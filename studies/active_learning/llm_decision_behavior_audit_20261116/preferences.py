#!/usr/bin/env python3
"""Fit descriptive choice models using frozen, label-free candidate features only."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/behavior_preferences_mpl')
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FEATURES = ['G_cw_distance', 'U_quantile_width', 'D_hamming', 'C_hamming_coverage']
INPUTS = {}


def read(path, expected=None):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    raw = path.read_bytes()
    assert raw == path.read_bytes(), ('unstable input', str(path))
    digest = hashlib.sha256(raw).hexdigest()
    if expected:
        assert digest == expected, ('hash mismatch', str(path))
    INPUTS[str(path.relative_to(ROOT))] = {'path': str(path.relative_to(ROOT)), 'sha256': digest, 'bytes': len(raw)}
    return raw


def load(path, expected=None):
    return json.loads(read(path, expected))


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def percentile(values):
    """Average ranks retain ties; a constant score has no within-state signal."""
    values = np.asarray(values, float)
    assert np.isfinite(values).all()
    return rankdata(values, method='average') / len(values)


def conditional_objective(beta, x, ridge=1.0, weights=None):
    """The chosen alternative is column zero; all alternatives enter the denominator."""
    x = np.asarray(x, float)
    utility = np.einsum('nkp,p->nk', x, beta)
    logden = logsumexp(utility, axis=1)
    prob = np.exp(utility - logden[:, None])
    weights = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    nll = np.dot(weights, logden - utility[:, 0]) + .5 * ridge * np.dot(beta, beta)
    grad = np.einsum('n,nk,nkp->p', weights, prob, x) - np.einsum('n,np->p', weights, x[:, 0]) + ridge * beta
    return float(nll), grad


def fit_logit(x, ridge=1.0, weights=None):
    assert x.ndim == 3 and np.isfinite(x).all()
    result = minimize(conditional_objective, np.zeros(x.shape[2]), args=(x, ridge, weights),
                      jac=True, method='L-BFGS-B', options={'ftol': 1e-11, 'gtol': 1e-7, 'maxiter': 300})
    if result.success or np.linalg.norm(result.jac) < 1e-4:
        return result.x, float(result.fun), True
    # Convex low-dimensional Newton fallback avoids L-BFGS line-search rounding failures.
    beta = result.x.copy()
    weight = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    for _ in range(50):
        objective, gradient = conditional_objective(beta, x, ridge, weight)
        if np.linalg.norm(gradient) < 1e-5:
            return beta, objective, True
        utility = np.einsum('nkp,p->nk', x, beta)
        prob = np.exp(utility - logsumexp(utility, axis=1)[:, None])
        mean = np.einsum('nk,nkp->np', prob, x)
        hessian = np.einsum('n,nk,nkp,nkq->pq', weight, prob, x, x) - np.einsum('n,np,nq->pq', weight, mean, mean)
        hessian += (ridge + 1e-8) * np.eye(x.shape[2])
        delta = np.linalg.solve(hessian, gradient)
        step = 1.0
        for _ in range(40):
            candidate = beta - step * delta
            new_objective, new_gradient = conditional_objective(candidate, x, ridge, weight)
            if new_objective <= objective:
                break
            step *= .5
        beta = candidate
        if abs(objective - new_objective) < 1e-9 and np.linalg.norm(new_gradient) < 1e-3:
            return beta, new_objective, True
    raise ArithmeticError('Conditional logit did not meet the gradient tolerance')


def canonical_factors(meta):
    c = meta['conditions']
    ratio = str(c['PE/EA']).strip()
    a, b = ratio.split('/')
    ratio = round(float(a) / float(b), 10)
    return (meta['smiles'], ratio, float(c['Density g/ml']), float(c['V/ul']),
            str(c['loading solvent']).strip(), float(c['Volume of loading solvent/ul']))


def fixed_reference(ids, seed, n=128):
    return sorted(ids, key=lambda value: hashlib.sha256(('preferences_static_hamming_v1_' + str(seed) + '_' + value).encode()).hexdigest())[:n]


def metadata_space(metadata):
    ids = sorted(metadata)
    factors = [canonical_factors(metadata[sid]) for sid in ids]
    code = np.zeros((len(ids), 6), np.int32)
    for col in range(6):
        labels = {v: i for i, v in enumerate(sorted({x[col] for x in factors}, key=str))}
        code[:, col] = [labels[x[col]] for x in factors]
    distance = np.zeros((len(ids), len(ids)), np.float32)
    for col in range(6):
        distance += code[:, col, None] != code[None, :, col]
    distance /= 6
    return ids, {sid: i for i, sid in enumerate(ids)}, distance


def frozen_catalog(state):
    source = Path(state['source'])
    directory = (ROOT / source).parent
    if not (directory / 'catalog.json').exists():
        return None, ''
    freeze = load(directory / 'batch_freeze.json', state.get('source_sha256'))
    hashes = freeze['artifact_hashes']
    contract = load(directory / 'contract.json', hashes.get('contract.json'))
    assert contract['checkpoint_sha256'] == state['checkpoint_sha256']
    assert set(contract['L_t_ids']) == set(state['labeled_ids'])
    assert set(contract['U_t_ids']) == set(state['unlabeled_ids'])
    catalog = load(directory / 'catalog.json', hashes.get('catalog.json', contract.get('catalog_sha256')))
    candidates = catalog['candidates'] + catalog['pending']
    assert {x['id'] for x in candidates} == set(state['unlabeled_ids'])
    assert len(candidates) == len(state['unlabeled_ids'])
    # The pending numerical half is part of the final accepted batch.
    assert set(freeze['batch_ids']) == set(state['selected_ids'])
    return {x['id']: x for x in candidates}, str((directory / 'catalog.json').relative_to(ROOT))


def cached_cw_scores(state, partitions, geometry):
    """Recompute only distances from a trusted existing gradient cache, never derivatives."""
    if state['seed'] not in (157, 6101) or state['L_after'] > 1005:
        return None, 'outside prespecified core cache audit'
    candidates = []
    source = ROOT / state['source']
    runtime = source.parent
    if '/selections/' in str(runtime):
        runtime = Path(str(runtime).replace('/selections/', '/runtime/'))
    if state['method'] == 'nocw':
        match = geometry[(geometry.method == 'nocw') & (geometry.seed == state['seed']) &
                         (geometry['round'] == state['round']) & (geometry.space == 'cw_gradient')]
        if len(match):
            row = match.iloc[0]
            candidates.append((ROOT / row.bank_path, row.bank_sha256, 'cw'))
        candidates.append((runtime / 'banks.npz', None, 'cw'))
    else:
        candidates.append((runtime / 'acquisition_artifacts/current_transformed_gradient_features.npz', None, 'features'))
        candidates.append((runtime / 'acquisition_artifacts/current_center_width_gradient_features.npz', None, 'features'))
        candidates.append((runtime / 'banks.npz', None, 'cw'))
    reverse = {int(v['canonical_index']): sid for sid, v in partitions[str(state['seed'])].items() if v['role'] in ('l0', 'u0')}
    reasons = []
    for path, expected, feature_key in candidates:
        if not path.exists():
            reasons.append('missing ' + str(path.relative_to(ROOT)))
            continue
        try:
            sidecar = path.with_suffix('.npz.contract.json') if feature_key == 'features' else path.parent / 'banks.json'
            md = load(sidecar)
            contract = md['contract']
            checkpoint = contract.get('checkpoint_sha256', contract.get('checkpoint'))
            assert checkpoint == state['checkpoint_sha256'], 'checkpoint mismatch'
            indices = contract.get('ordered_current_indices', contract.get('indices'))
            raw = read(path, expected or md['sha256'])
            with np.load(io.BytesIO(raw), allow_pickle=False) as bank:
                x = np.asarray(bank[feature_key], float)
                if 'canonical_indices' in bank:
                    assert np.array_equal(bank['canonical_indices'], indices)
            ordered_ids = [reverse[int(i)] for i in indices]
            assert len(set(ordered_ids)) == len(ordered_ids) == len(x)
            pos = {sid: i for i, sid in enumerate(ordered_ids)}
            assert set(ordered_ids) == set(state['labeled_ids']) | set(state['unlabeled_ids'])
            assert set(ordered_ids[:state['L_before']]) == set(state['labeled_ids'])
            u = x[[pos[sid] for sid in state['unlabeled_ids']]]
            h = x[[pos[sid] for sid in state['labeled_ids']]]
            # Squared Euclidean distance gives the same ranks as Euclidean.
            distances = cdist(u, h, metric='sqeuclidean').min(axis=1)
            return percentile(distances), str(path.relative_to(ROOT))
        except (AssertionError, KeyError, ValueError, OSError) as exc:
            reasons.append(str(exc))
    return None, '; '.join(reasons)


def choice_array(records, features, sample_seed=7001, nnegative=64):
    arrays, campaigns = [], []
    for state, values, mask in records:
        idx = [FEATURES.index(name) for name in features]
        if not np.isfinite(values[:, idx]).all():
            continue
        positive = np.flatnonzero(mask)
        negative = np.flatnonzero(~mask)
        assert len(positive) == 32 and len(negative) >= nnegative
        seed = int(stable([sample_seed, state['method'], state['seed'], state['round']])[:16], 16)
        rng = np.random.default_rng(seed)
        for picked in positive:
            alt = np.r_[picked, rng.choice(negative, nnegative, replace=False)]
            arrays.append(values[alt][:, idx])
            campaigns.append(int(state['seed']))
    return np.asarray(arrays), np.asarray(campaigns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--states', type=Path, default=HERE / 'states.json.xz')
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--negative-seeds', type=int, nargs='+', default=[7001, 7002, 7003])
    args = parser.parse_args()
    snapshot = json.loads(lzma.decompress(read(args.states)))
    states = snapshot['states']
    oldpath = ROOT / 'studies/active_learning/nocw_free_selection_diagnostics_20261101/snapshot_inputs.json.xz'
    old = json.loads(lzma.decompress(read(oldpath)))
    geometry = pd.read_csv(io.BytesIO(read('studies/active_learning/comprehensive_history_diagnostics_20261112/tables/behavior_geometry.csv')))
    ids, lookup, distance = metadata_space(snapshot['metadata'])
    records, summaries, coverage, correlations = [], [], [], []
    for count, state in enumerate(states):
        uid = state['unlabeled_ids']
        ui = [lookup[sid] for sid in uid]
        hi = [lookup[sid] for sid in state['labeled_ids']]
        mask = np.array([sid in set(state['selected_ids']) for sid in uid])
        assert mask.sum() == 32
        d = distance[np.ix_(ui, hi)].min(axis=1)
        ref = fixed_reference(snapshot['initial'][str(state['seed'])] + snapshot['pool'][str(state['seed'])], state['seed'])
        ri = [lookup[sid] for sid in ref]
        before = distance[np.ix_(ri, hi)].min(axis=1)
        gain = np.maximum(before[:, None] - distance[np.ix_(ri, ui)], 0).mean(axis=0)
        values = np.full((len(uid), 4), np.nan)
        values[:, 2] = percentile(d)
        values[:, 3] = percentile(gain)
        catalog, catalog_source = frozen_catalog(state)
        gsource = ''
        if catalog is not None:
            width = []
            for endpoint in ['V1', 'V2']:
                w = np.array([catalog[sid][f'pred_{endpoint}_q90_ml'] - catalog[sid][f'pred_{endpoint}_q10_ml'] for sid in uid])
                width.append(percentile(w))
            values[:, 1] = percentile(.5 * (width[0] + width[1]))
            if all('cw_distance_percentile' in catalog[sid] for sid in uid):
                values[:, 0] = percentile([catalog[sid]['cw_distance_percentile'] for sid in uid])
                gsource = catalog_source
        if not np.isfinite(values[:, 0]).all() and state['method'] in ('nocw', 'cw_gradient_lcmd32'):
            g, gsource = cached_cw_scores(state, old['partitions'], geometry)
            if g is not None:
                values[:, 0] = g
        cov = {k: state[k] for k in ['method', 'seed', 'round', 'L_before', 'L_after']}
        cov.update(candidate_n=len(uid), selected_n=int(mask.sum()), catalog_source=catalog_source,
                   G_source=gsource, information_time='frozen preselection state; static metadata only for D/C')
        for j, name in enumerate(FEATURES):
            cov[name + '_complete'] = bool(np.isfinite(values[:, j]).all())
            if cov[name + '_complete']:
                selected = values[mask, j]
                summaries.append({**{k: state[k] for k in ['method', 'seed', 'round', 'L_before', 'L_after']},
                                  'feature': name, 'candidate_n': len(uid), 'selected_n': 32,
                                  'selected_percentile_mean': selected.mean(),
                                  'pool_percentile_mean': values[:, j].mean(),
                                  'selected_minus_pool': selected.mean() - values[:, j].mean(),
                                  'top_decile_fraction': (selected > .9).mean()})
        coverage.append(cov)
        for j in range(4):
            for k in range(j + 1, 4):
                if np.isfinite(values[:, [j, k]]).all():
                    correlations.append({**{v: state[v] for v in ['method', 'seed', 'round', 'L_before']},
                                         'feature_a': FEATURES[j], 'feature_b': FEATURES[k],
                                         'pool_rank_correlation': float(np.corrcoef(values[:, j], values[:, k])[0, 1])})
        records.append((state, values, mask))
        if count % 100 == 0:
            print('candidate preference states', count, '/', len(states), flush=True)
    pd.DataFrame(coverage).to_csv(HERE / 'preference_state_coverage.csv', index=False)
    pd.DataFrame(summaries).to_csv(HERE / 'candidate_score_percentiles.csv', index=False)
    pd.DataFrame(correlations).to_csv(HERE / 'preference_collinearity.csv', index=False)
    # Equal actor, equal campaign, equal state reference; all scores are within-pool ranks.
    moments = []
    for state, values, _ in records:
        for j, name in enumerate(FEATURES):
            if np.isfinite(values[:, j]).all():
                moments.append({'method': state['method'], 'seed': state['seed'], 'feature': name,
                                'mean': values[:, j].mean(), 'second': np.mean(values[:, j] ** 2)})
    moments = pd.DataFrame(moments).groupby(['method', 'seed', 'feature'])[['mean', 'second']].mean().reset_index()
    moments = moments.groupby(['method', 'feature'])[['mean', 'second']].mean().reset_index()
    moments = moments.groupby('feature')[['mean', 'second']].mean()
    moments['sd'] = np.sqrt(moments.second - moments['mean'] ** 2)
    moments.to_csv(HERE / 'preference_standardization.csv')
    for _, values, _ in records:
        for j, name in enumerate(FEATURES):
            values[:, j] = (values[:, j] - moments.loc[name, 'mean']) / moments.loc[name, 'sd']
    models = [[name] for name in FEATURES] + [[FEATURES[0], FEATURES[1]], [FEATURES[0], FEATURES[2]], [FEATURES[1], FEATURES[2]],
                                            [FEATURES[2], FEATURES[3]], FEATURES]
    results, sensitivity, bootstrap_results = [], [], []
    scopes = {'common_1005': lambda s: s['seed'] in (157, 6101) and s['L_after'] <= 1005,
              'common_1453': lambda s: s['seed'] in (157, 6101) and s['L_after'] <= 1453,
              'all_available': lambda s: True}
    for scope, include in scopes.items():
        methods = sorted({s['method'] for s, _, _ in records})
        for method in methods:
            cohort = [r for r in records if r[0]['method'] == method and include(r[0])]
            for features in models:
                x, campaign = choice_array(cohort, features, args.negative_seeds[0])
                if len(x) < 64:
                    continue
                nstates = len(x) // 32
                available = len(cohort)
                # Avoid fitting a multivariate model to a small, selectively cached subset.
                complete = nstates == available
                if len(features) > 2 and not complete:
                    continue
                design = (x[:, 1:] - x[:, :1]).reshape(-1, len(features))
                condition = np.linalg.cond(design.T @ design) if len(features) > 1 else 1.0
                if len(features) > 2 and condition > 100:
                    continue
                beta, loss, success = fit_logit(x)
                model = '+'.join(features)
                base = {'scope': scope, 'method': method, 'model': model, 'n_states': nstates,
                        'eligible_states': available, 'state_coverage': nstates / available,
                        'n_choices': len(x), 'n_campaigns': len(np.unique(campaign)), 'ridge': 1,
                        'negative_n': 64, 'negative_seed': args.negative_seeds[0],
                        'design_condition_number': condition, 'optimizer_success': success,
                        'nll_penalized': loss, 'batch_selection_approximation': True}
                for feature, coefficient in zip(features, beta):
                    results.append({**base, 'aggregation': 'pooled', 'seed': 'ALL', 'feature': feature, 'coefficient': coefficient})
                for seed in np.unique(campaign):
                    b, l, ok = fit_logit(x[campaign == seed])
                    for feature, coefficient in zip(features, b):
                        results.append({**base, 'aggregation': 'campaign', 'seed': seed, 'feature': feature,
                                        'coefficient': coefficient, 'nll_penalized': l, 'optimizer_success': ok})
                if scope == 'common_1005':
                    for negseed in args.negative_seeds[1:]:
                        xs, _ = choice_array(cohort, features, negseed)
                        b, _, _ = fit_logit(xs)
                        for feature, coefficient in zip(features, b):
                            sensitivity.append({**base, 'negative_seed': negseed, 'feature': feature, 'coefficient': coefficient})
                # Campaign bootstrap is useful only for the larger five-seed descriptive cohort.
                if scope == 'all_available' and len(features) == 2 and len(np.unique(campaign)) >= 5 and complete:
                    rng = np.random.default_rng(60017)
                    seeds = np.unique(campaign)
                    boot = []
                    for _ in range(args.bootstrap):
                        chosen = rng.choice(seeds, len(seeds), replace=True)
                        weight = np.array([(chosen == s).sum() for s in campaign])
                        boot.append(fit_logit(x, weights=weight)[0])
                    boot = np.asarray(boot)
                    for j, feature in enumerate(features):
                        bootstrap_results.append({**base, 'feature': feature, 'coefficient': beta[j],
                                                  'cluster_bootstrap_n': args.bootstrap,
                                                  'ci025': np.quantile(boot[:, j], .025), 'ci975': np.quantile(boot[:, j], .975)})
            print('fit', scope, method, flush=True)
            pd.DataFrame(results).to_csv(HERE / 'behavior_preferences.csv', index=False)
            pd.DataFrame(sensitivity).to_csv(HERE / 'preference_negative_sensitivity.csv', index=False)
            pd.DataFrame(bootstrap_results).to_csv(HERE / 'preference_campaign_bootstrap.csv', index=False)
    pd.DataFrame(results).to_csv(HERE / 'behavior_preferences.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(HERE / 'preference_negative_sensitivity.csv', index=False)
    pd.DataFrame(bootstrap_results).to_csv(HERE / 'preference_campaign_bootstrap.csv', index=False)
    provenance = {'status': 'PASS', 'states': len(states), 'inputs': list(INPUTS.values()),
                  'new_training': 0, 'new_llm_calls': 0, 'unobserved_truth_reads': 0, 'test_truth_reads': 0,
                  'observed_labels_used_as_features': False,
                  'catalog_observed_section': 'Frozen JSON files contain legitimate past observations; that section is not accessed for scoring.',
                  'feature_definition': {'G': 'preselection CW-gradient nearest-history distance percentile, not gradient norm or value',
                                         'U': 'rank of average V1/V2 signed q90-q10 within-pool percentile; not calibrated epistemic uncertainty',
                                         'D': 'static six-factor minimum Hamming distance percentile',
                                         'C': 'singleton reduction in mean min-Hamming distance of 128 fixed references'},
                  'cache_distance_scope': 'seed157/6101 selected through L1005; catalog G reused at all available states',
                  'normalization': 'within-state pool average rank, then equal-actor/campaign/state shared z reference',
                  'negative_sampling': '64 distinct unselected feasible candidates per positive; entire final batch excluded',
                  'bootstrap': args.bootstrap, 'negative_seeds': args.negative_seeds,
                  'not_original_VU': True, 'retention_means_used_as_value': False}
    (HERE / 'preference_verification.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'states': len(states), 'coefficient_rows': len(results), 'percentile_rows': len(summaries)}))


if __name__ == '__main__':
    main()
