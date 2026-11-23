"""Strict single-factor analysis in physical units, using authorized label roles."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[2]
BASE = Path('experiments/e0_4g_baseline')
sys.modules.setdefault('run', sys.modules[__name__])
CAN = BASE / 'canonical_4g.csv'
QUAL = Path('studies/predictor/final_4g_qualification')
AL = Path('studies/active_learning')
HIST = AL / 'decision_mechanism_followup_20261117'
AXES = ('ea', 'V/ul', 'Volume of loading solvent/ul', 'Flow mL/min', 'loading solvent')
KEYS = ('canonical_smiles', 'CAS', 'column_specs', 'Purity', 'Density g/ml', *AXES)
TARGETS = ('V1', 'V2', 'C', 'W')
META = ('sample_id', *KEYS, 'PE/EA', 'quality_flags', 'source_row_1based')
SEEDS = (157, 887, 2357, 6101, 12203)
BUDGETS = (333, 525, 1005)
INPUTS = {}


def track(path):
    path = ROOT / path
    rel = str(path.relative_to(ROOT))
    if rel not in INPUTS:
        with path.open('rb') as stream:
            INPUTS[rel] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return path


def load(path):
    return json.loads(track(path).read_text())


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def table(name, data):
    pd.DataFrame(data).to_csv(OUT / 'tables' / f'{name}.csv', index=False, float_format='%.12g')


def fraction(value):
    numerator, denominator = map(float, value.split('/'))
    if min(numerator, denominator) < 0 or numerator + denominator <= 0:
        raise ValueError(value)
    return denominator / (numerator + denominator)


def physical(time, flow):
    return np.asarray(time, dtype=np.float64) * np.asarray(flow, dtype=np.float64) / 1200.0


def label_frame(path, allowed, loader=gzip.open):
    allowed = frozenset(allowed)
    with loader(track(path), 'rt', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['sample_id'] in allowed:
                yield row


def project_truth(allowed):
    frame = pd.DataFrame(label_frame(CAN, allowed, open))
    assert set(frame.sample_id) == set(allowed) and frame.sample_id.is_unique
    for target in (1, 2):
        computed = physical(frame[f't{target}'], frame['Flow mL/min'])
        np.testing.assert_allclose(computed, frame[f'V{target}_ml'].astype(float), atol=1e-12, rtol=0)
        frame[f'V{target}'] = computed
    return frame.set_index('sample_id')[['V1', 'V2']]


def verified_feedback(state, freeze, feedback, receipt):
    selected = state['selected_ids']
    digest = state['source_sha256']
    audit = receipt['audit']
    if not (len(selected) == len(set(selected)) == 32 and set(selected) <= set(state['unlabeled_ids'])
            and not set(selected) & set(state['labeled_ids'])
            and freeze['new_batch_labels_revealed'] is False and audit['roles'] == 'u0'
            and audit['acquisitions_frozen'] is True and audit['requested_rows'] == 32
            and audit['requested_ids_hash'] == stable(sorted(selected))
            and feedback['batch_freeze_sha256'] == receipt['batch_freeze_sha256'] == digest
            and stable(feedback) == receipt['feedback_sha256']):
        raise ValueError('invalid accepted-label receipt')
    predictions, records = freeze['premeasurement_predictions'], feedback['records']
    if any([row['candidate_id'] for row in source] != selected for source in (predictions, records)):
        raise ValueError('accepted IDs do not match')
    if any(pred[f'pred_{target}_ml'] != record[f'pred_{target}_ml']
           for pred, record in zip(predictions, records) for target in ('V1', 'V2')):
        raise ValueError('prediction provenance mismatch')
    return pd.DataFrame(records).set_index('candidate_id')


def enrich(frame):
    frame['C'] = (frame.V1 + frame.V2) / 2
    frame['W'] = frame.V2 - frame.V1
    return frame


def join_predictions(truth, pred):
    if not pred.index.is_unique or set(pred.index) != set(truth.index) or pred.checkpoint.nunique() != 1:
        raise ValueError('single-state prediction and explicit ID coverage required')
    assert pred[['V1', 'V2']].notna().all().all()
    return truth.join(pred.rename(columns={name: f'p_{name}' for name in TARGETS}))


def design(frame, scope, *, adjacent=True):
    from itertools import combinations

    pairs, levels, curvature = [], [], []
    values = list(TARGETS) + ([f'p_{target}' for target in TARGETS] if 'checkpoint' in frame else [])
    for axis in AXES:
        fixed = [key for key in KEYS if key != axis]
        for key, series in frame.groupby(fixed, sort=True, dropna=False):
            if series[axis].nunique() < 2:
                continue
            state = series.checkpoint.unique() if 'checkpoint' in series else ['']
            if len(state) != 1:
                raise ValueError('mixed checkpoints')
            info = {**scope, **dict(zip(fixed, key)), 'axis': axis, 'checkpoint': state[0],
                    'series': stable([axis, list(key)])[:20]}
            steps = []
            for level, group in series.groupby(axis, sort=True):
                ids = '|'.join(sorted(group.sample_id))
                steps.append({'level': level, 'ids': ids, 'n_rows': len(group),
                              **group[values].mean().to_dict()})
            levels.extend({**info, **step, 'n_levels': len(steps)} for step in steps)
            ordinal = axis != 'loading solvent'
            combos = zip(steps, steps[1:]) if ordinal and adjacent else combinations(steps, 2)
            for low, high in combos:
                dx = float(high['level'] - low['level']) if ordinal else np.nan
                step = {**info, 'low': low['level'], 'high': high['level'], 'dx': dx,
                        'low_ids': low['ids'], 'high_ids': high['ids'], 'n_levels': len(steps),
                        'delta_mass_mg': dx * info['Density g/ml'] if axis == 'V/ul' else np.nan}
                for name in values:
                    step[f'd_{name}'] = high[name] - low[name]
                    step[f's_{name}'] = step[f'd_{name}'] / dx
                pairs.append(step)
            if ordinal:
                for low, mid, high in zip(steps, steps[1:], steps[2:]):
                    left, right = mid['level'] - low['level'], high['level'] - mid['level']
                    item = {**info, **{f'{label}_ids': step['ids'] for label, step in zip(('low', 'mid', 'high'), (low, mid, high))},
                            'low': low['level'], 'mid': mid['level'], 'high': high['level']}
                    for name in values:
                        a, b, c = low[name], mid[name], high[name]
                        item[f'chord_{name}'] = b - (a + (c-a)*left/(left+right))
                        item[f'curv_{name}'] = -2 * item[f'chord_{name}'] / (left*right)
                        item[f'reversal_{name}'] = ((b-a)*(c-b) < 0)
                    curvature.append(item)
    return pairs, levels, curvature


def summarize(pairs, keys, thresholds=(0.1, 0.5, 1.0)):
    records = []
    rng = np.random.default_rng(20261123)
    for group, block in pairs.groupby(keys, sort=True):
        group = group if isinstance(group, tuple) else (group,)
        for target in TARGETS:
            actual = block[f'd_{target}'].to_numpy(float)
            predicted = block.get(f'd_p_{target}', pd.Series(np.nan, index=block.index)).to_numpy(float)
            base = pd.DataFrame({'mol': block.canonical_smiles.to_numpy(), 'actual': abs(actual), 'negative': actual < -1e-9,
                                 'positive': actual > 1e-9, 'pred': abs(predicted), 'error': abs(predicted-actual)})
            means = base.groupby('mol').mean()
            n_mol = len(means)
            boot = means.to_numpy()[rng.integers(n_mol, size=(1500, n_mol))].mean(axis=1)
            ratio = boot[:, 3] / np.where(boot[:, 0] > 0, boot[:, 0], np.nan)
            ci = lambda vector: np.quantile(vector, [.025, .975]) if n_mol >= 5 and np.isfinite(vector).all() else (np.nan, np.nan)
            actual_low, actual_high = ci(boot[:, 0]); error_low, error_high = ci(boot[:, 4]); ratio_low, ratio_high = ci(ratio)
            for threshold in thresholds:
                meaningful = abs(actual) > threshold
                score = base.loc[meaningful, ['mol']].copy()
                p, a = predicted[meaningful], actual[meaningful]
                score['flat'] = abs(p) < .1
                score['agree'] = (abs(p) >= .1) & (p*a > 0)
                score['opposite'] = (abs(p) >= .1) & (p*a < 0)
                score['ratio'] = abs(p)/abs(a)
                per_mol = score.groupby('mol').mean().mean() if np.isfinite(predicted).all() else pd.Series(dtype=float)
                records.append({**dict(zip(keys, group)), 'target': target, 'threshold': threshold,
                                'pairs': len(block), 'series': block.series.nunique(), 'molecules': n_mol,
                                'meaningful': int(meaningful.sum()), 'mean_abs_true': means.actual.mean(),
                                'mean_abs_pred': means.pred.mean(), 'difference_mae': means.error.mean(),
                                'actual_lo': actual_low, 'actual_hi': actual_high, 'mae_lo': error_low, 'mae_hi': error_high,
                                'ratio_lo': ratio_low, 'ratio_hi': ratio_high,
                                'amplitude_ratio': means.pred.mean()/means.actual.mean() if means.actual.mean() else np.nan,
                                'negative': means.negative.mean(), 'positive': means.positive.mean(),
                                'true_median_pair': np.median(actual), 'true_p10_pair': np.quantile(actual,.1),
                                'true_p90_pair': np.quantile(actual,.9),
                                'spearman_pair_descriptive': pd.Series(actual).corr(pd.Series(predicted), method='spearman')
                                if np.std(actual) > 1e-12 and np.std(predicted) > 1e-12 else np.nan, **per_mol.to_dict()})
    return pd.DataFrame(records)


def interpolation(levels):
    rows = []
    for _, block in levels.groupby('series'):
        if len(block) < 4 or block.axis.iloc[0] == 'loading solvent':
            continue
        block = block.sort_values('level')
        x = block.level.to_numpy(float)
        axis = block.axis.iloc[0]
        for target in TARGETS:
            y = block[target].to_numpy(float)
            full = np.polyval(np.polyfit(x, y, 1), x)
            for middle in range(1, len(x)-1):
                keep = np.arange(len(x)) != middle
                xfit, yfit = x[keep], y[keep]
                methods = {'linear': np.polyval(np.polyfit(xfit, yfit, 1), x[middle]),
                           'piecewise': np.interp(x[middle], xfit, yfit),
                           'monotonic': IsotonicRegression(increasing=bool(yfit[-1] > yfit[0])).fit(xfit, yfit).predict([x[middle]])[0],
                           'pchip': PchipInterpolator(xfit, yfit)(x[middle]), 'linear_resubstitution': full[middle]}
                rows.extend(dict(axis=axis, target=target, series=block.series.iloc[0], molecule=block.canonical_smiles.iloc[0],
                                 level=x[middle], method=method, abs_error=abs(value-y[middle])) for method, value in methods.items())
    return pd.DataFrame(rows)


def main():
    from al_models import al_models, historical
    from figures import figures
    meta = pd.read_csv(track(CAN), usecols=[name for name in META if name != 'ea'])
    meta['ea'] = meta['PE/EA'].map(fraction)
    assert meta[list(KEYS)].notna().all().all() and meta.sample_id.is_unique
    qualification = pd.read_csv(track(QUAL / 'splits/qualification_dataset_manifest.csv'), usecols=['sample_id','canonical_smiles','source_row'])
    identities = qualification.merge(meta, on=['sample_id','canonical_smiles'], validate='one_to_one')
    assert len(identities) == len(meta) and identities.source_row.eq(identities.source_row_1based).all()
    assert set(qualification.sample_id) == set(meta.sample_id)
    original = pd.read_csv(track('dataset/dataset_4g.csv'), usecols=lambda name: name not in ('t1','t2','verified t1','verified t2'))
    decisions = pd.read_csv(track(BASE / 'sample_decisions_4g.csv'), usecols=['decision','decision_reason','quality_flags'])
    split = pd.read_csv(track(QUAL / 'splits/row_seed_42.csv')).set_index('sample_id').split
    science_ids = split[split.isin(['train','validation'])].index
    physical_frame = meta.join(enrich(project_truth(science_ids)), on='sample_id', how='inner')
    coherent = physical_frame.quality_flags.isna() & physical_frame.V1.ge(0) & physical_frame.W.ge(0)
    series_results = [design(frame, {'scope': label}) for label, frame in
                      [('canonical_non_test', physical_frame), ('coherent', physical_frame[coherent])]]
    pairs, levels, curvature = (pd.DataFrame([row for result in series_results for row in result[index]]) for index in range(3))
    for label, output in [('physical_pairs',pairs),('physical_levels',levels),('physical_curvature',curvature)]:
        table(label, output)
    base_summary = summarize(pairs, ['scope','axis'], (.5,))
    table('physical_summary', base_summary)
    clean = pairs[pairs.scope.eq('coherent')].copy()
    clean['tpsa'] = clean.canonical_smiles.map(tpsa(clean.canonical_smiles.unique()))
    clean['tpsa_band'] = pd.cut(clean.tpsa, [-np.inf,20,40,np.inf], labels=['<=20','20–40','>40'])
    clean['loading_solvent_stratum'] = clean['loading solvent'].fillna('varies')
    table('physical_tpsa', summarize(clean, ['axis','tpsa_band'], (.5,)))
    table('physical_fixed_solvent', summarize(clean, ['axis','loading_solvent_stratum'], (.5,)))
    table('physical_by_molecule', summarize(clean, ['axis','canonical_smiles'], (.5,)))
    for axis, subset in clean[clean.axis.isin(['ea','V/ul'])].groupby('axis'):
        for target in ('V1','V2','W'):
            sign = 1 if axis == 'ea' or target == 'V1' else -1
            score = subset[f'd_{target}']*sign
            table(f'exceptions_{axis.replace("/","_")}_{target}', subset.loc[score[score > .5].nlargest(12).index])
    repeats = []
    for key, block in physical_frame.groupby(list(KEYS)):
        if len(block) >= 2:
            repeats.append({**dict(zip(KEYS,key)), 'n': len(block), 'ids': '|'.join(block.sample_id),
                            **{f'{t}_{name}': value for t in TARGETS for name, value in
                               [('sd', block[t].std()), ('range', block[t].max()-block[t].min())]}})
    table('repeats', repeats)
    checks = interpolation(levels[levels.scope.eq('coherent')]); table('interpolation', checks)
    if not checks.empty:
        table('interpolation_summary', checks.groupby(['axis','target','method','molecule']).abs_error.mean().groupby(level=[0,1,2]).agg(['size','mean','median']).reset_index())
    outputs = []
    global_rows = []
    for mode in ('row','compound'):
        for seed in (42,525,1101):
            path = QUAL / f'results/{mode}/seed_{seed}'
            run = load(path / 'run_summary.json')
            checkpoint = Path(run['checkpoint_path']); track(checkpoint)
            assert INPUTS[str(checkpoint)] == run['checkpoint_sha256']
            roles = pd.read_csv(track(QUAL / f'splits/{mode}_seed_{seed}.csv')).set_index('sample_id').split
            for role in ('train','validation'):
                ids = roles[roles.eq(role)].index
                cached = pd.DataFrame(label_frame(path / 'predictions.csv.gz', ids)).set_index('sample_id')
                assert set(cached.split) == {role}
                truth = enrich(project_truth(ids))
                np.testing.assert_allclose(truth.loc[cached.index,['V1','V2']], cached[['V1_true','V2_true']].astype(float), atol=1e-5, rtol=2e-7)
                pred = enrich(cached[['V1_q50','V2_q50']].astype(float).set_axis(['V1','V2'], axis=1))
                pred['checkpoint'] = run['checkpoint_sha256']
                frame = meta.join(join_predictions(truth,pred), on='sample_id', how='inner')
                if role == 'validation':
                    train_meta = meta[meta.sample_id.isin(roles[roles.eq('train')].index)]
                    train_keys = pd.MultiIndex.from_frame(train_meta[list(KEYS)])
                    evaluations = [(role, frame), ('validation_no_train_condition', frame[~pd.MultiIndex.from_frame(frame[list(KEYS)]).isin(train_keys)])]
                else:
                    evaluations = [(role, frame)]
                for role_name, rows in evaluations:
                    outputs.append(design(rows, dict(cohort=f'qualification_{mode}', seed=seed, budget=0, role=role_name)))
                global_rows.append(dict(cohort=f'qualification_{mode}', seed=seed,budget=0,role=role, **run['metrics'][role]))
    al_output, al_global = al_models(meta)
    outputs.extend(al_output)
    global_rows.extend(al_global)
    outputs.extend(historical(meta))
    mp, ml, mc = (pd.DataFrame([row for result in outputs for row in result[index]]) for index in range(3))
    for label, output in [('model_pairs',mp),('model_levels',ml),('model_curvature',mc)]:
        table(label, output)
    ms = summarize(mp, ['cohort','seed','budget','role','axis']); table('model_summary', ms)
    table('global_metrics',global_rows)
    audit = dict(raw_rows=len(original), raw_missing_metadata=original.isna().sum().to_dict(), canonical_rows=len(meta),
                 canonical_molecules=meta.canonical_smiles.nunique(), full_conditions=len(meta.drop_duplicates(list(KEYS))),
                 canonical_repeated_conditions=int((meta.groupby(list(KEYS)).size() > 1).sum()),
                 canonical_repeated_rows=int(meta.duplicated(list(KEYS),keep=False).sum()), n_science=len(physical_frame), n_coherent=int(coherent.sum()),
                 canonical_cas=meta.CAS.nunique(), multi_cas_for_smiles=int((meta.groupby('canonical_smiles').CAS.nunique()>1).sum()),
                 multi_smiles_for_cas=int((meta.groupby('CAS').canonical_smiles.nunique()>1).sum()),
                 original_decisions=Counter(decisions.decision), excluded_reasons=Counter(reason for cell in decisions[decisions.decision.ne('keep')].decision_reason.dropna() for reason in cell.split(';')),
                 ranges={key: dict(n=meta[key].nunique(),min=meta[key].min(),max=meta[key].max()) for key in KEYS[2:]},
                 qualification_split_sizes={file.stem:pd.read_csv(track(file)).split.value_counts().to_dict() for file in (ROOT/QUAL/'splits').glob('*seed*.csv')})
    (OUT/'tables/audit.json').write_text(json.dumps(audit, indent=2, default=lambda value:value.item(), ensure_ascii=False)+'\n')
    figures(pairs, levels, mp, ms, pd.DataFrame(global_rows))
    tracked_outputs = sorted(file for directory in ('tables','figures','scripts') for file in (OUT/directory).glob('*') if file.is_file())
    (OUT/'manifest.json').write_text(json.dumps(dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        inputs_sha256=INPUTS, outputs_sha256={str(file.relative_to(OUT)):hashlib.sha256(file.read_bytes()).hexdigest() for file in tracked_outputs},
        thresholds_ml=[.1,.5,1.], flat_ml=.1, bootstrap=dict(unit='canonical_smiles', draw=1500,seed=20261123),
        al_budgets=BUDGETS,al_seeds=SEEDS, new_fits=0,new_rollouts=0,new_label_reveals=0,test_truth_projected=0),indent=2)+'\n')


def tpsa(smiles):
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcTPSA
    return {mol: CalcTPSA(Chem.MolFromSmiles(mol)) for mol in smiles}


if __name__ == '__main__':
    main()
