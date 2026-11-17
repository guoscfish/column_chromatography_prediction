#!/usr/bin/env python3
"""Audit accepted hybrid halves and visible rationales without labels or model fits."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import lzma
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import rankdata
from scipy.stats import spearmanr
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
OLD = ROOT / 'studies/active_learning/nocw_free_selection_diagnostics_20261101'
FREV = ROOT / 'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions'
OWNERS = ((21, 'sol_xhigh_hybrid_free_to1005_20261013'),
          (26, 'sol_xhigh_all_strategies_to1165_20261014'),
          (30, 'sol_xhigh_all_strategies_to1293_20261015'),
          (34, 'sol_xhigh_all_strategies_to1421_20261017'),
          (36, 'sol_xhigh_all_strategies_to1549_20261018'))
sys.path.insert(0, str(OLD))
from condition_metrics import _canonical_conditions, compute_condition_metrics
from geometry_metrics import fixed_reference_ids

FILES = {}


def read(path):
    path = Path(path)
    value = path.read_bytes()
    assert value == path.read_bytes(), ('unstable source', str(path))
    FILES[str(path.relative_to(ROOT))] = {
        'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(value).hexdigest(),
        'bytes': len(value), 'role': 'hybrid/rationale metadata; responses never used'}
    return value


def js(path):
    return json.loads(read(path))


def accepted_selection(path):
    result = js(path)
    freeze = js(Path(path).with_name('batch_freeze.json'))
    assert FILES[str(Path(path).relative_to(ROOT))]['sha256'] == freeze['artifact_hashes']['selection.json']
    return result


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def split_halves(freeze, selection):
    """Use explicit pending IDs, never assume positional ownership."""
    all_ids = freeze['batch_ids']
    numerical = freeze['pending_ids']
    llm = [choice['id'] for choice in selection['choices']]
    assert len(numerical) == len(llm) == 16
    assert len(set(numerical)) == len(set(llm)) == 16
    assert not set(numerical) & set(llm)
    assert len(all_ids) == len(set(all_ids)) == 32
    assert set(all_ids) == set(numerical) | set(llm)
    return llm, numerical


def extract():
    snapshot = json.loads(lzma.decompress(read(OLD / 'snapshot_inputs.json.xz')))
    batches = []
    proposals = []
    for seed, n_rounds in [(157, 36), (6101, 35)]:
        previous = list(snapshot['initial'][str(seed)])
        for rnd in range(n_rounds):
            owner = next(name for end, name in OWNERS if rnd + 1 <= end)
            directory = FREV / owner / f'selections/seed_{seed}/cw16_llm16_scientist_v2/round_{rnd:02d}'
            freeze = js(directory / 'batch_freeze.json')
            contract = js(directory / 'contract.json')
            selection = js(directory / 'selection.json')
            for name in ['contract.json', 'selection.json']:
                assert FILES[str((directory / name).relative_to(ROOT))]['sha256'] == freeze['artifact_hashes'][name]
            llm, numerical = split_halves(freeze, selection)
            proposal_n = 0
            mismatch_n = 0
            for path in sorted(directory.glob('turn_*.json')):
                if path.name.endswith('.started.json'):
                    continue
                turn = js(path)
                if path.name in freeze['artifact_hashes']:
                    assert FILES[str(path.relative_to(ROOT))]['sha256'] == freeze['artifact_hashes'][path.name]
                answer = turn.get('answer', {})
                if isinstance(answer, str):
                    try:
                        answer = json.loads(answer)
                    except json.JSONDecodeError:
                        continue
                if not isinstance(answer, dict) or answer.get('type') != 'selection':
                    continue
                proposal_n += 1
                proposed = [v.get('id') for v in answer.get('choices', []) if isinstance(v, dict)]
                match = proposed == llm
                mismatch_n += not match
                proposals.append({'method': 'free16_lcmd16', 'seed': seed, 'round': rnd,
                                  'turn_source': str(path.relative_to(ROOT)),
                                  'proposal_n_choices': len(proposed), 'duplicate_ids': len(proposed) - len(set(proposed)),
                                  'outside_nonpending_U': len(set(proposed) - (set(contract['U_t_ids']) - set(numerical))),
                                  'matches_final_llm_order': match,
                                  'observable': 'structured answer only; rejection reason not inferred'})
            assert set(contract['L_t_ids']) == set(previous)
            assert len(previous) == 333 + 32 * rnd
            assert not set(freeze['batch_ids']) & set(previous)
            assert set(freeze['batch_ids']) <= set(contract['U_t_ids'])
            row = {'method': 'free16_lcmd16', 'seed': seed, 'round': rnd,
                   'L_before': 333 + 32 * rnd, 'L_after': 365 + 32 * rnd,
                   'selected_ids': freeze['batch_ids'], 'labeled_ids': contract['L_t_ids'],
                   'unlabeled_ids': contract['U_t_ids'], 'llm_ids': llm, 'numerical_ids': numerical,
                   'source': str((directory / 'batch_freeze.json').relative_to(ROOT)),
                   'source_sha256': FILES[str((directory / 'batch_freeze.json').relative_to(ROOT))]['sha256'],
                   'contract_source': str((directory / 'contract.json').relative_to(ROOT)),
                   'selection_source': str((directory / 'selection.json').relative_to(ROOT)),
                   'checkpoint_sha256': contract['checkpoint_sha256'],
                   'completed_fit_validation': True,
                   'pending_then_llm_order': freeze['batch_ids'] == numerical + llm,
                   'observable_selection_proposals': proposal_n,
                   'observable_nonfinal_proposals': mismatch_n,
                   'final_source_partition_verified': True}
            batches.append(row)
            previous += freeze['batch_ids']
    assert len(batches) == 71
    dump('hybrid_batches.json', batches)
    pd.DataFrame(proposals).to_csv(OUT / 'hybrid_proposal_audit.csv', index=False)
    return snapshot, batches


def marginal_coverage(nearest_h, dist_l, dist_n):
    """Distance decrease is a monotone, submodular coverage proxy."""
    h = np.asarray(nearest_h)
    l = np.minimum(h, np.asarray(dist_l))
    n = np.minimum(h, np.asarray(dist_n))
    union = np.minimum(l, n)
    result = {'llm_standalone': float(np.mean(h - l)),
              'numerical_standalone': float(np.mean(h - n)),
              'llm_after_numerical': float(np.mean(n - union)),
              'numerical_after_llm': float(np.mean(l - union)),
              'union_gain': float(np.mean(h - union))}
    result['llm_order_average'] = .5 * (result['llm_standalone'] + result['llm_after_numerical'])
    result['numerical_order_average'] = .5 * (result['numerical_standalone'] + result['numerical_after_llm'])
    result['shared_coverage_gain'] = result['llm_standalone'] + result['numerical_standalone'] - result['union_gain']
    assert min(result.values()) >= -1e-12
    assert abs(result['llm_order_average'] + result['numerical_order_average'] - result['union_gain']) < 1e-12
    return result


def fixed_geometry(meta):
    ids = sorted(meta)
    index = {sid: k for k, sid in enumerate(ids)}
    smiles = sorted({meta[sid]['smiles'] for sid in ids})
    sm_index = {s: k for k, s in enumerate(smiles)}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    molecular = 1 - np.asarray([DataStructs.BulkTanimotoSimilarity(f, fps) for f in fps])
    mol_index = np.asarray([sm_index[meta[sid]['smiles']] for sid in ids])
    canonical = {sid: _canonical_conditions(meta[sid]) for sid in ids}
    assert all(x is not None for x in canonical.values())
    values = np.asarray([[1 / (1 + float(canonical[sid][0])), float(canonical[sid][1]),
                         float(canonical[sid][2]), float(canonical[sid][4])] for sid in ids])
    spans = np.ptp(values, axis=0)
    spans[spans == 0] = 1
    values = (values - np.min(values, axis=0)) / spans
    solvents = np.asarray([canonical[sid][3] for sid in ids])

    def distances(a, b, space):
        ai = np.asarray([index[sid] for sid in a])
        bi = np.asarray([index[sid] for sid in b])
        if space == 'molecular':
            return molecular[np.ix_(mol_index[ai], mol_index[bi])]
        condition = (cdist(values[ai], values[bi], metric='cityblock') +
                     (solvents[ai, None] != solvents[bi][None, :])) / 5
        if space == 'condition':
            return condition
        return .5 * (condition + molecular[np.ix_(mol_index[ai], mol_index[bi])])

    return canonical, distances


def attribution(snapshot, batches):
    meta = snapshot['metadata']
    canonical, distance = fixed_geometry(meta)
    rows = []
    selected_rows = []
    for batch in batches:
        seed, rnd = batch['seed'], batch['round']
        h, l, n = batch['labeled_ids'], batch['llm_ids'], batch['numerical_ids']
        reference = fixed_reference_ids(snapshot['pool'][str(seed)], 128,
                                        'nocw_free_selection_diagnostics_v1_seed_' + str(seed))
        row = {k: batch[k] for k in ['method', 'seed', 'round', 'L_before', 'L_after', 'source',
                                     'selection_source', 'pending_then_llm_order', 'final_source_partition_verified']}
        h_sm = {meta[sid]['smiles'] for sid in h}
        l_sm, n_sm = ({meta[sid]['smiles'] for sid in subset} for subset in [l, n])
        h_cond = {canonical[sid] for sid in h}
        l_cond, n_cond = ({canonical[sid] for sid in subset} for subset in [l, n])
        h_state = {(meta[sid]['smiles'], canonical[sid]) for sid in h}
        for label, subset, other in [('llm', l, n), ('numerical', n, l)]:
            sm = {meta[sid]['smiles'] for sid in subset}
            states = {(meta[sid]['smiles'], canonical[sid]) for sid in subset}
            own, _ = compute_condition_metrics(subset, h, meta, return_examples=0)
            row.update({f'{label}_unique_molecules': len(sm),
                        f'{label}_new_molecules': len(sm - h_sm),
                        f'{label}_new_molecules_exclusive': len(sm - h_sm - {meta[sid]['smiles'] for sid in other}),
                        f'{label}_new_molecule_slots': sum(meta[sid]['smiles'] not in h_sm for sid in subset),
                        f'{label}_new_states': len(states - h_state),
                        f'{label}_same_molecule_single_factor_history_fraction': own['cross_any_single_selected_slots_fraction'],
                        f'{label}_same_molecule_single_factor_within_fraction': own['within_any_single_selected_slots_fraction'],
                        f'{label}_structure_nearest_history_mean': float(distance(subset, h, 'molecular').min(axis=1).mean()),
                        f'{label}_condition_nearest_history_mean': float(distance(subset, h, 'condition').min(axis=1).mean())})
            d = distance(subset, subset, 'joint')
            np.fill_diagonal(d, np.inf)
            row[f'{label}_within_nearest_joint_mean'] = float(d.min(axis=1).mean())
        row['molecule_overlap_halves'] = len(l_sm & n_sm)
        row['new_molecule_union'] = len((l_sm | n_sm) - h_sm)
        row['condition_overlap_halves'] = len(l_cond & n_cond)
        row['candidate_overlap_halves'] = len(set(l) & set(n))
        row['cross_half_nearest_joint_mean'] = float(distance(l, n, 'joint').min(axis=1).mean())
        for space in ['molecular', 'condition', 'joint']:
            gain = marginal_coverage(distance(reference, h, space).min(axis=1),
                                     distance(reference, l, space).min(axis=1),
                                     distance(reference, n, space).min(axis=1))
            row.update({f'{space}_{name}': value for name, value in gain.items()})
        directory = ROOT / batch['selection_source']
        catalog = js(directory.with_name('catalog.json'))
        assert FILES[str(directory.with_name('catalog.json').relative_to(ROOT))]['sha256'] == js(directory.with_name('batch_freeze.json'))['artifact_hashes']['catalog.json']
        candidates = {v['id']: v for v in catalog['candidates'] + catalog['pending']}
        assert set(candidates) == set(batch['unlabeled_ids'])
        for label, subset in [('llm', l), ('numerical', n)]:
            row[f'{label}_cw_distance_percentile_mean'] = np.mean([candidates[sid]['cw_distance_percentile'] for sid in subset])
            width = []
            for sid in subset:
                candidate = candidates[sid]
                u = .5 * sum(candidate[f'pred_{endpoint}_q90_ml'] - candidate[f'pred_{endpoint}_q10_ml'] for endpoint in ['V1', 'V2'])
                width.append(u)
                selected_rows.append({'method': 'free16_lcmd16', 'seed': seed, 'round': rnd,
                                      'sample_id': sid, 'source_half': label,
                                      'cw_distance_percentile': candidate['cw_distance_percentile'],
                                      'q90_q10_endpoint_average_ml': u,
                                      'information_time': 'historical same-checkpoint preselection catalog'})
            row[f'{label}_q90_q10_endpoint_average_ml'] = float(np.mean(width))
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / 'hybrid_attribution.csv', index=False)
    pd.DataFrame(selected_rows).to_csv(OUT / 'hybrid_selected_features.csv', index=False)
    return pd.DataFrame(rows)


def gradient_attribution(snapshot, batches):
    """Reuse each exact historical bank; compare normalized within-state gains."""
    rows = []
    missing = []
    for batch in batches:
        directory = Path(str(ROOT / batch['selection_source']).replace('/selections/', '/runtime/')).parent
        try:
            if not (directory / 'banks.json').exists():
                # Extension boundaries keep the already-fitted state in its prior owner.
                matches = [FREV / owner / f"runtime/seed_{batch['seed']}/cw16_llm16_scientist_v2/round_{batch['round']:02d}"
                           for _, owner in OWNERS]
                matches = [p for p in matches if (p / 'banks.json').exists()]
                matches = [p for p in matches if js(p / 'banks.json')['contract']['checkpoint'] == batch['checkpoint_sha256']]
                assert len(matches) == 1, ('ambiguous/missing historical bank', batch['seed'], batch['round'])
                directory = matches[0]
            manifest = js(directory / 'banks.json')
            assert manifest['contract']['checkpoint'] == batch['checkpoint_sha256']
            reverse = {int(v['canonical_index']): sid for sid, v in snapshot['partitions'][str(batch['seed'])].items()
                       if v['role'] in ['l0', 'u0']}
            ids = [reverse[int(i)] for i in manifest['contract']['indices']]
            assert len(ids) == len(set(ids)) == 3330
            n_h = len(batch['labeled_ids'])
            assert set(ids[:n_h]) == set(batch['labeled_ids'])
            assert set(ids[n_h:]) == set(batch['unlabeled_ids'])
            raw = read(directory / 'banks.npz')
            assert hashlib.sha256(raw).hexdigest() == manifest['sha256']
            positions = {sid: k for k, sid in enumerate(ids)}
            refs = fixed_reference_ids(snapshot['pool'][str(batch['seed'])], 128,
                                       'nocw_free_selection_diagnostics_v1_seed_' + str(batch['seed']))
            refs_i = [positions[sid] for sid in refs]
            l_i = [positions[sid] for sid in batch['llm_ids']]
            n_i = [positions[sid] for sid in batch['numerical_ids']]
            for space in ['cw', 'raw']:
                with np.load(io.BytesIO(raw), allow_pickle=False) as bank:
                    x = np.asarray(bank[space], dtype=np.float64)
                assert x.shape == (3330, 512) and np.isfinite(x).all()
                ref = x[refs_i]
                before = cdist(ref, x[:n_h]).min(axis=1)
                for k, sid in enumerate(refs):
                    if sid in set(batch['labeled_ids']):
                        before[k] = 0
                dl = cdist(ref, x[l_i]).min(axis=1)
                dn = cdist(ref, x[n_i]).min(axis=1)
                base = before.mean()
                gains = marginal_coverage(before, dl, dn)
                row = {k: batch[k] for k in ['method', 'seed', 'round', 'L_before', 'L_after', 'checkpoint_sha256']}
                row.update({'space': space + '_gradient', 'reference_n': len(refs),
                            'reference_ids_sha256': sha_json(refs),
                            'bank_path': str((directory / 'banks.npz').relative_to(ROOT)),
                            'bank_sha256': manifest['sha256'], 'within_state_reference_before_mean': base})
                row.update({key + '_relative': value / base if base > 0 else 0 for key, value in gains.items()})
                pool_norm = np.linalg.norm(x[n_h:], axis=1)
                ranks = (rankdata(pool_norm, method='average') - .5) / len(pool_norm)
                for label, indices in [('llm', l_i), ('numerical', n_i)]:
                    row[label + '_gradient_norm_pool_percentile'] = float(np.mean(ranks[np.asarray(indices) - n_h]))
                    row[label + '_nearest_h_raw_scale'] = float(cdist(x[indices], x[:n_h]).min(axis=1).mean())
                    d = cdist(x[indices], x[indices])
                    np.fill_diagonal(d, np.inf)
                    row[label + '_within_nn_raw_scale'] = float(d.min(axis=1).mean())
                rows.append(row)
        except (FileNotFoundError, KeyError) as exc:
            missing.append({'seed': batch['seed'], 'round': batch['round'], 'reason': str(exc)})
    pd.DataFrame(rows).to_csv(OUT / 'hybrid_gradient_attribution.csv', index=False)
    dump('hybrid_gradient_missing.json', missing)


def conditional_random(snapshot, batches, random_n=500):
    """Keep real CW16 fixed, randomize only the additional legal LLM16 slots."""
    meta = snapshot['metadata']
    _, distance = fixed_geometry(meta)
    rows = []
    for batch in batches:
        h, l, n = batch['labeled_ids'], batch['llm_ids'], batch['numerical_ids']
        eligible = [sid for sid in batch['unlabeled_ids'] if sid not in set(n)]
        assert len(eligible) == len(batch['unlabeled_ids']) - 16 and set(l) <= set(eligible)
        reference = fixed_reference_ids(snapshot['pool'][str(batch['seed'])], 128,
                                        'nocw_free_selection_diagnostics_v1_seed_' + str(batch['seed']))
        rng = np.random.default_rng(20261116 + batch['seed'] * 100 + batch['round'])
        samples = np.asarray([rng.choice(len(eligible), 16, replace=False) for _ in range(random_n)])
        positions = {sid: i for i, sid in enumerate(eligible)}
        actual_i = [positions[sid] for sid in l]
        record = {k: batch[k] for k in ['method', 'seed', 'round', 'L_before', 'L_after']}
        record.update({'random_batches': random_n, 'fixed_numeric_count': 16,
                       'eligible_llm_count': len(eligible), 'randomized_count': 16,
                       'reference_n': len(reference), 'reference_ids_sha256': sha_json(reference),
                       'baseline': 'uniform16 from U_t excluding actual fixed numerical16; no label access',
                       'hard_constraints': 'distinct unseen nonpending IDs; prompt has no molecule/control quota',
                       'viewed_ids_constraint': 'search-tool visibility is endogenous; candidate universe before queries'})
        for space in ['molecular', 'condition', 'joint']:
            nearest = distance(eligible, h, space).min(axis=1)
            candidate_to_ref = distance(reference, eligible, space)
            before = np.minimum(distance(reference, h, space).min(axis=1),
                                distance(reference, n, space).min(axis=1))
            random_nearest = nearest[samples].mean(axis=1)
            random_gain = (before[None, :] - np.minimum(before[None, :], candidate_to_ref[:, samples].min(axis=2).T)).mean(axis=1)
            actual_nearest = nearest[actual_i].mean()
            actual_gain = (before - np.minimum(before, candidate_to_ref[:, actual_i].min(axis=1))).mean()
            for name, actual, values in [('nearest_history', actual_nearest, random_nearest),
                                          ('coverage_after_numeric', actual_gain, random_gain)]:
                prefix = space + '_' + name
                record[prefix] = float(actual)
                record[prefix + '_random_mean'] = float(values.mean())
                record[prefix + '_random_delta'] = float(actual - values.mean())
                ties = np.isclose(values, actual, rtol=1e-10, atol=1e-12)
                record[prefix + '_random_percentile'] = float((np.sum((values < actual) & ~ties) + .5 * np.sum(ties)) / random_n)
                record[prefix + '_random_mcse'] = float(values.std(ddof=1) / np.sqrt(random_n))
                record[prefix + '_random_delta_first100'] = float(actual - values[:100].mean())
        rows.append(record)
    pd.DataFrame(rows).to_csv(OUT / 'hybrid_conditional_random.csv', index=False)


PATTERNS = {
    'exploration': r'\b(explor\w*|novel\w*|unobserved|unmeasured|sparse\w*|underrepresent\w*|coverage|divers\w*|new molecule\w*|new scaffold\w*)\b',
    'local_control': r'\b(local\w*|matched|control\w*|discriminat\w*|confirm\w*|single.factor)\b',
    'avoid_redundancy': r'\b(avoid\w*|redundan\w*|duplicat\w*)\b',
}


def text_flags(text):
    return {key + '_terms': len(re.findall(pattern, text.lower())) for key, pattern in PATTERNS.items()}


def visible_text(selection):
    parts = [selection.get('batch_strategy', ''), selection.get('batch_rationale', '')]
    for hyp in selection.get('hypotheses', []):
        if isinstance(hyp, dict):
            parts += [str(hyp.get(key, '')) for key in ['hypothesis', 'content', 'question', 'next_test', 'discriminating_question', 'next_discriminating_question']]
        else:
            parts.append(str(hyp))
    parts += [str(v.get('reason', '')) for v in selection.get('choices', [])]
    return '\n'.join(x for x in parts if x)


def rationale(snapshot, hybrids):
    records = []
    name_map = {'free': 'historical_free_mixed_model', 'nocw': 'nocw'}
    for batch in snapshot['batches']:
        if batch['method'] not in name_map or not batch['completed_fit_validation']:
            continue
        directory = ROOT / batch['source']
        selection_path = directory.with_name('selection.json')
        selection = accepted_selection(selection_path) if selection_path.exists() else batch['notes']
        records.append({'method': name_map[batch['method']], 'seed': batch['seed'], 'round': batch['round'],
                        'L_before': batch['L_before'], 'L_after': batch['L_after'],
                        'selected_ids': batch['selected_ids'], 'selection': selection,
                        'selection_source': str(selection_path.relative_to(ROOT)),
                        'text_extent': 'final accepted structured rationale; not internal reasoning'})
    for batch in hybrids:
        records.append({**{k: batch[k] for k in ['method', 'seed', 'round', 'L_before', 'L_after', 'selected_ids', 'selection_source']},
                        'selection': accepted_selection(ROOT / batch['selection_source']),
                        'text_extent': 'LLM16 final structured rationale; numerical16 selected before LLM'})
    state_path = OUT / 'states.json.xz'
    if state_path.exists():
        states = json.loads(lzma.decompress(read(state_path)))['states']
        existing = {(b['method'], b['seed'], b['round']) for b in records}
        for batch in states:
            key = batch['method'], batch['seed'], batch['round']
            if batch['method'] not in ['historical_free_mixed_model', 'nocw'] or key in existing:
                continue
            selection_path = (ROOT / batch['source']).with_name('selection.json')
            if not selection_path.exists():
                continue
            selection = accepted_selection(selection_path)
            assert {v['id'] for v in selection['choices']} == set(batch['selected_ids'])
            records.append({**{k: batch[k] for k in ['method', 'seed', 'round', 'L_before', 'L_after', 'selected_ids']},
                            'selection': selection, 'selection_source': str(selection_path.relative_to(ROOT)),
                            'text_extent': 'final accepted structured rationale; not internal reasoning'})
    archive_root = ROOT / 'reports/free_cw_final_20261112/scientific_artifacts'
    import gzip
    for seed in [157, 887, 2357, 6101, 12203]:
        for line in gzip.decompress(read(archive_root / f'seed_{seed}_results.jsonl.gz')).decode().splitlines():
            entry = json.loads(line)
            if Path(entry['path']).name != 'selection.json':
                continue
            assert hashlib.sha256(entry['raw_utf8'].encode()).hexdigest() == entry['sha256']
            selection = json.loads(entry['raw_utf8'])
            rnd = int(Path(entry['path']).parent.name.split('_')[-1])
            records.append({'method': 'new_free_cw_sol61_xhigh', 'seed': seed, 'round': rnd,
                            'L_before': 333 + 32 * rnd, 'L_after': 365 + 32 * rnd,
                            'selected_ids': [v['id'] for v in selection['choices']],
                            'selection': selection, 'selection_source': entry['path'],
                            'text_extent': 'final accepted structured rationale; not internal reasoning'})
    out = []
    for record in records:
        selection = record['selection']
        text = visible_text(selection)
        r = {k: v for k, v in record.items() if k not in ['selection', 'selected_ids']}
        r.update(text_flags(text))
        reasons = selection.get('choices', [])
        choice_ids = {v['id'] for v in reasons}
        r.update({'visible_characters': len(text), 'visible_choice_reasons': len(reasons),
                  'choice_ids_match_final': choice_ids <= set(record['selected_ids']),
                  'selected_ids_excerpt': '|'.join(record['selected_ids'][:4]),
                  'rationale_excerpt': (selection.get('batch_strategy', '') or selection.get('batch_rationale', '') or text)[:1200],
                  'candidate_reason_excerpt': ' | '.join(str(v.get('id', '')) + ': ' + str(v.get('reason', '')) for v in reasons[:2])[:1400]})
        out.append(r)
    frame = pd.DataFrame(out)
    assert not frame.duplicated(['method', 'seed', 'round']).any()
    frame.to_csv(OUT / 'rationale_metrics.csv', index=False)
    return frame


def preference_dynamics(snapshot, hybrids):
    meta = snapshot['metadata']
    records = []
    for batch in snapshot['batches']:
        if batch['method'] in ['free', 'nocw'] and batch['completed_fit_validation']:
            records.append({'method': {'free': 'historical_free_mixed_model', 'nocw': 'nocw'}[batch['method']],
                            'seed': batch['seed'], 'round': batch['round'], 'selected_ids': batch['selected_ids']})
    records += hybrids
    state_path = OUT / 'states.json.xz'
    if state_path.exists():
        records = json.loads(lzma.decompress(read(state_path)))['states']
    rows = []
    for method, seed in sorted({(b['method'], b['seed']) for b in records}):
        series = sorted([b for b in records if b['method'] == method and b['seed'] == seed], key=lambda x: x['round'])
        for before, after in zip(series[:-1], series[1:]):
            assert after['round'] == before['round'] + 1
            b = Counter(meta[sid]['smiles'] for sid in before['selected_ids'])
            a = Counter(meta[sid]['smiles'] for sid in after['selected_ids'])
            support = sorted(set(b) | set(a))
            # Raw action distributions have a stable definition; they are not beliefs.
            jsd = float(jensenshannon([b.get(x, 0) for x in support], [a.get(x, 0) for x in support], base=2) ** 2)
            rows.append({'method': method, 'seed': seed, 'round': after['round'],
                         'L_before': 333 + 32 * after['round'], 'molecule_action_js_divergence': jsd,
                         'previous_batch_molecule_slot_fraction': sum(a[s] for s in set(a) & set(b)) / 32,
                         'interpretation': 'Action Preference Dynamics; not calibrated belief movement'})
    pd.DataFrame(rows).to_csv(OUT / 'action_preference_dynamics.csv', index=False)


def attach_cases(frame):
    path = OUT / 'batch_metrics.csv'
    if not path.exists():
        return
    metrics = pd.read_csv(path)
    keys = ['method', 'seed', 'round']
    common = [name for name in metrics.columns if any(x in name.lower() for x in ['novel', 'rarity', 'coverage', 'random', 'percentile', 'single_factor', 'new_molecule'])]
    joined = frame.merge(metrics[keys + common], on=keys, how='left', validate='one_to_one')
    associations = []
    for scope, scoped in [('all_available', joined), ('common_1005', joined[joined.L_after <= 1005])]:
        for (method, seed), group in scoped.groupby(['method', 'seed']):
            rate = 1000 * group.exploration_terms / group.visible_characters
            for metric in ['hamming_novelty_random_delta', 'rarity_random_delta', 'joint_coverage_gain_random_delta']:
                if metric not in group:
                    continue
                valid = rate.notna() & group[metric].notna()
                coefficient = spearmanr(rate[valid], group.loc[valid, metric]).statistic if rate[valid].nunique() > 1 else np.nan
                associations.append({'scope': scope, 'method': method, 'seed': seed, 'batches': int(valid.sum()),
                                     'text_proxy': 'exploration keyword matches per 1000 visible characters',
                                     'action_metric': metric, 'spearman': coefficient,
                                     'interpretation': 'campaign-level descriptive association; not calibrated exploration intent'})
    pd.DataFrame(associations).to_csv(OUT / 'rationale_behavior_associations.csv', index=False)
    joined['stage'] = pd.cut(joined.L_before, [332, 525, 813, 1005, 1600], labels=['early', 'middle', 'late_common', 'extension'], right=False)
    metric = next((name for name in common if 'novel' in name.lower() and ('delta' in name.lower() or 'correct' in name.lower())), None)
    metric = metric or next((name for name in common if 'novel' in name.lower()), None)
    rows = []
    for _, group in joined.groupby(['method', 'stage'], observed=True):
        if metric is not None and group[metric].notna().any():
            valid = group[group[metric].notna()].sort_values([metric, 'seed', 'round'])
            chosen = [valid.iloc[0], valid.iloc[-1]]
        else:
            chosen = [group.sort_values(['seed', 'round']).iloc[0]]
        for index, record in enumerate(chosen):
            row = record.to_dict()
            row['selection_rule'] = 'within method/stage minimum' if index == 0 else 'within method/stage maximum'
            row['selection_metric'] = metric or 'earliest available metadata'
            row['judge_status'] = 'transparent keyword screening; human-reviewable final rationale only'
            rows.append(row)
    pd.DataFrame(rows).drop_duplicates(keys).to_csv(OUT / 'decision_cases.csv', index=False)


def write_results():
    hybrid = pd.read_csv(OUT / 'hybrid_attribution.csv')
    gradient = pd.read_csv(OUT / 'hybrid_gradient_attribution.csv')
    conditional = pd.read_csv(OUT / 'hybrid_conditional_random.csv')
    text = pd.read_csv(OUT / 'rationale_metrics.csv')
    proposals = pd.read_csv(OUT / 'hybrid_proposal_audit.csv')
    dynamics = pd.read_csv(OUT / 'action_preference_dynamics.csv')
    association = pd.read_csv(OUT / 'rationale_behavior_associations.csv')

    def table(frame):
        frame = frame.copy()
        for name in frame:
            if pd.api.types.is_numeric_dtype(frame[name]):
                frame[name] = frame[name].map(lambda value: f'{value:.5g}')
        return '| ' + ' | '.join(frame.columns) + ' |\n|' + '|'.join(['---'] * len(frame.columns)) + '|\n' + '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in frame.to_numpy())

    cols = ['llm_new_molecules', 'numerical_new_molecules',
            'llm_same_molecule_single_factor_history_fraction', 'numerical_same_molecule_single_factor_history_fraction',
            'llm_cw_distance_percentile_mean', 'numerical_cw_distance_percentile_mean',
            'llm_q90_q10_endpoint_average_ml', 'numerical_q90_q10_endpoint_average_ml']
    summary = hybrid.groupby('seed')[cols].mean().reset_index()
    gs = gradient[gradient.space == 'cw_gradient'].groupby('seed')[['llm_order_average_relative', 'numerical_order_average_relative', 'union_gain_relative', 'shared_coverage_gain_relative', 'llm_gradient_norm_pool_percentile', 'numerical_gradient_norm_pool_percentile']].mean().reset_index()
    rs = conditional.groupby('seed')[['joint_nearest_history_random_delta', 'joint_coverage_after_numeric_random_delta', 'joint_coverage_after_numeric_random_percentile', 'molecular_coverage_after_numeric_random_delta', 'condition_coverage_after_numeric_random_delta']].mean().reset_index()
    ts = text.groupby('method').agg(batches=('round', 'size'), exploration_keyword_fraction=('exploration_terms', lambda x: (x > 0).mean()), local_keyword_fraction=('local_control_terms', lambda x: (x > 0).mean())).reset_index()
    ds = dynamics.groupby('method').agg(transitions=('round', 'size'), molecule_action_js=('molecule_action_js_divergence', 'mean'), previous_batch_molecule_slots=('previous_batch_molecule_slot_fraction', 'mean')).reset_index()
    asso = association[association.scope == 'common_1005'].groupby(['method', 'action_metric']).spearman.agg(['mean', 'min', 'max']).reset_index()
    report = f'''# Hybrid 互补性、可见理由与行动偏好动态

## 证据范围与来源核对

本模块分析 seed 157 的36批（L333→1485）和6101的35批（L333→1453），合计71批、2272个接受名额。`contract.json`提供每轮历史与未标注候选ID；`batch_freeze.json.pending_ids`是数值半批，最终`selection.json.choices`是LLM半批。两者各16、无交集且并集等于32个最终ID，71批均为数值先冻结、LLM后补充。来源划分来自显式字段，不是依据位置猜测。最终结果未见自动补位或去重导致的来源混合。

`hybrid_proposal_audit.csv`保存{len(proposals)}个可见结构化selection答复，较71批多{len(proposals)-71}个；其中{int((~proposals.matches_final_llm_order).sum())}个答复的choice顺序列表不同于最终接受列表，累计{int(proposals.outside_nonpending_U.sum())}个ID提议事件不在对应非pending U内、{int(proposals.duplicate_ids.sum())}个批内重复ID。列表相同也不保证其科学证据字段合法，因此这些数字不能逐一等同于特定拒绝原因。可见最终理由中有ID拼写修订；未把它解释成静默自动修复。原始推理未恢复。

## 同状态半批行为

下表是逐批均值再按seed汇总，单位：分子数/批、比例、池内CW最近历史距离百分位、两个端点q90−q10宽度的平均mL。

{table(summary)}

LLM半批比数值半批选择更多首次出现的分子，也有更高的同分子单因素历史对照比例。这两件事可以同时成立：16个名额既可包含少量新分子，也可对历史已知分子进行局部条件实验。CW距离百分位说明数值半批更远离当前梯度历史；宽度不同只涉及单模型预测区间，不是经校准的认知不确定性。

## 梯度几何与覆盖顺序归因

每轮复用已有raw/CW CountSketch缓存，核对checkpoint、canonical index映射及L/U集合。扩展目录边界只在同checkpoint、同L/U及同轮索引全部匹配后引用原owner保存的bank；共{len(gradient)}个空间状态，缺失见`hybrid_gradient_missing.json`。不把跨轮原始梯度距离当作同一个固定表示空间。

固定参考集为每seed的U0中按旧脚本SHA规则确定的128个ID。定义覆盖为参考候选到历史集合的平均最近距离的负值，增量是距离下降；分别计算L单独、N单独、L给定N、N给定L。顺序平均贡献为两个进入顺序的平均；两份贡献相加等于联合增量。这是几何归因，绝非学习性能贡献。下表将每轮距离下降除以该轮参考集起始平均距离，再按seed汇总。

{table(gs)}

在CW梯度空间，数值半批的顺序平均相对覆盖贡献约为LLM半批的4–5倍；但是LLM梯度范数池内百分位反而更高。梯度范数大与集合几何覆盖高是不同性质。固定Morgan分子空间与固定条件空间的顺序平均贡献可查`hybrid_attribution.csv`：例如联合结构—条件空间中seed157数值稍大，6101则LLM稍大，不能将梯度结果泛化为所有空间。

静态条件距离使用EA体积分数、密度、上样体积、加载体积的固定候选元数据范围归一化，再与加载溶剂不等指示按五因素平均；分子距离是Morgan radius2/2048 Tanimoto距离；联合距离各占一半。范围来自无标签元数据并固定，不因训练更新变化。密度是记录因素且可能与分子绑定，单因素对照只对现有五个记录字段负责。几何共同参考覆盖的重叠不等于实验无效冗余。

## 保留真实数值16的条件随机基线

每个真实历史状态固定实际CW16，在当轮未标注池排除这16个pending后，不放回随机16个，重复500次，固定种子`20261116 + seed*100 + round`。同样排除已测量ID且保持预算；原Prompt没有分子配额、对照配额或固定探索比例。检索工具的已查看集合是策略行动的结果，基线以查询前完整候选宇宙作为可行域，未声称重现同一查询过程。

{table(rs)}

LLM的联合邻历史距离相对条件随机均值在两seed都偏低；但LLM给定CW的联合覆盖增量相对随机差值跨seed异号。LLM补充分子覆盖的平均差值为正，条件覆盖差值为负。这支持空间分工的描述，无法支持一致的几何协同，更无法支持Row Performance协同。

## 可见理由与实际行动

仅使用最终接受的`batch_strategy`、`batch_rationale`、假设文本和候选reason；关键词规则在脚本`PATTERNS`中公开。它检测探索相关措辞，不是原论文Judge估计的[0,1]探索意图。没有外部Judge、没有新LLM调用，不能恢复内部思维。最终选择文件与freeze哈希绑定。

{table(ts)}

按候选历史状态随机校正指标计算每个campaign内的Spearman关联；文本变量为每1000可见字符的探索关键词数。下表限L333–1005，每seed先单独计算，再列seed均值和范围。没有把候选行作为独立重复，没有强显著性检验，也没有声称严格复现论文的cycle correction。

{table(asso)}

New CW的关联方向随seed及探索代理变化；Historical和NoCW有部分正向关联。结果反对“探索措辞与行动绝对无关”的预设，也不能将其读作校准意图的测量。候选reason本身可能在描述已经选择的点，词频关联还存在文本长度、批次内容和后验叙述混杂。

`decision_cases.csv`按方法×阶段选取随机校正Hamming Novelty最小与最大批，合计32例，保存原可见英文摘录、候选ID、具体理由和全套数值。它明确展示反例而非挑选单向故事：

| 方法/seed/round | 可见设计 | Hamming随机百分位 | 联合Coverage随机百分位 | 含义 |
|---|---|---:|---:|---|
| Historical /6101/0 | condition-response curves、failure repair及novel-region exploration | 0 | 0.022 | 有探索措辞而选点接近历史，符合局部实验设计 |
| New CW /157/4 | 阈值条件扫点、matched loading、late-tail transfer anchors | 0.001 | 0.002 | 局部对照为明确设计，不能据此判断低效 |
| New CW /12203/3 | 未覆盖功能团转移并配对照 | 1 | 0.998 | 探索措辞与高探索代理一致的反例 |
| NoCW /2357/15 | tail thresholds、loading interactions与central controls | 0.001 | 0.982 | Hamming局部程度与覆盖贡献不等价 |

## Action Preference Dynamics

定义每批分子名额的经验分布，比较相邻批的Jensen–Shannon散度（log2，范围0–1）及与上一批分子重叠的名额比例。它们是行动分布的变化，不能当作信念校准或反馈因果效应。

{table(ds)}

LLM轨迹比CW-LCMD保留更多上一批分子名额，但不代表过度反应；科学对照、工具可见性和当轮训练状态均能解释这种持久性。当前可见假设不是统一定义、跨轮可比的概率信念，无法评估测量噪声，故不计算Martingale Score或Excess Belief Movement。

## 边界

两seedHybrid只支持描述性统计；与独立纯策略逐步分叉的历史不是同一冻结状态下的反事实。真实Row曲线必须由主分析审计，以上覆盖贡献均不作为性能贡献。分子新增数量随历史增长自然下降，原始距离也会随模型特征变换；因此所有强机制判断应限于同轮来源对照。零新训练、零新付费LLM、零未来标签查询。
'''
    (OUT / 'HYBRID_TEXT_RESULTS.md').write_text(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract-only', action='store_true')
    parser.add_argument('--attach-cases-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.attach_cases_only:
        attach_cases(pd.read_csv(OUT / 'rationale_metrics.csv'))
        write_results()
        return
    snapshot, batches = extract()
    if not args.extract_only:
        attribution(snapshot, batches)
        gradient_attribution(snapshot, batches)
        conditional_random(snapshot, batches)
        frame = rationale(snapshot, batches)
        preference_dynamics(snapshot, batches)
        attach_cases(frame)
        write_results()
    dump('hybrid_text_input_manifest.json', list(FILES.values()))
    print(json.dumps({'hybrid_batches': len(batches), 'input_files': len(FILES), 'new_training': 0, 'new_llm_calls': 0}))


if __name__ == '__main__':
    main()
