#!/usr/bin/env python3
"""Reconstruct exact frozen-state CW-LCMD32 prefixes; never load candidate truth."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
import lzma
import os
from pathlib import Path
import re
import subprocess
import sys

for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '1'
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PRIOR = ROOT / 'studies/active_learning/llm_decision_behavior_audit_20261116'
LEGACY = ROOT / 'studies/active_learning/nocw_free_selection_diagnostics_20261101'
LCMD_PATH = ROOT / 'src/qgeognn_al/active_learning_v2/lcmd.py'
INPUTS = {}


def read(path, expected=None):
    path = Path(path)
    raw = path.read_bytes()
    assert raw == path.read_bytes(), ('unstable source', str(path))
    digest = hashlib.sha256(raw).hexdigest()
    if expected:
        assert digest == expected, ('input hash mismatch', str(path))
    INPUTS[str(path.relative_to(ROOT))] = {'path': str(path.relative_to(ROOT)),
                                         'sha256': digest, 'bytes': len(raw)}
    return raw


def js(path, expected=None):
    return json.loads(read(path, expected))


def load_module(name, path):
    read(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def phase(budget):
    for low, high in [(333, 525), (525, 813), (813, 1005), (1005, 1453), (1453, 1485)]:
        if low <= budget < high:
            return f'{low}-{high}'
    raise ValueError('Unexpected budget')


def gain(before, distances):
    return float(np.mean(before - np.minimum(before, distances.min(axis=1))))


def pair_fraction(values):
    values = list(values)
    return float(np.mean([values[i] == values[j] for i in range(len(values)) for j in range(i + 1, len(values))]))


FACTOR_RULES = {
    'pe': r'\b(?:PE/EA|eluent|mobile.phase|solvent.response|solvent.strength|solvent ratio)\b',
    'sample_volume': r'\b(?:sample.volume|sample.amount|sample.load|loading.amount|\d+\s*(?:uL|μL|mg))\b',
    'loading_volume': r'\b(?:loading.solvent.volume|loading.volume|load.solvent.volume)\b',
    'loading_solvent': r'\b(?:loading.solvent|load.solvent|DCM|dichloromethane)\b',
    'structure_transfer': r'\b(?:transfer|scaffold|functional.group|unobserved|unmeasured|new.molecule|analogu?e)\w*\b',
}


def reason_flags(text):
    return {key: bool(re.search(pattern, text, re.I)) for key, pattern in FACTOR_RULES.items()}


def audit_prefix(selected, pending):
    same = selected[:len(pending)] == list(pending)
    mismatch = next((i for i, (a, b) in enumerate(zip(selected, pending)) if a != b), None)
    return same, mismatch


def group_stats(ids, h, n, meta, canonical, distance, x, position, refs,
                cw_before_h, cw_before_n, catalog, nearest_h, g_percentile,
                condition_metrics):
    indices = [position[sid] for sid in ids]
    hi, ni = [position[sid] for sid in h], [position[sid] for sid in n]
    ri = [position[sid] for sid in refs]
    widths = np.asarray([[catalog[sid][f'pred_{endpoint}_q90_ml'] - catalog[sid][f'pred_{endpoint}_q10_ml']
                          for endpoint in ['V1', 'V2']] for sid in ids])
    q50 = np.asarray([[catalog[sid][f'pred_{endpoint}_q50_ml'] for endpoint in ['V1', 'V2']] for sid in ids])
    actual, _ = condition_metrics(ids, h, meta, return_examples=0)
    after_n, _ = condition_metrics(ids, h + n, meta, return_examples=0)
    combined, _ = condition_metrics(n + ids, h, meta, return_examples=0)
    molecule = [meta[sid]['smiles'] for sid in ids]
    history_molecule = {meta[sid]['smiles'] for sid in h}
    hn_molecule = history_molecule | {meta[sid]['smiles'] for sid in n}
    stats = {'unique_molecules': len(set(molecule)),
             'new_molecules_h': len(set(molecule) - history_molecule),
             'new_molecules_hn': len(set(molecule) - hn_molecule),
             'new_molecule_slots_hn': sum(s not in hn_molecule for s in molecule),
             'single_factor_history_fraction': actual['cross_any_single_selected_slots_fraction'],
             'single_factor_hn_fraction': after_n['cross_any_single_selected_slots_fraction'],
             'single_factor_within_fraction': actual['within_any_single_selected_slots_fraction'],
             'full32_single_factor_within_fraction': combined['within_any_single_selected_slots_fraction'],
             'same_molecule_pairs_fraction': pair_fraction(molecule),
             'exact_condition_pairs_fraction': pair_fraction([canonical[sid] for sid in ids]),
             'cw_nearest_h_mean': float(nearest_h[np.asarray(indices) - len(h)].mean()),
             'cw_nearest_hn_mean': float(np.minimum(nearest_h[np.asarray(indices) - len(h)], cdist(x[indices], x[ni]).min(axis=1)).mean()),
             'cw_distance_pool_percentile': float(np.mean([catalog[sid]['cw_distance_percentile'] for sid in ids])),
             'gradient_norm_pool_percentile': float(g_percentile[np.asarray(indices) - len(h)].mean()),
             'cw_gain_after_n': gain(cw_before_n, cdist(x[ri], x[indices])),
             'pred_V1_q50_mean': float(q50[:, 0].mean()), 'pred_V2_q50_mean': float(q50[:, 1].mean()),
             'V1_width_mean': float(widths[:, 0].mean()), 'V2_width_mean': float(widths[:, 1].mean()),
             'width_mean': float(widths.mean()),
             'ea_fraction_mean': float(np.mean([1 / (1 + float(canonical[sid][0])) for sid in ids])),
             'sample_volume_mean': float(np.mean([float(canonical[sid][2]) for sid in ids])),
             'loading_volume_mean': float(np.mean([float(canonical[sid][4]) for sid in ids]))}
    stats['cw_gain_after_n_relative_h'] = stats['cw_gain_after_n'] / cw_before_h.mean() if cw_before_h.mean() > 0 else 0
    stats['cw_gain_after_n_relative_hn'] = stats['cw_gain_after_n'] / cw_before_n.mean() if cw_before_n.mean() > 0 else 0
    d = cdist(x[indices], x[indices])
    np.fill_diagonal(d, np.inf)
    stats['cw_within_nn_mean'] = float(d.min(axis=1).mean())
    for factor in ['pe', 'density', 'sample_volume', 'loading_solvent', 'loading_volume']:
        stats['single_' + factor + '_history_fraction'] = actual['cross_only_' + factor + '_selected_slots_fraction']
        stats['single_' + factor + '_hn_fraction'] = after_n['cross_only_' + factor + '_selected_slots_fraction']
        stats['full32_single_' + factor + '_within_fraction'] = combined['within_only_' + factor + '_selected_slots_fraction']
        stats['full32_single_' + factor + '_pairs'] = combined['within_only_' + factor + '_pairs']
    for space in ['molecular', 'condition', 'joint']:
        before = distance(refs, h + n, space).min(axis=1)
        stats[space + '_gain_after_n'] = gain(before, distance(refs, ids, space))
        stats[space + '_nearest_h_mean'] = float(distance(ids, h, space).min(axis=1).mean())
        d = distance(ids, ids, space)
        np.fill_diagonal(d, np.inf)
        stats[space + '_within_nn_mean'] = float(d.min(axis=1).mean())
    return stats


def run(limit=None, check_prior=True):
    if check_prior:
        result = subprocess.run([sys.executable, '-B', str(PRIOR / 'run.py'), '--check'], cwd=ROOT,
                                check=True, capture_output=True, text=True)
        old_check = result.stdout.strip()
    else:
        old_check = 'skipped by explicit command option; not a final sealed run'
    lcmd = load_module('followup_exact_lcmd', LCMD_PATH)
    sys.path.insert(0, str(LEGACY))
    reuse = load_module('followup_static_geometry', PRIOR / 'hybrid_text.py')
    for name in ['condition_metrics.py', 'geometry_metrics.py']:
        read(LEGACY / name)
    for name in ['scientist_study.py', 'scientist_lifecycle.py']:
        read(ROOT / 'src/qgeognn_al/active_learning_v2' / name)
    snapshot = json.loads(lzma.decompress(read(LEGACY / 'snapshot_inputs.json.xz')))
    batches = js(PRIOR / 'hybrid_batches.json')
    banks = pd.read_csv(io.BytesIO(read(PRIOR / 'hybrid_gradient_attribution.csv')))
    bank_rows = {(int(v.seed), int(v.round)): v for v in banks[banks.space == 'cw_gradient'].itertuples()}
    meta = snapshot['metadata']
    canonical, distance = reuse.fixed_geometry(meta)
    if limit is not None:
        batches = batches[:limit]
    state_rows, metric_rows, candidate_rows, traces, response_pairs = [], [], [], [], []
    for b in batches:
        seed, rnd = b['seed'], b['round']
        state = dict(seed=seed, round=rnd, L_before=b['L_before'], L_after=b['L_after'], phase=phase(b['L_before']))
        state_row = {**state, 'included': False, 'reason': ''}
        try:
            bank_row = bank_rows[(seed, rnd)]
            path = ROOT / bank_row.bank_path
            bank_meta = js(path.with_suffix('.json'))
            assert bank_meta['sha256'] == bank_row.bank_sha256
            assert bank_meta['contract']['checkpoint'] == b['checkpoint_sha256']
            contract = js(ROOT / b['contract_source'])
            freeze = js(ROOT / b['source'], b['source_sha256'])
            selection_path = ROOT / b['selection_source']
            selection = js(selection_path, freeze['artifact_hashes']['selection.json'])
            assert INPUTS[b['contract_source']]['sha256'] == freeze['artifact_hashes']['contract.json']
            assert b['numerical_ids'] == freeze['pending_ids']
            assert b['llm_ids'] == [v['id'] for v in selection['choices']]
            reverse = {int(v['canonical_index']): sid for sid, v in snapshot['partitions'][str(seed)].items() if v['role'] in ['l0', 'u0']}
            ids = [reverse[int(k)] for k in bank_meta['contract']['indices']]
            h = contract['L_t_ids']
            u = contract['U_t_ids']
            n_h = len(h)
            assert ids[:n_h] == h and ids[n_h:] == u
            assert len(ids) == len(set(ids)) == 3330
            position = {sid: k for k, sid in enumerate(ids)}
            with np.load(io.BytesIO(read(path, bank_meta['sha256'])), allow_pickle=False) as bank:
                original = bank['cw']
            assert original.shape == (3330, 512) and np.isfinite(original).all()
            result = lcmd.lcmd_tp_select(original[n_h:], original[:n_h], 32)
            selected = [u[int(k)] for k in result.selected_pool_positions]
            matched, mismatch = audit_prefix(selected, b['numerical_ids'])
            state_row.update(first16_ordered_equal=matched, first_mismatch_order=mismatch,
                             bank_path=bank_row.bank_path, bank_sha256=bank_row.bank_sha256,
                             checkpoint_sha256=b['checkpoint_sha256'], stored_dtype=str(original.dtype),
                             calculation_dtype='float64', ordered_history_and_pool_verified=True)
            if not matched:
                state_row['reason'] = 'ordered historical pending16 mismatch; excluded from counterfactual claims'
                state_rows.append(state_row)
                continue
            numerical = selected[:16]
            alternative = selected[16:]
            llm = b['llm_ids']
            assert not set(alternative) & set(h + numerical)
            catalog_source = selection_path.with_name('catalog.json')
            raw_catalog = js(catalog_source, freeze['artifact_hashes']['catalog.json'])
            catalog = {v['id']: v for v in raw_catalog['candidates'] + raw_catalog['pending']}
            assert set(catalog) == set(u)
            reasons = {v['id']: v['reason'] for v in selection['choices']}
            x = np.asarray(original, dtype=np.float64)
            refs = reuse.fixed_reference_ids(snapshot['pool'][str(seed)], 128, 'nocw_free_selection_diagnostics_v1_seed_' + str(seed))
            ri, ni = [position[sid] for sid in refs], [position[sid] for sid in numerical]
            before_h = cdist(x[ri], x[:n_h]).min(axis=1)
            before_n = np.minimum(before_h, cdist(x[ri], x[ni]).min(axis=1))
            norm_pct = rankdata(np.linalg.norm(x[n_h:], axis=1), method='average') / len(u)
            nearest_h = np.sqrt(result.initial_nearest_sq_distance)
            row = {**state, 'llm_cf_candidate_overlap': len(set(llm) & set(alternative)),
                   'llm_cf_molecule_overlap': len({meta[sid]['smiles'] for sid in llm} & {meta[sid]['smiles'] for sid in alternative}),
                   'llm_cf_condition_overlap': len({canonical[sid] for sid in llm} & {canonical[sid] for sid in alternative}),
                   'cw_first16_gain_relative_h': float(np.mean(before_h - before_n) / before_h.mean()),
                   'same_state_reference_before_h': float(before_h.mean()),
                   'same_state_reference_before_hn': float(before_n.mean()),
                   'remaining_unseen_molecules_hn': len({meta[sid]['smiles'] for sid in u} - {meta[sid]['smiles'] for sid in h + numerical}),
                   'reference_n': len(refs), 'counterfactual_kind': 'candidate set only; no training or truth outcome'}
            for group, chosen in [('llm16', llm), ('lcmd_back16', alternative)]:
                values = group_stats(chosen, h, numerical, meta, canonical, distance, x, position, refs,
                                     before_h, before_n, catalog, nearest_h, norm_pct, reuse.compute_condition_metrics)
                row.update({group + '_' + key: value for key, value in values.items()})
                for order, sid in enumerate(chosen):
                    c = canonical[sid]
                    reason = reasons.get(sid, '') if group == 'llm16' else ''
                    record = {**state, 'group': group, 'comparison_order': order, 'sample_id': sid,
                              'canonical_smiles': meta[sid]['smiles'], 'in_both_alternatives': sid in set(llm) & set(alternative),
                              'PE_EA': str(c[0]), 'density': float(c[1]), 'sample_volume_ul': float(c[2]),
                              'loading_solvent': c[3], 'loading_volume_ul': float(c[4]),
                              'known_molecule_h': meta[sid]['smiles'] in {meta[v]['smiles'] for v in h},
                              'known_molecule_hn': meta[sid]['smiles'] in {meta[v]['smiles'] for v in h + numerical},
                              'cw_distance_pool_percentile': catalog[sid]['cw_distance_percentile'],
                              'gradient_norm_pool_percentile': norm_pct[position[sid] - n_h],
                              'cw_nearest_h': nearest_h[position[sid] - n_h],
                              'cw_nearest_hn': float(min(nearest_h[position[sid] - n_h], cdist(x[[position[sid]]], x[ni]).min())),
                              'visible_reason': reason,
                              'reason_status': 'final visible LLM rationale' if group == 'llm16' else 'no LLM rationale exists for numerical counterfactual',
                              'true_labels_read': False}
                    for endpoint in ['V1', 'V2']:
                        record[endpoint + '_q50'] = catalog[sid][f'pred_{endpoint}_q50_ml']
                        record[endpoint + '_q90_q10'] = catalog[sid][f'pred_{endpoint}_q90_ml'] - catalog[sid][f'pred_{endpoint}_q10_ml']
                    flags = reason_flags(reason)
                    record.update({'reason_' + key: value if group == 'llm16' else None for key, value in flags.items()})
                    for factor_index, factor in enumerate(['pe', 'density', 'sample_volume', 'loading_solvent', 'loading_volume']):
                        anchors = [aid for aid in h + numerical if meta[aid]['smiles'] == meta[sid]['smiles']
                                   and [j for j in range(5) if canonical[aid][j] != c[j]] == [factor_index]]
                        record['single_' + factor + '_hn'] = bool(anchors)
                        record['single_' + factor + '_hn_anchors'] = len(anchors)
                        record['cw_distance_to_' + factor + '_hn_anchor'] = float(cdist(x[[position[sid]]], x[[position[aid] for aid in anchors]]).min()) if anchors else None
                        record['cw_distance_to_' + factor + '_hn_anchor_relative_reference_h'] = record['cw_distance_to_' + factor + '_hn_anchor'] / before_h.mean() if anchors and before_h.mean() > 0 else None
                    candidate_rows.append(record)
                for a_index, a in enumerate(chosen):
                    for z in chosen[a_index + 1:]:
                        if meta[a]['smiles'] != meta[z]['smiles']:
                            continue
                        changed = [j for j in range(5) if canonical[a][j] != canonical[z][j]]
                        if len(changed) != 1:
                            continue
                        changed_factor = ['pe', 'density', 'sample_volume', 'loading_solvent', 'loading_volume'][changed[0]]
                        response_pairs.append({**state, 'group': group, 'id_a': a, 'id_b': z, 'factor': changed_factor,
                                               'pred_V1_q50_abs_delta': abs(catalog[a]['pred_V1_q50_ml'] - catalog[z]['pred_V1_q50_ml']),
                                               'pred_V2_q50_abs_delta': abs(catalog[a]['pred_V2_q50_ml'] - catalog[z]['pred_V2_q50_ml']),
                                               'cw_distance': float(np.linalg.norm(x[position[a]] - x[position[z]])),
                                               'cw_distance_relative_reference_h': float(np.linalg.norm(x[position[a]] - x[position[z]]) / before_h.mean()),
                                               'actual_response': 'not accessed; prediction-only pair'})
            for key in values:
                row['delta_llm_minus_lcmd_' + key] = row['llm16_' + key] - row['lcmd_back16_' + key]
            for trace, sid in zip(result.trace, selected):
                traces.append({**state, 'sample_id': sid, 'half': 'first16_verified_history' if trace['selection_order'] < 16 else 'counterfactual_back16', **trace})
            state_row.update(included=True, pending16_count=16, reconstructed_count=32,
                             llm_count=16, counterfactual_back_count=16,
                             actual_llm_ids=llm, reconstructed_first16=numerical, reconstructed_back16=alternative)
            metric_rows.append(row)
        except (KeyError, OSError, ValueError, AssertionError) as exc:
            state_row['reason'] = str(exc)
        state_rows.append(state_row)
        print(json.dumps({'seed': seed, 'round': rnd, 'included': state_row['included'], 'first16_equal': state_row.get('first16_ordered_equal'), 'reason': state_row['reason']}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(state_rows).to_csv(OUT / 'counterfactual_state_coverage.csv', index=False)
    (OUT / 'counterfactual_selected_ids.json').write_text(json.dumps(state_rows, ensure_ascii=False, indent=2) + '\n')
    frame = pd.DataFrame(metric_rows)
    frame.to_csv(OUT / 'counterfactual_batches.csv', index=False)
    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(OUT / 'counterfactual_candidates.csv', index=False)
    pd.DataFrame(traces).to_csv(OUT / 'counterfactual_lcmd_trace.csv', index=False)
    pd.DataFrame(response_pairs).to_csv(OUT / 'counterfactual_predicted_response_pairs.csv', index=False)
    if not frame.empty:
        summaries(frame, candidates)
        write_report(frame, candidates, state_rows)
    verification = {'historical_seal_check': old_check, 'attempted_states': len(batches),
                    'exact_prefix_verified': int(sum(x['included'] for x in state_rows)),
                    'excluded_states': [x for x in state_rows if not x['included']],
                    'candidate_comparison_rows': len(candidates), 'lcmd_trace_rows': len(traces),
                    'algorithm_file': str(LCMD_PATH.relative_to(ROOT)), 'algorithm_sha256': INPUTS[str(LCMD_PATH.relative_to(ROOT))]['sha256'],
                    'new_training': 0, 'new_llm_calls': 0, 'candidate_truth_files_opened': 0,
                    'repository_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    (OUT / 'counterfactual_verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'counterfactual_input_manifest.json').write_text(json.dumps(list(INPUTS.values()), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: verification[k] for k in ['attempted_states', 'exact_prefix_verified', 'candidate_comparison_rows', 'lcmd_trace_rows']}), flush=True)


def summaries(frame, candidates):
    measures = [key for key in frame if key.startswith(('llm16_', 'lcmd_back16_', 'delta_'))]
    out = []
    for scope, subset in [('all_available', frame), ('common_1453', frame[frame.L_after <= 1453])]:
        for keys in [['seed'], ['seed', 'phase']]:
            summary = subset.groupby(keys)[measures].mean().reset_index()
            summary['batches'] = subset.groupby(keys).size().to_numpy()
            summary['scope'] = scope
            summary['level'] = 'seed' if len(keys) == 1 else 'seed_phase'
            out += summary.to_dict('records')
    pd.DataFrame(out).to_csv(OUT / 'counterfactual_seed_phase.csv', index=False)
    wins = []
    for (seed, ph), group in frame.groupby(['seed', 'phase']):
        for key in measures:
            if not key.startswith('delta_'):
                continue
            wins.append({'seed': seed, 'phase': ph, 'metric': key, 'batches': len(group),
                         'positive_n': int((group[key] > 1e-12).sum()), 'negative_n': int((group[key] < -1e-12).sum()),
                         'median_delta': group[key].median(), 'min_delta': group[key].min(), 'max_delta': group[key].max()})
    pd.DataFrame(wins).to_csv(OUT / 'counterfactual_paired_directions.csv', index=False)
    conditions = []
    for keys, group in candidates.groupby(['seed', 'phase', 'group']):
        for factor in ['PE_EA', 'sample_volume_ul', 'loading_volume_ul', 'loading_solvent']:
            for level, count in group[factor].value_counts().items():
                conditions.append(dict(zip(['seed', 'phase', 'group'], keys)) | {'factor': factor, 'level': level, 'count': count, 'fraction': count / len(group)})
    pd.DataFrame(conditions).to_csv(OUT / 'counterfactual_condition_levels.csv', index=False)
    llm = candidates[candidates.group == 'llm16']
    llm.groupby(['seed', 'phase'])[[key for key in candidates if key.startswith('reason_') and key != 'reason_status']].mean().reset_index().to_csv(OUT / 'counterfactual_reason_factors.csv', index=False)


def write_report(frame, candidates, state_rows):
    def table(data):
        data = data.copy()
        for key in data:
            if pd.api.types.is_numeric_dtype(data[key]):
                data[key] = data[key].map(lambda v: f'{v:.5g}')
        return '| ' + ' | '.join(data.columns) + ' |\n|' + '|'.join(['---'] * len(data.columns)) + '|\n' + '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in data.to_numpy())
    cols = ['llm_cf_candidate_overlap', 'llm16_cw_distance_pool_percentile', 'lcmd_back16_cw_distance_pool_percentile',
            'llm16_cw_gain_after_n_relative_h', 'lcmd_back16_cw_gain_after_n_relative_h',
            'llm16_single_factor_history_fraction', 'lcmd_back16_single_factor_history_fraction',
            'llm16_new_molecules_hn', 'lcmd_back16_new_molecules_hn']
    summary = frame.groupby('seed')[cols].mean().reset_index()
    ph = frame.groupby(['seed', 'phase'])[['delta_llm_minus_lcmd_cw_gain_after_n_relative_h', 'delta_llm_minus_lcmd_joint_gain_after_n', 'delta_llm_minus_lcmd_molecular_gain_after_n', 'delta_llm_minus_lcmd_single_sample_volume_history_fraction', 'delta_llm_minus_lcmd_single_pe_history_fraction', 'delta_llm_minus_lcmd_width_mean']].mean().reset_index()
    widths = frame.groupby('seed')[['llm16_pred_V1_q50_mean', 'lcmd_back16_pred_V1_q50_mean', 'llm16_pred_V2_q50_mean', 'lcmd_back16_pred_V2_q50_mean', 'llm16_width_mean', 'lcmd_back16_width_mean', 'llm16_same_molecule_pairs_fraction', 'lcmd_back16_same_molecule_pairs_fraction']].mean().reset_index()
    controls = frame.groupby('seed')[['llm16_single_sample_volume_history_fraction', 'lcmd_back16_single_sample_volume_history_fraction', 'llm16_single_sample_volume_hn_fraction', 'lcmd_back16_single_sample_volume_hn_fraction', 'llm16_full32_single_sample_volume_pairs', 'lcmd_back16_full32_single_sample_volume_pairs', 'llm16_single_pe_history_fraction', 'lcmd_back16_single_pe_history_fraction']].mean().reset_index()
    negative_cw = int((frame.delta_llm_minus_lcmd_cw_gain_after_n_relative_h < -1e-12).sum())
    positive_cw = int((frame.delta_llm_minus_lcmd_cw_gain_after_n_relative_h > 1e-12).sum())
    sample_higher = int((frame.delta_llm_minus_lcmd_single_sample_volume_hn_fraction > 1e-12).sum())
    overlap_zero = int((frame.llm_cf_candidate_overlap == 0).sum())
    report = f'''# Hybrid中LLM16替代了哪些LCMD候选？

**新增发现：LLM替代行为最突出的额外内容是现有梯度算法不选的上样量对照，同时减少了仍有明显边际收益的CW几何覆盖；这一取舍不是普遍的“条件对照 vs 不做对照”。** {overlap_zero}/{len(frame)}批两套后16无共同ID。样品体积单因素对照在LCMD后16中全部为零，LLM相对H+前16则在{sample_higher}/{len(frame)}批更高、其余持平。PE/EA对照在两组都很常见。冻结状态候选选择的分工得到直接识别，但实际训练收益仍未知。

## 新计算与精确复原资格

本模块尝试{len(state_rows)}个真实Hybrid冻结状态，{len(frame)}个状态通过有序前16一致性核验。全部原始bank为3330×512的CW CountSketch缓存，存储float32；直接复用缓存，不重算CW梯度、不加载模型。LCMD纯NumPy实现来自`src/qgeognn_al/active_learning_v2/lcmd.py`，计算转float64，以平方欧氏距离指定最近中心，选择最近中心簇内距离平方和最大的簇，再选择该簇最远点。`argmin/argmax`按原输入顺序处理并列；新中心只有距离严格更小时更新归属。每个状态保留训练/候选的原始有序canonical索引；32步逐步trace落盘。没有用farthest-first代替。

只有重建32步的前16与历史pending16逐个有序相同，才保留第17–32为合格候选集合反事实。排除项见`counterfactual_state_coverage.csv`。checkpoint、bank SHA、catalog与freeze哈希、H/U有序身份逐项核验。旧封存检查先运行；本次所有文件写入新目录，历史实验不变。候选预测仅取当轮catalog，未查询反事实候选真值，也不计算反事实Row Gain。

## LLM增加与牺牲的内容

同一H、U、checkpoint和数值前16下，实际LLM16与反事实LCMD后16可做严格候选集合比较；两组不是训练收益反事实。以下先逐批计算，再按seed汇总（157为36批、6101为35批；可另查共同L1453窗口）。

{table(summary)}

CW覆盖是固定128个U0参考ID到历史集合平均最近距离的下降。比较起点均为H+实际数值16，表中相对贡献除以同轮选前H参考距离；从未把不同checkpoint的原始距离合并成静态坐标。新分子是H+数值16尚未出现的SMILES数量。单因素对照以同分子、五个记录因素中恰一个不同定义；有科学对照含义，不能自动叫作冗余或低效。

**观察→证据→解释→边界：** LCMD后16并不是前16之后没有几何收益的补位：seed157/6101的给定前16CW边际覆盖分别约为起始参考距离的5.009%/5.333%，实际LLM为1.523%/1.173%。对应数值前16自身为7.079%/6.283%；后16边际虽往往递减，却远非接近零。LLM减去后16的CW贡献在{negative_cw}批为负、{positive_cw}批为正、其余平；两个seed的每个预定阶段均值都为负。因此平均梯度几何覆盖确有牺牲，但“每批LLM都更差”被反例否定，且几何损失不是Row学习损失。

## 被替换的是哪一类条件实验？

{table(controls)}

这里最清楚的差异是**样品体积对照关系**。LCMD后16相对H、相对H+数值前16的样品体积单因素名额都为零，完整数值32内样品体积only pair也全部为零。实际LLM在H+数值前16下约45.49%/52.32%的名额有此锚点；而完整Hybrid32内部仅平均1.194/1.086个这种实验对。两种分母不同：多数LLM条件对照以已知历史为锚，不是重复购买批内相同实验。

PE/EA历史对照比例则LLM约75.35%/75.71%、后16约71.18%/71.25%；不能笼统声称数值方法不选条件对照。平均样品体积本身也很接近或跨seed异向：157约97.83对97.55μL，6101约97.77对103.91μL。差异在“固定同分子其他因素进行对照”的选择结构，而非简单偏好较大上样量。加载体积和加载溶剂only对照稀少，完整表保留而不夸大。

上述结论还由`verify_counterfactual.py`独立以Fraction/Decimal逐因素枚举142个比较子批复核，没有调用被测condition函数；与主表最大误差为零。LCMD实现另用20套小整数例子，独立标量重算每一步簇分配、平方距离和、最远点及并列顺序，逐步trace完全一致。

## 阶段一致性与反例

以下均为LLM减去LCMD后16的配对差，CW列为相对于H起始距离的边际覆盖下降，joint/molecular列为固定结构/条件几何的绝对覆盖增量差。完整方向计数、逐批极值与条件水平分布分别见`counterfactual_paired_directions.csv`和`counterfactual_condition_levels.csv`。

{table(ph)}

静态分子空间为Morgan radius2/2048 Tanimoto距离；条件空间采用旧审计固定、无标签的四个连续范围与一个溶剂类别，联合距离各占一半。它们不等同于CW梯度空间。不能因某一覆盖变大而断定学习更好。

LLM相对后16的新增分子优势主要出现在L333–813，后续状态已没有未覆盖分子空间，不能从后期零分子增量推断探索意愿下降。联合静态覆盖差的逐批方向混合，157为14正/19负/3平，6101为19正/15负/1平；“LLM普遍增加一切覆盖”不成立。样品体积对照差却随预算明显扩大，L1005–1453两seed的LLM额外对照名额分别约61.61/72.32个百分点；这是与模型条件响应审计值得连接的机制信号，而不是通用切换阈值。

## 响应尺度、宽度与重复结构

{table(widths)}

q50为预测保留体积，不是候选实验价值；q90−q10为现有单模型区间宽度，未校准为epistemic uncertainty。两方案均取同一状态预测，不使用未来模型。重复SMILES的候选对比例衡量条件探索集中程度，不能单凭此指标认定浪费。

LLM内部61个样品体积only实验对的预测V1/V2 q50绝对变化均值为0.3066/0.2154mL；同组122个PE/EAonly对为5.394/11.598mL。分seed样品体积对分别34/27个，预测变化0.4047/0.3400及0.1831/0.05846mL。LCMD后16内部只有14个PE/EAonly对，没有样品体积only对。此处只描述候选当前预测，**没有读取这些对的真值**；不同因素的真实变化尺度不同，不能仅凭预测差大小比较学习价值。它提供“LLM补充的部分条件实验在当前表示/预测中变化很小”的具体依据，需与已揭示反馈的独立模型响应审计共同解释。

LLM可见理由的透明规则统计保存在`counterfactual_reason_factors.csv`，候选级原文见`counterfactual_candidates.csv`。规则只识别PE/EA、上样量、加载溶剂/体积及结构转移措辞；LCMD反事实没有原LLM理由，保持缺失，不生成解释。理由中出现某条件不证明该条件是唯一选择原因。

## 结论边界与下一步

这次新增证据回答“在相同状态下替换了谁”，而非“替换后训练收益是多少”。实际两组ID重叠、CW与固定几何边际、条件对照、分子覆盖及预测尺度均可在同状态比较；真实响应及行级损失贡献仍不可识别。任何下一轮学习收益验证都需要另行授权新训练或未揭示训练池标签；本轮未执行。两seed只能支持机制描述，阶段结果不产生通用预算切换阈值。

## 复现与核验

先运行旧审计`run.py --check`，再运行本模块`counterfactual.py`。默认自动检查旧封存；`--limit`仅用于调试，不能用部分状态冒充全71批。`counterfactual_input_manifest.json`保存读取路径及SHA；`counterfactual_verification.json`记录有序前缀、覆盖率、算法SHA和零训练/零LLM/零真值查询。关键单元测试见`tests/test_counterfactual.py`。
'''
    (OUT / 'HYBRID_COUNTERFACTUAL.md').write_text(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    parser.add_argument('--skip-prior-check', action='store_true')
    args = parser.parse_args()
    run(args.limit, not args.skip_prior_check)


if __name__ == '__main__':
    main()
