#!/usr/bin/env python3
"""Audit LLM rerun differences from frozen candidate metadata and predictions only."""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import importlib.util
import io
import itertools
import json
import lzma
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PREVIOUS = ROOT / 'studies/active_learning/llm_decision_behavior_audit_20261116'
METHODS = ['historical_free_mixed_model', 'new_free_cw_sol61_xhigh', 'nocw']
AXES = ['PE/EA', 'Density g/ml', 'V/ul', 'loading solvent', 'Volume of loading solvent/ul']
FILES = {}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def stable(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def read(path, expected=None):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    raw = path.read_bytes()
    assert raw == path.read_bytes(), ('unstable input', str(path))
    sha = digest(raw)
    if expected:
        assert sha == expected, ('source hash mismatch', str(path))
    key = str(path.relative_to(ROOT))
    if key in FILES:
        assert FILES[key]['sha256'] == sha
    FILES[key] = {'path': key, 'sha256': sha, 'bytes': len(raw)}
    return raw


def load(path, expected=None):
    return json.loads(read(path, expected))


def phase(budget):
    return '333-525' if budget < 525 else '525-813' if budget < 813 else '813-1005' if budget < 1005 else '1005-1485'


def rank(values):
    return rankdata(np.asarray(values, float), method='average') / len(values)


def strip_cw_packet(packet):
    result = copy.deepcopy(packet)
    result.pop('packet_sha256', None)
    for table in ['candidates', 'pending', 'observed']:
        columns = result[table]['columns']
        if 'cw_distance_percentile' in columns:
            column = columns.index('cw_distance_percentile')
            columns.pop(column)
            for row in result[table]['rows']:
                row.pop(column)
    return result


def condition_tuple(row):
    c = row['conditions']
    a, b = str(c['PE/EA']).split('/')
    return (round(float(a) / float(b), 10), float(c['Density g/ml']), float(c['V/ul']),
            str(c['loading solvent']).strip(), float(c['Volume of loading solvent/ul']))


def group_demean(values, group):
    values = np.asarray(values, float)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values[:, None]
    group = np.asarray(group, int)
    count = np.bincount(group)
    means = np.column_stack([np.bincount(group, weights=values[:, j]) / count for j in range(values.shape[1])])
    centered = values - means[group]
    return (centered[:, 0], means[group, 0]) if one_dimensional else (centered, means[group])


def residualize(score, covariates, groups=None):
    """Describe a score projection over the entire legal pool, without response labels."""
    score = np.asarray(score, float)
    x = np.asarray(covariates, float)
    if groups is None:
        design = np.column_stack([np.ones(len(score)), x])
        beta = np.linalg.lstsq(design, score, rcond=1e-10)[0]
        residual = score - design @ beta
        degrees = int(np.linalg.matrix_rank(design))
    else:
        yc, _ = group_demean(score, groups)
        xc, _ = group_demean(x, groups)
        beta = np.linalg.lstsq(xc, yc, rcond=1e-10)[0]
        residual = yc - xc @ beta
        degrees = len(np.unique(groups)) + int(np.linalg.matrix_rank(xc))
    total = np.sum((score - score.mean()) ** 2)
    r2 = float(1 - np.sum(residual ** 2) / total) if total > 1e-15 else np.nan
    return residual, r2, degrees


def matched_excess(score, mask, groups, minimum=5):
    """Exclude every accepted ID from the comparison pool."""
    scores = np.asarray(score, float)
    mask = np.asarray(mask, bool)
    references = {}
    for i, group in enumerate(groups):
        if not mask[i]:
            references.setdefault(group, []).append(scores[i])
    differences = []
    for i in np.flatnonzero(mask):
        candidates = references.get(groups[i], [])
        if len(candidates) >= minimum:
            differences.append(scores[i] - np.mean(candidates))
    return (float(np.mean(differences)) if differences else np.nan, len(differences))


def factor_allocation(selected, history, metadata):
    history_maps = [{} for _ in AXES]
    for sid in history:
        row = metadata[sid]
        c = condition_tuple(row)
        for j in range(5):
            key = (row['smiles'], c[:j] + c[j + 1:])
            history_maps[j].setdefault(key, set()).add(c[j])
    result, pairs = {}, {name: [] for name in AXES}
    for j, name in enumerate(AXES):
        supported = 0
        for sid in selected:
            row = metadata[sid]
            c = condition_tuple(row)
            values = history_maps[j].get((row['smiles'], c[:j] + c[j + 1:]), set())
            supported += any(v != c[j] for v in values)
        result[name] = supported / len(selected)
    for first, second in itertools.combinations(selected, 2):
        if metadata[first]['smiles'] != metadata[second]['smiles']:
            continue
        a, b = condition_tuple(metadata[first]), condition_tuple(metadata[second])
        changed = [j for j in range(5) if a[j] != b[j]]
        if len(changed) == 1:
            pairs[AXES[changed[0]]].append((first, second))
    return result, pairs


def model_covariates(cards):
    q1 = np.array([v['pred_V1_q50_ml'] for v in cards])
    q2 = np.array([v['pred_V2_q50_ml'] for v in cards])
    r1, r2 = rank(q1) - .5, rank(q2) - .5
    response = np.column_stack([r1, r2, r1 ** 2, r2 ** 2, r1 * r2])
    c = np.array([[1 / (1 + condition_tuple(v)[0]), condition_tuple(v)[1], condition_tuple(v)[2], condition_tuple(v)[4]] for v in cards])
    c = np.column_stack([rank(c[:, j]) - .5 for j in range(4)])
    solvent = pd.get_dummies(pd.Series([condition_tuple(v)[3] for v in cards]), dtype=float).to_numpy()
    conditions = np.column_stack([c, c ** 2, solvent])
    molecule_labels, group = np.unique([v['smiles'] for v in cards], return_inverse=True)
    return response, conditions, group, q1, q2


def accepted_receipt(directory, selection, freeze):
    matches = []
    names = sorted(n for n in freeze['artifact_hashes'] if Path(n).name.startswith('turn_') and n.endswith('.json') and not n.endswith('.started.json'))
    if not names:
        names = sorted(p.name for p in directory.glob('turn_*.json') if not p.name.endswith('.started.json'))
    for name in names:
        receipt = load(directory / name, freeze['artifact_hashes'].get(name))
        answer = receipt.get('answer')
        try:
            answer = json.loads(answer) if isinstance(answer, str) else answer
        except json.JSONDecodeError:
            continue
        if answer == selection:
            provenance = receipt.get('provenance', {})
            usage = provenance.get('usage', {})
            details = usage.get('output_tokens_details', {})
            matches.append({'served_model': provenance.get('served_model', ''),
                            'reasoning_effort': provenance.get('reasoning_effort', ''),
                            'input_tokens': usage.get('input_tokens'), 'output_tokens': usage.get('output_tokens'),
                            'reasoning_tokens': details.get('reasoning_tokens'),
                            'accepted_receipt': str((directory / name).relative_to(ROOT)),
                            'accepted_request_sha256': receipt.get('request_sha256', '')})
    if len(matches) == 1:
        return {**matches[0], 'original_frozen_turns_n': len(names), 'unique_accepted_receipt_match': True}
    return {'served_model': '', 'reasoning_effort': '', 'original_frozen_turns_n': len(names),
            'unique_accepted_receipt_match': False, 'accepted_receipt_matches': len(matches)}


def prompt_audit(states):
    paths = {}
    for state in states:
        if state['round'] == 0 and state['method'] == METHODS[0]:
            directory = (ROOT / state['source']).parent
            paths[state['seed']] = directory.parents[3] / 'selector_prompt.txt'
    prompts = {seed: read(path).decode() for seed, path in paths.items()}
    assert len({stable(v) for v in prompts.values()}) == 1
    historical = next(iter(prompts.values()))
    new = historical.replace('Pending-CW outcomes are identified separately.\n', '')
    source = ROOT / 'src/qgeognn_al/active_learning_v2/nocw_inputs.py'
    tree = ast.parse(read(source).decode())
    cw_description = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and
                          any(isinstance(t, ast.Name) and t.id == 'CW_DESCRIPTION' for t in n.targets))
    nocw = new.replace(cw_description, '')
    protocol = load(ROOT / 'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101/protocol.lock.json')
    assert stable(new) == protocol['prompt_hashes']['cw_aware']
    read(ROOT / 'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101/SCHEMA_DIFF.md')
    read(ROOT / 'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101/frozen_source/scripts/studies/run_free_cw_sol61_five_20261101.py')
    read(ROOT / 'scripts/studies/run_qgeognn_v2_row_llm_nocw_five.py')
    return {'historical': {'characters': len(historical), 'stable_json_sha256': stable(historical)},
            'new_cw': {'characters': len(new), 'stable_json_sha256': stable(new)},
            'nocw': {'characters': len(nocw), 'stable_json_sha256': stable(nocw)},
            'historical_new_difference': 'New removes exactly: Pending-CW outcomes are identified separately.\\n',
            'removed_characters': len(historical) - len(new),
            'new_nocw_difference': cw_description,
            'source_check': 'New prompt reconstructed from frozen Historical prompt and validated against protocol.lock hash; NoCW follows actual five-seed adapter.'}


def main(args):
    snapshot = json.loads(lzma.decompress(read(args.snapshot)))
    states = [x for x in snapshot['states'] if x['method'] in METHODS]
    assert len(states) == 540
    metadata = snapshot['metadata']
    oldpath = ROOT / 'studies/active_learning/nocw_free_selection_diagnostics_20261101/snapshot_inputs.json.xz'
    old = json.loads(lzma.decompress(read(oldpath)))
    geom = pd.read_csv(io.BytesIO(read(PREVIOUS / 'behavior_geometry.csv'))) if (PREVIOUS / 'behavior_geometry.csv').exists() else pd.read_csv(io.BytesIO(read(ROOT / 'studies/active_learning/comprehensive_history_diagnostics_20261112/tables/behavior_geometry.csv')))
    spec = importlib.util.spec_from_file_location('old_preferences_readonly', PREVIOUS / 'preferences.py')
    pref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pref)
    read(PREVIOUS / 'preferences.py')
    prompts = prompt_audit(states)
    initial_catalogs, initial_packets, catalogs = {}, {}, {}
    for state in states:
        if state['round'] != 0:
            continue
        directory = (ROOT / state['source']).parent
        freeze = load(directory / 'batch_freeze.json', state['source_sha256'])
        contract = load(directory / 'contract.json', freeze['artifact_hashes']['contract.json'])
        catalog = load(directory / 'catalog.json', freeze['artifact_hashes']['catalog.json'])
        packet = load(directory / 'input.json', freeze['artifact_hashes']['input.json'])
        assert contract['L_t_ids'] == state['labeled_ids'] and contract['U_t_ids'] == state['unlabeled_ids']
        initial_catalogs[(state['method'], state['seed'])] = catalog
        initial_packets[(state['method'], state['seed'])] = packet
    input_audits = []
    for seed in [157, 887, 2357, 6101, 12203]:
        triples = [next(b for b in states if b['method'] == method and b['seed'] == seed and b['round'] == 0) for method in METHODS]
        assert len({b['checkpoint_sha256'] for b in triples}) == 1
        assert all(b['labeled_ids'] == triples[0]['labeled_ids'] and b['unlabeled_ids'] == triples[0]['unlabeled_ids'] for b in triples)
        h, n, z = [initial_packets[(method, seed)] for method in METHODS]
        assert h == n
        assert strip_cw_packet(n) == strip_cw_packet(z)
        assert h['memory'] == n['memory'] == z['memory']
        assert all(not value for key, value in h['memory'].items() if key != 'memory_policy')
        input_audits.append({'seed': seed, 'same_checkpoint': True, 'same_ordered_L_U': True,
                             'historical_new_packet_equal': h == n, 'new_nocw_only_CW_column': True,
                             'empty_identical_initial_memory': True, 'historical_packet_sha256': h['packet_sha256'],
                             'new_packet_sha256': n['packet_sha256'], 'nocw_packet_sha256': z['packet_sha256'],
                             'full_system_prompt_identical_historical_new': False, 'system_prompt_removed_characters': 47,
                             'strict_shared_feature_state': True, 'strict_identical_full_request': False})
    allocation, levels, adjustments, matches, initials, selected_rows, contexts = [], [], [], [], [], [], []
    structures = {}
    for smiles in {v['smiles'] for v in metadata.values()}:
        molecule = Chem.MolFromSmiles(smiles)
        structures[smiles] = {'MW': Descriptors.MolWt(molecule), 'LogP': Descriptors.MolLogP(molecule),
                              'TPSA': rdMolDescriptors.CalcTPSA(molecule), 'HBD': Lipinski.NumHDonors(molecule),
                              'nitrogen': sum(a.GetAtomicNum() == 7 for a in molecule.GetAtoms())}
    for count, state in enumerate(states):
        method, seed, rnd = state['method'], state['seed'], state['round']
        directory = (ROOT / state['source']).parent
        freeze = load(directory / 'batch_freeze.json', state['source_sha256'])
        contract = load(directory / 'contract.json', freeze['artifact_hashes']['contract.json'])
        assert contract['checkpoint_sha256'] == state['checkpoint_sha256']
        assert set(contract['L_t_ids']) == set(state['labeled_ids']) and set(contract['U_t_ids']) == set(state['unlabeled_ids'])
        catalog = initial_catalogs[(method, seed)] if rnd == 0 else load(directory / 'catalog.json', freeze['artifact_hashes']['catalog.json'])
        cmap = {v['id']: v for v in catalog['candidates'] + catalog['pending']}
        uid = state['unlabeled_ids']
        assert set(uid) == set(cmap) and set(state['selected_ids']) == set(freeze['batch_ids'])
        cards = [cmap[sid] for sid in uid]
        mask = np.array([sid in set(state['selected_ids']) for sid in uid])
        assert mask.sum() == 32
        response, condition, group, q1, q2 = model_covariates(cards)
        w1 = np.array([v['pred_V1_q90_ml'] - v['pred_V1_q10_ml'] for v in cards])
        w2 = np.array([v['pred_V2_q90_ml'] - v['pred_V2_q10_ml'] for v in cards])
        scores = {'U_proxy': rank(.5 * (rank(w1) + rank(w2)))}
        g_source = ''
        if method != 'nocw':
            scores['G'] = rank([v['cw_distance_percentile'] for v in cards])
            g_source = str((directory / 'catalog.json').relative_to(ROOT))
        elif rnd == 0:
            shared = initial_catalogs[(METHODS[1], seed)]
            shared = {v['id']: v for v in shared['candidates'] + shared['pending']}
            for sid in uid:
                assert {k:v for k,v in shared[sid].items() if k != 'cw_distance_percentile'} == cmap[sid]
            scores['G'] = rank([shared[sid]['cw_distance_percentile'] for sid in uid])
            g_source = 'same-seed identical L333 checkpoint/catalog in New CW; omitted from NoCW visible input'
        else:
            g, g_source = pref.cached_cw_scores(state, old['partitions'], geom)
            if g is not None:
                scores['G'] = g
        key = {k: state[k] for k in ['method', 'seed', 'round', 'L_before', 'L_after']}
        key['phase'] = phase(state['L_before'])
        maps, pairs = factor_allocation(state['selected_ids'], state['labeled_ids'], metadata)
        for axis in AXES:
            ps = pairs[axis]
            row = {**key, 'factor': axis, 'history_single_factor_slot_fraction': maps[axis],
                   'within_batch_single_factor_pairs': len(ps), 'selected_n': 32,
                   'observability': 'candidate conditions and current frozen predictions; no outcome labels'}
            for target in ['V1', 'V2']:
                differences = [abs(cmap[a][f'pred_{target}_q50_ml'] - cmap[b][f'pred_{target}_q50_ml']) for a,b in ps]
                row[f'predicted_absolute_delta_{target}_mean'] = np.mean(differences) if differences else np.nan
                row[f'predicted_absolute_delta_{target}_median'] = np.median(differences) if differences else np.nan
            allocation.append(row)
            for level, number in pd.Series([str(metadata[sid]['conditions'][axis]) for sid in state['selected_ids']]).value_counts().items():
                levels.append({**key, 'factor': axis, 'level': level, 'slots': int(number), 'fraction': number/32})
        for feature, score in scores.items():
            variants = {'raw': (score-score.mean(), np.nan, 0),
                        'q50_polynomial': residualize(score, response),
                        'molecule_conditions': residualize(score, condition, group),
                        'q50_molecule_conditions': residualize(score, np.column_stack([response, condition]), group)}
            for name, (residual, r2, degrees) in variants.items():
                adjustments.append({**key, 'feature': feature, 'adjustment': name,
                                    'selected_excess_percentile': float(residual[mask].mean()), 'projection_R2': r2,
                                    'projection_rank': degrees, 'candidate_n': len(uid), 'selected_n': 32,
                                    'G_source': g_source if feature == 'G' else '', 'comparison': 'shared state' if rnd == 0 else 'descriptive across divergent states'})
            q1bin = np.minimum((rank(q1) * 5).astype(int), 4)
            q2bin = np.minimum((rank(q2) * 5).astype(int), 4)
            for name, grouping in [('q50_joint_quintiles', list(zip(q1bin, q2bin))),
                                    ('same_molecule', list(group)),
                                    ('same_molecule_q50V2_tertile', list(zip(group, np.minimum((rank(q2)*3).astype(int), 2))))]:
                difference, covered = matched_excess(score, mask, grouping)
                matches.append({**key, 'feature': feature, 'matching': name, 'matched_selected_n': covered,
                                'matched_selected_fraction': covered/32, 'matched_excess_percentile': difference,
                                'minimum_unselected_controls_per_pick': 5, 'selected_batch_excluded': True})
        packet_path = directory / 'input.json'
        packet = initial_packets[(method, seed)] if rnd == 0 else load(packet_path, freeze['artifact_hashes']['input.json'])
        memory = packet.get('memory', {})
        context = {**key, 'input_bytes': len(read(packet_path, freeze['artifact_hashes']['input.json'])), 'G_available': 'G' in scores,
                   'U_available': True, 'G_source': g_source, 'packet_sha256': packet['packet_sha256'],
                   'memory_characters':len(json.dumps(memory,ensure_ascii=False)),
                   'previous_hypotheses_n':len(memory.get('previous_hypotheses',[])),
                   'recent_outcomes_n':len(memory.get('recent_outcomes',[]))}
        if rnd == 0:
            selection = load(directory / 'selection.json', freeze['artifact_hashes']['selection.json'])
            receipt = accepted_receipt(directory, selection, freeze)
            context.update(receipt)
            context['memory_characters'] = len(json.dumps(packet['memory'], ensure_ascii=False))
            init = {**key, **receipt, 'checkpoint_sha256': state['checkpoint_sha256'],
                    'packet_sha256': packet['packet_sha256'], 'packet_bytes': packet_path.stat().st_size,
                    'initial_memory_empty': True, 'shared_state_verified': True,
                    'selected_unique_molecules': len({metadata[sid]['smiles'] for sid in state['selected_ids']}),
                    'sample_volume_mean': np.mean([metadata[sid]['conditions']['V/ul'] for sid in state['selected_ids']]),
                    'sample_volume_150_fraction': np.mean([metadata[sid]['conditions']['V/ul'] == 150 for sid in state['selected_ids']]),
                    'EA_fraction_mean': np.mean([1/(1+condition_tuple(metadata[sid])[0]) for sid in state['selected_ids']]),
                    'q50V1_mean': q1[mask].mean(), 'q50V2_mean': q2[mask].mean(),
                    'widthV1_mean': w1[mask].mean(), 'widthV2_mean': w2[mask].mean(),
                    'selected_G_percentile': scores['G'][mask].mean(), 'selected_U_percentile': scores['U_proxy'][mask].mean(),
                    'nitrogen_molecule_fraction': np.mean([structures[metadata[sid]['smiles']]['nitrogen'] > 0 for sid in state['selected_ids']]),
                    'HBD_molecule_fraction': np.mean([structures[metadata[sid]['smiles']]['HBD'] > 0 for sid in state['selected_ids']]),
                    'TPSA_mean': np.mean([structures[metadata[sid]['smiles']]['TPSA'] for sid in state['selected_ids']])}
            initials.append(init)
            for sid in state['selected_ids']:
                row = cmap[sid]
                index = uid.index(sid)
                selected_rows.append({**key, 'sample_id': sid, 'smiles': row['smiles'],
                                      **{axis:row['conditions'][axis] for axis in AXES},
                                      'pred_V1_q50_ml': row['pred_V1_q50_ml'], 'pred_V2_q50_ml': row['pred_V2_q50_ml'],
                                      'G_percentile': scores['G'][index], 'U_proxy_percentile': scores['U_proxy'][index]})
        contexts.append(context)
        if count % 100 == 0:
            print('LLM mechanism states',count,'/',len(states),flush=True)
    FILES.update(pref.INPUTS)
    for name, data in [('initial_input_audit.csv',input_audits),('initial_batch_comparison.csv',initials),
                       ('initial_selected_candidates.csv',selected_rows),('condition_factor_allocation.csv',allocation),
                       ('condition_level_allocation.csv',levels),('llm_score_adjustment.csv',adjustments),
                       ('llm_score_matching.csv',matches),('llm_context_audit.csv',contexts)]:
        pd.DataFrame(data).to_csv(args.output/name,index=False)
    (args.output/'llm_prompt_audit.json').write_text(json.dumps(prompts,ensure_ascii=False,indent=2)+'\n')
    pairwise = []
    for seed in [157,887,2357,6101,12203]:
        batch = {m:next(s for s in states if s['method']==m and s['seed']==seed and s['round']==0) for m in METHODS}
        for a,b in itertools.combinations(METHODS,2):
            first,second=set(batch[a]['selected_ids']),set(batch[b]['selected_ids'])
            mol_a,mol_b=({metadata[sid]['smiles'] for sid in ids} for ids in [first,second])
            pairwise.append({'seed':seed,'method_a':a,'method_b':b,'shared_checkpoint':True,
                             'selected_ID_overlap_n':len(first&second),'selected_ID_jaccard':len(first&second)/len(first|second),
                             'molecule_overlap_n':len(mol_a&mol_b),'molecule_jaccard':len(mol_a&mol_b)/len(mol_a|mol_b),
                             'same_actual_packet':initial_packets[(a,seed)]==initial_packets[(b,seed)],
                             'same_full_system_prompt':False})
    pd.DataFrame(pairwise).to_csv(args.output/'initial_pairwise_overlap.csv',index=False)
    allocation_frame=pd.DataFrame(allocation)
    numeric=['history_single_factor_slot_fraction','within_batch_single_factor_pairs']
    allocation_frame.groupby(['method','seed','phase','factor'])[numeric].mean().reset_index().to_csv(args.output/'llm_condition_allocation_seed_phase.csv',index=False)
    adjustment_frame=pd.DataFrame(adjustments)
    adjustment_frame.groupby(['method','seed','phase','feature','adjustment']).agg(
        states=('round','size'),mean_selected_excess=('selected_excess_percentile','mean'),mean_projection_R2=('projection_R2','mean')).reset_index().to_csv(args.output/'score_adjustment_seed_phase.csv',index=False)
    validation={'status':'PASS','states':len(states),'initial_states':len(initials),'initial_selected_events':len(selected_rows),
                'initial_same_checkpoint_and_ordered_L_U_all_five_seeds':True,'historical_new_initial_packet_equal_all_five_seeds':True,
                'new_nocw_packet_difference_only_CW_column_all_five_seeds':True,
                'same_full_request':False,'unobserved_truth_reads':0,'test_truth_reads':0,'new_training':0,'new_llm_calls':0,
                'observed_catalog_fields_used_as_response_targets':False,
                'G_states':int(sum(x['G_available'] for x in contexts)),'U_states':len(states),
                'score_adjustment':'within-state full-pool projection; not candidate value prediction, causal effect, calibrated uncertainty, or an OOF predictive model',
                'sources':list(FILES.values())}
    (args.output/'llm_reproducibility_inputs.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:validation[k] for k in ['states','initial_states','G_states','U_states','status']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',type=Path,default=PREVIOUS/'states.json.xz')
    parser.add_argument('--output',type=Path,default=OUT)
    main(parser.parse_args())
