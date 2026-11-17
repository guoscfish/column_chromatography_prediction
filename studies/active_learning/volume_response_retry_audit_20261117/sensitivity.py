#!/usr/bin/env python3
"""Frozen-checkpoint condition sensitivity with complete graph reconstruction, no fits."""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/volume_sensitivity_mpl')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import argparse, csv, hashlib, itertools, json, lzma, sys, io
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'studies/active_learning/nocw_free_selection_diagnostics_20261101'))
from condition_metrics import _canonical_conditions
from qgeognn_al.data import build_model_data, eluent_descriptor, LOADING_SOLVENT, qg
from qgeognn_al.models import load_predictor_checkpoint

SEEDS = (157, 887, 2357, 6101, 12203)
FIELDS = ('PE/EA', 'Density g/ml', 'V/ul', 'loading solvent', 'Volume of loading solvent/ul')
SOURCES = {}


def read(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    SOURCES[str(path.relative_to(ROOT))] = {'path': str(path.relative_to(ROOT)), 'sha256': digest, 'bytes': len(raw)}
    return raw


def load(path):
    return json.loads(read(path))


def write(name, rows):
    pd.DataFrame(rows).to_csv(HERE / name, index=False)


def metadata_frame(ids, metadata):
    return pd.DataFrame([{'sample_id': sid, 'canonical_smiles': metadata[sid]['smiles'],
                         **metadata[sid]['conditions'], 'V1_ml': 0., 'V2_ml': 0.} for sid in ids])


def ordinary_graphs(ids, metadata, cache, scaler):
    """Use the original public builder with explicit zero label sentinels."""
    atoms, angles = build_model_data(metadata_frame(ids, metadata), cache, pd.DataFrame(), scaler)
    for sid, atom in zip(ids, atoms):
        atom.sample_id = sid
        atom.sample_volume_ul = torch.tensor([float(metadata[sid]['conditions']['V/ul'])])
    return atoms, angles


def differentiable_graph(sid, metadata, cache, scaler, *, raw_volume=None, ea_fraction=None):
    """Reconstruct every edge field from physical conditions; retain a raw scalar leaf."""
    row = metadata[sid]
    conditions = row['conditions']
    graph = cache[row['smiles']]['graph']
    _, angle = ordinary_graphs([sid], metadata, cache, scaler)
    atom_ids = torch.from_numpy(np.stack([graph[n] for n in qg.atom_id_names], axis=1)).long()
    bond_ids = torch.from_numpy(np.stack([graph[n] for n in qg.bond_id_names], axis=1)).long()
    length = torch.from_numpy(np.asarray(graph['bond_length'], np.float32)).reshape(-1, 1)
    if ea_fraction is None:
        eluent = torch.tensor(eluent_descriptor(conditions['PE/EA']))
    else:
        eluent = (1 - ea_fraction) * torch.tensor(eluent_descriptor('1/0')) + ea_fraction * torch.tensor(eluent_descriptor('0/1'))
    minimum = torch.tensor(scaler['eluent']['min'], dtype=torch.float32)
    maximum = torch.tensor(scaler['eluent']['max'], dtype=torch.float32)
    eluent = (eluent - minimum) / (maximum - minimum + 1e-8)
    volume = torch.tensor(float(conditions['V/ul'])) if raw_volume is None else raw_volume
    extra = torch.stack([torch.tensor(LOADING_SOLVENT[conditions['loading solvent']]),
                         float(conditions['Density g/ml']) * volume,
                         torch.tensor(float(conditions['Volume of loading solvent/ul']))])
    count = len(bond_ids)
    edge_attr = torch.cat([bond_ids, length, eluent[None, :].expand(count, -1), extra[None, :].expand(count, -1)], dim=1)
    atom = Data(x=atom_ids, edge_index=torch.from_numpy(graph['edges'].T).long(), edge_attr=edge_attr,
                y=torch.zeros((1, 2)), canonical_position=torch.tensor(0), sample_id=sid,
                sample_volume_ul=volume.reshape(1))
    return atom, angle[0]


def response_groups(ids, metadata, axis):
    groups = defaultdict(list)
    for sid in ids:
        c = _canonical_conditions(metadata[sid])
        key = (metadata[sid]['smiles'],) + c[:axis] + c[axis + 1:]
        groups[key].append(sid)
    return [(key, group) for key, group in groups.items()
            if len({_canonical_conditions(metadata[sid])[axis] for sid in group}) >= 2]


def scalar_level(sid, metadata, axis):
    value = _canonical_conditions(metadata[sid])[axis]
    return 1 / (1 + float(value)) if axis == 0 else float(value)


def forward_arrays(model, atoms, angles, batch_size=128):
    result = []
    with torch.no_grad():
        for start in range(0, len(atoms), batch_size):
            a = Batch.from_data_list(atoms[start:start + batch_size])
            b = Batch.from_data_list(angles[start:start + batch_size])
            result.append(model(a, b).numpy())
    return np.vstack(result)


def collision_audit(metadata):
    rows = []
    for sid, meta in metadata.items():
        c = _canonical_conditions(meta)
        rows.append({'sample_id': sid, 'smiles': meta['smiles'], 'density': float(c[1]),
                     'volume_ul': float(c[2]), 'amount_mg_graph_float32': float(np.float32(float(c[1]) * float(c[2]))),
                     'condition_key': repr((c[0], c[3], c[4]))})
    frame = pd.DataFrame(rows)
    molecular = frame.groupby('smiles').agg(rows=('sample_id', 'size'), distinct_density=('density', 'nunique'),
                   distinct_volume=('volume_ul', 'nunique')).reset_index()
    write('metadata_density_by_molecule.csv', molecular)
    collisions = []
    for amount, group in frame.groupby('amount_mg_graph_float32'):
        combinations = group[['density', 'volume_ul']].drop_duplicates()
        if len(combinations) > 1:
            collisions.append({'amount_mg': amount, 'rows': len(group), 'molecules': group.smiles.nunique(),
                               'distinct_density_volume_combinations': len(combinations),
                               'distinct_volumes': combinations.volume_ul.nunique()})
    write('metadata_global_amount_collisions.csv', collisions)
    local = []
    for (smiles, condition), group in frame.groupby(['smiles', 'condition_key']):
        values = group[['density', 'volume_ul', 'amount_mg_graph_float32']].drop_duplicates().to_dict('records')
        for a, b in itertools.combinations(values, 2):
            if a['volume_ul'] == b['volume_ul']:
                continue
            gap = abs(a['amount_mg_graph_float32'] - b['amount_mg_graph_float32'])
            if gap <= .1:
                local.append({'smiles': smiles, 'other_conditions': condition, 'density_a': a['density'], 'density_b': b['density'],
                    'volume_a_ul': a['volume_ul'], 'volume_b_ul': b['volume_ul'], 'amount_gap_mg': gap,
                    'exact_float32_alias': gap == 0})
    write('metadata_same_molecule_near_aliases.csv', pd.DataFrame(local, columns=[
        'smiles', 'other_conditions', 'density_a', 'density_b', 'volume_a_ul', 'volume_b_ul', 'amount_gap_mg', 'exact_float32_alias']))
    return {'metadata_rows': len(frame), 'molecules': len(molecular),
        'molecules_with_multiple_density': int((molecular.distinct_density > 1).sum()),
        'global_amount_values_with_multiple_density_volume': len(collisions),
        'same_molecule_other_conditions_fixed_alias_pairs_within_0p1mg': len(local),
        'same_molecule_exact_alias_pairs': sum(r['exact_float32_alias'] for r in local)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = parser.parse_args()
    torch.set_num_threads(2)
    qg.device = torch.device('cpu')
    snapshot = json.loads(lzma.decompress(read(ROOT / 'studies/active_learning/llm_decision_behavior_audit_20261116/states.json.xz')))
    metadata = snapshot['metadata']
    collisions = collision_audit(metadata)
    cache = torch.load(ROOT / 'experiments/e0_4g_baseline/graph_cache_4g.pt', map_location='cpu', weights_only=False)
    read(ROOT / 'experiments/e0_4g_baseline/graph_cache_4g.pt')
    assert all(set(v) >= {'graph', 'descriptor'} for v in cache.values())
    pairs = []
    gradients = []
    graph_checks = []
    activations = []
    for seed in args.seeds:
        graph_contract = load(ROOT / f'studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_{seed}/scrubbed_graphs.pt.contract.json')
        partition = pd.read_csv(io.BytesIO(read(ROOT / f'studies/active_learning/qgeognn_v2_row_sequential_b32/splits/row_seed_{seed}.csv')))
        assert partition.canonical_index.tolist() == list(range(len(partition)))
        assert hashlib.sha256(json.dumps(partition.to_dict('list'), sort_keys=True, separators=(',', ':')).encode()).hexdigest() == graph_contract['contract']['partition_hash']
        identity_to_position = dict(zip(partition['sample_id'], partition['canonical_index']))
        blind_path = ROOT / f'studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_{seed}/scrubbed_graphs.pt'
        assert hashlib.sha256(read(blind_path)).hexdigest() == graph_contract['sha256']
        blind_atom, blind_angle = torch.load(blind_path, map_location='cpu', weights_only=False)
        assert all(torch.count_nonzero(a.y) == 0 for a in blind_atom)
        for budget in (525, 1005):
            state = next(b for b in snapshot['states'] if b['method'] == 'new_free_cw_sol61_xhigh' and b['seed'] == seed and b['L_before'] == budget)
            directory = (ROOT / state['source']).parent
            freeze = load(ROOT / state['source'])
            raw = read(directory / 'input.json')
            assert hashlib.sha256(raw).hexdigest() == freeze['artifact_hashes']['input.json']
            packet = json.loads(raw)
            columns = packet['observed']['columns']
            observed = {values[columns.index('id')]: (float(values[columns.index('V1_ml')]), float(values[columns.index('V2_ml')]))
                        for values in packet['observed']['rows']}
            assert set(observed) == set(state['labeled_ids'])
            runtime = Path(str(directory).replace('/selections/', '/runtime/'))
            receipt = load(runtime / 'prediction_freeze.json')
            assert receipt['checkpoint_sha256'] == state['checkpoint_sha256']
            checkpoint = Path(receipt['checkpoint_path'])
            assert hashlib.sha256(read(checkpoint)).hexdigest() == state['checkpoint_sha256']
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model = load_predictor_checkpoint(checkpoint)
            scaler = saved['preprocessing']['scaler']
            assert scaler == graph_contract['contract']['preprocessing']['scaler']
            groups = {axis: response_groups(state['labeled_ids'], metadata, axis) for axis in (2, 0)}
            ids = sorted({sid for values in groups.values() for _, group in values for sid in group})
            atom, angle = ordinary_graphs(ids, metadata, cache, scaler)
            prediction = forward_arrays(model, atom, angle)
            lookup = {sid: prediction[k] for k, sid in enumerate(ids)}
            selected = sorted(ids, key=lambda sid: hashlib.sha256(('graph_equivalence_' + sid).encode()).hexdigest())[:3]
            for sid in selected:
                k = ids.index(sid)
                a, b = differentiable_graph(sid, metadata, cache, scaler)
                index = identity_to_position[sid]
                errors = [float((atom[k].edge_attr - blind_atom[index].edge_attr).abs().max()),
                          float((angle[k].edge_attr - blind_angle[index].edge_attr).abs().max()),
                          float((a.edge_attr - atom[k].edge_attr).abs().max())]
                assert max(errors) < 2e-5, (seed, budget, sid, errors)
                p = forward_arrays(model, [blind_atom[index]], [blind_angle[index]])[0]
                assert np.max(np.abs(p - lookup[sid])) < 1e-3
                graph_checks.append({'seed': seed, 'budget': budget, 'sample_id': sid,
                                     'rebuilt_vs_blind_max_feature_error': max(errors),
                                     'rebuilt_vs_blind_max_prediction_error': float(np.max(np.abs(p - lookup[sid])))})
            for axis, values in groups.items():
                for key, group in values:
                    for a, b in itertools.combinations(group, 2):
                        va, vb = scalar_level(a, metadata, axis), scalar_level(b, metadata, axis)
                        if va == vb:
                            continue
                        if va > vb:
                            a, b, va, vb = b, a, vb, va
                        for endpoint, col in enumerate((1, 4)):
                            actual = observed[b][endpoint] - observed[a][endpoint]
                            pred = float(lookup[b][col] - lookup[a][col])
                            pairs.append({'seed': seed, 'budget': budget, 'axis': FIELDS[axis],
                                'molecule': metadata[a]['smiles'], 'low_id': a, 'high_id': b,
                                'low_level': va, 'high_level': vb, 'target': f'V{endpoint + 1}',
                                'actual_delta_ml': actual, 'predicted_delta_ml': pred,
                                'meaningful_0p5ml': abs(actual) > .5, 'prediction_flat_0p1ml': abs(pred) < .1,
                                'correct_nonflat': abs(actual) > .5 and abs(pred) >= .1 and actual * pred > 0,
                                'wrong_nonflat': abs(actual) > .5 and abs(pred) >= .1 and actual * pred < 0,
                                'checkpoint_sha256': state['checkpoint_sha256'],
                                'role': 'in-sample known history at this original frozen checkpoint, not held-out performance'})
                # Deterministic metadata-only AD cases avoid choosing by favorable errors.
                key, group = min(values, key=lambda value: hashlib.sha256(repr(value[0]).encode()).hexdigest())
                sid = min(group, key=lambda value: scalar_level(value, metadata, axis))
                raw_value = scalar_level(sid, metadata, axis)
                leaf = torch.tensor(raw_value, dtype=torch.float32, requires_grad=True)
                a, b = differentiable_graph(sid, metadata, cache, scaler,
                                          **({'raw_volume': leaf} if axis == 2 else {'ea_fraction': leaf}))
                aa, bb = Batch.from_data_list([a]), Batch.from_data_list([b])
                pooled = torch_geometric_pool(model.backbone(aa, bb), aa.batch)
                residual, hidden = model.condition_branch(aa)
                pre = model.head[0](pooled + residual)
                backbone_pre = torch.nn.functional.linear(pooled, model.head[0].weight, None)
                branch_pre = torch.nn.functional.linear(residual, model.head[0].weight, None)
                final = torch.clamp(torch.relu(pre), min=0, max=1e8)
                step = 1. if axis == 2 else .001
                left = raw_value - step if axis == 2 else max(0., raw_value - step)
                right = raw_value + step if axis == 2 else min(1., raw_value + step)
                finite = []
                for value in (left, right):
                    if axis == 2:
                        changed = {sid: {'smiles': metadata[sid]['smiles'], 'conditions': {**metadata[sid]['conditions'], 'V/ul': value}}}
                    else:
                        changed = {sid: {'smiles': metadata[sid]['smiles'], 'conditions': {**metadata[sid]['conditions'], 'PE/EA': f'{1-value:.12g}/{value:.12g}'}}}
                    x, y = ordinary_graphs([sid], changed, cache, scaler)
                    finite.append(forward_arrays(model, x, y)[0])
                for endpoint, col in enumerate((1, 4)):
                    dp = torch.autograd.grad(pre[0, col], leaf, retain_graph=True, allow_unused=True)[0]
                    dq = torch.autograd.grad(final[0, col], leaf, retain_graph=True, allow_unused=True)[0]
                    db = torch.autograd.grad(backbone_pre[0, col], leaf, retain_graph=True, allow_unused=True)[0]
                    dr = torch.autograd.grad(branch_pre[0, col], leaf, retain_graph=True, allow_unused=True)[0]
                    gradients.append({'seed': seed, 'budget': budget, 'axis': FIELDS[axis], 'sample_id': sid,
                        'molecule': metadata[sid]['smiles'], 'raw_level': raw_value, 'target': f'V{endpoint + 1}',
                        'ad_pre_relu': float(dp) if dp is not None else 0., 'ad_final': float(dq) if dq is not None else 0.,
                        'ad_backbone_path': float(db) if db is not None else 0.,
                        'ad_completion_branch_path': float(dr) if dr is not None else 0.,
                        'finite_difference': float((finite[1][col] - finite[0][col]) / (right - left)),
                        'fd_raw_step': right - left, 'head_pre_activation': float(pre[0, col].detach()),
                        'head_relu_active': bool(pre[0, col] > 0), 'typed_hidden_relu_active_fraction': float((hidden > 0).float().mean()),
                        'condition_residual_norm': float(residual.norm().detach()), 'backbone_pooled_norm': float(pooled.norm().detach())})
            with torch.no_grad():
                aa, bb = Batch.from_data_list(atom), Batch.from_data_list(angle)
                nodes = model.backbone(aa, bb)
                pooled = torch_geometric_pool(nodes, aa.batch)
                residual, hidden = model.condition_branch(aa)
                pre = model.head[0](pooled + residual)
                for endpoint, col in enumerate((1, 4)):
                    activations.append({'seed': seed, 'budget': budget, 'target': f'V{endpoint + 1}', 'rows': len(ids),
                        'q50_pre_relu_nonpositive_fraction': float((pre[:, col] <= 0).float().mean()),
                        'typed_hidden_relu_active_fraction': float((hidden > 0).float().mean()),
                        'condition_output_weight_norm': float(model.condition_branch.output.weight.norm()),
                        'condition_residual_norm_mean': float(residual.norm(dim=1).mean()),
                        'backbone_pooled_norm_mean': float(pooled.norm(dim=1).mean())})
            del model, saved, atom, angle
            print('sensitivity', seed, budget, len(ids), flush=True)
        del blind_atom, blind_angle
    write('frozen_response_pairs.csv', pairs)
    write('frozen_sensitivity_gradients.csv', gradients)
    write('frozen_activation_checks.csv', activations)
    write('graph_reconstruction_checks.csv', graph_checks)
    frame = pd.DataFrame(pairs)
    summaries = []
    for (seed, budget, axis, target), group in frame.groupby(['seed', 'budget', 'axis', 'target']):
        meaningful = group[group.meaningful_0p5ml]
        summaries.append({'seed': seed, 'budget': budget, 'axis': axis, 'target': target, 'pairs': len(group),
            'molecules': group.molecule.nunique(), 'observed_abs_mean': group.actual_delta_ml.abs().mean(),
            'predicted_abs_mean': group.predicted_delta_ml.abs().mean(),
            'difference_mae': (group.actual_delta_ml - group.predicted_delta_ml).abs().mean(),
            'meaningful_pairs': len(meaningful), 'flat_fraction_meaningful': meaningful.prediction_flat_0p1ml.mean(),
            'correct_nonflat_fraction_meaningful': meaningful.correct_nonflat.mean(),
            'wrong_nonflat_fraction_meaningful': meaningful.wrong_nonflat.mean()})
    write('frozen_response_seed_budget.csv', summaries)
    for p in (Path(__file__), ROOT / 'src/qgeognn_al/data.py', ROOT / 'src/qgeognn_al/models/qgeognn_v2.py',
              ROOT / 'src/qgeognn_al/schemas/conditions.py', ROOT / 'src/qgeognn_al/training/predictor.py', ROOT / 'application/QGeoGNN.py'):
        read(p)
    (HERE / 'sensitivity_manifest.json').write_text(json.dumps({'status': 'PASS', 'new_fits': 0, 'new_llm_calls': 0,
        'new_label_queries': 0, 'test_truth_reads': 0, 'checkpoint_states': len(args.seeds) * 2,
        'known_history_truth_role': 'same seed NewCW packet observed table exactly equals checkpoint H',
        'collision_metadata': collisions, 'sources': list(SOURCES.values())}, ensure_ascii=False, indent=2) + '\n')


def torch_geometric_pool(nodes, batch):
    from torch_geometric.nn import global_add_pool
    return global_add_pool(nodes, batch)


if __name__ == '__main__':
    main()
