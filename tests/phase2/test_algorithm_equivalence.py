"""Exact migrated-kernel regressions against pinned Git source; no training/LLM."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pytest
from src.qgeognn_al.al.acquisitions import select
from src.qgeognn_al.al.acquisitions.lcmd import lcmd_tp_select
from src.qgeognn_al.al.acquisitions.maxdet import conditional_gradient_maxdet
from src.qgeognn_al.al.acquisitions.ivr import conditional_batch_ivr
from src.qgeognn_al.al.acquisitions.hybrid import hybrid_select
from src.qgeognn_al.al.acquisitions.random import random_control_positions, random_control_seed
from src.qgeognn_al.al.representations import Authorization, FeatureSnapshot, RepresentationConfig
from src.qgeognn_al.infra.contracts import BaselineProvenance, canonical_bytes
import legacy_reference as legacy

RECORDS = []


@pytest.fixture(scope='module', autouse=True)
def evidence():
    yield
    destination = os.getenv('QGEOGNN_EQUIVALENCE_OUTPUT')
    if destination:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'source_commit': legacy.FROZEN_COMMIT,
                                    'scope': 'synthetic deterministic boundary regression; no actual AL selection',
                                    'records': RECORDS}, indent=2)+'\n')


def record(method, case, old_ids, new_ids, trace=None):
    RECORDS.append({'method': method, 'case': case, 'old_selected_ids': old_ids,
                    'new_selected_ids': new_ids, 'selected_order_exact': old_ids == new_ids,
                    'trace_sha256': hashlib.sha256(canonical_bytes(trace or [])).hexdigest(),
                    'numerical_scores_and_trace_equal': True})


def inputs(case, dimension=7):
    if case == 'ordinary':
        matrix = np.random.default_rng(157).normal(size=(20, dimension)); count, batch = 4, 5
    elif case == 'ties':
        matrix = np.ones((12, dimension)); count, batch = 2, 3
    else:
        matrix = np.zeros((16, dimension)); matrix[:, 0] = np.linspace(.1, 9.9, 16)
        count, batch = 3, 3
    return matrix, count, batch


def snapshot(matrix, count, kind='gradient'):
    l = tuple(f'L{i}' for i in range(count)); u = tuple(f'U{i}' for i in range(len(matrix)-count))
    return FeatureSnapshot(authorization=Authorization(l, u), train_ids=l, candidate_ids=u,
                           train_features=matrix[:count], candidate_features=matrix[count:],
                           config=RepresentationConfig(kind, matrix.shape[1]),
                           predictor=BaselineProvenance('1'*40, '2'*64, '3'*64, '4'*64, '5'*64))


@pytest.mark.parametrize('case', ['ordinary', 'ties', 'rank_deficient'])
def test_lcmd_matches_ids_initial_distances_and_all_trace_fields(case):
    matrix, count, batch = inputs(case)
    old = legacy.lcmd().lcmd_tp_select(matrix[count:], matrix[:count], batch)
    new = lcmd_tp_select(matrix[count:], matrix[:count], batch)
    np.testing.assert_array_equal(new.selected_pool_positions, old.selected_pool_positions)
    np.testing.assert_array_equal(new.initial_nearest_center, old.initial_nearest_center)
    np.testing.assert_array_equal(new.initial_nearest_sq_distance, old.initial_nearest_sq_distance)
    assert new.trace == old.trace
    result = select(snapshot(matrix, count), acquisition='lcmd', budget=batch, seed=157)
    ids = [f'U{i}' for i in old.selected_pool_positions.tolist()]
    assert list(result.selected_ids) == ids
    record('LCMD', case, ids, list(result.selected_ids), result.trace)


@pytest.mark.parametrize('case', ['ordinary', 'ties', 'rank_deficient'])
def test_maxdet_matches_ids_gains_scores_and_all_trace_fields(case):
    matrix, count, batch = inputs(case)
    l, u = np.arange(count), np.arange(count, len(matrix))
    old = legacy.maxdet().conditional_gradient_maxdet(matrix, l, u, batch)
    new = conditional_gradient_maxdet(matrix, l, u, batch)
    np.testing.assert_array_equal(new.selected_positions, old.selected_positions)
    np.testing.assert_array_equal(new.initial_marginal_gains, old.initial_marginal_gains)
    assert new.normalization_scale == old.normalization_scale
    assert new.trace == old.trace
    assert new.audit == {k: v for k, v in old.audit.items() if k != 'elapsed_seconds'}
    result = select(snapshot(matrix, count), acquisition='maxdet', budget=batch, seed=157)
    ids = [f'U{i}' for i in old.selected_candidate_positions.tolist()]
    assert list(result.selected_ids) == ids
    record('MaxDet', case, ids, list(result.selected_ids), result.trace)


@pytest.mark.parametrize('case', ['ordinary', 'ties', 'rank_deficient'])
def test_ivr_matches_ids_variance_reductions_and_all_trace_fields(case):
    matrix, count, batch = inputs(case)
    l, u = np.arange(count), np.arange(count, len(matrix))
    old = legacy.ivr().conditional_batch_ivr(matrix, l, u, batch)
    new = conditional_batch_ivr(matrix, l, u, batch)
    assert new['selected_candidate_positions'] == old['selected_candidate_positions']
    assert new['trace'] == old['trace']
    assert new['audit'] == {k: v for k, v in old['audit'].items() if k != 'elapsed_seconds'}
    result = select(snapshot(matrix, count), acquisition='ivr', budget=batch, seed=157)
    ids = [f'U{i}' for i in old['selected_candidate_positions']]
    assert list(result.selected_ids) == ids
    record('IVR', case, ids, list(result.selected_ids), result.trace)


@pytest.mark.parametrize('case', ['ordinary', 'ties', 'rank_deficient'])
def test_hybrid_matches_legacy_ranking_shortlist_uncertainty_and_ids(case):
    matrix, count, batch = inputs(case, 128)
    values = np.random.default_rng(887).normal(size=(3, len(matrix)-count, 6))
    if case == 'ties': values[:] = 0
    old = legacy.hybrid().v2_hybrid_select(matrix, values, count, (2., 3.), batch)
    new = hybrid_select(matrix, values, count, (2., 3.), batch)
    assert old.keys() == new.keys()
    for key in old: np.testing.assert_array_equal(new[key], old[key])
    features = snapshot(matrix, count, 'latent')
    result = select(features, acquisition='hybrid', budget=batch, seed=157,
                    numerical_signals={'candidate_ids': features.candidate_ids,
                                       'ensemble_predictions': values, 'target_scales': (2., 3.)})
    ids = [f'U{i}' for i in old['selected_positions'].tolist()]
    assert list(result.selected_ids) == ids
    np.testing.assert_array_equal(result.audit['uncertainty_ranking'], old['uncertainty_ranking'])
    record('Hybrid', case, ids, list(result.selected_ids), result.trace)


@pytest.mark.parametrize('outer_seed,control', [(17, 0), (157, 2), (12203, 4)])
def test_historical_random_controls_preserve_effective_seed_and_global_rng(outer_seed, control):
    old = legacy.load('src/qgeognn_al/active_learning_v2/protocol.py', ['random_control_positions'], injected={'RANDOM_CONTROLS': 5})
    np.random.seed(99); expected = np.random.random(5); np.random.seed(99)
    before = old.random_control_positions(20, 5, outer_seed, control)
    after = random_control_positions(20, 5, outer_seed, control)
    np.testing.assert_array_equal(before, after)
    assert random_control_seed(outer_seed, control) == outer_seed*1_000_003 + 50_000 + control
    matrix, count, _ = inputs('ordinary')
    result = select(snapshot(matrix, count), acquisition='random', budget=5,
                    seed=random_control_seed(outer_seed, control))
    mapped = old.random_control_positions(len(matrix)-count, 5, outer_seed, control)
    assert result.selected_ids == tuple(f'U{i}' for i in mapped)
    np.testing.assert_array_equal(np.random.random(5), expected)
    record('random', f'{outer_seed}/{control}', before.tolist(), after.tolist())


def test_lcmd_degenerate_historical_error_is_not_silently_fixed():
    for function in (legacy.lcmd().lcmd_tp_select, lcmd_tp_select):
        with pytest.raises(RuntimeError, match='empty largest cluster'):
            function(np.array([[10.], [10.]]), np.array([[0.], [10.]]), 1)


def test_hybrid_rejects_foreign_identity_signals_and_extra_truth_channels():
    matrix, count, batch = inputs('ordinary', 128); features = snapshot(matrix, count, 'latent')
    signals = {'candidate_ids': tuple(reversed(features.candidate_ids)),
               'ensemble_predictions': np.zeros((3, len(features.candidate_ids), 6)), 'target_scales': (2., 3.)}
    with pytest.raises(ValueError, match='identity/order'):
        select(features, acquisition='hybrid', budget=batch, seed=157, numerical_signals=signals)
    signals['candidate_ids'] = features.candidate_ids; signals['test_truth'] = [1.]
    with pytest.raises(ValueError, match='only identity-bound'):
        select(features, acquisition='hybrid', budget=batch, seed=157, numerical_signals=signals)


@pytest.mark.parametrize('sketch_seed', [17, 157, 887])
def test_raw_and_cw_countsketch_match_every_component_and_authorized_extractor(sketch_seed):
    import torch
    from torch_geometric.data import Data
    from src.qgeognn_al.al.representations.gradient import (
        extract_q50_gradient_sketches, extract_linear_output_gradient_sketches, extract_gradient,
    )
    from src.qgeognn_al.al.representations.center_width import center_width_transform
    from src.qgeognn_al.al.representations.latent import GraphRow, predictor_state_hash
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.linear = torch.nn.Linear(1, 6)
        def forward(self, atom, angle):
            return self.linear(atom.x.reshape(-1, 1).float())
    torch.set_num_threads(1); torch.manual_seed(31)
    model = Toy().eval()
    atom = [Data(x=torch.tensor([[float(i+1)]]), edge_index=torch.zeros((2, 0), dtype=torch.long),
                 edge_attr=torch.zeros((0, 13)), canonical_position=torch.tensor(i), num_nodes=1) for i in range(8)]
    angle = [Data(edge_index=torch.zeros((2, 0), dtype=torch.long), edge_attr=torch.zeros((0, 17)), num_nodes=0) for _ in atom]
    ref = legacy.gradient()
    old = ref.extract_q50_gradient_sketches(model, atom, angle, list(range(8)), (2., 3.), dimension=512, sketch_seed=sketch_seed)
    new = extract_q50_gradient_sketches(model, atom, angle, list(range(8)), (2., 3.), dimension=512, sketch_seed=sketch_seed)
    np.testing.assert_array_equal(new.features, old.features)
    assert new.parameter_audit == old.parameter_audit
    assert {k:v for k,v in new.audit.items() if k != 'elapsed_seconds'} == {k:v for k,v in old.audit.items() if k != 'elapsed_seconds'}
    l0 = np.array([[1., 3.], [3., 5.], [2., 9.]])
    transforms = legacy.load('src/qgeognn_al/active_learning_v2/gradient_transforms.py', ['center_width_transform', '_validated_targets'])
    old_transform, old_audit = transforms.center_width_transform(l0)
    transform, audit = center_width_transform(l0)
    np.testing.assert_array_equal(transform, old_transform); assert audit == old_audit
    old_cw = ref.extract_linear_output_gradient_sketches(model, atom, angle, list(range(8)), old_transform, dimension=512, sketch_seed=sketch_seed)
    new_cw = extract_linear_output_gradient_sketches(model, atom, angle, list(range(8)), transform, dimension=512, sketch_seed=sketch_seed)
    np.testing.assert_array_equal(new_cw.features, old_cw.features)
    ids = tuple(f'G{i}' for i in range(8)); auth = Authorization(ids[:3], ids[3:])
    rows = [GraphRow(identity, a, b) for identity, a, b in zip(ids, atom, angle)]
    provenance = BaselineProvenance('1'*40, '2'*64, '3'*64, '4'*64, '5'*64)
    kwargs = dict(expected_state_hash=predictor_state_hash(model), authorization=auth, rows=rows,
                  provenance=provenance, sketch_seed=sketch_seed)
    public = extract_gradient(model, scales=(2., 3.), **kwargs)
    public_cw = extract_gradient(model, kind='center_width', l0_targets=l0, l0_ids=auth.train_ids, **kwargs)
    np.testing.assert_array_equal(np.vstack([public.train, public.candidates]), old.features)
    np.testing.assert_array_equal(np.vstack([public_cw.train, public_cw.candidates]), old_cw.features)
    assert public.config.dimension == 512 and public_cw.config.dimension == 512
    assert all(parameter.grad is None for parameter in model.parameters())
    assert public.extraction.predictor_state_hash == public.extraction.predictor_state_after_hash
    for kwargs_bad in [dict(l0_ids=('future_U',), l0_targets=[[1., 2.]]),
                       dict(l0_ids=auth.train_ids, l0_targets=[[1., 2.]])]:
        with pytest.raises(ValueError, match='identity-aligned'):
            extract_gradient(model, kind='center_width', **kwargs, **kwargs_bad)
    record('q50/CW CountSketch512', f'sketch_seed={sketch_seed}',
           [old.audit['sketch_mapping_sha256']], [new.audit['sketch_mapping_sha256']],
           {'parameter_count': new.audit['parameter_count'], 'raw_exact': True, 'cw_exact': True})


@pytest.mark.parametrize('case,alpha', [('ordinary', .5), ('ties', .8), ('rank_deficient', 0.)])
def test_fused_math_and_snapshot_composition_match_frozen_block_scaling(case, alpha):
    from src.qgeognn_al.al.representations.fused import fuse_gradient_latent_features, fuse_snapshots
    old = legacy.load('src/qgeognn_al/active_learning_v2/fused_features.py',
                      ['_array_hash', 'FusionResult', 'fuse_gradient_latent_features'],
                      injected={'hashlib': hashlib, 'GRADIENT_DIMENSION': 512,
                                'LATENT_DIMENSION': 128, 'FUSED_DIMENSION': 640, 'ALPHA': .5})
    g, count, _ = inputs(case, 512); h, _, _ = inputs(case, 128)
    before = old.fuse_gradient_latent_features(g, h, count, alpha=alpha)
    after = fuse_gradient_latent_features(g, h, count, alpha=alpha)
    np.testing.assert_array_equal(after.features, before.features)
    assert after.audit == before.audit
    composed = fuse_snapshots(snapshot(g, count), snapshot(h, count, 'latent'), alpha=alpha)
    np.testing.assert_array_equal(np.vstack([composed.train, composed.candidates]), before.features)
    params = json.loads(composed.config.parameters_json)
    assert params['alpha'] == alpha
    assert params['normalization'] == 'sqrt(mean_L_t(||block_i||_2^2)); L_t prefix only'
    assert composed.config.dimension == 640
    changed = h.copy(); changed[count:] *= 77
    # Candidate changes may affect vectors but must never re-estimate L block scales.
    check = fuse_gradient_latent_features(g, changed, count, alpha=alpha)
    assert check.gradient_scale == before.gradient_scale and check.latent_scale == before.latent_scale
    record('fused640', case, [before.audit['fused_feature_sha256']], [after.audit['fused_feature_sha256']])


@pytest.mark.parametrize('bad_case', ['duplicate', 'foreign', 'wrong_count', 'wrong_packet'])
def test_free_candidate_identity_contract_matches_historical_direct_delivery(bad_case):
    import copy
    import types
    from src.qgeognn_al.al.agents import make_packet, validate_bound_plan
    ids = tuple(f'U{i}' for i in range(64))
    rows = [{'sample_id': identity, 'rank': i, 'nearest_center_sq_distance': float(i)} for i, identity in enumerate(ids)]
    packet = make_packet(rows, candidate_ids=ids, seed=157, round_index=0, budget=32, protocol_hash='a'*64)
    response = {'packet_sha256': packet['packet_sha256'], 'selected_ids': list(ids[:32])}
    old = legacy.load('src/qgeognn_al/active_learning_v2/scientist_full_delivery.py', ['validate_direct'],
                      injected={'copy': copy, 'MAX_HYPOTHESES': 8,
                                'decode_table': lambda packet, table: [{'id': identity} for identity in ids]})
    catalog = types.SimpleNamespace(candidates={identity: {} for identity in ids}, observed={})
    old_packet = {'packet_sha256': packet['packet_sha256'], 'selection_count': 32, 'round': 0}
    old_response = {'type': 'selection', 'packet_sha256': packet['packet_sha256'],
                    'choices': [{'id': identity, 'reason': 'synthetic reason', 'hypothesis_id': None} for identity in ids[:32]],
                    'hypotheses': [], 'batch_strategy': 'synthetic', 'batch_rationale': 'synthetic', 'feedback_interpretation': ''}
    assert validate_bound_plan(response, packet) == ids[:32]
    assert [choice['id'] for choice in old.validate_direct(old_response, catalog, old_packet)['choices']] == list(ids[:32])
    if bad_case == 'duplicate':
        response['selected_ids'][-1] = response['selected_ids'][0]
        old_response['choices'][-1]['id'] = old_response['choices'][0]['id']
    elif bad_case == 'foreign':
        response['selected_ids'][-1] = 'test_ID'
        old_response['choices'][-1]['id'] = 'test_ID'
    elif bad_case == 'wrong_count':
        response['selected_ids'].pop(); old_response['choices'].pop()
    else:
        response['packet_sha256'] = 'b'*64; old_response['packet_sha256'] = 'b'*64
    with pytest.raises(ValueError): validate_bound_plan(response, packet)
    with pytest.raises(ValueError): old.validate_direct(old_response, catalog, old_packet)


def test_provided_packet_has_no_truth_channel_and_cannot_cross_seed_or_round():
    from src.qgeognn_al.al.agents import make_packet, validate_bound_plan
    rows = [{'sample_id': 'U0', 'rank': 0}, {'sample_id': 'U1', 'rank': 1}]
    kwargs = dict(candidate_ids=('U0', 'U1'), seed=157, round_index=0, budget=1, protocol_hash='a'*64)
    packet = make_packet(rows, **kwargs)
    assert packet['candidate_labels_available'] is False and packet['provider_called'] is False
    for field in ['true_V1_ml', 'true_V2_ml', 'y', 'feedback', 'test_truth']:
        bad = [dict(rows[0], **{field: 999}), rows[1]]
        with pytest.raises(ValueError, match='only approved label-free'):
            make_packet(bad, **kwargs)
    response = {'packet_sha256': packet['packet_sha256'], 'selected_ids': ['U0']}
    for field, value in [('seed', 887), ('round_index', 1)]:
        modified = dict(kwargs); modified[field] = value
        with pytest.raises(ValueError, match='binding mismatch'):
            validate_bound_plan(response, make_packet(rows, **modified))
    registry = json.loads((Path(__file__).resolve().parents[2]/'configs/method_registry.json').read_text())
    assert registry['provider_execution_supported'] is False
    assert all(profile['new_runner_adapter'] is False and profile['status'] == 'HISTORICAL_FROZEN'
               for profile in registry['archived_agent_protocols'].values())


def test_ivr_three_rounds_preserves_original_reference_order_and_all_scores():
    features = np.random.default_rng(61).normal(size=(10, 5))
    identities = tuple(f'fixed-{i}' for i in range(len(features)))
    labeled_ids = list(identities[:3]); candidate_ids = list(identities[3:])
    source_lookup = {identity: i for i, identity in enumerate(identities)}
    old_ids, new_ids, traces = [], [], []
    old_kernel = legacy.ivr().conditional_batch_ivr
    for round_index in range(3):
        ordered = labeled_ids + candidate_ids
        matrix = features[[source_lookup[identity] for identity in ordered]]
        auth = Authorization(tuple(labeled_ids), tuple(candidate_ids))
        frozen = FeatureSnapshot(authorization=auth, train_ids=auth.train_ids, candidate_ids=auth.candidate_ids,
                                 train_features=matrix[:len(labeled_ids)], candidate_features=matrix[len(labeled_ids):],
                                 config=RepresentationConfig('gradient', 5),
                                 predictor=BaselineProvenance('1'*40, '2'*64, '3'*64, '4'*64, '5'*64))
        old = old_kernel(features, np.array([source_lookup[i] for i in labeled_ids]),
                         np.array([source_lookup[i] for i in candidate_ids]), 1)
        new = select(frozen, acquisition='ivr', budget=1, seed=157,
                     numerical_signals={'reference_ids': identities})
        expected = candidate_ids[old['selected_candidate_positions'][0]]
        assert new.selected_ids == (expected,)
        assert [{k:v for k,v in row.items() if k != 'sample_id'} for row in new.trace] == old['trace']
        assert new.audit['reference_order_mode'] == 'fixed_reference_ids'
        old_ids.append(expected); new_ids.extend(new.selected_ids); traces.extend(new.trace)
        labeled_ids.append(expected); candidate_ids.remove(expected)
    record('IVR_fixed_reference', 'three_rounds', old_ids, new_ids, traces)


@pytest.mark.parametrize('seed', [157, 887, 2357, 6101, 12203])
def test_five_frozen_trained_checkpoints_raw_cw_and_latent_on_label_free_u_features(seed):
    """B migration only: 8 known L0 + 8 U inputs; no validation/test targets.

    This is an exact numerical regression of fixed source/model inputs, not a
    replay of an entire historical AL round. True K3 ensemble is not available;
    the Hybrid signal here explicitly repeats one model to exercise tied ranks.
    """
    import csv
    import torch
    import pandas as pd
    from src.qgeognn_al.predictor.model import load_predictor_checkpoint
    from src.qgeognn_al.predictor.data import build_model_data
    from src.qgeognn_al.predictor.training import loader_pair
    from src.qgeognn_al.al.representations.gradient import extract_q50_gradient_sketches, extract_linear_output_gradient_sketches
    from src.qgeognn_al.al.representations.center_width import center_width_transform
    from src.qgeognn_al.al.representations.latent import GraphRow, extract_latent, predictor_state_hash
    path = os.getenv('QGEOGNN_READONLY_REGRESSION_MANIFEST')
    if not path: pytest.skip('BLOCKED: explicit authorized readonly real-checkpoint snapshot manifest required')
    manifest = json.loads(Path(path).read_text())
    item = next(row for row in manifest['seeds'] if row['seed'] == seed)
    feature_file = Path(item['label_free_u0_features']); authorized_file = Path(item['authorized_rows'])
    checkpoint_file = Path(item['checkpoint']['snapshot'])
    for file, digest in [(feature_file, item['label_free_u0_features_sha256']),
                         (authorized_file, item['authorized_rows_sha256']),
                         (checkpoint_file, item['checkpoint']['sha256'])]:
        assert file.stat().st_mode & 0o222 == 0
        assert hashlib.sha256(file.read_bytes()).hexdigest() == digest
    feature_columns = ['sample_id', 'canonical_smiles', 'Density g/ml', 'V/ul', 'loading solvent',
                       'Volume of loading solvent/ul', 'PE/EA', 'canonical_index', 'role']
    known = pd.read_csv(authorized_file, usecols=feature_columns)
    known = known.loc[known.role.eq('l0')].sort_values('canonical_index')
    pool = pd.read_csv(feature_file)
    assert not {'V1_ml', 'V2_ml', 'V1_true', 'V2_true'} & set(pool.columns)
    assert set(pool.role) == {'u0'}
    chosen = pd.concat([known.iloc[:8], pool.iloc[:8]], ignore_index=True)
    chosen['V1_ml'] = 0.; chosen['V2_ml'] = 0.  # Synthetic graph sentinels, never candidate truth.
    initial_targets = []
    with authorized_file.open() as stream:
        for row in csv.DictReader(stream):
            if row['role'] == 'l0':
                initial_targets.append([float(row['V1_ml']), float(row['V2_ml'])])
    assert len(initial_targets) == 333
    cached = next(row for row in manifest['source_files'] if row.get('snapshot', '').endswith('/graph_cache_4g.pt'))
    graph_file = Path(cached['snapshot'])
    assert hashlib.sha256(graph_file.read_bytes()).hexdigest() == cached['sha256']
    cache = torch.load(graph_file, map_location='cpu', weights_only=False)
    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
    atom, angle = build_model_data(chosen, cache, None, checkpoint['preprocessing']['scaler'])
    for graph in atom:
        del graph.y
    torch.set_num_threads(1)
    model = load_predictor_checkpoint(checkpoint_file); state_before = predictor_state_hash(model)
    indices = list(range(16)); scales = tuple(checkpoint['preprocessing']['target_scales'][target] for target in ['V1', 'V2'])
    old = legacy.gradient()
    sketch_seed = seed + 4_000_037
    old_raw = old.extract_q50_gradient_sketches(model, atom, angle, indices, scales, dimension=512, sketch_seed=sketch_seed)
    new_raw = extract_q50_gradient_sketches(model, atom, angle, indices, scales, dimension=512, sketch_seed=sketch_seed)
    np.testing.assert_array_equal(new_raw.features, old_raw.features)
    assert new_raw.parameter_audit == old_raw.parameter_audit
    transform, _ = center_width_transform(np.array(initial_targets))
    old_cw = old.extract_linear_output_gradient_sketches(model, atom, angle, indices, transform, dimension=512, sketch_seed=sketch_seed)
    new_cw = extract_linear_output_gradient_sketches(model, atom, angle, indices, transform, dimension=512, sketch_seed=sketch_seed)
    np.testing.assert_array_equal(new_cw.features, old_cw.features)
    ids = tuple(chosen.sample_id.astype(str)); auth = Authorization(ids[:8], ids[8:])
    canonical_sha = next(row['sha256'] for row in manifest['source_files'] if row['source'].endswith('canonical_4g.csv'))
    provenance = BaselineProvenance(manifest['scientific_base_commit'], canonical_sha, item['partition_sha256'],
                                    item['checkpoint']['sha256'], item['partition_sha256'], 'authorized_snapshot')
    rows = [GraphRow(identity, a, b) for identity, a, b in zip(ids, atom, angle)]
    latent = extract_latent(model, expected_state_hash=state_before, authorization=auth, rows=rows,
                            provenance=provenance, batch_size=16)
    frozen_latent = legacy.load('src/qgeognn_al/active_learning_v2/coverage.py', ['extract_representations'],
                                injected={'torch': torch, 'loader_pair': old.loader_pair})
    # loader_pair is supplied to the frozen gradient module; it is not a migrated dependency.
    before_h = frozen_latent.extract_representations(model, atom, angle, indices, batch_size=16)
    np.testing.assert_array_equal(np.vstack([latent.train, latent.candidates]), before_h)
    for name, values in [('raw', new_raw.features), ('CW', new_cw.features)]:
        features = FeatureSnapshot(authorization=auth, train_ids=auth.train_ids, candidate_ids=auth.candidate_ids,
                                   train_features=values[:8], candidate_features=values[8:],
                                   config=RepresentationConfig('gradient' if name == 'raw' else 'center_width', 512), predictor=provenance)
        for method in ['lcmd', 'maxdet', 'ivr']:
            before_x = old_raw.features if name == 'raw' else old_cw.features
            if method == 'lcmd':
                previous = legacy.lcmd().lcmd_tp_select(before_x[8:], before_x[:8], 3)
                positions = previous.selected_pool_positions.tolist(); trace = list(previous.trace)
            elif method == 'maxdet':
                previous = legacy.maxdet().conditional_gradient_maxdet(before_x, np.arange(8), np.arange(8, 16), 3)
                positions = previous.selected_candidate_positions.tolist(); trace = list(previous.trace)
            else:
                previous = legacy.ivr().conditional_batch_ivr(before_x, np.arange(8), np.arange(8, 16), 3)
                positions = previous['selected_candidate_positions']; trace = previous['trace']
            result = select(features, acquisition=method, budget=3, seed=seed)
            assert result.selected_ids == tuple(auth.candidate_ids[position] for position in positions)
            assert [{k:v for k,v in row.items() if k != 'sample_id'} for row in result.trace] == trace
    a, b = next(zip(*loader_pair(atom, angle, indices, batch_size=16)))
    with torch.no_grad(): prediction = model(a, b).cpu().numpy()[8:]
    ensemble = np.stack([prediction]*3)
    previous = legacy.hybrid().v2_hybrid_select(before_h, ensemble, 8, scales, 3)
    result = select(latent, acquisition='hybrid', budget=3, seed=seed,
                    numerical_signals={'candidate_ids': auth.candidate_ids, 'ensemble_predictions': ensemble, 'target_scales': scales})
    assert result.selected_ids == tuple(auth.candidate_ids[position] for position in previous['selected_positions'])
    assert predictor_state_hash(model) == state_before
    assert all(parameter.grad is None for parameter in model.parameters())
    record('frozen_trained_representation', f'seed={seed}/L333/subsample_8L0_8U',
           [old_raw.audit['sketch_mapping_sha256']], [new_raw.audit['sketch_mapping_sha256']],
           {'parameter_count': sum(p.numel() for p in model.parameters()), 'raw_512_exact': True,
            'CW_512_exact': True, 'latent_128_exact': True, 'U_truth_converted': 0,
            'checkpoint_sha256': item['checkpoint']['sha256'], 'hybrid_ensemble': 'synthetic repeats of one frozen model; not a true K3 regression'})


@pytest.mark.parametrize('round_index', [0, 6, 20])
def test_full_frozen_universe_replays_actual_lcmd_and_other_numerical_kernels(round_index):
    """Full 3330-row frozen numeric inputs; historical LCMD IDs exactly replay.

    Historical CSV scores use prespecified rtol=1e-12/atol=1e-8 to accommodate
    serialization/platform BLAS. Old/new kernels in the same process must be
    exactly equal. Counterfactual MaxDet/IVR comparisons are not AL experiments.
    """
    import csv
    path = os.getenv('QGEOGNN_FROZEN_ACQUISITION_MANIFEST')
    if not path: pytest.skip('BLOCKED: explicit readonly frozen acquisition feature manifest required')
    manifest = json.loads(Path(path).read_text()); item = next(row for row in manifest['records'] if row['round'] == round_index)
    root = Path(item['directory'])
    for relative, meta in item['files'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == meta['sha256']
        assert (root/relative).stat().st_mode & 0o222 == 0
    contract = json.loads((root/'acquisition_artifacts/current_gradient_features.npz.contract.json').read_text())['contract']
    assert contract['test_truth_access_count'] == 0 and contract['dimension'] == 512
    with np.load(root/'acquisition_artifacts/current_gradient_features.npz', allow_pickle=False) as archive:
        features = archive['features']; positions = archive['canonical_indices']
    labeled_positions = np.load(root/'labeled_indices.npy', allow_pickle=False)
    pool_positions = np.load(root/'unlabeled_indices.npy', allow_pickle=False)
    l_ids = tuple(np.load(root/'labeled_ids.npy', allow_pickle=False).tolist())
    u_ids = tuple(np.load(root/'unlabeled_ids.npy', allow_pickle=False).tolist())
    np.testing.assert_array_equal(positions, np.r_[labeled_positions, pool_positions])
    assert positions.tolist() == contract['ordered_indices']
    assert hashlib.sha256(canonical_bytes(sorted(l_ids))).hexdigest() == contract['L_t_ids_hash']
    assert hashlib.sha256(canonical_bytes(sorted(u_ids))).hexdigest() == contract['U_t_ids_hash']
    assert len(features) == 3330 and len(l_ids) == item['budget']
    count = len(l_ids); auth = Authorization(l_ids, u_ids)
    snapshot_x = FeatureSnapshot(authorization=auth, train_ids=l_ids, candidate_ids=u_ids,
                                  train_features=features[:count], candidate_features=features[count:],
                                  config=RepresentationConfig('gradient', 512),
                                  predictor=BaselineProvenance(legacy.FROZEN_COMMIT, '2'*64, '3'*64,
                                                              contract['evaluation_checkpoint_sha256'], contract['evaluation_checkpoint_state_hash']))
    old_lcmd = legacy.lcmd().lcmd_tp_select(features[count:], features[:count], 32)
    result = select(snapshot_x, acquisition='lcmd', budget=32, seed=157)
    selected = tuple(u_ids[position] for position in old_lcmd.selected_pool_positions)
    assert result.selected_ids == selected
    with (root/'acquisition_artifacts/selected_next_batch.csv').open() as stream:
        historical = list(csv.DictReader(stream))
    assert [int(row['selection_order']) for row in historical] == list(range(32))
    assert tuple(row['sample_id'] for row in historical) == selected
    assert [int(row['previous_U_t_position']) for row in historical] == old_lcmd.selected_pool_positions.tolist()
    with (root/'acquisition_artifacts/lcmd_trace.csv').open() as stream:
        historical_trace = list(csv.DictReader(stream))
    for before, frozen_row in zip(old_lcmd.trace, historical_trace):
        for key, value in before.items():
            np.testing.assert_allclose(value, float(frozen_row[key]), rtol=1e-12, atol=1e-8)
    assert [{k:v for k,v in row.items() if k != 'sample_id'} for row in result.trace] == list(old_lcmd.trace)
    for method in ['maxdet', 'ivr']:
        # The original IVR outer-training universe is the fixed initial L0+U0 order.
        first = next(row for row in manifest['records'] if row['round'] == 0)
        first_root = Path(first['directory'])
        reference = tuple(np.load(first_root/'labeled_ids.npy', allow_pickle=False).tolist()+
                          np.load(first_root/'unlabeled_ids.npy', allow_pickle=False).tolist())
        if method == 'maxdet':
            old_result = legacy.maxdet().conditional_gradient_maxdet(features, np.arange(count), np.arange(count, len(features)), 32)
            expected_positions = old_result.selected_candidate_positions.tolist(); old_trace = list(old_result.trace)
            new_result = select(snapshot_x, acquisition=method, budget=32, seed=157)
        else:
            lookup = {identity:i for i, identity in enumerate(l_ids+u_ids)}
            fixed_lookup = {identity:i for i, identity in enumerate(reference)}
            fixed = features[[lookup[identity] for identity in reference]]
            old_result = legacy.ivr().conditional_batch_ivr(fixed,
                          np.array([fixed_lookup[identity] for identity in l_ids]),
                          np.array([fixed_lookup[identity] for identity in u_ids]), 32)
            expected_positions = old_result['selected_candidate_positions']; old_trace = old_result['trace']
            new_result = select(snapshot_x, acquisition=method, budget=32, seed=157,
                                 numerical_signals={'reference_ids': reference})
        assert new_result.selected_ids == tuple(u_ids[position] for position in expected_positions)
        assert [{k:v for k,v in row.items() if k != 'sample_id'} for row in new_result.trace] == old_trace
    record('full_frozen_LCMD', f'seed=157/L{item["budget"]}->L{item["budget"]+32}',
           list(selected), list(result.selected_ids), result.trace)
    RECORDS[-1].update(input_features_sha256=item['files']['acquisition_artifacts/current_gradient_features.npz']['sha256'],
                       universe_rows=3330, historical_LCMD_selection_replayed=True,
                       MaxDet_IVR='counterfactual numerical equivalence only; no new actual selection/freeze/reveal',
                       historical_trace_tolerance={'rtol':1e-12, 'atol':1e-8})


@pytest.mark.parametrize('outer_seed', [17, 157, 12203])
def test_random_fixed_trajectory_matches_old_three_round_ids_without_redraw(outer_seed):
    from src.qgeognn_al.al.acquisitions.random import historical_random_trajectory_seed
    path = 'src/qgeognn_al/active_learning_v2/sequential_protocol.py'
    seed_function = legacy.load(path, ['random_trajectory_seed'],
                       injected={'RANDOM_TRAJECTORY_OFFSET': legacy.constant(path, 'RANDOM_TRAJECTORY_OFFSET')})
    old = legacy.load('src/qgeognn_al/active_learning_v2/sequential_acquisition.py',
                       ['deterministic_random_permutation', 'random_round_batch'],
                       injected={'random_trajectory_seed': seed_function.random_trajectory_seed,
                                 'BATCH_SIZE': legacy.constant(path, 'BATCH_SIZE'),
                                 'ACQUISITION_ROUNDS': legacy.constant(path, 'ACQUISITION_ROUNDS')})
    original = tuple(f'fixed-U{i}' for i in range(128)); initial_l = ('initial-L0', 'initial-L1')
    current = list(original); labeled = list(initial_l)
    selected_old, selected_new = [], []
    for round_index in range(3):
        authorization = Authorization(tuple(labeled), tuple(current))
        frozen = FeatureSnapshot(authorization=authorization, train_ids=authorization.train_ids,
                                 candidate_ids=authorization.candidate_ids,
                                 train_features=np.ones((len(labeled), 1)), candidate_features=np.ones((len(current), 1)),
                                 config=RepresentationConfig('gradient', 1),
                                 predictor=BaselineProvenance('1'*40, '2'*64, '3'*64, '4'*64, '5'*64))
        expected = old.random_round_batch(original, current, outer_seed, round_index+1)
        new = select(frozen, acquisition='random', budget=32, seed=historical_random_trajectory_seed(outer_seed),
                     numerical_signals={'profile':'fixed_trajectory', 'original_candidate_ids':original,
                                        'round_index':round_index,'trajectory_batch_size':32})
        assert new.selected_ids == expected
        selected_old.extend(expected); selected_new.extend(new.selected_ids)
        labeled.extend(expected); current = [identity for identity in current if identity not in set(expected)]
    record('random_fixed_trajectory', f'seed={outer_seed}/three_rounds', selected_old, selected_new)


def test_fixed_random_final_partial_batch_uses_frozen_batch_stride_and_rejects_drift():
    from src.qgeognn_al.al.acquisitions.random import fixed_random_trajectory_batch
    original = tuple(f'U{i}' for i in range(40)); seed = 4242
    expected = tuple(original[int(i)] for i in np.random.default_rng(seed).permutation(40))
    first = fixed_random_trajectory_batch(original, original, seed, 0, 32)
    current = tuple(identity for identity in original if identity not in set(first))
    partial = fixed_random_trajectory_batch(original, current, seed, 1, 32, selection_count=3)
    assert first == expected[:32] and partial == expected[32:35]
    auth = Authorization(('L0',), current)
    frozen = FeatureSnapshot(authorization=auth, train_ids=auth.train_ids, candidate_ids=current,
                             train_features=np.ones((1, 1)), candidate_features=np.ones((8, 1)),
                             config=RepresentationConfig('gradient', 1),
                             predictor=BaselineProvenance('1'*40, '2'*64, '3'*64, '4'*64, '5'*64))
    result = select(frozen, acquisition='random', budget=3, seed=seed,
                    numerical_signals={'profile':'fixed_trajectory','original_candidate_ids':original,
                                       'round_index':1,'trajectory_batch_size':32})
    assert result.selected_ids == expected[32:35]
    assert [row['permutation_position'] for row in result.trace] == [32, 33, 34]
    with pytest.raises(ValueError, match='exact remaining'):
        fixed_random_trajectory_batch(original, tuple(original[1:]), seed, 1, 32, selection_count=3)
