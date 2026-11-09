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
