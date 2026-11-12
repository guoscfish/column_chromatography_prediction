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
from tests.support import historical_reference as legacy
from tests.support.fixtures import inputs, snapshot, record


@pytest.mark.historical_archive
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



@pytest.mark.historical_archive
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



@pytest.mark.historical_archive
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



@pytest.mark.historical_archive
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



@pytest.mark.historical_archive
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



@pytest.mark.historical_archive
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



@pytest.mark.historical_archive
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



@pytest.mark.historical_archive
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




def test_lcmd_hand_computed_cluster_sum_and_stable_ties():
    result=lcmd_tp_select(np.array([[2.],[3.],[7.],[8.]]),np.array([[0.],[10.]]),3)
    np.testing.assert_array_equal(result.initial_nearest_center,[0,0,1,1])
    np.testing.assert_array_equal(result.initial_nearest_sq_distance,[4.,9.,9.,4.])
    np.testing.assert_array_equal(result.selected_pool_positions,[1,2,0])
    assert [r['largest_cluster_score'] for r in result.trace]==[13.,13.,1.]
    assert [r['selected_from_center'] for r in result.trace]==[0,1,2]


def test_random_effective_seed_and_global_rng_are_independent():
    from src.qgeognn_al.al.acquisitions.random import random_select
    np.random.seed(99);expected=np.random.random(5);np.random.seed(99)
    selected=random_select(12,4,17)
    np.testing.assert_array_equal(selected,[6,1,11,8])
    np.testing.assert_array_equal(np.random.random(5),expected)
    assert random_control_seed(157,2)==157*1_000_003+50_002
    for args in [(3,4,17),(3,0,17),(3,2,-1),(3,2,True)]:
        with pytest.raises(ValueError):random_select(*args)


@pytest.mark.parametrize('tied',[False,True])
def test_maxdet_matches_direct_log_determinant_oracle(tied):
    matrix=np.array([[1.,0.],[0.,2.],[2.,1.],[2.,1.] if tied else [1.,3.],[-1.,2.]])
    labeled=np.array([0,1]);candidates=np.array([3,2,4])
    result=conditional_gradient_maxdet(matrix,labeled,candidates,3)
    scale=np.sqrt(np.square(matrix[labeled]).sum(axis=1).mean());x=matrix/scale
    information=np.eye(2)+x[labeled].T@x[labeled]
    available=list(candidates);chosen=[];gains=[]
    for _ in range(3):
        base=np.linalg.slogdet(information)[1]
        values=[np.linalg.slogdet(information+np.outer(x[i],x[i]))[1]-base for i in available]
        winner=int(np.argmax(values));position=available.pop(winner);chosen.append(position);gains.append(values[winner])
        information=information+np.outer(x[position],x[position])
    assert result.selected_positions.tolist()==chosen
    np.testing.assert_allclose([r['marginal_logdet_gain'] for r in result.trace],gains,rtol=1e-13,atol=1e-14)
    assert result.normalization_scale==scale
    if tied:assert chosen[0]==3  # equal rows honor supplied candidate order


@pytest.mark.parametrize('candidate_order',[[2,3,4],[4,3,2]])
def test_ivr_matches_independent_row_covariance_oracle(candidate_order):
    matrix=np.array([[1.,0.],[0.,2.],[2.,1.],[2.,1.],[-1.,2.]])
    labeled=np.array([0,1]);candidates=np.array(candidate_order)
    result=conditional_batch_ivr(matrix,labeled,candidates,3)
    x=matrix/np.sqrt(np.square(matrix).sum(axis=1).mean());kernel=x@x.T
    covariance=kernel-kernel[:,labeled]@np.linalg.solve(np.eye(2)+kernel[np.ix_(labeled,labeled)],kernel[labeled])
    remaining=list(candidates);chosen=[];reductions=[]
    for _ in range(3):
        scores=np.array([np.square(covariance[:,i]).mean()/(1+covariance[i,i]) for i in remaining])
        winner=int(np.argmax(scores));position=remaining.pop(winner);chosen.append(position);reductions.append(scores[winner])
        direction=covariance[:,position].copy();covariance-=np.outer(direction,direction)/(1+covariance[position,position])
    assert [candidate_order[i] for i in result['selected_candidate_positions']]==chosen
    np.testing.assert_allclose([r['score'] for r in result['trace']],reductions,rtol=1e-12,atol=1e-14)
    assert all(r['integrated_variance_before']>=r['integrated_variance_after'] for r in result['trace'])
    with pytest.raises(ValueError,match='partition'):
        conditional_batch_ivr(matrix,np.array([0]),np.array([2,3,4]),1)


@pytest.mark.parametrize('tied',[False,True])
def test_hybrid_shortlist_and_latent_tie_semantics(tied):
    matrix=np.zeros((9,128));matrix[1:,0]=[1.,10.,3.,4.,5.,6.,7.,8.]
    predictions=np.zeros((3,8,6));amplitudes=np.ones(8) if tied else np.array([4.,4.,3.,2.,1.,0.,0.,0.])
    predictions[0,:,1]=-amplitudes;predictions[2,:,1]=amplitudes
    result=hybrid_select(matrix,predictions,1,(2.,3.),2)
    assert result['shortlist_size']==2
    assert result['shortlist_positions'].tolist()==[0,1]
    assert result['selected_positions'].tolist()==[1,0]
    with pytest.raises(ValueError,match='K=3'):
        hybrid_select(matrix,predictions[:2],1,(2.,3.),2)
