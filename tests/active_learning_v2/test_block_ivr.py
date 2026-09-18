import numpy as np
import pytest

from src.qgeognn_al.active_learning_v2.block_ivr import (
    block_batch_ivr, covariance_update, integrated_risk, normalize_blocks, posterior_covariance,
)
from src.qgeognn_al.active_learning_v2.ivr import conditional_batch_ivr


def brute_force(features, labeled, candidates, batch_size, backward):
    x, _ = normalize_blocks(features)
    chosen, trace = [], []
    for _ in range(min(2 * batch_size, len(candidates)) if backward else batch_size):
        current = np.r_[labeled, candidates[chosen]]
        before = integrated_risk(x, posterior_covariance(x, current))
        available = [i for i in range(len(candidates)) if i not in chosen]
        values = [before - integrated_risk(x, posterior_covariance(x, np.r_[current, candidates[i]]))
                  for i in available]
        at = int(np.argmax(values))
        chosen.append(available[at])
        trace.append(values[at])
    while len(chosen) > batch_size:
        before = integrated_risk(x, posterior_covariance(x, np.r_[labeled, candidates[chosen]]))
        available = sorted(chosen)
        values = [integrated_risk(x, posterior_covariance(x, np.r_[labeled, candidates[[j for j in chosen if j != i]]])) - before
                  for i in available]
        at = int(np.argmin(values))
        chosen.remove(available[at])
        trace.append(values[at])
    return chosen, trace


@pytest.mark.parametrize("outputs", [1, 2, 3])
@pytest.mark.parametrize("backward", [False, True])
def test_every_forward_backward_choice_matches_direct_reinversion(outputs, backward):
    x = np.random.default_rng(572).normal(size=(14, outputs, 8))
    l, u = np.arange(4), np.arange(4, 14)
    result = block_batch_ivr(x, l, u, 4, backward=backward)
    chosen, reductions = brute_force(x, l, u, 4, backward)
    assert result['selected_candidate_positions'] == chosen
    np.testing.assert_allclose([t['score'] for t in result['trace']], reductions, rtol=1e-10, atol=1e-13)


def test_rank_two_update_and_downdate_restore_covariance():
    x, _ = normalize_blocks(np.random.default_rng(14).normal(size=(9, 2, 6)))
    c = posterior_covariance(x, np.arange(3))
    added = covariance_update(c, x[5])
    np.testing.assert_allclose(added, posterior_covariance(x, [0, 1, 2, 5]), atol=1e-13)
    np.testing.assert_allclose(covariance_update(added, x[5], remove=True), c, atol=1e-13)


def test_scalar_path_matches_historical_selector_and_scale():
    x = np.random.default_rng(13).normal(size=(25, 9))
    l, u = np.arange(5), np.arange(5, 25)
    old = conditional_batch_ivr(x, l, u, 8)
    new = block_batch_ivr(x, l, u, 8)
    assert old['selected_candidate_positions'] == new['selected_candidate_positions']
    np.testing.assert_allclose([t['score'] for t in old['trace']], [t['score'] for t in new['trace']], atol=1e-13)


def test_endpoint_rotation_global_scale_and_endpoint_risk_accounting():
    x = np.random.default_rng(17).normal(size=(17, 2, 8))
    l, u = np.arange(4), np.arange(4, 17)
    rotation = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
    first = block_batch_ivr(x, l, u, 5)
    second = block_batch_ivr(np.einsum('ef,nfd->ned', rotation, x) * 137, l, u, 5)
    assert first['selected_candidate_positions'] == second['selected_candidate_positions']
    np.testing.assert_allclose(first['initial_scores'], second['initial_scores'], atol=1e-13)
    assert np.isclose(sum(first['audit']['endpoint_final_risk']), first['audit']['final_risk'])


def test_block_observations_keep_two_directions_and_duplicate_ties_are_stable():
    x = np.ones((9, 2, 5))
    result = block_batch_ivr(x, np.arange(2), np.arange(2, 9), 5)
    assert result['selected_candidate_positions'] == [0, 1, 2, 3, 4]
    x = np.zeros((4, 2, 2))
    x[1] = np.eye(2)
    x[2:, 0, 0] = 1
    result = block_batch_ivr(x, np.array([0]), np.array([1, 2, 3]), 1)
    assert result['selected_candidate_positions'] == [0]


@pytest.mark.parametrize('l,u', [([0], [0, 1, 2]), ([0], [1]), ([0], [1, 1, 2]), ([0], [1, 3])])
def test_invalid_partitions_rejected(l, u):
    with pytest.raises(ValueError):
        block_batch_ivr(np.ones((3, 2, 4)), np.array(l), np.array(u), 1)


def test_invalid_downdate_is_rejected():
    with pytest.raises(np.linalg.LinAlgError):
        covariance_update(np.eye(2), 2 * np.eye(2), remove=True)
