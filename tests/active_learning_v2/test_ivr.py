import numpy as np
import pytest

from src.qgeognn_al.active_learning_v2.ivr import conditional_batch_ivr


def direct_greedy(features, labeled, candidates, batch_size):
    x = features / np.sqrt(np.mean(np.sum(features ** 2, axis=1)))
    current = list(labeled)
    available = list(candidates)
    chosen, reductions = [], []
    for _ in range(batch_size):
        c = np.linalg.inv(np.eye(x.shape[1]) + x[current].T @ x[current])
        before = np.mean(np.sum((x @ c) * x, axis=1))
        scores = []
        for position in available:
            after_l = current + [position]
            after_c = np.linalg.inv(np.eye(x.shape[1]) + x[after_l].T @ x[after_l])
            after = np.mean(np.sum((x @ after_c) * x, axis=1))
            scores.append(before - after)
        winner = int(np.argmax(scores))
        reductions.append(scores[winner])
        current.append(available.pop(winner))
        chosen.append(current[-1])
    return chosen, reductions


def test_batch_matches_direct_reinversion_and_integrated_variance_reduction():
    x = np.random.default_rng(41).normal(size=(19, 7))
    l, u = np.arange(4), np.arange(4, 19)
    result = conditional_batch_ivr(x, l, u, 6)
    positions, reductions = direct_greedy(x, l, u, 6)
    assert u[result["selected_candidate_positions"]].tolist() == positions
    np.testing.assert_allclose([row["score"] for row in result["trace"]], reductions, rtol=1e-10)
    assert all(row["integrated_variance_after"] < row["integrated_variance_before"] for row in result["trace"])


def test_global_scale_invariance():
    x = np.random.default_rng(8).normal(size=(25, 9))
    original = conditional_batch_ivr(x, np.arange(5), np.arange(5, 25), 8)
    scaled = conditional_batch_ivr(x * 137, np.arange(5), np.arange(5, 25), 8)
    assert original["selected_candidate_positions"] == scaled["selected_candidate_positions"]
    np.testing.assert_allclose([r["score"] for r in original["trace"]],
                               [r["score"] for r in scaled["trace"]], rtol=1e-10)


def test_batch_conditions_on_pending_picks_instead_of_top_k():
    x = np.array([[0., 0.], [4., 0.], [4., 0.], [0., 4.]])
    result = conditional_batch_ivr(x, np.array([0]), np.array([1, 2, 3]), 2)
    assert result["selected_candidate_positions"] == [0, 2]


def test_existing_labels_reduce_redundant_direction():
    x = np.array([[10., 0.], [10., 0.], [0., 10.]])
    result = conditional_batch_ivr(x, np.array([0]), np.array([1, 2]), 1)
    assert result["selected_candidate_positions"] == [1]


def test_exact_duplicates_have_stable_ties_and_no_duplicate_selection():
    x = np.ones((7, 3))
    result = conditional_batch_ivr(x, np.array([0]), np.arange(1, 7), 6)
    assert result["selected_candidate_positions"] == list(range(6))


@pytest.mark.parametrize("l,u", [([0], [0, 1, 2]), ([0], [1]), ([0], [1, 1, 2]), ([0], [1, 3])])
def test_rejects_invalid_or_shrinking_reference_partition(l, u):
    with pytest.raises(ValueError):
        conditional_batch_ivr(np.ones((3, 2)), np.array(l), np.array(u), 1)


@pytest.mark.parametrize("x", [np.zeros((3, 2)), np.full((3, 2), np.nan)])
def test_rejects_nonfinite_or_uninformative_features(x):
    with pytest.raises(ValueError):
        conditional_batch_ivr(x, np.array([0]), np.array([1, 2]), 1)
