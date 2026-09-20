import inspect
import math

import numpy as np
import pytest

from src.qgeognn_al.active_learning_v2.maxdet import (
    conditional_gradient_maxdet,
    ensemble_top50_gate,
)


def _direct_greedy(features, labeled, candidates, batch_size):
    x = np.asarray(features, dtype=np.float64)
    scale = np.sqrt(np.square(x[labeled]).sum(axis=1).mean())
    psi = x / scale
    information = np.eye(x.shape[1]) + psi[labeled].T @ psi[labeled]
    available = list(candidates)
    selected = []
    for _ in range(batch_size):
        sign, base = np.linalg.slogdet(information)
        assert sign > 0
        gains = []
        for position in available:
            updated = information + np.outer(psi[position], psi[position])
            next_sign, next_value = np.linalg.slogdet(updated)
            assert next_sign > 0
            gains.append(next_value - base)
        winner = int(np.argmax(gains))
        position = available.pop(winner)
        selected.append(position)
        information += np.outer(psi[position], psi[position])
    return selected


def test_greedy_gains_and_order_match_direct_logdet_recomputation():
    rng = np.random.default_rng(17)
    features = rng.normal(size=(11, 5))
    labeled = np.array([0, 1, 2])
    candidates = np.array([9, 4, 7, 3, 5, 10])
    result = conditional_gradient_maxdet(features, labeled, candidates, 4)
    expected = _direct_greedy(features, labeled, candidates.tolist(), 4)
    assert result.selected_positions.tolist() == expected
    assert [row["marginal_logdet_gain"] for row in result.trace] == pytest.approx(
        [
            np.linalg.slogdet(
                np.eye(5)
                + (features[labeled] / result.normalization_scale).T
                @ (features[labeled] / result.normalization_scale)
                + sum(
                    np.outer(features[p] / result.normalization_scale, features[p] / result.normalization_scale)
                    for p in expected[: i + 1]
                )
            )[1]
            - np.linalg.slogdet(
                np.eye(5)
                + (features[labeled] / result.normalization_scale).T
                @ (features[labeled] / result.normalization_scale)
                + sum(
                    (
                        np.outer(features[p] / result.normalization_scale, features[p] / result.normalization_scale)
                        for p in expected[:i]
                    ),
                    start=np.zeros((5, 5)),
                )
            )[1]
            for i in range(4)
        ],
        abs=1e-11,
    )


def test_duplicate_and_collinear_rows_are_finite_and_stably_tied():
    features = np.array([[1.0, 0.0], [0.0, 1.0], [3.0, 0.0], [3.0, 0.0], [6.0, 0.0], [0.0, 2.0]])
    result = conditional_gradient_maxdet(features, [0, 1], [2, 3, 4, 5], 4)
    assert result.selected_positions[0] == 4
    assert result.selected_positions.tolist().index(2) < result.selected_positions.tolist().index(3)
    assert np.isfinite(result.initial_marginal_gains).all()
    assert all(np.isfinite(row["marginal_logdet_gain"]) for row in result.trace)


def test_labeled_conditioning_changes_selection_and_labeled_rows_are_excluded():
    features = np.array([[10.0, 0.0], [0.0, 1.0], [8.0, 0.0], [0.0, 8.0]])
    first = conditional_gradient_maxdet(features, [0, 1], [2, 3], 1)
    second = conditional_gradient_maxdet(features, [1], [0, 2, 3], 1)
    assert first.selected_positions.tolist() == [3]
    assert second.selected_positions.tolist() == [0]
    assert not set(first.selected_positions) & {0, 1}


def test_candidate_restriction_batch_size_32_and_repeat_determinism():
    rng = np.random.default_rng(29)
    features = rng.normal(size=(55, 9))
    labeled = np.arange(8)
    candidates = np.arange(12, 55)
    first = conditional_gradient_maxdet(features, labeled, candidates, 32)
    resumed = conditional_gradient_maxdet(features.copy(), labeled.copy(), candidates.copy(), 32)
    assert len(first.selected_positions) == len(set(first.selected_positions.tolist())) == 32
    assert set(first.selected_positions) <= set(candidates)
    assert np.array_equal(first.selected_positions, resumed.selected_positions)
    assert first.trace == resumed.trace


def test_invalid_or_nonfinite_inputs_fail_closed():
    x = np.eye(4)
    with pytest.raises(ValueError):
        conditional_gradient_maxdet(x, [0], [0, 1], 1)
    with pytest.raises(ValueError):
        conditional_gradient_maxdet(np.zeros((4, 2)), [0], [1, 2], 1)
    x[2, 1] = np.nan
    with pytest.raises(ValueError):
        conditional_gradient_maxdet(x, [0], [1, 2], 1)


def test_top50_gate_exact_membership_and_canonical_ties():
    predictions = np.zeros((3, 7, 6), dtype=float)
    # ddof=0 disagreement grows with this amplitude; positions 2/3 tie.
    amplitudes = np.array([0.0, 1.0, 4.0, 4.0, 3.0, 2.0, 0.5])
    predictions[0, :, 1] = amplitudes
    predictions[2, :, 1] = -amplitudes
    gate = ensemble_top50_gate(predictions, [1.0, 1.0])
    assert len(gate.candidate_positions) == math.ceil(0.5 * 7)
    assert gate.candidate_positions.tolist() == [2, 3, 4, 5]
    assert gate.cutoff_score == pytest.approx(gate.uncertainty_scores[5])


def test_selector_surface_has_no_label_or_truth_argument():
    parameters = set(inspect.signature(conditional_gradient_maxdet).parameters)
    assert not parameters & {"labels", "targets", "truth", "test", "test_truth", "pool_truth"}
