from __future__ import annotations

import numpy as np

from src.qgeognn_al.active_learning_v2.lcmd import lcmd_tp_select


def test_lcmd_selects_farthest_point_from_largest_cluster() -> None:
    train = np.array([[0.0], [10.0]])
    pool = np.array([[1.0], [3.0], [9.0]])
    result = lcmd_tp_select(pool, train, batch_size=1)
    assert result.selected_pool_positions.tolist() == [1]
    assert result.trace[0]["selected_from_center"] == 0
    assert result.trace[0]["nearest_center_sq_distance_before_selection"] == 9.0


def test_tp_semantics_install_l0_before_first_pool_point() -> None:
    train = np.array([[0.0], [10.0]])
    pool = np.array([[1.0], [6.0], [9.0]])
    result = lcmd_tp_select(pool, train, batch_size=1)
    # Correct TP mode selects 6 from the largest training-centered cluster;
    # the obsolete empty-set first-point rule would select largest-norm 9.
    assert result.selected_pool_positions.tolist() == [1]
    assert result.initial_nearest_center.tolist() == [0, 1, 1]


def test_lcmd_selection_is_unique_pool_only_and_deterministic() -> None:
    rng = np.random.default_rng(12)
    train = rng.normal(size=(7, 5))
    pool = rng.normal(size=(40, 5))
    first = lcmd_tp_select(pool, train, batch_size=13)
    second = lcmd_tp_select(pool.copy(), train.copy(), batch_size=13)
    assert np.array_equal(first.selected_pool_positions, second.selected_pool_positions)
    assert len(np.unique(first.selected_pool_positions)) == 13
    assert np.all((first.selected_pool_positions >= 0) & (first.selected_pool_positions < len(pool)))
