from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.qgeognn_al.active_learning_v2.gradient_transforms import center_width_transform, power_normalize_gradients
from src.qgeognn_al.active_learning_v2.short_sequential_reporting import partial_aulc
from src.qgeognn_al.active_learning_v2.short_sequential_reporting import _decisions
from src.qgeognn_al.active_learning_v2.short_sequential_runner import select_transformed_lcmd
from src.qgeognn_al.active_learning_v2.short_sequential_study import (
    ACQUISITION_ROUNDS,
    ACTIVE_LABEL_BUDGETS,
    BATCH_SIZE,
    FINAL_ACTIVE_LABELS,
    METHODS,
    SEEDS,
    protocol_record,
)


def test_scope_is_only_cw_and_direction_and_hard_stops_at_429():
    assert METHODS == ("center_width_lcmd", "direction_lcmd")
    assert SEEDS == (157, 6101)
    assert ACQUISITION_ROUNDS == 3
    assert ACTIVE_LABEL_BUDGETS == (333, 365, 397, 429)
    assert FINAL_ACTIVE_LABELS == 429
    protocol = protocol_record()
    assert protocol["explicitly_excluded_method"] == "ensemble_uncertainty"
    assert 461 not in protocol["active_label_budgets"]


def test_center_width_definition_is_exact_l0_population_scaling():
    targets = np.array([[1.0, 3.0], [2.0, 8.0], [6.0, 9.0], [8.0, 15.0]])
    transform, audit = center_width_transform(targets)
    s_center = np.std((targets[:, 0] + targets[:, 1]) / 2, ddof=0)
    s_width = np.std(targets[:, 1] - targets[:, 0], ddof=0)
    np.testing.assert_allclose(transform, [[.5 / s_center, .5 / s_center], [-1 / s_width, 1 / s_width]])
    assert audit["target_source"] == "L0_labels_only" and audit["scale_ddof"] == 0


def test_direction_is_unit_norm_and_retains_zero_rows():
    raw = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [1.0, 2.0, 2.0]])
    result = power_normalize_gradients(raw, 0.0)
    np.testing.assert_array_equal(result.features[0], np.zeros(3))
    np.testing.assert_allclose(np.linalg.norm(result.features[1:], axis=1), 1.0)
    assert result.audit["zero_norm_policy"] == "retain_zero_vector"


def test_selector_uses_every_current_labeled_row_as_center(monkeypatch):
    seen = {}

    class Selection:
        selected_pool_positions = np.arange(BATCH_SIZE)
        trace = []

    def fake_lcmd(pool, centers, batch_size):
        seen.update(pool_rows=len(pool), center_rows=len(centers), batch_size=batch_size)
        return Selection()

    monkeypatch.setattr("src.qgeognn_al.active_learning_v2.short_sequential_runner.lcmd_tp_select", fake_lcmd)
    result = select_transformed_lcmd(np.zeros((397 + 100, 512)), 397)
    assert seen == {"pool_rows": 100, "center_rows": 397, "batch_size": 32}
    assert result["center_count"] == 397


def test_selector_returns_exact_unique_current_u_positions():
    rng = np.random.default_rng(42)
    result = select_transformed_lcmd(rng.normal(size=(365 + 120, 512)), 365)
    selected = result["selected_pool_positions"]
    assert len(selected) == len(np.unique(selected)) == BATCH_SIZE
    assert np.all((selected >= 0) & (selected < 120))


def test_acquisition_surface_cannot_receive_truth_or_test_labels():
    parameters = set(inspect.signature(select_transformed_lcmd).parameters)
    assert not parameters & {"labels", "truth", "targets", "test", "test_truth", "u_truth"}


def test_partial_aulc_uses_all_four_registered_points():
    values = np.array([1.0, 0.8, 0.7, 0.6])
    expected = np.trapezoid(values, np.array(ACTIVE_LABEL_BUDGETS)) / 96
    assert partial_aulc(np.array(ACTIVE_LABEL_BUDGETS), values) == pytest.approx(expected)
    with pytest.raises(ValueError):
        partial_aulc(np.array(ACTIVE_LABEL_BUDGETS[:-1]), values[:-1])


def test_stop_precedes_improvement_against_weaker_baseline():
    import pandas as pd

    rows = []
    areas = []
    for seed in SEEDS:
        for method, area, endpoint in (
            ("gradient_maxdet", .70, .60),
            ("lcmd", .80, .75),
            ("center_width_lcmd", .75, .65),
            ("direction_lcmd", .85, .80),
        ):
            areas.append({"outer_seed": seed, "method": method, "AULC_333_429": area})
            rows.append({"outer_seed": seed, "method": method,
                         "active_label_count": 429, "combined_normalized_RMSE": endpoint})
    decisions = _decisions(pd.DataFrame(rows), pd.DataFrame(areas))
    assert decisions["methods"]["center_width_lcmd"]["condition_B"] is True
    assert decisions["methods"]["center_width_lcmd"]["stop_condition"] is True
    assert decisions["methods"]["center_width_lcmd"]["decision"] == "STOP"
