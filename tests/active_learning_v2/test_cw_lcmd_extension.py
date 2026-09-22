from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.qgeognn_al.active_learning_v2.cw_lcmd_extension_runner import select_current_cw_lcmd
from src.qgeognn_al.active_learning_v2.cw_lcmd_extension_study import ACTIVE_LABEL_BUDGETS, FINAL_ACTIVE_LABELS, METHOD, NEW_ACQUISITION_ROUNDS
from src.qgeognn_al.active_learning_v2.cw_lcmd_extension_reporting import (
    _assert_frozen_artifacts_unchanged,
    _aulc,
    _decision,
    _frozen_artifact_hashes,
    _per_seed_figure_name,
    _validate_budget_grid,
)
def test_extension_scope_and_hard_stop():
    assert METHOD == "center_width_lcmd" and NEW_ACQUISITION_ROUNDS == (4, 5, 6)
    assert ACTIVE_LABEL_BUDGETS[-1] == FINAL_ACTIVE_LABELS == 525 and 557 not in ACTIVE_LABEL_BUDGETS
def test_current_lcmd_uses_all_current_centers(monkeypatch):
    seen = {}
    class R: selected_pool_positions = np.arange(32); trace = []
    def fake(pool, centers, batch): seen.update(pool=len(pool), centers=len(centers), batch=batch); return R()
    monkeypatch.setattr("src.qgeognn_al.active_learning_v2.cw_lcmd_extension_runner.lcmd_tp_select", fake)
    result = select_current_cw_lcmd(np.zeros((429 + 100, 512)), 429)
    assert seen == {"pool": 100, "centers": 429, "batch": 32}; assert len(result["selected_pool_positions"]) == 32
def test_selection_rejects_invalid_shape():
    with pytest.raises(ValueError): select_current_cw_lcmd(np.zeros((429, 511)), 429)


def _curve(budgets=ACTIVE_LABEL_BUDGETS):
    return pd.DataFrame({
        "outer_seed": [157] * len(budgets),
        "method": [METHOD] * len(budgets),
        "active_label_count": list(budgets),
        "combined_normalized_RMSE": np.linspace(0.8, 0.5, len(budgets)),
    })


def test_full_aulc_requires_all_seven_budget_points():
    curve = _curve()
    result = _aulc(curve, 333, 525, "AULC_333_525", ACTIVE_LABEL_BUDGETS)
    assert len(ACTIVE_LABEL_BUDGETS) == 7
    assert result.loc[0, "AULC_333_525"] == pytest.approx(0.65)


@pytest.mark.parametrize("missing", [333, 365, 397, 429, 461, 493, 525])
def test_full_aulc_fails_loudly_for_any_missing_budget(missing):
    curve = _curve(tuple(budget for budget in ACTIVE_LABEL_BUDGETS if budget != missing))
    with pytest.raises(RuntimeError, match="incomplete AULC endpoints"):
        _aulc(curve, 333, 525, "AULC_333_525", ACTIVE_LABEL_BUDGETS)


def test_budget_grid_requires_every_seed_method_pair():
    with pytest.raises(RuntimeError, match="seed=6101"):
        _validate_budget_grid(_curve(), ACTIVE_LABEL_BUDGETS, (157, 6101), (METHOD,))


def _decision_inputs(endpoint_delta):
    mid_rows = []
    curve_rows = []
    for seed in (157, 6101):
        for method, area, endpoint in (
            (METHOD, 0.60, 0.60 + endpoint_delta),
            ("hybrid", 0.65, 0.60),
            ("gradient_maxdet", 0.70, 0.70),
        ):
            mid_rows.append({"outer_seed": seed, "method": method, "AULC_429_525": area})
            curve_rows.append({"outer_seed": seed, "method": method, "active_label_count": 525, "combined_normalized_RMSE": endpoint})
    return pd.DataFrame(curve_rows), pd.DataFrame(mid_rows)


def test_negative_endpoint_delta_is_not_deterioration():
    curves, mid = _decision_inputs(-0.08)
    decision = _decision(curves, mid)
    assert decision["decision_criteria"]["endpoint_deteriorated"] is False
    assert decision["decision"] == "STRONG_MIDSTAGE_SIGNAL"


def test_positive_endpoint_delta_above_tolerance_is_deterioration():
    curves, mid = _decision_inputs(0.05)
    decision = _decision(curves, mid)
    assert decision["decision_criteria"]["endpoint_deteriorated"] is True
    assert decision["decision"] == "MIXED"


def test_seed_6101_figure_filename_is_not_truncated():
    assert _per_seed_figure_name(157) == "per_seed_cw_continuation_157.png"
    assert _per_seed_figure_name(6101) == "per_seed_cw_continuation_6101.png"


def test_reporting_only_outputs_do_not_change_frozen_hashes(tmp_path):
    model = tmp_path / "runtime/seed_157/center_width_lcmd/round_04/model"
    acquisition = model.parent / "acquisition_artifacts"
    model.mkdir(parents=True)
    acquisition.mkdir(parents=True)
    (model / "best.pt").write_bytes(b"checkpoint")
    (model / "predictions.csv.gz").write_bytes(b"predictions")
    (acquisition / "selected_next_batch.csv").write_text("sample_id\na\n")
    before = _frozen_artifact_hashes(tmp_path)
    (tmp_path / "FINAL_REPORT.md").write_text("derived report")
    _assert_frozen_artifacts_unchanged(tmp_path, before)
