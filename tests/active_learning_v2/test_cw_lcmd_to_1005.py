from __future__ import annotations

import inspect
import numpy as np
import pandas as pd
import pytest

from src.qgeognn_al.active_learning_v2 import cw_lcmd_to_1005_reporting as reporting
from src.qgeognn_al.active_learning_v2 import cw_lcmd_to_1005_runner as runner
from src.qgeognn_al.active_learning_v2 import cw_lcmd_to_1005_study as study
from src.qgeognn_al.active_learning_v2.cw_lcmd_extension_runner import select_current_cw_lcmd


def test_scope_rounds_and_hard_stop():
    assert study.SEEDS == (157, 6101) and study.SOURCE_ROUND == 10
    assert study.NEW_ROUNDS == tuple(range(11, 22))
    assert len(study.ALL_BUDGETS) == 22 and study.ALL_BUDGETS[-1] == 1005
    assert len(study.PRIMARY_BUDGETS) == 12 and study.PRIMARY_BUDGETS[0] == 653


@pytest.mark.parametrize("seed", study.SEEDS)
def test_actual_l653_lineage_and_checkpoint_are_exact(seed):
    audit = study.audit_reuse(); anchor = next(x for x in audit["source_anchors"] if x["outer_seed"] == seed)
    assert anchor["source_active_labels"] == 653 and anchor["source_unlabeled_rows"] == 2677
    for field in ("ordered_L653_ids_hash", "ordered_U653_ids_hash", "L653_set_hash", "U653_set_hash",
                  "selected_lineage_ordered_hash", "checkpoint_sha256", "checkpoint_state_hash",
                  "prediction_sha256", "preprocessing_hash", "target_scale_hash",
                  "center_width_transform_hash", "split_hash", "source_protocol_sha256"):
        assert anchor[field]
    assert anchor["reuse_legal"] is True


def test_historical_comparators_are_complete_read_only():
    values = study.audit_reuse()["historical_comparators"]
    assert len(values) == 8
    assert all(x["rows"] == 22 and x["training_performed_by_continuation"] is False for x in values)


def test_all_current_l_are_centers_and_selection_is_unique(monkeypatch):
    seen = {}
    class Result: selected_pool_positions = np.arange(32); trace = []
    def fake(pool, centers, batch): seen.update(pool=len(pool), centers=len(centers), batch=batch); return Result()
    monkeypatch.setattr("src.qgeognn_al.active_learning_v2.cw_lcmd_extension_runner.lcmd_tp_select", fake)
    result = select_current_cw_lcmd(np.zeros((909 + 80, 512)), 909)
    assert seen == {"pool": 80, "centers": 909, "batch": 32}
    assert len(set(result["selected_pool_positions"])) == 32


def test_current_round_checkpoint_drives_each_acquisition():
    source = inspect.getsource(runner.run_continuation); acquire = inspect.getsource(runner._acquire)
    assert "_acquire(context, round_index" in source and "model_path" in acquire
    assert "static_round3_ranking_forbidden" in acquire


def test_no_test_reveal_in_execution_path():
    for function in (runner.run_continuation, runner.execute_seed, runner.finalize_pre_test):
        assert "final_test_evaluation" not in inspect.getsource(function)


def test_report_requires_complete_22_point_freeze(tmp_path):
    runner.atomic_json(tmp_path / "global_pre_test_freeze.json", {"status": "PENDING_CONTINUATIONS", "new_entries": {}})
    with pytest.raises(RuntimeError, match="22-point global freeze"):
        runner.reveal_and_report(tmp_path)


def _grid(budgets=study.ALL_BUDGETS):
    return pd.DataFrame([
        {"outer_seed": seed, "method": method, "active_label_count": budget,
         "combined_normalized_RMSE": 1 - budget / 2000}
        for seed in study.SEEDS for method in study.ALL_METHODS for budget in budgets
    ])


def test_full_aulc_requires_all_22_points():
    result = reporting._aulc(_grid(), study.ALL_BUDGETS, "AULC_333_1005")
    assert len(result) == 10


@pytest.mark.parametrize("missing", study.ALL_BUDGETS)
def test_missing_any_budget_fails_loudly(missing):
    frame = _grid(); frame = frame.loc[~(frame.outer_seed.eq(157) & frame.method.eq(study.METHOD) & frame.active_label_count.eq(missing))]
    with pytest.raises(RuntimeError, match="incomplete reporting budget grid"):
        reporting._aulc(frame, study.ALL_BUDGETS, "AULC_333_1005")


def test_decision_keeps_all_strong_comparators():
    source = inspect.getsource(reporting._decision)
    assert "STRONG_COMPARATORS" in source and "strongest" not in source.lower()


def test_final_round_has_no_acquisition_by_construction():
    source = inspect.getsource(runner.run_continuation)
    assert "if round_index < FINAL_ROUND" in source
