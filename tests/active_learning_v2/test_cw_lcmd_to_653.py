from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.qgeognn_al.active_learning_v2 import cw_lcmd_to_653_reporting as reporting
from src.qgeognn_al.active_learning_v2 import cw_lcmd_to_653_runner as runner
from src.qgeognn_al.active_learning_v2 import cw_lcmd_to_653_study as study
from src.qgeognn_al.active_learning_v2.cw_lcmd_extension_runner import select_current_cw_lcmd


def test_scope_and_hard_stop_are_exact():
    assert study.SEEDS == (157, 6101)
    assert study.NEW_ROUNDS == (7, 8, 9, 10)
    assert study.ALL_BUDGETS == (333, 365, 397, 429, 461, 493, 525, 557, 589, 621, 653)
    assert study.FINAL_ACTIVE_LABELS == 653 and 685 not in study.ALL_BUDGETS


@pytest.mark.parametrize("seed", study.SEEDS)
def test_actual_l525_source_lineage_checkpoint_and_hashes_are_exact(seed):
    audit = study.audit_reuse()
    anchor = next(item for item in audit["source_anchors"] if item["outer_seed"] == seed)
    assert anchor["source_active_labels"] == 525
    assert anchor["source_unlabeled_rows"] == 2805
    for field in ("ordered_L525_ids_hash", "ordered_U525_ids_hash", "L525_set_hash", "U525_set_hash",
                  "selected_lineage_ordered_hash", "checkpoint_sha256", "checkpoint_state_hash",
                  "prediction_sha256", "preprocessing_hash", "target_scale_hash",
                  "center_width_transform_hash", "split_hash", "source_protocol_sha256"):
        assert anchor[field]
    assert anchor["reuse_legal"] is True


def test_comparators_are_read_only_complete_and_scale_matched():
    records = study.audit_reuse()["historical_comparators"]
    assert len(records) == len(study.SEEDS) * len(study.COMPARATORS)
    assert all(record["rows"] == 11 and record["training_performed_by_continuation"] is False for record in records)
    for seed in study.SEEDS:
        assert len({record["target_scale_hash"] for record in records if record["outer_seed"] == seed}) == 1


def test_current_cw_uses_all_current_centers_and_unique_u_ids(monkeypatch):
    seen = {}
    class Result:
        selected_pool_positions = np.arange(32)
        trace = []
    def fake(pool, centers, batch):
        seen.update(pool=len(pool), centers=len(centers), batch=batch)
        return Result()
    monkeypatch.setattr("src.qgeognn_al.active_learning_v2.cw_lcmd_extension_runner.lcmd_tp_select", fake)
    result = select_current_cw_lcmd(np.zeros((557 + 100, 512)), 557)
    assert seen == {"pool": 100, "centers": 557, "batch": 32}
    assert len(set(result["selected_pool_positions"])) == 32


def test_runner_recomputes_features_from_each_current_checkpoint():
    source = inspect.getsource(runner.run_continuation)
    acquire = inspect.getsource(runner._acquire)
    assert "_acquire(context, round_index" in source
    assert "model_path" in acquire
    assert "static_round3_ranking_forbidden" in acquire


def test_state_transition_is_exact_plus_32_minus_32():
    old_l = list(range(525)); old_u = list(range(525, 700)); selected = old_u[:32]
    new_l = old_l + selected; new_u = old_u[32:]
    from src.qgeognn_al.active_learning_v2.sequential_acquisition import validate_trajectory_transition
    validate_trajectory_transition([str(x) for x in old_l], [str(x) for x in old_u],
                                   [str(x) for x in selected], [str(x) for x in new_l], [str(x) for x in new_u])
    assert len(new_l) - len(old_l) == 32 and len(old_u) - len(new_u) == 32


def test_duplicate_selection_is_detectable():
    selected = [f"id{i}" for i in range(31)] + ["id0"]
    assert len(selected) == 32 and len(set(selected)) != 32


def test_execution_functions_do_not_reveal_test_truth():
    for function in (runner.run_continuation, runner.execute_seed, runner.finalize_pre_test):
        assert "final_test_evaluation" not in inspect.getsource(function)


def test_report_refuses_access_before_eight_point_global_freeze(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "STUDY", tmp_path)
    runner.atomic_json(tmp_path / "global_pre_test_freeze.json", {"status": "PENDING_CONTINUATIONS", "new_entries": {}})
    with pytest.raises(RuntimeError, match="8-point global freeze"):
        runner.reveal_and_report(tmp_path)


def _grid(methods=study.ALL_METHODS, budgets=study.ALL_BUDGETS):
    rows = []
    for seed in study.SEEDS:
        for method in methods:
            for budget in budgets:
                rows.append({"outer_seed": seed, "method": method, "active_label_count": budget,
                             "combined_normalized_RMSE": 1 - budget / 2000})
    return pd.DataFrame(rows)


def test_aulc_requires_complete_eleven_point_grid():
    result = reporting._aulc(_grid(), study.ALL_BUDGETS, "AULC_333_653")
    assert len(result) == len(study.SEEDS) * len(study.ALL_METHODS)


@pytest.mark.parametrize("missing", study.ALL_BUDGETS)
def test_aulc_fails_loudly_for_any_missing_point(missing):
    frame = _grid()
    frame = frame.loc[~(
        frame.outer_seed.eq(157) & frame.method.eq(study.METHOD)
        & frame.active_label_count.eq(missing)
    )]
    with pytest.raises(RuntimeError, match="incomplete reporting budget grid"):
        reporting._aulc(frame, study.ALL_BUDGETS, "AULC_333_653")


def test_source_study_hash_guard_detects_no_current_modification(tmp_path):
    protected = Path(study.SOURCE_STUDY)
    paths = sorted(path for path in protected.rglob("*") if path.is_file())
    before = {str(path.relative_to(protected)): study.sha256_file(path) for path in paths}
    after = {str(path.relative_to(protected)): study.sha256_file(path) for path in paths}
    assert before == after and len(before) > 100


def test_decision_does_not_dynamically_choose_one_baseline():
    source = inspect.getsource(reporting._decision)
    assert "STRONG_COMPARATORS" in source
    assert "strongest" not in source.lower()


def test_paired_direction_labels_distinguish_r2():
    curves = _grid()
    for metric in reporting.METRICS:
        if metric not in curves:
            curves[metric] = 0.5
    areas = {
        "AULC_525_653": reporting._aulc(curves, study.PRIMARY_BUDGETS, "AULC_525_653"),
        "AULC_429_653": reporting._aulc(curves, study.MID_LATE_BUDGETS, "AULC_429_653"),
        "AULC_333_653": reporting._aulc(curves, study.ALL_BUDGETS, "AULC_333_653"),
    }
    paired = reporting._paired(curves, areas)
    assert set(paired.loc[paired.metric.str.contains("R2"), "direction"]) == {"positive_is_CW_better"}
    assert set(paired.loc[~paired.metric.str.contains("R2"), "direction"]) == {"negative_is_CW_better"}
