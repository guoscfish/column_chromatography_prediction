import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.qgeognn_al.active_learning_v2 import sequential_acquisition as acquisition
from src.qgeognn_al.active_learning_v2.benchmark_protocol import initialization_seed
from src.qgeognn_al.active_learning_v2.protocol import RestrictedLabelStore
from src.qgeognn_al.active_learning_v2.sequential_protocol import (
    ACQUISITION_ROUNDS,
    ACTIVE_LABEL_BUDGETS,
    BATCH_SIZE,
    CONFIRMATION_SEEDS,
    FINAL_ACTIVE_LABELS,
    INITIAL_ACTIVE_LABELS,
    INITIAL_POOL_ROWS,
    METHODS,
    label_budget,
    random_trajectory_seed,
)
from src.qgeognn_al.active_learning_v2.sequential_reporting import (
    incremental_label_saving,
    labels_to_target,
    normalized_aulc,
    write_final_outputs,
)


def _advance_random(seed: int, stop: int, state=None):
    original = tuple(f"id-{index}" for index in range(INITIAL_POOL_ROWS))
    if state is None:
        labeled = tuple(f"l0-{index}" for index in range(INITIAL_ACTIVE_LABELS))
        unlabeled = original
        selected_all = ()
        start = 1
    else:
        labeled, unlabeled, selected_all, start = state
    for round_index in range(start, stop + 1):
        selected = acquisition.random_round_batch(original, unlabeled, seed, round_index)
        selected_set = set(selected)
        next_labeled = labeled + selected
        next_unlabeled = tuple(value for value in unlabeled if value not in selected_set)
        acquisition.validate_trajectory_transition(
            labeled, unlabeled, selected, next_labeled, next_unlabeled
        )
        labeled, unlabeled = next_labeled, next_unlabeled
        selected_all += selected
    return labeled, unlabeled, selected_all, stop + 1


def test_frozen_cohort_methods_rounds_batch_and_budgets():
    assert CONFIRMATION_SEEDS == (157, 887, 2357, 6101, 12203)
    assert METHODS == ("random", "hybrid", "lcmd")
    assert BATCH_SIZE == 32 and ACQUISITION_ROUNDS == 21
    assert ACTIVE_LABEL_BUDGETS == tuple(range(333, 1006, 32))
    assert ACTIVE_LABEL_BUDGETS[-1] == FINAL_ACTIVE_LABELS == 1005


def test_each_transition_is_exact_plus_and_minus_32_without_duplicates():
    labeled, unlabeled, selected, _ = _advance_random(157, ACQUISITION_ROUNDS)
    assert len(labeled) == INITIAL_ACTIVE_LABELS + 21 * BATCH_SIZE
    assert len(unlabeled) == INITIAL_POOL_ROWS - 21 * BATCH_SIZE
    assert len(selected) == len(set(selected)) == 21 * BATCH_SIZE


def test_transition_rejects_selection_outside_previous_u_and_duplicate_ids():
    old_l = tuple(f"l{i}" for i in range(333))
    old_u = tuple(f"u{i}" for i in range(100))
    selected = old_u[:31] + ("not-in-u",)
    with pytest.raises(RuntimeError, match="previous U_t"):
        acquisition.validate_trajectory_transition(
            old_l, old_u, selected, old_l + selected, old_u[31:]
        )
    with pytest.raises(RuntimeError, match="duplicate"):
        acquisition.validate_trajectory_transition(
            old_l, old_u, old_u[:31] + (old_u[0],), old_l, old_u
        )


def test_random_trajectory_is_one_deterministic_nested_permutation():
    first = _advance_random(887, ACQUISITION_ROUNDS)
    second = _advance_random(887, ACQUISITION_ROUNDS)
    assert first[:3] == second[:3]
    assert random_trajectory_seed(887) == 887 * 1_000_003 + 900_001
    assert _advance_random(157, 1)[2] != _advance_random(887, 1)[2]


def test_random_resume_matches_uninterrupted_trajectory_exactly():
    uninterrupted = _advance_random(2357, ACQUISITION_ROUNDS)
    partial = _advance_random(2357, 7)
    resumed = _advance_random(2357, ACQUISITION_ROUNDS, partial)
    assert resumed[:3] == uninterrupted[:3]


def test_hybrid_recomputes_uncertainty_from_each_current_ensemble():
    rng = np.random.default_rng(9)
    latent = rng.normal(size=(20 + 128, 128))
    first = np.zeros((3, 128, 6))
    second = np.zeros_like(first)
    first[:, :, 1] = np.vstack([np.zeros(128), np.zeros(128), np.arange(128)])
    second[:, :, 1] = first[:, ::-1, 1]
    a = acquisition.v2_hybrid_select_current(latent, first, 20, (1.0, 1.0))
    b = acquisition.v2_hybrid_select_current(latent, second, 20, (1.0, 1.0))
    assert not np.array_equal(a["shortlist_pool_positions"], b["shortlist_pool_positions"])


def test_hybrid_shortlist_is_exact_current_u_top_quarter():
    rng = np.random.default_rng(12)
    latent = rng.normal(size=(33 + 129, 128))
    predictions = rng.normal(size=(3, 129, 6))
    result = acquisition.v2_hybrid_select_current(latent, predictions, 33, (2.0, 3.0))
    assert result["pool_size"] == 129
    assert result["shortlist_size"] == 33
    assert len(result["selected_pool_positions"]) == BATCH_SIZE
    assert set(result["selected_pool_positions"]) <= set(result["shortlist_pool_positions"])


def test_hybrid_installs_every_current_labeled_row_as_center(monkeypatch):
    seen = {}

    def fake_select(pool, centers, batch_size):
        seen["pool"] = len(pool)
        seen["centers"] = len(centers)
        return np.arange(batch_size)

    monkeypatch.setattr(acquisition, "coreset_tp_select", fake_select)
    rng = np.random.default_rng(18)
    result = acquisition.v2_hybrid_select_current(
        rng.normal(size=(397 + 200, 128)), rng.normal(size=(3, 200, 6)), 397, (1.0, 1.0)
    )
    assert seen == {"pool": result["shortlist_size"], "centers": 397}
    assert result["center_count"] == 397


def test_lcmd_installs_current_l_t_and_recomputes_supplied_features(monkeypatch):
    calls = []

    class Result:
        selected_pool_positions = np.arange(BATCH_SIZE)
        initial_nearest_center = np.zeros(80, dtype=int)
        initial_nearest_sq_distance = np.ones(80)
        trace = []

    def fake_lcmd(pool, centers, batch_size):
        calls.append((pool.copy(), centers.copy(), batch_size))
        return Result()

    monkeypatch.setattr(acquisition, "lcmd_tp_select", fake_lcmd)
    first = np.zeros((365 + 80, 512))
    second = first.copy(); second[-1, 0] = 1.0
    acquisition.acquire_sequential_batch(method="lcmd", labeled_count=365, gradient_features=first)
    acquisition.acquire_sequential_batch(method="lcmd", labeled_count=365, gradient_features=second)
    assert calls[0][1].shape == (365, 512)
    assert calls[0][0].shape == (80, 512)
    assert not np.array_equal(calls[0][0], calls[1][0])


def test_acquisition_surface_has_no_truth_label_or_test_input():
    parameters = set(inspect.signature(acquisition.acquire_sequential_batch).parameters)
    assert not parameters & {"labels", "targets", "truth", "test", "test_truth", "u_truth"}


def test_restricted_store_blocks_u0_before_selection_and_test_before_global_freeze(tmp_path):
    source = tmp_path / "targets.csv"
    source.write_text("sample_id,V1_ml,V2_ml\nl0,1,2\nu0,3,4\nvalidation,5,6\ntest,7,8\n")
    partition = pd.DataFrame({
        "sample_id": ["l0", "u0", "validation", "test"],
        "role": ["l0", "u0", "validation", "test"],
    })
    store = RestrictedLabelStore(source, partition)
    with pytest.raises(PermissionError):
        store.reveal(["u0"], "after_acquisition_fit")
    with pytest.raises(PermissionError):
        store.reveal(["test"], "final_test_evaluation")
    store.freeze_acquisitions(["u0"])
    assert store.reveal(["u0"], "after_acquisition_fit").shape == (1, 2)
    with pytest.raises(PermissionError):
        store.reveal(["test"], "final_test_evaluation")


def test_l0_target_scales_and_same_evaluation_initialization_are_frozen():
    runner = Path("src/qgeognn_al/active_learning_v2/sequential_runner.py").read_text()
    assert 'endpoint_scales_fit_ids_hash' in runner
    assert 'context.preprocessing["target_scales"]' in runner
    for seed in CONFIRMATION_SEEDS:
        assert len({initialization_seed(seed, 0) for _ in METHODS}) == 1


def test_dual_label_budget_accounting_includes_shared_validation():
    assert label_budget(333)["total_observed_non_test_labels"] == 749
    final = label_budget(1005)
    assert final["total_observed_non_test_labels"] == 1421
    assert final["active_label_fraction_outer_train"] == pytest.approx(1005 / 3330)
    assert final["total_observed_fraction_full_dataset"] == pytest.approx(1421 / 4163)


def test_normalized_aulc_trapezoid_and_units():
    errors = np.linspace(1.0, 0.5, len(ACTIVE_LABEL_BUDGETS))
    raw, normalized = normalized_aulc(np.asarray(ACTIVE_LABEL_BUDGETS), errors)
    assert raw == pytest.approx(0.75 * (1005 - 333))
    assert normalized == pytest.approx(0.75)


def test_labels_to_target_uses_first_adjacent_linear_crossing():
    labels = np.array([333, 365, 397, 429])
    errors = np.array([1.0, 0.9, 0.7, 0.6])
    assert labels_to_target(labels, errors, 0.8) == pytest.approx(381.0)
    assert labels_to_target(labels, errors, 1.0) == 333


def test_incremental_label_saving_uses_l0_as_sunk_cost():
    result = incremental_label_saving(750)
    assert result["saved_experimental_labels"] == 255
    assert result["saving_total"] == pytest.approx(255 / 1005)
    assert result["saving_incremental"] == pytest.approx(255 / 672)


def test_target_not_reached_is_censored_without_extrapolation():
    assert labels_to_target(np.array([333, 365, 397]), np.array([1.0, 0.9, 0.8]), 0.7) is None
    result = incremental_label_saving(None)
    assert result["status"] == "CENSORED_GT_1005"
    assert result["saving_incremental"] is None


def test_formal_runner_recomputes_current_features_and_has_global_test_barrier():
    source = Path("src/qgeognn_al/active_learning_v2/sequential_runner.py").read_text()
    assert "extract_representations(model, context.atom, context.angle, current)" in source
    assert "extract_q50_gradient_sketches(" in source
    assert 'current = np.r_[labeled_indices, unlabeled_indices]' in source
    assert "_verify_global_freeze(Path(study))" in source
    assert source.index("_verify_global_freeze(Path(study))", source.index("def reveal_test_and_report")) < source.index(
        'store.reveal(test_ids, "final_test_evaluation")', source.index("def reveal_test_and_report")
    )


def test_synthetic_complete_matrix_writes_every_registered_report_output(tmp_path):
    rows = []
    for seed_index, seed in enumerate(CONFIRMATION_SEEDS):
        for method, slope in (("random", 0.010), ("hybrid", 0.012), ("lcmd", 0.014)):
            for round_index, active in enumerate(ACTIVE_LABEL_BUDGETS):
                error = 1.0 + seed_index * 0.001 - slope * round_index
                rows.append({
                    "outer_seed": seed,
                    "method": method,
                    "round": round_index,
                    **label_budget(active),
                    "combined_normalized_RMSE": error,
                    "V1_RMSE": error * 10,
                    "V2_RMSE": error * 20,
                    "V1_MAE": error * 8,
                    "V2_MAE": error * 16,
                    "V1_R2": 1 - error / 2,
                    "V2_R2": 1 - error / 3,
                })
    full = pd.DataFrame([
        {
            "outer_seed": seed,
            **label_budget(3330),
            "combined_normalized_RMSE": 0.5 + index * 0.001,
            "V1_RMSE": 5.0,
            "V2_RMSE": 10.0,
            "V1_MAE": 4.0,
            "V2_MAE": 8.0,
            "V1_R2": 0.75,
            "V2_R2": 0.8,
        }
        for index, seed in enumerate(CONFIRMATION_SEEDS)
    ])
    audit = pd.DataFrame([{"status": "synthetic"}])
    result = write_final_outputs(tmp_path, pd.DataFrame(rows), full, audit, audit, audit, audit)
    assert result["decision"] in {
        "SEQUENTIAL_LCMD_BEST",
        "SEQUENTIAL_HYBRID_BEST",
        "SEQUENTIAL_LCMD_HYBRID_COMPARABLE",
        "NO_CLEAR_SEQUENTIAL_AL_GAIN",
    }
    expected_results = {
        "learning_curve_metrics.csv", "cohort_mean_learning_curve.csv",
        "normalized_aulc.csv", "labels_to_target.csv", "label_saving.csv",
        "full_data_reference.csv", "endpoint_learning_curves.csv", "round_runtime.csv",
        "trajectory_integrity_audit.csv", "label_access_audit.csv",
        "initialization_hash_audit.csv",
    }
    assert expected_results == {path.name for path in (tmp_path / "results").glob("*.csv")}
    assert len(list((tmp_path / "figures").glob("*.png"))) == 6
    assert (tmp_path / "decision.json").exists()
    assert (tmp_path / "FINAL_REPORT.md").exists()
