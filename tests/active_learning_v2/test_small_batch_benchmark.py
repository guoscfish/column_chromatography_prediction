import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.qgeognn_al.active_learning_v2.acquisition import acquire_batches
from src.qgeognn_al.active_learning_v2.benchmark import smoke, validate_freezes
from src.qgeognn_al.active_learning_v2.benchmark_protocol import (
    ALL_SEEDS, CONFIRMATION_SEEDS, DEVELOPMENT_SEEDS, label_budget,
)
from src.qgeognn_al.active_learning_v2.benchmark_reporting import decision_gate
from src.qgeognn_al.active_learning_v2.cache import seal_cache, verify_cache
from src.qgeognn_al.active_learning_v2.coverage import coreset_tp_select
from src.qgeognn_al.active_learning_v2.diagnostics import duplicate_audit
from src.qgeognn_al.active_learning_v2.uncertainty import ensemble_uncertainty_select


def test_coreset_uses_l0_and_new_centers_and_stable_ties():
    assert coreset_tp_select(np.array([[0.], [4.], [10.], [11.]]), np.array([[0.]]), 3).tolist() == [3, 1, 2]
    assert coreset_tp_select(np.zeros((5, 3)), np.zeros((1, 3)), 5).tolist() == list(range(5))


def test_uncertainty_is_epistemic_scaled_deterministic_and_ignores_quantile_width():
    values = np.zeros((3, 4, 6))
    values[:, 0, 1] = [-2, 0, 2]
    values[:, 1, 4] = [-4, 0, 4]
    values[:, 2, 1] = [-1, 0, 1]
    values[:, 3, 1] = [-2, 0, 2]
    selected, scores = ensemble_uncertainty_select(values, [1, 4], 4)
    assert selected.tolist() == [0, 3, 1, 2]
    assert scores[0] == pytest.approx(np.std([-2, 0, 2], ddof=0))
    values[:, :, [0, 2, 3, 5]] = 1e12
    assert np.array_equal(ensemble_uncertainty_select(values, [1, 4], 4)[0], selected)
    with pytest.raises(ValueError):
        ensemble_uncertainty_select(values[:2], [1, 4], 4)


@pytest.mark.parametrize("batch", [16, 32, 64])
def test_all_methods_exact_batch_and_independent_deterministic_controls(batch):
    rng = np.random.default_rng(14)
    args = dict(gradients=rng.normal(size=(160, 512)), representations=rng.normal(size=(160, 128)),
                ensemble_predictions=rng.normal(size=(3, 144, 6)), l0_count=16, scales=(10, 20),
                outer_seed=157, batch_size=batch)
    first, second = acquire_batches(**args), acquire_batches(**args)
    assert len(first) == (8 if batch == 32 else 6)
    for name, positions in first.items():
        assert len(positions) == len(set(positions)) == batch
        assert np.array_equal(positions, second[name])
    controls = {tuple(sorted(first[f"random_control_{i}"])) for i in range(5)}
    assert len(controls) == 5


def test_cache_contract_content_and_partial_mismatches_are_rejected(tmp_path):
    path = tmp_path / "array.npy"
    np.save(path, np.arange(3))
    seal_cache(path, {"checkpoint": "abc", "ordered_ids": ["a", "b"]})
    assert verify_cache(path, {"checkpoint": "abc", "ordered_ids": ["a", "b"]})
    with pytest.raises(RuntimeError, match="mismatch"):
        verify_cache(path, {"checkpoint": "def", "ordered_ids": ["a", "b"]})
    with pytest.raises(RuntimeError, match="mismatch"):
        verify_cache(path, {"checkpoint": "abc", "ordered_ids": ["b", "a"]})
    np.save(path, np.arange(4))
    with pytest.raises(RuntimeError, match="mismatch"):
        verify_cache(path, {"checkpoint": "abc", "ordered_ids": ["a", "b"]})
    path.unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        verify_cache(path, {})


def test_dual_budget_and_cohorts_are_frozen():
    assert label_budget(333, 416, 3330, 4163)["total_observed_label_count"] == 749
    assert label_budget(365, 416, 3330, 4163)["total_observed_label_count"] == 781
    assert CONFIRMATION_SEEDS == (157, 887, 2357, 6101, 12203)
    assert DEVELOPMENT_SEEDS == (73, 311, 1297, 4093, 8191)
    assert len(set(ALL_SEEDS)) == 10


def test_duplicate_audit_separates_collision_and_full_gradient_symmetry():
    identities = pd.DataFrame({"sample_id": list("abcde"), "canonical_index": range(5),
                               "input_hash": ["A", "A", "B", "C", "D"], "raw_X_hash": list("aabcd")})
    gradients = np.array([[0, 0], [0, 0], [1, 1], [1, 1], [2, 2]])
    selection = pd.DataFrame({"arm": ["random"] * 3, "selection_order": [0, 1, 2], "sample_id": list("bcd")})
    inputs, duplicate = duplicate_audit(identities, gradients, 1, selection, lambda i: str(i))
    assert inputs.selected_redundant_fraction.iloc[0] == pytest.approx(1 / 3)
    assert duplicate.explained_by_exact_X.tolist() == [True, False]
    assert duplicate.explanation.iloc[1] == "CountSketch_collision_between_distinct_full_Jacobians"
    _, duplicate = duplicate_audit(identities, gradients, 1, selection, lambda i: "same")
    assert duplicate.explanation.iloc[1] == "same_full_Jacobian_local_symmetry_or_saturation"


def test_complete_smoke_uses_production_freeze_fit_evaluate_pipeline(tmp_path):
    report = smoke(tmp_path / "smoke")
    assert report["status"] == "PASS" and report["formal_test_truth_access_count"] == 0
    for batch in (32, 16, 64):
        directory = tmp_path / f"smoke/runtime/seed_29/b{batch}"
        fits = pd.read_csv(directory / "fit_audit.csv")
        evaluation = fits.loc[~fits.arm.str.startswith("ensemble_member_")]
        assert evaluation.initialization_hash.nunique() == 1
        for key in ("l0_ids_hash", "validation_ids_hash", "test_ids_hash"):
            assert fits[key].nunique() == 1
        acquisitions = pd.read_csv(directory / "selected_batches.csv")
        assert acquisitions.groupby("arm").size().eq(batch).all()
        access = pd.read_csv(directory / "label_access_audit.csv")
        assert not access.purpose.eq("final_test_evaluation").any()
    path = tmp_path / "smoke/runtime/seed_29/b32/lcmd/predictions.csv.gz"
    with pytest.raises(RuntimeError, match="seed 31 is not frozen"):
        validate_freezes(tmp_path / "smoke/runtime", (29, 31), 32, True)
    path.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="frozen artifact changed"):
        validate_freezes(tmp_path / "smoke/runtime", (29,), 32, True)


def test_decision_gate_uses_five_outer_seeds_and_endpoint_guard():
    rows = [{"outer_seed": seed, "arm": arm, "combined_normalized_RMSE": 0.8 if arm == "lcmd" else 1.,
             "V1_RMSE": 0.9 if arm == "lcmd" else 1., "V2_RMSE": 0.9 if arm == "lcmd" else 1.}
            for seed in CONFIRMATION_SEEDS for arm in ["lcmd", "uncertainty", "coreset", *(f"random_control_{i}" for i in range(5))]]
    arms = pd.DataFrame(rows)
    assert decision_gate(arms, CONFIRMATION_SEEDS)["decision"] == "BEST_CURRENT_ROW_ACQUISITION"
    arms.loc[arms.arm.eq("lcmd"), "V1_RMSE"] = 1.02
    assert decision_gate(arms, CONFIRMATION_SEEDS)["decision"] == "BEST_CURRENT_ROW_ACQUISITION"
    arms.loc[arms.arm.eq("lcmd"), "V1_RMSE"] = 1.021
    assert decision_gate(arms, CONFIRMATION_SEEDS)["decision"] == "LCMD_BEATS_RANDOM_NOT_OTHER_BASELINES"
    arms.loc[arms.arm.eq("lcmd"), "combined_normalized_RMSE"] = 1.1
    assert decision_gate(arms, CONFIRMATION_SEEDS)["decision"] == "LARGE_BATCH_ONLY_SIGNAL"
    with pytest.raises(ValueError):
        decision_gate(arms.iloc[1:], CONFIRMATION_SEEDS)


def test_formal_execution_rejects_pre_preregistration_commit():
    from src.qgeognn_al.active_learning_v2.benchmark_protocol import BASE_COMMIT, assert_formal_authorized
    with pytest.raises(RuntimeError, match="requires Commit A"):
        assert_formal_authorized(BASE_COMMIT)


def test_secondary_cannot_start_before_primary_freeze(monkeypatch, tmp_path):
    from src.qgeognn_al.active_learning_v2 import benchmark
    monkeypatch.setattr(benchmark, "assert_formal_authorized", lambda commit, study: commit)
    with pytest.raises(RuntimeError, match="requires all primary results frozen"):
        benchmark.execute_secondary("test_commit", tmp_path)


def test_frozen_333_lcmd_selected_ids_regression():
    from src.qgeognn_al.active_learning_v2.benchmark_protocol import OLD_STUDY
    from src.qgeognn_al.active_learning_v2.lcmd import lcmd_tp_select
    for seed in DEVELOPMENT_SEEDS:
        runtime = OLD_STUDY / "runtime" / f"seed_{seed}"
        split = pd.read_csv(OLD_STUDY / "splits" / f"row_seed_{seed}.csv")
        with np.load(runtime / "gradient_features.npz") as cache:
            features, indices = cache["features"], cache["canonical_indices"]
        selected = lcmd_tp_select(features[333:], features[:333], 333).selected_pool_positions
        actual = split.iloc[indices[333:][selected]].sample_id.tolist()
        old = pd.read_csv(runtime / "selected_batches.csv")
        assert actual == old.loc[old.arm.eq("lcmd")].sort_values("selection_order").sample_id.tolist()
