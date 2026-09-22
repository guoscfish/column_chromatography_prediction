import inspect

import numpy as np
import pytest

from src.qgeognn_al.active_learning_v2 import same_state_branching as study


def synthetic_state(rows=96, labeled=40):
    return {
        "outer_indices": list(range(rows)),
        "labeled_indices": list(range(labeled)),
        "unlabeled_indices": list(range(labeled, rows)),
        "labeled_ids": [f"id{i}" for i in range(labeled)],
        "unlabeled_ids": [f"id{i}" for i in range(labeled, rows)],
    }


@pytest.mark.parametrize("budget,expected", [(429, 3), (653, 10)])
def test_anchor_budget_is_located_from_frozen_schedule(budget, expected):
    assert study._round_for_budget(budget) == expected


@pytest.mark.parametrize("seed", study.SEEDS)
@pytest.mark.parametrize("source", study.SOURCES)
@pytest.mark.parametrize("budget", study.ANCHOR_BUDGETS)
def test_actual_anchor_lineage_is_exact_and_test_blind(seed, source, budget):
    before = {}
    lineage = study.inspect_anchor(seed, source, budget)
    for path, digest in lineage["source_files"].items():
        before[path] = digest
    assert lineage["source_round"] == study._round_for_budget(budget)
    assert len(lineage["labeled_indices"]) == budget
    assert len(lineage["unlabeled_indices"]) == 3330 - budget
    assert len(set(lineage["labeled_indices"]) | set(lineage["unlabeled_indices"])) == 3330
    assert lineage["test_truth_access_count"] == 0
    assert lineage["checkpoint_sha256"]
    assert lineage["checkpoint_state_hash"]
    assert lineage["gradient_bank_hash"]
    study.assert_hashes(before)


@pytest.mark.parametrize("strategy", study.STRATEGIES)
def test_three_selectors_are_deterministic_unique_current_u_only(strategy):
    rng = np.random.default_rng(901)
    features = rng.normal(size=(96, 18))
    state = synthetic_state()
    first = study.select_batch(strategy, features, state)
    second = study.select_batch(strategy, features.copy(), state)
    assert first == second
    assert len(first["selected_ids"]) == len(set(first["selected_ids"])) == 32
    assert set(first["selected_indices"]) <= set(state["unlabeled_indices"])
    assert first["test_truth_access_count"] == 0


def test_selector_surface_has_no_label_or_performance_input():
    parameters = set(inspect.signature(study.select_batch).parameters)
    assert not parameters & {"labels", "truth", "targets", "test", "performance", "comparator"}


def test_method_defining_normalizations_are_preserved():
    rng = np.random.default_rng(117)
    features = rng.normal(size=(96, 12))
    state = synthetic_state()
    lcmd = study.select_batch("gradient_lcmd", features, state)["objective"]
    ivr = study.select_batch("kernel_ivr", features, state)["objective"]
    maxdet = study.select_batch("gradient_maxdet", features, state)["objective"]
    assert lcmd["definition"] == "raw_squared_euclidean_LCMD_TP"
    assert "global_RMS" in ivr["definition"] and ivr["raw_mean_squared_norm"] > 0
    assert "D-optimal" in maxdet["definition"] and maxdet["normalization_scale"] > 0


def test_feature_diagnostics_use_participation_ratio_and_current_l_coverage():
    features = np.array([[1., 0.], [0., 1.], [1., 1.], [2., 0.]])
    result = study.feature_diagnostics(features, np.array([0, 1]), np.array([2, 3]))
    gram = features.T @ features
    expected = np.trace(gram) ** 2 / np.square(gram).sum()
    assert result["gradient_effective_rank_participation_ratio"] == pytest.approx(expected)
    assert result["coverage_nearest_distance_mean"] == pytest.approx((1 + 1) / 2)
    assert result["active_label_count"] == 2 and result["candidate_pool_size"] == 2


def test_validation_slopes_have_frozen_window_semantics():
    history = [{"active_label_count": 333 + 32 * i, "validation_combined_nrmse": 1 - .1 * i}
               for i in range(4)]
    result = study._validation_slopes(history)
    assert result["recent_validation_last_step_delta"] == pytest.approx(-.1)
    assert result["recent_validation_2step_linear_slope_per_label"] == pytest.approx(-.1 / 32)
    assert result["recent_validation_3step_linear_slope_per_label"] == pytest.approx(-.1 / 32)


def test_advance_is_ordered_exact_b32_and_isolated():
    class Context:
        def ids(self, indices):
            return [f"id{i}" for i in indices]
    state = synthetic_state()
    selection = {"selected_indices": list(range(40, 72)), "selected_ids": [f"id{i}" for i in range(40, 72)]}
    updated = study._advance(Context(), state, selection)
    assert updated["labeled_indices"] == list(range(72))
    assert updated["unlabeled_indices"] == list(range(72, 96))
    assert state["labeled_indices"] == list(range(40))


def test_rollout_endpoints_are_exact():
    assert 429 + 2 * study.BATCH_SIZE == 493
    assert 653 + 2 * study.BATCH_SIZE == 717
    assert len(study.SEEDS) * len(study.SOURCES) * len(study.ANCHOR_BUDGETS) * len(study.STRATEGIES) * 2 == 48


def test_execution_functions_do_not_contain_test_reveal_purpose():
    for function in (study.select_batch, study._run_branch, study.execute_seed, study.global_freeze):
        assert "final_test_evaluation" not in inspect.getsource(function)


def test_report_refuses_label_access_before_global_freeze(monkeypatch):
    accessed = []
    def denied():
        raise RuntimeError("incomplete global branch freeze")
    monkeypatch.setattr(study, "validate_seal", lambda: {})
    monkeypatch.setattr(study, "_verify_global_branch_freeze", denied)
    monkeypatch.setattr(study, "RestrictedLabelStore", lambda *args: accessed.append(args))
    with pytest.raises(RuntimeError, match="incomplete"):
        study.reveal_test_and_report()
    assert accessed == []


def test_seed_6101_requires_seed_157_stage_gate(monkeypatch, tmp_path):
    destination = tmp_path / "study"
    (destination / "runtime").mkdir(parents=True)
    monkeypatch.setattr(study, "STUDY", destination)
    monkeypatch.setattr(study, "validate_seal", lambda: {})
    study.atomic_json(destination / "engineering_smoke.json", {"status": "PASSED", "test_truth_access_count": 0})
    with pytest.raises(FileNotFoundError):
        study.execute_seed(6101)
