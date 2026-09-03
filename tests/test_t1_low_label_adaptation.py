from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.studies.run_t1_low_label_adaptation import (
    _write_formal_analysis,
    audit_manifests,
    fit_affine,
    fit_ridge_residual,
    generate_outer_seeds,
    make_manifests,
    method_label_audit,
    run_formal,
)
from src.qgeognn_al.engine import AdaptationTrainConfig, SourceFreeTrainConfig, TrainConfig
from src.qgeognn_al.model import build_model, configure_trainable, install_monotonic_head
from src.qgeognn_al.t1_formal import (
    build_formal_fit_plan,
    capacity_crossover_summary,
    completion_gate,
    compute_aulc,
    execute_fit_plan,
    expected_fit_contract,
    inspect_fit_runtime,
    paired_aulc_effects,
    stability_gate,
    summarize_methods_by_budget,
    write_fit_contract,
)


ROOT = Path(__file__).resolve().parents[1]
T1_CONFIG = ROOT / "studies/track_b_transfer/t1_low_label_adaptation/config.json"


def authorized_config() -> dict:
    config = json.loads(T1_CONFIG.read_text())
    config["formal_authorized"] = True
    return config


def write_fake_completed_fit(fit_dir: Path, contract: dict, train_config_hash: str = "fake-train-config") -> None:
    fit_dir.mkdir(parents=True, exist_ok=True)
    (fit_dir / "best.pt").write_bytes(b"fake-checkpoint")
    (fit_dir / "history.csv").write_text("epoch,score\n1,1.0\n")
    from src.qgeognn_al.artifacts import sha256_file
    (fit_dir / "fit_result.json").write_text(json.dumps({
        "checkpoint_sha256": sha256_file(fit_dir / "best.pt"),
        "train_config_hash": train_config_hash,
        "trainable_parameters": contract["expected_trainable_parameters"],
        "total_parameters": contract["expected_total_parameters"],
        "labeled_ids_hash": contract["labeled_ids_hash"],
        "validation_ids_hash": contract["validation_ids_hash"],
    }))
    write_fit_contract(fit_dir, contract, train_config_hash)


class T1AdaptationContractTests(unittest.TestCase):
    def test_historical_train_config_still_rejects_other_transfer_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen Gate 0"):
            TrainConfig(transfer_mode="head_only").validate_frozen_predictor()

    def test_adaptation_config_accepts_preregistered_modes_and_rejects_unknown(self) -> None:
        for mode in ("head_only", "last1_head", "last2_head"):
            AdaptationTrainConfig(transfer_mode=mode).validate()
        with self.assertRaisesRegex(ValueError, "Unknown target adaptation mode"):
            AdaptationTrainConfig(transfer_mode="paper_style").validate()

    def test_configure_trainable_scopes_and_counts(self) -> None:
        model = build_model(torch.device("cpu"))
        install_monotonic_head(model)
        expected = {"head_only": 774, "last1_head": 93454, "last2_head": 186134}
        for mode, count in expected.items():
            trainable, total = configure_trainable(model, mode)
            self.assertEqual(trainable, count)
            self.assertEqual(total, 775476)
            names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
            self.assertTrue(all(name.startswith("graph_pred_linear") for name in names) if mode == "head_only" else True)

    def test_source_free_contract_remains_frozen(self) -> None:
        config = SourceFreeTrainConfig()
        config.validate_frozen_predictor()
        with self.assertRaisesRegex(ValueError, "E2 contract"):
            SourceFreeTrainConfig(transfer_mode="head_only").validate_frozen_predictor()


class T1ScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.identities = pd.DataFrame({
            "sample_id": [f"id-{index}" for index in range(120)],
            "canonical_index": np.arange(120),
            "canonical_smiles": [f"mol-{index // 3}" for index in range(120)],
        })
        self.seeds = generate_outer_seeds(20260902)

    def test_outer_seeds_are_frozen_and_distinct(self) -> None:
        self.assertEqual(self.seeds, [769539383, 1425370602, 536279090, 2767143051, 1362771960])
        self.assertEqual(len(set(self.seeds)), 5)

    def test_nested_schedules_are_deterministic_and_shared(self) -> None:
        first = make_manifests(self.identities, self.seeds, [30, 50, 70, 100], 8, 0.1)
        second = make_manifests(self.identities, self.seeds, [30, 50, 70, 100], 8, 0.1)
        pd.testing.assert_frame_equal(first[0], second[0])
        pd.testing.assert_frame_equal(first[1], second[1])
        audit = audit_manifests(first[0], first[1], [30, 50, 70, 100], 8)
        self.assertTrue(audit["nested_budget_pass"])
        self.assertTrue(audit["role_overlap_pass"])
        seed_rows = first[1].loc[first[1].outer_seed.eq(self.seeds[0])]
        for budget in [30, 50, 70, 100]:
            roles = seed_rows.loc[seed_rows.budget.eq(budget)].role.value_counts()
            self.assertEqual(roles["gradient_train"], budget - 8)
            self.assertEqual(roles["validation"], 8)

    def test_partition_builder_refuses_target_labels(self) -> None:
        labeled = self.identities.assign(V1_ml=1.0, V2_ml=2.0)
        with self.assertRaisesRegex(ValueError, "identity columns only"):
            make_manifests(labeled, [1], [30], 8, 0.1)

    def test_methods_receive_the_same_label_and_evaluation_ids(self) -> None:
        audit = method_label_audit(
            ["zero_shot", "affine", "target_head_only"],
            ["train-b", "train-a"], ["valid"], ["eval-b", "eval-a"],
        )
        adapted = audit.loc[~audit.method.eq("zero_shot")]
        self.assertEqual(adapted.gradient_train_ids_hash.nunique(), 1)
        self.assertEqual(audit.validation_ids_hash.nunique(), 1)
        self.assertEqual(audit.evaluation_ids_hash.nunique(), 1)
        self.assertEqual(int(audit.loc[audit.method.eq("zero_shot"), "gradient_train_rows"].iloc[0]), 0)


class T1SimpleMethodTests(unittest.TestCase):
    def test_affine_and_ridge_use_only_explicit_training_truth(self) -> None:
        source = np.column_stack([np.linspace(1, 5, 12), np.linspace(2, 10, 12)])
        truth = 1.5 * source + np.array([2.0, -1.0])
        predict_source = source[:3] + 0.25
        affine = fit_affine(truth, source, predict_source)
        conditions = np.column_stack([np.arange(12), np.ones(12)])
        ridge, alpha, policy = fit_ridge_residual(
            truth, source, conditions, np.array([f"g{i // 2}" for i in range(12)]),
            predict_source, conditions[:3], [0.01, 0.1, 1.0],
        )
        self.assertTrue(np.isfinite(affine).all())
        self.assertTrue(np.isfinite(ridge).all())
        self.assertIn(alpha, [0.01, 0.1, 1.0])
        self.assertIn("gradient_train_only", policy)

    def test_ridge_has_deterministic_tiny_set_fallback(self) -> None:
        source = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        truth = source + 1.0
        conditions = np.ones((3, 2))
        prediction, alpha, policy = fit_ridge_residual(
            truth, source, conditions, np.array(["same"] * 3), source[:1], conditions[:1],
            [0.01, 0.1, 1.0, 10.0, 100.0],
        )
        self.assertEqual(alpha, 1.0)
        self.assertEqual(policy, "deterministic_alpha_1_insufficient_groupkfold")
        self.assertTrue(np.isfinite(prediction).all())

    def test_formal_run_is_refused_when_not_authorized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"formal_authorized": False}))
            with self.assertRaisesRegex(RuntimeError, "formal_authorized=false"):
                run_formal(config)


class T1FormalPlanAndResumeTests(unittest.TestCase):
    def test_authorized_synthetic_path_enters_without_real_training(self) -> None:
        config = authorized_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            observed = {}
            def fake_executor(received, plan):
                observed["authorized"] = received["formal_authorized"]
                observed["fits"] = len(plan)
                return {"status": "synthetic"}
            result = run_formal(path, authorized_executor=fake_executor)
        self.assertEqual(result, {"status": "synthetic"})
        self.assertTrue(observed["authorized"])
        self.assertEqual(observed["fits"], 180)

    def test_expected_fit_plan_is_180_and_run_keys_are_unique(self) -> None:
        plan = build_formal_fit_plan(authorized_config())
        self.assertEqual(len(plan), 5 * 4 * 3 * 3)
        self.assertEqual(len({item.run_key for item in plan}), 180)
        self.assertEqual(plan[0].run_key, "seed_769539383/budget_30/target_head_only/member_42")

    def test_resume_reuses_complete_fit_without_executor_call(self) -> None:
        config = authorized_config(); spec = build_formal_fit_plan(config)[0]
        train_ids, validation_ids = ["train"], ["valid"]
        contract = expected_fit_contract(spec, config, train_ids, validation_ids)
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            write_fake_completed_fit(runtime / spec.run_key, contract)
            calls = []
            audit, rows = execute_fit_plan(
                [spec], runtime, lambda _: contract,
                lambda *_: calls.append("called"), max_same_config_retry=1,
            )
        self.assertEqual(calls, [])
        self.assertEqual(audit["reused"], 1)
        self.assertEqual(audit["completed"], 1)
        self.assertEqual(rows[0]["action"], "reused")

    def test_config_hash_mismatch_is_stale_and_not_reused(self) -> None:
        config = authorized_config(); spec = build_formal_fit_plan(config)[0]
        contract = expected_fit_contract(spec, config, ["train"], ["valid"])
        with tempfile.TemporaryDirectory() as directory:
            fit_dir = Path(directory) / spec.run_key
            write_fake_completed_fit(fit_dir, contract)
            changed = dict(contract); changed["formal_config_hash"] = "changed"
            state = inspect_fit_runtime(fit_dir, changed)
        self.assertEqual(state["status"], "stale")
        self.assertIn("formal_config_hash", state["mismatch"])

    def test_partial_runtime_is_quarantined_and_only_that_fit_reruns(self) -> None:
        config = authorized_config(); spec = build_formal_fit_plan(config)[0]
        contract = expected_fit_contract(spec, config, ["train"], ["valid"])
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory); fit_dir = runtime / spec.run_key
            fit_dir.mkdir(parents=True); (fit_dir / "history.csv").write_text("partial")
            calls = []
            def executor(_, destination, expected):
                calls.append(str(destination)); write_fake_completed_fit(destination, expected)
            audit, rows = execute_fit_plan([spec], runtime, lambda _: contract, executor)
            quarantined = list((runtime / "quarantine").iterdir())
        self.assertEqual(len(calls), 1)
        self.assertEqual(audit["partial"], 1)
        self.assertEqual(audit["rerun"], 1)
        self.assertEqual(audit["completed"], 1)
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(rows[0]["final_status"], "complete")

    def test_failed_fit_is_accounted_for_after_one_identical_retry(self) -> None:
        config = authorized_config(); spec = build_formal_fit_plan(config)[0]
        contract = expected_fit_contract(spec, config, ["train"], ["valid"])
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            def failing(*_):
                calls.append(1); raise RuntimeError("technical")
            audit, rows = execute_fit_plan([spec], Path(directory), lambda _: contract, failing, 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(audit["failed"], 1)
        self.assertTrue(audit["all_expected_fits_accounted_for"])
        self.assertEqual(rows[0]["final_status"], "failed")


class T1FormalAnalysisTests(unittest.TestCase):
    @staticmethod
    def complete_synthetic_metrics(config: dict) -> pd.DataFrame:
        rows = []
        offsets = {
            "zero_shot": .30, "affine": .12, "condition_ridge_residual": .14,
            "target_head_only": .08, "last1_head": .05, "current_last2_head": .06,
        }
        for seed_index, seed in enumerate(config["outer_seeds"]):
            for budget in config["target_label_budgets"]:
                for method in config["primary_methods"]:
                    rows.append({
                        "outer_seed": seed, "budget": budget, "method": method,
                        "combined_NRMSE": .7 - budget / 500 + offsets[method] + seed_index / 1000,
                    })
        return pd.DataFrame(rows)

    def test_aulc_and_normalized_aulc_use_trapezoidal_interval(self) -> None:
        metrics = pd.DataFrame({
            "outer_seed": [1] * 4, "budget": [30, 50, 70, 100],
            "method": ["m"] * 4, "combined_NRMSE": [1.0, 0.8, 0.6, 0.4],
        })
        result = compute_aulc(metrics, [30, 50, 70, 100]).iloc[0]
        expected = np.trapezoid([1.0, 0.8, 0.6, 0.4], [30, 50, 70, 100])
        self.assertAlmostEqual(result.AULC_30_100, expected)
        self.assertAlmostEqual(result.mean_NRMSE_over_budget_interval, expected / 70)

    def test_paired_delta_sign_candidate_minus_last2(self) -> None:
        aulc = pd.DataFrame([
            {"outer_seed": seed, "method": method, "mean_NRMSE_over_budget_interval": value}
            for seed in range(5)
            for method, value in [("candidate", 0.4 + seed * .01), ("current_last2_head", 0.5 + seed * .01)]
        ])
        differences, summary = paired_aulc_effects(aulc)
        self.assertTrue((differences.delta_normalized_AULC_candidate_minus_reference < 0).all())
        self.assertTrue(differences.candidate_better.all())
        self.assertTrue(bool(summary.stable_low_label_improvement.iloc[0]))

    def test_stability_gate_requires_all_three_conditions(self) -> None:
        passed = stability_gate([-0.2, -0.1, -0.05, -0.01, 0.02])
        failed_three_wins = stability_gate([-0.3, -0.2, -0.1, 0.01, 0.02])
        self.assertTrue(passed["pass"])
        self.assertEqual(passed["win_count"], 4)
        self.assertFalse(failed_three_wins["pass"])
        self.assertEqual(failed_three_wins["win_count"], 3)

    def test_capacity_summary_detects_descriptive_budget_change(self) -> None:
        rows = []
        methods = ["target_head_only", "last1_head", "current_last2_head"]
        for budget in [30, 50, 70, 100]:
            best = "target_head_only" if budget == 30 else "current_last2_head"
            for seed in range(5):
                for method in methods:
                    rows.append({"outer_seed": seed, "budget": budget, "method": method, "combined_NRMSE": 0.4 if method == best else 0.6})
        capacity = summarize_methods_by_budget(pd.DataFrame(rows), methods)
        summary = capacity_crossover_summary(capacity)
        self.assertEqual(summary["best_mean_method_per_budget"]["30"], "target_head_only")
        self.assertEqual(summary["best_mean_method_per_budget"]["100"], "current_last2_head")
        self.assertTrue(summary["descriptive_capacity_crossover"])

    def test_partial_results_forbid_final_scientific_decision(self) -> None:
        config = authorized_config()
        partial = pd.DataFrame([{"outer_seed": config["outer_seeds"][0], "budget": 30, "method": "zero_shot"}])
        gate = completion_gate(partial, {
            "completed": 179, "failed": 0, "missing": 1,
            "all_expected_fits_accounted_for": False,
        }, config)
        self.assertFalse(gate["pass"])
        self.assertFalse(gate["final_scientific_decision_allowed"])

    def test_complete_synthetic_analysis_writes_all_primary_outputs(self) -> None:
        config = authorized_config(); metrics = self.complete_synthetic_metrics(config)
        convergence = pd.DataFrame([
            {
                "outer_seed": seed, "budget": budget, "method": method,
                "source_member": member, "best_epoch": 10, "epochs_run": 120,
                "hit_max_epoch": False, "early_stopped": True,
                "normalized_valid_score": .5,
            }
            for seed in config["outer_seeds"] for budget in config["target_label_budgets"]
            for method in config["neural_modes"] for member in config["source_members"]
        ])
        resume = {
            "completed": 180, "reused": 0, "rerun": 0, "failed": 0,
            "missing": 0, "all_expected_fits_accounted_for": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            result = _write_formal_analysis(
                destination, config, metrics, convergence, pd.DataFrame(), resume, [],
                [
                    {"outer_seed": seed, "budget": budget, "method": method}
                    for seed in config["outer_seeds"] for budget in config["target_label_budgets"]
                    for method in config["primary_methods"]
                ],
            )
            expected = {
                "per_context_metrics.csv", "learning_curves.csv", "aulc_by_seed.csv",
                "paired_aulc_effects.csv", "capacity_by_budget.csv",
                "capacity_crossover_summary.json", "simple_vs_neural_by_budget.csv",
                "convergence_audit.csv", "convergence_summary.csv",
                "formal_run_audit.json", "resume_audit.json", "decision.json",
            }
            self.assertTrue(expected.issubset({path.name for path in destination.iterdir()}))
            self.assertTrue(json.loads((destination / "formal_run_audit.json").read_text())["pass"])
        self.assertTrue(result["formal_run_complete"])


if __name__ == "__main__":
    unittest.main()
