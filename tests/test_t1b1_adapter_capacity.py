from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.studies.run_t1b1_adapter_capacity import run_formal
from src.qgeognn_al.artifacts import sha256_file
from src.qgeognn_al.engine import GraphAdapterTrainConfig
from src.qgeognn_al.model import (
    ResidualGraphAdapterHead, build_model, configure_graph_adapter_trainable,
    configure_trainable, graph_representation_dim, install_graph_residual_adapter,
    install_monotonic_head,
)
from src.qgeognn_al.t1_formal import execute_fit_plan, inspect_fit_runtime, write_fit_contract
from src.qgeognn_al.t1b1 import (
    ADAPTER_METHODS, BUDGETS, CAPACITY_METHODS, OUTER_SEEDS,
    build_fit_plan, capacity_summaries, completion_gate, expected_contract,
    paired_effects,
)


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/track_b_transfer/t1b1_adapter_capacity"
CONFIG = STUDY / "config.json"


def config(authorized: bool = False) -> dict:
    value = json.loads(CONFIG.read_text())
    value["formal_authorized"] = authorized
    return value


def fake_completed_fit(path: Path, contract: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "best.pt").write_bytes(b"adapter-checkpoint")
    (path / "history.csv").write_text("epoch,score\n1,1\n")
    digest = sha256_file(path / "best.pt")
    (path / "fit_result.json").write_text(json.dumps({
        "checkpoint_sha256": digest,
        "train_config_hash": contract["adaptation_train_config_hash"],
        "trainable_parameters": contract["expected_trainable_parameters"],
        "total_parameters": contract["expected_total_parameters"],
        "labeled_ids_hash": contract["labeled_ids_hash"],
        "validation_ids_hash": contract["validation_ids_hash"],
    }))
    write_fit_contract(path, contract, contract["adaptation_train_config_hash"])


class AdapterArchitectureTests(unittest.TestCase):
    def test_zero_up_initialization_preserves_prediction(self) -> None:
        torch.manual_seed(4)
        model = build_model(torch.device("cpu")); install_monotonic_head(model)
        head = model.graph_pred_linear
        h = torch.randn(7, graph_representation_dim(model))
        head.eval()
        with torch.no_grad():
            before = head(h)
        actual = install_graph_residual_adapter(model, 8)
        model.graph_pred_linear.eval()
        with torch.no_grad():
            after = model.graph_pred_linear(h)
        self.assertEqual(actual, 128)
        self.assertLessEqual(float(torch.max(torch.abs(before - after))), 1e-7)

    def test_width_parameter_counts_increase_and_match_theory(self) -> None:
        observed = []
        for width in (8, 16, 32):
            model = build_model(torch.device("cpu")); install_monotonic_head(model)
            install_graph_residual_adapter(model, width)
            trainable, total, adapter, head = configure_graph_adapter_trainable(model)
            self.assertEqual(adapter, 257 * width + 128)
            self.assertEqual(head, 774)
            self.assertEqual(trainable, adapter + head)
            self.assertEqual(total, 775476 + adapter)
            observed.append(trainable)
        self.assertEqual(observed, [2958, 5014, 9126])
        self.assertLess(max(observed), 93454)

    def test_only_adapter_and_head_are_trainable(self) -> None:
        model = build_model(torch.device("cpu")); install_monotonic_head(model)
        install_graph_residual_adapter(model, 16); configure_graph_adapter_trainable(model)
        names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertTrue(names)
        self.assertTrue(all(name.startswith("graph_pred_linear.adapter.") or name.startswith("graph_pred_linear.head.") for name in names))
        self.assertFalse(any(name.startswith("gnn_node.") for name in names))

    def test_historical_model_and_configure_trainable_are_unchanged(self) -> None:
        model = build_model(torch.device("cpu")); install_monotonic_head(model)
        self.assertNotIsInstance(model.graph_pred_linear, ResidualGraphAdapterHead)
        self.assertEqual(configure_trainable(model, "head_only"), (774, 775476))
        self.assertEqual(configure_trainable(model, "last1_head"), (93454, 775476))
        self.assertEqual(configure_trainable(model, "last2_head"), (186134, 775476))

    def test_adapter_training_contract_rejects_unregistered_width(self) -> None:
        for width in (8, 16, 32):
            GraphAdapterTrainConfig(width).validate()
        with self.assertRaisesRegex(ValueError, "bottleneck_width"):
            GraphAdapterTrainConfig(64).validate()


class AdapterFormalContractTests(unittest.TestCase):
    def test_config_rejects_unknown_scientific_status(self) -> None:
        value = config()
        value["scientific_status"] = "unknown"
        with self.assertRaisesRegex(RuntimeError, "scientific_status"):
            build_fit_plan(value)

    def test_plan_has_180_unique_keys(self) -> None:
        plan = build_fit_plan(config())
        self.assertEqual(len(plan), 180)
        self.assertEqual(len({item.run_key for item in plan}), 180)
        self.assertEqual(plan[0].run_key, "seed_769539383/budget_30/graph_adapter_r8/member_42")

    def test_formal_refuses_without_authorization(self) -> None:
        value = config(False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "formal_authorized=false"):
                run_formal(path)

    def test_authorized_synthetic_entry_receives_180_fits(self) -> None:
        value = config(True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"; path.write_text(json.dumps(value))
            result = run_formal(path, lambda received, plan: {"authorized": received["formal_authorized"], "fits": len(plan)})
        self.assertEqual(result, {"authorized": True, "fits": 180})

    def test_resume_reuses_matching_adapter_fit(self) -> None:
        value = config(True); spec = build_fit_plan(value)[0]
        contract = expected_contract(spec, value, ["train"], ["valid"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fake_completed_fit(root / spec.run_key, contract)
            calls = []
            audit, _ = execute_fit_plan([spec], root, lambda _: contract, lambda *_: calls.append(1))
        self.assertEqual(calls, [])
        self.assertEqual(audit["reused"], 1)

    def test_stale_adapter_config_cannot_reuse_checkpoint(self) -> None:
        value = config(True); spec = build_fit_plan(value)[0]
        contract = expected_contract(spec, value, ["train"], ["valid"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory); fake_completed_fit(path, contract)
            changed = dict(contract); changed["adapter_config_hash"] = "changed"
            state = inspect_fit_runtime(path, changed)
        self.assertEqual(state["status"], "stale")
        self.assertIn("adapter_config_hash", state["mismatch"])

    def test_paired_delta_sign_and_stability_gate(self) -> None:
        rows = []
        adapter_deltas = {
            "graph_adapter_r8": [-0.1, -0.2, -0.1, -0.2, 0.1],
            "graph_adapter_r16": [-0.1, -0.2, -0.1, 0.1, 0.2],
            "graph_adapter_r32": [0.1] * 5,
        }
        for index, seed in enumerate(OUTER_SEEDS):
            rows.append({"outer_seed": seed, "method": "target_head_only", "mean_NRMSE_over_budget_interval": 1.0})
            for method, values in adapter_deltas.items():
                rows.append({"outer_seed": seed, "method": method, "mean_NRMSE_over_budget_interval": 1.0 + values[index]})
        details, summary = paired_effects(pd.DataFrame(rows))
        self.assertTrue(details.loc[details.candidate.eq("graph_adapter_r8") & details.outer_seed.eq(OUTER_SEEDS[0]), "candidate_better"].iloc[0])
        self.assertTrue(summary.loc[summary.candidate.eq("graph_adapter_r8"), "pass"].iloc[0])
        self.assertFalse(summary.loc[summary.candidate.eq("graph_adapter_r16"), "pass"].iloc[0])

    def test_incomplete_result_forbids_decision(self) -> None:
        metrics = pd.DataFrame([{"outer_seed": 1, "budget": 30, "method": "target_head_only"}])
        gate = completion_gate(metrics, {"completed": 179, "failed": 0, "missing": 1})
        self.assertFalse(gate["pass"])
        self.assertFalse(gate["final_scientific_decision_allowed"])

    def test_capacity_curve_summary(self) -> None:
        rows = []
        for seed_index, seed in enumerate(OUTER_SEEDS):
            for budget in BUDGETS:
                for method_index, method in enumerate(CAPACITY_METHODS):
                    rows.append({"outer_seed": seed, "budget": budget, "method": method, "combined_NRMSE": 1.0 + method_index * .1 + seed_index * .01})
        parameters = {method: index + 1 for index, method in enumerate(CAPACITY_METHODS)}
        curve, aulc = capacity_summaries(pd.DataFrame(rows), parameters)
        self.assertEqual(len(curve), 24)
        self.assertEqual(len(aulc), 6)
        self.assertEqual(set(curve.columns), {"method", "trainable_parameters", "budget", "mean_NRMSE", "median_NRMSE", "std_NRMSE", "best_by_seed_count"})


class AdapterSmokeArtifactTests(unittest.TestCase):
    def test_smoke_contract_artifacts(self) -> None:
        audit = json.loads((STUDY / "engineering_smoke_audit.json").read_text())
        self.assertTrue(audit["all_checks_pass"])
        self.assertFalse(audit["test_truth_read"])
        self.assertTrue(audit["checks"]["same_gradient_ids"])
        self.assertTrue(audit["checks"]["same_validation_ids"])
        self.assertTrue(audit["checks"]["same_evaluation_ids"])
        self.assertTrue(audit["checks"]["no_test_truth"])
        self.assertEqual(len(pd.read_csv(STUDY / "smoke_fit_audit.csv")), 9)


if __name__ == "__main__":
    unittest.main()
