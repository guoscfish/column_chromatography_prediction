from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.studies.run_t1_low_label_adaptation import (
    audit_manifests,
    fit_affine,
    fit_ridge_residual,
    generate_outer_seeds,
    make_manifests,
    method_label_audit,
    run_formal,
)
from src.qgeognn_al.engine import AdaptationTrainConfig, TrainConfig
from src.qgeognn_al.model import build_model, configure_trainable, install_monotonic_head


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


if __name__ == "__main__":
    unittest.main()
