"""Synthetic, label-free tests; no real LLM, training, gradient, or label access."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from numerical_critic import FrozenInputError, critique_frozen_plan, payload_hash


def mock_snapshot(*, pool_count=40, constant_features=False):
    """All values are synthetic; molecule strings are identities, not SMILES."""
    L = [f"L{i:02}" for i in range(8)]
    U = [f"U{i:02}" for i in range(pool_count)]
    ids = L + U
    state = dict(seed=157, round=1, selection_at="2026-10-30T10:00:00+08:00",
                 checkpoint_sha256="a" * 64, pool_partition="train_pool", synthetic=True,
                 L_t_ids=L, U_t_ids=U)

    def block(name, payload):
        return {"source": {"source_id": f"mock:{name}", "payload_sha256": payload_hash(payload),
                           "state_sha256": payload_hash(state), "checkpoint_sha256": state["checkpoint_sha256"],
                           "frozen_at": "2026-10-30T09:59:00+08:00"}, "payload": deepcopy(payload)}

    rows = []
    for j, identifier in enumerate(ids):
        rows.append({"id": identifier, "molecule": f"molecule-{j}",
                     "conditions": {"PE/EA": "4/1", "Density g/ml": .1, "V/ul": 50.,
                                    "loading solvent": "EA", "Volume of loading solvent/ul": 500.}})
    rows[8]["molecule"] = rows[0]["molecule"]  # U00 repeats L00 conditions.
    rows[9]["molecule"] = rows[0]["molecule"]  # U01 is a strict new loading contrast.
    rows[9]["conditions"]["V/ul"] = 100.
    rows[10]["molecule"] = rows[0]["molecule"]
    rows[10]["conditions"]["V/ul"] = 150.
    rows[11]["molecule"] = rows[0]["molecule"]
    rows[11]["conditions"]["V/ul"] = 50.
    rows[11]["conditions"]["PE/EA"] = "3/1"
    rng = np.random.default_rng(20261030)
    features = rng.normal(size=(len(ids), 5))
    features[9] = features[8]  # Geometry does not resolve this loading contrast.
    if constant_features:
        features[:] = 0.
    pred = np.arange(len(ids) * 2, dtype=float).reshape(len(ids), 2) / 10
    return {"schema_version": "numerical-critic-v1", "state": block("state", state),
            "metadata": block("metadata", {"ids": ids, "rows": rows}),
            "predictions": block("predictions", {"ids": ids, "q50": pred.tolist(),
                                                    "q10": (pred - 1).tolist(), "q90": (pred + 1).tolist()}),
            "gradient_spaces": [block("raw", {"name": "raw", "kind": "mock_frozen_gradient", "ids": ids,
                                                "features": features.tolist(), "preprocessing": "synthetic identity"})],
            "plan": block("plan", {"ids": U[:32]}),
            "feedback": block("feedback", [{"evidence_id": "mock-evidence-0", "id": L[0],
                                               "observed_at": "2026-10-30T09:00:00+08:00", "acquisition_round": 0,
                                               "error_v1_ml": -2., "error_v2_ml": 3.}])}


def rehash(block):
    block["source"]["payload_sha256"] = payload_hash(block["payload"])


class NumericalCriticTests(unittest.TestCase):
    def test_valid_plan_preserved_without_input_mutation(self):
        snapshot = mock_snapshot()
        before = deepcopy(snapshot)
        result = critique_frozen_plan(snapshot)
        self.assertEqual(snapshot, before)
        self.assertEqual(result["selected_ids"], snapshot["plan"]["payload"]["ids"])
        self.assertEqual(len(result["candidate_diagnostics"]), 32)
        self.assertEqual(len(result["candidate_pair_relations"]), 496)
        self.assertTrue(result["synthetic"])
        json.dumps(result, allow_nan=False)

    def test_strict_controls_and_replicates_separate_from_geometry(self):
        result = critique_frozen_plan(mock_snapshot())
        points = {p["id"]: p for p in result["candidate_diagnostics"]}
        self.assertEqual(points["U00"]["exact_condition_repeat_prior_ids"], ["L00"])
        self.assertEqual(points["U00"]["new_condition_curve_extension_prior_axes"], [])
        self.assertEqual(points["U01"]["new_condition_curve_extension_prior_axes"], ["V/ul"])
        self.assertEqual(points["U01"]["geometry"]["raw"]["exact_feature_peer_ids"], ["U00"])
        review = next(r for r in result["review_items"] if r["id"] == "U01")
        self.assertTrue(review["strict_control_relations"])
        self.assertEqual(review["action"], "human_or_future_scientific_review_only")
        # U01 versus U03 changes two conditions: never a strict single-axis pair.
        self.assertFalse(any({r["id"], r["peer_id"]} == {"U01", "U03"} for r in result["scientific_controls"]))

    def test_metric_parity_with_existing_free_audit(self):
        legacy_path = HERE.parent / "free_lcmd_mechanism_diagnostics_20261029" / "geometry_analysis.py"
        spec = importlib.util.spec_from_file_location("legacy_frozen_geometry", legacy_path)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        snapshot = mock_snapshot()
        features = np.array(snapshot["gradient_spaces"][0]["payload"]["features"])
        expected, _ = legacy.geometry(features, 8, list(range(32)))
        actual = critique_frozen_plan(snapshot)["geometry"]["raw"]["metrics"]
        for key, value in actual.items():
            with self.subTest(metric=key):
                if value is None:
                    self.assertIsNone(expected[key])
                else:
                    self.assertAlmostEqual(value, expected[key], places=11)

    def test_prefix_marginal_gains_add_up_on_common_remaining_pool(self):
        result = critique_frozen_plan(mock_snapshot())
        points = [p["geometry"]["raw"] for p in result["candidate_diagnostics"]]
        metrics = result["geometry"]["raw"]["metrics"]
        full = sum(p["prefix_marginal_mean_distance_drop_full_U"] for p in points)
        remaining = sum(p["prefix_marginal_common_remaining_U_drop_to_bank_RMS"] for p in points)
        self.assertAlmostEqual(full / metrics["pool_preselection_coverage_distance_mean"],
                               metrics["coverage_distance_relative_drop"])
        self.assertAlmostEqual(remaining, metrics["remaining_U_coverage_gain_to_bank_RMS"])
        self.assertTrue(all(p["prefix_marginal_mean_distance_drop_full_U"] >= 0 for p in points))

    def test_no_remaining_pool_and_zero_bank_are_explicit_null(self):
        result = critique_frozen_plan(mock_snapshot(pool_count=32, constant_features=True))
        metrics = result["geometry"]["raw"]["metrics"]
        self.assertIsNone(metrics["remaining_U_coverage_distance_relative_drop"])
        self.assertIsNone(metrics["remaining_U_coverage_gain_to_bank_RMS"])
        self.assertIsNone(metrics["coverage_distance_relative_drop"])
        self.assertEqual(metrics["centered_batch_effective_rank"], 0.)
        json.dumps(result, allow_nan=False)

    def test_empty_feedback_and_optional_quantiles(self):
        snapshot = mock_snapshot()
        snapshot["feedback"]["payload"] = []
        rehash(snapshot["feedback"])
        del snapshot["predictions"]["payload"]["q10"]
        del snapshot["predictions"]["payload"]["q90"]
        rehash(snapshot["predictions"])
        result = critique_frozen_plan(snapshot)
        self.assertIsNone(result["preselection_feedback_context"]["high8_same_molecule_enrichment"])
        self.assertIsNone(result["candidate_diagnostics"][0]["q90_minus_q10_v1_v2_ml"])

    def test_quantile_crossing_is_model_diagnostic_not_plan_rejection(self):
        snapshot = mock_snapshot()
        snapshot["predictions"]["payload"]["q10"][8][0] = 1e6
        rehash(snapshot["predictions"])
        result = critique_frozen_plan(snapshot)
        self.assertEqual(result["selected_ids"], snapshot["plan"]["payload"]["ids"])
        self.assertEqual(result["prediction_diagnostics"]["pool_quantile_crossing_count_v1_v2"], [1, 0])
        self.assertEqual(result["prediction_diagnostics"]["pool_negative_quantile_width_count_v1_v2"], [1, 0])
        self.assertEqual(result["candidate_diagnostics"][0]["quantile_crossing_v1_v2"], [True, False])
        self.assertLess(result["candidate_diagnostics"][0]["q90_minus_q10_v1_v2_ml"][0], 0.)

    def test_rejects_corrupt_hash_checkpoint_time_and_order(self):
        mutations = {
            "hash": lambda s: s["plan"]["source"].update(payload_sha256="0" * 64),
            "checkpoint": lambda s: s["predictions"]["source"].update(checkpoint_sha256="b" * 64),
            "state": lambda s: s["metadata"]["source"].update(state_sha256="c" * 64),
            "future_freeze": lambda s: s["gradient_spaces"][0]["source"].update(frozen_at="2026-10-30T10:00:01+08:00"),
            "naive_time": lambda s: s["plan"]["source"].update(frozen_at="2026-10-30T09:59:00"),
        }
        for name, mutate in mutations.items():
            with self.subTest(case=name):
                snapshot = mock_snapshot()
                mutate(snapshot)
                with self.assertRaises(FrozenInputError):
                    critique_frozen_plan(snapshot)
        snapshot = mock_snapshot()
        snapshot["predictions"]["payload"]["ids"].reverse()
        rehash(snapshot["predictions"])
        with self.assertRaisesRegex(FrozenInputError, "ordered IDs"):
            critique_frozen_plan(snapshot)

    def test_rejects_illegal_plan_including_count_duplicates_labeled_unknown(self):
        for replacement in ("L00", "not-in-pool", "U01", None):
            with self.subTest(replacement=replacement):
                snapshot = mock_snapshot()
                plan = snapshot["plan"]["payload"]["ids"]
                if replacement is None:
                    plan.pop()
                else:
                    plan[0] = replacement
                rehash(snapshot["plan"])
                with self.assertRaises(FrozenInputError):
                    critique_frozen_plan(snapshot)

    def test_rejects_current_unobserved_and_late_feedback(self):
        for change in ({"id": "U00"}, {"acquisition_round": 1},
                       {"observed_at": "2026-10-30T10:00:00+08:00"},
                       {"observed_at": "2026-10-30T09:59:30+08:00"}):
            with self.subTest(change=change):
                snapshot = mock_snapshot()
                snapshot["feedback"]["payload"][0].update(change)
                rehash(snapshot["feedback"])
                with self.assertRaises(FrozenInputError):
                    critique_frozen_plan(snapshot)

    def test_rejects_label_fields_and_nonfinite_or_mismatched_arrays(self):
        snapshot = mock_snapshot()
        snapshot["metadata"]["payload"]["rows"][8]["unmeasured_V1_truth"] = 3.
        rehash(snapshot["metadata"])
        with self.assertRaisesRegex(FrozenInputError, "unknown fields"):
            critique_frozen_plan(snapshot)
        for change in ("short_features", "short_prediction", "numeric_string"):
            with self.subTest(change=change):
                snapshot = mock_snapshot()
                if change == "short_features":
                    block = snapshot["gradient_spaces"][0]
                    block["payload"]["features"][0].pop()
                else:
                    block = snapshot["predictions"]
                    if change == "short_prediction":
                        block["payload"]["q50"].pop()
                    else:
                        block["payload"]["q50"][0][0] = "1.0"
                rehash(block)
                with self.assertRaises(FrozenInputError):
                    critique_frozen_plan(snapshot)
        snapshot = mock_snapshot()
        snapshot["gradient_spaces"][0]["payload"]["features"][0][0] = float("nan")
        with self.assertRaises(FrozenInputError):
            critique_frozen_plan(snapshot)


if __name__ == "__main__":
    unittest.main(verbosity=2)
