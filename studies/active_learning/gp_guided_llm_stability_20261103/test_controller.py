"""Synthetic, label-free tests for numerical diagnostics and frozen arm adapters."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from arm_adapter import (MockTransport, add_mock_controller_feedback, assert_frozen_input_invariants,
                         build_arm_inputs, freeze_shared_inputs)
from controller import conditional_information_gain, diagnose_batch, run_mock_controller, stable_hash


ROOT = Path(__file__).resolve().parent
SYNTHETIC_REFERENCE_SCOPE = {"batch_experiment_domain_id": "synthetic_domain",
    "reference_experiment_domain_id": "synthetic_domain", "batch_design_structure_sha256": "a" * 64,
    "reference_design_structure_sha256": "a" * 64, "freeze_manifest_sha256": "b" * 64}


def config(**thresholds):
    result = json.loads((ROOT / "configs/controller_development.json").read_text())
    result["thresholds"].update(thresholds)
    result["checkpoint_sha256"] = "c" * 64
    return result


class InformationGainTests(unittest.TestCase):
    def test_diagonal_analytic_and_float64(self):
        result = conditional_information_gain(np.diag([1, 3]).astype(np.float32), [2, 4], [0, 1])
        self.assertAlmostEqual(result["value_nats"], 0.5 * np.log(1.5 * 1.75))
        self.assertEqual(result["dtype"], "float64")
        self.assertEqual(result["cholesky_jitter"], 0)

    def test_joint_correlated_batch_is_not_sum_of_marginal_ig(self):
        result = conditional_information_gain([[1, 1], [1, 1]], 1, [0, 1])
        self.assertAlmostEqual(result["value_nats"], 0.5 * np.log(3))
        self.assertLess(result["value_nats"], result["independent_marginal_upper_bound_nats"])

    def test_heteroscedastic_noise_joint_formula(self):
        cov = np.array([[2., .5], [.5, 3.]])
        expected = 0.5 * np.linalg.slogdet(np.eye(2) + cov / np.array([1., 2.])[:, None])[1]
        self.assertAlmostEqual(conditional_information_gain(cov, [1, 2], [0, 1])["value_nats"], expected)

    def test_invalid_inputs_fail(self):
        cases = [([[1, 2], [2, 1]], 1, [0, 1]), ([[1, .2], [.1, 1]], 1, [0]),
                 (np.eye(2), 0, [0]), (np.eye(2), 1, [0, 0]), (np.eye(2), 1, [True]),
                 ([[float("nan")]], 1, [0]), (np.eye(2), [1], [0]), (np.eye(2), 1, [2])]
        for covariance, noise, batch in cases:
            with self.subTest(covariance=covariance, noise=noise, batch=batch):
                with self.assertRaises(ValueError):
                    conditional_information_gain(covariance, noise, batch)

    def test_conditional_covariance_reduces_information(self):
        prior = np.eye(3)
        posterior = prior * .1
        self.assertLess(conditional_information_gain(posterior, 1, [0, 1])["value_nats"],
                        conditional_information_gain(prior, 1, [0, 1])["value_nats"])


class ControllerDiagnosticTests(unittest.TestCase):
    def test_default_is_not_deployable_and_has_no_thresholds(self):
        result = diagnose_batch(np.eye(3), 1, [0, 1], config=config())
        self.assertEqual(result["decision_status"], "indeterminate")
        self.assertFalse(result["deployable"])
        self.assertFalse(result["triggered_constraints"])
        self.assertEqual(result["risk_blindspot"]["status"], "indeterminate")

    def test_all_batch_pairs_scanned_not_a_top_subset(self):
        result = diagnose_batch(np.ones((32, 32)), 1, list(range(32)), config=config(
            absolute_posterior_correlation=.99, maximum_unexempt_correlated_pairs=0))
        self.assertEqual(result["redundancy"]["scanned_pair_count"], 496)
        self.assertEqual(len(result["redundancy"]["unexempt_high_correlation_pairs"]), 496)
        self.assertTrue(result["triggered_constraints"])

    def test_reference_must_match_batch_size_and_joint_ig_ratio(self):
        cov = np.eye(4)
        cov[0, 1] = cov[1, 0] = 1
        result = diagnose_batch(cov, 1, [0, 1], config=config(minimum_reference_ig_ratio=.9),
                                reference_batch_indices=[2, 3], reference_scope=SYNTHETIC_REFERENCE_SCOPE)
        self.assertAlmostEqual(result["reference_ig_ratio"], np.log(3) / np.log(4))
        self.assertEqual(result["triggered_constraints"][0]["constraint"], "minimum_reference_ig_ratio")
        with self.assertRaises(ValueError):
            diagnose_batch(cov, 1, [0, 1], config=config(), reference_batch_indices=[2])

    def test_missing_or_zero_reference_is_indeterminate(self):
        result = diagnose_batch(np.zeros((3, 3)), 1, [0, 1], config=config(minimum_reference_ig_ratio=.5),
                                reference_batch_indices=[1, 2], reference_scope=SYNTHETIC_REFERENCE_SCOPE)
        self.assertIn("reference_information_gain_missing_or_zero", result["indeterminate_reasons"])

    def test_global_reference_cannot_force_local_design_to_change(self):
        scope = copy.deepcopy(SYNTHETIC_REFERENCE_SCOPE)
        scope["batch_experiment_domain_id"] = "local_condition_contrast"
        result = diagnose_batch(np.diag([.01, .01, 100., 100.]), 1, [0, 1],
            config=config(minimum_reference_ig_ratio=.9), reference_batch_indices=[2, 3], reference_scope=scope)
        self.assertFalse(result["triggered_constraints"])
        self.assertIn("ig_reference_experiment_domain_or_structure_mismatch", result["indeterminate_reasons"])
        self.assertLess(result["reference_ig_ratio"], .9)

    def test_no_reference_scope_means_no_IG_trigger(self):
        result = diagnose_batch(np.diag([.01, .01, 100., 100.]), 1, [0, 1],
            config=config(minimum_reference_ig_ratio=.9), reference_batch_indices=[2, 3])
        self.assertFalse(result["triggered_constraints"])
        self.assertIn("ig_reference_scope_not_frozen", result["indeterminate_reasons"])

    def test_exact_single_factor_has_only_structural_redundancy_exemption(self):
        cfg = config(absolute_posterior_correlation=.99, maximum_unexempt_correlated_pairs=0)
        cfg["condition_fields"] = ["loading", "solvent", "column"]
        metadata = [{"id": "a", "molecule_id": "m", "conditions": {"loading": 1, "solvent": .5, "column": 4}},
                    {"id": "b", "molecule_id": "m", "conditions": {"loading": 2, "solvent": .5, "column": 4}}]
        claim = {"candidate_ids": ["a", "b"], "hypothesis_id": "h", "varied_factor": "loading"}
        result = diagnose_batch(np.ones((2, 2)), 1, [0, 1], config=cfg, metadata=metadata,
                                matched_control_claims=[claim], approved_hypothesis_ids=["h"])
        self.assertEqual(len(result["redundancy"]["exempt_structural_control_pairs"]), 1)
        self.assertFalse(result["triggered_constraints"])
        metadata[1]["conditions"]["solvent"] = .6
        changed = diagnose_batch(np.ones((2, 2)), 1, [0, 1], config=cfg, metadata=metadata,
                                 matched_control_claims=[claim], approved_hypothesis_ids=["h"])
        self.assertEqual(changed["single_factor_structural_checks"][0]["status"], "rejected")
        self.assertTrue(changed["triggered_constraints"])

    def test_llm_rationale_and_unregistered_hypothesis_cannot_exempt(self):
        cfg = config(absolute_posterior_correlation=.99, maximum_unexempt_correlated_pairs=0)
        cfg["condition_fields"] = ["loading"]
        metadata = [{"id": "a", "molecule_id": "m", "conditions": {"loading": 1}},
                    {"id": "b", "molecule_id": "m", "conditions": {"loading": 2}}]
        result = diagnose_batch(np.ones((2, 2)), 1, [0, 1], config=cfg, metadata=metadata,
            matched_control_claims=[{"candidate_ids": ["a", "b"], "hypothesis_id": "fabricated",
                                     "varied_factor": "loading", "rationale": "This is scientifically useful."}])
        self.assertEqual(len(result["redundancy"]["exempt_structural_control_pairs"]), 0)

    def test_blindspot_requires_external_calibrated_same_checkpoint_evidence(self):
        evidence = {"calibration_status": "qualified_on_frozen_development", "checkpoint_sha256": "c" * 64,
                    "calibration_manifest_sha256": "d" * 64, "risk_scores": [1, 2, 3],
                    "blindspot_mask": [False, False, True]}
        result = diagnose_batch(np.eye(3), 1, [0, 1], config=config(minimum_blindspot_count=1), risk_evidence=evidence)
        self.assertEqual(result["triggered_constraints"][0]["constraint"], "minimum_blindspot_count")
        evidence["checkpoint_sha256"] = "other"
        mismatch = diagnose_batch(np.eye(3), 1, [0, 1], config=config(minimum_blindspot_count=1), risk_evidence=evidence)
        self.assertFalse(mismatch["triggered_constraints"])
        self.assertIn("blindspot_constraint_has_no_qualified_risk_evidence", mismatch["indeterminate_reasons"])

    def test_geometry_is_descriptive_and_has_no_global_minimum(self):
        result = diagnose_batch(np.eye(3), 1, [0, 1], config=config(), geometry={
            "candidate_features": [[0, 0], [1, 0], [10, 0]], "observed_features": [[0, 0]],
            "representation_id": "CW", "state_sha256": "state"})
        self.assertEqual(result["geometry"]["nearest_observed_distance"], [0, 1])
        self.assertIsNone(result["geometry"]["enforced_global_coverage_minimum"])


def frozen_inputs():
    return freeze_shared_inputs(checkpoint_sha256="c" * 64, labeled_ids=["o0"], candidate_ids=["a", "b"],
        scientific_memory={"hypotheses": ["h"]}, candidate_input=[{"id": "a"}, {"id": "b"}],
        candidate_metadata=[{"id": "a"}, {"id": "b"}], baseline_packet={"selection_count": 32},
        selector_protocol="frozen existing CW-Free protocol", llm_configuration={"model": "unchanged", "seed": None},
        tool_contract={"read_only": True, "schema_sha256": "s" * 64})


class FrozenInputTests(unittest.TestCase):
    def test_identical_baseline_and_B_C_information_with_C_only_feedback(self):
        inputs = build_arm_inputs(frozen_inputs(), {"posterior_variance": [1, 2]})
        self.assertEqual(inputs["B"], inputs["C"])
        updated = add_mock_controller_feedback(inputs, {"constraint": "synthetic_probe"})
        check = assert_frozen_input_invariants(updated)
        self.assertTrue(check["base_input_identical"])
        self.assertTrue(check["B_C_gp_information_identical"])
        self.assertEqual(check["C_only_feedback_count"], 1)
        self.assertEqual(inputs["C"]["controller_feedback"], [])

    def test_each_frozen_component_drift_is_rejected(self):
        inputs = build_arm_inputs(frozen_inputs(), {"variance": [1, 2]})
        for key in inputs["A"]["shared_base_payload"]["baseline_components"]:
            with self.subTest(component=key):
                drift = copy.deepcopy(inputs)
                drift["C"]["shared_base_payload"]["baseline_components"][key] = "changed"
                with self.assertRaises(ValueError):
                    assert_frozen_input_invariants(drift)

    def test_reforged_manifest_drift_and_gp_drift_are_rejected_across_arms(self):
        inputs = build_arm_inputs(frozen_inputs(), {"variance": [1, 2]})
        changed = inputs["C"]["shared_base_payload"]
        changed["baseline_components"]["scientific_memory"] = {"h": "changed"}
        changed["frozen_state_manifest"]["components_sha256"]["scientific_memory"] = stable_hash({"h": "changed"})
        with self.assertRaises(ValueError):
            assert_frozen_input_invariants(inputs)
        inputs = build_arm_inputs(frozen_inputs(), {"variance": [1, 2]})
        inputs["C"]["gp_information"]["variance"][0] = 3
        with self.assertRaises(ValueError):
            assert_frozen_input_invariants(inputs)

    def test_A_extra_gp_and_B_feedback_and_unregistered_differences_rejected(self):
        for arm, key, value in [("A", "gp_information", {"extra": 1}), ("B", "controller_feedback", ["x"]),
                                ("C", "unknown_hidden_prompt", "x")]:
            inputs = build_arm_inputs(frozen_inputs(), {})
            inputs[arm][key] = value
            with self.assertRaises(ValueError):
                assert_frozen_input_invariants(inputs)

    def test_mock_transport_has_no_network_fallback(self):
        response = {"choices": [{"id": "a"}]}
        transport = MockTransport([response])
        self.assertEqual(transport.respond({"input": 1}), response)
        self.assertEqual(len(transport.calls), 1)
        with self.assertRaises(RuntimeError):
            transport.respond({"input": 1})


class MockLoopTests(unittest.TestCase):
    def test_full_32_proposal_revision_and_complete_audit_trail(self):
        ids = [f"c{i}" for i in range(40)]
        covariance = np.eye(40)
        covariance[0, 1] = covariance[1, 0] = .999
        initial = {"choices": [{"id": value} for value in ids[:32]]}
        valid_calls = []

        def shared_validator(proposal):
            # This fixture stands in for the same existing production hook in A/B/C.
            valid_calls.append(copy.deepcopy(proposal))
            self.assertEqual(len(proposal["choices"]), 32)

        def revision(proposal, feedback):
            self.assertEqual(len(proposal["choices"]), 32)
            self.assertTrue(feedback["triggered_constraints"])
            proposal["choices"][1] = {"id": "c32"}
            proposal["revision_reason"] = "Replace the redundant non-control pair in this synthetic fixture."
            return proposal

        result = run_mock_controller(initial, candidate_ids=ids, shared_validate=shared_validator,
            diagnose=lambda idx: diagnose_batch(covariance, 1, idx, config=config(
                absolute_posterior_correlation=.99, maximum_unexempt_correlated_pairs=0)), revise=revision)
        self.assertTrue(result["revision_requested"])
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["raw_original_proposal"], initial)
        self.assertEqual(result["original_to_final_difference"]["removed_ids"], ["c1"])
        self.assertEqual(result["original_to_final_difference"]["added_ids"], ["c32"])
        self.assertEqual(result["final_status"], "indeterminate")
        self.assertGreater(result["attempts"][1]["numerical_diagnostics"]["conditional_information_gain"]["value_nats"],
                           result["attempts"][0]["numerical_diagnostics"]["conditional_information_gain"]["value_nats"])
        self.assertEqual(len(valid_calls), 3)
        self.assertEqual(len(initial["choices"]), 32)
        json.dumps(result, allow_nan=False)

    def test_unfixed_proposal_is_unresolved_and_never_auto_replaced(self):
        initial = {"choices": [{"id": "a"}, {"id": "b"}]}
        result = run_mock_controller(initial, candidate_ids=["a", "b"], shared_validate=lambda x: x,
            diagnose=lambda idx: diagnose_batch(np.ones((2, 2)), 1, idx, config=config(
                absolute_posterior_correlation=.99, maximum_unexempt_correlated_pairs=0)),
            revise=lambda proposal, feedback: proposal)
        self.assertEqual(result["final_status"], "unresolved_constraint")
        self.assertEqual(result["final_selection"], initial)

    def test_numerical_failure_records_indeterminate_audit_and_requests_no_revision(self):
        initial = {"choices": [{"id": "a"}, {"id": "b"}]}
        result = run_mock_controller(initial, candidate_ids=["a", "b"], shared_validate=lambda x: x,
            diagnose=lambda idx: diagnose_batch([[1, 2], [2, 1]], 1, idx, config=config()),
            revise=lambda proposal, feedback: self.fail("No revision may be requested for a failed numerical diagnostic"))
        self.assertEqual(result["final_status"], "indeterminate")
        self.assertFalse(result["revision_requested"])
        self.assertEqual(result["final_selection"], initial)
        self.assertEqual(result["attempts"][0]["numerical_diagnostics"]["error"]["class"], "ValueError")


if __name__ == "__main__":
    unittest.main()
