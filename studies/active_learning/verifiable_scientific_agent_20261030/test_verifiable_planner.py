"""Stdlib mock tests; no LLM, predictor, label-store, or test data access."""
import copy
import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from verifiable_planner import (FrozenContext, StableCandidateIndex, Unanalyzable,
                               apply_mock_patch, baseline_validate, build_repair_request,
                               diagnose, digest, format_only_repair, frozen_validator_bundle,
                               parse_complete_json, prepare_repair_from_response)


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = frozen_validator_bundle()

    def setUp(self):
        self.context = FrozenContext(("u0", "u1", "u2", "u3"), frozenset({"l0", "l1"}),
                                     frozenset({"l0", "l1"}), "frozen-packet", 2, 1)
        self.plan = {"type": "selection", "packet_sha256": "frozen-packet",
                     "choices": [{"id": "u0", "reason": "Contrast two solvent conditions.", "hypothesis_id": "h0"},
                                 {"id": "u1", "reason": "Retain the matched structural control.", "hypothesis_id": None}],
                     "hypotheses": [{"id": "h0", "content": "The solvent contrast may expose a local response.",
                                     "basis": "mixed", "confidence": "low", "status": "proposed",
                                     "supporting_observed_ids": ["l0"], "contradicting_observed_ids": [],
                                     "why_it_matters_for_model_learning": "Tests a current response ambiguity.",
                                     "next_discriminating_question": "Does the matched contrast persist?"}],
                     "batch_strategy": "Matched solvent contrast", "batch_rationale": "Discriminate local response.",
                     "feedback_interpretation": "The previous measured contrast is uncertain."}

    def patch(self, plan, changes, *, additions=None):
        request = build_repair_request(plan, self.context, self.bundle, intent_is_clear=True)
        patch = {"base_proposal_sha256": digest(plan), "context_sha256": self.context.binding_sha256,
                 "intent_preservation_statement": "Mock planner explicitly retains the original condition contrast.",
                 "operations": [{"op": "replace", "path": path, "value": value} for path, value in changes],
                 "append_choices": additions or []}
        return request, patch

    def test_valid_plan_matches_baseline(self):
        result = diagnose(self.plan, self.context, self.bundle)
        self.assertTrue(result.authoritative_legal)
        self.assertEqual(result.findings, [])
        self.assertEqual(result.legal_candidate_indices, [0, 1])

    def test_all_id_failures_are_localized_together(self):
        self.plan["choices"] += [copy.deepcopy(self.plan["choices"][0]),
                                 {"id": "l0", "reason": "A measured row.", "hypothesis_id": None},
                                 {"id": "invented", "reason": "An unknown row.", "hypothesis_id": None}]
        result = diagnose(self.plan, self.context, self.bundle)
        codes = {x["code"] for x in result.findings}
        self.assertTrue({"selection_count", "duplicate_candidate_id", "out_of_pool_id", "already_measured_id"} <= codes)
        self.assertEqual(result.legal_candidate_indices, [0, 1])
        self.assertTrue(result.full_replanning_required)  # Excess lawful items cannot be silently deleted.

    def test_evidence_and_hypothesis_reference(self):
        self.plan["hypotheses"][0]["supporting_observed_ids"] += ["u2", "unseen"]
        self.plan["choices"][1]["hypothesis_id"] = "h99"
        result = diagnose(self.plan, self.context, self.bundle)
        finding = next(x for x in result.findings if x["code"] == "unobserved_evidence")
        self.assertEqual(finding["invalid_ids"], ["u2", "unseen"])
        self.assertEqual(finding["retained_valid_ids"], ["l0"])
        self.assertIn("invalid_hypothesis_reference", {x["code"] for x in result.findings})

    def test_malformed_choices_never_become_a_determined_batch(self):
        self.plan["choices"] = {"id": "u0"}
        result = diagnose(self.plan, self.context, self.bundle)
        self.assertTrue(result.full_replanning_required)
        self.assertFalse(result.analyzable)
        self.assertEqual(result.legal_candidate_indices, [])

    def test_hypothesis_objects_in_choices_do_not_define_an_experiment_batch(self):
        self.plan["choices"].append({"id": "h_in_wrong_array", "content": "A hypothesis, not an experiment."})
        result = diagnose(self.plan, self.context, self.bundle)
        self.assertFalse(result.analyzable)
        self.assertTrue(result.full_replanning_required)
        self.assertEqual(result.legal_candidate_indices, [0, 1])  # Explicitly partial individual fields.

    def test_missing_count_is_explicit(self):
        self.plan["choices"].pop()
        diagnosis = diagnose(self.plan, self.context, self.bundle)
        self.assertEqual(diagnosis.missing_count, 1)
        request, patch = self.patch(self.plan, [], additions=[{"id": "u1", "reason": "Original matched control.",
                                                             "hypothesis_id": None}])
        repaired = apply_mock_patch(self.plan, self.context, request, patch, self.bundle)
        self.assertEqual(repaired["choices"][0], self.plan["choices"][0])

    def test_duplicate_patch_rewrites_invalid_slot_and_preserves_unaffected_fields(self):
        self.plan["choices"][1]["id"] = "u0"
        before = copy.deepcopy(self.plan)
        replacement = {"id": "u1", "reason": "Explicit revised reason for the actual matched control.", "hypothesis_id": None}
        request, patch = self.patch(self.plan, [("/choices/1", replacement)])
        repaired = apply_mock_patch(self.plan, self.context, request, patch, self.bundle)
        expected = copy.deepcopy(before)
        expected["choices"][1] = replacement
        self.assertEqual(repaired, expected)
        self.assertEqual(self.plan, before)
        self.assertTrue(diagnose(repaired, self.context, self.bundle).authoritative_legal)

    def test_patch_cannot_edit_legal_choice_or_mechanically_change_only_illegal_id(self):
        self.plan["choices"][1]["id"] = "invented"
        for path, value in (("/choices/0/id", "u3"), ("/choices/1/id", "u1")):
            with self.subTest(path=path):
                request, patch = self.patch(self.plan, [(path, value)])
                with self.assertRaises(ValueError):
                    apply_mock_patch(self.plan, self.context, request, patch, self.bundle)

    def test_patch_cannot_drop_valid_observed_evidence(self):
        self.plan["hypotheses"][0]["supporting_observed_ids"].append("u2")
        revised = copy.deepcopy(self.plan["hypotheses"][0])
        revised["supporting_observed_ids"] = []
        request, patch = self.patch(self.plan, [("/hypotheses/0", revised)])
        with self.assertRaises(ValueError):
            apply_mock_patch(self.plan, self.context, request, patch, self.bundle)
        # A planner may explicitly revise the scientific claim and evidence role.
        revised["contradicting_observed_ids"] = ["l0"]
        revised["status"] = "weakened"
        revised["content"] = "The observed contrast instead weakens the prior expectation."
        patch["operations"][0]["value"] = revised
        repaired = apply_mock_patch(self.plan, self.context, request, patch, self.bundle)
        self.assertEqual(repaired["choices"], self.plan["choices"])
        self.assertEqual(repaired["hypotheses"][0]["contradicting_observed_ids"], ["l0"])

    def test_stale_patch_and_tampered_request_rejected(self):
        self.plan["choices"][1]["id"] = "invented"
        replacement = {"id": "u1", "reason": "Explicit matched control.", "hypothesis_id": None}
        request, patch = self.patch(self.plan, [("/choices/1", replacement)])
        patch["base_proposal_sha256"] = "stale"
        with self.assertRaises(ValueError):
            apply_mock_patch(self.plan, self.context, request, patch, self.bundle)
        request, patch = self.patch(self.plan, [("/choices/1", replacement)])
        request["replace_paths"].append("/choices/0/id")
        with self.assertRaises(ValueError):
            apply_mock_patch(self.plan, self.context, request, patch, self.bundle)

    def test_ambiguous_intent_requires_full_replanning(self):
        self.plan["choices"][1]["id"] = "invented"
        request = build_repair_request(self.plan, self.context, self.bundle, intent_is_clear=False)
        self.assertEqual(request["mode"], "full_replanning_required")
        with self.assertRaises(ValueError):
            apply_mock_patch(self.plan, self.context, request, {}, self.bundle)

    def test_bad_binding_requires_replanning(self):
        self.plan["packet_sha256"] = "different-round"
        request = build_repair_request(self.plan, self.context, self.bundle, intent_is_clear=True)
        self.assertEqual(request["mode"], "full_replanning_required")

    def test_invalid_patch_cannot_pass_original_validator(self):
        self.plan["choices"][1]["id"] = "invented"
        replacement = {"id": "l1", "reason": "Explicit but illegal measured choice.", "hypothesis_id": None}
        request, patch = self.patch(self.plan, [("/choices/1", replacement)])
        with self.assertRaises(ValueError):
            apply_mock_patch(self.plan, self.context, request, patch, self.bundle)

    def test_context_rejects_overlap_and_unobserved_evidence_set(self):
        with self.assertRaises(ValueError):
            FrozenContext(("u0",), frozenset({"u0"}), frozenset(), "p", 1, 0)
        with self.assertRaises(ValueError):
            FrozenContext(("u0",), frozenset(), frozenset({"l0"}), "p", 1, 0)

    def test_nocw_schema_prose_and_content_match_frozen_validator(self):
        for variant, change, code in (
            ("cw_aware", lambda p: p.update(extra="unknown"), "schema_unauthorized_field"),
            ("cw_aware", lambda p: p["choices"][0].update(reason="See measured l0."), "untyped_evidence_prose"),
            ("no_cw", lambda p: p.update(batch_strategy="Select on cw_distance_percentile"), "nocw_forbidden_content")):
            with self.subTest(code=code):
                context = FrozenContext(self.context.candidate_ids, self.context.labeled_ids,
                                        self.context.observed_ids, self.context.packet_sha256, 2, 1, variant)
                plan = copy.deepcopy(self.plan)
                change(plan)
                result = diagnose(plan, context, self.bundle)
                self.assertFalse(result.authoritative_legal)
                self.assertEqual(result.authoritative_legal, baseline_validate(plan, context, self.bundle)[0])
                self.assertIn(code, {x["code"] for x in result.findings})

    def test_baseline_legal_not_redefined_by_extra_diagnostics(self):
        # Legacy direct allows extra fields; B must preserve exactly that acceptance.
        self.plan["extra"] = "legacy field"
        self.assertTrue(diagnose(self.plan, self.context, self.bundle).authoritative_legal)

    def test_actual_five_seed_nocw_keeps_legacy_free_prose_rules(self):
        context = FrozenContext(self.context.candidate_ids, self.context.labeled_ids,
                                self.context.observed_ids, self.context.packet_sha256, 2, 1, "no_cw")
        self.plan["choices"][0]["reason"] = "Compare the measured observation l0."
        self.assertEqual(context.validator_kind, "nocw_five")
        self.assertTrue(diagnose(self.plan, context, self.bundle).authoritative_legal)

    def packet(self):
        return {"packet_sha256": "frozen-packet", "selection_count": 2, "round": 1,
                "condition_columns": ["solvent"], "molecules": {"m0": {"smiles": "C"}},
                "candidates": {"columns": ["id", "molecule_id", "conditions", "pred_V1_ml"],
                               "rows": [[identity, "m0", ["EA"], 1.234567890123] for identity in self.context.candidate_ids]},
                "observed": {"columns": ["id"], "rows": [["l0"], ["l1"]]},
                "pending": {"columns": ["id"], "rows": []}}

    def test_index_preserves_full_pool_order_values_precision(self):
        source = self.packet()
        before = copy.deepcopy(source)
        index = StableCandidateIndex(source, self.context)
        shown = index.presentation()
        self.assertEqual(source, before)
        self.assertEqual(shown["candidate_count"], 4)
        self.assertEqual(shown["packet"]["molecules"], source["molecules"])
        self.assertEqual(shown["packet"]["observed"], source["observed"])
        self.assertEqual([r[:-1] for r in shown["packet"]["candidates"]["rows"]], source["candidates"]["rows"])
        for number, identity in enumerate(self.context.candidate_ids, 1):
            self.assertEqual(index.to_id(number, binding_sha256=index.binding_sha256), identity)
            self.assertEqual(index.to_index(identity, binding_sha256=index.binding_sha256), number)

    def test_index_rejects_bool_range_stale_ambiguous_or_duplicate_mapping(self):
        index = StableCandidateIndex(self.packet(), self.context)
        for value in (True, 0, -1, 5, "1", 1.0):
            with self.subTest(index=value):
                with self.assertRaises(ValueError):
                    index.to_id(value, binding_sha256=index.binding_sha256)
        with self.assertRaises(ValueError):
            index.to_id(1, binding_sha256="stale")
        source = self.packet()
        source["candidates"]["rows"][1][0] = "u0"
        with self.assertRaises(ValueError):
            StableCandidateIndex(source, self.context)

    def test_index_decode_preserves_all_scientific_text(self):
        index = StableCandidateIndex(self.packet(), self.context)
        proposal = copy.deepcopy(self.plan)
        proposal["index_binding_sha256"] = index.binding_sha256
        for choice in proposal["choices"]:
            choice["local_index"] = index.to_index(choice.pop("id"), binding_sha256=index.binding_sha256)
        self.assertEqual(index.decode_proposal(proposal, context=self.context, bundle=self.bundle), self.plan)
        proposal["choices"][1]["local_index"] = 1
        with self.assertRaises(ValueError):
            index.decode_proposal(proposal, context=self.context, bundle=self.bundle)

    def test_explicit_identity_contract_reads_only_packet_ids(self):
        packet = self.packet()
        packet["observed"]["columns"].append("V1_ml")
        packet["observed"]["rows"] = [["l0", {"unread": "opaque"}], ["l1", {"unread": "opaque"}]]
        context = FrozenContext.from_packet(packet, labeled_ids=["l0", "l1"], unlabeled_ids=self.context.candidate_ids)
        self.assertEqual(context, self.context)
        with self.assertRaises(ValueError):
            FrozenContext.from_packet(packet, labeled_ids=["l0", "l1"], unlabeled_ids=["u0"])

    def test_only_whole_json_fence_is_safe_format_repair(self):
        text = json.dumps(self.plan)
        output, operation = format_only_repair("\n```json\n" + text + "\n```\n")
        self.assertEqual(output, text)
        self.assertEqual(operation, "unwrap_single_json_fence")
        self.assertEqual(json.loads(output), self.plan)
        self.assertEqual(format_only_repair(text), (text, "identity"))

    def test_ambiguous_parse_never_recovers_candidate_batch(self):
        for text in ('{"choices": [], "choices": []}', '{"x": NaN}', '{"x": Infinity}',
                     '{"x":1e400}', '[{"nested":[-1e400]}]',
                     '{} {}', 'Here is the result: {}', '```json\n{}\n``` trailing',
                     '```json\n{}\n```\n```json\n{}\n```', '{"choices": ['):
            with self.subTest(text=text):
                with self.assertRaises(Unanalyzable):
                    format_only_repair(text)
        with self.assertRaises(Unanalyzable):
            parse_complete_json('{"choices": [], "choices": []}')

    def test_raw_response_repair_boundary_rejects_ambiguity(self):
        result = prepare_repair_from_response('{"choices": [], "choices": []}', self.context,
                                               self.bundle, intent_is_clear=True)
        self.assertFalse(result["unambiguous_proposal"])
        self.assertEqual(result["mode"], "full_replanning_required")
        self.assertNotIn("plan", result)
        result = prepare_repair_from_response(json.dumps(self.plan), self.context,
                                               self.bundle, intent_is_clear=True)
        self.assertTrue(result["unambiguous_proposal"])
        self.assertEqual(result["request"]["mode"], "already_verified")

    def test_offline_B_keeps_A_legacy_duplicate_key_acceptance_without_analyzing_batch(self):
        from offline_benchmark import benchmark_entry
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            packet = folder / "input.json"
            contract = folder / "contract.json"
            response = folder / "turn.json"
            packet.write_text(json.dumps(self.packet()))
            contract.write_text(json.dumps({"L_t_ids": ["l0", "l1"], "U_t_ids": ["u0", "u1", "u2", "u3"]}))
            answer = '{"batch_strategy":"ambiguous first value",' + json.dumps(self.plan)[1:]
            response.write_text(json.dumps({"success": True, "answer": answer}))
            entry = {"method": "mock", "seed": 0, "round": 1, "attempt": 0,
                     "packet_path": str(packet), "contract_path": str(contract), "response_path": str(response),
                     "response_sha256": hashlib.sha256(response.read_bytes()).hexdigest(),
                     "completion": "frozen_artifact", "validator_kind": "direct", "variant": "direct"}
            row, detail = benchmark_entry(entry, folder / "manifest.jsonl", self.bundle)
            self.assertTrue(row["A_legal"])
            self.assertTrue(row["B_legal"])
            self.assertFalse(row["C_legal"])
            self.assertFalse(row["raw_json_reliably_parseable"])
            self.assertIsNone(row["legal_candidate_occurrences"])
            self.assertEqual(detail["local_patch_assessment"], "full_replanning_required_for_safe_protocol")


if __name__ == "__main__":
    unittest.main(verbosity=2)
