"""Mutation tests for reporting arithmetic, ownership and truth boundaries."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("hybrid_provenance", HERE / "provenance_analysis.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


class ProvenanceTests(unittest.TestCase):
    def test_aulc_keeps_original_nrmse_scale(self):
        points = [{"labels": 333, "validation_nrmse": 2.0},
                  {"labels": 365, "validation_nrmse": 1.0},
                  {"labels": 397, "validation_nrmse": 0.0}]
        self.assertEqual(audit.aulc(points, 333, 397), 1.0)
        self.assertEqual(audit.aulc(points, 333, 365), 1.5)

    def test_aulc_forbids_missing_or_interpolated_endpoints(self):
        points = [{"labels": 333, "validation_nrmse": .7}, {"labels": 397, "validation_nrmse": .5}]
        with self.assertRaises(ValueError):
            audit.aulc(points, 333, 397)
        with self.assertRaises(ValueError):
            audit.aulc(points, 333, 365)

    def test_original_owner_switches_are_not_fit_off_by_one(self):
        self.assertEqual(audit.selection_owner(157, audit.FREE, 5), audit.ASTRA)
        self.assertEqual(audit.selection_owner(157, audit.FREE, 6), audit.FREE_CONT)
        self.assertEqual(audit.fit_owner(157, audit.FREE, 6), audit.ASTRA)
        self.assertEqual(audit.fit_owner(157, audit.FREE, 7), audit.FREE_CONT)
        self.assertEqual(audit.selection_owner(6101, audit.FREE, 9), audit.ASTRA)
        self.assertEqual(audit.selection_owner(6101, audit.FREE, 10), audit.FREE_CONT)
        self.assertEqual(audit.selection_owner(157, audit.CW, 2).name, "qgeognn_v2_row_short_sequential_b32")
        self.assertEqual(audit.selection_owner(157, audit.CW, 3).name, "qgeognn_v2_row_cw_lcmd_to_525")
        self.assertEqual(audit.fit_owner(157, audit.CW, 3).name, "qgeognn_v2_row_short_sequential_b32")
        self.assertEqual(audit.fit_owner(157, audit.CW, 4).name, "qgeognn_v2_row_cw_lcmd_to_525")

    def fixture(self):
        directory = audit.ASTRA / f"selections/seed_157/{audit.HYBRID}/round_00"
        return [json.loads((directory / f"{name}.json").read_text()) for name in
                ("input", "catalog", "contract", "selection", "batch_freeze", "feedback", "label_access_receipt")]

    def test_unmodified_frozen_batch_is_accepted(self):
        audit.validate_boundary(*self.fixture())

    def test_duplicate_or_foreign_selection_is_rejected(self):
        fixture = self.fixture()
        fixture[4]["batch_ids"][17] = fixture[4]["batch_ids"][16]
        with self.assertRaises(AssertionError):
            audit.validate_boundary(*fixture)
        fixture = self.fixture()
        fixture[4]["batch_ids"][17] = "foreign_candidate"
        with self.assertRaises(AssertionError):
            audit.validate_boundary(*fixture)

    def test_candidate_truth_injection_is_rejected(self):
        fixture = self.fixture()
        fixture[1]["candidates"][0]["true_V1_ml"] = 99
        with self.assertRaises(AssertionError):
            audit.validate_boundary(*fixture)

    def test_foreign_observed_evidence_is_rejected(self):
        fixture = self.fixture()
        fixture[3]["hypotheses"][0]["supporting_observed_ids"] = [fixture[1]["candidates"][0]["id"]]
        with self.assertRaises(AssertionError):
            audit.validate_boundary(*fixture)

    def test_early_reveal_or_unfrozen_receipt_is_rejected(self):
        fixture = self.fixture()
        fixture[4]["new_batch_labels_revealed"] = True
        with self.assertRaises(AssertionError):
            audit.validate_boundary(*fixture)
        fixture = self.fixture()
        fixture[6]["audit"]["acquisitions_frozen"] = False
        with self.assertRaises(AssertionError):
            audit.validate_boundary(*fixture)

    def test_full_delivery_omission_or_packet_mutation_is_rejected(self):
        fixture = self.fixture()
        fixture[0]["candidates"]["rows"].pop()
        with self.assertRaises(AssertionError):
            audit.validate_boundary(*fixture)

    def test_receipt_must_match_only_actual_selected_ids(self):
        fixture = self.fixture()
        fixture[6]["audit"]["requested_ids_hash"] = audit.ids_hash(fixture[2]["L_t_ids"])
        with self.assertRaises(AssertionError):
            audit.validate_boundary(*fixture)

    def test_delivered_outputs_have_full_paired_scope(self):
        summary = json.loads((HERE / "source_audit.json").read_text())
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["verified_fit_count"], 128)
        self.assertEqual(summary["verified_batch_count"], 120)
        self.assertEqual(summary["served_models_exact_accepted_selection"][f"157/{audit.HYBRID}"], {"gpt-6-astra": 15})
        self.assertEqual(summary["served_models_exact_accepted_selection"][f"157/{audit.FREE}"]["unknown_authorized_transcription"], 1)
        self.assertTrue(summary["scientist_system_prompts_bytes_equal"])
        self.assertEqual(summary["new_training"] + summary["new_llm_calls"] + summary["new_label_requests"], 0)
        rows = read_csv(HERE / "validation_rounds.csv")
        self.assertEqual(len(rows), 128)
        self.assertEqual({row["method"] for row in rows}, set(audit.METHODS))
        for seed in audit.SEEDS:
            for method in audit.METHODS:
                selected = [row for row in rows if int(row["seed"]) == seed and row["method"] == method]
                self.assertEqual([int(row["labels"]) for row in selected], list(range(333, 814, 32)))

    def test_stage_outputs_exactly_reproduce_round_integrals(self):
        rounds = read_csv(HERE / "validation_rounds.csv")
        stages = read_csv(HERE / "validation_stages.csv")
        self.assertEqual(len(stages), 32)
        for row in stages:
            points = [{"labels": int(x["labels"]), "validation_nrmse": float(x["validation_nrmse"])}
                      for x in rounds if (x["seed"], x["method"]) == (row["seed"], row["method"])]
            actual = audit.aulc(points, int(row["labels_start"]), int(row["labels_end"]))
            self.assertEqual(actual, float(row["validation_aulc"]))


if __name__ == "__main__":
    unittest.main()
