"""Boundaries that would materially invalidate strict loading inference."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("selection_analysis", HERE / "selection_analysis.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def row(identifier="x", v=100, density=1., pe="5/1", solvent="DCM", volume=300, smiles="CCO"):
    return {"id": identifier, "smiles": smiles, "conditions": {"PE/EA": pe, "Density g/ml": density, "V/ul": v,
                                                               "loading solvent": solvent, "Volume of loading solvent/ul": volume}}


class ConditionTests(unittest.TestCase):
    def test_each_original_axis_is_separate_and_mass_is_not_sixth(self):
        for axis, kwargs in (("v", {"v": 50}), ("density", {"density": .9}), ("loading_type", {"solvent": "PE"}),
                             ("loading_volume", {"volume": 200}), ("pe", {"pe": "10/1"})):
            with self.subTest(axis=axis):
                actual = audit.point_flags(row(), [row("p", **kwargs)], [])
                self.assertTrue(actual[axis + "_extension_prior"])
                self.assertEqual(actual["strict_loading_prior"], axis != "pe")
                self.assertEqual(sum(actual[a + "_extension_prior"] for a in audit.AXES), 1)
        self.assertNotIn("loading_amount_mg", audit.CONDITIONS)

    def test_multiaxis_is_broad_not_strict(self):
        flags = audit.point_flags(row(), [row("p", v=50, pe="10/1")], [])
        self.assertTrue(flags["broad_loading_prior"])
        self.assertTrue(flags["multiple_conditions_changed_prior"])
        self.assertFalse(flags["strict_loading_prior"])

    def test_molecule_identity_and_all_other_conditions_required(self):
        self.assertFalse(audit.point_flags(row(), [row("p", v=50, smiles="CCC")], [])["strict_loading_prior"])
        self.assertFalse(audit.point_flags(row(), [row("p", v=50, density=.9)], [])["strict_loading_prior"])

    def test_prior_tuple_repeat_is_not_missing_extension_even_with_batch_contrast(self):
        flags = audit.point_flags(row(), [row("p"), row("q", v=50)], [row("b", v=150)])
        self.assertTrue(flags["v_matched_contrast_prior"])
        self.assertTrue(flags["exact_prior_tuple_repeat"])
        self.assertFalse(flags["legacy_v_prior_extension"])
        self.assertFalse(flags["strict_loading_union"])

    def test_full_batch_peer_across_sources_and_union_counts_point_once(self):
        selected, pending = row("llm"), row("cw", v=50)
        flags = audit.point_flags(selected, [], [selected, pending, row("cw2", density=.9)])
        self.assertTrue(flags["strict_loading_batch_only"])
        self.assertTrue(flags["v_extension_batch"])
        self.assertTrue(flags["density_extension_batch"])
        self.assertIs(flags["strict_loading_union"], True)
        self.assertFalse(flags["known_molecule"])

    def test_prior_batch_both_disjoint_partition(self):
        flags = audit.point_flags(row(), [row("p", v=50)], [row("x"), row("b", volume=200)])
        self.assertTrue(flags["strict_loading_both"])
        self.assertFalse(flags["strict_loading_prior_only"])
        self.assertFalse(flags["strict_loading_batch_only"])

    def test_legacy_exactly_reused_for_v(self):
        for prior in ([], [row("p", v=50)], [row("p", v=50), row("q")], [row("p", v=50, pe="10/1")]):
            actual = audit.point_flags(row(), prior, [])
            previous = {"CCO": prior} if prior else {}
            self.assertEqual(actual["legacy_v_prior_extension"], audit.legacy.local_flags(row(), previous)["amount_curve_extension"])


class ReasonTests(unittest.TestCase):
    def test_text_is_proxy_and_exact_ids_are_checked(self):
        selected = row("12345678901234567890")
        other = row("aaaaaaaaaaaaaaaaaaaa", v=50)
        choice = {"reason": "Loading contrast to aaaaaaaaaaaaaaaaaaaa and bbbbbbbbbbbbbbbbbbbb for m1."}
        actual = audit.reason_features(choice, selected, {selected["id"]: selected, other["id"]: other}, [other["id"]], [], [selected["id"]], {"m1": {"smiles": "CCO"}})
        self.assertTrue(actual["reason_loading_mention"])
        self.assertEqual(actual["reason_strict_loading_reference_count"], 1)
        self.assertEqual(actual["reason_unknown_id_count"], 1)
        self.assertEqual(actual["reason_other_molecule_token_count"], 0)
        self.assertEqual(actual["reason_unknown_molecule_token_count"], 0)


class FrozenOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles = pd.read_csv(HERE / "behavior_profiles.csv")
        cls.points = pd.read_csv(HERE / "selection_point_flags.csv")

    def test_complete_exact_sources_and_denominators(self):
        p = self.profiles
        self.assertEqual(len(p), 180)
        self.assertEqual(len(self.points), 4800)
        self.assertEqual(set(p.source), {"hybrid_cw16", "hybrid_llm16", "hybrid_all32", "free32", "cw_lcmd32", "raw_lcmd32"})
        self.assertFalse(p.duplicated(["seed", "round", "source"]).any())
        self.assertTrue((p.labels_before == 333 + p["round"] * 32).all())
        self.assertTrue((p.batch_context_denominator == 32).all())
        self.assertTrue((p[p.source.isin(["hybrid_cw16", "hybrid_llm16"])].denominator == 16).all())
        self.assertTrue((p[~p.source.isin(["hybrid_cw16", "hybrid_llm16"])].denominator == 32).all())

    def test_hybrid_subsets_sum_to_full_and_point_flags_identical(self):
        cols = [c for c in self.profiles if c.endswith("_count") and c.startswith(("strict_loading_", "v_extension_", "density_extension_", "loading_type_extension_", "loading_volume_extension_", "pe_extension_"))]
        indexed = self.profiles.set_index(["seed", "round", "source"])
        for seed in (157, 6101):
            for rnd in range(15):
                for c in cols:
                    self.assertEqual(indexed.loc[(seed, rnd, "hybrid_all32"), c], indexed.loc[(seed, rnd, "hybrid_cw16"), c] + indexed.loc[(seed, rnd, "hybrid_llm16"), c])
        points = self.points.query("source.str.startswith('hybrid')", engine="python")
        self.assertTrue((points.groupby(["seed", "round", "sample_id"]).strict_loading_union.nunique() == 1).all())

    def test_partitions_and_no_tuple_repeat_counted(self):
        p = self.points
        self.assertTrue((p.strict_loading_union.astype(int) == p.strict_loading_prior_only.astype(int) + p.strict_loading_batch_only.astype(int) + p.strict_loading_both.astype(int)).all())
        self.assertFalse(p[p.exact_prior_tuple_repeat].strict_loading_union.any())
        self.assertTrue((p.legacy_v_prior_extension == p.v_extension_prior).all())

    def test_frozen_ids_and_pending_source(self):
        for _, g in self.points.groupby(["seed", "round", "source"]):
            path = audit.ROOT / g.selection_source.iloc[0]
            if path.suffix == ".json":
                freeze = json.loads(path.read_text())
                expected = freeze["pending_ids"] if g.source.iloc[0] == "hybrid_cw16" else [i for i in freeze["batch_ids"] if i not in freeze.get("pending_ids", [])] if g.source.iloc[0] == "hybrid_llm16" else freeze["batch_ids"]
            else:
                expected = pd.read_csv(path).sample_id.tolist()
            self.assertEqual(g.sort_values("source_order").sample_id.tolist(), expected)

    def test_no_candidate_or_test_truth_emitted(self):
        self.assertFalse(any(c.startswith(("true_", "error_")) or c in ("V1_ml", "V2_ml") for c in self.points))
        controls = self.profiles.query("source in ['cw_lcmd32','raw_lcmd32']")
        self.assertTrue(controls.high8_same_molecule_fraction.isna().all())
        llm = self.profiles.query("source in ['free32','hybrid_all32','hybrid_cw16','hybrid_llm16']")
        self.assertTrue((llm.history_error_available_count == llm["round"] * 32).all())

    def test_actual_legacy_alignment_window(self):
        old = pd.read_csv(audit.STUDIES / "free_lcmd_mechanism_diagnostics_20261029/behavior_profiles.csv")
        old = old.query("method == @audit.FREE and seed in [157,6101] and round < 15").sort_values(["seed", "round"])
        new = self.profiles.query("source == 'free32'").sort_values(["seed", "round"])
        np.testing.assert_array_equal(old.amount_curve_extension_fraction.to_numpy(), new.legacy_v_prior_extension_fraction.to_numpy())


if __name__ == "__main__":
    unittest.main()
