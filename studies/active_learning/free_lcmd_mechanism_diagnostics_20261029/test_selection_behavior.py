"""Lightweight tests of temporal/condition boundaries and emitted frozen IDs."""
import importlib.util
import json
from pathlib import Path
import unittest

import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("behavior",HERE/"selection_behavior_analysis.py")
behavior=importlib.util.module_from_spec(spec)
spec.loader.exec_module(behavior)


def row(amount=100,pe="5/1",volume=300,solvent="DCM",smiles="CCO"):
    return {"smiles":smiles,"conditions":{"PE/EA":pe,"Density g/ml":1.,"V/ul":amount,
                                          "loading solvent":solvent,"Volume of loading solvent/ul":volume}}


class ConditionBoundaryTests(unittest.TestCase):
    def test_unobserved_is_not_condition_completion(self):
        actual=behavior.local_flags(row(),{})
        self.assertFalse(actual["known_molecule"])
        self.assertFalse(actual["amount_curve_extension"])

    def test_amount_requires_other_four_conditions_match(self):
        good=behavior.local_flags(row(100),{"CCO":[row(50)]})
        bad=behavior.local_flags(row(100),{"CCO":[row(50,pe="10/1")]})
        self.assertTrue(good["amount_curve_extension"])
        self.assertFalse(bad["amount_curve_extension"])
        self.assertFalse(good["pe_curve_extension"])

    def test_interpolation_requires_observed_both_sides(self):
        self.assertFalse(behavior.local_flags(row(100),{"CCO":[row(50)]})["amount_interpolation"])
        self.assertTrue(behavior.local_flags(row(100),{"CCO":[row(50),row(150)]})["amount_interpolation"])
        # Future/same-batch rows are absent from this list and cannot count.

    def test_exact_conditions_repeat_and_pe_fraction(self):
        self.assertTrue(behavior.local_flags(row(),{"CCO":[row()]})["exact_condition_repeat"])
        self.assertAlmostEqual(behavior.ea_fraction("5/1"),1/6)
        self.assertEqual(behavior.ea_fraction("0/1"),1.)

    def test_replication_is_not_missing_condition_completion(self):
        actual=behavior.local_flags(row(100),{"CCO":[row(50),row(100),row(150)]})
        self.assertTrue(actual["amount_matched_contrast"])
        self.assertFalse(actual["amount_curve_extension"])
        self.assertFalse(actual["amount_interpolation"])


class FrozenOutputsTests(unittest.TestCase):
    def test_records_and_profiles_complete_without_truth_columns(self):
        records=pd.read_csv(HERE/"selection_records.csv")
        profiles=pd.read_csv(HERE/"behavior_profiles.csv")
        self.assertEqual(len(profiles),360)
        self.assertEqual(len(records),11520)
        self.assertEqual(records.groupby(["seed","method","round"]).sample_id.nunique().min(),32)
        self.assertFalse(any(c.startswith(("true_","error_")) or c in ("V1_ml","V2_ml") for c in records))
        self.assertTrue((profiles.labels_before == 333+32*profiles["round"]).all())
        self.assertTrue(profiles.query("method == 'lcmd'").high8_same_molecule_fraction.isna().all())

    def test_every_selected_id_matches_frozen_source_order(self):
        records=pd.read_csv(HERE/"selection_records.csv")
        for _,group in records.groupby(["seed","method","round"]):
            source=behavior.ROOT/group.selection_source.iloc[0]
            expected=json.loads(source.read_text())["batch_ids"] if source.suffix==".json" else pd.read_csv(source).sample_id.tolist()
            self.assertEqual(group.sort_values("selection_order").sample_id.tolist(),expected)

    def test_prior_error_history_cardinality(self):
        profiles=pd.read_csv(HERE/"behavior_profiles.csv")
        free=profiles.query("method == 'free_llm32_scientist_v2'")
        self.assertTrue((free.history_error_available_count == free["round"]*32).all())


class TextBoundaryTests(unittest.TestCase):
    def test_schema_keys_and_identifier_fields_do_not_count_as_prose(self):
        selection={"hypotheses":[{"id":"CW","supporting_observed_ids":["residual"],"contradicting_observed_ids":[],"content":"A molecule has two conditions."}]}
        actual=behavior.text_features(selection)
        self.assertFalse(actual["text_cw_mention"])
        self.assertFalse(actual["text_observed_feedback_mention"])
        self.assertFalse(actual["text_failure_repair_mention"])

    def test_physical_failure_is_not_model_failure(self):
        self.assertFalse(behavior.text_features({"batch_rationale":"A physical column failure needs correction."})["text_failure_repair_mention"])
        self.assertTrue(behavior.text_features({"batch_rationale":"The model underpredicts measured completion."})["text_failure_repair_mention"])


if __name__ == "__main__":
    unittest.main()
