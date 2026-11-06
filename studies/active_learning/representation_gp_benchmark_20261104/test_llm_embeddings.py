"""Feature-boundary and lossless-template tests require no downloaded model."""
import unittest
from llm_embeddings import render_experiment_text


def example():
    return {"canonical_smiles": "CC(=O)O", "conditions": {
        "PE/EA": "20/1", "Density g/ml": "1.0491", "V/ul": "34.01360544",
        "loading solvent": "DCM", "Volume of loading solvent/ul": "1000"}}


class TemplateTests(unittest.TestCase):
    def test_equivalent_templates_preserve_values_and_distinct_volumes(self):
        for template in ["A", "B"]:
            text = render_experiment_text(example(), template=template)
            for required in ["1.0491", "34.01360544", "1000", "20/1", "DCM", "4 g", "silica gel", "mg"]:
                self.assertIn(required, text)
            self.assertNotIn("V1", text)
            self.assertNotIn("V2", text)

    def test_smiles_only_ignores_conditions(self):
        a = example()
        b = example()
        b["conditions"]["V/ul"] = "150"
        self.assertEqual(render_experiment_text(a, "smiles_only"), render_experiment_text(b, "smiles_only"))
        self.assertNotEqual(render_experiment_text(a), render_experiment_text(b))

    def test_response_or_prediction_fields_rejected(self):
        for field in ["V1_ml", "V2_ml", "pred_V1_ml", "residual", "validation_score"]:
            row = example()
            row[field] = 1
            with self.assertRaises(ValueError):
                render_experiment_text(row)

    def test_invalid_condition_rejected(self):
        for value in [None, float("nan"), float("inf"), -1, 0, ""]:
            row = example()
            row["conditions"]["V/ul"] = value
            with self.assertRaises(ValueError):
                render_experiment_text(row)
        row = example()
        del row["conditions"]["Density g/ml"]
        with self.assertRaises(ValueError):
            render_experiment_text(row)

    def test_flat_form_and_nested_form_agree(self):
        row = example()
        self.assertEqual(render_experiment_text(row), render_experiment_text({"smiles": row["canonical_smiles"], **row["conditions"]}))

    def test_pure_ethyl_acetate_ratio_preserved(self):
        row = example()
        row["conditions"]["PE/EA"] = "0/1"
        self.assertIn("0/1", render_experiment_text(row))


if __name__ == "__main__":
    unittest.main()
