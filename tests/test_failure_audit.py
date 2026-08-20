import json, re, unittest
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.al_acquisition import fit_standardizer, transform_standardized
from scripts.run_e2_compound_failure_audit import paired_compound_effects

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/e2_compound_failure_audit"

class FailureAuditTests(unittest.TestCase):
    def test_readme_seed42_values_match_csv(self):
        frame = pd.read_csv(OUT / "test_labeled_coverage_trajectory.csv")
        expected = frame[(frame.outer_seed == 42) & (frame["round"] == 8)].set_index("strategy").relative_coverage_gain.to_dict()
        text = (OUT / "README.md").read_text()
        reported = json.loads(re.search(r"Seed42 final test-to-labeled relative coverage gain: (\{.*?\})", text).group(1))
        self.assertEqual(set(expected), set(reported))
        for key in expected: self.assertAlmostEqual(expected[key], reported[key], places=12)

    def test_gradient_row_semantics(self):
        frame = pd.read_csv(OUT / "test_labeled_coverage_trajectory.csv")
        self.assertNotIn("labeled_total", frame)
        self.assertEqual(set(frame.loc[frame["round"] == 0, "gradient_train_rows"]), {318})

    def test_paired_compound_delta(self):
        frame = pd.DataFrame([{"outer_seed":42,"round":8,"strategy":s,"canonical_smiles":c,"compound_rows":1,"normalized_RMSE":v,"normalized_MAE":v} for c in ("A","B") for s,v in (("random",1.0),("coverage",.8),("ensemble",1.1),("hybrid",1.2))])
        effects, _ = paired_compound_effects(frame)
        self.assertTrue(np.allclose(effects[effects.strategy == "coverage"].delta_vs_random, -.2))

    def test_condition_standardizer_is_l0_only(self):
        l0=np.array([[0.,10.],[2.,14.]])
        mean,scale=fit_standardizer(l0)
        self.assertTrue(np.allclose(transform_standardized(l0,mean,scale).mean(axis=0),0))
        # An extreme held-out point must not change fitted statistics.
        self.assertTrue(np.array_equal(mean, np.array([[1.,12.]])))
        relevance=pd.read_csv(OUT / "selected_test_relevance.csv")
        self.assertIn("mean_selected_to_test_standardized_condition_distance", relevance)
        self.assertNotIn("mean_selected_to_test_condition_distance", relevance)

if __name__ == "__main__": unittest.main()
