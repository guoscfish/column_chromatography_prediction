"""Data-boundary and frozen-representation tests using synthetic rows only."""
from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from representations import StateArtifacts, FEATURE_COLUMNS, structured_features


def synthetic_frame():
    return pd.DataFrame({
        "sample_id": ["a", "b", "c"], "canonical_smiles": ["CCO", "CCO", "CCO"],
        "Density g/ml": [1., 1., 1.], "V/ul": [10., 20., 10000.],
        "loading solvent": ["PE", "EA", "DCM"],
        "Volume of loading solvent/ul": [100., 200., 10000.], "PE/EA": ["1/1", "1/1", "1/1"]})


class RepresentationTests(unittest.TestCase):
    def test_state_rejects_test_and_duplicate_indices(self):
        state = StateArtifacts(Path("."), 157, 333, 0, Path("."), Path("model"), Path("raw"),
                               ["a"], ["b"], ["c"], np.array([1]), np.array([2]),
                               np.array([3]), np.array([1]), pd.DataFrame(), {})
        np.testing.assert_array_equal(state.validate_indices([3, 1, 2]), [3, 1, 2])
        with self.assertRaises(PermissionError):
            state.validate_indices([4])
        with self.assertRaises(ValueError):
            state.validate_indices([1, 1])

    def test_structured_fits_continuous_scales_only_on_current_training_rows(self):
        frame = synthetic_frame()
        values, audit = structured_features(frame, [0, 1])
        self.assertEqual(values.shape, (3, 2059))
        self.assertEqual(values.dtype, np.float64)
        # The extreme held-out row must not alter mean/scale of training rows.
        np.testing.assert_allclose(values[:2, 2054:2056], [[-1, -1], [1, 1]], atol=1e-12)
        np.testing.assert_allclose(audit["continuous_mean"][-2:], [15., 150.])
        np.testing.assert_array_equal(values[:, -3:], np.eye(3))
        np.testing.assert_array_equal(values[0, :2048], values[2, :2048])
        self.assertEqual(audit["standardizer_fit_role"], "current_L_t_only")

    def test_structured_rejects_response_columns_before_encoding(self):
        frame = synthetic_frame()
        self.assertEqual(set(frame), set(FEATURE_COLUMNS))
        frame["V1_ml"] = [1., 2., 3.]
        with self.assertRaises(PermissionError):
            structured_features(frame, [0, 1])

    def test_structured_conditions_change_for_identical_molecule(self):
        frame = synthetic_frame()
        values, _ = structured_features(frame, [0, 1])
        self.assertGreater(np.linalg.norm(values[0] - values[1]), 0)


if __name__ == "__main__":
    unittest.main()
