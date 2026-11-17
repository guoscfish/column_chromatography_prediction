"""Numerical and candidate-exclusion tests for the descriptive choice model."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

SPEC = importlib.util.spec_from_file_location('audit_preferences', Path(__file__).resolve().parents[1] / 'preferences.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class PreferenceTests(unittest.TestCase):
    def test_gradient(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(20, 65, 3))
        b = np.array([.2, -.3, .7])
        _, g = M.conditional_objective(b, x)
        numerical = np.array([(M.conditional_objective(b + np.eye(3)[j] * 1e-5, x)[0] -
                               M.conditional_objective(b - np.eye(3)[j] * 1e-5, x)[0]) / 2e-5 for j in range(3)])
        np.testing.assert_allclose(g, numerical, atol=1e-7)

    def test_extreme_utilities_are_stable(self):
        x = np.array([[[1e4], [-1e4], [0]], [[-1e4], [1e4], [0]]])
        loss, g = M.conditional_objective(np.array([1.]), x)
        self.assertTrue(np.isfinite(loss) and np.isfinite(g).all())
        self.assertTrue(np.isfinite(M.fit_logit(x)[0]).all())

    def test_uniform_model(self):
        x = np.ones((5, 65, 2))
        b, loss, _ = M.fit_logit(x)
        np.testing.assert_allclose(b, 0)
        self.assertAlmostEqual(loss, 5 * np.log(65))

    def test_recovers_known_preference_direction(self):
        rng = np.random.default_rng(71)
        x = rng.normal(size=(1200, 8, 2))
        utility = x @ np.array([.7, -.6])
        prob = np.exp(utility - utility.max(axis=1, keepdims=True))
        prob /= prob.sum(axis=1, keepdims=True)
        chosen = np.array([rng.choice(8, p=p) for p in prob])
        for i, pick in enumerate(chosen):
            x[i, [0, pick]] = x[i, [pick, 0]]
        estimate, _, _ = M.fit_logit(x)
        self.assertGreater(estimate[0], .4)
        self.assertLess(estimate[1], -.3)

    def test_newton_fallback(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=(40, 10, 2))
        failed = SimpleNamespace(success=False, jac=np.ones(2), x=np.zeros(2), fun=100)
        with patch.object(M, 'minimize', return_value=failed):
            beta, loss, ok = M.fit_logit(x)
        self.assertTrue(ok and np.isfinite(loss))
        self.assertLess(np.linalg.norm(M.conditional_objective(beta, x)[1]), 1e-3)

    def test_negative_pool_excludes_whole_batch(self):
        state = {'method': 'a', 'seed': 157, 'round': 0}
        values = np.zeros((100, 4))
        values[:, 2] = np.arange(100)
        mask = np.arange(100) < 32
        x, _ = M.choice_array([(state, values, mask)], ['D_hamming'])
        self.assertTrue((x[:, 0, 0] < 32).all())
        self.assertTrue((x[:, 1:, 0] >= 32).all())
        self.assertTrue(all(len(set(v[:, 0])) == 64 for v in x[:, 1:]))

    def test_tie_percentiles(self):
        np.testing.assert_allclose(M.percentile([0, 0, 1, 3]), [.375, .375, .75, 1])

    def test_hamming_coverage_monotone(self):
        meta = {str(i): {'smiles': 'A' if i < 2 else 'B', 'conditions': {'PE/EA': '1/1',
                'Density g/ml': 1, 'V/ul': i, 'loading solvent': 'PE', 'Volume of loading solvent/ul': 200}}
                for i in range(3)}
        _, _, d = M.metadata_space(meta)
        before = d[:, [0]].min(axis=1)
        after = d[:, [0, 1]].min(axis=1)
        self.assertTrue((after <= before).all())


if __name__ == '__main__':
    unittest.main()
