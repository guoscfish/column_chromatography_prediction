"""Numerical, nesting, covariance, and feature-boundary checks for fusion."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_fusion import AdditiveKernelGP, _old


class AdditiveKernelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(53)
        self.g = rng.normal(size=(25, 4))
        self.l = rng.normal(size=(25, 6))
        self.x = np.c_[self.g, self.l]
        self.y = np.sin(self.g[:20, 0]) + .3*self.l[:20, 1] + rng.normal(0, .1, 20)

    def model(self, mode="learned", **kwargs):
        return AdditiveKernelGP(gnn_dimension=4, lambda_mode=mode, max_iter=200, **kwargs).fit(self.x[:20], self.y)

    def test_lambda_zero_exact_gnn_nesting(self):
        fusion = self.model("zero")
        baseline = _old.FixedRepresentationGP(max_iter=200).fit(self.g[:20], self.y)
        a, va = fusion.predict(self.x[20:])
        b, vb = baseline.predict(self.g[20:])
        np.testing.assert_allclose(a, b, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(va, vb, atol=1e-7, rtol=1e-7)
        self.assertEqual(fusion.lambda_value, 0.)
        self.assertEqual(fusion.audit["scalar_parameter_count"], 3)

    def test_psd_unit_component_diagonals(self):
        model = self.model("fixed")
        kernel = model._base_kernel(model.x, model.x)
        np.testing.assert_allclose(np.diag(kernel), 2., atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(kernel).min(), -1e-10)
        self.assertEqual(model.lambda_value, 1.)

    def test_mll_gradient_numerical_difference(self):
        model = self.model()
        point = np.array([.2, -1.4, .3, -.2, -.5])
        _, analytic = model._objective(point)
        numerical = []
        for j in range(len(point)):
            plus, minus = point.copy(), point.copy()
            plus[j] += 1e-5
            minus[j] -= 1e-5
            numerical.append((model._objective(plus)[0]-model._objective(minus)[0])/2e-5)
        np.testing.assert_allclose(analytic, numerical, rtol=2e-5, atol=2e-5)

    def test_latent_observation_noise_separation(self):
        model = self.model()
        mean, latent, observation = model.predict_observation(self.x[20:])
        np.testing.assert_allclose(observation-latent, model.noise, rtol=1e-12, atol=1e-12)
        _, full = model.predict(self.x[20:], full_covariance=True)
        np.testing.assert_allclose(np.diag(full), latent, rtol=1e-9, atol=1e-9)
        self.assertGreaterEqual(np.linalg.eigvalsh(full).min(), -1e-9)

    def test_capacity_control_has_no_language_dependency(self):
        model = self.model(capacity_control=True)
        changed = self.x[20:].copy()
        changed[:, 4:] += 1e6
        a, va = model.predict(self.x[20:])
        b, vb = model.predict(changed)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(va, vb)
        self.assertEqual(model.audit["scalar_parameter_count"], 5)

    def test_scaler_uses_training_only_and_query_cannot_mutate(self):
        model = self.model()
        means, scales = model.feature_mean.copy(), model.feature_scale.copy()
        model.predict(self.x[20:]*1e4)
        np.testing.assert_array_equal(means, model.feature_mean)
        np.testing.assert_array_equal(scales, model.feature_scale)
        np.testing.assert_allclose(means, self.x[:20].mean(0))

    def test_invalid_lambda_and_features_rejected(self):
        with self.assertRaises(ValueError):
            AdditiveKernelGP(gnn_dimension=4, fixed_lambda=-1)
        with self.assertRaises(ValueError):
            AdditiveKernelGP(gnn_dimension=4).fit(self.g[:20], self.y)

    def test_duplicate_feature_rows_remain_numerically_stable(self):
        x = self.x[:20].copy()
        x[1] = x[0]
        model = AdditiveKernelGP(gnn_dimension=4, lambda_mode="fixed", max_iter=200).fit(x, self.y)
        mean, var = model.predict(x)
        self.assertTrue(np.isfinite(mean).all() and np.isfinite(var).all())
        self.assertGreater(model.audit["minimum_observation_eigenvalue"], 0.)

    def test_trajectory_seed_not_overwritten_by_optimizer_initialization(self):
        from run_phase3 import fit_record
        record = fit_record({"seed": 887, "budget": 813, "initialization": 17}, {"seed": 17, "noise_variance": 1.})
        self.assertEqual(record["seed"], 887)
        self.assertEqual(record["optimizer_seed"], 17)

    def test_sealed_feature_identity_and_test_exclusion(self):
        from run_phase3 import load_state
        state, data, x, xb, audit = load_state(157, 333)
        forbidden = set(state.partition.loc[state.partition.role.eq("test"), "canonical_index"].astype(int))
        self.assertFalse(forbidden & set(data["indices"].astype(int)))
        self.assertEqual(x.shape, xb.shape)
        self.assertEqual(len(audit["candidate_ids"]), 256)
        with self.assertRaises(PermissionError):
            state.validate_indices([next(iter(forbidden))])

    def test_feature_cache_hash_tampering_is_rejected(self):
        from unittest.mock import patch
        import run_phase3
        with patch.object(run_phase3, "file_hash", return_value="tampered"):
            with self.assertRaisesRegex(ValueError, "cache integrity"):
                run_phase3.load_state(157, 333)

    def test_posterior_reconstruction_without_refitting(self):
        model = self.model()
        restored = AdditiveKernelGP.restore_from_audit(self.x[:20], self.y, model.audit)
        for a, b in zip(model.predict(self.x[20:]), restored.predict(self.x[20:])):
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
