"""Synthetic-only mathematical tests for the representation GP extension."""
import inspect
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gp_models import (FixedRepresentationGP, ProjectedMaternGP, acquisition_diagnostics,
                       batch_information, regression_metrics, stable_cholesky)


class RepresentationGPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(917)
        cls.features = rng.normal(size=(35, 6))
        cls.responses = 4 + 2 * np.sin(cls.features[:, 0]) + .3 * rng.normal(size=35)
        cls.model = FixedRepresentationGP(max_iter=12).fit(cls.features[:22], cls.responses[:22])

    def test_matches_direct_conditioning_in_raw_response_units(self):
        gp = self.model
        query = self.features[22:]
        mean, covariance = gp.predict(query, full_covariance=True)
        q = gp.transform(query)
        train = gp.signal * gp._base_kernel(gp.x, gp.x) + gp.noise * np.eye(len(gp.x))
        cross = gp.signal * gp._base_kernel(q, gp.x)
        direct_mean = gp.mean + cross @ np.linalg.solve(train, self.responses[:22] - gp.mean)
        direct_cov = gp.signal * gp._base_kernel(q, q) - cross @ np.linalg.solve(train, cross.T)
        np.testing.assert_allclose(mean, direct_mean, atol=1e-9)
        np.testing.assert_allclose(covariance, direct_cov, atol=1e-9)
        self.assertEqual(gp.x.dtype, np.float64)
        self.assertGreater(gp.noise, 0)
        self.assertGreater(gp.lengthscale, 0)

    def test_training_only_feature_scaling(self):
        gp = self.model
        np.testing.assert_array_equal(gp.feature_mean, self.features[:22].mean(axis=0))
        rms = np.sqrt(np.mean((self.features[:22] - self.features[:22].mean(axis=0))**2))
        np.testing.assert_array_equal(gp.feature_scale, np.full(6, rms))
        before_mean, before_variance = gp.predict(self.features[22:])
        future_labels = self.responses[22:].copy()
        future_labels[:] = 1e12
        gp.predict(self.features[22:] + 10000)
        after_mean, after_variance = gp.predict(self.features[22:])
        np.testing.assert_array_equal(before_mean, after_mean)
        np.testing.assert_array_equal(before_variance, after_variance)
        self.assertEqual(set(inspect.signature(gp.fit).parameters), {"features", "targets"})

    def test_observation_variance_adds_noise_once(self):
        mean, latent, observation = self.model.predict_observation(self.features[22:])
        np.testing.assert_allclose(observation - latent, self.model.noise)
        _, latent_cov, observation_cov = self.model.predict_observation(self.features[22:], full_covariance=True)
        np.testing.assert_allclose(observation_cov - latent_cov, self.model.noise * np.eye(len(mean)))
        np.testing.assert_allclose(np.diag(latent_cov), latent)

    def test_constant_features_and_responses_remain_finite(self):
        gp = FixedRepresentationGP(max_iter=4).fit(np.ones((8, 3)), np.full(8, 9.))
        mu, var = gp.predict(np.ones((2, 3)))
        np.testing.assert_allclose(mu, 9.)
        self.assertTrue(np.all(np.isfinite(var)))
        self.assertEqual(gp.audit["constant_feature_count"], 3)

    def test_projection_parameters_learn_without_evaluation_data(self):
        gp = ProjectedMaternGP(projection_dim=4, max_iter=7, seed=17).fit(self.features[:18], self.responses[:18])
        weight0 = np.random.default_rng(17).normal(0, 1 / np.sqrt(6), size=(6, 4))
        self.assertGreater(np.linalg.norm(weight0 - gp.projection_weight), 1e-5)
        self.assertEqual(gp.audit["projection_parameters"], 28)
        self.assertLessEqual(gp.audit["loss_history_standardized"][-1], gp.audit["loss_history_standardized"][0])
        mean, covariance = gp.predict(self.features[22:], full_covariance=True)
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-8)
        gp2 = ProjectedMaternGP(projection_dim=4, max_iter=7, seed=17).fit(self.features[:18], self.responses[:18])
        np.testing.assert_array_equal(gp.projection_weight, gp2.projection_weight)
        np.testing.assert_array_equal(mean, gp2.predict(self.features[22:])[0])

    def test_linear_control_reuses_phase0(self):
        gp = FixedRepresentationGP(kernel="linear", max_iter=4).fit(self.features[:20], self.responses[:20])
        self.assertIn("Phase0", gp.audit["optimizer"]["method"])
        self.assertEqual(gp.audit["optimizer"]["starts"], 3)
        mu, var = gp.predict(self.features[20:])
        self.assertTrue(np.all(np.isfinite(mu)))
        self.assertTrue(np.all(var >= 0))

    def test_marginal_likelihood_record_matches_cholesky(self):
        gp = self.model
        residual = self.responses[:22] - gp.mean
        nll = .5 * residual @ gp.alpha + np.log(np.diag(gp.cholesky)).sum() + .5 * len(residual) * np.log(2 * np.pi)
        self.assertAlmostEqual(nll, gp.audit["negative_log_marginal_likelihood"])
        self.assertEqual(gp.audit["marginal_log_likelihood"], -nll)
        self.assertGreater(gp.audit["training_seconds"], 0)
        self.assertEqual(gp.audit["dtype"], "float64")

    def test_independent_target_instances_do_not_share_parameters(self):
        second = FixedRepresentationGP(max_iter=8).fit(self.features[:22], -3 * self.responses[:22] + 7)
        self.assertNotEqual(self.model.mean, second.mean)
        self.assertFalse(np.shares_memory(self.model.alpha, second.alpha))

    def test_maxvar_ivr_and_batch_information_are_label_free(self):
        result = acquisition_diagnostics(self.model, self.features, batch_size=5, candidate_limit=20, seed=31)
        repeat = acquisition_diagnostics(self.model, self.features, batch_size=5, candidate_limit=20, seed=31)
        self.assertEqual(result, repeat)
        self.assertEqual(len(result["candidate_positions"]), 20)
        self.assertEqual(result["candidate_count"], 35)
        positions = np.array(result["candidate_positions"])
        _, covariance = self.model.predict(self.features[positions], full_covariance=True)
        var = np.diag(covariance)
        first_ivr = np.sum(covariance**2, axis=0) / (len(positions) * (var + self.model.noise))
        self.assertEqual(result["methods"]["maxvar"]["selected_candidate_positions"][0], int(positions[np.argmax(var)]))
        self.assertEqual(result["methods"]["conditional_ivr"]["selected_candidate_positions"][0], int(positions[np.argmax(first_ivr)]))
        for method in result["methods"].values():
            self.assertEqual(len(method["selected_candidate_positions"]), 5)
            self.assertEqual(len(set(method["selected_candidate_positions"])), 5)
        trace = result["methods"]["batch_information_gain"]["trace"]
        joint = result["methods"]["batch_information_gain"]["batch_information"]
        self.assertAlmostEqual(sum(t["score"] for t in trace), joint, places=9)

    def test_conditional_ivr_matches_explicit_variance_reduction(self):
        result = acquisition_diagnostics(self.model, self.features[22:], batch_size=2)
        trace = result["methods"]["conditional_ivr"]["trace"]
        _, covariance = self.model.predict(self.features[22:], full_covariance=True)
        first = trace[0]["candidate_position"]
        after = covariance - np.outer(covariance[:, first], covariance[:, first]) / (covariance[first, first] + self.model.noise)
        expected = np.mean(np.diag(covariance) - np.diag(after))
        self.assertAlmostEqual(trace[0]["score"], expected)
        second = trace[1]["candidate_position"]
        score = np.sum(after[:, second]**2) / (len(after) * (after[second, second] + self.model.noise))
        self.assertAlmostEqual(trace[1]["score"], score)

    def test_reuses_phase0_metrics_and_cholesky_rejection(self):
        metrics = regression_metrics([1., 2., 3.], [1., 1., 2.], [1., 1., 1.])
        self.assertAlmostEqual(metrics["mae"], 2 / 3)
        self.assertIn("nlpd", metrics)
        self.assertIn("coverage_95", metrics)
        with self.assertRaises(ValueError):
            stable_cholesky(np.diag([-1., 1.]))
        duplicate, _ = batch_information(np.ones((2, 2)), 1.)
        independent, _ = batch_information(np.eye(2), 1.)
        self.assertLess(duplicate, independent)

    def test_invalid_shapes_fail(self):
        with self.assertRaises(ValueError):
            FixedRepresentationGP().fit(np.ones((3, 2)), [1., 2.])
        with self.assertRaises(ValueError):
            self.model.predict(np.ones((4, 5)))
        with self.assertRaises(ValueError):
            FixedRepresentationGP(starts=2)
        with self.assertRaises(ValueError):
            acquisition_diagnostics(self.model, self.features, candidate_limit=0)


if __name__ == "__main__":
    unittest.main()
