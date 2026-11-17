"""Probe the unchanged legacy optimizer closure using synthetic data only.

Run with the existing chromatography Python and the recorded single-thread
OpenMP environment. No production response table, checkpoint, or test is read.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LEGACY_PATH = HERE.parent / "representation_gp_benchmark_20261104" / "gp_models.py"
SPEC = importlib.util.spec_from_file_location("unchanged_legacy_gp_under_test", LEGACY_PATH)
LEGACY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEGACY)
RESULTS = {"legacy_source": str(LEGACY_PATH.relative_to(HERE.parents[2])),
           "legacy_source_sha256": hashlib.sha256(LEGACY_PATH.read_bytes()).hexdigest(),
           "scope": "synthetic-only actual legacy closure; no historical responses read",
           "finite_difference": {}, "kernel_consistency": {}, "limitations": [
               "Finite differences cover small synthetic float64 systems, not every historical fitted point.",
               "The zero-jitter branch is tested; gradients through a changing jitter branch are not claimed.",
               "No claim of convergence, calibration, or downstream AL performance follows from these tests."]}


@contextmanager
def captured_fit(projected=False, parameter_transform=None):
    """Intercept SciPy rather than reimplementing the objective being tested."""
    rng = np.random.default_rng(9071)
    x = rng.normal(size=(13, 5))
    y = 8.3 + 2.7 * np.sin(x[:, 0]) + .4 * x[:, 1] + .2 * rng.normal(size=len(x))
    capture = {}

    def fake_minimize(objective, initial, **kwargs):
        parameters = np.array(initial, dtype=np.float64, copy=True)
        if parameter_transform:
            parameters = parameter_transform(parameters)
        value, gradient = objective(parameters)
        capture.update(objective=objective, parameters=parameters, gradient=gradient,
                       value=value, optimizer_arguments=kwargs)
        return SimpleNamespace(x=parameters, fun=value, success=False,
                               message="Synthetic audit interception; no optimization requested",
                               nit=0, nfev=1)

    gp = (LEGACY.ProjectedMaternGP(projection_dim=3, max_iter=2, seed=17)
          if projected else LEGACY.FixedRepresentationGP(max_iter=2, seed=17))
    with patch.object(LEGACY, "minimize", fake_minimize):
        gp.fit(x, y)
    capture.update(model=gp, features=x, targets=y)
    yield capture


def finite_differences(objective, parameters, step=1e-5):
    """Central differences call the exact function handed to SciPy."""
    values = []
    for coordinate in range(len(parameters)):
        plus, minus = parameters.copy(), parameters.copy()
        h = step * max(1., abs(parameters[coordinate]))
        plus[coordinate] += h
        minus[coordinate] -= h
        values.append((objective(plus)[0] - objective(minus)[0]) / (2 * h))
    return np.asarray(values)


class LegacyGradientTests(unittest.TestCase):
    def _check_gradient(self, projected, shifted):
        def transform(vector):
            vector[-3:] = [-.3, -2.4, .4]
            if projected:
                vector[:-3] += .17
            return vector

        with captured_fit(projected, transform if shifted else None) as capture:
            parameters = capture["parameters"]
            value, analytic = capture["objective"](parameters)
            numerical = finite_differences(capture["objective"], parameters)
            error = np.abs(analytic - numerical)
            scaled_error = error / np.maximum(1., np.maximum(np.abs(analytic), np.abs(numerical)))
            np.testing.assert_allclose(analytic, numerical, atol=2e-6, rtol=2e-5)
            groups = {"log_signal": [-3], "log_noise": [-2], "log_lengthscale": [-1]}
            if projected:
                groups.update(projection_weight=list(range(15)), projection_bias=list(range(15, 18)))
            key = ("projected" if projected else "fixed") + ("_shifted" if shifted else "_initial")
            RESULTS["finite_difference"][key] = {
                "objective": value, "parameter_count": len(parameters), "step": 1e-5,
                "max_absolute_error": float(error.max()), "max_scaled_error": float(scaled_error.max()),
                "groups": {name: {"count": len(indices),
                    "analytic_gradient_norm": float(np.linalg.norm(analytic[indices])),
                    "max_absolute_error": float(error[indices].max()),
                    "max_scaled_error": float(scaled_error[indices].max())}
                    for name, indices in groups.items()}, "passed": True}

    def test_fixed_initial_gradients(self):
        self._check_gradient(False, False)

    def test_fixed_shifted_gradients(self):
        self._check_gradient(False, True)

    def test_projection_initial_gradients(self):
        self._check_gradient(True, False)

    def test_projection_shifted_gradients(self):
        self._check_gradient(True, True)

    def _check_kernel(self, projected):
        def transform(vector):
            vector[-3:] = [.2, -2.8, -.25]
            if projected:
                vector[:-3] += .21
            return vector

        with captured_fit(projected, transform) as capture:
            matrices = []
            actual_cholesky = torch.linalg.cholesky_ex

            def capture_cholesky(matrix, *args, **kwargs):
                matrices.append(matrix.detach().numpy().copy())
                return actual_cholesky(matrix, *args, **kwargs)

            with patch.object(torch.linalg, "cholesky_ex", capture_cholesky):
                torch_loss = capture["objective"](capture["parameters"])[0]
            self.assertEqual(len(matrices), 1)
            gp, x, y = capture["model"], capture["features"], capture["targets"]
            numpy_covariance = (gp.signal * gp._base_kernel(gp.x, gp.x)
                                + gp.noise * np.eye(len(x))) / gp.target_scale**2
            matrix_error = np.max(np.abs(matrices[0] - numpy_covariance))
            np.testing.assert_allclose(matrices[0], numpy_covariance, atol=2e-12, rtol=2e-12)
            # Standardization contributes exactly n * log(target_scale) in raw units.
            nll_adjusted = torch_loss + len(y) * np.log(gp.target_scale)
            nll_error = abs(nll_adjusted - gp.audit["negative_log_marginal_likelihood"])
            self.assertLess(nll_error, 2e-10)
            np.testing.assert_allclose(gp.embedding(x), gp.x, atol=1e-14, rtol=1e-14)
            query = x[:4] + .13
            z_query = gp.embedding(query)
            train = gp.signal * gp._base_kernel(gp.x, gp.x) + gp.noise * np.eye(len(x))
            cross = gp.signal * gp._base_kernel(z_query, gp.x)
            direct_mean = gp.mean + cross @ np.linalg.solve(train, y - gp.mean)
            direct_covariance = (gp.signal * gp._base_kernel(z_query, z_query)
                                 - cross @ np.linalg.solve(train, cross.T))
            mean, latent, observation = gp.predict_observation(query, full_covariance=True)
            np.testing.assert_allclose(mean, direct_mean, atol=2e-11, rtol=2e-11)
            np.testing.assert_allclose(latent, direct_covariance, atol=2e-11, rtol=2e-11)
            np.testing.assert_allclose(observation - latent, gp.noise * np.eye(4), atol=1e-13)
            RESULTS["kernel_consistency"]["projected" if projected else "fixed"] = {
                "max_training_covariance_absolute_error_standardized": float(matrix_error),
                "raw_nll_jacobian_error": float(nll_error),
                "max_posterior_mean_absolute_error": float(np.max(np.abs(mean - direct_mean))),
                "max_posterior_covariance_absolute_error": float(np.max(np.abs(latent - direct_covariance))),
                "jitter": gp.audit["numerical_jitter"], "observation_noise_added_once": True,
                "train_query_transform_consistent": True, "passed": True}

    def test_fixed_torch_numpy_kernel_and_posterior_consistency(self):
        self._check_kernel(False)

    def test_projection_torch_numpy_kernel_and_posterior_consistency(self):
        self._check_kernel(True)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LegacyGradientTests)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    RESULTS["test_count"] = outcome.testsRun
    RESULTS["success"] = outcome.wasSuccessful()
    RESULTS["failures"] = len(outcome.failures)
    RESULTS["errors"] = len(outcome.errors)
    RESULTS["legacy_source_unchanged"] = (hashlib.sha256(LEGACY_PATH.read_bytes()).hexdigest()
                                           == RESULTS["legacy_source_sha256"])
    (HERE / "legacy_gradient_results.json").write_text(json.dumps(RESULTS, indent=2) + "\n")
    raise SystemExit(0 if outcome.wasSuccessful() else 1)
