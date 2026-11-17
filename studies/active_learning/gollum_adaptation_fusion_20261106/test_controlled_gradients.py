"""Audit the real new SciPy/Torch closures without editing frozen study code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gp_training as TRAINING
from test_legacy_gradients import captured_fit, finite_differences

SOURCE_HASH = hashlib.sha256((HERE / "gp_training.py").read_bytes()).hexdigest()
RESULTS = {"source": "gp_training.py", "source_sha256": SOURCE_HASH,
           "scope": "Actual intercepted optimizer closures; synthetic float64 systems only",
           "finite_difference": {}, "kernel_consistency": {}}


def shifted(vector):
    value = np.array(vector, dtype=np.float64, copy=True)
    value[:-3] += .17
    value[-3:] = [-.3, -2.4, .4]
    return value


def capture_controlled(method):
    rng = np.random.default_rng(9071)
    x = rng.normal(size=(13, 5))
    y = 8.3 + 2.7 * np.sin(x[:, 0]) + .4 * x[:, 1] + .2 * rng.normal(size=len(x))
    calls = []

    def fake_minimize(objective, initial, **kwargs):
        parameters = shifted(initial)
        value, gradient = objective(parameters)
        calls.append({"objective": objective, "parameters": parameters, "gradient": gradient,
                      "value": value, "arguments": kwargs})
        return SimpleNamespace(x=parameters, fun=value, success=False,
                               message="Synthetic audit interception; no optimization requested",
                               nit=0, nfev=1)

    model = TRAINING.ControlledProjectionGP(method=method, projection_dim=3, max_iter=2,
                                           seed=17, ridge=.01, noise_floor=.02, stage_iterations=2)
    with patch.object(TRAINING, "minimize", fake_minimize):
        model.fit(x, y)
    return model, calls, x, y


class ControlledGradientTests(unittest.TestCase):
    def _check(self, method):
        gp, calls, x, y = capture_controlled(method)
        self.assertEqual(len(calls), 2 if method == "staged" else 1)
        for stage, capture in enumerate(calls):
            parameters, objective = capture["parameters"], capture["objective"]
            value, analytic = objective(parameters)
            numerical = finite_differences(objective, parameters)
            np.testing.assert_allclose(analytic, numerical, rtol=3e-5, atol=3e-6)
            error = np.abs(analytic - numerical)
            RESULTS["finite_difference"][f"{method}_stage{stage+1}"] = {
                "parameter_count": len(parameters), "objective": value,
                "max_absolute_error": float(error.max()),
                "max_scaled_error": float(np.max(error / np.maximum(1., np.maximum(abs(analytic), abs(numerical))))),
                "analytic_gradient_norm": float(np.linalg.norm(analytic)), "passed": True}
        final_call = calls[-1]
        matrices = []
        actual_cholesky = torch.linalg.cholesky_ex

        def capture_cholesky(matrix, *args, **kwargs):
            matrices.append(matrix.detach().numpy().copy())
            return actual_cholesky(matrix, *args, **kwargs)

        with patch.object(torch.linalg, "cholesky_ex", capture_cholesky):
            objective_value = final_call["objective"](final_call["parameters"])[0]
        self.assertEqual(len(matrices), 1)
        final_covariance = (gp.signal * gp._base_kernel(gp.x, gp.x)
                            + gp.noise * np.eye(len(x))) / gp.target_scale**2
        np.testing.assert_allclose(matrices[0], final_covariance, rtol=2e-12, atol=2e-12)
        penalty = (.5 * .01 * x.shape[1] * np.sum(final_call["parameters"][:-3]**2)
                   if method == "regularized" else 0.)
        raw_nll = objective_value - penalty + len(y) * np.log(gp.target_scale)
        self.assertAlmostEqual(raw_nll, gp.audit["negative_log_marginal_likelihood"], places=10)
        np.testing.assert_allclose(gp.embedding(x), gp.x, rtol=2e-14, atol=2e-14)
        if method == "noise_floor":
            lower = calls[0]["arguments"]["bounds"][-2][0]
            self.assertAlmostEqual(lower, np.log(.02))
        if method == "staged":
            self.assertEqual(len(calls[1]["parameters"]), 3)
            np.testing.assert_array_equal(gp.projection_weight.ravel(), calls[0]["parameters"][:15])
            np.testing.assert_array_equal(gp.projection_bias, calls[0]["parameters"][15:-3])
        RESULTS["kernel_consistency"][method] = {
            "covariance_error_standardized": float(np.max(abs(matrices[0] - final_covariance))),
            "raw_nll_jacobian_error": float(abs(raw_nll - gp.audit["negative_log_marginal_likelihood"])),
            "penalty": float(penalty), "train_query_transform_consistent": True,
            "fixed_representation_in_second_stage": method == "staged", "passed": True}

    def test_original_actual_closure(self):
        self._check("original")

    def test_converged_actual_closure(self):
        self._check("converged")

    def test_regularized_actual_closure(self):
        self._check("regularized")

    def test_noise_floor_actual_closure(self):
        self._check("noise_floor")

    def test_staged_actual_closures(self):
        self._check("staged")

    def test_original_objective_matches_unchanged_legacy(self):
        gp, calls, _, _ = capture_controlled("original")
        with captured_fit(True, shifted) as legacy:
            np.testing.assert_array_equal(calls[0]["parameters"], legacy["parameters"])
            old_value, old_gradient = legacy["objective"](legacy["parameters"])
            new_value, new_gradient = calls[0]["objective"](calls[0]["parameters"])
            self.assertAlmostEqual(old_value, new_value, places=12)
            np.testing.assert_allclose(old_gradient, new_gradient, rtol=2e-12, atol=2e-12)
            RESULTS["original_matches_legacy"] = True


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ControlledGradientTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    RESULTS.update(test_count=result.testsRun, success=result.wasSuccessful(), failures=len(result.failures),
                   errors=len(result.errors), source_unchanged=(hashlib.sha256((HERE / "gp_training.py").read_bytes()).hexdigest() == SOURCE_HASH))
    (HERE / "controlled_gradient_results.json").write_text(json.dumps(RESULTS, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
