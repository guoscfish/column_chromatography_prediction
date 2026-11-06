"""Small exact linear GP with explicit likelihood noise and float64 algebra."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.stats import norm, spearmanr


def stable_cholesky(matrix):
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise ValueError("finite square covariance required")
    if not np.allclose(a, a.T, atol=1e-11, rtol=1e-10):
        raise ValueError("asymmetric covariance")
    scale = max(1.0, float(np.max(np.diag(a))))
    if np.linalg.eigvalsh(a).min() < -1e-9 * scale:
        raise ValueError("covariance is materially indefinite")
    for relative in (0.0, 1e-12, 1e-10, 1e-8):
        jitter = relative * scale
        try:
            return np.linalg.cholesky(a + jitter * np.eye(len(a))), jitter
        except np.linalg.LinAlgError:
            pass
    raise ValueError("Cholesky failed within declared jitter budget")


class LinearGP:
    """Fit amplitude/noise by training evidence; never inspect evaluation labels."""

    def fit(self, features, targets, *, mean=None, bounds=(-9.0, 9.0)):
        self.x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if self.x.ndim != 2 or y.shape != (len(self.x),) or len(y) < 2:
            raise ValueError("training X/y dimensions are invalid")
        if not np.isfinite(self.x).all() or not np.isfinite(y).all():
            raise ValueError("nonfinite training values")
        self.mean = float(y.mean() if mean is None else mean)
        centered = y - self.mean
        kernel = self.x @ self.x.T
        eigenvalues, eigenvectors = np.linalg.eigh(kernel)
        eigenvalues = np.maximum(eigenvalues, 0)
        projections = eigenvectors.T @ centered

        def objective(log_parameters):
            signal, noise = np.exp(log_parameters)
            variances = signal * eigenvalues + noise
            return float(0.5 * np.sum(np.log(2 * np.pi * variances) + projections**2 / variances))

        # Multiple fixed starts address numerical/local optimization failure, not holdout selection.
        optimizers = [minimize(objective, start, bounds=[bounds, bounds], method="L-BFGS-B")
                      for start in ((0., -2.), (2., -1.), (-2., 0.))]
        valid = [r for r in optimizers if r.success and np.isfinite(r.fun)]
        if not valid:
            raise ValueError("all GP evidence optimizations failed")
        result = min(valid, key=lambda r: r.fun)
        self.signal, self.noise = np.exp(result.x)
        self.cholesky, jitter = stable_cholesky(self.signal * kernel + self.noise * np.eye(len(y)))
        self.alpha = cho_solve((self.cholesky, True), centered)
        self.audit = {"signal_variance": float(self.signal), "noise_variance": float(self.noise),
                      "numerical_jitter": jitter, "negative_log_marginal_likelihood": float(result.fun),
                      "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
                      "boundary_hit": bool(np.any(np.isclose(result.x, bounds[0], atol=1e-4)) or
                                           np.any(np.isclose(result.x, bounds[1], atol=1e-4))),
                      "training_rows": len(y), "feature_dimension": self.x.shape[1],
                      "constant_mean": self.mean, "dtype": "float64", "kernel": "linear"}
        return self

    def predict(self, features, *, full_covariance=False):
        query = np.asarray(features, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != self.x.shape[1] or not np.isfinite(query).all():
            raise ValueError("query features mismatch")
        cross = self.signal * (query @ self.x.T)
        projected = solve_triangular(self.cholesky, cross.T, lower=True)
        mean = self.mean + cross @ self.alpha
        if full_covariance:
            covariance = self.signal * (query @ query.T) - projected.T @ projected
            covariance = (covariance + covariance.T) / 2
            minimum = float(np.linalg.eigvalsh(covariance).min())
            if minimum < -1e-8 * max(1., self.signal):
                raise ValueError("invalid posterior covariance")
            return mean, covariance
        variance = self.signal * np.einsum("ij,ij->i", query, query) - np.sum(projected**2, axis=0)
        if variance.min() < -1e-8 * max(1., self.signal):
            raise ValueError("negative posterior variance")
        return mean, np.maximum(variance, 0.)


def probability_metrics(target, mean, observation_variance):
    y, mu, v = (np.asarray(a, dtype=np.float64) for a in (target, mean, observation_variance))
    if y.shape != mu.shape or y.shape != v.shape or np.any(v <= 0) or not all(np.isfinite(a).all() for a in (y, mu, v)):
        raise ValueError("invalid probability evaluation arrays")
    z = (y - mu) / np.sqrt(v)
    return {"n": len(y), "rmse": float(np.sqrt(np.mean((y-mu)**2))),
            "nlpd": float(np.mean(0.5 * (np.log(2*np.pi*v) + z*z))),
            "coverage_50": float(np.mean(np.abs(z) <= norm.ppf(.75))),
            "coverage_80": float(np.mean(np.abs(z) <= norm.ppf(.9))),
            "coverage_90": float(np.mean(np.abs(z) <= norm.ppf(.95))),
            "coverage_95": float(np.mean(np.abs(z) <= norm.ppf(.975))),
            "mean_interval_width_90": float(np.mean(2*norm.ppf(.95)*np.sqrt(v))),
            "standardized_error_rms": float(np.sqrt(np.mean(z*z))),
            "pit_mean": float(np.mean(norm.cdf(z))),
            "pit_deciles": np.histogram(norm.cdf(z), bins=np.linspace(0,1,11))[0].tolist()}


def rank_correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def fixed_kernel_signals(features, labeled_count):
    """Match historical IVR's full-universe RMS and unit prior/noise algebra."""
    x = np.asarray(features, dtype=np.float64)
    x = x / np.sqrt(np.mean(np.sum(x*x, axis=1)))
    chol, _ = stable_cholesky(np.eye(x.shape[1]) + x[:labeled_count].T @ x[:labeled_count])
    covariance = cho_solve((chol, True), np.eye(x.shape[1]))
    moment = x.T @ x / len(x)
    q = x @ covariance
    variance = np.einsum("ij,ij->i", q, x)
    scores = np.einsum("ij,ij->i", q, q @ moment) / (1 + variance)
    return variance, scores


def batch_information(covariance, noise_variance):
    """I(y_B; f_B | D_L) under independent Gaussian observation noise."""
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or len(covariance) == 0 or not np.isfinite(covariance).all():
        raise ValueError("finite nonempty covariance required")
    if not np.allclose(covariance, covariance.T, atol=1e-11, rtol=1e-10):
        raise ValueError("asymmetric posterior covariance")
    if np.linalg.eigvalsh(covariance).min() < -1e-9 * max(1., float(np.max(np.diag(covariance)))):
        raise ValueError("posterior covariance is not positive semidefinite")
    if not np.isfinite(noise_variance) or noise_variance <= 0:
        raise ValueError("strictly positive observation noise required")
    chol, jitter = stable_cholesky(np.eye(len(covariance)) + covariance / noise_variance)
    return float(np.log(np.diag(chol)).sum()), jitter
