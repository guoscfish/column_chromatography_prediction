"""Train-only Gaussian probability surrogates for prediction-oriented AL.

These models consume *only* explicitly supplied observed training labels.  They
do not know about a label store, validation set, or candidate labels.  GP output
channels are independent; the optional matrix-normal linear model has genuine
cross-output covariance and states its restrictive separability assumption.
All predictive values and covariance blocks are in the supplied physical units.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import ndtri


def _matrix(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2 or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite nonempty matrix")
    return result


def _prepare_y(y, mean, scales):
    y = _matrix(y, "y")
    mean = np.zeros_like(y) if mean is None else _matrix(mean, "mean_train")
    if mean.shape != y.shape:
        raise ValueError("mean_train and y shapes differ")
    scales = np.std(y, axis=0, ddof=1) if scales is None else np.asarray(scales, dtype=np.float64)
    scales = np.atleast_1d(scales)
    if scales.shape != (y.shape[1],) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("output_scales must be one positive scale per output")
    z = (y - mean) / scales
    center = z.mean(axis=0)
    return z - center, scales, center


@dataclass
class FeatureTransform:
    """Standardize and optionally reduce features using this fit's X only."""

    max_components: int | None = 16

    def fit(self, x):
        x = _matrix(x, "X")
        self.center = x.mean(axis=0)
        raw_scale = x.std(axis=0)
        self.scale = np.where(raw_scale > 1e-12, raw_scale, 1.0)
        z = (x - self.center) / self.scale
        components = min(z.shape[1], z.shape[0] - 1)
        if self.max_components is not None:
            if not isinstance(self.max_components, int) or self.max_components < 1:
                raise ValueError("max_components must be positive or None")
            components = min(components, self.max_components)
        if components < x.shape[1]:
            _, singular, vt = np.linalg.svd(z, full_matrices=False)
            self.rotation = vt[:components].T
            total = float(np.square(singular).sum())
            self.retained_variance_fraction = float(np.square(singular[:components]).sum() / total) if total else 1.0
        else:
            self.rotation = np.eye(x.shape[1])
            self.retained_variance_fraction = 1.0
        self.input_dimension = x.shape[1]
        return self

    def transform(self, x):
        x = _matrix(x, "X")
        if x.shape[1] != self.input_dimension:
            raise ValueError("X feature dimension differs from training")
        return ((x - self.center) / self.scale) @ self.rotation


class GaussianPosterior:
    """On-demand covariance over input rows; row-major/output-minor ordering.

    ``cov(..., noisy=True)`` denotes the *same* observation if indices coincide;
    noise is added on intersecting row positions. For distinct future replicates
    at the same input, use latent cross-covariance and noisy marginal blocks.
    """

    def __init__(self, model, x, mean=None, row_ids=None):
        self.model = model
        self.x = model.transform_.transform(x)
        self.n_outputs = model.n_outputs
        self.row_ids = np.arange(len(self.x)) if row_ids is None else np.asarray(row_ids)
        if self.row_ids.ndim != 1 or len(self.row_ids) != len(self.x) or len(set(self.row_ids.tolist())) != len(self.x):
            raise ValueError("row_ids must be unique with one ID per X row")
        base = np.zeros((len(self.x), self.n_outputs)) if mean is None else _matrix(mean, "mean")
        if base.shape != (len(self.x), self.n_outputs):
            raise ValueError("mean must have shape [rows, outputs]")
        self.mean = base + (model._predict_z(self.x) + model.center_) * model.scales_
        self.noise_cov = model.noise_cov_.copy()
        self._cache = model._posterior_cache(self.x)

    def _indices(self, indices):
        idx = np.asarray(indices)
        if idx.ndim != 1 or idx.dtype.kind not in "iu" or np.any(idx < 0) or np.any(idx >= len(self.x)):
            raise ValueError("posterior positions must be valid one-dimensional integer arrays")
        return idx

    def cov(self, indices_a, indices_b, noisy=False):
        a, b = self._indices(indices_a), self._indices(indices_b)
        blocks = self.model._cov_blocks(self.x, self._cache, a, b)
        if noisy:
            blocks = blocks + (a[:, None] == b[None, :])[:, :, None, None] * self.noise_cov
        return blocks.transpose(0, 2, 1, 3).reshape(len(a) * self.n_outputs, len(b) * self.n_outputs)

    def latent_diag(self, indices):
        idx = self._indices(indices)
        return self.model._diag_blocks(self.x, self._cache, idx)

    def noise_blocks(self, indices):
        idx = self._indices(indices)
        return np.broadcast_to(self.noise_cov, (len(idx), self.n_outputs, self.n_outputs)).copy()

    def latent_block(self, i, j):
        return self.cov(np.asarray([i]), np.asarray([j]))

    def label_block(self, i, j):
        return self.cov(np.asarray([i]), np.asarray([j]), noisy=True)

    def predict(self, noisy=True):
        blocks = self.latent_diag(np.arange(len(self.x)))
        if noisy:
            blocks += self.noise_cov
        return self.mean.copy(), np.diagonal(blocks, axis1=1, axis2=2).copy()


class GaussianProcessSurrogate:
    """Exact independent-output RBF/Matérn-3/2 empirical-Bayes GP.

    Each output learns signal variance, isotropic lengthscale, and observation
    variance by bounded marginal likelihood.  PCA is fit on training inputs.
    No neural-network predictive quantile is interpreted as joint covariance.
    """

    def __init__(self, kernel="rbf", max_components=16, maxiter=60, restarts=1):
        if kernel not in ("rbf", "matern32"):
            raise ValueError("kernel must be rbf or matern32")
        if maxiter < 1 or restarts < 1:
            raise ValueError("optimization budget must be positive")
        self.kernel = kernel
        self.max_components = max_components
        self.maxiter = int(maxiter)
        self.restarts = int(restarts)

    def _kernel(self, a, b, theta):
        signal, length, _ = np.exp(theta)
        d2 = np.maximum(cdist(a, b, "sqeuclidean"), 0) / length ** 2
        if self.kernel == "rbf":
            return signal * np.exp(-0.5 * d2)
        r = np.sqrt(3 * d2)
        return signal * (1 + r) * np.exp(-r)

    def fit(self, X_train, y_train, mean_train=None, output_scales=None):
        started = time.perf_counter()
        x = _matrix(X_train, "X_train")
        z, self.scales_, self.center_ = _prepare_y(y_train, mean_train, output_scales)
        if len(x) != len(z) or len(x) < 3:
            raise ValueError("X/y must have at least three matching rows")
        self.transform_ = FeatureTransform(self.max_components).fit(x)
        self.train_x_ = self.transform_.transform(x)
        self.n_outputs = z.shape[1]
        self.models_ = []
        median_distance = np.median(cdist(self.train_x_, self.train_x_))
        initial_length = max(float(median_distance), 0.1)
        train_distance_squared = cdist(self.train_x_, self.train_x_, "sqeuclidean")
        for endpoint in range(self.n_outputs):
            target = z[:, endpoint]
            def objective(theta):
                signal, length, noise = np.exp(theta)
                d2 = train_distance_squared / length ** 2
                if self.kernel == "rbf":
                    prior = signal * np.exp(-.5 * d2)
                    length_gradient = prior * d2
                else:
                    r = np.sqrt(3 * d2)
                    prior = signal * (1 + r) * np.exp(-r)
                    length_gradient = signal * r ** 2 * np.exp(-r)
                k = prior.copy()
                k.flat[::len(k) + 1] += noise
                try:
                    factor = cho_factor(k, lower=True, check_finite=False)
                except np.linalg.LinAlgError:
                    return 1e100, np.zeros(3)
                alpha = cho_solve(factor, target, check_finite=False)
                value = float(0.5 * target @ alpha + np.log(np.diag(factor[0])).sum() + len(k) / 2 * np.log(2 * np.pi))
                sensitivity = cho_solve(factor, np.eye(len(k)), check_finite=False) - np.outer(alpha, alpha)
                gradient = .5 * np.array([np.sum(sensitivity * prior),
                                          np.sum(sensitivity * length_gradient),
                                          noise * np.trace(sensitivity)])
                return value, gradient
            bounds = [(-9.21, 6.91), (np.log(initial_length / 100), np.log(initial_length * 100)), (-11.51, 3.0)]
            fits = []
            for restart in range(self.restarts):
                t0 = np.log([max(float(np.var(target)), 0.1), initial_length * 2 ** restart, 0.1 * 2 ** restart])
                fits.append(minimize(objective, t0, method="L-BFGS-B", jac=True, bounds=bounds,
                                     options={"maxiter": self.maxiter, "maxls": 20, "ftol": 1e-8}))
            fit = min(fits, key=lambda item: item.fun)
            if not np.isfinite(fit.fun) or fit.fun >= 1e99:
                raise np.linalg.LinAlgError("no valid GP marginal likelihood fit")
            theta = fit.x
            k = self._kernel(self.train_x_, self.train_x_, theta)
            k.flat[::len(k) + 1] += np.exp(theta[2])
            factor = cho_factor(k, lower=True, check_finite=False)
            self.models_.append({"theta": theta, "factor": factor,
                                 "alpha": cho_solve(factor, target, check_finite=False),
                                 "nll": float(fit.fun), "optimizer_success": bool(fit.success),
                                 "optimizer_message": str(fit.message), "nit": int(fit.nit), "nfev": int(fit.nfev)})
        self.noise_cov_ = np.diag([np.exp(m["theta"][2]) for m in self.models_]) * np.outer(self.scales_, self.scales_)
        self.audit_ = {"family": "independent_exact_gp", "kernel": self.kernel,
                       "n_train": len(x), "n_outputs": self.n_outputs,
                       "feature_dimension": self.train_x_.shape[1], "maxiter": self.maxiter,
                       "pca_retained_standardized_input_variance_fraction": self.transform_.retained_variance_fraction,
                       "restarts": self.restarts, "fit_seconds": time.perf_counter() - started,
                       "parameters": [{"signal_variance": float(np.exp(m["theta"][0])),
                                       "lengthscale": float(np.exp(m["theta"][1])),
                                       "noise_variance_scaled": float(np.exp(m["theta"][2])),
                                       "noise_log_bounds": [-11.51, 3.0],
                                       "noise_bound_hit": bool(min(abs(m["theta"][2] + 11.51), abs(m["theta"][2] - 3.0)) < 1e-4),
                                       **{k: m[k] for k in ("nll", "optimizer_success", "optimizer_message", "nit", "nfev")}}
                                      for m in self.models_]}
        return self

    def posterior(self, X, mean=None, row_ids=None):
        return GaussianPosterior(self, X, mean=mean, row_ids=row_ids)

    def _predict_z(self, x):
        return np.column_stack([self._kernel(x, self.train_x_, m["theta"]) @ m["alpha"] for m in self.models_])

    def _posterior_cache(self, x):
        return [(self._kernel(x, self.train_x_, m["theta"]),
                 cho_solve(m["factor"], self._kernel(self.train_x_, x, m["theta"]), check_finite=False))
                for m in self.models_]

    def _cov_blocks(self, x, cache, a, b):
        blocks = np.zeros((len(a), len(b), self.n_outputs, self.n_outputs), dtype=np.float64)
        for output, (m, (cross, solved)) in enumerate(zip(self.models_, cache)):
            blocks[:, :, output, output] = (self._kernel(x[a], x[b], m["theta"]) - cross[a] @ solved[:, b]) * self.scales_[output] ** 2
        return blocks

    def _diag_blocks(self, x, cache, indices):
        blocks = np.zeros((len(indices), self.n_outputs, self.n_outputs), dtype=np.float64)
        for output, (m, (cross, solved)) in enumerate(zip(self.models_, cache)):
            variance = np.exp(m["theta"][0]) - np.einsum("ij,ji->i", cross[indices], solved[:, indices])
            if np.min(variance, initial=0) < -1e-8 * np.exp(m["theta"][0]):
                raise np.linalg.LinAlgError("GP posterior has materially negative variance")
            blocks[:, output, output] = np.maximum(variance, 0) * self.scales_[output] ** 2
        return blocks


class BayesianLinearSurrogate:
    """Empirical-Bayes matrix-normal linear regression on train-only features.

    With ``coregionalized=True``, W ~ MN(0, alpha^-1 I, B) and observation
    noise ~ MN(0, sigma² I, B). B is a shrinkage covariance of centered training
    responses (after subtracting any supplied base mean), not fitted residuals.
    This is an explicit separable cross-output model, not an ensemble posterior
    or two independent EPIG scores relabeled as a coupled GP. Shared B in prior
    and noise limits how the cross-output dependence affects two-output MI.
    """

    def __init__(self, coregionalized=False, max_components=16, maxiter=60, correlation_shrinkage=0.2):
        if maxiter < 1 or not 0 < correlation_shrinkage <= 1:
            raise ValueError("invalid optimization budget or shrinkage")
        self.coregionalized = bool(coregionalized)
        self.max_components = max_components
        self.maxiter = int(maxiter)
        self.correlation_shrinkage = float(correlation_shrinkage)

    def fit(self, X_train, y_train, mean_train=None, output_scales=None):
        started = time.perf_counter()
        x = _matrix(X_train, "X_train")
        z, self.scales_, self.center_ = _prepare_y(y_train, mean_train, output_scales)
        if len(x) != len(z) or len(x) < 3:
            raise ValueError("X/y must have at least three matching rows")
        self.transform_ = FeatureTransform(self.max_components).fit(x)
        tx = self.transform_.transform(x)
        # Train-fitted RMS keeps a single prior precision meaningful as d changes.
        self.feature_rms_ = max(float(np.sqrt(np.mean(np.sum(tx * tx, axis=1)))), 1e-12)
        phi = self._phi(tx)
        self.n_outputs = z.shape[1]
        empirical = np.atleast_2d(np.cov(z, rowvar=False))
        if self.coregionalized:
            self.B_ = (1 - self.correlation_shrinkage) * empirical + self.correlation_shrinkage * np.diag(np.diag(empirical))
        else:
            self.B_ = np.diag(np.diag(empirical))
        self.B_ += np.eye(self.n_outputs) * 1e-8
        bfactor = cho_factor(self.B_, lower=True)
        whitened = cho_solve(bfactor, z.T).T
        p = phi.shape[1]
        gram, rhs = phi.T @ phi, phi.T @ z
        def objective(theta):
            alpha, noise = np.exp(theta)
            precision = alpha * np.eye(p) + gram / noise
            factor = cho_factor(precision, lower=True)
            weights = cho_solve(factor, rhs / noise)
            inv_y = z / noise - phi @ weights / noise
            quadratic = np.sum(whitened * inv_y)
            logdet = len(phi) * np.log(noise) + 2 * np.log(np.diag(factor[0])).sum() - p * np.log(alpha)
            return float(0.5 * (quadratic + self.n_outputs * logdet + len(phi) * 2 * np.log(np.diag(bfactor[0])).sum() + z.size * np.log(2 * np.pi)))
        fit = minimize(objective, np.log([1, .3]), method="L-BFGS-B", bounds=[(-9.21, 9.21), (-11.51, 4.61)],
                       options={"maxiter": self.maxiter, "ftol": 1e-9})
        if not np.isfinite(fit.fun):
            raise np.linalg.LinAlgError("invalid BLR marginal likelihood")
        alpha, self.noise_ = np.exp(fit.x)
        factor = cho_factor(alpha * np.eye(p) + gram / self.noise_, lower=True)
        self.coefficient_cov_ = cho_solve(factor, np.eye(p))
        self.weights_ = cho_solve(factor, rhs / self.noise_)
        self.physical_B_ = self.B_ * np.outer(self.scales_, self.scales_)
        self.noise_cov_ = self.noise_ * self.physical_B_
        self.audit_ = {"family": "matrix_normal_blr" if self.coregionalized else "independent_blr_shared_spatial_covariance",
                       "n_train": len(x), "n_outputs": self.n_outputs, "feature_dimension": p,
                       "pca_retained_standardized_input_variance_fraction": self.transform_.retained_variance_fraction,
                       "prior_precision": float(alpha), "noise_multiplier": float(self.noise_),
                       "noise_log_bounds": [-11.51, 4.61],
                       "noise_bound_hit": bool(min(abs(fit.x[1] + 11.51), abs(fit.x[1] - 4.61)) < 1e-4),
                       "coregion_scaled": self.B_.tolist(), "correlation_shrinkage": self.correlation_shrinkage,
                       "nll": float(fit.fun), "optimizer_success": bool(fit.success),
                       "optimizer_message": str(fit.message), "nit": int(fit.nit), "nfev": int(fit.nfev),
                       "maxiter": self.maxiter, "fit_seconds": time.perf_counter() - started}
        return self

    def _phi(self, x):
        return np.column_stack([np.ones(len(x)), x / self.feature_rms_])

    def posterior(self, X, mean=None, row_ids=None):
        return GaussianPosterior(self, X, mean=mean, row_ids=row_ids)

    def _predict_z(self, x):
        return self._phi(x) @ self.weights_

    def _posterior_cache(self, x):
        phi = self._phi(x)
        return phi, phi @ self.coefficient_cov_

    def _cov_blocks(self, x, cache, a, b):
        phi, projected = cache
        return (projected[a] @ phi[b].T)[:, :, None, None] * self.physical_B_

    def _diag_blocks(self, x, cache, indices):
        phi, projected = cache
        variances = np.einsum("ij,ij->i", projected[indices], phi[indices])
        return variances[:, None, None] * self.physical_B_


def gaussian_probability_metrics(y, mean, covariance_blocks):
    """Scores in physical units; no selection or tuning is performed here."""
    y, mean = _matrix(y, "y"), _matrix(mean, "mean")
    covariance = np.asarray(covariance_blocks, dtype=np.float64)
    if y.shape != mean.shape or covariance.shape != (len(y), y.shape[1], y.shape[1]):
        raise ValueError("prediction and covariance shapes differ")
    if not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.transpose(0, 2, 1)):
        raise ValueError("covariance blocks must be finite and symmetric")
    error = y - mean
    variances = np.diagonal(covariance, axis1=1, axis2=2)
    if np.any(variances <= 0):
        raise np.linalg.LinAlgError("predictive variances must be positive")
    sigma = np.sqrt(variances)
    standardized = error / sigma
    nll = .5 * (np.log(2 * np.pi * variances) + standardized ** 2)
    joint_nll = []
    for e, block in zip(error, covariance):
        factor = cho_factor(block, lower=True)
        joint_nll.append(.5 * (e @ cho_solve(factor, e) + 2 * np.log(np.diag(factor[0])).sum() + len(e) * np.log(2 * np.pi)))
    # Conditional coverage by predicted standard-deviation quartiles.
    conditional = []
    for output in range(y.shape[1]):
        groups = np.array_split(np.argsort(sigma[:, output], kind="stable"), 4)
        conditional.append([{"rows": len(g), "coverage80": float(np.mean(np.abs(standardized[g, output]) <= ndtri(.9))),
                             "mean_sigma": float(sigma[g, output].mean())} for g in groups if len(g)])
    return {"rows": len(y), "rmse": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
            "mae": np.mean(np.abs(error), axis=0).tolist(), "marginal_nll": nll.mean(axis=0).tolist(),
            "joint_nll": float(np.mean(joint_nll)),
            "coverage50": np.mean(np.abs(standardized) <= ndtri(.75), axis=0).tolist(),
            "coverage80": np.mean(np.abs(standardized) <= ndtri(.9), axis=0).tolist(),
            "coverage95": np.mean(np.abs(standardized) <= ndtri(.975), axis=0).tolist(),
            "standardized_residual_mean": standardized.mean(axis=0).tolist(),
            "standardized_residual_second_moment": np.mean(standardized ** 2, axis=0).tolist(),
            "conditional_coverage80_by_sigma_quartile": conditional}


def qualify_model(model_factory: Callable, X, y, n_splits=3, seed=0, output_scales=None,
                  cache_directory=None, cache_contract=None):
    """Honest train-only row OOF qualification, including full joint OOF NLL.

    X must be label-free or constructed without held-out supervision; this
    function cannot repair a feature bank pretrained on its held-out y labels.
    Scales, PCA, kernel/noise fitting, and intercepts use the training fold only.
    A fixed externally specified output scale is allowed and only changes units.
    Joint scores evaluate cross-row covariance within each held-out fold; they
    are diagnostic evidence, not proof of correct dependence from one data set.
    Optional fold caches require an explicit model/config contract. Input arrays,
    this source file, splits, and scales are additionally hashed. Interrupted
    fold attempts are recorded before fitting; completed folds are never refit.
    """
    x, y = _matrix(X, "X"), _matrix(y, "y")
    if len(x) != len(y) or not 2 <= n_splits <= len(x) // 2:
        raise ValueError("invalid fold count or X/y rows")
    fold_indices = np.array_split(np.random.default_rng(seed).permutation(len(x)), n_splits)
    cache = Path(cache_directory) if cache_directory is not None else None
    def file_hash(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def array_hash(array):
        array = np.ascontiguousarray(array)
        return {"shape": list(array.shape), "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    def atomic_json(path, value):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        tmp.replace(path)
    contract_hash = None
    if cache is not None:
        if cache_contract is None:
            raise ValueError("cached qualification requires an explicit model/config cache_contract")
        contract = {"schema": "epig-oof-fold-cache-v1", "model_config": cache_contract,
                    "X": array_hash(x), "y": array_hash(y), "n_splits": int(n_splits), "seed": int(seed),
                    "output_scales": None if output_scales is None else array_hash(np.asarray(output_scales, dtype=np.float64)),
                    "implementation_sha256": file_hash(__file__)}
        contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest()
        cache.mkdir(parents=True, exist_ok=True)
        contract_path = cache / "contract.json"
        if contract_path.exists():
            if json.loads(contract_path.read_text()) != contract:
                raise RuntimeError("OOF cache input/model/implementation contract drift")
        else:
            atomic_json(contract_path, contract)
    means = np.empty_like(y)
    blocks = np.empty((len(y), y.shape[1], y.shape[1]))
    folds = []
    all_rows = np.arange(len(x))
    for fold, held in enumerate(fold_indices):
        train = np.setdiff1d(all_rows, held)
        if cache is not None:
            receipt = cache / f"fold_{fold}_complete.json"
            array_path = cache / f"fold_{fold}.npz"
            metadata_path = cache / f"fold_{fold}.json"
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                if saved["contract_sha256"] != contract_hash or any(file_hash(cache / name) != expected for name, expected in saved["files"].items()):
                    raise RuntimeError("OOF completed fold cache drift")
                f = json.loads(metadata_path.read_text())
                if f["held_out_positions"] != held.tolist() or f["train_positions"] != train.tolist():
                    raise RuntimeError("OOF cached fold partition drift")
                with np.load(array_path, allow_pickle=False) as arrays:
                    means[held] = arrays["mean"]
                    blocks[held] = arrays["covariance_blocks"]
                folds.append(f)
                continue
            existing = list(cache.glob("attempt_*.json"))
            attempt = cache / f"attempt_{len(existing) + 1:04d}.json"
            # Exclusive creation protects the audit even if accidental parallel
            # invocations compete; the caller additionally owns its run lock.
            with attempt.open("x") as handle:
                json.dump({"fold": fold, "contract_sha256": contract_hash,
                           "started_unix": time.time(), "status": "STARTED"}, handle)
        model = model_factory().fit(x[train], y[train], output_scales=output_scales)
        posterior = model.posterior(x[held], row_ids=held)
        means[held] = posterior.mean
        blocks[held] = posterior.latent_diag(np.arange(len(held))) + posterior.noise_cov
        covariance = posterior.cov(np.arange(len(held)), np.arange(len(held)), noisy=True)
        factor = cho_factor(covariance, lower=True)
        residual = (y[held] - posterior.mean).ravel()
        mahalanobis = residual @ cho_solve(factor, residual)
        joint_nll = .5 * (mahalanobis + 2 * np.log(np.diag(factor[0])).sum() + len(residual) * np.log(2 * np.pi))
        diag = np.diag(covariance)
        diagonal_nll = .5 * np.sum(residual ** 2 / diag + np.log(diag) + np.log(2 * np.pi))
        block_metrics = gaussian_probability_metrics(y[held], posterior.mean, blocks[held])
        correlation = covariance / np.sqrt(np.outer(diag, diag))
        off_diagonal = correlation[np.triu_indices(len(correlation), 1)]
        fold_result = {"fold": fold, "train_rows": len(train), "held_out_rows": len(held),
                      "held_out_positions": held.tolist(), "train_positions": train.tolist(),
                      "full_joint_nll_per_row": float(joint_nll / len(held)),
                      "full_joint_nll_per_scalar": float(joint_nll / len(residual)),
                      "diagonal_nll_per_scalar": float(diagonal_nll / len(residual)),
                      "block_diagonal_nll_per_scalar": float(block_metrics["joint_nll"] / y.shape[1]),
                      "whitened_residual_rms": float(np.sqrt(mahalanobis / len(residual))),
                      "mean_absolute_offdiagonal_predictive_correlation": float(np.mean(np.abs(off_diagonal))),
                      "minimum_predictive_eigenvalue": float(np.linalg.eigvalsh(covariance)[0]),
                      "model": model.audit_}
        folds.append(fold_result)
        if cache is not None:
            tmp_arrays = array_path.with_suffix(".npz.tmp")
            with tmp_arrays.open("wb") as handle:
                np.savez_compressed(handle, mean=means[held], covariance_blocks=blocks[held])
            tmp_arrays.replace(array_path)
            atomic_json(metadata_path, fold_result)
            atomic_json(receipt, {"contract_sha256": contract_hash,
                                 "files": {p.name: file_hash(p) for p in (array_path, metadata_path)}})
            attempt_record = json.loads(attempt.read_text())
            attempt_record.update({"status": "COMPLETE", "finished_unix": time.time(),
                                   "receipt_sha256": file_hash(receipt)})
            atomic_json(attempt, attempt_record)
    return {"metrics": gaussian_probability_metrics(y, means, blocks), "folds": folds,
            "oof_mean": means, "oof_covariance_blocks": blocks,
            "fit_attempt_count": len(list(cache.glob("attempt_*.json"))) if cache is not None else n_splits,
            "qualification_scope": "train-only row OOF; requires label-free input features"}
