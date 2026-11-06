"""Exact float64 GPs; training labels are the only response input to fit.

Reuses Phase 0 validation, Cholesky, probability metrics and information gain.
All output variances, including ``noise``, use the original response units.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import time

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

_PHASE0 = Path(__file__).resolve().parent.parent / "gp_guided_llm_stability_20261103" / "gp.py"
_spec = importlib.util.spec_from_file_location("representation_benchmark_phase0_gp", _PHASE0)
_phase0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_phase0)
stable_cholesky = _phase0.stable_cholesky
probability_metrics = _phase0.probability_metrics
batch_information = _phase0.batch_information
rank_correlation = _phase0.rank_correlation


def regression_metrics(target, mean, observation_variance):
    """Extend, rather than replace, the already qualified Phase 0 metrics."""
    result = probability_metrics(target, mean, observation_variance)
    y, mu, variance = (np.asarray(a, dtype=np.float64) for a in (target, mean, observation_variance))
    result.update(mae=float(np.mean(np.abs(y - mu))),
                  mean_interval_width_95=float(np.mean(2 * 1.959963984540054 * np.sqrt(variance))),
                  absolute_error_uncertainty_spearman=rank_correlation(np.abs(y - mu), np.sqrt(variance)))
    return result


def _array(value):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all() or a.shape[1] == 0:
        raise ValueError("finite, nonempty feature dimension required")
    return a


def _matern52(a, b, lengthscale):
    """Isotropic distance per RMS coordinate after training-only scaling."""
    radius = np.sqrt(cdist(a, b, "sqeuclidean") / a.shape[1]) / lengthscale
    scaled = np.sqrt(5.) * radius
    return (1. + scaled + scaled * scaled / 3.) * np.exp(-scaled)


class FixedRepresentationGP:
    """Training-only centered/RMS features and standardized target, constant mean.

    Matérn hyperparameters are fit with exact MLL and analytic torch gradients.
    The optional linear control delegates optimization to Phase 0 LinearGP.
    The optimizer is never given validation labels or validation metrics.
    """
    def __init__(self, *, kernel="matern52", max_iter=40, starts=1, seed=0,
                 log_variance_bounds=(-9., 9.), log_lengthscale_bounds=(-5., 5.), threads=1):
        if kernel not in ("matern52", "linear") or max_iter < 1 or starts != 1:
            raise ValueError("kernel must be matern52/linear; declared single start and positive iterations required")
        self.kernel = kernel
        self.max_iter, self.seed, self.threads = int(max_iter), int(seed), int(threads)
        self.variance_bounds = tuple(log_variance_bounds)
        self.lengthscale_bounds = tuple(log_lengthscale_bounds)
        self.is_projection = False

    def _prepare(self, features, targets):
        features = _array(features)
        y = np.asarray(targets, dtype=np.float64)
        if y.shape != (len(features),) or len(y) < 2 or not np.isfinite(y).all():
            raise ValueError("at least two finite training responses matching features required")
        self.feature_mean = features.mean(axis=0)
        scale = features.std(axis=0)
        self.constant_features = int(np.sum(scale < 1e-12))
        global_rms = float(np.sqrt(np.mean((features - self.feature_mean)**2)))
        self.feature_scale = np.full(features.shape[1], global_rms if global_rms >= 1e-12 else 1.)
        self.mean, self.target_scale = float(y.mean()), float(y.std())
        if self.target_scale < 1e-12:
            self.target_scale = 1.
        self.feature_dimension = features.shape[1]
        return (features - self.feature_mean) / self.feature_scale, (y - self.mean) / self.target_scale, y

    def transform(self, features):
        x = _array(features)
        if x.shape[1] != self.feature_dimension:
            raise ValueError("query feature dimension mismatch")
        return (x - self.feature_mean) / self.feature_scale

    def _latent_features(self, features):
        x = self.transform(features)
        if self.is_projection:
            h = x @ self.projection_weight + self.projection_bias
            x = np.where(h > 0., h, np.expm1(np.minimum(h, 0.)))
        return x

    def embedding(self, features):
        """Return the actual fitted kernel representation without using labels."""
        return self._latent_features(features)

    def _base_kernel(self, a, b):
        if self.kernel == "linear":
            return (a @ b.T) / a.shape[1]
        return _matern52(a, b, self.lengthscale)

    def fit(self, features, targets):
        started = time.perf_counter()
        xs, ys, raw_y = self._prepare(features, targets)
        if self.kernel == "linear":
            old = _phase0.LinearGP().fit(xs / np.sqrt(xs.shape[1]), ys,
                                        bounds=self.variance_bounds)
            self.signal, self.noise = old.signal * self.target_scale**2, old.noise * self.target_scale**2
            self.lengthscale = None
            self.loss_history = []
            optimizer = dict(success=old.audit["optimizer_success"], message=old.audit["optimizer_message"],
                             iterations=None, evaluations=None, boundary_hit=old.audit["boundary_hit"],
                             method="Phase0 LinearGP L-BFGS-B", starts=3)
        else:
            self._optimize(xs, ys)
            optimizer = self._optimizer
        self.x = xs
        self._finalize(raw_y, optimizer, started)
        return self

    def _optimize(self, xs, ys):
        import torch
        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        tx, ty = torch.as_tensor(xs, dtype=torch.float64), torch.as_tensor(ys, dtype=torch.float64)
        n, p = xs.shape
        identity = torch.eye(n, dtype=torch.float64)
        if self.is_projection:
            rng = np.random.default_rng(self.seed)
            weight = rng.normal(0., 1. / np.sqrt(p), size=(p, self.projection_dim))
            initial = np.r_[weight.ravel(), np.zeros(self.projection_dim), 0., -2., 0.]
            offset = p * self.projection_dim
            bounds = [(None, None)] * (offset + self.projection_dim)
        else:
            initial, offset, bounds = np.array([0., -2., 0.]), 0, []
            base_d2 = torch.as_tensor(cdist(xs, xs, "sqeuclidean") / p, dtype=torch.float64)
        bounds += [self.variance_bounds, self.variance_bounds, self.lengthscale_bounds]
        self.loss_history, jitter_history = [], []

        def objective(parameters):
            vector = torch.tensor(parameters, dtype=torch.float64, requires_grad=True)
            log_signal, log_noise, log_ell = vector[-3:]
            if self.is_projection:
                weight = vector[:offset].reshape(p, self.projection_dim)
                bias = vector[offset:-3]
                z = torch.nn.functional.elu(tx @ weight + bias)
                square = (z * z).sum(dim=1)
                d2 = (square[:, None] + square[None, :] - 2 * (z @ z.T)).clamp_min(0.) / self.projection_dim
            else:
                d2 = base_d2
            radius = torch.sqrt(d2.clamp_min(1e-24)) / torch.exp(log_ell)
            scaled = np.sqrt(5.) * radius
            covariance = torch.exp(log_signal) * (1 + scaled + scaled * scaled / 3) * torch.exp(-scaled)
            covariance = covariance + torch.exp(log_noise) * identity
            scale = max(1., float(torch.exp(log_signal).detach() + torch.exp(log_noise).detach()))
            factor = None
            for relative in (0., 1e-12, 1e-10, 1e-8):
                jitter = relative * scale
                candidate, info = torch.linalg.cholesky_ex(covariance + jitter * identity)
                if int(info.max().detach()) == 0:
                    factor = candidate
                    break
            if factor is None:
                raise ValueError("training Cholesky exceeded declared Phase0 jitter budget")
            alpha = torch.cholesky_solve(ty[:, None], factor)
            loss = .5 * (ty[:, None] * alpha).sum() + torch.log(torch.diagonal(factor)).sum() + .5 * n * np.log(2 * np.pi)
            loss.backward()
            value, gradient = float(loss.detach()), vector.grad.detach().numpy().copy()
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                raise ValueError("nonfinite MLL optimization")
            self.loss_history.append(value)
            jitter_history.append(jitter)
            return value, gradient

        result = minimize(objective, initial, jac=True, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": self.max_iter, "maxls": 20, "ftol": 1e-9, "gtol": 1e-5,
                                   "maxcor": 10, "maxfun": self.max_iter * 10})
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise ValueError("no finite GP optimization result")
        self.signal, self.noise = np.exp(result.x[-3:-1]) * self.target_scale**2
        self.lengthscale = float(np.exp(result.x[-1]))
        if self.is_projection:
            self.projection_weight = result.x[:offset].reshape(p, self.projection_dim).copy()
            self.projection_bias = result.x[offset:-3].copy()
        boundary_hit = any(abs(value - bound) < 1e-4 for value, limits in zip(result.x[-3:], bounds[-3:])
                           for bound in limits)
        self._optimizer = dict(success=bool(result.success), message=str(result.message),
                               iterations=int(result.nit), evaluations=int(result.nfev),
                               boundary_hit=bool(boundary_hit), method="scipy L-BFGS-B + torch exact MLL gradients",
                               starts=1, max_training_jitter_standardized=float(max(jitter_history, default=0.)))

    def _finalize(self, raw_y, optimizer, started):
        # Final posterior factorization and covariance validation reuse Phase 0.
        self.cholesky, jitter = stable_cholesky(self.signal * self._base_kernel(self.x, self.x)
                                               + self.noise * np.eye(len(self.x)))
        centered = raw_y - self.mean
        self.alpha = cho_solve((self.cholesky, True), centered)
        nll = float(.5 * centered @ self.alpha + np.log(np.diag(self.cholesky)).sum()
                    + .5 * len(centered) * np.log(2 * np.pi))
        self.audit = dict(kernel=self.kernel, dtype="float64", training_rows=len(raw_y),
                          feature_dimension=self.feature_dimension, kernel_feature_dimension=self.x.shape[1],
                          mean_function="empirical training constant", constant_mean=self.mean,
                          target_scale=self.target_scale, feature_scaling="training coordinate mean + one global centered RMS",
                          feature_global_centered_rms=float(self.feature_scale[0]),
                          constant_feature_count=self.constant_features,
                          distance_normalization="mean squared coordinate distance",
                          signal_variance=float(self.signal), noise_variance=float(self.noise),
                          lengthscale=self.lengthscale, numerical_jitter=float(jitter),
                          negative_log_marginal_likelihood=nll, marginal_log_likelihood=-nll,
                          optimizer_success=optimizer["success"], optimizer_message=optimizer["message"],
                          optimizer=optimizer, seed=self.seed, max_iter=self.max_iter,
                          training_seconds=time.perf_counter() - started, prediction_seconds=[],
                          loss_history_standardized=self.loss_history, threads=self.threads,
                          trainable_projection=self.is_projection)
        if self.is_projection:
            self.audit.update(projection_dimension=self.projection_dim,
                              projection_parameters=self.projection_weight.size + self.projection_bias.size,
                              projection_activation="ELU", input_dimension_inferred=True,
                              projection_penalty="none; exact MLL objective only",
                              scale_identifiability="projection scale and lengthscale are not separately identifiable")

    def predict(self, features, *, full_covariance=False):
        started = time.perf_counter()
        query = self._latent_features(features)
        cross = self.signal * self._base_kernel(query, self.x)
        projected = solve_triangular(self.cholesky, cross.T, lower=True)
        mean = self.mean + cross @ self.alpha
        if full_covariance:
            covariance = self.signal * self._base_kernel(query, query) - projected.T @ projected
            covariance = (covariance + covariance.T) / 2
            if len(query) and np.linalg.eigvalsh(covariance).min() < -1e-8 * max(1., self.signal):
                raise ValueError("invalid posterior covariance")
            result = covariance
        else:
            diagonal = self.signal * (np.sum(query**2, axis=1) / query.shape[1] if self.kernel == "linear"
                                      else np.ones(len(query)))
            result = diagonal - np.sum(projected**2, axis=0)
            if len(query) and result.min() < -1e-8 * max(1., self.signal):
                raise ValueError("negative posterior variance")
            result = np.maximum(result, 0.)
        self.audit["prediction_seconds"].append(time.perf_counter() - started)
        return mean, result

    def predict_observation(self, features, *, full_covariance=False):
        mean, latent = self.predict(features, full_covariance=full_covariance)
        observation = latent + (self.noise * np.eye(len(latent)) if full_covariance else self.noise)
        return mean, latent, observation


class ProjectedMaternGP(FixedRepresentationGP):
    """Small Linear(inferred input dimension, 64)+ELU and exact GP joint MLL."""
    def __init__(self, *, projection_dim=64, **kwargs):
        if projection_dim < 1:
            raise ValueError("positive projection dimension required")
        super().__init__(kernel="matern52", **kwargs)
        self.projection_dim, self.is_projection = int(projection_dim), True

    def fit(self, features, targets):
        started = time.perf_counter()
        xs, ys, raw_y = self._prepare(features, targets)
        self._optimize(xs, ys)
        h = xs @ self.projection_weight + self.projection_bias
        self.x = np.where(h > 0., h, np.expm1(np.minimum(h, 0.)))
        self._finalize(raw_y, self._optimizer, started)
        return self


def acquisition_diagnostics(gp, candidate_features, *, batch_size=32, candidate_limit=256, seed=0):
    """Label-free offline diagnostics on one frozen and disclosed candidate subset.

    MaxVar is a one-shot marginal ranking. Conditional IVR averages variance
    reduction over the same frozen reference subset. Batch information gain
    uses sequential Gaussian conditioning with independent observation noise.
    No selected point is queried for a response and no production file is written.
    """
    x = _array(candidate_features)
    if len(x) == 0 or batch_size < 1 or candidate_limit < 1:
        raise ValueError("nonempty candidates and positive limits required")
    count = min(len(x), int(candidate_limit))
    positions = (np.arange(len(x)) if len(x) <= count else
                 np.sort(np.random.default_rng(seed).choice(len(x), count, replace=False)))
    mean, covariance = gp.predict(x[positions], full_covariance=True)
    covariance = (covariance + covariance.T) / 2
    diagonal = np.maximum(np.diag(covariance), 0.)
    b = min(batch_size, count)
    maxvar = np.argsort(-diagonal, kind="stable")[:b]

    def greedy(mode):
        current, available, selected, trace = covariance.copy(), np.ones(count, dtype=bool), [], []
        for step in range(b):
            variance = np.maximum(np.diag(current), 0.)
            if mode == "ivr":
                scores = np.sum(current**2, axis=0) / (count * (variance + gp.noise))
            else:
                scores = .5 * np.log1p(variance / gp.noise)
            scores[~available] = -np.inf
            index = int(np.argmax(scores))
            selected.append(index)
            trace.append(dict(step=step, candidate_position=int(positions[index]), score=float(scores[index]),
                              conditional_latent_variance=float(variance[index])))
            column = current[:, index].copy()
            current -= np.outer(column, column) / (variance[index] + gp.noise)
            current = (current + current.T) / 2
            available[index] = False
        return np.asarray(selected), trace

    ivr, ivr_trace = greedy("ivr")
    information, information_trace = greedy("information")
    methods = {}
    for name, selected, trace in (("maxvar", maxvar, []), ("conditional_ivr", ivr, ivr_trace),
                                   ("batch_information_gain", information, information_trace)):
        chosen_covariance = covariance[np.ix_(selected, selected)]
        info, jitter = batch_information(chosen_covariance, gp.noise)
        std = np.sqrt(np.maximum(np.diag(chosen_covariance), 1e-30))
        correlation = chosen_covariance / np.outer(std, std)
        upper = np.triu_indices(len(selected), 1)
        methods[name] = dict(selected_candidate_positions=positions[selected].astype(int).tolist(),
                             batch_information=float(info), information_jitter=float(jitter),
                             mean_absolute_batch_latent_correlation=float(np.abs(correlation[upper]).mean()) if b > 1 else None,
                             trace=trace)
    ivr_first = np.sum(covariance**2, axis=0) / (count * (diagonal + gp.noise))
    return dict(candidate_count=len(x), evaluated_candidate_count=count, candidate_limit=int(candidate_limit),
                candidate_seed=int(seed), candidate_positions=positions.astype(int).tolist(),
                candidate_positions_sha256=hashlib.sha256(positions.astype("<i8").tobytes()).hexdigest(),
                reference_pool="same fixed candidate subset; uniform mean IVR",
                latent_mean=mean.tolist(), latent_variance=diagonal.tolist(), initial_ivr_scores=ivr_first.tolist(),
                initial_information_scores=(.5 * np.log1p(diagonal / gp.noise)).tolist(), methods=methods)
