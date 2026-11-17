"""Audited exact additive kernels over frozen GNN and language features.

The two unit-diagonal Matérn kernels share one signal variance. The language
kernel has no amplitude of its own: its only relative amplitude is lambda.
No validation response is accepted by the fitter.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigvalsh
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

OLD = Path(__file__).resolve().parent.parent / "representation_gp_benchmark_20261104"
_spec = importlib.util.spec_from_file_location("fusion_previous_gp_models", OLD / "gp_models.py")
_old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_old)
regression_metrics = _old.regression_metrics
acquisition_diagnostics = _old.acquisition_diagnostics


class AdditiveKernelGP(_old.FixedRepresentationGP):
    """k = signal * (unit_Matern_G + lambda * unit_Matern_L).

    Inputs concatenate GNN and language feature blocks. Scaling is fitted
    separately on the training rows of each block. At fixed lambda=0 the
    implementation has exactly the GNN-only objective and three parameters.
    The capacity control replaces L by a second copy of G, giving the same
    five trainable scalars as learned additive fusion without language input.
    """
    def __init__(self, *, gnn_dimension, lambda_mode="learned", fixed_lambda=1.,
                 max_iter=200, capacity_control=False, seed=17, threads=1):
        if gnn_dimension < 1 or lambda_mode not in ("zero", "fixed", "learned"):
            raise ValueError("positive GNN dimension and zero/fixed/learned mode required")
        if fixed_lambda < 0 or not np.isfinite(fixed_lambda):
            raise ValueError("lambda must be finite and nonnegative")
        super().__init__(kernel="matern52", max_iter=max_iter, seed=seed, threads=threads)
        self.gnn_dimension = int(gnn_dimension)
        self.lambda_mode = lambda_mode
        self.fixed_lambda = 0. if lambda_mode == "zero" else float(fixed_lambda)
        self.capacity_control = bool(capacity_control)

    def _prepare(self, features, targets):
        features = _old._array(features)
        if features.shape[1] <= self.gnn_dimension:
            raise ValueError("GNN and language feature blocks both required")
        xs, ys, raw_y = super()._prepare(features, targets)
        means, scales = [], []
        for block in (features[:, :self.gnn_dimension], features[:, self.gnn_dimension:]):
            means.append(block.mean(0))
            rms = float(np.sqrt(np.mean((block - block.mean(0)) ** 2)))
            scales.append(np.full(block.shape[1], rms if rms > 1e-12 else 1.))
        self.feature_mean, self.feature_scale = np.concatenate(means), np.concatenate(scales)
        return self.transform(features), ys, raw_y

    def _split(self, x):
        g = x[:, :self.gnn_dimension]
        return g, (g if self.capacity_control else x[:, self.gnn_dimension:])

    @classmethod
    def restore_from_audit(cls, features, targets, audit):
        """Rebuild a frozen posterior without optimizing or reading new labels."""
        from scipy.linalg import cho_solve
        model = cls(gnn_dimension=audit["gnn_feature_dimension"], lambda_mode=audit["lambda_mode"],
                    fixed_lambda=audit["lambda_value"], max_iter=audit["max_iter"],
                    capacity_control=audit["capacity_control"], seed=audit.get("optimizer_seed", 17))
        model.x, _, raw_y = model._prepare(features, targets)
        for attribute, key in (("signal", "signal_variance"), ("noise", "noise_variance"),
                               ("lengthscale", "lengthscale"), ("language_lengthscale", "language_lengthscale"),
                               ("lambda_value", "lambda_value")):
            setattr(model, attribute, audit[key])
        covariance = model.signal * model._base_kernel(model.x, model.x) + model.noise * np.eye(len(raw_y))
        model.cholesky, _ = _old.stable_cholesky(covariance)
        model.alpha = cho_solve((model.cholesky, True), raw_y-model.mean)
        model.audit = {**audit, "prediction_seconds": []}
        return model

    def _base_kernel(self, a, b):
        ga, la = self._split(a)
        gb, lb = self._split(b)
        result = _old._matern52(ga, gb, self.lengthscale)
        if self.lambda_value:
            result += self.lambda_value * _old._matern52(la, lb, self.language_lengthscale)
        return result

    def _optimize(self, xs, ys):
        import torch
        torch.set_num_threads(self.threads)
        g, l = self._split(xs)
        dg = torch.tensor(cdist(g, g, "sqeuclidean") / g.shape[1], dtype=torch.float64)
        dl = torch.tensor(cdist(l, l, "sqeuclidean") / l.shape[1], dtype=torch.float64)
        ty = torch.tensor(ys, dtype=torch.float64)
        eye = torch.eye(len(ys), dtype=torch.float64)
        active_language = self.lambda_mode == "learned" or (self.lambda_mode == "fixed" and self.fixed_lambda != 0.)
        initial = [0., -2., 0.]
        bounds = [self.variance_bounds, self.variance_bounds, self.lengthscale_bounds]
        if active_language:
            initial.append(np.log(.5) if self.capacity_control else 0.)
            bounds.append(self.lengthscale_bounds)
        if self.lambda_mode == "learned":
            initial.append(0.)
            bounds.append((-9., 9.))
        self.loss_history, jitters, gradient_norms = [], [], []

        def matern(d2, ell):
            scaled = np.sqrt(5.) * torch.sqrt(d2.clamp_min(1e-24)) / torch.exp(ell)
            return (1 + scaled + scaled * scaled / 3) * torch.exp(-scaled)

        def objective(parameters):
            vector = torch.tensor(parameters, dtype=torch.float64, requires_grad=True)
            base = matern(dg, vector[2])
            if active_language:
                lam = torch.exp(vector[4]) if self.lambda_mode == "learned" else self.fixed_lambda
                base = base + lam * matern(dl, vector[3])
            covariance = torch.exp(vector[0]) * base + torch.exp(vector[1]) * eye
            factor = None
            scale = max(1., float(covariance.diagonal().mean().detach()))
            for relative in (0., 1e-12, 1e-10, 1e-8):
                jitter = relative * scale
                candidate, info = torch.linalg.cholesky_ex(covariance + jitter * eye)
                if int(info.max().detach()) == 0:
                    factor = candidate
                    break
            if factor is None:
                raise ValueError("fusion Cholesky exceeded the declared jitter budget")
            alpha = torch.cholesky_solve(ty[:, None], factor)
            loss = .5 * (ty[:, None] * alpha).sum() + factor.diagonal().log().sum() + .5 * len(ys) * np.log(2 * np.pi)
            loss.backward()
            value, gradient = float(loss.detach()), vector.grad.detach().numpy().copy()
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                raise ValueError("nonfinite additive-kernel MLL or gradient")
            self.loss_history.append(value)
            gradient_norms.append(float(np.max(np.abs(gradient))))
            jitters.append(jitter)
            return value, gradient

        # Exposed only for in-memory numerical gradient checks.
        self._objective = objective
        result = minimize(objective, initial, jac=True, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": self.max_iter, "maxls": 30, "maxcor": 10,
                                   "maxfun": self.max_iter * 10, "ftol": 1e-9, "gtol": 1e-5})
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise ValueError("no finite additive-kernel optimization result")
        self.signal, self.noise = np.exp(result.x[:2]) * self.target_scale ** 2
        self.lengthscale = float(np.exp(result.x[2]))
        self.language_lengthscale = float(np.exp(result.x[3])) if active_language else None
        self.lambda_value = float(np.exp(result.x[4])) if self.lambda_mode == "learned" else self.fixed_lambda
        boundary_parameters = [i for i, (v, pair) in enumerate(zip(result.x, bounds)) if any(abs(v-b) < 1e-4 for b in pair)]
        self.optimized_parameters = result.x.copy()
        self._optimizer = dict(success=bool(result.success), message=str(result.message),
                               iterations=int(result.nit), evaluations=int(result.nfev),
                               boundary_hit=bool(boundary_parameters), boundary_parameter_indices=boundary_parameters,
                               method="scipy L-BFGS-B + torch exact MLL gradients", starts=1,
                               max_training_jitter_standardized=float(max(jitters, default=0.)),
                               final_gradient_infinity_norm=float(np.max(np.abs(result.jac))),
                               gradient_infinity_norm_history=gradient_norms)

    def _finalize(self, raw_y, optimizer, started):
        super()._finalize(raw_y, optimizer, started)
        latent = self.signal * self._base_kernel(self.x, self.x)
        observation = latent + self.noise * np.eye(len(self.x))
        eig = eigvalsh(observation)
        latent_eig = eigvalsh(latent)
        self.audit.update(kernel="additive_matern52", lambda_mode=self.lambda_mode,
                          lambda_value=self.lambda_value, language_lengthscale=self.language_lengthscale,
                          capacity_control=self.capacity_control,
                          scalar_parameter_count=len(self.optimized_parameters),
                          kernel_amplitude="one shared signal; both component kernels have unit diagonal",
                          feature_scaling="separate L_t coordinate means and centered global RMS for each block",
                          gnn_feature_dimension=self.gnn_dimension,
                          language_feature_dimension=self.feature_dimension-self.gnn_dimension,
                          noise_to_total_latent_signal_ratio=float(self.noise/(self.signal*(1+self.lambda_value))),
                          condition_number_observation=float(eig[-1]/eig[0]),
                          condition_number_latent=float(latent_eig[-1]/latent_eig[0]) if latent_eig[0] > 0 else None,
                          minimum_latent_eigenvalue=float(latent_eig[0]),
                          minimum_observation_eigenvalue=float(eig[0]),
                          training_seconds_including_condition_number=time.perf_counter()-started)

    def predict(self, features, *, full_covariance=False):
        # The inherited diagonal assumes a single unit kernel, so use the
        # exact combined diagonal here for both latent and observation modes.
        started = time.perf_counter()
        query = self.transform(features)
        cross = self.signal * self._base_kernel(query, self.x)
        from scipy.linalg import solve_triangular
        projected = solve_triangular(self.cholesky, cross.T, lower=True)
        mean = self.mean + cross @ self.alpha
        if full_covariance:
            variance = self.signal * self._base_kernel(query, query) - projected.T @ projected
            variance = (variance + variance.T) / 2
            minimum = float(np.linalg.eigvalsh(variance).min())
        else:
            variance = self.signal * (1+self.lambda_value) - (projected**2).sum(0)
            minimum = float(variance.min())
        if minimum < -1e-8 * max(1., self.signal*(1+self.lambda_value)):
            raise ValueError("invalid additive posterior covariance")
        if not full_covariance:
            variance = np.maximum(variance, 0.)
        self.audit["prediction_seconds"].append(time.perf_counter()-started)
        return mean, variance
