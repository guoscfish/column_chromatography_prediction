"""Controlled float64 exact-MLL optimization and independent variance scaling."""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from data_boundary import LEGACY
from gp_models import ProjectedMaternGP, FixedRepresentationGP, regression_metrics


class ControlledProjectionGP(ProjectedMaternGP):
    """One-factor ablations of the historical implementation; no evaluation input."""
    def __init__(self, *, method='converged', max_iter=200, seed=17, projection_dim=64,
                 ridge=.01, noise_floor=.02, stage_iterations=100):
        super().__init__(projection_dim=projection_dim, max_iter=max_iter, seed=seed)
        if method not in ('original', 'converged', 'regularized', 'noise_floor', 'staged'):
            raise ValueError('unknown frozen method')
        self.method, self.ridge, self.noise_floor = method, ridge, noise_floor
        self.stage_iterations = stage_iterations

    def _optimize(self, xs, ys):
        import torch
        torch.set_num_threads(self.threads)
        tx, ty = torch.as_tensor(xs, dtype=torch.float64), torch.as_tensor(ys, dtype=torch.float64)
        n, p = xs.shape
        q = self.projection_dim
        offset = p * q
        rng = np.random.default_rng(self.seed)
        initial = np.r_[rng.normal(0., 1 / np.sqrt(p), size=(p, q)).ravel(), np.zeros(q), 0., -2., 0.]
        bounds = [(None, None)] * (offset + q) + [self.variance_bounds, self.variance_bounds, self.lengthscale_bounds]
        if self.method == 'noise_floor':
            bounds[-2] = (np.log(self.noise_floor), self.variance_bounds[1])
        self.loss_history, records = [], []
        identity = torch.eye(n, dtype=torch.float64)

        def objective(parameters, frozen=None):
            vector = torch.tensor(parameters, dtype=torch.float64, requires_grad=True)
            if frozen is None:
                z = torch.nn.functional.elu(tx @ vector[:offset].reshape(p, q) + vector[offset:-3])
                norm = (z * z).sum(1)
                d2 = (norm[:, None] + norm[None, :] - 2 * z @ z.T).clamp_min(0.) / q
            else:
                d2 = frozen
            scaled = np.sqrt(5.) * torch.sqrt(d2.clamp_min(1e-24)) / vector[-1].exp()
            covariance = vector[-3].exp() * (1 + scaled + scaled**2 / 3) * (-scaled).exp() + vector[-2].exp() * identity
            factor = None
            for relative in (0., 1e-12, 1e-10, 1e-8):
                jitter = relative * max(1., float((vector[-3].exp() + vector[-2].exp()).detach()))
                candidate, info = torch.linalg.cholesky_ex(covariance + jitter * identity)
                if int(info.max().detach()) == 0:
                    factor = candidate
                    break
            if factor is None:
                raise ValueError('declared Cholesky jitter budget exhausted')
            alpha = torch.cholesky_solve(ty[:, None], factor).ravel()
            nll = .5 * ty @ alpha + factor.diagonal().log().sum() + .5 * n * np.log(2 * np.pi)
            penalty = torch.tensor(0., dtype=torch.float64)
            if self.method == 'regularized' and frozen is None:
                # Gaussian weight prior: precision ridge * input_dimension.
                penalty = .5 * self.ridge * p * (vector[:offset+q]**2).sum()
            loss = nll + penalty
            loss.backward()
            value, grad = float(loss.detach()), vector.grad.detach().numpy().copy()
            if not np.isfinite(value) or not np.isfinite(grad).all():
                raise ValueError('nonfinite exact-MLL objective/gradient')
            self.loss_history.append(float(nll.detach()))
            records.append({'penalized_nll': value, 'nll': float(nll.detach()), 'penalty': float(penalty.detach()),
                            'gradient_inf': float(np.max(np.abs(grad))), 'noise_standardized': float(vector[-2].exp().detach()),
                            'jitter_standardized': jitter})
            return value, grad

        options = {'maxiter': self.max_iter, 'maxls': 20, 'maxfun': self.max_iter*10, 'maxcor': 10,
                   'ftol': 1e-9, 'gtol': 1e-5}
        result = minimize(objective, initial, jac=True, bounds=bounds, method='L-BFGS-B', options=options)
        first = result.x.copy()
        first_status = {'success': bool(result.success), 'message': str(result.message), 'iterations': int(result.nit)}
        if self.method == 'staged':
            h = xs @ first[:offset].reshape(p, q) + first[offset:-3]
            z = np.where(h > 0., h, np.expm1(np.minimum(h, 0.)))
            from scipy.spatial.distance import cdist
            frozen_d2 = torch.as_tensor(cdist(z, z, 'sqeuclidean') / q, dtype=torch.float64)
            result = minimize(lambda hp: objective(hp, frozen_d2), first[-3:], jac=True, bounds=bounds[-3:],
                              method='L-BFGS-B', options={**options, 'maxiter': self.stage_iterations,
                                                        'maxfun': self.stage_iterations*10})
            parameters = np.r_[first[:-3], result.x]
            value, gradient = objective(result.x, frozen_d2)
            final_bounds = bounds[-3:]
        else:
            parameters = result.x
            value, gradient = objective(parameters)
            final_bounds = bounds
        if not np.isfinite(parameters).all():
            raise ValueError('invalid fitted parameter vector')
        self.projection_weight = parameters[:offset].reshape(p, q).copy()
        self.projection_bias = parameters[offset:-3].copy()
        self.signal, self.noise = np.exp(parameters[-3:-1]) * self.target_scale**2
        self.lengthscale = float(np.exp(parameters[-1]))
        pg = gradient.copy()
        for k, (lower, upper) in enumerate(final_bounds):
            v = result.x[k]
            if lower is not None and v <= lower + 1e-8 and pg[k] > 0:
                pg[k] = 0
            if upper is not None and v >= upper - 1e-8 and pg[k] < 0:
                pg[k] = 0
        self._optimizer = {'success': bool(result.success), 'message': str(result.message),
                           'iterations': int(result.nit), 'evaluations': int(result.nfev),
                           'boundary_hit': bool(any(abs(v-b) < 1e-4 for v, bs in zip(parameters[-3:], bounds[-3:]) for b in bs)),
                           'method': 'scipy L-BFGS-B + torch exact MLL gradients', 'starts': 1,
                           'max_training_jitter_standardized': max(r['jitter_standardized'] for r in records),
                           'first_stage': first_status, 'final_gradient_inf': float(np.max(np.abs(gradient))),
                           'final_projected_gradient_inf': float(np.max(np.abs(pg))),
                           'final_objective': value, 'objective_evaluations': records,
                           'representation_displacement_rms': float(np.sqrt(np.mean((parameters[:-3]-initial[:-3])**2)))}
        self.initial_weight = initial[:offset].reshape(p, q)
        self.initial_bias = initial[offset:-3]

    def fit(self, features, targets):
        super().fit(features, targets)
        z = self.x
        xs = self.transform(features)
        hi = xs @ self.initial_weight + self.initial_bias
        zi = np.where(hi > 0., hi, np.expm1(np.minimum(hi, 0.)))
        initial_dist, final_dist = pdist(zi) / np.sqrt(self.projection_dim), pdist(z) / np.sqrt(self.projection_dim)
        valid = initial_dist > 1e-10
        ratio = final_dist[valid] / initial_dist[valid]
        gram = self.signal * self._base_kernel(z, z)
        eig = np.linalg.eigvalsh(gram)
        self.audit.update(method_ablation=self.method, projection_penalty=(f'0.5*{self.ridge}*p*||W,b||^2' if self.method=='regularized' else 'none'),
                          standardized_noise_floor=(self.noise_floor if self.method=='noise_floor' else float(np.exp(-9))),
                          noise_signal_ratio=float(self.noise/self.signal),
                          covariance_condition_number=float((eig[-1]+self.noise)/(max(eig[0],0)+self.noise)),
                          projection_weight_norm=float(np.linalg.norm(self.projection_weight)),
                          projection_bias_norm=float(np.linalg.norm(self.projection_bias)),
                          distance_ratio_quantiles=np.quantile(ratio,[0,.01,.1,.5,.9,.99,1]).tolist() if len(ratio) else [],
                          compressed_pair_fraction=float(np.mean(ratio<.1)) if len(ratio) else None,
                          kernel_distance_quantiles=np.quantile(final_dist/self.lengthscale,[0,.01,.1,.5,.9,.99,1]).tolist(),
                          identifiability='positive ELU branch is homogeneous; global projection scale can trade against ell; ELU negative branch breaks exact invariance')
        return self


def variance_temperature(targets, mean, observation_variance):
    """Gaussian variance MLE on the separate calibration partition only."""
    y, mu, v = map(lambda a: np.asarray(a, dtype=np.float64), (targets, mean, observation_variance))
    if y.shape != mu.shape or v.shape != y.shape or len(y) < 2 or not np.isfinite(y+mu+v).all() or np.any(v<=0):
        raise ValueError('finite positive aligned calibration inputs required')
    return float(np.clip(np.mean((y-mu)**2/v),1e-6,1e6))
