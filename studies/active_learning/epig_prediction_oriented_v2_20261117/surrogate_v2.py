"""Study-local feature/group GP ablations. No label-store or evaluation access.

F2 standardizes each column on the training fold, then weights the four
declared feature groups equally before optional PCA. This order matters:
weighting before another per-column standardization would cancel the weights.
The probability model remains Gaussian with independent endpoint noise.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from qgeognn_al.active_learning_v2.epig_models import (
    FeatureTransform, GaussianProcessSurrogate, _matrix, _prepare_y,
)


class GroupedTransform(FeatureTransform):
    def fit(self, x):
        x = _matrix(x, "X")
        if x.shape[1] != 31:
            raise ValueError("F2 requires F1's declared 31 columns")
        self.center = x.mean(0)
        self.scale = np.where(x.std(0) > 1e-12, x.std(0), 1.)
        # Molecular 16; eluent 6; solvent category 3; loading quantities 6.
        self.group_weights = np.concatenate([np.full(n, 1/np.sqrt(n)) for n in (16, 6, 3, 6)])
        z = (x-self.center)/self.scale*self.group_weights
        components = min(x.shape[1], len(x)-1)
        if self.max_components is not None:
            components = min(components, self.max_components)
        if components < x.shape[1]:
            _, s, vt = np.linalg.svd(z, full_matrices=False)
            self.rotation = vt[:components].T
            self.retained_variance_fraction = float((s[:components]**2).sum()/(s**2).sum())
        else:
            self.rotation = np.eye(x.shape[1])
            self.retained_variance_fraction = 1.
        self.input_dimension = x.shape[1]
        return self

    def transform(self, x):
        x = _matrix(x, "X")
        if x.shape[1] != self.input_dimension:
            raise ValueError("feature dimension drift")
        return (((x-self.center)/self.scale)*self.group_weights)@self.rotation


class StudyGP(GaussianProcessSurrogate):
    """Same Matérn likelihood as v1, with explicit group transform and starts.

    Kept study-local to preserve every source hash in the original freeze.
    ``restart_audit`` includes every optimization, rather than only the winner.
    """
    def __init__(self, grouped=False, max_components=16, maxiter=200, restarts=3):
        super().__init__("matern32", max_components, maxiter, restarts)
        self.grouped = grouped

    def fit(self, X_train, y_train, mean_train=None, output_scales=None):
        started = time.perf_counter()
        x = _matrix(X_train, "X_train")
        z, self.scales_, self.center_ = _prepare_y(y_train, mean_train, output_scales)
        if len(x) != len(z) or len(x)<3:
            raise ValueError("invalid training rows")
        transform = GroupedTransform if self.grouped else FeatureTransform
        self.transform_ = transform(self.max_components).fit(x)
        self.train_x_ = self.transform_.transform(x)
        self.n_outputs = z.shape[1]
        distance = cdist(self.train_x_, self.train_x_, "sqeuclidean")
        initial_length = max(float(np.median(np.sqrt(distance))), .1)
        bounds = [(-9.21, 6.91), (np.log(initial_length/100), np.log(initial_length*100)), (-11.51, 3.)]
        self.models_, restart_audit = [], []
        for endpoint in range(self.n_outputs):
            target = z[:, endpoint]
            def objective(theta):
                signal, length, noise = np.exp(theta)
                r = np.sqrt(3*distance/length**2)
                prior = signal*(1+r)*np.exp(-r)
                length_gradient = signal*r*r*np.exp(-r)
                k = prior.copy()
                k.flat[::len(k)+1] += noise
                try:
                    factor = cho_factor(k, lower=True, check_finite=False)
                except np.linalg.LinAlgError:
                    return 1e100, np.zeros(3)
                alpha = cho_solve(factor, target, check_finite=False)
                nll = .5*target@alpha+np.log(np.diag(factor[0])).sum()+len(k)/2*np.log(2*np.pi)
                sensitivity = cho_solve(factor, np.eye(len(k)), check_finite=False)-np.outer(alpha, alpha)
                gradient = .5*np.array([np.sum(sensitivity*prior), np.sum(sensitivity*length_gradient), noise*np.trace(sensitivity)])
                return float(nll), gradient
            fits = []
            for restart in range(self.restarts):
                t0 = np.log([max(float(np.var(target)), .1), initial_length*2**restart, .1*2**restart])
                fit = minimize(objective, t0, jac=True, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter":self.maxiter,"maxls":20,"ftol":1e-8})
                fits.append(fit)
                restart_audit.append({"endpoint":endpoint,"restart":restart,"nll":float(fit.fun),"optimizer_success":bool(fit.success),
                                      "nit":int(fit.nit),"nfev":int(fit.nfev),"gradient_max_abs":float(np.abs(fit.jac).max())})
            fit = min(fits, key=lambda r:r.fun)
            if not np.isfinite(fit.fun) or fit.fun>=1e99:
                raise np.linalg.LinAlgError("no valid GP optimization")
            k = self._kernel(self.train_x_, self.train_x_, fit.x)
            k.flat[::len(k)+1] += np.exp(fit.x[2])
            factor = cho_factor(k, lower=True, check_finite=False)
            self.models_.append({"theta":fit.x,"factor":factor,"alpha":cho_solve(factor,target,check_finite=False),
                                 "nll":float(fit.fun),"optimizer_success":bool(fit.success),"optimizer_message":str(fit.message),
                                 "nit":int(fit.nit),"nfev":int(fit.nfev),"gradient_max_abs":float(np.abs(fit.jac).max())})
        self.noise_cov_ = np.diag([np.exp(m["theta"][2]) for m in self.models_])*np.outer(self.scales_,self.scales_)
        self.audit_ = {"family":"study_local_independent_exact_gp","kernel":self.kernel,"n_train":len(x),"n_outputs":self.n_outputs,
                       "feature_dimension":self.train_x_.shape[1],"grouped":self.grouped,"maxiter":self.maxiter,"restarts":self.restarts,
                       "fit_seconds":time.perf_counter()-started,"pca_retained_standardized_input_variance_fraction":self.transform_.retained_variance_fraction,
                       "restart_audit":restart_audit,
                       "parameters":[{"signal_variance":float(np.exp(m["theta"][0])),"lengthscale":float(np.exp(m["theta"][1])),
                                      "noise_variance_scaled":float(np.exp(m["theta"][2])),
                                      **{k:m[k] for k in ("nll","optimizer_success","optimizer_message","nit","nfev","gradient_max_abs")}}
                                     for m in self.models_]}
        return self
