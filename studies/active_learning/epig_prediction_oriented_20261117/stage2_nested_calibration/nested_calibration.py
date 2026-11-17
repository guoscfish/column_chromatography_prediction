"""Bounded, honest nested calibration of the four stage-1 probability models.

This module never opens a data source. It consumes only the caller's currently
authorized X/y arrays. All three calibration schemes reuse the same inner OOF
fits. Outer held-out labels score predictions; they never set temperatures.
Implementation must be sealed before a real-data invocation by the driver.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtri

try:
    from qgeognn_al.active_learning_v2.epig_models import (
        GaussianProcessSurrogate, BayesianLinearSurrogate, gaussian_probability_metrics, qualify_model,
    )
except ModuleNotFoundError:
    from src.qgeognn_al.active_learning_v2.epig_models import (
        GaussianProcessSurrogate, BayesianLinearSurrogate, gaussian_probability_metrics, qualify_model,
    )


FAMILIES = ("gp_rbf", "gp_matern32", "blr_independent", "blr_coregionalized")
SCHEMES = ("identity", "nll", "central80")
OUTER_FOLDS = INNER_FOLDS = 3
OOF_SEED_OFFSET = 910007


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_hash(value):
    array = np.ascontiguousarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _arrays(path, **values):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def _read(path):
    return json.loads(Path(path).read_text())


def _validate(X, y):
    x, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(y) < 12 or y.shape[1] != 2:
        raise ValueError("nested calibration needs matching finite X and two-output y, at least 12 rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nested calibration input is nonfinite")
    return x, y


def _options(cache_contract):
    supplied = cache_contract.get("model_options", {})
    permitted = {"max_components", "maxiter", "restarts", "correlation_shrinkage"}
    if set(supplied) - permitted:
        raise ValueError("unregistered probability model option")
    return {"max_components": supplied.get("max_components", 16),
            "maxiter": supplied.get("maxiter", 60), "restarts": supplied.get("restarts", 1),
            "correlation_shrinkage": supplied.get("correlation_shrinkage", .2)}


def model_factory(family, cache_contract):
    if family not in FAMILIES:
        raise ValueError("unregistered stage-2 model family")
    options = _options(cache_contract)
    common = {key: options[key] for key in ("max_components", "maxiter")}
    if family.startswith("gp_"):
        return GaussianProcessSurrogate(kernel=family.removeprefix("gp_"), restarts=options["restarts"], **common)
    return BayesianLinearSurrogate(coregionalized=family == "blr_coregionalized",
                                   correlation_shrinkage=options["correlation_shrinkage"], **common)


def _contract(directory, kind, seed, family, x, y, supplied):
    if not isinstance(supplied, dict) or not supplied:
        raise ValueError("nonempty immutable cache_contract is required")
    if family not in FAMILIES:
        raise ValueError("unregistered stage-2 family")
    record = {"schema": "epig-nested-calibration-v1", "kind": kind, "seed": int(seed), "family": family,
              "X": _array_hash(x), "y": _array_hash(y), "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
              "outer_seed": int(seed) + OOF_SEED_OFFSET,
              "inner_seed_rule": "seed+910007+1009*(outer_fold+1)", "scaling": "full latent/noise/cross D Sigma D",
              "final_inner_seed": int(seed) + OOF_SEED_OFFSET + 1009*4,
              "schemes": list(SCHEMES), "model_options": _options(supplied), "caller_contract": supplied,
              "implementation_sha256": _hash(__file__),
              "model_implementation_sha256": _hash(GaussianProcessSurrogate.fit.__code__.co_filename)}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "contract.json"
    if path.exists() and _read(path) != record:
        raise RuntimeError("nested input/model/implementation contract drift")
    if not path.exists():
        _json(path, record)
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def _verify_complete(directory, receipt, contract_sha):
    if not receipt.exists():
        return False
    record = _read(receipt)
    if record["contract_sha256"] != contract_sha:
        raise RuntimeError("nested cache contract drift")
    for name, expected in record["files"].items():
        path = directory / name
        if not path.exists() or _hash(path) != expected:
            raise RuntimeError(f"nested cached output drift: {path}")
    return True


def estimate_temperatures(y, oof_mean, oof_covariance_blocks):
    """Predefined scales from inner held-out residuals; no outer y accepted."""
    y = np.asarray(y, dtype=np.float64)
    mean = np.asarray(oof_mean, dtype=np.float64)
    covariance = np.asarray(oof_covariance_blocks, dtype=np.float64)
    if y.shape != mean.shape or covariance.shape != (len(y), 2, 2):
        raise ValueError("inner prediction shape mismatch")
    variances = np.diagonal(covariance, axis1=1, axis2=2)
    if not np.isfinite(variances).all() or np.any(variances <= 0):
        raise np.linalg.LinAlgError("inner predictive variance must be positive")
    z = (y - mean) / np.sqrt(variances)
    temperatures = {"identity": np.ones(2), "nll": np.sqrt(np.mean(z*z, axis=0)),
                    "central80": np.quantile(np.abs(z), .8, axis=0, method="linear") / ndtri(.9)}
    if any(not np.isfinite(t).all() or np.any(t <= 0) for t in temperatures.values()):
        raise ValueError("calibration temperature is nonpositive or nonfinite; no clipping is registered")
    return temperatures


class TemperaturePosterior:
    """Scale every covariance block while preserving means and correlations.

    Full-output EPIG, including greedy conditional decisions, is invariant to
    this common invertible output transform. IVR trace endpoint weights change.
    """

    def __init__(self, posterior, temperature):
        self.base = posterior
        self.temperature = np.asarray(temperature, dtype=np.float64)
        self.n_outputs = posterior.n_outputs
        if self.temperature.shape != (self.n_outputs,) or not np.isfinite(self.temperature).all() or np.any(self.temperature <= 0):
            raise ValueError("temperature must be positive for every output")
        self.mean = posterior.mean.copy()
        self.row_ids = posterior.row_ids.copy()
        self.noise_cov = posterior.noise_cov * np.outer(self.temperature, self.temperature)

    def cov(self, left, right, noisy=False):
        scales_left = np.tile(self.temperature, len(left))
        scales_right = np.tile(self.temperature, len(right))
        return self.base.cov(left, right, noisy=noisy) * np.outer(scales_left, scales_right)

    def latent_diag(self, indices):
        return self.base.latent_diag(indices) * np.outer(self.temperature, self.temperature)

    def noise_blocks(self, indices):
        return self.base.noise_blocks(indices) * np.outer(self.temperature, self.temperature)

    def predict(self, noisy=True):
        blocks = self.latent_diag(np.arange(len(self.row_ids)))
        if noisy:
            blocks += self.noise_cov
        return self.mean.copy(), np.diagonal(blocks, axis1=1, axis2=2).copy()


class CalibratedPosterior(TemperaturePosterior):
    """Public formal-run adapter; width_temperature is a positive output vector."""

    def __init__(self, posterior, width_temperature):
        super().__init__(posterior, width_temperature)


def _fold_metrics(y, posterior, fold, train, held, temperature, inner_seed, model_audit):
    positions = np.arange(len(y))
    blocks = posterior.latent_diag(positions) + posterior.noise_blocks(positions)
    covariance = posterior.cov(positions, positions, noisy=True)
    factor = cho_factor(covariance, lower=True)
    residual = (y - posterior.mean).ravel()
    mahalanobis = float(residual @ cho_solve(factor, residual))
    joint = .5 * (mahalanobis + 2*np.log(np.diag(factor[0])).sum() + len(residual)*np.log(2*np.pi))
    diag = np.diag(covariance)
    diagonal_nll = .5 * np.sum(residual*residual/diag + np.log(diag) + np.log(2*np.pi))
    block = gaussian_probability_metrics(y, posterior.mean, blocks)
    correlation = covariance / np.sqrt(np.outer(diag, diag))
    result = {"fold": fold, "train_rows": len(train), "held_out_rows": len(held),
              "train_positions": train.tolist(), "held_out_positions": held.tolist(),
              "full_joint_nll_per_row": float(joint/len(y)), "full_joint_nll_per_scalar": float(joint/len(residual)),
              "diagonal_nll_per_scalar": float(diagonal_nll/len(residual)),
              "block_diagonal_nll_per_scalar": float(block["joint_nll"]/2),
              "whitened_residual_rms": float(np.sqrt(mahalanobis/len(residual))),
              "minimum_predictive_eigenvalue": float(np.linalg.eigvalsh(covariance)[0]),
              "mean_absolute_offdiagonal_predictive_correlation": float(np.mean(np.abs(correlation[np.triu_indices(len(correlation), 1)]))),
              "temperature": temperature.tolist(), "inner_seed": int(inner_seed), "model": model_audit}
    if result["minimum_predictive_eigenvalue"] <= 0:
        raise np.linalg.LinAlgError("calibrated outer predictive covariance is nonpositive")
    return result, posterior.mean.copy(), blocks


def _cached_inner(seed, family, x, y, directory, contract, event, before_fit):
    def factory():
        if before_fit is not None:
            # qualify_model has just persisted this attempt before invoking us.
            latest = sorted(directory.glob("attempt_*.json"))[-1]
            before_fit({**event, "phase": "inner_oof", "inner_fold": _read(latest)["fold"], "attempt_path": str(latest)})
        return model_factory(family, contract)
    return qualify_model(factory, x, y, n_splits=INNER_FOLDS, seed=seed,
                         cache_directory=directory, cache_contract={**contract, "nested_event": event})


def _cached_full_model(seed, family, x, y, directory, contract, event, before_fit):
    directory.mkdir(parents=True, exist_ok=True)
    # Caller/fold identity and X/y are already included in the outer contract;
    # bind this cache independently so final fitting can share the same helper.
    contract_sha = _contract(directory, "full_model", seed, family, x, y, contract)
    receipt = directory / "complete.json"
    if _verify_complete(directory, receipt, contract_sha):
        with (directory / "model.pkl").open("rb") as handle:
            return pickle.load(handle)
    attempts = sorted(directory.glob("attempt_*.json"))
    attempt = directory / f"attempt_{len(attempts)+1:04d}.json"
    with attempt.open("x") as handle:
        json.dump({**event, "phase": "outer_or_full_fit", "status": "STARTED", "started_unix": time.time()}, handle)
    if before_fit is not None:
        before_fit({**event, "phase": "outer_or_full_fit", "attempt_path": str(attempt)})
    model = model_factory(family, contract).fit(x, y)
    temporary = directory / "model.pkl.tmp"
    with temporary.open("wb") as handle:
        pickle.dump(model, handle)
    temporary.replace(directory / "model.pkl")
    _json(directory / "fit_audit.json", model.audit_)
    _json(receipt, {"contract_sha256": contract_sha,
                    "files": {name: _hash(directory/name) for name in ("model.pkl", "fit_audit.json")}})
    record = _read(attempt)
    record.update({"status": "COMPLETE", "finished_unix": time.time(), "receipt_sha256": _hash(receipt)})
    _json(attempt, record)
    return model


def run_nested_family(seed, family, X_L333, y_L333, cache_directory, cache_contract, before_fit=None):
    """Three outer folds × (three inner fits + one outer fit), twelve total.

    Completed folds and completed inner/outer model fits are reused after an
    interruption. ``before_fit(event)`` reserves a global budget before each
    actual model fit; cache replays never call it. Caller holds a run lock and
    enforces the global 240+20+3 attempt and cumulative wall budgets.
    """
    x, y = _validate(X_L333, y_L333)
    directory = Path(cache_directory)
    contract_sha = _contract(directory, "outer_oof", seed, family, x, y, cache_contract)
    outer_seed = int(seed) + OOF_SEED_OFFSET
    split = np.array_split(np.random.default_rng(outer_seed).permutation(len(y)), OUTER_FOLDS)
    all_positions = np.arange(len(y))
    candidates = {name: {"oof_mean": np.empty_like(y), "oof_covariance_blocks": np.empty((len(y),2,2)), "folds": []} for name in SCHEMES}
    temperatures = []
    for fold, held in enumerate(split):
        train = np.setdiff1d(all_positions, held)
        fold_dir = directory / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        complete = fold_dir / "complete.json"
        evaluation_path, arrays_path = fold_dir / "evaluation.json", fold_dir / "oof_arrays.npz"
        if _verify_complete(fold_dir, complete, contract_sha):
            saved = _read(evaluation_path)
            if saved["train_positions"] != train.tolist() or saved["held_out_positions"] != held.tolist():
                raise RuntimeError("cached outer partition drift")
            with np.load(arrays_path, allow_pickle=False) as arrays:
                for scheme in SCHEMES:
                    candidates[scheme]["oof_mean"][held] = arrays[f"{scheme}_mean"]
                    candidates[scheme]["oof_covariance_blocks"][held] = arrays[f"{scheme}_covariance_blocks"]
                    candidates[scheme]["folds"].append(saved["fold_metrics"][scheme])
            temperatures.append(saved["temperatures"])
            continue
        inner_seed = int(seed) + OOF_SEED_OFFSET + 1009*(fold+1)
        event = {"seed": int(seed), "family": family, "outer_fold": fold, "scope": "nested_qualification"}
        inner = _cached_inner(inner_seed, family, x[train], y[train], fold_dir/"inner", cache_contract, event, before_fit)
        scales = estimate_temperatures(y[train], inner["oof_mean"], inner["oof_covariance_blocks"])
        model = _cached_full_model(seed, family, x[train], y[train], fold_dir/"outer", cache_contract, event, before_fit)
        posterior = model.posterior(x[held], row_ids=held)
        arrays, fold_metrics = {}, {}
        for scheme in SCHEMES:
            wrapped = TemperaturePosterior(posterior, scales[scheme])
            result, means, blocks = _fold_metrics(y[held], wrapped, fold, train, held, scales[scheme], inner_seed, model.audit_)
            candidates[scheme]["oof_mean"][held] = means
            candidates[scheme]["oof_covariance_blocks"][held] = blocks
            candidates[scheme]["folds"].append(result)
            fold_metrics[scheme] = result
            arrays[f"{scheme}_mean"], arrays[f"{scheme}_covariance_blocks"] = means, blocks
        temperature_record = {"fold": fold, "inner_seed": inner_seed, **{key: value.tolist() for key,value in scales.items()}}
        temperatures.append(temperature_record)
        _arrays(arrays_path, **arrays)
        _json(evaluation_path, {"train_positions": train.tolist(), "held_out_positions": held.tolist(),
                               "temperatures": temperature_record, "fold_metrics": fold_metrics})
        _json(complete, {"contract_sha256": contract_sha,
                         "files": {path.name: _hash(path) for path in (evaluation_path, arrays_path)}})
    for value in candidates.values():
        value["metrics"] = gaussian_probability_metrics(y, value["oof_mean"], value["oof_covariance_blocks"])
    return {"candidates": candidates, "temperatures": temperatures,
            "fit_attempt_count": len(list(directory.rglob("attempt_*.json"))),
            "outer_seed": outer_seed, "qualification_scope": "nested train-only L333 OOF; outer held labels never set temperatures"}


def fit_calibrated_full(seed, family, scheme, X_L333, y_L333, cache_directory, cache_contract, before_fit=None):
    """Three inner OOF fits + one full fit after every seed qualifies.

    Returns a fitted uncalibrated model plus the selected temperature. Apply
    ``TemperaturePosterior(model.posterior(...), temperature)`` to all future
    prediction/acquisition blocks; never scale marginal variances alone.
    """
    if scheme not in SCHEMES:
        raise ValueError("unregistered calibration scheme")
    x, y = _validate(X_L333, y_L333)
    directory = Path(cache_directory)
    _contract(directory, "final_calibration", seed, family, x, y, {**cache_contract,"selected_scheme":scheme})
    event = {"seed": int(seed), "family": family, "scope": "final_calibrated_full", "scheme": scheme}
    inner_seed = int(seed) + OOF_SEED_OFFSET + 1009*4
    inner = _cached_inner(inner_seed, family, x, y, directory/"inner", cache_contract, event, before_fit)
    temperatures = estimate_temperatures(y, inner["oof_mean"], inner["oof_covariance_blocks"])
    model = _cached_full_model(seed, family, x, y, directory/"full", cache_contract, event, before_fit)
    _json(directory/"temperatures.json", {"inner_seed": inner_seed, "selected_scheme": scheme,
                                         **{k:v.tolist() for k,v in temperatures.items()}})
    return {"model": model, "temperature": temperatures[scheme], "scheme": scheme, "family": family,
            "fit_attempt_count": len(list(directory.rglob("attempt_*.json"))), "model_audit": model.audit_}
