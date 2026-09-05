"""Low-dimensional calibration models for cross-column transfer studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class CalibrationFit:
    prediction: np.ndarray
    coefficients: np.ndarray


def _two_targets(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    converted = tuple(np.asarray(value, dtype=float) for value in arrays)
    if any(value.ndim != 2 or value.shape[1] != 2 for value in converted):
        raise ValueError("calibration arrays must have shape (n, 2)")
    if any(not np.isfinite(value).all() for value in converted):
        raise ValueError("calibration arrays must be finite")
    return converted


def fit_scale_only(
    train_truth: np.ndarray, train_source: np.ndarray, predict_source: np.ndarray
) -> CalibrationFit:
    """Fit one through-origin scale per target using training labels only."""
    truth, source, values = _two_targets(train_truth, train_source, predict_source)
    if len(truth) != len(source) or not len(truth):
        raise ValueError("scale-only fit needs aligned non-empty training arrays")
    denominator = np.square(source).sum(axis=0)
    if np.any(denominator <= 1e-12):
        raise ValueError("scale-only fit has a degenerate source prediction")
    scale = (source * truth).sum(axis=0) / denominator
    return CalibrationFit(values * scale, scale.reshape(2, 1))


def fit_affine(
    train_truth: np.ndarray, train_source: np.ndarray, predict_source: np.ndarray
) -> CalibrationFit:
    """Fit one slope and intercept per target using training labels only."""
    truth, source, values = _two_targets(train_truth, train_source, predict_source)
    if len(truth) != len(source) or len(truth) < 2:
        raise ValueError("affine fit needs at least two aligned training rows")
    coefficients = np.empty((2, 2), dtype=float)
    prediction = np.empty_like(values)
    for target in range(2):
        design = np.column_stack([source[:, target], np.ones(len(source))])
        coefficients[target] = np.linalg.lstsq(design, truth[:, target], rcond=None)[0]
        prediction[:, target] = values[:, target] * coefficients[target, 0] + coefficients[target, 1]
    return CalibrationFit(prediction, coefficients)


def mass_ratio_prediction(source_prediction: np.ndarray, target_mass_g: float) -> np.ndarray:
    """Return the explicitly descriptive target-mass/4g scaling baseline."""
    (source,) = _two_targets(source_prediction)
    if target_mass_g <= 0:
        raise ValueError("target column mass must be positive")
    return source * (float(target_mass_g) / 4.0)


def fit_affine_condition_residual(
    train_truth: np.ndarray,
    train_source: np.ndarray,
    train_conditions: np.ndarray,
    train_groups: np.ndarray,
    predict_source: np.ndarray,
    predict_conditions: np.ndarray,
    alphas: tuple[float, ...] | list[float],
    metric_scales: np.ndarray,
) -> tuple[CalibrationFit, float, str]:
    """Fit affine plus condition Ridge with nested, group-isolated alpha selection.

    Each inner fold refits both the affine component and Ridge component. This
    prevents an affine fit on an inner validation row from leaking into alpha
    selection.
    """
    truth, source, values = _two_targets(train_truth, train_source, predict_source)
    conditions = np.asarray(train_conditions, dtype=float)
    predict_conditions = np.asarray(predict_conditions, dtype=float)
    groups = np.asarray(train_groups).astype(str)
    scales = np.asarray(metric_scales, dtype=float)
    alpha_grid = tuple(float(value) for value in alphas)
    if len(truth) != len(source) or len(truth) != len(conditions) or len(groups) != len(truth):
        raise ValueError("residual fit inputs are not aligned")
    if conditions.ndim != 2 or predict_conditions.ndim != 2 or conditions.shape[1] != predict_conditions.shape[1]:
        raise ValueError("condition matrices must be aligned 2D arrays")
    if scales.shape != (2,) or np.any(scales <= 0) or not alpha_grid:
        raise ValueError("invalid residual selection configuration")

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2 or len(truth) < 4:
        selected, policy = 1.0, "deterministic_alpha_1_insufficient_groups"
    else:
        folds = min(5, len(unique_groups))
        candidates: list[tuple[float, float]] = []
        for alpha in alpha_grid:
            scores = []
            for inner_train, inner_valid in GroupKFold(n_splits=folds).split(truth, groups=groups):
                inner_affine_train = fit_affine(
                    truth[inner_train], source[inner_train], source[inner_train]
                )
                inner_affine_valid = fit_affine(
                    truth[inner_train], source[inner_train], source[inner_valid]
                )
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                model.fit(
                    conditions[inner_train],
                    truth[inner_train] - inner_affine_train.prediction,
                )
                prediction = inner_affine_valid.prediction + model.predict(conditions[inner_valid])
                rmse = np.sqrt(np.mean(np.square(truth[inner_valid] - prediction), axis=0))
                scores.append(float(np.mean(rmse / scales)))
            candidates.append((float(np.mean(scores)), alpha))
        _, selected = min(candidates)
        policy = f"groupkfold_{folds}_gradient_train_only_refit_affine"

    affine_train = fit_affine(truth, source, source)
    affine_predict = fit_affine(truth, source, values)
    model = make_pipeline(StandardScaler(), Ridge(alpha=selected))
    model.fit(conditions, truth - affine_train.prediction)
    prediction = affine_predict.prediction + model.predict(predict_conditions)
    return CalibrationFit(prediction, affine_predict.coefficients), selected, policy
