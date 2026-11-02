"""Matched-benchmark calibration controls.

This module is separate from :mod:`calibration` so byte-frozen historical
studies that hash that module remain reproducible.  New matched studies import
the local identity-shrinkage helpers from the public transfer package.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .calibration import CalibrationFit


def _two_targets(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    converted = tuple(np.asarray(value, dtype=float) for value in arrays)
    if any(value.ndim != 2 or value.shape[1] != 2 for value in converted):
        raise ValueError("calibration arrays must have shape (n, 2)")
    if any(not np.isfinite(value).all() for value in converted):
        raise ValueError("calibration arrays must be finite")
    return converted


def fit_local_identity_shrinkage(
    train_truth: np.ndarray,
    train_source: np.ndarray,
    predict_source: np.ndarray,
    *,
    mass_ratio: float = 1.0,
    source_scales: Sequence[float] = (1.0, 1.0),
    strength: float = 1.0,
) -> CalibrationFit:
    """Fit independent affine maps shrunk toward mass-ratio identity.

    The fit consumes only ``train_truth``/``train_source`` labels.  In
    normalized coordinates each target is regularized toward slope 1 and
    intercept 0; ``strength=0`` reduces to ordinary affine calibration.
    """

    truth, source, values = _two_targets(train_truth, train_source, predict_source)
    scales = np.asarray(source_scales, dtype=float)
    ratio = float(mass_ratio)
    penalty = float(strength)
    if len(truth) == 0 or len(source) != len(truth):
        raise ValueError("calibration fit needs aligned non-empty training arrays")
    if scales.shape != (2,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("source_scales must contain two positive finite values")
    if not np.isfinite(ratio) or ratio <= 0 or not np.isfinite(penalty) or penalty < 0:
        raise ValueError("mass_ratio must be positive and strength non-negative")
    n = len(truth)
    coefficients = np.empty((2, 2), dtype=float)
    prediction = np.empty_like(values)
    for target in range(2):
        design = np.column_stack([source[:, target] / scales[target], np.ones(n)])
        if penalty == 0 and np.linalg.matrix_rank(design) < 2:
            raise ValueError("identity-shrinkage affine design is rank deficient")
        if penalty:
            augmented = np.vstack([design / np.sqrt(n), np.sqrt(penalty) * np.eye(2)])
            labels = np.r_[
                truth[:, target] / (ratio * scales[target] * np.sqrt(n)),
                np.sqrt(penalty) * np.array([1.0, 0.0]),
            ]
        else:
            augmented = design / np.sqrt(n)
            labels = truth[:, target] / (ratio * scales[target] * np.sqrt(n))
        coefficients[target] = np.linalg.lstsq(augmented, labels, rcond=None)[0]
        prediction[:, target] = (
            values[:, target] / scales[target] * coefficients[target, 0]
            + coefficients[target, 1]
        ) * ratio * scales[target]
    return CalibrationFit(prediction, coefficients)


def select_local_identity_shrinkage(
    train_truth: np.ndarray,
    train_source: np.ndarray,
    validation_truth: np.ndarray,
    validation_source: np.ndarray,
    strengths: Sequence[float],
    *,
    mass_ratio: float = 1.0,
    source_scales: Sequence[float] = (1.0, 1.0),
) -> tuple[CalibrationFit, float, str]:
    """Select a fixed shrinkage grid using validation labels only."""

    valid_truth, valid_source = _two_targets(validation_truth, validation_source)
    candidates = tuple(float(value) for value in strengths)
    if not candidates or any(not np.isfinite(value) or value < 0 for value in candidates):
        raise ValueError("strengths must contain non-negative finite values")
    scales = np.asarray(source_scales, dtype=float)
    fits = [
        fit_local_identity_shrinkage(
            train_truth,
            train_source,
            valid_source,
            mass_ratio=mass_ratio,
            source_scales=scales,
            strength=strength,
        )
        for strength in candidates
    ]
    scores = [float(np.sqrt(np.mean(np.square((valid_truth - fit.prediction) / scales)))) for fit in fits]
    selected = min(range(len(candidates)), key=lambda index: (scores[index], candidates[index]))
    return fits[selected], candidates[selected], "validation_only_minimum_normalized_rmse"


__all__ = ["fit_local_identity_shrinkage", "select_local_identity_shrinkage"]
