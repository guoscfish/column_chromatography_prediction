"""Fair nested extensions of the repository's Conditional EA calibration.

The historical ``fit_conditional`` implementation is intentionally untouched.
This module mirrors its exact normalization and Ridge prior, then appends at
most one extra varying-slope effect per endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ConditionalEANestedExtension:
    coefficients: np.ndarray
    ea_mean: float
    ea_feature_mean: np.ndarray
    ea_feature_std: np.ndarray
    extra_means: np.ndarray
    extra_stds: np.ndarray
    source_scales: np.ndarray
    mass_ratio: float
    penalty: float
    extra_names: tuple[str, ...]

    def predict(self, source: np.ndarray, ea: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
        source = np.asarray(source, dtype=float)
        ea = np.asarray(ea, dtype=float)
        if source.ndim != 2 or source.shape[1] != 2 or ea.shape != (len(source),):
            raise ValueError("source and EA are not aligned")
        if extra is None:
            extra = np.zeros((len(source), 2, len(self.extra_names)), dtype=float)
        extra = np.asarray(extra, dtype=float)
        if extra.shape != (len(source), 2, len(self.extra_names)):
            raise ValueError("extra effects are not aligned")
        u = source / self.source_scales
        ea_raw = u * (ea - self.ea_mean)[:, None]
        ea_feature = (ea_raw - self.ea_feature_mean) / self.ea_feature_std
        extra_raw = u[:, :, None] * ((extra - self.extra_means[None, :, :]) / self.extra_stds[None, :, :])
        design = np.concatenate([u[:, :, None], np.ones((len(u), 2, 1)),
                                 ea_feature[:, :, None], extra_raw], axis=2)
        normalized = np.einsum("ntk,tk->nt", design, self.coefficients)
        return normalized * self.mass_ratio * self.source_scales

    def audit(self) -> dict[str, object]:
        return {
            "coefficients": self.coefficients.tolist(),
            "ea_mean": float(self.ea_mean),
            "ea_feature_mean": self.ea_feature_mean.tolist(),
            "ea_feature_std": self.ea_feature_std.tolist(),
            "extra_means": self.extra_means.tolist(),
            "extra_stds": self.extra_stds.tolist(),
            "source_scales": self.source_scales.tolist(),
            "mass_ratio": float(self.mass_ratio),
            "penalty": float(self.penalty),
            "extra_names": list(self.extra_names),
            "free_coefficients": int(self.coefficients.size),
            "regularization": "SSE/n + penalty*((base_slope-1)^2 + intercept^2 + EA_effect^2 + extra_effect^2)",
        }


def fit_scalar_conditional(source: np.ndarray, truth: np.ndarray, ea: np.ndarray,
                           source_scale: float, mass_ratio: float, penalty: float) -> ConditionalEANestedExtension:
    """Fit one 3-coefficient Conditional-EA head for center or width."""

    source = np.asarray(source, dtype=float).reshape(-1, 1)
    truth = np.asarray(truth, dtype=float).reshape(-1, 1)
    fit = fit_conditional_nested(np.repeat(source, 2, axis=1), np.repeat(truth, 2, axis=1),
                                 np.asarray(ea, dtype=float), [source_scale, source_scale], mass_ratio, penalty)
    return fit


def _validate(source: np.ndarray, truth: np.ndarray, ea: np.ndarray, scales: np.ndarray, mass_ratio: float, penalty: float) -> None:
    if source.shape != truth.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("aligned source/truth with shape (n, 2) required")
    if ea.shape != (len(source),) or len(source) < 4:
        raise ValueError("aligned EA and sufficient rows required")
    if not all(np.isfinite(value).all() for value in (source, truth, ea, scales)):
        raise ValueError("non-finite calibration input")
    if scales.shape != (2,) or np.any(scales <= 0) or mass_ratio <= 0 or penalty < 0:
        raise ValueError("invalid source scales, mass ratio, or penalty")


def fit_conditional_nested(
    source: np.ndarray,
    truth: np.ndarray,
    ea: np.ndarray,
    source_scales: Sequence[float],
    mass_ratio: float,
    penalty: float,
    *,
    extra: np.ndarray | None = None,
    extra_names: Sequence[str] = (),
) -> ConditionalEANestedExtension:
    """Fit Conditional EA plus declared extra slope effects.

    The EA column is byte-for-byte equivalent in construction to
    :func:`fit_conditional`.  Every coefficient uses the same ``SSE/n``
    normalization and prior vector ``[1, 0, 0, ...]``.
    """

    source = np.asarray(source, dtype=float)
    truth = np.asarray(truth, dtype=float)
    ea = np.asarray(ea, dtype=float)
    scales = np.asarray(source_scales, dtype=float)
    names = tuple(str(value) for value in extra_names)
    if extra is None:
        extra = np.zeros((len(source), 2, 0), dtype=float)
    extra = np.asarray(extra, dtype=float)
    _validate(source, truth, ea, scales, float(mass_ratio), float(penalty))
    if extra.shape != (len(source), 2, len(names)) or not np.isfinite(extra).all():
        raise ValueError("extra effects do not align with declared names")
    u = source / scales
    e = ea - ea.mean()
    ea_raw = u * e[:, None]
    ea_mean, ea_std = ea_raw.mean(axis=0), ea_raw.std(axis=0)
    ea_std = np.where(ea_std < 1e-8, 1.0, ea_std)
    ea_feature = (ea_raw - ea_mean) / ea_std
    extra_means = extra.mean(axis=0) if extra.shape[2] else np.empty((2, 0))
    extra_stds = extra.std(axis=0) if extra.shape[2] else np.empty((2, 0))
    extra_stds = np.where(extra_stds < 1e-8, 1.0, extra_stds)
    extra_raw = u[:, :, None] * ((extra - extra_means[None, :, :]) / extra_stds[None, :, :])
    design = np.concatenate([u[:, :, None], np.ones((len(u), 2, 1)), ea_feature[:, :, None], extra_raw], axis=2)
    coefficients = np.empty((2, 3 + len(names)), dtype=float)
    for target in range(2):
        matrix = design[:, target, :]
        labels = truth[:, target] / (float(mass_ratio) * scales[target])
        prior = np.zeros(matrix.shape[1], dtype=float)
        prior[0] = 1.0
        augmented = np.vstack([matrix / np.sqrt(len(source)), np.sqrt(penalty) * np.eye(matrix.shape[1])])
        augmented_labels = np.r_[labels / np.sqrt(len(source)), np.sqrt(penalty) * prior]
        coefficients[target] = np.linalg.lstsq(augmented, augmented_labels, rcond=None)[0]
    return ConditionalEANestedExtension(coefficients, float(ea.mean()), ea_mean, ea_std,
                                        extra_means, extra_stds, scales.copy(), float(mass_ratio),
                                        float(penalty), names)


def endpoint_magnitude(source: np.ndarray, source_scales: Sequence[float]) -> np.ndarray:
    """Return endpoint-specific ``log1p(max(source_j, 0)/scale_j)``."""

    source = np.asarray(source, dtype=float)
    scales = np.asarray(source_scales, dtype=float)
    if source.ndim != 2 or source.shape[1] != 2 or scales.shape != (2,) or np.any(scales <= 0):
        raise ValueError("source and two positive scales required")
    return np.log1p(np.maximum(source, 0.0) / scales[None, :])


def center_width_transform(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("values must have shape (n, 2)")
    return values.mean(axis=1, keepdims=True), (values[:, 1] - values[:, 0]).reshape(-1, 1)


def center_width_inverse(center: np.ndarray, width: np.ndarray) -> np.ndarray:
    center, width = np.asarray(center, dtype=float).reshape(-1), np.asarray(width, dtype=float).reshape(-1)
    if center.shape != width.shape or not np.isfinite(center).all() or not np.isfinite(width).all():
        raise ValueError("center and width are not aligned")
    return np.column_stack([center - width / 2.0, center + width / 2.0])


def project_physical(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    raw = np.asarray(values, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 2 or not np.isfinite(raw).all():
        raise ValueError("values must have shape (n, 2)")
    projected = np.column_stack([np.maximum(raw[:, 0], 0.0), np.maximum(raw[:, 1], np.maximum(raw[:, 0], 0.0))])
    return projected, {
        "raw_v2_lt_v1_rate": float(np.mean(raw[:, 1] < raw[:, 0])),
        "raw_v1_negative_rate": float(np.mean(raw[:, 0] < 0.0)),
        "projection_frequency": float(np.mean(np.any(np.abs(projected - raw) > 1e-12, axis=1))),
    }


__all__ = ["ConditionalEANestedExtension", "center_width_inverse", "center_width_transform",
           "endpoint_magnitude", "fit_conditional_nested", "fit_scalar_conditional", "project_physical"]
