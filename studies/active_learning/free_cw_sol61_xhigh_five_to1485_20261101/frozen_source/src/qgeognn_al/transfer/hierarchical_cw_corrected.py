"""Dimensionally consistent hierarchical Center/Width transfer models.

The historical implementation in :mod:`architecture_headroom` is deliberately
left unchanged.  This module separates two questions: a byte-level nested
legacy-objective extension with independent deviation penalties, and a
dimensionless endpoint-aligned formulation whose unit-slope prior is the
mass-ratio source identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .architecture_headroom import HierarchicalCenterWidthFit, fit_hierarchical_scalar
from .controlled_conditional_extension import center_width_inverse, center_width_transform, project_physical


def _safe_scalar_scale(value: float) -> float:
    return float(value) if abs(float(value)) >= 1e-10 else 1.0


def fit_hierarchical_cw_separate_lambda_legacy_objective(
    source: np.ndarray, truth: np.ndarray, ea: np.ndarray, labels: Sequence[str],
    columns: Sequence[str], *, mass_ratios: dict[str, float], base_penalty: float,
    lambda_center_deviation: float, lambda_width_deviation: float,
) -> HierarchicalCenterWidthFit:
    """Exact nested extension of historical HIER; only the two delta penalties differ."""
    labels_array = np.asarray(labels, dtype=str)
    names = tuple(str(value) for value in columns)
    if set(mass_ratios) != set(names) or set(labels_array) - set(names):
        raise ValueError("column labels and mass-ratio contract do not match")
    ratios = np.asarray([mass_ratios[value] for value in labels_array], dtype=float)
    if np.any(ratios <= 0):
        raise ValueError("mass ratios must be positive")
    source_center, source_width = center_width_transform(source)
    truth_center, truth_width = center_width_transform(truth)
    common = dict(column=labels_array, columns=names, base_penalty=float(base_penalty))
    center = fit_hierarchical_scalar(
        source_center[:, 0], truth_center[:, 0] / ratios, ea,
        delta_penalty=float(lambda_center_deviation), **common,
    )
    width = fit_hierarchical_scalar(
        source_width[:, 0], truth_width[:, 0] / ratios, ea,
        delta_penalty=float(lambda_width_deviation), **common,
    )
    return HierarchicalCenterWidthFit(
        names, {key: float(value) for key, value in mass_ratios.items()}, center, width,
    )


def fit_hierarchical_cw_shared_lambda_corrected(
    source: np.ndarray, truth: np.ndarray, ea: np.ndarray, labels: Sequence[str],
    columns: Sequence[str], *, mass_ratios: dict[str, float], base_penalty: float,
    deviation_lambda: float,
) -> HierarchicalCenterWidthFit:
    """Matched shared-lambda control under the unchanged historical C/W objective."""
    return fit_hierarchical_cw_separate_lambda_legacy_objective(
        source, truth, ea, labels, columns, mass_ratios=mass_ratios,
        base_penalty=base_penalty, lambda_center_deviation=deviation_lambda,
        lambda_width_deviation=deviation_lambda,
    )


@dataclass(frozen=True)
class CorrectedHierarchicalBasis:
    center_scale: float
    width_scale: float
    ea_mean: float
    center_interaction_mean: float
    center_interaction_scale: float
    width_interaction_mean: float
    width_interaction_scale: float

    @classmethod
    def fit(cls, source: np.ndarray, ea: np.ndarray) -> "CorrectedHierarchicalBasis":
        center, width = center_width_transform(source)
        c, w = center[:, 0], width[:, 0]
        ea_array = np.asarray(ea, dtype=float).reshape(-1)
        cs, ws = _safe_scalar_scale(c.std(ddof=0)), _safe_scalar_scale(w.std(ddof=0))
        em = float(ea_array.mean())
        ci, wi = (c / cs) * (ea_array - em), (w / ws) * (ea_array - em)
        return cls(cs, ws, em, float(ci.mean()), _safe_scalar_scale(ci.std(ddof=0)),
                   float(wi.mean()), _safe_scalar_scale(wi.std(ddof=0)))

    def transform(self, source: np.ndarray, ea: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center, width = center_width_transform(source)
        c, w = center[:, 0] / self.center_scale, width[:, 0] / self.width_scale
        e = np.asarray(ea, dtype=float).reshape(-1) - self.ea_mean
        center_design = np.column_stack([
            c, np.ones(len(c)), (c * e - self.center_interaction_mean) / self.center_interaction_scale,
        ])
        width_design = np.column_stack([
            w, np.ones(len(w)), (w * e - self.width_interaction_mean) / self.width_interaction_scale,
        ])
        return center_design, width_design


def _expanded(design: np.ndarray, labels: np.ndarray, columns: tuple[str, ...], latent: np.ndarray | None) -> np.ndarray:
    latent_dimension = 0 if latent is None else latent.shape[1]
    result = np.zeros((len(design), 3 + 3 * len(columns) + latent_dimension), dtype=float)
    result[:, :3] = design
    for index, name in enumerate(columns):
        selected = labels == name
        result[selected, 3 + 3 * index:6 + 3 * index] = design[selected]
    if latent_dimension:
        result[:, 3 + 3 * len(columns):] = latent
    return result


def _endpoint_solve(
    center_design: np.ndarray, width_design: np.ndarray, truth: np.ndarray,
    ratios: np.ndarray, endpoint_scale: np.ndarray, center_source_scale: float,
    width_source_scale: float, center_penalty: np.ndarray, width_penalty: np.ndarray,
    row_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve endpoint-normalized loss with dimensionless C/W coefficients."""
    truth = np.asarray(truth, dtype=float)
    scales = np.asarray(endpoint_scale, dtype=float)
    if truth.shape != (len(center_design), 2) or scales.shape != (2,) or np.any(scales <= 0):
        raise ValueError("aligned truth and two positive endpoint scales required")
    n, pc = len(truth), center_design.shape[1]
    c = center_design * (ratios * center_source_scale)[:, None]
    w = width_design * (ratios * width_source_scale)[:, None]
    v1 = np.column_stack([c, -0.5 * w]) / scales[0]
    v2 = np.column_stack([c, +0.5 * w]) / scales[1]
    if row_weights is None:
        weights = np.ones((n, 2), dtype=float)
    else:
        weights = np.asarray(row_weights, dtype=float)
        if weights.shape != (n, 2) or np.any(weights <= 0) or not np.isfinite(weights).all():
            raise ValueError("row_weights must be finite positive n x 2")
    root_weights = np.sqrt(weights)
    design = np.vstack([v1 * root_weights[:, 0, None], v2 * root_weights[:, 1, None]]) / np.sqrt(n)
    target = np.r_[truth[:, 0] / scales[0] * root_weights[:, 0],
                   truth[:, 1] / scales[1] * root_weights[:, 1]] / np.sqrt(n)
    penalty = np.r_[center_penalty, width_penalty]
    prior = np.zeros(center_design.shape[1] + width_design.shape[1], dtype=float)
    prior[0], prior[pc] = 1.0, 1.0
    augmented = np.vstack([design, np.diag(np.sqrt(penalty))])
    coefficient = np.linalg.lstsq(
        augmented, np.r_[target, np.sqrt(penalty) * prior], rcond=None,
    )[0]
    return coefficient[:pc], coefficient[pc:]


@dataclass(frozen=True)
class EndpointAlignedCorrectedFit:
    columns: tuple[str, ...]
    mass_ratios: dict[str, float]
    basis: CorrectedHierarchicalBasis
    center_coefficients: np.ndarray
    width_coefficients: np.ndarray
    endpoint_scales: np.ndarray
    lambda_center_deviation: float
    lambda_width_deviation: float
    latent_mean: np.ndarray | None = None
    latent_scale: np.ndarray | None = None
    lambda_center_latent: float | None = None
    lambda_width_latent: float | None = None

    def predict_raw(self, source: np.ndarray, ea: np.ndarray, labels: Sequence[str], latent: np.ndarray | None = None) -> np.ndarray:
        labels_array = np.asarray(labels, dtype=str).reshape(-1)
        center, width = self.basis.transform(source, ea)
        standardized = None
        if self.latent_mean is not None:
            if latent is None:
                raise ValueError("latent representation required")
            standardized = (np.asarray(latent, dtype=float) - self.latent_mean) / self.latent_scale
        center = _expanded(center, labels_array, self.columns, standardized)
        width = _expanded(width, labels_array, self.columns, standardized)
        ratio = np.asarray([self.mass_ratios[value] for value in labels_array], dtype=float)
        predicted_center = (center @ self.center_coefficients) * ratio * self.basis.center_scale
        predicted_width = (width @ self.width_coefficients) * ratio * self.basis.width_scale
        return center_width_inverse(predicted_center, predicted_width)

    def predict(self, source: np.ndarray, ea: np.ndarray, labels: Sequence[str], latent: np.ndarray | None = None) -> np.ndarray:
        return project_physical(self.predict_raw(source, ea, labels, latent))[0]


def _fit_endpoint(
    source: np.ndarray, truth: np.ndarray, ea: np.ndarray, latent: np.ndarray | None,
    labels: Sequence[str], columns: Sequence[str], mass_ratios: dict[str, float], *,
    base_penalty: float, lambda_center_deviation: float, lambda_width_deviation: float,
    lambda_center_latent: float | None, lambda_width_latent: float | None,
    endpoint_scale: np.ndarray | None, row_weights: np.ndarray | None = None,
) -> EndpointAlignedCorrectedFit:
    source, truth = np.asarray(source, dtype=float), np.asarray(truth, dtype=float)
    labels_array, names = np.asarray(labels, dtype=str), tuple(str(value) for value in columns)
    if set(labels_array) - set(names) or set(mass_ratios) != set(names):
        raise ValueError("column labels and mass-ratio contract do not match")
    basis = CorrectedHierarchicalBasis.fit(source, ea)
    center, width = basis.transform(source, ea)
    latent_mean = latent_scale = standardized = None
    if (lambda_center_latent is None) != (lambda_width_latent is None):
        raise ValueError("both latent penalties must be present or absent")
    if lambda_center_latent is not None:
        latent_array = np.asarray(latent, dtype=float)
        if latent_array.ndim != 2 or len(latent_array) != len(source):
            raise ValueError("aligned two-dimensional latent representation required")
        latent_mean = latent_array.mean(axis=0)
        latent_scale = np.where(latent_array.std(axis=0) < 1e-10, 1.0, latent_array.std(axis=0))
        standardized = (latent_array - latent_mean) / latent_scale
    center = _expanded(center, labels_array, names, standardized)
    width = _expanded(width, labels_array, names, standardized)
    ratio = np.asarray([mass_ratios[value] for value in labels_array], dtype=float)
    scales = np.asarray(endpoint_scale if endpoint_scale is not None else truth.std(axis=0), dtype=float)
    scales = np.where(np.abs(scales) < 1e-10, 1.0, scales)
    center_penalty = np.r_[np.repeat(base_penalty, 3), np.repeat(lambda_center_deviation, 3 * len(names))]
    width_penalty = np.r_[np.repeat(base_penalty, 3), np.repeat(lambda_width_deviation, 3 * len(names))]
    if standardized is not None:
        center_penalty = np.r_[center_penalty, np.repeat(lambda_center_latent, standardized.shape[1])]
        width_penalty = np.r_[width_penalty, np.repeat(lambda_width_latent, standardized.shape[1])]
    center_coefficients, width_coefficients = _endpoint_solve(
        center, width, truth, ratio, scales, basis.center_scale, basis.width_scale,
        center_penalty, width_penalty, row_weights,
    )
    return EndpointAlignedCorrectedFit(
        names, {key: float(value) for key, value in mass_ratios.items()}, basis,
        center_coefficients, width_coefficients, scales.copy(),
        float(lambda_center_deviation), float(lambda_width_deviation),
        latent_mean, latent_scale,
        None if lambda_center_latent is None else float(lambda_center_latent),
        None if lambda_width_latent is None else float(lambda_width_latent),
    )


def fit_hierarchical_cw_endpoint_aligned_corrected(
    source: np.ndarray, truth: np.ndarray, ea: np.ndarray, labels: Sequence[str],
    columns: Sequence[str], *, mass_ratios: dict[str, float], base_penalty: float,
    lambda_center_deviation: float, lambda_width_deviation: float,
    endpoint_scale: np.ndarray | None = None, row_weights: np.ndarray | None = None,
) -> EndpointAlignedCorrectedFit:
    return _fit_endpoint(
        source, truth, ea, None, labels, columns, mass_ratios,
        base_penalty=base_penalty,
        lambda_center_deviation=lambda_center_deviation,
        lambda_width_deviation=lambda_width_deviation,
        lambda_center_latent=None, lambda_width_latent=None,
        endpoint_scale=endpoint_scale, row_weights=row_weights,
    )


def fit_corrected_joint_full128(
    source: np.ndarray, truth: np.ndarray, ea: np.ndarray, latent: np.ndarray,
    labels: Sequence[str], columns: Sequence[str], *, mass_ratios: dict[str, float],
    base_penalty: float, lambda_center_deviation: float, lambda_width_deviation: float,
    lambda_center_latent: float | None, lambda_width_latent: float | None,
    endpoint_scale: np.ndarray | None = None, row_weights: np.ndarray | None = None,
) -> EndpointAlignedCorrectedFit:
    """FULL128 joint model; ``None`` latent penalties force exact HIER nesting."""
    return _fit_endpoint(
        source, truth, ea, latent, labels, columns, mass_ratios,
        base_penalty=base_penalty,
        lambda_center_deviation=lambda_center_deviation,
        lambda_width_deviation=lambda_width_deviation,
        lambda_center_latent=lambda_center_latent,
        lambda_width_latent=lambda_width_latent,
        endpoint_scale=endpoint_scale, row_weights=row_weights,
    )


__all__ = [
    "CorrectedHierarchicalBasis", "EndpointAlignedCorrectedFit",
    "fit_corrected_joint_full128", "fit_hierarchical_cw_endpoint_aligned_corrected",
    "fit_hierarchical_cw_separate_lambda_legacy_objective",
    "fit_hierarchical_cw_shared_lambda_corrected",
]
