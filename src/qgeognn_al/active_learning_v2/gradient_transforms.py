"""Fixed, L0-only gradient geometries for the innovation screen."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureTransformResult:
    features: np.ndarray
    audit: dict[str, object]


def power_normalize_gradients(features: np.ndarray, alpha: float) -> FeatureTransformResult:
    """Return ``g / ||g||^(1-alpha)`` while retaining audited zero rows."""

    if float(alpha) not in (0.0, 0.5, 1.0):
        raise ValueError("innovation screen permits only alpha in {0, 0.5, 1}")
    values = np.asarray(features)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("gradient features must be a finite matrix")
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    zero = norms == 0
    denominator = np.ones_like(norms)
    denominator[~zero] = norms[~zero] ** (1.0 - float(alpha))
    transformed = values.astype(np.float64) / denominator[:, None]
    if not np.isfinite(transformed).all():
        raise RuntimeError("power-normalized gradients are non-finite")
    return FeatureTransformResult(
        transformed,
        {
            "definition": "g / ||g||^(1-alpha)",
            "alpha": float(alpha),
            "zero_norm_rows": int(zero.sum()),
            "zero_norm_indices": np.flatnonzero(zero).astype(int).tolist(),
            "zero_norm_policy": "retain_zero_vector",
            "input_norm_min": float(norms.min()) if len(norms) else None,
            "input_norm_max": float(norms.max()) if len(norms) else None,
        },
    )


def raw_output_transform(scales: tuple[float, float]) -> tuple[np.ndarray, dict[str, object]]:
    scale = _validated_scales(scales)
    transform = np.diag(1.0 / scale)
    return transform, {
        "definition": "diag(1/s_V1, 1/s_V2)",
        "target_scales": scale.tolist(),
        "transform_matrix": transform.tolist(),
        "target_scale_source": "L0_labels_only",
    }


def output_whitening_transform(
    l0_targets: np.ndarray,
    scales: tuple[float, float],
    *,
    absolute_eigenvalue_floor: float = 1e-8,
    relative_eigenvalue_floor: float = 1e-6,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build the fixed population-covariance whitening transform from L0."""

    targets = _validated_targets(l0_targets)
    scale = _validated_scales(scales)
    z = targets / scale[None, :]
    centered = z - z.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(absolute_eigenvalue_floor), float(relative_eigenvalue_floor) * float(eigenvalues.mean()))
    floored = np.maximum(eigenvalues, floor)
    inverse_sqrt = (eigenvectors * (1.0 / np.sqrt(floored))[None, :]) @ eigenvectors.T
    inverse_sqrt = 0.5 * (inverse_sqrt + inverse_sqrt.T)
    transform = inverse_sqrt @ np.diag(1.0 / scale)
    if not np.isfinite(transform).all() or not np.allclose(inverse_sqrt, inverse_sqrt.T, atol=1e-12):
        raise RuntimeError("invalid symmetric output-whitening transform")
    before_condition = float(np.inf if eigenvalues.min() <= 0 else eigenvalues.max() / eigenvalues.min())
    return transform, {
        "definition": "Sigma_z^(-1/2) @ diag(1/s_V1, 1/s_V2)",
        "target_scales": scale.tolist(),
        "population_covariance": covariance.tolist(),
        "eigenvalues_before_flooring": eigenvalues.tolist(),
        "eigenvalues_after_flooring": floored.tolist(),
        "absolute_eigenvalue_floor": float(absolute_eigenvalue_floor),
        "relative_eigenvalue_floor": float(relative_eigenvalue_floor),
        "applied_eigenvalue_floor": floor,
        "condition_number_before_flooring": before_condition,
        "condition_number_after_flooring": float(floored.max() / floored.min()),
        "inverse_sqrt_covariance": inverse_sqrt.tolist(),
        "transform_matrix": transform.tolist(),
        "target_source": "L0_labels_only",
        "covariance_ddof": 0,
    }


def center_width_transform(
    l0_targets: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build the L0-scaled center/width output coordinate transform."""

    targets = _validated_targets(l0_targets)
    center = 0.5 * (targets[:, 0] + targets[:, 1])
    width = targets[:, 1] - targets[:, 0]
    center_scale = float(center.std(ddof=0))
    width_scale = float(width.std(ddof=0))
    if center_scale <= 0 or width_scale <= 0:
        raise ValueError("L0 center and width population scales must be positive")
    transform = np.asarray(
        [[0.5 / center_scale, 0.5 / center_scale], [-1.0 / width_scale, 1.0 / width_scale]],
        dtype=np.float64,
    )
    return transform, {
        "definition": "[(V1+V2)/(2*s_C), (V2-V1)/s_W]",
        "center_population_std": center_scale,
        "width_population_std": width_scale,
        "transform_matrix": transform.tolist(),
        "target_source": "L0_labels_only",
        "scale_ddof": 0,
    }


def _validated_scales(scales: tuple[float, float]) -> np.ndarray:
    result = np.asarray(scales, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("two positive finite target scales required")
    return result


def _validated_targets(l0_targets: np.ndarray) -> np.ndarray:
    result = np.asarray(l0_targets, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2 or len(result) < 2 or not np.isfinite(result).all():
        raise ValueError("finite L0 targets with shape (N, 2), N >= 2 required")
    return result
