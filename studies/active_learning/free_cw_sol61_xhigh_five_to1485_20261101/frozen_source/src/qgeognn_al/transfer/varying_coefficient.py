"""Small, deterministic varying-coefficient transfer heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _finite_2d(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 2D array")
    return array


@dataclass(frozen=True)
class VaryingCoefficientFit:
    """Fit ``mass_ratio * source_scale * (beta0*u + beta1 + u*effects)``."""

    coefficients: np.ndarray
    source_scales: np.ndarray
    mass_ratio: float
    effect_means: np.ndarray
    effect_stds: np.ndarray
    penalty: float
    effect_names: tuple[str, ...]

    def _design(self, source: np.ndarray, effects: np.ndarray) -> np.ndarray:
        source = _finite_2d(source, "source")
        effects = _finite_2d(effects, "effects")
        if source.shape[1] != self.coefficients.shape[0]:
            raise ValueError("source target count differs from fitted head")
        if len(source) != len(effects) or effects.shape[1] != len(self.effect_names):
            raise ValueError("effects are not aligned with source")
        u = source / self.source_scales
        centered = (effects - self.effect_means) / self.effect_stds
        return np.concatenate([u[:, :, None], np.ones((len(u), u.shape[1], 1)),
                               u[:, :, None] * centered[:, None, :]], axis=2)

    def predict(self, source: np.ndarray, effects: np.ndarray) -> np.ndarray:
        design = self._design(source, effects)
        normalized = np.einsum("ntk,tk->nt", design, self.coefficients)
        return normalized * self.mass_ratio * self.source_scales

    def audit(self) -> dict[str, object]:
        return {
            "coefficients": self.coefficients.tolist(),
            "source_scales": self.source_scales.tolist(),
            "mass_ratio": float(self.mass_ratio),
            "effect_means": self.effect_means.tolist(),
            "effect_stds": self.effect_stds.tolist(),
            "penalty": float(self.penalty),
            "effect_names": list(self.effect_names),
            "free_coefficients": int(self.coefficients.size),
        }


def fit_varying_coefficient(
    source: np.ndarray,
    truth: np.ndarray,
    effects: np.ndarray,
    source_scales: Sequence[float],
    mass_ratio: float,
    penalty: float,
    *,
    effect_names: Sequence[str],
) -> VaryingCoefficientFit:
    """Fit one base slope/intercept and one varying slope per declared effect."""

    source = _finite_2d(source, "source")
    truth = _finite_2d(truth, "truth")
    effects = _finite_2d(effects, "effects")
    scales = np.asarray(source_scales, dtype=float)
    names = tuple(str(name) for name in effect_names)
    if source.shape != truth.shape or len(source) != len(effects):
        raise ValueError("aligned source/truth/effects required")
    if source.shape[1] < 1 or effects.shape[1] != len(names) or len(source) < 4:
        raise ValueError("invalid varying-coefficient dimensions")
    if scales.shape != (source.shape[1],) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("source scales must be positive and aligned")
    if not np.isfinite(mass_ratio) or mass_ratio <= 0 or penalty < 0:
        raise ValueError("invalid mass ratio or penalty")
    means = effects.mean(axis=0)
    stds = np.where(effects.std(axis=0) < 1e-8, 1.0, effects.std(axis=0))
    u = source / scales
    centered = (effects - means) / stds
    design = np.concatenate([u[:, :, None], np.ones((len(u), u.shape[1], 1)),
                             u[:, :, None] * centered[:, None, :]], axis=2)
    coefficients = np.empty((source.shape[1], 2 + len(names)), dtype=float)
    for target in range(source.shape[1]):
        matrix = design[:, target, :]
        labels = truth[:, target] / (mass_ratio * scales[target])
        penalty_vector = np.ones(matrix.shape[1], dtype=float)
        penalty_vector[1] = 0.0
        augmented = np.vstack([matrix, np.sqrt(penalty) * np.diag(penalty_vector)])
        coefficients[target] = np.linalg.lstsq(augmented, np.r_[labels, np.zeros(matrix.shape[1])], rcond=None)[0]
    return VaryingCoefficientFit(coefficients, scales.copy(), float(mass_ratio), means, stds,
                                 float(penalty), names)


__all__ = ["VaryingCoefficientFit", "fit_varying_coefficient"]
