"""Low-capacity center/width transfer reparameterization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .varying_coefficient import VaryingCoefficientFit, fit_varying_coefficient


@dataclass(frozen=True)
class CenterWidthFit:
    center: VaryingCoefficientFit
    width: VaryingCoefficientFit

    def predict_raw(self, source: np.ndarray, ea: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source = np.asarray(source, dtype=float)
        if source.ndim != 2 or source.shape[1] != 2:
            raise ValueError("center/width source must have shape (n, 2)")
        effects = np.asarray(ea, dtype=float).reshape(-1, 1)
        center = self.center.predict(source.mean(axis=1, keepdims=True), effects)[:, 0]
        width = self.width.predict((source[:, 1] - source[:, 0]).reshape(-1, 1), effects)[:, 0]
        return np.column_stack([center - width / 2.0, center + width / 2.0]), center, width

    def predict(self, source: np.ndarray, ea: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        raw, _, _ = self.predict_raw(source, ea)
        projected_v1 = np.maximum(raw[:, 0], 0.0)
        projected_v2 = np.maximum(raw[:, 1], projected_v1)
        prediction = np.column_stack([projected_v1, projected_v2])
        violation = (raw[:, 1] < raw[:, 0]) | (raw[:, 0] < 0.0)
        projection = np.any(np.abs(prediction - raw) > 1e-12, axis=1)
        return prediction, {
            "v2_lt_v1_violation_rate": float(np.mean(raw[:, 1] < raw[:, 0])),
            "v1_negative_rate": float(np.mean(raw[:, 0] < 0.0)),
            "physical_violation_rate": float(np.mean(violation)),
            "projection_frequency": float(np.mean(projection)),
        }


def fit_center_width(
    source: np.ndarray,
    truth: np.ndarray,
    ea: np.ndarray,
    mass_ratio: float,
    penalty: float,
) -> CenterWidthFit:
    """Fit EA-varying slopes for center and linear width."""

    source = np.asarray(source, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if source.shape != truth.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("aligned two-target source/truth required")
    ea = np.asarray(ea, dtype=float)
    if ea.shape != (len(source),):
        raise ValueError("EA must align with source rows")
    center_source = source.mean(axis=1, keepdims=True)
    width_source = (source[:, 1] - source[:, 0]).reshape(-1, 1)
    center_truth = truth.mean(axis=1, keepdims=True)
    width_truth = (truth[:, 1] - truth[:, 0]).reshape(-1, 1)
    # Normalize by source-side variation.  This keeps the transformation
    # identifiable from source predictions and does not use target scale as a
    # hidden normalization constant.
    center_scale = np.asarray([float(np.std(center_source))], dtype=float)
    width_scale = np.asarray([float(np.std(width_source))], dtype=float)
    if np.any(center_scale <= 0) or np.any(width_scale <= 0):
        raise ValueError("source center/width has no variation")
    effects = ea.reshape(-1, 1)
    center_fit = fit_varying_coefficient(center_source, center_truth, effects, center_scale, mass_ratio,
                                         penalty, effect_names=("EA_fraction",))
    width_fit = fit_varying_coefficient(width_source, width_truth, effects, width_scale, mass_ratio,
                                        penalty, effect_names=("EA_fraction",))
    return CenterWidthFit(center_fit, width_fit)


__all__ = ["CenterWidthFit", "fit_center_width"]
