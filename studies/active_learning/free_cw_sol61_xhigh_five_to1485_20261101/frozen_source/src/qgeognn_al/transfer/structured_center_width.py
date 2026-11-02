"""Low-capacity, strictly nested extensions of center/width transfer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controlled_conditional_extension import (
    ConditionalEANestedExtension,
    center_width_inverse,
    center_width_transform,
    fit_conditional_nested,
    fit_scalar_conditional,
    project_physical,
)


METHODS = ("M3_CENTER_WIDTH", "CENTER_MAGNITUDE", "CENTER_CONDITIONED_WIDTH")


def _scale(values: np.ndarray) -> float:
    scale = float(np.std(np.asarray(values, dtype=float)))
    return 1.0 if scale < 1e-8 else scale


def _repeat_effect(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    return np.repeat(values[:, None, None], 2, axis=1)


@dataclass(frozen=True)
class StructuredCenterWidthFit:
    method: str
    center_fit: ConditionalEANestedExtension
    width_fit: ConditionalEANestedExtension
    center_scale: float
    width_scale: float
    zero_extra: bool = False

    @property
    def free_coefficients(self) -> int:
        return 6 if self.method == "M3_CENTER_WIDTH" else 7

    def predict_raw(self, source: np.ndarray, ea: np.ndarray) -> np.ndarray:
        center, width = center_width_transform(source)
        center_pair = np.repeat(center, 2, axis=1)
        width_pair = np.repeat(width, 2, axis=1)
        center_extra = None
        width_extra = None
        if self.method == "CENTER_MAGNITUDE":
            magnitude = np.zeros(len(center)) if self.zero_extra else np.log1p(
                np.maximum(center[:, 0], 0.0) / self.center_scale
            )
            center_extra = _repeat_effect(magnitude)
        elif self.method == "CENTER_CONDITIONED_WIDTH":
            center_effect = np.zeros(len(center)) if self.zero_extra else center[:, 0]
            width_extra = _repeat_effect(center_effect)
        center_prediction = self.center_fit.predict(center_pair, ea, center_extra)[:, 0]
        width_prediction = self.width_fit.predict(width_pair, ea, width_extra)[:, 0]
        return center_width_inverse(center_prediction, width_prediction)

    def predict(self, source: np.ndarray, ea: np.ndarray) -> np.ndarray:
        return project_physical(self.predict_raw(source, ea))[0]

    def audit(self, source: np.ndarray, ea: np.ndarray) -> dict[str, object]:
        raw = self.predict_raw(source, ea)
        _, projection = project_physical(raw)
        extra_coefficient = None
        if self.method == "CENTER_MAGNITUDE":
            extra_coefficient = float(self.center_fit.coefficients[0, 3])
        elif self.method == "CENTER_CONDITIONED_WIDTH":
            extra_coefficient = float(self.width_fit.coefficients[0, 3])
        return {
            "method": self.method,
            "free_coefficients": self.free_coefficients,
            "center_scale": self.center_scale,
            "width_scale": self.width_scale,
            "extra_coefficient": extra_coefficient,
            **projection,
        }


def fit_structured_center_width(
    source: np.ndarray,
    truth: np.ndarray,
    ea: np.ndarray,
    mass_ratio: float,
    penalty: float,
    method: str,
    *,
    zero_extra: bool = False,
) -> StructuredCenterWidthFit:
    """Fit M3 or one of its declared one-coefficient nested extensions."""

    if method not in METHODS:
        raise ValueError(f"unknown structured center/width method: {method}")
    source = np.asarray(source, dtype=float)
    truth = np.asarray(truth, dtype=float)
    ea = np.asarray(ea, dtype=float)
    center_source, width_source = center_width_transform(source)
    center_truth, width_truth = center_width_transform(truth)
    center_scale, width_scale = _scale(center_source), _scale(width_source)
    if method == "CENTER_MAGNITUDE":
        magnitude = np.zeros(len(source)) if zero_extra else np.log1p(
            np.maximum(center_source[:, 0], 0.0) / center_scale
        )
        center_fit = fit_conditional_nested(
            np.repeat(center_source, 2, axis=1), np.repeat(center_truth, 2, axis=1), ea,
            [center_scale, center_scale], mass_ratio, penalty,
            extra=_repeat_effect(magnitude), extra_names=("center_magnitude",),
        )
    else:
        center_fit = fit_scalar_conditional(
            center_source, center_truth, ea, center_scale, mass_ratio, penalty
        )
    if method == "CENTER_CONDITIONED_WIDTH":
        center_effect = np.zeros(len(source)) if zero_extra else center_source[:, 0]
        width_fit = fit_conditional_nested(
            np.repeat(width_source, 2, axis=1), np.repeat(width_truth, 2, axis=1), ea,
            [width_scale, width_scale], mass_ratio, penalty,
            extra=_repeat_effect(center_effect), extra_names=("source_center",),
        )
    else:
        width_fit = fit_scalar_conditional(
            width_source, width_truth, ea, width_scale, mass_ratio, penalty
        )
    return StructuredCenterWidthFit(
        method, center_fit, width_fit, center_scale, width_scale, zero_extra
    )


__all__ = ["METHODS", "StructuredCenterWidthFit", "fit_structured_center_width"]
