"""Low-capacity shared center/width transfer with fixed column context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .controlled_conditional_extension import (
    ConditionalEANestedExtension,
    center_width_inverse,
    center_width_transform,
    fit_conditional_nested,
    project_physical,
)


MASS_FLOW_CONTEXT = ("packing_mass_g", "flow_ml_min")
ALL_CONTEXT = MASS_FLOW_CONTEXT + (
    "legacy_column_dia", "legacy_column_len", "legacy_column_den",
)


def _repeat_context(context: np.ndarray) -> np.ndarray:
    context = np.asarray(context, dtype=float)
    if context.ndim != 2 or not np.isfinite(context).all():
        raise ValueError("finite two-dimensional context required")
    return np.repeat(context[:, None, :], 2, axis=1)


def _scale(values: np.ndarray) -> float:
    value = float(np.std(np.asarray(values, dtype=float)))
    return 1.0 if value < 1e-8 else value


@dataclass(frozen=True)
class SharedCenterWidthFit:
    """A shared linear varying-slope fit across target columns."""

    context_names: tuple[str, ...]
    center_fit: ConditionalEANestedExtension
    width_fit: ConditionalEANestedExtension
    center_scale: float
    width_scale: float

    @property
    def free_coefficients(self) -> int:
        return 2 * (3 + len(self.context_names))

    def predict_raw(self, source: np.ndarray, ea: np.ndarray, context: np.ndarray) -> np.ndarray:
        source = np.asarray(source, dtype=float)
        ea = np.asarray(ea, dtype=float)
        context = np.asarray(context, dtype=float)
        if context.shape != (len(source), len(self.context_names)):
            raise ValueError("context does not match fitted contract")
        center, width = center_width_transform(source)
        extra = _repeat_context(context)
        center_prediction = self.center_fit.predict(
            np.repeat(center, 2, axis=1), ea, extra
        )[:, 0]
        width_prediction = self.width_fit.predict(
            np.repeat(width, 2, axis=1), ea, extra
        )[:, 0]
        return center_width_inverse(center_prediction, width_prediction)

    def predict(self, source: np.ndarray, ea: np.ndarray, context: np.ndarray) -> np.ndarray:
        return project_physical(self.predict_raw(source, ea, context))[0]

    def audit(self, source: np.ndarray, ea: np.ndarray, context: np.ndarray) -> dict[str, object]:
        raw = self.predict_raw(source, ea, context)
        _, projection = project_physical(raw)
        return {
            "context_names": list(self.context_names),
            "free_coefficients": self.free_coefficients,
            "center_scale": self.center_scale,
            "width_scale": self.width_scale,
            "center_fit": self.center_fit.audit(),
            "width_fit": self.width_fit.audit(),
            "normalization_fit_rows_only": True,
            **projection,
        }


def fit_shared_center_width(
    source: np.ndarray,
    truth: np.ndarray,
    ea: np.ndarray,
    context: np.ndarray,
    context_names: Sequence[str],
    penalty: float,
) -> SharedCenterWidthFit:
    """Fit one shared Center/Width model with standardized column context.

    The direct predictive formula is ``C_t = [a0 + aEA*EA + aZ'Z] C_s + bC``
    and likewise for width.  No mass-ratio multiplier is implicit: packing mass
    is present only when it is explicitly included in ``context_names``.
    """

    source = np.asarray(source, dtype=float)
    truth = np.asarray(truth, dtype=float)
    ea = np.asarray(ea, dtype=float)
    context = np.asarray(context, dtype=float)
    names = tuple(str(value) for value in context_names)
    if names not in (MASS_FLOW_CONTEXT, ALL_CONTEXT):
        raise ValueError("only the preregistered two- or five-variable context is allowed")
    if source.shape != truth.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("aligned source/truth arrays with shape (n,2) required")
    if ea.shape != (len(source),) or context.shape != (len(source), len(names)):
        raise ValueError("EA/context rows are not aligned")
    center_source, width_source = center_width_transform(source)
    center_truth, width_truth = center_width_transform(truth)
    center_scale, width_scale = _scale(center_source), _scale(width_source)
    extra = _repeat_context(context)
    center_fit = fit_conditional_nested(
        np.repeat(center_source, 2, axis=1), np.repeat(center_truth, 2, axis=1), ea,
        [center_scale, center_scale], 1.0, penalty, extra=extra, extra_names=names,
    )
    width_fit = fit_conditional_nested(
        np.repeat(width_source, 2, axis=1), np.repeat(width_truth, 2, axis=1), ea,
        [width_scale, width_scale], 1.0, penalty, extra=extra, extra_names=names,
    )
    return SharedCenterWidthFit(names, center_fit, width_fit, center_scale, width_scale)


__all__ = [
    "ALL_CONTEXT", "MASS_FLOW_CONTEXT", "SharedCenterWidthFit", "fit_shared_center_width",
]
