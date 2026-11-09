"""Budget-exact AULC and first-threshold interpolation, without extrapolation."""
import math
import numpy as np


def normalized_aulc(active_labels, errors, *, expected_budgets=None):
    """Return (raw trapezoidal area, area/observed span); optionally enforce a frozen grid."""
    labels, values = np.asarray(active_labels, np.float64), np.asarray(errors, np.float64)
    if labels.ndim != 1 or labels.shape != values.shape or len(labels) < 2 or not np.isfinite(labels).all() or not np.isfinite(values).all() or np.any(np.diff(labels) <= 0):
        raise ValueError('complete, finite, strictly increasing budget curve required')
    if expected_budgets is not None and not np.array_equal(labels, np.asarray(expected_budgets, np.float64)):
        raise ValueError('curve differs from the registered frozen budget grid')
    raw = float(np.trapezoid(values, labels))
    return raw, raw / float(labels[-1] - labels[0])


def labels_to_target(
    active_labels: np.ndarray,
    errors: np.ndarray,
    target: float,
) -> float | None:
    """First target crossing with adjacent-point interpolation; never extrapolate."""

    labels = np.asarray(active_labels, dtype=np.float64)
    values = np.asarray(errors, dtype=np.float64)
    if labels.ndim != 1 or values.shape != labels.shape or len(labels) == 0:
        raise ValueError("aligned nonempty one-dimensional curve required")
    if not np.all(np.diff(labels) > 0) or not np.isfinite(values).all() or not math.isfinite(float(target)):
        raise ValueError("labels must increase and errors/target must be finite")
    threshold = float(target)
    for index, value in enumerate(values):
        if value <= threshold:
            if index == 0:
                return float(labels[0])
            x0, x1 = labels[index - 1:index + 1]
            y0, y1 = values[index - 1:index + 1]
            if y0 <= threshold:
                return float(x0)
            if y1 == y0:
                return float(x1)
            fraction = (threshold - y0) / (y1 - y0)
            return float(x0 + fraction * (x1 - x0))
    return None
