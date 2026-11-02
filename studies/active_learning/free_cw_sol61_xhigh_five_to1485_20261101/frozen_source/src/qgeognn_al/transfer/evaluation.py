"""Evaluation primitives for matched cross-column transfer studies.

The functions in this module deliberately keep fitting and test evaluation
separate.  In particular, :func:`tail_error_metrics` receives the training
labels only to derive the volume cut points; it never searches thresholds on
the evaluation population.  Prediction arrays may contain either the current
two-column point output ``(n, 2)`` or the six quantiles used by QGeoGNN-V2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


TARGETS = ("V1", "V2")
STRATA = ("low", "mid", "high_tail")


def _as_truth(values: np.ndarray, name: str = "truth") -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def point_prediction(values: np.ndarray, name: str = "prediction") -> np.ndarray:
    """Extract V1/V2 point predictions from a two- or six-column array.

    QGeoGNN-V2 stores ``q10, q50, q90`` for each target.  The median is the
    point prediction used by every absolute-error metric.  A two-column array
    is already a point prediction and is returned unchanged (as a float copy).
    """

    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] not in (2, 6):
        raise ValueError(f"{name} must have shape (n, 2) or (n, 6)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    if array.shape[1] == 2:
        return array
    return array[:, (1, 4)]


def _scales(scales: Mapping[str, float] | Sequence[float] | None) -> np.ndarray | None:
    if scales is None:
        return None
    if isinstance(scales, Mapping):
        result = np.asarray([scales[target] for target in TARGETS], dtype=float)
    else:
        result = np.asarray(scales, dtype=float)
    if result.shape != (2,) or not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("scales must contain two positive finite values")
    return result


def absolute_error_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    scales: Mapping[str, float] | Sequence[float] | None = None,
) -> dict[str, float]:
    """Return primary absolute-error metrics and project-compatible NRMSE.

    ``combined_normalized_rmse`` is the RMS of the two source-normalized
    RMSEs, matching the checkpoint-selection score used by current transfer
    studies.  ``nrmse``/``combined_nrmse`` are the arithmetic mean retained by
    older AULC reports.  Keeping both names prevents accidental mixing of the
    two normalizations in summary code.
    """

    y_true = _as_truth(truth)
    y_pred = point_prediction(prediction)
    if len(y_true) != len(y_pred):
        raise ValueError("truth and prediction row counts differ")
    result: dict[str, float] = {}
    rmse_values = []
    for index, target in enumerate(TARGETS):
        residual = y_true[:, index] - y_pred[:, index]
        denominator = float(np.square(y_true[:, index] - y_true[:, index].mean()).sum())
        rmse = float(np.sqrt(np.square(residual).mean()))
        mae = float(np.abs(residual).mean())
        result[f"{target}_rmse"] = rmse
        result[f"{target}_mae"] = mae
        result[f"{target}_r2"] = float(1.0 - np.square(residual).sum() / denominator) if denominator else float("nan")
        rmse_values.append(rmse)
    normalized = _scales(scales)
    if normalized is not None:
        ratios = np.asarray(rmse_values) / normalized
        result["combined_normalized_rmse"] = float(np.sqrt(np.mean(np.square(ratios))))
        result["normalized_rmse"] = result["combined_normalized_rmse"]
        result["nrmse"] = float(np.mean(ratios))
        result["combined_nrmse"] = result["nrmse"]
    result["all_outputs_finite"] = bool(np.isfinite(y_pred).all())
    return result


# A short alias used by a few current study scripts.
regression_metrics = absolute_error_metrics


def metrics_from_arrays(
    truth: np.ndarray,
    prediction: np.ndarray,
    scales: Mapping[str, float] | Sequence[float] | None = None,
) -> dict[str, float]:
    """Return point metrics plus quantile coverage diagnostics for V2 output."""

    y_true = _as_truth(truth)
    raw = np.asarray(prediction, dtype=float)
    if raw.ndim != 2 or raw.shape[1] not in (2, 6) or len(raw) != len(y_true):
        raise ValueError("prediction must align with truth and have 2 or 6 columns")
    result = absolute_error_metrics(y_true, raw, scales)
    if raw.shape[1] == 6:
        for index, target in enumerate(TARGETS):
            quantiles = raw[:, index * 3 : index * 3 + 3]
            residuals = y_true[:, [index]] - quantiles
            levels = np.asarray([0.1, 0.5, 0.9], dtype=float).reshape(1, -1)
            result[f"{target}_mean_pinball_loss"] = float(np.maximum(levels * residuals, (levels - 1.0) * residuals).mean())
            result[f"{target}_interval_80_coverage"] = float(np.mean((y_true[:, index] >= quantiles[:, 0]) & (y_true[:, index] <= quantiles[:, 2])))
            result[f"{target}_interval_80_mean_width"] = float(np.mean(quantiles[:, 2] - quantiles[:, 0]))
        result["quantile_crossing_rate"] = float(np.mean((raw[:, 0] > raw[:, 1]) | (raw[:, 1] > raw[:, 2]) | (raw[:, 3] > raw[:, 4]) | (raw[:, 4] > raw[:, 5])))
    return result


def validation_scores(metrics: Mapping[str, float], target_variance: Mapping[str, float]) -> tuple[float, float]:
    """Return normalized and legacy validation scores used by old ledgers."""

    if any(float(target_variance[target]) <= 0 for target in TARGETS):
        raise ValueError("target variances must be positive")
    normalized = sum(float(metrics[f"{target}_rmse"]) ** 2 / float(target_variance[target]) for target in TARGETS)
    legacy = float(metrics["V1_rmse"]) ** 2 + 0.5 * float(metrics["V2_rmse"]) ** 2
    return float(normalized), legacy


def volume_strata(
    train_truth: np.ndarray,
    evaluation_truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign low/mid/high-tail labels using training-derived q50 and q80.

    The returned tuple is ``(labels, q50, q80)``.  ``labels`` has shape
    ``(n_evaluation, 2)`` and contains the stable values in :data:`STRATA`.
    No evaluation labels are used to calculate either cut point.
    """

    train = _as_truth(train_truth, "train_truth")
    evaluation = _as_truth(evaluation_truth, "evaluation_truth")
    q50 = np.quantile(train, 0.50, axis=0)
    q80 = np.quantile(train, 0.80, axis=0)
    labels = np.empty(evaluation.shape, dtype=object)
    for index in range(2):
        labels[:, index] = np.where(
            evaluation[:, index] <= q50[index],
            "low",
            np.where(evaluation[:, index] <= q80[index], "mid", "high_tail"),
        )
    return labels, q50.astype(float), q80.astype(float)


def tail_error_metrics(
    train_truth: np.ndarray,
    evaluation_truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    """Compute RMSE/MAE/count and high-tail SSE share by target stratum."""

    train = _as_truth(train_truth, "train_truth")
    truth = _as_truth(evaluation_truth, "evaluation_truth")
    pred = point_prediction(prediction)
    if len(truth) != len(pred):
        raise ValueError("evaluation truth and prediction row counts differ")
    labels, q50, q80 = volume_strata(train, truth)
    result: dict[str, float] = {
        "V1_train_q50": float(q50[0]),
        "V2_train_q50": float(q50[1]),
        "V1_train_q80": float(q80[0]),
        "V2_train_q80": float(q80[1]),
    }
    for index, target in enumerate(TARGETS):
        residual = truth[:, index] - pred[:, index]
        squared = np.square(residual)
        total_sse = float(squared.sum())
        for stratum in STRATA:
            mask = labels[:, index] == stratum
            count = int(mask.sum())
            prefix = f"{target}_{stratum}"
            result[f"{prefix}_n"] = count
            result[f"{prefix}_rmse"] = float(np.sqrt(squared[mask].mean())) if count else float("nan")
            result[f"{prefix}_mae"] = float(np.abs(residual[mask]).mean()) if count else float("nan")
            if stratum == "high_tail":
                result[f"{target}_high_tail_squared_error_fraction"] = (
                    float(squared[mask].sum() / total_sse) if total_sse else float("nan")
                )
                # Compact aliases make the primary tail table easy to build.
                result[f"{target}_tail_rmse"] = result[f"{prefix}_rmse"]
                result[f"{target}_tail_mae"] = result[f"{prefix}_mae"]
                result[f"{target}_tail_n"] = count
                result[f"{target}_tail_sse_fraction"] = result[f"{target}_high_tail_squared_error_fraction"]
    return result


def tail_error_rows(
    train_truth: np.ndarray,
    evaluation_truth: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    *,
    column: str | None = None,
    protocol: str | None = None,
    seed: int | None = None,
    budget: int | None = None,
) -> list[dict[str, object]]:
    """Return one long-form row per method/target/stratum."""

    train = _as_truth(train_truth, "train_truth")
    truth = _as_truth(evaluation_truth, "evaluation_truth")
    labels, q50, q80 = volume_strata(train, truth)
    rows: list[dict[str, object]] = []
    for method, prediction in predictions.items():
        point = point_prediction(prediction)
        if len(point) != len(truth):
            raise ValueError(f"prediction row count differs for method {method}")
        for index, target in enumerate(TARGETS):
            residual = truth[:, index] - point[:, index]
            squared = np.square(residual)
            total_sse = float(squared.sum())
            for stratum in STRATA:
                mask = labels[:, index] == stratum
                n = int(mask.sum())
                row: dict[str, object] = {
                    "column": column,
                    "protocol": protocol,
                    "seed": seed,
                    "budget": budget,
                    "method": method,
                    "target": target,
                    "stratum": stratum,
                    "n": n,
                    "rmse": float(np.sqrt(squared[mask].mean())) if n else float("nan"),
                    "mae": float(np.abs(residual[mask]).mean()) if n else float("nan"),
                    "squared_error_fraction": float(squared[mask].sum() / total_sse) if total_sse else float("nan"),
                    "train_q50": float(q50[index]),
                    "train_q80": float(q80[index]),
                }
                rows.append(row)
    return rows


def paired_comparison(
    truth: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    scales: Mapping[str, float] | Sequence[float] | None = None,
) -> dict[str, float]:
    """Compare two methods on exactly the same rows.

    Deltas are ``candidate - reference``; therefore a negative RMSE/MAE delta
    is an improvement.  The function does not select a method or inspect any
    labels outside the supplied evaluation rows.
    """

    y_true = _as_truth(truth)
    candidate_point = point_prediction(candidate)
    reference_point = point_prediction(reference)
    if candidate_point.shape != reference_point.shape or len(y_true) != len(candidate_point):
        raise ValueError("paired arrays are not aligned")
    candidate_metrics = absolute_error_metrics(y_true, candidate_point, scales)
    reference_metrics = absolute_error_metrics(y_true, reference_point, scales)
    result: dict[str, float] = {}
    for target in TARGETS:
        for metric in ("rmse", "mae"):
            metric_key = f"{target}_{metric}"
            result[f"{metric_key}_delta"] = candidate_metrics[metric_key] - reference_metrics[metric_key]
            result[f"{target}_{metric}_relative_gain"] = (
                float(1.0 - candidate_metrics[metric_key] / reference_metrics[metric_key])
                if reference_metrics[metric_key] else float("nan")
            )
    for key in ("combined_normalized_rmse", "nrmse"):
        if key in candidate_metrics and key in reference_metrics:
            result[f"{key}_delta"] = candidate_metrics[key] - reference_metrics[key]
            result[f"{key}_relative_gain"] = (
                float(1.0 - candidate_metrics[key] / reference_metrics[key])
                if reference_metrics[key] else float("nan")
            )
    result["wins_v1_rmse"] = float(candidate_metrics["V1_rmse"] < reference_metrics["V1_rmse"])
    result["wins_v2_rmse"] = float(candidate_metrics["V2_rmse"] < reference_metrics["V2_rmse"])
    return result


def compute_aulc(budgets: Sequence[float], scores: Sequence[float]) -> float:
    """Return the normalized trapezoidal area under a learning curve."""

    x = np.asarray(budgets, dtype=float)
    y = np.asarray(scores, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or len(x) < 2:
        raise ValueError("AULC needs aligned one-dimensional arrays with >=2 points")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x) <= 0):
        raise ValueError("AULC budgets must be finite and strictly increasing")
    span = float(x[-1] - x[0])
    return float(np.trapezoid(y, x) / span) if hasattr(np, "trapezoid") else float(np.trapz(y, x) / span)


@dataclass(frozen=True)
class LabelLedger:
    """Immutable role ledger used to guard matched target-label accounting."""

    gradient_train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    planned_budget: int | None = None

    def __post_init__(self) -> None:
        roles = [set(self.gradient_train), set(self.validation), set(self.test)]
        if any(not role for role in roles):
            raise ValueError("each label role must be non-empty")
        if roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2]:
            raise ValueError("label roles overlap")
        if self.planned_budget is not None and self.actual_budget != int(self.planned_budget):
            raise ValueError("planned budget does not equal train plus validation rows")

    @property
    def actual_budget(self) -> int:
        return len(self.gradient_train) + len(self.validation)

    def assert_fit_roles(self, ids: Sequence[str], *, allow_validation: bool = True) -> None:
        """Reject test IDs in a fit or checkpoint-selection input."""

        allowed = set(self.gradient_train) | (set(self.validation) if allow_validation else set())
        leaked = set(str(value) for value in ids) & set(self.test)
        if leaked:
            raise ValueError(f"test labels entered fit/selection: {sorted(leaked)[:3]}")
        unknown = set(str(value) for value in ids) - allowed
        if unknown:
            raise ValueError(f"fit IDs outside authorized labels: {sorted(unknown)[:3]}")

    def to_dict(self) -> dict[str, object]:
        return {
            "gradient_train": list(self.gradient_train),
            "validation": list(self.validation),
            "test": list(self.test),
            "planned_budget": self.planned_budget,
            "actual_budget": self.actual_budget,
            "test_rows_used_for_fit": 0,
            "test_rows_used_for_checkpoint_selection": 0,
        }
