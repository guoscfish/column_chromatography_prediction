"""Shared metric definitions used by frozen E4 and post-hoc diagnostics."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def regression_metric_row(
    truth: np.ndarray,
    prediction: np.ndarray,
    target_scales: Mapping[str, float],
) -> dict[str, float]:
    """Return the frozen E4 V1/V2 metrics and combined NRMSE."""
    truth_array = np.asarray(truth, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    if truth_array.shape != prediction_array.shape or truth_array.ndim != 2 or truth_array.shape[1] != 2:
        raise ValueError("truth and prediction must have aligned shape (rows, 2)")
    row: dict[str, float] = {}
    for column, target in enumerate(("V1", "V2")):
        residual = truth_array[:, column] - prediction_array[:, column]
        denominator = np.square(
            truth_array[:, column] - truth_array[:, column].mean()
        ).sum()
        row[f"{target}_MAE"] = float(np.abs(residual).mean())
        row[f"{target}_RMSE"] = float(np.sqrt(np.square(residual).mean()))
        row[f"{target}_R2"] = (
            float(1 - np.square(residual).sum() / denominator)
            if denominator
            else float("nan")
        )
    row["NRMSE"] = 0.5 * (
        row["V1_RMSE"] / target_scales["V1"]
        + row["V2_RMSE"] / target_scales["V2"]
    )
    return row
