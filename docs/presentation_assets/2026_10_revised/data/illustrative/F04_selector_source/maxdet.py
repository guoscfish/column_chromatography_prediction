"""Conditional D-optimal selection on fixed model feature rows."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Sequence

import numpy as np
from scipy.linalg import solve_triangular

from .uncertainty import ensemble_uncertainty_select


@dataclass(frozen=True)
class MaxDetResult:
    """A deterministic greedy batch and its label-free numerical audit."""

    selected_positions: np.ndarray
    selected_candidate_positions: np.ndarray
    normalization_scale: float
    initial_marginal_gains: np.ndarray
    trace: tuple[dict[str, float | int], ...]
    audit: dict[str, float | int | bool | str]


@dataclass(frozen=True)
class EnsembleTop50Gate:
    """The fixed uncertainty gate in canonical current-pool coordinates."""

    candidate_positions: np.ndarray
    ranking: np.ndarray
    uncertainty_scores: np.ndarray
    cutoff_score: float


def _positions(values: Sequence[int], *, name: str, row_count: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.int64)
    if result.ndim != 1 or not len(result):
        raise ValueError(f"{name} positions must be a nonempty one-dimensional array")
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{name} positions must be unique")
    if np.any(result < 0) or np.any(result >= row_count):
        raise ValueError(f"{name} positions escaped the feature matrix")
    return result


def conditional_gradient_maxdet(
    features: np.ndarray,
    labeled_positions: Sequence[int],
    candidate_positions: Sequence[int],
    batch_size: int,
) -> MaxDetResult:
    """Greedily maximize conditional log determinant in stable float64.

    The supplied candidate order is canonical and resolves exact score ties.
    Feature scaling and conditioning use only the current labeled feature rows.
    """

    started = time.perf_counter()
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("features must be a finite, nonempty two-dimensional matrix")
    labeled = _positions(labeled_positions, name="labeled", row_count=len(matrix))
    candidates = _positions(candidate_positions, name="candidate", row_count=len(matrix))
    if np.intersect1d(labeled, candidates).size:
        raise ValueError("labeled and candidate positions must be disjoint")
    size = int(batch_size)
    if not 0 < size <= len(candidates):
        raise ValueError("batch_size must be in [1, candidate_count]")

    labeled_norm_sq = np.einsum("ij,ij->i", matrix[labeled], matrix[labeled])
    scale = float(np.sqrt(labeled_norm_sq.mean()))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("current labeled features must have positive finite mean squared norm")

    labeled_scaled = matrix[labeled] / scale
    candidate_scaled = matrix[candidates] / scale
    information = np.eye(matrix.shape[1], dtype=np.float64) + labeled_scaled.T @ labeled_scaled
    try:
        cholesky = np.linalg.cholesky(information)
    except np.linalg.LinAlgError as error:
        raise RuntimeError("conditional MaxDet labeled information matrix is not positive definite") from error

    # In the A_L-whitened coordinates, subsequent greedy updates start from I.
    whitened = solve_triangular(
        cholesky,
        candidate_scaled.T,
        lower=True,
        check_finite=False,
    ).T
    precision = np.eye(matrix.shape[1], dtype=np.float64)
    leverages = np.einsum("ij,ij->i", whitened, whitened)
    if not np.isfinite(leverages).all() or np.any(leverages < -1e-12):
        raise RuntimeError("conditional MaxDet produced invalid initial leverages")
    leverages = np.maximum(leverages, 0.0)
    initial_gains = np.log1p(leverages)
    available = np.ones(len(candidates), dtype=bool)
    selected_local: list[int] = []
    trace: list[dict[str, float | int]] = []

    for selection_order in range(size):
        available_positions = np.flatnonzero(available)
        available_leverages = leverages[available_positions]
        if not np.isfinite(available_leverages).all():
            raise RuntimeError("conditional MaxDet encountered a non-finite marginal leverage")
        local = int(available_positions[np.argmax(available_leverages)])
        pivot = float(leverages[local])
        gain = float(np.log1p(pivot))
        gains = np.log1p(np.maximum(available_leverages, 0.0))
        trace.append(
            {
                "selection_order": selection_order,
                "candidate_position": local,
                "feature_position": int(candidates[local]),
                "conditional_leverage": pivot,
                "marginal_logdet_gain": gain,
                "available_count": int(len(available_positions)),
                "available_gain_min": float(gains.min()),
                "available_gain_q25": float(np.quantile(gains, 0.25)),
                "available_gain_median": float(np.quantile(gains, 0.5)),
                "available_gain_q75": float(np.quantile(gains, 0.75)),
                "available_gain_max": float(gains.max()),
            }
        )
        selected_local.append(local)
        available[local] = False

        direction = precision @ whitened[local]
        denominator = 1.0 + float(whitened[local] @ direction)
        if not np.isfinite(denominator) or denominator <= 0:
            raise RuntimeError("conditional MaxDet encountered a non-positive rank-one pivot")
        covariance = whitened @ direction
        leverages[available] -= np.square(covariance[available]) / denominator
        tolerance = 1e-10 * max(1.0, float(np.max(np.abs(leverages[available]), initial=0.0)))
        if np.any(leverages[available] < -tolerance):
            raise RuntimeError("conditional MaxDet rank-one update lost positive semidefiniteness")
        leverages = np.maximum(leverages, 0.0)
        leverages[~available] = 0.0
        precision -= np.outer(direction, direction) / denominator
        precision = 0.5 * (precision + precision.T)
        if not np.isfinite(precision).all():
            raise RuntimeError("conditional MaxDet precision update became non-finite")

    selected_candidate = np.asarray(selected_local, dtype=np.int64)
    selected = candidates[selected_candidate]
    if len(np.unique(selected)) != size or not set(selected.tolist()) <= set(candidates.tolist()):
        raise RuntimeError("conditional MaxDet returned an invalid batch")
    audit: dict[str, float | int | bool | str] = {
        "definition": "greedy conditional D-optimal logdet with current-L conditioning",
        "dtype": "float64",
        "feature_rows": int(len(matrix)),
        "feature_dimension": int(matrix.shape[1]),
        "labeled_count": int(len(labeled)),
        "candidate_count": int(len(candidates)),
        "batch_size": size,
        "normalization_scale": scale,
        "all_finite": True,
        "selected_unique": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return MaxDetResult(selected, selected_candidate, scale, initial_gains, tuple(trace), audit)


def ensemble_top50_gate(predictions: np.ndarray, scales: Sequence[float]) -> EnsembleTop50Gate:
    """Return exactly ``ceil(0.5 * |U_t|)`` candidates with stable ties."""

    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("ensemble predictions must be K x pool rows x outputs")
    pool_size = int(values.shape[1])
    if pool_size < 1:
        raise ValueError("ensemble uncertainty gate requires a nonempty pool")
    ranking, scores = ensemble_uncertainty_select(values, scales, pool_size)
    shortlist_size = int(math.ceil(0.50 * pool_size))
    candidates = np.asarray(ranking[:shortlist_size], dtype=np.int64)
    if len(candidates) != shortlist_size or len(np.unique(candidates)) != shortlist_size:
        raise RuntimeError("Top50 uncertainty gate returned an invalid candidate set")
    return EnsembleTop50Gate(
        candidate_positions=candidates,
        ranking=np.asarray(ranking, dtype=np.int64),
        uncertainty_scores=np.asarray(scores, dtype=np.float64),
        cutoff_score=float(scores[candidates[-1]]),
    )
