#!/usr/bin/env python3
"""Deterministic E2 acquisition and signal-agreement primitives."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


SIGNAL_COLUMNS = {
    "quantile_width": "quantile_width",
    "ensemble": "ensemble_score",
    "latent_distance": "latent_distance",
}


def fit_standardizer(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 2 or len(reference) == 0:
        raise ValueError("reference must be a non-empty 2D array")
    mean = reference.mean(axis=0, keepdims=True)
    scale = reference.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def transform_standardized(
    values: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - mean) / scale


def build_joint_representation(
    labeled_h: np.ndarray,
    pool_h: np.ndarray,
    labeled_conditions: np.ndarray,
    pool_conditions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Feature-wise standardize graph and condition blocks separately.

    The rule is frozen; statistics use the current non-validation labeled
    training reference only and never pool/test values.
    """

    h_mean, h_scale = fit_standardizer(labeled_h)
    c_mean, c_scale = fit_standardizer(labeled_conditions)
    labeled = np.column_stack(
        [
            transform_standardized(labeled_h, h_mean, h_scale),
            transform_standardized(labeled_conditions, c_mean, c_scale),
        ]
    )
    pool = np.column_stack(
        [
            transform_standardized(pool_h, h_mean, h_scale),
            transform_standardized(pool_conditions, c_mean, c_scale),
        ]
    )
    audit = {
        "embedding_normalization": "featurewise z-score fitted on current labeled train",
        "condition_normalization": "featurewise z-score fitted on current labeled train",
        "joint_representation": "concatenate standardized 128D h_graph and 9D conditions",
        "distance": "euclidean",
    }
    return labeled, pool, audit


def mean_knn_distance(
    reference: np.ndarray, candidates: np.ndarray, neighbors: int = 5
) -> np.ndarray:
    if len(reference) == 0 or len(candidates) == 0:
        raise ValueError("reference and candidates must be non-empty")
    k = min(int(neighbors), len(reference))
    model = NearestNeighbors(n_neighbors=k, metric="euclidean")
    model.fit(reference)
    distances, _ = model.kneighbors(candidates)
    return distances.mean(axis=1)


def minimum_reference_distance(
    reference: np.ndarray, candidates: np.ndarray, chunk_size: int = 512
) -> np.ndarray:
    if len(reference) == 0 or len(candidates) == 0:
        raise ValueError("reference and candidates must be non-empty")
    result = np.empty(len(candidates), dtype=np.float64)
    for start in range(0, len(candidates), chunk_size):
        stop = min(start + chunk_size, len(candidates))
        result[start:stop] = cdist(candidates[start:stop], reference).min(axis=1)
    return result


def deterministic_descending_order(scores: np.ndarray, sample_ids: Sequence[str]) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    ids = np.asarray([str(value) for value in sample_ids])
    if len(scores) != len(ids):
        raise ValueError("scores and sample_ids must align")
    return np.lexsort((ids, -scores))


def top_score_select(
    sample_ids: Sequence[str], scores: np.ndarray, batch_size: int
) -> list[str]:
    if batch_size < 1 or batch_size > len(sample_ids):
        raise ValueError("invalid batch_size")
    order = deterministic_descending_order(scores, sample_ids)
    ids = np.asarray([str(value) for value in sample_ids])
    return ids[order[:batch_size]].tolist()


def farthest_first_select(
    sample_ids: Sequence[str],
    candidate_representation: np.ndarray,
    reference_representation: np.ndarray,
    batch_size: int,
) -> list[str]:
    """Deterministic k-center with distances updated after every selection."""

    ids = np.asarray([str(value) for value in sample_ids])
    candidates = np.asarray(candidate_representation, dtype=np.float64)
    if len(ids) != len(candidates):
        raise ValueError("sample_ids and candidate_representation must align")
    if batch_size < 1 or batch_size > len(ids):
        raise ValueError("invalid batch_size")
    minimum = minimum_reference_distance(reference_representation, candidates)
    available = np.ones(len(ids), dtype=bool)
    selected_positions: list[int] = []
    for _ in range(batch_size):
        available_positions = np.flatnonzero(available)
        local_order = deterministic_descending_order(
            minimum[available_positions], ids[available_positions]
        )
        position = int(available_positions[local_order[0]])
        selected_positions.append(position)
        available[position] = False
        if available.any():
            new_distance = np.linalg.norm(candidates - candidates[position], axis=1)
            minimum = np.minimum(minimum, new_distance)
            minimum[~available] = -np.inf
    return ids[np.asarray(selected_positions)].tolist()


def hybrid_select(
    sample_ids: Sequence[str],
    ensemble_scores: np.ndarray,
    candidate_representation: np.ndarray,
    reference_representation: np.ndarray,
    batch_size: int,
    prefilter_fraction: float = 0.25,
) -> tuple[list[str], list[str]]:
    if prefilter_fraction != 0.25:
        raise ValueError("E2 Hybrid prefilter_fraction is frozen at 0.25")
    count = max(batch_size, int(math.ceil(prefilter_fraction * len(sample_ids))))
    order = deterministic_descending_order(ensemble_scores, sample_ids)[:count]
    ids = np.asarray([str(value) for value in sample_ids])
    subset_ids = ids[order]
    selected = farthest_first_select(
        subset_ids,
        np.asarray(candidate_representation)[order],
        reference_representation,
        batch_size,
    )
    return selected, subset_ids.tolist()


def batch_distance_summary(representation: np.ndarray) -> tuple[float, float]:
    if len(representation) < 2:
        return 0.0, 0.0
    distances = pdist(np.asarray(representation, dtype=np.float64), metric="euclidean")
    return float(distances.mean()), float(distances.min())


def signal_agreement_rows(
    frame: pd.DataFrame,
    metadata: dict,
    signal_columns: dict[str, str] | None = None,
) -> list[dict]:
    columns = signal_columns or SIGNAL_COLUMNS
    rows = []
    for signal_a, signal_b in combinations(columns, 2):
        values_a = frame[columns[signal_a]].to_numpy(dtype=float)
        values_b = frame[columns[signal_b]].to_numpy(dtype=float)
        spearman = (
            float(spearmanr(values_a, values_b).statistic)
            if len(frame) >= 3 and np.unique(values_a).size > 1 and np.unique(values_b).size > 1
            else float("nan")
        )
        row = {**metadata, "signal_A": signal_a, "signal_B": signal_b, "spearman": spearman}
        for fraction, label in ((0.05, "top5_overlap"), (0.10, "top10_overlap"), (0.20, "top20_overlap")):
            count = max(1, int(math.ceil(fraction * len(frame))))
            top_a = set(deterministic_descending_order(values_a, frame["sample_id"])[:count])
            top_b = set(deterministic_descending_order(values_b, frame["sample_id"])[:count])
            row[label] = float(len(top_a & top_b) / count)
            row[f"{label}_count"] = count
        rows.append(row)
    return rows
