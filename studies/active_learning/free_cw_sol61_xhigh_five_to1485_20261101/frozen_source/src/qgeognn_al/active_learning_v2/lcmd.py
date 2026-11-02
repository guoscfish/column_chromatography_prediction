"""Largest Cluster Maximum Distance selection in training-pool (TP) mode."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LCMDResult:
    """Selected pool positions and an auditable selection trace."""

    selected_pool_positions: np.ndarray
    initial_nearest_center: np.ndarray
    initial_nearest_sq_distance: np.ndarray
    trace: tuple[dict[str, float | int], ...]


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return a stable Euclidean squared-distance matrix."""

    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    values = (
        np.square(left64).sum(axis=1, keepdims=True)
        + np.square(right64).sum(axis=1)[None, :]
        - 2.0 * left64 @ right64.T
    )
    return np.maximum(values, 0.0)


def lcmd_tp_select(
    pool_features: np.ndarray,
    train_features: np.ndarray,
    batch_size: int,
) -> LCMDResult:
    """Select a batch with corrected LCMD-TP semantics.

    Existing training rows are installed as centers before the first pool
    selection.  At each step the cluster with the largest sum of squared
    nearest-center distances is chosen, then its farthest point becomes the
    next center.  Ties are resolved by the stable input order, matching
    ``argmax`` behavior in the reference implementation.
    """

    pool = np.asarray(pool_features, dtype=np.float64)
    train = np.asarray(train_features, dtype=np.float64)
    if pool.ndim != 2 or train.ndim != 2 or pool.shape[1] != train.shape[1]:
        raise ValueError("pool/train features must be aligned two-dimensional arrays")
    if not len(train):
        raise ValueError("TP mode requires at least one existing training center")
    if not 0 < int(batch_size) <= len(pool):
        raise ValueError("batch_size must be in [1, len(pool)]")
    if not np.isfinite(pool).all() or not np.isfinite(train).all():
        raise ValueError("LCMD features must be finite")

    initial = _squared_distances(pool, train)
    closest = np.argmin(initial, axis=1).astype(np.int64)
    minimum = initial[np.arange(len(pool)), closest]
    initial_closest = closest.copy()
    initial_minimum = minimum.copy()
    available = np.ones(len(pool), dtype=bool)
    selected: list[int] = []
    trace: list[dict[str, float | int]] = []
    center_count = len(train)

    for selection_order in range(int(batch_size)):
        cluster_scores = np.bincount(
            closest[available], weights=minimum[available], minlength=center_count
        )
        cluster = int(np.argmax(cluster_scores))
        candidates = np.flatnonzero(available & (closest == cluster))
        if not len(candidates):
            raise RuntimeError("LCMD selected an empty largest cluster")
        point = int(candidates[np.argmax(minimum[candidates])])
        selected.append(point)
        trace.append(
            {
                "selection_order": selection_order,
                "pool_position": point,
                "selected_from_center": cluster,
                "largest_cluster_score": float(cluster_scores[cluster]),
                "nearest_center_sq_distance_before_selection": float(minimum[point]),
            }
        )
        available[point] = False

        distance_to_new = _squared_distances(pool, pool[point : point + 1])[:, 0]
        improved = available & (distance_to_new < minimum)
        closest[improved] = center_count
        minimum[improved] = distance_to_new[improved]
        closest[point] = center_count
        minimum[point] = 0.0
        center_count += 1

    result = np.asarray(selected, dtype=np.int64)
    if len(np.unique(result)) != int(batch_size):
        raise RuntimeError("LCMD returned duplicate pool positions")
    return LCMDResult(result, initial_closest, initial_minimum, tuple(trace))
