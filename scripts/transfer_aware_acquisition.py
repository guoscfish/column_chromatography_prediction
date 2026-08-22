"""Pure, deterministic transfer-aware acquisition primitives for D43."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def transfer_prediction_shift(
    source_mean: np.ndarray, target_mean: np.ndarray, scales: dict[str, float]
) -> np.ndarray:
    source = np.asarray(source_mean, dtype=float)
    target = np.asarray(target_mean, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source_mean and target_mean must both have shape (rows, 2)")
    scale = np.asarray([scales["V1"], scales["V2"]], dtype=float)
    if np.any(scale <= 0):
        raise ValueError("target scales must be positive")
    return 0.5 * np.sum(np.abs(target - source) / scale, axis=1)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Average-tie percentile rank in [1/n, 1], deterministic for equal input."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("rank values must be a finite vector")
    return pd.Series(values).rank(method="average", pct=True).to_numpy(float)


def target_representativeness(representation: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return (-mean kNN distance, mean kNN distance) within the unlabeled pool."""
    z = np.asarray(representation, dtype=float)
    if z.ndim != 2 or len(z) <= k or k < 1:
        raise ValueError("pool must contain more than k rows")
    distance = cdist(z, z, metric="euclidean")
    np.fill_diagonal(distance, np.inf)
    knn = np.partition(distance, kth=k - 1, axis=1)[:, :k].mean(axis=1)
    return -knn, knn


def top_score(ids: list[str], score: np.ndarray, batch_size: int) -> list[str]:
    if len(ids) != len(score) or batch_size < 1 or batch_size > len(ids):
        raise ValueError("invalid ids/score/batch_size")
    normalized_ids = list(map(str, ids))
    order = sorted(range(len(normalized_ids)), key=lambda i: (-float(score[i]), normalized_ids[i]))
    return [normalized_ids[i] for i in order[:batch_size]]


def facility_location_select(
    ids: list[str], representation: np.ndarray, batch_size: int
) -> list[str]:
    """Greedy medoid-like coverage with lexicographic sample-id tie breaks."""
    ids = list(map(str, ids)); z = np.asarray(representation, dtype=float)
    if len(ids) != len(z) or len(set(ids)) != len(ids) or not 1 <= batch_size <= len(ids):
        raise ValueError("invalid facility-location inputs")
    distance = cdist(z, z, metric="euclidean")
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    total = distance.sum(axis=0)
    first = min(order, key=lambda i: (float(total[i]), ids[i]))
    chosen = [first]; nearest = distance[:, first].copy()
    while len(chosen) < batch_size:
        candidates = [i for i in order if i not in chosen]
        gain = {i: float(np.maximum(nearest - distance[:, i], 0.0).sum()) for i in candidates}
        next_index = min(candidates, key=lambda i: (-gain[i], ids[i]))
        chosen.append(next_index); nearest = np.minimum(nearest, distance[:, next_index])
    return [ids[i] for i in chosen]


def transfer_aware_selections(
    ids: list[str], shift: np.ndarray, uncertainty: np.ndarray,
    representation: np.ndarray, batch_size: int = 10, shortlist_fraction: float = 0.25,
) -> tuple[dict[str, list[str]], dict[str, np.ndarray | list[str]]]:
    ids = list(map(str, ids)); shift = np.asarray(shift, float); uncertainty = np.asarray(uncertainty, float)
    if len(ids) != len(shift) or len(ids) != len(uncertainty):
        raise ValueError("score lengths do not match ids")
    rank_shift = percentile_rank(shift); rank_uncertainty = percentile_rank(uncertainty)
    t2 = 0.5 * rank_shift + 0.5 * rank_uncertainty
    t1_ids = top_score(ids, shift, batch_size); t2_ids = top_score(ids, t2, batch_size)
    shortlist_size = max(batch_size, int(math.ceil(shortlist_fraction * len(ids))))
    shortlist_ids = top_score(ids, t2, shortlist_size)
    position = {sid: i for i, sid in enumerate(ids)}
    shortlist_rep = np.asarray(representation)[[position[x] for x in shortlist_ids]]
    t3_ids = facility_location_select(shortlist_ids, shortlist_rep, batch_size)
    return {
        "transfer_shift": t1_ids,
        "transfer_shift_uncertainty": t2_ids,
        "transfer_shift_uncertainty_representative": t3_ids,
    }, {
        "rank_shift": rank_shift, "rank_uncertainty": rank_uncertainty,
        "transfer_shift_uncertainty_score": t2,
        "shortlist_ids": shortlist_ids,
    }
