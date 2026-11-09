"""Frozen top-quarter ensemble-disagreement + latent k-center mathematics."""
import math
import numpy as np
from .lcmd import _squared_distances

SHORTLIST_FRACTION = .25
BATCH_SIZE = 32

def ensemble_uncertainty_select(predictions, scales, batch_size):
    values = np.asarray(predictions, dtype=np.float64)
    scale = np.asarray(scales, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != 3 or values.shape[2] != 6:
        raise ValueError("expected K=3 by pool rows by six outputs")
    if scale.shape != (2,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("two positive finite L0 scales required")
    if not np.isfinite(values).all() or not 0 < batch_size <= values.shape[1]:
        raise ValueError("invalid predictions or batch size")
    disagreement = values[:, :, [1, 4]].std(axis=0, ddof=0)
    scores = np.linalg.norm(disagreement / scale, axis=1)
    return np.argsort(-scores, kind="stable")[:batch_size], scores

def coreset_tp_select(pool_features, train_features, batch_size):
    pool, train = np.asarray(pool_features), np.asarray(train_features)
    if pool.ndim != 2 or train.ndim != 2 or pool.shape[1] != train.shape[1] or not len(train):
        raise ValueError("aligned pool and nonempty L0 matrices required")
    if not np.isfinite(pool).all() or not np.isfinite(train).all() or not 0 < batch_size <= len(pool):
        raise ValueError("invalid features or batch size")
    distance = _squared_distances(pool, train).min(axis=1)
    available = np.ones(len(pool), dtype=bool)
    selected = []
    for _ in range(batch_size):
        point = int(np.argmax(np.where(available, distance, -np.inf)))
        selected.append(point)
        available[point] = False
        distance = np.minimum(distance, _squared_distances(pool, pool[point:point + 1])[:, 0])
    return np.asarray(selected, dtype=np.int64)

def hybrid_select(
    representations: np.ndarray,
    ensemble_predictions: np.ndarray,
    l0_count: int,
    scales: tuple[float, float],
    batch_size: int = BATCH_SIZE,
) -> dict[str, np.ndarray | int]:
    """Top-25% current-V2 uncertainty followed by current-V2 latent k-center.

    This function accepts only label-free acquisition inputs.  The representation
    is required to contain the L0 rows followed by the canonical U0 rows.
    """

    representation = np.asarray(representations, dtype=np.float64)
    if representation.ndim != 2 or representation.shape[1] != 128:
        raise ValueError("current QGeoGNN-V2 128D representations are required")
    if not 0 < int(l0_count) < len(representation):
        raise ValueError("representation must contain nonempty L0 and U0 blocks")
    pool_size = len(representation) - int(l0_count)
    if not 0 < int(batch_size) <= pool_size:
        raise ValueError("invalid hybrid batch size")
    ranking, scores = ensemble_uncertainty_select(ensemble_predictions, scales, pool_size)
    # Historical Hybrid used ceil(fraction * |U0|).  For 2,997 U0 rows this
    # deterministically selects the smallest integer no smaller than 25%: 750.
    shortlist_size = max(int(batch_size), int(math.ceil(SHORTLIST_FRACTION * pool_size)))
    shortlist_positions = np.asarray(ranking[:shortlist_size], dtype=np.int64)
    selected_local = coreset_tp_select(
        representation[int(l0_count) + shortlist_positions], representation[:int(l0_count)], int(batch_size)
    )
    selected_positions = shortlist_positions[selected_local]
    if len(selected_positions) != int(batch_size) or len(np.unique(selected_positions)) != int(batch_size):
        raise RuntimeError("hybrid selection size/uniqueness violation")
    if not set(selected_positions).issubset(set(shortlist_positions)):
        raise RuntimeError("hybrid escaped its uncertainty shortlist")
    return {
        "selected_positions": selected_positions,
        "shortlist_positions": shortlist_positions,
        "uncertainty_ranking": np.asarray(ranking, dtype=np.int64),
        "uncertainty_scores": np.asarray(scores, dtype=np.float64),
        "shortlist_size": shortlist_size,
    }
