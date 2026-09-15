"""Label-free per-round acquisition for the sequential B=32 study."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .coverage import coreset_tp_select
from .lcmd import lcmd_tp_select
from .sequential_protocol import (
    ACQUISITION_ROUNDS,
    BATCH_SIZE,
    SHORTLIST_FRACTION,
    random_trajectory_seed,
)
from .uncertainty import ensemble_uncertainty_select


def deterministic_random_permutation(original_u0_ids: Sequence[str], outer_seed: int) -> tuple[str, ...]:
    """Return the one preregistered permutation shared by every Random round."""

    ids = tuple(str(value) for value in original_u0_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("original U0 IDs must be nonempty and unique")
    order = np.random.default_rng(random_trajectory_seed(outer_seed)).permutation(len(ids))
    return tuple(ids[int(position)] for position in order)


def random_round_batch(
    original_u0_ids: Sequence[str],
    current_u_ids: Sequence[str],
    outer_seed: int,
    acquisition_round: int,
) -> tuple[str, ...]:
    """Take the next exact slice of the single nested Random permutation."""

    round_index = int(acquisition_round)
    if not 1 <= round_index <= ACQUISITION_ROUNDS:
        raise ValueError("acquisition_round must be in [1, 21]")
    permutation = deterministic_random_permutation(original_u0_ids, outer_seed)
    start = (round_index - 1) * BATCH_SIZE
    stop = round_index * BATCH_SIZE
    selected = permutation[start:stop]
    if len(selected) != BATCH_SIZE:
        raise RuntimeError("random permutation cannot supply a complete registered batch")
    current = tuple(str(value) for value in current_u_ids)
    current_set = set(current)
    if len(current_set) != len(current):
        raise ValueError("current U_t IDs must be unique")
    if not set(selected) <= current_set:
        raise RuntimeError("Random round state is not the expected nested trajectory")
    if set(permutation[:start]) & current_set:
        raise RuntimeError("previous Random selections remain in current U_t")
    return selected


def v2_hybrid_select_current(
    representations: np.ndarray,
    ensemble_predictions: np.ndarray,
    labeled_count: int,
    scales: tuple[float, float],
    batch_size: int = BATCH_SIZE,
) -> dict[str, np.ndarray | int]:
    """Current-model Top-25% uncertainty then latent farthest-first.

    ``representations`` is ordered as all current L_t rows followed by all
    current U_t rows. Consequently every current labeled row, not only L0, is
    installed as a farthest-first center.
    """

    latent = np.asarray(representations, dtype=np.float64)
    labeled = int(labeled_count)
    if latent.ndim != 2 or latent.shape[1] != 128:
        raise ValueError("current QGeoGNN-V2 128D representations are required")
    if not 0 < labeled < len(latent):
        raise ValueError("representations must contain current L_t then current U_t")
    pool_size = len(latent) - labeled
    if not 0 < int(batch_size) <= pool_size:
        raise ValueError("invalid Hybrid batch size")
    ranking, scores = ensemble_uncertainty_select(ensemble_predictions, scales, pool_size)
    shortlist_size = max(int(batch_size), int(math.ceil(SHORTLIST_FRACTION * pool_size)))
    shortlist_positions = np.asarray(ranking[:shortlist_size], dtype=np.int64)
    local = coreset_tp_select(
        latent[labeled + shortlist_positions],
        latent[:labeled],
        int(batch_size),
    )
    selected = shortlist_positions[local]
    if len(selected) != int(batch_size) or len(np.unique(selected)) != int(batch_size):
        raise RuntimeError("Hybrid selection size/uniqueness violation")
    if not set(selected) <= set(shortlist_positions):
        raise RuntimeError("Hybrid selection escaped the current-U_t shortlist")
    return {
        "selected_pool_positions": selected,
        "shortlist_pool_positions": shortlist_positions,
        "uncertainty_ranking": np.asarray(ranking, dtype=np.int64),
        "uncertainty_scores": np.asarray(scores, dtype=np.float64),
        "shortlist_size": shortlist_size,
        "center_count": labeled,
        "pool_size": pool_size,
    }


def acquire_sequential_batch(
    *,
    method: str,
    labeled_count: int,
    batch_size: int = BATCH_SIZE,
    gradient_features: np.ndarray | None = None,
    representations: np.ndarray | None = None,
    ensemble_predictions: np.ndarray | None = None,
    scales: tuple[float, float] | None = None,
) -> dict[str, object]:
    """Run one current-round non-Random acquisition without any label input."""

    if int(batch_size) != BATCH_SIZE:
        raise ValueError("the sequential study freezes B=32")
    labeled = int(labeled_count)
    if method == "hybrid":
        if representations is None or ensemble_predictions is None or scales is None:
            raise ValueError("Hybrid requires current representations, ensemble predictions, and L0 scales")
        return v2_hybrid_select_current(
            representations, ensemble_predictions, labeled, scales, int(batch_size)
        )
    if method == "lcmd":
        features = np.asarray(gradient_features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != 512 or not 0 < labeled < len(features):
            raise ValueError("LCMD requires current L_t + U_t 512D gradient features")
        result = lcmd_tp_select(features[labeled:], features[:labeled], int(batch_size))
        return {
            "selected_pool_positions": result.selected_pool_positions,
            "center_count": labeled,
            "pool_size": len(features) - labeled,
            "initial_nearest_center": result.initial_nearest_center,
            "initial_nearest_sq_distance": result.initial_nearest_sq_distance,
            "trace": result.trace,
        }
    raise ValueError("non-Random sequential method must be 'hybrid' or 'lcmd'")


def validate_trajectory_transition(
    previous_labeled_ids: Sequence[str],
    previous_unlabeled_ids: Sequence[str],
    selected_ids: Sequence[str],
    next_labeled_ids: Sequence[str],
    next_unlabeled_ids: Sequence[str],
) -> None:
    """Enforce exact +32/-32 membership and ordering semantics."""

    previous_l = tuple(str(value) for value in previous_labeled_ids)
    previous_u = tuple(str(value) for value in previous_unlabeled_ids)
    selected = tuple(str(value) for value in selected_ids)
    next_l = tuple(str(value) for value in next_labeled_ids)
    next_u = tuple(str(value) for value in next_unlabeled_ids)
    for name, values in (("previous L_t", previous_l), ("previous U_t", previous_u),
                         ("selected", selected), ("next L_t", next_l), ("next U_t", next_u)):
        if len(set(values)) != len(values):
            raise RuntimeError(f"{name} contains duplicate IDs")
    if len(selected) != BATCH_SIZE:
        raise RuntimeError("every acquisition round must select exactly 32 IDs")
    if not set(selected) <= set(previous_u):
        raise RuntimeError("selected IDs must come from previous U_t")
    if next_l != previous_l + selected:
        raise RuntimeError("L_t must append the ordered selected batch")
    selected_set = set(selected)
    if next_u != tuple(value for value in previous_u if value not in selected_set):
        raise RuntimeError("U_t must preserve order while removing the selected batch")
    if len(next_l) != len(previous_l) + BATCH_SIZE or len(next_u) != len(previous_u) - BATCH_SIZE:
        raise RuntimeError("sequential trajectory must change by exactly one complete batch")
