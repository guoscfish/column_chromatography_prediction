"""CPU-only diagnostics of already frozen, ID-aligned historical feature banks.

This module never loads a model, target table, prediction table, or response. The
caller owns snapshotting and verification of the supplied historical bank. Each
feature space must be evaluated separately. Raw distances from different model
states or representations must not be compared as if they shared a metric.

Reference IDs are fixed by the caller, preferably the same SHA-sorted subset of
the seed's initial training candidate pool for all methods and rounds. They are
NOT removed as the labelled set grows, and actual/random selected IDs are NOT
removed. Already labelled and selected reference points have exact zero distance.
The same reference and preselection labelled set are used for every random batch.
"""
from __future__ import annotations

import os

# Set before importing NumPy; importing this module does not invoke GPU libraries.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import hashlib
import json
from typing import Iterable, Sequence

import numpy as np


DEFAULT_RANDOM_SEEDS = tuple(range(90210, 90226))


def fixed_reference_ids(candidate_ids: Iterable[str], size: int = 128,
                        salt: str = "nocw_free_selection_diagnostics_v1") -> list[str]:
    """Select a deterministic target-free reference once from a seed's U0.

    Keep the resulting IDs unchanged in subsequent calls to evaluate_geometry.
    A salt containing the seed is allowed; method/round must not change the salt.
    """
    ids = [str(v) for v in candidate_ids]
    if size <= 0 or len(set(ids)) != len(ids):
        raise ValueError("Reference candidates must be unique and size positive")
    return sorted(ids, key=lambda v: (hashlib.sha256(
        (salt + "\0" + v).encode("utf-8")).digest(), v))[:size]


def _ids(values: Iterable[str], name: str) -> list[str]:
    result = [str(v) for v in values]
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate stable row IDs")
    return result


def _nearest(left: np.ndarray, right: np.ndarray, chunk_size: int = 128) -> np.ndarray:
    """Nearest Euclidean distances with bounded 2-D blocks.

    Recalculate the winning pair using direct differences, so cancellation in
    norm-dot subtraction cannot turn a repeated vector into a nonzero distance.
    This allocates at most chunk_size**2 distances, never a pool-by-pool matrix.
    """
    if not len(right):
        raise ValueError("Nearest-distance reference cannot be empty")
    result = np.full(len(left), np.inf, dtype=np.float64)
    for a0 in range(0, len(left), chunk_size):
        a = left[a0:a0 + chunk_size]
        best = np.full(len(a), np.inf)
        for b0 in range(0, len(right), chunk_size):
            b = right[b0:b0 + chunk_size]
            squared = (np.einsum("ij,ij->i", a, a)[:, None]
                       + np.einsum("ij,ij->i", b, b)[None, :] - 2 * (a @ b.T))
            winner = np.argmin(squared, axis=1)
            difference = a - b[winner]
            exact_squared = np.einsum("ij,ij->i", difference, difference)
            best = np.minimum(best, exact_squared)
        result[a0:a0 + len(a)] = np.sqrt(best)
    return result


def _batch_nn(batch: np.ndarray, chunk_size: int) -> np.ndarray:
    if len(batch) < 2:
        return np.full(len(batch), np.nan)
    # A batch is only 32 points. Direct differences avoid cancellation entirely.
    values = np.full(len(batch), np.inf)
    for i, row in enumerate(batch):
        difference = batch - row
        d2 = np.einsum("ij,ij->i", difference, difference)
        d2[i] = np.inf
        values[i] = np.sqrt(d2.min())
    return values


def _metrics(batch: np.ndarray, labelled: np.ndarray, reference: np.ndarray,
             before: np.ndarray, chunk_size: int,
             reference_selected_mask: np.ndarray | None = None) -> dict[str, float | None]:
    to_l = _nearest(batch, labelled, chunk_size)
    nn = _batch_nn(batch, chunk_size)
    after = np.minimum(before, _nearest(reference, batch, chunk_size))
    if reference_selected_mask is not None:
        after[reference_selected_mask] = 0.0
    baseline = float(before.mean())
    drop = float((before - after).mean())
    return {
        "batch_nn_distance_mean": float(nn.mean()) if len(batch) >= 2 else None,
        "selected_to_l_distance_mean": float(to_l.mean()),
        "reference_coverage_distance_after_mean": float(after.mean()),
        "reference_coverage_distance_drop": drop,
        "reference_coverage_relative_drop": drop / baseline if baseline > 0 else None,
    }


def evaluate_geometry(features: np.ndarray, row_ids: Sequence[str],
                      labeled_ids: Sequence[str], unlabeled_ids: Sequence[str],
                      selected_ids: Sequence[str], reference_ids: Sequence[str] | None = None,
                      random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
                      reference_size: int = 128,
                      reference_seed: str = "nocw_free_selection_diagnostics_v1",
                      chunk_size: int = 128) -> dict[str, object]:
    """Return one state's actual metrics and within-state random midrank percentiles.

    Random batches are uniformly sampled without replacement from the SAME U_t,
    with size equal to the actual accepted batch. All randomness is local NumPy
    Generator state; the global random state is untouched. The percentile is
    (count(random < actual) + 0.5 * count(ties)) / number_of_random_batches;
    ties use np.isclose(rtol=1e-10, atol=1e-12) for numerical comparison only.
    With 16 reference batches its resolution is coarse. A high percentile means
    a larger metric; only coverage-drop metrics have an unambiguous 'more cover'
    direction. No percentile is an information-gain or learning-efficiency score.

    If reference_ids is None, a deterministic reference is selected from row_ids.
    Callers must pass the same list at later rounds to retain a fixed reference.
    features must contain only the explicitly supplied L_t and U_t, in any order.
    """
    all_ids = _ids(row_ids, "row_ids")
    labelled = _ids(labeled_ids, "labeled_ids")
    unlabelled = _ids(unlabeled_ids, "unlabeled_ids")
    selected = _ids(selected_ids, "selected_ids")
    refs = _ids(reference_ids if reference_ids is not None else fixed_reference_ids(
        all_ids, reference_size, reference_seed), "reference_ids")
    seeds = [int(s) for s in random_seeds]
    if not labelled or not selected or not refs or not seeds:
        raise ValueError("L, batch, reference and random seeds must be nonempty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Random reference seeds must be unique")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if set(labelled) & set(unlabelled):
        raise ValueError("L_t and U_t overlap")
    if set(labelled) | set(unlabelled) != set(all_ids):
        raise ValueError("Bank IDs must exactly cover L_t union U_t")
    if not set(selected) <= set(unlabelled) or len(selected) > len(unlabelled):
        raise ValueError("Accepted batch must be an unlabelled subset")
    if not set(refs) <= set(all_ids):
        raise ValueError("Fixed references are absent from the bank")
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != len(all_ids) or not np.isfinite(x).all():
        raise ValueError("Expected one finite feature row per stable ID")
    positions = {v: i for i, v in enumerate(all_ids)}
    def take(ids):
        return x[[positions[v] for v in ids]]
    l, u, batch, reference = map(take, (labelled, unlabelled, selected, refs))
    before = _nearest(reference, l, chunk_size)
    # Stable IDs, rather than floating-point distance cancellation, establish
    # exact self-distance for references already in L or in a selected batch.
    before[np.array([v in set(labelled) for v in refs])] = 0.0
    actual = _metrics(batch, l, reference, before, chunk_size,
                      np.array([v in set(selected) for v in refs]))
    simulated: dict[str, list[float]] = {name: [] for name in actual}
    for seed in seeds:
        indices = np.random.default_rng(seed).choice(len(u), len(batch), replace=False)
        random_ids = {unlabelled[i] for i in indices}
        metrics = _metrics(u[indices], l, reference, before, chunk_size,
                           np.array([v in random_ids for v in refs]))
        for name, value in metrics.items():
            if value is not None:
                simulated[name].append(value)
    result: dict[str, object] = {
        "feature_dimension": x.shape[1],
        "reference_n": len(refs),
        "reference_ids_sha256": hashlib.sha256(json.dumps(
            refs, separators=(",", ":")).encode()).hexdigest(),
        "reference_labeled_before_n": len(set(refs) & set(labelled)),
        "reference_selected_n": len(set(refs) & set(selected)),
        "reference_positive_before_n": int(np.count_nonzero(before > 0)),
        "reference_coverage_distance_before_mean": float(before.mean()),
        "random_n": len(seeds),
        "random_seeds": json.dumps(seeds, separators=(",", ":")),
        "random_sampling_pool_n": len(unlabelled),
        "random_reference_policy": "same_fixed_reference_including_labeled_and_selected",
        **actual,
    }
    for name, value in actual.items():
        values = np.asarray(simulated[name], dtype=np.float64)
        result[name + "_random_mean"] = float(values.mean()) if len(values) else None
        result[name + "_random_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
        if value is None or not len(values):
            result[name + "_random_percentile"] = None
        else:
            ties = np.isclose(values, value, rtol=1e-10, atol=1e-12)
            less = (values < value) & ~ties
            result[name + "_random_percentile"] = float((less.sum() + 0.5 * ties.sum()) / len(values))
    return result


def self_check() -> dict[str, object]:
    """Small, analytic checks of distances, fixed zeros, RNG and ID validation."""
    ids = ["a", "b", "c", "d", "e"]
    features = np.array([[0., 0.], [1., 0.], [3., 0.], [0., 4.], [0., 0.]])
    kwargs = dict(features=features, row_ids=ids, labeled_ids=["a"],
                  unlabeled_ids=["b", "c", "d", "e"], selected_ids=["b", "c"],
                  reference_ids=["a", "b", "c", "d", "e"], random_seeds=[31, 47])
    result = evaluate_geometry(**kwargs, chunk_size=2)
    assert result["batch_nn_distance_mean"] == 2.
    assert result["selected_to_l_distance_mean"] == 2.
    assert result["reference_coverage_distance_before_mean"] == 1.6
    assert result["reference_coverage_distance_after_mean"] == 0.8
    assert result["reference_coverage_relative_drop"] == 0.5
    assert result["reference_positive_before_n"] == 3
    assert result["reference_labeled_before_n"] == 1
    assert result["reference_selected_n"] == 2
    assert result == evaluate_geometry(**kwargs, chunk_size=128)
    assert fixed_reference_ids(reversed(ids), 3) == fixed_reference_ids(ids, 3)
    try:
        evaluate_geometry(**{**kwargs, "selected_ids": ["b", "b"]})
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate batch IDs were accepted")
    return {"passed": True, "analytic_distance_checks": 5,
            "fixed_reference_and_zero_checks": 3,
            "chunk_parity": True, "deterministic_reference": True,
            "duplicate_id_rejected": True}


if __name__ == "__main__":
    print(json.dumps(self_check(), ensure_ascii=False, indent=2))
