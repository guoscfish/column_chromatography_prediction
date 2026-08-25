"""Pure statistical and candidate-selection functions for D45."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

SCORE_COLUMNS = ("ensemble_score", "quantile_width", "coverage_score")
CHALLENGE_TAGS = ("ensemble_top8", "qwidth_top8", "coverage_top8")


def compute_marginal_utility(baseline_error: float, after_error: float) -> float:
    """Return error improvement; positive values mean the candidate helped."""
    return float(baseline_error - after_error)


def _deterministic_top_ids(pool: pd.DataFrame, score: str, top_n: int) -> set[str]:
    ranked = pool.sort_values(
        [score, "sample_id"], ascending=[False, True], kind="mergesort"
    )
    return set(ranked.head(min(top_n, len(ranked)))["sample_id"].astype(str))


def build_diagnostic_candidate_subset(
    pool: pd.DataFrame,
    seed: int = 4501,
    random_n: int = 48,
    top_n: int = 8,
    challenge_ids: Mapping[str, Sequence[str]] | None = None,
    include_challenge: bool = True,
) -> pd.DataFrame:
    """Build the deterministic representative/challenge union without truth.

    Only IDs and the three frozen acquisition-score columns are copied.  The
    optional challenge IDs allow exact E4 Coverage farthest-first selection.
    """
    required = {"sample_id", *SCORE_COLUMNS}
    missing = required - set(pool.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    ids = pool["sample_id"].astype(str)
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("sample_id must be non-null and unique")
    if random_n < 1 or top_n < 1:
        raise ValueError("random_n and top_n must be positive")

    safe_pool = pool[["sample_id", *SCORE_COLUMNS]].copy()
    safe_pool["sample_id"] = ids
    if not np.isfinite(safe_pool[list(SCORE_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("candidate scores must be finite")
    rng = np.random.default_rng(seed)
    positions = rng.choice(len(safe_pool), min(random_n, len(safe_pool)), replace=False)
    random_ids = set(safe_pool.iloc[positions]["sample_id"])
    supplied = challenge_ids or {}
    pool_ids = set(safe_pool["sample_id"])
    challenge_sets: dict[str, set[str]] = {}
    for score, tag in zip(SCORE_COLUMNS, CHALLENGE_TAGS, strict=True):
        selected = (
            set(map(str, supplied[tag]))
            if tag in supplied
            else _deterministic_top_ids(safe_pool, score, top_n)
        )
        if not selected <= pool_ids:
            raise ValueError(f"{tag} contains IDs outside the frozen pool")
        challenge_sets[tag] = selected

    rows: list[dict[str, object]] = []
    for row in safe_pool.itertuples(index=False):
        sample_id = str(row.sample_id)
        flags = {
            "random_sample": sample_id in random_ids,
            **{
                tag: include_challenge and sample_id in selected
                for tag, selected in challenge_sets.items()
            },
        }
        if any(flags.values()):
            tags = [tag for tag in ("random_sample", *CHALLENGE_TAGS) if flags[tag]]
            rows.append(
                {
                    "sample_id": sample_id,
                    "subset_type": "+".join(tags),
                    **flags,
                    **{column: float(getattr(row, column)) for column in SCORE_COLUMNS},
                }
            )
    return pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)


def summarize_utility_distribution(values: Sequence[float]) -> dict[str, float]:
    """Return the preregistered descriptive utility statistics."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("utility values must be a non-empty finite vector")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "IQR": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "P10": float(np.percentile(array, 10)),
        "P25": float(np.percentile(array, 25)),
        "P75": float(np.percentile(array, 75)),
        "P90": float(np.percentile(array, 90)),
        "fraction_utility_gt_0": float(np.mean(array > 0)),
    }


def bootstrap_spearman_ci(
    scores: Sequence[float],
    utility: Sequence[float],
    seed: int = 4502,
    bootstrap_replicates: int = 2000,
) -> dict[str, float | int]:
    """Return Spearman rho and a fixed-seed percentile bootstrap 95% CI."""
    score_array = np.asarray(scores, dtype=float)
    utility_array = np.asarray(utility, dtype=float)
    if score_array.shape != utility_array.shape or score_array.ndim != 1:
        raise ValueError("scores and utility must be aligned vectors")
    if len(score_array) < 3 or bootstrap_replicates < 1:
        raise ValueError("at least three rows and one bootstrap replicate are required")
    rho = float(spearmanr(score_array, utility_array).statistic)
    rng = np.random.default_rng(seed)
    bootstrapped: list[float] = []
    for _ in range(bootstrap_replicates):
        positions = rng.integers(0, len(score_array), len(score_array))
        statistic = float(
            spearmanr(score_array[positions], utility_array[positions]).statistic
        )
        if np.isfinite(statistic):
            bootstrapped.append(statistic)
    lower, upper = (
        np.percentile(bootstrapped, [2.5, 97.5])
        if bootstrapped
        else (float("nan"), float("nan"))
    )
    return {
        "rho": rho,
        "bootstrap_ci_lower": float(lower),
        "bootstrap_ci_upper": float(upper),
        "bootstrap_replicates_requested": int(bootstrap_replicates),
        "bootstrap_replicates_valid": len(bootstrapped),
        "bootstrap_seed": int(seed),
    }


def binary_auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Compute tie-aware binary AUROC."""
    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=bool)
    if score_array.shape != label_array.shape or score_array.ndim != 1:
        raise ValueError("scores and labels must be aligned vectors")
    positives = int(label_array.sum())
    negatives = int((~label_array).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(score_array, method="average")
    rank_sum = float(ranks[label_array].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def compute_enrichment(
    scores: Sequence[float], high_utility: Sequence[bool], fraction: float
) -> dict[str, float | int]:
    """Return hit rate and enrichment over the representative base rate."""
    score_array = np.asarray(scores, dtype=float)
    high_array = np.asarray(high_utility, dtype=bool)
    if score_array.shape != high_array.shape or score_array.ndim != 1:
        raise ValueError("scores and high_utility must be aligned vectors")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    count = max(1, int(np.ceil(len(score_array) * fraction)))
    selected = np.lexsort((np.arange(len(score_array)), -score_array))[:count]
    selected_rate = float(high_array[selected].mean())
    base_rate = float(high_array.mean())
    return {
        "fraction": float(fraction),
        "selected_count": count,
        "selected_high_count": int(high_array[selected].sum()),
        "selected_high_rate": selected_rate,
        "base_high_rate": base_rate,
        "enrichment": selected_rate / base_rate if base_rate else float("nan"),
    }


def source_reset_audit_rows(
    fit_records: Sequence[Mapping[str, object]],
    expected_source_hashes: Mapping[int, str],
    phase: str,
    candidate_id: str | None,
) -> list[dict[str, object]]:
    """Create the compact source-reset audit schema."""
    rows: list[dict[str, object]] = []
    for record in fit_records:
        member_seed = int(record["member_seed"])
        expected = expected_source_hashes[member_seed]
        observed = str(record["init_source_sha256"])
        rows.append(
            {
                "phase": phase,
                "candidate_id": candidate_id,
                "member_seed": member_seed,
                "init_source_sha256": observed,
                "expected_source_sha256": expected,
                "source_reset_pass": observed == expected,
                "checkpoint_sha256": str(record["checkpoint_sha256"]),
            }
        )
    return rows


def validate_candidate_fit_contract(
    initial_labeled: Sequence[str],
    fixed_validation: Sequence[str],
    candidate_labeled: Sequence[str],
    candidate_id: str,
    u0_ids: Sequence[str],
    test_ids: Sequence[str],
) -> dict[str, bool]:
    """Validate one-label reveal and fixed-validation isolation."""
    initial = list(map(str, initial_labeled))
    validation = list(map(str, fixed_validation))
    candidate_fit = list(map(str, candidate_labeled))
    candidate = str(candidate_id)
    checks = {
        "candidate_in_u0": candidate in set(map(str, u0_ids)),
        "candidate_not_test": candidate not in set(map(str, test_ids)),
        "candidate_not_validation": candidate not in set(validation),
        "fixed_validation_preserved": set(validation).issubset(candidate_fit),
        "exact_single_label_added": candidate_fit == initial + [candidate],
        "no_duplicate_training_identity": len(candidate_fit) == len(set(candidate_fit)),
    }
    if not all(checks.values()):
        raise ValueError(f"candidate fit contract failed: {checks}")
    return checks
