"""Pure functions for the D46 oracle-utility reliability audit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..metrics import regression_metric_row

STRATA = ("high_positive", "near_zero", "strongly_negative")


def select_reliability_candidates(
    d45_candidates: pd.DataFrame, per_stratum: int = 6
) -> pd.DataFrame:
    """Select deterministic D45-truth strata from representative candidates."""
    required = {
        "sample_id", "random_sample", "oracle_utility_V1", "oracle_utility_V2",
        "oracle_utility_combined", "ensemble_score", "quantile_width",
        "coverage_score", "ensemble_top8", "qwidth_top8", "coverage_top8",
    }
    missing = required - set(d45_candidates.columns)
    if missing:
        raise ValueError(f"D45 candidate table is missing: {sorted(missing)}")
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    representative = d45_candidates[d45_candidates["random_sample"].astype(bool)].copy()
    representative["sample_id"] = representative["sample_id"].astype(str)
    if representative["sample_id"].duplicated().any() or len(representative) < 3 * per_stratum:
        raise ValueError("representative candidates must be unique and sufficiently numerous")
    utility = "oracle_utility_combined"
    high = representative.sort_values(
        [utility, "sample_id"], ascending=[False, True], kind="mergesort"
    ).head(per_stratum)
    used = set(high["sample_id"])
    negative = representative[~representative["sample_id"].isin(used)].sort_values(
        [utility, "sample_id"], ascending=[True, True], kind="mergesort"
    ).head(per_stratum)
    used.update(negative["sample_id"])
    remaining = representative[~representative["sample_id"].isin(used)].assign(
        absolute_utility=lambda frame: frame[utility].abs()
    )
    near = remaining.sort_values(
        ["absolute_utility", "sample_id"], ascending=[True, True], kind="mergesort"
    ).head(per_stratum)
    if min(len(high), len(negative), len(near)) != per_stratum:
        raise ValueError("unable to construct disjoint reliability strata")
    selected = pd.concat(
        [high.assign(stratum="high_positive"),
         near.assign(stratum="near_zero"),
         negative.assign(stratum="strongly_negative")], ignore_index=True,
    )
    selected = selected.rename(columns={
        "oracle_utility_V1": "D45_oracle_utility_V1",
        "oracle_utility_V2": "D45_oracle_utility_V2",
        "oracle_utility_combined": "D45_oracle_utility_combined",
    })
    columns = [
        "sample_id", "stratum", "D45_oracle_utility_V1",
        "D45_oracle_utility_V2", "D45_oracle_utility_combined",
        "ensemble_score", "quantile_width", "coverage_score", "random_sample",
        "ensemble_top8", "qwidth_top8", "coverage_top8",
    ]
    result = selected[columns].copy()
    if result["sample_id"].duplicated().any():
        raise AssertionError("reliability selection produced duplicate candidates")
    return result.reset_index(drop=True)


def paired_utility(
    baseline_metrics: Mapping[str, float], candidate_metrics: Mapping[str, float]
) -> float:
    """Return within-repetition baseline minus candidate combined NRMSE."""
    return float(baseline_metrics["NRMSE"] - candidate_metrics["NRMSE"])


def summarize_candidate_reliability(
    utilities: pd.DataFrame,
    repetition_seeds: Sequence[int],
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """Summarize candidate utilities, sign counts, and sign consistency."""
    required = {"sample_id", "stratum", "repetition_seed", "oracle_utility_combined"}
    if missing := required - set(utilities.columns):
        raise ValueError(f"utility table is missing: {sorted(missing)}")
    seeds = list(map(int, repetition_seeds))
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    rows: list[dict[str, object]] = []
    for (sample_id, stratum), group in utilities.groupby(["sample_id", "stratum"], sort=False):
        indexed = group.set_index("repetition_seed")
        if set(indexed.index.astype(int)) != set(seeds) or len(indexed) != len(seeds):
            raise ValueError(f"candidate {sample_id} lacks exact repetitions")
        values = np.array([indexed.loc[seed, "oracle_utility_combined"] for seed in seeds], dtype=float)
        positive = int(np.sum(values > epsilon))
        negative = int(np.sum(values < -epsilon))
        zero = len(values) - positive - negative
        row: dict[str, object] = {"sample_id": str(sample_id), "stratum": str(stratum)}
        row.update({f"utility_rep{index + 1}": float(value) for index, value in enumerate(values)})
        row.update({
            "mean_utility": float(values.mean()),
            "std_utility": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min_utility": float(values.min()), "max_utility": float(values.max()),
            "positive_repetitions": positive, "negative_repetitions": negative,
            "near_zero_repetitions": zero,
            "sign_consistency": max(positive, negative, zero) / len(values),
            "epsilon": epsilon,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_ranking_stability(
    utilities: pd.DataFrame, repetition_seeds: Sequence[int]
) -> pd.DataFrame:
    """Compute every requested pairwise Spearman correlation by scope."""
    scopes = {
        "all_18": utilities,
        "high_plus_negative_12": utilities[utilities["stratum"] != "near_zero"],
        "representative_only": utilities[utilities.get("random_sample", True).astype(bool)]
        if "random_sample" in utilities else utilities,
    }
    rows: list[dict[str, object]] = []
    for scope, frame in scopes.items():
        pivot = frame.pivot(index="sample_id", columns="repetition_seed", values="oracle_utility_combined")
        for seed_a, seed_b in combinations(map(int, repetition_seeds), 2):
            aligned = pivot[[seed_a, seed_b]].dropna()
            rows.append({
                "scope": scope, "repetition_seed_A": seed_a,
                "repetition_seed_B": seed_b, "n": len(aligned),
                "spearman": float(spearmanr(aligned[seed_a], aligned[seed_b]).statistic),
            })
    return pd.DataFrame(rows)


def variance_decomposition(
    utilities: pd.DataFrame,
    candidate_column: str = "sample_id",
    value_column: str = "oracle_utility_combined",
) -> dict[str, float | int | str]:
    """Estimate balanced one-way random-effects variance and ICC(1,1)."""
    pivot = utilities.pivot(index=candidate_column, columns="repetition_seed", values=value_column)
    if pivot.isna().any().any() or pivot.shape[0] < 2 or pivot.shape[1] < 2:
        raise ValueError("variance decomposition requires a complete balanced panel")
    values = pivot.to_numpy(dtype=float)
    candidates, repetitions = values.shape
    grand_mean = float(values.mean())
    candidate_means = values.mean(axis=1)
    ms_between = float(repetitions * np.square(candidate_means - grand_mean).sum() / (candidates - 1))
    ms_within = float(np.square(values - candidate_means[:, None]).sum() / (candidates * (repetitions - 1)))
    between = max((ms_between - ms_within) / repetitions, 0.0)
    within = ms_within
    denominator = between + within
    reliability = between / denominator if denominator else float("nan")
    return {
        "estimator": "balanced one-way random effects; sigma_between=max((MSB-MSW)/k,0), sigma_within=MSW",
        "candidate_count": candidates, "repetition_count": repetitions,
        "grand_mean": grand_mean, "MS_between": ms_between, "MS_within": ms_within,
        "between_candidate_variance": between, "within_candidate_variance": within,
        "reliability_ratio": reliability, "icc": reliability,
    }


def paired_test_bootstrap(
    truth: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    target_scales: Mapping[str, float],
    bootstrap_replicates: int = 2000,
    seed: int = 4604,
    sampled_indices: np.ndarray | None = None,
    return_draws: bool = False,
) -> dict[str, object]:
    """Paired-row bootstrap of baseline-minus-candidate combined NRMSE."""
    truth_array = np.asarray(truth, dtype=float)
    baseline_array = np.asarray(baseline_prediction, dtype=float)
    candidate_array = np.asarray(candidate_prediction, dtype=float)
    if truth_array.shape != baseline_array.shape or truth_array.shape != candidate_array.shape:
        raise ValueError("truth and paired predictions must align")
    if sampled_indices is None:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(truth_array), size=(bootstrap_replicates, len(truth_array)))
    else:
        indices = np.asarray(sampled_indices, dtype=int)
        if indices.ndim != 2 or indices.shape[1] != len(truth_array):
            raise ValueError("sampled_indices must have shape (B, test_rows)")
    draws = np.empty(len(indices), dtype=float)
    for position, row_indices in enumerate(indices):
        baseline = regression_metric_row(truth_array[row_indices], baseline_array[row_indices], target_scales)
        candidate = regression_metric_row(truth_array[row_indices], candidate_array[row_indices], target_scales)
        draws[position] = paired_utility(baseline, candidate)
    lower, upper = np.percentile(draws, [2.5, 97.5])
    result: dict[str, object] = {
        "utility_bootstrap_mean": float(draws.mean()),
        "utility_bootstrap_ci_lower": float(lower),
        "utility_bootstrap_ci_upper": float(upper),
        "bootstrap_replicates": len(draws), "bootstrap_seed": int(seed),
        "paired_row_resampling": True,
    }
    if return_draws:
        result["draws"] = draws
        result["sampled_indices"] = indices
    return result


def compare_d45_d46(reliability: pd.DataFrame, candidate_subset: pd.DataFrame) -> pd.DataFrame:
    """Return candidate rows plus D45-versus-D46 summary statistics."""
    merged = candidate_subset[["sample_id", "stratum", "D45_oracle_utility_combined"]].merge(
        reliability[["sample_id", "mean_utility"]], on="sample_id", validate="one_to_one"
    )
    rho = float(spearmanr(merged["D45_oracle_utility_combined"], merged["mean_utility"]).statistic)
    difference = merged["D45_oracle_utility_combined"] - merged["mean_utility"]
    merged["absolute_difference"] = difference.abs()
    merged["sign_agreement"] = (
        np.sign(merged["D45_oracle_utility_combined"]) == np.sign(merged["mean_utility"])
    )
    merged["D45_D46_spearman"] = rho
    merged["D45_D46_MAE"] = float(merged["absolute_difference"].mean())
    merged["D45_D46_sign_agreement"] = float(merged["sign_agreement"].mean())
    return merged


def summarize_strata(
    reliability: pd.DataFrame, bootstrap: pd.DataFrame
) -> pd.DataFrame:
    """Summarize optimization and test-row reliability by D45 stratum."""
    pair_summary = bootstrap.groupby("stratum").agg(
        bootstrap_CI_excludes_zero_fraction=("bootstrap_CI_excludes_zero", "mean")
    )
    result = reliability.groupby("stratum").agg(
        candidate_count=("sample_id", "size"),
        mean_mean_utility=("mean_utility", "mean"),
        mean_std_utility=("std_utility", "mean"),
        mean_sign_consistency=("sign_consistency", "mean"),
    ).join(pair_summary).reset_index()
    return result
