#!/usr/bin/env python3
"""Create non-retraining mechanism diagnostics from finalized E2 row aggregates."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "experiments" / "e2_4g_active_learning"
STRATEGIES = ("random", "coverage", "ensemble", "hybrid")


def paired_rows(per_seed: pd.DataFrame, left: str, right: str, column: str) -> dict:
    pivot = per_seed.pivot(index="seed", columns="strategy", values=column)
    difference = pivot[left] - pivot[right]
    mean = float(difference.mean())
    standard_deviation = float(difference.std(ddof=1))
    margin = 4.303 * standard_deviation / math.sqrt(len(difference))
    return {
        "comparison": f"{left}-{right}",
        "metric": column,
        "mean_paired_difference": mean,
        "standard_deviation": standard_deviation,
        "descriptive_ci_low": mean - margin,
        "descriptive_ci_high": mean + margin,
        "win_count": int((difference > 0).sum()),
        "seeds": len(difference),
        "differences_by_seed": json.dumps({str(key): float(value) for key, value in difference.items()}),
    }


def descriptive_spearman(frame: pd.DataFrame, left: str, right: str) -> float:
    selected = frame[[left, right]].dropna()
    if len(selected) < 3 or selected[left].nunique() < 2 or selected[right].nunique() < 2:
        return float("nan")
    return float(spearmanr(selected[left], selected[right]).statistic)


def main() -> None:
    queried = pd.read_csv(OUTPUT_DIR / "queried_batch_diagnostics.csv")
    metrics = pd.read_csv(OUTPUT_DIR / "round_metrics.csv")
    diversity = queried.groupby("strategy", as_index=False).agg(
        mean_pairwise_distance_mean=("batch_mean_pairwise_latent_distance", "mean"),
        mean_pairwise_distance_std=("batch_mean_pairwise_latent_distance", "std"),
        min_pairwise_distance_mean=("batch_min_pairwise_latent_distance", "mean"),
        unique_compounds_mean=("selected_unique_compounds", "mean"),
    )
    diversity.to_csv(OUTPUT_DIR / "batch_diversity_summary.csv", index=False)

    seed_diversity = queried.groupby(["seed", "strategy"], as_index=False).agg(
        mean_pairwise_distance_mean=("batch_mean_pairwise_latent_distance", "mean"),
        min_pairwise_distance_mean=("batch_min_pairwise_latent_distance", "mean"),
        unique_compounds_mean=("selected_unique_compounds", "mean"),
    )
    effects = []
    for left, right in (("hybrid", "ensemble"), ("coverage", "ensemble")):
        for column in (
            "mean_pairwise_distance_mean",
            "min_pairwise_distance_mean",
            "unique_compounds_mean",
        ):
            effects.append(paired_rows(seed_diversity, left, right, column))
    pd.DataFrame(effects).to_csv(OUTPUT_DIR / "batch_diversity_paired_effects.csv", index=False)

    gain = queried.copy()
    gain["round"] = gain["round"] - 1
    after = metrics[["outer_seed", "strategy", "round", "nrmse"]].rename(
        columns={"outer_seed": "seed", "round": "next_round", "nrmse": "nrmse_after"}
    )
    after["round"] = after["next_round"] - 1
    gain = gain.merge(after.drop(columns="next_round"), on=["seed", "strategy", "round"], how="inner")
    before = metrics[["outer_seed", "strategy", "round", "nrmse"]].rename(
        columns={"outer_seed": "seed", "nrmse": "nrmse_before"}
    )
    gain = gain.merge(before, on=["seed", "strategy", "round"], how="inner")
    gain["delta_nrmse"] = gain["nrmse_before"] - gain["nrmse_after"]
    gain = gain[
        [
            "seed", "strategy", "round", "selected_mean_true_error_after_reveal",
            "selected_mean_ensemble_uncertainty", "selected_mean_latent_distance",
            "batch_mean_pairwise_latent_distance", "nrmse_before", "nrmse_after", "delta_nrmse",
        ]
    ].rename(columns={"selected_mean_true_error_after_reveal": "selected_mean_true_error"})
    gain.to_csv(OUTPUT_DIR / "selected_error_vs_learning_gain.csv", index=False)
    correlations = []
    for strategy, selected in gain.groupby("strategy"):
        correlations.append(
            {
                "strategy": strategy,
                "n_seed_round_observations": len(selected),
                "spearman_selected_error_vs_delta_nrmse": descriptive_spearman(
                    selected, "selected_mean_true_error", "delta_nrmse"
                ),
                "spearman_batch_diversity_vs_delta_nrmse": descriptive_spearman(
                    selected, "batch_mean_pairwise_latent_distance", "delta_nrmse"
                ),
            }
        )
    correlation_frame = pd.DataFrame(correlations)
    correlation_frame.to_csv(OUTPUT_DIR / "selected_error_learning_gain_summary.csv", index=False)

    diversity_key = diversity.set_index("strategy")
    effect_key = pd.DataFrame(effects).set_index(["comparison", "metric"])
    hybrid_diversity = float(diversity_key.loc["hybrid", "mean_pairwise_distance_mean"])
    ensemble_diversity = float(diversity_key.loc["ensemble", "mean_pairwise_distance_mean"])
    mechanism = f"""# E2 Row Hybrid Mechanism Audit

## What The Finalized Row Pilot Shows

Hybrid has lower mean normalized AULC than Ensemble, and its mean within-batch pairwise latent distance is {hybrid_diversity:.3f} versus {ensemble_diversity:.3f} for Ensemble. The paired Hybrid–Ensemble diversity effect is reported in `batch_diversity_paired_effects.csv` using the three outer seeds as the statistical unit.

This co-occurrence is consistent with the hypothesis that uncertainty filtering followed by diversity selection reduces batch redundancy. It does **not** prove that diversity selection caused the AULC improvement: Hybrid and Ensemble selected sets differ in more than one property.

## Selected Error And Learning Gain

`selected_error_vs_learning_gain.csv` defines next-round improvement as `NRMSE_t - NRMSE_t+1`. Its Spearman summaries are descriptive only: rounds within an outer seed are not independent and the sample is small. The results therefore neither establish causal mediation nor justify a strict claim that prediction risk equals, or fails to equal, training utility.

## Required Causal Control (Not Run Here)

The direct future control is **uncertainty-filter-random**: retain the same Ensemble Top-25% candidate subset as Hybrid, select 25 samples uniformly at random within that subset, and compare it with farthest-first Hybrid. This isolates the contribution of diversity selection more cleanly. It is registered here only; no such active-learning curve was run.
"""
    (OUTPUT_DIR / "hybrid_mechanism_summary.md").write_text(mechanism, encoding="utf-8")


if __name__ == "__main__":
    main()
