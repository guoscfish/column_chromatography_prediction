"""Post-freeze metric aggregation and preregistered decision logic."""

import numpy as np
import pandas as pd

from ..evaluation.point import point_metrics
from ..training.predictor import atomic_json


def metric_row(truth, prediction, scales):
    values = point_metrics(truth, prediction, scales)
    return {**{f"{target}_{metric}": values[f"{target}_{metric.lower()}"]
               for target in ("V1", "V2") for metric in ("RMSE", "MAE", "R2")},
            "combined_normalized_RMSE": values["combined_normalized_rmse"]}


def paired_summary(arms, baseline):
    records = []
    for seed, rows in arms.groupby("outer_seed", sort=False):
        base = float(baseline.loc[baseline.outer_seed.eq(seed), "combined_normalized_RMSE"].iloc[0])
        random = rows.loc[rows.arm.str.startswith("random_control_")]
        mean, median = random.combined_normalized_RMSE.mean(), random.combined_normalized_RMSE.median()
        for _, row in rows.iterrows():
            records.append({**row.to_dict(), "baseline_NRMSE": base,
                            "gain": base - row.combined_normalized_RMSE,
                            "method_gain_minus_random_mean_gain": mean - row.combined_normalized_RMSE,
                            "method_gain_minus_random_median_gain": median - row.combined_normalized_RMSE,
                            "beats_random_median": bool(row.combined_normalized_RMSE < median)})
    return pd.DataFrame(records)


def decision_gate(arms, expected_seeds):
    required = {"lcmd", "uncertainty", "coreset", *(f"random_control_{i}" for i in range(5))}
    if set(arms.outer_seed) != set(expected_seeds) or len(expected_seeds) != 5:
        raise ValueError("decision requires the complete preregistered five-seed cohort")
    if arms.duplicated(["outer_seed", "arm"]).any():
        raise ValueError("duplicate seed/arm metrics")
    if any(set(rows.arm) != required for _, rows in arms.groupby("outer_seed")):
        raise ValueError("incomplete method/control matrix")
    metrics = ["combined_normalized_RMSE", "V1_RMSE", "V2_RMSE"]
    if not np.isfinite(arms[metrics].to_numpy()).all():
        raise ValueError("non-finite decision metrics")
    grouped = arms.assign(method=arms.arm.where(~arms.arm.str.startswith("random_control_"), "random"))
    per_seed = grouped.groupby(["outer_seed", "method"])[metrics].mean().reset_index()
    means = per_seed.groupby("method")[metrics].mean()
    pivot = per_seed.pivot(index="outer_seed", columns="method", values=metrics[0])
    random_median = grouped.loc[grouped.method.eq("random")].groupby("outer_seed")[metrics[0]].median()
    wins = {"random_median": int((pivot.lcmd < random_median).sum()),
            "uncertainty": int((pivot.lcmd < pivot.uncertainty).sum()),
            "coreset": int((pivot.lcmd < pivot.coreset).sum())}
    random_gate = bool(means.loc["lcmd", metrics[0]] < means.loc["random", metrics[0]] and wins["random_median"] >= 4)
    endpoint_change = {target: float(means.loc["lcmd", f"{target}_RMSE"] /
                                    means.drop(index="lcmd")[f"{target}_RMSE"].min() - 1)
                       for target in ("V1", "V2")}
    conditions = {"beats_random_mean_and_4_of_5_medians": random_gate,
                  "beats_uncertainty_mean_and_3_of_5": bool(means.loc["lcmd", metrics[0]] < means.loc["uncertainty", metrics[0]] and wins["uncertainty"] >= 3),
                  "beats_coreset_mean_and_3_of_5": bool(means.loc["lcmd", metrics[0]] < means.loc["coreset", metrics[0]] and wins["coreset"] >= 3),
                  "endpoint_deterioration_at_most_2_percent": bool(all(
                      means.loc["lcmd", f"{target}_RMSE"] <= 1.02 * means.drop(index="lcmd")[f"{target}_RMSE"].min()
                      for target in ("V1", "V2")))}
    decision = ("BEST_CURRENT_ROW_ACQUISITION" if all(conditions.values()) else
                "LCMD_BEATS_RANDOM_NOT_OTHER_BASELINES" if random_gate else "LARGE_BATCH_ONLY_SIGNAL")
    return {"decision": decision, "conditions": conditions, "directional_wins": wins,
            "endpoint_fractional_change_vs_best_competitor": endpoint_change,
            "method_means": means.to_dict("index"), "sequential_recommendation_eligible": random_gate,
            "random_comparisons_are_not_independent_outer_seeds": True}


def write_primary_tables(results, metrics, cohorts, smoke=False):
    results.mkdir(parents=True, exist_ok=True)
    baseline = metrics.loc[metrics.arm.eq("baseline_l0")].copy()
    arms = metrics.loc[~metrics.arm.eq("baseline_l0")].copy()
    baseline.to_csv(results / "baseline_metrics.csv", index=False)
    arms.to_csv(results / "arm_metrics_b32.csv", index=False)
    summary = paired_summary(arms, baseline)
    summary.to_csv(results / "seed_summary_b32.csv", index=False)
    comparisons = []
    for seed, rows in arms.groupby("outer_seed"):
        for method in ("lcmd", "uncertainty", "coreset"):
            value = float(rows.loc[rows.arm.eq(method), "combined_normalized_RMSE"].iloc[0])
            for competitor in ("random_mean", "random_median", "lcmd", "uncertainty", "coreset"):
                if competitor == method:
                    continue
                other = rows.loc[rows.arm.str.startswith("random_control_"), "combined_normalized_RMSE"]
                reference = (other.mean() if competitor == "random_mean" else other.median()) if competitor.startswith("random_") else float(rows.loc[rows.arm.eq(competitor), "combined_normalized_RMSE"].iloc[0])
                comparisons.append({"outer_seed": seed, "method": method, "competitor": competitor,
                                    "NRMSE_advantage": reference - value, "directional_win": value < reference})
    pd.DataFrame(comparisons).to_csv(results / "pairwise_method_comparison.csv", index=False)
    if not smoke:
        decisions = {name: decision_gate(arms.loc[arms.outer_seed.isin(seeds)], seeds) for name, seeds in cohorts.items()}
        atomic_json(results / "cohort_decisions.json", decisions)
        return decisions
    return None


def plot_primary(results, figures):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = pd.read_csv(results / "arm_metrics_b32.csv")
    arms["method"] = arms.arm.where(~arms.arm.str.startswith("random_control_"), "random")
    figures.mkdir(parents=True, exist_ok=True)
    for metric, name in (("combined_normalized_RMSE", "b32_method_comparison"), ("V1_RMSE", "endpoint_v1_rmse"), ("V2_RMSE", "endpoint_v2_rmse")):
        fig, ax = plt.subplots(figsize=(7, 4))
        for seed, rows in arms.groupby("outer_seed"):
            values = rows.groupby("method")[metric].mean().reindex(["random", "uncertainty", "coreset", "lcmd"])
            ax.plot(values.index, values, marker="o", alpha=0.55, label=str(seed))
        ax.set_ylabel(metric)
        ax.set_title("B=32 matched row acquisition")
        fig.tight_layout()
        fig.savefig(figures / f"{name}.png", dpi=160)
        plt.close(fig)
