#!/usr/bin/env python3
"""Write the Phase 1 manuscript from completed, frozen result tables only."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qgeognn_adaptivity_matplotlib")
ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies/active_learning/qgeognn_v2_batch_adaptivity"


def show(value, digits=6):
    return f"{float(value):.{digits}f}" if pd.notna(value) else "censored / unavailable"


def markdown(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(show(x) if isinstance(x, (float, np.floating)) else str(x) for x in row) + " |")
    return "\n".join(lines)


def main():
    if not (STUDY / "global_pre_test_freeze.json").is_file():
        raise RuntimeError("global freeze required")
    decision = json.loads((STUDY / "decision.json").read_text())
    result = STUDY / "results"
    summary = pd.read_csv(result / "metric_summary.csv")
    per_seed = pd.read_csv(result / "per_seed_metrics.csv")
    paired = pd.read_csv(result / "paired_effects.csv")
    costs = pd.read_csv(result / "compute_cost.csv")
    targets = pd.read_csv(result / "labels_to_target.csv")
    saving = pd.read_csv(result / "label_saving.csv")
    overlap = pd.read_csv(result / "selection_overlap.csv")
    metrics = ["combined_normalized_RMSE", "V1_RMSE", "V2_RMSE", "V1_R2", "V2_R2"]
    mean = summary.pivot(index="method", columns="metric", values="endpoint_mean")[metrics].reset_index()
    one_shot_mean = mean.loc[mean.method.eq("static_lcmd")].copy()
    one_shot_mean["method"] = "oneshot_b320 (same final model)"
    mean = pd.concat([mean, one_shot_mean], ignore_index=True)
    areas = summary.pivot(index="method", columns="metric", values="aulc_mean")[metrics].reset_index()
    effect = paired.loc[(paired.method.eq("adaptive_lcmd")) & paired.comparator.eq("static_lcmd") & paired.metric.isin(metrics)]
    final_seed = per_seed.loc[per_seed.metric.eq("combined_normalized_RMSE")].pivot(index="outer_seed", columns="method", values="endpoint")
    final_seed["adaptive_minus_static"] = final_seed.adaptive_lcmd - final_seed.static_lcmd
    target_view = targets.loc[targets.scope.eq("cohort_mean") & targets.metric.isin(metrics),
                             ["method", "metric", "target", "interpolated_labels", "first_observed_labels", "sustained_labels", "status"]]
    cost_summary = costs.groupby("method")[["fit_count", "epochs", "training_seconds", "gradient_seconds", "new_fit_count", "new_training_seconds"]].sum().reset_index()
    complete = pd.read_csv(result / "complete_results.csv")
    one_shot = complete.loc[complete.method.eq("static_lcmd")].copy()
    one_shot["method"] = "oneshot_b320"
    for column in one_shot:
        if "AULC" in column or any(target in column for target in ("N80", "N90", "N95")) or "saving" in column:
            one_shot[column] = "NOT_EVALUATED" if column.endswith("status") else np.nan
    one_shot_cost = cost_summary.set_index("method").loc["oneshot_b320"]
    for column in ("fit_count", "epochs", "training_seconds", "gradient_seconds", "new_fit_count"):
        one_shot[column] = one_shot_cost[column]
    complete = pd.concat([complete, one_shot], ignore_index=True)
    complete["shared_endpoint_model"] = complete.method.isin(["static_lcmd", "oneshot_b320"])
    complete.to_csv(result / "complete_results_with_oneshot.csv", index=False)
    positive = decision["feedback_signal"]
    interpretation = (
        "The preregistered endpoint conditions support a feedback benefit in this cohort."
        if positive else
        "The preregistered endpoint conditions do not establish a consistent feedback benefit in this cohort.")
    curve_interpretation = (
        "The additional curve-efficiency conditions also pass."
        if decision["curve_efficiency_signal"] else
        "The combined endpoint-plus-curve conditions do not pass; inspect each component below.")
    lines = ["# Phase 1: matched-total-budget batch/adaptivity results", "",
             interpretation, curve_interpretation, "",
             "These are iterative development results on five previously evaluated row splits, not independent confirmation. "
             "All methods use 333 initial + 320 acquired = 653 active labels and the same 416 validation labels. "
             "No weighted overall score is used.", "", "## Prediction quality", "", markdown(mean), "",
             "OneShot-B320 and Static-B320-prefix share exactly the same 653-label model. "
             "Static intermediate fits are scratch fits and never influence acquisition. "
             "Their equality is by construction, not a second independent experimental replicate.", "",
             "### Per-seed endpoint NRMSE", "", markdown(final_seed.reset_index()), "",
             "### Paired Adaptive minus Static", "",
             "For endpoint and AULC, negative differences favor Adaptive for error metrics and positive differences favor it for R2. "
             "For gap_closed, positive differences favor Adaptive for every metric. "
             "Intervals are descriptive 95% paired bootstrap intervals over five overlapping splits.", "",
             markdown(effect[["metric", "statistic", "directional_wins", "n_pairs", "mean_difference", "median_difference", "sample_std_difference", "bootstrap_low", "bootstrap_high"]]),
             "", "## Label efficiency", "", "AULC is the trapezoidal integral divided by 653-333. "
             "This interval differs from Phase 0's 333..1005 interval, so those AULCs must not be ranked directly.", "",
             markdown(areas), "", "### Targets", "",
             "The following targets are computed on cohort-mean curves. Per-seed targets remain in the CSV. "
             "First crossing is not persistence. Sustained means every remaining observed point meets the threshold; "
             "a final-point crossing alone has no future validation. Censored values are not imputed or extrapolated.", "",
             markdown(target_view), "", "### Saving at Random@653", "",
             markdown(saving.loc[saving.scope.eq("cohort_mean") & saving.metric.isin(metrics) &
                                saving.target.eq("Random@653") & saving.crossing.eq("first_observed_labels"),
                                ["method", "metric", "random_labels", "method_labels", "saved_labels", "incremental_saving", "status"]]),
             "", "## Robustness and cost", "", markdown(cost_summary), "",
             "Training seconds include validation and final prediction. These are summed task times, not calendar time. "
             "Logical OneShot effort includes L0 plus the final model; its final model is already charged to Static "
             "in newly incurred work, so logical arm costs must not be added together. Historical machine loads differ. "
             "Full-data references are shared evaluation overhead, excluded from method operating costs. "
             "Interrupted-attempt compute was not fully measured; see EXECUTION_NOTES.md. "
             "The successful invocation's elapsed time is in execution_time.json, not a full wall-clock total.", "",
             "### Frozen gate components", "", "```json", json.dumps(decision, indent=2), "```", "",
             "## Selection mechanism", "",
             markdown(overlap.loc[overlap.active_label_count.eq(653)]), "",
             "The first 32 selected IDs match exactly for all five seeds. Later final membership differs. "
             "Because every evaluated model starts from the same initialization and uses the same fitting rule, "
             "the contrast measures the effect of the resulting acquisition trajectory and ordered training set. "
             "It is not an effect of warm-start or additional optimization applied to the final model. "
             "Training order is part of the fixed protocol; this experiment does not separately randomize set membership and order.", "",
             "## Interpretation and next action", "",
             "This control isolates fixed initial LCMD ranking versus B32 feedback at a matched final budget. "
             "It does not establish a monotone relationship across batch sizes; B64x5 and B160x2 were not run. "
             "It does not distinguish performance on new compounds or independent future experiments.", "",
             "The next acquisition implementation remains the separately specified matched B32 Gradient-MaxDet screen. "
             "Its advancement must depend on actual paired prediction and endpoint results. "
             "Do not automatically promote noise weighting, IVR variants, manual strategy switching or dynamic batches. "
             "See the Phase 2 implementation plan for prerequisites and stopping gates.", "",
             "Artifacts: `results/complete_results_with_oneshot.csv`, all per-seed tables, "
             "`figures/learning_curves.png`, and `figures/paired_endpoints.png`.", ""]
    (STUDY / "FINAL_REPORT.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, ("combined_normalized_RMSE", "V1_R2", "V2_R2")):
        data = per_seed.loc[per_seed.metric.eq(metric)].pivot(index="outer_seed", columns="method", values="endpoint")
        for seed, values in data.iterrows():
            ax.plot([0, 1], [values.static_lcmd, values.adaptive_lcmd], marker="o", label=str(seed))
        ax.set_xticks([0, 1], ["Static / OneShot", "Adaptive B32"])
        ax.set_title(metric)
        ax.grid(axis="y", alpha=.2)
    axes[-1].legend(title="Outer seed", fontsize=8)
    fig.suptitle("Matched 653-label endpoints")
    fig.tight_layout()
    fig.savefig(STUDY / "figures/paired_endpoints.png", dpi=180)
    plt.close(fig)
    print(STUDY / "FINAL_REPORT.md")


if __name__ == "__main__":
    main()
