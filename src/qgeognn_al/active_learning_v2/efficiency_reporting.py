"""Additive, endpoint-aware evaluation of already frozen AL trajectories.

This module does not read labels or change historical study decisions.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import sha256_file
from ..resources import ROOT
from ..training.predictor import atomic_json
from .sequential_protocol import CONFIRMATION_SEEDS, STUDY as BASELINE


METRICS = ("combined_normalized_RMSE", "V1_RMSE", "V2_RMSE", "V1_MAE", "V2_MAE", "V1_R2", "V2_R2")
IVR = ROOT / "studies/active_learning/qgeognn_v2_row_kernel_ivr_b32"
STUDY = ROOT / "studies/active_learning/qgeognn_v2_efficiency_review"


def read_json(path):
    return json.loads(Path(path).read_text())


def direction(metric):
    return 1 if metric.endswith("R2") else -1


def crossings(labels, values, target, higher=False):
    """Report first crossing and the first observed budget that stays reached."""
    x, y = np.asarray(labels, float), np.asarray(values, float)
    if (x.ndim != 1 or y.shape != x.shape or not len(x) or
            not np.isfinite(x).all() or not np.isfinite(y).all() or
            not np.isfinite(target) or not np.all(np.diff(x) > 0)):
        raise ValueError("finite, aligned, strictly increasing curve required")
    reached = y >= target if higher else y <= target
    hit = np.flatnonzero(reached)
    result = {"interpolated_labels": np.nan, "first_observed_labels": np.nan,
              "sustained_labels": np.nan, "status": "CENSORED", "censor_budget": float(x[-1])}
    if not len(hit):
        return result
    index = int(hit[0])
    interpolated = x[index] if index == 0 else (
        x[index - 1] + (target - y[index - 1]) / (y[index] - y[index - 1]) * (x[index] - x[index - 1]))
    sustained = [i for i in hit if reached[i:].all()]
    result.update(interpolated_labels=float(interpolated), first_observed_labels=float(x[index]),
                  sustained_labels=float(x[sustained[0]]) if sustained else np.nan, status="REACHED")
    return result


def validate_curves(curves, full):
    if curves.duplicated(["outer_seed", "method", "active_label_count"]).any():
        raise ValueError("duplicate curve budget")
    if full.duplicated("outer_seed").any() or set(full.outer_seed) != set(curves.outer_seed):
        raise ValueError("one matched full reference per seed required")
    if not np.isfinite(curves[list(METRICS)].to_numpy(float)).all() or not np.isfinite(full[list(METRICS)].to_numpy(float)).all():
        raise ValueError("nonfinite metrics")
    methods, seeds = set(curves.method), set(curves.outer_seed)
    if "random" not in methods:
        raise ValueError("nested random control required")
    expected = sorted(curves.active_label_count.unique())
    if len(expected) < 2:
        raise ValueError("at least two budgets required")
    for seed in seeds:
        rows = curves.loc[curves.outer_seed.eq(seed)]
        if set(rows.method) != methods:
            raise ValueError("incomplete seed/method matrix")
        for _, values in rows.groupby("method"):
            if sorted(values.active_label_count) != expected:
                raise ValueError("matched complete budget schedules required")
        initial = rows.loc[rows.active_label_count.eq(expected[0]), list(METRICS)].to_numpy(float)
        if not np.allclose(initial, initial[0], rtol=0, atol=1e-12):
            raise ValueError("initial model metrics differ across methods")


def evaluate(curves, full):
    """Return tidy per-seed metrics, targets and uncapped gap closure curves."""
    validate_curves(curves, full)
    summaries, gaps, targets = [], [], []
    mean = curves.groupby(["method", "active_label_count"])[list(METRICS)].mean().reset_index()
    scopes = [("seed", int(seed), rows, full.loc[full.outer_seed.eq(seed)].iloc[0])
              for seed, rows in curves.groupby("outer_seed")]
    scopes.append(("cohort_mean", None, mean, full[list(METRICS)].mean()))
    for scope, seed, rows, reference in scopes:
        random = rows.loc[rows.method.eq("random")].sort_values("active_label_count")
        start, end = random.active_label_count.iloc[[0, -1]].to_numpy(float)
        for metric in METRICS:
            initial, final_random = random[metric].iloc[[0, -1]].to_numpy(float)
            full_value = float(reference[metric])
            sign = direction(metric)
            denominator = sign * (full_value - initial)
            valid_gap = denominator > 1e-12
            thresholds = {f"Random@{int(end)}": final_random}
            if valid_gap:
                thresholds.update({f"N{int(100*fraction)}": initial + fraction * (full_value - initial)
                                   for fraction in (.8, .9, .95)})
            for method, values in rows.groupby("method"):
                values = values.sort_values("active_label_count")
                x, y = values.active_label_count.to_numpy(float), values[metric].to_numpy(float)
                gap = sign * (y - initial) / denominator if valid_gap else np.full(len(y), np.nan)
                if scope == "seed":
                    summaries.append({"outer_seed": seed, "method": method, "metric": metric,
                                      "endpoint": float(y[-1]), "aulc": float(np.trapezoid(y, x) / (end-start)),
                                      "gap_closed": float(gap[-1]), "full_reference": full_value,
                                      "gap_status": "VALID" if valid_gap else "NONPOSITIVE_FULL_GAP"})
                gaps.extend({"scope": scope, "outer_seed": seed, "method": method, "metric": metric,
                             "active_label_count": int(n), "gap_closed": float(g),
                             "gap_status": "VALID" if valid_gap else "NONPOSITIVE_FULL_GAP"}
                            for n, g in zip(x, gap))
                for name in (f"Random@{int(end)}", "N80", "N90", "N95"):
                    crossing = crossings(x, y, thresholds[name], sign == 1) if name in thresholds else {
                        "interpolated_labels": np.nan, "first_observed_labels": np.nan,
                        "sustained_labels": np.nan, "status": "NONPOSITIVE_FULL_GAP", "censor_budget": end}
                    targets.append({"scope": scope, "outer_seed": seed, "method": method, "metric": metric,
                                    "target": name, "threshold": thresholds.get(name, np.nan), **crossing})
    target_frame = pd.DataFrame(targets)
    savings = []
    for _, rows in target_frame.groupby(["scope", "outer_seed", "metric", "target"], dropna=False):
        control = rows.loc[rows.method.eq("random")].iloc[0]
        for _, row in rows.iterrows():
            for crossing in ("interpolated_labels", "first_observed_labels", "sustained_labels"):
                baseline, candidate = control[crossing], row[crossing]
                reached = pd.notna(baseline) and pd.notna(candidate)
                saved = baseline - candidate if reached else np.nan
                increment = baseline - curves.active_label_count.min()
                savings.append({**row[["scope", "outer_seed", "method", "metric", "target"]].to_dict(),
                                "crossing": crossing, "random_labels": baseline, "method_labels": candidate,
                                "saved_labels": saved,
                                "incremental_saving": saved / increment if reached and increment > 0 else np.nan,
                                "status": "BOTH_REACHED" if reached else "CENSORED_COMPARISON"})
    per_seed = pd.DataFrame(summaries)
    paired = []
    for (metric, statistic), rows in per_seed.melt(
            id_vars=["outer_seed", "method", "metric"], value_vars=["endpoint", "aulc", "gap_closed"],
            var_name="statistic").groupby(["metric", "statistic"]):
        pivot = rows.pivot(index="outer_seed", columns="method", values="value")
        for method, comparator in itertools.combinations(sorted(pivot.columns), 2):
            # Orient every pair so active vs Random and IVR vs LCMD read naturally.
            if method == "random" or (method == "lcmd" and comparator != "random"):
                method, comparator = comparator, method
            delta = (pivot[method] - pivot[comparator]).to_numpy(float)
            finite = np.isfinite(delta)
            delta = delta[finite]
            sign = 1 if statistic == "gap_closed" else direction(metric)
            bootstrap = np.random.default_rng(20260918).choice(delta, (10000, len(delta))).mean(axis=1) if len(delta) else np.array([np.nan])
            paired.append({"method": method, "comparator": comparator, "metric": metric,
                           "statistic": statistic, "n_pairs": len(delta),
                           "directional_wins": int(np.sum(sign*delta > 1e-12)),
                           "ties": int(np.sum(np.abs(delta) <= 1e-12)),
                           "mean_difference": float(delta.mean()) if len(delta) else np.nan,
                           "median_difference": float(np.median(delta)) if len(delta) else np.nan,
                           "sample_std_difference": float(delta.std(ddof=1)) if len(delta)>1 else np.nan,
                           "bootstrap_low": float(np.quantile(bootstrap, .025)),
                           "bootstrap_high": float(np.quantile(bootstrap, .975))})
    aggregate = per_seed.groupby(["method", "metric"])[["endpoint", "aulc", "gap_closed"]].agg(["mean", "median", "std"])
    aggregate.columns = ["_".join(parts) for parts in aggregate.columns]
    return {"learning_curve_metrics": curves, "cohort_mean_learning_curve": mean,
            "per_seed_metrics": per_seed, "metric_summary": aggregate.reset_index(),
            "gap_closure": pd.DataFrame(gaps), "labels_to_target": target_frame,
            "label_saving": pd.DataFrame(savings), "paired_effects": pd.DataFrame(paired)}


def historical_costs(final_round=21, include_ivr=True):
    records = []
    for seed in CONFIRMATION_SEEDS:
        for method in ("random", "hybrid", "lcmd"):
            runtime = BASELINE / "runtime" / f"seed_{seed}" / method
            fits = pd.read_csv(runtime / "fit_audit.csv")
            fits = fits.loc[(fits["round"] <= final_round) & ((fits.member == 0) | (fits["round"] < final_round))]
            gradient_s = sum(read_json(runtime / f"round_{r:02d}/acquisition_artifacts/gradient_audit.json")["elapsed_seconds"]
                             for r in range(final_round)) if method == "lcmd" else 0.
            round_elapsed = sum(read_json(runtime / f"round_{r:02d}/contract.json")["elapsed_seconds"] for r in range(final_round + 1))
            records.append({"outer_seed": seed, "method": method, "fit_count": len(fits),
                            "extra_ensemble_fits": int((fits.member > 0).sum()),
                            "epochs": int(fits.epochs_run.sum()),
                            "optimizer_steps": int((fits.epochs_run * np.ceil(fits.train_rows / 2048)).sum()),
                            "training_seconds": float(fits.training_seconds.sum()), "gradient_seconds": gradient_s,
                            "gradient_extractions": final_round if method == "lcmd" else 0,
                            "ensemble_inference_seconds": np.nan, "selector_seconds": np.nan,
                            "historical_round_elapsed_seconds": round_elapsed,
                            "new_fit_count": 0, "new_training_seconds": 0.,
                            "cost_note": "fit time includes validation/prediction; ensemble inference not separately timed; round elapsed overlaps components"})
        if include_ivr:
            runtime = IVR / "runtime" / f"seed_{seed}"
            fits = pd.read_csv(runtime / "fit_audit.csv")
            fits = fits.loc[fits["round"] <= final_round]
            acq = pd.read_csv(runtime / "acquisition_runtime.csv")
            acq = acq.loc[acq["round"] < final_round]
            records.append({"outer_seed": seed, "method": "kernel_ivr", "fit_count": len(fits),
                            "extra_ensemble_fits": 0, "epochs": int(fits.epochs_run.sum()),
                            "optimizer_steps": int(fits.epochs_run.sum()),
                            "training_seconds": float(fits.training_seconds.sum()),
                            "gradient_seconds": float(acq.gradient_seconds.sum()),
                            "gradient_extractions": len(acq), "ensemble_inference_seconds": 0.,
                            "selector_seconds": float(acq.ivr_seconds.sum()),
                            "historical_round_elapsed_seconds": np.nan,
                            "new_fit_count": 0, "new_training_seconds": 0.,
                            "cost_note": "historical effort reused by this report; original IVR used 21 new fits per seed"})
    return pd.DataFrame(records)


def write_evaluation(tables, costs, output):
    output = Path(output)
    results, figures = output / "results", output / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    for name, frame in {**tables, "compute_cost": costs}.items():
        frame.to_csv(results / f"{name}.csv", index=False)
    summary = tables["metric_summary"]
    wide = []
    for method in sorted(summary.method.unique()):
        values = summary.loc[summary.method.eq(method)].set_index("metric")
        row = {"method": method}
        for metric in METRICS:
            row[metric] = values.loc[metric, "endpoint_mean"]
            row[f"{metric}_AULC"] = values.loc[metric, "aulc_mean"]
        for metric in ("combined_normalized_RMSE", "V1_RMSE", "V2_RMSE", "V1_R2", "V2_R2"):
            for target in ("N80", "N90", "N95"):
                hit = tables["labels_to_target"].query("scope == 'cohort_mean' and method == @method and metric == @metric and target == @target").iloc[0]
                row[f"{metric}_{target}"] = hit.first_observed_labels
                row[f"{metric}_{target}_status"] = hit.status
        saving = tables["label_saving"]
        saving = saving.loc[saving.scope.eq("cohort_mean") & saving.method.eq(method) &
                            saving.metric.eq(METRICS[0]) & saving.target.str.startswith("Random@") &
                            saving.crossing.eq("first_observed_labels")].iloc[0]
        row["NRMSE_incremental_label_saving_observed"] = saving.incremental_saving
        effect = tables["paired_effects"].query("method == @method and comparator == 'random' and metric == 'combined_normalized_RMSE' and statistic == 'aulc'")
        row["NRMSE_AULC_wins_vs_random"] = int(effect.iloc[0].directional_wins) if len(effect) else np.nan
        cost = costs.loc[costs.method.eq(method)]
        for column in ("fit_count", "epochs", "training_seconds", "gradient_seconds", "new_fit_count"):
            row[column] = cost[column].sum(min_count=1)
        wide.append(row)
    pd.DataFrame(wide).to_csv(results / "complete_results.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    mean = tables["cohort_mean_learning_curve"]
    for ax, metric in zip(axes.flat, (METRICS[0], "V1_RMSE", "V2_RMSE", "V1_R2", "V2_R2")):
        for method, rows in mean.groupby("method"):
            ax.plot(rows.active_label_count, rows[metric], marker=".", ms=3, label=method)
        ax.set(title=metric, xlabel="Active training labels")
        ax.grid(alpha=.2)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center")
    fig.tight_layout()
    fig.savefig(figures / "learning_curves.png", dpi=180)
    plt.close(fig)


def run_phase0():
    sources = [BASELINE / "results/learning_curve_metrics.csv", BASELINE / "results/full_data_reference.csv",
               IVR / "results/learning_curve_metrics.csv"]
    before = {str(p.relative_to(ROOT)): sha256_file(p) for p in sources}
    curves = pd.read_csv(sources[0])
    ivr = pd.read_csv(sources[2])
    # Verify the extension preserved its reused baseline values before deduplicating.
    keys = ["outer_seed", "method", "active_label_count"]
    old = curves.loc[curves.method.isin(["random", "lcmd"])].sort_values(keys)
    reused = ivr.loc[ivr.method.isin(["random", "lcmd"])].sort_values(keys)
    if old[keys].values.tolist() != reused[keys].values.tolist() or not np.allclose(old[list(METRICS)], reused[list(METRICS)], atol=1e-12, rtol=0):
        raise RuntimeError("IVR reused controls differ from original baselines")
    curves = pd.concat([curves, ivr.loc[ivr.method.eq("kernel_ivr")]], ignore_index=True)
    full = pd.read_csv(sources[1])
    tables = evaluate(curves, full)
    costs = historical_costs()
    write_evaluation(tables, costs, STUDY)
    if before != {str(p.relative_to(ROOT)): sha256_file(p) for p in sources}:
        raise RuntimeError("historical inputs changed during reporting")
    atomic_json(STUDY / "provenance.json", {"source_files": before, "new_training_fits": 0,
                "new_test_truth_accesses": 0, "evidence": "retrospective_descriptive_reanalysis",
                "bootstrap": "10000 paired seed resamples, seed 20260918; overlapping splits, not independent confirmation",
                "aulc": "trapezoidal integral / budget span; no error/gap normalization",
                "gap": "unclipped; nonpositive initial-to-full gap undefined",
                "std": "sample standard deviation ddof=1", "validation_labels": 416})
    return {"output": str(STUDY), "curve_rows": len(curves), "new_fits": 0}
