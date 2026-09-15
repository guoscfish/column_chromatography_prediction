"""Preregistered learning-curve metrics and decisions for sequential B=32."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..training.predictor import atomic_json
from .sequential_protocol import (
    ACTIVE_LABEL_BUDGETS,
    BATCH_SIZE,
    COMPARABLE_AULC_RELATIVE_MARGIN,
    COMPARABLE_LABEL_MARGIN,
    CONFIRMATION_SEEDS,
    FINAL_ACTIVE_LABELS,
    INITIAL_ACTIVE_LABELS,
    METHODS,
)


METRIC_COLUMNS = (
    "combined_normalized_RMSE",
    "V1_RMSE",
    "V2_RMSE",
    "V1_MAE",
    "V2_MAE",
    "V1_R2",
    "V2_R2",
)


def normalized_aulc(active_labels: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    """Return trapezoidal area and average NRMSE over the frozen interval."""

    labels = np.asarray(active_labels, dtype=np.float64)
    values = np.asarray(errors, dtype=np.float64)
    expected = np.asarray(ACTIVE_LABEL_BUDGETS, dtype=np.float64)
    if labels.ndim != 1 or values.shape != labels.shape or not np.array_equal(labels, expected):
        raise ValueError("AULC requires every registered 333..1005 budget exactly once in order")
    if not np.isfinite(values).all():
        raise ValueError("AULC errors must be finite")
    raw = float(np.trapezoid(values, labels))
    return raw, raw / float(FINAL_ACTIVE_LABELS - INITIAL_ACTIVE_LABELS)


def labels_to_target(
    active_labels: np.ndarray,
    errors: np.ndarray,
    target: float,
) -> float | None:
    """First target crossing with adjacent-point interpolation; never extrapolate."""

    labels = np.asarray(active_labels, dtype=np.float64)
    values = np.asarray(errors, dtype=np.float64)
    if labels.ndim != 1 or values.shape != labels.shape or len(labels) == 0:
        raise ValueError("aligned nonempty one-dimensional curve required")
    if not np.all(np.diff(labels) > 0) or not np.isfinite(values).all() or not math.isfinite(float(target)):
        raise ValueError("labels must increase and errors/target must be finite")
    threshold = float(target)
    for index, value in enumerate(values):
        if value <= threshold:
            if index == 0:
                return float(labels[0])
            x0, x1 = labels[index - 1:index + 1]
            y0, y1 = values[index - 1:index + 1]
            if y0 <= threshold:
                return float(x0)
            if y1 == y0:
                return float(x1)
            fraction = (threshold - y0) / (y1 - y0)
            return float(x0 + fraction * (x1 - x0))
    return None


def incremental_label_saving(method_labels_to_target: float | None) -> dict[str, float | int | str | None]:
    """Savings against Random@1005, with shared L0 treated as sunk cost."""

    if method_labels_to_target is None:
        return {
            "status": "CENSORED_GT_1005",
            "labels_to_target": None,
            "saved_experimental_labels": None,
            "saving_total": None,
            "saving_incremental": None,
        }
    labels = float(method_labels_to_target)
    if not INITIAL_ACTIVE_LABELS <= labels <= FINAL_ACTIVE_LABELS:
        raise ValueError("labels-to-target lies outside the registered interval")
    saved = FINAL_ACTIVE_LABELS - labels
    return {
        "status": "REACHED",
        "labels_to_target": labels,
        "saved_experimental_labels": saved,
        "saving_total": saved / FINAL_ACTIVE_LABELS,
        "saving_incremental": saved / (FINAL_ACTIVE_LABELS - INITIAL_ACTIVE_LABELS),
    }


def validate_learning_curve(frame: pd.DataFrame) -> None:
    required = {
        "outer_seed", "method", "round", "active_label_count",
        "active_label_fraction_outer_train", "shared_validation_label_count",
        "total_observed_non_test_labels", "total_observed_fraction_full_dataset",
        *METRIC_COLUMNS,
    }
    if missing := required - set(frame):
        raise ValueError(f"learning curve is missing columns: {sorted(missing)}")
    if set(frame.outer_seed) != set(CONFIRMATION_SEEDS) or set(frame.method) != set(METHODS):
        raise ValueError("learning curve requires the exact five-seed/three-method matrix")
    if frame.duplicated(["outer_seed", "method", "round"]).any():
        raise ValueError("duplicate seed/method/round learning-curve row")
    for (_, _), rows in frame.groupby(["outer_seed", "method"], sort=False):
        ordered = rows.sort_values("round")
        if ordered["round"].tolist() != list(range(len(ACTIVE_LABEL_BUDGETS))):
            raise ValueError("learning curve lacks a complete round 0..21 trajectory")
        if ordered.active_label_count.tolist() != list(ACTIVE_LABEL_BUDGETS):
            raise ValueError("learning curve active-label schedule drift")
    if not np.isfinite(frame[list(METRIC_COLUMNS)].to_numpy(float)).all():
        raise ValueError("learning curve contains non-finite metrics")


def cohort_mean_curve(frame: pd.DataFrame) -> pd.DataFrame:
    validate_learning_curve(frame)
    numeric = [
        "active_label_count", "active_label_fraction_outer_train",
        "shared_validation_label_count", "total_observed_non_test_labels",
        "total_observed_fraction_full_dataset", *METRIC_COLUMNS,
    ]
    result = frame.groupby(["method", "round"], sort=False)[numeric].mean().reset_index()
    return result.sort_values(["method", "round"]).reset_index(drop=True)


def aulc_rows(frame: pd.DataFrame) -> pd.DataFrame:
    validate_learning_curve(frame)
    records: list[dict[str, object]] = []
    for (seed, method), rows in frame.groupby(["outer_seed", "method"], sort=False):
        ordered = rows.sort_values("active_label_count")
        raw, value = normalized_aulc(
            ordered.active_label_count.to_numpy(),
            ordered.combined_normalized_RMSE.to_numpy(),
        )
        records.append({
            "record_type": "per_seed",
            "outer_seed": int(seed),
            "method": method,
            "comparison": "",
            "AULC_raw": raw,
            "normalized_AULC": value,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
        })
    per_seed = pd.DataFrame(records)
    for method, rows in per_seed.groupby("method", sort=False):
        values = rows.normalized_AULC.to_numpy(float)
        records.append({
            "record_type": "method_summary", "outer_seed": np.nan, "method": method,
            "comparison": "", "AULC_raw": np.nan, "normalized_AULC": np.nan,
            "mean": float(values.mean()), "median": float(np.median(values)),
            "std": float(values.std(ddof=0)),
        })
    pivot = per_seed.pivot(index="outer_seed", columns="method", values="normalized_AULC")
    for left, right in (("hybrid", "random"), ("lcmd", "random"), ("lcmd", "hybrid")):
        differences = (pivot[left] - pivot[right]).to_numpy(float)
        records.append({
            "record_type": "paired_difference_summary", "outer_seed": np.nan,
            "method": "", "comparison": f"{left}_minus_{right}",
            "AULC_raw": np.nan, "normalized_AULC": np.nan,
            "mean": float(differences.mean()), "median": float(np.median(differences)),
            "std": float(differences.std(ddof=0)),
        })
    return pd.DataFrame(records)


def target_tables(
    mean_curve: pd.DataFrame,
    full_reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    if set(full_reference.outer_seed) != set(CONFIRMATION_SEEDS) or len(full_reference) != len(CONFIRMATION_SEEDS):
        raise ValueError("full-data reference requires exactly one row per confirmation seed")
    if not np.isfinite(full_reference.combined_normalized_RMSE.to_numpy(float)).all():
        raise ValueError("full-data reference metrics must be finite")
    random_curve = mean_curve.loc[mean_curve.method.eq("random")].sort_values("active_label_count")
    if random_curve.active_label_count.tolist() != list(ACTIVE_LABEL_BUDGETS):
        raise ValueError("cohort Random curve is incomplete")
    t_r30 = float(random_curve.loc[
        random_curve.active_label_count.eq(FINAL_ACTIVE_LABELS), "combined_normalized_RMSE"
    ].iloc[0])
    round_zero = mean_curve.loc[mean_curve["round"].eq(0), "combined_normalized_RMSE"].to_numpy(float)
    if len(round_zero) != len(METHODS) or np.ptp(round_zero) > 1e-12:
        raise RuntimeError("Round-0 evaluation must be identical across methods")
    e0 = float(round_zero[0])
    efull = float(full_reference.combined_normalized_RMSE.mean())
    t80 = efull + 0.20 * (e0 - efull)
    labels_records: list[dict[str, object]] = []
    savings_records: list[dict[str, object]] = []
    for target_name, target in (("T_R30", t_r30), ("T_80", t80)):
        for method in METHODS:
            rows = mean_curve.loc[mean_curve.method.eq(method)].sort_values("active_label_count")
            if target_name == "T_R30" and method == "random":
                labels = float(FINAL_ACTIVE_LABELS)
            else:
                labels = labels_to_target(
                    rows.active_label_count.to_numpy(),
                    rows.combined_normalized_RMSE.to_numpy(),
                    target,
                )
            labels_records.append({
                "target": target_name,
                "target_NRMSE": target,
                "method": method,
                "labels_to_target": labels,
                "status": "REACHED" if labels is not None else "CENSORED_GT_1005",
            })
            if target_name == "T_R30":
                savings_records.append({"method": method, **incremental_label_saving(labels)})
    return pd.DataFrame(labels_records), pd.DataFrame(savings_records), {
        "T_R30": t_r30, "E0": e0, "E_full": efull, "T_80": t80,
    }


def sequential_decision(aulc: pd.DataFrame, labels: pd.DataFrame, savings: pd.DataFrame) -> dict[str, object]:
    per_seed = aulc.loc[aulc.record_type.eq("per_seed")]
    pivot = per_seed.pivot(index="outer_seed", columns="method", values="normalized_AULC")
    if set(pivot.index) != set(CONFIRMATION_SEEDS) or set(pivot.columns) != set(METHODS):
        raise ValueError("decision requires complete per-seed AULC values")
    means = pivot.mean().to_dict()
    primary = labels.loc[labels.target.eq("T_R30")].set_index("method")
    saving = savings.set_index("method")
    target_values = {method: (None if pd.isna(primary.loc[method, "labels_to_target"])
                              else float(primary.loc[method, "labels_to_target"]))
                     for method in METHODS}
    saving_values = {method: (None if pd.isna(saving.loc[method, "saving_incremental"])
                              else float(saving.loc[method, "saving_incremental"]))
                     for method in METHODS}
    active_gain = {
        method: bool(means[method] < means["random"] or
                     (saving_values[method] is not None and saving_values[method] > 0))
        for method in ("hybrid", "lcmd")
    }
    relative_gap = abs(means["lcmd"] - means["hybrid"]) / min(means["lcmd"], means["hybrid"])
    target_gap = (None if target_values["lcmd"] is None or target_values["hybrid"] is None
                  else abs(target_values["lcmd"] - target_values["hybrid"]))
    wins = int((pivot.lcmd < pivot.random).sum())
    lcmd_conditions = {
        "mean_AULC_below_Random": bool(means["lcmd"] < means["random"]),
        "mean_AULC_below_Hybrid": bool(means["lcmd"] < means["hybrid"]),
        "positive_incremental_saving": bool(saving_values["lcmd"] is not None and saving_values["lcmd"] > 0),
        "labels_to_T_R30_no_more_than_Hybrid": bool(
            target_values["lcmd"] is not None and
            (target_values["hybrid"] is None or target_values["lcmd"] <= target_values["hybrid"])
        ),
        "AULC_directional_win_vs_Random_at_least_4_of_5": wins >= 4,
    }
    hybrid_best = bool(
        means["hybrid"] < means["lcmd"] and
        target_values["hybrid"] is not None and
        (target_values["lcmd"] is None or target_values["hybrid"] < target_values["lcmd"])
    )
    comparable = bool(
        relative_gap < COMPARABLE_AULC_RELATIVE_MARGIN and
        target_gap is not None and target_gap <= COMPARABLE_LABEL_MARGIN
    )
    if not any(active_gain.values()):
        decision = "NO_CLEAR_SEQUENTIAL_AL_GAIN"
    elif comparable:
        decision = "SEQUENTIAL_LCMD_HYBRID_COMPARABLE"
    elif all(lcmd_conditions.values()):
        decision = "SEQUENTIAL_LCMD_BEST"
    elif hybrid_best:
        decision = "SEQUENTIAL_HYBRID_BEST"
    else:
        decision = "NO_CLEAR_SEQUENTIAL_AL_GAIN"
    return {
        "decision": decision,
        "mean_normalized_AULC": means,
        "labels_to_T_R30": target_values,
        "incremental_saving": saving_values,
        "lcmd_conditions": lcmd_conditions,
        "lcmd_directional_wins_vs_random": wins,
        "hybrid_best_conditions_met": hybrid_best,
        "comparable": comparable,
        "lcmd_hybrid_relative_AULC_difference": relative_gap,
        "lcmd_hybrid_labels_to_target_difference": target_gap,
        "active_gain": active_gain,
        "thresholds_frozen_before_test_reveal": True,
    }


def _plot_results(mean_curve: pd.DataFrame, aulc: pd.DataFrame, labels: pd.DataFrame,
                  savings: pd.DataFrame, figures: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    colors = {"random": "#777777", "hybrid": "#377eb8", "lcmd": "#e41a1c"}
    for x_column, filename, xlabel in (
        ("active_label_count", "nrmse_vs_active_labels.png", "Active training labels"),
        ("total_observed_non_test_labels", "nrmse_vs_total_observed_labels.png", "Total observed non-test labels"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in METHODS:
            rows = mean_curve.loc[mean_curve.method.eq(method)].sort_values("round")
            ax.plot(rows[x_column], rows.combined_normalized_RMSE, marker="o", ms=3,
                    label=method, color=colors[method])
        ax.set_xlabel(xlabel); ax.set_ylabel("Combined normalized RMSE"); ax.legend()
        fig.tight_layout(); fig.savefig(figures / filename, dpi=180); plt.close(fig)

    summaries = aulc.loc[aulc.record_type.eq("method_summary")].set_index("method").loc[list(METHODS)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(METHODS), summaries["mean"], yerr=summaries["std"], color=[colors[m] for m in METHODS])
    ax.set_ylabel("Normalized AULC"); fig.tight_layout()
    fig.savefig(figures / "normalized_aulc_comparison.png", dpi=180); plt.close(fig)

    primary = labels.loc[labels.target.eq("T_R30")].copy()
    primary["plot_labels"] = primary.labels_to_target.fillna(FINAL_ACTIVE_LABELS + BATCH_SIZE)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(primary.method, primary.plot_labels, color=[colors[m] for m in primary.method])
    ax.set_ylabel("Active labels to T_R30"); fig.tight_layout()
    fig.savefig(figures / "labels_to_target.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    values = savings.set_index("method").loc[list(METHODS), "saving_incremental"].fillna(0) * 100
    ax.bar(list(METHODS), values, color=[colors[m] for m in METHODS])
    ax.set_ylabel("Incremental new-experiment saving (%)"); fig.tight_layout()
    fig.savefig(figures / "label_saving.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for endpoint, ax in zip(("V1", "V2"), axes):
        for method in METHODS:
            rows = mean_curve.loc[mean_curve.method.eq(method)].sort_values("round")
            ax.plot(rows.active_label_count, rows[f"{endpoint}_RMSE"], label=method, color=colors[method])
        ax.set_title(endpoint); ax.set_xlabel("Active labels"); ax.set_ylabel("RMSE")
    axes[1].legend(); fig.tight_layout()
    fig.savefig(figures / "endpoint_rmse_learning_curves.png", dpi=180); plt.close(fig)


def write_final_outputs(study: Path, learning: pd.DataFrame, full_reference: pd.DataFrame,
                        runtime_rows: pd.DataFrame, integrity: pd.DataFrame,
                        label_access: pd.DataFrame, initialization: pd.DataFrame) -> dict[str, object]:
    """Write metrics only after the caller has passed the global test barrier."""

    validate_learning_curve(learning)
    results = Path(study) / "results"
    results.mkdir(parents=True, exist_ok=True)
    mean_curve = cohort_mean_curve(learning)
    aulc = aulc_rows(learning)
    labels, savings, targets = target_tables(mean_curve, full_reference)
    decision = sequential_decision(aulc, labels, savings)
    learning.to_csv(results / "learning_curve_metrics.csv", index=False)
    mean_curve.to_csv(results / "cohort_mean_learning_curve.csv", index=False)
    aulc.to_csv(results / "normalized_aulc.csv", index=False)
    labels.to_csv(results / "labels_to_target.csv", index=False)
    savings.to_csv(results / "label_saving.csv", index=False)
    full_reference.to_csv(results / "full_data_reference.csv", index=False)
    learning[["outer_seed", "method", "round", "active_label_count", "V1_RMSE", "V2_RMSE",
              "V1_MAE", "V2_MAE", "V1_R2", "V2_R2"]].to_csv(
        results / "endpoint_learning_curves.csv", index=False
    )
    runtime_rows.to_csv(results / "round_runtime.csv", index=False)
    integrity.to_csv(results / "trajectory_integrity_audit.csv", index=False)
    label_access.to_csv(results / "label_access_audit.csv", index=False)
    initialization.to_csv(results / "initialization_hash_audit.csv", index=False)
    atomic_json(Path(study) / "decision.json", {**decision, "targets": targets})
    _plot_results(mean_curve, aulc, labels, savings, Path(study) / "figures")

    means = decision["mean_normalized_AULC"]
    primary = labels.loc[labels.target.eq("T_R30")].set_index("method")
    saving = savings.set_index("method")
    t80 = labels.loc[labels.target.eq("T_80")].set_index("method")
    def shown(value: object) -> str:
        return ">1005 / censored" if pd.isna(value) else f"{float(value):.2f}"
    report = f"""# Final Report — Sequential B=32 Active-Learning Efficiency

## Primary answers

Random, V2-Hybrid, and Gradient-LCMD mean normalized AULC values are
{means['random']:.6f}, {means['hybrid']:.6f}, and {means['lcmd']:.6f}, respectively.
The preregistered decision is **{decision['decision']}**.

T_R30 is {targets['T_R30']:.6f}. Labels-to-target are Random
{shown(primary.loc['random', 'labels_to_target'])}, Hybrid
{shown(primary.loc['hybrid', 'labels_to_target'])}, and LCMD
{shown(primary.loc['lcmd', 'labels_to_target'])} active labels.

Hybrid saves {shown(saving.loc['hybrid', 'saved_experimental_labels'])} new experiments
({shown(100 * saving.loc['hybrid', 'saving_incremental'])}%). LCMD saves
{shown(saving.loc['lcmd', 'saved_experimental_labels'])} new experiments
({shown(100 * saving.loc['lcmd', 'saving_incremental'])}%).

The secondary T_80 is {targets['T_80']:.6f}; labels-to-T_80 are Random
{shown(t80.loc['random', 'labels_to_target'])}, Hybrid
{shown(t80.loc['hybrid', 'labels_to_target'])}, and LCMD
{shown(t80.loc['lcmd', 'labels_to_target'])}.

Endpoint RMSE, MAE, and R2 are secondary predictor diagnostics in
`results/endpoint_learning_curves.csv`; they do not determine the active-learning winner.
"""
    (Path(study) / "FINAL_REPORT.md").write_text(report)
    return {**decision, "targets": targets}
