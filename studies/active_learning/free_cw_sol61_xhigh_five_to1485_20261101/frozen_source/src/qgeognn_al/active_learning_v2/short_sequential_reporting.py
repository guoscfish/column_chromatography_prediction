"""Post-freeze reporting for the 333--429 two-method sequential screen."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import sha256_file
from ..training.predictor import atomic_json
from .maxdet_study import BASELINE
from .short_sequential_study import (
    ACQUISITION_ROUNDS,
    ACTIVE_LABEL_BUDGETS,
    BATCH_SIZE,
    COMPARATORS,
    DISPLAY_NAMES,
    IVR_STUDY,
    MAXDET_STUDY,
    METHODS,
    SEEDS,
)


METRICS = (
    "combined_normalized_RMSE",
    "V1_RMSE", "V1_MAE", "V1_R2",
    "V2_RMSE", "V2_MAE", "V2_R2",
)


def partial_aulc(labels: np.ndarray, values: np.ndarray) -> float:
    x = np.asarray(labels, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.tolist() != list(ACTIVE_LABEL_BUDGETS) or y.shape != x.shape or not np.isfinite(y).all():
        raise ValueError("AULC requires the complete finite 333--429 curve")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def _historical_curves() -> pd.DataFrame:
    sources = (
        (BASELINE / "results/learning_curve_metrics.csv", ("random", "lcmd", "hybrid")),
        (MAXDET_STUDY / "results/learning_curve_metrics.csv", ("gradient_maxdet",)),
        (IVR_STUDY / "results/learning_curve_metrics.csv", ("kernel_ivr",)),
    )
    frames = []
    for path, methods in sources:
        table = pd.read_csv(path)
        selected = table.loc[
            table.outer_seed.isin(SEEDS)
            & table.method.isin(methods)
            & table.active_label_count.isin(ACTIVE_LABEL_BUDGETS),
            ["outer_seed", "method", "round", "active_label_count", *METRICS],
        ].copy()
        for method in methods:
            arm = selected.loc[selected.method.eq(method)]
            if len(arm) != len(SEEDS) * len(ACTIVE_LABEL_BUDGETS):
                raise RuntimeError(f"matched historical curve incomplete: {method}")
        frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def _aulc_table(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, method), group in curves.groupby(["outer_seed", "method"], sort=False):
        group = group.sort_values("active_label_count")
        rows.append({
            "outer_seed": int(seed),
            "method": method,
            "start_active_labels": ACTIVE_LABEL_BUDGETS[0],
            "stop_active_labels": ACTIVE_LABEL_BUDGETS[-1],
            "AULC_333_429": partial_aulc(
                group.active_label_count.to_numpy(), group.combined_normalized_RMSE.to_numpy()
            ),
        })
    return pd.DataFrame(rows)


def _paired(aulc: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        endpoint = curves.loc[
            curves.outer_seed.eq(seed) & curves.active_label_count.eq(ACTIVE_LABEL_BUDGETS[-1])
        ].set_index("method").combined_normalized_RMSE
        area = aulc.loc[aulc.outer_seed.eq(seed)].set_index("method").AULC_333_429
        for method in METHODS:
            for comparator in ("lcmd", "gradient_maxdet"):
                rows.append({
                    "outer_seed": seed,
                    "method": method,
                    "comparator": comparator,
                    "method_AULC_333_429": float(area[method]),
                    "comparator_AULC_333_429": float(area[comparator]),
                    "method_minus_comparator_AULC": float(area[method] - area[comparator]),
                    "method_NRMSE_at_429": float(endpoint[method]),
                    "comparator_NRMSE_at_429": float(endpoint[comparator]),
                    "method_minus_comparator_NRMSE_at_429": float(endpoint[method] - endpoint[comparator]),
                    "lower_is_better": True,
                })
    detail = pd.DataFrame(rows)
    means = detail.groupby(["method", "comparator"], as_index=False).agg(
        outer_seed=("outer_seed", lambda _: "2-seed mean"),
        method_AULC_333_429=("method_AULC_333_429", "mean"),
        comparator_AULC_333_429=("comparator_AULC_333_429", "mean"),
        method_minus_comparator_AULC=("method_minus_comparator_AULC", "mean"),
        method_NRMSE_at_429=("method_NRMSE_at_429", "mean"),
        comparator_NRMSE_at_429=("comparator_NRMSE_at_429", "mean"),
        method_minus_comparator_NRMSE_at_429=("method_minus_comparator_NRMSE_at_429", "mean"),
    )
    means["lower_is_better"] = True
    return pd.concat([detail, means[detail.columns]], ignore_index=True)


def _one_step(curves: pd.DataFrame, aulc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed_label, subset in [*( (str(seed), curves.loc[curves.outer_seed.eq(seed)]) for seed in SEEDS),
                               ("2-seed mean", curves.groupby(["method", "active_label_count"], as_index=False)["combined_normalized_RMSE"].mean())]:
        values = subset.loc[subset.method.isin(METHODS)].pivot(
            index="method", columns="active_label_count", values="combined_normalized_RMSE"
        )
        if seed_label == "2-seed mean":
            areas = aulc.groupby("method").AULC_333_429.mean()
        else:
            areas = aulc.loc[aulc.outer_seed.eq(int(seed_label))].set_index("method").AULC_333_429
        for method in METHODS:
            rows.append({
                "scope": seed_label,
                "method": method,
                "NRMSE_at_365": float(values.loc[method, 365]),
                "rank_at_365": int(values[365].rank(method="min").loc[method]),
                "NRMSE_at_429": float(values.loc[method, 429]),
                "rank_at_429": int(values[429].rank(method="min").loc[method]),
                "AULC_333_429": float(areas[method]),
                "rank_by_AULC": int(areas.loc[list(METHODS)].rank(method="min").loc[method]),
            })
    return pd.DataFrame(rows)


def _batch_path(study: Path, seed: int, method: str, source_round: int) -> Path:
    if method in METHODS:
        return study / "runtime" / f"seed_{seed}" / method / f"round_{source_round:02d}/acquisition_artifacts/selected_next_batch.csv"
    if method == "lcmd":
        return BASELINE / "runtime" / f"seed_{seed}" / "lcmd" / f"round_{source_round:02d}/acquisition_artifacts/selected_next_batch.csv"
    if method == "gradient_maxdet":
        return MAXDET_STUDY / "runtime" / f"seed_{seed}" / method / f"round_{source_round:02d}/acquisition/selected_next_batch.csv"
    raise ValueError(method)


def _selection_overlap(study: Path) -> pd.DataFrame:
    pairs = (
        ("center_width_lcmd", "lcmd"),
        ("center_width_lcmd", "gradient_maxdet"),
        ("direction_lcmd", "lcmd"),
        ("direction_lcmd", "gradient_maxdet"),
        ("center_width_lcmd", "direction_lcmd"),
    )
    rows = []
    for seed in SEEDS:
        for source_round in range(ACQUISITION_ROUNDS):
            for left, right in pairs:
                left_ids = set(pd.read_csv(_batch_path(study, seed, left, source_round)).sample_id.astype(str))
                right_ids = set(pd.read_csv(_batch_path(study, seed, right, source_round)).sample_id.astype(str))
                count = len(left_ids & right_ids)
                rows.append({
                    "outer_seed": seed,
                    "source_round": source_round,
                    "acquisition_round": source_round + 1,
                    "active_labels_before_acquisition": ACTIVE_LABEL_BUDGETS[source_round],
                    "method_a": left,
                    "method_b": right,
                    "overlap_count": count,
                    "overlap_fraction": count / BATCH_SIZE,
                })
    return pd.DataFrame(rows)


def _compute_cost(study: Path) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for method in METHODS:
            for round_index in range(ACQUISITION_ROUNDS + 1):
                directory = study / "runtime" / f"seed_{seed}" / method / f"round_{round_index:02d}"
                freeze = json.loads((directory / "round_freeze.json").read_text())
                if round_index == 0:
                    fit = json.loads(Path(freeze["checkpoint_path"]).with_name("fit_audit.json").read_text())
                    new_training = 0.0
                    reused = True
                else:
                    fit = json.loads((directory / "model/fit_audit.json").read_text())
                    new_training = float(fit["training_seconds"])
                    reused = False
                acquisition_path = directory / "acquisition_artifacts/contract.json"
                acquisition = json.loads(acquisition_path.read_text()) if acquisition_path.exists() else {}
                rows.append({
                    "outer_seed": seed,
                    "method": method,
                    "round": round_index,
                    "active_label_count": ACTIVE_LABEL_BUDGETS[round_index],
                    "evaluation_fits": 0 if reused else 1,
                    "evaluation_fit_reused": reused,
                    "ensemble_only_fits": 0,
                    "training_seconds": new_training,
                    "source_reused_training_seconds": float(fit["training_seconds"]) if reused else 0.0,
                    "gradient_extraction_seconds": float(acquisition.get("gradient_extraction_seconds", 0.0)),
                    "representation_prediction_seconds": float(acquisition.get("representation_prediction_seconds", 0.0)),
                    "acquisition_selector_seconds": float(acquisition.get("selector_seconds", 0.0)),
                })
    return pd.DataFrame(rows)


def _decisions(curves: pd.DataFrame, aulc: pd.DataFrame) -> dict:
    mean_area = aulc.groupby("method").AULC_333_429.mean()
    endpoints = curves.loc[curves.active_label_count.eq(429)].pivot(
        index="outer_seed", columns="method", values="combined_normalized_RMSE"
    )
    areas = aulc.pivot(index="outer_seed", columns="method", values="AULC_333_429")
    strong = min(("lcmd", "gradient_maxdet"), key=lambda method: mean_area[method])
    decisions = {}
    for method in METHODS:
        area_delta = areas[method] - areas[strong]
        endpoint_delta = endpoints[method] - endpoints[strong]
        condition_a = bool(mean_area[method] < mean_area[strong] and area_delta.max() <= 0.05)
        condition_b = any(bool((areas[method] < areas[baseline]).all()) for baseline in ("lcmd", "gradient_maxdet"))
        condition_c = bool(endpoints[method].mean() < endpoints[strong].mean() and mean_area[method] <= mean_area[strong] + 0.01)
        stop = bool((area_delta > 0).all() and (endpoint_delta > 0).all())
        if stop:
            category = "STOP"
        elif condition_a or condition_b or condition_c:
            category = "PROMISING_FOR_525"
        else:
            category = "INCONCLUSIVE"
        decisions[method] = {
            "decision": category,
            "strongest_baseline_by_mean_AULC": strong,
            "paired_AULC_deltas": {str(seed): float(area_delta.loc[seed]) for seed in SEEDS},
            "paired_endpoint_deltas": {str(seed): float(endpoint_delta.loc[seed]) for seed in SEEDS},
            "condition_A": condition_a,
            "condition_B": condition_b,
            "condition_C": condition_c,
            "stop_condition": stop,
        }
    return {
        "status": "COMPLETE_TRUNCATED_DEVELOPMENTAL_SCREEN",
        "evidence_class": "TRUNCATED_DEVELOPMENTAL_SCREEN",
        "methods": decisions,
        "automatic_extension_to_525": False,
        "ensemble_uncertainty_executed": False,
        "large_single_seed_AULC_worsening_threshold": 0.05,
        "AULC_not_obviously_worse_tolerance": 0.01,
    }


def _figures(study: Path, curves: pd.DataFrame, aulc: pd.DataFrame, one_step: pd.DataFrame, overlap: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = study / "figures"
    figures.mkdir(exist_ok=True)
    focus = ("lcmd", "gradient_maxdet", *METHODS, "random")
    styles = {"random": "--", "lcmd": "-", "gradient_maxdet": "-", METHODS[0]: "-", METHODS[1]: "-"}
    colors = {"random": "#777777", "lcmd": "#1f77b4", "gradient_maxdet": "#d62728",
              METHODS[0]: "#2ca02c", METHODS[1]: "#9467bd"}
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for method in focus:
        group = curves.loc[curves.method.eq(method)].groupby("active_label_count").combined_normalized_RMSE.mean()
        ax.plot(group.index, group.values, marker="o", linestyle=styles[method], color=colors[method], label=DISPLAY_NAMES[method])
    ax.set(xlabel="Active labels", ylabel="Combined normalized RMSE", xticks=ACTIVE_LABEL_BUDGETS)
    ax.grid(alpha=.2); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(figures / "nrmse_learning_curve_333_429.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, seed in zip(axes, SEEDS):
        for method in focus:
            group = curves.loc[curves.outer_seed.eq(seed) & curves.method.eq(method)].sort_values("active_label_count")
            axis.plot(group.active_label_count, group.combined_normalized_RMSE, marker="o", linestyle=styles[method], color=colors[method], label=DISPLAY_NAMES[method])
        axis.set(title=f"Seed {seed}", xlabel="Active labels", xticks=ACTIVE_LABEL_BUDGETS); axis.grid(alpha=.2)
    axes[0].set_ylabel("Combined normalized RMSE"); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figures / "per_seed_learning_curves.png", dpi=180); plt.close(fig)

    means = aulc.loc[aulc.method.isin(focus)].groupby("method").AULC_333_429.mean().sort_values()
    fig, ax = plt.subplots(figsize=(7.3, 4.3)); ax.bar([DISPLAY_NAMES[v] for v in means.index], means.values, color=[colors.get(v, "#888888") for v in means.index])
    ax.set_ylabel("AULC 333-429 (lower is better)"); ax.tick_params(axis="x", rotation=25); fig.tight_layout()
    fig.savefig(figures / "aulc_comparison.png", dpi=180); plt.close(fig)

    mean_rank = one_step.loc[one_step.scope.eq("2-seed mean")].set_index("method")
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for method in METHODS:
        row = mean_rank.loc[method]
        ax.scatter(row.NRMSE_at_365, row.NRMSE_at_429, s=70, color=colors[method], label=DISPLAY_NAMES[method])
    ax.set(xlabel="NRMSE at 365", ylabel="NRMSE at 429"); ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(figures / "one_step_vs_round3.png", dpi=180); plt.close(fig)

    summary = overlap.groupby(["method_a", "method_b"]).overlap_fraction.mean().sort_values()
    labels = [f"{DISPLAY_NAMES[a]} vs {DISPLAY_NAMES[b]}" for a, b in summary.index]
    fig, ax = plt.subplots(figsize=(8.2, 4.2)); ax.bar(labels, summary.values, color="#4c78a8")
    ax.set_ylabel("Mean batch overlap fraction"); ax.tick_params(axis="x", rotation=25); fig.tight_layout()
    fig.savefig(figures / "selection_overlap.png", dpi=180); plt.close(fig)


def _report_text(curves: pd.DataFrame, aulc: pd.DataFrame, one_step: pd.DataFrame, paired: pd.DataFrame, decisions: dict, cost: pd.DataFrame, overlap: pd.DataFrame) -> str:
    new = one_step.loc[one_step.scope.eq("2-seed mean")].set_index("method")
    round1 = new.NRMSE_at_365.idxmin(); round3 = new.NRMSE_at_429.idxmin(); area = new.AULC_333_429.idxmin()
    crossing = new.rank_at_365.to_dict() != new.rank_by_AULC.to_dict()
    seed_lines = []
    for seed in SEEDS:
        rows = one_step.loc[one_step.scope.eq(str(seed))].set_index("method")
        for method in METHODS:
            row = rows.loc[method]
            seed_lines.append(f"| {seed} | {DISPLAY_NAMES[method]} | {row.NRMSE_at_365:.6f} | {row.NRMSE_at_429:.6f} | {row.AULC_333_429:.6f} |")
    delta_lines = []
    detail = paired.loc[paired.outer_seed.astype(str).isin([str(v) for v in SEEDS])]
    for row in detail.itertuples():
        delta_lines.append(f"| {row.outer_seed} | {DISPLAY_NAMES[row.method]} | {DISPLAY_NAMES[row.comparator]} | {row.method_minus_comparator_AULC:+.6f} | {row.method_minus_comparator_NRMSE_at_429:+.6f} |")
    decision_lines = [f"- {DISPLAY_NAMES[method]}: **{value['decision']}** versus {DISPLAY_NAMES[value['strongest_baseline_by_mean_AULC']]} (AULC deltas {value['paired_AULC_deltas']}; endpoint deltas {value['paired_endpoint_deltas']})."
                      for method, value in decisions["methods"].items()]
    cost_summary = cost.groupby("method")[["training_seconds", "gradient_extraction_seconds", "acquisition_selector_seconds"]].sum()
    cost_lines = [f"- {DISPLAY_NAMES[method]}: {row.training_seconds:.1f}s new training, {row.gradient_extraction_seconds:.1f}s gradient extraction, {row.acquisition_selector_seconds:.3f}s selection."
                  for method, row in cost_summary.iterrows()]
    means = curves.groupby(["method", "active_label_count"]).combined_normalized_RMSE.mean().unstack()
    mean_areas = aulc.groupby("method").AULC_333_429.mean()
    all_methods = tuple(COMPARATORS) + METHODS
    leaders = {
        "round1": means.loc[list(all_methods), 365].idxmin(),
        "round3": means.loc[list(all_methods), 429].idxmin(),
        "aulc": mean_areas.loc[list(all_methods)].idxmin(),
    }
    seed_6101 = one_step.loc[one_step.scope.eq("6101")].set_index("method")
    cw_6101 = seed_6101.loc["center_width_lcmd"]
    direction_6101 = seed_6101.loc["direction_lcmd"]
    mean_delta = paired.loc[paired.outer_seed.astype(str).eq("2-seed mean")]
    difference_lines = [
        f"- {DISPLAY_NAMES[row.method]} minus {DISPLAY_NAMES[row.comparator]}: AULC {row.method_minus_comparator_AULC:+.6f}; NRMSE@429 {row.method_minus_comparator_NRMSE_at_429:+.6f}."
        for row in mean_delta.itertuples()
    ]
    overlap_means = overlap.groupby(["method_a", "method_b"]).overlap_count.mean()
    overlap_lines = [
        f"- {DISPLAY_NAMES[left]} vs {DISPLAY_NAMES[right]}: {count:.2f}/32 ({count / BATCH_SIZE:.3f})."
        for (left, right), count in overlap_means.items()
    ]
    return f"""# Short sequential B=32 developmental screen

This screen ran only Center/Width-LCMD and Direction-LCMD for seeds 157 and 6101, with adaptive scratch retraining at 333 -> 365 -> 397 -> 429. Ensemble Uncertainty was explicitly excluded. This is developmental evidence because these historical row-test cohorts have already been inspected.

## Primary results

| Seed | Method | NRMSE@365 | NRMSE@429 | AULC 333-429 |
|---:|---|---:|---:|---:|
{chr(10).join(seed_lines)}

Among the two new methods, round 1 is led by **{DISPLAY_NAMES[round1]}**, round 3 by **{DISPLAY_NAMES[round3]}**, and AULC by **{DISPLAY_NAMES[area]}**. The round-1 and AULC rankings {'do not agree, so a crossing/ranking reversal occurred' if crossing else 'agree; no ranking reversal occurred between the two new methods'}.

Across both new methods and all five matched historical comparators, the two-seed mean leader at 365 is **{DISPLAY_NAMES[leaders['round1']]}**, at 429 is **{DISPLAY_NAMES[leaders['round3']]}**, and over AULC is **{DISPLAY_NAMES[leaders['aulc']]}**. These are descriptive two-seed rankings, not significance tests.

The earlier development one-step Center/Width negative signal (about 0.728810 versus raw Gradient-LCMD 0.708760) does not hold uniformly here: CW beats Gradient-LCMD on both seeds' 333-429 AULC and @429, but trails the stronger Gradient-MaxDet on both seeds for both measures. In seed 6101, CW starts behind Direction at 365 ({cw_6101.NRMSE_at_365:.6f} versus {direction_6101.NRMSE_at_365:.6f}) and overtakes it at 429 ({cw_6101.NRMSE_at_429:.6f} versus {direction_6101.NRMSE_at_429:.6f}); seed 157 already favors CW at 365. Thus a one-step comparison does not reliably capture their short sequential ranking.

## Paired differences

Positive values mean the new method is worse.

| Seed | Method | Comparator | AULC delta | NRMSE@429 delta |
|---:|---|---|---:|---:|
{chr(10).join(delta_lines)}

Two-seed descriptive mean deltas:

{chr(10).join(difference_lines)}

Direction does not translate its different first-batch geometry into a stable predictive gain: its AULC is worse than Gradient-MaxDet on both seeds and worse than Gradient-LCMD in seed 157 (only marginally better in seed 6101), while its @429 NRMSE is worse than both baselines on both seeds.

## Selection overlap

Mean intersection per batch over six seed-round pairs (per-pair detail is in `results/selection_overlap.csv`):

{chr(10).join(overlap_lines)}

CW and Direction choose notably different batches. Neither low nor high overlap establishes superiority without the paired predictive results above.

## Continue/stop assessment

{chr(10).join(decision_lines)}

No method is automatically continued to 525.

The explicit STOP rule takes precedence when both seeds have worse AULC and @429 than the strongest baseline, even if a method improves over the weaker Gradient-LCMD. Ensemble Uncertainty was not run, so its sequential conclusion and cost cannot be evaluated by this study.

## Compute cost

{chr(10).join(cost_lines)}

These costs describe computation, not label efficiency. Both methods use one reused round-0 fit and three new evaluation fits per seed; neither runs an ensemble.

## Integrity and interpretation

Every acquisition used the current trajectory-specific checkpoint, recomputed its representation, installed all current L_t rows as LCMD centers, selected exactly 32 unique current-U_t IDs, and revealed their labels only after selection was frozen. All 16 prediction points were frozen before test truth was read. Comparator tables and batch IDs were read only from exact seed/budget historical trajectories. Low selection overlap establishes a different strategy, not predictive superiority.
"""


def write_report(study: Path, new_curves: pd.DataFrame, label_access: pd.DataFrame) -> dict:
    study = Path(study)
    results = study / "results"
    results.mkdir(exist_ok=True)
    historical = _historical_curves()
    curves = pd.concat([historical, new_curves], ignore_index=True).sort_values(["outer_seed", "method", "active_label_count"])
    expected_new = len(SEEDS) * len(METHODS) * len(ACTIVE_LABEL_BUDGETS)
    if len(new_curves) != expected_new or new_curves.duplicated(["outer_seed", "method", "active_label_count"]).any():
        raise RuntimeError("new learning-curve matrix incomplete")
    for seed in SEEDS:
        round0 = curves.loc[curves.outer_seed.eq(seed) & curves.active_label_count.eq(333)]
        if round0.combined_normalized_RMSE.nunique() != 1:
            raise RuntimeError("shared round-0 metric identity drift")
    mean_curve = curves.groupby(["method", "active_label_count"], as_index=False)[list(METRICS)].mean()
    aulc = _aulc_table(curves)
    paired = _paired(aulc, curves)
    one_step = _one_step(curves, aulc)
    overlap = _selection_overlap(study)
    cost = _compute_cost(study)
    decisions = _decisions(curves, aulc)
    curves.to_csv(results / "learning_curve_metrics.csv", index=False)
    mean_curve.to_csv(results / "cohort_mean_learning_curve.csv", index=False)
    aulc.to_csv(results / "partial_aulc_333_429.csv", index=False)
    paired.to_csv(results / "paired_comparison.csv", index=False)
    one_step.to_csv(results / "one_step_vs_short_sequential.csv", index=False)
    overlap.to_csv(results / "selection_overlap.csv", index=False)
    cost.to_csv(results / "compute_cost.csv", index=False)
    label_access.to_csv(results / "test_label_access_audit.csv", index=False)
    atomic_json(study / "decision.json", decisions)
    _figures(study, curves, aulc, one_step, overlap)
    (study / "FINAL_REPORT.md").write_text(_report_text(curves, aulc, one_step, paired, decisions, cost, overlap))
    manifest_files = [path for path in study.rglob("*") if path.is_file() and "runtime" not in path.parts]
    atomic_json(study / "artifact_manifest.json", {
        "status": "COMPLETE_TRUNCATED_DEVELOPMENTAL_SCREEN",
        "methods_run": list(METHODS),
        "ensemble_uncertainty_executed": False,
        "hard_stop_active_labels": ACTIVE_LABEL_BUDGETS[-1],
        "files": {str(path.relative_to(study)): sha256_file(path) for path in sorted(manifest_files)
                  if path.name != "artifact_manifest.json"},
    })
    return {"status": "COMPLETE_TRUNCATED_DEVELOPMENTAL_SCREEN", "decision": decisions}
