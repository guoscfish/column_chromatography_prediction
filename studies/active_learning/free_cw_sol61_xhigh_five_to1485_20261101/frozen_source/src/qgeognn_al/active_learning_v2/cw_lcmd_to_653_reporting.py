"""Derived reporting for the frozen CW-LCMD L525-to-L653 continuation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import sha256_file
from ..training.predictor import atomic_json
from .cw_lcmd_to_653_study import (
    ALL_BUDGETS, ALL_METHODS, COMPARATORS, DISPLAY_NAMES, IVR_STUDY,
    MAXDET_STUDY, METHOD, MID_LATE_BUDGETS, PRIMARY_BUDGETS, SEEDS,
    SOURCE_STUDY, STUDY, comparator_sources,
)
from .maxdet_study import BASELINE


METRICS = (
    "combined_normalized_RMSE", "V1_RMSE", "V1_MAE", "V1_R2",
    "V2_RMSE", "V2_MAE", "V2_R2",
)
STRONG_COMPARATORS = ("gradient_maxdet", "hybrid", "kernel_ivr")
ENDPOINT_DETERIORATION_TOLERANCE = 0.03
MATERIAL_DIRECTION_TOLERANCE = 0.01


def _validate_grid(frame: pd.DataFrame, budgets: tuple[int, ...], methods: tuple[str, ...] = ALL_METHODS) -> None:
    failures = []
    for seed in SEEDS:
        for method in methods:
            observed = sorted(frame.loc[
                frame.outer_seed.eq(seed) & frame.method.eq(method), "active_label_count"
            ].astype(int).tolist())
            if observed != list(budgets):
                failures.append(f"seed={seed} method={method} expected={list(budgets)} observed={observed}")
    if failures:
        raise RuntimeError("incomplete reporting budget grid: " + "; ".join(failures))


def _aulc(frame: pd.DataFrame, budgets: tuple[int, ...], column: str) -> pd.DataFrame:
    if len(budgets) < 2:
        raise ValueError("AULC needs at least two declared budgets")
    _validate_grid(frame.loc[frame.active_label_count.isin(budgets)], budgets)
    rows = []
    for (seed, method), group in frame.loc[frame.active_label_count.isin(budgets)].groupby(["outer_seed", "method"]):
        group = group.sort_values("active_label_count")
        x = group.active_label_count.to_numpy(float)
        y = group.combined_normalized_RMSE.to_numpy(float)
        rows.append({"outer_seed": int(seed), "method": method,
                     column: float(np.trapezoid(y, x) / (budgets[-1] - budgets[0]))})
    return pd.DataFrame(rows)


def _historical() -> pd.DataFrame:
    frames = []
    for method, path in comparator_sources().items():
        frame = pd.read_csv(path)
        frames.append(frame.loc[
            frame.outer_seed.isin(SEEDS) & frame.method.eq(method)
            & frame.active_label_count.isin(ALL_BUDGETS),
            ["outer_seed", "method", "round", "active_label_count", *METRICS],
        ])
    result = pd.concat(frames, ignore_index=True)
    _validate_grid(result, ALL_BUDGETS, COMPARATORS)
    return result


def _decision(curves: pd.DataFrame, primary: pd.DataFrame) -> dict:
    area = primary.pivot(index="outer_seed", columns="method", values="AULC_525_653")
    endpoint = curves.loc[curves.active_label_count.eq(653)].pivot(
        index="outer_seed", columns="method", values="combined_normalized_RMSE"
    )
    aulc_deltas = {method: area[METHOD] - area[method] for method in STRONG_COMPARATORS}
    endpoint_deltas = {method: endpoint[METHOD] - endpoint[method] for method in STRONG_COMPARATORS}
    beats_both = {method: bool((values < 0).all()) for method, values in aulc_deltas.items()}
    endpoint_vs_best = endpoint[METHOD] - endpoint[list(STRONG_COMPARATORS)].min(axis=1)
    marked_endpoint_deterioration = bool((endpoint_vs_best > ENDPOINT_DETERIORATION_TOLERANCE).any())
    all_three = all(beats_both.values())
    any_same = any(beats_both.values())
    loses_all_both = all(bool((values > ENDPOINT_DETERIORATION_TOLERANCE).all()) for values in aulc_deltas.values())
    no_endpoint_advantage = bool((endpoint[METHOD] >= endpoint[list(STRONG_COMPARATORS)].min(axis=1)).all())
    materially_opposite = {
        method: bool(
            np.sign(values.iloc[0]) != np.sign(values.iloc[1])
            and abs(values.iloc[0]) > MATERIAL_DIRECTION_TOLERANCE
            and abs(values.iloc[1]) > MATERIAL_DIRECTION_TOLERANCE
        ) for method, values in aulc_deltas.items()
    }
    if all_three and not marked_endpoint_deterioration:
        category = "STRONG_SUSTAINED_CW"
    elif any_same:
        category = "SUSTAINED_AGAINST_SOME"
    elif loses_all_both and no_endpoint_advantage:
        category = "STOP_CW"
    elif any(materially_opposite.values()):
        category = "MIXED_LATE_SIGNAL"
    else:
        category = "MIDSTAGE_ONLY"
    return {
        "status": "COMPLETE_DEVELOPMENTAL_CONTINUATION",
        "decision": category,
        "paired_AULC_525_653_deltas": {
            method: {str(seed): float(value) for seed, value in values.items()}
            for method, values in aulc_deltas.items()
        },
        "paired_NRMSE_653_deltas": {
            method: {str(seed): float(value) for seed, value in values.items()}
            for method, values in endpoint_deltas.items()
        },
        "two_seed_advantage": beats_both,
        "materially_opposite_seed_directions": materially_opposite,
        "endpoint_delta_vs_best_strong_comparator": {str(seed): float(value) for seed, value in endpoint_vs_best.items()},
        "endpoint_deterioration_tolerance": ENDPOINT_DETERIORATION_TOLERANCE,
        "marked_endpoint_deterioration": marked_endpoint_deterioration,
        "automatic_continuation_to_685_or_1005": False,
        "maxdet_to_cw_switch_implemented": False,
    }


def _paired(curves: pd.DataFrame, areas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    indices = {name: table.set_index(["outer_seed", "method"]) for name, table in areas.items()}
    curve = curves.set_index(["outer_seed", "method", "active_label_count"])
    for seed in SEEDS:
        for comparator in COMPARATORS:
            for name, table in indices.items():
                rows.append({"outer_seed": seed, "comparator": comparator, "metric": name,
                             "direction": "negative_is_CW_better",
                             "CW_minus_comparator": float(table.loc[(seed, METHOD), name] - table.loc[(seed, comparator), name])})
            for budget in (557, 589, 621, 653):
                rows.append({"outer_seed": seed, "comparator": comparator, "metric": f"NRMSE@{budget}",
                             "direction": "negative_is_CW_better",
                             "CW_minus_comparator": float(curve.loc[(seed, METHOD, budget), "combined_normalized_RMSE"] - curve.loc[(seed, comparator, budget), "combined_normalized_RMSE"])})
            for metric in ("V1_RMSE", "V2_RMSE", "V1_R2", "V2_R2"):
                direction = "positive_is_CW_better" if metric.endswith("R2") else "negative_is_CW_better"
                rows.append({"outer_seed": seed, "comparator": comparator, "metric": f"{metric}@653",
                             "direction": direction,
                             "CW_minus_comparator": float(curve.loc[(seed, METHOD, 653), metric] - curve.loc[(seed, comparator, 653), metric])})
    return pd.DataFrame(rows)


def _marginal(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    intervals = tuple(zip(ALL_BUDGETS[2:-1], ALL_BUDGETS[3:]))
    for (seed, method), group in curves.groupby(["outer_seed", "method"]):
        values = group.set_index("active_label_count").combined_normalized_RMSE
        for start, stop in intervals:
            before, after = float(values.loc[start]), float(values.loc[stop])
            rows.append({"outer_seed": int(seed), "method": method, "from_labels": start, "to_labels": stop,
                         "delta_NRMSE": after - before,
                         "relative_improvement": (before - after) / before})
    return pd.DataFrame(rows)


def _selection_overlap(study: Path) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for source_round, budget in zip((6, 7, 8, 9), (525, 557, 589, 621)):
            cw_path = study / f"runtime/seed_{seed}/{METHOD}/round_{source_round:02d}/acquisition_artifacts/selected_next_batch.csv"
            sources = {
                "gradient_maxdet": MAXDET_STUDY / f"runtime/seed_{seed}/gradient_maxdet/round_{source_round:02d}/acquisition/selected_next_batch.csv",
                "lcmd": BASELINE / f"runtime/seed_{seed}/lcmd/round_{source_round:02d}/acquisition_artifacts/selected_next_batch.csv",
            }
            cw = set(pd.read_csv(cw_path).sample_id.astype(str))
            if len(cw) != 32:
                raise RuntimeError(f"invalid CW batch for overlap: {cw_path}")
            for comparator, path in sources.items():
                other = set(pd.read_csv(path).sample_id.astype(str))
                if len(other) != 32:
                    raise RuntimeError(f"invalid comparator batch for overlap: {path}")
                count = len(cw & other)
                rows.append({"outer_seed": seed, "source_budget": budget, "source_round": source_round,
                             "method_a": METHOD, "method_b": comparator,
                             "overlap_count": count, "overlap_fraction": count / 32})
    return pd.DataFrame(rows)


def _cost(study: Path) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for evaluation_round in (7, 8, 9, 10):
            fit = json.loads((study / f"runtime/seed_{seed}/{METHOD}/round_{evaluation_round:02d}/model/fit_audit.json").read_text())
            source_round = evaluation_round - 1
            acquisition = json.loads((study / f"runtime/seed_{seed}/{METHOD}/round_{source_round:02d}/acquisition_artifacts/contract.json").read_text())
            training = float(fit["training_seconds"])
            gradient = float(acquisition["gradient_extraction_seconds"])
            selector = float(acquisition["selector_seconds"])
            rows.append({
                "outer_seed": seed, "method": METHOD, "source_round": source_round,
                "source_budget": ALL_BUDGETS[source_round], "evaluation_round": evaluation_round,
                "evaluation_budget": ALL_BUDGETS[evaluation_round], "checkpoint_reuse": False,
                "new_fit": True, "epochs_run": int(fit["epochs_run"]), "best_epoch": int(fit["best_epoch"]),
                "training_seconds": training, "gradient_extraction_seconds": gradient,
                "selector_seconds": selector, "total_elapsed_seconds": training + gradient + selector,
            })
    return pd.DataFrame(rows)


def _figures(study: Path, curves: pd.DataFrame, primary: pd.DataFrame, marginal: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = study / "figures"; figdir.mkdir(exist_ok=True)
    methods = ALL_METHODS
    colors = {METHOD: "#2ca02c", "gradient_maxdet": "#d62728", "hybrid": "#ff7f0e", "kernel_ivr": "#9467bd", "lcmd": "#1f77b4"}

    def curve_figure(subset: pd.DataFrame, budgets: tuple[int, ...], name: str) -> None:
        fig, ax = plt.subplots(figsize=(8.4, 4.9))
        for method in methods:
            group = subset.loc[subset.method.eq(method) & subset.active_label_count.isin(budgets)].sort_values("active_label_count")
            ax.plot(group.active_label_count, group.combined_normalized_RMSE, marker="o", label=DISPLAY_NAMES[method], color=colors[method])
        ax.set(xlabel="Active labels", ylabel="Combined normalized RMSE", xticks=list(budgets)); ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figdir / name, dpi=180); plt.close(fig)

    mean = curves.groupby(["method", "active_label_count"], as_index=False)[list(METRICS)].mean()
    curve_figure(mean, MID_LATE_BUDGETS, "cw_429_653.png")
    curve_figure(mean, PRIMARY_BUDGETS, "cw_525_653.png")
    for seed in SEEDS:
        curve_figure(curves.loc[curves.outer_seed.eq(seed)], MID_LATE_BUDGETS, f"per_seed_{seed}.png")
    areas = primary.groupby("method").AULC_525_653.mean().loc[list(methods)].sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.bar([DISPLAY_NAMES[m] for m in areas.index], areas.values, color=[colors[m] for m in areas.index]); ax.tick_params(axis="x", rotation=25); ax.set_ylabel("AULC 525-653 (lower is better)"); fig.tight_layout(); fig.savefig(figdir / "aulc_525_653.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    for method in methods:
        group = marginal.loc[marginal.method.eq(method) & marginal.from_labels.ge(429)].groupby("to_labels").delta_NRMSE.mean().sort_index()
        ax.plot(group.index, group.values, marker="o", label=DISPLAY_NAMES[method], color=colors[method])
    ax.axhline(0, color="black", linewidth=.8); ax.set(xlabel="Budget endpoint", ylabel="NRMSE(next) - NRMSE(previous)"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "marginal_improvement.png", dpi=180); plt.close(fig)


def _rank(table: pd.DataFrame, column: str) -> list[tuple[str, float]]:
    values = table.groupby("method")[column].mean().sort_values()
    return [(str(method), float(value)) for method, value in values.items()]


def _table_rows(table: pd.DataFrame, column: str) -> str:
    rows = []
    for label, subset in [*((str(seed), table.loc[table.outer_seed.eq(seed)]) for seed in SEEDS), ("Mean", table)]:
        values = subset.set_index("method")[column] if label != "Mean" else subset.groupby("method")[column].mean()
        rows.append("| " + label + " | " + " | ".join(f"{values[m]:.6f}" for m in ALL_METHODS) + " |")
    return "\n".join(rows)


def write_report(study: Path, extension: pd.DataFrame, access: pd.DataFrame) -> dict:
    study = Path(study); results = study / "results"; results.mkdir(exist_ok=True)
    source = pd.read_csv(SOURCE_STUDY / "results/learning_curve_metrics.csv")
    prefix = source.loc[
        source.outer_seed.isin(SEEDS) & source.method.eq(METHOD)
        & source.active_label_count.isin(ALL_BUDGETS[:7]),
        ["outer_seed", "method", "round", "active_label_count", *METRICS],
    ]
    _validate_grid(prefix, ALL_BUDGETS[:7], (METHOD,))
    _validate_grid(extension, ALL_BUDGETS[7:], (METHOD,))
    curves = pd.concat([prefix, extension, _historical()], ignore_index=True)
    _validate_grid(curves, ALL_BUDGETS)
    primary = _aulc(curves, PRIMARY_BUDGETS, "AULC_525_653")
    mid_late = _aulc(curves, MID_LATE_BUDGETS, "AULC_429_653")
    full = _aulc(curves, ALL_BUDGETS, "AULC_333_653")
    areas = {"AULC_525_653": primary, "AULC_429_653": mid_late, "AULC_333_653": full}
    paired = _paired(curves, areas)
    marginal = _marginal(curves)
    overlap = _selection_overlap(study)
    cost = _cost(study)
    decision = _decision(curves, primary)
    for name, table in {
        "learning_curve_metrics.csv": curves,
        "partial_aulc_525_653.csv": primary,
        "partial_aulc_429_653.csv": mid_late,
        "full_aulc_333_653.csv": full,
        "paired_comparison.csv": paired,
        "marginal_improvement.csv": marginal,
        "selection_overlap.csv": overlap,
        "compute_cost.csv": cost,
        "test_label_access_audit.csv": access,
    }.items():
        table.to_csv(results / name, index=False)
    atomic_json(study / "decision.json", decision)
    _figures(study, curves, primary, marginal)

    endpoint = curves.loc[curves.active_label_count.eq(653)].set_index(["outer_seed", "method"])
    endpoint_rows = []
    for seed in SEEDS:
        for method in ALL_METHODS:
            row = endpoint.loc[(seed, method)]
            endpoint_rows.append(f"| {seed} | {DISPLAY_NAMES[method]} | {row.combined_normalized_RMSE:.6f} | {row.V1_RMSE:.6f} | {row.V1_MAE:.6f} | {row.V1_R2:.6f} | {row.V2_RMSE:.6f} | {row.V2_MAE:.6f} | {row.V2_R2:.6f} |")
    ranking_lines = []
    for label, table, column in (("AULC 525-653", primary, "AULC_525_653"), ("AULC 429-653", mid_late, "AULC_429_653"), ("AULC 333-653", full, "AULC_333_653")):
        ranking_lines.append(f"- {label}: " + ", ".join(f"{DISPLAY_NAMES[m]} {v:.6f}" for m, v in _rank(table, column)))
    endpoint_rank = curves.loc[curves.active_label_count.eq(653)].groupby("method").combined_normalized_RMSE.mean().sort_values()
    ranking_lines.append("- NRMSE @653: " + ", ".join(f"{DISPLAY_NAMES[m]} {v:.6f}" for m, v in endpoint_rank.items()))
    late_cw = marginal.loc[marginal.method.eq(METHOD) & marginal.from_labels.ge(525)]
    pattern = " / ".join(
        f"seed {seed}: " + ", ".join(f"{int(r.from_labels)}->{int(r.to_labels)} {r.delta_NRMSE:+.6f}" for _, r in late_cw.loc[late_cw.outer_seed.eq(seed)].iterrows())
        for seed in SEEDS
    )
    cw_525 = curves.loc[curves.method.eq(METHOD) & curves.active_label_count.eq(525)].set_index("outer_seed")
    cw_653 = curves.loc[curves.method.eq(METHOD) & curves.active_label_count.eq(653)].set_index("outer_seed")
    component_lines = []
    for seed in SEEDS:
        component_lines.append(f"- Seed {seed}: V1 RMSE {cw_525.loc[seed, 'V1_RMSE']:.6f}->{cw_653.loc[seed, 'V1_RMSE']:.6f} ({cw_653.loc[seed, 'V1_RMSE']-cw_525.loc[seed, 'V1_RMSE']:+.6f}); V2 RMSE {cw_525.loc[seed, 'V2_RMSE']:.6f}->{cw_653.loc[seed, 'V2_RMSE']:.6f} ({cw_653.loc[seed, 'V2_RMSE']-cw_525.loc[seed, 'V2_RMSE']:+.6f}).")
    overlap_summary = overlap.groupby("method_b").agg(mean_overlap=("overlap_fraction", "mean"), max_overlap=("overlap_fraction", "max"))
    cost_total = cost[["training_seconds", "gradient_extraction_seconds", "selector_seconds", "total_elapsed_seconds"]].sum()
    (study / "FINAL_REPORT.md").write_text(f"""# Pure Center/Width-LCMD continuation: L525 to L653

This developmental continuation starts from each seed's exact frozen pure-CW L525 state. It performs four current-checkpoint CW acquisitions and four scratch QGeoGNN-V2 fits per seed, then hard-stops at L653. Historical comparators are read-only; no other acquisition method was trained. Test truth remained inaccessible until all eight new checkpoint/prediction pairs were globally frozen.

## Primary AULC 525-653

| Seed | Center/Width-LCMD | Gradient-MaxDet | Hybrid | Kernel-IVR | Gradient-LCMD |
|---:|---:|---:|---:|---:|---:|
{_table_rows(primary, 'AULC_525_653')}

## AULC 429-653

| Seed | Center/Width-LCMD | Gradient-MaxDet | Hybrid | Kernel-IVR | Gradient-LCMD |
|---:|---:|---:|---:|---:|---:|
{_table_rows(mid_late, 'AULC_429_653')}

## AULC 333-653

Every method/seed integral contains exactly the 11 preregistered points from 333 through 653.

| Seed | Center/Width-LCMD | Gradient-MaxDet | Hybrid | Kernel-IVR | Gradient-LCMD |
|---:|---:|---:|---:|---:|---:|
{_table_rows(full, 'AULC_333_653')}

## Fixed-budget endpoint

| Seed | Method | NRMSE | V1 RMSE | V1 MAE | V1 R2 | V2 RMSE | V2 MAE | V2 R2 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(endpoint_rows)}

## Rankings

{chr(10).join(ranking_lines)}

## Decision and interpretation

Decision: **{decision['decision']}**. The frozen two-seed AULC advantage flags versus MaxDet, Hybrid, and IVR are `{decision['two_seed_advantage']}`. Paired deltas for every comparator and metric are in `results/paired_comparison.csv`; NRMSE/AULC use negative-is-CW-better, while R2 uses positive-is-CW-better.

CW late marginal deltas are: {pattern}. The 493->525 seed-6101 rebound is therefore evaluated against the actual 525->653 sequence rather than a single endpoint.

Component changes from L525 to L653:
{chr(10).join(component_lines)}

Selection overlap is diagnostic only. Against Gradient-MaxDet, mean/max overlap fractions are {overlap_summary.loc['gradient_maxdet', 'mean_overlap']:.4f}/{overlap_summary.loc['gradient_maxdet', 'max_overlap']:.4f}; against Gradient-LCMD they are {overlap_summary.loc['lcmd', 'mean_overlap']:.4f}/{overlap_summary.loc['lcmd', 'max_overlap']:.4f}.

Compute cost was exactly 8 new evaluation fits and 0 ensemble fits. Totals: training {cost_total.training_seconds:.1f}s, gradient extraction {cost_total.gradient_extraction_seconds:.1f}s, selection {cost_total.selector_seconds:.1f}s, recorded combined elapsed {cost_total.total_elapsed_seconds:.1f}s. Per-seed and per-round values, epochs, and checkpoint reuse flags are in `results/compute_cost.csv`.

No MaxDet-to-CW switch was implemented: its L_t state differs from pure CW. The stage pattern may motivate a separately preregistered matched-state switch study, but this continuation cannot establish switch efficacy. No automatic continuation to 685 or 1005 is authorized; any further pure-CW experiment requires a new protocol and cost/benefit decision.
""")
    files = [path for path in study.rglob("*") if path.is_file() and "runtime" not in path.parts and path.name != "artifact_manifest.json"]
    atomic_json(study / "artifact_manifest.json", {
        "status": "COMPLETE_DEVELOPMENTAL_CONTINUATION", "hard_stop_active_labels": 653,
        "new_evaluation_fits": 8, "ensemble_fits": 0,
        "files": {str(path.relative_to(study)): sha256_file(path) for path in sorted(files)},
    })
    return {"status": "COMPLETE_DEVELOPMENTAL_CONTINUATION", "decision": decision}
