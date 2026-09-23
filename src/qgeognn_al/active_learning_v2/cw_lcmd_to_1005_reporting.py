"""Reporting for the frozen pure-CW L653-to-L1005 continuation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import sha256_file
from ..training.predictor import atomic_json
from .cw_lcmd_to_1005_study import (
    ALL_BUDGETS, ALL_METHODS, COMPARATORS, DISPLAY_NAMES, FROM_429_BUDGETS,
    FROM_525_BUDGETS, MAXDET_STUDY, METHOD, PRIMARY_BUDGETS, SEEDS,
    SOURCE_STUDY, STUDY, comparator_sources,
)
from .maxdet_study import BASELINE


METRICS = (
    "combined_normalized_RMSE", "V1_RMSE", "V1_MAE", "V1_R2",
    "V2_RMSE", "V2_MAE", "V2_R2",
)
STRONG_COMPARATORS = ("gradient_maxdet", "hybrid", "kernel_ivr")
MATERIAL_DIRECTION_TOLERANCE = 0.01
CLEAR_LOSS_TOLERANCE = 0.03


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
    subset = frame.loc[frame.active_label_count.isin(budgets)]
    _validate_grid(subset, budgets)
    rows = []
    for (seed, method), group in subset.groupby(["outer_seed", "method"]):
        group = group.sort_values("active_label_count")
        x = group.active_label_count.to_numpy(float)
        y = group.combined_normalized_RMSE.to_numpy(float)
        rows.append({"outer_seed": int(seed), "method": method,
                     column: float(np.trapezoid(y, x) / (budgets[-1] - budgets[0]))})
    return pd.DataFrame(rows)


def _historical() -> pd.DataFrame:
    frames = []
    for method, path in comparator_sources().items():
        table = pd.read_csv(path)
        frames.append(table.loc[
            table.outer_seed.isin(SEEDS) & table.method.eq(method)
            & table.active_label_count.isin(ALL_BUDGETS),
            ["outer_seed", "method", "round", "active_label_count", *METRICS],
        ])
    result = pd.concat(frames, ignore_index=True)
    _validate_grid(result, ALL_BUDGETS, COMPARATORS)
    return result


def _decision(curves: pd.DataFrame, primary: pd.DataFrame) -> dict:
    area = primary.pivot(index="outer_seed", columns="method", values="AULC_653_1005")
    endpoint = curves.loc[curves.active_label_count.eq(1005)].pivot(
        index="outer_seed", columns="method", values="combined_normalized_RMSE"
    )
    deltas = {method: area[METHOD] - area[method] for method in STRONG_COMPARATORS}
    endpoint_deltas = {method: endpoint[METHOD] - endpoint[method] for method in STRONG_COMPARATORS}
    beats_both = {method: bool((values < 0).all()) for method, values in deltas.items()}
    opposite = {
        method: bool(np.sign(values.iloc[0]) != np.sign(values.iloc[1])
                     and abs(values.iloc[0]) > MATERIAL_DIRECTION_TOLERANCE
                     and abs(values.iloc[1]) > MATERIAL_DIRECTION_TOLERANCE)
        for method, values in deltas.items()
    }
    loses_all = all(bool((values > CLEAR_LOSS_TOLERANCE).all()) for values in deltas.values())
    no_endpoint_advantage = bool((endpoint[METHOD] >= endpoint[list(STRONG_COMPARATORS)].min(axis=1)).all())
    if all(beats_both.values()):
        category = "SUSTAINED_TO_1005"
    elif any(beats_both.values()):
        category = "SUSTAINED_AGAINST_SOME"
    elif loses_all and no_endpoint_advantage:
        category = "STOP_CW"
    elif any(opposite.values()):
        category = "MIXED_LONG_HORIZON"
    else:
        category = "NO_CONSISTENT_ADVANTAGE"
    return {
        "status": "COMPLETE_DEVELOPMENTAL_CONTINUATION", "decision": category,
        "paired_AULC_653_1005_deltas": {
            method: {str(seed): float(value) for seed, value in values.items()}
            for method, values in deltas.items()
        },
        "paired_NRMSE_1005_deltas": {
            method: {str(seed): float(value) for seed, value in values.items()}
            for method, values in endpoint_deltas.items()
        },
        "two_seed_advantage": beats_both, "materially_opposite_seed_directions": opposite,
        "hard_stop_active_labels": 1005, "further_experiment_implemented": False,
    }


def _paired(curves: pd.DataFrame, areas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []; curve = curves.set_index(["outer_seed", "method", "active_label_count"])
    indexed = {name: table.set_index(["outer_seed", "method"]) for name, table in areas.items()}
    for seed in SEEDS:
        for comparator in COMPARATORS:
            for name, table in indexed.items():
                rows.append({"outer_seed": seed, "comparator": comparator, "metric": name,
                             "direction": "negative_is_CW_better",
                             "CW_minus_comparator": float(table.loc[(seed, METHOD), name] - table.loc[(seed, comparator), name])})
            for budget in ALL_BUDGETS[11:]:
                rows.append({"outer_seed": seed, "comparator": comparator, "metric": f"NRMSE@{budget}",
                             "direction": "negative_is_CW_better",
                             "CW_minus_comparator": float(curve.loc[(seed, METHOD, budget), "combined_normalized_RMSE"] - curve.loc[(seed, comparator, budget), "combined_normalized_RMSE"])})
            for metric in ("V1_RMSE", "V2_RMSE", "V1_R2", "V2_R2"):
                rows.append({"outer_seed": seed, "comparator": comparator, "metric": f"{metric}@1005",
                             "direction": "positive_is_CW_better" if metric.endswith("R2") else "negative_is_CW_better",
                             "CW_minus_comparator": float(curve.loc[(seed, METHOD, 1005), metric] - curve.loc[(seed, comparator, 1005), metric])})
    return pd.DataFrame(rows)


def _marginal(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, method), group in curves.groupby(["outer_seed", "method"]):
        values = group.set_index("active_label_count").combined_normalized_RMSE
        for start, stop in zip(ALL_BUDGETS[2:-1], ALL_BUDGETS[3:]):
            before, after = float(values.loc[start]), float(values.loc[stop])
            rows.append({"outer_seed": int(seed), "method": method, "from_labels": start, "to_labels": stop,
                         "delta_NRMSE": after - before, "relative_improvement": (before - after) / before})
    return pd.DataFrame(rows)


def _selection_overlap(study: Path) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for source_round, budget in zip(range(10, 21), ALL_BUDGETS[10:21]):
            cw_path = study / f"runtime/seed_{seed}/{METHOD}/round_{source_round:02d}/acquisition_artifacts/selected_next_batch.csv"
            sources = {
                "gradient_maxdet": MAXDET_STUDY / f"runtime/seed_{seed}/gradient_maxdet/round_{source_round:02d}/acquisition/selected_next_batch.csv",
                "lcmd": BASELINE / f"runtime/seed_{seed}/lcmd/round_{source_round:02d}/acquisition_artifacts/selected_next_batch.csv",
            }
            cw = set(pd.read_csv(cw_path).sample_id.astype(str))
            if len(cw) != 32: raise RuntimeError(f"invalid CW overlap batch: {cw_path}")
            for comparator, path in sources.items():
                other = set(pd.read_csv(path).sample_id.astype(str)); count = len(cw & other)
                if len(other) != 32: raise RuntimeError(f"invalid comparator overlap batch: {path}")
                rows.append({"outer_seed": seed, "source_budget": budget, "source_round": source_round,
                             "method_a": METHOD, "method_b": comparator,
                             "overlap_count": count, "overlap_fraction": count / 32})
    return pd.DataFrame(rows)


def _cost(study: Path) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for evaluation_round in range(11, 22):
            fit = json.loads((study / f"runtime/seed_{seed}/{METHOD}/round_{evaluation_round:02d}/model/fit_audit.json").read_text())
            source_round = evaluation_round - 1
            acquisition = json.loads((study / f"runtime/seed_{seed}/{METHOD}/round_{source_round:02d}/acquisition_artifacts/contract.json").read_text())
            training = float(fit["training_seconds"]); gradient = float(acquisition["gradient_extraction_seconds"]); selector = float(acquisition["selector_seconds"])
            rows.append({"outer_seed": seed, "method": METHOD, "source_round": source_round,
                         "source_budget": ALL_BUDGETS[source_round], "evaluation_round": evaluation_round,
                         "evaluation_budget": ALL_BUDGETS[evaluation_round], "checkpoint_reuse": False,
                         "new_fit": True, "epochs_run": int(fit["epochs_run"]), "best_epoch": int(fit["best_epoch"]),
                         "training_seconds": training, "gradient_extraction_seconds": gradient,
                         "selector_seconds": selector, "total_elapsed_seconds": training + gradient + selector})
    return pd.DataFrame(rows)


def _figures(study: Path, curves: pd.DataFrame, primary: pd.DataFrame, marginal: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = study / "figures"; figdir.mkdir(exist_ok=True)
    colors = {METHOD: "#2ca02c", "gradient_maxdet": "#d62728", "hybrid": "#ff7f0e", "kernel_ivr": "#9467bd", "lcmd": "#1f77b4"}

    def curve_plot(subset: pd.DataFrame, budgets: tuple[int, ...], name: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 5.2))
        for method in ALL_METHODS:
            group = subset.loc[subset.method.eq(method) & subset.active_label_count.isin(budgets)].sort_values("active_label_count")
            ax.plot(group.active_label_count, group.combined_normalized_RMSE, marker="o", markersize=4,
                    label=DISPLAY_NAMES[method], color=colors[method])
        ax.set(xlabel="Active labels", ylabel="Combined normalized RMSE", xticks=list(budgets)[::2]); ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figdir / name, dpi=180); plt.close(fig)

    mean = curves.groupby(["method", "active_label_count"], as_index=False)[list(METRICS)].mean()
    curve_plot(mean, PRIMARY_BUDGETS, "cw_653_1005.png")
    curve_plot(mean, ALL_BUDGETS, "cw_333_1005.png")
    for seed in SEEDS:
        curve_plot(curves.loc[curves.outer_seed.eq(seed)], PRIMARY_BUDGETS, f"per_seed_{seed}_653_1005.png")
    areas = primary.groupby("method").AULC_653_1005.mean().loc[list(ALL_METHODS)].sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.bar([DISPLAY_NAMES[m] for m in areas.index], areas.values,
        color=[colors[m] for m in areas.index]); ax.tick_params(axis="x", rotation=25); ax.set_ylabel("AULC 653-1005 (lower is better)")
    fig.tight_layout(); fig.savefig(figdir / "aulc_653_1005.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for method in ALL_METHODS:
        group = marginal.loc[marginal.method.eq(method) & marginal.from_labels.ge(653)].groupby("to_labels").delta_NRMSE.mean().sort_index()
        ax.plot(group.index, group.values, marker="o", label=DISPLAY_NAMES[method], color=colors[method])
    ax.axhline(0, color="black", linewidth=.8); ax.set(xlabel="Budget endpoint", ylabel="NRMSE(next) - NRMSE(previous)"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "marginal_653_1005.png", dpi=180); plt.close(fig)


def _table_rows(table: pd.DataFrame, column: str) -> str:
    rows = []
    for label, subset in [*((str(seed), table.loc[table.outer_seed.eq(seed)]) for seed in SEEDS), ("Mean", table)]:
        values = subset.set_index("method")[column] if label != "Mean" else subset.groupby("method")[column].mean()
        rows.append("| " + label + " | " + " | ".join(f"{values[m]:.6f}" for m in ALL_METHODS) + " |")
    return "\n".join(rows)


def _ranking(table: pd.DataFrame, column: str) -> str:
    values = table.groupby("method")[column].mean().sort_values()
    return ", ".join(f"{DISPLAY_NAMES[method]} {value:.6f}" for method, value in values.items())


def refresh_manifest(study: Path = STUDY) -> None:
    study = Path(study)
    files = [path for path in study.rglob("*") if path.is_file() and "runtime" not in path.parts and path.name != "artifact_manifest.json"]
    atomic_json(study / "artifact_manifest.json", {
        "status": "COMPLETE_DEVELOPMENTAL_CONTINUATION", "hard_stop_active_labels": 1005,
        "new_evaluation_fits": 22, "ensemble_fits": 0,
        "files": {str(path.relative_to(study)): sha256_file(path) for path in sorted(files)},
    })


def write_report(study: Path, extension: pd.DataFrame, access: pd.DataFrame) -> dict:
    study = Path(study); results = study / "results"; results.mkdir(exist_ok=True)
    source = pd.read_csv(SOURCE_STUDY / "results/learning_curve_metrics.csv")
    prefix = source.loc[
        source.outer_seed.isin(SEEDS) & source.method.eq(METHOD)
        & source.active_label_count.isin(ALL_BUDGETS[:11]),
        ["outer_seed", "method", "round", "active_label_count", *METRICS],
    ]
    _validate_grid(prefix, ALL_BUDGETS[:11], (METHOD,)); _validate_grid(extension, ALL_BUDGETS[11:], (METHOD,))
    curves = pd.concat([prefix, extension, _historical()], ignore_index=True); _validate_grid(curves, ALL_BUDGETS)
    primary = _aulc(curves, PRIMARY_BUDGETS, "AULC_653_1005")
    from525 = _aulc(curves, FROM_525_BUDGETS, "AULC_525_1005")
    from429 = _aulc(curves, FROM_429_BUDGETS, "AULC_429_1005")
    full = _aulc(curves, ALL_BUDGETS, "AULC_333_1005")
    areas = {"AULC_653_1005": primary, "AULC_525_1005": from525,
             "AULC_429_1005": from429, "AULC_333_1005": full}
    paired = _paired(curves, areas); marginal = _marginal(curves)
    overlap = _selection_overlap(study); cost = _cost(study); decision = _decision(curves, primary)
    tables = {
        "learning_curve_metrics.csv": curves, "partial_aulc_653_1005.csv": primary,
        "partial_aulc_525_1005.csv": from525, "partial_aulc_429_1005.csv": from429,
        "full_aulc_333_1005.csv": full, "paired_comparison.csv": paired,
        "marginal_improvement.csv": marginal, "selection_overlap.csv": overlap,
        "compute_cost.csv": cost, "test_label_access_audit.csv": access,
    }
    for name, table in tables.items(): table.to_csv(results / name, index=False)
    atomic_json(study / "decision.json", decision); _figures(study, curves, primary, marginal)

    endpoint = curves.loc[curves.active_label_count.eq(1005)].set_index(["outer_seed", "method"])
    endpoint_rows = []
    for seed in SEEDS:
        for method in ALL_METHODS:
            row = endpoint.loc[(seed, method)]
            endpoint_rows.append(f"| {seed} | {DISPLAY_NAMES[method]} | {row.combined_normalized_RMSE:.6f} | {row.V1_RMSE:.6f} | {row.V1_MAE:.6f} | {row.V1_R2:.6f} | {row.V2_RMSE:.6f} | {row.V2_MAE:.6f} | {row.V2_R2:.6f} |")
    primary_pivot = primary.pivot(index="outer_seed", columns="method", values="AULC_653_1005")
    paired_lines = []
    for comparator in COMPARATORS:
        values = primary_pivot[METHOD] - primary_pivot[comparator]
        paired_lines.append(f"- CW minus {DISPLAY_NAMES[comparator]}: seed 157 {values.loc[157]:+.6f}; seed 6101 {values.loc[6101]:+.6f}.")
    cw_marginal = marginal.loc[marginal.method.eq(METHOD) & marginal.from_labels.ge(653)]
    marginal_lines = []
    for seed in SEEDS:
        group = cw_marginal.loc[cw_marginal.outer_seed.eq(seed)]
        marginal_lines.append(f"- Seed {seed}: " + ", ".join(f"{int(row.from_labels)}->{int(row.to_labels)} {row.delta_NRMSE:+.6f}" for _, row in group.iterrows()))
    cw653 = curves.loc[curves.method.eq(METHOD) & curves.active_label_count.eq(653)].set_index("outer_seed")
    cw1005 = curves.loc[curves.method.eq(METHOD) & curves.active_label_count.eq(1005)].set_index("outer_seed")
    component_lines = []
    for seed in SEEDS:
        component_lines.append(f"- Seed {seed}: V1 RMSE {cw653.loc[seed, 'V1_RMSE']:.6f}->{cw1005.loc[seed, 'V1_RMSE']:.6f} ({cw1005.loc[seed, 'V1_RMSE']-cw653.loc[seed, 'V1_RMSE']:+.6f}); V2 RMSE {cw653.loc[seed, 'V2_RMSE']:.6f}->{cw1005.loc[seed, 'V2_RMSE']:.6f} ({cw1005.loc[seed, 'V2_RMSE']-cw653.loc[seed, 'V2_RMSE']:+.6f}).")
    overlap_summary = overlap.groupby("method_b").agg(mean_overlap=("overlap_fraction", "mean"), max_overlap=("overlap_fraction", "max"))
    total = cost[["training_seconds", "gradient_extraction_seconds", "selector_seconds", "total_elapsed_seconds"]].sum()
    endpoint_rank = curves.loc[curves.active_label_count.eq(1005)].groupby("method").combined_normalized_RMSE.mean().sort_values()
    endpoint_ranking = ", ".join(f"{DISPLAY_NAMES[m]} {v:.6f}" for m, v in endpoint_rank.items())
    (study / "FINAL_REPORT.md").write_text(f"""# Pure Center/Width-LCMD continuation: L653 to L1005

This developmental continuation starts from each seed's exact frozen pure-CW L653 state, performs eleven current-checkpoint CW acquisitions and eleven scratch QGeoGNN-V2 fits per seed, and hard-stops at L1005 without an L1005 acquisition. Historical comparators are read-only. Test truth remained inaccessible until all 22 new checkpoint/prediction pairs were globally frozen.

## Primary AULC 653-1005

| Seed | Center/Width-LCMD | Gradient-MaxDet | Hybrid | Kernel-IVR | Gradient-LCMD |
|---:|---:|---:|---:|---:|---:|
{_table_rows(primary, 'AULC_653_1005')}

Paired primary deltas (negative means CW is better):
{chr(10).join(paired_lines)}

## Longer-horizon AULCs

- AULC 525-1005: {_ranking(from525, 'AULC_525_1005')}
- AULC 429-1005: {_ranking(from429, 'AULC_429_1005')}
- AULC 333-1005: {_ranking(full, 'AULC_333_1005')}

Every full AULC uses exactly all 22 budgets from 333 through 1005 for every seed and method.

## Endpoint at L1005

| Seed | Method | NRMSE | V1 RMSE | V1 MAE | V1 R2 | V2 RMSE | V2 MAE | V2 R2 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(endpoint_rows)}

Two-seed mean NRMSE@1005 ranking: {endpoint_ranking}.

## Decision and trajectory analysis

Decision: **{decision['decision']}**. Two-seed AULC 653-1005 advantage flags versus MaxDet, Hybrid, and IVR are `{decision['two_seed_advantage']}`. Endpoint rankings are reported separately and do not override curve efficiency.

CW marginal NRMSE changes after L653:
{chr(10).join(marginal_lines)}

CW component changes from L653 to L1005:
{chr(10).join(component_lines)}

Selection overlap remained diagnostic only. Versus Gradient-MaxDet, mean/max fractions are {overlap_summary.loc['gradient_maxdet', 'mean_overlap']:.4f}/{overlap_summary.loc['gradient_maxdet', 'max_overlap']:.4f}; versus Gradient-LCMD they are {overlap_summary.loc['lcmd', 'mean_overlap']:.4f}/{overlap_summary.loc['lcmd', 'max_overlap']:.4f}.

Compute cost was exactly 22 new evaluation fits and 0 ensemble fits: training {total.training_seconds:.1f}s, gradient extraction {total.gradient_extraction_seconds:.1f}s, selection {total.selector_seconds:.1f}s, recorded combined elapsed {total.total_elapsed_seconds:.1f}s. Per-round epochs and timing are in `results/compute_cost.csv`.

This study ends at the established L1005 endpoint. No additional method search, switch experiment, or later acquisition was run.
""")
    refresh_manifest(study)
    return {"status": "COMPLETE_DEVELOPMENTAL_CONTINUATION", "decision": decision}
