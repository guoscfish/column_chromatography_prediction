"""Reporting for the CW-LCMD 429--525 continuation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import sha256_file
from ..training.predictor import atomic_json
from .cw_lcmd_extension_study import (
    ACTIVE_LABEL_BUDGETS, BATCH_SIZE, COMPARATORS, CONTINUATION_BUDGETS,
    DISPLAY_NAMES, METHOD, SEEDS, STUDY, SOURCE_STUDY, MAXDET_STUDY, IVR_STUDY, BASELINE,
)


METRICS = (
    "combined_normalized_RMSE", "V1_RMSE", "V1_MAE", "V1_R2",
    "V2_RMSE", "V2_MAE", "V2_R2",
)


def _historical() -> pd.DataFrame:
    sources = [
        (BASELINE / "results/learning_curve_metrics.csv", ("hybrid", "lcmd")),
        (MAXDET_STUDY / "results/learning_curve_metrics.csv", ("gradient_maxdet",)),
        (IVR_STUDY / "results/learning_curve_metrics.csv", ("kernel_ivr",)),
    ]
    frames = []
    for path, methods in sources:
        frame = pd.read_csv(path)
        frame = frame.loc[
            frame.outer_seed.isin(SEEDS) & frame.method.isin(methods)
            & frame.active_label_count.isin(CONTINUATION_BUDGETS),
            ["outer_seed", "method", "round", "active_label_count", *METRICS],
        ]
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    expected = len(SEEDS) * len(CONTINUATION_BUDGETS)
    for method in COMPARATORS:
        if len(result.loc[result.method.eq(method)]) != expected:
            raise RuntimeError(f"historical continuation comparator incomplete: {method}")
    return result


def _aulc(frame: pd.DataFrame, start: int, stop: int, name: str) -> pd.DataFrame:
    rows = []
    for (seed, method), group in frame.groupby(["outer_seed", "method"]):
        group = group.sort_values("active_label_count")
        x = group.active_label_count.to_numpy(float)
        y = group.combined_normalized_RMSE.to_numpy(float)
        rows.append({"outer_seed": int(seed), "method": method, name: float(np.trapezoid(y, x) / (stop - start))})
    return pd.DataFrame(rows)


def _batches(study: Path, seed: int, method: str, source_round: int) -> set[str]:
    if method == METHOD:
        path = study / "runtime" / f"seed_{seed}" / METHOD / f"round_{source_round:02d}/acquisition_artifacts/selected_next_batch.csv"
    elif method == "lcmd":
        path = BASELINE / "runtime" / f"seed_{seed}/lcmd/round_{source_round:02d}/acquisition_artifacts/selected_next_batch.csv"
    elif method == "gradient_maxdet":
        path = MAXDET_STUDY / "runtime" / f"seed_{seed}/gradient_maxdet/round_{source_round:02d}/acquisition/selected_next_batch.csv"
    else:
        raise ValueError(method)
    return set(pd.read_csv(path).sample_id.astype(str))


def _selection_overlap(study: Path) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for source_round in (3, 4, 5):
            for comparator in ("lcmd", "gradient_maxdet"):
                left = _batches(study, seed, METHOD, source_round)
                right = _batches(study, seed, comparator, source_round)
                count = len(left & right)
                rows.append({
                    "outer_seed": seed, "source_round": source_round,
                    "acquisition_round": source_round + 1,
                    "active_labels_before_acquisition": 429 + (source_round - 3) * 32,
                    "method_a": METHOD, "method_b": comparator,
                    "overlap_count": count, "overlap_fraction": count / BATCH_SIZE,
                })
    return pd.DataFrame(rows)


def _marginal(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in frame.groupby("method"):
        for seed, seed_group in group.groupby("outer_seed"):
            seed_group = seed_group.sort_values("active_label_count")
            values = dict(zip(seed_group.active_label_count, seed_group.combined_normalized_RMSE))
            for start, stop in zip((397, 429, 461, 493), (429, 461, 493, 525)):
                if start not in values or stop not in values:
                    continue
                delta = float(values[stop] - values[start])
                rows.append({"outer_seed": int(seed), "method": method,
                             "from_labels": start, "to_labels": stop,
                             "delta_NRMSE": delta,
                             "relative_improvement": float((values[start] - values[stop]) / values[start])})
    return pd.DataFrame(rows)


def _decision(curves: pd.DataFrame, mid: pd.DataFrame) -> dict:
    pivot = mid.pivot(index="outer_seed", columns="method", values="AULC_429_525")
    end = curves.loc[curves.active_label_count.eq(525)].pivot(index="outer_seed", columns="method", values="combined_normalized_RMSE")
    mean_mid = mid.groupby("method").AULC_429_525.mean()
    strongest = min(("gradient_maxdet", "hybrid"), key=lambda m: mean_mid[m])
    a = pivot[METHOD] - pivot[strongest]
    e = end[METHOD] - end[strongest]
    both_better = bool((a < 0).all())
    endpoint_better = bool((e <= 0).all())
    close = bool(abs(float(end[METHOD].mean() - end[strongest].mean())) <= 0.01)
    if both_better and endpoint_better and close:
        category = "STRONG_MIDSTAGE_SIGNAL"
    elif (mean_mid[METHOD] < mean_mid[strongest] and a.max() <= 0.05 and abs(float(e.mean())) <= 0.03):
        category = "PROMISING"
    elif (a > 0).all() and (e > 0).all():
        category = "STOP"
    else:
        category = "MIXED"
    return {
        "status": "COMPLETE_DEVELOPMENTAL_CONTINUATION",
        "decision": category,
        "strongest_baseline": strongest,
        "paired_AULC_429_525_deltas": {str(k): float(v) for k, v in a.items()},
        "paired_NRMSE_525_deltas": {str(k): float(v) for k, v in e.items()},
        "mean_AULC_429_525": {k: float(v) for k, v in mean_mid.items()},
        "automatic_continuation_to_653": False,
        "maxdet_to_cw_switch_implemented": False,
    }


def _figures(study: Path, curves: pd.DataFrame, marginal: pd.DataFrame, mid: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = study / "figures"; figdir.mkdir(exist_ok=True)
    methods = (METHOD, "gradient_maxdet", "hybrid", "lcmd", "kernel_ivr")
    colors = {METHOD: "#2ca02c", "gradient_maxdet": "#d62728", "hybrid": "#ff7f0e", "lcmd": "#1f77b4", "kernel_ivr": "#9467bd"}
    for scope, subset in (("mean", curves.groupby(["method", "active_label_count"], as_index=False)[list(METRICS)].mean()),
                          ("seed_157", curves.loc[curves.outer_seed.eq(157)]),
                          ("seed_6101", curves.loc[curves.outer_seed.eq(6101)])):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for method in methods:
            g = subset.loc[subset.method.eq(method)].sort_values("active_label_count")
            ax.plot(g.active_label_count, g.combined_normalized_RMSE, marker="o", label=DISPLAY_NAMES[method], color=colors[method])
        ax.set(xlabel="Active labels", ylabel="Combined normalized RMSE", xticks=[397, 429, 461, 493, 525]); ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figdir / ("cw_midstage_continuation_397_525.png" if scope == "mean" else f"per_seed_cw_continuation_{scope[-3:]}.png"), dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method in methods:
        g = marginal.loc[marginal.method.eq(method)].groupby("to_labels").delta_NRMSE.mean().sort_index()
        ax.plot(g.index, g.values, marker="o", label=DISPLAY_NAMES[method], color=colors[method])
    ax.axhline(0, color="black", linewidth=.8); ax.set(xlabel="New budget endpoint", ylabel="NRMSE(t+1) - NRMSE(t)"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "marginal_nrmse_improvement_per_batch.png", dpi=180); plt.close(fig)
    areas = mid.groupby("method").AULC_429_525.mean().loc[list(methods)].sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.bar([DISPLAY_NAMES[m] for m in areas.index], areas.values, color=[colors[m] for m in areas.index]); ax.tick_params(axis="x", rotation=25); ax.set_ylabel("AULC 429-525 (lower is better)"); fig.tight_layout(); fig.savefig(figdir / "aulc_429_525_comparison.png", dpi=180); plt.close(fig)


def write_report(study: Path, extension_curves: pd.DataFrame, access: pd.DataFrame) -> dict:
    study = Path(study); results = study / "results"; results.mkdir(exist_ok=True)
    historical = _historical()
    source = pd.read_csv(SOURCE_STUDY / "results/learning_curve_metrics.csv")
    cw_prefix = source.loc[source.outer_seed.isin(SEEDS) & source.method.eq(METHOD) & source.active_label_count.isin((397, 429)), ["outer_seed", "method", "round", "active_label_count", *METRICS]]
    curves = pd.concat([cw_prefix, extension_curves, historical], ignore_index=True).drop_duplicates(["outer_seed", "method", "active_label_count"])
    if len(extension_curves) != len(SEEDS) * 3 or extension_curves.duplicated(["outer_seed", "active_label_count"]).any():
        raise RuntimeError("extension metric matrix incomplete")
    mid = _aulc(curves.loc[curves.active_label_count.isin(CONTINUATION_BUDGETS)], 429, 525, "AULC_429_525")
    full = _aulc(curves.loc[curves.active_label_count.isin(ACTIVE_LABEL_BUDGETS)], 333, 525, "AULC_333_525")
    paired_rows = []
    for seed in SEEDS:
        for comparator in COMPARATORS:
            c = curves.loc[curves.outer_seed.eq(seed) & curves.method.isin((METHOD, comparator))].set_index(["method", "active_label_count"])
            m = mid.set_index(["outer_seed", "method"])
            f = full.set_index(["outer_seed", "method"])
            for budget in (461, 493, 525):
                paired_rows.append({"outer_seed": seed, "comparator": comparator, "metric": f"NRMSE@{budget}", "CW_minus_comparator": float(c.loc[(METHOD, budget), "combined_normalized_RMSE"] - c.loc[(comparator, budget), "combined_normalized_RMSE"])})
            paired_rows.extend([
                {"outer_seed": seed, "comparator": comparator, "metric": "AULC_429_525", "CW_minus_comparator": float(m.loc[(seed, METHOD), "AULC_429_525"] - m.loc[(seed, comparator), "AULC_429_525"])},
                {"outer_seed": seed, "comparator": comparator, "metric": "AULC_333_525", "CW_minus_comparator": float(f.loc[(seed, METHOD), "AULC_333_525"] - f.loc[(seed, comparator), "AULC_333_525"])},
            ])
            for metric in ("V1_RMSE", "V2_RMSE", "V1_R2", "V2_R2"):
                paired_rows.append({"outer_seed": seed, "comparator": comparator, "metric": f"{metric}@525", "CW_minus_comparator": float(c.loc[(METHOD, 525), metric] - c.loc[(comparator, 525), metric])})
    paired = pd.DataFrame(paired_rows)
    marginal = _marginal(curves)
    overlap = _selection_overlap(study)
    decision = _decision(curves, mid)
    cost_rows = []
    for seed in SEEDS:
        training_seconds = 0.0
        gradient_seconds = 0.0
        for round_index in (4, 5, 6):
            round_dir = study / "runtime" / f"seed_{seed}" / METHOD / f"round_{round_index:02d}"
            training_seconds += float(json.loads((round_dir / "model" / "fit_audit.json").read_text())["training_seconds"])
            # The hard-stop round has no acquisition and therefore no gradient contract.
            contract = round_dir / "acquisition_artifacts" / "contract.json"
            if contract.exists():
                gradient_seconds += float(json.loads(contract.read_text()).get("gradient_extraction_seconds", 0.0))
        cost_rows.append({"outer_seed": seed, "method": METHOD, "new_evaluation_fits": 3, "ensemble_fits": 0, "training_seconds": training_seconds, "gradient_seconds": gradient_seconds})
    cost = pd.DataFrame(cost_rows)
    curves.to_csv(results / "learning_curve_metrics.csv", index=False); paired.to_csv(results / "paired_comparison.csv", index=False); mid.to_csv(results / "partial_aulc_429_525.csv", index=False); full.to_csv(results / "full_aulc_333_525.csv", index=False); marginal.to_csv(results / "marginal_improvement.csv", index=False); overlap.to_csv(results / "selection_overlap.csv", index=False); cost.to_csv(results / "compute_cost.csv", index=False); access.to_csv(results / "test_label_access_audit.csv", index=False)
    atomic_json(study / "decision.json", decision)
    _figures(study, curves, marginal, mid)
    rows = []
    for seed in SEEDS:
        cw = mid.loc[(mid.outer_seed.eq(seed)) & mid.method.eq(METHOD), "AULC_429_525"].iloc[0]
        rows.append(f"| {seed} | {cw:.6f} |" + "".join(f" {mid.loc[(mid.outer_seed.eq(seed)) & mid.method.eq(m), 'AULC_429_525'].iloc[0]:.6f} |" for m in COMPARATORS))
    (study / "FINAL_REPORT.md").write_text(f"""# CW-LCMD 429 to 525 developmental continuation

This extension continues only the frozen Center/Width-LCMD trajectory from 429 to 461, 493, and 525 for seeds 157 and 6101. No comparator was retrained. The LCMD-to-IVR study remains sealed with engineering checks passed and formal execution not started.

## Primary AULC 429-525

| Seed | Center/Width-LCMD | Gradient-MaxDet | Hybrid | Gradient-LCMD | Kernel-IVR |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Decision: **{decision['decision']}** against strongest baseline **{decision['strongest_baseline']}**. Paired AULC deltas (CW minus strongest baseline) are {decision['paired_AULC_429_525_deltas']}; @525 deltas are {decision['paired_NRMSE_525_deltas']}. No continuation to 653 and no MaxDet->CW switch was implemented.

The marginal-improvement table separates 397->429 from 429->461, 461->493, and 493->525. A fast final drop at 397->429 is not treated as evidence by itself; the primary decision uses the complete 429->525 AULC and both seed directions.

All new batches were selected from current U_t using the current CW checkpoint and all current L_t as LCMD centers. Selected IDs were frozen before label reveal. Six new checkpoint/prediction artifacts were frozen before the single post-freeze test evaluation.
""")
    files = [p for p in study.rglob("*") if p.is_file() and "runtime" not in p.parts and p.name != "artifact_manifest.json"]
    atomic_json(study / "artifact_manifest.json", {"status": "COMPLETE_DEVELOPMENTAL_CONTINUATION", "hard_stop_active_labels": 525, "files": {str(p.relative_to(study)): sha256_file(p) for p in sorted(files)}})
    return {"status": "COMPLETE_DEVELOPMENTAL_CONTINUATION", "decision": decision}
