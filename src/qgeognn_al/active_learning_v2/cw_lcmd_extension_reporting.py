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
ALL_METHODS = (METHOD, *COMPARATORS)
ORIGINAL_REPORTING_CODE_SHA256 = "d607b1c5fdfa7c04edc1c36a0686743a8dff6fa322f94c22200d720f25462463"
ENDPOINT_DETERIORATION_TOLERANCE = 0.03


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
            & frame.active_label_count.isin(ACTIVE_LABEL_BUDGETS),
            ["outer_seed", "method", "round", "active_label_count", *METRICS],
        ]
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    _validate_budget_grid(result, ACTIVE_LABEL_BUDGETS, SEEDS, COMPARATORS)
    return result


def _validate_budget_grid(
    frame: pd.DataFrame,
    budgets: tuple[int, ...],
    seeds: tuple[int, ...],
    methods: tuple[str, ...],
) -> None:
    """Require one and only one metric row for every seed/method/budget."""
    expected_budgets = list(budgets)
    failures = []
    for seed in seeds:
        for method in methods:
            group = frame.loc[
                frame.outer_seed.eq(seed) & frame.method.eq(method),
                "active_label_count",
            ]
            observed = sorted(group.astype(int).tolist())
            if observed != expected_budgets:
                failures.append(
                    f"seed={seed} method={method} expected={expected_budgets} observed={observed}"
                )
    if failures:
        raise RuntimeError("incomplete AULC budget grid: " + "; ".join(failures))


def _aulc(
    frame: pd.DataFrame,
    start: int,
    stop: int,
    name: str,
    expected_budgets: tuple[int, ...],
) -> pd.DataFrame:
    if not expected_budgets or expected_budgets[0] != start or expected_budgets[-1] != stop:
        raise ValueError("AULC endpoints must match the declared budget grid")
    rows = []
    for (seed, method), group in frame.groupby(["outer_seed", "method"]):
        group = group.sort_values("active_label_count")
        x = group.active_label_count.to_numpy(float)
        y = group.combined_normalized_RMSE.to_numpy(float)
        observed = group.active_label_count.astype(int).tolist()
        if observed != list(expected_budgets):
            raise RuntimeError(
                f"incomplete AULC endpoints: seed={seed} method={method} "
                f"expected={list(expected_budgets)} observed={observed}"
            )
        rows.append({"outer_seed": int(seed), "method": method, name: float(np.trapezoid(y, x) / (stop - start))})
    return pd.DataFrame(rows)


def _per_seed_figure_name(seed: int) -> str:
    return f"per_seed_cw_continuation_{int(seed)}.png"


def _frozen_artifact_hashes(study: Path) -> dict[str, str]:
    """Hash immutable experiment artifacts touched by neither reporting nor plotting."""
    runtime = Path(study) / "runtime"
    names = {"best.pt", "predictions.csv.gz", "selected_next_batch.csv"}
    return {
        str(path.relative_to(study)): sha256_file(path)
        for path in sorted(runtime.rglob("*"))
        if path.is_file() and path.name in names
    }


def _assert_frozen_artifacts_unchanged(study: Path, before: dict[str, str]) -> None:
    after = _frozen_artifact_hashes(study)
    if after != before:
        changed = sorted(set(before) | set(after))
        changed = [path for path in changed if before.get(path) != after.get(path)]
        raise RuntimeError(f"reporting changed frozen experiment artifacts: {changed}")


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
    both_aulc_better = bool((a < 0).all())
    both_endpoints_better = bool((e <= 0).all())
    mean_endpoint_delta = float(e.mean())
    endpoint_deteriorated = mean_endpoint_delta > ENDPOINT_DETERIORATION_TOLERANCE
    if both_aulc_better and both_endpoints_better:
        category = "STRONG_MIDSTAGE_SIGNAL"
    elif (
        mean_mid[METHOD] < mean_mid[strongest]
        and a.max() <= 0.05
        and not endpoint_deteriorated
    ):
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
        "decision_criteria": {
            "both_seed_AULC_better": both_aulc_better,
            "both_seed_endpoints_better": both_endpoints_better,
            "mean_endpoint_delta": mean_endpoint_delta,
            "endpoint_deterioration_tolerance": ENDPOINT_DETERIORATION_TOLERANCE,
            "endpoint_deteriorated": endpoint_deteriorated,
        },
        "automatic_continuation_to_653": False,
        "maxdet_to_cw_switch_implemented": False,
    }


def _figures(
    study: Path,
    curves: pd.DataFrame,
    marginal: pd.DataFrame,
    mid: pd.DataFrame,
    full: pd.DataFrame,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = study / "figures"; figdir.mkdir(exist_ok=True)
    methods = (METHOD, "gradient_maxdet", "hybrid", "lcmd", "kernel_ivr")
    colors = {METHOD: "#2ca02c", "gradient_maxdet": "#d62728", "hybrid": "#ff7f0e", "lcmd": "#1f77b4", "kernel_ivr": "#9467bd"}
    for seed in (None, *SEEDS):
        subset = (
            curves.groupby(["method", "active_label_count"], as_index=False)[list(METRICS)].mean()
            if seed is None else curves.loc[curves.outer_seed.eq(seed)]
        )
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for method in methods:
            g = subset.loc[subset.method.eq(method)].sort_values("active_label_count")
            ax.plot(g.active_label_count, g.combined_normalized_RMSE, marker="o", label=DISPLAY_NAMES[method], color=colors[method])
        ax.set(xlabel="Active labels", ylabel="Combined normalized RMSE", xticks=list(ACTIVE_LABEL_BUDGETS)); ax.grid(alpha=.2); ax.legend(fontsize=8)
        name = "cw_full_trajectory_333_525.png" if seed is None else _per_seed_figure_name(seed)
        fig.tight_layout(); fig.savefig(figdir / name, dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method in methods:
        g = marginal.loc[marginal.method.eq(method)].groupby("to_labels").delta_NRMSE.mean().sort_index()
        ax.plot(g.index, g.values, marker="o", label=DISPLAY_NAMES[method], color=colors[method])
    ax.axhline(0, color="black", linewidth=.8); ax.set(xlabel="New budget endpoint", ylabel="NRMSE(t+1) - NRMSE(t)"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "marginal_nrmse_improvement_per_batch.png", dpi=180); plt.close(fig)
    areas = mid.groupby("method").AULC_429_525.mean().loc[list(methods)].sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.bar([DISPLAY_NAMES[m] for m in areas.index], areas.values, color=[colors[m] for m in areas.index]); ax.tick_params(axis="x", rotation=25); ax.set_ylabel("AULC 429-525 (lower is better)"); fig.tight_layout(); fig.savefig(figdir / "aulc_429_525_comparison.png", dpi=180); plt.close(fig)
    areas = full.groupby("method").AULC_333_525.mean().loc[list(methods)].sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.bar([DISPLAY_NAMES[m] for m in areas.index], areas.values, color=[colors[m] for m in areas.index]); ax.tick_params(axis="x", rotation=25); ax.set_ylabel("AULC 333-525 (lower is better)"); fig.tight_layout(); fig.savefig(figdir / "aulc_333_525_comparison.png", dpi=180); plt.close(fig)


def write_report(study: Path, extension_curves: pd.DataFrame, access: pd.DataFrame) -> dict:
    study = Path(study); results = study / "results"; results.mkdir(exist_ok=True)
    frozen_before = _frozen_artifact_hashes(study)
    historical = _historical()
    source = pd.read_csv(SOURCE_STUDY / "results/learning_curve_metrics.csv")
    cw_prefix_budgets = ACTIVE_LABEL_BUDGETS[:4]
    cw_prefix = source.loc[source.outer_seed.isin(SEEDS) & source.method.eq(METHOD) & source.active_label_count.isin(cw_prefix_budgets), ["outer_seed", "method", "round", "active_label_count", *METRICS]]
    _validate_budget_grid(cw_prefix, cw_prefix_budgets, SEEDS, (METHOD,))
    _validate_budget_grid(extension_curves, ACTIVE_LABEL_BUDGETS[4:], SEEDS, (METHOD,))
    curves = pd.concat([cw_prefix, extension_curves, historical], ignore_index=True).drop_duplicates(["outer_seed", "method", "active_label_count"])
    _validate_budget_grid(curves, ACTIVE_LABEL_BUDGETS, SEEDS, ALL_METHODS)
    mid = _aulc(
        curves.loc[curves.active_label_count.isin(CONTINUATION_BUDGETS)],
        429, 525, "AULC_429_525", CONTINUATION_BUDGETS,
    )
    full = _aulc(
        curves.loc[curves.active_label_count.isin(ACTIVE_LABEL_BUDGETS)],
        333, 525, "AULC_333_525", ACTIVE_LABEL_BUDGETS,
    )
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
    pd.DataFrame(
        [{"path": path, "sha256": digest} for path, digest in frozen_before.items()]
    ).to_csv(results / "frozen_artifact_hashes.csv", index=False)
    atomic_json(study / "decision.json", decision)
    _figures(study, curves, marginal, mid, full)

    def table_rows(table: pd.DataFrame, column: str) -> str:
        rows = []
        for label, subset in [*( (str(seed), table.loc[table.outer_seed.eq(seed)]) for seed in SEEDS), ("Mean", table)]:
            values = (
                subset.set_index("method")[column]
                if label != "Mean" else subset.groupby("method")[column].mean()
            )
            rows.append("| " + label + " | " + " | ".join(f"{values[m]:.6f}" for m in ALL_METHODS) + " |")
        return "\n".join(rows)

    full_rows = table_rows(full, "AULC_333_525")
    mid_rows = table_rows(mid, "AULC_429_525")
    criteria = decision["decision_criteria"]
    (study / "FINAL_REPORT.md").write_text(f"""# CW-LCMD 429 to 525 developmental continuation

This extension continued only the frozen Center/Width-LCMD trajectory from 429 to 461, 493, and 525 for seeds 157 and 6101. This reporting correction reused only frozen metrics and artifacts; it performed no training, acquisition, label reveal, or historical artifact rerun. No comparator was retrained.

## Full AULC 333-525

AULC 333-525 measures overall label efficiency from the initial active-learning budget. Every seed/method integral uses exactly 333, 365, 397, 429, 461, 493, and 525 active labels.

| Seed | Center/Width-LCMD | Gradient-MaxDet | Hybrid | Gradient-LCMD | Kernel-IVR |
|---:|---:|---:|---:|---:|---:|
{full_rows}

## Mid-stage AULC 429-525

AULC 429-525 measures local efficiency during the CW continuation window. It is distinct from the full AULC above and uses exactly 429, 461, 493, and 525 active labels.

| Seed | Center/Width-LCMD | Gradient-MaxDet | Hybrid | Gradient-LCMD | Kernel-IVR |
|---:|---:|---:|---:|---:|---:|
{mid_rows}

Decision: **{decision['decision']}** against strongest baseline **{decision['strongest_baseline']}**. Paired AULC deltas (CW minus strongest baseline) are {decision['paired_AULC_429_525_deltas']}; @525 deltas are {decision['paired_NRMSE_525_deltas']}. Both seeds improve mid-stage AULC, but both endpoints do not improve: seed 6101 is worse by {decision['paired_NRMSE_525_deltas']['6101']:.6f}. The mean endpoint delta is {criteria['mean_endpoint_delta']:.6f}, which is below the pre-existing +{criteria['endpoint_deterioration_tolerance']:.2f} deterioration tolerance. This supports PROMISING, not STRONG_MIDSTAGE_SIGNAL. No continuation to 653 is recommended under this completed developmental protocol, and no MaxDet-to-CW switch was implemented.

The marginal-improvement table separates 397->429 from 429->461, 461->493, and 493->525. A fast final drop at 397->429 is not treated as evidence by itself; the primary decision uses the complete 429->525 AULC and both seed directions.

All new batches were selected from current U_t using the current CW checkpoint and all current L_t as LCMD centers. Selected IDs were frozen before label reveal. Six new checkpoint/prediction artifacts were frozen before the single post-freeze test evaluation. Their hashes were verified unchanged by this report-only regeneration.

The separate LCMD-to-IVR two-seed development run has completed. It showed mixed late-AULC behavior and was stopped before the remaining three seeds. That separate study does not alter the CW decision here.
""")
    corrected_code_sha = sha256_file(Path(__file__))
    (study / "REPORTING_CORRECTION.md").write_text(f"""# Reporting correction provenance

- Original reporting code SHA-256: `{ORIGINAL_REPORTING_CODE_SHA256}`.
- Corrected reporting code SHA-256: `{corrected_code_sha}`.
- Frozen experiment protocol: unchanged.
- Checkpoints, predictions, selected-batch files/IDs, and test split: unchanged; {len(frozen_before)} protected artifacts were verified byte-for-byte before and after regeneration.
- Test truth was not read again and was not used for method design. This pass read only previously frozen metric/reporting artifacts.
- Training, acquisition, label reveal, and historical artifact reruns: none.

The former `AULC_333_525` calculation assembled only the 429, 461, 493, and 525 points and divided that partial trapezoidal area by `525 - 333`. It omitted the 333, 365, and 397 points, so it was not an integral over the claimed interval. The corrected calculation requires exactly seven points—333, 365, 397, 429, 461, 493, and 525—for every seed and method, and fails loudly on any missing or duplicate point.

The former PROMISING rule used `abs(mean endpoint delta) <= 0.03`. Because endpoint delta is CW minus the strongest baseline, a negative value is an improvement; taking the absolute value incorrectly rejected sufficiently large improvements as though they were deterioration. The corrected no-deterioration rule is `mean endpoint delta <= +0.03`.

The canonical decision changed from **MIXED** to **PROMISING**. It did not change to STRONG_MIDSTAGE_SIGNAL: both seeds have better AULC 429-525 than the same strongest baseline, but seed 6101 does not have a better endpoint. This correction does not authorize continuation to 653.

The formerly misnamed `per_seed_cw_continuation_101.png` is retained only under `figures/legacy/reporting_error/` and is deprecated. The canonical seed figures are `per_seed_cw_continuation_157.png` and `per_seed_cw_continuation_6101.png`.
""")
    _assert_frozen_artifacts_unchanged(study, frozen_before)
    files = [p for p in study.rglob("*") if p.is_file() and "runtime" not in p.parts and p.name != "artifact_manifest.json"]
    atomic_json(study / "artifact_manifest.json", {
        "status": "COMPLETE_DEVELOPMENTAL_CONTINUATION",
        "hard_stop_active_labels": 525,
        "reporting_correction": True,
        "deprecated_files": ["figures/legacy/reporting_error/per_seed_cw_continuation_101.png"],
        "files": {str(p.relative_to(study)): sha256_file(p) for p in sorted(files)},
    })
    return {"status": "COMPLETE_DEVELOPMENTAL_CONTINUATION", "decision": decision}


def regenerate_from_frozen_results(study: Path = STUDY) -> dict:
    """Regenerate derived reporting without reading labels or running a model."""
    study = Path(study)
    freeze = json.loads((study / "global_pre_test_freeze.json").read_text())
    if freeze.get("status") != "COMPLETE_DEVELOPMENTAL_CONTINUATION":
        raise RuntimeError("report-only regeneration requires a completed frozen continuation")
    curves = pd.read_csv(study / "results" / "learning_curve_metrics.csv")
    extension = curves.loc[
        curves.outer_seed.isin(SEEDS)
        & curves.method.eq(METHOD)
        & curves.active_label_count.isin(ACTIVE_LABEL_BUDGETS[4:]),
        ["outer_seed", "method", "round", "active_label_count", *METRICS],
    ].copy()
    access = pd.read_csv(study / "results" / "test_label_access_audit.csv")
    return write_report(study, extension, access)
