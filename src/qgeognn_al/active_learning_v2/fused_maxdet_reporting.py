"""Post-barrier comparison/reporting for parameterized Fusion-MaxDet studies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import sha256_file
from ..resources import SOURCE_DATA
from ..training.predictor import atomic_json
from .benchmark_reporting import metric_row
from .fused_maxdet_runner import _json, _phase_budget
from .fused_maxdet_study import DEFAULT_SPEC, FusedMaxDetStudySpec
from .protocol import RestrictedLabelStore


ROOT = DEFAULT_SPEC.study.parents[2]
GRADIENT_STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_maxdet_b32"
A050_STUDY = DEFAULT_SPEC.study
GRADIENT_METHOD = "gradient_maxdet"
A050_METHOD = DEFAULT_SPEC.method
METRICS = ("combined_normalized_RMSE", "V1_RMSE", "V1_R2", "V2_RMSE", "V2_R2")


def _verify_global_freeze(study: Path, spec: FusedMaxDetStudySpec) -> dict:
    freeze = _json(Path(study) / "global_pre_test_freeze.json")
    expected = len(spec.seeds) * len(spec.active_label_budgets)
    if (
        freeze.get("status") != "FROZEN_BEFORE_TEST_TRUTH"
        or freeze.get("alpha") != float(spec.alpha)
        or freeze.get("test_truth_access_count") != 0
        or len(freeze.get("entries", {})) != expected
    ):
        raise RuntimeError("fusion global pre-test barrier is incomplete")
    for key, entry in freeze["entries"].items():
        seed, round_name = key.split("/")
        path = Path(study) / "runtime" / seed / spec.method / round_name / "round_freeze.json"
        if sha256_file(path) != entry["round_freeze_sha256"]:
            raise RuntimeError(f"fusion round freeze changed: {key}")
        record = _json(path)
        if (
            sha256_file(Path(record["checkpoint_path"])) != record["checkpoint_sha256"]
            or sha256_file(Path(record["prediction_path"])) != record["prediction_sha256"]
        ):
            raise RuntimeError(f"fusion frozen prediction changed: {key}")
    return freeze


def _prediction_path(study: Path, spec: FusedMaxDetStudySpec, seed: int, method: str, round_index: int) -> Path:
    if method == spec.method:
        record_path = study / "runtime" / f"seed_{seed}" / spec.method / f"round_{round_index:02d}/round_freeze.json"
    elif method == GRADIENT_METHOD:
        record_path = GRADIENT_STUDY / "runtime" / f"seed_{seed}" / GRADIENT_METHOD / f"round_{round_index:02d}/round_freeze.json"
    elif method == A050_METHOD:
        record_path = A050_STUDY / "runtime" / f"seed_{seed}" / A050_METHOD / f"round_{round_index:02d}/round_freeze.json"
    else:
        raise ValueError(f"unknown comparison method: {method}")
    return Path(_json(record_path)["prediction_path"])


def _curves(study: Path, spec: FusedMaxDetStudySpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []
    overlap: list[dict[str, object]] = []
    methods = (GRADIENT_METHOD, A050_METHOD, spec.method)
    for seed in spec.seeds:
        partition = pd.read_csv(study / "splits" / f"row_seed_{seed}.csv")
        test_ids = partition.loc[partition.role.eq("test"), "sample_id"].astype(str).tolist()
        context = _json(study / "runtime" / f"seed_{seed}/context.json")
        scales = context["preprocessing"]["target_scales"]
        store = RestrictedLabelStore(SOURCE_DATA, partition)
        selected: list[str] = []
        for round_index in range(spec.acquisition_rounds):
            artifacts = study / "runtime" / f"seed_{seed}" / spec.method / f"round_{round_index:02d}/acquisition_artifacts"
            selected_frame = pd.read_csv(artifacts / "selected_batch.csv")
            selected_ids = selected_frame.sample_id.astype(str).tolist()
            if len(selected_ids) != 32 or len(set(selected_ids)) != 32:
                raise RuntimeError("fusion selection is not a unique batch of 32")
            selected.extend(selected_ids)
            profile = _json(artifacts / "fusion_profile.json")
            profiles.append(profile)
            overlap.append({
                "outer_seed": seed,
                "source_round": round_index,
                "active_label_count": spec.active_label_budgets[round_index],
                "a080_vs_gradient_overlap_count": profile["gradient_maxdet_overlap_count"],
                "a080_vs_gradient_overlap_fraction": profile["gradient_maxdet_overlap_fraction"],
                "a080_vs_a050_overlap_count": profile["a050_fusion_overlap_count"],
                "a080_vs_a050_overlap_fraction": profile["a050_fusion_overlap_fraction"],
            })
        store.freeze_acquisitions(selected)
        store.freeze_predictions()
        truth = store.reveal(test_ids, "final_test_evaluation")
        for method in methods:
            for round_index, active in enumerate(spec.active_label_budgets):
                table = pd.read_csv(_prediction_path(study, spec, seed, method, round_index))
                if table.sample_id.astype(str).tolist() != test_ids:
                    raise RuntimeError("frozen test prediction order drift")
                rows.append({
                    "outer_seed": seed,
                    "method": method,
                    "round": round_index,
                    **_phase_budget(active),
                    **metric_row(truth, table.drop(columns="sample_id").to_numpy(float), scales),
                })
    return pd.DataFrame(rows), pd.DataFrame(profiles), pd.DataFrame(overlap)


def _interval_aulc(group: pd.DataFrame, metric: str, start: int, stop: int) -> float:
    interval = group.loc[group.active_label_count.between(start, stop)].sort_values("active_label_count")
    if interval.active_label_count.tolist()[0] != start or interval.active_label_count.tolist()[-1] != stop:
        raise RuntimeError(f"AULC interval endpoints are missing: {start}-{stop}")
    x = interval.active_label_count.to_numpy(float)
    y = interval[metric].to_numpy(float)
    return float(np.trapezoid(y, x) / (stop - start))


def _summary(curves: pd.DataFrame, spec: FusedMaxDetStudySpec) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    intervals = (("full", 333, 653), ("early", 333, 525), ("middle", 525, 653))
    for (seed, method), group in curves.groupby(["outer_seed", "method"]):
        group = group.sort_values("active_label_count")
        for metric in METRICS:
            record: dict[str, object] = {
                "outer_seed": seed,
                "method": method,
                "metric": metric,
                "endpoint_budget": spec.final_active_labels,
                "endpoint": float(group.loc[group.active_label_count.eq(spec.final_active_labels), metric].iloc[0]),
            }
            for name, start, stop in intervals:
                record[f"{name}_aulc_{start}_{stop}"] = _interval_aulc(group, metric, start, stop)
            records.append(record)
    return pd.DataFrame(records)


def _paired(summary: pd.DataFrame, spec: FusedMaxDetStudySpec) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in spec.seeds:
        by_method = summary.loc[
            summary.outer_seed.eq(seed) & summary.metric.eq("combined_normalized_RMSE")
        ].set_index("method")
        gradient = by_method.loc[GRADIENT_METHOD]
        a050 = by_method.loc[A050_METHOD]
        a080 = by_method.loc[spec.method]
        record: dict[str, object] = {"outer_seed": seed}
        for column in ("full_aulc_333_653", "early_aulc_333_525", "middle_aulc_525_653", "endpoint"):
            record[f"gradient_{column}"] = gradient[column]
            record[f"a050_{column}"] = a050[column]
            record[f"a080_{column}"] = a080[column]
            record[f"a080_minus_gradient_{column}"] = a080[column] - gradient[column]
            record[f"a080_minus_a050_{column}"] = a080[column] - a050[column]
        for metric in ("V1_RMSE", "V1_R2", "V2_RMSE", "V2_R2"):
            subset = summary.loc[summary.outer_seed.eq(seed) & summary.metric.eq(metric)].set_index("method")
            for method, label in ((GRADIENT_METHOD, "gradient"), (A050_METHOD, "a050"), (spec.method, "a080")):
                record[f"{label}_{metric}_at_653"] = subset.loc[method, "endpoint"]
        rows.append(record)
    return pd.DataFrame(rows)


def _representation_tables(profiles: pd.DataFrame, spec: FusedMaxDetStudySpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    for row in profiles.to_dict("records"):
        common = {"outer_seed": row["outer_seed"], "source_round": row["source_round"], "active_label_count": row["active_label_count"]}
        variants = (
            ("gradient_a080_trajectory", "gradient_effective_rank", "gradient_pairwise_kernel_abs_corr"),
            ("latent_a080_trajectory", "latent_effective_rank", "latent_pairwise_kernel_abs_corr"),
            ("centered_latent_diagnostic", "centered_latent_effective_rank", "centered_latent_pairwise_kernel_abs_corr"),
            ("fused_a050_counterfactual_on_a080_trajectory", "fused_a050_counterfactual_effective_rank", "fused_a050_counterfactual_pairwise_kernel_abs_corr"),
            ("fused_a080", "fused_effective_rank", "fused_pairwise_kernel_abs_corr"),
        )
        for representation, rank_key, corr_key in variants:
            records.append({**common, "representation": representation, "effective_rank": row[rank_key], "pairwise_kernel_abs_corr": row[corr_key], "latent_mean_direction_ratio": row.get("latent_mean_direction_ratio")})
    for seed in spec.seeds:
        for round_index, active in enumerate(spec.active_label_budgets[:-1]):
            path = A050_STUDY / "runtime" / f"seed_{seed}" / A050_METHOD / f"round_{round_index:02d}/acquisition_artifacts/fusion_profile.json"
            row = _json(path)
            records.append({"outer_seed": seed, "source_round": round_index, "active_label_count": active,
                            "representation": "fused_a050_historical_trajectory", "effective_rank": row["fused_effective_rank"],
                            "pairwise_kernel_abs_corr": row["fused_pairwise_kernel_abs_corr"], "latent_mean_direction_ratio": np.nan})
    detail = pd.DataFrame(records)
    summary = detail.groupby(["outer_seed", "representation"], as_index=False).agg(
        rounds=("source_round", "count"),
        mean_effective_rank=("effective_rank", "mean"),
        mean_pairwise_kernel_abs_corr=("pairwise_kernel_abs_corr", "mean"),
        mean_latent_mean_direction_ratio=("latent_mean_direction_ratio", "mean"),
    )
    return detail, summary


def _decision(paired: pd.DataFrame, representation_summary: pd.DataFrame, spec: FusedMaxDetStudySpec) -> dict[str, object]:
    mean = paired.mean(numeric_only=True)
    full_diff = float(mean["a080_minus_gradient_full_aulc_333_653"])
    endpoint_diff = float(mean["a080_minus_gradient_endpoint"])
    a050_full_diff = float((paired.a050_full_aulc_333_653 - paired.gradient_full_aulc_333_653).mean())
    seed_full_diffs = paired.a080_minus_gradient_full_aulc_333_653.to_numpy(float)
    seed_endpoint_diffs = paired.a080_minus_gradient_endpoint.to_numpy(float)
    a080_rank = representation_summary.loc[representation_summary.representation.eq("fused_a080"), "mean_effective_rank"].mean()
    gradient_rank = representation_summary.loc[representation_summary.representation.eq("gradient_a080_trajectory"), "mean_effective_rank"].mean()
    if full_diff <= 0 and not np.all(seed_full_diffs > 0):
        category = "CONTINUE_TO_1005"
        conclusion = "Alpha=.8 matches or improves mean truncated AULC without directionally consistent paired degradation."
    elif full_diff < a050_full_diff and (
        np.any(seed_full_diffs < 0)
        or float(mean["a080_minus_gradient_early_aulc_333_525"]) < 0
        or float(mean["a080_minus_gradient_middle_aulc_525_653"]) < 0
    ):
        category = "WORTH_FULL_TRAJECTORY"
        conclusion = "Alpha=.8 remains slightly behind Gradient-MaxDet but recovers loss versus alpha=.5 and has a stable stage/seed gain."
    elif np.all(seed_full_diffs > 0) and np.all(seed_endpoint_diffs > 0) and a080_rank < gradient_rank:
        category = "STOP_SIMPLE_WEIGHTED_FUSION"
        conclusion = "Both seeds degrade in AULC and endpoint while fusion rank remains below Gradient; move to centered/residual latent fusion."
    else:
        category = "INCONCLUSIVE_DEVELOPMENTAL_SCREEN"
        conclusion = "The two-seed truncated evidence is mixed and does not justify a strong confirmation claim."
    return {
        "status": "COMPLETE_TRUNCATED_DEVELOPMENTAL_SCREEN",
        "evidence_class": spec.evidence_class,
        "alpha": float(spec.alpha),
        "decision": category,
        "conclusion": conclusion,
        "mean_a080_minus_gradient_nrmse_aulc_333_653": full_diff,
        "mean_a050_minus_gradient_nrmse_aulc_333_653": a050_full_diff,
        "mean_a080_minus_gradient_nrmse_at_653": endpoint_diff,
        "two_seed_inference_only": True,
        "statistical_significance_claimed": False,
    }


def _write_figures(study: Path, curves: pd.DataFrame, overlap: pd.DataFrame, representation: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = study / "figures"
    figures.mkdir(exist_ok=True)
    labels = {GRADIENT_METHOD: "Gradient-MaxDet", A050_METHOD: "Fusion alpha=.5"}
    for method in curves.method.unique():
        labels.setdefault(method, "Fusion alpha=.8")
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for method, group in curves.groupby("method"):
        mean = group.groupby("active_label_count").combined_normalized_RMSE.mean()
        ax.plot(mean.index, mean.values, marker="o", label=labels[method])
    ax.set(xlabel="Active labels", ylabel="Combined normalized RMSE")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(figures / "nrmse_learning_curve_333_653.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for seed, group in overlap.groupby("outer_seed"):
        axes[0].plot(group.source_round, group.a080_vs_gradient_overlap_fraction, marker="o", label=f"seed {seed}")
        axes[1].plot(group.source_round, group.a080_vs_a050_overlap_fraction, marker="o", label=f"seed {seed}")
    for axis, title in zip(axes, ("alpha=.8 vs Gradient", "alpha=.8 vs alpha=.5")):
        axis.set(xlabel="Acquisition round", title=title); axis.grid(alpha=.2); axis.legend(fontsize=8)
    axes[0].set_ylabel("Batch overlap fraction")
    fig.tight_layout(); fig.savefig(figures / "selection_overlap.png", dpi=180); plt.close(fig)

    selected = representation.loc[representation.representation.isin([
        "gradient_a080_trajectory", "latent_a080_trajectory", "centered_latent_diagnostic",
        "fused_a050_historical_trajectory", "fused_a080",
    ])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, group in selected.groupby("representation"):
        mean = group.groupby("active_label_count")[["effective_rank", "pairwise_kernel_abs_corr"]].mean()
        axes[0].plot(mean.index, mean.effective_rank, marker=".", label=name)
        axes[1].plot(mean.index, mean.pairwise_kernel_abs_corr, marker=".", label=name)
    axes[0].set(xlabel="Active labels", ylabel="Effective rank")
    axes[1].set(xlabel="Active labels", ylabel="Pairwise normalized-kernel |corr|")
    for axis in axes: axis.grid(alpha=.2)
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figures / "representation_diagnostics.png", dpi=180); plt.close(fig)


def _write_report(study: Path, paired: pd.DataFrame, decision: dict[str, object], representation_summary: pd.DataFrame, overlap: pd.DataFrame) -> None:
    mean = paired.mean(numeric_only=True)
    seed_lines = []
    for row in paired.to_dict("records"):
        seed_lines.append(
            f"| {int(row['outer_seed'])} | {row['gradient_full_aulc_333_653']:.6f} | {row['a050_full_aulc_333_653']:.6f} | "
            f"{row['a080_full_aulc_333_653']:.6f} | {row['a080_minus_gradient_endpoint']:+.6f} |"
        )
    common = representation_summary.loc[representation_summary.representation.eq("centered_latent_diagnostic")]
    common_text = ", ".join(
        f"seed {int(row.outer_seed)} centered rank={row.mean_effective_rank:.3f}, centered |corr|={row.mean_pairwise_kernel_abs_corr:.3f}, r_mean={row.mean_latent_mean_direction_ratio:.3f}"
        for row in common.itertuples()
    )
    report = f"""# Alpha=.8 Fusion-MaxDet truncated developmental screen

This is a two-seed developmental screen through 653 active labels, not an independent confirmation.

## Primary result

| Seed | Gradient AULC | alpha=.5 AULC | alpha=.8 AULC | alpha=.8 - Gradient NRMSE@653 |
|---:|---:|---:|---:|---:|
{chr(10).join(seed_lines)}

Mean alpha=.8 minus Gradient AULC (333-653): **{mean['a080_minus_gradient_full_aulc_333_653']:+.6f}**.  
Mean alpha=.8 minus alpha=.5 AULC (333-653): **{mean['a080_minus_a050_full_aulc_333_653']:+.6f}**.  
Early difference vs Gradient (333-525): **{mean['a080_minus_gradient_early_aulc_333_525']:+.6f}**.  
Middle difference vs Gradient (525-653): **{mean['a080_minus_gradient_middle_aulc_525_653']:+.6f}**.  
Mean endpoint difference vs Gradient: **{mean['a080_minus_gradient_endpoint']:+.6f}**.

## Mechanism diagnostics

{common_text}. Acquisition always used the original uncentered latent representation. Mean per-round batch overlap was {overlap.a080_vs_gradient_overlap_fraction.mean():.3f} with Gradient-MaxDet and {overlap.a080_vs_a050_overlap_fraction.mean():.3f} with alpha=.5 fusion.

## Decision

**{decision['decision']}** — {decision['conclusion']}

1. Alpha=.8 versus alpha=.5: determined from the paired truncated AULC and endpoint columns above.
2. Alpha=.8 versus Gradient-MaxDet: determined from the full curve, phase AULCs, and endpoint—not the endpoint alone.
3. Stage of improvement: compare the signed early and middle AULC differences above.

No statistical-significance claim is made from two seeds.
"""
    (study / "FINAL_REPORT.md").write_text(report)


def reveal_test_and_report(
    study: Path | None = None,
    *,
    spec: FusedMaxDetStudySpec = DEFAULT_SPEC,
) -> dict[str, object]:
    study = spec.study if study is None else Path(study)
    freeze = _verify_global_freeze(study, spec)
    curves, profiles, overlap = _curves(study, spec)
    summary = _summary(curves, spec)
    paired = _paired(summary, spec)
    representation, representation_summary = _representation_tables(profiles, spec)
    results = study / "results"
    results.mkdir(parents=True, exist_ok=True)
    curves.to_csv(results / "learning_curve_metrics.csv", index=False)
    summary.to_csv(results / "aulc_phase_and_endpoint_metrics.csv", index=False)
    paired.to_csv(results / "paired_comparison.csv", index=False)
    profiles.to_csv(results / "fusion_selection_profiles.csv", index=False)
    overlap.to_csv(results / "selection_overlap.csv", index=False)
    representation.to_csv(results / "representation_diagnostics.csv", index=False)
    representation_summary.to_csv(results / "representation_diagnostics_summary.csv", index=False)
    cost = pd.DataFrame([{
        "outer_seed": seed,
        "method": spec.method,
        "training_seconds": float(pd.read_csv(study / "runtime" / f"seed_{seed}/{spec.method}/fit_audit.csv").training_seconds.sum()),
        "gradient_seconds": float(profiles.loc[profiles.outer_seed.eq(seed), "gradient_seconds"].sum()),
        "selector_seconds": float(profiles.loc[profiles.outer_seed.eq(seed), "selector_seconds"].sum()),
    } for seed in spec.seeds])
    cost.to_csv(results / "compute_cost.csv", index=False)
    decision = _decision(paired, representation_summary, spec)
    decision.update({"methods": [GRADIENT_METHOD, A050_METHOD, spec.method], "seeds": list(spec.seeds),
                     "frozen_prediction_points": freeze["frozen_prediction_points"], "test_truth_access_after_barrier": True})
    atomic_json(study / "decision.json", decision)
    _write_figures(study, curves, overlap, representation)
    _write_report(study, paired, decision, representation_summary, overlap)
    return decision


__all__ = ["reveal_test_and_report"]
