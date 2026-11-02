"""Post-barrier evaluation and mechanism reporting for the MaxDet study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import sha256_file
from ..resources import SOURCE_DATA
from ..training.predictor import atomic_json
from .benchmark_protocol import load_features
from .benchmark_reporting import metric_row
from .efficiency_reporting import METRICS, evaluate, historical_costs, write_evaluation
from .maxdet_runner import _json, _partition
from .maxdet_study import (
    BASELINE,
    CONFIRMATION_SEEDS,
    IVR_BASELINE,
    METHODS,
    ROW_PHASES,
    STUDY,
    validate_pre_test_manifest,
)
from .protocol import RestrictedLabelStore
from .sequential_protocol import ACQUISITION_ROUNDS, ACTIVE_LABEL_BUDGETS, BATCH_SIZE


def _verify_barrier(study: Path) -> dict[str, object]:
    validate_pre_test_manifest(study)
    freeze = _json(study / "global_pre_test_freeze.json")
    expected = len(CONFIRMATION_SEEDS) * len(METHODS) * (ACQUISITION_ROUNDS + 1)
    if freeze.get("status") != "FROZEN_BEFORE_TEST_TRUTH" or freeze.get("test_truth_access_count") != 0:
        raise RuntimeError("MaxDet global pre-test barrier is not closed")
    if len(freeze.get("entries", {})) != expected:
        raise RuntimeError("MaxDet global pre-test matrix is incomplete")
    for key, entry in freeze["entries"].items():
        seed_text, method, round_text = key.split("/")
        path = study / "runtime" / seed_text / method / round_text / "round_freeze.json"
        if sha256_file(path) != entry["round_freeze_sha256"]:
            raise RuntimeError(f"MaxDet round freeze changed after global barrier: {key}")
        record = _json(path)
        if sha256_file(Path(record["checkpoint_path"])) != entry["checkpoint_sha256"]:
            raise RuntimeError(f"MaxDet checkpoint changed after global barrier: {key}")
        if sha256_file(Path(record["prediction_path"])) != entry["prediction_sha256"]:
            raise RuntimeError(f"MaxDet prediction changed after global barrier: {key}")
    return freeze


def _new_learning_curves(study: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_features()
    rows: list[dict[str, object]] = []
    access: list[dict[str, object]] = []
    for seed in CONFIRMATION_SEEDS:
        partition = _partition(seed, study)
        test_ids = partition.loc[partition.role.eq("test"), "sample_id"].astype(str).tolist()
        context = _json(study / "runtime" / f"seed_{seed}/context.json")
        scales = context["preprocessing"]["target_scales"]
        for method in METHODS:
            root = study / "runtime" / f"seed_{seed}" / method
            selected_ids = []
            for round_index in range(ACQUISITION_ROUNDS):
                selected_ids.extend(pd.read_csv(
                    root / f"round_{round_index:02d}/acquisition/selected_next_batch.csv"
                ).sample_id.astype(str))
            if len(selected_ids) != ACQUISITION_ROUNDS * BATCH_SIZE or len(set(selected_ids)) != len(selected_ids):
                raise RuntimeError("MaxDet selected-ID history is incomplete or duplicated")
            store = RestrictedLabelStore(SOURCE_DATA, partition)
            store.freeze_acquisitions(selected_ids)
            store.freeze_predictions()
            truth = store.reveal(test_ids, "final_test_evaluation")
            access.extend({"outer_seed": seed, "method": method, **item} for item in store.audit)
            for round_index, active in enumerate(ACTIVE_LABEL_BUDGETS):
                record = _json(root / f"round_{round_index:02d}/round_freeze.json")
                prediction = pd.read_csv(record["prediction_path"])
                if prediction.sample_id.astype(str).tolist() != test_ids:
                    raise RuntimeError("MaxDet frozen test-X prediction order drift")
                rows.append({
                    "outer_seed": seed,
                    "method": method,
                    "round": round_index,
                    "active_label_count": active,
                    "active_label_fraction_outer_train": active / 3330,
                    "shared_validation_label_count": 416,
                    "total_observed_non_test_labels": active + 416,
                    "total_observed_fraction_full_dataset": (active + 416) / len(data),
                    **metric_row(truth, prediction.drop(columns="sample_id").to_numpy(float), scales),
                })
    return pd.DataFrame(rows), pd.DataFrame(access)


def _historical_curves() -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_path = BASELINE / "results/learning_curve_metrics.csv"
    ivr_path = IVR_BASELINE / "results/learning_curve_metrics.csv"
    full_path = BASELINE / "results/full_data_reference.csv"
    baseline = pd.read_csv(baseline_path)
    ivr = pd.read_csv(ivr_path)
    keys = ["outer_seed", "method", "active_label_count"]
    for method in ("random", "lcmd"):
        left = baseline.loc[baseline.method.eq(method)].sort_values(keys)
        right = ivr.loc[ivr.method.eq(method)].sort_values(keys)
        if left[keys].values.tolist() != right[keys].values.tolist() or not np.allclose(
            left[list(METRICS)], right[list(METRICS)], rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(f"historical IVR reuse differs from sequential {method}")
    combined = pd.concat(
        [baseline, ivr.loc[ivr.method.eq("kernel_ivr")]], ignore_index=True
    )
    return combined, pd.read_csv(full_path)


def _partial_aulc(curves: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (seed, method), group in curves.groupby(["outer_seed", "method"]):
        ordered = group.sort_values("active_label_count")
        for phase, start, stop in ROW_PHASES:
            part = ordered.loc[ordered.active_label_count.between(start, stop)]
            if part.active_label_count.tolist() != [value for value in ACTIVE_LABEL_BUDGETS if start <= value <= stop]:
                raise RuntimeError(f"partial AULC boundary/grid mismatch: {phase}")
            for metric in METRICS:
                records.append({
                    "outer_seed": int(seed),
                    "method": method,
                    "phase": phase,
                    "start_labels": start,
                    "stop_labels": stop,
                    "metric": metric,
                    "normalized_partial_AULC": float(np.trapezoid(part[metric], part.active_label_count) / (stop - start)),
                })
    frame = pd.DataFrame(records)
    summary = frame.groupby(["method", "phase", "start_labels", "stop_labels", "metric"])[
        "normalized_partial_AULC"
    ].agg(["mean", "median", "std"]).reset_index()
    return frame, summary


def _paired_new_effects(partial: pd.DataFrame, evaluation: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    comparisons = (
        ("gradient_maxdet", "lcmd"),
        ("u50_gradient_maxdet", "gradient_maxdet"),
        ("u50_gradient_maxdet", "hybrid"),
        ("u50_gradient_maxdet", "lcmd"),
    )
    for phase in ("whole", "early", "middle", "late"):
        if phase == "whole":
            source = evaluation["per_seed_metrics"].loc[
                evaluation["per_seed_metrics"].metric.eq("combined_normalized_RMSE"),
                ["outer_seed", "method", "aulc"],
            ].rename(columns={"aulc": "value"})
        else:
            source = partial.loc[
                partial.phase.eq(phase) & partial.metric.eq("combined_normalized_RMSE"),
                ["outer_seed", "method", "normalized_partial_AULC"],
            ].rename(columns={"normalized_partial_AULC": "value"})
        pivot = source.pivot(index="outer_seed", columns="method", values="value")
        for method, comparator in comparisons:
            delta = (pivot[method] - pivot[comparator]).to_numpy(float)
            records.append({
                "phase": phase,
                "metric": "combined_normalized_RMSE",
                "method": method,
                "comparator": comparator,
                "mean_difference": float(delta.mean()),
                "median_difference": float(np.median(delta)),
                "directional_wins_lower_is_better": int((delta < 0).sum()),
                "per_seed_differences": json.dumps(dict(zip(pivot.index.astype(str), delta.tolist())), sort_keys=True),
            })
    return pd.DataFrame(records)


def _new_costs(study: Path) -> pd.DataFrame:
    rows = []
    for seed in CONFIRMATION_SEEDS:
        for method in METHODS:
            root = study / "runtime" / f"seed_{seed}" / method
            fits = pd.read_csv(root / "fit_audit.csv")
            profiles = pd.DataFrame([
                _json(root / f"round_{round_index:02d}/acquisition/selection_profile.json")
                for round_index in range(ACQUISITION_ROUNDS)
            ])
            member0 = fits.member.eq(0)
            extras = fits.member.gt(0)
            fresh = fits.reuse_status.eq("new_fit")
            rows.append({
                "outer_seed": seed,
                "method": method,
                "fit_count": int(len(fits)),
                "member0_evaluation_fits": int(member0.sum()),
                "member1_member2_extra_ensemble_fits": int(extras.sum()),
                "reused_fit_count": int((~fresh).sum()),
                "new_fit_count": int(fresh.sum()),
                "epochs": int(fits.loc[fresh, "epochs_run"].sum()),
                "training_seconds": float(fits.loc[fresh, "training_seconds"].sum()),
                "new_training_seconds": float(fits.loc[fresh, "training_seconds"].sum()),
                "gradient_extractions": int(len(profiles)),
                "reused_gradient_extractions": int(profiles.gradient_status.str.startswith("reused").sum()),
                "gradient_seconds": float(profiles.loc[~profiles.gradient_status.str.startswith("reused"), "gradient_seconds"].sum()),
                "selector_seconds": float(profiles.selector_seconds.sum()),
                "ensemble_inference_seconds": float(profiles.ensemble_inference_seconds_total.sum()),
                "historical_round_elapsed_seconds": np.nan,
                "cost_note": "new fit time uses the historical helper and includes validation plus its frozen prediction; separately clocked acquisition inference is reported without forming a weighted score",
            })
    return pd.DataFrame(rows)


def _mechanism_tables(study: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profiles = []
    overlaps = []
    selected = []
    for seed in CONFIRMATION_SEEDS:
        seed_root = study / "runtime" / f"seed_{seed}"
        overlaps.append(pd.read_csv(seed_root / "pure_u50_selection_overlap.csv"))
        for method in METHODS:
            root = seed_root / method
            for round_index in range(ACQUISITION_ROUNDS):
                profiles.append(_json(root / f"round_{round_index:02d}/acquisition/selection_profile.json"))
                selected.append(pd.read_csv(root / f"round_{round_index:02d}/acquisition/selected_next_batch.csv"))
    profile_frame = pd.DataFrame(profiles)
    overlap_frame = pd.concat(overlaps, ignore_index=True)
    selected_frame = pd.concat(selected, ignore_index=True)
    summary = profile_frame.groupby("method").agg(
        rounds=("source_round", "count"),
        mean_gradient_norm=("selected_gradient_norm_mean", "mean"),
        mean_effective_rank=("gram_effective_rank", "mean"),
        mean_abs_kernel_correlation=("within_batch_mean_absolute_normalized_kernel_correlation", "mean"),
        mean_overlap_lcmd=("overlap_lcmd_fraction", "mean"),
        mean_overlap_hybrid=("overlap_hybrid_fraction", "mean"),
        total_gradient_seconds=("gradient_seconds", "sum"),
        total_selector_seconds=("selector_seconds", "sum"),
    ).reset_index()
    pure_u50 = pd.DataFrame([{
        "comparison": "u50_gradient_maxdet_vs_gradient_maxdet",
        "rounds": len(overlap_frame),
        "mean_overlap_fraction": float(overlap_frame.overlap_fraction.mean()),
        "median_overlap_fraction": float(overlap_frame.overlap_fraction.median()),
        "min_overlap_fraction": float(overlap_frame.overlap_fraction.min()),
        "max_overlap_fraction": float(overlap_frame.overlap_fraction.max()),
        "identical_batches": int(overlap_frame.same_ordered_batch.sum()),
    }])
    summary = pd.concat([summary, pure_u50], ignore_index=True, sort=False)
    return profile_frame, overlap_frame, pd.concat([summary], ignore_index=True), selected_frame


def reveal_test_and_report(study: Path = STUDY) -> dict[str, object]:
    study = Path(study)
    freeze = _verify_barrier(study)
    new_curves, label_access = _new_learning_curves(study)
    historical, full = _historical_curves()
    curves = pd.concat([historical, new_curves], ignore_index=True)
    evaluation = evaluate(curves, full)
    partial, partial_summary = _partial_aulc(curves)
    paired_new = _paired_new_effects(partial, evaluation)
    costs = pd.concat([historical_costs(), _new_costs(study)], ignore_index=True, sort=False)
    write_evaluation(evaluation, costs, study)
    results = study / "results"
    new_curves.to_csv(results / "new_learning_curve_metrics.csv", index=False)
    partial.to_csv(results / "partial_aulc_per_seed.csv", index=False)
    partial_summary.to_csv(results / "partial_aulc_summary.csv", index=False)
    paired_new.to_csv(results / "paired_new_method_effects.csv", index=False)
    label_access.to_csv(results / "final_test_label_access_audit.csv", index=False)
    profiles, overlap, mechanism_summary, selected = _mechanism_tables(study)
    profiles.to_csv(results / "selection_profiles.csv", index=False)
    overlap.to_csv(results / "pure_u50_selection_overlap.csv", index=False)
    mechanism_summary.to_csv(results / "mechanism_summary.csv", index=False)
    selected.to_csv(results / "selected_batches.csv", index=False)
    aggregate = evaluation["metric_summary"]
    primary = aggregate.loc[aggregate.metric.eq("combined_normalized_RMSE")].set_index("method")
    whole_best = str(primary.aulc_mean.idxmin())
    endpoint_best = str(primary.endpoint_mean.idxmin())
    decision = {
        "status": "COMPLETE_DEVELOPMENT_EVIDENCE",
        "whole_curve_lowest_mean_NRMSE_AULC": whole_best,
        "endpoint_lowest_mean_NRMSE": endpoint_best,
        "new_methods": list(METHODS),
        "historical_baselines_reused_without_retraining": ["random", "hybrid", "lcmd", "kernel_ivr"],
        "global_pre_test_freeze_sha256": sha256_file(study / "global_pre_test_freeze.json"),
        "frozen_prediction_points": freeze["frozen_prediction_points"],
        "test_truth_access_after_global_barrier": True,
        "evidence_scope": "row-split development evidence; historical test cohort previously exposed",
    }
    atomic_json(study / "decision.json", decision)
    report = f"""# Gradient-MaxDet and fixed-U50 Gradient-MaxDet — final row study

Status: **{decision['status']}**. This is development evidence on the historically
exposed row cohort, not independent confirmation.

The lowest cohort-mean whole-curve combined-NRMSE AULC is `{whole_best}`. The
lowest mean endpoint combined NRMSE at 1005 labels is `{endpoint_best}`. Full
metrics, per-seed differences, early/middle/late partial AULCs, censored target
crossings, mechanism overlaps, and compute accounting are under `results/`.

All {freeze['frozen_prediction_points']} new-method prediction points and {len(CONFIRMATION_SEEDS) * ACQUISITION_ROUNDS} acquisition batches crossed one
global pre-test barrier before labels were opened. Historical Random, LCMD,
Hybrid, Kernel-IVR, matched full-data references, and exact reusable round-zero
artifacts were not retrained.
"""
    (study / "FINAL_REPORT.md").write_text(report)
    return decision
