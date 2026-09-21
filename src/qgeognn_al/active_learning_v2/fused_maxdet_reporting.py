"""Post-barrier metrics and compact mechanism report for Fusion-MaxDet."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import sha256_file
from ..resources import SOURCE_DATA
from ..training.predictor import atomic_json
from .benchmark_reporting import metric_row
from .fused_maxdet_runner import _json, _phase_budget
from .fused_maxdet_study import CONFIRMATION_SEEDS, METHOD, STUDY, historical_round0, validate_pre_test_manifest
from .protocol import RestrictedLabelStore
from .sequential_protocol import ACTIVE_LABEL_BUDGETS, ACQUISITION_ROUNDS, BATCH_SIZE
from .efficiency_reporting import crossings


BASELINE = STUDY.parents[2] / "studies/active_learning/qgeognn_v2_row_maxdet_b32"
SEQUENTIAL = STUDY.parents[2] / "studies/active_learning/qgeognn_v2_row_sequential_b32"


def _verify_global_freeze(study: Path) -> dict:
    freeze = _json(Path(study) / "global_pre_test_freeze.json")
    expected = len(CONFIRMATION_SEEDS) * (ACQUISITION_ROUNDS + 1)
    if freeze.get("status") != "FROZEN_BEFORE_TEST_TRUTH" or freeze.get("test_truth_access_count") != 0 or len(freeze.get("entries", {})) != expected:
        raise RuntimeError("fusion global pre-test barrier is incomplete")
    for key, entry in freeze["entries"].items():
        seed, round_name = key.split("/")
        path = Path(study) / "runtime" / seed / METHOD / round_name / "round_freeze.json"
        if sha256_file(path) != entry["round_freeze_sha256"]:
            raise RuntimeError(f"fusion round freeze changed: {key}")
        record = _json(path)
        if sha256_file(Path(record["checkpoint_path"])) != record["checkpoint_sha256"] or sha256_file(Path(record["prediction_path"])) != record["prediction_sha256"]:
            raise RuntimeError(f"fusion frozen prediction changed: {key}")
    return freeze


def _prediction_path(study: Path, seed: int, method: str, round_index: int) -> Path:
    if method == METHOD:
        record = _json(Path(study) / "runtime" / f"seed_{seed}" / METHOD / f"round_{round_index:02d}/round_freeze.json")
        return Path(record["prediction_path"])
    record_path = BASELINE / "runtime" / f"seed_{seed}" / "gradient_maxdet" / f"round_{round_index:02d}/round_freeze.json"
    record = _json(record_path)
    return Path(record["prediction_path"])


def _curves(study: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    profiles = []
    overlap = []
    for seed in CONFIRMATION_SEEDS:
        partition = pd.read_csv(Path(study) / "splits" / f"row_seed_{seed}.csv")
        test_ids = partition.loc[partition.role.eq("test"), "sample_id"].astype(str).tolist()
        context = _json(Path(study) / "runtime" / f"seed_{seed}/context.json")
        scales = context["preprocessing"]["target_scales"]
        store = RestrictedLabelStore(SOURCE_DATA, partition)
        selected = []
        for round_index in range(ACQUISITION_ROUNDS):
            selected_frame = pd.read_csv(Path(study) / "runtime" / f"seed_{seed}/{METHOD}/round_{round_index:02d}/acquisition_artifacts/selected_batch.csv")
            selected.extend(selected_frame.sample_id.astype(str).tolist())
            profile = _json(Path(study) / "runtime" / f"seed_{seed}/{METHOD}/round_{round_index:02d}/acquisition_artifacts/fusion_profile.json")
            profiles.append(profile)
            old_path = BASELINE / "runtime" / f"seed_{seed}/gradient_maxdet/round_{round_index:02d}/acquisition/selected_next_batch.csv"
            old = set(pd.read_csv(old_path).sample_id.astype(str)) if old_path.exists() else set()
            overlap.append({"outer_seed": seed, "source_round": round_index, "intersection_count": len(set(selected_frame.sample_id.astype(str)) & old), "overlap_fraction": profile["gradient_maxdet_overlap_fraction"]})
        store.freeze_acquisitions(selected)
        store.freeze_predictions()
        truth = store.reveal(test_ids, "final_test_evaluation")
        for method in ("gradient_maxdet", METHOD):
            for round_index, active in enumerate(ACTIVE_LABEL_BUDGETS):
                table = pd.read_csv(_prediction_path(study, seed, method, round_index))
                if table.sample_id.astype(str).tolist() != test_ids:
                    raise RuntimeError("frozen test prediction order drift")
                rows.append({"outer_seed": seed, "method": method, "round": round_index, **_phase_budget(active), **metric_row(truth, table.drop(columns="sample_id").to_numpy(float), scales)})
    return pd.DataFrame(rows), pd.DataFrame(profiles), pd.DataFrame(overlap)


def _summary(curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, crossings_rows = [], []
    for (seed, method), group in curves.groupby(["outer_seed", "method"]):
        group = group.sort_values("active_label_count")
        x = group.active_label_count.to_numpy(float)
        for metric in ("combined_normalized_RMSE", "V1_RMSE", "V2_RMSE", "V1_R2", "V2_R2"):
            y = group[metric].to_numpy(float)
            records.append({"outer_seed": seed, "method": method, "metric": metric, "aulc": float(np.trapezoid(y, x) / (x[-1] - x[0])), "final": float(y[-1])})
            # These are descriptive thresholds from the observed initial/final span.
            for name, fraction in (("N80", .8), ("N90", .9), ("N95", .95)):
                target = float(y[0] + fraction * (y[-1] - y[0]))
                crossings_rows.append({"outer_seed": seed, "method": method, "metric": metric, "target": name, **crossings(x, y, target, higher=metric.endswith("R2"))})
    return pd.DataFrame(records), pd.DataFrame(crossings_rows)


def reveal_test_and_report(study: Path = STUDY) -> dict[str, object]:
    study = Path(study)
    freeze = _verify_global_freeze(study)
    curves, profiles, overlap = _curves(study)
    summary, crossing = _summary(curves)
    results = study / "results"
    results.mkdir(parents=True, exist_ok=True)
    curves.to_csv(results / "learning_curve_metrics.csv", index=False)
    summary.to_csv(results / "aulc_and_final_metrics.csv", index=False)
    crossing.to_csv(results / "labels_to_target.csv", index=False)
    profiles.to_csv(results / "fusion_selection_profiles.csv", index=False)
    overlap.to_csv(results / "gradient_maxdet_selection_overlap.csv", index=False)
    paired = summary.pivot_table(index=["outer_seed", "metric"], columns="method", values=["aulc", "final"]).reset_index()
    paired.to_csv(results / "paired_gradient_vs_fusion.csv", index=False)
    cost = pd.DataFrame([{"outer_seed": seed, "method": METHOD, "training_seconds": float(pd.read_csv(study / "runtime" / f"seed_{seed}/{METHOD}/fit_audit.csv").training_seconds.sum()), "gradient_seconds": float(profiles.loc[profiles.outer_seed.eq(seed), "gradient_seconds"].sum()), "latent_seconds": float(profiles.loc[profiles.outer_seed.eq(seed), "latent_seconds"].sum()), "fusion_seconds": float(profiles.loc[profiles.outer_seed.eq(seed), "fusion_seconds"].sum()), "selector_seconds": float(profiles.loc[profiles.outer_seed.eq(seed), "selector_seconds"].sum())} for seed in CONFIRMATION_SEEDS])
    cost.to_csv(results / "compute_cost.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figures = study / "figures"
        figures.mkdir(exist_ok=True)
        for metric, filename in (("combined_normalized_RMSE", "nrmse_learning_curve.png"), ("V1_RMSE", "v1_rmse_learning_curve.png"), ("V2_RMSE", "v2_rmse_learning_curve.png")):
            fig, ax = plt.subplots(figsize=(7, 4))
            for method, group in curves.groupby("method"):
                mean = group.groupby("active_label_count")[metric].mean()
                ax.plot(mean.index, mean.values, marker=".", label=method)
            ax.set_xlabel("Active labels"); ax.set_ylabel(metric); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(figures / filename, dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4)); mean_overlap = overlap.groupby("source_round").overlap_fraction.mean(); ax.plot(mean_overlap.index, mean_overlap.values, marker="o"); ax.set_xlabel("Acquisition round"); ax.set_ylabel("Fusion/Gradient-MaxDet overlap"); fig.tight_layout(); fig.savefig(figures / "selection_overlap.png", dpi=160); plt.close(fig)
    except ImportError:
        pass
    decision = {"status": "COMPLETE_DEVELOPMENT_EVIDENCE", "methods": ["gradient_maxdet", METHOD], "seeds": list(CONFIRMATION_SEEDS), "frozen_prediction_points": freeze["frozen_prediction_points"], "mean_fusion_gradient_overlap_fraction": float(overlap.overlap_fraction.mean()), "test_truth_access_after_barrier": True}
    atomic_json(study / "decision.json", decision)
    return decision


__all__ = ["reveal_test_and_report"]
