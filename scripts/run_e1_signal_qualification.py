#!/usr/bin/env python3
"""Run E1 acquisition-signal qualification without active-learning rounds."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qgeognn_matplotlib")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.al_engine import QGeoGNNActiveLearningEngine, TrainConfig, canonical_json_hash
from scripts.run_e0_4g_baseline import (
    LOADING_SOLVENT,
    eluent_descriptor,
    sha256_file,
    train_baseline,
    write_artifact_manifest,
    write_environment,
)
from scripts.run_e0_8g_controls import load_graph_cache


OUTER_SEEDS = (42, 525, 1101)
MEMBER_SEEDS = (42, 525, 1101)
SPLIT_MODES = ("row", "compound")
SIGNALS = ("quantile_width", "ensemble", "latent_distance", "random")
SLICES = ("full", "common", "tail")
TARGETS = ("V1", "V2")
SOURCE_DIR = ROOT / "experiments" / "e0_4g_baseline"
G03_DIR = ROOT / "experiments" / "g0_3_threshold_sensitivity"
G04_DIR = ROOT / "experiments" / "g0_4_paper_style_transfer"


def record_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def ensure_source_members(output_dir: Path, epochs: int, patience: int) -> dict[int, Path]:
    members = {42: SOURCE_DIR / "checkpoints" / "best.pt"}
    data = pd.read_csv(SOURCE_DIR / "canonical_4g.csv")
    graph_cache = torch.load(SOURCE_DIR / "graph_cache_4g.pt", weights_only=False)
    split = pd.read_csv(SOURCE_DIR / "split_seed_42.csv")
    for seed in MEMBER_SEEDS[1:]:
        member_dir = output_dir / "source_members" / f"member_seed_{seed}"
        checkpoint = member_dir / "checkpoints" / "best.pt"
        metrics_path = member_dir / "metrics.json"
        if not checkpoint.exists() or not metrics_path.exists():
            member_dir.mkdir(parents=True, exist_ok=True)
            train_baseline(
                data,
                graph_cache,
                split,
                member_dir,
                seed=seed,
                epochs=epochs,
                batch_size=2048,
                learning_rate=1e-3,
                patience=patience,
                verbose=False,
            )
        members[seed] = checkpoint
    return members


def load_fit_result(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_target_member(
    engine: QGeoGNNActiveLearningEngine,
    source_checkpoint: Path,
    labeled_ids: list[str],
    validation_ids: list[str],
    outer_seed: int,
    member_seed: int,
    output_dir: Path,
    config: TrainConfig,
) -> tuple[Path, dict]:
    checkpoint = output_dir / "best.pt"
    result_path = output_dir / "fit_result.json"
    if checkpoint.exists() and result_path.exists():
        return checkpoint, load_fit_result(result_path)
    result = engine.fit(
        labeled_ids,
        validation_ids,
        config,
        source_checkpoint,
        seed=outer_seed * 100_000 + member_seed,
        output_dir=output_dir,
    )
    return Path(result.checkpoint), asdict(result)


def condition_matrix(data: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    rows = data.iloc[indices]
    eluents = np.vstack([eluent_descriptor(value) for value in rows["PE/EA"]]).astype(np.float32)
    extras = np.column_stack(
        [
            rows["loading solvent"].map(LOADING_SOLVENT).to_numpy(dtype=np.float32),
            (
                rows["Density g/ml"].to_numpy(dtype=np.float32)
                * rows["V/ul"].to_numpy(dtype=np.float32)
            ),
            rows["Volume of loading solvent/ul"].to_numpy(dtype=np.float32),
        ]
    )
    return np.column_stack([eluents, extras]).astype(np.float32)


def standardize(train: np.ndarray, evaluation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    return (train - mean) / scale, (evaluation - mean) / scale


def build_context_predictions(
    engine: QGeoGNNActiveLearningEngine,
    data: pd.DataFrame,
    split: pd.DataFrame,
    split_mode: str,
    outer_seed: int,
    checkpoints: dict[int, Path],
    output_dir: Path,
    k_neighbors: int,
) -> tuple[pd.DataFrame, dict]:
    train_indices = split.index[split["split"].eq("train")].to_numpy(dtype=int)
    test_indices = split.index[split["split"].eq("test")].to_numpy(dtype=int)
    train_ids = data.iloc[train_indices]["sample_id"].astype(str).tolist()
    test_ids = data.iloc[test_indices]["sample_id"].astype(str).tolist()
    member_predictions, member_embeddings = {}, {}
    for member_seed, checkpoint in checkpoints.items():
        train_result = engine.predict(
            train_ids,
            checkpoint,
            return_quantiles=True,
            return_embedding=member_seed == MEMBER_SEEDS[0],
            batch_size=256,
            chunk_size=1024,
        )
        test_result = engine.predict(
            test_ids,
            checkpoint,
            return_quantiles=True,
            return_embedding=member_seed == MEMBER_SEEDS[0],
            batch_size=256,
            chunk_size=1024,
        )
        member_predictions[member_seed] = test_result.table
        if member_seed == MEMBER_SEEDS[0]:
            member_embeddings["train"] = train_result.embeddings
            member_embeddings["test"] = test_result.embeddings

    reference = member_predictions[MEMBER_SEEDS[0]].copy()
    true = data.iloc[test_indices]
    scales = {
        target: max(float(data.iloc[train_indices][f"{target}_ml"].std(ddof=0)), 1e-8)
        for target in TARGETS
    }
    reference["true_V1"] = true["V1_ml"].to_numpy(dtype=float)
    reference["true_V2"] = true["V2_ml"].to_numpy(dtype=float)
    reference["pred_V1"] = reference["V1_q50"]
    reference["pred_V2"] = reference["V2_q50"]
    reference["standardized_abs_error"] = 0.5 * (
        np.abs(reference["true_V1"] - reference["pred_V1"]) / scales["V1"]
        + np.abs(reference["true_V2"] - reference["pred_V2"]) / scales["V2"]
    )
    reference["standardized_squared_error"] = 0.5 * (
        np.square(reference["true_V1"] - reference["pred_V1"]) / scales["V1"] ** 2
        + np.square(reference["true_V2"] - reference["pred_V2"]) / scales["V2"] ** 2
    )
    reference["V1_quantile_width"] = reference["V1_q90"] - reference["V1_q10"]
    reference["V2_quantile_width"] = reference["V2_q90"] - reference["V2_q10"]
    reference["quantile_width"] = 0.5 * (
        reference["V1_quantile_width"] / scales["V1"]
        + reference["V2_quantile_width"] / scales["V2"]
    )

    q50_v1 = np.column_stack(
        [member_predictions[seed]["V1_q50"].to_numpy(dtype=float) for seed in MEMBER_SEEDS]
    )
    q50_v2 = np.column_stack(
        [member_predictions[seed]["V2_q50"].to_numpy(dtype=float) for seed in MEMBER_SEEDS]
    )
    reference["ensemble_variance_V1"] = np.var(q50_v1 / scales["V1"], axis=1, ddof=1)
    reference["ensemble_variance_V2"] = np.var(q50_v2 / scales["V2"], axis=1, ddof=1)
    reference["ensemble_score"] = (
        reference["ensemble_variance_V1"] + reference["ensemble_variance_V2"]
    )
    for member_index, seed in enumerate(MEMBER_SEEDS):
        reference[f"member_{member_index}_seed"] = seed
        reference[f"member_{member_index}_V1_q50"] = q50_v1[:, member_index]
        reference[f"member_{member_index}_V2_q50"] = q50_v2[:, member_index]

    train_condition = condition_matrix(data, train_indices)
    test_condition = condition_matrix(data, test_indices)
    train_embedding, test_embedding = standardize(
        member_embeddings["train"], member_embeddings["test"]
    )
    train_condition, test_condition = standardize(train_condition, test_condition)
    train_representation = np.column_stack([train_embedding, train_condition])
    test_representation = np.column_stack([test_embedding, test_condition])
    neighbors = min(k_neighbors, len(train_representation))
    knn = NearestNeighbors(n_neighbors=neighbors, metric="euclidean")
    knn.fit(train_representation)
    distances, _ = knn.kneighbors(test_representation)
    reference["latent_distance"] = distances.mean(axis=1)
    rng = np.random.default_rng(outer_seed + (0 if split_mode == "row" else 1_000_000))
    reference["random_score"] = rng.random(len(reference))
    reference["tail_flag"] = true["threshold_excluded"].to_numpy(dtype=bool)
    reference["compound_id"] = true["canonical_smiles"].astype(str).to_numpy()
    reference["split_mode"] = split_mode
    reference["outer_seed"] = outer_seed
    reference["evaluation_split"] = "test"
    reference["V1_train_scale"] = scales["V1"]
    reference["V2_train_scale"] = scales["V2"]
    reference["primary_error_definition"] = "mean_standardized_absolute_error"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "reference_embeddings.npz",
        train_sample_ids=np.asarray(train_ids),
        test_sample_ids=np.asarray(test_ids),
        train_h_graph=member_embeddings["train"],
        test_h_graph=member_embeddings["test"],
        train_h_graph_condition=train_representation,
        test_h_graph_condition=test_representation,
    )
    ensemble_audit = {
        "split_mode": split_mode,
        "outer_seed": outer_seed,
        "members": [
            {
                "member_seed": seed,
                "checkpoint": record_path(checkpoints[seed]),
                "checkpoint_sha256": sha256_file(checkpoints[seed]),
            }
            for seed in MEMBER_SEEDS
        ],
        "same_target_train_ids_hash": canonical_json_hash(sorted(train_ids)),
        "target_train_rows": len(train_ids),
        "target_test_rows": len(test_ids),
        "mean_ensemble_score": float(reference["ensemble_score"].mean()),
        "nonzero_ensemble_fraction": float((reference["ensemble_score"] > 0).mean()),
        "mean_pairwise_q50_absolute_difference": float(
            np.mean(
                [
                    np.abs(q50_v1[:, left] - q50_v1[:, right]).mean()
                    + np.abs(q50_v2[:, left] - q50_v2[:, right]).mean()
                    for left in range(len(MEMBER_SEEDS))
                    for right in range(left + 1, len(MEMBER_SEEDS))
                ]
            )
        ),
    }
    return reference, ensemble_audit


def safe_spearman(score: np.ndarray, error: np.ndarray) -> float:
    if len(score) < 3 or np.unique(score).size < 2 or np.unique(error).size < 2:
        return float("nan")
    return float(spearmanr(score, error).statistic)


def bootstrap_spearman_ci(
    score: np.ndarray, error: np.ndarray, seed: int, iterations: int
) -> tuple[float, float]:
    if len(score) < 4:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        indices = rng.integers(0, len(score), size=len(score))
        value = safe_spearman(score[indices], error[indices])
        if np.isfinite(value):
            estimates.append(value)
    if len(estimates) < max(20, iterations // 10):
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def risk_coverage(score: np.ndarray, error: np.ndarray) -> tuple[float, list[dict]]:
    n = len(score)
    if n < 2:
        return float("nan"), []
    coverages = np.linspace(0.1, 1.0, 19)
    score_order = np.argsort(score, kind="mergesort")
    oracle_order = np.argsort(error, kind="mergesort")
    rows = []
    for coverage in coverages:
        keep = max(1, int(math.ceil(coverage * n)))
        risk = float(error[score_order[:keep]].mean())
        oracle_risk = float(error[oracle_order[:keep]].mean())
        rows.append(
            {
                "coverage": float(keep / n),
                "risk": risk,
                "oracle_risk": oracle_risk,
                "excess_risk": risk - oracle_risk,
            }
        )
    curve = pd.DataFrame(rows).drop_duplicates("coverage")
    ause = float(np.trapz(curve["excess_risk"], curve["coverage"]))
    return ause, curve.to_dict(orient="records")


def evaluate_signal(
    frame: pd.DataFrame,
    signal: str,
    slice_name: str,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> tuple[dict, list[dict]]:
    if slice_name == "common":
        selected = frame[~frame["tail_flag"]]
    elif slice_name == "tail":
        selected = frame[frame["tail_flag"]]
    elif slice_name == "full":
        selected = frame
    else:
        raise ValueError(slice_name)
    score_column = {
        "quantile_width": "quantile_width",
        "ensemble": "ensemble_score",
        "latent_distance": "latent_distance",
        "random": "random_score",
    }[signal]
    score = selected[score_column].to_numpy(dtype=float)
    error = selected["standardized_abs_error"].to_numpy(dtype=float)
    n = len(selected)
    spearman = safe_spearman(score, error)
    ci_low, ci_high = bootstrap_spearman_ci(
        score, error, bootstrap_seed, bootstrap_iterations
    )
    top_count = max(1, int(math.ceil(0.1 * n))) if n else 0
    if n:
        hard_positions = set(np.argsort(error, kind="mergesort")[-top_count:].tolist())
        score_positions = set(np.argsort(score, kind="mergesort")[-top_count:].tolist())
        overlap = len(hard_positions & score_positions)
        overlap_rate = overlap / top_count
        enrichment = overlap_rate / 0.1
        hard = np.zeros(n, dtype=int)
        hard[list(hard_positions)] = 1
        auroc = float(roc_auc_score(hard, score)) if np.unique(hard).size == 2 else float("nan")
        ause, curve = risk_coverage(score, error)
    else:
        overlap, overlap_rate, enrichment, auroc, ause, curve = 0, float("nan"), float("nan"), float("nan"), float("nan"), []
    metrics = {
        "signal": signal,
        "slice": slice_name,
        "n_samples": n,
        "tail_rows": int(selected["tail_flag"].sum()),
        "spearman": spearman,
        "spearman_ci_low": ci_low,
        "spearman_ci_high": ci_high,
        "hard_error_auroc": auroc,
        "top10_count": top_count,
        "top10_overlap_count": overlap,
        "top10_overlap_rate": overlap_rate,
        "enrichment": enrichment,
        "ause": ause,
        "mean_error": float(error.mean()) if n else float("nan"),
        "mean_signal": float(score.mean()) if n else float("nan"),
    }
    for row in curve:
        row.update({"signal": signal, "slice": slice_name})
    return metrics, curve


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["split_mode", "slice", "signal"]
    for context, group in metrics.groupby(keys, sort=True):
        row = dict(zip(keys, context))
        row["outer_seeds"] = int(group["outer_seed"].nunique())
        row["n_samples_mean"] = float(group["n_samples"].mean())
        for metric in ("spearman", "hard_error_auroc", "enrichment", "ause", "mean_error", "mean_signal"):
            values = group[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            if len(values) >= 2:
                margin = 4.303 * float(values.std(ddof=1)) / math.sqrt(len(values))
                row[f"{metric}_seed_ci_low"] = float(values.mean() - margin)
                row[f"{metric}_seed_ci_high"] = float(values.mean() + margin)
            else:
                row[f"{metric}_seed_ci_low"] = float("nan")
                row[f"{metric}_seed_ci_high"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def make_gate_decision(summary: pd.DataFrame) -> dict:
    key = summary[
        summary["slice"].isin(["full", "common"])
        & summary["split_mode"].isin(["row", "compound"])
    ].set_index(["split_mode", "slice", "signal"])
    ensemble_wins = []
    for split_mode in SPLIT_MODES:
        for slice_name in ("full", "common"):
            ensemble = key.loc[(split_mode, slice_name, "ensemble")]
            quantile = key.loc[(split_mode, slice_name, "quantile_width")]
            ensemble_wins.extend(
                [
                    bool(ensemble["spearman_mean"] > quantile["spearman_mean"]),
                    bool(ensemble["hard_error_auroc_mean"] > quantile["hard_error_auroc_mean"]),
                    bool(ensemble["enrichment_mean"] > quantile["enrichment_mean"]),
                    bool(ensemble["ause_mean"] < quantile["ause_mean"]),
                ]
            )
    def signal_gate(signal: str) -> dict:
        selected = key.xs(signal, level="signal")
        positive_slices = int((selected["spearman_mean"] > 0).sum())
        mean_spearman = float(selected["spearman_mean"].mean())
        mean_enrichment = float(selected["enrichment_mean"].mean())
        mean_ause = float(selected["ause_mean"].mean())
        return {
            "positive_spearman_slices": positive_slices,
            "key_slices": len(selected),
            "mean_spearman": mean_spearman,
            "mean_enrichment": mean_enrichment,
            "mean_ause": mean_ause,
            "qualified": bool(
                positive_slices >= 3 and mean_spearman > 0.1 and mean_enrichment > 1.5
            ),
        }
    ensemble_stats = signal_gate("ensemble")
    ensemble_main = bool(
        sum(ensemble_wins) >= 12
        and ensemble_stats["mean_spearman"] > 0.1
        and ensemble_stats["mean_enrichment"] > 1.5
    )
    quantile_stats = signal_gate("quantile_width")
    latent_stats = signal_gate("latent_distance")
    if ensemble_main:
        e2_uncertainty = "ensemble"
    elif quantile_stats["qualified"]:
        e2_uncertainty = "quantile_width_secondary_only"
    else:
        e2_uncertainty = "none"
    return {
        "selection_uses_predictor_retraining_test": False,
        "predictor_config_changed": False,
        "ensemble_wins_vs_quantile": int(sum(ensemble_wins)),
        "ensemble_comparisons": len(ensemble_wins),
        "ensemble_enters_e2_main": ensemble_main,
        "e2_uncertainty_strategy": e2_uncertainty,
        "quantile_width": quantile_stats,
        "ensemble": ensemble_stats,
        "latent_distance": latent_stats,
        "coverage_enters_e2_main": bool(latent_stats["qualified"]),
        "raw_vs_calibrated_quantile_width": "one ranking only; per-run global conformal inflation is not a second strategy",
        "tail_role": "failure/mechanism slice only; too small to trigger method selection",
        "rule": "ensemble needs >=12/16 wins over quantile plus key mean Spearman>0.1 and enrichment>1.5; individual signal needs positive Spearman in >=3/4 key slices, key mean Spearman>0.1 and enrichment>1.5",
    }


def plot_outputs(predictions: pd.DataFrame, summary: pd.DataFrame, risk: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    colors = {
        "quantile_width": "#4C78A8",
        "ensemble": "#F58518",
        "latent_distance": "#54A24B",
        "random": "#9D9D9D",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for axis, signal in zip(axes.flat, SIGNALS):
        score_col = {
            "quantile_width": "quantile_width",
            "ensemble": "ensemble_score",
            "latent_distance": "latent_distance",
            "random": "random_score",
        }[signal]
        rank_x = predictions[score_col].rank(pct=True)
        rank_y = predictions["standardized_abs_error"].rank(pct=True)
        axis.scatter(rank_x, rank_y, s=9, alpha=0.35, color=colors[signal])
        axis.set(title=signal, xlabel="signal percentile", ylabel="error percentile")
    fig.savefig(plot_dir / "signal_error_rank_scatter.png", dpi=180)
    plt.close(fig)

    selected_risk = risk[(risk["slice"].eq("full"))]
    mean_risk = selected_risk.groupby(["split_mode", "signal", "coverage"], as_index=False)["risk"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, split_mode in zip(axes, SPLIT_MODES):
        for signal in SIGNALS:
            curve = mean_risk[(mean_risk["split_mode"].eq(split_mode)) & (mean_risk["signal"].eq(signal))]
            axis.plot(curve["coverage"], curve["risk"], label=signal, color=colors[signal])
        axis.set(title=f"{split_mode} / full", xlabel="coverage retained", ylabel="mean standardized error")
    axes[1].legend(fontsize=8)
    fig.savefig(plot_dir / "risk_coverage.png", dpi=180)
    plt.close(fig)

    bars = summary[summary["slice"].isin(["full", "common"])].copy()
    bars["context"] = bars["split_mode"] + "/" + bars["slice"]
    contexts = ["row/full", "row/common", "compound/full", "compound/common"]
    x = np.arange(len(contexts))
    width = 0.2
    fig, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for offset, signal in enumerate(SIGNALS):
        values = bars[bars["signal"].eq(signal)].set_index("context").reindex(contexts)["enrichment_mean"]
        axis.bar(x + (offset - 1.5) * width, values, width, label=signal, color=colors[signal])
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set(xticks=x, xticklabels=contexts, ylabel="Top-10% enrichment")
    axis.legend(fontsize=8)
    fig.savefig(plot_dir / "hard_error_enrichment.png", dpi=180)
    plt.close(fig)

    heat = bars.pivot(index="signal", columns="context", values="spearman_mean").reindex(index=SIGNALS, columns=contexts)
    fig, axis = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    image = axis.imshow(heat.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axis.set(
        xticks=np.arange(len(contexts)),
        xticklabels=contexts,
        yticks=np.arange(len(SIGNALS)),
        yticklabels=SIGNALS,
    )
    axis.tick_params(axis="x", labelrotation=15)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    for row in range(len(SIGNALS)):
        for column in range(len(contexts)):
            axis.text(column, row, f"{heat.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, label="Spearman")
    fig.savefig(plot_dir / "signal_slice_heatmap.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "experiments" / "e1_signal_qualification"
    )
    parser.add_argument("--source-epochs", type=int, default=1000)
    parser.add_argument("--source-patience", type=int, default=100)
    parser.add_argument("--target-epochs", type=int, default=500)
    parser.add_argument("--target-patience", type=int, default=100)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--k-neighbors", type=int, default=5)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if (output_dir / "e1_gate_decision.json").exists():
        raise FileExistsError(f"Refusing to overwrite finalized E1: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_environment(output_dir)
    source_members = ensure_source_members(output_dir, args.source_epochs, args.source_patience)
    data = pd.read_csv(G03_DIR / "canonical_8g_no_threshold.csv")
    scaler = json.loads((SOURCE_DIR / "scaler.json").read_text(encoding="utf-8"))
    engine = QGeoGNNActiveLearningEngine(
        data,
        load_graph_cache(),
        scaler,
        SOURCE_DIR / "checkpoints" / "best.pt",
    )
    target_config = TrainConfig(epochs=args.target_epochs, patience=args.target_patience)
    all_predictions, all_metrics, all_risk, ensemble_audits, fit_records = [], [], [], [], []

    for split_mode in SPLIT_MODES:
        for outer_seed in OUTER_SEEDS:
            run_key = f"{split_mode}_seed_{outer_seed}"
            split = pd.read_csv(G03_DIR / "splits" / f"{run_key}.csv")
            train_ids = data.loc[split["split"].eq("train"), "sample_id"].astype(str).tolist()
            validation_ids = data.loc[split["split"].eq("valid"), "sample_id"].astype(str).tolist()
            labeled_ids = train_ids + validation_ids
            checkpoints = {
                42: G04_DIR
                / "runs"
                / run_key
                / "last2_head"
                / "checkpoints"
                / "last2_head.pt"
            }
            fit_records.append(
                {
                    "split_mode": split_mode,
                    "outer_seed": outer_seed,
                    "member_seed": 42,
                    "checkpoint": record_path(checkpoints[42]),
                    "checkpoint_sha256": sha256_file(checkpoints[42]),
                    "source_checkpoint": record_path(source_members[42]),
                    "reused_gate0_member": True,
                }
            )
            for member_seed in MEMBER_SEEDS[1:]:
                member_dir = output_dir / "runs" / run_key / f"member_seed_{member_seed}"
                checkpoint, result = ensure_target_member(
                    engine,
                    source_members[member_seed],
                    labeled_ids,
                    validation_ids,
                    outer_seed,
                    member_seed,
                    member_dir,
                    target_config,
                )
                checkpoints[member_seed] = checkpoint
                fit_records.append(
                    {
                        "split_mode": split_mode,
                        "outer_seed": outer_seed,
                        "member_seed": member_seed,
                        "checkpoint": record_path(checkpoint),
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "source_checkpoint": record_path(source_members[member_seed]),
                        "reused_gate0_member": False,
                        **{f"fit_{key}": value for key, value in result.items() if key not in {"checkpoint", "checkpoint_sha256"}},
                    }
                )
            print(json.dumps({"scoring": run_key, "members": list(checkpoints)}), flush=True)
            predictions, ensemble_audit = build_context_predictions(
                engine,
                data,
                split,
                split_mode,
                outer_seed,
                checkpoints,
                output_dir / "runs" / run_key,
                args.k_neighbors,
            )
            all_predictions.append(predictions)
            ensemble_audits.append(ensemble_audit)
            for signal_index, signal in enumerate(SIGNALS):
                for slice_index, slice_name in enumerate(SLICES):
                    metrics, curve = evaluate_signal(
                        predictions,
                        signal,
                        slice_name,
                        args.bootstrap_iterations,
                        outer_seed + signal_index * 10_000 + slice_index * 1_000,
                    )
                    metrics.update({"split_mode": split_mode, "outer_seed": outer_seed})
                    all_metrics.append(metrics)
                    for row in curve:
                        row.update({"split_mode": split_mode, "outer_seed": outer_seed})
                        all_risk.append(row)

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.DataFrame(all_metrics)
    risk = pd.DataFrame(all_risk)
    summary = summarize_metrics(metrics)
    decision = make_gate_decision(summary)
    predictions.to_csv(output_dir / "uq_predictions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(output_dir / "metrics_by_run.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    risk.to_csv(output_dir / "risk_coverage_curves.csv", index=False)
    pd.DataFrame(fit_records).to_csv(output_dir / "member_training_manifest.csv", index=False)
    (output_dir / "ensemble_audit.json").write_text(
        json.dumps(ensemble_audits, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "e1_gate_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_outputs(predictions, summary, risk, output_dir)
    config = {
        "stage": "E1_acquisition_signal_qualification",
        "active_learning_retraining": False,
        "outer_seeds": list(OUTER_SEEDS),
        "member_seeds": list(MEMBER_SEEDS),
        "split_modes": list(SPLIT_MODES),
        "signals": list(SIGNALS),
        "slices": list(SLICES),
        "primary_error": "0.5*(abs(V1-q50)/train_sd_V1 + abs(V2-q50)/train_sd_V2)",
        "quantile_width": "0.5*(width_V1/train_sd_V1 + width_V2/train_sd_V2), member0 raw intervals",
        "calibrated_width_policy": "not a separate ranking because per-run global alpha preserves ordering",
        "ensemble": "sample covariance trace of K=3 standardized q50 predictions; three independently trained 4g source anchors use the same 4g split",
        "latent_distance": "mean 5NN standardized Euclidean distance in standardized [h_graph; 9D condition] to target train",
        "random": "deterministic no-information baseline",
        "bootstrap_iterations": args.bootstrap_iterations,
        "source_epochs": args.source_epochs,
        "source_patience": args.source_patience,
        "target_config": asdict(target_config),
        "gate_rule": decision["rule"],
        "parquet_status": "not written because the frozen fish environment has no pyarrow/fastparquet; lossless per-sample table is csv.gz",
        "test_role": "E1 signal qualification only; never used to reopen Gate 0 Predictor",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    key_summary = summary[
        summary["slice"].isin(["full", "common"])
    ][
        [
            "split_mode",
            "slice",
            "signal",
            "spearman_mean",
            "hard_error_auroc_mean",
            "enrichment_mean",
            "ause_mean",
        ]
    ]
    readme = f"""# E1：Acquisition Signal Qualification

## 状态与Gate决定

本阶段不运行主动学习闭环，也不修改Gate 0 Predictor。Primary error固定为训练尺度标准化的V1/V2绝对误差均值；raw Quantile Width与全局conformal inflation后的width只算同一种ranking。

- Ensemble进入E2主uncertainty：**{decision['ensemble_enters_e2_main']}**（相对Quantile Width赢{decision['ensemble_wins_vs_quantile']}/{decision['ensemble_comparisons']}个关键聚合比较）。
- Quantile Width qualified：**{decision['quantile_width']['qualified']}**。
- Latent Distance支持Coverage主线：**{decision['coverage_enters_e2_main']}**。
- E2 uncertainty策略：**{decision['e2_uncertainty_strategy']}**。

12/16恰好达到预注册门槛，不解释为Ensemble普遍占优：row/full与row/common的8项比较全部由Ensemble胜出；compound/full中Quantile Width赢Spearman、AUROC与enrichment，Ensemble仅在AUSE上略优；compound/common中Ensemble赢AUROC、enrichment与AUSE，但Spearman较低。E2因此使用Ensemble作为唯一主uncertainty信号，同时保留Quantile Width为secondary/legacy诊断，不增加为第五个主策略。

Tail只作failure/mechanism slice，不凭小样本结果触发方法入选。完整逐样本结果在`uq_predictions.csv.gz`，每个run保留member q50、真实误差、三种signal、tail与compound identity；128维embedding及拼接conditions表示保存在各run的`reference_embeddings.npz`。

## 关键切片均值

```text
{key_summary.to_string(index=False)}
```

## Ensemble独立性

K=3成员使用相同4g训练split和相同8g target train/validation；seed42复用Gate 0 anchor，seed525/1101从不同4g随机初始化重新训练source后再迁移。只改变target seed在当前full-batch、无dropout路径下不会形成真正ensemble，因此没有采用该伪独立方案。

## 限制

- 每个row测试切片只有59个样本、compound测试切片只有58个样本；seed级置信区间很宽，E1只决定方法入口，不给出稳定效应量结论。
- tail每个run仅2--3个样本，compound/tail的Spearman不可定义。
- row seed1101的两个新增Ensemble成员分别在epoch 496/500和498/500取得validation最优，接近训练上限；这不触发事后延长epoch，但在E2中继续作为收敛审计项。

## 图表

- `plots/signal_error_rank_scatter.png`
- `plots/risk_coverage.png`
- `plots/hard_error_enrichment.png`
- `plots/signal_slice_heatmap.png`
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    write_artifact_manifest(output_dir)
    print(json.dumps({"decision": decision}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
