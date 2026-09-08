#!/usr/bin/env python3
"""Reproduce the paper-aligned 4g-to-25g/40g transfer protocol without test tuning."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_e0_4g_baseline import sha256_file, write_artifact_manifest, write_environment
from scripts.run_e0_8g_controls import load_graph_cache
from scripts.run_e0_8g_transfer import (
    SOURCE_4G_DIR,
    build_model,
    build_model_data,
    make_loaders,
    quantile_target_loss,
    set_training_mode,
    validation_scores,
)
from scripts.run_g0_1_quantile_monotonicity import install_monotonic_head, predict
from scripts.run_g0_4_paper_style_transfer import (
    add_column_spec,
    configure_g04_trainable,
    load_transferred_model,
    set_feature_schema,
)


STUDY = ROOT / "studies" / "transfer" / "paper_transfer_reproduction"
AUDIT_DIR = ROOT / "studies" / "transfer" / "cross_column" / "data_audit"
SOURCE_CHECKPOINT = SOURCE_4G_DIR / "checkpoints" / "best.pt"
SOURCE_SCALER = SOURCE_4G_DIR / "scaler.json"
SEEDS = (42, 525, 1101, 2025, 2026)
METHODS = ("direct", "paper_transfer")
TARGETS = ("V1", "V2")
METRICS = ("V1_r2", "V1_rmse", "V1_mae", "V2_r2", "V2_rmse", "V2_mae")
PAPER_R2 = {
    "8g": (0.759, 0.752),
    "25g": (0.747, 0.840),
    "40g": (0.826, 0.824),
}
COLUMN_CONFIG = {
    "25g": {
        "spec": (2.15, 15.6, 0.5248),
        "thresholds": (60.0, 120.0),
        "raw_hash": "6ef36adfa8a73d5b51560cd39776cae78c731a1107b3dfdf2b90661942ac5d06",
        "reader_rows": 490,
        "legacy_rows": 408,
    },
    "40g": {
        "spec": (2.15, 15.6, 0.5248),
        "thresholds": (150.0, 200.0),
        "raw_hash": "2d89921b4c84c720a7b3ae97e76d7a996a8f296c417ac42c29adf68d21574e0e",
        "reader_rows": 529,
        "legacy_rows": 456,
    },
}
SHARED_SPEC_FLAG = "25G_AND_40G_SHARE_LEGACY_COLUMN_SPEC_VALUES"


def target_canonical_path(column: str) -> Path:
    return AUDIT_DIR / f"canonical_{column}.csv"


def target_graph_path(column: str) -> Path:
    return AUDIT_DIR / f"graph_cache_{column}_only.pt"


def read_target_data(column: str, protocol: str) -> pd.DataFrame:
    """Load the audited raw-derived canonical table and apply only named protocol rules."""
    config = COLUMN_CONFIG[column]
    raw_path = ROOT / "dataset" / f"dataset_{column}.csv"
    if sha256_file(raw_path) != config["raw_hash"]:
        raise ValueError(f"Raw dataset hash changed: {raw_path}")
    data = pd.read_csv(target_canonical_path(column)).copy()
    if len(data) != config["reader_rows"]:
        raise ValueError(f"Unexpected reader-compatible row count for {column}: {len(data)}")
    if protocol == "legacy_filtered":
        v1_limit, v2_limit = config["thresholds"]
        data = data.loc[(data["V1_ml"] <= v1_limit) & (data["V2_ml"] <= v2_limit)].copy()
        if len(data) != config["legacy_rows"]:
            raise ValueError(f"Unexpected legacy-filtered row count for {column}: {len(data)}")
    elif protocol != "no_threshold":
        raise ValueError(protocol)
    data = data.reset_index(drop=True)
    data["column"] = column
    data["protocol"] = protocol
    data["repository_original_column_spec"] = [list(config["spec"])] * len(data)
    return data


def make_row_split(data: pd.DataFrame, column: str, protocol: str, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(len(data))
    train_end = int(0.8 * len(data))
    valid_end = train_end + int(0.1 * len(data))
    labels = np.full(len(data), "test", dtype=object)
    labels[permutation[:train_end]] = "train"
    labels[permutation[train_end:valid_end]] = "valid"
    split = data[["sample_id", "source_row_1based", "canonical_smiles"]].copy()
    split["canonical_index"] = np.arange(len(data), dtype=int)
    split["split"] = labels
    split["column"] = column
    split["protocol"] = protocol
    split["seed"] = seed
    expected = {"train": train_end, "valid": valid_end - train_end, "test": len(data) - valid_end}
    observed = split["split"].value_counts().to_dict()
    if observed != expected:
        raise AssertionError((column, protocol, seed, observed, expected))
    return split


def load_target_graph_cache(column: str) -> dict:
    cache = dict(load_graph_cache())
    cache.update(torch.load(target_graph_path(column), weights_only=False))
    return cache


def initialize_model(method: str, device: torch.device) -> tuple[nn.Module, list[str], list[str]]:
    set_feature_schema(True)
    if method == "paper_transfer":
        model, missing, unexpected = load_transferred_model("paper_style", device, SOURCE_CHECKPOINT)
        configure_g04_trainable(model, "paper_style")
        return model, missing, unexpected
    if method == "direct":
        model = build_model(device)
        install_monotonic_head(model)
        configure_g04_trainable(model, "full_finetune")
        return model, [], []
    raise ValueError(method)


def combined_normalized_rmse(metrics: dict[str, float], variance: dict[str, float]) -> float:
    value = 0.5 * (
        metrics["V1_rmse"] ** 2 / variance["V1"]
        + metrics["V2_rmse"] ** 2 / variance["V2"]
    )
    return float(np.sqrt(value))


def prediction_rows(
    model: nn.Module,
    loaders: dict,
    data: pd.DataFrame,
    column: str,
    protocol: str,
    method: str,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    for split_name in ("valid", "test"):
        _, truth, prediction, indices = predict(model, loaders[split_name], device)
        for row_index, true, quantiles in zip(indices, truth, prediction):
            source = data.iloc[int(row_index)]
            rows.append(
                {
                    "column": column,
                    "protocol": protocol,
                    "method": method,
                    "seed": seed,
                    "evaluation_split": split_name,
                    "sample_id": source["sample_id"],
                    "source_row_1based": int(source["source_row_1based"]),
                    "canonical_smiles": source["canonical_smiles"],
                    "V1_true": float(true[0]),
                    "V2_true": float(true[1]),
                    "V1_q10": float(quantiles[0]),
                    "V1_q50": float(quantiles[1]),
                    "V1_q90": float(quantiles[2]),
                    "V2_q10": float(quantiles[3]),
                    "V2_q50": float(quantiles[4]),
                    "V2_q90": float(quantiles[5]),
                    "column_spec_flag": SHARED_SPEC_FLAG,
                }
            )
    return pd.DataFrame(rows)


def train_run(
    column: str,
    protocol: str,
    method: str,
    seed: int,
    data: pd.DataFrame,
    split: pd.DataFrame,
    output_dir: Path,
    epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[dict, pd.DataFrame]:
    run_dir = output_dir / "runs" / protocol / column / f"row_seed_{seed}" / method
    checkpoint_dir = run_dir / "checkpoints"
    history_dir = run_dir / "histories"
    checkpoint_path = checkpoint_dir / "best.pt"
    result_path = run_dir / "result.json"
    predictions_path = run_dir / "predictions.csv.gz"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists() and result_path.exists() and predictions_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8")), pd.read_csv(predictions_path)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_cache = load_target_graph_cache(column)
    scaler = json.loads(SOURCE_SCALER.read_text(encoding="utf-8"))
    base_atom, angle_data = build_model_data(data, graph_cache, split, scaler)
    atom_data = add_column_spec(base_atom, COLUMN_CONFIG[column]["spec"])
    loaders = make_loaders(atom_data, angle_data, split, batch_size)
    train_mask = split["split"].eq("train").to_numpy()
    labels = data.loc[train_mask, ["V1_ml", "V2_ml"]]
    variance = {target: float(labels[f"{target}_ml"].var(ddof=0)) for target in TARGETS}
    if any(value <= 0 for value in variance.values()):
        raise ValueError(f"Invalid training variance: {variance}")

    model, missing, unexpected = initialize_model(method, device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
        weight_decay=1e-5,
    )

    history = []
    best_score, best_epoch, stale = float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        set_training_mode(model)
        total_loss, batches = 0.0, 0
        for atom_batch, angle_batch in zip(*loaders["train"]):
            atom_batch, angle_batch = atom_batch.to(device), angle_batch.to(device)
            prediction, _ = model(atom_batch, angle_batch)
            loss = quantile_target_loss(atom_batch.y[:, 0], prediction[:, 0:3])
            loss = loss + quantile_target_loss(atom_batch.y[:, 1], prediction[:, 3:6])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        valid_metrics, _, _, _ = predict(model, loaders["valid"], device)
        normalized_score, legacy_score = validation_scores(valid_metrics, variance)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(batches, 1),
                "normalized_valid_score": normalized_score,
                "legacy_valid_score": legacy_score,
                **valid_metrics,
            }
        )
        if normalized_score < best_score:
            best_score, best_epoch, stale = normalized_score, epoch, 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "column": column,
                    "protocol": protocol,
                    "method": method,
                    "seed": seed,
                    "epoch": epoch,
                    "normalized_valid_score": normalized_score,
                    "source_checkpoint_sha256": sha256_file(SOURCE_CHECKPOINT),
                    "column_spec": list(COLUMN_CONFIG[column]["spec"]),
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if stale >= patience:
            break

    pd.DataFrame(history).to_csv(history_dir / "history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    valid_metrics, _, _, _ = predict(model, loaders["valid"], device)
    test_metrics, _, _, _ = predict(model, loaders["test"], device)
    predictions = prediction_rows(model, loaders, data, column, protocol, method, seed, device)
    predictions.to_csv(predictions_path, index=False, compression="gzip")
    result = {
        "column": column,
        "protocol": protocol,
        "method": method,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "source_checkpoint": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
        "source_checkpoint_sha256": sha256_file(SOURCE_CHECKPOINT),
        "source_preprocessing": "experiments/e0_4g_baseline/scaler.json",
        "column_spec": list(COLUMN_CONFIG[column]["spec"]),
        "column_spec_flag": SHARED_SPEC_FLAG,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_parameter_names": trainable_names,
        "source_missing_keys": missing,
        "source_unexpected_keys": unexpected,
        "selection": "validation_only_minimum_combined_normalized_rmse",
        "normalized_valid_score": validation_scores(valid_metrics, variance)[0],
        "normalized_test_score": validation_scores(test_metrics, variance)[0],
        "combined_normalized_valid_rmse": combined_normalized_rmse(valid_metrics, variance),
        "combined_normalized_test_rmse": combined_normalized_rmse(test_metrics, variance),
        "valid": valid_metrics,
        "test": test_metrics,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result, predictions


def flat_metric_rows(result: dict) -> list[dict]:
    rows = []
    for split_name in ("valid", "test"):
        metrics = result[split_name]
        rows.append(
            {
                "column": result["column"],
                "protocol": result["protocol"],
                "method": result["method"],
                "seed": result["seed"],
                "evaluation_split": split_name,
                "best_epoch": result["best_epoch"],
                "epochs_run": result["epochs_run"],
                "selection": result["selection"],
                "column_spec_flag": result["column_spec_flag"],
                "combined_normalized_rmse": result[f"combined_normalized_{split_name}_rmse"],
                **{key: metrics[key] for key in METRICS},
            }
        )
    return rows


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics.loc[metrics["evaluation_split"].eq("test")].copy()
    groups = test.groupby(["column", "protocol", "method"], sort=False)
    rows = []
    for keys, group in groups:
        row = dict(zip(("column", "protocol", "method"), keys))
        row["seeds"] = int(group["seed"].nunique())
        for metric in (*METRICS, "combined_normalized_rmse"):
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sample_sd"] = float(values.std(ddof=1))
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def reference_rows() -> pd.DataFrame:
    source = json.loads((SOURCE_4G_DIR / "metrics.json").read_text(encoding="utf-8"))["test"]
    rows = [
        {
            "column": "4g",
            "protocol": "source_reference",
            "method": "source_reference",
            "seed": 42,
            "evaluation_split": "test",
            "best_epoch": 91,
            "epochs_run": 191,
            "selection": "existing_source_validation_best",
            "column_spec_flag": "not_applicable",
            "combined_normalized_rmse": np.nan,
            **{key: source[key] for key in METRICS},
        }
    ]
    for seed in (42, 525, 1101):
        result_path = (
            ROOT
            / "experiments"
            / "g0_4_paper_style_transfer"
            / "runs"
            / f"row_seed_{seed}"
            / "paper_style"
            / "paper_style.result.json"
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "column": "8g",
                "protocol": "existing_8g_no_threshold",
                "method": "paper_transfer",
                "seed": seed,
                "evaluation_split": "test",
                "best_epoch": result["best_epoch"],
                "epochs_run": result["epochs_run"],
                "selection": "existing_g0_4_validation_only",
                "column_spec_flag": "repository_original_8g_spec",
                "combined_normalized_rmse": np.nan,
                **{key: result["test"][key] for key in METRICS},
            }
        )
    return pd.DataFrame(rows)


def reference_summary(reference: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (column, protocol, method), group in reference.groupby(["column", "protocol", "method"], sort=False):
        row = {"column": column, "protocol": protocol, "method": method, "seeds": int(group.seed.nunique())}
        for metric in METRICS:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sample_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def paper_r2_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ("8g", "25g", "40g"):
        row = summary.loc[(summary["column"] == column) & (summary["method"] == "paper_transfer")]
        if column != "8g":
            row = row.loc[row["protocol"].eq("legacy_filtered")]
        if len(row) != 1:
            raise ValueError(f"Missing primary paper-transfer summary for {column}")
        value = row.iloc[0]
        paper_v1, paper_v2 = PAPER_R2[column]
        rows.append(
            {
                "column": column,
                "paper_r2_v1": paper_v1,
                "reproduction_r2_v1_mean": value["V1_r2_mean"],
                "difference_v1": value["V1_r2_mean"] - paper_v1,
                "paper_r2_v2": paper_v2,
                "reproduction_r2_v2_mean": value["V2_r2_mean"],
                "difference_v2": value["V2_r2_mean"] - paper_v2,
                "reference_note": "Published Figure 4 value supplied by the task; never used for selection.",
            }
        )
    return pd.DataFrame(rows)


def write_scale_note(output_dir: Path, summary: pd.DataFrame) -> None:
    scale_path = ROOT / "studies" / "transfer" / "physics_column_conditioned_transfer" / "budget100_metrics.csv"
    scale = pd.read_csv(scale_path)
    scale = scale.loc[
        scale["protocol"].eq("row")
        & scale["budget"].eq(100)
        & scale["method"].eq("scale_only")
        & scale["column"].isin(["8g", "25g", "40g"])
    ]
    scale_summary = scale.groupby("column")[["V1_rmse", "V2_rmse"]].mean()
    paper = summary.loc[summary["method"].eq("paper_transfer")].set_index("column")
    lines = [
        "# Paper transfer versus Scale: descriptive only",
        "",
        "This is not a head-to-head comparison. Paper-transfer primary runs use about 80% of legacy-filtered target rows, while the frozen Scale reference uses no target threshold and a budget of 100 labels. Neither table selected a model using the other table's test results.",
        "",
        "| column | paper-transfer V1 RMSE (mL) | paper-transfer V2 RMSE (mL) | frozen Scale budget-100 V1 RMSE (mL) | frozen Scale budget-100 V2 RMSE (mL) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for column in ("8g", "25g", "40g"):
        paper_row = paper.loc[column]
        scale_row = scale_summary.loc[column]
        lines.append(
            f"| {column} | {paper_row['V1_rmse_mean']:.2f} | {paper_row['V2_rmse_mean']:.2f} | {scale_row['V1_rmse']:.2f} | {scale_row['V2_rmse']:.2f} |"
        )
    (output_dir / "PAPER_TRANSFER_VS_SCALE_DESCRIPTIVE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(output_dir: Path, summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    primary = summary.loc[summary["protocol"].eq("legacy_filtered")].copy()
    transfer = primary.loc[primary["method"].eq("paper_transfer")].set_index("column")
    direct = primary.loc[primary["method"].eq("direct")].set_index("column")
    lines = [
        "# Paper-transfer reproduction report",
        "",
        "## Protocol",
        "",
        "This is a paper-aligned reconstructed reproduction, not an exact direct execution of a complete 25g/40g author training path. It loads the fixed 4g source, appends repository-original column values, function-preservingly initializes new adapters, uses the G0-4 monotonic head and last2-plus-head transfer scope, and selects the checkpoint using validation only. Direct training is random initialization with all parameters trainable. Both use identical random row splits, source preprocessing, loss, and held-out test rows.",
        "",
        "## Main results",
        "",
        "| 柱规格 | 方法 | V1 R2 (mean +/- SD) | V1 RMSE mL (mean +/- SD) | V1 MAE mL (mean +/- SD) | V2 R2 (mean +/- SD) | V2 RMSE mL (mean +/- SD) | V2 MAE mL (mean +/- SD) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for column in ("25g", "40g"):
        for method, label in (("direct", "direct"), ("paper_transfer", "paper_transfer")):
            row = primary.loc[(primary["column"] == column) & (primary["method"] == method)].iloc[0]
            lines.append(
                f"| {column} | {label} | {row['V1_r2_mean']:.3f} +/- {row['V1_r2_sample_sd']:.3f} | {row['V1_rmse_mean']:.2f} +/- {row['V1_rmse_sample_sd']:.2f} | {row['V1_mae_mean']:.2f} +/- {row['V1_mae_sample_sd']:.2f} | {row['V2_r2_mean']:.3f} +/- {row['V2_r2_sample_sd']:.3f} | {row['V2_rmse_mean']:.2f} +/- {row['V2_rmse_sample_sd']:.2f} | {row['V2_mae_mean']:.2f} +/- {row['V2_mae_sample_sd']:.2f} |"
            )
    lines.extend([
        "",
        "The full seed-level values, medians, minima, and maxima are in `all_metrics.csv` and `PAPER_TRANSFER_RMSE_SUMMARY.csv`.",
        "",
        "## Figure 4 R2 sanity check",
        "",
        "| column | paper R2 V1 | reproduction R2 V1 | paper R2 V2 | reproduction R2 V2 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.column} | {row.paper_r2_v1:.3f} | {row.reproduction_r2_v1_mean:.3f} | {row.paper_r2_v2:.3f} | {row.reproduction_r2_v2_mean:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "R2 measures explained variation within a target test distribution, whereas RMSE is an absolute deviation in mL. Larger-column data can have larger output variance, so an apparently high R2 can coexist with an operationally large RMSE. The 25g and 40g legacy input tuples are identical repository constants, not verified physical metadata; see `COLUMN_SPEC_PROVENANCE_AUDIT.md`.",
        "",
        "## Reproduction gap",
        "",
        "Differences from Figure 4 are descriptive only. Plausible non-identifiable sources include dataset version, legacy filtering, random split, unavailable exact freeze map, source checkpoint, preprocessing, output-head implementation, epoch/patience, and unverified column geometry. No item in this list was changed after observing a test metric.",
        "",
        f"Shared column-spec marker: `{SHARED_SPEC_FLAG}`.",
    ])
    text = "\n".join(lines) + "\n"
    (output_dir / "REPRODUCTION_REPORT.md").write_text(text, encoding="utf-8")
    (output_dir / "REPRODUCTION_GAP.md").write_text(
        "# Reproduction gap\n\n" + "\n".join(lines[-8:]) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=STUDY)
    parser.add_argument("--columns", nargs="+", choices=tuple(COLUMN_CONFIG), default=tuple(COLUMN_CONFIG))
    parser.add_argument("--protocols", nargs="+", choices=("legacy_filtered", "no_threshold"), default=("legacy_filtered",))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "splits").mkdir(parents=True, exist_ok=True)
    write_environment(output_dir)

    metric_rows, prediction_tables, split_tables = [], [], []
    frozen_data = {}
    for protocol in args.protocols:
        for column in args.columns:
            data = read_target_data(column, protocol)
            frozen_data[(column, protocol)] = data
            data.to_csv(output_dir / f"canonical_{column}_{protocol}.csv", index=False)
            for seed in args.seeds:
                split = make_row_split(data, column, protocol, seed)
                split.to_csv(output_dir / "splits" / f"{column}_{protocol}_row_seed_{seed}.csv", index=False)
                split_tables.append(split)
                for method in METHODS:
                    print(json.dumps({"column": column, "protocol": protocol, "seed": seed, "method": method}), flush=True)
                    result, predictions = train_run(
                        column,
                        protocol,
                        method,
                        seed,
                        data,
                        split,
                        output_dir,
                        2 if args.smoke else args.epochs,
                        2 if args.smoke else args.patience,
                        args.batch_size,
                    )
                    metric_rows.extend(flat_metric_rows(result))
                    prediction_tables.append(predictions)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    splits = pd.concat(split_tables, ignore_index=True)
    references = reference_rows()
    combined_metrics = pd.concat([references, metrics], ignore_index=True)
    summary = pd.concat([reference_summary(references), summarize_metrics(metrics)], ignore_index=True)
    comparison = paper_r2_comparison(summary)
    combined_metrics.to_csv(output_dir / "all_metrics.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv.gz", index=False, compression="gzip")
    splits.to_csv(output_dir / "split_manifest.csv", index=False)
    summary.to_csv(output_dir / "PAPER_TRANSFER_RMSE_SUMMARY.csv", index=False)
    comparison.to_csv(output_dir / "paper_r2_comparison.csv", index=False)
    write_scale_note(output_dir, summary)
    write_report(output_dir, summary, comparison)
    config = {
        "classification": "PAPER_ALIGNED_RECONSTRUCTED_REPRODUCTION",
        "primary_protocol": "legacy_filtered",
        "run_protocols": list(args.protocols),
        "columns": list(args.columns),
        "seeds": list(args.seeds),
        "epochs": 2 if args.smoke else args.epochs,
        "patience": 2 if args.smoke else args.patience,
        "batch_size": args.batch_size,
        "source_checkpoint": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
        "source_checkpoint_sha256": sha256_file(SOURCE_CHECKPOINT),
        "source_scaler": str(SOURCE_SCALER.relative_to(ROOT)),
        "column_spec_flag": SHARED_SPEC_FLAG,
        "selection": "validation_only_minimum_combined_normalized_rmse",
        "test_driven_reconstruction": False,
        "smoke": args.smoke,
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_artifact_manifest(output_dir)
    print(json.dumps({"completed": len(metric_rows) // 2, "output": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
