#!/usr/bin/env python3
"""Run D04: first-embedded versus lowest-energy conformer control."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qgeognn_graphs import build_graph_and_descriptor
from scripts.run_e0_4g_baseline import (
    eluent_descriptor,
    sha256_file,
    train_baseline,
    write_artifact_manifest,
    write_environment,
)
from scripts.run_e0_8g_controls import flatten_result
from scripts.run_e0_8g_transfer import build_model_data, make_loaders, run_configuration


LEGACY_4G = ROOT / "experiments" / "e0_4g_baseline"
LEGACY_8G = ROOT / "experiments" / "e0_8g_transfer"
LOSS_CONTROL = ROOT / "experiments" / "e0_3c_loss_controls"
FROZEN_SPLITS = ROOT / "experiments" / "e0_3b_controls" / "splits"
DEFAULT_OUTPUT = ROOT / "experiments" / "d04_conformer_selection"
SEEDS = [42, 525, 1101]
SPLIT_MODES = ["row", "compound"]
TRANSFER_CONFIG = "last2_head_equal_v2_lr1e-4"

TRANSFER_METRICS = [
    "normalized_valid_score",
    "normalized_test_score",
    "test_V1_r2",
    "test_V2_r2",
    "test_V1_mae",
    "test_V2_mae",
    "test_V1_rmse",
    "test_V2_rmse",
    "test_V1_interval_80_coverage",
    "test_V2_interval_80_coverage",
    "test_quantile_crossing_rate",
]


def geometry_deltas(new_graph: dict, old_graph: dict) -> dict[str, float | bool]:
    """Compare invariant geometry features, not arbitrarily oriented coordinates."""
    values = {}
    for key in ["bond_length", "bond_angle"]:
        new = np.asarray(new_graph[key], dtype=np.float64)
        old = np.asarray(old_graph[key], dtype=np.float64)
        if new.shape != old.shape:
            raise ValueError(f"geometry_shape_changed:{key}:{old.shape}->{new.shape}")
        values[f"mean_abs_{key}_delta"] = float(np.mean(np.abs(new - old)))
        values[f"max_abs_{key}_delta"] = float(np.max(np.abs(new - old)))
    values["geometry_changed"] = bool(
        values["max_abs_bond_length_delta"] > 1e-6
        or values["max_abs_bond_angle_delta"] > 1e-6
    )
    return values


def build_lowest_energy_caches(output_dir: Path, seed: int) -> tuple[dict, dict]:
    """Build 4g and target-only 8g caches with seeds paired to legacy audits."""
    cache_4g_path = output_dir / "graph_cache_4g_lowest_energy.pt"
    audit_4g_path = output_dir / "conformer_audit_4g.csv"
    cache_4g = (
        torch.load(cache_4g_path, weights_only=False) if cache_4g_path.exists() else {}
    )
    previous_4g = {}
    if audit_4g_path.exists():
        previous_4g = {
            row["canonical_smiles"]: row
            for row in pd.read_csv(audit_4g_path).to_dict(orient="records")
            if row.get("status") == "success"
        }
    legacy_cache_4g = torch.load(LEGACY_4G / "graph_cache_4g.pt", weights_only=False)
    legacy_audit_4g = pd.read_csv(LEGACY_4G / "graph_audit_4g.csv")
    audit_4g = []
    for index, row in legacy_audit_4g.reset_index(drop=True).iterrows():
        canonical = row["canonical_smiles"]
        if canonical in cache_4g and canonical in previous_4g:
            audit_4g.append(previous_4g[canonical])
            continue
        started = time.time()
        try:
            graph, descriptor, metadata = build_graph_and_descriptor(
                row["source_smiles"], seed + index, "lowest_energy"
            )
            old = legacy_cache_4g[canonical]
            cache_4g[canonical] = {"graph": graph, "descriptor": descriptor}
            record = {
                "canonical_smiles": canonical,
                "source_smiles": row["source_smiles"],
                "status": "success",
                **metadata,
                **geometry_deltas(graph, old["graph"]),
                "descriptor_changed": bool(not np.array_equal(descriptor, old["descriptor"])),
                "error": "",
            }
        except Exception as exc:
            record = {
                "canonical_smiles": canonical,
                "source_smiles": row["source_smiles"],
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        record["seconds"] = round(time.time() - started, 4)
        audit_4g.append(record)
        pd.DataFrame(audit_4g).to_csv(audit_4g_path, index=False)
        torch.save(cache_4g, cache_4g_path)
    if any(row["status"] != "success" for row in audit_4g):
        raise RuntimeError("4g lowest-energy cache contains graph failures")

    cache_8g_path = output_dir / "graph_cache_8g_only_lowest_energy.pt"
    audit_8g_path = output_dir / "conformer_audit_8g.csv"
    cache_8g_only = (
        torch.load(cache_8g_path, weights_only=False) if cache_8g_path.exists() else {}
    )
    legacy_cache_8g = dict(legacy_cache_4g)
    legacy_cache_8g.update(
        torch.load(LEGACY_8G / "graph_cache_8g_only.pt", weights_only=False)
    )
    legacy_audit_8g = pd.read_csv(LEGACY_8G / "graph_audit_8g.csv")
    audit_4g_by_smiles = {row["canonical_smiles"]: row for row in audit_4g}
    audit_8g = []
    for index, row in legacy_audit_8g.reset_index(drop=True).iterrows():
        canonical = row["canonical_smiles"]
        if canonical in cache_4g:
            source_record = audit_4g_by_smiles[canonical]
            audit_8g.append({
                "canonical_smiles": canonical,
                "source_smiles": row["source_smiles"],
                "status": "success",
                "graph_origin": "reused_4g_lowest_energy",
                "conformer_policy": "lowest_energy",
                "force_field": source_record.get("force_field"),
                "selected_conformer_id": source_record.get("selected_conformer_id"),
                "minimum_energy": source_record.get("minimum_energy"),
                "geometry_changed": source_record.get("geometry_changed"),
                "error": "",
                "seconds": 0.0,
            })
            continue
        started = time.time()
        try:
            graph, descriptor, metadata = build_graph_and_descriptor(
                row["source_smiles"], seed + index, "lowest_energy"
            )
            old = legacy_cache_8g[canonical]
            cache_8g_only[canonical] = {"graph": graph, "descriptor": descriptor}
            record = {
                "canonical_smiles": canonical,
                "source_smiles": row["source_smiles"],
                "status": "success",
                "graph_origin": "generated_8g_lowest_energy",
                **metadata,
                **geometry_deltas(graph, old["graph"]),
                "descriptor_changed": bool(not np.array_equal(descriptor, old["descriptor"])),
                "error": "",
            }
        except Exception as exc:
            record = {
                "canonical_smiles": canonical,
                "source_smiles": row["source_smiles"],
                "status": "failed",
                "graph_origin": "generated_8g_lowest_energy",
                "error": f"{type(exc).__name__}: {exc}",
            }
        record["seconds"] = round(time.time() - started, 4)
        audit_8g.append(record)
        torch.save(cache_8g_only, cache_8g_path)
    pd.DataFrame(audit_8g).to_csv(audit_8g_path, index=False)
    if any(row["status"] != "success" for row in audit_8g):
        raise RuntimeError("8g lowest-energy cache contains graph failures")
    torch.save(cache_8g_only, cache_8g_path)
    return cache_4g, cache_8g_only


def train_4g_source(
    output_dir: Path,
    work_dir: Path,
    graph_cache: dict,
    seed: int,
    smoke: bool,
) -> dict:
    source_work = work_dir / "source_4g"
    source_work.mkdir(parents=True, exist_ok=True)
    canonical = pd.read_csv(LEGACY_4G / "canonical_4g.csv")
    split = pd.read_csv(LEGACY_4G / "split_seed_42.csv")
    metrics = train_baseline(
        canonical,
        graph_cache,
        split,
        source_work,
        seed=seed,
        epochs=2 if smoke else 1000,
        batch_size=2048,
        learning_rate=1e-3,
        patience=2 if smoke else 100,
        verbose=False,
    )
    final_files = {
        source_work / "training_history.csv": output_dir / "history_4g_lowest_energy.csv",
        source_work / "metrics.json": output_dir / "metrics_4g_lowest_energy.json",
    }
    for source, destination in final_files.items():
        shutil.copy2(source, destination)
    return metrics


def run_8g_transfer(
    output_dir: Path,
    work_dir: Path,
    cache_4g: dict,
    cache_8g_only: dict,
    source_checkpoint: Path,
    seed_values: list[int],
    split_modes: list[str],
    smoke: bool,
) -> pd.DataFrame:
    canonical = pd.read_csv(LEGACY_8G / "canonical_8g.csv")
    graph_cache = dict(cache_4g)
    graph_cache.update(cache_8g_only)
    scaler = json.loads((LEGACY_4G / "scaler.json").read_text(encoding="utf-8"))
    results, predictions, histories = [], [], []
    for split_mode in split_modes:
        for seed in seed_values:
            run_key = f"{split_mode}_seed_{seed}"
            split = pd.read_csv(FROZEN_SPLITS / f"{run_key}.csv")
            atom_data, angle_data = build_model_data(canonical, graph_cache, split, scaler)
            loaders = make_loaders(atom_data, angle_data, split, batch_size=2048)
            train_mask = split["split"].eq("train").to_numpy()
            labels = canonical.loc[train_mask, ["V1_ml", "V2_ml"]]
            target_variance = {
                "V1": float(labels["V1_ml"].var(ddof=0)),
                "V2": float(labels["V2_ml"].var(ddof=0)),
            }
            run_dir = work_dir / "transfer_8g" / run_key
            result, prediction_rows = run_configuration(
                config_name=TRANSFER_CONFIG,
                mode="last2_head",
                learning_rate=1e-4,
                pretrained=True,
                loaders=loaders,
                canonical_df=canonical,
                output_dir=run_dir,
                source_checkpoint=source_checkpoint,
                seed=seed,
                epochs=2 if smoke else 500,
                patience=2 if smoke else 100,
                target_variance=target_variance,
                v2_weight=1.0,
                loss_scaling="legacy",
            )
            flat = flatten_result(result)
            flat.update({
                "split_mode": split_mode,
                "seed": seed,
                "conformer_policy": "lowest_energy",
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(split["split"].eq("valid").sum()),
                "test_rows": int(split["split"].eq("test").sum()),
            })
            results.append(flat)
            for row in prediction_rows:
                row.update({
                    "split_mode": split_mode,
                    "seed": seed,
                    "conformer_policy": "lowest_energy",
                })
                predictions.append(row)
            history_path = run_dir / "histories" / f"{TRANSFER_CONFIG}.csv"
            history = pd.read_csv(history_path)
            history.insert(0, "conformer_policy", "lowest_energy")
            history.insert(0, "seed", seed)
            history.insert(0, "split_mode", split_mode)
            histories.append(history)
            (run_dir / "checkpoints" / f"{TRANSFER_CONFIG}.pt").unlink(missing_ok=True)
    pd.DataFrame(predictions).to_csv(
        output_dir / "predictions_8g_lowest_energy.csv.gz", index=False, compression="gzip"
    )
    pd.concat(histories, ignore_index=True).to_csv(
        output_dir / "histories_8g_lowest_energy.csv.gz", index=False, compression="gzip"
    )
    return pd.DataFrame(results)


def flatten_metrics(policy: str, metrics: dict) -> dict:
    row = {
        "conformer_policy": policy,
        "best_epoch": metrics["best_epoch"],
        "best_valid_score": metrics["best_valid_score"],
        "epochs_run": metrics["epochs_run"],
    }
    for split in ["valid", "test"]:
        row.update({f"{split}_{key}": value for key, value in metrics[split].items()})
    return row


def write_comparisons(output_dir: Path, new_4g_metrics: dict, new_8g: pd.DataFrame) -> None:
    legacy_4g_metrics = json.loads((LEGACY_4G / "metrics.json").read_text(encoding="utf-8"))
    pd.DataFrame([
        flatten_metrics("first_embedded", legacy_4g_metrics),
        flatten_metrics("lowest_energy", new_4g_metrics),
    ]).to_csv(output_dir / "comparison_4g.csv", index=False)

    reference = pd.read_csv(LOSS_CONTROL / "comparison.csv")
    reference = reference[reference["config"].eq(TRANSFER_CONFIG)].copy()
    contexts = set(zip(new_8g["split_mode"], new_8g["seed"]))
    reference = reference[
        reference.apply(lambda row: (row["split_mode"], row["seed"]) in contexts, axis=1)
    ].copy()
    reference["conformer_policy"] = "first_embedded"
    combined = pd.concat([reference, new_8g], ignore_index=True, sort=False)
    combined.to_csv(output_dir / "comparison_8g.csv", index=False)

    paired_rows = []
    old_index = reference.set_index(["split_mode", "seed"])
    new_index = new_8g.set_index(["split_mode", "seed"])
    for context in sorted(contexts):
        old, new = old_index.loc[context], new_index.loc[context]
        row = {"split_mode": context[0], "seed": int(context[1])}
        for metric in TRANSFER_METRICS:
            row[f"delta_{metric}_lowest_minus_first"] = float(new[metric] - old[metric])
        paired_rows.append(row)
    pd.DataFrame(paired_rows).to_csv(output_dir / "paired_effects_8g.csv", index=False)

    summary = combined.groupby(["conformer_policy", "split_mode"])[TRANSFER_METRICS].agg(
        ["mean", "std"]
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(output_dir / "summary_8g.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the paired QGeoGNN lowest-energy conformer control."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()
    if (args.output_dir / "comparison_8g.csv").exists():
        raise FileExistsError(
            f"Refusing to overwrite finalized experiment: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_environment(args.output_dir)
    work_dir = args.output_dir / ".work"
    work_dir.mkdir(exist_ok=True)

    cache_4g, cache_8g = build_lowest_energy_caches(args.output_dir, args.seed)
    metrics_4g = train_4g_source(args.output_dir, work_dir, cache_4g, args.seed, args.smoke)
    source_checkpoint = work_dir / "source_4g" / "checkpoints" / "best.pt"
    seeds = [args.seed] if args.smoke else SEEDS
    split_modes = ["row"] if args.smoke else SPLIT_MODES
    results_8g = run_8g_transfer(
        args.output_dir,
        work_dir,
        cache_4g,
        cache_8g,
        source_checkpoint,
        seeds,
        split_modes,
        args.smoke,
    )
    write_comparisons(args.output_dir, metrics_4g, results_8g)
    config = {
        "stage": "D04_conformer_selection",
        "execution_script": "scripts/run_d04_conformer_selection.py",
        "data": {
            "canonical_4g": "experiments/e0_4g_baseline/canonical_4g.csv",
            "canonical_8g": "experiments/e0_8g_transfer/canonical_8g.csv",
            "thresholds": "legacy V1<=60 mL and V2<=120 mL",
            "repeated_experiments": "retain",
        },
        "conformer_protocol": {
            "generator": "RDKit ETKDGv3",
            "embedded_conformers": 10,
            "selection": "minimum optimized force-field energy",
            "force_field": "MMFF94 with recorded UFF fallback",
            "seed": args.seed,
        },
        "source_4g_training": {
            "loss": "legacy V1 + 0.5*V2",
            "learning_rate": 1e-3,
            "epochs": 2 if args.smoke else 1000,
            "patience": 2 if args.smoke else 100,
            "batch_size": 2048,
            "split": "experiments/e0_4g_baseline/split_seed_42.csv",
            "scaler": "experiments/e0_4g_baseline/scaler.json",
            "checkpoint_retention": "temporary validation-best; deleted after transfer",
        },
        "transfer_8g": {
            "configuration": TRANSFER_CONFIG,
            "loss": "V1 + V2",
            "learning_rate": 1e-4,
            "epochs": 2 if args.smoke else 500,
            "patience": 2 if args.smoke else 100,
            "seeds": seeds,
            "split_modes": split_modes,
            "frozen_split_directory": "experiments/e0_3b_controls/splits",
            "selection": "normalized validation score; test not used",
        },
        "reference_artifacts": {
            "legacy_4g_checkpoint_sha256": sha256_file(LEGACY_4G / "checkpoints" / "best.pt"),
            "canonical_4g_sha256": sha256_file(LEGACY_4G / "canonical_4g.csv"),
            "canonical_8g_sha256": sha256_file(LEGACY_8G / "canonical_8g.csv"),
        },
        "smoke": args.smoke,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.keep_work:
        shutil.rmtree(work_dir)
    write_artifact_manifest(args.output_dir)
    print(json.dumps({
        "4g_best_epoch": metrics_4g["best_epoch"],
        "8g_runs": len(results_8g),
        "output_dir": str(args.output_dir),
    }))


if __name__ == "__main__":
    main()
