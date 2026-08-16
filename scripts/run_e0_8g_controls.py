#!/usr/bin/env python3
"""Run paired 4g-to-8g robustness and loss-scaling control studies."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_e0_4g_baseline import (
    eluent_descriptor,
    make_compound_split,
    make_split,
    minmax_fit,
    sha256_file,
    write_environment,
)
from scripts.run_e0_8g_transfer import (
    SOURCE_4G_DIR,
    build_model_data,
    load_or_fit_scaler,
    make_loaders,
    metrics_from_arrays,
    run_configuration,
    write_artifact_manifest,
)


DATA_DIR = ROOT / "experiments" / "e0_8g_transfer"
FROZEN_SPLIT_DIR = ROOT / "experiments" / "e0_3b_controls" / "splits"

CORE_SPECS = {
    "scratch_lr1e-3": {
        "mode": "full", "lr": 1e-3, "pretrained": False, "scaler": "target_train"
    },
    "head_only_lr1e-4": {
        "mode": "head_only", "lr": 1e-4, "pretrained": True, "scaler": "source_4g"
    },
    "last2_head_lr1e-4": {
        "mode": "last2_head", "lr": 1e-4, "pretrained": True, "scaler": "source_4g"
    },
    "full_lr1e-4": {
        "mode": "full", "lr": 1e-4, "pretrained": True, "scaler": "source_4g"
    },
}

FACTOR_SPECS = {
    "head_reinit_lr1e-4": {
        "mode": "head_only", "lr": 1e-4, "pretrained": True, "scaler": "source_4g",
        "reinitialize_head": True,
    },
    "last2_head_reinit_lr1e-4": {
        "mode": "last2_head", "lr": 1e-4, "pretrained": True, "scaler": "source_4g",
        "reinitialize_head": True,
    },
    "last2_head_equal_v2_lr1e-4": {
        "mode": "last2_head", "lr": 1e-4, "pretrained": True, "scaler": "source_4g",
        "v2_weight": 1.0,
    },
    "last2_head_standardized_loss_lr1e-4": {
        "mode": "last2_head", "lr": 1e-4, "pretrained": True, "scaler": "source_4g",
        "v2_weight": 1.0, "loss_scaling": "target_standard_deviation",
    },
}

LOSS_SPECS = {
    "last2_head_lr1e-4": CORE_SPECS["last2_head_lr1e-4"],
    "last2_head_equal_v2_lr1e-4": FACTOR_SPECS["last2_head_equal_v2_lr1e-4"],
    "last2_head_standardized_loss_lr1e-4": FACTOR_SPECS[
        "last2_head_standardized_loss_lr1e-4"
    ],
}

METRIC_COLUMNS = [
    "normalized_valid_score",
    "normalized_test_score",
    "valid_V1_r2",
    "valid_V2_r2",
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


def flatten_result(result: dict) -> dict:
    row = {key: value for key, value in result.items() if key not in {"valid", "test"}}
    row.update({f"valid_{key}": value for key, value in result["valid"].items()})
    row.update({f"test_{key}": value for key, value in result["test"].items()})
    return row


def load_graph_cache() -> dict:
    cache = dict(torch.load(SOURCE_4G_DIR / "graph_cache_4g.pt", weights_only=False))
    cache.update(torch.load(DATA_DIR / "graph_cache_8g_only.pt", weights_only=False))
    return cache


def sensitivity_metrics(predictions: pd.DataFrame, excluded_source_row: int) -> pd.DataFrame:
    rows = []
    filtered = predictions[predictions["source_row_1based"].ne(excluded_source_row)]
    for keys, group in filtered.groupby(["split_mode", "seed", "config"], sort=True):
        split_mode, seed, config = keys
        y_true = group[["V1_true", "V2_true"]].to_numpy(dtype=np.float32)
        pred = group[["V1_q10", "V1_q50", "V1_q90", "V2_q10", "V2_q50", "V2_q90"]].to_numpy(
            dtype=np.float32
        )
        metrics = metrics_from_arrays(y_true, pred)
        rows.append({
            "split_mode": split_mode,
            "seed": int(seed),
            "config": config,
            "excluded_source_row_1based": excluded_source_row,
            "remaining_test_rows": len(group),
            **metrics,
        })
    return pd.DataFrame(rows)


def study_specs(study: str, split_mode: str, seed: int, first_seed: int) -> dict:
    """Return the frozen configuration matrix for one split/seed pair."""
    if study == "loss_controls":
        return dict(LOSS_SPECS)
    specs = dict(CORE_SPECS)
    if split_mode == "row" and seed == first_seed:
        specs.update(FACTOR_SPECS)
    return specs


def metric_summary(comparison: pd.DataFrame, configs: set[str]) -> pd.DataFrame:
    selected = comparison[comparison["config"].isin(configs)]
    summary = selected.groupby(["split_mode", "config"], as_index=False)[METRIC_COLUMNS].agg(
        ["mean", "std"]
    )
    summary.columns = [
        "_".join(part for part in column if part) for column in summary.columns.to_flat_index()
    ]
    return summary


def paired_effects(comparison: pd.DataFrame, reference_config: str) -> pd.DataFrame:
    """Calculate within-split, within-seed differences from a fixed reference."""
    rows = []
    for (split_mode, seed), group in comparison.groupby(["split_mode", "seed"]):
        reference = group[group["config"].eq(reference_config)]
        if reference.empty:
            continue
        baseline = reference.iloc[0]
        for _, result in group.iterrows():
            rows.append({
                "split_mode": split_mode,
                "seed": int(seed),
                "reference_config": reference_config,
                "config": result["config"],
                "delta_normalized_valid_vs_reference": (
                    result["normalized_valid_score"] - baseline["normalized_valid_score"]
                ),
                "delta_normalized_test_vs_reference": (
                    result["normalized_test_score"] - baseline["normalized_test_score"]
                ),
                "delta_test_mae_v1_vs_reference": (
                    result["test_V1_mae"] - baseline["test_V1_mae"]
                ),
                "delta_test_mae_v2_vs_reference": (
                    result["test_V2_mae"] - baseline["test_V2_mae"]
                ),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen paired controls for 4g-to-8g QGeoGNN transfer."
    )
    parser.add_argument(
        "--study", choices=["robustness", "loss_controls"], default="robustness"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 525, 1101])
    parser.add_argument("--split-modes", nargs="+", choices=["row", "compound"], default=["row", "compound"])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    default_outputs = {
        "robustness": ROOT / "experiments" / "e0_3b_controls",
        "loss_controls": ROOT / "experiments" / "e0_3c_loss_controls",
    }
    output_dir = args.output_dir or default_outputs[args.study]
    if (output_dir / "comparison.csv").exists():
        raise FileExistsError(
            f"Refusing to overwrite finalized experiment: {output_dir}. "
            "Pass a new --output-dir for a reproducibility run."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / ".work"
    work_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = output_dir / "splits"
    scalers_dir = output_dir / "scalers"
    if args.study == "robustness":
        splits_dir.mkdir(exist_ok=True)
    write_environment(output_dir)

    canonical_df = pd.read_csv(DATA_DIR / "canonical_8g.csv")
    graph_cache = load_graph_cache()
    raw_descriptors = np.vstack(
        [graph_cache[smiles]["descriptor"] for smiles in canonical_df["canonical_smiles"]]
    ).astype(np.float32)
    raw_eluents = np.vstack(
        [eluent_descriptor(value) for value in canonical_df["PE/EA"]]
    ).astype(np.float32)
    source_checkpoint = SOURCE_4G_DIR / "checkpoints" / "best.pt"

    all_results, all_predictions, all_histories = [], [], []
    split_manifest = {}
    for split_mode in args.split_modes:
        for seed in args.seeds:
            run_key = f"{split_mode}_seed_{seed}"
            run_dir = work_dir / run_key
            run_dir.mkdir(parents=True, exist_ok=True)
            if args.study == "loss_controls":
                split_source = FROZEN_SPLIT_DIR / f"{run_key}.csv"
                if not split_source.is_file():
                    raise FileNotFoundError(f"Frozen E0-3b split not found: {split_source}")
                split_df = pd.read_csv(split_source)
            elif split_mode == "row":
                split_df = make_split(canonical_df, run_dir, seed)
                split_source = run_dir / f"split_seed_{seed}.csv"
            else:
                split_df = make_compound_split(canonical_df, run_dir, seed)
                split_source = run_dir / f"compound_group_split_seed_{seed}.csv"
            if args.study == "robustness":
                split_source = shutil.copy2(split_source, splits_dir / f"{run_key}.csv")
            split_manifest[run_key] = {
                "path": str(Path(split_source).relative_to(ROOT)),
                "sha256": sha256_file(Path(split_source)),
            }

            specs = study_specs(args.study, split_mode, seed, args.seeds[0])
            if args.smoke:
                specs = {"last2_head_lr1e-4": CORE_SPECS["last2_head_lr1e-4"]}

            train_mask = split_df["split"].eq("train").to_numpy()
            required_scalers = {spec["scaler"] for spec in specs.values()}
            scalers = {}
            if "source_4g" in required_scalers:
                scalers["source_4g"] = load_or_fit_scaler(
                    raw_descriptors, raw_eluents, train_mask, "source_4g"
                )
            if "target_train" in required_scalers:
                scalers["target_train"] = {
                    "fit_split": f"8g_{split_mode}_train_seed_{seed}",
                    "descriptor": minmax_fit(raw_descriptors[train_mask]),
                    "eluent": minmax_fit(raw_eluents[train_mask]),
                }
                scalers_dir.mkdir(exist_ok=True)
                (scalers_dir / f"target_{run_key}.json").write_text(
                    json.dumps(scalers["target_train"], indent=2), encoding="utf-8"
                )
            data_by_scaler = {}
            for scaler_name, scaler in scalers.items():
                atom_data, angle_data = build_model_data(canonical_df, graph_cache, split_df, scaler)
                data_by_scaler[scaler_name] = make_loaders(
                    atom_data, angle_data, split_df, args.batch_size
                )
            train_labels = canonical_df.loc[train_mask, ["V1_ml", "V2_ml"]]
            target_variance = {
                "V1": float(train_labels["V1_ml"].var(ddof=0)),
                "V2": float(train_labels["V2_ml"].var(ddof=0)),
            }

            for config_name, spec in specs.items():
                state_path = run_dir / f"{config_name}.result.json"
                prediction_path = run_dir / f"{config_name}.predictions.csv"
                if state_path.exists() and prediction_path.exists():
                    result = json.loads(state_path.read_text(encoding="utf-8"))
                    prediction_rows = pd.read_csv(prediction_path).to_dict(orient="records")
                else:
                    result, prediction_rows = run_configuration(
                        config_name=config_name,
                        mode=spec["mode"],
                        learning_rate=spec["lr"],
                        pretrained=spec["pretrained"],
                        loaders=data_by_scaler[spec["scaler"]],
                        canonical_df=canonical_df,
                        output_dir=run_dir,
                        source_checkpoint=source_checkpoint,
                        seed=seed,
                        epochs=2 if args.smoke else args.epochs,
                        patience=2 if args.smoke else args.patience,
                        target_variance=target_variance,
                        v2_weight=spec.get("v2_weight", 0.5),
                        loss_scaling=spec.get("loss_scaling", "legacy"),
                        reinitialize_head=spec.get("reinitialize_head", False),
                    )
                    state_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                    pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False)
                result_row = flatten_result(result)
                result_row.update({
                    "split_mode": split_mode,
                    "seed": seed,
                    "scaler": spec["scaler"],
                    "train_rows": int(train_mask.sum()),
                    "valid_rows": int(split_df["split"].eq("valid").sum()),
                    "test_rows": int(split_df["split"].eq("test").sum()),
                })
                all_results.append(result_row)
                for row in prediction_rows:
                    row.update({"split_mode": split_mode, "seed": seed})
                    all_predictions.append(row)
                history_path = run_dir / "histories" / f"{config_name}.csv"
                if history_path.exists():
                    history = pd.read_csv(history_path)
                    history.insert(0, "config", config_name)
                    history.insert(0, "seed", seed)
                    history.insert(0, "split_mode", split_mode)
                    all_histories.append(history)
                checkpoint_path = run_dir / "checkpoints" / f"{config_name}.pt"
                checkpoint_path.unlink(missing_ok=True)

    comparison = pd.DataFrame(all_results)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    predictions = pd.DataFrame(all_predictions)
    predictions.to_csv(output_dir / "test_predictions.csv.gz", index=False, compression="gzip")
    if all_histories:
        pd.concat(all_histories, ignore_index=True).to_csv(
            output_dir / "training_histories.csv.gz", index=False, compression="gzip"
        )

    comparison_configs = set(LOSS_SPECS if args.study == "loss_controls" else CORE_SPECS)
    selected = comparison[comparison["config"].isin(comparison_configs)]
    metric_summary(comparison, comparison_configs).to_csv(
        output_dir / "paired_summary.csv", index=False
    )
    reference_config = (
        "last2_head_lr1e-4" if args.study == "loss_controls" else "scratch_lr1e-3"
    )
    effect_filename = (
        "paired_effects_vs_reference.csv"
        if args.study == "loss_controls"
        else "paired_effects_vs_scratch.csv"
    )
    paired_effects(selected, reference_config).to_csv(output_dir / effect_filename, index=False)
    sensitivity_metrics(predictions, 224).to_csv(
        output_dir / "sensitivity_excluding_source_row_224.csv", index=False
    )

    ranking_pool = selected[selected["config"].ne("scratch_lr1e-3")]
    ranking = ranking_pool.groupby("config")["normalized_valid_score"].agg(
        ["mean", "std", "count"]
    )
    ranking = ranking.sort_values("mean")
    provisional = ranking.reset_index().iloc[0].to_dict() if not ranking.empty else {}
    (output_dir / "provisional_candidate.json").write_text(
        json.dumps(provisional, indent=2), encoding="utf-8"
    )
    config = {
        "stage": (
            "E0-3c_paired_loss_controls"
            if args.study == "loss_controls"
            else "E0-3b_paired_robustness_and_factor_controls"
        ),
        "study": args.study,
        "data_dir": str(DATA_DIR.relative_to(ROOT)),
        "data_sha256": sha256_file(DATA_DIR / "canonical_8g.csv"),
        "source_checkpoint": str(source_checkpoint.relative_to(ROOT)),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "seeds": args.seeds,
        "split_modes": args.split_modes,
        "frozen_splits": split_manifest,
        "epochs": 2 if args.smoke else args.epochs,
        "patience": 2 if args.smoke else args.patience,
        "batch_size": args.batch_size,
        "study_specs": LOSS_SPECS if args.study == "loss_controls" else CORE_SPECS,
        "factor_specs_first_row_seed_only": (
            FACTOR_SPECS if args.study == "robustness" else None
        ),
        "reference_config": reference_config,
        "selection_rule": (
            "lowest mean normalized validation score across paired runs; test not used"
        ),
        "official_metrics_include_source_row_224": True,
        "sensitivity_exclusion_is_not_a_data_deletion": True,
        "smoke": args.smoke,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if not args.keep_work:
        shutil.rmtree(work_dir)
    write_artifact_manifest(output_dir)
    print(json.dumps({"runs": len(comparison), "provisional": provisional}, ensure_ascii=False))


if __name__ == "__main__":
    main()
