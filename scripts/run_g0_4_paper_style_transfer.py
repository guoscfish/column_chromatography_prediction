#!/usr/bin/env python3
"""Run Gate 0-4 paired transfer-structure qualification.

The experiment reuses the frozen G0-3 no-threshold data and six paired splits.
It compares the provisional last-two-layers-plus-head transfer, full fine-tuning,
and a restrained paper-aligned transfer that adds the repository's dormant
column diameter/length/packing-density edge inputs, transfers compatible 4g
weights, installs the frozen monotonic output head, and trains the new input
adapters together with the last two GNN layers and head.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_e0_4g_baseline import sha256_file, write_artifact_manifest, write_environment
from scripts.run_e0_8g_controls import load_graph_cache
from scripts.run_e0_8g_transfer import (
    SOURCE_4G_DIR,
    build_model,
    build_model_data,
    configure_trainable,
    make_loaders,
    quantile_target_loss,
    set_training_mode,
    validation_scores,
)
from scripts.run_g0_1_quantile_monotonicity import install_monotonic_head, predict
from scripts.run_g0_3_threshold_sensitivity import (
    add_sample_scores,
    fit_and_apply_calibration,
    slice_metrics,
)
from scripts.qgeognn_graphs import qg


CONFIGS = ("last2_head", "full_finetune", "paper_style")
TARGETS = ("V1", "V2")
SLICES = ("full", "common", "tail")
BASE_BOND_FLOAT_NAMES = ("bond_length", "prop", "e", "m", "V_e")
COLUMN_NAMES = ("column_dia", "column_len", "column_den")
# Values already encoded by the repository's original 8g and single-sample paths.
COLUMN_SPEC_4G = (1.5, 6.6, 0.4458)
COLUMN_SPEC_8G = (1.5, 13.2, 0.4458)
G03_DIR = ROOT / "experiments" / "g0_3_threshold_sensitivity"
PAPER_URL = "https://arxiv.org/pdf/2404.09114"


def set_feature_schema(paper_style: bool) -> None:
    qg.bond_float_names = list(BASE_BOND_FLOAT_NAMES + (COLUMN_NAMES if paper_style else ()))


def add_column_spec(atom_data: list, values: tuple[float, float, float]) -> list:
    result = []
    for item in atom_data:
        copied = item.clone()
        column = torch.tensor(values, dtype=torch.float32).repeat(copied.edge_attr.shape[0], 1)
        copied.edge_attr = torch.cat([copied.edge_attr, column], dim=1)
        result.append(copied)
    return result


def load_transferred_model(config_name: str, device: torch.device, source_checkpoint: Path):
    paper_style = config_name == "paper_style"
    set_feature_schema(paper_style)
    model = build_model(device)
    source = torch.load(source_checkpoint, map_location=device, weights_only=False)
    incompatible = model.load_state_dict(source["model_state_dict"], strict=not paper_style)
    missing = list(incompatible.missing_keys) if paper_style else []
    unexpected = list(incompatible.unexpected_keys) if paper_style else []
    if paper_style:
        expected_fragments = tuple(
            fragment
            for index in range(5, 8)
            for fragment in (f"linear_list.{index}.", f"rbf_list.{index}.")
        )
        invalid = [key for key in missing if not any(fragment in key for fragment in expected_fragments)]
        if unexpected or invalid or len(missing) != 54:
            raise ValueError(
                f"Unexpected paper-style transfer mismatch: missing={missing}, unexpected={unexpected}"
            )
        # A randomly added edge embedding changes every message-passing layer
        # before target training and therefore destroys the transferred source
        # function.  Zero-init only the new linear adapters so the model starts
        # exactly from the compatible 4g function; their weights still receive
        # gradients on the first target update.  RBF gamma remains at its
        # repository default and becomes trainable once adapter weights move.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                is_new_linear = "linear_list." in name and any(
                    f"linear_list.{index}." in name for index in range(5, 8)
                )
                if is_new_linear:
                    parameter.zero_()
    install_monotonic_head(model)
    return model, missing, unexpected


def configure_g04_trainable(model: nn.Module, config_name: str) -> tuple[int, int, list[str]]:
    if config_name == "full_finetune":
        trainable, total = configure_trainable(model, "full")
    else:
        trainable, total = configure_trainable(model, "last2_head")
        if config_name == "paper_style":
            for name, parameter in model.named_parameters():
                is_column_adapter = (
                    "bond_float_encoder.linear_list." in name
                    or "bond_float_encoder.rbf_list." in name
                    or "convs_bond_float." in name
                ) and any(
                    fragment in name
                    for index in range(5, 8)
                    for fragment in (f"linear_list.{index}.", f"rbf_list.{index}.")
                )
                if is_column_adapter:
                    parameter.requires_grad = True
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        elif config_name != "last2_head":
            raise ValueError(config_name)
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    return trainable, total, names


def prediction_frame(
    model: nn.Module,
    loaders: dict,
    data: pd.DataFrame,
    config_name: str,
    split_mode: str,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    set_feature_schema(config_name == "paper_style")
    for evaluation_split in ("valid", "test"):
        _, y_true, pred, indices = predict(model, loaders[evaluation_split], device)
        for row_index, true, quantiles in zip(indices, y_true, pred):
            source = data.iloc[int(row_index)]
            rows.append(
                {
                    "config": config_name,
                    "split_mode": split_mode,
                    "seed": seed,
                    "evaluation_split": evaluation_split,
                    "sample_id": source["sample_id"],
                    "source_row_1based": int(source["source_row_1based"]),
                    "canonical_smiles": source["canonical_smiles"],
                    "threshold_excluded": bool(source["threshold_excluded"]),
                    "V1_true": float(true[0]),
                    "V2_true": float(true[1]),
                    "V1_q10": float(quantiles[0]),
                    "V1_q50": float(quantiles[1]),
                    "V1_q90": float(quantiles[2]),
                    "V2_q10": float(quantiles[3]),
                    "V2_q50": float(quantiles[4]),
                    "V2_q90": float(quantiles[5]),
                }
            )
    return pd.DataFrame(rows)


def train_configuration(
    config_name: str,
    loaders: dict,
    run_dir: Path,
    source_checkpoint: Path,
    seed: int,
    epochs: int,
    patience: int,
    target_variance: dict[str, float],
) -> tuple[dict, nn.Module, torch.device]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, missing, unexpected = load_transferred_model(config_name, device, source_checkpoint)
    trainable, total, trainable_names = configure_g04_trainable(model, config_name)
    checkpoint_dir = run_dir / "checkpoints"
    history_dir = run_dir / "histories"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{config_name}.pt"
    result_path = run_dir / f"{config_name}.result.json"

    if checkpoint_path.exists() and result_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        return json.loads(result_path.read_text(encoding="utf-8")), model, device

    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
        weight_decay=1e-5,
    )
    history = []
    best_score, best_epoch, stale = float("inf"), 0, 0
    set_feature_schema(config_name == "paper_style")
    for epoch in range(1, epochs + 1):
        set_training_mode(model)
        total_loss, batches = 0.0, 0
        for atom_batch, angle_batch in zip(*loaders["train"]):
            atom_batch, angle_batch = atom_batch.to(device), angle_batch.to(device)
            pred, _ = model(atom_batch, angle_batch)
            loss = quantile_target_loss(atom_batch.y[:, 0], pred[:, 0:3])
            loss = loss + quantile_target_loss(atom_batch.y[:, 1], pred[:, 3:6])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        valid_metrics, _, _, _ = predict(model, loaders["valid"], device)
        score, legacy_score = validation_scores(valid_metrics, target_variance)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(batches, 1),
                "normalized_valid_score": score,
                "legacy_valid_score": legacy_score,
                **valid_metrics,
            }
        )
        if score < best_score:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "normalized_valid_score": score,
                    "config": config_name,
                    "seed": seed,
                    "learning_rate": 1e-4,
                    "trainable_parameters": trainable,
                    "column_spec_8g": COLUMN_SPEC_8G if config_name == "paper_style" else None,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if stale >= patience:
            break

    pd.DataFrame(history).to_csv(history_dir / f"{config_name}.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    valid_metrics, _, _, _ = predict(model, loaders["valid"], device)
    test_metrics, _, _, _ = predict(model, loaders["test"], device)
    result = {
        "config": config_name,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "normalized_valid_score": validation_scores(valid_metrics, target_variance)[0],
        "normalized_test_score": validation_scores(test_metrics, target_variance)[0],
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_parameter_names": trainable_names,
        "source_missing_keys": missing,
        "source_unexpected_keys": unexpected,
        "valid": valid_metrics,
        "test": test_metrics,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result, model, device


def import_g03_last2(
    run_key: str,
    loaders: dict,
    target_variance: dict[str, float],
    output_run_dir: Path,
) -> tuple[dict, nn.Module, torch.device]:
    source_dir = G03_DIR / "runs" / run_key / "no_threshold"
    source_checkpoint = source_dir / "checkpoints" / "monotonic_softplus.pt"
    source_result = source_dir / "monotonic_softplus.result.json"
    if not source_checkpoint.exists() or not source_result.exists():
        raise FileNotFoundError(f"Missing frozen G0-3 last2 run: {source_dir}")
    checkpoint_dir = output_run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target_checkpoint = checkpoint_dir / "last2_head.pt"
    if not target_checkpoint.exists():
        shutil.copy2(source_checkpoint, target_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_transferred_model("last2_head", device, SOURCE_4G_DIR / "checkpoints" / "best.pt")
    checkpoint = torch.load(target_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    valid_metrics, _, _, _ = predict(model, loaders["valid"], device)
    test_metrics, _, _, _ = predict(model, loaders["test"], device)
    original = json.loads(source_result.read_text(encoding="utf-8"))
    trainable, total, names = configure_g04_trainable(model, "last2_head")
    result = {
        "config": "last2_head",
        "best_epoch": int(original["best_epoch"]),
        "epochs_run": int(original["epochs_run"]),
        "normalized_valid_score": validation_scores(valid_metrics, target_variance)[0],
        "normalized_test_score": validation_scores(test_metrics, target_variance)[0],
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_parameter_names": names,
        "source": str(source_checkpoint.relative_to(ROOT)),
        "source_sha256": sha256_file(source_checkpoint),
        "valid": valid_metrics,
        "test": test_metrics,
    }
    (output_run_dir / "last2_head.result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result, model, device


def make_decision(training: pd.DataFrame) -> dict:
    valid = training.pivot(index=["split_mode", "seed"], columns="config", values="normalized_valid_score")
    relative = valid["paper_style"] / valid["last2_head"] - 1.0
    split_relative = (
        training.groupby(["split_mode", "config"])["normalized_valid_score"].mean().unstack()
    )
    split_relative = split_relative["paper_style"] / split_relative["last2_head"] - 1.0
    wins = int((relative < 0).sum())
    mean_change = float(valid["paper_style"].mean() / valid["last2_head"].mean() - 1.0)
    stable = bool(mean_change <= -0.02 and wins >= 4 and float(split_relative.max()) <= 0.05)
    return {
        "selected_transfer": "paper_style" if stable else "last2_head",
        "selection_uses_test": False,
        "paper_style_mean_validation_relative_change": mean_change,
        "paper_style_validation_wins": wins,
        "paired_contexts": len(relative),
        "split_mode_relative_changes": {key: float(value) for key, value in split_relative.items()},
        "rule": "select paper_style only if mean validation score improves >=2%, it wins >=4/6 paired contexts, and neither split-mode mean worsens >5%; otherwise retain last2+head",
        "full_finetune_role": "diagnostic control; not used to relax the pre-registered paper-style-versus-last2 rule",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "g0_4_paper_style_transfer")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 525, 1101])
    parser.add_argument("--split-modes", nargs="+", choices=["row", "compound"], default=["row", "compound"])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-reuse-g03-last2", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if (output_dir / "transfer_decision.json").exists():
        raise FileExistsError(f"Refusing to overwrite finalized experiment: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_environment(output_dir)

    data_path = G03_DIR / "canonical_8g_no_threshold.csv"
    data = pd.read_csv(data_path)
    if len(data) != 574:
        raise ValueError("G0-4 requires the frozen 574-row no-threshold target data")
    graph_cache = load_graph_cache()
    scaler = json.loads((SOURCE_4G_DIR / "scaler.json").read_text(encoding="utf-8"))
    source_checkpoint = SOURCE_4G_DIR / "checkpoints" / "best.pt"
    training_rows, prediction_tables, factor_rows, metric_rows = [], [], [], []

    for split_mode in args.split_modes:
        for seed in args.seeds:
            run_key = f"{split_mode}_seed_{seed}"
            split_path = G03_DIR / "splits" / f"{run_key}.csv"
            split = pd.read_csv(split_path)
            if not split["canonical_index"].equals(pd.Series(np.arange(len(data)))):
                raise ValueError(f"Index mismatch in {split_path}")
            base_atom, angle_data = build_model_data(data, graph_cache, split, scaler)
            paper_atom = add_column_spec(base_atom, COLUMN_SPEC_8G)
            base_loaders = make_loaders(base_atom, angle_data, split, args.batch_size)
            paper_loaders = make_loaders(paper_atom, angle_data, split, args.batch_size)
            train_mask = split["split"].eq("train").to_numpy()
            labels = data.loc[train_mask, ["V1_ml", "V2_ml"]]
            target_variance = {
                target: float(labels[f"{target}_ml"].var(ddof=0)) for target in TARGETS
            }
            reference_scales = {
                target: float(labels[f"{target}_ml"].std(ddof=0)) for target in TARGETS
            }

            for config_name in CONFIGS:
                print(json.dumps({"starting": run_key, "config": config_name}), flush=True)
                run_dir = output_dir / "runs" / run_key / config_name
                loaders = paper_loaders if config_name == "paper_style" else base_loaders
                if config_name == "last2_head" and not args.no_reuse_g03_last2 and not args.smoke:
                    result, model, device = import_g03_last2(run_key, loaders, target_variance, run_dir)
                else:
                    result, model, device = train_configuration(
                        config_name,
                        loaders,
                        run_dir,
                        source_checkpoint,
                        seed,
                        2 if args.smoke else args.epochs,
                        2 if args.smoke else args.patience,
                        target_variance,
                    )
                training_rows.append(
                    {
                        "split_mode": split_mode,
                        "seed": seed,
                        **{key: value for key, value in result.items() if key not in {"valid", "test", "trainable_parameter_names", "source_missing_keys", "source_unexpected_keys"}},
                    }
                )
                predictions = prediction_frame(
                    model, loaders, data, config_name, split_mode, seed, device
                )
                predictions, factors = fit_and_apply_calibration(predictions, config_name)
                predictions = add_sample_scores(predictions, reference_scales)
                prediction_tables.append(predictions)
                for factor in factors:
                    factor_rows.append(
                        {"split_mode": split_mode, "seed": seed, "config": config_name, **factor}
                    )
                calibration = predictions[predictions["evaluation_split"].eq("valid")]
                for evaluation_split in ("valid", "test"):
                    context = predictions[predictions["evaluation_split"].eq(evaluation_split)]
                    for slice_name in SLICES:
                        metric_rows.append(
                            {
                                "split_mode": split_mode,
                                "seed": seed,
                                "config": config_name,
                                "evaluation_split": evaluation_split,
                                "slice": slice_name,
                                **slice_metrics(context, calibration, reference_scales, slice_name),
                            }
                        )
                print(
                    json.dumps(
                        {"finished": run_key, "config": config_name, "best_epoch": result["best_epoch"]}
                    ),
                    flush=True,
                )

    training = pd.DataFrame(training_rows)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    factors = pd.DataFrame(factor_rows)
    metrics = pd.DataFrame(metric_rows)
    decision = make_decision(training)
    training.to_csv(output_dir / "training_comparison.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv.gz", index=False, compression="gzip")
    factors.to_csv(output_dir / "calibration_factors.csv", index=False)
    metrics.to_csv(output_dir / "slice_metrics.csv", index=False)
    summary = metrics.groupby(["evaluation_split", "slice", "config"], as_index=False).agg(
        rows_mean=("rows", "mean"),
        normalized_rmse_mean=("mean_normalized_rmse", "mean"),
        normalized_mae_mean=("mean_normalized_absolute_error", "mean"),
        V1_coverage_mean=("V1_calibrated_coverage", "mean"),
        V2_coverage_mean=("V2_calibrated_coverage", "mean"),
        auce_mean=("mean_auce", "mean"),
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "transfer_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    config = {
        "stage": "G0-4_paper_style_transfer",
        "smoke": args.smoke,
        "data": str(data_path.relative_to(ROOT)),
        "data_sha256": sha256_file(data_path),
        "rows": len(data),
        "source_checkpoint": str(source_checkpoint.relative_to(ROOT)),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "splits": [f"{mode}_seed_{seed}" for mode in args.split_modes for seed in args.seeds],
        "configs": list(CONFIGS),
        "shared": "monotonic softplus head, no-threshold target, source scaler, lr=1e-4, equal V1/V2 loss, validation-best checkpoint",
        "paper_style": "append diameter/length/packing-density to every Graph-G edge; transfer all compatible 4g weights; zero-initialize only the new column linear adapters to preserve the source function; initialize a new monotonic output head from source q50/differences; train new column RBF adapters plus last2 GNN layers plus head",
        "column_features": list(COLUMN_NAMES),
        "column_spec_4g_repository_values": list(COLUMN_SPEC_4G),
        "column_spec_8g_repository_values": list(COLUMN_SPEC_8G),
        "paper_source": PAPER_URL,
        "paper_scope_note": "paper specifies new column input, updated output layer and lr=1e-4, but not an exact freeze map; the adapter+last2 range is therefore an explicit minimal implementation choice",
        "selection": decision["rule"],
        "test_role": "final reporting only",
        "epochs": 2 if args.smoke else args.epochs,
        "patience": 2 if args.smoke else args.patience,
        "batch_size": args.batch_size,
        "reused_g03_last2": bool(not args.no_reuse_g03_last2 and not args.smoke),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    valid = training.groupby("config")["normalized_valid_score"].mean()
    test = training.groupby("config")["normalized_test_score"].mean()
    readme = f"""# G0-4：Paper-style transfer qualification

## 状态与决定

本实验在G0-3冻结的574行no-threshold 8g数据、3 seeds × row/compound paired splits上，只比较`last2+head`、`full fine-tune`和`paper-style`。三者共享source scaler、单调分位数头、V1/V2等权loss、`lr=1e-4`和validation-best checkpoint；test不参与结构选择。

Validation-only决定：**{decision['selected_transfer']}**。paper-style相对last2的平均validation normalized score变化为{decision['paper_style_mean_validation_relative_change']:+.1%}，赢{decision['paper_style_validation_wins']}/{decision['paired_contexts']}个paired contexts。预注册规则为：平均至少改善2%、至少赢4/6且任一split-mode均值不恶化超过5%；否则保留last2。

## 三种实现

- `last2_head`：复用G0-3完全同协议的冻结checkpoint。
- `full_finetune`：加载4g权重与单调头后更新全部参数。
- `paper_style`：把仓库预留的`column_dia/column_len/column_den`追加到Graph G每条edge；加载所有形状兼容的4g参数；新增column线性适配器零初始化以保持source初始函数；从source q50与分位差初始化新的单调输出头；更新新增column RBF adapters、末两层GNN和head。

8g柱规格沿用仓库原始实现中的`(1.5, 13.2, 0.4458)`；4g参照值为`(1.5, 6.6, 0.4458)`。论文明确要求新柱规格输入、输出层更新及`1e-4`微调，但没有给出逐层冻结图，因此本实验的adapter+last2范围是克制且显式记录的实现选择，而非声称逐行复刻论文。

## 汇总

平均normalized score（越低越好）：

| config | validation | test |
|---|---:|---:|
"""
    for name in CONFIGS:
        readme += f"| {name} | {valid[name]:.4f} | {test[name]:.4f} |\n"
    readme += "\nCalibration只在各run的validation估计per-target inflation；完整full/common/tail结果见`summary.csv`和`slice_metrics.csv`。\n"
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    write_artifact_manifest(output_dir)
    print(json.dumps({"decision": decision, "validation": valid.to_dict(), "test": test.to_dict()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
