#!/usr/bin/env python3
"""Run Gate 0-1 paired legacy-versus-monotonic quantile experiments.

This study intentionally changes only the quantile parameterisation.  It reuses
the frozen E0-3b 8g splits, the 4g anchor checkpoint, the source scaler, and the
provisional last-two-GNN-layers-plus-head transfer protocol.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_e0_4g_baseline import sha256_file, write_artifact_manifest, write_environment
from scripts.run_e0_8g_controls import DATA_DIR, FROZEN_SPLIT_DIR, flatten_result, load_graph_cache
from scripts.run_e0_8g_transfer import (
    SOURCE_4G_DIR,
    build_model,
    build_model_data,
    configure_trainable,
    make_loaders,
    metrics_from_arrays,
    quantile_target_loss,
    set_training_mode,
    validation_scores,
)


CONFIGS = ("legacy_independent", "monotonic_softplus")
TARGETS = ("V1", "V2")


class MonotonicQuantileHead(nn.Module):
    """Positive median with non-negative lower/upper softplus increments.

    For each target the head emits unconstrained ``(m_raw, d_low, d_high)`` and
    returns ``(m-softplus(d_low), m, m+softplus(d_high))`` with
    ``m=softplus(m_raw)``.  The centre and increment weights are initialised
    from the legacy q50 and pairwise quantile differences, respectively.
    """

    def __init__(self, legacy_linear: nn.Linear):
        super().__init__()
        self.linear = nn.Linear(legacy_linear.in_features, 6)
        with torch.no_grad():
            old_weight = legacy_linear.weight.detach()
            old_bias = legacy_linear.bias.detach()
            for offset in (0, 3):
                self.linear.weight[offset].copy_(old_weight[offset + 1])
                self.linear.bias[offset].copy_(old_bias[offset + 1])
                self.linear.weight[offset + 1].copy_(old_weight[offset + 1] - old_weight[offset])
                self.linear.bias[offset + 1].copy_(old_bias[offset + 1] - old_bias[offset])
                self.linear.weight[offset + 2].copy_(old_weight[offset + 2] - old_weight[offset + 1])
                self.linear.bias[offset + 2].copy_(old_bias[offset + 2] - old_bias[offset + 1])

    def forward(self, h_graph: torch.Tensor) -> torch.Tensor:
        raw = self.linear(h_graph)
        outputs = []
        for offset in (0, 3):
            median = F.softplus(raw[:, offset])
            lower = median - F.softplus(raw[:, offset + 1])
            upper = median + F.softplus(raw[:, offset + 2])
            outputs.extend((lower, median, upper))
        return torch.stack(outputs, dim=1)


def install_monotonic_head(model: nn.Module) -> None:
    legacy_head = model.graph_pred_linear
    if not isinstance(legacy_head, nn.Sequential) or not isinstance(legacy_head[0], nn.Linear):
        raise TypeError("Expected the legacy Sequential(Linear, ReLU) prediction head")
    model.graph_pred_linear = MonotonicQuantileHead(legacy_head[0]).to(
        next(model.parameters()).device
    )


def add_crossing_metrics(metrics: dict[str, float], pred: np.ndarray) -> dict[str, float]:
    result = dict(metrics)
    crossing_by_target = []
    for target_index, name in enumerate(TARGETS):
        quantiles = pred[:, target_index * 3 : target_index * 3 + 3]
        crossing = (quantiles[:, 0] > quantiles[:, 1]) | (quantiles[:, 1] > quantiles[:, 2])
        result[f"{name}_quantile_crossing_rate"] = float(crossing.mean())
        crossing_by_target.append(crossing)
    result["quantile_crossing_rate"] = float(np.logical_or.reduce(crossing_by_target).mean())
    return result


def predict(model, loaders, device: torch.device) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    true_values, predictions, indices = [], [], []
    with torch.no_grad():
        for atom_batch, angle_batch in zip(*loaders):
            atom_batch, angle_batch = atom_batch.to(device), angle_batch.to(device)
            pred, _ = model(atom_batch, angle_batch)
            predictions.append(pred.cpu().numpy())
            true_values.append(atom_batch.y.cpu().numpy())
            indices.append(atom_batch.canonical_position.cpu().numpy().reshape(-1))
    y_true = np.vstack(true_values)
    pred = np.vstack(predictions)
    data_indices = np.concatenate(indices)
    return add_crossing_metrics(metrics_from_arrays(y_true, pred), pred), y_true, pred, data_indices


def prediction_rows(
    canonical_df: pd.DataFrame,
    config_name: str,
    split_name: str,
    y_true: np.ndarray,
    pred: np.ndarray,
    indices: np.ndarray,
) -> list[dict]:
    rows = []
    for row_index, true, prediction in zip(indices, y_true, pred):
        source = canonical_df.iloc[int(row_index)]
        rows.append(
            {
                "config": config_name,
                "evaluation_split": split_name,
                "sample_id": source["sample_id"],
                "source_row_1based": int(source["source_row_1based"]),
                "canonical_index": int(row_index),
                "V1_true": float(true[0]),
                "V2_true": float(true[1]),
                "V1_q10": float(prediction[0]),
                "V1_q50": float(prediction[1]),
                "V1_q90": float(prediction[2]),
                "V2_q10": float(prediction[3]),
                "V2_q50": float(prediction[4]),
                "V2_q90": float(prediction[5]),
            }
        )
    return rows


def run_configuration(
    config_name: str,
    loaders: dict,
    canonical_df: pd.DataFrame,
    run_dir: Path,
    source_checkpoint: Path,
    seed: int,
    epochs: int,
    patience: int,
    target_variance: dict[str, float],
) -> tuple[dict, list[dict]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    source = torch.load(source_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(source["model_state_dict"])
    if config_name == "monotonic_softplus":
        install_monotonic_head(model)
    elif config_name != "legacy_independent":
        raise ValueError(config_name)
    trainable, total = configure_trainable(model, "last2_head")

    checkpoint_dir = run_dir / "checkpoints"
    history_dir = run_dir / "histories"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{config_name}.pt"
    history = []
    best_score, best_epoch, stale = float("inf"), 0, 0
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
        weight_decay=1e-5,
    )
    for epoch in range(1, epochs + 1):
        set_training_mode(model)
        total_loss, batches = 0.0, 0
        for atom_batch, angle_batch in zip(*loaders["train"]):
            atom_batch, angle_batch = atom_batch.to(device), angle_batch.to(device)
            pred, _ = model(atom_batch, angle_batch)
            loss_v1 = quantile_target_loss(atom_batch.y[:, 0], pred[:, 0:3])
            loss_v2 = quantile_target_loss(atom_batch.y[:, 1], pred[:, 3:6])
            loss = loss_v1 + loss_v2
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
                    "legacy_valid_score": legacy_score,
                    "config_name": config_name,
                    "quantile_parameterization": config_name,
                    "mode": "last2_head",
                    "learning_rate": 1e-4,
                    "v2_weight": 1.0,
                    "seed": seed,
                    "trainable_parameters": trainable,
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
    valid_metrics, valid_true, valid_pred, valid_indices = predict(model, loaders["valid"], device)
    test_metrics, test_true, test_pred, test_indices = predict(model, loaders["test"], device)
    best_score, legacy_score = validation_scores(valid_metrics, target_variance)
    rows = prediction_rows(
        canonical_df, config_name, "valid", valid_true, valid_pred, valid_indices
    )
    rows.extend(
        prediction_rows(canonical_df, config_name, "test", test_true, test_pred, test_indices)
    )
    result = {
        "config": config_name,
        "mode": "last2_head",
        "pretrained": True,
        "learning_rate": 1e-4,
        "v2_weight": 1.0,
        "loss_scaling": "legacy_units",
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "normalized_valid_score": best_score,
        "legacy_valid_score": legacy_score,
        "normalized_test_score": validation_scores(test_metrics, target_variance)[0],
        "valid": valid_metrics,
        "test": test_metrics,
    }
    return result, rows


def git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def summarize(comparison: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "normalized_valid_score",
        "normalized_test_score",
        "valid_V1_rmse",
        "valid_V2_rmse",
        "test_V1_rmse",
        "test_V2_rmse",
        "test_V1_interval_80_coverage",
        "test_V2_interval_80_coverage",
        "test_V1_interval_80_mean_width",
        "test_V2_interval_80_mean_width",
        "valid_quantile_crossing_rate",
        "test_quantile_crossing_rate",
    ]
    return comparison.groupby("config", as_index=False)[metric_columns].agg(["mean", "std"])


def paired_effects(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split_mode, seed), group in comparison.groupby(["split_mode", "seed"]):
        legacy = group[group["config"].eq("legacy_independent")].iloc[0]
        for _, candidate in group.iterrows():
            rows.append(
                {
                    "split_mode": split_mode,
                    "seed": int(seed),
                    "config": candidate["config"],
                    "delta_normalized_valid_vs_legacy": candidate["normalized_valid_score"]
                    - legacy["normalized_valid_score"],
                    "delta_valid_V1_rmse_vs_legacy": candidate["valid_V1_rmse"]
                    - legacy["valid_V1_rmse"],
                    "delta_valid_V2_rmse_vs_legacy": candidate["valid_V2_rmse"]
                    - legacy["valid_V2_rmse"],
                    "delta_test_crossing_vs_legacy": candidate["test_quantile_crossing_rate"]
                    - legacy["test_quantile_crossing_rate"],
                }
            )
    return pd.DataFrame(rows)


def write_readme(output_dir: Path, comparison: pd.DataFrame, smoke: bool) -> None:
    means = comparison.groupby("config").mean(numeric_only=True)
    legacy = means.loc["legacy_independent"]
    monotonic = means.loc["monotonic_softplus"]
    relative_changes = {
        metric: monotonic[metric] / legacy[metric] - 1
        for metric in ("normalized_valid_score", "valid_V1_rmse", "valid_V2_rmse")
    }
    gate_passed = all(value <= 0.05 for value in relative_changes.values()) and bool(
        comparison.loc[
            comparison["config"].eq("monotonic_softplus"), "valid_quantile_crossing_rate"
        ].max()
        == 0
    )
    status = "smoke test（不作方法结论）" if smoke else "正式配对实验"
    gate_text = "smoke 不作 Gate 判定" if smoke else ("通过" if gate_passed else "未通过")
    text = f"""# G0-1：分位数单调性对照

## 状态

本目录是{status}。比较只改变分位数输出形式；数据、split、4g anchor、迁移范围、学习率、损失权重和 checkpoint 选择规则保持一致。test 不参与模型选择。

## 协议

- Legacy：独立 q10/q50/q90 输出，并保留 crossing penalty。
- Monotonic：`q50=m`、`q10=m-softplus(d_low)`、`q90=m+softplus(d_high)`。
- 迁移：4g pretrained，last2+head，`lr=1e-4`，V1/V2 等权。
- 选择：仅按 validation 的 train-variance normalized q50 score 选择 checkpoint。
- 正式冻结标准：validation normalized score 与 V1/V2 RMSE 均不比 legacy 平均恶化超过 5%，且 crossing 为 0。

## 当前汇总

| 参数化 | valid normalized | test normalized | valid crossing | test crossing | test V1/V2 coverage |
|---|---:|---:|---:|---:|---:|
| Legacy | {legacy['normalized_valid_score']:.4f} | {legacy['normalized_test_score']:.4f} | {legacy['valid_quantile_crossing_rate']:.4f} | {legacy['test_quantile_crossing_rate']:.4f} | {legacy['test_V1_interval_80_coverage']:.3f} / {legacy['test_V2_interval_80_coverage']:.3f} |
| Monotonic | {monotonic['normalized_valid_score']:.4f} | {monotonic['normalized_test_score']:.4f} | {monotonic['valid_quantile_crossing_rate']:.4f} | {monotonic['test_quantile_crossing_rate']:.4f} | {monotonic['test_V1_interval_80_coverage']:.3f} / {monotonic['test_V2_interval_80_coverage']:.3f} |

## Gate 判定

G0-1：**{gate_text}**。Monotonic 相对 legacy 的 validation 变化为 normalized score {relative_changes['normalized_valid_score']:+.2%}、V1 RMSE {relative_changes['valid_V1_rmse']:+.2%}、V2 RMSE {relative_changes['valid_V2_rmse']:+.2%}；正式阈值均为不恶化超过5%。模型选择只使用 validation，test 仅作最终报告。

## 产物

- `comparison.csv`：逐 split/seed/config 指标。
- `summary.csv`：跨运行均值和标准差。
- `paired_effects_vs_legacy.csv`：配对差值。
- `predictions.csv.gz`：validation/test 的逐样本预测，为 G0-2 validation-only calibration 保留。
- 各运行目录中的 checkpoint 与 history：可复核 checkpoint 选择和继续 G0-2。
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "experiments" / "g0_1_quantile_monotonicity"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 525, 1101])
    parser.add_argument(
        "--split-modes", nargs="+", choices=["row", "compound"], default=["row", "compound"]
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if (output_dir / "comparison.csv").exists():
        raise FileExistsError(f"Refusing to overwrite finalized experiment: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_environment(output_dir)

    canonical_df = pd.read_csv(DATA_DIR / "canonical_8g.csv")
    graph_cache = load_graph_cache()
    scaler = json.loads((SOURCE_4G_DIR / "scaler.json").read_text(encoding="utf-8"))
    source_checkpoint = SOURCE_4G_DIR / "checkpoints" / "best.pt"
    split_manifest = {}
    all_results, all_predictions = [], []

    for split_mode in args.split_modes:
        for seed in args.seeds:
            run_key = f"{split_mode}_seed_{seed}"
            split_path = FROZEN_SPLIT_DIR / f"{run_key}.csv"
            split_df = pd.read_csv(split_path)
            split_manifest[run_key] = {
                "path": str(split_path.relative_to(ROOT)),
                "sha256": sha256_file(split_path),
            }
            atom_data, angle_data = build_model_data(canonical_df, graph_cache, split_df, scaler)
            loaders = make_loaders(atom_data, angle_data, split_df, args.batch_size)
            train_mask = split_df["split"].eq("train").to_numpy()
            train_labels = canonical_df.loc[train_mask, ["V1_ml", "V2_ml"]]
            target_variance = {
                "V1": float(train_labels["V1_ml"].var(ddof=0)),
                "V2": float(train_labels["V2_ml"].var(ddof=0)),
            }
            run_dir = output_dir / "runs" / run_key
            for config_name in CONFIGS:
                result_path = run_dir / f"{config_name}.result.json"
                prediction_path = run_dir / f"{config_name}.predictions.csv"
                if result_path.exists() and prediction_path.exists():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    rows = pd.read_csv(prediction_path).to_dict(orient="records")
                else:
                    print(json.dumps({"starting": run_key, "config": config_name}), flush=True)
                    result, rows = run_configuration(
                        config_name=config_name,
                        loaders=loaders,
                        canonical_df=canonical_df,
                        run_dir=run_dir,
                        source_checkpoint=source_checkpoint,
                        seed=seed,
                        epochs=2 if args.smoke else args.epochs,
                        patience=2 if args.smoke else args.patience,
                        target_variance=target_variance,
                    )
                    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                    pd.DataFrame(rows).to_csv(prediction_path, index=False)
                result_row = flatten_result(result)
                result_row.update(
                    {
                        "split_mode": split_mode,
                        "seed": seed,
                        "train_rows": int(train_mask.sum()),
                        "valid_rows": int(split_df["split"].eq("valid").sum()),
                        "test_rows": int(split_df["split"].eq("test").sum()),
                    }
                )
                all_results.append(result_row)
                for row in rows:
                    row.update({"split_mode": split_mode, "seed": seed})
                    all_predictions.append(row)
                print(
                    json.dumps(
                        {
                            "finished": run_key,
                            "config": config_name,
                            "normalized_valid_score": result["normalized_valid_score"],
                            "valid_crossing": result["valid"]["quantile_crossing_rate"],
                        }
                    ),
                    flush=True,
                )

    comparison = pd.DataFrame(all_results).sort_values(["split_mode", "seed", "config"])
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    summary = summarize(comparison)
    summary.columns = [
        "_".join(part for part in column if part) for column in summary.columns.to_flat_index()
    ]
    summary.to_csv(output_dir / "summary.csv", index=False)
    paired_effects(comparison).to_csv(output_dir / "paired_effects_vs_legacy.csv", index=False)
    pd.DataFrame(all_predictions).to_csv(
        output_dir / "predictions.csv.gz", index=False, compression="gzip"
    )
    config = {
        "stage": "G0-1_quantile_monotonicity",
        "smoke": args.smoke,
        "git_commit": git_commit(),
        "data": str((DATA_DIR / "canonical_8g.csv").relative_to(ROOT)),
        "source_checkpoint": str(source_checkpoint.relative_to(ROOT)),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "splits": split_manifest,
        "seeds": args.seeds,
        "split_modes": args.split_modes,
        "epochs": 2 if args.smoke else args.epochs,
        "patience": 2 if args.smoke else args.patience,
        "batch_size": args.batch_size,
        "transfer": "last2_head",
        "learning_rate": 1e-4,
        "loss": "L_V1 + L_V2; each target uses q10 pinball + q50 MSE + q90 pinball + crossing penalty",
        "checkpoint_selection": "validation-only train-variance normalized q50 MSE",
        "acceptance": {
            "crossing_rate": 0.0,
            "max_relative_mean_degradation_normalized_validation": 0.05,
            "max_relative_mean_degradation_each_validation_rmse": 0.05,
        },
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_readme(output_dir, comparison, args.smoke)
    write_artifact_manifest(output_dir)
    print(comparison.to_json(orient="records"), flush=True)


if __name__ == "__main__":
    main()
