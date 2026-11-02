#!/usr/bin/env python3
"""Run the E0-3 4g-to-8g transfer-learning comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qgeognn_graphs import build_graph_and_descriptor  # noqa: E402
from scripts.run_e0_4g_baseline import (  # noqa: E402
    ELUENT_SMILES,
    LOADING_SOLVENT,
    V1_LIMIT_ML,
    V2_LIMIT_ML,
    eluent_descriptor,
    make_compound_split,
    make_split,
    minmax_fit,
    minmax_transform,
    qg,
    ratio_valid,
    sha256_file,
    write_environment,
    write_artifact_manifest,
)


EXPECTED_SPEC = "Silica-CS 4g+4g"
SOURCE_4G_DIR = ROOT / "experiments" / "e0_4g_baseline"
RDLogger.DisableLog("rdApp.warning")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qgeognn_matplotlib")


def stable_sample_id(source_hash: str, source_row_1based: int) -> str:
    return hashlib.sha256(
        f"8g:{source_hash}:{source_row_1based}".encode("utf-8")
    ).hexdigest()[:20]


def prepare_8g(source: Path, output_dir: Path, seed: int) -> tuple[pd.DataFrame, dict]:
    source_hash = sha256_file(source)
    df = pd.read_csv(source).copy()
    df["source_row_1based"] = np.arange(len(df), dtype=int) + 2
    df["sample_id"] = [stable_sample_id(source_hash, row) for row in df["source_row_1based"]]
    numeric_columns = [
        "Density g/ml",
        "V/ul",
        "Volume of loading solvent/ul",
        "Flow mL/min",
        "t1",
        "t2",
    ]
    numeric = {name: pd.to_numeric(df[name], errors="coerce") for name in numeric_columns}
    df["V1_ml"] = numeric["t1"] * numeric["Flow mL/min"] / 1200.0
    df["V2_ml"] = numeric["t2"] * numeric["Flow mL/min"] / 1200.0
    decisions: list[list[str]] = [[] for _ in range(len(df))]
    quality_flags: list[list[str]] = [[] for _ in range(len(df))]

    def flag(target: list[list[str]], mask: pd.Series, reason: str) -> None:
        for index in df.index[mask.fillna(False)]:
            target[int(index)].append(reason)

    flag(decisions, ~df["column_specs"].eq(EXPECTED_SPEC), "unexpected_column_spec")
    flag(decisions, numeric["t1"].isna() | numeric["t1"].eq(-1), "reader_rejects_t1")
    core_missing = (
        df[["CAS", "loading solvent", "PE/EA", "column_specs", "smiles"]]
        .isna()
        .any(axis=1)
        | pd.concat([numeric[name].isna() for name in numeric_columns], axis=1).any(axis=1)
    )
    flag(quality_flags, core_missing, "missing_core")
    flag(quality_flags, numeric["t1"].lt(0) | numeric["t2"].lt(0), "negative_label")
    flag(quality_flags, numeric["t1"].gt(numeric["t2"]), "t1_gt_t2")
    flag(quality_flags, ~df["PE/EA"].map(ratio_valid), "invalid_ratio")
    smiles_valid = df["smiles"].map(
        lambda value: pd.notna(value)
        and bool(str(value).strip())
        and Chem.MolFromSmiles(str(value)) is not None
    )
    flag(decisions, ~smiles_valid, "invalid_smiles")
    flag(decisions, df["V1_ml"].gt(V1_LIMIT_ML), "over_v1_legacy_threshold")
    flag(decisions, df["V2_ml"].gt(V2_LIMIT_ML), "over_v2_legacy_threshold")

    source_cache = torch.load(SOURCE_4G_DIR / "graph_cache_4g.pt", weights_only=False)
    # Reuse source-domain graphs in memory, but persist only genuinely new 8g
    # structures.  The current 8g set overlaps the 4g cache almost completely.
    graph_cache: dict[str, dict] = {}
    target_only_graph_cache: dict[str, dict] = {}
    graph_audit = []
    cache_path = output_dir / "graph_cache_8g_only.pt"
    audit_path = output_dir / "graph_audit_8g.csv"
    pre_graph_keep = pd.Series([not reasons for reasons in decisions], index=df.index)
    smiles_by_canonical = {}
    for smiles in df.loc[pre_graph_keep, "smiles"].astype(str).unique():
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
        smiles_by_canonical.setdefault(canonical, smiles)
    for index, canonical in enumerate(sorted(smiles_by_canonical)):
        start = time.time()
        if canonical in source_cache:
            graph_cache[canonical] = source_cache[canonical]
            origin, status, error = "4g", "success", ""
        else:
            try:
                graph, descriptor, conformer = build_graph_and_descriptor(
                    smiles_by_canonical[canonical], seed + index
                )
                graph_cache[canonical] = {"graph": graph, "descriptor": descriptor}
                target_only_graph_cache[canonical] = graph_cache[canonical]
                origin, status, error = "8g_generated", "success", ""
            except Exception as exc:
                conformer, origin, status = {}, "8g_generated", "failed"
                error = f"{type(exc).__name__}: {exc}"
        graph_audit.append(
            {
                "canonical_smiles": canonical,
                "source_smiles": smiles_by_canonical[canonical],
                "status": status,
                "graph_origin": origin,
                **({"conformer_policy": "reused_4g_cache"} if origin == "4g" else conformer),
                "error": error,
                "seconds": round(time.time() - start, 4),
            }
        )
    failed = {row["canonical_smiles"] for row in graph_audit if row["status"] != "success"}
    for index in df.index[pre_graph_keep]:
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(str(df.at[index, "smiles"])), canonical=True)
        if canonical in failed:
            decisions[int(index)].append("graph_construction_failed")

    df["decision"] = ["keep" if not reasons else "drop" for reasons in decisions]
    df["decision_reason"] = [";".join(sorted(set(reasons))) for reasons in decisions]
    df["quality_flags"] = [";".join(sorted(set(flags))) for flags in quality_flags]
    df["reviewer"] = "user_confirmed_protocol_2026-08-13"
    canonical_df = df.loc[df["decision"].eq("keep")].copy().reset_index(drop=True)
    canonical_df["canonical_smiles"] = canonical_df["smiles"].map(
        lambda value: Chem.MolToSmiles(Chem.MolFromSmiles(str(value)), canonical=True)
    )
    df.to_csv(output_dir / "sample_decisions_8g.csv", index=False)
    canonical_df.to_csv(output_dir / "canonical_8g.csv", index=False)
    pd.DataFrame(graph_audit).to_csv(audit_path, index=False)
    torch.save(target_only_graph_cache, cache_path)
    metadata = {
        "source_file": str(source.relative_to(ROOT)),
        "source_sha256": source_hash,
        "source_rows": int(len(df)),
        "reader_compatible_rows": int(
            (df["column_specs"].eq(EXPECTED_SPEC) & numeric["t1"].ne(-1) & numeric["t1"].notna()).sum()
        ),
        "canonical_rows": int(len(canonical_df)),
        "unique_canonical_smiles": int(canonical_df["canonical_smiles"].nunique()),
        "graphs_reused_from_4g": int(sum(row["graph_origin"] == "4g" for row in graph_audit)),
        "graphs_generated_for_8g": int(sum(row["graph_origin"] == "8g_generated" for row in graph_audit)),
        "graph_failures": int(sum(row["status"] != "success" for row in graph_audit)),
        "repeated_experiments_policy": "retain",
        "quality_warnings_retained": {
            "negative_label_rows": int(canonical_df["quality_flags"].str.contains("negative_label", na=False).sum()),
            "t1_gt_t2_rows": int(canonical_df["quality_flags"].str.contains("t1_gt_t2", na=False).sum()),
        },
        "volume_formula": "V_ml = t_raw * Flow_mL_min / 1200",
        "volume_thresholds_ml": {"V1": V1_LIMIT_ML, "V2": V2_LIMIT_ML},
        "rdkit_seed": seed,
    }
    (output_dir / "data_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return canonical_df, graph_cache


def load_or_fit_scaler(
    descriptors: np.ndarray,
    eluents: np.ndarray,
    train_mask: np.ndarray,
    policy: str,
) -> dict:
    if policy == "source_4g":
        return json.loads((SOURCE_4G_DIR / "scaler.json").read_text(encoding="utf-8"))
    if policy == "target_train":
        return {
            "fit_split": "8g_train",
            "descriptor": minmax_fit(descriptors[train_mask]),
            "eluent": minmax_fit(eluents[train_mask]),
        }
    raise ValueError(policy)


def build_model_data(
    canonical_df: pd.DataFrame,
    graph_cache: dict,
    split_df: pd.DataFrame,
    scaler: dict,
) -> tuple[list[Data], list[Data]]:
    descriptors = np.vstack(
        [graph_cache[smiles]["descriptor"] for smiles in canonical_df["canonical_smiles"]]
    ).astype(np.float32)
    eluents = np.vstack([eluent_descriptor(value) for value in canonical_df["PE/EA"]]).astype(np.float32)
    descriptors = minmax_transform(descriptors, scaler["descriptor"]).astype(np.float32)
    eluents = minmax_transform(eluents, scaler["eluent"]).astype(np.float32)
    atom_bond, bond_angle = [], []
    for index, row in canonical_df.iterrows():
        graph = graph_cache[row["canonical_smiles"]]["graph"]
        atom_feature = torch.from_numpy(
            np.stack([graph[name] for name in qg.atom_id_names], axis=1)
        ).to(torch.int64)
        bond_ids = torch.from_numpy(
            np.stack([graph[name] for name in qg.bond_id_names], axis=1)
        ).to(torch.int64)
        edge_count = bond_ids.shape[0]
        bond_length = torch.from_numpy(graph["bond_length"].astype(np.float32)).reshape(-1, 1)
        prop = torch.tensor(eluents[index], dtype=torch.float32).repeat(edge_count, 1)
        extra = torch.tensor(
            [
                LOADING_SOLVENT[str(row["loading solvent"])],
                float(row["Density g/ml"]) * float(row["V/ul"]),
                float(row["Volume of loading solvent/ul"]),
            ],
            dtype=torch.float32,
        ).repeat(edge_count, 1)
        atom_bond.append(
            Data(
                x=atom_feature,
                edge_index=torch.from_numpy(graph["edges"].T).to(torch.int64),
                edge_attr=torch.cat([bond_ids, bond_length, prop, extra], dim=1),
                y=torch.tensor([[float(row["V1_ml"]), float(row["V2_ml"])]], dtype=torch.float32),
                canonical_position=torch.tensor(index, dtype=torch.int64),
            )
        )
        angle = torch.from_numpy(graph["bond_angle"].astype(np.float32)).reshape(-1, 1)
        descriptor = torch.tensor(descriptors[index], dtype=torch.float32).repeat(angle.shape[0], 1)
        bond_angle.append(
            Data(
                edge_index=torch.from_numpy(graph["BondAngleGraph_edges"].T).to(torch.int64),
                edge_attr=torch.cat([angle, descriptor], dim=1),
            )
        )
    return atom_bond, bond_angle


def make_loaders(
    atom_data: list[Data],
    angle_data: list[Data],
    split_df: pd.DataFrame,
    batch_size: int,
) -> dict:
    loaders = {}
    for split in ["train", "valid", "test"]:
        indices = split_df.index[split_df["split"].eq(split)].tolist()
        loaders[split] = (
            DataLoader([atom_data[i] for i in indices], batch_size=batch_size, shuffle=False),
            DataLoader([angle_data[i] for i in indices], batch_size=batch_size, shuffle=False),
        )
    return loaders


def metrics_from_arrays(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    result = {}
    for target, name in enumerate(["V1", "V2"]):
        quantiles = pred[:, target * 3 : target * 3 + 3]
        residual = y_true[:, target] - quantiles[:, 1]
        ss_res = float(np.square(residual).sum())
        ss_tot = float(np.square(y_true[:, target] - y_true[:, target].mean()).sum())
        errors = y_true[:, [target]] - quantiles
        levels = np.array([0.1, 0.5, 0.9], dtype=np.float32).reshape(1, -1)
        result[f"{name}_mae"] = float(np.abs(residual).mean())
        result[f"{name}_rmse"] = float(np.sqrt(np.square(residual).mean()))
        result[f"{name}_r2"] = float(1 - ss_res / ss_tot) if ss_tot else float("nan")
        result[f"{name}_mean_pinball_loss"] = float(
            np.maximum(levels * errors, (levels - 1) * errors).mean()
        )
        result[f"{name}_interval_80_coverage"] = float(
            np.mean((y_true[:, target] >= quantiles[:, 0]) & (y_true[:, target] <= quantiles[:, 2]))
        )
        result[f"{name}_interval_80_mean_width"] = float(
            np.mean(quantiles[:, 2] - quantiles[:, 0])
        )
    result["quantile_crossing_rate"] = float(
        np.mean(
            (pred[:, 0] > pred[:, 1])
            | (pred[:, 1] > pred[:, 2])
            | (pred[:, 3] > pred[:, 4])
            | (pred[:, 4] > pred[:, 5])
        )
    )
    return result


def predict(model, loaders, device: torch.device) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    true_values, predictions, indices = [], [], []
    with torch.no_grad():
        for atom_batch, angle_batch in zip(*loaders):
            atom_batch, angle_batch = atom_batch.to(device), angle_batch.to(device)
            predictions.append(model(atom_batch, angle_batch)[0].cpu().numpy())
            true_values.append(atom_batch.y.cpu().numpy())
            indices.append(atom_batch.canonical_position.cpu().numpy().reshape(-1))
    y_true, pred, data_indices = np.vstack(true_values), np.vstack(predictions), np.concatenate(indices)
    return metrics_from_arrays(y_true, pred), y_true, pred, data_indices


def build_model(device: torch.device) -> nn.Module:
    qg.device = str(device)
    return qg.GINGraphPooling(
        num_tasks=6,
        num_layers=5,
        emb_dim=128,
        drop_ratio=0.0,
        graph_pooling="sum",
        descriptor_dim=1827,
    ).to(device)


def configure_trainable(model: nn.Module, mode: str) -> tuple[int, int]:
    for parameter in model.parameters():
        parameter.requires_grad = mode == "full"
    if mode != "full":
        layers = []
        if mode == "last1_head":
            layers = [4]
        elif mode == "last2_head":
            layers = [3, 4]
        elif mode != "head_only":
            raise ValueError(mode)
        for name, parameter in model.named_parameters():
            if name.startswith("graph_pred_linear"):
                parameter.requires_grad = True
            if any(
                name.startswith(f"gnn_node.{branch}.{layer}.")
                for branch in [
                    "convs",
                    "convs_bond_angle",
                    "convs_bond_embeding",
                    "convs_bond_float",
                    "convs_angle_float",
                ]
                for layer in layers
            ):
                parameter.requires_grad = True
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def set_training_mode(model: nn.Module) -> None:
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            parameters = list(module.parameters(recurse=False))
            if parameters and not any(parameter.requires_grad for parameter in parameters):
                module.eval()


def validation_scores(metrics: dict, target_variance: dict[str, float]) -> tuple[float, float]:
    legacy = metrics["V1_rmse"] ** 2 + 0.5 * metrics["V2_rmse"] ** 2
    normalized = (
        metrics["V1_rmse"] ** 2 / target_variance["V1"]
        + metrics["V2_rmse"] ** 2 / target_variance["V2"]
    )
    return float(normalized), float(legacy)


def reset_prediction_head(model: nn.Module) -> None:
    """Reinitialize only the task-specific prediction head."""
    for module in model.graph_pred_linear.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()


def quantile_target_loss(true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Original q10/q50/q90 loss for one target, including crossing penalties."""
    return (
        qg.q_loss(0.1, true, pred[:, 0])
        + torch.mean((true - pred[:, 1]) ** 2)
        + qg.q_loss(0.9, true, pred[:, 2])
        + torch.mean(torch.relu(pred[:, 0] - pred[:, 1]))
        + torch.mean(torch.relu(pred[:, 1] - pred[:, 2]))
    )


def run_configuration(
    config_name: str,
    mode: str,
    learning_rate: float | None,
    pretrained: bool,
    loaders: dict,
    canonical_df: pd.DataFrame,
    output_dir: Path,
    source_checkpoint: Path,
    seed: int,
    epochs: int,
    patience: int,
    target_variance: dict[str, float],
    v2_weight: float = 0.5,
    loss_scaling: str = "legacy",
    reinitialize_head: bool = False,
) -> tuple[dict, list[dict]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    if pretrained:
        checkpoint = torch.load(source_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    if reinitialize_head:
        reset_prediction_head(model)
    trainable, total = (0, sum(p.numel() for p in model.parameters()))
    if mode != "zero_shot":
        trainable, total = configure_trainable(model, mode)

    checkpoint_dir = output_dir / "checkpoints"
    history_dir = output_dir / "histories"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{config_name}.pt"
    history = []
    best_score, best_epoch, stale = float("inf"), 0, 0
    if mode == "zero_shot":
        valid_metrics, *_ = predict(model, loaders["valid"], device)
        best_score, legacy_score = validation_scores(valid_metrics, target_variance)
    else:
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(learning_rate),
            weight_decay=1e-5,
        )
        for epoch in range(1, epochs + 1):
            set_training_mode(model)
            total_loss, batches = 0.0, 0
            for atom_batch, angle_batch in zip(*loaders["train"]):
                atom_batch, angle_batch = atom_batch.to(device), angle_batch.to(device)
                pred = model(atom_batch, angle_batch)[0]
                true_1, true_2 = atom_batch.y[:, 0], atom_batch.y[:, 1]
                if loss_scaling == "legacy":
                    loss_1 = quantile_target_loss(true_1, pred[:, 0:3])
                    loss_2 = quantile_target_loss(true_2, pred[:, 3:6])
                elif loss_scaling == "target_standard_deviation":
                    scale_1 = max(float(np.sqrt(target_variance["V1"])), 1e-8)
                    scale_2 = max(float(np.sqrt(target_variance["V2"])), 1e-8)
                    loss_1 = quantile_target_loss(true_1 / scale_1, pred[:, 0:3] / scale_1)
                    loss_2 = quantile_target_loss(true_2 / scale_2, pred[:, 3:6] / scale_2)
                else:
                    raise ValueError(f"Unknown loss scaling: {loss_scaling}")
                loss = loss_1 + float(v2_weight) * loss_2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                batches += 1
            valid_metrics, *_ = predict(model, loaders["valid"], device)
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
                        "mode": mode,
                        "learning_rate": learning_rate,
                        "v2_weight": v2_weight,
                        "loss_scaling": loss_scaling,
                        "reinitialize_head": reinitialize_head,
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
        valid_metrics, *_ = predict(model, loaders["valid"], device)
        best_score, legacy_score = validation_scores(valid_metrics, target_variance)

    test_metrics, y_true, pred, indices = predict(model, loaders["test"], device)
    prediction_rows = []
    for row_index, true, prediction in zip(indices, y_true, pred):
        row = canonical_df.iloc[int(row_index)]
        prediction_rows.append(
            {
                "config": config_name,
                "sample_id": row["sample_id"],
                "source_row_1based": int(row["source_row_1based"]),
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
    result = {
        "config": config_name,
        "mode": mode,
        "pretrained": pretrained,
        "learning_rate": learning_rate,
        "v2_weight": v2_weight,
        "loss_scaling": loss_scaling,
        "reinitialize_head": reinitialize_head,
        "best_epoch": best_epoch if mode != "zero_shot" else None,
        "epochs_run": len(history),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "normalized_valid_score": best_score,
        "legacy_valid_score": legacy_score,
        "valid": valid_metrics,
        "test": test_metrics,
    }
    result["normalized_test_score"] = validation_scores(test_metrics, target_variance)[0]
    return result, prediction_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "dataset" / "dataset_8g.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "e0_8g_transfer")
    parser.add_argument("--source-checkpoint", type=Path, default=SOURCE_4G_DIR / "checkpoints" / "best.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--configs", nargs="*", default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_environment(args.output_dir)
    if args.reuse_cache:
        canonical_df = pd.read_csv(args.output_dir / "canonical_8g.csv")
        graph_cache = dict(
            torch.load(SOURCE_4G_DIR / "graph_cache_4g.pt", weights_only=False)
        )
        graph_cache.update(
            torch.load(args.output_dir / "graph_cache_8g_only.pt", weights_only=False)
        )
    else:
        canonical_df, graph_cache = prepare_8g(args.source, args.output_dir, args.seed)
    split_df = make_split(canonical_df, args.output_dir, args.seed)
    compound_split_df = make_compound_split(canonical_df, args.output_dir, args.seed)
    raw_descriptors = np.vstack(
        [graph_cache[smiles]["descriptor"] for smiles in canonical_df["canonical_smiles"]]
    ).astype(np.float32)
    raw_eluents = np.vstack([eluent_descriptor(value) for value in canonical_df["PE/EA"]]).astype(np.float32)
    train_mask = split_df["split"].eq("train").to_numpy()
    scalers = {
        "source_4g": load_or_fit_scaler(raw_descriptors, raw_eluents, train_mask, "source_4g"),
        "target_train": load_or_fit_scaler(raw_descriptors, raw_eluents, train_mask, "target_train"),
    }
    scaler_dir = args.output_dir / "scalers"
    scaler_dir.mkdir(parents=True, exist_ok=True)
    (scaler_dir / "target_train.json").write_text(
        json.dumps(scalers["target_train"], indent=2), encoding="utf-8"
    )
    config = {
        "stage": "E0-3_4g_to_8g_transfer",
        "source": str(args.source.relative_to(ROOT)),
        "source_checkpoint": str(args.source_checkpoint.relative_to(ROOT)),
        "source_checkpoint_sha256": sha256_file(args.source_checkpoint),
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "training_loss": "legacy L_V1 + 0.5 * L_V2",
        "checkpoint_selection": "MSE_V1/Var_train_V1 + MSE_V2/Var_train_V2",
        "pretrained_scaler": "source_4g",
        "scratch_scaler": "target_train",
        "row_split_counts": split_df["split"].value_counts().to_dict(),
        "compound_group_split_counts": compound_split_df["split"].value_counts().to_dict(),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if args.prepare_only:
        write_artifact_manifest(args.output_dir)
        print(json.dumps({"prepared": len(canonical_df), **config}, ensure_ascii=False))
        return

    data_by_scaler = {}
    for scaler_name, scaler in scalers.items():
        atom_data, angle_data = build_model_data(canonical_df, graph_cache, split_df, scaler)
        data_by_scaler[scaler_name] = make_loaders(atom_data, angle_data, split_df, args.batch_size)
    train_labels = canonical_df.loc[train_mask, ["V1_ml", "V2_ml"]]
    target_variance = {
        "V1": float(train_labels["V1_ml"].var(ddof=0)),
        "V2": float(train_labels["V2_ml"].var(ddof=0)),
    }
    specifications = [("zero_shot", "zero_shot", None, True, "source_4g")]
    specifications.append(("scratch_lr1e-3", "full", 1e-3, False, "target_train"))
    for mode in ["head_only", "last1_head", "last2_head", "full"]:
        for learning_rate in [1e-5, 1e-4]:
            name = f"{mode}_lr{learning_rate:.0e}".replace("-0", "-")
            specifications.append((name, mode, learning_rate, True, "source_4g"))
    if args.configs:
        requested = set(args.configs)
        specifications = [spec for spec in specifications if spec[0] in requested]
        missing = requested - {spec[0] for spec in specifications}
        if missing:
            raise ValueError(f"Unknown configs: {sorted(missing)}")
    results, predictions = [], []
    partial_results_path = args.output_dir / ".partial_results.json"
    for name, mode, learning_rate, pretrained, scaler_name in specifications:
        print(json.dumps({"starting": name}, ensure_ascii=False), flush=True)
        result, rows = run_configuration(
            name,
            mode,
            learning_rate,
            pretrained,
            data_by_scaler[scaler_name],
            canonical_df,
            args.output_dir,
            args.source_checkpoint,
            args.seed,
            args.epochs,
            args.patience,
            target_variance,
        )
        result["scaler"] = scaler_name
        results.append(result)
        predictions.extend(rows)
        partial_results_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pd.DataFrame(predictions).to_csv(args.output_dir / "test_predictions.csv", index=False)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summary_rows = []
    for result in results:
        row = {key: value for key, value in result.items() if key not in {"valid", "test"}}
        row.update({f"valid_{key}": value for key, value in result["valid"].items()})
        row.update({f"test_{key}": value for key, value in result["test"].items()})
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("normalized_valid_score")
    summary.to_csv(args.output_dir / "comparison.csv", index=False)
    transfer_candidates = summary[(summary["pretrained"]) & (summary["mode"] != "zero_shot")]
    if not transfer_candidates.empty:
        selected = transfer_candidates.iloc[0].to_dict()
        selected_name = str(selected["config"])
        (args.output_dir / "selected_transfer.json").write_text(
            json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for checkpoint_path in (args.output_dir / "checkpoints").glob("*.pt"):
            if checkpoint_path.name != f"{selected_name}.pt":
                checkpoint_path.unlink()
    partial_results_path.unlink(missing_ok=True)
    write_artifact_manifest(args.output_dir)
    print(summary.to_json(orient="records"), flush=True)


if __name__ == "__main__":
    main()
