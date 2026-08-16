#!/usr/bin/env python3
"""Prepare and reproduce the E0-2 4g QGeoGNN baseline.

The script uses the repository CSV as the sole source, retains repeated
experiments, applies the legacy 4g volume limits, builds deterministic RDKit
3D graphs, freezes a row-level split, fits feature scalers on the training
split only, and saves the best validation checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qgeognn_graphs import build_graph_and_descriptor, qg  # noqa: E402


EXPECTED_SPEC = "Silica-CS 4g"
V1_LIMIT_ML = 60.0
V2_LIMIT_ML = 120.0
ELUENT_SMILES = ["CCCCCC", "CC(OCC)=O", "C(Cl)Cl", "CO", "CCOCC"]
LOADING_SOLVENT = {"PE": 0.0, "EA": 1.0, "DCM": 2.0}
RDLogger.DisableLog("rdApp.warning")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_environment(output_dir: Path) -> None:
    packages = [
        "numpy",
        "pandas",
        "rdkit",
        "scikit-learn",
        "scipy",
        "torch",
        "torch-geometric",
        "mordred",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_artifact_manifest(output_dir: Path) -> None:
    manifest_path = output_dir / "artifact_manifest.json"
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest_path.write_text(
        json.dumps({"artifacts": artifacts}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def stable_sample_id(source_hash: str, source_row_1based: int) -> str:
    value = f"4g:{source_hash}:{source_row_1based}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:20]


def ratio_valid(value: object) -> bool:
    try:
        left, right = str(value).strip().split("/")
        left_value, right_value = float(left), float(right)
        return left_value >= 0 and right_value >= 0 and left_value + right_value > 0
    except (TypeError, ValueError):
        return False


def eluent_descriptor(value: str) -> np.ndarray:
    left, right = (float(part) for part in value.split("/"))
    weights = np.array([left, right, 0.0, 0.0, 0.0], dtype=np.float32)
    weights /= weights.sum()
    result = np.zeros(6, dtype=np.float32)
    for smiles, weight in zip(ELUENT_SMILES, weights):
        if weight == 0:
            continue
        mol = Chem.MolFromSmiles(smiles)
        result += weight * np.array(
            [
                Descriptors.ExactMolWt(mol),
                Chem.rdMolDescriptors.CalcTPSA(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.MolLogP(mol),
            ],
            dtype=np.float32,
        )
    return result


def minmax_fit(values: np.ndarray) -> dict[str, list[float]]:
    return {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist()}


def minmax_transform(values: np.ndarray, scaler: dict[str, list[float]]) -> np.ndarray:
    minimum = np.asarray(scaler["min"], dtype=np.float32)
    maximum = np.asarray(scaler["max"], dtype=np.float32)
    return (values - minimum) / (maximum - minimum + 1e-8)


def prepare_data(
    source_path: Path,
    output_dir: Path,
    seed: int,
    conformer_policy: str = "first_embedded",
) -> tuple[pd.DataFrame, dict]:
    source_hash = sha256_file(source_path)
    df = pd.read_csv(source_path).copy()
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
    numeric = {column: pd.to_numeric(df[column], errors="coerce") for column in numeric_columns}
    v1 = numeric["t1"] * numeric["Flow mL/min"] / 1200.0
    v2 = numeric["t2"] * numeric["Flow mL/min"] / 1200.0
    df["V1_ml"] = v1
    df["V2_ml"] = v2

    decisions: list[list[str]] = [[] for _ in range(len(df))]
    quality_flags: list[list[str]] = [[] for _ in range(len(df))]

    def flag(target: list[list[str]], mask: pd.Series, reason: str) -> None:
        for index in df.index[mask.fillna(False)]:
            target[int(index)].append(reason)

    flag(decisions, ~df["column_specs"].eq(EXPECTED_SPEC), "unexpected_column_spec")
    flag(decisions, numeric["t1"].isna() | numeric["t1"].eq(-1), "reader_rejects_t1")
    core_missing = (
        df[["CAS", "loading solvent", "PE/EA", "column_specs", "smiles"]].isna().any(axis=1)
        | pd.concat([numeric[column].isna() for column in numeric_columns], axis=1).any(axis=1)
    )
    flag(quality_flags, core_missing, "missing_core")
    flag(quality_flags, numeric["t1"].lt(0) | numeric["t2"].lt(0), "negative_label")
    flag(quality_flags, numeric["t1"].gt(numeric["t2"]), "t1_gt_t2")
    flag(quality_flags, ~df["PE/EA"].map(ratio_valid), "invalid_ratio")
    smiles_parse = df["smiles"].map(
        lambda value: pd.notna(value) and bool(str(value).strip()) and Chem.MolFromSmiles(str(value)) is not None
    )
    flag(decisions, ~smiles_parse, "invalid_smiles")
    flag(decisions, v1.gt(V1_LIMIT_ML), "over_v1_legacy_threshold")
    flag(decisions, v2.gt(V2_LIMIT_ML), "over_v2_legacy_threshold")

    pre_graph_keep = pd.Series([not reasons for reasons in decisions], index=df.index)
    unique_smiles = sorted(df.loc[pre_graph_keep, "smiles"].astype(str).unique())
    graph_cache: dict[str, dict] = {}
    graph_audit: list[dict] = []
    cache_path = output_dir / "graph_cache_4g.pt"
    audit_path = output_dir / "graph_audit_4g.csv"
    if cache_path.exists():
        graph_cache = torch.load(cache_path, weights_only=False)
    previous_audit: dict[str, dict] = {}
    if audit_path.exists():
        previous_audit = {
            row["canonical_smiles"]: row
            for row in pd.read_csv(audit_path).to_dict(orient="records")
            if row.get("status") == "success"
        }
    for index, smiles in enumerate(unique_smiles):
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
        if canonical in graph_cache and canonical in previous_audit:
            graph_audit.append(previous_audit[canonical])
            continue
        start = time.time()
        try:
            graph, descriptor, conformer = build_graph_and_descriptor(
                smiles, seed + index, conformer_policy
            )
            graph_cache[canonical] = {"graph": graph, "descriptor": descriptor}
            status, error = "success", ""
        except Exception as exc:  # graph failures must be recorded, not hidden
            status, error, conformer = "failed", f"{type(exc).__name__}: {exc}", {}
        graph_audit.append(
            {
                "canonical_smiles": canonical,
                "source_smiles": smiles,
                "status": status,
                **conformer,
                "error": error,
                "seconds": round(time.time() - start, 4),
            }
        )
        pd.DataFrame(graph_audit).to_csv(audit_path, index=False)
        torch.save(graph_cache, cache_path)

    failed_smiles = {row["canonical_smiles"] for row in graph_audit if row["status"] != "success"}
    for index in df.index[pre_graph_keep]:
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(str(df.at[index, "smiles"])), canonical=True)
        if canonical in failed_smiles:
            decisions[int(index)].append("graph_construction_failed")

    df["decision"] = ["keep" if not reasons else "drop" for reasons in decisions]
    df["decision_reason"] = ["" if not reasons else ";".join(sorted(set(reasons))) for reasons in decisions]
    df["quality_flags"] = ["" if not flags else ";".join(sorted(set(flags))) for flags in quality_flags]
    df["reviewer"] = "user_confirmed_protocol_2026-08-13"
    canonical_df = df.loc[df["decision"].eq("keep")].copy().reset_index(drop=True)
    canonical_df["canonical_smiles"] = canonical_df["smiles"].map(
        lambda value: Chem.MolToSmiles(Chem.MolFromSmiles(str(value)), canonical=True)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "sample_decisions_4g.csv", index=False)
    canonical_df.to_csv(output_dir / "canonical_4g.csv", index=False)
    pd.DataFrame(graph_audit).to_csv(audit_path, index=False)
    torch.save(graph_cache, cache_path)

    metadata = {
        "source_file": str(source_path.relative_to(ROOT)),
        "source_sha256": source_hash,
        "source_rows": int(len(df)),
        "reader_compatible_rows": int(
            (df["column_specs"].eq(EXPECTED_SPEC) & numeric["t1"].ne(-1) & numeric["t1"].notna()).sum()
        ),
        "canonical_rows": int(len(canonical_df)),
        "unique_canonical_smiles": int(canonical_df["canonical_smiles"].nunique()),
        "graph_success_smiles": int(sum(row["status"] == "success" for row in graph_audit)),
        "graph_failed_smiles": int(sum(row["status"] != "success" for row in graph_audit)),
        "repeated_experiments_policy": "retain",
        "baseline_data_policy": "reproduce current reader and Construct_dataset legacy thresholds",
        "quality_warnings_retained": {
            "negative_label_rows": int(canonical_df["quality_flags"].str.contains("negative_label", na=False).sum()),
            "t1_gt_t2_rows": int(canonical_df["quality_flags"].str.contains("t1_gt_t2", na=False).sum()),
        },
        "volume_formula": "V_ml = t_raw * Flow_mL_min / 1200",
        "volume_thresholds_ml": {"V1": V1_LIMIT_ML, "V2": V2_LIMIT_ML},
        "rdkit_seed": seed,
        "conformer_policy": conformer_policy,
    }
    (output_dir / "data_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return canonical_df, graph_cache


def make_split(canonical_df: pd.DataFrame, output_dir: Path, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(len(canonical_df))
    train_end = int(0.8 * len(permutation))
    valid_end = train_end + int(0.1 * len(permutation))
    split = np.full(len(permutation), "test", dtype=object)
    split[permutation[:train_end]] = "train"
    split[permutation[train_end:valid_end]] = "valid"
    split_df = canonical_df[["sample_id", "source_row_1based"]].copy()
    split_df["canonical_index"] = np.arange(len(canonical_df), dtype=int)
    split_df["split"] = split
    split_df["seed"] = seed
    split_df.to_csv(output_dir / f"split_seed_{seed}.csv", index=False)
    return split_df


def make_compound_split(canonical_df: pd.DataFrame, output_dir: Path, seed: int) -> pd.DataFrame:
    """Save a comparison split without placing one compound in multiple subsets."""
    group_sizes = canonical_df.groupby("canonical_smiles", sort=True).size()
    groups = group_sizes.index.to_numpy(copy=True)
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    cumulative = np.cumsum([int(group_sizes[group]) for group in groups])
    train_cut = int(np.argmin(np.abs(cumulative - 0.8 * len(canonical_df)))) + 1
    valid_cut = int(np.argmin(np.abs(cumulative - 0.9 * len(canonical_df)))) + 1
    valid_cut = max(valid_cut, train_cut + 1)
    group_split = {
        group: "train" if index < train_cut else "valid" if index < valid_cut else "test"
        for index, group in enumerate(groups)
    }
    split_df = canonical_df[
        ["sample_id", "source_row_1based", "canonical_smiles"]
    ].copy()
    split_df["canonical_index"] = np.arange(len(canonical_df), dtype=int)
    split_df["split"] = split_df["canonical_smiles"].map(group_split)
    split_df["seed"] = seed
    split_df.to_csv(output_dir / f"compound_group_split_seed_{seed}.csv", index=False)
    return split_df


def build_model_data(
    canonical_df: pd.DataFrame,
    graph_cache: dict,
    split_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[list[Data], list[Data], dict]:
    descriptors = np.vstack(
        [graph_cache[smiles]["descriptor"] for smiles in canonical_df["canonical_smiles"]]
    ).astype(np.float32)
    eluents = np.vstack([eluent_descriptor(value) for value in canonical_df["PE/EA"]]).astype(np.float32)
    train_mask = split_df["split"].eq("train").to_numpy()
    scaler = {
        "fit_split": "train",
        "descriptor": minmax_fit(descriptors[train_mask]),
        "eluent": minmax_fit(eluents[train_mask]),
    }
    (output_dir / "scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")
    descriptors = minmax_transform(descriptors, scaler["descriptor"]).astype(np.float32)
    eluents = minmax_transform(eluents, scaler["eluent"]).astype(np.float32)

    atom_bond: list[Data] = []
    bond_angle: list[Data] = []
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
        loading_code = LOADING_SOLVENT[str(row["loading solvent"])]
        mass = float(row["Density g/ml"]) * float(row["V/ul"])
        extra = torch.tensor(
            [loading_code, mass, float(row["Volume of loading solvent/ul"])], dtype=torch.float32
        ).repeat(edge_count, 1)
        edge_attr = torch.cat([bond_ids, bond_length, prop, extra], dim=1)
        y = torch.tensor([[float(row["V1_ml"]), float(row["V2_ml"])]], dtype=torch.float32)
        atom_bond.append(
            Data(
                x=atom_feature,
                edge_index=torch.from_numpy(graph["edges"].T).to(torch.int64),
                edge_attr=edge_attr,
                y=y,
                data_index=torch.tensor(index, dtype=torch.int64),
            )
        )
        angle = torch.from_numpy(graph["bond_angle"].astype(np.float32)).reshape(-1, 1)
        angle_descriptor = torch.tensor(descriptors[index], dtype=torch.float32).repeat(angle.shape[0], 1)
        bond_angle.append(
            Data(
                edge_index=torch.from_numpy(graph["BondAngleGraph_edges"].T).to(torch.int64),
                edge_attr=torch.cat([angle, angle_descriptor], dim=1),
            )
        )
    return atom_bond, bond_angle, scaler


def evaluate(model, loader_atom, loader_angle, device: torch.device) -> dict:
    model.eval()
    true_values, predictions = [], []
    with torch.no_grad():
        for atom_batch, angle_batch in zip(loader_atom, loader_angle):
            atom_batch, angle_batch = atom_batch.to(device), angle_batch.to(device)
            pred = model(atom_batch, angle_batch)[0]
            true_values.append(atom_batch.y.cpu().numpy())
            predictions.append(pred.cpu().numpy())
    y_true = np.vstack(true_values)
    pred = np.vstack(predictions)
    q50 = pred[:, [1, 4]]
    result: dict[str, float] = {}
    for output_index, name in enumerate(["V1", "V2"]):
        residual = y_true[:, output_index] - q50[:, output_index]
        ss_res = float(np.square(residual).sum())
        ss_tot = float(np.square(y_true[:, output_index] - y_true[:, output_index].mean()).sum())
        result[f"{name}_mae"] = float(np.abs(residual).mean())
        result[f"{name}_rmse"] = float(np.sqrt(np.square(residual).mean()))
        result[f"{name}_r2"] = float(1.0 - ss_res / ss_tot) if ss_tot else float("nan")
        quantiles = pred[:, output_index * 3 : output_index * 3 + 3]
        errors = y_true[:, [output_index]] - quantiles
        levels = np.array([0.1, 0.5, 0.9], dtype=np.float32).reshape(1, -1)
        result[f"{name}_mean_pinball_loss"] = float(
            np.maximum(levels * errors, (levels - 1.0) * errors).mean()
        )
        result[f"{name}_interval_80_coverage"] = float(
            np.mean(
                (y_true[:, output_index] >= quantiles[:, 0])
                & (y_true[:, output_index] <= quantiles[:, 2])
            )
        )
        result[f"{name}_interval_80_mean_width"] = float(
            np.mean(quantiles[:, 2] - quantiles[:, 0])
        )
    result["quantile_crossing_rate"] = float(
        np.mean((pred[:, 0] > pred[:, 1]) | (pred[:, 1] > pred[:, 2]) | (pred[:, 3] > pred[:, 4]) | (pred[:, 4] > pred[:, 5]))
    )
    return result


def evaluate_checkpoint(
    canonical_df: pd.DataFrame,
    graph_cache: dict,
    split_df: pd.DataFrame,
    output_dir: Path,
    batch_size: int,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qg.device = str(device)
    atom_data, angle_data, _ = build_model_data(
        canonical_df, graph_cache, split_df, output_dir
    )
    loaders = {}
    for split in ["valid", "test"]:
        indices = split_df.index[split_df["split"].eq(split)].tolist()
        loaders[split] = (
            DataLoader([atom_data[i] for i in indices], batch_size=batch_size, shuffle=False),
            DataLoader([angle_data[i] for i in indices], batch_size=batch_size, shuffle=False),
        )
    model = qg.GINGraphPooling(
        num_tasks=6,
        num_layers=5,
        emb_dim=128,
        drop_ratio=0.0,
        graph_pooling="sum",
        descriptor_dim=1827,
    ).to(device)
    checkpoint = torch.load(
        output_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    previous = {}
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = {
        "best_epoch": int(checkpoint["epoch"]),
        "best_valid_score": float(checkpoint["valid_score"]),
        "epochs_run": previous.get("epochs_run"),
        "device": str(device),
        "valid": evaluate(model, *loaders["valid"], device),
        "test": evaluate(model, *loaders["test"], device),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def train_baseline(
    canonical_df: pd.DataFrame,
    graph_cache: dict,
    split_df: pd.DataFrame,
    output_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    verbose: bool = True,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qg.device = str(device)
    atom_data, angle_data, scaler = build_model_data(canonical_df, graph_cache, split_df, output_dir)

    loaders = {}
    for split in ["train", "valid", "test"]:
        indices = split_df.index[split_df["split"].eq(split)].tolist()
        loaders[split] = (
            DataLoader([atom_data[i] for i in indices], batch_size=batch_size, shuffle=False),
            DataLoader([angle_data[i] for i in indices], batch_size=batch_size, shuffle=False),
        )

    model = qg.GINGraphPooling(
        num_tasks=6,
        num_layers=5,
        emb_dim=128,
        drop_ratio=0.0,
        graph_pooling="sum",
        descriptor_dim=1827,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_valid = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, batches = 0.0, 0
        for atom_batch, angle_batch in zip(*loaders["train"]):
            atom_batch, angle_batch = atom_batch.to(device), angle_batch.to(device)
            pred = model(atom_batch, angle_batch)[0]
            true_1, true_2 = atom_batch.y[:, 0], atom_batch.y[:, 1]
            loss_1 = (
                qg.q_loss(0.1, true_1, pred[:, 0])
                + torch.mean((true_1 - pred[:, 1]) ** 2)
                + qg.q_loss(0.9, true_1, pred[:, 2])
                + torch.mean(torch.relu(pred[:, 0] - pred[:, 1]))
                + torch.mean(torch.relu(pred[:, 1] - pred[:, 2]))
            )
            loss_2 = (
                qg.q_loss(0.1, true_2, pred[:, 3])
                + torch.mean((true_2 - pred[:, 4]) ** 2)
                + qg.q_loss(0.9, true_2, pred[:, 5])
                + torch.mean(torch.relu(pred[:, 3] - pred[:, 4]))
                + torch.mean(torch.relu(pred[:, 4] - pred[:, 5]))
            )
            loss = loss_1 + 0.5 * loss_2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1

        valid_metrics = evaluate(model, *loaders["valid"], device)
        valid_score = valid_metrics["V1_rmse"] ** 2 + 0.5 * valid_metrics["V2_rmse"] ** 2
        record = {"epoch": epoch, "train_loss": total_loss / max(batches, 1), "valid_score": valid_score, **valid_metrics}
        history.append(record)
        if valid_score < best_valid:
            best_valid, best_epoch = valid_score, epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "valid_score": best_valid,
                    "seed": seed,
                    "source_sha256": sha256_file(ROOT / "dataset" / "dataset_4g.csv"),
                    "scaler": scaler,
                },
                checkpoint_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1
        if verbose:
            print(json.dumps(record, ensure_ascii=False), flush=True)
        if epochs_without_improvement >= patience:
            break

    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = {
        "best_epoch": best_epoch,
        "best_valid_score": best_valid,
        "epochs_run": len(history),
        "device": str(device),
        "valid": evaluate(model, *loaders["valid"], device),
        "test": evaluate(model, *loaders["test"], device),
    }
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "dataset" / "dataset_4g.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "e0_4g_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument(
        "--conformer-policy",
        choices=["first_embedded", "lowest_energy"],
        default="first_embedded",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_environment(args.output_dir)
    if args.reuse_cache:
        canonical_df = pd.read_csv(args.output_dir / "canonical_4g.csv")
        graph_cache = torch.load(args.output_dir / "graph_cache_4g.pt", weights_only=False)
    else:
        canonical_df, graph_cache = prepare_data(
            args.source, args.output_dir, args.seed, args.conformer_policy
        )
    split_df = make_split(canonical_df, args.output_dir, args.seed)
    compound_split_df = make_compound_split(canonical_df, args.output_dir, args.seed)
    config = {
        "stage": "E0-2_4g_baseline",
        "source": str(args.source.relative_to(ROOT)) if args.source.is_relative_to(ROOT) else str(args.source),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "thresholds_ml": {"V1": V1_LIMIT_ML, "V2": V2_LIMIT_ML},
        "repeated_experiments": "retain",
        "conformer_policy": args.conformer_policy,
        "split_counts": split_df["split"].value_counts().to_dict(),
        "compound_group_split_counts": compound_split_df["split"].value_counts().to_dict(),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if args.prepare_only:
        write_artifact_manifest(args.output_dir)
        print(json.dumps({"prepared": len(canonical_df), **config}, ensure_ascii=False))
        return
    if args.evaluate_only:
        metrics = evaluate_checkpoint(
            canonical_df, graph_cache, split_df, args.output_dir, args.batch_size
        )
        write_artifact_manifest(args.output_dir)
        print(json.dumps(metrics, ensure_ascii=False))
        return
    metrics = train_baseline(
        canonical_df,
        graph_cache,
        split_df,
        args.output_dir,
        args.seed,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.patience,
    )
    write_artifact_manifest(args.output_dir)
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
