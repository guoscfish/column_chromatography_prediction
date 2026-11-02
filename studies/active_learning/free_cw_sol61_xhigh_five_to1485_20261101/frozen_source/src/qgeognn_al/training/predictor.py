"""Shared source training primitives for the single standalone predictor API."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from ..evaluation.point import point_metrics
from ..data import build_model_data, eluent_descriptor, qg
from ..models import predictor_checkpoint
from ..schemas.conditions import fit_condition_normalization


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_source_preprocessing(data, split, graph_cache, scaler_path):
    joined = data[["sample_id"]].merge(split[["sample_id", "split"]], on="sample_id", how="left", validate="one_to_one")
    if len(joined) != len(data) or joined["split"].isna().any():
        raise ValueError("split must cover all source rows")
    positions = np.flatnonzero(joined["split"].eq("train"))
    train = data.iloc[positions]
    descriptors = np.vstack([graph_cache[s]["descriptor"] for s in train["canonical_smiles"]]).astype(np.float32)
    eluents = np.vstack([eluent_descriptor(v) for v in train["PE/EA"]]).astype(np.float32)
    scaler = {"descriptor": {"min": descriptors.min(0).tolist(), "max": descriptors.max(0).tolist()},
              "eluent": {"min": eluents.min(0).tolist(), "max": eluents.max(0).tolist()}}
    atomic_json(scaler_path, scaler)
    norm = fit_condition_normalization(data, split, Path(scaler_path))
    scales = {t: float(train[f"{t}_ml"].std(ddof=0)) for t in ("V1", "V2")}
    preprocessing = {"scaler": scaler, "target_scales": scales, "fit_role": "source_train",
                     "fit_rows": len(train), "fit_ids_hash": stable_hash(sorted(train.sample_id.astype(str))),
                     "validation_rows_used": 0, "test_rows_used": 0, "target_rows_used": 0}
    return norm, preprocessing


def loader_pair(atom_data, angle_data, indices, batch_size=2048):
    positions = [int(i) for i in indices]
    return (DataLoader([atom_data[i] for i in positions], batch_size=batch_size, shuffle=False),
            DataLoader([angle_data[i] for i in positions], batch_size=batch_size, shuffle=False))


def predict(model, atom, angle, indices):
    model.eval()
    predictions, truths, positions = [], [], []
    with torch.no_grad():
        for a, b in zip(*loader_pair(atom, angle, indices)):
            predictions.append(model(a, b).cpu().numpy())
            truths.append(a.y.cpu().numpy())
            positions.append(a.canonical_position.numpy().reshape(-1))
    return np.vstack(truths), np.vstack(predictions), np.concatenate(positions)


def target_loss(true, pred):
    return qg.q_loss(.1, true, pred[:, 0]) + torch.mean((true-pred[:, 1])**2) + qg.q_loss(.9, true, pred[:, 2]) + torch.mean(torch.relu(pred[:, 0]-pred[:, 1])) + torch.mean(torch.relu(pred[:, 1]-pred[:, 2]))


def train_source(model, atom, angle, train_indices, validation_indices, preprocessing, config, checkpoint_path, progress=None):
    """Test indices are deliberately absent from this fitting interface."""
    if set(train_indices) & set(validation_indices):
        raise ValueError("training/validation overlap")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    best, stale, best_epoch, history = float("inf"), 0, None, []
    for epoch in range(1, config["maximum_epochs"]+1):
        model.train()
        order = np.random.default_rng(config["seed"]*10000+epoch).permutation(train_indices)
        losses = []
        for a, b in zip(*loader_pair(atom, angle, order, config["batch_size"])):
            prediction = model(a, b)
            loss = target_loss(a.y[:, 0], prediction[:, :3]) + target_loss(a.y[:, 1], prediction[:, 3:])
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite source training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        truth, prediction, _ = predict(model, atom, angle, validation_indices)
        metrics = point_metrics(truth, prediction, preprocessing["target_scales"])
        score = metrics["combined_normalized_rmse"]
        if not math.isfinite(score):
            raise RuntimeError("non-finite validation score")
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_selection_score": score,
                        **{f"validation_{k}": v for k, v in metrics.items()}})
        if score < best:
            best, best_epoch, stale = score, epoch, 0
            torch.save(predictor_checkpoint(model, preprocessing=preprocessing, training_config=config,
                       provenance={"epoch": epoch, "validation_selection_score": score,
                                   "split_sha256": config["split_sha256"]}), checkpoint_path)
        else:
            stale += 1
        if progress and (epoch % 10 == 0 or epoch == 1):
            progress({"epoch": epoch, "best_epoch": best_epoch, "validation_score": score, "best_score": best})
        if stale >= config["patience"]:
            break
    return history, best_epoch
