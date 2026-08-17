#!/usr/bin/env python3
"""Shared QGeoGNN fit/predict and active-learning state primitives.

This module is the D28 engineering boundary used by E1/E2/E4.  Stable
``sample_id`` values are the primary identity.  ``canonical_index`` is retained
only as an auditable position in a frozen canonical table; filtered DataFrame
row numbers are never accepted as persistent identities.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader


ROOT = Path(__file__).resolve().parents[1]

from scripts.qgeognn_graphs import qg
from scripts.run_e0_4g_baseline import sha256_file
from scripts.run_e0_8g_transfer import (
    build_model,
    build_model_data,
    configure_trainable,
    metrics_from_arrays,
    quantile_target_loss,
    set_training_mode,
    validation_scores,
)
from scripts.run_g0_1_quantile_monotonicity import install_monotonic_head


BASE_BOND_FLOAT_NAMES = ["bond_length", "prop", "e", "m", "V_e"]
IDENTITY_COLUMNS = ["sample_id", "canonical_index", "source_canonical_index"]


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 500
    patience: int = 100
    batch_size: int = 2048
    transfer_mode: str = "last2_head"
    v1_weight: float = 1.0
    v2_weight: float = 1.0
    conformer_policy: str = "first_embedded"
    quantile_parameterization: str = "monotonic_softplus"
    scaler_policy: str = "source_train"
    checkpoint_selection: str = "validation_normalized_mse"

    def validate_frozen_predictor(self) -> None:
        expected = {
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "transfer_mode": "last2_head",
            "v1_weight": 1.0,
            "v2_weight": 1.0,
            "conformer_policy": "first_embedded",
            "quantile_parameterization": "monotonic_softplus",
            "scaler_policy": "source_train",
            "checkpoint_selection": "validation_normalized_mse",
        }
        actual = asdict(self)
        mismatches = {
            key: {"expected": value, "actual": actual[key]}
            for key, value in expected.items()
            if actual[key] != value
        }
        if mismatches:
            raise ValueError(f"TrainConfig changes frozen Gate 0 settings: {mismatches}")
        if self.epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("epochs, patience and batch_size must be positive")

    @property
    def config_hash(self) -> str:
        return canonical_json_hash(asdict(self))


@dataclass(frozen=True)
class SourceFreeTrainConfig(TrainConfig):
    """E2-only 4g contract that avoids using a full-data 4g checkpoint.

    With no source model there are no pretrained early layers to freeze, so all
    QGeoGNN parameters are trained from a seeded random initialization.  This
    is a separate protocol, not a change to the frozen 4g->8g transfer rule.
    """

    transfer_mode: str = "full_source_free"
    scaler_policy: str = "fixed_l0_train"
    initialization_policy: str = "seeded_random"

    def validate_frozen_predictor(self) -> None:
        expected = {
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "transfer_mode": "full_source_free",
            "initialization_policy": "seeded_random",
            "v1_weight": 1.0,
            "v2_weight": 1.0,
            "conformer_policy": "first_embedded",
            "quantile_parameterization": "monotonic_softplus",
            "scaler_policy": "fixed_l0_train",
            "checkpoint_selection": "validation_normalized_mse",
        }
        actual = asdict(self)
        mismatches = {
            key: {"expected": value, "actual": actual[key]}
            for key, value in expected.items()
            if actual[key] != value
        }
        if mismatches:
            raise ValueError(f"SourceFreeTrainConfig changes the E2 contract: {mismatches}")
        if self.epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("epochs, patience and batch_size must be positive")


@dataclass
class FitResult:
    checkpoint: str
    checkpoint_sha256: str
    best_epoch: int
    epochs_run: int
    normalized_valid_score: float
    train_rows: int
    validation_rows: int
    trainable_parameters: int
    total_parameters: int
    train_config_hash: str
    labeled_ids_hash: str
    validation_ids_hash: str
    initialization_policy: str
    trainable_scope: str
    scaler_hash: str


@dataclass
class PredictionResult:
    table: pd.DataFrame
    embeddings: np.ndarray | None


class QGeoGNNActiveLearningEngine:
    """Frozen QGeoGNN data/model adapter with stable identity handling."""

    def __init__(
        self,
        canonical_df: pd.DataFrame,
        graph_cache: dict,
        scaler: dict,
        init_checkpoint: Path,
        device: torch.device | None = None,
    ) -> None:
        required = {"sample_id", "canonical_smiles", "V1_ml", "V2_ml"}
        missing = required - set(canonical_df.columns)
        if missing:
            raise ValueError(f"Canonical data is missing required columns: {sorted(missing)}")
        data = canonical_df.copy().reset_index(drop=True)
        if data["sample_id"].isna().any() or data["sample_id"].duplicated().any():
            raise ValueError("sample_id must be non-null and globally unique")
        if "canonical_index" in data:
            expected = np.arange(len(data), dtype=int)
            if not np.array_equal(data["canonical_index"].to_numpy(dtype=int), expected):
                raise ValueError("Existing canonical_index must equal the frozen canonical-table order")
        else:
            data["canonical_index"] = np.arange(len(data), dtype=int)
        absent_graphs = sorted(set(data["canonical_smiles"]) - set(graph_cache))
        if absent_graphs:
            raise ValueError(f"Graph cache is missing {len(absent_graphs)} structures")

        self.data = data
        self.graph_cache = graph_cache
        self.scaler = scaler
        self.init_checkpoint = Path(init_checkpoint).resolve()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._sample_to_index = dict(zip(data["sample_id"].astype(str), data["canonical_index"]))
        qg.bond_float_names = list(BASE_BOND_FLOAT_NAMES)
        split_stub = pd.DataFrame(
            {
                "sample_id": data["sample_id"],
                "canonical_index": data["canonical_index"],
                "split": "pool",
            }
        )
        self.atom_data, self.angle_data = build_model_data(
            data, graph_cache, split_stub, scaler
        )

    def _resolve_ids(self, identities: Sequence[str | int]) -> list[int]:
        if not identities:
            raise ValueError("At least one identity is required")
        resolved: list[int] = []
        for identity in identities:
            if isinstance(identity, str):
                if identity not in self._sample_to_index:
                    raise KeyError(f"Unknown sample_id: {identity}")
                resolved.append(int(self._sample_to_index[identity]))
            elif isinstance(identity, (int, np.integer)):
                index = int(identity)
                if index < 0 or index >= len(self.data):
                    raise IndexError(index)
                resolved.append(index)
            else:
                raise TypeError(f"Identity must be sample_id or canonical_index, got {type(identity)}")
        if len(resolved) != len(set(resolved)):
            raise ValueError("Resolved identities contain duplicates")
        return resolved

    def candidate_table(self, identities: Sequence[str | int]) -> pd.DataFrame:
        indices = self._resolve_ids(identities)
        rows = self.data.iloc[indices][["sample_id", "canonical_index"]].copy()
        rows["source_canonical_index"] = rows["canonical_index"].astype(int)
        return rows.reset_index(drop=True)

    def _validate_candidate_table(self, candidates: pd.DataFrame) -> pd.DataFrame:
        missing = set(IDENTITY_COLUMNS) - set(candidates.columns)
        if missing:
            raise ValueError(f"Candidate table is missing identity columns: {sorted(missing)}")
        result = candidates[IDENTITY_COLUMNS].copy().reset_index(drop=True)
        if result["sample_id"].isna().any() or result["sample_id"].duplicated().any():
            raise ValueError("Candidate sample_id values must be unique")
        if result["canonical_index"].isna().any() or result["canonical_index"].duplicated().any():
            raise ValueError("Candidate canonical_index values must be unique")
        source = result["source_canonical_index"].to_numpy(dtype=int)
        if np.any(source < 0) or np.any(source >= len(self.data)):
            raise ValueError("source_canonical_index is outside the frozen canonical table")
        result["sample_id"] = result["sample_id"].astype(str)
        result["canonical_index"] = result["canonical_index"].astype(int)
        result["source_canonical_index"] = source
        return result

    def _load_model(self, checkpoint_path: Path) -> torch.nn.Module:
        qg.bond_float_names = list(BASE_BOND_FLOAT_NAMES)
        model = build_model(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state = checkpoint["model_state_dict"]
        monotonic = any(key.startswith("graph_pred_linear.linear.") for key in state)
        if monotonic:
            install_monotonic_head(model)
            model.load_state_dict(state)
        else:
            model.load_state_dict(state)
            install_monotonic_head(model)
        return model

    def _loaders_for_indices(
        self, indices: Sequence[int], batch_size: int, include_request_position: bool = False
    ) -> tuple[DataLoader, DataLoader]:
        atoms, angles = [], []
        for request_position, source_index in enumerate(indices):
            atom = self.atom_data[int(source_index)].clone()
            if include_request_position:
                atom.request_position = torch.tensor(request_position, dtype=torch.int64)
            atoms.append(atom)
            angles.append(self.angle_data[int(source_index)].clone())
        return (
            DataLoader(atoms, batch_size=batch_size, shuffle=False),
            DataLoader(angles, batch_size=batch_size, shuffle=False),
        )

    def fit(
        self,
        labeled_indices: Sequence[str | int],
        validation_indices: Sequence[str | int],
        train_config: TrainConfig | SourceFreeTrainConfig,
        init_checkpoint: Path | None,
        seed: int,
        output_dir: Path,
    ) -> FitResult:
        """Retrain from an anchor using a labeled set with a fixed validation subset.

        ``labeled_indices`` contains the full currently labeled budget.
        ``validation_indices`` must be a subset and is excluded from gradient
        updates while remaining counted in the label budget.
        """

        train_config.validate_frozen_predictor()
        all_labeled = self._resolve_ids(labeled_indices)
        validation = self._resolve_ids(validation_indices)
        if not set(validation).issubset(all_labeled):
            raise ValueError("validation_indices must be a subset of labeled_indices")
        train = [index for index in all_labeled if index not in set(validation)]
        if not train or len(validation) < 2:
            raise ValueError("fit requires at least one training row and two validation rows")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        source_free = isinstance(train_config, SourceFreeTrainConfig)
        if source_free:
            anchor = None
            model = build_model(self.device)
            install_monotonic_head(model)
            for parameter in model.parameters():
                parameter.requires_grad = True
            trainable = sum(parameter.numel() for parameter in model.parameters())
            total = trainable
            initialization_policy = train_config.initialization_policy
            trainable_scope = train_config.transfer_mode
        else:
            anchor = Path(init_checkpoint or self.init_checkpoint).resolve()
            model = self._load_model(anchor)
            trainable, total = configure_trainable(model, "last2_head")
            initialization_policy = "checkpoint"
            trainable_scope = "last2_head"
        train_loaders = self._loaders_for_indices(train, train_config.batch_size)
        validation_loaders = self._loaders_for_indices(validation, train_config.batch_size)
        train_labels = self.data.iloc[train][["V1_ml", "V2_ml"]]
        target_variance = {
            target: max(float(train_labels[f"{target}_ml"].var(ddof=0)), 1e-8)
            for target in ("V1", "V2")
        }
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "best.pt"
        history_path = output_dir / "history.csv"
        best_score, best_epoch, stale = float("inf"), 0, 0
        history: list[dict[str, float | int]] = []
        qg.bond_float_names = list(BASE_BOND_FLOAT_NAMES)

        for epoch in range(1, train_config.epochs + 1):
            set_training_mode(model)
            total_loss, batches = 0.0, 0
            for atom_batch, angle_batch in zip(*train_loaders):
                atom_batch = atom_batch.to(self.device)
                angle_batch = angle_batch.to(self.device)
                pred, _ = model(atom_batch, angle_batch)
                loss_v1 = quantile_target_loss(atom_batch.y[:, 0], pred[:, 0:3])
                loss_v2 = quantile_target_loss(atom_batch.y[:, 1], pred[:, 3:6])
                loss = train_config.v1_weight * loss_v1 + train_config.v2_weight * loss_v2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                batches += 1

            valid_metrics = self._evaluate_model(model, validation_loaders)
            score, legacy_score = validation_scores(valid_metrics, target_variance)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total_loss / max(batches, 1),
                    "normalized_valid_score": score,
                    "legacy_valid_score": legacy_score,
                }
            )
            if score < best_score:
                best_score, best_epoch, stale = score, epoch, 0
                temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp-{os.getpid()}")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "normalized_valid_score": score,
                        "seed": seed,
                        "train_config": asdict(train_config),
                        "train_config_hash": train_config.config_hash,
                        "scaler_hash": canonical_json_hash(self.scaler),
                        "init_checkpoint": str(anchor) if anchor is not None else None,
                        "init_checkpoint_sha256": sha256_file(anchor) if anchor is not None else None,
                        "initialization_policy": initialization_policy,
                        "trainable_scope": trainable_scope,
                        "labeled_ids": self.data.iloc[all_labeled]["sample_id"].tolist(),
                        "validation_ids": self.data.iloc[validation]["sample_id"].tolist(),
                        "test_used_for_selection": False,
                    },
                    temporary,
                )
                temporary.replace(checkpoint_path)
            else:
                stale += 1
            if stale >= train_config.patience:
                break

        pd.DataFrame(history).to_csv(history_path, index=False)
        result = FitResult(
            checkpoint=str(checkpoint_path),
            checkpoint_sha256=sha256_file(checkpoint_path),
            best_epoch=best_epoch,
            epochs_run=len(history),
            normalized_valid_score=float(best_score),
            train_rows=len(train),
            validation_rows=len(validation),
            trainable_parameters=trainable,
            total_parameters=total,
            train_config_hash=train_config.config_hash,
            labeled_ids_hash=canonical_json_hash(
                sorted(self.data.iloc[all_labeled]["sample_id"].astype(str).tolist())
            ),
            validation_ids_hash=canonical_json_hash(
                sorted(self.data.iloc[validation]["sample_id"].astype(str).tolist())
            ),
            initialization_policy=initialization_policy,
            trainable_scope=trainable_scope,
            scaler_hash=canonical_json_hash(self.scaler),
        )
        atomic_write_json(output_dir / "fit_result.json", asdict(result))
        return result

    def _evaluate_model(
        self, model: torch.nn.Module, loaders: tuple[DataLoader, DataLoader]
    ) -> dict[str, float]:
        model.eval()
        true_values, predictions = [], []
        with torch.no_grad():
            for atom_batch, angle_batch in zip(*loaders):
                atom_batch = atom_batch.to(self.device)
                angle_batch = angle_batch.to(self.device)
                pred, _ = model(atom_batch, angle_batch)
                true_values.append(atom_batch.y.cpu().numpy())
                predictions.append(pred.cpu().numpy())
        return metrics_from_arrays(np.vstack(true_values), np.vstack(predictions))

    def predict(
        self,
        indices: Sequence[str | int] | pd.DataFrame,
        checkpoint: Path,
        return_quantiles: bool = True,
        return_embedding: bool = True,
        batch_size: int = 256,
        chunk_size: int = 1024,
    ) -> PredictionResult:
        if batch_size < 1 or chunk_size < 1:
            raise ValueError("batch_size and chunk_size must be positive")
        candidates = (
            self._validate_candidate_table(indices)
            if isinstance(indices, pd.DataFrame)
            else self.candidate_table(indices)
        )
        model = self._load_model(Path(checkpoint).resolve())
        model.eval()
        prediction_chunks, embedding_chunks, observed_positions = [], [], []
        qg.bond_float_names = list(BASE_BOND_FLOAT_NAMES)

        with torch.no_grad():
            for chunk_start in range(0, len(candidates), chunk_size):
                chunk = candidates.iloc[chunk_start : chunk_start + chunk_size]
                source_indices = chunk["source_canonical_index"].to_numpy(dtype=int).tolist()
                loaders = self._loaders_for_indices(
                    source_indices, batch_size, include_request_position=True
                )
                local_predictions, local_embeddings, local_positions = [], [], []
                for atom_batch, angle_batch in zip(*loaders):
                    atom_batch = atom_batch.to(self.device)
                    angle_batch = angle_batch.to(self.device)
                    pred, h_graph = model(atom_batch, angle_batch)
                    local_predictions.append(pred.cpu().numpy())
                    local_embeddings.append(h_graph.cpu().numpy())
                    local_positions.append(atom_batch.request_position.cpu().numpy().reshape(-1))
                prediction_chunks.append(np.vstack(local_predictions))
                embedding_chunks.append(np.vstack(local_embeddings))
                observed_positions.extend((np.concatenate(local_positions) + chunk_start).tolist())

        expected_positions = list(range(len(candidates)))
        if observed_positions != expected_positions:
            raise AssertionError("Inference output order drifted from the requested candidate order")
        pred = np.vstack(prediction_chunks)
        embeddings = np.vstack(embedding_chunks)
        if len(pred) != len(candidates) or len(embeddings) != len(candidates):
            raise AssertionError("Inference output count does not match candidate count")
        table = candidates.copy()
        table["V1_q50"] = pred[:, 1]
        table["V2_q50"] = pred[:, 4]
        if return_quantiles:
            for column, values in zip(
                ("V1_q10", "V1_q90", "V2_q10", "V2_q90"),
                (pred[:, 0], pred[:, 2], pred[:, 3], pred[:, 5]),
            ):
                table[column] = values
        table["checkpoint_sha256"] = sha256_file(Path(checkpoint).resolve())
        return PredictionResult(table=table, embeddings=embeddings if return_embedding else None)


@dataclass
class ActiveLearningState:
    version: int
    round: int
    labeled_ids: list[str]
    pool_ids: list[str]
    selected_ids: list[str]
    checkpoint: str
    seed: int
    rng_state: dict[str, Any]
    split_hash: str
    config_hash: str

    def validate(self) -> None:
        if self.version != 1 or self.round < 0:
            raise ValueError("Unsupported or invalid active-learning state")
        labeled, pool, selected = set(self.labeled_ids), set(self.pool_ids), set(self.selected_ids)
        if len(labeled) != len(self.labeled_ids) or len(pool) != len(self.pool_ids):
            raise ValueError("labeled_ids and pool_ids must be internally unique")
        if labeled & pool:
            raise ValueError("labeled_ids and pool_ids must be disjoint")
        if not selected.issubset(labeled) or selected & pool:
            raise ValueError("selected_ids must be newly labeled and absent from pool")
        if len(selected) != len(self.selected_ids):
            raise ValueError("selected_ids contains duplicates")
        if not self.split_hash or not self.config_hash:
            raise ValueError("split_hash and config_hash are mandatory")


def initialize_round_state(
    labeled_ids: Sequence[str],
    pool_ids: Sequence[str],
    checkpoint: str,
    seed: int,
    split_hash: str,
    config_hash: str,
) -> ActiveLearningState:
    rng = np.random.default_rng(seed)
    state = ActiveLearningState(
        version=1,
        round=0,
        labeled_ids=[str(value) for value in labeled_ids],
        pool_ids=[str(value) for value in pool_ids],
        selected_ids=[],
        checkpoint=str(checkpoint),
        seed=int(seed),
        rng_state=rng.bit_generator.state,
        split_hash=split_hash,
        config_hash=config_hash,
    )
    state.validate()
    return state


def random_query(
    state: ActiveLearningState, batch_size: int, checkpoint: str | None = None
) -> ActiveLearningState:
    state.validate()
    if batch_size < 1 or batch_size > len(state.pool_ids):
        raise ValueError("batch_size must be between 1 and the remaining pool size")
    rng = np.random.default_rng()
    rng.bit_generator.state = state.rng_state
    positions = rng.choice(len(state.pool_ids), size=batch_size, replace=False)
    selected = [state.pool_ids[int(position)] for position in positions]
    selected_set = set(selected)
    next_state = ActiveLearningState(
        version=1,
        round=state.round + 1,
        labeled_ids=state.labeled_ids + selected,
        pool_ids=[value for value in state.pool_ids if value not in selected_set],
        selected_ids=selected,
        checkpoint=str(checkpoint or state.checkpoint),
        seed=state.seed,
        rng_state=rng.bit_generator.state,
        split_hash=state.split_hash,
        config_hash=state.config_hash,
    )
    next_state.validate()
    return next_state


def save_round_state(path: Path, state: ActiveLearningState) -> None:
    state.validate()
    atomic_write_json(Path(path), asdict(state))


def load_round_state(
    path: Path, expected_split_hash: str | None = None, expected_config_hash: str | None = None
) -> ActiveLearningState:
    state = ActiveLearningState(**json.loads(Path(path).read_text(encoding="utf-8")))
    state.validate()
    if expected_split_hash is not None and state.split_hash != expected_split_hash:
        raise ValueError("Resume refused: split hash changed")
    if expected_config_hash is not None and state.config_hash != expected_config_hash:
        raise ValueError("Resume refused: config hash changed")
    return state
