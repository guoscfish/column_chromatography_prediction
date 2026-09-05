"""Frozen typed input semantics for the standalone QGeoGNN-V2 predictor."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import global_mean_pool

from ..artifacts import sha256_file
from ..input_schema import CATEGORICAL_BOND_FEATURES


MODEL_VARIANT = "qgeognn_v2"
HIDDEN_DIM = 16
SOLVENT_VOCABULARY = ("PE", "EA", "DCM")
SOLVENT_EMBEDDING_DIM = 4
MISSING_CONDITION_FEATURES = (
    "eluent_h_acceptors",
    "eluent_logp",
    "loading_solvent",
    "loading_amount_density_x_volume",
    "loading_solvent_volume_ul",
)
INTENDED_CONDITION_FEATURES = (
    "eluent_exact_mol_wt",
    "eluent_tpsa",
    "eluent_rotatable_bonds",
    "eluent_h_donors",
    *MISSING_CONDITION_FEATURES,
)


@dataclass(frozen=True)
class ConditionNormalization:
    loading_amount_min: float
    loading_amount_max: float
    loading_volume_min: float
    loading_volume_max: float
    fit_dataset: str
    fit_role: str
    fit_row_count: int
    fit_ids_hash: str
    eluent_scaler_sha256: str

    def validate(self) -> None:
        if self.fit_dataset != "4g" or self.fit_role != "source_train":
            raise ValueError("V2 normalization must be fit on 4g source_train only")
        if self.fit_row_count < 1 or len(self.fit_ids_hash) != 64:
            raise ValueError("V2 normalization requires auditable source-train IDs")
        if self.loading_amount_max <= self.loading_amount_min:
            raise ValueError("loading amount scaler range must be positive")
        if self.loading_volume_max <= self.loading_volume_min:
            raise ValueError("loading volume scaler range must be positive")


def fit_condition_normalization(
    canonical_data: pd.DataFrame,
    source_split: pd.DataFrame,
    eluent_scaler_path: Path,
) -> ConditionNormalization:
    roles = source_split[["sample_id", "split"]]
    joined = canonical_data.merge(roles, on="sample_id", how="left", validate="one_to_one")
    if joined["split"].isna().any():
        raise ValueError("source split does not cover all canonical rows")
    train = joined.loc[joined["split"].eq("train")].copy()
    ids = sorted(train["sample_id"].astype(str))
    ids_hash = hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode()
    ).hexdigest()
    amount = train["Density g/ml"].to_numpy(dtype=np.float32) * train["V/ul"].to_numpy(dtype=np.float32)
    volume = train["Volume of loading solvent/ul"].to_numpy(dtype=np.float32)
    return ConditionNormalization(
        loading_amount_min=float(amount.min()),
        loading_amount_max=float(amount.max()),
        loading_volume_min=float(volume.min()),
        loading_volume_max=float(volume.max()),
        fit_dataset="4g",
        fit_role="source_train",
        fit_row_count=len(train),
        fit_ids_hash=ids_hash,
        eluent_scaler_sha256=sha256_file(Path(eluent_scaler_path)),
    )


def condition_schema() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_variant": MODEL_VARIANT,
        "integration": "after_fixed_sum_pooling_before_prediction_head",
        "legacy_condition_features": list(INTENDED_CONDITION_FEATURES[:4]),
        "completion_condition_features": list(MISSING_CONDITION_FEATURES),
        "all_intended_condition_features": list(INTENDED_CONDITION_FEATURES),
        "loading_solvent": {
            "type": "categorical_embedding",
            "vocabulary": list(SOLVENT_VOCABULARY),
            "embedding_dim": SOLVENT_EMBEDDING_DIM,
        },
        "continuous_completion_features": {
            "names": [
                "eluent_h_acceptors",
                "eluent_logp",
                "loading_amount_density_x_volume",
                "loading_solvent_volume_ul",
            ],
            "dtype": "float32",
            "eluent_normalization": "reuse_frozen_source_train_eluent_minmax_dimensions_4_5",
            "loading_normalization": "source_train_only_minmax",
        },
        "condition_branch": {
            "input_dim": 8,
            "hidden_dim": HIDDEN_DIM,
            "output_dim": 128,
            "activation": "relu",
            "output_initialization": "zeros",
        },
        "column_specification_features": [],
    }


def input_schema_hash(schema: dict[str, Any] | None = None) -> str:
    payload = condition_schema() if schema is None else schema
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def condition_branch_config() -> dict[str, Any]:
    schema = condition_schema()
    return {
        "model_variant": MODEL_VARIANT,
        "completion_features": schema["completion_condition_features"],
        **schema["condition_branch"],
        "solvent_vocabulary": list(SOLVENT_VOCABULARY),
        "solvent_embedding_dim": SOLVENT_EMBEDDING_DIM,
    }


def condition_branch_config_hash() -> str:
    encoded = json.dumps(condition_branch_config(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class TypedConditionCompletionBranch(nn.Module):
    def __init__(self, normalization: ConditionNormalization):
        super().__init__()
        normalization.validate()
        self.normalization = normalization
        self.solvent_embedding = nn.Embedding(len(SOLVENT_VOCABULARY), SOLVENT_EMBEDDING_DIM)
        self.hidden = nn.Linear(SOLVENT_EMBEDDING_DIM + 4, HIDDEN_DIM)
        self.activation = nn.ReLU()
        self.output = nn.Linear(HIDDEN_DIM, 128)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer(
            "loading_min",
            torch.tensor([normalization.loading_amount_min, normalization.loading_volume_min], dtype=torch.float32),
        )
        self.register_buffer(
            "loading_range",
            torch.tensor([
                normalization.loading_amount_max - normalization.loading_amount_min,
                normalization.loading_volume_max - normalization.loading_volume_min,
            ], dtype=torch.float32),
        )

    def typed_inputs(self, batched_atom_bond: Any) -> torch.Tensor:
        edge_attr = batched_atom_bond.edge_attr
        edge_batch = batched_atom_bond.batch[batched_atom_bond.edge_index[0]]
        offset = len(CATEGORICAL_BOND_FEATURES)
        continuous = edge_attr[:, offset:]
        edge_values = torch.stack(
            [continuous[:, 5], continuous[:, 6], continuous[:, 8], continuous[:, 9]], dim=1
        ).to(torch.float32)
        graph_values = global_mean_pool(edge_values, edge_batch)
        graph_values[:, 2:] = (graph_values[:, 2:] - self.loading_min) / self.loading_range
        solvent_code = global_mean_pool(continuous[:, 7:8], edge_batch).round().to(torch.long).squeeze(1)
        if torch.any(solvent_code < 0) or torch.any(solvent_code >= len(SOLVENT_VOCABULARY)):
            raise ValueError("loading solvent code is outside the V2 categorical vocabulary")
        return torch.cat([graph_values, self.solvent_embedding(solvent_code)], dim=1)

    def forward(self, batched_atom_bond: Any) -> tuple[torch.Tensor, torch.Tensor]:
        typed = self.typed_inputs(batched_atom_bond)
        internal = self.activation(self.hidden(typed))
        return self.output(internal), internal


