"""HISTORICAL_NEGATIVE_RESULT: failed Clean architecture, reproduction only."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_add_pool

from ..data import qg
from ..model import quantile_target_loss
from .clean_schema import (
    CLEAN_MODEL_VARIANT,
    SOLVENT_VOCABULARY,
    CleanConditionBatch,
    CleanConditionNormalization,
    clean_condition_schema_hash,
    clean_input_schema,
    clean_input_schema_hash,
)


GRAPH_HIDDEN_DIM = 128
MOLECULAR_LATENT_DIM = 64
SOLVENT_EMBEDDING_DIM = 4
CONDITION_HIDDEN_DIM = 32
CONDITION_LATENT_DIM = 64
FUSION_DIM = MOLECULAR_LATENT_DIM + CONDITION_LATENT_DIM
NUM_LAYERS = 5
FUSION_MODES = {"full", "molecule_only", "condition_only", "condition_disabled"}


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def molecular_architecture_config() -> dict[str, Any]:
    return {
        "backbone": "clean_qgeognn_geometry",
        "num_layers": NUM_LAYERS,
        "hidden_dim": GRAPH_HIDDEN_DIM,
        "pooling": "sum",
        "bond_continuous_features": ["bond_length"],
        "bond_angle_features": ["bond_angle", "molecular_descriptors"],
        "projection": [GRAPH_HIDDEN_DIM, MOLECULAR_LATENT_DIM],
        "projection_normalization": "layer_norm",
        "message_passing_activation": "relu_except_final_node_layer",
        "registered_legacy_dead_modules": [],
    }


def condition_architecture_config() -> dict[str, Any]:
    return {
        "continuous_features": ["ea_fraction", "loading_mass_mg", "loading_solvent_volume_ul"],
        "continuous_dim": 3,
        "solvent_vocabulary": list(SOLVENT_VOCABULARY),
        "solvent_embedding_dim": SOLVENT_EMBEDDING_DIM,
        "hidden_dim": CONDITION_HIDDEN_DIM,
        "output_dim": CONDITION_LATENT_DIM,
        "activation": "gelu",
        "output_normalization": "layer_norm",
    }


def fusion_config() -> dict[str, Any]:
    return {
        "molecular_latent_dim": MOLECULAR_LATENT_DIM,
        "condition_latent_dim": CONDITION_LATENT_DIM,
        "operator": "concatenate",
        "fused_dim": FUSION_DIM,
    }


def quantile_head_config() -> dict[str, Any]:
    return {
        "targets": ["V1_ml", "V2_ml"],
        "quantiles": [0.1, 0.5, 0.9],
        "within_target_parameterization": "softplus_median_plus_softplus_offsets",
        "training_output_policy": "ordered; q10 may be negative",
        "evaluation_output_policy": "clamp_each_quantile_to_[0,1e8]",
        "cross_target_constraint": None,
        "configured_target_weights": {"V1": 1.0, "V2": 1.0},
    }


class CleanGeometryBackbone(nn.Module):
    """Geometry-aware QGeoGNN core without known unreachable modules."""

    def __init__(self, num_layers: int = NUM_LAYERS, hidden_dim: int = GRAPH_HIDDEN_DIM):
        super().__init__()
        if num_layers < 2:
            raise ValueError("Clean QGeoGNN requires at least two graph layers")
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.atom_encoder = qg.AtomEncoder(hidden_dim)
        self.initial_bond_encoder = qg.BondEncoder(hidden_dim)
        self.initial_bond_length_encoder = qg.BondFloatRBF(["bond_length"], hidden_dim)
        self.node_convs = nn.ModuleList(qg.GINConv(hidden_dim) for _ in range(num_layers))

        # The last edge update in legacy code cannot reach a subsequent node layer.
        # Clean-QGeoGNN registers only the num_layers - 1 reachable updates.
        update_count = num_layers - 1
        self.edge_bond_encoders = nn.ModuleList(qg.BondEncoder(hidden_dim) for _ in range(update_count))
        self.edge_length_encoders = nn.ModuleList(qg.BondFloatRBF(["bond_length"], hidden_dim) for _ in range(update_count))
        self.angle_encoders = nn.ModuleList(qg.BondAngleFloatRBF(qg.bond_angle_float_names, hidden_dim) for _ in range(update_count))
        self.edge_convs = nn.ModuleList(qg.GINConv(hidden_dim) for _ in range(update_count))

    @staticmethod
    def _bond_inputs(edge_attr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        categorical_count = len(qg.bond_id_names)
        if edge_attr.ndim != 2 or edge_attr.shape[1] <= categorical_count:
            raise ValueError("clean molecular edge_attr requires categorical bonds plus bond length")
        categorical = edge_attr[:, :categorical_count].to(torch.int64)
        bond_length = edge_attr[:, categorical_count : categorical_count + 1].to(torch.float32)
        if not torch.isfinite(bond_length).all():
            raise ValueError("bond length must be finite")
        return categorical, bond_length

    def forward(self, atom_bond: Any, bond_angle: Any) -> torch.Tensor:
        categorical, bond_length = self._bond_inputs(atom_bond.edge_attr)
        if bond_angle.edge_attr.ndim != 2 or bond_angle.edge_attr.shape[1] != len(qg.bond_angle_float_names):
            raise ValueError("clean bond-angle features do not match the molecular descriptor contract")
        if not torch.isfinite(bond_angle.edge_attr).all():
            raise ValueError("bond-angle geometry and descriptors must be finite")

        node = self.atom_encoder(atom_bond.x)
        edge = self.initial_bond_encoder(categorical) + self.initial_bond_length_encoder(bond_length)
        for layer, node_conv in enumerate(self.node_convs):
            node = node_conv(node, atom_bond.edge_index, edge)
            if layer == self.num_layers - 1:
                continue
            updated_bond = self.edge_bond_encoders[layer](categorical)
            updated_bond = updated_bond + self.edge_length_encoders[layer](bond_length)
            angle = self.angle_encoders[layer](bond_angle.edge_attr.to(torch.float32))
            edge = self.edge_convs[layer](updated_bond, bond_angle.edge_index, angle)
            node = F.relu(node)
            edge = F.relu(edge)
        return node


class CleanConditionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.solvent_embedding = nn.Embedding(len(SOLVENT_VOCABULARY), SOLVENT_EMBEDDING_DIM)
        self.hidden = nn.Linear(3 + SOLVENT_EMBEDDING_DIM, CONDITION_HIDDEN_DIM)
        self.output = nn.Linear(CONDITION_HIDDEN_DIM, CONDITION_LATENT_DIM)
        self.normalization = nn.LayerNorm(CONDITION_LATENT_DIM)

    def forward(self, conditions: CleanConditionBatch) -> torch.Tensor:
        conditions.validate()
        encoded_solvent = self.solvent_embedding(conditions.loading_solvent)
        typed = torch.cat([conditions.continuous, encoded_solvent], dim=1)
        return self.normalization(self.output(F.gelu(self.hidden(typed))))


class CleanMonotonicQuantileHead(nn.Module):
    def __init__(self, input_dim: int = FUSION_DIM):
        super().__init__()
        self.linear = nn.Linear(input_dim, 6)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        raw = self.linear(fused)
        outputs = []
        for offset in (0, 3):
            median = F.softplus(raw[:, offset])
            outputs.extend((
                median - F.softplus(raw[:, offset + 1]),
                median,
                median + F.softplus(raw[:, offset + 2]),
            ))
        return torch.stack(outputs, dim=1)


class CleanQGeoGNN(nn.Module):
    model_variant = CLEAN_MODEL_VARIANT

    def __init__(self, normalization: CleanConditionNormalization):
        super().__init__()
        normalization.validate()
        self.condition_normalization = normalization
        self.molecular_backbone = CleanGeometryBackbone()
        self.molecular_projection = nn.Sequential(
            nn.Linear(GRAPH_HIDDEN_DIM, MOLECULAR_LATENT_DIM),
            nn.LayerNorm(MOLECULAR_LATENT_DIM),
        )
        self.condition_encoder = CleanConditionEncoder()
        self.quantile_head = CleanMonotonicQuantileHead()

    def representations(
        self,
        atom_bond: Any,
        bond_angle: Any,
        conditions: CleanConditionBatch,
        *,
        mode: str = "full",
    ) -> dict[str, torch.Tensor]:
        if mode not in FUSION_MODES:
            raise ValueError(f"unknown clean fusion mode: {mode}")
        conditions.validate()
        graph_count = int(atom_bond.batch.max().item()) + 1
        if graph_count != conditions.continuous.shape[0]:
            raise ValueError("graph and condition batch sizes differ")
        nodes = self.molecular_backbone(atom_bond, bond_angle)
        molecular = self.molecular_projection(global_add_pool(nodes, atom_bond.batch))
        condition = self.condition_encoder(conditions)
        if mode in {"molecule_only", "condition_disabled"}:
            condition = torch.zeros_like(condition)
        elif mode == "condition_only":
            molecular = torch.zeros_like(molecular)
        fused = torch.cat([molecular, condition], dim=1)
        return {"molecular": molecular, "condition": condition, "fused": fused}

    def forward(
        self,
        atom_bond: Any,
        bond_angle: Any,
        conditions: CleanConditionBatch,
        *,
        mode: str = "full",
    ) -> torch.Tensor:
        latent = self.representations(atom_bond, bond_angle, conditions, mode=mode)
        output = self.quantile_head(latent["fused"])
        if self.training:
            return output
        return torch.clamp(output, min=0, max=1e8)


def build_clean_model(normalization: CleanConditionNormalization, device: torch.device) -> CleanQGeoGNN:
    qg.device = str(device)
    return CleanQGeoGNN(normalization).to(device)


def latent_l2_norms(representations: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        "molecular_latent_l2_mean": float(representations["molecular"].norm(dim=1).mean().detach()),
        "condition_latent_l2_mean": float(representations["condition"].norm(dim=1).mean().detach()),
    }


def module_gradient_norm(module: nn.Module) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared += parameter.grad.detach().to(torch.float64).square().sum().cpu()
    return float(torch.sqrt(squared))


def parameter_reachability(model: nn.Module) -> dict[str, Any]:
    rows = []
    for name, parameter in model.named_parameters():
        reachable = parameter.grad is not None
        rows.append({
            "parameter_name": name,
            "numel": parameter.numel(),
            "requires_grad": parameter.requires_grad,
            "gradient_reachable": reachable,
            "gradient_norm": None if parameter.grad is None else float(parameter.grad.detach().norm()),
        })
    unreachable = sum(row["numel"] for row in rows if row["requires_grad"] and not row["gradient_reachable"])
    gradient_bearing = sum(row["numel"] for row in rows if row["requires_grad"] and row["gradient_reachable"])
    return {
        "nominal_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "requires_grad_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "gradient_bearing_parameters": gradient_bearing,
        "forward_unreachable_trainable_parameters": unreachable,
        "parameters": rows,
    }


def per_target_gradient_contribution(
    model: CleanQGeoGNN,
    atom_bond: Any,
    bond_angle: Any,
    conditions: CleanConditionBatch,
    targets: torch.Tensor,
) -> dict[str, dict[str, float]]:
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("targets must have shape [batch, 2]")
    result: dict[str, dict[str, float]] = {}
    for target_index, target_name in enumerate(("V1", "V2")):
        model.zero_grad(set_to_none=True)
        prediction = model(atom_bond, bond_angle, conditions)
        start = target_index * 3
        loss = quantile_target_loss(targets[:, target_index], prediction[:, start : start + 3])
        loss.backward()
        result[target_name] = {
            "loss": float(loss.detach()),
            "molecular_projection_gradient_l2": module_gradient_norm(model.molecular_projection),
            "condition_encoder_gradient_l2": module_gradient_norm(model.condition_encoder),
            "all_parameter_gradient_l2": module_gradient_norm(model),
        }
    model.zero_grad(set_to_none=True)
    return result


def permute_conditions(conditions: CleanConditionBatch, permutation: Iterable[int]) -> CleanConditionBatch:
    indices = torch.as_tensor(list(permutation), dtype=torch.long, device=conditions.continuous.device)
    if indices.numel() != conditions.continuous.shape[0] or sorted(indices.cpu().tolist()) != list(range(indices.numel())):
        raise ValueError("condition permutation must contain each batch index exactly once")
    result = CleanConditionBatch(
        continuous=conditions.continuous[indices],
        loading_solvent=conditions.loading_solvent[indices],
        sample_ids=tuple(conditions.sample_ids[index] for index in indices.cpu().tolist()),
    )
    result.validate()
    return result


REQUIRED_CLEAN_CHECKPOINT_FIELDS = {
    "model_variant", "input_schema", "input_schema_hash", "condition_schema_hash",
    "normalization_statistics", "normalization_fit_ids_hash",
    "molecular_architecture_config", "condition_architecture_config", "fusion_config",
    "quantile_head_config", "nominal_parameter_count", "requires_grad_parameter_count",
    "gradient_bearing_parameter_count", "git_commit_sha", "source_split_hash",
    "training_config", "training_config_hash", "model_state_dict",
}


def clean_checkpoint_payload(
    model: CleanQGeoGNN,
    *,
    gradient_bearing_parameter_count: int,
    git_commit_sha: str,
    source_split_hash: str,
    training_config: dict[str, Any],
) -> dict[str, Any]:
    schema = clean_input_schema()
    return {
        "model_variant": CLEAN_MODEL_VARIANT,
        "input_schema": schema,
        "input_schema_hash": clean_input_schema_hash(schema),
        "condition_schema_hash": clean_condition_schema_hash(),
        "normalization_statistics": asdict(model.condition_normalization),
        "normalization_fit_ids_hash": model.condition_normalization.fit_ids_hash,
        "molecular_architecture_config": molecular_architecture_config(),
        "condition_architecture_config": condition_architecture_config(),
        "fusion_config": fusion_config(),
        "quantile_head_config": quantile_head_config(),
        "nominal_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "requires_grad_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "gradient_bearing_parameter_count": int(gradient_bearing_parameter_count),
        "git_commit_sha": str(git_commit_sha),
        "source_split_hash": str(source_split_hash),
        "training_config": training_config,
        "training_config_hash": _stable_hash(training_config),
        "model_state_dict": model.state_dict(),
    }


def validate_clean_checkpoint(checkpoint: dict[str, Any]) -> None:
    missing = REQUIRED_CLEAN_CHECKPOINT_FIELDS - set(checkpoint)
    if missing:
        raise ValueError(f"clean checkpoint missing required fields: {sorted(missing)}")
    if checkpoint["model_variant"] != CLEAN_MODEL_VARIANT:
        raise ValueError("clean checkpoint model_variant mismatch")
    if checkpoint["input_schema_hash"] != clean_input_schema_hash(checkpoint["input_schema"]):
        raise ValueError("clean checkpoint input_schema_hash mismatch")
    if checkpoint["condition_schema_hash"] != clean_condition_schema_hash():
        raise ValueError("clean checkpoint condition_schema_hash mismatch")
    expected_configs = {
        "molecular_architecture_config": molecular_architecture_config(),
        "condition_architecture_config": condition_architecture_config(),
        "fusion_config": fusion_config(),
        "quantile_head_config": quantile_head_config(),
    }
    for name, expected in expected_configs.items():
        if checkpoint[name] != expected:
            raise ValueError(f"clean checkpoint {name} mismatch")
    if checkpoint["training_config_hash"] != _stable_hash(checkpoint["training_config"]):
        raise ValueError("clean checkpoint training_config_hash mismatch")
    normalization = CleanConditionNormalization(**checkpoint["normalization_statistics"])
    normalization.validate()
    if checkpoint["normalization_fit_ids_hash"] != normalization.fit_ids_hash:
        raise ValueError("clean checkpoint normalization fit IDs mismatch")


def load_clean_checkpoint(checkpoint_path: Path, device: torch.device) -> CleanQGeoGNN:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    validate_clean_checkpoint(checkpoint)
    normalization = CleanConditionNormalization(**checkpoint["normalization_statistics"])
    model = build_clean_model(normalization, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
