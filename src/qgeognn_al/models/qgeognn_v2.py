"""Standalone corrected QGeoGNN: five node layers and four effective edge updates.

This directly registers only the effective network; no Legacy/R2 construction,
pruning, alternative fusion, or head replacement occurs here.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool

from ..data import qg
from ..schemas.conditions import (
    ConditionNormalization, TypedConditionCompletionBranch, condition_schema,
    input_schema_hash,
)

MODEL_VARIANT = "qgeognn_v2"
PARAMETER_COUNT = 458952


class GeometryBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.atom_encoder = qg.AtomEncoder(128)
        self.bond_encoder = qg.BondEncoder(128)
        self.bond_float_encoder = qg.BondFloatRBF(qg.bond_float_names, 128)
        self.convs = nn.ModuleList(qg.GINConv(128) for _ in range(5))
        self.convs_bond_angle = nn.ModuleList(qg.GINConv(128) for _ in range(4))
        self.convs_bond_float = nn.ModuleList(qg.BondFloatRBF(qg.bond_float_names, 128) for _ in range(4))
        self.convs_bond_embeding = nn.ModuleList(qg.BondEncoder(128) for _ in range(4))
        self.convs_angle_float = nn.ModuleList(qg.BondAngleFloatRBF(qg.bond_angle_float_names, 128) for _ in range(4))

    def forward(self, atom_bond, bond_angle):
        edge_attr = atom_bond.edge_attr
        categorical = edge_attr[:, :len(qg.bond_id_names)].to(torch.int64)
        continuous = edge_attr[:, len(qg.bond_id_names):edge_attr.shape[1]+1].to(torch.float32)
        nodes = self.atom_encoder(atom_bond.x)
        edges = self.bond_float_encoder(continuous) + self.bond_encoder(categorical)
        for layer in range(5):
            nodes = self.convs[layer](nodes, atom_bond.edge_index, edges)
            if layer < 4:
                cur_edges = self.convs_bond_embeding[layer](categorical) + self.convs_bond_float[layer](continuous)
                angles = self.convs_angle_float[layer](bond_angle.edge_attr)
                edges = self.convs_bond_angle[layer](cur_edges, bond_angle.edge_index, angles)
                nodes = F.relu(nodes)
                edges = F.relu(edges)
        return nodes


class QGeoGNNV2(nn.Module):
    model_variant = MODEL_VARIANT

    def __init__(self, normalization: ConditionNormalization):
        super().__init__()
        self.backbone = GeometryBackbone()
        self.head = nn.Sequential(nn.Linear(128, 6), nn.ReLU())
        self.condition_branch = TypedConditionCompletionBranch(normalization)

    def extract_representation(self, atom_bond, bond_angle):
        nodes = self.backbone(atom_bond, bond_angle)
        pooled = global_add_pool(nodes, atom_bond.batch)
        residual, _ = self.condition_branch(atom_bond)
        return pooled + residual

    def forward(self, atom_bond, bond_angle):
        output = self.head(self.extract_representation(atom_bond, bond_angle))
        return output if self.training else torch.clamp(output, min=0, max=1e8)


def build_predictor(normalization: ConditionNormalization, device="cpu") -> QGeoGNNV2:
    if torch.device(device).type != "cpu":
        raise ValueError("The frozen graph/RBF runtime is CPU-qualified")
    # The retained QGeoGNN RBF primitives read their construction device from
    # the historical module global rather than from their constructor.
    qg.device = torch.device("cpu")
    return QGeoGNNV2(normalization).to(device)


def extract_representation(model, atom_bond, bond_angle):
    return model.extract_representation(atom_bond, bond_angle)


def predictor_checkpoint(model, *, preprocessing, training_config, provenance):
    return {
        "schema_version": 1, "model_variant": MODEL_VARIANT,
        "input_schema": condition_schema(), "input_schema_hash": input_schema_hash(),
        "output_order": ["V1_q10", "V1_q50", "V1_q90", "V2_q10", "V2_q50", "V2_q90"],
        "normalization": asdict(model.condition_branch.normalization),
        "preprocessing": preprocessing, "training_config": training_config,
        "provenance": provenance, "parameter_count": sum(p.numel() for p in model.parameters()),
        "model_state_dict": model.state_dict(),
    }


def validate_predictor_checkpoint(checkpoint):
    required = {"schema_version", "model_variant", "input_schema", "input_schema_hash", "output_order",
                "normalization", "preprocessing", "training_config", "provenance", "parameter_count", "model_state_dict"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"predictor checkpoint missing fields: {sorted(missing)}")
    if checkpoint["schema_version"] != 1 or checkpoint["model_variant"] != MODEL_VARIANT:
        raise ValueError("standalone predictor checkpoint version/variant mismatch")
    if checkpoint["input_schema"] != condition_schema() or checkpoint["input_schema_hash"] != input_schema_hash():
        raise ValueError("predictor input schema mismatch")
    if checkpoint["parameter_count"] != PARAMETER_COUNT:
        raise ValueError("predictor parameter count mismatch")
    if checkpoint["output_order"] != ["V1_q10", "V1_q50", "V1_q90", "V2_q10", "V2_q50", "V2_q90"]:
        raise ValueError("predictor output contract mismatch")
    ConditionNormalization(**checkpoint["normalization"]).validate()
    if checkpoint["preprocessing"].get("fit_role") != "source_train":
        raise ValueError("predictor preprocessing must be source-train only")


def load_predictor_checkpoint(path: Path, device="cpu") -> QGeoGNNV2:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    validate_predictor_checkpoint(checkpoint)
    model = build_predictor(ConditionNormalization(**checkpoint["normalization"]), device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model
