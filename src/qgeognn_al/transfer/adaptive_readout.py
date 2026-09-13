"""Residual, permutation-invariant readouts for the active QGeoGNN-V2 API.

The active source predictor uses ``global_add_pool``.  This module deliberately
keeps that representation and adds a gated attention residual, so a newly
constructed readout is exactly source-compatible at epoch zero.  It is not a
Transformer and does not alter the molecular backbone depth or width.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import softmax

from ..input_schema import CATEGORICAL_BOND_FEATURES


ReadoutKind = Literal["adaptive", "condition_query"]


def full_chromatographic_condition_inputs(atom_bond, condition_branch: nn.Module) -> torch.Tensor:
    """Return every audited sample-level condition for a condition query.

    The active graph already receives four source-scaled eluent descriptors
    (ExactMolWt, TPSA, nRotB, HBD), while the post-pooling completion branch
    receives HBA, LogP, loading solvent, loading amount and loading volume.
    This function makes the complete nine semantic conditions directly visible
    to the query without introducing column identity or physical metadata:

    ``6 source-scaled eluent values + 4D loading-solvent embedding + 2
    source-normalized loading values``.

    A test-only/custom branch can provide ``full_condition_inputs`` itself.
    """

    custom = getattr(condition_branch, "full_condition_inputs", None)
    if callable(custom):
        value = custom(atom_bond)
        if value.ndim != 2:
            raise ValueError("custom full condition inputs must be rank two")
        return value
    typed_inputs = getattr(condition_branch, "typed_inputs", None)
    if not callable(typed_inputs):
        raise TypeError("condition branch must expose typed_inputs")
    edge_attr = atom_bond.edge_attr
    offset = len(CATEGORICAL_BOND_FEATURES)
    continuous = edge_attr[:, offset:]
    if continuous.shape[1] < 10:
        raise ValueError("active graph lacks the audited 10 continuous edge features")
    edge_batch = atom_bond.batch[atom_bond.edge_index[0]]
    # Positions 1:7 are the six min-max-scaled eluent descriptors.  Edge-level
    # replication makes their graph mean exactly the sample-level value.
    eluent = global_mean_pool(continuous[:, 1:7].to(torch.float32), edge_batch)
    completion = typed_inputs(atom_bond)
    if completion.ndim != 2 or completion.shape[1] != 8:
        raise ValueError("active completion branch must emit 8 typed features")
    # typed = [HBA, LogP, normalized_loading_amount, normalized_loading_volume,
    #          4D loading-solvent embedding]. HBA/LogP are already in eluent.
    return torch.cat([eluent, completion[:, 2:]], dim=1)


def _graph_count(batch: torch.Tensor) -> int:
    if batch.ndim != 1 or batch.numel() == 0:
        raise ValueError("node batch vector must be a non-empty rank-one tensor")
    return int(batch.max().item()) + 1


class ResidualAttentionReadout(nn.Module):
    """Multi-head graph attention pooled as a residual to fixed sum pooling.

    ``gate`` is intentionally initialized to zero.  The attention parameters
    therefore receive useful gradients through the gate after its first update,
    while the source model's function remains unchanged before any optimization.
    """

    def __init__(self, *, hidden_dim: int = 128, heads: int = 4,
                 kind: ReadoutKind = "adaptive", condition_dim: int = 8):
        super().__init__()
        if hidden_dim < 1 or heads < 1 or hidden_dim % heads:
            raise ValueError("hidden_dim must be positive and divisible by heads")
        if kind not in ("adaptive", "condition_query"):
            raise ValueError(f"unsupported readout kind: {kind}")
        if condition_dim < 1:
            raise ValueError("condition_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = self.hidden_dim // self.heads
        self.kind: ReadoutKind = kind
        self.condition_dim = int(condition_dim)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.output = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        if self.kind == "adaptive":
            self.global_query = nn.Parameter(torch.empty(self.heads, self.head_dim))
            nn.init.xavier_uniform_(self.global_query)
            self.condition_encoder = None
        else:
            self.register_parameter("global_query", None)
            # The final layer has exactly hidden_dim outputs: for four heads it
            # reshapes to four 32-dimensional queries.
            self.condition_encoder = nn.Sequential(
                nn.Linear(self.condition_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        self.gate = nn.Parameter(torch.zeros(()))

    def _queries(self, conditions: torch.Tensor | None, graph_count: int) -> torch.Tensor:
        if self.kind == "adaptive":
            return self.global_query.unsqueeze(0).expand(graph_count, -1, -1)
        if conditions is None:
            raise ValueError("condition-query readout requires per-graph conditions")
        if conditions.ndim != 2 or conditions.shape != (graph_count, self.condition_dim):
            raise ValueError("condition-query inputs have an unexpected shape")
        assert self.condition_encoder is not None
        return self.condition_encoder(conditions).reshape(graph_count, self.heads, self.head_dim)

    def attention(self, node_embeddings: torch.Tensor, batch: torch.Tensor,
                  conditions: torch.Tensor | None = None) -> torch.Tensor:
        """Pool node embeddings without depending on their in-graph order."""

        if node_embeddings.ndim != 2 or node_embeddings.shape[1] != self.hidden_dim:
            raise ValueError("node embeddings have an unexpected shape")
        if batch.device != node_embeddings.device:
            raise ValueError("node embeddings and batch vector must share a device")
        graph_count = _graph_count(batch)
        query = self._queries(conditions, graph_count)
        keys = self.key(node_embeddings).reshape(-1, self.heads, self.head_dim)
        values = self.value(node_embeddings).reshape(-1, self.heads, self.head_dim)
        scores = (keys * query[batch]).sum(dim=-1) / math.sqrt(self.head_dim)
        weights = softmax(scores, batch, num_nodes=graph_count)
        pooled = global_add_pool((weights.unsqueeze(-1) * values).reshape(-1, self.hidden_dim),
                                 batch, size=graph_count)
        return self.output(pooled)

    def forward(self, fixed_sum: torch.Tensor, node_embeddings: torch.Tensor,
                batch: torch.Tensor, conditions: torch.Tensor | None = None) -> torch.Tensor:
        if fixed_sum.ndim != 2 or fixed_sum.shape[1] != self.hidden_dim:
            raise ValueError("fixed pooled representation has an unexpected shape")
        attended = self.attention(node_embeddings, batch, conditions)
        if attended.shape != fixed_sum.shape:
            raise RuntimeError("attention graph count does not match fixed pooling")
        return fixed_sum + self.gate * attended


class ConditionedReadoutQGeoGNNV2(nn.Module):
    """Active QGeoGNN-V2 with a gated residual attention readout.

    The source model's modules retain their original state-dict names.  This
    makes source checkpoint loading, the historical-shallow trainable scope,
    and source-function identity checks straightforward.
    """

    model_variant = "qgeognn_v2_conditioned_source_readout"

    def __init__(self, source_model: nn.Module, *, kind: ReadoutKind,
                 heads: int = 4, source_augmentation: nn.Module | None = None):
        super().__init__()
        required = ("backbone", "condition_branch", "head")
        if any(not isinstance(getattr(source_model, name, None), nn.Module) for name in required):
            raise TypeError("source_model must implement the active QGeoGNN-V2 module contract")
        self.backbone = source_model.backbone
        self.condition_branch = source_model.condition_branch
        self.head = source_model.head
        typed_inputs = getattr(self.condition_branch, "typed_inputs", None)
        if not callable(typed_inputs):
            raise TypeError("active condition branch must expose typed_inputs")
        self.adaptive_readout = ResidualAttentionReadout(
            hidden_dim=128, heads=heads, kind=kind, condition_dim=12
        )
        self.source_augmentation = source_augmentation
        self.readout_kind: ReadoutKind = kind
        self.readout_heads = int(heads)

    def fixed_sum_representation(self, atom_bond, bond_angle):
        nodes = self.backbone(atom_bond, bond_angle)
        summed = global_add_pool(nodes, atom_bond.batch)
        condition_residual, _ = self.condition_branch(atom_bond)
        return nodes, summed + condition_residual

    def extract_representation(self, atom_bond, bond_angle):
        nodes, fixed_sum = self.fixed_sum_representation(atom_bond, bond_angle)
        conditions = (full_chromatographic_condition_inputs(atom_bond, self.condition_branch)
                      if self.readout_kind == "condition_query" else None)
        representation = self.adaptive_readout(fixed_sum, nodes, atom_bond.batch, conditions)
        if self.source_augmentation is not None:
            representation = representation + self.source_augmentation(atom_bond)
        return representation

    def forward(self, atom_bond, bond_angle):
        output = self.head(self.extract_representation(atom_bond, bond_angle))
        return output if self.training else torch.clamp(output, min=0, max=1e8)


def build_conditioned_readout_model(source_model: nn.Module, *, kind: ReadoutKind,
                                    heads: int = 4,
                                    source_augmentation: nn.Module | None = None) -> ConditionedReadoutQGeoGNNV2:
    """Wrap a loaded source model without changing its epoch-zero function."""

    return ConditionedReadoutQGeoGNNV2(
        source_model, kind=kind, heads=heads, source_augmentation=source_augmentation
    )
