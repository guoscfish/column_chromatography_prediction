"""Prediction-function-preserving pruning of the frozen five-layer R2.

Only the audited unused registrations and terminal edge computation are removed.
R2 itself remains unchanged. Build from a canonical R2 to preserve initialization
and RNG consumption, including its zero-initialized condition completion output.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from .condition_complete_v2 import ConditionCompleteQGeoGNNV2
from .data import qg
from .model import build_model

MODEL_VARIANT = "qgeognn_condition_complete_v2_pruned"
DEAD_MODULE_PREFIXES = (
    "legacy_model.NN_descriptor",
    "legacy_model.gnn_node.bond_angle_encoder",
    "legacy_model.gnn_node.batch_norms",
    "legacy_model.gnn_node.batch_norms_ba",
    "legacy_model.gnn_node.convs_bond_angle.4",
    "legacy_model.gnn_node.convs_bond_embeding.4",
    "legacy_model.gnn_node.convs_bond_float.4",
    "legacy_model.gnn_node.convs_angle_float.4",
)


def validate_frozen_r2(model):
    node = model.legacy_model.gnn_node
    head = model.legacy_model.graph_pred_linear
    if (node.num_layers != 5 or node.JK != "last" or node.residual or node.drop_ratio != 0
            or not qg.Use_geometry_enhanced or model.legacy_model.pool is not qg.global_add_pool
            or not isinstance(head, nn.Sequential) or len(head) != 2
            or not isinstance(head[0], nn.Linear) or not isinstance(head[1], nn.ReLU)
            or (head[0].in_features, head[0].out_features) != (128, 6)):
        raise ValueError("Expected the frozen R2 five-layer geometry, sum pool and Linear/ReLU head")


def is_dead_key(name):
    return any(name.startswith(prefix + ".") for prefix in DEAD_MODULE_PREFIXES)


class PrunedGINNodeEmbedding(nn.Module):
    def __init__(self, original):
        super().__init__()
        if (original.num_layers != 5 or original.JK != "last" or
                original.residual or original.drop_ratio != 0 or not qg.Use_geometry_enhanced):
            raise ValueError("Pruning is qualified only for the frozen R2 geometry configuration")
        self.num_layers = original.num_layers
        self.JK, self.residual, self.drop_ratio = original.JK, original.residual, original.drop_ratio
        for name in ("atom_encoder", "bond_encoder", "bond_float_encoder", "convs"):
            self.add_module(name, getattr(original, name))
        for name in ("convs_bond_angle", "convs_bond_float", "convs_bond_embeding", "convs_angle_float"):
            self.add_module(name, nn.ModuleList(list(getattr(original, name)[:4])))

    def forward(self, batched_atom_bond, batched_bond_angle):
        x, edge_index, edge_attr = batched_atom_bond.x, batched_atom_bond.edge_index, batched_atom_bond.edge_attr
        edge_index_ba, edge_attr_ba = batched_bond_angle.edge_index, batched_bond_angle.edge_attr
        h_list = [self.atom_encoder(x)]
        h_list_ba = [self.bond_float_encoder(edge_attr[:, len(qg.bond_id_names):edge_attr.shape[1]+1].to(torch.float32))
                     + self.bond_encoder(edge_attr[:, 0:len(qg.bond_id_names)].to(torch.int64))]
        for layer in range(self.num_layers):
            h = self.convs[layer](h_list[layer], edge_index, h_list_ba[layer])
            if layer < self.num_layers - 1:
                cur_h_ba = self.convs_bond_embeding[layer](edge_attr[:, 0:len(qg.bond_id_names)].to(torch.int64)) + self.convs_bond_float[layer](edge_attr[:, len(qg.bond_id_names):edge_attr.shape[1]+1].to(torch.float32))
                cur_angle_hidden = self.convs_angle_float[layer](edge_attr_ba)
                h_ba = self.convs_bond_angle[layer](cur_h_ba, edge_index_ba, cur_angle_hidden)
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
                h_ba = F.dropout(F.relu(h_ba), self.drop_ratio, training=self.training)
                h_list_ba.append(h_ba)
            else:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            h_list.append(h)
        # R2 discards the second return value; no terminal edge output is exposed.
        return h_list[-1], None


class PrunedConditionCompleteQGeoGNNV2(ConditionCompleteQGeoGNNV2):
    model_variant = MODEL_VARIANT

    def __init__(self, canonical_r2):
        nn.Module.__init__(self)
        validate_frozen_r2(canonical_r2)
        # Legacy RBF centers are unregistered non-leaf views, so deepcopy is not
        # supported. Rebuild an independent source under an isolated RNG context
        # and load every registered value before transferring retained modules.
        device = next(canonical_r2.parameters()).device
        if device.type != "cpu":
            raise ValueError("This controlled pruning is qualified on CPU only")
        with torch.random.fork_rng(devices=[]):
            source = ConditionCompleteQGeoGNNV2(build_model(device), canonical_r2.condition_branch.normalization).to(device)
        source.load_state_dict(canonical_r2.state_dict(), strict=True)
        self.legacy_model = nn.Module()
        self.legacy_model.gnn_node = PrunedGINNodeEmbedding(source.legacy_model.gnn_node)
        self.legacy_model.pool = source.legacy_model.pool
        self.legacy_model.graph_pred_linear = source.legacy_model.graph_pred_linear
        self.condition_branch = source.condition_branch
        self.train(canonical_r2.training)


def convert_r2_state_dict_to_pruned(state, original, pruned):
    """Strict schema conversion, including buffers; never silently discard unknown keys."""
    validate_frozen_r2(original)
    source = original.state_dict()
    target = pruned.state_dict()
    expected_retained = {k for k in source if not is_dead_key(k)}
    if expected_retained != set(target):
        raise ValueError("pruned schema does not exactly match audited retained R2 keys")
    unexpected = sorted(set(state) - set(source))
    missing = sorted(set(source) - set(state))
    if unexpected or missing:
        raise ValueError(f"invalid R2 state: unexpected={unexpected}, missing={missing}")
    for key, value in state.items():
        if value.shape != source[key].shape or value.dtype != source[key].dtype:
            raise ValueError(f"R2 shape/dtype mismatch: {key}")
    converted = OrderedDict((k, state[k].detach().clone()) for k in target)
    parameters = dict(original.named_parameters())
    removed = sorted(k for k in source if is_dead_key(k))
    report = {
        "original_parameter_count": sum(p.numel() for p in parameters.values()),
        "removed_parameter_count": sum(p.numel() for k, p in parameters.items() if is_dead_key(k)),
        "retained_parameter_count": sum(p.numel() for k, p in parameters.items() if not is_dead_key(k)),
        "removed_keys": removed, "retained_keys": list(converted),
        "unexpected_keys": unexpected, "missing_keys": missing,
        "retained_values_bitwise_equal": all(torch.equal(state[k], v) for k, v in converted.items()),
        "conversion_status": "PASS",
    }
    return converted, report
