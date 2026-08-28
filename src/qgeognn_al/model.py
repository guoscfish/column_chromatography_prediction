"""Frozen QGeoGNN model, quantile-head, loss, and validation primitives."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import qg


class MonotonicQuantileHead(nn.Module):
    def __init__(self, legacy_linear: nn.Linear):
        super().__init__()
        self.linear = nn.Linear(legacy_linear.in_features, 6)
        with torch.no_grad():
            old_weight, old_bias = legacy_linear.weight.detach(), legacy_linear.bias.detach()
            for offset in (0, 3):
                self.linear.weight[offset].copy_(old_weight[offset + 1])
                self.linear.bias[offset].copy_(old_bias[offset + 1])
                self.linear.weight[offset + 1].copy_(old_weight[offset + 1] - old_weight[offset])
                self.linear.bias[offset + 1].copy_(old_bias[offset + 1] - old_bias[offset])
                self.linear.weight[offset + 2].copy_(old_weight[offset + 2] - old_weight[offset + 1])
                self.linear.bias[offset + 2].copy_(old_bias[offset + 2] - old_bias[offset + 1])

    def forward(self, h_graph: torch.Tensor) -> torch.Tensor:
        raw, outputs = self.linear(h_graph), []
        for offset in (0, 3):
            median = F.softplus(raw[:, offset])
            outputs.extend((median - F.softplus(raw[:, offset + 1]), median, median + F.softplus(raw[:, offset + 2])))
        return torch.stack(outputs, dim=1)


def install_monotonic_head(model: nn.Module) -> None:
    legacy_head = model.graph_pred_linear
    if not isinstance(legacy_head, nn.Sequential) or not isinstance(legacy_head[0], nn.Linear):
        raise TypeError("Expected the legacy Sequential(Linear, ReLU) prediction head")
    model.graph_pred_linear = MonotonicQuantileHead(legacy_head[0]).to(next(model.parameters()).device)


def build_model(device: torch.device) -> nn.Module:
    qg.device = str(device)
    return qg.GINGraphPooling(num_tasks=6, num_layers=5, emb_dim=128, drop_ratio=0.0, graph_pooling="sum", descriptor_dim=1827).to(device)


def configure_trainable(model: nn.Module, mode: str) -> tuple[int, int]:
    for parameter in model.parameters():
        parameter.requires_grad = mode == "full"
    if mode != "full":
        layers = [4] if mode == "last1_head" else [3, 4] if mode == "last2_head" else []
        if mode not in {"last1_head", "last2_head", "head_only"}:
            raise ValueError(mode)
        for name, parameter in model.named_parameters():
            if name.startswith("graph_pred_linear") or any(name.startswith(f"gnn_node.{branch}.{layer}.") for branch in ["convs", "convs_bond_angle", "convs_bond_embeding", "convs_bond_float", "convs_angle_float"] for layer in layers):
                parameter.requires_grad = True
    return (sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters()))


def set_training_mode(model: nn.Module) -> None:
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            parameters = list(module.parameters(recurse=False))
            if parameters and not any(parameter.requires_grad for parameter in parameters):
                module.eval()


def metrics_from_arrays(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for target, name in enumerate(("V1", "V2")):
        quantiles = pred[:, target * 3 : target * 3 + 3]
        residual = y_true[:, target] - quantiles[:, 1]
        ss_tot = float(np.square(y_true[:, target] - y_true[:, target].mean()).sum())
        errors = y_true[:, [target]] - quantiles
        levels = np.array([0.1, 0.5, 0.9], dtype=np.float32).reshape(1, -1)
        result[f"{name}_mae"] = float(np.abs(residual).mean())
        result[f"{name}_rmse"] = float(np.sqrt(np.square(residual).mean()))
        result[f"{name}_r2"] = float(1 - np.square(residual).sum() / ss_tot) if ss_tot else float("nan")
        result[f"{name}_mean_pinball_loss"] = float(np.maximum(levels * errors, (levels - 1) * errors).mean())
        result[f"{name}_interval_80_coverage"] = float(np.mean((y_true[:, target] >= quantiles[:, 0]) & (y_true[:, target] <= quantiles[:, 2])))
        result[f"{name}_interval_80_mean_width"] = float(np.mean(quantiles[:, 2] - quantiles[:, 0]))
    result["quantile_crossing_rate"] = float(np.mean((pred[:, 0] > pred[:, 1]) | (pred[:, 1] > pred[:, 2]) | (pred[:, 3] > pred[:, 4]) | (pred[:, 4] > pred[:, 5])))
    return result


def validation_scores(metrics: dict, target_variance: dict[str, float]) -> tuple[float, float]:
    legacy = metrics["V1_rmse"] ** 2 + 0.5 * metrics["V2_rmse"] ** 2
    normalized = metrics["V1_rmse"] ** 2 / target_variance["V1"] + metrics["V2_rmse"] ** 2 / target_variance["V2"]
    return float(normalized), float(legacy)


def quantile_target_loss(true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return qg.q_loss(0.1, true, pred[:, 0]) + torch.mean((true - pred[:, 1]) ** 2) + qg.q_loss(0.9, true, pred[:, 2]) + torch.mean(torch.relu(pred[:, 0] - pred[:, 1])) + torch.mean(torch.relu(pred[:, 1] - pred[:, 2]))
