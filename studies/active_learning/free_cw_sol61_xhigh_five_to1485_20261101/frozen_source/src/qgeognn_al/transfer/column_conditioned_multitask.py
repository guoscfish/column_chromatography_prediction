"""Column-conditioned multi-task QGeoGNN primitives.

This module contains only the preregistered A1 separate-head control and the A2
column-FiLM intervention. It does not implement calibration, loss weighting, or
post-result architecture selection.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool

from ..data import qg
from ..models import load_predictor_checkpoint


TASKS = ("4g", "25g", "40g")
TASK_TO_ID = {name: index for index, name in enumerate(TASKS)}
ARCHITECTURES = ("MULTITASK_SEPARATE_HEADS", "MULTITASK_COLUMN_FILM")
FILM_LAYERS = (3, 4)
FILM_EMBEDDING_DIM = 16


def task_ids(names: Sequence[str], *, device: torch.device | str | None = None) -> torch.Tensor:
    try:
        values = [TASK_TO_ID[str(name)] for name in names]
    except KeyError as error:
        raise ValueError(f"unknown column task: {error.args[0]}") from error
    return torch.tensor(values, dtype=torch.long, device=device)


class BalancedTaskBatcher:
    """Deterministic equal-participation batches for 4g/25g/40g.

    One epoch spans enough full equal-task steps to cover the largest target
    task (25g or 40g). Each task contributes exactly ``per_task_batch_size``
    draws per step. Draws cycle through deterministic permutations, so the
    smaller target may wrap and the much larger 4g source is subsampled without
    allowing source volume to define epoch length.
    """

    def __init__(self, indices: Mapping[str, Sequence[int]], *, per_task_batch_size: int, seed: int):
        if tuple(indices) != TASKS and set(indices) != set(TASKS):
            raise ValueError(f"balanced batching requires exactly {TASKS}")
        self.indices = {task: np.asarray(indices[task], dtype=np.int64) for task in TASKS}
        if any(len(values) == 0 for values in self.indices.values()):
            raise ValueError("every task must contain at least one training row")
        if int(per_task_batch_size) < 1:
            raise ValueError("per_task_batch_size must be positive")
        self.per_task_batch_size = int(per_task_batch_size)
        self.seed = int(seed)
        largest_target = max(len(self.indices["25g"]), len(self.indices["40g"]))
        self.steps_per_epoch = max(1, math.ceil(largest_target / self.per_task_batch_size))

    def _draws(self, task: str, epoch: int) -> np.ndarray:
        needed = self.steps_per_epoch * self.per_task_batch_size
        values = self.indices[task]
        task_offset = TASK_TO_ID[task] * 1_000_003
        rng = np.random.default_rng(self.seed * 10_007 + int(epoch) * 101 + task_offset)
        chunks = []
        while sum(len(chunk) for chunk in chunks) < needed:
            chunks.append(rng.permutation(values))
        return np.concatenate(chunks)[:needed]

    def epoch_batches(self, epoch: int) -> Iterable[dict[str, np.ndarray]]:
        draws = {task: self._draws(task, epoch) for task in TASKS}
        size = self.per_task_batch_size
        for step in range(self.steps_per_epoch):
            yield {task: values[step * size:(step + 1) * size] for task, values in draws.items()}

    def participation(self, epoch: int) -> dict[str, int]:
        counts = {task: 0 for task in TASKS}
        for batch in self.epoch_batches(epoch):
            for task, values in batch.items():
                counts[task] += len(values)
        return counts


def global_target_group_folds(
    canonical_smiles: Sequence[str], columns: Sequence[str], *, n_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return globally grouped target folds across joined 25g and 40g rows."""

    groups = np.asarray([str(value) for value in canonical_smiles])
    column_values = np.asarray([str(value) for value in columns])
    if len(groups) != len(column_values) or len(groups) == 0:
        raise ValueError("canonical_smiles and columns must be non-empty and aligned")
    if set(column_values) - {"25g", "40g"}:
        raise ValueError("global target folds accept only 25g and 40g")
    if len(np.unique(groups)) < int(n_splits):
        raise ValueError("fewer canonical groups than requested folds")
    positions = np.arange(len(groups), dtype=np.int64)
    folds = []
    for train, valid in GroupKFold(n_splits=int(n_splits)).split(positions, groups=groups):
        if set(groups[train]) & set(groups[valid]):
            raise RuntimeError("canonical molecule crosses a global inner fold")
        folds.append((positions[train], positions[valid]))
    return folds


class MultiTaskQGeoGNN(nn.Module):
    """A1/A2 wrapper around the active QGeoGNNV2 implementation."""

    def __init__(self, source: nn.Module, *, architecture: str):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"unsupported architecture: {architecture}")
        self.architecture = architecture
        self.backbone = source.backbone
        self.condition_branch = source.condition_branch
        self.heads = nn.ModuleDict({task: deepcopy(source.head) for task in TASKS})
        self.use_film = architecture == "MULTITASK_COLUMN_FILM"
        if self.use_film:
            self.column_embedding = nn.Embedding(len(TASKS), FILM_EMBEDDING_DIM)
            self.film_generators = nn.ModuleDict({str(layer): nn.Linear(FILM_EMBEDDING_DIM, 256) for layer in FILM_LAYERS})
            for generator in self.film_generators.values():
                nn.init.zeros_(generator.weight)
                nn.init.zeros_(generator.bias)
        self.configure_trainable_scope()

    def configure_trainable_scope(self) -> tuple[int, int]:
        for parameter in self.parameters():
            parameter.requires_grad = False
        prefixes = (
            "backbone.convs.3.", "backbone.convs.4.",
            "backbone.convs_bond_angle.3.", "backbone.convs_bond_float.3.",
            "backbone.convs_bond_embeding.3.", "backbone.convs_angle_float.3.",
            "condition_branch.", "heads.", "column_embedding.", "film_generators.",
        )
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.startswith(prefixes)
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.parameters())
        return trainable, total

    def optimizer(self, *, pretrained_lr: float = 1e-4, new_lr: float = 3e-4,
                  weight_decay: float = 1e-5) -> torch.optim.Optimizer:
        pretrained, new = [], []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith(("heads.25g.", "heads.40g.", "column_embedding.", "film_generators.")):
                new.append(parameter)
            else:
                pretrained.append(parameter)
        if not pretrained or not new:
            raise RuntimeError("optimizer requires both pretrained and new parameter groups")
        return torch.optim.Adam(
            [{"params": pretrained, "lr": float(pretrained_lr), "name": "pretrained"},
             {"params": new, "lr": float(new_lr), "name": "new_head_or_film"}],
            weight_decay=float(weight_decay),
        )

    def shared_gradient_parameters(self) -> tuple[nn.Parameter, ...]:
        prefixes = (
            "backbone.convs.3.", "backbone.convs.4.",
            "backbone.convs_bond_angle.3.", "backbone.convs_bond_float.3.",
            "backbone.convs_bond_embeding.3.", "backbone.convs_angle_float.3.",
        )
        return tuple(parameter for name, parameter in self.named_parameters()
                     if parameter.requires_grad and name.startswith(prefixes))

    def _apply_film(self, nodes: torch.Tensor, graph_batch: torch.Tensor,
                    task_index: torch.Tensor, layer: int) -> torch.Tensor:
        parameters = self.film_generators[str(layer)](self.column_embedding(task_index))
        gamma, beta = parameters.chunk(2, dim=1)
        return nodes * (1.0 + gamma[graph_batch]) + beta[graph_batch]

    def _backbone_forward(self, atom_bond, bond_angle, task_index: torch.Tensor) -> torch.Tensor:
        edge_attr = atom_bond.edge_attr
        categorical = edge_attr[:, :len(qg.bond_id_names)].to(torch.int64)
        continuous = edge_attr[:, len(qg.bond_id_names):edge_attr.shape[1] + 1].to(torch.float32)
        if hasattr(atom_bond, "frozen_nodes"):
            nodes, edges = atom_bond.frozen_nodes, atom_bond.frozen_edges
            first_layer = 3
        else:
            nodes = self.backbone.atom_encoder(atom_bond.x)
            edges = self.backbone.bond_float_encoder(continuous) + self.backbone.bond_encoder(categorical)
            first_layer = 0
        for layer in range(first_layer, 5):
            nodes = self.backbone.convs[layer](nodes, atom_bond.edge_index, edges)
            if self.use_film and layer in FILM_LAYERS:
                nodes = self._apply_film(nodes, atom_bond.batch, task_index, layer)
            if layer < 4:
                cur_edges = (self.backbone.convs_bond_embeding[layer](categorical)
                             + self.backbone.convs_bond_float[layer](continuous))
                angles = self.backbone.convs_angle_float[layer](bond_angle.edge_attr)
                edges = self.backbone.convs_bond_angle[layer](cur_edges, bond_angle.edge_index, angles)
                nodes = F.relu(nodes)
                edges = F.relu(edges)
        return nodes

    def extract_representation(self, atom_bond, bond_angle, task_index: torch.Tensor) -> torch.Tensor:
        task_index = torch.as_tensor(task_index, dtype=torch.long, device=atom_bond.x.device).reshape(-1)
        graph_count = int(atom_bond.num_graphs)
        if len(task_index) != graph_count or torch.any(task_index < 0) or torch.any(task_index >= len(TASKS)):
            raise ValueError("one valid task id is required per graph")
        nodes = self._backbone_forward(atom_bond, bond_angle, task_index)
        pooled = global_add_pool(nodes, atom_bond.batch)
        residual, _ = self.condition_branch(atom_bond)
        return pooled + residual

    def forward(self, atom_bond, bond_angle, task_index: torch.Tensor) -> torch.Tensor:
        task_index = torch.as_tensor(task_index, dtype=torch.long, device=atom_bond.x.device).reshape(-1)
        representation = self.extract_representation(atom_bond, bond_angle, task_index)
        all_heads = torch.stack([self.heads[task](representation) for task in TASKS], dim=1)
        row = torch.arange(len(task_index), device=representation.device)
        output = all_heads[row, task_index]
        return output if self.training else torch.clamp(output, min=0, max=1e8)


def build_multitask_model(source_checkpoint: Path, architecture: str, *, device: str = "cpu") -> MultiTaskQGeoGNN:
    source = load_predictor_checkpoint(Path(source_checkpoint), device=device)
    return MultiTaskQGeoGNN(source, architecture=architecture).to(device)


def head_copies_are_exact(model: MultiTaskQGeoGNN) -> bool:
    states = {task: model.heads[task].state_dict() for task in TASKS}
    return all(torch.equal(states["4g"][name], states[task][name])
               for task in ("25g", "40g") for name in states["4g"])


def film_is_zero_initialized(model: MultiTaskQGeoGNN) -> bool:
    if not model.use_film:
        raise ValueError("zero-FiLM check requires A2")
    return all(torch.count_nonzero(parameter).item() == 0
               for generator in model.film_generators.values() for parameter in generator.parameters())


def gradient_geometry(losses: Mapping[str, torch.Tensor], parameters: Sequence[nn.Parameter]) -> dict[str, float]:
    """Compute diagnostic-only per-task norms and pairwise cosine similarities."""

    gradients = {}
    for task in TASKS:
        values = torch.autograd.grad(losses[task], parameters, retain_graph=True, allow_unused=True)
        gradients[task] = tuple(torch.zeros_like(parameter) if value is None else value
                                for parameter, value in zip(parameters, values))
    result = {}
    norms = {}
    for task, values in gradients.items():
        square = sum(torch.sum(value.detach() ** 2) for value in values)
        norms[task] = torch.sqrt(square)
        result[f"grad_norm_{task}"] = float(norms[task].cpu())
    for left, right in (("4g", "25g"), ("4g", "40g"), ("25g", "40g")):
        dot = sum(torch.sum(a.detach() * b.detach()) for a, b in zip(gradients[left], gradients[right]))
        denominator = norms[left] * norms[right]
        cosine = dot / denominator if float(denominator) > 0 else dot.new_tensor(float("nan"))
        result[f"cos_{left}_{right}"] = float(cosine.cpu())
    return result


@torch.no_grad()
def cache_frozen_prefix(source, atoms, angles, *, batch_size=256):
    """Cache source-eval outputs feeding node block 3; no labels are consumed."""
    from torch_geometric.data import Batch

    source.eval()
    backbone = source.backbone
    for start in range(0, len(atoms), batch_size):
        atom = Batch.from_data_list(atoms[start:start + batch_size])
        angle = Batch.from_data_list(angles[start:start + batch_size])
        categorical = atom.edge_attr[:, :len(qg.bond_id_names)].to(torch.int64)
        continuous = atom.edge_attr[:, len(qg.bond_id_names):].to(torch.float32)
        nodes = backbone.atom_encoder(atom.x)
        edges = backbone.bond_float_encoder(continuous) + backbone.bond_encoder(categorical)
        for layer in range(3):
            nodes = F.relu(backbone.convs[layer](nodes, atom.edge_index, edges))
            cur_edges = backbone.convs_bond_embeding[layer](categorical) + backbone.convs_bond_float[layer](continuous)
            angle_hidden = backbone.convs_angle_float[layer](angle.edge_attr)
            edges = F.relu(backbone.convs_bond_angle[layer](cur_edges, angle.edge_index, angle_hidden))
        node_start = edge_start = 0
        for graph in atoms[start:start + batch_size]:
            node_count, edge_count = graph.num_nodes, graph.edge_index.shape[1]
            graph.frozen_nodes = nodes[node_start:node_start + node_count].clone()
            graph.frozen_edges = edges[edge_start:edge_start + edge_count].clone()
            node_start += node_count
            edge_start += edge_count
