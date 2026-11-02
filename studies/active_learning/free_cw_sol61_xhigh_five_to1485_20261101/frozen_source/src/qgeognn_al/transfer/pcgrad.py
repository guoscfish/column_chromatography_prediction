"""Deterministic PCGrad restricted to an explicitly supplied parameter tuple."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class PCGradDiagnostics:
    directed_projection_attempts: int
    directed_projections_triggered: int
    projection_trigger_fraction: float
    projected_minus_original_update_norm: float
    projected_original_update_norm_ratio: float


def deterministic_pcgrad_seed(outer_seed: int, inner_fold: int, epoch: int, step: int) -> int:
    """Mix only declared context coordinates into a stable unsigned 64-bit seed."""

    values = (int(outer_seed), int(inner_fold), int(epoch), int(step))
    state = 0x9E3779B97F4A7C15
    for value in values:
        state ^= (value & 0xFFFFFFFFFFFFFFFF) + 0x9E3779B97F4A7C15 + (state << 6) + (state >> 2)
        state &= 0xFFFFFFFFFFFFFFFF
    return state


def task_gradient_tuples(
    losses: Mapping[str, torch.Tensor], parameters: Sequence[nn.Parameter]
) -> dict[str, tuple[torch.Tensor, ...]]:
    """Read per-task gradients without mutating any ``.grad`` tensor."""

    parameters = tuple(parameters)
    if not parameters:
        raise ValueError("PCGrad requires at least one shared parameter")
    gradients: dict[str, tuple[torch.Tensor, ...]] = {}
    for task, loss in losses.items():
        values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        gradients[task] = tuple(
            torch.zeros_like(parameter) if value is None else value.detach().clone()
            for parameter, value in zip(parameters, values)
        )
    if len(gradients) < 2:
        raise ValueError("PCGrad requires at least two task losses")
    return gradients


def _dot(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> torch.Tensor:
    return sum((a * b).sum() for a, b in zip(left, right))


def _norm(values: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(sum((value * value).sum() for value in values))


def gradient_geometry(gradients: Mapping[str, Sequence[torch.Tensor]]) -> dict[str, float]:
    tasks = tuple(gradients)
    result: dict[str, float] = {}
    norms = {task: _norm(gradients[task]) for task in tasks}
    for task in tasks:
        result[f"grad_norm_{task}"] = float(norms[task].cpu())
    for index, left in enumerate(tasks):
        for right in tasks[index + 1:]:
            denominator = norms[left] * norms[right]
            cosine = _dot(gradients[left], gradients[right]) / denominator if float(denominator) > 0 else denominator.new_tensor(0.)
            result[f"cos_{left}_{right}"] = float(cosine.cpu())
    return result


def project_conflicting_gradients(
    gradients: Mapping[str, Sequence[torch.Tensor]], *, permutation_seed: int, eps: float = 1e-12
) -> tuple[dict[str, tuple[torch.Tensor, ...]], PCGradDiagnostics]:
    """Apply standard task-wise PCGrad with deterministic randomized order.

    Each task gradient is projected sequentially against independently shuffled
    *original* gradients of the other tasks. Projected task gradients are then
    averaged by :func:`pcgrad_backward`.
    """

    tasks = tuple(gradients)
    if len(tasks) < 2:
        raise ValueError("PCGrad requires at least two tasks")
    widths = {len(tuple(gradients[task])) for task in tasks}
    if len(widths) != 1 or not widths or next(iter(widths)) == 0:
        raise ValueError("task gradients must have the same nonzero parameter count")
    original = {task: tuple(value.detach().clone() for value in gradients[task]) for task in tasks}
    projected = {task: [value.detach().clone() for value in gradients[task]] for task in tasks}
    rng = np.random.default_rng(int(permutation_seed))
    task_order = [tasks[int(index)] for index in rng.permutation(len(tasks))]
    attempts = triggered = 0
    for task in task_order:
        others = [other for other in tasks if other != task]
        others = [others[int(index)] for index in rng.permutation(len(others))]
        for other in others:
            attempts += 1
            dot = _dot(projected[task], original[other])
            denominator = _dot(original[other], original[other])
            if float(dot) < 0.0 and float(denominator) > 0.0:
                coefficient = dot / (denominator + float(eps))
                projected[task] = [current - coefficient * reference for current, reference in zip(projected[task], original[other])]
                triggered += 1
    projected_tuple = {task: tuple(values) for task, values in projected.items()}
    original_update = tuple(torch.stack([original[task][i] for task in tasks]).mean(dim=0) for i in range(next(iter(widths))))
    projected_update = tuple(torch.stack([projected_tuple[task][i] for task in tasks]).mean(dim=0) for i in range(next(iter(widths))))
    difference = tuple(after - before for before, after in zip(original_update, projected_update))
    original_norm = _norm(original_update)
    ratio = _norm(projected_update) / original_norm if float(original_norm) > 0 else original_norm.new_tensor(1.)
    diagnostics = PCGradDiagnostics(
        directed_projection_attempts=attempts,
        directed_projections_triggered=triggered,
        projection_trigger_fraction=triggered / attempts,
        projected_minus_original_update_norm=float(_norm(difference).cpu()),
        projected_original_update_norm_ratio=float(ratio.cpu()),
    )
    return projected_tuple, diagnostics


def pcgrad_backward(
    losses: Mapping[str, torch.Tensor],
    shared_parameters: Sequence[nn.Parameter],
    *,
    permutation_seed: int,
) -> tuple[dict[str, tuple[torch.Tensor, ...]], dict[str, tuple[torch.Tensor, ...]], PCGradDiagnostics]:
    """Backpropagate mean loss, replacing only supplied shared gradients."""

    shared_parameters = tuple(shared_parameters)
    raw = task_gradient_tuples(losses, shared_parameters)
    projected, diagnostics = project_conflicting_gradients(raw, permutation_seed=permutation_seed)
    objective = torch.stack(tuple(losses.values())).mean()
    objective.backward()
    tasks = tuple(losses)
    for index, parameter in enumerate(shared_parameters):
        replacement = torch.stack([projected[task][index] for task in tasks]).mean(dim=0)
        parameter.grad = replacement.detach().clone()
    return raw, projected, diagnostics
