"""Preregistered low-capacity models for the full-data headroom study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .controlled_conditional_extension import center_width_inverse, center_width_transform, project_physical


def _scale(values: np.ndarray) -> float:
    value = float(np.std(np.asarray(values, dtype=float)))
    return value if value >= 1e-8 else 1.0


@dataclass(frozen=True)
class HierarchicalScalarFit:
    columns: tuple[str, ...]
    global_coefficients: np.ndarray
    deviations: np.ndarray
    source_scale: float
    ea_mean: float
    ea_feature_mean: float
    ea_feature_std: float
    base_penalty: float
    delta_penalty: float

    def predict(self, source: np.ndarray, ea: np.ndarray, column: Sequence[str]) -> np.ndarray:
        source = np.asarray(source, dtype=float).reshape(-1)
        ea = np.asarray(ea, dtype=float).reshape(-1)
        labels = np.asarray(column, dtype=str).reshape(-1)
        if not (source.shape == ea.shape == labels.shape) or not np.isfinite(source).all() or not np.isfinite(ea).all():
            raise ValueError("aligned finite source, EA, and column values required")
        u = source / self.source_scale
        interaction = (u * (ea - self.ea_mean) - self.ea_feature_mean) / self.ea_feature_std
        design = np.column_stack([u, np.ones(len(u)), interaction])
        lookup = {name: index for index, name in enumerate(self.columns)}
        try:
            coefficients = np.vstack([self.global_coefficients + self.deviations[lookup[value]] for value in labels])
        except KeyError as error:
            raise ValueError(f"unfitted column: {error.args[0]}") from error
        return np.sum(design * coefficients, axis=1) * self.source_scale

    def audit(self) -> dict[str, object]:
        return {
            "columns": list(self.columns), "global_coefficients": self.global_coefficients.tolist(),
            "deviations": self.deviations.tolist(), "source_scale": self.source_scale,
            "ea_mean": self.ea_mean, "ea_feature_mean": self.ea_feature_mean,
            "ea_feature_std": self.ea_feature_std, "base_penalty": self.base_penalty,
            "delta_penalty": self.delta_penalty,
        }


def fit_hierarchical_scalar(
    source: np.ndarray, normalized_truth: np.ndarray, ea: np.ndarray, column: Sequence[str],
    columns: Sequence[str], *, base_penalty: float, delta_penalty: float,
) -> HierarchicalScalarFit:
    """Fit global M3 coefficients plus strongly shrunk column deviations."""
    source = np.asarray(source, dtype=float).reshape(-1)
    truth = np.asarray(normalized_truth, dtype=float).reshape(-1)
    ea = np.asarray(ea, dtype=float).reshape(-1)
    labels = np.asarray(column, dtype=str).reshape(-1)
    names = tuple(str(value) for value in columns)
    if not (source.shape == truth.shape == ea.shape == labels.shape) or len(source) < 4:
        raise ValueError("aligned hierarchical training arrays required")
    if len(set(names)) != len(names) or set(labels) - set(names):
        raise ValueError("column contract mismatch")
    if base_penalty < 0 or delta_penalty < 0:
        raise ValueError("penalties must be nonnegative")
    scale = _scale(source)
    u = source / scale
    ea_mean = float(ea.mean())
    interaction_raw = u * (ea - ea_mean)
    interaction_mean = float(interaction_raw.mean())
    interaction_std = _scale(interaction_raw)
    base = np.column_stack([u, np.ones(len(u)), (interaction_raw - interaction_mean) / interaction_std])
    design = np.zeros((len(u), 3 + 3 * len(names)), dtype=float)
    design[:, :3] = base
    for index, name in enumerate(names):
        selected = labels == name
        design[selected, 3 + 3 * index:6 + 3 * index] = base[selected]
    prior = np.zeros(design.shape[1], dtype=float)
    prior[0] = 1.0
    penalty = np.r_[np.repeat(np.sqrt(base_penalty), 3), np.repeat(np.sqrt(delta_penalty), 3 * len(names))]
    augmented = np.vstack([design / np.sqrt(len(u)), np.diag(penalty)])
    augmented_truth = np.r_[truth / scale / np.sqrt(len(u)), penalty * prior]
    coefficients = np.linalg.lstsq(augmented, augmented_truth, rcond=None)[0]
    return HierarchicalScalarFit(names, coefficients[:3], coefficients[3:].reshape(len(names), 3),
                                 scale, ea_mean, interaction_mean, interaction_std,
                                 float(base_penalty), float(delta_penalty))


@dataclass(frozen=True)
class HierarchicalCenterWidthFit:
    columns: tuple[str, ...]
    mass_ratios: dict[str, float]
    center: HierarchicalScalarFit
    width: HierarchicalScalarFit

    def predict_raw(self, source: np.ndarray, ea: np.ndarray, column: Sequence[str]) -> np.ndarray:
        labels = np.asarray(column, dtype=str)
        center, width = center_width_transform(source)
        ratios = np.asarray([self.mass_ratios[value] for value in labels], dtype=float)
        normalized_center = self.center.predict(center[:, 0], ea, labels)
        normalized_width = self.width.predict(width[:, 0], ea, labels)
        return center_width_inverse(ratios * normalized_center, ratios * normalized_width)

    def predict(self, source: np.ndarray, ea: np.ndarray, column: Sequence[str]) -> np.ndarray:
        return project_physical(self.predict_raw(source, ea, column))[0]

    def audit(self) -> dict[str, object]:
        return {"columns": list(self.columns), "mass_ratios": self.mass_ratios,
                "center": self.center.audit(), "width": self.width.audit()}


def fit_hierarchical_center_width(
    source: np.ndarray, truth: np.ndarray, ea: np.ndarray, column: Sequence[str],
    columns: Sequence[str], *, mass_ratios: dict[str, float], base_penalty: float,
    delta_penalty: float,
) -> HierarchicalCenterWidthFit:
    labels = np.asarray(column, dtype=str)
    names = tuple(str(value) for value in columns)
    if set(mass_ratios) != set(names):
        raise ValueError("one mass ratio is required for every fitted column")
    ratios = np.asarray([mass_ratios[value] for value in labels], dtype=float)
    if np.any(ratios <= 0):
        raise ValueError("mass ratios must be positive")
    source_center, source_width = center_width_transform(source)
    truth_center, truth_width = center_width_transform(truth)
    kwargs = dict(column=labels, columns=names, base_penalty=base_penalty, delta_penalty=delta_penalty)
    center = fit_hierarchical_scalar(source_center[:, 0], truth_center[:, 0] / ratios, ea, **kwargs)
    width = fit_hierarchical_scalar(source_width[:, 0], truth_width[:, 0] / ratios, ea, **kwargs)
    return HierarchicalCenterWidthFit(names, {key: float(value) for key, value in mass_ratios.items()}, center, width)


@dataclass(frozen=True)
class LatentRidgeFit:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    residual_scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    alpha: float

    def predict_residual(self, latent: np.ndarray) -> np.ndarray:
        values = (np.asarray(latent, dtype=float) - self.feature_mean) / self.feature_scale
        return (values @ self.coefficients.T + self.intercept) * self.residual_scale


def fit_latent_ridge(latent: np.ndarray, residual: np.ndarray, alpha: float) -> LatentRidgeFit:
    from sklearn.linear_model import Ridge
    latent = np.asarray(latent, dtype=float)
    residual = np.asarray(residual, dtype=float)
    if latent.ndim != 2 or residual.shape != (len(latent), 2) or alpha < 0:
        raise ValueError("aligned latent and two-column residual arrays required")
    mean, scale = latent.mean(axis=0), latent.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    residual_scale = residual.std(axis=0)
    residual_scale = np.where(residual_scale < 1e-8, 1.0, residual_scale)
    model = Ridge(alpha=float(alpha)).fit((latent - mean) / scale, residual / residual_scale)
    return LatentRidgeFit(mean, scale, residual_scale, model.coef_.copy(), model.intercept_.copy(), float(alpha))


class TinyLatentAdapter(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden = int(hidden_dim or (16 if latent_dim * 16 + 50 < 5000 else 8))
        self.network = nn.Sequential(nn.Linear(latent_dim, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 2))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)


def tiny_adapter_parameter_count(latent_dim: int, hidden_dim: int | None = None) -> int:
    return sum(parameter.numel() for parameter in TinyLatentAdapter(latent_dim, hidden_dim).parameters())


__all__ = ["HierarchicalCenterWidthFit", "LatentRidgeFit", "TinyLatentAdapter",
           "fit_hierarchical_center_width", "fit_hierarchical_scalar", "fit_latent_ridge",
           "tiny_adapter_parameter_count"]
