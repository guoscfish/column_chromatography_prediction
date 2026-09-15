from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data

from src.qgeognn_al.active_learning_v2.gradient_features import (
    countsketch_mapping,
    extract_q50_gradient_sketches,
    sketch_flat_gradient,
)


class TinySixOutput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 6)

    def forward(self, atom, angle):
        del angle
        return self.linear(atom.x.reshape(-1, 1).float())


def _graphs():
    atom = [
        Data(x=torch.tensor([[value]]), y=torch.zeros((1, 2)), canonical_position=torch.tensor(index))
        for index, value in enumerate((1.0, 2.0, 4.0))
    ]
    angle = [Data(x=torch.zeros((1, 1))) for _ in atom]
    return atom, angle


def test_countsketch_mapping_and_projection_are_deterministic() -> None:
    first = countsketch_mapping(17, 8, 91)
    second = countsketch_mapping(17, 8, 91)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    gradient = torch.arange(17, dtype=torch.float32)
    assert torch.equal(
        sketch_flat_gradient(gradient, 0, *first, 8),
        sketch_flat_gradient(gradient, 0, *second, 8),
    )


def test_full_trainable_toy_gradient_sketch_is_repeatable_and_finite() -> None:
    torch.manual_seed(7)
    model = TinySixOutput()
    atom, angle = _graphs()
    kwargs = dict(
        model=model,
        atom_data=atom,
        angle_data=angle,
        indices=[0, 1, 2],
        scales=(2.0, 3.0),
        dimension=16,
        sketch_seed=123,
    )
    first = extract_q50_gradient_sketches(**kwargs)
    second = extract_q50_gradient_sketches(**kwargs)
    assert np.array_equal(first.features, second.features)
    assert first.features.shape == (3, 16)
    assert np.isfinite(first.features).all()
    assert first.audit["parameter_count"] == sum(p.numel() for p in model.parameters())
    assert first.audit["n_by_p_materialized"] is False
