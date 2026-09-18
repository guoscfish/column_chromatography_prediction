import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data

from src.qgeognn_al.active_learning_v2.gradient_bank import extract_gradient_bank, fold_sketch, project_pair
from src.qgeognn_al.active_learning_v2.gradient_features import countsketch_mapping, extract_q50_gradient_sketches


def test_nested_maps_match_legacy_maps_at_every_dimension():
    for seed in (19, 214, 4000194):
        large = countsketch_mapping(37, 2048, seed)
        for dimension in (256, 512, 1024, 2048):
            buckets, signs = countsketch_mapping(37, dimension, seed)
            assert torch.equal(large[0] // (2048 // dimension), buckets)
            assert torch.equal(large[1], signs)
            gradients = np.random.default_rng(90).normal(size=(2, 37))
            scalar, block = project_pair(gradients, *(t.numpy() for t in large), 2048)
            direct, direct_block = project_pair(gradients, buckets.numpy(), signs.numpy(), dimension)
            np.testing.assert_allclose(fold_sketch(scalar, dimension), direct, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(fold_sketch(block, dimension), direct_block, rtol=1e-5, atol=1e-6)


class SharedSixOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1., 2.]))

    def forward(self, a, b):
        v = a.x @ self.weight
        return torch.stack([v, v, v, 2*v, 2*v, 2*v], dim=-1)


def test_shared_parameter_mapping_preserves_endpoint_cross_information():
    atom = [Data(x=torch.tensor([[1., 3.]]), y=torch.zeros((1, 2)))]
    angle = [Data(x=torch.zeros((1, 1)))]
    model = SharedSixOutput()
    bank, _ = extract_gradient_bank(model, atom, angle, [0], (1., 1.), [7], dimension=512)
    np.testing.assert_array_equal(bank['block'][0, 0, 1], 2 * bank['block'][0, 0, 0])
    original = extract_q50_gradient_sketches(model, atom, angle, [0], (1., 1.), sketch_seed=7)
    np.testing.assert_allclose(bank['scalar'][0], original.features, atol=1e-7)
