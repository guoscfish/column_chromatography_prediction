"""Synthetic graph tensors only; never opens a dataset or label store."""
import torch
from torch_geometric.data import Data


def graph_pair(row=0):
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    continuous = torch.tensor([[1.25, .1, .2, .3, .4, .5, .6, float(row % 3), .7, .8]]).repeat(4, 1)
    atom = Data(x=torch.ones((3, 9), dtype=torch.long), edge_index=edges,
                edge_attr=torch.cat([torch.tensor([[1., 1., 0.]]).repeat(4, 1), continuous], dim=1),
                y=torch.zeros((1, 2)), canonical_position=torch.tensor(row), num_nodes=3)
    angles = torch.arange(6 * 17, dtype=torch.float32).reshape(6, 17) / 100
    angle = Data(edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
                 edge_attr=angles, num_nodes=4)
    return atom, angle
