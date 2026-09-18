"""One Jacobian pass, several nested CountSketch maps, no target access."""

from __future__ import annotations

import time
import numpy as np
import torch

from ..training.predictor import loader_pair
from .gradient_features import countsketch_mapping


def fold_sketch(values, dimension):
    """Power-of-two buckets coarsen by integer division, not a second sketch."""
    maximum = values.shape[-1]
    if dimension < 1 or maximum % dimension:
        raise ValueError("dimension must divide the bank dimension")
    return values.reshape(*values.shape[:-1], dimension, maximum // dimension).sum(axis=-1)


def project_pair(gradients, buckets, signs, dimension):
    """Float64 accumulation preserves small directions before storing float32."""
    p = gradients.shape[1]
    first = np.bincount(buckets[:p], weights=gradients[0] * signs[:p], minlength=dimension)
    second = np.bincount(buckets[p:], weights=gradients[1] * signs[p:], minlength=dimension)
    shared_second = np.bincount(buckets[:p], weights=gradients[1] * signs[:p], minlength=dimension)
    return (first + second).astype(np.float32), np.stack([first, shared_second]).astype(np.float32)


def extract_gradient_bank(model, atom, angle, indices, scales, sketch_seeds, *, dimension=2048, progress=None):
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0 or len(np.unique(indices)) != len(indices):
        raise ValueError("nonempty unique row positions required")
    if len(scales) != 2 or not np.isfinite(scales).all() or min(scales) <= 0:
        raise ValueError("two positive L0 endpoint scales required")
    if len(set(sketch_seeds)) != len(sketch_seeds) or not sketch_seeds:
        raise ValueError("unique sketch seeds required")
    if dimension < 1 or dimension & (dimension - 1):
        raise ValueError("bank dimension must be a power of two")
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    count = sum(p.numel() for p in parameters)
    maps = [tuple(v.numpy() for v in countsketch_mapping(count, dimension, seed)) for seed in sketch_seeds]
    scalar = np.empty((len(maps), len(indices), dimension), dtype=np.float32)
    block = np.empty((len(maps), len(indices), 2, dimension), dtype=np.float32)
    started = time.perf_counter()
    model.eval()
    for row, (a, b) in enumerate(zip(*loader_pair(atom, angle, indices, batch_size=1))):
        if torch.count_nonzero(a.y):
            raise RuntimeError("graph carries non-sentinel labels")
        output = model(a, b)
        if output.shape != (1, 6):
            raise ValueError("six-output model required")
        pair = []
        for endpoint, column in enumerate((1, 4)):
            gradients = torch.autograd.grad(output[0, column] / scales[endpoint], parameters,
                                            retain_graph=endpoint == 0, allow_unused=True)
            pair.append(torch.cat([torch.zeros(p.numel()) if g is None else g.detach().cpu().flatten()
                                   for p, g in zip(parameters, gradients)]).numpy())
        pair = np.asarray(pair)
        for j, (buckets, signs) in enumerate(maps):
            scalar[j, row], block[j, row] = project_pair(pair, buckets, signs, dimension)
        if progress and (row == 0 or (row + 1) % 250 == 0 or row + 1 == len(indices)):
            progress(dict(completed_rows=row + 1, total_rows=len(indices), elapsed_seconds=time.perf_counter() - started))
    if not np.isfinite(scalar).all() or not np.isfinite(block).all():
        raise RuntimeError("non-finite gradient bank")
    return dict(scalar=scalar, block=block, canonical_indices=indices,
                sketch_seeds=np.array(sketch_seeds, dtype=np.int64)), dict(
                    elapsed_seconds=time.perf_counter() - started, parameter_count=count,
                    representation="legacy concat and shared-parameter endpoint blocks",
                    label_access_count=0, new_fits=0, n_by_p_materialized=False)
