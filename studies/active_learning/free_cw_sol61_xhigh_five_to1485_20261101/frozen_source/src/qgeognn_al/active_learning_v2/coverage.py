"""Current-V2 latent-space training-pool greedy k-center."""

import numpy as np
import torch

from ..training.predictor import loader_pair
from .lcmd import _squared_distances


def extract_representations(model, atom, angle, indices, batch_size=2048):
    model.eval()
    chunks, positions = [], []
    with torch.no_grad():
        for a, b in zip(*loader_pair(atom, angle, indices, batch_size)):
            chunks.append(model.extract_representation(a, b).cpu().numpy())
            positions.extend(a.canonical_position.cpu().numpy().reshape(-1).tolist())
    if positions != list(indices):
        raise RuntimeError("latent representation order drift")
    result = np.vstack(chunks)
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite latent representation")
    return result


def coreset_tp_select(pool_features, train_features, batch_size):
    pool, train = np.asarray(pool_features), np.asarray(train_features)
    if pool.ndim != 2 or train.ndim != 2 or pool.shape[1] != train.shape[1] or not len(train):
        raise ValueError("aligned pool and nonempty L0 matrices required")
    if not np.isfinite(pool).all() or not np.isfinite(train).all() or not 0 < batch_size <= len(pool):
        raise ValueError("invalid features or batch size")
    distance = _squared_distances(pool, train).min(axis=1)
    available = np.ones(len(pool), dtype=bool)
    selected = []
    for _ in range(batch_size):
        point = int(np.argmax(np.where(available, distance, -np.inf)))
        selected.append(point)
        available[point] = False
        distance = np.minimum(distance, _squared_distances(pool, pool[point:point + 1])[:, 0])
    return np.asarray(selected, dtype=np.int64)
