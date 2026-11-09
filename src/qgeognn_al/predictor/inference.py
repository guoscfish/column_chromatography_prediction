"""Six raw outputs in fixed canonical order; no label reads or sorting."""
import numpy as np
import torch
from .training import loader_pair


def predict_outputs(model, atom, angle, indices, batch_size=2048):
    model.eval()
    chunks, positions = [], []
    with torch.no_grad():
        for a, b in zip(*loader_pair(atom, angle, indices, batch_size)):
            chunks.append(model(a, b).cpu().numpy())
            positions.extend(a.canonical_position.cpu().numpy().reshape(-1).tolist())
    if positions != list(indices):
        raise RuntimeError('prediction order drift')
    values = np.vstack(chunks)
    if values.shape != (len(indices), 6) or not np.isfinite(values).all():
        raise ValueError('six finite raw outputs required per authorized row')
    return values
