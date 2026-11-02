"""Epistemic disagreement of three independently initialized current V2 models."""

import numpy as np


def ensemble_uncertainty_select(predictions, scales, batch_size):
    values = np.asarray(predictions, dtype=np.float64)
    scale = np.asarray(scales, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != 3 or values.shape[2] != 6:
        raise ValueError("expected K=3 by pool rows by six outputs")
    if scale.shape != (2,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("two positive finite L0 scales required")
    if not np.isfinite(values).all() or not 0 < batch_size <= values.shape[1]:
        raise ValueError("invalid predictions or batch size")
    disagreement = values[:, :, [1, 4]].std(axis=0, ddof=0)
    scores = np.linalg.norm(disagreement / scale, axis=1)
    return np.argsort(-scores, kind="stable")[:batch_size], scores
