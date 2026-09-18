"""Conditional, greedy integrated variance reduction on a fixed row kernel."""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def conditional_batch_ivr(
    features: np.ndarray,
    labeled_positions: np.ndarray,
    candidate_positions: np.ndarray,
    batch_size: int = 32,
) -> dict:
    """Use every reference row uniformly, unit prior and unit observation noise.

    A single RMS normalization makes the rule invariant to global feature scale.
    Positions refer to the fixed original outer-training universe, never test X.
    No targets or residuals enter this scalar row-kernel surrogate.
    """
    started = time.perf_counter()
    raw = np.asarray(features, dtype=np.float64)
    if raw.ndim != 2 or min(raw.shape) == 0 or not np.isfinite(raw).all():
        raise ValueError("features must be a finite nonempty matrix")
    labeled = np.asarray(labeled_positions)
    candidates = np.asarray(candidate_positions)
    for values in (labeled, candidates):
        if values.ndim != 1 or values.dtype.kind not in "iu":
            raise ValueError("positions must be one-dimensional integer arrays")
        if len(set(values.tolist())) != len(values) or np.any(values < 0) or np.any(values >= len(raw)):
            raise ValueError("positions must be unique and inside the reference universe")
    if set(labeled.tolist()) & set(candidates.tolist()):
        raise ValueError("labeled and candidate positions must be disjoint")
    if set(labeled.tolist()) | set(candidates.tolist()) != set(range(len(raw))):
        raise ValueError("L and U must partition the fixed reference universe")
    if not isinstance(batch_size, int) or not 1 <= batch_size <= len(candidates):
        raise ValueError("invalid batch size")
    mean_sq_norm = float(np.mean(np.einsum("ij,ij->i", raw, raw)))
    if not np.isfinite(mean_sq_norm) or mean_sq_norm <= 0:
        raise ValueError("reference feature scale must be positive and finite")
    x = raw / np.sqrt(mean_sq_norm)
    precision = np.eye(x.shape[1]) + x[labeled].T @ x[labeled]
    covariance = cho_solve(cho_factor(precision, lower=True), np.eye(x.shape[1]))
    moment = x.T @ x / len(x)
    q = x @ covariance
    h = q @ moment
    available = np.ones(len(candidates), dtype=bool)
    selected, trace = [], []
    for step in range(batch_size):
        variance = np.einsum("ij,ij->i", q, x)
        numerator = np.einsum("ij,ij->i", q, h)
        if not np.isfinite(variance).all() or min(variance) < -1e-9 or min(numerator) < -1e-9:
            raise RuntimeError("conditional covariance lost numerical validity")
        scores = np.maximum(numerator[candidates], 0) / (1 + np.maximum(variance[candidates], 0))
        scores[~available] = -np.inf
        # np.argmax resolves exact ties by the original candidate order.
        winner = int(np.argmax(scores))
        position = int(candidates[winner])
        reduction = float(scores[winner])
        before = float(variance.mean())
        denominator = 1 + float(variance[position])
        v, hm = q[position].copy(), h[position].copy()
        cross = q @ x[position]
        # Sherman-Morrison updates both projected factors in O(n*d) per pick.
        q -= np.outer(cross, v) / denominator
        h -= np.outer(cross, hm) / denominator
        after = float(np.einsum("ij,ij->", q, x) / len(x))
        if not np.isclose(before - after, reduction, rtol=1e-7, atol=1e-11):
            raise RuntimeError("IVR score differs from realized covariance reduction")
        trace.append({"step": step + 1, "reference_position": position,
                      "candidate_position": winner, "score": reduction,
                      "conditional_variance": float(variance[position]),
                      "integrated_variance_before": before, "integrated_variance_after": after})
        selected.append(winner)
        available[winner] = False
    return {"selected_candidate_positions": selected, "trace": trace,
            "audit": {"reference_rows": len(x), "labeled_rows": len(labeled),
                      "candidate_rows": len(candidates), "feature_dimension": x.shape[1],
                      "raw_mean_squared_norm": mean_sq_norm, "prior_precision": 1.0,
                      "observation_noise_variance": 1.0, "batch_covariance_updates": batch_size,
                      "elapsed_seconds": time.perf_counter() - started}}
