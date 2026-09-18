"""Block-observation IVR and BAIT-style forward/backward batch optimization.

The reference universe, endpoint risk weights, prior, and noise remain fixed
throughout a batch. A block is one indivisible experiment, not one endpoint.
"""

from __future__ import annotations

import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def normalize_blocks(features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 2:
        x = x[:, None, :]
    if x.ndim != 3 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("finite nonempty N x outputs x features required")
    scale = float(np.mean(np.sum(x * x, axis=(1, 2))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("positive finite reference energy required")
    return x / np.sqrt(scale), scale


def validate_partition(n, labeled, candidates, batch_size):
    l, u = np.asarray(labeled), np.asarray(candidates)
    for values in (l, u):
        if values.ndim != 1 or values.dtype.kind not in "iu":
            raise ValueError("integer position vectors required")
        if len(np.unique(values)) != len(values) or np.any(values < 0) or np.any(values >= n):
            raise ValueError("unique in-range positions required")
    if len(l) + len(u) != n or len(np.unique(np.r_[l, u])) != n:
        raise ValueError("L and U must partition the fixed reference universe")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= len(u):
        raise ValueError("invalid batch size")
    return l, u


def posterior_covariance(x, labeled):
    flat = x[np.asarray(labeled, dtype=int)].reshape(-1, x.shape[-1])
    precision = np.eye(x.shape[-1]) + flat.T @ flat
    return cho_solve(cho_factor(precision, lower=True), np.eye(x.shape[-1]))


def covariance_update(covariance, block, *, remove=False):
    """Woodbury add/downdate, also used as a direct numerical audit oracle."""
    q = block @ covariance
    direction = -1 if remove else 1
    small = np.eye(len(block)) + direction * q @ block.T
    result = covariance - direction * q.T @ cho_solve(cho_factor(small, lower=True), q)
    return (result + result.T) / 2


def integrated_risk(x, covariance):
    return float(np.einsum("ned,ned->", x @ covariance, x) / len(x))


class BlockPosterior:
    """Maintain projected covariance factors in O(N * d * outputs) per pick."""

    def __init__(self, x, labeled):
        self.x = x
        flat = x.reshape(-1, x.shape[-1])
        self.moment = flat.T @ flat / len(x)
        self.q = x @ posterior_covariance(x, labeled)
        self.h = self.q @ self.moment

    def risk(self):
        return float(np.einsum("ned,ned->", self.q, self.x) / len(self.x))

    def scores(self, positions, *, remove=False):
        q, h, x = self.q[positions], self.h[positions], self.x[positions]
        direction = -1 if remove else 1
        small = np.eye(x.shape[1]) + direction * np.einsum("ned,nfd->nef", q, x)
        small = (small + small.swapaxes(-1, -2)) / 2
        numerator = np.einsum("ned,nfd->nef", q, h)
        # Cholesky rejects invalid downdates rather than silently clipping them.
        np.linalg.cholesky(small)
        values = np.trace(np.linalg.solve(small, numerator), axis1=1, axis2=2)
        if not np.isfinite(values).all() or np.min(values) < -1e-10:
            raise RuntimeError("invalid conditional risk reduction")
        return np.maximum(values, 0)

    def update(self, position, *, remove=False):
        q, h, x = self.q[position].copy(), self.h[position].copy(), self.x[position]
        direction = -1 if remove else 1
        small = np.eye(len(x)) + direction * q @ x.T
        factor = cho_factor((small + small.T) / 2, lower=True)
        cross = self.q @ x.T
        self.q -= direction * cross @ cho_solve(factor, q)
        self.h -= direction * cross @ cho_solve(factor, h)


def block_batch_ivr(features, labeled_positions, candidate_positions, batch_size=32, *, backward=False):
    """Unit-noise IVR; optional forward min(2B, |U|), backward to exactly B.

    Scalar input reproduces the historical surrogate. Multi-output input uses
    a single common RMS of each experiment's Frobenius norm, so the expected
    information trace per experiment is one in both representations.
    """
    started = time.perf_counter()
    x, scale = normalize_blocks(features)
    labeled, candidates = validate_partition(len(x), labeled_positions, candidate_positions, batch_size)
    state = BlockPosterior(x, labeled)
    initial_risk = state.risk()
    selected, trace = [], []
    available = np.ones(len(candidates), dtype=bool)
    initial_scores = state.scores(candidates)
    forward_count = min(2 * batch_size, len(candidates)) if backward else batch_size
    for step in range(forward_count):
        scores = initial_scores.copy() if step == 0 else state.scores(candidates)
        scores[~available] = -np.inf
        winner = int(np.argmax(scores))
        before = state.risk()
        state.update(int(candidates[winner]))
        after = state.risk()
        if not np.isclose(before - after, scores[winner], rtol=1e-7, atol=1e-11):
            raise RuntimeError("forward score/update mismatch")
        selected.append(winner)
        available[winner] = False
        trace.append(dict(phase="forward", candidate_position=winner, score=float(scores[winner]),
                          risk_before=before, risk_after=after))
    forward_b = selected[:batch_size].copy()
    while len(selected) > batch_size:
        # Stable original candidate order for exact backward ties.
        ordered = np.array(sorted(selected), dtype=int)
        scores = state.scores(candidates[ordered], remove=True)
        at = int(np.argmin(scores))
        winner = int(ordered[at])
        before = state.risk()
        state.update(int(candidates[winner]), remove=True)
        after = state.risk()
        if not np.isclose(after - before, scores[at], rtol=1e-7, atol=1e-11):
            raise RuntimeError("backward score/update mismatch")
        selected.remove(winner)
        trace.append(dict(phase="backward", candidate_position=winner, score=float(scores[at]),
                          risk_before=before, risk_after=after))
    direct = posterior_covariance(x, np.r_[labeled, candidates[selected]])
    error = float(np.max(np.abs(state.q - x @ direct)))
    if not np.allclose(state.q, x @ direct, rtol=1e-7, atol=1e-9):
        raise RuntimeError("incremental factors disagree with direct posterior")
    endpoint_risk = np.einsum("ned,ned->e", state.q, x) / len(x)
    return dict(selected_candidate_positions=selected, forward_b_candidate_positions=forward_b,
                initial_scores=initial_scores, trace=trace,
                audit=dict(reference_rows=len(x), outputs=x.shape[1], feature_dimension=x.shape[2],
                           raw_mean_squared_norm=scale, initial_risk=initial_risk, final_risk=state.risk(),
                           endpoint_final_risk=endpoint_risk.tolist(), direct_factor_max_error=error,
                           elapsed_seconds=time.perf_counter() - started))
