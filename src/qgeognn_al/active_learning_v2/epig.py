"""Exact Gaussian prediction-oriented information gain on a declared posterior.

The target label is a *future replicate*: observation noise is independent of
candidate noise even when the target and candidate have the same input/row ID.
All covariance algebra is float64.  A posterior supplies ``row_ids``,
``n_outputs`` (or ``mean``), ``noise_cov`` or ``noise_blocks(positions)``,
``cov(left_positions, right_positions, noisy=False)`` in row-major flat block
layout, and preferably ``latent_diag(positions)``.  It must already be fitted
using legally revealed training labels; this module never receives labels.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def _symmetric(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim < 2 or value.shape[-1] != value.shape[-2] or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite square matrices")
    scale = max(float(np.max(np.abs(value))), 1.0)
    if np.max(np.abs(value - np.swapaxes(value, -1, -2))) > 1e-10 * scale:
        raise ValueError(f"{name} is asymmetric")
    return (value + np.swapaxes(value, -1, -2)) * 0.5


def gaussian_pair_information(
    candidate_label_cov: np.ndarray,
    target_label_cov: np.ndarray,
    target_candidate_cross: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mutual information and trace variance reduction for block pairs.

    Inputs broadcast over leading dimensions; the last two dimensions hold
    outputs, and candidate/target output counts may differ.  Marginals already
    include observation noise, while the cross block contains latent covariance
    only for independent future labels.  No jitter, clipping of correlations,
    or repair of an invalid Gaussian is performed.
    """
    sxx = _symmetric(candidate_label_cov, "candidate_label_cov")
    stt = _symmetric(target_label_cov, "target_label_cov")
    stx = np.asarray(target_candidate_cross, dtype=np.float64)
    if stx.ndim < 2 or not np.isfinite(stx).all():
        raise ValueError("cross covariance must be finite")
    if stx.shape[-2:] != (stt.shape[-1], sxx.shape[-1]):
        raise ValueError("cross covariance output dimensions do not match")
    lx = np.linalg.cholesky(sxx)
    lt = np.linalg.cholesky(stt)
    if sxx.shape[-1] <= 2 and stt.shape[-1] <= 2:
        # Explicit triangular solves avoid millions of tiny LAPACK calls at
        # M=2048. This is Cholesky algebra, not a covariance inverse. log1p
        # preserves very small information gains near independent labels.
        b = np.empty_like(stx)
        b[..., :, 0] = stx[..., :, 0] / lx[..., 0, 0][..., None]
        if sxx.shape[-1] == 2:
            b[..., :, 1] = ((stx[..., :, 1] - b[..., :, 0] * lx[..., 1, 0][..., None]) /
                            lx[..., 1, 1][..., None])
        whitened = np.empty_like(b)
        whitened[..., 0, :] = b[..., 0, :] / lt[..., 0, 0][..., None]
        a = np.square(whitened[..., 0, :]).sum(axis=-1)
        if stt.shape[-1] == 1:
            loss = a
        else:
            whitened[..., 1, :] = ((b[..., 1, :] - whitened[..., 0, :] * lt[..., 1, 0][..., None]) /
                                    lt[..., 1, 1][..., None])
            d = np.square(whitened[..., 1, :]).sum(axis=-1)
            off = (whitened[..., 0, :] * whitened[..., 1, :]).sum(axis=-1)
            # det(I-WW^T)=1-(a+d-a*d+off^2). Sylvester's
            # criterion additionally checks its leading principal minor.
            loss = a + d - a*d + off*off
        if np.any(a >= 1) or np.any(loss >= 1) or np.any(loss < -1e-12):
            raise np.linalg.LinAlgError("Gaussian pair has nonpositive Schur complement")
        return -0.5 * np.log1p(-np.maximum(loss, 0)), np.square(b).sum(axis=(-2, -1))
    # B = Cov(Y_t,Y_x) L_x^-T, so B B^T is the Schur reduction.
    b = np.swapaxes(np.linalg.solve(lx, np.swapaxes(stx, -1, -2)), -1, -2)
    whitened = np.linalg.solve(lt, b)
    identity = np.eye(stt.shape[-1], dtype=np.float64)
    complement = _symmetric(identity - whitened @ np.swapaxes(whitened, -1, -2), "whitened Schur complement")
    lc = np.linalg.cholesky(complement)
    information = -np.log(np.diagonal(lc, axis1=-2, axis2=-1)).sum(axis=-1)
    reduction = np.square(b).sum(axis=(-2, -1))
    if np.any(information < -1e-10):
        raise np.linalg.LinAlgError("Gaussian information became negative")
    return np.maximum(information, 0.0), reduction


def _positions(values: Any, size: int, name: str, *, unique: bool = True) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1 or (arr.size and arr.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer position array")
    arr = arr.astype(np.int64)
    if np.any(arr < 0) or np.any(arr >= size):
        raise ValueError(f"{name} lies outside the declared training universe")
    if unique and len(np.unique(arr)) != len(arr):
        raise ValueError(f"{name} contains duplicate positions")
    return arr


class GaussianAcquisitionState:
    """Cached target/candidate covariance with exact rank-q batch updates.

    Memory is O(M*C*q^2), not O((M+C)^2*q^2). Candidate-to-winner covariance
    is requested once per greedy step. No candidate-to-candidate dense matrix
    is stored, and no probabilistic model is refit during a batch.
    """

    def __init__(self, posterior: Any, candidate_positions: Any,
                 reference_positions: Any, *, reference_weights: Any = None,
                 output_indices: Any = None, chunk_size: int = 128,
                 max_cache_bytes: int = 768 * 1024 ** 2):
        self.posterior = posterior
        self.row_ids = np.asarray(posterior.row_ids)
        if self.row_ids.ndim != 1 or len(set(self.row_ids.tolist())) != len(self.row_ids):
            raise ValueError("posterior.row_ids must be a unique one-dimensional training universe")
        self.n = len(self.row_ids)
        self.original_q = int(getattr(posterior, "n_outputs", np.asarray(getattr(posterior, "mean", [])).shape[-1]))
        if self.original_q < 1:
            raise ValueError("posterior must declare positive n_outputs")
        self.outputs = (_positions(output_indices, self.original_q, "output_indices")
                        if output_indices is not None else np.arange(self.original_q))
        if not len(self.outputs):
            raise ValueError("at least one output must be selected")
        self.q = len(self.outputs)
        self.candidates = _positions(candidate_positions, self.n, "candidate_positions")
        self.references = _positions(reference_positions, self.n, "reference_positions", unique=False)
        if not len(self.candidates) or not len(self.references):
            raise ValueError("candidate and target reference sets must be nonempty")
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self.chunk_size = chunk_size
        m, c, q = len(self.references), len(self.candidates), self.q
        bytes_required = m * c * q * q * 8
        if bytes_required > max_cache_bytes:
            raise MemoryError(f"target/candidate covariance requires {bytes_required} bytes, budget is {max_cache_bytes}")
        weights = (np.ones(m, dtype=np.float64) if reference_weights is None
                   else np.asarray(reference_weights, dtype=np.float64))
        if weights.shape != (m,) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("reference weights must be finite nonnegative masses with positive total")
        self.weights = weights / weights.sum()
        self.candidate_noise = self._noise(self.candidates)
        self.reference_noise = self._noise(self.references)
        self.candidate_latent = self._diag(self.candidates)
        self.reference_latent = self._diag(self.references)
        self.cross = np.empty((m * q, c * q), dtype=np.float64)
        for start in range(0, c, chunk_size):
            stop = min(start + chunk_size, c)
            self.cross[:, start*q:stop*q] = self._cov(self.references, self.candidates[start:stop])
        self.available = np.ones(c, dtype=bool)
        self.candidate_updates: list[np.ndarray] = []
        self.reference_updates: list[np.ndarray] = []
        self.selected_indices: list[int] = []
        self.cache_bytes = bytes_required

    def _cov(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.posterior.cov(left, right, noisy=False), dtype=np.float64)
        expected = (len(left) * self.original_q, len(right) * self.original_q)
        if raw.shape != expected or not np.isfinite(raw).all():
            raise ValueError("posterior.cov returned invalid shape or nonfinite values")
        blocks = raw.reshape(len(left), self.original_q, len(right), self.original_q)
        return blocks[:, self.outputs, :, :][:, :, :, self.outputs].reshape(len(left)*self.q, len(right)*self.q)

    def _diag(self, positions: np.ndarray) -> np.ndarray:
        if hasattr(self.posterior, "latent_diag"):
            raw = np.asarray(self.posterior.latent_diag(positions), dtype=np.float64)
            if raw.shape != (len(positions), self.original_q, self.original_q):
                raise ValueError("posterior.latent_diag returned invalid shape")
            raw = raw[:, self.outputs, :][:, :, self.outputs]
        else:
            # Compatibility path deliberately avoids a dense N-by-N covariance.
            raw = np.stack([self._cov(np.array([i]), np.array([i])) for i in positions])
        raw = _symmetric(raw, "latent diagonal blocks")
        if np.min(np.linalg.eigvalsh(raw)) < -1e-9 * max(1., float(np.max(np.abs(raw)))):
            raise np.linalg.LinAlgError("latent covariance has negative variance")
        return raw

    def _noise(self, positions: np.ndarray) -> np.ndarray:
        if hasattr(self.posterior, "noise_blocks"):
            raw = np.asarray(self.posterior.noise_blocks(positions), dtype=np.float64)
        else:
            raw = np.broadcast_to(np.asarray(self.posterior.noise_cov, dtype=np.float64),
                                  (len(positions), self.original_q, self.original_q))
        if raw.shape != (len(positions), self.original_q, self.original_q):
            raise ValueError("posterior noise blocks have invalid shape")
        raw = _symmetric(raw[:, self.outputs, :][:, :, self.outputs], "observation noise")
        if np.min(np.linalg.eigvalsh(raw)) < -1e-12:
            raise np.linalg.LinAlgError("observation noise must be positive semidefinite")
        return raw

    def scores(self, objective: str = "epig") -> np.ndarray:
        if objective not in {"epig", "ivr"}:
            raise ValueError("objective must be 'epig' or 'ivr'")
        c, m, q = len(self.candidates), len(self.references), self.q
        result = np.full(c, -np.inf, dtype=np.float64)
        target = self.reference_latent + self.reference_noise
        for start in range(0, c, self.chunk_size):
            stop = min(c, start + self.chunk_size)
            active = np.flatnonzero(self.available[start:stop]) + start
            if not len(active):
                continue
            cols = (active[:, None] * q + np.arange(q)).reshape(-1)
            # [C,M,q_target,q_candidate]
            cross = self.cross[:, cols].reshape(m, q, len(active), q).transpose(2, 0, 1, 3)
            info, reduction = gaussian_pair_information(
                (self.candidate_latent[active] + self.candidate_noise[active])[:, None],
                target[None], cross)
            result[active] = (info if objective == "epig" else reduction) @ self.weights
        return result

    def condition(self, candidate_index: int) -> None:
        """Condition on a candidate's future noisy observation without a label."""
        if not isinstance(candidate_index, (int, np.integer)) or not 0 <= candidate_index < len(self.candidates):
            raise ValueError("candidate_index is outside the candidate set")
        if not self.available[candidate_index]:
            raise ValueError("candidate was already selected")
        q = self.q
        lc = np.linalg.cholesky(_symmetric(self.candidate_latent[candidate_index] +
                                          self.candidate_noise[candidate_index], "selected label covariance"))
        winner = self.candidates[candidate_index:candidate_index+1]
        ccross = self._cov(self.candidates, winner).reshape(len(self.candidates), q, q)
        for update in self.candidate_updates:
            ccross -= update @ update[candidate_index].T
        rcross = self.cross[:, candidate_index*q:(candidate_index+1)*q].reshape(len(self.references), q, q).copy()
        cu = np.swapaxes(np.linalg.solve(lc, np.swapaxes(ccross, -1, -2)), -1, -2)
        ru = np.swapaxes(np.linalg.solve(lc, np.swapaxes(rcross, -1, -2)), -1, -2)
        self.candidate_latent -= cu @ np.swapaxes(cu, -1, -2)
        self.reference_latent -= ru @ np.swapaxes(ru, -1, -2)
        self.cross -= ru.reshape(-1, q) @ cu.reshape(-1, q).T
        self.candidate_updates.append(cu)
        self.reference_updates.append(ru)
        self.selected_indices.append(int(candidate_index))
        self.available[candidate_index] = False

    def integrated_label_variance(self) -> float:
        return float(self.weights @ np.trace(self.reference_latent + self.reference_noise, axis1=1, axis2=2))


def score_gaussian_acquisition(posterior: Any, candidate_positions: Any,
                               reference_positions: Any, *, objective: str = "epig",
                               selected_positions: Any = (), reference_weights: Any = None,
                               output_indices: Any = None, chunk_size: int = 128,
                               max_cache_bytes: int = 768 * 1024 ** 2) -> np.ndarray:
    """Score candidate positions, optionally conditioning selected candidates."""
    state = GaussianAcquisitionState(posterior, candidate_positions, reference_positions,
                                    reference_weights=reference_weights, output_indices=output_indices,
                                    chunk_size=chunk_size, max_cache_bytes=max_cache_bytes)
    selected = _positions(selected_positions, state.n, "selected_positions")
    lookup = {int(pos): i for i, pos in enumerate(state.candidates)}
    for position in selected:
        if int(position) not in lookup:
            raise ValueError("selected_positions must belong to the candidate set")
        state.condition(lookup[int(position)])
    return state.scores(objective)


def select_gaussian_batch(posterior: Any, candidate_positions: Any,
                          reference_positions: Any, *, objective: str = "epig",
                          batch_size: int = 32, batch_mode: str = "conditional_greedy",
                          labeled_positions: Any = (), reference_weights: Any = None,
                          output_indices: Any = None, chunk_size: int = 128,
                          max_cache_bytes: int = 768 * 1024 ** 2) -> dict:
    """Select an exact Gaussian EPIG/IVR batch, with distinct method names.

    ``conditional_greedy`` recomputes conditional MI/IVR after each selected
    observation; ``pointwise_topk`` ranks original one-point scores and merely
    measures the information actually obtained by that frozen batch.
    """
    started = time.perf_counter()
    state = GaussianAcquisitionState(posterior, candidate_positions, reference_positions,
                                    reference_weights=reference_weights, output_indices=output_indices,
                                    chunk_size=chunk_size, max_cache_bytes=max_cache_bytes)
    labeled = _positions(labeled_positions, state.n, "labeled_positions")
    if np.intersect1d(labeled, state.candidates).size:
        raise ValueError("candidates include already labeled rows")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or not 1 <= batch_size <= len(state.candidates):
        raise ValueError("batch_size must lie within the candidate count")
    if batch_mode not in {"conditional_greedy", "pointwise_topk"}:
        raise ValueError("unknown batch_mode")
    initial_scores = state.scores(objective)
    pointwise_order = np.argsort(-initial_scores, kind="stable")
    trace = []
    for step in range(batch_size):
        scores = initial_scores if step == 0 else state.scores(objective)
        index = int(np.argmax(scores) if batch_mode == "conditional_greedy" else pointwise_order[step])
        before = state.integrated_label_variance()
        state.condition(index)
        after = state.integrated_label_variance()
        if objective == "ivr" and not np.isclose(before - after, scores[index], rtol=1e-7, atol=1e-10):
            raise RuntimeError("IVR score disagrees with realized variance reduction")
        trace.append({"step": step+1, "candidate_index": index,
                      "row_position": int(state.candidates[index]),
                      "conditional_score": float(scores[index]),
                      "original_pointwise_score": float(initial_scores[index]),
                      "integrated_label_variance_before": before,
                      "integrated_label_variance_after": after})
    selected = state.candidates[state.selected_indices]
    return {"method": f"gaussian_{objective}_{batch_mode}",
            "selected_candidate_positions": state.selected_indices,
            "selected_positions": selected.tolist(),
            "selected_row_ids": state.row_ids[selected].tolist(),
            "initial_scores": initial_scores.tolist(), "trace": trace,
            "audit": {"dtype": "float64", "candidate_rows": len(state.candidates),
                      "reference_rows": len(state.references), "outputs": state.outputs.tolist(),
                      "target_label_semantics": "independent_future_replicate",
                      "cross_covariance": "latent_only", "cache_bytes": state.cache_bytes,
                      "conditional_batch_information_nats": (sum(t["conditional_score"] for t in trace)
                                                             if objective == "epig" else None),
                      "elapsed_seconds": time.perf_counter() - started}}
