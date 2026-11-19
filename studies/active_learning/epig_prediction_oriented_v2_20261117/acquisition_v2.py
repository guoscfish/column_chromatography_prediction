"""Exploratory v2 acquisition: shared reference ordering, BALD and bounded RAM.

No labels are accepted. Generic BALD uses the existing exact Gaussian state.
The disk-backed state is restricted to verified independent-output posteriors;
it computes exact full-population EPIG/IVR without requesting a U x U matrix.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import time

import numpy as np

from qgeognn_al.active_learning_v2.epig import GaussianAcquisitionState


def coupled_reference_positions(seed, allowed_positions, row_ids, size=None):
    values = np.asarray(allowed_positions, dtype=int)
    if values.ndim != 1 or len(set(values.tolist())) != len(values):
        raise ValueError("reference population must contain distinct positions")
    if np.any(values < 0) or np.any(values >= len(row_ids)):
        raise ValueError("reference position outside row universe")
    if size is not None and (not isinstance(size, int) or size < 1):
        raise ValueError("reference size must be positive")
    ranked = sorted(values.tolist(), key=lambda i: hashlib.sha256(
        f"epig_v2_crn:{seed}:{row_ids[i]}".encode()).digest())
    return np.asarray(ranked if size is None else ranked[:size], dtype=int)


def bald_scores(state):
    """I(Y_x; f_x | D,Y_B), exact for independent Gaussian observation noise."""
    noise = state.candidate_noise
    latent = state.candidate_latent
    if latent.ndim == 2:
        if np.any(noise <= 0):
            raise np.linalg.LinAlgError("BALD requires strictly positive noise")
        values = .5 * np.log1p(latent / noise).sum(axis=-1)
    else:
        # log det(A)-log det(N), using Cholesky rather than an inverse.
        ln = np.linalg.cholesky(noise)
        ly = np.linalg.cholesky(latent + noise)
        values = (np.log(np.diagonal(ly, axis1=-2, axis2=-1)) -
                  np.log(np.diagonal(ln, axis1=-2, axis2=-1))).sum(axis=-1)
    values = np.asarray(values, dtype=np.float64)
    values[~state.available] = -np.inf
    return values


class GaussianStateWithBALD(GaussianAcquisitionState):
    def scores(self, objective="epig"):
        return bald_scores(self) if objective == "bald" else super().scores(objective)


class IndependentStreamingCache:
    """Read-only float64 latent cross-channel diagonals stored on disk.

    max_covariance_rows bounds every posterior.cov request on both axes. Full
    U x U covariance is never requested or resident in RAM. Independent output
    structure is checked for every generated block, rather than assumed.
    """
    def __init__(self, posterior, candidates, directory, *, target_chunk=64, candidate_chunk=256):
        self.posterior = posterior
        self.candidates = np.asarray(candidates, dtype=int)
        self.n = len(posterior.row_ids)
        self.q = int(posterior.n_outputs)
        if self.candidates.ndim != 1 or not len(self.candidates) or len(set(self.candidates.tolist())) != len(self.candidates):
            raise ValueError("candidate positions must be nonempty and unique")
        if np.any(self.candidates < 0) or np.any(self.candidates >= self.n):
            raise ValueError("candidate outside posterior row universe")
        self.target_chunk = int(target_chunk)
        self.candidate_chunk = int(candidate_chunk)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "latent_cross.f64"
        self.shape = (self.n, len(self.candidates), self.q)
        self.bytes = int(np.prod(self.shape) * 8)
        self.max_covariance_shape = [0, 0]
        started = time.perf_counter()
        self.cross = np.memmap(self.path, mode="w+", dtype=np.float64, shape=self.shape)
        for a in range(0, self.n, self.target_chunk):
            refs = np.arange(a, min(self.n, a+self.target_chunk))
            for b in range(0, len(self.candidates), self.candidate_chunk):
                cand = self.candidates[b:b+self.candidate_chunk]
                raw = posterior.cov(refs, cand, noisy=False).reshape(len(refs), self.q, len(cand), self.q)
                blocks = raw.transpose(0, 2, 1, 3)
                off = blocks.copy()
                idx = np.arange(self.q)
                off[..., idx, idx] = 0
                if np.any(np.abs(off) > 1e-12):
                    raise ValueError("streaming diagonal cache requires independent outputs")
                self.cross[a:a+len(refs), b:b+len(cand)] = np.diagonal(blocks, axis1=-2, axis2=-1)
                self.max_covariance_shape = [max(self.max_covariance_shape[0], len(refs)),
                                             max(self.max_covariance_shape[1], len(cand))]
        self.cross.flush()
        self.build_seconds = time.perf_counter()-started

    def state(self, references):
        return IndependentStreamingState(self, references)


class IndependentStreamingState:
    def __init__(self, cache, references):
        self.cache = cache
        self.posterior = cache.posterior
        self.candidates = cache.candidates
        self.references = np.asarray(references, dtype=int)
        if self.references.ndim != 1 or not len(self.references) or len(set(self.references.tolist())) != len(self.references):
            raise ValueError("reference positions must be nonempty and unique")
        if np.any(self.references < 0) or np.any(self.references >= cache.n):
            raise ValueError("invalid reference positions")
        self.q = cache.q
        self.available = np.ones(len(self.candidates), dtype=bool)
        self.selected_indices = []
        self.candidate_updates = []
        self.candidate_latent = self._diag(self.candidates)
        self.reference_latent = self._diag(self.references)
        self.candidate_noise = self._noise(self.candidates)
        self.reference_noise = self._noise(self.references)
        self.temp = tempfile.TemporaryDirectory(prefix="state_", dir=cache.directory)
        self.path = Path(self.temp.name) / "conditioned.f64"
        self.cross = np.memmap(self.path, mode="w+", dtype=np.float64,
                               shape=(len(self.references), len(self.candidates), self.q))
        for a in range(0, len(self.references), cache.target_chunk):
            self.cross[a:a+cache.target_chunk] = cache.cross[self.references[a:a+cache.target_chunk]]

    def _diag(self, positions):
        raw = self.posterior.latent_diag(positions)
        off = raw.copy()
        idx = np.arange(self.q)
        off[..., idx, idx] = 0
        if np.any(np.abs(off) > 1e-12):
            raise ValueError("latent outputs are dependent")
        diag = np.diagonal(raw, axis1=-2, axis2=-1).copy()
        if np.any(diag < -1e-9):
            raise np.linalg.LinAlgError("negative latent variance")
        return diag

    def _noise(self, positions):
        raw = (self.posterior.noise_blocks(positions) if hasattr(self.posterior, "noise_blocks") else
               np.broadcast_to(self.posterior.noise_cov, (len(positions), self.q, self.q)))
        off = raw.copy()
        idx = np.arange(self.q)
        off[..., idx, idx] = 0
        if np.any(np.abs(off) > 1e-12):
            raise ValueError("noise outputs are dependent")
        diag = np.diagonal(raw, axis1=-2, axis2=-1).copy()
        if np.any(diag <= 0):
            raise np.linalg.LinAlgError("strictly positive noise required")
        return diag

    def scores(self, objective="epig"):
        if objective == "bald":
            return bald_scores(self)
        if objective not in {"epig", "ivr"}:
            raise ValueError("unknown acquisition objective")
        result = np.zeros(len(self.candidates))
        ccov = self.candidate_latent + self.candidate_noise
        for a in range(0, len(self.references), self.cache.target_chunk):
            b = min(a+self.cache.target_chunk, len(self.references))
            reduction = np.square(self.cross[a:b]) / ccov[None]
            if objective == "epig":
                ratio = reduction / (self.reference_latent[a:b] + self.reference_noise[a:b])[:, None]
                if np.any(ratio >= 1) or not np.isfinite(ratio).all():
                    raise np.linalg.LinAlgError("nonpositive independent Gaussian Schur complement")
                value = -.5*np.log1p(-ratio)
            else:
                value = reduction
            result += value.sum(axis=(0, 2)) / len(self.references)
        result[~self.available] = -np.inf
        return result

    def condition(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self.candidates) or not self.available[index]:
            raise ValueError("invalid or previously selected candidate")
        q = self.q
        raw = self.posterior.cov(self.candidates, self.candidates[index:index+1], noisy=False)
        blocks = raw.reshape(len(self.candidates), q, q)
        ccross = np.diagonal(blocks, axis1=-2, axis2=-1).copy()
        for previous in self.candidate_updates:
            ccross -= previous * previous[index]
        root = np.sqrt(self.candidate_latent[index] + self.candidate_noise[index])
        cu = ccross/root
        ru = np.asarray(self.cross[:, index]).copy()/root
        self.candidate_latent -= cu*cu
        self.reference_latent -= ru*ru
        for a in range(0, len(self.references), self.cache.target_chunk):
            b = min(a+self.cache.target_chunk, len(self.references))
            self.cross[a:b] -= ru[a:b, None]*cu[None]
        self.candidate_updates.append(cu)
        self.selected_indices.append(int(index))
        self.available[index] = False

    def close(self):
        if hasattr(self, "cross"):
            del self.cross
        self.temp.cleanup()


def select_streaming_batch(cache, references, objective="epig", batch_size=32, pointwise=False):
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or not 1 <= batch_size <= len(cache.candidates):
        raise ValueError("batch size outside candidate count")
    started = time.perf_counter()
    state = cache.state(references)
    try:
        initial = state.scores(objective)
        order = np.argsort(-initial, kind="stable")
        trace = []
        for step in range(batch_size):
            scores = initial if step == 0 else state.scores(objective)
            index = int(order[step] if pointwise else np.argmax(scores))
            trace.append({"step": step+1, "candidate_index": index,
                          "row_position": int(cache.candidates[index]), "conditional_score": float(scores[index])})
            state.condition(index)
        selected = cache.candidates[state.selected_indices]
        return {"objective": objective, "mode": "pointwise_topk" if pointwise else "conditional_greedy",
                "selected_positions": selected.tolist(), "initial_scores": initial.tolist(), "trace": trace,
                "seconds": time.perf_counter()-started}
    finally:
        state.close()


def batch_target_information(posterior, references, selected, chunk=64):
    """Integrated I(Y_t;Y_B), dense only in B*q (B=32), streamed in targets."""
    selected = np.asarray(selected, dtype=int)
    q = posterior.n_outputs
    noise = posterior.noise_blocks(selected) if hasattr(posterior, "noise_blocks") else np.broadcast_to(
        posterior.noise_cov, (len(selected), q, q))
    sbb = posterior.cov(selected, selected, noisy=False)
    for i in range(len(selected)):
        sbb[i*q:(i+1)*q, i*q:(i+1)*q] += noise[i]
    factor = np.linalg.cholesky(sbb)
    total = 0.
    for a in range(0, len(references), chunk):
        refs = np.asarray(references[a:a+chunk], dtype=int)
        st = posterior.latent_diag(refs)
        st += (posterior.noise_blocks(refs) if hasattr(posterior, "noise_blocks") else posterior.noise_cov)
        cross = posterior.cov(refs, selected, noisy=False).reshape(len(refs), q, len(selected)*q)
        solved = np.linalg.solve(factor, cross.transpose(0, 2, 1))
        after = st - np.swapaxes(solved, -1, -2) @ solved
        lt = np.linalg.cholesky(st)
        la = np.linalg.cholesky(after)
        total += (np.log(np.diagonal(lt, axis1=-2, axis2=-1)) -
                  np.log(np.diagonal(la, axis1=-2, axis2=-1))).sum()
    return float(total/len(references))
