"""Block-normalized gradient/latent features for Fusion-MaxDet.

The representation is deliberately label-free apart from the current labeled
rows used to estimate the two block scales.  The linear kernel is
``alpha * K_gradient + (1 - alpha) * K_latent``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

import numpy as np


DEFAULT_ALPHA = 0.5
# Backwards-compatible name used by the frozen alpha=.5 study.  New studies
# pass alpha explicitly through their study specification.
ALPHA = DEFAULT_ALPHA
GRADIENT_DIMENSION = 512
LATENT_DIMENSION = 128
FUSED_DIMENSION = GRADIENT_DIMENSION + LATENT_DIMENSION


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(repr(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FusionResult:
    features: np.ndarray
    gradient_scale: float
    latent_scale: float
    audit: dict[str, object]


def fuse_gradient_latent_features(
    gradient: np.ndarray,
    latent: np.ndarray,
    labeled_count: int,
    *,
    alpha: float = ALPHA,
) -> FusionResult:
    """Normalize each block using only ``L_t`` and concatenate the blocks.

    Rows must be ordered ``[L_t, U_t]``.  No target arrays are accepted by
    this API, which makes accidental label leakage at selection time harder.
    Zero or non-finite labeled block energy is rejected instead of repaired by
    an arbitrary epsilon.
    """

    g = np.asarray(gradient, dtype=np.float64)
    h = np.asarray(latent, dtype=np.float64)
    if g.ndim != 2 or h.ndim != 2 or len(g) != len(h):
        raise ValueError("gradient and latent matrices must have the same row count")
    if g.shape[1] != GRADIENT_DIMENSION or h.shape[1] != LATENT_DIMENSION:
        raise ValueError("expected 512D gradient and 128D latent representations")
    n_labeled = int(labeled_count)
    if not 0 < n_labeled <= len(g):
        raise ValueError("labeled_count must select a nonempty prefix of current rows")
    if not np.isfinite(g).all() or not np.isfinite(h).all():
        raise ValueError("gradient and latent matrices must be finite")
    if not np.isfinite(alpha) or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")

    g_scale = float(np.sqrt(np.mean(np.einsum("ij,ij->i", g[:n_labeled], g[:n_labeled]))))
    h_scale = float(np.sqrt(np.mean(np.einsum("ij,ij->i", h[:n_labeled], h[:n_labeled]))))
    if not np.isfinite(g_scale) or g_scale <= 0:
        raise ValueError("gradient block has zero or non-finite labeled energy")
    if not np.isfinite(h_scale) or h_scale <= 0:
        raise ValueError("latent block has zero or non-finite labeled energy")

    g_normalized = g / g_scale
    h_normalized = h / h_scale
    fused = np.concatenate(
        [np.sqrt(float(alpha)) * g_normalized,
         np.sqrt(1.0 - float(alpha)) * h_normalized],
        axis=1,
    ).astype(np.float32)
    if not np.isfinite(fused).all() or fused.shape[1] != FUSED_DIMENSION:
        raise RuntimeError("fusion produced an invalid feature matrix")
    audit: dict[str, object] = {
        "alpha": float(alpha),
        "gradient_dimension": GRADIENT_DIMENSION,
        "latent_dimension": LATENT_DIMENSION,
        "fused_dimension": FUSED_DIMENSION,
        "labeled_count": n_labeled,
        "gradient_block_scale": g_scale,
        "latent_block_scale": h_scale,
        "gradient_labeled_mean_norm_sq": float(np.mean(np.sum(g_normalized[:n_labeled] ** 2, axis=1))),
        "latent_labeled_mean_norm_sq": float(np.mean(np.sum(h_normalized[:n_labeled] ** 2, axis=1))),
        "fused_labeled_mean_norm_sq": float(np.mean(np.sum(fused[:n_labeled].astype(np.float64) ** 2, axis=1))),
        "gradient_feature_sha256": _array_hash(g),
        "latent_feature_sha256": _array_hash(h),
        "fused_feature_sha256": _array_hash(fused),
        "normalization": "sqrt(mean_L_t(||block_i||_2^2)); L_t prefix only",
        "kernel_definition": (
            f"{float(alpha):.12g}*K_gradient_normalized + "
            f"{1.0 - float(alpha):.12g}*K_latent_normalized"
        ),
        "all_finite": True,
    }
    return FusionResult(fused, g_scale, h_scale, audit)


def representation_diagnostics(features: np.ndarray) -> Mapping[str, float]:
    """Return effective rank and mean absolute normalized kernel correlation."""

    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("diagnostic features must be a finite nonempty matrix")
    # X X^T and X^T X have the same non-zero eigenvalues.  Computing the
    # feature-space spectrum avoids an O(N^3) decomposition of a ~3330-square
    # Gram matrix at every acquisition round while preserving the diagnostic.
    covariance = x.T @ x
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = float(eigenvalues.sum())
    if total == 0:
        rank = 0.0
    else:
        p = eigenvalues[eigenvalues > 0] / total
        rank = float(np.exp(-np.sum(p * np.log(p))))
    norms = np.linalg.norm(x, axis=1)
    normalized = np.divide(x, norms[:, None], out=np.zeros_like(x), where=norms[:, None] > 0)
    kernel = normalized @ normalized.T
    off_diagonal = np.abs(kernel[~np.eye(len(x), dtype=bool)])
    return {"effective_rank": rank, "pairwise_kernel_abs_corr": float(off_diagonal.mean()) if len(off_diagonal) else 0.0}


def latent_common_mode_diagnostics(
    latent: np.ndarray,
    labeled_count: int,
) -> Mapping[str, float]:
    """Diagnose a labeled-set mean direction without changing acquisition.

    The mean is estimated only from the current labeled prefix ``L_t``.  The
    centered effective-rank and kernel-correlation diagnostics are evaluated
    on all current ``[L_t, U_t]`` rows after subtracting that fixed mean.
    """

    h = np.asarray(latent, dtype=np.float64)
    n_labeled = int(labeled_count)
    if h.ndim != 2 or h.shape[1] != LATENT_DIMENSION or not 0 < n_labeled <= len(h):
        raise ValueError("latent must be a current 128D matrix with a nonempty labeled prefix")
    if not np.isfinite(h).all():
        raise ValueError("latent common-mode diagnostics require finite features")
    mean = h[:n_labeled].mean(axis=0)
    denominator = float(np.mean(np.einsum("ij,ij->i", h[:n_labeled], h[:n_labeled])))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("latent labeled mean squared norm must be positive and finite")
    centered = h - mean
    centered_diagnostics = representation_diagnostics(centered)
    return {
        "latent_mean_direction_ratio": float(mean @ mean) / denominator,
        "latent_mean_norm_sq": float(mean @ mean),
        "latent_labeled_mean_norm_sq": denominator,
        "centered_latent_effective_rank": centered_diagnostics["effective_rank"],
        "centered_latent_pairwise_kernel_abs_corr": centered_diagnostics["pairwise_kernel_abs_corr"],
    }


__all__ = [
    "DEFAULT_ALPHA", "ALPHA", "GRADIENT_DIMENSION", "LATENT_DIMENSION", "FUSED_DIMENSION",
    "FusionResult", "fuse_gradient_latent_features", "representation_diagnostics",
    "latent_common_mode_diagnostics",
]
