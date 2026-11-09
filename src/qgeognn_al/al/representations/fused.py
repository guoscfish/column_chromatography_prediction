"""Existing fixed-alpha gradient/latent block fusion, without research diagnostics."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
DEFAULT_ALPHA=.5
ALPHA=DEFAULT_ALPHA
GRADIENT_DIMENSION=512
LATENT_DIMENSION=128
FUSED_DIMENSION=640

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

def fuse_snapshots(gradient, latent, *, alpha=DEFAULT_ALPHA):
    """Compose two already-authorized snapshots without reading any labels."""
    from .base import FeatureSnapshot, RepresentationConfig
    from ...infra.contracts import canonical_bytes
    gradient.verify(); latent.verify()
    if gradient.authorization != latent.authorization or gradient.predictor != latent.predictor:
        raise ValueError('fusion inputs must share authorized identities and predictor provenance')
    if gradient.config.dimension != 512 or latent.config.dimension != 128:
        raise ValueError('fusion requires 512D gradient and 128D latent inputs')
    g = np.vstack([gradient.train, gradient.candidates]); h = np.vstack([latent.train, latent.candidates])
    cut = len(gradient.train_ids)
    result = fuse_gradient_latent_features(g, h, cut, alpha=alpha)
    config = RepresentationConfig('fused', 640, canonical_bytes(dict(result.audit,
                                   gradient_snapshot_sha256=gradient.sha256,
                                   latent_snapshot_sha256=latent.sha256)).decode())
    return FeatureSnapshot(authorization=gradient.authorization, train_ids=gradient.train_ids,
                           candidate_ids=gradient.candidate_ids, train_features=result.features[:cut],
                           candidate_features=result.features[cut:], config=config, predictor=gradient.predictor)
