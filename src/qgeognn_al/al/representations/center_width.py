"""Frozen L0-only center/width geometry; exploratory variants remain archived."""
import numpy as np

def center_width_transform(
    l0_targets: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build the L0-scaled center/width output coordinate transform."""

    targets = _validated_targets(l0_targets)
    center = 0.5 * (targets[:, 0] + targets[:, 1])
    width = targets[:, 1] - targets[:, 0]
    center_scale = float(center.std(ddof=0))
    width_scale = float(width.std(ddof=0))
    if center_scale <= 0 or width_scale <= 0:
        raise ValueError("L0 center and width population scales must be positive")
    transform = np.asarray(
        [[0.5 / center_scale, 0.5 / center_scale], [-1.0 / width_scale, 1.0 / width_scale]],
        dtype=np.float64,
    )
    return transform, {
        "definition": "[(V1+V2)/(2*s_C), (V2-V1)/s_W]",
        "center_population_std": center_scale,
        "width_population_std": width_scale,
        "transform_matrix": transform.tolist(),
        "target_source": "L0_labels_only",
        "scale_ddof": 0,
    }

def _validated_scales(scales: tuple[float, float]) -> np.ndarray:
    result = np.asarray(scales, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("two positive finite target scales required")
    return result

def _validated_targets(l0_targets: np.ndarray) -> np.ndarray:
    result = np.asarray(l0_targets, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2 or len(result) < 2 or not np.isfinite(result).all():
        raise ValueError("finite L0 targets with shape (N, 2), N >= 2 required")
    return result
