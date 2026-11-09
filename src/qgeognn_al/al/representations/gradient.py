"""Online full-network gradient sketches for QGeoGNN-V2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from ...predictor.training import loader_pair


@dataclass(frozen=True)
class ParameterSlice:
    name: str
    start: int
    stop: int
    shape: tuple[int, ...]


@dataclass(frozen=True)
class GradientSketchResult:
    features: np.ndarray
    audit: dict[str, object]
    parameter_audit: tuple[dict[str, object], ...]


def parameter_slices(model: torch.nn.Module) -> tuple[ParameterSlice, ...]:
    offset = 0
    result: list[ParameterSlice] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        result.append(ParameterSlice(name, offset, offset + parameter.numel(), tuple(parameter.shape)))
        offset += parameter.numel()
    return tuple(result)


def countsketch_mapping(parameter_count: int, dimension: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic buckets/signs for the concatenated two-output gradient."""

    if parameter_count < 1 or dimension < 1:
        raise ValueError("positive parameter_count and sketch dimension required")
    rng = np.random.default_rng(int(seed))
    buckets = rng.integers(0, int(dimension), size=2 * int(parameter_count), dtype=np.int64)
    signs = (2 * rng.integers(0, 2, size=2 * int(parameter_count), dtype=np.int8) - 1).astype(np.float32)
    return torch.from_numpy(buckets), torch.from_numpy(signs)


def sketch_flat_gradient(
    flat_gradient: torch.Tensor,
    output_index: int,
    buckets: torch.Tensor,
    signs: torch.Tensor,
    dimension: int,
) -> torch.Tensor:
    parameter_count = flat_gradient.numel()
    start = int(output_index) * parameter_count
    stop = start + parameter_count
    result = torch.zeros(int(dimension), dtype=torch.float32)
    result.scatter_add_(
        0,
        buckets[start:stop],
        flat_gradient.detach().to(torch.float32).cpu() * signs[start:stop],
    )
    return result


def extract_q50_gradient_sketches(
    model: torch.nn.Module,
    atom_data: Sequence[object],
    angle_data: Sequence[object],
    indices: Sequence[int],
    scales: tuple[float, float],
    *,
    dimension: int = 512,
    sketch_seed: int,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> GradientSketchResult:
    """Sketch per-row gradients of scaled V1/V2 q50 outputs.

    Only one row's full gradient is materialized at a time.  The returned
    matrix is N x sketch_dimension; an N x parameter_count Jacobian is never
    formed.
    """

    positions = [int(value) for value in indices]
    if not positions:
        raise ValueError("gradient extraction requires at least one row")
    if len(set(positions)) != len(positions):
        raise ValueError("gradient extraction indices must be unique")
    if len(scales) != 2 or not np.isfinite(scales).all() or min(scales) <= 0:
        raise ValueError("two positive finite L0 target scales required")
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    slices = parameter_slices(model)
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if parameter_count != (slices[-1].stop if slices else 0):
        raise RuntimeError("parameter slice accounting mismatch")
    buckets, signs = countsketch_mapping(parameter_count, int(dimension), int(sketch_seed))
    features = np.empty((len(positions), int(dimension)), dtype=np.float32)
    ever_nonzero = torch.zeros(parameter_count, dtype=torch.bool)
    tensor_observed = np.zeros(len(parameters), dtype=bool)
    started = time.perf_counter()
    model.eval()

    for row_index, (atom_batch, angle_batch) in enumerate(
        zip(*loader_pair(atom_data, angle_data, positions, batch_size=1))
    ):
        output = model(atom_batch, angle_batch)
        if output.shape != (1, 6):
            raise ValueError("QGeoGNN-V2 gradient extraction requires one six-output row")
        sketch = torch.zeros(int(dimension), dtype=torch.float32)
        for endpoint, output_column in enumerate((1, 4)):
            gradients = torch.autograd.grad(
                output[0, output_column] / float(scales[endpoint]),
                parameters,
                retain_graph=endpoint == 0,
                create_graph=False,
                allow_unused=True,
            )
            flat_parts: list[torch.Tensor] = []
            for tensor_index, (gradient, parameter) in enumerate(zip(gradients, parameters)):
                if gradient is None:
                    flat_parts.append(torch.zeros(parameter.numel(), dtype=torch.float32))
                else:
                    value = gradient.detach().reshape(-1).to(torch.float32).cpu()
                    flat_parts.append(value)
                    if bool(torch.any(value != 0)):
                        tensor_observed[tensor_index] = True
            flat = torch.cat(flat_parts)
            ever_nonzero |= flat != 0
            sketch += sketch_flat_gradient(flat, endpoint, buckets, signs, int(dimension))
        values = sketch.numpy()
        if not np.isfinite(values).all():
            raise RuntimeError("non-finite sketched gradient feature")
        features[row_index] = values
        if progress and (row_index == 0 or (row_index + 1) % 50 == 0 or row_index + 1 == len(positions)):
            progress(
                {
                    "completed_rows": row_index + 1,
                    "total_rows": len(positions),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )

    norms = np.linalg.norm(features.astype(np.float64), axis=1)
    unique_count = len(np.unique(features, axis=0))
    mapping_hash = hashlib.sha256(buckets.numpy().tobytes() + signs.numpy().tobytes()).hexdigest()
    audit: dict[str, object] = {
        "rows": len(positions),
        "parameter_count": parameter_count,
        "trainable_parameter_tensors": len(parameters),
        "gradient_bearing_parameter_count": int(ever_nonzero.sum()),
        "gradient_bearing_parameter_tensors": int(tensor_observed.sum()),
        "sketch_dimension": int(dimension),
        "sketch_seed": int(sketch_seed),
        "sketch_mapping_sha256": mapping_hash,
        "representation": "CountSketch(concat(grad(V1_q50/s_V1), grad(V2_q50/s_V2)))",
        "full_network": True,
        "n_by_p_materialized": False,
        "all_finite": bool(np.isfinite(features).all()),
        "norm_min": float(norms.min()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std(ddof=0)),
        "norm_max": float(norms.max()),
        "zero_norm_rows": int((norms == 0).sum()),
        "unique_feature_rows": unique_count,
        "duplicate_feature_rows": len(features) - unique_count,
        "elapsed_seconds": time.perf_counter() - started,
    }
    rows: list[dict[str, object]] = []
    for tensor_index, item in enumerate(slices):
        rows.append(
            {
                "parameter_name": item.name,
                "parameter_count": item.stop - item.start,
                "requires_grad": True,
                "gradient_tensor_observed": bool(tensor_observed[tensor_index]),
                "ever_nonzero_elements": int(ever_nonzero[item.start : item.stop].sum()),
                "shape": "x".join(str(value) for value in item.shape),
            }
        )
    return GradientSketchResult(features, audit, tuple(rows))


def extract_linear_output_gradient_sketches(
    model: torch.nn.Module,
    atom_data: Sequence[object],
    angle_data: Sequence[object],
    indices: Sequence[int],
    output_transform: np.ndarray,
    *,
    dimension: int = 512,
    sketch_seed: int,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> GradientSketchResult:
    """Sketch gradients of a fixed linear transform of the two q50 outputs.

    ``output_transform`` defines ``z = T @ [V1_q50, V2_q50]``.  The two
    transformed-output gradients use the same two-output CountSketch mapping as
    :func:`extract_q50_gradient_sketches`.  Only one row gradient is
    materialized at a time.
    """

    positions = [int(value) for value in indices]
    transform = np.asarray(output_transform, dtype=np.float64)
    if not positions:
        raise ValueError("gradient extraction requires at least one row")
    if len(set(positions)) != len(positions):
        raise ValueError("gradient extraction indices must be unique")
    if transform.shape != (2, 2) or not np.isfinite(transform).all():
        raise ValueError("output_transform must be a finite 2x2 matrix")
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    slices = parameter_slices(model)
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if parameter_count != (slices[-1].stop if slices else 0):
        raise RuntimeError("parameter slice accounting mismatch")
    buckets, signs = countsketch_mapping(parameter_count, int(dimension), int(sketch_seed))
    features = np.empty((len(positions), int(dimension)), dtype=np.float32)
    ever_nonzero = torch.zeros(parameter_count, dtype=torch.bool)
    tensor_observed = np.zeros(len(parameters), dtype=bool)
    started = time.perf_counter()
    model.eval()

    for row_index, (atom_batch, angle_batch) in enumerate(
        zip(*loader_pair(atom_data, angle_data, positions, batch_size=1))
    ):
        output = model(atom_batch, angle_batch)
        if output.shape != (1, 6):
            raise ValueError("QGeoGNN-V2 gradient extraction requires one six-output row")
        q50 = output[0, torch.tensor((1, 4), dtype=torch.long, device=output.device)]
        coefficients = torch.as_tensor(transform, dtype=output.dtype, device=output.device)
        transformed = coefficients @ q50
        sketch = torch.zeros(int(dimension), dtype=torch.float32)
        for endpoint in range(2):
            gradients = torch.autograd.grad(
                transformed[endpoint],
                parameters,
                retain_graph=endpoint == 0,
                create_graph=False,
                allow_unused=True,
            )
            flat_parts: list[torch.Tensor] = []
            for tensor_index, (gradient, parameter) in enumerate(zip(gradients, parameters)):
                if gradient is None:
                    flat_parts.append(torch.zeros(parameter.numel(), dtype=torch.float32))
                else:
                    value = gradient.detach().reshape(-1).to(torch.float32).cpu()
                    flat_parts.append(value)
                    if bool(torch.any(value != 0)):
                        tensor_observed[tensor_index] = True
            flat = torch.cat(flat_parts)
            ever_nonzero |= flat != 0
            sketch += sketch_flat_gradient(flat, endpoint, buckets, signs, int(dimension))
        values = sketch.numpy()
        if not np.isfinite(values).all():
            raise RuntimeError("non-finite transformed sketched gradient feature")
        features[row_index] = values
        if progress and (row_index == 0 or (row_index + 1) % 50 == 0 or row_index + 1 == len(positions)):
            progress(
                {
                    "completed_rows": row_index + 1,
                    "total_rows": len(positions),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )

    norms = np.linalg.norm(features.astype(np.float64), axis=1)
    unique_count = len(np.unique(features, axis=0))
    mapping_hash = hashlib.sha256(buckets.numpy().tobytes() + signs.numpy().tobytes()).hexdigest()
    audit: dict[str, object] = {
        "rows": len(positions),
        "parameter_count": parameter_count,
        "trainable_parameter_tensors": len(parameters),
        "gradient_bearing_parameter_count": int(ever_nonzero.sum()),
        "gradient_bearing_parameter_tensors": int(tensor_observed.sum()),
        "sketch_dimension": int(dimension),
        "sketch_seed": int(sketch_seed),
        "sketch_mapping_sha256": mapping_hash,
        "representation": "CountSketch(concat(grad(z1), grad(z2)))",
        "output_transform": transform.tolist(),
        "full_network": True,
        "n_by_p_materialized": False,
        "all_finite": bool(np.isfinite(features).all()),
        "norm_min": float(norms.min()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std(ddof=0)),
        "norm_max": float(norms.max()),
        "zero_norm_rows": int((norms == 0).sum()),
        "unique_feature_rows": unique_count,
        "duplicate_feature_rows": len(features) - unique_count,
        "elapsed_seconds": time.perf_counter() - started,
    }
    rows: list[dict[str, object]] = []
    for tensor_index, item in enumerate(slices):
        rows.append(
            {
                "parameter_name": item.name,
                "parameter_count": item.stop - item.start,
                "requires_grad": True,
                "gradient_tensor_observed": bool(tensor_observed[tensor_index]),
                "ever_nonzero_elements": int(ever_nonzero[item.start : item.stop].sum()),
                "shape": "x".join(str(value) for value in item.shape),
            }
        )
    return GradientSketchResult(features, audit, tuple(rows))



def extract_gradient(
    predictor: torch.nn.Module, *, expected_state_hash: str, authorization,
    rows, provenance, sketch_seed: int, scales=None,
    kind: str = 'gradient', l0_targets=None, l0_ids=None,
):
    """Extract the frozen 512D q50/CW representation from label-free graph rows.

    Standard mode differentiates q50/sigma exactly as the old scalar routine.
    CW mode uses the historical linear-output autograd routine, including its
    float32 coefficient conversion. It does not silently substitute the faster
    project-before-combination bank, whose rounding is a different profile.
    Supplied L0 labels are already authorized known information; their ordered
    IDs must be a subset of authorized L and are included in the feature hash.
    """
    from .latent import _qualified_graph, _graph_hash, predictor_state_hash
    from .base import FeatureSnapshot, RepresentationConfig, ExtractionProvenance
    from .center_width import center_width_transform, _validated_scales
    from ...infra.contracts import sample_ids, content_hash, canonical_bytes
    import re
    if kind not in ('gradient', 'center_width'):
        raise ValueError('only registered raw/CW gradient extraction modes are allowed')
    if type(sketch_seed) is not int or sketch_seed < 0:
        raise ValueError('sketch_seed must be an explicit nonnegative integer')
    if not isinstance(expected_state_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', expected_state_hash):
        raise ValueError('pinned predictor state SHA-256 required')
    if any(module.training for module in predictor.modules()):
        raise ValueError('predictor must already be frozen in eval mode')
    if any(value.device.type != 'cpu' for value in predictor.state_dict().values()):
        raise ValueError('CPU-qualified predictor required')
    if predictor_state_hash(predictor) != expected_state_hash:
        raise ValueError('predictor state does not match pinned fingerprint')
    if any(not parameter.requires_grad for parameter in predictor.parameters()):
        raise ValueError('full-network sketches require every predictor parameter to remain trainable')
    if provenance.source_kind == 'authorized_snapshot' and sum(p.numel() for p in predictor.parameters()) != 458952:
        raise ValueError('authorized QGeoGNN-V2 gradient snapshots require exactly 458952 parameters')
    ids = sample_ids([row.sample_id for row in rows], name='graph rows')
    if ids != authorization.train_ids + authorization.candidate_ids:
        raise ValueError('graph rows must exactly match authorized L then U order')
    atoms = [_qualified_graph(row.atom, atom=True) for row in rows]
    angles = [_qualified_graph(row.angle, atom=False) for row in rows]
    canonical_positions = [int(a.canonical_position.item()) for a in atoms]
    if len(set(canonical_positions)) != len(canonical_positions):
        raise ValueError('graph canonical positions must be unique')
    graph_hash = _graph_hash(atoms, angles)
    details = {'dimension': 512, 'sketch_seed': sketch_seed, 'ordered_sample_ids_hash': content_hash(ids)}
    if kind == 'gradient':
        scales = tuple(_validated_scales(scales).tolist())
        if l0_targets is not None or l0_ids is not None:
            raise ValueError('raw q50 gradient accepts only pinned target scales, not label arrays')
        result = extract_q50_gradient_sketches(predictor, atoms, angles, list(range(len(rows))),
                                               scales, dimension=512, sketch_seed=sketch_seed)
        details.update(target_scales=list(scales), normalization='q50 divided by frozen L0 population sigma')
    else:
        if scales is not None:
            raise ValueError('CW mode uses its own L0 center/width scales, not endpoint scaling')
        initial_ids = sample_ids(l0_ids, name='frozen L0 IDs') if l0_ids is not None else ()
        targets = np.asarray(l0_targets, dtype=np.float64)
        if not initial_ids or not set(initial_ids) <= set(authorization.train_ids) or targets.shape != (len(initial_ids), 2):
            raise ValueError('CW transform requires identity-aligned authorized frozen L0 labels')
        transform, transform_audit = center_width_transform(targets)
        result = extract_linear_output_gradient_sketches(predictor, atoms, angles, list(range(len(rows))),
                                                          transform, dimension=512, sketch_seed=sketch_seed)
        details.update(l0_ids_hash=content_hash(initial_ids), l0_labels_hash=content_hash(targets.tolist()),
                       output_transform=transform_audit, floating_profile='linear-output-autograd')
    state_after = predictor_state_hash(predictor)
    graph_after = _graph_hash(atoms, angles)
    if state_after != expected_state_hash or any(module.training for module in predictor.modules()):
        raise ValueError('predictor changed during gradient extraction')
    if graph_after != graph_hash:
        raise ValueError('graph inputs changed during gradient extraction')
    details.update({k: v for k, v in result.audit.items() if k != 'elapsed_seconds'})
    cut = len(authorization.train_ids)
    return FeatureSnapshot(authorization=authorization, train_ids=authorization.train_ids,
                           candidate_ids=authorization.candidate_ids, train_features=result.features[:cut],
                           candidate_features=result.features[cut:], config=RepresentationConfig(kind, 512),
                           predictor=provenance, extraction=ExtractionProvenance(
                               method='q50_countsketch' if kind == 'gradient' else 'cw_countsketch',
                               predictor_state_hash=expected_state_hash, predictor_state_after_hash=state_after,
                               graph_snapshot_hash=graph_hash, graph_snapshot_after_hash=graph_after,
                               batch_size=1, details_json=canonical_bytes(details).decode()))
