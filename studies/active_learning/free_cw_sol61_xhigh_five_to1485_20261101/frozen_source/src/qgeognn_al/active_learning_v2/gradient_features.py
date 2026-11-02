"""Online full-network gradient sketches for QGeoGNN-V2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from ..training.predictor import loader_pair


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


def state_dict_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


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


def extract_linear_output_gradient_sketches_many(
    model: torch.nn.Module,
    atom_data: Sequence[object],
    angle_data: Sequence[object],
    indices: Sequence[int],
    output_transforms: Mapping[str, np.ndarray],
    *,
    dimension: int = 512,
    sketch_seed: int,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, GradientSketchResult]:
    """Extract several fixed output geometries in one two-gradient row pass.

    CountSketch and differentiation are linear.  Each row's V1/V2 q50
    gradients are therefore computed once, projected into both output slots,
    and combined with every fixed 2x2 transform.  No cross-row Jacobian or
    parameter-sized cache is retained.
    """

    positions = [int(value) for value in indices]
    transforms = {str(name): np.asarray(value, dtype=np.float64) for name, value in output_transforms.items()}
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("gradient extraction requires unique nonempty row indices")
    if not transforms:
        raise ValueError("at least one output transform is required")
    if any(value.shape != (2, 2) or not np.isfinite(value).all() for value in transforms.values()):
        raise ValueError("every output transform must be a finite 2x2 matrix")
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    slices = parameter_slices(model)
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if parameter_count != (slices[-1].stop if slices else 0):
        raise RuntimeError("parameter slice accounting mismatch")
    buckets, signs = countsketch_mapping(parameter_count, int(dimension), int(sketch_seed))
    features = {
        name: np.empty((len(positions), int(dimension)), dtype=np.float32) for name in transforms
    }
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
        endpoint_projections: list[tuple[torch.Tensor, torch.Tensor]] = []
        for input_endpoint, output_column in enumerate((1, 4)):
            gradients = torch.autograd.grad(
                output[0, output_column],
                parameters,
                retain_graph=input_endpoint == 0,
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
            endpoint_projections.append(
                (
                    sketch_flat_gradient(flat, 0, buckets, signs, int(dimension)),
                    sketch_flat_gradient(flat, 1, buckets, signs, int(dimension)),
                )
            )
        for name, transform in transforms.items():
            values = sum(
                float(transform[output_endpoint, input_endpoint])
                * endpoint_projections[input_endpoint][output_endpoint]
                for output_endpoint in range(2)
                for input_endpoint in range(2)
            ).numpy()
            if not np.isfinite(values).all():
                raise RuntimeError(f"non-finite transformed gradient feature: {name}")
            features[name][row_index] = values
        if progress and (row_index == 0 or (row_index + 1) % 50 == 0 or row_index + 1 == len(positions)):
            progress(
                {
                    "completed_rows": row_index + 1,
                    "total_rows": len(positions),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )

    mapping_hash = hashlib.sha256(buckets.numpy().tobytes() + signs.numpy().tobytes()).hexdigest()
    parameter_rows = tuple(
        {
            "parameter_name": item.name,
            "parameter_count": item.stop - item.start,
            "requires_grad": True,
            "gradient_tensor_observed": bool(tensor_observed[tensor_index]),
            "ever_nonzero_elements": int(ever_nonzero[item.start : item.stop].sum()),
            "shape": "x".join(str(value) for value in item.shape),
        }
        for tensor_index, item in enumerate(slices)
    )
    elapsed = time.perf_counter() - started
    results = {}
    for name, values in features.items():
        norms = np.linalg.norm(values.astype(np.float64), axis=1)
        unique_count = len(np.unique(values, axis=0))
        results[name] = GradientSketchResult(
            values,
            {
                "rows": len(positions),
                "parameter_count": parameter_count,
                "trainable_parameter_tensors": len(parameters),
                "gradient_bearing_parameter_count": int(ever_nonzero.sum()),
                "gradient_bearing_parameter_tensors": int(tensor_observed.sum()),
                "sketch_dimension": int(dimension),
                "sketch_seed": int(sketch_seed),
                "sketch_mapping_sha256": mapping_hash,
                "representation": "CountSketch(concat(grad(z1), grad(z2)))",
                "output_transform": transforms[name].tolist(),
                "full_network": True,
                "n_by_p_materialized": False,
                "multi_transform_single_backward_pass": True,
                "all_finite": bool(np.isfinite(values).all()),
                "norm_min": float(norms.min()),
                "norm_mean": float(norms.mean()),
                "norm_std": float(norms.std(ddof=0)),
                "norm_max": float(norms.max()),
                "zero_norm_rows": int((norms == 0).sum()),
                "unique_feature_rows": unique_count,
                "duplicate_feature_rows": len(values) - unique_count,
                "elapsed_seconds": elapsed,
            },
            parameter_rows,
        )
    return results
