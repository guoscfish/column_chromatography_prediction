"""Authoritative frozen-predictor latent extraction for the new public API.

Inference semantics are migrated from immutable historical coverage source;
graph ordering, batching and extraction mathematics are unchanged. Historical
protocols recover their original tree from Git, rather than importing an editable
parallel implementation into the stable package. This entrypoint
accepts no label tables, graph paths, training API, provider or test role.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Sequence

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ...infra.contracts import BaselineProvenance, canonical_bytes, sample_ids
from .base import Authorization, ExtractionProvenance, FeatureSnapshot, RepresentationConfig


@dataclass(frozen=True)
class GraphRow:
    sample_id: str
    atom: Data
    angle: Data


def predictor_state_hash(model: torch.nn.Module) -> str:
    """Hash named weights/buffers only; never serialize or execute a checkpoint."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(canonical_bytes(list(array.shape)))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _qualified_graph(graph: Data, *, atom: bool) -> Data:
    if not isinstance(graph, Data):
        raise ValueError("GraphRow must contain individual torch_geometric Data objects")
    permitted = {"edge_index", "edge_attr", "num_nodes"}
    required = {"edge_index", "edge_attr"}
    if atom:
        permitted |= {"x", "canonical_position"}
        required |= {"x", "canonical_position"}
    keys = set(graph.keys())
    if keys - permitted or not required <= keys:
        raise ValueError("graph contains a label, unapproved field or missing inference input")
    for key in keys - {"num_nodes"}:
        value = graph[key]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or not torch.isfinite(value).all():
            raise ValueError("qualified graph fields must be finite CPU tensors")
    if atom:
        position = graph.canonical_position
        if position.numel() != 1 or position.dtype not in (torch.int32, torch.int64) or position.item() < 0:
            raise ValueError("one nonnegative canonical_position is required per graph row")
    # Inference only sees owned copies and cannot mutate the caller's graph.
    return graph.clone()


def _graph_hash(atoms: Sequence[Data], angles: Sequence[Data]) -> str:
    digest = hashlib.sha256()
    for graphs in (atoms, angles):
        for graph in graphs:
            for key in sorted(graph.keys()):
                value = graph[key]
                digest.update(key.encode())
                if isinstance(value, torch.Tensor):
                    array = value.detach().cpu().contiguous().numpy()
                    digest.update(str(array.dtype).encode())
                    digest.update(canonical_bytes(list(array.shape)))
                    digest.update(array.tobytes())
                else:
                    digest.update(canonical_bytes(value))
    return digest.hexdigest()


def extract_latent(
    predictor: torch.nn.Module, *, expected_state_hash: str,
    authorization: Authorization, rows: Sequence[GraphRow], provenance: BaselineProvenance,
    batch_size: int = 2048,
) -> FeatureSnapshot:
    """Return 128D vectors in authorized L-then-U order without changing mode.

    The caller must provide an already eval-mode, CPU-qualified predictor and
    its independently pinned state fingerprint. Both weights and copied graph
    inputs are checked before/after inference. `rows` must contain exactly L+U;
    extra or differently ordered identities are rejected before predictor use.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(expected_state_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_state_hash):
        raise ValueError("expected predictor state SHA-256 is required")
    if any(module.training for module in predictor.modules()):
        raise ValueError("predictor must already be frozen in eval mode")
    if any(value.device.type != "cpu" for value in predictor.state_dict().values()):
        raise ValueError("predictor must use the CPU-qualified runtime")
    if predictor_state_hash(predictor) != expected_state_hash:
        raise ValueError("predictor state does not match the pinned fingerprint")
    ordered_ids = sample_ids([row.sample_id for row in rows], name="graph rows")
    if ordered_ids != authorization.train_ids + authorization.candidate_ids:
        raise ValueError("graph rows must exactly match authorized L then U order")
    atoms = [_qualified_graph(row.atom, atom=True) for row in rows]
    angles = [_qualified_graph(row.angle, atom=False) for row in rows]
    expected_positions = [int(graph.canonical_position.item()) for graph in atoms]
    if len(set(expected_positions)) != len(expected_positions):
        raise ValueError("graph canonical positions must be unique")
    graph_hash = _graph_hash(atoms, angles)
    chunks, positions = [], []
    with torch.no_grad():
        for atom_batch, angle_batch in zip(DataLoader(atoms, batch_size=batch_size, shuffle=False),
                                           DataLoader(angles, batch_size=batch_size, shuffle=False)):
            representation = predictor.extract_representation(atom_batch, angle_batch)
            if representation.ndim != 2 or representation.shape[1] != 128:
                raise ValueError("frozen predictor must return 128-dimensional latent vectors")
            chunks.append(representation.cpu().numpy())
            positions.extend(atom_batch.canonical_position.cpu().numpy().reshape(-1).tolist())
    if positions != expected_positions:
        raise ValueError("latent graph order drift")
    state_after_hash = predictor_state_hash(predictor)
    if state_after_hash != expected_state_hash or any(module.training for module in predictor.modules()):
        raise ValueError("frozen predictor changed during latent extraction")
    graph_after_hash = _graph_hash(atoms, angles)
    if graph_after_hash != graph_hash:
        raise ValueError("graph inputs changed during latent extraction")
    matrix = np.vstack(chunks)
    if matrix.shape != (len(rows), 128) or not np.isfinite(matrix).all():
        raise ValueError("latent representation row count or finite-value check failed")
    cut = len(authorization.train_ids)
    return FeatureSnapshot(authorization=authorization, train_ids=authorization.train_ids,
                           candidate_ids=authorization.candidate_ids,
                           train_features=matrix[:cut], candidate_features=matrix[cut:],
                           config=RepresentationConfig("latent", 128), predictor=provenance,
                           extraction=ExtractionProvenance(
                               method="QGeoGNNV2.extract_representation",
                               predictor_state_hash=expected_state_hash,
                               predictor_state_after_hash=state_after_hash,
                               graph_snapshot_hash=graph_hash, graph_snapshot_after_hash=graph_after_hash,
                               batch_size=batch_size))
