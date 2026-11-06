"""Frozen, label-free representation extraction for the offline GP study.

All data access is explicit and rooted at a configurable repository directory.
This module never opens a response table and never trains a QGeoGNN model.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

BASE = Path("studies/active_learning/qgeognn_v2_row_sequential_b32")
EXTENSION = Path("studies/active_learning/qgeognn_v2_row_lcmd_to1485_20261026")
SEEDS = (157, 887, 2357, 6101, 12203)
BUDGETS = (333, 525, 813, 1005)
FEATURE_COLUMNS = ("sample_id", "canonical_smiles", "Density g/ml", "V/ul",
                   "loading solvent", "Volume of loading solvent/ul", "PE/EA")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _ids_hash(ids: Sequence[str]) -> str:
    return stable_hash(sorted(str(sid) for sid in ids))


@dataclass
class StateArtifacts:
    root: Path
    seed: int
    budget: int
    round_index: int
    directory: Path
    checkpoint: Path
    raw_cache: Path
    labeled_ids: list[str]
    unlabeled_ids: list[str]
    validation_ids: list[str]
    labeled_indices: np.ndarray
    unlabeled_indices: np.ndarray
    validation_indices: np.ndarray
    l0_indices: np.ndarray
    partition: pd.DataFrame
    audit: dict[str, Any]

    @property
    def allowed_indices(self) -> set[int]:
        return set(np.r_[self.labeled_indices, self.unlabeled_indices, self.validation_indices].astype(int))

    def validate_indices(self, indices: Sequence[int]) -> np.ndarray:
        values = np.asarray(indices, dtype=np.int64)
        if values.ndim != 1 or len(np.unique(values)) != len(values):
            raise ValueError("indices must be a one-dimensional unique sequence")
        if not set(values.astype(int)) <= self.allowed_indices:
            raise PermissionError("requested indices include test or unknown rows")
        return values


def audit_lcmd_state(root: Path | str, seed: int, budget: int) -> StateArtifacts:
    """Verify the exact historical checkpoint, train/pool state, and raw cache."""
    root = Path(root).resolve()
    seed, budget = int(seed), int(budget)
    if seed not in SEEDS or budget not in BUDGETS:
        raise ValueError("state is outside the preregistered five-seed/four-budget matrix")
    round_index = (budget - 333) // 32
    directory = root / BASE / f"runtime/seed_{seed}/lcmd/round_{round_index:02d}"
    acquisition_owner = BASE if round_index < 21 else EXTENSION
    acquisition = root / acquisition_owner / f"runtime/seed_{seed}/lcmd/round_{round_index:02d}/acquisition_artifacts"
    checkpoint = directory / "model/best.pt"
    raw_cache = acquisition / "current_gradient_features.npz"
    partition_path = root / BASE / f"splits/row_seed_{seed}.csv"
    partition = pd.read_csv(partition_path, dtype={"sample_id": str})
    required = {"sample_id", "canonical_index", "canonical_smiles", "role", "outer_seed"}
    if not required <= set(partition) or partition.sample_id.duplicated().any():
        raise ValueError("malformed identity-only historical partition")
    if set(partition.outer_seed.astype(int)) != {seed}:
        raise ValueError("partition seed mismatch")
    if partition.role.value_counts().to_dict() != {"u0": 2997, "test": 417, "validation": 416, "l0": 333}:
        raise ValueError("historical partition counts changed")
    labeled_ids = np.load(directory / "labeled_ids.npy", allow_pickle=False).astype(str).tolist()
    unlabeled_ids = np.load(directory / "unlabeled_ids.npy", allow_pickle=False).astype(str).tolist()
    labeled_indices = np.load(directory / "labeled_indices.npy", allow_pickle=False).astype(np.int64)
    unlabeled_indices = np.load(directory / "unlabeled_indices.npy", allow_pickle=False).astype(np.int64)
    val = partition.loc[partition.role.eq("validation")]
    validation_ids = val.sample_id.astype(str).tolist()
    validation_indices = val.canonical_index.to_numpy(np.int64)
    l0_indices = partition.loc[partition.role.eq("l0"), "canonical_index"].to_numpy(np.int64)
    if len(labeled_ids) != budget or len(labeled_ids) + len(unlabeled_ids) != 3330:
        raise ValueError("historical label budget mismatch")
    if len(set(labeled_ids + unlabeled_ids)) != 3330:
        raise ValueError("historical L/U partition is not disjoint and unique")
    identity = dict(zip(partition.sample_id, partition.canonical_index.astype(int)))
    for ids, indices in ((labeled_ids, labeled_indices), (unlabeled_ids, unlabeled_indices)):
        if [identity[sid] for sid in ids] != indices.tolist():
            raise ValueError("ID/canonical-index mapping drift")
    outer_ids = set(partition.loc[partition.role.isin(["l0", "u0"]), "sample_id"])
    if set(labeled_ids + unlabeled_ids) != outer_ids:
        raise ValueError("state L/U includes rows outside historical train pool")
    if not set(l0_indices) <= set(labeled_indices):
        raise ValueError("historical L0 was not retained")
    input_contract = _json(directory / "input_contract.json")
    frozen = _json(directory / "contract.json")
    fit = _json(directory / "model/fit_audit.json")
    expected_input = {"outer_seed": seed, "method": "lcmd", "round": round_index,
                      "active_label_count": budget, "L_t_ids_hash": _ids_hash(labeled_ids),
                      "U_t_ids_hash": _ids_hash(unlabeled_ids),
                      "ordered_L_t_hash": stable_hash(labeled_ids),
                      "ordered_U_t_hash": stable_hash(unlabeled_ids), "test_truth_access_count": 0}
    if any(input_contract.get(k) != v for k, v in expected_input.items()):
        raise ValueError("round input contract identity mismatch")
    if frozen["input"] != input_contract or frozen["test_truth_access_count"] != 0:
        raise ValueError("round freeze mismatch")
    if fit["train_ids_hash"] != _ids_hash(labeled_ids) or fit["validation_ids_hash"] != _ids_hash(validation_ids):
        raise ValueError("checkpoint training/validation IDs differ from state")
    if fit["train_rows"] != budget or fit["validation_rows"] != 416 or fit["test_labels_used_for_fit_or_checkpoint_selection"] != 0:
        raise ValueError("checkpoint label boundary violation")
    checkpoint_hash = file_hash(checkpoint)
    if checkpoint_hash != fit["checkpoint_sha256"] or checkpoint_hash != frozen["checkpoint_sha256"]:
        raise ValueError("checkpoint bytes differ from frozen state")
    for filename in ("labeled_ids.npy", "unlabeled_ids.npy", "labeled_indices.npy", "unlabeled_indices.npy",
                     "input_contract.json", "model/fit_audit.json", "model/best.pt"):
        if file_hash(directory / filename) != frozen["files"][filename]:
            raise ValueError("frozen state artifact drift: " + filename)
    metadata = _json(raw_cache.with_suffix(".npz.contract.json"))
    contract = metadata["contract"]
    acquisition_contract = _json(acquisition / "contract.json")
    for key, value in acquisition_contract["input"].items():
        if contract.get(key) != value:
            raise ValueError("raw bank/acquisition contract mismatch: " + key)
    expected = {"outer_seed": seed, "method": "lcmd", "source_round": round_index,
                "acquisition_round": round_index + 1, "batch_size": 32,
                "L_t_ids_hash": _ids_hash(labeled_ids), "U_t_ids_hash": _ids_hash(unlabeled_ids),
                "evaluation_checkpoint_sha256": checkpoint_hash,
                "evaluation_checkpoint_state_hash": fit["checkpoint_state_hash"],
                "endpoint_scales": input_contract["endpoint_scales"],
                "feature_recomputed_from_current_model": True, "centers_are_current_L_t": True,
                "dimension": 512, "sketch_seed": seed + 4_000_037,
                "ordered_indices": np.r_[labeled_indices, unlabeled_indices].tolist(),
                "kind": "current_full_network_q50_gradient_CountSketch", "test_truth_access_count": 0}
    if any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError("raw cache belongs to another checkpoint/state")
    l0_ids = partition.loc[partition.role.eq("l0"), "sample_id"].astype(str).tolist()
    if input_contract["endpoint_scales_fit_ids_hash"] != _ids_hash(l0_ids):
        raise ValueError("gradient endpoint scales were not fit on the historical L0")
    if contract["scales"] != [contract["endpoint_scales"][name] for name in ("V1", "V2")]:
        raise ValueError("raw cache scale aliases disagree")
    if metadata["contract_hash"] != stable_hash(contract) or metadata["sha256"] != file_hash(raw_cache):
        raise ValueError("raw bank integrity failure")
    audit = {"seed": seed, "budget": budget, "round": round_index, "trajectory": "historical_corrected_raw_LCMD32",
             "checkpoint": str(checkpoint.relative_to(root)), "checkpoint_sha256": checkpoint_hash,
             "checkpoint_state_hash": fit["checkpoint_state_hash"],
             "raw_cache": str(raw_cache.relative_to(root)), "raw_cache_sha256": metadata["sha256"],
             "partition": str(partition_path.relative_to(root)), "partition_sha256": file_hash(partition_path),
             "L_t_ids_hash": _ids_hash(labeled_ids), "U_t_ids_hash": _ids_hash(unlabeled_ids),
             "validation_ids_hash": _ids_hash(validation_ids), "labeled_rows": budget, "pool_rows": len(unlabeled_ids),
             "validation_rows": len(validation_ids), "historical_cw_available": False,
             "raw_sketch_seed": contract["sketch_seed"], "dimension": 512,
             "endpoint_scales": contract["endpoint_scales"], "label_table_reads": 0,
             "test_truth_access_count": 0, "new_gnn_fits": 0,
             "validation_previously_used_for_checkpoint_selection": True}
    return StateArtifacts(root, seed, budget, round_index, directory, checkpoint, raw_cache,
                          labeled_ids, unlabeled_ids, validation_ids, labeled_indices,
                          unlabeled_indices, validation_indices, l0_indices, partition, audit)


def load_raw_cache(state: StateArtifacts, indices: Sequence[int] | None = None) -> np.ndarray:
    """Return float64 raw features; requests outside cached L+U are rejected."""
    if file_hash(state.raw_cache) != state.audit["raw_cache_sha256"]:
        raise ValueError("raw cache changed after state audit")
    expected = np.r_[state.labeled_indices, state.unlabeled_indices]
    with np.load(state.raw_cache, allow_pickle=False) as bank:
        if set(bank.files) != {"features", "canonical_indices"} or not np.array_equal(bank["canonical_indices"], expected):
            raise ValueError("raw cache row contract mismatch")
        features = bank["features"].astype(np.float64)
    if features.shape != (3330, 512) or not np.isfinite(features).all():
        raise ValueError("invalid historical raw feature matrix")
    if indices is None:
        return features
    wanted = state.validate_indices(indices)
    positions = {int(index): i for i, index in enumerate(expected)}
    if not set(wanted) <= set(positions):
        raise ValueError("requested raw features are not in the historical bank")
    return features[[positions[int(index)] for index in wanted]]


def load_scrubbed_graphs(root: Path | str, seed: int):
    """Reuse the sealed historical graph cache and enforce zero sentinel labels."""
    import torch
    root = Path(root).resolve()
    directory = root / BASE / f"runtime/seed_{int(seed)}"
    path = directory / "scrubbed_graphs.pt"
    receipt = _json(path.with_suffix(".pt.contract.json"))
    context = _json(directory / "context.json")
    if receipt["contract"] != context["contract"] or receipt["contract_hash"] != stable_hash(receipt["contract"]):
        raise ValueError("scrubbed graph context contract mismatch")
    if receipt["sha256"] != file_hash(path):
        raise ValueError("scrubbed graph bytes changed")
    atom, angle = torch.load(path, map_location="cpu", weights_only=False)
    if len(atom) != 4163 or len(angle) != len(atom):
        raise ValueError("scrubbed graph row count mismatch")
    if any(torch.count_nonzero(item.y).item() for item in atom):
        raise PermissionError("graph cache carries response labels")
    if any(int(item.canonical_position) != i for i, item in enumerate(atom)):
        raise ValueError("scrubbed graph row mapping mismatch")
    return atom, angle


def load_model(state: StateArtifacts):
    """Load only the audited frozen predictor; no optimizer is constructed."""
    from qgeognn_al.models import load_predictor_checkpoint
    from qgeognn_al.active_learning_v2.gradient_features import state_dict_hash
    if file_hash(state.checkpoint) != state.audit["checkpoint_sha256"]:
        raise ValueError("checkpoint changed after audit")
    model = load_predictor_checkpoint(state.checkpoint, device="cpu")
    if state_dict_hash(model) != state.audit["checkpoint_state_hash"]:
        raise ValueError("loaded predictor state hash mismatch")
    return model


def _sentinel_check(atom, indices):
    import torch
    if any(torch.count_nonzero(atom[int(i)].y).item() for i in indices):
        raise PermissionError("feature extraction requires zero-sentinel graph labels")


def _validate_model(state: StateArtifacts, model):
    from qgeognn_al.active_learning_v2.gradient_features import state_dict_hash
    if state_dict_hash(model) != state.audit["checkpoint_state_hash"]:
        raise ValueError("provided predictor belongs to another frozen state")
    model.eval()


def extract_hidden(state: StateArtifacts, atom, angle, indices: Sequence[int], *, model=None,
                   batch_size: int = 256) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reuse the established 128D pre-head extractor and frozen q50 predictor."""
    from qgeognn_al.active_learning_v2.coverage import extract_representations
    from qgeognn_al.active_learning_v2.runner import predict_outputs
    indices = state.validate_indices(indices)
    _sentinel_check(atom, indices)
    model = load_model(state) if model is None else model
    _validate_model(state, model)
    started = time.perf_counter()
    hidden = extract_representations(model, atom, angle, indices.tolist(), batch_size=batch_size).astype(np.float64)
    output, order = predict_outputs(model, atom, angle, indices, batch_size=batch_size)
    if not np.array_equal(order, indices) or hidden.shape != (len(indices), 128):
        raise ValueError("hidden extraction shape/order mismatch")
    return hidden, output[:, [1, 4]].astype(np.float64), {
        "definition": "global_add_pool(backbone_nodes)+typed_condition_residual; before Linear(128,6)",
        "dimension": 128, "shape": list(hidden.shape), "normalization": "none; downstream train-only normalization",
        "model_compute_dtype": "float32", "output_dtype": "float64", "model_eval": True,
        "elapsed_seconds": time.perf_counter() - started, "new_gnn_fits": 0, "label_access_count": 0}


def extract_gradient(state: StateArtifacts, atom, angle, indices: Sequence[int], *, kind: str,
                     l0_targets: np.ndarray | None = None, model=None, progress=None) -> tuple[np.ndarray, dict]:
    """Reuse historical raw rows; calculate only missing raw rows or same-state CW."""
    from qgeognn_al.active_learning_v2.gradient_features import (
        extract_q50_gradient_sketches, extract_linear_output_gradient_sketches)
    from qgeognn_al.active_learning_v2.gradient_transforms import center_width_transform
    indices = state.validate_indices(indices)
    _sentinel_check(atom, indices)
    if kind not in {"raw", "cw"}:
        raise ValueError("gradient kind must be raw or cw")
    scales = tuple(float(state.audit["endpoint_scales"][name]) for name in ("V1", "V2"))
    sketch_seed = state.seed + 4_000_037
    started = time.perf_counter()
    cached_positions = set(np.r_[state.labeled_indices, state.unlabeled_indices].astype(int))
    reused = np.array([i in cached_positions and kind == "raw" for i in indices], dtype=bool)
    result = np.empty((len(indices), 512), dtype=np.float64)
    audit: dict[str, Any] = {"kind": kind, "dimension": 512, "sketch_seed": sketch_seed,
                             "historical_rows_reused": int(reused.sum()), "computed_rows": int((~reused).sum()),
                             "checkpoint_sha256": state.audit["checkpoint_sha256"],
                             "model_compute_dtype": "float32", "output_dtype": "float64",
                             "new_gnn_fits": 0, "label_access_count": 0}
    if reused.any():
        result[reused] = load_raw_cache(state, indices[reused])
    if (~reused).any():
        model = load_model(state) if model is None else model
        _validate_model(state, model)
        if kind == "raw":
            extraction = extract_q50_gradient_sketches(model, atom, angle, indices[~reused], scales,
                dimension=512, sketch_seed=sketch_seed, progress=progress)
        else:
            targets = np.asarray(l0_targets, dtype=np.float64)
            if targets.shape != (333, 2):
                raise ValueError("CW requires exactly the authorized historical L0 targets, ordered as state.l0_indices")
            if not np.allclose(targets.std(axis=0, ddof=0), scales, rtol=1e-6, atol=1e-5):
                raise ValueError("L0 target scales do not match the frozen historical checkpoint")
            transform, transform_audit = center_width_transform(targets)
            extraction = extract_linear_output_gradient_sketches(model, atom, angle, indices[~reused], transform,
                dimension=512, sketch_seed=sketch_seed, progress=progress)
            audit["output_transform"] = transform_audit
            audit["l0_canonical_indices_hash"] = stable_hash(state.l0_indices.tolist())
            audit["l0_targets_float64_sha256"] = hashlib.sha256(np.ascontiguousarray(targets).tobytes()).hexdigest()
        result[~reused] = extraction.features.astype(np.float64)
        audit["extraction"] = extraction.audit
    if not np.isfinite(result).all():
        raise ValueError("non-finite gradient representation")
    audit["elapsed_seconds"] = time.perf_counter() - started
    return result, audit


def structured_features(feature_frame: pd.DataFrame, train_indices: Sequence[int]) -> tuple[np.ndarray, dict]:
    """Morgan2048 plus existing eluent/amount/volume encoding and solvent one-hot.

    ``train_indices`` are positional indices into the supplied feature-only frame.
    Binary molecular and categorical features remain unscaled. Continuous
    conditions use means/scales from the current L_t rows only.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from qgeognn_al.data import condition_matrix, CONDITION_FEATURE_NAMES, fit_standardizer, apply_standardizer
    if set(feature_frame) != set(FEATURE_COLUMNS):
        raise PermissionError("structured encoder accepts exactly the explicit feature-only schema")
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if not len(train_indices) or len(set(train_indices)) != len(train_indices) or np.any(train_indices < 0) or np.any(train_indices >= len(feature_frame)):
        raise ValueError("valid nonempty unique training positions required")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = {}
    for smiles in feature_frame.canonical_smiles.astype(str).unique():
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError("invalid canonical SMILES")
        vector = np.zeros(2048, dtype=np.float64)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), vector)
        fingerprints[smiles] = vector
    molecular = np.vstack([fingerprints[s] for s in feature_frame.canonical_smiles.astype(str)])
    # Reuse the established condition encoder; replace its ordinal solvent code.
    conditions = condition_matrix(feature_frame).astype(np.float64)
    solvent = conditions[:, 6].astype(int)
    if np.any(solvent < 0) or np.any(solvent > 2):
        raise ValueError("unknown loading solvent")
    continuous = conditions[:, [0, 1, 2, 3, 4, 5, 7, 8]]
    mean, scale = fit_standardizer(continuous[train_indices])
    continuous = apply_standardizer(continuous, mean, scale)
    features = np.column_stack([molecular, continuous, np.eye(3, dtype=np.float64)[solvent]]).astype(np.float64)
    if not np.isfinite(features).all():
        raise ValueError("non-finite structured features")
    return features, {"dimension": features.shape[1], "morgan_radius": 2, "morgan_bits": 2048,
                      "molecular_scaling": "binary_unscaled", "solvent": ["PE", "EA", "DCM"],
                      "condition_encoder": "qgeognn_al.data.condition_matrix",
                      "continuous_names": [CONDITION_FEATURE_NAMES[i] for i in (0, 1, 2, 3, 4, 5, 7, 8)],
                      "continuous_mean": mean.tolist(), "continuous_scale": scale.tolist(),
                      "standardizer_fit_rows": len(train_indices), "standardizer_fit_role": "current_L_t_only",
                      "label_access_count": 0, "dtype": "float64"}
