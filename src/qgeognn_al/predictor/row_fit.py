"""Same-initialization Row AL fitting with explicit authorization and receipts.

Migrated from main b1a6ed7a active_learning_v2/{benchmark_protocol,runner}.py.
The model/loss/epoch shuffle/Adam/early-stop/best-checkpoint implementation is
predictor.training.train_source, shared with inference and numerical profiles.
No training starts by importing this module or by preparing a RowDataset.
"""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import stat
import time

import numpy as np
import pandas as pd
import torch

from .model import build_predictor, load_predictor_checkpoint
from .training import atomic_json, seed_everything, train_source
from .inference import predict_outputs
from .row import safe_output
from ..infra.artifacts import sha256_file
from ..infra.identity import ids_hash
from ..al.representations.latent import predictor_state_hash

TRAINING_CONFIG = {
    "model_variant": "qgeognn_v2", "optimizer": "Adam", "learning_rate": 0.001,
    "weight_decay": 0.0, "batch_size": 2048, "maximum_epochs": 1000, "patience": 100,
    "shuffle": "deterministic_each_epoch", "loss_weights": {"V1": 1.0, "V2": 1.0},
    "checkpoint_selection": "validation_combined_normalized_rmse", "test_during_training": False,
    "architecture_changed": False, "head_changed": False, "condition_branch_changed": False,
}
SOURCE_COMMIT = "b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28"


def _seed_member(outer_seed, member):
    if type(outer_seed) is not int or outer_seed < 0 or type(member) is not int or member not in range(3):
        raise ValueError("nonnegative integer outer seed and explicit K3 member 0/1/2 required")


def initialization_seed(outer_seed, member=0):
    _seed_member(outer_seed, member)
    return outer_seed + 2_000_003 + member * 1_000_033


def sketch_seed(outer_seed):
    _seed_member(outer_seed, 0)
    return outer_seed + 4_000_037


def row_training_config(outer_seed, split_sha256, member=0):
    _seed_member(outer_seed, member)
    if not isinstance(split_sha256, str) or len(split_sha256) != 64 or any(c not in "0123456789abcdef" for c in split_sha256):
        raise ValueError("training configuration requires an explicit split SHA256")
    return {**deepcopy(TRAINING_CONFIG), "seed": outer_seed + 3_000_017 + member * 1_000_033,
            "outer_seed": outer_seed, "cpu_threads": 2, "split_sha256": split_sha256}


def configure_row_runtime(*, cpu_threads=2, device="cpu"):
    """Apply the recorded CPU thread count; do not invent GPU determinism."""
    if device != "cpu" or type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("Row AL is qualified only for a positive CPU thread count")
    torch.set_num_threads(cpu_threads)
    return {"device": "cpu", "cpu_threads": torch.get_num_threads(),
            "gpu_training_equivalence": "NOT_VERIFIED",
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def initialize_row_model(normalization, outer_seed, member=0):
    """Use the historical initialization RNG, independent of epoch-order RNG."""
    configure_row_runtime()
    seed_everything(initialization_seed(outer_seed, member))
    model = build_predictor(normalization)
    return model, predictor_state_hash(model)


def legacy_row_state_hash(model):
    """Preserve historical receipt serialization, including shape JSON spaces.

    The public representation API uses a different compact shape encoding.
    Both fingerprints identify the same named tensors; never compare them as
    if the serialization profile were identical.
    """
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _array_hash(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def _contract_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _labels(atom_base, indices, truth):
    result = list(atom_base)
    positions = [int(i) for i in indices]
    values = np.asarray(truth, dtype=np.float32)
    if values.shape != (len(positions), 2) or not np.isfinite(values).all():
        raise ValueError("revealed targets must align with graph positions")
    for position, target in zip(positions, values):
        item = result[position].clone()
        item.y = torch.tensor(np.asarray([target]), dtype=torch.float32)
        result[position] = item
    return result


def _write_predictions(path, sample_ids, predictions):
    values = np.asarray(predictions, dtype=float)
    if values.shape != (len(sample_ids), 6):
        raise ValueError("prediction table must align with six-output model contract")
    frame = pd.DataFrame(values, columns=[f"{t}_{q}" for t in ("V1", "V2") for q in ("q10", "q50", "q90")])
    frame.insert(0, "sample_id", [str(v) for v in sample_ids])
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    return sha256_file(path)


def fit_row_from_initialization(*, atom_base, angle, normalization, preprocessing,
                                train_indices, train_truth, validation_indices,
                                validation_truth, prediction_indices, prediction_sample_ids,
                                initialization_seed, training_config, contract, runtime,
                                allow_training=False, forbidden_output_roots=()):
    """Fit one authorized arm or verify/reuse an exactly matching completed fit.

    Caller supplies only revealed L and validation targets. Completion requires
    checkpoint + predictions + audit and matching input/output hashes. Any partial
    runtime fails closed: it cannot silently launch a duplicate fit. This adapter
    intentionally does not reseed with training_config['seed'] before train_source;
    legacy uses that seed only for its per-epoch NumPy permutation.
    """
    if type(allow_training) is not bool:
        raise ValueError("training authorization must be an explicit boolean capability")
    runtime = safe_output(runtime, forbidden_output_roots)
    checkpoint_path, prediction_path, audit_path = (runtime / n for n in ("best.pt", "predictions.csv.gz", "fit_audit.json"))
    if runtime.exists() and not runtime.is_dir():
        raise ValueError("fit runtime must be an independent directory")
    # Never follow a restored receipt link into another experiment/worktree.
    # Completed read-only checkpoints remain valid, including K3 member files.
    for path in (checkpoint_path, prediction_path, audit_path,
                 runtime / "fit_started.json", runtime / "history.csv"):
        if path.is_symlink():
            raise ValueError("fit receipt symlinks are forbidden")
        if path.exists():
            metadata = path.stat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValueError("fit receipt must be an independent regular file, not a hardlink")
    complete = all(p.is_file() for p in (checkpoint_path, prediction_path, audit_path))
    if runtime.exists() and any(runtime.iterdir()) and not complete:
        raise RuntimeError("ambiguous partial fit requires inspection, not automatic retraining")
    if not complete and not allow_training:
        raise PermissionError("new scientific training requires separate explicit authorization")
    train_positions = [int(i) for i in train_indices]
    val_positions = [int(i) for i in validation_indices]
    prediction_positions = [int(i) for i in prediction_indices]
    for positions in (train_positions, val_positions, prediction_positions):
        if len(set(positions)) != len(positions) or any(i < 0 or i >= len(atom_base) for i in positions):
            raise ValueError("graph positions must be unique and within the canonical rows")
    if any("y" in graph for graph in (*atom_base, *angle)):
        raise ValueError("base graphs must be label-free; targets enter only as authorized arrays")
    if not train_positions or not val_positions or set(train_positions) & set(val_positions) or len(atom_base) != len(angle):
        raise ValueError("nonempty disjoint training/validation graph roles are required")
    if len(contract["train_sample_ids"]) != len(train_positions) or len(contract["validation_sample_ids"]) != len(val_positions):
        raise ValueError("fit identity rows must align with graph positions")
    if len(prediction_sample_ids) != len(prediction_positions):
        raise ValueError("prediction identities must align with canonical positions")
    if any(len(set(values)) != len(values) for values in (contract["train_sample_ids"], contract["validation_sample_ids"], prediction_sample_ids)):
        raise ValueError("fit sample identities must be unique within each role")
    if set(contract["train_sample_ids"]) & set(contract["validation_sample_ids"]):
        raise ValueError("fit identity roles overlap")
    train_values, val_values = np.asarray(train_truth, dtype=np.float32), np.asarray(validation_truth, dtype=np.float32)
    if train_values.shape != (len(train_positions), 2) or val_values.shape != (len(val_positions), 2) or not np.isfinite(train_values).all() or not np.isfinite(val_values).all():
        raise ValueError("authorized fit labels must have two finite values per row")
    if not isinstance(initialization_seed, int) or initialization_seed < 0:
        raise ValueError("initialization seed must be explicit")
    for name in ("seed", "split_sha256", "learning_rate", "weight_decay", "batch_size", "maximum_epochs", "patience"):
        if name not in training_config:
            raise ValueError("missing frozen training setting: " + name)
    configure_row_runtime(cpu_threads=int(training_config.get("cpu_threads", 2)))
    seed_everything(initialization_seed)
    initial_model = build_predictor(normalization)
    initialization_hash = legacy_row_state_hash(initial_model)
    public_initialization_hash = predictor_state_hash(initial_model)
    expected = {**dict(contract), "initialization_seed": initialization_seed,
                "initialization_hash": initialization_hash,
                "train_ids_hash": ids_hash(str(v) for v in contract["train_sample_ids"]),
                "validation_ids_hash": ids_hash(str(v) for v in contract["validation_sample_ids"]),
                "training_config": dict(training_config), "preprocessing": dict(preprocessing),
                "normalization": asdict(normalization), "train_truth_hash": _array_hash(train_values),
                "validation_truth_hash": _array_hash(val_values),
                "ordered_train_ids": list(contract["train_sample_ids"]),
                "ordered_validation_ids": list(contract["validation_sample_ids"]),
                "ordered_train_indices": train_positions, "ordered_validation_indices": val_positions,
                "prediction_sample_ids": list(prediction_sample_ids), "prediction_indices": prediction_positions}
    expected_hash = _contract_hash(expected)
    if complete:
        audit = json.loads(audit_path.read_text())
        if (audit.get("fit_contract_hash") != expected_hash or audit.get("checkpoint_sha256") != sha256_file(checkpoint_path)
                or audit.get("prediction_sha256") != sha256_file(prediction_path) or audit.get("initialization_hash") != initialization_hash):
            raise RuntimeError("refusing incompatible completed fit")
        return {**audit, "public_initialization_hash": public_initialization_hash,
                "public_checkpoint_state_hash": predictor_state_hash(load_predictor_checkpoint(checkpoint_path))}
    runtime.mkdir(parents=True, exist_ok=True)
    atomic_json(runtime / "fit_started.json", {"status": "FIT_STARTED", "fit_contract_hash": expected_hash,
                                              "training_authorized": True})
    atom = _labels(atom_base, train_positions, train_values)
    atom = _labels(atom, val_positions, val_values)
    started = time.perf_counter()
    history, best_epoch = train_source(initial_model, atom, angle, np.asarray(train_positions, dtype=int),
                                     np.asarray(val_positions, dtype=int), dict(preprocessing), dict(training_config), checkpoint_path)
    pd.DataFrame(history).to_csv(runtime / "history.csv", index=False)
    model = load_predictor_checkpoint(checkpoint_path)
    prediction = predict_outputs(model, atom_base, angle, prediction_positions, int(training_config["batch_size"]))
    prediction_hash = _write_predictions(prediction_path, prediction_sample_ids, prediction)
    audit = {"fit_contract_hash": expected_hash, "initialization_seed": initialization_seed,
             "initialization_hash": initialization_hash, "train_ids_hash": expected["train_ids_hash"],
             "validation_ids_hash": expected["validation_ids_hash"], "train_rows": len(train_positions),
             "validation_rows": len(val_positions), "best_epoch": int(best_epoch), "epochs_run": len(history),
             "best_validation_combined_normalized_rmse": float(min(row["validation_selection_score"] for row in history)),
             "checkpoint_sha256": sha256_file(checkpoint_path), "checkpoint_state_hash": legacy_row_state_hash(model),
             "public_initialization_hash": public_initialization_hash,
             "public_checkpoint_state_hash": predictor_state_hash(model),
             "prediction_sha256": prediction_hash, "prediction_rows": len(prediction_positions),
             "training_seconds": time.perf_counter() - started, "test_labels_used_for_fit_or_checkpoint_selection": 0,
             "test_graph_label_state": "absent", "model_variant": "qgeognn_v2",
             "parameter_count": sum(p.numel() for p in model.parameters())}
    atomic_json(audit_path, audit)
    return audit
