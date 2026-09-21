"""Frozen protocol for the two-seed Gradient+Latent Fusion-MaxDet study."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path
import shutil

import pandas as pd

from ..artifacts import sha256_file
from ..resources import ROOT, SOURCE_DATA, SOURCE_GRAPH_CACHE
from ..training.predictor import atomic_json
from .benchmark_protocol import CONFIRMATION_SEEDS as HISTORICAL_SEEDS, FEATURE_COLUMNS, TRAINING_CONFIG, load_features
from .fused_features import ALPHA, FUSED_DIMENSION, GRADIENT_DIMENSION, LATENT_DIMENSION
from .maxdet_study import BASELINE, historical_round0_paths
from .protocol import ids_hash, stable_hash, validate_row_protocol
from .sequential_protocol import (
    ACQUISITION_ROUNDS, ACTIVE_LABEL_BUDGETS, BATCH_SIZE, FINAL_ACTIVE_LABELS,
    INITIAL_ACTIVE_LABELS, INITIAL_POOL_ROWS, VALIDATION_ROWS,
)


STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_fused_maxdet_b32"
CONFIRMATION_SEEDS = (157, 6101)
METHOD = "gradient_latent_fusion_maxdet"


def code_paths() -> tuple[Path, ...]:
    return tuple(ROOT / value for value in (
        "scripts/studies/run_qgeognn_v2_row_fused_maxdet_b32.py",
        "src/qgeognn_al/active_learning_v2/fused_features.py",
        "src/qgeognn_al/active_learning_v2/fused_maxdet_study.py",
        "src/qgeognn_al/active_learning_v2/fused_maxdet_runner.py",
        "src/qgeognn_al/active_learning_v2/fused_maxdet_reporting.py",
        "src/qgeognn_al/active_learning_v2/gradient_features.py",
        "src/qgeognn_al/active_learning_v2/coverage.py",
        "src/qgeognn_al/active_learning_v2/maxdet.py",
        "src/qgeognn_al/active_learning_v2/runner.py",
    ))


def code_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths() if path.exists()}


def protocol_record() -> dict[str, object]:
    return {
        "study": STUDY.name,
        "method": METHOD,
        "historical_confirmation_seeds": list(HISTORICAL_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "row_split": True,
        "initial_active_labels": INITIAL_ACTIVE_LABELS,
        "initial_pool_rows": INITIAL_POOL_ROWS,
        "validation_rows": VALIDATION_ROWS,
        "batch_size": BATCH_SIZE,
        "acquisition_rounds": ACQUISITION_ROUNDS,
        "active_label_budgets": list(ACTIVE_LABEL_BUDGETS),
        "final_active_labels": FINAL_ACTIVE_LABELS,
        "gradient_dimension": GRADIENT_DIMENSION,
        "latent_dimension": LATENT_DIMENSION,
        "fused_dimension": FUSED_DIMENSION,
        "alpha": ALPHA,
        "normalization": "s_g/h=sqrt(mean_L_t(||block||^2)); phi=[sqrt(alpha)g/s_g,sqrt(1-alpha)h/s_h]",
        "kernel": "0.5*K_gradient_normalized + 0.5*K_latent_normalized",
        "selector": "conditional_gradient_maxdet(features=phi), unchanged tested selector",
        "training": TRAINING_CONFIG,
        "target_scales": "fixed original L0 population std ddof=0",
        "test_reveal_barrier": "all 2 seeds x 22 prediction points x 21 batches frozen",
        "source_sha256": sha256_file(SOURCE_DATA),
        "graph_cache_sha256": sha256_file(SOURCE_GRAPH_CACHE),
        "feature_columns": list(FEATURE_COLUMNS),
        "code_hashes": code_hashes(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "torch", "torch-geometric", "rdkit", "scipy", "scikit-learn")},
        "python": platform.python_version(),
    }


def prepare(study: Path = STUDY) -> dict[str, object]:
    study = Path(study)
    if (study / "runtime").exists():
        raise RuntimeError("refusing to prepare over an existing fusion runtime")
    data = load_features()
    splits = []
    for seed in CONFIRMATION_SEEDS:
        source = BASELINE / "splits" / f"row_seed_{seed}.csv"
        if not source.exists():
            raise RuntimeError(f"missing matched row split for seed {seed}")
        frame = pd.read_csv(source)
        validate_row_protocol(frame)
        if frame.sample_id.astype(str).tolist() != data.sample_id.astype(str).tolist():
            raise RuntimeError(f"split/source identity drift for seed {seed}")
        destination = study / "splits" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"refusing to replace changed split: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
        splits.append({"outer_seed": seed, "sha256": sha256_file(destination), "l0_ids_hash": ids_hash(frame.loc[frame.role.eq("l0"), "sample_id"]), "u0_ids_hash": ids_hash(frame.loc[frame.role.eq("u0"), "sample_id"]), "validation_ids_hash": ids_hash(frame.loc[frame.role.eq("validation"), "sample_id"]), "test_ids_hash": ids_hash(frame.loc[frame.role.eq("test"), "sample_id"])})
        paths = historical_round0_paths(seed)
        required = (paths["member0_model"], paths["member0_predictions"], paths["member0_audit"], paths["gradient"], paths["gradient_contract"], paths["gradient_audit"], paths["scrubbed_graphs"], paths["context"])
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"round-zero reuse is incomplete for seed {seed}: {missing}")
    record = protocol_record()
    study.mkdir(parents=True, exist_ok=True)
    atomic_json(study / "protocol.json", record)
    atomic_json(study / "splits/split_manifest.json", {"protocol_hash": stable_hash(record), "splits": splits})
    atomic_json(study / "reuse_audit.json", {"status": "PASS", "round_zero_reused": ["model", "gradient_features", "scrubbed_graphs"], "later_rounds_retrained": True, "seeds": list(CONFIRMATION_SEEDS)})
    atomic_json(study / "global_pre_test_freeze.json", {"status": "PENDING_PRE_TEST_EXECUTION", "test_truth_access_count": 0})
    atomic_json(study / "decision.json", {"status": "PENDING_COMPLETE_TRAJECTORIES"})
    atomic_json(study / "pre_test_manifest.json", {"phase": "PRE_TEST_PROTOCOL_FROZEN", "protocol_hash": stable_hash(record), "test_truth_access_count": 0, "files": {str(path.relative_to(ROOT)): sha256_file(path) for path in (*code_paths(), study / "protocol.json", study / "reuse_audit.json")}})
    return {"status": "PREPARED", "protocol_hash": stable_hash(record), "seeds": list(CONFIRMATION_SEEDS)}


def validate_pre_test_manifest(study: Path = STUDY) -> dict[str, object]:
    study = Path(study)
    manifest = json.loads((study / "pre_test_manifest.json").read_text())
    if manifest.get("phase") != "PRE_TEST_PROTOCOL_FROZEN":
        raise RuntimeError("fusion pre-test manifest is not frozen")
    if json.loads((study / "protocol.json").read_text()) != protocol_record():
        raise RuntimeError("fusion protocol differs from implementation")
    for relative, digest in manifest["files"].items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"fusion frozen input drift: {relative}")
    if json.loads((study / "reuse_audit.json").read_text()).get("status") != "PASS":
        raise RuntimeError("fusion reuse audit is not passing")
    return {"status": "PASS", "protocol_hash": manifest["protocol_hash"]}


def historical_round0(seed: int) -> dict[str, Path]:
    return historical_round0_paths(int(seed))


__all__ = ["STUDY", "METHOD", "CONFIRMATION_SEEDS", "prepare", "validate_pre_test_manifest", "historical_round0", "protocol_record"]
