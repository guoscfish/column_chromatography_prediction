"""Frozen protocol helpers for the row-first Gradient-MaxDet study.

The study deliberately lives beside, rather than inside, the historical
sequential runner.  This keeps the already-published LCMD/Hybrid trajectory
contracts immutable while allowing exact reuse of their round-zero artifacts.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from ..artifacts import sha256_file
from ..resources import ROOT, SOURCE_DATA, SOURCE_GRAPH_CACHE
from ..training.predictor import atomic_json
from .benchmark_protocol import (
    CONFIRMATION_SEEDS as HISTORICAL_CONFIRMATION_SEEDS,
    FEATURE_COLUMNS,
    TRAINING_CONFIG,
    load_features,
)
from .protocol import ids_hash, stable_hash, validate_row_protocol
from .sequential_protocol import (
    ACTIVE_LABEL_BUDGETS,
    BATCH_SIZE,
    FINAL_ACTIVE_LABELS,
    INITIAL_ACTIVE_LABELS,
    INITIAL_POOL_ROWS,
    OUTER_TRAIN_ROWS,
    SKETCH_DIMENSION,
    VALIDATION_ROWS,
)


STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_maxdet_b32"
BASELINE = ROOT / "studies/active_learning/qgeognn_v2_row_sequential_b32"
IVR_BASELINE = ROOT / "studies/active_learning/qgeognn_v2_row_kernel_ivr_b32"
METHODS = ("gradient_maxdet", "u50_gradient_maxdet")
ENSEMBLE_K = 3
UNCERTAINTY_GATE_FRACTION = 0.50
BASE_COMMIT = "e7f97e9b"
ROW_PHASES = (("early", 333, 525), ("middle", 525, 653), ("late", 653, 1005))
# This reduced run is explicitly developmental. The five-seed historical cohort
# remains available for baseline comparisons, but only these two seeds are run
# through the new MaxDet methods.
CONFIRMATION_SEEDS = (157, 6101)


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def study_code_paths() -> tuple[Path, ...]:
    relative = (
        "scripts/studies/run_qgeognn_v2_row_maxdet_b32.py",
        "src/qgeognn_al/active_learning_v2/maxdet.py",
        "src/qgeognn_al/active_learning_v2/maxdet_study.py",
        "src/qgeognn_al/active_learning_v2/maxdet_runner.py",
        "src/qgeognn_al/active_learning_v2/maxdet_reporting.py",
        "src/qgeognn_al/active_learning_v2/gradient_features.py",
        "src/qgeognn_al/active_learning_v2/runner.py",
        "src/qgeognn_al/active_learning_v2/uncertainty.py",
        "src/qgeognn_al/active_learning_v2/efficiency_reporting.py",
    )
    return tuple(ROOT / name for name in relative)


def code_hashes() -> dict[str, str]:
    missing = [path for path in study_code_paths() if not path.exists()]
    if missing:
        raise RuntimeError(f"MaxDet study code is incomplete: {missing}")
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in study_code_paths()}


def package_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "torch", "torch-geometric", "rdkit", "scipy", "scikit-learn")
    }


def protocol_record() -> dict[str, object]:
    return {
        "study": STUDY.name,
        "audit_head_commit": BASE_COMMIT,
        "evidence_status": "development_due_to_historical_row_test_exposure",
        "cohort_status": "two_seed_developmental_subset_of_historical_cohort",
        "historical_confirmation_seeds": list(HISTORICAL_CONFIRMATION_SEEDS),
        "research_priority": "ROW_SPLIT_PERFORMANCE_FIRST",
        "compound_split_role": "auxiliary_diagnostic_only_not_an_optimization_target",
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "methods": list(METHODS),
        "batch_size": BATCH_SIZE,
        "active_label_budgets": list(ACTIVE_LABEL_BUDGETS),
        "initial_active_labels": INITIAL_ACTIVE_LABELS,
        "final_active_labels": FINAL_ACTIVE_LABELS,
        "initial_pool_rows": INITIAL_POOL_ROWS,
        "outer_train_rows": OUTER_TRAIN_ROWS,
        "validation_rows": VALIDATION_ROWS,
        "gradient_sketch_dimension": SKETCH_DIMENSION,
        "gradient_sketch": "authoritative_full_network_q50_CountSketch",
        "maxdet": {
            "normalization": "sqrt(mean_L_t(||phi||^2))",
            "information": "I + sum_L_t psi psi^T",
            "gain": "log1p(psi_x^T A_inverse psi_x)",
            "batch": "greedy_conditional_rank_one_updates",
            "ridge": 1.0,
            "dtype": "float64",
            "tie_break": "canonical_current_U_t_order",
        },
        "uncertainty_gate": {
            "fraction": UNCERTAINTY_GATE_FRACTION,
            "size": "ceil(0.50 * current_U_t_rows)",
            "signal": "existing_K3_q50_ensemble_disagreement_on_V1_V2_scaled_by_fixed_L0_scales",
            "tie_break": "canonical_current_U_t_order",
            "maxdet_candidate_order": "canonical_current_U_t_order_after_gate_membership",
        },
        "ensemble_K": ENSEMBLE_K,
        "training": TRAINING_CONFIG,
        "target_scales": "fixed_original_L0_population_std_ddof_0",
        "row_phases": [{"name": name, "start": start, "stop": stop} for name, start, stop in ROW_PHASES],
        "test_reveal_barrier": "all_2_executed_seeds_x_2_methods_x_22_predictions_and_21_batches_frozen",
        "historical_baselines": ["random", "hybrid", "lcmd", "kernel_ivr"],
        "source_sha256": sha256_file(SOURCE_DATA),
        "graph_cache_sha256": sha256_file(SOURCE_GRAPH_CACHE),
        "feature_columns": list(FEATURE_COLUMNS),
        "code_hashes": code_hashes(),
        "packages": package_versions(),
        "python": platform.python_version(),
    }


def historical_round0_paths(seed: int) -> dict[str, Path]:
    base = BASELINE / "runtime" / f"seed_{int(seed)}"
    return {
        "context": base / "context.json",
        "scrubbed_graphs": base / "scrubbed_graphs.pt",
        "member0_model": base / "lcmd/round_00/model/best.pt",
        "member0_predictions": base / "lcmd/round_00/model/predictions.csv.gz",
        "member0_audit": base / "lcmd/round_00/model/fit_audit.json",
        "hybrid_member0_model": base / "hybrid/round_00/model/best.pt",
        "gradient": base / "lcmd/round_00/acquisition_artifacts/current_gradient_features.npz",
        "gradient_contract": base / "lcmd/round_00/acquisition_artifacts/current_gradient_features.npz.contract.json",
        "gradient_audit": base / "lcmd/round_00/acquisition_artifacts/gradient_audit.json",
        "ensemble_member1_model": base / "hybrid/round_00/acquisition_artifacts/ensemble_member_1/best.pt",
        "ensemble_member1_predictions": base / "hybrid/round_00/acquisition_artifacts/ensemble_member_1/predictions.csv.gz",
        "ensemble_member1_audit": base / "hybrid/round_00/acquisition_artifacts/ensemble_member_1/fit_audit.json",
        "ensemble_member2_model": base / "hybrid/round_00/acquisition_artifacts/ensemble_member_2/best.pt",
        "ensemble_member2_predictions": base / "hybrid/round_00/acquisition_artifacts/ensemble_member_2/predictions.csv.gz",
        "ensemble_member2_audit": base / "hybrid/round_00/acquisition_artifacts/ensemble_member_2/fit_audit.json",
    }


def _validate_split_and_reuse(seed: int) -> dict[str, object]:
    split_path = BASELINE / "splits" / f"row_seed_{seed}.csv"
    partition = pd.read_csv(split_path)
    validate_row_protocol(partition)
    data = load_features()
    if partition.sample_id.astype(str).tolist() != data.sample_id.astype(str).tolist():
        raise RuntimeError(f"historical row split identity drift for seed {seed}")
    expected = {"l0": 333, "u0": 2997, "validation": 416, "test": 417}
    if partition.role.value_counts().to_dict() != expected:
        raise RuntimeError(f"historical row split counts drift for seed {seed}")
    paths = historical_round0_paths(seed)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"round-zero reuse is incomplete for seed {seed}: {missing}")
    member0 = _json(paths["member0_audit"])
    hybrid0 = _json(base := paths["hybrid_member0_model"].parent / "fit_audit.json")
    if member0["checkpoint_state_hash"] != hybrid0["checkpoint_state_hash"]:
        raise RuntimeError(f"LCMD/Hybrid round-zero member0 mismatch for seed {seed}")
    if sha256_file(paths["member0_model"]) != sha256_file(paths["hybrid_member0_model"]):
        raise RuntimeError(f"LCMD/Hybrid round-zero checkpoint bytes differ for seed {seed}")
    context = _json(paths["context"])["contract"]
    if context["source_sha256"] != sha256_file(SOURCE_DATA):
        raise RuntimeError(f"round-zero source drift for seed {seed}")
    with np.load(paths["gradient"]) as gradient:
        indices = gradient["canonical_indices"]
        features = gradient["features"]
    expected_indices = partition.loc[partition.role.isin(["l0", "u0"]), "canonical_index"].to_numpy(int)
    # Historical current order is L0 followed by U0, not canonical row order.
    expected_indices = np.r_[
        partition.loc[partition.role.eq("l0"), "canonical_index"].to_numpy(int),
        partition.loc[partition.role.eq("u0"), "canonical_index"].to_numpy(int),
    ]
    if not np.array_equal(indices, expected_indices) or features.shape != (3330, SKETCH_DIMENSION):
        raise RuntimeError(f"round-zero gradient bank order/shape drift for seed {seed}")
    gradient_contract = _json(paths["gradient_contract"])["contract"]
    if gradient_contract.get("sketch_seed") != seed + 4_000_037:
        raise RuntimeError(f"round-zero CountSketch seed drift for seed {seed}")
    u0_ids = partition.loc[partition.role.eq("u0"), "sample_id"].astype(str).tolist()
    for member in (1, 2):
        table = pd.read_csv(paths[f"ensemble_member{member}_predictions"])
        if table.sample_id.astype(str).tolist() != u0_ids:
            raise RuntimeError(f"round-zero ensemble pool identity drift for seed {seed} member {member}")
    return {
        "outer_seed": int(seed),
        "split_sha256": sha256_file(split_path),
        "l0_ids_hash": ids_hash(partition.loc[partition.role.eq("l0"), "sample_id"]),
        "u0_ids_hash": ids_hash(partition.loc[partition.role.eq("u0"), "sample_id"]),
        "validation_ids_hash": ids_hash(partition.loc[partition.role.eq("validation"), "sample_id"]),
        "test_ids_hash": ids_hash(partition.loc[partition.role.eq("test"), "sample_id"]),
        "member0_checkpoint_sha256": sha256_file(paths["member0_model"]),
        "member0_checkpoint_state_hash": member0["checkpoint_state_hash"],
        "gradient_sha256": sha256_file(paths["gradient"]),
        "gradient_contract_sha256": sha256_file(paths["gradient_contract"]),
        "ensemble_member1_checkpoint_sha256": sha256_file(paths["ensemble_member1_model"]),
        "ensemble_member2_checkpoint_sha256": sha256_file(paths["ensemble_member2_model"]),
        "target_scales": context["preprocessing"]["target_scales"],
        "training_config": TRAINING_CONFIG,
        "reuse_legal": True,
    }


def prepare(study: Path = STUDY, *, allow_existing_runtime: bool = False) -> dict[str, object]:
    """Freeze the new protocol after proving exact historical reuse eligibility."""

    study = Path(study)
    previous_manifest = _json(study / "pre_test_manifest.json") if (study / "pre_test_manifest.json").exists() else {}
    if (study / "runtime").exists() and not allow_existing_runtime:
        raise RuntimeError("refusing to prepare over MaxDet runtime artifacts")
    record = protocol_record()
    study.mkdir(parents=True, exist_ok=True)
    (study / "splits").mkdir(exist_ok=True)
    reuse = []
    for seed in CONFIRMATION_SEEDS:
        reuse.append(_validate_split_and_reuse(seed))
        source = BASELINE / "splits" / f"row_seed_{seed}.csv"
        destination = study / "splits" / source.name
        if destination.exists() and sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"refusing to replace changed split: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
    atomic_json(study / "protocol.json", record)
    atomic_json(study / "reuse_audit.json", {
        "status": "PASS",
        "interpretation": "Only exact round-zero model/gradient/K3 ensemble artifacts are reusable; later Hybrid states are trajectory-specific and forbidden.",
        "historical_work_not_retrained": ["random", "gradient_lcmd", "hybrid", "kernel_ivr"],
        "seeds": reuse,
    })
    atomic_json(study / "global_pre_test_freeze.json", {
        "status": "PENDING_PRE_TEST_EXECUTION",
        "test_truth_access_count": 0,
    })
    atomic_json(study / "decision.json", {"status": "PENDING_COMPLETE_TRAJECTORIES"})
    manifest = {
        "phase": "PRE_TEST_PROTOCOL_FROZEN",
        "head_commit_at_audit": BASE_COMMIT,
        "protocol_hash": stable_hash(record),
        "test_truth_access_count": 0,
        # Existing trajectories carry the old cohort-wide hash. Keep that
        # continuity token separate from the new two-seed protocol hash so
        # completed per-seed artifacts remain resumable and auditable.
        "runtime_protocol_hash": previous_manifest.get("runtime_protocol_hash", previous_manifest.get("protocol_hash", stable_hash(record))),
        "files": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (*study_code_paths(), study / "protocol.json", study / "reuse_audit.json")
        },
    }
    atomic_json(study / "pre_test_manifest.json", manifest)
    return {
        "status": "PREPARED",
        "protocol_hash": stable_hash(record),
        "runtime_protocol_hash": manifest["runtime_protocol_hash"],
        "reuse_seeds": len(reuse),
    }


def validate_pre_test_manifest(study: Path = STUDY) -> dict[str, object]:
    study = Path(study)
    manifest = _json(study / "pre_test_manifest.json")
    if manifest.get("phase") != "PRE_TEST_PROTOCOL_FROZEN":
        raise RuntimeError("MaxDet pre-test manifest is not frozen")
    for relative, digest in manifest["files"].items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"MaxDet frozen input drift: {relative}")
    if _json(study / "protocol.json") != protocol_record():
        raise RuntimeError("MaxDet protocol/code/environment drift")
    if _json(study / "reuse_audit.json").get("status") != "PASS":
        raise RuntimeError("round-zero reuse audit is not passing")
    return {
        "status": "PASS",
        "protocol_hash": manifest["protocol_hash"],
        "runtime_protocol_hash": manifest.get("runtime_protocol_hash", manifest["protocol_hash"]),
    }


def gate_size(pool_size: int) -> int:
    if int(pool_size) < 1:
        raise ValueError("pool_size must be positive")
    return int(math.ceil(UNCERTAINTY_GATE_FRACTION * int(pool_size)))
