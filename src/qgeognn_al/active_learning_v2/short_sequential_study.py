"""Frozen protocol for the two-method 333--429 sequential screen."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd

from ..artifacts import sha256_file
from ..resources import ROOT, SOURCE_DATA, SOURCE_GRAPH_CACHE
from ..training.predictor import atomic_json
from .benchmark_protocol import FEATURE_COLUMNS, TRAINING_CONFIG, load_features
from .maxdet_study import BASELINE, historical_round0_paths
from .protocol import ids_hash, stable_hash, validate_row_protocol


STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_short_sequential_b32"
MAXDET_STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_maxdet_b32"
IVR_STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_kernel_ivr_b32"
INNOVATION_STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_innovation_screen"
METHODS = ("center_width_lcmd", "direction_lcmd")
SEEDS = (157, 6101)
BATCH_SIZE = 32
ACQUISITION_ROUNDS = 3
ACTIVE_LABEL_BUDGETS = (333, 365, 397, 429)
INITIAL_ACTIVE_LABELS = 333
FINAL_ACTIVE_LABELS = 429
SKETCH_DIMENSION = 512
COMPARATORS = ("random", "lcmd", "hybrid", "kernel_ivr", "gradient_maxdet")
DISPLAY_NAMES = {
    "random": "Random",
    "lcmd": "Gradient-LCMD",
    "hybrid": "Hybrid",
    "kernel_ivr": "Kernel-IVR",
    "gradient_maxdet": "Gradient-MaxDet",
    "center_width_lcmd": "Center/Width-LCMD",
    "direction_lcmd": "Direction-LCMD",
}


def code_paths() -> tuple[Path, ...]:
    relative = (
        "scripts/studies/run_qgeognn_v2_row_short_sequential_b32.py",
        "src/qgeognn_al/active_learning_v2/short_sequential_study.py",
        "src/qgeognn_al/active_learning_v2/short_sequential_runner.py",
        "src/qgeognn_al/active_learning_v2/short_sequential_reporting.py",
        "src/qgeognn_al/active_learning_v2/gradient_features.py",
        "src/qgeognn_al/active_learning_v2/gradient_transforms.py",
        "src/qgeognn_al/active_learning_v2/lcmd.py",
        "src/qgeognn_al/active_learning_v2/runner.py",
        "src/qgeognn_al/active_learning_v2/protocol.py",
        "src/qgeognn_al/active_learning_v2/sequential_acquisition.py",
    )
    return tuple(ROOT / value for value in relative)


def code_hashes() -> dict[str, str]:
    missing = [str(path) for path in code_paths() if not path.exists()]
    if missing:
        raise RuntimeError(f"short-sequential implementation is incomplete: {missing}")
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths()}


def protocol_record() -> dict[str, object]:
    return {
        "study": STUDY.name,
        "evidence_class": "TRUNCATED_DEVELOPMENTAL_SCREEN",
        "research_question": "whether one-step rankings persist across three adaptive retraining rounds",
        "split": "ROW",
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "explicitly_excluded_method": "ensemble_uncertainty",
        "batch_size": BATCH_SIZE,
        "acquisition_rounds": ACQUISITION_ROUNDS,
        "active_label_budgets": list(ACTIVE_LABEL_BUDGETS),
        "hard_stop_active_labels": FINAL_ACTIVE_LABELS,
        "gradient_dimension": SKETCH_DIMENSION,
        "gradient_sketch_seed": "outer_seed + 4_000_037",
        "center_width": "grad((V1+V2)/(2*s_C)), grad((V2-V1)/s_W); s_C,s_W are L0 population std",
        "direction": "power_normalize_gradients(raw_q50_gradient, alpha=0.0); zero rows retained",
        "selector": "LCMD-TP with all current L_t rows as centers",
        "retraining": "scratch current-trajectory QGeoGNN-V2 after every frozen batch",
        "training": TRAINING_CONFIG,
        "comparators": list(COMPARATORS),
        "primary_metric": "trapezoidal mean combined_normalized_RMSE AULC over 333--429",
        "test_reveal_barrier": "all 2 methods x 2 seeds x 4 prediction points frozen",
        "source_sha256": sha256_file(SOURCE_DATA),
        "graph_cache_sha256": sha256_file(SOURCE_GRAPH_CACHE),
        "feature_columns": list(FEATURE_COLUMNS),
        "code_hashes": code_hashes(),
    }


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def split_path(seed: int) -> Path:
    return STUDY / "splits" / f"row_seed_{int(seed)}.csv"


def _historical_result_sources() -> dict[str, Path]:
    return {
        "sequential": BASELINE / "results/learning_curve_metrics.csv",
        "maxdet": MAXDET_STUDY / "results/learning_curve_metrics.csv",
        "ivr": IVR_STUDY / "results/learning_curve_metrics.csv",
    }


def audit_reuse() -> dict[str, object]:
    data = load_features()
    records: list[dict[str, object]] = []
    for seed in SEEDS:
        source_split = BASELINE / "splits" / f"row_seed_{seed}.csv"
        partition = pd.read_csv(source_split)
        validate_row_protocol(partition)
        if partition.sample_id.astype(str).tolist() != data.sample_id.astype(str).tolist():
            raise RuntimeError(f"historical split/source identity drift for seed {seed}")
        paths = historical_round0_paths(seed)
        required = {
            "context": paths["context"],
            "scrubbed_graphs": paths["scrubbed_graphs"],
            "checkpoint": paths["member0_model"],
            "predictions": paths["member0_predictions"],
            "fit_audit": paths["member0_audit"],
            "raw_gradient": paths["gradient"],
            "raw_gradient_contract": paths["gradient_contract"],
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise RuntimeError(f"round-0 reuse incomplete for seed {seed}: {missing}")
        context = _json(paths["context"])["contract"]
        fit = _json(paths["member0_audit"])
        if context["source_sha256"] != sha256_file(SOURCE_DATA):
            raise RuntimeError("historical source hash drift")
        if context["source_graph_cache_sha256"] != sha256_file(SOURCE_GRAPH_CACHE):
            raise RuntimeError("historical graph-cache hash drift")
        records.append({
            "outer_seed": seed,
            "split_sha256": sha256_file(source_split),
            "split_hash": stable_hash(partition.to_dict("list")),
            **{f"{role}_ids_hash": ids_hash(partition.loc[partition.role.eq(role), "sample_id"])
               for role in ("l0", "u0", "validation", "test")},
            "checkpoint_sha256": sha256_file(paths["member0_model"]),
            "checkpoint_state_hash": fit["checkpoint_state_hash"],
            "prediction_sha256": sha256_file(paths["member0_predictions"]),
            "raw_gradient_sha256": sha256_file(paths["gradient"]),
            "preprocessing_hash": stable_hash(context["preprocessing"]),
            "target_scale_hash": stable_hash(context["preprocessing"]["target_scales"]),
            "sample_order_hash": stable_hash(data.sample_id.astype(str).tolist()),
            "reuse_legal": True,
        })

    comparator_rows: list[dict[str, object]] = []
    for source_name, path in _historical_result_sources().items():
        if not path.exists():
            raise RuntimeError(f"missing historical comparator table: {path}")
        table = pd.read_csv(path)
        methods = {
            "sequential": ("random", "lcmd", "hybrid"),
            "maxdet": ("gradient_maxdet",),
            "ivr": ("kernel_ivr",),
        }[source_name]
        subset = table.loc[
            table.outer_seed.isin(SEEDS)
            & table.method.isin(methods)
            & table.active_label_count.isin(ACTIVE_LABEL_BUDGETS)
        ]
        for method in methods:
            arm = subset.loc[subset.method.eq(method)]
            expected = len(SEEDS) * len(ACTIVE_LABEL_BUDGETS)
            if len(arm) != expected or arm.duplicated(["outer_seed", "active_label_count"]).any():
                raise RuntimeError(f"incomplete matched comparator: {method}")
            comparator_rows.append({
                "method": method,
                "source": str(path.relative_to(ROOT)),
                "source_sha256": sha256_file(path),
                "rows": len(arm),
                "seeds": list(SEEDS),
                "budgets": list(ACTIVE_LABEL_BUDGETS),
                "identity_consistent": True,
            })
    return {
        "status": "AUDITED_BEFORE_NEW_TRAINING",
        "new_methods": list(METHODS),
        "ensemble_uncertainty_executed": False,
        "same_seed_split_budget_model_acquisition_result_found": False,
        "round0_reuse": records,
        "comparators": comparator_rows,
        "innovation_screen": {
            "path": str(INNOVATION_STUDY.relative_to(ROOT)),
            "status": "selection_only_development_seeds_no_selected-label_retraining",
        },
    }


def prepare(study: Path = STUDY) -> dict[str, object]:
    study = Path(study)
    if (study / "runtime").exists():
        raise RuntimeError("refusing to prepare over short-sequential runtime")
    study.mkdir(parents=True, exist_ok=True)
    (study / "splits").mkdir(exist_ok=True)
    protocol = protocol_record()
    reuse = audit_reuse()
    for seed in SEEDS:
        source = BASELINE / "splits" / f"row_seed_{seed}.csv"
        destination = study / "splits" / source.name
        if destination.exists() and sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"refusing changed split: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
    atomic_json(study / "protocol.json", protocol)
    atomic_json(study / "reuse_audit.json", reuse)
    atomic_json(study / "global_pre_test_freeze.json", {
        "status": "PENDING_TRAJECTORIES",
        "expected_methods": list(METHODS),
        "expected_seeds": list(SEEDS),
        "expected_budgets": list(ACTIVE_LABEL_BUDGETS),
        "test_truth_access_count": 0,
    })
    atomic_json(study / "decision.json", {"status": "PENDING_RESULTS", "automatic_extension_to_525": False})
    return {"status": "PREPARED", "protocol_hash": stable_hash(protocol), "reuse_audit": reuse}


def validate_prepared(study: Path = STUDY) -> dict[str, object]:
    study = Path(study)
    current = protocol_record()
    if _json(study / "protocol.json") != current:
        raise RuntimeError("short-sequential protocol/code hash drift")
    if _json(study / "reuse_audit.json") != audit_reuse():
        raise RuntimeError("short-sequential reuse audit drift")
    return {"status": "VALID", "protocol_hash": stable_hash(current)}
