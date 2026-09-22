"""Protocol and lineage audit for the CW-LCMD continuation from 429 to 525."""

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


STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_cw_lcmd_to_525"
SOURCE_STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_short_sequential_b32"
MAXDET_STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_maxdet_b32"
IVR_STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_kernel_ivr_b32"
LCMD_TO_IVR_STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_lcmd_to_ivr_b32"
METHOD = "center_width_lcmd"
SEEDS = (157, 6101)
BATCH_SIZE = 32
SOURCE_ROUND = 3
FINAL_ROUND = 6
NEW_ACQUISITION_ROUNDS = (4, 5, 6)
ACTIVE_LABEL_BUDGETS = (333, 365, 397, 429, 461, 493, 525)
CONTINUATION_BUDGETS = (429, 461, 493, 525)
FINAL_ACTIVE_LABELS = 525
SKETCH_DIMENSION = 512
COMPARATORS = ("gradient_maxdet", "hybrid", "lcmd", "kernel_ivr")
DISPLAY_NAMES = {
    METHOD: "Center/Width-LCMD",
    "gradient_maxdet": "Gradient-MaxDet",
    "hybrid": "Hybrid",
    "lcmd": "Gradient-LCMD",
    "kernel_ivr": "Kernel-IVR",
}


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def code_paths() -> tuple[Path, ...]:
    relative = (
        "scripts/studies/run_qgeognn_v2_row_cw_lcmd_to_525.py",
        "src/qgeognn_al/active_learning_v2/cw_lcmd_extension_study.py",
        "src/qgeognn_al/active_learning_v2/cw_lcmd_extension_runner.py",
        "src/qgeognn_al/active_learning_v2/cw_lcmd_extension_reporting.py",
        "src/qgeognn_al/active_learning_v2/gradient_features.py",
        "src/qgeognn_al/active_learning_v2/gradient_transforms.py",
        "src/qgeognn_al/active_learning_v2/lcmd.py",
        "src/qgeognn_al/active_learning_v2/protocol.py",
        "src/qgeognn_al/active_learning_v2/runner.py",
        "src/qgeognn_al/active_learning_v2/sequential_acquisition.py",
    )
    return tuple(ROOT / value for value in relative)


def code_hashes() -> dict[str, str]:
    missing = [str(path) for path in code_paths() if not path.exists()]
    if missing:
        raise RuntimeError(f"CW extension implementation incomplete: {missing}")
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths()}


def protocol_record() -> dict[str, object]:
    return {
        "study": STUDY.name,
        "evidence_class": "DEVELOPMENTAL_CONTINUATION",
        "source_study": str(SOURCE_STUDY.relative_to(ROOT)),
        "source_method": METHOD,
        "source_round": SOURCE_ROUND,
        "source_active_labels": 429,
        "method": METHOD,
        "seeds": list(SEEDS),
        "batch_size": BATCH_SIZE,
        "new_acquisition_rounds": list(NEW_ACQUISITION_ROUNDS),
        "active_label_budgets": list(ACTIVE_LABEL_BUDGETS),
        "continuation_budgets": list(CONTINUATION_BUDGETS),
        "hard_stop_active_labels": FINAL_ACTIVE_LABELS,
        "forbidden_next_budget": 557,
        "gradient_dimension": SKETCH_DIMENSION,
        "gradient_sketch_seed": "outer_seed + 4_000_037",
        "center_width": "grad((V1+V2)/(2*s_C)), grad((V2-V1)/s_W); fixed L0 population scales",
        "selector": "LCMD-TP with all current L_t rows as centers",
        "retraining": "scratch QGeoGNN-V2 on the CW-specific trajectory; no warm start",
        "training": TRAINING_CONFIG,
        "comparators": list(COMPARATORS),
        "comparator_training": "forbidden; exact historical result reuse only",
        "primary_metric": "normalized trapezoidal combined-NRMSE AULC over 429--525",
        "secondary_metric": "normalized trapezoidal combined-NRMSE AULC over 333--525",
        "test_reveal_barrier": "all 2 seeds x 3 new checkpoints/predictions frozen",
        "source_sha256": sha256_file(SOURCE_DATA),
        "graph_cache_sha256": sha256_file(SOURCE_GRAPH_CACHE),
        "feature_columns": list(FEATURE_COLUMNS),
        "code_hashes": code_hashes(),
    }


def split_path(seed: int) -> Path:
    return STUDY / "splits" / f"row_seed_{int(seed)}.csv"


def source_method_root(seed: int) -> Path:
    return SOURCE_STUDY / "runtime" / f"seed_{int(seed)}" / METHOD


def source_anchor(seed: int) -> dict[str, Path]:
    base = source_method_root(seed)
    round_dir = base / f"round_{SOURCE_ROUND:02d}"
    freeze = _json(round_dir / "round_freeze.json")
    return {
        "state": round_dir / "state.csv",
        "input_contract": round_dir / "input_contract.json",
        "round_freeze": round_dir / "round_freeze.json",
        "trajectory_freeze": base / "trajectory_freeze.json",
        "checkpoint": Path(freeze["checkpoint_path"]),
        "predictions": Path(freeze["prediction_path"]),
        "fit_audit": Path(freeze["checkpoint_path"]).with_name("fit_audit.json"),
        "context": SOURCE_STUDY / "runtime" / f"seed_{int(seed)}" / "context.json",
        "scrubbed_graphs": historical_round0_paths(seed)["scrubbed_graphs"],
        "split": SOURCE_STUDY / "splits" / f"row_seed_{int(seed)}.csv",
    }


def _source_selected_batches(seed: int) -> list[str]:
    values: list[str] = []
    for source_round in range(SOURCE_ROUND):
        path = source_method_root(seed) / f"round_{source_round:02d}/acquisition_artifacts/selected_next_batch.csv"
        batch = pd.read_csv(path).sample_id.astype(str).tolist()
        if len(batch) != BATCH_SIZE or len(set(batch)) != BATCH_SIZE:
            raise RuntimeError(f"invalid historical CW batch: seed={seed} round={source_round}")
        values.extend(batch)
    if len(values) != SOURCE_ROUND * BATCH_SIZE or len(set(values)) != len(values):
        raise RuntimeError(f"historical CW lineage contains duplicates: seed={seed}")
    return values


def _comparator_sources() -> dict[str, Path]:
    return {
        "gradient_maxdet": MAXDET_STUDY / "results/learning_curve_metrics.csv",
        "hybrid": BASELINE / "results/learning_curve_metrics.csv",
        "lcmd": BASELINE / "results/learning_curve_metrics.csv",
        "kernel_ivr": IVR_STUDY / "results/learning_curve_metrics.csv",
    }


def audit_reuse() -> dict[str, object]:
    data = load_features()
    source_global = _json(SOURCE_STUDY / "global_pre_test_freeze.json")
    if source_global.get("status") != "FROZEN_BEFORE_TEST_TRUTH":
        raise RuntimeError("source short-sequential study is not globally frozen")
    source_protocol = _json(SOURCE_STUDY / "protocol.json")
    records = []
    for seed in SEEDS:
        paths = source_anchor(seed)
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise RuntimeError(f"missing source CW anchor artifacts for seed {seed}: {missing}")
        partition = pd.read_csv(paths["split"])
        validate_row_protocol(partition)
        if partition.sample_id.astype(str).tolist() != data.sample_id.astype(str).tolist():
            raise RuntimeError(f"source split/sample order drift for seed {seed}")
        state = pd.read_csv(paths["state"])
        labeled = state.loc[state.role.eq("labeled"), "sample_id"].astype(str).tolist()
        unlabeled = state.loc[state.role.eq("unlabeled"), "sample_id"].astype(str).tolist()
        if len(labeled) != 429 or len(unlabeled) != 2901 or len(set(labeled + unlabeled)) != 3330:
            raise RuntimeError(f"source CW L429/U429 state invalid for seed {seed}")
        l0 = partition.loc[partition.role.eq("l0"), "sample_id"].astype(str).tolist()
        u0 = partition.loc[partition.role.eq("u0"), "sample_id"].astype(str).tolist()
        selected = _source_selected_batches(seed)
        if labeled != l0 + selected:
            raise RuntimeError(f"source CW ordered L429 lineage mismatch for seed {seed}")
        selected_set = set(selected)
        if unlabeled != [value for value in u0 if value not in selected_set]:
            raise RuntimeError(f"source CW ordered U429 lineage mismatch for seed {seed}")
        input_contract = _json(paths["input_contract"])
        round_freeze = _json(paths["round_freeze"])
        trajectory = _json(paths["trajectory_freeze"])
        global_entry = source_global["entries"][f"seed_{seed}/{METHOD}/round_03"]
        if input_contract["L_t_ids_hash"] != ids_hash(labeled) or input_contract["U_t_ids_hash"] != ids_hash(unlabeled):
            raise RuntimeError(f"source CW state/input hashes mismatch for seed {seed}")
        if sha256_file(paths["round_freeze"]) != global_entry["round_freeze_sha256"]:
            raise RuntimeError(f"source CW global-freeze binding mismatch for seed {seed}")
        if sha256_file(paths["checkpoint"]) != round_freeze["checkpoint_sha256"] != global_entry["checkpoint_sha256"]:
            raise RuntimeError(f"source CW checkpoint hash mismatch for seed {seed}")
        if sha256_file(paths["predictions"]) != round_freeze["prediction_sha256"] != global_entry["prediction_sha256"]:
            raise RuntimeError(f"source CW prediction hash mismatch for seed {seed}")
        if trajectory.get("final_active_labels") != 429 or trajectory.get("round_points") != 4:
            raise RuntimeError(f"source CW trajectory freeze invalid for seed {seed}")
        context = _json(paths["context"])
        records.append({
            "outer_seed": seed,
            "source_round": SOURCE_ROUND,
            "source_active_labels": len(labeled),
            "source_unlabeled_rows": len(unlabeled),
            "ordered_L429_ids_hash": stable_hash(labeled),
            "L429_ids_hash": ids_hash(labeled),
            "ordered_U429_ids_hash": stable_hash(unlabeled),
            "U429_ids_hash": ids_hash(unlabeled),
            "selected_lineage_ordered_hash": stable_hash(selected),
            "split_sha256": sha256_file(paths["split"]),
            "split_hash": stable_hash(partition.to_dict("list")),
            "source_round_freeze_sha256": sha256_file(paths["round_freeze"]),
            "checkpoint_sha256": sha256_file(paths["checkpoint"]),
            "checkpoint_state_hash": round_freeze["checkpoint_state_hash"],
            "prediction_sha256": sha256_file(paths["predictions"]),
            "preprocessing_hash": stable_hash(context["preprocessing"]),
            "target_scale_hash": stable_hash(context["preprocessing"]["target_scales"]),
            "center_width_transform_hash": stable_hash(context["center_width_transform"]),
            "source_protocol_sha256": sha256_file(SOURCE_STUDY / "protocol.json"),
            "reuse_legal": True,
        })

    comparators = []
    for method, path in _comparator_sources().items():
        table = pd.read_csv(path)
        arm = table.loc[
            table.outer_seed.isin(SEEDS)
            & table.method.eq(method)
            & table.active_label_count.isin(ACTIVE_LABEL_BUDGETS)
        ]
        expected = len(SEEDS) * len(ACTIVE_LABEL_BUDGETS)
        if len(arm) != expected or arm.duplicated(["outer_seed", "active_label_count"]).any():
            raise RuntimeError(f"matched comparator incomplete: {method}")
        comparators.append({
            "method": method,
            "source": str(path.relative_to(ROOT)),
            "source_sha256": sha256_file(path),
            "rows": len(arm),
            "seeds": list(SEEDS),
            "budgets": list(ACTIVE_LABEL_BUDGETS),
            "training_performed_by_extension": False,
        })
    lcmd_to_ivr_readme = LCMD_TO_IVR_STUDY / "README.md"
    status = "absent"
    if lcmd_to_ivr_readme.exists():
        text = lcmd_to_ivr_readme.read_text()
        status = "sealed_engineering_checks_passed_formal_execution_not_started" if "formal execution not started" in text else "present_status_unrecognized"
    return {
        "status": "AUDITED_BEFORE_CONTINUATION",
        "source_protocol": source_protocol,
        "source_anchors": records,
        "historical_comparators": comparators,
        "other_method_training_authorized": False,
        "lcmd_to_ivr_read_only_status": status,
    }


def prepare(study: Path = STUDY) -> dict[str, object]:
    study = Path(study)
    if (study / "runtime").exists():
        raise RuntimeError("refusing to prepare over CW extension runtime")
    study.mkdir(parents=True, exist_ok=True)
    (study / "splits").mkdir(exist_ok=True)
    protocol = protocol_record()
    reuse = audit_reuse()
    for seed in SEEDS:
        source = SOURCE_STUDY / "splits" / f"row_seed_{seed}.csv"
        destination = study / "splits" / source.name
        if destination.exists() and sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"refusing changed extension split: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
    atomic_json(study / "protocol.json", protocol)
    atomic_json(study / "reuse_audit.json", reuse)
    atomic_json(study / "global_pre_test_freeze.json", {
        "status": "PENDING_CONTINUATIONS",
        "expected_method": METHOD,
        "expected_seeds": list(SEEDS),
        "expected_new_budgets": list(ACTIVE_LABEL_BUDGETS[4:]),
        "test_truth_access_count": 0,
    })
    atomic_json(study / "decision.json", {"status": "PENDING_RESULTS", "automatic_continuation": False})
    return {"status": "PREPARED", "protocol_hash": stable_hash(protocol), "source_anchors": len(reuse["source_anchors"])}


def validate_prepared(study: Path = STUDY) -> dict[str, object]:
    study = Path(study)
    protocol = protocol_record()
    recorded_protocol = _json(study / "protocol.json")
    if recorded_protocol != protocol:
        # Once the developmental continuation is complete, preserve the exact
        # frozen protocol used for the runs while reporting later source-only
        # fixes as an explicit, non-rerun code drift.
        decision_path = study / "decision.json"
        completed = decision_path.exists() and _json(decision_path).get("status") == "COMPLETE_DEVELOPMENTAL_CONTINUATION"
        if not completed:
            raise RuntimeError("CW extension protocol/code hash drift")
        return {
            "status": "VALID_COMPLETED_FROZEN_PROTOCOL",
            "protocol_hash": stable_hash(recorded_protocol),
            "code_hash_drift": True,
        }
    if _json(study / "reuse_audit.json") != audit_reuse():
        raise RuntimeError("CW extension reuse audit drift")
    return {"status": "VALID", "protocol_hash": stable_hash(protocol)}
