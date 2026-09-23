"""Frozen protocol and exact L653 lineage audit for pure CW-LCMD to L1005."""

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


STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_cw_lcmd_to_1005"
SOURCE_STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_cw_lcmd_to_653"
MAXDET_STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_maxdet_b32"
IVR_STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_kernel_ivr_b32"
METHOD = "center_width_lcmd"
SEEDS = (157, 6101)
BATCH_SIZE = 32
SOURCE_ROUND = 10
FINAL_ROUND = 21
NEW_ROUNDS = tuple(range(11, 22))
ALL_BUDGETS = tuple(range(333, 1006, 32))
PRIMARY_BUDGETS = ALL_BUDGETS[10:]
FROM_525_BUDGETS = ALL_BUDGETS[6:]
FROM_429_BUDGETS = ALL_BUDGETS[3:]
FINAL_ACTIVE_LABELS = 1005
SKETCH_DIMENSION = 512
COMPARATORS = ("gradient_maxdet", "hybrid", "kernel_ivr", "lcmd")
ALL_METHODS = (METHOD, *COMPARATORS)
DISPLAY_NAMES = {
    METHOD: "Center/Width-LCMD",
    "gradient_maxdet": "Gradient-MaxDet",
    "hybrid": "Hybrid",
    "kernel_ivr": "Kernel-IVR",
    "lcmd": "Gradient-LCMD",
}


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def code_paths() -> tuple[Path, ...]:
    relative = (
        "scripts/studies/run_qgeognn_v2_row_cw_lcmd_to_1005.py",
        "src/qgeognn_al/active_learning_v2/cw_lcmd_to_1005_study.py",
        "src/qgeognn_al/active_learning_v2/cw_lcmd_to_1005_runner.py",
        "src/qgeognn_al/active_learning_v2/cw_lcmd_to_1005_reporting.py",
        "src/qgeognn_al/active_learning_v2/cw_lcmd_extension_runner.py",
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
        raise RuntimeError(f"CW-to-1005 implementation incomplete: {missing}")
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths()}


def split_path(seed: int) -> Path:
    return STUDY / "splits" / f"row_seed_{int(seed)}.csv"


def source_root(seed: int) -> Path:
    return SOURCE_STUDY / "runtime" / f"seed_{int(seed)}" / METHOD


def source_anchor(seed: int) -> dict[str, Path]:
    directory = source_root(seed) / f"round_{SOURCE_ROUND:02d}"
    freeze = read_json(directory / "round_freeze.json")
    return {
        "state": directory / "state.csv",
        "input_contract": directory / "input_contract.json",
        "round_freeze": directory / "round_freeze.json",
        "trajectory_freeze": source_root(seed) / "trajectory_freeze.json",
        "checkpoint": Path(freeze["checkpoint_path"]),
        "predictions": Path(freeze["prediction_path"]),
        "fit_audit": Path(freeze["checkpoint_path"]).with_name("fit_audit.json"),
        "context": SOURCE_STUDY / "runtime" / f"seed_{int(seed)}" / "context.json",
        "scrubbed_graphs": historical_round0_paths(seed)["scrubbed_graphs"],
        "split": SOURCE_STUDY / "splits" / f"row_seed_{int(seed)}.csv",
        "protocol": SOURCE_STUDY / "protocol.json",
        "global_freeze": SOURCE_STUDY / "global_pre_test_freeze.json",
    }


def comparator_sources() -> dict[str, Path]:
    return {
        "gradient_maxdet": MAXDET_STUDY / "results/learning_curve_metrics.csv",
        "hybrid": BASELINE / "results/learning_curve_metrics.csv",
        "kernel_ivr": IVR_STUDY / "results/learning_curve_metrics.csv",
        "lcmd": BASELINE / "results/learning_curve_metrics.csv",
    }


def _preprocessing(seed: int, method: str) -> dict:
    if method == "gradient_maxdet":
        return read_json(MAXDET_STUDY / f"runtime/seed_{seed}/context.json")["preprocessing"]
    return read_json(BASELINE / f"runtime/seed_{seed}/context.json")["contract"]["preprocessing"]


def audit_reuse() -> dict[str, object]:
    data = load_features()
    global_freeze = read_json(SOURCE_STUDY / "global_pre_test_freeze.json")
    if global_freeze.get("status") != "COMPLETE_DEVELOPMENTAL_CONTINUATION":
        raise RuntimeError("source CW-to-653 study is not complete")
    anchors, comparators = [], []
    for seed in SEEDS:
        paths = source_anchor(seed)
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise RuntimeError(f"missing L653 source artifacts: {missing}")
        partition = pd.read_csv(paths["split"])
        validate_row_protocol(partition)
        if partition.sample_id.astype(str).tolist() != data.sample_id.astype(str).tolist():
            raise RuntimeError(f"source split/sample order drift: seed={seed}")
        state = pd.read_csv(paths["state"])
        labeled = state.loc[state.role.eq("labeled"), "sample_id"].astype(str).tolist()
        unlabeled = state.loc[state.role.eq("unlabeled"), "sample_id"].astype(str).tolist()
        l0 = partition.loc[partition.role.eq("l0"), "sample_id"].astype(str).tolist()
        u0 = partition.loc[partition.role.eq("u0"), "sample_id"].astype(str).tolist()
        lineage = labeled[len(l0):]
        if len(lineage) != SOURCE_ROUND * BATCH_SIZE or len(set(lineage)) != len(lineage):
            raise RuntimeError(f"invalid CW L653 selected lineage: seed={seed}")
        if labeled != l0 + lineage or len(labeled) != 653 or len(unlabeled) != 2677:
            raise RuntimeError(f"ordered CW L653 lineage mismatch: seed={seed}")
        if unlabeled != [value for value in u0 if value not in set(lineage)]:
            raise RuntimeError(f"ordered CW U653 lineage mismatch: seed={seed}")
        contract = read_json(paths["input_contract"])
        freeze = read_json(paths["round_freeze"])
        trajectory = read_json(paths["trajectory_freeze"])
        global_entry = global_freeze["new_entries"][f"seed_{seed}/round_10"]
        context = read_json(paths["context"])
        if contract["L_t_ids_hash"] != ids_hash(labeled) or contract["U_t_ids_hash"] != ids_hash(unlabeled):
            raise RuntimeError(f"source L653 input hashes mismatch: seed={seed}")
        if sha256_file(paths["round_freeze"]) != global_entry["round_freeze_sha256"]:
            raise RuntimeError(f"source L653 global binding mismatch: seed={seed}")
        if sha256_file(paths["checkpoint"]) != freeze["checkpoint_sha256"] or freeze["checkpoint_sha256"] != global_entry["checkpoint_sha256"]:
            raise RuntimeError(f"source L653 checkpoint mismatch: seed={seed}")
        if sha256_file(paths["predictions"]) != freeze["prediction_sha256"] or freeze["prediction_sha256"] != global_entry["prediction_sha256"]:
            raise RuntimeError(f"source L653 prediction mismatch: seed={seed}")
        fit = read_json(paths["fit_audit"])
        if fit["checkpoint_state_hash"] != freeze["checkpoint_state_hash"] or trajectory["final_active_labels"] != 653:
            raise RuntimeError(f"source L653 state/trajectory mismatch: seed={seed}")
        anchors.append({
            "outer_seed": seed, "source_round": SOURCE_ROUND, "source_active_labels": 653,
            "source_unlabeled_rows": 2677, "ordered_L653_ids_hash": stable_hash(labeled),
            "ordered_U653_ids_hash": stable_hash(unlabeled), "L653_set_hash": ids_hash(labeled),
            "U653_set_hash": ids_hash(unlabeled), "selected_lineage_ordered_hash": stable_hash(lineage),
            "split_sha256": sha256_file(paths["split"]), "split_hash": stable_hash(partition.to_dict("list")),
            "source_round_freeze_sha256": sha256_file(paths["round_freeze"]),
            "checkpoint_sha256": sha256_file(paths["checkpoint"]),
            "checkpoint_state_hash": freeze["checkpoint_state_hash"],
            "prediction_sha256": sha256_file(paths["predictions"]),
            "preprocessing_hash": stable_hash(context["preprocessing"]),
            "target_scale_hash": stable_hash(context["preprocessing"]["target_scales"]),
            "center_width_transform_hash": stable_hash(context["center_width_transform"]),
            "source_protocol_sha256": sha256_file(paths["protocol"]), "reuse_legal": True,
        })
        for method, path in comparator_sources().items():
            table = pd.read_csv(path)
            arm = table.loc[
                table.outer_seed.eq(seed) & table.method.eq(method)
                & table.active_label_count.isin(ALL_BUDGETS)
            ]
            if len(arm) != len(ALL_BUDGETS) or arm.duplicated("active_label_count").any():
                raise RuntimeError(f"historical comparator incomplete: seed={seed} method={method}")
            preprocessing = _preprocessing(seed, method)
            if preprocessing["target_scales"] != context["preprocessing"]["target_scales"]:
                raise RuntimeError(f"target-scale mismatch: seed={seed} method={method}")
            comparators.append({
                "outer_seed": seed, "method": method, "source": str(path.relative_to(ROOT)),
                "source_sha256": sha256_file(path), "budgets": list(ALL_BUDGETS), "rows": len(arm),
                "target_scale_hash": stable_hash(preprocessing["target_scales"]),
                "split_hash": stable_hash(partition.to_dict("list")),
                "training_performed_by_continuation": False,
            })
    return {
        "status": "AUDITED_EXACT_CW_L653_BEFORE_CONTINUATION",
        "source_anchors": anchors, "historical_comparators": comparators,
        "other_method_training_authorized": False, "test_truth_access_count": 0,
    }


def protocol_record() -> dict[str, object]:
    return {
        "study": STUDY.name, "evidence_class": "DEVELOPMENTAL_CONTINUATION",
        "source_study": str(SOURCE_STUDY.relative_to(ROOT)), "source_method": METHOD,
        "source_round": SOURCE_ROUND, "source_active_labels": 653, "method": METHOD,
        "seeds": list(SEEDS), "batch_size": BATCH_SIZE, "new_rounds": list(NEW_ROUNDS),
        "active_label_budgets": list(ALL_BUDGETS), "continuation_budgets": list(PRIMARY_BUDGETS),
        "hard_stop_active_labels": FINAL_ACTIVE_LABELS, "acquisition_at_1005": False,
        "gradient_dimension": SKETCH_DIMENSION, "gradient_sketch_seed": "outer_seed + 4_000_037",
        "center_width": "grad((V1+V2)/(2*s_C)), grad((V2-V1)/s_W); frozen L0 population scales",
        "selector": "LCMD-TP with all current L_t rows as centers and current U_t candidates",
        "retraining": "scratch QGeoGNN-V2 every round; no warm start", "training": TRAINING_CONFIG,
        "comparators": list(COMPARATORS), "comparator_training": "forbidden; exact historical reuse only",
        "primary_metric": "normalized trapezoidal combined-NRMSE AULC over 653--1005",
        "secondary_metrics": ["AULC_525_1005", "AULC_429_1005", "AULC_333_1005"],
        "decision_rule": {
            "SUSTAINED_TO_1005": "both seeds beat MaxDet, Hybrid, and IVR on AULC_653_1005",
            "SUSTAINED_AGAINST_SOME": "both seeds beat at least one of MaxDet, Hybrid, or IVR",
            "STOP_CW": "both seeds lose to all three by more than 0.03 and have no endpoint advantage",
            "MIXED_LONG_HORIZON": "opposite seed directions exceed 0.01 for at least one strong comparator",
            "NO_CONSISTENT_ADVANTAGE": "none of the preceding rules applies",
        },
        "test_reveal_barrier": "all 2 seeds x 11 new checkpoint/prediction pairs frozen",
        "source_sha256": sha256_file(SOURCE_DATA), "graph_cache_sha256": sha256_file(SOURCE_GRAPH_CACHE),
        "feature_columns": list(FEATURE_COLUMNS), "code_hashes": code_hashes(),
    }


def prepare(study: Path = STUDY) -> dict[str, object]:
    study = Path(study)
    if (study / "runtime").exists():
        raise RuntimeError("refusing to prepare over CW-to-1005 runtime")
    study.mkdir(parents=True, exist_ok=True); (study / "splits").mkdir(exist_ok=True)
    reuse = audit_reuse(); protocol = protocol_record()
    for seed in SEEDS:
        source = source_anchor(seed)["split"]; destination = study / "splits" / source.name
        if destination.exists() and sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"refusing changed extension split: {destination}")
        if not destination.exists(): shutil.copy2(source, destination)
    atomic_json(study / "protocol.json", protocol); atomic_json(study / "reuse_audit.json", reuse)
    atomic_json(study / "global_pre_test_freeze.json", {
        "status": "PENDING_CONTINUATIONS", "expected_method": METHOD,
        "expected_seeds": list(SEEDS), "expected_new_budgets": list(ALL_BUDGETS[11:]),
        "test_truth_access_count": 0,
    })
    atomic_json(study / "decision.json", {"status": "PENDING_RESULTS"})
    return {"status": "PREPARED", "protocol_hash": stable_hash(protocol), "source_anchors": 2}


def validate_prepared(study: Path = STUDY) -> dict[str, object]:
    study = Path(study); protocol = protocol_record()
    if read_json(study / "protocol.json") != protocol:
        raise RuntimeError("CW-to-1005 protocol/code hash drift")
    if read_json(study / "reuse_audit.json") != audit_reuse():
        raise RuntimeError("CW-to-1005 source/comparator reuse audit drift")
    return {"status": "VALID", "protocol_hash": stable_hash(protocol)}
