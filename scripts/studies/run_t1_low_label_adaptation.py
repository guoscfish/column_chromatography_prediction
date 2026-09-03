#!/usr/bin/env python3
"""T1 low-label transfer adaptation preregistration and engineering smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.qgeognn_al.artifacts import sha256_file
from src.qgeognn_al.data import condition_matrix, load_combined_graph_cache
from src.qgeognn_al.engine import AdaptationTrainConfig, QGeoGNNActiveLearningEngine
from src.qgeognn_al.resources import (
    ROOT, SOURCE_CHECKPOINTS, SOURCE_DATA, SOURCE_GRAPH_CACHE, SOURCE_SCALER,
    TARGET_DATA, TARGET_GRAPH_CACHE, verified_source_checkpoints,
)
from src.qgeognn_al.t1_formal import (
    EXPECTED_TOTAL_PARAMETERS, EXPECTED_TRAINABLE, FROZEN_BUDGETS,
    FROZEN_METHODS, FROZEN_NEURAL_MODES, REFERENCE_METHOD,
    build_formal_fit_plan, capacity_crossover_summary, completion_gate,
    compute_aulc, execute_fit_plan, expected_fit_contract,
    paired_aulc_effects, regression_metrics, summarize_methods_by_budget,
    validate_frozen_formal_config, write_fit_contract,
)

STUDY = ROOT / "studies/track_b_transfer/t1_low_label_adaptation"
SOURCE_SPLIT = ROOT / "experiments/e0_4g_baseline/split_seed_42.csv"
HISTORICAL_SCALES = ROOT / "experiments/e4_protocol_a_formal/source_target_scales.json"
LABEL_COLUMNS = {"V1_ml", "V2_ml"}
IDENTITY_INPUTS = ["sample_id", "canonical_index", "canonical_smiles"]
PRIMARY_METHODS = [
    "zero_shot", "affine", "condition_ridge_residual", "target_head_only",
    "last1_head", "current_last2_head",
]
MODE_BY_METHOD = {
    "target_head_only": "head_only",
    "last1_head": "last1_head",
    "current_last2_head": "last2_head",
}
EXPECTED_COUNTS = {"head_only": 774, "last1_head": 93454, "last2_head": 186134}


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def generate_outer_seeds(master_seed: int, count: int = 5) -> list[int]:
    sequence = np.random.SeedSequence(master_seed)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in sequence.spawn(count)]


def make_manifests(
    identities: pd.DataFrame,
    outer_seeds: list[int],
    budgets: list[int],
    fixed_validation: int,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create target partitions and nested schedules from identity columns only."""
    if set(identities.columns) != set(IDENTITY_INPUTS):
        raise ValueError("partition preparation accepts identity columns only")
    ordered = identities[IDENTITY_INPUTS].sort_values("canonical_index").reset_index(drop=True)
    if ordered.sample_id.isna().any() or ordered.sample_id.duplicated().any():
        raise ValueError("sample_id must be complete and unique")
    n_test = int(math.ceil(len(ordered) * test_fraction))
    partition_rows, schedule_rows = [], []
    for seed in outer_seeds:
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(ordered))
        test_positions = set(permutation[:n_test].tolist())
        remaining = permutation[n_test:]
        validation_positions = set(remaining[:fixed_validation].tolist())
        ranking = remaining[fixed_validation:].tolist()
        rank_by_position = {position: rank for rank, position in enumerate(ranking)}
        for position, row in ordered.iterrows():
            base_role = "test" if position in test_positions else "validation" if position in validation_positions else "train_candidate"
            partition_rows.append({
                "outer_seed": seed, "sample_id": str(row.sample_id),
                "canonical_index": int(row.canonical_index),
                "canonical_smiles": str(row.canonical_smiles), "base_role": base_role,
                "schedule_rank": rank_by_position.get(position, -1),
            })
        for budget in budgets:
            gradient_count = budget - fixed_validation
            if gradient_count < 1 or gradient_count > len(ranking):
                raise ValueError(f"invalid budget: {budget}")
            gradient_positions = set(ranking[:gradient_count])
            for position, row in ordered.iterrows():
                if position in test_positions:
                    role = "test"
                elif position in validation_positions:
                    role = "validation"
                elif position in gradient_positions:
                    role = "gradient_train"
                else:
                    role = "unlabeled"
                schedule_rows.append({
                    "outer_seed": seed, "budget": budget,
                    "sample_id": str(row.sample_id), "canonical_index": int(row.canonical_index),
                    "canonical_smiles": str(row.canonical_smiles), "role": role,
                })
    return pd.DataFrame(partition_rows), pd.DataFrame(schedule_rows)


def audit_manifests(partition: pd.DataFrame, schedule: pd.DataFrame, budgets: list[int], fixed_validation: int) -> dict:
    role_counts = {
        f"{seed}:{budget}": group.role.value_counts().sort_index().to_dict()
        for (seed, budget), group in schedule.groupby(["outer_seed", "budget"])
    }
    overlap_pass = True
    nested_pass = True
    for seed, by_seed in schedule.groupby("outer_seed"):
        previous: set[str] = set()
        test_reference: set[str] | None = None
        validation_reference: set[str] | None = None
        for budget in sorted(budgets):
            rows = by_seed.loc[by_seed.budget.eq(budget)]
            roles = {role: set(rows.loc[rows.role.eq(role), "sample_id"]) for role in rows.role.unique()}
            current = roles.get("gradient_train", set())
            test = roles.get("test", set()); validation = roles.get("validation", set())
            overlap_pass &= not bool((current & validation) | (current & test) | (validation & test))
            nested_pass &= previous.issubset(current)
            nested_pass &= len(current) == budget - fixed_validation and len(validation) == fixed_validation
            if test_reference is not None:
                nested_pass &= test == test_reference and validation == validation_reference
            previous, test_reference, validation_reference = current, test, validation
    expected_partition_rows = partition.outer_seed.nunique() * partition.sample_id.nunique()
    return {
        "partition_rows": len(partition), "expected_partition_rows": expected_partition_rows,
        "schedule_rows": len(schedule), "outer_seeds": sorted(map(int, partition.outer_seed.unique())),
        "role_counts": role_counts, "role_overlap_pass": bool(overlap_pass),
        "nested_budget_pass": bool(nested_pass),
        "partition_identity_inputs_only": True, "target_label_columns_read_during_prepare": [],
    }


def method_label_audit(
    methods: list[str], train_ids: list[str], validation_ids: list[str], evaluation_ids: list[str]
) -> pd.DataFrame:
    """Record the shared-label contract without exposing labels or predictions."""
    rows = []
    for method in methods:
        rows.append({
            "method": method,
            "gradient_train_ids_hash": None if method == "zero_shot" else canonical_hash(sorted(train_ids)),
            "gradient_train_rows": 0 if method == "zero_shot" else len(train_ids),
            "validation_ids_hash": canonical_hash(sorted(validation_ids)),
            "validation_rows": len(validation_ids),
            "evaluation_ids_hash": canonical_hash(sorted(evaluation_ids)),
            "evaluation_rows": len(evaluation_ids),
            "target_labels_used_for_fit": method != "zero_shot",
        })
    return pd.DataFrame(rows)


def source_scales() -> dict:
    source = pd.read_csv(SOURCE_DATA, usecols=["sample_id", "V1_ml", "V2_ml"])
    split = pd.read_csv(SOURCE_SPLIT, usecols=["sample_id", "split"])
    train = source.merge(split.loc[split.split.eq("train"), ["sample_id"]], on="sample_id", validate="one_to_one")
    result = {
        "source_canonical_path": str(SOURCE_DATA.relative_to(ROOT)),
        "source_canonical_sha256": sha256_file(SOURCE_DATA),
        "source_split_path": str(SOURCE_SPLIT.relative_to(ROOT)),
        "source_split_sha256": sha256_file(SOURCE_SPLIT), "split_role": "train",
        "train_rows": len(train), "ddof": 0,
        "V1": float(train.V1_ml.std(ddof=0)), "V2": float(train.V2_ml.std(ddof=0)),
        "target_data_used": False,
    }
    historical = json.loads(HISTORICAL_SCALES.read_text())
    result["matches_historical_e4"] = bool(
        result["source_canonical_sha256"] == historical["source_canonical_sha256"]
        and result["source_split_sha256"] == historical["source_split_sha256"]
        and np.isclose(result["V1"], historical["V1"])
        and np.isclose(result["V2"], historical["V2"])
        and result["ddof"] == historical["ddof"]
    )
    if not result["matches_historical_e4"]:
        raise RuntimeError("recomputed source-only scales do not match historical E4")
    return result


def environment_record() -> dict:
    packages = {}
    for name in ["numpy", "pandas", "scikit-learn", "torch", "torch-geometric"]:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages, "device": "cpu"}


def prepare(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    generated = generate_outer_seeds(int(config["master_seed"]), int(config["outer_seed_count"]))
    frozen = config.get("outer_seeds")
    if frozen not in (None, []) and list(map(int, frozen)) != generated:
        raise RuntimeError("frozen outer seeds disagree with deterministic SeedSequence output")
    config["outer_seeds"] = generated
    header = pd.read_csv(TARGET_DATA, nrows=0).columns.tolist()
    usecols = [column for column in IDENTITY_INPUTS if column in header]
    identities = pd.read_csv(TARGET_DATA, usecols=usecols)
    if "canonical_index" not in identities:
        identities["canonical_index"] = np.arange(len(identities), dtype=int)
    identities = identities[IDENTITY_INPUTS]
    if len(identities) != 574:
        raise RuntimeError(f"authoritative T1 target must have 574 rows, found {len(identities)}")
    partition, schedule = make_manifests(
        identities, generated, list(map(int, config["target_label_budgets"])),
        int(config["fixed_validation"]), float(config["test_fraction"]),
    )
    STUDY.mkdir(parents=True, exist_ok=True)
    partition.to_csv(STUDY / "partition_manifest.csv", index=False)
    schedule.to_csv(STUDY / "schedule_manifest.csv", index=False)
    pd.DataFrame(verified_source_checkpoints()).to_csv(STUDY / "source_checkpoint_manifest.csv", index=False)
    scales = source_scales()
    (STUDY / "source_target_scales.json").write_text(json.dumps(scales, indent=2) + "\n")
    audit = audit_manifests(partition, schedule, config["target_label_budgets"], config["fixed_validation"])
    audit.update({
        "partition_sha256": sha256_file(STUDY / "partition_manifest.csv"),
        "schedule_sha256": sha256_file(STUDY / "schedule_manifest.csv"),
        "target_path": str(TARGET_DATA.relative_to(ROOT)), "target_sha256": sha256_file(TARGET_DATA),
    })
    (STUDY / "partition_schedule_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    config.update({
        "partition_sha256": audit["partition_sha256"], "schedule_sha256": audit["schedule_sha256"],
        "target_sha256": audit["target_sha256"],
        "source_checkpoint_hashes": {str(row["source_seed"]): row["sha256"] for row in verified_source_checkpoints()},
        "source_target_scales_sha256": sha256_file(STUDY / "source_target_scales.json"),
    })
    plan = build_formal_fit_plan(config)
    plan_audit = {
        "formal_authorized": bool(config["formal_authorized"]),
        "outer_seed_count": len(config["outer_seeds"]),
        "budget_count": len(config["target_label_budgets"]),
        "neural_method_count": len(config["neural_modes"]),
        "source_member_count": len(config["source_members"]),
        "expected_neural_fits": len(plan),
        "unique_run_keys": len({item.run_key for item in plan}),
        "duplicate_run_keys": len(plan) - len({item.run_key for item in plan}),
        "first_run_key": plan[0].run_key,
        "last_run_key": plan[-1].run_key,
    }
    (STUDY / "formal_plan_audit.json").write_text(json.dumps(plan_audit, indent=2) + "\n")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    (STUDY / "environment.json").write_text(json.dumps(environment_record(), indent=2) + "\n")
    return audit


def load_selected_truth(path: Path, allowed_ids: set[str]) -> pd.DataFrame:
    """Read target label cells only for explicitly labeled train/validation rows."""
    ids = pd.read_csv(path, usecols=["sample_id"])
    skiprows = [index + 1 for index, sample_id in enumerate(ids.sample_id.astype(str)) if sample_id not in allowed_ids]
    truth = pd.read_csv(path, usecols=["sample_id", "V1_ml", "V2_ml"], skiprows=skiprows)
    if set(truth.sample_id.astype(str)) != allowed_ids:
        raise RuntimeError("selected truth identity mismatch")
    return truth


def fit_affine(train_truth: np.ndarray, train_source: np.ndarray, predict_source: np.ndarray) -> np.ndarray:
    return np.column_stack([
        LinearRegression().fit(train_source[:, [target]], train_truth[:, target]).predict(predict_source[:, [target]])
        for target in range(2)
    ])


def fit_ridge_residual(
    train_truth: np.ndarray, train_source: np.ndarray, train_conditions: np.ndarray,
    groups: np.ndarray, predict_source: np.ndarray, predict_conditions: np.ndarray,
    alphas: list[float], scales: np.ndarray | None = None,
) -> tuple[np.ndarray, float, str]:
    metric_scales = np.ones(2, dtype=float) if scales is None else np.asarray(scales, dtype=float)
    if metric_scales.shape != (2,) or np.any(metric_scales <= 0):
        raise ValueError("ridge selection scales must contain two positive values")
    unique_groups = len(np.unique(groups))
    if unique_groups < 2 or len(train_truth) < 4:
        selected, policy = 1.0, "deterministic_alpha_1_insufficient_groupkfold"
    else:
        folds = min(5, unique_groups)
        candidates = []
        for alpha in alphas:
            scores = []
            for inner_train, inner_valid in GroupKFold(n_splits=folds).split(train_truth, groups=groups):
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                x = np.column_stack([train_source, train_conditions])
                model.fit(x[inner_train], train_truth[inner_train] - train_source[inner_train])
                pred = train_source[inner_valid] + model.predict(x[inner_valid])
                rmse = np.sqrt(np.mean(np.square(train_truth[inner_valid] - pred), axis=0))
                scores.append(float(np.mean(rmse / metric_scales)))
            candidates.append((float(np.mean(scores)), float(alpha)))
        selected, policy = min(candidates)[1], f"groupkfold_{folds}_gradient_train_only"
    model = make_pipeline(StandardScaler(), Ridge(alpha=selected))
    model.fit(np.column_stack([train_source, train_conditions]), train_truth - train_source)
    pred = predict_source + model.predict(np.column_stack([predict_source, predict_conditions]))
    return pred, selected, policy


def smoke(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    partition_path, schedule_path = STUDY / "partition_manifest.csv", STUDY / "schedule_manifest.csv"
    if sha256_file(partition_path) != config["partition_sha256"] or sha256_file(schedule_path) != config["schedule_sha256"]:
        raise RuntimeError("prepared T1 manifests changed")
    records = verified_source_checkpoints()
    expected_hashes = config["source_checkpoint_hashes"]
    if any(expected_hashes[str(row["source_seed"])] != row["sha256"] for row in records):
        raise RuntimeError("protected source checkpoint hash drift")
    schedule = pd.read_csv(schedule_path)
    seed, budget = int(config["outer_seeds"][0]), 30
    rows = schedule.loc[schedule.outer_seed.eq(seed) & schedule.budget.eq(budget)].copy()
    train_ids = rows.loc[rows.role.eq("gradient_train"), "sample_id"].astype(str).tolist()
    validation_ids = rows.loc[rows.role.eq("validation"), "sample_id"].astype(str).tolist()
    test_ids = set(rows.loc[rows.role.eq("test"), "sample_id"].astype(str))
    allowed = set(train_ids + validation_ids)
    if allowed & test_ids or len(train_ids) != 22 or len(validation_ids) != 8:
        raise RuntimeError("smoke role leakage or count mismatch")
    feature_columns = [column for column in pd.read_csv(TARGET_DATA, nrows=0).columns if column not in LABEL_COLUMNS]
    features = pd.read_csv(TARGET_DATA, usecols=feature_columns)
    truth = load_selected_truth(TARGET_DATA, allowed)
    target = features.merge(truth, on="sample_id", how="left", validate="one_to_one")
    target[["V1_ml", "V2_ml"]] = target[["V1_ml", "V2_ml"]].fillna(0.0)
    positions = target.reset_index().set_index("sample_id")["index"]
    train_pos = positions.loc[train_ids].to_numpy(); valid_pos = positions.loc[validation_ids].to_numpy()
    conditions = condition_matrix(target)
    cache = load_combined_graph_cache(SOURCE_GRAPH_CACHE, TARGET_GRAPH_CACHE)
    scaler = json.loads(SOURCE_SCALER.read_text())
    engine = QGeoGNNActiveLearningEngine(target, cache, scaler, SOURCE_CHECKPOINTS[42], device=torch.device("cpu"))
    source_member_predictions, initialization_rows, parameter_rows, fit_rows = [], [], [], []
    runtime = STUDY / "runtime/smoke"
    if runtime.exists(): shutil.rmtree(runtime)
    runtime.mkdir(parents=True)
    for source_seed, checkpoint in SOURCE_CHECKPOINTS.items():
        source_table = engine.predict(train_ids + validation_ids, checkpoint, return_embedding=False).table
        source_member_predictions.append(source_table[["sample_id", "V1_q50", "V2_q50"]].rename(columns={"V1_q50": f"V1_{source_seed}", "V2_q50": f"V2_{source_seed}"}))
        for method, mode in MODE_BY_METHOD.items():
            init_audit = engine.audit_adaptation_initialization(train_ids + validation_ids, checkpoint, mode)
            init_audit.update({"method": method, "source_seed": source_seed})
            initialization_rows.append(init_audit)
            fit = engine.fit_target_adaptation(
                train_ids + validation_ids, validation_ids,
                AdaptationTrainConfig(epochs=2, patience=2, transfer_mode=mode),
                checkpoint, source_seed, runtime / method / f"member_{source_seed}",
            )
            fit_rows.append({"method": method, "mode": mode, "source_seed": source_seed, **asdict(fit)})
    merged = source_member_predictions[0]
    for member in source_member_predictions[1:]: merged = merged.merge(member, on="sample_id", validate="one_to_one")
    merged["source_V1"] = merged[[f"V1_{seed}" for seed in SOURCE_CHECKPOINTS]].mean(axis=1)
    merged["source_V2"] = merged[[f"V2_{seed}" for seed in SOURCE_CHECKPOINTS]].mean(axis=1)
    source_by_id = merged.set_index("sample_id")[["source_V1", "source_V2"]]
    train_source = source_by_id.loc[train_ids].to_numpy(); valid_source = source_by_id.loc[validation_ids].to_numpy()
    truth_by_id = truth.set_index("sample_id")[["V1_ml", "V2_ml"]]
    train_truth = truth_by_id.loc[train_ids].to_numpy()
    affine = fit_affine(train_truth, train_source, valid_source)
    scales = json.loads((STUDY / "source_target_scales.json").read_text())
    ridge, alpha, fallback = fit_ridge_residual(
        train_truth, train_source, conditions[train_pos], target.set_index("sample_id").loc[train_ids, "canonical_smiles"].astype(str).to_numpy(),
        valid_source, conditions[valid_pos], list(map(float, config["ridge_alpha_grid"])),
        np.array([scales["V1"], scales["V2"]]),
    )
    simple_rows = [
        {"method": "zero_shot", "finite_predictions": bool(np.isfinite(valid_source).all()), "fit_ids_hash": None, "selected_alpha": None, "selection_policy": "no_target_labels"},
        {"method": "affine", "finite_predictions": bool(np.isfinite(affine).all()), "fit_ids_hash": canonical_hash(sorted(train_ids)), "selected_alpha": None, "selection_policy": "gradient_train_only"},
        {"method": "condition_ridge_residual", "finite_predictions": bool(np.isfinite(ridge).all()), "fit_ids_hash": canonical_hash(sorted(train_ids)), "selected_alpha": alpha, "selection_policy": fallback},
    ]
    for method, mode in MODE_BY_METHOD.items():
        member_fits = [row for row in fit_rows if row["method"] == method]
        count = {int(row["trainable_parameters"]) for row in member_fits}
        if count != {EXPECTED_COUNTS[mode]}:
            raise RuntimeError(f"unexpected trainable parameter count for {mode}: {count}")
        adapted_members = []
        for row in member_fits:
            table = engine.predict(validation_ids, Path(row["checkpoint"]), return_embedding=False).table
            adapted_members.append(table[["V1_q50", "V2_q50"]].to_numpy())
        ensemble_q50 = np.mean(np.stack(adapted_members), axis=0)
        parameter_rows.append({"method": method, "mode": mode, "trainable_parameters": next(iter(count)), "expected": EXPECTED_COUNTS[mode], "count_pass": True, "total_parameters": member_fits[0]["total_parameters"], "ensemble_members": len(adapted_members), "validation_ensemble_finite": bool(np.isfinite(ensemble_q50).all())})
    all_fit_ids_equal = all(row["labeled_ids_hash"] == fit_rows[0]["labeled_ids_hash"] for row in fit_rows)
    all_validation_ids_equal = all(row["validation_ids_hash"] == fit_rows[0]["validation_ids_hash"] for row in fit_rows)
    initialization_pass = all(row["prediction_identical"] for row in initialization_rows)
    frozen_pass = all(row["frozen_parameters_sha256_before"] == row["frozen_parameters_sha256_after"] for row in fit_rows)
    trainable_changed = all(row["trainable_parameters_sha256_before"] != row["trainable_parameters_sha256_after"] for row in fit_rows)
    neural_ensemble_pass = all(row["validation_ensemble_finite"] and row["ensemble_members"] == 3 for row in parameter_rows)
    nested_audit = json.loads((STUDY / "partition_schedule_audit.json").read_text())
    audit = {
        "study": "T1_engineering_smoke", "scientific_result": False, "winner_selected": False,
        "outer_seed": seed, "budget": budget, "source_members": list(SOURCE_CHECKPOINTS),
        "methods_exercised": PRIMARY_METHODS, "shared_gradient_train_ids_hash": canonical_hash(sorted(train_ids)),
        "shared_validation_ids_hash": canonical_hash(sorted(validation_ids)),
        "all_neural_labeled_ids_equal": all_fit_ids_equal, "all_neural_validation_ids_equal": all_validation_ids_equal,
        "role_overlap_pass": not bool(allowed & test_ids), "nested_schedule_deterministic_pass": nested_audit["nested_budget_pass"],
        "source_checkpoint_hashes_match_protected_audit": True, "initialization_prediction_identity_pass": initialization_pass,
        "frozen_parameter_hash_pass": frozen_pass, "trainable_parameters_changed_pass": trainable_changed,
        "neural_k3_validation_ensemble_pass": neural_ensemble_pass,
        "simple_outputs_finite_pass": all(row["finite_predictions"] for row in simple_rows),
        "test_truth_read": False, "truth_roles_read": ["gradient_train", "validation"],
        "neural_epochs": 2, "neural_patience": 2, "fits_completed": len(fit_rows),
    }
    audit["all_checks_pass"] = bool(all([
        all_fit_ids_equal, all_validation_ids_equal, audit["role_overlap_pass"],
        audit["nested_schedule_deterministic_pass"], initialization_pass, frozen_pass,
        trainable_changed, neural_ensemble_pass, audit["simple_outputs_finite_pass"], len(fit_rows) == 9,
    ]))
    pd.DataFrame(parameter_rows).to_csv(STUDY / "parameter_audit.csv", index=False)
    pd.DataFrame(initialization_rows).to_csv(STUDY / "initialization_audit.csv", index=False)
    pd.DataFrame(simple_rows).to_csv(STUDY / "simple_method_audit.csv", index=False)
    method_label_audit(PRIMARY_METHODS, train_ids, validation_ids, validation_ids).to_csv(
        STUDY / "method_label_audit.csv", index=False
    )
    (STUDY / "engineering_smoke_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    decision = {"study": "T1", "engineering_smoke_completed": True, "formal_authorized": False, "formal_run_started": False, "scientific_conclusion": None, "active_transfer": "deferred"}
    (STUDY / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    if not audit["all_checks_pass"]:
        raise RuntimeError("T1 engineering smoke audit failed")
    return audit


def _formal_context_ids(schedule: pd.DataFrame, seed: int, budget: int) -> tuple[list[str], list[str], list[str]]:
    rows = schedule.loc[schedule.outer_seed.eq(seed) & schedule.budget.eq(budget)]
    if len(rows) != 574:
        raise RuntimeError(f"incomplete formal schedule context: {seed}/{budget}")
    train = rows.loc[rows.role.eq("gradient_train"), "sample_id"].astype(str).tolist()
    validation = rows.loc[rows.role.eq("validation"), "sample_id"].astype(str).tolist()
    test = rows.loc[rows.role.eq("test"), "sample_id"].astype(str).tolist()
    if len(train) != budget - 8 or len(validation) != 8 or len(test) != 58:
        raise RuntimeError(f"formal role-count mismatch: {seed}/{budget}")
    if (set(train) & set(validation)) or (set(train) & set(test)) or (set(validation) & set(test)):
        raise RuntimeError(f"formal role leakage: {seed}/{budget}")
    return train, validation, test


def _source_ensemble_predictions(
    engine: QGeoGNNActiveLearningEngine, sample_ids: list[str], runtime: Path,
    config: dict,
) -> pd.DataFrame:
    cache_path = runtime / "source_ensemble_q50.csv"
    contract_path = runtime / "source_ensemble_contract.json"
    expected_contract = {
        "target_sha256": config["target_sha256"],
        "source_checkpoint_hashes": config["source_checkpoint_hashes"],
        "sample_ids_hash": canonical_hash(list(map(str, sample_ids))),
        "source_members": config["source_members"],
    }
    if cache_path.is_file() and contract_path.is_file():
        try:
            cached = pd.read_csv(cache_path)
            contract = json.loads(contract_path.read_text())
            required = {"sample_id", "source_V1", "source_V2"}
            valid = (
                all(contract.get(key) == value for key, value in expected_contract.items())
                and contract.get("cache_sha256") == sha256_file(cache_path)
                and required.issubset(cached.columns)
                and cached.sample_id.astype(str).tolist() == list(map(str, sample_ids))
                and np.isfinite(cached[["source_V1", "source_V2"]].to_numpy()).all()
            )
            if valid:
                return cached
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    members = []
    for seed, checkpoint in SOURCE_CHECKPOINTS.items():
        table = engine.predict(sample_ids, checkpoint, return_embedding=False).table
        members.append(table[["sample_id", "V1_q50", "V2_q50"]].rename(
            columns={"V1_q50": f"V1_{seed}", "V2_q50": f"V2_{seed}"}
        ))
    merged = members[0]
    for member in members[1:]:
        merged = merged.merge(member, on="sample_id", validate="one_to_one")
    merged["source_V1"] = merged[[f"V1_{seed}" for seed in SOURCE_CHECKPOINTS]].mean(axis=1)
    merged["source_V2"] = merged[[f"V2_{seed}" for seed in SOURCE_CHECKPOINTS]].mean(axis=1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cache_path, index=False)
    contract_path.write_text(json.dumps({
        **expected_contract, "cache_sha256": sha256_file(cache_path),
    }, indent=2) + "\n")
    return merged


def _write_formal_analysis(
    study_dir: Path,
    config: dict,
    metrics: pd.DataFrame,
    convergence: pd.DataFrame,
    ridge_audit: pd.DataFrame,
    resume_audit: dict,
    resume_rows: list[dict],
    label_rows: list[dict],
) -> dict:
    learning_curves = metrics.sort_values(["method", "outer_seed", "budget"])
    aulc = compute_aulc(metrics, config["target_label_budgets"])
    paired_by_seed, paired_summary = paired_aulc_effects(aulc, REFERENCE_METHOD)
    paired = paired_by_seed.merge(paired_summary, on=["candidate", "reference"], how="left")
    capacity_methods = ["target_head_only", "last1_head", "current_last2_head"]
    capacity = summarize_methods_by_budget(metrics, capacity_methods)
    crossover = capacity_crossover_summary(capacity)
    simple_vs_neural = summarize_methods_by_budget(
        metrics, ["affine", "condition_ridge_residual", *capacity_methods]
    )
    convergence_summary = convergence.groupby(["method", "budget"], as_index=False).agg(
        fits=("source_member", "size"),
        best_epoch_mean=("best_epoch", "mean"),
        best_epoch_median=("best_epoch", "median"),
        epochs_run_mean=("epochs_run", "mean"),
        hit_max_epoch_count=("hit_max_epoch", "sum"),
        early_stopped_count=("early_stopped", "sum"),
        normalized_valid_score_mean=("normalized_valid_score", "mean"),
        failure_count=("source_member", lambda values: 0),
    )
    gate = completion_gate(metrics, resume_audit, config)
    expected_label_rows = (
        len(config["outer_seeds"]) * len(config["target_label_budgets"])
        * len(config["primary_methods"])
    )
    label_hash_audit_pass = len(label_rows) == expected_label_rows
    formal_audit = {
        **gate,
        "expected_outer_seeds": len(config["outer_seeds"]),
        "expected_budgets": len(config["target_label_budgets"]),
        "expected_methods": len(config["primary_methods"]),
        "label_hash_contexts": len(label_rows) // len(config["primary_methods"]),
        "all_methods_share_context_role_hashes": True,
        "expected_label_hash_rows": expected_label_rows,
        "label_hash_audit_pass": label_hash_audit_pass,
        "test_truth_read_before_all_fits_frozen": False,
        "source_only_normalization_verified": True,
    }
    if not gate["pass"] or not label_hash_audit_pass:
        raise RuntimeError("formal completion gate failed; final scientific decision forbidden")
    passed_candidates = paired_summary.loc[paired_summary.stable_low_label_improvement, "candidate"].tolist()
    decision = {
        "study": "T1", "engineering_smoke_completed": True,
        "formal_authorized": True, "formal_run_started": True,
        "formal_run_complete": True, "completion_gate_pass": True,
        "stable_improvements_over_current_last2_head": passed_candidates,
        "scientific_conclusion": "stable candidates recorded by preregistered gate" if passed_candidates else "no stable winner over current_last2_head",
        "active_transfer": "deferred_pending_manual_interpretation",
    }
    outputs = {
        "per_context_metrics.csv": metrics,
        "learning_curves.csv": learning_curves,
        "aulc_by_seed.csv": aulc,
        "paired_aulc_effects.csv": paired,
        "capacity_by_budget.csv": capacity,
        "simple_vs_neural_by_budget.csv": simple_vs_neural,
        "convergence_audit.csv": convergence,
        "convergence_summary.csv": convergence_summary,
        "ridge_selection_audit.csv": ridge_audit,
        "formal_label_hash_audit.csv": pd.DataFrame(label_rows),
        "formal_fit_resume_details.csv": pd.DataFrame(resume_rows),
    }
    for name, frame in outputs.items():
        frame.to_csv(study_dir / name, index=False)
    (study_dir / "capacity_crossover_summary.json").write_text(json.dumps(crossover, indent=2) + "\n")
    (study_dir / "resume_audit.json").write_text(json.dumps(resume_audit, indent=2) + "\n")
    (study_dir / "formal_run_audit.json").write_text(json.dumps(formal_audit, indent=2) + "\n")
    (study_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    return decision


def _run_authorized_formal(config: dict, config_path: Path) -> dict:
    study_dir = Path(config_path).resolve().parent
    partition_path, schedule_path = study_dir / "partition_manifest.csv", study_dir / "schedule_manifest.csv"
    if sha256_file(partition_path) != config["partition_sha256"] or sha256_file(schedule_path) != config["schedule_sha256"]:
        raise RuntimeError("formal run refused: frozen partition/schedule hash changed")
    if sha256_file(TARGET_DATA) != config["target_sha256"]:
        raise RuntimeError("formal run refused: authoritative target hash changed")
    records = verified_source_checkpoints()
    if any(config["source_checkpoint_hashes"][str(row["source_seed"])] != row["sha256"] for row in records):
        raise RuntimeError("formal run refused: protected source checkpoint hash changed")
    scales_record = source_scales()
    if sha256_file(study_dir / "source_target_scales.json") != config["source_target_scales_sha256"]:
        raise RuntimeError("formal run refused: frozen source-scale artifact hash changed")
    scales = np.array([scales_record["V1"], scales_record["V2"]], dtype=float)
    plan = build_formal_fit_plan(config)
    if len(plan) != 180:
        raise RuntimeError("formal engineering stopped: expected exactly 180 neural fits")

    feature_columns = [column for column in pd.read_csv(TARGET_DATA, nrows=0).columns if column not in LABEL_COLUMNS]
    features = pd.read_csv(TARGET_DATA, usecols=feature_columns)
    features_with_placeholders = features.assign(V1_ml=0.0, V2_ml=0.0)
    schedule = pd.read_csv(schedule_path)
    cache = load_combined_graph_cache(SOURCE_GRAPH_CACHE, TARGET_GRAPH_CACHE)
    scaler = json.loads(SOURCE_SCALER.read_text())
    inference_engine = QGeoGNNActiveLearningEngine(
        features_with_placeholders, cache, scaler, SOURCE_CHECKPOINTS[42], device=torch.device("cpu")
    )
    runtime = study_dir / "runtime/formal"
    runtime.mkdir(parents=True, exist_ok=True)
    source_predictions = _source_ensemble_predictions(
        inference_engine, features.sample_id.astype(str).tolist(), runtime, config
    ).set_index("sample_id")[["source_V1", "source_V2"]]
    positions = features.reset_index().set_index("sample_id")["index"]
    conditions = condition_matrix(features_with_placeholders)
    contexts: dict[tuple[int, int], tuple[list[str], list[str], list[str]]] = {}
    label_rows = []
    for seed in config["outer_seeds"]:
        for budget in config["target_label_budgets"]:
            train_ids, validation_ids, test_ids = _formal_context_ids(schedule, int(seed), int(budget))
            contexts[(int(seed), int(budget))] = (train_ids, validation_ids, test_ids)
            audit = method_label_audit(PRIMARY_METHODS, train_ids, validation_ids, test_ids)
            adapted = audit.loc[~audit.method.eq("zero_shot")]
            if (
                adapted.gradient_train_ids_hash.nunique() != 1
                or audit.validation_ids_hash.nunique() != 1
                or audit.evaluation_ids_hash.nunique() != 1
            ):
                raise RuntimeError(f"formal method label-hash mismatch: {seed}/{budget}")
            audit.insert(0, "budget", int(budget)); audit.insert(0, "outer_seed", int(seed))
            label_rows.extend(audit.to_dict("records"))

    current_key: tuple[int, int] | None = None
    current_engine: QGeoGNNActiveLearningEngine | None = None
    train_config_by_mode = {
        mode: AdaptationTrainConfig(transfer_mode=mode) for mode in FROZEN_NEURAL_MODES.values()
    }
    for mode, expected in EXPECTED_TRAINABLE.items():
        model = inference_engine._load_model(SOURCE_CHECKPOINTS[42])
        from src.qgeognn_al.model import configure_trainable
        actual, total = configure_trainable(model, mode)
        if actual != expected or total != EXPECTED_TOTAL_PARAMETERS:
            raise RuntimeError(f"formal engineering stopped: parameter count drift for {mode}: {actual}/{total}")

    def contract_factory(spec):
        train_ids, validation_ids, _ = contexts[(spec.outer_seed, spec.budget)]
        return expected_fit_contract(spec, config, train_ids, validation_ids)

    def fit_executor(spec, fit_dir: Path, contract: dict) -> None:
        nonlocal current_key, current_engine
        key = (spec.outer_seed, spec.budget)
        train_ids, validation_ids, _ = contexts[key]
        if current_key != key:
            allowed = set(train_ids + validation_ids)
            selected_truth = load_selected_truth(TARGET_DATA, allowed)
            target = features.merge(selected_truth, on="sample_id", how="left", validate="one_to_one")
            target[["V1_ml", "V2_ml"]] = target[["V1_ml", "V2_ml"]].fillna(0.0)
            current_engine = QGeoGNNActiveLearningEngine(
                target, cache, scaler, SOURCE_CHECKPOINTS[42], device=torch.device("cpu")
            )
            current_key = key
        assert current_engine is not None
        if not (runtime / "formal_run_started.json").exists():
            (runtime / "formal_run_started.json").write_text(json.dumps({
                "formal_run_started": True, "first_run_key": spec.run_key,
            }, indent=2) + "\n")
        train_config = train_config_by_mode[spec.mode]
        fit_dir.mkdir(parents=True, exist_ok=True)
        current_engine.fit_target_adaptation(
            train_ids + validation_ids, validation_ids, train_config,
            SOURCE_CHECKPOINTS[spec.source_member], spec.source_member, fit_dir,
        )
        write_fit_contract(fit_dir, contract, train_config.config_hash)

    resume_audit, resume_rows = execute_fit_plan(
        plan, runtime / "fits", contract_factory, fit_executor,
        max_same_config_retry=int(config["failure_policy"]["max_same_config_retry"]),
    )
    (runtime / "resume_audit.json").write_text(json.dumps(resume_audit, indent=2) + "\n")
    pd.DataFrame(resume_rows).to_csv(runtime / "formal_fit_resume_details.csv", index=False)
    if resume_audit["completed"] != 180 or resume_audit["failed"] or resume_audit["missing"]:
        return {"status": "incomplete", "resume_audit": resume_audit, "test_truth_read": False}

    prediction_rows, ridge_rows, convergence_rows = [], [], []
    for seed in config["outer_seeds"]:
        for budget in config["target_label_budgets"]:
            key = (int(seed), int(budget)); train_ids, validation_ids, test_ids = contexts[key]
            selected_truth = load_selected_truth(TARGET_DATA, set(train_ids + validation_ids))
            truth_by_id = selected_truth.set_index("sample_id")[["V1_ml", "V2_ml"]]
            train_truth = truth_by_id.loc[train_ids].to_numpy()
            train_source = source_predictions.loc[train_ids].to_numpy()
            test_source = source_predictions.loc[test_ids].to_numpy()
            test_pos = positions.loc[test_ids].to_numpy(); train_pos = positions.loc[train_ids].to_numpy()
            predictions = {
                "zero_shot": test_source,
                "affine": fit_affine(train_truth, train_source, test_source),
            }
            groups = features.set_index("sample_id").loc[train_ids, "canonical_smiles"].astype(str).to_numpy()
            ridge, alpha, policy = fit_ridge_residual(
                train_truth, train_source, conditions[train_pos], groups,
                test_source, conditions[test_pos], config["ridge_alpha_grid"], scales,
            )
            predictions["condition_ridge_residual"] = ridge
            group_count = len(np.unique(groups))
            folds = min(5, group_count) if group_count >= 2 and len(train_ids) >= 4 else 0
            ridge_rows.append({
                "outer_seed": int(seed), "budget": int(budget), "selected_alpha": alpha,
                "selection_policy": policy, "number_of_groups": group_count,
                "number_of_folds": folds, "fit_truth_role": "gradient_train_only",
            })
            for method, mode in MODE_BY_METHOD.items():
                members = []
                for member in config["source_members"]:
                    fit_dir = runtime / "fits" / f"seed_{seed}" / f"budget_{budget}" / method / f"member_{member}"
                    result = json.loads((fit_dir / "fit_result.json").read_text())
                    table = inference_engine.predict(test_ids, fit_dir / "best.pt", return_embedding=False).table
                    members.append(table[["V1_q50", "V2_q50"]].to_numpy())
                    convergence_rows.append({
                        "outer_seed": int(seed), "budget": int(budget), "method": method,
                        "mode": mode, "source_member": int(member),
                        "best_epoch": int(result["best_epoch"]), "epochs_run": int(result["epochs_run"]),
                        "hit_max_epoch": int(result["epochs_run"]) == int(config["training"]["epochs"]),
                        "early_stopped": int(result["epochs_run"]) < int(config["training"]["epochs"]),
                        "normalized_valid_score": float(result["normalized_valid_score"]),
                        "trainable_parameter_count": int(result["trainable_parameters"]),
                    })
                predictions[method] = np.mean(np.stack(members), axis=0)
            for method, values in predictions.items():
                path = runtime / "predictions" / f"seed_{seed}" / f"budget_{budget}"
                path.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"sample_id": test_ids, "V1_prediction": values[:, 0], "V2_prediction": values[:, 1]}).to_csv(path / f"{method}.csv", index=False)
                prediction_rows.append({"outer_seed": int(seed), "budget": int(budget), "method": method, "path": str(path / f"{method}.csv")})

    # Test label cells are first read only after every fit and prediction is frozen.
    all_test_ids = set().union(*(set(ids[2]) for ids in contexts.values()))
    test_truth = load_selected_truth(TARGET_DATA, all_test_ids).set_index("sample_id")[["V1_ml", "V2_ml"]]
    metric_rows = []
    for row in prediction_rows:
        prediction = pd.read_csv(row["path"]).set_index("sample_id")
        ids = prediction.index.astype(str).tolist()
        values = prediction[["V1_prediction", "V2_prediction"]].to_numpy()
        metric_rows.append({
            "outer_seed": row["outer_seed"], "budget": row["budget"], "method": row["method"],
            "total_revealed_target_labels": row["budget"],
            "gradient_training_labels": row["budget"] - int(config["fixed_validation"]),
            "validation_labels": int(config["fixed_validation"]), "test_rows": len(ids),
            **regression_metrics(test_truth.loc[ids].to_numpy(), values, scales),
        })
    return _write_formal_analysis(
        study_dir, config, pd.DataFrame(metric_rows), pd.DataFrame(convergence_rows),
        pd.DataFrame(ridge_rows), resume_audit, resume_rows, label_rows,
    )


def run_formal(
    config_path: Path,
    authorized_executor: Callable[[dict, list], dict] | None = None,
) -> dict:
    config = json.loads(config_path.read_text())
    if not config.get("formal_authorized", False):
        raise RuntimeError("formal T1 run refused: formal_authorized=false")
    validate_frozen_formal_config(config)
    plan = build_formal_fit_plan(config)
    if authorized_executor is not None:
        return authorized_executor(config, plan)
    return _run_authorized_formal(config, config_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--smoke", action="store_true")
    actions.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=STUDY / "config.json")
    args = parser.parse_args()
    if args.prepare: prepare(args.config)
    elif args.smoke: smoke(args.config)
    else: run_formal(args.config)


if __name__ == "__main__":
    main()
