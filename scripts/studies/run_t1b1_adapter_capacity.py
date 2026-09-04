#!/usr/bin/env python3
"""Prepare, smoke-test, and eventually run the frozen T1b-1 capacity sweep."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
from dataclasses import asdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch

from src.qgeognn_al.artifacts import sha256_file
from src.qgeognn_al.data import load_combined_graph_cache
from src.qgeognn_al.engine import GraphAdapterTrainConfig, QGeoGNNActiveLearningEngine
from src.qgeognn_al.model import (
    ResidualGraphAdapterHead, build_model, configure_graph_adapter_trainable,
    configure_trainable, graph_representation_dim, install_graph_residual_adapter,
    install_monotonic_head,
)
from src.qgeognn_al.resources import (
    ROOT, SOURCE_CHECKPOINTS, SOURCE_GRAPH_CACHE, SOURCE_SCALER, TARGET_DATA,
    TARGET_GRAPH_CACHE, verified_source_checkpoints,
)
from src.qgeognn_al.t1_formal import (
    execute_fit_plan, regression_metrics, write_fit_contract,
)
from src.qgeognn_al.t1b1 import (
    ADAPTER_METHODS, BUDGETS, CAPACITY_METHODS, PRIMARY_REFERENCE,
    adapter_config, build_fit_plan, capacity_summaries, completion_gate,
    expected_contract, paired_effects, validate_config,
)


STUDY = ROOT / "studies/track_b_transfer/t1b1_adapter_capacity"
T1A = ROOT / "studies/track_b_transfer/t1_low_label_adaptation"
LABEL_COLUMNS = {"V1_ml", "V2_ml"}


def _verify_frozen_inputs(config: dict) -> None:
    paths = {
        "partition_sha256": T1A / "partition_manifest.csv",
        "schedule_sha256": T1A / "schedule_manifest.csv",
        "per_context_metrics_sha256": T1A / "per_context_metrics.csv",
        "convergence_audit_sha256": T1A / "convergence_audit.csv",
        "source_target_scales_sha256": T1A / "source_target_scales.json",
        "target_sha256": TARGET_DATA,
    }
    for key, path in paths.items():
        if sha256_file(path) != config["t1a_artifacts"][key]:
            raise RuntimeError(f"frozen T1a artifact changed: {key}")
    observed = {str(row["source_seed"]): row["sha256"] for row in verified_source_checkpoints()}
    if observed != config["source_checkpoint_hashes"]:
        raise RuntimeError("protected source checkpoint hashes changed")


def _context_ids(schedule: pd.DataFrame, seed: int, budget: int) -> tuple[list[str], list[str], list[str]]:
    rows = schedule.loc[schedule.outer_seed.eq(seed) & schedule.budget.eq(budget)]
    roles = {
        role: rows.loc[rows.role.eq(role), "sample_id"].astype(str).tolist()
        for role in ("gradient_train", "validation", "test")
    }
    if len(rows) != 574 or len(roles["gradient_train"]) != budget - 8 or len(roles["validation"]) != 8 or len(roles["test"]) != 58:
        raise RuntimeError(f"invalid frozen T1a context: {seed}/{budget}")
    sets = [set(roles[name]) for name in ("gradient_train", "validation", "test")]
    if (sets[0] & sets[1]) or (sets[0] & sets[2]) or (sets[1] & sets[2]):
        raise RuntimeError(f"role leakage in frozen T1a context: {seed}/{budget}")
    return roles["gradient_train"], roles["validation"], roles["test"]


def _load_selected_truth(allowed_ids: set[str]) -> pd.DataFrame:
    ids = pd.read_csv(TARGET_DATA, usecols=["sample_id"])
    skiprows = [index + 1 for index, value in enumerate(ids.sample_id.astype(str)) if value not in allowed_ids]
    truth = pd.read_csv(TARGET_DATA, usecols=["sample_id", "V1_ml", "V2_ml"], skiprows=skiprows)
    if set(truth.sample_id.astype(str)) != allowed_ids:
        raise RuntimeError("selected target truth IDs do not match allowed train/validation IDs")
    return truth


def _parameter_hash(model: torch.nn.Module, prefix: str) -> str:
    digest = hashlib.sha256()
    matched = False
    for name, parameter in model.named_parameters():
        if name.startswith(prefix):
            matched = True
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    if not matched:
        raise RuntimeError(f"parameter scope is empty: {prefix}")
    return digest.hexdigest()


def prepare(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    validate_config(config)
    if config["formal_authorized"]:
        raise RuntimeError("engineering preparation requires formal_authorized=false")
    _verify_frozen_inputs(config)
    base = build_model(torch.device("cpu"))
    install_monotonic_head(base)
    actual_dim = graph_representation_dim(base)
    if actual_dim != int(config["graph_representation_dim"]):
        raise RuntimeError(f"graph representation dimension mismatch: config={config['graph_representation_dim']} actual={actual_dim}")
    rows = []
    baseline_modes = {
        "target_head_only": "head_only", "last1_head": "last1_head",
        "current_last2_head": "last2_head",
    }
    for method, mode in baseline_modes.items():
        model = build_model(torch.device("cpu")); install_monotonic_head(model)
        trainable, total = configure_trainable(model, mode)
        expected = config["parameter_counts"][method]
        rows.append({
            "method": method, "bottleneck_width": None, "graph_representation_dim": actual_dim,
            "adapter_parameters": 0, "head_parameters": 774,
            "total_trainable_parameters": trainable, "total_model_parameters": total,
            "expected_trainable_parameters": expected["total_trainable_parameters"],
            "count_pass": trainable == expected["total_trainable_parameters"] and total == expected["total_model_parameters"],
        })
    for method, width in ADAPTER_METHODS.items():
        model = build_model(torch.device("cpu")); install_monotonic_head(model)
        detected = install_graph_residual_adapter(model, width)
        trainable, total, adapter_parameters, head_parameters = configure_graph_adapter_trainable(model)
        theoretical = 2 * detected * width + width + detected
        expected = config["parameter_counts"][method]
        rows.append({
            "method": method, "bottleneck_width": width, "graph_representation_dim": detected,
            "adapter_parameters": adapter_parameters, "head_parameters": head_parameters,
            "total_trainable_parameters": trainable, "total_model_parameters": total,
            "expected_trainable_parameters": expected["total_trainable_parameters"],
            "count_pass": adapter_parameters == theoretical and trainable == expected["total_trainable_parameters"] and total == expected["total_model_parameters"],
        })
    audit = pd.DataFrame(rows)
    audit["bottleneck_width"] = audit["bottleneck_width"].astype("Int64")
    if not audit.count_pass.all():
        raise RuntimeError("parameter capacity audit failed")
    adapter_counts = audit.loc[audit.method.isin(ADAPTER_METHODS), "total_trainable_parameters"].tolist()
    if adapter_counts != sorted(adapter_counts) or len(set(adapter_counts)) != 3 or max(adapter_counts) >= 93454:
        raise RuntimeError("adapter capacities must increase strictly and remain below last1_head")
    audit.to_csv(STUDY / "parameter_capacity_audit.csv", index=False)
    plan = build_fit_plan(config)
    plan_audit = {
        "formal_authorized": False, "expected_new_adapter_fits": 180,
        "actual_new_adapter_fits": len(plan), "unique_run_keys": len({item.run_key for item in plan}),
        "duplicate_run_keys": len(plan) - len({item.run_key for item in plan}),
        "first_run_key": plan[0].run_key, "last_run_key": plan[-1].run_key,
        "t1a_frozen_baseline_fits_not_repeated": True, "test_truth_read": False,
    }
    (STUDY / "formal_plan_audit.json").write_text(json.dumps(plan_audit, indent=2) + "\n")
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "torch", "torch-geometric"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    (STUDY / "environment.json").write_text(json.dumps({
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages, "device": "cpu",
    }, indent=2) + "\n")
    return plan_audit


def preflight(config_path: Path) -> dict:
    """Verify the frozen formal contract without reading any target label cell."""
    config = json.loads(config_path.read_text())
    validate_config(config)
    if config.get("formal_authorized") is not False:
        raise RuntimeError("formal preflight requires formal_authorized=false")
    _verify_frozen_inputs(config)
    checks: dict[str, bool] = {}
    checks["outer_seeds"] = config["outer_seeds"] == [769539383, 1425370602, 536279090, 2767143051, 1362771960]
    checks["budgets"] = config["target_label_budgets"] == [30, 50, 70, 100]
    checks["fixed_validation"] = config["fixed_validation"] == 8
    checks["adapter_widths"] = config["bottleneck_widths"] == [8, 16, 32]
    checks["adapter_structure"] = (
        config["adapter_type"] == "graph_level_residual"
        and config["activation"] == "relu"
        and config["up_projection_initialization"] == "zeros"
    )
    checks["primary_reference"] = config["primary_reference"] == "target_head_only"
    checks["stable_gate"] = config["stable_improvement_gate"] == {
        "mean_delta_lt": 0, "median_delta_lt": 0,
        "minimum_wins": 4, "outer_seed_count": 5,
    }
    expected_training = {
        "learning_rate": 1e-4, "weight_decay": 1e-5, "epochs": 500,
        "patience": 100, "batch_size": 2048, "v1_weight": 1.0,
        "v2_weight": 1.0, "scaler_policy": "source_train",
        "quantile_parameterization": "monotonic_softplus",
        "checkpoint_selection": "validation_normalized_mse",
    }
    checks["training_contract"] = config["training"] == expected_training

    schedule = pd.read_csv(T1A / "schedule_manifest.csv")
    role_counts_pass = True
    shared_roles_pass = True
    role_hashes = []
    for seed in config["outer_seeds"]:
        for budget in config["target_label_budgets"]:
            train_ids, validation_ids, test_ids = _context_ids(schedule, int(seed), int(budget))
            role_counts_pass &= (
                len(train_ids) == int(budget) - 8
                and len(validation_ids) == 8 and len(test_ids) == 58
            )
            hashes = {
                "outer_seed": int(seed), "budget": int(budget),
                "gradient_train_ids_hash": hashlib.sha256(json.dumps(sorted(train_ids)).encode()).hexdigest(),
                "validation_ids_hash": hashlib.sha256(json.dumps(sorted(validation_ids)).encode()).hexdigest(),
                "test_ids_hash": hashlib.sha256(json.dumps(sorted(test_ids)).encode()).hexdigest(),
            }
            role_hashes.append(hashes)
            shared_roles_pass &= len(config["adapter_methods"]) == 3
    checks["role_counts"] = bool(role_counts_pass)
    checks["same_role_ids_for_all_adapters"] = bool(shared_roles_pass)

    parameter_rows = []
    base = build_model(torch.device("cpu")); install_monotonic_head(base)
    detected_dim = graph_representation_dim(base)
    checks["graph_representation_dim_128"] = detected_dim == config["graph_representation_dim"] == 128
    for method, width in config["adapter_methods"].items():
        model = build_model(torch.device("cpu")); install_monotonic_head(model)
        actual_dim = install_graph_residual_adapter(model, int(width))
        trainable, total, adapter_parameters, head_parameters = configure_graph_adapter_trainable(model)
        expected = config["parameter_counts"][method]
        parameter_rows.append({
            "method": method, "width": int(width), "input_dim": actual_dim,
            "adapter_parameters": adapter_parameters, "head_parameters": head_parameters,
            "trainable_parameters": trainable, "total_parameters": total,
            "pass": (
                adapter_parameters == expected["adapter_parameters"]
                and head_parameters == expected["head_parameters"]
                and trainable == expected["total_trainable_parameters"]
                and total == expected["total_model_parameters"]
            ),
        })
        GraphAdapterTrainConfig(bottleneck_width=int(width)).validate()
    for method, mode in {
        "target_head_only": "head_only", "last1_head": "last1_head",
        "current_last2_head": "last2_head",
    }.items():
        model = build_model(torch.device("cpu")); install_monotonic_head(model)
        trainable, total = configure_trainable(model, mode)
        expected = config["parameter_counts"][method]
        parameter_rows.append({
            "method": method, "width": None, "input_dim": detected_dim,
            "adapter_parameters": 0, "head_parameters": 774,
            "trainable_parameters": trainable, "total_parameters": total,
            "pass": trainable == expected["total_trainable_parameters"] and total == expected["total_model_parameters"],
        })
    checks["parameter_counts"] = all(row["pass"] for row in parameter_rows)

    target_header = pd.read_csv(TARGET_DATA, nrows=0).columns.tolist()
    feature_columns = [column for column in target_header if column not in LABEL_COLUMNS]
    features = pd.read_csv(TARGET_DATA, usecols=feature_columns)
    placeholders = features.assign(V1_ml=0.0, V2_ml=0.0)
    cache = load_combined_graph_cache(SOURCE_GRAPH_CACHE, TARGET_GRAPH_CACHE)
    scaler = json.loads(SOURCE_SCALER.read_text())
    engine = QGeoGNNActiveLearningEngine(placeholders, cache, scaler, SOURCE_CHECKPOINTS[42], device=torch.device("cpu"))
    train_ids, validation_ids, _ = _context_ids(schedule, int(config["outer_seeds"][0]), 30)
    identity_rows = []
    for method, width in config["adapter_methods"].items():
        for member in config["source_members"]:
            row = engine.audit_graph_adapter_initialization(
                train_ids + validation_ids, SOURCE_CHECKPOINTS[int(member)], int(width)
            )
            row.update({"method": method, "source_member": int(member)})
            identity_rows.append(row)
    checks["source_function_identity"] = all(
        row["prediction_identical"] and row["maximum_absolute_prediction_difference"] <= 1e-7
        for row in identity_rows
    )
    plan = build_fit_plan(config)
    checks["expected_new_adapter_fits_180"] = len(plan) == 180
    checks["unique_run_keys_180"] = len({item.run_key for item in plan}) == 180
    checks["duplicate_run_keys_zero"] = len(plan) - len({item.run_key for item in plan}) == 0
    failed = sorted(key for key, passed in checks.items() if not passed)
    audit = {
        "schema_version": 1,
        "status": "passed" if not failed else "failed",
        "formal_authorized": False,
        "test_truth_read": False,
        "target_columns_read": feature_columns,
        "checks": checks,
        "failed_checks": failed,
        "parameter_audit": parameter_rows,
        "initialization_identity_audit": identity_rows,
        "role_hash_contexts": role_hashes,
        "fit_plan": {
            "expected": 180, "actual": len(plan),
            "unique": len({item.run_key for item in plan}),
            "duplicates": len(plan) - len({item.run_key for item in plan}),
            "first_run_key": plan[0].run_key, "last_run_key": plan[-1].run_key,
        },
    }
    (STUDY / "formal_preflight_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if failed:
        raise RuntimeError(f"T1b-1 formal preflight failed: {failed}")
    return audit


def smoke(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    validate_config(config)
    if config["formal_authorized"]:
        raise RuntimeError("engineering smoke requires formal_authorized=false")
    _verify_frozen_inputs(config)
    schedule = pd.read_csv(T1A / "schedule_manifest.csv")
    seed, budget = int(config["outer_seeds"][0]), 30
    train_ids, validation_ids, test_ids = _context_ids(schedule, seed, budget)
    allowed = set(train_ids + validation_ids)
    if allowed & set(test_ids):
        raise RuntimeError("smoke label leakage")
    feature_columns = [column for column in pd.read_csv(TARGET_DATA, nrows=0).columns if column not in LABEL_COLUMNS]
    features = pd.read_csv(TARGET_DATA, usecols=feature_columns)
    truth = _load_selected_truth(allowed)
    target = features.merge(truth, on="sample_id", how="left", validate="one_to_one")
    target[["V1_ml", "V2_ml"]] = target[["V1_ml", "V2_ml"]].fillna(0.0)
    cache = load_combined_graph_cache(SOURCE_GRAPH_CACHE, TARGET_GRAPH_CACHE)
    scaler = json.loads(SOURCE_SCALER.read_text())
    engine = QGeoGNNActiveLearningEngine(target, cache, scaler, SOURCE_CHECKPOINTS[42], device=torch.device("cpu"))
    runtime = STUDY / "runtime/smoke"
    runtime.mkdir(parents=True, exist_ok=True)
    initialization_rows, fit_rows, label_rows, ensemble_rows = [], [], [], []
    for method, width in ADAPTER_METHODS.items():
        method_fits = []
        for member in config["source_members"]:
            initialization = engine.audit_graph_adapter_initialization(
                train_ids + validation_ids, SOURCE_CHECKPOINTS[int(member)], width,
            )
            initialization.update({"method": method, "source_member": int(member)})
            initialization_rows.append(initialization)
            random.seed(int(member)); np.random.seed(int(member)); torch.manual_seed(int(member))
            initial_model = engine._load_model(SOURCE_CHECKPOINTS[int(member)])
            install_graph_residual_adapter(initial_model, width)
            configure_graph_adapter_trainable(initial_model)
            adapter_before = _parameter_hash(initial_model, "graph_pred_linear.adapter.")
            head_before = _parameter_hash(initial_model, "graph_pred_linear.head.")
            fit_dir = runtime / method / f"member_{member}"
            result = engine.fit_graph_adapter(
                train_ids + validation_ids, validation_ids,
                GraphAdapterTrainConfig(bottleneck_width=width, epochs=2, patience=2),
                SOURCE_CHECKPOINTS[int(member)], int(member), fit_dir,
            )
            trained = engine._load_model(Path(result.checkpoint))
            configure_graph_adapter_trainable(trained)
            row = {
                "method": method, "bottleneck_width": width, "source_member": int(member),
                **asdict(result),
                "frozen_gnn_unchanged": result.frozen_parameters_sha256_before == result.frozen_parameters_sha256_after,
                "adapter_changed": adapter_before != _parameter_hash(trained, "graph_pred_linear.adapter."),
                "head_changed": head_before != _parameter_hash(trained, "graph_pred_linear.head."),
            }
            fit_rows.append(row); method_fits.append(row)
            label_rows.append({
                "method": method, "source_member": int(member),
                "gradient_train_ids_hash": hashlib.sha256(json.dumps(sorted(train_ids)).encode()).hexdigest(),
                "validation_ids_hash": hashlib.sha256(json.dumps(sorted(validation_ids)).encode()).hexdigest(),
                "evaluation_ids_hash": hashlib.sha256(json.dumps(sorted(validation_ids)).encode()).hexdigest(),
            })
        predictions = []
        for row in method_fits:
            table = engine.predict(validation_ids, Path(row["checkpoint"]), return_embedding=False).table
            predictions.append(table[["V1_q50", "V2_q50"]].to_numpy())
        ensemble_rows.append({
            "method": method, "members": len(predictions),
            "k3_q50_mean_finite": bool(np.isfinite(np.mean(np.stack(predictions), axis=0)).all()),
        })
    initialization = pd.DataFrame(initialization_rows)
    fits = pd.DataFrame(fit_rows)
    labels = pd.DataFrame(label_rows)
    ensembles = pd.DataFrame(ensemble_rows)
    checks = {
        "smoke_fits_9": len(fits) == 9,
        "initialization_identity": bool(initialization.prediction_identical.all()),
        "maximum_initialization_difference_le_1e_7": bool(initialization.maximum_absolute_prediction_difference.max() <= 1e-7),
        "graph_representation_dim_128": set(initialization.graph_representation_dim) == {128},
        "frozen_gnn_hash_unchanged": bool(fits.frozen_gnn_unchanged.all()),
        "adapter_parameters_changed": bool(fits.adapter_changed.all()),
        "prediction_head_parameters_changed": bool(fits.head_changed.all()),
        "same_gradient_ids": labels.gradient_train_ids_hash.nunique() == 1,
        "same_validation_ids": labels.validation_ids_hash.nunique() == 1,
        "same_evaluation_ids": labels.evaluation_ids_hash.nunique() == 1,
        "k3_inference_finite": bool(ensembles.k3_q50_mean_finite.all()),
        "no_test_truth": True,
    }
    all_pass = all(value is True for value in checks.values())
    initialization.to_csv(STUDY / "initialization_audit.csv", index=False)
    fits.to_csv(STUDY / "smoke_fit_audit.csv", index=False)
    labels.to_csv(STUDY / "smoke_label_hash_audit.csv", index=False)
    ensembles.to_csv(STUDY / "smoke_ensemble_audit.csv", index=False)
    audit = {
        "study": "T1b1_adapter_capacity", "engineering_only": True,
        "outer_seed": seed, "budget": budget, "epochs": 2, "patience": 2,
        "test_truth_read": False, "checks": checks, "all_checks_pass": all_pass,
        "scientific_result": None, "winner_selected": False,
    }
    (STUDY / "engineering_smoke_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    decision = json.loads((STUDY / "decision.json").read_text())
    decision["engineering_smoke_completed"] = bool(all_pass)
    (STUDY / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    if not all_pass:
        raise RuntimeError(f"T1b-1 smoke failed: {[key for key, value in checks.items() if value is not True]}")
    return audit


def _run_authorized_formal(config: dict, config_path: Path) -> dict:
    _verify_frozen_inputs(config)
    schedule = pd.read_csv(T1A / "schedule_manifest.csv")
    feature_columns = [column for column in pd.read_csv(TARGET_DATA, nrows=0).columns if column not in LABEL_COLUMNS]
    features = pd.read_csv(TARGET_DATA, usecols=feature_columns)
    placeholders = features.assign(V1_ml=0.0, V2_ml=0.0)
    cache = load_combined_graph_cache(SOURCE_GRAPH_CACHE, TARGET_GRAPH_CACHE)
    scaler = json.loads(SOURCE_SCALER.read_text())
    inference = QGeoGNNActiveLearningEngine(placeholders, cache, scaler, SOURCE_CHECKPOINTS[42], device=torch.device("cpu"))
    contexts = {
        (int(seed), int(budget)): _context_ids(schedule, int(seed), int(budget))
        for seed in config["outer_seeds"] for budget in config["target_label_budgets"]
    }
    runtime = Path(config_path).resolve().parent / "runtime/formal"
    plan = build_fit_plan(config)
    current_key, current_engine = None, None

    def contract_factory(spec):
        train_ids, validation_ids, _ = contexts[(spec.outer_seed, spec.budget)]
        return expected_contract(spec, config, train_ids, validation_ids)

    def fit_executor(spec, fit_dir, contract):
        nonlocal current_key, current_engine
        key = (spec.outer_seed, spec.budget)
        train_ids, validation_ids, _ = contexts[key]
        if current_key != key:
            truth = _load_selected_truth(set(train_ids + validation_ids))
            target = features.merge(truth, on="sample_id", how="left", validate="one_to_one")
            target[["V1_ml", "V2_ml"]] = target[["V1_ml", "V2_ml"]].fillna(0.0)
            current_engine = QGeoGNNActiveLearningEngine(target, cache, scaler, SOURCE_CHECKPOINTS[42], device=torch.device("cpu"))
            current_key = key
        train_config = adapter_config(spec)
        current_engine.fit_graph_adapter(
            train_ids + validation_ids, validation_ids, train_config,
            SOURCE_CHECKPOINTS[spec.source_member], spec.source_member, fit_dir,
        )
        write_fit_contract(fit_dir, contract, train_config.config_hash)

    resume, details = execute_fit_plan(
        plan, runtime / "fits", contract_factory, fit_executor,
        max_same_config_retry=int(config["failure_policy"]["max_same_config_retry"]),
    )
    if resume["completed"] != 180 or resume["failed"] or resume["missing"]:
        return {"status": "incomplete", "test_truth_read": False, "resume_audit": resume}
    prediction_rows = []
    convergence = []
    for seed in config["outer_seeds"]:
        for budget in config["target_label_budgets"]:
            _, _, test_ids = contexts[(int(seed), int(budget))]
            for method in ADAPTER_METHODS:
                members = []
                for member in config["source_members"]:
                    fit_dir = runtime / "fits" / f"seed_{seed}/budget_{budget}/{method}/member_{member}"
                    table = inference.predict(test_ids, fit_dir / "best.pt", return_embedding=False).table
                    members.append(table[["V1_q50", "V2_q50"]].to_numpy())
                    result = json.loads((fit_dir / "fit_result.json").read_text())
                    convergence.append({
                        "outer_seed": seed, "budget": budget, "method": method, "source_member": member,
                        "mode": "graph_adapter_head",
                        "best_epoch": result["best_epoch"], "epochs_run": result["epochs_run"],
                        "early_stopped": result["epochs_run"] < 500, "hit_max_epoch": result["epochs_run"] == 500,
                        "normalized_valid_score": result["normalized_valid_score"],
                        "trainable_parameter_count": result["trainable_parameters"],
                    })
                values = np.mean(np.stack(members), axis=0)
                path = runtime / f"predictions/seed_{seed}/budget_{budget}/{method}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"sample_id": test_ids, "V1_prediction": values[:, 0], "V2_prediction": values[:, 1]}).to_csv(path, index=False)
                prediction_rows.append({"outer_seed": seed, "budget": budget, "method": method, "path": str(path)})
    all_test_ids = set().union(*(set(value[2]) for value in contexts.values()))
    test_truth = _load_selected_truth(all_test_ids).set_index("sample_id")[["V1_ml", "V2_ml"]]
    scales_record = json.loads((T1A / "source_target_scales.json").read_text())
    scales = [scales_record["V1"], scales_record["V2"]]
    adapter_metrics = []
    for row in prediction_rows:
        pred = pd.read_csv(row["path"]).set_index("sample_id")
        ids = pred.index.astype(str).tolist()
        adapter_metrics.append({
            "outer_seed": row["outer_seed"], "budget": row["budget"], "method": row["method"],
            **regression_metrics(test_truth.loc[ids].to_numpy(), pred[["V1_prediction", "V2_prediction"]].to_numpy(), scales),
        })
    frozen = pd.read_csv(T1A / "per_context_metrics.csv")
    frozen = frozen.loc[frozen.method.isin(["target_head_only", "last1_head", "current_last2_head"])]
    metrics = pd.concat([frozen, pd.DataFrame(adapter_metrics)], ignore_index=True, sort=False)
    parameter_map = {method: values["total_trainable_parameters"] for method, values in config["parameter_counts"].items()}
    curve, aulc_summary = capacity_summaries(metrics, parameter_map)
    from src.qgeognn_al.t1_formal import compute_aulc
    aulc = compute_aulc(metrics, BUDGETS)
    paired_rows, paired_summary = paired_effects(aulc)
    gate = completion_gate(metrics, resume)
    if not gate["pass"]:
        raise RuntimeError("incomplete T1b-1 result forbids final decision")
    study_dir = Path(config_path).resolve().parent
    metrics.to_csv(study_dir / "per_context_metrics.csv", index=False)
    curve.to_csv(study_dir / "capacity_curve.csv", index=False)
    aulc_summary.to_csv(study_dir / "capacity_aulc_summary.csv", index=False)
    paired_rows.merge(paired_summary, on=["candidate", "reference"]).to_csv(study_dir / "paired_aulc_effects.csv", index=False)
    frozen_convergence = pd.read_csv(T1A / "convergence_audit.csv")
    frozen_convergence = frozen_convergence.loc[
        frozen_convergence.method.isin(["target_head_only", "last1_head", "current_last2_head"])
    ]
    all_convergence = pd.concat([frozen_convergence, pd.DataFrame(convergence)], ignore_index=True, sort=False)
    all_convergence.to_csv(study_dir / "convergence_audit.csv", index=False)
    pd.DataFrame(details).to_csv(study_dir / "formal_fit_resume_details.csv", index=False)
    (study_dir / "resume_audit.json").write_text(json.dumps(resume, indent=2) + "\n")
    (study_dir / "formal_run_audit.json").write_text(json.dumps({**gate, "test_truth_read_before_all_predictions_frozen": False}, indent=2) + "\n")
    import matplotlib.pyplot as plt
    order = CAPACITY_METHODS
    plot = aulc_summary.set_index("method").loc[order]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.log10(plot.trainable_parameters), plot.mean_normalized_AULC, marker="o")
    labels = {
        "target_head_only": ("Head", (-8, -16)),
        "graph_adapter_r8": ("r8", (-8, 9)),
        "graph_adapter_r16": ("r16", (-4, -16)),
        "graph_adapter_r32": ("r32", (5, 9)),
        "last1_head": ("Last1", (0, 8)),
        "current_last2_head": ("Last2", (-28, 8)),
    }
    for method, row in plot.iterrows():
        label, offset = labels[method]
        ax.annotate(
            label,
            (np.log10(row.trainable_parameters), row.mean_normalized_AULC),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )
    ax.set(xlabel="log10(trainable parameters)", ylabel="mean normalized AULC")
    ax.margins(y=0.1)
    fig.tight_layout(); fig.savefig(study_dir / "t1b1_capacity_curve.png", dpi=180); plt.close(fig)
    winners = paired_summary.loc[paired_summary["pass"], "candidate"].tolist()
    decision = {
        "study": "T1b1_adapter_capacity", "scientific_status": "formal_complete",
        "engineering_smoke_completed": True, "formal_authorized": True,
        "formal_run_started": True, "scientific_conclusion": "stable adapters recorded" if winners else "no adapter stably better than target_head_only",
        "stable_improvements_over_target_head_only": winners,
        "t1b2_authorized": False, "active_transfer": "deferred",
    }
    (study_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    return decision


def run_formal(config_path: Path, authorized_executor=None) -> dict:
    config = json.loads(config_path.read_text())
    if not config.get("formal_authorized", False):
        raise RuntimeError("formal T1b-1 run refused: formal_authorized=false")
    validate_config(config)
    plan = build_fit_plan(config)
    if authorized_executor is not None:
        return authorized_executor(config, plan)
    return _run_authorized_formal(config, config_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--smoke", action="store_true")
    actions.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=STUDY / "config.json")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.config)
    elif args.preflight:
        preflight(args.config)
    elif args.smoke:
        smoke(args.config)
    else:
        run_formal(args.config)


if __name__ == "__main__":
    main()
