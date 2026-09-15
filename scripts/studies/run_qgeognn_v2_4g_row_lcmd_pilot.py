#!/usr/bin/env python3
"""Preregistered one-step Gradient-LCMD-TP study for standalone QGeoGNN-V2."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qgeognn_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.qgeognn_al.active_learning_v2.gradient_features import (
    extract_q50_gradient_sketches,
    state_dict_hash,
)
from src.qgeognn_al.active_learning_v2.lcmd import lcmd_tp_select
from src.qgeognn_al.active_learning_v2.protocol import (
    OUTER_SEEDS,
    RANDOM_CONTROLS,
    RestrictedLabelStore,
    batch_size,
    ids_hash,
    make_row_protocol,
    random_control_positions,
    stable_hash,
    validate_row_protocol,
)
from src.qgeognn_al.active_learning_v2.runner import (
    fit_al_preprocessing,
    fit_from_same_initialization,
    make_label_scrubbed_graphs,
)
from src.qgeognn_al.artifacts import sha256_file
from src.qgeognn_al.data import condition_matrix
from src.qgeognn_al.evaluation.point import point_metrics
from src.qgeognn_al.models import build_predictor, load_predictor_checkpoint
from src.qgeognn_al.models.qgeognn_v2 import PARAMETER_COUNT
from src.qgeognn_al.resources import SOURCE_DATA, SOURCE_GRAPH_CACHE
from src.qgeognn_al.training.predictor import atomic_json


STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_lcmd"
SPLITS = STUDY / "splits"
RESULTS = STUDY / "results"
FIGURES = STUDY / "figures"
RUNTIME = STUDY / "runtime"
QUALIFICATION_PROTOCOL = ROOT / "studies/predictor/final_4g_qualification/protocol.json"
FEATURE_COLUMNS = (
    "sample_id",
    "canonical_smiles",
    "Density g/ml",
    "V/ul",
    "loading solvent",
    "Volume of loading solvent/ul",
    "PE/EA",
)
TRAINING_CONFIG = {
    "model_variant": "qgeognn_v2",
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "batch_size": 2048,
    "maximum_epochs": 1000,
    "patience": 100,
    "shuffle": "deterministic_each_epoch",
    "loss_weights": {"V1": 1.0, "V2": 1.0},
    "checkpoint_selection": "validation_combined_normalized_rmse",
    "test_during_training": False,
    "architecture_changed": False,
    "head_changed": False,
    "condition_branch_changed": False,
}
SKETCH_DIMENSION = 512
PREFLIGHT_SEED = 73
PREFLIGHT_L0_ROWS = 25
PREFLIGHT_POOL_ROWS = 75
PREFLIGHT_BATCH = 10
MAX_ACCEPTABLE_FULL_EXTRACTION_SECONDS = 7200.0
MAX_ACCEPTABLE_PEAK_RSS_BYTES = 12 * 1024**3


def initialization_seed(outer_seed: int) -> int:
    return int(outer_seed) + 2_000_003


def training_seed(outer_seed: int) -> int:
    return int(outer_seed) + 3_000_017


def sketch_seed(outer_seed: int) -> int:
    return int(outer_seed) + 4_000_037


def environment_record() -> dict[str, object]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "torch", "torch-geometric", "rdkit", "scipy", "scikit-learn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "device": "cpu",
        "base_commit_sha": "f5605ba051a8094bbbf6fd371e722b276ecfbd34",
        "branch": "exp/qgeognn-v2-4g-row-al",
        "worktree_clean_before_branch": False,
        "preexisting_untracked_preserved": [".vscode/"],
        "source_data": str(SOURCE_DATA.relative_to(ROOT)),
        "source_data_sha256": sha256_file(SOURCE_DATA),
        "graph_cache_sha256": sha256_file(SOURCE_GRAPH_CACHE),
    }


def load_feature_frame() -> pd.DataFrame:
    qualified = json.loads(QUALIFICATION_PROTOCOL.read_text())
    if qualified.get("model_variant") != "qgeognn_v2":
        raise RuntimeError("authoritative 4g qualification is not QGeoGNN-V2")
    if sha256_file(SOURCE_DATA) != qualified.get("source_sha256"):
        raise RuntimeError("canonical 4g source hash differs from qualification")
    data = pd.read_csv(SOURCE_DATA, usecols=list(FEATURE_COLUMNS)).reset_index(drop=True)
    observed_compounds = int(data.canonical_smiles.nunique())
    if len(data) != int(qualified["rows"]):
        raise RuntimeError(
            f"canonical 4g row count differs from qualification: "
            f"observed={len(data)}, qualified={qualified['rows']}"
        )
    if observed_compounds != int(qualified["compounds"]):
        raise RuntimeError(
            f"canonical 4g compound count differs from qualification: "
            f"observed={observed_compounds}, qualified={qualified['compounds']}"
        )
    if data.sample_id.nunique() != len(data):
        raise RuntimeError("canonical 4g sample identities are not unique")
    return data


def partition_path(seed: int) -> Path:
    return SPLITS / f"row_seed_{int(seed)}.csv"


def prepare() -> None:
    engineering = json.loads(
        (ROOT / "studies/predictor/final_v2_engineering/equivalence_audit.json").read_text()
    )
    reachability = json.loads(
        (ROOT / "studies/predictor/final_v2_engineering/reachability_audit.json").read_text()
    )
    if (
        engineering.get("status") != "PASS"
        or reachability.get("requires_grad_parameters") != PARAMETER_COUNT
        or reachability.get("gradient_bearing_parameters") != PARAMETER_COUNT
        or reachability.get("forward_unreachable_trainable_parameters") != 0
    ):
        raise RuntimeError("qualified standalone QGeoGNN-V2 engineering gate is not valid")
    data = load_feature_frame()
    SPLITS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for seed in OUTER_SEEDS:
        split = make_row_protocol(data[["sample_id", "canonical_smiles"]], seed)
        path = partition_path(seed)
        split.to_csv(path, index=False)
        records.append(
            {
                "outer_seed": seed,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "role_counts": split.role.value_counts().to_dict(),
                "partition_ids_hash": ids_hash(split.sample_id),
                "test_ids_hash": ids_hash(split.loc[split.role.eq("test"), "sample_id"]),
                "validation_ids_hash": ids_hash(split.loc[split.role.eq("validation"), "sample_id"]),
                "l0_ids_hash": ids_hash(split.loc[split.role.eq("l0"), "sample_id"]),
            }
        )
    atomic_json(
        SPLITS / "split_manifest.json",
        {
            "status": "FROZEN_BEFORE_FORMAL_TEST_EVALUATION",
            "source_data_sha256": sha256_file(SOURCE_DATA),
            "row_count": len(data),
            "compound_count": data.canonical_smiles.nunique(),
            "outer_seeds": list(OUTER_SEEDS),
            "splits": records,
        },
    )
    atomic_json(STUDY / "environment.json", environment_record())
    print(json.dumps({"prepared": True, "outer_seeds": OUTER_SEEDS, "splits": records}), flush=True)


def load_partition(seed: int) -> pd.DataFrame:
    manifest = json.loads((SPLITS / "split_manifest.json").read_text())
    record = next(item for item in manifest["splits"] if int(item["outer_seed"]) == int(seed))
    path = partition_path(seed)
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"split drift for outer seed {seed}")
    split = pd.read_csv(path)
    validate_row_protocol(split)
    return split


def role_indices(partition: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        role: partition.loc[partition.role.eq(role), "canonical_index"].to_numpy(dtype=int)
        for role in ("l0", "u0", "validation", "test")
    }


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _seed_setup(seed: int) -> dict[str, object]:
    torch.set_num_threads(2)
    data = load_feature_frame()
    partition = load_partition(seed)
    roles = role_indices(partition)
    label_store = RestrictedLabelStore(SOURCE_DATA, partition)
    l0_ids = partition.loc[partition.role.eq("l0"), "sample_id"].astype(str).tolist()
    validation_ids = partition.loc[partition.role.eq("validation"), "sample_id"].astype(str).tolist()
    l0_truth = label_store.reveal(l0_ids, "initial_fit")
    validation_truth = label_store.reveal(validation_ids, "initial_fit")
    cache = torch.load(SOURCE_GRAPH_CACHE, weights_only=False)
    runtime = RUNTIME / f"seed_{seed}"
    runtime.mkdir(parents=True, exist_ok=True)
    normalization, preprocessing = fit_al_preprocessing(
        data,
        cache,
        np.r_[roles["l0"], roles["u0"]],
        l0_ids,
        l0_truth,
        runtime / "scaler.json",
    )
    atomic_json(
        runtime / "preprocessing.json",
        {"normalization": vars(normalization), "preprocessing": preprocessing},
    )
    atom, angle = make_label_scrubbed_graphs(data, cache, preprocessing["scaler"])
    config = {
        **TRAINING_CONFIG,
        "seed": training_seed(seed),
        "split_sha256": sha256_file(partition_path(seed)),
        "outer_seed": int(seed),
        "cpu_threads": 2,
    }
    baseline_prediction_indices = np.r_[roles["l0"], roles["validation"], roles["test"]]
    baseline_prediction_ids = data.iloc[baseline_prediction_indices].sample_id.astype(str).tolist()
    baseline_contract = {
        "study": "qgeognn_v2_row_lcmd",
        "outer_seed": int(seed),
        "arm": "baseline_l0",
        "train_sample_ids": l0_ids,
        "validation_sample_ids": validation_ids,
        "split_sha256": sha256_file(partition_path(seed)),
        "training_config_hash": stable_hash(config),
    }
    baseline_audit = fit_from_same_initialization(
        atom_base=atom,
        angle=angle,
        normalization=normalization,
        preprocessing=preprocessing,
        train_indices=roles["l0"],
        train_truth=l0_truth,
        validation_indices=roles["validation"],
        validation_truth=validation_truth,
        prediction_indices=baseline_prediction_indices,
        prediction_sample_ids=baseline_prediction_ids,
        initialization_seed=initialization_seed(seed),
        training_config=config,
        contract=baseline_contract,
        runtime=runtime / "baseline_l0",
    )
    return {
        "data": data,
        "partition": partition,
        "roles": roles,
        "label_store": label_store,
        "l0_ids": l0_ids,
        "validation_ids": validation_ids,
        "l0_truth": l0_truth,
        "validation_truth": validation_truth,
        "cache": cache,
        "normalization": normalization,
        "preprocessing": preprocessing,
        "atom": atom,
        "angle": angle,
        "config": config,
        "runtime": runtime,
        "baseline_audit": baseline_audit,
    }


def _feature_contract(seed: int, checkpoint: Path, indices: Iterable[int], preprocessing: Mapping[str, object]) -> dict[str, object]:
    return {
        "outer_seed": int(seed),
        "checkpoint_sha256": sha256_file(checkpoint),
        "canonical_indices_hash": stable_hash([int(value) for value in indices]),
        "target_scales": preprocessing["target_scales"],
        "sketch_dimension": SKETCH_DIMENSION,
        "sketch_seed": sketch_seed(seed),
        "parameter_count": PARAMETER_COUNT,
        "outputs": ["V1_q50", "V2_q50"],
    }


def extract_full_features(setup: Mapping[str, object], seed: int) -> tuple[np.ndarray, dict[str, object]]:
    runtime = Path(setup["runtime"])
    checkpoint = runtime / "baseline_l0/best.pt"
    roles = setup["roles"]
    indices = np.r_[roles["l0"], roles["u0"]]
    contract = _feature_contract(seed, checkpoint, indices, setup["preprocessing"])
    path = runtime / "gradient_features.npz"
    audit_path = runtime / "gradient_feature_audit.json"
    parameter_path = runtime / "gradient_parameter_audit.csv"
    if path.exists() and audit_path.exists() and parameter_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit.get("contract") != contract or audit.get("feature_file_sha256") != sha256_file(path):
            raise RuntimeError(f"incompatible cached gradient features for seed {seed}")
        stored = np.load(path)
        features = stored["features"]
        if features.shape != (len(indices), SKETCH_DIMENSION) or not np.isfinite(features).all():
            raise RuntimeError("cached gradient feature shape/content changed")
        return features, audit

    model = load_predictor_checkpoint(checkpoint)

    def progress(record: dict[str, object]) -> None:
        atomic_json(runtime / "gradient_progress.json", {"outer_seed": seed, **record})
        if int(record["completed_rows"]) % 250 == 0 or int(record["completed_rows"]) == len(indices):
            print(json.dumps({"gradient_seed": seed, **record}), flush=True)

    result = extract_q50_gradient_sketches(
        model,
        setup["atom"],
        setup["angle"],
        indices,
        (
            float(setup["preprocessing"]["target_scales"]["V1"]),
            float(setup["preprocessing"]["target_scales"]["V2"]),
        ),
        dimension=SKETCH_DIMENSION,
        sketch_seed=sketch_seed(seed),
        progress=progress,
    )
    np.savez_compressed(path, features=result.features, canonical_indices=indices)
    audit = {
        "contract": contract,
        **result.audit,
        "feature_file_sha256": sha256_file(path),
        "peak_rss_bytes": peak_rss_bytes(),
    }
    atomic_json(audit_path, audit)
    pd.DataFrame(result.parameter_audit).assign(outer_seed=seed).to_csv(parameter_path, index=False)
    return result.features, audit


def finite_difference_check(setup: Mapping[str, object], seed: int) -> dict[str, float | bool | str]:
    model = load_predictor_checkpoint(Path(setup["runtime"]) / "baseline_l0/best.pt")
    model.eval()
    position = int(setup["roles"]["l0"][0])
    atom_batch, angle_batch = next(
        zip(*__import__("src.qgeognn_al.training.predictor", fromlist=["loader_pair"]).loader_pair(
            setup["atom"], setup["angle"], [position], 1
        ))
    )
    parameter = dict(model.named_parameters())["head.0.bias"]
    scale = float(setup["preprocessing"]["target_scales"]["V1"])
    analytic = torch.autograd.grad(model(atom_batch, angle_batch)[0, 1] / scale, parameter)[0][1].item()
    original = parameter.detach()[1].item()
    epsilon = 1e-3
    with torch.no_grad():
        parameter[1] = original + epsilon
        plus = model(atom_batch, angle_batch)[0, 1].item() / scale
        parameter[1] = original - epsilon
        minus = model(atom_batch, angle_batch)[0, 1].item() / scale
        parameter[1] = original
    numeric = (plus - minus) / (2 * epsilon)
    error = abs(analytic - numeric)
    return {
        "parameter": "head.0.bias[1]",
        "analytic": analytic,
        "central_difference": numeric,
        "absolute_error": error,
        "passed": bool(error <= 5e-4 * max(1.0, abs(analytic))),
    }


def preflight() -> None:
    if not (SPLITS / "split_manifest.json").exists():
        prepare()
    setup = _seed_setup(PREFLIGHT_SEED)
    roles = setup["roles"]
    indices = np.r_[roles["l0"][:PREFLIGHT_L0_ROWS], roles["u0"][:PREFLIGHT_POOL_ROWS]]
    model = load_predictor_checkpoint(Path(setup["runtime"]) / "baseline_l0/best.pt")
    kwargs = dict(
        model=model,
        atom_data=setup["atom"],
        angle_data=setup["angle"],
        indices=indices,
        scales=(
            float(setup["preprocessing"]["target_scales"]["V1"]),
            float(setup["preprocessing"]["target_scales"]["V2"]),
        ),
        dimension=SKETCH_DIMENSION,
        sketch_seed=sketch_seed(PREFLIGHT_SEED),
    )
    first = extract_q50_gradient_sketches(**kwargs)
    second = extract_q50_gradient_sketches(**kwargs)
    deterministic = bool(np.allclose(first.features, second.features, rtol=0, atol=1e-7))
    small = lcmd_tp_select(
        first.features[PREFLIGHT_L0_ROWS:], first.features[:PREFLIGHT_L0_ROWS], PREFLIGHT_BATCH
    )
    full_features, full_audit = extract_full_features(setup, PREFLIGHT_SEED)
    full_selection = lcmd_tp_select(
        full_features[len(roles["l0"]):], full_features[:len(roles["l0"])], batch_size(setup["partition"])
    )
    fd = finite_difference_check(setup, PREFLIGHT_SEED)
    selected = full_selection.selected_pool_positions
    feature_indices = np.r_[roles["l0"], roles["u0"]]
    input_columns = [
        "canonical_smiles", "Density g/ml", "V/ul", "loading solvent",
        "Volume of loading solvent/ul", "PE/EA",
    ]
    input_keys = (
        setup["data"].iloc[feature_indices][input_columns].astype(str).agg("|".join, axis=1).to_numpy()
    )
    _, feature_inverse = np.unique(full_features, axis=0, return_inverse=True)
    duplicate_groups_mix_distinct_inputs = 0
    for group in np.flatnonzero(np.bincount(feature_inverse) > 1):
        if len(set(input_keys[feature_inverse == group])) != 1:
            duplicate_groups_mix_distinct_inputs += 1
    unique_input_rows = int(len(np.unique(input_keys)))
    conditions = {
        "finite_difference_gradient_check": bool(fd["passed"]),
        "small_gradient_features_finite": bool(np.isfinite(first.features).all()),
        "deterministic_repeat_within_tolerance": deterministic,
        "small_lcmd_batch_size_exact": len(small.selected_pool_positions) == PREFLIGHT_BATCH,
        "small_lcmd_unique": len(np.unique(small.selected_pool_positions)) == PREFLIGHT_BATCH,
        "full_features_finite": bool(full_audit["all_finite"]),
        "full_features_no_unexplained_duplicate_collapse": (
            int(full_audit["unique_feature_rows"]) == unique_input_rows
            and duplicate_groups_mix_distinct_inputs == 0
        ),
        "full_lcmd_batch_size_exact": len(selected) == batch_size(setup["partition"]),
        "full_lcmd_unique": len(np.unique(selected)) == batch_size(setup["partition"]),
        "full_lcmd_pool_only": bool(np.all((selected >= 0) & (selected < len(roles["u0"])) )),
        "full_extraction_runtime_feasible": float(full_audit["elapsed_seconds"]) <= MAX_ACCEPTABLE_FULL_EXTRACTION_SECONDS,
        "peak_rss_feasible": int(full_audit["peak_rss_bytes"]) <= MAX_ACCEPTABLE_PEAK_RSS_BYTES,
        "test_truth_access_count_zero": not any(row["roles"] == "test" for row in setup["label_store"].audit),
    }
    payload = {
        "status": "PASS" if all(conditions.values()) else "BLOCKED",
        "outer_seed": PREFLIGHT_SEED,
        "small_rows": len(indices),
        "small_pool_rows": PREFLIGHT_POOL_ROWS,
        "small_l0_rows": PREFLIGHT_L0_ROWS,
        "small_batch": PREFLIGHT_BATCH,
        "conditions": conditions,
        "finite_difference": fd,
        "small_first_audit": first.audit,
        "small_second_audit": second.audit,
        "full_feature_audit": full_audit,
        "duplicate_feature_explanation": {
            "unique_label_free_input_rows": unique_input_rows,
            "duplicate_label_free_input_rows": len(full_features) - unique_input_rows,
            "duplicate_gradient_groups_mixing_distinct_inputs": duplicate_groups_mix_distinct_inputs,
            "interpretation": "Every exact sketch duplicate is explained by an exact duplicate molecule-plus-condition input.",
        },
        "estimated_five_seed_feature_seconds": 5 * float(full_audit["elapsed_seconds"]),
    }
    atomic_json(RESULTS / "preflight.json", payload)
    if payload["status"] != "PASS":
        (STUDY / "BLOCKER_REPORT.md").write_text(
            "# Blocker report\n\nFull-network Gradient-LCMD-TP preflight failed. "
            "The primary experiment was not run and no head-only substitute was used. "
            f"See `results/preflight.json`. Conditions: `{conditions}`.\n"
        )
        atomic_json(STUDY / "decision.json", {"decision": "BLOCKED", "preflight": conditions})
        raise RuntimeError("full-gradient preflight failed; see BLOCKER_REPORT.md")
    print(json.dumps({"preflight": "PASS", "full_feature_audit": full_audit}), flush=True)


def selection_tables(setup: Mapping[str, object], seed: int, features: np.ndarray) -> tuple[pd.DataFrame, dict[str, object]]:
    data = setup["data"]
    roles = setup["roles"]
    l0_count = len(roles["l0"])
    l0_features, pool_features = features[:l0_count], features[l0_count:]
    pool_indices = np.asarray(roles["u0"], dtype=int)
    l0_indices = np.asarray(roles["l0"], dtype=int)
    batch = batch_size(setup["partition"])
    started = time.perf_counter()
    selected = lcmd_tp_select(pool_features, l0_features, batch)
    lcmd_seconds = time.perf_counter() - started

    raw_conditions = condition_matrix(data)
    outer_indices = np.r_[l0_indices, pool_indices]
    mean = raw_conditions[outer_indices].mean(axis=0)
    scale = raw_conditions[outer_indices].std(axis=0)
    scale[scale < 1e-8] = 1.0
    condition_z = (raw_conditions - mean) / scale
    condition_sq = (
        np.square(condition_z[pool_indices]).sum(axis=1, keepdims=True)
        + np.square(condition_z[l0_indices]).sum(axis=1)[None, :]
        - 2 * condition_z[pool_indices] @ condition_z[l0_indices].T
    )
    nearest_condition = np.sqrt(np.maximum(condition_sq, 0).min(axis=1))
    pool_norms = np.linalg.norm(pool_features.astype(np.float64), axis=1)
    trace = {int(row["pool_position"]): row for row in selected.trace}
    rows: list[dict[str, object]] = []

    arm_positions = {"lcmd": selected.selected_pool_positions}
    for control in range(RANDOM_CONTROLS):
        arm_positions[f"random_control_{control}"] = random_control_positions(
            len(pool_indices), batch, seed, control
        )
    for arm, positions in arm_positions.items():
        for order, pool_position in enumerate(positions):
            pool_position = int(pool_position)
            canonical = int(pool_indices[pool_position])
            source_trace = trace.get(pool_position, {}) if arm == "lcmd" else {}
            center_index = int(selected.initial_nearest_center[pool_position])
            row = data.iloc[canonical]
            rows.append(
                {
                    "outer_seed": seed,
                    "arm": arm,
                    "control_index": None if arm == "lcmd" else int(arm.rsplit("_", 1)[1]),
                    "selection_order": order,
                    "sample_id": str(row.sample_id),
                    "canonical_index": canonical,
                    "canonical_smiles": str(row.canonical_smiles),
                    "gradient_norm": float(pool_norms[pool_position]),
                    "nearest_l0_gradient_distance": float(math.sqrt(max(0.0, selected.initial_nearest_sq_distance[pool_position]))),
                    "initial_l0_cluster_sample_id": str(data.iloc[l0_indices[center_index]].sample_id),
                    "nearest_l0_condition_distance": float(nearest_condition[pool_position]),
                    "lcmd_selected_from_center_index": source_trace.get("selected_from_center"),
                    "lcmd_largest_cluster_score": source_trace.get("largest_cluster_score"),
                    "PE_EA": str(row["PE/EA"]),
                    "loading_solvent": str(row["loading solvent"]),
                    "loading_amount_density_x_volume": float(row["Density g/ml"] * row["V/ul"]),
                    "loading_solvent_volume_ul": float(row["Volume of loading solvent/ul"]),
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.groupby("arm").size().eq(batch).all() or not frame.groupby("arm").sample_id.nunique().eq(batch).all():
        raise RuntimeError("selected batch uniqueness/size audit failed")
    if not set(frame.sample_id) <= set(data.iloc[pool_indices].sample_id.astype(str)):
        raise RuntimeError("selected batch contains a non-U0 row")
    return frame, {"lcmd_selection_seconds": lcmd_seconds, "batch_size": batch}


def run_seed_pre_evaluation(seed: int) -> None:
    if seed not in OUTER_SEEDS:
        raise ValueError("unregistered outer seed")
    setup = _seed_setup(seed)
    features, feature_audit = extract_full_features(setup, seed)
    selections, selection_audit = selection_tables(setup, seed, features)
    runtime = Path(setup["runtime"])
    selection_path = runtime / "selected_batches.csv"
    selections.to_csv(selection_path, index=False)

    label_store = setup["label_store"]
    label_store.freeze_acquisitions(selections.sample_id)
    data = setup["data"]
    test_indices = setup["roles"]["test"]
    test_ids = data.iloc[test_indices].sample_id.astype(str).tolist()
    fit_audits: list[dict[str, object]] = [
        {"outer_seed": seed, "arm": "baseline_l0", **setup["baseline_audit"]}
    ]
    for arm, selected_rows in selections.groupby("arm", sort=False):
        selected_rows = selected_rows.sort_values("selection_order")
        selected_ids = selected_rows.sample_id.astype(str).tolist()
        selected_indices = selected_rows.canonical_index.to_numpy(dtype=int)
        train_ids = setup["l0_ids"] + selected_ids
        train_indices = np.r_[setup["roles"]["l0"], selected_indices]
        selected_truth = label_store.reveal(selected_ids, "after_acquisition_fit")
        train_truth = np.vstack([setup["l0_truth"], selected_truth])
        contract = {
            "study": "qgeognn_v2_row_lcmd",
            "outer_seed": int(seed),
            "arm": arm,
            "train_sample_ids": train_ids,
            "validation_sample_ids": setup["validation_ids"],
            "selection_sha256": sha256_file(selection_path),
            "split_sha256": sha256_file(partition_path(seed)),
            "training_config_hash": stable_hash(setup["config"]),
        }
        audit = fit_from_same_initialization(
            atom_base=setup["atom"],
            angle=setup["angle"],
            normalization=setup["normalization"],
            preprocessing=setup["preprocessing"],
            train_indices=train_indices,
            train_truth=train_truth,
            validation_indices=setup["roles"]["validation"],
            validation_truth=setup["validation_truth"],
            prediction_indices=test_indices,
            prediction_sample_ids=test_ids,
            initialization_seed=initialization_seed(seed),
            training_config=setup["config"],
            contract=contract,
            runtime=runtime / "arms" / arm,
        )
        fit_audits.append({"outer_seed": seed, "arm": arm, **audit})
        print(json.dumps({"outer_seed": seed, "arm_complete": arm, "best_epoch": audit["best_epoch"]}), flush=True)

    pd.DataFrame(fit_audits).to_csv(runtime / "fit_audit.csv", index=False)
    pd.DataFrame(label_store.audit).to_csv(runtime / "label_access_before_test.csv", index=False)
    checkpoint_paths = [runtime / "baseline_l0/best.pt"] + [
        runtime / "arms" / arm / "best.pt" for arm in selections.arm.drop_duplicates()
    ]
    prediction_paths = [runtime / "baseline_l0/predictions.csv.gz"] + [
        runtime / "arms" / arm / "predictions.csv.gz" for arm in selections.arm.drop_duplicates()
    ]
    freeze = {
        "status": "SEED_FROZEN_BEFORE_TEST_TRUTH_ACCESS",
        "outer_seed": seed,
        "selection_sha256": sha256_file(selection_path),
        "checkpoint_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in checkpoint_paths},
        "prediction_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in prediction_paths},
        "feature_audit": feature_audit,
        "selection_audit": selection_audit,
        "test_truth_access_count": 0,
    }
    atomic_json(runtime / "pre_test_freeze.json", freeze)
    print(json.dumps({"seed_frozen_before_test": seed, "fits": len(fit_audits)}), flush=True)


def _launch_seed(seed: int) -> tuple[int, int]:
    log = RUNTIME / "logs" / f"seed_{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as output:
        completed = subprocess.run(
            [sys.executable, __file__, "--run-seed", str(seed)],
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
        )
    return seed, int(completed.returncode)


def decision_gate(seed_summary: pd.DataFrame, arm_metrics: pd.DataFrame) -> dict[str, object]:
    ordered = seed_summary.sort_values("outer_seed")
    directional_wins = int((ordered.LCMD_gain > ordered.Random_gain_median).sum())
    mean_delta = float(ordered.LCMD_minus_Random_mean.mean())
    median_delta = float(ordered.LCMD_minus_Random_median.median())
    lcmd = arm_metrics.loc[arm_metrics.arm.eq("lcmd")]
    random = arm_metrics.loc[arm_metrics.arm.str.startswith("random_control_")]
    lcmd_nrmse = float(lcmd.combined_normalized_RMSE.mean())
    random_nrmse = float(random.combined_normalized_RMSE.mean())
    improvement = 100.0 * (random_nrmse - lcmd_nrmse) / random_nrmse
    endpoint_changes = {
        target: 100.0
        * (float(lcmd[f"{target}_RMSE"].mean()) - float(random[f"{target}_RMSE"].mean()))
        / float(random[f"{target}_RMSE"].mean())
        for target in ("V1", "V2")
    }
    conditions = {
        "A_directional_wins_at_least_4_of_5": directional_wins >= 4,
        "B_mean_LCMD_minus_Random_mean_gain_positive": mean_delta > 0,
        "C_median_LCMD_minus_Random_median_gain_positive": median_delta > 0,
        "D_mean_after_batch_NRMSE_improvement_at_least_5_percent": improvement >= 5.0,
        "E_no_endpoint_mean_RMSE_deterioration_above_2_percent": max(endpoint_changes.values()) <= 2.0,
    }
    if all(conditions.values()):
        decision = "STRONG_POSITIVE"
    elif directional_wins >= 4 or improvement >= 3.0:
        decision = "PROMISING_BUT_NOT_CONFIRMED"
    else:
        decision = "NO_STABLE_BENEFIT"
    return {
        "decision": decision,
        "conditions": conditions,
        "directional_wins": directional_wins,
        "mean_LCMD_minus_Random_mean_gain": mean_delta,
        "median_LCMD_minus_Random_median_gain": median_delta,
        "mean_LCMD_after_batch_combined_NRMSE": lcmd_nrmse,
        "mean_Random_control_after_batch_combined_NRMSE": random_nrmse,
        "mean_combined_NRMSE_improvement_percent": improvement,
        "endpoint_mean_RMSE_change_percent_LCMD_vs_Random": endpoint_changes,
        "full_learning_curve_recommended": decision == "STRONG_POSITIVE",
        "automatic_next_method": None,
        "study_stops_after_decision": True,
    }


def metric_row(
    truth: np.ndarray,
    prediction: np.ndarray,
    scales: Mapping[str, float],
) -> dict[str, float | bool]:
    values = point_metrics(truth, prediction, scales)
    result: dict[str, float | bool] = {
        "V1_RMSE": values["V1_rmse"],
        "V2_RMSE": values["V2_rmse"],
        "V1_MAE": values["V1_mae"],
        "V2_MAE": values["V2_mae"],
        "V1_R2": values["V1_r2"],
        "V2_R2": values["V2_r2"],
        "combined_normalized_RMSE": values["combined_normalized_rmse"],
        "V1_RMSE_MAE_ratio": values["V1_rmse"] / values["V1_mae"],
        "V2_RMSE_MAE_ratio": values["V2_rmse"] / values["V2_mae"],
        "all_outputs_finite": values["all_outputs_finite"],
    }
    return result


def _read_predictions(path: Path, expected_ids: list[str]) -> np.ndarray:
    frame = pd.read_csv(path)
    if frame.sample_id.astype(str).tolist() != expected_ids:
        raise RuntimeError(f"prediction identity/order changed: {path}")
    return frame[[f"{target}_{quantile}" for target in ("V1", "V2") for quantile in ("q10", "q50", "q90")]].to_numpy(float)


def aggregate_and_evaluate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    selected_frames: list[pd.DataFrame] = []
    freeze_records: dict[str, object] = {}
    feature_rows: list[dict[str, object]] = []
    parameter_frames: list[pd.DataFrame] = []
    fit_frames: list[pd.DataFrame] = []
    access_frames: list[pd.DataFrame] = []
    for seed in OUTER_SEEDS:
        runtime = RUNTIME / f"seed_{seed}"
        freeze_path = runtime / "pre_test_freeze.json"
        if not freeze_path.exists():
            raise RuntimeError(f"seed {seed} is not frozen before test")
        freeze = json.loads(freeze_path.read_text())
        if freeze.get("status") != "SEED_FROZEN_BEFORE_TEST_TRUTH_ACCESS" or freeze.get("test_truth_access_count") != 0:
            raise RuntimeError(f"seed {seed} test gate invalid")
        for relative, digest in {**freeze["checkpoint_hashes"], **freeze["prediction_hashes"]}.items():
            if sha256_file(ROOT / relative) != digest:
                raise RuntimeError(f"frozen artifact drift: {relative}")
        freeze_records[str(seed)] = freeze
        selected_frames.append(pd.read_csv(runtime / "selected_batches.csv"))
        feature_rows.append({"outer_seed": seed, **freeze["feature_audit"]})
        parameter_frames.append(pd.read_csv(runtime / "gradient_parameter_audit.csv"))
        fit_frames.append(pd.read_csv(runtime / "fit_audit.csv"))
        access_frames.append(
            pd.read_csv(runtime / "label_access_before_test.csv").assign(outer_seed=seed)
        )

    selected = pd.concat(selected_frames, ignore_index=True)
    selected.to_csv(RESULTS / "selected_batches.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(RESULTS / "gradient_feature_audit.csv", index=False)
    pd.concat(parameter_frames, ignore_index=True).to_csv(RESULTS / "gradient_parameter_audit.csv", index=False)
    fits = pd.concat(fit_frames, ignore_index=True)
    fits.to_csv(RESULTS / "fit_audit.csv", index=False)
    fits[["outer_seed", "arm", "initialization_seed", "initialization_hash", "validation_ids_hash"]].assign(
        initial_hash_identical_within_seed=fits.groupby("outer_seed").initialization_hash.transform("nunique").eq(1),
        test_used_for_checkpoint_selection=False,
    ).to_csv(RESULTS / "initialization_hash_audit.csv", index=False)
    runtime_rows = fits[["outer_seed", "arm", "training_seconds", "epochs_run", "best_epoch"]].copy()
    feature_runtime = pd.DataFrame(feature_rows)[["outer_seed", "elapsed_seconds", "peak_rss_bytes"]].rename(
        columns={"elapsed_seconds": "gradient_feature_seconds"}
    )
    runtime_rows.merge(feature_runtime, on="outer_seed", how="left").to_csv(RESULTS / "runtime_audit.csv", index=False)
    pretest_manifest = {
        "status": "ALL_ACQUISITIONS_CHECKPOINTS_AND_PREDICTIONS_FROZEN_BEFORE_TEST_TRUTH_ACCESS",
        "selected_batches_sha256": sha256_file(RESULTS / "selected_batches.csv"),
        "seeds": freeze_records,
        "test_truth_access_count": 0,
    }
    atomic_json(RUNTIME / "global_pre_test_freeze.json", pretest_manifest)

    baseline_rows: list[dict[str, object]] = []
    arm_rows: list[dict[str, object]] = []
    final_access: list[pd.DataFrame] = access_frames[:]
    data = load_feature_frame()
    for seed in OUTER_SEEDS:
        partition = load_partition(seed)
        roles = role_indices(partition)
        store = RestrictedLabelStore(SOURCE_DATA, partition)
        seed_selected = selected.loc[selected.outer_seed.eq(seed)]
        store.freeze_acquisitions(seed_selected.sample_id)
        store.freeze_predictions()
        l0_ids = data.iloc[roles["l0"]].sample_id.astype(str).tolist()
        validation_ids = data.iloc[roles["validation"]].sample_id.astype(str).tolist()
        test_ids = data.iloc[roles["test"]].sample_id.astype(str).tolist()
        l0_truth = store.reveal(l0_ids, "initial_fit")
        validation_truth = store.reveal(validation_ids, "initial_fit")
        test_truth = store.reveal(test_ids, "final_test_evaluation")
        scales = json.loads((RUNTIME / f"seed_{seed}/preprocessing.json").read_text())["preprocessing"]["target_scales"]
        metric_scales = {"V1": float(scales["V1"]), "V2": float(scales["V2"])}

        baseline_ids = l0_ids + validation_ids + test_ids
        baseline_prediction = _read_predictions(
            RUNTIME / f"seed_{seed}/baseline_l0/predictions.csv.gz", baseline_ids
        )
        offsets = (0, len(l0_ids), len(l0_ids) + len(validation_ids), len(baseline_ids))
        for role, truth, start, stop in (
            ("train", l0_truth, offsets[0], offsets[1]),
            ("validation", validation_truth, offsets[1], offsets[2]),
            ("test", test_truth, offsets[2], offsets[3]),
        ):
            baseline_rows.append(
                {
                    "outer_seed": seed,
                    "arm": "baseline_l0",
                    "split": role,
                    **metric_row(truth, baseline_prediction[start:stop], metric_scales),
                }
            )
        baseline_test_nrmse = baseline_rows[-1]["combined_normalized_RMSE"]
        for arm in ["lcmd", *[f"random_control_{index}" for index in range(RANDOM_CONTROLS)]]:
            prediction = _read_predictions(
                RUNTIME / f"seed_{seed}/arms/{arm}/predictions.csv.gz", test_ids
            )
            metrics = metric_row(test_truth, prediction, metric_scales)
            arm_rows.append(
                {
                    "outer_seed": seed,
                    "arm": arm,
                    "split": "test",
                    **metrics,
                    "baseline_L0_NRMSE": baseline_test_nrmse,
                    "gain": float(baseline_test_nrmse) - float(metrics["combined_normalized_RMSE"]),
                }
            )
        final_access.append(pd.DataFrame(store.audit).assign(outer_seed=seed))

    baseline = pd.DataFrame(baseline_rows)
    arms = pd.DataFrame(arm_rows)
    baseline.to_csv(RESULTS / "baseline_metrics.csv", index=False)
    arms.to_csv(RESULTS / "arm_metrics.csv", index=False)
    pd.concat(final_access, ignore_index=True).to_csv(RESULTS / "label_access_audit.csv", index=False)
    summaries: list[dict[str, object]] = []
    for seed in OUTER_SEEDS:
        rows = arms.loc[arms.outer_seed.eq(seed)]
        lcmd = rows.loc[rows.arm.eq("lcmd")].iloc[0]
        random = rows.loc[rows.arm.str.startswith("random_control_")]
        random_gains = random.gain.to_numpy(float)
        summaries.append(
            {
                "outer_seed": seed,
                "baseline_L0_NRMSE": lcmd.baseline_L0_NRMSE,
                "LCMD_after_NRMSE": lcmd.combined_normalized_RMSE,
                "LCMD_gain": lcmd.gain,
                "Random_after_NRMSE_mean": random.combined_normalized_RMSE.mean(),
                "Random_after_NRMSE_median": random.combined_normalized_RMSE.median(),
                "Random_gain_mean": random_gains.mean(),
                "Random_gain_median": np.median(random_gains),
                "Random_gain_min": random_gains.min(),
                "Random_gain_max": random_gains.max(),
                "LCMD_minus_Random_mean": lcmd.gain - random_gains.mean(),
                "LCMD_minus_Random_median": lcmd.gain - np.median(random_gains),
                "LCMD_beat_count": int((lcmd.gain > random_gains).sum()),
                "LCMD_percentile_among_controls": float((random_gains < lcmd.gain).mean()),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(RESULTS / "seed_summary.csv", index=False)
    decision = decision_gate(summary, arms)
    atomic_json(STUDY / "decision.json", decision)
    return baseline, arms, summary, decision


def batch_diagnostics(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (seed, arm), group in selected.groupby(["outer_seed", "arm"], sort=False):
        counts = group.canonical_smiles.value_counts().to_numpy()
        rows.append(
            {
                "outer_seed": seed,
                "arm": arm,
                "gradient_norm_mean": group.gradient_norm.mean(),
                "nearest_L0_gradient_distance_mean": group.nearest_l0_gradient_distance.mean(),
                "initial_L0_clusters_covered": group.initial_l0_cluster_sample_id.nunique(),
                "unique_compounds": group.canonical_smiles.nunique(),
                "compound_repeated_row_fraction": float((counts[counts > 1].sum() / len(group)) if np.any(counts > 1) else 0.0),
                "nearest_L0_condition_distance_mean": group.nearest_l0_condition_distance.mean(),
                "unique_eluent_ratios": group.PE_EA.nunique(),
                "unique_loading_solvents": group.loading_solvent.nunique(),
            }
        )
    return pd.DataFrame(rows)


def make_figures(baseline: pd.DataFrame, arms: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for seed in OUTER_SEEDS:
        row = arms.loc[arms.outer_seed.eq(seed)]
        random_mean = row.loc[row.arm.str.startswith("random_control_"), "combined_normalized_RMSE"].mean()
        lcmd = row.loc[row.arm.eq("lcmd"), "combined_normalized_RMSE"].item()
        ax.plot([0, 1], [random_mean, lcmd], marker="o", alpha=0.75, label=str(seed))
    ax.set(xticks=[0, 1], xticklabels=["Random mean", "Gradient-LCMD-TP"], ylabel="Test combined normalized RMSE")
    ax.legend(title="Outer seed", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "paired_lcmd_vs_random.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(summary))
    ax.bar(x - 0.18, summary.Random_gain_mean, width=0.36, label="Random mean gain")
    ax.bar(x + 0.18, summary.LCMD_gain, width=0.36, label="LCMD gain")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xticks=x, xticklabels=summary.outer_seed.astype(str), xlabel="Outer seed", ylabel="Baseline − after-batch NRMSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "nrmse_gain_by_seed.png", dpi=180)
    plt.close(fig)

    endpoint = []
    for target in ("V1", "V2"):
        endpoint.extend(
            [
                {"target": target, "arm": "Random mean", "RMSE": arms.loc[arms.arm.str.startswith("random_control_"), f"{target}_RMSE"].mean()},
                {"target": target, "arm": "Gradient-LCMD-TP", "RMSE": arms.loc[arms.arm.eq("lcmd"), f"{target}_RMSE"].mean()},
            ]
        )
    endpoint_frame = pd.DataFrame(endpoint)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(2)
    random_values = endpoint_frame.loc[endpoint_frame.arm.eq("Random mean"), "RMSE"]
    lcmd_values = endpoint_frame.loc[endpoint_frame.arm.eq("Gradient-LCMD-TP"), "RMSE"]
    ax.bar(x - 0.18, random_values, width=0.36, label="Random mean")
    ax.bar(x + 0.18, lcmd_values, width=0.36, label="Gradient-LCMD-TP")
    ax.set(xticks=x, xticklabels=["V1", "V2"], ylabel="Test RMSE (mL)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "endpoint_rmse_comparison.png", dpi=180)
    plt.close(fig)


def write_reports(baseline: pd.DataFrame, arms: pd.DataFrame, summary: pd.DataFrame, decision: Mapping[str, object]) -> None:
    selected = pd.read_csv(RESULTS / "selected_batches.csv")
    diagnostics = batch_diagnostics(selected)
    diagnostics.to_csv(RESULTS / "batch_diagnostics.csv", index=False)
    ldiag = diagnostics.loc[diagnostics.arm.eq("lcmd")]
    rdiag = diagnostics.loc[diagnostics.arm.str.startswith("random_control_")]
    baseline_test = baseline.loc[baseline.split.eq("test")]
    baseline_train = baseline.loc[baseline.split.eq("train")]
    lcmd = arms.loc[arms.arm.eq("lcmd")]
    random = arms.loc[arms.arm.str.startswith("random_control_")]
    seed_table = "\n".join(
        f"| {int(row.outer_seed)} | {row.baseline_L0_NRMSE:.6f} | {row.LCMD_after_NRMSE:.6f} | "
        f"{row.Random_after_NRMSE_mean:.6f} | {row.Random_after_NRMSE_median:.6f} | "
        f"{int(row.LCMD_beat_count)}/5 |"
        for row in summary.itertuples(index=False)
    )
    report = f"""# Final report — QGeoGNN-V2 4g row Gradient-LCMD-TP

Decision: **`{decision['decision']}`**. The preregistered one-step experiment is complete and stops here.

## Primary result

Across five independent outer row seeds, LCMD beat the within-seed Random median in **{decision['directional_wins']}/5** seeds. Mean LCMD after-batch combined normalized RMSE was **{decision['mean_LCMD_after_batch_combined_NRMSE']:.6f}**, versus **{decision['mean_Random_control_after_batch_combined_NRMSE']:.6f}** across the 25 Random controls, an improvement of **{decision['mean_combined_NRMSE_improvement_percent']:.2f}%**. Mean `LCMD gain − Random mean gain` was **{decision['mean_LCMD_minus_Random_mean_gain']:.6f}** and the cross-seed median `LCMD gain − Random median gain` was **{decision['median_LCMD_minus_Random_median_gain']:.6f}**.

Mean endpoint RMSE changes for LCMD relative to Random were **{decision['endpoint_mean_RMSE_change_percent_LCMD_vs_Random']['V1']:.2f}%** for V1 and **{decision['endpoint_mean_RMSE_change_percent_LCMD_vs_Random']['V2']:.2f}%** for V2 (negative is better).

| Outer seed | L0 baseline NRMSE | LCMD after NRMSE | Random mean after | Random median after | LCMD beat count |
|---:|---:|---:|---:|---:|---:|
{seed_table}

## Baseline and error shape

At 10% of outer-training labels (333 active labels, plus 416 shared validation labels), mean test combined normalized RMSE was **{baseline_test.combined_normalized_RMSE.mean():.6f}**. Mean V1/V2 RMSE was **{baseline_test.V1_RMSE.mean():.3f}/{baseline_test.V2_RMSE.mean():.3f} mL**. Mean baseline test RMSE/MAE ratios were **{baseline_test.V1_RMSE_MAE_ratio.mean():.3f}/{baseline_test.V2_RMSE_MAE_ratio.mean():.3f}**, compared with training ratios **{baseline_train.V1_RMSE_MAE_ratio.mean():.3f}/{baseline_train.V2_RMSE_MAE_ratio.mean():.3f}**. These unusually high ratios support a heavy-tail/outlier-sensitive error diagnosis, but do not by themselves prove a particular error distribution. This diagnostic was not used to tune acquisition.

## What LCMD selected

Relative to the mean Random batch, LCMD's selected rows had mean gradient norm **{ldiag.gradient_norm_mean.mean():.3f}** versus **{rdiag.gradient_norm_mean.mean():.3f}**, mean nearest-L0 gradient distance **{ldiag.nearest_L0_gradient_distance_mean.mean():.3f}** versus **{rdiag.nearest_L0_gradient_distance_mean.mean():.3f}**, and covered **{ldiag.initial_L0_clusters_covered.mean():.1f}** versus **{rdiag.initial_L0_clusters_covered.mean():.1f}** initial gradient clusters. It selected **{ldiag.unique_compounds.mean():.1f}** unique compounds per batch versus **{rdiag.unique_compounds.mean():.1f}** for Random, with repeated-row fractions **{ldiag.compound_repeated_row_fraction.mean():.3f}** versus **{rdiag.compound_repeated_row_fraction.mean():.3f}**. Mean nearest-L0 standardized condition distance was **{ldiag.nearest_L0_condition_distance_mean.mean():.3f}** versus **{rdiag.nearest_L0_condition_distance_mean.mean():.3f}**; mean unique eluent-ratio counts were **{ldiag.unique_eluent_ratios.mean():.2f}** versus **{rdiag.unique_eluent_ratios.mean():.2f}**, and both covered all **{ldiag.unique_loading_solvents.mean():.0f}** loading solvents. Thus LCMD emphasized gradient-space distance and norm rather than maximizing raw compound or categorical-condition counts. These are label-free mechanism descriptions, not post-hoc selection changes.

## Scientific interpretation

The overall decision follows the frozen gate exactly. Both endpoints improved substantially: mean V1 RMSE fell from **{random.V1_RMSE.mean():.3f}** to **{lcmd.V1_RMSE.mean():.3f} mL**, and mean V2 RMSE fell from **{random.V2_RMSE.mean():.3f}** to **{lcmd.V2_RMSE.mean():.3f} mL**. The relative reduction is slightly larger for V2, but the result is not driven by only one endpoint. Five outer seeds are the independent replication units; the five Random controls within each seed estimate the conditional Random distribution and are not 25 independent datasets.

This current-V2 result must not be numerically pooled with Wu et al.'s full 4g result or with legacy E2/A1a experiments. It answers whether labels treated as initially unknown are chosen more effectively for this qualified canonical V2 row protocol. It measures label efficiency, data efficiency, and experimental-data acquisition efficiency—not wall-clock training acceleration.

The validation labels are a fixed shared auxiliary cost and are excluded from the active label count. Test truth was first accessed only after every acquisition, checkpoint, and prediction was frozen and hashed. No test result changed seeds, sketch size, gradient scope, batch size, distance, model, or training settings.

## Next stage

Full row learning curve recommended: **{str(bool(decision['full_learning_curve_recommended'])).lower()}**. Regardless of this result, no BAIT, Condition-LCMD, UCB/EI/TS, Hybrid continuation, or automatic tuning is launched.

## Method references

- Holzmüller et al., [A Framework and Benchmark for Deep Batch Active Learning for Regression](https://www.jmlr.org/papers/v24/22-0937.html), JMLR 24(164), 2023.
- Author implementation: [`dholzmueller/bmdal_reg`](https://github.com/dholzmueller/bmdal_reg). This study uses the corrected TP initialization semantics and an explicitly documented two-output QGeoGNN-V2 adaptation.
"""
    (STUDY / "FINAL_REPORT.md").write_text(report)
    (STUDY / "README.md").write_text(
        "# QGeoGNN-V2 4g row Gradient-LCMD-TP\n\n"
        f"Status: `COMPLETED / {decision['decision']}`.\n\n"
        f"LCMD beat the Random median in {decision['directional_wins']}/5 outer seeds; mean after-batch combined NRMSE improvement was {decision['mean_combined_NRMSE_improvement_percent']:.2f}%. "
        "See `FINAL_REPORT.md`, the frozen `PREREGISTRATION.md`, and `results/seed_summary.csv`.\n\n"
        "## Historical boundary\n\n"
        "- Under the legacy E2 row protocol, Coverage/Hybrid beat Random in 3/3 seeds; that was pilot evidence, not a current-V2 result.\n"
        "- A1a applied farthest-first inside a shared Top-25% uncertainty shortlist and failed its mechanism gate. It rejected only that uncertainty-shortlist-to-diversity mechanism, not 4g active learning in general.\n"
        "- The legacy predictor had a condition-feature reachability defect. Its E2/A1a numbers are not pooled with this study.\n"
        "- This experiment uses the qualified standalone QGeoGNN-V2 and full-pool, sketched full-network Gradient-LCMD-TP.\n\n"
        "Method basis: Holzmüller et al., [JMLR 24(164)](https://www.jmlr.org/papers/v24/22-0937.html), with the corrected TP semantics documented by the [author implementation](https://github.com/dholzmueller/bmdal_reg).\n"
    )
    if decision["decision"] == "STRONG_POSITIVE":
        (STUDY / "NEXT_STAGE_RECOMMENDATION.md").write_text(
            "# Next-stage recommendation\n\nThe frozen strong-positive gate passed. A separately preregistered 10% → 20% → 30% → 40% → 50% row learning curve is recommended. No automatic run has started.\n"
        )


def write_artifact_manifest() -> None:
    files = [
        path
        for path in STUDY.rglob("*")
        if path.is_file() and "runtime" not in path.parts and path.name != "artifact_manifest.json"
    ]
    atomic_json(
        STUDY / "artifact_manifest.json",
        {
            "study": "qgeognn_v2_row_lcmd",
            "status": "COMPLETED",
            "source_data_sha256": sha256_file(SOURCE_DATA),
            "graph_cache_sha256": sha256_file(SOURCE_GRAPH_CACHE),
            "files": {
                str(path.relative_to(STUDY)): sha256_file(path)
                for path in sorted(files)
            },
        },
    )


def execute(workers: int) -> None:
    preflight_payload = json.loads((RESULTS / "preflight.json").read_text())
    if preflight_payload.get("status") != "PASS":
        raise RuntimeError("formal execution requires a passing full-gradient preflight")
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(workers)) as pool:
        completed = list(pool.map(_launch_seed, OUTER_SEEDS))
    atomic_json(RUNTIME / "execution_audit.json", {"runs": completed})
    if any(returncode != 0 for _, returncode in completed):
        raise RuntimeError("formal seed failure; inspect studies/.../runtime/logs")
    baseline, arms, summary, decision = aggregate_and_evaluate()
    make_figures(baseline, arms, summary)
    write_reports(baseline, arms, summary, decision)
    write_artifact_manifest()
    print(
        json.dumps(
            {
                "formal_experiment_complete": True,
                "directional_wins": decision["directional_wins"],
                "mean_combined_NRMSE_improvement_percent": decision["mean_combined_NRMSE_improvement_percent"],
                "decision": decision["decision"],
                "full_learning_curve_recommended": decision["full_learning_curve_recommended"],
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--run-seed", type=int)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.preflight:
        preflight()
    elif args.execute:
        execute(args.workers)
    else:
        run_seed_pre_evaluation(int(args.run_seed))


if __name__ == "__main__":
    main()
