#!/usr/bin/env python3
"""POST-PRIMARY matched V2 Hybrid extension; never modifies the frozen benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qgeognn_matplotlib")

import numpy as np
import pandas as pd
import torch

from src.qgeognn_al.active_learning_v2.benchmark_protocol import (
    CONFIRMATION_SEEDS,
    STUDY as PRIMARY_STUDY,
    TRAINING_CONFIG,
    initialization_seed,
    load_features,
    seed_config,
)
from src.qgeognn_al.active_learning_v2.benchmark_reporting import metric_row
from src.qgeognn_al.active_learning_v2.cache import verify_cache
from src.qgeognn_al.active_learning_v2.coverage import coreset_tp_select, extract_representations
from src.qgeognn_al.active_learning_v2.diagnostics import input_identities, mechanism_variables
from src.qgeognn_al.active_learning_v2.lcmd import _squared_distances
from src.qgeognn_al.active_learning_v2.protocol import RestrictedLabelStore, ids_hash, make_row_protocol, stable_hash
from src.qgeognn_al.active_learning_v2.runner import (
    fit_al_preprocessing,
    fit_from_same_initialization,
    predict_outputs,
)
from src.qgeognn_al.active_learning_v2.uncertainty import ensemble_uncertainty_select
from src.qgeognn_al.artifacts import sha256_file
from src.qgeognn_al.models import load_predictor_checkpoint
from src.qgeognn_al.resources import SOURCE_DATA, SOURCE_GRAPH_CACHE
from src.qgeognn_al.training.predictor import atomic_json


STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_hybrid_extension"
BATCH_SIZE = 32
ENSEMBLE_K = 3
SHORTLIST_FRACTION = 0.25
# This is a descriptive, preregistered interpretation margin, not a p-value.
COMPARABLE_MEAN_NRMSE_MARGIN = 0.01


def _canonical_descending(scores: np.ndarray) -> np.ndarray:
    """Descending score order; stable U0 canonical order resolves exact ties."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("uncertainty scores must be one finite vector")
    return np.argsort(-values, kind="stable")


def v2_hybrid_select(
    representations: np.ndarray,
    ensemble_predictions: np.ndarray,
    l0_count: int,
    scales: tuple[float, float],
    batch_size: int = BATCH_SIZE,
) -> dict[str, np.ndarray | int]:
    """Top-25% current-V2 uncertainty followed by current-V2 latent k-center.

    This function accepts only label-free acquisition inputs.  The representation
    is required to contain the L0 rows followed by the canonical U0 rows.
    """

    representation = np.asarray(representations, dtype=np.float64)
    if representation.ndim != 2 or representation.shape[1] != 128:
        raise ValueError("current QGeoGNN-V2 128D representations are required")
    if not 0 < int(l0_count) < len(representation):
        raise ValueError("representation must contain nonempty L0 and U0 blocks")
    pool_size = len(representation) - int(l0_count)
    if not 0 < int(batch_size) <= pool_size:
        raise ValueError("invalid hybrid batch size")
    ranking, scores = ensemble_uncertainty_select(ensemble_predictions, scales, pool_size)
    # Historical Hybrid used ceil(fraction * |U0|).  For 2,997 U0 rows this
    # deterministically selects the smallest integer no smaller than 25%: 750.
    shortlist_size = max(int(batch_size), int(math.ceil(SHORTLIST_FRACTION * pool_size)))
    shortlist_positions = np.asarray(ranking[:shortlist_size], dtype=np.int64)
    selected_local = coreset_tp_select(
        representation[int(l0_count) + shortlist_positions], representation[:int(l0_count)], int(batch_size)
    )
    selected_positions = shortlist_positions[selected_local]
    if len(selected_positions) != int(batch_size) or len(np.unique(selected_positions)) != int(batch_size):
        raise RuntimeError("hybrid selection size/uniqueness violation")
    if not set(selected_positions).issubset(set(shortlist_positions)):
        raise RuntimeError("hybrid escaped its uncertainty shortlist")
    return {
        "selected_positions": selected_positions,
        "shortlist_positions": shortlist_positions,
        "uncertainty_ranking": np.asarray(ranking, dtype=np.int64),
        "uncertainty_scores": np.asarray(scores, dtype=np.float64),
        "shortlist_size": shortlist_size,
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _assert_file_hashes(record: dict, root: Path, label: str) -> None:
    for relative, digest in record.get("files", {}).items():
        path = root / relative
        if not path.exists() or sha256_file(path) != digest:
            raise RuntimeError(f"{label} frozen artifact changed: {relative}")


def _primary_freeze() -> dict:
    path = PRIMARY_STUDY / "primary_results_freeze.json"
    if not path.exists():
        raise RuntimeError("the primary B=32 result freeze is missing")
    freeze = _read_json(path)
    _assert_file_hashes(freeze, PRIMARY_STUDY, "primary")
    if freeze.get("all_primary_seeds") is None:
        raise RuntimeError("primary freeze lacks its complete cohort declaration")
    return freeze


def _old_seed_paths(seed: int) -> tuple[Path, Path]:
    runtime = PRIMARY_STUDY / "runtime" / f"seed_{seed}"
    split = PRIMARY_STUDY / "splits" / f"row_seed_{seed}.csv"
    if not runtime.exists() or not split.exists():
        raise RuntimeError(f"missing frozen B=32 runtime or split for seed {seed}")
    return runtime, split


def _seed_partition(seed: int, data: pd.DataFrame) -> tuple[pd.DataFrame, dict, Path, Path]:
    runtime, split_path = _old_seed_paths(seed)
    partition = pd.read_csv(split_path)
    manifest = _read_json(PRIMARY_STUDY / "splits/split_manifest.json")
    manifest_row = next((row for row in manifest["splits"] if int(row["outer_seed"]) == int(seed)), None)
    if manifest_row is None or sha256_file(split_path) != manifest_row["sha256"]:
        raise RuntimeError(f"frozen split hash mismatch for seed {seed}")
    expected = make_row_protocol(data, seed)
    expected["study"] = PRIMARY_STUDY.name
    if not partition.equals(expected):
        raise RuntimeError(f"frozen row split differs from its canonical deterministic definition for seed {seed}")
    return partition, manifest_row, runtime, split_path


def _fit_rows(seed: int) -> pd.DataFrame:
    rows = pd.read_csv(PRIMARY_STUDY / "formal_results/fit_audit_b32.csv")
    rows = rows.loc[rows.outer_seed.eq(seed)].copy()
    required = {"baseline_l0", "ensemble_member_1", "ensemble_member_2", "b32/lcmd",
                *(f"b32/random_control_{i}" for i in range(5))}
    if set(rows.arm) & required != required:
        raise RuntimeError(f"formal fit audit is incomplete for seed {seed}")
    return rows


def _evaluation_config(seed: int, partition: pd.DataFrame) -> dict:
    """The exact V2 evaluation config, including the frozen row-split hash."""

    config = seed_config(seed)
    config["split_sha256"] = stable_hash(partition.to_dict("list"))
    return config


def _check_seed_reuse(seed: int, data: pd.DataFrame, protocol: dict) -> tuple[dict, list[dict], list[dict]]:
    partition, split_manifest, runtime, split_path = _seed_partition(seed, data)
    context = _read_json(runtime / "context.json")
    graph_receipt = _read_json(runtime / "scrubbed_graphs.pt.contract.json")
    graph_contract = graph_receipt.get("contract", {})
    if not verify_cache(runtime / "scrubbed_graphs.pt", graph_contract):
        raise RuntimeError(f"scrubbed graph cache is not reusable for seed {seed}")
    if stable_hash(graph_contract) != context.get("contract_hash"):
        raise RuntimeError(f"context contract hash mismatch for seed {seed}")
    roles = {role: partition.loc[partition.role.eq(role), "sample_id"].astype(str).tolist()
             for role in ("l0", "u0", "validation", "test")}
    expected_role_hashes = {role: ids_hash(values) for role, values in roles.items()}
    if context.get("roles") != expected_role_hashes:
        raise RuntimeError(f"context role identities mismatch for seed {seed}")
    if graph_contract.get("source_sha256") != sha256_file(SOURCE_DATA):
        raise RuntimeError(f"source-data contract mismatch for seed {seed}")
    if graph_contract.get("graph_cache_sha256") != sha256_file(SOURCE_GRAPH_CACHE):
        raise RuntimeError(f"graph-cache contract mismatch for seed {seed}")
    if graph_contract.get("ordered_sample_ids") != data.sample_id.astype(str).tolist():
        raise RuntimeError(f"canonical sample order mismatch for seed {seed}")
    if graph_contract.get("partition") != partition.to_dict("list"):
        raise RuntimeError(f"partition contract mismatch for seed {seed}")
    preprocessing = context.get("preprocessing")
    if graph_contract.get("preprocessing") != preprocessing:
        raise RuntimeError(f"preprocessing contract mismatch for seed {seed}")
    if graph_contract.get("smoke") is not False:
        raise RuntimeError(f"formal cache is marked smoke for seed {seed}")
    scales = preprocessing.get("target_scales", {})
    if set(scales) != {"V1", "V2"} or not all(float(scales[x]) > 0 for x in scales):
        raise RuntimeError(f"invalid L0-derived target scales for seed {seed}")

    fit_rows = _fit_rows(seed)
    expected_init_seed = initialization_seed(seed)
    expected_init_hash = fit_rows.loc[fit_rows.arm.eq("baseline_l0"), "initialization_hash"].iloc[0]
    compared = fit_rows.loc[fit_rows.arm.eq("b32/lcmd") | fit_rows.arm.str.startswith("b32/random_control_")]
    for column in ("initialization_seed", "initialization_hash", "validation_ids_hash", "test_ids_hash"):
        if compared[column].nunique() != 1:
            raise RuntimeError(f"old Random/LCMD initialization audit differs for seed {seed}: {column}")
    if int(compared.initialization_seed.iloc[0]) != expected_init_seed or compared.initialization_hash.iloc[0] != expected_init_hash:
        raise RuntimeError(f"old evaluation initialization differs from the frozen V2 rule for seed {seed}")
    if compared.validation_ids_hash.iloc[0] != expected_role_hashes["validation"] or compared.test_ids_hash.iloc[0] != expected_role_hashes["test"]:
        raise RuntimeError(f"old evaluation role IDs differ for seed {seed}")
    if not compared.model_variant.eq("qgeognn_v2").all() or not compared.test_labels_used_for_fit_or_checkpoint_selection.eq(0).all():
        raise RuntimeError(f"old evaluation provenance fails for seed {seed}")
    if not protocol.get("training") == TRAINING_CONFIG:
        raise RuntimeError("the frozen training configuration no longer matches the qualified V2 protocol")
    if protocol.get("primary_batch") != BATCH_SIZE or protocol.get("ensemble_K") != ENSEMBLE_K:
        raise RuntimeError("the frozen formal B=32/ensemble contract is incompatible")
    if protocol["training"].get("checkpoint_selection") != "validation_combined_normalized_rmse":
        raise RuntimeError("the frozen checkpoint criterion is incompatible")

    b32 = _read_json(runtime / "b32/pre_test_freeze.json")
    if (b32.get("status") != "FROZEN_BEFORE_TEST_TRUTH" or b32.get("batch_size") != BATCH_SIZE
            or b32.get("outer_seed") != seed or b32.get("test_truth_access_count") != 0):
        raise RuntimeError(f"old B=32 pre-test freeze is invalid for seed {seed}")
    _assert_file_hashes(b32, runtime, f"seed {seed} B=32")

    gradient_path = runtime / "gradient_features.npz"
    gradient_receipt = _read_json(gradient_path.with_suffix(".npz.contract.json"))
    gradient_contract = gradient_receipt.get("contract", {})
    if not verify_cache(gradient_path, gradient_contract):
        raise RuntimeError(f"gradient cache is not reusable for seed {seed}")
    outer_ids = roles["l0"] + roles["u0"]
    if (gradient_contract.get("context_hash") != context["contract_hash"]
            or gradient_contract.get("sample_ids") != outer_ids
            or tuple(gradient_contract.get("scales", ())) != (float(scales["V1"]), float(scales["V2"]))
            or gradient_contract.get("dimension") != 512):
        raise RuntimeError(f"gradient cache contract mismatch for seed {seed}")
    baseline_checkpoint = runtime / "baseline_l0/best.pt"
    if gradient_contract.get("checkpoint_sha256") != sha256_file(baseline_checkpoint):
        raise RuntimeError(f"gradient cache checkpoint mismatch for seed {seed}")
    with np.load(gradient_path) as gradient_cache:
        canonical_indices = gradient_cache["canonical_indices"]
        expected_indices = np.r_[partition.loc[partition.role.eq("l0"), "canonical_index"],
                                 partition.loc[partition.role.eq("u0"), "canonical_index"]]
        if not np.array_equal(canonical_indices, expected_indices):
            raise RuntimeError(f"gradient cache canonical order mismatch for seed {seed}")

    for member in (1, 2):
        member_row = fit_rows.loc[fit_rows.arm.eq(f"ensemble_member_{member}")].iloc[0]
        if (int(member_row.initialization_seed) != initialization_seed(seed, member)
                or member_row.model_variant != "qgeognn_v2"):
            raise RuntimeError(f"ensemble member {member} provenance mismatch for seed {seed}")
        member_path = runtime / f"ensemble_member_{member}/predictions.csv.gz"
        if sha256_file(member_path) != member_row.prediction_sha256:
            raise RuntimeError(f"ensemble prediction cache changed for seed {seed}, member {member}")
        table = pd.read_csv(member_path)
        if table.sample_id.astype(str).tolist() != roles["u0"] or tuple(table.columns) != (
            "sample_id", "V1_q10", "V1_q50", "V1_q90", "V2_q10", "V2_q50", "V2_q90"
        ):
            raise RuntimeError(f"ensemble prediction identity/output contract mismatch for seed {seed}")

    selections = pd.read_csv(runtime / "b32/selected_batches.csv")
    for arm in ["lcmd", *(f"random_control_{i}" for i in range(5))]:
        actual = selections.loc[selections.arm.eq(arm)].sort_values("selection_order")
        if len(actual) != BATCH_SIZE or actual.sample_id.nunique() != BATCH_SIZE or not set(actual.sample_id) <= set(roles["u0"]):
            raise RuntimeError(f"old {arm} batch does not match the B=32 U0 contract for seed {seed}")

    cache_rows = []
    for artifact, path, contract_hash in (
        ("scrubbed_graphs", runtime / "scrubbed_graphs.pt", graph_receipt["contract_hash"]),
        ("gradient_features", gradient_path, gradient_receipt["contract_hash"]),
        ("baseline_checkpoint", baseline_checkpoint, context["contract_hash"]),
        ("ensemble_member_1_predictions", runtime / "ensemble_member_1/predictions.csv.gz", None),
        ("ensemble_member_2_predictions", runtime / "ensemble_member_2/predictions.csv.gz", None),
    ):
        cache_rows.append({"outer_seed": seed, "artifact": artifact, "path": str(path.relative_to(ROOT)),
                           "sha256": sha256_file(path), "contract_hash": contract_hash,
                           "reuse_allowed": True, "reason": "all required content/protocol hashes match"})
    init_rows = []
    for _, row in compared.iterrows():
        init_rows.append({"outer_seed": seed, "method": "Random" if str(row.arm).startswith("b32/random") else "Gradient-LCMD",
                          "arm": row.arm, "initialization_seed": int(row.initialization_seed),
                          "initialization_hash": row.initialization_hash,
                          "l0_ids_hash": ids_hash(roles["l0"]), "validation_ids_hash": row.validation_ids_hash,
                          "test_ids_hash": row.test_ids_hash, "target_scales_hash": stable_hash(scales),
                          "training_config_hash": stable_hash(_evaluation_config(seed, partition)), "source": "frozen_primary"})
    record = {
        "outer_seed": seed,
        "split_sha256": sha256_file(split_path),
        "split_ids_hash": stable_hash(partition[["sample_id", "role"]].to_dict("list")),
        "l0_ids_hash": expected_role_hashes["l0"], "u0_ids_hash": expected_role_hashes["u0"],
        "validation_ids_hash": expected_role_hashes["validation"], "test_ids_hash": expected_role_hashes["test"],
        "initialization_hash": expected_init_hash, "initialization_seed": expected_init_seed,
        "training_config_hash": stable_hash(_evaluation_config(seed, partition)), "target_scales_hash": stable_hash(scales),
        "target_scales": scales, "context_contract_hash": context["contract_hash"],
        "primary_b32_pre_test_freeze_sha256": sha256_file(runtime / "b32/pre_test_freeze.json"),
        "random_reusable": True, "lcmd_reusable": True,
    }
    return record, cache_rows, init_rows


def audit_reuse(write: bool = True) -> dict:
    """Validate every reuse prerequisite before acquisition labels are touched."""

    try:
        freeze = _primary_freeze()
        protocol = _read_json(PRIMARY_STUDY / "protocol.json")
        if tuple(protocol.get("confirmation_seeds", ())) != CONFIRMATION_SEEDS:
            raise RuntimeError("the old confirmation cohort does not equal the extension cohort")
        if protocol.get("model_variant", "qgeognn_v2") != "qgeognn_v2":
            raise RuntimeError("the old primary predictor is not current QGeoGNN-V2")
        data = load_features()
        records, cache_rows, initialization_rows = [], [], []
        for seed in CONFIRMATION_SEEDS:
            record, caches, inits = _check_seed_reuse(seed, data, protocol)
            records.append(record)
            cache_rows.extend(caches)
            initialization_rows.extend(inits)
        result = {
            "status": "PASS",
            "study": "POST-PRIMARY MATCHED EXTENSION",
            "primary_study": str(PRIMARY_STUDY.relative_to(ROOT)),
            "primary_results_freeze_sha256": sha256_file(PRIMARY_STUDY / "primary_results_freeze.json"),
            "primary_decision": _read_json(PRIMARY_STUDY / "formal_decision.json")["primary"]["decision"],
            "confirmation_seeds": list(CONFIRMATION_SEEDS),
            "B": BATCH_SIZE,
            "ensemble_K": ENSEMBLE_K,
            "reuse_old_random": "YES",
            "reuse_old_lcmd": "YES",
            "checks": {
                "confirmation_cohort": True, "row_splits": True, "l0_ids": True,
                "validation_test_ids": True, "predictor": True, "training_config": True,
                "evaluation_initialization": True, "target_scales": True,
                "checkpoint_criterion": True, "b32": True,
                "primary_freeze": True, "cache_contracts": True,
            },
            "seeds": records,
        }
    except Exception as error:
        result = {"status": "BLOCKED", "study": "POST-PRIMARY MATCHED EXTENSION", "error": str(error),
                  "reuse_old_random": "NO", "reuse_old_lcmd": "NO"}
        cache_rows, initialization_rows = [], []
    if write:
        STUDY.mkdir(parents=True, exist_ok=True)
        (STUDY / "results").mkdir(exist_ok=True)
        atomic_json(STUDY / "reuse_audit.json", result)
        pd.DataFrame(cache_rows).to_csv(STUDY / "results/cache_reuse_audit.csv", index=False)
        pd.DataFrame(initialization_rows).to_csv(STUDY / "results/initialization_hash_audit.csv", index=False)
    if result["status"] != "PASS":
        raise RuntimeError(f"reuse audit blocked: {result['error']}")
    return result


def _load_seed_inputs(seed: int) -> dict:
    """Load only the audited old fit/cache artifacts; this reads no U0/test labels."""

    data = load_features()
    partition, _, old_runtime, _ = _seed_partition(seed, data)
    context = _read_json(old_runtime / "context.json")
    atom, angle = torch.load(old_runtime / "scrubbed_graphs.pt", weights_only=False)
    if any(torch.count_nonzero(item.y) for item in atom):
        raise RuntimeError("reused graph cache contains non-sentinel labels")
    roles = {role: partition.loc[partition.role.eq(role), "canonical_index"].to_numpy(int)
             for role in ("l0", "u0", "validation", "test")}
    outer = np.r_[roles["l0"], roles["u0"]]
    model = load_predictor_checkpoint(old_runtime / "baseline_l0/best.pt")
    representations = extract_representations(model, atom, angle, outer)
    if representations.shape != (len(outer), 128):
        raise RuntimeError(f"current V2 extract_representation did not return 128D rows for seed {seed}")
    baseline_pool, order = predict_outputs(model, atom, angle, roles["u0"])
    if not np.array_equal(order, roles["u0"]):
        raise RuntimeError(f"baseline U0 prediction order drift for seed {seed}")
    member_tables = [baseline_pool]
    for member in (1, 2):
        table = pd.read_csv(old_runtime / f"ensemble_member_{member}/predictions.csv.gz")
        expected_ids = data.iloc[roles["u0"]].sample_id.astype(str).tolist()
        if table.sample_id.astype(str).tolist() != expected_ids:
            raise RuntimeError(f"ensemble U0 identity mismatch for seed {seed}")
        member_tables.append(table.drop(columns="sample_id").to_numpy())
    ensemble_predictions = np.stack(member_tables)
    with np.load(old_runtime / "gradient_features.npz") as cached:
        gradients = cached["features"]
        gradient_indices = cached["canonical_indices"]
    if not np.array_equal(gradient_indices, outer):
        raise RuntimeError(f"gradient canonical order drift for seed {seed}")
    return {
        "seed": seed, "data": data, "partition": partition, "old_runtime": old_runtime,
        "context": context, "atom": atom, "angle": angle, "roles": roles, "outer": outer,
        "model": model, "representations": representations, "ensemble_predictions": ensemble_predictions,
        "gradients": gradients,
    }


def _safe_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write deterministic CSV once; an interrupted rerun cannot silently change it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = pd.read_csv(path)
        same = list(current.columns) == list(frame.columns) and len(current) == len(frame)
        if same:
            for column in frame.columns:
                left, right = current[column], frame[column]
                if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
                    same = bool(np.allclose(left.to_numpy(float), right.to_numpy(float), rtol=0.0, atol=1e-12,
                                              equal_nan=True))
                else:
                    same = left.astype(str).tolist() == right.astype(str).tolist()
                if not same:
                    break
        if not same:
            raise RuntimeError(f"refusing to overwrite a nonidentical frozen artifact: {path}")
        return
    frame.to_csv(path, index=False)


def _hybrid_tables(inputs: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    seed, data, roles = inputs["seed"], inputs["data"], inputs["roles"]
    l0_count = len(roles["l0"])
    scales = tuple(float(inputs["context"]["preprocessing"]["target_scales"][target]) for target in ("V1", "V2"))
    selection = v2_hybrid_select(inputs["representations"], inputs["ensemble_predictions"], l0_count, scales)
    u0_ids = data.iloc[roles["u0"]].sample_id.astype(str).to_numpy()
    ranks = np.empty(len(u0_ids), dtype=int)
    ranks[selection["uncertainty_ranking"]] = np.arange(1, len(u0_ids) + 1)
    shortlist_set = set(selection["shortlist_positions"].tolist())
    shortlist = pd.DataFrame({
        "outer_seed": seed,
        "shortlist_rank": np.arange(1, int(selection["shortlist_size"]) + 1),
        "canonical_u0_position": selection["shortlist_positions"],
        "canonical_index": roles["u0"][selection["shortlist_positions"]],
        "sample_id": u0_ids[selection["shortlist_positions"]],
        "uncertainty_score": selection["uncertainty_scores"][selection["shortlist_positions"]],
    })
    selected = pd.DataFrame({
        "outer_seed": seed,
        "selection_order": np.arange(BATCH_SIZE),
        "canonical_u0_position": selection["selected_positions"],
        "canonical_index": roles["u0"][selection["selected_positions"]],
        "sample_id": u0_ids[selection["selected_positions"]],
        "uncertainty_score": selection["uncertainty_scores"][selection["selected_positions"]],
        "uncertainty_rank": ranks[selection["selected_positions"]],
        "in_top25_shortlist": [int(value) in shortlist_set for value in selection["selected_positions"]],
    })
    if len(shortlist) != int(selection["shortlist_size"]) or len(selected) != BATCH_SIZE or not selected.in_top25_shortlist.all():
        raise RuntimeError("hybrid shortlist/selection output contract failed")
    return selection, shortlist, selected


def _freeze_hybrid_seed(seed: int) -> dict:
    inputs = _load_seed_inputs(seed)
    selection, shortlist, selected = _hybrid_tables(inputs)
    runtime = STUDY / "runtime" / f"seed_{seed}"
    _safe_write_csv(shortlist, runtime / "shortlist.csv")
    _safe_write_csv(selected, runtime / "hybrid_selection.csv")
    store = RestrictedLabelStore(SOURCE_DATA, inputs["partition"])
    roles, data = inputs["roles"], inputs["data"]
    l0_ids = data.iloc[roles["l0"]].sample_id.astype(str).tolist()
    validation_ids = data.iloc[roles["validation"]].sample_id.astype(str).tolist()
    l0_truth = store.reveal(l0_ids, "initial_fit")
    validation_truth = store.reveal(validation_ids, "initial_fit")
    graph_cache = torch.load(SOURCE_GRAPH_CACHE, weights_only=False)
    normalization, preprocessing = fit_al_preprocessing(
        data, graph_cache, inputs["outer"], l0_ids, l0_truth, runtime / "scaler.json"
    )
    old_preprocessing = inputs["context"]["preprocessing"]
    graph_receipt = _read_json(inputs["old_runtime"] / "scrubbed_graphs.pt.contract.json")
    if stable_hash(preprocessing) != stable_hash(old_preprocessing) or stable_hash(asdict(normalization)) != stable_hash(graph_receipt["contract"]["normalization"]):
        raise RuntimeError(f"recomputed V2 preprocessing/normalization differs from the frozen B=32 contract for seed {seed}")
    if tuple(preprocessing["target_scales"][target] for target in ("V1", "V2")) != tuple(old_preprocessing["target_scales"][target] for target in ("V1", "V2")):
        raise RuntimeError(f"target scales differ from the frozen B=32 contract for seed {seed}")
    selected_indices = roles["u0"][selected.canonical_u0_position.to_numpy(int)]
    selected_ids = selected.sample_id.astype(str).tolist()
    store.freeze_acquisitions(selected_ids)
    acquired_truth = store.reveal(selected_ids, "after_acquisition_fit")
    fit = fit_from_same_initialization(
        atom_base=inputs["atom"], angle=inputs["angle"], normalization=normalization, preprocessing=preprocessing,
        train_indices=np.r_[roles["l0"], selected_indices], train_truth=np.vstack([l0_truth, acquired_truth]),
        validation_indices=roles["validation"], validation_truth=validation_truth,
        prediction_indices=roles["test"], prediction_sample_ids=data.iloc[roles["test"]].sample_id.astype(str).tolist(),
        initialization_seed=initialization_seed(seed), training_config=_evaluation_config(seed, inputs["partition"]),
        contract={"study": STUDY.name, "arm": "hybrid", "old_context_hash": inputs["context"]["contract_hash"],
                  "train_sample_ids": l0_ids + selected_ids, "validation_sample_ids": validation_ids,
                  "selected_ids_hash": stable_hash(selected_ids),
                  "shortlist_ids_hash": stable_hash(shortlist.sample_id.astype(str).tolist()),
                  "hybrid_definition": "top25_epistemic_then_current_v2_latent_farthest_first"},
        runtime=runtime / "hybrid",
    )
    old_fit = _fit_rows(seed)
    expected_init = old_fit.loc[old_fit.arm.eq("b32/lcmd"), "initialization_hash"].iloc[0]
    if fit["initialization_hash"] != expected_init or fit["initialization_seed"] != initialization_seed(seed):
        raise RuntimeError(f"Hybrid evaluation initialization differs from Random/LCMD for seed {seed}")
    _safe_write_csv(pd.DataFrame(store.audit), runtime / "label_access_audit.csv")
    protected = [runtime / "shortlist.csv", runtime / "hybrid_selection.csv", runtime / "label_access_audit.csv",
                 runtime / "hybrid/best.pt", runtime / "hybrid/predictions.csv.gz", runtime / "hybrid/fit_audit.json",
                 runtime / "hybrid/history.csv"]
    freeze = {
        "status": "FROZEN_BEFORE_TEST_TRUTH", "outer_seed": seed, "batch_size": BATCH_SIZE,
        "test_truth_access_count": 0, "selected_ids_hash": stable_hash(selected_ids),
        "shortlist_ids_hash": stable_hash(shortlist.sample_id.astype(str).tolist()),
        "initialization_hash": fit["initialization_hash"],
        "training_config_hash": stable_hash(_evaluation_config(seed, inputs["partition"])),
        "files": {str(path.relative_to(runtime)): sha256_file(path) for path in protected},
    }
    atomic_json(runtime / "pre_test_freeze.json", freeze)
    return {"fit": fit, "selected": selected, "shortlist": shortlist, "freeze": freeze}


def _validate_hybrid_freezes() -> dict:
    records = {}
    for seed in CONFIRMATION_SEEDS:
        runtime = STUDY / "runtime" / f"seed_{seed}"
        path = runtime / "pre_test_freeze.json"
        if not path.exists():
            raise RuntimeError(f"Hybrid seed {seed} did not reach the global pre-test barrier")
        freeze = _read_json(path)
        if (freeze.get("status") != "FROZEN_BEFORE_TEST_TRUTH" or freeze.get("outer_seed") != seed
                or freeze.get("batch_size") != BATCH_SIZE or freeze.get("test_truth_access_count") != 0):
            raise RuntimeError(f"invalid Hybrid pre-test freeze for seed {seed}")
        _assert_file_hashes(freeze, runtime, f"Hybrid seed {seed}")
        records[str(seed)] = {"sha256": sha256_file(path), **freeze}
    return records


def _evaluate_hybrid_frozen() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Only called after all five Hybrid fits and predictions are frozen."""

    rows, accesses = [], []
    for seed in CONFIRMATION_SEEDS:
        data = load_features()
        partition, _, _, _ = _seed_partition(seed, data)
        runtime = STUDY / "runtime" / f"seed_{seed}"
        selection = pd.read_csv(runtime / "hybrid_selection.csv").sort_values("selection_order")
        store = RestrictedLabelStore(SOURCE_DATA, partition)
        store.freeze_acquisitions(selection.sample_id.astype(str).tolist())
        store.freeze_predictions()
        test_ids = partition.loc[partition.role.eq("test"), "sample_id"].astype(str).tolist()
        truth = store.reveal(test_ids, "final_test_evaluation")
        prediction = pd.read_csv(runtime / "hybrid/predictions.csv.gz")
        if prediction.sample_id.astype(str).tolist() != test_ids:
            raise RuntimeError(f"Hybrid test prediction identity/order mismatch for seed {seed}")
        context = _read_json(PRIMARY_STUDY / "runtime" / f"seed_{seed}/context.json")
        scales = context["preprocessing"]["target_scales"]
        base = pd.read_csv(PRIMARY_STUDY / "formal_results/all_metrics_b32.csv")
        base_nrmse = float(base.loc[(base.outer_seed.eq(seed)) & base.arm.eq("baseline_l0"), "combined_normalized_RMSE"].iloc[0])
        values = metric_row(truth, prediction.drop(columns="sample_id").to_numpy(), scales)
        rows.append({"outer_seed": seed, "batch_size": BATCH_SIZE, "arm": "hybrid", "cohort": "confirmation",
                     "active_label_count": 365, "shared_validation_label_count": 416,
                     "combined_normalized_RMSE": values["combined_normalized_RMSE"],
                     "V1_RMSE": values["V1_RMSE"], "V1_MAE": values["V1_MAE"], "V1_R2": values["V1_R2"],
                     "V2_RMSE": values["V2_RMSE"], "V2_MAE": values["V2_MAE"], "V2_R2": values["V2_R2"],
                     "baseline_L0_NRMSE": base_nrmse, "gain": base_nrmse - values["combined_normalized_RMSE"]})
        accesses.extend({"outer_seed": seed, **entry} for entry in store.audit)
    return pd.DataFrame(rows), pd.DataFrame(accesses)


def _selected_mechanism_profiles(seed: int, hybrid: pd.DataFrame) -> tuple[list[dict], dict]:
    inputs = _load_seed_inputs(seed)
    data, roles = inputs["data"], inputs["roles"]
    selected_old = pd.read_csv(inputs["old_runtime"] / "b32/selected_batches.csv")
    lcmd = selected_old.loc[selected_old.arm.eq("lcmd")].sort_values("selection_order")
    hybrid = hybrid.sort_values("selection_order")
    selection, _, _ = _hybrid_tables(inputs)
    scores = selection["uncertainty_scores"]
    l0_count = len(roles["l0"])
    prediction, order = predict_outputs(inputs["model"], inputs["atom"], inputs["angle"], inputs["outer"])
    if not np.array_equal(order, inputs["outer"]):
        raise RuntimeError(f"mechanism prediction order drift for seed {seed}")
    variables = mechanism_variables(data, inputs["outer"], l0_count, inputs["gradients"], prediction)
    variables["ensemble_uncertainty"] = np.r_[np.full(l0_count, np.nan), scores]
    latent_distance = np.sqrt(_squared_distances(inputs["representations"], inputs["representations"][:l0_count]).min(axis=1))
    variables["nearest_l0_latent_distance"] = latent_distance
    identities = input_identities(data, torch.load(SOURCE_GRAPH_CACHE, weights_only=False), inputs["atom"], inputs["angle"], inputs["outer"])
    variables = variables.merge(identities[["sample_id", "input_hash"]], on="sample_id", validate="one_to_one")
    u0_ids = data.iloc[roles["u0"]].sample_id.astype(str).tolist()
    position = {sample_id: index for index, sample_id in enumerate(u0_ids)}

    def profile(method: str, selection_frame: pd.DataFrame) -> dict:
        ids = selection_frame.sample_id.astype(str).tolist()
        subset = variables.set_index("sample_id").loc[ids]
        seen = set(identities.iloc[:l0_count].input_hash.astype(str))
        duplicate_count = 0
        for value in subset.input_hash.astype(str):
            duplicate_count += value in seen
            seen.add(value)
        positions = np.asarray([position[item] for item in ids], dtype=int)
        return {"outer_seed": seed, "method": method, "batch_size": len(ids),
                "mean_ensemble_uncertainty": float(subset.ensemble_uncertainty.mean()),
                "mean_gradient_norm": float(subset.gradient_norm.mean()),
                "mean_nearest_l0_gradient_distance": float(subset.nearest_l0_gradient_distance.mean()),
                "mean_nearest_l0_latent_distance": float(subset.nearest_l0_latent_distance.mean()),
                "unique_compounds": int(data.set_index("sample_id").loc[ids, "canonical_smiles"].nunique()),
                "exact_X_duplicate_rate": duplicate_count / len(ids),
                "mean_predicted_V1_q50": float(subset.predicted_V1_q50.mean()),
                "mean_predicted_V2_q50": float(subset.predicted_V2_q50.mean()),
                "mean_molecular_weight": float(subset.MolWt.mean()), "mean_atom_count": float(subset.atom_count.mean()),
                "mean_nearest_l0_condition_distance": float(subset.nearest_l0_condition_distance.mean()),
                "mean_uncertainty_rank": float(np.mean(np.empty(len(ids), dtype=int))),
                "fraction_within_top25": float(np.mean([value in set(selection["shortlist_positions"]) for value in positions])),
            }

    profiles = [profile("Hybrid", hybrid), profile("Gradient-LCMD", lcmd)]
    ranks = np.empty(len(u0_ids), dtype=int)
    ranks[selection["uncertainty_ranking"]] = np.arange(1, len(u0_ids) + 1)
    for row, selection_frame in zip(profiles, (hybrid, lcmd)):
        positions = np.asarray([position[item] for item in selection_frame.sample_id.astype(str)], dtype=int)
        row["mean_uncertainty_rank"] = float(ranks[positions].mean())
    hybrid_ids, lcmd_ids = set(hybrid.sample_id.astype(str)), set(lcmd.sample_id.astype(str))
    outside = lcmd.loc[~lcmd.canonical_index.isin(roles["u0"][selection["shortlist_positions"]])]
    overlap = {"outer_seed": seed, "hybrid_lcmd_overlap_count": len(hybrid_ids & lcmd_ids),
               "hybrid_lcmd_overlap_fraction": len(hybrid_ids & lcmd_ids) / BATCH_SIZE,
               "lcmd_outside_hybrid_shortlist_count": len(outside),
               "lcmd_outside_hybrid_shortlist_fraction": len(outside) / BATCH_SIZE,
               "shortlist_size": int(selection["shortlist_size"]),
               "hybrid_all_from_shortlist": bool(set(hybrid.canonical_index).issubset(set(roles["u0"][selection["shortlist_positions"]])))}
    return profiles, overlap


def _old_primary_arms() -> pd.DataFrame:
    rows = pd.read_csv(PRIMARY_STUDY / "formal_results/all_metrics_b32.csv")
    rows = rows.loc[rows.outer_seed.isin(CONFIRMATION_SEEDS)].copy()
    required = {"baseline_l0", "lcmd", *(f"random_control_{i}" for i in range(5))}
    if any(set(frame.arm) & required != required for _, frame in rows.groupby("outer_seed")):
        raise RuntimeError("old B=32 formal metrics lack Random/LCMD arms")
    return rows


def _decision_and_tables(hybrid_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    old = _old_primary_arms()
    records = []
    for seed in CONFIRMATION_SEEDS:
        rows = old.loc[old.outer_seed.eq(seed)]
        random = rows.loc[rows.arm.str.startswith("random_control_")]
        lcmd = rows.loc[rows.arm.eq("lcmd")].iloc[0]
        hybrid = hybrid_metrics.loc[hybrid_metrics.outer_seed.eq(seed)].iloc[0]
        random_mean = float(random.combined_normalized_RMSE.mean())
        random_median = float(random.combined_normalized_RMSE.median())
        records.append({"outer_seed": seed, "Random_mean_NRMSE": random_mean, "Random_median_NRMSE": random_median,
                        "Hybrid_NRMSE": float(hybrid.combined_normalized_RMSE), "LCMD_NRMSE": float(lcmd.combined_normalized_RMSE),
                        "Hybrid_vs_Random_mean": random_mean - float(hybrid.combined_normalized_RMSE),
                        "Hybrid_vs_Random_median": random_median - float(hybrid.combined_normalized_RMSE),
                        "Hybrid_vs_LCMD": float(lcmd.combined_normalized_RMSE) - float(hybrid.combined_normalized_RMSE),
                        "LCMD_vs_Hybrid": float(hybrid.combined_normalized_RMSE) - float(lcmd.combined_normalized_RMSE),
                        "Hybrid_beats_Random_median": bool(float(hybrid.combined_normalized_RMSE) < random_median),
                        "Hybrid_beats_LCMD": bool(float(hybrid.combined_normalized_RMSE) < float(lcmd.combined_normalized_RMSE)),
                        "LCMD_beats_Hybrid": bool(float(lcmd.combined_normalized_RMSE) < float(hybrid.combined_normalized_RMSE))})
    pairwise = pd.DataFrame(records)
    metric_columns = ["combined_normalized_RMSE", "V1_RMSE", "V1_MAE", "V1_R2", "V2_RMSE", "V2_MAE", "V2_R2"]
    summary_rows = []
    for method, frame in (("Random", old.loc[old.arm.str.startswith("random_control_")]),
                          ("Hybrid", hybrid_metrics), ("Gradient-LCMD", old.loc[old.arm.eq("lcmd")])):
        summary_rows.append({"method": method, "n_fits": len(frame), "n_outer_seeds": frame.outer_seed.nunique(),
                             **{f"mean_{column}": float(frame[column].mean()) for column in metric_columns}})
    summary = pd.DataFrame(summary_rows)
    means = summary.set_index("method")
    hybrid_random = bool(means.loc["Hybrid", "mean_combined_normalized_RMSE"] < means.loc["Random", "mean_combined_normalized_RMSE"]
                         and int(pairwise.Hybrid_beats_Random_median.sum()) >= 4)
    hybrid_lcmd = bool(means.loc["Hybrid", "mean_combined_normalized_RMSE"] < means.loc["Gradient-LCMD", "mean_combined_normalized_RMSE"]
                       and int(pairwise.Hybrid_beats_LCMD.sum()) >= 3)
    lcmd_hybrid = bool(means.loc["Gradient-LCMD", "mean_combined_normalized_RMSE"] < means.loc["Hybrid", "mean_combined_normalized_RMSE"]
                       and int(pairwise.LCMD_beats_Hybrid.sum()) >= 3)
    comparable = bool(abs(means.loc["Gradient-LCMD", "mean_combined_normalized_RMSE"] - means.loc["Hybrid", "mean_combined_normalized_RMSE"])
                      <= COMPARABLE_MEAN_NRMSE_MARGIN
                      and int(pairwise.Hybrid_beats_LCMD.sum()) < 3 and int(pairwise.LCMD_beats_Hybrid.sum()) < 3)
    if hybrid_random and hybrid_lcmd:
        decision = "HYBRID_BEST_IN_EXTENSION"
    elif lcmd_hybrid:
        decision = "LCMD_REMAINS_BEST"
    elif comparable:
        decision = "LCMD_HYBRID_COMPARABLE"
    elif not hybrid_random:
        decision = "HYBRID_DOES_NOT_BEAT_RANDOM"
    else:
        decision = "BLOCKED"
    endpoint_warnings = []
    for first, second in (("Hybrid", "Gradient-LCMD"), ("Gradient-LCMD", "Hybrid")):
        if means.loc[first, "mean_combined_normalized_RMSE"] < means.loc[second, "mean_combined_normalized_RMSE"]:
            for endpoint in ("V1", "V2"):
                change = means.loc[first, f"mean_{endpoint}_RMSE"] / means.loc[second, f"mean_{endpoint}_RMSE"] - 1
                if change > 0.02:
                    endpoint_warnings.append({"better_combined_method": first, "comparison_method": second,
                                              "endpoint": endpoint, "relative_rmse_change": float(change)})
    decision_record = {"study": "POST-PRIMARY MATCHED EXTENSION", "decision": decision,
                       "primary_decision_unchanged": "BEST_CURRENT_ROW_ACQUISITION",
                       "rules": {"hybrid_beats_random": "mean lower and >=4/5 Random-median wins",
                                 "hybrid_best": "mean lower than LCMD and >=3/5 LCMD wins",
                                 "lcmd_remains_best": "mean lower than Hybrid and >=3/5 Hybrid wins",
                                 "comparable_margin_nrmse": COMPARABLE_MEAN_NRMSE_MARGIN},
                       "conditions": {"HYBRID_BEATS_RANDOM": hybrid_random, "HYBRID_BEST_IN_EXTENSION": hybrid_lcmd,
                                      "LCMD_REMAINS_BEST": lcmd_hybrid, "LCMD_HYBRID_COMPARABLE": comparable},
                       "directional_wins": {"Hybrid_vs_Random_median": int(pairwise.Hybrid_beats_Random_median.sum()),
                                              "Hybrid_vs_LCMD": int(pairwise.Hybrid_beats_LCMD.sum()),
                                              "LCMD_vs_Hybrid": int(pairwise.LCMD_beats_Hybrid.sum())},
                       "method_means": means.to_dict("index"), "endpoint_guard_warnings": endpoint_warnings}
    return pairwise, summary, old, decision_record


def _plot_results(pairwise: pd.DataFrame, profiles: pd.DataFrame, overlap: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = STUDY / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    ordered = pairwise.sort_values("outer_seed")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for column, label, color in (("Random_mean_NRMSE", "Random mean", "#8c8c8c"),
                                 ("Hybrid_NRMSE", "V2-Hybrid", "#377eb8"),
                                 ("LCMD_NRMSE", "Gradient-LCMD", "#e41a1c")):
        ax.plot(ordered.outer_seed.astype(str), ordered[column], marker="o", label=label, color=color)
    ax.set(ylabel="combined normalized RMSE", title="POST-PRIMARY matched B=32 extension")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "random_hybrid_lcmd_nrmse.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(ordered)); width = .26
    ax.bar(x - width, ordered.Random_mean_NRMSE, width, label="Random mean", color="#8c8c8c")
    ax.bar(x, ordered.Hybrid_NRMSE, width, label="V2-Hybrid", color="#377eb8")
    ax.bar(x + width, ordered.LCMD_NRMSE, width, label="Gradient-LCMD", color="#e41a1c")
    ax.set(xticks=x, xticklabels=ordered.outer_seed.astype(str), ylabel="combined normalized RMSE")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "paired_seed_comparison.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(ordered.outer_seed.astype(str), overlap.set_index("outer_seed").loc[ordered.outer_seed, "hybrid_lcmd_overlap_fraction"], label="Hybrid–LCMD overlap")
    ax.bar(ordered.outer_seed.astype(str), overlap.set_index("outer_seed").loc[ordered.outer_seed, "lcmd_outside_hybrid_shortlist_fraction"],
           bottom=0, alpha=.45, label="LCMD outside Hybrid shortlist")
    ax.set(ylabel="fraction of B=32", title="Selection-set relation")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "hybrid_lcmd_overlap.png", dpi=160); plt.close(fig)
    variables = ["mean_ensemble_uncertainty", "mean_gradient_norm", "mean_nearest_l0_gradient_distance", "mean_nearest_l0_latent_distance"]
    means = profiles.groupby("method")[variables].mean().T
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for axis, variable in zip(axes.flat, variables):
        means.loc[variable].plot.bar(ax=axis, color=["#e41a1c", "#377eb8"])
        axis.set_title(variable.replace("mean_", "").replace("_", " "))
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(figures / "hybrid_lcmd_mechanism.png", dpi=160); plt.close(fig)


def _write_results(hybrid_metrics: pd.DataFrame, label_access: pd.DataFrame, frozen: dict, runs: list[dict]) -> dict:
    results = STUDY / "results"
    results.mkdir(parents=True, exist_ok=True)
    pairwise, summary, old, decision = _decision_and_tables(hybrid_metrics)
    _safe_write_csv(hybrid_metrics.sort_values("outer_seed"), results / "hybrid_arm_metrics.csv")
    _safe_write_csv(pairwise.sort_values("outer_seed"), results / "pairwise_comparison.csv")
    _safe_write_csv(summary, results / "three_method_summary.csv")
    _safe_write_csv(label_access.sort_values(["outer_seed", "purpose"]), results / "hybrid_label_access_audit.csv")
    shortlists = pd.concat([item["shortlist"] for item in runs], ignore_index=True)
    selections = pd.concat([item["selected"] for item in runs], ignore_index=True)
    _safe_write_csv(shortlists, results / "shortlists.csv")
    _safe_write_csv(selections, results / "hybrid_selected_batches.csv")
    profiles, overlaps = [], []
    for item in runs:
        seed_profiles, seed_overlap = _selected_mechanism_profiles(int(item["selected"].outer_seed.iloc[0]), item["selected"])
        profiles.extend(seed_profiles); overlaps.append(seed_overlap)
    profile_frame, overlap_frame = pd.DataFrame(profiles), pd.DataFrame(overlaps)
    _safe_write_csv(profile_frame.sort_values(["outer_seed", "method"]), results / "mechanism_profiles.csv")
    _safe_write_csv(overlap_frame.sort_values("outer_seed"), results / "selection_overlap.csv")
    init = pd.read_csv(results / "initialization_hash_audit.csv")
    hybrid_init = []
    for item in runs:
        seed = int(item["selected"].outer_seed.iloc[0])
        fit = item["fit"]
        context = _read_json(PRIMARY_STUDY / "runtime" / f"seed_{seed}/context.json")
        partition = pd.read_csv(PRIMARY_STUDY / "splits" / f"row_seed_{seed}.csv")
        hybrid_init.append({"outer_seed": seed, "method": "Hybrid", "arm": "hybrid",
                            "initialization_seed": fit["initialization_seed"], "initialization_hash": fit["initialization_hash"],
                            "l0_ids_hash": ids_hash(partition.loc[partition.role.eq("l0"), "sample_id"]),
                            "validation_ids_hash": fit["validation_ids_hash"],
                            "test_ids_hash": ids_hash(partition.loc[partition.role.eq("test"), "sample_id"]),
                            "target_scales_hash": stable_hash(context["preprocessing"]["target_scales"]),
                            "training_config_hash": stable_hash(_evaluation_config(seed, partition)), "source": "extension"})
    init = pd.concat([init, pd.DataFrame(hybrid_init)], ignore_index=True)
    if not init.groupby("outer_seed").initialization_hash.nunique().eq(1).all():
        raise RuntimeError("Hybrid did not share the frozen evaluation initialization with Random/LCMD")
    # `audit_reuse` intentionally writes the old-only audit first; the formal
    # result atomically upgrades it to the complete three-method audit.
    init.sort_values(["outer_seed", "method", "arm"]).to_csv(results / "initialization_hash_audit.csv", index=False)
    atomic_json(results / "global_pre_test_freeze.json", {"seeds": frozen, "test_truth_revealed_after_global_freeze": True})
    decision["global_pre_test_freeze_sha256"] = sha256_file(results / "global_pre_test_freeze.json")
    atomic_json(STUDY / "decision.json", decision)
    _plot_results(pairwise, profile_frame, overlap_frame)
    mean = summary.set_index("method")
    result_text = f"""# Results — V2 Hybrid matched extension

This is a **POST-PRIMARY MATCHED EXTENSION**.  It does not amend the frozen primary decision `BEST_CURRENT_ROW_ACQUISITION`.

## Decision

`{decision['decision']}`

| Method | Mean combined NRMSE | Mean V1 RMSE | Mean V2 RMSE |
| --- | ---: | ---: | ---: |
| Random | {mean.loc['Random','mean_combined_normalized_RMSE']:.6f} | {mean.loc['Random','mean_V1_RMSE']:.6f} | {mean.loc['Random','mean_V2_RMSE']:.6f} |
| V2-Hybrid | {mean.loc['Hybrid','mean_combined_normalized_RMSE']:.6f} | {mean.loc['Hybrid','mean_V1_RMSE']:.6f} | {mean.loc['Hybrid','mean_V2_RMSE']:.6f} |
| Gradient-LCMD | {mean.loc['Gradient-LCMD','mean_combined_normalized_RMSE']:.6f} | {mean.loc['Gradient-LCMD','mean_V1_RMSE']:.6f} | {mean.loc['Gradient-LCMD','mean_V2_RMSE']:.6f} |

Hybrid beat its within-seed Random median in {decision['directional_wins']['Hybrid_vs_Random_median']}/5 seeds. Hybrid beat LCMD in {decision['directional_wins']['Hybrid_vs_LCMD']}/5 seeds; LCMD beat Hybrid in {decision['directional_wins']['LCMD_vs_Hybrid']}/5 seeds.

The selection and mechanism diagnostics are descriptive and did not feed back into acquisition.  Endpoint guard warnings: {json.dumps(decision['endpoint_guard_warnings'])}.
"""
    (STUDY / "RESULTS.md").write_text(result_text)
    return decision


def execute() -> dict:
    audit_reuse(write=True)
    if (STUDY / "decision.json").exists():
        raise RuntimeError("extension already has a formal decision; refusing to overwrite it")
    runs = []
    for seed in CONFIRMATION_SEEDS:
        runs.append(_freeze_hybrid_seed(seed))
        print(json.dumps({"hybrid_seed_frozen": seed, "test_truth_access_count": 0}), flush=True)
    frozen = _validate_hybrid_freezes()
    hybrid_metrics, label_access = _evaluate_hybrid_frozen()
    decision = _write_results(hybrid_metrics, label_access, frozen, runs)
    print(json.dumps({"decision": decision["decision"]}, indent=2), flush=True)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audit", action="store_true", help="strict read-only reuse audit")
    group.add_argument("--run", action="store_true", help="fit/evaluate the five Hybrid extension arms")
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit_reuse(write=True), indent=2))
    else:
        execute()


if __name__ == "__main__":
    main()
