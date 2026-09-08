#!/usr/bin/env python3
"""Run and summarize the frozen absolute-error cross-column benchmark.

The benchmark is deliberately a thin orchestration layer.  It consumes the
frozen schedule from ``cross_column`` and reuses only artifacts whose source,
canonical data, roles, budget and test population pass an explicit contract
audit.  The one missing primary arm, ``paper_style_current_v2``, is fit from
the qualified current-V2 representation with a target head and a
zero-initialized column-context adapter.

Runtime checkpoints, histories and per-sample predictions are written below
the ignored ``runtime/`` directory.  CSV/JSON/Markdown files at the study
root are compact scientific records only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PARENT = ROOT / "studies/transfer/cross_column"
STUDY = ROOT / "studies/transfer/matched_rmse_benchmark"
SOURCE = ROOT / "studies/predictor/final_4g_qualification/runtime/row/seed_42/best.pt"
SOURCE_SHA256 = "fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0"
SCHEDULE_SHA256 = "b52b937aa750586ec8f2efb2f10f93dfddd2252bb03afaf304556e16b4fab5ee"
COLUMNS = ("8g", "25g", "40g")
PROTOCOLS = ("row", "compound")
SEEDS = (769539383, 1425370602, 536279090, 2767143051, 1362771960)
BUDGETS = (30, 50, 70, 100)
PRIMARY_METHODS = (
    "zero_shot",
    "scale_only",
    "affine",
    "local_identity_shrinkage",
    "conditional_EA",
    "target_head_only",
    "standard_shallow_finetune",
    "paper_style_current_v2",
)
REUSED_METHODS = {
    "zero_shot": "cross_column",
    "scale_only": "cross_column",
    "affine": "cross_column",
    "target_head_only": "cross_column",
    "local_identity_shrinkage": "source_anchored_shared_transfer",
    "conditional_EA": "source_anchored_shared_transfer",
    "standard_shallow_finetune": "source_anchored_shared_transfer",
}
TARGETS = ("V1", "V2")
CONTEXT_NAMES = ("mass_ratio_to_4g", "is_25g", "is_40g")
PAPER_CONFIG = {
    "scope": "current_v2_shallow_final_layer_condition_head_plus_column_adapter",
    "context_dim": 3,
    "context_names": list(CONTEXT_NAMES),
    "context_formula": "[target_mass_g / 4, 1(column=25g), 1(column=40g)]",
    "column_spec_provenance": "COLUMN_ID_AND_MASS_FROM_CANONICAL_COLUMN_SPEC; no legacy tuple used",
    "legacy_tuple_status": "REPOSITORY_LEGACY_COLUMN_SPEC_NOT_USED",
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "maximum_epochs": 500,
    "patience": 100,
    "batch_size": 2048,
    "cpu_threads": 1,
    "selection": "validation_only_minimum_combined_source_normalized_rmse",
}
DECISION_RULES = {
    "material_paper_style_gain": (
        "paper_style_current_v2 must improve both B100 combined normalized RMSE "
        "and label-efficiency AULC by >=5% versus each preregistered simple "
        "reference, with >=4/5 paired seed wins, in both 25g and 40g"
    ),
    "small_incremental_gain": "positive paired mean gain below 5%",
    "tail_dominance": (
        "a high-tail stratum is called dominant only when its mean share of "
        "total target SSE is >=50%; otherwise its contribution is reported "
        "without calling it the primary source of total RMSE"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_parent_protocol() -> tuple[dict, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Validate the already-frozen parent protocol without rerandomizing it."""

    protocol_path = PARENT / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _assert(protocol["source_checkpoint_sha256"] == SOURCE_SHA256, "frozen source SHA changed")
    _assert(sha256_file(SOURCE) == SOURCE_SHA256, "source checkpoint hash mismatch")
    schedule_path = PARENT / "splits/schedule_manifest.csv"
    _assert(protocol["schedule_sha256"] == SCHEDULE_SHA256, "frozen schedule record changed")
    _assert(sha256_file(schedule_path) == SCHEDULE_SHA256, "frozen schedule hash mismatch")
    _assert(protocol.get("target_threshold") is None, "matched benchmark requires no target threshold")
    _assert(protocol.get("budget_definition") == "gradient_train plus validation revealed target rows", "budget contract changed")
    _assert(tuple(protocol.get("columns_g", {})) == COLUMNS, "frozen columns changed")
    _assert(tuple(protocol.get("protocols", ())) == PROTOCOLS, "frozen split protocols changed")
    _assert(tuple(protocol.get("outer_seeds", ())) == SEEDS, "frozen outer seeds changed")
    _assert(tuple(protocol.get("planned_budgets", ())) == BUDGETS, "frozen budgets changed")
    _assert(protocol.get("neural_selection") == "validation only", "neural selection contract changed")
    _assert(protocol.get("simple_fit") == "gradient_train only", "simple-fit contract changed")
    _assert(protocol.get("test_tuning") is False, "test tuning must remain disabled")
    schedule = pd.read_csv(schedule_path)
    _assert(not {"V1_ml", "V2_ml"}.intersection(schedule.columns), "schedule must not expose truth labels")
    grouped = schedule.groupby(["column", "protocol", "outer_seed", "planned_budget"], sort=False)
    _assert(len(grouped) == 120, f"expected 120 frozen contexts, found {len(grouped)}")
    for key, frame in grouped:
        roles = {role: set(frame.loc[frame.role.eq(role), "sample_id"].astype(str)) for role in ("gradient_train", "validation", "test", "pool")}
        _assert(all(roles.values()), f"empty role in {key}")
        _assert(sum(map(len, roles.values())) == len(frame), f"duplicate role IDs in {key}")
        _assert(int(frame.actual_budget.iloc[0]) == len(roles["gradient_train"]) + len(roles["validation"]), f"budget mismatch in {key}")
        _assert(not (roles["gradient_train"] | roles["validation"]) & roles["test"], f"test leakage in {key}")
    canonical: dict[str, pd.DataFrame] = {}
    for column in COLUMNS:
        path = PARENT / "data_audit" / f"canonical_{column}.csv"
        _assert(sha256_file(path) == protocol["canonical_sha256"][column], f"canonical hash mismatch: {column}")
        frame = pd.read_csv(path).reset_index(drop=True)
        _assert(frame.sample_id.astype(str).is_unique, f"duplicate sample IDs: {column}")
        _assert({"V1_ml", "V2_ml"}.issubset(frame.columns), f"missing target labels: {column}")
        canonical[column] = frame
    return protocol, schedule, canonical


def build_protocol(parent: dict) -> dict:
    """Freeze the matched design before any new paper-style test result exists."""

    protocol = {
        "study_name": "MATCHED_CROSS_COLUMN_ABSOLUTE_ERROR_BENCHMARK",
        "classification": "STRATEGY_MATCHED_ABSOLUTE_ERROR_BENCHMARK",
        "parent_protocol": str((PARENT / "protocol.json").relative_to(ROOT)),
        "parent_protocol_sha256": sha256_file(PARENT / "protocol.json"),
        "source_checkpoint": str(SOURCE.relative_to(ROOT)),
        "source_checkpoint_sha256": SOURCE_SHA256,
        "columns": list(COLUMNS),
        "protocols": list(PROTOCOLS),
        "outer_seeds": list(SEEDS),
        "budgets": list(BUDGETS),
        "budget_definition": "gradient_train + validation revealed target rows",
        "target_threshold": None,
        "schedule": str((PARENT / "splits/schedule_manifest.csv").relative_to(ROOT)),
        "schedule_sha256": SCHEDULE_SHA256,
        "canonical_sha256": parent["canonical_sha256"],
        "methods": list(PRIMARY_METHODS),
        "primary_metrics": [
            "V1_rmse", "V1_mae", "V1_r2", "V2_rmse", "V2_mae", "V2_r2",
            "combined_normalized_rmse", "nrmse", "aulc",
        ],
        "simple_fit_labels": "gradient_train only",
        "neural_fit_labels": "gradient_train; frozen validation only for checkpoint selection",
        "test_labels_used_for_fit_or_selection": False,
        "test_tuning": False,
        "global_test_evaluation_boundary": "all 120 paper-style predictions freeze before any new test truth is read",
        "preprocessing": "qualified source checkpoint preprocessing; source_train fit only",
        "tail_definition": {
            "threshold_source": "target gradient_train labels within each seed/budget context",
            "quantiles": {"q50": 0.5, "q80": 0.8},
            "strata": {"low": "y <= q50", "mid": "q50 < y <= q80", "high_tail": "y > q80"},
            "test_truth_used_only_after_freeze": True,
        },
        "paper_style_current_v2": PAPER_CONFIG,
        "decision_rules": DECISION_RULES,
        "status": "FROZEN_BEFORE_NEW_TEST_EVALUATION",
    }
    path = STUDY / "protocol.json"
    if path.exists():
        _assert(json.loads(path.read_text(encoding="utf-8")) == protocol, "matched protocol is frozen and cannot change")
    else:
        atomic_json(path, protocol)
    return protocol


def _context(schedule: pd.DataFrame, column: str, protocol: str, seed: int, budget: int) -> pd.DataFrame:
    frame = schedule.loc[
        schedule.column.eq(column)
        & schedule.protocol.eq(protocol)
        & schedule.outer_seed.eq(seed)
        & schedule.planned_budget.eq(budget)
    ].copy()
    _assert(len(frame) > 0, f"missing schedule context: {column}/{protocol}/{seed}/{budget}")
    _assert(frame.sample_id.astype(str).is_unique, "duplicate context sample IDs")
    return frame


def validate_reuse_provenance(parent: dict) -> dict[str, str]:
    """Verify that candidate reused studies share all frozen parent anchors."""

    anchored_path = ROOT / "studies/transfer/source_anchored_shared_transfer/protocol.json"
    anchored = json.loads(anchored_path.read_text(encoding="utf-8"))
    hashes = anchored.get("hashes", {})
    _assert(hashes.get(str(SOURCE.relative_to(ROOT))) == SOURCE_SHA256, "source-anchored source hash is not matched")
    schedule_path = str((PARENT / "splits/schedule_manifest.csv").relative_to(ROOT))
    _assert(hashes.get(schedule_path) == SCHEDULE_SHA256, "source-anchored schedule is not matched")
    for column in COLUMNS:
        canonical_path = str((PARENT / "data_audit" / f"canonical_{column}.csv").relative_to(ROOT))
        _assert(hashes.get(canonical_path) == parent["canonical_sha256"][column], f"source-anchored canonical mismatch: {column}")
    _assert(tuple(anchored.get("columns", ())) == COLUMNS, "source-anchored columns are not matched")
    _assert(tuple(anchored.get("seeds", ())) == SEEDS, "source-anchored seeds are not matched")
    _assert(tuple(anchored.get("budgets", ())) == BUDGETS, "source-anchored budgets are not matched")
    _assert(anchored.get("test_used_for_selection") is False, "source-anchored protocol used test selection")
    _assert(anchored.get("donor_target_labels") == 0, "source-anchored protocol used donor labels")
    return {
        "parent_stable_hash": stable_hash(parent),
        "source_anchored_protocol_sha256": sha256_file(anchored_path),
    }


def _read_reused_predictions(
    column: str,
    protocol: str,
    seed: int,
    budget: int,
    schedule: pd.DataFrame,
    reuse_contract: dict[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Read and identity-align exact matched predictions from frozen studies."""

    frame = _context(schedule, column, protocol, seed, budget)
    test_ids = frame.loc[frame.role.eq("test"), "sample_id"].astype(str).tolist()
    cross_dir = PARENT / column / protocol / f"seed_{seed}" / f"budget_{budget}"
    completion_path = cross_dir / "completion.json"
    _assert(completion_path.exists(), f"missing cross-column completion: {completion_path}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    expected_completion = {
        "protocol_hash": reuse_contract["parent_stable_hash"],
        "column": column,
        "split_protocol": protocol,
        "seed": seed,
        "planned_budget": budget,
        "actual_budget": int(frame.actual_budget.iloc[0]),
    }
    _assert(completion.get("contract") == expected_completion, f"cross-column completion contract mismatch: {cross_dir}")
    for name, digest in completion.get("files", {}).items():
        path = cross_dir / name
        _assert(path.exists() and sha256_file(path) == digest, f"cross-column artifact hash mismatch: {path}")
    cross = pd.read_csv(cross_dir / "predictions.csv.gz").set_index("sample_id")
    _assert(cross.index.astype(str).is_unique, f"duplicate cross-column prediction IDs: {cross_dir}")
    _assert(set(cross.index.astype(str)) == set(test_ids), f"cross-column test population mismatch: {column}/{protocol}/{seed}/{budget}")
    cross = cross.loc[test_ids]
    source_dir = ROOT / "studies/transfer/source_anchored_shared_transfer/contexts" / column / protocol / f"seed_{seed}" / f"budget_{budget}"
    frozen_path = source_dir / "frozen.json"
    _assert(frozen_path.exists(), f"missing source-anchored freeze: {frozen_path}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    _assert(frozen.get("protocol_sha256") == reuse_contract["source_anchored_protocol_sha256"], f"source-anchored protocol mismatch: {source_dir}")
    for name, digest in frozen.get("files", {}).items():
        path = source_dir / name
        _assert(path.exists() and sha256_file(path) == digest, f"source-anchored artifact hash mismatch: {path}")
    usage = json.loads((source_dir / "label_usage.json").read_text(encoding="utf-8"))
    for role in ("gradient_train", "validation", "test"):
        expected = set(frame.loc[frame.role.eq(role), "sample_id"].astype(str))
        _assert(set(map(str, usage[role])) == expected, f"reused label ledger mismatch ({role})")
    _assert(usage.get("other_target_column_labels_used", 0) == 0, "reused fit used donor target labels")
    _assert(usage.get("test_labels_used_for_training_or_selection", 0) == 0, "reused fit used test labels")
    blind = pd.read_csv(source_dir / "predictions_blind.csv.gz").set_index("sample_id")
    _assert(blind.index.astype(str).is_unique, f"duplicate source-anchored prediction IDs: {source_dir}")
    _assert(set(blind.index.astype(str)) == set(test_ids), f"source-anchored test population mismatch: {column}/{protocol}/{seed}/{budget}")
    blind = blind.loc[test_ids]
    predictions = {
        "zero_shot": cross[["zero_shot_V1", "zero_shot_V2"]].to_numpy(float),
        "scale_only": cross[["scale_only_V1", "scale_only_V2"]].to_numpy(float),
        "affine": cross[["affine_V1", "affine_V2"]].to_numpy(float),
        "target_head_only": cross[["target_head_only_V1", "target_head_only_V2"]].to_numpy(float),
        "local_identity_shrinkage": blind[["local_identity_shrinkage_V1", "local_identity_shrinkage_V2"]].to_numpy(float),
        "conditional_EA": blind[["conditional_EA_V1", "conditional_EA_V2"]].to_numpy(float),
        "standard_shallow_finetune": blind[["standard_shallow_finetune_V1", "standard_shallow_finetune_V2"]].to_numpy(float),
    }
    _assert(all(np.isfinite(value).all() for value in predictions.values()), "non-finite reused prediction")
    audit = {
        "status": "REUSED_MATCHED_ARTIFACT",
        "cross_column_completion": str(completion_path.relative_to(ROOT)),
        "source_anchored_freeze": str(frozen_path.relative_to(ROOT)),
        "cross_column_predictions_sha256": sha256_file(cross_dir / "predictions.csv.gz"),
        "source_anchored_predictions_sha256": sha256_file(source_dir / "predictions_blind.csv.gz"),
        "numerical_metric_recheck": "performed_after_identity_alignment",
        "developmental_evidence": True,
        "developmental_caveat": "reused source-anchored evidence was historically test-exposed; this is a matched developmental ranking, not an independent confirmation set",
        "methods": {method: REUSED_METHODS[method] for method in REUSED_METHODS},
    }
    return predictions, audit


def _column_context(column: str, count: int) -> np.ndarray:
    mass = {"8g": 8.0, "25g": 25.0, "40g": 40.0}[column]
    values = np.asarray([mass / 4.0, float(column == "25g"), float(column == "40g")], dtype=np.float32)
    return np.repeat(values[None, :], count, axis=0)


def _source_graph_inputs(
    column: str,
    canonical: pd.DataFrame,
    preprocessing: dict,
) -> tuple[list[object], list[object], np.ndarray]:
    """Build label-scrubbed graphs for paper-style fitting and inference.

    Graph construction requires a ``y`` tensor for API compatibility, but
    target truth is not a model input.  We deliberately replace *every* target
    label with zero here.  The per-context fitter places real values only on
    gradient-train and validation rows, making it impossible for a held-out
    test label to enter its input graph collection.
    """

    source_cache = dict(torch.load(ROOT / "experiments/e0_4g_baseline/graph_cache_4g.pt", weights_only=False))
    target_cache_path = PARENT / "data_audit" / f"graph_cache_{column}_only.pt"
    source_cache.update(torch.load(target_cache_path, weights_only=False))
    from src.qgeognn_al.data import build_model_data
    from src.qgeognn_al.transfer import attach_column_context

    scrubbed = canonical.copy()
    scrubbed[["V1_ml", "V2_ml"]] = 0.0
    atom, angle = build_model_data(scrubbed, source_cache, pd.DataFrame(), preprocessing["scaler"])
    return attach_column_context(atom, _column_context(column, len(atom))), angle, _column_context(column, len(atom))


def _paper_fit(
    column: str,
    protocol: str,
    seed: int,
    budget: int,
    context_frame: pd.DataFrame,
    sample_ids: Iterable[str],
    base_atom: list[object],
    angle: list[object],
    train_truth: np.ndarray,
    validation_truth: np.ndarray,
    preprocessing: dict,
    source_model: torch.nn.Module,
) -> tuple[np.ndarray, dict[str, object], dict[str, object]]:
    """Fit current-V2 shallow paper-style adaptation using train/valid only.

    The interface intentionally does not accept a canonical target frame or a
    test-truth array.  ``base_atom`` was built with zero labels everywhere;
    only copies of revealed train/validation rows receive labels below.
    """

    from src.qgeognn_al.transfer import PaperStyleCurrentV2, loader_pair, quantile_target_loss

    ids = [str(value) for value in sample_ids]
    index = {sample_id: i for i, sample_id in enumerate(ids)}
    _assert(len(index) == len(ids), "duplicate canonical sample ID in paper-style input")
    train_ids = context_frame.loc[context_frame.role.eq("gradient_train"), "sample_id"].astype(str).tolist()
    valid_ids = context_frame.loc[context_frame.role.eq("validation"), "sample_id"].astype(str).tolist()
    test_ids = context_frame.loc[context_frame.role.eq("test"), "sample_id"].astype(str).tolist()
    train_idx = np.asarray([index[value] for value in train_ids], dtype=int)
    valid_idx = np.asarray([index[value] for value in valid_ids], dtype=int)
    test_idx = np.asarray([index[value] for value in test_ids], dtype=int)
    train_truth = np.asarray(train_truth, dtype=np.float32)
    validation_truth = np.asarray(validation_truth, dtype=np.float32)
    _assert(train_truth.shape == (len(train_idx), 2), "paper-style training labels do not align")
    _assert(validation_truth.shape == (len(valid_idx), 2), "paper-style validation labels do not align")
    _assert(np.isfinite(train_truth).all() and np.isfinite(validation_truth).all(), "non-finite revealed target label")
    scales = preprocessing["target_scales"]
    contract = {
        "protocol_sha256": sha256_file(STUDY / "protocol.json"),
        "column": column, "protocol": protocol, "seed": seed, "budget": budget,
        "gradient_train_ids_hash": stable_hash(sorted(train_ids)),
        "validation_ids_hash": stable_hash(sorted(valid_ids)),
        "test_ids_hash": stable_hash(sorted(test_ids)),
        "gradient_train_rows": len(train_ids),
        "validation_rows": len(valid_ids),
        "test_labels_used_for_fit_or_selection": 0,
    }
    runtime = STUDY / "runtime" / "paper_style_current_v2" / column / protocol / f"seed_{seed}" / f"budget_{budget}"
    runtime.mkdir(parents=True, exist_ok=True)
    fit_path = runtime / "fit_audit.json"
    pred_path = runtime / "predictions.csv.gz"
    if fit_path.exists() and pred_path.exists():
        audit = json.loads(fit_path.read_text(encoding="utf-8"))
        if audit.get("contract") != contract:
            raise RuntimeError(f"paper-style runtime contract changed: {runtime}")
        _assert(audit.get("source_checkpoint_sha256") == SOURCE_SHA256, "paper-style source hash changed")
        _assert(audit.get("test_labels_used_for_fit_or_selection") == 0, "paper-style audit reports test leakage")
        _assert(audit.get("prediction_sha256") == sha256_file(pred_path), "paper-style prediction hash mismatch")
        stored = pd.read_csv(pred_path)
        _assert(stored.sample_id.astype(str).is_unique, "duplicate paper-style prediction IDs")
        _assert(stored.sample_id.astype(str).tolist() == test_ids, "paper-style test IDs changed")
        return stored.loc[:, ["V1_pred", "V2_pred"]].to_numpy(float), audit, {"runtime": str(runtime.relative_to(ROOT)), "status": "REUSED_NEW_FIT"}

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(int(PAPER_CONFIG["cpu_threads"]))
    model = PaperStyleCurrentV2(source_model, context_dim=3, scope="shallow")
    # This is intentionally the current-V2 shallow scope: the final message
    # layer, condition completion, transferred target head and new context
    # adapter update; the source head remains frozen as an audit reference.
    _assert(not any(parameter.requires_grad for parameter in model.source_head.parameters()), "source head must remain frozen")
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=PAPER_CONFIG["learning_rate"], weight_decay=PAPER_CONFIG["weight_decay"],
    )
    # Clone the zero-labelled graph inputs and reveal labels only in the two
    # authorized roles.  Test graph labels stay at their hard zero sentinel.
    atom = [value.clone() for value in base_atom]
    for row, value in zip(train_idx, train_truth):
        atom[int(row)].y = torch.tensor([value], dtype=torch.float32)
    for row, value in zip(valid_idx, validation_truth):
        atom[int(row)].y = torch.tensor([value], dtype=torch.float32)
    best_score, best_epoch, stale = float("inf"), 0, 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, int(PAPER_CONFIG["maximum_epochs"]) + 1):
        model.training_target()
        losses: list[float] = []
        order = np.random.default_rng(seed * 10000 + epoch).permutation(train_idx)
        for atom_batch, angle_batch in zip(*loader_pair(atom, angle, order, int(PAPER_CONFIG["batch_size"]))):
            prediction = model(atom_batch, angle_batch)
            loss = quantile_target_loss(atom_batch.y[:, 0], prediction[:, :3]) + quantile_target_loss(atom_batch.y[:, 1], prediction[:, 3:])
            _assert(bool(torch.isfinite(loss)), "non-finite paper-style training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            valid_prediction = [model(atom_batch, angle_batch).cpu().numpy() for atom_batch, angle_batch in zip(*loader_pair(atom, angle, valid_idx, int(PAPER_CONFIG["batch_size"])))]
            valid_point = np.vstack(valid_prediction)[:, [1, 4]]
        valid_error = validation_truth - valid_point
        score = float(np.sqrt(np.mean(np.square(valid_error / np.asarray([scales["V1"], scales["V2"]])))))
        _assert(math.isfinite(score), "non-finite paper-style validation score")
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)), "validation_combined_normalized_rmse": score})
        if score < best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= int(PAPER_CONFIG["patience"]):
            break
    _assert(best_state is not None, "paper-style fit did not produce a validation checkpoint")
    model.load_state_dict(best_state, strict=False)
    model.eval()
    with torch.no_grad():
        test_prediction = [model(atom_batch, angle_batch).cpu().numpy() for atom_batch, angle_batch in zip(*loader_pair(atom, angle, test_idx, int(PAPER_CONFIG["batch_size"])))]
    point = np.vstack(test_prediction)[:, [1, 4]]
    pd.DataFrame({"sample_id": test_ids, "V1_pred": point[:, 0], "V2_pred": point[:, 1]}).to_csv(pred_path, index=False, compression={"method": "gzip", "mtime": 0})
    torch.save({"source_checkpoint_sha256": SOURCE_SHA256, "protocol_sha256": contract["protocol_sha256"], "state_dict": best_state, "best_epoch": best_epoch}, runtime / "best.pt")
    audit = {
        "method": "paper_style_current_v2", "column": column, "protocol": protocol, "seed": seed, "budget": budget,
        "contract": contract, "best_epoch": best_epoch, "epochs_run": len(history),
        "validation_score": best_score, "trainable_parameters": trainable, "total_parameters": total,
        "trainable_scope": "final message layer + condition branch + target head + column adapter; source head frozen",
        "source_checkpoint_sha256": SOURCE_SHA256, "column_context": PAPER_CONFIG,
        "history": history, "test_labels_used_for_fit_or_selection": 0,
        "test_graph_labels": "zero_sentinel; test truth was not loaded by this fit",
        "prediction_sha256": sha256_file(pred_path),
    }
    atomic_json(fit_path, audit)
    return point, audit, {"runtime": str(runtime.relative_to(ROOT)), "status": "RERUN_MATCHED_ARTIFACT"}


def _metrics(truth: np.ndarray, prediction: np.ndarray, scales: dict[str, float]) -> dict[str, float]:
    from src.qgeognn_al.transfer import absolute_error_metrics
    return absolute_error_metrics(truth, prediction, scales)


def _load_frozen_paper_prediction(
    column: str,
    split_protocol: str,
    seed: int,
    budget: int,
    context: pd.DataFrame,
) -> np.ndarray:
    """Verify a frozen paper prediction without accessing target truth."""

    train_ids = context.loc[context.role.eq("gradient_train"), "sample_id"].astype(str).tolist()
    valid_ids = context.loc[context.role.eq("validation"), "sample_id"].astype(str).tolist()
    test_ids = context.loc[context.role.eq("test"), "sample_id"].astype(str).tolist()
    expected_contract = {
        "protocol_sha256": sha256_file(STUDY / "protocol.json"),
        "column": column, "protocol": split_protocol, "seed": seed, "budget": budget,
        "gradient_train_ids_hash": stable_hash(sorted(train_ids)),
        "validation_ids_hash": stable_hash(sorted(valid_ids)),
        "test_ids_hash": stable_hash(sorted(test_ids)),
        "gradient_train_rows": len(train_ids),
        "validation_rows": len(valid_ids),
        "test_labels_used_for_fit_or_selection": 0,
    }
    runtime = STUDY / "runtime" / "paper_style_current_v2" / column / split_protocol / f"seed_{seed}" / f"budget_{budget}"
    audit_path, prediction_path = runtime / "fit_audit.json", runtime / "predictions.csv.gz"
    _assert(audit_path.exists() and prediction_path.exists(), f"missing frozen paper-style runtime: {runtime}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    _assert(audit.get("contract") == expected_contract, f"paper-style freeze contract mismatch: {runtime}")
    _assert(audit.get("source_checkpoint_sha256") == SOURCE_SHA256, f"paper-style source mismatch: {runtime}")
    _assert(audit.get("test_labels_used_for_fit_or_selection") == 0, f"paper-style test leakage: {runtime}")
    _assert(audit.get("prediction_sha256") == sha256_file(prediction_path), f"paper-style prediction hash mismatch: {runtime}")
    frame = pd.read_csv(prediction_path)
    _assert(frame.sample_id.astype(str).is_unique, f"duplicate paper-style prediction IDs: {runtime}")
    _assert(frame.sample_id.astype(str).tolist() == test_ids, f"paper-style prediction IDs mismatch: {runtime}")
    values = frame[["V1_pred", "V2_pred"]].to_numpy(float)
    _assert(np.isfinite(values).all(), f"non-finite paper-style predictions: {runtime}")
    return values


def execute_paper_style(protocol: dict, schedule: pd.DataFrame, canonical: dict[str, pd.DataFrame]) -> tuple[list[dict], dict]:
    """Freeze the missing paper-style arm before reading any test truth."""

    del protocol  # The file is verified by each per-context contract.
    source_payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    preprocessing = source_payload["preprocessing"]
    from src.qgeognn_al.models import load_predictor_checkpoint
    source_model = load_predictor_checkpoint(SOURCE)
    atom_inputs: dict[str, list[object]] = {}
    angle_inputs: dict[str, list[object]] = {}
    for column in COLUMNS:
        atom_inputs[column], angle_inputs[column], _ = _source_graph_inputs(column, canonical[column], preprocessing)
    rows: list[dict] = []
    audit: dict[str, object] = {
        "method": "paper_style_current_v2",
        "contexts": {},
        "fit_phase": "TEST_TRUTH_NOT_READ",
        "status": "PREDICTIONS_NOT_YET_GLOBALLY_FROZEN",
    }
    for column in COLUMNS:
        labels = canonical[column].set_index("sample_id")
        for split_protocol in PROTOCOLS:
            for seed in SEEDS:
                for budget in BUDGETS:
                    context = _context(schedule, column, split_protocol, seed, budget)
                    train_ids = context.loc[context.role.eq("gradient_train"), "sample_id"].astype(str).tolist()
                    valid_ids = context.loc[context.role.eq("validation"), "sample_id"].astype(str).tolist()
                    # These are the only target-label reads during the fit phase.
                    train_truth = labels.loc[train_ids, ["V1_ml", "V2_ml"]].to_numpy(float)
                    validation_truth = labels.loc[valid_ids, ["V1_ml", "V2_ml"]].to_numpy(float)
                    prediction, fit_audit, context_audit = _paper_fit(
                        column, split_protocol, seed, budget, context,
                        canonical[column].sample_id.astype(str), atom_inputs[column], angle_inputs[column],
                        train_truth, validation_truth, preprocessing, source_model,
                    )
                    print(
                        f"paper-style {column}/{split_protocol}/seed={seed}/B={budget}: "
                        f"{context_audit['status']} epoch={fit_audit['best_epoch']}",
                        flush=True,
                    )
                    rows.append({"column": column, "protocol": split_protocol, "seed": seed, "budget": budget})
                    audit["contexts"][f"{column}/{split_protocol}/{seed}/{budget}"] = {
                        "fit": {key: value for key, value in fit_audit.items() if key not in {"history"}},
                        "runtime": context_audit,
                    }
    _assert(len(rows) == len(COLUMNS) * len(PROTOCOLS) * len(SEEDS) * len(BUDGETS), "paper-style freeze is incomplete")
    audit["status"] = "PREDICTIONS_GLOBALLY_FROZEN_BEFORE_TEST_EVALUATION"
    audit["prediction_context_count"] = len(rows)
    audit["test_labels_used_for_fit_or_selection"] = 0
    atomic_json(STUDY / "paper_style_audit.json", audit)
    return rows, audit


def _tail_rows_for_context(column: str, split_protocol: str, seed: int, budget: int, context: pd.DataFrame, canonical: pd.DataFrame, predictions: dict[str, np.ndarray]) -> list[dict]:
    from src.qgeognn_al.transfer import tail_error_rows
    train_ids = context.loc[context.role.eq("gradient_train"), "sample_id"].astype(str).tolist()
    test_ids = context.loc[context.role.eq("test"), "sample_id"].astype(str).tolist()
    indexed = canonical.set_index("sample_id")
    train_truth = indexed.loc[train_ids, ["V1_ml", "V2_ml"]].to_numpy(float)
    test_truth = indexed.loc[test_ids, ["V1_ml", "V2_ml"]].to_numpy(float)
    return tail_error_rows(
        train_truth,
        test_truth,
        predictions,
        column=column,
        protocol=split_protocol,
        seed=seed,
        budget=budget,
    )


def build_records(protocol: dict, schedule: pd.DataFrame, canonical: dict[str, pd.DataFrame], paper_rows: list[dict] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """Recompute all metrics from frozen predictions and build tail diagnostics."""

    source_payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    scales = source_payload["preprocessing"]["target_scales"]
    paper_by_key = {(row["column"], row["protocol"], int(row["seed"]), int(row["budget"])): row for row in (paper_rows or [])}
    parent = json.loads((PARENT / "protocol.json").read_text(encoding="utf-8"))
    reuse_contract = validate_reuse_provenance(parent)
    metric_rows: list[dict] = []
    tail_rows: list[dict] = []
    prediction_rows: list[dict] = []
    target_distribution_rows: list[dict] = []
    context_audit: dict[str, object] = {}
    for column in COLUMNS:
        for split_protocol in PROTOCOLS:
            for seed in SEEDS:
                for budget in BUDGETS:
                    context = _context(schedule, column, split_protocol, seed, budget)
                    reused, audit = _read_reused_predictions(column, split_protocol, seed, budget, schedule, reuse_contract)
                    key = (column, split_protocol, seed, budget)
                    predictions = dict(reused)
                    if key in paper_by_key:
                        predictions["paper_style_current_v2"] = _load_frozen_paper_prediction(
                            column, split_protocol, seed, budget, context
                        )
                    else:
                        raise RuntimeError(f"missing paper-style result for {key}; run --execute first")
                    test_ids = context.loc[context.role.eq("test"), "sample_id"].astype(str).tolist()
                    truth = canonical[column].set_index("sample_id").loc[test_ids, ["V1_ml", "V2_ml"]].to_numpy(float)
                    for method in PRIMARY_METHODS:
                        metric = _metrics(truth, predictions[method], scales)
                        metric_rows.append({"column": column, "protocol": split_protocol, "seed": seed, "budget": budget,
                                            "actual_budget": int(context.actual_budget.iloc[0]), "method": method, "artifact_status": "RERUN_MATCHED_ARTIFACT" if method == "paper_style_current_v2" else "REUSED_MATCHED_ARTIFACT", **metric})
                        if budget == 100:
                            for sample_id, actual, predicted in zip(test_ids, truth, predictions[method]):
                                prediction_rows.append({
                                    "column": column, "protocol": split_protocol, "seed": seed, "budget": budget,
                                    "method": method, "sample_id": sample_id,
                                    "V1_true": float(actual[0]), "V2_true": float(actual[1]),
                                    "V1_pred": float(predicted[0]), "V2_pred": float(predicted[1]),
                                })
                    if budget == 100:
                        for target_index, target in enumerate(TARGETS):
                            values = truth[:, target_index]
                            target_distribution_rows.append({
                                "column": column, "protocol": split_protocol, "seed": seed, "budget": budget,
                                "target": target, "n": len(values), "mean": float(values.mean()),
                                "sample_sd": float(values.std(ddof=1)), "variance": float(values.var(ddof=1)),
                                "min": float(values.min()), "median": float(np.median(values)), "max": float(values.max()),
                            })
                    tail_rows.extend(_tail_rows_for_context(column, split_protocol, seed, budget, context, canonical[column], predictions))
                    context_audit[f"{column}/{split_protocol}/{seed}/{budget}"] = audit
    metrics = pd.DataFrame(metric_rows)
    tails = pd.DataFrame(tail_rows)
    return metrics, tails, context_audit, pd.DataFrame(prediction_rows), pd.DataFrame(target_distribution_rows)


def _summary_row(values: pd.Series, prefix: str) -> dict[str, float]:
    """Return the explicitly requested five-seed descriptive statistics."""

    values = values.astype(float)
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_sample_sd": float(values.std(ddof=1)),
        f"{prefix}_median": float(values.median()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
    }


def _paired_stats(candidate: pd.Series, reference: pd.Series, prefix: str) -> dict[str, float | int]:
    """Summarize paired candidate-minus-reference values in matched seed order."""

    _assert(candidate.index.equals(reference.index), f"unmatched paired seeds for {prefix}")
    values = candidate.to_numpy(float)
    baseline = reference.to_numpy(float)
    delta = values - baseline
    return {
        f"{prefix}_paired_mean_delta": float(delta.mean()),
        f"{prefix}_paired_median_delta": float(np.median(delta)),
        f"{prefix}_wins": int((delta < 0).sum()),
        f"{prefix}_relative_gain": float(1.0 - values.mean() / baseline.mean()),
        f"{prefix}_relative_percent_gain": float(100.0 * (1.0 - values.mean() / baseline.mean())),
    }


def _tail_summary(tails: pd.DataFrame) -> pd.DataFrame:
    """Summarize predeclared volume strata with five-seed descriptive stats."""

    rows: list[dict] = []
    group_cols = ["column", "protocol", "budget", "method", "target", "stratum"]
    for keys, group in tails.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys))
        row["seeds"] = int(group.seed.nunique())
        for name in ("n", "rmse", "mae", "squared_error_fraction", "train_q50", "train_q80"):
            row.update(_summary_row(group[name], name))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(
    metrics: pd.DataFrame,
    tails: pd.DataFrame,
    prediction_rows: pd.DataFrame,
    target_distributions: pd.DataFrame,
    protocol: dict,
) -> dict[str, pd.DataFrame]:
    """Write compact primary tables, paired comparisons and explanatory plots."""

    del protocol
    STUDY.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(STUDY / "all_metrics.csv", index=False)
    tails.to_csv(STUDY / "tail_error_metrics.csv", index=False)
    target_distributions.to_csv(STUDY / "test_target_distribution_by_seed.csv", index=False)
    test = metrics.loc[metrics.budget.eq(100)].copy()
    stat_rows: list[dict] = []
    group_cols = ["column", "protocol", "method"]
    value_cols = ["V1_rmse", "V2_rmse", "V1_mae", "V2_mae", "V1_r2", "V2_r2", "combined_normalized_rmse", "nrmse"]
    for keys, group in test.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys)); row["seeds"] = int(group.seed.nunique())
        for name in value_cols:
            row.update(_summary_row(group[name], name))
        stat_rows.append(row)
    budget100 = pd.DataFrame(stat_rows)
    budget100.to_csv(STUDY / "budget100_summary.csv", index=False)

    aulc_rows: list[dict] = []
    for keys, group in metrics.groupby(["column", "protocol", "seed", "method"], sort=False):
        group = group.sort_values("budget")
        _assert(tuple(group.budget.astype(int)) == BUDGETS, f"incomplete AULC grid: {keys}")
        score = group.nrmse.to_numpy(float)
        area = np.trapezoid(score, group.budget.to_numpy(float)) if hasattr(np, "trapezoid") else np.trapz(score, group.budget.to_numpy(float))
        aulc_rows.append({
            "column": keys[0], "protocol": keys[1], "seed": int(keys[2]), "method": keys[3],
            "aulc": float(area / (max(BUDGETS) - min(BUDGETS))),
            "planned_budget_min": int(group.budget.min()), "planned_budget_max": int(group.budget.max()),
        })
    aulc_by_seed = pd.DataFrame(aulc_rows)
    aulc_by_seed.to_csv(STUDY / "aulc_by_seed.csv", index=False)
    aulc_rows_summary: list[dict] = []
    for keys, group in aulc_by_seed.groupby(["column", "protocol", "method"], sort=False):
        row = dict(zip(["column", "protocol", "method"], keys)); row["seeds"] = int(group.seed.nunique())
        row.update(_summary_row(group.aulc, "aulc")); aulc_rows_summary.append(row)
    aulc_summary = pd.DataFrame(aulc_rows_summary)
    aulc_summary.to_csv(STUDY / "aulc_summary.csv", index=False)

    pairs: list[dict] = []
    references = ("scale_only", "affine", "local_identity_shrinkage")
    for (column, split_protocol), group in metrics.groupby(["column", "protocol"], sort=False):
        b100 = group.loc[group.budget.eq(100)]
        aulc_group = aulc_by_seed.loc[(aulc_by_seed.column.eq(column)) & (aulc_by_seed.protocol.eq(split_protocol))]
        b100_by_metric = {name: b100.pivot(index="seed", columns="method", values=name).sort_index() for name in ("V1_rmse", "V2_rmse", "combined_normalized_rmse", "nrmse")}
        aulc_pivot = aulc_group.pivot(index="seed", columns="method", values="aulc").sort_index()
        _assert(
            len(aulc_pivot.index) == len(SEEDS) and set(aulc_pivot.index.astype(int)) == set(SEEDS),
            f"missing AULC seed: {column}/{split_protocol}",
        )
        for method in PRIMARY_METHODS:
            for reference in references:
                if method == reference:
                    continue
                row: dict[str, object] = {"column": column, "protocol": split_protocol, "method": method, "reference": reference, "seeds": len(SEEDS)}
                for name, pivot in b100_by_metric.items():
                    _assert(method in pivot and reference in pivot, f"missing paired B100 method: {method}/{reference}")
                    row.update(_paired_stats(pivot[method], pivot[reference], name))
                row.update(_paired_stats(aulc_pivot[method], aulc_pivot[reference], "aulc"))
                relative = float(row["aulc_relative_gain"])
                wins = int(row["aulc_wins"])
                row["interpretation"] = (
                    "replicated_material_gain" if relative >= 0.05 and wins >= 4 else
                    "small_incremental_gain" if 0 < relative < 0.05 else
                    "not_replicated_across_seeds" if relative > 0 else
                    "no_gain_vs_reference"
                )
                pairs.append(row)
    paired = pd.DataFrame(pairs)
    paired.to_csv(STUDY / "paired_comparisons.csv", index=False)

    tail_summary = _tail_summary(tails)
    tail_summary.to_csv(STUDY / "tail_error_summary.csv", index=False)
    distribution_summary_rows: list[dict] = []
    for keys, group in target_distributions.groupby(["column", "protocol", "target"], sort=False):
        row = dict(zip(["column", "protocol", "target"], keys)); row["seeds"] = int(group.seed.nunique())
        for name in ("mean", "sample_sd", "variance", "min", "median", "max"):
            row.update(_summary_row(group[name], f"test_{name}"))
        distribution_summary_rows.append(row)
    distribution_summary = pd.DataFrame(distribution_summary_rows)
    distribution_summary.to_csv(STUDY / "test_target_distribution_summary.csv", index=False)
    _write_plots(metrics, tails, prediction_rows)
    return {
        "budget100": budget100,
        "aulc_by_seed": aulc_by_seed,
        "aulc_summary": aulc_summary,
        "paired": paired,
        "tails": tails,
        "tail_summary": tail_summary,
        "target_distribution_summary": distribution_summary,
    }


def _write_plots(metrics: pd.DataFrame, tails: pd.DataFrame, predictions: pd.DataFrame) -> None:
    """Produce predeclared descriptive plots from globally frozen predictions."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = STUDY / "plots"; plots.mkdir(parents=True, exist_ok=True)
    b100 = metrics.loc[metrics.budget.eq(100)]
    # 1. Method x column B100 RMSE comparison, retaining both targets.
    for split_protocol in PROTOCOLS:
        frame = b100.loc[b100.protocol.eq(split_protocol)].groupby(["column", "method"])[["V1_rmse", "V2_rmse"]].mean().reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
        x = np.arange(len(COLUMNS)); width = .8 / len(PRIMARY_METHODS)
        for target, ax in zip(TARGETS, axes):
            for index, method in enumerate(PRIMARY_METHODS):
                values = [float(frame.loc[(frame.column.eq(column)) & (frame.method.eq(method)), f"{target}_rmse"].iloc[0]) for column in COLUMNS]
                ax.bar(x + (index - (len(PRIMARY_METHODS) - 1) / 2) * width, values, width, label=method)
            ax.set_xticks(x, COLUMNS); ax.set_ylabel(f"{target} RMSE (mL)"); ax.set_title(f"B=100 {target}: {split_protocol}")
        axes[-1].legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(plots / f"method_column_b100_rmse_{split_protocol}.png", dpi=160); plt.close(fig)
    # 2. Absolute RMSE budget curves (not a relabelled NRMSE curve).
    for column in COLUMNS:
        for split_protocol in PROTOCOLS:
            frame = metrics.loc[(metrics.column.eq(column)) & (metrics.protocol.eq(split_protocol))]
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            for target, ax in zip(TARGETS, axes):
                grouped = frame.groupby(["budget", "method"])[f"{target}_rmse"].mean().reset_index()
                for method in PRIMARY_METHODS:
                    part = grouped.loc[grouped.method.eq(method)].sort_values("budget")
                    ax.plot(part.budget, part[f"{target}_rmse"], marker="o", label=method)
                ax.set_xlabel("purchased target labels (train + validation)"); ax.set_ylabel(f"{target} RMSE (mL)"); ax.set_title(f"{column}/{split_protocol}")
            axes[-1].legend(fontsize=7); fig.tight_layout(); fig.savefig(plots / f"rmse_vs_budget_{column}_{split_protocol}.png", dpi=160); plt.close(fig)
    # 3. Low/mid/high RMSE strata at B100.
    for column in COLUMNS:
        for split_protocol in PROTOCOLS:
            frame = tails.loc[(tails.column.eq(column)) & (tails.protocol.eq(split_protocol)) & (tails.budget.eq(100))]
            fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
            for target, ax in zip(TARGETS, axes):
                subset = frame.loc[frame.target.eq(target)].groupby(["stratum", "method"]).rmse.mean().reset_index()
                for method in PRIMARY_METHODS:
                    part = subset.loc[subset.method.eq(method)].set_index("stratum").reindex(("low", "mid", "high_tail"))
                    ax.plot(("low", "mid", "high-tail"), part.rmse, marker="o", label=method)
                ax.set_ylabel(f"{target} RMSE (mL)"); ax.set_title(f"Volume strata: {column}/{split_protocol}")
            axes[-1].legend(fontsize=7); fig.tight_layout(); fig.savefig(plots / f"rmse_by_volume_stratum_{column}_{split_protocol}.png", dpi=160); plt.close(fig)
    # 4--6. Predicted-vs-true, residual-vs-true and absolute-error-vs-true.
    for column in COLUMNS:
        for split_protocol in PROTOCOLS:
            frame = predictions.loc[(predictions.column.eq(column)) & (predictions.protocol.eq(split_protocol))]
            for target in TARGETS:
                true_name, pred_name = f"{target}_true", f"{target}_pred"
                fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
                lower = float(min(frame[true_name].min(), frame[pred_name].min())); upper = float(max(frame[true_name].max(), frame[pred_name].max()))
                for method in PRIMARY_METHODS:
                    part = frame.loc[frame.method.eq(method)]
                    residual = part[pred_name] - part[true_name]
                    axes[0].scatter(part[true_name], part[pred_name], s=9, alpha=.25, label=method)
                    axes[1].scatter(part[true_name], residual, s=9, alpha=.25, label=method)
                    axes[2].scatter(part[true_name], np.abs(residual), s=9, alpha=.25, label=method)
                axes[0].plot((lower, upper), (lower, upper), "k--", linewidth=1); axes[0].set(xlabel="True volume (mL)", ylabel="Predicted volume (mL)", title="Predicted vs true")
                axes[1].axhline(0, color="k", linewidth=1); axes[1].set(xlabel="True volume (mL)", ylabel="Prediction residual (mL)", title="Residual vs true")
                axes[2].set(xlabel="True volume (mL)", ylabel="Absolute error (mL)", title="Absolute error vs true")
                axes[-1].legend(fontsize=6, ncol=2); fig.suptitle(f"B=100 {column}/{split_protocol}/{target}"); fig.tight_layout(); fig.savefig(plots / f"prediction_diagnostics_{column}_{split_protocol}_{target}.png", dpi=160); plt.close(fig)


def write_audits(protocol: dict, context_audit: dict, paper_audit: dict) -> None:
    historical = {
        "paper_transfer_reproduction": {
            "status": "NOT_PROTOCOL_MATCHED",
            "classification": "PAPER_ALIGNED_RECONSTRUCTED_REPRODUCTION",
            "reasons": ["legacy V1/V2 threshold filtering", "larger target-label fraction", "old E0 source checkpoint", "different split/test population"],
            "ranking_eligible": False,
        },
        "physics_column_conditioned_transfer": {
            "status": "PROTOCOL_MATCHED_APPENDIX_ELIGIBLE",
            "reasons": [
                "current source, no-threshold canonical data, frozen schedule and equal per-column budget are matched",
                "physics/context architecture is not a preregistered primary strategy and is retained outside the primary ranking",
                "physical metadata remains an engineering proxy rather than verified geometry",
            ],
            "ranking_eligible": False,
        },
        "residual_diagnostics": {"status": "NOT_PROTOCOL_MATCHED", "reasons": ["historical protocol and test exposure metadata"], "ranking_eligible": False},
    }
    audit = {
        "schema_version": 1,
        "study": protocol["study_name"],
        "protocol_sha256": sha256_file(STUDY / "protocol.json"),
        "source_checkpoint": protocol["source_checkpoint"],
        "source_checkpoint_sha256": sha256_file(SOURCE),
        "schedule_sha256": sha256_file(PARENT / "splits/schedule_manifest.csv"),
        "contexts": 120,
        "methods": {
            method: {
                "status": "RERUN_MATCHED_ARTIFACT" if method == "paper_style_current_v2" else "REUSED_MATCHED_ARTIFACT",
                "source": "new current-V2 shallow paper-style fit" if method == "paper_style_current_v2" else REUSED_METHODS[method],
                "developmental_evidence": method in {"local_identity_shrinkage", "conditional_EA", "standard_shallow_finetune"},
            }
            for method in PRIMARY_METHODS
        },
        "context_artifacts": context_audit,
        "paper_style": paper_audit,
        "historical_references": historical,
        "test_labels_used_for_fit_or_selection": 0,
        "global_test_evaluation_boundary": paper_audit.get("status"),
        "target_threshold": None,
    }
    atomic_json(STUDY / "artifact_audit.json", audit)


def derive_scientific_decision(summaries: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Apply the protocol's fixed decision rules only after frozen evaluation."""

    paired = summaries["paired"]
    paper = paired.loc[
        paired.method.eq("paper_style_current_v2")
        & paired.column.isin(("25g", "40g"))
        & paired.reference.isin(("scale_only", "affine", "local_identity_shrinkage"))
    ].copy()
    expected = 2 * len(PROTOCOLS) * 3
    _assert(len(paper) == expected, "paper-style paired comparison matrix is incomplete")
    material = bool(
        (paper.combined_normalized_rmse_relative_gain >= 0.05).all()
        and (paper.combined_normalized_rmse_wins >= 4).all()
        and (paper.aulc_relative_gain >= 0.05).all()
        and (paper.aulc_wins >= 4).all()
    )
    tails = summaries["tail_summary"].loc[
        lambda frame: frame.budget.eq(100)
        & frame.column.isin(("25g", "40g"))
        & frame.stratum.eq("high_tail")
    ]
    tail_ranges = []
    for keys, group in tails.groupby(["column", "protocol", "target"], sort=False):
        tail_ranges.append({
            "column": keys[0], "protocol": keys[1], "target": keys[2],
            "minimum_high_tail_sse_percent": float(100.0 * group.squared_error_fraction_mean.min()),
            "maximum_high_tail_sse_percent": float(100.0 * group.squared_error_fraction_mean.max()),
            "all_methods_tail_dominant": bool((group.squared_error_fraction_mean >= 0.50).all()),
        })
    all_tail_dominant = bool(tail_ranges) and all(item["all_methods_tail_dominant"] for item in tail_ranges)
    tags = ["PAPER_STYLE_MATERIALLY_IMPROVES_MATCHED_ABSOLUTE_ERROR" if material else "SIMPLE_CALIBRATION_REMAINS_COMPETITIVE"]
    tags.append("TAIL_ERROR_DOMINATES_LARGE_COLUMN_RMSE" if all_tail_dominant else "TAIL_ERROR_IS_QUANTIFIED_BUT_NOT_UNIFORMLY_DOMINANT")
    return {
        "decision": tags[0],
        "decision_tags": tags,
        "decision_rules": DECISION_RULES,
        "paper_style_material_rule_passed": material,
        "paper_style_comparison_rows": int(len(paper)),
        "large_column_tail_sse_ranges_percent": tail_ranges,
        "r2_role": "secondary trend metric only; RMSE/MAE in mL are primary",
        "evidence_scope": "matched developmental benchmark; reused source-anchored artifacts were historically test-exposed",
        "complexity_gate": "A new complex model needs an independently replicated, preregistered paired absolute-error gain; this result does not by itself identify a causal mechanism.",
    }


def _best_rmse_lines(budget100: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for column in COLUMNS:
        for split_protocol in PROTOCOLS:
            for target in TARGETS:
                candidate = budget100.loc[(budget100.column.eq(column)) & (budget100.protocol.eq(split_protocol))]
                best = candidate.sort_values(f"{target}_rmse_mean").iloc[0]
                lines.append(
                    f"- `{column}/{split_protocol}` {target}: `{best['method']}` has the lowest B=100 mean RMSE "
                    f"({best[f'{target}_rmse_mean']:.2f} mL; MAE {best[f'{target}_mae_mean']:.2f} mL; R2 {best[f'{target}_r2_mean']:.3f})."
                )
    return lines


def write_report(protocol: dict, summaries: dict[str, pd.DataFrame]) -> dict[str, object]:
    b = summaries["budget100"]
    a = summaries["aulc_summary"]
    tail = summaries["tail_summary"]
    distributions = summaries["target_distribution_summary"]
    decision = derive_scientific_decision(summaries)
    lines = [
        "# Matched cross-column absolute-error benchmark",
        "",
        "Decision is based on RMSE/MAE in mL; R2 is secondary.  Every method uses the same qualified current-V2 source checkpoint, canonical no-threshold targets, frozen split schedule, target-label budget and test population.",
        "",
        "## Protocol",
        "",
        f"Source: `{protocol['source_checkpoint']}` (SHA-256 `{protocol['source_checkpoint_sha256']}`). Columns: 8g, 25g, 40g. Protocols: row and target-compound holdout. Outer seeds: `{', '.join(map(str, SEEDS))}`. Budgets: `{', '.join(map(str, BUDGETS))}`. Target threshold: `NONE`.",
        "",
        "Simple calibration fits use gradient-train labels only. Neural checkpoint selection uses frozen validation labels only. The paper-style runner builds zero-labelled graphs, reveals only train/validation labels, freezes and hashes all 120 prediction files, and only then reads test truth. The schedule is inherited byte-for-byte from `cross_column`; no split was rerandomized.",
        "",
        "## B=100 primary table",
        "",
        "The CSV contains mean, sample SD, median, minimum and maximum across the five seeds. The table below shows mean values and keeps row/compound protocols explicit.",
        "",
        "| column | protocol | method | B | V1 RMSE | V2 RMSE | V1 MAE | V2 MAE | V1 R2 | V2 R2 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in b.sort_values(["column", "protocol", "method"]).iterrows():
        lines.append(f"| {row['column']} | {row['protocol']} | {row['method']} | 100 | {row['V1_rmse_mean']:.2f} | {row['V2_rmse_mean']:.2f} | {row['V1_mae_mean']:.2f} | {row['V2_mae_mean']:.2f} | {row['V1_r2_mean']:.3f} | {row['V2_r2_mean']:.3f} |")
    lines += ["", "## Label efficiency", "", "AULC is the trapezoidal area of the project-compatible arithmetic mean source-normalized RMSE over planned budgets 30/50/70/100; lower is better.", "", "| column | protocol | method | AULC mean | sample SD | median | min | max |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for _, row in a.sort_values(["column", "protocol", "aulc_mean"]).iterrows():
        lines.append(f"| {row['column']} | {row['protocol']} | {row['method']} | {row['aulc_mean']:.3f} | {row['aulc_sample_sd']:.3f} | {row['aulc_median']:.3f} | {row['aulc_min']:.3f} | {row['aulc_max']:.3f} |")
    lines += ["", "## Tail diagnosis", "", "For each seed and budget, q50 and q80 are computed from that context's gradient-train labels only. Test rows are then classified as low, mid or high-tail using true test volume. `squared_error_fraction` is that stratum's contribution to total SSE. `tail_error_summary.csv` retains mean, sample SD, median, min and max for all low/mid/high strata.", "", "| column | protocol | method | target | tail RMSE | tail MAE | tail SSE fraction |", "| --- | --- | --- | --- | ---: | ---: | ---: |"]
    high = tail.loc[(tail.stratum.eq("high_tail")) & (tail.budget.eq(100))].copy()
    for _, row in high.sort_values(["column", "protocol", "target", "rmse_mean"]).iterrows():
        lines.append(f"| {row['column']} | {row['protocol']} | {row['method']} | {row['target']} | {row['rmse_mean']:.2f} | {row['mae_mean']:.2f} | {100*row['squared_error_fraction_mean']:.1f}% |")
    lines += [
        "", "## Reuse and protocol boundaries", "",
        "Existing baseline and current-V2 source-anchored predictions are included only as `REUSED_MATCHED_ARTIFACT` after source/canonical/schedule/role/hash checks. The new paper arm is `RERUN_MATCHED_ARTIFACT`. Reused source-anchored results retain their `DEVELOPMENTAL` caveat because their historical test populations had already been examined. The legacy paper reproduction remains `NOT_PROTOCOL_MATCHED` because it uses threshold filtering, a larger label fraction, an old E0 source checkpoint and different test population; it is not ranked here.",
        "", "Paper-style current-V2 starts from the qualified source and keeps the six-output current-V2 contract. It trains the final message layer, condition branch, transferred target head and a zero-initialized three-dimensional context adapter. It uses only nominal mass and column identity from the audited canonical targets; it does not use the historical hardcoded `(2.15, 15.6, 0.5248)` tuple and makes no verified physical-geometry claim. Since context is constant within each independently fit target column, its learned context shift is not separately identifiable from a target-specific latent shift; this is a paper-inspired strategy control, not proof of a physical column mechanism.",
        "", "## Direct answers to the scientific questions", "",
        "### 1–3. B=100 absolute-error floor", "",
        "The following lists the lowest matched mean RMSE for each column/protocol/target; these are comparative observations, not operational acceptance thresholds.",
        "",
        *_best_rmse_lines(b),
        "", "### 4. Equal-budget paper-style strategy", "",
        f"The fixed material-gain rule is `{'PASSED' if decision['paper_style_material_rule_passed'] else 'NOT PASSED'}`. It requires >=5% B=100 combined-normalized-RMSE and AULC gain with >=4/5 paired wins versus scale-only, affine and local shrinkage in both 25g and 40g. See `paired_comparisons.csv`; a one-seed win is not treated as evidence.",
        "", "### 5. Why the historical paper reproduction can look much lower", "",
        "The paper-aligned reproduction is not an estimand in this table: it confounds legacy V1/V2 filtering, roughly 80% target training labels, an old E0 source, different random splits/test population and a monotonic-head architecture. Its apparent low RMSE cannot be causally apportioned among those differences from the existing artifacts, so it cannot establish a matched strategy advantage.",
        "", "### 6. Is high-volume tail the primary driver?", "",
        "The predeclared B=100 tail SSE ranges are recorded in `SCIENTIFIC_DECISION.json` and the full tail table above. A tail is called dominant only at >=50% SSE. Results below that threshold show a material tail contribution but do not support attributing the entire large-column RMSE to a small tail alone.",
        "", "### 7. Why high R2 can coexist with large RMSE", "",
        "R2 is normalized by the variance of the evaluated target population. The B=100 test-distribution summary below shows the target-scale context; a larger test standard deviation permits a larger absolute RMSE at the same R2. It is therefore secondary to mL RMSE/MAE for operational interpretation.",
        "",
        "| column | protocol | target | mean test sample SD (mL) | mean test variance (mL²) |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for _, row in distributions.sort_values(["column", "protocol", "target"]).iterrows():
        lines.append(f"| {row['column']} | {row['protocol']} | {row['target']} | {row['test_sample_sd_mean']:.2f} | {row['test_variance_mean']:.2f} |")
    lines += [
        "", "### 8–9. Bottleneck and next-model decision", "",
        f"`{decision['decision']}`. This benchmark can distinguish matched predictive error patterns and paired reproducibility, but it cannot causally isolate calibration bias, tail extrapolation, representation capacity, low-label estimation, or measurement noise. Unless the preregistered paper-style material-gain rule passes, the evidence does not justify appending a more complex model solely to chase this test result; independent batches, replicated measurements and tail coverage remain the cleaner next evidence.",
        "", "## Scientific decision", "",
        f"Primary outcome: `{decision['decision']}`. Tags: `{', '.join(decision['decision_tags'])}`. RMSE/MAE, tail fractions and paired five-seed comparisons should be read together. High R2 alone is not evidence of operationally small absolute error.",
        "",
    ]
    (STUDY / "MATCHED_RMSE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    atomic_json(STUDY / "SCIENTIFIC_DECISION.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="validate and freeze protocol")
    parser.add_argument("--execute", action="store_true", help="fit, globally freeze, then evaluate the missing paper-style arm")
    parser.add_argument("--summarize", action="store_true", help="rebuild summaries from frozen runtime predictions")
    parser.add_argument("--smoke", action="store_true", help="validate the parent and public APIs without writing a study record")
    args = parser.parse_args()
    parent, schedule, canonical = validate_parent_protocol()
    if args.smoke:
        from src.qgeognn_al.models import load_predictor_checkpoint
        from src.qgeognn_al.transfer import PaperStyleCurrentV2, absolute_error_metrics, loader_pair, quantile_target_loss, tail_error_rows
        payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
        atom, angle, _ = _source_graph_inputs("8g", canonical["8g"], payload["preprocessing"])
        model = PaperStyleCurrentV2(load_predictor_checkpoint(SOURCE), context_dim=3, scope="shallow")
        first_atom, first_angle = next(zip(*loader_pair(atom, angle, (0, 1), 2)))
        with torch.no_grad():
            output = model(first_atom, first_angle)
        _assert(tuple(output.shape) == (2, 6) and bool(torch.isfinite(output).all()), "paper-style smoke forward failed")
        model.training_target()
        training_output = model(first_atom, first_angle)
        smoke_loss = quantile_target_loss(first_atom.y[:, 0], training_output[:, :3]) + quantile_target_loss(first_atom.y[:, 1], training_output[:, 3:])
        smoke_loss.backward()
        trainable_gradient = any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
        source_gradient = any(parameter.grad is not None for parameter in model.source_head.parameters())
        _assert(trainable_gradient and not source_gradient, "paper-style shallow gradient scope failed")
        print(json.dumps({"smoke": True, "contexts": 120, "forward_shape": list(output.shape), "public_apis": [PaperStyleCurrentV2.__name__, absolute_error_metrics.__name__, tail_error_rows.__name__]}))
        return
    protocol = build_protocol(parent)
    if args.prepare and not args.execute and not args.summarize:
        print(json.dumps({"prepared": True, "protocol_sha256": sha256_file(STUDY / "protocol.json"), "contexts": 120}))
        return
    paper_rows: list[dict] = []
    paper_audit: dict = {}
    if args.execute:
        paper_rows, paper_audit = execute_paper_style(protocol, schedule, canonical)
    elif (STUDY / "paper_style_audit.json").exists():
        paper_audit = json.loads((STUDY / "paper_style_audit.json").read_text(encoding="utf-8"))
        _assert(paper_audit.get("status") == "PREDICTIONS_GLOBALLY_FROZEN_BEFORE_TEST_EVALUATION", "paper-style predictions were not globally frozen")
        _assert(paper_audit.get("test_labels_used_for_fit_or_selection") == 0, "paper-style freeze reports test leakage")
        for key, value in paper_audit.get("contexts", {}).items():
            column, split_protocol, seed, budget = key.split("/")
            paper_rows.append({"column": column, "protocol": split_protocol, "seed": int(seed), "budget": int(budget)})
    else:
        raise RuntimeError("paper-style arm is missing; run with --execute")
    _assert(len(paper_rows) == 120, "paper-style freeze does not contain all 120 contexts")
    metrics, tails, context_audit, prediction_rows, target_distributions = build_records(protocol, schedule, canonical, paper_rows)
    summaries = summarize(metrics, tails, prediction_rows, target_distributions, protocol)
    write_audits(protocol, context_audit, paper_audit)
    decision = write_report(protocol, summaries)
    print(json.dumps({"completed": True, "contexts": 120, "metric_rows": len(metrics), "tail_rows": len(tails), "decision": decision["decision"], "study": str(STUDY)}))


if __name__ == "__main__":
    main()
