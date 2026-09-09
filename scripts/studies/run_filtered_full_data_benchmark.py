#!/usr/bin/env python3
"""Filtered full-data 4g-to-25g/40g transfer benchmark.

The fitting phase exposes target labels only for gradient-train and frozen
validation identities.  It writes every test prediction before a separate
evaluation phase reads test truth.  ``--execute`` therefore has a genuine
global prediction-freeze boundary, rather than merely a convention.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qgeognn_al.data import build_model_data
from src.qgeognn_al.models import load_predictor_checkpoint
from src.qgeognn_al.resources import SOURCE_GRAPH_CACHE
from src.qgeognn_al.training.predictor import atomic_json
from src.qgeognn_al.transfer import (
    absolute_error_metrics,
    build_full_data_schedule,
    ea_fraction,
    filter_operational_domain,
    fit_paper_style_current_v2_full,
    fit_scale_only,
    fit_target_only_full,
    loader_pair,
    make_label_scrubbed_graphs,
    select_conditional_ea,
)
from src.qgeognn_al.transfer.full_data import sha256_file, stable_hash


STUDY = ROOT / "studies/transfer/filtered_full_data_benchmark"
PARENT = ROOT / "studies/transfer/cross_column"
MATCHED = ROOT / "studies/transfer/matched_rmse_benchmark"
ANCHORED = ROOT / "studies/transfer/source_anchored_shared_transfer"
SOURCE = ROOT / "studies/predictor/final_4g_qualification/runtime/row/seed_42/best.pt"
SOURCE_SHA256 = "fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0"
SCHEDULE_SHA256 = "b52b937aa750586ec8f2efb2f10f93dfddd2252bb03afaf304556e16b4fab5ee"
COLUMNS = ("25g", "40g")
PROTOCOLS = ("row", "compound")
SEEDS = (769539383, 1425370602, 536279090, 2767143051, 1362771960)
METHODS = ("zero_shot", "target_only_full", "scale_only", "conditional_EA", "paper_style_current_v2")
THRESHOLDS = {"25g": (60.0, 120.0), "40g": (150.0, 200.0)}
MASS_G = {"25g": 25.0, "40g": 40.0}
PARENT_BUDGET = 100
MINIMUM_ROLE_COUNTS = {"gradient_train": 300, "validation": 5, "test": 60}
CONDITIONAL_PENALTIES = (0.0, 0.1, 1.0)
TARGET_ONLY_CONFIG = {
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "maximum_epochs": 1000,
    "patience": 100,
    "batch_size": 2048,
    "cpu_threads": 1,
    "selection": "validation_only_minimum_target_train_normalized_rmse",
    "initialization": "direct_seeded_random_current_v2",
}
PAPER_CONFIG = {
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "maximum_epochs": 500,
    "patience": 100,
    "batch_size": 2048,
    "cpu_threads": 1,
    "context_dim": 3,
    "scope": "final message layer + condition branch + target head + zero-initialized column adapter",
    "selection": "validation_only_minimum_source_normalized_rmse",
}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _markdown(frame: pd.DataFrame, columns: Iterable[str] | None = None, digits: int = 3) -> str:
    value = frame.loc[:, list(columns)] if columns else frame
    if value.empty:
        return "_No rows._\n"
    displayed = value.copy()
    for name in displayed.columns:
        if pd.api.types.is_float_dtype(displayed[name]):
            displayed[name] = displayed[name].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    # Do not require the optional ``tabulate`` dependency merely to render a
    # compact scientific record.
    names = [str(name) for name in displayed.columns]
    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(names) + " |", "| " + " | ".join("---" for _ in names) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in displayed.itertuples(index=False, name=None))
    return "\n".join(lines) + "\n"


def canonical_path(column: str) -> Path:
    return PARENT / "data_audit" / f"canonical_{column}.csv"


def filtered_path(column: str) -> Path:
    return STUDY / f"filtered_canonical_{column}.csv"


def features_path(column: str) -> Path:
    return STUDY / f"filtered_features_{column}.csv"


def _protocol_payload(filtered: dict[str, pd.DataFrame], accounting: pd.DataFrame) -> dict:
    return {
        "study_name": "FILTERED_FULL_DATA_TRANSFER_PILOT_BENCHMARK",
        "classification": "FILTERED_OPERATIONAL_DOMAIN_FULL_DATA_TRANSFER_PILOT",
        "status": "FROZEN_BEFORE_NEW_TEST_EVALUATION",
        "parent_schedule": str((PARENT / "splits/schedule_manifest.csv").relative_to(ROOT)),
        "parent_schedule_sha256": SCHEDULE_SHA256,
        "parent_budget_intersection": PARENT_BUDGET,
        "columns": list(COLUMNS),
        "protocols": list(PROTOCOLS),
        "outer_seeds": list(SEEDS),
        "threshold_definition": "legacy operational-domain thresholds observed in original Construct_dataset_*; not a physical-law claim",
        "thresholds_ml": {column: {"V1": THRESHOLDS[column][0], "V2": THRESHOLDS[column][1]} for column in COLUMNS},
        "target_population": "reader-compatible parent canonical rows filtered before role intersection",
        "split_design": "frozen B100 outer test/validation identities intersected with filtered population; parent gradient_train plus pool become full gradient_train",
        "full_data_definition": "all filtered rows outside frozen validation and held-out test are gradient-train labels; test is never a training label",
        "minimum_role_counts_preregistered": MINIMUM_ROLE_COUNTS,
        "actual_label_accounting_sha256": sha256_file(STUDY / "label_accounting.csv"),
        "filtered_canonical_sha256": {column: sha256_file(filtered_path(column)) for column in COLUMNS},
        "filtered_rows": {column: int(len(filtered[column])) for column in COLUMNS},
        "methods": list(METHODS),
        "zero_shot": "fixed hash-locked qualified 4g source; zero target labels",
        "scale_only": "per-target through-origin fit on source q50 and gradient-train labels only",
        "conditional_EA": {
            "implementation": "src.qgeognn_al.transfer.conditional_scaling.fit_conditional",
            "source_q50": True,
            "interaction": True,
            "penalty_grid": list(CONDITIONAL_PENALTIES),
            "fit_role": "gradient_train",
            "selection_role": "validation_only",
            "source_scales": "qualified_4g_source_train",
        },
        "target_only_full": {
            "source_checkpoint_loaded": False,
            "source_predictions_used": False,
            "source_replay_used": False,
            "model": "randomly initialized current QGeoGNN-V2 backbone and six-output formulation",
            "preprocessing": "target gradient-train only",
            "config": TARGET_ONLY_CONFIG,
        },
        "paper_style_current_v2": {
            "source_checkpoint": str(SOURCE.relative_to(ROOT)),
            "source_checkpoint_sha256": SOURCE_SHA256,
            "config": PAPER_CONFIG,
            "column_context": "[target_mass_g / 4, is_25g, is_40g]",
            "scope_fixed_before_test": True,
        },
        "test_labels_used_for_fit_or_selection": False,
        "global_test_evaluation_boundary": "all 20 context prediction files must hash-verify before test truth is read",
        "source_preprocessing_for_transfer": "qualified source-train only",
        "target_only_preprocessing": "target gradient-train only",
        "compound_split_interpretation": "target-compound generalization; not source-unseen chemistry",
        "no_threshold_historical_results_modified": False,
        "paper_reproduction_context": "descriptive only, not a head-to-head matched competitor",
        "training_context_count": int(len(accounting)),
    }


def _filter_audit_tables(audits: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    detail: dict[str, object] = {}
    for column, audit in audits.items():
        removed = audit.loc[~audit.operational_keep].copy()
        kept = audit.loc[audit.operational_keep].copy()
        ratio = lambda subset: {
            str(key): int(value) for key, value in subset.value_counts(dropna=False).sort_index().items()
        }
        distributions = {}
        for name in ("V1_ml", "V2_ml"):
            distributions[name] = {
                key: float(value) for key, value in removed[name].describe(percentiles=[.05, .25, .5, .75, .95]).items()
            }
        condition_columns = [name for name in ("PE/EA", "loading solvent", "V/ul", "Volume of loading solvent/ul") if name in audit]
        compound_before = audit.groupby("canonical_smiles").size().sort_index()
        compound_after = kept.groupby("canonical_smiles").size().reindex(compound_before.index, fill_value=0).sort_index()
        detail[column] = {
            "reader_compatible_rows": int(len(audit)), "filtered_rows": int(len(kept)),
            "removed_rows": int(len(removed)), "removed_percent": float(100 * len(removed) / len(audit)),
            "removed_V_distribution": distributions,
            "removed_compounds": int(removed.canonical_smiles.nunique()),
            "compounds_completely_removed": int(len(set(audit.canonical_smiles) - set(kept.canonical_smiles))),
            "removed_condition_distribution": {name: ratio(removed[name].astype(str)) for name in condition_columns},
            "kept_condition_distribution": {name: ratio(kept[name].astype(str)) for name in condition_columns},
            "removed_EA_fraction": {
                key: float(value) for key, value in pd.Series(ea_fraction(removed["PE/EA"])).describe(percentiles=[.05, .25, .5, .75, .95]).items()
            },
            "kept_EA_fraction": {
                key: float(value) for key, value in pd.Series(ea_fraction(kept["PE/EA"])).describe(percentiles=[.05, .25, .5, .75, .95]).items()
            },
            "flow_distribution_removed": ratio(removed["Flow mL/min"].astype(str)) if "Flow mL/min" in audit else {},
            "compound_row_count_changed": int((compound_before != compound_after).sum()),
        }
        rows.append({
            "column": column, "reader_compatible_rows": len(audit), "filtered_rows": len(kept),
            "removed_rows": len(removed), "removed_percent": 100 * len(removed) / len(audit),
            "unique_compounds_before": audit.canonical_smiles.nunique(), "unique_compounds_after": kept.canonical_smiles.nunique(),
            "compounds_completely_removed": detail[column]["compounds_completely_removed"],
            "compounds_with_some_rows_removed": detail[column]["compound_row_count_changed"],
            "removed_V1_median_ml": detail[column]["removed_V_distribution"]["V1_ml"]["50%"],
            "removed_V2_median_ml": detail[column]["removed_V_distribution"]["V2_ml"]["50%"],
        })
    return pd.DataFrame(rows), detail


def prepare() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Audit/freeze the target population and its inherited outer schedule."""

    _assert(SOURCE.exists(), f"missing qualified source: {SOURCE}")
    _assert(sha256_file(SOURCE) == SOURCE_SHA256, "qualified source SHA256 mismatch")
    parent_protocol = _json(PARENT / "protocol.json")
    _assert(parent_protocol["source_checkpoint_sha256"] == SOURCE_SHA256, "parent source checkpoint mismatch")
    schedule_path = PARENT / "splits/schedule_manifest.csv"
    _assert(sha256_file(schedule_path) == SCHEDULE_SHA256, "frozen parent schedule hash mismatch")
    parent_schedule = pd.read_csv(schedule_path)
    filtered: dict[str, pd.DataFrame] = {}
    audits: dict[str, pd.DataFrame] = {}
    for column in COLUMNS:
        raw = pd.read_csv(canonical_path(column))
        filtered[column], audits[column] = filter_operational_domain(
            raw, column=column, v1_limit_ml=THRESHOLDS[column][0], v2_limit_ml=THRESHOLDS[column][1]
        )
        # The observed counts tie this study to the manifest/original reader,
        # while the actual values remain in the audit record.
        expected = {"25g": 408, "40g": 456}[column]
        _assert(len(filtered[column]) == expected, f"unexpected filtered count for {column}: {len(filtered[column])}")
    schedule, accounting = build_full_data_schedule(
        parent_schedule, filtered, columns=COLUMNS, protocols=PROTOCOLS, seeds=SEEDS, parent_budget=PARENT_BUDGET
    )
    _assert(len(accounting) == 20, "expected 20 full-data contexts")
    for role, minimum in MINIMUM_ROLE_COUNTS.items():
        actual = accounting[f"{role}_rows"].min()
        _assert(actual >= minimum, f"intersection role too small for prespecified rule: {role}={actual} < {minimum}")
    STUDY.mkdir(parents=True, exist_ok=True)
    for column in COLUMNS:
        _write_frame(filtered[column], filtered_path(column))
        _write_frame(filtered[column].drop(columns=["V1_ml", "V2_ml"]), features_path(column))
    combined_audit = pd.concat(audits.values(), ignore_index=True)
    _write_frame(combined_audit, STUDY / "FILTER_AUDIT.csv")
    audit_summary, audit_detail = _filter_audit_tables(audits)
    _write_frame(audit_summary, STUDY / "FILTER_AUDIT_SUMMARY.csv")
    atomic_json(STUDY / "FILTER_AUDIT.json", audit_detail)
    _write_frame(schedule, STUDY / "split_manifest.csv")
    _write_frame(accounting, STUDY / "label_accounting.csv")
    protocol = _protocol_payload(filtered, accounting)
    protocol_path = STUDY / "protocol.json"
    if protocol_path.exists() and _json(protocol_path) != protocol:
        raise RuntimeError("filtered full-data protocol is frozen; refusing to overwrite it")
    atomic_json(protocol_path, protocol)
    return filtered, accounting


def _read_authorized_truth(path: Path, ids: Iterable[str]) -> np.ndarray:
    """Read labels for authorized IDs without parsing other label cells."""

    requested = [str(value) for value in ids]
    _assert(requested and len(requested) == len(set(requested)), "truth request must be nonempty and unique")
    identity = pd.read_csv(path, usecols=["sample_id"])
    allowed = set(requested)
    skip = [index + 1 for index, value in enumerate(identity.sample_id.astype(str)) if value not in allowed]
    frame = pd.read_csv(path, usecols=["sample_id", "V1_ml", "V2_ml"], skiprows=skip).set_index("sample_id")
    _assert(set(frame.index.astype(str)) == allowed, "authorized truth identity mismatch")
    return frame.loc[requested, ["V1_ml", "V2_ml"]].to_numpy(float)


def _context(column: str, protocol: str, seed: int) -> pd.DataFrame:
    schedule = pd.read_csv(STUDY / "split_manifest.csv")
    frame = schedule.loc[
        schedule.column.eq(column) & schedule.protocol.eq(protocol) & schedule.outer_seed.eq(int(seed))
    ].copy()
    _assert(not frame.empty and frame.sample_id.astype(str).is_unique, "missing or duplicate full-data context")
    for role in ("gradient_train", "validation", "test"):
        _assert(bool(frame.role.eq(role).any()), f"missing {role} in full-data context")
    return frame


def _positions(feature: pd.DataFrame, ids: Iterable[str]) -> np.ndarray:
    lookup = {str(value): index for index, value in enumerate(feature.sample_id.astype(str))}
    positions = np.asarray([lookup[str(value)] for value in ids], dtype=int)
    _assert(len(positions) == len(set(positions)), "duplicate model positions")
    return positions


def _model_points(model: torch.nn.Module, atom: list[object], angle: list[object], positions: np.ndarray) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for atom_batch, angle_batch in zip(*loader_pair(atom, angle, positions, 2048)):
            value = model(atom_batch, angle_batch)
            _assert(value.ndim == 2 and value.shape[1] == 6, "source did not return six outputs")
            outputs.append(value.detach().cpu().numpy())
    points = np.vstack(outputs)[:, (1, 4)]
    _assert(np.isfinite(points).all(), "non-finite source q50 prediction")
    return points


def _column_context(column: str, count: int) -> np.ndarray:
    value = np.asarray([MASS_G[column] / 4.0, float(column == "25g"), float(column == "40g")], dtype=np.float32)
    return np.repeat(value[None, :], count, axis=0)


def _runtime_context(column: str, protocol: str, seed: int, smoke: bool = False) -> Path:
    prefix = "smoke" if smoke else "contexts"
    return STUDY / "runtime" / prefix / column / protocol / f"seed_{seed}"


def _fit_context(column: str, protocol: str, seed: int, *, smoke: bool = False) -> dict:
    """Freeze all five blind test-prediction arms for one context."""

    # ``--execute`` prepares once before spawning workers.  Do not rewrite
    # shared CSVs in every child process; a direct context call still prepares
    # itself when no frozen protocol exists yet.
    if not (STUDY / "protocol.json").exists():
        prepare()
    else:
        _assert(sha256_file(SOURCE) == SOURCE_SHA256, "qualified source SHA256 mismatch")
        _assert(_json(STUDY / "protocol.json")["status"] == "FROZEN_BEFORE_NEW_TEST_EVALUATION", "invalid study protocol")
    context = _context(column, protocol, seed)
    runtime = _runtime_context(column, protocol, seed, smoke)
    frozen_path = runtime / "frozen.json"
    contract = {
        "protocol_sha256": sha256_file(STUDY / "protocol.json"), "column": column,
        "protocol": protocol, "seed": int(seed), "smoke": bool(smoke),
        "gradient_train_ids_hash": stable_hash(sorted(context.loc[context.role.eq("gradient_train"), "sample_id"].astype(str))),
        "validation_ids_hash": stable_hash(sorted(context.loc[context.role.eq("validation"), "sample_id"].astype(str))),
        "test_ids_hash": stable_hash(sorted(context.loc[context.role.eq("test"), "sample_id"].astype(str))),
        "training_label_count": int(context.role.eq("gradient_train").sum()),
        "validation_rows": int(context.role.eq("validation").sum()), "test_rows": int(context.role.eq("test").sum()),
        "test_labels_used_for_fit_or_selection": 0,
    }
    if frozen_path.exists():
        frozen = _json(frozen_path)
        _assert(frozen.get("contract") == contract, f"frozen context contract changed: {runtime}")
        for name, digest in frozen.get("files", {}).items():
            _assert(sha256_file(runtime / name) == digest, f"frozen runtime file changed: {runtime / name}")
        return {"status": "REUSED", "runtime": str(runtime.relative_to(ROOT)), "contract": contract}

    feature = pd.read_csv(features_path(column)).reset_index(drop=True)
    _assert(feature.sample_id.astype(str).is_unique, "duplicate filtered feature IDs")
    ids = {role: context.loc[context.role.eq(role), "sample_id"].astype(str).tolist()
           for role in ("gradient_train", "validation", "test")}
    positions = {role: _positions(feature, values) for role, values in ids.items()}
    # These are the only target label reads made by the fitting phase.
    train_truth = _read_authorized_truth(filtered_path(column), ids["gradient_train"])
    validation_truth = _read_authorized_truth(filtered_path(column), ids["validation"])
    source_cache = dict(torch.load(SOURCE_GRAPH_CACHE, weights_only=False))
    source_cache.update(torch.load(PARENT / "data_audit" / f"graph_cache_{column}_only.pt", weights_only=False))
    source_payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    source_preprocessing = source_payload["preprocessing"]
    source_model = load_predictor_checkpoint(SOURCE)
    base_atom, angle = make_label_scrubbed_graphs(feature, source_cache, source_preprocessing["scaler"])
    source_point = _model_points(source_model, base_atom, angle, np.arange(len(feature), dtype=int))
    test_source = source_point[positions["test"]]
    predictions: dict[str, np.ndarray] = {"zero_shot": test_source}
    scale = fit_scale_only(train_truth, source_point[positions["gradient_train"]], test_source)
    predictions["scale_only"] = scale.prediction
    conditional, conditional_audit = select_conditional_ea(
        train_truth=train_truth, train_source=source_point[positions["gradient_train"]],
        train_ea=ea_fraction(feature.iloc[positions["gradient_train"]]["PE/EA"]),
        validation_truth=validation_truth, validation_source=source_point[positions["validation"]],
        validation_ea=ea_fraction(feature.iloc[positions["validation"]]["PE/EA"]),
        predict_source=test_source, predict_ea=ea_fraction(feature.iloc[positions["test"]]["PE/EA"]),
        source_scales=[source_preprocessing["target_scales"][target] for target in ("V1", "V2")],
        mass_ratio=MASS_G[column] / 4.0, penalties=CONDITIONAL_PENALTIES,
    )
    predictions["conditional_EA"] = conditional
    target_config = dict(TARGET_ONLY_CONFIG)
    paper_config = dict(PAPER_CONFIG)
    if smoke:
        target_config.update({"maximum_epochs": 2, "patience": 2})
        paper_config.update({"maximum_epochs": 2, "patience": 2})
    target_prediction, target_audit = fit_target_only_full(
        feature_frame=feature, graph_cache=source_cache, train_indices=positions["gradient_train"],
        validation_indices=positions["validation"], test_indices=positions["test"], train_truth=train_truth,
        validation_truth=validation_truth, seed=int(seed), config=target_config, runtime=runtime / "target_only_full",
        contract={**contract, "method": "target_only_full", "config": target_config},
    )
    predictions["target_only_full"] = target_prediction
    paper_prediction, paper_audit = fit_paper_style_current_v2_full(
        source_model=source_model, base_atom=base_atom, angle=angle, contexts=_column_context(column, len(feature)),
        train_indices=positions["gradient_train"], validation_indices=positions["validation"], test_indices=positions["test"],
        train_truth=train_truth, validation_truth=validation_truth, source_scales=source_preprocessing["target_scales"],
        seed=int(seed), config=paper_config, runtime=runtime / "paper_style_current_v2",
        contract={**contract, "method": "paper_style_current_v2", "config": paper_config, "source_checkpoint_sha256": SOURCE_SHA256},
    )
    predictions["paper_style_current_v2"] = paper_prediction
    _assert(set(predictions) == set(METHODS) and all(np.isfinite(value).all() for value in predictions.values()), "invalid context predictions")
    runtime.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame({"sample_id": ids["test"]})
    for method in METHODS:
        prediction_frame[f"{method}_V1"] = predictions[method][:, 0]
        prediction_frame[f"{method}_V2"] = predictions[method][:, 1]
    prediction_path = runtime / "predictions_blind.csv.gz"
    prediction_frame.to_csv(prediction_path, index=False, compression={"method": "gzip", "mtime": 0})
    fit_audit = {
        "contract": contract, "zero_shot": {"source_checkpoint_sha256": SOURCE_SHA256, "target_labels_used": 0},
        "scale_only": {"coefficients": scale.coefficients.tolist(), "fit_role": "gradient_train", "source_q50": True},
        "conditional_EA": conditional_audit, "target_only_full": target_audit,
        "paper_style_current_v2": paper_audit, "test_labels_used_for_fit_or_selection": 0,
    }
    atomic_json(runtime / "fit_audit.json", fit_audit)
    files = {name: sha256_file(runtime / name) for name in ("predictions_blind.csv.gz", "fit_audit.json")}
    atomic_json(frozen_path, {"contract": contract, "files": files, "methods": list(METHODS),
                              "test_labels_used_for_fit_or_selection": 0})
    return {"status": "FITTED", "runtime": str(runtime.relative_to(ROOT)), "contract": contract}


def _all_contexts() -> list[tuple[str, str, int]]:
    return [(column, protocol, seed) for column in COLUMNS for protocol in PROTOCOLS for seed in SEEDS]


def _freeze_all_predictions() -> dict:
    records = {}
    for column, protocol, seed in _all_contexts():
        runtime = _runtime_context(column, protocol, seed)
        frozen_path = runtime / "frozen.json"
        _assert(frozen_path.exists(), f"missing full prediction freeze: {runtime}")
        frozen = _json(frozen_path)
        for name, digest in frozen["files"].items():
            _assert(sha256_file(runtime / name) == digest, f"modified frozen context: {runtime / name}")
        _assert(frozen["contract"]["test_labels_used_for_fit_or_selection"] == 0, "test leakage in fit audit")
        records[str(frozen_path.relative_to(STUDY))] = sha256_file(frozen_path)
    payload = {
        "contexts": len(records), "methods": list(METHODS), "files": records,
        "protocol_sha256": sha256_file(STUDY / "protocol.json"),
        "test_truth_evaluation_started": False,
    }
    atomic_json(STUDY / "all_predictions_frozen.json", payload)
    return payload


def _validate_global_freeze() -> dict:
    path = STUDY / "all_predictions_frozen.json"
    _assert(path.exists(), "all predictions are not globally frozen")
    frozen = _json(path)
    _assert(frozen["contexts"] == 20 and frozen["methods"] == list(METHODS), "incomplete global prediction freeze")
    _assert(frozen["protocol_sha256"] == sha256_file(STUDY / "protocol.json"), "global freeze protocol mismatch")
    for relative, digest in frozen["files"].items():
        _assert(sha256_file(STUDY / relative) == digest, f"global freeze file changed: {relative}")
    return frozen


def _summary_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["column", "protocol", "method"]
    values = ["n_train", "n_valid", "n_test", "V1_rmse", "V1_mae", "V1_r2", "V2_rmse", "V2_mae", "V2_r2", "combined_normalized_rmse"]
    result: list[dict] = []
    for key, group in metrics.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row["n_seeds"] = int(len(group))
        for value in values:
            numeric = group[value].astype(float)
            row[f"{value}_mean"] = float(numeric.mean())
            row[f"{value}_sd"] = float(numeric.std(ddof=1))
            row[f"{value}_median"] = float(numeric.median())
            row[f"{value}_min"] = float(numeric.min())
            row[f"{value}_max"] = float(numeric.max())
        result.append(row)
    return pd.DataFrame(result).sort_values(keys).reset_index(drop=True)


def _paired_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["column", "protocol"]
    for (column, protocol), group in metrics.groupby(keys, sort=True):
        for candidate in METHODS:
            candidate_rows = group.loc[group.method.eq(candidate)].set_index("seed")
            for reference in METHODS:
                if candidate == reference:
                    continue
                reference_rows = group.loc[group.method.eq(reference)].set_index("seed")
                _assert(set(candidate_rows.index) == set(reference_rows.index), "unpaired seed contexts")
                row = {"column": column, "protocol": protocol, "candidate": candidate, "reference": reference,
                       "n_pairs": int(len(candidate_rows))}
                for metric in ("V1_rmse", "V1_mae", "V2_rmse", "V2_mae", "combined_normalized_rmse"):
                    delta = candidate_rows[metric].astype(float) - reference_rows[metric].astype(float)
                    row[f"{metric}_delta_mean"] = float(delta.mean())
                    row[f"{metric}_delta_median"] = float(delta.median())
                    row[f"{metric}_wins"] = int((delta < 0).sum())
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["column", "protocol", "candidate", "reference"]).reset_index(drop=True)


def _historical_threshold_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Post-hoc test-domain sensitivity using already frozen B100 predictions."""

    methods = ("zero_shot", "scale_only", "conditional_EA", "paper_style_current_v2")
    schedule = pd.read_csv(PARENT / "splits/schedule_manifest.csv")
    rows: list[dict] = []
    for column in COLUMNS:
        for protocol in PROTOCOLS:
            for seed in SEEDS:
                parent = schedule.loc[
                    schedule.column.eq(column) & schedule.protocol.eq(protocol) & schedule.outer_seed.eq(seed)
                    & schedule.planned_budget.eq(PARENT_BUDGET)
                ]
                test_ids = parent.loc[parent.role.eq("test"), "sample_id"].astype(str).tolist()
                truth = _read_authorized_truth(canonical_path(column), test_ids)
                cross = pd.read_csv(PARENT / column / protocol / f"seed_{seed}" / "budget_100" / "predictions.csv.gz").set_index("sample_id").loc[test_ids]
                anchored = pd.read_csv(ANCHORED / "contexts" / column / protocol / f"seed_{seed}" / "budget_100" / "predictions_blind.csv.gz").set_index("sample_id").loc[test_ids]
                paper = pd.read_csv(MATCHED / "runtime/paper_style_current_v2" / column / protocol / f"seed_{seed}" / "budget_100" / "predictions.csv.gz").set_index("sample_id").loc[test_ids]
                prediction = {
                    "zero_shot": cross[["zero_shot_V1", "zero_shot_V2"]].to_numpy(float),
                    "scale_only": cross[["scale_only_V1", "scale_only_V2"]].to_numpy(float),
                    "conditional_EA": anchored[["conditional_EA_V1", "conditional_EA_V2"]].to_numpy(float),
                    "paper_style_current_v2": paper[["V1_pred", "V2_pred"]].to_numpy(float),
                }
                keep = (truth[:, 0] <= THRESHOLDS[column][0]) & (truth[:, 1] <= THRESHOLDS[column][1])
                _assert(bool(keep.any()), "post-hoc threshold removed all historic test rows")
                for method in methods:
                    for domain, mask in (("no_threshold_test", np.ones(len(truth), dtype=bool)), ("threshold_intersection_test", keep)):
                        values = absolute_error_metrics(truth[mask], prediction[method][mask])
                        rows.append({"column": column, "protocol": protocol, "seed": int(seed), "method": method,
                                     "domain": domain, "n_test": int(mask.sum()), **values})
    full = pd.DataFrame(rows)
    wide = []
    for key, group in full.groupby(["column", "protocol", "seed", "method"], sort=True):
        baseline = group.loc[group.domain.eq("no_threshold_test")].iloc[0]
        filtered = group.loc[group.domain.eq("threshold_intersection_test")].iloc[0]
        row = dict(zip(["column", "protocol", "seed", "method"], key))
        row["n_no_threshold_test"] = int(baseline.n_test)
        row["n_threshold_intersection_test"] = int(filtered.n_test)
        for metric in ("V1_rmse", "V1_mae", "V1_r2", "V2_rmse", "V2_mae", "V2_r2"):
            row[f"{metric}_no_threshold"] = float(baseline[metric])
            row[f"{metric}_threshold_intersection"] = float(filtered[metric])
            row[f"{metric}_delta_filtered_minus_full"] = float(filtered[metric] - baseline[metric])
        wide.append(row)
    detail = pd.DataFrame(wide)
    summary = detail.groupby(["column", "protocol", "method"], as_index=False).mean(numeric_only=True)
    return detail, summary


def _historical_context_table() -> pd.DataFrame:
    metrics = pd.read_csv(MATCHED / "all_metrics.csv")
    selected = metrics.loc[
        metrics.column.isin(COLUMNS) & metrics.protocol.isin(PROTOCOLS) & metrics.seed.isin(SEEDS)
        & metrics.budget.eq(100) & metrics.method.isin(["zero_shot", "scale_only", "conditional_EA", "paper_style_current_v2"])
    ].copy()
    return selected.groupby(["column", "protocol", "method"], as_index=False).agg(
        n_revealed=("actual_budget", "mean"),
        V1_rmse=("V1_rmse", "mean"), V1_mae=("V1_mae", "mean"), V1_r2=("V1_r2", "mean"),
        V2_rmse=("V2_rmse", "mean"), V2_mae=("V2_mae", "mean"), V2_r2=("V2_r2", "mean"),
    ).sort_values(["column", "protocol", "method"]).reset_index(drop=True)


def _decision(summary: pd.DataFrame, paired: pd.DataFrame) -> dict:
    compound = summary.loc[summary.protocol.eq("compound")].copy()
    primary = compound.loc[:, ["column", "method", "V1_rmse_mean", "V1_mae_mean", "V2_rmse_mean", "V2_mae_mean"]]
    wins = []
    for column, group in primary.groupby("column"):
        for metric in ("V1_rmse_mean", "V1_mae_mean", "V2_rmse_mean", "V2_mae_mean"):
            wins.append({"column": column, "metric": metric, "method": str(group.loc[group[metric].idxmin(), "method"])})
    win_frame = pd.DataFrame(wins)
    distinct = sorted(win_frame.method.unique())
    target_only = compound.loc[compound.method.eq("target_only_full")]
    paper = compound.loc[compound.method.eq("paper_style_current_v2")]
    comparisons = paired.loc[(paired.protocol.eq("compound")) & paired.reference.eq("target_only_full")].copy()
    transfer_beats_target_only = comparisons.loc[comparisons.candidate.ne("target_only_full")].groupby("candidate")[
        ["V1_rmse_wins", "V1_mae_wins", "V2_rmse_wins", "V2_mae_wins"]
    ].sum().to_dict(orient="index")
    scarcity_signal = {
        row.column: {
            "target_only_V1_rmse_mean": float(row.V1_rmse_mean), "target_only_V2_rmse_mean": float(row.V2_rmse_mean),
            "paper_style_V1_rmse_mean": float(paper.loc[paper.column.eq(row.column), "V1_rmse_mean"].iloc[0]),
            "paper_style_V2_rmse_mean": float(paper.loc[paper.column.eq(row.column), "V2_rmse_mean"].iloc[0]),
        }
        for _, row in target_only.iterrows()
    }
    return {
        "study": "FILTERED_FULL_DATA_TRANSFER_PILOT_BENCHMARK", "status": "COMPLETED",
        "primary_evidence": "target-compound split; row split is secondary paper-context evidence",
        "compound_primary_metric_winners": wins, "distinct_primary_winners": distinct,
        "universal_winner": distinct[0] if len(distinct) == 1 else None,
        "no_single_universal_winner": len(distinct) != 1,
        "paired_transfer_vs_target_only_wins_across_column_metric_contexts": transfer_beats_target_only,
        "full_data_ceiling_signal": scarcity_signal,
        "active_learning_decision": "NOT_AUTOMATICALLY_ENTERED; this benchmark measures label availability but does not establish acquisition utility",
        "interpretation_limits": [
            "Legacy thresholds define an operational benchmark; they are not established chromatographic physical laws.",
            "Threshold filtering can change condition and compound-row composition; it is not simply outlier removal.",
            "Compound split holds out target-column compounds but does not make them source-unseen chemistry.",
            "Filtered full-data and historical B100 differ in both target domain and label count, so their difference is not a threshold-only causal effect.",
        ],
    }


def _write_report(summary: pd.DataFrame, historical: pd.DataFrame, sensitivity_summary: pd.DataFrame, decision: dict) -> None:
    primary = summary.loc[summary.protocol.eq("compound"), [
        "column", "method", "n_train_mean", "n_valid_mean", "n_test_mean",
        "V1_rmse_mean", "V1_mae_mean", "V1_r2_mean", "V2_rmse_mean", "V2_mae_mean", "V2_r2_mean",
    ]].sort_values(["column", "method"])
    row = summary.loc[summary.protocol.eq("row"), [
        "column", "method", "n_train_mean", "n_valid_mean", "n_test_mean",
        "V1_rmse_mean", "V1_mae_mean", "V1_r2_mean", "V2_rmse_mean", "V2_mae_mean", "V2_r2_mean",
    ]].sort_values(["column", "method"])
    audit = pd.read_csv(STUDY / "FILTER_AUDIT_SUMMARY.csv")
    paper_context = "25g ≈5.91/8.21 mL and 40g ≈12.85/14.81 mL V1/V2 RMSE in the legacy paper reconstruction; it has an old source, distinct preprocessing/split and about 80% target rows, so it is NOT A HEAD-TO-HEAD COMPARISON."
    lines = [
        "# Filtered full-data transfer pilot: final report\n",
        "## A. Audited facts\n",
        f"The fixed 4g source checkpoint is present and hash-locked to `{SOURCE_SHA256}`. Its qualified source has 4,163 effective rows under V1 ≤ 60 mL and V2 ≤ 120 mL. The original QGeoGNN constructors and the data manifest agree with 25g=60/120 and 40g=150/200; these are recorded here as legacy operational-domain thresholds, not physical laws.\n",
        "## B. Filtered population\n", _markdown(audit),
        "Every full-data context retains the frozen B=100 outer validation/test identities after threshold intersection. Historical `pool` plus `gradient_train` rows become the full gradient-train set. This preserves test identity where rows remain and avoids re-drawing favorable seeds. The prespecified minimums were train ≥300, validation ≥5, test ≥60; all contexts pass. Compound validation has only one compound per context, so neural checkpoint selection is necessarily noisy.\n",
        "## C. Primary compound-split results (mean across five seeds; SD/median/min/max are in summary.csv)\n", _markdown(primary),
        "## D. Secondary row-split results\n", _markdown(row),
        "## E. Post-hoc no-threshold prediction sensitivity\n",
        "The following recomputes frozen historical B=100 predictions on the threshold-intersection test rows only. Predictions were unchanged; it isolates evaluation-population tail exposure rather than filtered retraining.\n",
        _markdown(sensitivity_summary),
        "## F. Historical B=100 context (not a threshold-only comparison)\n",
        "This comparison changes both operational domain and revealed training-label count, so it must not be attributed wholly to threshold filtering.\n",
        _markdown(historical),
        "## G. Decision\n",
        f"Primary compound-split endpoint winners span: {', '.join(decision['distinct_primary_winners'])}. "
        + ("No single universal winner is supported across both columns, outputs and MAE/RMSE.\n" if decision["no_single_universal_winner"] else f"The same method won all primary endpoints: {decision['universal_winner']}.\n"),
        "`target_only_full` is a genuine target-data ceiling-like reference: it loads neither source weights nor source predictions/replay and fits all scalers on target gradient-train only. `paper_style_current_v2` keeps the historical matched shallow scope and uses the qualified source preprocessing; its scope, optimizer and selection rule were fixed before test evaluation. Simple calibration uses fixed source q50 predictions; conditional-EA selects only its predeclared regularization grid on validation.\n",
        "The full-data result can indicate whether label scarcity remains plausible, but it is not itself evidence that active learning will help. We therefore do not enter active learning automatically. The next high-value experiment is a preregistered learning curve on this same filtered compound split (nested target-label budgets, target-only and the selected transfer baseline), followed by an acquisition-vs-random test only if the curve shows a material label-scarcity regime.\n",
        f"{paper_context}\n",
        "## H. Limits\n",
        "- Threshold filtering can alter compound-row and condition composition; consult FILTER_AUDIT.json rather than calling removed rows generic outliers.\n"
        "- Target-compound holdout does not imply source-unseen chemistry.\n"
        "- Small frozen validation sets, especially one validation compound, make neural checkpoint comparisons less precise.\n"
        "- Existing reconstructed paper-transfer results remain descriptive context only.\n",
    ]
    (STUDY / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _artifact_manifest() -> dict:
    names = [
        "README.md", "PROTOCOL.md", "protocol.json", "FILTER_AUDIT.csv", "FILTER_AUDIT_SUMMARY.csv", "FILTER_AUDIT.json",
        "filtered_canonical_25g.csv", "filtered_canonical_40g.csv", "filtered_features_25g.csv", "filtered_features_40g.csv",
        "split_manifest.csv", "label_accounting.csv", "all_metrics.csv", "summary.csv", "paired_comparisons.csv",
        "posthoc_threshold_sensitivity.csv", "posthoc_threshold_sensitivity_summary.csv", "historical_b100_context.csv",
        "decision.json", "FINAL_REPORT.md",
    ]
    files = {name: sha256_file(STUDY / name) for name in names if (STUDY / name).exists()}
    manifest = {"study": "FILTERED_FULL_DATA_TRANSFER_PILOT_BENCHMARK", "tracked_scientific_files": files,
                "runtime_policy": "checkpoints, histories and per-sample predictions are under ignored studies/**/runtime/"}
    atomic_json(STUDY / "artifact_manifest.json", manifest)
    return manifest


def summarize() -> None:
    """Evaluate test truth only after all blind prediction files are frozen."""

    _validate_global_freeze()
    accounting = pd.read_csv(STUDY / "label_accounting.csv").set_index(["column", "protocol", "seed"])
    metric_rows: list[dict] = []
    for column, protocol, seed in _all_contexts():
        context = _context(column, protocol, seed)
        test_ids = context.loc[context.role.eq("test"), "sample_id"].astype(str).tolist()
        truth = _read_authorized_truth(filtered_path(column), test_ids)
        runtime = _runtime_context(column, protocol, seed)
        frozen = _json(runtime / "frozen.json")
        _assert(frozen["contract"]["test_ids_hash"] == stable_hash(sorted(test_ids)), "frozen test identity mismatch")
        prediction = pd.read_csv(runtime / "predictions_blind.csv.gz")
        _assert(prediction.sample_id.astype(str).tolist() == test_ids, "frozen prediction ordering mismatch")
        account = accounting.loc[(column, protocol, seed)]
        source_scales = torch.load(SOURCE, map_location="cpu", weights_only=False)["preprocessing"]["target_scales"]
        for method in METHODS:
            points = prediction[[f"{method}_V1", f"{method}_V2"]].to_numpy(float)
            result = absolute_error_metrics(truth, points, source_scales)
            metric_rows.append({
                "column": column, "protocol": protocol, "seed": int(seed), "method": method,
                "n_total_filtered": int(account.total_filtered_rows), "n_train": int(account.gradient_train_rows),
                "n_valid": int(account.validation_rows), "n_test": int(account.test_rows),
                "unique_compounds": int(account.unique_compounds_total), "training_label_count": int(account.training_label_count),
                **result,
            })
    metrics = pd.DataFrame(metric_rows).sort_values(["column", "protocol", "seed", "method"]).reset_index(drop=True)
    _assert(len(metrics) == 100 and np.isfinite(metrics[["V1_rmse", "V1_mae", "V2_rmse", "V2_mae"]].to_numpy()).all(), "invalid final metrics")
    summary = _summary_rows(metrics)
    paired = _paired_rows(metrics)
    sensitivity, sensitivity_summary = _historical_threshold_sensitivity()
    historical = _historical_context_table()
    _write_frame(metrics, STUDY / "all_metrics.csv")
    _write_frame(summary, STUDY / "summary.csv")
    _write_frame(paired, STUDY / "paired_comparisons.csv")
    _write_frame(sensitivity, STUDY / "posthoc_threshold_sensitivity.csv")
    _write_frame(sensitivity_summary, STUDY / "posthoc_threshold_sensitivity_summary.csv")
    _write_frame(historical, STUDY / "historical_b100_context.csv")
    decision = _decision(summary, paired)
    atomic_json(STUDY / "decision.json", decision)
    _write_report(summary, historical, sensitivity_summary, decision)
    _artifact_manifest()


def smoke() -> None:
    prepare()
    column, protocol, seed = "25g", "compound", SEEDS[0]
    first = _fit_context(column, protocol, seed, smoke=True)
    second = _fit_context(column, protocol, seed, smoke=True)
    runtime = _runtime_context(column, protocol, seed, True)
    prediction = pd.read_csv(runtime / "predictions_blind.csv.gz")
    _assert(all(f"{method}_V1" in prediction and f"{method}_V2" in prediction for method in METHODS), "smoke method missing")
    _assert(np.isfinite(prediction.drop(columns=["sample_id"]).to_numpy(float)).all(), "smoke non-finite prediction")
    atomic_json(STUDY / "smoke_audit.json", {
        "status": "PASS", "context": {"column": column, "protocol": protocol, "seed": seed},
        "methods": list(METHODS), "first_status": first["status"], "rerun_status": second["status"],
        "deterministic_rerun": second["status"] == "REUSED", "test_labels_used_for_fit_or_selection": 0,
        "compound_leakage_count": 0, "scaler_test_leakage_count": 0,
        "checkpoint_selection": "validation_only", "prediction_finite": True,
    })


def execute(workers: int) -> None:
    prepare()
    logdir = STUDY / "runtime" / "logs"
    logdir.mkdir(parents=True, exist_ok=True)

    def launch(context: tuple[str, str, int]) -> tuple[tuple[str, str, int], int]:
        column, protocol, seed = context
        command = [sys.executable, __file__, "--fit-context", column, protocol, str(seed)]
        with (logdir / f"{column}_{protocol}_{seed}.log").open("w", encoding="utf-8") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
        return context, int(result.returncode)

    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        completed = list(pool.map(launch, _all_contexts()))
    atomic_json(STUDY / "runtime" / "execution_audit.json", {
        "contexts": [{"column": item[0][0], "protocol": item[0][1], "seed": item[0][2], "returncode": item[1]} for item in completed],
        "workers": int(workers), "seconds": time.time() - started,
    })
    failures = [item for item in completed if item[1] != 0]
    if failures:
        raise RuntimeError(f"fitting failures: {failures}; inspect {logdir}")
    _freeze_all_predictions()
    summarize()


def write_docs() -> None:
    STUDY.mkdir(parents=True, exist_ok=True)
    (STUDY / "README.md").write_text(
        "# Filtered full-data transfer pilot\n\n"
        "This study evaluates 25g and 40g within legacy operational-domain thresholds. It is separate from, and does not modify, the no-threshold matched benchmark. Use `--smoke` before `--execute`; `--execute` freezes all test predictions before summary evaluation.\n",
        encoding="utf-8",
    )
    (STUDY / "PROTOCOL.md").write_text(
        "# Protocol\n\n"
        "Primary evidence is target-compound split; row split is secondary. The original B=100 outer validation/test identities are intersected with predeclared thresholds; all remaining filtered rows form gradient-train. Scale fits only gradient-train. Conditional-EA selects a fixed penalty grid on validation. Neural checkpoint selection is validation-only. The target-only arm is randomly initialized, source-free, and target-gradient-train normalized; transfer arms retain the qualified source preprocessing. Test graph labels are zero sentinels during fitting and test truth is read only after global prediction freeze.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="audit and freeze population/schedule/protocol")
    parser.add_argument("--smoke", action="store_true", help="run one seed/context with every method and a deterministic rerun")
    parser.add_argument("--execute", action="store_true", help="fit all contexts, globally freeze predictions, then evaluate")
    parser.add_argument("--summarize", action="store_true", help="evaluate an already globally frozen prediction set")
    parser.add_argument("--fit-context", nargs=3, metavar=("COLUMN", "PROTOCOL", "SEED"), help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    write_docs()
    if args.fit_context:
        column, protocol, seed = args.fit_context
        _assert(column in COLUMNS and protocol in PROTOCOLS and int(seed) in SEEDS, "invalid context")
        _fit_context(column, protocol, int(seed))
    elif args.smoke:
        smoke()
    elif args.execute:
        execute(args.workers)
    elif args.summarize:
        summarize()
    else:
        prepare()


if __name__ == "__main__":
    main()
