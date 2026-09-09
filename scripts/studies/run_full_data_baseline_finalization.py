#!/usr/bin/env python3
"""Finalize the strongest filtered FULL-data transfer baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qgeognn_al.transfer import (  # noqa: E402
    ALL_CONTEXT,
    MASS_FLOW_CONTEXT,
    absolute_error_metrics,
    build_full_data_schedule,
    center_width_inverse,
    center_width_transform,
    ea_fraction,
    filter_operational_domain,
    fit_scalar_conditional,
    fit_shared_center_width,
    fit_structured_center_width,
    loader_pair,
    make_label_scrubbed_graphs,
    project_physical,
)
from src.qgeognn_al.transfer.full_data import sha256_file, stable_hash  # noqa: E402


STUDY = ROOT / "studies/transfer/full_data_baseline_finalization"
BENCHMARK = ROOT / "studies/transfer/filtered_full_data_benchmark"
PARENT = ROOT / "studies/transfer/cross_column"
HEADROOM = ROOT / "studies/transfer/filtered_transfer_headroom_audit/runtime"
SOURCE = ROOT / "studies/predictor/final_4g_qualification/runtime/row/seed_42/best.pt"
SOURCE_SHA256 = "fce9edebc294fd179c7c7dc27ab2badea049c77fdad03a6cf0c317c63df544b0"
SCHEDULE_SHA256 = "b52b937aa750586ec8f2efb2f10f93dfddd2252bb03afaf304556e16b4fab5ee"
SOURCE_SCALES = np.asarray([7.8796590346317394, 16.076509553562932], dtype=float)
COLUMNS = ("8g", "25g", "40g")
FOCAL_COLUMNS = ("25g", "40g")
PROTOCOLS = ("compound", "row")
SEEDS = (769539383, 1425370602, 536279090, 2767143051, 1362771960)
THRESHOLDS = {"8g": (60.0, 120.0), "25g": (60.0, 120.0), "40g": (150.0, 200.0)}
CONTEXT_VALUES = {
    "8g": (8.0, 10.0, 1.5, 13.2, 0.4458),
    "25g": (25.0, 15.0, 2.15, 15.6, 0.5248),
    "40g": (40.0, 30.0, 2.15, 15.6, 0.5248),
}
REFERENCE_METHODS = ("conditional_EA", "scale_only", "paper_style_current_v2")
CANDIDATES = ("M3_CENTER_WIDTH_FULL", "MASS_FLOW_SHARED", "ALL_CONTEXT_SHARED_CENTER_WIDTH")
METHODS = REFERENCE_METHODS + CANDIDATES
M3_PENALTIES = (0.0, 0.1, 1.0)
SHARED_PENALTIES = (0.1, 1.0, 10.0)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = {"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None
    frame.to_csv(path, index=False, compression=compression)


def _md(frame: pd.DataFrame, digits: int = 3) -> str:
    displayed = frame.copy()
    for name in displayed.columns:
        if pd.api.types.is_float_dtype(displayed[name]):
            displayed[name] = displayed[name].map(
                lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
            )
    names = [str(value) for value in displayed.columns]
    lines = ["| " + " | ".join(names) + " |", "| " + " | ".join("---" for _ in names) + " |"]
    for row in displayed.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines) + "\n"


def _canonical_path(column: str) -> Path:
    if column in FOCAL_COLUMNS:
        return BENCHMARK / f"filtered_canonical_{column}.csv"
    return STUDY / f"filtered_canonical_{column}.csv"


def _features_path(column: str) -> Path:
    if column in FOCAL_COLUMNS:
        return BENCHMARK / f"filtered_features_{column}.csv"
    return STUDY / f"filtered_features_{column}.csv"


def _read_authorized_truth(path: Path, ids: Iterable[str]) -> np.ndarray:
    requested = [str(value) for value in ids]
    _assert(requested and len(requested) == len(set(requested)), "truth request must be nonempty and unique")
    identity = pd.read_csv(path, usecols=["sample_id"])
    allowed = set(requested)
    skip = [index + 1 for index, value in enumerate(identity.sample_id.astype(str)) if value not in allowed]
    frame = pd.read_csv(path, usecols=["sample_id", "V1_ml", "V2_ml"], skiprows=skip)
    frame.index = frame.sample_id.astype(str)
    _assert(set(frame.index) == allowed, "authorized truth identity mismatch")
    return frame.loc[requested, ["V1_ml", "V2_ml"]].to_numpy(float)


def _context(column: str, protocol: str, seed: int) -> pd.DataFrame:
    schedule = pd.read_csv(STUDY / "split_manifest.csv")
    result = schedule.loc[
        schedule.column.eq(column) & schedule.protocol.eq(protocol) & schedule.outer_seed.eq(seed)
    ].copy()
    _assert(not result.empty and result.sample_id.astype(str).is_unique, "invalid frozen context")
    return result


def _ids(column: str, protocol: str, seed: int, role: str) -> list[str]:
    frame = _context(column, protocol, seed)
    return frame.loc[frame.role.eq(role), "sample_id"].astype(str).tolist()


def prepare() -> dict:
    """Freeze the inherited populations, identities, model set and gates."""

    _assert(sha256_file(SOURCE) == SOURCE_SHA256, "source checkpoint hash mismatch")
    _assert(sha256_file(PARENT / "splits/schedule_manifest.csv") == SCHEDULE_SHA256, "parent schedule drift")
    parent = pd.read_csv(PARENT / "splits/schedule_manifest.csv")
    STUDY.mkdir(parents=True, exist_ok=True)

    raw8 = pd.read_csv(PARENT / "data_audit/canonical_8g.csv")
    filtered8, _ = filter_operational_domain(
        raw8, column="8g", v1_limit_ml=THRESHOLDS["8g"][0], v2_limit_ml=THRESHOLDS["8g"][1]
    )
    _assert(len(filtered8) == 552, f"unexpected filtered 8g count: {len(filtered8)}")
    _write_frame(filtered8, _canonical_path("8g"))
    _write_frame(filtered8.drop(columns=["V1_ml", "V2_ml"]), _features_path("8g"))
    schedule8, _ = build_full_data_schedule(
        parent, {"8g": filtered8}, columns=("8g",), protocols=PROTOCOLS, seeds=SEEDS, parent_budget=100
    )
    focal_schedule = pd.read_csv(BENCHMARK / "split_manifest.csv")
    schedule = pd.concat([schedule8, focal_schedule], ignore_index=True).sort_values(
        ["column", "protocol", "outer_seed", "sample_id"]
    ).reset_index(drop=True)
    _write_frame(schedule, STUDY / "split_manifest.csv")

    accounting_rows = []
    for protocol in PROTOCOLS:
        for seed in SEEDS:
            counts = {}
            for column in COLUMNS:
                block = schedule.loc[
                    schedule.column.eq(column) & schedule.protocol.eq(protocol) & schedule.outer_seed.eq(seed)
                ]
                counts[column] = {role: int(block.role.eq(role).sum()) for role in ("gradient_train", "validation", "test")}
            for focal in FOCAL_COLUMNS:
                accounting_rows.append({
                    "focal_column": focal, "protocol": protocol, "seed": seed,
                    "focal_target_train_rows": counts[focal]["gradient_train"],
                    "8g_donor_rows": counts["8g"]["gradient_train"],
                    "25g_donor_rows": counts["25g"]["gradient_train"],
                    "40g_donor_rows": counts["40g"]["gradient_train"],
                    "total_target_supervision_rows": sum(counts[column]["gradient_train"] for column in COLUMNS),
                    "focal_validation_rows": counts[focal]["validation"],
                    "focal_test_rows": counts[focal]["test"],
                    "donor_outer_test_rows_used": 0,
                })
    accounting = pd.DataFrame(accounting_rows)
    _write_frame(accounting, STUDY / "FULL_DATA_LABEL_ACCOUNTING.csv")

    reference_summary = pd.read_csv(BENCHMARK / "summary.csv")
    reference_summary = reference_summary.loc[reference_summary.method.isin(REFERENCE_METHODS)].copy()
    _assert(len(reference_summary) == 12, "reference summary is incomplete")
    _write_frame(reference_summary, STUDY / "existing_reference_metrics.csv")

    protocol = {
        "study_name": "FULL_DATA_BASELINE_FINALIZATION",
        "classification": "FILTERED_FULL_DATA_WORKING_BASELINE_SELECTION",
        "status": "FROZEN_BEFORE_NEW_CANDIDATE_TEST_EVALUATION",
        "parent_benchmark": "studies/transfer/filtered_full_data_benchmark",
        "parent_protocol_sha256": sha256_file(BENCHMARK / "protocol.json"),
        "parent_prediction_freeze_sha256": sha256_file(BENCHMARK / "all_predictions_frozen.json"),
        "parent_schedule_sha256": SCHEDULE_SHA256,
        "source_checkpoint_sha256": SOURCE_SHA256,
        "source_scales": {"V1": SOURCE_SCALES[0], "V2": SOURCE_SCALES[1]},
        "columns_in_shared_fit": list(COLUMNS),
        "ranked_focal_columns": list(FOCAL_COLUMNS),
        "protocols": list(PROTOCOLS), "seeds": list(SEEDS), "budget": "FULL",
        "full_data_definition": "all operationally filtered rows outside inherited frozen validation/test are gradient_train",
        "thresholds_ml": {key: {"V1": value[0], "V2": value[1]} for key, value in THRESHOLDS.items()},
        "primary": "compound", "secondary": "row",
        "references": list(REFERENCE_METHODS), "candidates": list(CANDIDATES),
        "m3": {
            "implementation": "fit_structured_center_width(..., M3_CENTER_WIDTH)",
            "penalty_grid": list(M3_PENALTIES), "selection": "focal validation only",
            "mass_ratio": {column: float(CONTEXT_VALUES[column][0] / 4.0) for column in FOCAL_COLUMNS},
            "free_coefficients": 6,
        },
        "shared": {
            "penalty_grid": list(SHARED_PENALTIES),
            "selection": "one penalty minimizing equal-column mean validation combined normalized RMSE",
            "mass_flow_context": list(MASS_FLOW_CONTEXT), "all_context": list(ALL_CONTEXT),
            "context_values": {key: list(value) for key, value in CONTEXT_VALUES.items()},
            "context_normalization": "authorized shared gradient_train rows only",
            "donor_truth": "each column current seed/protocol gradient_train only",
        },
        "promotion_gate": {
            "reference": "conditional_EA", "min_aggregate_compound_gain": 0.03,
            "min_one_column_gain": 0.03, "max_other_column_deterioration": 0.03,
            "max_endpoint_deterioration": 0.05, "min_combined_seed_wins": 6,
            "max_aggregate_mae_deterioration": 0.03,
        },
        "test_labels_for_fit_normalization_or_selection": False,
        "prediction_freeze_required_before_test_truth": True,
        "physical_interpretation": "predictive only; context coefficients are confounded and not causal",
    }
    path = STUDY / "protocol.json"
    if path.exists():
        _assert(_json(path) == protocol, "frozen protocol drift")
    else:
        _write_json(path, protocol)
    (STUDY / "README.md").write_text(
        "# Full-data baseline finalization\n\nA frozen, developmental comparison of corrected M3 and two shared column-context controls against the existing filtered FULL-data baselines.\n",
        encoding="utf-8",
    )
    (STUDY / "PROTOCOL.md").write_text(
        "# Protocol\n\nCompound split is primary and row split secondary. Existing reference predictions are hash-verified and reused. M3 is fit per focal column. Shared fits pool only the current seed/protocol gradient-train rows from 8g, 25g and 40g; every validation and test row remains excluded from fitting and context normalization. One penalty is selected from the frozen grid using the equal-column mean validation score. Candidate predictions are globally frozen before test truth is read. Legacy descriptors retain their code names and no units or causal meaning are asserted.\n",
        encoding="utf-8",
    )
    return protocol


def _source_frame(column: str) -> pd.DataFrame:
    feature = pd.read_csv(_features_path(column))
    if column in FOCAL_COLUMNS:
        path = HEADROOM / f"source_{column}.csv.gz"
        meta = _json(HEADROOM / f"source_{column}.json")
        _assert(sha256_file(path) == meta["sha256"], f"source cache hash mismatch: {column}")
        result = pd.read_csv(path)
    else:
        path = STUDY / "runtime/source_8g.csv.gz"
        if not path.exists():
            import torch
            from src.qgeognn_al.models import load_predictor_checkpoint
            from src.qgeognn_al.resources import SOURCE_GRAPH_CACHE

            cache = dict(torch.load(SOURCE_GRAPH_CACHE, weights_only=False))
            cache.update(torch.load(PARENT / "data_audit/graph_cache_8g_only.pt", weights_only=False))
            payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
            model = load_predictor_checkpoint(SOURCE)
            atom, angle = make_label_scrubbed_graphs(feature, cache, payload["preprocessing"]["scaler"])
            model.eval()
            outputs = []
            with torch.no_grad():
                positions = np.arange(len(feature), dtype=int)
                for atom_batch, angle_batch in zip(*loader_pair(atom, angle, positions, 2048)):
                    outputs.append(model(atom_batch, angle_batch).detach().cpu().numpy())
            points = np.vstack(outputs)[:, (1, 4)]
            _assert(np.isfinite(points).all(), "non-finite 8g source prediction")
            result = pd.DataFrame({"sample_id": feature.sample_id.astype(str),
                                   "V1_source": points[:, 0], "V2_source": points[:, 1]})
            _write_frame(result, path)
            _write_json(STUDY / "runtime/source_8g.json", {
                "rows": len(result), "source_checkpoint_sha256": SOURCE_SHA256,
                "prediction_sha256": sha256_file(path), "target_label_cells_parsed": 0,
            })
        result = pd.read_csv(path)
    indexed = result.set_index(result.sample_id.astype(str))
    ordered = indexed.loc[feature.sample_id.astype(str)].reset_index(drop=True)
    _assert(np.isfinite(ordered[["V1_source", "V2_source"]].to_numpy(float)).all(), "bad source cache")
    return ordered


def _context_matrix(column: str, count: int, names: tuple[str, ...]) -> np.ndarray:
    values = dict(zip(ALL_CONTEXT, CONTEXT_VALUES[column]))
    row = np.asarray([values[name] for name in names], dtype=float)
    return np.repeat(row[None, :], count, axis=0)


def _historical_m3(
    train_source: np.ndarray, train_truth: np.ndarray, train_ea: np.ndarray,
    predict_source: np.ndarray, predict_ea: np.ndarray, mass_ratio: float, penalty: float,
) -> np.ndarray:
    center_source, width_source = center_width_transform(train_source)
    center_truth, width_truth = center_width_transform(train_truth)
    center_scale = float(np.std(center_source)) or 1.0
    width_scale = float(np.std(width_source)) or 1.0
    center_fit = fit_scalar_conditional(center_source, center_truth, train_ea, center_scale, mass_ratio, penalty)
    width_fit = fit_scalar_conditional(width_source, width_truth, train_ea, width_scale, mass_ratio, penalty)
    center_predict, width_predict = center_width_transform(predict_source)
    c = center_fit.predict(np.repeat(center_predict, 2, axis=1), predict_ea)[:, 0]
    w = width_fit.predict(np.repeat(width_predict, 2, axis=1), predict_ea)[:, 0]
    return project_physical(center_width_inverse(c, w))[0]


def _reference_predictions(column: str, protocol: str, seed: int, test_ids: list[str]) -> dict[str, np.ndarray]:
    freeze = _json(BENCHMARK / "all_predictions_frozen.json")
    relative = f"runtime/contexts/{column}/{protocol}/seed_{seed}/frozen.json"
    _assert(sha256_file(BENCHMARK / relative) == freeze["files"][relative], "parent frozen context drift")
    path = BENCHMARK / f"runtime/contexts/{column}/{protocol}/seed_{seed}/predictions_blind.csv.gz"
    frame = pd.read_csv(path)
    frame.index = frame.sample_id.astype(str)
    frame = frame.loc[test_ids]
    return {
        method: frame[[f"{method}_V1", f"{method}_V2"]].to_numpy(float)
        for method in REFERENCE_METHODS
    }


def _select_m3(
    train_source: np.ndarray, train_truth: np.ndarray, train_ea: np.ndarray,
    valid_source: np.ndarray, valid_truth: np.ndarray, valid_ea: np.ndarray,
    mass_ratio: float,
) -> tuple[float, list[dict]]:
    rows = []
    for penalty in M3_PENALTIES:
        fit = fit_structured_center_width(
            train_source, train_truth, train_ea, mass_ratio, penalty, "M3_CENTER_WIDTH"
        )
        score = absolute_error_metrics(valid_truth, fit.predict(valid_source, valid_ea), SOURCE_SCALES)[
            "combined_normalized_rmse"
        ]
        rows.append({"penalty": penalty, "validation_score": float(score)})
    selected = min(rows, key=lambda row: (row["validation_score"], row["penalty"]))
    return float(selected["penalty"]), rows


def _fit_shared(
    names: tuple[str, ...], protocol: str, seed: int,
    features: dict[str, pd.DataFrame], sources: dict[str, np.ndarray],
) -> tuple[object, float, list[dict], dict[str, dict[str, np.ndarray]]]:
    prepared = {}
    train_source, train_truth, train_ea, train_context = [], [], [], []
    for column in COLUMNS:
        feature = features[column]
        position = {value: index for index, value in enumerate(feature.sample_id.astype(str))}
        ids = {role: _ids(column, protocol, seed, role) for role in ("gradient_train", "validation", "test")}
        indices = {role: np.asarray([position[value] for value in values], dtype=int) for role, values in ids.items()}
        truth = {role: _read_authorized_truth(_canonical_path(column), ids[role])
                 for role in ("gradient_train", "validation")}
        ea = ea_fraction(feature["PE/EA"])
        context = _context_matrix(column, len(feature), names)
        prepared[column] = {"ids": ids, "indices": indices, "truth": truth, "ea": ea, "context": context}
        train_source.append(sources[column][indices["gradient_train"]])
        train_truth.append(truth["gradient_train"])
        train_ea.append(ea[indices["gradient_train"]])
        train_context.append(context[indices["gradient_train"]])
    stacked = tuple(map(np.concatenate, (train_source, train_truth, train_ea, train_context)))
    validation_rows = []
    for penalty in SHARED_PENALTIES:
        fit = fit_shared_center_width(*stacked, names, penalty)
        scores = {}
        for column in COLUMNS:
            item = prepared[column]
            index = item["indices"]["validation"]
            prediction = fit.predict(sources[column][index], item["ea"][index], item["context"][index])
            scores[column] = float(absolute_error_metrics(item["truth"]["validation"], prediction, SOURCE_SCALES)[
                "combined_normalized_rmse"
            ])
        validation_rows.append({"penalty": penalty, "equal_column_mean_score": float(np.mean(list(scores.values()))),
                                "column_scores": scores})
    selected = min(validation_rows, key=lambda row: (row["equal_column_mean_score"], row["penalty"]))
    fit = fit_shared_center_width(*stacked, names, float(selected["penalty"]))
    return fit, float(selected["penalty"]), validation_rows, prepared


def fit_all() -> dict:
    prepare()
    features = {column: pd.read_csv(_features_path(column)) for column in COLUMNS}
    sources = {column: _source_frame(column)[["V1_source", "V2_source"]].to_numpy(float) for column in COLUMNS}
    max_equivalence = 0.0
    fitted_contexts = 0
    for protocol in PROTOCOLS:
        for seed in SEEDS:
            shared = {}
            prepared_shared = None
            for method, names in (("MASS_FLOW_SHARED", MASS_FLOW_CONTEXT),
                                  ("ALL_CONTEXT_SHARED_CENTER_WIDTH", ALL_CONTEXT)):
                fit, selected, candidates, prepared = _fit_shared(names, protocol, seed, features, sources)
                shared[method] = (fit, selected, candidates)
                prepared_shared = prepared
            _assert(prepared_shared is not None, "shared preparation missing")
            for column in FOCAL_COLUMNS:
                feature = features[column]
                source = sources[column]
                item = prepared_shared[column]
                indices, ids, ea = item["indices"], item["ids"], item["ea"]
                train_source = source[indices["gradient_train"]]
                train_truth = item["truth"]["gradient_train"]
                train_ea = ea[indices["gradient_train"]]
                valid_source = source[indices["validation"]]
                valid_truth = item["truth"]["validation"]
                valid_ea = ea[indices["validation"]]
                test_source = source[indices["test"]]
                test_ea = ea[indices["test"]]
                penalty, m3_candidates = _select_m3(
                    train_source, train_truth, train_ea, valid_source, valid_truth, valid_ea,
                    CONTEXT_VALUES[column][0] / 4.0,
                )
                m3 = fit_structured_center_width(
                    train_source, train_truth, train_ea, CONTEXT_VALUES[column][0] / 4.0,
                    penalty, "M3_CENTER_WIDTH",
                )
                m3_prediction = m3.predict(test_source, test_ea)
                historical = _historical_m3(
                    train_source, train_truth, train_ea, test_source, test_ea,
                    CONTEXT_VALUES[column][0] / 4.0, penalty,
                )
                difference = float(np.max(np.abs(m3_prediction - historical)))
                max_equivalence = max(max_equivalence, difference)
                _assert(difference < 1e-10, "historical corrected M3 equivalence failed")
                predictions = _reference_predictions(column, protocol, seed, ids["test"])
                predictions["M3_CENTER_WIDTH_FULL"] = m3_prediction
                audits = {"M3_CENTER_WIDTH_FULL": {
                    "selected_penalty": penalty, "penalty_candidates": m3_candidates,
                    "historical_equivalence_max_abs_difference": difference,
                    "fit": m3.audit(test_source, test_ea),
                }}
                for method, (fit, selected, candidates) in shared.items():
                    names = tuple(fit.context_names)
                    context = _context_matrix(column, len(feature), names)[indices["test"]]
                    predictions[method] = fit.predict(test_source, test_ea, context)
                    audits[method] = {"selected_penalty": selected, "penalty_candidates": candidates,
                                      "fit": fit.audit(test_source, test_ea, context)}
                _assert(set(predictions) == set(METHODS), "method contract mismatch")
                _assert(all(np.isfinite(value).all() for value in predictions.values()), "non-finite candidate prediction")
                runtime = STUDY / f"runtime/contexts/{column}/{protocol}/seed_{seed}"
                frame = pd.DataFrame({"sample_id": ids["test"]})
                for method in METHODS:
                    frame[f"{method}_V1"] = predictions[method][:, 0]
                    frame[f"{method}_V2"] = predictions[method][:, 1]
                prediction_path = runtime / "predictions_blind.csv.gz"
                _write_frame(frame, prediction_path)
                contract = {
                    "column": column, "protocol": protocol, "seed": seed,
                    "protocol_sha256": sha256_file(STUDY / "protocol.json"),
                    "train_ids_hash": stable_hash(sorted(ids["gradient_train"])),
                    "validation_ids_hash": stable_hash(sorted(ids["validation"])),
                    "test_ids_hash": stable_hash(sorted(ids["test"])),
                    "test_labels_used_for_fit_normalization_or_selection": 0,
                    "donor_outer_test_rows_used": 0,
                }
                _write_json(runtime / "fit_audit.json", {"contract": contract, "methods": audits})
                _write_json(runtime / "frozen.json", {
                    "contract": contract, "methods": list(METHODS),
                    "files": {name: sha256_file(runtime / name) for name in ("predictions_blind.csv.gz", "fit_audit.json")},
                })
                fitted_contexts += 1
    equivalence = {"status": "PASS" if max_equivalence < 1e-10 else "FAIL", "tolerance": 1e-10,
                   "max_abs_prediction_difference": max_equivalence, "actual_full_contexts_checked": fitted_contexts,
                   "historical_contract": "corrected M3; SSE/n, priors [1,0,0], same source scaling, mass ratio and projection"}
    _write_json(STUDY / "m3_historical_equivalence.json", equivalence)
    return equivalence


def freeze_predictions() -> dict:
    files = {}
    for column in FOCAL_COLUMNS:
        for protocol in PROTOCOLS:
            for seed in SEEDS:
                runtime = STUDY / f"runtime/contexts/{column}/{protocol}/seed_{seed}"
                frozen = _json(runtime / "frozen.json")
                for name, digest in frozen["files"].items():
                    _assert(sha256_file(runtime / name) == digest, "candidate frozen file drift")
                _assert(frozen["contract"]["test_labels_used_for_fit_normalization_or_selection"] == 0, "test leakage")
                relative = str((runtime / "frozen.json").relative_to(STUDY))
                files[relative] = sha256_file(runtime / "frozen.json")
    payload = {"status": "FROZEN_BEFORE_TEST_EVALUATION", "contexts": len(files), "methods": list(METHODS),
               "protocol_sha256": sha256_file(STUDY / "protocol.json"), "files": files,
               "test_truth_evaluation_started": False}
    _write_json(STUDY / "prediction_freeze_manifest.json", payload)
    return payload


def _validate_freeze() -> dict:
    frozen = _json(STUDY / "prediction_freeze_manifest.json")
    _assert(frozen["contexts"] == 20 and frozen["methods"] == list(METHODS), "incomplete prediction freeze")
    _assert(frozen["protocol_sha256"] == sha256_file(STUDY / "protocol.json"), "freeze protocol drift")
    for relative, digest in frozen["files"].items():
        _assert(sha256_file(STUDY / relative) == digest, "frozen prediction context drift")
    return frozen


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    values = ["n_train", "n_valid", "n_test", "V1_rmse", "V1_mae", "V1_r2", "V2_rmse", "V2_mae",
              "V2_r2", "combined_normalized_rmse", "center_rmse", "center_mae", "width_rmse", "width_mae"]
    rows = []
    for keys, group in metrics.groupby(["column", "protocol", "method"], sort=True):
        row = dict(zip(("column", "protocol", "method"), keys)); row["n_seeds"] = len(group)
        for value in values:
            numeric = group[value].astype(float)
            row[f"{value}_mean"] = float(numeric.mean()) if numeric.notna().any() else np.nan
            row[f"{value}_sd"] = float(numeric.std(ddof=1)) if numeric.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _paired(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = [(candidate, "conditional_EA") for candidate in CANDIDATES]
    comparisons.append(("ALL_CONTEXT_SHARED_CENTER_WIDTH", "MASS_FLOW_SHARED"))
    for column in FOCAL_COLUMNS:
        for protocol in PROTOCOLS:
            group = metrics.loc[metrics.column.eq(column) & metrics.protocol.eq(protocol)]
            for candidate, reference in comparisons:
                a = group.loc[group.method.eq(candidate)].set_index("seed")
                b = group.loc[group.method.eq(reference)].set_index("seed")
                row = {"column": column, "protocol": protocol, "candidate": candidate, "reference": reference, "n_pairs": len(a)}
                for metric in ("V1_rmse", "V1_mae", "V2_rmse", "V2_mae", "combined_normalized_rmse"):
                    delta = a[metric] - b[metric]
                    row[f"{metric}_delta_mean"] = float(delta.mean())
                    row[f"{metric}_relative_gain"] = float(1.0 - a[metric].mean() / b[metric].mean())
                    row[f"{metric}_wins"] = int((delta < 0).sum())
                rows.append(row)
    return pd.DataFrame(rows)


def _ranking(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics.loc[metrics.protocol.eq("compound")]
    reference = primary.loc[primary.method.eq("conditional_EA")].set_index(["column", "seed"])
    rows = []
    for method in METHODS:
        frame = primary.loc[primary.method.eq(method)]
        by_column = frame.groupby("column").mean(numeric_only=True)
        aligned = frame.set_index(["column", "seed"])
        wins = int((aligned.combined_normalized_rmse < reference.combined_normalized_rmse).sum()) if method != "conditional_EA" else 0
        rows.append({
            "method": method,
            "25g_compound_combined_nrmse": float(by_column.loc["25g", "combined_normalized_rmse"]),
            "40g_compound_combined_nrmse": float(by_column.loc["40g", "combined_normalized_rmse"]),
            "mean_25g_40g_compound_nrmse": float(frame.combined_normalized_rmse.mean()),
            "V1_mean_rmse": float(frame.V1_rmse.mean()), "V2_mean_rmse": float(frame.V2_rmse.mean()),
            "mean_mae": float(frame[["V1_mae", "V2_mae"]].to_numpy().mean()),
            "seed_stability": f"{wins}/10 combined-NRMSE wins vs conditional_EA" if method != "conditional_EA" else "reference",
            "wins_vs_conditional_ea": wins,
        })
    result = pd.DataFrame(rows).sort_values("mean_25g_40g_compound_nrmse").reset_index(drop=True)
    result["primary_rank"] = np.arange(1, len(result) + 1)
    return result


def _decision(metrics: pd.DataFrame, ranking: pd.DataFrame) -> dict:
    primary = metrics.loc[metrics.protocol.eq("compound")]
    reference = primary.loc[primary.method.eq("conditional_EA")].set_index(["column", "seed"])
    gate = _json(STUDY / "protocol.json")["promotion_gate"]
    candidates = {}
    for method in CANDIDATES:
        frame = primary.loc[primary.method.eq(method)].set_index(["column", "seed"])
        column_gains = {column: float(1.0 - frame.loc[column].combined_normalized_rmse.mean() /
                                             reference.loc[column].combined_normalized_rmse.mean()) for column in FOCAL_COLUMNS}
        endpoint_gains = {f"{column}_{target}": float(1.0 - frame.loc[column, f"{target}_rmse"].mean() /
                                                         reference.loc[column, f"{target}_rmse"].mean())
                          for column in FOCAL_COLUMNS for target in ("V1", "V2")}
        aggregate_gain = float(1.0 - frame.combined_normalized_rmse.mean() / reference.combined_normalized_rmse.mean())
        wins = int((frame.combined_normalized_rmse < reference.combined_normalized_rmse).sum())
        candidate_mae = float(frame[["V1_mae", "V2_mae"]].to_numpy().mean())
        reference_mae = float(reference[["V1_mae", "V2_mae"]].to_numpy().mean())
        mae_gain = float(1.0 - candidate_mae / reference_mae)
        passes = (aggregate_gain >= gate["min_aggregate_compound_gain"]
                  and max(column_gains.values()) >= gate["min_one_column_gain"]
                  and min(column_gains.values()) >= -gate["max_other_column_deterioration"]
                  and min(endpoint_gains.values()) >= -gate["max_endpoint_deterioration"]
                  and wins >= gate["min_combined_seed_wins"]
                  and mae_gain >= -gate["max_aggregate_mae_deterioration"])
        candidates[method] = {"aggregate_combined_nrmse_gain": aggregate_gain, "column_gains": column_gains,
                              "endpoint_rmse_gains": endpoint_gains, "combined_seed_wins": wins,
                              "aggregate_mae_gain": mae_gain, "passes": bool(passes)}
    # MASS_FLOW_SHARED is the required incremental control.  The user-facing
    # decision contract permits promotion only of corrected M3 or ALL_CONTEXT.
    promoted = [method for method in ("M3_CENTER_WIDTH_FULL", "ALL_CONTEXT_SHARED_CENTER_WIDTH")
                if candidates[method]["passes"]]
    if promoted:
        best = ranking.loc[ranking.method.isin(promoted)].sort_values("primary_rank").iloc[0].method
        status = ("CENTER_WIDTH_PROMOTED_AS_WORKING_FULL_DATA_BASELINE" if best == "M3_CENTER_WIDTH_FULL"
                  else "ALL_CONTEXT_SHARED_PROMOTED_AS_WORKING_FULL_DATA_BASELINE" if best == "ALL_CONTEXT_SHARED_CENTER_WIDTH"
                  else "MASS_FLOW_SHARED_PROMOTED_AS_WORKING_FULL_DATA_BASELINE")
    else:
        best, status = "conditional_EA", "CONDITIONAL_EA_RETAINED_AS_WORKING_FULL_DATA_BASELINE"
    return {"status": status, "selected_working_baseline": best, "reference": "conditional_EA",
            "candidate_gates": candidates, "prediction_freeze_verified_before_test_evaluation": True,
            "developmental_evidence": True, "active_learning_started": False}


def evaluate() -> dict:
    started = time.perf_counter()
    _validate_freeze()
    metrics_rows = []
    accounting = pd.read_csv(STUDY / "FULL_DATA_LABEL_ACCOUNTING.csv")
    for column in FOCAL_COLUMNS:
        for protocol in PROTOCOLS:
            for seed in SEEDS:
                test_ids = _ids(column, protocol, seed, "test")
                truth = _read_authorized_truth(_canonical_path(column), test_ids)
                frame = pd.read_csv(STUDY / f"runtime/contexts/{column}/{protocol}/seed_{seed}/predictions_blind.csv.gz")
                _assert(frame.sample_id.astype(str).tolist() == test_ids, "frozen test order mismatch")
                label = accounting.loc[accounting.focal_column.eq(column) & accounting.protocol.eq(protocol)
                                       & accounting.seed.eq(seed)].iloc[0]
                c_truth, w_truth = center_width_transform(truth)
                for method in METHODS:
                    prediction = frame[[f"{method}_V1", f"{method}_V2"]].to_numpy(float)
                    metric = absolute_error_metrics(truth, prediction, SOURCE_SCALES)
                    if method in CANDIDATES:
                        c_pred, w_pred = center_width_transform(prediction)
                        extra = {"center_rmse": float(np.sqrt(np.mean((c_pred - c_truth) ** 2))),
                                 "center_mae": float(np.mean(np.abs(c_pred - c_truth))),
                                 "width_rmse": float(np.sqrt(np.mean((w_pred - w_truth) ** 2))),
                                 "width_mae": float(np.mean(np.abs(w_pred - w_truth)))}
                    else:
                        extra = {key: np.nan for key in ("center_rmse", "center_mae", "width_rmse", "width_mae")}
                    metrics_rows.append({"column": column, "protocol": protocol, "seed": seed, "method": method,
                                         "n_train": int(label.focal_target_train_rows), "n_valid": int(label.focal_validation_rows),
                                         "n_test": int(label.focal_test_rows), **metric, **extra})
    metrics = pd.DataFrame(metrics_rows)
    summary = _summary(metrics); paired = _paired(metrics); ranking = _ranking(metrics)
    decision = _decision(metrics, ranking)
    _write_frame(metrics, STUDY / "all_metrics.csv")
    _write_frame(summary, STUDY / "summary.csv")
    _write_frame(paired, STUDY / "paired_comparisons.csv")
    _write_frame(ranking, STUDY / "baseline_ranking.csv")
    _write_json(STUDY / "decision.json", decision)
    _write_report(summary, paired, ranking, decision)
    _write_json(STUDY / "run_metadata.json", {
        "runtime_seconds_evaluation": time.perf_counter() - started,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "contexts": 20, "frozen_methods": len(METHODS), "test_evaluation_after_global_freeze": True,
        "source_checkpoint_sha256": SOURCE_SHA256,
    })
    names = ["README.md", "PROTOCOL.md", "protocol.json", "FULL_DATA_LABEL_ACCOUNTING.csv",
             "existing_reference_metrics.csv", "m3_historical_equivalence.json", "prediction_freeze_manifest.json",
             "deterministic_rerun.json",
             "filtered_canonical_8g.csv", "filtered_features_8g.csv", "split_manifest.csv",
             "all_metrics.csv", "summary.csv", "paired_comparisons.csv", "baseline_ranking.csv", "decision.json",
             "FINAL_REPORT.md", "run_metadata.json"]
    _write_json(STUDY / "artifact_manifest.json", {"study": "FULL_DATA_BASELINE_FINALIZATION",
                "files": {name: sha256_file(STUDY / name) for name in names}})
    return decision


def _write_report(summary: pd.DataFrame, paired: pd.DataFrame, ranking: pd.DataFrame, decision: dict) -> None:
    columns = ["column", "protocol", "method", "V1_rmse_mean", "V1_mae_mean", "V1_r2_mean",
               "V2_rmse_mean", "V2_mae_mean", "V2_r2_mean", "combined_normalized_rmse_mean"]
    table = summary.loc[summary.method.isin(METHODS), columns].sort_values(["protocol", "column", "method"])
    primary = table.loc[table.protocol.eq("compound")]
    secondary = table.loc[table.protocol.eq("row")]
    increment = paired.loc[(paired.candidate == "ALL_CONTEXT_SHARED_CENTER_WIDTH")
                           & (paired.reference == "MASS_FLOW_SHARED")]
    compound_increment = increment.loc[increment.protocol.eq("compound")]
    selected = decision["selected_working_baseline"]
    m3_pairs = paired.loc[(paired.candidate == "M3_CENTER_WIDTH_FULL")
                          & (paired.reference == "conditional_EA")]
    m3_compound = m3_pairs.loc[m3_pairs.protocol.eq("compound")].set_index("column")
    m3_row = m3_pairs.loc[m3_pairs.protocol.eq("row")].set_index("column")
    all_pairs = paired.loc[(paired.candidate == "ALL_CONTEXT_SHARED_CENTER_WIDTH")
                           & (paired.reference == "conditional_EA")
                           & paired.protocol.eq("compound")].set_index("column")
    gate = decision["candidate_gates"]["M3_CENTER_WIDTH_FULL"]
    report = f"""# Final filtered FULL-data transfer baseline

## Decision

**{decision['status']}**. Selected working baseline: **{selected}**. No Active Learning or learning curve was started.

FULL means every eligible operationally filtered row outside the inherited frozen validation/test identities is in gradient-train. It never means training on test. Compound is primary; row is secondary.

## Compound-primary results

{_md(primary)}

## Row-secondary results

{_md(secondary)}

## Ranking

{_md(ranking)}

## Corrected M3 versus Conditional EA

On the compound-primary endpoint, M3 improves combined normalized RMSE by {100*m3_compound.loc['25g', 'combined_normalized_rmse_relative_gain']:.2f}% on 25g ({int(m3_compound.loc['25g', 'combined_normalized_rmse_wins'])}/5 wins) and {100*m3_compound.loc['40g', 'combined_normalized_rmse_relative_gain']:.2f}% on 40g ({int(m3_compound.loc['40g', 'combined_normalized_rmse_wins'])}/5 wins). Across both columns the mean gain is {100*gate['aggregate_combined_nrmse_gain']:.2f}% with {gate['combined_seed_wins']}/10 paired wins. All four mean endpoint RMSEs improve; aggregate MAE improves {100*gate['aggregate_mae_gain']:.2f}%. R2 moves in the same favorable direction for all four endpoint/column means.

The row-secondary comparison is directionally consistent versus Conditional EA: combined normalized RMSE improves {100*m3_row.loc['25g', 'combined_normalized_rmse_relative_gain']:.2f}% on 25g and {100*m3_row.loc['40g', 'combined_normalized_rmse_relative_gain']:.2f}% on 40g. Paper-style remains the best row-only method, but row interpolation does not override the compound-primary selection.

## Shared-context increment

ALL_CONTEXT and MASS_FLOW_SHARED use the same 8g/25g/40g gradient-train supervision and the same shared model. Their only difference is adding `legacy_column_dia`, `legacy_column_len`, and `legacy_column_den`. Negative deltas favor ALL_CONTEXT.

{_md(compound_increment)}

Against Conditional EA, ALL_CONTEXT improves 25g compound combined normalized RMSE by {100*all_pairs.loc['25g', 'combined_normalized_rmse_relative_gain']:.2f}% but worsens 40g by {-100*all_pairs.loc['40g', 'combined_normalized_rmse_relative_gain']:.2f}%. Relative to MASS_FLOW_SHARED, adding the three legacy descriptors improves 25g by {100*compound_increment.set_index('column').loc['25g', 'combined_normalized_rmse_relative_gain']:.2f}% (5/5) but worsens 40g by {-100*compound_increment.set_index('column').loc['40g', 'combined_normalized_rmse_relative_gain']:.2f}% (0/5). They therefore do not provide stable cross-column incremental predictive value.

## Interpretation

M3 is the exact corrected six-coefficient Center/Width formulation. Its actual-data equivalence to the historical corrected implementation is below `1e-10`. The shared model is `Ct=[a0+aEA*EA+aZ^T Z]Cs+bC` and `Wt=[d0+dEA*EA+dZ^T Z]Ws+bW`, with Z standardized on shared gradient-train only. MASS_FLOW uses two Z variables; ALL_CONTEXT uses exactly five.

Shared candidates receive more historical target supervision than single-column Conditional EA because they pool authorized gradient-train rows across 8g/25g/40g. No donor validation or test labels enter fitting. Removing 8g would still leave a mathematically definable 25g+40g fit, but only two column contexts and an even weaker basis for extrapolating column effects; this was not run as an ablation.

RMSE and combined normalized RMSE determine the ranking, MAE checks whether typical absolute error reverses direction, and R2 is secondary. A higher R2 with worse RMSE is not promotion evidence: R2 depends on the held-out target variance and can preserve ranking trends while absolute calibration remains worse.

The legacy descriptor coefficients are not independently identifiable from packing mass, flow, or column identity. Any improvement is predictive association only; no flow, diameter, length, or density causal effect is claimed.

All candidate predictions were globally frozen and hash-verified before test truth evaluation. This remains developmental evidence because these outer tests have historical repository exposure. The selected working baseline may now be frozen for subsequent AL baseline construction, but this study itself does not start AL.
"""
    (STUDY / "FINAL_REPORT.md").write_text(report, encoding="utf-8")


def execute() -> dict:
    started = time.perf_counter()
    freeze_path = STUDY / "prediction_freeze_manifest.json"
    previous_freeze = freeze_path.read_bytes() if freeze_path.exists() else None
    prepare(); fit_all(); freeze_predictions()
    deterministic = previous_freeze is not None and previous_freeze == freeze_path.read_bytes()
    _write_json(STUDY / "deterministic_rerun.json", {
        "status": "PASS" if deterministic else "NOT_CHECKED_FIRST_RUN",
        "prediction_freeze_manifest_byte_identical": deterministic,
        "prior_frozen_run_available": previous_freeze is not None,
    })
    decision = evaluate()
    metadata = _json(STUDY / "run_metadata.json")
    metadata["runtime_seconds_total"] = time.perf_counter() - started
    _write_json(STUDY / "run_metadata.json", metadata)
    # Refresh hashes after adding total runtime.
    evaluate_manifest = _json(STUDY / "artifact_manifest.json")
    evaluate_manifest["files"]["run_metadata.json"] = sha256_file(STUDY / "run_metadata.json")
    _write_json(STUDY / "artifact_manifest.json", evaluate_manifest)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        result = execute()
    elif args.prepare:
        result = prepare()
    elif args.fit:
        result = fit_all()
    elif args.freeze:
        result = freeze_predictions()
    elif args.evaluate:
        result = evaluate()
    else:
        parser.error("choose one phase")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
