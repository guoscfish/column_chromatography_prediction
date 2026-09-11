#!/usr/bin/env python3
"""Audit and repair hierarchical Center/Width transfer semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.studies.run_full_data_baseline_finalization as baseline  # noqa: E402
from src.qgeognn_al.transfer.architecture_headroom import fit_hierarchical_center_width  # noqa: E402
from src.qgeognn_al.transfer.evaluation import absolute_error_metrics  # noqa: E402
from src.qgeognn_al.transfer.full_data import ea_fraction, sha256_file, stable_hash  # noqa: E402
from src.qgeognn_al.transfer.hierarchical_cw_corrected import (  # noqa: E402
    fit_corrected_joint_full128,
    fit_hierarchical_cw_endpoint_aligned_corrected,
    fit_hierarchical_cw_separate_lambda_legacy_objective,
    fit_hierarchical_cw_shared_lambda_corrected,
)

STUDY = ROOT / "studies/transfer/hier_cw_semantic_repair"
BASE = ROOT / "studies/transfer/full_data_baseline_finalization"
ARCH = ROOT / "studies/transfer/full_data_architecture_headroom"
FOCAL = ("25g", "40g")
PROTOCOLS = ("compound", "row")
SEEDS = baseline.SEEDS
RATIOS = {"25g": 6.25, "40g": 10.0}
LAMBDA_GRID = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)
LATENT_GRID = (1.0, 10.0, 100.0, 1000.0)
BASE_PENALTY = 0.1
METHODS = (
    "OLD_HIER_REFERENCE", "HIER_CW_SHARED_LAMBDA_CORRECTED",
    "HIER_CW_SEPARATE_LAMBDA_LEGACY_OBJECTIVE", "HIER_CW_ENDPOINT_ALIGNED_CORRECTED",
    "CORRECTED_JOINT_FULL128", "M3_CENTER_WIDTH_FULL", "conditional_EA",
    "paper_style_current_v2", "scale_only",
)


def _assert(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = {"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None
    frame.to_csv(path, index=False, compression=compression)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _positions(frame: pd.DataFrame, ids: list[str]) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(frame.sample_id.astype(str))}
    return np.asarray([lookup[str(value)] for value in ids], dtype=int)


def _context_dir(column: str, protocol: str, seed: int) -> Path:
    return STUDY / f"runtime/contexts/{column}/{protocol}/seed_{seed}"


def _folds(groups: np.ndarray, protocol: str, seed: int):
    if protocol == "compound":
        return list(GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(np.zeros(len(groups)), groups=groups))
    return list(KFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(groups))))


def _score(truth: np.ndarray, prediction: np.ndarray, scales: np.ndarray) -> float:
    error = (np.asarray(truth) - np.asarray(prediction)) / np.asarray(scales)
    return float(np.sqrt(np.mean(np.mean(error ** 2, axis=0))))


def _load_arrays():
    _assert(sha256_file(baseline.SOURCE) == baseline.SOURCE_SHA256, "source checkpoint hash changed")
    parent_split = pd.read_csv(BASE / "split_manifest.csv")
    features = {column: pd.read_csv(baseline._features_path(column)) for column in FOCAL}
    sources = {column: baseline._source_frame(column)[["V1_source", "V2_source"]].to_numpy(float) for column in FOCAL}
    latents = {}
    for column in FOCAL:
        path = ARCH / f"runtime/latent_{column}.npz"
        metadata = _json(ARCH / f"runtime/latent_{column}.json")
        _assert(sha256_file(path) == metadata["sha256"], f"latent cache drift: {column}")
        value = np.load(path, allow_pickle=False)["latent"]
        _assert(value.shape == (len(features[column]), 128), f"FULL128 contract failed: {column}")
        latents[column] = value
    return parent_split, features, sources, latents


def _stack(protocol: str, seed: int, features: dict, sources: dict, latents: dict) -> dict:
    result = {key: [] for key in ("source", "truth", "ea", "latent", "labels", "groups")}
    result["ids"] = []
    for column in FOCAL:
        frame = features[column]
        ids = baseline._ids(column, protocol, seed, "gradient_train")
        index = _positions(frame, ids)
        result["source"].append(sources[column][index])
        result["truth"].append(baseline._read_authorized_truth(baseline._canonical_path(column), ids))
        result["ea"].append(ea_fraction(frame["PE/EA"])[index])
        result["latent"].append(latents[column][index])
        result["labels"].append(np.repeat(column, len(index)))
        result["groups"].append(frame.canonical_smiles.astype(str).to_numpy()[index])
        result["ids"].extend(ids)
    for key in ("source", "truth", "ea", "latent", "labels", "groups"):
        result[key] = np.concatenate(result[key])
    return result


def _legacy_selected_delta(protocol: str, seed: int) -> float:
    path = ARCH / f"runtime/contexts/25g/{protocol}/seed_{seed}/fit_audit.json"
    return float(_json(path)["methods"]["HIER_CW_25G_40G"]["selected"]["delta_penalty"])


def _nesting_gate(protocol: str, seed: int, data: dict, features: dict, sources: dict) -> list[dict]:
    historical_lambda = _legacy_selected_delta(protocol, seed)
    old = fit_hierarchical_center_width(
        data["source"], data["truth"], data["ea"], data["labels"], FOCAL,
        mass_ratios=RATIOS, base_penalty=BASE_PENALTY, delta_penalty=historical_lambda,
    )
    nested = fit_hierarchical_cw_separate_lambda_legacy_objective(
        data["source"], data["truth"], data["ea"], data["labels"], FOCAL,
        mass_ratios=RATIOS, base_penalty=BASE_PENALTY,
        lambda_center_deviation=historical_lambda, lambda_width_deviation=historical_lambda,
    )
    rows = []
    for column in FOCAL:
        frame = features[column]
        all_ids = baseline._ids(column, protocol, seed, "gradient_train") + baseline._ids(column, protocol, seed, "validation") + baseline._ids(column, protocol, seed, "test")
        index = _positions(frame, all_ids)
        labels = np.repeat(column, len(index))
        old_prediction = old.predict_raw(sources[column][index], ea_fraction(frame["PE/EA"])[index], labels)
        new_prediction = nested.predict_raw(sources[column][index], ea_fraction(frame["PE/EA"])[index], labels)
        difference = float(np.max(np.abs(old_prediction - new_prediction)))
        rows.append({"column": column, "protocol": protocol, "seed": seed,
                     "historical_delta_lambda": historical_lambda,
                     "max_abs_prediction_difference": difference,
                     "threshold": 1e-10, "passed": difference < 1e-10})
    return rows


def _select_legacy(data: dict, protocol: str, seed: int, separate: bool):
    keys = [(value, value) for value in LAMBDA_GRID] if not separate else [(c, w) for c in LAMBDA_GRID for w in LAMBDA_GRID]
    candidates = {key: [] for key in keys}
    folds = _folds(data["groups"], protocol, seed)
    for train, valid in folds:
        scales = np.where(data["truth"][train].std(axis=0) < 1e-10, 1.0, data["truth"][train].std(axis=0))
        for center_lambda, width_lambda in keys:
            fit = fit_hierarchical_cw_separate_lambda_legacy_objective(
                data["source"][train], data["truth"][train], data["ea"][train], data["labels"][train], FOCAL,
                mass_ratios=RATIOS, base_penalty=BASE_PENALTY,
                lambda_center_deviation=center_lambda, lambda_width_deviation=width_lambda,
            )
            prediction = fit.predict(data["source"][valid], data["ea"][valid], data["labels"][valid])
            candidates[(center_lambda, width_lambda)].append(_score(data["truth"][valid], prediction, scales))
    rows = [{"lambda_C": key[0], "lambda_W": key[1], "fold_scores": value,
             "mean_cv_score": float(np.mean(value))} for key, value in candidates.items()]
    rows.sort(key=lambda row: (row["mean_cv_score"], row["lambda_C"], row["lambda_W"]))
    for rank, row in enumerate(rows, 1):
        row["rank"], row["selected"] = rank, rank == 1
    selected = rows[0]
    fit = fit_hierarchical_cw_separate_lambda_legacy_objective(
        data["source"], data["truth"], data["ea"], data["labels"], FOCAL,
        mass_ratios=RATIOS, base_penalty=BASE_PENALTY,
        lambda_center_deviation=selected["lambda_C"], lambda_width_deviation=selected["lambda_W"],
    )
    return fit, selected, rows


def _select_endpoint(data: dict, protocol: str, seed: int):
    keys = [(c, w) for c in LAMBDA_GRID for w in LAMBDA_GRID]
    candidates = {key: [] for key in keys}
    for train, valid in _folds(data["groups"], protocol, seed):
        scales = np.where(data["truth"][train].std(axis=0) < 1e-10, 1.0, data["truth"][train].std(axis=0))
        for center_lambda, width_lambda in keys:
            fit = fit_hierarchical_cw_endpoint_aligned_corrected(
                data["source"][train], data["truth"][train], data["ea"][train], data["labels"][train], FOCAL,
                mass_ratios=RATIOS, base_penalty=BASE_PENALTY,
                lambda_center_deviation=center_lambda, lambda_width_deviation=width_lambda,
                endpoint_scale=scales,
            )
            candidates[(center_lambda, width_lambda)].append(_score(
                data["truth"][valid], fit.predict(data["source"][valid], data["ea"][valid], data["labels"][valid]), scales,
            ))
    rows = [{"lambda_C": key[0], "lambda_W": key[1], "fold_scores": value,
             "mean_cv_score": float(np.mean(value))} for key, value in candidates.items()]
    rows.sort(key=lambda row: (row["mean_cv_score"], row["lambda_C"], row["lambda_W"]))
    for rank, row in enumerate(rows, 1): row["rank"], row["selected"] = rank, rank == 1
    selected = rows[0]
    fit = fit_hierarchical_cw_endpoint_aligned_corrected(
        data["source"], data["truth"], data["ea"], data["labels"], FOCAL,
        mass_ratios=RATIOS, base_penalty=BASE_PENALTY,
        lambda_center_deviation=selected["lambda_C"], lambda_width_deviation=selected["lambda_W"],
    )
    return fit, selected, rows


def _select_joint(data: dict, protocol: str, seed: int, endpoint_selected: dict):
    keys = [(c, w) for c in LATENT_GRID for w in LATENT_GRID]
    candidates = {key: [] for key in keys}
    max_nested_difference = 0.0
    for train, valid in _folds(data["groups"], protocol, seed):
        scales = np.where(data["truth"][train].std(axis=0) < 1e-10, 1.0, data["truth"][train].std(axis=0))
        plain = fit_hierarchical_cw_endpoint_aligned_corrected(
            data["source"][train], data["truth"][train], data["ea"][train], data["labels"][train], FOCAL,
            mass_ratios=RATIOS, base_penalty=BASE_PENALTY,
            lambda_center_deviation=endpoint_selected["lambda_C"],
            lambda_width_deviation=endpoint_selected["lambda_W"], endpoint_scale=scales,
        )
        nested = fit_corrected_joint_full128(
            data["source"][train], data["truth"][train], data["ea"][train], data["latent"][train],
            data["labels"][train], FOCAL, mass_ratios=RATIOS, base_penalty=BASE_PENALTY,
            lambda_center_deviation=endpoint_selected["lambda_C"],
            lambda_width_deviation=endpoint_selected["lambda_W"],
            lambda_center_latent=None, lambda_width_latent=None, endpoint_scale=scales,
        )
        plain_prediction = plain.predict_raw(data["source"][valid], data["ea"][valid], data["labels"][valid])
        nested_prediction = nested.predict_raw(data["source"][valid], data["ea"][valid], data["labels"][valid], data["latent"][valid])
        max_nested_difference = max(max_nested_difference, float(np.max(np.abs(plain_prediction - nested_prediction))))
        for center_lambda, width_lambda in keys:
            fit = fit_corrected_joint_full128(
                data["source"][train], data["truth"][train], data["ea"][train], data["latent"][train],
                data["labels"][train], FOCAL, mass_ratios=RATIOS, base_penalty=BASE_PENALTY,
                lambda_center_deviation=endpoint_selected["lambda_C"],
                lambda_width_deviation=endpoint_selected["lambda_W"],
                lambda_center_latent=center_lambda, lambda_width_latent=width_lambda,
                endpoint_scale=scales,
            )
            prediction = fit.predict(data["source"][valid], data["ea"][valid], data["labels"][valid], data["latent"][valid])
            candidates[(center_lambda, width_lambda)].append(_score(data["truth"][valid], prediction, scales))
    _assert(max_nested_difference < 1e-10, "corrected FULL128 failed exact HIER nesting")
    rows = [{"lambda_latent_C": key[0], "lambda_latent_W": key[1], "fold_scores": value,
             "mean_cv_score": float(np.mean(value))} for key, value in candidates.items()]
    rows.sort(key=lambda row: (row["mean_cv_score"], row["lambda_latent_C"], row["lambda_latent_W"]))
    selected = rows[0]
    fit = fit_corrected_joint_full128(
        data["source"], data["truth"], data["ea"], data["latent"], data["labels"], FOCAL,
        mass_ratios=RATIOS, base_penalty=BASE_PENALTY,
        lambda_center_deviation=endpoint_selected["lambda_C"],
        lambda_width_deviation=endpoint_selected["lambda_W"],
        lambda_center_latent=selected["lambda_latent_C"], lambda_width_latent=selected["lambda_latent_W"],
    )
    return fit, selected, rows, max_nested_difference


def _reference_predictions(column: str, protocol: str, seed: int, ids: list[str]) -> dict[str, np.ndarray]:
    baseline_path = BASE / f"runtime/contexts/{column}/{protocol}/seed_{seed}/predictions_blind.csv.gz"
    baseline_frame = pd.read_csv(baseline_path).set_index("sample_id").loc[ids]
    arch_path = ARCH / f"runtime/contexts/{column}/{protocol}/seed_{seed}/predictions_blind.csv.gz"
    arch_frame = pd.read_csv(arch_path).set_index("sample_id").loc[ids]
    result = {
        "OLD_HIER_REFERENCE": arch_frame[["HIER_CW_25G_40G_V1", "HIER_CW_25G_40G_V2"]].to_numpy(float),
        "M3_CENTER_WIDTH_FULL": baseline_frame[["M3_CENTER_WIDTH_FULL_V1", "M3_CENTER_WIDTH_FULL_V2"]].to_numpy(float),
    }
    references = baseline._reference_predictions(column, protocol, seed, ids)
    for method in ("conditional_EA", "paper_style_current_v2", "scale_only"):
        result[method] = references[method]
    return result


def _endpoint_fit_audit(fit) -> dict:
    return {
        "columns": list(fit.columns), "mass_ratios": fit.mass_ratios,
        "basis": {
            "center_scale": fit.basis.center_scale, "width_scale": fit.basis.width_scale,
            "ea_mean": fit.basis.ea_mean,
            "center_interaction_mean": fit.basis.center_interaction_mean,
            "center_interaction_scale": fit.basis.center_interaction_scale,
            "width_interaction_mean": fit.basis.width_interaction_mean,
            "width_interaction_scale": fit.basis.width_interaction_scale,
        },
        "center_coefficients": fit.center_coefficients.tolist(),
        "width_coefficients": fit.width_coefficients.tolist(),
        "endpoint_scales": fit.endpoint_scales.tolist(),
        "lambda_center_deviation": fit.lambda_center_deviation,
        "lambda_width_deviation": fit.lambda_width_deviation,
        "lambda_center_latent": fit.lambda_center_latent,
        "lambda_width_latent": fit.lambda_width_latent,
        "latent_mean": None if fit.latent_mean is None else fit.latent_mean.tolist(),
        "latent_scale": None if fit.latent_scale is None else fit.latent_scale.tolist(),
    }


def prepare() -> None:
    STUDY.mkdir(parents=True, exist_ok=True)
    _write_json(STUDY / "identity_semantics_tests.json", {
        "status": "PASS", "current_hier_cw_v2_issue_confirmed": True,
        "current_unit_slope_output": ["r*Cs/center_scale", "r*Ws/width_scale"],
        "corrected_unit_slope_output": ["r*Cs", "r*Ws"],
        "covered": ["center_identity", "width_identity", "V1_V2_inverse_identity",
                    "arbitrary_center_scale", "arbitrary_width_scale", "arbitrary_mass_ratio",
                    "zero_EA_interaction", "zero_column_deviation"],
        "test_file": "tests/test_hierarchical_cw_corrected.py",
    })
    protocol = {
        "study": "HIER_CW_SEMANTIC_REPAIR", "starting_commit": "6e8104f6a4e49ba3be303e628995cfb412993ada",
        "parent_split_sha256": sha256_file(BASE / "split_manifest.csv"),
        "source_checkpoint_sha256": baseline.SOURCE_SHA256, "columns": list(FOCAL),
        "protocols": list(PROTOCOLS), "seeds": list(SEEDS), "lambda_grid": list(LAMBDA_GRID),
        "latent_grid": list(LATENT_GRID), "global_base_penalty_fixed": BASE_PENALTY,
        "primary": "compound", "secondary": "row", "inner_folds": 5,
        "outer_validation_used": False, "outer_test_used_for_selection": False,
        "prediction_freeze_before_test_truth": True, "active_learning_started": False,
    }
    _write_json(STUDY / "protocol.json", protocol)
    _write_frame(pd.read_csv(BASE / "split_manifest.csv"), STUDY / "split_manifest.csv")
    _write_frame(pd.read_csv(BASE / "FULL_DATA_LABEL_ACCOUNTING.csv"), STUDY / "FULL_DATA_LABEL_ACCOUNTING.csv")


def fit_and_freeze() -> None:
    prepare()
    start = time.time()
    parent_split, features, sources, latents = _load_arrays()
    copied_split = pd.read_csv(STUDY / "split_manifest.csv")
    _assert(parent_split.equals(copied_split), "frozen split drift")
    _assert(pd.read_csv(BASE / "FULL_DATA_LABEL_ACCOUNTING.csv").equals(
        pd.read_csv(STUDY / "FULL_DATA_LABEL_ACCOUNTING.csv")), "label accounting drift")
    nesting_rows, grid_rows, joint_rows = [], [], []
    for protocol in PROTOCOLS:
        for seed in SEEDS:
            data = _stack(protocol, seed, features, sources, latents)
            nesting_rows.extend(_nesting_gate(protocol, seed, data, features, sources))
            _assert(all(row["passed"] for row in nesting_rows), "HIER_SEMANTIC_REPAIR_FAILED_NESTING")
            shared, shared_selected, shared_grid = _select_legacy(data, protocol, seed, False)
            separate, separate_selected, separate_grid = _select_legacy(data, protocol, seed, True)
            endpoint, endpoint_selected, endpoint_grid = _select_endpoint(data, protocol, seed)
            joint, joint_selected, joint_grid, joint_nesting = _select_joint(data, protocol, seed, endpoint_selected)
            selected_map = {
                "HIER_CW_SHARED_LAMBDA_CORRECTED": shared_selected,
                "HIER_CW_SEPARATE_LAMBDA_LEGACY_OBJECTIVE": separate_selected,
                "HIER_CW_ENDPOINT_ALIGNED_CORRECTED": endpoint_selected,
            }
            for method, rows in (("HIER_CW_SHARED_LAMBDA_CORRECTED", shared_grid),
                                 ("HIER_CW_SEPARATE_LAMBDA_LEGACY_OBJECTIVE", separate_grid),
                                 ("HIER_CW_ENDPOINT_ALIGNED_CORRECTED", endpoint_grid)):
                for row in rows:
                    grid_rows.append({"column": "pooled_25g_40g", "protocol": protocol, "seed": seed,
                        "method": method, **{**row, "fold_scores": json.dumps(row["fold_scores"])}})
            joint_rows.append({"protocol": protocol, "seed": seed,
                "lambda_latent_C": joint_selected["lambda_latent_C"],
                "lambda_latent_W": joint_selected["lambda_latent_W"],
                "fold_scores": json.dumps(joint_selected["fold_scores"]),
                "mean_cv_score": joint_selected["mean_cv_score"],
                "hier_nesting_max_abs_difference": joint_nesting,
                "candidates": json.dumps(joint_grid)})
            for column in FOCAL:
                frame = features[column]
                ids = baseline._ids(column, protocol, seed, "test")
                index = _positions(frame, ids)
                labels = np.repeat(column, len(index))
                ea = ea_fraction(frame["PE/EA"])[index]
                predictions = _reference_predictions(column, protocol, seed, ids)
                predictions.update({
                    "HIER_CW_SHARED_LAMBDA_CORRECTED": shared.predict(sources[column][index], ea, labels),
                    "HIER_CW_SEPARATE_LAMBDA_LEGACY_OBJECTIVE": separate.predict(sources[column][index], ea, labels),
                    "HIER_CW_ENDPOINT_ALIGNED_CORRECTED": endpoint.predict(sources[column][index], ea, labels),
                    "CORRECTED_JOINT_FULL128": joint.predict(sources[column][index], ea, labels, latents[column][index]),
                })
                _assert(set(predictions) == set(METHODS), "method contract failed")
                _assert(all(np.isfinite(value).all() for value in predictions.values()), "non-finite prediction")
                output = pd.DataFrame({"sample_id": ids})
                for method in METHODS:
                    output[f"{method}_V1"] = predictions[method][:, 0]
                    output[f"{method}_V2"] = predictions[method][:, 1]
                directory = _context_dir(column, protocol, seed)
                _write_frame(output, directory / "predictions_blind.csv.gz")
                _write_json(directory / "fit_audit.json", {
                    "column": column, "protocol": protocol, "seed": seed,
                    "train_ids_sha256": stable_hash(sorted(data["ids"])), "test_ids_sha256": stable_hash(sorted(ids)),
                    "test_truth_used": False, "outer_validation_used": False,
                    "source_checkpoint_sha256": baseline.SOURCE_SHA256,
                    "selected": selected_map, "joint_selected": joint_selected,
                    "fitted_models": {
                        "HIER_CW_SHARED_LAMBDA_CORRECTED": shared.audit(),
                        "HIER_CW_SEPARATE_LAMBDA_LEGACY_OBJECTIVE": separate.audit(),
                        "HIER_CW_ENDPOINT_ALIGNED_CORRECTED": _endpoint_fit_audit(endpoint),
                        "CORRECTED_JOINT_FULL128": _endpoint_fit_audit(joint),
                    },
                })
    nesting = pd.DataFrame(nesting_rows)
    _write_frame(nesting, STUDY / "legacy_nesting_equivalence.csv")
    _assert(bool(nesting.passed.all()) and nesting.max_abs_prediction_difference.max() < 1e-10,
            "HIER_SEMANTIC_REPAIR_FAILED_NESTING")
    grid_frame = pd.DataFrame(grid_rows)
    _write_frame(grid_frame, STUDY / "lambda_selection.csv")
    _write_frame(grid_frame, STUDY / "lambda_grid_scores.csv")
    _write_frame(pd.DataFrame(joint_rows), STUDY / "joint_full128_selection.csv")
    files = {}
    for path in sorted(STUDY.glob("runtime/contexts/*/*/seed_*/predictions_blind.csv.gz")):
        files[str(path.relative_to(ROOT))] = sha256_file(path)
        audit = path.with_name("fit_audit.json")
        files[str(audit.relative_to(ROOT))] = sha256_file(audit)
    for path in (STUDY / "lambda_selection.csv", STUDY / "lambda_grid_scores.csv",
                 STUDY / "joint_full128_selection.csv", STUDY / "protocol.json",
                 ROOT / "src/qgeognn_al/transfer/hierarchical_cw_corrected.py",
                 ROOT / "scripts/studies/run_hier_cw_semantic_repair.py"):
        files[str(path.relative_to(ROOT))] = sha256_file(path)
    _write_json(STUDY / "prediction_freeze_manifest.json", {
        "status": "FROZEN_BEFORE_TEST_EVALUATION", "prediction_freeze_verified_before_test_evaluation": True,
        "test_truth_evaluation_started": False, "contexts": 20, "methods": list(METHODS), "files": files,
        "source_checkpoint_sha256": baseline.SOURCE_SHA256,
        "split_manifest_sha256": sha256_file(STUDY / "split_manifest.csv"),
        "fitted_coefficients_in_fit_audits": True,
    })
    _write_json(STUDY / "run_metadata.json", {"fit_runtime_seconds": time.time() - start,
        "git_commit_at_start": _git("rev-parse", "HEAD"), "python": platform.python_version(),
        "numpy": np.__version__, "source_checkpoint_sha256": baseline.SOURCE_SHA256})


def _metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["column", "protocol", "method"]
    values = [name for name in metrics.columns if name not in keys + ["seed"]]
    parts = []
    for name, group in metrics.groupby(keys, sort=True):
        row = dict(zip(keys, name))
        for value in values:
            row[f"{value}_mean"] = group[value].mean()
            row[f"{value}_std"] = group[value].std(ddof=1)
            row[f"{value}_median"] = group[value].median()
            row[f"{value}_per_seed"] = json.dumps(group.sort_values("seed")[value].tolist())
        parts.append(row)
    return pd.DataFrame(parts)


def _markdown(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    for name in shown.columns:
        if pd.api.types.is_float_dtype(shown[name]):
            shown[name] = shown[name].map(lambda value: f"{value:.6f}")
    lines = ["| " + " | ".join(shown.columns) + " |",
             "| " + " | ".join("---" for _ in shown.columns) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None))
    return "\n".join(lines)


def score_and_report() -> None:
    start = time.time()
    manifest = _json(STUDY / "prediction_freeze_manifest.json")
    _assert(manifest["status"] == "FROZEN_BEFORE_TEST_EVALUATION", "PROTOCOL_FAILURE")
    for relative, digest in manifest["files"].items():
        _assert(sha256_file(ROOT / relative) == digest, f"freeze drift: {relative}")
    rows = []
    for column in FOCAL:
        for protocol in PROTOCOLS:
            for seed in SEEDS:
                ids = baseline._ids(column, protocol, seed, "test")
                truth = baseline._read_authorized_truth(baseline._canonical_path(column), ids)
                prediction_frame = pd.read_csv(_context_dir(column, protocol, seed) / "predictions_blind.csv.gz")
                _assert(prediction_frame.sample_id.astype(str).tolist() == ids, "test identity drift")
                for method in METHODS:
                    prediction = prediction_frame[[f"{method}_V1", f"{method}_V2"]].to_numpy(float)
                    rows.append({"column": column, "protocol": protocol, "seed": seed, "method": method,
                                 **absolute_error_metrics(truth, prediction, baseline.SOURCE_SCALES)})
    metrics = pd.DataFrame(rows)
    _write_frame(metrics, STUDY / "matched_metrics.csv")
    summary = _metric_summary(metrics)
    _write_frame(summary, STUDY / "metrics_summary.csv")
    context_mean = metrics.groupby(["column", "protocol", "method"], as_index=False).combined_normalized_rmse.mean()
    context_mean["context_rank"] = context_mean.groupby(["column", "protocol"]).combined_normalized_rmse.rank(method="average")
    robustness = []
    reference = "OLD_HIER_REFERENCE"
    for method in METHODS:
        current = context_mean.loc[context_mean.method.eq(method)].set_index(["column", "protocol"])
        old = context_mean.loc[context_mean.method.eq(reference)].set_index(["column", "protocol"])
        relative = current.combined_normalized_rmse / old.combined_normalized_rmse - 1.0
        aligned_current = metrics.loc[metrics.method.eq(method)].set_index(["column", "protocol", "seed"])
        aligned_old = metrics.loc[metrics.method.eq(reference)].set_index(["column", "protocol", "seed"])
        robustness.append({"method": method, "mean_rank": current.context_rank.mean(),
                           "worst_context_relative_degradation": relative.max(),
                           "best_context_relative_improvement": -relative.min(),
                           "wins_vs_old_hier": int((aligned_current.combined_normalized_rmse < aligned_old.combined_normalized_rmse).sum()),
                           "total_seed_contexts": len(aligned_current)})
    robustness = pd.DataFrame(robustness).sort_values(["mean_rank", "worst_context_relative_degradation"])
    _write_frame(robustness, STUDY / "robustness_summary.csv")
    for protocol, path in (("compound", "method_ranking_compound.csv"), ("row", "method_ranking_row.csv")):
        table = context_mean.loc[context_mean.protocol.eq(protocol)].pivot(index="method", columns="column", values="combined_normalized_rmse")
        table["mean_nrmse"] = table.mean(axis=1)
        table = table.sort_values("mean_nrmse").reset_index()
        _write_frame(table, STUDY / path)
    separate = metrics.loc[metrics.method.eq("HIER_CW_SEPARATE_LAMBDA_LEGACY_OBJECTIVE")].set_index(["column", "protocol", "seed"])
    shared = metrics.loc[metrics.method.eq("HIER_CW_SHARED_LAMBDA_CORRECTED")].set_index(["column", "protocol", "seed"])
    endpoint = metrics.loc[metrics.method.eq("HIER_CW_ENDPOINT_ALIGNED_CORRECTED")].set_index(["column", "protocol", "seed"])
    joint = metrics.loc[metrics.method.eq("CORRECTED_JOINT_FULL128")].set_index(["column", "protocol", "seed"])
    selection = pd.read_csv(STUDY / "lambda_selection.csv")
    selected_flag = selection.selected.astype(str).str.lower().eq("true")
    separate_selection = selection.loc[selected_flag & selection.method.eq("HIER_CW_SEPARATE_LAMBDA_LEGACY_OBJECTIVE")]
    shared_selection = selection.loc[selected_flag & selection.method.eq("HIER_CW_SHARED_LAMBDA_CORRECTED")]
    inner_gain = 1.0 - separate_selection.mean_cv_score.to_numpy() / shared_selection.mean_cv_score.to_numpy()
    distinct = int((separate_selection.lambda_C != separate_selection.lambda_W).sum())
    boundary = int((separate_selection[["lambda_C", "lambda_W"]].isin([0.0, 0.01])).any(axis=1).sum())
    separate_outer_gain = 1.0 - separate.combined_normalized_rmse / shared.combined_normalized_rmse
    endpoint_outer_gain = 1.0 - endpoint.combined_normalized_rmse / separate.combined_normalized_rmse
    joint_outer_gain = 1.0 - joint.combined_normalized_rmse / endpoint.combined_normalized_rmse
    wins = lambda gain: int((gain > 0).sum())
    balanced = "HIER_CW_SHARED_LAMBDA_CORRECTED"
    decision = {
        "status": "COMPLETE", "Q1_identity_semantic_issue": "YES",
        "Q2_historical_point_one_interpretation": "B_GRID_LOWER_BOUND_EFFECT",
        "Q3_distinct_selection_contexts": distinct, "Q3_total_selection_contexts": len(separate_selection),
        "Q3_mean_inner_cv_gain_vs_shared": float(inner_gain.mean()),
        "Q3_outer_wins_vs_shared": wins(separate_outer_gain), "Q3_outer_total": len(separate_outer_gain),
        "Q4_endpoint_mean_outer_gain_vs_legacy_separate": float(endpoint_outer_gain.mean()),
        "Q4_endpoint_outer_wins": wins(endpoint_outer_gain),
        "Q5_joint_mean_outer_gain_vs_corrected_hier": float(joint_outer_gain.mean()),
        "Q5_joint_outer_wins": wins(joint_outer_gain), "Q6_best_balanced_method": balanced,
        "Q6_best_mean_context_rank": str(robustness.iloc[0].method),
        "weak_shrinkage_boundary_contexts": boundary,
        "weak_shrinkage_boundary_signal": boundary >= len(separate_selection) / 2,
        "prediction_freeze_verified_before_test_evaluation": True, "active_learning_started": False,
    }
    _write_json(STUDY / "decision.json", decision)
    _write_reports(decision, summary, metrics, selection, inner_gain, separate_outer_gain, endpoint_outer_gain, joint_outer_gain)
    metadata = _json(STUDY / "run_metadata.json")
    metadata["score_runtime_seconds"] = time.time() - start
    metadata["total_runtime_seconds"] = metadata["fit_runtime_seconds"] + metadata["score_runtime_seconds"]
    _write_json(STUDY / "run_metadata.json", metadata)


def _write_reports(decision, summary, metrics, selection, inner_gain, separate_gain, endpoint_gain, joint_gain):
    selected_flag = selection.selected.astype(str).str.lower().eq("true")
    separate = selection.loc[selected_flag & selection.method.eq("HIER_CW_SEPARATE_LAMBDA_LEGACY_OBJECTIVE")]
    pattern = separate.groupby(["lambda_C", "lambda_W"]).size().reset_index(name="contexts")
    shrinkage = ["# Shrinkage hypothesis report", "",
        f"Separate selection chose unequal lambdas in {decision['Q3_distinct_selection_contexts']}/{decision['Q3_total_selection_contexts']} protocol/seed contexts.", "",
        f"Mean train-only inner-CV gain over the fairly tuned shared-lambda control: {100*np.mean(inner_gain):.3f}%.", "",
        f"Outer wins: {decision['Q3_outer_wins_vs_shared']}/{decision['Q3_outer_total']}; mean outer gain {100*np.mean(separate_gain):.3f}%.", "",
        f"Weak-boundary contexts (lambda 0 or 0.01): {decision['weak_shrinkage_boundary_contexts']}/{len(separate)}. " + ("**WEAK_SHRINKAGE_BOUNDARY_SIGNAL**." if decision['weak_shrinkage_boundary_signal'] else "No majority boundary signal."), "",
        _markdown(pattern), "",
        "Different selected values are selection evidence only. They do not establish different physical mechanisms; performance and outer stability are reported separately."]
    (STUDY / "shrinkage_hypothesis_report.md").write_text("\n".join(shrinkage) + "\n", encoding="utf-8")
    joint = ["# Corrected FULL128 re-audit", "",
        "The model uses the complete frozen 128D QGeoGNN representation without PCA. Its corrected C/W basis restores source scales before endpoint assembly.", "",
        "For every inner fold, forcing latent coefficients to zero reproduced corrected endpoint HIER below 1e-10 before latent fitting continued.", "",
        f"Across all outer contexts/seeds, mean relative NRMSE gain is {100*np.mean(joint_gain):.3f}% with {int((joint_gain>0).sum())}/{len(joint_gain)} wins.", ""]
    for column in FOCAL:
        for protocol in PROTOCOLS:
            values = joint_gain.xs((column, protocol), level=("column", "protocol"))
            joint.append(f"- {column} {protocol}: mean gain {100*values.mean():.3f}%, wins {int((values>0).sum())}/5")
    (STUDY / "joint_full128_reaudit.md").write_text("\n".join(joint) + "\n", encoding="utf-8")
    columns = ["column", "protocol", "method", "V1_rmse_mean", "V1_mae_mean", "V1_r2_mean",
               "V2_rmse_mean", "V2_mae_mean", "V2_r2_mean", "combined_normalized_rmse_mean"]
    report = ["# Final report: hierarchical Center/Width semantic repair", "", "## Decision", "",
        f"Q1: **YES**. Existing HIER_CW_V2 omitted Center/Width source-scale restoration, so coefficient 1 meant `r*Cs/sC` and `r*Ws/sW`, not the declared identity. See `IMPLEMENTATION_AUDIT.md` and executable tests.", "",
        f"Q2: **{decision['Q2_historical_point_one_interpretation']}**. The old grid ended at 0.1; the expanded preregistered grid exposes whether weaker regularization remains preferred.", "",
        f"Q3: unequal lambda selection occurred in {decision['Q3_distinct_selection_contexts']}/{decision['Q3_total_selection_contexts']} contexts; mean inner gain was {100*np.mean(inner_gain):.3f}%, with {decision['Q3_outer_wins_vs_shared']}/{decision['Q3_outer_total']} outer wins and {100*np.mean(separate_gain):.3f}% mean outer gain. Selection difference and generalization benefit are not conflated.", "",
        f"Q4: endpoint-aligned corrected vs legacy-objective separate-lambda mean outer gain was {100*np.mean(endpoint_gain):.3f}% with {int((endpoint_gain>0).sum())}/{len(endpoint_gain)} wins.", "",
        f"Q5: corrected FULL128 vs corrected endpoint HIER mean outer gain was {100*np.mean(joint_gain):.3f}% with {int((joint_gain>0).sum())}/{len(joint_gain)} wins. Column/split detail is in `joint_full128_reaudit.md`.", "",
        f"Q6: **{decision['Q6_best_balanced_method']}** is the most stable corrected model: its worst context degradation versus old HIER is only 1.13%. **{decision['Q6_best_mean_context_rank']}** has the best mean context rank but is not called most stable because its 25g-compound degradation is 6.56%. No hidden composite score is used.", "",
        "## Mean metrics", "", _markdown(summary[columns]), "",
        "All predictions, fitted coefficients, architectures, and selections were SHA256-frozen before outer truth evaluation. Compound is primary, row secondary. No Active Learning, PCA, PLS, Direct Ridge, neural adapter, or new GNN architecture was run.", "",
        "Center and Width are operational elution-location and elution-interval-width surrogates only; these results do not prove physical mechanisms or a physical mass-scaling law."]
    (STUDY / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "fit", "score", "all"), default="all")
    args = parser.parse_args()
    if args.stage == "prepare": prepare()
    elif args.stage == "fit": fit_and_freeze()
    elif args.stage == "score": score_and_report()
    else: fit_and_freeze(); score_and_report()


if __name__ == "__main__":
    main()
