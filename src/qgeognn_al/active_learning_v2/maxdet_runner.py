"""Resumable execution for pure and fixed-U50 Gradient-MaxDet trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from ..artifacts import sha256_file
from ..models import load_predictor_checkpoint
from ..resources import SOURCE_DATA
from ..schemas.conditions import ConditionNormalization
from ..training.predictor import atomic_json
from .benchmark_protocol import initialization_seed, load_features, seed_config, sketch_seed
from .cache import array_hash
from .gradient_features import extract_q50_gradient_sketches
from .maxdet import conditional_gradient_maxdet, ensemble_top50_gate
from .maxdet_study import (
    BASELINE,
    CONFIRMATION_SEEDS,
    ENSEMBLE_K,
    METHODS,
    SKETCH_DIMENSION,
    STUDY,
    gate_size,
    historical_round0_paths,
    validate_pre_test_manifest,
)
from .protocol import RestrictedLabelStore, ids_hash, stable_hash, validate_row_protocol
from .runner import fit_from_same_initialization, predict_outputs
from .sequential_acquisition import validate_trajectory_transition
from .sequential_protocol import (
    ACQUISITION_ROUNDS,
    ACTIVE_LABEL_BUDGETS,
    BATCH_SIZE,
    FINAL_ACTIVE_LABELS,
    INITIAL_ACTIVE_LABELS,
)


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _write_json_once(path: Path, value: dict) -> None:
    path = Path(path)
    if path.exists():
        if _json(path) != value:
            raise RuntimeError(f"refusing to replace mismatched JSON: {path}")
        return
    atomic_json(path, value)


def _write_csv_once(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    if path.exists():
        current = pd.read_csv(path, keep_default_na=False)
        expected = frame.copy()
        expected.columns = expected.columns.astype(str)
        same = list(current.columns) == list(expected.columns) and len(current) == len(expected)
        if same:
            for column in current:
                if pd.api.types.is_numeric_dtype(current[column]) and pd.api.types.is_numeric_dtype(expected[column]):
                    same = bool(np.allclose(current[column].to_numpy(float), expected[column].to_numpy(float),
                                              rtol=0.0, atol=1e-12, equal_nan=True))
                else:
                    left = current[column].astype(object)
                    right = expected[column].astype(object)
                    same = all(
                        (pd.isna(a) and (pd.isna(b) or b == ""))
                        or (pd.isna(b) and (pd.isna(a) or a == ""))
                        or str(a) == str(b)
                        for a, b in zip(left.tolist(), right.tolist())
                    )
                if not same:
                    break
        if not same:
            raise RuntimeError(f"refusing to replace mismatched CSV: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _phase_label_budget(active: int) -> dict[str, int | float]:
    return {
        "active_label_count": int(active),
        "active_label_fraction_outer_train": int(active) / 3330,
        "shared_validation_label_count": 416,
        "total_observed_non_test_labels": int(active) + 416,
        "total_observed_fraction_full_dataset": (int(active) + 416) / 4163,
    }


@dataclass
class MaxDetSeedContext:
    seed: int
    partition: pd.DataFrame
    runtime: Path
    protocol_hash: str

    def __post_init__(self) -> None:
        torch.set_num_threads(2)
        self.seed = int(self.seed)
        self.runtime = Path(self.runtime)
        self.runtime.mkdir(parents=True, exist_ok=True)
        validate_row_protocol(self.partition)
        self.data = load_features()
        if self.data.sample_id.astype(str).tolist() != self.partition.sample_id.astype(str).tolist():
            raise RuntimeError("MaxDet split/source identity mismatch")
        self.roles = {
            role: self.partition.loc[self.partition.role.eq(role), "canonical_index"].to_numpy(int)
            for role in ("l0", "u0", "validation", "test")
        }
        baseline = BASELINE / "runtime" / f"seed_{self.seed}"
        historical = _json(baseline / "context.json")["contract"]
        self.preprocessing = historical["preprocessing"]
        self.normalization = ConditionNormalization(**historical["normalization"])
        self.atom, self.angle = torch.load(baseline / "scrubbed_graphs.pt", weights_only=False)
        if any(torch.count_nonzero(item.y) for item in self.atom):
            raise RuntimeError("historical scrubbed graph cache contains target labels")
        store = RestrictedLabelStore(SOURCE_DATA, self.partition)
        self.l0_truth = store.reveal(self.ids(self.roles["l0"]), "initial_fit")
        self.validation_truth = store.reveal(self.ids(self.roles["validation"]), "initial_fit")
        context = {
            "study": STUDY.name,
            "protocol_hash": self.protocol_hash,
            "outer_seed": self.seed,
            "split_hash": stable_hash(self.partition.to_dict("list")),
            "l0_ids_hash": ids_hash(self.ids(self.roles["l0"])),
            "u0_ids_hash": ids_hash(self.ids(self.roles["u0"])),
            "validation_ids_hash": ids_hash(self.ids(self.roles["validation"])),
            "test_ids_hash": ids_hash(self.ids(self.roles["test"])),
            "normalization": historical["normalization"],
            "preprocessing": self.preprocessing,
            "historical_context_hash": stable_hash(historical),
            "test_truth_access_count": 0,
        }
        _write_json_once(self.runtime / "context.json", context)
        self.context = context

    def ids(self, indices: Sequence[int]) -> list[str]:
        return self.data.iloc[list(indices)].sample_id.astype(str).tolist()

    def new_store(self) -> RestrictedLabelStore:
        store = RestrictedLabelStore(SOURCE_DATA, self.partition)
        l0 = store.reveal(self.ids(self.roles["l0"]), "initial_fit")
        validation = store.reveal(self.ids(self.roles["validation"]), "initial_fit")
        if not np.array_equal(l0, self.l0_truth) or not np.array_equal(validation, self.validation_truth):
            raise RuntimeError("MaxDet initial labels differ across method stores")
        return store

    def training_config(self, member: int) -> dict:
        config = seed_config(self.seed, member, smoke=False)
        config["split_sha256"] = stable_hash(self.partition.to_dict("list"))
        return config

    def fit(
        self,
        *,
        method: str,
        round_index: int,
        member: int,
        labeled_indices: np.ndarray,
        labeled_truth: np.ndarray,
        prediction_indices: np.ndarray,
        prediction_role: str,
        runtime: Path,
    ) -> dict[str, object]:
        train_ids = self.ids(labeled_indices)
        prediction_ids = self.ids(prediction_indices)
        contract = {
            "study": STUDY.name,
            "protocol_hash": self.protocol_hash,
            "outer_seed": self.seed,
            "method": method,
            "round": int(round_index),
            "member": int(member),
            "train_sample_ids": train_ids,
            "validation_sample_ids": self.ids(self.roles["validation"]),
            "L_t_ids_hash": ids_hash(train_ids),
            "prediction_role": prediction_role,
            "test_truth_access_count": 0,
        }
        audit = fit_from_same_initialization(
            atom_base=self.atom,
            angle=self.angle,
            normalization=self.normalization,
            preprocessing=self.preprocessing,
            train_indices=labeled_indices,
            train_truth=labeled_truth,
            validation_indices=self.roles["validation"],
            validation_truth=self.validation_truth,
            prediction_indices=prediction_indices,
            prediction_sample_ids=prediction_ids,
            initialization_seed=initialization_seed(self.seed, member),
            training_config=self.training_config(member),
            contract=contract,
            runtime=runtime,
        )
        return {
            "outer_seed": self.seed,
            "method": method,
            "round": int(round_index),
            "member": int(member),
            "reuse_status": "new_fit",
            **audit,
            **_phase_label_budget(len(labeled_indices)),
        }


def _partition(seed: int, study: Path) -> pd.DataFrame:
    path = Path(study) / "splits" / f"row_seed_{int(seed)}.csv"
    frame = pd.read_csv(path)
    validate_row_protocol(frame)
    return frame


def _round0_evaluation(context: MaxDetSeedContext, method: str) -> tuple[Path, dict[str, object]]:
    paths = historical_round0_paths(context.seed)
    audit = _json(paths["member0_audit"])
    prediction = pd.read_csv(paths["member0_predictions"])
    if prediction.sample_id.astype(str).tolist() != context.ids(context.roles["test"]):
        raise RuntimeError("historical round-zero test prediction identity drift")
    row = {
        "outer_seed": context.seed,
        "method": method,
        "round": 0,
        "member": 0,
        "reuse_status": "reused_historical_round0",
        "source_checkpoint": str(paths["member0_model"].relative_to(BASELINE.parent.parent.parent)),
        **audit,
        **_phase_label_budget(INITIAL_ACTIVE_LABELS),
    }
    return paths["member0_model"], row


def _gradient_features(
    context: MaxDetSeedContext,
    method: str,
    round_index: int,
    model_path: Path,
    current_indices: np.ndarray,
    directory: Path,
) -> tuple[np.ndarray, dict[str, object], str]:
    if round_index == 0:
        paths = historical_round0_paths(context.seed)
        with np.load(paths["gradient"]) as cached:
            features = cached["features"]
            order = cached["canonical_indices"]
        if not np.array_equal(order, current_indices):
            raise RuntimeError("historical round-zero gradient order drift")
        return features, _json(paths["gradient_audit"]), "reused_historical_round0"
    artifacts = directory / "acquisition"
    artifacts.mkdir(parents=True, exist_ok=True)
    path = artifacts / "current_gradient_features.npz"
    contract_path = artifacts / "current_gradient_features.contract.json"
    contract = {
        "study": STUDY.name,
        "outer_seed": context.seed,
        "method": method,
        "round": round_index,
        "checkpoint_sha256": sha256_file(model_path),
        "ordered_indices": current_indices.tolist(),
        "dimension": SKETCH_DIMENSION,
        "sketch_seed": sketch_seed(context.seed),
        "scales": context.preprocessing["target_scales"],
    }
    if path.exists() or contract_path.exists():
        if not path.exists() or not contract_path.exists() or _json(contract_path) != contract:
            raise RuntimeError(f"incomplete or incompatible gradient cache: {path}")
        with np.load(path) as cached:
            features = cached["features"]
            order = cached["canonical_indices"]
        if not np.array_equal(order, current_indices):
            raise RuntimeError("cached MaxDet gradient order drift")
        return features, _json(artifacts / "gradient_audit.json"), "resumed_cache"
    model = load_predictor_checkpoint(model_path)
    scales = tuple(float(context.preprocessing["target_scales"][key]) for key in ("V1", "V2"))
    result = extract_q50_gradient_sketches(
        model,
        context.atom,
        context.angle,
        current_indices,
        scales,
        dimension=SKETCH_DIMENSION,
        sketch_seed=sketch_seed(context.seed),
    )
    np.savez_compressed(path, features=result.features, canonical_indices=current_indices)
    atomic_json(contract_path, contract)
    atomic_json(artifacts / "gradient_audit.json", result.audit)
    return result.features, result.audit, "new_extraction"


def _pool_prediction(
    context: MaxDetSeedContext, model_path: Path, unlabeled_indices: np.ndarray
) -> tuple[np.ndarray, float]:
    model = load_predictor_checkpoint(model_path)
    started = time.perf_counter()
    values, order = predict_outputs(model, context.atom, context.angle, unlabeled_indices)
    elapsed = time.perf_counter() - started
    if not np.array_equal(order, unlabeled_indices):
        raise RuntimeError("MaxDet member-zero pool prediction order drift")
    return values, elapsed


def _ensemble_predictions(
    context: MaxDetSeedContext,
    round_index: int,
    labeled_indices: np.ndarray,
    labeled_truth: np.ndarray,
    unlabeled_indices: np.ndarray,
    member0: np.ndarray,
    directory: Path,
) -> tuple[np.ndarray, list[dict[str, object]], float]:
    predictions = [member0]
    fit_rows: list[dict[str, object]] = []
    measured_inference = 0.0
    u_ids = context.ids(unlabeled_indices)
    paths = historical_round0_paths(context.seed)
    for member in range(1, ENSEMBLE_K):
        if round_index == 0:
            table = pd.read_csv(paths[f"ensemble_member{member}_predictions"])
            if table.sample_id.astype(str).tolist() != u_ids:
                raise RuntimeError("historical round-zero ensemble prediction order drift")
            predictions.append(table.drop(columns="sample_id").to_numpy(float))
            audit = _json(paths[f"ensemble_member{member}_audit"])
            fit_rows.append({
                "outer_seed": context.seed,
                "method": "u50_gradient_maxdet",
                "round": 0,
                "member": member,
                "reuse_status": "reused_historical_hybrid_round0",
                **audit,
                **_phase_label_budget(len(labeled_indices)),
            })
            continue
        runtime = directory / "acquisition" / f"ensemble_member_{member}"
        fit = context.fit(
            method="u50_gradient_maxdet",
            round_index=round_index,
            member=member,
            labeled_indices=labeled_indices,
            labeled_truth=labeled_truth,
            prediction_indices=unlabeled_indices,
            prediction_role="current_U_t",
            runtime=runtime,
        )
        table = pd.read_csv(runtime / "predictions.csv.gz")
        if table.sample_id.astype(str).tolist() != u_ids:
            raise RuntimeError("new ensemble prediction identity drift")
        expected = table.drop(columns="sample_id").to_numpy(float)
        predictions.append(expected)
        # The historical fit helper reports training+frozen prediction together.
        # Repeat the exact inference once so ensemble inference has an independent clock.
        model = load_predictor_checkpoint(runtime / "best.pt")
        started = time.perf_counter()
        repeated, order = predict_outputs(model, context.atom, context.angle, unlabeled_indices)
        inference_seconds = time.perf_counter() - started
        measured_inference += inference_seconds
        if not np.array_equal(order, unlabeled_indices) or not np.allclose(repeated, expected, atol=1e-6, rtol=0):
            raise RuntimeError("timed ensemble inference differs from frozen prediction")
        fit_rows.append(fit)
    return np.stack(predictions), fit_rows, measured_inference


def _effective_rank_and_correlation(features: np.ndarray) -> tuple[float, float]:
    x = np.asarray(features, dtype=np.float64)
    gram = x @ x.T
    eigenvalues = np.maximum(np.linalg.eigvalsh((gram + gram.T) / 2), 0.0)
    total = float(eigenvalues.sum())
    if total <= 0:
        effective_rank = 0.0
    else:
        probabilities = eigenvalues[eigenvalues > 0] / total
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    norms = np.linalg.norm(x, axis=1)
    normalized = np.divide(x, norms[:, None], out=np.zeros_like(x), where=norms[:, None] > 0)
    kernel = normalized @ normalized.T
    off_diagonal = np.abs(kernel[~np.eye(len(x), dtype=bool)])
    return effective_rank, float(off_diagonal.mean()) if len(off_diagonal) else 0.0


def _historical_batch(seed: int, method: str, round_index: int) -> list[str]:
    path = BASELINE / "runtime" / f"seed_{seed}" / method / f"round_{round_index:02d}" / "acquisition_artifacts/selected_next_batch.csv"
    return [] if not path.exists() else pd.read_csv(path).sample_id.astype(str).tolist()


def _selection_diagnostics(
    *,
    context: MaxDetSeedContext,
    method: str,
    round_index: int,
    features: np.ndarray,
    labeled_count: int,
    selected_feature_positions: np.ndarray,
    selected_pool_positions: np.ndarray,
    unlabeled_indices: np.ndarray,
    member0_predictions: np.ndarray,
    selection,
    gradient_status: str,
    gradient_audit: dict[str, object],
    member0_inference_seconds: float,
    extra_member_inference_seconds: float,
    ensemble: np.ndarray | None,
    gate,
    directory: Path,
) -> dict[str, object]:
    selected_ids = np.asarray(context.ids(unlabeled_indices))[selected_pool_positions].astype(str)
    selected_features = features[selected_feature_positions]
    norms = np.linalg.norm(selected_features.astype(np.float64), axis=1)
    labeled_features = features[:labeled_count].astype(np.float64)
    nearest = np.sqrt(np.maximum(
        np.square(selected_features.astype(np.float64)).sum(axis=1, keepdims=True)
        + np.square(labeled_features).sum(axis=1)[None, :]
        - 2 * selected_features.astype(np.float64) @ labeled_features.T,
        0.0,
    ).min(axis=1))
    effective_rank, correlation = _effective_rank_and_correlation(selected_features)
    lcmd = set(_historical_batch(context.seed, "lcmd", round_index))
    hybrid = set(_historical_batch(context.seed, "hybrid", round_index))
    selected_set = set(selected_ids.tolist())
    rows = pd.DataFrame({
        "outer_seed": context.seed,
        "method": method,
        "source_round": round_index,
        "acquisition_round": round_index + 1,
        "selection_order": np.arange(BATCH_SIZE),
        "pool_position": selected_pool_positions,
        "canonical_index": unlabeled_indices[selected_pool_positions],
        "sample_id": selected_ids,
        "gradient_norm": norms,
        "nearest_L_t_gradient_distance": nearest,
        "predicted_V1_q50": member0_predictions[selected_pool_positions, 1],
        "predicted_V2_q50": member0_predictions[selected_pool_positions, 4],
        "marginal_logdet_gain": [item["marginal_logdet_gain"] for item in selection.trace],
    })
    if gate is not None:
        rank = np.empty(len(gate.ranking), dtype=int)
        rank[gate.ranking] = np.arange(len(gate.ranking))
        percentile = 1.0 - rank / max(1, len(rank) - 1)
        rows["ensemble_uncertainty"] = gate.uncertainty_scores[selected_pool_positions]
        rows["uncertainty_percentile"] = percentile[selected_pool_positions]
    _write_csv_once(directory / "acquisition" / "selected_next_batch.csv", rows)
    _write_csv_once(directory / "acquisition" / "maxdet_trace.csv", pd.DataFrame(selection.trace))
    uncertainty_summary: dict[str, object] = {}
    if gate is not None and ensemble is not None:
        gate_membership = np.zeros(len(unlabeled_indices), dtype=bool)
        gate_membership[gate.candidate_positions] = True
        selected_mask = np.zeros(len(unlabeled_indices), dtype=bool)
        selected_mask[selected_pool_positions] = True
        rank = np.empty(len(gate.ranking), dtype=int)
        rank[gate.ranking] = np.arange(len(gate.ranking))
        distribution = pd.DataFrame({
            "pool_position": np.arange(len(unlabeled_indices)),
            "canonical_index": unlabeled_indices,
            "sample_id": context.ids(unlabeled_indices),
            "uncertainty_score": gate.uncertainty_scores,
            "uncertainty_rank": rank + 1,
            "in_top50_gate": gate_membership,
            "selected": selected_mask,
        })
        _write_csv_once(directory / "acquisition" / "uncertainty_distribution.csv.gz", distribution)
        categories = {
            "selected": selected_mask,
            "gate_nonselected": gate_membership & ~selected_mask,
            "rejected_by_gate": ~gate_membership,
        }
        for name, mask in categories.items():
            values = gate.uncertainty_scores[mask]
            uncertainty_summary[f"{name}_uncertainty_count"] = int(mask.sum())
            uncertainty_summary[f"{name}_uncertainty_mean"] = float(values.mean())
            uncertainty_summary[f"{name}_uncertainty_median"] = float(np.median(values))
    profile = {
        "outer_seed": context.seed,
        "method": method,
        "source_round": round_index,
        "active_label_count": labeled_count,
        "pool_size": len(unlabeled_indices),
        "candidate_pool_size": len(unlabeled_indices) if gate is None else len(gate.candidate_positions),
        "top50_cutoff": None if gate is None else gate.cutoff_score,
        "normalization_scale": selection.normalization_scale,
        "selected_gradient_norm_mean": float(norms.mean()),
        "selected_gradient_norm_median": float(np.median(norms)),
        "nearest_L_t_gradient_distance_mean": float(nearest.mean()),
        "gram_effective_rank": effective_rank,
        "within_batch_mean_absolute_normalized_kernel_correlation": correlation,
        "overlap_lcmd_count": len(selected_set & lcmd),
        "overlap_lcmd_fraction": len(selected_set & lcmd) / BATCH_SIZE,
        "overlap_hybrid_count": len(selected_set & hybrid),
        "overlap_hybrid_fraction": len(selected_set & hybrid) / BATCH_SIZE,
        "gradient_status": gradient_status,
        "gradient_seconds": float(gradient_audit["elapsed_seconds"]),
        "selector_seconds": float(selection.audit["elapsed_seconds"]),
        "member0_pool_inference_seconds": member0_inference_seconds,
        "extra_member_measured_repeat_inference_seconds": extra_member_inference_seconds,
        "ensemble_inference_seconds_total": member0_inference_seconds + extra_member_inference_seconds,
        **uncertainty_summary,
    }
    atomic_json(directory / "acquisition" / "selection_profile.json", profile)
    return profile


def _acquire(
    context: MaxDetSeedContext,
    method: str,
    round_index: int,
    labeled_indices: np.ndarray,
    unlabeled_indices: np.ndarray,
    labeled_truth: np.ndarray,
    model_path: Path,
    directory: Path,
) -> tuple[list[str], dict[str, object], list[dict[str, object]]]:
    acquisition = directory / "acquisition"
    selection_path = acquisition / "selected_next_batch.csv"
    contract_path = acquisition / "contract.json"
    input_contract = {
        "study": STUDY.name,
        "outer_seed": context.seed,
        "method": method,
        "source_round": round_index,
        "L_t_ids_hash": ids_hash(context.ids(labeled_indices)),
        "U_t_ids_hash": ids_hash(context.ids(unlabeled_indices)),
        "checkpoint_sha256": sha256_file(model_path),
        "batch_size": BATCH_SIZE,
        "test_truth_access_count": 0,
    }
    if contract_path.exists() or selection_path.exists():
        if not contract_path.exists() or not selection_path.exists():
            raise RuntimeError(f"partial MaxDet acquisition cache: {directory}")
        saved = _json(contract_path)
        if saved["input"] != input_contract:
            raise RuntimeError(f"MaxDet acquisition input drift: {directory}")
        if sha256_file(selection_path) != saved["selected_table_sha256"]:
            raise RuntimeError(f"MaxDet acquisition selection hash drift: {directory}")
        ids = pd.read_csv(selection_path).sample_id.astype(str).tolist()
        extra = []
        if method == "u50_gradient_maxdet":
            for member in range(1, ENSEMBLE_K):
                if round_index == 0:
                    audit = _json(historical_round0_paths(context.seed)[f"ensemble_member{member}_audit"])
                    extra.append({"outer_seed": context.seed, "method": method, "round": 0, "member": member,
                                  "reuse_status": "reused_historical_hybrid_round0", **audit,
                                  **_phase_label_budget(len(labeled_indices))})
                else:
                    audit = _json(acquisition / f"ensemble_member_{member}/fit_audit.json")
                    extra.append({"outer_seed": context.seed, "method": method, "round": round_index,
                                  "member": member, "reuse_status": "new_fit", **audit,
                                  **_phase_label_budget(len(labeled_indices))})
        return ids, saved, extra

    acquisition.mkdir(parents=True, exist_ok=True)
    current = np.r_[labeled_indices, unlabeled_indices]
    features, gradient_audit, gradient_status = _gradient_features(
        context, method, round_index, model_path, current, directory
    )
    member0, member0_inference_seconds = _pool_prediction(context, model_path, unlabeled_indices)
    extra_fits: list[dict[str, object]] = []
    ensemble = None
    gate = None
    extra_member_inference_seconds = 0.0
    if method == "u50_gradient_maxdet":
        ensemble, extra_fits, extra_member_inference_seconds = _ensemble_predictions(
            context, round_index, labeled_indices, labeled_truth, unlabeled_indices, member0, directory
        )
        gate = ensemble_top50_gate(
            ensemble,
            tuple(float(context.preprocessing["target_scales"][key]) for key in ("V1", "V2")),
        )
        if len(gate.candidate_positions) != gate_size(len(unlabeled_indices)):
            raise RuntimeError("fixed Top50 gate size drift")
        # Gate membership is uncertainty-ranked, but MaxDet ties use the same
        # canonical current-U order as the pure arm.
        candidate_feature_positions = len(labeled_indices) + np.sort(gate.candidate_positions)
    else:
        candidate_feature_positions = np.arange(len(labeled_indices), len(current), dtype=int)
    selection = conditional_gradient_maxdet(
        features,
        np.arange(len(labeled_indices), dtype=int),
        candidate_feature_positions,
        BATCH_SIZE,
    )
    selected_pool_positions = selection.selected_positions - len(labeled_indices)
    if np.any(selected_pool_positions < 0) or np.any(selected_pool_positions >= len(unlabeled_indices)):
        raise RuntimeError("MaxDet selection escaped current U_t")
    profile = _selection_diagnostics(
        context=context,
        method=method,
        round_index=round_index,
        features=features,
        labeled_count=len(labeled_indices),
        selected_feature_positions=selection.selected_positions,
        selected_pool_positions=selected_pool_positions,
        unlabeled_indices=unlabeled_indices,
        member0_predictions=member0,
        selection=selection,
        gradient_status=gradient_status,
        gradient_audit=gradient_audit,
        member0_inference_seconds=member0_inference_seconds,
        extra_member_inference_seconds=extra_member_inference_seconds,
        ensemble=ensemble,
        gate=gate,
        directory=directory,
    )
    selected_ids = np.asarray(context.ids(unlabeled_indices))[selected_pool_positions].astype(str).tolist()
    if len(selected_ids) != BATCH_SIZE or len(set(selected_ids)) != BATCH_SIZE:
        raise RuntimeError("MaxDet did not return exactly 32 unique current-U IDs")
    contract = {
        "input": input_contract,
        "status": "FROZEN_BEFORE_TEST_TRUTH",
        "selected_ids_ordered_hash": stable_hash(selected_ids),
        "selected_ids_set_hash": ids_hash(selected_ids),
        "selected_table_sha256": sha256_file(selection_path),
        "selection_profile": profile,
        "test_truth_access_count": 0,
    }
    atomic_json(contract_path, contract)
    return selected_ids, contract, extra_fits


def run_method(context: MaxDetSeedContext, method: str) -> dict[str, object]:
    if method not in METHODS:
        raise ValueError(f"unknown MaxDet method: {method}")
    store = context.new_store()
    labeled = context.roles["l0"].copy()
    unlabeled = context.roles["u0"].copy()
    truth = context.l0_truth.copy()
    incoming: list[str] = []
    selected_all: list[str] = []
    fits: list[dict[str, object]] = []
    round_hashes: list[str] = []
    for round_index in range(ACQUISITION_ROUNDS + 1):
        directory = context.runtime / method / f"round_{round_index:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        input_contract = {
            "study": STUDY.name,
            "protocol_hash": context.protocol_hash,
            "outer_seed": context.seed,
            "method": method,
            "round": round_index,
            "active_label_count": len(labeled),
            "L_t_ids_hash": ids_hash(context.ids(labeled)),
            "U_t_ids_hash": ids_hash(context.ids(unlabeled)),
            "incoming_selected_ids_hash": stable_hash(incoming),
            "test_truth_access_count": 0,
        }
        _write_json_once(directory / "input_contract.json", input_contract)
        _write_csv_once(directory / "state.csv", pd.DataFrame({
            "role": ["labeled"] * len(labeled) + ["unlabeled"] * len(unlabeled),
            "position": list(range(len(labeled))) + list(range(len(unlabeled))),
            "canonical_index": np.r_[labeled, unlabeled],
            "sample_id": context.ids(np.r_[labeled, unlabeled]),
        }))
        if round_index == 0:
            model_path, evaluation = _round0_evaluation(context, method)
            prediction_path = historical_round0_paths(context.seed)["member0_predictions"]
        else:
            model_runtime = directory / "model"
            evaluation = context.fit(
                method=method,
                round_index=round_index,
                member=0,
                labeled_indices=labeled,
                labeled_truth=truth,
                prediction_indices=context.roles["test"],
                prediction_role="test_X_without_test_truth",
                runtime=model_runtime,
            )
            model_path = model_runtime / "best.pt"
            prediction_path = model_runtime / "predictions.csv.gz"
        fits.append(evaluation)
        outgoing: list[str] = []
        acquisition_contract = None
        if round_index < ACQUISITION_ROUNDS:
            outgoing, acquisition_contract, extra = _acquire(
                context, method, round_index, labeled, unlabeled, truth, model_path, directory
            )
            fits.extend(extra)
        round_contract = {
            "input": input_contract,
            "status": "FROZEN_BEFORE_TEST_TRUTH",
            "checkpoint_path": str(model_path),
            "checkpoint_sha256": sha256_file(model_path),
            "prediction_path": str(prediction_path),
            "prediction_sha256": sha256_file(prediction_path),
            "outgoing_selected_ids_hash": stable_hash(outgoing),
            "acquisition_contract_hash": None if acquisition_contract is None else stable_hash(acquisition_contract),
            "test_truth_access_count": 0,
        }
        _write_json_once(directory / "round_freeze.json", round_contract)
        round_hashes.append(sha256_file(directory / "round_freeze.json"))
        if round_index == ACQUISITION_ROUNDS:
            break
        previous_l_ids, previous_u_ids = context.ids(labeled), context.ids(unlabeled)
        by_id = {sample_id: position for position, sample_id in enumerate(previous_u_ids)}
        selected_positions = np.asarray([by_id[sample_id] for sample_id in outgoing], dtype=int)
        selected_indices = unlabeled[selected_positions]
        selected_set = set(outgoing)
        next_unlabeled = np.asarray([
            index for index, sample_id in zip(unlabeled, previous_u_ids) if sample_id not in selected_set
        ], dtype=int)
        next_labeled = np.r_[labeled, selected_indices]
        validate_trajectory_transition(
            previous_l_ids, previous_u_ids, outgoing, context.ids(next_labeled), context.ids(next_unlabeled)
        )
        selected_all.extend(outgoing)
        if len(selected_all) != len(set(selected_all)):
            raise RuntimeError("MaxDet trajectory reacquired an earlier sample")
        store.freeze_acquisitions(selected_all)
        new_truth = store.reveal(outgoing, "after_acquisition_fit")
        truth = np.vstack([truth, new_truth])
        labeled, unlabeled, incoming = next_labeled, next_unlabeled, outgoing
    if len(labeled) != FINAL_ACTIVE_LABELS or len(selected_all) != ACQUISITION_ROUNDS * BATCH_SIZE:
        raise RuntimeError("MaxDet trajectory did not reach the registered final budget")
    fit_frame = pd.DataFrame(fits)
    _write_csv_once(context.runtime / method / "fit_audit.csv", fit_frame)
    _write_csv_once(context.runtime / method / "label_access_audit.csv", pd.DataFrame(store.audit))
    trajectory = {
        "status": "FROZEN_BEFORE_TEST_TRUTH",
        "outer_seed": context.seed,
        "method": method,
        "round_points": ACQUISITION_ROUNDS + 1,
        "acquisition_rounds": ACQUISITION_ROUNDS,
        "final_active_labels": len(labeled),
        "selected_ids_ordered_hash": stable_hash(selected_all),
        "final_L_t_ids_hash": ids_hash(context.ids(labeled)),
        "final_U_t_ids_hash": ids_hash(context.ids(unlabeled)),
        "round_freeze_sha256": round_hashes,
        "fit_audit_sha256": sha256_file(context.runtime / method / "fit_audit.csv"),
        "label_access_audit_sha256": sha256_file(context.runtime / method / "label_access_audit.csv"),
        "test_truth_access_count": 0,
    }
    _write_json_once(context.runtime / method / "trajectory_freeze.json", trajectory)
    return trajectory


def _write_pairwise_selection_overlap(context: MaxDetSeedContext) -> None:
    rows = []
    for round_index in range(ACQUISITION_ROUNDS):
        batches = {}
        for method in METHODS:
            path = context.runtime / method / f"round_{round_index:02d}/acquisition/selected_next_batch.csv"
            batches[method] = pd.read_csv(path).sample_id.astype(str).tolist()
        left, right = map(set, batches.values())
        rows.append({
            "outer_seed": context.seed,
            "source_round": round_index,
            "active_label_count": ACTIVE_LABEL_BUDGETS[round_index],
            "intersection_count": len(left & right),
            "overlap_fraction": len(left & right) / BATCH_SIZE,
            "same_ordered_batch": batches[METHODS[0]] == batches[METHODS[1]],
        })
    _write_csv_once(context.runtime / "pure_u50_selection_overlap.csv", pd.DataFrame(rows))


def execute_seed(seed: int, study: Path = STUDY) -> dict[str, object]:
    validation = validate_pre_test_manifest(study)
    if int(seed) not in CONFIRMATION_SEEDS:
        raise ValueError("seed is outside the frozen confirmation cohort")
    placeholder = _json(Path(study) / "global_pre_test_freeze.json")
    if placeholder.get("status") != "PENDING_PRE_TEST_EXECUTION":
        raise RuntimeError("MaxDet seed execution is closed after the global barrier")
    context = MaxDetSeedContext(
        int(seed),
        _partition(int(seed), Path(study)),
        Path(study) / "runtime" / f"seed_{int(seed)}",
        validation["runtime_protocol_hash"],
    )
    trajectories = {}
    for method in METHODS:
        trajectories[method] = run_method(context, method)
        print(json.dumps({"seed": int(seed), "method_frozen": method, "test_truth_access_count": 0}), flush=True)
    _write_pairwise_selection_overlap(context)
    return {"seed": int(seed), "status": "SEED_FROZEN_BEFORE_TEST_TRUTH", "trajectories": trajectories}


def build_global_pre_test_freeze(study: Path = STUDY) -> dict[str, object]:
    validate_pre_test_manifest(study)
    study = Path(study)
    entries = {}
    for seed in CONFIRMATION_SEEDS:
        for method in METHODS:
            root = study / "runtime" / f"seed_{seed}" / method
            trajectory_path = root / "trajectory_freeze.json"
            trajectory = _json(trajectory_path)
            if trajectory.get("status") != "FROZEN_BEFORE_TEST_TRUTH" or trajectory.get("test_truth_access_count") != 0:
                raise RuntimeError(f"trajectory is not frozen: seed={seed} method={method}")
            if sha256_file(root / "fit_audit.csv") != trajectory["fit_audit_sha256"]:
                raise RuntimeError("MaxDet fit audit changed after trajectory freeze")
            for round_index in range(ACQUISITION_ROUNDS + 1):
                path = root / f"round_{round_index:02d}/round_freeze.json"
                record = _json(path)
                if record["status"] != "FROZEN_BEFORE_TEST_TRUTH" or record["test_truth_access_count"] != 0:
                    raise RuntimeError("MaxDet round escaped the pre-test barrier")
                if sha256_file(Path(record["checkpoint_path"])) != record["checkpoint_sha256"]:
                    raise RuntimeError("MaxDet frozen checkpoint changed")
                if sha256_file(Path(record["prediction_path"])) != record["prediction_sha256"]:
                    raise RuntimeError("MaxDet frozen prediction changed")
                entries[f"seed_{seed}/{method}/round_{round_index:02d}"] = {
                    "round_freeze_sha256": sha256_file(path),
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "prediction_sha256": record["prediction_sha256"],
                    "active_label_count": record["input"]["active_label_count"],
                }
    freeze = {
        "status": "FROZEN_BEFORE_TEST_TRUTH",
        "description": f"{len(CONFIRMATION_SEEDS)} executed seeds x 2 MaxDet methods x 22 prediction points; all {len(CONFIRMATION_SEEDS) * 21} batches frozen",
        "entries": entries,
        "frozen_prediction_points": len(entries),
        "test_truth_access_count": 0,
    }
    atomic_json(study / "global_pre_test_freeze.json", freeze)
    return freeze
