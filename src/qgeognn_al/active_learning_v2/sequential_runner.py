"""Resumable formal orchestration for sequential B=32 active learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from ..artifacts import sha256_file
from ..models import load_predictor_checkpoint
from ..resources import SOURCE_DATA, SOURCE_GRAPH_CACHE
from ..training.predictor import atomic_json
from .benchmark_protocol import initialization_seed, load_features, seed_config, sketch_seed
from .benchmark_reporting import metric_row
from .cache import array_hash, seal_cache, verify_cache
from .coverage import extract_representations
from .gradient_features import extract_q50_gradient_sketches
from .protocol import RestrictedLabelStore, ids_hash, stable_hash, validate_row_protocol
from .runner import (
    fit_al_preprocessing,
    fit_from_same_initialization,
    make_label_scrubbed_graphs,
    predict_outputs,
)
from .sequential_acquisition import (
    acquire_sequential_batch,
    random_round_batch,
    validate_trajectory_transition,
)
from .sequential_protocol import (
    ACQUISITION_ROUNDS,
    ACTIVE_LABEL_BUDGETS,
    BATCH_SIZE,
    CONFIRMATION_SEEDS,
    ENSEMBLE_K,
    FINAL_ACTIVE_LABELS,
    INITIAL_ACTIVE_LABELS,
    INITIAL_POOL_ROWS,
    METHODS,
    OUTER_TRAIN_ROWS,
    SKETCH_DIMENSION,
    STUDY,
    VALIDATION_ROWS,
    assert_formal_authorized,
    code_hashes,
    label_budget,
    protocol_record,
    random_trajectory_seed,
)
from .sequential_reporting import write_final_outputs


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _write_json_once(path: Path, value: dict) -> None:
    path = Path(path)
    if path.exists():
        if _json(path) != value:
            raise RuntimeError(f"refusing to reuse mismatched JSON artifact: {path}")
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
            for column in current.columns:
                if pd.api.types.is_numeric_dtype(current[column]) and pd.api.types.is_numeric_dtype(expected[column]):
                    same = bool(np.allclose(current[column].to_numpy(float), expected[column].to_numpy(float),
                                              rtol=0.0, atol=1e-12, equal_nan=True))
                else:
                    same = current[column].astype(str).tolist() == expected[column].astype(str).tolist()
                if not same:
                    break
        if not same:
            raise RuntimeError(f"refusing to reuse mismatched CSV artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_npy_once(path: Path, values: Sequence[object], contract: dict) -> None:
    path = Path(path)
    array = np.asarray(values)
    if verify_cache(path, contract):
        current = np.load(path)
        if not np.array_equal(current, array):
            raise RuntimeError(f"cached state array differs despite contract: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    seal_cache(path, contract)


def _assert_protected(root: Path, files: dict[str, str]) -> None:
    for relative, digest in files.items():
        path = root / relative
        if not path.exists() or sha256_file(path) != digest:
            raise RuntimeError(f"frozen artifact changed: {path}")


def _assert_complete_fit_or_empty(runtime: Path) -> None:
    paths = (runtime / "best.pt", runtime / "predictions.csv.gz", runtime / "fit_audit.json")
    present = [path.exists() for path in paths]
    if any(present) and not all(present):
        raise RuntimeError(f"partial fit cache cannot be resumed safely: {runtime}")


@dataclass
class SequentialSeedContext:
    """Immutable X/split/preprocessing state shared by all arms for one seed."""

    seed: int
    partition: pd.DataFrame
    runtime: Path
    commit: str
    source: Path = SOURCE_DATA

    def __post_init__(self) -> None:
        torch.set_num_threads(2)
        self.runtime = Path(self.runtime)
        self.runtime.mkdir(parents=True, exist_ok=True)
        validate_row_protocol(self.partition)
        if set(self.partition.outer_seed) != {int(self.seed)}:
            raise ValueError("partition seed mismatch")
        self.data = load_features(self.source)
        if self.data.sample_id.astype(str).tolist() != self.partition.sample_id.astype(str).tolist():
            raise ValueError("partition/source sample identity drift")
        if self.data.canonical_smiles.astype(str).tolist() != self.partition.canonical_smiles.astype(str).tolist():
            raise ValueError("partition/source molecule identity drift")
        self.roles = {
            role: self.partition.loc[self.partition.role.eq(role), "canonical_index"].to_numpy(int)
            for role in ("l0", "u0", "validation", "test")
        }
        if (len(self.roles["l0"]), len(self.roles["u0"]), len(self.roles["validation"]), len(self.roles["test"])) != (
            INITIAL_ACTIVE_LABELS, INITIAL_POOL_ROWS, VALIDATION_ROWS, 417
        ):
            raise ValueError("formal row protocol counts changed")
        self.outer = np.r_[self.roles["l0"], self.roles["u0"]]
        initial_store = RestrictedLabelStore(self.source, self.partition)
        self.l0_truth = initial_store.reveal(self.ids(self.roles["l0"]), "initial_fit")
        self.validation_truth = initial_store.reveal(self.ids(self.roles["validation"]), "initial_fit")
        raw_graphs = torch.load(SOURCE_GRAPH_CACHE, weights_only=False)
        self.normalization, self.preprocessing = fit_al_preprocessing(
            self.data,
            raw_graphs,
            self.outer,
            self.ids(self.roles["l0"]),
            self.l0_truth,
            self.runtime / "scaler.json",
        )
        self.context_contract = {
            "study": STUDY.name,
            "preregistration_commit": self.commit,
            "outer_seed": int(self.seed),
            "source_sha256": sha256_file(self.source),
            "source_graph_cache_sha256": sha256_file(SOURCE_GRAPH_CACHE),
            "partition_hash": stable_hash(self.partition.to_dict("list")),
            "l0_ids_hash": ids_hash(self.ids(self.roles["l0"])),
            "u0_ids_hash": ids_hash(self.ids(self.roles["u0"])),
            "validation_ids_hash": ids_hash(self.ids(self.roles["validation"])),
            "test_ids_hash": ids_hash(self.ids(self.roles["test"])),
            "normalization": asdict(self.normalization),
            "preprocessing": self.preprocessing,
            "code_hashes": code_hashes(),
        }
        graph_path = self.runtime / "scrubbed_graphs.pt"
        if verify_cache(graph_path, self.context_contract):
            self.atom, self.angle = torch.load(graph_path, weights_only=False)
        else:
            self.atom, self.angle = make_label_scrubbed_graphs(
                self.data, raw_graphs, self.preprocessing["scaler"]
            )
            torch.save((self.atom, self.angle), graph_path)
            seal_cache(graph_path, self.context_contract)
        if any(torch.count_nonzero(item.y) for item in self.atom):
            raise RuntimeError("sequential graph cache contains non-sentinel labels")
        _write_json_once(self.runtime / "context.json", {
            "contract": self.context_contract,
            "contract_hash": stable_hash(self.context_contract),
            "test_truth_access_count": 0,
        })

    def ids(self, indices: Sequence[int]) -> list[str]:
        return self.data.iloc[list(indices)].sample_id.astype(str).tolist()

    def new_method_store(self) -> RestrictedLabelStore:
        store = RestrictedLabelStore(self.source, self.partition)
        l0 = store.reveal(self.ids(self.roles["l0"]), "initial_fit")
        validation = store.reveal(self.ids(self.roles["validation"]), "initial_fit")
        if not np.array_equal(l0, self.l0_truth) or not np.array_equal(validation, self.validation_truth):
            raise RuntimeError("method label store does not match frozen initial truth")
        return store

    def training_config(self, member: int = 0) -> dict:
        config = seed_config(self.seed, member, smoke=False)
        config["split_sha256"] = stable_hash(self.partition.to_dict("list"))
        return config

    def fit(
        self,
        *,
        method: str,
        round_index: int,
        labeled_indices: Sequence[int],
        labeled_truth: np.ndarray,
        runtime: Path,
        member: int = 0,
        prediction_indices: Sequence[int] | None = None,
    ) -> dict[str, object]:
        _assert_complete_fit_or_empty(runtime)
        predict_indices = self.roles["test"] if prediction_indices is None else np.asarray(prediction_indices, dtype=int)
        predict_ids = self.ids(predict_indices)
        labeled_ids = self.ids(labeled_indices)
        contract = {
            "study": STUDY.name,
            "context_hash": stable_hash(self.context_contract),
            "method": method,
            "round": int(round_index),
            "member": int(member),
            "train_sample_ids": labeled_ids,
            "validation_sample_ids": self.ids(self.roles["validation"]),
            "L_t_ids_hash": ids_hash(labeled_ids),
            "prediction_role": "current_U_t" if prediction_indices is not None else "test",
        }
        audit = fit_from_same_initialization(
            atom_base=self.atom,
            angle=self.angle,
            normalization=self.normalization,
            preprocessing=self.preprocessing,
            train_indices=np.asarray(labeled_indices, dtype=int),
            train_truth=np.asarray(labeled_truth, dtype=np.float32),
            validation_indices=self.roles["validation"],
            validation_truth=self.validation_truth,
            prediction_indices=predict_indices,
            prediction_sample_ids=predict_ids,
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
            **audit,
            **label_budget(len(labeled_indices)),
        }


def _round_input_contract(
    context: SequentialSeedContext,
    method: str,
    round_index: int,
    labeled_indices: Sequence[int],
    unlabeled_indices: Sequence[int],
    incoming_selected_ids: Sequence[str],
) -> dict[str, object]:
    labeled_ids = context.ids(labeled_indices)
    unlabeled_ids = context.ids(unlabeled_indices)
    return {
        "study": STUDY.name,
        "preregistration_commit": context.commit,
        "outer_seed": context.seed,
        "method": method,
        "round": int(round_index),
        "active_label_count": len(labeled_ids),
        "unlabeled_count": len(unlabeled_ids),
        "L_t_ids_hash": ids_hash(labeled_ids),
        "U_t_ids_hash": ids_hash(unlabeled_ids),
        "ordered_L_t_hash": stable_hash(labeled_ids),
        "ordered_U_t_hash": stable_hash(unlabeled_ids),
        "incoming_selected_ids_hash": stable_hash(list(incoming_selected_ids)),
        "model_initialization_seed": initialization_seed(context.seed, 0),
        "training_config": context.training_config(0),
        "endpoint_scales": context.preprocessing["target_scales"],
        "endpoint_scales_fit_ids_hash": context.preprocessing["target_scale_fit_ids_hash"],
        "acquisition_config": {
            "batch_size": BATCH_SIZE,
            "hybrid": "current_K3_q50_top25_then_current_latent_farthest_first",
            "lcmd": "current_full_network_q50_512D_CountSketch_corrected_TP_squared_euclidean",
            "random_seed": random_trajectory_seed(context.seed),
        },
        "code_hash": stable_hash(code_hashes()),
        "test_truth_access_count": 0,
    }


def _save_round_state(
    directory: Path,
    contract: dict,
    labeled_indices: Sequence[int],
    unlabeled_indices: Sequence[int],
    labeled_ids: Sequence[str],
    unlabeled_ids: Sequence[str],
    incoming_selected: pd.DataFrame,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_json_once(directory / "input_contract.json", contract)
    state_contract = {"round_contract_hash": stable_hash(contract), "kind": "ordered_L_t_sample_ids"}
    _write_npy_once(directory / "labeled_ids.npy", labeled_ids, state_contract)
    state_contract = {"round_contract_hash": stable_hash(contract), "kind": "ordered_U_t_sample_ids"}
    _write_npy_once(directory / "unlabeled_ids.npy", unlabeled_ids, state_contract)
    state_contract = {"round_contract_hash": stable_hash(contract), "kind": "L_t_canonical_indices"}
    _write_npy_once(directory / "labeled_indices.npy", labeled_indices, state_contract)
    state_contract = {"round_contract_hash": stable_hash(contract), "kind": "U_t_canonical_indices"}
    _write_npy_once(directory / "unlabeled_indices.npy", unlabeled_indices, state_contract)
    _write_csv_once(directory / "selected_batch.csv", incoming_selected)


def _load_completed_acquisition(directory: Path, expected: dict) -> tuple[list[str], dict] | None:
    contract_path = directory / "acquisition_artifacts/contract.json"
    selection_path = directory / "acquisition_artifacts/selected_next_batch.csv"
    present = (contract_path.exists(), selection_path.exists())
    if any(present) and not all(present):
        raise RuntimeError(f"partial acquisition cache cannot be resumed safely: {directory}")
    if not all(present):
        return None
    saved = _json(contract_path)
    if saved.get("input") != expected:
        raise RuntimeError(f"acquisition contract mismatch: {directory}")
    _assert_protected(directory, saved["files"])
    selected = pd.read_csv(selection_path).sample_id.astype(str).tolist()
    if stable_hash(selected) != saved["selected_ids_ordered_hash"]:
        raise RuntimeError("cached acquisition selection hash mismatch")
    return selected, saved


def _cached_hybrid_fit_rows(
    context: SequentialSeedContext,
    round_index: int,
    directory: Path,
    active_labels: int,
) -> list[dict[str, object]]:
    """Restore acquisition-only fit audit rows when a Hybrid round resumes."""

    rows: list[dict[str, object]] = []
    for member in range(1, ENSEMBLE_K):
        audit = _json(directory / "acquisition_artifacts" / f"ensemble_member_{member}" / "fit_audit.json")
        rows.append({
            "outer_seed": context.seed,
            "method": "hybrid",
            "round": int(round_index),
            "member": member,
            **audit,
            **label_budget(active_labels),
        })
    return rows


def _acquire_next(
    context: SequentialSeedContext,
    method: str,
    round_index: int,
    labeled_indices: np.ndarray,
    unlabeled_indices: np.ndarray,
    labeled_truth: np.ndarray,
    evaluation_fit: dict[str, object],
    directory: Path,
) -> tuple[list[str], dict[str, object], list[dict[str, object]]]:
    acquisition_round = round_index + 1
    labeled_ids = context.ids(labeled_indices)
    unlabeled_ids = context.ids(unlabeled_indices)
    input_contract = {
        "outer_seed": context.seed,
        "method": method,
        "source_round": round_index,
        "acquisition_round": acquisition_round,
        "L_t_ids_hash": ids_hash(labeled_ids),
        "U_t_ids_hash": ids_hash(unlabeled_ids),
        "evaluation_checkpoint_sha256": evaluation_fit["checkpoint_sha256"],
        "evaluation_checkpoint_state_hash": evaluation_fit["checkpoint_state_hash"],
        "endpoint_scales": context.preprocessing["target_scales"],
        "batch_size": BATCH_SIZE,
        "feature_recomputed_from_current_model": method in {"hybrid", "lcmd"},
        "centers_are_current_L_t": method in {"hybrid", "lcmd"},
        "test_truth_access_count": 0,
    }
    if cached := _load_completed_acquisition(directory, input_contract):
        cached_fits = (
            _cached_hybrid_fit_rows(context, round_index, directory, len(labeled_indices))
            if method == "hybrid" else []
        )
        return cached[0], cached[1], cached_fits

    artifacts = directory / "acquisition_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    selected_positions: np.ndarray
    extra_fits: list[dict[str, object]] = []
    protected: list[Path] = []
    metadata: dict[str, object]
    if method == "random":
        selected_ids = list(random_round_batch(
            context.ids(context.roles["u0"]), unlabeled_ids, context.seed, acquisition_round
        ))
        position_by_id = {sample_id: position for position, sample_id in enumerate(unlabeled_ids)}
        selected_positions = np.asarray([position_by_id[value] for value in selected_ids], dtype=int)
        metadata = {
            "definition": "next_slice_of_single_original_U0_permutation",
            "random_trajectory_seed": random_trajectory_seed(context.seed),
        }
    else:
        model = load_predictor_checkpoint(directory / "model/best.pt")
        current = np.r_[labeled_indices, unlabeled_indices]
        scales = tuple(float(context.preprocessing["target_scales"][target]) for target in ("V1", "V2"))
        if method == "hybrid":
            representation_contract = {
                **input_contract,
                "kind": "current_QGeoGNN_V2_128D_representation",
                "ordered_indices": current.tolist(),
            }
            representation_path = artifacts / "current_representations.npz"
            if verify_cache(representation_path, representation_contract):
                with np.load(representation_path) as cached_representation:
                    representations = cached_representation["features"]
                    stored_indices = cached_representation["canonical_indices"]
                if not np.array_equal(stored_indices, current):
                    raise RuntimeError("cached current representation order mismatch")
            else:
                representations = extract_representations(model, context.atom, context.angle, current)
                np.savez_compressed(representation_path, features=representations, canonical_indices=current)
                seal_cache(representation_path, representation_contract)
            predictions: list[np.ndarray] = []
            member0, order = predict_outputs(model, context.atom, context.angle, unlabeled_indices)
            if not np.array_equal(order, unlabeled_indices):
                raise RuntimeError("Hybrid member-0 current-U_t order mismatch")
            predictions.append(member0)
            for member in range(1, ENSEMBLE_K):
                member_runtime = artifacts / f"ensemble_member_{member}"
                fit = context.fit(
                    method=method,
                    round_index=round_index,
                    labeled_indices=labeled_indices,
                    labeled_truth=labeled_truth,
                    runtime=member_runtime,
                    member=member,
                    prediction_indices=unlabeled_indices,
                )
                extra_fits.append(fit)
                table = pd.read_csv(member_runtime / "predictions.csv.gz")
                if table.sample_id.astype(str).tolist() != unlabeled_ids:
                    raise RuntimeError("Hybrid ensemble current-U_t prediction identity mismatch")
                predictions.append(table.drop(columns="sample_id").to_numpy(float))
                protected.extend(member_runtime / name for name in ("best.pt", "predictions.csv.gz", "fit_audit.json"))
            ensemble = np.stack(predictions)
            selection = acquire_sequential_batch(
                method="hybrid",
                labeled_count=len(labeled_indices),
                representations=representations,
                ensemble_predictions=ensemble,
                scales=scales,
            )
            selected_positions = np.asarray(selection["selected_pool_positions"], dtype=int)
            shortlist = np.asarray(selection["shortlist_pool_positions"], dtype=int)
            shortlist_frame = pd.DataFrame({
                "rank": np.arange(1, len(shortlist) + 1),
                "pool_position": shortlist,
                "canonical_index": unlabeled_indices[shortlist],
                "sample_id": np.asarray(unlabeled_ids)[shortlist],
                "uncertainty_score": np.asarray(selection["uncertainty_scores"])[shortlist],
            })
            _write_csv_once(artifacts / "shortlist.csv", shortlist_frame)
            protected.extend((representation_path, representation_path.with_suffix(".npz.contract.json"),
                              artifacts / "shortlist.csv"))
            metadata = {
                "definition": "current_K3_q50_top25_then_current_V2_latent_farthest_first",
                "shortlist_size": int(selection["shortlist_size"]),
                "center_count": int(selection["center_count"]),
                "uncertainty_recomputed_round": round_index,
                "representation_recomputed_round": round_index,
            }
        else:
            gradient_contract = {
                **input_contract,
                "kind": "current_full_network_q50_gradient_CountSketch",
                "ordered_indices": current.tolist(),
                "dimension": SKETCH_DIMENSION,
                "sketch_seed": sketch_seed(context.seed),
                "scales": scales,
            }
            gradient_path = artifacts / "current_gradient_features.npz"
            if verify_cache(gradient_path, gradient_contract):
                with np.load(gradient_path) as cached_gradient:
                    gradients = cached_gradient["features"]
                    stored_indices = cached_gradient["canonical_indices"]
                if not np.array_equal(stored_indices, current):
                    raise RuntimeError("cached current gradient order mismatch")
                gradient_audit = _json(artifacts / "gradient_audit.json")
            else:
                result = extract_q50_gradient_sketches(
                    model,
                    context.atom,
                    context.angle,
                    current,
                    scales,
                    dimension=SKETCH_DIMENSION,
                    sketch_seed=sketch_seed(context.seed),
                )
                gradients, gradient_audit = result.features, result.audit
                np.savez_compressed(gradient_path, features=gradients, canonical_indices=current)
                seal_cache(gradient_path, gradient_contract)
                atomic_json(artifacts / "gradient_audit.json", gradient_audit)
            selection = acquire_sequential_batch(
                method="lcmd", labeled_count=len(labeled_indices), gradient_features=gradients
            )
            selected_positions = np.asarray(selection["selected_pool_positions"], dtype=int)
            trace = pd.DataFrame(selection["trace"])
            _write_csv_once(artifacts / "lcmd_trace.csv", trace)
            protected.extend((gradient_path, gradient_path.with_suffix(".npz.contract.json"),
                              artifacts / "gradient_audit.json", artifacts / "lcmd_trace.csv"))
            metadata = {
                "definition": "current_full_network_q50_512D_CountSketch_corrected_LCMD_TP",
                "center_count": int(selection["center_count"]),
                "gradient_features_recomputed_round": round_index,
                "gradient_semantic_hash": array_hash(gradients),
                "gradient_audit": gradient_audit,
            }
        selected_ids = np.asarray(unlabeled_ids)[selected_positions].astype(str).tolist()

    if len(selected_ids) != BATCH_SIZE or len(set(selected_ids)) != BATCH_SIZE:
        raise RuntimeError("sequential acquisition did not produce exactly 32 unique IDs")
    selection_frame = pd.DataFrame({
        "outer_seed": context.seed,
        "method": method,
        "acquisition_round": acquisition_round,
        "selection_order": np.arange(BATCH_SIZE),
        "previous_U_t_position": selected_positions,
        "canonical_index": unlabeled_indices[selected_positions],
        "sample_id": selected_ids,
    })
    selection_path = artifacts / "selected_next_batch.csv"
    _write_csv_once(selection_path, selection_frame)
    protected.append(selection_path)
    acquisition_contract = {
        "input": input_contract,
        "metadata": metadata,
        "selected_ids_ordered_hash": stable_hash(selected_ids),
        "selected_ids_set_hash": ids_hash(selected_ids),
        "files": {str(path.relative_to(directory)): sha256_file(path) for path in protected},
    }
    _write_json_once(artifacts / "contract.json", acquisition_contract)
    return selected_ids, acquisition_contract, extra_fits


def _completed_round(directory: Path, expected_input: dict) -> dict | None:
    path = directory / "contract.json"
    if not path.exists():
        return None
    contract = _json(path)
    if contract.get("input") != expected_input:
        raise RuntimeError(f"completed round contract mismatch: {directory}")
    _assert_protected(directory, contract["files"])
    return contract


def _verify_round_artifacts(directory: Path, record: dict) -> None:
    """Verify a round receipt and its nested acquisition receipt recursively."""

    _assert_protected(directory, record["files"])
    acquisition_hash = record.get("acquisition_contract_hash")
    acquisition_path = directory / "acquisition_artifacts/contract.json"
    if acquisition_hash is None:
        if acquisition_path.exists():
            raise RuntimeError("final round unexpectedly contains acquisition artifacts")
        return
    acquisition = _json(acquisition_path)
    if stable_hash(acquisition) != acquisition_hash:
        raise RuntimeError("round/acquisition contract binding mismatch")
    _assert_protected(directory, acquisition["files"])


def run_method_trajectory(context: SequentialSeedContext, method: str) -> dict[str, object]:
    """Fit rounds 0..21 and reacquire after rounds 0..20, with exact resume."""

    if method not in METHODS:
        raise ValueError("unknown sequential method")
    store = context.new_method_store()
    labeled_indices = np.asarray(context.roles["l0"], dtype=int)
    unlabeled_indices = np.asarray(context.roles["u0"], dtype=int)
    labeled_truth = np.asarray(context.l0_truth, dtype=np.float32)
    incoming_selected_ids: list[str] = []
    all_selected_ids: list[str] = []
    round_freezes: list[dict[str, object]] = []
    all_fits: list[dict[str, object]] = []
    for round_index in range(ACQUISITION_ROUNDS + 1):
        started = time.perf_counter()
        directory = context.runtime / method / f"round_{round_index:02d}"
        incoming = pd.DataFrame({
            "outer_seed": [context.seed] * len(incoming_selected_ids),
            "method": [method] * len(incoming_selected_ids),
            "round": [round_index] * len(incoming_selected_ids),
            "selection_order": np.arange(len(incoming_selected_ids)),
            "sample_id": incoming_selected_ids,
        })
        input_contract = _round_input_contract(
            context, method, round_index, labeled_indices, unlabeled_indices, incoming_selected_ids
        )
        if len(labeled_indices) != ACTIVE_LABEL_BUDGETS[round_index]:
            raise RuntimeError("active-label schedule drift before fit")
        _save_round_state(
            directory,
            input_contract,
            labeled_indices,
            unlabeled_indices,
            context.ids(labeled_indices),
            context.ids(unlabeled_indices),
            incoming,
        )
        completed = _completed_round(directory, input_contract)
        model_runtime = directory / "model"
        evaluation_fit = context.fit(
            method=method,
            round_index=round_index,
            labeled_indices=labeled_indices,
            labeled_truth=labeled_truth,
            runtime=model_runtime,
        )
        _write_json_once(directory / "validation_metrics.json", {
            "outer_seed": context.seed,
            "method": method,
            "round": round_index,
            "selection_metric": "validation_combined_normalized_rmse",
            "best_validation_combined_normalized_rmse": evaluation_fit[
                "best_validation_combined_normalized_rmse"
            ],
            "best_epoch": evaluation_fit["best_epoch"],
            "epochs_run": evaluation_fit["epochs_run"],
            "test_labels_used": 0,
        })
        all_fits.append(evaluation_fit)
        if completed is not None:
            if completed["evaluation_fit_contract_hash"] != evaluation_fit["fit_contract_hash"]:
                raise RuntimeError("resumed evaluation fit differs from completed round")
        next_selected_ids: list[str] = []
        acquisition_contract: dict[str, object] | None = None
        extra_fits: list[dict[str, object]] = []
        if round_index < ACQUISITION_ROUNDS:
            next_selected_ids, acquisition_contract, extra_fits = _acquire_next(
                context,
                method,
                round_index,
                labeled_indices,
                unlabeled_indices,
                labeled_truth,
                evaluation_fit,
                directory,
            )
            all_fits.extend(extra_fits)
        protected = [
            directory / "labeled_ids.npy", directory / "labeled_ids.npy.contract.json",
            directory / "unlabeled_ids.npy", directory / "unlabeled_ids.npy.contract.json",
            directory / "labeled_indices.npy", directory / "labeled_indices.npy.contract.json",
            directory / "unlabeled_indices.npy", directory / "unlabeled_indices.npy.contract.json",
            directory / "selected_batch.csv", directory / "input_contract.json",
            directory / "validation_metrics.json",
            model_runtime / "best.pt", model_runtime / "predictions.csv.gz", model_runtime / "fit_audit.json",
        ]
        if acquisition_contract is not None:
            protected.extend((directory / "acquisition_artifacts/contract.json",
                              directory / "acquisition_artifacts/selected_next_batch.csv"))
        round_contract = {
            "input": input_contract,
            "status": "FROZEN_BEFORE_TEST_TRUTH",
            "evaluation_fit_contract_hash": evaluation_fit["fit_contract_hash"],
            "initialization_seed": evaluation_fit["initialization_seed"],
            "initialization_hash": evaluation_fit["initialization_hash"],
            "checkpoint_sha256": evaluation_fit["checkpoint_sha256"],
            "checkpoint_state_hash": evaluation_fit["checkpoint_state_hash"],
            "prediction_sha256": evaluation_fit["prediction_sha256"],
            "outgoing_selected_ids_hash": stable_hash(next_selected_ids),
            "acquisition_contract_hash": None if acquisition_contract is None else stable_hash(acquisition_contract),
            "test_truth_access_count": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "files": {str(path.relative_to(directory)): sha256_file(path) for path in protected},
        }
        if completed is None:
            _write_json_once(directory / "contract.json", round_contract)
        else:
            # Wall time is provenance, not a semantic resume input.
            left, right = dict(completed), dict(round_contract)
            left.pop("elapsed_seconds", None); right.pop("elapsed_seconds", None)
            if left != right:
                raise RuntimeError("resumed round artifacts differ from uninterrupted semantics")
            round_contract = completed
        round_freezes.append(round_contract)

        if round_index < ACQUISITION_ROUNDS:
            old_l_ids, old_u_ids = context.ids(labeled_indices), context.ids(unlabeled_indices)
            position_by_id = {sample_id: position for position, sample_id in enumerate(old_u_ids)}
            selected_positions = np.asarray([position_by_id[value] for value in next_selected_ids], dtype=int)
            selected_indices = unlabeled_indices[selected_positions]
            selected_set = set(next_selected_ids)
            next_unlabeled = np.asarray([
                index for index, sample_id in zip(unlabeled_indices, old_u_ids) if sample_id not in selected_set
            ], dtype=int)
            next_labeled = np.r_[labeled_indices, selected_indices]
            validate_trajectory_transition(
                old_l_ids,
                old_u_ids,
                next_selected_ids,
                context.ids(next_labeled),
                context.ids(next_unlabeled),
            )
            all_selected_ids.extend(next_selected_ids)
            if len(set(all_selected_ids)) != len(all_selected_ids):
                raise RuntimeError("trajectory selected an ID more than once")
            store.freeze_acquisitions(all_selected_ids)
            new_truth = store.reveal(next_selected_ids, "after_acquisition_fit")
            labeled_truth = np.vstack([labeled_truth, new_truth])
            labeled_indices, unlabeled_indices = next_labeled, next_unlabeled
            incoming_selected_ids = next_selected_ids

    if len(labeled_indices) != FINAL_ACTIVE_LABELS or len(all_selected_ids) != BATCH_SIZE * ACQUISITION_ROUNDS:
        raise RuntimeError("trajectory did not reach the frozen final budget")
    pd.DataFrame(store.audit).to_csv(context.runtime / method / "label_access_audit.csv", index=False)
    pd.DataFrame(all_fits).to_csv(context.runtime / method / "fit_audit.csv", index=False)
    trajectory = {
        "status": "FROZEN_BEFORE_TEST_TRUTH",
        "outer_seed": context.seed,
        "method": method,
        "rounds": ACQUISITION_ROUNDS,
        "round_points": ACQUISITION_ROUNDS + 1,
        "final_active_labels": len(labeled_indices),
        "selected_ids_hash": stable_hash(all_selected_ids),
        "final_L_t_ids_hash": ids_hash(context.ids(labeled_indices)),
        "final_U_t_ids_hash": ids_hash(context.ids(unlabeled_indices)),
        "test_truth_access_count": 0,
        "round_contract_hashes": [stable_hash(value) for value in round_freezes],
        "files": {
            "label_access_audit.csv": sha256_file(context.runtime / method / "label_access_audit.csv"),
            "fit_audit.csv": sha256_file(context.runtime / method / "fit_audit.csv"),
        },
    }
    _write_json_once(context.runtime / method / "trajectory_freeze.json", trajectory)
    return trajectory


def run_full_data_reference(context: SequentialSeedContext) -> dict[str, object]:
    """Train the matched full-outer reference once; it never enters acquisition."""

    runtime = context.runtime / "full_data_reference"
    store = context.new_method_store()
    u0_ids = context.ids(context.roles["u0"])
    store.freeze_acquisitions(u0_ids)
    u0_truth = store.reveal(u0_ids, "after_acquisition_fit")
    truth = np.vstack([context.l0_truth, u0_truth])
    fit = context.fit(
        method="full_data_reference",
        round_index=-1,
        labeled_indices=context.outer,
        labeled_truth=truth,
        runtime=runtime / "model",
    )
    pd.DataFrame(store.audit).to_csv(runtime / "label_access_audit.csv", index=False)
    freeze = {
        "status": "FROZEN_BEFORE_TEST_TRUTH",
        "outer_seed": context.seed,
        "active_label_count": OUTER_TRAIN_ROWS,
        "acquisition_use": False,
        "endpoint_scales": context.preprocessing["target_scales"],
        "initialization_seed": fit["initialization_seed"],
        "initialization_hash": fit["initialization_hash"],
        "checkpoint_sha256": fit["checkpoint_sha256"],
        "checkpoint_state_hash": fit["checkpoint_state_hash"],
        "prediction_sha256": fit["prediction_sha256"],
        "test_truth_access_count": 0,
        "files": {
            "model/best.pt": sha256_file(runtime / "model/best.pt"),
            "model/predictions.csv.gz": sha256_file(runtime / "model/predictions.csv.gz"),
            "model/fit_audit.json": sha256_file(runtime / "model/fit_audit.json"),
            "label_access_audit.csv": sha256_file(runtime / "label_access_audit.csv"),
        },
    }
    _write_json_once(runtime / "pre_test_freeze.json", freeze)
    return freeze


def _partition(seed: int, study: Path = STUDY) -> pd.DataFrame:
    manifest = _json(study / "splits/split_manifest.json")
    record = next(item for item in manifest["splits"] if int(item["outer_seed"]) == int(seed))
    path = study / "splits" / f"row_seed_{seed}.csv"
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"frozen sequential split changed for seed {seed}")
    return pd.read_csv(path)


def build_global_pre_test_freeze(study: Path = STUDY) -> dict[str, object]:
    """Verify every trajectory/checkpoint/prediction before any test reveal."""

    entries: dict[str, dict[str, object]] = {}
    for seed in CONFIRMATION_SEEDS:
        seed_runtime = Path(study) / "runtime" / f"seed_{seed}"
        for method in METHODS:
            trajectory_path = seed_runtime / method / "trajectory_freeze.json"
            trajectory = _json(trajectory_path)
            if (trajectory.get("status") != "FROZEN_BEFORE_TEST_TRUTH" or
                    trajectory.get("round_points") != ACQUISITION_ROUNDS + 1 or
                    trajectory.get("test_truth_access_count") != 0):
                raise RuntimeError(f"trajectory is not at the pre-test barrier: seed={seed} method={method}")
            _assert_protected(seed_runtime / method, trajectory["files"])
            for round_index in range(ACQUISITION_ROUNDS + 1):
                directory = seed_runtime / method / f"round_{round_index:02d}"
                record = _json(directory / "contract.json")
                if record.get("status") != "FROZEN_BEFORE_TEST_TRUTH" or record.get("test_truth_access_count") != 0:
                    raise RuntimeError("invalid round pre-test freeze")
                _verify_round_artifacts(directory, record)
                key = f"seed_{seed}/{method}/round_{round_index:02d}"
                entries[key] = {
                    "outer_seed": seed,
                    "method": method,
                    "round": round_index,
                    "active_ids_hash": record["input"]["L_t_ids_hash"],
                    "checkpoint_hash": record["checkpoint_sha256"],
                    "checkpoint_state_hash": record["checkpoint_state_hash"],
                    "prediction_hash": record["prediction_sha256"],
                    "round_contract_sha256": sha256_file(directory / "contract.json"),
                }
        full_directory = seed_runtime / "full_data_reference"
        full = _json(full_directory / "pre_test_freeze.json")
        if full.get("status") != "FROZEN_BEFORE_TEST_TRUTH" or full.get("test_truth_access_count") != 0:
            raise RuntimeError("invalid full-data pre-test freeze")
        _assert_protected(full_directory, full["files"])
        entries[f"seed_{seed}/full_data_reference"] = {
            "outer_seed": seed,
            "method": "full_data_reference",
            "active_ids_hash": ids_hash(_partition(seed, Path(study)).loc[
                lambda rows: rows.role.isin(["l0", "u0"]), "sample_id"
            ]),
            "checkpoint_hash": full["checkpoint_sha256"],
            "checkpoint_state_hash": full["checkpoint_state_hash"],
            "prediction_hash": full["prediction_sha256"],
            "round_contract_sha256": sha256_file(full_directory / "pre_test_freeze.json"),
        }
    expected = len(CONFIRMATION_SEEDS) * (len(METHODS) * (ACQUISITION_ROUNDS + 1) + 1)
    if len(entries) != expected:
        raise RuntimeError("global freeze matrix is incomplete")
    freeze = {
        "status": "FROZEN_BEFORE_TEST_TRUTH",
        "trajectory_description": "5 seeds x 3 methods x rounds 0..21 plus 5 full-data references",
        "registered_acquisition_rounds": ACQUISITION_ROUNDS,
        "frozen_round_points": len(CONFIRMATION_SEEDS) * len(METHODS) * (ACQUISITION_ROUNDS + 1),
        "test_truth_access_count": 0,
        "entries": entries,
    }
    atomic_json(Path(study) / "global_pre_test_freeze.json", freeze)
    return freeze


def execute_pre_test(commit: str, study: Path = STUDY) -> dict[str, object]:
    """The formal long run. This function never reads test truth."""

    commit = assert_formal_authorized(commit, Path(study))
    placeholder = _json(Path(study) / "global_pre_test_freeze.json")
    if placeholder.get("status") == "FROZEN_BEFORE_TEST_TRUTH":
        return build_global_pre_test_freeze(Path(study))
    if placeholder.get("status") != "PENDING_FORMAL_RUN":
        raise RuntimeError("unexpected global freeze state")
    for seed in CONFIRMATION_SEEDS:
        execute_pre_test_seed(commit, seed, Path(study))
    return build_global_pre_test_freeze(Path(study))


def execute_pre_test_seed(commit: str, seed: int, study: Path = STUDY) -> dict[str, object]:
    """Run one complete seed shard, permitting independent seed-level scheduling."""

    commit = assert_formal_authorized(commit, Path(study))
    if int(seed) not in CONFIRMATION_SEEDS:
        raise ValueError("seed shard must belong to the frozen confirmation cohort")
    placeholder = _json(Path(study) / "global_pre_test_freeze.json")
    if placeholder.get("status") != "PENDING_FORMAL_RUN":
        raise RuntimeError("seed execution is closed after the global pre-test freeze")
    context = SequentialSeedContext(
        seed=int(seed),
        partition=_partition(int(seed), Path(study)),
        runtime=Path(study) / "runtime" / f"seed_{int(seed)}",
        commit=commit,
    )
    frozen: dict[str, object] = {}
    for method in METHODS:
        frozen[method] = run_method_trajectory(context, method)
        print(json.dumps({"seed": int(seed), "method_frozen": method,
                          "test_truth_access_count": 0}), flush=True)
    frozen["full_data_reference"] = run_full_data_reference(context)
    print(json.dumps({"seed": int(seed), "full_data_reference_frozen": True,
                      "test_truth_access_count": 0}), flush=True)
    return {"seed": int(seed), "status": "SEED_FROZEN_BEFORE_TEST_TRUTH", "artifacts": frozen}


def finalize_pre_test(commit: str, study: Path = STUDY) -> dict[str, object]:
    """Close the global barrier after all independently run seed shards exist."""

    assert_formal_authorized(commit, Path(study))
    return build_global_pre_test_freeze(Path(study))


def _verify_global_freeze(study: Path) -> dict[str, object]:
    freeze = _json(study / "global_pre_test_freeze.json")
    if freeze.get("status") != "FROZEN_BEFORE_TEST_TRUTH" or freeze.get("test_truth_access_count") != 0:
        raise RuntimeError("test evaluation requires the complete global pre-test freeze")
    expected = len(CONFIRMATION_SEEDS) * (len(METHODS) * (ACQUISITION_ROUNDS + 1) + 1)
    if len(freeze.get("entries", {})) != expected:
        raise RuntimeError("global pre-test freeze entry count changed")
    for key, entry in freeze["entries"].items():
        if entry["method"] == "full_data_reference":
            path = study / "runtime" / f"seed_{entry['outer_seed']}" / "full_data_reference/pre_test_freeze.json"
        else:
            path = study / "runtime" / f"seed_{entry['outer_seed']}" / entry["method"] / f"round_{entry['round']:02d}/contract.json"
        if sha256_file(path) != entry["round_contract_sha256"]:
            raise RuntimeError(f"global freeze artifact changed: {key}")
        record = _json(path)
        if entry["method"] == "full_data_reference":
            _assert_protected(path.parent, record["files"])
        else:
            _verify_round_artifacts(path.parent, record)
    return freeze


def reveal_test_and_report(commit: str, study: Path = STUDY) -> dict[str, object]:
    """Reveal test labels only after the global barrier, then compute all outputs."""

    assert_formal_authorized(commit, Path(study))
    _verify_global_freeze(Path(study))
    learning_rows: list[dict[str, object]] = []
    full_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    integrity_rows: list[dict[str, object]] = []
    label_access_rows: list[dict[str, object]] = []
    initialization_rows: list[dict[str, object]] = []
    for seed in CONFIRMATION_SEEDS:
        partition = _partition(seed, Path(study))
        data = load_features(SOURCE_DATA)
        test_ids = partition.loc[partition.role.eq("test"), "sample_id"].astype(str).tolist()
        scales = _json(Path(study) / "runtime" / f"seed_{seed}" / "context.json")["contract"]["preprocessing"]["target_scales"]
        seed_initialization_hashes: set[str] = set()
        for method in METHODS:
            method_runtime = Path(study) / "runtime" / f"seed_{seed}" / method
            trajectory = _json(method_runtime / "trajectory_freeze.json")
            final_labeled = np.load(method_runtime / f"round_{ACQUISITION_ROUNDS:02d}/labeled_indices.npy")
            selected_ids = [value for value in data.iloc[final_labeled].sample_id.astype(str).tolist()
                            if value not in set(partition.loc[partition.role.eq("l0"), "sample_id"].astype(str))]
            store = RestrictedLabelStore(SOURCE_DATA, partition)
            store.freeze_acquisitions(selected_ids)
            store.freeze_predictions()
            truth = store.reveal(test_ids, "final_test_evaluation")
            label_access_rows.extend({"outer_seed": seed, "method": method, **row} for row in store.audit)
            previous_l: list[str] | None = None
            previous_u: list[str] | None = None
            for round_index in range(ACQUISITION_ROUNDS + 1):
                directory = method_runtime / f"round_{round_index:02d}"
                contract = _json(directory / "contract.json")
                prediction = pd.read_csv(directory / "model/predictions.csv.gz")
                if prediction.sample_id.astype(str).tolist() != test_ids:
                    raise RuntimeError("test prediction identity/order mismatch after global freeze")
                current_l = np.load(directory / "labeled_ids.npy").astype(str).tolist()
                current_u = np.load(directory / "unlabeled_ids.npy").astype(str).tolist()
                incoming = pd.read_csv(directory / "selected_batch.csv").sample_id.astype(str).tolist()
                if round_index:
                    validate_trajectory_transition(previous_l, previous_u, incoming, current_l, current_u)
                previous_l, previous_u = current_l, current_u
                integrity_rows.append({
                    "outer_seed": seed, "method": method, "round": round_index,
                    "active_label_count": len(current_l), "unlabeled_count": len(current_u),
                    "incoming_batch_count": len(incoming), "selected_ids_unique": len(set(current_l)) == len(current_l),
                    "selected_from_previous_U_t": True, "trajectory_transition_pass": True,
                })
                fit = _json(directory / "model/fit_audit.json")
                seed_initialization_hashes.add(fit["initialization_hash"])
                initialization_rows.append({
                    "outer_seed": seed, "method": method, "round": round_index,
                    "initialization_seed": fit["initialization_seed"],
                    "initialization_hash": fit["initialization_hash"],
                    "checkpoint_state_hash": fit["checkpoint_state_hash"],
                })
                runtime_rows.append({
                    "outer_seed": seed, "method": method, "round": round_index,
                    "training_seconds": fit["training_seconds"], "epochs_run": fit["epochs_run"],
                    "best_epoch": fit["best_epoch"],
                    "acquisition_seconds_included_in_round": contract["elapsed_seconds"],
                })
                learning_rows.append({
                    "outer_seed": seed, "method": method, "round": round_index,
                    **label_budget(len(current_l)),
                    **metric_row(truth, prediction.drop(columns="sample_id").to_numpy(float), scales),
                })
            if trajectory["final_active_labels"] != FINAL_ACTIVE_LABELS:
                raise RuntimeError("frozen trajectory final active-label count changed")
        if len(seed_initialization_hashes) != 1:
            raise RuntimeError(f"evaluation initialization differs across methods/rounds for seed {seed}")

        full_runtime = Path(study) / "runtime" / f"seed_{seed}" / "full_data_reference"
        store = RestrictedLabelStore(SOURCE_DATA, partition)
        store.freeze_acquisitions(partition.loc[partition.role.eq("u0"), "sample_id"].astype(str))
        store.freeze_predictions()
        truth = store.reveal(test_ids, "final_test_evaluation")
        prediction = pd.read_csv(full_runtime / "model/predictions.csv.gz")
        if prediction.sample_id.astype(str).tolist() != test_ids:
            raise RuntimeError("full-data prediction identity/order mismatch")
        fit = _json(full_runtime / "model/fit_audit.json")
        full_rows.append({
            "outer_seed": seed, **label_budget(OUTER_TRAIN_ROWS),
            **metric_row(truth, prediction.drop(columns="sample_id").to_numpy(float), scales),
        })
        runtime_rows.append({
            "outer_seed": seed, "method": "full_data_reference", "round": -1,
            "training_seconds": fit["training_seconds"], "epochs_run": fit["epochs_run"],
            "best_epoch": fit["best_epoch"], "acquisition_seconds_included_in_round": 0.0,
        })
        label_access_rows.extend({"outer_seed": seed, "method": "full_data_reference", **row} for row in store.audit)

    return write_final_outputs(
        Path(study),
        pd.DataFrame(learning_rows),
        pd.DataFrame(full_rows),
        pd.DataFrame(runtime_rows),
        pd.DataFrame(integrity_rows),
        pd.DataFrame(label_access_rows),
        pd.DataFrame(initialization_rows),
    )
