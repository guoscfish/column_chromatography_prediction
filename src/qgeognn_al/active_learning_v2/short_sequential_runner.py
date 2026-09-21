"""Resumable execution for Center/Width-LCMD and Direction-LCMD through 429."""

from __future__ import annotations

from dataclasses import asdict
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
from .gradient_features import (
    extract_linear_output_gradient_sketches,
    extract_q50_gradient_sketches,
    state_dict_hash,
)
from .gradient_transforms import center_width_transform, power_normalize_gradients
from .lcmd import lcmd_tp_select
from .maxdet_study import historical_round0_paths
from .protocol import RestrictedLabelStore, ids_hash, stable_hash, validate_row_protocol
from .runner import fit_from_same_initialization
from .sequential_acquisition import validate_trajectory_transition
from .short_sequential_study import (
    ACQUISITION_ROUNDS,
    ACTIVE_LABEL_BUDGETS,
    BATCH_SIZE,
    FINAL_ACTIVE_LABELS,
    METHODS,
    SEEDS,
    SKETCH_DIMENSION,
    STUDY,
    protocol_record,
    split_path,
    validate_prepared,
)


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _write_json_once(path: Path, value: dict) -> None:
    if path.exists():
        if _json(path) != value:
            raise RuntimeError(f"refusing mismatched JSON artifact: {path}")
        return
    atomic_json(path, value)


def _write_csv_once(path: Path, value: pd.DataFrame) -> None:
    if path.exists():
        current = pd.read_csv(path, keep_default_na=False)
        expected = value.copy()
        if list(current.columns) != list(expected.columns) or len(current) != len(expected):
            raise RuntimeError(f"refusing mismatched CSV artifact: {path}")
        for column in current:
            if pd.api.types.is_numeric_dtype(current[column]) and pd.api.types.is_numeric_dtype(expected[column]):
                same = np.allclose(current[column].to_numpy(float), expected[column].to_numpy(float), atol=1e-8, rtol=1e-7)
            else:
                same = current[column].astype(str).tolist() == expected[column].astype(str).tolist()
            if not same:
                raise RuntimeError(f"refusing mismatched CSV artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    value.to_csv(path, index=False)


class ShortSequentialContext:
    def __init__(self, seed: int, study: Path = STUDY):
        self.seed = int(seed)
        self.study = Path(study)
        self.runtime = self.study / "runtime" / f"seed_{self.seed}"
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.partition = pd.read_csv(split_path(self.seed))
        validate_row_protocol(self.partition)
        self.data = load_features(SOURCE_DATA)
        if self.partition.sample_id.astype(str).tolist() != self.data.sample_id.astype(str).tolist():
            raise RuntimeError("short-sequential split/source order drift")
        self.roles = {role: self.partition.loc[self.partition.role.eq(role), "canonical_index"].to_numpy(int)
                      for role in ("l0", "u0", "validation", "test")}
        old = _json(historical_round0_paths(self.seed)["context"])["contract"]
        self.preprocessing = old["preprocessing"]
        self.normalization = ConditionNormalization(**old["normalization"])
        self.atom, self.angle = torch.load(historical_round0_paths(self.seed)["scrubbed_graphs"], weights_only=False)
        if any(torch.count_nonzero(item.y) for item in self.atom):
            raise RuntimeError("historical scrubbed graph cache contains labels")
        store = RestrictedLabelStore(SOURCE_DATA, self.partition)
        self.l0_truth = store.reveal(self.ids(self.roles["l0"]), "initial_fit")
        self.validation_truth = store.reveal(self.ids(self.roles["validation"]), "initial_fit")
        self.cw_transform, self.cw_audit = center_width_transform(self.l0_truth)
        self.protocol_hash = stable_hash(protocol_record())
        _write_json_once(self.runtime / "context.json", {
            "study": self.study.name,
            "protocol_hash": self.protocol_hash,
            "outer_seed": self.seed,
            "split_hash": stable_hash(self.partition.to_dict("list")),
            "l0_ids_hash": ids_hash(self.ids(self.roles["l0"])),
            "u0_ids_hash": ids_hash(self.ids(self.roles["u0"])),
            "validation_ids_hash": ids_hash(self.ids(self.roles["validation"])),
            "test_ids_hash": ids_hash(self.ids(self.roles["test"])),
            "normalization": asdict(self.normalization),
            "preprocessing": self.preprocessing,
            "center_width_transform": self.cw_audit,
            "test_truth_access_count": 0,
        })

    def ids(self, indices: Sequence[int]) -> list[str]:
        return self.data.iloc[list(indices)].sample_id.astype(str).tolist()

    def new_store(self) -> RestrictedLabelStore:
        store = RestrictedLabelStore(SOURCE_DATA, self.partition)
        if not np.array_equal(store.reveal(self.ids(self.roles["l0"]), "initial_fit"), self.l0_truth):
            raise RuntimeError("L0 truth drift")
        if not np.array_equal(store.reveal(self.ids(self.roles["validation"]), "initial_fit"), self.validation_truth):
            raise RuntimeError("validation truth drift")
        return store

    def training_config(self) -> dict:
        config = seed_config(self.seed, 0, smoke=False)
        config["split_sha256"] = stable_hash(self.partition.to_dict("list"))
        return config

    def fit(self, method: str, round_index: int, labeled: np.ndarray, truth: np.ndarray, runtime: Path) -> dict:
        train_ids = self.ids(labeled)
        contract = {
            "study": self.study.name,
            "protocol_hash": self.protocol_hash,
            "outer_seed": self.seed,
            "method": method,
            "round": round_index,
            "member": 0,
            "train_sample_ids": train_ids,
            "validation_sample_ids": self.ids(self.roles["validation"]),
            "L_t_ids_hash": ids_hash(train_ids),
            "prediction_role": "test_X_without_test_truth",
            "test_truth_access_count": 0,
        }
        audit = fit_from_same_initialization(
            atom_base=self.atom,
            angle=self.angle,
            normalization=self.normalization,
            preprocessing=self.preprocessing,
            train_indices=labeled,
            train_truth=truth,
            validation_indices=self.roles["validation"],
            validation_truth=self.validation_truth,
            prediction_indices=self.roles["test"],
            prediction_sample_ids=self.ids(self.roles["test"]),
            initialization_seed=initialization_seed(self.seed, 0),
            training_config=self.training_config(),
            contract=contract,
            runtime=runtime,
        )
        return {"reuse_status": "new_fit", **audit}


def select_transformed_lcmd(features: np.ndarray, labeled_count: int, batch_size: int = BATCH_SIZE) -> dict:
    values = np.asarray(features)
    labeled = int(labeled_count)
    if values.ndim != 2 or values.shape[1] != SKETCH_DIMENSION or not 0 < labeled < len(values):
        raise ValueError("ordered current L_t + U_t 512D features required")
    selected = lcmd_tp_select(values[labeled:], values[:labeled], int(batch_size))
    positions = np.asarray(selected.selected_pool_positions, dtype=int)
    if len(positions) != batch_size or len(np.unique(positions)) != batch_size:
        raise RuntimeError("LCMD selection must contain exactly 32 unique U_t positions")
    return {"selected_pool_positions": positions, "center_count": labeled, "trace": selected.trace}


def _round0_evaluation(context: ShortSequentialContext) -> tuple[Path, Path, dict]:
    paths = historical_round0_paths(context.seed)
    fit = _json(paths["member0_audit"])
    prediction = pd.read_csv(paths["member0_predictions"])
    if prediction.sample_id.astype(str).tolist() != context.ids(context.roles["test"]):
        raise RuntimeError("historical round-0 prediction identity drift")
    return paths["member0_model"], paths["member0_predictions"], {
        "reuse_status": "reused_historical_round0",
        "source_checkpoint": str(paths["member0_model"]),
        **fit,
    }


def _load_cached_features(path: Path, contract: dict, current: np.ndarray) -> tuple[np.ndarray, dict] | None:
    receipt = path.with_suffix(path.suffix + ".contract.json")
    if not path.exists() and not receipt.exists():
        return None
    if not path.exists() or not receipt.exists():
        raise RuntimeError(f"partial feature cache: {path}")
    saved = _json(receipt)
    if saved.get("contract") != contract or saved.get("sha256") != sha256_file(path):
        raise RuntimeError(f"feature cache contract mismatch: {path}")
    with np.load(path) as cache:
        features = cache["features"]
        indices = cache["canonical_indices"]
    if not np.array_equal(indices, current) or features.shape != (len(current), SKETCH_DIMENSION):
        raise RuntimeError("feature cache order/shape mismatch")
    return features, saved


def _cache_features(path: Path, contract: dict, current: np.ndarray, features: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=np.asarray(features), canonical_indices=np.asarray(current, dtype=int))
    atomic_json(path.with_suffix(path.suffix + ".contract.json"), {"contract": contract, "sha256": sha256_file(path)})


def _extract_features(
    context: ShortSequentialContext,
    method: str,
    round_index: int,
    model_path: Path,
    current: np.ndarray,
    labeled_count: int,
    directory: Path,
) -> tuple[np.ndarray, dict, float]:
    artifact = directory / "acquisition_artifacts"
    artifact.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = sha256_file(model_path)
    contract = {
        "study": context.study.name,
        "protocol_hash": context.protocol_hash,
        "outer_seed": context.seed,
        "method": method,
        "source_round": round_index,
        "checkpoint_sha256": checkpoint_sha,
        "ordered_current_indices": current.tolist(),
        "ordered_current_ids_hash": stable_hash(context.ids(current)),
        "L_t_ids_hash": ids_hash(context.ids(current[:labeled_count])),
        "U_t_ids_hash": ids_hash(context.ids(current[labeled_count:])),
        "center_count": labeled_count,
        "dimension": SKETCH_DIMENSION,
        "sketch_seed": sketch_seed(context.seed),
        "transform": method,
        "test_truth_access_count": 0,
    }
    path = artifact / "current_transformed_gradient_features.npz"
    cached = _load_cached_features(path, contract, current)
    if cached is not None:
        return cached[0], _json(artifact / "gradient_audit.json"), 0.0
    model = load_predictor_checkpoint(model_path)
    if state_dict_hash(model) != _json(model_path.parent / "fit_audit.json").get("checkpoint_state_hash") and round_index != 0:
        raise RuntimeError("current checkpoint state hash mismatch")
    started = time.perf_counter()
    if method == "direction_lcmd":
        if round_index == 0:
            source = historical_round0_paths(context.seed)["gradient"]
            with np.load(source) as cache:
                raw = np.asarray(cache["features"])
                indices = np.asarray(cache["canonical_indices"])
            if not np.array_equal(indices, current):
                raise RuntimeError("historical round-0 raw gradient order drift")
            raw_audit = _json(historical_round0_paths(context.seed)["gradient_audit"])
            raw_status = "reused_historical_round0"
        else:
            scales = tuple(float(context.preprocessing["target_scales"][key]) for key in ("V1", "V2"))
            raw_result = extract_q50_gradient_sketches(
                model, context.atom, context.angle, current, scales,
                dimension=SKETCH_DIMENSION, sketch_seed=sketch_seed(context.seed),
            )
            raw, raw_audit, raw_status = raw_result.features, raw_result.audit, "new_current_model_extraction"
        transformed = power_normalize_gradients(raw, 0.0)
        features = transformed.features
        audit = {
            "method": method,
            "raw_gradient_status": raw_status,
            "raw_gradient_audit": raw_audit,
            "transform_audit": transformed.audit,
            "current_checkpoint_sha256": checkpoint_sha,
            "current_model_round": round_index,
            "feature_hash": array_hash(features),
        }
    elif method == "center_width_lcmd":
        result = extract_linear_output_gradient_sketches(
            model, context.atom, context.angle, current, context.cw_transform,
            dimension=SKETCH_DIMENSION, sketch_seed=sketch_seed(context.seed),
        )
        features = result.features
        audit = {
            "method": method,
            "gradient_audit": result.audit,
            "transform_audit": context.cw_audit,
            "current_checkpoint_sha256": checkpoint_sha,
            "current_model_round": round_index,
            "feature_hash": array_hash(features),
        }
    else:
        raise ValueError(f"unsupported short-sequential method: {method}")
    elapsed = time.perf_counter() - started
    audit["gradient_extraction_seconds"] = elapsed
    _cache_features(path, contract, current, features)
    _write_json_once(artifact / "gradient_audit.json", audit)
    return features, audit, elapsed


def _acquire(
    context: ShortSequentialContext,
    method: str,
    round_index: int,
    labeled: np.ndarray,
    unlabeled: np.ndarray,
    model_path: Path,
    directory: Path,
) -> tuple[list[str], dict]:
    acquisition = directory / "acquisition_artifacts"
    contract_path = acquisition / "contract.json"
    selected_path = acquisition / "selected_next_batch.csv"
    input_contract = {
        "study": context.study.name,
        "protocol_hash": context.protocol_hash,
        "outer_seed": context.seed,
        "method": method,
        "source_round": round_index,
        "acquisition_round": round_index + 1,
        "L_t_ids_hash": ids_hash(context.ids(labeled)),
        "U_t_ids_hash": ids_hash(context.ids(unlabeled)),
        "checkpoint_sha256": sha256_file(model_path),
        "current_model_round": round_index,
        "batch_size": BATCH_SIZE,
        "centers_are_all_current_L_t": True,
        "test_truth_access_count": 0,
    }
    if contract_path.exists():
        saved = _json(contract_path)
        if saved.get("input") != input_contract or saved.get("selected_table_sha256") != sha256_file(selected_path):
            raise RuntimeError(f"cached acquisition mismatch: {directory}")
        return pd.read_csv(selected_path).sample_id.astype(str).tolist(), saved
    current = np.r_[labeled, unlabeled]
    features, feature_audit, gradient_seconds = _extract_features(
        context, method, round_index, model_path, current, len(labeled), directory
    )
    started = time.perf_counter()
    selection = select_transformed_lcmd(features, len(labeled))
    selector_seconds = time.perf_counter() - started
    positions = selection["selected_pool_positions"]
    selected_ids = np.asarray(context.ids(unlabeled))[positions].astype(str).tolist()
    if len(selected_ids) != BATCH_SIZE or len(set(selected_ids)) != BATCH_SIZE:
        raise RuntimeError("acquisition did not produce 32 unique IDs")
    table = pd.DataFrame({
        "outer_seed": context.seed,
        "method": method,
        "source_round": round_index,
        "acquisition_round": round_index + 1,
        "selection_order": np.arange(BATCH_SIZE),
        "pool_position": positions,
        "canonical_index": unlabeled[positions],
        "sample_id": selected_ids,
    })
    _write_csv_once(selected_path, table)
    _write_csv_once(acquisition / "lcmd_trace.csv", pd.DataFrame(selection["trace"]))
    contract = {
        "input": input_contract,
        "selected_ids_ordered_hash": stable_hash(selected_ids),
        "selected_ids_set_hash": ids_hash(selected_ids),
        "selected_table_sha256": sha256_file(selected_path),
        "feature_hash": feature_audit["feature_hash"],
        "feature_current_model_round": feature_audit["current_model_round"],
        "center_count": selection["center_count"],
        "gradient_extraction_seconds": gradient_seconds,
        "representation_prediction_seconds": 0.0,
        "selector_seconds": selector_seconds,
        "test_truth_access_count": 0,
    }
    _write_json_once(contract_path, contract)
    return selected_ids, contract


def run_trajectory(context: ShortSequentialContext, method: str) -> dict:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    store = context.new_store()
    labeled = context.roles["l0"].copy()
    unlabeled = context.roles["u0"].copy()
    truth = context.l0_truth.copy()
    incoming: list[str] = []
    selected_all: list[str] = []
    round_hashes: list[str] = []
    fit_rows: list[dict] = []
    for round_index in range(ACQUISITION_ROUNDS + 1):
        if len(labeled) != ACTIVE_LABEL_BUDGETS[round_index]:
            raise RuntimeError("active-label budget drift")
        directory = context.runtime / method / f"round_{round_index:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        input_contract = {
            "study": context.study.name,
            "protocol_hash": context.protocol_hash,
            "outer_seed": context.seed,
            "method": method,
            "round": round_index,
            "active_label_count": len(labeled),
            "unlabeled_count": len(unlabeled),
            "L_t_ids_hash": ids_hash(context.ids(labeled)),
            "U_t_ids_hash": ids_hash(context.ids(unlabeled)),
            "incoming_selected_ids_hash": stable_hash(incoming),
            "trajectory_specific_model": True,
            "test_truth_access_count": 0,
        }
        _write_json_once(directory / "input_contract.json", input_contract)
        _write_csv_once(directory / "state.csv", pd.DataFrame({
            "role": ["labeled"] * len(labeled) + ["unlabeled"] * len(unlabeled),
            "position": list(range(len(labeled))) + list(range(len(unlabeled))),
            "canonical_index": np.r_[labeled, unlabeled],
            "sample_id": context.ids(np.r_[labeled, unlabeled]),
        }))
        _write_csv_once(directory / "selected_batch.csv", pd.DataFrame({
            "outer_seed": [context.seed] * len(incoming),
            "method": [method] * len(incoming),
            "round": [round_index] * len(incoming),
            "selection_order": np.arange(len(incoming)),
            "sample_id": incoming,
        }))
        if round_index == 0:
            model_path, prediction_path, evaluation = _round0_evaluation(context)
        else:
            evaluation = context.fit(method, round_index, labeled, truth, directory / "model")
            model_path = directory / "model/best.pt"
            prediction_path = directory / "model/predictions.csv.gz"
        fit_rows.append({"outer_seed": context.seed, "method": method, "round": round_index, **evaluation})
        outgoing: list[str] = []
        acquisition_contract = None
        if round_index < ACQUISITION_ROUNDS:
            outgoing, acquisition_contract = _acquire(
                context, method, round_index, labeled, unlabeled, model_path, directory
            )
        freeze = {
            "input": input_contract,
            "status": "FROZEN_BEFORE_TEST_TRUTH",
            "checkpoint_path": str(model_path),
            "checkpoint_sha256": sha256_file(model_path),
            "checkpoint_state_hash": evaluation["checkpoint_state_hash"],
            "prediction_path": str(prediction_path),
            "prediction_sha256": sha256_file(prediction_path),
            "fit_contract_hash": evaluation["fit_contract_hash"],
            "outgoing_selected_ids_hash": stable_hash(outgoing),
            "acquisition_contract_hash": None if acquisition_contract is None else stable_hash(acquisition_contract),
            "test_truth_access_count": 0,
        }
        _write_json_once(directory / "round_freeze.json", freeze)
        round_hashes.append(sha256_file(directory / "round_freeze.json"))
        if round_index == ACQUISITION_ROUNDS:
            break
        old_l, old_u = context.ids(labeled), context.ids(unlabeled)
        position_by_id = {value: index for index, value in enumerate(old_u)}
        selected_indices = unlabeled[np.asarray([position_by_id[value] for value in outgoing], dtype=int)]
        selected_set = set(outgoing)
        next_unlabeled = np.asarray([index for index, value in zip(unlabeled, old_u) if value not in selected_set])
        next_labeled = np.r_[labeled, selected_indices]
        validate_trajectory_transition(old_l, old_u, outgoing, context.ids(next_labeled), context.ids(next_unlabeled))
        selected_all.extend(outgoing)
        if len(selected_all) != len(set(selected_all)):
            raise RuntimeError("trajectory selected an ID more than once")
        store.freeze_acquisitions(selected_all)
        truth = np.vstack([truth, store.reveal(outgoing, "after_acquisition_fit")])
        labeled, unlabeled, incoming = next_labeled, next_unlabeled, outgoing
    if len(labeled) != FINAL_ACTIVE_LABELS or len(selected_all) != ACQUISITION_ROUNDS * BATCH_SIZE:
        raise RuntimeError("trajectory did not stop exactly at 429")
    _write_csv_once(context.runtime / method / "fit_audit.csv", pd.DataFrame(fit_rows))
    _write_csv_once(context.runtime / method / "label_access_audit.csv", pd.DataFrame(store.audit))
    freeze = {
        "status": "FROZEN_BEFORE_TEST_TRUTH",
        "outer_seed": context.seed,
        "method": method,
        "round_points": ACQUISITION_ROUNDS + 1,
        "final_active_labels": len(labeled),
        "selected_ids_hash": stable_hash(selected_all),
        "round_freeze_sha256": round_hashes,
        "test_truth_access_count": 0,
    }
    _write_json_once(context.runtime / method / "trajectory_freeze.json", freeze)
    return freeze


def execute_seed(seed: int, study: Path = STUDY) -> dict:
    if int(seed) not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    validate_prepared(study)
    placeholder = _json(Path(study) / "global_pre_test_freeze.json")
    if placeholder.get("status") != "PENDING_TRAJECTORIES":
        raise RuntimeError("new trajectory execution is closed after global freeze")
    torch.set_num_threads(2)
    context = ShortSequentialContext(int(seed), Path(study))
    result = {}
    for method in METHODS:
        result[method] = run_trajectory(context, method)
        print(json.dumps({"seed": seed, "method_frozen": method, "final_active_labels": FINAL_ACTIVE_LABELS}), flush=True)
    return {"status": "SEED_FROZEN_BEFORE_TEST_TRUTH", "seed": int(seed), "methods": result}


def finalize_pre_test(study: Path = STUDY) -> dict:
    validate_prepared(study)
    entries = {}
    for seed in SEEDS:
        for method in METHODS:
            base = Path(study) / "runtime" / f"seed_{seed}" / method
            trajectory = _json(base / "trajectory_freeze.json")
            if trajectory.get("final_active_labels") != FINAL_ACTIVE_LABELS or trajectory.get("test_truth_access_count") != 0:
                raise RuntimeError("trajectory is not frozen at the 429 pre-test barrier")
            for round_index in range(ACQUISITION_ROUNDS + 1):
                path = base / f"round_{round_index:02d}" / "round_freeze.json"
                record = _json(path)
                if record.get("status") != "FROZEN_BEFORE_TEST_TRUTH" or record.get("test_truth_access_count") != 0:
                    raise RuntimeError("round is not frozen before test truth")
                if sha256_file(Path(record["checkpoint_path"])) != record["checkpoint_sha256"]:
                    raise RuntimeError("checkpoint hash drift before global freeze")
                if sha256_file(Path(record["prediction_path"])) != record["prediction_sha256"]:
                    raise RuntimeError("prediction hash drift before global freeze")
                entries[f"seed_{seed}/{method}/round_{round_index:02d}"] = {
                    "outer_seed": seed,
                    "method": method,
                    "round": round_index,
                    "active_label_count": ACTIVE_LABEL_BUDGETS[round_index],
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "checkpoint_state_hash": record["checkpoint_state_hash"],
                    "prediction_sha256": record["prediction_sha256"],
                    "round_freeze_sha256": sha256_file(path),
                }
    expected = len(SEEDS) * len(METHODS) * (ACQUISITION_ROUNDS + 1)
    if len(entries) != expected:
        raise RuntimeError("global pre-test matrix incomplete")
    freeze = {
        "status": "FROZEN_BEFORE_TEST_TRUTH",
        "entries": entries,
        "frozen_prediction_points": expected,
        "hard_stop_active_labels": FINAL_ACTIVE_LABELS,
        "ensemble_uncertainty_executed": False,
        "test_truth_access_count": 0,
    }
    atomic_json(Path(study) / "global_pre_test_freeze.json", freeze)
    return freeze


def reveal_and_report(study: Path = STUDY) -> dict:
    from .short_sequential_reporting import write_report

    freeze = _json(Path(study) / "global_pre_test_freeze.json")
    expected = len(SEEDS) * len(METHODS) * (ACQUISITION_ROUNDS + 1)
    if freeze.get("status") != "FROZEN_BEFORE_TEST_TRUTH" or len(freeze.get("entries", {})) != expected:
        raise RuntimeError("test reveal requires the complete global pre-test freeze")
    learning_rows: list[dict] = []
    label_access_rows: list[dict] = []
    for seed in SEEDS:
        context = ShortSequentialContext(seed, Path(study))
        test_ids = context.ids(context.roles["test"])
        for method in METHODS:
            method_runtime = Path(study) / "runtime" / f"seed_{seed}" / method
            final_state = pd.read_csv(method_runtime / f"round_{ACQUISITION_ROUNDS:02d}/state.csv")
            final_ids = final_state.loc[final_state.role.eq("labeled"), "sample_id"].astype(str).tolist()
            l0 = set(context.ids(context.roles["l0"]))
            selected = [value for value in final_ids if value not in l0]
            store = RestrictedLabelStore(SOURCE_DATA, context.partition)
            store.freeze_acquisitions(selected)
            store.freeze_predictions()
            truth = store.reveal(test_ids, "final_test_evaluation")
            label_access_rows.extend({"outer_seed": seed, "method": method, **row} for row in store.audit)
            for round_index, active in enumerate(ACTIVE_LABEL_BUDGETS):
                record = _json(method_runtime / f"round_{round_index:02d}/round_freeze.json")
                prediction = pd.read_csv(record["prediction_path"])
                if prediction.sample_id.astype(str).tolist() != test_ids:
                    raise RuntimeError("frozen prediction/test identity mismatch")
                from .benchmark_reporting import metric_row
                learning_rows.append({
                    "outer_seed": seed,
                    "method": method,
                    "round": round_index,
                    "active_label_count": active,
                    **metric_row(truth, prediction.drop(columns="sample_id").to_numpy(float), context.preprocessing["target_scales"]),
                })
    return write_report(Path(study), pd.DataFrame(learning_rows), pd.DataFrame(label_access_rows))
