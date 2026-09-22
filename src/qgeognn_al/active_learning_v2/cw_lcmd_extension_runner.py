"""Resumable CW-LCMD continuation from the exact 429 state through 525."""

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
from .cw_lcmd_extension_study import (
    ACTIVE_LABEL_BUDGETS,
    BATCH_SIZE,
    FINAL_ACTIVE_LABELS,
    FINAL_ROUND,
    METHOD,
    NEW_ACQUISITION_ROUNDS,
    SEEDS,
    SKETCH_DIMENSION,
    SOURCE_ROUND,
    STUDY,
    protocol_record,
    source_anchor,
    source_method_root,
    split_path,
    validate_prepared,
)
from .gradient_features import extract_linear_output_gradient_sketches, state_dict_hash
from .gradient_transforms import center_width_transform
from .lcmd import lcmd_tp_select
from .protocol import RestrictedLabelStore, ids_hash, stable_hash, validate_row_protocol
from .runner import fit_from_same_initialization
from .sequential_acquisition import validate_trajectory_transition


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


class CWExtensionContext:
    def __init__(self, seed: int, study: Path = STUDY, *, reuse_frozen_protocol_hash: bool = False):
        self.seed = int(seed)
        self.study = Path(study)
        self.runtime = self.study / "runtime" / f"seed_{self.seed}"
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.partition = pd.read_csv(split_path(self.seed))
        validate_row_protocol(self.partition)
        self.data = load_features(SOURCE_DATA)
        if self.partition.sample_id.astype(str).tolist() != self.data.sample_id.astype(str).tolist():
            raise RuntimeError("extension split/source order drift")
        self.roles = {role: self.partition.loc[self.partition.role.eq(role), "canonical_index"].to_numpy(int)
                      for role in ("l0", "u0", "validation", "test")}
        source_context = _json(source_anchor(self.seed)["context"])
        self.preprocessing = source_context["preprocessing"]
        self.normalization = ConditionNormalization(**source_context["normalization"])
        self.atom, self.angle = torch.load(source_anchor(self.seed)["scrubbed_graphs"], weights_only=False)
        if any(torch.count_nonzero(item.y) for item in self.atom):
            raise RuntimeError("source scrubbed graph cache contains labels")
        store = RestrictedLabelStore(SOURCE_DATA, self.partition)
        self.l0_truth = store.reveal(self.ids(self.roles["l0"]), "initial_fit")
        self.validation_truth = store.reveal(self.ids(self.roles["validation"]), "initial_fit")
        self.cw_transform, self.cw_audit = center_width_transform(self.l0_truth)
        if self.cw_audit != source_context["center_width_transform"]:
            raise RuntimeError("Center/Width transform differs from source trajectory")
        self.protocol_hash = stable_hash(protocol_record())
        context_payload = {
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
            "source_context_sha256": sha256_file(source_anchor(self.seed)["context"]),
            "test_truth_access_count": 0,
        }
        context_path = self.runtime / "context.json"
        if reuse_frozen_protocol_hash and context_path.exists():
            frozen_context = _json(context_path)
            frozen_without_hash = {key: value for key, value in frozen_context.items() if key != "protocol_hash"}
            current_without_hash = {key: value for key, value in context_payload.items() if key != "protocol_hash"}
            if frozen_without_hash != current_without_hash:
                raise RuntimeError(f"refusing mismatched frozen context: {context_path}")
            self.protocol_hash = frozen_context["protocol_hash"]
            context_payload["protocol_hash"] = self.protocol_hash
        _write_json_once(context_path, context_payload)

    def ids(self, indices: Sequence[int]) -> list[str]:
        return self.data.iloc[list(indices)].sample_id.astype(str).tolist()

    def training_config(self) -> dict:
        config = seed_config(self.seed, 0, smoke=False)
        config["split_sha256"] = stable_hash(self.partition.to_dict("list"))
        return config

    def fit(self, round_index: int, labeled: np.ndarray, truth: np.ndarray, runtime: Path) -> dict:
        train_ids = self.ids(labeled)
        contract = {
            "study": self.study.name,
            "protocol_hash": self.protocol_hash,
            "outer_seed": self.seed,
            "method": METHOD,
            "round": round_index,
            "member": 0,
            "train_sample_ids": train_ids,
            "validation_sample_ids": self.ids(self.roles["validation"]),
            "L_t_ids_hash": ids_hash(train_ids),
            "source_CW_trajectory": True,
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
        return {"reuse_status": "new_extension_fit", **audit}


def select_current_cw_lcmd(features: np.ndarray, labeled_count: int) -> dict[str, object]:
    values = np.asarray(features)
    labeled = int(labeled_count)
    if values.ndim != 2 or values.shape[1] != SKETCH_DIMENSION or not 0 < labeled < len(values):
        raise ValueError("ordered current L_t + U_t 512D Center/Width features required")
    result = lcmd_tp_select(values[labeled:], values[:labeled], BATCH_SIZE)
    positions = np.asarray(result.selected_pool_positions, dtype=int)
    if len(positions) != BATCH_SIZE or len(np.unique(positions)) != BATCH_SIZE:
        raise RuntimeError("CW-LCMD must select exactly 32 unique U_t positions")
    if np.any(positions < 0) or np.any(positions >= len(values) - labeled):
        raise RuntimeError("CW-LCMD selected outside current U_t")
    return {"selected_pool_positions": positions, "center_count": labeled, "trace": result.trace}


def _load_or_extract_features(
    context: CWExtensionContext,
    round_index: int,
    model_path: Path,
    labeled: np.ndarray,
    unlabeled: np.ndarray,
    directory: Path,
) -> tuple[np.ndarray, dict, float]:
    artifacts = directory / "acquisition_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    current = np.r_[labeled, unlabeled]
    checkpoint_sha = sha256_file(model_path)
    contract = {
        "study": context.study.name,
        "protocol_hash": context.protocol_hash,
        "outer_seed": context.seed,
        "method": METHOD,
        "source_round": round_index,
        "active_label_count": len(labeled),
        "checkpoint_sha256": checkpoint_sha,
        "ordered_current_indices": current.tolist(),
        "ordered_current_ids_hash": stable_hash(context.ids(current)),
        "L_t_ids_hash": ids_hash(context.ids(labeled)),
        "U_t_ids_hash": ids_hash(context.ids(unlabeled)),
        "center_count": len(labeled),
        "dimension": SKETCH_DIMENSION,
        "sketch_seed": sketch_seed(context.seed),
        "transform": context.cw_audit,
        "current_checkpoint_required": True,
        "test_truth_access_count": 0,
    }
    path = artifacts / "current_center_width_gradient_features.npz"
    receipt_path = path.with_suffix(path.suffix + ".contract.json")
    if path.exists() or receipt_path.exists():
        if not path.exists() or not receipt_path.exists():
            raise RuntimeError(f"partial CW gradient cache: {path}")
        receipt = _json(receipt_path)
        if receipt.get("contract") != contract or receipt.get("sha256") != sha256_file(path):
            raise RuntimeError(f"CW gradient cache contract mismatch: {path}")
        with np.load(path) as cache:
            features = cache["features"]
            indices = cache["canonical_indices"]
        if not np.array_equal(indices, current) or features.shape != (len(current), SKETCH_DIMENSION):
            raise RuntimeError("CW gradient cache order/shape mismatch")
        return features, _json(artifacts / "gradient_audit.json"), 0.0
    model = load_predictor_checkpoint(model_path)
    expected_state_hash = _json(model_path.with_name("fit_audit.json"))["checkpoint_state_hash"]
    if state_dict_hash(model) != expected_state_hash:
        raise RuntimeError("current CW checkpoint state hash mismatch")
    started = time.perf_counter()
    result = extract_linear_output_gradient_sketches(
        model,
        context.atom,
        context.angle,
        current,
        context.cw_transform,
        dimension=SKETCH_DIMENSION,
        sketch_seed=sketch_seed(context.seed),
    )
    elapsed = time.perf_counter() - started
    features = result.features
    audit = {
        "method": METHOD,
        "current_model_round": round_index,
        "current_active_labels": len(labeled),
        "current_checkpoint_sha256": checkpoint_sha,
        "current_checkpoint_state_hash": expected_state_hash,
        "gradient_audit": result.audit,
        "transform_audit": context.cw_audit,
        "feature_hash": array_hash(features),
        "gradient_extraction_seconds": elapsed,
    }
    np.savez_compressed(path, features=features, canonical_indices=current)
    atomic_json(receipt_path, {"contract": contract, "sha256": sha256_file(path)})
    _write_json_once(artifacts / "gradient_audit.json", audit)
    return features, audit, elapsed


def _acquire(
    context: CWExtensionContext,
    round_index: int,
    labeled: np.ndarray,
    unlabeled: np.ndarray,
    model_path: Path,
    directory: Path,
) -> tuple[list[str], dict]:
    artifacts = directory / "acquisition_artifacts"
    contract_path = artifacts / "contract.json"
    selected_path = artifacts / "selected_next_batch.csv"
    input_contract = {
        "study": context.study.name,
        "protocol_hash": context.protocol_hash,
        "outer_seed": context.seed,
        "method": METHOD,
        "source_round": round_index,
        "acquisition_round": round_index + 1,
        "active_label_count": len(labeled),
        "L_t_ids_hash": ids_hash(context.ids(labeled)),
        "U_t_ids_hash": ids_hash(context.ids(unlabeled)),
        "checkpoint_sha256": sha256_file(model_path),
        "current_model_round": round_index,
        "batch_size": BATCH_SIZE,
        "centers_are_all_current_L_t": True,
        "static_round3_ranking_forbidden": True,
        "test_truth_access_count": 0,
    }
    if contract_path.exists():
        saved = _json(contract_path)
        if saved.get("input") != input_contract or saved.get("selected_table_sha256") != sha256_file(selected_path):
            raise RuntimeError(f"cached extension acquisition mismatch: {directory}")
        return pd.read_csv(selected_path).sample_id.astype(str).tolist(), saved
    features, feature_audit, gradient_seconds = _load_or_extract_features(
        context, round_index, model_path, labeled, unlabeled, directory
    )
    started = time.perf_counter()
    selection = select_current_cw_lcmd(features, len(labeled))
    selector_seconds = time.perf_counter() - started
    positions = np.asarray(selection["selected_pool_positions"], dtype=int)
    selected_ids = np.asarray(context.ids(unlabeled))[positions].astype(str).tolist()
    if len(selected_ids) != BATCH_SIZE or len(set(selected_ids)) != BATCH_SIZE:
        raise RuntimeError("extension acquisition did not return 32 unique IDs")
    table = pd.DataFrame({
        "outer_seed": context.seed,
        "method": METHOD,
        "source_round": round_index,
        "acquisition_round": round_index + 1,
        "active_labels_before_acquisition": len(labeled),
        "selection_order": np.arange(BATCH_SIZE),
        "pool_position": positions,
        "canonical_index": unlabeled[positions],
        "sample_id": selected_ids,
    })
    _write_csv_once(selected_path, table)
    _write_csv_once(artifacts / "lcmd_trace.csv", pd.DataFrame(selection["trace"]))
    receipt = {
        "input": input_contract,
        "selected_ids_ordered_hash": stable_hash(selected_ids),
        "selected_ids_set_hash": ids_hash(selected_ids),
        "selected_table_sha256": sha256_file(selected_path),
        "feature_hash": feature_audit["feature_hash"],
        "feature_current_model_round": feature_audit["current_model_round"],
        "center_count": int(selection["center_count"]),
        "gradient_extraction_seconds": gradient_seconds,
        "selector_seconds": selector_seconds,
        "test_truth_access_count": 0,
    }
    _write_json_once(contract_path, receipt)
    return selected_ids, receipt


def _source_state(context: CWExtensionContext) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    paths = source_anchor(context.seed)
    state = pd.read_csv(paths["state"])
    labeled = state.loc[state.role.eq("labeled"), "canonical_index"].to_numpy(int)
    unlabeled = state.loc[state.role.eq("unlabeled"), "canonical_index"].to_numpy(int)
    selected: list[str] = []
    for source_round in range(SOURCE_ROUND):
        batch = pd.read_csv(source_method_root(context.seed) / f"round_{source_round:02d}/acquisition_artifacts/selected_next_batch.csv")
        selected.extend(batch.sample_id.astype(str).tolist())
    freeze = _json(paths["round_freeze"])
    if context.ids(labeled) != context.ids(context.roles["l0"]) + selected:
        raise RuntimeError("source L429 ordered lineage changed after preparation")
    return labeled, unlabeled, selected, freeze


def run_continuation(context: CWExtensionContext) -> dict:
    labeled, unlabeled, selected_all, source_freeze = _source_state(context)
    store = RestrictedLabelStore(SOURCE_DATA, context.partition)
    if not np.array_equal(store.reveal(context.ids(context.roles["l0"]), "initial_fit"), context.l0_truth):
        raise RuntimeError("extension L0 truth drift")
    if not np.array_equal(store.reveal(context.ids(context.roles["validation"]), "initial_fit"), context.validation_truth):
        raise RuntimeError("extension validation truth drift")
    store.freeze_acquisitions(selected_all)
    selected_truth = store.reveal(selected_all, "after_acquisition_fit")
    truth = np.vstack([context.l0_truth, selected_truth])
    if truth.shape != (429, 2):
        raise RuntimeError("extension did not reconstruct exactly L429 truth")
    incoming = pd.read_csv(source_method_root(context.seed) / "round_03/selected_batch.csv").sample_id.astype(str).tolist()
    fit_rows: list[dict] = []
    new_round_hashes: list[str] = []

    for round_index in range(SOURCE_ROUND, FINAL_ROUND + 1):
        expected_budget = ACTIVE_LABEL_BUDGETS[round_index]
        if len(labeled) != expected_budget:
            raise RuntimeError("CW extension active-label schedule drift")
        directory = context.runtime / METHOD / f"round_{round_index:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        input_contract = {
            "study": context.study.name,
            "protocol_hash": context.protocol_hash,
            "outer_seed": context.seed,
            "method": METHOD,
            "round": round_index,
            "active_label_count": len(labeled),
            "unlabeled_count": len(unlabeled),
            "L_t_ids_hash": ids_hash(context.ids(labeled)),
            "U_t_ids_hash": ids_hash(context.ids(unlabeled)),
            "incoming_selected_ids_hash": stable_hash(incoming),
            "source_CW_trajectory": True,
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
            "method": [METHOD] * len(incoming),
            "round": [round_index] * len(incoming),
            "selection_order": np.arange(len(incoming)),
            "sample_id": incoming,
        }))
        if round_index == SOURCE_ROUND:
            model_path = Path(source_freeze["checkpoint_path"])
            prediction_path = Path(source_freeze["prediction_path"])
            evaluation = {
                "reuse_status": "reused_exact_source_CW_L429",
                **_json(model_path.with_name("fit_audit.json")),
            }
        else:
            evaluation = context.fit(round_index, labeled, truth, directory / "model")
            model_path = directory / "model/best.pt"
            prediction_path = directory / "model/predictions.csv.gz"
        fit_rows.append({"outer_seed": context.seed, "method": METHOD, "round": round_index, **evaluation})
        outgoing: list[str] = []
        acquisition_receipt = None
        if round_index < FINAL_ROUND:
            outgoing, acquisition_receipt = _acquire(
                context, round_index, labeled, unlabeled, model_path, directory
            )
        freeze = {
            "input": input_contract,
            "status": "SOURCE_ANCHOR_FROZEN" if round_index == SOURCE_ROUND else "FROZEN_BEFORE_EXTENSION_TEST_TRUTH",
            "checkpoint_path": str(model_path),
            "checkpoint_sha256": sha256_file(model_path),
            "checkpoint_state_hash": evaluation["checkpoint_state_hash"],
            "prediction_path": str(prediction_path),
            "prediction_sha256": sha256_file(prediction_path),
            "fit_contract_hash": evaluation["fit_contract_hash"],
            "outgoing_selected_ids_hash": stable_hash(outgoing),
            "acquisition_contract_hash": None if acquisition_receipt is None else stable_hash(acquisition_receipt),
            "source_round_freeze_sha256": sha256_file(source_anchor(context.seed)["round_freeze"]) if round_index == SOURCE_ROUND else None,
            "test_truth_access_count": 0,
        }
        _write_json_once(directory / "round_freeze.json", freeze)
        if round_index > SOURCE_ROUND:
            new_round_hashes.append(sha256_file(directory / "round_freeze.json"))
        if round_index == FINAL_ROUND:
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
            raise RuntimeError("CW extension selected an ID twice across source and continuation")
        store.freeze_acquisitions(selected_all)
        new_truth = store.reveal(outgoing, "after_acquisition_fit")
        truth = np.vstack([truth, new_truth])
        labeled, unlabeled, incoming = next_labeled, next_unlabeled, outgoing

    if len(labeled) != FINAL_ACTIVE_LABELS or len(selected_all) != FINAL_ACTIVE_LABELS - 333:
        raise RuntimeError("CW extension did not hard-stop exactly at 525")
    method_root = context.runtime / METHOD
    _write_csv_once(method_root / "fit_audit.csv", pd.DataFrame(fit_rows))
    _write_csv_once(method_root / "label_access_audit.csv", pd.DataFrame(store.audit))
    trajectory = {
        "status": "FROZEN_BEFORE_EXTENSION_TEST_TRUTH",
        "outer_seed": context.seed,
        "method": METHOD,
        "source_active_labels": 429,
        "new_round_points": 3,
        "new_acquisition_rounds": 3,
        "final_active_labels": len(labeled),
        "final_unlabeled_rows": len(unlabeled),
        "all_selected_ids_hash": stable_hash(selected_all),
        "new_round_freeze_sha256": new_round_hashes,
        "source_round_freeze_sha256": sha256_file(source_anchor(context.seed)["round_freeze"]),
        "test_truth_access_count": 0,
    }
    _write_json_once(method_root / "trajectory_freeze.json", trajectory)
    return trajectory


def execute_seed(seed: int, study: Path = STUDY) -> dict:
    if int(seed) not in SEEDS:
        raise ValueError(f"extension seed must be one of {SEEDS}")
    validate_prepared(study)
    placeholder = _json(Path(study) / "global_pre_test_freeze.json")
    if placeholder.get("status") != "PENDING_CONTINUATIONS":
        raise RuntimeError("CW extension execution closed after global freeze")
    torch.set_num_threads(2)
    context = CWExtensionContext(int(seed), Path(study))
    trajectory = run_continuation(context)
    print(json.dumps({"seed": int(seed), "method_frozen": METHOD, "final_active_labels": 525}), flush=True)
    return {"status": "SEED_CONTINUATION_FROZEN", "seed": int(seed), "trajectory": trajectory}


def finalize_pre_test(study: Path = STUDY) -> dict:
    validate_prepared(study)
    entries = {}
    anchors = {}
    for seed in SEEDS:
        root = Path(study) / "runtime" / f"seed_{seed}" / METHOD
        trajectory = _json(root / "trajectory_freeze.json")
        if trajectory.get("final_active_labels") != FINAL_ACTIVE_LABELS or trajectory.get("test_truth_access_count") != 0:
            raise RuntimeError("CW extension trajectory not frozen at 525")
        source_path = root / "round_03/round_freeze.json"
        source_record = _json(source_path)
        if source_record.get("status") != "SOURCE_ANCHOR_FROZEN":
            raise RuntimeError("CW source anchor receipt invalid")
        anchors[f"seed_{seed}/round_03"] = {
            "outer_seed": seed,
            "active_label_count": 429,
            "source_round_freeze_sha256": source_record["source_round_freeze_sha256"],
            "extension_anchor_sha256": sha256_file(source_path),
        }
        for round_index in NEW_ACQUISITION_ROUNDS:
            path = root / f"round_{round_index:02d}/round_freeze.json"
            record = _json(path)
            if record.get("status") != "FROZEN_BEFORE_EXTENSION_TEST_TRUTH" or record.get("test_truth_access_count") != 0:
                raise RuntimeError("new CW extension round not frozen")
            if sha256_file(Path(record["checkpoint_path"])) != record["checkpoint_sha256"]:
                raise RuntimeError("CW extension checkpoint hash drift")
            if sha256_file(Path(record["prediction_path"])) != record["prediction_sha256"]:
                raise RuntimeError("CW extension prediction hash drift")
            entries[f"seed_{seed}/round_{round_index:02d}"] = {
                "outer_seed": seed,
                "method": METHOD,
                "round": round_index,
                "active_label_count": ACTIVE_LABEL_BUDGETS[round_index],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "checkpoint_state_hash": record["checkpoint_state_hash"],
                "prediction_sha256": record["prediction_sha256"],
                "round_freeze_sha256": sha256_file(path),
            }
    if len(entries) != len(SEEDS) * len(NEW_ACQUISITION_ROUNDS):
        raise RuntimeError("CW extension global freeze matrix incomplete")
    freeze = {
        "status": "FROZEN_BEFORE_EXTENSION_TEST_TRUTH",
        "source_anchors": anchors,
        "new_entries": entries,
        "new_frozen_prediction_points": len(entries),
        "hard_stop_active_labels": FINAL_ACTIVE_LABELS,
        "other_methods_trained": [],
        "test_truth_access_count": 0,
    }
    atomic_json(Path(study) / "global_pre_test_freeze.json", freeze)
    return freeze


def reveal_and_report(study: Path = STUDY) -> dict:
    from .benchmark_reporting import metric_row
    from .cw_lcmd_extension_reporting import write_report

    freeze = _json(Path(study) / "global_pre_test_freeze.json")
    if (freeze.get("status") != "FROZEN_BEFORE_EXTENSION_TEST_TRUTH"
            or len(freeze.get("new_entries", {})) != len(SEEDS) * len(NEW_ACQUISITION_ROUNDS)):
        raise RuntimeError("extension test reveal requires complete global freeze")
    rows = []
    access_rows = []
    for seed in SEEDS:
        context = CWExtensionContext(seed, Path(study), reuse_frozen_protocol_hash=True)
        final_state = pd.read_csv(Path(study) / "runtime" / f"seed_{seed}" / METHOD / "round_06/state.csv")
        final_ids = final_state.loc[final_state.role.eq("labeled"), "sample_id"].astype(str).tolist()
        selected = [value for value in final_ids if value not in set(context.ids(context.roles["l0"]))]
        store = RestrictedLabelStore(SOURCE_DATA, context.partition)
        store.freeze_acquisitions(selected)
        store.freeze_predictions()
        test_ids = context.ids(context.roles["test"])
        truth = store.reveal(test_ids, "final_test_evaluation")
        access_rows.extend({"outer_seed": seed, "method": METHOD, **item} for item in store.audit)
        for round_index in NEW_ACQUISITION_ROUNDS:
            record = _json(Path(study) / "runtime" / f"seed_{seed}" / METHOD / f"round_{round_index:02d}/round_freeze.json")
            prediction = pd.read_csv(record["prediction_path"])
            if prediction.sample_id.astype(str).tolist() != test_ids:
                raise RuntimeError("extension prediction/test identity mismatch")
            rows.append({
                "outer_seed": seed,
                "method": METHOD,
                "round": round_index,
                "active_label_count": ACTIVE_LABEL_BUDGETS[round_index],
                **metric_row(truth, prediction.drop(columns="sample_id").to_numpy(float), context.preprocessing["target_scales"]),
            })
    result = write_report(Path(study), pd.DataFrame(rows), pd.DataFrame(access_rows))
    # Persist the post-reveal state so the study no longer advertises a
    # pre-test freeze after its authorized developmental evaluation.
    freeze["status"] = "COMPLETE_DEVELOPMENTAL_CONTINUATION"
    freeze["test_truth_access_count"] = len(SEEDS)
    atomic_json(Path(study) / "global_pre_test_freeze.json", freeze)
    return result
