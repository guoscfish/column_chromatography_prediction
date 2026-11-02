"""Resumable pure-CW continuation from exact frozen L653 to L1005."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..artifacts import sha256_file
from ..resources import SOURCE_DATA
from ..schemas.conditions import ConditionNormalization
from ..training.predictor import atomic_json
from .benchmark_protocol import load_features
from .cw_lcmd_extension_runner import CWExtensionContext, _acquire, _write_csv_once, _write_json_once
from .cw_lcmd_to_1005_study import (
    ALL_BUDGETS, FINAL_ACTIVE_LABELS, FINAL_ROUND, METHOD, NEW_ROUNDS, SEEDS,
    SOURCE_ROUND, STUDY, protocol_record, source_anchor, source_root, split_path,
    validate_prepared,
)
from .gradient_transforms import center_width_transform
from .protocol import RestrictedLabelStore, ids_hash, stable_hash, validate_row_protocol
from .sequential_acquisition import validate_trajectory_transition


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


class CW1005Context(CWExtensionContext):
    def __init__(self, seed: int, study: Path = STUDY, *, reuse_frozen_protocol_hash: bool = False):
        self.seed = int(seed); self.study = Path(study)
        self.runtime = self.study / "runtime" / f"seed_{self.seed}"; self.runtime.mkdir(parents=True, exist_ok=True)
        self.partition = pd.read_csv(split_path(self.seed)); validate_row_protocol(self.partition)
        self.data = load_features(SOURCE_DATA)
        if self.partition.sample_id.astype(str).tolist() != self.data.sample_id.astype(str).tolist():
            raise RuntimeError("CW-to-1005 split/source order drift")
        self.roles = {role: self.partition.loc[self.partition.role.eq(role), "canonical_index"].to_numpy(int)
                      for role in ("l0", "u0", "validation", "test")}
        source_context = read_json(source_anchor(self.seed)["context"])
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
            raise RuntimeError("Center/Width transform differs from frozen CW source")
        self.protocol_hash = stable_hash(protocol_record())
        payload = {
            "study": self.study.name, "protocol_hash": self.protocol_hash, "outer_seed": self.seed,
            "split_hash": stable_hash(self.partition.to_dict("list")),
            "l0_ids_hash": ids_hash(self.ids(self.roles["l0"])), "u0_ids_hash": ids_hash(self.ids(self.roles["u0"])),
            "validation_ids_hash": ids_hash(self.ids(self.roles["validation"])),
            "test_ids_hash": ids_hash(self.ids(self.roles["test"])), "normalization": asdict(self.normalization),
            "preprocessing": self.preprocessing, "center_width_transform": self.cw_audit,
            "source_context_sha256": sha256_file(source_anchor(self.seed)["context"]), "test_truth_access_count": 0,
        }
        path = self.runtime / "context.json"
        if reuse_frozen_protocol_hash and path.exists():
            frozen = read_json(path)
            if {k: v for k, v in frozen.items() if k != "protocol_hash"} != {k: v for k, v in payload.items() if k != "protocol_hash"}:
                raise RuntimeError(f"refusing mismatched frozen context: {path}")
            self.protocol_hash = frozen["protocol_hash"]; payload["protocol_hash"] = self.protocol_hash
        _write_json_once(path, payload)


def _source_state(context: CW1005Context) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    paths = source_anchor(context.seed); state = pd.read_csv(paths["state"])
    labeled = state.loc[state.role.eq("labeled"), "canonical_index"].to_numpy(int)
    unlabeled = state.loc[state.role.eq("unlabeled"), "canonical_index"].to_numpy(int)
    selected = context.ids(labeled)[len(context.roles["l0"]):]
    if len(labeled) != 653 or len(selected) != 320 or context.ids(labeled) != context.ids(context.roles["l0"]) + selected:
        raise RuntimeError("source is not the exact ordered pure-CW L653 state")
    return labeled, unlabeled, selected, read_json(paths["round_freeze"])


def run_continuation(context: CW1005Context) -> dict:
    labeled, unlabeled, selected_all, source_freeze = _source_state(context)
    store = RestrictedLabelStore(SOURCE_DATA, context.partition)
    if not np.array_equal(store.reveal(context.ids(context.roles["l0"]), "initial_fit"), context.l0_truth):
        raise RuntimeError("L0 truth drift")
    if not np.array_equal(store.reveal(context.ids(context.roles["validation"]), "initial_fit"), context.validation_truth):
        raise RuntimeError("validation truth drift")
    store.freeze_acquisitions(selected_all)
    truth = np.vstack([context.l0_truth, store.reveal(selected_all, "after_acquisition_fit")])
    if truth.shape != (653, 2): raise RuntimeError("failed to reconstruct exact observed L653 labels")
    incoming = pd.read_csv(source_root(context.seed) / "round_10/selected_batch.csv").sample_id.astype(str).tolist()
    fit_rows, new_round_hashes = [], []

    for round_index in range(SOURCE_ROUND, FINAL_ROUND + 1):
        if len(labeled) != ALL_BUDGETS[round_index]: raise RuntimeError("CW-to-1005 active-label schedule drift")
        directory = context.runtime / METHOD / f"round_{round_index:02d}"; directory.mkdir(parents=True, exist_ok=True)
        input_contract = {
            "study": context.study.name, "protocol_hash": context.protocol_hash, "outer_seed": context.seed,
            "method": METHOD, "round": round_index, "active_label_count": len(labeled),
            "unlabeled_count": len(unlabeled), "L_t_ids_hash": ids_hash(context.ids(labeled)),
            "U_t_ids_hash": ids_hash(context.ids(unlabeled)), "incoming_selected_ids_hash": stable_hash(incoming),
            "source_CW_trajectory": True, "test_truth_access_count": 0,
        }
        _write_json_once(directory / "input_contract.json", input_contract)
        _write_csv_once(directory / "state.csv", pd.DataFrame({
            "role": ["labeled"] * len(labeled) + ["unlabeled"] * len(unlabeled),
            "position": list(range(len(labeled))) + list(range(len(unlabeled))),
            "canonical_index": np.r_[labeled, unlabeled], "sample_id": context.ids(np.r_[labeled, unlabeled]),
        }))
        _write_csv_once(directory / "selected_batch.csv", pd.DataFrame({
            "outer_seed": [context.seed] * len(incoming), "method": [METHOD] * len(incoming),
            "round": [round_index] * len(incoming), "selection_order": np.arange(len(incoming)), "sample_id": incoming,
        }))
        if round_index == SOURCE_ROUND:
            model_path = Path(source_freeze["checkpoint_path"]); prediction_path = Path(source_freeze["prediction_path"])
            evaluation = {"reuse_status": "reused_exact_source_CW_L653", **read_json(model_path.with_name("fit_audit.json"))}
        else:
            evaluation = context.fit(round_index, labeled, truth, directory / "model")
            model_path = directory / "model/best.pt"; prediction_path = directory / "model/predictions.csv.gz"
        fit_rows.append({"outer_seed": context.seed, "method": METHOD, "round": round_index, **evaluation})
        outgoing, acquisition_receipt = [], None
        if round_index < FINAL_ROUND:
            outgoing, acquisition_receipt = _acquire(context, round_index, labeled, unlabeled, model_path, directory)
        freeze = {
            "input": input_contract,
            "status": "SOURCE_ANCHOR_FROZEN" if round_index == SOURCE_ROUND else "FROZEN_BEFORE_EXTENSION_TEST_TRUTH",
            "checkpoint_path": str(model_path), "checkpoint_sha256": sha256_file(model_path),
            "checkpoint_state_hash": evaluation["checkpoint_state_hash"],
            "prediction_path": str(prediction_path), "prediction_sha256": sha256_file(prediction_path),
            "fit_contract_hash": evaluation["fit_contract_hash"], "outgoing_selected_ids_hash": stable_hash(outgoing),
            "acquisition_contract_hash": None if acquisition_receipt is None else stable_hash(acquisition_receipt),
            "source_round_freeze_sha256": sha256_file(source_anchor(context.seed)["round_freeze"]) if round_index == SOURCE_ROUND else None,
            "test_truth_access_count": 0,
        }
        _write_json_once(directory / "round_freeze.json", freeze)
        if round_index > SOURCE_ROUND: new_round_hashes.append(sha256_file(directory / "round_freeze.json"))
        if round_index == FINAL_ROUND: break
        old_l, old_u = context.ids(labeled), context.ids(unlabeled)
        positions = {value: index for index, value in enumerate(old_u)}
        selected_indices = unlabeled[np.asarray([positions[value] for value in outgoing], dtype=int)]
        selected_set = set(outgoing)
        next_unlabeled = np.asarray([index for index, value in zip(unlabeled, old_u) if value not in selected_set])
        next_labeled = np.r_[labeled, selected_indices]
        validate_trajectory_transition(old_l, old_u, outgoing, context.ids(next_labeled), context.ids(next_unlabeled))
        selected_all.extend(outgoing)
        if len(selected_all) != len(set(selected_all)): raise RuntimeError("CW selected an ID twice")
        store.freeze_acquisitions(selected_all); truth = np.vstack([truth, store.reveal(outgoing, "after_acquisition_fit")])
        labeled, unlabeled, incoming = next_labeled, next_unlabeled, outgoing

    if len(labeled) != FINAL_ACTIVE_LABELS or len(selected_all) != FINAL_ACTIVE_LABELS - 333:
        raise RuntimeError("CW continuation did not hard-stop exactly at 1005")
    root = context.runtime / METHOD
    _write_csv_once(root / "fit_audit.csv", pd.DataFrame(fit_rows)); _write_csv_once(root / "label_access_audit.csv", pd.DataFrame(store.audit))
    trajectory = {
        "status": "FROZEN_BEFORE_EXTENSION_TEST_TRUTH", "outer_seed": context.seed, "method": METHOD,
        "source_active_labels": 653, "new_round_points": 11, "new_acquisition_rounds": 11,
        "new_evaluation_fits": 11, "final_active_labels": len(labeled), "final_unlabeled_rows": len(unlabeled),
        "all_selected_ids_hash": stable_hash(selected_all), "new_round_freeze_sha256": new_round_hashes,
        "source_round_freeze_sha256": sha256_file(source_anchor(context.seed)["round_freeze"]),
        "test_truth_access_count": 0,
    }
    _write_json_once(root / "trajectory_freeze.json", trajectory); return trajectory


def execute_seed(seed: int, study: Path = STUDY) -> dict:
    if int(seed) not in SEEDS: raise ValueError(f"seed must be one of {SEEDS}")
    validate_prepared(study)
    if read_json(Path(study) / "global_pre_test_freeze.json").get("status") != "PENDING_CONTINUATIONS":
        raise RuntimeError("execution closed after global freeze")
    torch.set_num_threads(2); trajectory = run_continuation(CW1005Context(int(seed), Path(study)))
    print(json.dumps({"seed": int(seed), "method_frozen": METHOD, "final_active_labels": 1005}), flush=True)
    return {"status": "SEED_CONTINUATION_FROZEN", "seed": int(seed), "trajectory": trajectory}


def finalize_pre_test(study: Path = STUDY) -> dict:
    validate_prepared(study); entries, anchors = {}, {}
    for seed in SEEDS:
        root = Path(study) / f"runtime/seed_{seed}/{METHOD}"
        trajectory = read_json(root / "trajectory_freeze.json")
        if trajectory.get("final_active_labels") != 1005 or trajectory.get("test_truth_access_count") != 0:
            raise RuntimeError("trajectory not frozen at 1005")
        source_path = root / "round_10/round_freeze.json"; source_record = read_json(source_path)
        if source_record.get("status") != "SOURCE_ANCHOR_FROZEN": raise RuntimeError("invalid L653 anchor receipt")
        anchors[f"seed_{seed}/round_10"] = {
            "outer_seed": seed, "active_label_count": 653,
            "source_round_freeze_sha256": source_record["source_round_freeze_sha256"],
            "extension_anchor_sha256": sha256_file(source_path),
        }
        for round_index in NEW_ROUNDS:
            path = root / f"round_{round_index:02d}/round_freeze.json"; record = read_json(path)
            if record.get("status") != "FROZEN_BEFORE_EXTENSION_TEST_TRUTH" or record.get("test_truth_access_count") != 0:
                raise RuntimeError("new round not frozen behind test barrier")
            if sha256_file(Path(record["checkpoint_path"])) != record["checkpoint_sha256"]: raise RuntimeError("checkpoint hash drift")
            if sha256_file(Path(record["prediction_path"])) != record["prediction_sha256"]: raise RuntimeError("prediction hash drift")
            entries[f"seed_{seed}/round_{round_index:02d}"] = {
                "outer_seed": seed, "method": METHOD, "round": round_index,
                "active_label_count": ALL_BUDGETS[round_index], "checkpoint_sha256": record["checkpoint_sha256"],
                "checkpoint_state_hash": record["checkpoint_state_hash"], "prediction_sha256": record["prediction_sha256"],
                "round_freeze_sha256": sha256_file(path),
            }
    if len(entries) != 22: raise RuntimeError("global freeze requires exactly 22 new checkpoint/prediction pairs")
    freeze = {
        "status": "FROZEN_BEFORE_EXTENSION_TEST_TRUTH", "source_anchors": anchors, "new_entries": entries,
        "new_frozen_prediction_points": 22, "hard_stop_active_labels": 1005,
        "acquisition_at_1005": False, "other_methods_trained": [], "test_truth_access_count": 0,
    }
    atomic_json(Path(study) / "global_pre_test_freeze.json", freeze); return freeze


def reveal_and_report(study: Path = STUDY) -> dict:
    from .benchmark_reporting import metric_row
    from .cw_lcmd_to_1005_reporting import refresh_manifest, write_report
    freeze = read_json(Path(study) / "global_pre_test_freeze.json")
    if freeze.get("status") != "FROZEN_BEFORE_EXTENSION_TEST_TRUTH" or len(freeze.get("new_entries", {})) != 22:
        raise RuntimeError("test reveal requires the complete 22-point global freeze")
    rows, access_rows = [], []
    for seed in SEEDS:
        context = CW1005Context(seed, Path(study), reuse_frozen_protocol_hash=True)
        final_state = pd.read_csv(Path(study) / f"runtime/seed_{seed}/{METHOD}/round_21/state.csv")
        final_ids = final_state.loc[final_state.role.eq("labeled"), "sample_id"].astype(str).tolist()
        selected = final_ids[len(context.roles["l0"]):]
        store = RestrictedLabelStore(SOURCE_DATA, context.partition); store.freeze_acquisitions(selected); store.freeze_predictions()
        test_ids = context.ids(context.roles["test"]); truth = store.reveal(test_ids, "final_test_evaluation")
        access_rows.extend({"outer_seed": seed, "method": METHOD, **item} for item in store.audit)
        for round_index in NEW_ROUNDS:
            record = read_json(Path(study) / f"runtime/seed_{seed}/{METHOD}/round_{round_index:02d}/round_freeze.json")
            prediction = pd.read_csv(record["prediction_path"])
            if prediction.sample_id.astype(str).tolist() != test_ids: raise RuntimeError("prediction/test identity mismatch")
            rows.append({"outer_seed": seed, "method": METHOD, "round": round_index,
                         "active_label_count": ALL_BUDGETS[round_index],
                         **metric_row(truth, prediction.drop(columns="sample_id").to_numpy(float), context.preprocessing["target_scales"])})
    result = write_report(Path(study), pd.DataFrame(rows), pd.DataFrame(access_rows))
    freeze["status"] = "COMPLETE_DEVELOPMENTAL_CONTINUATION"; freeze["test_truth_access_count"] = len(SEEDS)
    atomic_json(Path(study) / "global_pre_test_freeze.json", freeze); refresh_manifest(Path(study))
    return result
