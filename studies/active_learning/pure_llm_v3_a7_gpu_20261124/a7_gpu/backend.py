"""Qualified public 4g resources and an independent QGeoGNN-V2 CUDA fit path."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from unittest.mock import patch
from uuid import uuid4

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

from src.qgeognn_al.active_learning_v2 import runner
from src.qgeognn_al.active_learning_v2.benchmark_protocol import (
    TRAINING_CONFIG, initialization_seed, load_features, seed_config,
)
from src.qgeognn_al.active_learning_v2.protocol import RestrictedLabelStore, ids_hash, stable_hash, validate_row_protocol
from src.qgeognn_al.active_learning_v2.scientist_full_delivery import encode_catalog
from src.qgeognn_al.active_learning_v2.scientist_selector import CONDITIONS, PREDICTIONS, ReadOnlyCatalog
from src.qgeognn_al.evaluation.point import point_metrics
from src.qgeognn_al.models import build_predictor
from src.qgeognn_al.models.qgeognn_v2 import validate_predictor_checkpoint
from src.qgeognn_al.schemas.conditions import ConditionNormalization
from src.qgeognn_al.training.predictor import seed_everything
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.frozen_state import FrozenState, SEEDS, SPLITS
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import round_lock

from .cuda_runtime import configure_device, cuda_adapter, cuda_contract, qualified_packages


SEQUENTIAL_STUDY = SPLITS.parent
CANONICAL = Path("experiments/e0_4g_baseline/canonical_4g.csv")
GRAPH_CACHE = Path("experiments/e0_4g_baseline/graph_cache_4g.pt")
METHOD = "pure_llm_v3"
INITIAL_BUDGET = 333
BATCH_SIZE = 32
TOTAL_ROUNDS = 36
REFERENCE_PREREGISTRATION = "a419ea75e7894004aef393cb3734a7ae5e1356a1"
REFERENCE_KIND = "formal_cuda_three_methods"
MODEL_FILES = {
    "application/QGeoGNN.py",
    "application/utils.py",
    "src/qgeognn_al/active_learning_v2/benchmark_protocol.py",
    "src/qgeognn_al/active_learning_v2/runner.py",
    "src/qgeognn_al/data.py",
    "src/qgeognn_al/evaluation/point.py",
    "src/qgeognn_al/input_schema.py",
    "src/qgeognn_al/models/qgeognn_v2.py",
    "src/qgeognn_al/schemas/conditions.py",
    "src/qgeognn_al/training/predictor.py",
}
PREDICTION_COLUMNS = [f"{target}_{q}" for target in ("V1", "V2") for q in ("q10", "q50", "q90")]
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _verify(record, expected, name):
    mismatches = sorted(key for key, value in expected.items() if record.get(key) != value)
    if mismatches:
        raise ValueError(f"{name} differs at {', '.join(mismatches)}")


def _publish_graphs(path, atom, angle):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save((atom, angle), stream)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _copy_checkpoint(source, destination, expected_sha256):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with source.open("rb") as original:
                shutil.copyfileobj(original, stream)
            stream.flush()
            os.fsync(stream.fileno())
            if file_digest(temporary) != expected_sha256:
                raise ValueError("L333 source checkpoint changed while copying")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def _validate_graphs(data, atom, angle):
    if len(atom) != len(data) or len(angle) != len(data):
        raise ValueError("scrubbed graph coverage differs from canonical feature rows")
    for index, graph in enumerate(atom):
        if graph.canonical_position.item() != index or tuple(graph.y.shape) != (1, 2) or torch.count_nonzero(graph.y):
            raise ValueError(f"scrubbed graph position or sentinel mismatch at position {index}")


@contextmanager
def _preserve_completed_training(fit_dir, contract):
    """Seal the completed optimizer work before inference and downstream metrics."""
    path = fit_dir / "training_complete.json"
    original = runner.train_source

    def training_once(model, atom, angle, train_indices, validation_indices, preprocessing, config, checkpoint,
                      **kwargs):
        if path.exists():
            record = read(path)
            _verify(record, {"contract": contract, "checkpoint_sha256": file_digest(checkpoint)},
                    "completed training receipt")
            return record["history"], record["best_epoch"]
        history, best = original(model, atom, angle, train_indices, validation_indices,
                                 preprocessing, config, checkpoint, **kwargs)
        once(path, {"contract": contract, "checkpoint_sha256": file_digest(checkpoint),
                    "history": history, "best_epoch": best})
        return history, best

    with patch.object(runner, "train_source", training_once):
        yield


class TrainingBackend:
    """Provide legal observations, full-U predictions, and frozen-config CUDA fits for one seed."""

    def __init__(self, root, cache_dir, seed, *, gpu_slot=nullcontext):
        if seed not in SEEDS:
            raise ValueError("the seed is outside the formal A7 cohort")
        self.root = Path(root)
        self.cache_dir = Path(cache_dir)
        self.seed = seed
        self.gpu_slot = gpu_slot
        self.source = self.root / CANONICAL
        self.graph_source = self.root / GRAPH_CACHE
        self.source_sha256 = file_digest(self.source)
        self.graph_sha256 = file_digest(self.graph_source)
        protocol = read(self.root / SEQUENTIAL_STUDY / "protocol.json")
        _verify(protocol, {
            "confirmation_seeds": list(SEEDS), "training": TRAINING_CONFIG, "batch_size": BATCH_SIZE,
            "initial_active_labels": INITIAL_BUDGET, "initial_pool_rows": 2997, "validation_rows": 416,
            "test_rows": 417, "outer_train_rows": 3330, "source_sha256": self.source_sha256,
            "graph_cache_sha256": self.graph_sha256,
            "target_scales": "original_L0_population_std_ddof_0_fixed_for_trajectory",
        }, "public Sequential B32 protocol")
        split_manifest = read(self.root / SPLITS / "split_manifest.json")
        _verify(split_manifest, {"source_sha256": self.source_sha256, "protocol_hash": stable_hash(protocol)},
                "split manifest provenance")
        split_record = next(record for record in split_manifest["splits"] if record["outer_seed"] == seed)
        split_file = self.root / SPLITS / f"row_seed_{seed}.csv"
        self.split_sha256 = file_digest(split_file)
        _verify(split_record, {"sha256": self.split_sha256,
            "role_counts": {"l0": 333, "u0": 2997, "validation": 416, "test": 417}}, "split file")
        self.partition = pd.read_csv(split_file)
        validate_row_protocol(self.partition)
        self.data = load_features(self.source)
        if (set(self.partition["outer_seed"]) != {seed} or
                self.partition["canonical_index"].tolist() != list(range(len(self.data))) or
                not self.partition[["sample_id", "canonical_smiles"]].equals(
                    self.data[["sample_id", "canonical_smiles"]])):
            raise ValueError("canonical graph-row and split identities disagree")
        self.roles = {role: tuple(self.partition.loc[self.partition.role.eq(role), "sample_id"])
                      for role in split_record["role_counts"]}
        for role, sample_ids in self.roles.items():
            if ids_hash(sample_ids) != split_record[f"{role}_ids_hash"]:
                raise ValueError(f"frozen role identities differ for {role}")
        self.l0_ids, self.u0_ids = self.roles["l0"], self.roles["u0"]
        self.indices = dict(zip(self.data.sample_id, self.partition.canonical_index, strict=True))
        self.u0_set = frozenset(self.u0_ids)
        self.training_config = {**seed_config(seed), "split_sha256": stable_hash(self.partition.to_dict("list"))}
        checked_paths = MODEL_FILES | {
            (SPLITS / f"row_seed_{seed}.csv").as_posix(), (SEQUENTIAL_STUDY / "protocol.json").as_posix(),
            (SPLITS / "split_manifest.json").as_posix(),
        }
        self.code_and_split_hashes = {p: file_digest(self.root / p) for p in sorted(checked_paths)}
        torch.set_num_threads(self.training_config["cpu_threads"])
        with round_lock(self.cache_dir):
            self._prepare_cache()

    def _prepare_cache(self):
        store = RestrictedLabelStore(self.source, self.partition)
        self.l0_truth = store.reveal(self.l0_ids, "initial_fit")
        self.validation_truth = store.reveal(self.roles["validation"], "initial_fit")
        self.initial_label_audit = store.audit
        context = {
            "seed": self.seed, "source_sha256": self.source_sha256, "graph_source_sha256": self.graph_sha256,
            "split_sha256": self.split_sha256, "training_config": self.training_config,
            "ordered_canonical_ids_sha256": stable_hash(self.data.sample_id.tolist()),
            "ordered_canonical_structures_sha256": stable_hash(self.data.canonical_smiles.tolist()),
            "role_id_hashes": {role: ids_hash(ids) for role, ids in self.roles.items()},
            "training_source_sha256": self.code_and_split_hashes,
            "label_access": self.initial_label_audit,
        }
        graph_path = self.cache_dir / "scrubbed_graphs.pt"
        scaler_path = self.cache_dir / "scaler.json"
        receipt_path = self.cache_dir / "resource_receipt.json"
        if receipt_path.exists():
            receipt = read(receipt_path)
            _verify(receipt, {"context": context, "graph_sha256": file_digest(graph_path),
                              "scaler_sha256": file_digest(scaler_path)}, "prepared resource receipt")
            if read(scaler_path) != receipt["preprocessing"]["scaler"]:
                raise ValueError("scrubbed graph scaler differs from preprocessing")
            self.normalization = ConditionNormalization(**receipt["normalization"])
            self.normalization.validate()
            self.preprocessing = receipt["preprocessing"]
            self.atom, self.angle = torch.load(graph_path, map_location="cpu", weights_only=False)
        else:
            if graph_path.exists():
                archive = graph_path.with_name(f"unsealed_graphs_{uuid4().hex}.pt")
                graph_path.rename(archive)
            raw_graphs = torch.load(self.graph_source, map_location="cpu", weights_only=False)
            if not set(self.data.canonical_smiles) <= raw_graphs.keys():
                raise ValueError("raw conformer graph cache does not cover canonical molecules")
            self.normalization, self.preprocessing = runner.fit_al_preprocessing(
                self.data, raw_graphs, self._positions(self.l0_ids + self.u0_ids), self.l0_ids,
                self.l0_truth, scaler_path,
            )
            self.atom, self.angle = runner.make_label_scrubbed_graphs(self.data, raw_graphs, self.preprocessing["scaler"])
            _validate_graphs(self.data, self.atom, self.angle)
            _publish_graphs(graph_path, self.atom, self.angle)
            receipt = {
                "context": context, "graph_sha256": file_digest(graph_path), "scaler_sha256": file_digest(scaler_path),
                "normalization": asdict(self.normalization), "preprocessing": self.preprocessing,
            }
            once(receipt_path, receipt)
        _validate_graphs(self.data, self.atom, self.angle)
        _verify(self.preprocessing, {
            "target_scale_fit_role": "l0_labels_only", "target_scale_fit_ids_hash": ids_hash(self.l0_ids),
            "target_scale_fit_rows": INITIAL_BUDGET, "test_rows_used": 0,
            "target_scales": dict(zip(("V1", "V2"), map(float, self.l0_truth.std(axis=0, ddof=0)), strict=True)),
            "fit_ids_hash": ids_hash(self.l0_ids + self.u0_ids), "fit_rows": len(self.l0_ids + self.u0_ids),
        }, "fixed L333 and label-free covariate preprocessing")
        if self.normalization.eluent_scaler_sha256 != receipt["scaler_sha256"]:
            raise ValueError("normalization is not bound to the prepared scaler")
        seed_everything(initialization_seed(self.seed))
        self.initialization_hash = runner.state_dict_hash(build_predictor(self.normalization))
        self.cache_receipt = receipt
        self.cache_receipt_sha256 = file_digest(receipt_path)

    def _positions(self, ids):
        return np.asarray([self.indices[value] for value in ids], dtype=int)

    @contextmanager
    def _cuda(self):
        packages = qualified_packages(self.root)
        with self.gpu_slot():
            device = configure_device()
            environment = {**cuda_contract(device), "packages": packages}
            with cuda_adapter(device):
                yield environment

    def _anchor(self, anchor_run, environment):
        run_path = Path(anchor_run) / "run_contract.json"
        entry = read(run_path)
        method = entry["method"]
        if method not in {"random", "hybrid", "cw_lcmd"}:
            raise ValueError("L333 anchor is not a qualified numeric CUDA run")
        _verify(entry, {
            "kind": REFERENCE_KIND, "outer_seed": self.seed, "canonical_source_sha256": self.source_sha256,
            "canonical_graph_cache_sha256": self.graph_sha256, "training": TRAINING_CONFIG,
            "reference_preregistration_commit": REFERENCE_PREREGISTRATION,
            "reference_initial_L": "common_original_L333_per_seed", "batch_size": BATCH_SIZE,
            "selection_rounds": TOTAL_ROUNDS, "budgets": [INITIAL_BUDGET + BATCH_SIZE * r for r in range(TOTAL_ROUNDS + 1)],
            **environment,
        }, "numeric CUDA L333 anchor run")
        _verify(entry["code_and_config_sha256"], self.code_and_split_hashes, "numeric CUDA shared training source")
        round_dir = Path(anchor_run) / "rounds" / method / "round_00"
        freeze_path = round_dir / "contract.json"
        freeze = read(freeze_path)
        _verify(freeze["input"], {
            "outer_seed": self.seed, "method": method, "round": 0, "active_label_count": INITIAL_BUDGET,
            "unlabeled_count": len(self.u0_ids), "L_t_ids_hash": ids_hash(self.l0_ids), "U_t_ids_hash": ids_hash(self.u0_ids),
            "ordered_L_t_hash": stable_hash(list(self.l0_ids)), "ordered_U_t_hash": stable_hash(list(self.u0_ids)),
            "model_initialization_seed": initialization_seed(self.seed), "training_config": self.training_config,
            "endpoint_scales": self.preprocessing["target_scales"], "endpoint_scales_fit_ids_hash": ids_hash(self.l0_ids),
            "test_truth_access_count": 0,
        }, "numeric CUDA L333 ordered-state freeze")
        _verify(freeze, {"status": "FROZEN_BEFORE_TEST_TRUTH", "test_truth_access_count": 0,
                         "initialization_hash": self.initialization_hash}, "numeric CUDA L333 seal")
        checkpoint = round_dir / "model/best.pt"
        fit_file = round_dir / "model/fit_audit.json"
        checkpoint_digest = file_digest(checkpoint)
        _verify(freeze["files"], {"model/best.pt": checkpoint_digest, "model/fit_audit.json": file_digest(fit_file)},
                "numeric CUDA sealed L333 model")
        fit = read(fit_file)
        _verify(fit, {
            "fit_contract_hash": freeze["evaluation_fit_contract_hash"], "train_ids_hash": ids_hash(self.l0_ids),
            "validation_ids_hash": ids_hash(self.roles["validation"]), "train_rows": INITIAL_BUDGET,
            "validation_rows": len(self.roles["validation"]), "initialization_seed": initialization_seed(self.seed),
            "initialization_hash": self.initialization_hash, "checkpoint_sha256": checkpoint_digest,
            "checkpoint_state_hash": freeze["checkpoint_state_hash"], "test_labels_used_for_fit_or_checkpoint_selection": 0,
        }, "numeric CUDA L333 fit audit")
        if freeze["checkpoint_sha256"] != checkpoint_digest:
            raise ValueError("numeric freeze checkpoint digest drifted")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        validate_predictor_checkpoint(payload)
        _verify(payload, {"preprocessing": self.preprocessing, "normalization": asdict(self.normalization),
                          "training_config": self.training_config}, "numeric CUDA L333 checkpoint")
        provenance = {"kind": REFERENCE_KIND, "run_contract_sha256": file_digest(run_path),
                      "round_freeze_sha256": file_digest(freeze_path), "fit_audit_sha256": file_digest(fit_file),
                      "source_kind": "shared_numeric_L333_model_only"}
        return checkpoint, fit, provenance

    def _infer(self, model, ids):
        positions = self._positions(ids)
        predictions, returned = runner.predict_outputs(model, self.atom, self.angle, positions)
        if not np.array_equal(positions, returned):
            raise ValueError("checkpoint prediction order drifted")
        return predictions

    def initial_state(self, anchor_run, fit_dir, *, allow_fresh_fit=False):
        """Create the first packet from a reusable qualifying model or an authorized fresh fit."""
        fit_dir = Path(fit_dir)
        prediction_file = fit_dir / "predictions.csv.gz"
        receipt_path = fit_dir / "initial_state_receipt.json"
        if receipt_path.exists():
            receipt = read(receipt_path)
            _verify(receipt, {"seed": self.seed, "resources_sha256": self.cache_receipt_sha256,
                              "prediction_sha256": file_digest(prediction_file),
                              "checkpoint_sha256": file_digest(fit_dir / "best.pt"),
                              "validation_metrics_sha256": file_digest(fit_dir / "validation_metrics.json"),
                              "ordered_U_ids_sha256": stable_hash(list(self.u0_ids))}, "recorded L333 state")
            values = self._read_predictions(prediction_file, self.u0_ids)
            return self.state_from_predictions(values, receipt["checkpoint_sha256"], receipt=receipt)
        if anchor_run is None:
            if not allow_fresh_fit:
                raise FileNotFoundError("a qualifying existing CUDA L333 anchor is required for preflight")
            fit = self._fit(self.l0_ids, self.l0_truth, self.u0_ids, 0, None, fit_dir)
            source = {"source_kind": "fresh_authorized_L333", "fit_audit_sha256": file_digest(fit_dir / "fit_audit.json")}
            environment = read(fit_dir / "fit_device.json")
            values = self._read_predictions(prediction_file, self.u0_ids)
        else:
            with self._cuda() as environment:
                checkpoint, fit, source = self._anchor(anchor_run, environment)
                model = runner.load_predictor_checkpoint(checkpoint)
                if runner.state_dict_hash(model) != fit["checkpoint_state_hash"]:
                    raise ValueError("numeric CUDA checkpoint tensor digest disagrees with the L333 seal")
                values = self._infer(model, self.u0_ids)
                validation = self._infer(model, self.roles["validation"])
            metrics = point_metrics(self.validation_truth, validation, self.preprocessing["target_scales"])
            self._save_metrics(fit_dir, fit, metrics)
            runner.write_predictions(prediction_file, self.u0_ids, values)
            _copy_checkpoint(checkpoint, fit_dir / "best.pt", fit["checkpoint_sha256"])
        receipt = {
            "seed": self.seed, "resources_sha256": self.cache_receipt_sha256,
            "prediction_sha256": file_digest(prediction_file), "checkpoint_sha256": fit["checkpoint_sha256"],
            "ordered_U_ids_sha256": stable_hash(list(self.u0_ids)), "source": source,
            "device": environment, "validation_metrics_sha256": file_digest(fit_dir / "validation_metrics.json"),
        }
        once(receipt_path, receipt)
        return self.state_from_predictions(self._read_predictions(prediction_file, self.u0_ids),
                                           fit["checkpoint_sha256"], receipt=receipt)

    @staticmethod
    def _read_predictions(path, ids):
        table = pd.read_csv(path, float_precision="round_trip")
        if list(table.columns) != ["sample_id", *PREDICTION_COLUMNS] or table.sample_id.tolist() != list(ids):
            raise ValueError("prediction file differs from the complete ordered U")
        return table[PREDICTION_COLUMNS].to_numpy(dtype=float)

    def state_from_predictions(self, values, checkpoint, *, receipt=None):
        """Build the complete round-0 adapter packet from U predictions and authorized L333 observations."""
        values = np.asarray(values)
        if not isinstance(checkpoint, str) or not SHA256.fullmatch(checkpoint):
            raise ValueError("initial state requires a SHA256 checkpoint identity")
        if values.shape != (len(self.u0_ids), 6) or not np.isfinite(values).all():
            raise ValueError("initial predictions must contain all six finite outputs for every U row")

        def card(sample_id):
            row = self.data.iloc[self.indices[sample_id]]
            return {"id": sample_id, "smiles": row.canonical_smiles,
                    "conditions": {name: row[name].item() if hasattr(row[name], "item") else row[name]
                                   for name in CONDITIONS}}

        candidates = [
            {**card(sample_id), **dict(zip(PREDICTIONS, map(float, prediction), strict=True)),
             "pred_V1_ml": float(prediction[1]), "pred_V2_ml": float(prediction[4])}
            for sample_id, prediction in zip(self.u0_ids, values, strict=True)
        ]
        observed = [
            {**card(sample_id), "V1_ml": float(truth[0]), "V2_ml": float(truth[1]), "record_kind": "initial_L333"}
            for sample_id, truth in zip(self.l0_ids, self.l0_truth, strict=True)
        ]
        catalog = ReadOnlyCatalog(candidates, observed, [], salt=f"pure-llm-v3:{self.seed}:0")
        packet = {
            "seed": self.seed, "round": 0, "study_version": METHOD, "active_label_count": len(observed),
            "selection_count": BATCH_SIZE, "objective": "overall Row V1/V2 predictive accuracy after retraining",
            "validation_test_records": 0, "delivery": "all_candidates_one_request",
            "candidate_count": len(candidates), "observed_count": len(observed), **encode_catalog(catalog),
            "memory": {"previous_hypotheses": [], "previous_feedback_interpretation": "", "previous_batch_rationale": "",
                       "previous_selected_ids": [], "recent_outcomes": [], "high_error_measured_examples": [],
                       "memory_policy": "V3 host memory"},
        }
        packet["packet_sha256"] = digest(packet)
        provenance = {"split_sha256": self.split_sha256,
            "expected_candidates": sorted(self.u0_ids), "expected_observed": sorted(self.l0_ids),
            "resource_receipt_sha256": self.cache_receipt_sha256}
        if receipt is not None:
            provenance["initial_state_receipt_sha256"] = digest(receipt)
        return FrozenState(self.seed, 0, checkpoint, packet, [], [], provenance)

    def reveal_batch(self, prior_ids, batch_ids):
        prior = tuple(prior_ids)
        batch = tuple(batch_ids)
        if (len(batch) != BATCH_SIZE or len(set(prior + batch)) != len(prior + batch)
                or not set(prior + batch) <= self.u0_set or len(prior) % BATCH_SIZE != 0):
            raise ValueError("revealed batch must comprise 32 fresh IDs in the frozen U trajectory")
        store = RestrictedLabelStore(self.source, self.partition)
        store.freeze_acquisitions(prior + batch)
        return store.reveal(batch, "after_acquisition_fit"), store.audit[-1]

    def fit_next(self, *, labeled_ids, labeled_truth, unlabeled_ids, round_index, freeze_sha256, fit_dir):
        if not 1 <= round_index <= TOTAL_ROUNDS or not isinstance(freeze_sha256, str) or not SHA256.fullmatch(freeze_sha256):
            raise ValueError("formal post-acquisition fit requires a frozen acquisition digest")
        return self._fit(tuple(labeled_ids), labeled_truth, tuple(unlabeled_ids), round_index,
                         freeze_sha256, Path(fit_dir))

    def _fit(self, labeled, truth, unlabeled, round_index, freeze_sha256, fit_dir):
        acquired = labeled[INITIAL_BUDGET:]
        acquired_set = set(acquired)
        truth = np.asarray(truth, dtype=np.float32)
        if (labeled[:INITIAL_BUDGET] != self.l0_ids
                or len(acquired) != BATCH_SIZE * round_index or len(acquired) != len(acquired_set) or not acquired_set <= self.u0_set
                or unlabeled != tuple(value for value in self.u0_ids if value not in acquired_set)
                or truth.shape != (len(labeled), 2) or not np.array_equal(truth[:INITIAL_BUDGET], self.l0_truth)):
            raise ValueError("fit arrays differ from exact L333, acquisitions, or canonical U order")
        if not np.isfinite(truth).all():
            raise ValueError("fit contains non-finite labels")
        sealed_artifacts = [fit_dir / name for name in ("best.pt", "predictions.csv.gz", "fit_audit.json")]
        present = [path.exists() for path in sealed_artifacts]
        if any(present) and not all(present) and not (fit_dir / "training_complete.json").exists():
            raise RuntimeError("an interrupted fit must be retained, then retried in a new fit directory")
        train_contract = {
            "method": METHOD, "round": round_index, "outer_seed": self.seed, "device": "cuda",
            "train_sample_ids": list(labeled), "validation_sample_ids": list(self.roles["validation"]),
            "selection_freeze_sha256": freeze_sha256, "prediction_role": "current_U_t",
            "test_truth_access_count": 0, "resource_receipt_sha256": self.cache_receipt_sha256,
        }
        fit_device = fit_dir / "fit_device.json"
        if all(present) and not fit_device.exists():
            raise RuntimeError("sealed fit is missing its CUDA provenance")
        scope = nullcontext() if all(present) else self._cuda()
        with scope as environment:
            if environment is not None:
                once(fit_device, environment)
            optimizer_contract = {"fit": train_contract, "training_config": self.training_config,
                                  "train_truth_sha256": runner.array_hash(truth),
                                  "validation_truth_sha256": runner.array_hash(self.validation_truth)}
            with _preserve_completed_training(fit_dir, optimizer_contract):
                fit = runner.fit_from_same_initialization(
                    atom_base=self.atom, angle=self.angle, normalization=self.normalization, preprocessing=self.preprocessing,
                    train_indices=self._positions(labeled), train_truth=truth,
                    validation_indices=self._positions(self.roles["validation"]), validation_truth=self.validation_truth,
                    prediction_indices=self._positions(unlabeled), prediction_sample_ids=unlabeled,
                    initialization_seed=initialization_seed(self.seed), training_config=self.training_config,
                    contract=train_contract, runtime=fit_dir,
                )
        if fit["initialization_hash"] != self.initialization_hash:
            raise ValueError("same-seed predictor initialization drifted")
        history = pd.read_csv(fit_dir / "history.csv", float_precision="round_trip")
        selected_epoch = history.loc[history.epoch.eq(fit["best_epoch"])]
        if len(selected_epoch) != 1:
            raise ValueError("history does not prove the selected validation epoch")
        best = selected_epoch.iloc[0]
        metric_names = ("combined_normalized_rmse", *(f"{target}_{k}" for target in ("V1", "V2")
                                                      for k in ("rmse", "mae", "r2")))
        metrics = {name: float(best[f"validation_{name}"]) for name in metric_names}
        validation_metrics = self._save_metrics(fit_dir, fit, metrics)
        once(fit_dir / "fit_artifact_receipt.json", {
            name: file_digest(fit_dir / name) for name in (
                "best.pt", "predictions.csv.gz", "fit_audit.json", "history.csv", "training_complete.json",
                "validation_metrics.json", "fit_device.json",
            )
        })
        return {**fit, "validation_metrics": validation_metrics}

    @staticmethod
    def _save_metrics(fit_dir, fit, metrics):
        target = fit["best_validation_combined_normalized_rmse"]
        if not np.isclose(metrics["combined_normalized_rmse"], target, atol=1e-6, rtol=0):
            raise ValueError("validation diagnostics differ from checkpoint selection metric")
        result = {**metrics, "best_epoch": fit["best_epoch"], "checkpoint_sha256": fit["checkpoint_sha256"],
                  "epochs_run": fit["epochs_run"], "training_seconds": fit["training_seconds"]}
        once(Path(fit_dir) / "validation_metrics.json", result)
        return result


@contextmanager
def _qualification_lease(path):
    with Path(path).open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("the designated qualification GPU is still leased") from error
        yield


def main():
    """Prepare public resources and optionally qualify existing anchors without fitting."""
    from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.runner import stage_state

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append")
    parser.add_argument("--anchor-root", type=Path, help="parent of the existing random-<seed> CUDA run directories")
    parser.add_argument("--initial-root", type=Path, help="write qualified L333 model and request receipts here")
    parser.add_argument("--gpu-lease", type=Path, help="shared numeric-suite lock for this physical GPU")
    args = parser.parse_args()
    mode_inputs = (args.anchor_root, args.initial_root, args.gpu_lease)
    if any(mode_inputs) and not all(mode_inputs):
        parser.error("qualifying a CUDA anchor requires --anchor-root, --initial-root and --gpu-lease")
    slot = (lambda: _qualification_lease(args.gpu_lease)) if args.anchor_root else nullcontext
    for seed in args.seed or SEEDS:
        backend = TrainingBackend(args.root, args.cache_root / f"seed_{seed}", seed, gpu_slot=slot)
        result = {"seed": seed, "roles": {role: len(ids) for role, ids in backend.roles.items()},
                  "initialization_hash": backend.initialization_hash, "target_scales": backend.preprocessing["target_scales"],
                  "graph_rows": len(backend.atom), "graph_fit": "label_scrubbed_cpu",
                  "resource_receipt_sha256": backend.cache_receipt_sha256, "training_fits": 0, "api_requests": 0}
        if args.anchor_root:
            directory = args.initial_root / f"seed_{seed}"
            state = backend.initial_state(args.anchor_root / f"random-{seed}", directory / "fit")
            stage = stage_state(args.root, state, arm="A7", memory_mode="M2")
            once(directory / "qualified_source.json", asdict(state))
            once(directory / "qualified_request.json", {
                "binding": stage.binding, "messages": stage.messages,
                "cuda_predictor_qualified": True, "provider_called": False,
            })
            result.update({"status": "qualified_CUDA_L333", "checkpoint_sha256": state.checkpoint,
                           "request_packet_sha256": stage.binding["packet_sha256"]})
        else:
            result["status"] = "public_resource_verified"
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
