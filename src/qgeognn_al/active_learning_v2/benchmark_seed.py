"""One seed: train L0, freeze acquisitions, reveal selected labels, freeze fits."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..artifacts import sha256_file
from ..models import load_predictor_checkpoint
from ..resources import SOURCE_GRAPH_CACHE
from ..training.predictor import atomic_json
from .acquisition import acquire_batches
from .benchmark_protocol import (code_hashes, initialization_seed, label_budget, load_features, package_versions,
                                 seed_config, sketch_seed)
from .cache import array_hash, seal_cache, verify_cache
from .coverage import extract_representations
from .gradient_features import extract_q50_gradient_sketches
from .protocol import RestrictedLabelStore, ids_hash, stable_hash, validate_row_protocol
from .runner import fit_al_preprocessing, fit_from_same_initialization, make_label_scrubbed_graphs


@dataclass
class SeedContext:
    source: Path
    runtime: Path
    seed: int
    partition: pd.DataFrame
    smoke: bool = False

    def __post_init__(self):
        torch.set_num_threads(2)
        self.runtime.mkdir(parents=True, exist_ok=True)
        validate_row_protocol(self.partition)
        if set(self.partition.outer_seed) != {self.seed}:
            raise ValueError("context and partition outer seed mismatch")
        self.data = load_features(self.source)
        if self.data.sample_id.tolist() != self.partition.sample_id.tolist():
            raise ValueError("partition/source identity order mismatch")
        if self.data.canonical_smiles.tolist() != self.partition.canonical_smiles.tolist():
            raise ValueError("partition/source molecule identity mismatch")
        self.roles = {role: self.partition.loc[self.partition.role.eq(role), "canonical_index"].to_numpy(int)
                      for role in ("l0", "u0", "validation", "test")}
        if self.partition.canonical_index.tolist() != list(range(len(self.data))):
            raise ValueError("canonical index drift")
        self.store = RestrictedLabelStore(self.source, self.partition)
        self.l0_truth = self.store.reveal(self.ids(self.roles["l0"]), "initial_fit")
        self.validation_truth = self.store.reveal(self.ids(self.roles["validation"]), "initial_fit")
        self.graph_cache = torch.load(SOURCE_GRAPH_CACHE, weights_only=False)
        self.outer = np.r_[self.roles["l0"], self.roles["u0"]]
        self.normalization, self.preprocessing = fit_al_preprocessing(
            self.data, self.graph_cache, self.outer, self.ids(self.roles["l0"]), self.l0_truth,
            self.runtime / "scaler.json")
        self.contract = {
            "source_sha256": sha256_file(self.source), "graph_cache_sha256": sha256_file(SOURCE_GRAPH_CACHE),
            "code_hashes": code_hashes(), "ordered_sample_ids": self.ids(range(len(self.data))),
            "partition": self.partition.to_dict("list"), "normalization": asdict(self.normalization),
            "preprocessing": self.preprocessing, "smoke": self.smoke,
            "packages": package_versions(),
        }
        graph_path = self.runtime / "scrubbed_graphs.pt"
        if verify_cache(graph_path, self.contract):
            self.atom, self.angle = torch.load(graph_path, weights_only=False)
        else:
            self.atom, self.angle = make_label_scrubbed_graphs(self.data, self.graph_cache, self.preprocessing["scaler"])
            torch.save((self.atom, self.angle), graph_path)
            seal_cache(graph_path, self.contract)
        if any(torch.count_nonzero(a.y) for a in self.atom):
            raise RuntimeError("cached graph labels are not scrubbed")
        atomic_json(self.runtime / "context.json", {"contract_hash": stable_hash(self.contract),
                    "source_sha256": self.contract["source_sha256"], "preprocessing": self.preprocessing,
                    "roles": {role: ids_hash(self.ids(indices)) for role, indices in self.roles.items()}})

    def ids(self, indices):
        return self.data.iloc[list(indices)].sample_id.astype(str).tolist()

    def fit(self, arm, train_indices, truth, member=0, predict_pool=False):
        config = seed_config(self.seed, member, self.smoke)
        config["split_sha256"] = stable_hash(self.partition.to_dict("list"))
        prediction_indices = self.roles["u0"] if predict_pool else self.roles["test"]
        contract = {"context_hash": stable_hash(self.contract), "arm": arm,
                    "train_sample_ids": self.ids(train_indices),
                    "validation_sample_ids": self.ids(self.roles["validation"])}
        audit = fit_from_same_initialization(
            atom_base=self.atom, angle=self.angle, normalization=self.normalization,
            preprocessing=self.preprocessing, train_indices=train_indices, train_truth=truth,
            validation_indices=self.roles["validation"], validation_truth=self.validation_truth,
            prediction_indices=prediction_indices, prediction_sample_ids=self.ids(prediction_indices),
            initialization_seed=initialization_seed(self.seed, member), training_config=config,
            contract=contract, runtime=self.runtime / arm)
        return {"outer_seed": self.seed, "arm": arm, "member": member,
                "l0_ids_hash": ids_hash(self.ids(self.roles["l0"])),
                "test_ids_hash": ids_hash(self.ids(self.roles["test"])), **audit,
                **label_budget(len(train_indices), len(self.roles["validation"]), len(self.outer), len(self.data))}

    def initial_features(self, primary=True):
        baseline = self.fit("baseline_l0", self.roles["l0"], self.l0_truth)
        model = load_predictor_checkpoint(self.runtime / "baseline_l0/best.pt")
        scales = tuple(self.preprocessing["target_scales"][t] for t in ("V1", "V2"))
        feature_contract = {"context_hash": stable_hash(self.contract), "checkpoint_sha256": baseline["checkpoint_sha256"],
                            "sample_ids": self.ids(self.outer), "sketch_seed": sketch_seed(self.seed),
                            "scales": scales, "dimension": 512}
        path = self.runtime / "gradient_features.npz"
        if verify_cache(path, feature_contract):
            with np.load(path) as stored:
                gradients = stored["features"]
        else:
            result = extract_q50_gradient_sketches(model, self.atom, self.angle, self.outer, scales,
                                                 dimension=512, sketch_seed=sketch_seed(self.seed))
            gradients = result.features
            np.savez_compressed(path, features=gradients, canonical_indices=self.outer)
            seal_cache(path, feature_contract)
            atomic_json(self.runtime / "gradient_audit.json", result.audit)
        if gradients.shape != (len(self.outer), 512) or not np.isfinite(gradients).all():
            raise RuntimeError("gradient cache shape/content mismatch")
        fits = [baseline]
        representations, predictions = None, None
        if primary:
            representations = extract_representations(model, self.atom, self.angle, self.outer)
            predictions = []
            # Member 0 reuses the exact baseline initialization/checkpoint, without a second fit.
            from .runner import predict_outputs
            values, order = predict_outputs(model, self.atom, self.angle, self.roles["u0"])
            if not np.array_equal(order, self.roles["u0"]):
                raise RuntimeError("ensemble member 0 order mismatch")
            predictions.append(values)
            for member in (1, 2):
                arm = f"ensemble_member_{member}"
                fits.append(self.fit(arm, self.roles["l0"], self.l0_truth, member, predict_pool=True))
                table = pd.read_csv(self.runtime / arm / "predictions.csv.gz")
                if table.sample_id.tolist() != self.ids(self.roles["u0"]):
                    raise RuntimeError("ensemble member order mismatch")
                predictions.append(table.drop(columns="sample_id").to_numpy())
            predictions = np.stack(predictions)
        return baseline, fits, gradients, representations, predictions


def freeze_seed(context, batch_size):
    started = time.perf_counter()
    baseline, fits, gradients, representations, predictions = context.initial_features(primary=batch_size == 32)
    arms = acquire_batches(gradients=gradients, representations=representations,
                           ensemble_predictions=predictions, l0_count=len(context.roles["l0"]),
                           scales=tuple(context.preprocessing["target_scales"][t] for t in ("V1", "V2")),
                           outer_seed=context.seed, batch_size=batch_size)
    out = context.runtime / f"b{batch_size}"
    out.mkdir(exist_ok=True)
    selections = pd.DataFrame([
        {"arm": arm, "selection_order": rank, "canonical_index": int(context.roles["u0"][pos]),
         "sample_id": context.ids([context.roles["u0"][pos]])[0]}
        for arm, positions in arms.items() for rank, pos in enumerate(positions)])
    selections.to_csv(out / "selected_batches.csv", index=False)
    from .diagnostics import audit_context
    audit_context(context, gradients, selections, out)
    context.store.freeze_acquisitions(selections.sample_id)
    for arm, positions in arms.items():
        selected = context.roles["u0"][positions]
        truth = context.store.reveal(context.ids(selected), "after_acquisition_fit")
        fits.append(context.fit(f"b{batch_size}/{arm}", np.r_[context.roles["l0"], selected],
                                np.vstack([context.l0_truth, truth])))
    fit_frame = pd.DataFrame(fits)
    evaluation = fit_frame.loc[~fit_frame.arm.str.startswith("ensemble_member_")]
    if evaluation.initialization_hash.nunique() != 1:
        raise RuntimeError("evaluation initialization mismatch")
    for column in ("l0_ids_hash", "validation_ids_hash", "test_ids_hash"):
        if fit_frame[column].nunique() != 1:
            raise RuntimeError(f"matched identities mismatch: {column}")
    fit_frame.to_csv(out / "fit_audit.csv", index=False)
    pd.DataFrame(context.store.audit).to_csv(out / "label_access_audit.csv", index=False)
    evaluation[["arm", "initialization_seed", "initialization_hash", "validation_ids_hash", "test_ids_hash"]].to_csv(
        out / "initialization_hash_audit.csv", index=False)
    protected = [out / "selected_batches.csv", out / "fit_audit.csv", out / "label_access_audit.csv",
                 context.runtime / "context.json", context.runtime / "gradient_features.npz"]
    protected.extend(out / name for name in ("input_redundancy_audit.csv", "gradient_duplicate_audit.csv",
                                            "gradient_mechanism_correlations.csv", "gradient_mechanism_selection_profiles.csv"))
    for arm in fit_frame.arm:
        protected.extend(context.runtime / arm / name for name in ("best.pt", "predictions.csv.gz", "fit_audit.json"))
    freeze = {"status": "FROZEN_BEFORE_TEST_TRUTH", "outer_seed": context.seed, "batch_size": batch_size,
              "smoke": context.smoke, "context_hash": stable_hash(context.contract),
              "test_truth_access_count": 0, "gradient_semantic_hash": array_hash(gradients),
              "selected_ids_hash": stable_hash(selections[["arm", "sample_id"]].to_dict("list")),
              "checkpoint_state_hash": baseline["checkpoint_state_hash"],
              "seconds": time.perf_counter() - started,
              "files": {str(p.relative_to(context.runtime)): sha256_file(p) for p in protected}}
    atomic_json(out / "pre_test_freeze.json", freeze)
    return freeze
