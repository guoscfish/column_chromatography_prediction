"""Parameterized Scientist lifecycle, isolated from frozen legacy study globals.

Extracted from the Scientist implementation; model fitting, initialization and
checkpoint rules remain in ShortSequentialContext and the existing runner.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..artifacts import sha256_file
from ..models import load_predictor_checkpoint
from ..resources import SOURCE_DATA
from ..schemas.conditions import ConditionNormalization
from .benchmark_protocol import load_features
from .dialog_bridge import once, read
from .dialog_study import _card as old_card, _observed
from .gradient_transforms import center_width_transform
from .lcmd import _squared_distances, lcmd_tp_select
from .llm_screen import banks
from .maxdet_study import historical_round0_paths
from .protocol import RestrictedLabelStore, ids_hash, stable_hash, validate_row_protocol
from .runner import predict_outputs
from .short_sequential_runner import ShortSequentialContext
from .short_sequential_study import STUDY as CW_SHORT
from .scientist_selector import VERSION, ReadOnlyCatalog
from .scientist_study import SEEDS, METHODS, prediction_freeze


class WorkflowContext(ShortSequentialContext):
    def __init__(self, seed, study_root):
        if seed not in SEEDS:
            raise ValueError("unregistered seed")
        self.seed, self.study = seed, Path(study_root)
        self.runtime = self.study / f"runtime/seed_{seed}"
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.partition = pd.read_csv(CW_SHORT / f"splits/row_seed_{seed}.csv")
        validate_row_protocol(self.partition)
        self.data = load_features()
        if self.data.sample_id.astype(str).tolist() != self.partition.sample_id.astype(str).tolist():
            raise RuntimeError("source order mismatch")
        self.roles = {r: self.partition.loc[self.partition.role.eq(r), "canonical_index"].to_numpy(int)
                      for r in ("l0", "u0", "validation", "test")}
        old = read(historical_round0_paths(seed)["context"])["contract"]
        self.preprocessing = old["preprocessing"]
        self.normalization = ConditionNormalization(**old["normalization"])
        self.atom, self.angle = torch.load(historical_round0_paths(seed)["scrubbed_graphs"], weights_only=False)
        if any(torch.count_nonzero(item.y) for item in self.atom):
            raise RuntimeError("scrubbed graph labels must be zero")
        store = RestrictedLabelStore(SOURCE_DATA, self.partition)
        self.l0_truth = store.reveal(self.ids(self.roles["l0"]), "initial_fit")
        self.validation_truth = None
        self.cw_transform, self.cw_audit = center_width_transform(self.l0_truth)
        self.protocol_hash = stable_hash(read(self.study / "protocol.json"))
        once(self.runtime / "context.json", {"seed": seed, "protocol_hash": self.protocol_hash,
             "split_hash": stable_hash(self.partition.to_dict("list")), "preprocessing": self.preprocessing,
             "normalization": asdict(self.normalization), "initial_label_access": store.audit})

    def new_store(self):
        return RestrictedLabelStore(SOURCE_DATA, self.partition)


def state(context, method, round_index, workflow):
    labeled, unlabeled, truth = context.roles["l0"].copy(), context.roles["u0"].copy(), context.l0_truth.copy()
    history = []
    for r in range(round_index):
        directory = workflow.selection_directory(context.seed, method, r)
        frozen = workflow.audit_batch(directory)
        feedback = read(directory / "feedback.json")
        receipt = read(directory / "label_access_receipt.json")
        if feedback["batch_freeze_sha256"] != sha256_file(directory / "batch_freeze.json") or receipt["feedback_sha256"] != stable_hash(feedback):
            raise RuntimeError("feedback lineage drift")
        if (receipt["batch_freeze_sha256"] != feedback["batch_freeze_sha256"]
                or receipt["audit"]["purpose"] != "after_acquisition_fit"
                or not receipt["audit"]["acquisitions_frozen"]
                or receipt["audit"]["requested_ids_hash"] != ids_hash(frozen["batch_ids"])):
            raise RuntimeError("label receipt does not prove frozen batch reveal")
        if (feedback["seed"], feedback["method"], feedback["round"]) != (context.seed, method, r):
            raise RuntimeError("foreign trajectory feedback")
        ids = frozen["batch_ids"]
        if [row["candidate_id"] for row in feedback["records"]] != ids:
            raise RuntimeError("feedback ID alignment mismatch")
        for row, pred in zip(feedback["records"], frozen["premeasurement_predictions"]):
            for target in ("V1", "V2"):
                if (row[f"pred_{target}_ml"] != pred[f"pred_{target}_ml"] or
                        row[f"error_{target}_ml"] != row[f"pred_{target}_ml"]-row[f"true_{target}_ml"]):
                    raise RuntimeError("feedback premeasurement prediction/error mismatch")
        lookup = dict(zip(context.ids(unlabeled), unlabeled))
        indices = np.array([lookup[i] for i in ids], dtype=int)
        labeled = np.r_[labeled, indices]
        unlabeled = np.array([i for i in unlabeled if i not in set(indices)], dtype=int)
        truth = np.vstack([truth, [[row["true_V1_ml"], row["true_V2_ml"]] for row in feedback["records"]]])
        history.append(feedback)
    if len(labeled) != workflow.BUDGETS[round_index]:
        raise RuntimeError("active budget mismatch")
    return labeled, unlabeled, truth, history


def make_catalog(context, method, r, labeled, unlabeled, truth, history, checkpoint, directory,
                 pending_count=None, version=VERSION):
    current, n = np.r_[labeled, unlabeled], len(labeled)
    cw, _ = banks(context, current, checkpoint, r, directory)
    if pending_count is None:
        pending_count = 16 if method == METHODS[0] else 0
    positions = lcmd_tp_select(cw[n:], cw[:n], pending_count).selected_pool_positions if pending_count else []
    pending_ids = context.ids(unlabeled[np.asarray(positions, dtype=int)])
    predictions, order = predict_outputs(load_predictor_checkpoint(checkpoint), context.atom, context.angle, unlabeled)
    if not np.array_equal(order, unlabeled) or predictions.shape != (len(unlabeled), 6):
        raise RuntimeError("prediction identity/quantile schema mismatch")
    percentiles = pd.Series(np.sqrt(_squared_distances(cw[n:], cw[:n]).min(axis=1))).rank(pct=True).to_numpy()
    cards = []
    for index, pred, distance in zip(unlabeled, predictions, percentiles):
        card = old_card(context, index, pred, distance)
        card.update({f"pred_{target}_{quantile}_ml": float(pred[j]) for j, (target, quantile) in enumerate(
            (t, q) for t in ("V1", "V2") for q in ("q10", "q50", "q90"))})
        cards.append(card)
    lookup = {row["id"]: row for row in cards}
    raw = {"candidates": [row for row in cards if row["id"] not in set(pending_ids)],
           "pending": [lookup[i] for i in pending_ids],
           "observed": _observed(context, labeled, truth, history),
           "salt": f"{version}:{context.seed}:{method}:{r}"}
    return ReadOnlyCatalog(**raw), raw


def advance(seed, method, r, workflow):
    """Explicitly reveal one frozen batch and fit exactly one next-budget predictor."""
    workflow.validate()
    directory = workflow.selection_directory(seed, method, r)
    frozen = workflow.audit_batch(directory)
    context = workflow.Context(seed)
    labeled, unlabeled, truth, history = workflow.state(context, method, r)
    result = read(directory / "selection.json")
    batch = frozen["batch_ids"]
    all_ids = [row["candidate_id"] for h in history for row in h["records"]] + batch
    store = context.new_store()
    store.freeze_acquisitions(all_ids)
    if (directory / "feedback.json").exists():
        # state() verifies the existing feedback and receipt before any resumed fit.
        new_labeled, _, new_truth, _ = workflow.state(context, method, r+1)
    else:
        values = store.reveal(batch, "after_acquisition_fit")
        reasons = {c["id"]: c for c in result["choices"]}
        records = []
        for pred, value in zip(frozen["premeasurement_predictions"], values):
            i = pred["candidate_id"]
            records.append({**pred, "source": "CW" if i in frozen["pending_ids"] else "LLM",
                "true_V1_ml": float(value[0]), "true_V2_ml": float(value[1]),
                "error_V1_ml": pred["pred_V1_ml"]-float(value[0]),
                "error_V2_ml": pred["pred_V2_ml"]-float(value[1]),
                "reason": reasons.get(i, {}).get("reason", "pending geometric coverage"),
                "hypothesis_id": reasons.get(i, {}).get("hypothesis_id")})
        feedback = {"seed": seed, "method": method, "round": r,
                    "batch_freeze_sha256": sha256_file(directory / "batch_freeze.json"),
                    "records": records, **{k: result[k] for k in ("hypotheses", "feedback_interpretation", "batch_rationale")}}
        if hasattr(workflow, "feedback_metadata"):
            feedback.update(workflow.feedback_metadata(result, directory))
        once(directory / "label_access_receipt.json", {"feedback_sha256": stable_hash(feedback),
             "batch_freeze_sha256": sha256_file(directory / "batch_freeze.json"), "audit": store.audit[-1]})
        once(directory / "feedback.json", feedback)
        new_labeled, _, new_truth, _ = workflow.state(context, method, r+1)
    context.validation_truth = store.reveal(context.ids(context.roles["validation"]), "initial_fit")
    next_dir = context.runtime / method / f"round_{r+1:02d}"
    torch.set_num_threads(2)
    context.fit(method, r+1, new_labeled, new_truth, next_dir / "model")
    workflow.prediction_freeze(context, method, r+1, new_labeled)
    if r == len(workflow.BUDGETS)-2:
        hashes = {}
        for j in range(len(workflow.BUDGETS)):
            p = context.runtime / method / f"round_{j:02d}/prediction_freeze.json"
            hashes[str(p.relative_to(workflow.STUDY))] = sha256_file(p)
        for j in range(len(workflow.BUDGETS)-1):
            d = workflow.selection_directory(seed, method, j)
            for name in ("batch_freeze.json", "feedback.json", "label_access_receipt.json"):
                p = d / name
                hashes[str(p.relative_to(workflow.STUDY))] = sha256_file(p)
        once(context.runtime / method / "trajectory_freeze.json", {"seed": seed, "method": method,
             "selected_ids": all_ids, "final_active_labels": len(new_labeled), "files": hashes})
    return {"status": "ONE_BATCH_TRAINED", "active_label_count": len(new_labeled), "next_round": r+1,
            "next_selection_started": False}
