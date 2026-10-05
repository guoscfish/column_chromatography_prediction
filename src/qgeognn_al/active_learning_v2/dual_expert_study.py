"""Independent Chemistry Scientist + ML Scientist -> Planner study.

This module stops at round-0 selection. It has no reveal, retraining, or test-label path.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..artifacts import sha256_file
from ..models import load_predictor_checkpoint
from ..resources import ROOT, SOURCE_DATA
from ..schemas.conditions import ConditionNormalization
from .benchmark_protocol import TRAINING_CONFIG, load_features, package_versions
from .dialog_bridge import once, read
from .dialog_study import _card as old_card, _observed
from .gradient_transforms import center_width_transform
from .llm_screen import banks
from .maxdet_study import historical_round0_paths
from .protocol import RestrictedLabelStore, ids_hash, stable_hash, validate_row_protocol
from .runner import predict_outputs
from .short_sequential_runner import ShortSequentialContext
from .short_sequential_study import STUDY as CW_SHORT
from .scientist_selector import ReadOnlyCatalog
from .dual_expert_selector import (CHEMISTRY_SYSTEM_PROMPT, ML_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT,
                                   METHOD, QUERY_BUDGET, SELECTION_COUNT, VERSION, VIEW_BUDGET,
                                   trajectory_memory, validate_planner_selection)
from .dual_expert_transport import run_agent, settings

STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_llm_dual_expert_v1"
SEEDS = (157, 6101)
METHODS = (METHOD,)
CONTROLS = ("random32", "center_width_lcmd")
BUDGETS = (333, 365, 397, 429, 461, 493, 525)
BASE_COMMIT = "647271df406cb30f606fd58de97b182b87fe2fd5"


def code_hashes():
    paths = [ROOT / "scripts/studies/run_qgeognn_v2_row_llm_dual_expert.py"]
    paths += [ROOT / "src/qgeognn_al/active_learning_v2" / name for name in (
        "dual_expert_selector.py", "dual_expert_transport.py", "dual_expert_study.py",
        "scientist_selector.py", "scientist_transport.py", "dialog_study.py", "dialog_bridge.py",
        "llm_screen.py", "protocol.py", "runner.py", "short_sequential_runner.py")]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)}


def source_manifest():
    paths = [SOURCE_DATA, CW_SHORT / "protocol.json"]
    for seed in SEEDS:
        paths += [CW_SHORT / "splits" / f"row_seed_{seed}.csv"]
        historic = historical_round0_paths(seed)
        paths += [historic[k] for k in ("context", "scrubbed_graphs", "member0_model", "gradient", "gradient_contract")]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


def prepare(config=None):
    config = config or settings()
    if (STUDY / "protocol.json").exists():
        if read(STUDY / "protocol.json")["transport"] != config:
            raise RuntimeError("transport cannot change after protocol freeze")
        return validate()
    STUDY.mkdir(parents=True, exist_ok=True)
    record = {
        "study_version": VERSION, "base_commit": BASE_COMMIT, "status": "FROZEN_BEFORE_NEW_SELECTION",
        "scientific_question": "Does independent chemistry and ML advice, synthesized by a free planner, improve QGeoGNN label efficiency?",
        "split": "Row", "seeds": list(SEEDS), "methods": [*CONTROLS, METHOD], "budgets": list(BUDGETS),
        "batch_size": 32, "selection_count": SELECTION_COUNT, "training": TRAINING_CONFIG,
        "training_control": "unchanged QGeoGNN configuration; no training is started by this implementation",
        "transport": config, "query_budget_per_role": QUERY_BUDGET, "view_budget_per_role": VIEW_BUDGET,
        "model_call_budget": {"chemistry_scientist": 12, "ml_scientist": 12, "scientific_planner": 16},
        "roles": ["chemistry_scientist", "ml_scientist", "scientific_planner"],
        "fixed_quota": None, "hidden_reranking": None,
        "context_isolation": "independent role transcripts and seed/method/round directories",
        "truth_barrier": "batch_freeze.json must exist before any future reveal; this module has no reveal/advance path",
        "selector_boundary": "trajectory-local L_t/U_t features and predictions only; no validation/test/other seed truth",
        "prompts": {"chemistry": stable_hash(CHEMISTRY_SYSTEM_PROMPT), "ml": stable_hash(ML_SYSTEM_PROMPT),
                    "planner": stable_hash(PLANNER_SYSTEM_PROMPT)},
        "source_manifest": source_manifest(), "code_hashes": code_hashes(), "packages": package_versions(),
        "historical_v2_base_commit": BASE_COMMIT,
    }
    once(STUDY / "protocol.json", record)
    for name, prompt in (("chemistry_prompt.txt", CHEMISTRY_SYSTEM_PROMPT),
                         ("ml_prompt.txt", ML_SYSTEM_PROMPT), ("planner_prompt.txt", PLANNER_SYSTEM_PROMPT)):
        path = STUDY / name
        if path.exists() and path.read_text() != prompt:
            raise RuntimeError(f"prompt artifact changed: {name}")
        path.write_text(prompt)
    return {"status": "PREPARED", "new_training_runs_started": 0, "labels_revealed": 0}


def validate():
    record = read(STUDY / "protocol.json")
    if record["base_commit"] != BASE_COMMIT or record["code_hashes"] != code_hashes():
        raise RuntimeError("frozen code/base drift")
    if record["source_manifest"] != source_manifest():
        raise RuntimeError("source or control artifact drift")
    if ((STUDY / "chemistry_prompt.txt").read_text() != CHEMISTRY_SYSTEM_PROMPT
            or (STUDY / "ml_prompt.txt").read_text() != ML_SYSTEM_PROMPT
            or (STUDY / "planner_prompt.txt").read_text() != PLANNER_SYSTEM_PROMPT):
        raise RuntimeError("prompt artifact drift")
    return {"status": "VALID", "protocol_sha256": stable_hash(record), "old_runtime_modified": False,
            "labels_revealed": 0, "fits_started": 0}


class Context(ShortSequentialContext):
    def __init__(self, seed):
        if seed not in SEEDS:
            raise ValueError("unregistered seed")
        self.seed, self.study = seed, STUDY
        self.runtime = STUDY / f"runtime/seed_{seed}"
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.partition = pd.read_csv(CW_SHORT / f"splits/row_seed_{seed}.csv")
        validate_row_protocol(self.partition)
        self.data = load_features()
        if self.data.sample_id.astype(str).tolist() != self.partition.sample_id.astype(str).tolist():
            raise RuntimeError("source order mismatch")
        self.roles = {role: self.partition.loc[self.partition.role.eq(role), "canonical_index"].to_numpy(int)
                      for role in ("l0", "u0", "validation", "test")}
        old = read(historical_round0_paths(seed)["context"])["contract"]
        self.preprocessing = old["preprocessing"]
        self.normalization = ConditionNormalization(**old["normalization"])
        self.atom, self.angle = torch.load(historical_round0_paths(seed)["scrubbed_graphs"], weights_only=False)
        if any(torch.count_nonzero(item.y) for item in self.atom):
            raise RuntimeError("scrubbed graph labels are not allowed")
        store = RestrictedLabelStore(SOURCE_DATA, self.partition)
        self.l0_truth = store.reveal(self.ids(self.roles["l0"]), "initial_fit")
        self.validation_truth = None
        self.cw_transform, self.cw_audit = center_width_transform(self.l0_truth)
        self.protocol_hash = stable_hash(read(STUDY / "protocol.json"))
        once(self.runtime / "context.json", {"seed": seed, "protocol_hash": self.protocol_hash,
             "split_hash": stable_hash(self.partition.to_dict("list")), "preprocessing": self.preprocessing,
             "normalization": asdict(self.normalization), "initial_label_access": store.audit,
             "validation_test_truth_access_count": 0})

    def new_store(self):
        return RestrictedLabelStore(SOURCE_DATA, self.partition)


def selection_directory(seed, method=METHOD, round_index=0):
    if seed not in SEEDS or method != METHOD or round_index != 0:
        raise ValueError("only registered round-0 dual-expert trajectory is available")
    return STUDY / f"selections/seed_{seed}/{method}/round_{round_index:02d}"


def make_catalog(context, checkpoint, directory):
    labeled, unlabeled = context.roles["l0"], context.roles["u0"]
    current = np.r_[labeled, unlabeled]
    cw, _ = banks(context, current, checkpoint, 0, directory)
    predictions, order = predict_outputs(load_predictor_checkpoint(checkpoint), context.atom, context.angle, unlabeled)
    if not np.array_equal(order, unlabeled) or predictions.shape != (len(unlabeled), 6):
        raise RuntimeError("prediction identity/quantile schema mismatch")
    distances = np.sqrt(((cw[len(labeled):, None, :] - cw[None, :len(labeled), :]) ** 2).sum(axis=2)).min(axis=1)
    percentiles = pd.Series(distances).rank(pct=True).to_numpy()
    cards = []
    for index, pred, distance in zip(unlabeled, predictions, percentiles):
        card = old_card(context, index, pred, distance)
        card.update({f"pred_{target}_{quantile}_ml": float(pred[j]) for j, (target, quantile) in enumerate(
            (pair for target in ("V1", "V2") for pair in ((target, "q10"), (target, "q50"), (target, "q90"))))})
        cards.append(card)
    raw = {"candidates": cards, "pending": [], "observed": _observed(context, labeled, context.l0_truth, []),
           "salt": f"{VERSION}:{context.seed}:{METHOD}:0"}
    return ReadOnlyCatalog(**raw), raw


def stage(seed, method=METHOD, r=0):
    validate()
    directory = selection_directory(seed, method, r)
    directory.mkdir(parents=True, exist_ok=True)
    context = Context(seed)
    checkpoint = historical_round0_paths(seed)["member0_model"]
    catalog, raw = make_catalog(context, checkpoint, directory)
    base_packet = {"study_version": VERSION, "seed": seed, "method": method, "round": r,
                   "active_label_count": len(context.roles["l0"]), "selection_count": SELECTION_COUNT,
                   "objective": "overall Row V1/V2 predictive accuracy after retraining",
                   "memory": trajectory_memory([], seed, method), "candidate_overview": catalog.overview(),
                   "initial_cards": catalog.initial_cards(), "observed_examples": sorted(catalog.observed.values(), key=catalog.key)[:12],
                   "observed_count": len(catalog.observed), "pending_experiments": [],
                   "validation_test_records": 0, "truth_access": {"validation": 0, "test": 0}}
    base_packet["packet_sha256"] = stable_hash(base_packet)
    once(directory / "input.json", base_packet)
    once(directory / "catalog.json", raw)
    once(directory / "contract.json", {"seed": seed, "method": method, "round": r,
         "protocol_hash": context.protocol_hash, "checkpoint_sha256": sha256_file(checkpoint),
         "L_t_ids": context.ids(context.roles["l0"]), "U_t_ids": context.ids(context.roles["u0"]),
         "input_sha256": sha256_file(directory / "input.json"), "catalog_sha256": sha256_file(directory / "catalog.json"),
         "pending_ids": [], "new_labels_revealed": 0})
    return {"status": "READY_FOR_FIRST_DUAL_EXPERT_SELECTION", "seed": seed, "method": method, "round": r,
            "packet": str(directory / "input.json"), "eligible_count": len(catalog.candidates), "pending_count": 0,
            "new_labels_revealed": 0, "validation_test_truth_access_count": 0, "llm_calls": 0, "fits": 0}


def select(seed, method=METHOD, r=0, call=None):
    stage(seed, method, r)
    directory = selection_directory(seed, method, r)
    if (directory / "batch_freeze.json").exists():
        return {"status": "ALREADY_FROZEN"}
    packet, raw = read(directory / "input.json"), read(directory / "catalog.json")
    catalog = ReadOnlyCatalog(**raw)
    catalog.initial_cards()
    config = read(STUDY / "protocol.json")["transport"]
    role_root = directory / "agents"
    chemistry_packet = {**packet, "role": "chemistry_scientist", "expert_memo_required": True}
    ml_packet = {**packet, "role": "ml_scientist", "expert_memo_required": True}
    chemistry = run_agent("chemistry_scientist", chemistry_packet, catalog, role_root / "chemistry_scientist", config, call=call)
    # Each role receives an independent catalog instance and transcript directory.
    ml_catalog = ReadOnlyCatalog(**raw)
    ml_catalog.initial_cards()
    ml = run_agent("ml_scientist", ml_packet, ml_catalog, role_root / "ml_scientist", config, call=call)
    once(directory / "expert_memos.json", {"chemistry_scientist": chemistry, "ml_scientist": ml})
    planner_catalog = ReadOnlyCatalog(**raw)
    planner_catalog.initial_cards()
    planner_packet = {**packet, "role": "scientific_planner", "chemistry_memo": chemistry, "ml_memo": ml,
                      "expert_memos_are_advisory": True}
    planner_packet["packet_sha256"] = stable_hash(planner_packet)
    planner = run_agent("scientific_planner", planner_packet, planner_catalog, role_root / "scientific_planner", config, call=call)
    once(directory / "planner_input.json", planner_packet)
    once(directory / "selection.json", planner)
    batch = [item["id"] for item in planner["choices"]]
    contract = read(directory / "contract.json")
    if len(batch) != 32 or len(set(batch)) != 32 or not set(batch) <= set(contract["U_t_ids"]):
        raise ValueError("planner selected an illegal batch")
    lookup = catalog.candidates
    artifact_names = ["input.json", "catalog.json", "contract.json", "expert_memos.json", "planner_input.json", "selection.json"]
    artifact_names += [str(p.relative_to(directory)) for p in sorted(role_root.rglob("turn_*.json"))]
    once(directory / "batch_freeze.json", {"batch_ids": batch, "pending_ids": [],
         "premeasurement_predictions": [{"candidate_id": i, "pred_V1_ml": lookup[i]["pred_V1_ml"],
                                          "pred_V2_ml": lookup[i]["pred_V2_ml"]} for i in batch],
         "artifact_hashes": {name: sha256_file(directory / name) for name in artifact_names},
         "viewed_candidate_ids": sorted(planner_catalog.viewed), "new_batch_labels_revealed": False,
         "label_access_count": 0, "training_runs_started": 0})
    return {"status": "BATCH_FROZEN_BEFORE_REVEAL", "batch_count": 32, "new_labels_revealed": 0,
            "training_runs_started": 0, "expert_roles": ["chemistry_scientist", "ml_scientist", "scientific_planner"]}


def advance(*args, **kwargs):
    raise RuntimeError("Dual Expert v1 intentionally stops before label reveal and retraining")


def report(*args, **kwargs):
    raise RuntimeError("Dual Expert v1 has no formal report before an approved future extension")
