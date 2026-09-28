"""Independent LLM Scientist V2 study with explicit selection/reveal/fit phases."""
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
from .benchmark_reporting import metric_row
from .cw_lcmd_extension_study import STUDY as CW_EXTENSION
from .dialog_bridge import once, read
from .dialog_study import _card as old_card, _observed
from .gradient_transforms import center_width_transform
from .lcmd import _squared_distances, lcmd_tp_select
from .llm_screen import banks
from .maxdet_study import BASELINE, historical_round0_paths
from .protocol import RestrictedLabelStore, ids_hash, stable_hash, validate_row_protocol
from .runner import predict_outputs
from .short_sequential_runner import ShortSequentialContext, _round0_evaluation
from .short_sequential_study import STUDY as CW_SHORT
from .scientist_selector import (VERSION, SYSTEM_PROMPT, ReadOnlyCatalog, scientific_memory,
                                 validate_selection, QUERY_BUDGET, VIEW_BUDGET, MODEL_CALL_BUDGET)
from .scientist_transport import settings, run_selector

STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_llm_scientist_v2"
SEEDS = (157, 6101)
METHODS = ("cw16_llm16_scientist_v2", "free_llm32_scientist_v2")
CONTROLS = ("random32", "center_width_lcmd")
BUDGETS = (333, 365, 397, 429, 461, 493, 525)


def code_hashes():
    paths = list((ROOT / "src/qgeognn_al").rglob("*.py"))
    paths += [ROOT / "scripts/studies/run_qgeognn_v2_row_llm_scientist.py"]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)}


def historical_manifest():
    paths = []
    for name in ("qgeognn_v2_row_llm16_screen", "qgeognn_v2_row_llm_full_pool_feedback_v1",
                 "qgeognn_v2_row_llm_dialog_feedback_v1"):
        paths.extend(p for p in (STUDY.parent / name).rglob("*") if p.is_file())
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)}


def control_reuse():
    """Verify provenance without reading any historical performance metrics or truth."""
    source_protocols = {str(base.relative_to(ROOT)): read(base / "protocol.json")
                        for base in (BASELINE, CW_SHORT, CW_EXTENSION)}
    if any(p["training"] != TRAINING_CONFIG for p in source_protocols.values()):
        raise RuntimeError("control training configuration mismatch")
    rows, sources = [], {str(SOURCE_DATA.relative_to(ROOT)): sha256_file(SOURCE_DATA)}
    for base in (BASELINE, CW_SHORT, CW_EXTENSION):
        sources[str((base / "protocol.json").relative_to(ROOT))] = sha256_file(base / "protocol.json")
    for seed in SEEDS:
        split = pd.read_csv(CW_SHORT / f"splits/row_seed_{seed}.csv")
        other = pd.read_csv(BASELINE / f"splits/row_seed_{seed}.csv")
        validate_row_protocol(split)
        for column in ("sample_id", "canonical_index", "role", "outer_seed"):
            if not split[column].equals(other[column]):
                raise RuntimeError("control split mismatch")
        l0 = split.loc[split.role.eq("l0"), "sample_id"].astype(str).tolist()
        validation_ids = split.loc[split.role.eq("validation"), "sample_id"].astype(str).tolist()
        historical = historical_round0_paths(seed)
        initial_hash = read(historical["member0_audit"])["initialization_hash"]
        for key in ("context", "scrubbed_graphs", "member0_model", "member0_predictions", "member0_audit",
                    "gradient", "gradient_contract"):
            p = historical[key]
            sources[str(p.relative_to(ROOT))] = sha256_file(p)
        for base in (CW_SHORT, BASELINE):
            p = base / f"splits/row_seed_{seed}.csv"
            sources[str(p.relative_to(ROOT))] = sha256_file(p)
        cw_features = CW_SHORT / f"runtime/seed_{seed}/center_width_lcmd/round_00/acquisition_artifacts/current_transformed_gradient_features.npz"
        for p in (cw_features, cw_features.with_suffix(".npz.contract.json")):
            sources[str(p.relative_to(ROOT))] = sha256_file(p)
        for method in CONTROLS:
            for r, budget in enumerate(BUDGETS):
                if method == "random32":
                    directory = BASELINE / f"runtime/seed_{seed}/random/round_{r:02d}"
                    freeze_path = directory / "contract.json"
                    frozen = read(freeze_path)
                    checkpoint, prediction = directory / "model/best.pt", directory / "model/predictions.csv.gz"
                    if frozen["input"]["active_label_count"] != budget:
                        raise RuntimeError("random budget mismatch")
                    if any(frozen["input"]["training_config"][k] != v for k, v in TRAINING_CONFIG.items()):
                        raise RuntimeError("random round training mismatch")
                    if r == 0 and frozen["input"]["L_t_ids_hash"] != ids_hash(l0):
                        raise RuntimeError("random L333 mismatch")
                else:
                    base = CW_SHORT if r <= 3 else CW_EXTENSION
                    directory = base / f"runtime/seed_{seed}/center_width_lcmd/round_{r:02d}"
                    freeze_path = directory / "round_freeze.json"
                    frozen = read(freeze_path)
                    checkpoint, prediction = Path(frozen["checkpoint_path"]), Path(frozen["prediction_path"])
                audit_path = checkpoint.parent / "fit_audit.json"
                audit = read(audit_path)
                if audit["initialization_hash"] != initial_hash or audit["train_rows"] != budget:
                    raise RuntimeError("control initialization/budget mismatch")
                if audit["validation_ids_hash"] != ids_hash(validation_ids):
                    raise RuntimeError("control validation mismatch")
                if r == 0 and audit["train_ids_hash"] != ids_hash(l0):
                    raise RuntimeError("control L333 mismatch")
                for kind, p in (("checkpoint", checkpoint), ("prediction", prediction)):
                    if sha256_file(p) != frozen[f"{kind}_sha256"] or sha256_file(p) != audit[f"{kind}_sha256"]:
                        raise RuntimeError("control artifact hash mismatch")
                for p in (checkpoint, prediction, freeze_path, audit_path):
                    sources[str(p.relative_to(ROOT))] = sha256_file(p)
                rows.append({"seed": seed, "method": method, "budget": budget,
                             "prediction_path": str(prediction.relative_to(ROOT)),
                             "checkpoint_path": str(checkpoint.relative_to(ROOT)),
                             "freeze_path": str(freeze_path.relative_to(ROOT)),
                             "initialization_hash": initial_hash, "L333_ids_hash": ids_hash(l0)})
    return {"entries": rows, "source_hashes": sources, "test_metrics_read": False, "refits": 0}


def prepare(config=None):
    config = config or settings()
    if (STUDY / "protocol.json").exists():
        if read(STUDY / "protocol.json")["transport"] != config:
            raise RuntimeError("transport cannot change after protocol freeze")
        return validate()
    STUDY.mkdir(parents=True, exist_ok=True)
    reuse = control_reuse()
    once(STUDY / "control_reuse.json", reuse)
    once(STUDY / "historical_artifacts.json", historical_manifest())
    record = {"study_version": VERSION, "split": "Row", "seeds": list(SEEDS),
        "methods": [*CONTROLS, *METHODS], "budgets": list(BUDGETS), "batch_size": 32,
        "selection_counts": dict(zip(METHODS, (16, 32))), "training": TRAINING_CONFIG,
        "retraining": "same initialization; from scratch; same optimizer and validation checkpoint rule",
        "primary_metric": "combined NRMSE using fixed L333 target scales; trapezoidal label-AULC 333..525",
        "secondary_metrics": ["V1/V2 RMSE MAE R2", "NRMSE_525", "per-seed AULC", "query/token cost"],
        "comparison": "each LLM arm against random32 and CW32; free versus hybrid is descriptive",
        "development_gate": "lower mean AULC and wins on both seeds versus both controls",
        "attribution": "LLM scientist package; chemistry/ML prior causal effects not isolated",
        "evidence_class": "two-seed development; historically exposed test; no significance claims",
        "transport": config, "query_budget": QUERY_BUDGET, "view_budget": VIEW_BUDGET,
        "model_call_budget": MODEL_CALL_BUDGET, "initial_cards": 20,
        "quotas": None, "automatic_fallback": False,
        "memory": "last LLM hypotheses/interpretation/rationale; last batch outcomes; exact top-8 error records; all selected IDs",
        "truth_barrier": "freeze all 32 IDs plus premeasurement predictions before reveal",
        "global_test_barrier": "both seeds x both LLM arms x all seven predictions before test reveal",
        "selector_boundary": "own L_t truth and U_t features/predictions only; no validation/test or other trajectories",
        "code_hashes": code_hashes(), "packages": package_versions(),
        "control_reuse_sha256": sha256_file(STUDY / "control_reuse.json"),
        "historical_manifest_sha256": sha256_file(STUDY / "historical_artifacts.json"),
        "prompt_sha256": stable_hash(SYSTEM_PROMPT)}
    once(STUDY / "protocol.json", record)
    prompt_path = STUDY / "selector_prompt.txt"
    if prompt_path.exists() and prompt_path.read_text() != SYSTEM_PROMPT:
        raise RuntimeError("prompt artifact changed")
    prompt_path.write_text(SYSTEM_PROMPT)
    return {"status": "PREPARED", "control_refits": 0, "new_training_runs_started": 0}


def validate():
    record = read(STUDY / "protocol.json")
    if record["code_hashes"] != code_hashes() or record["prompt_sha256"] != stable_hash(SYSTEM_PROMPT):
        raise RuntimeError("frozen code/prompt drift")
    if (STUDY / "selector_prompt.txt").read_text() != SYSTEM_PROMPT:
        raise RuntimeError("prompt file drift")
    for name, key in (("control_reuse.json", "control_reuse_sha256"),
                      ("historical_artifacts.json", "historical_manifest_sha256")):
        if sha256_file(STUDY / name) != record[key]:
            raise RuntimeError("frozen manifest drift")
    for relative, digest in read(STUDY / "control_reuse.json")["source_hashes"].items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"control input drift: {relative}")
    if historical_manifest() != read(STUDY / "historical_artifacts.json"):
        raise RuntimeError("historical artifacts modified")
    return {"status": "VALID", "protocol_sha256": stable_hash(record), "historical_artifacts_unchanged": True}


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
        self.protocol_hash = stable_hash(read(STUDY / "protocol.json"))
        once(self.runtime / "context.json", {"seed": seed, "protocol_hash": self.protocol_hash,
             "split_hash": stable_hash(self.partition.to_dict("list")), "preprocessing": self.preprocessing,
             "normalization": asdict(self.normalization), "initial_label_access": store.audit})

    def new_store(self):
        return RestrictedLabelStore(SOURCE_DATA, self.partition)


def selection_directory(seed, method, round_index):
    if seed not in SEEDS or method not in METHODS or round_index not in range(6):
        raise ValueError("unregistered seed/method/round")
    return STUDY / f"selections/seed_{seed}/{method}/round_{round_index:02d}"


def verify_files(directory, hashes):
    for name, digest in hashes.items():
        if sha256_file(directory / name) != digest:
            raise RuntimeError(f"frozen artifact drift: {name}")


def audit_batch(directory):
    frozen = read(directory / "batch_freeze.json")
    verify_files(directory, frozen["artifact_hashes"])
    packet = read(directory / "input.json")
    raw = read(directory / "catalog.json")
    catalog = ReadOnlyCatalog(**raw)
    catalog.initial_cards()
    config = read(STUDY / "protocol.json")["transport"]
    def unavailable(*args):
        raise RuntimeError("audit may only replay existing selector turns")
    result = run_selector(packet, catalog, directory, config, call=unavailable)
    expected = list(catalog.pending) + [c["id"] for c in result["choices"]]
    if frozen["batch_ids"] != expected or len(set(expected)) != 32:
        raise RuntimeError("batch/selection mismatch")
    lookup = {**catalog.candidates, **catalog.pending}
    if frozen["premeasurement_predictions"] != [{"candidate_id": i,
            "pred_V1_ml": lookup[i]["pred_V1_ml"], "pred_V2_ml": lookup[i]["pred_V2_ml"]} for i in expected]:
        raise RuntimeError("frozen predictions mismatch")
    return frozen


def state(context, method, round_index):
    labeled, unlabeled, truth = context.roles["l0"].copy(), context.roles["u0"].copy(), context.l0_truth.copy()
    history = []
    for r in range(round_index):
        directory = selection_directory(context.seed, method, r)
        frozen = audit_batch(directory)
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
    if len(labeled) != BUDGETS[round_index]:
        raise RuntimeError("active budget mismatch")
    return labeled, unlabeled, truth, history


def prediction_freeze(context, method, r, labeled):
    directory = context.runtime / method / f"round_{r:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    if r == 0:
        checkpoint, prediction, fit = _round0_evaluation(context)
    else:
        checkpoint, prediction = directory / "model/best.pt", directory / "model/predictions.csv.gz"
        fit = read(directory / "model/fit_audit.json")
        if fit["train_ids_hash"] != ids_hash(context.ids(labeled)):
            raise RuntimeError("predictor trained on wrong labels")
    record = {"checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
              "prediction_path": str(prediction), "prediction_sha256": sha256_file(prediction),
              "L_t_ids_hash": ids_hash(context.ids(labeled)), "initialization_hash": fit["initialization_hash"],
              "test_truth_access_count": 0}
    once(directory / "prediction_freeze.json", record)
    return checkpoint, directory


def make_catalog(context, method, r, labeled, unlabeled, truth, history, checkpoint, directory):
    current, n = np.r_[labeled, unlabeled], len(labeled)
    cw, _ = banks(context, current, checkpoint, r, directory)
    positions = lcmd_tp_select(cw[n:], cw[:n], 16).selected_pool_positions if method == METHODS[0] else []
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
           "salt": f"{VERSION}:{context.seed}:{method}:{r}"}
    return ReadOnlyCatalog(**raw), raw


def stage(seed, method, r=0):
    validate()
    selection_directory(seed, method, r)
    torch.set_num_threads(2)
    context = Context(seed)
    labeled, unlabeled, truth, history = state(context, method, r)
    checkpoint, runtime = prediction_freeze(context, method, r, labeled)
    directory = selection_directory(seed, method, r)
    directory.mkdir(parents=True, exist_ok=True)
    catalog, raw = make_catalog(context, method, r, labeled, unlabeled, truth, history, checkpoint, runtime)
    packet = {"study_version": VERSION, "seed": seed, "method": method, "round": r,
              "active_label_count": len(labeled), "selection_count": 16 if method == METHODS[0] else 32,
              "objective": "overall Row V1/V2 predictive accuracy after retraining",
              "memory": scientific_memory(history, seed, method), "pending_experiments": list(catalog.pending.values()),
              "candidate_overview": catalog.overview(), "initial_cards": catalog.initial_cards(),
              "observed_examples": sorted(catalog.observed.values(), key=catalog.key)[:12],
              "observed_count": len(catalog.observed), "observed_record_access": "get_observed/search_observed",
              "validation_test_records": 0}
    packet["packet_sha256"] = stable_hash(packet)
    once(directory / "input.json", packet)
    once(directory / "catalog.json", raw)
    once(directory / "contract.json", {"seed": seed, "method": method, "round": r,
         "protocol_hash": context.protocol_hash, "checkpoint_sha256": sha256_file(checkpoint),
         "L_t_ids": context.ids(labeled), "U_t_ids": context.ids(unlabeled),
         "input_sha256": sha256_file(directory / "input.json"), "catalog_sha256": sha256_file(directory / "catalog.json")})
    return {"status": "READY_FOR_FIRST_SELECTION" if r == 0 else "READY_FOR_SELECTION",
            "seed": seed, "method": method, "round": r, "packet": str(directory / "input.json"),
            "eligible_count": len(catalog.candidates), "pending_count": len(catalog.pending),
            "new_labels_revealed": 0, "validation_test_truth_access_count": 0, "llm_calls": 0, "fits": 0}


def select(seed, method, r=0):
    stage(seed, method, r)
    directory = selection_directory(seed, method, r)
    if (directory / "batch_freeze.json").exists():
        audit_batch(directory)
        return {"status": "ALREADY_FROZEN"}
    packet, raw = read(directory / "input.json"), read(directory / "catalog.json")
    catalog = ReadOnlyCatalog(**raw)
    catalog.initial_cards()
    result = run_selector(packet, catalog, directory, read(STUDY / "protocol.json")["transport"])
    validate_selection(result, catalog, packet)
    batch = list(catalog.pending) + [c["id"] for c in result["choices"]]
    if len(batch) != 32 or len(set(batch)) != 32 or not set(batch) <= set(read(directory / "contract.json")["U_t_ids"]):
        raise ValueError("invalid batch; no labels revealed")
    lookup = {**catalog.candidates, **catalog.pending}
    names = ["input.json", "catalog.json", "contract.json", "selection.json"]
    names += [p.name for p in sorted(directory.glob("turn_*.json"))]
    names += [p.name for p in sorted(directory.glob("query_*.json"))]
    once(directory / "batch_freeze.json", {"batch_ids": batch, "pending_ids": list(catalog.pending),
         "premeasurement_predictions": [{"candidate_id": i, "pred_V1_ml": lookup[i]["pred_V1_ml"],
              "pred_V2_ml": lookup[i]["pred_V2_ml"]} for i in batch],
         "artifact_hashes": {n: sha256_file(directory / n) for n in names},
         "viewed_candidate_ids": sorted(catalog.viewed), "new_batch_labels_revealed": False})
    return {"status": "BATCH_FROZEN_BEFORE_REVEAL", "batch_count": 32, "new_labels_revealed": 0}


def advance(seed, method, r=0):
    """Explicitly reveal one frozen batch and fit exactly one next-budget predictor."""
    validate()
    directory = selection_directory(seed, method, r)
    frozen = audit_batch(directory)
    context = Context(seed)
    labeled, unlabeled, truth, history = state(context, method, r)
    result = read(directory / "selection.json")
    batch = frozen["batch_ids"]
    all_ids = [row["candidate_id"] for h in history for row in h["records"]] + batch
    store = context.new_store()
    store.freeze_acquisitions(all_ids)
    if (directory / "feedback.json").exists():
        # state() verifies the existing feedback and receipt before any resumed fit.
        new_labeled, _, new_truth, _ = state(context, method, r+1)
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
        once(directory / "label_access_receipt.json", {"feedback_sha256": stable_hash(feedback),
             "batch_freeze_sha256": sha256_file(directory / "batch_freeze.json"), "audit": store.audit[-1]})
        once(directory / "feedback.json", feedback)
        new_labeled, _, new_truth, _ = state(context, method, r+1)
    context.validation_truth = store.reveal(context.ids(context.roles["validation"]), "initial_fit")
    next_dir = context.runtime / method / f"round_{r+1:02d}"
    torch.set_num_threads(2)
    context.fit(method, r+1, new_labeled, new_truth, next_dir / "model")
    prediction_freeze(context, method, r+1, new_labeled)
    if r == 5:
        hashes = {}
        for j in range(7):
            p = context.runtime / method / f"round_{j:02d}/prediction_freeze.json"
            hashes[str(p.relative_to(STUDY))] = sha256_file(p)
        for j in range(6):
            d = selection_directory(seed, method, j)
            for name in ("batch_freeze.json", "feedback.json", "label_access_receipt.json"):
                p = d / name
                hashes[str(p.relative_to(STUDY))] = sha256_file(p)
        once(context.runtime / method / "trajectory_freeze.json", {"seed": seed, "method": method,
             "selected_ids": all_ids, "final_active_labels": len(new_labeled), "files": hashes})
    return {"status": "ONE_BATCH_TRAINED", "active_label_count": len(new_labeled), "next_round": r+1,
            "next_selection_started": False}


def report():
    """Global audit must complete before this function opens any test labels."""
    validate()
    entries = {}
    for seed in SEEDS:
        for method in METHODS:
            trajectory = read(STUDY / f"runtime/seed_{seed}/{method}/trajectory_freeze.json")
            if trajectory["final_active_labels"] != 525 or len(set(trajectory["selected_ids"])) != 192:
                raise RuntimeError("incomplete trajectory")
            verify_files(STUDY, trajectory["files"])
            state(Context(seed), method, 6)
            for r in range(6):
                frozen = audit_batch(selection_directory(seed, method, r))
                if frozen["batch_ids"] != trajectory["selected_ids"][r*32:(r+1)*32]:
                    raise RuntimeError("trajectory selection mismatch")
            for r in range(7):
                p = STUDY / f"runtime/seed_{seed}/{method}/round_{r:02d}/prediction_freeze.json"
                rec = read(p)
                for kind in ("checkpoint", "prediction"):
                    if sha256_file(Path(rec[f"{kind}_path"])) != rec[f"{kind}_sha256"]:
                        raise RuntimeError("prediction/checkpoint drift")
                entries[str(p.relative_to(STUDY))] = sha256_file(p)
    if len(entries) != 28:
        raise RuntimeError("incomplete global prediction grid")
    once(STUDY / "global_pre_test_freeze.json", {"entries": entries})
    rows, audits = [], []
    reused = read(STUDY / "control_reuse.json")["entries"]
    for seed in SEEDS:
        context = Context(seed)
        store = context.new_store()
        store.freeze_acquisitions([])
        store.freeze_predictions()
        values = store.reveal(context.ids(context.roles["test"]), "final_test_evaluation")
        audits += [{"seed": seed, **a} for a in store.audit]
        for method in (*CONTROLS, *METHODS):
            for r, budget in enumerate(BUDGETS):
                if method in CONTROLS:
                    path = ROOT / next(e["prediction_path"] for e in reused if (e["seed"], e["method"], e["budget"]) == (seed, method, budget))
                else:
                    path = Path(read(STUDY / f"runtime/seed_{seed}/{method}/round_{r:02d}/prediction_freeze.json")["prediction_path"])
                pred = pd.read_csv(path)
                if pred.sample_id.astype(str).tolist() != context.ids(context.roles["test"]):
                    raise RuntimeError("test prediction alignment mismatch")
                rows.append({"seed": seed, "method": method, "budget": budget,
                    **metric_row(values, pred.drop(columns="sample_id").to_numpy(float), context.preprocessing["target_scales"])})
    frame = pd.DataFrame(rows)
    areas = [{"seed": seed, "method": method, "AULC_333_525": float(np.trapezoid(
        group.sort_values("budget").combined_normalized_RMSE, BUDGETS)/192)}
        for (seed, method), group in frame.groupby(["seed", "method"])]
    once(STUDY / "results.json", {"learning_curves": rows, "aulc": areas, "test_access": audits})
    return {"status": "REPORTED", "evidence": "two-seed development; no causal knowledge attribution"}
