"""Six-round CW16 + full-pool random/LLM active-learning experiment."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from ..artifacts import sha256_file
from ..models import load_predictor_checkpoint
from ..resources import ROOT, SOURCE_DATA, SOURCE_GRAPH_CACHE
from ..schemas.conditions import ConditionNormalization
from ..training.predictor import atomic_json
from .benchmark_protocol import TRAINING_CONFIG, load_features
from .benchmark_reporting import metric_row
from .cw_lcmd_extension_study import STUDY as CW_EXTENSION
from .full_pool_selector import (INITIAL_VIEW_COUNT, MODEL_CALL_BUDGET, PER_QUERY_LIMIT,
                                 PROMPT_VERSION, QUERY_BUDGET, SYSTEM_PROMPT, VIEW_BUDGET,
                                 ReadOnlyCatalog, configured, run_openai_selector,
                                 validate_selection)
from .gradient_transforms import center_width_transform
from .lcmd import lcmd_tp_select, _squared_distances
from .llm_screen import banks, rng, shortlist
from .maxdet_study import historical_round0_paths
from .protocol import RestrictedLabelStore, ids_hash, stable_hash, validate_row_protocol
from .runner import predict_outputs
from .short_sequential_runner import ShortSequentialContext, _round0_evaluation, _write_json_once
from .short_sequential_study import STUDY as CW_SHORT


STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_llm_full_pool_feedback_v1"
VERSION = "full_pool_feedback_v1"
SEEDS = (157, 6101)
METHODS = ("cw16_random16_full_pool", "cw16_llm16_full_pool")
BUDGETS = (333, 365, 397, 429, 461, 493, 525)
MODEL = "gpt-5.4"
REASONING_EFFORT = "medium"
CONDITION_FIELDS = ("PE/EA", "Density g/ml", "V/ul", "loading solvent", "Volume of loading solvent/ul")


def read(path):
    return json.loads(Path(path).read_text())


def code_hashes():
    relative = (
        "src/qgeognn_al/active_learning_v2/full_pool_selector.py",
        "src/qgeognn_al/active_learning_v2/full_pool_study.py",
        "scripts/studies/run_qgeognn_v2_row_llm_full_pool.py",
        "src/qgeognn_al/active_learning_v2/llm_screen.py",
        "src/qgeognn_al/active_learning_v2/short_sequential_runner.py",
        "src/qgeognn_al/active_learning_v2/runner.py",
        "src/qgeognn_al/active_learning_v2/protocol.py",
        "src/qgeognn_al/active_learning_v2/gradient_features.py",
        "src/qgeognn_al/active_learning_v2/gradient_transforms.py",
        "src/qgeognn_al/active_learning_v2/lcmd.py",
        "src/qgeognn_al/training/predictor.py",
        "src/qgeognn_al/evaluation/point.py",
    )
    return {name: sha256_file(ROOT / name) for name in relative}


def _source_hashes():
    paths = [SOURCE_DATA, SOURCE_GRAPH_CACHE,
             CW_SHORT / "protocol.json", CW_EXTENSION / "protocol.json",
             CW_EXTENSION / "results/learning_curve_metrics.csv"]
    for seed in SEEDS:
        paths += [CW_SHORT / "splits" / f"row_seed_{seed}.csv",
                  historical_round0_paths(seed)["member0_model"],
                  historical_round0_paths(seed)["member0_predictions"]]
        for round_index in range(4):
            paths.append(CW_SHORT / "runtime" / f"seed_{seed}/center_width_lcmd/round_{round_index:02d}/round_freeze.json")
        for round_index in range(3, 7):
            paths.append(CW_EXTENSION / "runtime" / f"seed_{seed}/center_width_lcmd/round_{round_index:02d}/round_freeze.json")
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def audit_cw_reuse():
    result = []
    historical = pd.read_csv(CW_EXTENSION / "results/learning_curve_metrics.csv")
    for seed in SEEDS:
        split = pd.read_csv(CW_SHORT / "splits" / f"row_seed_{seed}.csv")
        validate_row_protocol(split)
        l0_ids = split.loc[split.role.eq("l0"), "sample_id"].astype(str).tolist()
        expected = historical.loc[historical.outer_seed.eq(seed) & historical.method.eq("center_width_lcmd")]
        if sorted(expected.active_label_count.tolist()) != list(BUDGETS):
            raise RuntimeError("CW historical budget grid incomplete")
        frozen = []
        for round_index, budget in enumerate(BUDGETS):
            base = CW_SHORT if round_index <= 3 else CW_EXTENSION
            path = base / f"runtime/seed_{seed}/center_width_lcmd/round_{round_index:02d}/round_freeze.json"
            record = read(path)
            for key in ("checkpoint", "prediction"):
                if sha256_file(Path(record[f"{key}_path"])) != record[f"{key}_sha256"]:
                    raise RuntimeError("CW historical artifact hash mismatch")
            frozen.append(str(path.relative_to(ROOT)))
        source_protocol = read(CW_SHORT / "protocol.json")
        if source_protocol["training"] != TRAINING_CONFIG:
            raise RuntimeError("CW training configuration drift")
        result.append({"seed": seed, "L333_ids_hash": ids_hash(l0_ids),
                       "split_sha256": sha256_file(CW_SHORT / "splits" / f"row_seed_{seed}.csv"),
                       "reused_round_freezes": frozen, "matched_budgets": list(BUDGETS)})
    return result


def protocol_record():
    return {
        "study_version": VERSION, "status": "FROZEN_BEFORE_NEW_SELECTION",
        "research_question": "incremental generalization and label efficiency of CW16 plus full-pool response-feedback LLM16",
        "split": "Row", "seeds": list(SEEDS),
        "methods": ["center_width_lcmd", *METHODS], "budgets": list(BUDGETS),
        "batch": {"CW": 16, "supplement": 16, "total": 32},
        "random_scope": "all own current U_t minus pending CW16, uniform without replacement",
        "llm_scope": "all own current U_t minus pending CW16; 128 numerical IDs are reference only",
        "llm_model": MODEL, "reasoning_effort": REASONING_EFFORT,
        "selector_settings": {"prompt_version": PROMPT_VERSION, "model_calls_max": MODEL_CALL_BUDGET,
                              "catalog_queries_max": QUERY_BUDGET, "distinct_candidate_views_max": VIEW_BUDGET,
                              "per_query_limit": PER_QUERY_LIMIT, "initial_reference_cards": INITIAL_VIEW_COUNT,
                              "provider": "OpenAI Responses API", "store": False,
                              "temperature": "provider default", "automatic_model_fallback": False},
        "training": TRAINING_CONFIG, "retraining": "same initialization, from scratch at each budget",
        "truth_barrier": "freeze all 32 IDs and their measurement-time predictions before revealing either half",
        "context_boundary": "fresh stateless request per seed/method/round with only current trajectory L_t, its earlier reasons/hypotheses/feedback and current feature-only candidate catalog; no development chat",
        "observation_units": {"V1_ml": "mL", "V2_ml": "mL", "pred_V1_ml": "mL", "pred_V2_ml": "mL",
                              "V2_minus_V1": "mL elution width, not uncertainty"},
        "metric": "combined NRMSE fixed L333 target scales; trapezoidal label-AULC 333-525",
        "test_barrier": "both new methods x both seeds x all 7 prediction budgets frozen before new test evaluation",
        "development_evidence": True, "no_statistical_significance_claim": True,
        "no_posthoc_selector_change": True, "old_failure_standard_preserved": True,
        "code_hashes": code_hashes(), "source_hashes": _source_hashes(),
    }


def prepare():
    if (STUDY / "protocol.json").exists():
        return validate()
    audit = audit_cw_reuse()
    STUDY.mkdir(parents=True, exist_ok=True)
    (STUDY / "splits").mkdir(exist_ok=True)
    for seed in SEEDS:
        shutil.copy2(CW_SHORT / "splits" / f"row_seed_{seed}.csv", STUDY / "splits" / f"row_seed_{seed}.csv")
    atomic_json(STUDY / "cw_reuse_audit.json", audit)
    atomic_json(STUDY / "protocol.json", protocol_record())
    (STUDY / "selector_prompt.txt").write_text(SYSTEM_PROMPT)
    return {"status": "PREPARED", "CW_reused": True, "new_fit_count": 24,
            "LLM_decisions": 12, "selector_configured": configured()}


def refreeze_preselection():
    """Archive an engineering dry run, only before any scientific batch freezes."""
    if list((STUDY / "selections").glob("**/batch_freeze.json")):
        raise RuntimeError("cannot refreeze after a scientific batch freeze")
    revision = STUDY / "revisions" / "preselection_draft"
    if revision.exists():
        raise RuntimeError("preselection draft already archived")
    revision.mkdir(parents=True)
    for name in ("protocol.json", "selector_prompt.txt", "cw_reuse_audit.json", "selections", "runtime"):
        path = STUDY / name
        if path.exists():
            shutil.move(str(path), str(revision / name))
    return prepare()


def validate():
    record = read(STUDY / "protocol.json")
    if record != protocol_record() or (STUDY / "selector_prompt.txt").read_text() != SYSTEM_PROMPT:
        raise RuntimeError("frozen protocol/code/input/prompt drift")
    for seed in SEEDS:
        if sha256_file(STUDY / "splits" / f"row_seed_{seed}.csv") != sha256_file(CW_SHORT / "splits" / f"row_seed_{seed}.csv"):
            raise RuntimeError("split drift")
    return {"status": "VALID", "protocol_hash": stable_hash(record)}


class Context(ShortSequentialContext):
    def __init__(self, seed):
        self.seed, self.study = int(seed), STUDY
        self.runtime = STUDY / "runtime" / f"seed_{seed}"
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.partition = pd.read_csv(STUDY / "splits" / f"row_seed_{seed}.csv")
        validate_row_protocol(self.partition)
        self.data = load_features()
        if self.partition.sample_id.astype(str).tolist() != self.data.sample_id.astype(str).tolist():
            raise RuntimeError("split/source order drift")
        self.roles = {role: self.partition.loc[self.partition.role.eq(role), "canonical_index"].to_numpy(int)
                      for role in ("l0", "u0", "validation", "test")}
        old = read(historical_round0_paths(seed)["context"])["contract"]
        self.preprocessing = old["preprocessing"]
        self.normalization = ConditionNormalization(**old["normalization"])
        self.atom, self.angle = torch.load(historical_round0_paths(seed)["scrubbed_graphs"], weights_only=False)
        if any(torch.count_nonzero(item.y) for item in self.atom):
            raise RuntimeError("scrubbed graphs contain labels")
        store = RestrictedLabelStore(SOURCE_DATA, self.partition)
        self.l0_truth = store.reveal(self.ids(self.roles["l0"]), "initial_fit")
        self.validation_truth = store.reveal(self.ids(self.roles["validation"]), "initial_fit")
        self.cw_transform, self.cw_audit = center_width_transform(self.l0_truth)
        self.protocol_hash = stable_hash(read(STUDY / "protocol.json"))
        _write_json_once(self.runtime / "context.json", {"study_version": VERSION, "seed": seed,
                          "protocol_hash": self.protocol_hash,
                          "split_hash": stable_hash(self.partition.to_dict("list")),
                          "L333_ids_hash": ids_hash(self.ids(self.roles["l0"])),
                          "preprocessing": self.preprocessing, "normalization": asdict(self.normalization)})


def _card(context, index, pred=None, distance=None):
    row = context.data.iloc[int(index)]
    card = {"id": str(row.sample_id), "smiles": str(row.canonical_smiles),
            "conditions": {key: row[key].item() if hasattr(row[key], "item") else row[key]
                           for key in CONDITION_FIELDS}}
    if pred is not None:
        card["pred_V1_ml"] = float(pred[1])
        card["pred_V2_ml"] = float(pred[4])
    if distance is not None:
        card["cw_distance_percentile"] = round(float(distance), 5)
    return card


def _observed(context, labeled, truth, feedback):
    earlier = {row["candidate_id"]: row for item in feedback for row in item["records"]}
    rows = []
    for index, values in zip(labeled, truth):
        card = _card(context, index)
        card.update({"V1_ml": float(values[0]), "V2_ml": float(values[1]),
                     "record_kind": "acquired_with_premeasurement_prediction" if card["id"] in earlier else "initial_L333"})
        if card["id"] in earlier:
            prior = earlier[card["id"]]
            card["premeasurement_prediction_V1_ml"] = prior["pred_V1_ml"]
            card["premeasurement_prediction_V2_ml"] = prior["pred_V2_ml"]
            card["premeasurement_error_V1_ml"] = prior["error_V1_ml"]
            card["premeasurement_error_V2_ml"] = prior["error_V2_ml"]
        rows.append(card)
    return rows


def _catalog(context, labeled, unlabeled, truth, feedback, model_path, cw, reference_positions):
    current = np.r_[labeled, unlabeled]
    n = len(labeled)
    cw_pending = lcmd_tp_select(cw[n:], cw[:n], 16).selected_pool_positions + n
    pending_ids = set(context.ids(current[cw_pending]))
    preds, order = predict_outputs(load_predictor_checkpoint(model_path), context.atom, context.angle, unlabeled)
    if not np.array_equal(order, unlabeled):
        raise RuntimeError("pool prediction ID alignment failed")
    distances = np.sqrt(_squared_distances(cw[n:], cw[:n]).min(axis=1))
    percentile = pd.Series(distances).rank(method="average", pct=True).to_numpy()
    cards = [_card(context, idx, pred, percentile[i]) for i, (idx, pred) in enumerate(zip(unlabeled, preds))]
    by_id = {card["id"]: card for card in cards}
    pending = [by_id[value] for value in context.ids(current[cw_pending])]
    candidates = [card for card in cards if card["id"] not in pending_ids]
    observed = _observed(context, labeled, truth, feedback)
    ref_ids = [context.ids([current[pos]])[0] for pos in reference_positions]
    return ReadOnlyCatalog(candidates, observed, pending, ref_ids), cards


def _feedback_records(batch, cards, observed_values, choices):
    lookup = {card["id"]: card for card in cards}
    reasons = {choice["id"]: choice for choice in choices}
    records = []
    for position, (candidate_id, truth) in enumerate(zip(batch, observed_values)):
        pred = lookup[candidate_id]
        selected = reasons.get(candidate_id, {})
        records.append({"candidate_id": candidate_id, "source": "CW" if position < 16 else "LLM" if choices else "random",
                        "pred_V1_ml": pred["pred_V1_ml"], "pred_V2_ml": pred["pred_V2_ml"],
                        "true_V1_ml": float(truth[0]), "true_V2_ml": float(truth[1]),
                        "error_V1_ml": float(pred["pred_V1_ml"] - truth[0]),
                        "error_V2_ml": float(pred["pred_V2_ml"] - truth[1]),
                        "reason": selected.get("reason", "CW geometric coverage" if position < 16 else "uniform full-pool random"),
                        "hypothesis_id": selected.get("hypothesis_id")})
    return records


def _acquire(context, method, round_index, labeled, unlabeled, truth, model_path, directory,
             prior_feedback, *, retry_selector=False):
    selection_dir = STUDY / "selections" / f"seed_{context.seed}" / method / f"round_{round_index:02d}"
    selection_dir.mkdir(parents=True, exist_ok=True)
    current = np.r_[labeled, unlabeled]
    n = len(labeled)
    cw, raw = banks(context, current, model_path, round_index, directory)
    numeric = shortlist(cw, raw, n, context.seed, round_index)
    catalog, cards = _catalog(context, labeled, unlabeled, truth, prior_feedback, model_path,
                              cw, numeric["candidate_positions"])
    pending_ids = [card["id"] for card in catalog.pending.values()]
    if len(pending_ids) != 16 or len(catalog.candidates) != len(unlabeled) - 16:
        raise RuntimeError("CW/full-pool cardinality mismatch")
    history = []
    for previous in prior_feedback:
        history.append({"round": previous["round"], "selected_choices": previous.get("choices", []),
                        "hypotheses": previous.get("hypotheses", []), "measured_feedback": previous["records"]})
    observed = list(catalog.observed.values())
    overview = catalog.overview()
    packet = {"study_version": VERSION, "seed": context.seed, "method": method,
              "round": round_index, "active_label_count": n, "objective": "reduce fixed-denominator combined V1/V2 NRMSE",
              "units": {"V1": "mL", "V2": "mL", "V2_minus_V1": "mL elution width, not uncertainty"},
              "L_t_summary": {"count": len(observed), "V1_min_max": [min(v["V1_ml"] for v in observed), max(v["V1_ml"] for v in observed)],
                              "V2_min_max": [min(v["V2_ml"] for v in observed), max(v["V2_ml"] for v in observed)]},
              "L_t_example_records": observed[:24], "L_t_observed_ids": [v["id"] for v in observed],
              "L_t_full_record_access": "get_observed/search_observed tools",
              "trajectory_history": history, "pending_CW16": list(catalog.pending.values()),
              "candidate_overview": overview,
              "numeric_reference_initial_cards": catalog.initial_cards() if method == METHODS[1] else [],
              "numerical_references_only": True, "observed_validation_test_records": 0}
    packet["packet_sha256"] = stable_hash(packet)
    contract = {"protocol_hash": context.protocol_hash, "checkpoint_sha256": sha256_file(model_path),
                "L_t_ids": context.ids(labeled), "U_t_ids": context.ids(unlabeled)}
    _write_json_once(selection_dir / "contract.json", contract)
    _write_json_once(selection_dir / "input.json", packet)
    _write_json_once(selection_dir / "observed_records.json", {"records": observed})
    _write_json_once(selection_dir / "numeric_reference.json", {"proposal": numeric,
                    "candidate_ids": catalog.reference_ids, "pending_CW16_ids": pending_ids})
    frozen_path = selection_dir / "batch_freeze.json"
    if frozen_path.exists():
        frozen = read(frozen_path)
        if frozen["contract_sha256"] != sha256_file(selection_dir / "contract.json"):
            raise RuntimeError("frozen batch contract drift")
        return frozen["batch_ids"], cards, frozen.get("choices", []), frozen.get("hypotheses", []), selection_dir
    if method == METHODS[0]:
        pool = list(catalog.candidates)
        picks = rng(context.seed, round_index, 36).choice(pool, 16, replace=False).tolist()
        choices, hypotheses = [], []
    else:
        result_path = selection_dir / "validated_selection.json"
        if result_path.exists():
            result = read(result_path)
        else:
            attempts = sorted(selection_dir.glob("selector_attempt_*.json"))
            completed = [read(path) for path in attempts if read(path).get("final_text")]
            if completed:
                raw_response = json.loads(completed[-1]["final_text"])
            elif not configured():
                return {"status": "AWAITING_SELECTOR_CONFIG", "seed": context.seed, "method": method,
                        "round": round_index, "required": "OPENAI_API_KEY for the frozen OpenAI Responses API model",
                        "input": str(selection_dir / "input.json")}
            elif attempts and not retry_selector:
                return {"status": "SELECTOR_PAUSED", "reason": "incomplete/failed attempt; pass --retry-selector",
                        "attempt": str(attempts[-1])}
            else:
                attempt = len(attempts)
                raw_response = run_openai_selector(packet, catalog, selection_dir, MODEL, REASONING_EFFORT, attempt)
            result = validate_selection(raw_response, catalog, packet_sha256=packet["packet_sha256"],
                                        require_feedback=round_index > 0)
            _write_json_once(result_path, result)
        choices, hypotheses = result["choices"], result["hypotheses"]
        picks = [choice["id"] for choice in choices]
    batch = pending_ids + picks
    if len(batch) != len(set(batch)) or len(batch) != 32 or not set(batch) <= set(context.ids(unlabeled)):
        raise RuntimeError("invalid batch; no labels revealed")
    prediction_lookup = {card["id"]: card for card in cards}
    frozen_predictions = [{"candidate_id": value, "pred_V1_ml": prediction_lookup[value]["pred_V1_ml"],
                           "pred_V2_ml": prediction_lookup[value]["pred_V2_ml"]} for value in batch]
    _write_json_once(selection_dir / "batch_freeze.json", {"contract_sha256": sha256_file(selection_dir / "contract.json"),
                     "batch_ids": batch, "CW16_ids": pending_ids, "supplement16_ids": picks,
                     "premeasurement_predictions": frozen_predictions, "choices": choices,
                     "hypotheses": hypotheses,
                     "feedback_interpretation": result.get("feedback_interpretation", "") if method == METHODS[1] else "",
                     "numeric_reference_overlap": len(set(picks) & set(catalog.reference_ids)),
                     "viewed_candidate_ids": sorted(catalog.viewed) if method == METHODS[1] else [],
                     "eligible_candidate_count": len(catalog.candidates), "new_batch_labels_revealed": False})
    return batch, cards, choices, hypotheses, selection_dir


def execute(seed, method, *, retry_selector=False):
    if seed not in SEEDS or method not in METHODS:
        raise ValueError("unregistered seed/method")
    validate()
    if (STUDY / "global_pre_test_freeze.json").exists():
        raise RuntimeError("all new trajectories already globally frozen")
    torch.set_num_threads(2)
    context = Context(seed)
    store = context.new_store()
    labeled, unlabeled, truth = context.roles["l0"].copy(), context.roles["u0"].copy(), context.l0_truth.copy()
    all_selected, feedback_history, prediction_freezes = [], [], []
    for round_index, budget in enumerate(BUDGETS):
        if len(labeled) != budget:
            raise RuntimeError("budget drift")
        directory = context.runtime / method / f"round_{round_index:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        _write_json_once(directory / "state.json", {"labeled_ids": context.ids(labeled),
                         "unlabeled_ids": context.ids(unlabeled)})
        if round_index == 0:
            model_path, prediction_path, fit = _round0_evaluation(context)
        else:
            print(json.dumps({"event": "fit_start", "seed": seed, "method": method,
                              "budget": budget}), flush=True)
            fit = context.fit(method, round_index, labeled, truth, directory / "model")
            model_path, prediction_path = directory / "model/best.pt", directory / "model/predictions.csv.gz"
        _write_json_once(directory / "prediction_freeze.json", {
            "checkpoint_path": str(model_path), "checkpoint_sha256": sha256_file(model_path),
            "prediction_path": str(prediction_path), "prediction_sha256": sha256_file(prediction_path),
            "L_t_ids_hash": ids_hash(context.ids(labeled)), "initialization_hash": fit["initialization_hash"],
            "test_truth_access_count": 0})
        prediction_freezes.append(str(directory / "prediction_freeze.json"))
        if round_index == 6:
            break
        acquired = _acquire(context, method, round_index, labeled, unlabeled, truth, model_path,
                            directory, feedback_history, retry_selector=retry_selector)
        if isinstance(acquired, dict):
            return acquired
        batch, cards, choices, hypotheses, selection_dir = acquired
        frozen = read(selection_dir / "batch_freeze.json")
        all_selected += batch
        store.freeze_acquisitions(all_selected)
        feedback_path = selection_dir / "feedback.json"
        if feedback_path.exists():
            feedback = read(feedback_path)
            if feedback["batch_freeze_sha256"] != sha256_file(selection_dir / "batch_freeze.json"):
                raise RuntimeError("feedback/freeze mismatch")
            if not (selection_dir / "label_access_receipt.json").exists():
                raise RuntimeError("feedback exists without label-access receipt")
            values = np.asarray([[row["true_V1_ml"], row["true_V2_ml"]] for row in feedback["records"]], dtype=np.float32)
        else:
            values = store.reveal(batch, "after_acquisition_fit")
            _write_json_once(selection_dir / "label_access_receipt.json", {
                "batch_freeze_sha256": sha256_file(selection_dir / "batch_freeze.json"),
                "audit": store.audit[-1]})
            feedback = {"round": round_index, "batch_freeze_sha256": sha256_file(selection_dir / "batch_freeze.json"),
                        "choices": choices, "hypotheses": hypotheses,
                        "feedback_interpretation": frozen.get("feedback_interpretation", ""),
                        "records": _feedback_records(batch, cards, values, choices)}
            _write_json_once(feedback_path, feedback)
        feedback_history.append(feedback)
        lookup = dict(zip(context.ids(unlabeled), unlabeled))
        selected_indices = np.asarray([lookup[value] for value in batch], dtype=int)
        labeled = np.r_[labeled, selected_indices]
        truth = np.vstack([truth, values])
        excluded = set(selected_indices)
        unlabeled = np.asarray([value for value in unlabeled if value not in excluded], dtype=int)
    _write_json_once(context.runtime / method / "trajectory_freeze.json", {
        "status": "FROZEN_BEFORE_NEW_TEST_TRUTH", "seed": seed, "method": method,
        "prediction_freezes": prediction_freezes, "selected_ids": all_selected,
        "final_active_labels": len(labeled), "test_truth_access_count": 0})
    return {"status": "TRAJECTORY_COMPLETE", "seed": seed, "method": method,
            "final_active_labels": len(labeled)}


def report():
    """Open test truth only after every new trajectory and prediction has frozen."""
    validate()
    entries = {}
    for seed in SEEDS:
        for method in METHODS:
            trajectory = read(STUDY / f"runtime/seed_{seed}/{method}/trajectory_freeze.json")
            if trajectory["final_active_labels"] != 525 or len(trajectory["selected_ids"]) != 192:
                raise RuntimeError("incomplete trajectory")
            for path in trajectory["prediction_freezes"]:
                record = read(path)
                if any(sha256_file(Path(record[f"{kind}_path"])) != record[f"{kind}_sha256"]
                       for kind in ("checkpoint", "prediction")):
                    raise RuntimeError("prediction/checkpoint hash mismatch")
                entries[path] = sha256_file(Path(path))
            for round_index in range(6):
                selection = STUDY / f"selections/seed_{seed}/{method}/round_{round_index:02d}"
                freeze = read(selection / "batch_freeze.json")
                if freeze["batch_ids"] != trajectory["selected_ids"][round_index*32:(round_index+1)*32]:
                    raise RuntimeError("selection lineage mismatch")
                if read(selection / "feedback.json")["batch_freeze_sha256"] != sha256_file(selection / "batch_freeze.json"):
                    raise RuntimeError("feedback lineage mismatch")
                receipt = read(selection / "label_access_receipt.json")
                if (receipt["batch_freeze_sha256"] != sha256_file(selection / "batch_freeze.json")
                        or not receipt["audit"]["acquisitions_frozen"]
                        or receipt["audit"]["purpose"] != "after_acquisition_fit"):
                    raise RuntimeError("label reveal did not follow batch freeze")
    if len(entries) != 28:
        raise RuntimeError("incomplete global prediction grid")
    _write_json_once(STUDY / "global_pre_test_freeze.json", {"status": "FROZEN_BEFORE_TEST_TRUTH", "entries": entries})
    rows, test_access = [], []
    for seed in SEEDS:
        context = Context(seed)
        store = context.new_store()
        store.freeze_acquisitions([])
        store.freeze_predictions()
        test_truth = store.reveal(context.ids(context.roles["test"]), "final_test_evaluation")
        test_access += [{"seed": seed, **row} for row in store.audit]
        for method in ("center_width_lcmd", *METHODS):
            for round_index, budget in enumerate(BUDGETS):
                if method == "center_width_lcmd":
                    base = CW_SHORT if round_index <= 3 else CW_EXTENSION
                    freeze = read(base / f"runtime/seed_{seed}/center_width_lcmd/round_{round_index:02d}/round_freeze.json")
                    path = freeze["prediction_path"]
                else:
                    path = read(context.runtime / method / f"round_{round_index:02d}/prediction_freeze.json")["prediction_path"]
                predictions = pd.read_csv(path)
                if predictions.sample_id.astype(str).tolist() != context.ids(context.roles["test"]):
                    raise RuntimeError("test prediction ID alignment mismatch")
                rows.append({"seed": seed, "method": method, "budget": budget,
                             **metric_row(test_truth, predictions.drop(columns="sample_id").to_numpy(float),
                                          context.preprocessing["target_scales"])})
    result_dir = STUDY / "results"
    result_dir.mkdir(exist_ok=True)
    curves = pd.DataFrame(rows)
    curves.to_csv(result_dir / "learning_curves.csv", index=False)
    pd.DataFrame(test_access).to_csv(result_dir / "test_label_access_audit.csv", index=False)
    areas = []
    for (seed, method), group in curves.groupby(["seed", "method"]):
        group = group.sort_values("budget")
        areas.append({"seed": seed, "method": method,
                      "label_AULC_333_525": float(np.trapz(group.combined_normalized_RMSE, group.budget) / 192),
                      "NRMSE_525": float(group.iloc[-1].combined_normalized_RMSE)})
    pd.DataFrame(areas).to_csv(result_dir / "aulc.csv", index=False)
    diagnostics, costs, cases = [], [], []
    feature_lookup = load_features()
    feature_lookup["sample_id"] = feature_lookup.sample_id.astype(str)
    feature_lookup = feature_lookup.set_index("sample_id")
    for seed in SEEDS:
        for method in METHODS:
            for round_index in range(6):
                selection = STUDY / f"selections/seed_{seed}/{method}/round_{round_index:02d}"
                freeze = read(selection / "batch_freeze.json")
                before = read(selection / "observed_records.json")["records"]
                feedback = read(selection / "feedback.json")
                before_smiles = {row["smiles"] for row in before}
                before_conditions = {(row["smiles"], stable_hash(row["conditions"])) for row in before}
                picked = [feature_lookup.loc[value] for value in freeze["batch_ids"]]
                picked_smiles = [str(row.canonical_smiles) for row in picked]
                new_molecules = sum(value not in before_smiles for value in picked_smiles)
                condition_contrasts = sum((str(row.canonical_smiles) in before_smiles and
                                           (str(row.canonical_smiles), stable_hash({key: row[key].item() if hasattr(row[key], "item") else row[key]
                                                                                   for key in CONDITION_FIELDS})) not in before_conditions)
                                          for row in picked)
                diagnostics.append({"seed": seed, "method": method, "round": round_index,
                                    "budget_before": BUDGETS[round_index], "new_molecule_rows": new_molecules,
                                    "existing_molecule_new_condition_rows": condition_contrasts,
                                    "within_batch_same_molecule_pairs": sum(picked_smiles.count(value) *
                                        (picked_smiles.count(value) - 1) // 2 for value in set(picked_smiles)),
                                    "numeric_reference_overlap": freeze["numeric_reference_overlap"],
                                    "eligible_full_pool_count": freeze["eligible_candidate_count"],
                                    "LLM_viewed_candidate_count": len(freeze["viewed_candidate_ids"]),
                                    "LLM_outside_128_count": len(set(freeze["supplement16_ids"]) -
                                                                 set(read(selection / "numeric_reference.json")["candidate_ids"]))
                                    if method == METHODS[1] else 0,
                                    "unique_molecules_after_batch": len(before_smiles | set(picked_smiles)),
                                    "unique_condition_tuples_after_batch": len(before_conditions |
                                        {(str(row.canonical_smiles), stable_hash({key: row[key].item() if hasattr(row[key], "item") else row[key]
                                                                                  for key in CONDITION_FIELDS})) for row in picked})})
                for hypothesis in feedback.get("hypotheses", []):
                    cases.append({"seed": seed, "round": round_index, "hypothesis_id": hypothesis["id"],
                                  "content": hypothesis["content"],
                                  "supporting_observed_ids": json.dumps(hypothesis["supporting_observed_ids"]),
                                  "contradicting_observed_ids": json.dumps(hypothesis["contradicting_observed_ids"]),
                                  "evidence_strength": hypothesis["evidence_strength"],
                                  "feedback_interpretation": feedback.get("feedback_interpretation", "")})
            seconds = 0.0
            for round_index in range(1, 7):
                audit = read(STUDY / f"runtime/seed_{seed}/{method}/round_{round_index:02d}/model/fit_audit.json")
                seconds += float(audit["training_seconds"])
            tokens_in = tokens_out = calls = 0
            if method == METHODS[1]:
                for path in (STUDY / f"selections/seed_{seed}/{method}").glob("round_*/selector_attempt_*.json"):
                    transcript = read(path)
                    tokens_in += transcript["usage"]["input_tokens"]
                    tokens_out += transcript["usage"]["output_tokens"]
                    calls += len(transcript["turns"])
            costs.append({"seed": seed, "method": method, "new_training_fits": 6,
                          "new_training_seconds": seconds, "LLM_api_calls": calls,
                          "LLM_input_tokens": tokens_in, "LLM_output_tokens": tokens_out})
    pd.DataFrame(diagnostics).to_csv(result_dir / "selection_diagnostics.csv", index=False)
    pd.DataFrame(costs).to_csv(result_dir / "costs.csv", index=False)
    pd.DataFrame(cases).to_csv(result_dir / "hypothesis_history.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    colors = {"center_width_lcmd": "#3669a7", METHODS[0]: "#da8a27", METHODS[1]: "#258563"}
    for axis, seed in zip(axes, SEEDS):
        for method in colors:
            group = curves.loc[curves.seed.eq(seed) & curves.method.eq(method)].sort_values("budget")
            axis.plot(group.budget, group.combined_normalized_RMSE, marker="o", color=colors[method], label=method)
        axis.set(title=f"Seed {seed}", xlabel="Active labels", ylabel="Combined NRMSE")
        axis.grid(alpha=.25)
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    (STUDY / "figures").mkdir(exist_ok=True)
    figure.savefig(STUDY / "figures/learning_curves.png", dpi=180)
    plt.close(figure)
    area_frame = pd.DataFrame(areas)
    pivot = area_frame.pivot(index="seed", columns="method", values="label_AULC_333_525")
    comparisons = []
    for control in ("center_width_lcmd", METHODS[0]):
        delta = pivot[METHODS[1]] - pivot[control]
        comparisons.append({"control": control, "per_seed_LLM_minus_control": delta.to_dict(),
                            "mean_difference": float(delta.mean()), "wins_of_two": int((delta < 0).sum())})
    report_lines = ["# Full-pool feedback LLM active learning: development results", "",
                    "Two development seeds; Row split; lower combined NRMSE and label-AULC are better.",
                    "All new trajectories and prediction files were frozen before this test evaluation.",
                    "", "## Main result", "", area_frame.to_markdown(index=False), "",
                    "![Per-seed learning curves](figures/learning_curves.png)", "",
                    "## Paired comparisons", "", pd.DataFrame(comparisons).to_markdown(index=False), "",
                    "## L525 outcome", "", curves.loc[curves.budget.eq(525)].to_markdown(index=False), "",
                    "## Selection diagnosis", "", pd.DataFrame(diagnostics).to_markdown(index=False), "",
                    "## Costs", "", pd.DataFrame(costs).to_markdown(index=False), "",
                    "The hypothesis history is in `results/hypothesis_history.csv`; its interpretation is model-generated",
                    "and must be checked against the cited measured records before treating a mechanism as supported.",
                    "This version jointly changed pool access, observed responses, and context management.",
                    "The two seed mean is descriptive, not evidence of statistical significance.",
                    "The old 128-candidate screen and its failure standard remain unchanged.",
                    "The next validation is a same-interface, same-budget ablation of real-response feedback; "
                    "then consider identity masking with its concurrent input changes and an independent larger seed set."]
    (STUDY / "FINAL_REPORT.md").write_text("\n".join(report_lines) + "\n")
    return {"status": "REPORTED", "AULC": areas, "comparisons": comparisons}
