"""Chemistry + ML advisers -> free planner, with the shared Scientist lifecycle."""
from __future__ import annotations

from pathlib import Path
import sys
import json
import time
import shutil
from contextlib import contextmanager

import numpy as np
import torch

from ..artifacts import sha256_file
from ..resources import ROOT
from . import scientist_study as scientist
from . import scientist_lifecycle as lifecycle
from .benchmark_protocol import TRAINING_CONFIG, package_versions
from .dialog_bridge import once, read
from .protocol import stable_hash
from .protocol import ids_hash
from .scientist_full_delivery import encode_catalog
from .scientist_selector import ReadOnlyCatalog
from .scientist_transport import FULL_SELECTION_ATTEMPTS, check_transport
from .dual_expert_selector import METHOD, VERSION, FULL_ROLE_PROMPTS, trajectory_memory
from .dual_expert_transport import run_agent, settings

ARCHIVE_STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_llm_dual_expert_v1"
DEFAULT_REVISION = "full_pool_feedback_isolated_20261010"
STUDY = ARCHIVE_STUDY / "revisions" / DEFAULT_REVISION
SEEDS = scientist.SEEDS
METHODS = (METHOD,)
CONTROLS = scientist.CONTROLS
BUDGETS = tuple(range(333, 814, 32))


def configure_revision(revision=None):
    global STUDY
    name = revision or DEFAULT_REVISION
    if Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("revision must be a single directory name")
    STUDY = ARCHIVE_STUDY / "revisions" / name
    return STUDY


def code_hashes():
    paths = list((ROOT / "src/qgeognn_al").rglob("*.py"))
    paths += [ROOT / "scripts/studies/run_qgeognn_v2_row_llm_dual_expert.py"]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)}


def control_reuse():
    reuse = scientist.control_reuse()
    base = ROOT / "studies/active_learning"
    for seed in SEEDS:
        initial = next(e for e in reuse["entries"] if e["seed"] == seed and e["budget"] == 333)
        initial_audit = read((ROOT / initial["checkpoint_path"]).parent / "fit_audit.json")
        for r, budget in enumerate(BUDGETS):
            if budget <= scientist.BUDGETS[-1]:
                continue
            for method, name, folder, freeze_name in (
                ("random32", "qgeognn_v2_row_sequential_b32", "random", "contract.json"),
                ("center_width_lcmd", "qgeognn_v2_row_cw_lcmd_to_653" if budget <= 653 else "qgeognn_v2_row_cw_lcmd_to_1005", "center_width_lcmd", "round_freeze.json")):
                directory = base / name / f"runtime/seed_{seed}/{folder}/round_{r:02d}"
                frozen = read(directory / freeze_name)
                checkpoint = Path(frozen["checkpoint_path"]) if "checkpoint_path" in frozen else directory / "model/best.pt"
                prediction = Path(frozen["prediction_path"]) if "prediction_path" in frozen else directory / "model/predictions.csv.gz"
                audit_path = checkpoint.parent / "fit_audit.json"
                audit = read(audit_path)
                if (audit["train_rows"] != budget or audit["initialization_hash"] != initial["initialization_hash"]
                        or audit["validation_ids_hash"] != initial_audit["validation_ids_hash"]):
                    raise RuntimeError("extended control budget/initialization/validation mismatch")
                protocol_path = base / name / "protocol.json"
                if method == "random32":
                    if any(frozen["input"]["training_config"][k] != v for k, v in TRAINING_CONFIG.items()):
                        raise RuntimeError("extended random training mismatch")
                elif read(protocol_path)["training"] != TRAINING_CONFIG:
                    raise RuntimeError("extended CW training mismatch")
                for kind, path in (("checkpoint", checkpoint), ("prediction", prediction)):
                    if sha256_file(path) != frozen[f"{kind}_sha256"] or sha256_file(path) != audit[f"{kind}_sha256"]:
                        raise RuntimeError("extended control artifact drift")
                for path in (checkpoint, prediction, audit_path, directory / freeze_name, protocol_path):
                    reuse["source_hashes"][str(path.relative_to(ROOT))] = sha256_file(path)
                reuse["entries"].append({"seed": seed, "method": method, "budget": budget,
                    "checkpoint_path": str(checkpoint.relative_to(ROOT)), "prediction_path": str(prediction.relative_to(ROOT)),
                    "freeze_path": str((directory / freeze_name).relative_to(ROOT)),
                    "initialization_hash": initial["initialization_hash"], "L333_ids_hash": initial["L333_ids_hash"]})
    return reuse


def prepare(config=None, inheritance=None, format_repair=None):
    config = config or settings()
    if (STUDY / "protocol.json").exists():
        if read(STUDY / "protocol.json")["transport"] != config:
            raise RuntimeError("transport cannot change after protocol freeze")
        if inheritance is not None and read(STUDY / "protocol.json").get("inheritance") != inheritance:
            raise RuntimeError("inherited trajectory cannot change after protocol freeze")
        return validate()
    STUDY.mkdir(parents=True, exist_ok=True)
    once(STUDY / "control_reuse.json", control_reuse())
    record = {
        "study_version": VERSION, "implementation": "full_pool_feedback_v2",
        "split": "Row", "seeds": list(SEEDS), "methods": [*CONTROLS, METHOD],
        "budgets": list(BUDGETS), "batch_size": 32, "selection_count": 32,
        "training": TRAINING_CONFIG,
        "retraining": "shared Scientist workflow; same initialization, optimizer and validation checkpoint rule",
        "roles": list(FULL_ROLE_PROMPTS), "transport": config,
        "candidate_delivery": "all candidates and observed records; lossless tables; no retrieval",
        "model_call_budget_per_role": FULL_SELECTION_ATTEMPTS,
        "fixed_quota": None, "hidden_reranking": None,
        "memory": "previous selected plans and scientist memos, hypotheses, measured outcomes and errors",
        "context_isolation": "independent role transcripts; seed/method-local history",
        "truth_barrier": "all 32 IDs and premeasurement predictions frozen before label reveal",
        "selector_boundary": "own L_t truth and U_t features/predictions only; no validation/test or other trajectories",
        "pilot_evaluation": "validation fit audits only; report never reveals test truth",
        "code_hashes": code_hashes(), "packages": package_versions(),
        "prompts": {role: stable_hash(prompt) for role, prompt in FULL_ROLE_PROMPTS.items()},
        "control_reuse_sha256": sha256_file(STUDY / "control_reuse.json"),
    }
    if inheritance is not None:
        record["inheritance"] = inheritance
    if format_repair is not None:
        if format_repair.get("role") not in FULL_ROLE_PROMPTS or format_repair.get("extra_attempts") != 1:
            raise ValueError("invalid bounded format repair")
        record["format_repair"] = format_repair
    once(STUDY / "protocol.json", record)
    for role, prompt in FULL_ROLE_PROMPTS.items():
        (STUDY / f"{role}_prompt.txt").write_text(prompt)
    return {"status": "PREPARED", "control_refits": 0, "new_training_runs_started": 0}


def make_inheritance(parent_revision, seed, method, start_round):
    """Bind an unchanged selection/feedback/checkpoint prefix before continuation."""
    if (Path(parent_revision).name != parent_revision or parent_revision in ("", ".", "..")
            or seed not in SEEDS or method != METHOD or start_round not in range(1, len(BUDGETS))):
        raise ValueError("invalid inherited trajectory")
    parent = ARCHIVE_STUDY / "revisions" / parent_revision
    if parent == STUDY:
        raise ValueError("continuation requires an independent revision")
    protocol = read(parent / "protocol.json")
    for name, digest in protocol["code_hashes"].items():
        if sha256_file(ROOT / name) != digest:
            preserved = parent / "runtime/frozen_source_before_sol61" / name
            if not preserved.exists() or sha256_file(preserved) != digest:
                raise RuntimeError(f"parent code drift without preserved source: {name}")
    if protocol["prompts"] != {role: stable_hash(prompt) for role, prompt in FULL_ROLE_PROMPTS.items()}:
        raise RuntimeError("parent role prompts differ")
    copies, sources, selections, transports, extras = {}, {}, [], {}, {}
    def source(path):
        sources[str(path)] = sha256_file(path)
    source(parent / "protocol.json")
    source(parent / "control_reuse.json")
    for role in FULL_ROLE_PROMPTS:
        source(parent / f"{role}_prompt.txt")
    for r in range(start_round):
        rel = f"selections/seed_{seed}/{method}/round_{r:02d}"
        selections.append(rel)
        inherited = protocol.get("inheritance", {})
        if rel in inherited.get("selection_directories", []):
            transports[rel] = inherited.get("transports", {}).get(rel, inherited["transport"])
        else:
            transports[rel] = protocol["transport"]
        for role in FULL_ROLE_PROMPTS:
            key = f"{rel}/agents/{role}"
            extras[key] = inherited.get("extra_format_attempts", {}).get(key, 0)
            repair = protocol.get("format_repair", {})
            if repair.get("directory") == rel and repair.get("role") == role:
                extras[key] = repair["extra_attempts"]
        directory = parent / rel
        scientist.verify_files(directory, read(directory / "batch_freeze.json")["artifact_hashes"])
        for name in ("feedback.json", "label_access_receipt.json"):
            read(directory / name)
        for path in sorted(directory.rglob("*.json")):
            copies[str(path.relative_to(parent))] = str(path)
            source(path)
    for r in range(start_round+1):
        rel = f"runtime/seed_{seed}/{method}/round_{r:02d}/prediction_freeze.json"
        path = parent / rel
        frozen = read(path)
        copies[rel] = str(path)
        source(path)
        for kind in ("checkpoint", "prediction"):
            artifact = Path(frozen[f"{kind}_path"])
            if sha256_file(artifact) != frozen[f"{kind}_sha256"]:
                raise RuntimeError("parent checkpoint/prediction drift")
            source(artifact)
        source(Path(frozen["checkpoint_path"]).parent / "fit_audit.json")
    return {"parent_revision": parent_revision, "seed": seed, "method": method,
        "start_round": start_round, "active_label_count": BUDGETS[start_round],
        "transport": protocol["transport"], "selection_directories": selections,
        "transports": transports, "extra_format_attempts": extras,
        "copied_files": copies, "source_hashes": sources,
        "meaning": "historical rounds replay with historical transport; new rounds use current transport"}


def inherit_files(inheritance):
    """Copy original bytes so all existing freeze and receipt hashes stay valid."""
    for name, digest in inheritance["source_hashes"].items():
        if sha256_file(Path(name)) != digest:
            raise RuntimeError(f"inherited source drift: {name}")
    for relative, name in inheritance["copied_files"].items():
        destination = STUDY / relative
        digest = inheritance["source_hashes"][name]
        if destination.exists():
            if sha256_file(destination) != digest:
                raise RuntimeError(f"inherited destination drift: {relative}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(name, destination)


def validate():
    record = read(STUDY / "protocol.json")
    if record["code_hashes"] != code_hashes():
        raise RuntimeError("frozen code drift; use an isolated revision for new code")
    if record["prompts"] != {role: stable_hash(prompt) for role, prompt in FULL_ROLE_PROMPTS.items()}:
        raise RuntimeError("frozen prompt drift")
    for role, prompt in FULL_ROLE_PROMPTS.items():
        if (STUDY / f"{role}_prompt.txt").read_text() != prompt:
            raise RuntimeError("prompt artifact drift")
    if sha256_file(STUDY / "control_reuse.json") != record["control_reuse_sha256"]:
        raise RuntimeError("control manifest drift")
    for name, digest in read(STUDY / "control_reuse.json")["source_hashes"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError(f"control artifact drift: {name}")
    inheritance = record.get("inheritance")
    if inheritance:
        for name, digest in inheritance["source_hashes"].items():
            if sha256_file(Path(name)) != digest:
                raise RuntimeError(f"inherited source drift: {name}")
        for relative, name in inheritance["copied_files"].items():
            if sha256_file(STUDY / relative) != inheritance["source_hashes"][name]:
                raise RuntimeError(f"inherited destination drift: {relative}")
    return {"status": "VALID", "protocol_sha256": stable_hash(record)}


class Context(lifecycle.WorkflowContext):
    def __init__(self, seed):
        super().__init__(seed, study_root=STUDY)


def selection_directory(seed, method=METHOD, round_index=0):
    if seed not in SEEDS or method != METHOD or round_index not in range(len(BUDGETS)-1):
        raise ValueError("unregistered seed/method/round")
    return STUDY / f"selections/seed_{seed}/{method}/round_{round_index:02d}"


def state(context, method, round_index):
    return lifecycle.state(context, method, round_index, workflow=sys.modules[__name__])


def prediction_freeze(context, method, r, labeled):
    inheritance = read(STUDY / "protocol.json").get("inheritance")
    if inheritance and (context.seed, method) == (inheritance["seed"], inheritance["method"]) and r <= inheritance["start_round"]:
        directory = context.runtime / method / f"round_{r:02d}"
        record = read(directory / "prediction_freeze.json")
        if record["L_t_ids_hash"] != ids_hash(context.ids(labeled)):
            raise RuntimeError("inherited checkpoint trained on different labels")
        for kind in ("checkpoint", "prediction"):
            if sha256_file(Path(record[f"{kind}_path"])) != record[f"{kind}_sha256"]:
                raise RuntimeError("inherited checkpoint/prediction drift")
        return Path(record["checkpoint_path"]), directory
    return scientist.prediction_freeze(context, method, r, labeled)


def role_packet(packet, role, **extra):
    value = {k: v for k, v in packet.items() if k != "packet_sha256"}
    value.update(role=role, **extra)
    value["packet_sha256"] = stable_hash(value)
    return value


def stage(seed, method=METHOD, r=0):
    validate()
    directory = selection_directory(seed, method, r)
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    context = Context(seed)
    labeled, unlabeled, truth, history = state(context, method, r)
    checkpoint, runtime = prediction_freeze(context, method, r, labeled)
    catalog, raw = lifecycle.make_catalog(context, method, r, labeled, unlabeled, truth,
        history, checkpoint, runtime, pending_count=0, version=VERSION)
    packet = {"study_version": VERSION, "seed": seed, "method": method, "round": r,
        "active_label_count": len(labeled), "selection_count": 32,
        "objective": "overall Row V1/V2 predictive accuracy after retraining",
        "memory": trajectory_memory(history, seed, method),
        "delivery": "all_candidates_one_request", "candidate_count": len(catalog.candidates),
        "observed_count": len(catalog.observed), "validation_test_records": 0,
        **encode_catalog(catalog)}
    packet["packet_sha256"] = stable_hash(packet)
    once(directory / "input.json", packet)
    once(directory / "catalog.json", raw)
    once(directory / "contract.json", {"seed": seed, "method": method, "round": r,
        "protocol_hash": context.protocol_hash, "checkpoint_sha256": sha256_file(checkpoint),
        "L_t_ids": context.ids(labeled), "U_t_ids": context.ids(unlabeled),
        "input_sha256": sha256_file(directory / "input.json"),
        "catalog_sha256": sha256_file(directory / "catalog.json")})
    return {"status": "READY_FOR_SELECTION", "seed": seed, "method": method, "round": r,
        "packet": str(directory / "input.json"), "eligible_count": len(catalog.candidates),
        "pending_count": 0, "new_labels_revealed": 0, "llm_calls": 0, "fits": 0}


def transport_for(directory):
    protocol = read(STUDY / "protocol.json")
    inheritance = protocol.get("inheritance")
    relative = str(directory.relative_to(STUDY))
    if inheritance and relative in inheritance["selection_directories"]:
        return inheritance.get("transports", {}).get(relative, inheritance["transport"])
    return protocol["transport"]


def _select_roles(directory, call=None):
    packet, raw = read(directory / "input.json"), read(directory / "catalog.json")
    config = transport_for(directory)
    repair = read(STUDY / "protocol.json").get("format_repair", {})
    def extra_attempts(role):
        relative = str(directory.relative_to(STUDY))
        inherited = read(STUDY / "protocol.json").get("inheritance", {})
        if relative in inherited.get("selection_directories", []):
            return inherited.get("extra_format_attempts", {}).get(f"{relative}/agents/{role}", 0)
        return 1 if (repair.get("role") == role and repair.get("directory") == relative) else 0
    memos = {}
    for role in ("chemistry_scientist", "ml_scientist"):
        # Neither scientist receives the current other scientist's memo.
        memos[role] = run_agent(role, role_packet(packet, role), ReadOnlyCatalog(**raw),
            directory / "agents" / role, config, call=call, extra_format_attempts=extra_attempts(role))
    once(directory / "expert_memos.json", memos)
    planner_packet = role_packet(packet, "scientific_planner",
        chemistry_memo=memos["chemistry_scientist"], ml_memo=memos["ml_scientist"],
        expert_memos_are_advisory=True)
    once(directory / "planner_input.json", planner_packet)
    catalog = ReadOnlyCatalog(**raw)
    result = run_agent("scientific_planner", planner_packet, catalog,
        directory / "agents/scientific_planner", config, call=call, extra_format_attempts=extra_attempts("scientific_planner"))
    once(directory / "selection.json", result)
    return result, catalog


def _verify_batch(directory, result, catalog, frozen):
    contract = read(directory / "contract.json")
    batch = [c["id"] for c in result["choices"]]
    if len(batch) != 32 or len(set(batch)) != 32 or not set(batch) <= set(contract["U_t_ids"]):
        raise RuntimeError("illegal planner batch")
    predictions = [{"candidate_id": i, "pred_V1_ml": catalog.candidates[i]["pred_V1_ml"],
        "pred_V2_ml": catalog.candidates[i]["pred_V2_ml"]} for i in batch]
    if frozen["batch_ids"] != batch or frozen["pending_ids"] or frozen["premeasurement_predictions"] != predictions:
        raise RuntimeError("batch/selection/prediction mismatch")
    return frozen


def audit_batch(directory):
    frozen = read(directory / "batch_freeze.json")
    scientist.verify_files(directory, frozen["artifact_hashes"])
    def unavailable(*args):
        raise RuntimeError("audit may only replay existing role turns")
    result, catalog = _select_roles(directory, call=unavailable)
    return _verify_batch(directory, result, catalog, frozen)


def select(seed, method=METHOD, r=0, call=None):
    validate()
    directory = selection_directory(seed, method, r)
    if (directory / "batch_freeze.json").exists():
        audit_batch(directory)
        return {"status": "ALREADY_FROZEN"}
    stage(seed, method, r)
    result, catalog = _select_roles(directory, call=call)
    batch = [c["id"] for c in result["choices"]]
    frozen = {"batch_ids": batch, "pending_ids": [],
        "premeasurement_predictions": [{"candidate_id": i,
            "pred_V1_ml": catalog.candidates[i]["pred_V1_ml"],
            "pred_V2_ml": catalog.candidates[i]["pred_V2_ml"]} for i in batch],
        "viewed_candidate_ids": sorted(catalog.viewed), "new_batch_labels_revealed": False}
    _verify_batch(directory, result, catalog, frozen)
    names = ["input.json", "catalog.json", "contract.json", "expert_memos.json", "planner_input.json", "selection.json"]
    names += [str(p.relative_to(directory)) for p in sorted((directory / "agents").rglob("*.json"))]
    frozen["artifact_hashes"] = {name: sha256_file(directory / name) for name in names}
    once(directory / "batch_freeze.json", frozen)
    return {"status": "BATCH_FROZEN_BEFORE_REVEAL", "batch_count": 32, "new_labels_revealed": 0}


def feedback_metadata(result, directory):
    memos = read(directory / "expert_memos.json")
    return {"chemistry_memo": memos["chemistry_scientist"], "ml_memo": memos["ml_scientist"],
        "selected_plan": result, **{key: result[key] for key in (
            "expert_synthesis", "batch_strategy", "expected_learning_value")}}


def advance(seed, method=METHOD, r=0):
    return lifecycle.advance(seed, method, r, workflow=sys.modules[__name__])


def report(seed=157, method=METHOD, rounds=2):
    """Compare completed fits at equal budgets using validation only."""
    validate()
    if rounds not in range(1, len(BUDGETS)):
        raise ValueError(f"rounds must be 1..{len(BUDGETS)-1}")
    context = Context(seed)
    state(context, method, rounds)
    rows, sources = [], {}
    def add(current_method, r, checkpoint):
        path = Path(checkpoint).parent / "fit_audit.json"
        audit = read(path)
        if audit["train_rows"] != BUDGETS[r]:
            raise RuntimeError("comparison budget mismatch")
        sources[str(path.relative_to(ROOT))] = sha256_file(path)
        rows.append({"seed": seed, "method": current_method, "budget": BUDGETS[r],
            "validation_NRMSE": audit["best_validation_combined_normalized_rmse"]})
    for r in range(rounds+1):
        p = context.runtime / method / f"round_{r:02d}/prediction_freeze.json"
        record = read(p)
        for kind in ("checkpoint", "prediction"):
            if sha256_file(Path(record[f"{kind}_path"])) != record[f"{kind}_sha256"]:
                raise RuntimeError("prediction/checkpoint drift")
        sources[str(p.relative_to(ROOT))] = sha256_file(p)
        add(method, r, record["checkpoint_path"])
    for entry in read(STUDY / "control_reuse.json")["entries"]:
        if entry["seed"] == seed and entry["budget"] <= BUDGETS[rounds]:
            add(entry["method"], BUDGETS.index(entry["budget"]), ROOT / entry["checkpoint_path"])
    for reference in scientist.METHODS:
        for r in range(rounds+1):
            p = scientist.STUDY / f"runtime/seed_{seed}/{reference}/round_{r:02d}/prediction_freeze.json"
            if p.exists():
                record = read(p)
                if sha256_file(Path(record["checkpoint_path"])) != record["checkpoint_sha256"]:
                    raise RuntimeError("reference checkpoint drift")
                add(reference, r, record["checkpoint_path"])
    summaries = []
    for current in dict.fromkeys(row["method"] for row in rows):
        curve = sorted((row for row in rows if row["method"] == current), key=lambda row: row["budget"])
        if len(curve) == rounds+1:
            summaries.append({"method": current, "endpoint_validation_NRMSE": curve[-1]["validation_NRMSE"],
                "validation_AULC": float(np.trapezoid([row["validation_NRMSE"] for row in curve],
                    [row["budget"] for row in curve])/(BUDGETS[rounds]-BUDGETS[0]))})
    calls = [{"round": r, "role": role, "turn": p.name, **read(p)["provenance"]}
        for r in range(rounds) for role in FULL_ROLE_PROMPTS
        for p in sorted((selection_directory(seed, method, r) / "agents" / role).glob("turn_[0-9][0-9].json"))]
    result = {"seed": seed, "method": method, "rounds": rounds,
        "learning_curves": rows, "summaries": summaries, "llm_calls": calls, "source_hashes": sources,
        "test_truth_access_count": 0, "evidence": "single-seed exploratory validation comparison; no test evaluation"}
    path = STUDY / f"reports/seed_{seed}/{method}/rounds_{rounds:02d}/results.json"
    once(path, result)
    return {"status": "VALIDATION_REPORTED", "results": str(path), "summaries": summaries}


def run(seed=157, method=METHOD, rounds=2, config=None, status_interval=30.0,
        start_round=None, parent_revision=None):
    if seed not in SEEDS or method != METHOD or rounds not in range(1, len(BUDGETS)):
        raise ValueError("unregistered trajectory or rounds")
    with scientist.exclusive_lock(STUDY / ".operation.lock"):
        if start_round is None:
            start_round = read(STUDY / "protocol.json").get("inheritance", {}).get("start_round", 0) if (STUDY / "protocol.json").exists() else 0
        if not 0 <= start_round < rounds:
            raise ValueError("start round must precede final round")
        inheritance = make_inheritance(parent_revision, seed, method, start_round) if parent_revision else None
        if inheritance:
            inherit_files(inheritance)
        prepare(config, inheritance=inheritance)
        registered = read(STUDY / "protocol.json").get("inheritance")
        if registered and (seed, method, start_round) != (registered["seed"], registered["method"], registered["start_round"]):
            raise RuntimeError("continuation identity does not match inherited prefix")
        if start_round and not registered:
            raise RuntimeError("nonzero start round requires a frozen inherited prefix")
        state(Context(seed), method, start_round)
        check_transport(read(STUDY / "protocol.json")["transport"])
        once(STUDY / f"run_scope_seed_{seed}_to_{BUDGETS[rounds]}.json",
            {"seed": seed, "method": method, "rounds": rounds,
             "start_round": start_round, "additional_rounds": rounds-start_round,
             "budgets": list(BUDGETS[:rounds+1]), "evaluation": "validation only"})
        for r in range(start_round, rounds):
            with status_phase(f"dual expert seed={seed} round={r} selection", interval=status_interval):
                select(seed, method, r)
            with status_phase(f"dual expert seed={seed} round={r} fit", interval=status_interval):
                advance(seed, method, r)
        return report(seed, method, rounds)


@contextmanager
def status_phase(message, interval=30.0):
    import math
    import threading
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("status interval must be finite and positive")
    started, stopped = time.monotonic(), threading.Event()
    def progress(event):
        path = STUDY / "execution_progress.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": time.time(), **event}, ensure_ascii=False)+"\n")
    def heartbeat():
        while not stopped.wait(interval):
            elapsed = round(time.monotonic()-started, 1)
            print(f"[Dual Expert] {message}; elapsed={elapsed}s", flush=True)
            progress({"phase": message, "elapsed_seconds": elapsed})
    print(f"[Dual Expert] {message}; started", flush=True)
    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join()
