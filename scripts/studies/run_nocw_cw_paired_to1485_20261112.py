#!/usr/bin/env python3
"""Resume NoCW and complete matched CW-gradient LCMD: five seeds, L1485."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SCRIPT = Path(__file__).resolve()
STUDY = ROOT / "studies/active_learning/nocw_vs_cw_lcmd_five_to1485_20261112"
NOCW = STUDY / "nocw"
CW = STUDY / "cw_lcmd"
SOURCE = ROOT / "studies/active_learning/qgeognn_v2_row_llm_nocw_ablation/revisions/five_seed_nocw_to1389_20261101"
CW_SOURCE = ROOT / "studies/active_learning/qgeognn_v2_row_cw_lcmd_to_1005"
BASELINE = ROOT / "studies/active_learning/qgeognn_v2_row_sequential_b32"
SEEDS = (157, 887, 2357, 6101, 12203)
BUDGETS = tuple(range(333, 1486, 32))
ARM = "llm32_nocw"
METHOD = "center_width_lcmd"
POLICY = STUDY / "protocol.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def once(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise RuntimeError(f"Immutable record differs: {path}")
    else:
        atomic(path, value)


@contextmanager
def lock(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def compute_slot():
    handles = []
    try:
        while not handles:
            for slot in range(2):
                path = STUDY / f"locks/compute_{slot}.lock"
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = path.open("a+")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                else:
                    handles.append(handle)
                    break
            if not handles:
                time.sleep(2)
        yield
    finally:
        for handle in handles:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()


def last_round(root, seed, method):
    points = sorted((root / f"runtime/seed_{seed}/{method}").glob("round_*/prediction_freeze.json"))
    rounds = [int(p.parent.name.split("_")[1]) for p in points]
    if rounds and rounds != list(range(rounds[-1] + 1)):
        raise RuntimeError("Noncontiguous saved NoCW fits")
    return rounds[-1] if rounds else -1


def remaining_rounds(completed):
    if completed not in range(37):
        raise ValueError("Completed round must lie between zero and 36")
    return range(completed, 36)


def reconcile_interruption(started, stop, comparison_hash):
    """Record a locally interrupted stream; remote completion/billing stay unknown."""
    started = Path(started)
    terminal = started.with_name(started.name.replace(".started.json", ".json"))
    if terminal.exists():
        return
    seed = int(started.parent.parent.parent.name.split("_")[1])
    entry = next(row for row in stop["seeds"] if row["seed"] == seed)
    relative = str(started.relative_to(NOCW))
    if (not stop["all_target_processes_stopped"] or not entry["lock_released"]
            or relative not in entry["unresolved_started_preserved"]
            or (started.parent / "batch_freeze.json").exists()):
        raise RuntimeError("Interrupted request is outside the verified user-stop record")
    completed = started.with_name(started.name.replace("attempt_", "stream_").replace(".started.json", ".completed.json"))
    if completed.exists():
        raise RuntimeError("A durable completed answer must be recovered instead of resent")
    original = read(started)
    once(terminal, {**original, "success": False, "error_type": "LocallyInterruptedByVerifiedUserStop",
        "error_message": "Local stream ended at the recorded user stop; no complete response was saved. Explicitly resumed in a new revision.",
        "elapsed_seconds": 0, "elapsed_seconds_unknown": True,
        "remote_completion_status": "unknown", "billing_status": "unknown",
        "local_interruption_reconciliation": True, "started_sha256": sha(started),
        "comparison_protocol_sha256": comparison_hash, "reconciled_utc": now()})


def prepare():
    if POLICY.exists():
        return validate()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/nocw_cw_paired_matplotlib")
    from scripts.studies import run_qgeognn_v2_row_llm_nocw_to1389 as old
    old.activate()
    original = old.validate()
    stop_path = SOURCE / "progress/user_stop_20261101T112435Z.json"
    stop = read(stop_path)
    counts = {seed: last_round(SOURCE, seed, ARM) for seed in SEEDS}
    if counts != {157: 23, 887: 21, 2357: 25, 6101: 26, 12203: 28}:
        raise RuntimeError("NoCW source advanced since the inspected stopped snapshot")
    for seed in SEEDS:
        if old.lock_busy(SOURCE, seed):
            raise RuntimeError("A source NoCW worker is active")
    files = {str(SCRIPT.relative_to(ROOT)): sha(SCRIPT),
             "tests/active_learning_v2/test_nocw_cw_paired_1485.py": sha(ROOT / "tests/active_learning_v2/test_nocw_cw_paired_1485.py")}
    record = {"created_utc": now(), "user_request": "llm我主要对应nocwllm，先帮我把实验补齐吧",
        "user_budget_reply": "5个 seed 均补到 L1485，按上一条36轮方案",
        "seeds": list(SEEDS), "budgets": list(BUDGETS), "batch_size": 32,
        "methods": [ARM, METHOD], "lcmd_geometry": "center_width_gradient",
        "primary_metric": "raw normalized trapezoidal validation NRMSE AULC, L333-L1485",
        "secondary_metrics": ["validation NRMSE at L1005", "validation NRMSE at L1389", "validation NRMSE at L1485"],
        "primary_scope": "descriptive paired comparison on established five Row splits; test evaluation deferred",
        "nocw_source": str(SOURCE.relative_to(ROOT)), "nocw_source_rounds": counts,
        "cw_source_rounds": {seed: 21 if seed in (157, 6101) else 0 for seed in SEEDS},
        "new_nocw_batches": sum(36-r for r in counts.values()), "new_cw_batches": 138,
        "nocw_protocol_sha256": original["protocol_sha256"],
        "source_stop_record_sha256": sha(stop_path), "code_hashes": files,
        "maximum_concurrent_compute_jobs": 2, "cpu_threads_per_job": 2,
        "new_llm_models": False, "new_llm_restarts_from_L333": False,
        "llm_retry_policy": "retain existing strict full-pool corrections and network backoff; preserve failures and unknown interrupted usage",
        "test_truth_reads": 0, "source_files_modified": False}
    STUDY.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "protocol.json", "protocol.lock.json", "served_model_lock.json"):
        target = NOCW / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / name, target)
    imported = {}
    for seed, end in counts.items():
        selection_source = SOURCE / f"selections/seed_{seed}/{ARM}"
        selection_target = NOCW / f"selections/seed_{seed}/{ARM}"
        shutil.copytree(selection_source, selection_target, dirs_exist_ok=True)
        for path in selection_source.glob("round_*/*"):
            if path.is_file():
                relative = path.relative_to(SOURCE)
                if sha(path) != sha(NOCW / relative):
                    raise RuntimeError("NoCW prefix copy differs")
                imported[str(relative)] = sha(path)
        for rnd in range(end + 1):
            relative = Path(f"runtime/seed_{seed}/{ARM}/round_{rnd:02d}/prediction_freeze.json")
            target = NOCW / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE / relative, target)
            frozen = read(target)
            for kind in ("checkpoint", "prediction"):
                if sha(Path(frozen[f"{kind}_path"])) != frozen[f"{kind}_sha256"]:
                    raise RuntimeError("NoCW predictor hash drift")
            imported[str(relative)] = sha(target)
    atomic(NOCW / "prefix_import.json", {"source": str(SOURCE), "files": imported, "new_provider_calls": 0})
    once(POLICY, record)
    digest = sha(POLICY)
    for seed in SEEDS:
        for started in (NOCW / f"selections/seed_{seed}/{ARM}").glob("round_*/attempt_*.started.json"):
            reconcile_interruption(started, stop, digest)
    atomic(STUDY / "formal_authorization.json", {"formal_run_authorized": True,
        "protocol_sha256": digest, "user_budget_reply": record["user_budget_reply"], "recorded_utc": now()})
    (STUDY / "README.md").write_text(
        "# NoCW LLM versus CW-gradient LCMD32: five paired seeds to L1485\n\n"
        "User authorized both methods on seeds 157, 887, 2357, 6101, 12203, with 36 batches of 32 after shared L333 starts. "
        "Reuse all existing scientific trajectories without changing old files. NoCW needs 57 additional batches; CW-LCMD needs 138. "
        "Primary outcome: raw normalized trapezoidal validation NRMSE AULC, L333-L1485; endpoints are secondary. "
        "These are established Row splits, not a new untouched confirmation cohort. No test truth is read in this run.\n\n"
        "The imported NoCW requests/answers/freezes remain byte-identical. Four interrupted local streams get separate "
        "local reconciliation receipts in this revision; provider completion and billing remain unknown. "
        "Original records remain intact. Do not substitute partial answers or reconstruct selections. "
        "Model, effort, prompt, input encoding, candidate pool, correction rules and training settings are retained.\n\n"
        "Two shared compute slots, two CPU threads each; five independent NoCW keys. "
        "Each worker is dispatched once and stops on an unexpected error; no automatic abnormal-worker restart. "
        "The supervisor writes status.json and validation snapshots. It does not send reports to acquisition.\n",
        encoding="utf-8")
    return validate()


def validate():
    record = read(POLICY)
    if record["seeds"] != list(SEEDS) or record["budgets"] != list(BUDGETS):
        raise RuntimeError("Paired scope drift")
    for name, digest in record["code_hashes"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"Paired adapter source drift: {name}")
    source_protocol = read(NOCW / "protocol.lock.json")
    for name, digest in source_protocol["code_hashes"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"Frozen scientific code drift: {name}")
    if read(NOCW / "protocol.json") != source_protocol:
        raise RuntimeError("NoCW frozen protocol drift")
    return {"status": "VALID", "protocol_sha256": record["nocw_protocol_sha256"],
            "comparison_protocol_sha256": sha(POLICY), "target_budget": 1485}


def status(method, seed, phase, **details):
    atomic(STUDY / f"progress/{method}_{seed}.json", {"method": method, "seed": seed,
        "phase": phase, "pid": os.getpid(), "updated_utc": now(), **details})


def activate_nocw(seed):
    from scripts.studies import run_qgeognn_v2_row_llm_nocw_to1389 as old
    old.activate(seed)
    base, retry, stream, legacy, workflow = old.base, old.retry, old.stream, old.legacy, old.s
    for module in (base, retry, stream, legacy, workflow):
        module.STUDY = NOCW
    workflow.BUDGETS = BUDGETS
    workflow.validate = validate
    def authorized():
        result = validate()
        if read(STUDY / "formal_authorization.json")["protocol_sha256"] != result["comparison_protocol_sha256"]:
            raise PermissionError("Paired execution authorization mismatch")
    workflow.require_authorization = authorized
    workflow.audit_batch = old.ORIGINAL_AUDIT_BATCH
    workflow.write_stage = old.ORIGINAL_WRITE_STAGE
    end = int(read(POLICY)["nocw_source_rounds"][str(seed)])
    def frozen(context, method, rnd, labeled):
        if rnd <= end:
            directory = NOCW / f"runtime/seed_{seed}/{method}/round_{rnd:02d}"
            saved = read(directory / "prediction_freeze.json")
            if saved["L_t_ids_hash"] != workflow.ids_hash(context.ids(labeled)) or saved["test_truth_access_count"] != 0:
                raise RuntimeError("Imported NoCW state identity drift")
            return Path(saved["checkpoint_path"]), directory
        return old.ORIGINAL_PREDICTION_FREEZE(context, method, rnd, labeled)
    workflow.prediction_freeze = frozen
    original_fit = base.Context.fit
    def fit(context, *args, **kwargs):
        with compute_slot():
            return original_fit(context, *args, **kwargs)
    # This subclass avoids modifying shared or frozen source files.
    class ResumeContext(base.Context):
        pass
    ResumeContext.fit = fit
    workflow.Context = ResumeContext
    base.credential_audit()
    os.environ.pop("SCIENTIST_API_KEY", None)
    os.environ["SCIENTIST_API_KEY_FILE"] = str(base.KEY_FILES[seed])
    base.torch.set_num_threads(2)
    return old


def run_nocw(seed):
    old = activate_nocw(seed)
    s = old.s
    with lock(STUDY / f"locks/nocw_{seed}.lock"):
        end = last_round(NOCW, seed, ARM)
        for rnd in remaining_rounds(end):
            directory = NOCW / f"selections/seed_{seed}/{ARM}/round_{rnd:02d}"
            status("nocw", seed, "SELECTING", completed_rounds=rnd, budget=BUDGETS[rnd])
            with s.exclusive_lock(NOCW / ".operation.lock"):
                if not (directory / "contract.json").exists():
                    with compute_slot():
                        s.stage(seed, ARM, rnd, allow_feature_compute=True)
                if (directory / "batch_freeze.json").exists():
                    s.audit_batch(directory)
                else:
                    packet = read(directory / "input.json")
                    catalog = s.ReadOnlyCatalog(**read(directory / "catalog.json"))
                    plan = s.run_selection(packet, catalog, directory, "no_cw",
                        read(NOCW / "config.json")["transport"], s.live_call)
                    s.freeze_batch(directory, plan, catalog, packet)
            status("nocw", seed, "TRAINING_OR_WAITING_FOR_SLOT", completed_rounds=rnd, budget=BUDGETS[rnd])
            s.advance(seed, ARM, rnd, allow_reveal=True, allow_training=True)
            frozen = read(NOCW / f"runtime/seed_{seed}/{ARM}/round_{rnd+1:02d}/prediction_freeze.json")
            audit = read(Path(frozen["checkpoint_path"]).with_name("fit_audit.json"))
            assert audit["train_rows"] == BUDGETS[rnd+1]
            assert audit["test_labels_used_for_fit_or_checkpoint_selection"] == 0
            status("nocw", seed, "BATCH_COMPLETE", completed_rounds=rnd+1, budget=BUDGETS[rnd+1])
        status("nocw", seed, "COMPLETE", completed_rounds=36, budget=1485)


def run_cw(seed):
    import numpy as np
    import pandas as pd
    import torch
    from scripts.studies import run_qgeognn_v2_row_llm_nocw_five as base
    from src.qgeognn_al.active_learning_v2.cw_lcmd_extension_runner import CWExtensionContext, _acquire, _write_csv_once
    from src.qgeognn_al.active_learning_v2.gradient_transforms import center_width_transform
    from src.qgeognn_al.active_learning_v2.protocol import RestrictedLabelStore, stable_hash, validate_row_protocol
    from src.qgeognn_al.active_learning_v2.sequential_acquisition import validate_trajectory_transition
    from src.qgeognn_al.active_learning_v2.benchmark_protocol import load_features
    from src.qgeognn_al.schemas.conditions import ConditionNormalization
    torch.set_num_threads(2)
    with lock(STUDY / f"locks/cw_{seed}.lock"):
        ctx = CWExtensionContext.__new__(CWExtensionContext)
        ctx.seed, ctx.study = seed, CW
        ctx.runtime = CW / f"runtime/seed_{seed}"
        ctx.runtime.mkdir(parents=True, exist_ok=True)
        paths = base.initial_paths(seed)
        ctx.partition = pd.read_csv(BASELINE / f"splits/row_seed_{seed}.csv")
        validate_row_protocol(ctx.partition)
        ctx.data = load_features()
        assert ctx.data.sample_id.astype(str).tolist() == ctx.partition.sample_id.astype(str).tolist()
        ctx.roles = {r: ctx.partition.loc[ctx.partition.role.eq(r), "canonical_index"].to_numpy(int)
                     for r in ("l0", "u0", "validation", "test")}
        initial = read(paths["context"])["contract"]
        ctx.preprocessing = initial["preprocessing"]
        ctx.normalization = ConditionNormalization(**initial["normalization"])
        ctx.atom, ctx.angle = torch.load(paths["scrubbed_graphs"], weights_only=False)
        assert all(not torch.count_nonzero(graph.y) for graph in ctx.atom)
        store = RestrictedLabelStore(base.s.SOURCE_DATA, ctx.partition)
        ctx.l0_truth = store.reveal(ctx.ids(ctx.roles["l0"]), "initial_fit")
        ctx.validation_truth = store.reveal(ctx.ids(ctx.roles["validation"]), "initial_fit")
        ctx.cw_transform, ctx.cw_audit = center_width_transform(ctx.l0_truth)
        ctx.protocol_hash = sha(POLICY)
        first = 21 if seed in (157, 6101) else 0
        if first:
            anchor = CW_SOURCE / f"runtime/seed_{seed}/{METHOD}/round_21"
            state = pd.read_csv(anchor / "state.csv")
            labeled = state.loc[state.role.eq("labeled"), "canonical_index"].to_numpy(int)
            unlabeled = state.loc[state.role.eq("unlabeled"), "canonical_index"].to_numpy(int)
            source_freeze = read(anchor / "round_freeze.json")
        else:
            labeled, unlabeled = ctx.roles["l0"].copy(), ctx.roles["u0"].copy()
            source_freeze = read(NOCW / f"runtime/seed_{seed}/{ARM}/round_00/prediction_freeze.json")
        selected = ctx.ids(labeled)[333:]
        store.freeze_acquisitions(selected)
        truth = np.vstack([ctx.l0_truth, store.reveal(selected, "after_acquisition_fit")]) if selected else ctx.l0_truth.copy()
        initial_audit = read(Path(source_freeze["checkpoint_path"]).with_name("fit_audit.json"))
        for rnd in range(first, 37):
            validate()
            directory = ctx.runtime / METHOD / f"round_{rnd:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            assert len(labeled) == BUDGETS[rnd] and len(selected) == len(set(selected)) == 32*rnd
            _write_csv_once(directory / "state.csv", pd.DataFrame({
                "role": ["labeled"]*len(labeled)+["unlabeled"]*len(unlabeled),
                "canonical_index": np.r_[labeled, unlabeled], "sample_id": ctx.ids(np.r_[labeled, unlabeled])}))
            status("cw", seed, "TRAINING_OR_WAITING_FOR_SLOT", completed_rounds=rnd-1 if rnd>first else first, budget=BUDGETS[rnd])
            if rnd == first:
                checkpoint = Path(source_freeze["checkpoint_path"])
                prediction = Path(source_freeze["prediction_path"])
                audit = initial_audit
                assert sha(checkpoint) == source_freeze["checkpoint_sha256"]
                assert sha(prediction) == source_freeze["prediction_sha256"]
            else:
                with compute_slot():
                    audit = ctx.fit(rnd, labeled, truth, directory / "model")
                checkpoint, prediction = directory / "model/best.pt", directory / "model/predictions.csv.gz"
            assert audit["initialization_hash"] == initial_audit["initialization_hash"]
            assert audit["validation_ids_hash"] == initial_audit["validation_ids_hash"]
            assert audit["train_rows"] == BUDGETS[rnd] and audit["test_labels_used_for_fit_or_checkpoint_selection"] == 0
            once(directory / "fit_receipt.json", {"checkpoint_path": str(checkpoint), "fit_audit_sha256": sha(checkpoint.with_name("fit_audit.json")),
                "train_rows": audit["train_rows"], "validation_nrmse": audit["best_validation_combined_normalized_rmse"]})
            outgoing = []
            if rnd < 36:
                status("cw", seed, "ACQUIRING_OR_WAITING_FOR_SLOT", completed_rounds=rnd, budget=BUDGETS[rnd])
                with compute_slot():
                    outgoing, _ = _acquire(ctx, rnd, labeled, unlabeled, checkpoint, directory)
            once(directory / "round_freeze.json", {"round": rnd, "budget": BUDGETS[rnd],
                "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
                "prediction_path": str(prediction), "prediction_sha256": sha(prediction),
                "selected_next_batch": outgoing, "test_truth_access_count": 0,
                "comparison_protocol_sha256": ctx.protocol_hash})
            status("cw", seed, "COMPLETE" if rnd==36 else "BATCH_COMPLETE", completed_rounds=rnd, budget=BUDGETS[rnd])
            if rnd == 36:
                break
            before_l, before_u = ctx.ids(labeled), ctx.ids(unlabeled)
            lookup = dict(zip(before_u, unlabeled))
            next_labeled = np.r_[labeled, [lookup[x] for x in outgoing]].astype(int)
            selected_set = set(outgoing)
            next_unlabeled = np.asarray([i for i,x in zip(unlabeled, before_u) if x not in selected_set], dtype=int)
            validate_trajectory_transition(before_l, before_u, outgoing, ctx.ids(next_labeled), ctx.ids(next_unlabeled))
            selected.extend(outgoing)
            store.freeze_acquisitions(selected)
            truth = np.vstack([truth, store.reveal(outgoing, "after_acquisition_fit")])
            once(directory / "label_access_receipt.json", {"batch_ids": outgoing, "audit": store.audit[-1],
                "round_freeze_sha256": sha(directory / "round_freeze.json")})
            labeled, unlabeled = next_labeled, next_unlabeled


def report():
    rows = []
    historical = ROOT / "reports/five_methods_cw_gradient_20261103/validation_curves.csv"
    with historical.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "cw_gradient_lcmd32":
                rows.append({"method": "cw", "seed": int(row["seed"]), "budget": int(row["budget"]),
                             "validation_nrmse": float(row["validation_nrmse"])})
    for seed in SEEDS:
        for frozen in sorted((NOCW / f"runtime/seed_{seed}/{ARM}").glob("round_*/prediction_freeze.json")):
            rnd = int(frozen.parent.name.split("_")[1])
            audit = read(Path(read(frozen)["checkpoint_path"]).with_name("fit_audit.json"))
            rows.append({"method": "nocw", "seed": seed, "budget": BUDGETS[rnd],
                         "validation_nrmse": audit["best_validation_combined_normalized_rmse"]})
        for receipt in sorted((CW / f"runtime/seed_{seed}/{METHOD}").glob("round_*/fit_receipt.json")):
            row = read(receipt)
            value = {"method": "cw", "seed": seed, "budget": row["train_rows"], "validation_nrmse": row["validation_nrmse"]}
            existing = next((x for x in rows if all(x[k]==value[k] for k in ("method", "seed", "budget"))), None)
            if existing:
                assert existing == value
            else:
                rows.append(value)
    rows.sort(key=lambda row: (row["method"], row["seed"], row["budget"]))
    output = STUDY / "reports"
    output.mkdir(exist_ok=True)
    with (output / "validation_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    by = {(m,s): [r for r in rows if r["method"]==m and r["seed"]==s] for m in ("nocw","cw") for s in SEEDS}
    common = min((g[-1]["budget"] if g else 0) for g in by.values())
    summaries = []
    if common >= 365:
        for budget in (1005,1389,1485):
            if budget > common:
                continue
            for seed in SEEDS:
                group = {}
                for method in ("nocw","cw"):
                    points = [r for r in by[method,seed] if r["budget"]<=budget]
                    assert [p["budget"] for p in points] == list(range(333,budget+1,32))
                    area = sum((a["validation_nrmse"]+b["validation_nrmse"])*16 for a,b in zip(points,points[1:]))/(budget-333)
                    group[method] = {"endpoint": points[-1]["validation_nrmse"], "raw_normalized_AULC": area}
                summaries.append({"seed": seed, "budget": budget, **group,
                    "endpoint_delta_nocw_minus_cw": group["nocw"]["endpoint"]-group["cw"]["endpoint"],
                    "AULC_delta_nocw_minus_cw": group["nocw"]["raw_normalized_AULC"]-group["cw"]["raw_normalized_AULC"]})
    result = {"updated_utc": now(), "common_budget_all_five": common, "target_budget": 1485,
        "complete": common==1485, "paired_milestones": summaries, "test_truth_reads": 0}
    atomic(output / "paired_summary.json", result)
    return result


def supervise():
    validate()
    with lock(STUDY / "locks/supervisor.lock"):
        children = {}
        jobs = [("nocw",s) for s in SEEDS] + [("cw",s) for s in SEEDS]
        for method, seed in jobs:
            receipt = STUDY / f"progress/dispatch_{method}_{seed}.json"
            if receipt.exists():
                raise RuntimeError("A previous dispatch requires explicit reconciliation")
            log = STUDY / f"progress/{method}_{seed}.log"
            log.parent.mkdir(exist_ok=True)
            env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", MPLCONFIGDIR="/tmp/nocw_cw_paired_matplotlib",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            with log.open("ab") as output:
                child = subprocess.Popen([sys.executable, str(SCRIPT), "worker", "--method", method, "--seed", str(seed)],
                    cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            children[method,seed] = child
            once(receipt, {"method": method, "seed": seed, "pid": child.pid, "started_utc": now(), "protocol_sha256": sha(POLICY)})
        while True:
            workers = []
            for (method,seed), child in children.items():
                path = STUDY / f"progress/{method}_{seed}.json"
                saved = read(path) if path.exists() else {"phase":"STARTING"}
                workers.append({**saved, "method": method, "seed": seed, "pid": child.pid, "exit_code": child.poll()})
            summary = report()
            atomic(STUDY / "status.json", {"updated_utc": now(), "workers": workers,
                "target_budget": 1485, "common_budget_all_five": summary["common_budget_all_five"],
                "complete": summary["complete"], "supervisor_pid": os.getpid()})
            if all(child.poll() is not None for child in children.values()):
                return {"status": "COMPLETE" if summary["complete"] else "NEEDS_ATTENTION", "workers": workers}
            time.sleep(20)


def launch():
    validate()
    receipt = STUDY / "launch.json"
    if receipt.exists():
        raise RuntimeError("Experiment already dispatched; inspect status instead of launching twice")
    with (STUDY / "supervisor.log").open("ab") as output:
        child = subprocess.Popen([sys.executable, str(SCRIPT), "supervise"], cwd=ROOT,
            env=dict(os.environ), stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    result = {"supervisor_pid": child.pid, "started_utc": now(), "protocol_sha256": sha(POLICY), "target_budget": 1485}
    once(receipt, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare","validate","launch","supervise","worker","report","status"))
    parser.add_argument("--method", choices=("nocw","cw"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action == "worker":
        if args.method is None or args.seed is None:
            parser.error("worker requires --method and --seed")
        try:
            validate()
            (run_nocw if args.method=="nocw" else run_cw)(args.seed)
        except BaseException as error:
            status(args.method, args.seed, "FAILED_CLOSED", error_type=type(error).__name__, reason=str(error)[:500])
            raise
        result = {"status":"COMPLETE", "method":args.method, "seed":args.seed}
    elif args.action == "status":
        result = read(STUDY / "status.json")
    else:
        result = globals()[args.action]()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
