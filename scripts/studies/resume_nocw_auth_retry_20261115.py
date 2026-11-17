#!/usr/bin/env python3
"""Explicit user-requested retry of five saved401 stops; future401 stays fatal."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
READER = ROOT / "scripts/studies/resume_nocw_upstream_reconciled_20261114.py"
spec = importlib.util.spec_from_file_location("auth_retry_saved_failure_reader", READER)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
reader.ERROR = "provider HTTP 401"  # Memory only; the frozen historical adapter is unchanged.
r, m = reader.r, reader.m
SEEDS = {157: 28, 887: 26, 2357: 31, 6101: 30, 12203: 31}
PLAN = m.STUDY / "auth_retry_plan_20261115.json"
APPROVAL = m.STUDY / "auth_retry_authorization_20261115.json"
LAUNCH = m.STUDY / "auth_retry_launch_20261115.json"
TEST = ROOT / "tests/active_learning_v2/test_nocw_auth_retry_20261115.py"
KEY_NAMES = {157: "seed157-continuation.api-key", 887: "dual-expert-813.api-key",
             2357: "dual-expert-813-seed6101.api-key", 6101: "seed6101-token4research.api-key",
             12203: "free-12203-20261023.api-key"}


def credentials():
    paths = {s: Path.home() / ".config/qgeognn-scientist" / name for s, name in KEY_NAMES.items()}
    values = [paths[s].read_bytes().strip() for s in SEEDS]
    if any(not value or any(chr(c).isspace() for c in value) for value in values) or len(set(values)) != 5:
        raise RuntimeError("Existing five distinct single-token key files must be valid locally")
    return {str(s): {"path": str(path), "mtime": path.stat().st_mtime} for s, path in paths.items()}


def preflight():
    m.validate()
    entries = []
    for seed, rnd in SEEDS.items():
        progress_path = m.STUDY / f"progress/nocw_{seed}.json"
        progress = m.read(progress_path)
        if progress.get("phase") != "FAILED_CLOSED" or progress.get("reason") != reader.ERROR:
            raise RuntimeError("Only the reviewed saved401 stops may retry")
        if m.last_round(m.NOCW, seed, m.ARM) != rnd:
            raise RuntimeError("Saved fitted prefix changed")
        with m.lock(m.STUDY / f"locks/nocw_{seed}.lock"):
            pass
        directory = m.NOCW / f"selections/seed_{seed}/{m.ARM}/round_{rnd:02d}"
        if any((directory / n).exists() for n in ("batch_freeze.json", "feedback.json", "label_access_receipt.json")):
            raise RuntimeError("Pending batch is already frozen or revealed")
        for started in directory.glob("attempt_*.started.json"):
            if not started.with_name(started.name.replace(".started.json", ".json")).exists():
                raise RuntimeError("Unresolved request requires separate reconciliation")
        fatal = [q for q in directory.glob("attempt_*.json")
                 if re.fullmatch(r"attempt_\d+_\d+\.json", q.name) and m.read(q).get("fatal_protocol_error")]
        if len(fatal) != 1:
            raise RuntimeError("Expected exactly one saved fatal receipt")
        saved = m.read(fatal[0])
        if saved.get("error_message") != reader.ERROR or saved.get("success") is not False:
            raise RuntimeError("Saved error differs from the explicit authentication retry")
        entries.append({"seed": seed, "completed_rounds": rnd, "completed_budget": 333 + 32 * rnd,
            "stopped_pid": progress["pid"], "progress_sha256": m.sha(progress_path),
            "terminal_receipt": str(fatal[0].relative_to(ROOT)), "terminal_receipt_sha256": m.sha(fatal[0]),
            "request_sha256": saved["request_sha256"], "unresolved_requests": 0,
            "pending_batch_labels_revealed": False})
    return entries, credentials()


def prepare():
    entries, keys = preflight()
    record = {"prepared_utc": m.now(), "entries": entries, "credential_files": keys,
        "script_sha256": m.sha(SCRIPT), "reader_sha256": m.sha(READER), "test_sha256": m.sha(TEST),
        "endpoint": "https://token4research.cn/v1", "model": "gpt-6.1-sol", "reasoning_effort": "xhigh",
        "initial_fitted_rounds": 146, "remaining_rounds": 34, "target_budget": 1485,
        "authorization_needed": "Explicit user request to retry after401; external data consent remains unchanged",
        "semantics": "Reconcile only the five hash-bound saved401 receipts. Future401 remains fatal. Existing upstream/network retries remain unchanged. Real next selections, no discarded model probes.",
        "scientific_changes": False, "original_receipts_modified": False, "test_truth_reads": 0,
        "billing_status": "unknown"}
    m.once(PLAN, record)
    return record


def validate():
    m.validate()
    plan = m.read(PLAN)
    for path, key in [(SCRIPT, "script_sha256"), (READER, "reader_sha256"), (TEST, "test_sha256")]:
        if m.sha(path) != plan[key]:
            raise RuntimeError("Authentication retry implementation drift")
    approval = m.read(APPROVAL)
    if not approval.get("approved") or approval.get("plan_sha256") != m.sha(PLAN) or approval.get("seeds") != list(SEEDS):
        raise PermissionError("Explicit five-seed retry approval is missing or mismatched")
    credentials()
    return plan


def worker(seed):
    plan = validate()
    entry = next(e for e in plan["entries"] if e["seed"] == seed)
    receipt = ROOT / entry["terminal_receipt"]
    reconciliation = receipt.with_name("explicit_auth_retry_reconciliation_20261115.json")
    m.once(reconciliation, {"original_failure_sha256": entry["terminal_receipt_sha256"],
        "request_sha256": entry["request_sha256"], "original_error": reader.ERROR, "success": False,
        "accepted_answer": False, "approval_sha256": m.sha(APPROVAL), "plan_sha256": m.sha(PLAN),
        "future_http401": "fatal stop retained", "billing_status": "unknown"})
    from scripts.studies import run_qgeognn_v2_row_llm_nocw_stream as stream
    stream.s.read = reader.reconciled_reader(stream.s.read, receipt, entry["terminal_receipt_sha256"],
        entry["request_sha256"], reconciliation, m.sha(reconciliation))
    return r.worker(seed)


def launch():
    plan = validate()
    entries, keys = preflight()
    if entries != plan["entries"] or LAUNCH.exists():
        raise RuntimeError("Retry state advanced or already dispatched")
    jobs = []
    for e in entries:
        seed = e["seed"]
        env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", MPLCONFIGDIR="/tmp/nocw_cw_paired_matplotlib",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        log = m.STUDY / f"progress/nocw_{seed}_auth_retry_20261115.log"
        with log.open("ab") as output:
            child = subprocess.Popen([sys.executable, str(SCRIPT), "worker", "--seed", str(seed)],
                cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        with (m.STUDY / f"progress/keep_awake_auth_retry_{seed}.log").open("ab") as output:
            awake = subprocess.Popen(["/usr/bin/caffeinate", "-is", "-w", str(child.pid)],
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        jobs.append({"seed": seed, "pid": child.pid, "keep_awake_pid": awake.pid,
                     "log": str(log.relative_to(ROOT)), "dispatched_utc": m.now()})
    result = {"plan_sha256": m.sha(PLAN), "approval_sha256": m.sha(APPROVAL), "jobs": jobs,
              "future_http401": "fatal stop retained", "active_workers_interrupted": False}
    m.once(LAUNCH, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "worker"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action == "worker" and args.seed is None:
        parser.error("worker requires --seed")
    result = worker(args.seed) if args.action == "worker" else globals()[args.action]()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
