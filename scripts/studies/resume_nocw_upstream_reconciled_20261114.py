#!/usr/bin/env python3
"""Reconcile the exact saved upstream failures under the approved five-seed restart.

Original attempts, requests, scientific sources and the approved plan stay intact.
Only an in-memory copy of each hash-bound upstream failure loses its fatal flag.
The original retry ledger then skips that unsuccessful attempt as usual.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
PARENT = ROOT / "scripts/studies/resume_nocw_upstream_all_five_20261114.py"
spec = importlib.util.spec_from_file_location("approved_five_seed_recovery", PARENT)
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
r, m = parent.r, parent.r.m
SUPPLEMENT = m.STUDY / "upstream_saved_failure_reconciliation_20261114.json"
LAUNCH = m.STUDY / "upstream_reconciled_launch_20261114.json"
TEST = ROOT / "tests/active_learning_v2/test_nocw_saved_upstream_reconciliation_20261114.py"
ERROR = "provider terminal failure code: upstream_error"


def reconciled_reader(read, receipt, receipt_hash, request_hash, reconciliation, reconciliation_hash):
    """Keep all other receipts and fatal guards unchanged, including path aliases."""
    target = Path(receipt).resolve()
    def checked(path):
        value = read(path)
        if Path(path).resolve() != target:
            return value
        if m.sha(target) != receipt_hash or m.sha(reconciliation) != reconciliation_hash:
            raise RuntimeError("Saved upstream reconciliation source drift")
        if (value.get("request_sha256") != request_hash or value.get("success") is not False
                or value.get("fatal_protocol_error") is not True
                or value.get("error_message") != ERROR):
            raise RuntimeError("Reconciliation is restricted to the audited upstream failure")
        return {**value, "fatal_protocol_error": False,
                "fatal_protocol_error_reconciled": True,
                "reconciliation_sha256": reconciliation_hash}
    return checked


def preflight():
    r.require_approval()
    approved = m.read(r.PLAN)
    first = m.read(m.STUDY / "upstream_recovery_launch_20261114.json")
    if first["plan_sha256"] != m.sha(r.PLAN):
        raise RuntimeError("First recovery dispatch plan drift")
    jobs = {job["seed"]: job for job in first["jobs"]}
    entries = []
    for entry in approved["entries"]:
        seed, rnd = entry["seed"], entry["completed_rounds"]
        progress = m.read(ROOT / entry["progress"])
        if (progress.get("pid") != jobs[seed]["pid"]
                or progress.get("phase") != "FAILED_CLOSED"
                or progress.get("reason") != "recorded fatal protocol error requires reconciliation"):
            raise RuntimeError("Only the first dispatch's pre-request reconciliation stop may resume")
        if m.last_round(m.NOCW, seed, m.ARM) != rnd:
            raise RuntimeError("Saved fitted prefix changed")
        with m.lock(m.STUDY / f"locks/nocw_{seed}.lock"):
            pass
        receipt = ROOT / entry["terminal_receipt"]
        directory = receipt.parent
        if m.sha(receipt) != entry["terminal_receipt_sha256"]:
            raise RuntimeError("Original terminal failure receipt changed")
        saved = m.read(receipt)
        if (saved.get("error_message") != ERROR or not saved.get("fatal_protocol_error")
                or saved.get("success") is not False
                or saved["request_sha256"] != entry["request_sha256"]):
            raise RuntimeError("Original failure is not the approved exact upstream code")
        terminal_indices = tuple(int(x) for x in receipt.stem.split("_")[1:])
        for started in directory.glob("attempt_*.started.json"):
            result = started.with_name(started.name.replace(".started.json", ".json"))
            indices = tuple(int(x) for x in result.stem.split("_")[1:])
            if not result.exists() or indices > terminal_indices:
                raise RuntimeError("Unresolved or new request appeared after the approved stop")
        if any((directory / n).exists() for n in ("batch_freeze.json", "feedback.json", "label_access_receipt.json")):
            raise RuntimeError("The pending selection already revealed or froze a batch")
        fatal = [q for q in directory.glob("attempt_*.json")
                 if not q.name.endswith(".started.json") and m.read(q).get("fatal_protocol_error")]
        if fatal != [receipt]:
            raise RuntimeError("Another fatal receipt requires its own reconciliation")
        entries.append({**entry, "first_recovery_pid": jobs[seed]["pid"],
                        "pre_request_stop_sha256": m.sha(ROOT / entry["progress"])})
    return entries


def prepare():
    entries = preflight()
    record = {"prepared_utc": m.now(), "entries": entries,
              "approved_plan_sha256": m.sha(r.PLAN), "approval_sha256": m.sha(r.APPROVAL),
              "first_dispatch_sha256": m.sha(m.STUDY / "upstream_recovery_launch_20261114.json"),
              "script_sha256": m.sha(SCRIPT), "test_sha256": m.sha(TEST),
              "parent_script_sha256": m.sha(PARENT),
              "scope": "Implement the approved upstream_error restart for the existing saved failures; no scope expansion",
              "first_dispatch_new_requests": 0, "original_receipts_modified": False,
              "scientific_changes": False, "test_truth_reads": 0, "billing_status": "unknown",
              "implementation": "Hash-bound additive reconciliation receipt and narrow in-memory read overlay; original retry/correction policy unchanged"}
    m.once(SUPPLEMENT, record)
    return record


def validate():
    r.require_approval()
    record = m.read(SUPPLEMENT)
    for f, key in [(SCRIPT, "script_sha256"), (TEST, "test_sha256"), (PARENT, "parent_script_sha256"),
                   (r.PLAN, "approved_plan_sha256"), (r.APPROVAL, "approval_sha256")]:
        if m.sha(f) != record[key]:
            raise RuntimeError("Saved-failure recovery implementation or approval drift")
    return record


def worker(seed):
    record = validate()
    entry = next(e for e in record["entries"] if e["seed"] == seed)
    receipt = ROOT / entry["terminal_receipt"]
    reconciliation = receipt.with_name("approved_upstream_failure_reconciliation.json")
    m.once(reconciliation, {"original_failure_path": entry["terminal_receipt"],
        "original_failure_sha256": entry["terminal_receipt_sha256"],
        "request_sha256": entry["request_sha256"], "success": False,
        "provider_code": "upstream_error", "accepted_answer": False,
        "recovery_approval_sha256": m.sha(r.APPROVAL), "implementation_sha256": m.sha(SUPPLEMENT),
        "billing_status": "unknown", "handling": "skip saved unsuccessful attempt; exact upstream failures use original network backoff"})
    from scripts.studies import run_qgeognn_v2_row_llm_nocw_stream as stream
    stream.s.read = reconciled_reader(stream.s.read, receipt, entry["terminal_receipt_sha256"],
                                     entry["request_sha256"], reconciliation, m.sha(reconciliation))
    return r.worker(seed)


def launch():
    validate()
    entries = preflight()
    if LAUNCH.exists():
        raise RuntimeError("Reconciled recovery already dispatched; do not launch duplicates")
    jobs = []
    for e in entries:
        seed = e["seed"]
        env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", MPLCONFIGDIR="/tmp/nocw_cw_paired_matplotlib",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        log = m.STUDY / f"progress/nocw_{seed}_reconciled_recovery_20261114.log"
        with log.open("ab") as output:
            child = subprocess.Popen([sys.executable, str(SCRIPT), "worker", "--seed", str(seed)],
                cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        with (m.STUDY / f"progress/keep_awake_reconciled_{seed}.log").open("ab") as output:
            awake = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(child.pid)],
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        jobs.append({"seed": seed, "pid": child.pid, "keep_awake_pid": awake.pid,
                     "log": str(log.relative_to(ROOT)), "dispatched_utc": m.now()})
    result = {"approved_plan_sha256": m.sha(r.PLAN), "implementation_sha256": m.sha(SUPPLEMENT),
              "jobs": jobs, "active_workers_interrupted": False,
              "monitoring": "Use these PIDs and actual frozen fits; original supervisor is finished. Refresh derivative paired reports from saved fits only."}
    m.once(LAUNCH, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "worker"))
    parser.add_argument("--seed", type=int, choices=r.SEEDS)
    args = parser.parse_args()
    if args.action == "worker":
        if args.seed is None:
            parser.error("worker requires --seed")
        result = worker(args.seed)
    else:
        result = globals()[args.action]()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
