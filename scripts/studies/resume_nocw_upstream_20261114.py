#!/usr/bin/env python3
"""Audited, explicitly approved recovery of two stopped NoCW seed workers."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location("paired_frozen_runner", ROOT / "scripts/studies/run_nocw_cw_paired_to1485_20261112.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
SEEDS = {887: 24, 2357: 28}
PLAN = m.STUDY / "upstream_recovery_plan_20261114.json"
APPROVAL = m.STUDY / "upstream_recovery_authorization_20261114.json"


def retry_upstream(call, protocol_error, directory, turn, attempt, *args):
    """Only the exact upstream terminal code changes to existing network backoff."""
    try:
        return call(*args, directory, turn, attempt)
    except protocol_error as error:
        if str(error) != "provider terminal failure code: upstream_error":
            raise
        import httpx
        m.once(Path(directory) / f"upstream_retry_policy_{turn:02d}_{attempt:02d}.json", {
            "provider_code": "upstream_error", "original_error_type": type(error).__name__,
            "handling": "existing persistent network retry/backoff; no answer accepted",
            "recovery_plan_sha256": m.sha(PLAN), "billing_status": "unknown"})
        raise httpx.RemoteProtocolError("provider terminal failure: upstream_error") from error


def preflight():
    m.validate()
    entries = []
    for seed, rnd in SEEDS.items():
        progress_path = m.STUDY / f"progress/nocw_{seed}.json"
        progress = m.read(progress_path)
        if (progress.get("phase") != "FAILED_CLOSED" or
                progress.get("reason") != "provider terminal failure code: upstream_error"):
            raise RuntimeError("Recovery is restricted to the two audited upstream stops")
        if m.last_round(m.NOCW, seed, m.ARM) != rnd:
            raise RuntimeError("NoCW source advanced after the stopped snapshot")
        with m.lock(m.STUDY / f"locks/nocw_{seed}.lock"):
            pass
        directory = m.NOCW / f"selections/seed_{seed}/{m.ARM}/round_{rnd:02d}"
        if any((directory / name).exists() for name in ("batch_freeze.json", "feedback.json", "label_access_receipt.json")):
            raise RuntimeError("Stopped selection already froze or revealed a new batch")
        unresolved = [p for p in directory.glob("attempt_*.started.json")
                      if not p.with_name(p.name.replace(".started.json", ".json")).exists()]
        if unresolved:
            raise RuntimeError("An ambiguous request requires separate reconciliation")
        failed = [p for p in directory.glob("attempt_*.json") if not p.name.endswith(".started.json")
                  and m.read(p).get("fatal_protocol_error")]
        if len(failed) != 1 or m.read(failed[0]).get("error_message") != progress["reason"]:
            raise RuntimeError("Terminal failure receipt does not match the stopped worker")
        entries.append({"seed": seed, "completed_rounds": rnd, "completed_budget": m.BUDGETS[rnd],
            "pending_round": rnd, "stopped_pid": progress["pid"],
            "progress": str(progress_path.relative_to(ROOT)), "progress_sha256": m.sha(progress_path),
            "terminal_receipt": str(failed[0].relative_to(ROOT)), "terminal_receipt_sha256": m.sha(failed[0]),
            "request_sha256": m.read(failed[0])["request_sha256"], "unresolved_requests": 0,
            "new_labels_revealed_for_pending_batch": False})
    return entries


def prepare():
    entries = preflight()
    record = {"prepared_utc": m.now(), "status": "PREPARED_AWAITING_EXPLICIT_RECOVERY_APPROVAL",
        "entries": entries, "endpoint": "https://token4research.cn/v1",
        "model": "gpt-6.1-sol", "reasoning_effort": "xhigh", "target_budget": 1485,
        "only_transport_change": "Treat the exact provider response.failed code upstream_error as a retryable network error using the existing backoff; retain every other fatal guard.",
        "approval_reason": "Monitoring instruction: 不要自动重启异常工作进程",
        "scientific_changes": False, "frozen_source_edits": False,
        "new_provider_requests_during_preparation": 0, "test_truth_reads": 0,
        "active_workers_interrupted": False, "completed_results_deleted": False,
        "recovery_script_sha256": m.sha(SCRIPT),
        "test_sha256": m.sha(ROOT / "tests/active_learning_v2/test_nocw_upstream_recovery_20261114.py")}
    m.once(PLAN, record)
    return record


def require_approval():
    record = m.read(PLAN)
    if (m.sha(SCRIPT) != record["recovery_script_sha256"] or
            m.sha(ROOT / "tests/active_learning_v2/test_nocw_upstream_recovery_20261114.py") != record["test_sha256"]):
        raise RuntimeError("Recovery adapter or tests changed after offline preparation")
    if not APPROVAL.exists():
        raise PermissionError("Explicit recovery approval is pending; no worker may restart")
    approval = m.read(APPROVAL)
    if not approval.get("approved") or approval.get("plan_sha256") != m.sha(PLAN):
        raise PermissionError("Recovery approval does not bind this reviewed plan")
    m.validate()


def worker(seed):
    require_approval()
    from scripts.studies import run_qgeognn_v2_row_llm_nocw_stream as stream
    original = stream.live_call
    def live(messages, config, directory, turn, attempt):
        return retry_upstream(original, stream.ProtocolViolation, directory, turn, attempt, messages, config)
    stream.live_call = live
    try:
        m.run_nocw(seed)
    except BaseException as error:
        m.status("nocw", seed, "FAILED_CLOSED", error_type=type(error).__name__, reason=str(error)[:500])
        raise


def launch():
    require_approval()
    entries = preflight()
    receipt = m.STUDY / "upstream_recovery_launch_20261114.json"
    if receipt.exists():
        raise RuntimeError("Recovery already dispatched; do not restart workers twice")
    jobs = []
    for entry in entries:
        seed = entry["seed"]
        env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", MPLCONFIGDIR="/tmp/nocw_cw_paired_matplotlib",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        log = m.STUDY / f"progress/nocw_{seed}_upstream_recovery_20261114.log"
        with log.open("ab") as output:
            child = subprocess.Popen([sys.executable, str(SCRIPT), "worker", "--seed", str(seed)],
                cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        with (m.STUDY / f"progress/keep_awake_recovery_{seed}.log").open("ab") as output:
            awake = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(child.pid)],
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        jobs.append({"seed": seed, "pid": child.pid, "old_pid": entry["stopped_pid"],
                     "keep_awake_pid": awake.pid, "log": str(log.relative_to(ROOT)), "dispatched_utc": m.now()})
    result = {"plan_sha256": m.sha(PLAN), "jobs": jobs, "active_workers_interrupted": False,
              "original_supervisor_retains_historical_exit_codes": True,
              "monitoring": "Use these recovery PIDs and current per-seed progress files instead of the original supervisor's old exit codes. Refresh final reports from saved fits if the original supervisor exits first."}
    m.once(receipt, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "worker"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action == "worker":
        if args.seed is None:
            parser.error("worker requires --seed")
        result = worker(args.seed)
    else:
        result = globals()[args.action]()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
