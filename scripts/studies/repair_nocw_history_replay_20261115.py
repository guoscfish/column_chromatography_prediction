#!/usr/bin/env python3
"""Carry approved saved-failure reconciliations through full frozen-history replay.

Offline audit never creates request/answer/selection artifacts or calls a provider.
Restart is separately gated; original failures and successful answers are intact.
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
CAP = ROOT / "scripts/studies/run_nocw_request_limit3_20261115.py"
spec = importlib.util.spec_from_file_location("limited_history_repair_base", CAP)
cap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cap)
auth, m, P = cap.auth, cap.m, cap.P
TEST = ROOT / "tests/active_learning_v2/test_nocw_history_replay_20261115.py"
AUDIT = P / "reports/frozen_history_replay_audit_20261115.json"
PLAN = P / "history_replay_repair_plan_20261115.json"
APPROVAL = P / "history_replay_repair_authorization_20261115.json"
LAUNCH = P / "history_replay_repair_launch_20261115.json"


def approved_entries():
    cap.validate()
    entries = []
    configurations = [
        (auth.r.PLAN, auth.r.APPROVAL, "provider terminal failure code: upstream_error",
         "approved_upstream_failure_reconciliation.json", "recovery_approval_sha256"),
        (auth.PLAN, auth.APPROVAL, "provider HTTP 401",
         "explicit_auth_retry_reconciliation_20261115.json", "approval_sha256")]
    for plan_path, approval_path, error, name, key in configurations:
        plan, approval = m.read(plan_path), m.read(approval_path)
        if not approval.get("approved") or approval["plan_sha256"] != m.sha(plan_path):
            raise RuntimeError("Historical reconciliation lacks its explicit approval")
        for entry in plan["entries"]:
            path = ROOT / entry["terminal_receipt"]
            reconciliation = path.with_name(name)
            value, resolved = m.read(path), m.read(reconciliation)
            if (m.sha(path) != entry["terminal_receipt_sha256"]
                    or value.get("error_message") != error or value.get("success") is not False
                    or value.get("fatal_protocol_error") is not True
                    or value["request_sha256"] != entry["request_sha256"]
                    or resolved["original_failure_sha256"] != m.sha(path)
                    or resolved[key] != m.sha(approval_path)):
                raise RuntimeError("Historical failure/reconciliation lineage drift")
            entries.append({"seed": entry["seed"], "path": str(path), "error": error,
                "failure_sha256": m.sha(path), "request_sha256": entry["request_sha256"],
                "reconciliation": str(reconciliation), "reconciliation_sha256": m.sha(reconciliation)})
    return entries


def history_reader(read, entries):
    lookup = {Path(e["path"]).resolve(): e for e in entries}
    def checked(path):
        value = read(path)
        entry = lookup.get(Path(path).resolve())
        if entry is None:
            return value
        if (m.sha(Path(path)) != entry["failure_sha256"]
                or m.sha(Path(entry["reconciliation"])) != entry["reconciliation_sha256"]
                or value.get("error_message") != entry["error"]
                or value.get("request_sha256") != entry["request_sha256"]
                or value.get("success") is not False or value.get("fatal_protocol_error") is not True):
            raise RuntimeError("Approved historical failure changed")
        return {**value, "fatal_protocol_error": False, "fatal_protocol_error_reconciled": True,
                "reconciliation_sha256": entry["reconciliation_sha256"]}
    return checked


def reader_for_seed(read, entries, seed, include_upstream=True, include_auth=True):
    chosen = [e for e in entries if e["seed"] == seed]
    upstream = [e for e in chosen if "upstream_error" in e["error"]]
    result = history_reader(read, upstream if include_upstream else [])
    if include_auth:
        entry = next(e for e in chosen if e["error"] == "provider HTTP 401")
        result = auth.reader.reconciled_reader(result, entry["path"], entry["failure_sha256"],
            entry["request_sha256"], Path(entry["reconciliation"]), entry["reconciliation_sha256"])
    return result


def offline_audit():
    entries = approved_entries()
    from scripts.studies import run_qgeognn_v2_row_llm_nocw_stream as stream
    original_read, original_once = stream.s.read, stream.s.once
    base_once, retry_call, physical = stream.base.once, stream.retry.request_with_retries, stream.live_call
    def immutable_once(path, value):
        path = Path(path)
        if not path.exists() or original_read(path) != value:
            raise RuntimeError("Offline replay cannot create or modify an artifact: " + str(path))
    def saved_only(messages, config, directory, turn, call):
        request_hash = stream.s.stable_hash({"messages": messages, "config": config})
        receipts = list(Path(directory).glob(f"attempt_{turn:02d}_*.json"))
        if not any(not p.name.endswith(".started.json") and original_read(p).get("success")
                   and original_read(p).get("request_sha256") == request_hash for p in receipts):
            raise RuntimeError("Offline replay has no successful saved response")
        return retry_call(messages, config, directory, turn, call)
    def forbidden(*args, **kwargs):
        raise SystemExit("OFFLINE AUDIT MUST NOT CALL A PROVIDER")
    before, reproduced, checked, source_hashes = {}, [], [], {}
    for seed in auth.SEEDS:
        for path in (P / f"nocw/selections/seed_{seed}/llm32_nocw").glob("round_*/batch_freeze.json"):
            before[str(path)] = m.sha(path)
    try:
        for seed, completed in auth.SEEDS.items():
            old = m.activate_nocw(seed)
            workflow = old.s
            stream.s.once = stream.base.once = immutable_once
            stream.retry.request_with_retries, stream.live_call = saved_only, forbidden
            # Reproduce the deployed bug with only the current401 allowlist.
            stream.s.read = reader_for_seed(original_read, entries, seed, include_upstream=False)
            old_round = next(int(Path(e["path"]).parent.name.split("_")[-1]) for e in entries
                             if e["seed"] == seed and "upstream_error" in e["error"])
            directory = P / f"nocw/selections/seed_{seed}/llm32_nocw/round_{old_round:02d}"
            try:
                workflow.audit_batch(directory)
            except stream.FatalTransportStop as error:
                reproduced.append({"seed": seed, "round": old_round, "error": str(error)})
            else:
                raise RuntimeError("Expected historical fatal-ledger bug did not reproduce")
            stream.s.read = reader_for_seed(original_read, entries, seed)
            prior_labeled, prior_unlabeled, prior_batch = None, None, None
            for rnd in range(completed + 1):
                directory = P / f"nocw/selections/seed_{seed}/llm32_nocw/round_{rnd:02d}"
                frozen = workflow.audit_batch(directory)
                contract = original_read(directory / "contract.json")
                labeled, unlabeled, batch = contract["L_t_ids"], contract["U_t_ids"], frozen["batch_ids"]
                assert len(labeled) == 333 + 32 * rnd and len(set(labeled)) == len(labeled)
                assert len(batch) == len(set(batch)) == 32 and set(batch) <= set(unlabeled)
                assert not set(labeled) & set(batch)
                if prior_labeled is not None:
                    assert labeled == prior_labeled + prior_batch
                    assert unlabeled == [x for x in prior_unlabeled if x not in set(prior_batch)]
                prior_labeled, prior_unlabeled, prior_batch = labeled, unlabeled, batch
                source_hashes[str(directory / "batch_freeze.json")] = m.sha(directory / "batch_freeze.json")
                checked.append({"seed": seed, "round": rnd, "batch_size": 32,
                                "strict_original_selection_replay": True})
    finally:
        stream.s.read, stream.s.once, stream.base.once = original_read, original_once, base_once
        stream.retry.request_with_retries, stream.live_call = retry_call, physical
    assert all(m.sha(Path(path)) == digest for path, digest in before.items())
    result = {"checked_utc": m.now(), "status": "PASS", "historical_failures_reproduced": reproduced,
        "frozen_batches_replayed": len(checked), "batches": checked, "source_hashes": source_hashes,
        "approved_historical_failure_entries": entries, "provider_calls": 0, "new_attempt_markers": 0,
        "source_test_truth_reads": 0, "original_frozen_batches_modified": False,
        "scientific_rules_or_accepted_selection_modified": False}
    m.once(AUDIT, result)
    return result


def prepare():
    audit = m.read(AUDIT)
    if audit["status"] != "PASS" or audit["frozen_batches_replayed"] != 151:
        raise RuntimeError("Full151-batch offline replay is required")
    entries = []
    for seed, rnd in auth.SEEDS.items():
        progress = m.read(P / f"progress/nocw_{seed}.json")
        directory = P / f"nocw/selections/seed_{seed}/llm32_nocw/round_{rnd:02d}"
        if progress.get("phase") != "FAILED_CLOSED" or progress.get("reason") != "recorded fatal protocol error requires reconciliation":
            raise RuntimeError("Only the reviewed stopped history-replay workers may resume")
        if m.last_round(m.NOCW, seed, m.ARM) != rnd:
            raise RuntimeError("Fitted prefix advanced")
        if any((directory / n).exists() for n in ("feedback.json", "label_access_receipt.json")):
            raise RuntimeError("Pending labels were already revealed")
        entries.append({"seed": seed, "completed_rounds": rnd, "completed_budget": 333 + 32 * rnd,
            "batch_freeze": str((directory / "batch_freeze.json").relative_to(ROOT)),
            "batch_freeze_sha256": m.sha(directory / "batch_freeze.json"), "pending_frozen_ids": 32,
            "old_pid": progress["pid"], "progress_sha256": m.sha(P / f"progress/nocw_{seed}.json")})
    plan = {"prepared_utc": m.now(), "entries": entries, "script_sha256": m.sha(SCRIPT),
        "test_sha256": m.sha(TEST), "request_limit_script_sha256": m.sha(CAP), "audit_sha256": m.sha(AUDIT),
        "repair": "Carry already-approved upstream reconciliation through historical replay, alongside the existing saved401 reconciliation. No new fatal code is waived.",
        "first_action": "Reuse the five saved validated32-ID batches and fit; no repeat model calls for these batches",
        "remaining_fits": 34, "request_limit": 3, "independent_compute_slots": 2,
        "scientific_changes": False, "original_records_modified": False, "test_truth_reads": 0,
        "billing_status": "unknown; future necessary selections use existing consent and retry policy",
        "approval_reason": "Monitoring instruction: 不要自动重启异常工作进程"}
    m.once(PLAN, plan)
    return plan


def validate():
    cap.validate()
    plan = m.read(PLAN)
    for path, key in [(SCRIPT, "script_sha256"), (TEST, "test_sha256"), (CAP, "request_limit_script_sha256"), (AUDIT, "audit_sha256")]:
        if m.sha(path) != plan[key]:
            raise RuntimeError("Reviewed history repair drift")
    approval = m.read(APPROVAL) if APPROVAL.exists() else {}
    if not approval.get("approved") or approval.get("plan_sha256") != m.sha(PLAN):
        raise PermissionError("Explicit restart approval for the repaired workers is pending")
    return plan


def worker(seed):
    validate()
    from scripts.studies import run_qgeognn_v2_row_llm_nocw_stream as stream
    # cap.worker/auth.worker will install their one saved401 overlay on top.
    stream.s.read = reader_for_seed(stream.s.read, approved_entries(), seed, include_auth=False)
    return cap.worker(seed)


def launch():
    plan = validate()
    if LAUNCH.exists():
        raise RuntimeError("History repair already dispatched")
    jobs = []
    for entry in plan["entries"]:
        seed = entry["seed"]
        if m.sha(ROOT / entry["batch_freeze"]) != entry["batch_freeze_sha256"]:
            raise RuntimeError("Saved batch changed")
        with m.lock(P / f"locks/nocw_{seed}.lock"):
            pass
    previous = m.read(cap.STATUS)
    m.once(P / "request_limit3_status_before_history_repair_20261115.json", previous)
    for entry in plan["entries"]:
        seed = entry["seed"]
        env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", MPLCONFIGDIR="/tmp/nocw_cw_paired_matplotlib",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        log = P / f"progress/nocw_{seed}_history_repair_20261115.log"
        with log.open("ab") as output:
            child = subprocess.Popen([sys.executable, str(SCRIPT), "worker", "--seed", str(seed)],
                cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        with (P / f"progress/keep_awake_history_repair_{seed}.log").open("ab") as output:
            awake = subprocess.Popen(["/usr/bin/caffeinate", "-is", "-w", str(child.pid)],
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        jobs.append({"seed": seed, "pid": child.pid, "keep_awake_pid": awake.pid, "mode": "limited",
            "worker_script": SCRIPT.name, "log": str(log.relative_to(ROOT)), "not_before_utc": m.now(),
            "dispatched_utc": m.now()})
        # Publish before the new child reaches cap.worker's job lookup.
        combined = jobs + [j for j in previous["jobs"] if j["seed"] not in {x["seed"] for x in jobs}]
        m.atomic(cap.STATUS, {"updated_utc": m.now(), "request_limit": 3, "jobs": combined,
                            "legacy_workers_reserved": 0, "transition_complete": True})
    record = {"created_utc": m.now(), "jobs": jobs, "plan_sha256": m.sha(PLAN),
        "request_limit": 3, "compute_slots": 2, "reuse_saved_pending_batches": True}
    m.once(LAUNCH, record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "prepare", "launch", "worker"))
    parser.add_argument("--seed", type=int, choices=auth.SEEDS)
    args = parser.parse_args()
    if args.action == "worker" and args.seed is None:
        parser.error("worker requires --seed")
    result = worker(args.seed) if args.action == "worker" else offline_audit() if args.action == "audit" else globals()[args.action]()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
