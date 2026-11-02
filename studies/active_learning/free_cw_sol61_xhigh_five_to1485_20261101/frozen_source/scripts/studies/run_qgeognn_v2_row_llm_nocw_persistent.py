#!/usr/bin/env python3
"""User-authorized persistent transport retries; frozen scientific protocol intact."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.studies import run_qgeognn_v2_row_llm_nocw_five as base

s = base.s
STUDY = base.STUDY
SCRIPT = Path(__file__).resolve()
AMENDMENT = STUDY / "persistent_retry_amendment.json"
BASE_VALIDATE = s.validate
BASE_REPORT = base.report
WORKER_SEED = None


class AmbiguousRequestError(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare():
    base.activate()
    validation = BASE_VALIDATE()
    record = {"version": "persistent_transport_retry_v1", "base_protocol_sha256": validation["protocol_sha256"],
        "user_authorization": "一直失败就一直重试", "authorized_utc": now(),
        "maximum_transport_attempts": None, "backoff_seconds": [30, 60, 120, 240, 300],
        "scope": "retry completed transport failures indefinitely; no reset of attempts or request hashes",
        "inflight_requests": "wait for existing worker; never run two workers for a seed",
        "ambiguous_started_without_result": "stop for reconciliation; do not delete or resend silently",
        "scientific_validation": "unchanged; malformed/illegal plans never replaced or accepted",
        "selection_correction_budget": "unchanged at 3 answers", "key_model_endpoint": "unchanged",
        "base_config_note": "frozen config retains original transport_attempts=3; this explicit amendment supersedes that limit only",
        "fees": "failed/timeout requests may be billed; repeated requests now explicitly authorized",
        "code_hashes": {str(SCRIPT.relative_to(s.ROOT)): s.sha256_file(SCRIPT),
            "tests/active_learning_v2/test_nocw_persistent_retry.py":
                s.sha256_file(s.ROOT / "tests/active_learning_v2/test_nocw_persistent_retry.py")}}
    if AMENDMENT.exists():
        old = s.read(AMENDMENT)
        for name in record:
            if name != "authorized_utc" and old.get(name) != record[name]:
                raise RuntimeError("retry amendment drift")
        return old
    base.once(AMENDMENT, record)
    return record


def validate():
    result = BASE_VALIDATE()
    policy = s.read(AMENDMENT)
    if policy["base_protocol_sha256"] != result["protocol_sha256"]:
        raise RuntimeError("retry amendment bound to a different protocol")
    for path, digest in policy["code_hashes"].items():
        if s.sha256_file(s.ROOT / path) != digest:
            raise RuntimeError("retry amendment source drift")
    return result


def activate(seed=None):
    global WORKER_SEED
    WORKER_SEED = seed
    base.activate(seed)
    s.validate = validate
    s.request_with_retries = request_with_retries


def safe_error(error):
    message = str(error)
    for path in base.KEY_FILES.values():
        value = path.read_text().strip()
        if value:
            message = message.replace(value, "[REDACTED]")
    message = re.sub(r"(?i)Bearer\s+\S+|sk-[A-Za-z0-9_-]+", "[REDACTED]", message)
    return message[:700]


def retry_status(**details):
    if WORKER_SEED is not None:
        base.atomic_json(STUDY / f"progress/retry_seed_{WORKER_SEED}.json",
                         {"seed": WORKER_SEED, "pid": os.getpid(), "updated_utc": now(), **details})


def request_with_retries(messages, config, directory, turn, call):
    directory = Path(directory)
    request_hash = s.stable_hash({"messages": messages, "config": config})
    policy_hash = s.sha256_file(AMENDMENT)
    attempt, consecutive_new_failures = 0, 0
    while True:
        path = directory / f"attempt_{turn:02d}_{attempt:02d}.json"
        started = path.with_suffix(".started.json")
        if path.exists():
            saved = s.read(path)
            if saved["request_sha256"] != request_hash:
                raise RuntimeError("request drift")
            if saved.get("success"):
                return saved
            attempt += 1
            continue
        if started.exists():
            raise AmbiguousRequestError("ambiguous interrupted provider request; preserve marker and reconcile before resend")
        # File/protocol/permission errors fail before incurring another request.
        s.require_authorization()
        s.once(started, {"request_sha256": request_hash, "retry_amendment_sha256": policy_hash})
        retry_status(phase="REQUESTING", turn=turn, attempt_index=attempt,
                     request_sha256=request_hash, amendment_sha256=policy_hash)
        begin = time.monotonic()
        try:
            answer, provenance = call(copy.deepcopy(messages), copy.deepcopy(config))
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("empty answer")
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            saved = {"request_sha256": request_hash, "success": False,
                "retry_amendment_sha256": policy_hash, "error_type": type(error).__name__,
                "error_message": safe_error(error), "http_status": getattr(error, "status_code", None),
                "cause_type": type(error.__cause__).__name__ if error.__cause__ else None,
                "elapsed_seconds": time.monotonic()-begin}
            s.once(path, saved)
            consecutive_new_failures += 1
            delay = min(300, 30 * 2**min(consecutive_new_failures-1, 4))
            retry_status(phase="WAITING_RETRY", turn=turn, attempt_index=attempt,
                last_error_type=saved["error_type"], last_error_message=saved["error_message"],
                retry_at_utc=(datetime.now(timezone.utc)+timedelta(seconds=delay)).isoformat(),
                backoff_seconds=delay, amendment_sha256=policy_hash)
            time.sleep(delay)
            attempt += 1
            continue
        saved = {"request_sha256": request_hash, "success": True, "answer": answer,
                 "provenance": provenance, "elapsed_seconds": time.monotonic()-begin,
                 "retry_amendment_sha256": policy_hash}
        s.once(path, saved)
        retry_status(phase="RESPONSE_COMPLETE", turn=turn, attempt_index=attempt,
                     amendment_sha256=policy_hash)
        return saved


def worker(seed, allow_paid_llm=False, allow_training=False, allow_reveal=False):
    activate(seed)
    return base.worker(seed, allow_paid_llm=allow_paid_llm,
                       allow_training=allow_training, allow_reveal=allow_reveal)


def seed_busy(seed):
    p = STUDY / f"locks/worker_seed_{seed}.lock"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
        return False


def status():
    records = []
    for seed in base.SEEDS:
        p = STUDY / f"progress/seed_{seed}.json"
        progress = s.read(p) if p.exists() else {}
        retry = STUDY / f"progress/retry_seed_{seed}.json"
        records.append({"seed": seed, "worker_lock_held": seed_busy(seed),
            "progress": progress, "retry": s.read(retry) if retry.exists() else None})
    return {"workers": records, "retry_policy": "persistent with backoff", "updated_utc": now()}


def report(budget):
    # The base validation report uses the same scientific scores; augment costs
    # with every physical attempt, including indices above99.
    activate()
    result = BASE_REPORT(budget)
    all_attempts = []
    for seed in base.SEEDS:
        root = STUDY / f"selections/seed_{seed}/llm32_nocw"
        for p in sorted(root.glob("round_*/attempt_*.json")):
            if not re.fullmatch(r"attempt_\d+_\d+\.json", p.name):
                continue
            if int(p.parent.name.split("_")[-1]) >= (budget-333)//32:
                continue
            a = s.read(p)
            all_attempts.append({"seed": seed, "path": str(p.relative_to(STUDY)),
                "success": a["success"], "usage": a.get("provenance", {}).get("usage"),
                "elapsed_seconds": a["elapsed_seconds"]})
    result["all_physical_attempts"] = all_attempts
    result["retry_amendment_sha256"] = s.sha256_file(AMENDMENT)
    base.once(STUDY / f"reports/validation_to{budget}_persistent_retry.json", result)
    return result


def supervise(allow_paid_llm=False, allow_training=False, allow_reveal=False):
    if not (allow_paid_llm and allow_training and allow_reveal):
        raise PermissionError("all explicit paid/reveal/training flags required")
    activate()
    s.require_authorization()
    base.credential_audit()
    with base.BASE_EXCLUSIVE_LOCK(STUDY / "locks/persistent_supervisor.lock"):
        children = {}
        launch = {"launched_utc": now(), "supervisor_pid": os.getpid(),
            "amendment_sha256": s.sha256_file(AMENDMENT), "original_launch_sha256": s.sha256_file(STUDY / "launch.json"),
            "existing_workers": "leave inflight requests untouched; take over only after worker lock releases"}
        original = STUDY / "persistent_retry_launch.json"
        if not original.exists():
            base.once(original, launch)
        elif s.read(original)["amendment_sha256"] != launch["amendment_sha256"]:
            raise RuntimeError("supervisor retry policy drift")
        base.once(STUDY / f"progress/supervisor_sessions/{os.getpid()}.json", launch)
        while True:
            validate()
            records = []
            for seed in base.SEEDS:
                path = STUDY / f"progress/seed_{seed}.json"
                progress = s.read(path) if path.exists() else {}
                busy = seed_busy(seed)
                record = {"seed": seed, "worker_lock_held": busy, "phase": progress.get("phase"),
                          "pid": progress.get("pid"), "round": progress.get("round")}
                if progress.get("phase") == "COMPLETE":
                    records.append(record)
                    continue
                if seed in children and children[seed].poll() is None:
                    record.update(child_pid=children[seed].pid, supervision="CHILD_RUNNING")
                elif busy:
                    record["supervision"] = "EXISTING_WORKER_RUNNING"
                elif seed in children:
                    record.update(supervision="NEEDS_RECONCILIATION", child_exit_code=children[seed].returncode)
                elif progress.get("phase") == "FAILED_CLOSED" and "transport attempt budget exhausted" in progress.get("reason", ""):
                    env = os.environ.copy()
                    env.pop("SCIENTIST_API_KEY", None)
                    env.update(SCIENTIST_API_KEY_FILE=str(base.KEY_FILES[seed]), KMP_DUPLICATE_LIB_OK="TRUE",
                               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
                    with (STUDY / f"progress/seed_{seed}_persistent.log").open("ab") as out:
                        child = subprocess.Popen([sys.executable, str(SCRIPT), "worker", "--seed", str(seed),
                            "--allow-paid-llm", "--allow-training", "--allow-label-reveal"],
                            cwd=s.ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
                    children[seed] = child
                    record.update(child_pid=child.pid, supervision="RESUMING_WITH_EXISTING_LEDGER")
                else:
                    record["supervision"] = "NEEDS_RECONCILIATION"
                records.append(record)
            base.atomic_json(STUDY / "progress/persistent_supervisor.json", {
                "updated_utc": now(), "pid": os.getpid(), "workers": records,
                "retry_amendment_sha256": s.sha256_file(AMENDMENT)})
            if all(r["phase"] == "COMPLETE" for r in records):
                return {"status": "ALL_FIVE_COMPLETE", "target_budget": 1005}
            time.sleep(15)


def launch(allow_paid_llm=False, allow_training=False, allow_reveal=False):
    if not (allow_paid_llm and allow_training and allow_reveal):
        raise PermissionError("all explicit paid/reveal/training flags required")
    activate()
    s.require_authorization()
    log = STUDY / "progress/persistent_supervisor.log"
    with log.open("ab") as out:
        child = subprocess.Popen([sys.executable, str(SCRIPT), "supervise", "--allow-paid-llm",
            "--allow-training", "--allow-label-reveal"], cwd=s.ROOT, env=os.environ.copy(),
            stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    return {"status": "SUPERVISOR_DISPATCHED", "pid": child.pid}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", nargs="?", default="status", choices=("prepare", "validate", "status", "launch", "supervise", "worker", "report"))
    p.add_argument("--seed", type=int, choices=base.SEEDS)
    p.add_argument("--budget", type=int, choices=(525, 1005), default=1005)
    p.add_argument("--allow-paid-llm", action="store_true")
    p.add_argument("--allow-training", action="store_true")
    p.add_argument("--allow-label-reveal", action="store_true")
    a = p.parse_args()
    if a.action == "prepare": result = prepare()
    elif a.action == "status": result = status()
    elif a.action == "report": result = report(a.budget)
    elif a.action == "validate":
        activate(); result = validate()
    else:
        if a.action == "worker" and a.seed is None:
            p.error("worker requires --seed")
        kwargs = dict(allow_paid_llm=a.allow_paid_llm, allow_training=a.allow_training, allow_reveal=a.allow_label_reveal)
        result = worker(a.seed, **kwargs) if a.action == "worker" else globals()[a.action](**kwargs)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
