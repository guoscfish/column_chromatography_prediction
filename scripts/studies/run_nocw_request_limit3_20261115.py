#!/usr/bin/env python3
"""Gracefully drain legacy requests and bound new physical requests to three.

Only verified between-request failure backoff permits a healthy worker handover.
No active request or training job is cancelled. Scientific sources stay frozen.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
AUTH = ROOT / "scripts/studies/resume_nocw_auth_retry_20261115.py"
spec = importlib.util.spec_from_file_location("authorized_nocw_concurrency_base", AUTH)
auth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auth)
m = auth.m
P = m.STUDY
POLICY = P / "request_limit3_policy_20261115.json"
LAUNCH = P / "request_limit3_launch_20261115.json"
STATUS = P / "request_limit3_status_20261115.json"
TEST = ROOT / "tests/active_learning_v2/test_nocw_request_limit3_20261115.py"
CHILDREN = {}


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def legacy_count(status):
    return sum(j["mode"] == "legacy" for j in status["jobs"])


@contextmanager
def request_slot(seed, turn, attempt, study=None):
    study = Path(study) if study else P
    locks = study / "locks/request_limit3"
    locks.mkdir(parents=True, exist_ok=True)
    gate = study / f"progress/request_gate_seed_{seed}.json"
    gate.parent.mkdir(parents=True, exist_ok=True)
    queued = m.now()
    record = {"seed": seed, "pid": os.getpid(), "turn": turn, "attempt_index": attempt,
              "queued_utc": queued, "phase": "WAITING_FOR_REQUEST_SLOT", "limit": 3}
    m.atomic(gate, {**record, "updated_utc": m.now()})
    held = None
    try:
        while held is None:
            with (locks / "admission.lock").open("a+") as admission:
                fcntl.flock(admission, fcntl.LOCK_EX)
                status = m.read(study / STATUS.name)
                reserved = legacy_count(status)
                waiting = []
                for path in gate.parent.glob("request_gate_seed_*.json"):
                    value = m.read(path)
                    if value.get("phase") == "WAITING_FOR_REQUEST_SLOT" and alive(value["pid"]):
                        waiting.append(value)
                waiting.sort(key=lambda v: (v["queued_utc"], v["seed"]))
                if waiting and waiting[0]["seed"] == seed:
                    for index in range(reserved, 3):
                        candidate = (locks / f"request_{index}.lock").open("a+")
                        try:
                            fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            candidate.close()
                        else:
                            held = candidate
                            m.atomic(gate, {**record, "phase": "REQUEST_ACTIVE", "slot": index,
                                "legacy_workers_reserved": reserved, "updated_utc": m.now()})
                            break
                fcntl.flock(admission, fcntl.LOCK_UN)
            if held is None:
                time.sleep(0.2)
        yield
    finally:
        if held is not None:
            fcntl.flock(held, fcntl.LOCK_UN)
            held.close()
        m.atomic(gate, {**record, "phase": "REQUEST_RELEASED", "updated_utc": m.now()})


def validate():
    auth.validate()
    policy = m.read(POLICY)
    for path, key in [(SCRIPT, "script_sha256"), (TEST, "test_sha256"), (AUTH, "auth_script_sha256")]:
        if m.sha(path) != policy[key]:
            raise RuntimeError("Request concurrency implementation drift")
    return policy


def worker(seed):
    validate()
    job = next(j for j in m.read(STATUS)["jobs"] if j["seed"] == seed)
    not_before = datetime.fromisoformat(job["not_before_utc"])
    while datetime.now(timezone.utc) < not_before:
        time.sleep(min(1, max(0, (not_before - datetime.now(timezone.utc)).total_seconds())))
    from scripts.studies import run_qgeognn_v2_row_llm_nocw_stream as stream
    original = stream.live_call
    def limited(messages, config, directory, turn, attempt):
        with request_slot(seed, turn, attempt):
            return original(messages, config, directory, turn, attempt)
    stream.live_call = limited
    return auth.worker(seed)


def safe_backoff(job, retry, study=None):
    study = Path(study) if study else P
    if retry.get("pid") != job["pid"] or retry.get("phase") != "WAITING_RETRY":
        return False
    if (datetime.fromisoformat(retry["retry_at_utc"]) - datetime.now(timezone.utc)).total_seconds() < 3:
        return False
    progress = m.read(study / f"progress/nocw_{job['seed']}.json")
    if progress.get("pid") != job["pid"] or progress.get("phase") != "SELECTING":
        return False
    root = study / f"nocw/selections/seed_{job['seed']}/llm32_nocw"
    for started in root.glob("round_*/attempt_*.started.json"):
        if not started.with_name(started.name.replace(".started.json", ".json")).exists():
            return False
    return True


def dispatch(job):
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", MPLCONFIGDIR="/tmp/nocw_cw_paired_matplotlib",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    log = P / f"progress/nocw_{job['seed']}_limit3_20261115.log"
    with log.open("ab") as output:
        child = subprocess.Popen([sys.executable, str(SCRIPT), "worker", "--seed", str(job["seed"])],
            cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    with (P / f"progress/keep_awake_limit3_{job['seed']}.log").open("ab") as output:
        awake = subprocess.Popen(["/usr/bin/caffeinate", "-is", "-w", str(child.pid)],
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    CHILDREN[child.pid] = child
    return {**job, "mode": "limited", "old_pid": job["pid"], "pid": child.pid,
            "keep_awake_pid": awake.pid, "worker_script": SCRIPT.name,
            "log": str(log.relative_to(ROOT)), "dispatched_utc": m.now()}


def supervise():
    validate()
    with m.lock(P / "locks/request_limit3_controller.lock"):
        status = m.read(STATUS)
        while True:
            for index, job in enumerate(status["jobs"]):
                progress = m.read(P / f"progress/nocw_{job['seed']}.json")
                child = CHILDREN.get(job["pid"])
                running = child.poll() is None if child else alive(job["pid"])
                if not running:
                    job["mode"] = "complete" if progress.get("phase") == "COMPLETE" else "failed_no_restart"
                    continue
                if job["mode"] != "legacy":
                    continue
                retry = m.read(P / f"nocw/progress/retry_seed_{job['seed']}.json")
                if not safe_backoff(job, retry):
                    continue
                info = subprocess.check_output(["ps", "-p", str(job["pid"]), "-o", "state=,command="], text=True)
                if str(AUTH) not in info or not re.search(rf"--seed {job['seed']}(\s|$)", info):
                    raise RuntimeError("Legacy worker identity changed")
                stopped = False
                try:
                    os.kill(job["pid"], signal.SIGSTOP)
                    stopped = True
                    for _ in range(20):
                        if subprocess.check_output(["ps", "-p", str(job["pid"]), "-o", "state="], text=True).strip().startswith("T"):
                            break
                        time.sleep(0.02)
                    else:
                        continue
                    current = m.read(P / f"nocw/progress/retry_seed_{job['seed']}.json")
                    if not safe_backoff(job, current) or current != retry:
                        continue
                    receipt = P / f"progress/request_limit3_handover_seed_{job['seed']}.json"
                    m.once(receipt, {"seed": job["seed"], "old_pid": job["pid"], "retry_state": current,
                        "verified_utc": m.now(), "unresolved_requests": 0, "active_request_interrupted": False,
                        "policy_sha256": m.sha(POLICY), "billing_status": "unknown"})
                    os.kill(job["pid"], signal.SIGTERM)
                finally:
                    if stopped:
                        try:
                            os.kill(job["pid"], signal.SIGCONT)
                        except ProcessLookupError:
                            pass
                for _ in range(50):
                    if not alive(job["pid"]):
                        break
                    time.sleep(0.1)
                else:
                    raise RuntimeError("Legacy worker did not exit; no replacement launched")
                job["mode"] = "handover"
                job["not_before_utc"] = retry["retry_at_utc"]
                m.atomic(STATUS, status)
                status["jobs"][index] = dispatch(job)
                m.atomic(STATUS, status)
            status.update(updated_utc=m.now(), legacy_workers_reserved=legacy_count(status),
                          transition_complete=legacy_count(status) == 0)
            m.atomic(STATUS, status)
            if all(j["mode"] in ("complete", "failed_no_restart") for j in status["jobs"]):
                return status
            time.sleep(0.5)


def prepare():
    auth.validate()
    source = m.read(auth.LAUNCH)
    jobs = [{**j, "mode": "legacy", "worker_script": AUTH.name} for j in source["jobs"]]
    policy = {"created_utc": m.now(), "user_instruction": "请求的时候最多三个并行，训练可以多个一起训练",
        "request_limit": 3, "request_fairness": "FIFO admission", "training": "existing independent two compute slots; no request slot held while training",
        "legacy_transition": "Let in-flight requests finish; handover only during verified completed-failure backoff with no unresolved request. Reserve one request slot per remaining legacy worker.",
        "scientific_changes": False, "active_requests_cancelled": False,
        "script_sha256": m.sha(SCRIPT), "test_sha256": m.sha(TEST), "auth_script_sha256": m.sha(AUTH),
        "source_launch_sha256": m.sha(auth.LAUNCH), "billing_status": "unknown"}
    m.once(POLICY, policy)
    m.once(STATUS, {"updated_utc": m.now(), "request_limit": 3, "jobs": jobs,
        "legacy_workers_reserved": len(jobs), "transition_complete": False})
    return policy


def launch():
    validate()
    if LAUNCH.exists():
        raise RuntimeError("Concurrency controller already launched")
    log = P / "progress/request_limit3_controller_20261115.log"
    with log.open("ab") as output:
        child = subprocess.Popen([sys.executable, str(SCRIPT), "supervise"], cwd=ROOT,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    with (P / "progress/keep_awake_request_limit3_controller.log").open("ab") as output:
        awake = subprocess.Popen(["/usr/bin/caffeinate", "-is", "-w", str(child.pid)],
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    record = {"created_utc": m.now(), "controller_pid": child.pid, "keep_awake_pid": awake.pid,
        "policy_sha256": m.sha(POLICY), "log": str(log.relative_to(ROOT)),
        "status_file": STATUS.name, "request_limit": 3}
    m.once(LAUNCH, record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "supervise", "worker"))
    parser.add_argument("--seed", type=int, choices=auth.SEEDS)
    args = parser.parse_args()
    if args.action == "worker" and args.seed is None:
        parser.error("worker requires --seed")
    value = worker(args.seed) if args.action == "worker" else globals()[args.action]()
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)
