#!/usr/bin/env python3
"""Repair missing SSE record separators without changing scientific requests.

Frozen transport, retry, selection, and historical reconciliation code is reused.
Only an exact completed JSON event followed by another JSON data record permits
inserting a missing blank separator. Malformed/incomplete JSON is not repaired.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
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
sys.path.insert(0, str(ROOT))
from scripts.studies import run_qgeognn_v2_row_llm_nocw_stream as stream

spec = importlib.util.spec_from_file_location("existing_authorized_history_repair", ROOT / "scripts/studies/repair_nocw_history_replay_20261115.py")
history = importlib.util.module_from_spec(spec)
spec.loader.exec_module(history)
cap, m, P = history.cap, history.m, history.P
SCRIPT = Path(__file__).resolve()
TEST = ROOT / "tests/active_learning_v2/test_nocw_sse_frames_20261116.py"
PLAN = P / "sse_frame_repair_plan_20261116.json"
APPROVAL = P / "sse_frame_repair_authorization_20261116.json"
LAUNCH = P / "sse_frame_repair_launch_20261116.json"
CONTROL = P / "sse_frame_repair_status_20261116.json"
ORIGINAL_LIVE = stream.live_call
ORIGINAL_CLIENT = stream.client_for


def complete_event(data):
    try:
        value = json.loads(data)
    except (ValueError, TypeError):
        return False
    return (isinstance(value, dict) and isinstance(value.get("type"), str)
            and (value["type"].startswith("response.") or value["type"] in ("error", "keepalive")))


def reframe_lines(lines, capture, stats):
    pending = []
    for line in lines:
        capture(line)
        if line.startswith("data:"):
            body = line[5:].lstrip(" ")
            if pending and complete_event("\n".join(pending)) and body.lstrip().startswith("{"):
                # A complete JSON object cannot be continued by another object.
                # Supply the missing SSE delimiter; the frozen parser continues
                # to validate and process each event and response itself.
                stats["inserted_blank_separators"] += 1
                yield ""
                pending = []
            pending.append(body)
        elif line == "":
            pending = []
        yield line


class CapturedResponse:
    def __init__(self, response, directory, turn, attempt):
        self.response = response
        self.prefix = Path(directory) / f"stream_{turn:02d}_{attempt:02d}"
        self.path = self.prefix.with_suffix(".wire.jsonl")
        self.handle = None
        self.stats = {"inserted_blank_separators": 0, "received_lines": 0}

    def __getattr__(self, name):
        return getattr(self.response, name)

    def capture(self, line):
        self.handle.write(json.dumps({"received_utc": m.now(), "line": line}, ensure_ascii=False) + "\n")
        self.stats["received_lines"] += 1
        if self.stats["received_lines"] % 50 == 0:
            self.handle.flush()

    def iter_lines(self):
        self.handle = self.path.open("x")
        yield from reframe_lines(self.response.iter_lines(), self.capture, self.stats)

    def finish(self):
        if self.handle is not None and not self.handle.closed:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()
            m.once(self.prefix.with_suffix(".framing_receipt.json"), {
                "finished_utc": m.now(), "wire_file": self.path.name, "wire_sha256": m.sha(self.path),
                "plan_sha256": m.sha(PLAN), "script_sha256": m.sha(SCRIPT), **self.stats,
                "request_body_changed": False, "frozen_completion_validation_reused": True})


class CapturedClient:
    def __init__(self, client, directory, turn, attempt):
        self.client, self.directory, self.turn, self.attempt = client, directory, turn, attempt

    def __enter__(self):
        self.client.__enter__()
        return self

    def __exit__(self, *args):
        return self.client.__exit__(*args)

    @contextmanager
    def stream(self, *args, **kwargs):
        with self.client.stream(*args, **kwargs) as response:
            wrapped = CapturedResponse(response, self.directory, self.turn, self.attempt)
            try:
                yield wrapped
            finally:
                wrapped.finish()


def live_call(messages, config, directory, turn, attempt):
    original_factory = stream.client_for
    stream.client_for = lambda cfg: CapturedClient(ORIGINAL_CLIENT(cfg), directory, turn, attempt)
    try:
        return ORIGINAL_LIVE(messages, config, directory, turn, attempt)
    finally:
        stream.client_for = original_factory


def validate():
    history.validate()
    plan = m.read(PLAN)
    for path, key in [(SCRIPT, "script_sha256"), (TEST, "test_sha256"), (Path(history.__file__), "history_repair_sha256")]:
        if m.sha(path) != plan[key]:
            raise RuntimeError("SSE frame repair source drift")
    approval = m.read(APPROVAL)
    if not approval.get("approved") or approval["plan_sha256"] != m.sha(PLAN):
        raise PermissionError("SSE transport repair has not been authorized")
    return plan


def worker(seed):
    plan = validate()
    if seed not in plan["seeds"]:
        raise RuntimeError("Seed outside transport repair scope")
    deadline = time.monotonic() + 30
    while True:
        job = next(j for j in m.read(cap.STATUS)["jobs"] if j["seed"] == seed)
        if job["pid"] == os.getpid() and job.get("worker_script") == SCRIPT.name:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("Replacement identity was not published; no request sent")
        time.sleep(.1)
    stream.live_call = live_call
    return history.worker(seed)


def safe(job, retry):
    if not cap.alive(job["pid"]) or job.get("worker_script") != Path(history.__file__).name:
        return False
    return cap.safe_backoff(job, retry)


def handover(job, retry, status):
    if not safe(job, retry):
        return None
    info = subprocess.check_output(["ps", "-p", str(job["pid"]), "-o", "state=,command="], text=True)
    if str(Path(history.__file__)) not in info or not re.search(rf"--seed {job['seed']}(\s|$)", info):
        raise RuntimeError("Existing healthy worker identity differs")
    stopped = False
    try:
        os.kill(job["pid"], signal.SIGSTOP)
        stopped = True
        for _ in range(20):
            if subprocess.check_output(["ps", "-p", str(job["pid"]), "-o", "state="], text=True).strip().startswith("T"):
                break
            time.sleep(.02)
        else:
            return None
        current = m.read(P / f"nocw/progress/retry_seed_{job['seed']}.json")
        if current != retry or not safe(job, current):
            return None
        with m.lock(P / f"locks/sse_frame_handover_{job['seed']}.lock"):
            m.once(P / f"progress/sse_frame_handover_seed_{job['seed']}_20261116.json", {
                "verified_utc": m.now(), "seed": job["seed"], "old_pid": job["pid"],
                "retry_state": retry, "plan_sha256": m.sha(PLAN), "unresolved_requests": 0,
                "request_or_training_interrupted": False, "billing_status": "unknown"})
            os.kill(job["pid"], signal.SIGTERM)
    finally:
        if stopped:
            try:
                os.kill(job["pid"], signal.SIGCONT)
            except ProcessLookupError:
                pass
    for _ in range(50):
        if not cap.alive(job["pid"]):
            break
        time.sleep(.1)
    else:
        raise RuntimeError("Old worker did not exit; no replacement launched")
    replacement = {**job, "old_pid": job["pid"], "mode": "limited", "worker_script": SCRIPT.name,
                   "not_before_utc": retry["retry_at_utc"], "handover_utc": m.now()}
    # A short start gate lets the job identity be published before cap.worker
    # reads it. The preserved retry deadline remains the effective call gate.
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", MPLCONFIGDIR="/tmp/nocw_cw_paired_matplotlib",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    log = P / f"progress/nocw_{job['seed']}_sse_frames_20261116.log"
    with log.open("ab") as output:
        child = subprocess.Popen([sys.executable, str(SCRIPT), "worker", "--seed", str(job["seed"])],
                                 cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    with (P / f"progress/keep_awake_sse_frames_{job['seed']}.log").open("ab") as output:
        awake = subprocess.Popen(["/usr/bin/caffeinate", "-is", "-w", str(child.pid)],
                                 stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    replacement.update(pid=child.pid, keep_awake_pid=awake.pid, log=str(log.relative_to(ROOT)), dispatched_utc=m.now())
    status["jobs"] = [replacement if x["seed"] == job["seed"] else x for x in status["jobs"]]
    m.atomic(cap.STATUS, status)
    return replacement


def supervise():
    plan = validate()
    with m.lock(P / "locks/sse_frame_controller_20261116.lock"):
        while True:
            status = m.read(cap.STATUS)
            pending, adopted, excluded = [], [], []
            for job in status["jobs"]:
                if job["seed"] not in plan["seeds"]:
                    continue
                if job.get("worker_script") == SCRIPT.name:
                    adopted.append(job["seed"])
                    continue
                if job["pid"] != plan["source_workers"][str(job["seed"])]["pid"]:
                    excluded.append({"seed": job["seed"], "reason": "worker identity changed; no adoption"})
                    continue
                if not cap.alive(job["pid"]):
                    excluded.append({"seed": job["seed"], "reason": "ended; never automatically restarted"})
                    continue
                retry = m.read(P / f"nocw/progress/retry_seed_{job['seed']}.json")
                result = handover(job, retry, status)
                (adopted if result else pending).append(job["seed"])
            m.atomic(CONTROL, {"updated_utc": m.now(), "controller_pid": os.getpid(),
                              "adopted": adopted, "pending_safe_backoff": pending, "excluded": excluded})
            if not pending:
                return
            time.sleep(1)


def launch():
    validate()
    intent = P / "sse_frame_repair_dispatch_intent_20261116.json"
    with m.lock(P / "locks/sse_frame_dispatch_20261116.lock"):
        if LAUNCH.exists() or intent.exists():
            raise RuntimeError("SSE frame repair already dispatched or pending reconciliation")
        m.once(P / "request_limit3_status_before_sse_frames_20261116.json", m.read(cap.STATUS))
        m.once(intent, {"created_utc": m.now(), "plan_sha256": m.sha(PLAN),
                        "authorization_sha256": m.sha(APPROVAL), "single_dispatch": True})
        env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", MPLCONFIGDIR="/tmp/nocw_cw_paired_matplotlib",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        log = P / "progress/sse_frame_repair_controller_20261116.log"
        with log.open("ab") as output:
            child = subprocess.Popen([sys.executable, str(SCRIPT), "supervise"], cwd=ROOT, env=env,
                                     stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        with (P / "progress/sse_frame_controller_keep_awake_20261116.log").open("ab") as output:
            awake = subprocess.Popen(["/usr/bin/caffeinate", "-is", "-w", str(child.pid)],
                                     stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {"created_utc": m.now(), "controller_pid": child.pid,
                   "controller_keep_awake_pid": awake.pid, "log": str(log.relative_to(ROOT)),
                   "plan_sha256": m.sha(PLAN), "authorization_sha256": m.sha(APPROVAL),
                   "request_limit": 3, "compute_slots": 2, "billing_status": "unknown"}
        m.once(LAUNCH, receipt)
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "supervise", "worker"))
    parser.add_argument("--seed", type=int, choices=(157, 887))
    args = parser.parse_args()
    if args.action == "worker" and args.seed is None:
        parser.error("worker requires --seed")
    if args.action == "worker":
        worker(args.seed)
    elif args.action == "supervise":
        supervise()
    else:
        print(json.dumps(launch(), ensure_ascii=False), flush=True)
