#!/usr/bin/env python3
"""Audited SSE transport and safe, between-request migration for five NoCW seeds.

No scientific source, request body, existing attempt, or frozen protocol is edited.
No background generation, response retrieval, probe, or additional model call.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.studies import run_qgeognn_v2_row_llm_nocw_persistent as retry

base, s = retry.base, retry.s
STUDY = base.STUDY
SCRIPT = Path(__file__).resolve()
POLICY = STUDY / "stream_transport_amendment.json"
TEST = s.ROOT / "tests/active_learning_v2/test_nocw_stream_transport.py"


class ProtocolViolation(RuntimeError):
    """A completed response with the wrong identity/status/tools is never used."""


class FatalTransportStop(SystemExit):
    pass


def prepare():
    retry.activate()
    valid = retry.validate()
    record = {
        "version": "audited_sse_transport_v1", "base_protocol_sha256": valid["protocol_sha256"],
        "persistent_retry_amendment_sha256": s.sha256_file(retry.AMENDMENT),
        "user_authorization": "帮我处理一下（修复长请求传输与流事件记录）",
        "request_body": "same model/input/reasoning/tools/store/max_output_tokens/sampling_parameters; stream=true",
        "route": "direct; proxy=None; trust_env=False; same endpoint and independent key mapping",
        "tcp_keepalive_seconds": 60, "http_read_timeout": "unchanged 600 seconds of socket inactivity",
        "observability": "metadata only: response id, event type/count/cursor/timestamps; no reasoning or input text",
        "durability": "validated completed answer saved before closing stream; recover this local receipt without resending",
        "handover": "retain inflight workers; stop only in verified retry backoff with zero unresolved started markers",
        "retry_and_correction": "persistent network retry policy and three answer correction budget unchanged",
        "background": False, "store": False, "paid_probes": False,
        "code_hashes": {str(SCRIPT.relative_to(s.ROOT)): s.sha256_file(SCRIPT),
                        str(TEST.relative_to(s.ROOT)): s.sha256_file(TEST)},
    }
    base.once(POLICY, record)
    return record


def validate():
    result = retry.validate()
    policy = s.read(POLICY)
    if (policy["base_protocol_sha256"] != result["protocol_sha256"] or
            policy["persistent_retry_amendment_sha256"] != s.sha256_file(retry.AMENDMENT)):
        raise RuntimeError("stream amendment protocol drift")
    for path, digest in policy["code_hashes"].items():
        if s.sha256_file(s.ROOT / path) != digest:
            raise RuntimeError("stream amendment source drift")
    return result


def activate(seed=None):
    retry.activate(seed)
    s.validate = validate
    s.request_with_retries = request_with_retries


def body_for(messages, config):
    return {"model": config["model"], "input": messages,
            "reasoning": {"effort": config["reasoning_effort"]}, "tools": [], "store": False,
            "max_output_tokens": config["max_output_tokens"], **config["sampling_parameters"], "stream": True}


def completed_result(response, config):
    model = response.get("model", "")
    if (response.get("status") != "completed" or not isinstance(model, str) or
            not (model == config["model"] or model.startswith(config["model"] + "-")) or
            any(item.get("type") not in ("message", "reasoning") for item in response.get("output", []))):
        raise ProtocolViolation("incomplete/wrong-model/tool response; fail closed")
    answer = "".join(part["text"] for item in response.get("output", []) if item.get("type") == "message"
                     for part in item.get("content", []) if part.get("type") == "output_text")
    if not answer.strip() or not response.get("id"):
        raise ProtocolViolation("completed response missing answer or response id")
    try:
        base.once(STUDY / "served_model_lock.json", {"served_model": model})
    except RuntimeError as error:
        raise ProtocolViolation("served model differs from locked identity") from error
    return answer, {"response_id": response["id"], "served_model": model,
                    "usage": response.get("usage"), "native_tool_calls": 0}


def client_for(config):
    import httpx
    opts = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    for name in ("TCP_KEEPIDLE", "TCP_KEEPALIVE"):
        if hasattr(socket, name):
            opts.append((socket.IPPROTO_TCP, getattr(socket, name), 60))
            break
    transport = httpx.HTTPTransport(retries=0, socket_options=opts, trust_env=False)
    return httpx.Client(transport=transport, trust_env=False, timeout=config["timeout_seconds"])


def live_call(messages, config, directory, turn, attempt):
    """Use the same SSE request as Free, preserving every frozen body parameter."""
    import httpx
    from src.qgeognn_al.active_learning_v2.scientist_transport import provider_key
    s.require_authorization()
    prefix = Path(directory) / f"stream_{turn:02d}_{attempt:02d}"
    request_hash = s.stable_hash({"messages": messages, "config": config})
    state = {"seed": retry.WORKER_SEED, "pid": os.getpid(), "turn": turn, "attempt_index": attempt,
             "request_sha256": request_hash, "amendment_sha256": s.sha256_file(POLICY),
             "phase": "CONNECTING", "started_utc": retry.now(), "event_count": 0, "line_count": 0,
             "response_id": None, "last_sequence_number": None}
    last_write = 0
    durable_result = None

    def persist(force=False):
        nonlocal last_write
        if force or time.monotonic() - last_write >= 15:
            state["updated_utc"] = retry.now()
            base.atomic_json(prefix.with_suffix(".json"), state)
            if retry.WORKER_SEED is not None:
                base.atomic_json(STUDY / f"progress/stream_seed_{retry.WORKER_SEED}.json", state)
            last_write = time.monotonic()

    def event(data):
        nonlocal durable_result
        kind = data.get("type", "unknown")
        state.update(event_count=state["event_count"] + 1, last_event_type=kind,
                     last_event_utc=retry.now())
        if isinstance(data.get("sequence_number"), int) and kind != "keepalive":
            state["last_sequence_number"] = data["sequence_number"]
        if isinstance(data.get("response"), dict) and data["response"].get("id"):
            state["response_id"] = data["response"]["id"]
        state["phase"] = {"response.created": "CREATED", "response.queued": "QUEUED",
                          "response.in_progress": "IN_PROGRESS"}.get(kind, state["phase"])
        persist(force=kind in ("response.created", "response.queued", "response.in_progress"))
        if kind == "response.completed":
            answer, provenance = completed_result(data["response"], config)
            record = {"request_sha256": request_hash, "answer": answer, "provenance": provenance,
                      "amendment_sha256": s.sha256_file(POLICY), "validated_completed": True}
            base.once(prefix.with_suffix(".completed.json"), record)
            durable_result = answer, provenance
            state["phase"] = "COMPLETED"
            persist(True)
            return answer, provenance
        if kind == "response.failed":
            code = data.get("response", {}).get("error", {}).get("code", "unknown")
            if code in ("server_error", "rate_limit_exceeded"):
                raise httpx.RemoteProtocolError("provider terminal failure: " + code)
            raise ProtocolViolation("provider terminal failure code: " + str(code)[:80])
        if kind == "response.incomplete":
            raise ProtocolViolation("provider terminal event: " + kind)
        if kind == "error":
            # Only type/code, never provider message (which may echo sensitive input).
            code = data.get("code", "unknown")
            if code in ("invalid_api_key", "insufficient_quota", "model_not_found", "invalid_request_error"):
                raise ProtocolViolation("provider stream error code: " + code)
            raise httpx.RemoteProtocolError("provider stream error: " + str(code)[:80])
        return None

    persist(True)
    try:
        with client_for(config) as client:
            with client.stream("POST", config["base_url"].rstrip("/") + "/responses",
                               headers={"Authorization": "Bearer " + provider_key(config),
                                        "Accept": "text/event-stream"}, json=body_for(messages, config)) as response:
                state.update(http_status=response.status_code, headers_received_utc=retry.now(), phase="CONNECTED")
                persist(True)
                if response.status_code in (400, 401, 403, 404, 405, 422):
                    raise ProtocolViolation("provider HTTP " + str(response.status_code))
                response.raise_for_status()
                data_lines = []
                for line in response.iter_lines():
                    state.update(line_count=state["line_count"] + 1, last_line_utc=retry.now())
                    persist()
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip(" "))
                    elif line == "" and data_lines:
                        payload, data_lines = "\n".join(data_lines), []
                        if payload.strip() == "[DONE]":
                            continue
                        result = event(json.loads(payload))
                        if result is not None:
                            return result  # Do not wait for trailer/EOF after durable completion.
                if data_lines and "\n".join(data_lines).strip() != "[DONE]":
                    result = event(json.loads("\n".join(data_lines)))
                    if result is not None:
                        return result
                raise httpx.RemoteProtocolError("stream ended before response.completed")
    except Exception as error:
        if durable_result is not None:
            # The complete, validated answer is authoritative even if __exit__
            # fails while closing the already completed HTTP connection.
            return durable_result
        state.update(phase="FAILED", error_type=type(error).__name__, ended_utc=retry.now())
        persist(True)
        raise


def request_with_retries(messages, config, directory, turn, ignored_call):
    """Keep the existing retry ledger and budget; add safe local completion recovery."""
    directory = Path(directory)
    request_hash = s.stable_hash({"messages": messages, "config": config})
    for completed in directory.glob(f"stream_{turn:02d}_*.completed.json"):
        attempt = int(completed.name.split("_")[2].split(".")[0])
        receipt = directory / f"attempt_{turn:02d}_{attempt:02d}.json"
        if receipt.exists():
            continue
        saved = s.read(completed)
        started = receipt.with_suffix(".started.json")
        if (not started.exists() or s.read(started)["request_sha256"] != request_hash or
                saved["request_sha256"] != request_hash or not saved.get("validated_completed") or
                saved["amendment_sha256"] != s.sha256_file(POLICY)):
            raise RuntimeError("local completed response does not match frozen request")
        model = saved["provenance"]["served_model"]
        if not (model == config["model"] or model.startswith(config["model"] + "-")):
            raise ProtocolViolation("local completed response model mismatch")
        base.once(STUDY / "served_model_lock.json", {"served_model": model})
        base.once(receipt, {"request_sha256": request_hash, "success": True, "answer": saved["answer"],
            "provenance": saved["provenance"], "elapsed_seconds": max(0, completed.stat().st_mtime-started.stat().st_mtime),
            "recovered_local_completion": True, "stream_amendment_sha256": s.sha256_file(POLICY)})

    def call(m, c):
        indices = [int(p.name.split("_")[2].split(".")[0]) for p in directory.glob(f"attempt_{turn:02d}_*.started.json")
                   if not p.with_name(p.name.replace(".started.json", ".json")).exists()]
        if len(indices) != 1:
            raise FatalTransportStop("unexpected unresolved request ledger")
        attempt = indices[0]
        begin = time.monotonic()
        try:
            return live_call(m, c, directory, turn, attempt)
        except ProtocolViolation as error:
            base.once(directory / f"attempt_{turn:02d}_{attempt:02d}.json", {
                "request_sha256": request_hash, "success": False, "fatal_protocol_error": True,
                "error_type": type(error).__name__, "error_message": str(error),
                "elapsed_seconds": time.monotonic()-begin, "stream_amendment_sha256": s.sha256_file(POLICY)})
            raise FatalTransportStop(str(error)) from error

    # Never replay a recorded protocol failure as a transient network retry.
    for p in directory.glob(f"attempt_{turn:02d}_*.json"):
        if re.fullmatch(r"attempt_\d+_\d+\.json", p.name) and s.read(p).get("fatal_protocol_error"):
            raise FatalTransportStop("recorded fatal protocol error requires reconciliation")
    return retry.request_with_retries(messages, config, directory, turn, call)


def unresolved(seed):
    root = STUDY / f"selections/seed_{seed}/llm32_nocw"
    return [str(p.relative_to(STUDY)) for p in root.glob("round_*/attempt_*.started.json")
            if not p.with_name(p.name.replace(".started.json", ".json")).exists()]


def process_info(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "stat=,command="], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def migrate_backoff(seed, progress, retry_state):
    """Freeze and recheck before terminating a sleeping old worker, never a call."""
    pid = progress.get("pid")
    if not (pid and retry_state.get("pid") == pid and retry_state.get("phase") == "WAITING_RETRY"):
        return False
    if unresolved(seed):
        return False
    try:
        info = process_info(pid)
    except subprocess.CalledProcessError:
        return False
    if str(SCRIPT) in info:
        return False
    if str(retry.SCRIPT) not in info or f"--seed {seed}" not in info:
        raise RuntimeError("old worker process identity mismatch")
    os.kill(pid, signal.SIGSTOP)
    try:
        for _ in range(20):
            if process_info(pid).startswith("T"):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("could not verify old worker stopped")
        current = s.read(STUDY / f"progress/retry_seed_{seed}.json")
        if current.get("pid") != pid or current.get("phase") != "WAITING_RETRY" or unresolved(seed):
            return False
        base.once(STUDY / f"progress/stream_handover_seed_{seed}_{pid}.json", {
            "seed": seed, "old_pid": pid, "verified_utc": retry.now(), "unresolved_started": 0,
            "retry_state": current, "action": "stop old worker during completed-failure backoff"})
        os.kill(pid, signal.SIGTERM)
        return True
    finally:
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass


def status():
    result = retry.status()
    for record in result["workers"]:
        seed = record["seed"]
        p = STUDY / f"progress/stream_seed_{seed}.json"
        record["stream"] = s.read(p) if p.exists() else None
        record["unresolved_started"] = len(unresolved(seed))
    p = STUDY / "progress/stream_supervisor.json"
    result["stream_supervisor"] = s.read(p) if p.exists() else None
    return result


def supervise():
    def stop_controller(signum, frame):
        # Unwind migrate_backoff's finally block if stopped during handover.
        raise SystemExit("stream controller stopped")
    signal.signal(signal.SIGTERM, stop_controller)
    activate()
    s.require_authorization()
    base.credential_audit()
    with base.BASE_EXCLUSIVE_LOCK(STUDY / "locks/stream_supervisor.lock"):
        # Owning this same legacy lock guarantees the old launcher cannot compete.
        with base.BASE_EXCLUSIVE_LOCK(STUDY / "locks/persistent_supervisor.lock"):
            children = {}
            handed = {seed for seed in base.SEEDS if list((STUDY / "progress").glob(f"stream_handover_seed_{seed}_*.json"))}
            last_validation = 0
            while True:
                if time.monotonic() - last_validation >= 15:
                    validate()
                    last_validation = time.monotonic()
                records = []
                for seed in base.SEEDS:
                    p = STUDY / f"progress/seed_{seed}.json"
                    progress = s.read(p) if p.exists() else {}
                    record = {"seed": seed, "phase": progress.get("phase"), "pid": progress.get("pid"),
                              "round": progress.get("round"), "current_budget": progress.get("current_budget")}
                    busy = retry.seed_busy(seed)
                    if progress.get("phase") == "COMPLETE":
                        record["supervision"] = "COMPLETE"
                    elif seed in children:
                        child = children[seed]
                        record.update(child_pid=child.pid, supervision="STREAM_WORKER_RUNNING" if child.poll() is None
                                      else "NEEDS_RECONCILIATION", exit_code=child.returncode)
                    elif busy:
                        r = STUDY / f"progress/retry_seed_{seed}.json"
                        rs = s.read(r) if r.exists() else {}
                        if seed not in handed and migrate_backoff(seed, progress, rs):
                            handed.add(seed)
                            record["supervision"] = "HANDOVER_AT_SAFE_BACKOFF"
                        else:
                            record["supervision"] = "WAITING_FOR_INFLIGHT_LEGACY_WORKER"
                    elif unresolved(seed):
                        record["supervision"] = "NEEDS_RECONCILIATION"
                    elif (STUDY / f"progress/stream_worker_seed_{seed}.json").exists():
                        record["supervision"] = "NEEDS_RECONCILIATION"
                    elif seed in handed or (progress.get("phase") == "FAILED_CLOSED" and
                            "transport attempt budget exhausted" in progress.get("reason", "")):
                        rp = STUDY / f"progress/retry_seed_{seed}.json"
                        rs = s.read(rp) if rp.exists() else {}
                        until = rs.get("retry_at_utc") if rs.get("phase") == "WAITING_RETRY" else None
                        if until and datetime.fromisoformat(until) > datetime.now(timezone.utc):
                            record.update(supervision="WAITING_EXISTING_BACKOFF", retry_at_utc=until)
                            records.append(record)
                            continue
                        env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="2",
                                   MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
                        env.pop("SCIENTIST_API_KEY", None)
                        with (STUDY / f"progress/seed_{seed}_stream.log").open("ab") as out:
                            child = subprocess.Popen([sys.executable, str(SCRIPT), "worker", "--seed", str(seed),
                                "--allow-paid-llm", "--allow-training", "--allow-label-reveal"], cwd=s.ROOT,
                                env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
                        children[seed] = child
                        base.atomic_json(STUDY / f"progress/stream_worker_seed_{seed}.json", {
                            "seed": seed, "pid": child.pid, "started_utc": retry.now(),
                            "amendment_sha256": s.sha256_file(POLICY)})
                        record.update(child_pid=child.pid, supervision="STREAM_WORKER_DISPATCHED")
                    else:
                        record["supervision"] = "NEEDS_RECONCILIATION"
                    records.append(record)
                base.atomic_json(STUDY / "progress/stream_supervisor.json", {
                    "pid": os.getpid(), "updated_utc": retry.now(), "workers": records,
                    "amendment_sha256": s.sha256_file(POLICY)})
                if all(r["supervision"] == "COMPLETE" for r in records):
                    return {"status": "ALL_FIVE_COMPLETE"}
                time.sleep(2)


def launch():
    activate()
    validate()
    s.require_authorization()
    # Stop the controller only. It never sends a model request or holds a worker lease.
    old = s.read(STUDY / "progress/persistent_supervisor.json")["pid"]
    if not (retry.seed_busy(base.SEEDS[0]) or (STUDY / "progress/stream_supervisor.json").exists()):
        raise RuntimeError("unexpected legacy state; inspect before migration")
    with base.BASE_EXCLUSIVE_LOCK(STUDY / "locks/stream_launch.lock"):
        with (STUDY / "locks/stream_supervisor.lock").open("a+") as handle:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("stream supervisor already running; do not launch a duplicate")
            fcntl.flock(handle, fcntl.LOCK_UN)
        try:
            info = process_info(old)
        except subprocess.CalledProcessError:
            info = ""
        if info:
            if str(retry.SCRIPT) not in info or " supervise " not in info:
                raise RuntimeError("legacy supervisor identity mismatch")
            os.kill(old, signal.SIGTERM)
        with (STUDY / "progress/stream_supervisor.log").open("ab") as out:
            child = subprocess.Popen([sys.executable, str(SCRIPT), "supervise", "--allow-paid-llm",
                "--allow-training", "--allow-label-reveal"], cwd=s.ROOT, env=dict(os.environ),
                stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        record = {"pid": child.pid, "started_utc": retry.now(), "old_controller_pid": old,
                  "amendment_sha256": s.sha256_file(POLICY), "inflight_workers_interrupted": False}
        base.once(STUDY / f"progress/stream_launch_{child.pid}.json", record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="status", choices=("prepare", "validate", "status", "launch", "supervise", "worker", "report"))
    parser.add_argument("--seed", type=int, choices=base.SEEDS)
    parser.add_argument("--budget", type=int, choices=(525, 1005), default=1005)
    for flag in ("allow-paid-llm", "allow-training", "allow-label-reveal"):
        parser.add_argument("--" + flag, action="store_true")
    a = parser.parse_args()
    if a.action in ("launch", "supervise", "worker") and not (a.allow_paid_llm and a.allow_training and a.allow_label_reveal):
        parser.error("all three explicit cost/reveal/training flags required")
    if a.action == "prepare":
        result = prepare()
    elif a.action == "validate":
        activate(); result = validate()
    elif a.action == "worker":
        if a.seed is None:
            parser.error("worker requires --seed")
        activate(a.seed)
        result = base.worker(a.seed, allow_paid_llm=True, allow_training=True, allow_reveal=True)
    elif a.action == "report":
        result = retry.report(a.budget)
    else:
        result = globals()[a.action]()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
