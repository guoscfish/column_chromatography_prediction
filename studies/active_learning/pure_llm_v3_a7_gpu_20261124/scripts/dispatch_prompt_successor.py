#!/usr/bin/env python3
"""Dispatch one approved seed 887 prompt-successor request through the live Mac Key Pool."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import read_bridge_json, verify_job
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import (
    Dispatcher, RemotePaused, _exclusive, _private_directory, verify_local_source,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import OFFLINE
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.key_pool import LEASE, KeyPool, load_pool
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import sdk_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import (
    ATTEMPT_ID, GPU_PYTHON, SHA256, CourierUnavailable, SSHCourier, verify_info, verify_list,
)

PILOT_CONTRACT_SHA256 = "24f01791f4e6e16e9b644dc2165fe1557eeaa7030087df2b726485ad9878e410"
ORIGINAL_ATTEMPT = "e8642075f00b422690e769f1921c03e4"
PROMPT = OFFLINE / "PROMPT_PURE_LLM_V3.md"
DATA_BINDINGS = ("checkpoint_sha256", "packet_sha256", "wire_packet_sha256", "mapping_sha256",
                 "state_sha256", "split_sha256", "wire_representation")
SELECTION = "one_shot_selection.json"
JOURNAL_DIRS = ("jobs", "slots", "attempts", "results", "delivered")


def _until(operation, deadline, poll, description):
    while True:
        try:
            item = operation()
        except CourierUnavailable:
            item = None
        if item is not None:
            return item
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(description)
        time.sleep(min(poll, remaining))


def _pool(pilot_spool):
    _private_directory(pilot_spool)
    binding = read_bridge_json(pilot_spool / "run_binding.json")
    if binding.get("contract_sha256") != PILOT_CONTRACT_SHA256 or binding.get("mock") is not False:
        raise PermissionError("the shared Key Pool must belong to the original live pilot")
    old = read_bridge_json(pilot_spool / "jobs" / f"{ORIGINAL_ATTEMPT}.json")
    request, transport, _ = verify_job(old, contract_sha256=PILOT_CONTRACT_SHA256,
                                       authorization_sha256=binding["authorization_sha256"])
    pool_dir = pilot_spool / "key_pool"
    _private_directory(pool_dir)
    _private_directory(pool_dir / "tickets")
    config = load_pool(Path.home() / ".config/purellm/key_pool.json")
    pool = KeyPool(pool_dir, config)
    status = pool.status(readonly=True)
    if (not status.get("initialized") or status["limits"]["cap"] != config.parallelism
            or {row["id"] for row in status["keys"]} != set(config.keys)):
        raise PermissionError("local key configuration differs from the initialized live pilot pool")
    return pool, request, transport


def _approved_prompt_sha(info):
    prompt = ROOT / PROMPT
    if file_digest(prompt) != info["contract"]["source_sha256"].get(PROMPT.as_posix()):
        raise PermissionError("local updated prompt differs from the independently approved successor")
    return digest(prompt.read_text().strip())


def _validate(dispatcher, job, original, prompt_sha):
    request, _, started = dispatcher.validate(job)
    source, updated = original["binding"], request["binding"]
    messages = request["payload"]["input"]
    if ((updated["seed"], updated["round"], request["logical_index"], started["attempt"]) != (887, 3, 0, 0)
            or [message["role"] for message in messages] != ["system", "user"]
            or any(updated[field] != source[field] for field in DATA_BINDINGS)
            or messages[1] != original["payload"]["input"][1]
            or updated["prompt_sha256"] != prompt_sha or updated["prompt_sha256"] == source["prompt_sha256"]):
        raise PermissionError("one-shot job must contain only the updated prompt and original seed 887 L429 state")
    return job


def _selection(dispatcher, courier, validate, expected_attempt, deadline, poll):
    path = dispatcher.path / SELECTION
    if path.exists():
        selected = read_bridge_json(path)
        if (type(selected) is not dict or set(selected) != {"attempt_id", "job_sha256"}
                or not ATTEMPT_ID.fullmatch(selected["attempt_id"])
                or not SHA256.fullmatch(selected["job_sha256"])
                or expected_attempt is not None and selected["attempt_id"] != expected_attempt):
            raise PermissionError("the immutable one-shot selection differs from the requested attempt")
        identity = selected["attempt_id"]
        local = dispatcher.path / "jobs" / f"{identity}.json"
        job = (read_bridge_json(local) if local.exists() else
               _until(lambda: courier.get(identity), deadline, poll, "staged job is still unavailable"))
        validate(job)
        if (job["attempt_id"], job["job_sha256"]) != (identity, selected["job_sha256"]):
            raise PermissionError("the pinned one-shot job changed")
        return job
    if any(any((dispatcher.path / name).iterdir()) for name in JOURNAL_DIRS):
        raise PermissionError("the one-shot spool contains work with no immutable selection")

    def pending():
        inventory = verify_list(courier.list(), dispatcher.info)
        if inventory["paused"]:
            raise RemotePaused("the successor coordinator is paused")
        ids = inventory["attempt_ids"]
        if not ids:
            return None
        if len(ids) != 1 or expected_attempt is not None and ids[0] != expected_attempt:
            raise PermissionError("the successor must publish only the one approved attempt")
        job = validate(courier.get(ids[0]))
        if job["attempt_id"] != ids[0]:
            raise PermissionError("the fetched job differs from the pending attempt")
        return job

    job = _until(pending, deadline, poll, "no approved one-shot job was published in time")
    once(path, {"attempt_id": job["attempt_id"], "job_sha256": job["job_sha256"]})
    return job


def _check_journal_scope(spool, identity):
    suffix = f"{identity}.json"
    entries = (suffix, "887-3-0-0.json", identity, suffix, suffix)
    for directory, expected in zip(JOURNAL_DIRS, entries, strict=True):
        if any(path.name != expected for path in (spool / directory).iterdir()):
            raise PermissionError("the one-shot spool contains a foreign attempt")


def _check_seed_lease(pool):
    lease = LEASE.get()
    if lease is None or lease.pool is not pool:
        raise PermissionError("one-shot dispatch requires the live pilot Key Pool lease")
    with pool._status_connection(readonly=True) as connection:
        if connection.execute("SELECT 1 FROM queue WHERE seed=887 AND state IN ('queued','busy') AND ticket!=?",
                              (lease.ticket,)).fetchone():
            raise PermissionError("another seed 887 attempt already holds or awaits a Key")


def run(args):
    deadline = time.monotonic() + args.wait_seconds
    pilot_spool = args.pilot_spool.expanduser().resolve(strict=True)
    spool = args.spool_dir.expanduser().resolve()
    if spool == pilot_spool or spool.is_relative_to(pilot_spool) or pilot_spool.is_relative_to(spool):
        raise PermissionError("the successor needs a separate private Mac spool")
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = _until(courier.info, deadline, args.poll, "the successor is unreachable over SSH")
    verify_info(info, expected_contract=args.contract_sha256, expected_auth=args.authorization_sha256)
    if info["contract"].get("run_directory") != args.remote_run_dir:
        raise PermissionError("the authorized successor run directory differs")
    verify_local_source(info, root=ROOT)
    prompt_sha = _approved_prompt_sha(info)
    pool, original, pilot_transport = _pool(pilot_spool)
    transport = TransportConfig(**info["contract"]["profile"]["transport"])
    if transport != pilot_transport or not sdk_status(transport)["matches"]:
        raise PermissionError("the one-shot transport and SDK must match the live pilot")

    dispatcher = Dispatcher(spool, info=info, contract_sha256=args.contract_sha256,
                            transport=transport, mock=False, pool=pool)

    def validate(job):
        return _validate(dispatcher, job, original, prompt_sha)

    def before_call(identity, job):
        if dispatcher.check_remote(courier.info())["paused"]:
            raise RemotePaused("the successor coordinator is paused")
        if validate(courier.claim(identity, dispatcher.dispatcher_id)) != job:
            raise PermissionError("the durable remote claim returned another job")
        _check_seed_lease(pool)

    with _exclusive(spool / "dispatcher.lock"):
        once(spool / "provider_model_lock.json", read_bridge_json(pilot_spool / "provider_model_lock.json"))
        job = _selection(dispatcher, courier, validate, args.attempt_id, deadline, args.poll)
        _check_journal_scope(spool, job["attempt_id"])
        identity = dispatcher.accept(job)
        dispatcher.recover_captured()
        status = _until(lambda: dispatcher.dispatch(identity, before_call=before_call,
                        pause=lambda: time.monotonic() >= deadline), deadline, args.poll,
                        "the selected request could not dispatch before its wait expired")
        view = {"seed": 887, "round": 3, "logical_index": 0, "attempt_id": identity,
                "request_id": job["request_id"], "status": status}
        if status != "sealed":
            return view
        result = dispatcher.result(identity)
        if identity in dispatcher.undelivered():
            ack = _until(lambda: courier.put(identity, dispatcher.dispatcher_id, result), deadline, args.poll,
                         "result is durably captured on Mac and awaits SSH delivery")
            dispatcher.mark_delivered(identity, result, ack)
        outcome = result["outcome"]
        return {**view, "delivered": True, "provider_dispatched": result["provider_dispatched"],
                "outcome": outcome["status"], "failure": outcome.get("failure", {}).get("code")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", default="eit-hpc")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--remote-python", default=GPU_PYTHON)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--pilot-spool", type=Path, required=True)
    parser.add_argument("--spool-dir", type=Path, required=True)
    parser.add_argument("--attempt-id")
    parser.add_argument("--wait-seconds", type=float, default=4500)
    parser.add_argument("--poll", type=float, default=2)
    args = parser.parse_args()
    if (not SHA256.fullmatch(args.contract_sha256) or not SHA256.fullmatch(args.authorization_sha256)
            or args.attempt_id is not None and not ATTEMPT_ID.fullmatch(args.attempt_id)
            or not 0 < args.poll <= args.wait_seconds or not math.isfinite(args.wait_seconds)):
        parser.error("provide exact authorization hashes, an optional attempt ID, and finite positive waits")
    os.umask(0o077)
    report = run(args)
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("outcome") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
