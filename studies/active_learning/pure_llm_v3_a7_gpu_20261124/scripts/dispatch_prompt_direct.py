#!/usr/bin/env python3
"""Dispatch one approved direct-network repeat from the original seed state."""
import argparse
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import (
    read_bridge_json, validate_result, verify_job,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import (
    Dispatcher, RemotePaused, _exclusive, _private_directory, verify_local_source,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    STUDY, RequestTimeouts, request_timeouts,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.key_pool import LEASE
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import sdk_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import (
    ATTEMPT_ID, GPU_PYTHON, SHA256, SSHCourier, verify_info,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.dispatch_prompt_successor import (
    DATA_BINDINGS, _approved_prompt_sha, _pool, _selection, _until,
)

JOURNAL_DIRS = ("jobs", "slots", "attempts", "results", "delivered")


@dataclass(frozen=True)
class ParentEvidence:
    contract_sha256: str
    round_index: int
    attempt_ids: tuple[str, ...]


PARENTS = {
    157: ParentEvidence("95ea110912318583f7d5fe06846d424366a48fa4f86782cbee34fc7f739b1f72", 1,
                        ("04250380f9664bc49fb92653e12d452a",)),
    887: ParentEvidence("9bde1b5347e8c867a5add1a7e6281f21f814773bbeddd532ddc9b648501af5e4", 4,
                        ("d2bcf8042a3e47bba3d0dc23e1e5503e", "0ee3b4971d004591950706e7a1392ea7")),
    2357: ParentEvidence("d62a74be435f2cc69c634948502cba3e53e339cf65ccbc85447f6131d72ea0ac", 1,
                         ("2ecb1d4afd264d9fab22a28ee0f336f7",)),
}
DIRECT_PROVIDER = STUDY / "a7_gpu/provider.py"
DIRECT_PROVIDER_SHA256 = "a0b4d47dc973aa5368acead4dc0157ac230f91cc39bf78b8794f468bf1623c11"
HARD_DEADLINE = 3600
DIRECT_TIMEOUTS = RequestTimeouts(total_seconds=HARD_DEADLINE)


def _parent(seed, spool):
    _private_directory(spool)
    evidence = PARENTS[seed]
    binding = read_bridge_json(spool / "run_binding.json")
    if binding.get("contract_sha256") != evidence.contract_sha256 or binding.get("mock") is not False:
        raise PermissionError("the parent spool does not belong to the approved seed continuation")
    original = None
    for attempt, attempt_id in enumerate(evidence.attempt_ids):
        checked = _parent_attempt(seed, evidence, spool, binding["authorization_sha256"], attempt, attempt_id)
        if original is not None and checked != original:
            raise PermissionError("the parent physical attempts must share the exact request and configuration")
        original = checked
    return original


def _parent_attempt(seed, evidence, spool, authorization_sha, attempt, attempt_id):
    filename = f"{attempt_id}.json"
    job = read_bridge_json(spool / "jobs" / filename)
    request, transport, started = verify_job(job, contract_sha256=evidence.contract_sha256,
                                              authorization_sha256=authorization_sha)
    result = validate_result(read_bridge_json(spool / "results" / filename), job)
    ack = read_bridge_json(spool / "delivered" / filename)
    failure = result["outcome"].get("failure", {})
    coords = request["binding"]["seed"], request["binding"]["round"], request["logical_index"], started["attempt"]
    if (coords != (seed, evidence.round_index, 0, attempt) or started["attempt_id"] != attempt_id
            or result["provider_dispatched"] is not True or failure.get("ambiguous") is not True
            or result["raw_response"] is not None
            or (spool / "attempts" / attempt_id / "raw_provider_response.json").exists()
            or ack != {"attempt_id": attempt_id, "result_sha256": result["result_sha256"]}):
        raise PermissionError("the parent must retain the exact unresolved request and bridge receipt")
    return request, transport


def _direct_transport(info, seed, parent_transport, pilot_transport):
    profile = info["contract"]["profile"]
    boundaries = profile.get("resume_boundaries", {})
    expected = replace(parent_transport, timeout_seconds=HARD_DEADLINE, transport_attempts=1, logical_replies=1)
    transport = TransportConfig(**profile["transport"])
    if (parent_transport != pilot_transport or transport != expected
            or request_timeouts(profile) != DIRECT_TIMEOUTS or set(boundaries) != {str(seed)}
            or boundaries[str(seed)].get("round") != PARENTS[seed].round_index):
        raise PermissionError("direct execution needs the approved single-attempt seed scope and timeouts")
    if (info["contract"]["source_sha256"].get(DIRECT_PROVIDER.as_posix()) != DIRECT_PROVIDER_SHA256
            or file_digest(ROOT / DIRECT_PROVIDER) != DIRECT_PROVIDER_SHA256
            or not sdk_status(transport)["matches"]):
        raise PermissionError("the approved direct HTTPS provider and SDK must be installed")
    return transport


def _validate(dispatcher, job, original, seed, prompt_sha):
    request, _, started = dispatcher.validate(job)
    source, updated = original["binding"], request["binding"]
    coords = updated["seed"], updated["round"], request["logical_index"], started["attempt"]
    if (coords != (seed, PARENTS[seed].round_index, 0, 0)
            or [m["role"] for m in request["payload"]["input"]] != ["system", "user"]
            or updated["prompt_sha256"] != prompt_sha or updated["prompt_sha256"] != source["prompt_sha256"]
            or request["payload_sha256"] != original["payload_sha256"] or request["payload"] != original["payload"]
            or any(updated[field] != source[field] for field in DATA_BINDINGS)
            or request["request_id"] == original["request_id"]):
        raise PermissionError("the direct attempt must carry the exact original prompt and seed-state payload")
    return job


def _check_journal_scope(spool, identity, seed):
    filename = f"{identity}.json"
    entries = (filename, f"{seed}-{PARENTS[seed].round_index}-0-0.json", identity, filename, filename)
    for directory, expected in zip(JOURNAL_DIRS, entries, strict=True):
        if any(path.name != expected for path in (spool / directory).iterdir()):
            raise PermissionError("the direct one-shot spool contains a foreign attempt")


def _check_seed_lease(pool, seed):
    lease = LEASE.get()
    if lease is None or lease.pool is not pool:
        raise PermissionError("direct dispatch requires the original shared Key Pool")
    with pool._status_connection(readonly=True) as connection:
        if connection.execute("SELECT 1 FROM queue WHERE seed=? AND state IN ('queued','busy') AND ticket!=?",
                              (seed, lease.ticket)).fetchone():
            raise PermissionError("another attempt for the same seed already holds or awaits a Key")


def run(args):
    deadline = time.monotonic() + args.wait_seconds
    pilot = args.pilot_spool.expanduser().resolve(strict=True)
    parent = args.parent_spool.expanduser().resolve(strict=True)
    spool = args.spool_dir.expanduser().resolve()
    if any(spool.is_relative_to(existing) or existing.is_relative_to(spool) for existing in (pilot, parent)):
        raise PermissionError("direct dispatch needs a new private spool separate from the existing runs")
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = _until(courier.info, deadline, args.poll, "the direct successor is unreachable over SSH")
    verify_info(info, expected_contract=args.contract_sha256, expected_auth=args.authorization_sha256)
    if info["contract"].get("run_directory") != args.remote_run_dir:
        raise PermissionError("the authorized direct run directory differs")
    verify_local_source(info, root=ROOT)
    pool, _, pilot_transport = _pool(pilot)
    original, parent_transport = _parent(args.seed, parent)
    transport = _direct_transport(info, args.seed, parent_transport, pilot_transport)
    prompt_sha = _approved_prompt_sha(info)
    dispatcher = Dispatcher(spool, info=info, contract_sha256=args.contract_sha256,
                            transport=transport, mock=False, pool=pool)

    def validate(job):
        return _validate(dispatcher, job, original, args.seed, prompt_sha)

    def before_call(identity, job):
        if dispatcher.check_remote(courier.info())["paused"]:
            raise RemotePaused("the direct coordinator is paused")
        if validate(courier.claim(identity, dispatcher.dispatcher_id)) != job:
            raise PermissionError("the durable remote claim returned another job")
        _check_seed_lease(pool, args.seed)

    with _exclusive(spool / "dispatcher.lock"):
        once(spool / "provider_model_lock.json", read_bridge_json(pilot / "provider_model_lock.json"))
        job = _selection(dispatcher, courier, validate, args.attempt_id, deadline, args.poll)
        _check_journal_scope(spool, job["attempt_id"], args.seed)
        identity = dispatcher.accept(job)
        dispatcher.recover_captured()

        def dispatch():
            return dispatcher.dispatch(identity, before_call=before_call, pause=lambda: time.monotonic() >= deadline)

        status = _until(dispatch, deadline, args.poll, "the direct attempt could not claim a Key before its wait expired")
        view = {"seed": args.seed, "round": PARENTS[args.seed].round_index, "logical_index": 0, "attempt_id": identity,
                "request_id": job["request_id"], "status": status}
        if status != "sealed":
            return view
        result = dispatcher.result(identity)
        if identity in dispatcher.undelivered():
            ack = _until(lambda: courier.put(identity, dispatcher.dispatcher_id, result), deadline, args.poll,
                         "direct result is durably captured on Mac and awaits SSH delivery")
            dispatcher.mark_delivered(identity, result, ack)
        outcome = result["outcome"]
        return {**view, "delivered": True, "provider_dispatched": result["provider_dispatched"],
                "outcome": outcome["status"], "failure": outcome.get("failure", {}).get("code")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=tuple(PARENTS), required=True)
    parser.add_argument("--ssh-host", default="eit-hpc")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--remote-python", default=GPU_PYTHON)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--pilot-spool", type=Path, required=True)
    parser.add_argument("--parent-spool", type=Path, required=True)
    parser.add_argument("--spool-dir", type=Path, required=True)
    parser.add_argument("--attempt-id")
    parser.add_argument("--wait-seconds", type=float, default=8500)
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
