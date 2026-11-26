"""Dispatch the exact seed 2357 V13 request once under its separate recovery contract."""

import argparse
from dataclasses import replace
import math
import os
from pathlib import Path
import time

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_STREAM_IPV4, TransportConfig,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import (
    read_bridge_json, validate_result, verify_job,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAL_WIRE_REPRESENTATION
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import (
    Dispatcher, RemotePaused, _exclusive, _private_directory, verify_local_source,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    ROOT, SSH_API, STUDY, RequestTimeouts, request_timeouts,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import sdk_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import (
    ATTEMPT_ID, GPU_PYTHON, SHA256, SSHCourier, verify_info,
)
from .dispatch_prompt_direct import _check_journal_scope, _check_seed_lease
from .dispatch_prompt_pilot import SharedKeyPool
from .dispatch_prompt_successor import (
    DATA_BINDINGS, PILOT_CONTRACT_SHA256, _approved_prompt_sha, _pool, _selection, _until,
)


SEED, ROUND = 2357, 1
RUN = "a7-gpt62-astra-xhigh-ipv4-v13-2357-extension-01"
PARENT = "a7-gpt62-astra-xhigh-ipv4-v13"
PARENT_CONTRACT = "b17eee7b4d584d5522ca990b47aefdf9d628685557683c83c5bc24f20a998a85"
PARENT_ATTEMPTS = ("710a7f399cb34b6c8a28b8cd5c5e6098", "2b35a1f6928d4b7c8ed8dd6475c64bc9")
MODULE = "dispatch_prepared_ipv4_recovery_2357"
SOURCES = tuple(STUDY / "scripts" / f"{name}.py" for name in (
    MODULE, "dispatch_prompt_direct", "dispatch_prompt_pilot", "dispatch_prompt_successor",
))
TIMEOUTS = RequestTimeouts(read_idle_seconds=1800, total_seconds=3600)
BOUND_HASHES = ("checkpoint_sha256", "base_messages_sha256", "packet_sha256", "wire_packet_sha256")
BUDGET = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 1}


def _scope(info, run):
    record = info["contract"]
    profile = record["profile"]
    predecessor = record["predecessor"]
    run_path = Path(run)
    expected = {"provider_capability": "gated_ipv4", "arm": "A7", "memory": "M2", "device": "cuda",
                "initial_labels": 333, "target_labels": 397, "batch_size": 32, "api_parallelism": 1,
                "store": False, "confirmation_seeds": [SEED], "qualification_seed": SEED,
                "requires_initial_qualification": False, "wire_representation": FORMAL_WIRE_REPRESENTATION}
    boundaries = profile.get("resume_boundaries", {})
    if (run_path.name != RUN or record["run_directory"] != run or record["api_execution"] != SSH_API
            or record["gpu_slots"] != [0] or record["budget"] != BUDGET
            or predecessor.get("dispatcher_module") != MODULE
            or predecessor.get("parent_run_directory") != str(run_path.with_name(PARENT))
            or predecessor.get("parent_contract_sha256") != PARENT_CONTRACT
            or (predecessor.get("seed"), predecessor.get("round"), predecessor.get("logical_index")) != (SEED, ROUND, 0)
            or "fixed_reference_repair" in profile
            or any(type(profile.get(key)) is not type(value) or profile[key] != value for key, value in expected.items())
            or set(boundaries) != {str(SEED)} or boundaries[str(SEED)].get("round") != ROUND
            or request_timeouts(profile) != TIMEOUTS):
        raise PermissionError("the authorized run differs from the single V13 seed 2357 recovery")
    verify_local_source(info, root=ROOT)
    for path in SOURCES:
        if record["source_sha256"].get(path.as_posix()) != file_digest(ROOT / path):
            raise PermissionError(f"the approved recovery source differs: {path.name}")
    return predecessor


def _parent_request(spool, scope):
    _private_directory(spool)
    binding = read_bridge_json(spool / "run_binding.json")
    auth = scope.get("parent_authorization_sha256")
    if (not isinstance(auth, str) or not SHA256.fullmatch(auth)
            or binding != {"contract_sha256": PARENT_CONTRACT, "authorization_sha256": auth, "mock": False}
            or binding["mock"] is not False):
        raise PermissionError("the parent Mac spool differs from the authorized V13 archive")
    attempts = scope.get("prior_attempts")
    if type(attempts) is not list or len(attempts) != len(PARENT_ATTEMPTS):
        raise PermissionError("the recovery must bind the two original V13 attempts")
    original = None
    for index, (item, identity) in enumerate(zip(attempts, PARENT_ATTEMPTS, strict=True)):
        filename = f"{identity}.json"
        job = read_bridge_json(spool / "jobs" / filename)
        request, transport, started = verify_job(job, contract_sha256=PARENT_CONTRACT, authorization_sha256=auth)
        result = validate_result(read_bridge_json(spool / "results" / filename), job)
        ack = read_bridge_json(spool / "delivered" / filename)
        saved = spool / "attempts" / identity
        terminal_sha = item.get("terminal_sha256") if isinstance(item, dict) else None
        expected = {"attempt": index, "attempt_id": identity, "job_sha256": job["job_sha256"],
                    "result_sha256": result["result_sha256"], "terminal_sha256": terminal_sha}
        state = request["binding"]
        failure = result["outcome"].get("failure", {})
        if (item != expected or not isinstance(terminal_sha, str) or not SHA256.fullmatch(terminal_sha)
                or (state["seed"], state["round"], request["logical_index"], started["attempt"]) != (SEED, ROUND, 0, index)
                or started["attempt_id"] != identity or request["request_id"] != scope.get("prior_request_id")
                or request["payload_sha256"] != scope.get("payload_sha256")
                or any(state[key] != scope.get(key) for key in BOUND_HASHES)
                or result["provider_dispatched"] is not True or result["raw_response"] is not None
                or failure.get("code") != "timeout" or failure.get("ambiguous") is not True
                or ack != {"attempt_id": identity, "result_sha256": result["result_sha256"]}
                or (saved / "raw_provider_response.json").exists() or (saved / "provider_capture_receipt.json").exists()
                or original is not None and request != original):
            raise PermissionError("the recovery requires both original sealed timeouts and no captured completion")
        original = request
    model = read_bridge_json(spool / "provider_model_lock.json")
    if model != scope.get("model_identity"):
        raise PermissionError("the recovery model identity differs from the original V13 model lock")
    return original, transport, model


def _validate(dispatcher, job, original, prompt_sha):
    request, _, started = dispatcher.validate(job)
    source, current = original["binding"], request["binding"]
    if ((current["seed"], current["round"], request["logical_index"], started["attempt"]) != (SEED, ROUND, 0, 0)
            or current["prompt_sha256"] != prompt_sha or request["request_id"] == original["request_id"]
            or started["attempt_id"] in PARENT_ATTEMPTS or request["payload"] != original["payload"]
            or request["payload_sha256"] != original["payload_sha256"]
            or any(current[key] != source[key] for key in (*DATA_BINDINGS, "base_messages_sha256"))):
        raise PermissionError("the one-shot job differs from the exact original 2357 payload or checkpoint")
    return job


def run(args):
    deadline = time.monotonic() + args.wait_seconds
    spools = Path.home() / ".config/purellm/spool"
    if args.contract_sha256 in (PILOT_CONTRACT_SHA256, PARENT_CONTRACT):
        raise PermissionError("the recovery must have its own contract and spool")
    spool = spools / args.contract_sha256
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = _until(courier.info, deadline, args.poll, "the one-shot recovery is unreachable over SSH")
    verify_info(info, expected_contract=args.contract_sha256, expected_auth=args.authorization_sha256)
    scope = _scope(info, args.remote_run_dir)
    original, base_transport, model = _parent_request(spools / PARENT_CONTRACT, scope)
    pilot, _, pilot_transport = _pool(spools / PILOT_CONTRACT_SHA256)
    transport = TransportConfig(**info["contract"]["profile"]["transport"])
    expected_base = replace(pilot_transport, timeout_seconds=TIMEOUTS.total_seconds, wire_mode=RESPONSES_STREAM_IPV4)
    if (base_transport != expected_base or transport != replace(base_transport, logical_replies=1, transport_attempts=1)
            or (transport.provider, transport.base_url, transport.requested_model, transport.served_model,
                transport.reasoning_effort, transport.max_output_tokens, transport.sdk_version)
            != ("token4research", "https://token4research.cn/v1", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh", 32768, "2.28.0")
            or not sdk_status(transport)["matches"]):
        raise PermissionError("the approved recovery must retain the exact V13 wire transport with one attempt")
    prompt_sha = _approved_prompt_sha(info)
    pool = SharedKeyPool(pilot.directory, pilot.config)
    dispatcher = Dispatcher(spool, info=info, contract_sha256=args.contract_sha256,
                            transport=transport, mock=False, pool=pool)

    def validate(job):
        return _validate(dispatcher, job, original, prompt_sha)

    def before_call(identity, job):
        if dispatcher.check_remote(courier.info())["paused"]:
            raise RemotePaused("the one-shot recovery is paused")
        if validate(courier.claim(identity, dispatcher.dispatcher_id)) != job:
            raise PermissionError("the remote claim does not match the pinned one-shot job")
        _check_seed_lease(pool, SEED)

    with _exclusive(spool / "dispatcher.lock"):
        pool.prepare()
        once(spool / "provider_model_lock.json", model)
        job = _selection(dispatcher, courier, validate, None, deadline, args.poll)
        identity = job["attempt_id"]
        _check_journal_scope(spool, identity, SEED)
        dispatcher.accept(job)
        dispatcher.recover_captured()
        status = _until(lambda: dispatcher.dispatch(identity, before_call=before_call,
                                                     pause=lambda: time.monotonic() >= deadline),
                        deadline, args.poll, "the one-shot recovery could not acquire a shared Key")
        view = {"seed": SEED, "round": ROUND, "request_id": job["request_id"], "attempt_id": identity, "status": status}
        if status != "sealed":
            return view
        result = dispatcher.result(identity)
        if identity in dispatcher.undelivered():
            ack = _until(lambda: courier.put(identity, dispatcher.dispatcher_id, result), deadline, args.poll,
                         "the one-shot result is captured on Mac and awaits SSH delivery")
            dispatcher.mark_delivered(identity, result, ack)
        outcome = result["outcome"]
        return {**view, "delivered": True, "provider_dispatched": result["provider_dispatched"],
                "outcome": outcome["status"], "failure": outcome.get("failure", {}).get("code")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-paid", action="store_true", required=True)
    parser.add_argument("--ssh-host", default="eit-hpc")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--remote-python", default=GPU_PYTHON)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--wait-seconds", type=float, default=8500)
    parser.add_argument("--poll", type=float, default=2)
    args = parser.parse_args()
    if (not all(SHA256.fullmatch(value) for value in (args.contract_sha256, args.authorization_sha256))
            or not math.isfinite(args.wait_seconds) or not 0 < args.poll <= min(60, args.wait_seconds)):
        parser.error("provide exact authorization hashes and finite positive waits")
    os.umask(0o077)
    print(dumps(run(args)))


if __name__ == "__main__":
    main()
