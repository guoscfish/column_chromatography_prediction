"""Dispatch one separately authorized, shorter recovery attempt for a specific V13 lane."""

import argparse
from dataclasses import dataclass, replace
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
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import GPU_PYTHON, SHA256, SSHCourier, verify_info
from .dispatch_prompt_direct import JOURNAL_DIRS, _check_seed_lease
from .dispatch_prompt_pilot import SharedKeyPool
from .dispatch_prompt_successor import PILOT_CONTRACT_SHA256, _approved_prompt_sha, _pool, _selection, _until


@dataclass(frozen=True)
class Lane:
    seed: int
    round_index: int
    prior_attempts: tuple[str, str]
    posted_error: str
    posted_transport_error: str

    @property
    def name(self):
        return f"{PARENT_RUN}-{self.seed}-extension-01"


PARENT_RUN = "a7-gpt62-astra-xhigh-ipv4-v13"
PARENT_CONTRACT = "b17eee7b4d584d5522ca990b47aefdf9d628685557683c83c5bc24f20a998a85"
PARENT_AUTHORIZATION = "d43ca2ba5a3ac93173466bdb638735805d0092b321dd42e7d8aac862af922410"
LANES = {
    6101: Lane(6101, 1, ("f7c8c634b4a047e7a00cc1e70d630583", "02baf5606f2f4c5395ca9167c9c560b5"),
               "timeout", "hard_wall_deadline"),
    887: Lane(887, 6, ("8092f217a6234c6aa72cd86df60fb862", "0716cc9aca894e31b95c5f81b8a1d427"),
              "response_id_mismatch", "response_id_mismatch"),
}
MODULE = "dispatch_v13_lane_extension"
SOURCES = tuple(STUDY / "scripts" / f"{name}.py" for name in (
    MODULE, "dispatch_prompt_direct", "dispatch_prompt_pilot", "dispatch_prompt_successor",
))
PARENT_WALL_SECONDS = 3600
TIMEOUTS = RequestTimeouts(read_idle_seconds=900, total_seconds=1800)
BRIDGE_WAIT_SECONDS = PARENT_WALL_SECONDS + TIMEOUTS.total_seconds + TIMEOUTS.bridge_margin_seconds
BUDGET = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 1}
BOUND_HASHES = ("checkpoint_sha256", "base_messages_sha256", "packet_sha256", "wire_packet_sha256")
AUTHORITATIVE_MESSAGE_HASH = "authoritative_base_messages_sha256"


def _approved_scope(info, run, lane):
    record = info["contract"]
    profile, predecessor = record["profile"], record["predecessor"]
    run_path = Path(run)
    expected = {"provider_capability": "gated_ipv4", "arm": "A7", "memory": "M2", "device": "cuda",
                "initial_labels": 333, "target_labels": 333 + (lane.round_index + 1) * 32,
                "batch_size": 32, "api_parallelism": 1, "store": False, "confirmation_seeds": [lane.seed],
                "qualification_seed": lane.seed, "requires_initial_qualification": False,
                "wire_representation": FORMAL_WIRE_REPRESENTATION}
    boundaries = profile.get("resume_boundaries", {})
    if (run_path.name != lane.name or record["run_directory"] != run or record["api_execution"] != SSH_API
            or record["gpu_slots"] != [0] or record["budget"] != BUDGET
            or predecessor.get("dispatcher_module") != MODULE
            or type(predecessor.get("bridge_wait_seconds")) is not int
            or predecessor["bridge_wait_seconds"] != BRIDGE_WAIT_SECONDS
            or predecessor.get("parent_run_directory") != str(run_path.with_name(PARENT_RUN))
            or (predecessor.get("parent_contract_sha256"), predecessor.get("parent_authorization_sha256"))
            != (PARENT_CONTRACT, PARENT_AUTHORIZATION)
            or (predecessor.get("seed"), predecessor.get("round"), predecessor.get("logical_index"))
            != (lane.seed, lane.round_index, 0)
            or "fixed_reference_repair" in profile
            or any(type(profile.get(key)) is not type(value) or profile[key] != value for key, value in expected.items())
            or set(boundaries) != {str(lane.seed)} or boundaries[str(lane.seed)].get("round") != lane.round_index
            or request_timeouts(profile) != TIMEOUTS):
        raise PermissionError("the authorized run does not match the exact one-attempt lane and shorter timeouts")
    verify_local_source(info, root=ROOT)
    for path in SOURCES:
        if record["source_sha256"].get(path.as_posix()) != file_digest(ROOT / path):
            raise PermissionError(f"the approved recovery source differs: {path.name}")
    return predecessor


def _archived_attempt(spool, lane, scope, index, evidence):
    identity = lane.prior_attempts[index]
    filename = f"{identity}.json"
    job = read_bridge_json(spool / "jobs" / filename)
    request, transport, started = verify_job(job, contract_sha256=PARENT_CONTRACT,
                                              authorization_sha256=PARENT_AUTHORIZATION,
                                              wire_representation=FORMAL_WIRE_REPRESENTATION)
    result = validate_result(read_bridge_json(spool / "results" / filename), job)
    ack = read_bridge_json(spool / "delivered" / filename)
    saved = spool / "attempts" / identity
    progress = read_bridge_json(saved / "provider_progress.json")
    posted = index == 1
    error = lane.posted_error if posted else "connect_error"
    transport_error = lane.posted_transport_error if posted else "direct_ipv4_preflight_failed"
    terminal_sha = evidence.get("terminal_sha256") if isinstance(evidence, dict) else None
    expected = {"attempt": index, "attempt_id": identity, "job_sha256": job["job_sha256"],
                "result_sha256": result["result_sha256"], "terminal_sha256": terminal_sha,
                "error": error, "provider_dispatched": posted, "ambiguous": posted}
    state = request["binding"]
    failure = result["outcome"].get("failure", {})
    if (evidence != expected or not isinstance(terminal_sha, str) or not SHA256.fullmatch(terminal_sha)
            or (state["seed"], state["round"], request["logical_index"], started["attempt"])
            != (lane.seed, lane.round_index, 0, index)
            or started["attempt_id"] != identity or request["request_id"] != scope.get("prior_request_id")
            or request["payload_sha256"] != scope.get("payload_sha256")
            or any(state[key] != scope.get(key) for key in BOUND_HASHES)
            or result["provider_dispatched"] is not posted or result["raw_response"] is not None
            or failure.get("code") != error or failure.get("ambiguous") is not posted
            or (progress.get("attempt_id"), progress.get("job_sha256"), progress.get("error_type"), progress.get("http_status"))
            != (identity, job["job_sha256"], transport_error, 200 if posted else None)
            or ack != {"attempt_id": identity, "result_sha256": result["result_sha256"]}
            or (saved / "raw_provider_response.json").exists() or (saved / "provider_capture_receipt.json").exists()):
        raise PermissionError("the first preflight failure and second unresolved posted attempt must match their original receipts")
    return request, transport


def _parent(spool, lane, scope):
    _private_directory(spool)
    binding = read_bridge_json(spool / "run_binding.json")
    if (binding != {"contract_sha256": PARENT_CONTRACT, "authorization_sha256": PARENT_AUTHORIZATION, "mock": False}
            or binding["mock"] is not False):
        raise PermissionError("the original V13 Mac journal does not match the approved archive")
    attempts = scope.get("prior_attempts")
    if type(attempts) is not list or len(attempts) != len(lane.prior_attempts):
        raise PermissionError("the recovery requires the original two physical reservations")
    first, second = (_archived_attempt(spool, lane, scope, index, entry) for index, entry in enumerate(attempts))
    if first != second:
        raise PermissionError("both original attempts must bind the same scientific request")
    model = read_bridge_json(spool / "provider_model_lock.json")
    if model != scope.get("model_identity"):
        raise PermissionError("the approved model identity differs from the original V13 model lock")
    request, transport = first
    return request, transport, model


def _validate(dispatcher, job, lane, original, prompt_sha, reconstruction=None):
    request, _, started = dispatcher.validate(job)
    source, state = original["binding"], request["binding"]
    if reconstruction is not None:
        if ((lane.seed, lane.round_index) != (887, 6) or type(reconstruction) is not dict
                or set(reconstruction) != {"archived_sha256", "rebuilt_sha256"}
                or any(not isinstance(value, str) or not SHA256.fullmatch(value) for value in reconstruction.values())
                or reconstruction["archived_sha256"] == reconstruction["rebuilt_sha256"]
                or source.get(AUTHORITATIVE_MESSAGE_HASH) != reconstruction["archived_sha256"]
                or state.get(AUTHORITATIVE_MESSAGE_HASH) != reconstruction["rebuilt_sha256"]):
            raise PermissionError("the archived and rebuilt host-only message digests are outside the signed seed 887 scope")
        source = {**source, AUTHORITATIVE_MESSAGE_HASH: reconstruction["rebuilt_sha256"]}
    science = {key: value for key, value in state.items() if key != "a7_formal"}
    if ((state["seed"], state["round"], request["logical_index"], started["attempt"]) != (lane.seed, lane.round_index, 0, 0)
            or state["prompt_sha256"] != prompt_sha or request["request_id"] == original["request_id"]
            or started["attempt_id"] in lane.prior_attempts or request["payload"] != original["payload"]
            or request["payload_sha256"] != original["payload_sha256"]
            or science != {key: value for key, value in source.items() if key != "a7_formal"}):
        raise PermissionError("the new physical attempt must use the exact archived scientific payload")
    return job


def run(args):
    lane = LANES[args.seed]
    deadline = time.monotonic() + args.wait_seconds
    if args.contract_sha256 in (PILOT_CONTRACT_SHA256, PARENT_CONTRACT):
        raise PermissionError("the separate recovery requires its own authorized contract and spool")
    spools = Path.home() / ".config/purellm/spool"
    spool = spools / args.contract_sha256
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = _until(courier.info, deadline, args.poll, "the lane recovery is unreachable over SSH")
    verify_info(info, expected_contract=args.contract_sha256, expected_auth=args.authorization_sha256)
    scope = _approved_scope(info, args.remote_run_dir, lane)
    original, parent_transport, model = _parent(spools / PARENT_CONTRACT, lane, scope)
    pilot, _, pilot_transport = _pool(spools / PILOT_CONTRACT_SHA256)
    registered = replace(pilot_transport, timeout_seconds=PARENT_WALL_SECONDS, wire_mode=RESPONSES_STREAM_IPV4)
    transport = TransportConfig(**info["contract"]["profile"]["transport"])
    expected = replace(parent_transport, timeout_seconds=TIMEOUTS.total_seconds, logical_replies=1, transport_attempts=1)
    if (parent_transport != registered or transport != expected
            or (transport.provider, transport.base_url, transport.requested_model, transport.served_model,
                transport.reasoning_effort, transport.max_output_tokens, transport.sdk_version)
            != ("token4research", "https://token4research.cn/v1", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh", 32768, "2.28.0")
            or not sdk_status(transport)["matches"]):
        raise PermissionError("the lane must retain the original V13 wire protocol and approved one-attempt timeout")
    pool = SharedKeyPool(pilot.directory, pilot.config)
    prompt_sha = _approved_prompt_sha(info)
    dispatcher = Dispatcher(spool, info=info, contract_sha256=args.contract_sha256, transport=transport, mock=False, pool=pool)

    def validate(job):
        return _validate(dispatcher, job, lane, original, prompt_sha, scope.get("authoritative_message_reconstruction"))

    def before_call(identity, job):
        if dispatcher.check_remote(courier.info())["paused"]:
            raise RemotePaused("the lane recovery is paused")
        if validate(courier.claim(identity, dispatcher.dispatcher_id)) != job:
            raise PermissionError("the remote claim does not match the pinned recovery job")
        _check_seed_lease(pool, lane.seed)

    with _exclusive(spool / "dispatcher.lock"):
        pool.prepare()
        once(spool / "provider_model_lock.json", model)
        job = _selection(dispatcher, courier, validate, None, deadline, args.poll)
        identity = job["attempt_id"]
        filename = f"{identity}.json"
        entries = (filename, f"{lane.seed}-{lane.round_index}-0-0.json", identity, filename, filename)
        if any(path.name != expected for directory, expected in zip(JOURNAL_DIRS, entries, strict=True)
               for path in (spool / directory).iterdir()):
            raise PermissionError("the one-shot recovery journal contains a foreign physical attempt")
        dispatcher.accept(job)
        dispatcher.recover_captured()
        status = _until(lambda: dispatcher.dispatch(identity, before_call=before_call,
                                                     pause=lambda: time.monotonic() >= deadline),
                        deadline, args.poll, "the lane recovery could not acquire a shared Key")
        view = {"seed": lane.seed, "round": lane.round_index, "request_id": job["request_id"],
                "attempt_id": identity, "status": status}
        if status != "sealed":
            return view
        result = dispatcher.result(identity)
        if identity in dispatcher.undelivered():
            ack = _until(lambda: courier.put(identity, dispatcher.dispatcher_id, result), deadline, args.poll,
                         "the recovery result is captured on Mac and awaits SSH delivery")
            dispatcher.mark_delivered(identity, result, ack)
        outcome = result["outcome"]
        return {**view, "delivered": True, "provider_dispatched": result["provider_dispatched"],
                "outcome": outcome["status"], "failure": outcome.get("failure", {}).get("code")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-paid", action="store_true", required=True)
    parser.add_argument("--seed", type=int, choices=tuple(LANES), required=True)
    parser.add_argument("--ssh-host", default="eit-hpc")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--remote-python", default=GPU_PYTHON)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--wait-seconds", type=float, default=BRIDGE_WAIT_SECONDS)
    parser.add_argument("--poll", type=float, default=2)
    args = parser.parse_args()
    if (not all(SHA256.fullmatch(value) for value in (args.contract_sha256, args.authorization_sha256))
            or not math.isfinite(args.wait_seconds) or not 0 < args.poll <= min(60, args.wait_seconds)):
        parser.error("provide exact authorization hashes and finite positive waits")
    os.umask(0o077)
    print(dumps(run(args)))


if __name__ == "__main__":
    main()
