"""Dispatch one authorized WebSocket recovery of the exact 6101 R1 input."""

import argparse
from dataclasses import replace
from importlib.metadata import version
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_WEBSOCKET_IPV4, RESPONSES_STREAM_IPV4, TransportConfig,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import (
    read_bridge_json, validate_result, verify_job,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAL_WIRE_REPRESENTATION
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import (
    Dispatcher, _private_directory, run_dispatcher, verify_local_source,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    ROOT, SSH_API, STUDY, RequestTimeouts, request_timeouts,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import sdk_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import GPU_PYTHON, SHA256, SSHCourier, verify_info
from .dispatch_prompt_direct import _check_seed_lease
from .dispatch_prompt_pilot import SharedKeyPool
from .dispatch_prompt_successor import PILOT_CONTRACT_SHA256, _approved_prompt_sha, _pool
from .dispatch_v13_lane_extension import (
    BUDGET, LANES, PARENT_AUTHORIZATION, PARENT_CONTRACT, PARENT_RUN, PARENT_WALL_SECONDS,
    _parent,
)


MODULE = "dispatch_v13_6101_ws_extension"
LANE = LANES[6101]
RUN = f"{PARENT_RUN}-6101-ws-extension-01"
PREVIOUS = "3fbd15709e318e3f7cce6fe57257be4c5a123d08769a83c041da6c25868c7d07"
PREVIOUS_AUTH = "178ae4d043b4ed4e625254583b4b602ce7773d10cd414860ecfa2d8e903459b4"
PREVIOUS_ATTEMPT = "55f8c0bc852c452ba1f46f3c157861b1"
PREVIOUS_REQUEST = "56e6016470e0353ba97df3e8a956454f29f1efc6bf5ed370be95a88f46905f56"
PREVIOUS_JOB = "7ff0face0800083aef78058198ae037523d6aa14a2a5101d8e8966db682bae7c"
PREVIOUS_RESULT = "2437350cc6fbbcb0d351cfeb86ca987849178f9f987bfc54d16e286bd48026fd"
PAYLOAD_SHA = "55c213c38e8c6b5fe3832f05060930bc436ec530d437da3551dc74f6d1324eeb"
TIMEOUTS = RequestTimeouts(read_idle_seconds=1800, total_seconds=3600)
SSE_INACTIVITY_SECONDS = 900
WEBSOCKETS_VERSION = "15.0.1"
SOURCES = tuple(STUDY / "scripts" / f"{name}.py" for name in (
    MODULE, "dispatch_v13_lane_extension", "dispatch_prompt_direct", "dispatch_prompt_pilot", "dispatch_prompt_successor",
))


def _scope(info, run):
    record = info["contract"]
    profile, prior = record["profile"], record["predecessor"]
    location = Path(run)
    expected_profile = {
        "provider_capability": "gated_ipv4", "arm": "A7", "memory": "M2", "device": "cuda",
        "initial_labels": 333, "target_labels": 397, "batch_size": 32, "api_parallelism": 1,
        "store": False, "confirmation_seeds": [6101], "qualification_seed": 6101,
        "sse_inactivity_seconds": SSE_INACTIVITY_SECONDS, "websockets_version": WEBSOCKETS_VERSION,
        "requires_initial_qualification": False, "wire_representation": FORMAL_WIRE_REPRESENTATION,
    }
    expected_prior = {
        "dispatcher_module": MODULE, "parent_run_directory": str(location.with_name(PARENT_RUN)),
        "parent_contract_sha256": PARENT_CONTRACT, "parent_authorization_sha256": PARENT_AUTHORIZATION,
        "previous_run_directory": str(location.with_name(LANE.name)),
        "previous_contract_sha256": PREVIOUS, "previous_authorization_sha256": PREVIOUS_AUTH,
        "previous_request_id": PREVIOUS_REQUEST, "payload_sha256": PAYLOAD_SHA,
        "seed": 6101, "round": 1, "logical_index": 0, "bridge_wait_seconds": TIMEOUTS.bridge_wait_seconds,
    }
    boundary = profile.get("resume_boundaries", {})
    if (location.name != RUN or record["run_directory"] != run or record["api_execution"] != SSH_API
            or record["gpu_slots"] != [0] or record["budget"] != BUDGET or request_timeouts(profile) != TIMEOUTS
            or "fixed_reference_repair" in profile
            or any(type(profile.get(k)) is not type(v) or profile[k] != v for k, v in expected_profile.items())
            or any(type(prior.get(k)) is not type(v) or prior[k] != v for k, v in expected_prior.items())
            or set(boundary) != {"6101"} or boundary["6101"].get("round") != 1):
        raise PermissionError("WebSocket transport is limited to one authorized 6101 R1 request and its L365 boundary")
    verify_local_source(info, root=ROOT)
    for path in SOURCES:
        if record["source_sha256"].get(path.as_posix()) != file_digest(ROOT / path):
            raise PermissionError(f"the approved recovery source differs: {path.name}")
    return prior


def _previous(spool, original, model, prior):
    _private_directory(spool)
    if read_bridge_json(spool / "run_binding.json") != {
            "contract_sha256": PREVIOUS, "authorization_sha256": PREVIOUS_AUTH, "mock": False}:
        raise PermissionError("wrong Mac journal for the preceding failed recovery")
    name = f"{PREVIOUS_ATTEMPT}.json"
    job = read_bridge_json(spool / "jobs" / name)
    request, config, started = verify_job(job, contract_sha256=PREVIOUS, authorization_sha256=PREVIOUS_AUTH,
                                          wire_representation=FORMAL_WIRE_REPRESENTATION)
    evidence = prior.get("previous_attempt", {})
    terminal = evidence.get("terminal_sha256")
    if not isinstance(terminal, str) or not SHA256.fullmatch(terminal):
        raise PermissionError("the preceding GPU terminal must be bound")
    expected = {"attempt": 0, "attempt_id": PREVIOUS_ATTEMPT, "job_sha256": PREVIOUS_JOB,
                "result_sha256": PREVIOUS_RESULT, "terminal_sha256": terminal,
                "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True}
    result = validate_result(read_bridge_json(spool / "results" / name), job)
    failure = result["outcome"].get("failure", {})
    ack = read_bridge_json(spool / "delivered" / name)
    saved = spool / "attempts" / PREVIOUS_ATTEMPT
    progress = read_bridge_json(saved / "provider_progress.json")
    science = {k: v for k, v in request["binding"].items() if k != "a7_formal"}
    if (evidence != expected or job["job_sha256"] != PREVIOUS_JOB or result["result_sha256"] != PREVIOUS_RESULT
            or (started["attempt_id"], started["attempt"], request["request_id"]) != (PREVIOUS_ATTEMPT, 0, PREVIOUS_REQUEST)
            or request["payload"] != original["payload"] or request["payload_sha256"] != original["payload_sha256"]
            or science != {k: v for k, v in original["binding"].items() if k != "a7_formal"}
            or result["provider_dispatched"] is not True or result["raw_response"] is not None
            or failure.get("code") != "response_id_mismatch" or failure.get("ambiguous") is not True
            or (progress.get("attempt_id"), progress.get("job_sha256"), progress.get("http_status"), progress.get("error_type"))
            != (PREVIOUS_ATTEMPT, PREVIOUS_JOB, 200, "response_id_mismatch")
            or ack != {"attempt_id": PREVIOUS_ATTEMPT, "result_sha256": PREVIOUS_RESULT}
            or any((saved / item).exists() for item in ("raw_provider_response.json", "provider_capture_receipt.json"))
            or read_bridge_json(spool / "provider_model_lock.json") != model):
        raise PermissionError("the prior one-shot must be the exact delivered ID-mismatch failure without a complete response")
    return config


class WebSocketRecoveryDispatcher(Dispatcher):
    def validate(self, job):
        request, config, started = super().validate(job)
        original, binding = self.original, request["binding"]
        science = {k: v for k, v in binding.items() if k != "a7_formal"}
        if ((binding["seed"], binding["round"], request["logical_index"], started["attempt"]) != (6101, 1, 0, 0)
                or binding["prompt_sha256"] != self.prompt_sha
                or request["request_id"] in {original["request_id"], PREVIOUS_REQUEST}
                or started["attempt_id"] in {*LANE.prior_attempts, PREVIOUS_ATTEMPT}
                or request["payload"] != original["payload"] or request["payload_sha256"] != PAYLOAD_SHA
                or science != {k: v for k, v in original["binding"].items() if k != "a7_formal"}):
            raise PermissionError("the WebSocket attempt must preserve the exact archived scientific request")
        return request, config, started

    def dispatch(self, attempt_id, *, before_call, pause=None):
        def check(identity, job):
            before_call(identity, job)
            _check_seed_lease(self.pool, 6101)

        return super().dispatch(attempt_id, before_call=check, pause=pause)


def run(args):
    if args.contract_sha256 in {PILOT_CONTRACT_SHA256, PARENT_CONTRACT, PREVIOUS}:
        raise PermissionError("the WebSocket recovery has a distinct scoped spool")
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.info(), expected_contract=args.contract_sha256, expected_auth=args.authorization_sha256)
    prior = _scope(info, args.remote_run_dir)
    spools = Path.home() / ".config/purellm/spool"
    original, old_transport, model = _parent(spools / PARENT_CONTRACT, LANE, prior)
    previous_transport = _previous(spools / PREVIOUS, original, model, prior)
    pilot, _, pilot_transport = _pool(spools / PILOT_CONTRACT_SHA256)
    registered = replace(pilot_transport, timeout_seconds=PARENT_WALL_SECONDS, wire_mode=RESPONSES_STREAM_IPV4)
    transport = TransportConfig(**info["contract"]["profile"]["transport"])
    expected = replace(registered, wire_mode=RESPONSES_WEBSOCKET_IPV4, timeout_seconds=TIMEOUTS.total_seconds,
                       logical_replies=1, transport_attempts=1)
    if (registered != old_transport or transport != expected
            or previous_transport != replace(expected, wire_mode=RESPONSES_STREAM_IPV4, timeout_seconds=1800)
            or (transport.provider, transport.base_url, transport.requested_model, transport.served_model,
                transport.reasoning_effort, transport.max_output_tokens, transport.sdk_version)
            != ("token4research", "https://token4research.cn/v1", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh", 32768, "2.28.0")
            or not sdk_status(transport)["matches"] or version("websockets") != WEBSOCKETS_VERSION):
        raise PermissionError("the WebSocket mode and dependencies differ from the signed one-shot")
    if original["payload"].get("stream") is not True or digest(original["payload"]) != PAYLOAD_SHA:
        raise PermissionError("the archived scientific request differs from the signed WebSocket payload")
    spool = spools / args.contract_sha256
    pool = SharedKeyPool(pilot.directory, pilot.config)
    dispatcher = WebSocketRecoveryDispatcher(spool, info=info, contract_sha256=args.contract_sha256,
                                        transport=transport, mock=False, pool=pool)
    dispatcher.original = original
    dispatcher.prompt_sha = _approved_prompt_sha(info)
    once(spool / "provider_model_lock.json", model)
    return run_dispatcher(dispatcher, courier, watch=True, workers=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-paid", action="store_true", required=True)
    parser.add_argument("--ssh-host", default="eit-hpc")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--remote-python", default=GPU_PYTHON)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    args = parser.parse_args()
    if not all(SHA256.fullmatch(value) for value in (args.contract_sha256, args.authorization_sha256)):
        parser.error("exact contract and authorization hashes required")
    os.umask(0o077)
    print(dumps(run(args)))


if __name__ == "__main__":
    main()
