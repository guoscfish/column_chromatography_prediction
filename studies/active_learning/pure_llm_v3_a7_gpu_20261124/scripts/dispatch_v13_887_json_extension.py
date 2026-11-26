"""Dispatch one authorized whole-response HTTP retry of the exact 887 R6 input."""

import argparse
from dataclasses import replace
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_JSON_IPV4, RESPONSES_STREAM_IPV4, TransportConfig,
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
    BUDGET, BRIDGE_WAIT_SECONDS, LANES, PARENT_AUTHORIZATION, PARENT_CONTRACT, PARENT_RUN, PARENT_WALL_SECONDS,
    _parent,
)


MODULE = "dispatch_v13_887_json_extension"
LANE = LANES[887]
RUN = f"{PARENT_RUN}-887-json-extension-01"
PREVIOUS = "9d5516fcaf672333a1d99bf7d9e0f182720986743391d010dc653dae74704963"
PREVIOUS_AUTH = "68c486cff2a14a721a6a3dc9264683ee9a611e04ce9ea906f4eae757277ced50"
PREVIOUS_ATTEMPT = "4aa0bd5f4a7c40b48cb3705fdd1b1807"
PREVIOUS_REQUEST = "c9f8d4cd13555828c65ec03a988b486110dcdd1250b2ee4c17bee58ac7bde709"
PREVIOUS_JOB = "5aa5a6c21f578b4d14ba07b565c654c31e03e378cab5382cae4ca87582c6261b"
PREVIOUS_RESULT = "02123f992d0d2f9246091e5bb9a7181f37213c3f2b1615e37968ffca16076101"
PAYLOAD_SHA = "415a29f679ad7542bf7bd6756cd29d1ff455a487d58d81b56784c71a91d3c3ad"
TIMEOUTS = RequestTimeouts(read_idle_seconds=1800, total_seconds=1800)
SOURCES = tuple(STUDY / "scripts" / f"{name}.py" for name in (
    MODULE, "dispatch_v13_lane_extension", "dispatch_prompt_direct", "dispatch_prompt_pilot", "dispatch_prompt_successor",
))


def _scope(info, run):
    record = info["contract"]
    profile, prior = record["profile"], record["predecessor"]
    location = Path(run)
    expected_profile = {
        "provider_capability": "gated_ipv4", "arm": "A7", "memory": "M2", "device": "cuda",
        "initial_labels": 333, "target_labels": 557, "batch_size": 32, "api_parallelism": 1,
        "store": False, "confirmation_seeds": [887], "qualification_seed": 887,
        "requires_initial_qualification": False, "wire_representation": FORMAL_WIRE_REPRESENTATION,
    }
    expected_prior = {
        "dispatcher_module": MODULE, "parent_run_directory": str(location.with_name(PARENT_RUN)),
        "parent_contract_sha256": PARENT_CONTRACT, "parent_authorization_sha256": PARENT_AUTHORIZATION,
        "previous_run_directory": str(location.with_name(LANE.name)),
        "previous_contract_sha256": PREVIOUS, "previous_authorization_sha256": PREVIOUS_AUTH,
        "previous_request_id": PREVIOUS_REQUEST, "json_payload_sha256": PAYLOAD_SHA,
        "seed": 887, "round": 6, "logical_index": 0, "bridge_wait_seconds": BRIDGE_WAIT_SECONDS,
    }
    boundary = profile.get("resume_boundaries", {})
    if (location.name != RUN or record["run_directory"] != run or record["api_execution"] != SSH_API
            or record["gpu_slots"] != [0] or record["budget"] != BUDGET or request_timeouts(profile) != TIMEOUTS
            or "fixed_reference_repair" in profile or profile.get("sse_inactivity_seconds") is not None
            or any(type(profile.get(k)) is not type(v) or profile[k] != v for k, v in expected_profile.items())
            or any(type(prior.get(k)) is not type(v) or prior[k] != v for k, v in expected_prior.items())
            or set(boundary) != {"887"} or boundary["887"].get("round") != 6):
        raise PermissionError("whole-response transport is limited to one authorized 887 R6 request and its L525 boundary")
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


class JsonRecoveryDispatcher(Dispatcher):
    def validate(self, job):
        request, config, started = super().validate(job)
        original, binding = self.original, request["binding"]
        science = {k: v for k, v in binding.items() if k != "a7_formal"}
        if ((binding["seed"], binding["round"], request["logical_index"], started["attempt"]) != (887, 6, 0, 0)
                or binding["prompt_sha256"] != self.prompt_sha
                or request["request_id"] in {original["request_id"], PREVIOUS_REQUEST}
                or started["attempt_id"] in {*LANE.prior_attempts, PREVIOUS_ATTEMPT}
                or request["payload"] != original["payload"] or request["payload_sha256"] != PAYLOAD_SHA
                or science != {k: v for k, v in original["binding"].items() if k != "a7_formal"}):
            raise PermissionError("the whole-response attempt must preserve the exact archived scientific request")
        return request, config, started

    def dispatch(self, attempt_id, *, before_call, pause=None):
        def check(identity, job):
            before_call(identity, job)
            _check_seed_lease(self.pool, 887)

        return super().dispatch(attempt_id, before_call=check, pause=pause)


def run(args):
    if args.contract_sha256 in {PILOT_CONTRACT_SHA256, PARENT_CONTRACT, PREVIOUS}:
        raise PermissionError("the whole-response recovery has a distinct scoped spool")
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.info(), expected_contract=args.contract_sha256, expected_auth=args.authorization_sha256)
    prior = _scope(info, args.remote_run_dir)
    spools = Path.home() / ".config/purellm/spool"
    original, old_transport, model = _parent(spools / PARENT_CONTRACT, LANE, prior)
    previous_transport = _previous(spools / PREVIOUS, original, model, prior)
    pilot, _, pilot_transport = _pool(spools / PILOT_CONTRACT_SHA256)
    registered = replace(pilot_transport, timeout_seconds=PARENT_WALL_SECONDS, wire_mode=RESPONSES_STREAM_IPV4)
    transport = TransportConfig(**info["contract"]["profile"]["transport"])
    expected = replace(registered, wire_mode=RESPONSES_JSON_IPV4, timeout_seconds=TIMEOUTS.total_seconds,
                       logical_replies=1, transport_attempts=1)
    if (registered != old_transport or transport != expected
            or previous_transport != replace(expected, wire_mode=RESPONSES_STREAM_IPV4)
            or (transport.provider, transport.base_url, transport.requested_model, transport.served_model,
                transport.reasoning_effort, transport.max_output_tokens, transport.sdk_version)
            != ("token4research", "https://token4research.cn/v1", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh", 32768, "2.28.0")
            or not sdk_status(transport)["matches"]):
        raise PermissionError("only the signed whole-response wire mode may differ from the prior one-shot")
    payload = {**original["payload"], "stream": False}
    if original["payload"].get("stream") is not True or digest(payload) != PAYLOAD_SHA:
        raise PermissionError("the whole-response payload must differ from the archived payload only in stream=false")
    spool = spools / args.contract_sha256
    pool = SharedKeyPool(pilot.directory, pilot.config)
    dispatcher = JsonRecoveryDispatcher(spool, info=info, contract_sha256=args.contract_sha256,
                                        transport=transport, mock=False, pool=pool)
    dispatcher.original = {**original, "payload": payload, "payload_sha256": PAYLOAD_SHA}
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
