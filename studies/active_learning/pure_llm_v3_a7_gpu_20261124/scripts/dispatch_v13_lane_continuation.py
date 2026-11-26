"""Dispatch the remaining V13 rounds for a lane with a verified one-shot CUDA handover."""

import argparse
from dataclasses import replace
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_STREAM_IPV4, TransportConfig,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import (
    read_bridge_json, validate_result, verify_job,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAL_WIRE_REPRESENTATION
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import (
    _private_directory, run_dispatcher, verify_local_source,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    ROOT, SSH_API, STUDY, request_timeouts,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import sdk_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import (
    ATTEMPT_ID, GPU_PYTHON, SHA256, SSHCourier, verify_info,
)
from .dispatch_prompt_pilot import PromptDispatcher, SharedKeyPool
from .dispatch_prompt_successor import PILOT_CONTRACT_SHA256, SELECTION, _approved_prompt_sha, _pool
from .dispatch_v13_lane_extension import (
    BRIDGE_WAIT_SECONDS, LANES, PARENT_AUTHORIZATION, PARENT_CONTRACT, PARENT_RUN, PARENT_WALL_SECONDS, TIMEOUTS,
)


END_ROUND = 36
AUTHORITATIVE_MESSAGE_HASH = "authoritative_base_messages_sha256"
MODULE = "dispatch_v13_lane_continuation"
SOURCES = tuple(STUDY / "scripts" / f"{name}.py" for name in (
    MODULE, "dispatch_v13_lane_extension", "dispatch_prompt_pilot", "dispatch_prompt_successor",
))


def _sha(value):
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _verified_scope(info, run, lane, extension_contract, extension_authorization):
    record = info["contract"]
    profile, predecessor = record["profile"], record["predecessor"]
    first_round = lane.round_index + 1
    remaining = END_ROUND - first_round
    budget = {"first_logical_requests": remaining, "maximum_logical_requests": remaining * 3,
              "maximum_physical_attempts": remaining * 3 * 2}
    run_path = Path(run)
    expected = {"provider_capability": "gated_ipv4", "arm": "A7", "memory": "M2", "device": "cuda",
                "initial_labels": 333, "target_labels": 1485, "batch_size": 32, "api_parallelism": 1,
                "store": False, "confirmation_seeds": [lane.seed], "qualification_seed": lane.seed,
                "requires_initial_qualification": False, "wire_representation": FORMAL_WIRE_REPRESENTATION}
    if (run_path.name != f"{PARENT_RUN}-{lane.seed}-continuation-01" or record["run_directory"] != run
            or record["api_execution"] != SSH_API or record["gpu_slots"] != [0] or record["budget"] != budget
            or predecessor.get("dispatcher_module") != MODULE or predecessor.get("execution_budget") != budget
            or predecessor.get("parent_run_directory") != str(run_path.with_name(PARENT_RUN))
            or predecessor.get("extension_run_directory") != str(run_path.with_name(lane.name))
            or (predecessor.get("parent_contract_sha256"), predecessor.get("parent_authorization_sha256"))
            != (PARENT_CONTRACT, PARENT_AUTHORIZATION)
            or (predecessor.get("extension_contract_sha256"), predecessor.get("extension_authorization_sha256"))
            != (extension_contract, extension_authorization)
            or (predecessor.get("seed"), predecessor.get("round")) != (lane.seed, first_round)
            or type(predecessor.get("bridge_wait_seconds")) is not int
            or predecessor["bridge_wait_seconds"] != BRIDGE_WAIT_SECONDS or "fixed_reference_repair" in profile
            or any(type(profile.get(key)) is not type(value) or profile[key] != value for key, value in expected.items())
            or request_timeouts(profile) != TIMEOUTS):
        raise PermissionError("the continuation differs from the authorized lane, transport, or remaining rounds")
    boundaries = profile.get("resume_boundaries", {})
    boundary = boundaries.get(str(lane.seed), {})
    if (set(boundaries) != {str(lane.seed)} or set(boundary) != {"round", "state_sha256", "memory_sha256"}
            or boundary["round"] != first_round
            or not all(_sha(boundary.get(key)) for key in ("state_sha256", "memory_sha256"))
            or not all(_sha(predecessor.get(key)) for key in ("checkpoint_sha256", "parent_seed_pause_sha256",
                "extension_transition_sha256", "extension_completion_sha256", "extension_fit_audit_sha256",
                "extension_job_sha256", "extension_result_sha256"))
            or not isinstance(attempt := predecessor.get("extension_attempt_id"), str)
            or not ATTEMPT_ID.fullmatch(attempt)):
        raise PermissionError("the continuation must bind a verified CUDA handover and completed response")
    marker = {"contract_sha256": extension_contract, "seed": lane.seed, "round": first_round,
              "active_labels": 333 + first_round * 32, "checkpoint_sha256": predecessor["checkpoint_sha256"],
              "state_sha256": boundary["state_sha256"], "memory_sha256": boundary["memory_sha256"],
              "transition_sha256": predecessor["extension_transition_sha256"], "original_future_budget": budget}
    if digest(marker) != predecessor["extension_completion_sha256"]:
        raise PermissionError("the signed CUDA handover does not match its completion marker")
    verify_local_source(info, root=ROOT)
    for path in SOURCES:
        if record["source_sha256"].get(path.as_posix()) != file_digest(ROOT / path):
            raise PermissionError(f"the approved continuation source differs: {path.name}")
    return predecessor


def _read_bound_job(spool, identity, contract, authorization):
    _private_directory(spool)
    binding = read_bridge_json(spool / "run_binding.json")
    if (binding != {"contract_sha256": contract, "authorization_sha256": authorization, "mock": False}
            or binding["mock"] is not False):
        raise PermissionError("the archived Mac spool differs from its exact original authorization")
    job = read_bridge_json(spool / "jobs" / f"{identity}.json")
    return job, verify_job(job, contract_sha256=contract, authorization_sha256=authorization,
                           wire_representation=FORMAL_WIRE_REPRESENTATION)


def _delivered_extension(spools, lane, scope):
    contract, authorization = scope["extension_contract_sha256"], scope["extension_authorization_sha256"]
    identity = scope["extension_attempt_id"]
    spool, parent = spools / contract, spools / PARENT_CONTRACT
    job, (request, config, started) = _read_bound_job(spool, identity, contract, authorization)
    _, (original, parent_transport, old_started) = _read_bound_job(
        parent, lane.prior_attempts[1], PARENT_CONTRACT, PARENT_AUTHORIZATION,
    )
    filename = f"{identity}.json"
    result = validate_result(read_bridge_json(spool / "results" / filename), job)
    ack = read_bridge_json(spool / "delivered" / filename)
    selected = read_bridge_json(spool / SELECTION)
    attempt = spool / "attempts" / identity
    capture = read_bridge_json(attempt / "provider_capture_receipt.json")
    raw = read_bridge_json(attempt / "raw_provider_response.json")
    model = read_bridge_json(spool / "provider_model_lock.json")
    binding, source = request["binding"], original["binding"]
    reconstruction = scope.get("authoritative_message_reconstruction")
    if reconstruction is not None:
        if ((lane.seed, lane.round_index) != (887, 6) or type(reconstruction) is not dict
                or set(reconstruction) != {"archived_sha256", "rebuilt_sha256"}
                or not all(_sha(value) for value in reconstruction.values())
                or reconstruction["archived_sha256"] == reconstruction["rebuilt_sha256"]
                or source.get(AUTHORITATIVE_MESSAGE_HASH) != reconstruction["archived_sha256"]
                or binding.get(AUTHORITATIVE_MESSAGE_HASH) != reconstruction["rebuilt_sha256"]):
            raise PermissionError("the archived and rebuilt host-only message digests are outside the signed seed 887 scope")
        source = {**source, AUTHORITATIVE_MESSAGE_HASH: reconstruction["rebuilt_sha256"]}
    science = {key: value for key, value in binding.items() if key != "a7_formal"}
    if ((binding["seed"], binding["round"], request["logical_index"], started["attempt"]) != (lane.seed, lane.round_index, 0, 0)
            or started["attempt_id"] != identity or identity in lane.prior_attempts
            or old_started["attempt_id"] != lane.prior_attempts[1] or old_started["attempt"] != 1
            or request["request_id"] == original["request_id"] or request["payload"] != original["payload"]
            or request["payload_sha256"] != original["payload_sha256"]
            or science != {key: value for key, value in source.items() if key != "a7_formal"}
            or job["job_sha256"] != scope["extension_job_sha256"]
            or selected != {"attempt_id": identity, "job_sha256": job["job_sha256"]}
            or result["result_sha256"] != scope["extension_result_sha256"] or result["provider_dispatched"] is not True
            or result["outcome"]["status"] != "completed" or raw != result["raw_response"] or raw.get("status") != "completed"
            or capture["attempt_id"] != identity or capture["job_sha256"] != job["job_sha256"]
            or capture["metadata"]["raw_response_sha256"] != result["raw_response_sha256"]
            or ack != {"attempt_id": identity, "result_sha256": result["result_sha256"]}
            or model != result["model_identity"] or model != scope.get("model_identity")
            or model != read_bridge_json(parent / "provider_model_lock.json")):
        raise PermissionError("the continuation requires the exact original payload and durably delivered extension response")
    return config, parent_transport, model


class ContinuationDispatcher(PromptDispatcher):
    def validate(self, job):
        request, config, started = super().validate(job)
        state = request["binding"]
        predecessor = self.info["contract"]["predecessor"]
        first_round = predecessor["round"]
        if (state["seed"] != predecessor["seed"] or not first_round <= state["round"] < END_ROUND
                or state["round"] == first_round and state["checkpoint_sha256"] != predecessor["checkpoint_sha256"]):
            raise PermissionError("the continuation job is outside its lane, verified checkpoint, and round range")
        return request, config, started


def run(args):
    lane = LANES[args.seed]
    extension = args.extension_contract_sha256
    ancestors = {PILOT_CONTRACT_SHA256, PARENT_CONTRACT}
    if extension in ancestors or args.contract_sha256 in {*ancestors, extension}:
        raise PermissionError("the continuation must use distinct authorized extension and continuation spools")
    spools = Path.home() / ".config/purellm/spool"
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.info(), expected_contract=args.contract_sha256, expected_auth=args.authorization_sha256)
    scope = _verified_scope(info, args.remote_run_dir, lane, extension, args.extension_authorization_sha256)
    extension_transport, parent_transport, model = _delivered_extension(spools, lane, scope)
    pilot, _, pilot_transport = _pool(spools / PILOT_CONTRACT_SHA256)
    transport = TransportConfig(**info["contract"]["profile"]["transport"])
    registered = replace(pilot_transport, timeout_seconds=PARENT_WALL_SECONDS, wire_mode=RESPONSES_STREAM_IPV4)
    if (parent_transport != registered or transport != replace(registered, timeout_seconds=TIMEOUTS.total_seconds)
            or extension_transport != replace(transport, logical_replies=1, transport_attempts=1)
            or (transport.provider, transport.base_url, transport.requested_model, transport.served_model,
                transport.reasoning_effort, transport.max_output_tokens, transport.sdk_version,
                transport.logical_replies, transport.transport_attempts)
            != ("token4research", "https://token4research.cn/v1", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh", 32768, "2.28.0", 3, 2)
            or not sdk_status(transport)["matches"]):
        raise PermissionError("the continuation must retain the exact V13 wire configuration and per-round budgets")
    spool = spools / args.contract_sha256
    pool = SharedKeyPool(pilot.directory, pilot.config)
    dispatcher = ContinuationDispatcher(spool, info=info, contract_sha256=args.contract_sha256,
                                        transport=transport, mock=False, pool=pool)
    dispatcher.prompt_sha = _approved_prompt_sha(info)
    once(spool / "provider_model_lock.json", model)
    return run_dispatcher(dispatcher, courier, watch=True, workers=1)


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
    parser.add_argument("--extension-contract-sha256", required=True)
    parser.add_argument("--extension-authorization-sha256", required=True)
    args = parser.parse_args()
    if not all(_sha(value) for value in (args.contract_sha256, args.authorization_sha256,
                                        args.extension_contract_sha256, args.extension_authorization_sha256)):
        parser.error("provide exact continuation and one-shot extension contract and authorization hashes")
    os.umask(0o077)
    print(dumps(run(args)))


if __name__ == "__main__":
    main()
