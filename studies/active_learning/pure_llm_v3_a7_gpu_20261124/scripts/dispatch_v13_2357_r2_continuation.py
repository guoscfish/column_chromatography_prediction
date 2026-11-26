"""Continue seed 2357 from the verified L429 state of its R2 recovery."""

import argparse
from dataclasses import replace
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_STREAM_IPV4, TransportConfig,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import read_bridge_json, validate_result
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAL_WIRE_REPRESENTATION
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import run_dispatcher, verify_local_source
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, SSH_API, STUDY, request_timeouts
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import sdk_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import ATTEMPT_ID, GPU_PYTHON, SSHCourier, verify_info
from .dispatch_prompt_pilot import SharedKeyPool
from .dispatch_prompt_successor import PILOT_CONTRACT_SHA256, SELECTION, _approved_prompt_sha, _pool
from .dispatch_v13_2357_r2_sse_extension import (
    PARENT_ATTEMPTS, PARENT_AUTHORIZATION, PARENT_CONTRACT, PARENT_PAYLOAD_SHA,
    PARENT_RUN, ROUND, RUN as EXTENSION_RUN, SEED, SSE_INACTIVITY_SECONDS, TIMEOUTS,
)
from .dispatch_v13_lane_continuation import ContinuationDispatcher, _read_bound_job, _sha


MODULE = "dispatch_v13_2357_r2_continuation"
RUN = "a7-gpt62-astra-xhigh-ipv4-v13-2357-r2-continuation-01"
NEXT_ROUND = ROUND + 1
BUDGET = {"first_logical_requests": 33, "maximum_logical_requests": 99, "maximum_physical_attempts": 198}
SOURCES = tuple(STUDY / "scripts" / f"{name}.py" for name in (
    MODULE, "dispatch_v13_lane_continuation", "dispatch_v13_lane_extension",
    "dispatch_v13_2357_r2_sse_extension", "dispatch_prompt_pilot", "dispatch_prompt_successor",
))


def _scope(info, run, extension_contract, extension_auth):
    record = info["contract"]
    profile, prior = record["profile"], record["predecessor"]
    location = Path(run)
    expected_profile = {
        "provider_capability": "gated_ipv4", "arm": "A7", "memory": "M2", "device": "cuda",
        "initial_labels": 333, "target_labels": 1485, "batch_size": 32, "api_parallelism": 1,
        "store": False, "confirmation_seeds": [SEED], "qualification_seed": SEED,
        "requires_initial_qualification": False, "wire_representation": FORMAL_WIRE_REPRESENTATION,
        "sse_inactivity_seconds": SSE_INACTIVITY_SECONDS,
    }
    expected_prior = {
        "dispatcher_module": MODULE, "parent_run_directory": str(location.with_name(PARENT_RUN)),
        "parent_contract_sha256": PARENT_CONTRACT, "parent_authorization_sha256": PARENT_AUTHORIZATION,
        "extension_run_directory": str(location.with_name(EXTENSION_RUN)), "extension_contract_sha256": extension_contract,
        "extension_authorization_sha256": extension_auth, "seed": SEED, "round": NEXT_ROUND,
        "execution_budget": BUDGET, "bridge_wait_seconds": TIMEOUTS.bridge_wait_seconds,
    }
    boundaries = profile.get("resume_boundaries", {})
    boundary = boundaries.get(str(SEED), {})
    hash_fields = ("checkpoint_sha256", "parent_seed_pause_sha256", "extension_completion_sha256",
                   "extension_transition_sha256", "extension_fit_audit_sha256", "extension_job_sha256",
                   "extension_result_sha256")
    if (location.name != RUN or record["run_directory"] != run or record["api_execution"] != SSH_API
            or record["gpu_slots"] != [0] or record["budget"] != BUDGET
            or request_timeouts(profile) != TIMEOUTS or "fixed_reference_repair" in profile
            or any(type(profile.get(k)) is not type(v) or profile[k] != v for k, v in expected_profile.items())
            or any(type(prior.get(k)) is not type(v) or prior[k] != v for k, v in expected_prior.items())
            or not all(_sha(prior.get(key)) for key in hash_fields)
            or not isinstance(attempt := prior.get("extension_attempt_id"), str) or not ATTEMPT_ID.fullmatch(attempt)
            or set(boundaries) != {str(SEED)} or set(boundary) != {"round", "state_sha256", "memory_sha256"}
            or boundary["round"] != NEXT_ROUND
            or not all(_sha(boundary.get(key)) for key in ("state_sha256", "memory_sha256"))):
        raise PermissionError("2357 continuation is outside the verified L429 boundary or 33-round budget")
    marker = {
        "contract_sha256": extension_contract, "seed": SEED, "round": NEXT_ROUND, "active_labels": 429,
        "checkpoint_sha256": prior["checkpoint_sha256"], "state_sha256": boundary["state_sha256"],
        "memory_sha256": boundary["memory_sha256"], "transition_sha256": prior["extension_transition_sha256"],
        "original_future_budget": BUDGET,
    }
    if digest(marker) != prior["extension_completion_sha256"]:
        raise PermissionError("2357 continuation does not bind the sealed L429 completion")
    verify_local_source(info, root=ROOT)
    for path in SOURCES:
        if record["source_sha256"].get(path.as_posix()) != file_digest(ROOT / path):
            raise PermissionError(f"the approved continuation source differs: {path.name}")
    return prior


def _archived_result(spools, prior):
    identity = prior["extension_attempt_id"]
    contract, approval = prior["extension_contract_sha256"], prior["extension_authorization_sha256"]
    spool, parent = spools / contract, spools / PARENT_CONTRACT
    job, (request, transport, started) = _read_bound_job(spool, identity, contract, approval)
    _, (original, original_transport, old_started) = _read_bound_job(
        parent, PARENT_ATTEMPTS[0], PARENT_CONTRACT, PARENT_AUTHORIZATION,
    )
    filename = f"{identity}.json"
    result = validate_result(read_bridge_json(spool / "results" / filename), job)
    ack = read_bridge_json(spool / "delivered" / filename)
    selected = read_bridge_json(spool / SELECTION)
    attempt = spool / "attempts" / identity
    receipt = read_bridge_json(attempt / "provider_capture_receipt.json")
    raw = read_bridge_json(attempt / "raw_provider_response.json")
    model = read_bridge_json(spool / "provider_model_lock.json")
    binding = request["binding"]
    science = {key: value for key, value in binding.items() if key != "a7_formal"}
    if ((binding["seed"], binding["round"], request["logical_index"], started["attempt"]) != (SEED, ROUND, 0, 0)
            or started["attempt_id"] != identity or identity in PARENT_ATTEMPTS
            or (old_started["attempt_id"], old_started["attempt"]) != (PARENT_ATTEMPTS[0], 0)
            or request["request_id"] == original["request_id"] or request["payload"] != original["payload"]
            or request["payload_sha256"] != original["payload_sha256"] or request["payload_sha256"] != PARENT_PAYLOAD_SHA
            or science != {key: value for key, value in original["binding"].items() if key != "a7_formal"}
            or job["job_sha256"] != prior["extension_job_sha256"]
            or selected != {"attempt_id": identity, "job_sha256": job["job_sha256"]}
            or result["result_sha256"] != prior["extension_result_sha256"] or result["provider_dispatched"] is not True
            or result["outcome"]["status"] != "completed" or raw != result["raw_response"] or raw.get("status") != "completed"
            or receipt["attempt_id"] != identity or receipt["job_sha256"] != job["job_sha256"]
            or receipt["metadata"]["raw_response_sha256"] != result["raw_response_sha256"]
            or ack != {"attempt_id": identity, "result_sha256": result["result_sha256"]}
            or model != result["model_identity"] or model != prior.get("model_identity")
            or model != read_bridge_json(parent / "provider_model_lock.json")):
        raise PermissionError("2357 continuation needs the exact original payload and a sealed, delivered provider result")
    return transport, original_transport, model


def run(args):
    extension = args.extension_contract_sha256
    distinct = {PILOT_CONTRACT_SHA256, PARENT_CONTRACT, extension, args.contract_sha256}
    if len(distinct) != 4:
        raise PermissionError("the continuation must use a separate authorized spool")
    spools = Path.home() / ".config/purellm/spool"
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.info(), expected_contract=args.contract_sha256, expected_auth=args.authorization_sha256)
    prior = _scope(info, args.remote_run_dir, extension, args.extension_authorization_sha256)
    extension_transport, original, model = _archived_result(spools, prior)
    pilot, _, pilot_transport = _pool(spools / PILOT_CONTRACT_SHA256)
    registered = replace(pilot_transport, timeout_seconds=TIMEOUTS.total_seconds, wire_mode=RESPONSES_STREAM_IPV4)
    transport = TransportConfig(**info["contract"]["profile"]["transport"])
    if (transport != registered or transport != original
            or extension_transport != replace(registered, logical_replies=1, transport_attempts=1)
            or (transport.provider, transport.base_url, transport.requested_model, transport.served_model,
                transport.reasoning_effort, transport.max_output_tokens, transport.sdk_version,
                transport.logical_replies, transport.transport_attempts)
            != ("token4research", "https://token4research.cn/v1", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh", 32768, "2.28.0", 3, 2)
            or not sdk_status(transport)["matches"]):
        raise PermissionError("the continuation must retain the exact V13 transport and per-round budget")
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
    parser.add_argument("--ssh-host", default="eit-hpc")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--remote-python", default=GPU_PYTHON)
    arguments = ("contract-sha256", "authorization-sha256", "extension-contract-sha256", "extension-authorization-sha256")
    for name in arguments:
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    if not all(_sha(getattr(args, name.replace("-", "_"))) for name in arguments):
        parser.error("provide the exact continuation and extension contract and authorization hashes")
    os.umask(0o077)
    print(dumps(run(args)))


if __name__ == "__main__":
    main()
