"""Dispatch seed 2357's remaining rounds from its verified L397 recovery checkpoint."""

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
    ROOT, SSH_API, STUDY, RequestTimeouts, request_timeouts,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import sdk_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import (
    ATTEMPT_ID, GPU_PYTHON, SHA256, SSHCourier, verify_info,
)
from .dispatch_prompt_pilot import PromptDispatcher, SharedKeyPool
from .dispatch_prompt_successor import PILOT_CONTRACT_SHA256, SELECTION, _approved_prompt_sha, _pool


SEED, START_ROUND, END_ROUND = 2357, 2, 36
RUN = "a7-gpt62-astra-xhigh-ipv4-v13-2357-continuation-01"
PARENT_RUN = "a7-gpt62-astra-xhigh-ipv4-v13"
EXTENSION_RUN = "a7-gpt62-astra-xhigh-ipv4-v13-2357-extension-01"
PARENT_CONTRACT = "b17eee7b4d584d5522ca990b47aefdf9d628685557683c83c5bc24f20a998a85"
PARENT_AUTHORIZATION = "d43ca2ba5a3ac93173466bdb638735805d0092b321dd42e7d8aac862af922410"
EXTENSION_CONTRACT = "c07d946937930e575096e167fcdd80df3e261e2bb6f603350f4d6a447085d300"
EXTENSION_AUTHORIZATION = "5c3e30de81150b69ce4fe125bb760bcfcd776b86ed0099b1137c283612ac4785"
MODULE = "dispatch_prepared_ipv4_continue_2357"
SOURCES = tuple(STUDY / "scripts" / f"{name}.py" for name in (MODULE, "dispatch_prompt_pilot", "dispatch_prompt_successor"))
TIMEOUTS = RequestTimeouts(read_idle_seconds=1800, total_seconds=3600)
BUDGET = {"first_logical_requests": 34, "maximum_logical_requests": 102, "maximum_physical_attempts": 204}


def _sha(value):
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _verified_scope(info, run):
    record = info["contract"]
    profile, predecessor = record["profile"], record["predecessor"]
    run_path = Path(run)
    expected = {"provider_capability": "gated_ipv4", "arm": "A7", "memory": "M2", "device": "cuda",
                "initial_labels": 333, "target_labels": 1485, "batch_size": 32, "api_parallelism": 1,
                "store": False, "confirmation_seeds": [SEED], "qualification_seed": SEED,
                "requires_initial_qualification": False, "wire_representation": FORMAL_WIRE_REPRESENTATION}
    if (run_path.name != RUN or record["run_directory"] != run or record["api_execution"] != SSH_API
            or record["gpu_slots"] != [0] or record["budget"] != BUDGET
            or predecessor.get("dispatcher_module") != MODULE or predecessor.get("execution_budget") != BUDGET
            or predecessor.get("parent_run_directory") != str(run_path.with_name(PARENT_RUN))
            or predecessor.get("extension_run_directory") != str(run_path.with_name(EXTENSION_RUN))
            or (predecessor.get("parent_contract_sha256"), predecessor.get("parent_authorization_sha256"))
            != (PARENT_CONTRACT, PARENT_AUTHORIZATION)
            or (predecessor.get("extension_contract_sha256"), predecessor.get("extension_authorization_sha256"))
            != (EXTENSION_CONTRACT, EXTENSION_AUTHORIZATION)
            or (predecessor.get("seed"), predecessor.get("round")) != (SEED, START_ROUND)
            or "fixed_reference_repair" in profile
            or any(type(profile.get(key)) is not type(value) or profile[key] != value for key, value in expected.items())
            or request_timeouts(profile) != TIMEOUTS):
        raise PermissionError("the continuation differs from the authorized 34-round V13 scope")
    boundaries = profile.get("resume_boundaries", {})
    boundary = boundaries.get(str(SEED), {})
    if (set(boundaries) != {str(SEED)} or set(boundary) != {"round", "state_sha256", "memory_sha256"}
            or boundary["round"] != START_ROUND
            or not all(_sha(boundary.get(key)) for key in ("state_sha256", "memory_sha256"))
            or not all(_sha(predecessor.get(key)) for key in ("checkpoint_sha256", "extension_transition_sha256",
                "extension_completion_sha256", "extension_fit_audit_sha256", "extension_job_sha256",
                "extension_result_sha256"))
            or not isinstance(attempt := predecessor.get("extension_attempt_id"), str)
            or not ATTEMPT_ID.fullmatch(attempt)):
        raise PermissionError("the continuation must bind the verified L397 transition and original completed result")
    marker = {"contract_sha256": EXTENSION_CONTRACT, "seed": SEED, "round": START_ROUND, "active_labels": 397,
              "checkpoint_sha256": predecessor["checkpoint_sha256"], "state_sha256": boundary["state_sha256"],
              "memory_sha256": boundary["memory_sha256"], "transition_sha256": predecessor["extension_transition_sha256"],
              "original_future_budget": BUDGET}
    if digest(marker) != predecessor["extension_completion_sha256"]:
        raise PermissionError("the signed L397 handover does not match its completion marker")
    verify_local_source(info, root=ROOT)
    for path in SOURCES:
        if record["source_sha256"].get(path.as_posix()) != file_digest(ROOT / path):
            raise PermissionError(f"the approved continuation source differs: {path.name}")
    return predecessor


def _delivered_extension(spool, scope):
    _private_directory(spool)
    binding = read_bridge_json(spool / "run_binding.json")
    if (binding != {"contract_sha256": EXTENSION_CONTRACT, "authorization_sha256": EXTENSION_AUTHORIZATION, "mock": False}
            or binding["mock"] is not False):
        raise PermissionError("the parent spool is not the authorized L397 extension")
    identity = scope["extension_attempt_id"]
    filename = f"{identity}.json"
    job = read_bridge_json(spool / "jobs" / filename)
    request, config, started = verify_job(job, contract_sha256=EXTENSION_CONTRACT,
                                          authorization_sha256=EXTENSION_AUTHORIZATION,
                                          wire_representation=FORMAL_WIRE_REPRESENTATION)
    result = validate_result(read_bridge_json(spool / "results" / filename), job)
    ack = read_bridge_json(spool / "delivered" / filename)
    selected = read_bridge_json(spool / SELECTION)
    attempt = spool / "attempts" / identity
    capture = read_bridge_json(attempt / "provider_capture_receipt.json")
    raw = read_bridge_json(attempt / "raw_provider_response.json")
    model = read_bridge_json(spool / "provider_model_lock.json")
    state = request["binding"]
    if ((state["seed"], state["round"], request["logical_index"], started["attempt"]) != (SEED, START_ROUND - 1, 0, 0)
            or started["attempt_id"] != identity or job["job_sha256"] != scope["extension_job_sha256"]
            or selected != {"attempt_id": identity, "job_sha256": job["job_sha256"]}
            or result["result_sha256"] != scope["extension_result_sha256"] or result["provider_dispatched"] is not True
            or result["outcome"]["status"] != "completed" or raw != result["raw_response"] or raw.get("status") != "completed"
            or capture["attempt_id"] != identity or capture["job_sha256"] != job["job_sha256"]
            or capture["metadata"]["raw_response_sha256"] != result["raw_response_sha256"]
            or ack != {"attempt_id": identity, "result_sha256": result["result_sha256"]}
            or model != result["model_identity"] or model != scope.get("model_identity")):
        raise PermissionError("the continuation requires the same durably captured and delivered extension response")
    return config, model


class ContinuationDispatcher(PromptDispatcher):
    def validate(self, job):
        request, config, started = super().validate(job)
        state = request["binding"]
        predecessor = self.info["contract"]["predecessor"]
        if (state["seed"] != SEED or not START_ROUND <= state["round"] < END_ROUND
                or state["round"] == START_ROUND and state["checkpoint_sha256"] != predecessor["checkpoint_sha256"]):
            raise PermissionError("the continuation job is outside seed 2357's verified checkpoint and round range")
        return request, config, started


def run(args):
    if args.contract_sha256 in (PILOT_CONTRACT_SHA256, PARENT_CONTRACT, EXTENSION_CONTRACT):
        raise PermissionError("the continuation must have its own authorized run and Mac spool")
    spools = Path.home() / ".config/purellm/spool"
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.info(), expected_contract=args.contract_sha256, expected_auth=args.authorization_sha256)
    scope = _verified_scope(info, args.remote_run_dir)
    extension_transport, model = _delivered_extension(spools / EXTENSION_CONTRACT, scope)
    pilot, _, pilot_transport = _pool(spools / PILOT_CONTRACT_SHA256)
    transport = TransportConfig(**info["contract"]["profile"]["transport"])
    expected = replace(pilot_transport, timeout_seconds=TIMEOUTS.total_seconds, wire_mode=RESPONSES_STREAM_IPV4)
    if (transport != expected or extension_transport != replace(transport, logical_replies=1, transport_attempts=1)
            or (transport.provider, transport.base_url, transport.requested_model, transport.served_model,
                transport.reasoning_effort, transport.max_output_tokens, transport.sdk_version,
                transport.logical_replies, transport.transport_attempts)
            != ("token4research", "https://token4research.cn/v1", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh", 32768, "2.28.0", 3, 2)
            or not sdk_status(transport)["matches"]):
        raise PermissionError("the continuation must retain the original V13 transport and per-round budgets")
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
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    args = parser.parse_args()
    if not all(SHA256.fullmatch(value) for value in (args.contract_sha256, args.authorization_sha256)):
        parser.error("provide the exact approved continuation contract and authorization hashes")
    os.umask(0o077)
    print(dumps(run(args)))


if __name__ == "__main__":
    main()
