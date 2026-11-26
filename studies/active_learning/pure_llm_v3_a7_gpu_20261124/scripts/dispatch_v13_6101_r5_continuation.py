"""Dispatch ordinary 6101 traffic once the GPU attests its zero-POST L493 handover."""

import argparse
from dataclasses import replace
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import RESPONSES_STREAM_IPV4, TransportConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import read_bridge_json, validate_result, verify_job
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAT_SUFFIX, FORMAL_WIRE_REPRESENTATION
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import run_dispatcher, verify_local_source
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, SSH_API, request_timeouts
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import sdk_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import ATTEMPT_ID, GPU_PYTHON, SSHCourier, verify_info
from . import dispatch_v13_resilient_sse as resilient
from . import run_v13_6101_r5_continuation as gpu
from .dispatch_v13_relay_retry import RelayRetryDispatcher


def _scope(info, run, cached_contract, cached_auth, identity):
    cached = gpu.cached
    record = info["contract"]
    profile, proof = record["profile"], record["predecessor"]
    path = Path(run)
    expected = {
        "dispatcher_module": gpu.DISPATCHER, "seed": gpu.SEED, "round": gpu.START,
        "parent_run_directory": str(path.with_name(cached.OLD.name)),
        "parent_contract_sha256": cached.OLD.contract, "parent_authorization_sha256": cached.OLD.approval,
        "parent_seed_pause_sha256": cached.PAUSE_SHA, "extension_run_directory": str(path.with_name(cached.RUN_NAME)),
        "extension_contract_sha256": cached_contract, "extension_authorization_sha256": cached_auth,
        "model_identity": identity, "execution_budget": gpu.BUDGET, "recovery_budget": gpu.BOUNDARY_BUDGET,
        "bridge_wait_seconds": resilient.TIMEOUTS.bridge_wait_seconds,
    }
    settings = {
        "arm": "A7", "memory": "M2", "device": "cuda", "initial_labels": 333, "target_labels": 1485, "batch_size": 32,
        "confirmation_seeds": [gpu.SEED], "qualification_seed": gpu.SEED, "requires_initial_qualification": False,
        "api_parallelism": 1, "store": False, "wire_representation": FORMAL_WIRE_REPRESENTATION,
        "provider_capability": "gated_ipv4", "ipv4_readiness_windows": 3, "ipv4_preflight_retry": True,
        "sse_inactivity_seconds": resilient.INACTIVITY, "normalize_observed_reference_aliases": True,
        "audited_future_transport_replay": True, "sealed_relay_retry_seconds": 180, "replay_eligible_from_round": gpu.START,
    }
    boundaries = profile.get("resume_boundaries", {})
    boundary = boundaries.get(str(gpu.SEED), {})
    evidence = ("checkpoint_sha256", "cached_local_terminal_sha256", "cached_selection_sha256", "cached_selection_file_sha256",
                "extension_transition_sha256", "extension_source_sha256", "extension_fit_audit_sha256", "extension_freeze_sha256")
    custody = proof.get("reviewed_cached_selection", {})
    local_attempt = proof.get("cached_local_attempt_id")
    if (path.name != gpu.RUN_NAME or record["run_directory"] != run or record["api_execution"] != SSH_API
            or record["gpu_slots"] != [0] or record["budget"] != gpu.BUDGET or "fixed_reference_repair" in profile
            or request_timeouts(profile) != resilient.TIMEOUTS
            or any(type(proof.get(k)) is not type(v) or proof[k] != v for k, v in expected.items())
            or any(type(profile.get(k)) is not type(v) or profile[k] != v for k, v in settings.items())
            or any(not resilient._sha(proof.get(k)) for k in evidence)
            or not isinstance(local_attempt, str) or not ATTEMPT_ID.fullmatch(local_attempt)
            or local_attempt in (cached.ATTEMPT, cached.FAILED_ATTEMPT["attempt_id"])
            or type(custody) is not dict or custody.get("provider_post") is not False
            or proof["cached_selection_sha256"] != digest(custody)
            or set(boundaries) != {str(gpu.SEED)} or set(boundary) != {"round", "state_sha256", "memory_sha256"}
            or boundary["round"] != gpu.START or any(not resilient._sha(boundary[k]) for k in ("state_sha256", "memory_sha256"))):
        raise PermissionError("the approved continuation does not describe a verified normal L493 handover")
    verify_local_source(info, root=ROOT)
    for source in (*resilient.SOURCES, *gpu.SOURCES):
        if record["source_sha256"].get(source.as_posix()) != file_digest(ROOT / source):
            raise PermissionError(f"the continuation source is outside its signed manifest: {source.name}")
    return proof


def _archive(spool, custody):
    cached = gpu.cached
    filename = f"{cached.ATTEMPT}.json"
    job = read_bridge_json(spool / "jobs" / filename)
    request, transport, started = verify_job(job, contract_sha256=cached.OLD.contract,
                                              authorization_sha256=cached.OLD.approval, wire_representation=FORMAL_WIRE_REPRESENTATION)
    result = validate_result(read_bridge_json(spool / "results" / filename), job)
    attempt = spool / "attempts" / cached.ATTEMPT
    raw = read_bridge_json(attempt / "raw_provider_response.json")
    receipt = read_bridge_json(attempt / "provider_capture_receipt.json")
    reply = result["outcome"].get("reply", {})
    identity = read_bridge_json(spool / "provider_model_lock.json")
    if ((request["request_id"], request["payload_sha256"], started["attempt_id"], job["job_sha256"],
         result["result_sha256"], digest(raw), reply.get("response_id")) != (
            cached.REQUEST, cached.PAYLOAD, cached.ATTEMPT, cached.JOB, cached.RESULT, cached.RAW, cached.RESPONSE)
            or result["outcome"]["status"] != "completed" or result["provider_dispatched"] is not True
            or raw != result["raw_response"] or result["model_identity"] != identity
            or receipt.get("attempt_id") != cached.ATTEMPT or receipt.get("job_sha256") != cached.JOB
            or receipt.get("metadata", {}).get("raw_response_sha256") != cached.RAW
            or digest(reply.get("text")) != cached.TEXT
            or read_bridge_json(spool / "delivered" / filename) != {"attempt_id": cached.ATTEMPT, "result_sha256": cached.RESULT}):
        raise PermissionError("the original completed response and capture acknowledgment are required")
    failed = cached.FAILED_ATTEMPT
    retries, correction_transport, retry_started, retry_job, retry_result = resilient._sealed(
        spool, failed["attempt_id"], cached.OLD.contract, cached.OLD.approval,
    )
    failure = retry_result["outcome"]["failure"]
    progress = spool / "attempts" / failed["attempt_id"] / "provider_progress.json"
    trace = read_bridge_json(progress)
    phases = trace.get("network_phases", {})
    if (correction_transport != transport or retries["binding"] != request["binding"]
            or (retries["request_id"], retries["payload_sha256"], retries["logical_index"], retry_started["attempt"])
            != (cached.FAILED_REQUEST, cached.FAILED_PAYLOAD, 1, 0)
            or retries["payload"]["input"][:2] != request["payload"]["input"]
            or (retry_job["job_sha256"], retry_result["result_sha256"], failure["code"],
                retry_result["provider_dispatched"], failure["ambiguous"]) != (
                    failed["job_sha256"], failed["result_sha256"], failed["error"], True, True)
            or file_digest(progress) != cached.FAILED_PROGRESS
            or phases.get("send_body", {}).get("state") != "complete"
            or phases.get("receive_headers", {}).get("state") != "failed"
            or any(key in trace for key in ("http_status", "response_id", "provider_request_id"))):
        raise PermissionError("the separate dispatched and ambiguous R4 correction must remain in the original ledger")
    plan, paths = cached._project_mirrors(cached.parse_response(reply["text"]), stage=None)
    if (digest(plan), digest(paths), custody.get("normalized_plan_sha256"), custody.get("removed_paths"),
        custody.get("response_id"), custody.get("failed_correction", {}).get("result_sha256")) != (
            cached.PLAN, cached.PATHS, cached.PLAN, paths, cached.RESPONSE, failed["result_sha256"]):
        raise PermissionError("the original response does not reproduce the exact reviewed fitted plan")
    return request, transport


class ContinuationDispatcher(RelayRetryDispatcher):
    def validate(self, job):
        request, config, started = super().validate(job)
        if ("fixed_reference_repair" in request["binding"] or started["attempt_id"] in (
                gpu.cached.ATTEMPT, gpu.cached.FAILED_ATTEMPT["attempt_id"], self.scope["cached_local_attempt_id"])):
            raise PermissionError("the ordinary continuation cannot submit any archived R4 attempt")
        return request, config, started

    def _check_boundary(self, request, started):
        if (request["binding"]["checkpoint_sha256"] != self.scope["checkpoint_sha256"]
                or request["logical_index"] != 0 or started["attempt"] not in (0, 1)):
            raise PermissionError("R5 permits one ordinary logical decision from its fitted checkpoint")
        if started["attempt"] == 1:
            self._check_boundary_retry(request, started)


def run(args):
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.retry_exchange("info"), expected_contract=args.contract_sha256,
                       expected_auth=args.authorization_sha256)
    cached = gpu.cached
    spools = Path.home() / ".config/purellm/spool"
    old_spool = resilient._spool(spools, cached.OLD.contract, cached.OLD.approval)
    model = read_bridge_json(old_spool / "provider_model_lock.json")
    proof = _scope(info, args.remote_run_dir, args.cached_contract_sha256, args.cached_authorization_sha256, model)
    original, prior_transport = _archive(old_spool, proof["reviewed_cached_selection"])
    system = original["payload"]["input"][0]["content"]
    prompt_sha = resilient._approved_prompt_sha(info)
    if not system.endswith(FORMAT_SUFFIX) or prompt_sha != digest(system.removesuffix(FORMAT_SUFFIX)):
        raise PermissionError("the ordinary W08 Prompt is required")
    pilot, _, pilot_transport = resilient._pool(spools / resilient.PILOT_CONTRACT_SHA256)
    transport = TransportConfig(**info["contract"]["profile"]["transport"])
    registered = replace(pilot_transport, timeout_seconds=resilient.TIMEOUTS.total_seconds, wire_mode=RESPONSES_STREAM_IPV4)
    if (transport != registered or prior_transport != registered or not sdk_status(transport)["matches"]
            or args.contract_sha256 in (cached.OLD.contract, args.cached_contract_sha256, resilient.PILOT_CONTRACT_SHA256)):
        raise PermissionError("use a new spool with the original Astra/xhigh wire, timeout and SDK")
    if args.verify_only:
        return {"seed": gpu.SEED, "round": gpu.START, "contract_sha256": args.contract_sha256, "verified_no_dispatch": True}
    spool = spools / args.contract_sha256
    dispatcher = ContinuationDispatcher(spool, info=info, contract_sha256=args.contract_sha256, transport=transport,
                                        mock=False, pool=resilient.SharedKeyPool(pilot.directory, pilot.config))
    dispatcher.scope = proof
    dispatcher.prompt_sha = prompt_sha
    once(spool / "provider_model_lock.json", model)
    return run_dispatcher(dispatcher, courier, watch=True, workers=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--execute-paid", action="store_true")
    parser.add_argument("--ssh-host", default="eit-hpc")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--remote-python", default=GPU_PYTHON)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--cached-contract-sha256", required=True)
    parser.add_argument("--cached-authorization-sha256", required=True)
    args = parser.parse_args()
    if not all(resilient._sha(value) for value in (args.contract_sha256, args.authorization_sha256,
                                                  args.cached_contract_sha256, args.cached_authorization_sha256)):
        parser.error("exact continuation and cached authorization hashes are required")
    os.umask(0o077)
    print(dumps(run(args)))


if __name__ == "__main__":
    main()
