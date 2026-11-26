"""Dispatch normal seed 887 traffic only after the signed L781 repair handover."""

import argparse
from dataclasses import replace
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import parse_response
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import RESPONSES_STREAM_IPV4, TransportConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import read_bridge_json, validate_result, verify_job
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAT_SUFFIX, FORMAL_WIRE_REPRESENTATION
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import run_dispatcher, verify_local_source
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, SSH_API, request_timeouts
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import sdk_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import ATTEMPT_ID, GPU_PYTHON, SHA256, SSHCourier, verify_info
from . import dispatch_v13_887_r13_targeted_repair as old_mac
from . import dispatch_v13_resilient_sse as resilient
from . import run_v13_887_r14_continuation as gpu
from .dispatch_v13_relay_retry import RelayRetryDispatcher


def _scope(info, run, original, identity):
    record = info["contract"]
    profile, proof = record["profile"], record["predecessor"]
    path = Path(run)
    expected = {
        "dispatcher_module": gpu.DISPATCHER, "seed": gpu.SEED, "round": gpu.START,
        "parent_run_directory": str(path.with_name(gpu.repair.PARENT.name)),
        "parent_contract_sha256": gpu.repair.PARENT.contract, "parent_authorization_sha256": gpu.repair.PARENT.approval,
        "parent_seed_pause_sha256": gpu.repair.OLD_PAUSE, "extension_run_directory": str(path.with_name(gpu.cached.RUN_NAME)),
        "extension_contract_sha256": gpu.CACHED_CONTRACT, "extension_authorization_sha256": gpu.CACHED_AUTH,
        "paid_correction_run_directory": str(path.with_name(gpu.cached.CORRECTION.name)),
        "paid_correction_contract_sha256": gpu.cached.CORRECTION.contract,
        "paid_correction_authorization_sha256": gpu.cached.CORRECTION.approval,
        "checkpoint_sha256": gpu.CHECKPOINT, "extension_transition_sha256": gpu.TRANSITION,
        "extension_source_sha256": gpu.STATE, "cached_correction_file_sha256": gpu.CUSTODY,
        "extension_projected_plan_sha256": gpu.cached.PROJECTED,
        "model_identity": identity, "execution_budget": gpu.BUDGET, "recovery_budget": gpu.BOUNDARY_BUDGET,
        "bridge_wait_seconds": resilient.TIMEOUTS.bridge_wait_seconds, "r14_request": gpu.R14_REQUEST,
        "cached_local_attempt_id": gpu.LOCAL_ATTEMPT, "cached_local_terminal_sha256": gpu.LOCAL_TERMINAL,
        "cached_correction_sha256": gpu.CUSTODY_SHA, "extension_fit_audit_sha256": gpu.FIT_AUDIT,
        "extension_freeze_sha256": gpu.FREEZE,
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
    evidence = ("cached_local_terminal_sha256", "cached_correction_sha256", "extension_fit_audit_sha256", "extension_freeze_sha256")
    custody = proof.get("cached_correction") or {}
    if (path.name != gpu.RUN_NAME or record["run_directory"] != run or record["api_execution"] != SSH_API
            or record["gpu_slots"] != [0] or record["budget"] != gpu.BUDGET or "fixed_reference_repair" in profile
            or request_timeouts(profile) != resilient.TIMEOUTS
            or any(type(proof.get(k)) is not type(v) or proof[k] != v for k, v in expected.items())
            or any(type(profile.get(k)) is not type(v) or profile[k] != v for k, v in settings.items())
            or any(not resilient._sha(proof.get(k)) for k in evidence)
            or not isinstance(proof.get("cached_local_attempt_id"), str) or not ATTEMPT_ID.fullmatch(proof["cached_local_attempt_id"])
            or proof["cached_local_attempt_id"] == gpu.cached.ATTEMPT or custody.get("provider_post") is not False
            or proof["cached_correction_sha256"] != digest(custody)
            or Path(proof.get("extension_release_directory", "")).name != gpu.CACHED_RELEASE
            or set(boundaries) != {str(gpu.SEED)} or set(boundary) != {"round", "state_sha256", "memory_sha256"}
            or boundary != {"round": gpu.START, "state_sha256": gpu.STATE_SHA, "memory_sha256": gpu.MEMORY_SHA}):
        raise PermissionError("the signed scope is not the normal L781–L1485 continuation")
    verify_local_source(info, root=ROOT)
    for source in (*resilient.SOURCES, *gpu.repair.SOURCES, *gpu.cached.SOURCES, *gpu.SOURCES):
        if record["source_sha256"].get(source.as_posix()) != file_digest(ROOT / source):
            raise PermissionError(f"the continuation's source is outside its signed manifest: {source.name}")
    system = original["payload"]["input"][0]["content"]
    if not system.endswith(FORMAT_SUFFIX) or resilient._approved_prompt_sha(info) != digest(system.removesuffix(FORMAT_SUFFIX)):
        raise PermissionError("the regular W10 Prompt must be restored before dispatch")
    return proof


def _completed_mac(spools, proof, original, old_transport, identity):
    cached = gpu.cached
    spool = resilient._spool(spools, cached.CORRECTION.contract, cached.CORRECTION.approval)
    attempt = cached.ATTEMPT
    filename = f"{attempt}.json"
    job = read_bridge_json(spool / "jobs" / filename)
    request, transport, started = verify_job(job, contract_sha256=cached.CORRECTION.contract,
                                              authorization_sha256=cached.CORRECTION.approval, wire_representation=FORMAL_WIRE_REPRESENTATION)
    gpu.repair.verify_patch_request(request, original, gpu.repair._metadata(original), attempt=started["attempt"])
    result = validate_result(read_bridge_json(spool / "results" / filename), job)
    attempt_dir = spool / "attempts" / attempt
    raw = read_bridge_json(attempt_dir / "raw_provider_response.json")
    capture = read_bridge_json(attempt_dir / "provider_capture_receipt.json")
    custody = proof["cached_correction"]
    if (transport != replace(old_transport, logical_replies=1, transport_attempts=1)
            or (request["request_id"], request["payload_sha256"], started["attempt_id"], job["job_sha256"],
                result["result_sha256"], digest(raw)) != (cached.REQUEST, cached.PAYLOAD, attempt, cached.JOB, cached.RESULT, cached.RAW)
            or (custody.get("attempt_id"), custody.get("result_sha256"), custody.get("raw_response_sha256"),
                custody.get("reply_text_sha256"), custody.get("ordered_candidate_ids_sha256"))
            != (attempt, cached.RESULT, cached.RAW, cached.TEXT, gpu.repair.ORDER)
            or result["outcome"] != {"status": "failed", "failure": cached.FAILURE}
            or result["provider_dispatched"] is not True or raw != result["raw_response"]
            or capture.get("attempt_id") != attempt or capture.get("job_sha256") != job["job_sha256"]
            or capture.get("metadata", {}).get("raw_response_sha256") != digest(raw)
            or result["model_identity"] is not None or read_bridge_json(spool / "provider_model_lock.json") != identity
            or file_digest(spool / "delivered" / filename) != cached.MAC_ACK or digest(capture) != cached.MAC_CAPTURE
            or file_digest(spool / "jobs" / filename) != cached.JOB_FILE
            or file_digest(spool / "results" / filename) != cached.RESULT_FILE
            or read_bridge_json(spool / "delivered" / filename) != {"attempt_id": attempt, "result_sha256": result["result_sha256"]}):
        raise PermissionError("the original correction is not complete and acknowledged by the same GPU run")
    parent = resilient._spool(spools, gpu.repair.PARENT.contract, gpu.repair.PARENT.approval)
    original_reply = read_bridge_json(parent / "results" / f"{gpu.repair.ATTEMPT}.json")["outcome"]["reply"]
    plan = gpu.repair._original_plan(original_reply["text"])
    archived = cached._extract_final(raw, transport, identity)
    patch = parse_response(archived["text"])
    if (digest(archived["text"]) != cached.TEXT
            or digest(cached.project(patch, original=plan)) != proof["extension_projected_plan_sha256"]):
        raise PermissionError("the delivered correction does not reproduce the signed fitted plan")


class ContinuationDispatcher(RelayRetryDispatcher):
    def validate(self, job):
        request, config, started = super().validate(job)
        if "fixed_reference_repair" in request["binding"] or started["attempt_id"] in {
            gpu.repair.ATTEMPT, gpu.cached.ATTEMPT, self.scope["cached_local_attempt_id"],
        }:
            raise PermissionError("the normal continuation cannot replay the completed selection or its repair")
        return request, config, started

    def _check_boundary(self, request, started):
        if request["logical_index"] != 0 or gpu.initial_request_identity(request) != self.scope["r14_request"]:
            raise PermissionError("R14 authorizes one logical decision from its exact fitted checkpoint")
        if started["attempt"] == 1:
            self._check_boundary_retry(request, started)


def run(args):
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.retry_exchange("info"), expected_contract=args.contract_sha256, expected_auth=args.authorization_sha256)
    spools = Path.home() / ".config/purellm/spool"
    original, prior_transport, identity = old_mac._archive(spools)
    proof = _scope(info, args.remote_run_dir, original, identity)
    _completed_mac(spools, proof, original, prior_transport, identity)
    pilot, _, pilot_transport = resilient._pool(spools / resilient.PILOT_CONTRACT_SHA256)
    transport = TransportConfig(**info["contract"]["profile"]["transport"])
    registered = replace(pilot_transport, timeout_seconds=resilient.TIMEOUTS.total_seconds, wire_mode=RESPONSES_STREAM_IPV4)
    if transport != registered or prior_transport != registered or not sdk_status(transport)["matches"]:
        raise PermissionError("the restored Prompt must use the original Astra/xhigh transport and SDK")
    if args.verify_only:
        return {"seed": gpu.SEED, "round": gpu.START, "verified_no_dispatch": True}
    spool = spools / args.contract_sha256
    dispatcher = ContinuationDispatcher(spool, info=info, contract_sha256=args.contract_sha256, transport=transport,
                                        mock=False, pool=resilient.SharedKeyPool(pilot.directory, pilot.config))
    dispatcher.scope = proof
    dispatcher.prompt_sha = resilient._approved_prompt_sha(info)
    once(spool / "provider_model_lock.json", identity)
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
    args = parser.parse_args()
    if not all(SHA256.fullmatch(value) for value in (args.contract_sha256, args.authorization_sha256)):
        parser.error("provide both exact continuation authorization hashes")
    os.umask(0o077)
    print(dumps(run(args)))


if __name__ == "__main__":
    main()
