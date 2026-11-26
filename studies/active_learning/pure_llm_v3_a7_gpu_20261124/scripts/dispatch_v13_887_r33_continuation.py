"""Dispatch only the signed ordinary R33–R35 887 requests after the exact targeted fit."""

import argparse
from dataclasses import replace
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_STREAM_IPV4, TransportConfig, is_sealed_relay_failure,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import (
    ATTEMPT_ID, read_bridge_json, validate_result, verify_job,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAL_WIRE_REPRESENTATION
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import run_dispatcher, verify_local_source
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, SSH_API, request_timeouts
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import _reply_from_raw, sdk_status, verify_reported_model
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import GPU_PYTHON, SHA256, SSHCourier, verify_info
from . import dispatch_v13_887_r32_targeted_repair as archived
from . import dispatch_v13_resilient_sse as resilient
from . import run_v13_887_r33_continuation as gpu
from .dispatch_v13_relay_retry import RelayRetryDispatcher


def _scope(info, run, identity):
    record = info["contract"]
    profile, proof = record["profile"], record["predecessor"]
    path = Path(run)
    expected = {
        "dispatcher_module": gpu.DISPATCHER, "seed": gpu.SEED, "round": gpu.START,
        "previous_run_directory": str(path.with_name(gpu.REPAIRED.name)),
        "previous_contract_sha256": gpu.REPAIRED.contract, "previous_authorization_sha256": gpu.REPAIRED.approval,
        "normal_run_directory": str(path.with_name(gpu.repair.PARENT.name)),
        "normal_contract_sha256": gpu.repair.PARENT.contract, "normal_authorization_sha256": gpu.repair.PARENT.approval,
        "normal_seed_pause_file_sha256": gpu.repair.OLD_PAUSE, "model_identity": identity,
        "recovery_budget": gpu.BOUNDARY_BUDGET, "execution_budget": gpu.BUDGET,
        "bridge_wait_seconds": resilient.TIMEOUTS.bridge_wait_seconds,
    }
    settings = {
        "arm": "A7", "memory": "M2", "device": "cuda", "initial_labels": 333, "target_labels": gpu.TARGET, "batch_size": 32,
        "confirmation_seeds": [gpu.SEED], "qualification_seed": gpu.SEED, "requires_initial_qualification": False,
        "api_parallelism": 1, "store": False, "wire_representation": FORMAL_WIRE_REPRESENTATION,
        "provider_capability": "gated_ipv4", "ipv4_readiness_windows": 3, "ipv4_preflight_retry": True,
        "sse_inactivity_seconds": resilient.INACTIVITY, "normalize_observed_reference_aliases": True,
        "full_imported_boundary_round": gpu.START, "audited_future_transport_replay": True,
        "sealed_relay_retry_seconds": 180, "replay_eligible_from_round": gpu.START,
    }
    handoff, preview = proof["r32_handoff"], proof["r33_request"]
    boundary = {"round": gpu.START, "state_sha256": handoff["state_sha256"], "memory_sha256": handoff["memory_sha256"]}
    if (path.name != gpu.RUN_NAME or record["run_directory"] != run or record["api_execution"] != SSH_API
            or record["gpu_slots"] != [0] or record["budget"] != gpu.BUDGET or "fixed_reference_repair" in profile
            or request_timeouts(profile) != resilient.TIMEOUTS
            or any(type(proof.get(key)) is not type(value) or proof[key] != value for key, value in expected.items())
            or any(type(profile.get(key)) is not type(value) or profile[key] != value for key, value in settings.items())
            or profile.get("resume_boundaries") != {str(gpu.SEED): boundary}
            or Path(proof["previous_release_directory"]).name != gpu.REPAIR_RELEASE
            or (handoff["contract_sha256"], handoff["authorization_sha256"], handoff["candidate_order_sha256"])
            != (gpu.REPAIRED.contract, gpu.REPAIRED.approval, gpu.repair.ORDER)
            or not handoff["fit_files"] or not all(SHA256.fullmatch(value) for value in handoff["fit_files"].values())
            or preview != gpu.R33_REQUEST
            or not all(SHA256.fullmatch(value) for value in (*preview.values(), handoff["freeze_sha256"],
                                                             handoff["state_sha256"], handoff["memory_sha256"]))
            or (handoff["checkpoint_sha256"], handoff["memory_sha256"], handoff["freeze_sha256"])
            != (gpu.REPAIRED_CHECKPOINT, gpu.REPAIRED_MEMORY, gpu.REPAIRED_FREEZE)
            or proof["checkpoint_sha256"] != handoff["checkpoint_sha256"] or preview["checkpoint_sha256"] != proof["checkpoint_sha256"]):
        raise PermissionError("the supplied Mac scope is not the signed normal L1389–L1485 continuation")
    verify_local_source(info, root=ROOT)
    fixed_sources = {**gpu.SOURCE_BASE, **gpu.REPAIR_SOURCES, gpu.SEED_RUNNER: gpu.RUNNER_PATCH}
    for source in (*resilient.SOURCES, *gpu.SOURCES, *(Path(name) for name in fixed_sources)):
        key = source.as_posix()
        actual = file_digest(ROOT / source)
        if record["source_sha256"].get(key) != actual or fixed_sources.get(key, actual) != actual:
            raise PermissionError(f"the installed continuation source differs from approval: {source.name}")
    if resilient._approved_prompt_sha(info) != preview["prompt_sha256"]:
        raise PermissionError("the first request does not bind the normal A7 prompt")
    return proof


def _repair_archive(spools, proof, original, model):
    handoff = proof["r32_handoff"]["bridge"]
    attempt = handoff["attempt_id"]
    if not ATTEMPT_ID.fullmatch(attempt) or attempt != gpu.REPAIR_ATTEMPT or handoff["provider_post"] is not True:
        raise PermissionError("the fitted R32 correction requires one exact completed provider attempt")
    spool = resilient._spool(spools, gpu.REPAIRED.contract, gpu.REPAIRED.approval)
    name = f"{attempt}.json"
    slot = spool / "slots" / f"{gpu.SEED}-{gpu.START - 1}-0-0.json"
    job_path, result_path = spool / "jobs" / name, spool / "results" / name
    job = read_bridge_json(job_path)
    request, config, started = verify_job(job, contract_sha256=gpu.REPAIRED.contract,
                                          authorization_sha256=gpu.REPAIRED.approval,
                                          wire_representation=FORMAL_WIRE_REPRESENTATION)
    result = validate_result(read_bridge_json(result_path), job)
    attempt_path = spool / "attempts" / attempt
    raw = read_bridge_json(attempt_path / "raw_provider_response.json")
    capture = read_bridge_json(attempt_path / "provider_capture_receipt.json")
    ack_path = spool / "delivered" / name
    ack = read_bridge_json(ack_path)
    gpu.repair.verify_patch_request(request, original, gpu.repair._metadata(original), attempt=started["attempt"])
    if ({path.name for path in (spool / "slots").glob(f"{gpu.SEED}-{gpu.START - 1}-*-*.json")} != {slot.name}
            or read_bridge_json(slot) != {"attempt_id": attempt, "job_sha256": handoff["job_sha256"]}
            or (request["request_id"], request["payload_sha256"], started["attempt_id"])
            != (handoff["request_id"], handoff["payload_sha256"], attempt)
            or (job["job_sha256"], result["result_sha256"], digest(raw), result["raw_response_sha256"])
            != (handoff["job_sha256"], handoff["result_sha256"], handoff["raw_sha256"], handoff["raw_sha256"])
            or (file_digest(job_path), file_digest(result_path)) != (handoff["job_file_sha256"], handoff["result_file_sha256"])
            or result["outcome"]["status"] != "completed" or result["provider_dispatched"] is not True
            or raw != result["raw_response"] or raw["status"] != "completed" or result["model_identity"] != model
            or read_bridge_json(spool / "provider_model_lock.json") != model
            or ack != {"attempt_id": attempt, "result_sha256": handoff["result_sha256"]}
            or file_digest(ack_path) != gpu.REPAIR_ACK
            or capture.get("attempt_id") != attempt or capture.get("job_sha256") != handoff["job_sha256"]
            or capture.get("metadata", {}).get("raw_response_sha256") != handoff["raw_sha256"]
            or capture["metadata"].get("provider_request_id") != handoff["provider_request_id"]):
        raise PermissionError("the Mac repair archive does not match the signed GPU custody")
    reply = _reply_from_raw(raw, handoff["provider_request_id"])
    raw_model = {"model_version": raw.get("model_version"), "reported_reasoning_effort": raw["reasoning"]["effort"]}
    if ((reply.response_id, digest(reply.text), reply.usage) != (
            handoff["response_id"], handoff["answer_sha256"], handoff["usage"])
            or verify_reported_model(config, reply, raw_model) != model):
        raise PermissionError("the completed R32 repair answer or actual model identity changed")
    return config


class ContinuationDispatcher(RelayRetryDispatcher):
    def _prior(self, request, logical, physical):
        binding = request["binding"]
        slot = self.path / "slots" / f"{binding['seed']}-{binding['round']}-{logical}-{physical}.json"
        if not slot.exists():
            return None
        reservation = read_bridge_json(slot)
        identity = reservation["attempt_id"]
        if not ATTEMPT_ID.fullmatch(identity):
            raise PermissionError("invalid prior Mac physical reservation")
        name = f"{identity}.json"
        job = read_bridge_json(self.path / "jobs" / name)
        old, config, started = verify_job(job, contract_sha256=self.info["contract_sha256"],
                                          authorization_sha256=self.info["authorization_sha256"],
                                          wire_representation=FORMAL_WIRE_REPRESENTATION)
        result = validate_result(read_bridge_json(self.path / "results" / name), job)
        ack = read_bridge_json(self.path / "delivered" / name)
        if (old["binding"] != binding or config != self.transport or old["logical_index"] != logical
                or old["payload"]["input"][:2] != request["payload"]["input"][:2]
                or started["attempt"] != physical or started["attempt_id"] != identity
                or reservation != {"attempt_id": identity, "job_sha256": job["job_sha256"]}
                or ack != {"attempt_id": identity, "result_sha256": result["result_sha256"]}):
            raise PermissionError("the prior attempt does not match the current R33 scientific context")
        return old, result

    def _check_boundary(self, request, started):
        gpu.check_first_round(request, self.scope["r33_request"])
        logical = request["logical_index"]
        if logical:
            prior = [item for index in (0, 1) if (item := self._prior(request, logical - 1, index)) is not None]
            complete = [result for _, result in prior if result["outcome"]["status"] == "completed"]
            if len(complete) != 1 or complete[0]["raw_response"] is None:
                raise PermissionError("R33 correction requires the previous completed and delivered response")
        if started["attempt"] == 1:
            pair = self._prior(request, logical, 0)
            if pair is None:
                raise PermissionError("R33 physical retry requires its own delivered P0")
            old, result = pair
            failure = result["outcome"].get("failure") or {}
            evidence = failure.get("reply_identity") or {}
            attempt_path = self.path / "attempts" / result["attempt_id"]
            if (old != request or result["attempt_id"] == started["attempt_id"]
                    or result["provider_dispatched"] is not True or result["raw_response"] is not None
                    or result["outcome"]["status"] != "failed" or failure.get("ambiguous") is not True
                    or failure.get("retryable") is not False or evidence.get("usage") != {}
                    or evidence.get("served_model") not in (None, self.transport.served_model)
                    or not is_sealed_relay_failure({**evidence, "error": failure.get("code")})
                    or any((attempt_path / name).exists() for name in ("raw_provider_response.json", "provider_capture_receipt.json"))):
                raise PermissionError("R33 P1 requires the exact sealed no-answer relay failure")

    def validate(self, job):
        request, config, started = super().validate(job)
        if "fixed_reference_repair" in request["binding"] or started["attempt_id"] in {
            gpu.repair.ATTEMPT, self.scope["r32_handoff"]["bridge"]["attempt_id"],
        }:
            raise PermissionError("the normal continuation does not replay the archived R32 answer or correction")
        return request, config, started


def run(args):
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.retry_exchange("info"), expected_contract=args.contract_sha256, expected_auth=args.authorization_sha256)
    spools = Path.home() / ".config/purellm/spool"
    if args.contract_sha256 in {gpu.repair.PARENT.contract, gpu.REPAIRED.contract, resilient.PILOT_CONTRACT_SHA256}:
        raise PermissionError("the normal R33 continuation requires a distinct authorized Mac spool")
    original, old_transport, identity = archived._archive(spools)
    proof = _scope(info, args.remote_run_dir, identity)
    repaired_transport = _repair_archive(spools, proof, original, identity)
    pilot, _, pilot_transport = resilient._pool(spools / resilient.PILOT_CONTRACT_SHA256)
    transport = TransportConfig(**info["contract"]["profile"]["transport"])
    expected = replace(pilot_transport, timeout_seconds=resilient.TIMEOUTS.total_seconds, wire_mode=RESPONSES_STREAM_IPV4)
    if (transport != expected or old_transport != expected
            or repaired_transport != replace(expected, logical_replies=1, transport_attempts=1)
            or not sdk_status(transport)["matches"]):
        raise PermissionError("the continuation and the two archived requests must retain Astra/xhigh and the reviewed protocol")
    if args.verify_only:
        return {"seed": gpu.SEED, "round": gpu.START, "contract_sha256": args.contract_sha256, "verified_no_dispatch": True}
    spool = spools / args.contract_sha256
    dispatcher = ContinuationDispatcher(spool, info=info, contract_sha256=args.contract_sha256, transport=transport,
                                        mock=False, pool=resilient.SharedKeyPool(pilot.directory, pilot.config))
    dispatcher.scope = proof
    dispatcher.prompt_sha = resilient._approved_prompt_sha(info)
    once(spool / "provider_model_lock.json", identity)
    return run_dispatcher(dispatcher, courier, watch=True, workers=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify-only", action="store_true")
    action.add_argument("--execute-paid", action="store_true")
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
