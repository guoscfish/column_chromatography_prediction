"""Dispatch the signed 6101 ordinary R11–R35 after its verified zero-POST L685 fit."""

import argparse
from dataclasses import replace
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import parse_response
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_STREAM_IPV4, TransportConfig, is_sealed_relay_failure,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import read_bridge_json, verify_job
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAL_WIRE_REPRESENTATION
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import run_dispatcher, verify_local_source
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, SSH_API, request_timeouts
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import _reply_from_raw, sdk_status, verify_reported_model
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import GPU_PYTHON, SHA256, SSHCourier, verify_info
from . import dispatch_v13_6101_r9_continuation as normal
from . import dispatch_v13_resilient_sse as resilient
from . import run_v13_6101_r11_continuation as gpu
from .dispatch_v13_relay_retry import RelayRetryDispatcher


def _archived_mac(spools, custody):
    old = gpu.cached
    spool = resilient._spool(spools, old.OLD.contract, old.OLD.approval)
    old.verify_mac_archive(spool)
    slot = spool / "slots" / f"{gpu.SEED}-{old.ROUND}-0-0.json"
    name = f"{old.ATTEMPT}.json"
    job = read_bridge_json(spool / "jobs" / name)
    request, transport, started = verify_job(job, contract_sha256=old.OLD.contract,
                                              authorization_sha256=old.OLD.approval,
                                              wire_representation=FORMAL_WIRE_REPRESENTATION)
    attempt = spool / "attempts" / old.ATTEMPT
    raw = read_bridge_json(attempt / "raw_provider_response.json")
    identity = read_bridge_json(spool / "provider_model_lock.json")
    if (set((spool / "slots").glob(f"{gpu.SEED}-{old.ROUND}-*-*.json")) != {slot}
            or read_bridge_json(slot) != {"attempt_id": old.ATTEMPT, "job_sha256": old.JOB}
            or (request["request_id"], request["payload_sha256"], job["job_sha256"], started["attempt_id"])
            != (old.REQUEST, old.PAYLOAD, old.JOB, old.ATTEMPT)
            or type(custody) is not dict or custody.get("provider_post") is not False
            or digest(custody) != gpu.HANDOFF["cached_selection_sha256"]):
        raise PermissionError("the Mac archive differs from the signed zero-POST L685 proof")
    declared = {"model_version": raw.get("model_version"), "reported_reasoning_effort": raw["reasoning"]["effort"]}
    reply = _reply_from_raw(raw, old.PROVIDER_REQUEST)
    final = parse_response(reply.text)
    if ((reply.response_id, digest(reply.text), reply.usage) != (old.RESPONSE, old.TEXT, old.USAGE)
            or verify_reported_model(transport, reply, declared) != identity or digest(final) != old.PLAN
            or digest([choice["candidate_id"] for choice in final["choices"]]) != old.ORDER):
        raise PermissionError("the exact patched parser does not reproduce the archived R10 final answer")
    return transport, identity


def _scope(info, run, identity):
    record = info["contract"]
    profile, proof, path = record["profile"], record["predecessor"], Path(run)
    handoff, first = gpu.validate_pins({key: proof[key] for key in (
        "r10_handoff", "r11_request", "cached_response_import_file_sha256",
    )})
    expected = {
        "dispatcher_module": gpu.DISPATCHER, "seed": gpu.SEED, "round": gpu.START,
        "previous_run_directory": str(path.with_name(gpu.CACHED.name)),
        "previous_contract_sha256": gpu.CACHED.contract, "previous_authorization_sha256": gpu.CACHED.approval,
        "normal_run_directory": str(path.with_name(gpu.cached.OLD.name)),
        "normal_contract_sha256": gpu.cached.OLD.contract, "normal_authorization_sha256": gpu.cached.OLD.approval,
        "normal_seed_pause_file_sha256": gpu.cached.ARCHIVE_FILES[f"seed_{gpu.SEED}/pause.request"],
        "checkpoint_sha256": handoff["checkpoint_sha256"], "model_identity": identity,
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
    boundary = {"round": gpu.START, "state_sha256": handoff["state_sha256"],
                "memory_sha256": handoff["scientific_memory_sha256"]}
    fixed_sources = {
        gpu.PARSER: gpu.NEW_PARSER, gpu.normal.SEED_RUNNER: gpu.RUNNER_PATCH, gpu.CACHED_MODULE: gpu.CACHED_HELPER,
        gpu.seed_runner.PROMPT_PATH.as_posix(): gpu.NORMAL_PROMPT,
    }
    signed_sources = record["source_sha256"]
    if (path.name != gpu.RUN_NAME or record["run_directory"] != run or record["api_execution"] != SSH_API
            or record["gpu_slots"] != [0] or record["budget"] != gpu.BUDGET or "fixed_reference_repair" in profile
            or request_timeouts(profile) != resilient.TIMEOUTS
            or any(type(proof.get(key)) is not type(value) or proof[key] != value for key, value in expected.items())
            or any(type(profile.get(key)) is not type(value) or profile[key] != value for key, value in settings.items())
            or profile.get("resume_boundaries") != {str(gpu.SEED): boundary}
            or Path(proof["previous_release_directory"]).name != gpu.CACHED_RELEASE
            or Path(proof["normal_release_directory"]).name != gpu.cached.OLD_RELEASE
            or any(signed_sources.get(key) != value or file_digest(ROOT / key) != value for key, value in fixed_sources.items())):
        raise PermissionError("the new Mac scope differs from the signed normal L685–L1485 contract")
    verify_local_source(info, root=ROOT)
    for source in (*gpu.SOURCES, *resilient.SOURCES, *gpu.normal.SOURCES):
        if signed_sources.get(source.as_posix()) != file_digest(ROOT / source):
            raise PermissionError(f"the installed continuation source differs from approval: {source.name}")
    if resilient._approved_prompt_sha(info) != first["prompt_sha256"]:
        raise PermissionError("the first request does not bind the unchanged normal A7 prompt")
    return proof


class ContinuationDispatcher(normal.ContinuationDispatcher):
    def _check_boundary(self, request, started):
        gpu.check_first_round(request, self.scope["r11_request"])
        logical = request["logical_index"]
        if logical:
            prior = [item for index in (0, 1) if (item := self._prior(request, logical - 1, index)) is not None]
            complete = [result for _, result in prior if result["outcome"]["status"] == "completed"]
            if len(complete) != 1 or complete[0]["raw_response"] is None:
                raise PermissionError("R11 correction requires the previous completed and delivered response")
        if started["attempt"] == 1:
            pair = self._prior(request, logical, 0)
            if pair is None:
                raise PermissionError("R11 physical retry requires its own delivered P0")
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
                raise PermissionError("R11 P1 requires the exact sealed no-answer relay failure")

    def validate(self, job):
        request, config, started = RelayRetryDispatcher.validate(self, job)
        if "fixed_reference_repair" in request["binding"] or started["attempt_id"] in {
            gpu.cached.ATTEMPT, gpu.HANDOFF["local_attempt_id"],
        }:
            raise PermissionError("the new ordinary continuation cannot replay the R10 paid or local attempt")
        return request, config, started


def run(args):
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.retry_exchange("info"), expected_contract=args.contract_sha256, expected_auth=args.authorization_sha256)
    spools = Path.home() / ".config/purellm/spool"
    if args.contract_sha256 in {gpu.cached.OLD.contract, gpu.CACHED.contract, resilient.PILOT_CONTRACT_SHA256}:
        raise PermissionError("the normal R11 route needs its own authorized Mac spool")
    archived_transport, identity = _archived_mac(spools, info["contract"]["predecessor"]["cached_selection"])
    proof = _scope(info, args.remote_run_dir, identity)
    pilot, _, pilot_transport = resilient._pool(spools / resilient.PILOT_CONTRACT_SHA256)
    transport = TransportConfig(**info["contract"]["profile"]["transport"])
    expected = replace(pilot_transport, timeout_seconds=resilient.TIMEOUTS.total_seconds, wire_mode=RESPONSES_STREAM_IPV4)
    if transport != expected or archived_transport != expected or not sdk_status(transport)["matches"]:
        raise PermissionError("the normal route must preserve Astra/xhigh and the approved Responses protocol")
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
