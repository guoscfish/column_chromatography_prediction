"""Dispatch only the approved normal 887 continuation anchored to the archived R19 fit."""

import argparse
from dataclasses import replace
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import parse_response
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import RESPONSES_STREAM_IPV4, TransportConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import read_bridge_json, validate_result, verify_job
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAL_WIRE_REPRESENTATION
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import run_dispatcher, verify_local_source
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, SSH_API, request_timeouts
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import _reply_from_raw, sdk_status, verify_reported_model
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import GPU_PYTHON, SHA256, SSHCourier, verify_info
from . import dispatch_v13_resilient_sse as resilient
from . import run_v13_887_r20_continuation as gpu
from .dispatch_v13_relay_retry import RelayRetryDispatcher


def _archived_mac(spools, custody):
    old = gpu.cached
    spool = resilient._spool(spools, old.OLD.contract, old.OLD.approval)
    name = f"{old.ATTEMPT}.json"
    slot = spool / f"slots/{gpu.SEED}-19-0-0.json"
    job = read_bridge_json(spool / "jobs" / name)
    request, transport, started = verify_job(job, contract_sha256=old.OLD.contract,
                                              authorization_sha256=old.OLD.approval,
                                              wire_representation=FORMAL_WIRE_REPRESENTATION)
    result = validate_result(read_bridge_json(spool / "results" / name), job)
    attempt = spool / "attempts" / old.ATTEMPT
    raw = read_bridge_json(attempt / "raw_provider_response.json")
    capture = read_bridge_json(attempt / "provider_capture_receipt.json")
    identity = read_bridge_json(spool / "provider_model_lock.json")
    if (set((spool / "slots").glob(f"{gpu.SEED}-19-*-*.json")) != {slot}
            or read_bridge_json(slot) != {"attempt_id": old.ATTEMPT, "job_sha256": old.JOB}
            or (request["request_id"], request["payload_sha256"], job["job_sha256"], started["attempt_id"])
            != (old.REQUEST, old.PAYLOAD, old.JOB, old.ATTEMPT)
            or result["outcome"] != {"status": "failed", "failure": old.FAILURE}
            or result["model_identity"] is not None or result["provider_dispatched"] is not True
            or (result["result_sha256"], digest(raw), result["raw_response_sha256"]) != (old.RESULT, old.RAW, old.RAW)
            or raw != result["raw_response"] or digest(capture) != old.CAPTURE
            or file_digest(spool / "delivered" / name) != old.ACK
            or file_digest(attempt / "provider_progress.json") != old.PROGRESS
            or read_bridge_json(spool / "delivered" / name) != {"attempt_id": old.ATTEMPT, "result_sha256": old.RESULT}
            or any(file_digest(spool / name.replace("bridge/outbox/", "jobs/").replace("bridge/results/", "results/")) != sha
                   for name, sha in old.ARCHIVE_FILES.items() if name.startswith(("bridge/outbox/", "bridge/results/")))
            or not isinstance(custody, dict) or custody.get("provider_post") is not False
            or digest(custody) != gpu.HANDOFF["cached_selection_sha256"]):
        raise PermissionError("the immutable Mac archive differs from the signed R19 cached handoff")
    declared = {"model_version": raw.get("model_version"), "reported_reasoning_effort": raw["reasoning"]["effort"]}
    reply = _reply_from_raw(raw, old.PROVIDER_REQUEST)
    plan = parse_response(reply.text)
    if ((reply.response_id, digest(reply.text), reply.usage) != (old.RESPONSE, old.TEXT, old.USAGE)
            or verify_reported_model(transport, reply, declared) != identity or digest(plan) != old.PLAN
            or digest([choice["candidate_id"] for choice in plan["choices"]]) != old.ORDER):
        raise PermissionError("the reviewed parser does not reproduce the exact archived R19 final")
    return transport, identity


def _scope(info, run, identity):
    record = info["contract"]
    profile, proof = record["profile"], record["predecessor"]
    path = Path(run)
    expected = {
        "dispatcher_module": gpu.DISPATCHER, "seed": gpu.SEED, "round": gpu.START,
        "parent_run_directory": str(path.with_name(gpu.cached.OLD.name)),
        "parent_contract_sha256": gpu.cached.OLD.contract, "parent_authorization_sha256": gpu.cached.OLD.approval,
        "parent_seed_pause_sha256": gpu.cached.ARCHIVE_FILES["seed_887/pause.request"],
        "extension_run_directory": str(path.with_name(gpu.CACHED.name)), "r19_handoff": gpu.HANDOFF,
        "cached_response_import_file_sha256": gpu.CUSTODY_FILE, "r20_request": gpu.R20_REQUEST,
        "checkpoint_sha256": gpu.HANDOFF["checkpoint_sha256"], "model_identity": identity,
        "execution_budget": gpu.BUDGET, "recovery_budget": gpu.BOUNDARY_BUDGET,
        "bridge_wait_seconds": resilient.TIMEOUTS.bridge_wait_seconds,
    }
    settings = {
        "arm": "A7", "memory": "M2", "device": "cuda", "initial_labels": 333, "target_labels": gpu.TARGET, "batch_size": 32,
        "confirmation_seeds": [gpu.SEED], "qualification_seed": gpu.SEED, "requires_initial_qualification": False,
        "api_parallelism": 1, "store": False, "wire_representation": FORMAL_WIRE_REPRESENTATION,
        "provider_capability": "gated_ipv4", "ipv4_readiness_windows": 3, "ipv4_preflight_retry": True,
        "sse_inactivity_seconds": resilient.INACTIVITY, "normalize_observed_reference_aliases": True,
        "audited_future_transport_replay": True, "sealed_relay_retry_seconds": 180, "replay_eligible_from_round": gpu.START,
    }
    boundary = {"round": gpu.START, "state_sha256": gpu.HANDOFF["state_sha256"],
                "memory_sha256": gpu.HANDOFF["scientific_memory_sha256"]}
    if (path.name != gpu.RUN_NAME or record["run_directory"] != run or record["api_execution"] != SSH_API
            or record["gpu_slots"] != [0] or record["budget"] != gpu.BUDGET or "fixed_reference_repair" in profile
            or request_timeouts(profile) != resilient.TIMEOUTS
            or any(type(proof.get(k)) is not type(v) or proof[k] != v for k, v in expected.items())
            or any(type(profile.get(k)) is not type(v) or profile[k] != v for k, v in settings.items())
            or profile.get("resume_boundaries") != {str(gpu.SEED): boundary}
            or Path(proof.get("parent_release_directory", "")).name != gpu.cached.OLD_RELEASE
            or Path(proof.get("extension_release_directory", "")).name != gpu.CACHED_RELEASE
            or record["source_sha256"].get(gpu.PARSER) != gpu.NEW_PARSER
            or record["source_sha256"].get(gpu.CACHED_MODULE) != gpu.CACHED_HELPER
            or record["source_sha256"].get(gpu.seed_runner.PROMPT_PATH.as_posix()) != gpu.NORMAL_PROMPT):
        raise PermissionError("the signed scope is not the reviewed normal L973–L1485 continuation")
    verify_local_source(info, root=ROOT)
    for source in (*resilient.SOURCES, *gpu.SOURCES, Path(gpu.CACHED_MODULE)):
        if record["source_sha256"].get(source.as_posix()) != file_digest(ROOT / source):
            raise PermissionError(f"continuation source is outside its signed manifest: {source.name}")
    if resilient._approved_prompt_sha(info) != gpu.R20_REQUEST["prompt_sha256"]:
        raise PermissionError("the normal scientific Prompt differs from the signed input")
    return proof


class ContinuationDispatcher(RelayRetryDispatcher):
    def validate(self, job):
        request, config, started = super().validate(job)
        if "fixed_reference_repair" in request["binding"] or started["attempt_id"] in {
            gpu.cached.ATTEMPT, gpu.HANDOFF["local_attempt_id"],
        }:
            raise PermissionError("the normal continuation cannot replay the archived R19 selection")
        return request, config, started

    def _check_boundary(self, request, started):
        if request["logical_index"] != 0 or gpu.request_identity(request) != self.scope["r20_request"]:
            raise PermissionError("R20 allows one logical decision from the exact signed L973 request")
        if started["attempt"] == 1:
            self._check_boundary_retry(request, started)


def run(args):
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.retry_exchange("info"), expected_contract=args.contract_sha256, expected_auth=args.authorization_sha256)
    spools = Path.home() / ".config/purellm/spool"
    archived_transport, identity = _archived_mac(spools, info["contract"]["predecessor"].get("cached_selection"))
    proof = _scope(info, args.remote_run_dir, identity)
    pilot, _, pilot_transport = resilient._pool(spools / resilient.PILOT_CONTRACT_SHA256)
    transport = TransportConfig(**info["contract"]["profile"]["transport"])
    expected = replace(pilot_transport, timeout_seconds=resilient.TIMEOUTS.total_seconds, wire_mode=RESPONSES_STREAM_IPV4)
    if transport != expected or archived_transport != expected or not sdk_status(transport)["matches"]:
        raise PermissionError("the signed continuation must preserve the original Astra/xhigh transport profile")
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
