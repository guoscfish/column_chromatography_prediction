"""Dispatch the single signed repair job for the rejected 12203 R5 answer."""

import argparse
from dataclasses import replace
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import RESPONSES_STREAM_IPV4, TransportConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import read_bridge_json
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAL_WIRE_REPRESENTATION
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import Dispatcher, run_dispatcher, verify_local_source
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, SSH_API, request_timeouts
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import sdk_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import GPU_PYTHON, SHA256, SSHCourier, verify_info
from . import dispatch_v13_resilient_sse as resilient
from . import dispatch_v13_retry_successor as successor_mac
from . import run_v13_12203_r5_targeted_repair as repair


def _archive(spools):
    parent = resilient._spool(spools, repair.PARENT.contract, repair.PARENT.approval)
    name = f"{repair.ATTEMPT}.json"
    job = read_bridge_json(parent / "jobs" / name)
    result = read_bridge_json(parent / "results" / name)
    original, old_transport, _, _ = repair._archive_job(job, result, FORMAL_WIRE_REPRESENTATION)
    slot = parent / "slots" / f"{repair.SEED}-{repair.ROUND}-0-0.json"
    if (set((parent / "slots").glob(f"{repair.SEED}-{repair.ROUND}-*-*.json")) != {slot}
            or read_bridge_json(slot) != {"attempt_id": repair.ATTEMPT, "job_sha256": repair.JOB}):
        raise PermissionError("the original R5 round has a second reserved Mac physical attempt")
    attempt = parent / "attempts" / repair.ATTEMPT
    raw = read_bridge_json(attempt / "raw_provider_response.json")
    capture = read_bridge_json(attempt / "provider_capture_receipt.json")
    identity = read_bridge_json(parent / "provider_model_lock.json")
    archive_files = ((parent / "jobs" / name, repair.ORIGINAL_JOB_FILE),
                     (parent / "results" / name, repair.ORIGINAL_RESULT_FILE),
                     (parent / "delivered" / name, repair.ORIGINAL_ACK_FILE),
                     (attempt / "raw_provider_response.json", repair.ORIGINAL_RAW_FILE),
                     (attempt / "provider_capture_receipt.json", repair.ORIGINAL_CAPTURE_FILE))
    if (any(file_digest(path) != sha for path, sha in archive_files)
            or read_bridge_json(parent / "delivered" / name) != {"attempt_id": repair.ATTEMPT, "result_sha256": repair.RESULT}
            or raw != result["raw_response"] or digest(raw) != repair.RAW
            or capture.get("attempt_id") != repair.ATTEMPT or capture.get("job_sha256") != repair.JOB
            or capture.get("metadata", {}).get("raw_response_sha256") != repair.RAW
            or identity != result["model_identity"]):
        raise PermissionError("the Mac archive lacks the exact completed response and delivery evidence")
    return original, old_transport, identity


def _scope(info, run, original, identity):
    record = info["contract"]
    profile, proof = record["profile"], record["predecessor"]
    expected_profile = {
        "arm": "A7", "memory": "M2", "device": "cuda", "initial_labels": 333, "target_labels": repair.TARGET,
        "batch_size": 32, "confirmation_seeds": [repair.SEED], "qualification_seed": repair.SEED,
        "requires_initial_qualification": False, "api_parallelism": 1, "store": False,
        "wire_representation": FORMAL_WIRE_REPRESENTATION, "provider_capability": "gated_ipv4",
        "ipv4_readiness_windows": 3, "ipv4_preflight_retry": True, "sse_inactivity_seconds": resilient.INACTIVITY,
        "fixed_reference_repair": repair._metadata(original),
    }
    expected_proof = {
        "dispatcher_module": repair.DISPATCHER, "seed": repair.SEED, "round": repair.ROUND, "logical_index": 0,
        "previous_run_directory": str(Path(run).with_name(repair.PARENT.name)),
        "previous_contract_sha256": repair.PARENT.contract, "previous_authorization_sha256": repair.PARENT.approval,
        "previous_request_id": repair.REQUEST, "previous_seed_pause_file_sha256": repair.OLD_PAUSE,
        "previous_terminal_file_sha256": repair.OLD_TERMINAL, "model_identity": identity,
        "previous_worker_file_sha256": repair.OLD_WORKER,
        "old_boundary_files": repair.BOUNDARY_FILES, "original_rejection_sha256": repair.OLD_DECISION_SHA256,
        "fixed_reference_repair_sha256": digest(expected_profile["fixed_reference_repair"]), "prior_response_reposted": False,
        "execution_budget": repair.BUDGET, "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 1},
        "bridge_wait_seconds": resilient.TIMEOUTS.bridge_wait_seconds,
        "original_reviewed_archive_sha256": repair.ARCHIVE_PROOF_SHA256,
    }
    boundaries = profile.get("resume_boundaries", {})
    if (Path(run).name != repair.RUN_NAME or record["run_directory"] != run or record["api_execution"] != SSH_API
            or record["gpu_slots"] != [0] or record["budget"] != repair.BUDGET
            or request_timeouts(profile) != resilient.TIMEOUTS or "response_headers_seconds" in profile
            or digest(proof.get("original_reviewed_archive")) != repair.ARCHIVE_PROOF_SHA256
            or any(type(profile.get(k)) is not type(v) or profile[k] != v for k, v in expected_profile.items())
            or any(type(proof.get(k)) is not type(v) or proof[k] != v for k, v in expected_proof.items())
            or {"audited_future_transport_replay", "sealed_relay_retry_seconds", "replay_eligible_from_round",
                "normalize_observed_reference_aliases"} & profile.keys()
            or set(boundaries) != {str(repair.SEED)} or boundaries[str(repair.SEED)].get("round") != repair.ROUND
            or boundaries[str(repair.SEED)].get("user_message_sha256") != digest(original["payload"]["input"][1])):
        raise PermissionError("the signed contract does not authorize this exact one-shot correction")
    verify_local_source(info, root=ROOT)
    for source in (*resilient.SOURCES, *repair.SOURCES):
        if record["source_sha256"].get(source.as_posix()) != file_digest(ROOT / source):
            raise PermissionError(f"the local correction source differs from its approved file: {source.name}")
    if resilient._approved_prompt_sha(info) != digest(repair.patch_system(original)):
        raise PermissionError("the prompt differs from the archived science with the approved correction guide")
    return expected_profile["fixed_reference_repair"]


class RepairDispatcher(Dispatcher):
    def validate(self, job):
        request, config, started = super().validate(job)
        if started["attempt_id"] == repair.ATTEMPT:
            raise PermissionError("the completed scientific request cannot be submitted again")
        repair.verify_patch_request(request, self.original, self.metadata, attempt=started["attempt"])
        return request, config, started


def run(args):
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.retry_exchange("info"), expected_contract=args.contract_sha256, expected_auth=args.authorization_sha256)
    if args.contract_sha256 in (repair.PARENT.contract, resilient.PILOT_CONTRACT_SHA256):
        raise PermissionError("the correction requires an independent signed spool")
    spools = Path.home() / ".config/purellm/spool"
    original, archived_transport, identity = _archive(spools)
    metadata = _scope(info, args.remote_run_dir, original, identity)
    local_record = spools.parent / "reviewed_recovery" / f"12203-72-{repair.OLD_REVIEWED_RECORD}.json"
    _, lane = repair._reviewed_scope(local_record)
    prior = info["contract"]["predecessor"]["original_reviewed_archive"]
    successor_mac._archive(spools, prior, lane)
    original_pool, _, pilot_transport = resilient._pool(spools / resilient.PILOT_CONTRACT_SHA256)
    approved = replace(pilot_transport, timeout_seconds=resilient.TIMEOUTS.total_seconds, wire_mode=RESPONSES_STREAM_IPV4)
    transport = TransportConfig(**info["contract"]["profile"]["transport"])
    if (archived_transport != approved or transport != replace(approved, logical_replies=1, transport_attempts=1)
            or not sdk_status(transport)["matches"]):
        raise PermissionError("the correction must retain the exact Astra/xhigh IPv4 model profile and SDK")
    if args.verify_only:
        return {"seed": repair.SEED, "round": repair.ROUND, "contract_sha256": args.contract_sha256,
                "verified_no_dispatch": True}
    spool = spools / args.contract_sha256
    dispatcher = RepairDispatcher(spool, info=info, contract_sha256=args.contract_sha256, transport=transport,
                                  mock=False, pool=resilient.SharedKeyPool(original_pool.directory, original_pool.config))
    dispatcher.original, dispatcher.metadata = original, metadata
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
        parser.error("provide both exact authorization hashes")
    os.umask(0o077)
    print(dumps(run(args)))


if __name__ == "__main__":
    main()
