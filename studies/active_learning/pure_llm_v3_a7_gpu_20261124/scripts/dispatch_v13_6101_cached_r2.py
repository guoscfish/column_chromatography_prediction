"""Dispatch seed 6101 future rounds only after its archived R2 answer fits L429."""

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
from time import monotonic, sleep

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import RESPONSES_STREAM_IPV4, TransportConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import read_bridge_json, validate_result, verify_job
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAL_WIRE_REPRESENTATION
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import run_dispatcher, verify_local_source
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, SSH_API, STUDY, request_timeouts
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import sdk_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import CourierUnavailable, GPU_PYTHON, SHA256, SSHCourier, verify_info
from . import dispatch_v13_resilient_sse as resilient
from . import run_v13_6101_cached_r2 as gpu
from .dispatch_v13_bounded_sse_recovery import _spool
from .dispatch_v13_relay_retry import RelayRetryDispatcher


HANDOFF_CHECK = """\
import sys
from pathlib import Path
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import STUDY, verify_authorization
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.run_prepared_five_seed import PreparedCohortContract

run, contract_sha, authorization_sha = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
record, approval = read(run / "formal_contract.json"), read(run / "authorization.json")
source = STUDY / "scripts/run_v13_6101_cached_r2.py"
if (digest(record), digest(approval), record.get("run_directory")) != (contract_sha, authorization_sha, str(run)):
    raise PermissionError("the GPU handover belongs to a different signed run")
if file_digest(source) != record["source_sha256"].get(source.as_posix()):
    raise PermissionError("the GPU handover verifier differs from its signed source")
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_6101_cached_r2 as gpu

contract = PreparedCohortContract(record["profile"], TransportConfig(**record["profile"]["transport"]),
    record["source_sha256"], record["asset_sha256"], record["anchor_root"], tuple(record["gpu_slots"]), record["lease_root"],
    str(run), record["api_execution"], record["predecessor"])
if contract.record() != record:
    raise PermissionError("the GPU handover contract could not be reproduced")
verify_authorization(contract, approval)
report = gpu.verify_handoff(run, contract, approval)
binding = {"contract_sha256": contract_sha, "authorization_sha256": authorization_sha}
if report is None:
    print(dumps({**binding, "status": "pending"}))
else:
    print(dumps({**binding, **report, "status": "complete", "seed": gpu.SEED,
                 "cached_import_sha256": digest(contract.predecessor["cached_import"])}))
"""


def _scope(info, run):
    record = info["contract"]
    profile, proof = record["profile"], record["predecessor"]
    cached = proof["cached_import"]
    expected = {
        "attempt_id": gpu.ATTEMPT, "job_sha256": gpu.JOB_SHA, "result_sha256": gpu.RESULT_SHA,
        "raw_response_sha256": gpu.RAW_SHA, "final_text_sha256": gpu.FINAL_SHA, "normalized_plan_sha256": gpu.PLAN_SHA,
        "ordered_candidate_ids_sha256": gpu.ORDER_SHA, "archived_terminal_file_sha256": gpu.TERMINAL_SHA,
        "archived_decision_file_sha256": gpu.DECISION_SHA, "archive_boundary_sha256": gpu.BOUNDARY_FILES, "provider_post": False,
    }
    settings = {"confirmation_seeds": [gpu.SEED], "target_labels": 1485, "replay_eligible_from_round": gpu.ROUND + 1,
                "arm": "A7", "memory": "M2", "device": "cuda", "batch_size": 32, "api_parallelism": 1,
                "requires_initial_qualification": False, "store": False, "audited_future_transport_replay": True,
                "normalize_observed_reference_aliases": True, "sealed_relay_retry_seconds": 180,
                "wire_representation": FORMAL_WIRE_REPRESENTATION,
                "ipv4_preflight_retry": True, "ipv4_readiness_windows": 3, "sse_inactivity_seconds": resilient.INACTIVITY}
    prefix = cached.get("confirmed_boundary_prefix", {})
    if (Path(run).name != gpu.RUN_NAME or record["run_directory"] != run or record["api_execution"] != SSH_API
            or record["gpu_slots"] != [0] or record["budget"] != gpu.BUDGET or proof["recovery_budget"] != gpu.IMPORT_BUDGET
            or proof["dispatcher_module"] != gpu.DISPATCHER or proof["previous_run_directory"] != str(Path(run).with_name(gpu.OLD.name))
            or (proof["previous_contract_sha256"], proof["previous_authorization_sha256"], proof["previous_request_id"])
            != (gpu.OLD.contract, gpu.OLD.approval, gpu.REQUEST)
            or (proof["seed"], proof["round"], proof["payload_sha256"]) != (gpu.SEED, gpu.ROUND, gpu.PAYLOAD_SHA)
            or any(type(profile.get(key)) is not type(value) or profile[key] != value for key, value in settings.items())
            or profile["resume_boundaries"][str(gpu.SEED)]["round"] != gpu.ROUND or "fixed_reference_repair" in profile
            or request_timeouts(profile) != resilient.TIMEOUTS or proof["bridge_wait_seconds"] != resilient.TIMEOUTS.bridge_wait_seconds
            or set(cached) != {*expected, "removed_paths", "confirmed_boundary_prefix"}
            or any(type(cached.get(key)) is not type(value) or cached[key] != value for key, value in expected.items())
            or type(cached.get("removed_paths")) is not list or len(cached["removed_paths"]) != 74
            or digest(cached["removed_paths"]) != gpu.PATHS_SHA or type(prefix) is not dict or not prefix
            or any(not resilient._sha(value) for value in prefix.values())
            or record["source_sha256"].get(gpu.SEED_RUNNER_FILE) != gpu.SEED_RUNNER_SHA):
        raise PermissionError("this scope may dispatch only after the exact no-POST R2 cache and fit")
    verify_local_source(info, root=ROOT)
    sources = (*resilient.SOURCES, STUDY / "scripts" / "dispatch_v13_relay_retry.py",
               STUDY / "scripts" / "run_v13_relay_retry.py",
               STUDY / "scripts" / "run_v13_6101_cached_r2.py", STUDY / "scripts" / f"{gpu.DISPATCHER}.py")
    for path in sources:
        if record["source_sha256"].get(path.as_posix()) != file_digest(ROOT / path):
            raise PermissionError(f"the approved cached continuation source differs: {path.name}")
    return proof


def _archive(spools, proof):
    spool = _spool(spools, gpu.OLD.contract, gpu.OLD.approval)
    name = f"{gpu.ATTEMPT}.json"
    job = read_bridge_json(spool / "jobs" / name)
    original, transport, started = verify_job(job, contract_sha256=gpu.OLD.contract, authorization_sha256=gpu.OLD.approval,
                                                wire_representation=FORMAL_WIRE_REPRESENTATION)
    result = validate_result(read_bridge_json(spool / "results" / name), job)
    reply = result["outcome"].get("reply", {})
    ack = read_bridge_json(spool / "delivered" / name)
    attempt = spool / "attempts" / gpu.ATTEMPT
    raw = read_bridge_json(attempt / "raw_provider_response.json")
    capture = read_bridge_json(attempt / "provider_capture_receipt.json")
    slot_dir = spool / "slots"
    prefix = f"{gpu.SEED}-{gpu.ROUND}"
    slot = slot_dir / f"{prefix}-0-0.json"
    if ((job["attempt_id"], original["request_id"], original["payload_sha256"], job["job_sha256"], result["result_sha256"],
         result["raw_response_sha256"]) != (gpu.ATTEMPT, gpu.REQUEST, gpu.PAYLOAD_SHA, gpu.JOB_SHA, gpu.RESULT_SHA, gpu.RAW_SHA)
            or (original["binding"]["seed"], original["binding"]["round"], original["logical_index"], started["attempt"])
            != (gpu.SEED, gpu.ROUND, 0, 0)
            or any(original["binding"][key] != proof[key] for key in ("checkpoint_sha256", "packet_sha256",
                                                                      "wire_packet_sha256", "base_messages_sha256"))
            or result["outcome"]["status"] != "completed" or result["provider_dispatched"] is not True
            or reply.get("response_id") != gpu.RESPONSE_ID or digest(reply.get("text")) != gpu.FINAL_SHA
            or result["model_identity"] != proof["model_identity"]
            or read_bridge_json(spool / "provider_model_lock.json") != proof["model_identity"]
            or result["raw_response"] != raw or digest(raw) != gpu.RAW_SHA
            or capture.get("attempt_id") != gpu.ATTEMPT or capture.get("job_sha256") != gpu.JOB_SHA
            or capture.get("metadata", {}).get("raw_response_sha256") != gpu.RAW_SHA
            or ack != {"attempt_id": gpu.ATTEMPT, "result_sha256": gpu.RESULT_SHA}
            or read_bridge_json(slot) != {"attempt_id": gpu.ATTEMPT, "job_sha256": gpu.JOB_SHA}
            or set(slot_dir.glob(f"{prefix}-*-*.json")) != {slot}):
        raise PermissionError("the original completed raw reply, capture, reservation or acknowledgment differs")
    gpu._normalized(raw)
    return transport


def _read_handoff(courier, info):
    argv = [courier.remote_python, "-B", "-c", HANDOFF_CHECK, courier.remote_run_directory,
            info["contract_sha256"], info["authorization_sha256"]]
    command = f"cd {shlex.quote(courier.remote_root)} && exec {shlex.join(argv)}"
    ssh = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=yes",
           "--", courier.host, command]
    try:
        response = subprocess.run(ssh, input="", capture_output=True, text=True, check=True, timeout=courier.timeout)
    except subprocess.CalledProcessError as error:
        if error.returncode == 255:
            raise CourierUnavailable("the L429 GPU handover read did not complete") from None
        raise PermissionError("the authenticated GPU refused the L429 handover check") from None
    except (OSError, subprocess.SubprocessError):
        raise CourierUnavailable("the L429 GPU handover read did not complete") from None
    try:
        return json.loads(response.stdout)
    except (TypeError, json.JSONDecodeError):
        raise CourierUnavailable("the L429 GPU handover did not return authenticated JSON") from None


def _verified_checkpoint(handoff, info, proof):
    expected = {"contract_sha256": info["contract_sha256"], "authorization_sha256": info["authorization_sha256"],
                "seed": gpu.SEED, "round": gpu.ROUND + 1, "active_labels": 429,
                "cached_import_sha256": digest(proof["cached_import"]), "provider_post": False}
    if type(handoff) is dict and handoff == {"status": "pending", "contract_sha256": info["contract_sha256"],
                                            "authorization_sha256": info["authorization_sha256"]}:
        return None
    if (type(handoff) is not dict or handoff.get("status") != "complete"
            or any(type(handoff.get(k)) is not type(value) or handoff[k] != value for k, value in expected.items())
            or not resilient._sha(handoff.get("checkpoint_sha256")) or not resilient._sha(handoff.get("transition_file_sha256"))):
        raise PermissionError("the GPU handover does not prove the exact no-POST L429 checkpoint")
    return handoff["checkpoint_sha256"]


def _wait_handoff(courier, info, proof, wait):
    deadline = monotonic() + wait
    while True:
        try:
            checkpoint = _verified_checkpoint(_read_handoff(courier, info), info, proof)
            if checkpoint is not None:
                return checkpoint
        except CourierUnavailable:
            if wait == 0:
                raise
        if wait == 0:
            return None
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise CourierUnavailable("no verified L429 handover before the finite local wait expired")
        sleep(min(3, remaining))


class FutureDispatcher(RelayRetryDispatcher):
    def _check_boundary(self, request, started):
        raise PermissionError("R2 is a local cached decision and cannot be sent to the provider")

    def validate(self, job):
        request, transport, started = super().validate(job)
        binding = request["binding"]
        if (binding["round"] == gpu.ROUND + 1 and binding["checkpoint_sha256"] != self.handoff_checkpoint
                or started["attempt_id"] == gpu.ATTEMPT or request["request_id"] == gpu.REQUEST):
            raise PermissionError("future R3 must descend from the verified L429 checkpoint")
        return request, transport, started


def run(args):
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.retry_exchange("info"), expected_contract=args.contract_sha256,
                       expected_auth=args.authorization_sha256)
    proof = _scope(info, args.remote_run_dir)
    spools = Path.home() / ".config/purellm/spool"
    archived = _archive(spools, proof)
    pilot, _, base = resilient._pool(spools / resilient.PILOT_CONTRACT_SHA256)
    transport = TransportConfig(**info["contract"]["profile"]["transport"])
    if (args.contract_sha256 in {gpu.OLD.contract, resilient.PILOT_CONTRACT_SHA256} or archived != transport
            or transport != replace(base, timeout_seconds=resilient.TIMEOUTS.total_seconds, wire_mode=RESPONSES_STREAM_IPV4)
            or (transport.requested_model, transport.served_model, transport.reasoning_effort, transport.sdk_version,
                transport.logical_replies, transport.transport_attempts) != ("gpt-6.2-astra", "gpt-6.2-astra", "xhigh", "2.28.0", 3, 2)
            or not sdk_status(transport)["matches"]):
        raise PermissionError("future rounds require the original model and the signed separate spool")
    checkpoint = _wait_handoff(courier, info, proof, 0 if args.verify_only else args.handoff_wait)
    if args.verify_only:
        return {"seed": gpu.SEED, "round": gpu.ROUND, "verified_no_dispatch": True,
                "future_starts_at_round": gpu.ROUND + 1, "l429_handoff_verified": checkpoint is not None}
    if checkpoint is None:
        raise PermissionError("future provider work requires a verified L429 checkpoint")
    spool = spools / args.contract_sha256
    dispatcher = FutureDispatcher(spool, info=info, contract_sha256=args.contract_sha256, transport=transport,
                                  mock=False, pool=resilient.SharedKeyPool(pilot.directory, pilot.config))
    dispatcher.scope = proof
    dispatcher.handoff_checkpoint = checkpoint
    dispatcher.prompt_sha = resilient._approved_prompt_sha(info)
    once(spool / "provider_model_lock.json", proof["model_identity"])
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
    parser.add_argument("--handoff-wait", type=float, default=1800)
    args = parser.parse_args()
    if (not all(SHA256.fullmatch(value) for value in (args.contract_sha256, args.authorization_sha256))
            or not math.isfinite(args.handoff_wait) or not 1 <= args.handoff_wait <= 7800):
        parser.error("provide exact authorization hashes and a finite L429 handover wait")
    os.umask(0o077)
    print(dumps(run(args)))


if __name__ == "__main__":
    main()
