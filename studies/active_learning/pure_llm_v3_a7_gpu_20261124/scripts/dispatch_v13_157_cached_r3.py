"""Use provider transport only for future rounds after the verified 157 cached import."""

import argparse
from dataclasses import replace
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import RESPONSES_STREAM_IPV4, TransportConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import read_bridge_json, validate_result, verify_job
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import run_dispatcher, verify_local_source
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, SSH_API, STUDY, request_timeouts
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import sdk_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import GPU_PYTHON, SHA256, SSHCourier, verify_info
from . import dispatch_v13_resilient_sse as resilient
from . import run_v13_157_cached_r3 as gpu
from .dispatch_v13_bounded_sse_recovery import _spool
from .dispatch_v13_relay_retry import RelayRetryDispatcher


def _scope(info, run):
    record = info["contract"]
    profile, proof = record["profile"], record["predecessor"]
    scope = proof["cached_import"]
    expected = {
        "attempt_id": gpu.ATTEMPT, "job_sha256": gpu.JOB_SHA, "result_sha256": gpu.RESULT_SHA,
        "raw_response_sha256": gpu.RAW_SHA, "final_text_sha256": gpu.FINAL_SHA, "normalized_plan_sha256": gpu.PLAN_SHA,
        "ordered_candidate_ids_sha256": gpu.ORDER_SHA, "role_field_projection": gpu.ROLE_PROJECTION,
        "archived_terminal_file_sha256": gpu.TERMINAL_SHA, "archived_decision_file_sha256": gpu.DECISION_SHA,
        "archive_boundary_sha256": gpu.BOUNDARY_FILES, "provider_post": False,
    }
    if (Path(run).name != gpu.RUN_NAME or record["run_directory"] != run or record["api_execution"] != SSH_API
            or record["gpu_slots"] != [0] or record["budget"] != gpu.BUDGET or proof["recovery_budget"] != gpu.IMPORT_BUDGET
            or proof["dispatcher_module"] != gpu.DISPATCHER or proof["previous_run_directory"] != str(Path(run).with_name(gpu.OLD.name))
            or (proof["previous_contract_sha256"], proof["previous_authorization_sha256"], proof["previous_request_id"])
            != (gpu.OLD.contract, gpu.OLD.approval, gpu.REQUEST)
            or (proof["seed"], proof["round"], proof["payload_sha256"]) != (gpu.SEED, gpu.ROUND, gpu.PAYLOAD_SHA)
            or (profile["confirmation_seeds"], profile["target_labels"], profile["replay_eligible_from_round"])
            != ([gpu.SEED], 1485, gpu.ROUND + 1)
            or profile["resume_boundaries"][str(gpu.SEED)]["round"] != gpu.ROUND
            or (profile["arm"], profile["memory"], profile["device"], profile["batch_size"]) != ("A7", "M2", "cuda", 32)
            or profile.get("audited_future_transport_replay") is not True or profile["sealed_relay_retry_seconds"] != 180
            or request_timeouts(profile) != resilient.TIMEOUTS or proof["bridge_wait_seconds"] != resilient.TIMEOUTS.bridge_wait_seconds
            or any(type(scope.get(key)) is not type(value) or scope[key] != value for key, value in expected.items())
            or digest(scope["removed_paths"]) != gpu.PATHS_SHA
            or record["source_sha256"].get(gpu.SEED_RUNNER_FILE) != gpu.SEED_RUNNER_SHA):
        raise PermissionError("this scope may dispatch only after the exact zero-POST R3 cached decision")
    verify_local_source(info, root=ROOT)
    sources = (*resilient.SOURCES, STUDY / "scripts" / "dispatch_v13_relay_retry.py",
               STUDY / "scripts" / "run_v13_relay_retry.py",
               STUDY / "scripts" / "run_v13_157_cached_r3.py", STUDY / "scripts" / f"{gpu.DISPATCHER}.py")
    for path in sources:
        if record["source_sha256"].get(path.as_posix()) != file_digest(ROOT / path):
            raise PermissionError(f"the approved cached continuation source differs: {path.name}")
    return proof


def _archive(spools, proof):
    spool = _spool(spools, gpu.OLD.contract, gpu.OLD.approval)
    name = f"{gpu.ATTEMPT}.json"
    job = read_bridge_json(spool / "jobs" / name)
    original, transport, started = verify_job(job, contract_sha256=gpu.OLD.contract, authorization_sha256=gpu.OLD.approval)
    result = validate_result(read_bridge_json(spool / "results" / name), job)
    ack = read_bridge_json(spool / "delivered" / name)
    raw = read_bridge_json(spool / "attempts" / gpu.ATTEMPT / "raw_provider_response.json")
    slot_dir = spool / "slots"
    prefix = f"{gpu.SEED}-{gpu.ROUND}"
    if ((job["attempt_id"], original["request_id"], original["payload_sha256"], job["job_sha256"], result["result_sha256"],
         result["raw_response_sha256"]) != (gpu.ATTEMPT, gpu.REQUEST, gpu.PAYLOAD_SHA, gpu.JOB_SHA, gpu.RESULT_SHA, gpu.RAW_SHA)
            or (original["binding"]["seed"], original["binding"]["round"], original["logical_index"], started["attempt"])
            != (gpu.SEED, gpu.ROUND, 0, 0)
            or any(original["binding"][key] != proof[key] for key in ("checkpoint_sha256", "packet_sha256",
                                                                      "wire_packet_sha256", "base_messages_sha256"))
            or result["outcome"]["status"] != "completed" or result["provider_dispatched"] is not True
            or result["model_identity"] != proof["model_identity"]
            or read_bridge_json(spool / "provider_model_lock.json") != proof["model_identity"]
            or result["raw_response"] != raw or ack != {"attempt_id": gpu.ATTEMPT, "result_sha256": gpu.RESULT_SHA}
            or set(slot_dir.glob(f"{prefix}-*-*.json")) != {slot_dir / f"{prefix}-0-0.json"}):
        raise PermissionError("the actual completed original reply or its single prior reservation differs")
    gpu._normalized(raw)
    return transport


class FutureDispatcher(RelayRetryDispatcher):
    def _check_boundary(self, request, started):
        raise PermissionError("R3 is a local cached decision and cannot be sent to the provider")


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
            or (transport.requested_model, transport.served_model, transport.reasoning_effort, transport.sdk_version)
            != ("gpt-6.2-astra", "gpt-6.2-astra", "xhigh", "2.28.0") or not sdk_status(transport)["matches"]):
        raise PermissionError("future rounds require the original provider protocol and a separate private spool")
    if args.verify_only:
        return {"seed": gpu.SEED, "round": gpu.ROUND, "verified_no_dispatch": True, "future_starts_at_round": gpu.ROUND + 1}
    spool = spools / args.contract_sha256
    dispatcher = FutureDispatcher(spool, info=info, contract_sha256=args.contract_sha256, transport=transport,
                                  mock=False, pool=resilient.SharedKeyPool(pilot.directory, pilot.config))
    dispatcher.scope = proof
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
    args = parser.parse_args()
    if not all(SHA256.fullmatch(value) for value in (args.contract_sha256, args.authorization_sha256)):
        parser.error("provide the exact contract and authorization hashes")
    os.umask(0o077)
    print(dumps(run(args)))


if __name__ == "__main__":
    main()
