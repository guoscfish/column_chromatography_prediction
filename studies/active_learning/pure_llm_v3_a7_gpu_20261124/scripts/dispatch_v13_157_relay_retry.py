"""Dispatch one new 157 R3 slot after the two exhausted relay failures."""

import argparse
from dataclasses import replace
import os
from pathlib import Path
from time import time

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import RESPONSES_STREAM_IPV4, TransportConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import read_bridge_json, validate_result
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAL_WIRE_REPRESENTATION
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import run_dispatcher, verify_local_source
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, SSH_API, STUDY, request_timeouts
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import sdk_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import GPU_PYTHON, SSHCourier, verify_info
from . import dispatch_v13_resilient_sse as resilient
from . import run_v13_157_relay_retry as gpu
from .dispatch_v13_bounded_sse_recovery import _sha, _spool
from .dispatch_v13_relay_retry import RelayRetryDispatcher


def _scope(info, run):
    record = info["contract"]
    profile, proof = record["profile"], record["predecessor"]
    path = Path(run)
    lane = gpu.LANE
    expected = {
        "dispatcher_module": gpu.DISPATCHER, "seed": lane.seed, "round": lane.round, "logical_index": 0,
        "previous_contract_sha256": gpu.OLD.contract, "previous_authorization_sha256": gpu.OLD.approval,
        "previous_run_directory": str(path.with_name(gpu.OLD.name)), "previous_request_id": gpu.REQUEST,
        "payload_sha256": gpu.PAYLOAD, "previous_terminal_file_sha256": [item.terminal_file_sha256 for item in gpu.PRIOR],
        "previous_transport_progress_file_sha256": [item.trace_sha256 for item in gpu.PRIOR],
        "previous_transport_observations": [item.observation for item in gpu.PRIOR],
        "previous_replay_files": gpu.REPLAY_FILES,
        "recovery_budget": gpu.RECOVERY_BUDGET, "execution_budget": gpu.execution_budget(),
        "bridge_wait_seconds": resilient.TIMEOUTS.bridge_wait_seconds, "relay_not_before_unix": gpu.NOT_BEFORE_UNIX,
        "prior_boundary_budget": {"physical_authorized": 2, "physical_already_used": 2, "physical_remaining": 0},
        "newly_authorized_boundary_physical_attempts": 1,
    }
    settings = {
        "provider_capability": "gated_ipv4", "arm": "A7", "memory": "M2", "device": "cuda",
        "initial_labels": 333, "target_labels": 1485, "batch_size": 32, "api_parallelism": 1,
        "store": False, "confirmation_seeds": [lane.seed], "qualification_seed": lane.seed,
        "requires_initial_qualification": False, "wire_representation": FORMAL_WIRE_REPRESENTATION,
        "sse_inactivity_seconds": 900, "ipv4_preflight_retry": True, "ipv4_readiness_windows": 3,
        "normalize_observed_reference_aliases": True, "audited_future_transport_replay": True,
        "replay_eligible_from_round": lane.round + 1, "sealed_relay_retry_seconds": gpu.retry.COOLDOWN_SECONDS,
    }
    boundaries = profile.get("resume_boundaries", {})
    if (path.name != gpu.RUN_NAME or record["run_directory"] != run or record["api_execution"] != SSH_API
            or record["gpu_slots"] != [0] or record["budget"] != gpu.execution_budget() or "fixed_reference_repair" in profile
            or request_timeouts(profile) != resilient.TIMEOUTS
            or any(type(proof.get(k)) is not type(v) or proof[k] != v for k, v in expected.items())
            or any(type(profile.get(k)) is not type(v) or profile[k] != v for k, v in settings.items())
            or set(boundaries) != {str(lane.seed)} or boundaries[str(lane.seed)].get("round") != lane.round
            or not isinstance(proof.get("previous_prefix_files"), dict)
            or digest(proof["previous_prefix_files"]) != proof.get("previous_prefix_sha256")
            or not all(_sha(proof.get(k)) for k in ("checkpoint_sha256", "packet_sha256", "wire_packet_sha256",
                                                  "base_messages_sha256", "previous_prefix_sha256"))):
        raise PermissionError("the session does not match one additional physical attempt and the exact original model and cooldown")
    verify_local_source(info, root=ROOT)
    sources = (*resilient.SOURCES, STUDY / "scripts" / "dispatch_v13_relay_retry.py",
               STUDY / "scripts" / "run_v13_relay_retry.py", STUDY / "scripts" / "run_v13_157_relay_retry.py",
               STUDY / "scripts" / f"{gpu.DISPATCHER}.py")
    for source in sources:
        if record["source_sha256"].get(source.as_posix()) != file_digest(ROOT / source):
            raise PermissionError(f"the approved 157 retry source differs: {source.name}")
    return proof


def _archive(spools, proof):
    lane = gpu.LANE
    spool = _spool(spools, gpu.OLD.contract, gpu.OLD.approval)
    original, transport = resilient._attempts(spool, gpu.OLD.contract, gpu.OLD.approval, proof["previous_attempts"],
                                               lane.seed, lane.round, gpu.REQUEST, 2)
    model = read_bridge_json(spool / "provider_model_lock.json")
    slots = spool / "slots"
    prefix = f"{lane.seed}-{lane.round}"
    if (set(slots.glob(f"{prefix}-*-*.json")) != {slots / f"{prefix}-0-{index}.json" for index in (0, 1)}
            or model != proof["model_identity"] or original["payload_sha256"] != gpu.PAYLOAD
            or original["payload"].get("stream") is not True
            or original["binding"].get("normalize_observed_reference_aliases") is not True
            or any(original["binding"][k] != proof[k] for k in ("checkpoint_sha256", "packet_sha256",
                                                               "wire_packet_sha256", "base_messages_sha256"))):
        raise PermissionError("the original two-attempt archive, model or scientific input differs")
    for prior, receipt in zip(gpu.PRIOR, proof["previous_attempts"], strict=True):
        filename = f"{prior.attempt_id}.json"
        trace = spool / "attempts" / prior.attempt_id / "provider_progress.json"
        progress = read_bridge_json(trace)
        job = read_bridge_json(spool / "jobs" / filename)
        result = validate_result(read_bridge_json(spool / "results" / filename), job)
        failure = result["outcome"]["failure"]
        evidence = failure.get("reply_identity") or {}
        expected = (prior.attempt_id, prior.job_sha256, prior.result_sha256, "response_id_mismatch", True, True)
        if (tuple(receipt[k] for k in ("attempt_id", "job_sha256", "result_sha256", "error", "provider_dispatched",
                                      "ambiguous")) != expected or file_digest(trace) != prior.trace_sha256
                or any(progress.get(k) != v for k, v in prior.observation.items())
                or any(k in progress for k in ("first_text_seconds", "complete_terminal_captured"))
                or evidence != {"provider_status": "failed", "provider_request_id": prior.provider_request_id,
                                "response_id": prior.observation["conflicting_response_id"],
                                "served_model": transport.served_model, "usage": {}}):
            raise PermissionError("one original delivery does not match its archived relay failure and no-output trace")
    return original, transport, model


class AdditionalRelayDispatcher(RelayRetryDispatcher):
    def _check_boundary(self, request, started):
        if (started["attempt"] != 0 or time() < gpu.NOT_BEFORE_UNIX
                or started["attempt_id"] in {prior.attempt_id for prior in gpu.PRIOR}):
            raise PermissionError("157 R3 authorizes one distinct physical attempt after its latest cooldown")
        return super()._check_boundary(request, started)


def run(args):
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.retry_exchange("info"), expected_contract=args.contract_sha256,
                       expected_auth=args.authorization_sha256)
    proof = _scope(info, args.remote_run_dir)
    if args.contract_sha256 in {gpu.OLD.contract, resilient.PILOT_CONTRACT_SHA256}:
        raise PermissionError("the additional slot needs its own signed, isolated Mac spool")
    spools = Path.home() / ".config/purellm/spool"
    original, old_transport, model = _archive(spools, proof)
    pilot, _, pilot_transport = resilient._pool(spools / resilient.PILOT_CONTRACT_SHA256)
    registered = replace(pilot_transport, timeout_seconds=resilient.TIMEOUTS.total_seconds, wire_mode=RESPONSES_STREAM_IPV4)
    transport = TransportConfig(**info["contract"]["profile"]["transport"])
    if (transport != registered or old_transport != registered
            or (transport.provider, transport.base_url, transport.requested_model, transport.served_model,
                transport.reasoning_effort, transport.logical_replies, transport.transport_attempts,
                transport.max_output_tokens, transport.sdk_version)
            != ("token4research", "https://token4research.cn/v1", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh", 3, 2,
                32768, "2.28.0") or not sdk_status(transport)["matches"]):
        raise PermissionError("the additional retry must retain the approved model and wire settings")
    if args.verify_only:
        return {"seed": gpu.LANE.seed, "round": gpu.LANE.round, "contract_sha256": args.contract_sha256,
                "verified_no_dispatch": True, "relay_not_before_unix": gpu.NOT_BEFORE_UNIX}
    if time() < gpu.NOT_BEFORE_UNIX:
        raise PermissionError("the original 180-second relay cooldown is still in force")
    spool = spools / args.contract_sha256
    dispatcher = AdditionalRelayDispatcher(spool, info=info, contract_sha256=args.contract_sha256, transport=transport,
                                           mock=False, pool=resilient.SharedKeyPool(pilot.directory, pilot.config))
    dispatcher.scope, dispatcher.original = proof, original
    dispatcher.prompt_sha = resilient._approved_prompt_sha(info)
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
    args = parser.parse_args()
    if not all(_sha(value) for value in (args.contract_sha256, args.authorization_sha256)):
        parser.error("exact contract and approval hashes required")
    os.umask(0o077)
    print(dumps(run(args)))


if __name__ == "__main__":
    main()
