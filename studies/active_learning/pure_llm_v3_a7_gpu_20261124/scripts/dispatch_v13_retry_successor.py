"""Dispatch one separately authorized physical response request after verified failed waves."""

import argparse
from dataclasses import replace
import os
from pathlib import Path
from time import time

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import RESPONSES_STREAM_IPV4, TransportConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import read_bridge_json
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAL_WIRE_REPRESENTATION
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import run_dispatcher, verify_local_source
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, SSH_API, STUDY, request_timeouts
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import sdk_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import GPU_PYTHON, SSHCourier, verify_info
from . import dispatch_v13_resilient_sse as resilient
from . import run_v13_retry_successor as gpu
from .dispatch_v13_bounded_sse_recovery import _sha, _spool
from .dispatch_v13_relay_retry import RelayRetryDispatcher


def _scope(info, run, lane):
    record = info["contract"]
    profile, proof = record["profile"], record["predecessor"]
    path = Path(run)
    prior = lane.archive[-1]
    expected = {
        "dispatcher_module": gpu.DISPATCHER, "seed": lane.seed, "round": lane.round, "logical_index": 0,
        "previous_contract_sha256": prior.identity.contract, "previous_authorization_sha256": prior.identity.approval,
        "previous_run_directory": str(path.with_name(prior.identity.name)), "previous_request_id": prior.request_id,
        "payload_sha256": lane.payload_sha256, "previous_terminal_file_sha256": prior.terminal_file_sha256,
        "previous_transport_progress_file_sha256": prior.progress_file_sha256, "previous_transport_observation": prior.progress,
        "recovery_budget": gpu.RECOVERY_BUDGET, "execution_budget": lane.budget,
        "bridge_wait_seconds": resilient.TIMEOUTS.bridge_wait_seconds, "relay_not_before_unix": lane.not_before_unix,
        "relay_cooldown_clock": lane.cooldown_field, "newly_authorized_boundary_physical_attempts": 1,
    }
    if lane.fitted_prefix_files:
        expected["reviewed_fitted_prefix_files"] = dict(lane.fitted_prefix_files)
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
    if (path.name != lane.run_name or record["run_directory"] != run or record["api_execution"] != SSH_API
            or record["gpu_slots"] != [0] or record["budget"] != lane.budget or "fixed_reference_repair" in profile
            or request_timeouts(profile) != resilient.TIMEOUTS
            or any(type(proof.get(k)) is not type(v) or proof[k] != v for k, v in expected.items())
            or any(type(profile.get(k)) is not type(v) or profile[k] != v for k, v in settings.items())
            or set(boundaries) != {str(lane.seed)} or boundaries[str(lane.seed)].get("round") != lane.round
            or not isinstance(proof.get("previous_prefix_files"), dict)
            or digest(proof["previous_prefix_files"]) != proof.get("previous_prefix_sha256")
            or not all(_sha(proof.get(k)) for k in ("checkpoint_sha256", "packet_sha256", "wire_packet_sha256",
                                                  "base_messages_sha256", "previous_prefix_sha256"))):
        raise PermissionError("the successor does not match its reviewed archive, finite new slot and cooldown")
    verify_local_source(info, root=ROOT)
    sources = (*resilient.SOURCES, STUDY / "scripts" / "dispatch_v13_relay_retry.py",
               STUDY / "scripts" / "run_v13_relay_retry.py", STUDY / "scripts" / "run_v13_retry_successor.py",
               STUDY / "scripts" / "run_v13_12203_relay_retry.py", STUDY / "scripts" / "run_v13_157_relay_retry.py",
               STUDY / "scripts" / f"{gpu.DISPATCHER}.py")
    for source in sources:
        if record["source_sha256"].get(source.as_posix()) != file_digest(ROOT / source):
            raise PermissionError(f"the approved retry source differs: {source.name}")
    return proof


def _archive(spools, proof, lane):
    chain = proof["reviewed_retry_chain"]
    if type(chain) is not list or len(chain) != len(lane.archive) or chain[-1]["previous_attempts"] != proof["previous_attempts"]:
        raise PermissionError("the recovery requires all prior signed archives")
    previous = None
    run = Path(proof["previous_run_directory"])
    for review, item in zip(lane.archive, chain, strict=True):
        identity = review.identity
        expected = {"contract_sha256": identity.contract, "authorization_sha256": identity.approval,
                    "run_directory": str(run.with_name(identity.name)), "request_id": review.request_id,
                    "terminal_file_sha256": review.terminal_file_sha256, "progress_file_sha256": review.progress_file_sha256}
        if any(item.get(k) != value for k, value in expected.items()):
            raise PermissionError("a prior signed proof differs from its reviewed failed wave")
        spool = _spool(spools, identity.contract, identity.approval)
        attempts = review.attempts
        original, transport = resilient._attempts(spool, identity.contract, identity.approval, item["previous_attempts"],
                                                   lane.seed, lane.round, review.request_id, len(attempts))
        slots = spool / "slots"
        prefix = f"{lane.seed}-{lane.round}"
        if (set(slots.glob(f"{prefix}-*-*.json")) != {slots / f"{prefix}-0-{i}.json" for i in range(len(attempts))}
                or original["payload_sha256"] != lane.payload_sha256 or original["payload"].get("stream") is not True
                or read_bridge_json(spool / "provider_model_lock.json") != proof["model_identity"]
                or any(original["binding"][k] != proof[k] for k in ("checkpoint_sha256", "packet_sha256",
                                                                     "wire_packet_sha256", "base_messages_sha256"))
                or (previous is not None and not gpu.retry._compare_science(original, previous))):
            raise PermissionError("a delivered old failure, original input or absence of answer could not be verified")
        for prior, receipt in zip(attempts, item["previous_attempts"], strict=True):
            trace = spool / "attempts" / prior.attempt_id / "provider_progress.json"
            progress = read_bridge_json(trace)
            if (tuple(receipt[k] for k in ("attempt_id", "job_sha256", "result_sha256", "error",
                                          "provider_dispatched", "ambiguous"))
                    != (prior.attempt_id, prior.job_sha256, prior.result_sha256, prior.failure, True, True)
                    or file_digest(trace) != prior.progress_file_sha256
                    or any(progress.get(k) != v for k, v in prior.progress.items())
                    or any(k in progress for k in ("first_text_seconds", "complete_terminal_captured"))):
                raise PermissionError("a reviewed relay failure differs from the archived no-answer trace")
        previous = original
    return original, transport


class SuccessorDispatcher(RelayRetryDispatcher):
    def _check_boundary(self, request, started):
        lane = self.reviewed_scope
        if (started["attempt"] != 0 or time() < lane.not_before_unix
                or started["attempt_id"] in {old.attempt_id for review in lane.archive for old in review.attempts}):
            raise PermissionError("the reviewed round allows one distinct physical attempt after its last cooldown")
        return super()._check_boundary(request, started)


def run(args, lane):
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.retry_exchange("info"), expected_contract=args.contract_sha256,
                       expected_auth=args.authorization_sha256)
    proof = _scope(info, args.remote_run_dir, lane)
    if args.contract_sha256 in {resilient.PILOT_CONTRACT_SHA256, *(old.identity.contract for old in lane.archive)}:
        raise PermissionError("the new slot needs its own signed, isolated Mac spool")
    spools = Path.home() / ".config/purellm/spool"
    original, old_transport = _archive(spools, proof, lane)
    pilot, _, pilot_transport = resilient._pool(spools / resilient.PILOT_CONTRACT_SHA256)
    registered = replace(pilot_transport, timeout_seconds=resilient.TIMEOUTS.total_seconds, wire_mode=RESPONSES_STREAM_IPV4)
    transport = TransportConfig(**info["contract"]["profile"]["transport"])
    if (transport != registered or old_transport != registered
            or (transport.provider, transport.base_url, transport.requested_model, transport.served_model,
                transport.reasoning_effort, transport.logical_replies, transport.transport_attempts,
                transport.max_output_tokens, transport.sdk_version)
            != ("token4research", "https://token4research.cn/v1", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh", 3, 2,
                32768, "2.28.0") or not sdk_status(transport)["matches"]):
        raise PermissionError("the successor must retain the approved model and wire settings")
    if args.verify_only:
        return {"seed": lane.seed, "round": lane.round, "contract_sha256": args.contract_sha256,
                "verified_no_dispatch": True, "relay_not_before_unix": lane.not_before_unix}
    if time() < lane.not_before_unix:
        raise PermissionError("the original 180-second relay cooldown is still in force")
    spool = spools / args.contract_sha256
    dispatcher = SuccessorDispatcher(spool, info=info, contract_sha256=args.contract_sha256, transport=transport,
                                           mock=False, pool=resilient.SharedKeyPool(pilot.directory, pilot.config))
    dispatcher.scope, dispatcher.original, dispatcher.reviewed_scope = proof, original, lane
    dispatcher.prompt_sha = resilient._approved_prompt_sha(info)
    once(spool / "provider_model_lock.json", proof["model_identity"])
    return run_dispatcher(dispatcher, courier, watch=True, workers=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--execute-paid", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--wave", type=int, required=True)
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
    print(dumps(run(args, gpu.read_scope(parser, args))))


if __name__ == "__main__":
    main()
