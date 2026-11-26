"""Dispatch the signed, bounded continuation of a confirmed relay failure."""

import argparse
from dataclasses import replace
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_STREAM_IPV4, TransportConfig, is_sealed_relay_failure,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import read_bridge_json, validate_result, verify_job
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAL_WIRE_REPRESENTATION
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import Dispatcher, run_dispatcher, verify_local_source
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, SSH_API, STUDY, request_timeouts
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import sdk_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import GPU_PYTHON, SSHCourier, verify_info
from . import dispatch_v13_resilient_sse as resilient
from .dispatch_v13_bounded_sse_recovery import _sha, _science, _spool
from .run_v13_relay_retry import DISPATCHER, LANES


def _scope(info, run, lane):
    record = info["contract"]
    profile, proof = record["profile"], record["predecessor"]
    path = Path(run)
    future = 35 - lane.round
    budget = {
        "first_logical_requests": 1 + future,
        "maximum_logical_requests": 1 + 3 * future,
        "maximum_physical_attempts": 2 + 6 * future,
    }
    expected = {
        "dispatcher_module": DISPATCHER, "seed": lane.seed, "round": lane.round, "logical_index": 0,
        "previous_contract_sha256": lane.contract_sha256,
        "previous_authorization_sha256": lane.authorization_sha256,
        "previous_run_directory": str(path.with_name(lane.parent.name)),
        "previous_request_id": lane.request_id, "payload_sha256": lane.payload_sha256,
        "previous_terminal_file_sha256": lane.terminal_file_sha256,
        "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 2},
        "execution_budget": budget, "bridge_wait_seconds": resilient.TIMEOUTS.bridge_wait_seconds,
    }
    settings = {
        "provider_capability": "gated_ipv4", "arm": "A7", "memory": "M2", "device": "cuda",
        "initial_labels": 333, "target_labels": 1485, "batch_size": 32, "api_parallelism": 1,
        "store": False, "confirmation_seeds": [lane.seed], "qualification_seed": lane.seed,
        "requires_initial_qualification": False, "wire_representation": FORMAL_WIRE_REPRESENTATION,
        "sse_inactivity_seconds": 900, "ipv4_preflight_retry": True, "ipv4_readiness_windows": 3,
        "normalize_observed_reference_aliases": True, "audited_future_transport_replay": True,
        "replay_eligible_from_round": lane.round, "sealed_relay_retry_seconds": 180,
    }
    boundaries = profile.get("resume_boundaries", {})
    if (path.name != lane.run_name or record["run_directory"] != run or record["api_execution"] != SSH_API
            or record["gpu_slots"] != [0] or record["budget"] != budget or "fixed_reference_repair" in profile
            or request_timeouts(profile) != resilient.TIMEOUTS
            or any(type(proof.get(k)) is not type(v) or proof[k] != v for k, v in expected.items())
            or any(type(profile.get(k)) is not type(v) or profile[k] != v for k, v in settings.items())
            or set(boundaries) != {str(lane.seed)} or boundaries[str(lane.seed)].get("round") != lane.round
            or not isinstance(proof.get("previous_prefix_files"), dict)
            or digest(proof["previous_prefix_files"]) != proof.get("previous_prefix_sha256")
            or not all(_sha(proof.get(k)) for k in ("checkpoint_sha256", "packet_sha256", "wire_packet_sha256",
                                                  "base_messages_sha256", "previous_prefix_sha256"))):
        raise PermissionError("the relay retry scope, cooldown or remaining budget differs from approval")
    verify_local_source(info, root=ROOT)
    sources = (*resilient.SOURCES, STUDY / "scripts" / f"{DISPATCHER}.py",
               STUDY / "scripts/run_v13_relay_retry.py")
    for source in sources:
        if record["source_sha256"].get(source.as_posix()) != file_digest(ROOT / source):
            raise PermissionError(f"the approved relay retry source differs: {source.name}")
    return proof


def _archive(spools, proof, lane):
    spool = _spool(spools, lane.contract_sha256, lane.authorization_sha256)
    old, transport = resilient._attempts(
        spool, lane.contract_sha256, lane.authorization_sha256, proof["previous_attempts"],
        lane.seed, lane.round, lane.request_id, 1,
    )
    receipt = proof["previous_attempts"][0]
    observation = {
        "error_type": "response_id_mismatch", "conflicting_event_type": "response.failed",
        "http_status": 200, "sse_event_count": 41, "provider_request_id": lane.provider_request_id,
        "response_id": lane.response_id, "conflicting_response_id": "resp_" + lane.provider_request_id.replace("-", ""),
    }
    attempt = spool / "attempts" / lane.attempt_id
    progress = read_bridge_json(attempt / "provider_progress.json")
    model = read_bridge_json(spool / "provider_model_lock.json")
    binding = old["binding"]
    if (tuple(receipt[k] for k in ("attempt_id", "job_sha256", "result_sha256", "error", "provider_dispatched", "ambiguous"))
            != (lane.attempt_id, lane.job_sha256, lane.result_sha256, "response_id_mismatch", True, True)
            or proof.get("previous_transport_observation") != observation
            or any(progress.get(k) != v for k, v in observation.items())
            or progress.get("response_id_matches_transport_alias") is not True
            or progress.get("complete_terminal_captured") is True
            or model != proof["model_identity"] or old["payload_sha256"] != lane.payload_sha256
            or old["payload"].get("stream") is not True
            or binding.get("normalize_observed_reference_aliases") is not True
            or any(binding[k] != proof[k] for k in ("checkpoint_sha256", "packet_sha256", "wire_packet_sha256",
                                                    "base_messages_sha256"))):
        raise PermissionError("the retired request, model or explicit relay failure differs from its archive")
    return old, transport, model


class RelayRetryDispatcher(resilient.ResilientDispatcher):
    def validate(self, job):
        request, config, started = Dispatcher.validate(self, job)
        binding = request["binding"]
        seed, first = self.scope["seed"], self.scope["round"]
        index, physical = binding["round"], started["attempt"]
        if (binding["seed"] != seed or not first <= index < 36 or binding["prompt_sha256"] != self.prompt_sha
                or binding.get("normalize_observed_reference_aliases") is not True
                or not 0 <= request["logical_index"] < config.logical_replies
                or not 0 <= physical < config.transport_attempts):
            raise PermissionError("the bridge job is outside this exact continued trajectory")
        if index == first:
            self._check_boundary(request, started)
        return request, config, started

    def _check_boundary(self, request, started):
        current, archived = _science(request), _science(self.original)
        reconstruction = self.scope.get("authoritative_message_reconstruction")
        if reconstruction is not None:
            field = "authoritative_base_messages_sha256"
            if reconstruction != {"previous_sha256": archived.get(field), "reconstructed_sha256": current.get(field)}:
                raise PermissionError("the authoritative message reconstruction differs from approval")
            current.pop(field)
            archived.pop(field)
        if (request["logical_index"] != 0 or started["attempt"] not in (0, 1)
                or request["payload"] != self.original["payload"] or current != archived
                or request["request_id"] == self.original["request_id"]
                or started["attempt_id"] == self.scope["previous_attempts"][0]["attempt_id"]):
            raise PermissionError("the boundary authorizes only its exact archived logical input")
        if started["attempt"] == 1:
            self._check_boundary_retry(request, started)

    def _check_boundary_retry(self, request, started):
        seed, index = self.scope["seed"], self.scope["round"]
        slot = read_bridge_json(self.path / "slots" / f"{seed}-{index}-0-0.json")
        identity = slot["attempt_id"]
        if identity == started["attempt_id"] or not _sha(slot.get("job_sha256")):
            raise PermissionError("the relay retry requires a separate first reservation")
        filename = f"{identity}.json"
        prior = read_bridge_json(self.path / "jobs" / filename)
        old, _, old_started = verify_job(prior)
        result = read_bridge_json(self.path / "results" / filename)
        validate_result(result, prior)
        ack = read_bridge_json(self.path / "delivered" / filename)
        failure = result["outcome"].get("failure") or {}
        evidence = failure.get("reply_identity") or {}
        if (old != request or old_started["attempt"] != 0 or slot["job_sha256"] != prior["job_sha256"]
                or ack != {"attempt_id": identity, "result_sha256": result["result_sha256"]}
                or result["provider_dispatched"] is not True or result["raw_response"] is not None
                or failure.get("ambiguous") is not True or failure.get("retryable") is not False
                or set(evidence) != {"provider_status", "provider_request_id", "response_id", "served_model", "usage"}
                or evidence.get("usage") != {} or evidence.get("served_model") not in (None, self.transport.served_model)
                or not is_sealed_relay_failure({**evidence, "error": failure.get("code")})):
            raise PermissionError("the second boundary slot requires the first sealed, delivered relay failure")


def run(args):
    lane = LANES[args.seed]
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.retry_exchange("info"), expected_contract=args.contract_sha256,
                       expected_auth=args.authorization_sha256)
    proof = _scope(info, args.remote_run_dir, lane)
    if args.contract_sha256 in {lane.contract_sha256, resilient.PILOT_CONTRACT_SHA256}:
        raise PermissionError("the new relay retry needs a separate Mac spool")
    spools = Path.home() / ".config/purellm/spool"
    original, old_transport, model = _archive(spools, proof, lane)
    pilot, _, pilot_transport = resilient._pool(spools / resilient.PILOT_CONTRACT_SHA256)
    registered = replace(pilot_transport, timeout_seconds=resilient.TIMEOUTS.total_seconds, wire_mode=RESPONSES_STREAM_IPV4)
    transport = TransportConfig(**info["contract"]["profile"]["transport"])
    if (transport != registered or old_transport != registered
            or (transport.provider, transport.base_url, transport.requested_model, transport.served_model,
                transport.reasoning_effort, transport.logical_replies, transport.transport_attempts,
                transport.max_output_tokens, transport.sdk_version)
            != ("token4research", "https://token4research.cn/v1", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh", 3, 2,
                32768, "2.28.0") or not sdk_status(transport)["matches"]):
        raise PermissionError("the retry must retain the approved model, timeout and wire settings")
    if args.verify_only:
        return {"seed": lane.seed, "round": lane.round, "contract_sha256": args.contract_sha256, "verified_no_dispatch": True}
    spool = spools / args.contract_sha256
    dispatcher = RelayRetryDispatcher(spool, info=info, contract_sha256=args.contract_sha256, transport=transport,
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
    parser.add_argument("--seed", type=int, choices=tuple(LANES), required=True)
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
