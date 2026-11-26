"""Dispatch signed A7 resumes with immutable prior receipts and a shared Key pool."""

import argparse
from dataclasses import replace
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import RESPONSES_STREAM_IPV4, TransportConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import read_bridge_json
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAL_WIRE_REPRESENTATION
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import Dispatcher, run_dispatcher, verify_local_source
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, SSH_API, STUDY, request_timeouts
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import sdk_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import GPU_PYTHON, SSHCourier, verify_info
from .dispatch_prompt_direct import _check_seed_lease
from .dispatch_prompt_pilot import SharedKeyPool
from .dispatch_prompt_successor import PILOT_CONTRACT_SHA256, _approved_prompt_sha, _pool
from .dispatch_v13_bounded_sse_recovery import (
    FOLLOW_2357, FOLLOW_2357_AUTH, FOLLOW_2357_RUN, INACTIVITY, TIMEOUTS, _science, _sealed, _sha, _spool,
)
from .dispatch_v13_lane_extension import PARENT_AUTHORIZATION, PARENT_CONTRACT, PARENT_RUN


MODULE = "dispatch_v13_resilient_sse"
ROUNDS = {157: 3, 887: 6, 2357: 2, 6101: 1, 12203: 4}
RECOVERY_BUDGET = {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
SOURCES = tuple(STUDY / "scripts" / f"{name}.py" for name in (
    MODULE, "dispatch_v13_bounded_sse_recovery", "dispatch_prompt_direct",
    "dispatch_prompt_pilot", "dispatch_prompt_successor", "dispatch_v13_lane_extension",
))


def _origin(seed):
    if seed == 2357:
        return FOLLOW_2357, FOLLOW_2357_AUTH, FOLLOW_2357_RUN
    return PARENT_CONTRACT, PARENT_AUTHORIZATION, PARENT_RUN


def _scope(info, run, seed):
    record = info["contract"]
    profile, proof = record["profile"], record["predecessor"]
    location = Path(run)
    start = ROUNDS[seed]
    origin, auth, name = _origin(seed)
    previous = name if seed == 12203 else f"{PARENT_RUN}-{seed}-bounded-sse-01"
    expected_profile = {
        "provider_capability": "gated_ipv4", "arm": "A7", "memory": "M2", "device": "cuda",
        "initial_labels": 333, "target_labels": 1485, "batch_size": 32, "api_parallelism": 1,
        "store": False, "confirmation_seeds": [seed], "qualification_seed": seed,
        "requires_initial_qualification": False, "wire_representation": FORMAL_WIRE_REPRESENTATION,
        "sse_inactivity_seconds": INACTIVITY, "ipv4_preflight_retry": True, "ipv4_readiness_windows": 3,
        "normalize_observed_reference_aliases": True, "audited_future_transport_replay": True,
        "replay_eligible_from_round": start + 1,
    }
    remaining = 35 - start
    budget = {"first_logical_requests": 1 + remaining, "maximum_logical_requests": 1 + 3 * remaining,
              "maximum_physical_attempts": 1 + 6 * remaining}
    expected_proof = {
        "dispatcher_module": MODULE, "seed": seed, "round": start, "logical_index": 0,
        "parent_contract_sha256": origin, "parent_authorization_sha256": auth,
        "parent_run_directory": str(location.with_name(name)),
        "previous_run_directory": str(location.with_name(previous)), "execution_budget": budget,
        "recovery_budget": RECOVERY_BUDGET, "bridge_wait_seconds": TIMEOUTS.bridge_wait_seconds,
    }
    boundaries = profile.get("resume_boundaries", {})
    bound_fields = ("prior_request_id", "payload_sha256", "checkpoint_sha256", "packet_sha256",
                    "wire_packet_sha256", "base_messages_sha256", "previous_contract_sha256",
                    "previous_authorization_sha256", "previous_request_id")
    if (location.name != f"{PARENT_RUN}-{seed}-resilient-sse-01" or record["run_directory"] != run
            or record["api_execution"] != SSH_API or record["gpu_slots"] != [0] or record["budget"] != budget
            or request_timeouts(profile) != TIMEOUTS or "fixed_reference_repair" in profile
            or any(type(profile.get(k)) is not type(v) or profile[k] != v for k, v in expected_profile.items())
            or any(type(proof.get(k)) is not type(v) or proof[k] != v for k, v in expected_proof.items())
            or not all(_sha(proof.get(key)) for key in bound_fields)
            or set(boundaries) != {str(seed)} or boundaries[str(seed)].get("round") != start):
        raise PermissionError("the signed resume differs from the exact original round or finite L1485 budget")
    verify_local_source(info, root=ROOT)
    for path in SOURCES:
        if record["source_sha256"].get(path.as_posix()) != file_digest(ROOT / path):
            raise PermissionError(f"the approved resume source differs: {path.name}")
    return proof


def _attempts(spool, contract, approval, evidence, seed, round_index, request_id, count):
    if type(evidence) is not list or len(evidence) != count:
        raise PermissionError("the old physical attempt receipts must all be bound")
    original = None
    for index, item in enumerate(evidence):
        identity = item["attempt_id"]
        request, transport, started, job, result = _sealed(spool, identity, contract, approval)
        failure = result["outcome"]["failure"]
        expected = {
            "attempt": index, "attempt_id": identity, "job_sha256": job["job_sha256"],
            "result_sha256": result["result_sha256"], "terminal_sha256": item.get("terminal_sha256"),
            "error": failure["code"], "provider_dispatched": result["provider_dispatched"],
            "ambiguous": failure["ambiguous"],
        }
        slot = read_bridge_json(spool / "slots" / f"{seed}-{round_index}-0-{index}.json")
        if (item != expected or not _sha(item.get("terminal_sha256"))
                or slot != {"attempt_id": identity, "job_sha256": job["job_sha256"]}
                or (started["attempt"], request["binding"]["seed"], request["binding"]["round"], request["logical_index"])
                != (index, seed, round_index, 0) or request["request_id"] != request_id
                or (original is not None and original != (request, transport))):
            raise PermissionError("a previous physical reservation differs from the archived request or failure")
        original = request, transport
    return original


def _archive(spools, proof):
    seed, index = proof["seed"], proof["round"]
    previous_contract, previous_auth = proof["previous_contract_sha256"], proof["previous_authorization_sha256"]
    origin_contract, origin_auth, _ = _origin(seed)
    origin = _spool(spools, origin_contract, origin_auth)
    previous = _spool(spools, previous_contract, previous_auth)
    original, old_transport = _attempts(origin, origin_contract, origin_auth, proof["prior_attempts"],
                                         seed, index, proof["prior_request_id"], 2)
    spent, _ = _attempts(previous, previous_contract, previous_auth, proof["previous_attempts"],
                          seed, index, proof["previous_request_id"], 2 if seed == 12203 else 1)
    model = read_bridge_json(origin / "provider_model_lock.json")
    if (original["payload_sha256"] != proof["payload_sha256"] or spent["payload"] != original["payload"]
            or _science(spent) != _science(original) or original["payload"].get("stream") is not True
            or any(original["binding"][k] != proof[k] for k in (
                "checkpoint_sha256", "packet_sha256", "wire_packet_sha256", "base_messages_sha256"))
            or model != proof["model_identity"] or read_bridge_json(previous / "provider_model_lock.json") != model):
        raise PermissionError("the resume does not reproduce the original A7 science and model lock")
    return original, old_transport, model


class ResilientDispatcher(Dispatcher):
    def validate(self, job):
        request, config, started = super().validate(job)
        binding = request["binding"]
        seed, first = self.scope["seed"], self.scope["round"]
        round_index = binding["round"]
        if (binding["seed"] != seed or not first <= round_index < 36
                or binding["prompt_sha256"] != self.prompt_sha
                or binding.get("normalize_observed_reference_aliases") is not True
                or not 0 <= request["logical_index"] < config.logical_replies
                or not 0 <= started["attempt"] < config.transport_attempts):
            raise PermissionError("this bridge job is outside the signed resumed A7 trajectory")
        if round_index == first:
            science = _science(request)
            science.pop("normalize_observed_reference_aliases")
            if ((request["logical_index"], started["attempt"]) != (0, 0)
                    or request["payload"] != self.original["payload"] or science != _science(self.original)
                    or request["request_id"] in {self.original["request_id"], self.scope["previous_request_id"]}
                    or started["attempt_id"] in {r["attempt_id"] for r in self.scope["prior_attempts"] + self.scope["previous_attempts"]}):
                raise PermissionError("the first resumed round allows one exact archived input and new physical identity")
        return request, config, started

    def dispatch(self, attempt_id, *, before_call, pause=None):
        def check(identity, job):
            before_call(identity, job)
            _check_seed_lease(self.pool, self.scope["seed"])

        return super().dispatch(attempt_id, before_call=check, pause=pause, before_no_post=before_call)


def run(args):
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.retry_exchange("info"), expected_contract=args.contract_sha256,
                       expected_auth=args.authorization_sha256)
    proof = _scope(info, args.remote_run_dir, args.seed)
    if args.contract_sha256 in {PILOT_CONTRACT_SHA256, proof["parent_contract_sha256"], proof["previous_contract_sha256"]}:
        raise PermissionError("the resumed lane needs its own scoped Mac spool")
    spools = Path.home() / ".config/purellm/spool"
    original, old_transport, model = _archive(spools, proof)
    pilot, _, pilot_transport = _pool(spools / PILOT_CONTRACT_SHA256)
    registered = replace(pilot_transport, timeout_seconds=TIMEOUTS.total_seconds, wire_mode=RESPONSES_STREAM_IPV4)
    transport = TransportConfig(**info["contract"]["profile"]["transport"])
    if (old_transport != registered or transport != registered
            or (transport.provider, transport.base_url, transport.requested_model, transport.served_model,
                transport.reasoning_effort, transport.logical_replies, transport.transport_attempts,
                transport.max_output_tokens, transport.sdk_version)
            != ("token4research", "https://token4research.cn/v1", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh", 3, 2,
                32768, "2.28.0") or not sdk_status(transport)["matches"]):
        raise PermissionError("the resume must retain the exact original direct SSE settings and future budgets")
    spool = spools / args.contract_sha256
    dispatcher = ResilientDispatcher(spool, info=info, contract_sha256=args.contract_sha256,
                                     transport=transport, mock=False, pool=SharedKeyPool(pilot.directory, pilot.config))
    dispatcher.scope, dispatcher.original = proof, original
    dispatcher.prompt_sha = _approved_prompt_sha(info)
    once(spool / "provider_model_lock.json", model)
    return run_dispatcher(dispatcher, courier, watch=True, workers=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-paid", action="store_true", required=True)
    parser.add_argument("--seed", type=int, choices=tuple(ROUNDS), required=True)
    parser.add_argument("--ssh-host", default="eit-hpc")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--remote-python", default=GPU_PYTHON)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    args = parser.parse_args()
    if not all(_sha(value) for value in (args.contract_sha256, args.authorization_sha256)):
        parser.error("exact contract and authorization hashes required")
    os.umask(0o077)
    print(dumps(run(args)))


if __name__ == "__main__":
    main()
