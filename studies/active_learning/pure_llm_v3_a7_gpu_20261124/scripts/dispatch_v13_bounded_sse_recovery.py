"""Dispatch one audited direct SSE recovery for each approved exhausted A7 lane."""

import argparse
from dataclasses import replace
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import RESPONSES_STREAM_IPV4, TransportConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import read_bridge_json, validate_result, verify_job
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAL_WIRE_REPRESENTATION
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import (
    Dispatcher, _private_directory, run_dispatcher, verify_local_source,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    ROOT, SSH_API, STUDY, RequestTimeouts, request_timeouts,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import sdk_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import GPU_PYTHON, SHA256, SSHCourier, verify_info
from .dispatch_prompt_direct import _check_seed_lease
from .dispatch_prompt_pilot import SharedKeyPool
from .dispatch_prompt_successor import PILOT_CONTRACT_SHA256, _approved_prompt_sha, _pool
from .dispatch_v13_lane_extension import BUDGET, PARENT_AUTHORIZATION, PARENT_CONTRACT, PARENT_RUN


MODULE = "dispatch_v13_bounded_sse_recovery"
TIMEOUTS = RequestTimeouts(read_idle_seconds=1800, total_seconds=3600)
INACTIVITY = 900
FOLLOW_2357 = "99fd5844f05bca5d7d6be41bc82dbccc9dc77d90faba61101ff5a0c490403c2b"
FOLLOW_2357_AUTH = "01dce8c6c2feef6f9b042858189fddcd95f77f0678867fc9f3f3731a39134f16"
FOLLOW_2357_RUN = f"{PARENT_RUN}-2357-continuation-01"
LANES = {157: (3, "-157-extension-01"), 887: (6, "-887-json-extension-01"),
         2357: (2, "-2357-r2-extension-01"), 6101: (1, "-6101-ws-extension-01")}
SOURCES = tuple(STUDY / "scripts" / f"{name}.py" for name in (
    MODULE, "dispatch_prompt_direct", "dispatch_prompt_pilot", "dispatch_prompt_successor",
))


def _sha(value):
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _scope(info, run, seed):
    record = info["contract"]
    profile, prior = record["profile"], record["predecessor"]
    location = Path(run)
    round_index, last_suffix = LANES[seed]
    origin, auth, origin_run = ((FOLLOW_2357, FOLLOW_2357_AUTH, FOLLOW_2357_RUN) if seed == 2357
                                else (PARENT_CONTRACT, PARENT_AUTHORIZATION, PARENT_RUN))
    expected_profile = {"provider_capability": "gated_ipv4", "arm": "A7", "memory": "M2", "device": "cuda",
                        "initial_labels": 333, "target_labels": 333 + (round_index + 1) * 32, "batch_size": 32,
                        "api_parallelism": 1, "store": False, "confirmation_seeds": [seed],
                        "qualification_seed": seed, "requires_initial_qualification": False,
                        "wire_representation": FORMAL_WIRE_REPRESENTATION, "sse_inactivity_seconds": INACTIVITY}
    expected_prior = {"dispatcher_module": MODULE, "seed": seed, "round": round_index, "logical_index": 0,
                      "parent_contract_sha256": origin, "parent_authorization_sha256": auth,
                      "parent_run_directory": str(location.with_name(origin_run)),
                      "previous_run_directory": str(location.with_name(PARENT_RUN + last_suffix)),
                      "execution_budget": BUDGET, "bridge_wait_seconds": TIMEOUTS.bridge_wait_seconds}
    hash_fields = ("prior_request_id", "payload_sha256", "checkpoint_sha256", "packet_sha256", "wire_packet_sha256",
                   "base_messages_sha256", "previous_contract_sha256", "previous_authorization_sha256", "previous_request_id")
    boundaries = profile.get("resume_boundaries", {})
    if (location.name != f"{PARENT_RUN}-{seed}-bounded-sse-01" or record["run_directory"] != run
            or record["api_execution"] != SSH_API or record["gpu_slots"] != [0] or record["budget"] != BUDGET
            or request_timeouts(profile) != TIMEOUTS or "fixed_reference_repair" in profile
            or any(type(profile.get(k)) is not type(v) or profile[k] != v for k, v in expected_profile.items())
            or any(type(prior.get(k)) is not type(v) or prior[k] != v for k, v in expected_prior.items())
            or not all(_sha(prior.get(key)) for key in hash_fields)
            or set(boundaries) != {str(seed)} or boundaries[str(seed)].get("round") != round_index):
        raise PermissionError("the signed recovery differs from the original seed boundary or one-attempt budget")
    verify_local_source(info, root=ROOT)
    for path in SOURCES:
        if record["source_sha256"].get(path.as_posix()) != file_digest(ROOT / path):
            raise PermissionError(f"the approved recovery source differs: {path.name}")
    return prior


def _spool(spools, contract, approval):
    path = spools / contract
    _private_directory(path)
    if read_bridge_json(path / "run_binding.json") != {
            "contract_sha256": contract, "authorization_sha256": approval, "mock": False}:
        raise PermissionError("the archived Mac spool is outside the original authorization")
    return path


def _sealed(spool, identity, contract, approval):
    name = f"{identity}.json"
    job = read_bridge_json(spool / "jobs" / name)
    request, transport, started = verify_job(job, contract_sha256=contract, authorization_sha256=approval,
                                             wire_representation=FORMAL_WIRE_REPRESENTATION)
    result = validate_result(read_bridge_json(spool / "results" / name), job)
    ack = read_bridge_json(spool / "delivered" / name)
    attempt = spool / "attempts" / identity
    if (started["attempt_id"] != identity or result["outcome"]["status"] != "failed"
            or result["raw_response"] is not None
            or ack != {"attempt_id": identity, "result_sha256": result["result_sha256"]}
            or any((attempt / path).exists() for path in ("raw_provider_response.json", "provider_capture_receipt.json"))):
        raise PermissionError("complete or undelivered provider evidence must be resolved before new dispatch")
    return request, transport, started, job, result


def _science(request):
    return {k: v for k, v in request["binding"].items() if k != "a7_formal"}


def _archive(spools, scope):
    seed, round_index = scope["seed"], scope["round"]
    original_contract, original_auth = scope["parent_contract_sha256"], scope["parent_authorization_sha256"]
    previous_contract, previous_auth = scope["previous_contract_sha256"], scope["previous_authorization_sha256"]
    origin = _spool(spools, original_contract, original_auth)
    previous = _spool(spools, previous_contract, previous_auth)
    prior_attempts = scope.get("prior_attempts")
    if type(prior_attempts) is not list or len(prior_attempts) != 2:
        raise PermissionError("the original two spent reservations must be bound")
    old = []
    for index, evidence in enumerate(prior_attempts):
        slot = read_bridge_json(origin / "slots" / f"{seed}-{round_index}-0-{index}.json")
        request, transport, started, job, result = _sealed(origin, slot["attempt_id"], original_contract, original_auth)
        failure = result["outcome"]["failure"]
        expected = {"attempt": index, "attempt_id": started["attempt_id"], "job_sha256": job["job_sha256"],
                    "result_sha256": result["result_sha256"], "terminal_sha256": evidence.get("terminal_sha256"),
                    "error": failure["code"], "provider_dispatched": result["provider_dispatched"],
                    "ambiguous": failure["ambiguous"]}
        if (evidence != expected or not _sha(evidence.get("terminal_sha256"))
                or job["job_sha256"] != slot["job_sha256"] or started["attempt"] != index
                or (request["binding"]["seed"], request["binding"]["round"], request["logical_index"]) != (seed, round_index, 0)
                or (request["request_id"], request["payload_sha256"]) != (scope["prior_request_id"], scope["payload_sha256"])
                or any(request["binding"][k] != scope[k] for k in (
                    "checkpoint_sha256", "packet_sha256", "wire_packet_sha256", "base_messages_sha256"))):
            raise PermissionError("the original physical reservations differ from the signed archived request")
        old.append((request, transport))
    if old[0] != old[1]:
        raise PermissionError("the two original physical slots do not bind the same request")
    original, old_transport = old[0]
    evidence = scope["previous_attempt"]
    identity = evidence["attempt_id"]
    request, _, started, job, result = _sealed(previous, identity, previous_contract, previous_auth)
    failure = result["outcome"]["failure"]
    expected = {"attempt": 0, "attempt_id": identity, "job_sha256": job["job_sha256"],
                "result_sha256": result["result_sha256"], "terminal_sha256": evidence.get("terminal_sha256"),
                "error": failure["code"], "provider_dispatched": True, "ambiguous": True}
    expected_payload = {**original["payload"], "stream": False} if seed == 887 else original["payload"]
    model = read_bridge_json(origin / "provider_model_lock.json")
    if (not _sha(evidence.get("terminal_sha256")) or evidence != expected or started["attempt"] != 0
            or request["request_id"] != scope["previous_request_id"] or request["payload"] != expected_payload
            or _science(request) != _science(original) or result["provider_dispatched"] is not True
            or failure["ambiguous"] is not True or model != scope["model_identity"]
            or read_bridge_json(previous / "provider_model_lock.json") != model):
        raise PermissionError("latest spent attempt or original scientific request differs from the new exact scope")
    return original, old_transport, model


class BoundedRecoveryDispatcher(Dispatcher):
    def validate(self, job):
        request, config, started = super().validate(job)
        original, state = self.original, request["binding"]
        seed, round_index = self.scope["seed"], self.scope["round"]
        if ((state["seed"], state["round"], request["logical_index"], started["attempt"]) != (seed, round_index, 0, 0)
                or state["prompt_sha256"] != self.prompt_sha or _science(request) != _science(original)
                or request["payload"] != original["payload"] or request["payload_sha256"] != self.scope["payload_sha256"]
                or request["request_id"] in {original["request_id"], self.scope["previous_request_id"]}
                or started["attempt_id"] == self.scope["previous_attempt"]["attempt_id"]):
            raise PermissionError("bounded recovery cannot submit a different scientific request or an extra attempt")
        return request, config, started

    def dispatch(self, attempt_id, *, before_call, pause=None):
        def check(identity, job):
            before_call(identity, job)
            _check_seed_lease(self.pool, self.scope["seed"])

        return super().dispatch(attempt_id, before_call=check, pause=pause)


def run(args):
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.info(), expected_contract=args.contract_sha256, expected_auth=args.authorization_sha256)
    scope = _scope(info, args.remote_run_dir, args.seed)
    if args.contract_sha256 in {PILOT_CONTRACT_SHA256, scope["parent_contract_sha256"], scope["previous_contract_sha256"]}:
        raise PermissionError("each new recovery must use its own scoped Mac spool")
    spools = Path.home() / ".config/purellm/spool"
    original, old_transport, model = _archive(spools, scope)
    pilot, _, pilot_transport = _pool(spools / PILOT_CONTRACT_SHA256)
    registered = replace(pilot_transport, timeout_seconds=TIMEOUTS.total_seconds, wire_mode=RESPONSES_STREAM_IPV4)
    transport = TransportConfig(**info["contract"]["profile"]["transport"])
    if (old_transport != registered or transport != replace(registered, logical_replies=1, transport_attempts=1)
            or (transport.provider, transport.base_url, transport.requested_model, transport.served_model,
                transport.reasoning_effort, transport.max_output_tokens, transport.sdk_version)
            != ("token4research", "https://token4research.cn/v1", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh", 32768, "2.28.0")
            or not sdk_status(transport)["matches"]):
        raise PermissionError("the new physical slot must retain the original Token4Research direct SSE settings")
    spool = spools / args.contract_sha256
    pool = SharedKeyPool(pilot.directory, pilot.config)
    dispatcher = BoundedRecoveryDispatcher(spool, info=info, contract_sha256=args.contract_sha256,
                                           transport=transport, mock=False, pool=pool)
    dispatcher.original = original
    dispatcher.scope = scope
    dispatcher.prompt_sha = _approved_prompt_sha(info)
    once(spool / "provider_model_lock.json", model)
    return run_dispatcher(dispatcher, courier, watch=True, workers=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-paid", action="store_true", required=True)
    parser.add_argument("--seed", type=int, choices=tuple(LANES), required=True)
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
