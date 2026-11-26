"""Dispatch one exact archived resilient recovery, then its original future rounds."""

import argparse
from dataclasses import replace
import math
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import RESPONSES_STREAM_IPV4, TransportConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import read_bridge_json
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAL_WIRE_REPRESENTATION
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import run_dispatcher, verify_local_source
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, SSH_API, STUDY, request_timeouts
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import sdk_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import GPU_PYTHON, SSHCourier, verify_info
from . import dispatch_v13_resilient_sse as resilient
from .dispatch_v13_bounded_sse_recovery import _sha, _science, _spool
from .run_v13_resilient_failure_recovery import DISPATCHER, FAILURES


def _scope(info, run, failed):
    record = info["contract"]
    profile, proof = record["profile"], record["predecessor"]
    location = Path(run)
    index = failed.lane.round
    parent, parent_auth, parent_name = resilient._origin(failed.seed)
    future = 35 - index
    budget = {"first_logical_requests": 1 + future, "maximum_logical_requests": 1 + 3 * future,
              "maximum_physical_attempts": 1 + 6 * future}
    expected = {
        "dispatcher_module": DISPATCHER, "seed": failed.seed, "round": index, "logical_index": 0,
        "parent_contract_sha256": parent, "parent_authorization_sha256": parent_auth,
        "parent_run_directory": str(location.with_name(parent_name)),
        "previous_contract_sha256": failed.contract_sha256, "previous_authorization_sha256": failed.authorization_sha256,
        "previous_run_directory": str(location.with_name(failed.archived.name)), "previous_request_id": failed.request_id,
        "recovery_budget": resilient.RECOVERY_BUDGET, "execution_budget": budget,
        "bridge_wait_seconds": resilient.TIMEOUTS.bridge_wait_seconds,
    }
    expected_profile = {
        "provider_capability": "gated_ipv4", "arm": "A7", "memory": "M2", "device": "cuda",
        "initial_labels": 333, "target_labels": 1485, "batch_size": 32, "api_parallelism": 1,
        "store": False, "confirmation_seeds": [failed.seed], "qualification_seed": failed.seed,
        "requires_initial_qualification": False, "wire_representation": FORMAL_WIRE_REPRESENTATION,
        "sse_inactivity_seconds": 900, "ipv4_preflight_retry": True, "ipv4_readiness_windows": 3,
        "normalize_observed_reference_aliases": True, "audited_future_transport_replay": True,
        "replay_eligible_from_round": index + 1,
    }
    boundaries = profile.get("resume_boundaries", {})
    if (location.name != failed.run_name or record["run_directory"] != run or record["api_execution"] != SSH_API
            or record["gpu_slots"] != [0] or record["budget"] != budget or "fixed_reference_repair" in profile
            or request_timeouts(profile) != resilient.TIMEOUTS
            or any(type(proof.get(k)) is not type(v) or proof[k] != v for k, v in expected.items())
            or any(type(profile.get(k)) is not type(v) or profile[k] != v for k, v in expected_profile.items())
            or set(boundaries) != {str(failed.seed)} or boundaries[str(failed.seed)].get("round") != index):
        raise PermissionError("the signed recovery does not preserve the exact prior science and finite slot")
    verify_local_source(info, root=ROOT)
    sources = (*resilient.SOURCES, STUDY / "scripts" / f"{DISPATCHER}.py",
               STUDY / "scripts/run_v13_resilient_failure_recovery.py")
    for path in sources:
        if record["source_sha256"].get(path.as_posix()) != file_digest(ROOT / path):
            raise PermissionError(f"the approved recovery source differs: {path.name}")
    return proof


def _no_model_post(spool, attempt, failed):
    for window in range(1, 4):
        receipt = read_bridge_json(attempt / f"readiness_{window:02d}.json")
        retry = receipt.get("retry_unix")
        if ((receipt.get("attempt_id"), receipt.get("job_sha256"), receipt.get("window"))
                != (failed.attempt_id, failed.job_sha256, window)
                or type(retry) not in (float, int) or not math.isfinite(retry) or retry < 0):
            raise PermissionError("the retired no-POST claim lacks the three exact readiness receipts")
    if (attempt / "started.json").exists():
        raise PermissionError("a possible old model POST conflicts with the no-POST receipt")
    slot = spool / "slots" / f"{failed.seed}-{failed.lane.round}-0-1.json"
    if slot.exists():
        raise PermissionError("the rejected extra GPU reservation unexpectedly reached the Mac")


def _previous(spools, proof, failed, original, model):
    spool = _spool(spools, failed.contract_sha256, failed.authorization_sha256)
    spent, _ = resilient._attempts(spool, failed.contract_sha256, failed.authorization_sha256,
                                    proof["previous_attempts"], failed.seed, failed.lane.round, failed.request_id, 1)
    receipt = proof["previous_attempts"][0]
    expected = (failed.attempt_id, failed.job_sha256, failed.result_sha256, failed.code,
                failed.provider_dispatched, failed.provider_dispatched)
    old_binding = _science(spent)
    if (old_binding.pop("normalize_observed_reference_aliases", None) is not True
            or old_binding != _science(original) or spent["payload"] != original["payload"]
            or read_bridge_json(spool / "provider_model_lock.json") != model
            or tuple(receipt[key] for key in ("attempt_id", "job_sha256", "result_sha256", "error",
                                             "provider_dispatched", "ambiguous")) != expected):
        raise PermissionError("the immediate sealed failure is not the known exact archived request")

    attempt = spool / "attempts" / failed.attempt_id
    progress = read_bridge_json(attempt / "provider_progress.json")
    observation = {"http_status": failed.http_status, "error_type": failed.provider_error,
                   "sse_event_count": failed.sse_event_count}
    if failed.provider_request_id is not None:
        observation.update(provider_request_id=failed.provider_request_id, response_id=failed.response_id,
                           conflicting_response_id="resp_" + failed.provider_request_id.replace("-", ""))
    if (proof.get("previous_transport_observation") != observation
            or any(progress.get(key) != value for key, value in observation.items())
            or progress.get("complete_terminal_captured") is True):
        raise PermissionError("the original route evidence differs from the exact sealed failure")
    if not failed.provider_dispatched:
        _no_model_post(spool, attempt, failed)
        aborted = proof.get("previous_aborted_reservation")
        if aborted is not None:
            if ((aborted.get("attempt"), aborted.get("attempt_id"), aborted.get("provider_dispatched"))
                    != (1, failed.aborted_attempt_id, False) or not _sha(aborted.get("started_sha256"))
                    or any((spool / folder / f"{failed.aborted_attempt_id}.json").exists() for folder in ("jobs", "results", "delivered"))
                    or (spool / "attempts" / failed.aborted_attempt_id).exists()):
                raise PermissionError("the rejected local reservation unexpectedly reached the Mac")


def run(args):
    failed = FAILURES[args.seed]
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.info(), expected_contract=args.contract_sha256, expected_auth=args.authorization_sha256)
    proof = _scope(info, args.remote_run_dir, failed)
    if args.contract_sha256 in {resilient.PILOT_CONTRACT_SHA256, proof["parent_contract_sha256"],
                               failed.contract_sha256, proof["archived_previous"]["contract_sha256"]}:
        raise PermissionError("the newly approved lane requires its own Mac spool")
    spools = Path.home() / ".config/purellm/spool"
    older = {**proof, **{f"previous_{key}": value for key, value in proof["archived_previous"].items()}}
    original, old_transport, model = resilient._archive(spools, older)
    _previous(spools, proof, failed, original, model)
    pilot, _, pilot_transport = resilient._pool(spools / resilient.PILOT_CONTRACT_SHA256)
    registered = replace(pilot_transport, timeout_seconds=resilient.TIMEOUTS.total_seconds, wire_mode=RESPONSES_STREAM_IPV4)
    transport = TransportConfig(**info["contract"]["profile"]["transport"])
    if (transport != registered or old_transport != registered
            or (transport.provider, transport.base_url, transport.requested_model, transport.served_model,
                transport.reasoning_effort, transport.logical_replies, transport.transport_attempts,
                transport.max_output_tokens, transport.sdk_version)
            != ("token4research", "https://token4research.cn/v1", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh", 3, 2,
                32768, "2.28.0") or not sdk_status(transport)["matches"]):
        raise PermissionError("the new recovery must retain the exact previously approved SSE wire settings")
    if args.verify_only:
        return {"contract_sha256": args.contract_sha256, "seed": failed.seed,
                "round": failed.lane.round, "verified_no_dispatch": True}
    spool = spools / args.contract_sha256
    dispatcher = resilient.ResilientDispatcher(
        spool, info=info, contract_sha256=args.contract_sha256, transport=transport,
        mock=False, pool=resilient.SharedKeyPool(pilot.directory, pilot.config),
    )
    dispatcher.scope, dispatcher.original = proof, original
    dispatcher.prompt_sha = resilient._approved_prompt_sha(info)
    once(spool / "provider_model_lock.json", model)
    return run_dispatcher(dispatcher, courier, watch=True, workers=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--execute-paid", action="store_true")
    parser.add_argument("--seed", type=int, choices=tuple(FAILURES), required=True)
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
