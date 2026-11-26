"""Continue either V13 lane after its one-shot extension seals the next fitted state."""

import argparse
from dataclasses import asdict, replace
from functools import partial
import os
from pathlib import Path
import sys
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import BridgePending
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, STUDY, verify_authorization
from . import run_v13_lane_extension as extension
from .run_prepared_five_seed import SHA256
from .run_v13_seed2357_continuation import _completed_attempt


DISPATCHER = "dispatch_v13_lane_continuation"


def _budget(parent, lane, round_index):
    if round_index != lane.round + 1:
        raise PermissionError("the continuation needs the next fitted round from its one-shot extension")
    remaining = parent.rounds - round_index
    logical = remaining * parent.transport.logical_replies
    return {"first_logical_requests": remaining, "maximum_logical_requests": logical,
            "maximum_physical_attempts": logical * parent.transport.transport_attempts}


def _retirement(parent, lane, *, expected=None):
    seed, _ = extension._uncommitted(parent, lane, require_paused=True)
    evidence = file_digest(seed / "pause.request")
    if expected is not None and evidence != expected:
        raise PermissionError("the original V13 seed retirement marker changed")
    return evidence


def _extension_approval(contract, run, expected_contract, expected_approval):
    if not all(SHA256.fullmatch(value) for value in (expected_contract, expected_approval)):
        raise PermissionError("supply the exact released extension and authorization SHA-256 values")
    record = read(run / "formal_contract.json")
    approval = verify_authorization(contract, read(run / "authorization.json"))
    if record != contract.record() or contract.digest != expected_contract or digest(approval) != expected_approval:
        raise PermissionError("the sealed one-shot contract or authorization differs from the expected release")
    return approval


def _require_fixed_reconstruction(lane):
    if lane.seed == 887 and (os.environ.get("PYTHONHASHSEED") != "0" or sys.flags.hash_randomization):
        raise PermissionError("start this 887 continuation with PYTHONHASHSEED=0 to reproduce the signed request checks")


def _reconstruction_proof(one_shot, lane):
    proof = one_shot.predecessor.get("authoritative_message_reconstruction")
    if proof is None:
        return {}
    if (lane.seed != 887 or type(proof) is not dict or set(proof) != {"archived_sha256", "rebuilt_sha256"}
            or any(not isinstance(value, str) or not SHA256.fullmatch(value) for value in proof.values())
            or proof["archived_sha256"] == proof["rebuilt_sha256"]):
        raise PermissionError("the one-shot contract has an invalid host serialization proof")
    return {"authoritative_message_reconstruction": proof}


def _completion(path, contract, lane, state, memory, transition, budget):
    expected = {"contract_sha256": contract.digest, "seed": lane.seed, "round": lane.round + 1,
                "active_labels": contract.target_labels, "checkpoint_sha256": state.checkpoint,
                "state_sha256": digest(asdict(state)), "memory_sha256": digest(memory),
                "transition_sha256": file_digest(transition), "original_future_budget": budget}
    if read(path) != expected or state.seed != lane.seed or state.round != lane.round + 1:
        raise PermissionError("the completed extension marker differs from the verified CUDA transition")
    return expected


def construct(root, original_release, parent_run, extension_release, extension_run, run, lane, *,
              expected_extension_contract, expected_extension_approval):
    """Authenticate the exact child request, legal freeze, fit lineage and post-fit state."""
    _require_fixed_reconstruction(lane)
    run_name = f"{extension.previous.PARENT_NAME}-{lane.seed}-continuation-01"
    if run.name != run_name or run.parent != parent_run.parent or root in (original_release, extension_release):
        raise PermissionError("use this seed's registered continuation sibling and new immutable release")
    retired = _retirement(parent_run, lane)
    parent, parent_approval = extension.previous._original_contract(parent_run)
    one_shot, prior, backend, original = extension.construct(
        extension_release, original_release, parent_run, extension_run, lane,
    )
    approval = _extension_approval(one_shot, extension_run, expected_extension_contract, expected_extension_approval)
    reconstruction = _reconstruction_proof(one_shot, lane)
    sources = dict(one_shot.source_sha256)
    if any(file_digest(root / name) != expected for name, expected in sources.items()):
        raise PermissionError("sources changed from the verified one-shot release")
    for module in ("run_v13_lane_continuation", DISPATCHER, "run_v13_seed2357_continuation"):
        name = (STUDY / "scripts" / f"{module}.py").as_posix()
        sources[name] = file_digest(root / name)

    active = seed_runner.round_directory(seed_runner.seed_directory(extension_run, lane.seed), lane.round)
    following = seed_runner.round_directory(active.parent, lane.round + 1)
    scope = seed_runner._scope(one_shot, approval, lane.seed, False)
    if (read(active / "source.json") != asdict(prior[0]) or read(active / "boundary_memory.json") != prior[1]
            or read(active.parent / "boundary.json") != {
                "formal_run": scope, **one_shot.profile["resume_boundaries"][str(lane.seed)],
            }):
        raise PermissionError("the extension's starting boundary differs from the original V13 state")
    request = extension._request(extension_release, one_shot, approval, lane, prior[0], prior[1], original)
    attempt = _completed_attempt(extension_run, one_shot, approval, active, request)
    state, memory = seed_runner._successor(active, prior[0], following, scope, one_shot.transport, one_shot, backend)
    budget = _budget(parent, lane, state.round)
    marker = _completion(extension_run / "extension_completion.json", one_shot, lane, state, memory,
                         following / "transition.json", budget)
    lock = read(extension_run / "provider_model_lock.json")
    if lock != one_shot.predecessor["model_identity"]:
        raise PermissionError("the completed extension has a different served model lock")

    transport = replace(one_shot.transport, logical_replies=parent.transport.logical_replies,
                        transport_attempts=parent.transport.transport_attempts)
    profile = {
        **parent.profile, "confirmation_seeds": [lane.seed], "qualification_seed": lane.seed,
        "requires_initial_qualification": False, "api_parallelism": 1, "transport": asdict(transport),
        "request_timeouts": one_shot.profile["request_timeouts"],
        "resume_boundaries": {str(lane.seed): {"round": state.round, "state_sha256": marker["state_sha256"],
                                               "memory_sha256": marker["memory_sha256"]}},
    }
    predecessor = {
        "dispatcher_module": DISPATCHER, "parent_run_directory": str(parent_run),
        "parent_release_directory": str(original_release), "parent_contract_sha256": parent.digest,
        "parent_authorization_sha256": digest(parent_approval), "parent_seed_pause_sha256": retired,
        "extension_run_directory": str(extension_run), "extension_release_directory": str(extension_release),
        "extension_contract_sha256": one_shot.digest, "extension_authorization_sha256": digest(approval),
        "extension_completion_sha256": digest(marker), "extension_transition_sha256": marker["transition_sha256"],
        "extension_fit_audit_sha256": state.provenance["fit_audit_sha256"], **attempt,
        "seed": lane.seed, "round": state.round, "checkpoint_sha256": state.checkpoint,
        "cache_directory": str(backend.cache_dir), "public_resource_receipt_sha256": backend.cache_receipt_sha256,
        "model_identity": lock, "execution_budget": budget,
        "bridge_wait_seconds": one_shot.predecessor["bridge_wait_seconds"], **reconstruction,
    }
    contract = replace(parent, profile=profile, transport=transport, source_sha256=sources,
                       run_directory=str(run), predecessor=predecessor)
    return contract, (state, memory, state.round), backend


def execute(root, parent, run, lane, contract, approval, boundary, backend, *, resume):
    _require_fixed_reconstruction(lane)
    _retirement(parent, lane, expected=contract.predecessor["parent_seed_pause_sha256"])
    with scheduler._lock(run / "coordinator.lock", busy="this seed continuation is already running"):
        if scheduler._pause_requested(run) and not resume:
            raise PermissionError("this continuation is paused; use resume")
        once(run / "formal_contract.json", contract.record())
        once(run / "authorization.json", approval)
        once(run / "provider_model_lock.json", contract.predecessor["model_identity"])
        if resume:
            (run / "pause.request").unlink(missing_ok=True)
            (seed_runner.seed_directory(run, lane.seed) / "pause.request").unlink(missing_ok=True)
        scheduler._atomic_status(run / "coordinator_generation.json", {"id": uuid4().hex, "pid": os.getpid()})
        return _run(root, parent, run, lane, contract, approval, boundary, backend)


def _run(root, parent, run, lane, contract, approval, boundary, backend):
    status = {"seed": lane.seed, "pid": os.getpid(), "gpu": contract.gpu_slots[0]}
    path = run / "workers" / f"seed_{lane.seed}.json"
    scheduler._atomic_status(path, {**status, "status": "initializing"})
    try:
        call, options = scheduler._provider_wiring(run, contract, approval, None, lane.seed)
        managed = options["managed_attempt"]
        managed.wait_seconds = contract.predecessor["bridge_wait_seconds"]

        def continuation_attempt(request, config, started):
            _retirement(parent, lane, expected=contract.predecessor["parent_seed_pause_sha256"])
            binding = request["binding"]
            if binding["seed"] != lane.seed or not lane.round < binding["round"] < contract.rounds:
                raise PermissionError("this continuation cannot submit a completed or exhausted round")
            return managed(request, config, started)

        result = seed_runner.run_seed(
            root=root, run_directory=run, contract=contract, authorization=approval, seed=lane.seed, backend=backend,
            call=call, resume_boundary=boundary, transport_options={"managed_attempt": continuation_attempt},
            fit_slot=partial(scheduler.gpu_slot, run, lane.seed, status["gpu"], contract.lease_root),
        )
        scheduler._atomic_status(path, {**status, **asdict(result)})
        return result
    except Exception as error:
        phase = "paused" if scheduler._pause_requested(run, lane.seed) else (
            "bridge_pending" if isinstance(error, BridgePending) else "failed")
        scheduler._atomic_status(path, {**status, "status": phase, "error_type": type(error).__name__,
                                        "reason": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run", "resume"))
    parser.add_argument("--seed", type=int, choices=tuple(extension.LANES), required=True)
    for name in ("original-release", "original-run", "extension-release", "extension-run", "run-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--extension-contract-sha256", required=True)
    parser.add_argument("--extension-authorization-sha256", required=True)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    original_release, parent, extension_release, extension_run, run = (
        path.resolve() for path in (args.original_release, args.original_run, args.extension_release,
                                    args.extension_run, args.run_dir)
    )
    lane = extension.LANES[args.seed]
    contract, boundary, backend = construct(
        ROOT, original_release, parent, extension_release, extension_run, run, lane,
        expected_extension_contract=args.extension_contract_sha256,
        expected_extension_approval=args.extension_authorization_sha256,
    )
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        parser.error("run and resume require the exact continuation authorization")
    approval = verify_authorization(contract, read(args.authorization))
    result = execute(ROOT, parent, run, lane, contract, approval, boundary, backend, resume=args.action == "resume")
    print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
