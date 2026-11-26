"""Continue seed 2357 from a verified L397 extension with the original remaining V13 budget."""

import argparse
from dataclasses import asdict, replace
from functools import partial
import os
from pathlib import Path
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import terminal_record
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import (
    BridgePending, read_bridge_json, verify_job, verify_result,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, STUDY, verify_authorization
from . import run_v13_seed2357_extension as extension


SEED = extension.SEED
START_ROUND = extension.START_ROUND + 1
RUN_NAME = "a7-gpt62-astra-xhigh-ipv4-v13-2357-continuation-01"
DISPATCHER = "dispatch_prepared_ipv4_continue_2357"
EXTENSION_CONTRACT_SHA256 = "c07d946937930e575096e167fcdd80df3e261e2bb6f603350f4d6a447085d300"
EXTENSION_AUTHORIZATION_SHA256 = "5c3e30de81150b69ce4fe125bb760bcfcd776b86ed0099b1137c283612ac4785"


def _budget(parent, round_index):
    if round_index != START_ROUND:
        raise PermissionError("only a verified L397 boundary releases the unused V13 budget")
    remaining = parent.rounds - round_index
    logical = remaining * parent.transport.logical_replies
    return {"first_logical_requests": remaining, "maximum_logical_requests": logical,
            "maximum_physical_attempts": logical * parent.transport.transport_attempts}


def _extension_principal(contract, run):
    record = read(run / "formal_contract.json")
    approval = verify_authorization(contract, read(run / "authorization.json"))
    if (record != contract.record() or contract.digest != EXTENSION_CONTRACT_SHA256
            or digest(approval) != EXTENSION_AUTHORIZATION_SHA256):
        raise PermissionError("the deployed one-attempt extension contract or authorization differs")
    return approval


def _completed_attempt(run, contract, approval, active, request):
    """Link the validated selection to the sole real submission in the completed extension."""
    freeze = read(active / "batch_freeze.json")
    turn = active / "turn_00"
    if ((freeze["accepted_logical_index"], freeze["accepted_attempt"]) != (0, 0)
            or read(turn / "request.json") != request
            or set(active.glob("turn_*/attempt_*.started.json")) != {turn / "attempt_00.started.json"}):
        raise PermissionError("the extension selection must come from one exact physical attempt")

    started = read(turn / "attempt_00.started.json")
    terminal = terminal_record(turn, 0)
    name = f"{started['attempt_id']}.json"
    job = read_bridge_json(run / "bridge" / "outbox" / name)
    result = read_bridge_json(run / "bridge" / "results" / name)
    bound = verify_job(job, authorization_sha256=digest(approval), contract_sha256=contract.digest,
                       wire_representation=contract.profile["wire_representation"])
    reply = verify_result(result, job)
    if (bound != (request, contract.transport, started) or started["attempt"] != 0
            or any(terminal.get(key) != value for key, value in started.items())
            or terminal["status"] != "completed" or "reconciliation_sha256" in terminal
            or terminal.get("bridge_job_sha256") != job["job_sha256"]
            or terminal.get("bridge_result_sha256") != result["result_sha256"]
            or terminal["text"] != reply.text or terminal["response_id"] != reply.response_id
            or result["model_identity"] != contract.predecessor["model_identity"]):
        raise PermissionError("the extension freeze is detached from its completed Mac receipt")
    return {"extension_attempt_id": started["attempt_id"], "extension_job_sha256": job["job_sha256"],
            "extension_result_sha256": result["result_sha256"]}


def _completion(path, contract, state, memory, transition, budget):
    expected = {"contract_sha256": contract.digest, "seed": SEED, "round": START_ROUND, "active_labels": 397,
                "checkpoint_sha256": state.checkpoint, "state_sha256": digest(asdict(state)),
                "memory_sha256": digest(memory), "transition_sha256": file_digest(transition),
                "original_future_budget": budget}
    if read(path) != expected or state.seed != SEED or state.round != START_ROUND:
        raise PermissionError("the completed extension marker differs from the verified L397 transition")
    return expected


def _retirement(parent, *, expected=None):
    extension._parent_paused(parent)
    pause = seed_runner.seed_directory(parent, SEED) / "pause.request"
    evidence = file_digest(pause)
    if expected is not None and evidence != expected:
        raise PermissionError("the original V13 seed retirement marker changed")
    return evidence


def construct(root, original_release, parent_run, extension_release, extension_run, run):
    """Reconstruct the extension and authenticate its selection, CUDA fit and L397 state."""
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    if run.name != RUN_NAME or run.parent != parent_run.parent or root in (original_release, extension_release):
        raise PermissionError("use the registered continuation sibling and its own immutable release")
    retired = _retirement(parent_run)
    parent, parent_approval = extension._original_contract(parent_run)
    one_shot, boundary, cache, original = extension.construct(extension_release, original_release, parent_run, extension_run)
    one_shot_approval = _extension_principal(one_shot, extension_run)
    sources = dict(one_shot.source_sha256)
    if any(file_digest(root / name) != expected for name, expected in sources.items()):
        raise PermissionError("sources changed from the published extension release")
    for module in ("run_v13_seed2357_continuation", DISPATCHER):
        name = (STUDY / "scripts" / f"{module}.py").as_posix()
        sources[name] = file_digest(root / name)

    backend = TrainingBackend(root, cache, SEED)
    active = seed_runner.round_directory(seed_runner.seed_directory(extension_run, SEED), extension.START_ROUND)
    following = seed_runner.round_directory(active.parent, START_ROUND)
    scope = seed_runner._scope(one_shot, one_shot_approval, SEED, False)
    if (read(active / "source.json") != asdict(boundary[0]) or read(active / "boundary_memory.json") != boundary[1]
            or read(active.parent / "boundary.json") != {
                "formal_run": scope, **one_shot.profile["resume_boundaries"][str(SEED)],
            }):
        raise PermissionError("the extension's starting boundary differs from the original L365 state")
    request = extension._extension_request(extension_release, one_shot, one_shot_approval, boundary, original)
    attempt = _completed_attempt(extension_run, one_shot, one_shot_approval, active, request)
    state, memory = seed_runner._successor(active, boundary[0], following, scope, one_shot.transport, one_shot, backend)
    budget = _budget(parent, state.round)
    marker = _completion(extension_run / "extension_completion.json", one_shot, state, memory,
                         following / "transition.json", budget)
    lock = read(extension_run / "provider_model_lock.json")
    if lock != one_shot.predecessor["model_identity"]:
        raise PermissionError("the completed extension has a different served model lock")

    profile = {
        **parent.profile, "confirmation_seeds": [SEED], "qualification_seed": SEED,
        "requires_initial_qualification": False, "api_parallelism": 1,
        "resume_boundaries": {str(SEED): {"round": START_ROUND, "state_sha256": marker["state_sha256"],
                                         "memory_sha256": marker["memory_sha256"]}},
    }
    predecessor = {
        "dispatcher_module": DISPATCHER, "parent_run_directory": str(parent_run),
        "parent_release_directory": str(original_release), "parent_contract_sha256": parent.digest,
        "parent_authorization_sha256": digest(parent_approval), "parent_seed_pause_sha256": retired,
        "extension_run_directory": str(extension_run), "extension_release_directory": str(extension_release),
        "extension_contract_sha256": one_shot.digest, "extension_authorization_sha256": digest(one_shot_approval),
        "extension_completion_sha256": digest(marker), "extension_transition_sha256": marker["transition_sha256"],
        "extension_fit_audit_sha256": state.provenance["fit_audit_sha256"], **attempt,
        "seed": SEED, "round": START_ROUND, "checkpoint_sha256": state.checkpoint,
        "cache_directory": str(cache), "public_resource_receipt_sha256": backend.cache_receipt_sha256,
        "model_identity": lock, "execution_budget": budget,
    }
    contract = replace(parent, profile=profile, source_sha256=sources, run_directory=str(run), predecessor=predecessor)
    return contract, (state, memory, START_ROUND), backend


def execute(root, parent_run, run, contract, approval, boundary, backend, *, resume):
    _retirement(parent_run, expected=contract.predecessor["parent_seed_pause_sha256"])
    with scheduler._lock(run / "coordinator.lock", busy="the seed 2357 continuation is already running"):
        if scheduler._pause_requested(run) and not resume:
            raise PermissionError("the seed 2357 continuation is paused; use resume")
        once(run / "formal_contract.json", contract.record())
        once(run / "authorization.json", approval)
        once(run / "provider_model_lock.json", contract.predecessor["model_identity"])
        status_path = run / "workers" / f"seed_{SEED}.json"
        status = {"seed": SEED, "pid": os.getpid(), "gpu": contract.gpu_slots[0]}
        with scheduler._lock(run / "workers" / f"seed_{SEED}.lock", busy="the seed 2357 continuation still has a worker"):
            scheduler._atomic_status(run / "coordinator_generation.json", {"id": uuid4().hex, "pid": os.getpid()})
            if resume:
                (run / "pause.request").unlink(missing_ok=True)
                (seed_runner.seed_directory(run, SEED) / "pause.request").unlink(missing_ok=True)
            return _run_seed(root, parent_run, run, contract, approval, boundary, backend, status_path, status)


def _run_seed(root, parent_run, run, contract, approval, boundary, backend, status_path, status):
    scheduler._atomic_status(status_path, {**status, "status": "initializing"})
    try:
        call, options = scheduler._provider_wiring(run, contract, approval, None, SEED)
        managed = options["managed_attempt"]

        def continuation_attempt(request, config, started):
            _retirement(parent_run, expected=contract.predecessor["parent_seed_pause_sha256"])
            binding = request["binding"]
            if binding["seed"] != SEED or not START_ROUND <= binding["round"] < contract.rounds:
                raise PermissionError("the continuation cannot submit a previously completed or exhausted round")
            return managed(request, config, started)

        result = seed_runner.run_seed(
            root=root, run_directory=run, contract=contract, authorization=approval,
            seed=SEED, backend=backend, call=call, resume_boundary=boundary,
            transport_options={"managed_attempt": continuation_attempt},
            fit_slot=partial(scheduler.gpu_slot, run, SEED, status["gpu"], contract.lease_root),
        )
        scheduler._atomic_status(status_path, {**status, **asdict(result)})
        return result
    except Exception as error:
        phase = "paused" if scheduler._pause_requested(run, SEED) else (
            "bridge_pending" if isinstance(error, BridgePending) else "failed")
        scheduler._atomic_status(status_path, {**status, "status": phase, "error_type": type(error).__name__,
                                               "reason": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run", "resume"))
    for name in ("original-release", "original-run", "extension-release", "extension-run", "run-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
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
    contract, boundary, backend = construct(ROOT, original_release, parent, extension_release, extension_run, run)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        parser.error("run and resume need the exact continuation authorization file")
    approval = verify_authorization(contract, read(args.authorization))
    result = execute(ROOT, parent, run, contract, approval, boundary, backend, resume=args.action == "resume")
    print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
