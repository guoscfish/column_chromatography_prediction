"""Resume ordinary A7 at the verified 6101 R5/L493 zero-POST handover."""

import argparse
from dataclasses import asdict, replace
from functools import partial
import os
from pathlib import Path
import sys
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import make_request, terminal_record
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import BridgePending
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, STUDY, verify_authorization
from . import run_v13_6101_cached_r4 as cached


SEED, START = cached.SEED, cached.ROUND + 1
RUN_NAME = f"{cached.relay.bounded.PREFIX}-6101-r5-continuation-01"
DISPATCHER = "dispatch_v13_6101_r5_continuation"
BOUNDARY_BUDGET = {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
BUDGET = {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182}
SOURCES = tuple(STUDY / "scripts" / f"{module}.py" for module in (
    cached.MODULE, DISPATCHER, "run_v13_6101_r5_continuation",
))


def _original_profile(parent, descriptor):
    return {**parent.profile, "resume_boundaries": {str(SEED): descriptor}, "replay_eligible_from_round": START,
            "sealed_relay_retry_seconds": 180, "audited_future_transport_replay": True}


def _require_complete(run):
    worker = read(run / "workers" / f"seed_{SEED}.json")
    if (worker.get("status"), worker.get("completed_rounds"), worker.get("active_labels")) != (
            "complete", START, cached.TARGET):
        raise PermissionError("the zero-POST cached run must stop at a fully fitted L493")
    cached._require_no_bridge(run)
    scheduler._assert_no_old_workers(run, (SEED,))


def _completed_local(run, active, imported, approval, expected, plan):
    custody = imported.predecessor["cached_selection"]
    artifact = run / "cached_response_import.json"
    scope = seed_runner._scope(imported, approval, SEED, False)
    turn = active / "turn_00"
    started = read(turn / "attempt_00.started.json")
    terminal = terminal_record(turn, 0)
    grant = seed_runner._selection_grant(scope, expected["request_id"])
    if (read(artifact) != custody or custody.get("provider_post") is not False
            or expected["request_id"] == cached.REQUEST or read(turn / "request.json") != expected
            or read(turn / "formal_request_authorization.json") != grant or read(active / "authorization.json") != grant
            or set(active.glob("turn_*/attempt_*.started.json")) != {turn / "attempt_00.started.json"}
            or terminal.get("request_id") != expected["request_id"] or terminal.get("status") != "completed"
            or terminal.get("provider_dispatched") is not False or terminal.get("usage") != {}
            or terminal.get("cached_selection_sha256") != digest(custody)
            or terminal.get("imported_from_response_id") != cached.RESPONSE or terminal.get("response_id") != cached.RESPONSE
            or terminal.get("answer_sha256") != digest(dumps(plan)) or terminal.get("text") != dumps(plan)
            or any(terminal.get(key) != value for key, value in started.items())):
        raise PermissionError("the local terminal is detached from the exact cached R4 answer")
    return {"cached_local_attempt_id": started["attempt_id"],
            "cached_local_terminal_sha256": file_digest(turn / "attempt_00.terminal.json"),
            "cached_selection_file_sha256": file_digest(artifact), "cached_selection_sha256": digest(custody),
            "reviewed_cached_selection": custody}


def construct(root, old_release, old_run, imported_release, imported_run, run, *, imported_contract, imported_auth):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    if (root in (old_release, imported_release) or run.name != RUN_NAME or imported_run.name != cached.RUN_NAME
            or len({old_run.parent, imported_run.parent, run.parent}) != 1):
        raise PermissionError("the ordinary continuation requires a separate release and run")
    _require_complete(imported_run)
    parent, parent_approval = cached.relay.bounded._archived(old_run, old_release, cached.OLD)
    imported, boundary, cache, archived, _, plan = cached.construct(imported_release, old_release, old_run, imported_run)
    approval = verify_authorization(imported, read(imported_run / "authorization.json"))
    if (imported.digest != imported_contract or digest(approval) != imported_auth
            or read(imported_run / "formal_contract.json") != imported.record()):
        raise PermissionError("the sealed zero-POST contract or authorization differs from its approved identity")
    if imported.record()["budget"] != cached.BUDGET or imported.target_labels != cached.TARGET:
        raise PermissionError("the original R4 cache has to be terminal-only with zero provider demand")
    sources = {name: file_digest(root / name) for name in imported.source_sha256}
    if sources != imported.source_sha256:
        raise PermissionError("the continuation must preserve every byte of the ordinary W08 implementation and Prompt")
    for source in SOURCES:
        sources[source.as_posix()] = file_digest(root / source)

    directory = seed_runner.seed_directory(imported_run, SEED)
    active, following = (seed_runner.round_directory(directory, index) for index in (cached.ROUND, START))
    state, memory, _ = boundary
    scope = seed_runner._scope(imported, approval, SEED, False)
    if (read(directory / "boundary.json") != {"formal_run": scope, **imported.profile["resume_boundaries"][str(SEED)]}
            or read(active / "source.json") != asdict(state) or read(active / "boundary_memory.json") != memory
            or read(imported_run / "provider_model_lock.json") != imported.predecessor["model_identity"]):
        raise PermissionError("the exact original training boundary or model lock changed")
    user = archived["payload"]["input"][1]
    stage = seed_runner.pin_boundary_user_message(seed_runner._stage(root, state, memory, scope, imported), user, digest(user))
    expected = make_request(stage.binding, 0, stage.messages, imported.transport)
    if expected["payload"] != archived["payload"]:
        raise PermissionError("the local cache did not reuse the original complete R4 payload")
    custody = _completed_local(imported_run, active, imported, approval, expected, plan)
    freeze = seed_runner._verify_selection(active, state, scope, imported.transport, imported.batch_size, stage=stage)
    if (freeze["plan"] != plan or digest(freeze["batch_ids"]) != cached.FROZEN_ORDER
            or (freeze["accepted_logical_index"], freeze["accepted_attempt"]) != (0, 0)):
        raise PermissionError("the ordered frozen batch differs from the legal cached projection")
    backend = TrainingBackend(root, cache, SEED)
    next_state, next_memory = seed_runner._successor(active, state, following, scope, imported.transport, imported, backend)
    if (next_state.seed, next_state.round, len(next_state.provenance["expected_observed"])) != (SEED, START, cached.TARGET):
        raise PermissionError("the verified CUDA transition must be exactly 32 new labels from R4")
    descriptor = {"round": START, "state_sha256": digest(asdict(next_state)), "memory_sha256": digest(next_memory)}
    proof = {
        "dispatcher_module": DISPATCHER, "seed": SEED, "round": START,
        "parent_run_directory": str(old_run), "parent_release_directory": str(old_release),
        "parent_contract_sha256": parent.digest, "parent_authorization_sha256": digest(parent_approval),
        "parent_seed_pause_sha256": cached.PAUSE_SHA, "extension_run_directory": str(imported_run),
        "extension_release_directory": str(imported_release), "extension_contract_sha256": imported.digest,
        "extension_authorization_sha256": digest(approval), **custody,
        "extension_transition_sha256": file_digest(following / "transition.json"),
        "extension_source_sha256": file_digest(following / "source.json"),
        "extension_fit_audit_sha256": next_state.provenance["fit_audit_sha256"],
        "extension_freeze_sha256": freeze["freeze_sha256"], "checkpoint_sha256": next_state.checkpoint,
        "cache_directory": str(cache), "public_resource_receipt_sha256": parent.predecessor["public_resource_receipt_sha256"],
        "model_identity": imported.predecessor["model_identity"], "bridge_wait_seconds": parent.predecessor["bridge_wait_seconds"],
        "recovery_budget": BOUNDARY_BUDGET, "execution_budget": BUDGET,
    }
    profile = _original_profile(parent, descriptor)
    contract = replace(parent, profile=profile, source_sha256=sources, run_directory=str(run), predecessor=proof)
    return contract, (next_state, next_memory, START), backend


def execute(root, old, imported, run, contract, approval, boundary, backend, *, resume):
    proof = contract.predecessor
    gpu = contract.gpu_slots[0]
    directory = seed_runner.seed_directory(run, SEED)
    status = {"seed": SEED, "pid": os.getpid(), "gpu": gpu}
    worker = run / f"workers/seed_{SEED}.json"
    with scheduler._lock(imported / "coordinator.lock", busy="the zero-POST child still owns its worker"):
        _require_complete(imported)
        cached._sealed_parent(old, require_pause=True)
        scheduler._assert_no_old_workers(old, (SEED,))
        with scheduler._lock(run / "coordinator.lock", busy="the R5 continuation already has a coordinator"):
            if scheduler._pause_requested(run, SEED) and not resume:
                raise PermissionError("this continuation is paused; use resume")
            once(run / "formal_contract.json", contract.record())
            once(run / "authorization.json", approval)
            once(run / "provider_model_lock.json", proof["model_identity"])
            if resume:
                (run / "pause.request").unlink(missing_ok=True)
                (directory / "pause.request").unlink(missing_ok=True)
            scheduler._atomic_status(run / "coordinator_generation.json", {"id": uuid4().hex, "pid": os.getpid()})
            scheduler._atomic_status(worker, {**status, "status": "initializing"})
            try:
                call, options = scheduler._provider_wiring(run, contract, approval, None, SEED)
                managed = options["managed_attempt"]
                managed.wait_seconds = proof["bridge_wait_seconds"]

                def checked(request, config, started):
                    binding = request["binding"]
                    if (binding["seed"] != SEED or not START <= binding["round"] < contract.rounds
                            or binding["round"] == START and (request["logical_index"] != 0
                                or binding["checkpoint_sha256"] != proof["checkpoint_sha256"] or started["attempt"] not in (0, 1))):
                        raise PermissionError("the continuation cannot exceed its signed R5 boundary")
                    return managed(request, config, started)

                result = seed_runner.run_seed(
                    root=root, run_directory=run, contract=contract, authorization=approval, seed=SEED,
                    backend=backend, call=call, resume_boundary=boundary, transport_options={"managed_attempt": checked},
                    fit_slot=partial(scheduler.gpu_slot, run, SEED, gpu, contract.lease_root),
                )
                scheduler._atomic_status(worker, {**status, **asdict(result)})
                return result
            except Exception as error:
                phase = "paused" if scheduler._pause_requested(run, SEED) else (
                    "bridge_pending" if isinstance(error, BridgePending) else "failed")
                scheduler._atomic_status(worker, {**status, "status": phase, "error_type": type(error).__name__})
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run", "resume"))
    for name in ("old-release", "old-run", "cached-release", "cached-run", "run-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--cached-contract-sha256", required=True)
    parser.add_argument("--cached-authorization-sha256", required=True)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0" or sys.flags.hash_randomization:
        parser.error("run with PYTHONHASHSEED=0 to preserve the exact W08 request reconstruction")
    os.umask(0o077)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    old_release, old, imported_release, imported, run = (p.resolve() for p in (
        args.old_release, args.old_run, args.cached_release, args.cached_run, args.run_dir,
    ))
    contract, boundary, backend = construct(ROOT, old_release, old, imported_release, imported, run,
                                            imported_contract=args.cached_contract_sha256,
                                            imported_auth=args.cached_authorization_sha256)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        parser.error("provide the exact ordinary continuation authorization")
    approval = verify_authorization(contract, read(args.authorization))
    result = execute(ROOT, old, imported, run, contract, approval, boundary, backend, resume=args.action == "resume")
    print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
