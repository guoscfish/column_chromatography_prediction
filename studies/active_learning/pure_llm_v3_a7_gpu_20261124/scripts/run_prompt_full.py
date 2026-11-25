"""Continue a verified revised-prompt seed from L653 to L1485 on the shared GPU."""
import argparse
from dataclasses import asdict, replace
import multiprocessing
import os
from pathlib import Path
import time
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.memory_manager import MemoryManager
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, SSH_API, STUDY, verify_authorization

from . import run_prompt_pilot as pilot

RUN_PREFIX = "a7-gpt62-astra-xhigh-ssh-v6-prompt-full"
START_ROUND = 10
START_LABELS = 653
TARGET_LABELS = 1485


def construct(root, old_release, old_run, first_release, first_run, parent_release, parent_run, run, seed):
    """Bind the new run to a verified completed V5 prefix and its last scientific memory."""
    if (seed not in pilot.PHASE_BOUNDARIES or run.name != f"{RUN_PREFIX}-{seed}"
            or run.parent != old_run.parent or parent_run == run):
        raise ValueError("the full continuation needs its own registered persistent run directory")

    directory = seed_runner.seed_directory(parent_run, seed)
    endpoint = seed_runner.round_directory(directory, START_ROUND)
    if not (endpoint / "transition.json").is_file():
        raise ValueError("the revised-prompt seed has not reached its sealed L653 endpoint")
    if (endpoint / "batch_freeze.json").exists():
        raise ValueError("the parent has already started acquisitions past its L653 endpoint")

    # These are read on the verification path and must already be sealed.
    cache = old_run / "public_graphs" / f"seed_{seed}"
    old_initial = seed_runner.seed_directory(old_run, seed) / "initial"
    if not all(path.is_file() for path in (cache / "round.lock", cache / "resource_receipt.json",
                                           old_initial / "initial.json", old_initial / "source.json")):
        raise ValueError("read-only verification requires the existing sealed parent resources")

    parent, boundaries = pilot.construct(parent_release, old_release, old_run, first_release, first_run, parent_run, seed)
    parent_record = read(parent_run / "formal_contract.json")
    if (parent.record() != parent_record or parent.api_execution != SSH_API or parent.gpu_slots != (0,)
            or parent.initial_labels != 333 or parent.batch_size != 32 or parent.target_labels != START_LABELS
            or parent.rounds != START_ROUND or parent.requires_initial_qualification
            or parent.transport.transport_attempts != 2):
        raise PermissionError("the parent is outside the reviewed revised-prompt single-GPU pilot")
    parent_approval = verify_authorization(parent, read(parent_run / "authorization.json"))
    parent_lock = read(parent_run / "provider_model_lock.json")
    if parent_lock != read(first_run / "provider_model_lock.json"):
        raise ValueError("verified provider identity differs across the revised-prompt segments")

    state, memory, first_round = boundaries[seed]
    boundary_scope = seed_runner._scope(parent, parent_approval, seed, False)
    parent_boundary = parent.profile["resume_boundaries"][str(seed)]
    imported = seed_runner.round_directory(directory, first_round)
    if (read(directory / "boundary.json") != {"formal_run": boundary_scope, **parent_boundary}
            or read(imported / "source.json") != asdict(state)
            or read(imported / "boundary_memory.json") != memory):
        raise ValueError("the sealed V5 start differs from its original completed boundary")

    backend = TrainingBackend(old_release, cache, seed)
    transitions = {}
    for index in range(first_round, START_ROUND):
        state, memory, transition = pilot._take_transition(parent_run, index, seed, state, backend,
                                                           parent, parent_approval)
        transitions[str(index + 1)] = transition

    MemoryManager.from_snapshot(memory, seed=seed)
    if (state.round != START_ROUND or len(state.provenance["expected_observed"]) != START_LABELS
            or set(memory["rounds"]) != {str(index) for index in range(START_ROUND)}):
        raise ValueError("the L653 boundary lacks the complete ten-round scientific lineage")

    descriptor = {"round": START_ROUND, "state_sha256": digest(asdict(state)), "memory_sha256": digest(memory)}
    profile = {**parent.profile, "confirmation_seeds": [seed], "target_labels": TARGET_LABELS,
               "resume_boundaries": {str(seed): descriptor}}
    if any(file_digest(root / name) != expected for name, expected in parent.source_sha256.items()):
        raise PermissionError("the full release changed an approved source or the revised prompt")
    wrapper_path = (STUDY / "scripts/run_prompt_full.py").as_posix()
    sources = {**parent.source_sha256, wrapper_path: file_digest(root / wrapper_path)}
    remaining = (TARGET_LABELS - START_LABELS) // parent.batch_size
    logical = remaining * parent.transport.logical_replies
    evidence = {
        "pilot_contract_sha256": parent.predecessor["pilot_contract_sha256"],
        "contract_sha256": parent.digest, "authorization_sha256": digest(parent_approval),
        "run_directory": str(parent_run), "start_label_budget": START_LABELS,
        "checkpoint_sha256": state.checkpoint, "transition_sha256": transitions[str(START_ROUND)],
        "verified_v5_transition_sha256_by_completed_round": transitions,
        "provider_model_lock_sha256": digest(parent_lock),
        "extension_budget": {"first_logical_requests": remaining, "maximum_logical_requests": logical,
                             "maximum_physical_attempts": logical * parent.transport.transport_attempts},
        "uses_previous_incomplete_round_responses": False,
    }
    contract = replace(parent, profile=profile, source_sha256=sources, run_directory=str(run), predecessor=evidence)
    return contract, (state, memory, START_ROUND)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run", "resume"))
    for name in ("old-release", "old-run", "first-release", "first-run", "parent-release", "parent-run", "run-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=tuple(pilot.PHASE_BOUNDARIES), required=True)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    run, old_run, parent_run = args.run_dir.resolve(), args.old_run.resolve(), args.parent_run.resolve()
    contract, boundary = construct(ROOT, args.old_release.resolve(), old_run, args.first_release.resolve(),
                                   args.first_run.resolve(), args.parent_release.resolve(), parent_run, run, args.seed)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        raise PermissionError("the full continuation requires exact contract authorization")
    approval = verify_authorization(contract, read(args.authorization))
    ctx = multiprocessing.get_context("spawn")
    with scheduler._lock(run / "coordinator.lock", busy="the full continuation coordinator is already running"):
        scheduler._assert_no_old_workers(run, contract.seed_ids)
        if scheduler._pause_requested(run) and args.action != "resume":
            raise PermissionError("the full continuation is paused; use resume")
        once(run / "formal_contract.json", contract.record())
        once(run / "authorization.json", approval)
        model_lock = read(parent_run / "provider_model_lock.json")
        if digest(model_lock) != contract.predecessor["provider_model_lock_sha256"]:
            raise ValueError("the parent provider model lock changed during verification")
        once(run / "provider_model_lock.json", model_lock)
        generation = uuid4().hex
        once(run / "events" / f"{generation}.json", {"action": args.action, "timestamp_unix": time.time(),
                                                     "authorization_sha256": digest(approval)})
        scheduler._atomic_status(run / "coordinator_generation.json", {"id": generation, "pid": os.getpid()})
        if args.action == "resume":
            (run / "pause.request").unlink(missing_ok=True)
        child = ctx.Process(target=pilot.worker,
                            args=(ROOT, run, old_run, contract, approval, args.seed, boundary, generation),
                            name=f"a7-full-seed-{args.seed}")
        child.start()
        child.join()
        print(dumps({"seed": args.seed, "target_labels": TARGET_LABELS, "worker_exit": child.exitcode}))
        raise SystemExit(int(child.exitcode != 0))


if __name__ == "__main__":
    main()
