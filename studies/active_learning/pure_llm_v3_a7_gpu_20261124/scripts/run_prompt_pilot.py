"""Continue the three A7 pilot trajectories with the revised prompt and sealed checkpoints."""
import argparse
from dataclasses import asdict, replace
from functools import partial
import multiprocessing
import os
from pathlib import Path
import time
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    PROFILE_V4_PILOT, ROOT, SSH_API, STUDY, load_contract, verify_authorization,
)
from . import run_prompt_successor as first

RUN_PREFIX = "a7-gpt62-astra-xhigh-ssh-v5-prompt-continuation"
PHASE_BOUNDARIES = {157: 1, 887: 4, 2357: 1}


def _take_transition(run, index, seed, state, backend, contract, approval):
    directory = seed_runner.seed_directory(run, seed)
    before, after = (seed_runner.round_directory(directory, n) for n in (index, index + 1))
    scope = seed_runner._scope(contract, approval, seed, False)
    next_state, snapshot = seed_runner._successor(before, state, after, scope, contract.transport, contract, backend)
    return next_state, snapshot, file_digest(after / "transition.json")


def construct(root, old_release, old_run, first_release, first_run, run, seed):
    if run.name != f"{RUN_PREFIX}-{seed}" or run.parent != old_run.parent or first_run.parent != old_run.parent:
        raise ValueError("the continuation requires its own registered persistent run directory")
    record = read(old_run / "formal_contract.json")
    base = load_contract(old_release, old_release / PROFILE_V4_PILOT, anchor_root=record["anchor_root"],
                         gpu_slots=tuple(record["gpu_slots"]), lease_root=record["lease_root"],
                         run_directory=old_run, api_execution=SSH_API,
                         predecessor_run=record["predecessor"]["run_directory"])
    if base.digest != first.PILOT_SHA or base.record() != record or not (old_run / "pause.request").is_file():
        raise PermissionError("the original pilot must have the reviewed identity and remain paused")
    old_approval = verify_authorization(base, read(old_run / "authorization.json"))
    first_record = read(first_run / "formal_contract.json")
    first_approval_record = read(first_run / "authorization.json")
    if (digest(first_record) != first_approval_record["contract_sha256"]
            or first_record["predecessor"]["contract_sha256"] != base.digest):
        raise PermissionError("the previous prompt authorization differs from the original pilot")
    prompt = seed_runner.PROMPT_PATH
    if ((root / prompt).read_text().strip() != (old_release / prompt).read_text().strip() + "\n\n" + first.PROMPT_UPDATE
            or (root / prompt).read_bytes() != (first_release / prompt).read_bytes()):
        raise ValueError("the continuation must use the same revised system prompt")
    sources = {name: file_digest(root / name) for name in first_record["source_sha256"]}
    if sources != first_record["source_sha256"]:
        raise ValueError("the continuation changes sources outside its coordinator and dispatcher")
    for name in ("run_prompt_pilot.py", "dispatch_prompt_pilot.py"):
        path = (STUDY / "scripts" / name).as_posix()
        sources[path] = file_digest(root / path)

    backend = TrainingBackend(old_release, old_run / "public_graphs" / f"seed_{seed}", seed)
    if seed == first.SEED:
        first_contract, state, _, _ = first._contract(first_release, old_release, old_run, first_run)
        if first_contract.record() != first_record:
            raise ValueError("the revised fourth round contract differs from its source manifest")
        first_approval = verify_authorization(first_contract, first_approval_record)
        state, snapshot, transition = _take_transition(first_run, first.INDEX, seed, state,
                                                       backend, first_contract, first_approval)
        source_contract, source_run = first_contract, first_run
    else:
        old_seed = seed_runner.seed_directory(old_run, seed)
        scope = seed_runner._scope(base, old_approval, seed, False)
        anchor = Path(base.anchor_root) / f"random-{seed}"
        initial = seed_runner._initial_state(old_seed, backend, base, scope, anchor, None)
        state, snapshot, transition = _take_transition(old_run, 0, seed, initial, backend, base, old_approval)
        source_contract, source_run = base, old_run
    endpoint = PHASE_BOUNDARIES[seed]
    if state.round != endpoint:
        raise ValueError("continuation source did not reach the promised completed round")
    descriptor = {"round": endpoint, "state_sha256": digest(asdict(state)), "memory_sha256": digest(snapshot)}
    evidence = {"contract_sha256": source_contract.digest, "run_directory": str(source_run),
                "transition_sha256": transition, "checkpoint_sha256": state.checkpoint}
    profile = {**base.profile, "resume_boundaries": {str(seed): descriptor}}
    predecessor = {"pilot_contract_sha256": base.digest, "boundary_evidence": {str(seed): evidence},
                   "fresh_first_request_limit": base.rounds - endpoint,
                   "uses_previous_incomplete_round_responses": False}
    contract = replace(base, profile=profile, source_sha256=sources, run_directory=str(run), predecessor=predecessor)
    return contract, {seed: (state, snapshot, endpoint)}


def worker(root, run, old_run, contract, approval, seed, boundary, generation):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    record = run / "workers" / f"seed_{seed}.json"
    try:
        with scheduler._lock(run / "workers" / f"seed_{seed}.lock", busy="seed worker is already running"):
            if read(run / "coordinator_generation.json")["id"] != generation:
                raise ValueError("seed belongs to an older coordinator")
            scheduler._atomic_status(record, {"status": "initializing", "seed": seed, "pid": os.getpid(), "gpu": 0})
            backend = TrainingBackend(root, old_run / "public_graphs" / f"seed_{seed}", seed)
            call, options = scheduler._provider_wiring(run, contract, approval, None, seed)
            output = seed_runner.run_seed(root=root, run_directory=run, contract=contract, authorization=approval,
                seed=seed, backend=backend, call=call, resume_boundary=boundary,
                fit_slot=partial(scheduler.gpu_slot, run, seed, 0, contract.lease_root), transport_options=options)
            scheduler._atomic_status(record, {**asdict(output), "pid": os.getpid(), "gpu": 0})
    except Exception as error:
        from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import BridgePending

        status = "paused" if scheduler._pause_requested(run, seed) else (
            "bridge_pending" if isinstance(error, BridgePending) else "failed")
        scheduler._atomic_status(record, {"status": status, "seed": seed, "pid": os.getpid(), "gpu": 0,
                                          "error_type": type(error).__name__, "reason": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run", "resume"))
    for name in ("old-release", "old-run", "first-release", "first-run", "run-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=tuple(PHASE_BOUNDARIES), required=True)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    run, old_run = args.run_dir.resolve(), args.old_run.resolve()
    contract, boundaries = construct(ROOT, args.old_release.resolve(), old_run,
                                     args.first_release.resolve(), args.first_run.resolve(), run, args.seed)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        raise PermissionError("continuation requires contract-bound authorization")
    approval = verify_authorization(contract, read(args.authorization))
    ctx = multiprocessing.get_context("spawn")
    with scheduler._lock(run / "coordinator.lock", busy="prompt continuation already running"):
        scheduler._assert_no_old_workers(run, tuple(boundaries))
        if scheduler._pause_requested(run) and args.action != "resume":
            raise PermissionError("continuation is paused; use resume")
        once(run / "formal_contract.json", contract.record())
        once(run / "authorization.json", approval)
        once(run / "provider_model_lock.json", read(args.first_run / "provider_model_lock.json"))
        generation = uuid4().hex
        once(run / "events" / f"{generation}.json", {"action": args.action, "timestamp_unix": time.time(),
                                                     "authorization_sha256": digest(approval)})
        scheduler._atomic_status(run / "coordinator_generation.json", {"id": generation, "pid": os.getpid()})
        if args.action == "resume":
            (run / "pause.request").unlink(missing_ok=True)
        workers = [ctx.Process(target=worker, args=(ROOT, run, old_run, contract, approval, seed, boundary, generation),
                               name=f"a7-seed-{seed}") for seed, boundary in boundaries.items()]
        for child in workers:
            child.start()
        for child in workers:
            child.join()
        print(dumps({"children": {child.name: child.exitcode for child in workers}}))
        raise SystemExit(any(child.exitcode != 0 for child in workers))


if __name__ == "__main__":
    main()
