"""Continue one authorized A7 lane after an ambiguous transport stop, within its recorded budget."""
import argparse
from dataclasses import asdict
from functools import partial
import multiprocessing
import os
from pathlib import Path
import time
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    UNKNOWN_REPLAY_DECISION, TransportConfig, TransportStop, authorize_ambiguous_replay,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import (
    ATTEMPT_ID, RETRYABLE_CODES, validate_result, verify_job,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import FormalContract, OFFLINE, SSH_API, verify_authorization

TRANSPORT = (OFFLINE / "pure_llm_v3/transport_manager.py").as_posix()
AMBIGUOUS_NETWORK = frozenset({"connection_interrupted", "timeout", "sse_frame_error", "sse_missing_completion",
                               "upstream_http2_stream_error", "http_500", "http_502", "http_503", "http_504", "http_529"})
BUSY = "existing seed coordinator is still active"
WORKER_BUSY = "an earlier seed worker is still alive"
INTERRUPTED_WORKER_STATES = {"initializing", "interrupted", "bridge_pending"}
REJECTED_DECISIONS = {"rejected", "malformed_complete_response", "model_requested_replan", "patch_rejected"}


def verify_launcher(policy, policy_sha, launcher_sha, approval=None, approval_sha=None):
    if (approval is None) != (approval_sha is None):
        raise PermissionError("launcher update requires both the approval and its exact digest")
    if approval is None:
        if launcher_sha != policy.get("launcher_sha256"):
            raise PermissionError("rolling launcher differs from the original authorized source")
        return
    expected = {"schema": "a7_rolling_launcher_update_v1", "policy_sha256": policy_sha,
                "old_launcher_sha256": policy["launcher_sha256"], "new_launcher_sha256": launcher_sha,
                "transport_replacement_sha256": policy["transport_replacement_sha256"], "runs": policy["runs"]}
    if (type(approval) is not dict or digest(approval) != approval_sha
            or set(approval) != {*expected, "authorized_by", "authorization_reference"}
            or any(approval[key] != value for key, value in expected.items())
            or launcher_sha == policy["launcher_sha256"]
            or any(not isinstance(approval[key], str) or not approval[key].strip() or len(approval[key]) > 300
                   for key in ("authorized_by", "authorization_reference"))):
        raise PermissionError("launcher approval must pin the watcher and preserve the original policy and transport scope")


def load_run(run, base, overlay, policy, policy_sha, seed, *, launcher_approval=None, launcher_approval_sha=None):
    if (digest(policy) != policy_sha or policy.get("schema") != "a7_v5_bounded_transport_replay_v1"
            or policy.get("additional_physical_attempts_per_logical") != 1
            or policy.get("accept_possible_duplicate_billing") is not True
            or policy.get("provider_lookup") != "unavailable_on_audited_store_false_route"
            or file_digest(overlay / TRANSPORT) != policy.get("transport_replacement_sha256")):
        raise PermissionError("rolling transport replay needs an intact scoped authorization and runtime")
    verify_launcher(policy, policy_sha, file_digest(Path(__file__)), launcher_approval, launcher_approval_sha)
    record = read(run / "formal_contract.json")
    if digest(record) != policy["runs"].get(str(run)) or record.get("run_directory") != str(run):
        raise PermissionError("run is outside the approved recovery scope")
    if (record.get("api_execution") != SSH_API or record.get("gpu_slots") != [0]
            or set(record["profile"].get("resume_boundaries", {})) != {str(seed)}):
        raise PermissionError("rolling replay applies only to the exact one-GPU seed continuation")
    sources = record["source_sha256"]
    if any(file_digest(base / name) != sha for name, sha in sources.items()):
        raise PermissionError("the original continuation release source changed")
    if any(file_digest(overlay / name) != sha for name, sha in sources.items() if name != TRANSPORT):
        raise PermissionError("rolling runtime may only replace the registered transport module")
    contract = FormalContract(record["profile"], TransportConfig(**record["profile"]["transport"]), sources,
                              record["asset_sha256"], record["anchor_root"], tuple(record["gpu_slots"]),
                              record["lease_root"], str(run), record["api_execution"], record["predecessor"])
    if contract.record() != record or contract.transport.transport_attempts != 2:
        raise PermissionError("rolling replay must preserve the two-attempt contract and scientific state")
    approval = verify_authorization(contract, read(run / "authorization.json"))
    descriptor = record["profile"]["resume_boundaries"][str(seed)]
    start = descriptor["round"]
    directory = seed_runner.round_directory(seed_runner.seed_directory(run, seed), start)
    source = read(directory / "source.json")
    memory = read(directory / "boundary_memory.json")
    if digest(source) != descriptor["state_sha256"] or digest(memory) != descriptor["memory_sha256"]:
        raise PermissionError("the original continuation checkpoint or memory changed")
    return contract, approval, (seed_runner.FrozenState(**source), memory, start)


def latest_round(directory, initial_index, limit):
    current = initial_index
    while current < limit and (seed_runner.round_directory(directory, current + 1) / "transition.json").is_file():
        current += 1
    return current


def original_result(run, turn, request, config, approval, contract):
    started = read(turn / "attempt_00.started.json")
    name = f"{started['attempt_id']}.json"
    job = read(run / "bridge/outbox" / name)
    result = read(run / "bridge/results" / name)
    staged_request, staged_config, staged_started = verify_job(job, authorization_sha256=digest(approval),
                contract_sha256=contract.digest, wire_representation=contract.profile["wire_representation"])
    validate_result(result, job)
    previous = read(turn / "attempt_00.terminal.json")
    failure = result["outcome"].get("failure", {})
    if ((staged_request, staged_config, staged_started) != (request, config, started)
            or previous.get("bridge_result_sha256") != result["result_sha256"]
            or result["provider_dispatched"] is not True or result.get("raw_response") is not None
            or failure.get("code") != previous.get("error") or failure.get("ambiguous") is not True):
        raise PermissionError("the archived attempt is neither the exact request nor a sealed ambiguous result")
    return started, previous


def resume_plan(run, directory, state_round, worker, contract, approval, policy_sha):
    round_dir = seed_runner.round_directory(directory, state_round)
    # The caller owns the coordinator lock and has verified that no seed worker owns its lock.
    interrupted = worker.get("status") in INTERRUPTED_WORKER_STATES
    permits = {}
    active = None
    for turn in sorted(round_dir.glob("turn_[0-9][0-9]")):
        if not (turn / "request.json").is_file():
            if any(turn.glob("attempt_*.json")) or (turn / "decision.json").exists():
                raise PermissionError("attempt or decision evidence has no archived request")
            continue
        request = read(turn / "request.json")
        if ((request["binding"]["seed"], request["binding"]["round"], request["logical_index"])
                != (worker["seed"], state_round, int(turn.name.removeprefix("turn_")))):
            raise PermissionError("archived request lies outside the selected seed round")
        active = (turn, request)
        auth_path = turn / "attempt_00.replay_authorization.json"
        if auth_path.is_file():
            recorded = read(auth_path)
            if not recorded["authorization_reference"].endswith(policy_sha):
                raise PermissionError("recorded replay belongs to another operator approval")
            permits[request["request_id"]] = digest(recorded)
    if active is None:
        if (interrupted and (round_dir / "source.json").is_file()
                and not (round_dir / "batch_freeze.json").exists()):
            return "run", "continue from the sealed state after an interrupted worker", permits
        return "stop", "missing recoverable current request", permits
    turn, request = active
    if (round_dir / "batch_freeze.json").exists():
        if interrupted:
            return "run", "resume the already frozen batch", permits
        return "stop", "scientific or training stop after batch freeze", permits
    for attempt in (1, 0):
        started_path = turn / f"attempt_{attempt:02d}.started.json"
        terminal_path = turn / f"attempt_{attempt:02d}.terminal.json"
        if not started_path.is_file():
            if terminal_path.exists():
                raise PermissionError("terminal evidence has no archived attempt reservation")
            continue
        if not terminal_path.is_file():
            if not interrupted:
                return "stop", "non-transport stop with a reserved physical attempt", permits
            job_name = f"{read(started_path)['attempt_id']}.json"
            if (run / "bridge/results" / job_name).is_file() or not (run / "bridge/outbox" / job_name).is_file():
                return "run", "reattach to the original physical attempt", permits
            return "wait", "existing physical attempt awaiting Mac receipt", permits
        terminal = read(terminal_path)
        if terminal["status"] == "completed":
            decision_path = turn / "decision.json"
            decision = read(decision_path)["state"] if decision_path.is_file() else None
            if (decision in REJECTED_DECISIONS
                    and request["logical_index"] + 1 == contract.transport.logical_replies):
                return "stop", "logical response budget exhausted", permits
            if interrupted and (decision is None or decision == "legal" or decision in REJECTED_DECISIONS):
                return "run", "process the existing completed response", permits
            return "stop", "non-transport stop", permits
        if attempt == 1:
            return "stop", "physical attempt budget exhausted", permits
        if (interrupted and terminal["status"] == "retryable" and terminal.get("ambiguous") is False
                and terminal.get("circuit_open") is False and terminal.get("error") in RETRYABLE_CODES):
            return "run", "resume the already authorized ordinary second attempt", permits
        if (terminal.get("status") != "stopped" or terminal.get("ambiguous") is not True
                or terminal.get("error") not in AMBIGUOUS_NETWORK or terminal.get("circuit_open") is not False):
            return "stop", terminal.get("error", "non-transport stop"), permits
        started, previous = original_result(run, turn, request, contract.transport, approval, contract)
        permit = {"decision": UNKNOWN_REPLAY_DECISION, "authorized_by": "pilot operator",
                  "authorization_reference": "2026-11-25 ongoing pilot replay approval " + policy_sha,
                  "accept_duplicate_billing_risk": True, "contract_sha256": contract.digest,
                  "request_id": request["request_id"], "attempt_id": started["attempt_id"],
                  "bridge_result_sha256": previous["bridge_result_sha256"],
                  "original_terminal_sha256": digest(previous), "next_attempt": 1}
        authorize_ambiguous_replay(round_dir, request, authorization=permit, config=contract.transport,
                                  expected_authorization_sha256=digest(permit))
        permits[request["request_id"]] = digest(permit)
        return "run", "one authorized replay of sealed ambiguous transport", permits
    if interrupted and not any(turn.glob("attempt_*.json")):
        return "run", "resume the staged request before its first reservation", permits
    return "stop", "no reserved original request", permits


def _read_worker(run, directory, seed):
    path = run / "workers" / f"seed_{seed}.json"
    if not path.is_file():
        return {"seed": seed, "status": "missing"}
    worker = read(path)
    if worker.get("seed") != seed:
        raise PermissionError("worker custody belongs to another seed")
    since = path.stat().st_mtime_ns
    # If a failure was sealed just before the worker lost its chance to publish status, keep that stop.
    if worker.get("status") == "initializing":
        for event_path in (directory / "events").glob("*.json"):
            if event_path.stat().st_mtime_ns >= since:
                event = read(event_path)
                if event.get("status") == "failed" and event.get("error_type") != "BridgePending":
                    return {**worker, "status": "failed"}
    return worker


def reserve_resume(run, directory, current, reason, permits):
    """Allow one automatic restart at each distinct durable recovery state."""
    current_dir = seed_runner.round_directory(directory, current)
    next_dir = seed_runner.round_directory(directory, current + 1)
    started = list(current_dir.glob("turn_[0-9][0-9]/attempt_*.started.json"))
    paths = [current_dir / name for name in ("source.json", "binding.json", "batch_freeze.json", "selection.json",
                                           "memory.json")]
    paths.extend(current_dir.glob("turn_[0-9][0-9]/request.json"))
    paths.extend(current_dir.glob("turn_[0-9][0-9]/decision.json"))
    paths.extend(current_dir.glob("turn_[0-9][0-9]/attempt_*.terminal.json"))
    paths.extend(started)
    for item in started:
        identity = read(item)["attempt_id"]
        if not ATTEMPT_ID.fullmatch(identity):
            raise PermissionError("recovery cannot track an invalid physical attempt identity")
        paths.append(run / "bridge/results" / f"{identity}.json")
    paths.extend(next_dir / name for name in ("fit_authorization.json", "authorized_feedback.json", "transition.json"))
    paths.extend(next_dir.glob("fit_[0-9][0-9][0-9]/training_complete.json"))
    paths.extend(next_dir.glob("fit_[0-9][0-9][0-9]/fit_artifact_receipt.json"))
    progress = {"round": current, "replay_authorizations": permits, "artifacts": {
        path.relative_to(run).as_posix(): file_digest(path) for path in sorted(paths) if path.is_file()
    }}
    marker = run / "operator/rolling_resume_markers" / f"{digest(progress)}.json"
    if marker.is_file():
        if read(marker)["progress"] != progress:
            raise PermissionError("automatic recovery marker differs from its durable state")
        return False
    once(marker, {"progress": progress, "reason": reason})
    return True


def execute(base, run, cache, contract, approval, seed, boundary, permits, generation):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend
    worker = run / "workers" / f"seed_{seed}.json"
    status = {"seed": seed, "pid": os.getpid(), "gpu": 0}
    with scheduler._lock(run / "workers" / f"seed_{seed}.lock", busy="seed worker is still active"):
        if read(run / "coordinator_generation.json")["id"] != generation:
            raise TransportStop("worker belongs to an older generation")
        try:
            scheduler._atomic_status(worker, {**status, "status": "initializing"})
            backend = TrainingBackend(base, cache, seed)
            call, options = scheduler._provider_wiring(run, contract, approval, None, seed)
            result = seed_runner.run_seed(root=base, run_directory=run, contract=contract, authorization=approval,
                seed=seed, backend=backend, call=call, resume_boundary=boundary,
                fit_slot=partial(scheduler.gpu_slot, run, seed, 0, contract.lease_root),
                transport_options={**options, "replay_authorizations": permits})
            scheduler._atomic_status(worker, {**status, **asdict(result)})
        except BaseException as error:
            from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import BridgePending
            phase = "paused" if scheduler._pause_requested(run, seed) else (
                "bridge_pending" if isinstance(error, BridgePending) else "failed")
            scheduler._atomic_status(worker, {**status, "status": phase, "reason": str(error),
                                               "error_type": type(error).__name__})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "base-release", "overlay", "cache", "policy"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--policy-sha", required=True)
    parser.add_argument("--launcher-approval", type=Path)
    parser.add_argument("--launcher-approval-sha")
    parser.add_argument("--seed", type=int, choices=(157, 887, 2357), required=True)
    args = parser.parse_args()
    run, base, overlay, cache, policy_path = (path.resolve() for path in
        (args.run, args.base_release, args.overlay, args.cache, args.policy))
    seed = args.seed
    policy = read(policy_path)
    launcher_approval = read(args.launcher_approval.resolve()) if args.launcher_approval is not None else None
    contract, approval, boundary = load_run(run, base, overlay, policy, args.policy_sha, seed,
        launcher_approval=launcher_approval, launcher_approval_sha=args.launcher_approval_sha)
    directory = seed_runner.seed_directory(run, seed)
    status_path = run / "operator/rolling_status.json"
    ctx = multiprocessing.get_context("spawn")
    with scheduler._lock(run / "operator/rolling.lock", busy="rolling watcher is already active"):
        once(run / "operator/rolling_transport_authorization.json", policy)
        if launcher_approval is not None:
            once(run / "operator/rolling_launcher_approvals" / f"{args.launcher_approval_sha}.json", launcher_approval)
        previous_status = None

        def report(phase, reason):
            nonlocal previous_status
            if previous_status != (phase, reason):
                data = {"seed": seed, "pid": os.getpid(), "status": phase, "reason": reason,
                        "policy_sha256": args.policy_sha, "time": time.time()}
                scheduler._atomic_status(status_path, data)
                print(dumps(data), flush=True)
                previous_status = (phase, reason)

        while not scheduler._pause_requested(run, seed):
            try:
                with scheduler._lock(run / "coordinator.lock", busy=BUSY):
                    scheduler._assert_no_old_workers(run, (seed,))
                    current = latest_round(directory, boundary[2], contract.rounds)
                    if current == contract.rounds:
                        report("complete", f"{contract.initial_labels + current * contract.batch_size} sealed labels")
                        return
                    worker = _read_worker(run, directory, seed)
                    action, reason, permits = resume_plan(run, directory, current, worker, contract, approval, args.policy_sha)
                    if action == "run" and not reserve_resume(run, directory, current, reason, permits):
                        report("stop", "automatic restart already used at this unchanged durable state")
                        return
                    report(action, reason)
                    if action == "stop":
                        return
                    if action == "run":
                        generation = uuid4().hex
                        once(run / "events" / f"{generation}.json", {"action": "rolling_transport_resume",
                             "authorization_sha256": digest(approval), "replay_policy_sha256": args.policy_sha,
                             "timestamp_unix": time.time(), "seed": seed, "round": current})
                        scheduler._atomic_status(run / "coordinator_generation.json", {"id": generation, "pid": os.getpid()})
                        process = ctx.Process(target=execute, args=(base, run, cache, contract, approval, seed,
                                                                   boundary, permits, generation), name=f"a7-seed-{seed}")
                        process.start()
                        process.join()
            except TransportStop as error:
                if str(error) not in (BUSY, WORKER_BUSY):
                    raise
                report("watching", str(error))
            time.sleep(3)
        report("paused", "operator pause")


if __name__ == "__main__":
    main()
