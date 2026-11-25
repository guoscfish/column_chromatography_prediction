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
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import validate_result, verify_job
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import FormalContract, OFFLINE, SSH_API, verify_authorization

TRANSPORT = (OFFLINE / "pure_llm_v3/transport_manager.py").as_posix()
AMBIGUOUS_NETWORK = frozenset({"connection_interrupted", "timeout", "sse_frame_error", "sse_missing_completion",
                               "upstream_http2_stream_error", "http_500", "http_502", "http_503", "http_504", "http_529"})
BUSY = "existing seed coordinator is still active"


def load_run(run, base, overlay, policy, policy_sha, seed):
    if (digest(policy) != policy_sha or policy.get("schema") != "a7_v5_bounded_transport_replay_v1"
            or policy.get("additional_physical_attempts_per_logical") != 1
            or policy.get("accept_possible_duplicate_billing") is not True
            or policy.get("provider_lookup") != "unavailable_on_audited_store_false_route"
            or file_digest(Path(__file__)) != policy.get("launcher_sha256")
            or file_digest(overlay / TRANSPORT) != policy.get("transport_replacement_sha256")):
        raise PermissionError("rolling transport replay needs an intact scoped authorization and runtime")
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
    permits = {}
    active = None
    for turn in sorted(round_dir.glob("turn_[0-9][0-9]")):
        if not (turn / "request.json").is_file():
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
        return "stop", "missing recoverable current request", permits
    turn, request = active
    if (round_dir / "batch_freeze.json").exists():
        if worker.get("status") in {"bridge_pending", "initializing"}:
            return "run", "resume the already frozen batch", permits
        return "stop", "scientific or training stop after batch freeze", permits
    for attempt in (1, 0):
        started_path = turn / f"attempt_{attempt:02d}.started.json"
        if not started_path.is_file():
            continue
        terminal_path = turn / f"attempt_{attempt:02d}.terminal.json"
        if not terminal_path.is_file():
            job_name = f"{read(started_path)['attempt_id']}.json"
            if (run / "bridge/results" / job_name).is_file() or not (run / "bridge/outbox" / job_name).is_file():
                return "run", "reattach to the original physical attempt", permits
            return "wait", "existing physical attempt awaiting Mac receipt", permits
        terminal = read(terminal_path)
        if attempt == 1:
            return "stop", "physical attempt budget exhausted" if terminal["status"] != "completed" else "non-transport stop", permits
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
    return "stop", "no reserved original request", permits


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
    parser.add_argument("--seed", type=int, choices=(157, 887, 2357), required=True)
    args = parser.parse_args()
    run, base, overlay, cache, policy_path = (path.resolve() for path in
        (args.run, args.base_release, args.overlay, args.cache, args.policy))
    seed = args.seed
    policy = read(policy_path)
    contract, approval, boundary = load_run(run, base, overlay, policy, args.policy_sha, seed)
    directory = seed_runner.seed_directory(run, seed)
    status_path = run / "operator/rolling_status.json"
    ctx = multiprocessing.get_context("spawn")
    with scheduler._lock(run / "operator/rolling.lock", busy="rolling watcher is already active"):
        once(run / "operator/rolling_transport_authorization.json", policy)
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
                    worker = read(run / "workers" / f"seed_{seed}.json")
                    action, reason, permits = resume_plan(run, directory, current, worker, contract, approval, args.policy_sha)
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
                if str(error) != BUSY:
                    raise
                report("watching", BUSY)
            time.sleep(3)
        report("paused", "operator pause")


if __name__ == "__main__":
    main()
