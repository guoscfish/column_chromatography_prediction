"""Prepare one exact WebSocket request for the sealed 6101 L365 boundary."""

import argparse
from dataclasses import asdict, dataclass, replace
from functools import partial
import os
from pathlib import Path
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_WEBSOCKET_IPV4, TransportStop, terminal_record, wire_payload,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import (
    BridgePending, read_bridge_json, validate_result, verify_job,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    OFFLINE, ROOT, STUDY, RequestTimeouts, verify_authorization,
)
from . import run_v13_lane_extension as earlier


LANE = earlier.LANES[6101]
RUN_NAME = "a7-gpt62-astra-xhigh-ipv4-v13-6101-ws-extension-01"
DISPATCHER = "dispatch_v13_6101_ws_extension"
TIMEOUTS = RequestTimeouts(read_idle_seconds=1800, total_seconds=3600)
INACTIVITY_SECONDS = 900
WEBSOCKETS_VERSION = "15.0.1"
OVERLAYS = frozenset(((STUDY / "a7_gpu/provider.py").as_posix(),
                      (STUDY / "a7_gpu/dispatcher.py").as_posix(),
                      (OFFLINE / "pure_llm_v3/transport_manager.py").as_posix()))


@dataclass(frozen=True)
class FailedOneShot:
    contract_sha256: str
    authorization_sha256: str
    attempt_id: str
    request_id: str
    job_sha256: str
    result_sha256: str


PREVIOUS = FailedOneShot(
    "3fbd15709e318e3f7cce6fe57257be4c5a123d08769a83c041da6c25868c7d07",
    "178ae4d043b4ed4e625254583b4b602ce7773d10cd414860ecfa2d8e903459b4",
    "55f8c0bc852c452ba1f46f3c157861b1",
    "56e6016470e0353ba97df3e8a956454f29f1efc6bf5ed370be95a88f46905f56",
    "7ff0face0800083aef78058198ae037523d6aa14a2a5101d8e8966db682bae7c",
    "2437350cc6fbbcb0d351cfeb86ca987849178f9f987bfc54d16e286bd48026fd",
)


def _retired(original_run, previous_run):
    for run in (original_run, previous_run):
        earlier._uncommitted(run, LANE, require_paused=True)


def _sources(root, previous):
    sources = {name: file_digest(root / name) for name in previous.source_sha256}
    if {name for name, actual in sources.items() if actual != previous.source_sha256[name]} != OVERLAYS:
        raise PermissionError("the immutable sibling may replace only the signed WebSocket transport sources")
    for module in ("run_v13_6101_ws_extension", DISPATCHER):
        name = (STUDY / "scripts" / f"{module}.py").as_posix()
        sources[name] = file_digest(root / name)
    return sources


def _previous_attempt(run, active, lane, contract, approval, request, spec):
    """Bind the preceding spent SSE unit to its exact failure and verify the POST was made."""
    turn = active / "turn_00"
    reserved = turn / "attempt_00.started.json"
    if (set(active.glob("turn_*/attempt_*.started.json")) != {reserved}
            or any(turn.glob("*.reconciliation.json")) or any(turn.glob("attempt_*.replay*.json"))):
        raise PermissionError("the preceding failure requires exactly one exhausted and unreconciled attempt")
    name = f"{spec.attempt_id}.json"
    started = read(reserved)
    terminal = terminal_record(turn, 0)
    job = read_bridge_json(run / "bridge" / "outbox" / name)
    result = read_bridge_json(run / "bridge" / "results" / name)
    binding = verify_job(job, contract_sha256=spec.contract_sha256, authorization_sha256=digest(approval),
                         wire_representation=contract.profile["wire_representation"])
    validate_result(result, job)
    failure = result["outcome"].get("failure", {})
    scope = seed_runner._scope(contract, approval, lane.seed, False)
    if (binding != (request, contract.transport, started) or request != read(turn / "request.json")
            or (request["request_id"], request["payload_sha256"]) != (spec.request_id, lane.payload_sha256)
            or (started["attempt"], started["attempt_id"]) != (0, spec.attempt_id)
            or (job["job_sha256"], result["result_sha256"]) != (spec.job_sha256, spec.result_sha256)
            or any(terminal.get(key) != value for key, value in started.items())
            or terminal.get("status") != "stopped" or terminal.get("error") != "response_id_mismatch"
            or terminal.get("ambiguous") is not True or terminal.get("provider_dispatched") is not True
            or terminal.get("circuit_open") is not False or "answer_sha256" in terminal
            or "reconciliation_sha256" in terminal or terminal.get("provider_status") == "completed"
            or (terminal.get("bridge_job_sha256"), terminal.get("bridge_result_sha256")) != (spec.job_sha256, spec.result_sha256)
            or result["outcome"]["status"] != "failed" or failure.get("code") != "response_id_mismatch"
            or failure.get("ambiguous") is not True or result["provider_dispatched"] is not True
            or result["raw_response"] is not None
            or read(turn / "formal_request_authorization.json")
            != seed_runner._selection_grant(scope, spec.request_id)):
        raise PermissionError("the preceding SSE request is not its exact sealed ID-mismatch failure")
    return {"attempt": 0, "attempt_id": spec.attempt_id, "job_sha256": spec.job_sha256,
            "result_sha256": spec.result_sha256, "terminal_sha256": digest(terminal),
            "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True}


def _same_wire(original, transport, lane):
    payload = wire_payload(original["payload"]["input"], transport)
    if (payload != original["payload"] or digest(payload) != lane.payload_sha256
            or payload.get("stream") is not True or payload.get("store") is not False or "background" in payload):
        raise PermissionError("the signed WebSocket request must retain the entire archived scientific payload")


def construct(root, original_release, original_run, previous_release, previous_run, run):
    if run.name != RUN_NAME or run.parent != original_run.parent or root in (original_release, previous_release):
        raise PermissionError("use the registered 6101 WebSocket sibling and its immutable release")
    _, active = earlier._uncommitted(previous_run, LANE)
    previous, boundary, backend, original = earlier.construct(
        previous_release, original_release, original_run, previous_run, LANE,
    )
    approval = verify_authorization(previous, read(previous_run / "authorization.json"))
    if (read(previous_run / "formal_contract.json") != previous.record()
            or previous.digest != PREVIOUS.contract_sha256 or digest(approval) != PREVIOUS.authorization_sha256):
        raise PermissionError("the published 6101 one-attempt extension or its authorization changed")
    state, memory, _ = boundary
    scope = seed_runner._scope(previous, approval, LANE.seed, False)
    if (read(active / "source.json") != asdict(state) or read(active / "boundary_memory.json") != memory
            or read(active.parent / "boundary.json") != {
                "formal_run": scope, **previous.profile["resume_boundaries"][str(LANE.seed)],
            }):
        raise PermissionError("the preceding extension does not own the exact L365 state and memory")
    request = earlier._request(previous_release, previous, approval, LANE, state, memory, original)
    if not earlier.previous._same_science(request, original):
        raise PermissionError("the preceding SSE extension changed the original scientific payload")
    prior = _previous_attempt(previous_run, active, LANE, previous, approval, request, PREVIOUS)
    if read(previous_run / "provider_model_lock.json") != previous.predecessor["model_identity"]:
        raise PermissionError("the previous served model identity changed")

    transport = replace(previous.transport, wire_mode=RESPONSES_WEBSOCKET_IPV4, timeout_seconds=TIMEOUTS.total_seconds)
    _same_wire(original, transport, LANE)
    profile = {
        **previous.profile, "transport": asdict(transport), "request_timeouts": asdict(TIMEOUTS),
        "sse_inactivity_seconds": INACTIVITY_SECONDS, "websockets_version": WEBSOCKETS_VERSION,
    }
    predecessor = {
        **previous.predecessor, "dispatcher_module": DISPATCHER, "bridge_wait_seconds": TIMEOUTS.bridge_wait_seconds,
        "previous_run_directory": str(previous_run), "previous_release_directory": str(previous_release),
        "previous_contract_sha256": previous.digest, "previous_authorization_sha256": digest(approval),
        "previous_request_id": PREVIOUS.request_id, "previous_attempt": prior,
    }
    contract = replace(previous, profile=profile, transport=transport, source_sha256=_sources(root, previous),
                       run_directory=str(run), predecessor=predecessor)
    return contract, boundary, backend, original


def execute(root, parent, previous_run, run, contract, approval, boundary, backend, original, *, resume):
    _retired(parent, previous_run)
    state, memory, _ = boundary
    request = earlier._request(root, contract, approval, LANE, state, memory, original)
    if not earlier.previous._same_science(request, original):
        raise PermissionError("the authorized WebSocket request changed the full archived scientific input")
    with scheduler._lock(run / "coordinator.lock", busy="this WebSocket extension is already running"):
        if scheduler._pause_requested(run) and not resume:
            raise PermissionError("this WebSocket extension is paused; use resume")
        once(run / "formal_contract.json", contract.record())
        once(run / "authorization.json", approval)
        once(run / "provider_model_lock.json", contract.predecessor["model_identity"])
        once(run / "extension_request_scope.json", {"request_id": request["request_id"],
             "payload_sha256": request["payload_sha256"], "seed": LANE.seed, "round": LANE.round,
             "logical_index": 0, "attempt": 0})
        if resume:
            (run / "pause.request").unlink(missing_ok=True)
            (seed_runner.seed_directory(run, LANE.seed) / "pause.request").unlink(missing_ok=True)
        scheduler._atomic_status(run / "coordinator_generation.json", {"id": uuid4().hex, "pid": os.getpid()})
        return _run(root, parent, previous_run, run, contract, approval, boundary, backend, original, request)


def _run(root, parent, previous_run, run, contract, approval, boundary, backend, original, request):
    status = {"seed": LANE.seed, "pid": os.getpid(), "gpu": contract.gpu_slots[0]}
    path = run / "workers" / f"seed_{LANE.seed}.json"
    scheduler._atomic_status(path, {**status, "status": "initializing"})
    try:
        call, options = scheduler._provider_wiring(run, contract, approval, None, LANE.seed)
        managed = options["managed_attempt"]
        managed.wait_seconds = contract.predecessor["bridge_wait_seconds"]

        def one_request(candidate, config, started):
            _retired(parent, previous_run)
            prior = [*contract.predecessor["prior_attempts"], contract.predecessor["previous_attempt"]]
            if not earlier.previous._one_attempt(request, contract.transport, prior, candidate, config, started):
                raise TransportStop("the signed WebSocket extension allows one exact new physical attempt")
            return managed(candidate, config, started)

        result = seed_runner.run_seed(
            root=root, run_directory=run, contract=contract, authorization=approval, seed=LANE.seed, backend=backend,
            call=call, resume_boundary=boundary, boundary_user_message=original["payload"]["input"][1],
            transport_options={"managed_attempt": one_request},
            fit_slot=partial(scheduler.gpu_slot, run, LANE.seed, status["gpu"], contract.lease_root),
        )
        if result.status == "complete":
            earlier._completion(run, contract, approval, LANE, boundary, backend)
        scheduler._atomic_status(path, {**status, **asdict(result)})
        return result
    except Exception as error:
        phase = "paused" if scheduler._pause_requested(run, LANE.seed) else (
            "bridge_pending" if isinstance(error, BridgePending) else "failed")
        scheduler._atomic_status(path, {**status, "status": phase, "error_type": type(error).__name__,
                                        "reason": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run", "resume"))
    for name in ("original-release", "original-run", "previous-release", "previous-run", "run-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    original_release, parent, previous_release, previous, run = (
        path.resolve() for path in (args.original_release, args.original_run, args.previous_release,
                                    args.previous_run, args.run_dir)
    )
    contract, boundary, backend, original = construct(ROOT, original_release, parent, previous_release, previous, run)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        parser.error("run and resume require exact authorization for the new WebSocket contract")
    approval = verify_authorization(contract, read(args.authorization))
    result = execute(ROOT, parent, previous, run, contract, approval, boundary, backend, original,
                     resume=args.action == "resume")
    print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
