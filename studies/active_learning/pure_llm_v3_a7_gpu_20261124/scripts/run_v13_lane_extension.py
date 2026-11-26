"""Stage at most one new request for each of two explicitly scoped V13 failures."""

import argparse
from dataclasses import asdict, dataclass, replace
from functools import partial
import os
from pathlib import Path
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    TransportStop, make_request, terminal_record,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import (
    BridgePending, read_bridge_json, validate_result, verify_job,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import AUTHORITATIVE_PACKET_FILE
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    ROOT, STUDY, RequestTimeouts, verify_authorization,
)
from . import run_v13_seed2357_extension as previous
from .run_prepared_five_seed import _qualification_model, read_boundary


DISPATCHER = "dispatch_v13_lane_extension"
TIMEOUTS = RequestTimeouts(read_idle_seconds=900, total_seconds=1800)
# The shared pool might first be occupied by requests with the original V13 hard wall.
BRIDGE_WAIT_SECONDS = 3600 + TIMEOUTS.total_seconds + TIMEOUTS.bridge_margin_seconds
LOGICAL = 0


@dataclass(frozen=True)
class Lane:
    seed: int
    round: int
    posted_error: str
    attempt_ids: tuple[str, str]
    payload_sha256: str

    @property
    def run_name(self):
        return f"{previous.PARENT_NAME}-{self.seed}-extension-01"


LANES = {
    6101: Lane(6101, 1, "timeout", (
        "f7c8c634b4a047e7a00cc1e70d630583", "02baf5606f2f4c5395ca9167c9c560b5",
    ), "55c213c38e8c6b5fe3832f05060930bc436ec530d437da3551dc74f6d1324eeb"),
    887: Lane(887, 6, "response_id_mismatch", (
        "8092f217a6234c6aa72cd86df60fb862", "0716cc9aca894e31b95c5f81b8a1d427",
    ), "b2ae00711d597149ee2bd6d19dc37002a0b138b2a0ed9cb6f006959e11627167"),
}


def _uncommitted(parent, lane, *, require_paused=False):
    seed = seed_runner.seed_directory(parent, lane.seed)
    active = seed_runner.round_directory(seed, lane.round)
    following = seed_runner.round_directory(seed, lane.round + 1)
    if any((active / name).exists() for name in ("batch_freeze.json", "selection.json", "memory.json")) or any(
        (following / name).exists() for name in ("fit_authorization.json", "authorized_feedback.json",
                                                "source.json", "transition.json")
    ):
        raise PermissionError("finish already selected work under its original V13 contract")
    if require_paused and not (seed / "pause.request").is_file():
        raise PermissionError("pause just this original V13 seed before starting its new authorized attempt")
    return seed, active


def _boundary(root, parent, lane, contract, approval):
    """Reconstruct the exact current state from a signed prepared state and sealed fits."""
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    seed_dir, active = _uncommitted(parent, lane)
    old_lane = contract.predecessor["lanes"][str(lane.seed)]
    preparation = Path(old_lane["boundary_directory"]).parents[1]
    draft = read(preparation / "five_seed_boundary_draft.json")
    row = draft["boundaries"][str(lane.seed)]
    state, memory, start = read_boundary(row, lane.seed, preparation)
    descriptor = contract.profile["resume_boundaries"][str(lane.seed)]
    if (digest(draft) != contract.predecessor["prepared_draft_sha256"] or row["descriptor"] != descriptor
            or row["cache_directory"] != old_lane["cache_directory"] or row["boundary_directory"] != old_lane["boundary_directory"]
            or state.checkpoint != old_lane["checkpoint_sha256"] or start != old_lane["start_round"]
            or lane.round != start + 1):
        raise PermissionError("the prepared seed boundary no longer matches the original V13 contract")
    expected_resource = (state.provenance["resource_receipt_sha256"] if start == 0
                         else row["evidence"]["public_resource_receipt_sha256"])
    cache = Path(old_lane["cache_directory"]).resolve(strict=True)
    if file_digest(cache / "resource_receipt.json") != expected_resource:
        raise PermissionError("the training cache differs from the signed prepared resource receipt")
    scope = seed_runner._scope(contract, approval, lane.seed, False)
    before = seed_runner.round_directory(seed_dir, start)
    if (read(seed_dir / "boundary.json") != {"formal_run": scope, **descriptor}
            or read(before / "source.json") != asdict(state) or read(before / "boundary_memory.json") != memory):
        raise PermissionError("the original seed start does not match its prepared state and memory")
    backend = TrainingBackend(root, cache, lane.seed)
    state, memory = seed_runner._successor(before, state, active, scope, contract.transport, contract, backend)
    return active, backend, state, memory


def _retired_attempts(parent, turn, lane, request, contract, approval):
    """Bind both originals, separating the GET preflight from the ambiguous paid POST."""
    expected = {turn / f"attempt_{index:02d}.started.json" for index in range(2)}
    if (set(turn.glob("attempt_*.started.json")) != expected or any(turn.glob("*.reconciliation.json"))
            or any(turn.glob("attempt_*.replay*.json"))):
        raise PermissionError("the original turn needs its two exact, unreconciled reservations")
    prior = []
    for index, attempt_id in enumerate(lane.attempt_ids):
        posted = bool(index)
        error = lane.posted_error if posted else "connect_error"
        status = "stopped" if posted else "retryable"
        started = read(turn / f"attempt_{index:02d}.started.json")
        terminal = terminal_record(turn, index)
        name = f"{attempt_id}.json"
        job = read_bridge_json(parent / "bridge" / "outbox" / name)
        result = read_bridge_json(parent / "bridge" / "results" / name)
        bound = verify_job(job, authorization_sha256=digest(approval), contract_sha256=contract.digest,
                           wire_representation=contract.profile["wire_representation"])
        validate_result(result, job)
        failure = result["outcome"].get("failure", {})
        if (bound != (request, contract.transport, started) or (started["attempt"], started["attempt_id"]) != (index, attempt_id)
                or any(terminal.get(key) != value for key, value in started.items())
                or terminal.get("status") != status or terminal.get("error") != error
                or terminal.get("ambiguous") is not posted or terminal.get("provider_dispatched") is not posted
                or terminal.get("circuit_open") is not False or "answer_sha256" in terminal
                or "reconciliation_sha256" in terminal
                or terminal.get("bridge_job_sha256") != job["job_sha256"]
                or terminal.get("bridge_result_sha256") != result["result_sha256"]
                or result["outcome"]["status"] != "failed" or failure.get("code") != error
                or failure.get("ambiguous") is not posted or failure.get("retryable") is posted
                or result["provider_dispatched"] is not posted or result["raw_response"] is not None):
            raise PermissionError("the original evidence differs from the sealed preflight and posted-failure pair")
        prior.append({"attempt": index, "attempt_id": attempt_id, "job_sha256": job["job_sha256"],
                      "result_sha256": result["result_sha256"], "terminal_sha256": digest(terminal),
                      "error": error, "provider_dispatched": posted, "ambiguous": posted})
    return prior


def _request(root, contract, approval, lane, state, memory, original):
    scope = seed_runner._scope(contract, approval, lane.seed, False)
    stage = seed_runner._stage(root, state, memory, scope, contract)
    message = original["payload"]["input"][1]
    stage = seed_runner.pin_boundary_user_message(stage, message, digest(message))
    return make_request(stage.binding, LOGICAL, stage.messages, contract.transport)


def construct(root, original_release, parent, run, lane):
    if run.name != lane.run_name or run.parent != parent.parent or root == original_release:
        raise PermissionError("use this seed's registered sibling run and separate immutable release")
    old, approval = previous._original_contract(parent)
    sources = dict(old.source_sha256)
    if any(file_digest(release / name) != expected for release in (root, original_release)
           for name, expected in sources.items()):
        raise PermissionError("a source referenced by the original V13 contract changed")
    for module in ("run_v13_lane_extension", DISPATCHER, "run_v13_seed2357_extension", "dispatch_prompt_direct"):
        name = (STUDY / "scripts" / f"{module}.py").as_posix()
        sources[name] = file_digest(root / name)

    active, backend, state, memory = _boundary(root, parent, lane, old, approval)
    turn = active / "turn_00"
    original = read(turn / "request.json")
    binding = original["binding"]
    if (state.round != lane.round or original["payload_sha256"] != lane.payload_sha256
            or binding["packet_sha256"] != digest(read(active / AUTHORITATIVE_PACKET_FILE))
            or _request(root, old, approval, lane, state, memory, original) != original):
        raise PermissionError("the V13 scientific request differs from its sealed state or exact input")
    prior = _retired_attempts(parent, turn, lane, original, old, approval)
    identity = _qualification_model({"model_identity": read(parent / "provider_model_lock.json")}, old.transport)
    if identity != old.predecessor["provider_qualification"]["model_identity"]:
        raise PermissionError("the provider model lock differs from the approved V13 route")

    descriptor = {"round": lane.round, "state_sha256": digest(asdict(state)), "memory_sha256": digest(memory),
                  "user_message_sha256": digest(original["payload"]["input"][1])}
    transport = replace(old.transport, logical_replies=1, transport_attempts=1, timeout_seconds=TIMEOUTS.total_seconds)
    profile = {
        **old.profile, "confirmation_seeds": [lane.seed], "qualification_seed": lane.seed,
        "requires_initial_qualification": False, "api_parallelism": 1,
        "target_labels": old.initial_labels + old.batch_size * (lane.round + 1),
        "transport": asdict(transport), "request_timeouts": asdict(TIMEOUTS),
        "resume_boundaries": {str(lane.seed): descriptor},
    }
    predecessor = {
        "dispatcher_module": DISPATCHER, "parent_run_directory": str(parent),
        "parent_release_directory": str(original_release), "parent_contract_sha256": old.digest,
        "parent_authorization_sha256": digest(approval), "seed": lane.seed, "round": lane.round,
        "logical_index": LOGICAL, "prior_request_id": original["request_id"], "prior_attempts": prior,
        "payload_sha256": original["payload_sha256"], "model_identity": identity,
        **{key: binding[key] for key in previous.MESSAGE_HASHES},
        "prior_execution_and_billing": "unresolved", "execution_budget": previous.ONE_SHOT_BUDGET,
        "bridge_wait_seconds": BRIDGE_WAIT_SECONDS,
        "parent_transition_sha256": file_digest(active / "transition.json"),
        "parent_fit_audit_sha256": state.provenance["fit_audit_sha256"],
        "public_resource_receipt_sha256": backend.cache_receipt_sha256,
    }
    contract = replace(old, profile=profile, transport=transport, source_sha256=sources,
                       run_directory=str(run), predecessor=predecessor)
    return contract, (state, memory, lane.round), backend, original


def _completion(run, contract, approval, lane, boundary, backend):
    active = seed_runner.round_directory(seed_runner.seed_directory(run, lane.seed), lane.round)
    following = seed_runner.round_directory(active.parent, lane.round + 1)
    scope = seed_runner._scope(contract, approval, lane.seed, False)
    state, memory = seed_runner._successor(active, boundary[0], following, scope, contract.transport, contract, backend)
    remaining = 36 - state.round
    once(run / "extension_completion.json", {
        "contract_sha256": contract.digest, "seed": lane.seed, "round": state.round,
        "active_labels": contract.target_labels, "checkpoint_sha256": state.checkpoint,
        "state_sha256": digest(asdict(state)), "memory_sha256": digest(memory),
        "transition_sha256": file_digest(following / "transition.json"),
        "original_future_budget": {"first_logical_requests": remaining,
                                   "maximum_logical_requests": remaining * 3,
                                   "maximum_physical_attempts": remaining * 3 * 2},
    })


def execute(root, parent, run, lane, contract, approval, boundary, backend, original, *, resume):
    _uncommitted(parent, lane, require_paused=True)
    state, memory, _ = boundary
    request = _request(root, contract, approval, lane, state, memory, original)
    if not previous._same_science(request, original):
        raise PermissionError("the authorized extension does not reproduce the entire original payload")
    with scheduler._lock(run / "coordinator.lock", busy="this seed extension is already running"):
        if scheduler._pause_requested(run) and not resume:
            raise PermissionError("this extension is paused; use resume")
        once(run / "formal_contract.json", contract.record())
        once(run / "authorization.json", approval)
        once(run / "provider_model_lock.json", contract.predecessor["model_identity"])
        once(run / "extension_request_scope.json", {"request_id": request["request_id"],
            "payload_sha256": request["payload_sha256"], "seed": lane.seed, "round": lane.round,
            "logical_index": LOGICAL, "attempt": 0})
        if resume:
            (run / "pause.request").unlink(missing_ok=True)
            (seed_runner.seed_directory(run, lane.seed) / "pause.request").unlink(missing_ok=True)
        scheduler._atomic_status(run / "coordinator_generation.json", {"id": uuid4().hex, "pid": os.getpid()})
        return _run(root, parent, run, lane, contract, approval, boundary, backend, original, request)


def _run(root, parent, run, lane, contract, approval, boundary, backend, original, request):
    status = {"seed": lane.seed, "pid": os.getpid(), "gpu": contract.gpu_slots[0]}
    path = run / "workers" / f"seed_{lane.seed}.json"
    scheduler._atomic_status(path, {**status, "status": "initializing"})
    try:
        call, options = scheduler._provider_wiring(run, contract, approval, None, lane.seed)
        managed = options["managed_attempt"]
        managed.wait_seconds = contract.predecessor["bridge_wait_seconds"]

        def one_request(candidate, config, started):
            _uncommitted(parent, lane, require_paused=True)
            if not previous._one_attempt(request, contract.transport, contract.predecessor["prior_attempts"],
                                         candidate, config, started):
                raise TransportStop("this contract authorizes one exact new physical attempt")
            return managed(candidate, config, started)

        result = seed_runner.run_seed(
            root=root, run_directory=run, contract=contract, authorization=approval, seed=lane.seed, backend=backend,
            call=call, resume_boundary=boundary, boundary_user_message=original["payload"]["input"][1],
            transport_options={"managed_attempt": one_request},
            fit_slot=partial(scheduler.gpu_slot, run, lane.seed, status["gpu"], contract.lease_root),
        )
        if result.status == "complete":
            _completion(run, contract, approval, lane, boundary, backend)
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
    parser.add_argument("--seed", type=int, choices=tuple(LANES), required=True)
    parser.add_argument("--original-release", type=Path, required=True)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    original_release, parent, run = (path.resolve() for path in (args.original_release, args.original_run, args.run_dir))
    lane = LANES[args.seed]
    contract, boundary, backend, original = construct(ROOT, original_release, parent, run, lane)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        parser.error("run and resume require the exact one-attempt authorization")
    approval = verify_authorization(contract, read(args.authorization))
    result = execute(ROOT, parent, run, lane, contract, approval, boundary, backend, original,
                     resume=args.action == "resume")
    print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
