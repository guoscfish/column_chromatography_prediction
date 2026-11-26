"""Authorize one exact R2 request after the earlier 2357 replay stopped before POST."""

import argparse
from dataclasses import asdict, replace
from functools import partial
import os
from pathlib import Path
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    _link_replay_attempt, make_request, terminal_record, TransportStop,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import (
    BridgePending, read_bridge_json, validate_result, verify_job,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import AUTHORITATIVE_PACKET_FILE
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    ROOT, STUDY, RequestTimeouts, verify_authorization,
)
from . import run_v13_seed2357_continuation as continuation
from . import run_v13_seed2357_extension as previous


SEED, ROUND, LOGICAL = 2357, 2, 0
PARENT_CONTRACT = "99fd5844f05bca5d7d6be41bc82dbccc9dc77d90faba61101ff5a0c490403c2b"
PARENT_AUTHORIZATION = "01dce8c6c2feef6f9b042858189fddcd95f77f0678867fc9f3f3731a39134f16"
RUN_NAME = "a7-gpt62-astra-xhigh-ipv4-v13-2357-r2-extension-01"
DISPATCHER = "dispatch_v13_2357_r2_sse_extension"
REQUEST = "42cebcd7e63ca4e60b5b8ee6d957e7b0b82f2adaec93038e88b386a392f72f87"
PAYLOAD = "a2bd678e1246366cdb3f2870f5ff235465e30b7629932857cde2e3bf3b54b60e"
ATTEMPTS = ("2e3790a917714ed6bc9a2bcdc4c9db09", "985cdc73312f40558a79bd6902c01dd4")
TIMEOUTS = RequestTimeouts(read_idle_seconds=1800, total_seconds=3600)
SSE_INACTIVITY_SECONDS = 900
TRANSPORT_OVERLAYS = frozenset(
    (STUDY / "a7_gpu" / name).as_posix() for name in ("provider.py", "dispatcher.py")
)


def _retirement(parent, *, expected=None):
    seed = seed_runner.seed_directory(parent, SEED)
    active = seed_runner.round_directory(seed, ROUND)
    following = seed_runner.round_directory(seed, ROUND + 1)
    if any((active / name).exists() for name in ("batch_freeze.json", "selection.json", "memory.json")) or any(
        (following / name).exists() for name in ("fit_authorization.json", "authorized_feedback.json",
                                                "source.json", "transition.json")
    ):
        raise PermissionError("the original continuation already has scientific progress for R2")
    pause = seed / "pause.request"
    if not pause.is_file():
        raise PermissionError("pause the original 2357 continuation seed before establishing the sibling attempt")
    proof = file_digest(pause)
    if expected is not None and proof != expected:
        raise PermissionError("the original 2357 retirement marker changed")
    return active, proof


def _original(parent_release, parent):
    record = read(parent / "formal_contract.json")
    approval = read(parent / "authorization.json")
    if (parent.name != continuation.RUN_NAME or record["run_directory"] != str(parent)
            or digest(record) != PARENT_CONTRACT or digest(approval) != PARENT_AUTHORIZATION):
        raise PermissionError("the recovery requires the exact original 2357 continuation and approval")
    history = record["predecessor"]
    original_release, original_run, extension_release, extension_run = (
        Path(history[key]) for key in ("parent_release_directory", "parent_run_directory",
                                       "extension_release_directory", "extension_run_directory")
    )
    contract, boundary, backend = continuation.construct(
        parent_release, original_release, original_run, extension_release, extension_run, parent,
    )
    approval = verify_authorization(contract, approval)
    if (contract.record() != record or contract.digest != PARENT_CONTRACT or boundary[2] != ROUND
            or read(parent / "provider_model_lock.json") != history["model_identity"]
            or any(file_digest(parent_release / name) != expected for name, expected in contract.source_sha256.items())):
        raise PermissionError("the original continuation source, CUDA boundary, or model lock changed")
    return contract, approval, boundary, backend


def _source_hashes(root, old):
    sources = {name: file_digest(root / name) for name in old.source_sha256}
    if {name for name, actual in sources.items() if old.source_sha256[name] != actual} != TRANSPORT_OVERLAYS:
        raise PermissionError("only the new provider and dispatcher may differ from the original continuation")
    for module in ("run_v13_2357_r2_sse_extension", DISPATCHER, "dispatch_prompt_direct"):
        name = (STUDY / "scripts" / f"{module}.py").as_posix()
        sources[name] = file_digest(root / name)
    return sources


def _prior_attempts(parent, turn, request, contract, approval):
    expected = {turn / f"attempt_{index:02d}.started.json" for index in range(2)}
    if set(turn.glob("attempt_*.started.json")) != expected or any(turn.glob("*.reconciliation.json")):
        raise PermissionError("the original turn needs two exact sealed attempts with no recovered completion")
    permit_sha = digest(read(turn / "attempt_00.replay_authorization.json"))
    replay = terminal_record(turn, 0, replay_authorization_sha256=permit_sha)
    _link_replay_attempt(turn, read(turn / "attempt_01.started.json"), replay, allow_create=False)
    prior = []
    for index, attempt_id in enumerate(ATTEMPTS):
        posted = index == 0
        error = "timeout" if posted else "connect_error"
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
                or terminal.get("status") != "stopped" or terminal.get("error") != error
                or terminal.get("ambiguous") is not posted or terminal.get("provider_dispatched") is not posted
                or terminal.get("circuit_open") is not False or "answer_sha256" in terminal
                or "reconciliation_sha256" in terminal
                or terminal.get("bridge_job_sha256") != job["job_sha256"]
                or terminal.get("bridge_result_sha256") != result["result_sha256"]
                or result["outcome"]["status"] != "failed" or failure.get("code") != error
                or failure.get("ambiguous") is not posted or failure.get("retryable") is posted
                or result["provider_dispatched"] is not posted or result["raw_response"] is not None):
            raise PermissionError("the original posted timeout and preflight-only replay do not match their receipts")
        prior.append({"attempt": index, "attempt_id": attempt_id, "job_sha256": job["job_sha256"],
                      "result_sha256": result["result_sha256"], "terminal_sha256": digest(terminal),
                      "error": error, "provider_dispatched": posted, "ambiguous": posted})
    return prior, permit_sha, digest(read(turn / "attempt_01.replay_link.json"))


def _stage_request(root, contract, approval, boundary, original=None):
    state, memory, _ = boundary
    scope = seed_runner._scope(contract, approval, SEED, False)
    stage = seed_runner._stage(root, state, memory, scope, contract)
    if original is not None:
        message = original["payload"]["input"][1]
        stage = seed_runner.pin_boundary_user_message(stage, message, digest(message))
    return make_request(stage.binding, LOGICAL, stage.messages, contract.transport)


def construct(root, parent_release, parent, run):
    if run.name != RUN_NAME or run.parent != parent.parent or root == parent_release:
        raise PermissionError("use the registered sibling and its immutable overlay release")
    active, pause_sha = _retirement(parent)
    old, old_approval, boundary, backend = _original(parent_release, parent)
    sources = _source_hashes(root, old)
    state, memory, _ = boundary
    scope = seed_runner._scope(old, old_approval, SEED, False)
    if (read(active / "source.json") != asdict(state) or read(active / "boundary_memory.json") != memory
            or read(active.parent / "boundary.json") != {
                "formal_run": scope, **old.profile["resume_boundaries"][str(SEED)],
            }):
        raise PermissionError("the original L397 CUDA boundary or its memory changed")
    turn = active / "turn_00"
    original = read(turn / "request.json")
    binding = original["binding"]
    if (original["request_id"] != REQUEST or original["payload_sha256"] != PAYLOAD
            or read(turn / "formal_request_authorization.json") != seed_runner._selection_grant(scope, REQUEST)
            or binding["packet_sha256"] != digest(read(active / AUTHORITATIVE_PACKET_FILE))
            or _stage_request(root, old, old_approval, boundary) != original):
        raise PermissionError("the exact staged R2 request no longer matches its original scientific boundary")
    prior, permit, link = _prior_attempts(parent, turn, original, old, old_approval)
    transport = replace(old.transport, logical_replies=1, transport_attempts=1)
    descriptor = {**old.profile["resume_boundaries"][str(SEED)],
                  "user_message_sha256": digest(original["payload"]["input"][1])}
    profile = {
        **old.profile, "target_labels": 429, "transport": asdict(transport),
        "request_timeouts": asdict(TIMEOUTS), "sse_inactivity_seconds": SSE_INACTIVITY_SECONDS,
        "resume_boundaries": {str(SEED): descriptor},
    }
    predecessor = {
        "dispatcher_module": DISPATCHER, "parent_run_directory": str(parent),
        "parent_release_directory": str(parent_release), "parent_contract_sha256": old.digest,
        "parent_authorization_sha256": digest(old_approval), "parent_seed_pause_sha256": pause_sha,
        "seed": SEED, "round": ROUND, "logical_index": LOGICAL, "prior_request_id": REQUEST,
        "prior_attempts": prior, "prior_replay_authorization_sha256": permit, "prior_replay_link_sha256": link,
        "payload_sha256": PAYLOAD, "model_identity": old.predecessor["model_identity"],
        **{key: binding[key] for key in previous.MESSAGE_HASHES},
        "bridge_wait_seconds": TIMEOUTS.bridge_wait_seconds, "execution_budget": previous.ONE_SHOT_BUDGET,
        "prior_execution_and_billing": "unresolved", "parent_fit_audit_sha256": state.provenance["fit_audit_sha256"],
        "public_resource_receipt_sha256": backend.cache_receipt_sha256,
    }
    contract = replace(old, profile=profile, transport=transport, source_sha256=sources,
                       run_directory=str(run), predecessor=predecessor)
    return contract, boundary, backend, original


def _completion(run, contract, approval, boundary, backend):
    seed = seed_runner.seed_directory(run, SEED)
    active = seed_runner.round_directory(seed, ROUND)
    following = seed_runner.round_directory(seed, ROUND + 1)
    scope = seed_runner._scope(contract, approval, SEED, False)
    state, memory = seed_runner._successor(active, boundary[0], following, scope, contract.transport, contract, backend)
    remaining = 36 - state.round
    once(run / "extension_completion.json", {
        "contract_sha256": contract.digest, "seed": SEED, "round": state.round, "active_labels": 429,
        "checkpoint_sha256": state.checkpoint, "state_sha256": digest(asdict(state)), "memory_sha256": digest(memory),
        "transition_sha256": file_digest(following / "transition.json"),
        "original_future_budget": {"first_logical_requests": remaining,
                                   "maximum_logical_requests": remaining * 3,
                                   "maximum_physical_attempts": remaining * 6},
    })


def execute(root, parent, run, contract, approval, boundary, backend, original, *, resume):
    _retirement(parent, expected=contract.predecessor["parent_seed_pause_sha256"])
    request = _stage_request(root, contract, approval, boundary, original)
    if not previous._same_science(request, original):
        raise PermissionError("the child request differs from the exact original R2 payload or scientific bindings")
    with scheduler._lock(run / "coordinator.lock", busy="the R2 extension is already running"):
        if scheduler._pause_requested(run) and not resume:
            raise PermissionError("the R2 extension is paused; use resume")
        once(run / "formal_contract.json", contract.record())
        once(run / "authorization.json", approval)
        once(run / "provider_model_lock.json", contract.predecessor["model_identity"])
        once(run / "extension_request_scope.json", {"request_id": request["request_id"], "payload_sha256": PAYLOAD,
            "seed": SEED, "round": ROUND, "logical_index": LOGICAL, "attempt": 0})
        if resume:
            (run / "pause.request").unlink(missing_ok=True)
            (seed_runner.seed_directory(run, SEED) / "pause.request").unlink(missing_ok=True)
        scheduler._atomic_status(run / "coordinator_generation.json", {"id": uuid4().hex, "pid": os.getpid()})
        return _run(root, parent, run, contract, approval, boundary, backend, original, request)


def _run(root, parent, run, contract, approval, boundary, backend, original, request):
    status = {"seed": SEED, "pid": os.getpid(), "gpu": contract.gpu_slots[0]}
    path = run / "workers" / f"seed_{SEED}.json"
    scheduler._atomic_status(path, {**status, "status": "initializing"})
    try:
        call, options = scheduler._provider_wiring(run, contract, approval, None, SEED)
        managed = options["managed_attempt"]
        managed.wait_seconds = contract.predecessor["bridge_wait_seconds"]

        def one_request(candidate, config, started):
            _retirement(parent, expected=contract.predecessor["parent_seed_pause_sha256"])
            if not previous._one_attempt(request, contract.transport, contract.predecessor["prior_attempts"],
                                         candidate, config, started):
                raise TransportStop("this recovery authorizes only one exact new physical attempt")
            return managed(candidate, config, started)

        result = seed_runner.run_seed(
            root=root, run_directory=run, contract=contract, authorization=approval, seed=SEED, backend=backend,
            call=call, resume_boundary=boundary, boundary_user_message=original["payload"]["input"][1],
            transport_options={"managed_attempt": one_request},
            fit_slot=partial(scheduler.gpu_slot, run, SEED, status["gpu"], contract.lease_root),
        )
        if result.status == "complete":
            _completion(run, contract, approval, boundary, backend)
        scheduler._atomic_status(path, {**status, **asdict(result)})
        return result
    except Exception as error:
        phase = "paused" if scheduler._pause_requested(run, SEED) else (
            "bridge_pending" if isinstance(error, BridgePending) else "failed")
        scheduler._atomic_status(path, {**status, "status": phase, "error_type": type(error).__name__,
                                        "reason": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run", "resume"))
    for name in ("continuation-release", "continuation-run", "run-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parent_release, parent, run = (path.resolve() for path in (args.continuation_release, args.continuation_run, args.run_dir))
    contract, boundary, backend, original = construct(ROOT, parent_release, parent, run)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        parser.error("run and resume require this exact one-attempt authorization")
    approval = verify_authorization(contract, read(args.authorization))
    result = execute(ROOT, parent, run, contract, approval, boundary, backend, original, resume=args.action == "resume")
    print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
