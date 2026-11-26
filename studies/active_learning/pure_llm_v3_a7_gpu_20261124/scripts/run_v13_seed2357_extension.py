"""Authorize one extra transport attempt for the exact V13 seed 2357 L365 request."""

import argparse
from dataclasses import asdict, replace
from functools import partial
import os
from pathlib import Path
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_STREAM_IPV4, TransportConfig, TransportStop, make_request, terminal_record,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import (
    BridgePending, read_bridge_json, validate_result, verify_job,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import AUTHORITATIVE_PACKET_FILE
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    ROOT, SSH_API, STUDY, RequestTimeouts, request_timeouts, verify_authorization,
)
from .run_prepared_five_seed import PreparedCohortContract, _qualification_model


PARENT_NAME = "a7-gpt62-astra-xhigh-ipv4-v13"
PARENT_SHA256 = "b17eee7b4d584d5522ca990b47aefdf9d628685557683c83c5bc24f20a998a85"
RUN_NAME = "a7-gpt62-astra-xhigh-ipv4-v13-2357-extension-01"
DISPATCHER = "dispatch_prepared_ipv4_recovery_2357"
SEED, START_ROUND, LOGICAL_INDEX = 2357, 1, 0
ONE_SHOT_BUDGET = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 1}
MESSAGE_HASHES = ("checkpoint_sha256", "packet_sha256", "wire_packet_sha256", "base_messages_sha256")


def _original_contract(parent):
    record = read(parent / "formal_contract.json")
    if parent.name != PARENT_NAME or record["run_directory"] != str(parent) or digest(record) != PARENT_SHA256:
        raise PermissionError("the seed 2357 extension requires the exact original V13 contract")
    transport = TransportConfig(**record["profile"]["transport"])
    contract = PreparedCohortContract(
        record["profile"], transport, record["source_sha256"], record["asset_sha256"],
        record["anchor_root"], tuple(record["gpu_slots"]), record["lease_root"],
        str(parent), record["api_execution"], record["predecessor"],
    )
    expected_transport = ("token4research", "https://token4research.cn/v1", "gpt-6.2-astra",
                          "gpt-6.2-astra", "xhigh", RESPONSES_STREAM_IPV4, 32768, 3600, 3, 2)
    actual_transport = (
        transport.provider, transport.base_url, transport.requested_model, transport.served_model,
        transport.reasoning_effort, transport.wire_mode, transport.max_output_tokens,
        transport.timeout_seconds, transport.logical_replies, transport.transport_attempts,
    )
    if (contract.record() != record or actual_transport != expected_transport
            or contract.api_execution != SSH_API or contract.gpu_slots != (0,)
            or contract.initial_labels != 333 or contract.batch_size != 32 or contract.target_labels != 1485
            or contract.profile["store"] is not False
            or request_timeouts(contract.profile) != RequestTimeouts(read_idle_seconds=1800, total_seconds=3600)):
        raise PermissionError("the original V13 protocol differs from the approved continuation scope")
    return contract, verify_authorization(contract, read(parent / "authorization.json"))


def _prior_attempts(parent, turn, request, contract, approval):
    """Verify both original reservations and their own unmodified terminal failures."""
    expected = {turn / f"attempt_{index:02d}.started.json" for index in range(2)}
    if set(turn.glob("attempt_*.started.json")) != expected or any(turn.glob("*.reconciliation.json")):
        raise PermissionError("the original round needs exactly its two unreconciled physical reservations")
    attempts = []
    for index in range(2):
        started = read(turn / f"attempt_{index:02d}.started.json")
        terminal = terminal_record(turn, index)
        name = f"{started['attempt_id']}.json"
        job = read_bridge_json(parent / "bridge" / "outbox" / name)
        result = read_bridge_json(parent / "bridge" / "results" / name)
        bound = verify_job(job, authorization_sha256=digest(approval), contract_sha256=contract.digest,
                           wire_representation=contract.profile["wire_representation"])
        validate_result(result, job)
        failure = result["outcome"].get("failure", {})
        if (bound != (request, contract.transport, started) or started["attempt"] != index
                or any(terminal.get(key) != value for key, value in started.items())
                or terminal["status"] != "stopped" or terminal.get("error") != "timeout"
                or terminal.get("ambiguous") is not True or terminal.get("provider_dispatched") is not True
                or terminal.get("circuit_open") is not False or "answer_sha256" in terminal
                or "reconciliation_sha256" in terminal
                or terminal.get("bridge_job_sha256") != job["job_sha256"]
                or terminal.get("bridge_result_sha256") != result["result_sha256"]
                or result["outcome"]["status"] != "failed" or failure.get("code") != "timeout"
                or failure.get("ambiguous") is not True or result["provider_dispatched"] is not True
                or result["raw_response"] is not None):
            raise PermissionError("the old physical attempt is not the sealed ambiguous timeout")
        attempts.append({"attempt": index, "attempt_id": started["attempt_id"], "job_sha256": job["job_sha256"],
                         "result_sha256": result["result_sha256"], "terminal_sha256": digest(terminal)})
    if attempts[0]["attempt_id"] == attempts[1]["attempt_id"]:
        raise PermissionError("the two old attempts must have distinct identities")
    return attempts


def _uncommitted_parent_round(parent):
    seed = seed_runner.seed_directory(parent, SEED)
    active = seed_runner.round_directory(seed, START_ROUND)
    following = seed_runner.round_directory(seed, START_ROUND + 1)
    selected = ("batch_freeze.json", "selection.json", "memory.json")
    revealed = ("fit_authorization.json", "authorized_feedback.json", "source.json", "transition.json")
    if any((active / name).exists() for name in selected) or any((following / name).exists() for name in revealed):
        raise PermissionError("the original round already has selection, reveal, or fit evidence")
    return seed, active


def _read_boundary(parent, contract, approval):
    seed, active = _uncommitted_parent_round(parent)
    descriptor = contract.profile["resume_boundaries"][str(SEED)]
    scope = seed_runner._scope(contract, approval, SEED, False)
    state_record = read(active / "source.json")
    memory = read(active / "boundary_memory.json")
    lane = contract.predecessor["lanes"][str(SEED)]
    prepared = Path(lane["boundary_directory"])
    if (read(seed / "boundary.json") != {"formal_run": scope, **descriptor}
            or descriptor != {"round": START_ROUND, "state_sha256": digest(state_record),
                              "memory_sha256": digest(memory)}
            or state_record != read(prepared / "state.json") or memory != read(prepared / "memory.json")
            or (state_record["seed"], state_record["round"], state_record["checkpoint"])
            != (SEED, START_ROUND, lane["checkpoint_sha256"])
            or lane["start_round"] != START_ROUND or len(state_record["provenance"]["expected_observed"]) != 365):
        raise PermissionError("the L365 state, memory, or checkpoint differs from the signed V13 boundary")
    cache = _fitted_cache(contract, lane, prepared, descriptor, state_record)
    return active, cache, seed_runner.FrozenState(**state_record), memory


def _fitted_cache(contract, lane, prepared, descriptor, state):
    # The fitted L365 state links its fit audit. The original preparation draft
    # links that fit audit and the resource receipt to the signed V13 boundary.
    draft = read(prepared.parents[1] / "five_seed_boundary_draft.json")
    row = draft["boundaries"][str(SEED)]
    evidence = row["evidence"]
    cache = Path(lane["cache_directory"]).resolve(strict=True)
    if (digest(draft) != contract.predecessor["prepared_draft_sha256"]
            or row["descriptor"] != descriptor or row["boundary_directory"] != str(prepared)
            or row["cache_directory"] != str(cache) or row["checkpoint_sha256"] != state["checkpoint"]
            or evidence["source"] != "sealed_v11"
            or evidence["contract_sha256"] != contract.predecessor["prior_v11_contract_sha256_by_seed"][str(SEED)]
            or evidence["fit_audit_sha256"] != state["provenance"]["fit_audit_sha256"]
            or file_digest(cache / "resource_receipt.json") != evidence["public_resource_receipt_sha256"]):
        raise PermissionError("the seed 2357 training cache differs from its signed fitted boundary")
    return cache


def construct(root, original_release, parent, run):
    """Compute the new paying contract without writing artifacts or contacting the provider."""
    if run.name != RUN_NAME or run.parent != parent.parent or root == original_release:
        raise PermissionError("use the registered sibling run directory and a separate immutable release")
    contract, approval = _original_contract(parent)
    sources = dict(contract.source_sha256)
    if any(file_digest(release / name) != expected for release in (root, original_release)
           for name, expected in sources.items()):
        raise PermissionError("a source bound by the original V13 contract changed or is missing")
    for module in ("run_v13_seed2357_extension", DISPATCHER, "dispatch_prompt_direct"):
        name = (STUDY / "scripts" / f"{module}.py").as_posix()
        sources[name] = file_digest(root / name)

    active, cache, state, memory = _read_boundary(parent, contract, approval)
    turn = active / "turn_00"
    request = read(turn / "request.json")
    scope = seed_runner._scope(contract, approval, SEED, False)
    binding = request["binding"]
    if ((binding["seed"], binding["round"], request["logical_index"]) != (SEED, START_ROUND, LOGICAL_INDEX)
            or binding["a7_formal"] != scope or binding["checkpoint_sha256"] != state.checkpoint
            or binding["packet_sha256"] != digest(read(active / AUTHORITATIVE_PACKET_FILE))
            or request["payload"]["store"] is not False):
        raise PermissionError("the archived request does not bind the unchanged scientific state and full packet")
    message = request["payload"]["input"][1]
    stage = seed_runner._stage(root, state, memory, scope, contract)
    stage = seed_runner.pin_boundary_user_message(stage, message, digest(message))
    if make_request(stage.binding, LOGICAL_INDEX, stage.messages, contract.transport) != request:
        raise PermissionError("the original V13 model input no longer matches the sealed scientific state")
    attempts = _prior_attempts(parent, turn, request, contract, approval)
    identity = _qualification_model({"model_identity": read(parent / "provider_model_lock.json")}, contract.transport)
    if identity != contract.predecessor["provider_qualification"]["model_identity"]:
        raise PermissionError("the original served-model lock differs from the qualified route")

    boundary = {**contract.profile["resume_boundaries"][str(SEED)], "user_message_sha256": digest(message)}
    transport = replace(contract.transport, logical_replies=1, transport_attempts=1)
    profile = {
        **contract.profile, "confirmation_seeds": [SEED], "qualification_seed": SEED,
        "requires_initial_qualification": False, "target_labels": 397, "api_parallelism": 1,
        "transport": asdict(transport), "resume_boundaries": {str(SEED): boundary},
    }
    predecessor = {
        "dispatcher_module": DISPATCHER, "parent_run_directory": str(parent),
        "parent_release_directory": str(original_release), "parent_contract_sha256": contract.digest,
        "parent_authorization_sha256": digest(approval), "seed": SEED, "round": START_ROUND,
        "logical_index": LOGICAL_INDEX, "prior_request_id": request["request_id"], "prior_attempts": attempts,
        "payload_sha256": request["payload_sha256"], "model_identity": identity,
        **{key: binding[key] for key in MESSAGE_HASHES},
        "prior_execution_and_billing": "unresolved", "execution_budget": ONE_SHOT_BUDGET,
    }
    extension = replace(contract, profile=profile, transport=transport, source_sha256=sources,
                        run_directory=str(run), predecessor=predecessor)
    return extension, (state, memory, START_ROUND), cache, request


def _same_science(candidate, original):
    """A new formal identity may carry only the original complete model payload."""
    before = {key: value for key, value in original["binding"].items() if key != "a7_formal"}
    after = {key: value for key, value in candidate["binding"].items() if key != "a7_formal"}
    return (candidate["request_id"] != original["request_id"] and before == after
            and candidate["logical_index"] == original["logical_index"]
            and candidate["payload"] == original["payload"]
            and candidate["payload_sha256"] == original["payload_sha256"])


def _extension_request(root, contract, approval, boundary, original):
    state, memory, _ = boundary
    stage = seed_runner._stage(root, state, memory, seed_runner._scope(contract, approval, SEED, False), contract)
    descriptor = contract.profile["resume_boundaries"][str(SEED)]
    stage = seed_runner.pin_boundary_user_message(
        stage, original["payload"]["input"][1], descriptor["user_message_sha256"],
    )
    candidate = make_request(stage.binding, LOGICAL_INDEX, stage.messages, contract.transport)
    if not _same_science(candidate, original):
        raise PermissionError("the extended attempt must have the exact original model input and scientific binding")
    return candidate


def _parent_paused(parent):
    seed, _ = _uncommitted_parent_round(parent)
    if not (seed / "pause.request").is_file():
        raise PermissionError("pause only the original V13 seed 2357 lane before the separately approved extension")


def _one_attempt(expected, transport, prior, candidate, config, started):
    return (candidate == expected and config == transport
            and (started["request_id"], started["attempt"]) == (expected["request_id"], 0)
            and started["attempt_id"] not in {item["attempt_id"] for item in prior})


def execute(root, parent, run, contract, approval, boundary, cache, original, *, resume):
    _parent_paused(parent)
    request = _extension_request(root, contract, approval, boundary, original)
    with scheduler._lock(run / "coordinator.lock", busy="the seed 2357 extension is already running"):
        if scheduler._pause_requested(run) and not resume:
            raise PermissionError("the seed 2357 extension is paused; use resume")
        once(run / "formal_contract.json", contract.record())
        once(run / "authorization.json", approval)
        once(run / "provider_model_lock.json", contract.predecessor["model_identity"])
        mark = {"seed": SEED, "round": START_ROUND, "logical_index": LOGICAL_INDEX, "attempt": 0,
                "request_id": request["request_id"], "payload_sha256": request["payload_sha256"]}
        once(run / "extension_request_scope.json", mark)
        status_path = run / "workers" / f"seed_{SEED}.json"
        status = {"seed": SEED, "pid": os.getpid(), "gpu": contract.gpu_slots[0]}
        with scheduler._lock(run / "workers" / f"seed_{SEED}.lock", busy="the seed 2357 extension still has a worker"):
            scheduler._atomic_status(run / "coordinator_generation.json", {"id": uuid4().hex, "pid": os.getpid()})
            if resume:
                (run / "pause.request").unlink(missing_ok=True)
                (seed_runner.seed_directory(run, SEED) / "pause.request").unlink(missing_ok=True)
            return _run_round(root, parent, run, contract, approval, boundary, cache, original,
                              request, status_path, status)


def _run_round(root, parent, run, contract, approval, boundary, cache, original, request, status_path, status):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    scheduler._atomic_status(status_path, {**status, "status": "initializing"})
    try:
        backend = TrainingBackend(root, cache, SEED)
        call, options = scheduler._provider_wiring(run, contract, approval, None, SEED)
        managed = options["managed_attempt"]

        def one_request(candidate, config, started):
            _parent_paused(parent)
            if not _one_attempt(request, contract.transport, contract.predecessor["prior_attempts"],
                                candidate, config, started):
                raise TransportStop("the signed extension allows only one exact physical attempt")
            return managed(candidate, config, started)

        result = seed_runner.run_seed(
            root=root, run_directory=run, contract=contract, authorization=approval, seed=SEED, backend=backend,
            call=call, resume_boundary=boundary, boundary_user_message=original["payload"]["input"][1],
            transport_options={"managed_attempt": one_request},
            fit_slot=partial(scheduler.gpu_slot, run, SEED, status["gpu"], contract.lease_root),
        )
        if result.status == "complete":
            old = seed_runner.round_directory(seed_runner.seed_directory(run, SEED), START_ROUND)
            next_round = old.parent / "round_02"
            state, memory = seed_runner._successor(
                old, boundary[0], next_round, seed_runner._scope(contract, approval, SEED, False),
                contract.transport, contract, backend,
            )
            if result.active_labels != 397 or state.round != START_ROUND + 1:
                raise ValueError("the completed extension did not produce the expected L397 boundary")
            remaining = 36 - state.round
            once(run / "extension_completion.json", {
                "contract_sha256": contract.digest, "seed": SEED, "round": state.round,
                "active_labels": result.active_labels, "checkpoint_sha256": state.checkpoint,
                "state_sha256": digest(asdict(state)), "memory_sha256": digest(memory),
                "transition_sha256": file_digest(next_round / "transition.json"),
                "original_future_budget": {"first_logical_requests": remaining,
                                           "maximum_logical_requests": remaining * 3,
                                           "maximum_physical_attempts": remaining * 3 * 2},
            })
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
    contract, boundary, cache, original = construct(ROOT, original_release, parent, run)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        parser.error("run and resume need the exact extension authorization file")
    approval = verify_authorization(contract, read(args.authorization))
    result = execute(ROOT, parent, run, contract, approval, boundary, cache, original, resume=args.action == "resume")
    print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
