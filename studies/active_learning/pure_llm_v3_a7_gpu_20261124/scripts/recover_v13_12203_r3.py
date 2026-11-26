"""Inspect or reattach seed 12203 R3 under the exact original V13 physical budget."""

import argparse
from dataclasses import asdict
from functools import partial
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_STREAM_IPV4, UNKNOWN_REPLAY_DECISION, TransportStop, _link_replay_attempt,
    authorize_ambiguous_replay, make_request, terminal_record,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import (
    BridgePending, read_bridge_json, validate_result, verify_job, verify_result,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import verify_authorization
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_prepared_five_seed as cohort


A7 = Path("/1/column_chromatography_prediction/purellm_a7")
RELEASE = A7 / "releases/ipv4-v13-5190067ed6d96089d1262337f45d23e29541adc0d1a589518d72a3754bee784b"
RUN = A7 / "runs/a7-gpt62-astra-xhigh-ipv4-v13"
PREPARATION = A7 / "preparation/a7-five-v12"
DRAFT = "80ece707d734b468f64814710c2db83cae189293d17cc2f8395916a49be64311"
CONTRACT = "b17eee7b4d584d5522ca990b47aefdf9d628685557683c83c5bc24f20a998a85"
APPROVAL = "d43ca2ba5a3ac93173466bdb638735805d0092b321dd42e7d8aac862af922410"
SEED, ROUND, TURN = 12203, 3, 0
ATTEMPT = "209c8285d97c450fac5dff50fd559bd1"
REQUEST = "f3ac75817f85484b3ee466ec9787cc44d61c88c245ee0a4bc9e2c4abd43b2f1e"
PAYLOAD = "29b2b6f416288dff87e9b9b1d45bd53b0322f1c071e5644ae1f4d0a555fbe909"
RESULT = "96fe904df2aa3699471e7e67c194914b840f5256cdc3e3303c35a8f56603dd28"
R1_ATTEMPT = "f478cb82243b4bca8499055fa0b3c6da"
R1_RESULT = "6977bea2fe405f869581bfd76cff1d1f4ccfbd3967a14a135d6e4979615ecef7"


def _report(state, **fields):
    print(dumps({"state": state, "contract_sha256": CONTRACT, "seed": SEED, "round": ROUND, **fields}), flush=True)


def _target(job):
    request, config, started = verify_job(job, contract_sha256=CONTRACT, authorization_sha256=APPROVAL)
    binding = request["binding"]
    if ((binding["seed"], binding["round"], request["logical_index"], started["attempt"]) != (SEED, ROUND, TURN, 0)
            or (request["request_id"], request["payload_sha256"], started["attempt_id"]) != (REQUEST, PAYLOAD, ATTEMPT)):
        raise PermissionError("the archived request is not the exact 12203 R3 first attempt")
    return request, config, started


def _failure(job, result):
    validate_result(result, job)
    failure = result["outcome"].get("failure", {})
    if (result["result_sha256"] != RESULT or result["provider_dispatched"] is not True
            or result["outcome"]["status"] != "failed" or result["raw_response"] is not None
            or failure.get("code") != "timeout" or failure.get("ambiguous") is not True
            or failure.get("retryable") is not False):
        raise PermissionError("the original result is not the exact paid ambiguous inactivity timeout")
    return {"authorization_sha256": APPROVAL, "request_id": REQUEST, "payload_sha256": PAYLOAD,
            "attempt_id": ATTEMPT, "job_sha256": job["job_sha256"], "result_sha256": RESULT,
            "mac_complete_response_absent": True, "mac_result_delivered": True}


def mac_evidence(spool):
    if read_bridge_json(spool / "run_binding.json") != {
        "contract_sha256": CONTRACT, "authorization_sha256": APPROVAL, "mock": False,
    }:
        raise PermissionError("the Mac journal is not the original V13 spool")
    name = f"{ATTEMPT}.json"
    saved = spool / "attempts" / ATTEMPT
    job = read_bridge_json(spool / "jobs" / name)
    _target(job)
    capture = (saved / "raw_provider_response.json", saved / "provider_capture_receipt.json")
    if any(path.exists() for path in capture):
        _report("use_archived_mac_completion")
        return
    if not (spool / "results" / name).is_file() or not (spool / "delivered" / name).is_file():
        _report("waiting_for_original_result")
        return
    result = read_bridge_json(spool / "results" / name)
    if result["outcome"]["status"] == "completed":
        _report("use_archived_mac_completion")
        return
    evidence = _failure(job, result)
    ack = read_bridge_json(spool / "delivered" / name)
    progress = read_bridge_json(saved / "provider_progress.json")
    if (ack != {"attempt_id": ATTEMPT, "result_sha256": RESULT}
            or (progress.get("attempt_id"), progress.get("job_sha256"), progress.get("phase"), progress.get("error_type"))
            != (ATTEMPT, job["job_sha256"], "transport_failure", "sse_inactivity_timeout")
            or progress.get("complete_terminal_captured") or any(path.exists() for path in capture)):
        raise PermissionError("the delivered Mac artifact does not prove an inactivity failure without completion")
    _report("sealed_mac_failure_no_completion", **evidence)


def _load():
    if Path.cwd().resolve() != RELEASE:
        raise PermissionError("run this operator launcher from the immutable original V13 release")
    contract, draft = cohort.construct(RELEASE, PREPARATION / "five_seed_boundary_draft.json", DRAFT,
                                       PREPARATION / "ipv4-v13-policy.json", RUN)
    approval = verify_authorization(contract, read(RUN / "authorization.json"))
    if (contract.digest != CONTRACT or digest(approval) != APPROVAL or contract.record() != read(RUN / "formal_contract.json")
            or contract.gpu_slots != (0,) or contract.transport.wire_mode != RESPONSES_STREAM_IPV4
            or contract.transport.transport_attempts != 2
            or read(RUN / "provider_model_lock.json") != contract.predecessor["provider_qualification"]["model_identity"]):
        raise PermissionError("the original V13 authorization, model, or execution budget changed")
    gate = contract.predecessor["ipv4_route_gate"]
    gate_seed, gate_round = gate["seed"], gate["verified_round"]
    fit = seed_runner.round_directory(seed_runner.seed_directory(RUN, gate_seed), gate_round) / "transition.json"
    gate_record = {"seed": gate_seed, "completed_rounds": gate_round,
                   "active_labels": contract.initial_labels + contract.batch_size * gate_round,
                   "checkpoint_sha256": read(fit)["checkpoint_sha256"], "transition_sha256": file_digest(fit),
                   "contract_sha256": CONTRACT}
    if read(RUN / "ipv4_route_gate.json") != gate_record:
        raise PermissionError("the original approved CUDA gate is missing or changed")
    row = draft["boundaries"][str(SEED)]
    return contract, approval, cohort.read_boundary(row, SEED, PREPARATION), row, gate["bridge_wait_seconds"]


def _paths(run):
    seed = seed_runner.seed_directory(run, SEED)
    active = seed_runner.round_directory(seed, ROUND)
    return seed, active, active / "turn_00"


def _inspect(run):
    seed, active, turn = _paths(run)
    if (seed_runner.round_directory(seed, ROUND + 1) / "transition.json").is_file():
        return "R3_already_sealed_resume_normally"
    if (active / "batch_freeze.json").is_file():
        return "finish_frozen_R3"
    first, second = (turn / f"attempt_{i:02d}.terminal.json" for i in range(2))
    if not first.is_file():
        return "waiting_for_original_terminal"
    if any(turn.glob("attempt_00.recovery*.json")) or (turn / "attempt_00.reconciliation.json").exists():
        return "use_original_reconciliation"
    if read(first).get("status") == "completed":
        return "use_original_completion"
    if second.is_file() and read(second).get("status") != "completed":
        return "physical_budget_exhausted"
    permitted = (turn / "attempt_00.replay_authorization.json").is_file()
    if not permitted and ((turn / "attempt_01.started.json").exists() or second.exists()):
        raise PermissionError("the second physical slot has already been reserved under different evidence")
    return "reattach_existing_replay" if permitted else "sealed_original_candidate"


def _previous_replay(run, seed, contract):
    """Audit the sealed R1 path without granting or supplying a replay for it again."""
    active = seed_runner.round_directory(seed, 1)
    freeze = read(active / "batch_freeze.json")
    turn = active / "turn_00"
    permit_sha = digest(read(turn / "attempt_00.replay_authorization.json"))
    replay = terminal_record(turn, 0, replay_authorization_sha256=permit_sha)
    started = read(turn / "attempt_01.started.json")
    _link_replay_attempt(turn, started, replay, allow_create=False)
    if ((freeze["accepted_logical_index"], freeze["accepted_attempt"]) != (0, 1)
            or started["attempt_id"] != R1_ATTEMPT):
        raise PermissionError("R1 lacks its previously accepted replay")
    name = f"{R1_ATTEMPT}.json"
    job = read_bridge_json(run / "bridge" / "outbox" / name)
    result = read_bridge_json(run / "bridge" / "results" / name)
    bound = verify_job(job, contract_sha256=CONTRACT, authorization_sha256=APPROVAL,
                       wire_representation=contract.profile["wire_representation"])
    terminal = terminal_record(turn, 1)
    reply = verify_result(result, job)
    if (bound != (read(turn / "request.json"), contract.transport, started)
            or result["result_sha256"] != R1_RESULT or terminal["bridge_result_sha256"] != R1_RESULT
            or terminal["bridge_job_sha256"] != job["job_sha256"]
            or freeze["provider_response_id"] != reply.response_id or terminal["text"] != reply.text):
        raise PermissionError("the completed R1 replay receipt no longer matches its freeze")
    return permit_sha


def _original_result(run, active, turn, contract, approval, path):
    job = read_bridge_json(run / "bridge" / "outbox" / f"{ATTEMPT}.json")
    result = read_bridge_json(run / "bridge" / "results" / f"{ATTEMPT}.json")
    request, config, started = _target(job)
    evidence = _failure(job, result)
    terminal = terminal_record(turn, 0)
    selection = seed_runner._selection_grant(seed_runner._scope(contract, approval, SEED, False), REQUEST)
    exact_evidence = {"state": "sealed_mac_failure_no_completion", "contract_sha256": CONTRACT,
                      "seed": SEED, "round": ROUND, **evidence}
    if (read(path) != exact_evidence or (request, config, started)
            != (read(turn / "request.json"), contract.transport, read(turn / "attempt_00.started.json"))
            or terminal.get("status") != "stopped" or terminal.get("error") != "timeout"
            or terminal.get("provider_dispatched") is not True or terminal.get("ambiguous") is not True
            or terminal.get("circuit_open") is not False or terminal.get("bridge_result_sha256") != RESULT
            or terminal.get("bridge_job_sha256") != job["job_sha256"]
            or any(terminal.get(key) != value for key, value in started.items())
            or read(active / "authorization.json") != selection or read(turn / "formal_request_authorization.json") != selection):
        raise PermissionError("the signed scientific grant and sealed Mac/GPFS original failure do not match")
    return request, terminal, exact_evidence


def _permit(args, turn, terminal):
    required = {"decision": UNKNOWN_REPLAY_DECISION, "contract_sha256": CONTRACT, "request_id": REQUEST,
                "attempt_id": ATTEMPT, "bridge_result_sha256": RESULT, "original_terminal_sha256": digest(terminal),
                "next_attempt": 1, "accept_duplicate_billing_risk": True}
    saved = turn / "attempt_00.replay_authorization.json"
    if saved.is_file():
        permit = read(saved)
        if any(permit.get(key) != value for key, value in required.items()):
            raise PermissionError("the durable replay authorization identifies another original request")
        return permit
    if not (args.accept_duplicate_billing_risk and args.authorized_by and args.authorized_by.strip()
            and args.authorization_reference and args.authorization_reference.strip()):
        raise PermissionError("a new replay requires explicit acceptance of duplicate billing and its authorization reference")
    return {**required, "authorized_by": args.authorized_by, "authorization_reference": args.authorization_reference}


def _restage(contract, approval, boundary, backend, seed, original):
    state, memory, start = boundary
    scope = seed_runner._scope(contract, approval, SEED, False)
    for index in range(start, ROUND):
        before = seed_runner.round_directory(seed, index)
        after = seed_runner.round_directory(seed, index + 1)
        state, memory = seed_runner._successor(before, state, after, scope, contract.transport, contract, backend)
    stage = seed_runner._stage(RELEASE, state, memory, scope, contract)
    if make_request(stage.binding, TURN, stage.messages, contract.transport) != original:
        raise PermissionError("this process cannot reproduce the exact already archived R3 request")


def activate(args, contract, approval, boundary, row, bridge_wait):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    if args.mac_evidence is None:
        raise PermissionError("activation needs sealed failure evidence from the original Mac spool")
    worker = RUN / "workers" / f"seed_{SEED}.json"
    with scheduler._lock(worker.with_suffix(".lock"), busy="the V13 seed 12203 worker still owns this lane"):
        if scheduler._pause_requested(RUN, SEED):
            raise PermissionError("the V13 run or seed 12203 is paused")
        mode = _inspect(RUN)
        if mode not in {"sealed_original_candidate", "reattach_existing_replay"}:
            _report(mode)
            return
        seed, active, turn = _paths(RUN)
        if mode == "sealed_original_candidate" and any(path.parent != turn for path in active.glob("turn_*/request.json")):
            raise PermissionError("R3 has already staged another logical decision")
        request, terminal, evidence = _original_result(RUN, active, turn, contract, approval, args.mac_evidence)
        previous_permit = _previous_replay(RUN, seed, contract)
        backend = TrainingBackend(RELEASE, row["cache_directory"], SEED)
        _restage(contract, approval, boundary, backend, seed, request)
        permit = _permit(args, turn, terminal)
        permit_sha = digest(permit)
        authorize_ambiguous_replay(active, request, authorization=permit, config=contract.transport,
                                   expected_authorization_sha256=permit_sha)
        once(RUN / "operator" / "v13-12203-r3-recovery" / f"{permit_sha}.json", {
            "contract_sha256": CONTRACT, "request_id": REQUEST, "original_attempt_id": ATTEMPT,
            "mac_evidence_sha256": digest(evidence), "archived_r1_replay_sha256": previous_permit,
            "replay_authorization_sha256": permit_sha, "launcher_sha256": file_digest(Path(__file__)),
        })
        _report(mode, replay_authorization_sha256=permit_sha)
        _resume(worker, contract, approval, boundary, backend, bridge_wait, request, permit_sha)


def _resume(worker, contract, approval, boundary, backend, bridge_wait, request, permit_sha):
    status = {"seed": SEED, "pid": os.getpid(), "gpu": contract.gpu_slots[0]}
    scheduler._atomic_status(worker, {**status, "status": "initializing"})
    try:
        call, options = scheduler._provider_wiring(RUN, contract, approval, None, SEED)
        managed = options["managed_attempt"]
        managed.wait_seconds = bridge_wait

        def checked_attempt(candidate, config, started):
            binding = candidate["binding"]
            if binding["seed"] != SEED or not ROUND <= binding["round"] < contract.rounds:
                raise TransportStop("a V13 recovery cannot submit a finished round or another seed")
            if binding["round"] == ROUND and candidate["logical_index"] == TURN and (
                candidate != request or config != contract.transport or started["attempt"] != 1
            ):
                raise TransportStop("R3's audited replay permits only attempt 1 of its exact original request")
            return managed(candidate, config, started)

        result = seed_runner.run_seed(
            root=RELEASE, run_directory=RUN, contract=contract, authorization=approval, seed=SEED, backend=backend,
            call=call, resume_boundary=boundary,
            transport_options={"managed_attempt": checked_attempt, "replay_authorizations": {REQUEST: permit_sha}},
            fit_slot=partial(scheduler.gpu_slot, RUN, SEED, status["gpu"], contract.lease_root),
        )
        scheduler._atomic_status(worker, {**status, **asdict(result)})
        _report(result.status, active_labels=result.active_labels, completed_rounds=result.completed_rounds)
    except Exception as error:
        phase = "paused" if scheduler._pause_requested(RUN, SEED) else (
            "bridge_pending" if isinstance(error, BridgePending) else "failed")
        scheduler._atomic_status(worker, {**status, "status": phase, "error_type": type(error).__name__,
                                        "reason": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "mac-evidence", "activate"))
    parser.add_argument("--mac-spool", type=Path, default=Path.home() / ".config/purellm/spool" / CONTRACT)
    parser.add_argument("--mac-evidence", type=Path)
    parser.add_argument("--accept-duplicate-billing-risk", action="store_true")
    parser.add_argument("--authorized-by")
    parser.add_argument("--authorization-reference")
    args = parser.parse_args()
    if args.action == "mac-evidence":
        mac_evidence(args.mac_spool)
        return
    if args.action == "inspect":
        _report(_inspect(RUN))
        return
    os.umask(0o077)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    activate(args, *_load())


if __name__ == "__main__":
    main()
