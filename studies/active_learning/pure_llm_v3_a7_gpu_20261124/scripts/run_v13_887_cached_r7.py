"""Import the known completed seed 887 round 7 without a provider submission."""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from functools import partial
import os
from pathlib import Path
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import ResponseFormatError, parse_response
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.runner import _validation
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    ProviderReply, TransportStop, make_request, terminal_record,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import (
    ATTEMPT_ID, BridgePending, read_bridge_json, validate_result, verify_job,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import AUTHORITATIVE_PACKET_FILE
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import OFFLINE, ROOT, STUDY, verify_authorization
from . import run_v13_bounded_sse_recovery as bounded
from . import run_v13_resilient_failure_recovery as failure


SEED, ROUND = 887, 7
DISPATCHER = "dispatch_v13_887_cached_r7"
OLD = bounded.RunIdentity(
    "-887-resilient-failure-01", "f4e2782ba061c1c00dd9a1f46198380bc095492e38b467e6d9bf20b4b828b33d",
    "37bcb9bd06d31622ce29ff56363bf1d9bfb8772e2e3016e133fb3e9bcd29b7b3",
)
RUN_NAME = f"{bounded.PREFIX}-887-cached-r7-01"
BUDGET = {"first_logical_requests": 28, "maximum_logical_requests": 84, "maximum_physical_attempts": 168}
IMPORT_BUDGET = {"maximum_logical_requests": 1, "maximum_physical_attempts": 0}
ATTEMPTS = (
    ("15dc3b1b993d449690b004700245c98d", "839bd8a576de8d3148b518cf7d992cf11c78ffbd43009810af8ed64d4114c0b2",
     "cb3d198112709b12cf78542527cabd6d8e075fb85cc309654ea795a5f85892d1"),
    ("232ab8efcc6a414c851c3025278bfd3d", "dd2658bef869ee213f1bf708102bc7580b55a71e337443f713301480e1166e44",
     "08a77585687d9c77676a14f72276f64099b72ff59fc18fff183e2643203a4848"),
)
RESULT_SHA = "7b2a7440c29f3cefeca62672ec162c365c596c0e0d9518bafec6f2be81c434a6"
RAW_SHA = "a943a2f542ce4ff53a77b135d8ef2d882d6176a133618e047ff0af44a1200e7b"
PAYLOAD_SHA = "f987f5490942a6817d996e7c19176b48f73a5df5376e964d8a6c3be58feccf08"
CHECKPOINT_SHA = "00d4bd97b632c8a38213e1d950667af052cd34e2ebfef333ce80a5d61f7460af"
PACKET_SHA = "73adb6e960ed5550530d0310cbb3bfeec5d08555a443719e8dc332dfe5af57ff"
WIRE_SHA = "68a3491b743845141750a4baf9eaf4a1d4893d126c251eb30417a7558beea54b"
FINAL_SHA = "d77e4c98c17e3252ccb2676aa6c9b6409004641d2f8c432cf2b6d42ca3fe6348"
PARSED_SHA = "7afd7a36e70af95195c16ce6416853bc8b0d0c7ccf52eff8f394688bcb4f251a"
PLAN_SHA = "d8acfb0d43235068f932f18b7f09c08554e811060f04878fe1a5a9afe8386d47"
PATHS_SHA = "6b3ea2afcec9cef75413d41293c9826ca1d16fc01e81966fbc9d93864f160c44"
ORDER_SHA = "e1206991a537ae92e6ec65b2b3f7306cf94ed2e69cb5dc9310867fb0b44f9861"
OLD_TERMINAL_SHA = "68448c0d8edf22fb40aad51d8582630ff381849badd8216544a8e6373f4fbe39"
PAUSE_MARKER_SHA = "e2a003428ab33bd15ace0ed8a498579f59639a85950f7676eca6e256842eb41b"
PAUSED_WAIT_REASON = "formal run paused while waiting for the existing Mac attempt"
BOUNDARY_FILES = {
    "round_07/source.json": "3af9bc2c889c48f8ed994faef836448996c1125f7ccafa111f4e76ed52c75ba2",
    "round_07/transition.json": "412e9b9215968c966b7349ba1c07c6ec4de87f2ef07f446adefba6671fc2b60a",
    "round_06/memory.json": "20bba37d6f17e7e522b0a99378adcb8088562769f7bb937a65097ff6c69783ad",
}


def _final_plan(raw):
    if digest(raw) != RAW_SHA or raw.get("status") != "completed":
        raise PermissionError("cached answer is not the exact complete provider object")
    final = [item for item in raw["output"] if item.get("phase") == "final_answer"]
    if (len(final) != 1 or final[0] != raw["output"][-1]
            or (final[0].get("type"), final[0].get("role"), final[0].get("status"))
            != ("message", "assistant", "completed")
            or len(final[0]["content"]) != 1 or final[0]["content"][0].get("type") != "output_text"):
        raise PermissionError("cached answer lacks its one final message")
    text = final[0]["content"][0]["text"]
    plan = parse_response(text)
    if digest(text) != FINAL_SHA or digest(plan) != PARSED_SHA:
        raise PermissionError("cached final message differs from the verified plan")
    normalized = deepcopy(plan)
    paths = []
    for index, choice in enumerate(normalized["choices"]):
        for slot, reference in enumerate(choice["references"]):
            if (reference.get("reference_kind") != "observed" or type(reference.get("reference_id")) is not str
                    or reference.get("observed_id") != reference["reference_id"]
                    or reference.get("candidate_id") != choice["candidate_id"]):
                raise PermissionError("cached reference does not carry both exact nested mirrors")
            for alias in ("observed_id", "candidate_id"):
                del reference[alias]
                paths.append(f"/choices/{index}/references/{slot}/{alias}")
    if (len(paths) != 72 or digest(paths) != PATHS_SHA or digest(normalized) != PLAN_SHA
            or digest([choice["candidate_id"] for choice in normalized["choices"]]) != ORDER_SHA):
        raise PermissionError("cached plan or ordered candidates changed during alias removal")
    return text, normalized, paths


def _correction_failure(result, job):
    validate_result(result, job)
    outcome = result["outcome"]
    failure = outcome.get("failure", {})
    if (outcome["status"] != "failed" or result["raw_response"] is not None
            or (failure.get("reply_identity") or {}).get("provider_status") == "completed"):
        raise PermissionError("leave a successfully returned correction with the original runner")
    return failure


def _paused_correction_receipt(started, job, result, claim):
    failure = _correction_failure(result, job)
    if (result["provider_dispatched"] is not True or set(claim) != {"attempt_id", "dispatcher_id", "job_sha256"}
            or (claim["attempt_id"], claim["job_sha256"]) != (job["attempt_id"], job["job_sha256"])
            or not isinstance(claim["dispatcher_id"], str) or not ATTEMPT_ID.fullmatch(claim["dispatcher_id"])):
        raise PermissionError("the paused correction has no exact dispatched bridge claim")
    return {"started_sha256": digest(started), "claim": claim, "failure_sha256": digest(failure)}


def _old_receipt(run, active, index, transport, wire_representation, *, paused_waiter=False):
    turn = active / f"turn_{index:02d}"
    attempt, request_id, job_sha = ATTEMPTS[index]
    without_terminal = index == 1 and paused_waiter
    expected = {turn / "attempt_00.started.json"}
    if not without_terminal:
        expected.add(turn / "attempt_00.terminal.json")
    if set(turn.glob("attempt_*")) != expected:
        raise PermissionError("both archived turns must have exactly the approved physical reservation")
    request, started = read(turn / "request.json"), read(turn / "attempt_00.started.json")
    terminal = None if without_terminal else terminal_record(turn, 0)
    name = f"{attempt}.json"
    job = read_bridge_json(run / "bridge/outbox" / name)
    result = read_bridge_json(run / "bridge/results" / name)
    if (verify_job(job, authorization_sha256=OLD.approval, contract_sha256=OLD.contract,
                   wire_representation=wire_representation) != (request, transport, started)
            or (job["attempt_id"], request["request_id"], job["job_sha256"], request["logical_index"])
            != (attempt, request_id, job_sha, index)
            or read(turn / "formal_request_authorization.json") != seed_runner._selection_grant(
                request["binding"]["a7_formal"], request_id)
            or (request["binding"]["seed"], request["binding"]["round"], started["attempt"]) != (SEED, ROUND, 0)
            or result["result_sha256"] != digest({k: v for k, v in result.items() if k != "result_sha256"})
            or terminal is not None and (any(terminal.get(key) != value for key, value in started.items())
                or terminal.get("bridge_job_sha256") != job_sha
                or terminal.get("bridge_result_sha256") != result["result_sha256"]
                or terminal.get("provider_dispatched") is not result["provider_dispatched"])):
        raise PermissionError("sealed archived request, result or terminal identity differs")
    evidence = {"attempt_id": attempt, "request_id": request_id, "job_sha256": job_sha,
                "result_sha256": result["result_sha256"], "terminal_sha256": digest(terminal) if terminal else None}
    if index == 0:
        # This one archived row used the original envelope text, which included commentary.
        original = result["outcome"]["reply"]
        if (result["result_sha256"] != RESULT_SHA or result["raw_response_sha256"] != RAW_SHA
                or result["provider_dispatched"] is not True or terminal["status"] != "completed"
                or file_digest(turn / "attempt_00.terminal.json") != OLD_TERMINAL_SHA
                or original["text"] != terminal["text"] or digest(terminal["text"]) != terminal["answer_sha256"]
                or original["response_id"] != terminal["response_id"]
                or original["served_model"] != terminal["served_model"] or original["usage"] != terminal["usage"]
                or original["provider_request_id"] != terminal["provider_request_id"]
                or read(turn / "decision.json") != {"state": "malformed_complete_response"}):
            raise PermissionError("the original cached completion or prior parser failure changed")
        try:
            parse_response(terminal["text"])
        except ResponseFormatError:
            pass
        else:
            raise PermissionError("the original parser failure no longer reproduces")
    elif without_terminal:
        claim = read_bridge_json(run / "bridge/claims" / name)
        evidence.update(_paused_correction_receipt(started, job, result, claim))
    else:
        failure = _correction_failure(result, job)
        if (terminal["status"] != "stopped" or terminal.get("error") != failure["code"]
                or terminal.get("ambiguous") is not failure["ambiguous"]):
            raise PermissionError("leave a successfully returned correction with the original runner")
    return request, result, evidence


def _verify_old_retired(old, directory):
    worker = read(old / f"workers/seed_{SEED}.json")
    active, following = (seed_runner.round_directory(directory, index) for index in (ROUND, ROUND + 1))
    if (worker.get("seed") != SEED or worker.get("status") not in {"failed", "paused"}
            or {path.name for path in active.glob("turn_*")} != {"turn_00", "turn_01"}
            or any((active / name).exists() for name in ("batch_freeze.json", "selection.json", "memory.json"))
            or (following.is_dir() and any(following.iterdir()))):
        raise PermissionError("wait for the old worker to stop with no selected or fitted batch")
    if worker["status"] == "failed":
        return active, None
    pause_marker = old / "pause.request"
    if (worker.get("error_type") != "BridgePending" or worker.get("reason") != PAUSED_WAIT_REASON
            or not pause_marker.is_file() or pause_marker.is_symlink() or file_digest(pause_marker) != PAUSE_MARKER_SHA):
        raise PermissionError("the old worker was not fenced while waiting for the exact correction")
    return active, {"pause_marker_sha256": PAUSE_MARKER_SHA, "worker_status_sha256": digest(worker)}


def construct(root, old_release, old, run):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    if run.name != RUN_NAME or run.parent != old.parent or root == old_release:
        raise PermissionError("the cached continuation requires its own sibling run and source release")
    previous, approval = bounded._archived(old, old_release, OLD)
    directory = seed_runner.seed_directory(old, SEED)
    active, paused_waiter = _verify_old_retired(old, directory)
    if any(file_digest(directory / name) != sha for name, sha in BOUNDARY_FILES.items()):
        raise PermissionError("the archived L557 fitted boundary files changed")
    args = (previous.transport, previous.profile["wire_representation"])
    request, result, first = _old_receipt(old, active, 0, *args)
    _, _, second = _old_receipt(old, active, 1, *args, paused_waiter=paused_waiter is not None)
    binding = request["binding"]
    if ((request["payload_sha256"], binding["checkpoint_sha256"], binding["packet_sha256"], binding["wire_packet_sha256"])
            != (PAYLOAD_SHA, CHECKPOINT_SHA, PACKET_SHA, WIRE_SHA)
            or result["model_identity"] != previous.predecessor["model_identity"]
            or read(old / "provider_model_lock.json") != result["model_identity"]):
        raise PermissionError("the cached request changed scientific input or provider model identity")

    cache = Path(previous.predecessor["cache_directory"]).resolve(strict=True)
    if file_digest(cache / "resource_receipt.json") != previous.predecessor["public_resource_receipt_sha256"]:
        raise PermissionError("the original public training cache changed")
    backend = TrainingBackend(old_release, cache, SEED)
    previous_round = seed_runner.round_directory(directory, ROUND - 1)
    start = seed_runner.FrozenState(**read(previous_round / "source.json"))
    scope = seed_runner._scope(previous, approval, SEED, False)
    previous_boundary = previous.profile["resume_boundaries"][str(SEED)]
    if (digest(asdict(start)) != previous_boundary["state_sha256"]
            or digest(read(previous_round / "boundary_memory.json")) != previous_boundary["memory_sha256"]
            or read(directory / "boundary.json") != {"formal_run": scope, **previous_boundary}):
        raise PermissionError("the archived L525 predecessor state changed")
    state, memory = seed_runner._successor(previous_round, start, active, scope,
                                          previous.transport, previous, backend)
    stage = seed_runner._stage(old_release, state, memory, scope, previous)
    user = request["payload"]["input"][1]
    stage = seed_runner.pin_boundary_user_message(stage, user, digest(user))
    text, plan, paths = _final_plan(result["raw_response"])
    validation = _validation(stage, plan)
    # The archived lossless packet is SHA-exact even when its unsent message keys were serialized in a different order.
    archived_binding = {**stage.binding, "authoritative_base_messages_sha256":
                        binding["authoritative_base_messages_sha256"]}
    if (digest(read(active / AUTHORITATIVE_PACKET_FILE)) != PACKET_SHA or digest(stage.packet) != PACKET_SHA
            or make_request(archived_binding, 0, stage.messages, previous.transport) != request
            or not validation.valid or validation.valid_choice_count != 32):
        raise PermissionError("cached final answer is not fully legal on the exact authoritative A7 stage")

    descriptor = {"round": ROUND, "state_sha256": digest(asdict(state)),
                  "memory_sha256": digest(memory), "user_message_sha256": digest(user)}
    proof = {
        "dispatcher_module": DISPATCHER, "previous_run_directory": str(old), "previous_release_directory": str(old_release),
        "previous_contract_sha256": previous.digest, "previous_authorization_sha256": digest(approval),
        "seed": SEED, "round": ROUND, "payload_sha256": PAYLOAD_SHA, "checkpoint_sha256": CHECKPOINT_SHA,
        "packet_sha256": PACKET_SHA, "wire_packet_sha256": WIRE_SHA, "prompt_sha256": binding["prompt_sha256"],
        "model_identity": result["model_identity"], "cache_directory": str(cache),
        "execution_budget": BUDGET, "recovery_budget": IMPORT_BUDGET,
        "cached_import": {"prior": [first, second], "raw_response_sha256": RAW_SHA, "final_text_sha256": digest(text),
                          "normalized_plan_sha256": digest(plan), "removed_paths": paths,
                          "ordered_candidate_ids_sha256": ORDER_SHA, "archive_boundary_sha256": BOUNDARY_FILES,
                          "archived_terminal_file_sha256": OLD_TERMINAL_SHA, "provider_post": False,
                          **({"paused_waiter": paused_waiter} if paused_waiter else {})},
    }
    profile = {**previous.profile, "resume_boundaries": {str(SEED): descriptor},
               "replay_eligible_from_round": ROUND + 1}
    sources = {name: file_digest(root / name) for name in previous.source_sha256}
    allowed = failure.OVERLAYS | {(STUDY / "a7_gpu/bridge_protocol.py").as_posix(),
                                 (OFFLINE / "pure_llm_v3/runner.py").as_posix()}
    if {name for name, sha in sources.items() if sha != previous.source_sha256[name]} - allowed:
        raise PermissionError("cached continuation changed source outside the audited API extraction and dispatcher")
    for module in ("run_v13_887_cached_r7", DISPATCHER):
        name = (STUDY / "scripts" / f"{module}.py").as_posix()
        sources[name] = file_digest(root / name)
    contract = replace(previous, profile=profile, source_sha256=sources, run_directory=str(run), predecessor=proof)
    return contract, (state, memory, ROUND), cache, request, result["outcome"]["reply"], plan


def execute(root, run, old, contract, approval, boundary, cache, archived, old_reply, plan, *, resume, import_only):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    scope = seed_runner._scope(contract, approval, SEED, False)
    user = archived["payload"]["input"][1]
    stage = seed_runner.pin_boundary_user_message(seed_runner._stage(root, *boundary[:2], scope, contract),
                                                  user, digest(user))
    expected = make_request(stage.binding, 0, stage.messages, contract.transport)
    if expected["payload"] != archived["payload"] or not _validation(stage, plan).valid:
        raise PermissionError("cached plan does not reproduce the approved scientific decision on the new stage")
    evidence = {"provider_dispatched": False, "cached_import_sha256": digest(contract.predecessor["cached_import"]),
                "imported_from_response_id": old_reply["response_id"]}
    reply = replace(ProviderReply(**old_reply), text=dumps(plan), transport_evidence=evidence)
    with scheduler._lock(old / "coordinator.lock", busy="the original 887 coordinator must exit before import"):
        scheduler._assert_no_old_workers(old, (SEED,))
        active, paused_waiter = _verify_old_retired(old, seed_runner.seed_directory(old, SEED))
        scope_record = contract.predecessor["cached_import"]
        if paused_waiter != scope_record.get("paused_waiter"):
            raise PermissionError("the old worker state differs from the approved import receipt")
        args = (contract.transport, contract.profile["wire_representation"])
        _, _, first = _old_receipt(old, active, 0, *args)
        _, _, second = _old_receipt(old, active, 1, *args, paused_waiter=paused_waiter is not None)
        if [first, second] != scope_record["prior"]:
            raise PermissionError("the old bridge receipts differ from the approved import")
        with scheduler._lock(run / "coordinator.lock", busy="cached 887 continuation is already active"):
            if scheduler._pause_requested(run, SEED) and not resume:
                raise PermissionError("cached continuation is paused")
            once(run / "formal_contract.json", contract.record())
            once(run / "authorization.json", approval)
            once(run / "provider_model_lock.json", contract.predecessor["model_identity"])
            once(run / "cached_response_import.json", contract.predecessor["cached_import"])
            if resume:
                (run / "pause.request").unlink(missing_ok=True)
                (seed_runner.seed_directory(run, SEED) / "pause.request").unlink(missing_ok=True)
            scheduler._atomic_status(run / "coordinator_generation.json", {"id": uuid4().hex, "pid": os.getpid()})
            call, options = scheduler._provider_wiring(run, contract, approval, None, SEED)
            bridge = options["managed_attempt"]
            past = seed_runner.seed_directory(run, SEED) / "round_07/turn_00/attempt_00.terminal.json"
            if past.exists():
                returned = read(past)
                if ((returned.get("status"), returned.get("request_id"), returned.get("answer_sha256"))
                        != ("completed", expected["request_id"], digest(reply.text))
                        or any(returned.get(key) != value for key, value in evidence.items())):
                    raise PermissionError("the saved local import differs from the approved cached reply")

            def import_or_bridge(candidate, config, started):
                number = candidate["binding"]["round"]
                if number == ROUND and (candidate, config, started["attempt"]) == (expected, contract.transport, 0):
                    return reply
                if candidate["binding"]["seed"] != SEED or not ROUND < number < contract.rounds:
                    raise TransportStop("the cached decision has no provider transport")
                return bridge(candidate, config, started)

            status = {"seed": SEED, "pid": os.getpid(), "gpu": contract.gpu_slots[0]}
            path = run / f"workers/seed_{SEED}.json"
            scheduler._atomic_status(path, {**status, "status": "initializing"})
            try:
                progress = seed_runner.run_seed(
                    root=root, run_directory=run, contract=contract, authorization=approval, seed=SEED,
                    backend=TrainingBackend(root, cache, SEED), call=call, resume_boundary=boundary,
                    max_rounds=ROUND + 1 if import_only else None,
                    boundary_user_message=user, transport_options={"managed_attempt": import_or_bridge},
                    fit_slot=partial(scheduler.gpu_slot, run, SEED, status["gpu"], contract.lease_root),
                )
                scheduler._atomic_status(path, {**status, **asdict(progress)})
                return progress
            except Exception as error:
                phase = "paused" if scheduler._pause_requested(run, SEED) else (
                    "bridge_pending" if isinstance(error, BridgePending) else "failed")
                scheduler._atomic_status(path, {**status, "status": phase, "error_type": type(error).__name__,
                                                 "reason": str(error)})
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run", "resume"))
    parser.add_argument("--old-release", type=Path, required=True)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--import-only", action="store_true", help="stop at the verified L589 fit")
    args = parser.parse_args()
    os.umask(0o077)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    old_release, old, run = (path.resolve() for path in (args.old_release, args.old_run, args.run_dir))
    contract, boundary, cache, archived, old_reply, plan = construct(ROOT, old_release, old, run)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        parser.error("run and resume require the exact new recovery authorization")
    approval = verify_authorization(contract, read(args.authorization))
    print(dumps(asdict(execute(ROOT, run, old, contract, approval, boundary, cache, archived, old_reply, plan,
                              resume=args.action == "resume", import_only=args.import_only))))


if __name__ == "__main__":
    main()
