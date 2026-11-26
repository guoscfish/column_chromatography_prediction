"""Recover the exactly reviewed 6101 R2 response with no new provider request."""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from functools import partial
import os
from pathlib import Path
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import parse_response
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.runner import _validation
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    ProviderReply, TransportStop, make_request, response_output_text, terminal_record,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import (
    BridgePending, read_bridge_json, validate_result, verify_job,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, STUDY, verify_authorization
from . import run_v13_relay_retry as relay


SEED, ROUND = 6101, 2
DISPATCHER = "dispatch_v13_6101_cached_r2"
OLD = relay.bounded.RunIdentity("-6101-relay-retry-06",
    "3380abf0b3fbde69a4d4c7a9d8e64dc815cc64c258736088baa3041fff209023",
    "ae353451f7f46dd0b36a4515e8e469defe36348f50e9be8752999d7999a228c1")
RUN_NAME = f"{relay.bounded.PREFIX}-6101-cached-r2-01"
BUDGET = {"first_logical_requests": 33, "maximum_logical_requests": 99, "maximum_physical_attempts": 198}
IMPORT_BUDGET = {"maximum_logical_requests": 1, "maximum_physical_attempts": 0}
ATTEMPT = "18c94b2993e1444686651504f7aeb992"
REQUEST = "c4b95593b381246aaa76046f717daa92912cd3afa7d17754cb55e333014af79f"
JOB_SHA = "f02620a8b8bc34c516dc4a9b2d31b01e065ea2dc1f969c6d2f7c3be12de086cd"
RESULT_SHA = "9497827cae957a42049346b25bb15650eb3213bc7db11a95d2e6af9fd83c3e60"
RAW_SHA = "d7f5586ac48d83e1af98faa24ad17317aa200c5148b8037451493f09bea82ea4"
RESPONSE_ID = "resp_04415e2e4258a85c016b08202301c087d283892240b2871ffd"
PAYLOAD_SHA = "f3f3dac061e9baca67095771305e0ea47827c20d35e7d1018c89059172533b1a"
FINAL_SHA = "fd09b1b73ab1fee937b4ffe32b14ee09b1f394a4d46b5ae760425d5645065646"
PARSED_SHA = "a7446c33a0bdd20f52f7dca3b9c4d305ad0712ca4d1118d3e850a20c716a6b8e"
PLAN_SHA = "c13db811a7e733ebf186dbeb03959c7fe917218c0cd8f244812824669d593789"
PATHS_SHA = "ccf160994bcb7d187e414728416a56cc05391edebc2264a5e818409b4be6cb69"
ORDER_SHA = "7d6f34ae97691850af98ebbc46f14010aec547d24f06f020d916f73f0423ce06"
TERMINAL_SHA = "b49f95e69d5517ddd433423ee3de039260d28aa40b9aed8cd368a8e69617ea97"
DECISION_SHA = "a03701ebfb59012d1e04898b472d414a94bde8be0debf12754b47c8eeb47c778"
SEED_RUNNER_FILE = (STUDY / "a7_gpu/seed_runner.py").as_posix()
SEED_RUNNER_SHA = "d872883ca1d987d36b9610ab59f1a2957dc25a697750879f41839783dda147ff"
BOUNDARY_FILES = {
    "boundary.json": "5384af42e8bc07d61c093b77530d148ace7307547415af0e079b86cae8fda67f",
    "round_02/source.json": "c33ecb89a166f5f2d112f2bb434fab6529d4bf108fe0d4f84b28d3b7e39648b0",
    "round_02/boundary_memory.json": "989f5e746a5398725bd4bcb37ee199fab8b1f13516e0e58c6d47b66365b07cec",
}
LANE = replace(relay.LANES[SEED], round=ROUND)


def _project_mirrors(plan, stage):
    projected = deepcopy(plan)
    paths = []
    fields = {"observed": "observed_id", "candidate": "candidate_id"}
    tables = None if stage is None else {"observed": stage.context.observed, "candidate": stage.context.candidates}
    for choice_index, choice in enumerate(projected["choices"]):
        for reference_index, reference in enumerate(choice["references"]):
            aliases = reference.keys() & fields.values()
            if not aliases:
                continue
            kind = reference["reference_kind"]
            field = fields.get(kind)
            canonical = reference["reference_id"]
            if (aliases != {field} or reference[field] != canonical
                    or tables is not None and canonical not in tables[kind]):
                raise PermissionError("a nested field does not mirror its typed, authoritative reference ID")
            del reference[field]
            paths.append(f"/choices/{choice_index}/references/{reference_index}/{field}")
    return projected, paths


def _normalized(raw, stage=None):
    if digest(raw) != RAW_SHA or raw.get("status") != "completed" or raw["id"] != RESPONSE_ID:
        raise PermissionError("the cached result must be the exact complete provider object")
    text = response_output_text(raw["output"])
    plan = parse_response(text)
    if digest(text) != FINAL_SHA or digest(plan) != PARSED_SHA:
        raise PermissionError("the original final provider text differs from the exact reviewed result")
    projected, paths = _project_mirrors(plan, stage)
    if (len(paths) != 74 or digest(paths) != PATHS_SHA or digest(projected) != PLAN_SHA
            or digest([choice["candidate_id"] for choice in projected["choices"]]) != ORDER_SHA):
        raise PermissionError("the reviewed mirrors, normalized plan, or ordered candidates changed")
    return plan, projected, paths


def _old_receipt(run, active, previous, approval):
    turn = relay.bounded._turn(active, 1)
    request = relay.bounded._request(active, turn, previous, approval, LANE)
    started = read(turn / "attempt_00.started.json")
    terminal = terminal_record(turn, 0)
    name = f"{ATTEMPT}.json"
    job = read_bridge_json(run / "bridge/outbox" / name)
    result = validate_result(read_bridge_json(run / "bridge/results" / name), job)
    reply = result["outcome"].get("reply", {})
    if (verify_job(job, contract_sha256=OLD.contract, authorization_sha256=OLD.approval,
                   wire_representation=previous.profile["wire_representation"]) != (request, previous.transport, started)
            or (request["request_id"], request["payload_sha256"], job["job_sha256"], result["result_sha256"],
                result["raw_response_sha256"], job["attempt_id"]) != (REQUEST, PAYLOAD_SHA, JOB_SHA, RESULT_SHA, RAW_SHA, ATTEMPT)
            or result["provider_dispatched"] is not True or terminal["provider_dispatched"] is not True
            or result["outcome"]["status"] != "completed" or terminal["status"] != "completed"
            or any(terminal.get(k) != v for k, v in started.items())
            or terminal["bridge_job_sha256"] != JOB_SHA or terminal["bridge_result_sha256"] != RESULT_SHA
            or file_digest(turn / "attempt_00.terminal.json") != TERMINAL_SHA
            or file_digest(turn / "decision.json") != DECISION_SHA or terminal["answer_sha256"] != FINAL_SHA
            or any(terminal[k] != reply[k] for k in ("text", "response_id", "served_model", "provider_request_id", "usage"))
            or (read(run / "provider_model_lock.json"), result["model_identity"])
            != (previous.predecessor["model_identity"],) * 2):
        raise PermissionError("the original posted response, terminal or prior rejected decision changed")
    return request, result


def construct(root, old_release, old, run):
    if run.name != RUN_NAME or run.parent != old.parent or root == old_release:
        raise PermissionError("the cached continuation needs its own sibling run and immutable release")
    previous, approval = relay.bounded._archived(old, old_release, OLD)
    directory, active = relay._retired(old, LANE, require_pause=False)
    if any(file_digest(directory / name) != sha for name, sha in BOUNDARY_FILES.items()):
        raise PermissionError("the exact archived L397 state and memory changed")
    request, result = _old_receipt(old, active, previous, approval)
    previous_proof = previous.predecessor
    cache = Path(previous_proof["cache_directory"]).resolve(strict=True)
    if file_digest(cache / "resource_receipt.json") != previous_proof["public_resource_receipt_sha256"]:
        raise PermissionError("the archived training cache differs")
    boundary, prefix = relay._boundary(old_release, directory, active, previous, approval, LANE, cache)
    state, memory, number = boundary
    user = request["payload"]["input"][1]
    scope = seed_runner._scope(previous, approval, SEED, False)
    stage = seed_runner.pin_boundary_user_message(seed_runner._stage(old_release, state, memory, scope, previous),
                                                  user, digest(user))
    original, plan, paths = _normalized(result["raw_response"], stage)
    before = _validation(stage, original)
    checked = _validation(stage, plan)
    archived_binding = {**stage.binding, relay.ALIAS: request["binding"][relay.ALIAS]}
    if (make_request(archived_binding, 0, stage.messages, previous.transport) != request
            or read(active / "turn_00/decision.json") != {"state": "rejected", "diagnosis": before.as_dict()}
            or before.valid or not checked.valid or checked.valid_choice_count != 32):
        raise PermissionError("the original diagnostics or fully revalidated normalized plan cannot be reproduced")

    proof = {**previous_proof, "dispatcher_module": DISPATCHER, "previous_run_directory": str(old),
             "previous_release_directory": str(old_release), "previous_contract_sha256": previous.digest,
             "previous_authorization_sha256": digest(approval), "previous_request_id": REQUEST,
             "prompt_sha256": stage.binding["prompt_sha256"], "recovery_budget": IMPORT_BUDGET, "execution_budget": BUDGET,
             "cached_import": {"attempt_id": ATTEMPT, "job_sha256": JOB_SHA, "result_sha256": RESULT_SHA,
                 "raw_response_sha256": RAW_SHA, "final_text_sha256": FINAL_SHA, "normalized_plan_sha256": PLAN_SHA,
                 "removed_paths": paths, "ordered_candidate_ids_sha256": ORDER_SHA, "archived_terminal_file_sha256": TERMINAL_SHA,
                 "archived_decision_file_sha256": DECISION_SHA,
                 "archive_boundary_sha256": BOUNDARY_FILES, "confirmed_boundary_prefix": prefix, "provider_post": False}}
    descriptor = {"round": number, "state_sha256": digest(asdict(state)),
                  "memory_sha256": digest(memory), "user_message_sha256": digest(user)}
    profile = {**previous.profile, "resume_boundaries": {str(SEED): descriptor}, "replay_eligible_from_round": ROUND + 1}
    sources = {name: file_digest(root / name) for name in previous.source_sha256}
    if (sources.get(SEED_RUNNER_FILE) != SEED_RUNNER_SHA
            or {name for name, sha in sources.items() if sha != previous.source_sha256[name]} - {SEED_RUNNER_FILE}):
        raise PermissionError("the import release must keep the original science and exact audited cached runner")
    for module in ("run_v13_6101_cached_r2", DISPATCHER):
        path = (STUDY / "scripts" / f"{module}.py").as_posix()
        sources[path] = file_digest(root / path)
    contract = replace(previous, profile=profile, source_sha256=sources, run_directory=str(run), predecessor=proof)
    return contract, boundary, cache, request, result["outcome"]["reply"], plan


def verify_handoff(run, contract, approval, *, root=ROOT):
    """Check the immutable L429 CUDA transition before allowing future rounds."""
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    run = Path(run)
    verify_authorization(contract, approval)
    proof = contract.predecessor
    if (run.name != RUN_NAME or contract.run_directory != str(run) or read(run / "formal_contract.json") != contract.record()
            or read(run / "authorization.json") != approval or proof["dispatcher_module"] != DISPATCHER
            or proof["recovery_budget"] != IMPORT_BUDGET or proof["execution_budget"] != BUDGET):
        raise PermissionError("the cached continuation does not match the exact signed run")
    directory = seed_runner.seed_directory(run, SEED)
    imported, fitted = (seed_runner.round_directory(directory, n) for n in (ROUND, ROUND + 1))
    transition = fitted / "transition.json"
    if not transition.is_file():
        return None
    custody = proof["cached_import"]
    terminal = terminal_record(relay.bounded._turn(imported, 1), 0)
    original_state = read(imported / "source.json")
    boundary = contract.profile["resume_boundaries"][str(SEED)]
    if (custody["provider_post"] is not False or read(run / "cached_response_import.json") != custody
            or terminal.get("status") != "completed" or terminal.get("provider_dispatched") is not False
            or terminal.get("usage") != {} or terminal.get("cached_import_sha256") != digest(custody)
            or terminal.get("imported_from_response_id") != RESPONSE_ID or digest(parse_response(terminal["text"])) != PLAN_SHA
            or (run / "bridge/outbox" / f"{terminal['attempt_id']}.json").exists()
            or (digest(original_state), digest(read(imported / "boundary_memory.json")))
            != (boundary["state_sha256"], boundary["memory_sha256"])):
        raise PermissionError("the zero-POST cached decision or its fitted boundary changed")
    scope = seed_runner._scope(contract, approval, SEED, False)
    cache = Path(proof["cache_directory"])
    if file_digest(cache / "resource_receipt.json") != proof["public_resource_receipt_sha256"]:
        raise PermissionError("the original training cache changed")
    backend = TrainingBackend(root, cache, SEED)
    state, _ = seed_runner._successor(imported, seed_runner.FrozenState(**original_state), fitted, scope,
                                      contract.transport, contract, backend)
    acquired = seed_runner._state_roles(state, backend, scope, ROUND + 1, contract.batch_size)
    labels = len(backend.l0_ids) + len(acquired)
    if (state.round, labels) != (ROUND + 1, 429):
        raise PermissionError("the cached fit did not yield the intended L429 state")
    return {"round": state.round, "active_labels": labels, "checkpoint_sha256": state.checkpoint,
            "transition_file_sha256": file_digest(transition), "provider_post": False}


def execute(root, run, old, contract, approval, boundary, cache, archived, old_reply, plan, *, resume, import_only):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    scope = seed_runner._scope(contract, approval, SEED, False)
    user = archived["payload"]["input"][1]
    stage = seed_runner.pin_boundary_user_message(seed_runner._stage(root, *boundary[:2], scope, contract), user, digest(user))
    expected = make_request(stage.binding, 0, stage.messages, contract.transport)
    if expected["payload"] != archived["payload"] or digest(plan) != PLAN_SHA or not _validation(stage, plan).valid:
        raise PermissionError("the approved cached decision changed before import")
    evidence = {"provider_dispatched": False, "cached_import_sha256": digest(contract.predecessor["cached_import"]),
                "imported_from_response_id": old_reply["response_id"]}
    reply = replace(ProviderReply(**old_reply), text=dumps(plan), usage={}, transport_evidence=evidence)
    with scheduler._lock(old / "coordinator.lock", busy="the original 6101 coordinator must exit before import"):
        scheduler._assert_no_old_workers(old, (SEED,))
        _, active = relay._retired(old, LANE, require_pause=True)
        prior, _ = relay.bounded._archived(old, Path(contract.predecessor["previous_release_directory"]), OLD)
        _old_receipt(old, active, prior, read(old / "authorization.json"))
        with scheduler._lock(run / "coordinator.lock", busy="the cached 6101 continuation is already active"):
            if scheduler._pause_requested(run, SEED) and not resume:
                raise PermissionError("cached continuation is paused; use resume")
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
            bridge.wait_seconds = contract.predecessor["bridge_wait_seconds"]
            past = seed_runner.round_directory(seed_runner.seed_directory(run, SEED), ROUND) / "turn_00/attempt_00.terminal.json"
            if not import_only and verify_handoff(run, contract, approval, root=root) is None:
                raise PermissionError("the cached L429 fit must be verified before future provider transport")
            if past.exists():
                saved = read(past)
                if ((saved.get("status"), saved.get("request_id"), saved.get("answer_sha256"))
                        != ("completed", expected["request_id"], digest(reply.text))
                        or any(saved.get(key) != value for key, value in evidence.items())):
                    raise PermissionError("the saved local import differs from the approved cached response")

            def import_or_bridge(candidate, config, started):
                binding = candidate["binding"]
                if binding["seed"] != SEED:
                    raise TransportStop("a cached continuation cannot submit for a different seed")
                if binding["round"] == ROUND and (candidate, config, started["attempt"]) == (expected, contract.transport, 0):
                    return reply
                if import_only or not ROUND < binding["round"] < contract.rounds:
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
            except Exception as exc:
                phase = "paused" if scheduler._pause_requested(run, SEED) else (
                    "bridge_pending" if isinstance(exc, BridgePending) else "failed")
                scheduler._atomic_status(path, {**status, "status": phase, "error_type": type(exc).__name__, "reason": str(exc)})
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run", "resume"))
    parser.add_argument("--old-release", type=Path, required=True)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--continue-after-fit", action="store_true", help="resume later rounds after verifying the cached L429 fit")
    args = parser.parse_args()
    if args.continue_after_fit and args.action != "resume":
        parser.error("future rounds require resume after the zero-POST L429 fit")
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
        parser.error("run and resume require the exact new zero-POST import authorization")
    approval = verify_authorization(contract, read(args.authorization))
    result = execute(ROOT, run, old, contract, approval, boundary, cache, archived, old_reply, plan,
                     resume=args.action == "resume", import_only=not args.continue_after_fit)
    print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
