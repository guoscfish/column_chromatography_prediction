"""Run one paid reference repair for the rejected seed 887 decision at L461."""
import argparse
from dataclasses import replace
from functools import partial
import os
from pathlib import Path
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import parse_response
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.runner import _validation
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    TransportStop, make_request, terminal_record,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import BridgePending, verify_result
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAT_SUFFIX
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    OFFLINE, ROOT, STUDY, verify_authorization,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.targeted_reference_repair import (
    normalize_original, project_reply, repair_guide,
)

from . import run_prompt_direct as direct

RUN_NAME = "a7-gpt62-astra-xhigh-ssh-v10-targeted-repair-887"
SEED = 887
ROUND = 4
CHOICE_INDEX = 15
REFERENCE_INDEX = 0
CANDIDATE = "C-0249"
OBSERVED = "O-2702"


def _original_reply(v9_run, v9_release, contract, approval, boundary, v5_request):
    directory = seed_runner.seed_directory(v9_run, SEED)
    active = seed_runner.round_directory(directory, ROUND)
    following = seed_runner.round_directory(directory, ROUND + 1)
    if (any((active / name).exists() for name in ("batch_freeze.json", "selection.json", "memory.json"))
            or any((following / name).exists() for name in ("transition.json", "authorized_feedback.json",
                                                          "fit_authorization.json"))):
        raise ValueError("the rejected decision already has scientific advancement")
    turn = active / "turn_00"
    request = read(turn / "request.json")
    if request != direct.verify_fresh_input(v9_release, contract, approval, boundary, v5_request):
        raise ValueError("the rejected request differs from the one approved for V9")
    scope = seed_runner._scope(contract, approval, SEED, False)
    grant = seed_runner._selection_grant(scope, request["request_id"])
    started = read(turn / "attempt_00.started.json")
    terminal = terminal_record(turn, 0)
    identity = started["attempt_id"]
    if (read(turn / "formal_request_authorization.json") != grant
            or read(active / "authorization.json") != grant
            or terminal["status"] != "completed" or terminal.get("provider_status") != "completed"
            or (started["request_id"], started["attempt"]) != (request["request_id"], 0)
            or any(terminal.get(key) != value for key, value in started.items())
            or terminal["answer_sha256"] != digest(terminal["text"])):
        raise ValueError("the rejected decision lacks its exact completed request and authorization")
    job = read(v9_run / "bridge/outbox" / f"{identity}.json")
    result = read(v9_run / "bridge/results" / f"{identity}.json")
    if (job["request"] != request or job["started"] != started
            or terminal.get("bridge_result_sha256") != result["result_sha256"]
            or terminal.get("bridge_job_sha256") != job["job_sha256"]):
        raise ValueError("the rejected decision and Mac result do not share request custody")
    reply = verify_result(result, job)
    if (reply.text != terminal["text"] or reply.response_id != terminal["response_id"]
            or reply.served_model != contract.transport.served_model
            or result["raw_response_sha256"] is None):
        raise ValueError("the completed archival reply differs from its terminal evidence")
    state, memory, _ = boundary
    stage = seed_runner._stage(v9_release, state, memory, scope, contract)
    parsed = parse_response(reply.text)
    expected_decision = {"state": "rejected", "diagnosis": _validation(stage, parsed).as_dict()}
    if read(turn / "decision.json") != expected_decision:
        raise ValueError("the prior decision does not reproduce the archived rejection")
    return request, started, result, stage, parsed, reply.text


def _verified_plan(stage, original, text):
    canonical, removed_count = normalize_original(text)
    ids = [choice["candidate_id"] for choice in canonical["choices"]]
    if (len(ids) != 32 or len(set(ids)) != 32
            or ids != [choice["candidate_id"] for choice in original["choices"]]):
        raise ValueError("normalization changed the original ordered 32 candidates")
    target = canonical["choices"][CHOICE_INDEX]
    if (target["candidate_id"], target["references"][REFERENCE_INDEX]["reference_id"]) != (CANDIDATE, OBSERVED):
        raise PermissionError("the repair target differs from the approved candidate and reference")
    validation = _validation(stage, canonical)
    errors = [(item["path"], item["code"]) for item in validation.errors]
    if (validation.valid_choice_count != 31
            or errors != [("/choices/15/references/0/relationship", "multifactor_mismatch")]
            or validation.errors[0].get("actual_changed_axes") != ["flow"]
            or str(stage.context.observed[OBSERVED]["conditions"]["flow"]) != "5"
            or str(stage.context.candidates[CANDIDATE]["conditions"]["flow"]) != "10"):
        raise PermissionError("normalization did not reduce the old rejection to the authorized reference edit")
    return canonical, ids, removed_count


def construct(root, old_release, old_run, first_release, first_run, parent_release, parent_run, v9_release, v9_run, run):
    """Bind a new one-shot repair to the immutable V9 rejection and L461 state."""
    if run.name != RUN_NAME or run.parent != old_run.parent or run == v9_run:
        raise ValueError("the targeted repair needs its own registered run directory")
    old, boundary, v5_request = direct.construct(v9_release, old_release, old_run, first_release, first_run,
                                                   parent_release, parent_run, v9_run, SEED)
    if (read(v9_run / "formal_contract.json") != old.record() or old.seed_ids != (SEED,)
            or old.target_labels != 493 or old.transport.transport_attempts != 1 or old.transport.logical_replies != 1
            or old.transport.timeout_seconds != 3600 or boundary[2] != ROUND):
        raise PermissionError("the parent is not the immutable one-shot seed887 V9 run")
    approval = verify_authorization(old, read(v9_run / "authorization.json"))
    original, started, result, stage, parsed, text = _original_reply(
        v9_run, v9_release, old, approval, boundary, v5_request)
    plan, ids, removed = _verified_plan(stage, parsed, text)
    metadata = {
        "original_request_id": original["request_id"], "original_attempt_id": started["attempt_id"],
        "original_raw_sha256": result["raw_response_sha256"], "original_answer_sha256": digest(text),
        "original_result_sha256": result["result_sha256"], "canonical_plan_sha256": digest(plan),
        "ordered_candidate_ids_sha256": digest(ids), "repair_index": CHOICE_INDEX,
        "reference_index": REFERENCE_INDEX, "candidate_id": CANDIDATE, "reference_id": OBSERVED,
        "changed_axis": "flow", "removed_nonsemantic_fields": removed,
    }

    prompt = seed_runner.PROMPT_PATH
    if (root / prompt).read_text().strip() != (v9_release / prompt).read_text().strip() + "\n\n" + repair_guide():
        raise PermissionError("the prompt may only append the authorized reference repair guide")
    editable = {prompt.as_posix(), (STUDY / "a7_gpu/seed_runner.py").as_posix(),
                (OFFLINE / "pure_llm_v3/runner.py").as_posix()}
    sources = {name: file_digest(root / name) for name in old.source_sha256}
    if any(sources[name] != expected for name, expected in old.source_sha256.items() if name not in editable):
        raise PermissionError("the repair release changed code outside the projection and prompt")
    for relative in ("a7_gpu/targeted_reference_repair.py", "scripts/run_prompt_targeted_repair.py",
                     "scripts/dispatch_prompt_targeted_repair.py"):
        name = (STUDY / relative).as_posix()
        sources[name] = file_digest(root / name)
    lock = read(v9_run / "provider_model_lock.json")
    if digest(lock) != old.predecessor["provider_model_lock_sha256"] or lock != result["model_identity"]:
        raise ValueError("the V9 response provider identity differs from the sealed direct protocol")
    profile = {**old.profile, "fixed_reference_repair": metadata}
    evidence = {"contract_sha256": old.digest, "authorization_sha256": digest(approval), "run_directory": str(v9_run),
                "checkpoint_sha256": boundary[0].checkpoint, "start_label_budget": 461,
                "original_user_message_sha256": digest(original["payload"]["input"][1]),
                "provider_model_lock_sha256": digest(lock), "fixed_reference_repair_sha256": digest(metadata),
                "max_new_physical_attempts": 1, "uses_previous_incomplete_round_responses": False,
                "mac_connection_policy": old.predecessor["mac_connection_policy"]}
    contract = replace(old, profile=profile, source_sha256=sources, run_directory=str(run), predecessor=evidence)
    return contract, boundary, original, plan


def verify_fresh_input(root, contract, approval, boundary, original):
    state, memory, start = boundary
    scope = seed_runner._scope(contract, approval, SEED, False)
    stage = seed_runner._stage(root, state, memory, scope, contract)
    old_system, old_user = original["payload"]["input"]
    stage = seed_runner.pin_boundary_user_message(
        stage, old_user, contract.profile["resume_boundaries"][str(SEED)]["user_message_sha256"])
    request = make_request(stage.binding, 0, stage.messages, contract.transport)
    expected_system = {**old_system, "content": old_system["content"].removesuffix(FORMAT_SUFFIX)
                       + "\n\n" + repair_guide() + FORMAT_SUFFIX}
    if (start != ROUND or not old_system["content"].endswith(FORMAT_SUFFIX)
            or request["payload"] != {**original["payload"], "input": [expected_system, old_user]}
            or request["request_id"] == original["request_id"]
            or any(stage.binding[name] != original["binding"][name] for name in (
                "checkpoint_sha256", "packet_sha256", "wire_packet_sha256", "mapping_sha256"))):
        raise PermissionError("the repair request may only add the exact system guide to the original data")
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run", "resume"))
    for name in ("old-release", "old-run", "first-release", "first-run", "parent-release", "parent-run",
                 "v9-release", "v9-run", "run-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    old_run, v9_run, run = args.old_run.resolve(), args.v9_run.resolve(), args.run_dir.resolve()
    contract, boundary, original, plan = construct(
        ROOT, args.old_release.resolve(), old_run, args.first_release.resolve(), args.first_run.resolve(),
        args.parent_release.resolve(), args.parent_run.resolve(), args.v9_release.resolve(), v9_run, run)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        raise PermissionError("one paid reference repair requires the exact new authorization")
    approval = verify_authorization(contract, read(args.authorization))
    request = verify_fresh_input(ROOT, contract, approval, boundary, original)
    with scheduler._lock(run / "coordinator.lock", busy="targeted repair coordinator already running"):
        scheduler._assert_no_old_workers(run, contract.seed_ids)
        if scheduler._pause_requested(run) and args.action != "resume":
            raise PermissionError("the targeted repair is paused; use resume")
        once(run / "formal_contract.json", contract.record())
        once(run / "authorization.json", approval)
        once(run / "fixed_reference_repair.json", contract.profile["fixed_reference_repair"])
        once(run / "canonical_repair_plan.json", plan)
        once(run / "fresh_request_scope.json", {"request_id": request["request_id"], "seed": SEED,
             "round": ROUND, "logical_index": 0, "attempt": 0, "prior_complete_response_in_payload": False,
             "payload_sha256": request["payload_sha256"]})
        lock = read(v9_run / "provider_model_lock.json")
        if digest(lock) != contract.predecessor["provider_model_lock_sha256"]:
            raise ValueError("the sealed provider model identity changed during verification")
        once(run / "provider_model_lock.json", lock)
        scheduler._atomic_status(run / "coordinator_generation.json", {"id": uuid4().hex, "pid": os.getpid()})
        if args.action == "resume":
            (run / "pause.request").unlink(missing_ok=True)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        record = run / "workers" / f"seed_{SEED}.json"
        with scheduler._lock(run / "workers" / f"seed_{SEED}.lock", busy="targeted repair worker already running"):
            scheduler._atomic_status(record, {"status": "initializing", "seed": SEED, "pid": os.getpid(), "gpu": 0})
            try:
                backend = TrainingBackend(ROOT, old_run / "public_graphs" / f"seed_{SEED}", SEED)
                call, options = scheduler._provider_wiring(run, contract, approval, None, SEED)
                managed = options["managed_attempt"]

                def one_request(candidate, config, started):
                    if candidate != request or started["attempt"] != 0:
                        raise TransportStop("the repair authorizes one exact physical request")
                    return managed(candidate, config, started)

                output = seed_runner.run_seed(root=ROOT, run_directory=run, contract=contract, authorization=approval,
                    seed=SEED, backend=backend, call=call, resume_boundary=boundary,
                    boundary_user_message=original["payload"]["input"][1], max_rounds=ROUND + 1,
                    fit_slot=partial(scheduler.gpu_slot, run, SEED, 0, contract.lease_root),
                    transport_options={"managed_attempt": one_request},
                    response_projection=partial(project_reply, plan=plan))
                scheduler._atomic_status(record, {**vars(output), "pid": os.getpid(), "gpu": 0})
            except Exception as error:
                status = "paused" if scheduler._pause_requested(run, SEED) else (
                    "bridge_pending" if isinstance(error, BridgePending) else "failed")
                scheduler._atomic_status(record, {"status": status, "seed": SEED, "pid": os.getpid(), "gpu": 0,
                                                  "error_type": type(error).__name__, "reason": str(error)})
                raise
        print(dumps(vars(output)))


if __name__ == "__main__":
    main()
