"""Recover the exactly reviewed 157 R3 response with no new provider request."""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from functools import partial
import os
from pathlib import Path
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.metadata_view import condition_changes
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import parse_response
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.runner import _observed_alias_projection, _validation
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    ProviderReply, TransportStop, make_request, response_output_text, terminal_record,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import (
    BridgePending, read_bridge_json, validate_result, verify_job,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, STUDY, verify_authorization
from . import run_v13_relay_retry as relay


SEED, ROUND = 157, 3
DISPATCHER = "dispatch_v13_157_cached_r3"
OLD = relay.bounded.RunIdentity("-157-relay-retry-06",
    "a135a2a3d10e80f12864be66eb5bb65046ced589f8767969b3f9cddf3f6c006e",
    "dc505def300d7dd7024819d6cb4494e0d042bc6c1fd8ea5971aea90b602e58a5")
RUN_NAME = f"{relay.bounded.PREFIX}-157-cached-r3-01"
BUDGET = {"first_logical_requests": 32, "maximum_logical_requests": 96, "maximum_physical_attempts": 192}
IMPORT_BUDGET = {"maximum_logical_requests": 1, "maximum_physical_attempts": 0}
ATTEMPT = "d788b24870c14eb3ae7968c7172244ee"
REQUEST = "5d046f81deba596ad7b28ad436f61f91d9210d93d8d94a1dd6fcece207837606"
JOB_SHA = "b901812e38b61ef0f9ca9bfe91d53cc7d35ffd1ea16e39a849451cb31df60da8"
RESULT_SHA = "9fa7fdefe0d76a3b8e80709032ca1a6d06bfa4ed925f3d742c60058190c1eca2"
RAW_SHA = "e3abc12a7e91aaa92c14670298b51940c5461f5a303504c8b810b526597fde21"
RESPONSE_ID = "resp_0403535e9e00decd016b08039e0b4887d2994dfd88faabbaea"
PAYLOAD_SHA = "594ff66ced4fe668cdde6bcbdcc07e9f8cf49933a45accc89ef81a4c97e42107"
FINAL_SHA = "2e2c5140e8c95dc871bccd5f71a39aeb4a95c32e81db0843dd691a007a413233"
PARSED_SHA = "ce00808d4151b1ca3555296c90c15c67c994394a53ff74ec26e96d3c75f1343a"
PLAN_SHA = "eaecf42e4679501be50fb4f938224816972317961e58781d05964f0f77c8fc4d"
PATHS_SHA = "b505c2ba7de680aa9f77ca5945c5dc720a9e29c8906ef7118ef1fedf889403fc"
ORDER_SHA = "6b581d414db7d5559a28af69f775b5e0d13f79812a609642c27c76f1644f4acd"
TERMINAL_SHA = "39edae1cb53b26776fe529166753348de3a46b374a2f52db2915ab0aef6f8d95"
DECISION_SHA = "a49dafb6821b789ed51373e9df957ffded1345f769984e3f178881945f9c7976"
SEED_RUNNER_FILE = (STUDY / "a7_gpu/seed_runner.py").as_posix()
SEED_RUNNER_SHA = "150c7f1f0dce45351b13d5789f99bfc1b41c219251b4c9c6398482a5fb47e8ac"
ROLE_PROJECTION = {"path": "/choices/18/references/0/reference_kind", "from": "single_factor_contrast",
                   "to": "observed", "observed_id": "O-0959", "candidate_id": "C-2926", "axis": "PE/EA"}
BOUNDARY_FILES = {
    "boundary.json": "ea7769698750fb47f0a21fec6abefe5be473545d9658f2dee258e6063a03d0a5",
    "round_03/source.json": "7025deb16c4db0aed7596a70788c12c49a34d182bdfed789326f6c0cdb84e1bc",
    "round_03/boundary_memory.json": "cd61725b8ffdd533c4d185ee3d54693d7237eb55d6add0ac3b83cf2e22a57e37",
}
LANE = relay.LANES[SEED]


def _normalized(raw, stage=None):
    if digest(raw) != RAW_SHA or raw.get("status") != "completed" or raw["id"] != RESPONSE_ID:
        raise PermissionError("the cached result must be the exact complete provider object")
    text = response_output_text(raw["output"])
    plan = parse_response(text)
    if digest(text) != FINAL_SHA or digest(plan) != PARSED_SHA:
        raise PermissionError("the original final provider text differs from the exact reviewed result")
    normalized = deepcopy(plan)
    choice = normalized["choices"][18]
    reference = choice["references"][0]
    if ((reference["reference_kind"], reference["relationship"], reference["reference_id"], reference["observed_id"],
         reference["candidate_id"], choice["candidate_id"], reference["changed_axis"])
            != ("single_factor_contrast", "single_factor_contrast", "O-0959", "O-0959", "C-2926", "C-2926", "PE/EA")):
        raise PermissionError("the scoped role field does not repeat its unchanged relationship and exact observed mirror")
    if stage is not None:
        observed, candidates = stage.context.observed, stage.context.candidates
        selected, compared = candidates["C-2926"], observed["O-0959"]
        if ("O-0959" in candidates or "C-2926" in observed
                or (selected["molecule_id"], compared["molecule_id"], choice["molecule_id"]) != ("M-0130",) * 3
                or condition_changes(selected["conditions"], compared["conditions"]) != ("PE/EA",)):
            raise PermissionError("the exact recorded comparison cannot establish the observed role")
    reference["reference_kind"] = "observed"
    projection = _observed_alias_projection(normalized)
    if projection is None:
        raise PermissionError("the remaining aliases must be exact observed and enclosing candidate mirrors")
    plan_out, paths = projection
    if (len(paths) != 79 or digest(paths) != PATHS_SHA or digest(plan_out) != PLAN_SHA
            or digest([c["candidate_id"] for c in plan_out["choices"]]) != ORDER_SHA):
        raise PermissionError("the reviewed normalization or ordered candidate list changed")
    return plan, plan_out, paths


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
        raise PermissionError("the exact archived L429 state and memory changed")
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
                 "archived_decision_file_sha256": DECISION_SHA, "role_field_projection": ROLE_PROJECTION,
                 "archive_boundary_sha256": BOUNDARY_FILES, "confirmed_boundary_prefix": prefix, "provider_post": False}}
    descriptor = {"round": number, "state_sha256": digest(asdict(state)),
                  "memory_sha256": digest(memory), "user_message_sha256": digest(user)}
    profile = {**previous.profile, "resume_boundaries": {str(SEED): descriptor}, "replay_eligible_from_round": ROUND + 1}
    sources = {name: file_digest(root / name) for name in previous.source_sha256}
    if (sources.get(SEED_RUNNER_FILE) != SEED_RUNNER_SHA
            or {name for name, sha in sources.items() if sha != previous.source_sha256[name]} - {SEED_RUNNER_FILE}):
        raise PermissionError("the import release must keep the original science and exact audited cached runner")
    for module in ("run_v13_157_cached_r3", DISPATCHER):
        path = (STUDY / "scripts" / f"{module}.py").as_posix()
        sources[path] = file_digest(root / path)
    contract = replace(previous, profile=profile, source_sha256=sources, run_directory=str(run), predecessor=proof)
    return contract, boundary, cache, request, result["outcome"]["reply"], plan


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
    reply = replace(ProviderReply(**old_reply), text=dumps(plan), transport_evidence=evidence)
    with scheduler._lock(old / "coordinator.lock", busy="the original 157 coordinator must exit before import"):
        scheduler._assert_no_old_workers(old, (SEED,))
        _, active = relay._retired(old, LANE, require_pause=True)
        prior, _ = relay.bounded._archived(old, Path(contract.predecessor["previous_release_directory"]), OLD)
        _old_receipt(old, active, prior, read(old / "authorization.json"))
        with scheduler._lock(run / "coordinator.lock", busy="the cached 157 continuation is already active"):
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
            past = seed_runner.seed_directory(run, SEED) / "round_03/turn_00/attempt_00.terminal.json"
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
    parser.add_argument("--import-only", action="store_true", help="stop after the cached fit reaches L461")
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
        parser.error("run and resume require the exact new zero-POST import authorization")
    approval = verify_authorization(contract, read(args.authorization))
    result = execute(ROOT, run, old, contract, approval, boundary, cache, archived, old_reply, plan,
                     resume=args.action == "resume", import_only=args.import_only)
    print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
