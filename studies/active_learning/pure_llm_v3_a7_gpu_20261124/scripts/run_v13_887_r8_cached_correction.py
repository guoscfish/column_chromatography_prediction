"""Import one validated archived 887 correction from its exact raw reply, with no model submission."""

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
    ProviderReply, TransportStop, make_request, terminal_record,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import validate_result, verify_job
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, STUDY, verify_authorization
from . import run_v13_887_r8_targeted_repair as repair
from .run_v13_lane_continuation import _extension_approval


SEED, ROUND = 887, 8
RUN_NAME = f"{repair.relay.bounded.PREFIX}-887-r8-cached-correction-01"
MODULE = "run_v13_887_r8_cached_correction"
CORRECTION = repair.relay.bounded.RunIdentity(
    "-887-r8-targeted-repair-01", "a176ab59cc1c069ac77f67dc15516844bae3d3f6fa13fde6e7a77db73afa2247",
    "666fca40036ab340100c819bb0ccd58fec9239aab6b2ee792f6ffcf8e2b98c92",
)
ATTEMPT = "3a024e9affe84055a845f45c0eb62fda"
REQUEST = "4ffa1afd898b3653ef7b6040503686181db14ae8529ba5a7f7087edeae00211e"
PAYLOAD = "4b16f6c293651f488fcd23cab41edfa0273135fe29627b87a7eaf0d5435bb3c4"
JOB = "55d6c1476cf450be2cecd3551960978341bb60dd0105b06bfb775bb8d0dfae68"
RESULT = "ba851f873f1c483f7570eb2ccd81fd7926543249bdb906be738edd1beb61f01d"
RAW = "dbff342eb8cf08d49f07d50539b77b364f90485dcafb37042df12a02baa22938"
TEXT = "bbba2f8023d8b0cc98cfa8f44636a0dc5b0edc2dae085a5fbac5f753c35d8703"
PATCH = "f220ca3ce48139444b52f267b5fd433344df8cad76a3a22d53f77a9103aa7948"
PROJECTED = "401c78f2d7317fa6ae7dc0473382ecf7880753a2dd805fc73936188516516860"
RESPONSE = "resp_0f7f8409a0f2e1d9016b080fe43d5087d0b431e6b5b08f2efd"
PAUSE = "5e0eaeee65c3d37558af6ffedf292e474bcf920e82bf96dc78e09144a948064a"
DECISION = "4c81960657aef06ac303f8402f751712c9f9672d17f3028411ef617a960e210a"
TERMINAL = "74fd6c4a2949d92c11873c4fbfb2748742e1455b1ef6aa6824f470f13a897c74"
BUDGET = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 0}
SOURCES = tuple(STUDY / "scripts" / f"{module}.py" for module in (MODULE, "run_v13_lane_continuation"))


def project(patch, *, original):
    if digest(patch) != PATCH or digest(original) != repair.PLAN:
        raise PermissionError("only this exact archived model patch may be imported")
    reason = patch["reason"]
    negated = "isolates neither effect"
    if (patch["candidate_id"] != repair.CANDIDATE or patch["reference_index"] != repair.REFERENCE_INDEX
            or patch["reference"] != repair.REQUIRED_REFERENCE or reason.count(negated) != 1
            or repair._checked_reason(reason)
            or not repair._checked_reason(reason.replace(negated, "does not separate either effect"))):
        raise PermissionError("the archived correction differs from the single verified negated-phrase exception")
    plan = deepcopy(original)
    choice = plan["choices"][repair.CHOICE_INDEX]
    choice["reason"] = reason
    choice["references"][repair.REFERENCE_INDEX] = deepcopy(patch["reference"])
    if digest(plan) != PROJECTED:
        raise PermissionError("the exact ordered projection changed")
    return plan


def _read_archive(old, active, contract, approval, original):
    turn = repair.relay.bounded._turn(active, 1)
    if (file_digest(active.parent / "pause.request"), file_digest(turn / "decision.json"),
        file_digest(turn / "attempt_00.terminal.json")) != (PAUSE, DECISION, TERMINAL):
        raise PermissionError("the paid correction's fence, rejection or completed terminal changed")
    if read(turn / "decision.json") != {"state": "patch_rejected", "reason": "outside_scoped_reference_repair"}:
        raise PermissionError("the completed correction is not the identified software-only rejection")
    request = repair.relay.bounded._request(active, turn, contract, approval, repair.LANE)
    repair.verify_patch_request(request, original, contract.profile["fixed_reference_repair"], attempt=0)
    started = read(turn / "attempt_00.started.json")
    terminal = terminal_record(turn, 0)
    job = read(old / "bridge/outbox" / f"{ATTEMPT}.json")
    result = validate_result(read(old / "bridge/results" / f"{ATTEMPT}.json"), job)
    bound = verify_job(job, contract_sha256=contract.digest, authorization_sha256=digest(approval),
                       wire_representation=contract.profile["wire_representation"])
    if (bound != (request, contract.transport, started) or started["attempt_id"] != ATTEMPT
            or (request["request_id"], request["payload_sha256"], job["job_sha256"], result["result_sha256"],
                result["raw_response_sha256"]) != (REQUEST, PAYLOAD, JOB, RESULT, RAW)
            or result["outcome"]["status"] != "completed" or result["provider_dispatched"] is not True
            or result["model_identity"] != contract.predecessor["model_identity"]
            or read(old / "provider_model_lock.json") != result["model_identity"]
            or terminal["status"] != "completed" or terminal.get("provider_status") != "completed"
            or terminal.get("bridge_job_sha256") != JOB or terminal.get("bridge_result_sha256") != RESULT
            or any(terminal.get(k) != v for k, v in started.items())):
        raise PermissionError("the cached repair is detached from its exact completed provider reply")
    reply = result["outcome"]["reply"]
    if ((reply["response_id"], digest(reply["text"])) != (RESPONSE, TEXT)
            or (terminal["response_id"], terminal["answer_sha256"], terminal["text"]) != (RESPONSE, TEXT, reply["text"])):
        raise PermissionError("the original fully delivered repair response changed")
    return request, reply


def construct(root, ordinary_root, ordinary_run, paid_root, paid, run):
    if (root in (ordinary_root, paid_root) or run.name != RUN_NAME or paid.name != CORRECTION.name
            or len({ordinary_run.parent, paid.parent, run.parent}) != 1):
        raise PermissionError("the cached correction requires its own signed sibling release and run")
    directory, active = repair.relay._retired(paid, repair.LANE, require_pause=True)
    previous, boundary, cache, original, original_plan = repair.construct(paid_root, ordinary_root, ordinary_run, paid)
    approval = _extension_approval(previous, paid, CORRECTION.contract, CORRECTION.approval)
    sources = {name: file_digest(root / name) for name in previous.source_sha256}
    if sources != previous.source_sha256:
        raise PermissionError("the archival import cannot change any of the approved correction source or Prompt")
    sources.update({path.as_posix(): file_digest(root / path) for path in SOURCES})
    request, reply = _read_archive(paid, active, previous, approval, original)
    scope = seed_runner._scope(previous, approval, SEED, False)
    state, memory, _ = boundary
    user = original["payload"]["input"][1]
    if (read(directory / "boundary.json") != {"formal_run": scope, **previous.profile["resume_boundaries"][str(SEED)]}
            or read(active / "source.json") != asdict(state) or read(active / "boundary_memory.json") != memory):
        raise PermissionError("the completed paid repair's L589 input state or scientific memory changed")
    stage = seed_runner.pin_boundary_user_message(seed_runner._stage(paid_root, state, memory, scope, previous), user, digest(user))
    if any(stage.binding[k] != request["binding"][k] for k in repair.DATA_BINDINGS):
        raise PermissionError("the archived correction's authoritative stage is no longer reproducible")
    plan = project(parse_response(reply["text"]), original=original_plan)
    validation = _validation(stage, plan)
    if not validation.valid or validation.valid_choice_count != 32:
        raise PermissionError("the unchanged correction failed the authoritative A7 validator")
    imported = {"provider_post": False, "attempt_id": ATTEMPT, "request_id": REQUEST, "job_sha256": JOB,
                "result_sha256": RESULT, "raw_response_sha256": RAW, "response_id": RESPONSE, "reply_text_sha256": TEXT,
                "patch_sha256": PATCH, "projected_plan_sha256": PROJECTED, "ordered_candidate_ids_sha256": repair.ORDER,
                "previous_seed_pause_file_sha256": PAUSE, "old_decision_file_sha256": DECISION,
                "old_terminal_file_sha256": TERMINAL}
    proof = {**previous.predecessor, "dispatcher_module": MODULE, "cached_correction": imported,
             "previous_run_directory": str(paid), "previous_release_directory": str(paid_root),
             "previous_contract_sha256": CORRECTION.contract, "previous_authorization_sha256": CORRECTION.approval,
             "previous_request_id": REQUEST, "execution_budget": BUDGET,
             "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 0}}
    contract = replace(previous, predecessor=proof, run_directory=str(run), source_sha256=sources)
    return contract, boundary, cache, original, original_plan, reply


def _no_provider(*args, **kwargs):
    raise TransportStop("a cached correction has no provider transport")


def _require_no_bridge(run):
    if any(path.is_file() for folder in ("outbox", "claims", "results") for path in (run / "bridge" / folder).glob("*.json")):
        raise PermissionError("a zero-POST archival import may not have any bridge attempts")


def execute(root, paid, run, contract, approval, boundary, cache, original, plan, archived, *, resume):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    approval = verify_authorization(contract, approval)
    scope = seed_runner._scope(contract, approval, SEED, False)
    user = original["payload"]["input"][1]
    stage = seed_runner.pin_boundary_user_message(seed_runner._stage(root, *boundary[:2], scope, contract), user, digest(user))
    expected = make_request(stage.binding, 0, stage.messages, contract.transport)
    repair.verify_patch_request(expected, original, contract.profile["fixed_reference_repair"], attempt=0)
    custody = contract.predecessor["cached_correction"]
    if (expected["request_id"] == REQUEST or custody["provider_post"] is not False
            or contract.predecessor["execution_budget"] != BUDGET):
        raise PermissionError("the local decision requires isolated provenance and zero paid submissions")
    evidence = {"provider_dispatched": False, "cached_correction_sha256": digest(custody),
                "imported_from_response_id": RESPONSE}
    reply = replace(ProviderReply(**archived), usage={}, transport_evidence=evidence)

    def local(candidate, config, started):
        _require_no_bridge(run)
        if candidate != expected or config != contract.transport or started["attempt"] != 0:
            raise TransportStop("only the exact archived repair may occupy the local decision slot")
        return reply

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(contract.gpu_slots[0])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with scheduler._lock(paid / "coordinator.lock", busy="the paid repair worker must remain retired"):
        retired, _ = repair.relay._retired(paid, repair.LANE, require_pause=True)
        if file_digest(retired / "pause.request") != PAUSE:
            raise PermissionError("the paid repair fence changed")
        with scheduler._lock(run / "coordinator.lock", busy="the cached GPU coordinator is already active"):
            _require_no_bridge(run)
            scheduler._assert_no_old_workers(run, (SEED,))
            directory = seed_runner.seed_directory(run, SEED)
            if scheduler._pause_requested(run, SEED) and not resume:
                raise PermissionError("use resume for the paused local import")
            once(run / "formal_contract.json", contract.record())
            once(run / "authorization.json", approval)
            once(run / "provider_model_lock.json", contract.predecessor["model_identity"])
            once(run / "cached_correction.json", custody)
            once(run / "canonical_repair_plan.json", plan)
            prior = directory / "round_08/turn_00/attempt_00.terminal.json"
            if prior.is_file():
                terminal = read(prior)
                if (terminal.get("request_id") != expected["request_id"] or terminal.get("answer_sha256") != TEXT
                        or any(terminal.get(k) != v for k, v in evidence.items())):
                    raise PermissionError("the local resume receipt differs from the cached completion")
            if resume:
                (run / "pause.request").unlink(missing_ok=True)
                (directory / "pause.request").unlink(missing_ok=True)
            scheduler._atomic_status(run / "coordinator_generation.json", {"id": uuid4().hex, "pid": os.getpid()})
            gpu = contract.gpu_slots[0]
            status = {"seed": SEED, "pid": os.getpid(), "gpu": gpu}
            worker = run / f"workers/seed_{SEED}.json"
            scheduler._atomic_status(worker, {**status, "status": "initializing"})
            try:
                result = seed_runner.run_seed(
                    root=root, run_directory=run, contract=contract, authorization=approval, seed=SEED,
                    backend=TrainingBackend(root, cache, SEED), call=_no_provider, resume_boundary=boundary,
                    boundary_user_message=user, transport_options={"managed_attempt": local},
                    response_projection=partial(project, original=plan),
                    fit_slot=partial(scheduler.gpu_slot, run, SEED, gpu, contract.lease_root),
                )
                _require_no_bridge(run)
                scheduler._atomic_status(worker, {**status, **asdict(result)})
                return result
            except Exception as error:
                phase = "paused" if scheduler._pause_requested(run, SEED) else "failed"
                scheduler._atomic_status(worker, {**status, "status": phase, "error_type": type(error).__name__, "reason": str(error)})
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run", "resume"))
    for name in ("ordinary-release", "ordinary-run", "paid-release", "paid-run", "run-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    ordinary_root, ordinary, paid_root, paid, run = (p.resolve() for p in (
        args.ordinary_release, args.ordinary_run, args.paid_release, args.paid_run, args.run_dir,
    ))
    contract, boundary, cache, original, plan, reply = construct(ROOT, ordinary_root, ordinary, paid_root, paid, run)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        parser.error("the exact new local import and CUDA authorization is required")
    progress = execute(ROOT, paid, run, contract, read(args.authorization), boundary, cache, original, plan, reply,
                       resume=args.action == "resume")
    print(dumps(asdict(progress)))


if __name__ == "__main__":
    main()
