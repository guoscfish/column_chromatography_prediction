"""Import one validated archived 887 correction from its exact raw reply, with no model submission."""

import argparse
from dataclasses import asdict, replace
from functools import partial
import os
from pathlib import Path
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import parse_response
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.runner import _validation
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    ProviderReply, ResponseMessagePhaseError, TransportStop, make_request, response_output_text, terminal_record,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import provider, scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import validate_result, verify_job
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, STUDY, verify_authorization
from . import run_v13_887_r13_targeted_repair as repair
from .run_v13_lane_continuation import _extension_approval


SEED, ROUND = repair.SEED, repair.ROUND
RUN_NAME = f"{repair.relay.bounded.PREFIX}-887-r13-cached-correction-01"
MODULE = "run_v13_887_r13_cached_correction"
CORRECTION = repair.relay.bounded.RunIdentity(
    "-887-r13-targeted-repair-01", "1e19e1e8c13e5b7b0c630fb7c79787f75ff4ffa885c1c075fb3a641cf49151f0",
    "91a621dedc98c5be52d374d3c40584e9b85257685c1cf6a5b9fd11c91dd916ed",
)
ATTEMPT = "97c807dafdd64bdb92fe4b47065727b5"
REQUEST = "30100cb9ccb8f6a30335c44d6f03d176352644a76e39b90e4765c61076db264f"
PAYLOAD = "3a15f8e8e1cf540ee5c9ceff059f3bb544ea00a3f507690c746f2e45878b56c8"
JOB = "8aa48f7483ed84dcaa31c454d30082629ecc454f2d1c72f8e30a5d83ee7aa7cd"
JOB_FILE = "382415bf4adc72585e96b0cce05af4c8c25a602f3d57104033186330fdce32e1"
RESULT = "4162b8bfa0be7ac1439c1c096109c868db2c2d9f71e42a916166224ff1392a82"
RESULT_FILE = "393f20efe790939be240138a7e86b88ee98a01b2c7522d08947c02638493e585"
RAW = "d760937aa2611f70516be15fba0a6027e1ecd76f6299f9e68cb87e49c3e93926"
PLACEHOLDER_ITEM = "3ee13ae0a68689b8983b7bed9a12da37bf584a288aea23559b0c19cfc9d87a34"
FINAL_ITEM = "cb6791f01920f40320691e9768fb803c8d448447dac4ac00b58972b0d2b30f47"
TEXT = "b4f1e2dd57336bdd51a20750be0c6e13814d84463882075dc988075e34591113"
PATCH = "47c1275ceb3de3da2868bcdab3ebf30c354e94cf1160a651c55589468029c8fd"
PROJECTED = "cf78fb629ea49be94be539a05fb4ef96d7ae81121940b68879b81e8a3302f3eb"
RESPONSE = "resp_ft_89f9256d2c114dbcb45fb9db64e3044c"
PROVIDER_REQUEST_ID = ("65bc227c-bda2-4762-8563-f1a14422aa80, d6f0e867-cb8a-4051-a09f-7bd8a2b5afef, "
                       "841fbf9f-b2da-4717-aaf1-a1a1169fb35c")
MAC_CAPTURE = "073ee9f552d89ff3e29b99fd0072dd3d271b380126abf0898ef821285b12b54c"
MAC_ACK = "a5405ea5506a8a2e82a42ecada0d22454c3bea660bd9cebba961b0a65b728770"
PAUSE = "71eaff5600feba3a24aedccec9ed41efc4fd0495ca21dc43d3e01f5ae3602d1c"
WORKER = "e04eb0c6fa1f02e871b446937412bb03cde64f65290f8e92043a229d1d19a0ea"
TERMINAL = "5acfb0d53f27fc2c8262e453dfaf595e6535b761cd8bb8d70c8f404d37fc6a45"
FAILURE = {"code": "invalid_response_message_phase", "ambiguous": False, "retryable": False,
           "retry_after": None, "reply_identity": None}
USAGE = {"input_tokens": 353339, "output_tokens": 285, "total_tokens": 353624,
         "cached_tokens": 0, "reasoning_tokens": 29}
BUDGET = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 0}
SOURCES = tuple(STUDY / "scripts" / f"{module}.py" for module in (MODULE, "run_v13_lane_continuation"))


def project(patch, *, original):
    if digest(patch) != PATCH or digest(original) != repair.PLAN:
        raise PermissionError("only the archived final model patch may be imported")
    plan = repair.project_reply(patch, original=original)
    if digest(plan) != PROJECTED:
        raise PermissionError("the exact ordered correction changed")
    return plan


def _extract_final(raw, transport, model_lock):
    if (type(raw) is not dict or digest(raw) != RAW
            or (raw.get("id"), raw.get("status"), raw.get("model"), raw.get("error"), raw.get("incomplete_details"))
            != (RESPONSE, "completed", "gpt-6.2-astra", None, None)):
        raise PermissionError("the archived response digest, identity or completion changed")
    items = raw.get("output")
    if type(items) is not list or len(items) != 3 or any(type(item) is not dict for item in items):
        raise PermissionError("the archived output item count or shape changed")
    placeholder, reasoning, final = items
    if (digest(placeholder) != PLACEHOLDER_ITEM or placeholder.get("type") != "message"
            or placeholder.get("phase") is not None or placeholder.get("role") != "assistant"
            or placeholder.get("status") != "completed" or placeholder.get("content") != [{"type": "output_text", "text": " "}]
            or reasoning.get("type") != "reasoning" or digest(final) != FINAL_ITEM
            or (final.get("type"), final.get("role"), final.get("phase"), final.get("status"))
            != ("message", "assistant", "final_answer", "completed")):
        raise PermissionError("only the pinned one-space placeholder may be ignored beside the explicit final answer")
    try:
        response_output_text(items)
    except ResponseMessagePhaseError:
        pass
    else:
        raise PermissionError("the original output does not reproduce the archived message-phase rejection")
    declared = {"model_version": raw.get("model_version"), "reported_reasoning_effort": raw["reasoning"]["effort"]}
    capture = {"attempt_id": ATTEMPT, "job_sha256": JOB,
               "metadata": {"raw_response_sha256": RAW, "provider_request_id": PROVIDER_REQUEST_ID, **declared}}
    if digest(capture) != MAC_CAPTURE:
        raise PermissionError("the model and request identity differ from the exact local capture receipt")
    # This temporary projection never replaces the stored provider response or the failed bridge receipt.
    isolated = {**raw, "output": [reasoning, final]}
    reply = provider._reply_from_raw(isolated, PROVIDER_REQUEST_ID)
    if (digest(reply.text) != TEXT or reply.usage != USAGE
            or provider.verify_reported_model(transport, reply, declared) != model_lock):
        raise PermissionError("the explicit final answer, original usage or served model changed")
    return asdict(reply)


def _read_archive(old, active, contract, approval, original):
    turn = repair.relay.bounded._turn(active, 1)
    if ((file_digest(active.parent / "pause.request"), file_digest(turn / "attempt_00.terminal.json"),
         file_digest(old / f"workers/seed_{SEED}.json")) != (PAUSE, TERMINAL, WORKER)
            or (turn / "decision.json").exists()):
        raise PermissionError("the prior sealed transport rejection or worker fence changed")
    request = repair.relay.bounded._request(active, turn, contract, approval, repair.LANE)
    repair.verify_patch_request(request, original, contract.profile["fixed_reference_repair"], attempt=0)
    started = read(turn / "attempt_00.started.json")
    terminal = terminal_record(turn, 0)
    filename = f"{ATTEMPT}.json"
    if any({path.name for path in (old / "bridge" / folder).glob("*.json")} != {filename}
           for folder in ("outbox", "claims", "results")):
        raise PermissionError("the closed one-physical-request mailbox changed")
    job_path, result_path = (old / "bridge" / folder / filename for folder in ("outbox", "results"))
    if (file_digest(job_path), file_digest(result_path)) != (JOB_FILE, RESULT_FILE):
        raise PermissionError("the archived bridge files differ from the delivered Mac files")
    job = read(job_path)
    result = validate_result(read(result_path), job)
    bound = verify_job(job, contract_sha256=contract.digest, authorization_sha256=digest(approval),
                       wire_representation=contract.profile["wire_representation"])
    model_lock = read(old / "provider_model_lock.json")
    if (bound != (request, contract.transport, started) or started["attempt_id"] != ATTEMPT
            or (request["request_id"], request["payload_sha256"], job["job_sha256"], result["result_sha256"],
                result["raw_response_sha256"]) != (REQUEST, PAYLOAD, JOB, RESULT, RAW)
            or result["outcome"] != {"status": "failed", "failure": FAILURE} or result["provider_dispatched"] is not True
            or result["model_identity"] is not None or model_lock != contract.predecessor["model_identity"]
            or terminal["status"] != "stopped" or terminal.get("error") != FAILURE["code"]
            or terminal.get("ambiguous") is not False or terminal.get("provider_dispatched") is not True
            or terminal.get("bridge_job_sha256") != JOB or terminal.get("bridge_result_sha256") != RESULT
            or any(terminal.get(k) != v for k, v in started.items())):
        raise PermissionError("the local correction is detached from its exact paid raw response and settled failure")
    return request, _extract_final(result["raw_response"], contract.transport, model_lock)


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
        raise PermissionError("the completed paid repair's L749 input state or scientific memory changed")
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
                "previous_seed_pause_file_sha256": PAUSE, "old_terminal_file_sha256": TERMINAL,
                "previous_worker_file_sha256": WORKER, "prior_failure_code": FAILURE["code"],
                "ignored_placeholder_item_sha256": PLACEHOLDER_ITEM, "explicit_final_item_sha256": FINAL_ITEM,
                "mac_capture_sha256": MAC_CAPTURE, "mac_ack_file_sha256": MAC_ACK,
                "archived_provider_request_id": PROVIDER_REQUEST_ID, "archived_usage": reply["usage"]}
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
            or contract.predecessor["execution_budget"] != BUDGET or contract.target_labels != repair.TARGET):
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
            prior = directory / f"round_{ROUND:02d}/turn_00/attempt_00.terminal.json"
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
