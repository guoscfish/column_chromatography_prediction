"""Fit seed 887 through L973 from its exact archived R19 final, with zero provider posts."""

import argparse
import csv
from dataclasses import asdict, replace
from functools import partial
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.frozen_state import SPLITS
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import parse_response
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.runner import _validation
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    ResponseMessagePhaseError, TransportStop, make_request, response_output_text, terminal_record,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import provider, scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import validate_result, verify_job
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, STUDY, verify_authorization
from .run_v13_bounded_sse_recovery import PREFIX, RunIdentity, _archived
from .run_v13_887_r13_cached_correction import _no_provider, _require_no_bridge


SEED, ROUND, TARGET = 887, 19, 973
MODULE = "run_v13_887_r19_cached"
RUN_NAME = f"{PREFIX}-887-r19-cached-01"
OLD_RELEASE = "ipv4-v13-887-r14-continuation-01-07fa47eb124f9d391186594d50ea0190e9de933163d240e21c3c8e78420a2bf3"
OLD = RunIdentity("-887-r14-continuation-01", "1cb3a62bf07bda39f88c82b86fdce3c1a729b0fac35ae016662c3f9d9a3d5485",
                  "cb91385d64b48dedfbf819b69e7e1a9da0fb99679ac642470f34e0907d991226")
ATTEMPT = "a4c01eea321444ba9b40dd9ba205fab6"
REQUEST = "7edf87cd8a09b2acf3ec0d31a9de942ef8020a980c36a7fb37b4df3eba380e7f"
PAYLOAD = "84ba6c4287f6d0e87ce387dfc0524695dc73cf87f4c20f9826f12b326341bac9"
JOB = "91156cf729d2eaf1394b7226a1f5b0737ff3aa08ea09898ffa16a5573aea9bbc"
RESULT = "0c1c0dee4c8033874f20c03377c354f6bd4480d4c439a656f49885eaef67e38d"
RAW = "bd5cc293ae0bce087b867e57ac3952abc51e8f925f71c68dbcef5b0989ee5077"
PLACEHOLDER = "5245395d4068007d015c28718df830eee63129cefa76890151cb9008e2be8bd7"
FINAL = "76abf13909888f623c1e853231308281b3fd41eb61407999880d6abf945da955"
TEXT = "8699f5f047984c8b5fef18ee7c1b119b7ae19ec7558a1fca1520f99d70a0bd1a"
PLAN = "023fd20b0d630b51df92353b1c8a6de69a5bbd90ed4a1d71f7af97aa70a5169d"
ORDER = "bdc2cbebfce55b823eeeffa453dc0c2d15b185075ea50d8e2243a3aaaee12be8"
CHECKPOINT = "f4003ffe1e52133ca2c994ca9c281aea6d4a19215c4b1afa497a8f8fc881e4c4"
MEMORY = "2dafbda36f27349a4d40fc5a5292d85a85f885df91757cf228f041cf51d3224b"
RESPONSE = "resp_ft_b3057b24d2204c009332156fc0aa411c"
PROVIDER_REQUEST = ("544ab56c-bcad-4ac1-9a9a-bc47d38012dd, a1701488-60c6-4bef-8b0c-c42faf5ac47b, "
                    "6abb12bc-f341-4220-8083-0ca10081cb04")
CAPTURE = "7e73fcf8818385a43651954e383dea17d2b3eb24be8af34ad54d9cec46d8fa7b"
ACK = "655e804988d9541232e1ccce5404466852441c514a32cc7809f0bbf0999741bf"
PROGRESS = "4a65fd4fea8667b6dff0dc5d841a8b0d9d9d44bcc79f8a4d70b3f44865b12ff7"
USAGE = {"input_tokens": 358211, "output_tokens": 14039, "reasoning_tokens": 5435,
         "cached_tokens": 0, "total_tokens": 372250}
FAILURE = {"code": "invalid_response_message_phase", "ambiguous": False, "retryable": False,
           "retry_after": None, "reply_identity": None}
ARCHIVE_FILES = {
    "bridge/outbox/" + ATTEMPT + ".json": "7917bfb59bf21aba318c7e6bec909ce87f4374b918aa52a2a85a981ce50b8d56",
    "bridge/results/" + ATTEMPT + ".json": "61d12c9eb1ad43fadd24d19a28e07d6fc0898f28f8278f81b26bb4f39f80bdb8",
    "seed_887/round_19/turn_00/request.json": "dbe993270c33a62bb2069cd2ea8f0affcc91d59d65ec0f8c2b84c61bbe73a20b",
    "seed_887/round_19/turn_00/attempt_00.started.json": "574d219c78cac39c824798aa2cef3ab3f9a3b21ad4503027ad7308c0ab637fcd",
    "seed_887/round_19/turn_00/attempt_00.terminal.json": "ca6bb28ff65af881f458a618855aa12c3da726183271327cd14e2bf470eb21ba",
    "seed_887/round_19/source.json": "7227174710df93445c90b9877b5ca9e25b3696767e541d0659832d5c103fd255",
    "seed_887/round_19/transition.json": "16109d74cc5f25ba8befc9713bb166917b7da5745eb3ceb666ad1a7e97f6bd56",
    "seed_887/round_18/memory.json": "fa3e172111974ce21a20f339674cf122242bfa830a536d5ebadd2ce2ab4fe45b",
    "workers/seed_887.json": "d634bf087526621edc945aa017d7ea08fa2f4a9ad69a181b152150447f92c473",
    "seed_887/pause.request": "e2615ea52a16df716103c2a664de2b785ed13540c9a757f4c6cfcf389472ffe4",
    "provider_model_lock.json": "950c0fd7b1d3135305a6fa7024419205130d90d8962b46b7fbeed5d050fa5041",
}
RECOVERY_BUDGET = {"maximum_logical_requests": 1, "maximum_physical_attempts": 0}
BUDGET = {"first_logical_requests": 1, **RECOVERY_BUDGET}


def _extract_final(raw, transport, lock):
    if (digest(raw) != RAW or (raw.get("id"), raw.get("status"), raw.get("model"), raw.get("error"),
            raw.get("incomplete_details")) != (RESPONSE, "completed", "gpt-6.2-astra", None, None)):
        raise PermissionError("the exact canonical provider completion changed")
    items = raw.get("output")
    if type(items) is not list or len(items) != 10 or any(type(item) is not dict for item in items):
        raise PermissionError("the archived output item count or type changed")
    placeholder, *reasoning, final = items
    if (digest(placeholder) != PLACEHOLDER or placeholder.get("phase") is not None
            or (placeholder.get("type"), placeholder.get("role"), placeholder.get("status"))
            != ("message", "assistant", "completed")
            or placeholder.get("content") != [{"type": "output_text", "text": " "}]
            or any(item.get("type") != "reasoning" for item in reasoning)
            or digest(final) != FINAL or (final.get("type"), final.get("role"), final.get("phase"), final.get("status"))
            != ("message", "assistant", "final_answer", "completed")):
        raise PermissionError("only the one ASCII space placeholder may be ignored beside the explicit final")
    try:
        response_output_text(items)
    except ResponseMessagePhaseError:
        pass
    else:
        raise PermissionError("the immutable parser did not reproduce the archived phase rejection")
    declared = {"model_version": raw.get("model_version"), "reported_reasoning_effort": raw["reasoning"]["effort"]}
    metadata = {"raw_response_sha256": RAW, "provider_request_id": PROVIDER_REQUEST, **declared}
    if digest({"attempt_id": ATTEMPT, "job_sha256": JOB, "metadata": metadata}) != CAPTURE:
        raise PermissionError("the recorded Mac capture identity changed")
    reply = provider._reply_from_raw({**raw, "output": items[1:]}, PROVIDER_REQUEST)
    if digest(reply.text) != TEXT or reply.usage != USAGE or provider.verify_reported_model(transport, reply, declared) != lock:
        raise PermissionError("the explicit final, original usage or served model differs")
    return reply


def _fenced(old):
    directory = old / f"seed_{SEED}"
    active = directory / f"round_{ROUND:02d}"
    turn = active / "turn_00"
    worker = read(old / f"workers/seed_{SEED}.json")
    pid = worker.get("pid")
    try:
        arguments = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, ProcessLookupError):
        arguments = ()
    if (worker.get("status") != "failed" or type(pid) is not int or str(old).encode() in arguments
            or set(active.glob("turn_*")) != {turn}
            or {p.name for p in turn.glob("attempt_*")} != {"attempt_00.started.json", "attempt_00.terminal.json"}
            or any((active / name).exists() for name in ("selection.json", "batch_freeze.json", "memory.json"))
            or (turn / "decision.json").exists() or (directory / f"round_{ROUND + 1:02d}").exists()
            or any(file_digest(old / name) != expected for name, expected in ARCHIVE_FILES.items())):
        raise PermissionError("the stopped predecessor, exact paid receipts or pause fence changed")
    return directory, turn


def _split(root):
    path = root / SPLITS / f"row_seed_{SEED}.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return SimpleNamespace(l0_ids=tuple(row["sample_id"] for row in rows if row["role"] == "l0"),
                           u0_ids=tuple(row["sample_id"] for row in rows if row["role"] == "u0"),
                           split_sha256=file_digest(path))


def _boundary(old_release, directory, previous, approval):
    scope = seed_runner._scope(previous, approval, SEED, False)
    descriptor = previous.profile["resume_boundaries"][str(SEED)]
    start = descriptor["round"]
    initial = directory / f"round_{start:02d}"
    state = seed_runner.FrozenState(**read(initial / "source.json"))
    memory = read(initial / "boundary_memory.json")
    if (start != 14 or read(directory / "boundary.json") != {"formal_run": scope, **descriptor}
            or digest(asdict(state)) != descriptor["state_sha256"] or digest(memory) != descriptor["memory_sha256"]):
        raise PermissionError("the signed prior L781 boundary changed")
    prefix = {"initial_boundary": file_digest(directory / "boundary.json")}
    split = _split(old_release)
    for index in range(start, ROUND):
        current = directory / f"round_{index:02d}"
        following = directory / f"round_{index + 1:02d}"
        state, memory = seed_runner._successor(current, state, following, scope, previous.transport, previous, split)
        prefix[f"transition_{index + 1:02d}"] = file_digest(following / "transition.json")
    if (state.round != ROUND or state.checkpoint != CHECKPOINT or digest(memory) != MEMORY
            or asdict(state) != read(directory / f"round_{ROUND:02d}" / "source.json")):
        raise PermissionError("the exact R14–R18 prefix did not close at the R19 boundary")
    return (state, memory, ROUND), prefix


def _archive(old, turn, contract, approval, stage):
    request = read(turn / "request.json")
    started = read(turn / "attempt_00.started.json")
    terminal = terminal_record(turn, 0)
    filename = f"{ATTEMPT}.json"
    job = read(old / "bridge/outbox" / filename)
    result = validate_result(read(old / "bridge/results" / filename), job)
    scope = seed_runner._scope(contract, approval, SEED, False)
    binding = verify_job(job, contract_sha256=contract.digest, authorization_sha256=digest(approval),
                         wire_representation=contract.profile["wire_representation"])
    grant = seed_runner._selection_grant(scope, REQUEST)
    lock = read(old / "provider_model_lock.json")
    if (request != make_request(stage.binding, 0, stage.messages, contract.transport)
            or read(turn / "formal_request_authorization.json") != grant or read(turn.parent / "authorization.json") != grant
            or binding != (request, contract.transport, started) or started["attempt_id"] != ATTEMPT
            or (request["request_id"], request["payload_sha256"], job["job_sha256"], result["result_sha256"],
                result["raw_response_sha256"]) != (REQUEST, PAYLOAD, JOB, RESULT, RAW)
            or result["outcome"] != {"status": "failed", "failure": FAILURE} or result["provider_dispatched"] is not True
            or result["model_identity"] is not None or lock != contract.predecessor["model_identity"]
            or terminal["status"] != "stopped" or terminal.get("error") != FAILURE["code"]
            or terminal.get("ambiguous") is not False or terminal.get("provider_dispatched") is not True
            or terminal.get("bridge_job_sha256") != JOB or terminal.get("bridge_result_sha256") != RESULT
            or "answer_sha256" in terminal or any(terminal.get(key) != value for key, value in started.items())):
        raise PermissionError("the exact archived final is detached from its paid R19 request and stopped receipt")
    return request, _extract_final(result["raw_response"], contract.transport, lock)


def _verified_plan(stage, reply):
    plan = parse_response(reply.text)
    validation = _validation(stage, plan)
    order = [choice["candidate_id"] for choice in plan["choices"]]
    if digest(plan) != PLAN or digest(order) != ORDER or len(set(order)) != 32 or not validation.valid or validation.valid_choice_count != 32:
        raise PermissionError("the exact final answer failed the unchanged official A7 validator")
    return plan


def construct(root, old_release, old, run):
    if root == old_release or old_release.name != OLD_RELEASE or run.name != RUN_NAME or run.parent != old.parent:
        raise PermissionError("the zero-POST fit requires its exact immutable predecessor and separate sibling")
    previous, approval = _archived(old, old_release, OLD)
    directory, turn = _fenced(old)
    boundary, prefix = _boundary(old_release, directory, previous, approval)
    state, memory, _ = boundary
    scope = seed_runner._scope(previous, approval, SEED, False)
    stage = seed_runner._stage(old_release, state, memory, scope, previous)
    request, archived = _archive(old, turn, previous, approval, stage)
    _verified_plan(stage, archived)
    cache = Path(previous.predecessor["cache_directory"]).resolve(strict=True)
    resource = file_digest(cache / "resource_receipt.json")
    if resource != previous.predecessor["public_resource_receipt_sha256"]:
        raise PermissionError("the signed public QGeoGNN training cache changed")
    user = request["payload"]["input"][1]
    descriptor = {"round": ROUND, "state_sha256": digest(asdict(state)), "memory_sha256": digest(memory),
                  "user_message_sha256": digest(user)}
    transport = replace(previous.transport, logical_replies=1, transport_attempts=1)
    profile = {**previous.profile, "target_labels": TARGET, "transport": asdict(transport),
               "resume_boundaries": {str(SEED): descriptor}}
    for attr in ("audited_future_transport_replay", "sealed_relay_retry_seconds", "replay_eligible_from_round"):
        profile.pop(attr, None)
    sources = {key: file_digest(root / key) for key in previous.source_sha256}
    if sources != previous.source_sha256:
        raise PermissionError("the local import cannot change existing signed code or the standard A7 prompt")
    for module in (MODULE, "run_v13_887_r13_cached_correction"):
        path = (STUDY / "scripts" / f"{module}.py").as_posix()
        sources[path] = file_digest(root / path)
    custody = {"provider_post": False, "attempt_id": ATTEMPT, "request_id": REQUEST, "payload_sha256": PAYLOAD,
               "job_sha256": JOB, "result_sha256": RESULT, "raw_response_sha256": RAW, "response_id": RESPONSE,
               "reply_text_sha256": TEXT, "plan_sha256": PLAN, "ordered_candidate_ids_sha256": ORDER,
               "ignored_placeholder_item_sha256": PLACEHOLDER, "explicit_final_item_sha256": FINAL,
               "archived_usage": archived.usage, "archived_provider_request_id": PROVIDER_REQUEST,
               "mac_capture_sha256": CAPTURE, "mac_ack_file_sha256": ACK, "mac_progress_file_sha256": PROGRESS,
               "original_failure": FAILURE, "archived_files": ARCHIVE_FILES, "confirmed_boundary_prefix": prefix}
    proof = {"dispatcher_module": MODULE, "seed": SEED, "round": ROUND, "cached_selection": custody,
             "previous_run_directory": str(old), "previous_release_directory": str(old_release),
             "previous_contract_sha256": OLD.contract, "previous_authorization_sha256": OLD.approval,
             "recovery_budget": RECOVERY_BUDGET, "execution_budget": BUDGET, "checkpoint_sha256": CHECKPOINT,
             "cache_directory": str(cache), "public_resource_receipt_sha256": resource,
             "model_identity": previous.predecessor["model_identity"]}
    contract = replace(previous, profile=profile, transport=transport, source_sha256=sources,
                       run_directory=str(run), predecessor=proof)
    return contract, boundary, cache, request, archived


def _local_stage(root, contract, approval, boundary, original, archived):
    scope = seed_runner._scope(contract, approval, SEED, False)
    user = original["payload"]["input"][1]
    stage = seed_runner.pin_boundary_user_message(seed_runner._stage(root, *boundary[:2], scope, contract), user, digest(user))
    request = make_request(stage.binding, 0, stage.messages, contract.transport)
    custody = contract.predecessor["cached_selection"]
    if (request["payload"] != original["payload"] or request["request_id"] == REQUEST
            or custody["provider_post"] is not False or contract.predecessor["execution_budget"] != BUDGET
            or contract.predecessor["recovery_budget"] != RECOVERY_BUDGET or contract.target_labels != TARGET):
        raise PermissionError("the isolated local decision cannot differ from the original request or submit a provider post")
    _verified_plan(stage, archived)
    return stage, user, request


def _virtual_evidence(custody):
    return {"provider_dispatched": False, "cached_selection_sha256": digest(custody),
            "imported_from_response_id": RESPONSE}


def _local_terminal(turn, request, archived, custody):
    terminal = terminal_record(turn, 0)
    started = read(turn / "attempt_00.started.json")
    if (read(turn / "request.json") != request or terminal.get("request_id") != request["request_id"]
            or terminal.get("status") != "completed" or terminal.get("usage") != {}
            or terminal.get("answer_sha256") != TEXT or terminal.get("response_id") != RESPONSE
            or terminal.get("text") != archived.text or started.get("attempt_id") == ATTEMPT
            or any(terminal.get(key) != value for key, value in started.items())
            or any(terminal.get(key) != value for key, value in _virtual_evidence(custody).items())):
        raise PermissionError("the local attempt differs from the exact zero-POST final")
    return terminal


def verify_handoff(root, old_release, old, run, approved_contract_sha256, approved_authorization_sha256):
    """Read the owned immutable release and return the signed zero-POST L973 boundary proof."""
    contract, boundary, _, original, archived = construct(root, old_release, old, run)
    approval = verify_authorization(contract, read(run / "authorization.json"))
    if ((contract.digest, digest(approval)) != (approved_contract_sha256, approved_authorization_sha256)
            or read(run / "formal_contract.json") != contract.record()):
        raise PermissionError("the cached L973 contract or authorization differs from its signed identity")
    _require_no_bridge(run)
    worker = read(run / f"workers/seed_{SEED}.json")
    if (worker.get("status"), worker.get("completed_rounds"), worker.get("active_labels")) != ("complete", ROUND + 1, TARGET):
        raise PermissionError("the one-batch archival worker has not completed the exact L973 fit")
    stage, _, request = _local_stage(root, contract, approval, boundary, original, archived)
    scope = seed_runner._scope(contract, approval, SEED, False)
    directory = run / f"seed_{SEED}"
    active, following = (directory / f"round_{index:02d}" for index in (ROUND, ROUND + 1))
    turn = active / "turn_00"
    custody = contract.predecessor["cached_selection"]
    terminal = _local_terminal(turn, request, archived, custody)
    if (read(run / "cached_response_import.json") != custody
            or read(directory / "boundary.json") != {"formal_run": scope, **contract.profile["resume_boundaries"][str(SEED)]}
            or read(active / "source.json") != asdict(boundary[0]) or read(active / "boundary_memory.json") != boundary[1]
            or {p.name for p in active.glob("turn_*")} != {"turn_00"}
            or {p.name for p in turn.glob("attempt_*")} != {"attempt_00.started.json", "attempt_00.terminal.json"}):
        raise PermissionError("the local archival custody and isolated R19 boundary changed")
    freeze = seed_runner._verify_selection(active, boundary[0], scope, contract.transport, contract.batch_size, stage=stage)
    if (freeze["plan"] != _verified_plan(stage, archived) or freeze["accepted_logical_index"] != 0
            or freeze["accepted_attempt"] != 0 or len(set(freeze["batch_ids"])) != 32):
        raise PermissionError("the sealed L973 selection is not the exact ordered archived model decision")
    state, memory = seed_runner._successor(active, boundary[0], following, scope, contract.transport, contract, _split(root))
    if (state.seed, state.round, len(state.provenance["expected_observed"])) != (SEED, ROUND + 1, TARGET):
        raise PermissionError("the fitted state does not close the one-batch L973 contract")
    return {"contract_sha256": contract.digest, "authorization_sha256": digest(approval),
            "checkpoint_sha256": state.checkpoint, "transition_file_sha256": file_digest(following / "transition.json"),
            "source_file_sha256": file_digest(following / "source.json"), "state_sha256": digest(asdict(state)),
            "scientific_memory_sha256": digest(memory), "fit_audit_file_sha256": state.provenance["fit_audit_sha256"],
            "freeze_sha256": freeze["freeze_sha256"], "cached_selection_sha256": digest(custody),
            "local_terminal_sha256": file_digest(turn / "attempt_00.terminal.json"),
            "local_attempt_id": terminal["attempt_id"], "archived_provider_response_id": RESPONSE,
            "candidate_order_sha256": ORDER, "provider_post": False}


def execute(root, old, run, contract, approval, boundary, cache, original, archived, *, resume):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    approval = verify_authorization(contract, approval)
    _, user, request = _local_stage(root, contract, approval, boundary, original, archived)
    custody = contract.predecessor["cached_selection"]
    reply = replace(archived, usage={}, transport_evidence=_virtual_evidence(custody))

    def local(candidate, config, started):
        _require_no_bridge(run)
        if candidate != request or config != contract.transport or started.get("attempt") != 0:
            raise TransportStop("only the exact archived R19 result may fill the local attempt")
        return reply

    gpu = contract.gpu_slots[0]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with scheduler._lock(old / "coordinator.lock", busy="the old R19 coordinator must remain retired"):
        scheduler._assert_no_old_workers(old, (SEED,))
        _fenced(old)
        with scheduler._lock(run / "coordinator.lock", busy="the local R19 coordinator is already active"):
            _require_no_bridge(run)
            scheduler._assert_no_old_workers(run, (SEED,))
            directory = seed_runner.seed_directory(run, SEED)
            if scheduler._pause_requested(run, SEED) and not resume:
                raise PermissionError("use resume for the paused local archival fit")
            once(run / "formal_contract.json", contract.record())
            once(run / "authorization.json", approval)
            once(run / "provider_model_lock.json", contract.predecessor["model_identity"])
            once(run / "cached_response_import.json", custody)
            turn = directory / f"round_{ROUND:02d}/turn_00"
            if (turn / "attempt_00.terminal.json").is_file():
                _local_terminal(turn, request, archived, custody)
            if resume:
                (run / "pause.request").unlink(missing_ok=True)
                (directory / "pause.request").unlink(missing_ok=True)
            scheduler._atomic_status(run / "coordinator_generation.json", {"id": uuid4().hex, "pid": os.getpid()})
            status = {"seed": SEED, "pid": os.getpid(), "gpu": gpu}
            worker = run / f"workers/seed_{SEED}.json"
            scheduler._atomic_status(worker, {**status, "status": "initializing"})
            try:
                result = seed_runner.run_seed(
                    root=root, run_directory=run, contract=contract, authorization=approval, seed=SEED,
                    backend=TrainingBackend(root, cache, SEED), call=_no_provider, resume_boundary=boundary,
                    boundary_user_message=user, transport_options={"managed_attempt": local},
                    fit_slot=partial(scheduler.gpu_slot, run, SEED, gpu, contract.lease_root),
                )
                _require_no_bridge(run)
                scheduler._atomic_status(worker, {**status, **asdict(result)})
                return result
            except Exception as error:
                state = "paused" if scheduler._pause_requested(run, SEED) else "failed"
                scheduler._atomic_status(worker, {**status, "status": state, "error_type": type(error).__name__})
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run", "resume", "verify-handoff"))
    for name in ("old-release", "old-run", "run-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--contract-sha256")
    parser.add_argument("--authorization-sha256")
    args = parser.parse_args()
    os.umask(0o077)
    old_release, old, run = (path.resolve() for path in (args.old_release, args.old_run, args.run_dir))
    if args.action == "verify-handoff":
        if args.contract_sha256 is None or args.authorization_sha256 is None:
            parser.error("provide both digests of the signed archival import")
        print(dumps(verify_handoff(ROOT, old_release, old, run, args.contract_sha256, args.authorization_sha256)))
        return
    contract, boundary, cache, original, archived = construct(ROOT, old_release, old, run)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        parser.error("provide the signed zero-POST and one-CUDA-fit authorization")
    result = execute(ROOT, old, run, contract, read(args.authorization), boundary, cache, original, archived,
                     resume=args.action == "resume")
    print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
