"""Fit seed 6101 through L685 from its exact archived R10 final, with zero provider posts."""

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
from .run_v13_6101_r9_continuation import _archival_lock, _require_fixed_reconstruction


SEED, ROUND, TARGET = 6101, 10, 685
MODULE = "run_v13_6101_r10_cached"
RUN_NAME = f"{PREFIX}-6101-r10-cached-01"
OLD_RELEASE = "ipv4-v13-6101-r9-continuation-01-c492b4c8896191eb2a330efc24e76d3dade7e80919dcb67cfbff48d8de54f06a"
OLD = RunIdentity("-6101-r9-continuation-01", "e63dff127877bed96403a8c4174dcddb1b43c68913ed60bae7631ce0b03be5c2",
                  "989a6b41a0efc0363336bc9fa0af2b9c47f2d32eb52843a51b1f9341f7464768")
ATTEMPT = "bb2d8e5e07034ead99c65028309958d3"
REQUEST = "4c56106c2d3b8ca0b128e8de966215070beb839ed27b586a110a1c60248c98bc"
PAYLOAD = "7d31903d2a80f99d16f61fc8fd45992a8b5fe7ac7df10386860811788b7cbeb9"
JOB = "d280900759cbf700206cf04adb6ee9cc1f527e6152e300cf02f1d134789dca45"
RESULT = "5b6b9526753a5988142c28c0fd459410ee72fe2d2b8e5cb59d6afbac69809dd0"
RAW = "ce57eaee71de85fc7d91901ecac83d40285984cd0675f15f60e7abf27fd6209e"
PLACEHOLDER = "9a2e4cd783976ef07cd1c5d1974714523516fb13560a6efec7abc99448042a11"
FINAL = "ac4c430bc58487a7f361690b4302c80f9526309ada844ab001d5d693d9dde243"
TEXT = "7df4677768fe854e96c60afcca850d6a63fc47706f3032e8bc220a21f1a6546d"
PLAN = "96bc176a52b1dbafa6acee045b10ed9f373405a373b348870c3aa9d7f999c3ab"
ORDER = "cf0e319868866717efeabf81d8752c65a49c5f01c82a34308745785c1a09e276"
CHECKPOINT = "dbeba66c3bd67937dfc0d5070104fb24a9c539960c8cb0f07c20efbec167b4fc"
MEMORY = "b2bda9d5df5de414da38d0a4ccf75757f29a7b2692ceebcd96c51bb10dc25deb"
RESPONSE = "resp_ft_9a8a0de08c1445c0baec11aeb4614f51"
PROVIDER_REQUEST = ("d079c16d-4994-4965-b335-fa6e7479108a, c6518934-0a43-41e5-871f-cd7e41b492f9, "
                    "9625ae0d-e131-48bb-945d-f995756e85d3")
CAPTURE = "28b6354002a89ecd7808696801fdc98e5ae2ed371335641a59a0efe3d6c32683"
CAPTURE_FILE = "b086741020a35a7db217184a8d5524789040e65429384b44d719a32dc14b1550"
RAW_FILE = "b7572b4060862e8732491af379a44932ad948aa0ab2ee5f3028f8a35a4fa3017"
ACK = "46162d1dbda1533eb439674327991009c5d075be85d20181b56b41e474e7bce9"
PROGRESS = "fc3af32835dd5baa8db9dfb8616bfb19f5724d2f335f1219abdaf045351a743d"
USAGE = {"input_tokens": 366104, "output_tokens": 16056, "reasoning_tokens": 5517,
         "cached_tokens": 0, "total_tokens": 382160}
FAILURE = {"code": "invalid_response_message_phase", "ambiguous": False, "retryable": False,
           "retry_after": None, "reply_identity": None}
ARCHIVE_FILES = {
    "bridge/outbox/" + ATTEMPT + ".json": "3631185a747283873f3b1fea93eebe5f0a5b84dbcaafe06edaca1c331ea96bf3",
    "bridge/results/" + ATTEMPT + ".json": "82d1431e109dcdbc7142915dc6dbd9608b3b0ade08cfc3fda7b651893b35d82d",
    "seed_6101/round_10/turn_00/request.json": "45222fe1387c51bad3dd9ab81aaef4f8d56b2f8c8ddd743866e67c734bd090b7",
    "seed_6101/round_10/turn_00/attempt_00.started.json": "a8696dbd4b46ac2e71b7ee422bdfe513cb12139c69b1e19635720c32c771319d",
    "seed_6101/round_10/turn_00/attempt_00.terminal.json": "56b05a4c8d87d60b75494d38fa17303038d81b988c25dd4200a275cd59c1dc3b",
    "seed_6101/round_10/source.json": "042db5cd51c244218b5f54246bdd9a030f8bdf0af25e9f9312137df04c43c867",
    "seed_6101/round_10/transition.json": "b777346763d79eb9d030ec4d1cf8c34d88f3faaf92867e99a359bedf5eed9dff",
    "seed_6101/round_09/memory.json": "552eb30b181dc694331017e081c4cba62667ceb5b16585da0f143cf6fe69c5fe",
    "workers/seed_6101.json": "ed2247f926bf73ecb8ca165b655933743aeaf520d2a8c44d828fb21dcb35d4d3",
    "seed_6101/pause.request": "26fa5d44a24da9580923614c984ad64fdbce7100ff3a49e226771f949cfecca9",
    "provider_model_lock.json": "950c0fd7b1d3135305a6fa7024419205130d90d8962b46b7fbeed5d050fa5041",
}
RECOVERY_BUDGET = {"maximum_logical_requests": 1, "maximum_physical_attempts": 0}
BUDGET = {"first_logical_requests": 1, **RECOVERY_BUDGET}


def _no_provider(*args, **kwargs):
    raise TransportStop("an archived final selection has no provider transport")


def _require_no_bridge(run):
    if any(path.is_file() for folder in ("outbox", "claims", "results")
           for path in (run / "bridge" / folder).glob("*.json")):
        raise PermissionError("a zero-POST archival import may not have any bridge attempts")


def verify_mac_archive(spool):
    """Check the settled local capture and ACK without opening credential files."""
    attempt = spool / "attempts" / ATTEMPT
    job = spool / "jobs" / f"{ATTEMPT}.json"
    result = spool / "results" / f"{ATTEMPT}.json"
    ack = spool / "delivered" / f"{ATTEMPT}.json"
    capture, raw, progress = (attempt / name for name in (
        "provider_capture_receipt.json", "raw_provider_response.json", "provider_progress.json",
    ))
    expected = {
        job: ARCHIVE_FILES["bridge/outbox/" + ATTEMPT + ".json"],
        result: ARCHIVE_FILES["bridge/results/" + ATTEMPT + ".json"],
        ack: ACK, capture: CAPTURE_FILE, raw: RAW_FILE, progress: PROGRESS,
    }
    if (any(file_digest(path) != sha for path, sha in expected.items())
            or read(ack) != {"attempt_id": ATTEMPT, "result_sha256": RESULT}
            or digest(read(capture)) != CAPTURE or digest(read(raw)) != RAW
            or validate_result(read(result), read(job))["raw_response"] != read(raw)):
        raise PermissionError("the local complete capture or settled GPFS delivery acknowledgement changed")


def _extract_final(raw, transport, lock):
    if (type(raw) is not dict or digest(raw) != RAW or raw.get("truncation") != "disabled" or (raw.get("id"), raw.get("status"), raw.get("model"), raw.get("error"),
            raw.get("incomplete_details")) != (RESPONSE, "completed", "gpt-6.2-astra", None, None)):
        raise PermissionError("the exact canonical provider completion changed")
    items = raw.get("output")
    if type(items) is not list or len(items) != 9 or any(type(item) is not dict for item in items):
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
    if worker.get("status") != "failed" or type(pid) is not int or pid <= 0:
        raise PermissionError("the previous R10 worker is not in a sealed failed state")
    try:
        arguments = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, ProcessLookupError):
        arguments = ()
    if (str(old).encode() in arguments
            or set(active.glob("turn_*")) != {turn}
            or {p.name for p in turn.glob("attempt_*")} != {"attempt_00.started.json", "attempt_00.terminal.json"}
            or any((active / name).exists() for name in ("selection.json", "batch_freeze.json", "memory.json"))
            or (turn / "decision.json").exists() or (directory / f"round_{ROUND + 1:02d}").exists()
            or any(not (old / name).is_file() or file_digest(old / name) != expected
                   for name, expected in ARCHIVE_FILES.items())):
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
    if (start != 9 or read(directory / "boundary.json") != {"formal_run": scope, **descriptor}
            or digest(asdict(state)) != descriptor["state_sha256"] or digest(memory) != descriptor["memory_sha256"]):
        raise PermissionError("the signed prior L621 boundary changed")
    prefix = {"initial_boundary": file_digest(directory / "boundary.json")}
    split = _split(old_release)
    for index in range(start, ROUND):
        current = directory / f"round_{index:02d}"
        following = directory / f"round_{index + 1:02d}"
        state, memory = seed_runner._successor(current, state, following, scope, previous.transport, previous, split)
        prefix[f"transition_{index + 1:02d}"] = file_digest(following / "transition.json")
    if (state.round != ROUND or state.checkpoint != CHECKPOINT or digest(memory) != MEMORY
            or asdict(state) != read(directory / f"round_{ROUND:02d}" / "source.json")):
        raise PermissionError("the exact R9 prefix did not close at the R10 boundary")
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
        raise PermissionError("the exact archived final is detached from its paid R10 request and stopped receipt")
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
    _require_fixed_reconstruction()
    with _archival_lock(old):
        directory, turn = _fenced(old)
        previous, approval = _archived(old, old_release, OLD)
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
    for attr in ("audited_future_transport_replay", "sealed_relay_retry_seconds",
                 "replay_eligible_from_round", "full_imported_boundary_round"):
        profile.pop(attr, None)
    sources = {key: file_digest(root / key) for key in previous.source_sha256}
    if sources != previous.source_sha256:
        raise PermissionError("the local import cannot change existing signed code or the standard A7 prompt")
    path = (STUDY / "scripts" / f"{MODULE}.py").as_posix()
    sources[path] = file_digest(root / path)
    custody = {"provider_post": False, "attempt_id": ATTEMPT, "request_id": REQUEST, "payload_sha256": PAYLOAD,
               "job_sha256": JOB, "result_sha256": RESULT, "raw_response_sha256": RAW, "response_id": RESPONSE,
               "reply_text_sha256": TEXT, "plan_sha256": PLAN, "ordered_candidate_ids_sha256": ORDER,
               "ignored_placeholder_item_sha256": PLACEHOLDER, "explicit_final_item_sha256": FINAL,
               "archived_usage": archived.usage, "archived_provider_request_id": PROVIDER_REQUEST,
               "mac_capture_sha256": CAPTURE, "mac_capture_file_sha256": CAPTURE_FILE, "mac_raw_file_sha256": RAW_FILE,
               "mac_ack_file_sha256": ACK, "mac_progress_file_sha256": PROGRESS,
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
    """Read the owned immutable release and return the signed zero-POST L685 boundary proof."""
    contract, boundary, _, original, archived = construct(root, old_release, old, run)
    approval = verify_authorization(contract, read(run / "authorization.json"))
    if ((contract.digest, digest(approval)) != (approved_contract_sha256, approved_authorization_sha256)
            or read(run / "formal_contract.json") != contract.record()):
        raise PermissionError("the cached L685 contract or authorization differs from its signed identity")
    _require_no_bridge(run)
    worker = read(run / f"workers/seed_{SEED}.json")
    if (worker.get("status"), worker.get("completed_rounds"), worker.get("active_labels")) != ("complete", ROUND + 1, TARGET):
        raise PermissionError("the one-batch archival worker has not completed the exact L685 fit")
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
        raise PermissionError("the local archival custody and isolated R10 boundary changed")
    freeze = seed_runner._verify_selection(active, boundary[0], scope, contract.transport, contract.batch_size, stage=stage)
    if (freeze["plan"] != _verified_plan(stage, archived) or freeze["accepted_logical_index"] != 0
            or freeze["accepted_attempt"] != 0 or len(set(freeze["batch_ids"])) != 32):
        raise PermissionError("the sealed L685 selection is not the exact ordered archived model decision")
    state, memory = seed_runner._successor(active, boundary[0], following, scope, contract.transport, contract, _split(root))
    if (state.seed, state.round, len(state.provenance["expected_observed"])) != (SEED, ROUND + 1, TARGET):
        raise PermissionError("the fitted state does not close the one-batch L685 contract")
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
            raise TransportStop("only the exact archived R10 result may fill the local attempt")
        return reply

    gpu = contract.gpu_slots[0]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with scheduler._lock(old / "coordinator.lock", busy="the old R10 coordinator must remain retired"):
        scheduler._assert_no_old_workers(old, (SEED,))
        _fenced(old)
        with scheduler._lock(run / "coordinator.lock", busy="the local R10 coordinator is already active"):
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
