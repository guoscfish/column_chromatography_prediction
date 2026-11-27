"""Continue 2357 from a verified R11 targeted fit using the normal A7 prompt and budget."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import fcntl
from functools import partial
import os
from pathlib import Path
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import parse_response
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.runner import REPAIR_INSTRUCTIONS, REPLAN_INSTRUCTIONS
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    SEALED_RELAY_RETRY_SECONDS, _link_replay_attempt, make_request, terminal_record,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import (
    ATTEMPT_ID, BridgePending, validate_result, verify_job,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import OFFLINE, ROOT, STUDY, verify_authorization
from . import run_v13_2357_r11_targeted_repair as repair
from .run_v13_bounded_sse_recovery import RunIdentity, _archived
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend


SEED, START, TARGET = 2357, 12, 1485
MODULE = "run_v13_2357_r12_continuation"
DISPATCHER = "dispatch_v13_2357_r12_continuation"
RUN_NAME = f"{repair.relay.bounded.PREFIX}-2357-r12-continuation-01"
REPAIR_RELEASE = "ipv4-v13-2357-r11-targeted-repair-01-d8bcc642e20a55071026d3ad7679e9ba719e992a6605ed88a5ddd9ac7ec757c5"
REPAIRED = RunIdentity(
    "-2357-r11-targeted-repair-01", "2e2c2919fc76305aa08f973c7a2a253089f19afbb44519d30afadd30427790dc",
    "2b20a6e2e1e097519fe93a260c1e18f0650f3383f754111b269c5002e8495d88",
)
REPAIR_ATTEMPT = "ef2405d0983b48008f5ac8bbbd26bc99"
REPAIR_AUTHORITATIVE_MESSAGES = "ff4082b6be7f1e7f8a5562b28e954b244c80b2bbe2cd39f2a30971979d0072d0"
REPAIRED_CHECKPOINT = "484852bdaef0412c43736a70622d99c11b8123c8b75046f8ed6465d9bcda55bc"
REPAIRED_MEMORY = 'f522e00cd1f56f718057b3781fc806798e31b586ea9ceb1bc9fcacedcf9ddfc5'
REPAIRED_FREEZE = "a8389ac86daf6808ea9f30b60fd9f331098a3c13fa1a4556c9a678533088c62c"
REPAIR_ACK = "87761bc01f862763b44188fa27ce5c5d690b6401c0619ca29ceb58f5c1feca79"
BUDGET = {"first_logical_requests": 24, "maximum_logical_requests": 72, "maximum_physical_attempts": 144}
BOUNDARY_BUDGET = {"maximum_logical_requests": 3, "maximum_physical_attempts": 2}
SOURCE_BASE = {
    seed_runner.PROMPT_PATH.as_posix(): "d656796e97dee76bd12bb7ce94bc3056d042e858e417759eb4108710744a594f",
    (OFFLINE / "pure_llm_v3/transport_manager.py").as_posix(): "25bb9cd757b1d14153bf1f758427d195a4a2299a9a9f840dd7619c783e76c739",
    (STUDY / "a7_gpu/seed_runner.py").as_posix(): "19122c372294d5dd0f08d2ba4b1388b027ac1bacf65d84004c5eacf5b498e9f6",
}
SEED_RUNNER = (STUDY / "a7_gpu/seed_runner.py").as_posix()
RUNNER_PATCH = "a1560f9dd386ea100db485edb155f59e5c421c13cd2828758b8303ca517a53e5"
REPAIR_SOURCES = dict(zip((path.as_posix() for path in repair.SOURCES), (
    "df955af2a16d3112df2841f6e12fed1f55974ed4c4f37e109ccf89db532ec75a",
    "ecf9857d6ea95c290b442e051e5c3b0bb06cf84189dfa1fe32947a1fe889c9d7",
), strict=True))
SOURCES = tuple(STUDY / "scripts" / f"{name}.py" for name in (MODULE, DISPATCHER))
STAGE_FIELDS = (
    "checkpoint_sha256", "packet_sha256", "wire_packet_sha256", "base_messages_sha256",
    "authoritative_base_messages_sha256", "prompt_sha256", "mapping_sha256", "split_sha256", "state_sha256",
)
R12_REQUEST = {'checkpoint_sha256': '484852bdaef0412c43736a70622d99c11b8123c8b75046f8ed6465d9bcda55bc',
 'packet_sha256': '6b9acd4c0d2f0ac2db310c56754a4a3c816fb0d9c1c9922dd733573b9733bc95',
 'wire_packet_sha256': '9fce2bd9479187785090423563b211b1c39c1bb5eeace3fb4c1ff4ee105c51c8',
 'base_messages_sha256': 'e9f494ddd93b45066f3e4b4673347ef23561ced9a2abdac0339d3f52aa03e956',
 'authoritative_base_messages_sha256': '5756456d96237fd221d929cda5112e9067bcc30290ec26a53335d7b97bba7c3d',
 'prompt_sha256': '0872b8ddd262b319d9a1463ca0696fa28a5afbd51da8cd3ff477729db6b08d40',
 'mapping_sha256': '57d7e0fdad33dda05f1235e17276e0c62d7ce0ab316bd8c0220ba39b63816d42',
 'split_sha256': '360d81a9e239d7bbf6e2f89f5ce7db7e119f0783f5e134df122ac08a6fc1f1e7',
 'state_sha256': 'caed7c6ce8a7481825b371655f62d3aa23799742876f5f16dfce6942e8451c51',
 'payload_sha256': '1465731406b9cd73f7cfe3ad6c9af31ab6230556b24667c7211b550954041086'}
HANDOFF_SHA256 = '2ead72b6a2097bc5667c4715181aed3fb1841aa7e49769fb4334a27eff008fdf'
CORRECTION_STATES = {"rejected", "malformed_complete_response", "patch_rejected", "model_requested_replan"}


def request_identity(request):
    return {**{field: request["binding"][field] for field in STAGE_FIELDS},
            "payload_sha256": request["payload_sha256"]}


def check_first_round(request, identity):
    """Bind all R12 logical requests to its independently compiled base messages."""
    binding = request["binding"]
    index, messages = request["logical_index"], request["payload"]["input"]
    if ((binding["seed"], binding["round"]) != (SEED, START)
            or type(index) is not int or not 0 <= index < 3
            or "fixed_reference_repair" in binding
            or any(binding[field] != identity[field] for field in STAGE_FIELDS)
            or digest(messages[:2]) != identity["base_messages_sha256"]):
        raise PermissionError("the request differs from the approved normal R12 scientific context")
    if index == 0:
        if len(messages) != 2 or request["payload_sha256"] != identity["payload_sha256"]:
            raise PermissionError("the first R12 request no longer matches its signed full payload")
        return
    if len(messages) != 3 or messages[2]["role"] != "user":
        raise PermissionError("an R12 correction requires its dedicated user message")
    text = messages[2]["content"]
    for instruction, mode in ((REPAIR_INSTRUCTIONS, "patch"), (REPLAN_INSTRUCTIONS, "full_replan")):
        prefix = instruction + "\n"
        if text.startswith(prefix):
            content = parse_response(text[len(prefix):])
            if type(content) is dict and content.get("mode") == mode:
                return
            break
    raise PermissionError("R12 correction is outside the normal repair protocol")


def _correction_ready(round_dir, request):
    """Require official rejection and a complete previous logical result before a correction."""
    index = request["logical_index"]
    if not index:
        return
    turn = round_dir / f"turn_{index - 1:02d}"
    prior_request = read(turn / "request.json")
    if (read(turn / "decision.json").get("state") not in CORRECTION_STATES
            or prior_request["binding"] != request["binding"] or prior_request["logical_index"] != index - 1
            or prior_request["payload"]["input"][:2] != request["payload"]["input"][:2]
            or (round_dir / "batch_freeze.json").exists()):
        raise PermissionError("an R12 correction must follow the preceding rejected decision")
    completed = [item for path in turn.glob("attempt_*.terminal.json")
                 if (item := read(path)).get("status") == "completed"]
    if len(completed) != 1 or completed[0].get("request_id") != prior_request["request_id"]:
        raise PermissionError("an R12 correction must follow one complete preceding response")


def _check_second_slot(round_dir, request, started):
    if started["attempt"] != 1:
        return
    turn = round_dir / f"turn_{request['logical_index']:02d}"
    sha = digest(read(turn / "attempt_00.replay_authorization.json"))
    replay = terminal_record(turn, 0, replay_authorization_sha256=sha)
    if (replay.get("sealed_relay_retry_seconds") != SEALED_RELAY_RETRY_SECONDS
            or replay.get("request_id") != request["request_id"]):
        raise PermissionError("R12 physical retry requires its own sealed relay proof")
    _link_replay_attempt(turn, started, replay, allow_create=False)


@contextmanager
def _archival_lock(run):
    """Refuse an active old coordinator without modifying its archived lock file."""
    with (run / "coordinator.lock").open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PermissionError("an archived coordinator still holds its run lock") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _normal_sources(root, normal, targeted):
    if any(normal.source_sha256.get(name) != sha for name, sha in SOURCE_BASE.items()):
        raise PermissionError("the normal R11 runtime does not match its reviewed prompt, parser and seed runner")
    if any(targeted.source_sha256.get(name) != sha for name, sha in REPAIR_SOURCES.items()):
        raise PermissionError("the targeted R11 verifier does not match its separately signed source")
    expected = {**normal.source_sha256, **REPAIR_SOURCES, SEED_RUNNER: RUNNER_PATCH}
    actual = {name: file_digest(root / name) for name in expected}
    if actual != expected:
        raise PermissionError("the normal continuation changed an inherited source beyond the reviewed budget guard")
    actual.update({path.as_posix(): file_digest(root / path) for path in SOURCES})
    return actual


def _repair_bridge(run, active, contract, approval, expected, original):
    """Read the only signed targeted repair and verify agreement through its complete terminal."""
    turn = active / "turn_00"
    terminal = terminal_record(turn, 0)
    attempt = terminal["attempt_id"]
    if not ATTEMPT_ID.fullmatch(attempt) or attempt != REPAIR_ATTEMPT:
        raise PermissionError("the targeted repair needs its own physical attempt")
    name = f"{attempt}.json"
    if ({path.name for path in active.glob("turn_*")} != {"turn_00"}
            or {path.name for path in turn.glob("attempt_*")}
            != {"attempt_00.started.json", "attempt_00.terminal.json"}
            or any({path.name for path in (run / "bridge" / folder).glob("*.json")} != {name}
                   for folder in ("outbox", "claims", "results"))):
        raise PermissionError("the targeted scope must have exactly one bridge attempt")
    job_path, result_path = (run / "bridge" / area / name for area in ("outbox", "results"))
    job = read(job_path)
    request, config, started = verify_job(job, contract_sha256=contract.digest,
                                          authorization_sha256=digest(approval),
                                          wire_representation=contract.profile["wire_representation"])
    result = validate_result(read(result_path), job)
    reply = result["outcome"].get("reply") or {}
    raw = result["raw_response"]
    model_lock = contract.predecessor["model_identity"]
    repair.verify_patch_request(request, original, contract.profile["fixed_reference_repair"], attempt=started["attempt"])
    if (request != expected or read(turn / "request.json") != request or config != contract.transport
            or read(turn / "attempt_00.started.json") != started
            or terminal.get("status") != "completed" or terminal.get("provider_status") != "completed"
            or any(terminal.get(key) != value for key, value in started.items())
            or terminal.get("bridge_job_sha256") != job["job_sha256"]
            or terminal.get("bridge_result_sha256") != result["result_sha256"]
            or result["outcome"]["status"] != "completed" or result["provider_dispatched"] is not True
            or type(raw) is not dict or raw.get("status") != "completed"
            or result["model_identity"] != model_lock or read(run / "provider_model_lock.json") != model_lock
            or raw.get("reasoning", {}).get("effort") != config.reasoning_effort
            or not reply.get("provider_request_id") or reply["served_model"] != config.served_model
            or terminal.get("answer_sha256") != digest(reply["text"])
            or any(terminal.get(field) != reply[field] for field in (
                "text", "response_id", "served_model", "usage", "provider_request_id",
            ))):
        raise PermissionError("the R11 repair does not have one exact delivered complete provider response")
    return reply, {
        "attempt_id": attempt, "request_id": request["request_id"], "payload_sha256": request["payload_sha256"],
        "job_sha256": job["job_sha256"], "result_sha256": result["result_sha256"], "raw_sha256": digest(raw),
        "job_file_sha256": file_digest(job_path), "result_file_sha256": file_digest(result_path),
        "answer_sha256": terminal["answer_sha256"], "response_id": reply["response_id"],
        "provider_request_id": reply["provider_request_id"], "usage": reply["usage"], "provider_post": True,
    }


def verify_handoff(old_root, old, targeted_root, targeted_run):
    """Read the current fitted R11 archive and its original answered decision under both locks."""
    if (targeted_root.name != REPAIR_RELEASE or targeted_run.name != REPAIRED.name
            or old.name != repair.PARENT.name or targeted_run.parent != old.parent):
        raise PermissionError("use the exact retired normal scope and signed R11 targeted release")
    targeted, approval = _archived(targeted_run, targeted_root, REPAIRED)
    if Path(targeted.predecessor["previous_release_directory"]) != old_root:
        raise PermissionError("the targeted scope was constructed from another ordinary release")
    reconstructed, boundary, cache, original, plan = repair.construct(targeted_root, old_root, old, targeted_run)
    if reconstructed.record() != targeted.record():
        raise PermissionError("the signed targeted contract no longer reconstructs")
    worker_path = targeted_run / f"workers/seed_{SEED}.json"
    worker = read(worker_path)
    if (worker.get("status"), worker.get("completed_rounds"), worker.get("active_labels"), worker.get("seed")) != (
        "complete", START, 717, SEED,
    ):
        raise PermissionError("R11 must first complete and seal the exact CUDA L717 state")
    directory = seed_runner.seed_directory(targeted_run, SEED)
    active, following = (seed_runner.round_directory(directory, index) for index in (START - 1, START))
    turn = active / "turn_00"
    if set(path.name for path in directory.glob("round_*")) != {active.name, following.name} or any(
        (following / name).exists() for name in ("turn_00", "turn_01", "turn_02", "batch_freeze.json", "selection.json")
    ):
        raise PermissionError("the one-batch targeted scope contains an unexpected later decision")
    metadata = targeted.profile["fixed_reference_repair"]
    scope = seed_runner._scope(targeted, approval, SEED, False)
    user = original["payload"]["input"][1]
    stage = seed_runner.pin_boundary_user_message(
        seed_runner._stage(targeted_root, *boundary[:2], scope, targeted), user, digest(user),
    )
    # The internal authority hash was reconstructed differently when sealing the original wire input.
    # Restore only its independently verified archival value before comparing the signed request.
    rebuilt_authority = stage.binding["authoritative_base_messages_sha256"]
    stage = replace(stage, binding={**stage.binding, "authoritative_base_messages_sha256": REPAIR_AUTHORITATIVE_MESSAGES})
    expected = make_request(stage.binding, 0, stage.messages, targeted.transport)
    reply, bridge = _repair_bridge(targeted_run, active, targeted, approval, expected, original)
    projected = repair.project_reply(parse_response(reply["text"]), original=plan)
    freeze = seed_runner._verify_selection(active, boundary[0], scope, targeted.transport, targeted.batch_size, stage=stage)
    if (read(targeted_run / "fixed_reference_repair.json") != metadata
            or read(targeted_run / "canonical_repair_plan.json") != plan
            or read(directory / "boundary.json") != {"formal_run": scope, **targeted.profile["resume_boundaries"][str(SEED)]}
            or read(active / "source.json") != asdict(boundary[0]) or read(active / "boundary_memory.json") != boundary[1]
            or read(active / "boundary_user_message.json") != user
            or freeze["plan"] != projected or freeze["accepted_logical_index"] != 0 or freeze["accepted_attempt"] != 0
            or len(set(freeze["batch_ids"])) != 32 or digest(repair._ordered(projected)) != repair.ORDER):
        raise PermissionError("the projected archive and frozen 32 R11 selections disagree")
    state, memory = seed_runner._successor(active, boundary[0], following, scope, targeted.transport,
                                          targeted, TrainingBackend(targeted_root, cache, SEED))
    if ((state.seed, state.round, len(state.provenance["expected_observed"]), state.checkpoint)
            != (SEED, START, 717, REPAIRED_CHECKPOINT)
            or digest(memory) != REPAIRED_MEMORY
            or freeze["freeze_sha256"] != REPAIRED_FREEZE):
        raise PermissionError("the verified R11 fit does not furnish L717")
    fit = seed_runner._fit_directory(following)
    device = read(fit / "fit_device.json")
    if not device.get("cuda") or device.get("deterministic_algorithms") is not True:
        raise PermissionError("the exact next boundary must come from a verified CUDA fit")
    paths = (worker_path, targeted_run / "coordinator_generation.json", active / "memory.json", active / "selection.json",
             active / "authoritative_packet.json", turn / "response_projection.json", turn / "decision.json",
             turn / "attempt_00.terminal.json", following / "source.json", following / "transition.json",
             following / "authorized_feedback.json", fit / "fit_audit.json", fit / "validation_metrics.json")
    proof = {
        "contract_sha256": targeted.digest, "authorization_sha256": digest(approval), "bridge": bridge,
        "checkpoint_sha256": state.checkpoint, "state_sha256": digest(asdict(state)),
        "memory_sha256": digest(memory), "freeze_sha256": freeze["freeze_sha256"], "candidate_order_sha256": repair.ORDER,
        "fit_files": {path.relative_to(targeted_run).as_posix(): file_digest(path) for path in paths},
        "repair_authoritative_messages": {"archive": REPAIR_AUTHORITATIVE_MESSAGES, "reconstructed": rebuilt_authority},
    }
    return proof, (state, memory, START), cache, targeted


def construct(root, old_root, old, targeted_root, targeted_run, run, pins):
    if root in (old_root, targeted_root) or run.name != RUN_NAME or run.parent != old.parent:
        raise PermissionError("the normal continuation requires a distinct sibling run and release")
    handoff, boundary, cache, targeted = verify_handoff(old_root, old, targeted_root, targeted_run)
    if pins != handoff or digest(handoff) != HANDOFF_SHA256:
        raise PermissionError("the exact verified L717 handoff does not match the supplied pins")
    normal, approval = _archived(old, old_root, repair.PARENT)
    sources = _normal_sources(root, normal, targeted)
    if (normal.target_labels, normal.seed_ids, normal.transport.logical_replies, normal.transport.transport_attempts,
            normal.profile.get("sealed_relay_retry_seconds"), normal.profile.get("audited_future_transport_replay")) != (
        TARGET, (SEED,), 3, 2, SEALED_RELAY_RETRY_SECONDS, True,
    ) or "fixed_reference_repair" in normal.profile:
        raise PermissionError("the inherited route does not use the reviewed normal scientific protocol and budgets")
    profile = {**normal.profile, "resume_boundaries": {str(SEED): {"round": START,
        "state_sha256": handoff["state_sha256"], "memory_sha256": handoff["memory_sha256"]}},
        "full_imported_boundary_round": START, "replay_eligible_from_round": START}
    stage = seed_runner._stage(root, *boundary[:2], seed_runner._scope(normal, approval, SEED, False), normal)
    if (stage.binding["checkpoint_sha256"] != handoff["checkpoint_sha256"]
            or (stage.binding["seed"], stage.binding["round"]) != (SEED, START)
            or "fixed_reference_repair" in stage.binding):
        raise PermissionError("the normal R12 request did not reconstruct from the verified L717 anchor")
    first = request_identity(make_request(stage.binding, 0, stage.messages, normal.transport))
    if first != R12_REQUEST:
        raise PermissionError("the reconstructed normal R12 input differs from its independently reviewed payload")
    proof = {
        "dispatcher_module": DISPATCHER, "seed": SEED, "round": START,
        "previous_run_directory": str(targeted_run), "previous_release_directory": str(targeted_root),
        "previous_contract_sha256": targeted.digest, "previous_authorization_sha256": REPAIRED.approval,
        "normal_run_directory": str(old), "normal_release_directory": str(old_root),
        "normal_contract_sha256": normal.digest, "normal_authorization_sha256": digest(approval),
        "normal_seed_pause_file_sha256": repair.OLD_PAUSE, "r11_handoff": handoff,
        "r12_request": first,
        "checkpoint_sha256": boundary[0].checkpoint, "cache_directory": str(cache),
        "public_resource_receipt_sha256": normal.predecessor["public_resource_receipt_sha256"],
        "model_identity": normal.predecessor["model_identity"],
        "bridge_wait_seconds": normal.predecessor["bridge_wait_seconds"],
        "recovery_budget": BOUNDARY_BUDGET, "execution_budget": BUDGET,
    }
    return replace(normal, profile=profile, source_sha256=sources, run_directory=str(run), predecessor=proof), boundary, cache


def execute(root, run, contract, authorization, boundary, cache, *, resume):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    approval = verify_authorization(contract, authorization)
    with scheduler._lock(run / "coordinator.lock", busy="the normal R12 coordinator is already active"):
        scheduler._assert_no_old_workers(run, (SEED,))
        directory = seed_runner.seed_directory(run, SEED)
        if scheduler._pause_requested(run, SEED) and not resume:
            raise PermissionError("the normal R12 continuation is paused; use resume")
        once(run / "formal_contract.json", contract.record())
        once(run / "authorization.json", approval)
        once(run / "provider_model_lock.json", contract.predecessor["model_identity"])
        if resume:
            (run / "pause.request").unlink(missing_ok=True)
            (directory / "pause.request").unlink(missing_ok=True)
        scheduler._atomic_status(run / "coordinator_generation.json", {"id": uuid4().hex, "pid": os.getpid()})
        call, options = scheduler._provider_wiring(run, contract, approval, None, SEED)
        bridge = options["managed_attempt"]

        def managed(request, config, started):
            binding = request["binding"]
            if (config != contract.transport or binding["seed"] != SEED or not START <= binding["round"] < 36
                    or "fixed_reference_repair" in binding
                    or started["attempt_id"] in {repair.ATTEMPT, contract.predecessor["r11_handoff"]["bridge"]["attempt_id"]}):
                raise PermissionError("the bridge job is outside the normal R12–R35 scope")
            if binding["round"] == START:
                check_first_round(request, contract.predecessor["r12_request"])
                active = seed_runner.round_directory(directory, START)
                _correction_ready(active, request)
                _check_second_slot(active, request, started)
            return bridge(request, config, started)

        gpu = contract.gpu_slots[0]
        worker = run / f"workers/seed_{SEED}.json"
        status = {"seed": SEED, "pid": os.getpid(), "gpu": gpu}
        scheduler._atomic_status(worker, {**status, "status": "initializing"})
        try:
            progress = seed_runner.run_seed(
                root=root, run_directory=run, contract=contract, authorization=approval, seed=SEED,
                backend=TrainingBackend(root, cache, SEED), call=call, resume_boundary=boundary,
                transport_options={"managed_attempt": managed}, fit_slot=partial(scheduler.gpu_slot, run, SEED, gpu, contract.lease_root),
            )
            scheduler._atomic_status(worker, {**status, **asdict(progress)})
            return progress
        except Exception as error:
            phase = "paused" if scheduler._pause_requested(run, SEED) else (
                "bridge_pending" if isinstance(error, BridgePending) else "failed")
            scheduler._atomic_status(worker, {**status, "status": phase, "error_type": type(error).__name__, "reason": str(error)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect-handoff", "contract", "run", "resume"))
    for name in ("old-release", "old-run", "targeted-release", "targeted-run"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--handoff-pins", type=Path)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    old_root, old, targeted_root, targeted_run = (value.resolve(strict=True) for value in (
        args.old_release, args.old_run, args.targeted_release, args.targeted_run,
    ))
    with _archival_lock(old), _archival_lock(targeted_run):
        if args.action == "inspect-handoff":
            handoff, boundary, _, _ = verify_handoff(old_root, old, targeted_root, targeted_run)
            normal, approval = _archived(old, old_root, repair.PARENT)
            stage = seed_runner._stage(old_root, *boundary[:2], seed_runner._scope(normal, approval, SEED, False), normal)
            print(dumps({"handoff": handoff, "r12_request": request_identity(make_request(stage.binding, 0, stage.messages, normal.transport))}))
            return
        if args.run_dir is None or args.handoff_pins is None:
            parser.error("contract, run and resume require the isolated run path and verified handoff pins")
        run = args.run_dir.resolve()
        contract, boundary, cache = construct(ROOT, old_root, old, targeted_root, targeted_run, run, read(args.handoff_pins))
        if args.action == "contract":
            print(dumps({"contract_sha256": contract.digest, **contract.record()}))
            return
        if args.authorization is None:
            parser.error("run and resume require the exact continuation authorization")
        result = execute(ROOT, run, contract, read(args.authorization), boundary, cache, resume=args.action == "resume")
        print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
