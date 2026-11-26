"""Continue 887 from a verified R32 targeted fit using the normal A7 prompt and budget."""

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
from . import run_v13_887_r19_cached as cached
from . import run_v13_887_r32_targeted_repair as repair
from . import run_v13_lane_continuation as continuation
from .run_v13_bounded_sse_recovery import RunIdentity, _archived


SEED, START, TARGET = 887, 33, 1485
MODULE = "run_v13_887_r33_continuation"
DISPATCHER = "dispatch_v13_887_r33_continuation"
RUN_NAME = f"{repair.relay.bounded.PREFIX}-887-r33-continuation-01"
LANE = repair.SimpleNamespace(seed=SEED, round=START - 1)
REPAIR_RELEASE = "ipv4-v13-887-r32-targeted-repair-01-bc1b3178866e02fb00e5844b1bdc8a8e8690dd0f8120f01d7d4b510004b4d882"
REPAIRED = RunIdentity(
    "-887-r32-targeted-repair-01", "653ec5d2732e72bfc317c084c98671890030ffda0c361cd52570fd1ef22bde50",
    "7a429145f5aaf60783e82fad5ca18975d3b53fd180066cd25946b8ebfe95956a",
)
REPAIR_ATTEMPT = "6527fb2228284a8cb84fcf8b5084f0d0"
REPAIRED_CHECKPOINT = "98f69f161819ef20d0921b33b5fa60403de90595057535f71d9fdfb9e134a7f0"
REPAIRED_MEMORY = "eb0af3e7de942ef0bd099a3c9597efca22a734ff5652fedb357c3b754c3886bd"
REPAIRED_FREEZE = "b74f38a12bbd9e12e160d860382d12ae2cfbbf82c09249ee15380d6b6a7afb9a"
REPAIR_ACK = "18cb994aa1f958ba25b9ce998ed2bcf77a406e80e064f4d1de35bba47b08ab8c"
BUDGET = {"first_logical_requests": 3, "maximum_logical_requests": 9, "maximum_physical_attempts": 18}
BOUNDARY_BUDGET = {"maximum_logical_requests": 3, "maximum_physical_attempts": 2}
SOURCE_BASE = {
    seed_runner.PROMPT_PATH.as_posix(): "d656796e97dee76bd12bb7ce94bc3056d042e858e417759eb4108710744a594f",
    (OFFLINE / "pure_llm_v3/transport_manager.py").as_posix(): "2cdc0990f81012f040a73f8d5b7226fe1d35b934f477dc1654ee071126193a3c",
    (STUDY / "a7_gpu/seed_runner.py").as_posix(): "19122c372294d5dd0f08d2ba4b1388b027ac1bacf65d84004c5eacf5b498e9f6",
}
SEED_RUNNER = (STUDY / "a7_gpu/seed_runner.py").as_posix()
RUNNER_PATCH = "417f41255d53d7794abcdc98c0a9258141d01c076df845bfa877013c84de1792"
REPAIR_SOURCES = dict(zip((path.as_posix() for path in repair.SOURCES), (
    "c09912fd0433c9bfcec8fb99afdf4c353153fa7eb029750393f5a06c5a9fd219",
    "15fb8912a486f30b910bed622c692a51e0850adea9897f791d3993ee74b569c9",
), strict=True))
SOURCES = tuple(STUDY / "scripts" / f"{name}.py" for name in (MODULE, DISPATCHER))
STAGE_FIELDS = (
    "checkpoint_sha256", "packet_sha256", "wire_packet_sha256", "base_messages_sha256",
    "authoritative_base_messages_sha256", "prompt_sha256", "mapping_sha256", "split_sha256", "state_sha256",
)
R33_REQUEST = {'payload_sha256': 'f440da696e0571f647ebf15fd2fd9d622c03a77d4bb13a8eb8b861dbb3ada59f',
 'checkpoint_sha256': '98f69f161819ef20d0921b33b5fa60403de90595057535f71d9fdfb9e134a7f0',
 'packet_sha256': 'ab4d0f0c88c0801726ce9d230514748d69cd520f61b90f9d627b6e760acb1063',
 'wire_packet_sha256': '17e2a5582f3d993ab6b8b4f23253accf98fcc96fcc8fe4370c00b43205a0d76f',
 'base_messages_sha256': '8260d257c5556878c9d2915630862e4141f8a71c7b8c950b942e6142c3d78b6b',
 'authoritative_base_messages_sha256': 'b4a81d350fd611ee076adc55a18412d6470ee3d4eac2250268efaee307e82dd6',
 'prompt_sha256': '0872b8ddd262b319d9a1463ca0696fa28a5afbd51da8cd3ff477729db6b08d40',
 'mapping_sha256': '5e004a11706553497a363ae77a32cea7da424e2c24124144b950154c698b033f',
 'split_sha256': '0413fb5c356dc14b1d2c056379ef09f1925835e64fa41fa86dc12aa8703a1807',
 'state_sha256': '6ac9d9e9b4e46914e6d0b6951b0570992855af5dc94bd6a71880bec82aa823d6'}
CORRECTION_STATES = {"rejected", "malformed_complete_response", "patch_rejected", "model_requested_replan"}


def request_identity(request):
    return {**{field: request["binding"][field] for field in STAGE_FIELDS},
            "payload_sha256": request["payload_sha256"]}


def check_first_round(request, identity):
    """Bind all R33 logical requests to its independently compiled base messages."""
    binding = request["binding"]
    index, messages = request["logical_index"], request["payload"]["input"]
    if ((binding["seed"], binding["round"]) != (SEED, START)
            or type(index) is not int or not 0 <= index < 3
            or "fixed_reference_repair" in binding
            or any(binding[field] != identity[field] for field in STAGE_FIELDS)
            or digest(messages[:2]) != identity["base_messages_sha256"]):
        raise PermissionError("the request differs from the approved normal R33 scientific context")
    if index == 0:
        if len(messages) != 2 or request["payload_sha256"] != identity["payload_sha256"]:
            raise PermissionError("the first R33 request no longer matches its signed full payload")
        return
    if len(messages) != 3 or messages[2]["role"] != "user":
        raise PermissionError("an R33 correction requires its dedicated user message")
    text = messages[2]["content"]
    for instruction, mode in ((REPAIR_INSTRUCTIONS, "patch"), (REPLAN_INSTRUCTIONS, "full_replan")):
        prefix = instruction + "\n"
        if text.startswith(prefix):
            content = parse_response(text[len(prefix):])
            if type(content) is dict and content.get("mode") == mode:
                return
            break
    raise PermissionError("R33 correction is outside the normal repair protocol")


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
        raise PermissionError("an R33 correction must follow the preceding rejected decision")
    completed = [item for path in turn.glob("attempt_*.terminal.json")
                 if (item := read(path)).get("status") == "completed"]
    if len(completed) != 1 or completed[0].get("request_id") != prior_request["request_id"]:
        raise PermissionError("an R33 correction must follow one complete preceding response")


def _check_second_slot(round_dir, request, started):
    if started["attempt"] != 1:
        return
    turn = round_dir / f"turn_{request['logical_index']:02d}"
    sha = digest(read(turn / "attempt_00.replay_authorization.json"))
    replay = terminal_record(turn, 0, replay_authorization_sha256=sha)
    if (replay.get("sealed_relay_retry_seconds") != SEALED_RELAY_RETRY_SECONDS
            or replay.get("request_id") != request["request_id"]):
        raise PermissionError("R33 physical retry requires its own sealed relay proof")
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
        raise PermissionError("the normal R32 runtime does not match its reviewed prompt, parser and seed runner")
    if any(targeted.source_sha256.get(name) != sha for name, sha in REPAIR_SOURCES.items()):
        raise PermissionError("the targeted R32 verifier does not match its separately signed source")
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
        raise PermissionError("the R32 repair does not have one exact delivered complete provider response")
    return reply, {
        "attempt_id": attempt, "request_id": request["request_id"], "payload_sha256": request["payload_sha256"],
        "job_sha256": job["job_sha256"], "result_sha256": result["result_sha256"], "raw_sha256": digest(raw),
        "job_file_sha256": file_digest(job_path), "result_file_sha256": file_digest(result_path),
        "answer_sha256": terminal["answer_sha256"], "response_id": reply["response_id"],
        "provider_request_id": reply["provider_request_id"], "usage": reply["usage"], "provider_post": True,
    }


def verify_handoff(old_root, old, targeted_root, targeted_run):
    """Read the current fitted R32 archive and its original answered decision under both locks."""
    if (targeted_root.name != REPAIR_RELEASE or targeted_run.name != REPAIRED.name
            or old.name != repair.PARENT.name or targeted_run.parent != old.parent):
        raise PermissionError("use the exact retired normal scope and signed R32 targeted release")
    targeted, approval = _archived(targeted_run, targeted_root, REPAIRED)
    if Path(targeted.predecessor["previous_release_directory"]) != old_root:
        raise PermissionError("the targeted scope was constructed from another ordinary release")
    reconstructed, boundary, cache, original, plan = repair.construct(targeted_root, old_root, old, targeted_run)
    if reconstructed.record() != targeted.record():
        raise PermissionError("the signed targeted contract no longer reconstructs")
    worker_path = targeted_run / f"workers/seed_{SEED}.json"
    worker = read(worker_path)
    if (worker.get("status"), worker.get("completed_rounds"), worker.get("active_labels"), worker.get("seed")) != (
        "complete", START, 1389, SEED,
    ):
        raise PermissionError("R32 must first complete and seal the exact CUDA L1389 state")
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
        raise PermissionError("the projected archive and frozen 32 R32 selections disagree")
    state, memory = seed_runner._successor(active, boundary[0], following, scope, targeted.transport,
                                          targeted, cached._split(targeted_root))
    if ((state.seed, state.round, len(state.provenance["expected_observed"]), state.checkpoint)
            != (SEED, START, 1389, REPAIRED_CHECKPOINT)
            or digest(memory) != REPAIRED_MEMORY or freeze["freeze_sha256"] != REPAIRED_FREEZE):
        raise PermissionError("the verified R32 fit does not furnish L1389")
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
    }
    return proof, (state, memory, START), cache, targeted


def construct(root, old_root, old, targeted_root, targeted_run, run, pins):
    continuation._require_fixed_reconstruction(LANE)
    if root in (old_root, targeted_root) or run.name != RUN_NAME or run.parent != old.parent:
        raise PermissionError("the normal continuation requires a distinct sibling run and release")
    handoff, boundary, cache, targeted = verify_handoff(old_root, old, targeted_root, targeted_run)
    if pins != handoff:
        raise PermissionError("the exact verified L1389 handoff does not match the supplied pins")
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
        raise PermissionError("the normal R33 request did not reconstruct from the verified L1389 anchor")
    first = request_identity(make_request(stage.binding, 0, stage.messages, normal.transport))
    if first != R33_REQUEST:
        raise PermissionError("the reconstructed normal R33 input differs from its independently reviewed payload")
    proof = {
        "dispatcher_module": DISPATCHER, "seed": SEED, "round": START,
        "previous_run_directory": str(targeted_run), "previous_release_directory": str(targeted_root),
        "previous_contract_sha256": targeted.digest, "previous_authorization_sha256": REPAIRED.approval,
        "normal_run_directory": str(old), "normal_release_directory": str(old_root),
        "normal_contract_sha256": normal.digest, "normal_authorization_sha256": digest(approval),
        "normal_seed_pause_file_sha256": repair.OLD_PAUSE, "r32_handoff": handoff,
        "r33_request": first,
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
    with scheduler._lock(run / "coordinator.lock", busy="the normal R33 coordinator is already active"):
        scheduler._assert_no_old_workers(run, (SEED,))
        directory = seed_runner.seed_directory(run, SEED)
        if scheduler._pause_requested(run, SEED) and not resume:
            raise PermissionError("the normal R33 continuation is paused; use resume")
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
                    or started["attempt_id"] in {repair.ATTEMPT, contract.predecessor["r32_handoff"]["bridge"]["attempt_id"]}):
                raise PermissionError("the bridge job is outside the normal R33–R35 scope")
            if binding["round"] == START:
                check_first_round(request, contract.predecessor["r33_request"])
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
    continuation._require_fixed_reconstruction(LANE)
    with _archival_lock(old), _archival_lock(targeted_run):
        if args.action == "inspect-handoff":
            print(dumps(verify_handoff(old_root, old, targeted_root, targeted_run)[0]))
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
