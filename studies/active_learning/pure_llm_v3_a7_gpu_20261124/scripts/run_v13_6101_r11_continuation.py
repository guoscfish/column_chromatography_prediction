"""Continue 6101 from its verified zero-POST L685 fit with the reviewed phase parser."""

import argparse
from dataclasses import asdict, replace
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import parse_response
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.runner import REPAIR_INSTRUCTIONS, REPLAN_INSTRUCTIONS
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import SEALED_RELAY_RETRY_SECONDS, make_request
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import BridgePending
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import OFFLINE, ROOT, STUDY, verify_authorization
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import SHA256
from . import run_v13_6101_r10_cached as cached
from . import run_v13_6101_r9_continuation as normal
from .run_v13_bounded_sse_recovery import RunIdentity, _archived


SEED, START, TARGET = 6101, 11, 1485
MODULE = "run_v13_6101_r11_continuation"
DISPATCHER = "dispatch_v13_6101_r11_continuation"
RUN_NAME = f"{cached.PREFIX}-6101-r11-continuation-01"
CACHED = RunIdentity("-6101-r10-cached-01", "555ea880ed86c6e9347be1967874a9512c73aa8cd1142f90c343233828aa6f58",
                     "345ac4f06cb2940848743e053723780e6b31153002479b3607a936471864b45d")
CACHED_RELEASE = "ipv4-v13-6101-r10-cached-01-140259eb08a1e6caf27edbc26e51b5e63fb3d9e8b501f892f6bdc445fd4d204b"
CACHED_MODULE = (STUDY / "scripts" / f"{cached.MODULE}.py").as_posix()
CACHED_HELPER = "356e3f1a49922b9a07489a4ed061e573c2ffe937189c5f0b98fc9b98947420f8"
PARSER = (OFFLINE / "pure_llm_v3/transport_manager.py").as_posix()
OLD_PARSER = "25bb9cd757b1d14153bf1f758427d195a4a2299a9a9f840dd7619c783e76c739"
NEW_PARSER = "2cdc0990f81012f040a73f8d5b7226fe1d35b934f477dc1654ee071126193a3c"
RUNNER_PATCH = "8368fc31377202dd4fb451a069cdfd28bcc88e08ca5ba44963cf187e18fc93e0"
NORMAL_PROMPT = normal.SOURCE_BASE[seed_runner.PROMPT_PATH.as_posix()]
BUDGET = {"first_logical_requests": 25, "maximum_logical_requests": 75, "maximum_physical_attempts": 150}
BOUNDARY_BUDGET = {"maximum_logical_requests": 3, "maximum_physical_attempts": 2}
SOURCES = tuple(STUDY / "scripts" / f"{name}.py" for name in (MODULE, DISPATCHER))
HANDOFF = {
    "contract_sha256": CACHED.contract, "authorization_sha256": CACHED.approval,
    "checkpoint_sha256": "07214f81948e1edbb1a7500cfa9e4dbe4604504a136defcd33f460f8fb842239",
    "transition_file_sha256": "2490fc5d2537cc5ad92217d1f1d4cbedef5b70bbae03f951b231885738f95160",
    "source_file_sha256": "13d6d959557a4ecffc79737cdd5b07dcabff22880472cb3d5b1c0ad49346ece5",
    "state_sha256": "877489dcf354335b22f530309ca923c6ad38687d950e2ebab41d5a848124ac43",
    "scientific_memory_sha256": "958041f961972e6a274f25712d484724116026fbbdd930acf72e6559d415b146",
    "fit_audit_file_sha256": "f21f1774af7241b0098c1b5d40143847231c4dba10f6f4c63a13a22de11c0cf2",
    "freeze_sha256": "7385ece32aeb66d8e1228451f23a154ab71aabb25e5c8900b53184513153ebcc",
    "cached_selection_sha256": "c996df723e45ee1161fa57c1704fbeda0c1e8d8f4aeb07da8c9506778bcc9c5f",
    "local_terminal_sha256": "c942bc3bb66e934c7d8374729faef0c40ed6b83ed2d8246a8f49a213c34fd848",
    "local_attempt_id": "ac3f82ce61c44d8ca06f9bff8fcdde38",
    "archived_provider_response_id": cached.RESPONSE, "candidate_order_sha256": cached.ORDER, "provider_post": False,
}
REQUEST_FIELDS = {"payload_sha256", *normal.STAGE_FIELDS}


def validate_pins(pins):
    """Validate the fit and first-request identities in the independently verified pin record."""
    if type(pins) is not dict or set(pins) != {"r10_handoff", "r11_request", "cached_response_import_file_sha256"}:
        raise PermissionError("the handoff must contain the fit, custody and first request pins")
    handoff, first = pins["r10_handoff"], pins["r11_request"]
    if (type(handoff) is not dict or handoff != HANDOFF or handoff["provider_post"] is not False
            or type(first) is not dict or set(first) != REQUEST_FIELDS
            or not all(isinstance(x, str) and SHA256.fullmatch(x) for x in first.values())
            or first["checkpoint_sha256"] != handoff["checkpoint_sha256"]
            or not isinstance(pins["cached_response_import_file_sha256"], str)
            or not SHA256.fullmatch(pins["cached_response_import_file_sha256"])):
        raise PermissionError("the read-only R10 fit proof and ordinary R11 request pins are incomplete or changed")
    return handoff, first


def check_first_round(request, identity):
    binding = request["binding"]
    index, messages = request["logical_index"], request["payload"]["input"]
    if ((binding["seed"], binding["round"]) != (SEED, START)
            or type(index) is not int or not 0 <= index < 3 or "fixed_reference_repair" in binding
            or any(binding[field] != identity[field] for field in normal.STAGE_FIELDS)
            or digest(messages[:2]) != identity["base_messages_sha256"]):
        raise PermissionError("the request differs from the approved normal R11 scientific context")
    if index == 0:
        if len(messages) != 2 or request["payload_sha256"] != identity["payload_sha256"]:
            raise PermissionError("the first R11 request no longer matches the signed full payload")
        return
    if len(messages) != 3 or messages[2]["role"] != "user":
        raise PermissionError("an R11 correction requires a dedicated user message")
    text = messages[2]["content"]
    for instruction, mode in ((REPAIR_INSTRUCTIONS, "patch"), (REPLAN_INSTRUCTIONS, "full_replan")):
        prefix = instruction + "\n"
        if text.startswith(prefix):
            parsed = parse_response(text[len(prefix):])
            if type(parsed) is dict and parsed.get("mode") == mode:
                return
            break
    raise PermissionError("R11 correction is outside the normal repair protocol")


def _normal_sources(root, parent, imported):
    inherited = parent.source_sha256
    if (inherited.get(PARSER) != OLD_PARSER or inherited.get(normal.SEED_RUNNER) != normal.RUNNER_PATCH
            or inherited.get(seed_runner.PROMPT_PATH.as_posix()) != NORMAL_PROMPT
            or imported.source_sha256 != {**inherited, CACHED_MODULE: CACHED_HELPER}):
        raise PermissionError("the original Prompt, runner or signed legacy archival source changed")
    expected = {**imported.source_sha256, PARSER: NEW_PARSER, normal.SEED_RUNNER: RUNNER_PATCH}
    if any(file_digest(root / name) != sha for name, sha in expected.items()):
        raise PermissionError("the normal release changed source beyond the reviewed parser and R11 boundary guard")
    return {**expected, **{path.as_posix(): file_digest(root / path) for path in SOURCES}}


def _legacy_handoff(cached_root, old_root, old, imported_run):
    if file_digest(cached_root / PARSER) != OLD_PARSER or file_digest(cached_root / CACHED_MODULE) != CACHED_HELPER:
        raise PermissionError("the original archive must be verified with the signed legacy phase parser")
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONHOME", "PYTHONPATH", "PYTHONSAFEPATH")}
    env["PYTHONHASHSEED"] = "0"
    module = ".".join(Path(CACHED_MODULE).with_suffix("").parts)
    child = subprocess.run([
        sys.executable, "-B", "-s", "-m", module, "verify-handoff",
        "--old-release", str(old_root), "--old-run", str(old), "--run-dir", str(imported_run),
        "--contract-sha256", CACHED.contract, "--authorization-sha256", CACHED.approval,
    ], cwd=cached_root, env=env, capture_output=True, text=True, timeout=180, check=False)
    if child.returncode:
        raise PermissionError(f"the immutable R10 archive verifier failed (exit {child.returncode})")
    proof = json.loads(child.stdout)
    if proof != HANDOFF or proof["provider_post"] is not False:
        raise PermissionError("the sealed zero-POST L685 fit no longer reproduces its exact evidence")
    scheduler._assert_no_old_workers(imported_run, (SEED,))
    return proof


def inspect(root, old_root, old, cached_root, imported_run):
    normal._require_fixed_reconstruction()
    if (root in (old_root, cached_root) or old_root.name != cached.OLD_RELEASE
            or cached_root.name != CACHED_RELEASE or old.parent != imported_run.parent):
        raise PermissionError("use the exact immutable predecessor releases in one runs root")
    parent, parent_approval = _archived(old, old_root, cached.OLD)
    imported, _ = _archived(imported_run, cached_root, CACHED)
    sources = _normal_sources(root, parent, imported)
    if ((parent.target_labels, parent.seed_ids, parent.transport.logical_replies, parent.transport.transport_attempts,
            parent.profile.get("sealed_relay_retry_seconds"), parent.profile.get("audited_future_transport_replay"))
            != (TARGET, (SEED,), 3, 2, SEALED_RELAY_RETRY_SECONDS, True)
            or "fixed_reference_repair" in parent.profile or imported.target_labels != 685
            or imported.predecessor.get("execution_budget") != cached.BUDGET):
        raise PermissionError("the scientific profiles or authorized transport budgets changed")
    handoff = _legacy_handoff(cached_root, old_root, old, imported_run)
    directory = imported_run / f"seed_{SEED}"
    active, following = (directory / f"round_{index:02d}" for index in (START - 1, START))
    source, transition = (following / name for name in ("source.json", "transition.json"))
    fit = seed_runner._fit_directory(following)
    device = read(fit / "fit_device.json")
    state, memory = seed_runner.FrozenState(**read(source)), read(active / "memory.json")
    custody_file = imported_run / "cached_response_import.json"
    custody = read(custody_file)
    cache = Path(parent.predecessor["cache_directory"]).resolve(strict=True)
    resource = parent.predecessor["public_resource_receipt_sha256"]
    if ((state.seed, state.round, len(state.provenance["expected_observed"]), state.checkpoint)
            != (SEED, START, 685, handoff["checkpoint_sha256"])
            or not device.get("cuda") or device.get("deterministic_algorithms") is not True
            or file_digest(source) != handoff["source_file_sha256"] or file_digest(transition) != handoff["transition_file_sha256"]
            or digest(asdict(state)) != handoff["state_sha256"] or digest(memory) != handoff["scientific_memory_sha256"]
            or state.provenance["fit_audit_sha256"] != handoff["fit_audit_file_sha256"]
            or file_digest(fit / "fit_audit.json") != handoff["fit_audit_file_sha256"]
            or custody != imported.predecessor["cached_selection"] or custody.get("provider_post") is not False
            or digest(custody) != handoff["cached_selection_sha256"]
            or file_digest(cache / "resource_receipt.json") != resource):
        raise PermissionError("the verified zero-POST L685 state, model memory, fit or training cache changed")
    stage = seed_runner._stage(root, state, memory, seed_runner._scope(parent, parent_approval, SEED, False), parent)
    first_request = make_request(stage.binding, 0, stage.messages, parent.transport)
    first = normal.request_identity(first_request)
    check_first_round(first_request, first)
    pins = {"r10_handoff": handoff, "r11_request": first, "cached_response_import_file_sha256": file_digest(custody_file)}
    validate_pins(pins)
    return pins, (state, memory, START), cache, parent, imported, sources


def construct(root, old_root, old, cached_root, imported_run, run, pins):
    if run.name != RUN_NAME or run.parent != old.parent:
        raise PermissionError("the ordinary continuation requires its distinct sibling run")
    current, boundary, cache, parent, imported, sources = inspect(root, old_root, old, cached_root, imported_run)
    if pins != current:
        raise PermissionError("the generated L685 proof and full R11 request differ from the reviewed pins")
    handoff, _ = validate_pins(pins)
    profile = {**parent.profile, "resume_boundaries": {str(SEED): {
        "round": START, "state_sha256": handoff["state_sha256"], "memory_sha256": handoff["scientific_memory_sha256"],
    }}, "full_imported_boundary_round": START, "replay_eligible_from_round": START}
    proof = {
        "dispatcher_module": DISPATCHER, "seed": SEED, "round": START, **pins,
        "previous_run_directory": str(imported_run), "previous_release_directory": str(cached_root),
        "previous_contract_sha256": CACHED.contract, "previous_authorization_sha256": CACHED.approval,
        "normal_run_directory": str(old), "normal_release_directory": str(old_root),
        "normal_contract_sha256": cached.OLD.contract, "normal_authorization_sha256": cached.OLD.approval,
        "normal_seed_pause_file_sha256": cached.ARCHIVE_FILES[f"seed_{SEED}/pause.request"],
        "cached_selection": imported.predecessor["cached_selection"], "checkpoint_sha256": boundary[0].checkpoint,
        "cache_directory": str(cache), "public_resource_receipt_sha256": parent.predecessor["public_resource_receipt_sha256"],
        "model_identity": parent.predecessor["model_identity"], "bridge_wait_seconds": parent.predecessor["bridge_wait_seconds"],
        "recovery_budget": BOUNDARY_BUDGET, "execution_budget": BUDGET,
    }
    return replace(parent, profile=profile, source_sha256=sources, run_directory=str(run), predecessor=proof), boundary, cache


def execute(root, run, contract, authorization, boundary, cache, *, resume):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    approval = verify_authorization(contract, authorization)
    with scheduler._lock(run / "coordinator.lock", busy="the normal R11 coordinator is already active"):
        scheduler._assert_no_old_workers(run, (SEED,))
        directory = seed_runner.seed_directory(run, SEED)
        if scheduler._pause_requested(run, SEED) and not resume:
            raise PermissionError("the normal R11 continuation is paused; use resume")
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
                    or started["attempt_id"] in {cached.ATTEMPT, contract.predecessor["r10_handoff"]["local_attempt_id"]}):
                raise PermissionError("the bridge job is outside the normal R11–R35 scope")
            if binding["round"] == START:
                check_first_round(request, contract.predecessor["r11_request"])
                active = seed_runner.round_directory(directory, START)
                normal._correction_ready(active, request)
                normal._check_second_slot(active, request, started)
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
    for name in ("old-release", "old-run", "cached-release", "cached-run"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--handoff-pins", type=Path)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    old_root, old, cached_root, imported_run = (value.resolve(strict=True) for value in (
        args.old_release, args.old_run, args.cached_release, args.cached_run,
    ))
    normal._require_fixed_reconstruction()
    with normal._archival_lock(imported_run):
        if args.action == "inspect-handoff":
            print(dumps(inspect(ROOT, old_root, old, cached_root, imported_run)[0]))
            return
        if args.run_dir is None or args.handoff_pins is None:
            parser.error("contract, run and resume require the distinct run path and reviewed handoff pins")
        run = args.run_dir.resolve()
        contract, boundary, cache = construct(ROOT, old_root, old, cached_root, imported_run, run, read(args.handoff_pins))
        if args.action == "contract":
            print(dumps({"contract_sha256": contract.digest, **contract.record()}))
            return
        if args.authorization is None:
            parser.error("run and resume require the new exact continuation authorization")
        result = execute(ROOT, run, contract, read(args.authorization), boundary, cache, resume=args.action == "resume")
        print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
