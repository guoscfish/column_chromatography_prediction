"""Continue each sealed A7 seed to L1485 using the common prompt and direct Mac route."""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.memory_manager import MemoryManager
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, read, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import parse_response
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportConfig, terminal_record
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import verify_result
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    FormalContract, ROOT, SSH_API, STUDY, RequestTimeouts, verify_authorization,
)

from . import run_prompt_pilot as pilot

RUN_PREFIX = "a7-gpt62-astra-xhigh-ssh-v11-direct-full"
V10_NAME = "a7-gpt62-astra-xhigh-ssh-v10-targeted-repair-887"
V10_MODULE = "studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.run_prompt_targeted_repair"
DIRECT_PROVIDER_SHA256 = "a0b4d47dc973aa5368acead4dc0157ac230f91cc39bf78b8794f468bf1623c11"
BOUNDARIES = {157: 1, 2357: 1, 887: 5}
TARGET = 1485


class _ArchivedOneShot(FormalContract):
    def record(self):
        record = super().record()
        record["budget"] = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 1}
        return record


def _verify_v10_contract(release, run, paths):
    record = read(run / "formal_contract.json")
    if run.name != V10_NAME or record.get("run_directory") != str(run):
        raise PermissionError("the repair predecessor must be the exact registered V10 run")
    contract = _ArchivedOneShot(record["profile"], TransportConfig(**record["profile"]["transport"]),
                record["source_sha256"], record["asset_sha256"], record["anchor_root"], tuple(record["gpu_slots"]),
                record["lease_root"], str(run), record["api_execution"], record["predecessor"])
    approval = verify_authorization(contract, read(run / "authorization.json"))
    if (record != contract.record() or contract.seed_ids != (887,) or contract.target_labels != 493
            or contract.api_execution != SSH_API or contract.gpu_slots != (0,)
            or not isinstance(contract.profile.get("fixed_reference_repair"), dict)
            or any(file_digest(release / name) != sha for name, sha in record["source_sha256"].items())):
        raise PermissionError("the V10 repair contract or its isolated source release changed")
    command = [sys.executable, "-m", V10_MODULE, "contract", "--run-dir", str(run)]
    for name, path in paths.items():
        command.extend((f"--{name}", str(path)))
    report = subprocess.run(command, cwd=release, env={**os.environ, "PYTHONPATH": str(release),
                            "PYTHONDONTWRITEBYTECODE": "1"}, check=True, capture_output=True, text=True, timeout=300)
    if parse_response(report.stdout) != {"contract_sha256": contract.digest, **record}:
        raise PermissionError("the preserved V10 source did not reproduce its signed repair contract")
    return contract, approval


def _complete_repair(v10, seed, state, memory, backend, contract, approval):
    seed_dir = seed_runner.seed_directory(v10, seed)
    active = seed_runner.round_directory(seed_dir, state.round)
    scope = seed_runner._scope(contract, approval, seed, False)
    boundary = contract.profile["resume_boundaries"][str(seed)]
    if (read(seed_dir / "boundary.json") != {"formal_run": scope, **boundary}
            or read(active / "source.json") != asdict(state) or read(active / "boundary_memory.json") != memory):
        raise PermissionError("the sealed V10 boundary is outside the original L461 trajectory")
    freeze = read(active / "batch_freeze.json")
    if (freeze["accepted_logical_index"], freeze["accepted_attempt"]) != (0, 0):
        raise PermissionError("the V10 selection did not use its one approved patch")
    turn = active / "turn_00"
    accepted = terminal_record(turn, 0)
    request = read(turn / "request.json")
    started = read(turn / "attempt_00.started.json")
    identity = started["attempt_id"]
    job = read(v10 / "bridge/outbox" / f"{identity}.json")
    result = read(v10 / "bridge/results" / f"{identity}.json")
    if (accepted["status"] != "completed" or job["request"] != request or job["started"] != started
            or accepted.get("bridge_result_sha256") != result["result_sha256"]
            or accepted.get("bridge_job_sha256") != job["job_sha256"]):
        raise ValueError("the V10 repair response lacks its complete linked bridge evidence")
    reply = verify_result(result, job)
    if reply.text != accepted["text"] or reply.response_id != accepted["response_id"]:
        raise PermissionError("the V10 patch differs from the validated Mac response")
    data = contract.profile["fixed_reference_repair"]
    original = read(v10 / "canonical_repair_plan.json")
    chosen = read(active / "selection.json")
    patch = parse_response(accepted["text"])
    if (digest(original) != data["canonical_plan_sha256"] or type(patch) is not dict
            or set(patch) != {"reference", "reason"} or patch["reference"] != {
                "reference_id": data["reference_id"], "reference_kind": "observed",
                "relationship": "single_factor_contrast", "changed_axis": data["changed_axis"]}):
        raise PermissionError("the accepted V10 edit exceeds the signed reference repair")
    expected = deepcopy(original)
    edit = expected["choices"][data["repair_index"]]
    edit["references"][data["reference_index"]] = patch["reference"]
    edit["reason"] = patch["reason"]
    if (chosen != expected or digest([item["candidate_id"] for item in chosen["choices"]])
            != data["ordered_candidate_ids_sha256"]):
        raise PermissionError("the repaired V10 choice changed the original batch or another field")
    return pilot._take_transition(v10, state.round, seed, state, backend, contract, approval)


def construct(root, old_release, old_run, first_release, first_run, parent_release, parent_run, run, seed, *,
              v9_release=None, v9_run=None, v10_release=None, v10_run=None):
    """Construct an immutable V11 run using only completed V5/V10 scientific states."""
    if (seed not in BOUNDARIES or run.name != f"{RUN_PREFIX}-{seed}" or run.parent != old_run.parent
            or run in (old_run, first_run, parent_run, v9_run, v10_run)):
        raise ValueError("the continuation requires its own registered run directory")
    repair_paths = v9_release, v9_run, v10_release, v10_run
    if (seed == 887 and any(path is None for path in repair_paths)
            or seed != 887 and any(path is not None for path in repair_paths)):
        raise ValueError("only seed887 requires the archived V9 and V10 release and run paths")
    if seed == 887 and any(path.parent != run.parent for path in (v9_run, v10_run)):
        raise ValueError("the repair runs must be siblings on persistent storage")

    cache = old_run / "public_graphs" / f"seed_{seed}"
    initial = seed_runner.seed_directory(old_run, seed) / "initial"
    if not all(path.is_file() for path in (cache / "round.lock", cache / "resource_receipt.json",
                                           initial / "initial.json", initial / "source.json")):
        raise ValueError("read-only verification needs the existing sealed training resources")
    parent, boundaries = pilot.construct(parent_release, old_release, old_run, first_release, first_run, parent_run, seed)
    if (read(parent_run / "formal_contract.json") != parent.record() or parent.device != "cuda"
            or parent.initial_labels != 333 or parent.target_labels != 653 or parent.batch_size != 32
            or parent.api_execution != SSH_API or parent.gpu_slots != (0,) or parent.requires_initial_qualification
            or parent.transport.transport_attempts != 2 or parent.transport.logical_replies != 3):
        raise PermissionError("the parent is outside the registered revised-prompt V5 protocol")
    parent_approval = verify_authorization(parent, read(parent_run / "authorization.json"))
    model_lock = read(parent_run / "provider_model_lock.json")
    if model_lock != read(first_run / "provider_model_lock.json"):
        raise PermissionError("the V5 provider model identity differs from the sealed revised route")

    state, memory, start = boundaries[seed]
    parent_dir = seed_runner.seed_directory(parent_run, seed)
    parent_stage = seed_runner.round_directory(parent_dir, start)
    parent_scope = seed_runner._scope(parent, parent_approval, seed, False)
    if (read(parent_dir / "boundary.json") != {"formal_run": parent_scope, **parent.profile["resume_boundaries"][str(seed)]}
            or read(parent_stage / "source.json") != asdict(state)
            or read(parent_stage / "boundary_memory.json") != memory):
        raise PermissionError("the V5 start does not match its complete predecessor trajectory")
    if seed != 887 and ((parent_stage / "batch_freeze.json").exists()
            or (seed_runner.round_directory(parent_dir, start + 1) / "transition.json").exists()):
        raise ValueError("the V5 seed already advanced beyond its registered L365 endpoint")
    evidence = {"contract_sha256": parent.digest, "authorization_sha256": digest(parent_approval),
                "run_directory": str(parent_run), "boundary_source_sha256": file_digest(parent_stage / "source.json"),
                "transition_sha256": parent.predecessor["boundary_evidence"][str(seed)]["transition_sha256"]}
    if seed == 887:
        paths = {"old-release": old_release, "old-run": old_run, "first-release": first_release, "first-run": first_run,
                 "parent-release": parent_release, "parent-run": parent_run, "v9-release": v9_release, "v9-run": v9_run}
        repaired, repaired_approval = _verify_v10_contract(v10_release, v10_run, paths)
        if read(v10_run / "provider_model_lock.json") != model_lock:
            raise PermissionError("the repair segment used a different provider identity")
        backend = TrainingBackend(old_release, cache, seed)
        state, memory, transition = _complete_repair(v10_run, seed, state, memory, backend, repaired, repaired_approval)
        evidence = {"contract_sha256": repaired.digest, "authorization_sha256": digest(repaired_approval),
                    "run_directory": str(v10_run), "transition_sha256": transition, "verified_v5_boundary": evidence}

    start = BOUNDARIES[seed]
    start_labels = parent.initial_labels + parent.batch_size * start
    MemoryManager.from_snapshot(memory, seed=seed)
    if (state.round != start or len(state.provenance["expected_observed"]) != start_labels
            or set(memory["rounds"]) != {str(index) for index in range(start)}):
        raise ValueError("the seed does not have its complete approved predecessor history")
    provider_file = (STUDY / "a7_gpu/provider.py").as_posix()
    sources = {name: file_digest(root / name) for name in parent.source_sha256}
    if (sources[provider_file] != DIRECT_PROVIDER_SHA256
            or any(sources[name] != sha for name, sha in parent.source_sha256.items() if name != provider_file)):
        raise PermissionError("V11 must retain the common V5 prompt and use only the exact direct provider change")
    for script in ("run_prompt_finish.py", "dispatch_prompt_finish.py"):
        name = (STUDY / "scripts" / script).as_posix()
        sources[name] = file_digest(root / name)

    transport = replace(parent.transport, timeout_seconds=3600)
    timeout = RequestTimeouts(connect_seconds=30, write_seconds=120, read_idle_seconds=1800,
                             total_seconds=3600, bridge_margin_seconds=600)
    descriptor = {"round": start, "state_sha256": digest(asdict(state)), "memory_sha256": digest(memory)}
    profile = {**parent.profile, "confirmation_seeds": [seed], "target_labels": TARGET,
               "transport": asdict(transport), "request_timeouts": asdict(timeout),
               "resume_boundaries": {str(seed): descriptor}}
    remaining = (TARGET - start_labels) // parent.batch_size
    logical = remaining * transport.logical_replies
    predecessor = {**evidence, "start_label_budget": start_labels, "checkpoint_sha256": state.checkpoint,
                   "provider_model_lock_sha256": digest(model_lock), "mac_connection_policy": "direct_no_proxy",
                   "uses_previous_incomplete_round_responses": False,
                   "extension_budget": {"first_logical_requests": remaining, "maximum_logical_requests": logical,
                                        "maximum_physical_attempts": logical * transport.transport_attempts}}
    contract = replace(parent, profile=profile, transport=transport, source_sha256=sources,
                       run_directory=str(run), predecessor=predecessor)
    return contract, (state, memory, start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run", "resume"))
    for name in ("old-release", "old-run", "first-release", "first-run", "parent-release", "parent-run", "run-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("v9-release", "v9-run", "v10-release", "v10-run"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--seed", choices=tuple(BOUNDARIES), type=int, required=True)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    run, old_run = args.run_dir.resolve(), args.old_run.resolve()
    evidence_paths = {key: path.resolve() if path is not None else None for key, path in
                      ((key, getattr(args, key)) for key in ("v9_release", "v9_run", "v10_release", "v10_run"))}
    contract, boundary = construct(ROOT, args.old_release.resolve(), old_run, args.first_release.resolve(),
        args.first_run.resolve(), args.parent_release.resolve(), args.parent_run.resolve(), run, args.seed,
        **evidence_paths)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        raise PermissionError("the direct full continuation requires exact contract authorization")
    approval = verify_authorization(contract, read(args.authorization))
    ctx = multiprocessing.get_context("spawn")
    with scheduler._lock(run / "coordinator.lock", busy="the direct full continuation is already running"):
        scheduler._assert_no_old_workers(run, contract.seed_ids)
        if scheduler._pause_requested(run) and args.action != "resume":
            raise PermissionError("the direct full continuation is paused; use resume")
        model_lock = read(Path(contract.predecessor["run_directory"]) / "provider_model_lock.json")
        if digest(model_lock) != contract.predecessor["provider_model_lock_sha256"]:
            raise PermissionError("the predecessor model lock changed during verification")
        once(run / "formal_contract.json", contract.record())
        once(run / "authorization.json", approval)
        once(run / "provider_model_lock.json", model_lock)
        generation = uuid4().hex
        once(run / "events" / f"{generation}.json", {"action": args.action, "timestamp_unix": time.time(),
                                                     "authorization_sha256": digest(approval)})
        scheduler._atomic_status(run / "coordinator_generation.json", {"id": generation, "pid": os.getpid()})
        if args.action == "resume":
            (run / "pause.request").unlink(missing_ok=True)
        child = ctx.Process(target=pilot.worker,
                            args=(ROOT, run, old_run, contract, approval, args.seed, boundary, generation),
                            name=f"a7-direct-full-{args.seed}")
        child.start()
        child.join()
        print(dumps({"seed": args.seed, "target_labels": TARGET, "worker_exit": child.exitcode}))
        raise SystemExit(int(child.exitcode != 0))


if __name__ == "__main__":
    main()
