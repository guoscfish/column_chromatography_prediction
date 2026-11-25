"""Authorize one direct Mac request from a sealed revised-prompt seed state."""
import argparse
from dataclasses import asdict, replace
from functools import partial
import os
from pathlib import Path
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.memory_manager import MemoryManager
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportStop, check_request, make_request
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import BridgePending, validate_result
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    FormalContract, ROOT, SSH_API, STUDY, request_timeouts, verify_authorization,
)

from . import run_prompt_pilot as pilot

SEED_RUNS = {157: (1, "v8"), 2357: (1, "v8"), 887: (4, "v9")}
SEEDS = tuple(SEED_RUNS)
TOTAL_DEADLINE = 3600


def run_name(seed):
    _, version = SEED_RUNS[seed]
    return f"a7-gpt62-astra-xhigh-ssh-{version}-direct-one-{seed}"


class DirectContract(FormalContract):
    def record(self):
        record = super().record()
        record["budget"] = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 1}
        return record


def _failed_attempts(parent_run, turn, request):
    records = []
    for index in range(2):
        prefix = f"attempt_{index:02d}"
        started = read(turn / f"{prefix}.started.json")
        terminal = read(turn / f"{prefix}.terminal.json")
        identity = started["attempt_id"]
        if ((started["request_id"], started["attempt"], started.get("transport"))
                != (request["request_id"], index, "mac_bridge")
                or any(terminal.get(key) != value for key, value in started.items())
                or terminal["status"] not in ("stopped", "retryable") or "answer_sha256" in terminal
                or terminal.get("provider_status") == "completed"
                or (turn / f"{prefix}.recovery.terminal.json").exists()):
            raise ValueError("the parent physical ledger has recovered or nonterminal evidence")
        job = read(parent_run / "bridge/outbox" / f"{identity}.json")
        if job["request"] != request or job["started"] != started:
            raise ValueError("the parent physical job differs from its immutable attempt")
        result = validate_result(read(parent_run / "bridge/results" / f"{identity}.json"), job)
        if (result["outcome"]["status"] != "failed" or result["raw_response"] is not None
                or result["result_sha256"] != terminal.get("bridge_result_sha256")
                or result["job_sha256"] != terminal.get("bridge_job_sha256")):
            raise ValueError("the parent bridge ledger contains a completion or has changed")
        records.append({"attempt": index, "attempt_id": identity, "terminal_sha256": digest(terminal),
                        "bridge_result_sha256": result["result_sha256"], "failure": terminal["error"]})
    if records[0]["attempt_id"] == records[1]["attempt_id"] or terminal["status"] != "stopped":
        raise ValueError("the parent has not exhausted two distinct transport attempts")
    return records


def construct(root, old_release, old_run, first_release, first_run, parent_release, parent_run, run, seed):
    """Read the verified seed boundary and parent ledger without creating a new run."""
    if (seed not in SEEDS or run.name != run_name(seed) or run.parent != old_run.parent
            or parent_run == run):
        raise ValueError("one-shot direct acquisition requires its own registered seed run directory")
    start, _ = SEED_RUNS[seed]
    directory = seed_runner.seed_directory(parent_run, seed)
    active = seed_runner.round_directory(directory, start)
    following = seed_runner.round_directory(directory, start + 1)
    if ((active / "batch_freeze.json").exists() or (following / "transition.json").exists()
            or any(path.name != "turn_00" for path in active.glob("turn_[0-9][0-9]"))):
        raise ValueError("the parent already contains a later logical response, frozen batch, or subsequent transition")
    cache = old_run / "public_graphs" / f"seed_{seed}"
    old_seed = seed_runner.seed_directory(old_run, seed)
    if not all(path.is_file() for path in (cache / "round.lock", cache / "resource_receipt.json",
                                           old_seed / "initial/initial.json", old_seed / "initial/source.json")):
        raise ValueError("read-only verification requires the existing sealed parent resources")
    parent, boundaries = pilot.construct(parent_release, old_release, old_run, first_release, first_run, parent_run, seed)
    if (read(parent_run / "formal_contract.json") != parent.record() or parent.api_execution != SSH_API
            or parent.gpu_slots != (0,) or parent.initial_labels != 333 or parent.batch_size != 32
            or parent.transport.transport_attempts != 2):
        raise PermissionError("one-shot direct acquisition requires the exact reviewed V5 parent")
    approval = verify_authorization(parent, read(parent_run / "authorization.json"))
    scope = seed_runner._scope(parent, approval, seed, False)
    state, memory, index = boundaries[seed]
    MemoryManager.from_snapshot(memory, seed=seed)
    descriptor = parent.profile["resume_boundaries"][str(seed)]
    start_labels = parent.initial_labels + parent.batch_size * start
    target_labels = start_labels + parent.batch_size
    if (index != start or state.round != start or len(state.provenance["expected_observed"]) != start_labels
            or set(memory["rounds"]) != {str(round_index) for round_index in range(start)}
            or read(directory / "boundary.json") != {"formal_run": scope, **descriptor}
            or read(active / "source.json") != asdict(state)
            or read(active / "boundary_memory.json") != memory):
        raise ValueError(f"V5 start differs from its sealed original L{start_labels} state")

    turn = active / "turn_00"
    original = read(turn / "request.json")
    binding = original["binding"]
    check_request(original, binding, parent.transport)
    messages = original["payload"]["input"]
    if (binding.get("a7_formal") != scope or binding["checkpoint_sha256"] != state.checkpoint
            or binding["round"] != start or original["logical_index"] != 0
            or [message["role"] for message in messages] != ["system", "user"]
            or read(turn / "formal_request_authorization.json")
            != seed_runner._selection_grant(scope, original["request_id"])):
        raise ValueError("the exhausted request is outside the authorized round scope")
    failures = _failed_attempts(parent_run, turn, original)

    old_policy = request_timeouts(parent.profile)
    if old_policy is None or old_policy.total_seconds != parent.transport.timeout_seconds:
        raise ValueError("the original reviewed deadline is missing")
    timeouts = replace(old_policy, total_seconds=TOTAL_DEADLINE)
    transport = replace(parent.transport, transport_attempts=1, logical_replies=1, timeout_seconds=TOTAL_DEADLINE)
    profile = {**parent.profile, "confirmation_seeds": [seed], "target_labels": target_labels, "transport": asdict(transport),
               "request_timeouts": asdict(timeouts),
               "resume_boundaries": {str(seed): {**descriptor, "user_message_sha256": digest(messages[1])}}}
    model_lock = read(parent_run / "provider_model_lock.json")
    if model_lock != read(first_run / "provider_model_lock.json"):
        raise ValueError("provider identity changed across the revised-prompt segments")
    provider_path = (STUDY / "a7_gpu/provider.py").as_posix()
    sources = {name: file_digest(root / name) for name in parent.source_sha256}
    if (any(sources[name] != expected for name, expected in parent.source_sha256.items() if name != provider_path)
            or sources[provider_path] == parent.source_sha256[provider_path]):
        raise PermissionError("the direct release must change only the reviewed provider route and add direct scripts")
    for name in ("run_prompt_direct.py", "dispatch_prompt_direct.py"):
        path = (STUDY / "scripts" / name).as_posix()
        sources[path] = file_digest(root / path)
    evidence = {
        "pilot_contract_sha256": parent.predecessor["pilot_contract_sha256"],
        "contract_sha256": parent.digest, "authorization_sha256": digest(approval), "run_directory": str(parent_run),
        "checkpoint_sha256": state.checkpoint, "start_label_budget": start_labels,
        "original_request_id": original["request_id"], "original_payload_sha256": original["payload_sha256"],
        "exhausted_attempts": failures, "prior_execution_and_billing": "unresolved",
        "accept_duplicate_billing_risk": True, "uses_previous_incomplete_round_responses": False,
        "max_new_physical_attempts": 1, "provider_model_lock_sha256": digest(model_lock),
        "mac_connection_policy": "direct_no_proxy",
    }
    contract = DirectContract(**{**vars(parent), "profile": profile, "transport": transport,
                                "source_sha256": sources, "run_directory": str(run), "predecessor": evidence})
    return contract, (state, memory, start), original


def verify_fresh_input(root, contract, approval, boundary, original):
    state, memory, start = boundary
    scope = seed_runner._scope(contract, approval, state.seed, False)
    stage = seed_runner._stage(root, state, memory, scope, contract)
    previous = original["binding"]
    messages = original["payload"]["input"]
    stage = seed_runner.pin_boundary_user_message(
        stage, messages[1], contract.profile["resume_boundaries"][str(state.seed)]["user_message_sha256"])
    request = make_request(stage.binding, 0, stage.messages, contract.transport)
    if (start != SEED_RUNS[state.seed][0] or previous["round"] != start
            or request["payload"] != original["payload"] or request["request_id"] == original["request_id"]
            or digest(request["payload"]) != contract.predecessor["original_payload_sha256"]
            or any(stage.binding[name] != previous[name] for name in (
                "checkpoint_sha256", "packet_sha256", "wire_packet_sha256", "mapping_sha256", "prompt_sha256"))):
        raise PermissionError("direct authorization preserves the exact original two-message scientific payload")
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run", "resume"))
    for name in ("old-release", "old-run", "first-release", "first-run", "parent-release", "parent-run", "run-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    run, old_run, parent_run = args.run_dir.resolve(), args.old_run.resolve(), args.parent_run.resolve()
    contract, boundary, original = construct(ROOT, args.old_release.resolve(), old_run, args.first_release.resolve(),
        args.first_run.resolve(), args.parent_release.resolve(), parent_run, run, args.seed)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        raise PermissionError("one direct request requires the exact contract authorization")
    approval = verify_authorization(contract, read(args.authorization))
    request = verify_fresh_input(ROOT, contract, approval, boundary, original)
    seed, start = args.seed, boundary[2]
    with scheduler._lock(run / "coordinator.lock", busy="direct continuation already running"):
        scheduler._assert_no_old_workers(run, contract.seed_ids)
        if scheduler._pause_requested(run) and args.action != "resume":
            raise PermissionError("the direct continuation is paused; use resume")
        once(run / "formal_contract.json", contract.record())
        once(run / "authorization.json", approval)
        once(run / "fresh_request_scope.json", {"request_id": request["request_id"], "seed": seed,
             "round": start, "logical_index": 0, "attempt": 0, "prior_complete_response_in_payload": False,
             "payload_sha256": request["payload_sha256"]})
        model_lock = read(parent_run / "provider_model_lock.json")
        if digest(model_lock) != contract.predecessor["provider_model_lock_sha256"]:
            raise ValueError("the parent provider identity changed during verification")
        once(run / "provider_model_lock.json", model_lock)
        scheduler._atomic_status(run / "coordinator_generation.json", {"id": uuid4().hex, "pid": os.getpid()})
        if args.action == "resume":
            (run / "pause.request").unlink(missing_ok=True)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        record = run / "workers" / f"seed_{seed}.json"
        with scheduler._lock(run / "workers" / f"seed_{seed}.lock", busy="direct seed worker already running"):
            scheduler._atomic_status(record, {"status": "initializing", "seed": seed, "pid": os.getpid(), "gpu": 0})
            try:
                backend = TrainingBackend(ROOT, old_run / "public_graphs" / f"seed_{seed}", seed)
                call, options = scheduler._provider_wiring(run, contract, approval, None, seed)
                managed = options["managed_attempt"]

                def one_request(candidate, config, started):
                    if candidate != request or started["attempt"] != 0:
                        raise TransportStop("the direct contract authorizes one exact physical request")
                    return managed(candidate, config, started)

                output = seed_runner.run_seed(root=ROOT, run_directory=run, contract=contract, authorization=approval,
                    seed=seed, backend=backend, call=call, resume_boundary=boundary,
                    boundary_user_message=original["payload"]["input"][1], max_rounds=start + 1,
                    fit_slot=partial(scheduler.gpu_slot, run, seed, 0, contract.lease_root),
                    transport_options={"managed_attempt": one_request})
                scheduler._atomic_status(record, {**asdict(output), "pid": os.getpid(), "gpu": 0})
            except Exception as error:
                status = "paused" if scheduler._pause_requested(run, seed) else (
                    "bridge_pending" if isinstance(error, BridgePending) else "failed")
                scheduler._atomic_status(record, {"status": status, "seed": seed, "pid": os.getpid(), "gpu": 0,
                                                  "error_type": type(error).__name__, "reason": str(error)})
                raise
        print(dumps(asdict(output)))


if __name__ == "__main__":
    main()
