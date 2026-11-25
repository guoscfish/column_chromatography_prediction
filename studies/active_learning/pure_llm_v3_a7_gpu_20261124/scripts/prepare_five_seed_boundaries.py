"""Prepare five authenticated A7 boundaries without an API request or new fit."""

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from functools import partial
import fcntl
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.memory_manager import MemoryManager
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportConfig, terminal_record
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import validate_result, verify_job
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    FormalContract, LEASE_ROOT, SEEDS, SSH_API, V4_PILOT_RUN_NAME, verify_authorization,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.supersession import verify_v3_anchors


V11_RUN_PREFIX = "a7-gpt62-astra-xhigh-ssh-v11-direct-full"
V11_CONTRACTS = {
    157: "23301d646bb9c6d704bd4a0dea64357de7d8069a8f378abbed88624fbac0497b",
    887: "ef7b71daeb6e8e0dc926f1d34d1d582f6e8abb0db4ed97ccfab1a293dcbb749d",
    2357: "13690e90aeccb52b850a86c83263a201962f8ab75ec67d75eae472b4506f87a5",
}
V11_START = {157: 1, 887: 5, 2357: 1}
PROMPT_SHA256 = "d656796e97dee76bd12bb7ce94bc3056d042e858e417759eb4108710744a594f"


@contextmanager
def _stopped(run, seed):
    paths = (run / "coordinator.lock", run / "workers" / f"seed_{seed}.lock",
             run / "operator" / "rolling.lock")
    with ExitStack() as locks:
        for path in paths:
            descriptor = locks.enter_context(path.open("r+"))
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise PermissionError(f"V11 seed {seed} is still active") from None
        yield


def _open_archive(run, seed):
    record = read(run / "formal_contract.json")
    if digest(record) != V11_CONTRACTS[seed] or record["run_directory"] != str(run):
        raise PermissionError(f"V11 contract identity changed for seed {seed}")
    profile = record["profile"]
    contract = FormalContract(profile, TransportConfig(**profile["transport"]), record["source_sha256"],
                              record["asset_sha256"], record["anchor_root"], tuple(record["gpu_slots"]),
                              record["lease_root"], str(run), record["api_execution"], record["predecessor"])
    if (contract.record() != record or contract.seed_ids != (seed,) or contract.api_execution != SSH_API
            or contract.gpu_slots != (0,) or contract.requires_initial_qualification
            or Path(contract.lease_root) != LEASE_ROOT or Path(contract.anchor_root) != LEASE_ROOT / "trajectories"):
        raise PermissionError("V11 archive is outside the reviewed single-GPU A7/M2 scope")
    approval = verify_authorization(contract, read(run / "authorization.json"))
    model_lock = read(run / "provider_model_lock.json")
    if digest(model_lock) != contract.predecessor["provider_model_lock_sha256"]:
        raise PermissionError(f"V11 provider identity changed for seed {seed}")
    return contract, approval, model_lock


def _scientific_profile(contract):
    excluded = {"confirmation_seeds", "resume_boundaries", "store", "transport",
                "request_timeouts", "provider_capability", "verified_served_model_reference"}
    scientific = {key: value for key, value in contract.profile.items() if key not in excluded}
    expected = {"arm": "A7", "memory": "M2", "device": "cuda", "initial_labels": 333,
                "target_labels": 1485, "initial_pool": 2997, "validation": 416, "test": 417,
                "batch_size": 32, "requires_initial_qualification": False}
    if any(scientific.get(key) != value for key, value in expected.items()):
        raise PermissionError("scientific settings differ from the approved A7/M2 trajectory")
    return {**scientific, "confirmation_seeds": list(SEEDS)}


def _retired_attempts(run, directory, contract, approval, start):
    """Keep links to complete failure evidence. Existing answers require in-place recovery."""
    scope = seed_runner._scope(contract, approval, contract.seed_ids[0], False)
    seen = {}
    for turn in sorted(directory.glob("turn_[0-9][0-9]")):
        request_path = turn / "request.json"
        reservations = sorted(turn.glob("attempt_*.started.json"))
        if not request_path.is_file():
            if reservations:
                raise ValueError("V11 reservations are missing their original request")
            continue
        request = read(request_path)
        binding = request["binding"]
        if ((binding["seed"], binding["round"], request["logical_index"], binding["a7_formal"])
                != (scope["seed"], start, int(turn.name[5:]), scope)):
            raise PermissionError("a V11 request differs from its exact scientific and authorization scope")
        for index, reservation in enumerate(reservations):
            if index >= contract.transport.transport_attempts or reservation.name != f"attempt_{index:02d}.started.json":
                raise PermissionError("V11 attempt sequence is outside the original finite budget")
            started = read(reservation)
            terminal = terminal_record(turn, index)
            name = f"{started['attempt_id']}.json"
            job = read(run / "bridge" / "outbox" / name)
            result = read(run / "bridge" / "results" / name)
            validated = verify_job(job, authorization_sha256=digest(approval), contract_sha256=contract.digest,
                                   wire_representation=contract.profile["wire_representation"])
            validate_result(result, job)
            if (validated != (request, contract.transport, started)
                    or terminal["bridge_result_sha256"] != result["result_sha256"]
                    or terminal["bridge_job_sha256"] != job["job_sha256"]):
                raise PermissionError("the V11 failure is detached from its original request or receipt")
            if (result["outcome"]["status"] != "failed" or result.get("raw_response") is not None
                    or terminal["status"] not in {"stopped", "retryable"}
                    or terminal.get("error") != result["outcome"]["failure"]["code"]):
                raise PermissionError("recover the existing V11 response before starting a successor round")
            seen[started["attempt_id"]] = {"request_id": request["request_id"],
                "payload_sha256": request["payload_sha256"], "bridge_result_sha256": result["result_sha256"],
                "terminal_sha256": digest(terminal), "code": terminal["error"],
                "execution_status_unknown": result["outcome"]["failure"]["ambiguous"]}
    return seen


def _archive_boundary(run, seed, contract, approval, backend):
    directory = seed_runner.seed_directory(run, seed)
    descriptor = contract.profile["resume_boundaries"][str(seed)]
    start = descriptor["round"]
    active = seed_runner.round_directory(directory, start)
    scope = seed_runner._scope(contract, approval, seed, False)
    source, snapshot = read(active / "source.json"), read(active / "boundary_memory.json")
    if (start != V11_START[seed] or read(directory / "boundary.json") != {"formal_run": scope, **descriptor}
            or descriptor != {"round": start, "state_sha256": digest(source), "memory_sha256": digest(snapshot)}):
        raise PermissionError(f"V11 seed {seed} start boundary changed")
    state = seed_runner.FrozenState(**source)
    if state.checkpoint != contract.predecessor["checkpoint_sha256"]:
        raise PermissionError(f"V11 seed {seed} initial checkpoint changed")
    transitions = {}
    while start < contract.rounds:
        following = seed_runner.round_directory(directory, start + 1)
        seal = following / "transition.json"
        if not seal.is_file():
            if any((active / name).exists() for name in ("batch_freeze.json", "selection.json", "memory.json")) or any(
                (following / name).exists() for name in ("authorized_feedback.json", "fit_authorization.json")
            ):
                raise PermissionError(f"finish already frozen V11 seed {seed} work before preparing a successor")
            break
        state, snapshot = seed_runner._successor(active, state, following, scope, contract.transport, contract, backend)
        start += 1
        transitions[str(start)] = file_digest(seal)
        active = following
    expected_seals = {seed_runner.round_directory(directory, int(round_index)) / "transition.json"
                      for round_index in transitions}
    if set(directory.glob("round_*/transition.json")) != expected_seals:
        raise PermissionError(f"V11 seed {seed} transition history is not contiguous")
    seed_runner._state_roles(state, backend, scope, start, contract.batch_size)
    fit_audit_sha = state.provenance.get("fit_audit_sha256")
    if not isinstance(fit_audit_sha, str) or len(fit_audit_sha) != 64:
        raise PermissionError("completed V11 state lacks its original fit audit identity")
    evidence = {"source": "sealed_v11", "run_directory": str(run), "contract_sha256": contract.digest,
                "authorization_sha256": digest(approval), "transition_sha256_by_round": transitions,
                "fit_audit_sha256": fit_audit_sha, "public_resource_receipt_sha256": backend.cache_receipt_sha256,
                "retained_transport_failures": _retired_attempts(run, active, contract, approval, start)}
    return state, snapshot, evidence


def _anchor_boundary(seed, backend, anchor_root, temporary, checkpoint):
    anchor = anchor_root / f"random-{seed}"
    destination = temporary / f"seed_{seed}" / "anchor"
    state = backend.initial_state(anchor, destination, allow_fresh_fit=False)
    receipt = read(destination / "initial_state_receipt.json")
    if (state.checkpoint != checkpoint or receipt["checkpoint_sha256"] != checkpoint
            or receipt["source"]["source_kind"] != "shared_numeric_L333_model_only"
            or state.provenance["resource_receipt_sha256"] != backend.cache_receipt_sha256):
        raise PermissionError(f"seed {seed} differs from its approved shared L333 CUDA anchor")
    seed_runner._state_roles(state, backend, {"seed": seed}, 0, 32)
    return state, MemoryManager(seed).snapshot(), {
        "source": "shared_cuda_l333_anchor", "anchor_directory": str(anchor),
        "inference_receipt_sha256": file_digest(destination / "initial_state_receipt.json"),
    }


def prepare(v11_release, runs, preparation):
    """Materialize exact private boundary inputs and return a transport-independent draft."""
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    if any(root == preparation or root in preparation.parents for root in (runs, v11_release)):
        raise ValueError("preparation must have a dedicated directory outside the historical runs and release")
    original = runs / V4_PILOT_RUN_NAME
    baseline = None
    prepared = {}
    with scheduler._lock(preparation / "preparation.lock", busy="five-seed boundary preparation is already active"):
        for seed in V11_CONTRACTS:
            run = runs / f"{V11_RUN_PREFIX}-{seed}"
            with _stopped(run, seed):
                contract, approval, model_lock = _open_archive(run, seed)
                common = (_scientific_profile(contract), contract.source_sha256, contract.asset_sha256, model_lock)
                if baseline is None:
                    baseline = common
                    if (contract.source_sha256[seed_runner.PROMPT_PATH.as_posix()] != PROMPT_SHA256
                            or any(file_digest(v11_release / name) != sha for name, sha in contract.source_sha256.items())
                            or any(file_digest(v11_release / name) != sha for name, sha in contract.asset_sha256.items())):
                        raise PermissionError("the reviewed V11 improved prompt, source or asset differs on disk")
                elif common != baseline:
                    raise PermissionError("V11 source, assets, improved prompt or model differs across the existing seeds")
                cache = original / "public_graphs" / f"seed_{seed}"
                if not all((cache / name).is_file() for name in (
                    "round.lock", "resource_receipt.json", "scaler.json", "scrubbed_graphs.pt"
                )):
                    raise FileNotFoundError(f"seed {seed} needs the original sealed public resource cache")
                backend = TrainingBackend(v11_release, cache, seed)
                state, snapshot, evidence = _archive_boundary(run, seed, contract, approval, backend)
                prepared[str(seed)] = _save_boundary(preparation, state, snapshot, cache, evidence)

        anchor_root = LEASE_ROOT / "trajectories"
        anchors = verify_v3_anchors(anchor_root)
        for seed in SEEDS:
            if seed in V11_CONTRACTS:
                continue
            if any(runs.glob(f"*/seed_{seed}/round_*/turn_*/attempt_*.started.json")) or any(
                runs.glob(f"*/seed_{seed}/round_*/*freeze.json")
            ) or any(runs.glob(f"*/seed_{seed}/round_*/transition.json")):
                raise PermissionError(f"seed {seed} already has scientific or transport custody that needs review")
            cache = preparation / "public_graphs" / f"seed_{seed}"
            slot = partial(scheduler.gpu_slot, preparation, seed, 0, LEASE_ROOT)
            backend = TrainingBackend(v11_release, cache, seed, gpu_slot=slot)
            state, snapshot, evidence = _anchor_boundary(seed, backend, anchor_root, preparation, anchors[str(seed)])
            prepared[str(seed)] = _save_boundary(preparation, state, snapshot, cache, evidence)

        scientific, _, assets, model_lock = baseline
        starts = [row["descriptor"]["round"] for row in prepared.values()]
        draft = {"schema": "a7_five_seed_boundary_draft_v1", "scientific_profile": scientific,
                 "asset_sha256": assets, "improved_prompt_sha256": PROMPT_SHA256,
                 "previous_provider_model_identity": model_lock, "gpu_slots": [0],
                 "gpu_lease_root": str(LEASE_ROOT), "initial_anchor_sha256_by_seed": anchors,
                 "boundaries": prepared, "remaining_first_logical_requests": sum(36 - start for start in starts),
                 "paid_requests_authorized": False}
        once(preparation / "five_seed_boundary_draft.json", draft)
        return draft


def _save_boundary(preparation, state, snapshot, cache, evidence):
    seed_runner.MemoryManager.from_snapshot(snapshot, seed=state.seed)
    if set(snapshot["rounds"]) != {str(index) for index in range(state.round)}:
        raise ValueError("the boundary lacks one or more earlier scientific memory rounds")
    saved = preparation / "boundaries" / f"seed_{state.seed}"
    source = asdict(state)
    once(saved / "state.json", source)
    once(saved / "memory.json", snapshot)
    return {"descriptor": {"round": state.round, "state_sha256": digest(source), "memory_sha256": digest(snapshot)},
            "boundary_directory": str(saved), "cache_directory": str(cache), "checkpoint_sha256": state.checkpoint,
            "active_labels": len(state.provenance["expected_observed"]), "evidence": evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v11-release", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    result = prepare(args.v11_release.resolve(strict=True), args.runs.resolve(strict=True), args.preparation.resolve())
    print(dumps({"draft_sha256": digest(result), "draft": result}))


if __name__ == "__main__":
    main()
