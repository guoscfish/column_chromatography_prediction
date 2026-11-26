"""Resume an exact failed A7 round with one initial POST and one audited relay retry."""

import argparse
from dataclasses import asdict, dataclass, replace
from functools import partial
import os
from pathlib import Path
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportStop, make_request
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import BridgePending
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, STUDY, verify_authorization
from . import run_v13_bounded_sse_recovery as bounded
from . import run_v13_resilient_failure_recovery as failure


DISPATCHER = "dispatch_v13_relay_retry"
ALIAS = "authoritative_base_messages_sha256"
RECOVERY_BUDGET = {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
COOLDOWN_SECONDS = 180
OVERLAYS = failure.OVERLAYS | {(STUDY / "a7_gpu/bridge_protocol.py").as_posix()}


@dataclass(frozen=True)
class ArchivedLane:
    seed: int
    round: int
    contract_sha256: str
    authorization_sha256: str
    attempt_id: str
    request_id: str
    payload_sha256: str
    job_sha256: str
    result_sha256: str
    terminal_file_sha256: str
    provider_request_id: str
    response_id: str

    @property
    def run_name(self):
        return f"{bounded.PREFIX}-{self.seed}-relay-retry-01"

    @property
    def parent(self):
        return bounded.RunIdentity(f"-{self.seed}-resilient-failure-01",
                                   self.contract_sha256, self.authorization_sha256)

    @property
    def observation(self):
        return {"error_type": "response_id_mismatch", "conflicting_event_type": "response.failed", "http_status": 200,
                "sse_event_count": 41, "provider_request_id": self.provider_request_id, "response_id": self.response_id,
                "conflicting_response_id": "resp_" + self.provider_request_id.replace("-", "")}


LANES = {
    157: ArchivedLane(
        157, 3, "3b35720e9795bfabacfea545d8e26349b84004ed2b9f3aab84bd4c19fde18bdf",
        "da0e03b4f2d9a8c0bdeb8377bda1cb592f9f615361c44e2e5cdd3e80343c1b42",
        "40f86013a8634cb6922eb676e3cbd6d9", "a4c49ed113517a7a848af845092da9f11b0412780100e1e613996d9cae484a99",
        "594ff66ced4fe668cdde6bcbdcc07e9f8cf49933a45accc89ef81a4c97e42107",
        "14892c91c2bb614f9aeced2f21453e120642b7372a52a5b37cee1ae9b0836b97",
        "5c69c8437c9a4aed04e0ab072a8fc1254fd6659120ecb0d77acca1d435948f1a",
        "df7a48814be4e6da01904a0e5844b62b9b807d596f4ced79498ce0080681b699",
        "f2fadb17-e9f5-4bdd-b5b6-dd9d3705c875", "resp_062d62002ff66c2a016b07e5f28a7887d2a6d430f92b50a429"),
    2357: ArchivedLane(
        2357, 4, "aa26cf2fae0779131218c62db7fbb8cd95fa89383fadfa46496d264db23669d3",
        "716db75648f44c1994603edc724f374d638da7b46922445a475b25d595823afd",
        "42bab9007c5748a09b03f6e84062fdfd", "e8bc3ee90e66cdd639703df3375f5f2738525191a5a0ff0a62fcf96d44aeb373",
        "302fe66d28c6c9aa4cfd604a56d623cc0f6d7a8d4bd53f4ee67e194732bf39ea",
        "74963b18eed4c5b39177e76c57e9f087464bb13894a8880ed02dfcad7046a5bf",
        "e2f5b2d48609993e3d67c9dda04c9bdb20e1f14cf02b6d2eb121d5fe530c3f77",
        "70c6aa40de727dd098248e478dcd3353ad4b6a9c29f5661c57bb274f9291c63f",
        "4de42ecb-2f3f-45a6-81fa-586a1d105f39", "resp_0ad54a2b7e56cc0b016b07e87f7b0c87d19305c9ccc72031a4"),
    6101: ArchivedLane(
        6101, 1, "735a7fddda1a5b2dd52ff5afaf0726c7220713bbdbc6d735b9dbdecbcbe92ae8",
        "6f18d91f7dd411c204347ba44c6514650d14cf50d910057d383d9e481de6dafd",
        "39339b8a78ea42de97f33616d90558f6", "c0b1413dc15be84caaea4c5ff3568950bad3f01dd18f54f48389daeba52c0449",
        "55c213c38e8c6b5fe3832f05060930bc436ec530d437da3551dc74f6d1324eeb",
        "b460d88f2b2109f39fa7462c04965975ee3c40f57548dd29136cfe5d76c8f3b5",
        "5c976eeeab8d2ca7d5308a9d01a75fe2d5edc2a4501c4199228fbf2f68958653",
        "a060fb897c3dc58f8ecab4b564f552f44137974fdc1ee2dfaa4001d7324e9464",
        "bbee93e7-15f0-4bd3-b13a-9e6e74c3a798", "resp_0eaf0e5ceebd7dd4016b07e5f34ffc87d28c6511837db7aa36"),
    12203: ArchivedLane(
        12203, 4, "95236d24b4bd3ddcfce419ff7f3ca8bf004f7cde838253d1bc75a7667dd3d88f",
        "78b7594b6be7f2f44f4eae4db1dbec8bd3aa102f9f3a52fd6be51d87067bdb8c",
        "c5ca8e4d76204482a4ac24f53659736a", "8c6a6633fa0127051252bdfe35c71591a68458346070c77e9bfd0b2ab9f0ca04",
        "0bd98eeac5b9b9049b7e00d7e8e439d0d1c07e7bd419370440d0259dfc91e1f5",
        "fe3571b44ed20986a549339ff5e98341e4c1c6caf74bbdf3c5514a6ba2e06668",
        "ed657fa9f3d37d5ccd5b8aaf2c799b4f58c36a6bbebf925e31715150e1a198fb",
        "2342a52f5617d8a620561f53f9510535f19aea4d489fc114aabfb63ce8d7b6bd",
        "aa06d60c-1ffd-46b0-b0a3-0c87397aab2f", "resp_02135f064ea4759f016b07e6c52f8c87d28e71676d3a4c2796"),
}


def execution_budget(total_rounds, start):
    remaining = total_rounds - start
    return {"first_logical_requests": remaining, "maximum_logical_requests": 1 + 3 * (remaining - 1),
            "maximum_physical_attempts": 2 + 6 * (remaining - 1)}


def _retired(run, lane, *, require_pause):
    directory, active = bounded.earlier._uncommitted(run, lane, require_paused=require_pause)
    worker = read(run / "workers" / f"seed_{lane.seed}.json")
    if worker.get("seed") != lane.seed or worker.get("status") != "failed":
        raise PermissionError("the exact failed worker has not retired")
    pid = worker.get("pid")
    if type(pid) is int:
        try:
            arguments = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            arguments = ()
        if str(run).encode() in arguments:
            raise PermissionError("the original worker still owns the same seed")
    return directory, active


def _failure_receipt(previous_run, active, previous, approval, lane):
    turn = bounded._turn(active, 1)
    request = bounded._request(active, turn, previous, approval, lane)
    receipt = bounded._failure(previous_run, turn, previous, approval, request, 0)
    expected = (lane.attempt_id, lane.job_sha256, lane.result_sha256, "response_id_mismatch", True, True)
    if (tuple(receipt[k] for k in ("attempt_id", "job_sha256", "result_sha256", "error", "provider_dispatched",
                                   "ambiguous")) != expected
            or (request["request_id"], request["payload_sha256"]) != (lane.request_id, lane.payload_sha256)
            or file_digest(turn / "attempt_00.terminal.json") != lane.terminal_file_sha256):
        raise PermissionError("the original posted relay failure differs from the sealed ambiguous receipt")
    return request, receipt


def _boundary(root, directory, current, previous, approval, lane, cache):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    scope = seed_runner._scope(previous, approval, lane.seed, False)
    old = previous.profile["resume_boundaries"][str(lane.seed)]
    start = old["round"]
    initial = seed_runner.round_directory(directory, start)
    state, memory = (seed_runner.FrozenState(**read(initial / "source.json")), read(initial / "boundary_memory.json"))
    if (not 0 <= start <= lane.round < previous.rounds
            or read(directory / "boundary.json") != {"formal_run": scope, **old}
            or (digest(asdict(state)), digest(memory), digest(read(initial / "boundary_user_message.json")))
            != (old["state_sha256"], old["memory_sha256"], old["user_message_sha256"])):
        raise PermissionError("the previous signed imported boundary or its memory changed")
    proof = {"initial_boundary": file_digest(directory / "boundary.json")}
    backend = TrainingBackend(root, cache, lane.seed)
    for index in range(start, lane.round):
        complete = seed_runner.round_directory(directory, index)
        follow = seed_runner.round_directory(directory, index + 1)
        state, memory = seed_runner._successor(complete, state, follow, scope, previous.transport, previous, backend)
        proof[f"transition_{index + 1:02d}"] = file_digest(follow / "transition.json")
    if state.round != lane.round or asdict(state) != read(current / "source.json"):
        raise PermissionError("the retry checkpoint is not the latest completed signed fit")
    proof["latest_source"] = file_digest(current / "source.json")
    proof["latest_memory"] = digest(memory)
    return (state, memory, lane.round), proof


def _compare_science(request, archived, observation=None):
    scientific = {k: v for k, v in request["binding"].items() if k != "a7_formal"}
    original = {k: v for k, v in archived["binding"].items() if k != "a7_formal"}
    if observation is not None:
        if (original.get(ALIAS), scientific.get(ALIAS)) != (observation["previous_sha256"],
                                                            observation["reconstructed_sha256"]):
            return False
        scientific[ALIAS] = original[ALIAS]
    return (scientific == original and request["payload"] == archived["payload"]
            and request["payload_sha256"] == archived["payload_sha256"] and request["logical_index"] == 0)


def construct(root, failed_release, failed_run, run, lane):
    if run.name != lane.run_name or run.parent != failed_run.parent or root == failed_release:
        raise PermissionError("use the exact successor sibling and a separate immutable release")
    previous, approval = bounded._archived(failed_run, failed_release, lane.parent)
    directory, active = _retired(failed_run, lane, require_pause=False)
    request, receipt = _failure_receipt(failed_run, active, previous, approval, lane)
    old_proof = previous.predecessor
    cache = Path(old_proof["cache_directory"]).resolve(strict=True)
    resource = file_digest(cache / "resource_receipt.json")
    if resource != old_proof["public_resource_receipt_sha256"]:
        raise PermissionError("the original public training cache changed")
    boundary, prefix = _boundary(failed_release, directory, active, previous, approval, lane, cache)
    state, memory, index = boundary
    scope = seed_runner._scope(previous, approval, lane.seed, False)
    user = request["payload"]["input"][1]
    stage = seed_runner._stage(failed_release, state, memory, scope, previous)
    stage = seed_runner.pin_boundary_user_message(stage, user, digest(user))
    rebuilt = make_request(stage.binding, 0, stage.messages, previous.transport)
    host = {}
    archived_host, rebuilt_host = request["binding"][ALIAS], stage.binding[ALIAS]
    if archived_host != rebuilt_host:
        host = {"previous_sha256": archived_host, "reconstructed_sha256": rebuilt_host}
    identity = old_proof["model_identity"]
    if not _compare_science(rebuilt, request, host or None) or read(failed_run / "provider_model_lock.json") != identity:
        raise PermissionError("the full archived A7 request or served model cannot be reproduced")
    descriptor = {"round": index, "state_sha256": digest(asdict(state)),
                  "memory_sha256": digest(memory), "user_message_sha256": digest(user)}
    proof = {"dispatcher_module": DISPATCHER, "seed": lane.seed, "round": index, "logical_index": 0,
             "previous_contract_sha256": previous.digest, "previous_authorization_sha256": digest(approval),
             "previous_run_directory": str(failed_run), "previous_release_directory": str(failed_release),
             "previous_request_id": request["request_id"], "previous_attempts": [receipt],
             "previous_transport_observation": lane.observation, "previous_prefix_sha256": digest(prefix),
             "previous_prefix_files": prefix,
             "previous_terminal_file_sha256": lane.terminal_file_sha256,
             "payload_sha256": request["payload_sha256"], "model_identity": identity,
             "recovery_budget": RECOVERY_BUDGET, "execution_budget": execution_budget(previous.rounds, index),
             "bridge_wait_seconds": old_proof["bridge_wait_seconds"], "cache_directory": str(cache),
             "public_resource_receipt_sha256": resource}
    proof.update({key: request["binding"][key] for key in (
                  "checkpoint_sha256", "packet_sha256", "wire_packet_sha256", "base_messages_sha256")})
    if host:
        proof["authoritative_message_reconstruction"] = host
    profile = {**previous.profile, "resume_boundaries": {str(lane.seed): descriptor},
               "sealed_relay_retry_seconds": COOLDOWN_SECONDS, "replay_eligible_from_round": index}
    sources = {name: file_digest(root / name) for name in previous.source_sha256}
    if {name for name, actual in sources.items() if actual != previous.source_sha256[name]} - OVERLAYS:
        raise PermissionError("the new attempt changed source outside audited transport and adapter paths")
    for name in ("run_v13_relay_retry", DISPATCHER):
        path = (STUDY / "scripts" / f"{name}.py").as_posix()
        sources[path] = file_digest(root / path)
    contract = replace(previous, profile=profile, source_sha256=sources, run_directory=str(run), predecessor=proof)
    return contract, boundary, cache, request


def execute(root, failed_run, run, lane, contract, approval, boundary, cache, original, *, resume):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    _retired(failed_run, lane, require_pause=True)
    exact = bounded.earlier._request(root, contract, approval, lane, boundary[0], boundary[1], original)
    proof = contract.predecessor
    if not _compare_science(exact, original, proof.get("authoritative_message_reconstruction")):
        raise PermissionError("the signed request changed before activation")
    status = {"seed": lane.seed, "pid": os.getpid(), "gpu": contract.gpu_slots[0]}
    path = run / "workers" / f"seed_{lane.seed}.json"
    with scheduler._lock(run / "coordinator.lock", busy="this relay retry lane is already running"):
        if scheduler._pause_requested(run, lane.seed) and not resume:
            raise PermissionError("this relay retry is paused; use resume")
        once(run / "formal_contract.json", contract.record())
        once(run / "authorization.json", approval)
        once(run / "provider_model_lock.json", proof["model_identity"])
        if resume:
            (run / "pause.request").unlink(missing_ok=True)
            (seed_runner.seed_directory(run, lane.seed) / "pause.request").unlink(missing_ok=True)
        scheduler._atomic_status(run / "coordinator_generation.json", {"id": uuid4().hex, "pid": os.getpid()})
        scheduler._atomic_status(path, {**status, "status": "initializing"})
        try:
            call, options = scheduler._provider_wiring(run, contract, approval, None, lane.seed)
            managed = options["managed_attempt"]
            managed.wait_seconds = proof["bridge_wait_seconds"]

            def checked_attempt(candidate, config, started):
                _retired(failed_run, lane, require_pause=True)
                binding = candidate["binding"]
                if binding["seed"] != lane.seed or not lane.round <= binding["round"] < contract.rounds:
                    raise TransportStop("the retry scope cannot submit a completed round or another seed")
                if binding["round"] == lane.round and (
                    candidate != exact or config != contract.transport
                    or started["request_id"] != exact["request_id"] or started["attempt"] not in (0, 1)
                    or started["attempt_id"] == lane.attempt_id
                ):
                    raise TransportStop("the current round is limited to one exact logical request and two physical slots")
                return managed(candidate, config, started)

            result = seed_runner.run_seed(
                root=root, run_directory=run, contract=contract, authorization=approval, seed=lane.seed,
                backend=TrainingBackend(root, cache, lane.seed), call=call, resume_boundary=boundary,
                boundary_user_message=original["payload"]["input"][1], transport_options={"managed_attempt": checked_attempt},
                fit_slot=partial(scheduler.gpu_slot, run, lane.seed, status["gpu"], contract.lease_root),
            )
            scheduler._atomic_status(path, {**status, **asdict(result)})
            return result
        except Exception as exc:
            phase = "paused" if scheduler._pause_requested(run, lane.seed) else (
                "bridge_pending" if isinstance(exc, BridgePending) else "failed")
            scheduler._atomic_status(path, {**status, "status": phase,
                                            "error_type": type(exc).__name__, "reason": str(exc)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run", "resume"))
    parser.add_argument("--seed", type=int, choices=tuple(LANES), required=True)
    parser.add_argument("--failed-release", type=Path, required=True)
    parser.add_argument("--failed-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    failed_release, failed_run, run = (path.resolve() for path in (args.failed_release, args.failed_run, args.run_dir))
    lane = LANES[args.seed]
    contract, boundary, cache, original = construct(ROOT, failed_release, failed_run, run, lane)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        parser.error("run and resume require the exact new relay retry authorization")
    approval = verify_authorization(contract, read(args.authorization))
    result = execute(ROOT, failed_run, run, lane, contract, approval, boundary, cache, original,
                     resume=args.action == "resume")
    print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
