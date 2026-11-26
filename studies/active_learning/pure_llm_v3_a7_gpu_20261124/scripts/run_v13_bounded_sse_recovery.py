"""Authorize one exact SSE attempt for each exhausted V13 decision, after archive checks."""

import argparse
from dataclasses import asdict, dataclass, replace
from functools import partial
import os
from pathlib import Path
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_STREAM_IPV4, TransportConfig, TransportStop, _link_replay_attempt, terminal_record,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import (
    BridgePending, read_bridge_json, validate_result, verify_job,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import AUTHORITATIVE_PACKET_FILE
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    OFFLINE, ROOT, SSH_API, STUDY, RequestTimeouts, verify_authorization,
)
from . import run_v13_lane_extension as earlier
from .run_prepared_five_seed import PreparedCohortContract


PREFIX = "a7-gpt62-astra-xhigh-ipv4-v13"
DISPATCHER = "dispatch_v13_bounded_sse_recovery"
TIMEOUTS = RequestTimeouts(read_idle_seconds=1800, total_seconds=3600)
TRANSPORT_SOURCES = frozenset(((STUDY / "a7_gpu/provider.py").as_posix(),
                               (STUDY / "a7_gpu/dispatcher.py").as_posix(),
                               (OFFLINE / "pure_llm_v3/transport_manager.py").as_posix()))


@dataclass(frozen=True)
class RunIdentity:
    suffix: str
    contract: str
    approval: str

    @property
    def name(self):
        return PREFIX + self.suffix


@dataclass(frozen=True)
class Lane:
    seed: int
    round: int
    parent: RunIdentity
    previous: RunIdentity
    previous_request: str
    previous_attempt: str
    previous_job: str
    previous_result: str
    previous_error: str
    previous_wire: str

    @property
    def run_name(self):
        return f"{PREFIX}-{self.seed}-bounded-sse-01"


V13 = RunIdentity("", "b17eee7b4d584d5522ca990b47aefdf9d628685557683c83c5bc24f20a998a85",
                  "d43ca2ba5a3ac93173466bdb638735805d0092b321dd42e7d8aac862af922410")
R2_PARENT = RunIdentity("-2357-continuation-01", "99fd5844f05bca5d7d6be41bc82dbccc9dc77d90faba61101ff5a0c490403c2b",
                        "01dce8c6c2feef6f9b042858189fddcd95f77f0678867fc9f3f3731a39134f16")
LANES = {
    157: Lane(157, 3, V13, RunIdentity("-157-extension-01",
        "afb6fbbdbdc77105ebd311131df12c50af1149054d63cef0d81319cb217e5ccd",
        "33b41c17652e0c3b3352f2e1b5f81d74f576ac79dd1dc26307ffb419f1f048bf"),
        "7c356ae49249888621cd1d8c2597be183202d517bf8bdaaa0eece8f681375145",
        "d7c4e03dc9ec42c989080cd45fa89699", "a01f306d87f7a7f2e159a9a31a83fa24f3d7f2ded0ffb46a45b91225f5deac77",
        "ce921d90144b0e43ac1f8d8b4158c3bb96cdc9109a2f4ddf2c1a88e1473adb5d", "connection_interrupted", RESPONSES_STREAM_IPV4),
    887: Lane(887, 6, V13, RunIdentity("-887-json-extension-01",
        "9224d5871a9c3b691aa8062ba918f3e4360cf5bd0c781382f6f845a905d10e14",
        "bff974d129f0b073ca8e9e9506b9887b7619f620635c79ced3fab0445c8491df"),
        "b17dd55c904c452c26f21d815288b0cec20b53f554d99e0511ac1e46b4a86f08",
        "e84dfc1886c3487ba4b2fa82f0d3ce8f", "f5f6b41270d0debe982819ce5565b31ea4e629babf28e4ba39c0a2fde2b626d3",
        "c056c0b7c2b0c20e8a6db7ec6097793d7a6bc91d8c4ee92e2b3f48e936bf0d18", "http_524", "responses_json_ipv4"),
    2357: Lane(2357, 2, R2_PARENT, RunIdentity("-2357-r2-extension-01",
        "ee77d4caa9660acc0bebd482f27493e4b74a01eae7c2e7f0806467acfa06a445",
        "9e69e76109c76d16c8f278cc4bcf647b06b1c9ef6de33a9340d22c16634b2c32"),
        "5cdd04e6256879ef419ce1bac95e7c092dfe618b96651f4c6b52ba0ebcd23ab2",
        "0ca4396d005a4aa080f952ee4f4d8826", "4dccad6a54e2f5acd02c6fea9726711299ed0b1eddd7e4924ab388ab0797a3a3",
        "6b6c6eee0e5eee33474f994fc686b43969f94b4a26d82e94ff63a487b27a1cfc", "timeout", RESPONSES_STREAM_IPV4),
    6101: Lane(6101, 1, V13, RunIdentity("-6101-ws-extension-01",
        "840cb7d237ee37a4ba643579c61dc79905f683efcd61e626505e6f94da2a23eb",
        "c8a78ca92ff78764b21ffedade86bed78c7ab1a025f33cb8be8228ec2bea7c78"),
        "684abac3e62f5d212b3e686dc0bc874524f22af2258b573375b1b7964a89c784",
        "37d804ea3c25480eb678ef3eaf51eb03", "fe27e8cce97fbc7b3ae13221eff2e0e2a08cfa55d655d233bf1ed1f849c40ea3",
        "890c72068d6b274fdb04d08c6951dec7e363f37c3a6bf79fa86e59248706636d", "websocket_interrupted", "responses_websocket_ipv4"),
}


def _archived_source(release, name, expected, inherited_release):
    path = release / name
    try:
        return file_digest(path)
    except FileNotFoundError:
        # A sparse previous release may omit unchanged scientific source. The
        # fallback must already be signed and verified in the original release.
        if path.is_symlink() or inherited_release is None:
            raise
        original, sources = inherited_release
        if sources.get(name) != expected:
            raise
        return file_digest(original / name)


def _archived(run, release, identity, *, inherited_release=None):
    record = read(run / "formal_contract.json")
    approval = read(run / "authorization.json")
    if (run.name != identity.name or record["run_directory"] != str(run)
            or digest(record) != identity.contract or digest(approval) != identity.approval
            or record["api_execution"] != SSH_API or record["gpu_slots"] != [0]):
        raise PermissionError("an archived run, approval or CUDA scope differs from its pinned record")
    model = TransportConfig(**record["profile"]["transport"])
    contract = PreparedCohortContract(record["profile"], model, record["source_sha256"], record["asset_sha256"],
                                      record["anchor_root"], tuple(record["gpu_slots"]), record["lease_root"],
                                      str(run), record["api_execution"], record["predecessor"])
    if (contract.record() != record
            or any(_archived_source(release, name, expected, inherited_release) != expected
                   for name, expected in contract.source_sha256.items())):
        raise PermissionError("an archived source release differs from its signed manifest")
    return contract, verify_authorization(contract, approval)


def _sources(root, original):
    sources = {name: file_digest(root / name) for name in original.source_sha256}
    if {name for name, actual in sources.items() if actual != original.source_sha256[name]} - TRANSPORT_SOURCES:
        raise PermissionError("the bounded release changed source outside the authorized transport path")
    for module in ("run_v13_bounded_sse_recovery", DISPATCHER, "run_v13_lane_extension", "run_v13_seed2357_extension",
                   "dispatch_prompt_direct", "dispatch_prompt_pilot", "dispatch_prompt_successor", "dispatch_v13_lane_extension"):
        name = (STUDY / "scripts" / f"{module}.py").as_posix()
        sources[name] = file_digest(root / name)
    return sources


def _failure(run, turn, contract, approval, request, index):
    started = read(turn / f"attempt_{index:02d}.started.json")
    terminal = terminal_record(turn, index)
    attempt = started["attempt_id"]
    job = read_bridge_json(run / "bridge" / "outbox" / f"{attempt}.json")
    result = read_bridge_json(run / "bridge" / "results" / f"{attempt}.json")
    bound = verify_job(job, contract_sha256=contract.digest, authorization_sha256=digest(approval),
                       wire_representation=contract.profile["wire_representation"])
    validate_result(result, job)
    failure = result["outcome"].get("failure", {})
    if (bound != (request, contract.transport, started) or started["attempt"] != index
            or any(terminal.get(key) != value for key, value in started.items())
            or terminal.get("status") not in {"retryable", "stopped"} or terminal.get("circuit_open") is not False
            or "answer_sha256" in terminal or "reconciliation_sha256" in terminal
            or terminal.get("provider_status") == "completed" or result["outcome"]["status"] != "failed"
            or terminal.get("bridge_job_sha256") != job["job_sha256"]
            or terminal.get("bridge_result_sha256") != result["result_sha256"]
            or terminal.get("error") != failure.get("code") or terminal.get("ambiguous") is not failure.get("ambiguous")
            or terminal.get("provider_dispatched") is not result["provider_dispatched"] or result["raw_response"] is not None):
        raise PermissionError("an archived attempt is not a sealed original failure without completion")
    return {"attempt": index, "attempt_id": attempt, "job_sha256": job["job_sha256"],
            "result_sha256": result["result_sha256"], "terminal_sha256": digest(terminal),
            "error": failure["code"], "provider_dispatched": result["provider_dispatched"],
            "ambiguous": failure["ambiguous"]}


def _turn(active, count):
    turn = active / "turn_00"
    if (set(active.glob("turn_*/attempt_*.started.json")) != {
            turn / f"attempt_{index:02d}.started.json" for index in range(count)}
            or any(turn.glob("*.reconciliation.json")) or any(turn.glob("*.recovery.terminal.json"))):
        raise PermissionError("the archived decision has different reservations or recovered completion evidence")
    return turn


def _request(active, turn, contract, approval, lane):
    request = read(turn / "request.json")
    scope = seed_runner._scope(contract, approval, lane.seed, False)
    grant = seed_runner._selection_grant(scope, request["request_id"])
    binding = request["binding"]
    if ((binding["seed"], binding["round"], request["logical_index"], binding["a7_formal"]) != (
            lane.seed, lane.round, 0, scope)
            or binding["packet_sha256"] != digest(read(active / AUTHORITATIVE_PACKET_FILE))
            or read(turn / "formal_request_authorization.json") != grant):
        raise PermissionError("a sealed request differs from its current scientific scope")
    return request


def _same_payload(original, previous, lane):
    old, new = original["binding"], previous["binding"]
    expected = {**original["payload"], "stream": lane.previous_wire != "responses_json_ipv4"}
    return (previous["payload"] == expected and previous["payload_sha256"] == digest(expected)
            and {k: v for k, v in old.items() if k != "a7_formal"} == {k: v for k, v in new.items() if k != "a7_formal"})


def _boundary(lane, original, original_active, previous, approval, active):
    scope = seed_runner._scope(previous, approval, lane.seed, False)
    descriptor = previous.profile["resume_boundaries"][str(lane.seed)]
    state, memory = read(active / "source.json"), read(active / "boundary_memory.json")
    if (read(active.parent / "boundary.json") != {"formal_run": scope, **descriptor}
            or tuple(descriptor[k] for k in ("round", "state_sha256", "memory_sha256")) != (
                lane.round, digest(state), digest(memory))
            or (state["seed"], state["round"], len(state["provenance"]["expected_observed"])) != (
                lane.seed, lane.round, 333 + 32 * lane.round)
            or state != read(original_active / "source.json")):
        raise PermissionError("the previous one-shot and original archive do not share the exact fitted boundary")
    if lane.parent == V13:
        transition = read(original_active / "transition.json")
        fit = original_active / transition["fit_path"]
        if (transition["checkpoint_sha256"] != state["checkpoint"]
                or transition["source_sha256"] != file_digest(original_active / "source.json")
                or transition["fit_audit_sha256"] != file_digest(fit / "fit_audit.json")
                or transition["fit_audit_sha256"] != state["provenance"]["fit_audit_sha256"]
                or file_digest(fit / "best.pt") != state["checkpoint"]
                or memory != read(seed_runner.round_directory(original_active.parent, lane.round - 1) / "memory.json")):
            raise PermissionError("checkpoint, fit audit, or saved scientific memory differs from the original transition")
        cache = Path(original.predecessor["lanes"][str(lane.seed)]["cache_directory"])
    else:
        original_descriptor = original.profile["resume_boundaries"][str(lane.seed)]
        if (read(original_active / "boundary_memory.json") != memory
                or original_descriptor != {k: descriptor[k] for k in ("round", "state_sha256", "memory_sha256")}):
            raise PermissionError("the SSE retry must reuse the original 2357 continuation boundary")
        cache = Path(original.predecessor["cache_directory"])
    expected = previous.predecessor["public_resource_receipt_sha256"]
    if file_digest(cache / "resource_receipt.json") != expected:
        raise PermissionError("the original qualified training cache differs from the signed resource receipt")
    return (seed_runner.FrozenState(**state), memory, lane.round), cache


def construct(root, original_release, original_run, previous_release, previous_run, run, lane):
    if run.name != lane.run_name or run.parent != original_run.parent or root in (original_release, previous_release):
        raise PermissionError("use the registered bounded sibling and a separate release")
    original, original_approval = _archived(original_run, original_release, lane.parent)
    previous, approval = _archived(previous_run, previous_release, lane.previous,
                                   inherited_release=(original_release, original.source_sha256))
    if (original.transport.wire_mode != RESPONSES_STREAM_IPV4 or original.transport.transport_attempts != 2
            or previous.transport.wire_mode != lane.previous_wire or previous.seed_ids != (lane.seed,)
            or (previous.transport.logical_replies, previous.transport.transport_attempts) != (1, 1)
            or previous.target_labels != 333 + 32 * (lane.round + 1)
            or previous.predecessor["parent_run_directory"] != str(original_run)
            or previous.predecessor["parent_contract_sha256"] != original.digest):
        raise PermissionError("the exact original and prior physical budgets do not match this bounded lane")
    _, original_active = earlier._uncommitted(original_run, lane)
    _, active = earlier._uncommitted(previous_run, lane)
    original_turn, turn = _turn(original_active, 2), _turn(active, 1)
    archived = _request(original_active, original_turn, original, original_approval, lane)
    spent = _request(active, turn, previous, approval, lane)
    if (not _same_payload(archived, spent, lane) or archived["request_id"] != previous.predecessor["prior_request_id"]
            or archived["payload_sha256"] != previous.predecessor["payload_sha256"]
            or spent["request_id"] != lane.previous_request or archived["payload"].get("stream") is not True):
        raise PermissionError("the latest attempt differs from the original scientific request")
    prior = [_failure(original_run, original_turn, original, original_approval, archived, i) for i in (0, 1)]
    if prior != previous.predecessor["prior_attempts"]:
        raise PermissionError("the original two failures differ from the already signed archive")
    if lane.parent == R2_PARENT:
        permit = digest(read(original_turn / "attempt_00.replay_authorization.json"))
        replay = terminal_record(original_turn, 0, replay_authorization_sha256=permit)
        _link_replay_attempt(original_turn, read(original_turn / "attempt_01.started.json"), replay, allow_create=False)
        if (permit != previous.predecessor["prior_replay_authorization_sha256"]
                or digest(read(original_turn / "attempt_01.replay_link.json")) != previous.predecessor["prior_replay_link_sha256"]):
            raise PermissionError("the earlier 2357 replay authorization or linkage changed")
    latest = _failure(previous_run, turn, previous, approval, spent, 0)
    expected = (lane.previous_attempt, lane.previous_job, lane.previous_result, lane.previous_error, True, True)
    if tuple(latest[k] for k in ("attempt_id", "job_sha256", "result_sha256", "error", "provider_dispatched", "ambiguous")) != expected:
        raise PermissionError("the prior one-attempt recovery is not the exact sealed failure")
    identity = previous.predecessor["model_identity"]
    if any(read(folder / "provider_model_lock.json") != identity for folder in (original_run, previous_run)):
        raise PermissionError("previous served model or reasoning identity drifted")
    boundary, cache = _boundary(lane, original, original_active, previous, approval, active)
    transport = replace(original.transport, timeout_seconds=TIMEOUTS.total_seconds, logical_replies=1, transport_attempts=1)
    profile = {k: v for k, v in previous.profile.items() if k != "websockets_version"}
    profile.update(transport=asdict(transport), request_timeouts=asdict(TIMEOUTS), sse_inactivity_seconds=900)
    predecessor = {
        **previous.predecessor, "dispatcher_module": DISPATCHER, "previous_run_directory": str(previous_run),
        "previous_release_directory": str(previous_release), "previous_contract_sha256": previous.digest,
        "previous_authorization_sha256": digest(approval), "previous_request_id": spent["request_id"],
        "previous_attempt": latest, "bridge_wait_seconds": TIMEOUTS.bridge_wait_seconds,
        "prior_attempts": prior, "execution_budget": earlier.previous.ONE_SHOT_BUDGET,
    }
    contract = replace(previous, profile=profile, transport=transport, source_sha256=_sources(root, original),
                       run_directory=str(run), predecessor=predecessor)
    return contract, boundary, cache, archived


def _retired(parent, previous, lane):
    for run in (parent, previous):
        earlier._uncommitted(run, lane, require_paused=True)


def execute(root, parent, previous, run, lane, contract, approval, boundary, cache, original, *, resume):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    _retired(parent, previous, lane)
    request = earlier._request(root, contract, approval, lane, boundary[0], boundary[1], original)
    if not earlier._same_science(request, original, contract.predecessor.get("authoritative_message_reconstruction")):
        raise PermissionError("the new SSE request must reproduce the original scientific body")
    with scheduler._lock(run / "coordinator.lock", busy="this bounded SSE coordinator is already running"):
        if scheduler._pause_requested(run) and not resume:
            raise PermissionError("this bounded run is paused; use resume")
        once(run / "formal_contract.json", contract.record())
        once(run / "authorization.json", approval)
        once(run / "provider_model_lock.json", contract.predecessor["model_identity"])
        once(run / "extension_request_scope.json", {"request_id": request["request_id"],
            "payload_sha256": request["payload_sha256"], "seed": lane.seed, "round": lane.round,
            "logical_index": 0, "attempt": 0})
        if resume:
            (run / "pause.request").unlink(missing_ok=True)
            (seed_runner.seed_directory(run, lane.seed) / "pause.request").unlink(missing_ok=True)
        scheduler._atomic_status(run / "coordinator_generation.json", {"id": uuid4().hex, "pid": os.getpid()})
        return _run(root, parent, previous, run, lane, contract, approval, boundary,
                    TrainingBackend(root, cache, lane.seed), original, request)


def _run(root, parent, previous, run, lane, contract, approval, boundary, backend, original, request):
    status = {"seed": lane.seed, "pid": os.getpid(), "gpu": contract.gpu_slots[0]}
    path = run / "workers" / f"seed_{lane.seed}.json"
    scheduler._atomic_status(path, {**status, "status": "initializing"})
    try:
        call, options = scheduler._provider_wiring(run, contract, approval, None, lane.seed)
        managed = options["managed_attempt"]
        managed.wait_seconds = contract.predecessor["bridge_wait_seconds"]

        def one_request(candidate, config, started):
            _retired(parent, previous, lane)
            prior = [*contract.predecessor["prior_attempts"], contract.predecessor["previous_attempt"]]
            if not earlier.previous._one_attempt(request, contract.transport, prior, candidate, config, started):
                raise TransportStop("the new scope authorizes only one exact new physical request")
            return managed(candidate, config, started)

        result = seed_runner.run_seed(
            root=root, run_directory=run, contract=contract, authorization=approval, seed=lane.seed, backend=backend,
            call=call, resume_boundary=boundary, boundary_user_message=original["payload"]["input"][1],
            transport_options={"managed_attempt": one_request},
            fit_slot=partial(scheduler.gpu_slot, run, lane.seed, status["gpu"], contract.lease_root),
        )
        if result.status == "complete":
            earlier._completion(run, contract, approval, lane, boundary, backend)
        scheduler._atomic_status(path, {**status, **asdict(result)})
        return result
    except Exception as error:
        phase = "paused" if scheduler._pause_requested(run, lane.seed) else (
            "bridge_pending" if isinstance(error, BridgePending) else "failed")
        scheduler._atomic_status(path, {**status, "status": phase, "error_type": type(error).__name__,
                                        "reason": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run", "resume"))
    parser.add_argument("--seed", type=int, choices=tuple(LANES), required=True)
    for name in ("original-release", "original-run", "previous-release", "previous-run", "run-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    original_release, parent, previous_release, previous, run = (
        path.resolve() for path in (args.original_release, args.original_run, args.previous_release,
                                    args.previous_run, args.run_dir)
    )
    lane = LANES[args.seed]
    contract, boundary, cache, original = construct(ROOT, original_release, parent, previous_release, previous, run, lane)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        parser.error("run and resume require the exact new single-attempt authorization")
    approval = verify_authorization(contract, read(args.authorization))
    result = execute(ROOT, parent, previous, run, lane, contract, approval, boundary, cache, original,
                     resume=args.action == "resume")
    print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
