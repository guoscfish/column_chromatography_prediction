"""Resume each paused V13 seed from its exact archived state through L1485."""

import argparse
from dataclasses import asdict, dataclass, replace
from functools import partial
import os
from pathlib import Path
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportStop
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import BridgePending
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, OFFLINE, STUDY, verify_authorization
from . import run_v13_bounded_sse_recovery as bounded
from .run_prepared_five_seed import read_boundary


DISPATCHER = "dispatch_v13_resilient_sse"
SOURCES_ALLOWED = bounded.TRANSPORT_SOURCES | {
    (STUDY / "a7_gpu/seed_runner.py").as_posix(), (OFFLINE / "pure_llm_v3/runner.py").as_posix(),
}
REPLAY_BUDGET = {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
ALIAS = "normalize_observed_reference_aliases"


@dataclass(frozen=True)
class Lane:
    seed: int
    round: int
    previous: bounded.RunIdentity
    previous_attempt: str
    previous_request: str
    previous_job: str
    previous_result: str
    posted: bool

    @property
    def parent(self):
        return bounded.R2_PARENT if self.seed == 2357 else bounded.V13

    @property
    def run_name(self):
        return f"{bounded.PREFIX}-{self.seed}-resilient-sse-01"


LANES = {
    157: Lane(157, 3, bounded.RunIdentity("-157-bounded-sse-01",
        "6c0f4d632fb6500e3ec9523d84bea5867790339b7184d2990e538a923b93fdd9",
        "fb37bb97b0128072e8bf42a245f6f10713ba1d016095b101a9049338a121ffd3"),
        "0d0deb7f7f6a485fa631891130cc794d", "066cc76dd0539f0ec58391dbafc143e703580cb9d8932b239984f6927a3399ee",
        "7e635b367588cba89a5a73d9518bbb4c5c6ee77b0e4cc0746121f5dfe9be1216",
        "9459d36006e790d5447526865aaaf966d9a532c29f455c87a53d1334fb3cdc76", False),
    887: Lane(887, 6, bounded.RunIdentity("-887-bounded-sse-01",
        "6f2f834a6c696e9b453b2441fa2fbfdd991c8419478c140f1a1c2fbd2cb805e5",
        "d6268d9baff64b60f5cce0f83d880a150ee9abbc004d92470655fc50bfd9c6a4"),
        "72cbf757729e435caad524b6dd39ca74", "f718a38f0f78cb2b0e939e9dff7fe21436cbc1ddbe1ad120a27ef20ab35437b4",
        "25266cf8ad03fa371bed69a703aeb08ab3184c6d5e0c0908a7e764e6344e204b",
        "26384e20d20d1b89481af9a84b213a8eab088b98cde2aeec9728a35ad517e1b7", False),
    2357: Lane(2357, 2, bounded.RunIdentity("-2357-bounded-sse-01",
        "d072e9baf728b36549dcf55db61289b5848d115b6c245f4e56109a60cebb3eea",
        "efe41b051b63a6d01eda7d9fcecdd1c55952579c880268c68237326e75e9b9cd"),
        "6072424d7c7a4bdebad50b7cfe0047b2", "29a25b4321e1355a5901aa5304dcaecaada09af15ed5e3a435cb81e51e553d9d",
        "c71e7a5a68c39f801fe17a189cdaf2d67d6e1fbedb993851567573cb43c3c2ea",
        "f690e3c20b22d63a37c0cfd6665b45ba7dc8207ff13e584672f76520dd7a4223", False),
    6101: Lane(6101, 1, bounded.RunIdentity("-6101-bounded-sse-01",
        "cc1868ce36febd76734018b6d9220f8e60ee08b133b2acf36e9a6d98460af85e",
        "2a37e4674c5700bde48f8e9783d74a25119c88f4e8de67b92b24c5ea524470a0"),
        "c9a031c8fbe040c28ec89b6b45f8a585", "a5b881a96b8c80e0d5af0372f35292ccf11a3acd817d4d72dd6aee92e2f48722",
        "7c6f3f951a9f2b9689629a1de978478c58d615b42d5f85d25ccd69e77143c49c",
        "ec2511683f99474602d99e07bd0a0b1a5e8a0cca5eec4ecfc066ec6ca98f8683", True),
    12203: Lane(12203, 4, bounded.V13,
        "9681a798adb84324ae0e90ba2718ba17", "3abf9aaefbb16493ee795217ff606ba485d0e255c0919b3b3c82a17c677c5802",
        "1c83c8fb5ad0ac74bbe2db9a6e2365281bd4718099edcb6288dc0dc40a3850d7",
        "fd5e56e15b51948b010c2d11a5a71ec58e53114fb581db578e0a23a81121dd29", True),
}
R4_FIRST = ("a824dd3d312c480ea6de02f04994e910", "72ea005cf0230e1834e2e654d328b93d5d435f6340083998651b18d119d41e99",
            "8c479421a91566bf6a25ea58331ea04508324567c53e3b21794221e17d49d667")
R4_PAYLOAD = "0bd98eeac5b9b9049b7e00d7e8e439d0d1c07e7bd419370440d0259dfc91e1f5"
R4_BOUNDARY = {"round": 4, "state_sha256": "1e2a50d6864feaca1ebdcf306a7869918d4d28e90af0b40e3c2fb8c5d72c3c37",
               "memory_sha256": "7d17980960e64cd4f335758cc977d09b6981b017a77fbe41b94979c62b5c4631",
               "user_message_sha256": "a4bbc1dae4add9be012725eb098e3ad7b2027f2126c57de0903b11ac90a020c4"}


def _sources(root, original):
    sources = {name: file_digest(root / name) for name in original.source_sha256}
    if {name for name, actual in sources.items() if actual != original.source_sha256[name]} - SOURCES_ALLOWED:
        raise PermissionError("the resilient release changed source outside the authorized transport and local alias code")
    for module in (
        "run_v13_resilient_sse", DISPATCHER, "run_v13_bounded_sse_recovery", "run_v13_lane_extension",
        "run_v13_seed2357_extension", "dispatch_v13_bounded_sse_recovery", "dispatch_v13_lane_extension",
        "dispatch_prompt_direct", "dispatch_prompt_pilot", "dispatch_prompt_successor",
    ):
        name = (STUDY / "scripts" / f"{module}.py").as_posix()
        sources[name] = file_digest(root / name)
    return sources


def _descriptor(boundary, request):
    state, memory, index = boundary
    return {"round": index, "state_sha256": digest(asdict(state)), "memory_sha256": digest(memory),
            "user_message_sha256": digest(request["payload"]["input"][1])}


def _latest_failure(lane, failure):
    expected = (lane.previous_attempt, lane.previous_job, lane.previous_result, lane.posted, lane.posted,
                "timeout" if lane.posted else "connect_error")
    if tuple(failure[key] for key in ("attempt_id", "job_sha256", "result_sha256", "provider_dispatched",
                                     "ambiguous", "error")) != expected:
        raise PermissionError("the latest exhausted result differs from its exact sealed prior failure")


def _root_12203(root, run, lane, original, approval, active, archived, attempts):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    if (archived["request_id"] != lane.previous_request or archived["payload_sha256"] != R4_PAYLOAD
            or tuple(attempts[0][k] for k in ("attempt_id", "job_sha256", "result_sha256")) != R4_FIRST
            or attempts[0]["provider_dispatched"] is not False or attempts[0]["ambiguous"] is not False
            or attempts[0]["error"] != "connect_error"):
        raise PermissionError("the exact two prior R4 attempts are unavailable")
    _latest_failure(lane, attempts[1])
    item = original.predecessor["lanes"][str(lane.seed)]
    prepared = Path(item["boundary_directory"])
    draft = read(prepared.parents[1] / "five_seed_boundary_draft.json")
    row = draft["boundaries"][str(lane.seed)]
    if digest(draft) != original.predecessor["prepared_draft_sha256"] or row["descriptor"] != (
        original.profile["resume_boundaries"][str(lane.seed)]
    ):
        raise PermissionError("the original 12203 L333 preparation changed")
    state, memory, initial = read_boundary(row, lane.seed, prepared.parents[1])
    cache = Path(item["cache_directory"])
    if file_digest(cache / "resource_receipt.json") != state.provenance["resource_receipt_sha256"]:
        raise PermissionError("the original 12203 training cache changed")
    backend = TrainingBackend(root, cache, lane.seed)
    scope = seed_runner._scope(original, approval, lane.seed, False)
    seed = seed_runner.seed_directory(run, lane.seed)
    for index in range(initial, lane.round):
        before = seed_runner.round_directory(seed, index)
        after = seed_runner.round_directory(seed, index + 1)
        state, memory = seed_runner._successor(before, state, after, scope, original.transport, original, backend)
    boundary = (state, memory, lane.round)
    if _descriptor(boundary, archived) != R4_BOUNDARY or asdict(state) != read(active / "source.json"):
        raise PermissionError("the fitted 12203 L461 checkpoint and memory differ from the exact archived decision")
    return boundary, cache


def _previous(root, original_release, parent, previous_release, previous_run, lane,
              original, approval, archived, active, attempts):
    if lane.seed == 12203:
        if previous_run != parent or previous_release != original_release:
            raise PermissionError("12203 continues directly from the original V13 fitted prefix")
        boundary, cache = _root_12203(root, parent, lane, original, approval, active, archived, attempts)
        return original, approval, boundary, cache, attempts

    previous, previous_approval = bounded._archived(
        previous_run, previous_release, lane.previous, inherited_release=(original_release, original.source_sha256),
    )
    if ((previous.transport.logical_replies, previous.transport.transport_attempts) != (1, 1)
            or previous.seed_ids != (lane.seed,) or previous.predecessor["parent_contract_sha256"] != original.digest
            or previous.predecessor["parent_run_directory"] != str(parent)
            or previous.predecessor["prior_attempts"] != attempts):
        raise PermissionError("the exact bounded predecessor does not bind the original exhausted decision")
    _, prior_active = bounded.earlier._uncommitted(previous_run, lane)
    turn = bounded._turn(prior_active, 1)
    request = bounded._request(prior_active, turn, previous, previous_approval, lane)
    if (request["request_id"] != lane.previous_request or request["payload"] != archived["payload"]
            or not bounded.earlier.previous._same_science(request, archived)):
        raise PermissionError("the latest bounded request differs from the original archived science")
    latest = bounded._failure(previous_run, turn, previous, previous_approval, request, 0)
    _latest_failure(lane, latest)
    boundary, cache = bounded._boundary(lane, original, active, previous, previous_approval, prior_active)
    return previous, previous_approval, boundary, cache, [latest]


def _same_science(request, archived):
    original = {k: v for k, v in archived["binding"].items() if k != "a7_formal"}
    current = {k: v for k, v in request["binding"].items() if k not in ("a7_formal", ALIAS)}
    return (request["binding"].get(ALIAS) is True and original == current
            and request["request_id"] != archived["request_id"] and request["logical_index"] == 0
            and request["payload"] == archived["payload"] and request["payload_sha256"] == archived["payload_sha256"])


def _budget(original, index):
    remaining = original.rounds - index
    future = remaining - 1
    logical = future * original.transport.logical_replies
    return {"first_logical_requests": remaining, "maximum_logical_requests": 1 + logical,
            "maximum_physical_attempts": 1 + logical * original.transport.transport_attempts}


def construct(root, original_release, parent, previous_release, previous_run, run, lane):
    if run.name != lane.run_name or run.parent != parent.parent or root in (original_release, previous_release):
        raise PermissionError("use the registered resilient sibling and its immutable release")
    original, approval = bounded._archived(parent, original_release, lane.parent)
    if (original.transport.wire_mode != bounded.RESPONSES_STREAM_IPV4
            or (original.transport.logical_replies, original.transport.transport_attempts) != (3, 2)
            or original.rounds != 36):
        raise PermissionError("the original V13 contract does not have the exact 36-round physical budget")
    _, active = bounded.earlier._uncommitted(parent, lane)
    turn = bounded._turn(active, 2)
    archived = bounded._request(active, turn, original, approval, lane)
    attempts = [bounded._failure(parent, turn, original, approval, archived, index) for index in (0, 1)]
    previous, prior_approval, boundary, cache, latest = _previous(
        root, original_release, parent, previous_release, previous_run, lane, original, approval, archived, active, attempts,
    )
    identity = read(parent / "provider_model_lock.json")
    if identity != read(previous_run / "provider_model_lock.json"):
        raise PermissionError("the original and latest Mac model identity records differ")
    transport = replace(original.transport, timeout_seconds=bounded.TIMEOUTS.total_seconds)
    if bounded.earlier.make_request(archived["binding"], 0, archived["payload"]["input"], transport)["payload"] != archived["payload"]:
        raise PermissionError("the new SSE transport changed the full archived scientific payload")
    profile = {
        **original.profile, "confirmation_seeds": [lane.seed], "qualification_seed": lane.seed,
        "requires_initial_qualification": False, "api_parallelism": 1, "transport": asdict(transport),
        "request_timeouts": asdict(bounded.TIMEOUTS), "sse_inactivity_seconds": 900,
        "resume_boundaries": {str(lane.seed): _descriptor(boundary, archived)},
        "ipv4_preflight_retry": True, "ipv4_readiness_windows": 3, ALIAS: True,
        "audited_future_transport_replay": True, "replay_eligible_from_round": lane.round + 1,
    }
    predecessor = {
        "dispatcher_module": DISPATCHER, "parent_run_directory": str(parent),
        "parent_release_directory": str(original_release), "parent_contract_sha256": original.digest,
        "parent_authorization_sha256": digest(approval), "prior_request_id": archived["request_id"],
        "prior_attempts": attempts, "previous_run_directory": str(previous_run),
        "previous_release_directory": str(previous_release), "previous_contract_sha256": previous.digest,
        "previous_authorization_sha256": digest(prior_approval), "previous_request_id": lane.previous_request,
        "previous_attempts": latest, "payload_sha256": archived["payload_sha256"], "model_identity": identity,
        **{key: archived["binding"][key] for key in bounded.earlier.previous.MESSAGE_HASHES},
        "seed": lane.seed, "round": lane.round, "logical_index": 0, "bridge_wait_seconds": bounded.TIMEOUTS.bridge_wait_seconds,
        "recovery_budget": REPLAY_BUDGET, "execution_budget": _budget(original, lane.round),
        "cache_directory": str(cache), "public_resource_receipt_sha256": file_digest(cache / "resource_receipt.json"),
    }
    if lane.seed != 12203:
        predecessor["previous_attempt"] = latest[0]
    contract = replace(original, profile=profile, transport=transport, source_sha256=_sources(root, original),
                       run_directory=str(run), predecessor=predecessor)
    return contract, boundary, cache, archived


def execute(root, parent, previous, run, lane, contract, approval, boundary, cache, original, *, resume):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    bounded._retired(parent, previous, lane)
    request = bounded.earlier._request(root, contract, approval, lane, boundary[0], boundary[1], original)
    if not _same_science(request, original):
        raise PermissionError("the exact authorized SSE payload differs from the original full scientific request")
    with scheduler._lock(run / "coordinator.lock", busy="this resilient SSE lane is already running"):
        if scheduler._pause_requested(run) and not resume:
            raise PermissionError("this resilient run is paused; use resume")
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

        def checked_attempt(candidate, config, started):
            bounded._retired(parent, previous, lane)
            binding = candidate["binding"]
            if binding["seed"] != lane.seed or not lane.round <= binding["round"] < contract.rounds:
                raise TransportStop("the resilient scope cannot submit a completed round or another seed")
            if binding["round"] == lane.round:
                prior = [*contract.predecessor["prior_attempts"], *contract.predecessor["previous_attempts"]]
                if not bounded.earlier.previous._one_attempt(request, contract.transport, prior, candidate, config, started):
                    raise TransportStop("only the exact first logical request and physical slot are authorized in the recovery round")
            return managed(candidate, config, started)

        result = seed_runner.run_seed(
            root=root, run_directory=run, contract=contract, authorization=approval, seed=lane.seed, backend=backend,
            call=call, resume_boundary=boundary, boundary_user_message=original["payload"]["input"][1],
            transport_options={"managed_attempt": checked_attempt},
            fit_slot=partial(scheduler.gpu_slot, run, lane.seed, status["gpu"], contract.lease_root),
        )
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
        parser.add_argument(f"--{name}", type=Path, required=not name.startswith("previous"))
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    lane = LANES[args.seed]
    if lane.seed != 12203 and (args.previous_release is None or args.previous_run is None):
        parser.error("supply the exact latest one-attempt archive for this seed")
    original_release, parent, previous_release, previous, run = (
        path.resolve() for path in (args.original_release, args.original_run, args.previous_release or args.original_release,
                                    args.previous_run or args.original_run, args.run_dir)
    )
    contract, boundary, cache, original = construct(ROOT, original_release, parent, previous_release, previous, run, lane)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        parser.error("run and resume require the exact signed resilient continuation")
    approval = verify_authorization(contract, read(args.authorization))
    result = execute(ROOT, parent, previous, run, lane, contract, approval, boundary, cache, original,
                     resume=args.action == "resume")
    print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
