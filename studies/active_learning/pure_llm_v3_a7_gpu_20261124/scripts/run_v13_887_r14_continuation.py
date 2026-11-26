"""Resume normal A7 only after the signed 887 targeted repair completes its CUDA fit."""

import argparse
from dataclasses import asdict, replace
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import parse_response
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import make_request, terminal_record
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, STUDY, verify_authorization
from . import run_v13_887_r13_cached_correction as cached
from . import run_v13_887_r13_targeted_repair as repair
from . import run_v13_lane_continuation as continuation


SEED, START = 887, 14
RUN_NAME = f"{repair.relay.bounded.PREFIX}-887-r14-continuation-01"
DISPATCHER = "dispatch_v13_887_r14_continuation"
CACHED_CONTRACT = "fea67a7dbab3a3153d224a4cdc810366f015d05d46df4343a55a2712ac6369f0"
CACHED_AUTH = "4ef97ab2a321faf84945bd4a4ee5b82a742f32342f68d4e41b121b93d9209be2"
CACHED_RELEASE = "ipv4-v13-887-r13-cached-correction-01-38267e97515ffdcc7dc468409a4ae53e6746f58fb82d5c03f5719077481e1a25"
CHECKPOINT = "571ac7aee1760bb1c73f16269342083acd30e26de6e21c4ad16125caec0eda97"
TRANSITION = "b14bb70220accd998cb9f1c5401773124fcc2c2c033146518f22a868be296c42"
STATE = "fd24737bdc63d4c8a93a7b7af81a4b573d42907a8ed5198795a9d8eaae3d6a31"
CUSTODY = "e421565b5c353fccbaf818293e45130017a479c45252b5ab8356f914d39a2b93"
LOCAL_TERMINAL = "33b35e3ea84802e402d4b7829202548f32a310b4351eca4f38cca3412cf46aa2"
LOCAL_ATTEMPT = "59b26706198b4b239f02fadb5267c13e"
NORMAL_PROMPT = "d656796e97dee76bd12bb7ce94bc3056d042e858e417759eb4108710744a594f"
CUSTODY_SHA = "500eae760f7b9b28de0e0ee13c522c2bd4f87b41e71edb0db95e271c6ca1d2e5"
FIT_AUDIT = "3f760b7834b7cc359dcc76f8f3aad61e6d8794d29f79af7968dde6522815b29e"
FREEZE = "6038bd1e11b157e833800294d75885117b5d7051f37f5d0394ba0732dce791a9"
STATE_SHA = "fa036aaf7f664d46c5f26043316a040fb6e8ebba3f65884e26d77676d08ce6cc"
MEMORY_SHA = "d56373643194525cf45302bdd83aca10a2bb774447dd32a9c16a49cdf4e24d1b"
R14_REQUEST = {
    "payload_sha256": "42556b0ac60f818003e8ba9ec24d1ceb3124edee82eb6f2734410029a69d69ca",
    "checkpoint_sha256": "571ac7aee1760bb1c73f16269342083acd30e26de6e21c4ad16125caec0eda97",
    "packet_sha256": "4c79fbe05e64bb492340eb60fffb9bfad9ec7666e07136a579ac57a51ae8fb79",
    "wire_packet_sha256": "9da7c0bcccd5846bb3d7d8893c313517ba1b61d0e1796de7183983b16289612c",
    "prompt_sha256": "0872b8ddd262b319d9a1463ca0696fa28a5afbd51da8cd3ff477729db6b08d40",
    "base_messages_sha256": "51cc88e3b104a1accc72c085002c05ff59153586d204aea531ba828e26ab2366",
    "authoritative_base_messages_sha256": "1ef8ae7081c699eaf548c18d9583d7b8d9067570a6448db4b63b4c5ec462aa9d",
    "mapping_sha256": "001022a7ff7bfe0397292898c4aa50503c6cb8625db417e3c2234970a2d458cc",
    "split_sha256": "0413fb5c356dc14b1d2c056379ef09f1925835e64fa41fa86dc12aa8703a1807",
    "state_sha256": "0167ad0a5a94a093ac239448ccef51fc85ad6a350eda4c8f50fea6ac88fcf72c",
}
BOUNDARY_BUDGET = {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
BUDGET = {"first_logical_requests": 22, "maximum_logical_requests": 64, "maximum_physical_attempts": 128}
SOURCES = tuple(STUDY / "scripts" / f"{module}.py" for module in (
    "run_v13_887_r14_continuation", DISPATCHER, "run_v13_lane_continuation", "run_v13_seed2357_continuation",
    "dispatch_v13_887_r13_targeted_repair",
))


def initial_request_identity(request):
    return {key: request["payload_sha256"] if key == "payload_sha256" else request["binding"][key] for key in R14_REQUEST}


def _original_profile(parent, descriptor):
    return {**parent.profile, "resume_boundaries": {str(SEED): descriptor}, "replay_eligible_from_round": START,
            "sealed_relay_retry_seconds": 180, "audited_future_transport_replay": True}


def _require_complete(run):
    phase = read(run / "workers" / f"seed_{SEED}.json")
    round_dir = run / f"seed_{SEED}/round_{START:02d}"
    if (phase.get("status"), phase.get("completed_rounds"), phase.get("active_labels")) != ("complete", START, 781):
        raise PermissionError("the one-shot correction must stop at its fully fitted L781 boundary")
    for name in ("source.json", "transition.json", "authorized_feedback.json", "fit_authorization.json"):
        if not (round_dir / name).is_file():
            raise PermissionError("the L781 training and reveal receipts are incomplete")
    scheduler._assert_no_old_workers(run, (SEED,))


def _completed_local(run, active, contract, approval, request, archived, custody):
    cached._require_no_bridge(run)
    turn = repair.relay.bounded._turn(active, 1)
    started = read(turn / "attempt_00.started.json")
    terminal = terminal_record(turn, 0)
    scope = seed_runner._scope(contract, approval, SEED, False)
    grant = seed_runner._selection_grant(scope, request["request_id"])
    if (request["request_id"] in (repair.REQUEST, cached.REQUEST) or read(turn / "request.json") != request
            or read(turn / "formal_request_authorization.json") != grant or read(active / "authorization.json") != grant
            or read(run / "cached_correction.json") != custody or custody.get("provider_post") is not False
            or file_digest(run / "cached_correction.json") != CUSTODY or digest(custody) != CUSTODY_SHA
            or started["attempt_id"] != LOCAL_ATTEMPT
            or file_digest(turn / "attempt_00.terminal.json") != LOCAL_TERMINAL
            or terminal.get("request_id") != request["request_id"] or terminal.get("status") != "completed"
            or terminal.get("provider_dispatched") is not False or terminal.get("usage") != {}
            or terminal.get("cached_correction_sha256") != digest(custody)
            or terminal.get("imported_from_response_id") != cached.RESPONSE
            or terminal.get("response_id") != cached.RESPONSE or terminal.get("answer_sha256") != cached.TEXT
            or terminal.get("text") != archived["text"] or any(terminal.get(k) != v for k, v in started.items())):
        raise PermissionError("the L781 state is detached from the exact zero-POST cached correction")
    return {"cached_local_attempt_id": started["attempt_id"],
            "cached_local_terminal_sha256": file_digest(turn / "attempt_00.terminal.json"),
            "cached_correction": custody, "cached_correction_sha256": digest(custody)}


def construct(root, original_root, original_run, paid_root, paid_run, cached_root, imported_run, run):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    if (root in (original_root, paid_root, cached_root) or imported_run.name != cached.RUN_NAME
            or cached_root.name != CACHED_RELEASE or run.name != RUN_NAME
            or len({original_run.parent, paid_run.parent, imported_run.parent, run.parent}) != 1):
        raise PermissionError("use the signed correction and a separate normal-Prompt sibling release")
    continuation._require_fixed_reconstruction(repair.LANE)
    _require_complete(imported_run)
    parent, parent_approval = repair.relay.bounded._archived(original_run, original_root, repair.PARENT)
    sources = {name: file_digest(root / name) for name in parent.source_sha256}
    if sources != parent.source_sha256 or parent.source_sha256[seed_runner.PROMPT_PATH.as_posix()] != NORMAL_PROMPT:
        raise PermissionError("restore every byte of the regular W10 source and Prompt before resuming")
    imported, initial, cache, original, plan, archived = cached.construct(
        cached_root, original_root, original_run, paid_root, paid_run, imported_run,
    )
    approval = continuation._extension_approval(imported, imported_run, CACHED_CONTRACT, CACHED_AUTH)
    for path in (*repair.SOURCES, *cached.SOURCES, *SOURCES):
        sources[path.as_posix()] = file_digest(root / path)
    if any(sources[path.as_posix()] != imported.source_sha256[path.as_posix()] for path in (*repair.SOURCES, *cached.SOURCES)):
        raise PermissionError("the completed repair implementation differs from the approved archive")

    directory = seed_runner.seed_directory(imported_run, SEED)
    active = seed_runner.round_directory(directory, repair.ROUND)
    following = seed_runner.round_directory(directory, START)
    initial_state, initial_memory, _ = initial
    scope = seed_runner._scope(imported, approval, SEED, False)
    user = original["payload"]["input"][1]
    if (read(directory / "boundary.json") != {"formal_run": scope, **imported.profile["resume_boundaries"][str(SEED)]}
            or read(active / "source.json") != asdict(initial_state) or read(active / "boundary_memory.json") != initial_memory):
        raise PermissionError("the repair's original L749 state or scientific memory changed")
    stage = seed_runner.pin_boundary_user_message(seed_runner._stage(cached_root, initial_state, initial_memory, scope, imported),
                                                  user, digest(user))
    request = make_request(stage.binding, 0, stage.messages, imported.transport)
    repair.verify_patch_request(request, original, imported.profile["fixed_reference_repair"], attempt=0)
    evidence = _completed_local(imported_run, active, imported, approval, request, archived, imported.predecessor["cached_correction"])
    projection = cached.project(parse_response(archived["text"]), original=plan)
    freeze = seed_runner._verify_selection(active, initial_state, scope, imported.transport, imported.batch_size, stage=stage)
    if projection != freeze["plan"] or freeze["freeze_sha256"] != FREEZE:
        raise PermissionError("the L781 boundary is detached from the actual archived correction")
    backend = TrainingBackend(root, cache, SEED)
    state, memory = seed_runner._successor(active, initial_state, following, scope, imported.transport, imported, backend)
    if ((state.seed, state.round, len(state.provenance["expected_observed"]), state.checkpoint) != (SEED, START, 781, CHECKPOINT)
            or (file_digest(following / "source.json"), file_digest(following / "transition.json")) != (STATE, TRANSITION)
            or (digest(asdict(state)), digest(memory), state.provenance["fit_audit_sha256"]) != (STATE_SHA, MEMORY_SHA, FIT_AUDIT)):
        raise PermissionError("the fitted state does not close the validated 32-point correction at L781")
    descriptor = {"round": START, "state_sha256": digest(asdict(state)), "memory_sha256": digest(memory)}
    stage = seed_runner._stage(root, state, memory, seed_runner._scope(parent, parent_approval, SEED, False), parent)
    request = make_request(stage.binding, 0, stage.messages, parent.transport)
    if initial_request_identity(request) != R14_REQUEST:
        raise PermissionError("the normal R14 packet differs from the exact fitted boundary")
    proof = {
        "r14_request": R14_REQUEST,
        "dispatcher_module": DISPATCHER, "seed": SEED, "round": START,
        "parent_run_directory": str(original_run), "parent_release_directory": str(original_root),
        "parent_contract_sha256": parent.digest, "parent_authorization_sha256": digest(parent_approval),
        "parent_seed_pause_sha256": repair.OLD_PAUSE, "extension_run_directory": str(imported_run),
        "extension_release_directory": str(cached_root), "extension_contract_sha256": imported.digest,
        "extension_authorization_sha256": digest(approval), **evidence,
        "paid_correction_run_directory": str(paid_run), "paid_correction_release_directory": str(paid_root),
        "paid_correction_contract_sha256": cached.CORRECTION.contract,
        "paid_correction_authorization_sha256": cached.CORRECTION.approval,
        "extension_transition_sha256": TRANSITION, "extension_source_sha256": STATE, "cached_correction_file_sha256": CUSTODY,
        "extension_fit_audit_sha256": state.provenance["fit_audit_sha256"],
        "extension_freeze_sha256": freeze["freeze_sha256"], "extension_projected_plan_sha256": digest(projection),
        "checkpoint_sha256": state.checkpoint, "cache_directory": str(cache),
        "public_resource_receipt_sha256": parent.predecessor["public_resource_receipt_sha256"],
        "model_identity": imported.predecessor["model_identity"], "bridge_wait_seconds": parent.predecessor["bridge_wait_seconds"],
        "recovery_budget": BOUNDARY_BUDGET, "execution_budget": BUDGET,
    }
    contract = replace(parent, profile=_original_profile(parent, descriptor), source_sha256=sources,
                       run_directory=str(run), predecessor=proof)
    return contract, (state, memory, START), backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run", "resume"))
    for name in ("original-release", "original-run", "paid-release", "paid-run", "cached-release", "cached-run", "run-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    old_root, old, paid_root, paid, cached_root, imported, run = (p.resolve() for p in (
        args.original_release, args.original_run, args.paid_release, args.paid_run,
        args.cached_release, args.cached_run, args.run_dir,
    ))
    contract, boundary, backend = construct(ROOT, old_root, old, paid_root, paid, cached_root, imported, run)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        parser.error("provide the new continuation's exact paid and CUDA authorization")
    approval = verify_authorization(contract, read(args.authorization))
    with scheduler._lock(imported / "coordinator.lock", busy="the zero-POST import still owns its worker"):
        _require_complete(imported)
        result = continuation.execute(ROOT, old, run, repair.LANE, contract, approval, boundary, backend,
                                      resume=args.action == "resume")
    print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
