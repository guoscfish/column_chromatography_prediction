"""Fit the exact reviewed 6101 R4 archived answer once, with no provider submission."""

import argparse
from dataclasses import asdict, replace
from functools import partial
import os
from pathlib import Path
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import parse_response
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.runner import _validation
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    ProviderReply, TransportStop, make_request, response_output_text, terminal_record,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import read_bridge_json, validate_result, verify_job
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, STUDY, verify_authorization

from . import run_v13_relay_retry as relay
from .run_v13_6101_cached_r2 import _project_mirrors


SEED, ROUND, TARGET = 6101, 4, 493
MODULE = "run_v13_6101_cached_r4"
OLD = relay.bounded.RunIdentity("-6101-relay-retry-08",
    "d25e49bde420072e42f30871dbd08b66b18330e5c9c3c74b6d14de54f43cc95c",
    "a03d6c829065244f62c87e3628bf54c3ae80e4dd420db3b02cb7c44aeeb70837")
RUN_NAME = f"{relay.bounded.PREFIX}-6101-cached-r4-01"
LANE = replace(relay.LANES[SEED], round=ROUND)
BUDGET = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 0}
IMPORT_BUDGET = {"maximum_logical_requests": 1, "maximum_physical_attempts": 0}
PAUSE_SHA = "f888e524590442f261821b7156caf1de057401b329993a189cc8198d48ea567a"
CHECKPOINT = "1b8181b779824faa4f292e69889cc55d98e2f4f83b0124469349282f42b9bdd6"
ATTEMPT = "c4ac95fc97414c9aa78ce5e9ffe8d4b9"
REQUEST = "611835b9380be5195601e09bf90efbbc131fdfa04b241ac514031d92c453738d"
JOB = "e958c64ad2ad557c35cd0ed53abfe7b1f1cb9921bf6ed7c9da0afc5101a098bc"
RESULT = "4d1f7cd08e6283bb10964f30fa49c3bd61c7a4899f03ba7693934ed20b2fe834"
RAW = "d4f7267158fbb7acda8b048003d7e6096e366ebb0fbbdef8f211af168056bb7e"
RESPONSE = "resp_0c7893dbeb805928016b084b21d93087d29a9ce0051335c26c"
PAYLOAD = "796adee159f5c628c18e888aa412bb46bc91d54d1210e559ab36acc2e32aaff1"
TEXT = "71bc2bfaf31f9fc318c2dac716d6afff24d0318099872d51c555b686a7ebf5c2"
RAW_PLAN = "880493450e756eb1487654b70df20b03a8f93e098297f6698ea7e688c736b9ca"
PLAN = "ac348f5ed60165e9aa7c0a8589be4a1dbe2a69644694b8261c0a4b00418f2d41"
PATHS = "9e4e35ff3080066c185268f9f3adc5e752ed72c7eeae8484e8185c96e45a04a8"
ORDER = "089314d92777b889207fd3e1fb2a42a12c049ea66834ded095fdf8c3e8582946"
REASONS = "1f381544255cafc82faf366407eb315c472cc6a2f6d8e1aa10a635ca412a2049"
FROZEN_ORDER = "27781e476a036022f8de5816008c94c9b3ceefe7aa611e0e18417728b890349d"
TERMINAL = "279c2cba1a8e758d1fabf9afdf282e0ef2f274eac879917d9b4cda94c68f8931"
DECISION = "e2318b2ce793487b7149a9577f815a1e8610d74de93b90f96dad975225b06bbb"
FAILED_REQUEST = "91d3d8cdae7459a30d7edcb1a0a980db64e67a62e3ae8eb8f5be56a0254ab311"
FAILED_PAYLOAD = "20821d863ce378581cb2d531287b73f3cc1b22548eef93e84464452965dea576"
FAILED_ATTEMPT = {"attempt": 0, "attempt_id": "c56c1f108e2e44a99c84c061251974ff",
                  "job_sha256": "c0b5f2df29d3dd2fa83687d0b3c593b3ef012a438a8eb672bbef749de015c5ff",
                  "result_sha256": "a1d7732438ffcd116e6a90c62590ca880cceb9de231d900abac938943624baaa",
                  "error": "connection_interrupted", "provider_dispatched": True, "ambiguous": True}
FAILED_TERMINAL = "108438e95b58118d79032ecc4e6906b2be5866c1a8e77925094ae421b53c3f7f"
FAILED_PROGRESS = "37346ba002baa871cacf381c005ef497ec25b373dd3b47051e32dbc306cd590f"
BOUNDARY_FILES = {
    "round_03/memory.json": "2953230055565dfa2b15e5542b228c4f6df6f2e05667aa252cce3eef49d03f9f",
    "round_04/source.json": "4ac4de887ab7b56d894389a2b8a087a32ddc450ab878331a19da3d1ec2031b47",
    "round_04/transition.json": "ec45b661e47a89c3d04123732c0d897c68fdc938c1ed167f2476dfc6fd2697e4",
    "round_04/authoritative_packet.json": "92755d8994db1301620dcf002e588dbfe9d3d3a49257f0878659330269926db2",
}


def _sealed_parent(run, *, require_pause):
    directory, active = relay._retired(run, LANE, require_pause=require_pause)
    needed = {active / f"turn_{index:02d}/attempt_00.started.json" for index in (0, 1)}
    if (set(active.glob("turn_*/attempt_*.started.json")) != needed
            or any(active.glob("turn_*/*.reconciliation.json")) or any(active.glob("turn_*/*.recovery.terminal.json"))
            or any(file_digest(directory / name) != sha for name, sha in BOUNDARY_FILES.items())
            or require_pause and file_digest(directory / "pause.request") != PAUSE_SHA):
        raise PermissionError("the retired R4 boundary or its two exact original reservations changed")
    return directory, active


def _archived_answer(run, active, previous, approval):
    scope = seed_runner._scope(previous, approval, SEED, False)
    turn, correction = (active / f"turn_{index:02d}" for index in (0, 1))
    request, retried = (read(path / "request.json") for path in (turn, correction))
    for path, original in ((turn, request), (correction, retried)):
        if read(path / "formal_request_authorization.json") != seed_runner._selection_grant(scope, original["request_id"]):
            raise PermissionError("the prior selection request authorization changed")
    job = read_bridge_json(run / "bridge/outbox" / f"{ATTEMPT}.json")
    result = validate_result(read_bridge_json(run / "bridge/results" / f"{ATTEMPT}.json"), job)
    terminal = terminal_record(turn, 0)
    reply = result["outcome"].get("reply", {})
    if (verify_job(job, contract_sha256=OLD.contract, authorization_sha256=OLD.approval,
                   wire_representation=previous.profile["wire_representation"])
            != (request, previous.transport, read(turn / "attempt_00.started.json"))
            or (request["request_id"], request["payload_sha256"], job["job_sha256"], result["result_sha256"],
                result["raw_response_sha256"], reply.get("response_id"), job["attempt_id"])
            != (REQUEST, PAYLOAD, JOB, RESULT, RAW, RESPONSE, ATTEMPT)
            or result["outcome"]["status"] != "completed" or result["provider_dispatched"] is not True
            or terminal["status"] != "completed" or terminal["provider_dispatched"] is not True
            or terminal["bridge_job_sha256"] != JOB or terminal["bridge_result_sha256"] != RESULT
            or file_digest(turn / "attempt_00.terminal.json") != TERMINAL or terminal["answer_sha256"] != TEXT
            or file_digest(turn / "decision.json") != DECISION
            or any(terminal[k] != reply[k] for k in ("text", "response_id", "served_model", "provider_request_id", "usage"))
            or (read(run / "provider_model_lock.json"), result["model_identity"]) != (previous.predecessor["model_identity"],) * 2):
        raise PermissionError("the sealed original provider response or the exact rejected decision changed")
    failed = relay.bounded._failure(run, correction, previous, approval, retried, 0)
    if ((retried["request_id"], retried["payload_sha256"], retried["logical_index"]) != (FAILED_REQUEST, FAILED_PAYLOAD, 1)
            or request["binding"] != retried["binding"] or request["payload"]["input"] != retried["payload"]["input"][:2]
            or file_digest(correction / "attempt_00.terminal.json") != FAILED_TERMINAL
            or any(failed[k] != v for k, v in FAILED_ATTEMPT.items())):
        raise PermissionError("the separate ambiguous paid correction ledger changed")
    return request, result, failed


def _validated_projection(stage, raw, decision):
    if digest(raw) != RAW or (raw.get("id"), raw.get("status")) != (RESPONSE, "completed"):
        raise PermissionError("the archived provider object changed")
    text = response_output_text(raw["output"])
    original = parse_response(text)
    if digest(text) != TEXT or digest(original) != RAW_PLAN:
        raise PermissionError("the exact original provider plan changed")
    normalized, paths = _project_mirrors(original, stage)
    before, after = _validation(stage, original), _validation(stage, normalized)
    roles = [choice["candidate_id"] for choice in normalized["choices"]]
    reasons = [choice["reason"] for choice in normalized["choices"]]
    if (decision != {"state": "rejected", "diagnosis": before.as_dict()} or before.valid
            or not after.valid or after.valid_choice_count != 32 or len(paths) != 42
            or (digest(normalized), digest(paths), digest(roles), digest(reasons),
                digest([stage.context.all_row_ids[role] for role in roles])) != (PLAN, PATHS, ORDER, REASONS, FROZEN_ORDER)
            or roles != [choice["candidate_id"] for choice in original["choices"]]
            or reasons != [choice["reason"] for choice in original["choices"]]):
        raise PermissionError("the reviewed role mirrors or the fully revalidated original selections changed")
    return normalized, paths


def construct(root, old_release, old, run):
    if run.name != RUN_NAME or run.parent != old.parent or root == old_release:
        raise PermissionError("the archival fit requires a new immutable sibling release and run")
    previous, approval = relay.bounded._archived(old, old_release, OLD)
    directory, active = _sealed_parent(old, require_pause=True)
    request, result, failed = _archived_answer(old, active, previous, approval)
    cache = Path(previous.predecessor["cache_directory"]).resolve(strict=True)
    if file_digest(cache / "resource_receipt.json") != previous.predecessor["public_resource_receipt_sha256"]:
        raise PermissionError("the signed training resource changed")
    boundary, prefix = relay._boundary(old_release, directory, active, previous, approval, LANE, cache)
    state, memory, number = boundary
    user = request["payload"]["input"][1]
    scope = seed_runner._scope(previous, approval, SEED, False)
    stage = seed_runner.pin_boundary_user_message(seed_runner._stage(old_release, state, memory, scope, previous), user, digest(user))
    if state.checkpoint != CHECKPOINT or make_request(stage.binding, 0, stage.messages, previous.transport) != request:
        raise PermissionError("the full original R4 scientific request could not be reproduced")
    normalized, paths = _validated_projection(stage, result["raw_response"], read(active / "turn_00/decision.json"))
    custody = {"provider_post": False, "attempt_id": ATTEMPT, "request_id": REQUEST, "job_sha256": JOB,
               "result_sha256": RESULT, "raw_response_sha256": RAW, "response_id": RESPONSE, "final_text_sha256": TEXT,
               "original_plan_sha256": RAW_PLAN, "normalized_plan_sha256": PLAN, "removed_paths": paths,
               "ordered_candidate_ids_sha256": ORDER, "ordered_reasons_sha256": REASONS, "frozen_order_sha256": FROZEN_ORDER,
               "archived_terminal_file_sha256": TERMINAL, "archived_decision_file_sha256": DECISION,
               "failed_correction": {**failed, "request_id": FAILED_REQUEST, "terminal_file_sha256": FAILED_TERMINAL,
                                     "progress_file_sha256": FAILED_PROGRESS},
               "previous_seed_pause_file_sha256": PAUSE_SHA, "archived_boundary_files": BOUNDARY_FILES,
               "confirmed_boundary_prefix": prefix}
    proof = {**previous.predecessor, "dispatcher_module": MODULE, "cached_selection": custody,
             "previous_run_directory": str(old), "previous_release_directory": str(old_release),
             "previous_contract_sha256": OLD.contract, "previous_authorization_sha256": OLD.approval,
             "previous_request_id": REQUEST, "execution_budget": BUDGET, "recovery_budget": IMPORT_BUDGET,
             "seed": SEED, "round": ROUND, "checkpoint_sha256": CHECKPOINT,
             "newly_authorized_boundary_physical_attempts": 0, "previous_prefix_files": prefix,
             "previous_prefix_sha256": digest(prefix)}
    proof.update({key: request["binding"][key] for key in ("packet_sha256", "wire_packet_sha256", "base_messages_sha256")})
    for superseded in ("cached_import", "relay_not_before_unix", "relay_cooldown_clock", "previous_attempts",
                       "previous_terminal_file_sha256", "previous_transport_progress_file_sha256",
                       "previous_transport_observation", "reviewed_retry_chain", "inherited_boundary_budget",
                       "authoritative_message_reconstruction"):
        proof.pop(superseded, None)
    descriptor = {"round": number, "state_sha256": digest(asdict(state)),
                  "memory_sha256": digest(memory), "user_message_sha256": digest(user)}
    transport = replace(previous.transport, logical_replies=1, transport_attempts=1)
    profile = {**previous.profile, "target_labels": TARGET, "resume_boundaries": {str(SEED): descriptor},
               "transport": asdict(transport)}
    for option in ("audited_future_transport_replay", "replay_eligible_from_round", "sealed_relay_retry_seconds"):
        profile.pop(option, None)
    sources = {name: file_digest(root / name) for name in previous.source_sha256}
    if sources != previous.source_sha256:
        raise PermissionError("the archival fit cannot change existing signed source or Prompt")
    module = (STUDY / "scripts" / f"{MODULE}.py").as_posix()
    sources[module] = file_digest(root / module)
    contract = replace(previous, profile=profile, transport=transport, source_sha256=sources,
                       run_directory=str(run), predecessor=proof)
    return contract, boundary, cache, request, result["outcome"]["reply"], normalized


def _no_provider(*args, **kwargs):
    raise TransportStop("the archival fit has no provider transport")


def _require_no_bridge(run):
    if any(path.is_file() for folder in ("outbox", "claims", "results") for path in (run / "bridge" / folder).glob("*.json")):
        raise PermissionError("a zero-POST archival fit cannot contain provider bridge attempts")


def execute(root, old, run, contract, approval, boundary, cache, archived, old_reply, plan, *, resume):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    verify_authorization(contract, approval)
    scope = seed_runner._scope(contract, approval, SEED, False)
    user = archived["payload"]["input"][1]
    stage = seed_runner.pin_boundary_user_message(seed_runner._stage(root, *boundary[:2], scope, contract), user, digest(user))
    expected = make_request(stage.binding, 0, stage.messages, contract.transport)
    custody = contract.predecessor["cached_selection"]
    if (expected["payload"] != archived["payload"] or expected["request_id"] == REQUEST or digest(plan) != PLAN
            or not _validation(stage, plan).valid or custody["provider_post"] is not False
            or contract.predecessor["execution_budget"] != BUDGET or contract.target_labels != TARGET):
        raise PermissionError("the local decision cannot use a changed scientific plan or submit a provider request")
    evidence = {"provider_dispatched": False, "cached_selection_sha256": digest(custody),
                "imported_from_response_id": RESPONSE}
    reply = replace(ProviderReply(**old_reply), text=dumps(plan), usage={}, transport_evidence=evidence)

    def local(candidate, config, started):
        _require_no_bridge(run)
        if candidate != expected or config != contract.transport or started["attempt"] != 0:
            raise TransportStop("only the exact archived decision may occupy the local slot")
        return reply

    gpu = contract.gpu_slots[0]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with scheduler._lock(old / "coordinator.lock", busy="the archived coordinator must remain retired"):
        scheduler._assert_no_old_workers(old, (SEED,))
        _sealed_parent(old, require_pause=True)
        with scheduler._lock(run / "coordinator.lock", busy="the archival fit is already active"):
            _require_no_bridge(run)
            scheduler._assert_no_old_workers(run, (SEED,))
            directory = seed_runner.seed_directory(run, SEED)
            if scheduler._pause_requested(run, SEED) and not resume:
                raise PermissionError("use resume for the paused archival fit")
            once(run / "formal_contract.json", contract.record())
            once(run / "authorization.json", approval)
            once(run / "provider_model_lock.json", contract.predecessor["model_identity"])
            once(run / "cached_response_import.json", custody)
            past = directory / "round_04/turn_00/attempt_00.terminal.json"
            if past.is_file():
                terminal = read(past)
                if ((terminal.get("status"), terminal.get("request_id"), terminal.get("answer_sha256"), terminal.get("usage"))
                        != ("completed", expected["request_id"], digest(reply.text), {})
                        or any(terminal.get(k) != v for k, v in evidence.items())):
                    raise PermissionError("the saved local decision differs from the reviewed archive")
            if resume:
                (run / "pause.request").unlink(missing_ok=True)
                (directory / "pause.request").unlink(missing_ok=True)
            scheduler._atomic_status(run / "coordinator_generation.json", {"id": uuid4().hex, "pid": os.getpid()})
            status = {"seed": SEED, "pid": os.getpid(), "gpu": gpu}
            worker = run / f"workers/seed_{SEED}.json"
            scheduler._atomic_status(worker, {**status, "status": "initializing"})
            try:
                result = seed_runner.run_seed(
                    root=root, run_directory=run, contract=contract, authorization=approval, seed=SEED,
                    backend=TrainingBackend(root, cache, SEED), call=_no_provider, resume_boundary=boundary,
                    boundary_user_message=user, transport_options={"managed_attempt": local},
                    fit_slot=partial(scheduler.gpu_slot, run, SEED, gpu, contract.lease_root),
                )
                _require_no_bridge(run)
                scheduler._atomic_status(worker, {**status, **asdict(result)})
                return result
            except Exception as error:
                phase = "paused" if scheduler._pause_requested(run, SEED) else "failed"
                scheduler._atomic_status(worker, {**status, "status": phase, "error_type": type(error).__name__})
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run", "resume"))
    for name in ("old-release", "old-run", "run-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        parser.error("run the exact archived reconstruction under PYTHONHASHSEED=0")
    os.umask(0o077)
    old_release, old, run = (p.resolve() for p in (args.old_release, args.old_run, args.run_dir))
    contract, boundary, cache, request, reply, plan = construct(ROOT, old_release, old, run)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        parser.error("the exact signed zero-POST archival fit authorization is required")
    result = execute(ROOT, old, run, contract, read(args.authorization), boundary, cache, request, reply, plan,
                     resume=args.action == "resume")
    print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
