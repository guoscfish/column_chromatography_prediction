"""Run one scoped A7 correction of the completed, rejected 2357 R11 decision."""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from functools import partial
import json
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import ResponseFormatError, parse_response
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.runner import _validation
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportStop, make_request, terminal_record
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import BridgePending, validate_result, verify_job
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAT_SUFFIX
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, STUDY, verify_authorization
from . import run_v13_relay_retry as relay
from . import run_v13_retry_successor as successor


SEED, ROUND, TARGET = 2357, 11, 717
RUN_NAME = f"{relay.bounded.PREFIX}-2357-r11-targeted-repair-01"
DISPATCHER = "dispatch_v13_2357_r11_targeted_repair"
PARENT = relay.bounded.RunIdentity("-2357-relay-retry-57", "105def65b9818ce48027ea57fee5afe8797985f0342a1d51012bcf06f19afca6", "f97d559649f6a4da095a1a4eb6f14cffaa5a5659e57c56f91781eb73fea2a08f")
LANE = SimpleNamespace(seed=SEED, round=ROUND)
ATTEMPT = "e66838b3d4734ccca71c1877017a8723"
REQUEST = "2ebe90720e10fadf251e2e5cde7e0ede0a0857c21b3a866bdd1c129047f207a1"
JOB = "a3ed54c79a05f713574a5407fa1e440a4681674c18e7625a33d91bf077a7b07f"
RESULT = "57c8bb42fa400e9e287d79c1057d00334a27f3f54be90b8678a931abba6c77f5"
RAW = "3f947342770c0e1ac07331cf7488349327fb3006b073378acd7e101ea7f4ac1d"
ANSWER = "09423a9f1331f41139caf986ec8cbf744ed739a1aa748a4bac7e450cdb22e062"
PLAN = "ed95e10ae80606fa0adab1615d2d359d456c4286af879e70111135133a5874a6"
ORDER = "84bf76d485f843468a58dffb53528d2530b77de616cc5fb5e8fe8112eab7d5e3"
RESPONSE = "resp_09ee7467ed6a394e016b094dc54ad487d287e212a56ab8db1a"
PAYLOAD = "bebb62d54a05af78db26d5f190841f6d5a68239dc9b89a7ada26c19aafa9282e"
OLD_PAUSE = "7d663284735091b737e673082f3ddffaad3067cb7aae44512cbac55df20416b5"
OLD_WORKER = "0702b584b6a5e6be16b378fb660966e3c56db848ea55b3a570ed08cd5d557bdf"
OLD_TERMINAL = "5864da7def00ccf41a6dca462d1de710b9b41e11341a33d13e13e3fad0b2efc4"
OLD_DECISION = "697a8242b714c52fb0cb2012b5c872c43644e94b44f7a7830720e60cdcc289a6"
OLD_DECISION_SHA256 = "79ededaa3c357d57a58b6f1ae05361a52fed80a278671aa966476ba8b67edc7e"
CHECKPOINT = "4ebe85a88fac11bdb21e9b75a9b0617db4fda6233d2027549b588942661f7344"
ORIGINAL_JOB_FILE = "897150049e81c71aa11392d4a5b6d9ae6bff2c7a8db11686e20baff62c00f9fc"
ORIGINAL_RESULT_FILE = "b94e2d70dded2a02e5cc48cdd93a053aff97dd7a7bb334b6c2935da425c97509"
ORIGINAL_CAPTURE_FILE = "7423383c92f81326310a27b0bd297b663f8edaca6f789b8d383d178773e0122a"
ORIGINAL_RAW_FILE = "4508ea86f363a1e6f95f7cadecaaf5e2e44fbc619f0edb0e0a623fc818d574fb"
ORIGINAL_ACK_FILE = "7e8abfd712041a8f66a39513afeacb6e421f9eca4e6a79fb9475ae3100dfc540"
BOUNDARY_FILES = {'boundary.json': 'bd178217e10dbe879a997ecd58ab77960b000cffda755028cd427e033c8a0174',
 'round_11/authoritative_packet.json': 'd33d49a9a9cee6ea902efb2a30747e0cc76dcb1ef0b3e268835ad2e734e2f733',
 'round_11/boundary_memory.json': '0925bd733aeda54ff6d551d327a215c713c7d801448a1d273e8848709b100125',
 'round_11/boundary_user_message.json': '0fabad82858816d994585630013453ed19d61e7ee79895c94c57d09eceb8ac24',
 'round_11/source.json': '89e83649f38ef6645e8fdcdfc87d57dcfcaef89c104ceae250f9581ea71c42ab',
 'round_11/turn_00/attempt_00.started.json': '920760a7700e79c084c875cdb045d5421185e9850724e40992c975d727da2a66',
 'round_11/turn_00/request.json': 'b239aef2eac6b5b2d06ebead9622e52dff63f0e759b3f4ae8e252ab09afa9584'}
OLD_HISTORY_SHA256 = "3b7b80caaf653ed8a714c9036f56956684b992a591765cb76e81c26854904117"
OLD_PREFIX_SHA256 = "b7b050c8e34bc167ffaa16498d1f6729510c6c4561ad85e899f1838dfe8cacc2"
CHOICE_INDEX = 31
CANDIDATE, MOLECULE, INVALID_HYPOTHESIS = "C-2733", "M-0115", "H-MethoxyKetoneRise"
REJECTION = {'code': 'unknown_hypothesis_reference', 'hypothesis_id': 'H-MethoxyKetoneRise', 'path': '/choices/31/hypothesis_id'}
DATA_BINDINGS = ("seed", "round", "arm", "schema_version", "prompt_version", "checkpoint_sha256", "packet_sha256",
                 "wire_packet_sha256", "wire_representation", "mapping_sha256", "split_sha256", "state_sha256")
BUDGET = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 1}
SOURCES = tuple(STUDY / "scripts" / f"{module}.py" for module in ("run_v13_2357_r11_targeted_repair", DISPATCHER))


def guide():
    return (
        "Scoped correction of one optional citation in the existing 32-choice A7 batch. The final choice "
        "C-2733 (M-0115) cited H-MethoxyKetoneRise from archived memory. Only the active hypothesis collection "
        "can be used for this field, and its unrelated entry is not a substitute. Keep the choice, reason, "
        "observed evidence, and all hypothesis memory exactly as provided. Use null for its unsupported citation. "
        'Respond with one JSON object containing exactly {"candidate_id":"C-2733","hypothesis_id":null}. '
        "Do not select candidates or include any other output."
    )


def patch_system(original):
    system = original["payload"]["input"][0]["content"]
    if not system.endswith(FORMAT_SUFFIX):
        raise PermissionError("the completed request is missing its exact compact wire guide")
    return system.removesuffix(FORMAT_SUFFIX) + "\n\n" + guide()


def _ordered(plan):
    ids = [choice["candidate_id"] for choice in plan["choices"]]
    if (len(ids), len(set(ids)), digest(ids)) != (32, 32, ORDER):
        raise PermissionError("the cached batch differs from the 32 ordered provider choices")
    return ids


def _original_plan(text):
    plan = parse_response(text)
    _ordered(plan)
    choice = plan["choices"][CHOICE_INDEX]
    if (digest(text), digest(plan), choice.get("candidate_id"), choice.get("molecule_id"), choice.get("hypothesis_id")) != (
            ANSWER, PLAN, CANDIDATE, MOLECULE, INVALID_HYPOTHESIS):
        raise PermissionError("the original answer or sole rejected hypothesis citation changed")
    return plan


def project_reply(patch, *, original):
    if type(patch) is not dict or patch != {"candidate_id": CANDIDATE, "hypothesis_id": None}:
        raise ResponseFormatError("the sole authorized repair removes the optional rank-32 hypothesis citation")
    if digest(original) != PLAN:
        raise PermissionError("the archived 2357 provider plan changed")
    plan = deepcopy(original)
    plan["choices"][CHOICE_INDEX]["hypothesis_id"] = patch["hypothesis_id"]
    _ordered(plan)
    return plan


def _archive_job(job, receipt, wire_representation):
    request, transport, started = verify_job(job, contract_sha256=PARENT.contract, authorization_sha256=PARENT.approval,
                                              wire_representation=wire_representation)
    result = validate_result(receipt, job)
    if ((request["request_id"], request["payload_sha256"], started["attempt_id"], started["attempt"],
         job["job_sha256"], result["result_sha256"], result["raw_response_sha256"], result["provider_dispatched"])
            != (REQUEST, PAYLOAD, ATTEMPT, 0, JOB, RESULT, RAW, True)
            or (request["binding"]["seed"], request["binding"]["round"], request["logical_index"]) != (SEED, ROUND, 0)
            or result["outcome"]["status"] != "completed"):
        raise PermissionError("the archived provider and bridge receipts differ from the exact completed request")
    reply = result["outcome"]["reply"]
    if (reply["response_id"], reply["served_model"], result["raw_response"].get("reasoning", {}).get("effort")) != (
            RESPONSE, transport.served_model, transport.reasoning_effort):
        raise PermissionError("the archived response model or identity changed")
    return request, transport, reply, _original_plan(reply["text"])


def _metadata(original):
    return {
        "original_request_id": REQUEST, "original_attempt_id": ATTEMPT, "original_raw_sha256": RAW,
        "original_answer_sha256": ANSWER, "original_result_sha256": RESULT, "original_plan_sha256": PLAN,
        "ordered_candidate_ids_sha256": ORDER, "repair_index": CHOICE_INDEX, "field": "hypothesis_id",
        "candidate_id": CANDIDATE, "old_hypothesis_id": INVALID_HYPOTHESIS, "required_hypothesis_id": None,
        "starting_checkpoint_sha256": CHECKPOINT,
        "old_decision_file_sha256": OLD_DECISION, "original_payload_sha256": original["payload_sha256"],
    }


def verify_patch_request(request, original, metadata, *, attempt):
    system = patch_system(original)
    expected = {**original["payload"], "input": [{"role": "system", "content": system + FORMAT_SUFFIX},
                                                    original["payload"]["input"][1]]}
    binding = request["binding"]
    if (type(attempt) is not int or attempt != 0 or request["logical_index"] != 0 or request["payload"] != expected
            or request["request_id"] == REQUEST or binding["prompt_sha256"] != digest(system)
            or binding.get("fixed_reference_repair") != metadata
            or "normalize_observed_reference_aliases" in binding
            or any(binding[field] != original["binding"][field] for field in DATA_BINDINGS)):
        raise PermissionError("the correction may add only the signed guide to the completed request's input")


def construct(root, old_root, old, run):
    if root == old_root or run.name != RUN_NAME or run.parent != old.parent:
        raise PermissionError("use the distinct signed correction release and sibling run")
    parent, approval = relay.bounded._archived(old, old_root, PARENT)
    archive_scope = successor.SCOPES[(SEED, 57)]
    old_proof = parent.predecessor
    expected_lineage = successor.fitted_reviewed_predecessor(archive_scope)
    if (old_proof.get(successor.FITTED_REVIEWED_PREDECESSOR_FIELD) != expected_lineage
            or digest(old_proof.get("reviewed_retry_chain")) != OLD_HISTORY_SHA256
            or old_proof.get("previous_prefix_sha256") != OLD_PREFIX_SHA256
            or digest(old_proof.get("previous_prefix_files")) != OLD_PREFIX_SHA256
            or parent.profile.get("response_headers_seconds") != 300
            or parent.profile.get("sse_inactivity_seconds") != 900):
        raise PermissionError("the signed 2357 R11 fitted lineage or transport settings changed")
    chain = successor._archives(Path(old_proof["previous_run_directory"]),
                                Path(old_proof["previous_release_directory"]), archive_scope)
    successor._verify_w57_fitted_predecessor(chain, expected_lineage)
    directory, active = relay._retired(old, LANE, require_pause=True)
    turn = relay.bounded._turn(active, 1)
    if (file_digest(directory / "pause.request"), file_digest(turn / "attempt_00.terminal.json"),
        file_digest(turn / "decision.json"), file_digest(old / f"workers/seed_{SEED}.json")) != (
            OLD_PAUSE, OLD_TERMINAL, OLD_DECISION, OLD_WORKER):
        raise PermissionError("the retired 2357 pause, terminal or decision receipt changed")
    if any(file_digest(directory / path) != sha for path, sha in BOUNDARY_FILES.items()):
        raise PermissionError("the verified L685 boundary or complete packet changed")
    original = relay.bounded._request(active, turn, parent, approval, LANE)
    if any({path.name for path in (old / "bridge" / folder).glob("*.json")} != {f"{ATTEMPT}.json"}
           for folder in ("outbox", "claims", "results")):
        raise PermissionError("the previous one-attempt mailbox changed")
    job_path = old / "bridge/outbox" / f"{ATTEMPT}.json"
    result_path = old / "bridge/results" / f"{ATTEMPT}.json"
    if (file_digest(job_path), file_digest(result_path)) != (ORIGINAL_JOB_FILE, ORIGINAL_RESULT_FILE):
        raise PermissionError("the original bridge bytes differ from the Mac custody records")
    outbox, result = read(job_path), read(result_path)
    archived, transport, reply, plan = _archive_job(outbox, result, parent.profile["wire_representation"])
    terminal = terminal_record(turn, 0)
    started = read(turn / "attempt_00.started.json")
    scope = seed_runner._scope(parent, approval, SEED, False)
    model_lock = read(old / "provider_model_lock.json")
    if (archived != original or transport != parent.transport or outbox["started"] != started
            or read(active / "authorization.json") != seed_runner._selection_grant(scope, REQUEST)
            or terminal["status"] != "completed" or terminal.get("provider_status") != "completed"
            or terminal.get("bridge_job_sha256") != JOB or terminal.get("bridge_result_sha256") != RESULT
            or terminal.get("answer_sha256") != ANSWER or terminal.get("text") != reply["text"]
            or terminal.get("response_id") != RESPONSE or any(terminal.get(k) != v for k, v in started.items())
            or model_lock != result["model_identity"] or model_lock != parent.predecessor["model_identity"]):
        raise PermissionError("the rejected boundary lacks exact completed provider custody")
    cache = Path(parent.predecessor["cache_directory"]).resolve(strict=True)
    cache_sha = parent.predecessor["public_resource_receipt_sha256"]
    if file_digest(cache / "resource_receipt.json") != cache_sha:
        raise PermissionError("the signed public CUDA cache changed")
    boundary, prefix = relay._boundary(old_root, directory, active, parent, approval, LANE, cache)
    state, memory, _ = boundary
    user = original["payload"]["input"][1]
    stage = seed_runner.pin_boundary_user_message(seed_runner._stage(old_root, state, memory, scope, parent), user, digest(user))
    host_field = relay.ALIAS
    archived_stage = replace(stage, binding={**stage.binding, host_field: original["binding"][host_field]})
    if make_request(archived_stage.binding, 0, archived_stage.messages, transport) != original:
        raise PermissionError("the old request differs from the archived L685 packet")
    evaluation = _validation(stage, plan)
    diagnosis = {"state": "rejected", "diagnosis": evaluation.as_dict()}
    context_memory = json.loads(user["content"])["memory"]
    active_ids = {item["persistent_hypothesis_id"] for item in context_memory["hypotheses"]}
    archived_ids = {item["persistent_hypothesis_id"] for item in context_memory["archived_hypotheses"]}
    if (state.checkpoint != CHECKPOINT or len(state.provenance["expected_observed"]) != 685
            or digest(diagnosis) != OLD_DECISION_SHA256 or read(turn / "decision.json") != diagnosis
            or evaluation.valid or evaluation.valid_choice_count != 31 or list(evaluation.errors) != [REJECTION]
            or INVALID_HYPOTHESIS not in archived_ids or INVALID_HYPOTHESIS in active_ids
            or not _validation(stage, project_reply({"candidate_id": CANDIDATE, "hypothesis_id": None}, original=plan)).valid):
        raise PermissionError("the archived citation is not the only invalid field in the exact L685 decision")
    prompt = seed_runner.PROMPT_PATH
    if (root / prompt).read_text().strip() != patch_system(original):
        raise PermissionError("the signed correction release may apply only the exact scoped guide")
    sources = {name: file_digest(root / name) for name in parent.source_sha256}
    if any(sources[name] != sha for name, sha in parent.source_sha256.items() if name != prompt.as_posix()):
        raise PermissionError("the correction release changed previously signed non-prompt sources")
    sources.update({path.as_posix(): file_digest(root / path) for path in SOURCES})
    metadata = _metadata(original)
    one = replace(transport, logical_replies=1, transport_attempts=1)
    profile = {**parent.profile, "target_labels": TARGET, "transport": asdict(one), "fixed_reference_repair": metadata}
    for key in ("audited_future_transport_replay", "replay_eligible_from_round", "sealed_relay_retry_seconds",
                "normalize_observed_reference_aliases"):
        profile.pop(key, None)
    previous_evidence = parent.predecessor
    proof = {"dispatcher_module": DISPATCHER, "seed": SEED, "round": ROUND, "logical_index": 0,
             "previous_run_directory": str(old), "previous_release_directory": str(old_root),
             "previous_contract_sha256": PARENT.contract, "previous_authorization_sha256": PARENT.approval,
             "previous_request_id": REQUEST, "model_identity": model_lock, "cache_directory": str(cache),
             "public_resource_receipt_sha256": cache_sha, "bridge_wait_seconds": previous_evidence["bridge_wait_seconds"],
             "previous_prefix_files": prefix, "previous_prefix_sha256": digest(prefix),
             "previous_seed_pause_file_sha256": OLD_PAUSE, "previous_terminal_file_sha256": OLD_TERMINAL,
             "previous_worker_file_sha256": OLD_WORKER, "old_boundary_files": BOUNDARY_FILES,
             "original_rejection_sha256": digest(diagnosis), "fixed_reference_repair_sha256": digest(metadata),
             "prior_response_reposted": False, "execution_budget": BUDGET,
             "reviewed_fitted_predecessor": expected_lineage, "reviewed_retry_chain": old_proof["reviewed_retry_chain"],
             "original_retry_history_sha256": OLD_HISTORY_SHA256,
             "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}}
    contract = replace(parent, profile=profile, transport=one, run_directory=str(run), predecessor=proof, source_sha256=sources)
    return contract, boundary, cache, original, plan


def execute(root, old, run, contract, approval, boundary, cache, original, plan, *, resume=False):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    approval = verify_authorization(contract, approval)
    user = original["payload"]["input"][1]
    scope = seed_runner._scope(contract, approval, SEED, False)
    stage = seed_runner.pin_boundary_user_message(seed_runner._stage(root, *boundary[:2], scope, contract), user, digest(user))
    expected = make_request(stage.binding, 0, stage.messages, contract.transport)
    metadata = contract.profile["fixed_reference_repair"]
    verify_patch_request(expected, original, metadata, attempt=0)
    with scheduler._lock(old / "coordinator.lock", busy="the answered predecessor must remain retired"):
        scheduler._assert_no_old_workers(old, (SEED,))
        original_seed, _ = relay._retired(old, LANE, require_pause=True)
        if file_digest(original_seed / "pause.request") != OLD_PAUSE:
            raise PermissionError("the retired 2357 worker fence changed")
        with scheduler._lock(run / "coordinator.lock", busy="the targeted correction coordinator is already active"):
            scheduler._assert_no_old_workers(run, (SEED,))
            directory = seed_runner.seed_directory(run, SEED)
            if scheduler._pause_requested(run, SEED) and not resume:
                raise PermissionError("the targeted correction is paused; use resume")
            once(run / "formal_contract.json", contract.record())
            once(run / "authorization.json", approval)
            once(run / "provider_model_lock.json", contract.predecessor["model_identity"])
            once(run / "fixed_reference_repair.json", metadata)
            once(run / "canonical_repair_plan.json", plan)
            if resume:
                (run / "pause.request").unlink(missing_ok=True)
                (directory / "pause.request").unlink(missing_ok=True)
            scheduler._atomic_status(run / "coordinator_generation.json", {"id": uuid4().hex, "pid": os.getpid()})
            call, options = scheduler._provider_wiring(run, contract, approval, None, SEED)
            bridge = options["managed_attempt"]

            def one_request(candidate, config, started):
                if candidate != expected or config != contract.transport or started["attempt"] != 0:
                    raise TransportStop("the correction authorizes only one exact new request")
                return bridge(candidate, config, started)

            gpu = contract.gpu_slots[0]
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            status = {"seed": SEED, "pid": os.getpid(), "gpu": gpu}
            worker = run / f"workers/seed_{SEED}.json"
            scheduler._atomic_status(worker, {**status, "status": "initializing"})
            try:
                progress = seed_runner.run_seed(root=root, run_directory=run, contract=contract, authorization=approval, seed=SEED,
                    backend=TrainingBackend(root, cache, SEED), call=call, resume_boundary=boundary, boundary_user_message=user,
                    transport_options={"managed_attempt": one_request}, response_projection=partial(project_reply, original=plan),
                    fit_slot=partial(scheduler.gpu_slot, run, SEED, gpu, contract.lease_root))
                scheduler._atomic_status(worker, {**status, **asdict(progress)})
            except Exception as error:
                phase = "paused" if scheduler._pause_requested(run, SEED) else (
                    "bridge_pending" if isinstance(error, BridgePending) else "failed")
                scheduler._atomic_status(worker, {**status, "status": phase, "error_type": type(error).__name__, "reason": str(error)})
                raise
            return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prompt", "contract", "run", "resume"))
    parser.add_argument("--old-release", type=Path, required=True)
    parser.add_argument("--old-run", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    if args.action == "prompt":
        print((args.old_release / seed_runner.PROMPT_PATH).read_text().strip() + "\n\n" + guide())
        return
    if args.old_run is None or args.run_dir is None:
        parser.error("the old and new run paths are required")
    old, run = args.old_run.resolve(), args.run_dir.resolve()
    contract, boundary, cache, original, plan = construct(ROOT, args.old_release.resolve(), old, run)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        parser.error("an exact separate signed authorization is required")
    result = execute(ROOT, old, run, contract, read(args.authorization), boundary, cache, original, plan,
                     resume=args.action == "resume")
    print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
