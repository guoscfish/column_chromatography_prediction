"""Run one scoped A7 correction of the completed, rejected 887 R32 decision."""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from functools import partial
import os
from pathlib import Path
import re
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


SEED, ROUND, TARGET = 887, 32, 1389
RUN_NAME = f"{relay.bounded.PREFIX}-887-r32-targeted-repair-01"
DISPATCHER = "dispatch_v13_887_r32_targeted_repair"
PARENT = relay.bounded.RunIdentity(
    "-887-relay-retry-11", "a25b339b3663b90ca6932d106cf03c34ef21799784a1eb303dd8fb64f6829417",
    "7a458fa0066f4a6f47a01f26d1f1759cb14a69635e9c50de9d6a1369bef0b784",
)
LANE = SimpleNamespace(seed=SEED, round=ROUND)
ATTEMPT = "bafc2f8bfad14ca89af0b79a79e12064"
REQUEST = "82b2a0a662af10665cf17cbd0a8924a4c5838ac3b0e9a4672888c2e42e50be81"
JOB = "fe964ccc69c2849b04caf95c1160a1cc9ccef07f02f79a888389be8ec96b71ae"
RESULT = "2be1fc78b8bd58d3f0eefc28fbb28eb03200d93668c2305652b76ae32cb57c69"
RAW = "f25ad4db3a50a24740322cdc62237a324cec9fe441f4ef2fb2ebff55aea690ac"
ANSWER = "f0208d5f9c79a0bf51addefc9527bb2090c20ad634a0a3f8fc238726521b3cec"
PLAN = "269e572064f24d18f3e29c65d556528f8ab7bae281e1aa42cb2d0d431f3e9396"
ORDER = "d87b81dfa15bd37b6c199dfde43faa0bf3bc0eabe7f4a99bbb7476aae67104c2"
RESPONSE = "resp_ft_c57cc6230356493ebd5da65b44652d7f"
PAYLOAD = "302b17f2672c1cb477425d7aaa8ec077e443c07f6b4015a4499fb54b5e5d9b65"
OLD_PAUSE = "a6b52c37675de268e7e7f2c4e1635d1e37d446c4a7ea29984b62e825efdc2c7a"
OLD_WORKER = "a821c7be21f22f2c37d8fecdfbaa408322fe200221461b1966567315fa35e408"
OLD_TERMINAL = "631109e6c9adafb0521cf495836a87c5660e88466336f3854f0e55331d32e306"
OLD_DECISION = "72c96738f1829101c021a7ef2d6746a6a29795bcdae739cb3e336986a38897e4"
OLD_DECISION_SHA256 = "24da2fc1a35f67ad0eed6d0424681c7048c157c759675a46177cace9497d073d"
CHECKPOINT = "ac23374d4dd7f7276053d0d333707ccc03862aae40c7c419db01ac40806100a7"
ORIGINAL_JOB_FILE = "a6a742b312df327b705bdb89f31f798b7c681786f4ec115c7c4438a8a6fcb86c"
ORIGINAL_RESULT_FILE = "47242366342c2626881bd236e8ab4d7b06d952bfe12a07266b0a8428e2e98238"
ORIGINAL_CAPTURE_FILE = "26fa3ff9794705bfdfc3254fb32e7c1b4665714a791f785e107920666e2471a3"
ORIGINAL_RAW_FILE = "6e0d5e52a921cadd720d93c0fe8170e167ba4563fe5d7ab00ee478209f866169"
ORIGINAL_ACK_FILE = "ef13d6b0bc1ed65dae7c4a7836bd8379cf52fa2a30d77df5f4f0960b19d118cb"
BOUNDARY_FILES = {
    "boundary.json": "6444b5ce97b4aa9707a3a73cf978468df3ebbd0d4a8b6a6b4e7d5e3e5c7807b2",
    "round_32/source.json": "35b8b5f750460dd39c9fd98e70b1e9ae167f25889d438d2720c7d9745aa1e9d3",
    "round_32/boundary_memory.json": "ec18fb358a4bbd94dc384ad0671f568e9be700ee1d379f8cc16920167057ca3c",
    "round_32/boundary_user_message.json": "9a50a382beba7e6c639cf32e42a4ccdecdf2a2e4f0585c4833ebae4fd1422fe4",
    "round_32/authoritative_packet.json": "8c9099c4600386baae1435633754edccf14291839d59fa252d997d1431357fda",
}
CHOICE_INDEX, REFERENCE_INDEX = 22, 0
CANDIDATE, MOLECULE, OBSERVED = "C-1591", "M-0010", "O-1261"
PRESERVED_REASON = ("The two preparations already give similar finishes at 2/1 and 20/1, "
                    "providing context for transfer across eluents.")
UNCHANGED_REFERENCES = [
    {"reference_id": "O-2975", "reference_kind": "observed", "relationship": "single_factor_contrast", "changed_axis": "PE/EA"},
    {"reference_id": "O-2573", "reference_kind": "observed", "relationship": "condition_response"},
]
DATA_BINDINGS = ("seed", "round", "arm", "schema_version", "prompt_version", "checkpoint_sha256", "packet_sha256",
                 "wire_packet_sha256", "wire_representation", "mapping_sha256", "split_sha256", "state_sha256")
BUDGET = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 1}
SOURCES = tuple(STUDY / "scripts" / f"{module}.py" for module in ("run_v13_887_r32_targeted_repair", DISPATCHER))
WRONG_REFERENCE = {"reference_id": OBSERVED, "reference_kind": "observed", "relationship": "multifactor_contrast"}
REQUIRED_REFERENCE = {
    "reference_id": OBSERVED, "reference_kind": "observed", "relationship": "single_factor_contrast",
    "changed_axis": "loading_solvent",
}
REJECTION = {"code": "multifactor_mismatch", "path": "/choices/22/references/0/relationship",
             "actual_changed_axes": ["loading_solvent"]}


def guide():
    return (
        "Scoped correction for the existing 32-choice A7 batch. Keep every candidate ID, its order, and every "
        "choice field except one relationship and its associated reason. C-1591 (M-0010) and observed O-1261 "
        "(M-0010) are both at neat EA (PE/EA 0/1). Their only experimental difference is loading solvent "
        "(C-1591 DCM, O-1261 PE). Retain the other two references and every hypothesis. Output one JSON object "
        "with exactly candidate_id, reference_index, reference, and reason. Set candidate_id to C-1591 and "
        "reference_index to 0. Set reference to "
        '{"reference_id":"O-1261","reference_kind":"observed","relationship":"single_factor_contrast",'
        '"changed_axis":"loading_solvent"}. '
        "For reason, replace VERB with assess, examine, probe, test, or evaluate in this two-sentence form: "
        "Compare the DCM-loaded C-1591 with the PE-loaded O-1261 under neat EA to VERB the effect of loading solvent. "
        f"{PRESERVED_REASON} "
        "The single scoped JSON object is the entire output."
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
    if (digest(text), digest(plan), choice.get("candidate_id"), choice.get("molecule_id"), choice["references"]) != (
            ANSWER, PLAN, CANDIDATE, MOLECULE, [WRONG_REFERENCE, *UNCHANGED_REFERENCES]):
        raise PermissionError("the full original answer or the sole rejected relationship changed")
    return plan


def _checked_reason(reason):
    return type(reason) is str and re.fullmatch(
        r"Compare the DCM-loaded C-1591 with the PE-loaded O-1261 under neat EA to "
        r"(?:assess|examine|probe|test|evaluate) the effect of loading solvent\. " + re.escape(PRESERVED_REASON),
        reason,
    ) is not None


def project_reply(patch, *, original):
    if (type(patch) is not dict or set(patch) != {"candidate_id", "reference_index", "reference", "reason"}
            or patch["candidate_id"] != CANDIDATE or type(patch["reference_index"]) is not int
            or patch["reference_index"] != REFERENCE_INDEX or patch["reference"] != REQUIRED_REFERENCE
            or not _checked_reason(patch["reason"])):
        raise ResponseFormatError("response is outside the sole affected 887 reference and reason")
    if digest(original) != PLAN:
        raise PermissionError("the cached 887 provider plan changed")
    plan = deepcopy(original)
    choice = plan["choices"][CHOICE_INDEX]
    choice["references"][REFERENCE_INDEX] = deepcopy(patch["reference"])
    choice["reason"] = patch["reason"]
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
        "ordered_candidate_ids_sha256": ORDER, "repair_index": CHOICE_INDEX, "reference_index": REFERENCE_INDEX,
        "candidate_id": CANDIDATE, "reference_id": OBSERVED, "relationship": REQUIRED_REFERENCE["relationship"],
        "changed_axis": REQUIRED_REFERENCE["changed_axis"], "starting_checkpoint_sha256": CHECKPOINT,
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
    directory, active = relay._retired(old, LANE, require_pause=True)
    turn = relay.bounded._turn(active, 1)
    if (file_digest(directory / "pause.request"), file_digest(turn / "attempt_00.terminal.json"),
        file_digest(turn / "decision.json"), file_digest(old / f"workers/seed_{SEED}.json")) != (
            OLD_PAUSE, OLD_TERMINAL, OLD_DECISION, OLD_WORKER):
        raise PermissionError("the retired 887 pause, terminal or decision receipt changed")
    if any(file_digest(directory / path) != sha for path, sha in BOUNDARY_FILES.items()):
        raise PermissionError("the verified L1357 boundary or complete packet changed")
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
        raise PermissionError("the old request differs from the archived L1357 packet")
    evaluation = _validation(stage, plan)
    diagnosis = {"state": "rejected", "diagnosis": evaluation.as_dict()}
    candidate, observed = stage.context.candidates[CANDIDATE], stage.context.observed[OBSERVED]
    if (state.checkpoint != CHECKPOINT or len(state.provenance["expected_observed"]) != 1357
            or digest(diagnosis) != OLD_DECISION_SHA256 or read(turn / "decision.json") != diagnosis
            or evaluation.valid or evaluation.valid_choice_count != 31 or list(evaluation.errors) != [REJECTION]
            or candidate["molecule_id"] != MOLECULE or observed["molecule_id"] != MOLECULE
            or candidate["conditions"]["loading_solvent"] != "DCM" or observed["conditions"]["loading_solvent"] != "PE"
            or candidate["conditions"]["PE/EA"] != "0/1" or observed["conditions"]["PE/EA"] != "0/1"):
        raise PermissionError("the rejection or the authoritative evidence differs from the exact repair scope")
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
            raise PermissionError("the retired 887 worker fence changed")
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
