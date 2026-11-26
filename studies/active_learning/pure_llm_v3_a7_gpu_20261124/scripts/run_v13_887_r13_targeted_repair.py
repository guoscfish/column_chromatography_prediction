"""Run one scoped A7 correction of the completed, rejected 887 R13 decision."""

import argparse
from collections import Counter
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
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.runner import _observed_alias_projection, _validation
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportStop, make_request, terminal_record
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import BridgePending, validate_result, verify_job
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAT_SUFFIX
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, STUDY, verify_authorization
from . import run_v13_relay_retry as relay


SEED, ROUND, TARGET = 887, 13, 781
RUN_NAME = f"{relay.bounded.PREFIX}-887-r13-targeted-repair-01"
DISPATCHER = "dispatch_v13_887_r13_targeted_repair"
PARENT = relay.bounded.RunIdentity(
    "-887-relay-retry-10", "cc90be12c858a5b2a9d2e1687c0dceec5dd249d1d7fee4753c9210a9f054950d",
    "f31d42cdfed27143ce9a42d5c72c4a4bdd054d8496f5e78bbf90f07b7b169cb2",
)
LANE = SimpleNamespace(seed=SEED, round=ROUND)
ATTEMPT = "15a9a6bed1ad4b52b30a5671ee37787b"
REQUEST = "1ad7a37fcc64c26caaa0e21ad63359ed4fd23ac743e0b0cddb86f7ac7d11ec37"
JOB = "77fab8e002df56d5f3b28ae322efcb2c0e867553db6ecf0d16162ec4e4ad4194"
RESULT = "1d4517a66cc5c3e4aff10da0972680712bf893cb3a8832ae91e5832181f99914"
RAW = "81313ed247f06e1ddf0b268a1c08bdaade328cf1b8e55d487329cd807b6c203a"
ANSWER = "7c037be03e806d8994faf691a22e0f2900f383bff1eec0aa30b388b760467ed9"
ORIGINAL_PLAN = "24b0f0e11c944c1bbfa8d04d3a851b36356e6ace7571298c8c978ebc62cbbf0b"
PLAN = "dc878f267936c40d66e504aee28c5c03c6523e0d3c31afaceb514be31789bed1"
ORDER = "34ecb37bceb1398406ada92880f9338bd34bdf685c65f42fdb4c8fe2959bbf8e"
REMOVED_ALIASES = 61
ALIAS_PATHS = "6cb9370df72ef53756765610a27aea96ae280da36e4b7bb634e8f1dea0dd2946"
RESPONSE = "resp_01ad18f5edbfe55c016b087a6ddcf487d2a186e70389448bcc"
PAYLOAD = "20b9b243194cbc330493d023e3f27928fea6d7f05ae8d900c33b413a9670dcb4"
OLD_PAUSE = "9e8a2593f7abc3064585766ec741308373a39b2d03bbcc165723efefcbd76e90"
OLD_WORKER = "cf0d1fd15220f13344c896b35cd1d01870f520dc439170b45cc41fc249d0ed24"
OLD_TERMINAL = "79f9398f843d000b623be350c5905fdf1d21af05dbf9e2b6a4b97b8f9486aad7"
OLD_DECISION = "40a656f25cb2e2f65580797187d6a7aa190a71d6a7a006fbc952b0812abcfe8c"
OLD_DECISION_SHA256 = "98eaee8698258f64c642548ff1f435a6cc226754b2e9549c7e7916e6160c2789"
CHECKPOINT = "dced0207b5677e2b5b1d1937a40af0d26bae92cdfc6d72518a95410ac431a152"
ORIGINAL_JOB_FILE = "152d31c6eec99a54cdfc9c67e10e100206711867d1c782437675b9415fae0ecd"
ORIGINAL_RESULT_FILE = "b9cd51396899b4c037f5a9aa0616650070b39ceaa18d28a1511c8ba1d6bf71b1"
ORIGINAL_CAPTURE_FILE = "5ab9b64a611c26d03af7da5b9857d652df2f17b49f9fdbd3a8fa1bf66e340785"
ORIGINAL_RAW_FILE = "3671dfb6a2eed8fda574d3e9dd1d6b1c17b7f9a76ace077274fdd1087758c74c"
ORIGINAL_ACK_FILE = "56b13e4b6e81986cb3c85c6f865bd01a6dc632fb8600d91beb2da6ea3e7a3cd2"
BOUNDARY_FILES = {
    "boundary.json": "48586f45bc0d1f81b5aab6ad475a435b3a1f72942bf7dee8ccbd98b197570c61",
    "round_13/source.json": "242f308cc4fa10fbe86d31bde6e349f31c5745c2a8b47a3588468c92fb008be1",
    "round_13/boundary_memory.json": "58763db2a3995f943fb84f7f9b32c937705b4606b1c43a988079e84132380410",
    "round_13/boundary_user_message.json": "89ca2fa0e3be6409890dbcd40a219d1762481ec63345037a2048b1ec2fc39f33",
    "round_13/authoritative_packet.json": "fc15e56dd9c646aa7155986e15343c4c5548529023a8a136b9fde024106b0c9c",
}
CHOICE_INDEX, REFERENCE_INDEX = 11, 0
CANDIDATE, MOLECULE, OBSERVED, OBSERVED_MOLECULE, SAME_MOLECULE_OBSERVED = "C-0400", "M-0061", "O-3192", "M-0062", "O-2659"
CONDITIONS = {
    CANDIDATE: (MOLECULE, {"sample_volume": "150", "PE/EA": "2/1"}),
    OBSERVED: (OBSERVED_MOLECULE, {"sample_volume": "100", "PE/EA": "2/1"}),
    SAME_MOLECULE_OBSERVED: (MOLECULE, {"sample_volume": "150", "PE/EA": "0/1"}),
}
DATA_BINDINGS = ("seed", "round", "arm", "schema_version", "prompt_version", "checkpoint_sha256", "packet_sha256",
                 "wire_packet_sha256", "wire_representation", "mapping_sha256", "split_sha256", "state_sha256")
BUDGET = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 1}
SOURCES = tuple(STUDY / "scripts" / f"{module}.py" for module in ("run_v13_887_r13_targeted_repair", DISPATCHER))
WRONG_REFERENCE = {"reference_id": OBSERVED, "reference_kind": "observed", "relationship": "condition_response"}
REQUIRED_REFERENCE = {"reference_id": OBSERVED, "reference_kind": "observed", "relationship": "cross_molecule_analogy"}
UNCHANGED_REFERENCE = {"reference_id": SAME_MOLECULE_OBSERVED, "reference_kind": "observed", "relationship": "condition_response"}


def guide():
    return (
        "Scoped correction for the already chosen A7 batch. Keep all 32 candidate IDs and their order. "
        "The sole affected choice is index 11, C-0400 (M-0061), and its reference index 0 to observed O-3192 (M-0062). "
        "These are distinct compounds. Assign that reference the relationship cross_molecule_analogy. "
        "Retain the second reference O-2659 (M-0061) as existing within-compound evidence. "
        "Return one JSON object with exactly candidate_id, reference_index, reference, reason. "
        "Set candidate_id to C-0400 and reference_index to 0. The reference object has exactly reference_id, "
        "reference_kind, relationship with values O-3192, observed, cross_molecule_analogy. "
        "Write the reason in this two-sentence form, using one of assess, examine, probe, test, or evaluate for VERB: "
        "Compare C-0400 (M-0061, PE/EA 2/1, 150 µL) with O-2659 (M-0061, PE/EA 0/1, 150 µL) to VERB "
        "the within-compound PE/EA response. O-3192 (M-0062, PE/EA 2/1, 100 µL) provides a cross-molecule analogy. "
        "Preserve every other choice, reference, and hypothesis. The scoped object is the complete output for this request."
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
    original = parse_response(text)
    _ordered(original)
    projection = _observed_alias_projection(original)
    if (digest(text), digest(original)) != (ANSWER, ORIGINAL_PLAN) or projection is None:
        raise PermissionError("the raw provider plan or its exact observed mirrors changed")
    plan, paths = projection
    choice = plan["choices"][CHOICE_INDEX]
    if (digest(plan), len(paths), digest(paths), choice.get("candidate_id"), choice.get("molecule_id"),
        choice["references"]) != (PLAN, REMOVED_ALIASES, ALIAS_PATHS, CANDIDATE, MOLECULE,
                                   [WRONG_REFERENCE, UNCHANGED_REFERENCE]):
        raise PermissionError("the approved mirror removal does not reproduce the rejected contrast")
    return plan


def _checked_reason(reason):
    if type(reason) is not str:
        return False
    pattern = (
        r"Compare C-0400 \(M-0061, PE/EA 2/1, 150 [µμu]L\) with O-2659 \(M-0061, PE/EA 0/1, 150 [µμu]L\) "
        r"to (?:assess|examine|probe|test|evaluate) the within-compound PE/EA response\. "
        r"O-3192 \(M-0062, PE/EA 2/1, 100 [µμu]L\) provides a cross-molecule analogy\."
    )
    return re.fullmatch(pattern, reason) is not None


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
    if reply["response_id"] != RESPONSE or reply["served_model"] != transport.served_model:
        raise PermissionError("the archived response model or identity changed")
    return request, transport, reply, _original_plan(reply["text"])


def _metadata(original):
    return {
        "original_request_id": REQUEST, "original_attempt_id": ATTEMPT, "original_raw_sha256": RAW,
        "original_answer_sha256": ANSWER, "original_result_sha256": RESULT, "original_plan_sha256": ORIGINAL_PLAN,
        "canonical_plan_sha256": PLAN, "removed_alias_count": REMOVED_ALIASES, "removed_alias_paths_sha256": ALIAS_PATHS,
        "ordered_candidate_ids_sha256": ORDER, "repair_index": CHOICE_INDEX, "reference_index": REFERENCE_INDEX,
        "candidate_id": CANDIDATE, "reference_id": OBSERVED, "relationship": REQUIRED_REFERENCE["relationship"],
        "unchanged_reference_id": SAME_MOLECULE_OBSERVED, "starting_checkpoint_sha256": CHECKPOINT,
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
        raise PermissionError("the verified L749 boundary or complete packet changed")
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
        raise PermissionError("the old request differs from the archived L749 packet")
    raw = parse_response(reply["text"])
    raw_evaluation = _validation(stage, raw)
    diagnosis = {"state": "rejected", "diagnosis": raw_evaluation.as_dict()}
    normalized_evaluation = _validation(stage, plan)
    errors = [(error["code"], error["path"], error.get("reference_id")) for error in normalized_evaluation.errors]
    role_rows = {CANDIDATE: stage.context.candidates[CANDIDATE], OBSERVED: stage.context.observed[OBSERVED],
                 SAME_MOLECULE_OBSERVED: stage.context.observed[SAME_MOLECULE_OBSERVED]}
    aliases = _observed_alias_projection(raw)[1]
    alias_counts = Counter(path.rsplit("/", 1)[0] for path in aliases)
    raw_alias_counts = Counter({item["path"]: item["unexpected_field_count"] for item in raw_evaluation.errors
                               if item["code"] == "unexpected_field"})
    raw_contrasts = [(item["code"], item["path"], item.get("reference_id"))
                     for item in raw_evaluation.errors if item["code"] != "unexpected_field"]
    contrast = [("different_molecules", f"/choices/{CHOICE_INDEX}/references/{REFERENCE_INDEX}/relationship", OBSERVED)]
    if (state.checkpoint != CHECKPOINT or len(state.provenance["expected_observed"]) != 749
            or digest(diagnosis) != OLD_DECISION_SHA256 or read(turn / "decision.json") != diagnosis
            or raw_alias_counts != alias_counts or raw_contrasts != contrast
            or normalized_evaluation.valid or normalized_evaluation.valid_choice_count != 31
            or errors != contrast
            or any(row["molecule_id"] != CONDITIONS[key][0]
                   or any(row["conditions"][field] != value for field, value in CONDITIONS[key][1].items())
                   for key, row in role_rows.items())):
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
