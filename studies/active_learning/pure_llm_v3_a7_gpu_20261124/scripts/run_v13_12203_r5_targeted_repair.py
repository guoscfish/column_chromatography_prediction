"""Run one scoped A7 correction of the completed, rejected 12203 R5 decision."""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from functools import partial
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


SEED, ROUND, TARGET = 12203, 5, 525
RUN_NAME = f"{relay.bounded.PREFIX}-12203-r5-targeted-repair-01"
DISPATCHER = "dispatch_v13_12203_r5_targeted_repair"
PARENT = relay.bounded.RunIdentity("-12203-relay-retry-72", "3e65dbdca4f5c7cfae88580fdb9483e4fe4dd52ee65211b31627ddaa085db33d", "2e4aab2d681d901af5d31fcf9f6eca6f87884a5df680f82493a8a9bd5848cb66")
LANE = SimpleNamespace(seed=SEED, round=ROUND)
ATTEMPT = "eb5ab73df24b49519dccce0cb1438854"
REQUEST = "ed843d3c27de1ccea41d83abf477b5b1e784f412dcba40474381715f0996b580"
JOB = "456c31d29cc128f4158501caf7b746baaa7e6dab1084c5c2af4f04204d0843e8"
RESULT = "1e63e3bb4d48014c7322add7e5aa53051eb19ba0ed794a7568dc928726c4e8df"
RAW = "96c8dfa98c149bca6d8e4d75607983f29ea03b0e03038e6ceac4dd4fb41b18e9"
ANSWER = "753401c1ff46f171f5a96fbe31e5d1b42e7ca2b008d3ac0782ba9171fb66949d"
PLAN = "8619e1a5f27faafe50b3f64306273745a499b025b70f6f7608175f03d38a79c9"
ORDER = "922b53b7a6ced64664e9bb45cee26f6b9955931a177cf4909b6d7ea6c1178a67"
RESPONSE = "resp_0d5097fbed9b751e016b09513b1d6c87d0a9aaff312cfac33a"
PAYLOAD = "00d53155b752dc54eed4e6aad7bfbdb01d19b21c7f0866cebd85391f8d456373"
OLD_PAUSE = "738b50ad7e9503d28071d263546d591a9a405b1257bb4ffc917a1411a244ea82"
OLD_WORKER = "e8e787ba20f8374149a2aa5ebbca3bb17cd5be0cc1c67887adda135f5d4549c2"
OLD_TERMINAL = "aaf422d6c50e5e6d3040288d3d82e721a682a33a95fe1227a124ee8122fba158"
OLD_DECISION = "6b0b3d456347b7c91b3be883bceb0e766e6b5a08e6c0b14eff3644a4250f311d"
OLD_DECISION_SHA256 = "00b5434affa4e47df76f5a7600876aa978d55da7859d9af3329f97f8e6324622"
CHECKPOINT = "34cf4e966602c4eed13ea70922aa80313ad415dd5b3cefe7fd2e7650f05a4c07"
ORIGINAL_JOB_FILE = "ddf415bb61de6d0031243d2eea04c6633ea14b87f82e41dd4ce0bb14c99e9075"
ORIGINAL_RESULT_FILE = "270571ac959e0b1f45be0deccd31205e9f9a269cb61846486e6c7f0597c32bf3"
ORIGINAL_CAPTURE_FILE = "007b9510cbe6ae97c4567d6b99fb6ed50bd68de5e9e6d8d4b3d4d6f0a2af5bbc"
ORIGINAL_RAW_FILE = "97aac8d0a896ca07742cb6a7a4fa47c434f802dd73387a5c8d012e5aee0faa2c"
ORIGINAL_ACK_FILE = "c68004d84405f96f642bf84bc59b657e3d2809a7a41da1bfabe89f8eacbb1974"
OLD_REVIEWED_RECORD = "ce895aa0b5446c1d0e6b4f2147989f837d5b56b269f291ced5f56eb8f80d797a"
OLD_REVIEWED_RECORD_FILE = "2f05da428be56ef6422ef3108feb3639ddedf24bce2a3df170b6a50f0feb685f"
ARCHIVE_PROOF_SHA256 = "2c0afa8616ea625ac8bb780bf8cd82192b54997e654500c0c3b6eb047745ee02"
ARCHIVE_FIELDS = ('checkpoint_sha256',
 'packet_sha256',
 'wire_packet_sha256',
 'base_messages_sha256',
 'previous_run_directory',
 'previous_contract_sha256',
 'previous_authorization_sha256',
 'previous_release_directory',
 'previous_attempts',
 'reviewed_retry_chain',
 'reviewed_recovery_record_sha256',
 'retired_reviewed_recovery_records',
 'model_identity')
BOUNDARY_FILES = {'boundary.json': '771d35527905c7b4721334a373cee9477ef450310a4797ad866895f6b8a04c23',
 'round_05/authoritative_packet.json': '0b3208712b8f4606a7b63b65444003926bf456a999a5520acce238001b538108',
 'round_05/boundary_memory.json': '0150070586759a4a2200f593df0af5207b6c63a84b92c3081996e4ddca04ce11',
 'round_05/boundary_user_message.json': '1ac1c4c0feb0cac699179f63007bd71c8baf1fb38aee2ce3d5beb29e8387320e',
 'round_05/source.json': '9b290ab2a6c5a7da8334eb77ca003a10ac1b52ed10a76c0baf3b02b2b9f64eac',
 'round_05/turn_00/attempt_00.started.json': 'eb005e68ac9f2cbc063dce134a0d9c4f6d7068ef260ad125b9f4ca635943bde1',
 'round_05/turn_00/request.json': '0bcf6cb5ce007d6ac98b70058f0c2ce52950b0fe5c00e2bab86ab85b0a4b43b3'}
REPAIRS = ({'choice_index': 12,
  'ref_index': 0,
  'candidate': 'C-3068',
  'molecule': 'M-0073',
  'reference': 'O-2880',
  'reference_molecule': 'M-0073',
  'old_reference': {'reference_id': 'O-2880',
                    'reference_kind': 'observed',
                    'relationship': 'single_factor_contrast',
                    'changed_axis': 'loading_solvent_volume'},
  'changed_axes': ['flow'],
  'candidate_conditions': {'flow': '10', 'sample_volume': '50', 'PE/EA': '2/1'},
  'reference_conditions': {'flow': '5', 'sample_volume': '50', 'PE/EA': '2/1'},
  'corrected_reference': {'reference_id': 'O-2880',
                          'reference_kind': 'observed',
                          'relationship': 'single_factor_contrast',
                          'changed_axis': 'flow'},
  'reason': 'Compare C-3068 at 10 mL/min with O-2880 at 5 mL/min to vary flow while holding the other recorded '
            'conditions fixed.'},
 {'choice_index': 15,
  'ref_index': 0,
  'candidate': 'C-0489',
  'molecule': 'M-0090',
  'reference': 'O-2389',
  'reference_molecule': 'M-0090',
  'old_reference': {'reference_id': 'O-2389',
                    'reference_kind': 'observed',
                    'relationship': 'single_factor_contrast',
                    'changed_axis': 'PE/EA'},
  'changed_axes': ['sample_volume', 'PE/EA'],
  'candidate_conditions': {'flow': '10', 'sample_volume': '100', 'PE/EA': '50/1'},
  'reference_conditions': {'flow': '10', 'sample_volume': '150', 'PE/EA': '20/1'},
  'corrected_reference': {'reference_id': 'O-2389',
                          'reference_kind': 'observed',
                          'relationship': 'multifactor_contrast'},
  'reason': 'Compare C-0489 at PE/EA 50/1 and 100 µL with O-2389 at PE/EA 20/1 and 150 µL to extend the weak-elution '
            'range. The volume and ratio both change, so neither effect is isolated.'},
 {'choice_index': 30,
  'ref_index': 0,
  'candidate': 'C-1600',
  'molecule': 'M-0207',
  'reference': 'O-0579',
  'reference_molecule': 'M-0068',
  'old_reference': {'reference_id': 'O-0579',
                    'reference_kind': 'observed',
                    'relationship': 'single_factor_contrast',
                    'changed_axis': 'PE/EA'},
  'changed_axes': ['CAS', 'density', 'PE/EA'],
  'candidate_conditions': {'flow': '10', 'sample_volume': '50', 'PE/EA': '2/1'},
  'reference_conditions': {'flow': '10', 'sample_volume': '50', 'PE/EA': '20/1'},
  'corrected_reference': {'reference_id': 'O-0579',
                          'reference_kind': 'observed',
                          'relationship': 'cross_molecule_analogy'},
  'reason': 'Compare C-1600 (M-0207, PE/EA 2/1) with O-0579 (M-0068, PE/EA 20/1) only as a cross-molecule analogy. '
            'The different compounds and solvent ratios do not isolate a ratio effect.'})
REJECTIONS = [{'actual_changed_axes': ['flow'],
  'code': 'single_factor_mismatch',
  'path': '/choices/12/references/0/changed_axis'},
 {'actual_changed_axes': ['sample_volume', 'PE/EA'],
  'code': 'single_factor_mismatch',
  'path': '/choices/15/references/0/changed_axis'},
 {'code': 'different_molecules', 'path': '/choices/30/references/0/relationship', 'reference_id': 'O-0579'},
 {'actual_changed_axes': ['CAS', 'density', 'PE/EA'],
  'code': 'single_factor_mismatch',
  'path': '/choices/30/references/0/changed_axis'}]
DATA_BINDINGS = ("seed", "round", "arm", "schema_version", "prompt_version", "checkpoint_sha256", "packet_sha256",
                 "wire_packet_sha256", "wire_representation", "mapping_sha256", "split_sha256", "state_sha256")
BUDGET = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 1}
SOURCES = tuple(STUDY / "scripts" / f"{module}.py" for module in ("run_v13_12203_r5_targeted_repair", DISPATCHER))


def guide():
    requested = []
    for scope in REPAIRS:
        requested.append({"candidate_id": scope["candidate"], "reference_index": scope["ref_index"],
                          "reference": scope["corrected_reference"], "reason": scope["reason"]})
    return (
        "Scoped correction of three comparisons in the existing 32-choice A7 batch. Keep all candidate IDs, "
        "ranks, other references, choices, strategy, and hypotheses exactly as provided. "
        "Apply only these precise relationships and evidence-based reasons; no new proposal is authorized. "
        "Return a single JSON object with exactly a repairs array ordered as shown. "
        "Use Compare, Assess, or Examine as the leading verb of each reason, preserving its scientific content. "
        f"Required JSON shape and scoped content: {dumps({'repairs': requested})}"
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
    if (digest(text), digest(plan)) != (ANSWER, PLAN):
        raise PermissionError("the exact completed 12203 provider plan changed")
    for scope in REPAIRS:
        choice = plan["choices"][scope["choice_index"]]
        if (choice["candidate_id"], choice["molecule_id"], choice["references"][scope["ref_index"]]) != (
                scope["candidate"], scope["molecule"], scope["old_reference"]):
            raise PermissionError("a scoped comparison is no longer the exact archived choice")
    return plan


def _checked_reason(reason, expected):
    if type(reason) is not str:
        return False
    verb, _, text = reason.partition(" ")
    return (verb in {"Compare", "Assess", "Examine"}
            and text.replace("μL", "µL").replace("uL", "µL") == expected.removeprefix("Compare "))


def project_reply(patch, *, original):
    if (type(patch) is not dict or set(patch) != {"repairs"} or type(patch["repairs"]) is not list
            or len(patch["repairs"]) != len(REPAIRS)):
        raise ResponseFormatError("expected the three scoped reference and reason repairs")
    if digest(original) != PLAN:
        raise PermissionError("the archived 12203 provider plan changed")
    plan = deepcopy(original)
    for item, scope in zip(patch["repairs"], REPAIRS, strict=True):
        if (type(item) is not dict or set(item) != {"candidate_id", "reference_index", "reference", "reason"}
                or item["candidate_id"] != scope["candidate"] or type(item["reference_index"]) is not int
                or item["reference_index"] != scope["ref_index"] or item["reference"] != scope["corrected_reference"]
                or not _checked_reason(item["reason"], scope["reason"])):
            raise ResponseFormatError("response is outside the three scoped reference and reason repairs")
        choice = plan["choices"][scope["choice_index"]]
        choice["references"][scope["ref_index"]] = deepcopy(item["reference"])
        choice["reason"] = item["reason"]
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
        "ordered_candidate_ids_sha256": ORDER, "starting_checkpoint_sha256": CHECKPOINT,
        "repairs": [{"choice_index": scope["choice_index"], "reference_index": scope["ref_index"],
                     "candidate_id": scope["candidate"], **scope["corrected_reference"]} for scope in REPAIRS],
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


def _reviewed_scope(path):
    if file_digest(path) != OLD_REVIEWED_RECORD_FILE:
        raise PermissionError("the original W72 immutable reviewed record changed")
    record = read(path)
    return record, successor._record_scope(record, OLD_REVIEWED_RECORD, SEED, 72)


def _reviewed_archive(old, parent):
    prior = {key: parent.predecessor[key] for key in ARCHIVE_FIELDS}
    if digest(prior) != ARCHIVE_PROOF_SHA256:
        raise PermissionError("the frozen W72 reviewed archive proof changed")
    record, scope = _reviewed_scope(old / successor.REVIEWED_RECORD_FILE)
    if prior[successor.REVIEWED_RECORD_FIELD] != OLD_REVIEWED_RECORD:
        raise PermissionError("W72 does not authenticate its original reviewed archive")
    records = successor._archives(Path(prior["previous_run_directory"]), Path(prior["previous_release_directory"]), scope)
    # W69 is a signed static anchor. W70 and W71 are the first two exact prefixes of W72.
    for count, entry in enumerate(reversed(records[:2]), 1):
        expected = {**record, "steps": record["steps"][:count]}
        if (entry["contract"].predecessor.get(successor.REVIEWED_RECORD_FIELD) != digest(expected)
                or read(entry["run"] / successor.REVIEWED_RECORD_FILE) != expected):
            raise PermissionError("a retired W70–W71 reviewed record prefix changed")
    latest = records[0]
    result = read(latest["run"] / "bridge/results" / f"{scope.attempt_id}.json")
    successor._check_reviewed_parent(scope, latest["contract"].predecessor, latest["request"], result)
    return prior


def construct(root, old_root, old, run):
    if root == old_root or run.name != RUN_NAME or run.parent != old.parent:
        raise PermissionError("use the distinct signed correction release and sibling run")
    parent, approval = relay.bounded._archived(old, old_root, PARENT)
    original_archive = _reviewed_archive(old, parent)
    directory, active = relay._retired(old, LANE, require_pause=True)
    turn = relay.bounded._turn(active, 1)
    if (file_digest(directory / "pause.request"), file_digest(turn / "attempt_00.terminal.json"),
        file_digest(turn / "decision.json"), file_digest(old / f"workers/seed_{SEED}.json")) != (
            OLD_PAUSE, OLD_TERMINAL, OLD_DECISION, OLD_WORKER):
        raise PermissionError("the retired 12203 pause, terminal or decision receipt changed")
    if any(file_digest(directory / path) != sha for path, sha in BOUNDARY_FILES.items()):
        raise PermissionError("the verified L493 boundary or complete packet changed")
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
        raise PermissionError("the old request differs from the archived L493 packet")
    evaluation = _validation(stage, plan)
    diagnosis = {"state": "rejected", "diagnosis": evaluation.as_dict()}
    patch = {"repairs": [{"candidate_id": item["candidate"], "reference_index": item["ref_index"],
                          "reference": item["corrected_reference"], "reason": item["reason"]} for item in REPAIRS]}
    if (state.checkpoint != CHECKPOINT or len(state.provenance["expected_observed"]) != 493
            or digest(diagnosis) != OLD_DECISION_SHA256 or read(turn / "decision.json") != diagnosis
            or evaluation.valid or evaluation.valid_choice_count != 29 or list(evaluation.errors) != REJECTIONS
            or not _validation(stage, project_reply(patch, original=plan)).valid):
        raise PermissionError("the authoritative L493 rejection changed or the three scoped repairs fail A7")
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
             "original_reviewed_archive": original_archive, "original_reviewed_archive_sha256": ARCHIVE_PROOF_SHA256,
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
            raise PermissionError("the retired 12203 worker fence changed")
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
