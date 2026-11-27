"""Run one scoped A7 correction of the completed, rejected 6101 R8 decision."""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from functools import partial
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.metadata_view import condition_changes
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import ResponseFormatError, parse_response
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.runner import _validation
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportStop, make_request, terminal_record
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import scheduler, seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import BridgePending, validate_result, verify_job
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAT_SUFFIX
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, STUDY, verify_authorization
from . import run_v13_relay_retry as relay


SEED, ROUND, TARGET = 6101, 8, 621
RUN_NAME = f"{relay.bounded.PREFIX}-6101-r8-targeted-repair-01"
DISPATCHER = "dispatch_v13_6101_r8_targeted_repair"
PARENT = relay.bounded.RunIdentity(
    "-6101-relay-retry-28", "d482a87af7a5b4340739d810e50c76b8f3d0abae590aca107b5a5a8716248a67",
    "c8513c9d0350e8bf0a32b83121fe81421857b05cc8e4aa3f4d42fd959387f86b",
)
LANE = SimpleNamespace(seed=SEED, round=ROUND)
ATTEMPT = "72d0fe476b3842a5a428b3fadb4b20bd"
REQUEST = "776ca26679900435061486d83b0680228b98b27048886e3c5fefd0208c78cc24"
JOB = "6ac0c8d6203d56de2c563801881540abc3d06f7ad143e5ae883cafd85ab4fcdf"
RESULT = "fb5e25d8ed837bb2a32b8d09950b74c3db9169674c39c34b9bb84b7bce06d478"
RAW = "8068384abe2b49a59c0f40f76d9343d08327a8a913139ae47cf3c38a19d52508"
ANSWER = "3a1bc0bec0363b903478f849e14a51327aaab8445ce0aa97035f49e95b0a44a1"
PLAN = "56e6a534709d9156e03345ceaec2c260ea85aac711d330515936d6d8b49fe614"
ORDER = "82b3b4b16ac76980e46764263e34d51dac3d11a7a13d686643b26f34f89e0355"
RESPONSE = "resp_054a6642cafcff24016b08c34f7d0c87d09106ab9378800f5b"
PAYLOAD = "2958938fcb0b6cea2e5ef5fb8b379cc02a4ed8a35b4d5635a2fd7f3eb95a006c"
OLD_PAUSE = "ae6a2801b839ae4e61cfc310cab353718ccc57ad538624351ac03121d23a167b"
OLD_WORKER = "c48b19f0271329016a11b940c5359c4b588445e09b197fed19f219299db79bf9"
OLD_TERMINAL = "014ea1b74f0a475e16492d7d4d7aa8e51cca986310cbbd19ef907d90cad03d8b"
OLD_DECISION = "ee52b1d6f15c5cc6306909aabfd8fb74479417d675b5035aa2b7f81b10f8d882"
OLD_DECISION_SHA256 = "996ecccea301ef8f000cde085f64e0e8dc1c9014e9ab54b15c28bc3ee0d5a3be"
CHECKPOINT = "244c8df2fc40b2b83f0abdf5f710e419d15ebd91918b7bad2c1c33a48d45da8c"
ORIGINAL_JOB_FILE = "554129f92606ebf696882f24e3d8a5858b74cfaea7764d0adeb8f4b7abc51875"
ORIGINAL_RESULT_FILE = "c1dd9ee2874df337895f0fa8024d11663510e0537c68d633bda54901b5d25ddb"
ORIGINAL_CAPTURE_FILE = "cbefd79649cdd3cfbd614506a3a6fac1dfc567135c4cf331a2dbb4ce9834b28a"
ORIGINAL_RAW_FILE = "a1f25aabcc4ba339f63c2e9bc95f402fa33cb6b7be84c39e5c36db96e49239f8"
ORIGINAL_ACK_FILE = "dbcb8393d110c12413b83f613862f222773aa0ddd3419c64f872990d14e9048b"
BOUNDARY_FILES = {'boundary.json': '8b6617fa11495dd45c2042a659118454119b5adc20deb91c45b1a683e6e384ac',
 'round_08/source.json': '3230dd9d4a8cf6c0c1222ef322f5240b679d4d7ad17bfde42b16b3134a67270d',
 'round_08/boundary_memory.json': '926c576d6a1694029b1dc34745d7fbde45934a9d3b87078d19e714dfe62a660c',
 'round_08/boundary_user_message.json': '1259fd41051438f3244a7572c39d65bc077c1bac36c59b42d2bc05e0972c5e4b',
 'round_08/authoritative_packet.json': '6b20eb5704f7071f78e9ffb057c8e0bc9f38d1ae9c95d31e187a1fa63ea0334a'}
OLD_P0_FILES = {'attempt_00.terminal.json': '3a2d62dd802f599e84b8c670b7fe583c5472488e06af0b841f9477f6839f8ae2',
 'attempt_00.replay_authorization.json': 'd72e567c92ae2ae3d01293beafac234655d86d8a35acf73153945540e4504cc6',
 'attempt_00.replay.terminal.json': '9acc60ca7aaa34416df1351bece583941ab6bbdb3f8d22c245cce26f6dcedd29',
 'attempt_01.replay_link.json': 'e333099e4fd1f5e9a66d47ad2bf0931caafbfb1f1ebfcddbd85899ac935bdc7c'}
P0_ATTEMPT = "8acc680cc00147df9ea32238a0a9a565"
P0_JOB = "e4eb29524be24442a03785c62806af6a05a23c44723f8a44272257dc61e8117f"
P0_RESULT = "8430fe2f1bcd3fc4510d642995bc04afbced729e3a48087e087a08f21d085281"
P0_OLD_ACK = "eaeab8a3a95eaeef3f24f3936a4d73f6a153b4be26bc473884c6b2f3b0f2c00b"
REPAIRS = (
    {
        "index": 16, "candidate": "C-1496", "molecule": "M-0126", "reference": "O-0124", "reference_molecule": "M-0178",
        "old_relationship": "replicate", "candidate_conditions": {"PE/EA": "5/1", "sample_volume": "100"},
        "reference_conditions": {"PE/EA": "5/1", "sample_volume": "100"},
        "changed_axes": ("CAS", "density", "loading_solvent_volume"),
        "same_molecule_evidence": ("O-1543", "O-1072"),
        "reason": ("Compare C-1496 (M-0126, PE/EA 5/1, 100 µL) with O-0124 (M-0178, PE/EA 5/1, 100 µL) "
                   "as a cross-molecule analogy. O-1543 and O-1072 provide same-molecule observed context for C-1496."),
    },
    {
        "index": 17, "candidate": "C-0884", "molecule": "M-0147", "reference": "O-3082", "reference_molecule": "M-0212",
        "old_relationship": "condition_response", "candidate_conditions": {"PE/EA": "0/1", "sample_volume": "150"},
        "reference_conditions": {"PE/EA": "2/1", "sample_volume": "50"},
        "changed_axes": ("CAS", "purity_raw", "density", "sample_volume", "PE/EA"),
        "same_molecule_evidence": ("O-2050",),
        "reason": ("Compare C-0884 (M-0147, PE/EA 0/1, 150 µL) with O-3082 (M-0212, PE/EA 2/1, 50 µL) "
                   "as a cross-molecule analogy. O-2050 is same-molecule observed evidence for C-0884."),
    },
)
DATA_BINDINGS = ("seed", "round", "arm", "schema_version", "prompt_version", "checkpoint_sha256", "packet_sha256",
                 "wire_packet_sha256", "wire_representation", "mapping_sha256", "split_sha256", "state_sha256")
BUDGET = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 1}
SOURCES = tuple(STUDY / "scripts" / f"{module}.py" for module in ("run_v13_6101_r8_targeted_repair", DISPATCHER))
REJECTIONS = [{'code': 'different_molecules', 'path': '/choices/16/references/0/relationship', 'reference_id': 'O-0124'},
 {'actual_changed_axes': ['CAS', 'density', 'loading_solvent_volume'],
  'code': 'replicate_condition_mismatch',
  'path': '/choices/16/references/0/relationship'},
 {'code': 'different_molecules', 'path': '/choices/17/references/0/relationship', 'reference_id': 'O-3082'}]


def _reference(repair, relationship):
    return {"reference_id": repair["reference"], "reference_kind": "observed", "relationship": relationship}


def guide():
    return (
        "Scoped correction for the existing A7 batch. All 32 selected candidates and their order are fixed. "
        "Output one JSON object with exactly the repairs key, whose value is an ordered array of two patch objects. "
        "Each patch object has exactly candidate_id, reference_index, reference, and reason. "
        "Use reference_index 0 for both patches. Each reference has exactly reference_id, reference_kind, and relationship. "
        "In this order use C-1496 and O-0124, then C-0884 and O-3082. Both references have "
        "reference_kind observed and relationship cross_molecule_analogy. No changed_axis is needed for an analogy. "
        "All other references, selection fields, and hypotheses are fixed. "
        "For each reason use its corresponding two-sentence form below. The leading verb Compare may instead be Assess or Examine. "
        f"Use this reason for C-1496. {REPAIRS[0]['reason']} "
        f"Use this reason for C-0884. {REPAIRS[1]['reason']} "
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
    if (digest(text), digest(plan)) != (ANSWER, PLAN):
        raise PermissionError("the completed provider plan changed")
    for repair in REPAIRS:
        choice = plan["choices"][repair["index"]]
        if (choice.get("candidate_id"), choice.get("molecule_id"), choice["references"][0]) != (
                repair["candidate"], repair["molecule"], _reference(repair, repair["old_relationship"])):
            raise PermissionError("the completed provider plan differs from the two scoped reference slips")
    return plan


def _checked_reason(reason, expected):
    if type(reason) is not str:
        return False
    verb, _, prose = reason.partition(" ")
    return (
        verb in {"Compare", "Assess", "Examine"}
        and prose.replace("μL", "µL").replace("uL", "µL") == expected.removeprefix("Compare ")
    )


def project_reply(patch, *, original):
    if type(patch) is not dict or set(patch) != {"repairs"} or type(patch["repairs"]) is not list or len(patch["repairs"]) != 2:
        raise ResponseFormatError("response must contain the two scoped reference and reason corrections")
    if digest(original) != PLAN:
        raise PermissionError("the cached 6101 provider plan changed")
    plan = deepcopy(original)
    for item, repair in zip(patch["repairs"], REPAIRS, strict=True):
        if (type(item) is not dict or set(item) != {"candidate_id", "reference_index", "reference", "reason"}
                or item["candidate_id"] != repair["candidate"] or type(item["reference_index"]) is not int
                or item["reference_index"] != 0 or item["reference"] != _reference(repair, "cross_molecule_analogy")
                or not _checked_reason(item["reason"], repair["reason"])):
            raise ResponseFormatError("response is outside the two scoped reference and reason corrections")
        choice = plan["choices"][repair["index"]]
        choice["references"][0] = deepcopy(item["reference"])
        choice["reason"] = item["reason"]
    _ordered(plan)
    return plan


def _archive_job(job, receipt, wire_representation):
    request, transport, started = verify_job(job, contract_sha256=PARENT.contract, authorization_sha256=PARENT.approval,
                                              wire_representation=wire_representation)
    result = validate_result(receipt, job)
    if ((request["request_id"], request["payload_sha256"], started["attempt_id"], started["attempt"],
         job["job_sha256"], result["result_sha256"], result["raw_response_sha256"], result["provider_dispatched"])
            != (REQUEST, PAYLOAD, ATTEMPT, 1, JOB, RESULT, RAW, True)
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
        "old_decision_file_sha256": OLD_DECISION, "original_payload_sha256": original["payload_sha256"],
        "repairs": [{"choice_index": repair["index"], "reference_index": 0, "candidate_id": repair["candidate"],
                     **_reference(repair, "cross_molecule_analogy")} for repair in REPAIRS],
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
    turn = relay.bounded._turn(active, 2)
    if {path for path in active.glob("turn_*") if path.is_dir()} != {turn}:
        raise PermissionError("the archive contains another logical proposal")
    if (file_digest(directory / "pause.request"), file_digest(turn / "attempt_01.terminal.json"),
        file_digest(turn / "decision.json"), file_digest(old / f"workers/seed_{SEED}.json")) != (
            OLD_PAUSE, OLD_TERMINAL, OLD_DECISION, OLD_WORKER):
        raise PermissionError("the retired 6101 pause, terminal or decision receipt changed")
    if any(file_digest(turn / path) != sha for path, sha in OLD_P0_FILES.items()):
        raise PermissionError("the first failed transport and its sanctioned replay changed")
    if any(file_digest(directory / path) != sha for path, sha in BOUNDARY_FILES.items()):
        raise PermissionError("the verified L589 boundary or complete packet changed")
    original = relay.bounded._request(active, turn, parent, approval, LANE)
    if any({path.name for path in (old / "bridge" / folder).glob("*.json")} != {f"{P0_ATTEMPT}.json", f"{ATTEMPT}.json"}
           for folder in ("outbox", "claims", "results")):
        raise PermissionError("the previous two-attempt mailbox changed")
    job_path = old / "bridge/outbox" / f"{ATTEMPT}.json"
    result_path = old / "bridge/results" / f"{ATTEMPT}.json"
    if (file_digest(job_path), file_digest(result_path)) != (ORIGINAL_JOB_FILE, ORIGINAL_RESULT_FILE):
        raise PermissionError("the original bridge bytes differ from the Mac custody records")
    outbox, result = read(job_path), read(result_path)
    archived, transport, reply, plan = _archive_job(outbox, result, parent.profile["wire_representation"])
    terminal = terminal_record(turn, 1)
    started = read(turn / "attempt_01.started.json")
    old_job = read(old / "bridge/outbox" / f"{P0_ATTEMPT}.json")
    old_receipt = validate_result(read(old / "bridge/results" / f"{P0_ATTEMPT}.json"), old_job)
    prior, _, old_started = verify_job(old_job, contract_sha256=PARENT.contract, authorization_sha256=PARENT.approval)
    old_terminal = terminal_record(turn, 0)
    if (prior != original or old_started != read(turn / "attempt_00.started.json")
            or (old_started["attempt_id"], old_job["job_sha256"], old_receipt["result_sha256"]) != (P0_ATTEMPT, P0_JOB, P0_RESULT)
            or old_receipt["outcome"]["status"] != "failed" or old_receipt["raw_response"] is not None
            or old_receipt["outcome"]["failure"]["code"] != "response_id_mismatch"
            or old_receipt["provider_dispatched"] is not True
            or (old_terminal.get("bridge_job_sha256"), old_terminal.get("bridge_result_sha256")) != (P0_JOB, P0_RESULT)):
        raise PermissionError("the first physical attempt was not the exact reviewed incomplete failure")
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
        raise PermissionError("the old request differs from the archived L589 packet")
    evaluation = _validation(stage, plan)
    diagnosis = {"state": "rejected", "diagnosis": evaluation.as_dict()}
    if (state.checkpoint != CHECKPOINT or len(state.provenance["expected_observed"]) != 589
            or digest(diagnosis) != OLD_DECISION_SHA256 or read(turn / "decision.json") != diagnosis
            or evaluation.valid or evaluation.valid_choice_count != 30 or len(evaluation.legal_candidate_indices) != 32
            or list(evaluation.errors) != REJECTIONS):
        raise PermissionError("the A7 rejection differs from the two reviewed reference slips")
    for repair in REPAIRS:
        candidate = stage.context.candidates[repair["candidate"]]
        observed = stage.context.observed[repair["reference"]]
        if (candidate["molecule_id"] != repair["molecule"] or observed["molecule_id"] != repair["reference_molecule"]
                or condition_changes(candidate["conditions"], observed["conditions"]) != repair["changed_axes"]
                or any(candidate["conditions"][k] != v for k,v in repair["candidate_conditions"].items())
                or any(observed["conditions"][k] != v for k,v in repair["reference_conditions"].items())
                or any(stage.context.observed[ref]["molecule_id"] != repair["molecule"]
                       for ref in repair["same_molecule_evidence"])):
            raise PermissionError("the authoritative packet differs from the two reviewed cross-molecule evidence records")
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
             "previous_worker_file_sha256": OLD_WORKER, "old_boundary_files": BOUNDARY_FILES, "prior_transport_replay_files": OLD_P0_FILES,
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
            raise PermissionError("the retired 6101 worker fence changed")
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
