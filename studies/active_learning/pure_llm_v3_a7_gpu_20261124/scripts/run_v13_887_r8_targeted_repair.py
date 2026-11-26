"""Run one scoped A7 correction of the completed, rejected 887 R8 decision."""

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


SEED, ROUND = 887, 8
RUN_NAME = f"{relay.bounded.PREFIX}-887-r8-targeted-repair-01"
DISPATCHER = "dispatch_v13_887_r8_targeted_repair"
PARENT = relay.bounded.RunIdentity(
    "-887-relay-retry-04", "dd0501bb9680a2af013c4a5b22976b64ff1b7d691b2645e0b9dab0191564de65",
    "51a2e734219ce337b158573a200975487019bdafb94c2d859823d969c88f7d4e",
)
LANE = SimpleNamespace(seed=SEED, round=ROUND)
ATTEMPT = "708d8c9376434c6ebf1651a4aca1128b"
REQUEST = "6d6ec5097eeb40717d6c8718bd9d6d587a3afe762fb5a8989194ac4a58e2dacc"
JOB = "6b93f5570f550f0ab3656c95c463ffe5ba58a93bc547e87a13970b797064e1f7"
RESULT = "9df26d546983731feb6fa4352f08bcc49c76bc92ed9dbf121ff2cca64424b0ae"
RAW = "0f6e0705ac59a6573aa94f7cab6c8d05a4c053d3770508919cc32ac638d9089f"
ANSWER = "cfa037654ba76bd0972e514d972ca8ccdc9622a198730a3e9896fba6f4816e38"
PLAN = "9e0aa7668dd45226bd69f56930423a4e75e856fa0e65e66183720481b5b6ace9"
ORDER = "65f00d6a0ee14007e78e794de8b81e1a8305d091d82b3c290094d3dfdd2fa9d2"
RESPONSE = "resp_07796c664d79b8f7016b080747141087d0ae405706b6db2c5c"
PAYLOAD = "ac09bcae3019cc81899e246e4769bc20ec1b764ac08e52cfc6b4c35dc8324c85"
OLD_PAUSE = "059523d356888364b9ff1a2a327c88707872ee531fd9d908cd3c39da09f02dfa"
OLD_TERMINAL = "08e38a2f6d75212228b067079e659ebce6993dd8556356fe2d42717f33a09210"
OLD_DECISION = "6c90c654c8fd9c197cce242b7a2127b6b2194b0704b1d81c44ed8677fdc0ec7d"
CHOICE_INDEX, REFERENCE_INDEX = 31, 0
CANDIDATE, MOLECULE, OBSERVED = "C-1232", "M-0083", "O-2392"
CONDITIONS = {CANDIDATE: {"sample_volume": "100", "flow": "10", "PE/EA": "20/1"},
              OBSERVED: {"sample_volume": "150", "flow": "5", "PE/EA": "20/1"}}
DATA_BINDINGS = ("seed", "round", "arm", "schema_version", "prompt_version", "checkpoint_sha256", "packet_sha256",
                 "wire_packet_sha256", "wire_representation", "mapping_sha256", "split_sha256", "state_sha256")
BUDGET = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 1}
SOURCES = tuple(STUDY / "scripts" / f"{module}.py" for module in ("run_v13_887_r8_targeted_repair", DISPATCHER))
WRONG_REFERENCE = {"reference_id": OBSERVED, "reference_kind": "observed",
                   "relationship": "single_factor_contrast", "changed_axis": "sample_volume"}
REQUIRED_REFERENCE = {"reference_id": OBSERVED, "reference_kind": "observed", "relationship": "multifactor_contrast"}


def guide():
    return (
        "Scoped correction for the already chosen A7 batch. Keep all 32 selected candidate IDs in their original order. "
        "The only rejected entry is choice index 31, C-1232 (M-0083), reference index 0 to the observed row O-2392. "
        "It was incorrectly characterized as a single-factor sample-volume contrast. Consult the same rows in the supplied table. "
        "Check both sample volume and flow, and give an accurate reason for the comparison without isolating either effect. "
        "Keep the observation reference identity. Return one JSON object with keys candidate_id, reference_index, reference, reason. "
        "The reference object contains reference_id, reference_kind, relationship and only fields required by that relationship. "
        "The reason must be one concise English statement specifying both rows' sample volumes in µL and flow rates in mL/min, "
        "naming each row by its ID just before stating that row's sample volume then flow. "
        "Do not emit a full batch, replace a candidate or observation, change another choice, or update hypotheses. This scoped JSON "
        "response replaces the general batch output format for this one correction."
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
    if (digest(text), digest(plan), choice.get("molecule_id"), choice["references"]) != (
        ANSWER, PLAN, MOLECULE, [WRONG_REFERENCE],
    ):
        raise PermissionError("the completed provider plan differs from the specific rejected contrast")
    return plan


def _checked_reason(reason):
    if type(reason) is not str or not 25 <= len(reason) <= 240 or reason != reason.strip() or any(
        ord(character) < 32 for character in reason
    ):
        return False
    facts = tuple((row.upper() if row else int(number), unit.lower().endswith("l"))
                  for row, number, unit in re.findall(
        r"\b(C-1232|O-2392)\b|(?<![\d.])(\d+)\s*([µμu]l|ml\s*/\s*min)\b", reason, re.I,
    ))
    selected = ((CANDIDATE, False), (100, True), (10, False))
    observed = ((OBSERVED, False), (150, True), (5, False))
    return facts in (selected + observed, observed + selected) and not re.search(
        r"\b(?:only|solely|single[- ]factor|isolat\w*|fixed|unchanged)\b", reason, re.I,
    )


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
        "original_answer_sha256": ANSWER, "original_result_sha256": RESULT, "canonical_plan_sha256": PLAN,
        "ordered_candidate_ids_sha256": ORDER, "repair_index": CHOICE_INDEX, "reference_index": REFERENCE_INDEX,
        "candidate_id": CANDIDATE, "reference_id": OBSERVED, "actual_changed_axes": ["sample_volume", "flow"],
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
        file_digest(turn / "decision.json")) != (OLD_PAUSE, OLD_TERMINAL, OLD_DECISION):
        raise PermissionError("the retired 887 pause, terminal or decision receipt changed")
    original = relay.bounded._request(active, turn, parent, approval, LANE)
    outbox = read(old / "bridge/outbox" / f"{ATTEMPT}.json")
    result = read(old / "bridge/results" / f"{ATTEMPT}.json")
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
        raise PermissionError("the old request differs from the archived L589 packet")
    evaluation = _validation(stage, plan)
    diagnosis = {"state": "rejected", "diagnosis": evaluation.as_dict()}
    errors = [(error["code"], error["path"], error.get("actual_changed_axes")) for error in evaluation.errors]
    role_rows = {CANDIDATE: stage.context.candidates[CANDIDATE], OBSERVED: stage.context.observed[OBSERVED]}
    if (evaluation.valid or evaluation.valid_choice_count != 31 or read(turn / "decision.json") != diagnosis
            or errors != [("single_factor_mismatch", "/choices/31/references/0/changed_axis", ["sample_volume", "flow"])]
            or any(row["molecule_id"] != MOLECULE or any(row["conditions"][k] != v for k, v in CONDITIONS[key].items())
                   for key, row in role_rows.items())):
        raise PermissionError("the historical rejection differs from the single authorized semantic error")
    prompt = seed_runner.PROMPT_PATH
    if (root / prompt).read_text().strip() != patch_system(original):
        raise PermissionError("the signed correction release may apply only the exact scoped guide")
    sources = {name: file_digest(root / name) for name in parent.source_sha256}
    if any(sources[name] != sha for name, sha in parent.source_sha256.items() if name != prompt.as_posix()):
        raise PermissionError("the correction release changed previously signed non-prompt sources")
    sources.update({path.as_posix(): file_digest(root / path) for path in SOURCES})
    metadata = _metadata(original)
    one = replace(transport, logical_replies=1, transport_attempts=1)
    profile = {**parent.profile, "target_labels": 621, "transport": asdict(one), "fixed_reference_repair": metadata}
    for key in ("audited_future_transport_replay", "replay_eligible_from_round", "sealed_relay_retry_seconds",
                "normalize_observed_reference_aliases"):
        profile.pop(key, None)
    previous_evidence = parent.predecessor
    proof = {"dispatcher_module": DISPATCHER, "seed": SEED, "round": ROUND, "logical_index": 0,
             "previous_run_directory": str(old), "previous_release_directory": str(old_root),
             "previous_contract_sha256": PARENT.contract, "previous_authorization_sha256": PARENT.approval,
             "previous_request_id": REQUEST, "model_identity": model_lock, "cache_directory": str(cache),
             "public_resource_receipt_sha256": cache_sha, "bridge_wait_seconds": previous_evidence["bridge_wait_seconds"],
             "reviewed_cached_import": previous_evidence["reviewed_cached_import"],
             "previous_prefix_files": prefix, "previous_prefix_sha256": digest(prefix),
             "previous_seed_pause_file_sha256": OLD_PAUSE, "previous_terminal_file_sha256": OLD_TERMINAL,
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
