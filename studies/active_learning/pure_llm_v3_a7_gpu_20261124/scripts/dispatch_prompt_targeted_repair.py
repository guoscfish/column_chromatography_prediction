#!/usr/bin/env python3
"""Dispatch the one approved 887 reference correction over direct HTTPS."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, file_digest, once
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import (
    read_bridge_json, validate_result, verify_job,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAT_SUFFIX
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import (
    Dispatcher, RemotePaused, _exclusive, _private_directory, verify_local_source,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import STUDY
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import (
    ATTEMPT_ID, GPU_PYTHON, SHA256, SSHCourier, verify_info,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.targeted_reference_repair import (
    CANDIDATE, CHOICE_INDEX, OBSERVATION, REFERENCE_INDEX, normalize_original, repair_guide,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.dispatch_prompt_direct import (
    _check_journal_scope, _check_seed_lease, _direct_transport,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.dispatch_prompt_successor import (
    DATA_BINDINGS, _approved_prompt_sha, _pool, _selection, _until,
)

SEED, ROUND = 887, 4
RUN_NAME = "a7-gpt62-astra-xhigh-ssh-v10-targeted-repair-887"
PARENT_CONTRACT = "b4f1d4cdb65e50dce57c4800258d9c4bf9732d3c5fbbf8eeaaba54d2f14c4a01"
PARENT_ATTEMPT = "8dab698f5847490cab3592131b5bab22"
PARENT_REQUEST = "540031c048a321fb9afd9f07a1c776c11e6a9032134fd07bde59e1a517ea5192"
PARENT_RESULT = "24c3b0d0d3ff7fdccef6d16fdb78ae74d96b9ec57e630c678d0efa31bd6176d9"
PARENT_RAW = "81e30ab883ff5a7fffd795660f9a0af0f806a71dd5baa289f978aefbf1a539b2"
PROJECTOR = STUDY / "a7_gpu/targeted_reference_repair.py"


def _parent(spool):
    _private_directory(spool)
    binding = read_bridge_json(spool / "run_binding.json")
    if binding.get("contract_sha256") != PARENT_CONTRACT or binding.get("mock") is not False:
        raise PermissionError("the parent spool must belong to the exact completed V9 attempt")
    filename = f"{PARENT_ATTEMPT}.json"
    job = read_bridge_json(spool / "jobs" / filename)
    request, transport, started = verify_job(job, contract_sha256=PARENT_CONTRACT,
                                              authorization_sha256=binding["authorization_sha256"])
    result = validate_result(read_bridge_json(spool / "results" / filename), job)
    attempt = spool / "attempts" / PARENT_ATTEMPT
    receipt = read_bridge_json(attempt / "provider_capture_receipt.json")
    raw = read_bridge_json(attempt / "raw_provider_response.json")
    ack = read_bridge_json(spool / "delivered" / filename)
    coords = request["binding"]["seed"], request["binding"]["round"], request["logical_index"], started["attempt"]
    if (coords != (SEED, ROUND, 0, 0) or request["request_id"] != PARENT_REQUEST
            or started["attempt_id"] != PARENT_ATTEMPT or result["outcome"]["status"] != "completed"
            or result["provider_dispatched"] is not True or result["result_sha256"] != PARENT_RESULT
            or receipt["attempt_id"] != PARENT_ATTEMPT or receipt["job_sha256"] != job["job_sha256"]
            or result["raw_response"] != raw or digest(raw) != PARENT_RAW
            or receipt["metadata"]["raw_response_sha256"] != PARENT_RAW
            or ack != {"attempt_id": PARENT_ATTEMPT, "result_sha256": PARENT_RESULT}):
        raise PermissionError("the approved repair requires the original archived V9 response and delivery proof")
    answer = result["outcome"]["reply"]["text"]
    plan, removed = normalize_original(answer)
    profile = {
        "original_request_id": PARENT_REQUEST, "original_attempt_id": PARENT_ATTEMPT,
        "original_raw_sha256": PARENT_RAW, "original_answer_sha256": digest(answer),
        "original_result_sha256": PARENT_RESULT, "canonical_plan_sha256": digest(plan),
        "ordered_candidate_ids_sha256": digest([choice["candidate_id"] for choice in plan["choices"]]),
        "repair_index": CHOICE_INDEX, "reference_index": REFERENCE_INDEX,
        "candidate_id": CANDIDATE, "reference_id": OBSERVATION, "changed_axis": "flow",
        "removed_nonsemantic_fields": removed,
    }
    return request, transport, profile


def _updated_system(original):
    prior = original["payload"]["input"][0]["content"]
    if not prior.endswith(FORMAT_SUFFIX):
        raise PermissionError("the original V9 system message has no registered compact input format")
    return prior.removesuffix(FORMAT_SUFFIX) + "\n\n" + repair_guide()


def _validate(dispatcher, job, original, repair_profile, system):
    request, _, started = dispatcher.validate(job)
    old, new = original["binding"], request["binding"]
    coords = new["seed"], new["round"], request["logical_index"], started["attempt"]
    expected_wire = {**original["payload"], "input": [
        {"role": "system", "content": system + FORMAT_SUFFIX}, original["payload"]["input"][1],
    ]}
    if (coords != (SEED, ROUND, 0, 0) or request["payload"] != expected_wire
            or new["prompt_sha256"] != digest(system) or new.get("fixed_reference_repair") != repair_profile
            or any(new[field] != old[field] for field in DATA_BINDINGS)
            or request["request_id"] == original["request_id"]):
        raise PermissionError("targeted repair must keep the approved user data and use only the registered system guide")
    return job


def run(args):
    deadline = time.monotonic() + args.wait_seconds
    pilot = args.pilot_spool.expanduser().resolve(strict=True)
    parent = args.parent_spool.expanduser().resolve(strict=True)
    spool = args.spool_dir.expanduser().resolve()
    if any(spool.is_relative_to(previous) or previous.is_relative_to(spool) for previous in (pilot, parent)):
        raise PermissionError("the targeted repair must use a new private Mac spool")
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = _until(courier.info, deadline, args.poll, "the targeted repair is unreachable over SSH")
    verify_info(info, expected_contract=args.contract_sha256, expected_auth=args.authorization_sha256)
    if info["contract"].get("run_directory") != args.remote_run_dir or Path(args.remote_run_dir).name != RUN_NAME:
        raise PermissionError("the targeted repair needs its own authorized run directory")
    verify_local_source(info, root=ROOT)
    pool, _, pilot_transport = _pool(pilot)
    original, original_transport, repair_profile = _parent(parent)
    transport = _direct_transport(info, SEED, pilot_transport, pilot_transport)
    system = _updated_system(original)
    contract = info["contract"]
    if (transport != original_transport or contract["profile"].get("fixed_reference_repair") != repair_profile
            or file_digest(ROOT / PROJECTOR) != contract["source_sha256"].get(PROJECTOR.as_posix())
            or _approved_prompt_sha(info) != digest(system)):
        raise PermissionError("the approved repair scope, implementation, prompt or transport differs")
    model_lock = read_bridge_json(parent / "provider_model_lock.json")
    if model_lock != read_bridge_json(pilot / "provider_model_lock.json"):
        raise PermissionError("the archived V9 model identity differs from the registered pilot")
    dispatcher = Dispatcher(spool, info=info, contract_sha256=args.contract_sha256,
                            transport=transport, mock=False, pool=pool)

    def validate(job):
        return _validate(dispatcher, job, original, repair_profile, system)

    def before_call(identity, job):
        if dispatcher.check_remote(courier.info())["paused"]:
            raise RemotePaused("the targeted repair coordinator is paused")
        if validate(courier.claim(identity, dispatcher.dispatcher_id)) != job:
            raise PermissionError("the remote claim does not match the targeted repair job")
        _check_seed_lease(pool, SEED)

    with _exclusive(spool / "dispatcher.lock"):
        once(spool / "provider_model_lock.json", model_lock)
        job = _selection(dispatcher, courier, validate, args.attempt_id, deadline, args.poll)
        _check_journal_scope(spool, job["attempt_id"], SEED)
        identity = dispatcher.accept(job)
        dispatcher.recover_captured()

        def dispatch():
            return dispatcher.dispatch(identity, before_call=before_call, pause=lambda: time.monotonic() >= deadline)

        status = _until(dispatch, deadline, args.poll, "the targeted repair could not claim its Key before timeout")
        view = {"seed": SEED, "round": ROUND, "logical_index": 0, "attempt_id": identity,
                "request_id": job["request_id"], "status": status}
        if status != "sealed":
            return view
        result = dispatcher.result(identity)
        if identity in dispatcher.undelivered():
            ack = _until(lambda: courier.put(identity, dispatcher.dispatcher_id, result), deadline, args.poll,
                         "targeted repair result is captured on Mac and awaits SSH delivery")
            dispatcher.mark_delivered(identity, result, ack)
        outcome = result["outcome"]
        return {**view, "delivered": True, "provider_dispatched": result["provider_dispatched"],
                "outcome": outcome["status"], "failure": outcome.get("failure", {}).get("code")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(SEED,), default=SEED)
    parser.add_argument("--ssh-host", default="eit-hpc")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--remote-python", default=GPU_PYTHON)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--pilot-spool", type=Path, required=True)
    parser.add_argument("--parent-spool", type=Path, required=True)
    parser.add_argument("--spool-dir", type=Path, required=True)
    parser.add_argument("--attempt-id")
    parser.add_argument("--wait-seconds", type=float, default=8500)
    parser.add_argument("--poll", type=float, default=2)
    args = parser.parse_args()
    if (not SHA256.fullmatch(args.contract_sha256) or not SHA256.fullmatch(args.authorization_sha256)
            or args.attempt_id is not None and not ATTEMPT_ID.fullmatch(args.attempt_id)
            or not 0 < args.poll <= args.wait_seconds or not math.isfinite(args.wait_seconds)):
        parser.error("provide exact authorization hashes, an optional attempt ID, and finite positive waits")
    os.umask(0o077)
    report = run(args)
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("outcome") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
