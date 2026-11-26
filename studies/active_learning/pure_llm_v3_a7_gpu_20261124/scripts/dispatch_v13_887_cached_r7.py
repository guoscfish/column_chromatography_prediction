"""Dispatch normal future rounds after the seed 887 cached round 7 import."""

import argparse
from dataclasses import replace
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import RESPONSES_STREAM_IPV4, TransportConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import read_bridge_json, verify_job
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import Dispatcher, run_dispatcher, verify_local_source
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, SSH_API, STUDY
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import sdk_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import GPU_PYTHON, SHA256, SSHCourier, verify_info
from .dispatch_prompt_direct import _check_seed_lease
from .dispatch_prompt_pilot import SharedKeyPool
from .dispatch_prompt_successor import PILOT_CONTRACT_SHA256, _approved_prompt_sha, _pool
from .dispatch_v13_bounded_sse_recovery import _spool
from .run_v13_887_cached_r7 import (
    ATTEMPTS, BUDGET, DISPATCHER, FINAL_SHA, IMPORT_BUDGET, OLD, ORDER_SHA, PATHS_SHA, PLAN_SHA,
    RAW_SHA, RESULT_SHA, ROUND, RUN_NAME, SEED,
)


def _scope(info, run):
    record = info["contract"]
    profile, proof = record["profile"], record["predecessor"]
    old = Path(run).with_name(OLD.name)
    scope = proof["cached_import"]
    prior = scope["prior"]
    if (Path(run).name != RUN_NAME or record["run_directory"] != run or record["api_execution"] != SSH_API
            or record["budget"] != BUDGET or proof["recovery_budget"] != IMPORT_BUDGET
            or proof["dispatcher_module"] != DISPATCHER or proof["previous_run_directory"] != str(old)
            or proof["previous_contract_sha256"] != OLD.contract or proof["previous_authorization_sha256"] != OLD.approval
            or (proof["seed"], proof["round"]) != (SEED, ROUND)
            or (profile["confirmation_seeds"], profile["target_labels"], profile["replay_eligible_from_round"])
            != ([SEED], 1485, ROUND + 1)
            or profile["resume_boundaries"][str(SEED)]["round"] != ROUND
            or scope["provider_post"] is not False or len(prior) != len(ATTEMPTS)
            or (scope["raw_response_sha256"], scope["final_text_sha256"], scope["normalized_plan_sha256"],
                digest(scope["removed_paths"]), scope["ordered_candidate_ids_sha256"])
            != (RAW_SHA, FINAL_SHA, PLAN_SHA, PATHS_SHA, ORDER_SHA)):
        raise PermissionError("this Mac scope allows only future rounds of the verified cached import")
    verify_local_source(info, root=ROOT)
    for module in (DISPATCHER, "run_v13_887_cached_r7"):
        path = STUDY / "scripts" / f"{module}.py"
        if file_digest(ROOT / path) != record["source_sha256"].get(path.as_posix()):
            raise PermissionError("the approved cached continuation source differs")
    return proof


def _archive(spools, proof):
    spool = _spool(spools, OLD.contract, OLD.approval)
    transport = None
    model = proof["model_identity"]
    if read_bridge_json(spool / "provider_model_lock.json") != model:
        raise PermissionError("the old provider model lock differs from the new authorization")
    for index, (item, (identity, request_id, job_sha)) in enumerate(zip(proof["cached_import"]["prior"], ATTEMPTS, strict=True)):
        name = f"{identity}.json"
        job = read_bridge_json(spool / "jobs" / name)
        result = read_bridge_json(spool / "results" / name)
        ack = read_bridge_json(spool / "delivered" / name)
        request, current, started = verify_job(job, contract_sha256=OLD.contract, authorization_sha256=OLD.approval)
        if ((item["attempt_id"], item["request_id"], item["job_sha256"]) != (identity, request_id, job_sha)
                or (job["attempt_id"], job["request_id"], job["job_sha256"]) != (identity, request_id, job_sha)
                or result["result_sha256"] != item["result_sha256"]
                or result["result_sha256"] != digest({key: value for key, value in result.items() if key != "result_sha256"})
                or ack != {"attempt_id": identity, "result_sha256": item["result_sha256"]}
                or (request["binding"]["seed"], request["binding"]["round"], request["logical_index"], started["attempt"])
                != (SEED, ROUND, index, 0) or (transport is not None and current != transport)):
            raise PermissionError("an original cached receipt or delivery acknowledgement differs")
        if index == 0 and (result["result_sha256"] != RESULT_SHA or result["raw_response_sha256"] != RAW_SHA
                           or digest(result["raw_response"]) != RAW_SHA or result["model_identity"] != model):
            raise PermissionError("the cached raw completion differs from the original")
        if index == 1 and result["outcome"]["status"] != "failed":
            raise PermissionError("leave the returned correction with the previous runner")
        transport = current
    return transport


class FutureDispatcher(Dispatcher):
    def validate(self, job):
        request, config, started = super().validate(job)
        binding = request["binding"]
        if (binding["seed"] != SEED or not ROUND < binding["round"] < 36
                or binding["prompt_sha256"] != self.prompt_sha
                or binding.get("normalize_observed_reference_aliases") is not True
                or not 0 <= request["logical_index"] < config.logical_replies
                or not 0 <= started["attempt"] < config.transport_attempts):
            raise PermissionError("the cached decision is excluded from Mac provider dispatch")
        return request, config, started

    def dispatch(self, attempt_id, *, before_call, pause=None):
        def check(identity, job):
            before_call(identity, job)
            _check_seed_lease(self.pool, SEED)

        return super().dispatch(attempt_id, before_call=check, pause=pause)


def run(args):
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.retry_exchange("info"), expected_contract=args.contract_sha256,
                       expected_auth=args.authorization_sha256)
    proof = _scope(info, args.remote_run_dir)
    spools = Path.home() / ".config/purellm/spool"
    archived = _archive(spools, proof)
    pilot, _, base = _pool(spools / PILOT_CONTRACT_SHA256)
    transport = TransportConfig(**info["contract"]["profile"]["transport"])
    if (args.contract_sha256 in {OLD.contract, PILOT_CONTRACT_SHA256} or archived != transport
            or transport != replace(base, timeout_seconds=transport.timeout_seconds, wire_mode=RESPONSES_STREAM_IPV4)
            or not sdk_status(transport)["matches"]):
        raise PermissionError("future rounds require a private spool and the original provider protocol")
    spool = spools / args.contract_sha256
    dispatcher = FutureDispatcher(spool, info=info, contract_sha256=args.contract_sha256, transport=transport,
                                  mock=False, pool=SharedKeyPool(pilot.directory, pilot.config))
    dispatcher.prompt_sha = _approved_prompt_sha(info)
    once(spool / "provider_model_lock.json", proof["model_identity"])
    return run_dispatcher(dispatcher, courier, watch=True, workers=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-paid", action="store_true", required=True)
    parser.add_argument("--ssh-host", default="eit-hpc")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--remote-python", default=GPU_PYTHON)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    args = parser.parse_args()
    if not all(SHA256.fullmatch(value) for value in (args.contract_sha256, args.authorization_sha256)):
        parser.error("provide the exact contract and authorization hashes")
    os.umask(0o077)
    print(dumps(run(args)))


if __name__ == "__main__":
    main()
