#!/usr/bin/env python3
"""Run a direct HTTPS Dispatcher for one authorized A7 V11 trajectory."""
import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import read_bridge_json
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAL_WIRE_REPRESENTATION
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import run_dispatcher, verify_local_source
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    SSH_API, STUDY, RequestTimeouts, request_timeouts,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import sdk_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import (
    GPU_PYTHON, SHA256, SSHCourier, verify_info,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.dispatch_prompt_pilot import (
    PromptDispatcher, SharedKeyPool,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.dispatch_prompt_successor import (
    _approved_prompt_sha, _pool,
)

BOUNDARIES = {157: 1, 887: 5, 2357: 1}
RUN_PREFIX = "a7-gpt62-astra-xhigh-ssh-v11-direct-full"
PROMPT_SHA256 = "0872b8ddd262b319d9a1463ca0696fa28a5afbd51da8cd3ff477729db6b08d40"
TIMEOUTS = RequestTimeouts(read_idle_seconds=1800, total_seconds=3600)
DIRECT_PROVIDER = STUDY / "a7_gpu/provider.py"
DIRECT_PROVIDER_SHA256 = "a0b4d47dc973aa5368acead4dc0157ac230f91cc39bf78b8794f468bf1623c11"
DISPATCH_SOURCE = STUDY / "scripts/dispatch_prompt_finish.py"
V5_DISPATCH_SOURCE = STUDY / "scripts/dispatch_prompt_pilot.py"


class FinishDispatcher(PromptDispatcher):
    def validate(self, job):
        request, config, started = super().validate(job)
        profile = self.info["contract"]["profile"]
        rounds = (profile["target_labels"] - profile["initial_labels"]) // profile["batch_size"]
        if request["binding"]["round"] >= rounds:
            raise PermissionError("the job exceeds the authorized L1485 trajectory")
        return request, config, started


def _approved_scope(info, seed, remote_run):
    contract = info["contract"]
    profile = contract["profile"]
    boundaries = profile.get("resume_boundaries", {})
    scientific = {"arm": "A7", "memory": "M2", "device": "cuda", "initial_labels": 333,
                  "target_labels": 1485, "batch_size": 32, "api_parallelism": 3, "store": False,
                  "wire_representation": FORMAL_WIRE_REPRESENTATION}
    if (contract.get("run_directory") != remote_run or Path(remote_run).name != f"{RUN_PREFIX}-{seed}"
            or contract.get("api_execution") != SSH_API
            or set(boundaries) != {str(seed)} or boundaries[str(seed)].get("round") != BOUNDARIES[seed]
            or profile.get("confirmation_seeds") not in ([seed], list(BOUNDARIES))
            or "fixed_reference_repair" in profile
            or any(type(profile.get(key)) is not type(value) or profile[key] != value
                   for key, value in scientific.items())):
        raise PermissionError("the remote contract is outside the approved V11 A7 trajectory")
    for relative in (DISPATCH_SOURCE, V5_DISPATCH_SOURCE):
        if contract["source_sha256"].get(relative.as_posix()) != file_digest(ROOT / relative):
            raise PermissionError("the installed V11 dispatch source differs from its approval")
    prompt = _approved_prompt_sha(info)
    if prompt != PROMPT_SHA256:
        raise PermissionError("the installed prompt differs from the approved improved V5 prompt")
    return prompt


def _approved_transport(info, pilot_transport):
    contract = info["contract"]
    profile = contract["profile"]
    transport = TransportConfig(**profile["transport"])
    expected = replace(pilot_transport, timeout_seconds=TIMEOUTS.total_seconds)
    if (transport != expected or (transport.provider, transport.requested_model, transport.served_model,
                                  transport.reasoning_effort, transport.logical_replies, transport.transport_attempts)
            != ("token4research", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh", 3, 2)
            or request_timeouts(profile) != TIMEOUTS):
        raise PermissionError("the V11 model, timeouts or finite request budget differs")
    if (contract["source_sha256"].get(DIRECT_PROVIDER.as_posix()) != DIRECT_PROVIDER_SHA256
            or file_digest(ROOT / DIRECT_PROVIDER) != DIRECT_PROVIDER_SHA256
            or not sdk_status(transport)["matches"]):
        raise PermissionError("the approved direct HTTPS provider and SDK must be installed")
    return transport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=tuple(BOUNDARIES), required=True)
    parser.add_argument("--ssh-host", default="eit-hpc")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--remote-python", default=GPU_PYTHON)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--pilot-spool", type=Path, required=True)
    parser.add_argument("--spool-dir", type=Path, required=True)
    args = parser.parse_args()
    if not SHA256.fullmatch(args.contract_sha256) or not SHA256.fullmatch(args.authorization_sha256):
        parser.error("provide exact contract and authorization hashes")
    os.umask(0o077)
    pilot = args.pilot_spool.expanduser().resolve(strict=True)
    spool = args.spool_dir.expanduser().resolve()
    if spool.is_relative_to(pilot) or pilot.is_relative_to(spool):
        raise PermissionError("V11 needs a new private Mac spool separate from its shared Key Pool")
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.info(), expected_contract=args.contract_sha256, expected_auth=args.authorization_sha256)
    verify_local_source(info, root=ROOT)
    prompt_sha = _approved_scope(info, args.seed, args.remote_run_dir)
    pilot_pool, _, pilot_transport = _pool(pilot)
    transport = _approved_transport(info, pilot_transport)
    pool = SharedKeyPool(pilot_pool.directory, pilot_pool.config)
    dispatcher = FinishDispatcher(spool, info=info, contract_sha256=args.contract_sha256,
                                  transport=transport, mock=False, pool=pool)
    dispatcher.prompt_sha = prompt_sha
    once(spool / "provider_model_lock.json", read_bridge_json(pilot / "provider_model_lock.json"))
    print(dumps(run_dispatcher(dispatcher, courier, watch=True, workers=1)))


if __name__ == "__main__":
    main()
