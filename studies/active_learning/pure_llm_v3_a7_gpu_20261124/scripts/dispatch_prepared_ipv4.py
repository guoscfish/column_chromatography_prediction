"""Serial Mac dispatcher for the authorized five-seed direct IPv4 continuation."""

import argparse
from dataclasses import replace
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_STREAM_IPV4, TransportConfig,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAL_WIRE_REPRESENTATION
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import run_dispatcher, verify_local_source
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    ROOT, SEEDS, SSH_API, STUDY, request_timeouts,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import sdk_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import (
    GPU_PYTHON, SHA256, SSHCourier, verify_info,
)
from .dispatch_prompt_pilot import PromptDispatcher, SharedKeyPool
from .dispatch_prompt_successor import _approved_prompt_sha, _pool


MODULE = "dispatch_prepared_ipv4"
SOURCES = tuple(STUDY / "scripts" / f"{name}.py" for name in (
    MODULE, "dispatch_prompt_pilot", "dispatch_prompt_successor",
))


def _approved(info, run, pilot_transport):
    record = info["contract"]
    profile = record["profile"]
    predecessor = record.get("predecessor") or {}
    expected_scope = {
        "provider_capability": "gated_ipv4", "arm": "A7", "memory": "M2", "device": "cuda",
        "initial_labels": 333, "target_labels": 1485, "batch_size": 32, "api_parallelism": 1,
        "store": False, "confirmation_seeds": list(SEEDS),
        "wire_representation": FORMAL_WIRE_REPRESENTATION,
    }
    if (record.get("run_directory") != run or record.get("api_execution") != SSH_API
            or predecessor.get("dispatcher_module") != MODULE
            or set(profile.get("resume_boundaries", {})) != set(map(str, SEEDS))
            or "fixed_reference_repair" in profile
            or any(type(profile.get(key)) is not type(value) or profile[key] != value
                   for key, value in expected_scope.items())):
        raise PermissionError("the remote contract does not authorize this serial A7 IPv4 dispatcher")

    transport = TransportConfig(**profile["transport"])
    timing = request_timeouts(profile)
    if (timing is None or
            transport != replace(pilot_transport, timeout_seconds=timing.total_seconds, wire_mode=RESPONSES_STREAM_IPV4)
            or (transport.provider, transport.base_url, transport.requested_model, transport.served_model,
                transport.reasoning_effort, transport.max_output_tokens, transport.logical_replies,
                transport.transport_attempts) != (
                    "token4research", "https://token4research.cn/v1", "gpt-6.2-astra", "gpt-6.2-astra",
                    "xhigh", 32768, 3, 2,
                )
            or not sdk_status(transport)["matches"]):
        raise PermissionError("the exact approved IPv4 transport, limits and SDK must be installed")

    qualification = predecessor.get("provider_qualification") or {}
    identity = qualification.get("model_identity") or {}
    expected_identity = {
        "provider": transport.provider, "requested_model": transport.requested_model,
        "served_model": transport.served_model, "requested_reasoning_effort": transport.reasoning_effort,
    }
    if (set(identity) != {*expected_identity, "reported_model_version", "reported_reasoning_effort"}
            or any(identity[key] != value for key, value in expected_identity.items())
            or identity["reported_reasoning_effort"] not in (None, transport.reasoning_effort)):
        raise PermissionError("the approved historical model lock differs from this IPv4 continuation")

    verify_local_source(info, root=ROOT)
    for relative in SOURCES:
        if record["source_sha256"].get(relative.as_posix()) != file_digest(ROOT / relative):
            raise PermissionError("the installed serial dispatcher source differs from the approved release")
    return transport, identity, _approved_prompt_sha(info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-paid", action="store_true", required=True)
    parser.add_argument("--ssh-host", default="eit-hpc")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--remote-python", default=GPU_PYTHON)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--pilot-spool", type=Path, required=True)
    parser.add_argument("--spool-dir", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if not SHA256.fullmatch(args.contract_sha256) or not SHA256.fullmatch(args.authorization_sha256):
        parser.error("provide exact contract and authorization hashes")
    pilot = args.pilot_spool.expanduser().resolve(strict=True)
    spool = args.spool_dir.expanduser().resolve()
    if spool.is_relative_to(pilot) or pilot.is_relative_to(spool):
        raise PermissionError("IPv4 continuation needs a new private spool separate from the original shared Key Pool")

    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.info(), expected_contract=args.contract_sha256, expected_auth=args.authorization_sha256)
    pilot_pool, _, pilot_transport = _pool(pilot)
    transport, identity, prompt_sha = _approved(info, args.remote_run_dir, pilot_transport)
    pool = SharedKeyPool(pilot_pool.directory, pilot_pool.config)
    dispatcher = PromptDispatcher(spool, info=info, contract_sha256=args.contract_sha256,
                                  transport=transport, mock=False, pool=pool)
    dispatcher.prompt_sha = prompt_sha
    once(spool / "provider_model_lock.json", identity)
    print(dumps(run_dispatcher(dispatcher, courier, watch=True, workers=1)))


if __name__ == "__main__":
    main()
