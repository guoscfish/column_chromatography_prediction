"""Mac entry point for a signed, qualified five-seed background cohort."""

import argparse
from contextlib import contextmanager
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_BACKGROUND, TransportConfig,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.background_dispatcher import BackgroundDispatcher
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import _private_directory, verify_local_source
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, SEEDS, SSH_API, STUDY, request_timeouts
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.key_pool import KeyPool, load_pool
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import sdk_status
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import GPU_PYTHON, SHA256, SSHCourier, verify_info

MODULE = "dispatch_prepared_background"


def _approved(info, run):
    record = info["contract"]
    profile = record["profile"]
    predecessor = record.get("predecessor") or {}
    qualification = predecessor.get("provider_qualification") or {}
    transport = TransportConfig(**profile["transport"])
    if (record.get("run_directory") != run or record.get("api_execution") != SSH_API
            or tuple(profile.get("confirmation_seeds", ())) != SEEDS or profile.get("arm") != "A7"
            or profile.get("memory") != "M2" or profile.get("target_labels") != 1485
            or profile.get("batch_size") != 32 or profile.get("provider_capability") != "verified_route"
            or transport.wire_mode != RESPONSES_BACKGROUND or profile.get("store") is not False
            or qualification.get("same_id_retrieval_completed") is not True
            or predecessor.get("dispatcher_module") != MODULE or not sdk_status(transport)["matches"]):
        raise PermissionError("the remote contract does not authorize this exact qualified A7 background route")
    verify_local_source(info, root=ROOT)
    for relative in (STUDY / "a7_gpu/background_transport.py", STUDY / "a7_gpu/background_dispatcher.py",
                     STUDY / f"scripts/{MODULE}.py"):
        if record["source_sha256"].get(relative.as_posix()) != file_digest(ROOT / relative):
            raise PermissionError("the installed background source differs from this authorization")
    return transport, request_timeouts(profile), qualification["model_identity"]


def _client_factory(config, policy):
    import httpx
    import openai

    # A POST acknowledges an asynchronous job. A GET fetches one status. Neither
    # depends on a single connection staying open for the full generation.
    timeout = httpx.Timeout(connect=policy.connect_seconds, pool=policy.connect_seconds,
                            write=policy.write_seconds, read=min(90, policy.read_idle_seconds))

    @contextmanager
    def client(key):
        with httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False) as http:
            with openai.OpenAI(api_key=key, base_url=config.base_url, timeout=timeout,
                               http_client=http, max_retries=0) as sdk:
                yield sdk
    return client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", default="eit-hpc")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--remote-python", default=GPU_PYTHON)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--key-pool-config", type=Path, required=True)
    parser.add_argument("--shared-key-pool-dir", type=Path, required=True)
    parser.add_argument("--spool-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=8)
    args = parser.parse_args()
    os.umask(0o077)
    if not SHA256.fullmatch(args.contract_sha256) or not SHA256.fullmatch(args.authorization_sha256):
        parser.error("exact contract and authorization hashes required")
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.info(), expected_contract=args.contract_sha256, expected_auth=args.authorization_sha256)
    config, timeouts, identity = _approved(info, args.remote_run_dir)
    if timeouts is None:
        raise PermissionError("the qualified route must bind its HTTP timeouts")
    pool_directory = args.shared_key_pool_dir.expanduser().absolute()
    _private_directory(pool_directory, create=True)
    pool = KeyPool(pool_directory, load_pool(args.key_pool_config.expanduser()))
    runner = BackgroundDispatcher(args.spool_dir, info=info, contract_sha256=args.contract_sha256,
                                  transport=config, mock=False, pool=pool, client_for_key=_client_factory(config, timeouts))
    once(runner.path / "provider_model_lock.json", identity)
    runner.watch(courier, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
