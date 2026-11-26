"""Resume an existing V13 Mac spool with an explicitly approved scheduling addendum."""

import argparse
import os
from pathlib import Path

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import read_bridge_json
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import _private_directory, run_dispatcher
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import (
    ATTEMPT_ID, GPU_PYTHON, SHA256, SSHCourier, verify_info,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.dispatch_prepared_ipv4 import _approved
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.dispatch_prompt_pilot import PromptDispatcher, SharedKeyPool
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.dispatch_prompt_successor import (
    PILOT_CONTRACT_SHA256, _pool,
)


SCHEMA = "a7_ipv4_parallel_schedule_authorization_v1"
PARALLELISM = 3
JOURNAL_DIRS = ("jobs", "slots", "attempts", "results", "delivered")


def _existing_spool(directory, contract_sha, authorization_sha):
    _private_directory(directory)
    for folder in JOURNAL_DIRS:
        _private_directory(directory / folder)
    binding = read_bridge_json(directory / "run_binding.json")
    expected = {"contract_sha256": contract_sha, "authorization_sha256": authorization_sha, "mock": False}
    identity = read_bridge_json(directory / "dispatcher_identity.json")
    dispatcher_id = identity.get("dispatcher_id") if isinstance(identity, dict) else None
    if (binding != expected or binding.get("mock") is not False or type(identity) is not dict
            or set(identity) != {"dispatcher_id"}
            or not isinstance(dispatcher_id, str) or not ATTEMPT_ID.fullmatch(dispatcher_id)):
        raise PermissionError("the schedule update requires the existing authorized Mac dispatcher spool")
    return dispatcher_id


def _verify_schedule(approval, approval_sha, contract_sha, authorization_sha, launcher_sha, dispatcher_id):
    expected = {"schema": SCHEMA, "contract_sha256": contract_sha, "authorization_sha256": authorization_sha,
                "launcher_sha256": launcher_sha, "dispatcher_id": dispatcher_id,
                "previous_api_parallelism": 1, "api_parallelism": PARALLELISM,
                "authorize_paid_api_scheduling": True}
    if (type(approval) is not dict or digest(approval) != approval_sha
            or set(approval) != {*expected, "authorized_by", "authorization_reference"}
            or any(type(approval[key]) is not type(value) or approval[key] != value for key, value in expected.items())
            or any(not isinstance(approval[key], str) or not approval[key].strip() or len(approval[key]) > 300
                   for key in ("authorized_by", "authorization_reference"))):
        raise PermissionError("the exact schedule approval must authorize the original V13 spool and three API workers")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute-paid", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    parser.add_argument("--ssh-host", default="eit-hpc")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-run-dir", required=True)
    parser.add_argument("--remote-python", default=GPU_PYTHON)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--schedule-authorization", type=Path, required=True)
    parser.add_argument("--schedule-authorization-sha256", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if not all(SHA256.fullmatch(value) for value in (
        args.contract_sha256, args.authorization_sha256, args.schedule_authorization_sha256,
    )):
        parser.error("provide exact contract, original authorization and schedule authorization hashes")

    spools = Path.home() / ".config/purellm/spool"
    spool = spools / args.contract_sha256
    dispatcher_id = _existing_spool(spool, args.contract_sha256, args.authorization_sha256)
    schedule = read_bridge_json(args.schedule_authorization)
    _verify_schedule(schedule, args.schedule_authorization_sha256, args.contract_sha256,
                     args.authorization_sha256, file_digest(Path(__file__)), dispatcher_id)

    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = verify_info(courier.info(), expected_contract=args.contract_sha256, expected_auth=args.authorization_sha256)
    pilot_pool, _, pilot_transport = _pool(spools / PILOT_CONTRACT_SHA256)
    transport, identity, prompt_sha = _approved(info, args.remote_run_dir, pilot_transport)
    if read_bridge_json(spool / "provider_model_lock.json") != identity:
        raise PermissionError("the existing V13 model lock differs from its approved identity")
    pool = SharedKeyPool(pilot_pool.directory, pilot_pool.config)
    workers = min(PARALLELISM, pool.config.parallelism)
    if args.verify_only:
        print(dumps({"verified": True, "api_workers": workers, "contract_sha256": args.contract_sha256,
                     "schedule_authorization_sha256": args.schedule_authorization_sha256}))
        return
    dispatcher = PromptDispatcher(spool, info=info, contract_sha256=args.contract_sha256,
                                  transport=transport, mock=False, pool=pool)
    if dispatcher.dispatcher_id != dispatcher_id:
        raise PermissionError("the original dispatcher identity changed during startup")
    dispatcher.prompt_sha = prompt_sha
    print(dumps(run_dispatcher(dispatcher, courier, watch=True, workers=workers)))


if __name__ == "__main__":
    main()
