"""Move only the unspent 12203 R4 relay slot to an independently signed successor."""

import argparse
from dataclasses import asdict, dataclass, replace
import os
from pathlib import Path
from time import time

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import make_request
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, STUDY, verify_authorization
from . import run_v13_relay_retry as retry


DISPATCHER = "dispatch_v13_12203_relay_retry"
OLD = retry.bounded.RunIdentity("-12203-relay-retry-01",
    "1f1f225de4f396e639ba97bd56b2ab145307e66626b316b115ba7c82beaf11f5",
    "77b8f85d33eed696a3ec19147804a00a963b37bea56b3fcdb2eb5d6dfbf0f16d")
RUN_NAME = f"{retry.bounded.PREFIX}-12203-relay-retry-02"
ATTEMPT = "bc63c0206e284af3b707fc725c213f2e"
REQUEST = "82e9c57d8543ab15d8161a379e7d6496a04fe7679b521b7e505861f48d4ed514"
PAYLOAD = "0bd98eeac5b9b9049b7e00d7e8e439d0d1c07e7bd419370440d0259dfc91e1f5"
JOB = "92b91b93570b344bc4be91a675c1f12ca0cadbdc4b9bda3cf5ea7617cf27e232"
RESULT = "4d60143bed2f089d3f9ba72c5f960f0b53be49754559f89bfee7b08d3bf0d420"
TERMINAL_FILE_SHA = "4abb86ed9cc04506fe0e0f3e9110fbb7cba2ad52503b1d551bdb9073339de07b"
TRACE_SHA = "88d1642871364a746b6a452f2dcb8fc3fe06619dc8969c45b60e41b53fe6519f"
FAILED_UNIX = 1795683178.4218476
NOT_BEFORE_UNIX = FAILED_UNIX + retry.COOLDOWN_SECONDS
RECOVERY_BUDGET = {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
OBSERVATION = {"http_status": 200, "phase": "transport_failure", "event_type": "error", "error_type": "error",
               "provider_error_code": "stream_timeout", "sse_event_count": 14, "model": "gpt-6.2-astra",
               "provider_request_id": "6aa2e132-afca-45fc-b620-afbfaef37064",
               "response_id": "resp_00d1a4d1c3b15b2f016b07f26d843c87d185f77aebc2e55d83"}


@dataclass(frozen=True)
class Boundary:
    seed: int = 12203
    round: int = 4
    attempt_id: str = ATTEMPT


LANE = Boundary()


def execution_budget(rounds=36):
    budget = retry.execution_budget(rounds, LANE.round)
    budget["maximum_physical_attempts"] -= 1
    return budget


def _failure(previous_run, active, previous, approval):
    turn = retry.bounded._turn(active, 1)
    request = retry.bounded._request(active, turn, previous, approval, LANE)
    receipt = retry.bounded._failure(previous_run, turn, previous, approval, request, 0)
    terminal = turn / "attempt_00.terminal.json"
    failed = read(terminal)
    if (tuple(receipt[k] for k in ("attempt_id", "job_sha256", "result_sha256", "error", "provider_dispatched",
                                   "ambiguous")) != (ATTEMPT, JOB, RESULT, "error", True, True)
            or (request["request_id"], request["payload_sha256"]) != (REQUEST, PAYLOAD)
            or file_digest(terminal) != TERMINAL_FILE_SHA
            or failed.get("failed_unix") != FAILED_UNIX or failed.get("not_before_unix", NOT_BEFORE_UNIX) > NOT_BEFORE_UNIX):
        raise PermissionError("the signed original stream-timeout receipt differs")
    return request, receipt


def construct(root, failed_release, failed_run, run):
    if run.name != RUN_NAME or run.parent != failed_run.parent or root == failed_release:
        raise PermissionError("use the registered 12203 successor and its own immutable release")
    previous, approval = retry.bounded._archived(failed_run, failed_release, OLD)
    old = previous.predecessor
    if (old["recovery_budget"] != retry.RECOVERY_BUDGET or old["execution_budget"] != retry.execution_budget(36, LANE.round)
            or previous.profile["sealed_relay_retry_seconds"] != retry.COOLDOWN_SECONDS
            or previous.profile["replay_eligible_from_round"] != LANE.round):
        raise PermissionError("the previously signed boundary does not have exactly one remaining physical slot")
    directory, active = retry._retired(failed_run, LANE, require_pause=False)
    request, receipt = _failure(failed_run, active, previous, approval)
    cache = Path(old["cache_directory"]).resolve(strict=True)
    if file_digest(cache / "resource_receipt.json") != old["public_resource_receipt_sha256"]:
        raise PermissionError("the original public training cache changed")
    boundary, prefix = retry._boundary(failed_release, directory, active, previous, approval, LANE, cache)
    state, memory, index = boundary
    scope = seed_runner._scope(previous, approval, LANE.seed, False)
    user = request["payload"]["input"][1]
    stage = seed_runner._stage(failed_release, state, memory, scope, previous)
    stage = seed_runner.pin_boundary_user_message(stage, user, digest(user))
    rebuilt = make_request(stage.binding, 0, stage.messages, previous.transport)
    old_message = request["binding"][retry.ALIAS]
    new_message = stage.binding[retry.ALIAS]
    host = {"previous_sha256": old_message, "reconstructed_sha256": new_message} if old_message != new_message else None
    if (not retry._compare_science(rebuilt, request, host)
            or read(failed_run / "provider_model_lock.json") != old["model_identity"]):
        raise PermissionError("the exact archived scientific input or served model does not match")
    proof = {
        **old, "dispatcher_module": DISPATCHER, "previous_run_directory": str(failed_run),
        "previous_release_directory": str(failed_release), "previous_contract_sha256": previous.digest,
        "previous_authorization_sha256": digest(approval), "previous_request_id": REQUEST, "previous_attempts": [receipt],
        "previous_transport_observation": OBSERVATION, "previous_transport_progress_file_sha256": TRACE_SHA,
        "previous_terminal_file_sha256": TERMINAL_FILE_SHA, "previous_prefix_files": prefix,
        "previous_prefix_sha256": digest(prefix), "recovery_budget": RECOVERY_BUDGET,
        "execution_budget": execution_budget(), "relay_not_before_unix": NOT_BEFORE_UNIX,
        "inherited_boundary_budget": {"physical_authorized": 2, "physical_already_used": 1, "physical_remaining": 1},
        "archived_previous": {key: value for key, value in old.items() if key.startswith("previous_")},
    }
    proof.pop("authoritative_message_reconstruction", None)
    if host is not None:
        proof["authoritative_message_reconstruction"] = host
    profile = {**previous.profile, "replay_eligible_from_round": index + 1}
    sources = {name: file_digest(root / name) for name in previous.source_sha256}
    if {path for path, actual in sources.items() if actual != previous.source_sha256[path]} - retry.OVERLAYS:
        raise PermissionError("the new attempt changed source outside audited provider and retry code")
    for module in ("run_v13_12203_relay_retry", DISPATCHER):
        path = (STUDY / "scripts" / f"{module}.py").as_posix()
        sources[path] = file_digest(root / path)
    contract = replace(previous, profile=profile, source_sha256=sources, run_directory=str(run), predecessor=proof)
    return contract, boundary, cache, request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run", "resume"))
    parser.add_argument("--failed-release", type=Path, required=True)
    parser.add_argument("--failed-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    failed_release, failed_run, run = (path.resolve() for path in (args.failed_release, args.failed_run, args.run_dir))
    contract, boundary, cache, original = construct(ROOT, failed_release, failed_run, run)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        parser.error("run and resume require the exact new authorization inheriting one remaining slot")
    approval = verify_authorization(contract, read(args.authorization))
    if time() < NOT_BEFORE_UNIX:
        raise PermissionError("the original 180-second relay cooldown is still in force")
    progress = retry.execute(ROOT, failed_run, run, LANE, contract, approval, boundary, cache, original,
                             resume=args.action == "resume")
    print(dumps(asdict(progress)))


if __name__ == "__main__":
    main()
