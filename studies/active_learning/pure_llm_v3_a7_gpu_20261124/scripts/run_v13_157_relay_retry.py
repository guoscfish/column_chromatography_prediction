"""Resume 157 R3 with one new physical request after both archived relay failures."""

import argparse
from dataclasses import asdict, dataclass, replace
import os
from pathlib import Path
from time import time

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    _link_replay_attempt, make_request, terminal_record,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, STUDY, verify_authorization
from . import run_v13_relay_retry as retry


DISPATCHER = "dispatch_v13_157_relay_retry"
OLD = retry.bounded.RunIdentity("-157-relay-retry-01",
    "51022ae153326c4d3259ebb183632564184671955e3ca86646debdfc9615862d",
    "0ffd0471f4a5ed444f5e67a3c0bbf0e956c7c2fcaaec49a7d64840fff6a27a45")
RUN_NAME = f"{retry.bounded.PREFIX}-157-relay-retry-02"
REQUEST = "a39894954355f629d0fb102a9bf04086d6c4235a3d54a4e90350042bbea429a8"
PAYLOAD = "594ff66ced4fe668cdde6bcbdcc07e9f8cf49933a45accc89ef81a4c97e42107"
FAILED_UNIX = 1795683673.804615
NOT_BEFORE_UNIX = FAILED_UNIX + retry.COOLDOWN_SECONDS
RECOVERY_BUDGET = {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
REPLAY_FILES = {
    "attempt_00.replay_authorization.json": "823c1a3bb2eead727ffa842c81a7ffa52b7f5d02d82ccafc7bb7cdd0f400fe64",
    "attempt_00.replay.terminal.json": "11ec2f5a20b02678c5034db2a93fc73f207579797562d45eb7b55d6cdff8f761",
    "attempt_01.replay_link.json": "7d50b61ea9b110a16195f0f5db883612f4b0f046904a94fec00de4dc135bb67e",
}


@dataclass(frozen=True)
class PriorAttempt:
    attempt_id: str
    job_sha256: str
    result_sha256: str
    terminal_file_sha256: str
    trace_sha256: str
    provider_request_id: str
    response_id: str

    @property
    def observation(self):
        return {"error_type": "response_id_mismatch", "conflicting_event_type": "response.failed", "http_status": 200,
                "sse_event_count": 41, "provider_request_id": self.provider_request_id, "response_id": self.response_id,
                "conflicting_response_id": "resp_" + self.provider_request_id.replace("-", ""),
                "model": "gpt-6.2-astra", "response_id_matches_transport_alias": True}


PRIOR = (
    PriorAttempt("82772d4260c34644828f128a47361c0c",
        "157d0542dd6969cf0117980163a27bbb7ce8260b05fe70ec00cc295b6d6b42da",
        "405c5798bb5b62601db8dd9208867996de213ea4797611b11574b97eb9b8390d",
        "10d4a7458074ff1521026f377ec7a6677cbf9e7239eba2acd3ac0d7c2a04ed42",
        "b7725b88e950a6323b17fa57bd83bdcf0d17c691e22fdc77655190a01dcba736",
        "1e6ea4e0-c61f-4a8b-af12-baa65b0e7e18", "resp_0858e0ddfb638efe016b07f26bf77087d294c650cfcef28d51"),
    PriorAttempt("f5ead92b68594b23ae04b891547534cc",
        "e11cd1da6f518e3a0fa37537df9c1da3d76a47644764a4a2d1858bc52a8ce631",
        "01726eb1a162e6678546907aad160bb6d19aeaafef56ce5d8cf23e90671990bf",
        "4827695332f7c35a9bd821a3077406e9f4779127c1b873726fab5bf9b4a698cc",
        "a85b13ad9b2394da4ce6133a466f967b6ff20834234ae03bb302b2e5f014952d",
        "fc46b6b3-65b7-4e56-ba67-9b2fd79290e5", "resp_025819138fff10a0016b07f48ea85487d29f5ac12fce924b33"),
)


@dataclass(frozen=True)
class Boundary:
    seed: int = 157
    round: int = 3
    attempt_id: str = PRIOR[-1].attempt_id


LANE = Boundary()


def execution_budget(rounds=36):
    budget = retry.execution_budget(rounds, LANE.round)
    budget["maximum_physical_attempts"] -= 1
    return budget


def _failure(previous_run, active, previous, approval):
    turn = retry.bounded._turn(active, 2)
    request = retry.bounded._request(active, turn, previous, approval, LANE)
    receipts = []
    for index, prior in enumerate(PRIOR):
        receipt = retry.bounded._failure(previous_run, turn, previous, approval, request, index)
        terminal = turn / f"attempt_{index:02d}.terminal.json"
        expected = (prior.attempt_id, prior.job_sha256, prior.result_sha256, "response_id_mismatch", True, True)
        if (tuple(receipt[k] for k in ("attempt_id", "job_sha256", "result_sha256", "error", "provider_dispatched",
                                       "ambiguous")) != expected or file_digest(terminal) != prior.terminal_file_sha256):
            raise PermissionError("both prior physical outcomes must match the exact stopped failure receipts")
        receipts.append(receipt)
    latest = read(terminal)
    if ((request["request_id"], request["payload_sha256"]) != (REQUEST, PAYLOAD)
            or latest.get("failed_unix") != FAILED_UNIX or latest["not_before_unix"] > NOT_BEFORE_UNIX
            or any(file_digest(turn / path) != sha for path, sha in REPLAY_FILES.items())):
        raise PermissionError("the archived logical request, two-attempt replay chain or last failure time differs")
    authorization = digest(read(turn / "attempt_00.replay_authorization.json"))
    replay = terminal_record(turn, 0, replay_authorization_sha256=authorization)
    _link_replay_attempt(turn, read(turn / "attempt_01.started.json"), replay, allow_create=False)
    return request, receipts


def construct(root, failed_release, failed_run, run):
    if run.name != RUN_NAME or run.parent != failed_run.parent or root == failed_release:
        raise PermissionError("use the registered 157 successor and its own immutable release")
    previous, approval = retry.bounded._archived(failed_run, failed_release, OLD)
    old = previous.predecessor
    if (old["recovery_budget"] != retry.RECOVERY_BUDGET or old["execution_budget"] != retry.execution_budget(36, LANE.round)
            or previous.profile["sealed_relay_retry_seconds"] != retry.COOLDOWN_SECONDS
            or previous.profile["replay_eligible_from_round"] != LANE.round):
        raise PermissionError("the previous contract does not match the exhausted two-physical boundary")
    directory, active = retry._retired(failed_run, LANE, require_pause=False)
    request, receipts = _failure(failed_run, active, previous, approval)
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
        "previous_authorization_sha256": digest(approval), "previous_request_id": REQUEST, "previous_attempts": receipts,
        "previous_transport_observations": [item.observation for item in PRIOR],
        "previous_transport_progress_file_sha256": [item.trace_sha256 for item in PRIOR],
        "previous_terminal_file_sha256": [item.terminal_file_sha256 for item in PRIOR],
        "previous_replay_files": REPLAY_FILES, "previous_prefix_files": prefix,
        "previous_prefix_sha256": digest(prefix), "recovery_budget": RECOVERY_BUDGET,
        "execution_budget": execution_budget(), "relay_not_before_unix": NOT_BEFORE_UNIX,
        "prior_boundary_budget": {"physical_authorized": 2, "physical_already_used": 2, "physical_remaining": 0},
        "newly_authorized_boundary_physical_attempts": 1,
        "archived_previous": {key: value for key, value in old.items() if key.startswith("previous_")},
    }
    proof.pop("authoritative_message_reconstruction", None)
    if host is not None:
        proof["authoritative_message_reconstruction"] = host
    profile = {**previous.profile, "replay_eligible_from_round": index + 1}
    sources = {name: file_digest(root / name) for name in previous.source_sha256}
    if {path for path, actual in sources.items() if actual != previous.source_sha256[path]} - retry.OVERLAYS:
        raise PermissionError("the new attempt changed source outside audited provider and retry code")
    for module in ("run_v13_157_relay_retry", DISPATCHER):
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
        parser.error("run and resume require the exact new authorization for one additional physical attempt")
    approval = verify_authorization(contract, read(args.authorization))
    if time() < NOT_BEFORE_UNIX:
        raise PermissionError("the original 180-second relay cooldown is still in force")
    progress = retry.execute(ROOT, failed_run, run, LANE, contract, approval, boundary, cache, original,
                             resume=args.action == "resume")
    print(dumps(asdict(progress)))


if __name__ == "__main__":
    main()
