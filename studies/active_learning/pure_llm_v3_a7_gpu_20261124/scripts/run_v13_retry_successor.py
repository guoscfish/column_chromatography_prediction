"""Continue a reviewed failed boundary with exactly one new physical request."""

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
from . import run_v13_12203_relay_retry as wave02
from . import run_v13_157_relay_retry as wave157
from . import run_v13_relay_retry as retry


DISPATCHER = "dispatch_v13_retry_successor"
RECOVERY_BUDGET = wave02.RECOVERY_BUDGET


@dataclass(frozen=True)
class Review:
    identity: retry.bounded.RunIdentity
    request_id: str
    attempt_id: str
    job_sha256: str
    result_sha256: str
    terminal_file_sha256: str
    progress_file_sha256: str
    failure: str
    progress: dict
    earlier: tuple["Review", ...] = ()
    replay_files: dict | None = None

    @property
    def attempts(self):
        return (*self.earlier, self)


@dataclass(frozen=True)
class Scope:
    seed: int
    round: int
    wave: int
    payload_sha256: str
    archive: tuple[Review, ...]
    cooldown_field: str
    cooldown_unix: float
    original_start_round: int | None = None
    fitted_prefix_files: tuple[tuple[str, str], ...] = ()

    @property
    def run_name(self):
        return f"{retry.bounded.PREFIX}-{self.seed}-relay-retry-{self.wave:02d}"

    @property
    def attempt_id(self):
        return self.archive[-1].attempt_id

    @property
    def not_before_unix(self):
        return self.cooldown_unix + retry.COOLDOWN_SECONDS

    @property
    def budget(self):
        record = retry.execution_budget(36, self.round)
        record["maximum_physical_attempts"] -= 1
        return record


_157_PRIOR = tuple(Review(wave157.OLD, wave157.REQUEST, item.attempt_id, item.job_sha256, item.result_sha256,
                          item.terminal_file_sha256, item.trace_sha256, "response_id_mismatch", item.observation)
                   for item in wave157.PRIOR)

_6101_OLD = retry.bounded.RunIdentity("-6101-relay-retry-01",
    "978ef32b62d20f330b3fd9f4fa40afc9831c098d01a1a2ee013655ae9f18ab4d",
    "f8dc94f7d0c9c68b9023230675fcb1b476864ada0efc136123f991c88f1736f6")
_6101_REQUEST = "2f30189c779807c27f260f361864c4a88e777f0677b6d1cdaa81247023803303"
_6101_ATTEMPTS = tuple(Review(_6101_OLD, _6101_REQUEST, item.attempt_id, item.job_sha256, item.result_sha256,
                             item.terminal_file_sha256, item.trace_sha256, "response_id_mismatch", item.observation)
                       for item in (
    wave157.PriorAttempt("70aeee7f486449bba0e3344bece900cc",
        "0f7db8326ea1e093e10a9629c42fec235ef0f30a92583041b0d90722d7d93711",
        "7ad3f71eba6ad06fc442e87d5cbc65e98f99038af24635e983f021c1c07f89c9",
        "64a972a3d630c138d0b891eead861eb1ff71ee9c3d3bcab2289022dab6b2ad1d",
        "4c4aa9211f2c0b562023225ea6d91b66ae2e822b17207603635954e0a89f8972",
        "cd73475e-e4df-4d26-8421-4d4a3241caad", "resp_0fecee31ec692ca4016b07f695302c87d2ae8bc170e6fb334e"),
    wave157.PriorAttempt("2547ab46579b4341bc04f26550be98fc",
        "c1da929e150505f240cf13a6b590f605a6b6dc438bd23893fed4640ee7f7c70d",
        "755fedb86b30db88d74fb6c9eb45faeb4dd4c017f8b3ccf5d6c4e2bff5bfccfd",
        "42e00584598175ae64e7af0a569f7402a7071d7b9c73a5ab373264a91b82512f",
        "7b5b71d36c42c24bf39ea4ecce27d11e7b3c8d1dbe2ee57f963c1d97fd82c1bd",
        "fea68c61-2267-41ae-9992-9229df01bd45", "resp_0cd67c1e155d7c81016b07f82d785887d2b2fb1cde21b55276"),
))

SCOPES = {
    (6101, 2): Scope(
        6101, 2, 2, "f3f3dac061e9baca67095771305e0ea47827c20d35e7d1018c89059172533b1a", (
            replace(_6101_ATTEMPTS[-1], earlier=_6101_ATTEMPTS[:-1], replay_files={
                "attempt_00.replay_authorization.json": "ed3b198cd669492bfc478c373ce14b7a23e711bbe11b4cc61d60bcfd24699514",
                "attempt_00.replay.terminal.json": "5b6c603bab148aa3cc0ae301ceab2ce4adf6b5c60ee56c10f7135b32b9194a89",
                "attempt_01.replay_link.json": "34275ff1746b61630db679d5f829a1d8a68303195cbb471868e5fce842ff3af0",
            }),
        ), "failed_unix", 1795684592.3081465, original_start_round=1, fitted_prefix_files=(
            ("round_02/source.json", "c33ecb89a166f5f2d112f2bb434fab6529d4bf108fe0d4f84b28d3b7e39648b0"),
            ("round_02/transition.json", "90e14a3d1ab7fd8719c68b5ec86c3d3d580cf5c733a880bc6581f371995cf54c"),
            ("round_01/memory.json", "989f5e746a5398725bd4bcb37ee199fab8b1f13516e0e58c6d47b66365b07cec"),
        )),
    (157, 3): Scope(
        157, 3, 3, wave157.PAYLOAD, (
            replace(_157_PRIOR[-1], earlier=_157_PRIOR[:-1]),
            Review(retry.bounded.RunIdentity("-157-relay-retry-02",
                "a277fb5b7a0806c2f8f69443ad4b05821659783b8fa65d0b9ba6f7c814f052cd",
                "498d339f217f9b4e1d3d4b16cd1e4602048e3ad15715bca927fe1d6c41b797aa"),
                "d8fed3a9c76a3a4bc6fa27bac6a705b202b8d90d9565606ed65b1783046feb52", "ddd2a7ca64a74a75a20d198d72501cb1",
                "dabdd32ea8889eb86fd6ff2228ab66b8b6a3dc149a7087fbc839c2205bcdadd8",
                "5f0b04be5af4d1476810de3335c077034074e0e101d93f443ee9a2af3a6f2283",
                "0dcc718d9aa7ca2c4b6df268e04e6644c96103d9acd17c90f058ba0631b472cf",
                "b0b8b6ad3a6366329afb407ed8f6025d26983e594389dd032946ce8976d8dd7c", "sse_frame_error",
                {"phase": "transport_failure", "error_type": "ReadError", "http_status": 200, "sse_event_count": 7,
                 "provider_request_id": "e503bbf4-d451-4de0-ba50-d835437c913a",
                 "response_id": "resp_0dd4ac642b956f4a016b07f782808887d2a9bbd4fb60d1f9fc"}),
        ), "not_before_unix", 1795684354.7931683),
    (12203, 3): Scope(
        12203, 4, 3, wave02.PAYLOAD, (
            Review(wave02.OLD, wave02.REQUEST, wave02.ATTEMPT, wave02.JOB, wave02.RESULT, wave02.TERMINAL_FILE_SHA,
                   wave02.TRACE_SHA, "error", wave02.OBSERVATION),
            Review(retry.bounded.RunIdentity("-12203-relay-retry-02",
                "f373999a7c88aae8134bf2db212f6c52c8c7d75aedc1779cf5d4daede4b7a86c",
                "fe8cbc69384fb5561d63a91a72173882462d70ddaa87796990e8da24caae7680"),
                "3bd1d62bce670c46ad2b10584769a0bfdce2fa9d0d9256000a3b7bbdf1441b79", "ff520a3b471840e68a1f9ed34bee41f1",
                "0fafc1c0274a578b04eb5a21ac9a78981d72fefecfa313455d2b814a4f3516ad",
                "898014f9ac17cf60b55317aae8e4dadb56243c9ea689e1883b2c1c8044327803",
                "d62cef08ea21bbb26fa238eb1994093c15dc79aeacac9948f5b8f3bbb7130099",
                "cbda54406f564c8869a77f1ad09de5ada161f9b03d02cabb621bc367009bcfbd", "connection_interrupted",
                {"phase": "transport_failure", "error_type": "ReadError", "http_status": None,
                 "first_sse_seconds": None, "sse_event_count": None, "provider_request_id": None, "response_id": None}),
        ), "not_before_unix", 1795683909.169368),
}


def _evidence(run, active, contract, approval, scope, review):
    if review.identity == wave157.OLD:
        request, receipts = wave157._failure(run, active, contract, approval)
        return request, receipts, read(active / "turn_00/attempt_01.terminal.json")
    turn = retry.bounded._turn(active, len(review.attempts))
    request = retry.bounded._request(active, turn, contract, approval, scope)
    if (request["request_id"], request["payload_sha256"]) != (review.request_id, scope.payload_sha256):
        raise PermissionError("the reviewed logical request changed")
    receipts = []
    for index, attempt in enumerate(review.attempts):
        receipt = retry.bounded._failure(run, turn, contract, approval, request, index)
        terminal_path = turn / f"attempt_{index:02d}.terminal.json"
        if (file_digest(terminal_path) != attempt.terminal_file_sha256
                or tuple(receipt[k] for k in ("attempt_id", "job_sha256", "result_sha256", "error",
                                              "provider_dispatched", "ambiguous"))
                != (attempt.attempt_id, attempt.job_sha256, attempt.result_sha256, attempt.failure, True, True)):
            raise PermissionError("a reviewed stopped ambiguous receipt changed")
        receipts.append(receipt)
    if review.replay_files:
        if any(file_digest(turn / path) != sha for path, sha in review.replay_files.items()):
            raise PermissionError("the original delayed replay linkage changed")
        permit = digest(read(turn / "attempt_00.replay_authorization.json"))
        replay = terminal_record(turn, 0, replay_authorization_sha256=permit)
        _link_replay_attempt(turn, read(turn / "attempt_01.started.json"), replay, allow_create=False)
    return request, receipts, read(terminal_path)


def _archives(failed_run, failed_release, scope):
    run, release = failed_run, failed_release
    records = []
    for review in reversed(scope.archive):
        contract, approval = retry.bounded._archived(run, release, review.identity)
        directory, active = retry._retired(run, scope, require_pause=bool(records))
        request, receipts, terminal = _evidence(run, active, contract, approval, scope, review)
        if read(run / "provider_model_lock.json") != contract.predecessor["model_identity"]:
            raise PermissionError("the archived confirmed model identity changed")
        if records:
            current = records[-1]
            inherited = current["contract"].predecessor
            if (inherited["previous_contract_sha256"] != contract.digest
                    or inherited["previous_authorization_sha256"] != digest(approval)
                    or inherited["previous_attempts"] != receipts
                    or inherited["model_identity"] != contract.predecessor["model_identity"]
                    or not retry._compare_science(current["request"], request,
                                                  inherited.get("authoritative_message_reconstruction"))):
                raise PermissionError("a retained successor does not authenticate its immediate predecessor")
        records.append({"run": run, "release": release, "directory": directory, "active": active, "contract": contract,
                        "approval": approval, "request": request, "receipts": receipts, "terminal": terminal})
        run, release = (Path(contract.predecessor[k]) for k in ("previous_run_directory", "previous_release_directory"))
    return records


def construct(root, failed_release, failed_run, run, scope):
    if run.name != scope.run_name or run.parent != failed_run.parent or root == failed_release:
        raise PermissionError("use the registered successor and an independent immutable source release")
    records = _archives(failed_run, failed_release, scope)
    last = records[0]
    parent, approval = last["contract"], last["approval"]
    old = parent.predecessor
    parent_start = parent.profile["resume_boundaries"][str(scope.seed)]["round"]
    if scope.original_start_round is None:
        expected_budget = (scope.round, RECOVERY_BUDGET, scope.budget)
    else:
        expected_budget = (scope.original_start_round, retry.RECOVERY_BUDGET,
                           retry.execution_budget(36, scope.original_start_round))
    if (last["terminal"].get(scope.cooldown_field) != scope.cooldown_unix
            or (parent_start, old["recovery_budget"], old["execution_budget"]) != expected_budget
            or any(file_digest(last["directory"] / name) != sha for name, sha in scope.fitted_prefix_files)):
        raise PermissionError("the retired physical budget or original cooldown clock differs from its record")
    cache = Path(old["cache_directory"]).resolve(strict=True)
    if file_digest(cache / "resource_receipt.json") != old["public_resource_receipt_sha256"]:
        raise PermissionError("the original public training cache changed")
    boundary, prefix = retry._boundary(failed_release, last["directory"], last["active"], parent, approval, scope, cache)
    request = last["request"]
    state, memory, index = boundary
    binding = request["binding"]
    stage = seed_runner._stage(failed_release, state, memory, seed_runner._scope(parent, approval, scope.seed, False), parent)
    user = request["payload"]["input"][1]
    stage = seed_runner.pin_boundary_user_message(stage, user, digest(user))
    before, after = request["binding"][retry.ALIAS], stage.binding[retry.ALIAS]
    host = {"previous_sha256": before, "reconstructed_sha256": after} if before != after else None
    if not retry._compare_science(make_request(stage.binding, 0, stage.messages, parent.transport), request, host):
        raise PermissionError("the full prior scientific request cannot be reproduced")
    history = [{"contract_sha256": item.identity.contract, "authorization_sha256": item.identity.approval,
                "run_directory": str(record["run"]), "request_id": item.request_id, "terminal_file_sha256": item.terminal_file_sha256,
                "progress_file_sha256": item.progress_file_sha256, "previous_attempts": record["receipts"]}
               for item, record in zip(scope.archive, reversed(records), strict=True)]
    proof = {
        **old, "dispatcher_module": DISPATCHER, "seed": scope.seed, "round": index, "logical_index": 0,
        "payload_sha256": request["payload_sha256"], "previous_run_directory": str(failed_run),
        "previous_release_directory": str(failed_release), "previous_contract_sha256": parent.digest,
        "previous_authorization_sha256": digest(approval), "previous_request_id": request["request_id"],
        "previous_attempts": last["receipts"], "previous_terminal_file_sha256": scope.archive[-1].terminal_file_sha256,
        "previous_transport_progress_file_sha256": scope.archive[-1].progress_file_sha256,
        "previous_transport_observation": scope.archive[-1].progress, "previous_prefix_sha256": digest(prefix),
        "previous_prefix_files": prefix, "recovery_budget": RECOVERY_BUDGET, "execution_budget": scope.budget,
        "relay_not_before_unix": scope.not_before_unix, "relay_cooldown_clock": scope.cooldown_field,
        "newly_authorized_boundary_physical_attempts": 1, "reviewed_retry_chain": history,
    }
    proof.update({key: binding[key] for key in ("checkpoint_sha256", "packet_sha256", "wire_packet_sha256", "base_messages_sha256")})
    if scope.fitted_prefix_files:
        proof["reviewed_fitted_prefix_files"] = dict(scope.fitted_prefix_files)
    for stale in ("authoritative_message_reconstruction", "inherited_boundary_budget"):
        proof.pop(stale, None)
    if host:
        proof["authoritative_message_reconstruction"] = host
    descriptor = {"round": index, "state_sha256": digest(asdict(state)),
                  "memory_sha256": digest(memory), "user_message_sha256": digest(user)}
    profile = {**parent.profile, "replay_eligible_from_round": index + 1, "resume_boundaries": {str(scope.seed): descriptor}}
    sources = {name: file_digest(root / name) for name in parent.source_sha256}
    if {name for name, actual in sources.items() if actual != parent.source_sha256[name]} - retry.OVERLAYS:
        raise PermissionError("the successor changed source outside the audited transport paths")
    for module in ("run_v13_retry_successor", DISPATCHER, "run_v13_12203_relay_retry", "run_v13_157_relay_retry"):
        name = (STUDY / "scripts" / f"{module}.py").as_posix()
        sources[name] = file_digest(root / name)
    contract = replace(parent, profile=profile, source_sha256=sources, run_directory=str(run), predecessor=proof)
    return contract, boundary, cache, request


def read_scope(parser, args):
    scope = SCOPES.get((args.seed, args.wave))
    if scope is None:
        parser.error("this seed and wave have no individually reviewed prior attempt and finite scope")
    return scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run", "resume"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--wave", type=int, required=True)
    parser.add_argument("--failed-release", type=Path, required=True)
    parser.add_argument("--failed-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    scope = read_scope(parser, args)
    os.umask(0o077)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    failed_release, failed_run, run = (path.resolve() for path in (args.failed_release, args.failed_run, args.run_dir))
    contract, boundary, cache, original = construct(ROOT, failed_release, failed_run, run, scope)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        parser.error("run and resume require the exact new authorization for one physical attempt")
    approval = verify_authorization(contract, read(args.authorization))
    if time() < scope.not_before_unix:
        raise PermissionError("the original 180-second relay cooldown is still in force")
    result = retry.execute(ROOT, failed_run, run, scope, contract, approval, boundary, cache, original,
                           resume=args.action == "resume")
    print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
