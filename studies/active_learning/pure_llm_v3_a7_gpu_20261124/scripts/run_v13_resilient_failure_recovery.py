"""Recover one exact failed resilient SSE boundary, then continue its original trajectory."""

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import os

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, read
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import ROOT, STUDY, verify_authorization
from . import run_v13_bounded_sse_recovery as bounded
from . import run_v13_resilient_sse as resilient


DISPATCHER = "dispatch_v13_resilient_failure_recovery"
OVERLAYS = resilient.SOURCES_ALLOWED | {(STUDY / name).as_posix() for name in (
    "a7_gpu/provider.py", "a7_gpu/dispatcher.py", "a7_gpu/ssh_courier.py",
    "scripts/dispatch_v13_resilient_sse.py",
)}


@dataclass(frozen=True)
class FailedRecovery:
    seed: int
    contract_sha256: str
    authorization_sha256: str
    request_id: str
    attempt_id: str
    job_sha256: str
    result_sha256: str
    code: str
    provider_error: str | None
    http_status: int | None
    sse_event_count: int | None
    provider_dispatched: bool = True
    provider_request_id: str | None = None
    response_id: str | None = None
    aborted_attempt_id: str | None = None

    @property
    def lane(self):
        return resilient.LANES[self.seed]

    @property
    def run_name(self):
        return f"{bounded.PREFIX}-{self.seed}-resilient-failure-01"

    @property
    def archived(self):
        return bounded.RunIdentity(f"-{self.seed}-resilient-sse-01", self.contract_sha256, self.authorization_sha256)


FAILURES = {
    157: FailedRecovery(
        157, "3220882105bb609a69e160ceac26673498a7b56f81c7604f28da03394b38f578",
        "8ee80f4e661cc20e1a86d42cf8bd62c86be39f449b7297c2b7a08859d0c5baf9",
        "c575dddc0f719c7c3a5e43c489e1803db3d878606912b4c18713793418abae1a",
        "b6b840a7a7064edeb095d4a6e2686571", "c83fc44afb732a682c7921e916c17a7e04c0e79bb1483a17baa6115208ec6a88",
        "3ae99f04fcdfb70a461f2c1f1817fb7c2057ecb875dc8540a2c8428c5005fd80", "connect_error", None, None, None,
        provider_dispatched=False, aborted_attempt_id="6755030e99c64959ad4f1ded1997d5c9",
    ),
    2357: FailedRecovery(
        2357, "7b18ce2e724a57e69b7837cea84804492f36a62228b42745c79e8ab2d202a9a6",
        "744811f8ed200e9cf4b563a2057b06c3fc7494a718d41235452372547ef29fe5",
        "1198d309ea31b3a8436985e4133c1365eec933a517c447d051d82a1c5409f45a",
        "fd65176edc9b4e59b87c347c30832f7c", "301a68125973199dad50b2e05535fa670a2b9fe4b4f37ec5e73df3c9d586bfdd",
        "9ebbdf7e79ae37a8501b49d1bac3aa99544b1e626b463fd3e9df88e8e0815198", "sdk_error", "APIError", 200, 5,
    ),
    887: FailedRecovery(
        887, "0c01992f30072c0cac2c3af36a89bc43378e87ffd54723efdef3e0404bcc19a6",
        "8d18696951379be8db01f0c00490eb8c36b69f77d22617a86375d7a834c61730",
        "bc7f144779e831a56d3cd0561a18f817ca62d7d26716dad227ade697704daa1f",
        "b0de2bfdcafe4be9900b43c21e733755", "ecd275c0f09d5588441c19c272c0a201fb13e3c7f04c7a8be8fb4f2bec7afb9d",
        "fddc9a5e99c0c257409d388f527d92bb779759f110f8d2daa5f0ac22a373fb50", "http_524", "http_524", 524, None,
    ),
    6101: FailedRecovery(
        6101, "10637fb22d3950ef6896bd118c08b21cd4b1eb825a423b62c153b83f12563adc",
        "35ec24a11486213c1959271fa28dd466a280014252c256a09c1a72f7b1a07046",
        "5e3133c10d4ea06b4bd7776c58abbe74f2c4350b581cb2b3b73a09f7cd363e02",
        "13eb0474c82d4cca825847830414fb50", "9d3afa0a064e6e36598af9ccff88bd5fc8d5758c0344853476b74d551895cd76",
        "ccf77bff4e1db96fdec42c2b55907bc5658c0422f10e02670100e00e08f777e3",
        "response_id_mismatch", "response_id_mismatch", 200, 41,
        provider_request_id="9c40f8c3-08ac-4f19-b5d1-ee58eb1b4b56",
        response_id="resp_0afd2d1a115ae7d1016b07e2a09b8887d295da02fe98ac25ee",
    ),
    12203: FailedRecovery(
        12203, "af5348664ad8dd786766e516b434a736a1d0cf4798165c4a4b3b09665e427911",
        "e5e8fb28c04663e94de92fbed624c1c10cc93cac73ad419488474278165a45d1",
        "5f4d9b604b0fb43430512d1e70787475a7d43f379c3280f73a3bb37e52f24676",
        "a2a3974d680844c2b3d8f61d9ece1f74", "e802ee42e0e23366d360c50f03c1d02e66e213560e9813ffbc1c3b57d8331dc9",
        "d641d12fc752c1f50b8068757e918c6977269ca8a9eb60d435f10725c8c91646",
        "response_id_mismatch", "response_id_mismatch", 200, 41,
        provider_request_id="04ff76e2-7444-43c9-84d5-045b59acc961",
        response_id="resp_0afd5a927289ce48016b07e286589c87d2ba2fbbc22d020ad0",
    ),
}


def _failure_scope(failed, preceding, request, original, receipt):
    expected = (failed.attempt_id, failed.job_sha256, failed.result_sha256, failed.code,
                failed.provider_dispatched, failed.provider_dispatched)
    if (request["request_id"] != failed.request_id or not resilient._same_science(request, original)
            or tuple(receipt[k] for k in ("attempt_id", "job_sha256", "result_sha256", "error",
                                          "provider_dispatched", "ambiguous")) != expected):
        raise PermissionError("the immediate parent must preserve the exact archived science and sealed failure")
    earlier = {k.removeprefix("previous_"): value for k, value in preceding.items() if k.startswith("previous_")}
    observation = {"http_status": failed.http_status, "error_type": failed.provider_error,
                   "sse_event_count": failed.sse_event_count}
    if failed.provider_request_id is not None:
        observation.update(provider_request_id=failed.provider_request_id, response_id=failed.response_id,
                           conflicting_response_id="resp_" + failed.provider_request_id.replace("-", ""))
    return {**preceding, "dispatcher_module": DISPATCHER, "archived_previous": earlier,
            "previous_contract_sha256": failed.contract_sha256, "previous_authorization_sha256": failed.authorization_sha256,
            "previous_request_id": failed.request_id, "previous_attempts": [receipt], "previous_attempt": receipt,
            "previous_transport_observation": observation}


def _failed_turn(run, active, failed):
    turn = active / "turn_00"
    aborted = turn / "attempt_01.started.json"
    if failed.aborted_attempt_id is None or not aborted.exists():
        return bounded._turn(active, 1), None
    bounded._turn(active, 2)
    started = read(aborted)
    name = f"{failed.aborted_attempt_id}.json"
    if ((started.get("attempt"), started.get("attempt_id"), started.get("request_id"), started.get("transport"))
            != (1, failed.aborted_attempt_id, failed.request_id, "mac_bridge")
            or any((run / "bridge" / folder / name).exists() for folder in ("outbox", "results", "claims"))
            or any(turn.glob("attempt_01.*terminal.json"))):
        raise PermissionError("the extra local reservation cannot be proven aborted before its bridge submission")
    return turn, {"attempt": 1, "attempt_id": failed.aborted_attempt_id,
                  "started_sha256": file_digest(aborted), "provider_dispatched": False}


def construct(root, failed_release, failed_run, run, failed):
    if run.name != failed.run_name or run.parent != failed_run.parent or root == failed_release:
        raise PermissionError("use the registered recovery sibling and its own immutable release")
    lane = failed.lane
    previous, approval = bounded._archived(failed_run, failed_release, failed.archived)
    proof = previous.predecessor
    original_release, original_run, older_release, older_run = (
        Path(proof[key]) for key in ("parent_release_directory", "parent_run_directory",
                                     "previous_release_directory", "previous_run_directory")
    )
    verified, boundary, cache, original = resilient.construct(
        failed_release, original_release, original_run, older_release, older_run, failed_run, lane,
    )
    if verified.record() != previous.record() or verified.digest != failed.contract_sha256:
        raise PermissionError("the exact previous resilient contract can no longer be reconstructed")
    _, active = bounded.earlier._uncommitted(failed_run, lane)
    turn, aborted = _failed_turn(failed_run, active, failed)
    request = bounded._request(active, turn, previous, approval, lane)
    reconstituted = bounded.earlier._request(failed_release, previous, approval, lane, boundary[0], boundary[1], original)
    if request != reconstituted:
        raise PermissionError("the failed scope does not match its archived L397/L525 boundary")
    receipt = bounded._failure(failed_run, turn, previous, approval, request, 0)
    successor = _failure_scope(failed, proof, request, original, receipt)
    if aborted is not None:
        successor["previous_aborted_reservation"] = aborted
    if (read(active / "source.json") != asdict(boundary[0]) or read(active / "boundary_memory.json") != boundary[1]
            or read(failed_run / "provider_model_lock.json") != proof["model_identity"]):
        raise PermissionError("the failed original fit, memory or served model identity changed")
    sources = {name: file_digest(root / name) for name in previous.source_sha256}
    if {name for name, actual in sources.items() if actual != previous.source_sha256[name]} - OVERLAYS:
        raise PermissionError("the new recovery changed code outside its authorized transport overlays")
    for module in ("run_v13_resilient_failure_recovery", DISPATCHER):
        name = (STUDY / "scripts" / f"{module}.py").as_posix()
        sources[name] = file_digest(root / name)
    successor.update(previous_run_directory=str(failed_run), previous_release_directory=str(failed_release))
    contract = replace(previous, source_sha256=sources, run_directory=str(run), predecessor=successor)
    return contract, boundary, cache, original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contract", "run", "resume"))
    parser.add_argument("--seed", type=int, choices=tuple(FAILURES), required=True)
    parser.add_argument("--failed-release", type=Path, required=True)
    parser.add_argument("--failed-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    failed_release, failed_run, run = (p.resolve() for p in (args.failed_release, args.failed_run, args.run_dir))
    failed = FAILURES[args.seed]
    contract, boundary, cache, original = construct(ROOT, failed_release, failed_run, run, failed)
    if args.action == "contract":
        print(dumps({"contract_sha256": contract.digest, **contract.record()}))
        return
    if args.authorization is None:
        parser.error("run and resume require the exact new recovery authorization")
    approval = verify_authorization(contract, read(args.authorization))
    parent = Path(contract.predecessor["parent_run_directory"])
    result = resilient.execute(ROOT, parent, failed_run, run, failed.lane, contract, approval, boundary,
                               cache, original, resume=args.action == "resume")
    print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
