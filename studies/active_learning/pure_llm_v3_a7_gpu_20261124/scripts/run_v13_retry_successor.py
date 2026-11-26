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
SIGNED_RETRY_SOURCES = tuple(STUDY / "scripts" / f"{module}.py" for module in (
    "run_v13_retry_successor", DISPATCHER, "run_v13_relay_retry", "dispatch_v13_relay_retry",
    "run_v13_12203_relay_retry", "run_v13_157_relay_retry",
))
QUEUE_SOURCES = (
    ((STUDY / "a7_gpu/dispatcher.py").as_posix(), "cde7ba080ba728b7600764d733cfab8a87bbe8e537840929bc1f47cd6c2cb3fc"),
)


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
class CachedOrigin:
    ancestor: retry.bounded.RunIdentity
    proof_sha256: str
    receipt_file_sha256: str
    resource_sha256: str

    @property
    def review(self):
        return {"proof_sha256": self.proof_sha256, "receipt_file_sha256": self.receipt_file_sha256,
                "resource_contract_sha256": self.ancestor.contract, "resource_authorization_sha256": self.ancestor.approval,
                "provider_post": False}


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
    required_sources: tuple[tuple[str, str], ...] = ()
    cached_origin: CachedOrigin | None = None

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

SCOPES[(6101, 3)] = replace(SCOPES[(6101, 2)], wave=3, cooldown_field="not_before_unix", cooldown_unix=1795686404.2017465,
                              original_start_round=None, fitted_prefix_files=(), required_sources=QUEUE_SOURCES, archive=(
    *SCOPES[(6101, 2)].archive,
    Review(retry.bounded.RunIdentity("-6101-relay-retry-02",
        "19d8da15a90d182094a65f8693239061a71603958a34a352d58da0ae54aff595",
        "619b51b8177eb1cd071ba03dbb29a1e6c78f837ed7e250123a57cab032f4e1d7"),
        "0b921fac494749af3083c5ada05ea22f4be80a973a5e1db455d9811e7d439204", "094b37236e0f42b19cf30adcda11deb8",
        "592dd5651e0ac75f15968d288cba37d53713a9a456f67d9bbbf441781bf5a5d9",
        "cd93a0e2bb33b38528e8acdee763926d38ece22af79a89c71391ae0176777578",
        "2305a231866b9862b33d73c300c5580f05f8cacbc5fa01867c8a9c3e722e018d",
        "9f558de4e29358510a6f35254a82c3705c93b974eba37d30424ebbc7ed051948", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch",
         "sse_event_count": 41, "provider_request_id": "679f9855-2eda-4e86-9ea5-4fe75ed45719",
         "response_id": "resp_02c95f0c40e3ffc3016b07fd73ff1087d199a88a399a8c5583",
         "conflicting_response_id": "resp_679f98552eda4e869ea54fe75ed45719", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True}),
))

SCOPES[(157, 4)] = replace(SCOPES[(157, 3)], wave=4, cooldown_unix=1795685449.213155, archive=(
    *SCOPES[(157, 3)].archive,
    Review(retry.bounded.RunIdentity("-157-relay-retry-03",
        "269543cf13a42ca75542f5fd24311fbe437b0e30eb18c8b9c3711ae7b4b0aea6",
        "d93de895ea293405d4b6095a6399c6ded0d6f6005d660e31d90937dbbb7b535c"),
        "f6c9b03c951b91dcb38b42364a05dbb24e12af8e45eb34cb365ccdcada2d564e", "3a466e59049640aba687e95c19dc5054",
        "a6e211ffbc55414081fad2f38ac71c9c14f86535d730f7bdc6fc3be80afdb9db",
        "42b39ab5902e340d0df666232c5ac641185e50d6cf2e62067849cc271bdc9d83",
        "4cb10ccd3f8e9e487554d81b354d6c24f1a263758e3b87f9ef3e1fbfa7753159",
        "01437d7c316baec695cef4512effb38f539b5f404b333950fb034f41746c4d36", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch",
         "sse_event_count": 41, "provider_request_id": "d06bdd54-1c8e-4263-8f9c-680c5416de2a",
         "response_id": "resp_0ef680a41e384a91016b07fb37e25c87d295425992514870b2",
         "conflicting_response_id": "resp_d06bdd541c8e42638f9c680c5416de2a", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True}),
))

SCOPES[(157, 5)] = replace(SCOPES[(157, 4)], wave=5, cooldown_unix=1795686187.2452364, archive=(
    *SCOPES[(157, 4)].archive,
    Review(retry.bounded.RunIdentity("-157-relay-retry-04",
        "5445e576c203814df1aadad7b761d98405d5acc80894733b8840c45ece3aa2ac",
        "56b7acaef8144e4af34fd9cc90c33c815d00f2162be43c60bd23b1d1653b7be2"),
        "e9c640564d39591d242226f44d1bf60a408d92a67dc59c83336b9ae645a43f71", "8135ea87aba647e2af90541b2455cb43",
        "39bc1fa4acf97e2366dea34b6aa42b319f27422fcd0d1193b51ebe895e3d9208",
        "051c4598f9bb78b273188c40d1a3684f1373cf3c423c1da2946da06b9a803fa2",
        "af97c1b4949373341e7d7c13121db119c79283515c5de768629f9b9ed300aef2",
        "d65d849a9fea1c225073c00df258c6439a40818312e35fdc91553a59ef7aa17e", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch",
         "sse_event_count": 41, "provider_request_id": "db357841-73d5-4ff8-9e35-976dfaac3a86",
         "response_id": "resp_05dc474a5f199ecb016b07fd86c7d487d29ee2782852ebf58d",
         "conflicting_response_id": "resp_db35784173d54ff89e35976dfaac3a86", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True}),
), required_sources=QUEUE_SOURCES)

SCOPES[(12203, 4)] = replace(SCOPES[(12203, 3)], wave=4, cooldown_unix=1795685708.5045788, archive=(
    *SCOPES[(12203, 3)].archive,
    Review(retry.bounded.RunIdentity("-12203-relay-retry-03",
        "a62436a562a1007d441cd88aac5f92e2c53609a4232d49d4c6d87410dbe49598",
        "886e8993730af0dcafa5ce286e0e7701a6397e778324db72e48053efdf93b46a"),
        "b1dd1488f577af8ede100513f1f25647bbe5c4837d3e377d8cb2b7c0aef18722", "7f4f602283a14b5ebd89fdb98247e306",
        "ae6b8232e425c1f74afc6125588558c9b5a09b0be7774ee49d662d2df767ae5c",
        "03b6c645d58e3f42285b5d176bfeec3606a582f915b01a7d67e4ae55a30e556f",
        "0dd0f1b70b18755e4c825f705247cdc6f2aed747d78c565ce08cee555f912dc2",
        "5c8b9bb2953d7232f92e7d4cf9592113706d39f08187e9bb236d7cd72dc7f309", "timeout",
        {"http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
         "sse_event_count": 16, "provider_request_id": "b5b53dcc-c91b-42dd-8be1-4e689bd3f5f3",
         "response_id": "resp_0377120a5eb95a6d016b07f8c54f7887d2bf3179451cf31d5a"}),
), required_sources=QUEUE_SOURCES)

SCOPES[(12203, 5)] = replace(SCOPES[(12203, 4)], wave=5, cooldown_unix=1795686451.363517, archive=(
    *SCOPES[(12203, 4)].archive,
    Review(retry.bounded.RunIdentity("-12203-relay-retry-04",
        "5650df1c3ec2bfc02b3d414263bd99753f9882c93d8bc1910102cf262451055e",
        "e0772b0349f742db1934f33beedfc83c45d2f8b0cd043512257651d0e44bd4f9"),
        "078866fe9c060d20a54586a439458673db6c26d43ff88fa626c831c152d8a5a0", "57b87f54c0ce486c83b7d0a5ffcb1699",
        "d97c642d070c3d7524a45460ff913e210cc852575897ddde55fd5aaf916180ef",
        "dc3b965716c613d665136853bc6aa4a67a021c66ab6483645f913fef4d8813f1",
        "67de43373ff29535f34bdc399ff499f8033a33c20efc6ef6a399cf4bb234b1fe",
        "37c6dba9fdda950fb0c3179552026ed6b8a3df61eb31e05ef9368445c1dec665", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch",
         "sse_event_count": 41, "provider_request_id": "2df9333e-c775-46a7-8dfc-89aa54cf3749",
         "response_id": "resp_00271412b94c2133016b07ff67b75487d2ba227b0979fa30a1",
         "conflicting_response_id": "resp_2df9333ec77546a78dfc89aa54cf3749", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True}),
))

SCOPES[(887, 1)] = Scope(
    887, 8, 1, "ac09bcae3019cc81899e246e4769bc20ec1b764ac08e52cfc6b4c35dc8324c85", (
        Review(retry.bounded.RunIdentity("-887-cached-r7-01",
            "85f7448d3f2bf60e28b778be01cceca626fc9c26f03a5f32c1b6bf0fd2777763",
            "c87bd63366932852ea9a96c40346446fd8f5aa3091e68b851280a9bf67f62f9d"),
            "299309f8e73e0c13a720b0537a368c61ee691d9d109f828c49dcf5e24c8eea22", "fa7a605ce7f0490fbe8a39854981dd38",
            "4218b6a7f7dfbb4940aff87d879ed297578f09c77fa4775f65e13735e4b73ecc",
            "6b77162673d7ead51444c80c400a3c5195e4777beaf333b5fff5c6d200a31fe4",
            "09a9dd7c2ceb5aee0e6a3188c9ac9ce1cb6175debc530e015c8fed18b316f2c2",
            "e4b2958bfab5d6c0f75db71c545b88ee07899a72a5270a70c30d85f9a9d4d6ca", "response_id_mismatch",
            {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch",
             "sse_event_count": 41, "provider_request_id": "2b53b779-42cc-472e-8b2c-509604178690",
             "response_id": "resp_06e929eed61879f8016b07fc5dabdc87d2a1e2ec641e7fabc1",
             "conflicting_response_id": "resp_2b53b77942cc472e8b2c509604178690", "conflicting_event_type": "response.failed",
             "response_id_matches_transport_alias": True}),
    ), "not_before_unix", 1795685708.1872327, original_start_round=7, fitted_prefix_files=(
        ("round_08/source.json", "44b96e6448131a1a37eb3a0b2a80d8a13eaca7c7e16945ce85686f1b9eb039e2"),
        ("round_08/transition.json", "7ca5f9400e1a3152a0cc9e920bfdc6fb90739bf0a872d811b8d364bc39217c72"),
        ("round_07/memory.json", "771b3cbbe0f28005ff04a9dab30084035b9e1eebd70a3e5137dc9cee1d367de1"),
        ("round_07/batch_freeze.json", "1e0e41cb74a3525d7b30f3d80d1c6aae3e65d031eb8699b3f35c3caa404d37f4"),
        ("round_07/turn_00/attempt_00.terminal.json", "9324a781e9596a7ed994ad9e275aad7aca02cbfbfed04ddfafb6276430ef07dc"),
    ), required_sources=QUEUE_SOURCES, cached_origin=CachedOrigin(
        retry.bounded.RunIdentity("-887-resilient-failure-01",
            "f4e2782ba061c1c00dd9a1f46198380bc095492e38b467e6d9bf20b4b828b33d",
            "37bcb9bd06d31622ce29ff56363bf1d9bfb8772e2e3016e133fb3e9bcd29b7b3"),
        "20c979db30ef6f959e873df178d42cef2ab5b000d857e564b59e92b4d6b8d5b7",
        "cf345efd407710e3a11d34196b5bd7befda0a229e552f1745d9848b1c5dccb28",
        "c44a0d1a1f2961e99c5f1cd0032599259de09eef77a3aed93480cfb658ec5cf1",
    ),
)
SCOPES[(887, 2)] = replace(SCOPES[(887, 1)], wave=2)


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


def _cached_resources(failed_run, directory, old, scope):
    origin = scope.cached_origin
    prior_run, prior_release = (Path(old[key]) for key in ("previous_run_directory", "previous_release_directory"))
    prior, _ = retry.bounded._archived(prior_run, prior_release, origin.ancestor)
    receipt = failed_run / "cached_response_import.json"
    terminal = read(directory / f"round_{scope.original_start_round:02d}/turn_00/attempt_00.terminal.json")
    resources = prior.predecessor
    if ((old["previous_contract_sha256"], old["previous_authorization_sha256"])
            != (origin.ancestor.contract, origin.ancestor.approval) or prior_run.parent != failed_run.parent
            or resources["model_identity"] != old["model_identity"] or resources["cache_directory"] != old["cache_directory"]
            or resources["public_resource_receipt_sha256"] != origin.resource_sha256
            or digest(old["cached_import"]) != origin.proof_sha256 or file_digest(receipt) != origin.receipt_file_sha256
            or read(receipt) != old["cached_import"] or old["cached_import"].get("provider_post") is not False
            or terminal.get("status") != "completed" or terminal.get("provider_dispatched") is not False
            or (failed_run / "bridge/outbox" / f"{terminal['attempt_id']}.json").exists()):
        raise PermissionError("the cached fit or its original no-POST and resource evidence changed")
    return resources


def _source_manifest(root, parent, scope):
    sources = {name: file_digest(root / name) for name in parent.source_sha256}
    launchers = {(STUDY / "scripts" / f"{module}.py").as_posix() for module in ("run_v13_retry_successor", DISPATCHER)}
    if {name for name, actual in sources.items() if actual != parent.source_sha256[name]} - retry.OVERLAYS - launchers:
        raise PermissionError("the successor changed source outside the audited transport paths")
    if any(sources.get(name) != expected for name, expected in scope.required_sources):
        raise PermissionError("the successor is missing the exact reviewed transport source")
    sources.update({path.as_posix(): file_digest(root / path) for path in SIGNED_RETRY_SOURCES})
    return sources


def construct(root, failed_release, failed_run, run, scope):
    if run.name != scope.run_name or run.parent != failed_run.parent or root == failed_release:
        raise PermissionError("use the registered successor and an independent immutable source release")
    records = _archives(failed_run, failed_release, scope)
    last = records[0]
    parent, approval = last["contract"], last["approval"]
    old = parent.predecessor
    parent_start = parent.profile["resume_boundaries"][str(scope.seed)]["round"]
    if scope.cached_origin is not None:
        remaining = parent.rounds - scope.round
        expected_budget = (scope.original_start_round, {"maximum_logical_requests": 1, "maximum_physical_attempts": 0},
                           {"first_logical_requests": remaining, "maximum_logical_requests": 3 * remaining,
                            "maximum_physical_attempts": 6 * remaining})
    elif scope.original_start_round is None:
        expected_budget = (scope.round, RECOVERY_BUDGET, scope.budget)
    else:
        expected_budget = (scope.original_start_round, retry.RECOVERY_BUDGET,
                           retry.execution_budget(36, scope.original_start_round))
    if (last["terminal"].get(scope.cooldown_field) != scope.cooldown_unix
            or (parent_start, old["recovery_budget"], old["execution_budget"]) != expected_budget
            or any(file_digest(last["directory"] / name) != sha for name, sha in scope.fitted_prefix_files)):
        raise PermissionError("the retired physical budget or original cooldown clock differs from its record")
    resources = old if scope.cached_origin is None else _cached_resources(failed_run, last["directory"], old, scope)
    cache = Path(old["cache_directory"]).resolve(strict=True)
    if file_digest(cache / "resource_receipt.json") != resources["public_resource_receipt_sha256"]:
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
        "public_resource_receipt_sha256": resources["public_resource_receipt_sha256"],
        "bridge_wait_seconds": resources["bridge_wait_seconds"],
    }
    proof.update({key: binding[key] for key in ("checkpoint_sha256", "packet_sha256", "wire_packet_sha256", "base_messages_sha256")})
    if scope.fitted_prefix_files:
        proof["reviewed_fitted_prefix_files"] = dict(scope.fitted_prefix_files)
    if scope.cached_origin is not None:
        proof.pop("cached_import")
        proof["reviewed_cached_import"] = scope.cached_origin.review
    for stale in ("authoritative_message_reconstruction", "inherited_boundary_budget"):
        proof.pop(stale, None)
    if host:
        proof["authoritative_message_reconstruction"] = host
    descriptor = {"round": index, "state_sha256": digest(asdict(state)),
                  "memory_sha256": digest(memory), "user_message_sha256": digest(user)}
    profile = {**parent.profile, "replay_eligible_from_round": index + 1, "resume_boundaries": {str(scope.seed): descriptor},
               "sealed_relay_retry_seconds": retry.COOLDOWN_SECONDS}
    sources = _source_manifest(root, parent, scope)
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
