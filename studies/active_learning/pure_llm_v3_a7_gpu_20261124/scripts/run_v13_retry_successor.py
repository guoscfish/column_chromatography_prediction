"""Continue a reviewed failed boundary within its signed physical request budget."""

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
PROVIDER_SOURCE = (STUDY / "a7_gpu/provider.py").as_posix()
HEADER_DEADLINE_SOURCES = {
    PROVIDER_SOURCE: "4b3f0f181401f03f3b54bc8a35861fa91a40a497c93a0162fbb7f65dd2ffa443",
    (STUDY / "a7_gpu/dispatcher.py").as_posix(): "073fe84d02496a4e6ca8187922df58399d87e40b062d0521a925716fe84d576b",
}
HEADER_DEADLINE_FIELD = "response_headers_seconds"
SAME_RESPONSE_RETRY_SOURCES = (
    *QUEUE_SOURCES,
    (PROVIDER_SOURCE, "9b77e82920a303b9429957372d0943a1c729fecca3c80c00d86e0411a1c74bd0"),
    ((STUDY.parent / "pure_llm_v3_20261124/pure_llm_v3/transport_manager.py").as_posix(),
     "25bb9cd757b1d14153bf1f758427d195a4a2299a9a9f840dd7619c783e76c739"),
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

    @property
    def pre_post_connect_failure(self):
        return self == _12203_WAVE12_PRE_POST_CONNECT


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
    boundary_physical_attempts: int = 1
    original_boundary_physical_attempts: int = 2
    response_headers_seconds: int | None = None

    @property
    def header_deadline_profile(self):
        seconds = self.response_headers_seconds
        if seconds is None:
            return {}
        sources = dict(self.required_sources)
        if (self.seed != 2357 or self.wave < 19 or type(seconds) is not int or seconds != 300
                or any(sources.get(name) != sha for name, sha in HEADER_DEADLINE_SOURCES.items())):
            raise PermissionError("the response headers deadline lacks its reviewed lane and source pins")
        return {HEADER_DEADLINE_FIELD: seconds}

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
        return self.execution_budget(self.boundary_physical_attempts)

    def execution_budget(self, boundary_physical_attempts):
        record = retry.execution_budget(36, self.round)
        record["maximum_physical_attempts"] -= 2 - boundary_physical_attempts
        return record

    @property
    def recovery_budget(self):
        return {"maximum_logical_requests": 1, "maximum_physical_attempts": self.boundary_physical_attempts}

    @property
    def replay_start_round(self):
        return self.round + (self.boundary_physical_attempts == 1)


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

SCOPES[(6101, 4)] = replace(SCOPES[(6101, 3)], wave=4, cooldown_unix=1795690616.5505776,
                              boundary_physical_attempts=2, archive=(
    *SCOPES[(6101, 3)].archive,
    Review(retry.bounded.RunIdentity("-6101-relay-retry-03",
        "250e3b87e46a3013f37af35bbe4a911c4214c8efa419a44b52d0c52c56fe496c",
        "a43dd05f6020dd6b27f5e91f703082817f7f74699e99717e3685abb7d6fa805a"),
        "d30c0ed67afa97864f63c0c76c73e62a1854747845583c335d110b42b70735f8", "688a8fd0d3524eabb491b7fcaddddd6a",
        "c0781c7eff611bc630c3bf66c831c4a07410b9b1ced87cc3fe1b5ae69e722e42",
        "1f73d28e528dd56011f30eb13a09911700480675fc6c1eb1da725386382ea1c1",
        "dc14eca9467812a4b3c58a8c9e6bdd0d77c8bdb0fba8cf7306457a5169becefb",
        "ca16f3084e0265ffdf2042996a8d646facc69e33901ebdb6af2014a6be021d12", "timeout",
        {"http_status": 200, "phase": "transport_failure", "error_type": "hard_wall_deadline",
         "event_type": "response.output_text.delta", "sse_event_count": 3016, "last_sse_seconds": 3531.8199360000435,
         "provider_request_id": "bff33246-96b7-4c53-8bf2-8e477b1f16ba", "model": "gpt-6.2-astra",
         "response_id": "resp_07eba1f236437c1c016b08027af25487d0981f8a128889e114"}),
))

SCOPES[(6101, 5)] = replace(SCOPES[(6101, 4)], wave=5, cooldown_field="failed_unix", cooldown_unix=1795692351.0271559,
                              original_start_round=2, original_boundary_physical_attempts=2,
                              boundary_physical_attempts=1, archive=(
    *SCOPES[(6101, 4)].archive,
    Review(retry.bounded.RunIdentity("-6101-relay-retry-04",
        "d91884196aced55e77bb9e9cd2c7d790653060553c5cf573760b092fcc05d303",
        "54215e67efd44263b9b4956d03dc26b28a7914cf6f318e75ff851febf8c42e53"),
        "45a83e9d69c4b3c4e5abbfb48d2007fc032ab9cc9d1cc711a7c276aa4f62d1a5", "1b132db24b2c493dacaa226f3b8a9d03",
        "63673125a094b1307479e6f334b8d4cae3b82d86ac189ee324d6bde74b56c172",
        "103d35e0e06c00d82f603b71f71deabd462a563d5f74b25e349b69bf4016afae",
        "b1e38d716655fe6e3d6716bf91d48c7f127845f12f9f047693e92236309cf61b",
        "060a67cb60da1119d693d2b800c914d6a46acd9b7ecb5a5b8be26670d43d7d25", "sse_frame_error",
        {"http_status": 200, "phase": "transport_failure", "error_type": "ReadError",
         "event_type": "response.output_text.delta", "sse_event_count": 252, "last_sse_seconds": 1201.01432175003,
         "provider_request_id": "bfe68056-9b30-465c-9a09-a1632d2350c4", "model": "gpt-6.2-astra",
         "response_id": "resp_0fdef8f9826fb9c4016b08126348a087d299ea05eb53d3e6d5"}),
))

SCOPES[(6101, 6)] = replace(SCOPES[(6101, 5)], wave=6, cooldown_field="not_before_unix", cooldown_unix=1795692993.4806566,
                              original_start_round=None, original_boundary_physical_attempts=1, boundary_physical_attempts=2,
                              archive=(
    *SCOPES[(6101, 5)].archive,
    Review(retry.bounded.RunIdentity("-6101-relay-retry-05",
        "e391fb682d43acd1e452fc3a09e7be5596274cee26e23ee1d1f4f91e0ae69a0d",
        "2f4aae25cfaf9bfcbdf98063cc2f701138274cf46c756ff2825ab8798e705219"),
        "9a69c8b4d228bf5a14134f0b4a2eede7b23f7de1e9a33f49109c2d47d185d1fa", "afcc54a04b93475ba7edcab4471d784e",
        "cb1b1a191293150babaedfdf77ab9e7bc6c5fb141cc87ada6828679d57659eff",
        "d9c055dbb26c36730e2072b738665821937a732d02a78292a33fe2c7b97acf37",
        "6ccadea081c37e6fc2695d61c517222d9c42b90874172e3b2005ee56eff211b4",
        "83f034ab7b6150e487311e0b5a213b5d16a55e5e11d42b4dd8312744f5e4aa73", "response.failed",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response.failed", "event_type": "response.failed",
         "provider_error_code": "server_error", "sse_event_count": 41, "last_sse_seconds": 318.5360944159329,
         "provider_request_id": "42fe0655-daa5-4c5c-863c-7ff3c5a4f33e", "model": "gpt-6.2-astra",
         "response_id": "resp_0361f6be4607ecc5016b0818828c8c87d29fd7a92b30d25ab7"}),
))

SCOPES[(6101, 7)] = Scope(
    6101, 3, 7, "ee08e66f7541e5081b341ae3b104b5839294dadbd37afe8a09c24200dae49ad6", (
        Review(retry.bounded.RunIdentity("-6101-cached-r2-01",
            "802ae4d2191ecfb834bfe7bed995f54d9ffe40c36a2ec5cb48236fdc4c97ebad",
            "40098571aeb8b13c3eaa0f0ffb1870ba87f318bbfcdc11218480b5d97dd96c59"),
            "d800a262bd72eed1dba8cfa0d8d5d2f82b2841da7f28320be4ab24ab1707b61d", "290284ba893c42f79a67a9b5bf88159b",
            "10768096f58639ab0558274872135f1b412983a80531c74ca3ad7d88529cf1f1",
            "0ab6189a6f4dd5e1c2dc3986701aaeb20d681aa51a9b867a4a62bd387742acda",
            "bccac7285af180969617c3eca42b03480695f0d25f2e2aafcc7db88047f671bd",
            "bdb1e5376d1223808aa8df262ee6308df6f779a0f4fbae8bfdfb326812cf5e78", "timeout",
            {"http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
             "event_type": "response.output_text.delta", "sse_event_count": 1092, "last_sse_seconds": 1930.1032520001754,
             "provider_request_id": "135efa95-4f78-4992-930c-2d2aefe18baf", "model": "gpt-6.2-astra",
             "response_id": "resp_058c376bd0a872b0016b082ba0164887d2a9347a98d0dca4c4"}),
    ), "failed_unix", 1795700386.0270398, original_start_round=2, fitted_prefix_files=(
        ("round_02/turn_00/attempt_00.terminal.json", "cc3ef06bb9e2798bacdfadd15bfd3e5c4a8f95d196463ce5bf08be55c52c82fa"),
        ("round_02/batch_freeze.json", "ed3d2e410799a947e28302669e15ef2c11f269af6a20d990fa1a3ffc056fa31e"),
        ("round_02/selection.json", "08a9e6d66e59c3b2497984c4ffab2090d91e72d5cffd6e91beea02298c3490f3"),
        ("round_02/memory.json", "8d6cd100558f57f85be74755824ef9073b805bfe77a0cfe30ee985fcb5938b92"),
        ("round_03/source.json", "cea613518df5533dde899b691f5f3db7a1bb2c761dcf18ab30c8f116675b10f9"),
        ("round_03/transition.json", "8e5f9d4cc67593e008734a4ba72194970560a5d7d495ee4c507299391f2d9996"),
        ("round_03/fit_000/fit_audit.json", "30b84b35840f7e79999c6a6ff0895ba8fcac504e3c6f059e52b4ce033b0051e6"),
        ("round_03/fit_000/validation_metrics.json", "5cc16664c462cbd711d3c61990c19cdf117458f0c55c49bafa35ad0483580586"),
        ("round_03/authoritative_packet.json", "1180c495a1a3b44e21a7bcd65b018a39df312d60fe58cdc2653ae7b7e2666f0b"),
    ), required_sources=tuple(
        (source, "aacbc199e896091d4395fc036cb60bc604d91ec326a79fcff26e157d9a73705f" if source == PROVIDER_SOURCE else sha)
        for source, sha in SAME_RESPONSE_RETRY_SOURCES
    ), cached_origin=CachedOrigin(
        retry.bounded.RunIdentity("-6101-relay-retry-06",
            "3380abf0b3fbde69a4d4c7a9d8e64dc815cc64c258736088baa3041fff209023",
            "ae353451f7f46dd0b36a4515e8e469defe36348f50e9be8752999d7999a228c1"),
        "257e1b849cd8466d7eb599784c30ce5fc72f6afc974930b0c88ad3e675a61c57",
        "55c1aaa458c9e105ce9d9b94ff7434ccfb347b69ea1b96a335175ecec713b244",
        "ad483af2538715ea4f70f6e7bbbdbc01b590058a0b4923ec0595d0719cb3785e",
    ),
)


SCOPES[(6101, 8)] = replace(
    SCOPES[(6101, 7)], wave=8, cooldown_field="not_before_unix", cooldown_unix=1795704665.0379763,
    original_start_round=None, original_boundary_physical_attempts=1, fitted_prefix_files=(),
    boundary_physical_attempts=2, archive=(
        *SCOPES[(6101, 7)].archive,
        Review(retry.bounded.RunIdentity("-6101-relay-retry-07",
            "1da70806e319c8cbacff6e16894d1667fd0f7ad859075410c9ef7ccb49f33422",
            "3e4e510b3c06e0fe26fe39e33155ff85d346c48ab5047e0518cca2bbf91ddc15"),
            "2b2b9d5ba368ff9e49e4dbc610bed027fbbad49039ca732604a36314758fd177", "e674f38eefbb4e9fa50824f61f932400",
            "fa409a947fd079c30940a83194672613070eb2aa500ba1071df77759566e743a",
            "f3b0f8be705e9914600ace137fe9bce9fa036189cb2c65eb6a6866c7b1b74a98",
            "a774e46c0f0989967d750a3756a85e639822a11ba29acb67e2ad4448e1a0a383",
            "070f12991d2f5b8980c5bb944ac54971bdc4cfae4dd1191b8147a4f1188846ca", "timeout",
            {"http_status": 200, "phase": "transport_failure", "error_type": "hard_wall_deadline",
             "event_type": "response.output_text.delta", "sse_event_count": 648, "last_sse_seconds": 3314.1049271670636,
             "provider_request_id": "16af7ace-dc40-4203-8032-039d98bfae0b", "model": "gpt-6.2-astra",
             "response_id": "resp_0af34571a9ede41c016b0839433f6087d2afc55981c5f34a34"}),
    ),
)


_6101_R5_CONTINUATION = retry.bounded.RunIdentity("-6101-r5-continuation-01",
    "e061a3321283a52928bc97de3f3f704abdc91aad435a79ca8f374f36af4c92f0",
    "7ed54929fa4513e5b4b9060705b5d4cd582094c6f225b4e1cdd8e1056b60a8e0")
_6101_R6_CHECKPOINT = "5f2b2b644f40d3e2dbc91ca5da7bab9003eadbeb37623ee9bd0c2d777d1f1a9a"
SCOPES[(6101, 9)] = Scope(
    6101, 6, 9, "a59f16ba3c689a594bcbc566b972da3f22a6d00adb1483dad0290bc75dc4cb9f", (
        Review(_6101_R5_CONTINUATION,
            "e4920fc471d314f94e30ddd5cae8296b2cc00c61b644fd0befad4f796e8923e2", "69cc4c61d3c841528eac07c9648f93ea",
            "91183e20f85e4502336ee692afdb57a1dc44b7084cf85304aab63a35b60ddef9",
            "3cf63864aed8906b58a961d818f7de0fa0026085759ef5abcfd751873e320a1b",
            "714cd62cda2b9507ebfa0a200f0d60304cce0aa9d48f3d5120f448746c20c7dd",
            "8ab403e2d74c70a9d8d2f806f71b4c0a9643c7e7bc68b765be7aa50dcd84b0aa", "connection_interrupted",
            {"phase": "transport_failure", "error_type": "ReadError", "elapsed_seconds": 22.604738916968927,
             "network_phases": {
                 "direct_ipv4_preflight": {"duration_seconds": 1.0998312919400632, "observed_seconds": 1.1062773340381682, "state": "complete"},
                 "direct_ipv4_preflight_1": {"duration_seconds": 1.0998312919400632, "observed_seconds": 1.1040142921265215, "state": "complete"},
                 "direct_ipv4_preflight_1_dns_tcp": {"duration_seconds": 0.3846792501863092, "observed_seconds": 0.3892637090757489, "state": "complete"},
                 "direct_ipv4_preflight_1_tls": {"duration_seconds": 0.3268952078651637, "observed_seconds": 0.7169538750313222, "state": "complete"},
                 "receive_headers": {"duration_seconds": 16.4222404579632, "observed_seconds": 22.599786292063072, "state": "failed"},
                 "send_body": {"duration_seconds": 2.0461973340716213, "observed_seconds": 6.1744883339852095, "state": "complete"},
                 "send_headers": {"duration_seconds": 0.00022587482817471027, "observed_seconds": 4.127967624925077, "state": "complete"},
             }}),
    ), "not_before_unix", 1795710270.8833244, original_start_round=5, fitted_prefix_files=(
        ("boundary.json", "d9315cd18eff5ae0ec26414f398d210929caabfb02c4fc3a3cff5b9346e6cdff"),
        ("round_05/source.json", "ff67b9cd13896991283e9491fd5856b0fe88ae31cff4067d40da46b4503bfb94"),
        ("round_05/boundary_memory.json", "3bfe5d7f06bd1c4162b6ef14e307a186b9d2c3775d600622bb42fd731911fbce"),
        ("round_05/selection.json", "4d40f61edac613254719cfb3a8e2a06d5123d8c90bd6837fab8fea39e0ed2316"),
        ("round_05/batch_freeze.json", "6dbade356cac58ef256082277d41241afe8d1433b9cfc07e2f1dfc95c0bd9c7e"),
        ("round_05/memory.json", "7b6180f12da4b0d87e428f8e7b7a42ddd656a34115c69f6a4b4646fc52062997"),
        ("round_05/turn_00/attempt_00.terminal.json", "bacecbda3c91638c2c07fb52c0dcc490fedfcab27bfab9d2c9b13f62ef0cf273"),
        ("round_06/source.json", "896eb6ebbb89b83c6cfd6d34e4af8d322d5ee593e7dfc0c73a9277f93283219e"),
        ("round_06/transition.json", "97315c1d7c2f92b940d3ec9e30841b4496eec6dcd5f04a84b9ab832b72602e18"),
        ("round_06/fit_000/fit_audit.json", "4d03eece5896e116fc35aee450309264bb87a96925f37bdd71772da538752df6"),
        ("round_06/fit_000/validation_metrics.json", "55e2385893ab29c2f0d5b8d0d49b37cff5aa0c5861230a6f53deee377cec6cd5"),
        ("round_06/authoritative_packet.json", "49d34f1bda4e39ea826fb37be1c094a8072c2be63b40f17b9128e44ad4a5183c"),
    ), required_sources=SCOPES[(6101, 8)].required_sources,
)


_6101_WAVE09 = retry.bounded.RunIdentity("-6101-relay-retry-09",
    "c5a2f7184c17057fe9b9691dd3ba9ad5edef8a7c4d912adcc1dc9056927ba37f",
    "12285ae28447c96809134b8c38b20512ef5ab5040f36416fd15ff67ae8ccde1a")
_6101_WAVE09_R8_REQUEST = "f357136bf6563f4c50d12932d5ee638c90ea9f33cddd1cabf5437694dcc975f2"
_6101_WAVE09_R8_FIRST = Review(_6101_WAVE09, _6101_WAVE09_R8_REQUEST, "ef5051cc66524f29966df442211f160b",
    "4867d3f8bfae897d5d11943b9d440474dedfa4ebc8ef89be17a914b9caab0ea7",
    "5e3d759fe78b7d471b6f0e60bf5748f88a712c1af25b62ebe854853635b0d5e4",
    "48cde2fd1f636e008383f035e88d98f3b4d888ac61446e5992d8d42f71802685",
    "043707f73fbdcbe8e63f42d9b6021233774a66185d227c3d03ce41f68a6e76f0", "response_id_mismatch",
    {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 24,
     "provider_request_id": "0834e05c-b997-4dfe-aa81-607e095749c8", "model": "gpt-6.2-astra",
     "response_id": "resp_0805a7463a074bfc016b086615949887d1bb24fdfac647e27a",
     "conflicting_response_id": "resp_0834e05cb9974dfeaa81607e095749c8", "conflicting_event_type": "response.failed",
     "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"})
SCOPES[(6101, 10)] = Scope(
    6101, 8, 10, "2958938fcb0b6cea2e5ef5fb8b379cc02a4ed8a35b4d5635a2fd7f3eb95a006c", (
        Review(_6101_WAVE09, _6101_WAVE09_R8_REQUEST, "450d602d7de84cbd937750f11456c549",
            "088d082ca7ad73b25523ebf3b2b05c8d4283eae651b078dc3bc0eb0d28f8bc95",
            "8b895d8a1fb1aa8a883b7c8a55b60ac9548bd8ecd2844a297105dd4d16aea7ae",
            "12ef48c3857621ffb6098984e82adbd4345a032c2fb095b1631213af8babb6d1",
            "bf63990032e2cd82564f0da6cf99f7bf8519251e25f5869e579f2effe8de4527", "response_id_mismatch",
            {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 25,
             "provider_request_id": "cf4c6250-dcff-4f63-a2c0-5ed6e3d40527", "model": "gpt-6.2-astra",
             "response_id": "resp_0424cc53454e7b6e016b0868b5a8a487d192173d72e36d844a",
             "conflicting_response_id": "resp_cf4c6250dcff4f63a2c05ed6e3d40527", "conflicting_event_type": "response.failed",
             "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"},
            earlier=(_6101_WAVE09_R8_FIRST,), replay_files={
                "attempt_00.replay_authorization.json": "cb6f2bb92c02f76c645a66e9441799f025309d53400fd81098adb0f6942dd829",
                "attempt_00.replay.terminal.json": "a5a190fa2f36d05fac6960a9834dff11859cfb8aef955684444ff9a698a8286c",
                "attempt_01.replay_link.json": "7a082d4289a51447d8ea47391fbcff458799d5ef43f6076a5be1f2fd9abd6467",
            }),
    ), "not_before_unix", 1795713420.7855213, original_start_round=6, fitted_prefix_files=(
        ("boundary.json", "12df247e9e7c256220eb13cf455304ebd63f6ac92563d6dd7eb29ca352d5052c"),
        ("round_06/source.json", "896eb6ebbb89b83c6cfd6d34e4af8d322d5ee593e7dfc0c73a9277f93283219e"),
        ("round_06/boundary_memory.json", "7b6180f12da4b0d87e428f8e7b7a42ddd656a34115c69f6a4b4646fc52062997"),
        ("round_06/boundary_user_message.json", "a62e1542e0d54e567ee19c270aed97db9d5a6620421c846abf16ba32b6ddf83a"),
        ("round_06/selection.json", "fcfb53630734b4d6c3b9ee60f25f9dd08581ffdb1a38dea1b3d75d8e96b7caa5"),
        ("round_06/batch_freeze.json", "dfd739851d29a43be92ef58dff8d651ccaa8aca937e1584d7d32b002549313be"),
        ("round_06/memory.json", "e5201f76f9d49c89ae278d6d2e82ef58cecdc120d145c6037364a2536595da20"),
        ("round_06/turn_00/attempt_00.terminal.json", "97a5a6e8648b1e9bd58aa7f9251b63ad5d1519e929ae070dd80122f8141f4216"),
        ("round_07/source.json", "74098c938e8ed4e1ff3f09af639b3bf668bbb2ad6514ddc04a40ddec1046578b"),
        ("round_07/transition.json", "56631c6ef5a417125f6def6f8a4492ec3dd4897b15d1c7e381b57ee66fe772a5"),
        ("round_07/fit_000/fit_audit.json", "b6b5a085b8b902276e079d1169c4c5824c8c32557b49eeb5261e927aadeabd9a"),
        ("round_07/fit_000/validation_metrics.json", "52d462ac148ce7dc8f9926f696e0473b7fd8147f2618ebe3ea2c046db0a2d385"),
        ("round_07/selection.json", "6e8d8121c2439198a7fa0c62f1ea588451b6adea3befdbd786562dff459aff72"),
        ("round_07/batch_freeze.json", "356d5a9e0851fe4a77c9df4f26e667121ac6e9639c55ba4469bc7c2d04e7cdad"),
        ("round_07/memory.json", "926c576d6a1694029b1dc34745d7fbde45934a9d3b87078d19e714dfe62a660c"),
        ("round_07/turn_00/attempt_00.terminal.json", "27ca3270bdf039c451d79e0930f225abdd434027fad0700fdcff9a3902b92f1d"),
        ("round_08/source.json", "3230dd9d4a8cf6c0c1222ef322f5240b679d4d7ad17bfde42b16b3134a67270d"),
        ("round_08/transition.json", "b857ed3cb6c60ad991758756502374a320e5e67b19b07ce31a636c69b57b99c8"),
        ("round_08/fit_000/fit_audit.json", "1a56dba2787c88d2343742556c375bebcfa6172d41d338c42f2eab2548576cab"),
        ("round_08/fit_000/validation_metrics.json", "ffb5d6aedb73511ae3146a61a5c7f9725d9957b287651f9b977dca5625ba6364"),
        ("round_08/authoritative_packet.json", "6b20eb5704f7071f78e9ffb057c8e0bc9f38d1ae9c95d31e187a1fa63ea0334a"),
    ), required_sources=SCOPES[(6101, 9)].required_sources,
    boundary_physical_attempts=2, original_boundary_physical_attempts=1,
)


_6101_WAVE10 = retry.bounded.RunIdentity("-6101-relay-retry-10",
    "fb0342ccd7700c266ee0a460096aaec248884996955edc20a9af8e30fe2e2137",
    "c1ab36d0ff790098b4782cd9438e893aa060bd8e980cb4b3fb6242086c801ea7")
_6101_WAVE10_REQUEST = "5b8ac84b3646090579664633a02389af05be6e42684cea78ddaf25fcf540c0d4"
_6101_WAVE10_FIRST = Review(_6101_WAVE10, _6101_WAVE10_REQUEST, "db13d3243ddc4591acea7c94d2fe4b75",
    "c8fe143b6d00331d23ffcdf2e8e66194887e4dddfc00dcfcf0171bf75c56b3d6",
    "8e2719bc25c2f5ebdbb1b165301bdc48133a104ae62062d463a08c0e37ca9364",
    "6e31bafc4dc07f499a8c8f63ff4b96d27a5418357dc128d169d1eecaae7e42eb",
    "8448dd3f2ee15b5b70b32b2635cdb86c41923fdb6a9b73fa613156b7669a0909", "response_id_mismatch",
    {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 41,
     "provider_request_id": "4be5bc98-ffff-4fb2-bf68-fdbbde51455a", "model": "gpt-6.2-astra",
     "response_id": "resp_0e117d81a9715d81016b086bdd85bc87d28170abd29550100c",
     "conflicting_response_id": "resp_4be5bc98ffff4fb2bf68fdbbde51455a", "conflicting_event_type": "response.failed",
     "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"})
SCOPES[(6101, 11)] = replace(SCOPES[(6101, 10)], wave=11, cooldown_unix=1795714824.6630635,
                              original_start_round=8, original_boundary_physical_attempts=2, fitted_prefix_files=(), archive=(
    *SCOPES[(6101, 10)].archive,
    Review(_6101_WAVE10, _6101_WAVE10_REQUEST, "55a4b2e1c9294a7184ca133aecca3577",
        "0a9afe0b83dfdccf855f6e57c5341732e007930a04def44c1c2c103da3121b48",
        "2a446ca7c25cf49d7a40ab02c95e84fa78ee14067fefeee2d43496cda9c1100c",
        "9bc303dad617a073f4d5876441d8e556e2884c4649d9d09ce630bdfb0dc755f6",
        "cbd90d57c55e7561775e41c8fdfcd6cd6601b5542d9be84352bcc0854af69966", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 25,
         "provider_request_id": "52fa244b-8600-49f8-8101-f301db1daddc", "model": "gpt-6.2-astra",
         "response_id": "resp_02114b47e9dc9ab9016b086e2931e487d199dc03f20afa22e1",
         "conflicting_response_id": "resp_52fa244b860049f88101f301db1daddc", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"},
        earlier=(_6101_WAVE10_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "d4aa7af3b434358d8ba854c8425f61b930e4ae9fecf5d2c5447a7306f5e76b67",
            "attempt_00.replay.terminal.json": "07c005d90d5a536a1fc547d8f822989750272ad93678863933e96ccef102fc9b",
            "attempt_01.replay_link.json": "aa1a28b08103f5353c6dfc0b0a256bf4079b917b693ca80ae74a9f40032f4436",
        }),
))


_6101_WAVE11 = retry.bounded.RunIdentity("-6101-relay-retry-11",
    "2eeb40775aad7cb3364c88bb61fda34302649421818f96d993496a4b5ef7bb37",
    "1208097b53d07ee3ef6bc753abf7b839a717e3e2eb11f23e97ef2ccfc028f2b1")
SCOPES[(6101, 12)] = replace(SCOPES[(6101, 11)], wave=12, cooldown_unix=1795716992.261229,
                              boundary_physical_attempts=1, archive=(
    *SCOPES[(6101, 11)].archive,
    Review(_6101_WAVE11, "b20fb55f997040325e772db5ec201fb544bf865c80cf4caf348899f1537e08b0", "5b0bd7c2d18c4d19969336f0e21bc48b",
        "81870dcb524a73897baed11656fa066e1d18784c44db962f6ad083f278353db2",
        "1c60263f7204b8d55d4ce4361ff1170d029abc84f6996c5983e2ac168d63661d",
        "5ad5c1f04b9ed476f609a395a5f725c1846d2c522586ff83df5ca927e1c9dee7",
        "00de6efcaf267be3986f35d013b6f9b3f22f6ebdb056fb97d356e3bc75c2b2a7", "timeout",
        {"http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
         "event_type": "response.output_item.added", "sse_event_count": 11, "last_sse_seconds": 98.36479816609062,
         "last_sse_activity_seconds": 98.36479816609062, "elapsed_seconds": 998.379541415954,
         "provider_request_id": "92245a07-bfb7-49d3-b38a-6de5dac41c94", "model": "gpt-6.2-astra",
         "response_id": "resp_0ed0dda73e39e4c1016b0871d5f55087d1a9cbbb40341c4100"}),
))


SCOPES[(6101, 13)] = replace(SCOPES[(6101, 12)], wave=13, cooldown_field="not_before_unix",
                              cooldown_unix=1795717954.8001642,
                              boundary_physical_attempts=2, original_boundary_physical_attempts=1, archive=(
    *SCOPES[(6101, 12)].archive,
    Review(retry.bounded.RunIdentity("-6101-relay-retry-12",
        "ec792611cee180e42dec88cb93e2285d5adbf44ef35070889b90553ecb42856f",
        "3383ca07d78e1d3d53cfa6e68a0fc5fb74a2309d61470ff931b329272bce0b7a"),
        "76165309548af59ee119b62329afae6bf21af11a7605a89651565026ae585068", "4931381e0e68465ba95d5bde6d7250b5",
        "6a41520e25cea9c87772f7ffd72044d92f835d1339e2a6c0887538f4b4831a32",
        "60217d22b4a367ab3a0115c136a29a90b6d7f75d0b767bc098a11f791b2d1122",
        "c533a79262db1228331fdd145f3c474f66e504d48cafe49ae57f3c1e5e70d929",
        "b77f4fe150f0b5e120c97ad47cb231ab992bb47fe6cc884d737ebaed889c19f8", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 41,
         "provider_request_id": "dad38e94-87e4-4511-b337-8fd9053b5ab0", "model": "gpt-6.2-astra",
         "response_id": "resp_0c1d1ca729a0b81e016b087a78ab2c87d290fba89d87269c80",
         "conflicting_response_id": "resp_dad38e9487e44511b3378fd9053b5ab0", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"}),
))


_6101_WAVE13 = retry.bounded.RunIdentity("-6101-relay-retry-13",
    "eaf4be1327059e4428c44f2b18dbead1bc6939348ff4a42a1a38db63d5c1d58d",
    "bb4a095411844fbb77264300f8768648919bc837cbfa22e806a2f1eb7639702f")
_6101_WAVE13_REQUEST = "2e8bb0c438ec58d8835e619404593f50eab00f6ddf6bb5ac968ae69330a6d56f"
_6101_WAVE13_FIRST = Review(_6101_WAVE13, _6101_WAVE13_REQUEST, "5838b6000a814025b2d1d852f34fdbec",
    "0f2e69d2f255c9c5049b62eedf28dfe04b37989fe4f13bb808b6d2e03ce1a4ae",
    "f182276436b741654d035b46600990bb74070eb7581530bf322e5d12b8ccadbc",
    "643680653c93985cc863f6449c531baaf78c8a4dbcfd336b1af681c4e34f4310",
    "645b43961ecd91f02f205a9409e6ed3dc7c749239b6290be6343dce58062cfbb", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'sse_event_count': 41,
     'model': 'gpt-6.2-astra',
     'provider_request_id': '99bc486f-f525-4a96-8bbc-d77424cde734',
     'response_id': 'resp_09ad142159470aac016b087cbc710087d28f16d278d37083ff',
     'conflicting_response_id': 'resp_99bc486ff5254a968bbcd77424cde734',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 200.6954515420366,
     'elapsed_seconds': 203.1509125418961,
     'sse_milestones': {'response.created': {'elapsed_seconds': 25.452258958946913, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 25.45321649988182,
                                                 'interval_seconds': 0.0009575409349054098},
                        'response.output_item.added': {'elapsed_seconds': 25.4539109589532,
                                                       'interval_seconds': 0.0006944590713828802},
                        'response.output_item.done': {'elapsed_seconds': 31.959784999955446,
                                                      'interval_seconds': 6.505874041002244}}})

SCOPES[(6101, 14)] = replace(SCOPES[(6101, 13)], wave=14, cooldown_field="not_before_unix",
                              cooldown_unix=1795718929.4587774,
                              original_boundary_physical_attempts=2, archive=(
    *SCOPES[(6101, 13)].archive,
    Review(_6101_WAVE13, _6101_WAVE13_REQUEST, "3c292dd317494e3c819dfd4d2d91e11f",
        "89f7c6019a51fb94bccdc21ddfbe6836718a6c401b5d851196039c8d7b638b55",
        "0f65080f38dc543b90be42086428d6f5156ece7bc288c979fd537257151ca73f",
        "1f9916f971363cc387bbbe876cfc808b8392dfc1c43c3b589c12c5fbe81b7ab5",
        "0f88e98983372369406a082573daf3410a7c25c774c40a370337170d5fb58f60", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'sse_event_count': 41,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '0234a87c-69be-42c9-9892-09978844c446',
         'response_id': 'resp_0ea0a1e91e9a3f39016b087e445cdc87d285e631ffd5c54de8',
         'conflicting_response_id': 'resp_0234a87c69be42c9989209978844c446',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 197.1219058749266,
         'elapsed_seconds': 199.487576500047,
         'sse_milestones': {'response.created': {'elapsed_seconds': 19.544332250021398, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 19.54631866607815,
                                                     'interval_seconds': 0.001986416056752205},
                            'response.output_item.added': {'elapsed_seconds': 19.548579999944195,
                                                           'interval_seconds': 0.002261333866044879},
                            'response.output_item.done': {'elapsed_seconds': 28.862903374945745,
                                                          'interval_seconds': 9.31432337500155}}},
        earlier=(_6101_WAVE13_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "b3c07aa957c95612cefcec3a0402b516127d8d36075b2f487b5288fa0e95163d",
            "attempt_00.replay.terminal.json": "754e073857d2b1b6ef25250256ec83dc2a20eadd459cfd26d28e213e685f1a85",
            "attempt_01.replay_link.json": "1047670bc75a59f934af03d6654074437de21d2c8c4d8faaa9c3288a92640af0",
        }),
))


_6101_WAVE14 = retry.bounded.RunIdentity("-6101-relay-retry-14",
    "eb4f325faeb5ba5dd07d2af86521c2c87834b2f7284ed10eb398cba0ffbfaad7",
    "7ea360bc7bdbe2554e698f4f572395bd7c90acfdd038b97ef34ba9bc04fab3fe")
_6101_WAVE14_REQUEST = "152b0e1afcee48181046a4d56dc9549d0311eb91a847967f808e439375b1ba59"
_6101_WAVE14_FIRST = Review(_6101_WAVE14, _6101_WAVE14_REQUEST, "28a3cea9b05d46acb43d47c85fb1f019",
    "c6fe885fcf54eb804e419a28c7d77500036e0f0be1391377a6645c7d695f6233",
    "130e22f72fa4f09382241283192a3dda46a37832efae218b92f6f76c590a267f",
    "d06cba12bacec23e9cb9119101094016068b0c917f3214163fe5f3b43ce4222d",
    "c843fa4112bcb160c10e37582ae858d0286b0465763013f336d05eb523545516", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'sse_event_count': 41,
     'model': 'gpt-6.2-astra',
     'provider_request_id': '0daae4c8-6ed9-4d06-b84a-ec8add898bb8',
     'response_id': 'resp_065299b1de693b2b016b088107a04087d2a8511d1c4dfb6426',
     'conflicting_response_id': 'resp_0daae4c86ed94d06b84aec8add898bb8',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 198.2515598749742,
     'elapsed_seconds': 201.38565191696398,
     'sse_milestones': {'response.created': {'elapsed_seconds': 21.23378008394502, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 21.23482720903121,
                                                 'interval_seconds': 0.0010471250861883163},
                        'response.output_item.added': {'elapsed_seconds': 21.235555541934446,
                                                       'interval_seconds': 0.0007283329032361507},
                        'response.output_item.done': {'elapsed_seconds': 30.532186292111874,
                                                      'interval_seconds': 9.296630750177428}}})

SCOPES[(6101, 15)] = replace(SCOPES[(6101, 14)], wave=15, cooldown_field="not_before_unix",
                              cooldown_unix=1795721123.602687,
                              original_boundary_physical_attempts=2, archive=(
    *SCOPES[(6101, 14)].archive,
    Review(_6101_WAVE14, _6101_WAVE14_REQUEST, "88e82c797f1647ef9bc5b30796f4ce44",
        "ffa3ace4ffa5ac6914c767a154aed4ba0ec2cd0ad3c3c12fb48360d5cd22adfb",
        "23a76da405428d3ddbe046b01b96f7f2201a4249fce7cf19050f47001210dffb",
        "e575b6d466dd59367264e0199e0604f317fdc3d70434714517deec1048601994",
        "58b78f27278f6d1d9dd56446b70d95aff13fea57a7221a7a20d991bc50ca14f3", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'sse_event_count': 41,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '33e82de3-9228-42ff-a99d-29ab758c79a3',
         'response_id': 'resp_0dcd77ba68925457016b0886d00d6487d2bb1b2ae88af05f16',
         'conflicting_response_id': 'resp_33e82de3922842ffa99d29ab758c79a3',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 200.20737820793875,
         'elapsed_seconds': 202.82111550006084,
         'sse_milestones': {'response.created': {'elapsed_seconds': 22.877719667041674, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 22.880311083048582,
                                                     'interval_seconds': 0.0025914160069078207},
                            'response.output_item.added': {'elapsed_seconds': 22.882771166972816,
                                                           'interval_seconds': 0.0024600839242339134},
                            'response.output_item.done': {'elapsed_seconds': 32.2034473749809,
                                                          'interval_seconds': 9.32067620800808}}},
        earlier=(_6101_WAVE14_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "44751e2c62e1936d9b68cb9843c6f5d43fb63b6a86a130006e41fd7d48100d1a",
            "attempt_00.replay.terminal.json": "29446f425f5a360dac2c5e56274367ce5af0deb1200829e1c22c7a43e693d60b",
            "attempt_01.replay_link.json": "0ceab90e3e9c14e153ccd97cc87cb3a941b93ea62cf95864fdd45b8d641f3ade",
        }),
))


_6101_WAVE15 = retry.bounded.RunIdentity("-6101-relay-retry-15",
    "723888e402d1a35102e85343c6d467008d95b4780d334efb8adb8828d55abe04",
    "1f223585dcc62458f3f0ca03bd4b17fbd19bddc84186a814923286c7daa1e956")
_6101_WAVE15_REQUEST = "02285fe9d216e386e4920b659665ca66b29c2ca43ef0f7eb43ec319538f3fbb2"
_6101_WAVE15_FIRST = Review(_6101_WAVE15, _6101_WAVE15_REQUEST, "39b38386aaf54b998a6c652f004dbbdc",
    "b8f5ad7f5306595851bcf713b14aef5e88650c2a9521aa8044634a473f60fea9",
    "c41d84efad4bcfb88bc55c386aba085bbf33e1c2515f128aa38bcc670df0bb72",
    "aeb856024cab2c5e5a79e17973fdf34ae8186d63705558c238fc32046e8bed45",
    "a8b4fa799a15c4b8e6da9515ef8012eb461a330ca567edd90d3574b614b613fd", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'sse_event_count': 41,
     'model': 'gpt-6.2-astra',
     'provider_request_id': '9b7d8812-a352-4629-979f-68127c88c136',
     'response_id': 'resp_02034872cf78e435016b08890c91e487d2bb7facc6d304b335',
     'conflicting_response_id': 'resp_9b7d8812a3524629979f68127c88c136',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 195.81578575004824,
     'elapsed_seconds': 198.80683895898983,
     'sse_milestones': {'response.created': {'elapsed_seconds': 18.901981834089383, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 18.903225583955646,
                                                 'interval_seconds': 0.0012437498662620783},
                        'response.output_item.added': {'elapsed_seconds': 18.903933375142515,
                                                       'interval_seconds': 0.0007077911868691444},
                        'response.output_item.done': {'elapsed_seconds': 28.061106084147468,
                                                      'interval_seconds': 9.157172709004954}}})

SCOPES[(6101, 16)] = replace(SCOPES[(6101, 15)], wave=16, cooldown_field="not_before_unix",
                              cooldown_unix=1795722108.4976568, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(6101, 15)].archive,
    Review(_6101_WAVE15, _6101_WAVE15_REQUEST, "c9dea397edc7441490ab906504560983",
        "5040f87fcf2b84bee9bb2da33aaefcf84ca1bbd7c6eeeb4c7eea2a37193588d0",
        "1abeb6c5d35975c07021d81f6e02759e9cef7c2694f605b36b0560911d7c0a6b",
        "17ab86b5f20098b8227442fe27eff12973bddaee3a34f13511b8d3f269a520bf",
        "338a5c46c9ca48dd39ee9cb87b02a8ba6e2fb46dbc5209ea034a46137b592238", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'sse_event_count': 25,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '284acfc7-42ce-492e-91be-22250147e1e1',
         'response_id': 'resp_0026a862684924ce016b088a9d52e887d19b47da308bb19652',
         'conflicting_response_id': 'resp_284acfc742ce492e91be22250147e1e1',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 213.05853545898572,
         'elapsed_seconds': 213.06899487483315,
         'sse_milestones': {'response.created': {'elapsed_seconds': 33.16971124988049, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 33.17170966695994,
                                                     'interval_seconds': 0.0019984170794487},
                            'response.output_item.added': {'elapsed_seconds': 33.17382129188627,
                                                           'interval_seconds': 0.002111624926328659},
                            'response.output_item.done': {'elapsed_seconds': 50.751691666897386,
                                                          'interval_seconds': 17.577870375011116}}},
        earlier=(_6101_WAVE15_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "5a0c729d6f585c09bc54a695b2ee424e55103ef82201d0295b8ae2ead113fe73",
            "attempt_00.replay.terminal.json": "cfa77fa69645205519bca8ac56852406b059486defcde7a5922e97857469dbb9",
            "attempt_01.replay_link.json": "46c395a008f59306e1d106630a061c9e44882f898605d99df303e4880d9d6cce",
        }),
))


_6101_WAVE16 = retry.bounded.RunIdentity("-6101-relay-retry-16",
    "2089fb222a06744f47099a0cfceeb0afaa5504af7abf7e4d8a3393ea92c4351c",
    "01b5321fb259735fc5472494861c66d0a893eb4f66717d9817231eea8a033eed")
_6101_WAVE16_REQUEST = "d3d16c7894ab2873d14938dcfed40cd2d9de7685302b9726890a081fa6120a06"
_6101_WAVE16_FIRST = Review(_6101_WAVE16, _6101_WAVE16_REQUEST, "f63231e3fdbe466b9672a8d670113637",
    "139fed5fb4e868aad97bc2e22d48b2e4bf52c5f6ec0d62752d0ff8fe116b97de",
    "be58f008e5e65d9858f4c3616d1a54a44c13a9a6ecdfd5eaf7c0de501a82ce9e",
    "c0468ae4cc19065d4f8f6fc8c0e92e62524bcaec62eb2302a8b5733c2372eded",
    "6df25729f58efc92fcb813eae0526b811a3a4aafe56576df979bb85832e5b3a9", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'sse_event_count': 41,
     'model': 'gpt-6.2-astra',
     'provider_request_id': 'e9b0ecef-6f41-4b70-8097-106b21471470',
     'response_id': 'resp_0290f97fd770ed78016b088dcdd09c87d199dc50192ada328b',
     'conflicting_response_id': 'resp_e9b0ecef6f414b708097106b21471470',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 197.44853199995123,
     'elapsed_seconds': 200.42632666695863,
     'sse_milestones': {'response.created': {'elapsed_seconds': 20.310905708931386, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 20.311886749928817,
                                                 'interval_seconds': 0.0009810409974306822},
                        'response.output_item.added': {'elapsed_seconds': 20.312735375016928,
                                                       'interval_seconds': 0.0008486250881105661},
                        'response.output_item.done': {'elapsed_seconds': 29.621836334001273,
                                                      'interval_seconds': 9.309100958984345}}})

SCOPES[(6101, 17)] = replace(SCOPES[(6101, 16)], wave=17, cooldown_field="not_before_unix",
                              cooldown_unix=1795723319.4104714, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(6101, 16)].archive,
    Review(_6101_WAVE16, _6101_WAVE16_REQUEST, "c17bc04f17594d9e9acf4b4ce7357d51",
        "19d9f5842f70bd6fde83efbff2872ec38582bc7ae69e2d0528effd16e187e03e",
        "bdf590a00e8e445f0325c83f24e16ae3e529c60192e5bd2e7002d7d2196ef483",
        "f37adbe6ee937f470dc7b0ea450be5d8922d3a34688134c6612953681786e1f6",
        "27ca1b503cc7383c747ceef031d0ed2956c5dcf66850f922ea05f0e917361146", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'sse_event_count': 25,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '076fd6a9-c824-405c-b5a0-9f7ebec2e1be',
         'response_id': 'resp_0b1ed3fb81c788a1016b088f581da487d186a38ed538d35102',
         'conflicting_response_id': 'resp_076fd6a9c824405cb5a09f7ebec2e1be',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 211.01859904197045,
         'elapsed_seconds': 211.68580766697414,
         'sse_milestones': {'response.created': {'elapsed_seconds': 31.684189249994233, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 31.68774904194288,
                                                     'interval_seconds': 0.003559791948646307},
                            'response.output_item.added': {'elapsed_seconds': 31.691758417058736,
                                                           'interval_seconds': 0.004009375115856528},
                            'response.output_item.done': {'elapsed_seconds': 47.924547374947,
                                                          'interval_seconds': 16.23278895788826}}},
        earlier=(_6101_WAVE16_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "1059bbc3f4a26dbd8c43bfe6d21e16f384466a783eb3e36357b9ae2a0ac985b8",
            "attempt_00.replay.terminal.json": "9120a4377c264d898c2dfdaa68be28bf393bd068053260da31def3cfd212f425",
            "attempt_01.replay_link.json": "2ef5c7ccf7983860781c80c2cc2846dfc8f84daa1f7c7f31b3ab7871e129e903",
        }),
))


SCOPES[(6101, 18)] = replace(SCOPES[(6101, 17)], wave=18, cooldown_field="not_before_unix",
                              cooldown_unix=1795724893.1359603, boundary_physical_attempts=1,
                              original_boundary_physical_attempts=2, archive=(
    *SCOPES[(6101, 17)].archive,
    Review(retry.bounded.RunIdentity("-6101-relay-retry-17",
        "d9eb108d8544c497d71e69804b1082a90fd6667f15bbbabc35863d0114b2d63b",
        "55164af8c4bd97ef8ccbe708254fde01b0464d18b6cd6a1c856b80f219ff17ab"),
        "5e5ce772bd6ee11f1b2362784562e1d683f48f3dcbca9ad83d049b0ad8fc9afe", "c93f82325eef449d965c6cd07d7370d1",
        "f2d2c713536ee18ee9971e391faa150063d3299d338829f85866b1ed782b0589",
        "87e159b75f44d519990e2ea50ad519f911c88346214a1489b42e63aef30b56c8",
        "e644e1eb54cb464a4e4f449fb078e805f55f38d5a19a554565c45990d9440b00",
        "e35c4092a7cc76dfd64af41353d41fb14f49584ff7412545d80954f3cae7371c", "timeout",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'sse_inactivity_timeout',
         'event_type': 'response.output_item.added',
         'sse_event_count': 23,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '28af9e2a-3e94-44b6-8a70-3266dce9c85b',
         'response_id': 'resp_0311909841327144016b08919e406887d282946c24629f2c82',
         'last_sse_seconds': 195.7894462090917,
         'last_sse_activity_seconds': 195.7894462090917,
         'elapsed_seconds': 1095.799670791952,
         'sse_milestones': {'response.created': {'elapsed_seconds': 29.485547916963696, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 29.486534500028938,
                                                     'interval_seconds': 0.0009865830652415752},
                            'response.output_item.added': {'elapsed_seconds': 29.4872592499014,
                                                           'interval_seconds': 0.0007247498724609613},
                            'response.output_item.done': {'elapsed_seconds': 47.20936862495728,
                                                          'interval_seconds': 17.72210937505588}}}),
))


SCOPES[(6101, 19)] = replace(SCOPES[(6101, 18)], wave=19, cooldown_field="not_before_unix",
                              cooldown_unix=1795725469.7751193, boundary_physical_attempts=2,
                              original_boundary_physical_attempts=1, archive=(
    *SCOPES[(6101, 18)].archive,
    Review(retry.bounded.RunIdentity("-6101-relay-retry-18",
        "3ef1cdc126867d2ded8d926965bd80f70d8c3687a61270ee2e402ff7a058e30e",
        "6217b415f93b219bd903da1f75c703c216ec1e61a68452dd45d9f452683b5ea9"),
        "c912c28c98bf1ab461cb5653076104ccaf929918f0aa91bef73a031b11b41996", "a25420736ed14156b62c45aa63147857",
        "4059f593b6e6917069905a47a867de2b563dd2b7bc1a57cfe45a82755bfd43d5",
        "60b22555f35a6e6628ee7cdb1a04e744cfcca60806f289a7474a87356711f204",
        "3b18be441d41efa746520a1c3734a79d36a3b9b0422fe85fd7ebf13f81a239a6",
        "c30806f0431fc6cf55e2932f7162cb8e7598eb3b173ef9251d1246dffb28e019", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'event_type': 'response.output_item.added',
         'sse_event_count': 41,
         'model': 'gpt-6.2-astra',
         'provider_request_id': 'aa7814f3-9ae2-4821-a0fd-0b5c36f0b8ea',
         'response_id': 'resp_0f163772d4233cbf016b0897d1b23487d184a585e7ad294752',
         'conflicting_response_id': 'resp_aa7814f39ae24821a0fd0b5c36f0b8ea',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 195.2267327089794,
         'last_sse_activity_seconds': 195.2267327089794,
         'elapsed_seconds': 198.17091620899737,
         'sse_milestones': {'response.created': {'elapsed_seconds': 18.134589292109013, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 18.13622883404605,
                                                     'interval_seconds': 0.0016395419370383024},
                            'response.output_item.added': {'elapsed_seconds': 18.13797491695732,
                                                           'interval_seconds': 0.0017460829112678766},
                            'response.output_item.done': {'elapsed_seconds': 27.2943535419181,
                                                          'interval_seconds': 9.15637862496078}},
         'network_phases': {'direct_ipv4_http11': {'duration_seconds': 18.065469166031107,
                                                   'observed_seconds': 18.07537891692482,
                                                   'state': 'complete'},
                            'direct_ipv4_preflight': {'duration_seconds': 0.7129365000873804,
                                                      'observed_seconds': 0.7251982090529054,
                                                      'state': 'complete'},
                            'direct_ipv4_preflight_1': {'duration_seconds': 0.7129365000873804,
                                                        'observed_seconds': 0.7229494170751423,
                                                        'state': 'complete'},
                            'direct_ipv4_preflight_1_dns_tcp': {'duration_seconds': 0.21449062507599592,
                                                                'observed_seconds': 0.22649749997071922,
                                                                'state': 'complete'},
                            'direct_ipv4_preflight_1_tls': {'duration_seconds': 0.2253213340882212,
                                                            'observed_seconds': 0.4546462919097394,
                                                            'state': 'complete'},
                            'receive_headers': {'duration_seconds': 13.56521450006403,
                                                'observed_seconds': 18.073338417103514,
                                                'state': 'complete'},
                            'send_body': {'duration_seconds': 0.8663502091076225,
                                          'observed_seconds': 4.506517709000036,
                                          'state': 'complete'},
                            'send_headers': {'duration_seconds': 0.00022504106163978577,
                                             'observed_seconds': 3.6397787500172853,
                                             'state': 'complete'}},
         'first_phase_seconds': {'http_headers': 18.07646029209718,
                                 'http_phase': 0.22649749997071922,
                                 'request_dispatched': 3.493617249885574,
                                 'sse_event': 18.134589292109013,
                                 'transport_failure': 198.17091620899737}}),
))


_6101_WAVE19 = retry.bounded.RunIdentity("-6101-relay-retry-19",
    "caafe807a20d37213b06c1880cc7065834a971c7588fb21b26f6a98f44c54da0",
    "d07b681c836770cc9f5943ad16d6863cfe6d1b3bb2dc587581e96292be3f7208")
_6101_WAVE19_REQUEST = "c20cdba9520247c56e56bf545fd4d321de60be193086b2040af2e0e3fc8af0b9"
_6101_WAVE19_FIRST = Review(_6101_WAVE19, _6101_WAVE19_REQUEST, "ab403f7d4a1641118e5f70e701ae33f3",
    "4e168b730f4f87b05e020379d6bcee445d69dc321b2bf1bc9678b1ff45df9355",
    "f77b0444ff8ec9f44957305b3fa357673a717f9c725c85bbad3ed292ad747ce5",
    "8240975e329e0bc59daf21b1bda05287e71f4b61dfaad635225da802ee56f6e1",
    "8379b08195b7d3df67be43d64a2b7a1a333d5a2627450cb74abb21a25f5b3984", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'event_type': 'response.output_item.added',
     'sse_event_count': 25,
     'model': 'gpt-6.2-astra',
     'provider_request_id': 'a8bef927-671d-42a2-8254-35cc2c785752',
     'response_id': 'resp_0eb1f0b0efef4968016b089bc505ac87d2aaf1e20e0f627ffa',
     'conflicting_response_id': 'resp_a8bef927671d42a2825435cc2c785752',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 205.44007624988444,
     'last_sse_activity_seconds': 205.44007624988444,
     'elapsed_seconds': 211.36252345796674,
     'sse_milestones': {'response.created': {'elapsed_seconds': 31.243470165878534, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 31.244651000015438,
                                                 'interval_seconds': 0.001180834136903286},
                        'response.output_item.added': {'elapsed_seconds': 31.245323000010103,
                                                       'interval_seconds': 0.0006719999946653843},
                        'response.output_item.done': {'elapsed_seconds': 47.60814254102297,
                                                      'interval_seconds': 16.36281954101287}},
     'network_phases': {'direct_ipv4_http11': {'duration_seconds': 31.175461791921407,
                                               'observed_seconds': 31.186278624925762,
                                               'state': 'complete'},
                        'direct_ipv4_preflight': {'duration_seconds': 0.9089106251485646,
                                                  'observed_seconds': 0.9221106658224016,
                                                  'state': 'complete'},
                        'direct_ipv4_preflight_1': {'duration_seconds': 0.9089106251485646,
                                                    'observed_seconds': 0.9198283329606056,
                                                    'state': 'complete'},
                        'direct_ipv4_preflight_1_dns_tcp': {'duration_seconds': 0.24095262493938208,
                                                            'observed_seconds': 0.25314695783890784,
                                                            'state': 'complete'},
                        'direct_ipv4_preflight_1_tls': {'duration_seconds': 0.24738812493160367,
                                                        'observed_seconds': 0.5032165830489248,
                                                        'state': 'complete'},
                        'receive_headers': {'duration_seconds': 26.901228416943923,
                                            'observed_seconds': 31.181038999930024,
                                            'state': 'complete'},
                        'send_body': {'duration_seconds': 0.9552391669712961,
                                      'observed_seconds': 4.277509207837284,
                                      'state': 'complete'},
                        'send_headers': {'duration_seconds': 0.00021779187954962254,
                                         'observed_seconds': 3.3216014578938484,
                                         'state': 'complete'}},
     'first_phase_seconds': {'http_headers': 31.18765987502411,
                             'http_phase': 0.25314695783890784,
                             'request_dispatched': 3.220779249910265,
                             'sse_event': 31.243470165878534,
                             'transport_failure': 211.36252345796674}})

SCOPES[(6101, 20)] = replace(SCOPES[(6101, 19)], wave=20, cooldown_field="not_before_unix",
                              cooldown_unix=1795727052.2344866, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(6101, 19)].archive,
    Review(_6101_WAVE19, _6101_WAVE19_REQUEST, "57d10f4bffd649b7aa2c50065d1efd78",
        "b89626bf11297e9ff26711cd3cb0a3e48d45d2c0b6ccdc8c889bf8d9c849a6ca",
        "1d5bff315f60b87d126009b55f9cf3f1009d7b89a9ce5f77818664bdc0d21c19",
        "157563bb66d5aa09c884b71f2a2287585a147915426ba5fc541845a3121d0dab",
        "be0ca2e33ffd0feeb44d6206b7f9c311de670115b4488293cb105f07e9aa9977", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'event_type': 'response.output_item.added',
         'sse_event_count': 25,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '69e231d6-ad02-4895-a3ee-7599c8fc4ee2',
         'response_id': 'resp_0e21f1eae3a3dc7a016b089df4968887d19ab7ad892c5f5f75',
         'conflicting_response_id': 'resp_69e231d6ad024895a3ee7599c8fc4ee2',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 198.82416954101063,
         'last_sse_activity_seconds': 198.82416954101063,
         'elapsed_seconds': 208.37481295806356,
         'sse_milestones': {'response.created': {'elapsed_seconds': 28.345941541017964, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 28.347996375057846,
                                                     'interval_seconds': 0.0020548340398818254},
                            'response.output_item.added': {'elapsed_seconds': 28.349738458171487,
                                                           'interval_seconds': 0.0017420831136405468},
                            'response.output_item.done': {'elapsed_seconds': 45.84804979106411,
                                                          'interval_seconds': 17.498311332892627}},
         'network_phases': {'direct_ipv4_http11': {'duration_seconds': 28.338568459032103,
                                                   'observed_seconds': 28.34198641614057,
                                                   'state': 'complete'},
                            'direct_ipv4_preflight': {'duration_seconds': 0.9083195000421256,
                                                      'observed_seconds': 0.9142159160692245,
                                                      'state': 'complete'},
                            'direct_ipv4_preflight_1': {'duration_seconds': 0.9083195000421256,
                                                        'observed_seconds': 0.9119096251670271,
                                                        'state': 'complete'},
                            'direct_ipv4_preflight_1_dns_tcp': {'duration_seconds': 0.23898166697472334,
                                                                'observed_seconds': 0.24294458306394517,
                                                                'state': 'complete'},
                            'direct_ipv4_preflight_1_tls': {'duration_seconds': 0.24755812506191432,
                                                            'observed_seconds': 0.4930660000536591,
                                                            'state': 'complete'},
                            'receive_headers': {'duration_seconds': 24.252684667007998,
                                                'observed_seconds': 28.336896458175033,
                                                'state': 'complete'},
                            'send_body': {'duration_seconds': 0.9519454170949757,
                                          'observed_seconds': 4.081764833070338,
                                          'state': 'complete'},
                            'send_headers': {'duration_seconds': 0.0001302498858422041,
                                             'observed_seconds': 3.1295068750623614,
                                             'state': 'complete'}},
         'first_phase_seconds': {'http_headers': 28.3435335829854,
                                 'http_phase': 0.24294458306394517,
                                 'request_dispatched': 3.127337541198358,
                                 'sse_event': 28.345941541017964,
                                 'transport_failure': 208.37481295806356}},
        earlier=(_6101_WAVE19_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "1e19f8ddd5fd6ff51cb6ae1d0790a841485517e5412f4cd8e29162854342a9aa",
            "attempt_00.replay.terminal.json": "043cd54edde70fdef5a116464bd352695d07616e2dca24b2acdd6fb669ea5209",
            "attempt_01.replay_link.json": "03b57e1738b5fe7e4f02ed95f05b845abf34b3f336636102027ba681a007bfbe",
        }),
))


_6101_WAVE20 = retry.bounded.RunIdentity("-6101-relay-retry-20",
    "f13bc63f8695256d37ae9e742819cd764a415989eb1298b30a4f97b68d77aecb",
    "f132ae5cafe4d1219151c232d1b854cc6dd7fc24534fe1210d723bab28757ea2")
_6101_WAVE20_REQUEST = "65fc4b7238cebd3a256d204cc58e21e0843038a229e24f4b568da2811a4a6bf0"
_6101_WAVE20_FIRST = Review(_6101_WAVE20, _6101_WAVE20_REQUEST, "94ff77fc975a486bbc6123e0eb12a505",
    "3ba4d25064a9b19f41852219238935475d2494c7e71152bc2abe9061053d0439",
    "d403e2ef21db87d95d17cd4104ba21e612c81ec26858127d0fb1867ba4f50173",
    "76fa8a43fc6de6b13deb5c86eacf9f5ea464fd7e10bea1fbf3fb651b5d47592d",
    "1e2b97967c97ba92693e4c573131273c39a2c0cb36972b8e7454eedd30279d81", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'event_type': 'response.output_item.added',
     'sse_event_count': 41,
     'model': 'gpt-6.2-astra',
     'provider_request_id': '035d553e-1024-4ade-b2a5-58e2294d7c22',
     'response_id': 'resp_0555ebbaffa6f800016b08a068afbc87d29fea49f1fd1eda72',
     'conflicting_response_id': 'resp_035d553e10244adeb2a558e2294d7c22',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 196.8691331250593,
     'last_sse_activity_seconds': 196.8691331250593,
     'elapsed_seconds': 199.88039808301255,
     'sse_milestones': {'response.created': {'elapsed_seconds': 19.772214166121557, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 19.773239915957674,
                                                 'interval_seconds': 0.0010257498361170292},
                        'response.output_item.added': {'elapsed_seconds': 19.77408150001429,
                                                       'interval_seconds': 0.0008415840566158295},
                        'response.output_item.done': {'elapsed_seconds': 29.07511308300309,
                                                      'interval_seconds': 9.301031582988799}}})

SCOPES[(6101, 21)] = replace(SCOPES[(6101, 20)], wave=21, cooldown_field="not_before_unix",
                              cooldown_unix=1795728070.404925, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(6101, 20)].archive,
    Review(_6101_WAVE20, _6101_WAVE20_REQUEST, "7d19e415a09c4786858a353cab95ee30",
        "e152b8f1833386671086a210b3496bce13eb72829d1fefcfb9fc6e3ce924424c",
        "379e4e50356101777c624c2f7c9d378dffafcd5eb655ed855175b89f3d102ac1",
        "479980cf2f24c2f29b490a0874ace6e807a41e4c3fbf026e4421c77cdf50c7d8",
        "fdf40eb4625389acd65f895aff58bcb8c5ec5ef1ef720d7494c32538f58423c1", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'event_type': 'response.output_item.added',
         'sse_event_count': 41,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '83146006-5842-4829-9801-14e67f4afa32',
         'response_id': 'resp_0d9c83b16628e94b016b08a1fb64c487d2964040c60e9e1060',
         'conflicting_response_id': 'resp_8314600658424829980114e67f4afa32',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 195.67230829200707,
         'last_sse_activity_seconds': 195.67230829200707,
         'elapsed_seconds': 198.37909508310258,
         'sse_milestones': {'response.created': {'elapsed_seconds': 18.436576958047226, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 18.43958095810376,
                                                     'interval_seconds': 0.0030040000565350056},
                            'response.output_item.added': {'elapsed_seconds': 18.441653833026066,
                                                           'interval_seconds': 0.0020728749223053455},
                            'response.output_item.done': {'elapsed_seconds': 27.749191500013694,
                                                          'interval_seconds': 9.307537666987628}}},
        earlier=(_6101_WAVE20_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "700129167dc6339227b90ffce83b2db887600f6d5ba9e4f1ca47586d66468cec",
            "attempt_00.replay.terminal.json": "91a84f0a64be6398a32ca1256dbc06a876d78833ce70443456c99a590a65eda7",
            "attempt_01.replay_link.json": "77be623c93c7e72ccbedcf6c9b540e5097006fd7d737d47ece4859a91eab0387",
        }),
))


_6101_WAVE21 = retry.bounded.RunIdentity("-6101-relay-retry-21",
    "6d4c68b8234db17c79c0b2626e90fe6c29178c42b1aa45da0d88e5ba5a8d7fbf",
    "9f31bb918783ef8f1d1ece1be1c2bd887e73b8322a4176d46fd4fd00513725d4")
_6101_WAVE21_REQUEST = "e60ce7b55b14a635a97022530ebc5902464ba634192022644732fec8292553aa"
_6101_WAVE21_FIRST = Review(_6101_WAVE21, _6101_WAVE21_REQUEST, "ed0e9772e418412fbb46cf4c8adad047",
    "1461923e08d5c987a455b9c53efa7202cdbd922772e4a2d92a3c3388a4df0d23",
    "34e09ebe39b0660ab1bb9810282b4f18325aa533716ee7b8a75c49c0d7b5c7fc",
    "dc2fa0c5acf9e6b4b67d63fec16d54598cbfa88eb8876c8248df336e0623f4b9",
    "48bc05cf3a652378dd49ce9bea47a621885ba041201f38c811caa53911acb8bb", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'event_type': 'response.output_item.added',
     'sse_event_count': 41,
     'model': 'gpt-6.2-astra',
     'provider_request_id': 'f0f8bf9d-305d-4f5c-a5af-faa347dfcbe6',
     'response_id': 'resp_00dd34a24f8fb445016b08a48c7a1c87d2af5db8bbfb742dce',
     'conflicting_response_id': 'resp_f0f8bf9d305d4f5ca5affaa347dfcbe6',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 196.35937079205178,
     'last_sse_activity_seconds': 196.35937079205178,
     'elapsed_seconds': 199.36438170890324,
     'sse_milestones': {'response.created': {'elapsed_seconds': 19.346235875040293, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 19.34709187503904,
                                                 'interval_seconds': 0.0008559999987483025},
                        'response.output_item.added': {'elapsed_seconds': 19.347907416988164,
                                                       'interval_seconds': 0.0008155419491231441},
                        'response.output_item.done': {'elapsed_seconds': 28.60579074989073,
                                                      'interval_seconds': 9.257883332902566}}})

SCOPES[(6101, 22)] = replace(SCOPES[(6101, 21)], wave=22, cooldown_field="not_before_unix",
                              cooldown_unix=1795729129.1526127, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(6101, 21)].archive,
    Review(_6101_WAVE21, _6101_WAVE21_REQUEST, "d61ca71a6ba34c72815eda95a4902d22",
        "ac1ac4fcfe9ee11f8e47e29c514c99e9ee2bc1c10ef98f20862d1432253e78a9",
        "61f4095b9cb65ac097354d46eb361c41fbc605b96b8d9b353fefc22e9db7a83b",
        "2eaec0f27a87734206b9ecf09953df388a03e8fecf9de3938d428aa9353f56a1",
        "c0ca9b967dd2c65ff23c7617b45d9762555cb9307f39b4d6fdeb19e8dd9007e6", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'event_type': 'response.output_item.added',
         'sse_event_count': 41,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '3387dd77-b91a-4820-b19f-6f6e0de255d7',
         'response_id': 'resp_0f78f455098ebd82016b08a6199b5087d2b3cb3c0f8310bca3',
         'conflicting_response_id': 'resp_3387dd77b91a4820b19f6f6e0de255d7',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 196.45068449992687,
         'last_sse_activity_seconds': 196.45068449992687,
         'elapsed_seconds': 199.22330374992453,
         'sse_milestones': {'response.created': {'elapsed_seconds': 19.283509582979605, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 19.285732958000153,
                                                     'interval_seconds': 0.0022233750205487013},
                            'response.output_item.added': {'elapsed_seconds': 19.288397416938096,
                                                           'interval_seconds': 0.0026644589379429817},
                            'response.output_item.done': {'elapsed_seconds': 28.753125875024125,
                                                          'interval_seconds': 9.464728458086029}}},
        earlier=(_6101_WAVE21_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "c0cbc7a2c1a9623c52b150334a2e7fbaf7b2f3c3866778a65342bce52baf6f0d",
            "attempt_00.replay.terminal.json": "37393cf53dff08c8a023505b64fc1721f30457dd9da7b74f17529097abd23ab9",
            "attempt_01.replay_link.json": "03b4ae57ba5cbb66135628db23acd709757d11af8ec6f4d2171a18da98af77d0",
        }),
))


_6101_WAVE22 = retry.bounded.RunIdentity("-6101-relay-retry-22",
    "c4d6f97a21f6acde7f1279cba678fd0c9e0bce7e122cfd5fc859c0cda3dff3de",
    "e3b51bc0b106cb3f30c436472da5744454b6dc1688f5d172d4fcacc2eb39b624")
_6101_WAVE22_REQUEST = "6c8eba5102c55312a8c24941179a119eb843a392d46af3fc5f703a3127e9d152"
_6101_WAVE22_FIRST = Review(_6101_WAVE22, _6101_WAVE22_REQUEST, "d2275b39cbea4e1ab69ed8c951bf489d",
    "9895a2be9a9e4df1f63b96abfb84cc4f3d5ab57f74614697d22ddf14cca7f533",
    "47fe579e9fa4ccc5e5f51785d641aae08cb69769f6478e4e6d246675b7149d6c",
    "6d294a04c3b4599a7e47e966ff0dfd719963167fb4d9030248578cdbbef5e6a8",
    "966fc06eca94e78be39c39f6fb4b7671cb42eb09720e6e2b6bbca1afb255fb85", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'event_type': 'response.output_item.added',
     'sse_event_count': 41,
     'model': 'gpt-6.2-astra',
     'provider_request_id': 'acdc29ef-23f1-4e73-9df3-186adf732dbc',
     'response_id': 'resp_0f23cf58fd69c464016b08a83cba6c87d28989337684aa2921',
     'conflicting_response_id': 'resp_acdc29ef23f14e739df3186adf732dbc',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 195.71191366692074,
     'last_sse_activity_seconds': 195.71191366692074,
     'elapsed_seconds': 198.56211329204962,
     'sse_milestones': {'response.created': {'elapsed_seconds': 18.49276416702196, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 18.49377804202959,
                                                 'interval_seconds': 0.0010138750076293945},
                        'response.output_item.added': {'elapsed_seconds': 18.494646958075464,
                                                       'interval_seconds': 0.0008689160458743572},
                        'response.output_item.done': {'elapsed_seconds': 28.081966874888167,
                                                      'interval_seconds': 9.587319916812703}}})

SCOPES[(6101, 23)] = replace(SCOPES[(6101, 22)], wave=23, cooldown_field="not_before_unix",
                              cooldown_unix=1795730080.231831, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(6101, 22)].archive,
    Review(_6101_WAVE22, _6101_WAVE22_REQUEST, "d6530498d3dc436e8b05b2dbd49f74cd",
        "405a8fbdc8ef2f9e8bea2f0b11699f2d6730a2b3e865f4d080d59f438e72ec0b",
        "034af23b0eb61133d21968855d206edb36202c0dd795da57362b4e9878e10adf",
        "b8a3797b80079c6d75de4fd998461a7ace2c0ccfbb3f034a10379d6c6a2b75a8",
        "08ec4a8a3b1e3875128876f1c0b28746572aa8fe386f0f3b89c755517092e1d6", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'event_type': 'response.output_item.added',
         'sse_event_count': 25,
         'model': 'gpt-6.2-astra',
         'provider_request_id': 'd99d61c4-faee-4ddf-ab28-0ba80446db4d',
         'response_id': 'resp_01f3239edf5c0689016b08a9c4647087d09954ecd17ee242ff',
         'conflicting_response_id': 'resp_d99d61c4faee4ddfab280ba80446db4d',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 206.12672008317895,
         'last_sse_activity_seconds': 206.12672008317895,
         'elapsed_seconds': 211.91308645810932,
         'sse_milestones': {'response.created': {'elapsed_seconds': 31.908563708188012, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 31.912377125117928,
                                                     'interval_seconds': 0.0038134169299155474},
                            'response.output_item.added': {'elapsed_seconds': 31.915689000161365,
                                                           'interval_seconds': 0.003311875043436885},
                            'response.output_item.done': {'elapsed_seconds': 48.24363920814358,
                                                          'interval_seconds': 16.327950207982212}}},
        earlier=(_6101_WAVE22_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "15171368767598339210258575e06ef7e2480c709997b202408c1389579b7e9f",
            "attempt_00.replay.terminal.json": "ef09e775423c9517b6543d7adfe61cd644248cea10c9f8d0f5cc279c019b0a17",
            "attempt_01.replay_link.json": "5aa37eefa70e023f6ea16f2de51de80f582b98ddad73236f3a2d839f3c5f0d81",
        }),
))


_6101_WAVE23 = retry.bounded.RunIdentity("-6101-relay-retry-23",
    "43cf05aae93480ca623c10032d3a5820a33cf0f617e10a3bb7115e4570c4b292",
    "062791ed190d3e0c41e9e71d235eb3e225206458597b27ec3c4228f6278121ec")
_6101_WAVE23_REQUEST = "66c41ac9e6e77ba34bb32059c59dd2382a71b5aa955826f5c4b6956ae4f02e7c"
_6101_WAVE23_FIRST = Review(_6101_WAVE23, _6101_WAVE23_REQUEST, "f02ae69a17f84c60948f9686d5e9c494",
    "56f9fd279b72dff4e755c3ec7e8d0e0d91ade05923db5f736bd1230dab188f2e",
    "36e1a958d5a40f45b107dba36c5894721321a0f07653024650172120f71d8ad0",
    "5e37437642c86d92ae986740a9d9b960526fdfb66357c308854bb6607f11004e",
    "203efd6962ca1c5366cd869ab9ba6013301614793804a67eec02ce5e800080b7", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'event_type': 'response.output_item.added',
     'sse_event_count': 41,
     'model': 'gpt-6.2-astra',
     'provider_request_id': 'de0ba174-1621-4f93-bc29-c181a24c8f0d',
     'response_id': 'resp_0ddbfb9bed1f3a89016b08abdf0ad487d2861e43774e3ede86',
     'conflicting_response_id': 'resp_de0ba17416214f93bc29c181a24c8f0d',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 194.72265804186463,
     'last_sse_activity_seconds': 194.72265804186463,
     'elapsed_seconds': 197.64505687495694,
     'sse_milestones': {'response.created': {'elapsed_seconds': 17.5323133750353, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 17.533455916913226,
                                                 'interval_seconds': 0.001142541877925396},
                        'response.output_item.added': {'elapsed_seconds': 17.534116874914616,
                                                       'interval_seconds': 0.0006609580013900995},
                        'response.output_item.done': {'elapsed_seconds': 26.921417959034443,
                                                      'interval_seconds': 9.387301084119827}}})

SCOPES[(6101, 24)] = replace(SCOPES[(6101, 23)], wave=24, cooldown_field="not_before_unix",
                              cooldown_unix=1795731002.7983212, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(6101, 23)].archive,
    Review(_6101_WAVE23, _6101_WAVE23_REQUEST, "390ae9ee72354d5dbeaa69559af7765c",
        "892572bc928a0202e5b8b0d535da26fefe83114a96e46193533f69b9c969964f",
        "a300bec6b36c4de1a0790caa78e03e2745a13a41ef870596f7c89380236b3370",
        "9090345385a0b06099ecb705e2b030f95a7794879cbad377a197a94560b1a0ff",
        "104e0ca02ca392e01004728ff656b032b4bbe592bbe9e79161052df61fa87377", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'event_type': 'response.output_item.added',
         'sse_event_count': 41,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '32bff7da-8bec-48ed-95bc-1ddcfc8e6734',
         'response_id': 'resp_00b39143f7411dad016b08ad7069a487d284446055f038c8ff',
         'conflicting_response_id': 'resp_32bff7da8bec48ed95bc1ddcfc8e6734',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 195.0334075000137,
         'last_sse_activity_seconds': 195.0334075000137,
         'elapsed_seconds': 197.85632333299145,
         'sse_milestones': {'response.created': {'elapsed_seconds': 17.906683332985267, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 17.908817833056673,
                                                     'interval_seconds': 0.0021345000714063644},
                            'response.output_item.added': {'elapsed_seconds': 17.910795415984467,
                                                           'interval_seconds': 0.0019775829277932644},
                            'response.output_item.done': {'elapsed_seconds': 27.154793333029374,
                                                          'interval_seconds': 9.243997917044908}}},
        earlier=(_6101_WAVE23_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "1a451a95cecc32c215d00058b7c8a5e5a6f25d20dddac165d5fe815be6cf42da",
            "attempt_00.replay.terminal.json": "f72bb313c783b9af255d6ec92b383f5df43d6d9a7d7f6ec665ad830167ddff64",
            "attempt_01.replay_link.json": "5a8eebbb1c534842c9c3ce0cfce197d77d1a80bba2362f90f518c234fb4d3472",
        }),
))


_2357_OLD = retry.bounded.RunIdentity("-2357-relay-retry-01",
    "3c20dad3f4c772011e82d8684e7bfe4204a988e32f59b26b90390507ad6bb1d8",
    "5a507f2471b94d81b3eaae590d6c027bc2e8d3369a424550af0def0aea8c25f8")
_2357_REQUEST = "0c88dc80f734af1eee98a8d01d5992725e8a2ce68787a1a572150e714beb71b9"
_2357_FIRST = Review(_2357_OLD, _2357_REQUEST, "bde86b9f77d44228bc6d1d2ee34287e4",
    "e39f6aaa75b557ad92a973bdd9a2b85be915d1de98c3b07d8b0e0e574e9ba78f",
    "940cf534df4872eee6cbb51367170d9f882e0c59e0f3e4bfb7da1c0764aa2ab7",
    "4c50f3cf04f5dab96cd254ebe18e6d17e2c98d45f9dd35a38ff90eeeceafedc5",
    "619ed8ba393e46bd6e815dc6d4799bb89195b69569b1d3a0b9514a5b66b05679", "response_id_mismatch",
    {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch",
     "sse_event_count": 41, "provider_request_id": "8b970794-584a-46d5-95fe-b0f8cb601c9e",
     "response_id": "resp_084ae7d43e6d33e1016b07f373b83087d1955ad6ecedc533c7",
     "conflicting_response_id": "resp_8b970794584a46d595feb0f8cb601c9e", "conflicting_event_type": "response.failed",
     "response_id_matches_transport_alias": True})
SCOPES[(2357, 2)] = Scope(2357, 4, 2, "302fe66d28c6c9aa4cfd604a56d623cc0f6d7a8d4bd53f4ee67e194732bf39ea", (
    Review(_2357_OLD, _2357_REQUEST, "29e6b83977b14450904542b81b5efbd2",
        "8cea52ac8526860bf3a297c691238295482304125889bfef518cf1f89c243ead",
        "14b2e29fe47a84bd1c194d11f0adcb39f35f8283eb89a73d93a822117dd1fd76",
        "984f751b7e162ab3b73efec0d5637884c8af8df1b75a6122c7e7068f8ff9a198",
        "2f7e4ac250c929d338b61c3ad8f25a3ee678900468649a1f041f5ed8c08368b6", "timeout",
        {"http_status": 200, "phase": "transport_failure", "error_type": "hard_wall_deadline",
         "sse_event_count": 5325, "event_type": "response.output_text.delta", "last_sse_seconds": 2957.5922856668476,
         "provider_request_id": "48fda888-11da-47f5-9c99-222688c89689",
         "response_id": "resp_096ae4c1a60efb8b016b07f5cf8e7887d1b0e15f434504077f"},
        earlier=(_2357_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "3e6e8f5986a2dc8ea161d69ee967490341fbdcfb05bc8d8949c056d9a74ced77",
            "attempt_00.replay.terminal.json": "4d6e133baff411b80f58ff3a7ea15724f6ef8c40cc4a6616907e81a2ba08de70",
            "attempt_01.replay_link.json": "1d865c07dd2f269ad17f8b5991a3e0be6a1c14b3f5645fa8da1722b994a587cb",
        }),
), "failed_unix", 1795687279.5865693, original_start_round=4, fitted_prefix_files=(
    ("round_04/source.json", "2506c383d9e1a2de8e27635a36a4198121cc6772ad917685ecbdcbd31b605cc9"),
    ("round_04/boundary_memory.json", "e70a0d42f6aa9b94fc73046408bf4e2ce9d3082d896f5993909b29f3ca0d5620"),
), required_sources=QUEUE_SOURCES)

SCOPES[(2357, 3)] = replace(SCOPES[(2357, 2)], wave=3, cooldown_field="not_before_unix", cooldown_unix=1795688074.0102787,
                              original_start_round=None, fitted_prefix_files=(), boundary_physical_attempts=2, archive=(
    *SCOPES[(2357, 2)].archive,
    Review(retry.bounded.RunIdentity("-2357-relay-retry-02",
        "035481e8d0864a48445d4a604497bd9d934d21b1b6f6492c228222e3f606b5a5",
        "d2e03688bf51035fdb24f73e0d1c7dfe39a3306935a528a9bf2000fc42d01fd7"),
        "58764c4b906283071162ec91fca7e918ef30268454eb2ec804fcbec7b224330e", "8061ec9fd687497ebac1ef472bc086d3",
        "8a657b0c0ecfc0ae8b2f8656849bc8492b0b225695d84f8935fe27c9084749f4",
        "130c86c96e74c6476716d522bebc048364ea4f396cb404885c3d799ba204630f",
        "4c9f3ba23256ec6798f73e3ef54daa4eaa70bc99accb63d428cb329fe9bc0450",
        "b9d4d97dc0a314a9406930830f389f9dde3960ff3ac26329fc058c4d88fb2dcc", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch",
         "sse_event_count": 41, "provider_request_id": "2efb8fb0-cffc-49db-bcb2-18794136153c",
         "response_id": "resp_05dfcfb75072f710016b0805ba45a087d0ae49da3cdad433f8",
         "conflicting_response_id": "resp_2efb8fb0cffc49dbbcb218794136153c", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True}),
))

_2357_WAVE03 = retry.bounded.RunIdentity("-2357-relay-retry-03",
    "6adf6cdae83afb6ca89b6e2b04483f1b1436cfe7188579f3b250ee6548818c3c",
    "985d25fec830d958301ae43b786bf678590a38c890971e992216a0dc543026f9")
_2357_WAVE03_REQUEST = "68deaab12dd523312fcafd7eaeb27ec0589e876157b964ebfb52e6caf29bd417"
_2357_WAVE03_FIRST = Review(_2357_WAVE03, _2357_WAVE03_REQUEST, "86d12e7086c14d0aadd056d318ce3ebb",
    "6973100ab086b43dedaddcd5e649a574acf1c77acf531bdb901370171a425084",
    "e4e45a46db36440cf600def9745a19f0d244124d00ddcec17427f07da1a0a4e2",
    "91db59ce07f81d97fd6dd317614f188d7457422d43e0b773a5b8b39da12fb427",
    "0ca52c6243e664b6788c275457d8ae9d20192113a8bc287c6f6ec475adfee897", "response_id_mismatch",
    {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 41,
     "provider_request_id": "9e1b78e5-c013-45e6-80cd-d056c6a5b1ae", "model": "gpt-6.2-astra",
     "response_id": "resp_0bf066c8f8427198016b08099357e087d0aeae4c5257b4afa6",
     "conflicting_response_id": "resp_9e1b78e5c01345e680cdd056c6a5b1ae", "conflicting_event_type": "response.failed",
     "response_id_matches_transport_alias": True})
SCOPES[(2357, 4)] = replace(SCOPES[(2357, 3)], wave=4, cooldown_field="failed_unix", cooldown_unix=1795690463.3580766,
                              original_start_round=4, archive=(
    *SCOPES[(2357, 3)].archive,
    Review(_2357_WAVE03, _2357_WAVE03_REQUEST, "c7b932c208cd4e6f83d0879ef7c1878b",
        "ff68e1fdfa784741d52f4df56a79cc350303b9b6e6174863a8a49c059e18cc60",
        "a548417ce201488b9ca203e8b02edc4c179ec0c0e4233adc90683056e82ebb90",
        "09858195dcb9316f86f45d19d1ac8c5eecd9e7346f4e0541e64e71ad1be40661",
        "3bf15dcf933a56dcf5788a55559b0f23b47021d300bfa3cd2acf416566208257", "timeout",
        {"http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout", "sse_event_count": 14,
         "event_type": "response.output_item.done", "last_sse_seconds": 289.73664454207756, "model": "gpt-6.2-astra",
         "provider_request_id": "87a5a63a-102a-4c9a-8e13-c5c93cfa0ca0",
         "response_id": "resp_086e7bc1ad0aba55016b080b4527bc87d0bc905983bc86a586"},
        earlier=(_2357_WAVE03_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "b38e8214726b1296a5c0c27f2d0d60ff9daed00575a2bef2fffeff270b291a8a",
            "attempt_00.replay.terminal.json": "9546fb7633e5944238ce86d87d73b5e4a32a0a3d58f94b51af9a5821fa61d431",
            "attempt_01.replay_link.json": "808a709c47b4ff8e435eaa7559754c206aa81d55ce67adec6237adb97f8bf901",
        }),
))

SCOPES[(2357, 5)] = replace(SCOPES[(2357, 4)], wave=5, cooldown_unix=1795694625.301756,
                              boundary_physical_attempts=1, required_sources=SAME_RESPONSE_RETRY_SOURCES, archive=(
    *SCOPES[(2357, 4)].archive,
    Review(retry.bounded.RunIdentity("-2357-relay-retry-04",
        "fc19d508b0374d79fb7a015f50b68c606afaf4811c3c87fbaeaa216699759f47",
        "dc83d4755b0859fd1a7db4989f23541a6cd552f1418ea5745aa27ad2dff361fd"),
        "3aea456efc01f2a7a0d4b82e1c0b7b3a8bf1270079dfa7065811c33ae1b18d8f", "fff7561203064189b1f10fdd96e26e61",
        "685292e3160694d3cd861dab060c2247668d36aff62279e5a0f6df2c12853e87",
        "9e09937ddd425f157988ffeee0389317c50e388c583a49aea7f106c65fdd7b6c",
        "5f48a0457c89a6bfa32b0353993d3228d920faeef7d27e4089e792ad09ea5ec0",
        "17f568c466f3bfdd75e23f4082ab57c599007bce939db3231d5ad0c7b63387a0", "timeout",
        {"http_status": 200, "phase": "transport_failure", "error_type": "hard_wall_deadline",
         "event_type": "response.output_text.delta", "sse_event_count": 3170, "last_sse_seconds": 3181.4656675001606,
         "provider_request_id": "df645dc3-e434-4b35-a26b-14b507b18be4", "model": "gpt-6.2-astra",
         "response_id": "resp_05749099683d24b3016b0812a205dc87d090ff8bd9db36de56"}),
))

SCOPES[(2357, 6)] = replace(SCOPES[(2357, 5)], wave=6, cooldown_field="not_before_unix", cooldown_unix=1795697966.9968047,
                              original_start_round=None, original_boundary_physical_attempts=1, boundary_physical_attempts=2,
                              archive=(
    *SCOPES[(2357, 5)].archive,
    Review(retry.bounded.RunIdentity("-2357-relay-retry-05",
        "0ea96d7929e9bdc395bd7bbf05ccdeaebceb5805c5ba73aa8f44985635d57967",
        "ec14cc82c953da5ca45361aa2abec38f35a88d3b7319504f63941536e40db20c"),
        "d1003c4a55978f13c25302a0e22e335e6b3e0553809c65931ec6874b102ac4e6", "a6b141a821d649e5bc2a33ec5cab1638",
        "411bd92256e1c1cf38498e48dc1fcb655af4d69070402034af0fea9d2afd8654",
        "42399347447ba316b457e8b8df2c0ed48913ad03c4e864ff61597a13e23cc477",
        "62945fd3e4bb2a9c4765cb9a15fce22cd2f2687748a42e55694f3efc5bb6427c",
        "beb8b632fc743457b12c9d1328d37806cd739d849337b93ebfb8159ffa4599e1", "timeout",
        {"http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
         "event_type": "response.output_text.delta", "sse_event_count": 831, "last_sse_seconds": 1604.6706121659372,
         "provider_request_id": "b1a6449c-09ec-4b05-a253-aad6e5207e9f", "model": "gpt-6.2-astra",
         "response_id": "resp_0dbb7f15ca0e4a12016b08237b136887d293cf127a7e3cd4bb"}),
))

SCOPES[(2357, 7)] = replace(SCOPES[(2357, 6)], wave=7, cooldown_field="failed_unix", cooldown_unix=1795700982.1268728,
                              original_start_round=4, original_boundary_physical_attempts=2, boundary_physical_attempts=1,
                              required_sources=tuple(
    (source, "aacbc199e896091d4395fc036cb60bc604d91ec326a79fcff26e157d9a73705f" if source == PROVIDER_SOURCE else sha)
    for source, sha in SAME_RESPONSE_RETRY_SOURCES
), archive=(
    *SCOPES[(2357, 6)].archive,
    Review(retry.bounded.RunIdentity("-2357-relay-retry-06",
        "be3a0f8b24263bfec2b336a21d584ca70e65a578604f7c9d6e3b4085c7431d5c",
        "c3e3390acc7c3e6bf0b12105d79b6c87c4429b789fa72d1de4ac5e2920841eb4"),
        "02be42da96eb22bbe91aa1c6b9288c13303f964330728e6b1ed0d843d898f140", "46684da6118f47f5b6144ac8d60c11ad",
        "1ac263c754aaa0f5a44200f2815e4d1e8a17b1806e05f5600ab1ec5dae74d94a",
        "2ae5aea87b849a8d3ddd0cbc908855a348a574ac1bc2d6f9aef6e892a6e800b4",
        "691773641afc17c78915325644fc2918b45571d02a63e742566e9604415930dc",
        "5b44830f2e69aa1585764e233a89f6b4d786e8ef4a634915bf12c1f3e719c2b6", "timeout",
        {"http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
         "event_type": "response.output_text.delta", "sse_event_count": 440, "last_sse_seconds": 991.6895799168851,
         "provider_request_id": None, "model": "gpt-6.2-astra",
         "response_id": "resp_0d6c9b78cc72e853016b0831a918dc87d18fca9c2e4227ca18"}),
))

_2357_WAVE07 = retry.bounded.RunIdentity("-2357-relay-retry-07",
    "9982670e76d85b98ee971ac612c2f9126463946b7712c3cb2ff9802b1fae077b",
    "d2088d434902bcfd7a98df850f2757e88654788d656e4f88aa18a373a939a62e")
_2357_WAVE07_REQUEST = "6858cfa565da51909475a563cd8f8dd8ec02b7191b16009f521365af28d42a56"
_2357_WAVE07_FIRST = Review(_2357_WAVE07, _2357_WAVE07_REQUEST, "2c4159b1abb44af2abe95938b6f5014d",
    "e356e8e375891801a51898530271468ae6da6afd44e4c0f4b4d6d9602e1d293a",
    "6ed590d163f7915c8f43699fac87df073216e778d1daab129a4b1390a12ea918",
    "eb7fdd44e5d569638f5a6b54eaf5127ed6530660b4f1fcef63cefedd48ccce55",
    "4d592bf78daa2e0d5e73c2b7bb43681a746946a312dd8d6c20c1e330ebf9b40e", "response_id_mismatch",
    {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 23,
     "provider_request_id": "72e1def0-82cd-4141-a93a-3a8101295447", "model": "gpt-6.2-astra",
     "response_id": "resp_0b7f5126ab268cfd016b083e428d4887d0923f17b406214281",
     "conflicting_response_id": "resp_72e1def082cd4141a93a3a8101295447", "conflicting_event_type": "response.failed",
     "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"})
SCOPES[(2357, 8)] = Scope(
    2357, 5, 8, "9bbfd69fb3e88c99e81a9ae6d12df1df4488c8a4466bd2c86e925ec8f041db28", (
        Review(_2357_WAVE07, _2357_WAVE07_REQUEST, "80c5ef6e55cc497891e88860772de966",
            "3982bbff222cd93ddf924579a63e58f18d769a838d6f1e38c847ecae86745909",
            "da5ac22f12d9517cb611246fad8f575d44200b993f93d5e2154dc14767f14f9c",
            "33af82fe12d27779bc805489d407cefb2cf23278d2284c70d98faaf0abac7394",
            "068aa3de0d96fe4f4052c3e164adcc2f7cd95fbed2b2a466a3955e7db42c4691", "response_id_mismatch",
            {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 25,
             "provider_request_id": "22215ed1-3cc1-4de6-83e7-300e9681da73", "model": "gpt-6.2-astra",
             "response_id": "resp_08301c347a6f76ee016b084759604c87d0abb04a7ee20a94b8",
             "conflicting_response_id": "resp_22215ed13cc14de683e7300e9681da73", "conflicting_event_type": "response.failed",
             "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"},
            earlier=(_2357_WAVE07_FIRST,), replay_files={
                "attempt_00.replay_authorization.json": "7a4036560f2d270a3a2077dec3f44c27f19ce8dbad7e15e9acb11fec54a2a031",
                "attempt_00.replay.terminal.json": "fb62b42b673fd39235d6953e0f61574d4f40e96e45d3ab80c5e210fa5504d5e5",
                "attempt_01.replay_link.json": "969c43ad504e1c4bb2641ef163f0fa757513bcd0447a9d9e4cee144d69acc892",
            }),
    ), "failed_unix", 1795704880.1421165, original_start_round=4, fitted_prefix_files=(
        ("boundary.json", "15f0871e497fb397fa8b9088ccb8dede87e7b42812ed18471b79385d4b685d20"),
        ("round_04/selection.json", "418c30b287d441f4cac20717bd3bc341bcf6cb35382634e6d8263c57ef142dca"),
        ("round_04/batch_freeze.json", "54660ed0cb3910c18f5215b195ba3691aa091283b47e28a8e9124962152e3da0"),
        ("round_04/memory.json", "ca68e2011ba224b51a5d718fbdda208aa64896fa606bcd484623e44aec9823d8"),
        ("round_05/source.json", "a60e0c92322ffb2a7f40fc9e5a20894aa6905e5048eaf88520088e08c5b25d00"),
        ("round_05/transition.json", "bf855d43455d4ae2e1d2ff7d50a17b94be5c1adff2782c47fa5011d12f84703a"),
        ("round_05/fit_000/fit_audit.json", "4e89b08f06578ea4d901a062403667ab3508787629b93b70ccc5ecdd7b69aa80"),
        ("round_05/fit_000/validation_metrics.json", "f6397e258dcaf052022b207166e93ddeb677ce02fa02a9baacecfd0e537f20eb"),
        ("round_05/authoritative_packet.json", "7b6ba694100cf4b61f17e5a69a50565e8ad8b290d8d45f068478af2b5dcfe101"),
    ), required_sources=SCOPES[(2357, 7)].required_sources, boundary_physical_attempts=2, original_boundary_physical_attempts=1,
)


_2357_WAVE08 = retry.bounded.RunIdentity("-2357-relay-retry-08",
    "56abeeb8d3c4ad38c286a3b258a34298659f5596a8ee6bd970473d5e450c06ad",
    "56127c93850689717abf865b3b856e07d73a26b9779d608d673121988f8990e9")
_2357_WAVE08_REQUEST = "4e74466519da52c01150e90ec5407401b9b890bf016b88c64ed61095da9af971"
_2357_WAVE08_FIRST = Review(_2357_WAVE08, _2357_WAVE08_REQUEST, "dd59937e8b7a49a085984f24ee0effb7",
    "27cd276c4ee00893de0c177dde6ddb8d0bd9854bb092d3af761a5ff1c00195d2",
    "d9cf7e90039b6e994ce800d71db2cd3aea812601412cfc2c87d6ae8bc5e64169",
    "262bdbf16241edb3f02344ec071637a5aac806a03012a111a8a66ce2bef2df40",
    "2373314db6652c9ec8f995e650d7b401cc678d4e5f8f9d2183240062aad0de24", "response_id_mismatch",
    {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 25,
     "provider_request_id": "9f91c938-6161-47c5-9146-4ca36eb8cca6", "model": "gpt-6.2-astra",
     "response_id": "resp_06e394fba6f72e74016b084a44f8a487d0aa73ff60e493ac62",
     "conflicting_response_id": "resp_9f91c938616147c591464ca36eb8cca6", "conflicting_event_type": "response.failed",
     "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"})
SCOPES[(2357, 9)] = replace(SCOPES[(2357, 8)], wave=9, cooldown_unix=1795706488.8023124,
                              original_start_round=5, original_boundary_physical_attempts=2, fitted_prefix_files=(), archive=(
    *SCOPES[(2357, 8)].archive,
    Review(_2357_WAVE08, _2357_WAVE08_REQUEST, "f4b70a1204d74013a247a4d8653ec3e9",
        "a39c2a0d58719bb6441da89c1eb641815e24f1a3ace0e4e8df39ccf4803aba6d",
        "c85d5f55acef98c9b571e82abe97fc4b544eb557f08a8a11b27a6194262c3ca9",
        "d9ff8d5817c358605ff5d6e7fb3d2df1e2d9c12e3346fa5eef25997880432987",
        "184e2ac31b3c9abfa6ab43290d46207eef16888366fdc653a46bcdc7e9949115", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 25,
         "provider_request_id": "48ad4b49-24e9-4e69-97ed-3f8dc51d1ade", "model": "gpt-6.2-astra",
         "response_id": "resp_0fff34fcd9c0ee76016b084da47b5087d0887fa5dc3059c3a7",
         "conflicting_response_id": "resp_48ad4b4924e94e6997ed3f8dc51d1ade", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"},
        earlier=(_2357_WAVE08_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "2290e6d843b58e45aa5c266c49f994258261eee5a9c1dd7830b0f890f263dc9e",
            "attempt_00.replay.terminal.json": "e5cc4ee4ad93c3cf6015e4f49293c1fab3012c7c0760bcae96755c8cda92fbd0",
            "attempt_01.replay_link.json": "33b610278022a90ddea6d0ab36f042fed266d1a31611c70fee05bef7d4e6b1e1",
        }),
))


SCOPES[(2357, 10)] = replace(SCOPES[(2357, 9)], wave=10, cooldown_field="not_before_unix", cooldown_unix=1795709483.170326,
                               boundary_physical_attempts=1, archive=(
    *SCOPES[(2357, 9)].archive,
    Review(retry.bounded.RunIdentity("-2357-relay-retry-09",
        "f85aa65e0caeb48b419927c074278f0b97ee483c66180c06eab2e2588aa85f77",
        "fd640132095e581fbab5377114ceff22f06d9931f30adb5bd9b9c6b31a9df663"),
        "cdcdf87d1d2ccd568decbe711b4ecda2c3dcf2f7d86620cf603c0d97ff304b65", "c75a2df1d54c4abeb9c11e242dbde066",
        "b581e90bd4f97439462554ed51aa58de9b5b993022b3ac1cc7f6037e23d009c7",
        "5624186190f2ea4cb79f2053e53c1a02b873708dc33b334cb645d51ee983fa39",
        "2653d4924e63d5999b2f78fb6b860460d820b7a730395a9dfc6352588f355d14",
        "909a5d5e6d3654dfc7259ac8bdd39a0d07085293162ef05e2b7b262aae148f3a", "timeout",
        {"phase": "transport_failure", "error_type": "ReadTimeout", "elapsed_seconds": 1828.1092890840955,
         "network_phases": {
             "direct_ipv4_preflight": {"duration_seconds": 14.24446979095228, "observed_seconds": 14.263882874976844, "state": "complete"},
             "direct_ipv4_preflight_1": {"duration_seconds": 14.24446979095228, "observed_seconds": 14.261098542017862, "state": "complete"},
             "direct_ipv4_preflight_1_dns_tcp": {"duration_seconds": 1.6504104998894036, "observed_seconds": 1.6687907089944929, "state": "complete"},
             "direct_ipv4_preflight_1_tls": {"duration_seconds": 2.2314766249619424, "observed_seconds": 3.903982625110075, "state": "complete"},
             "receive_headers": {"duration_seconds": 1800.0037664161064, "observed_seconds": 1828.0736891250126, "state": "failed"},
             "send_body": {"duration_seconds": 10.512595750158653, "observed_seconds": 28.06884575006552, "state": "complete"},
             "send_headers": {"duration_seconds": 0.00021974998526275158, "observed_seconds": 17.555847208946943, "state": "complete"},
         }}),
))


SCOPES[(2357, 11)] = replace(SCOPES[(2357, 10)], wave=11, cooldown_unix=1795711578.937361,
                               original_boundary_physical_attempts=1, boundary_physical_attempts=2, archive=(
    *SCOPES[(2357, 10)].archive,
    Review(retry.bounded.RunIdentity("-2357-relay-retry-10",
        "fd02d26a0e700633a21bc4f26214be9393d76fad1a852d4482262371b1b8959f",
        "779a4a91f62c6686dd6979c14624e3bccbf7522f895431e06685eeca21525388"),
        "261148011c2e9354de302ebbd9323dae15536a1130677ea9b79244bb601a0d20", "d2a8dbcdfbf14301b3e91b626cdfa716",
        "c8de05a1c0d4409c6cb8a89341b0e12def7d356a2685057f52c81e928766a4fd",
        "377af21c447cbb62d61d5c1a9153adfba7ddfc5783f4605de4e84790343e2689",
        "28c5247df4511c6d2f8b58efc48322f3eafcc6e492ed1bb1a2ff01d002151f18",
        "df8b5fd031ffc15dc39538d9fa88cfdcdd7cba664468dbe7bf2558236dc139e7", "timeout",
        {"http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
         "event_type": "response.output_item.added", "sse_event_count": 17, "last_sse_seconds": 501.662876291899,
         "elapsed_seconds": 1401.6690100000706, "provider_request_id": "7b25cdeb-4598-44ba-98fb-b5a3c924696f",
         "model": "gpt-6.2-astra", "response_id": "resp_0bc4c24a3c1b9d40016b085cfec74887d28fa81ed2a6343b27"}),
))


_2357_WAVE11 = retry.bounded.RunIdentity("-2357-relay-retry-11",
    "e51b23c0ce9271c6c506fc0bb10510e972f5821f4741c793385b8aa014748ce6",
    "91f1d4e2238a5189ff2cdc3ae96607dfcea2346f17ed7d98c114f92104b473f9")
_2357_WAVE11_REQUEST = "c734d0704db6d98ff61321c2ceae8a0e06100c976cd681662b0e96fec98d38ee"
_2357_WAVE11_FIRST = Review(_2357_WAVE11, _2357_WAVE11_REQUEST, "67bb14f783184999a8ba51aecb620cad",
    "56e9a3b53001938eaa18d4cc489f36854c34f43723b966a70b47abcac57ff08f",
    "61786745aeae39d1a1689828b47373f88250f6f2b99ae878eda51734195c77e6",
    "e0f45cde999a2402b61a3fd80b70d5ea0835979d4ddff2f709c01a51afb58123",
    "03a53d21cb73f381911cb56662ce28a864762e2e43378da0b8c9fabff6183427", "response_id_mismatch",
    {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 41,
     "provider_request_id": "59786f11-f5a4-4b14-bf0c-08c0fe88807e", "model": "gpt-6.2-astra",
     "response_id": "resp_0d1745a9a89ad2d5016b086549206c87d2a230cd03218aa896",
     "conflicting_response_id": "resp_59786f11f5a44b14bf0c08c0fe88807e", "conflicting_event_type": "response.failed",
     "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"})
SCOPES[(2357, 12)] = replace(SCOPES[(2357, 11)], wave=12, cooldown_unix=1795713181.660755,
                               original_boundary_physical_attempts=2, archive=(
    *SCOPES[(2357, 11)].archive,
    Review(_2357_WAVE11, _2357_WAVE11_REQUEST, "5ee7f72f23d3432db2cb674ee0c809e3",
        "ff996b9c94bc01ca7f563a9c43f436403eb7d99e3f99818fa6cc7b370b5407cc",
        "b60125bd3428d75e95c88c06f0da44acf7ac3175b678c119464019b7af7b3ffc",
        "652f819e1f820de798db4ee803a6b34508ca26a4970f080471858c6372ffc47a",
        "ad27ef2a28d57adc22b7ac1a50cf57dfb511ef3df6c51ffee9940f5b5642c42a", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 25,
         "provider_request_id": "6e90dbbd-eaaa-48e2-a396-2d6fcbb1219a", "model": "gpt-6.2-astra",
         "response_id": "resp_0701b3cd45663007016b0867c08f5c87d184be9b44d19aa5a3",
         "conflicting_response_id": "resp_6e90dbbdeaaa48e2a3962d6fcbb1219a", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"},
        earlier=(_2357_WAVE11_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "95475c02ab29c825ea18ff55690ebdb428a343ad60861618c44a9a6f07f7f1d3",
            "attempt_00.replay.terminal.json": "b8e7925c3f608af1045608445f2688a1781d45e4d4722317be8dc1d3b46e6134",
            "attempt_01.replay_link.json": "5717bfca88e6f0297cff08de37a963734045718a50fe705f4f801e95837f8d6f",
        }),
))


_2357_WAVE12 = retry.bounded.RunIdentity("-2357-relay-retry-12",
    "b78ed3a580e8c22077b87818c988b83e30d05a41bb93971ee8f65bbe35af8f60",
    "60dcf56a1b35c79b3971bf6fa94e1d7809326509d479cae5be49cff4454d286b")
_2357_WAVE12_REQUEST = "7e4eaa2366d785ac178e03befa08cf5582eae475865a33a27d9c17fd518e1b65"
_2357_WAVE12_FIRST = Review(_2357_WAVE12, _2357_WAVE12_REQUEST, "60096c8a5cf849d6a5c3a06bd203691b",
    "f4662af19613dded4e1be6f2064320c90239c34e07497b2480eb3b2d4a98cbed",
    "19b91ff37a2cb42fc31b29f662e2d89210098816fc261a65d3b3d279eeaa4a4a",
    "6dc8d03507b5f8b8e8c87969a7a67997d7d6a8bdc14243113f6228b55444d658",
    "9c04614797dc4507f6e96b300b7cefc51c2c1690847f07edb7b791eb81487b68", "response_id_mismatch",
    {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch",
     "event_type": "response.output_item.added", "sse_event_count": 23,
     "provider_request_id": "c9b80ff2-3fdf-4c7d-9cab-35e55da76f08", "model": "gpt-6.2-astra",
     "response_id": "resp_05a5af9255b94efa016b086b5670bc87d1a2b4f78ebff2b992",
     "conflicting_response_id": "resp_c9b80ff23fdf4c7d9cab35e55da76f08", "conflicting_event_type": "response.failed",
     "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"})
SCOPES[(2357, 13)] = replace(SCOPES[(2357, 12)], wave=13, cooldown_field="failed_unix",
                               cooldown_unix=1795714511.0751874, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(2357, 12)].archive,
    Review(_2357_WAVE12, _2357_WAVE12_REQUEST, "938a1fbd8a66479f87505a2824ba1c45",
        "269c709de787770924539b40516ac3840a5ab09efcd9fffbf23b58e7cb6ca1ae",
        "e069044a1b0da28194f44205ffc8a796d1e1917e3104d7dc23c6b5220f9d7d91",
        "19fd3b81b8ccdce193cb59ae5d4c1edca3f356bd8b3de739b3688cafcd6684e2",
        "5c341138e6cf4e2eaaa990cdc7a518b54164a2176a1ef0b247676772450a584b", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch",
         "event_type": "response.output_item.added", "sse_event_count": 23,
         "provider_request_id": "61398b0f-99ec-49ed-852b-50749894b751", "model": "gpt-6.2-astra",
         "response_id": "resp_0414566f3ddd6909016b086cfea8e887d1a92c9ad38fbbdd41",
         "conflicting_response_id": "resp_61398b0f99ec49ed852b50749894b751", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"},
        earlier=(_2357_WAVE12_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "70682c9eaded0a16258580f17f2e24d29381504f061e494456607fcd8c708cb4",
            "attempt_00.replay.terminal.json": "51e39e4bed4fb1c52d889d6264e2a6b3b339f0bf48f31cc02b35a2ba4a22ed18",
            "attempt_01.replay_link.json": "8352325daa3f308d30727c140285464dc4ab68dda2cebc8714169505e595cb25",
        }),
))


_2357_WAVE13 = retry.bounded.RunIdentity("-2357-relay-retry-13",
    "971509317d68b4d787bb09d8f41e6d0d3dd7d2dad86e446a8b1aa1be3881a578",
    "b885108529b6475fa90ad880d4bd28565951fbd48f48b08746373db6038086ea")
_2357_WAVE13_REQUEST = "42e6dd57dacdd98571a74ef264ffb6ec15a910cfc7fd4cb4a007ea32798ba815"
_2357_WAVE13_FIRST = Review(_2357_WAVE13, _2357_WAVE13_REQUEST, "4e9bf9c29a3842c6aeb16187db906b32",
    "f46af29b4109dbd02959c2ae52c6bf09665f11fa42f7958162085b95dae1b6e4",
    "2a4da8598d7fba845a5916bf3a9757c3f08a8af342b9b97633e1cabe2555f799",
    "599db9bf2b092041e422b4f5e3af0c12bb133a8f1e9b1958768ed880ac43bfb3",
    "46c1f0cb04167938f884fed7b54fa8c5794e623a14ff8bb377d0d6129987dc2b", "response_id_mismatch",
    {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 25,
     "provider_request_id": "578fca6e-f57e-4a4c-9263-bd361f7f67b7", "model": "gpt-6.2-astra",
     "response_id": "resp_0bc77e5b85e85de7016b087089989087d1b95e166f1fae42b6",
     "conflicting_response_id": "resp_578fca6ef57e4a4c9263bd361f7f67b7", "conflicting_event_type": "response.failed",
     "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"})
SCOPES[(2357, 14)] = replace(SCOPES[(2357, 13)], wave=14, cooldown_field="not_before_unix",
                               cooldown_unix=1795717206.3467476, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(2357, 13)].archive,
    Review(_2357_WAVE13, _2357_WAVE13_REQUEST, "c7838e3483fb4e3c8217729061109eb5",
        "e7588b4ebc499c3650490fab13cfc741ac864d0070bba06c79b64a59b7adb106",
        "adb99ba9e070fa4b4c1fa08a3aab0b2b84bd2b67dde4ba0e8a103f0784a04010",
        "3ff6a01fd8e989857a6d0a62854505beea16ce569b2960fb78dd8874c6a3c926",
        "354ce997e058bf28997db4560adb2b4ee9cc27d4d52cde2502fa5190ceb3eb08", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 41,
         "provider_request_id": "debb12eb-8ca4-4182-b805-e508591d8f51", "model": "gpt-6.2-astra",
         "response_id": "resp_0e0ee04155c2ceed016b087787aeb087d2b946c64f4a174fda",
         "conflicting_response_id": "resp_debb12eb8ca44182b805e508591d8f51", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"},
        earlier=(_2357_WAVE13_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "0dea3618537cb37d373ec63f6d73e0d7a5514d8006dcd8c795819c920290af30",
            "attempt_00.replay.terminal.json": "420c9ddf47dafe744208c97e9363b32339a6249b87e00b7eb74c9628ba038579",
            "attempt_01.replay_link.json": "6a33bded3147f04c4ba4e1d25c41645ad0668b9a233956f8f2d483a424016d2c",
        }),
))


_2357_WAVE14 = retry.bounded.RunIdentity("-2357-relay-retry-14",
    "16bb17dff9acf067e8ac53500b4795c5a08bc599b2be0fc8aed6e4b4bf13ef4e",
    "0d377d4a8a095733404e6e6453e396bd8900d2c06a0cf9d02f43f940d8639a5e")
_2357_WAVE14_REQUEST = "83813669549a2b0a9761c2ad32f8fd26efd76068923f814ba7aed08ce787f6ba"
_2357_WAVE14_FIRST = Review(_2357_WAVE14, _2357_WAVE14_REQUEST, "d623a45515c943cd88a4c93633a670c4",
    "2c856b8ce3d8f6e5bd6dc6b955a8b660ddeda4d556beb5d10ce36b2145620210",
    "8fa25d857d5d236c805211f3d38547a689f83bc11cce783dc71fce58524ca591",
    "2211e2ceba9fdb9cc17703aa0b53cd4845410f9c8432baebbeecb60ea789b767",
    "115f017697f4e7348cdb0c469193107762660a036d99968b9ff2265c039c577d", "response_id_mismatch",
    {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 23,
     "provider_request_id": "20b06f18-166a-4ccf-ad33-2c18a9cbbfff", "model": "gpt-6.2-astra",
     "response_id": "resp_032e03a58b6e1473016b087b443bd887d1bfabc4a2a84fd8ba",
     "conflicting_response_id": "resp_20b06f18166a4ccfad332c18a9cbbfff", "conflicting_event_type": "response.failed",
     "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"})
SCOPES[(2357, 15)] = replace(SCOPES[(2357, 14)], wave=15, cooldown_field="not_before_unix",
                               cooldown_unix=1795718582.0906043, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(2357, 14)].archive,
    Review(_2357_WAVE14, _2357_WAVE14_REQUEST, "bedbc8d4462b41859c8ed5cc4415fbda",
        "0bb64f5cc47aba3f5272fe6ce7697924668dbd63f327c7c7e5746f018a731b8b",
        "a1d258bf3228429b31d5a96410dca079d80078e07565066b17409464ea5be46f",
        "b0b295186d112c9495e45452d97cfc6182e0ac711fc71349ddc90c402612c06f",
        "35c7d6025d1f19f26e35a2464a7ade46030a115f62ff5b08ac2631ee23a016b5", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 25,
         "provider_request_id": "4ce87799-dd31-4af2-af55-93bba3339782", "model": "gpt-6.2-astra",
         "response_id": "resp_0a14140aa36c782e016b087cdc282887d1b93327570b3de8b1",
         "conflicting_response_id": "resp_4ce87799dd314af2af5593bba3339782", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"},
        earlier=(_2357_WAVE14_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "5ace23c35815a5f7e9386385034582f11997ac6f734e96ba79ce07c74d523459",
            "attempt_00.replay.terminal.json": "0c52248723d54dd7ad196d0046598c2f93612ecf85cdccdfb901798c8b75b090",
            "attempt_01.replay_link.json": "811c37ed8a98533b809696e54486a7c3ad987c735eaa8dae3262068449769c42",
        }),
))


_2357_WAVE15 = retry.bounded.RunIdentity("-2357-relay-retry-15",
    "128f918c7a57c055ec4aa6cb1e1e587363f5f667ece2a68ac3b6e5715d64ff62",
    "c071817654fc799bad48e2d0433635315f2e2d5207df65d00eb77bf43ff9622b")
_2357_WAVE15_REQUEST = "d3cfdf7c122f2ce4b33247003d974f2fe883c4ad6effe6ec340703cfb5ef6b8a"
_2357_WAVE15_FIRST = Review(_2357_WAVE15, _2357_WAVE15_REQUEST, "8e36b8d104ca4862997060e83c02b4b9",
    "788145f476d9f48f80872424b2eac52ced920ebc8717843da4470fcba61478c6",
    "d4589d7016e0d3a44d8fcc59a4375e71cfd9be807784abb7674431e359daffe8",
    "053f6d85a2873f0bc6fe564f9a344852f2c5815349702c040351da5f910a70bf",
    "27f21966c6678f9a88dd871c0e2731a3c934560f78a6229511b616746e17e095", "response_id_mismatch",
    {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch",
     "event_type": "response.output_item.added", "sse_event_count": 41,
     "provider_request_id": "552b7589-7b14-4d84-aecc-5944b46f72c4", "model": "gpt-6.2-astra",
     "response_id": "resp_03a8e6f9d458d54d016b087f55ddbc87d2bd3c6d569db02f7f",
     "conflicting_response_id": "resp_552b75897b144d84aecc5944b46f72c4", "conflicting_event_type": "response.failed",
     "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error",
     "last_sse_seconds": 194.83617429202422, "elapsed_seconds": 197.62899016682059})
SCOPES[(2357, 16)] = replace(SCOPES[(2357, 15)], wave=16, cooldown_unix=1795719604.1234941, archive=(
    *SCOPES[(2357, 15)].archive,
    Review(_2357_WAVE15, _2357_WAVE15_REQUEST, "6b1a6ffe4864450abae7e02daed30584",
        "a1e68f4d485e7c664454574e48b0195ec94fd17b81e14406c97d6cbdff4fad2b",
        "e4728fe60caf902a4a944ba42a5f70f3dcbca298b4a361934614f8f6a976e966",
        "91497035974234aa6433cc222c6fd76231712933b8843927d0051c6a46f7ed99",
        "30fa5ce4b14c2684c03d211a117c0bcb0db629eea383e12db25c39f395e977a2", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch",
         "event_type": "response.output_item.added", "sse_event_count": 41,
         "provider_request_id": "b8c41ed7-e61e-44da-b3d5-862c071825ca", "model": "gpt-6.2-astra",
         "response_id": "resp_0c05a0f34761c7f1016b0880e4c5d487d2bccf69184f1dc200",
         "conflicting_response_id": "resp_b8c41ed7e61e44dab3d5862c071825ca", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error",
         "last_sse_seconds": 196.09210079186596, "elapsed_seconds": 197.80961004202254},
        earlier=(_2357_WAVE15_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "5f788909ce9bb3b7499d2288b461066802163b4c1a2617a6721187ad9d218d38",
            "attempt_00.replay.terminal.json": "d1dfd7da1195530e09083797cb188c33d6a28f32f7f94da96c86acf855144ebb",
            "attempt_01.replay_link.json": "a9fb5a334645c5debd1c7854b260dfca8489bd990d419fc7ab925e986d8afd74",
        }),
))


_2357_WAVE16 = retry.bounded.RunIdentity("-2357-relay-retry-16",
    "a1012cd98e9968430ff976bef25c470aa30ba3c75be21e8ccdf3253f1fd77a99",
    "e643c5a7cf8ec93ee251eb94ce181b5278e739250bba9e7d2f5067784d0c1590")
_2357_WAVE16_REQUEST = "c8623b69e487601a7b2cde8e4e317863e313ffccfa8a6ec8cf1e8676e77e9dd0"
_2357_WAVE16_FIRST = Review(_2357_WAVE16, _2357_WAVE16_REQUEST, "1cded70dd18448a4b1e4338f03c05c51",
    "44125edd93b272f16ed1dbe67e05ed369a4491cc3b72de79f83dfa96b5f8ec0f",
    "b35d28d86a0f7633fa14e16315bbef0a2d0eb8b8b78856fcd32bbe339a11afbd",
    "9e008bdaad6940eb8a612c335036c44e4369ccf227f3870ba0fe23996b1aa66f",
    "33fd604f1581d4680a964d3681ad68d276c3b94761141a3a44002f47ab4b587d", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'sse_event_count': 23,
     'model': 'gpt-6.2-astra',
     'provider_request_id': '92cf3afd-5811-4aa4-b9f7-1a2d30ca9312',
     'response_id': 'resp_0ae25716f798cf27016b0887f8fc3087d19ea9e9cb1195a9ac',
     'conflicting_response_id': 'resp_92cf3afd58114aa4b9f71a2d30ca9312',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 198.4292452079244,
     'elapsed_seconds': 212.25270220800303,
     'sse_milestones': {'response.created': {'elapsed_seconds': 32.12956970813684, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 32.13065887498669,
                                                 'interval_seconds': 0.0010891668498516083},
                        'response.output_item.added': {'elapsed_seconds': 32.13132962491363,
                                                       'interval_seconds': 0.0006707499269396067},
                        'response.output_item.done': {'elapsed_seconds': 49.37365833297372,
                                                      'interval_seconds': 17.242328708060086}}})

SCOPES[(2357, 17)] = replace(SCOPES[(2357, 16)], wave=17, cooldown_field="not_before_unix",
                              cooldown_unix=1795721826.3807504, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(2357, 16)].archive,
    Review(_2357_WAVE16, _2357_WAVE16_REQUEST, "1f7361cc146f48fbad269764e61a17b8",
        "441149d2ba7c85290e5bbcda4500a5bcb4ebf8e04e6988743a4448a39267de03",
        "0919f07fb7d787c4e3c472bdb275437122c276fb4071f7625264f894aecba141",
        "b990236af5b0b825d2d6d7e197f40a45bec236e944fbe2b8319919f35e862fe0",
        "fdbf4cee9c5ba8eb79516db622e1dc2df4f267288d37877a6a6e76fa340c7035", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'sse_event_count': 41,
         'model': 'gpt-6.2-astra',
         'provider_request_id': 'd5147b62-a362-4996-9450-dfea6471f409',
         'response_id': 'resp_00867897eadda728016b08899466f087d2b58b0392a99d2e02',
         'conflicting_response_id': 'resp_d5147b62a36249969450dfea6471f409',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 198.49031345779076,
         'elapsed_seconds': 201.8770019998774,
         'sse_milestones': {'response.created': {'elapsed_seconds': 21.899365790886804, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 21.90066649997607,
                                                     'interval_seconds': 0.0013007090892642736},
                            'response.output_item.added': {'elapsed_seconds': 21.901749040931463,
                                                           'interval_seconds': 0.0010825409553945065},
                            'response.output_item.done': {'elapsed_seconds': 30.416553374845535,
                                                          'interval_seconds': 8.514804333914071}}},
        earlier=(_2357_WAVE16_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "064979c82fa2e4a42adb9596e95d3e7e16c476a344a3b01ca6c58365120e2377",
            "attempt_00.replay.terminal.json": "0669996a4aa5c3d09a1d3bac854e626b45f5f5673f7823043199d55bf971eb22",
            "attempt_01.replay_link.json": "2327fccd4d88ecc08b64f58a46a768af9fb1a3559fbd74efdacf62af7b1842a9",
        }),
))


_2357_WAVE17 = retry.bounded.RunIdentity("-2357-relay-retry-17",
    "9c5b7c8c7850af16feb75642d5ea7f29fe45a5d503e2d2ad1c65450bc4d79b3f",
    "a56bb51b861e1a6fb07dd52e7d140a43903cebb159c585ab325f55c301b876b9")
_2357_WAVE17_REQUEST = "7c0d8d6d5f4ba90e4b52ceffd9fe27a40689aaf3b61d2c1a1d1dd0ef89d5038b"
_2357_WAVE17_FIRST = Review(_2357_WAVE17, _2357_WAVE17_REQUEST, "90186ecb2e534a0c89e883969ad487d3",
    "df044a8493d99e31eef3700cb99389ded7fe753f4c603723f6e55302bb7fa8c3",
    "a5a8eed6cff2cc264dd35deb0a62de2ca361b6c2be4b8dbaeb8b88b9d6d0a4ef",
    "5504e96e4ad6699a39ce568b006b06c37aff849ea206b3fccea599f1f85cad62",
    "65ae88bcef37f19107c204d2c9ddebdf1a52a1cf417ffa6fecc0ec386654e873", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'sse_event_count': 41,
     'model': 'gpt-6.2-astra',
     'provider_request_id': 'c60f9384-dccf-43ac-ae17-fb2560710eb4',
     'response_id': 'resp_0e116bc4e545a2f2016b088ce2cc3887d199f9e38772327215',
     'conflicting_response_id': 'resp_c60f9384dccf43acae17fb2560710eb4',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 198.61201649997383,
     'elapsed_seconds': 201.6338657499291,
     'sse_milestones': {'response.created': {'elapsed_seconds': 21.59259279211983, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 21.59367175004445,
                                                 'interval_seconds': 0.001078957924619317},
                        'response.output_item.added': {'elapsed_seconds': 21.594360042130575,
                                                       'interval_seconds': 0.0006882920861244202},
                        'response.output_item.done': {'elapsed_seconds': 30.835152457933873,
                                                      'interval_seconds': 9.240792415803298}}})

SCOPES[(2357, 18)] = replace(SCOPES[(2357, 17)], wave=18, cooldown_field="not_before_unix",
                              cooldown_unix=1795723090.9339292, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(2357, 17)].archive,
    Review(_2357_WAVE17, _2357_WAVE17_REQUEST, "749dd3bcaf984c778355742ea5a8937e",
        "be33da3381ea93ae51ca1af4e0a41c335202e971f4fe7a1117d612323be9e2bc",
        "403ff997086ef987b8c4d5819b03e8cfc2d2ee505f5af13a1bce09e98ee520c9",
        "ed30f2de9cf8e0567446f8652306f0e22551efa4ebef49ecd36a4dbc71d722ed",
        "3b44ad22c98b1c3d1ad28eb91423552d87fe482ccfba94c53e05d97a197a4f6a", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'sse_event_count': 23,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '8ed055a3-9381-495c-9076-5aa211ebf3fa',
         'response_id': 'resp_0c36d980c860b812016b088e72485c87d2adbce42f8bfd723c',
         'conflicting_response_id': 'resp_8ed055a39381495c90765aa211ebf3fa',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 201.00235691596754,
         'elapsed_seconds': 215.57813533302397,
         'sse_milestones': {'response.created': {'elapsed_seconds': 35.58365466585383, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 35.58796533290297,
                                                     'interval_seconds': 0.004310667049139738},
                            'response.output_item.added': {'elapsed_seconds': 35.591414040885866,
                                                           'interval_seconds': 0.0034487079828977585},
                            'response.output_item.done': {'elapsed_seconds': 53.547473415965214,
                                                          'interval_seconds': 17.95605937507935}}},
        earlier=(_2357_WAVE17_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "be92d5ec1248b306e7897b38f7238388cddc2b56ba3ccbe8e895a12bcac21e15",
            "attempt_00.replay.terminal.json": "ff95e7ffbbff3cc7d42676c8fe9240db1411eb0a30637cefef606833327c26e6",
            "attempt_01.replay_link.json": "ff44e05691806c1483e2abd76882961a104c8275e6c19f2750febcff6db703ec",
        }),
))


_2357_WAVE18 = retry.bounded.RunIdentity("-2357-relay-retry-18",
    "e6382ee8b0d441879aea0045e712183ed72ab305c046ad3320502c84f366bdc0",
    "82ef39e050f9306309e4b92679fdaefc481b41ab22dd60305f3d2f4fd4bafbb6")
_2357_WAVE18_REQUEST = "6cfa51b09c6efe21d11eb3eda0568d105ed608f285b7022d2c53facf40959985"
_2357_WAVE18_FIRST = Review(_2357_WAVE18, _2357_WAVE18_REQUEST, "33970472c3ba4a81963eaa903fcf1ef6",
    "d2b3f2880dbf0ed91a547fa4841d6b34e08fac1bbca93e65645c67db7ccb2a9a",
    "f7743c112bfa58179baf69a1b8ca7491638652c304efa10820f5ff4a5f05764a",
    "b9682c0ac55fe8ef819b8ccf415575404dd3f961468849cca35335f6fd0b45a7",
    "62c36834aba7ffab664954cc82a1d830b2ba0bdb43f83fdd5bf2bab00dfa5b74", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'event_type': 'response.output_item.added',
     'sse_event_count': 41,
     'model': 'gpt-6.2-astra',
     'provider_request_id': '4d92eb23-ae2a-426a-b265-907bd621ff15',
     'response_id': 'resp_0897dfa5d70cbdca016b0890b3fd1c87d185708b2749e8980d',
     'conflicting_response_id': 'resp_4d92eb23ae2a426ab265907bd621ff15',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 196.7272894999478,
     'last_sse_activity_seconds': 196.7272894999478,
     'elapsed_seconds': 199.51655941689387,
     'sse_milestones': {'response.created': {'elapsed_seconds': 19.443529791897163, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 19.444197417004034,
                                                 'interval_seconds': 0.0006676251068711281},
                        'response.output_item.added': {'elapsed_seconds': 19.444826582912356,
                                                       'interval_seconds': 0.0006291659083217382},
                        'response.output_item.done': {'elapsed_seconds': 28.731172292027622,
                                                      'interval_seconds': 9.286345709115267}},
     'network_phases': {'direct_ipv4_http11': {'duration_seconds': 19.392041499959305,
                                               'observed_seconds': 19.401977458037436,
                                               'state': 'complete'},
                        'direct_ipv4_preflight': {'duration_seconds': 0.7654818748123944,
                                                  'observed_seconds': 0.7777235829271376,
                                                  'state': 'complete'},
                        'direct_ipv4_preflight_1': {'duration_seconds': 0.7654818748123944,
                                                    'observed_seconds': 0.7755179579835385,
                                                    'state': 'complete'},
                        'direct_ipv4_preflight_1_dns_tcp': {'duration_seconds': 0.19213845790363848,
                                                            'observed_seconds': 0.20421016705222428,
                                                            'state': 'complete'},
                        'direct_ipv4_preflight_1_tls': {'duration_seconds': 0.1960709160193801,
                                                        'observed_seconds': 0.40249891695566475,
                                                        'state': 'complete'},
                        'receive_headers': {'duration_seconds': 15.04099424998276,
                                            'observed_seconds': 19.39806770789437,
                                            'state': 'complete'},
                        'send_body': {'duration_seconds': 0.7332742500584573,
                                      'observed_seconds': 4.35571466688998,
                                      'state': 'complete'},
                        'send_headers': {'duration_seconds': 0.00022858381271362305,
                                         'observed_seconds': 3.6220299168489873,
                                         'state': 'complete'}},
     'first_phase_seconds': {'http_headers': 19.4034240830224,
                             'http_phase': 0.20421016705222428,
                             'request_dispatched': 3.4714355829637498,
                             'sse_event': 19.443529791897163,
                             'transport_failure': 199.51655941689387}})

SCOPES[(2357, 19)] = replace(SCOPES[(2357, 18)], wave=19, cooldown_field="not_before_unix",
                              cooldown_unix=1795725658.6105614, original_boundary_physical_attempts=2,
                              response_headers_seconds=300, required_sources=tuple({
    **dict(SCOPES[(2357, 18)].required_sources), **HEADER_DEADLINE_SOURCES,
}.items()), archive=(
    *SCOPES[(2357, 18)].archive,
    Review(_2357_WAVE18, _2357_WAVE18_REQUEST, "a4ca21791ef94e748d7f414a4b36ee6a",
        "296e43ec72f4fe9268d94467fd0811f8fda2894cf31978cb1aa0b1cbcf8286c1",
        "560db268fa4923f480008eceff8be087964aa316085b5aa5290195a03a7a638c",
        "eb5ed8a5bcc4a0d830972f13f7a53db4169125805b099c9f3d16ba46a2bebfcb",
        "061635bd12af3fd206726d2c488ee673bee0d4ca4a8304b1798134ed40f2d11d", "timeout",
        {'phase': 'transport_failure',
         'error_type': 'ReadTimeout',
         'event_type': 'ipv4_preflight_http_401',
         'elapsed_seconds': 1804.6336837920826,
         'network_phases': {'direct_ipv4_preflight': {'duration_seconds': 0.8712731250561774,
                                                      'observed_seconds': 0.8765098340809345,
                                                      'state': 'complete'},
                            'direct_ipv4_preflight_1': {'duration_seconds': 0.8712731250561774,
                                                        'observed_seconds': 0.8744371249340475,
                                                        'state': 'complete'},
                            'direct_ipv4_preflight_1_dns_tcp': {'duration_seconds': 0.4168070419691503,
                                                                'observed_seconds': 0.420157833956182,
                                                                'state': 'complete'},
                            'direct_ipv4_preflight_1_tls': {'duration_seconds': 0.22410049987956882,
                                                            'observed_seconds': 0.6463227500207722,
                                                            'state': 'complete'},
                            'receive_headers': {'duration_seconds': 1800.004173916066,
                                                'observed_seconds': 1804.6267460000236,
                                                'state': 'failed'},
                            'send_body': {'duration_seconds': 0.807349625043571,
                                          'observed_seconds': 4.620853041997179,
                                          'state': 'complete'},
                            'send_headers': {'duration_seconds': 0.0001915409229695797,
                                             'observed_seconds': 3.8131987499073148,
                                             'state': 'complete'}},
         'first_phase_seconds': {'http_phase': 0.420157833956182,
                                 'request_dispatched': 3.810599209042266,
                                 'transport_failure': 1804.6336837920826}},
        earlier=(_2357_WAVE18_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "89780823bf9c7ea7f8a41de1132a53322a887aa0653c352b06de9517fd1cf1ea",
            "attempt_00.replay.terminal.json": "9a63e5b1325420d18b1cd24a048c800e93582cdc5951cb7b22d3afcd4dc669c5",
            "attempt_01.replay_link.json": "ecde4dd5cb8ac497e0ef115cbdcfb9c22f6e7c54b6f29249239be9deeb008efb",
        }),
))


_2357_WAVE19 = retry.bounded.RunIdentity("-2357-relay-retry-19",
    "cbb985566055d926c39413abd8e111a27b99df2b69ebb09490db374c50fb403a",
    "9baa2c6fb711fc8ef67797c7bb83e8b4542d79c03b8973bbf102e7b42f488a94")
_2357_WAVE19_REQUEST = "76bc9f2e4fb0bf76780b7b5cde86008f56d051a02d2f15cb96956db667901d6a"
_2357_WAVE19_FIRST = Review(_2357_WAVE19, _2357_WAVE19_REQUEST, "6babdfd158a0464fa5ed42e9a2fe50bd",
    "92dd5d2e6aaf3986abd2bc2102c4388c0a08c4becb191043efe240992a8a1ad2",
    "0401a0f57d7b505fd5a2494edb0f6e516f8cce5375086ac074401b85c0e63321",
    "192af9f7a98e0ea3f758921145b97f73c84810a896f2b130bb6aa438a2ab1c63",
    "e059276c5f50bfc98b3d82e196913d42060b45cb89b7dc59b0786447b59afadf", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'event_type': 'response.output_item.added',
     'sse_event_count': 25,
     'model': 'gpt-6.2-astra',
     'provider_request_id': '2e230c7f-0769-4fcc-bb67-bd17656e1bc6',
     'response_id': 'resp_02bb336d87a0366a016b089b82f81087d28fbcd17817871345',
     'conflicting_response_id': 'resp_2e230c7f07694fccbb67bd17656e1bc6',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 208.70969949988648,
     'last_sse_activity_seconds': 208.70969949988648,
     'elapsed_seconds': 213.7336898748763,
     'sse_milestones': {'response.created': {'elapsed_seconds': 33.72797837504186, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 33.72926050005481,
                                                 'interval_seconds': 0.001282125012949109},
                        'response.output_item.added': {'elapsed_seconds': 33.730225540930405,
                                                       'interval_seconds': 0.0009650408755987883},
                        'response.output_item.done': {'elapsed_seconds': 50.687942165881395,
                                                      'interval_seconds': 16.95771662495099}},
     'network_phases': {'direct_ipv4_http11': {'duration_seconds': 32.703545124968514,
                                               'observed_seconds': 32.71424008300528,
                                               'state': 'complete'},
                        'direct_ipv4_preflight': {'duration_seconds': 1.0368483329657465,
                                                  'observed_seconds': 1.0499527079518884,
                                                  'state': 'complete'},
                        'direct_ipv4_preflight_1': {'duration_seconds': 1.0368483329657465,
                                                    'observed_seconds': 1.0476399578619748,
                                                    'state': 'complete'},
                        'direct_ipv4_preflight_1_dns_tcp': {'duration_seconds': 0.17838599998503923,
                                                            'observed_seconds': 0.1908271659631282,
                                                            'state': 'complete'},
                        'direct_ipv4_preflight_1_tls': {'duration_seconds': 0.18564320798031986,
                                                        'observed_seconds': 0.37862212490290403,
                                                        'state': 'complete'},
                        'receive_headers': {'duration_seconds': 27.709287750069052,
                                            'observed_seconds': 32.70993712497875,
                                            'state': 'complete'},
                        'send_body': {'duration_seconds': 0.7065159999765456,
                                      'observed_seconds': 4.998077082913369,
                                      'state': 'complete'},
                        'send_headers': {'duration_seconds': 0.0002656669821590185,
                                         'observed_seconds': 4.29113220795989,
                                         'state': 'complete'}},
     'first_phase_seconds': {'http_headers': 32.71566304098815,
                             'http_phase': 0.1908271659631282,
                             'request_dispatched': 4.112716665957123,
                             'sse_event': 33.72797837504186,
                             'transport_failure': 213.7336898748763}})

SCOPES[(2357, 20)] = replace(SCOPES[(2357, 19)], wave=20, cooldown_field="not_before_unix",
                              cooldown_unix=1795726840.8317, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(2357, 19)].archive,
    Review(_2357_WAVE19, _2357_WAVE19_REQUEST, "b7cb05a627f84af6bcf8315e3fc0a2c7",
        "2cf1546c0c98ec26fb2659caab97ecf2d206e67ee2280b4c3435ba0acf0634bb",
        "be2f33513a0d8c2ee018168aacac302028c6449131d96ec81e1b08bf747d4867",
        "41ffc8a8d97689087fe32b7daf0b4f8e879f343b19863c65a32c1f58dcc54bf1",
        "476f22b37894c96ec270f9551c09a36e39084ec7308287ca4132089426da1f64", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'event_type': 'response.output_item.added',
         'sse_event_count': 25,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '405c046b-b04d-4c0a-ba57-6ee85ae7e3d9',
         'response_id': 'resp_0f06b0c9194e9ee9016b089d20b3f887d1bd8ecb7647ea8473',
         'conflicting_response_id': 'resp_405c046bb04d4c0aba576ee85ae7e3d9',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 203.49816245795228,
         'last_sse_activity_seconds': 203.49816245795228,
         'elapsed_seconds': 211.15037337481044,
         'sse_milestones': {'response.created': {'elapsed_seconds': 31.156765665858984, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 31.158781749894843,
                                                     'interval_seconds': 0.002016084035858512},
                            'response.output_item.added': {'elapsed_seconds': 31.161257416009903,
                                                           'interval_seconds': 0.0024756661150604486},
                            'response.output_item.done': {'elapsed_seconds': 46.374838665826246,
                                                          'interval_seconds': 15.213581249816343}},
         'network_phases': {'direct_ipv4_http11': {'duration_seconds': 31.149189125047997,
                                                   'observed_seconds': 31.152318165870383,
                                                   'state': 'complete'},
                            'direct_ipv4_preflight': {'duration_seconds': 1.427600583061576,
                                                      'observed_seconds': 1.4329195828177035,
                                                      'state': 'complete'},
                            'direct_ipv4_preflight_1': {'duration_seconds': 1.427600583061576,
                                                        'observed_seconds': 1.4308044998906553,
                                                        'state': 'complete'},
                            'direct_ipv4_preflight_1_dns_tcp': {'duration_seconds': 0.4483224591240287,
                                                                'observed_seconds': 0.4517007078975439,
                                                                'state': 'complete'},
                            'direct_ipv4_preflight_1_tls': {'duration_seconds': 0.23820479190908372,
                                                            'observed_seconds': 0.6919662079308182,
                                                            'state': 'complete'},
                            'receive_headers': {'duration_seconds': 25.120877207955346,
                                                'observed_seconds': 31.146710332948714,
                                                'state': 'complete'},
                            'send_body': {'duration_seconds': 1.0123755419626832,
                                          'observed_seconds': 6.023157499963418,
                                          'state': 'complete'},
                            'send_headers': {'duration_seconds': 0.00016820896416902542,
                                             'observed_seconds': 5.010503249941394,
                                             'state': 'complete'}},
         'first_phase_seconds': {'http_headers': 31.15373837482184,
                                 'http_phase': 0.4517007078975439,
                                 'request_dispatched': 5.008145582862198,
                                 'sse_event': 31.156765665858984,
                                 'transport_failure': 211.15037337481044}},
        earlier=(_2357_WAVE19_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "7e06679c9b8134c9c2fbb7f867356c23b20dffd9410091856e926e00cda24068",
            "attempt_00.replay.terminal.json": "05db658e51edfc666fa5b9fda2aa5638a1c554258cd51e712719bb1b8d5a4654",
            "attempt_01.replay_link.json": "ac1e4973b9abf9d960d3489d3008c99ccbe8bf3b23ce647a99587a912b86f446",
        }),
))


_2357_WAVE20 = retry.bounded.RunIdentity("-2357-relay-retry-20",
    "d84128e0f85eb584c4defd92003499f3b9a33f482c6d409548f363fd0cf3204c",
    "ec1fe4c70edcd97b28ae8118e0d43cffb0dd504b7eb3699b08de629139079329")
_2357_WAVE20_REQUEST = "b6221350e82464e0f7cfcc8f4a1e3055b63d5c40fda361f075ffc970c6cbb150"
_2357_WAVE20_FIRST = Review(_2357_WAVE20, _2357_WAVE20_REQUEST, "d3d52223789440a9bfa650ce7df19d34",
    "47d066f6328e78705f3e7e6e1be53de19c26829301fcc3d8c2410a1f4939daba",
    "f616149eef125ab0a90943a637cbe77ec386f74be2863ba9a32a4e853908b30a",
    "574059ad3a5a86f12d6b906bf9d80466db808f95fc80e44dadc7052995ca8b63",
    "4c7142cbbb885d06bc0f7c2a0cbdf88104a549eb7c8a47bf45f24f41a2d9e5d9", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'event_type': 'response.output_item.added',
     'sse_event_count': 41,
     'model': 'gpt-6.2-astra',
     'provider_request_id': 'e3e8c59c-1442-4a69-bee8-423aa55c4afd',
     'response_id': 'resp_0956a27db9eb39f9016b089f8c8d9c87d29089075e43208152',
     'conflicting_response_id': 'resp_e3e8c59c14424a69bee8423aa55c4afd',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 196.31199462502263,
     'last_sse_activity_seconds': 196.31199462502263,
     'elapsed_seconds': 199.1518699170556,
     'sse_milestones': {'response.created': {'elapsed_seconds': 19.053688500076532, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 19.054406125098467,
                                                 'interval_seconds': 0.0007176250219345093},
                        'response.output_item.added': {'elapsed_seconds': 19.055042500142008,
                                                       'interval_seconds': 0.000636375043541193},
                        'response.output_item.done': {'elapsed_seconds': 28.42000629194081,
                                                      'interval_seconds': 9.3649637917988}},
     'network_phases': {'direct_ipv4_http11': {'duration_seconds': 19.005770791089162,
                                               'observed_seconds': 19.01555679203011,
                                               'state': 'complete'},
                        'direct_ipv4_preflight': {'duration_seconds': 1.053683916805312,
                                                  'observed_seconds': 1.0650741669815034,
                                                  'state': 'complete'},
                        'direct_ipv4_preflight_1': {'duration_seconds': 1.053683916805312,
                                                    'observed_seconds': 1.0635573340114206,
                                                    'state': 'complete'},
                        'direct_ipv4_preflight_1_dns_tcp': {'duration_seconds': 0.19202670804224908,
                                                            'observed_seconds': 0.20348508399911225,
                                                            'state': 'complete'},
                        'direct_ipv4_preflight_1_tls': {'duration_seconds': 0.189551749965176,
                                                        'observed_seconds': 0.39525216701440513,
                                                        'state': 'complete'},
                        'receive_headers': {'duration_seconds': 14.424594874959439,
                                            'observed_seconds': 19.01263429201208,
                                            'state': 'complete'},
                        'send_body': {'duration_seconds': 0.7280653328634799,
                                      'observed_seconds': 4.5856094169430435,
                                      'state': 'complete'},
                        'send_headers': {'duration_seconds': 0.0002262911293655634,
                                         'observed_seconds': 3.857118375133723,
                                         'state': 'complete'}},
     'first_phase_seconds': {'http_headers': 19.01673616701737,
                             'http_phase': 0.20348508399911225,
                             'request_dispatched': 3.735099625075236,
                             'sse_event': 19.053688500076532,
                             'transport_failure': 199.1518699170556}})

SCOPES[(2357, 21)] = replace(SCOPES[(2357, 20)], wave=21, cooldown_field="not_before_unix",
                              cooldown_unix=1795727871.4221308, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(2357, 20)].archive,
    Review(_2357_WAVE20, _2357_WAVE20_REQUEST, "5732ea68547d4b0798f601e648609dee",
        "3f7052fcba8ac14eb3ee2617a57306a3749c7d0054f8bb313654b9396e43b0fc",
        "a20d3798a77d57e456e58f47d73894fc8216a0c4d1d5d3f4f5bccc08abed61d8",
        "ba5175cd5179686b82d0e387ed9ad5a902588f28c31d6c87683e13e2b72b6806",
        "efb7fbb10c618c6ef56e5d98a523290cf3c3d92fcb6e8187c4a23cb72c28b071", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'event_type': 'response.output_item.added',
         'sse_event_count': 41,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '158e8426-960f-46a5-a130-2e8888a52cc3',
         'response_id': 'resp_0afcc4244e6426d7016b08a131d4a087d29df4d0ef07a6d6db',
         'conflicting_response_id': 'resp_158e8426960f46a5a1302e8888a52cc3',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 194.24631462502293,
         'last_sse_activity_seconds': 194.24631462502293,
         'elapsed_seconds': 196.9795132910367,
         'sse_milestones': {'response.created': {'elapsed_seconds': 17.0129694158677, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 17.01522958301939,
                                                     'interval_seconds': 0.0022601671516895294},
                            'response.output_item.added': {'elapsed_seconds': 17.017461540875956,
                                                           'interval_seconds': 0.002231957856565714},
                            'response.output_item.done': {'elapsed_seconds': 26.367308540968224,
                                                          'interval_seconds': 9.349847000092268}},
         'network_phases': {'direct_ipv4_http11': {'duration_seconds': 17.00531958299689,
                                                   'observed_seconds': 17.008419207995757,
                                                   'state': 'complete'},
                            'direct_ipv4_preflight': {'duration_seconds': 0.7112214169465005,
                                                      'observed_seconds': 0.7159068749751896,
                                                      'state': 'complete'},
                            'direct_ipv4_preflight_1': {'duration_seconds': 0.7112214169465005,
                                                        'observed_seconds': 0.7143886659760028,
                                                        'state': 'complete'},
                            'direct_ipv4_preflight_1_dns_tcp': {'duration_seconds': 0.23093754099681973,
                                                                'observed_seconds': 0.23430004087276757,
                                                                'state': 'complete'},
                            'direct_ipv4_preflight_1_tls': {'duration_seconds': 0.2309152497909963,
                                                            'observed_seconds': 0.4672612079884857,
                                                            'state': 'complete'},
                            'receive_headers': {'duration_seconds': 12.65418216586113,
                                                'observed_seconds': 17.002798915840685,
                                                'state': 'complete'},
                            'send_body': {'duration_seconds': 0.8904972081072628,
                                          'observed_seconds': 4.34403554094024,
                                          'state': 'complete'},
                            'send_headers': {'duration_seconds': 0.00018158298917114735,
                                             'observed_seconds': 3.453265040880069,
                                             'state': 'complete'}},
         'first_phase_seconds': {'http_headers': 17.009910540888086,
                                 'http_phase': 0.23430004087276757,
                                 'request_dispatched': 3.450836875010282,
                                 'sse_event': 17.0129694158677,
                                 'transport_failure': 196.9795132910367}},
        earlier=(_2357_WAVE20_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "ab35df7db5f4759938f49eecf09e0e457a22ff221cc150f9721f7d5880acb09a",
            "attempt_00.replay.terminal.json": "3a540a024b85272599c9f594aa0e98d2e157ae5418f330b9625b7fdd4fc04bbf",
            "attempt_01.replay_link.json": "251ed732388bd0b0aa19e10e0d3333fe604da368fe8aef812364fa63a74618a5",
        }),
))


_2357_WAVE21 = retry.bounded.RunIdentity("-2357-relay-retry-21",
    "2de8b2c7ea584ca8fa62262906212ea44eba8e8fb3f447b8e1fae7d12528fc98",
    "b71eb34048b44fc35ba4b44cdb72e76eed898e8b19ce67aa1988f2d48695e7b5")
_2357_WAVE21_REQUEST = "eb51a587aff6f4e7a4761ea218ed68a5163e30dfc140311fc47fa07226f979db"
_2357_WAVE21_FIRST = Review(_2357_WAVE21, _2357_WAVE21_REQUEST, "61ca8f0d3b024d18a30097d130d3dd52",
    "9d6722586a15fe95202e0e28469901cc0c5c0f9a29b59d97940139a2c13889fe",
    "33d79ff4943d12c2df836c22e89e7c4357066465062021a1a228aa308e036208",
    "29bd0e510d1b94e7a444b3438f19e3b687ca21d2705481ee887dfa9249c0b43e",
    "bcc277769a51b64b30ae57484a583d7346f328bd59e5060a633fb9d5465189b6", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'event_type': 'response.output_item.added',
     'sse_event_count': 25,
     'model': 'gpt-6.2-astra',
     'provider_request_id': '97d81e06-776a-48ab-8275-973cbb87a8de',
     'response_id': 'resp_0a54f9964cb2421f016b08a3398f4887d0932085a77a13e5cc',
     'conflicting_response_id': 'resp_97d81e06776a48ab8275973cbb87a8de',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 203.82668470800854,
     'last_sse_activity_seconds': 203.82668470800854,
     'elapsed_seconds': 213.16435074992478,
     'sse_milestones': {'response.created': {'elapsed_seconds': 33.03505620802753, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 33.035868332954124,
                                                 'interval_seconds': 0.00081212492659688},
                        'response.output_item.added': {'elapsed_seconds': 33.03668929100968,
                                                       'interval_seconds': 0.0008209580555558205},
                        'response.output_item.done': {'elapsed_seconds': 48.336624457966536,
                                                      'interval_seconds': 15.299935166956857}},
     'network_phases': {'direct_ipv4_http11': {'duration_seconds': 31.904800791991875,
                                               'observed_seconds': 31.91462470800616,
                                               'state': 'complete'},
                        'direct_ipv4_preflight': {'duration_seconds': 0.7694391668774188,
                                                  'observed_seconds': 0.7819424159824848,
                                                  'state': 'complete'},
                        'direct_ipv4_preflight_1': {'duration_seconds': 0.7694391668774188,
                                                    'observed_seconds': 0.7793557909317315,
                                                    'state': 'complete'},
                        'direct_ipv4_preflight_1_dns_tcp': {'duration_seconds': 0.1915207919664681,
                                                            'observed_seconds': 0.20343387499451637,
                                                            'state': 'complete'},
                        'direct_ipv4_preflight_1_tls': {'duration_seconds': 0.19694904098287225,
                                                        'observed_seconds': 0.4022008329629898,
                                                        'state': 'complete'},
                        'receive_headers': {'duration_seconds': 27.838718084152788,
                                            'observed_seconds': 31.90957450005226,
                                            'state': 'complete'},
                        'send_body': {'duration_seconds': 0.7586000831797719,
                                      'observed_seconds': 4.06822441611439,
                                      'state': 'complete'},
                        'send_headers': {'duration_seconds': 0.00024458300322294235,
                                         'observed_seconds': 3.3091921659652144,
                                         'state': 'complete'}},
     'first_phase_seconds': {'http_headers': 31.915515583008528,
                             'http_phase': 0.20343387499451637,
                             'request_dispatched': 3.1556710829026997,
                             'sse_event': 33.03505620802753,
                             'transport_failure': 213.16435074992478}})

SCOPES[(2357, 22)] = replace(SCOPES[(2357, 21)], wave=22, cooldown_field="not_before_unix",
                              cooldown_unix=1795728813.6545773, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(2357, 21)].archive,
    Review(_2357_WAVE21, _2357_WAVE21_REQUEST, "8c97ccc80ce94f7cba136bfc8af99824",
        "2887a6a6ecb9101d65770ee8abf0de4da9167b3b90afb6105667d13de873edaf",
        "3eabbf13228e33bde8eb4deb1ed0aef9f5fcd9b85a87d8c0b18898454355f34a",
        "ff4ad99d1554d7c7cbf9392cb99f630b2fdfa0e7411ee5cbd09a5c8542931bc1",
        "ff0d25471d74df99eef924cb392c7fed6aefc7048d13ec69f7ef915f71beddde", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'event_type': 'response.output_item.added',
         'sse_event_count': 25,
         'model': 'gpt-6.2-astra',
         'provider_request_id': 'a4f2e440-a571-42d9-9528-f2e89d5d27d8',
         'response_id': 'resp_0f68cf21289a73f2016b08a4d3492487d0983c58a9d4449ff9',
         'conflicting_response_id': 'resp_a4f2e440a57142d99528f2e89d5d27d8',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 200.32678483310156,
         'last_sse_activity_seconds': 200.32678483310156,
         'elapsed_seconds': 207.71423870814033,
         'sse_milestones': {'response.created': {'elapsed_seconds': 27.688676041085273, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 27.691013916162774,
                                                     'interval_seconds': 0.0023378750775009394},
                            'response.output_item.added': {'elapsed_seconds': 27.693921083118767,
                                                           'interval_seconds': 0.0029071669559925795},
                            'response.output_item.done': {'elapsed_seconds': 41.860319041181356,
                                                          'interval_seconds': 14.16639795806259}},
         'network_phases': {'direct_ipv4_http11': {'duration_seconds': 27.680657875025645,
                                                   'observed_seconds': 27.684117125114426,
                                                   'state': 'complete'},
                            'direct_ipv4_preflight': {'duration_seconds': 0.758600499946624,
                                                      'observed_seconds': 0.764687665970996,
                                                      'state': 'complete'},
                            'direct_ipv4_preflight_1': {'duration_seconds': 0.758600499946624,
                                                        'observed_seconds': 0.7622066251933575,
                                                        'state': 'complete'},
                            'direct_ipv4_preflight_1_dns_tcp': {'duration_seconds': 0.1926773339509964,
                                                                'observed_seconds': 0.1965194579679519,
                                                                'state': 'complete'},
                            'direct_ipv4_preflight_1_tls': {'duration_seconds': 0.1985441658180207,
                                                            'observed_seconds': 0.39667525002732873,
                                                            'state': 'complete'},
                            'receive_headers': {'duration_seconds': 23.544121667044237,
                                                'observed_seconds': 27.67908920813352,
                                                'state': 'complete'},
                            'send_body': {'duration_seconds': 0.7592190410941839,
                                          'observed_seconds': 4.1326604161877185,
                                          'state': 'complete'},
                            'send_headers': {'duration_seconds': 0.00014233309775590897,
                                             'observed_seconds': 3.3731649580877274,
                                             'state': 'complete'}},
         'first_phase_seconds': {'http_headers': 27.68557966616936,
                                 'http_phase': 0.1965194579679519,
                                 'request_dispatched': 3.37110049999319,
                                 'sse_event': 27.688676041085273,
                                 'transport_failure': 207.71423870814033}},
        earlier=(_2357_WAVE21_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "8957527ada6db3bf72afb1dff7cc16832c50b04a924d530bfbf449f6fad0a558",
            "attempt_00.replay.terminal.json": "595698cead04d8cc320ab403e6c05494150461b64d5002b26a4e14b828946ed6",
            "attempt_01.replay_link.json": "c2cebcf186572b0cb5bda9ccb60e29a6c5ee0f1cfa60c3f7a2594d41cd30387e",
        }),
))


_2357_WAVE22 = retry.bounded.RunIdentity("-2357-relay-retry-22",
    "2f5e7a5f9af4a903b1401d4f061b51b9a6fd9dfad2c0474754a0134d4648c035",
    "bb76dddf00fd94f1564913be2e5f3ab0571045ecba1b24128a0320b2e233b3ea")
_2357_WAVE22_REQUEST = "53fb823c8dd341349d631495e056b5740f72fa7caa05e16bb424bf0221091ea2"
_2357_WAVE22_FIRST = Review(_2357_WAVE22, _2357_WAVE22_REQUEST, "7497055372d545b4baeb0c3ee3d6705d",
    "dc8b1a4270af653c68717b6bdb4a07558d1f8759c6e42a39096c2d38bf7a5331",
    "aff09024a572ab3684742d6fe14cd0565b5a79be44c9a46c3fd03f6cc817e0ca",
    "e3a394aa2503e3510b084d9c0a7a0135d92924f3fb0de92ebd929c10dc62b777",
    "ecbbb9bae8d98069672be6ec95f1a0d1e203887416e15e0916f3b444383e66ce", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'event_type': 'response.output_item.added',
     'sse_event_count': 41,
     'model': 'gpt-6.2-astra',
     'provider_request_id': '076a5c3d-6141-45b4-bf49-fcc1e239c4b8',
     'response_id': 'resp_0d7f27ac87a6bf55016b08a7073a0887d2850b06941dc9d1a8',
     'conflicting_response_id': 'resp_076a5c3d614145b4bf49fcc1e239c4b8',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 194.97610895894468,
     'last_sse_activity_seconds': 194.97610895894468,
     'elapsed_seconds': 197.9509128748905,
     'sse_milestones': {'response.created': {'elapsed_seconds': 17.812170499935746, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 17.81302125006914,
                                                 'interval_seconds': 0.000850750133395195},
                        'response.output_item.added': {'elapsed_seconds': 17.81368820904754,
                                                       'interval_seconds': 0.0006669589783996344},
                        'response.output_item.done': {'elapsed_seconds': 27.12607733393088,
                                                      'interval_seconds': 9.312389124883339}}})

SCOPES[(2357, 23)] = replace(SCOPES[(2357, 22)], wave=23, cooldown_field="not_before_unix",
                              cooldown_unix=1795729775.532721, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(2357, 22)].archive,
    Review(_2357_WAVE22, _2357_WAVE22_REQUEST, "165867cccd6b46678fb7d6238adfb53e",
        "b1b55820e277d79883c3bec7ae72d2496f9e8585db384ef65c572d958dbdb6db",
        "48b248ec4e759a6ca797e3369b7bdc6d8861a4f872d7bc50db4720666328a2da",
        "952bc63f4deacdfaad9f7f310ed2c593020ea10def642325ce7477c7100ed486",
        "ff87e785259deda019b25d2aac70bbd26ca31e4e0609f60914a652474e502036", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'event_type': 'response.output_item.added',
         'sse_event_count': 25,
         'model': 'gpt-6.2-astra',
         'provider_request_id': 'd3397732-79e5-45fd-97db-e297be7c7a65',
         'response_id': 'resp_057dfee8e10ebbec016b08a895091087d09081252127029931',
         'conflicting_response_id': 'resp_d339773279e545fd97dbe297be7c7a65',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 207.89976579113863,
         'last_sse_activity_seconds': 207.89976579113863,
         'elapsed_seconds': 210.13794500008225,
         'sse_milestones': {'response.created': {'elapsed_seconds': 30.186397791141644, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 30.18774054106325,
                                                     'interval_seconds': 0.001342749921604991},
                            'response.output_item.added': {'elapsed_seconds': 30.189011791022494,
                                                           'interval_seconds': 0.0012712499592453241},
                            'response.output_item.done': {'elapsed_seconds': 46.978341083042324,
                                                          'interval_seconds': 16.78932929201983}}},
        earlier=(_2357_WAVE22_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "1ba6833d2bcf1f72bf179eafb686ccfecec3e870c4d760402af3a44c5c070252",
            "attempt_00.replay.terminal.json": "14d1db3b694faf7bf65e3d0a112524ea8933fc543d01f2f8df1d92ed6765e3a8",
            "attempt_01.replay_link.json": "8e7bf861d31f52666f857cc0ede7de7765d24484449d81d9837a343053951648",
        }),
))


_2357_WAVE23 = retry.bounded.RunIdentity("-2357-relay-retry-23",
    "1d4544da7e7f97d4a1073b95602dd8cc0552092d8a0357d74e0793440ec2a9b5",
    "8f58632ec77f732357a0b7e4e9f40b5a6b068ae4aeea830a030ffa37e97fdde7")
_2357_WAVE23_REQUEST = "16907a55840c66c655bc7a166bd050d24d463a9e9356109d86ea412c8ce872d8"
_2357_WAVE23_FIRST = Review(_2357_WAVE23, _2357_WAVE23_REQUEST, "c08afd5215ad43f7bea313fd70b0b32c",
    "3c043793280766350efc5c7070977f7e09b5abd9e4fd99de45eec1e00c1f5385",
    "c6f16be59f1934086749edf118ee3a03f122daaf5b3c9b4619d03de930d473f9",
    "e8cc17354c9aa2ee79a3ad97e62562718f770815388dbc0e77d99d0a619803a9",
    "72a5e9b2fb8f0613e5e1f2c6658a1ba66ec35776ec3c021e4c5fb22fe36ebf01", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'event_type': 'response.output_item.added',
     'sse_event_count': 25,
     'model': 'gpt-6.2-astra',
     'provider_request_id': '31943418-d5f9-46f7-a324-ac6812206747',
     'response_id': 'resp_0116c64b1c22ec34016b08ab05997c87d0b44593da98f22f56',
     'conflicting_response_id': 'resp_31943418d5f946f7a324ac6812206747',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 207.97548883291893,
     'last_sse_activity_seconds': 207.97548883291893,
     'elapsed_seconds': 215.05301429098472,
     'sse_milestones': {'response.created': {'elapsed_seconds': 34.900967041030526, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 34.90213208296336,
                                                 'interval_seconds': 0.001165041932836175},
                        'response.output_item.added': {'elapsed_seconds': 34.90302037494257,
                                                       'interval_seconds': 0.0008882919792085886},
                        'response.output_item.done': {'elapsed_seconds': 50.75682487501763,
                                                      'interval_seconds': 15.853804500075057}}})

SCOPES[(2357, 24)] = replace(SCOPES[(2357, 23)], wave=24, cooldown_field="not_before_unix",
                              cooldown_unix=1795730806.2878697, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(2357, 23)].archive,
    Review(_2357_WAVE23, _2357_WAVE23_REQUEST, "188beb8c58bb4483b724f55ae7b8a4a7",
        "1392f69bbdd26e22a0e5f5e0efeca59084fb87203fc8e5148fe1873bfec47ad9",
        "455a9ec44e4758ce514659790955dd60bdc8181233fae3d33f3180167075e6a1",
        "cbb859d09251fcaacff360a360b59f5b4ae25224cdf26b3a97245ba550dfbcac",
        "9fc52d20cb52972a7c9aadaae7c6ebeaadabb30f892cdc4ac5fdc54796e3fd51", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'event_type': 'response.output_item.added',
         'sse_event_count': 41,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '32f9a784-2f6c-41e7-b686-66d074cbb300',
         'response_id': 'resp_0f4dce7737962e7b016b08aca6fd5c87d2ae703cae6675280a',
         'conflicting_response_id': 'resp_32f9a7842f6c41e7b68666d074cbb300',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 195.03721862495877,
         'last_sse_activity_seconds': 195.03721862495877,
         'elapsed_seconds': 197.8289520831313,
         'sse_milestones': {'response.created': {'elapsed_seconds': 17.904734125128016, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 17.906998415943235,
                                                     'interval_seconds': 0.002264290815219283},
                            'response.output_item.added': {'elapsed_seconds': 17.90989683312364,
                                                           'interval_seconds': 0.002898417180404067},
                            'response.output_item.done': {'elapsed_seconds': 27.42339575011283,
                                                          'interval_seconds': 9.513498916989192}}},
        earlier=(_2357_WAVE23_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "1cae94b20aaba7d83ab2d86828dcd6475f0a68a6f4f2923104a09474d47a0aaf",
            "attempt_00.replay.terminal.json": "ed21087d2d4608b8bac955f78179c1ff9f3bcdae4f9220f5aab62469856b7e90",
            "attempt_01.replay_link.json": "e10421c1b8218c81c948cc8a64ab812fc3c8f1b1e349abc726fb4e2a2ca95605",
        }),
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

SCOPES[(157, 6)] = replace(SCOPES[(157, 5)], wave=6, cooldown_unix=1795686851.0807147, archive=(
    *SCOPES[(157, 5)].archive,
    Review(retry.bounded.RunIdentity("-157-relay-retry-05",
        "360ce022071f58f7b45c80e6554a69bb4433e58cea803bd1f03efafe5e28de50",
        "c78dc92578f193670200e5abda1d4f76357b29e419e174916fe16e28df63a584"),
        "1fb54c4cd562bbd605c70778c6e56c6b36e4d30547af99bf82f41e21dfb38137", "f9946e76e81e432997a39dcfdcc6867a",
        "84713abfe1eca6c3035570210640f432224455e5510c36ddcf9a0c7e7327c3c0",
        "62caa4e7ba18d9a67b8aa259bb0a5100769df181c88da75e5434b7a11e4cc6c7",
        "c0e6b8d8d1aae5048b22aa62eb64ac0833981b2c49086ab33f07b312bffcf506",
        "950630f627fbbbd4a995506248fa32ea9ea1ff34f903fd3764c6a55da4ffc31f", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch",
         "sse_event_count": 41, "provider_request_id": "1310ea7a-5244-48d2-a158-fd30d4ec4db2",
         "response_id": "resp_02f8dae9ee92f0d3016b0800fb4f2487d2b2ec67c61d50888a",
         "conflicting_response_id": "resp_1310ea7a524448d2a158fd30d4ec4db2", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True}),
))

SCOPES[(157, 7)] = Scope(
    157, 4, 7, "57e28171cede075156b2a3331b8322cb994805651134644ab0cadb56e59639c7", (
        Review(retry.bounded.RunIdentity("-157-cached-r3-01",
            "f142af55eb54dc184e3cd7d19f16717712605df25044a16c19918a62884ab776",
            "17e8e477bba8760b46d6977d279573e234b807a9fdd7c4a637be5dda12dea37a"),
            "46ebd031c878dadd645b2161415a45cdf838dafc086c7f45834eb25ac03589de", "ed2d0f8f95ef40c9ae73c25f397d5468",
            "26ed0ad0b1c1f12e9818cb0289dec1ec4c7810903a2e8fad76585c9c7fcbe193",
            "14552112cb46b531dd6f5990213ad7a9620b1ea14bc3e0c6346737dcd40d9cf0",
            "9dbd308f49082eac9bf6e240402eb9886496590fb3c809d27545c72a0bcdb9a3",
            "4bea8bef10d47c4583368202fe8ca34e1c0dffa7d34902d7ba368667a9323f92", "timeout",
            {"http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
             "event_type": "response.output_text.delta", "sse_event_count": 1264, "last_sse_seconds": 1748.327394583961,
             "provider_request_id": None, "model": "gpt-6.2-astra",
             "response_id": "resp_035fa91e2ecafa2a016b080d900bac87d1ba82be06c4badff7"}),
    ), "failed_unix", 1795692500.964523, original_start_round=3, fitted_prefix_files=(
        ("round_03/turn_00/attempt_00.terminal.json", "b9e5023a6795b1e1a8f6a39a0211250454875fc3d957fc5200169b0b37f82be5"),
        ("round_03/batch_freeze.json", "c08566979fb97938c67a50a9e66840a7bdf827e65608679ca5578ebeca60dbd4"),
        ("round_03/selection.json", "ad05e526557e8ab84d9ee0854b6b4e88301c47ef187a2c1ac76a514cec1a3071"),
        ("round_03/memory.json", "b540d8fc9d10020b5e34b2dc8eec8a68fb0d8b46f12bdb76f1f1394e9e733bb7"),
        ("round_04/source.json", "bfb9e71acf549811b50c221546a987851c06f42550df5f64838461f0defd0324"),
        ("round_04/transition.json", "9b6d125b72b9eab66943687213d9d86b934c4c50884e006931a04043b2aff37a"),
    ), required_sources=QUEUE_SOURCES, cached_origin=CachedOrigin(
        retry.bounded.RunIdentity("-157-relay-retry-06",
            "a135a2a3d10e80f12864be66eb5bb65046ced589f8767969b3f9cddf3f6c006e",
            "dc505def300d7dd7024819d6cb4494e0d042bc6c1fd8ea5971aea90b602e58a5"),
        "84a173798d13de2596449dd53d21052972be058f11b41ac6bdc5b2edbf2fcb36",
        "ed23e6b33ff06a331031599d8b885d8d233a396d092bc6d68c401d99152d44e0",
        "f06b8cc813aa4124c4be9fe4375ff1eba7594b4e3a7f612e26417c33c535d16c",
    ),
)

SCOPES[(157, 8)] = replace(SCOPES[(157, 7)], wave=8, cooldown_field="not_before_unix", cooldown_unix=1795696600.5165217,
                             original_start_round=None, fitted_prefix_files=(), boundary_physical_attempts=2,
                             required_sources=SCOPES[(2357, 5)].required_sources, archive=(
    *SCOPES[(157, 7)].archive,
    Review(retry.bounded.RunIdentity("-157-relay-retry-07",
        "652e1945d0298f1fc99946cd58e398d15de80020cc72e32af56ce2ff3ffc9016",
        "49c86ded7d5c1f26c6d728c8e5877e10cf6866b2abf3d0cf3cda0b25adaf70ad"),
        "eefcf01bbf10877374e72f79d848310e02a9f1c14e8d17aa9c6921543e567865", "480fca7d77f74e2bbbb8e6bb0bd2429e",
        "669348c7843eb21d30691c54ab83d4f28b5b70148efddbd717fe9adb6f388035",
        "07678784ab39ef104e604566806acd5d2e187e88015637b28c7d215391c07625",
        "dde8e232534c842fb25cb2a0bcc8b5ff3951f1703fa0e96696c19e24e7c92a58",
        "462eedc9bab98efeba02074297c9a51eb32b89cb2a0b6f2373cb78dee93ca47b", "timeout",
        {"http_status": 200, "phase": "transport_failure", "error_type": "hard_wall_deadline",
         "event_type": "response.output_text.delta", "sse_event_count": 2783, "last_sse_seconds": 3561.158813457936,
         "provider_request_id": None, "model": "gpt-6.2-astra",
         "response_id": "resp_0f418a2e94c1bd0b016b0819d504a887d1af55f85e4323df53"}),
))

SCOPES[(157, 9)] = Scope(
    157, 7, 9, "a107c416d5d99771b3b14a88ee77b81c73378807da79cc849c3da05edae9e1e3", (
        Review(retry.bounded.RunIdentity("-157-relay-retry-08",
            "c522b2d38f3f07dfa2eba566dd069bf44c21885518760e5179fa6315615f4435",
            "f116399a5c511f2fdf9c064f1d052c8662d635cada4c2ff6a925d9f9703a64cb"),
            "43d657284193e17a47274bf0660ccdb31dfdb12baa4e91cb686e65ea5ceebfa0", "c72548bb89114b64b95972fc24b5910b",
            "fe18ea111eadef8a26bb9f9526fea10dd6459b3c3b4f65f18bff2975fa2df16f",
            "4db176100825a81b20a949d476a801a957fc4286385add839054fbaf32ac85c6",
            "21bcafe25908c82563fbc9a2c3fd597142c3671ffdaa9b2b1bf9585735117ad6",
            "aeab3db281169048857cc9bdb376f946b02e51bcc8a5fd8495a577ea79257f72", "timeout",
            {"http_status": 200, "phase": "transport_failure", "error_type": "hard_wall_deadline",
             "event_type": "response.output_text.delta", "sse_event_count": 5937, "last_sse_seconds": 2942.470113791991,
             "provider_request_id": None, "model": "gpt-6.2-astra",
             "response_id": "resp_0670888a12db153b016b083a3afb7487d18710c78143d1258d"}),
    ), "failed_unix", 1795704901.8307962, original_start_round=4, fitted_prefix_files=(
        ("boundary.json", "61cd70dc48adbd2958bd81844bb2e5dec91030d3ea3d55bcf6713dec7ed59e0d"),
        ("round_04/boundary_user_message.json", "819e36858efbf1dece8620035ccd13728a9f17aa6c737e71d21dda20a7366c34"),
        ("round_04/turn_00/attempt_00.terminal.json", "64073d8a4cfa0d269b9c1540ac139f678dc9fe9dd821c5c138720097a80b1704"),
        ("round_04/batch_freeze.json", "750f620462b175b0902e090621bdd69df7926ae67a8e0ddc8fb8906c9c2ed19f"),
        ("round_04/selection.json", "60e78ac73c0544366f7d777bdb66e19561f77a7eae0703f8d1e6de00c19815ca"),
        ("round_04/memory.json", "0f1f4667d916a7086a4bb413a0ad9c485cb221ae61d23251291bb81a6373d5f8"),
        ("round_05/transition.json", "624e3130925174062977c5552e68ec1df1ddbb184309eef96454b5495945e6f1"),
        ("round_05/turn_00/attempt_00.terminal.json", "f05f175e9f37b29a7c3cb5076cd75f1b356bd70e78c23bf96961d5cef49d6ceb"),
        ("round_05/batch_freeze.json", "42bfba810e4b1257cc25da446edda28d667ae3ca426c2df24e03d66ab9bee9f4"),
        ("round_05/selection.json", "ae6d9e24d794d614e9a7e517ac36be4a2369950819db11b28ef6bae89f9c1605"),
        ("round_05/memory.json", "c3d98a983778ecfabea09529fb2394e3e72758f4e731c7c484a545025d168969"),
        ("round_06/transition.json", "556978252cb8fa1533ee985a85fcd06ea1ec4ce6aa866ec3fab1df0939acc34c"),
        ("round_06/turn_00/attempt_00.terminal.json", "78aaed50b975b4bb54db16455d1b2449ca8e4bff31665faa7fc3fcaba96ce784"),
        ("round_06/batch_freeze.json", "aecc0b4addc25dd58f61b955f882fcfbcbfa6e0591b44c56d67823c5fee6f8de"),
        ("round_06/selection.json", "8633cb14dce70f801c48066a1dc8004232fe0d3f91313c91b2b66b04ab23ecf0"),
        ("round_06/memory.json", "2ddffdf429aabee1493cec68248e696aaf52578433bf61308ab2841317cdc897"),
        ("round_07/transition.json", "26cf746fe168b0942a8aeb55a407b366dba8356b15f50da4041924120307b047"),
        ("round_07/source.json", "ac42a77b7b18963db4be8ca807aa68bc04636643cdb8b0f9d0226a3f515a1aee"),
        ("round_07/fit_000/fit_audit.json", "7566158a2562efe3570d2cf43458481a606649b8d2215f3262357e0d482cc542"),
        ("round_07/fit_000/validation_metrics.json", "2a990a91faedc98ee4e7d214acbed5a6d2de75f6597e9d09565355271bc05b70"),
        ("round_07/authoritative_packet.json", "767f945a9801c53434e2dea7eb7756d5499cc6b0af3d5f77c2a633c800957793"),
    ), required_sources=SCOPES[(6101, 8)].required_sources, boundary_physical_attempts=2,
)

_157_WAVE09 = retry.bounded.RunIdentity("-157-relay-retry-09",
    "d0a49373985c1c6bd255894f7f35f4ca79c6fd59a9a6c939857f87cb788d7b79",
    "28eb169d3d948e19d52c6e2dbf951db74c525a908ce2207dc4629783cb93f287")
_157_WAVE09_REQUEST = "e59ccde4457fa92dfe70834781d7f769f6c94532db40c62b6dbd658040db2c42"
_157_WAVE09_FIRST = Review(_157_WAVE09, _157_WAVE09_REQUEST, "ee1c6e1526ee4fd5b8985667fd5a7acd",
    "1cb274a35e10ad4baa1b410d0ea5fe9e2b25add6c3784f4982b61a65add44612",
    "e9fcb62f40fc685b20fcd2635a1a1404afe126918f341effb4730f5b9608ef7a",
    "81c73c24dd749e6c148cce60f23cf1466d609619a4b289128c4c532a256e7c25",
    "8fc130fb135f72ffb26c5b71f091c6a5567c2d4a0309d38a82c67f543f665a77", "response_id_mismatch",
    {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch",
     "event_type": "response.output_item.added", "sse_event_count": 25, "last_sse_seconds": 241.4274480000604,
     "provider_request_id": "7cce46bc-8ffb-4421-9ca4-9c8990eacd97", "model": "gpt-6.2-astra",
     "response_id": "resp_019e9920d73374d1016b084c6dfb0887d09da6aedddf87c36e",
     "conflicting_response_id": "resp_7cce46bc8ffb44219ca49c8990eacd97", "conflicting_event_type": "response.failed",
     "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"})
SCOPES[(157, 10)] = replace(SCOPES[(157, 9)], wave=10, cooldown_unix=1795707648.740078,
                            original_start_round=7, original_boundary_physical_attempts=2, fitted_prefix_files=(), archive=(
    *SCOPES[(157, 9)].archive,
    Review(_157_WAVE09, _157_WAVE09_REQUEST, "a36eb5dadf3848c480a6111e30fae29a",
        "8028423b74bdc556a37ca0a48d59218faad18a5c8db697646dd89f9f2a18e822",
        "9f59cc0819cefc4ecadb30992a3ff000d83010e75c998336e1aba6eb5cb3f960",
        "75193d8180ad37624473ba3579c45b376aff7df4d830abb0c372476623199c79",
        "36071445278b32df8d46f8b772cae7c9cde591dd1e5b0f7a135d49819e7cd239", "timeout",
        {"http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
         "event_type": "response.output_item.done", "sse_event_count": 16, "last_sse_seconds": 355.43698091688566,
         "provider_request_id": "9c1bb9f3-9ba1-4dd9-b8c2-f0ab809c1a13", "model": "gpt-6.2-astra",
         "response_id": "resp_064c196a68bb4cad016b084e30589087d09f71396652606830"},
        earlier=(_157_WAVE09_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "5863950fdd116c32d8127d13f10de98debfb08094c808781f4cc200e1486a9a9",
            "attempt_00.replay.terminal.json": "d87205578d47784081525c1119ff44137b4c6e9e7290822dadcf3863c730eada",
            "attempt_01.replay_link.json": "58dfea77b6e9be2212170b4f3b9b7865c481480e594dc7623e46decfae43591a",
        }),
))


SCOPES[(157, 11)] = replace(
    SCOPES[(157, 10)], wave=11, cooldown_field="not_before_unix", cooldown_unix=1795709532.7965784,
    boundary_physical_attempts=1, archive=(
        *SCOPES[(157, 10)].archive,
        Review(retry.bounded.RunIdentity("-157-relay-retry-10",
            "da528d3431e8029204cf1e4bc9f0cb80e26b43824ca8fd2b50b91c61e21e3193",
            "779450ec6b1fa6ac85653d79887d8df64a0f7a9e25fc212ac0b2e84c6af43e8f"),
            "7a9a81fe57c993a4d8725a7f19a6e19d71688b80227af45684742361f547452a", "1403a2ad4fd640f38431fc5b41e8c63b",
            "135ac7ab824e0583c94f2b7eddee0c45e0133a96152fd772d444dd7e962ebf83",
            "75f28f0d88372241d4cfffdbb8c14c79c35192ecb44eb1c900c4881dfd04ecc1",
            "ef3d7e834b035d862625770e776f7b31d7fa7abfb7cffe2b5cd1b67b34c0de40",
            "c69ee67ffbec7fe9865edce0b086b6fb6abc810f25c5b128b55e36c960bcdd35", "connection_interrupted",
            {"http_status": None, "phase": "transport_failure", "error_type": "ReadError",
             "event_type": "ipv4_preflight_http_401", "provider_request_id": None, "model": None, "response_id": None}),
    ),
)


SCOPES[(157, 12)] = replace(
    SCOPES[(157, 11)], wave=12, cooldown_unix=1795713791.6597245,
    boundary_physical_attempts=2, original_boundary_physical_attempts=1, archive=(
        *SCOPES[(157, 11)].archive,
        Review(retry.bounded.RunIdentity("-157-relay-retry-11",
            "b3160c0d7152da8cd96c435797608ed5f4ba1599a8bbb25d9abf78e3688f7f72",
            "275c02cc9719bd58133cc4518c3872f27eb90695a6ce3d9a6949859379ae483b"),
            "fe395a7de89bfbf4eb8d3518e1a11e50ac02d0373c15a81f8e543245229e64a5", "1c721d7485b5425aa43fa84b40cb906c",
            "bceccf52e34a0d9c921ad46307ebd2414e613bbed7bdc080485ea614501930ea",
            "2cf9c48c9fc755ce6bfd996d05e15a1b9dda4fbcf7b8f8f3d0fe34436a355199",
            "6ab68e428ac4326ddaf6a9a8a73df763cffc6b7fd5395ff769968c1f4b37f717",
            "93bbdbbea43e96442938cf26d2ff1d4c0d52a17fe6624251e4b8ef24398a7267", "timeout",
            {"http_status": 200, "phase": "transport_failure", "error_type": "hard_wall_deadline",
             "event_type": "response.output_text.delta", "sse_event_count": 3544, "last_sse_seconds": 3500.5466470001265,
             "elapsed_seconds": 3600.0235151669476, "provider_request_id": None, "model": "gpt-6.2-astra",
             "response_id": "resp_07cb3bb3604ee842016b085cedae5087d2b7ede07e8464d479"}),
    ),
)


_157_WAVE12 = retry.bounded.RunIdentity("-157-relay-retry-12",
    "fbf1de0b1b25fef8003e01ee3b098554f356d150ab005f7ff065e51efb36c7c2",
    "706ed123af0f865e22ef4140fbf55df1ba79d2569e42900d6c917fd56b0f1fb6")
_157_WAVE12_REQUEST = "65e609464f666e9b1e5e498195a8435755e1f90f8e5c6fa85cefbbfaea093283"
_157_WAVE12_FIRST = Review(_157_WAVE12, _157_WAVE12_REQUEST, "8973b62b1ef34dbc92b9140dff4757cd",
    "59b123e6667bb8e011b68958b8bd47f867c9000b6e77472f4e9d42454594534b",
    "8e2a0488a295e9261e37937a87551a0e779f21f06284e97edf094a1dfedf77bc",
    "3803111604c2bdd209b2ce65188aa0a3437045ef22f36da61596854ad229889b",
    "cab864e6c29438cb2ee3c2690c3ddb5b578c1480491d462548a3b0a6ed37b5a9", "response_id_mismatch",
    {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch",
     "event_type": "response.output_item.added", "sse_event_count": 41,
     "provider_request_id": "f281eab1-e62f-4484-9d41-113ef43eb67d", "model": "gpt-6.2-astra",
     "response_id": "resp_0c7fea1a20f51769016b086dd6fb2487d281e08b6cce8d7cd6",
     "conflicting_response_id": "resp_f281eab1e62f44849d41113ef43eb67d", "conflicting_event_type": "response.failed",
     "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"})
SCOPES[(157, 13)] = replace(
    SCOPES[(157, 12)], wave=13, cooldown_field="failed_unix", cooldown_unix=1795716989.7967703,
    original_boundary_physical_attempts=2, archive=(
        *SCOPES[(157, 12)].archive,
        Review(_157_WAVE12, _157_WAVE12_REQUEST, "3ea1f797475745898d5e73835cdf80ab",
            "ada5ab05c0e72c67ad17410aac2c4c92a77fd317d8fed908453d108f49cde755",
            "74835ac5a00770ee44d84e33440c5c6bf81dc24304193097a6de580c0d9edb90",
            "e07fe0e864788b7c4608b565757dadc78e94f475f7ccd3e2d3b5dd1e54307807",
            "687b243fe19d362494bd44777f70f2a4fdc58b30fdfac13701e753361bb2d4ba", "timeout",
            {"http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
             "event_type": "response.output_item.done", "sse_event_count": 24,
             "last_sse_seconds": 698.6975491670892, "last_sse_activity_seconds": 698.6975491670892,
             "elapsed_seconds": 1598.7060428331606, "provider_request_id": "a1c30542-f1fc-45a6-8cae-bf0e234b81ba",
             "model": "gpt-6.2-astra", "response_id": "resp_090fb9d352760183016b086f72a42487d2808f1febffe6f78c"},
            earlier=(_157_WAVE12_FIRST,), replay_files={
                "attempt_00.replay_authorization.json": "bbe0bcf9ca35132f2caa1af53b00212771775b6686833ab855f60a9a1e20e5ab",
                "attempt_00.replay.terminal.json": "8ca191ce30982e122f4c55c903dd40f7574d3c98c9d9dff03f35ca063e7177eb",
                "attempt_01.replay_link.json": "1b9229d17f2971f125b0063a2401d60700412466fac065ccf2300dafd1d8b5c6",
            }),
    ),
)


_157_WAVE13 = retry.bounded.RunIdentity("-157-relay-retry-13",
    "1bd3a790d40f1d8b382c4d9544e78ae5c4046f287dffff9ea897754328335297",
    "de7355f6653bfbdbb504cc3d96fb641f60082f5059a6094ba1055f3538b7c9f1")
_157_WAVE13_REQUEST = "ff136e4d10960d07d35821f64ef2fefe0a13779f5e8879ebbe5bb26a6e74ee9c"
_157_WAVE13_FIRST = Review(_157_WAVE13, _157_WAVE13_REQUEST, "2dec2b706caa412cb1251aff0a2b0b42",
    "80ef8ee32467461876c7f0dc29cb9d5e25605d3c4d64d0e9a44673d38294d2aa",
    "8c21b0c5c3737de3702ff38c83b1b9fb8e74c49e34c8de62d6cad6ab1ad26ddf",
    "f371380bea7f1aee557960d2f95ebd8a9f3cf1dc667c3ecaf1d169d185fab537",
    "823e3443403ed3077cfb2cfb53f065184460fdc7a90f0edcca7162fa42dcf8d6", "response_id_mismatch",
    {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 25,
     "provider_request_id": "bc3f8a19-4457-4d88-9c96-c9f442e11240", "model": "gpt-6.2-astra",
     "response_id": "resp_069a70ca424c912a016b087a7909a487d1a39714038564ae74",
     "conflicting_response_id": "resp_bc3f8a1944574d889c96c9f442e11240", "conflicting_event_type": "response.failed",
     "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"})
SCOPES[(157, 14)] = replace(SCOPES[(157, 13)], wave=14, cooldown_field="not_before_unix",
                              cooldown_unix=1795718382.9140031, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(157, 13)].archive,
    Review(_157_WAVE13, _157_WAVE13_REQUEST, "74e79fbb04454079a41244ae4ee35c53",
        "ad4fc6b9129272d2e5133885b4d51de64e3f83c31d3cbdbd270e5169f2c151bd",
        "16da9655cf1985c1d2c98698df2b5d5aaaa323e80b50e02c56f41b7bdcb33dd3",
        "9fb2c7073ef9dc8adaf022561df003cfed381025a2c698c147b68d6ffc4905bb",
        "8ad5796e24d840b54fdfb10f4d1ec200a7caa4a7cebba49006a97982b5c0b0f4", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 41,
         "provider_request_id": "5f2aa5a4-9d8a-423b-b466-bc86827d489b", "model": "gpt-6.2-astra",
         "response_id": "resp_0995a2322c47b270016b087c1c135087d28df33a3dc759e82b",
         "conflicting_response_id": "resp_5f2aa5a49d8a423bb466bc86827d489b", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"},
        earlier=(_157_WAVE13_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "71386e5b9d504db42ae5597e75b8ee747a0cb3215376977cf7ec77aba3e74a8d",
            "attempt_00.replay.terminal.json": "5a64314230eed776c69072a22a345b80bd696bcbb809dd8f78a59219486092fd",
            "attempt_01.replay_link.json": "09fa2525b10dfb27e2a0cf712dae256571fb33389c7c7a5d405b8d6649adf4ae",
        }),
))


_157_WAVE14 = retry.bounded.RunIdentity("-157-relay-retry-14",
    "0f15c53f5fb8be71c7a92536de9afb1e5cbf90775550c876fdd93b3d216716d7",
    "4841cbe651fd6bb8299eed6c02b6c2007fb4cc9c099e6b01b96e15b1951b2db2")
_157_WAVE14_REQUEST = "24c686ea0fdf4d99a23996fedd965dcc5f8f6e9120ec619c9e27a2107241f053"
_157_WAVE14_FIRST = Review(_157_WAVE14, _157_WAVE14_REQUEST, "2b15775e3fbf432bac822bf518b3f6fd",
    "f11b0e51e5ef96cc00907c98f3842ff65bd08191f54cb9538f06aa17d956dca7",
    "2af5a898d006fbafcf2372e4ad8cf10249c0005fa980b4fffe2cb80c1ca9ad9d",
    "c973eff85c8b6fe49414450082face5f86c9cf15db46f0aa357287d3705cc1eb",
    "6a00a680a2fd820367e9c932817c710bb06772c469696ef9e77136607a974e33", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'sse_event_count': 41,
     'model': 'gpt-6.2-astra',
     'provider_request_id': 'b24f30b4-784a-497b-bf0c-a4673d093090',
     'response_id': 'resp_02328ab21578efd7016b087e6d4fdc87d2bb8494944645337f',
     'conflicting_response_id': 'resp_b24f30b4784a497bbf0ca4673d093090',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 195.9210442909971,
     'elapsed_seconds': 198.9187320829369,
     'sse_milestones': {'response.created': {'elapsed_seconds': 18.82977058296092, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 18.830546875018626,
                                                 'interval_seconds': 0.0007762920577079058},
                        'response.output_item.added': {'elapsed_seconds': 18.83104758313857,
                                                       'interval_seconds': 0.000500708119943738},
                        'response.output_item.done': {'elapsed_seconds': 28.38752175006084,
                                                      'interval_seconds': 9.556474166922271}}})

SCOPES[(157, 15)] = replace(SCOPES[(157, 14)], wave=15, cooldown_field="not_before_unix",
                              cooldown_unix=1795719366.705628,
                              original_boundary_physical_attempts=2, archive=(
    *SCOPES[(157, 14)].archive,
    Review(_157_WAVE14, _157_WAVE14_REQUEST, "140178c833074f75986a63a927d8ca00",
        "7af1838da572e2ce29e3fb8a0d512969560de0691f729fda00febf11720a9f0d",
        "37da7320abe13cd61b2fee37b218b0a6804fdcf58355b56999f897400283f370",
        "96240c037c9f6683c7a041ca92cce890dd30b1629049101f6b30b442feb02adb",
        "7859aae069be83625ecddc707d2bb28a7fb87c0063cede76f16185ab91adc3f2", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'sse_event_count': 41,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '20ce98dc-d807-4b77-a529-9fd5cf12d408',
         'response_id': 'resp_00cbeebad1f62198016b087ff7e7f887d2a4d4e69b792c7d34',
         'conflicting_response_id': 'resp_20ce98dcd8074b77a5299fd5cf12d408',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 196.38259412511252,
         'elapsed_seconds': 198.8618995831348,
         'sse_milestones': {'response.created': {'elapsed_seconds': 18.934528667014092, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 18.93730491702445,
                                                     'interval_seconds': 0.002776250010356307},
                            'response.output_item.added': {'elapsed_seconds': 18.939459042157978,
                                                           'interval_seconds': 0.0021541251335293055},
                            'response.output_item.done': {'elapsed_seconds': 28.093643208034337,
                                                          'interval_seconds': 9.154184165876359}}},
        earlier=(_157_WAVE14_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "08f0528839a68bed3293dfd5ee8410dc17b14786890ce10a69a61252c5f6bcd7",
            "attempt_00.replay.terminal.json": "a4d027b79027202d75008b27e4d5076b8b43e120fb0e8a4bbc9adf61a969dd61",
            "attempt_01.replay_link.json": "0cfdf2f1371926546b8237e098d632858467ee60faa46ac68e194f933448a9bb",
        }),
))


_157_WAVE15 = retry.bounded.RunIdentity("-157-relay-retry-15",
    "f0ca4d3bc4342d6b4887238415123d113545734117a03c30c51482cfc3bfb301",
    "e8df9f783658106ad197dbeeb2d5a34f1d6e43296167aa558ffe9856dc53e47a")
_157_WAVE15_REQUEST = "bde99de6dc28ffb35e0d5083936a6c49d733fc0333fcb9f2d27127abe7403e2f"
_157_WAVE15_FIRST = Review(_157_WAVE15, _157_WAVE15_REQUEST, "86e32af54c0e486285bb634fb55cbef8",
    "39c86ffb72f1af94555e7fe79b856b0c67dcfc4348b87cd7ffc7169c106e9dd2",
    "673f0600aafa83a0350589bb84d9ec3e3c35f593f98b0224d4430e43de8a848a",
    "812fe89ce0153b3965857e329590ab95a05d667eec8ac22479697a91d216fbf1",
    "16412bdb0663541f3d6e26bc4cdc59bd824bc9be6392999a2dcd5e2e6921ed0d", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'sse_event_count': 41,
     'model': 'gpt-6.2-astra',
     'provider_request_id': 'c927251d-5dd7-49c7-b53f-36856389ad89',
     'response_id': 'resp_0d2cf26dfb0307f9016b0886ed085887d2a3920fd4106892e5',
     'conflicting_response_id': 'resp_c927251d5dd749c7b53f36856389ad89',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 197.81280629104003,
     'elapsed_seconds': 200.8359460409265,
     'sse_milestones': {'response.created': {'elapsed_seconds': 20.72371495794505, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 20.724735165946186,
                                                 'interval_seconds': 0.0010202080011367798},
                        'response.output_item.added': {'elapsed_seconds': 20.72522229095921,
                                                       'interval_seconds': 0.0004871250130236149},
                        'response.output_item.done': {'elapsed_seconds': 30.01478095795028,
                                                      'interval_seconds': 9.28955866699107}}})

SCOPES[(157, 16)] = replace(SCOPES[(157, 15)], wave=16, cooldown_field="not_before_unix",
                              cooldown_unix=1795721561.325949,
                              original_boundary_physical_attempts=2, archive=(
    *SCOPES[(157, 15)].archive,
    Review(_157_WAVE15, _157_WAVE15_REQUEST, "0753108b819c4adcbcb54938c6d5bdda",
        "9be554f96e916da6cb3235968b30febc51e0035b71a58b15538da30795972a61",
        "8c9494df914662c33dab8866e1750166ce3e0885baecace9bd516f2bed7660fd",
        "c4fd4fabef6dfd1c48ce3e3c796622d0cf327c98f4bcada47ba64f9258192b4e",
        "9bcc2b90c83b0c8187d7a45beeaed71ce755b610b3bca414677d58741d61d02a", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'sse_event_count': 25,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '44585c25-2dca-40d7-accc-ce2a1147bcf2',
         'response_id': 'resp_037ec881b43e4a94016b08887ce2c887d09f04149b6aa43962',
         'conflicting_response_id': 'resp_44585c252dca40d7acccce2a1147bcf2',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 210.7801252498757,
         'elapsed_seconds': 213.25315304100513,
         'sse_milestones': {'response.created': {'elapsed_seconds': 33.2462880408857, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 33.249730583047494,
                                                     'interval_seconds': 0.0034425421617925167},
                            'response.output_item.added': {'elapsed_seconds': 33.25367245799862,
                                                           'interval_seconds': 0.003941874951124191},
                            'response.output_item.done': {'elapsed_seconds': 49.5187750000041,
                                                          'interval_seconds': 16.26510254200548}}},
        earlier=(_157_WAVE15_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "66a0b7411dd935b383e0d55a3b7852c432b8dffdf7635663abf62c0d07e18a90",
            "attempt_00.replay.terminal.json": "6bbd71fc06ebcb41679a8d421c52895c2d0d6fd4caf972ce8c52fca96790a839",
            "attempt_01.replay_link.json": "16fd0eed58cd80c74f3ce6578ad23d32777d362465cf9da56e727b9587c27ac4",
        }),
))


_157_WAVE16 = retry.bounded.RunIdentity("-157-relay-retry-16",
    "60ef6daf955bb7b11b8390cdeb08fb13db769ce47176a1ca9c09f3ec2d52bad6",
    "c69d3995ca87f89c466f71d79c23300f302028446a93d1c2f6605df63d409e8c")
_157_WAVE16_REQUEST = "83794c225bc9400d78e15ddbd0a6891ee8688f270fc0ad0b6be55cf6f5862bed"
_157_WAVE16_FIRST = Review(_157_WAVE16, _157_WAVE16_REQUEST, "a428b0b0658443ef9b4d1e3de274a8ef",
    "4a359bee4ec1b1d62cac5fad95f928dd67d0349d886762a71a251e8095331ec3",
    "4062f102d1575e541bc7bdbff43615828176f868143e75b7335d495638784d32",
    "d8d89a2ad5994beae4c1ebc18a5486a8096c8953cb4ed7e57380bafe6f0b75cb",
    "2578939021b3135022710399ac179fddd975a9102c0b0c4d06bcf16934fbaed7", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'sse_event_count': 25,
     'model': 'gpt-6.2-astra',
     'provider_request_id': '6aaabdb8-88f7-42c7-a5a6-e8ca2babb74d',
     'response_id': 'resp_004b3026e48b490e016b088b03518487d1ae2547559d4afb17',
     'conflicting_response_id': 'resp_6aaabdb888f742c7a5a6e8ca2babb74d',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 207.71317308279686,
     'elapsed_seconds': 212.28305095783435,
     'sse_milestones': {'response.created': {'elapsed_seconds': 32.13775283284485, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 32.13957424997352,
                                                 'interval_seconds': 0.0018214171286672354},
                        'response.output_item.added': {'elapsed_seconds': 32.14059720793739,
                                                       'interval_seconds': 0.0010229579638689756},
                        'response.output_item.done': {'elapsed_seconds': 48.89199149981141,
                                                      'interval_seconds': 16.75139429187402}}})

SCOPES[(157, 17)] = replace(SCOPES[(157, 16)], wave=17, cooldown_field="not_before_unix",
                              cooldown_unix=1795722602.7178502, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(157, 16)].archive,
    Review(_157_WAVE16, _157_WAVE16_REQUEST, "39039b9e9017416984944f65a4eb37a8",
        "e0fbe6cab9255cf8a2afe3edab074d65398277331cef9d3f6de6da78dbb63c51",
        "5f7ad7609a3d333f8001d62d75dfb353d2bf0711e4bda8d31f63c5a6982ce917",
        "6bde11c304287cefe9b956f7deb61e6372aeaab1e4f8f70fcdcabc2273067766",
        "d3f46271df0f3e915ddab5bb8c2ef797cb5fafb1f3f10d650eb203179c6dee31", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'sse_event_count': 41,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '6358dd8f-9cf1-40a4-a7fa-fdd375c9b4a0',
         'response_id': 'resp_0b10fe3feb21d73b016b088c9905c887d1b7ed2b30e6e4abbe',
         'conflicting_response_id': 'resp_6358dd8f9cf140a4a7fafdd375c9b4a0',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 199.81623791716993,
         'elapsed_seconds': 202.5548870000057,
         'sse_milestones': {'response.created': {'elapsed_seconds': 22.720137584023178, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 22.74440075014718,
                                                     'interval_seconds': 0.024263166124001145},
                            'response.output_item.added': {'elapsed_seconds': 22.77850366709754,
                                                           'interval_seconds': 0.03410291695035994},
                            'response.output_item.done': {'elapsed_seconds': 31.997200459009036,
                                                          'interval_seconds': 9.218696791911498}}},
        earlier=(_157_WAVE16_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "e2110504777b56aebf1394ed95df2f4c5fedc8c2a157cbad1c6cba78f4d18d3c",
            "attempt_00.replay.terminal.json": "606e09d1c2170c9878111680594d91db4477657930f591898bbf0e3f5b314fe1",
            "attempt_01.replay_link.json": "4ba4adfeb0abce35c6c304c78251ab76dd23d28cc831f29f60f6358ea66ffbad",
        }),
))


_157_WAVE17 = retry.bounded.RunIdentity("-157-relay-retry-17",
    "a3eff4a581df80f6fede6fb3e4961ce775744457f01598247e9849ce3eba4ab2",
    "1d2a3b2b9c0aef5b8b704ba8fc42b395dbf510d82c4023359526a5cb79f90417")
_157_WAVE17_REQUEST = "be139e4d191fe3f8feae14eae42e4df765baf4c3d57cd33b5ed0b1f362a89006"
_157_WAVE17_FIRST = Review(_157_WAVE17, _157_WAVE17_REQUEST, "21368d1641eb45258b3c573d8e05c6c8",
    "dfd2f7d6b28c1d225e3601b81dd0c9b861b7a1dbfd60f83825ee1bf1ab7aaf5d",
    "06a08942620bd2e3e8097478bba5b326d0b7caa285b1c910310bda5e17d01e96",
    "f085c4c7f362da41c9ecc9683153707ef4a7b6ecc3f488a1d2363d2f39364378",
    "f7edbe61d8df4b43ee39b3824092acf0c412361a79fae411d45026812c9e1dfc", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'sse_event_count': 25,
     'model': 'gpt-6.2-astra',
     'provider_request_id': '09619b4b-6367-46a5-a2ab-24a1c170cfd1',
     'response_id': 'resp_0eab6d0895810bdc016b088f26d33087d2bbcf7074b8d98801',
     'conflicting_response_id': 'resp_09619b4b636746a5a2ab24a1c170cfd1',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 210.11558154202066,
     'elapsed_seconds': 211.47636504215188,
     'sse_milestones': {'response.created': {'elapsed_seconds': 31.36002600006759, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 31.36092337500304,
                                                 'interval_seconds': 0.0008973749354481697},
                        'response.output_item.added': {'elapsed_seconds': 31.361740458989516,
                                                       'interval_seconds': 0.0008170839864760637},
                        'response.output_item.done': {'elapsed_seconds': 49.12950575002469,
                                                      'interval_seconds': 17.767765291035175}}})

SCOPES[(157, 18)] = replace(SCOPES[(157, 17)], wave=18, cooldown_field="not_before_unix",
                              cooldown_unix=1795723671.023255, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(157, 17)].archive,
    Review(_157_WAVE17, _157_WAVE17_REQUEST, "9e607209d29c4dc0a5553f3f3cce9942",
        "8f3a2a85edbca768c6880ff692c3a033c7a05595f64b91299829d23093b41565",
        "7d18c1e6f509cbea40d1f7299686d9167830ee9926d1278002a63a4a9614e645",
        "8927aac5d427f40752ae4d3c50da49309df7317448d2292c8c20b92d859ae786",
        "78f37204a861717cf0dfac6027c7e43870d6acba946b0bea057b87b7039bdaa3", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'sse_event_count': 39,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '8bf07202-460b-47e7-a0e1-760d769b2c16',
         'response_id': 'resp_0ef06a18ce3cbe09016b0890cabf0c87d1b9ff4b2e5b8cef53',
         'conflicting_response_id': 'resp_8bf07202460b47e7a0e1760d769b2c16',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 199.53460570890456,
         'elapsed_seconds': 206.96631662501022,
         'sse_milestones': {'response.created': {'elapsed_seconds': 31.603634458966553, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 31.605146833928302,
                                                     'interval_seconds': 0.0015123749617487192},
                            'response.output_item.added': {'elapsed_seconds': 31.611024416983128,
                                                           'interval_seconds': 0.005877583054825664},
                            'response.output_item.done': {'elapsed_seconds': 44.093582833884284,
                                                          'interval_seconds': 12.482558416901156}}},
        earlier=(_157_WAVE17_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "094722a58836331606ed36181fd430f2cb2c573820a8c94b8ae872dc19d84686",
            "attempt_00.replay.terminal.json": "7a626bedb178f43eaf0d43eccb083f514baf8bc3ec53b11363cb3833143152fa",
            "attempt_01.replay_link.json": "8d2a0d9d90accd864ad831053362c0d2541642d2c20090b6b0ab2678bc136ca9",
        }),
))


_157_WAVE18 = retry.bounded.RunIdentity("-157-relay-retry-18",
    "e8c52f6f5e746c73b56b9e34e674f390b435146ed6f300805b47c954fae186ba",
    "ff6fe7f2950033d6cb21f34d36e6a70c5db658f2f57694bcea029db6af1fbadf")
_157_WAVE18_REQUEST = "6e0635eb183a9921de26ff86560f2ad11d0dde26b82ee5315b31e8f911d5801b"
_157_WAVE18_FIRST = Review(_157_WAVE18, _157_WAVE18_REQUEST, "dc7cfa5063e54558beb40e16fba7a0cc",
    "ae3ab8a0daa8f3849f9ebc77d761083c98566a5dcf8b66a8f930fe078ea08b7f",
    "c721ef963f44c6139f7212d423d561d3dd53bec84915cb3709103f1cdf820b64",
    "5a1c97ba41c1dd3c3c5df2a699e53c8059264e70f5b4e3bf4dde0dcd209f731c",
    "497d68907a91c0067c8d1e1e61e032cb66dbd8727e8a04e2b89148bafd4ee97b", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'sse_event_count': 25,
     'model': 'gpt-6.2-astra',
     'provider_request_id': 'c4d85a15-6120-4198-9163-a174448ee76d',
     'response_id': 'resp_067bd812780dc5fc016b089735ee8c87d182856fcaeb262045',
     'conflicting_response_id': 'resp_c4d85a15612041989163a174448ee76d',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 206.87087195785716,
     'elapsed_seconds': 210.5370893748477,
     'sse_milestones': {'response.created': {'elapsed_seconds': 30.38787754205987, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 30.38887708284892,
                                                 'interval_seconds': 0.000999540789052844},
                        'response.output_item.added': {'elapsed_seconds': 30.389658582862467,
                                                       'interval_seconds': 0.0007815000135451555},
                        'response.output_item.done': {'elapsed_seconds': 45.28178858291358,
                                                      'interval_seconds': 14.892130000051111}}})

SCOPES[(157, 19)] = replace(SCOPES[(157, 18)], wave=19, cooldown_field="not_before_unix",
                              cooldown_unix=1795725723.7298713, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(157, 18)].archive,
    Review(_157_WAVE18, _157_WAVE18_REQUEST, "6e2250d2243c49b3888b1c92dd4ebcb1",
        "a0e850ac8eecc4059420a6344d009a83c443b5a17b381893a46e4932885b9732",
        "b0bbe0a3a2cc2a876ec12c221b00fbd11283fac27bf2c4ff53d3916095857779",
        "616c2e1b13491406c543ddb968d66cb3f3dd919b9ed0ab51d802d0482d29adde",
        "84bbab0a142c49200987787113fce71461ad57c6d750840dd96fbd4bea1c6416", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'sse_event_count': 41,
         'model': 'gpt-6.2-astra',
         'provider_request_id': 'b76a116d-7b48-42da-b76a-b2c515091c21',
         'response_id': 'resp_06ef494d4a709fd9016b0898ce801887d2bdafdc5109af59b9',
         'conflicting_response_id': 'resp_b76a116d7b4842dab76ab2c515091c21',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 197.2681829999201,
         'elapsed_seconds': 200.00862212502398,
         'sse_milestones': {'response.created': {'elapsed_seconds': 20.034967333078384, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 20.03709787502885,
                                                     'interval_seconds': 0.0021305419504642487},
                            'response.output_item.added': {'elapsed_seconds': 20.03945575002581,
                                                           'interval_seconds': 0.002357874996960163},
                            'response.output_item.done': {'elapsed_seconds': 29.1976025828626,
                                                          'interval_seconds': 9.158146832836792}}},
        earlier=(_157_WAVE18_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "44adc2554e9507e69a247c98de94a2cd785573fad4b6354c5f2ab214c473dae7",
            "attempt_00.replay.terminal.json": "46fe156552c86721fafd3db97c63372916e4a7c6bec6ea17700813808de3f8e7",
            "attempt_01.replay_link.json": "eff15bd38ffae5997f2f4bfbdc129854ec5a5d679df2fd7ee0e2075838ffdf41",
        }),
))


_157_WAVE19 = retry.bounded.RunIdentity("-157-relay-retry-19",
    "d613f006d5563188c15a76e345a902e7b9c9a8cc134aea74fc25760fa8aab862",
    "24bc96d658f867936a1f306317e09429af05552af5f6ed787694554faad9b854")
_157_WAVE19_REQUEST = "172ebd981ada49931004781d55dfb28c6ebba1543c062b617962310c48f2e7ec"
_157_WAVE19_FIRST = Review(_157_WAVE19, _157_WAVE19_REQUEST, "3eee29e09a504e3ca05604684fd42780",
    "833fd0b9125ea36eea71d760036b033301011d9f4d86d7a105e8a3ec236bcffd",
    "1821aadcaf1498e1d192a09374af9e16d84b53f7a1a4e3a747f09d0a9aceb74d",
    "4c58a5ef02f9d710f4f28e53f2bf214876a8dc1b6f52bef665487df5ce1c121e",
    "626457e44823842db2bf3bf5681804b4cd27abe78a89982a53a914e1e4bc2c7d", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'event_type': 'response.output_item.added',
     'sse_event_count': 25,
     'model': 'gpt-6.2-astra',
     'provider_request_id': 'f85f0269-1b20-4542-8a3c-c9485865d8f3',
     'response_id': 'resp_0db7e393568f5b70016b089c63e12c87d1ab61a101499e91f3',
     'conflicting_response_id': 'resp_f85f02691b2045428a3cc9485865d8f3',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 199.4403174170293,
     'last_sse_activity_seconds': 199.4403174170293,
     'elapsed_seconds': 210.07300991704687,
     'sse_milestones': {'response.created': {'elapsed_seconds': 29.947434958070517, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 29.948451332980767,
                                                 'interval_seconds': 0.0010163749102503061},
                        'response.output_item.added': {'elapsed_seconds': 29.949253208003938,
                                                       'interval_seconds': 0.0008018750231713057},
                        'response.output_item.done': {'elapsed_seconds': 46.910151625052094,
                                                      'interval_seconds': 16.960898417048156}},
     'network_phases': {'direct_ipv4_http11': {'duration_seconds': 29.88038829085417,
                                               'observed_seconds': 29.89017525012605,
                                               'state': 'complete'},
                        'direct_ipv4_preflight': {'duration_seconds': 0.7483586249873042,
                                                  'observed_seconds': 0.7605074171442538,
                                                  'state': 'complete'},
                        'direct_ipv4_preflight_1': {'duration_seconds': 0.7483586249873042,
                                                    'observed_seconds': 0.7582412920892239,
                                                    'state': 'complete'},
                        'direct_ipv4_preflight_1_dns_tcp': {'duration_seconds': 0.1934168750885874,
                                                            'observed_seconds': 0.20489329216070473,
                                                            'state': 'complete'},
                        'direct_ipv4_preflight_1_tls': {'duration_seconds': 0.19366504182107747,
                                                        'observed_seconds': 0.4008761669974774,
                                                        'state': 'complete'},
                        'receive_headers': {'duration_seconds': 25.362807207973674,
                                            'observed_seconds': 29.885123749962077,
                                            'state': 'complete'},
                        'send_body': {'duration_seconds': 0.7409001251216978,
                                      'observed_seconds': 4.520404958166182,
                                      'state': 'complete'},
                        'send_headers': {'duration_seconds': 0.000208541052415967,
                                         'observed_seconds': 3.7790979580022395,
                                         'state': 'complete'}},
     'first_phase_seconds': {'http_headers': 29.89180558314547,
                             'http_phase': 0.20489329216070473,
                             'request_dispatched': 3.634974041953683,
                             'sse_event': 29.947434958070517,
                             'transport_failure': 210.07300991704687}})

SCOPES[(157, 20)] = replace(SCOPES[(157, 19)], wave=20, cooldown_field="not_before_unix",
                              cooldown_unix=1795727063.4760141, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(157, 19)].archive,
    Review(_157_WAVE19, _157_WAVE19_REQUEST, "c027d0bb376640a59a9127deff102363",
        "102c350d59fea52cf9f18eada25f6e93ab77c89bdd96df0683406d970cc8ff4d",
        "caf621298a8eab94cfb378d9e29baeb741c318f8820c4c6e573077f053e8375c",
        "313bf93d351ae6ca5ed2fb177c76b517aef5ae9f02ae245b6851c790c9d8029a",
        "e2259b99a395b4fce239ddc98ab11c1c264078539ea168381efaa3c3b45f9399", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'event_type': 'response.output_item.added',
         'sse_event_count': 25,
         'model': 'gpt-6.2-astra',
         'provider_request_id': 'e9e45cf3-5ee8-45e3-adea-de3b115d824d',
         'response_id': 'resp_05f7313513895fa0016b089dfeab4487d186f98ab482f5d3d4',
         'conflicting_response_id': 'resp_e9e45cf35ee845e3adeade3b115d824d',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 207.11027549998835,
         'last_sse_activity_seconds': 207.11027549998835,
         'elapsed_seconds': 209.26974008302204,
         'sse_milestones': {'response.created': {'elapsed_seconds': 29.327437083004043, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 29.329424374969676,
                                                     'interval_seconds': 0.0019872919656336308},
                            'response.output_item.added': {'elapsed_seconds': 29.332867791876197,
                                                           'interval_seconds': 0.0034434169065207243},
                            'response.output_item.done': {'elapsed_seconds': 47.64363854192197,
                                                          'interval_seconds': 18.310770750045776}},
         'network_phases': {'direct_ipv4_http11': {'duration_seconds': 29.3200689998921,
                                                   'observed_seconds': 29.323291207896546,
                                                   'state': 'complete'},
                            'direct_ipv4_preflight': {'duration_seconds': 1.0522868339903653,
                                                      'observed_seconds': 1.057941332925111,
                                                      'state': 'complete'},
                            'direct_ipv4_preflight_1': {'duration_seconds': 1.0522868339903653,
                                                        'observed_seconds': 1.0556331248953938,
                                                        'state': 'complete'},
                            'direct_ipv4_preflight_1_dns_tcp': {'duration_seconds': 0.18832250009290874,
                                                                'observed_seconds': 0.19192058290354908,
                                                                'state': 'complete'},
                            'direct_ipv4_preflight_1_tls': {'duration_seconds': 0.1945246250834316,
                                                            'observed_seconds': 0.38822079193778336,
                                                            'state': 'complete'},
                            'receive_headers': {'duration_seconds': 25.0867345421575,
                                                'observed_seconds': 29.318083750084043,
                                                'state': 'complete'},
                            'send_body': {'duration_seconds': 0.7547209588810802,
                                          'observed_seconds': 4.228899291949347,
                                          'state': 'complete'},
                            'send_headers': {'duration_seconds': 0.0001393752172589302,
                                             'observed_seconds': 3.4739071670919657,
                                             'state': 'complete'}},
         'first_phase_seconds': {'http_headers': 29.324781416915357,
                                 'http_phase': 0.19192058290354908,
                                 'request_dispatched': 3.471844542073086,
                                 'sse_event': 29.327437083004043,
                                 'transport_failure': 209.26974008302204}},
        earlier=(_157_WAVE19_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "4749661d7fca63e1c4d427e2d221547b43a735d5828334dace6f8a9ec4558f06",
            "attempt_00.replay.terminal.json": "4074fb1342c26de8c1eb15743970d4a9610f501eb23799bc738380a4391b07b4",
            "attempt_01.replay_link.json": "7c09a96596a2f6027f5b1c37ed004804e4afe90cb9f91cf065bf15e86993cd4c",
        }),
))


_157_WAVE20 = retry.bounded.RunIdentity("-157-relay-retry-20",
    "9935b79b98c7f107beee8dd9ec1d90d95e778411099c869367dba3b8d2792ee9",
    "45733b57666b64bc529abbdb6f6b40b4a38c147ae36e419d0d4510d9767b5f6c")
_157_WAVE20_REQUEST = "23a23024f1f3a784e66302e5e055fb64d6e854cb8c5f8c4df3bcc4e002f41679"
_157_WAVE20_FIRST = Review(_157_WAVE20, _157_WAVE20_REQUEST, "aa56b198c9b04413ace9ca158b0fc34b",
    "247833a35e6cd63cff4e50fb0dab6ea1d666378563aee4d65744c6863143cf8d",
    "2a8c59d26dc03d24d2bd473b90255d4ca8e97cb4140a53ca9bc1c9e9ad34365b",
    "9082931239072337a8e46317dc0c8ae1cb19b47613cfbf12d62ffbfcc90ec047",
    "2d096ea423991b13d2c5fb98522b3756d61ff6960ced3db92174f5028745752d", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'event_type': 'response.output_item.added',
     'sse_event_count': 41,
     'model': 'gpt-6.2-astra',
     'provider_request_id': '86bb8e9b-3e2f-4a56-8796-d96974feea28',
     'response_id': 'resp_03b86e4b747c15be016b08a0cd28ec87d29c28646011eb78b4',
     'conflicting_response_id': 'resp_86bb8e9b3e2f4a568796d96974feea28',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 195.33912854199298,
     'last_sse_activity_seconds': 195.33912854199298,
     'elapsed_seconds': 198.35154516692273,
     'sse_milestones': {'response.created': {'elapsed_seconds': 18.248514167033136, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 18.249285416910425,
                                                 'interval_seconds': 0.0007712498772889376},
                        'response.output_item.added': {'elapsed_seconds': 18.250103499973193,
                                                       'interval_seconds': 0.0008180830627679825},
                        'response.output_item.done': {'elapsed_seconds': 27.523103625047952,
                                                      'interval_seconds': 9.27300012507476}}})

SCOPES[(157, 21)] = replace(SCOPES[(157, 20)], wave=21, cooldown_field="not_before_unix",
                              cooldown_unix=1795728200.9793127, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(157, 20)].archive,
    Review(_157_WAVE20, _157_WAVE20_REQUEST, "7a9c1ecb0d024f189769ef2d2954374b",
        "86a84ff9706c6b23ae2949ea5066c586878aa62762ec6c92f5f6a9b25348f5f3",
        "6c1854ee27ef0ab3b1d558f579c8393f25c008e3dccc0de5a4bec8f692b1bf77",
        "b1c6f17416f5ca38fc74e34747afdd360366becf7c838a23cb582262a5a08e9b",
        "ff27b14a8e2d06c20ab40929a88bee4226c731b14b69b1570467ebe9868578ad", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'event_type': 'response.output_item.added',
         'sse_event_count': 25,
         'model': 'gpt-6.2-astra',
         'provider_request_id': 'fc9191d2-7cb7-49a6-9ca9-5cad7e93d45c',
         'response_id': 'resp_0bcbdb82d1ac7eb2016b08a274c11887d083d338f413c3a9a1',
         'conflicting_response_id': 'resp_fc9191d27cb749a69ca95cad7e93d45c',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 205.73171916720457,
         'last_sse_activity_seconds': 205.73171916720457,
         'elapsed_seconds': 207.95196458417922,
         'sse_milestones': {'response.created': {'elapsed_seconds': 27.94564904202707, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 27.94786670920439,
                                                     'interval_seconds': 0.0022176671773195267},
                            'response.output_item.added': {'elapsed_seconds': 27.949778834125027,
                                                           'interval_seconds': 0.0019121249206364155},
                            'response.output_item.done': {'elapsed_seconds': 44.31273345905356,
                                                          'interval_seconds': 16.362954624928534}}},
        earlier=(_157_WAVE20_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "0d6a128e4b2a0e9dafede3d861318d517818f7c505b7f5bd4e4eb464cbd43cf4",
            "attempt_00.replay.terminal.json": "74d3c428e925e351ffed88eeeb7578f3bbb9b3dcf20638064e025cfefe135338",
            "attempt_01.replay_link.json": "62712afb21826d24a80ff2cc5aa9e6f61188eb4f44cd7e6169cca2ddbd7a0b43",
        }),
))


_157_WAVE21 = retry.bounded.RunIdentity("-157-relay-retry-21",
    "0e2c3407f84bb056870db36755a830924586ef467107e729a0e61cdf87abe81b",
    "16fc32b62b823f3ce82de86055ef803d08c4c6d4c4b25d60cb42ba777f4ebede")
_157_WAVE21_REQUEST = "f0504b4a24091d4da009bbbb3a38ab3e4894ada1e1a08ea8fcbfe95104e62c29"
_157_WAVE21_FIRST = Review(_157_WAVE21, _157_WAVE21_REQUEST, "b41e1069da06490dba37e4cf783a19d5",
    "95e5b2a5e685451399da4eb1a2e27f3e96daf15acd58b409703c07fc71e885c1",
    "35d81f4b88e226e6184041c92f96ab7738c6ac202523d26b83a8d60ffc9c6fd5",
    "47f6405a933f740def543e2948d852b08eb1a7a1ae718a19051c8d60a730c5af",
    "8b751695d95ab861769c6a4bf9ab6fa93d219d3820c5d8ca8bbe58a1a19589ec", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'event_type': 'response.output_item.added',
     'sse_event_count': 25,
     'model': 'gpt-6.2-astra',
     'provider_request_id': '01978cb3-b6bb-41c8-a42a-89739cfc7967',
     'response_id': 'resp_089bbe29dafc2e3d016b08a5556e6887d0bd5f48d38228468d',
     'conflicting_response_id': 'resp_01978cb3b6bb41c8a42a89739cfc7967',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 197.751758125145,
     'last_sse_activity_seconds': 197.751758125145,
     'elapsed_seconds': 208.89500066707842,
     'sse_milestones': {'response.created': {'elapsed_seconds': 28.73804650013335, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 28.738887374987826,
                                                 'interval_seconds': 0.0008408748544752598},
                        'response.output_item.added': {'elapsed_seconds': 28.73972333408892,
                                                       'interval_seconds': 0.0008359591010957956},
                        'response.output_item.done': {'elapsed_seconds': 44.40929100010544,
                                                      'interval_seconds': 15.669567666016519}}})

SCOPES[(157, 22)] = replace(SCOPES[(157, 21)], wave=22, cooldown_field="not_before_unix",
                              cooldown_unix=1795729347.1188662, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(157, 21)].archive,
    Review(_157_WAVE21, _157_WAVE21_REQUEST, "4bc6d117ceb145dba341f1e694f903e8",
        "2dd2c0c083dce95aff243c6883e8eaeb15b30dcb5db1489a20adfc61473b218f",
        "10d1482c1c46879f132f4edb8a5a1f54f30a6fce1a064e771e4861c88a532d4b",
        "3f5029b698dbbac214f7826a53111deb066b980c791abfbdcac9d0a87cc0fc00",
        "1290f9fc10cbe9ce3c036c2e1b57ab8dc42ce6fda84c330ea8d123cc62893473", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'event_type': 'response.output_item.added',
         'sse_event_count': 41,
         'model': 'gpt-6.2-astra',
         'provider_request_id': 'b8951463-48bd-49d5-8022-f46189ddccb4',
         'response_id': 'resp_0a2662c58705c65e016b08a6ef67a887d2ab7e6f3a1d86c0c0',
         'conflicting_response_id': 'resp_b895146348bd49d58022f46189ddccb4',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 194.418092665961,
         'last_sse_activity_seconds': 194.418092665961,
         'elapsed_seconds': 197.21956591587514,
         'sse_milestones': {'response.created': {'elapsed_seconds': 17.275104415835813, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 17.27725474978797,
                                                     'interval_seconds': 0.0021503339521586895},
                            'response.output_item.added': {'elapsed_seconds': 17.280243249842897,
                                                           'interval_seconds': 0.00298850005492568},
                            'response.output_item.done': {'elapsed_seconds': 26.639862957876176,
                                                          'interval_seconds': 9.359619708033279}}},
        earlier=(_157_WAVE21_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "5286bf394052d7ba82a7b9504cf98d693831d4dec9c1152d6d8bac064cdce869",
            "attempt_00.replay.terminal.json": "184f09c5cd9ec56827e5dc39db0bb282149f7da4ed43639ac3f225871a6d32e0",
            "attempt_01.replay_link.json": "6622c6a7d32e4b70efc6e25120a9ba721668f2a3e9cb629af7fc9292a0d4f07b",
        }),
))


_157_WAVE22 = retry.bounded.RunIdentity("-157-relay-retry-22",
    "752e6a56d1796ed6817b05da2afc873efc4c0b6db404c7565905c2d1ad9b43d6",
    "7e7684b8033561fa040381c883074e09b31525e69fd795a3d97f0b7150ef2873")
_157_WAVE22_REQUEST = "b10df8e26573ee3dce765103d9c1a455cfbd554ca5b1055be4cb50abe8d460dc"
_157_WAVE22_FIRST = Review(_157_WAVE22, _157_WAVE22_REQUEST, "2b9b653c5a98441eb5d17613841cbcc6",
    "3465b4a67d97669c197a2bfbc167f366f0370306327178f5b1a2cbf8fc0eb448",
    "24abd4bb288190d75117686c1e70fa00c0e422e2e60c8700802cc4f27287a50d",
    "17d93807928b22536fc71f443bb5833048a55597cde2ca74391a2aade757d744",
    "6fc528fb22f950f9260d793214c156edbd3f216cc86b9462095bd60d82020c4b", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'event_type': 'response.output_item.added',
     'sse_event_count': 27,
     'model': 'gpt-6.2-astra',
     'provider_request_id': 'b001f522-d40a-4cee-a024-f7d1d32a7782',
     'response_id': 'resp_0a2bb209af7392b5016b08a93aa37887d18e7aa8e69a147480',
     'conflicting_response_id': 'resp_b001f522d40a4ceea024f7d1d32a7782',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 206.6090013750363,
     'last_sse_activity_seconds': 206.6090013750363,
     'elapsed_seconds': 211.886689750012,
     'sse_milestones': {'response.created': {'elapsed_seconds': 31.848059833049774, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 31.84893950005062,
                                                 'interval_seconds': 0.0008796670008450747},
                        'response.output_item.added': {'elapsed_seconds': 31.91873516701162,
                                                       'interval_seconds': 0.06979566696099937},
                        'response.output_item.done': {'elapsed_seconds': 47.82585004204884,
                                                      'interval_seconds': 15.907114875037223}}})

SCOPES[(157, 23)] = replace(SCOPES[(157, 22)], wave=23, cooldown_field="not_before_unix",
                              cooldown_unix=1795730335.952652, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(157, 22)].archive,
    Review(_157_WAVE22, _157_WAVE22_REQUEST, "33fabf638c754f0ca1655aeb45b3f763",
        "0b99590ab8b4e2af8663bd7507316eb402dfe66d2622315b179f3fba5db6f6a6",
        "c876dcb80c7ab5946c4e0f453cc5dbd3db309e69a6fb1890d92b56cddd5b7eae",
        "33b6bf126eba81acccbce695d4e0de5ae1da57612e1ccbf8dbdf48c6af3a6432",
        "7579d68bf8edf91209f4f7d88188ce9cde91ec6a306ce129aa4ddd258df9f480", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'event_type': 'response.output_item.added',
         'sse_event_count': 41,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '0c304b28-2b77-4c4d-a33f-3742db1d64ce',
         'response_id': 'resp_09a0a8dd60f15e58016b08aad27b5087d29915b446ce8d5a9b',
         'conflicting_response_id': 'resp_0c304b282b774c4da33f3742db1d64ce',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 195.4820779999718,
         'last_sse_activity_seconds': 195.4820779999718,
         'elapsed_seconds': 198.22378837503493,
         'sse_milestones': {'response.created': {'elapsed_seconds': 18.278119042050093, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 18.279653459088877,
                                                     'interval_seconds': 0.001534417038783431},
                            'response.output_item.added': {'elapsed_seconds': 18.28185429191217,
                                                           'interval_seconds': 0.002200832823291421},
                            'response.output_item.done': {'elapsed_seconds': 27.594514833996072,
                                                          'interval_seconds': 9.312660542083904}}},
        earlier=(_157_WAVE22_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "e891e2aa0d289c9c1b7999a748a5d01a68de9ea5d57b62654fa99a4e1668ea4e",
            "attempt_00.replay.terminal.json": "a28dd9a9f3404204650fa22e31ce59359a2b2635fe1236ec4a9f43cccaa6a90e",
            "attempt_01.replay_link.json": "1bbe0332724f4ee700bf577ab568840e926e3d2a5a379628437e466df2c2d6f1",
        }),
))


_157_WAVE23 = retry.bounded.RunIdentity("-157-relay-retry-23",
    "0e8357f1ce54a69ed9f98d462fa9bff9cfd2a902b1ab419e934251fb9235f72e",
    "35c74688d1ee0a102bdb11098b5bbbfaa80d941ffbab8c6b764289bdee83e647")
_157_WAVE23_REQUEST = "bd9d1fa27271276b7e3f0f7972e301c64bae808a521ecc9a16eb5b0cd04964d2"
_157_WAVE23_FIRST = Review(_157_WAVE23, _157_WAVE23_REQUEST, "afbcb8adb16847d097eb676509fe245a",
    "b317bec12e06552acfc750df7d4f16793c5b0df6eb0963a0bb0dfad37b15f215",
    "d6e980d10c178e73249a77a271049227f2791241c809765316c15d17a55f5d05",
    "83eccebb686f2e0a10d24956edf72a1c2a8caf2c61930c8c886b5ae037193309",
    "732cacac964b0027203050e4c49983280d90f66c31d5a06e701fe8bd1c8ef958", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'event_type': 'response.output_item.added',
     'sse_event_count': 41,
     'model': 'gpt-6.2-astra',
     'provider_request_id': '07b82922-352d-4e4a-803b-4c1c08fbe0d6',
     'response_id': 'resp_0584f6a69ceef0e8016b08ace4e6bc87d29f88098f598925b5',
     'conflicting_response_id': 'resp_07b82922352d4e4a803b4c1c08fbe0d6',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 196.29554429184645,
     'last_sse_activity_seconds': 196.29554429184645,
     'elapsed_seconds': 199.2967434169259,
     'sse_milestones': {'response.created': {'elapsed_seconds': 19.180507499957457, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 19.181389333913103,
                                                 'interval_seconds': 0.0008818339556455612},
                        'response.output_item.added': {'elapsed_seconds': 19.182028874987736,
                                                       'interval_seconds': 0.0006395410746335983},
                        'response.output_item.done': {'elapsed_seconds': 28.55891704186797,
                                                      'interval_seconds': 9.376888166880235}}})

SCOPES[(157, 24)] = replace(SCOPES[(157, 23)], wave=24, cooldown_field="not_before_unix",
                              cooldown_unix=1795731255.068868, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(157, 23)].archive,
    Review(_157_WAVE23, _157_WAVE23_REQUEST, "56dcb2c4d8384847872b8cda68e4de43",
        "5903bac054ceb9405f7e8aee0de110c5116cfdc891807bfdcb0cd40531906baf",
        "6a5aa138d0781a9ed8c98ae7019fbb9e2a65390d56498b4cca10dfa02d0251d6",
        "5a1c837c7398939e407815fee96cfedbd7e158400ff9fa3d466e8c5f772fb32c",
        "bfe944c6c4c0bd43e8f30b44443bb09ce31597e58799104dd4c1ba887ae57dca", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'event_type': 'response.output_item.added',
         'sse_event_count': 41,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '4a62968f-fc78-44ce-96f3-26e9af0fc8aa',
         'response_id': 'resp_0c5cd60d3e13f521016b08ae6cdea487d083ae473d6ffacb66',
         'conflicting_response_id': 'resp_4a62968ffc7844ce96f326e9af0fc8aa',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 196.19101420906372,
         'last_sse_activity_seconds': 196.19101420906372,
         'elapsed_seconds': 199.0167657500133,
         'sse_milestones': {'response.created': {'elapsed_seconds': 19.083091042004526, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 19.086583208991215,
                                                     'interval_seconds': 0.0034921669866889715},
                            'response.output_item.added': {'elapsed_seconds': 19.088602042058483,
                                                           'interval_seconds': 0.002018833067268133},
                            'response.output_item.done': {'elapsed_seconds': 28.337056583957747,
                                                          'interval_seconds': 9.248454541899264}}},
        earlier=(_157_WAVE23_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "5b5516ae557ff1cbb8f3baed23945892c80208f7475148bf9649c086b19c2e43",
            "attempt_00.replay.terminal.json": "629c73e30e5ac97720e0fc0bfda9b03d1af11e1d2a166688ed51a4078a859bba",
            "attempt_01.replay_link.json": "de2327c88b26f99a6baaad0da64ae73bd974a62adccd4adf4b9f6967447d5e93",
        }),
))


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

SCOPES[(12203, 6)] = replace(SCOPES[(12203, 5)], wave=6, cooldown_unix=1795687035.0316763, archive=(
    *SCOPES[(12203, 5)].archive,
    Review(retry.bounded.RunIdentity("-12203-relay-retry-05",
        "e2b5af09a15ec1c32dcc7ddd3d0f418ac12309178154c304379a0f69dd614a77",
        "435c4957488c881e7132ef1541403328a5b6cf0d0ee383a7e9ff11de5ca400b0"),
        "29f5cd07c5babf292526190b1d4ffc831e81b4973996e377aec62108241bcdcc", "e729a3ed4b8646eeb4f138afa5cdfd92",
        "44ea8435296a6b7135a027d4ef7ac5f6b13be2b3cc308a8e6b84caf1c1e667f9",
        "85ab759d4150348aea0821c1b55cc0a4bd77dab56aed4316d5b40ef5b7f92160",
        "d8f625b0cf3f3fc44a8174c81d2e21be6aaa984c5d2fa69fd0ceef1ef585610a",
        "c68e7df78ed3a63c4cf2ec4982b00ff1a5eb6649d920e391728f9485a7772824", "error",
        {"http_status": 200, "phase": "transport_failure", "event_type": "error", "error_type": "error",
         "provider_error_code": "upstream_stream_read_error", "sse_event_count": 10,
         "provider_request_id": "e8a4abba-c6fc-43b4-863a-340949691acb",
         "response_id": "resp_03e8605b24debb20016b080227447887d2ac1a1e5f5313b441"}),
))

_12203_R5_OLD = retry.bounded.RunIdentity("-12203-relay-retry-06",
    "25877c1f9d94e21b6edef515716ec3131961bc584cd6e7a7d3cd293fe6e57b55",
    "505603229a4568db10a5d830ae602005910e37a89baee011ee5474b5b8aed760")
_12203_R5_REQUEST = "980bebc61700c1aa38cea2122393f8edce7690379d71e8cedc35844317644bb2"
_12203_R5_ATTEMPTS = tuple(Review(_12203_R5_OLD, _12203_R5_REQUEST, item.attempt_id, item.job_sha256, item.result_sha256,
                                 item.terminal_file_sha256, item.trace_sha256, "response_id_mismatch", item.observation)
                           for item in (
    wave157.PriorAttempt("e3f44e8d707943c3addae8000ea5c342",
        "c400e12117d7d83cbf39dd2ce20e77f3a6d26206e4706eabf28d85e55e06b1b0",
        "35b9107cccc252238008048eafe50b4e0de3ba0b46dad62dab281ee5f1f99bca",
        "665e80f3da6b65115b03853bb3228b59df135478d7dc7b51a1a43ce5d7d18384",
        "3ea88cb3443dffebc3104376b03d7b6f15c8464fc2cf997ef1878821d0030842",
        "27b81e7c-120f-4d55-be10-a23183a1ee81", "resp_021bc56c61ab87f4016b080685e44087d2847891aa246d8686"),
    wave157.PriorAttempt("94b1558e7c0047e9b37b96e26b42e203",
        "cab8059942f00cca4b9dec94e9a92e44400bdfc512f3e61b9d2f6a7272df3880",
        "45b7bc50571e263baf4e2cef1312c404be6699e9986855068d82ee1681cd9ed9",
        "d79bb2cf536690154f461551eb03b2781063f2b7e10dec91f6368d2dfb8ea19e",
        "0d98d86ef6d1088ddd53e57b2b4cb94ea5459e46271550f566acc8eb07bec856",
        "c9fc2a09-7679-4b58-a20d-0ee33729728a", "resp_0c871587f2f473db016b080824636487d0a93ee1bdab4872ac"),
))
SCOPES[(12203, 7)] = Scope(
    12203, 5, 7, "00d53155b752dc54eed4e6aad7bfbdb01d19b21c7f0866cebd85391f8d456373", (
        replace(_12203_R5_ATTEMPTS[-1], earlier=_12203_R5_ATTEMPTS[:-1], replay_files={
            "attempt_00.replay_authorization.json": "6673d2374dbc481ff2d1c1ea52f734a2d4fc93d36f5b377a88b3c1afcdb7cd43",
            "attempt_00.replay.terminal.json": "a423f8707d1db0eb4b637dc9e799401908b4c026aa163814033680d4d0a95d42",
            "attempt_01.replay_link.json": "3c0326655d089be53a55e26ca0f5ebb01bbea864b3bc5461f93adcf47d3891ea",
        }),
    ), "failed_unix", 1795688681.139692, original_start_round=4, fitted_prefix_files=(
        ("round_05/source.json", "9b290ab2a6c5a7da8334eb77ca003a10ac1b52ed10a76c0baf3b02b2b9f64eac"),
        ("round_05/transition.json", "54470e8d33fab447b9e3704826af595f044199b0c8d2880f64a2cdc1b4cc2c2b"),
        ("round_04/memory.json", "0150070586759a4a2200f593df0af5207b6c63a84b92c3081996e4ddca04ce11"),
    ), required_sources=QUEUE_SOURCES, boundary_physical_attempts=2, original_boundary_physical_attempts=1,
)

_12203_WAVE07 = retry.bounded.RunIdentity("-12203-relay-retry-07",
    "cbe9c095142e5e0ccf2f6db546b012e2a028ec19a3360df1e5030954d430dc4a",
    "c4c3a24bc680d92435d9c9c16637fbe31618f56c11f477a91b8124a6b5a98f54")
_12203_WAVE07_REQUEST = "3ec4ff97a04342586a18a980e0e10fdac0fb6940299263a471f4eb1b11c0874c"
_12203_WAVE07_FIRST = Review(_12203_WAVE07, _12203_WAVE07_REQUEST, "79b3dbd9ae574275bdf5b358a6fb0d94",
    "70e22b03ac3869c06f56cc9283fbf5184695e1a472b19e1b83e44b7f9d3c4ede",
    "229b4d1b83302cf2471715025fec255c476bc4c01e31a17f99fa3c6da1ce9eff",
    "e6957c8821f8fb9f7d463fa30fb58f5f84527df1f8207977b13c3856ff78c0b7",
    "abd06785c3207fd67a1836ad0a359cd5297ea4717f7221ae4b2ecb6e97600966", "response_id_mismatch",
    {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 41,
     "provider_request_id": "762f77b7-e931-4e59-8a6c-ac3903ff8ac0", "model": "gpt-6.2-astra",
     "response_id": "resp_0e105f0a0cdc31b0016b080b6dc65487d088ce8ac384e71dcf",
     "conflicting_response_id": "resp_762f77b7e9314e598a6cac3903ff8ac0", "conflicting_event_type": "response.failed",
     "response_id_matches_transport_alias": True})
SCOPES[(12203, 8)] = replace(SCOPES[(12203, 7)], wave=8, cooldown_unix=1795690739.9466105,
                             original_start_round=5, original_boundary_physical_attempts=2, fitted_prefix_files=(), archive=(
    *SCOPES[(12203, 7)].archive,
    Review(_12203_WAVE07, _12203_WAVE07_REQUEST, "a23ebac4293547a7bae5dd5d862ba37a",
        "e3fb82f78d7c7761b4ff81c96ac50de296e06003482505d66a426877fecaef9e",
        "b2371c6afe4fbbd9525ef9097025ef661546c8eb07df8256f05c4aa173168666",
        "40149ac3a1373cdb77683e491972ce31113f9fd8b45fe4f33bf3b36577b34ee1",
        "775ebcaab79c6f3da6f2474a79c0a68beef93b5f4a23dd4a7af24a77c0988305", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 23,
         "provider_request_id": "6bb7602d-b97a-4472-b11b-35a7970d6dc9", "model": "gpt-6.2-astra",
         "response_id": "resp_0e33b0170fc4fd53016b08100f08b487d08dd9fa602b7f5aa0",
         "conflicting_response_id": "resp_6bb7602db97a4472b11b35a7970d6dc9", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True},
        earlier=(_12203_WAVE07_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "752e4554bd8a33e0bb475158b5f206e579d83090375afea485b6666cac9f6929",
            "attempt_00.replay.terminal.json": "b6d06b60daa1be86c15e605eb6c96a245db67b29ab021433a885741af5f815d4",
            "attempt_01.replay_link.json": "ee26381d4987507172431e0f8f3309fcbe9c4bdf717cf2877fb30eb873cfa028",
        }),
))

_12203_WAVE08 = retry.bounded.RunIdentity("-12203-relay-retry-08",
    "5f65ea60af0ce2440d123a21f350e60d53de9d40e53b85557a3ec8a43aa7456a",
    "76fd9ea06a3de5a1b3e26c6fa0755e4ad0065b0a26841d8798f97074367397c1")
_12203_WAVE08_REQUEST = "bc13a59e4d51133e819d127814ff58e6b9d2657f5c69a1320f21aaf1d9c6a316"
_12203_WAVE08_FIRST = Review(_12203_WAVE08, _12203_WAVE08_REQUEST, "22736e2c597d42bba2f5730825e76a65",
    "09c94cf6eae1273dfde2076230f3ec75608874c2d1c45ac80c28b152459a2ed7",
    "487ce8f74f48e74f41964dd4656c4562ae192a8486b9fd6a8025d78f302fa7f3",
    "fdca433ee02d59c560b3374f10da8e85512550ec2bf6f71db914e61aebd3ceac",
    "1a720d2f5218ef8658d2fe469193bf265abb83a346a5b0e8a9b64c5bd538b3c3", "response_id_mismatch",
    {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 24,
     "provider_request_id": "beb6d4fe-5392-4e55-a5f9-51bbc251a142", "model": "gpt-6.2-astra",
     "response_id": "resp_08e5131532e68c3f016b08173f0dbc87d2ab78d3678686e13a",
     "conflicting_response_id": "resp_beb6d4fe53924e55a5f951bbc251a142", "conflicting_event_type": "response.failed",
     "response_id_matches_transport_alias": True})
SCOPES[(12203, 9)] = replace(SCOPES[(12203, 8)], wave=9, cooldown_unix=1795692975.7197583, archive=(
    *SCOPES[(12203, 8)].archive,
    Review(_12203_WAVE08, _12203_WAVE08_REQUEST, "f6543a336b2c4f04a3c75d585b9a7083",
        "cab5a0446f3d36a6bfc8e237183b22330b71d5995c3c7e6dcd09083acc84a0a1",
        "24abca31698e415c935f3030214e243c3cad42c998efd4c8dd4883b3bdb605b1",
        "a0673027c885037ec65346ea56523e7268e919f7be69392171c0c346b6a8d65d",
        "fb1de8ba5997deb7378cf75f47ea62533deee34fa51be135b5820dbe128fbef6", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 41,
         "provider_request_id": "9bd44948-f4f4-40f5-a811-098b71481d85", "model": "gpt-6.2-astra",
         "response_id": "resp_0736a44e75a90c12016b0818e6209087d28545d7eaea36f9c5",
         "conflicting_response_id": "resp_9bd44948f4f440f5a811098b71481d85", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True},
        earlier=(_12203_WAVE08_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "968d3d51677033bf8f14283fe42d4195f9aca0f162e5f78ab852c33456788660",
            "attempt_00.replay.terminal.json": "c7c1876782e113a092c7431cce9b9269309d3c40eac074fcad8ab61ba2dd513f",
            "attempt_01.replay_link.json": "05a826f44aff6fbf994f43dcb50ce60aafe83d9af7b3f07a136f948e26f20ac3",
        }),
))

SCOPES[(12203, 10)] = replace(SCOPES[(12203, 9)], wave=10, cooldown_unix=1795695678.1715395,
                               boundary_physical_attempts=1, required_sources=SAME_RESPONSE_RETRY_SOURCES, archive=(
    *SCOPES[(12203, 9)].archive,
    Review(retry.bounded.RunIdentity("-12203-relay-retry-09",
        "25793611aa80ddf279067ac651908154ecc65ad49a485b4e69c2df6e16322928",
        "4d19e1f3fff41d5af1f09181b8f63f5b876b92f2d5c76f16dc08ab8899d91fb8"),
        "a0bbbe0bcb9362c5ef9262f279efb9a3515d7d0407e1ae6666403f84a634bc22", "47604b0004ba4cd98dd083e22aae017f",
        "8ccd651794938a33dc51fdcbfc077158f675e89d145e0d3724b69a8a2d8597e3",
        "6cb112f81e78254912da39ab4539ae51fc9443308614e93b0424dadc203311cb",
        "3a1d44dfccb67d510169ab721c4eca7e13dedc2fe4e6afdf28cf72c4e171ec3b",
        "33bb92f90edd5c1fd19f9ef1c1ed6bbce1e43297716e536b2f1cb469981bdd41", "timeout",
        {"http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
         "event_type": "response.output_item.done", "sse_event_count": 28, "last_sse_seconds": 980.5284760829527,
         "provider_request_id": "f870028a-25eb-4703-8571-6b0b1dec1870", "model": "gpt-6.2-astra",
         "response_id": "resp_0dc94406a9c8a9b1016b081d14b61887d285421dd0467c6173"}),
))

SCOPES[(12203, 11)] = replace(SCOPES[(12203, 10)], wave=11, cooldown_field="not_before_unix",
                               cooldown_unix=1795696851.8253946, original_start_round=None,
                               original_boundary_physical_attempts=1, boundary_physical_attempts=2, archive=(
    *SCOPES[(12203, 10)].archive,
    Review(retry.bounded.RunIdentity("-12203-relay-retry-10",
        "168df83d138c13c153bc05e91f8ca0e73208affaf08dc3292135c269c30ec8b5",
        "a004554f8d14f8b75b9c40cd084b6f1dba28cf677c246abd882c191869d0934d"),
        "dbe5eb527a0236fc9e81d6714290fe8068bd747d9408362924daac4ed2ac8923", "1f2c7f3be5294230b6a3699b9a653db0",
        "eff1ba75a1cfe1657ddda50d768db4decf4c33c77be3310a39bbbf5d48db1e5b",
        "759839401c4679744ad291df409d6ae9caec11bc349773245e5349be3ae6942b",
        "545be789c7eb182cc8cff91c3c3e3e88b777acc2395f2f63d520e12894ae9f56",
        "1bbd1344e6ad2268c365f851f2b42c363b3d76b6620fbfb89cb7363f80c68d26", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 25,
         "provider_request_id": "e28495c5-d4e2-42ef-8e04-e706fae6752d", "model": "gpt-6.2-astra",
         "response_id": "resp_045a309a88ab9a96016b0827f5520887d0a8494b64e6666c61",
         "conflicting_response_id": "resp_e28495c5d4e242ef8e04e706fae6752d", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True}),
))

_12203_WAVE11 = retry.bounded.RunIdentity("-12203-relay-retry-11",
    "6083e9eb70e70d536952f4c9faa91b310c3560e6934875675f55f91d7135855d",
    "914a2a3187fc33a84ff6678bd7ecbab1f14aeb3692fd4d2dc1a340757d71001a")
_12203_WAVE11_REQUEST = "71b0130320deaf04a0f36006a8886756c66532c5adc43d318d0c19b23ee94878"
_12203_WAVE11_FIRST = Review(_12203_WAVE11, _12203_WAVE11_REQUEST, "6f75c90ab45748eaa054c070473ec296",
    "ee7626ee5b00810e19377551f121599d010da056f3f34236f322b92c3cbf49bc",
    "aec0ced68b9a2daff882ad7ab3a09ac731d14055421c41d2201ac8c747e2d4d9",
    "903b43ba21fbaabc38bbb313d6d902975dc05737b100be0102f0d0fbe2699fa2",
    "76d8acccb74a7380c5969fbc24cf0fe82e504d558a24c96035c132c826ee5319", "response_id_mismatch",
    {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 25,
     "provider_request_id": "2bea3d9f-8d7c-4f4b-bc53-5af4384b54db", "model": "gpt-6.2-astra",
     "response_id": "resp_0be161e1cc6bd6b0016b082d30612c87d1b23550823cea8f04",
     "conflicting_response_id": "resp_2bea3d9f8d7c4f4bbc535af4384b54db", "conflicting_event_type": "response.failed",
     "response_id_matches_transport_alias": True})
SCOPES[(12203, 12)] = replace(SCOPES[(12203, 11)], wave=12, cooldown_field="failed_unix",
                               cooldown_unix=1795699095.9433548, original_start_round=5,
                               original_boundary_physical_attempts=2, required_sources=tuple(
    (source, "aacbc199e896091d4395fc036cb60bc604d91ec326a79fcff26e157d9a73705f" if source == PROVIDER_SOURCE else sha)
    for source, sha in SAME_RESPONSE_RETRY_SOURCES
), archive=(
    *SCOPES[(12203, 11)].archive,
    Review(_12203_WAVE11, _12203_WAVE11_REQUEST, "6bcb5f6a31ec42a0b5c254b787202c52",
        "6d87de5c3b77b273442ca8480732588e6f45d1a57b60462794394aa8dbd92220",
        "efae225e919525904cf79acdd10bf7cd2c4e076ada6e2e168c7be88250bc319f",
        "6ff9d185708909d691c4986010b25d0b8188aa096dfa21fb95ae78ddac08a874",
        "d38fa2f6920488a978e98f9c1b6c5a9a74d9a5f44a613020e52f602b0127e77e", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 25,
         "provider_request_id": "0dac1eab-fa9b-4e8c-95a7-b6425611386b", "model": "gpt-6.2-astra",
         "response_id": "resp_0eecca46bad6c6e3016b0830bcf4dc87d0bf5ebeabbf62be5a",
         "conflicting_response_id": "resp_0dac1eabfa9b4e8c95a7b6425611386b", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True},
        earlier=(_12203_WAVE11_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "0b19a39b3fe1948be32132803058b98d417a21420ae7ac08a3978dba61fa3517",
            "attempt_00.replay.terminal.json": "ba9dda32d3025c6434533d53a1a3de0b4ecf978758fe4c6fd6cfbf84e65bc3a2",
            "attempt_01.replay_link.json": "603fb01451eca42bee560998d456bddf67d1e50a0004dcf609ea31ad33b6fd29",
        }),
))

_12203_WAVE12 = retry.bounded.RunIdentity("-12203-relay-retry-12",
    "0fea4698db29545c7f8981c83c2fbb0891131832a40913d542cde8db38d24c61",
    "e5a55b77d5e384811cd71c158fc058eda2d139523a45ac0480f281e37d964768")
_12203_WAVE12_REQUEST = "0f9446bd6febf0fcda32cfcddb70af5ecd7e097a4cbaed4071c0fba504482860"
_12203_WAVE12_FIRST = Review(_12203_WAVE12, _12203_WAVE12_REQUEST, "7b8dd2a160914ff293622380dd1d8cae",
    "402f872ecbfdb2cc509973c39da844f5518d278fbe25971fa14f767f0c298e6c",
    "9ccf002474dc31a8fdb3176655b390b6ca57823943b8062132ac9145d30516a4",
    "9da7c12c45a23332ed86f2d208f352192e53108f95de8f32bfe4102e56829cf2",
    "19f356dd06a9e6bd1f3b32883709ad7eb0f16ec40d43aeb2d8fa895dbf807264", "response_id_mismatch",
    {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 23,
     "provider_request_id": "470d4f0a-1b3a-4ed0-8460-d71f12a292a8", "model": "gpt-6.2-astra",
     "response_id": "resp_065c53b252d45ce2016b0837cab6b887d0b6ff67a016f45b7c",
     "conflicting_response_id": "resp_470d4f0a1b3a4ed08460d71f12a292a8", "conflicting_event_type": "response.failed",
     "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"})
_12203_WAVE12_PRE_POST_CONNECT = Review(_12203_WAVE12, _12203_WAVE12_REQUEST, "ecb84ce8e8214cca9bb3005c6ff672e0",
    "b941ade623118f585b2e828a8b513f290055b25ec4b8ad34564e1625d6acc4f0",
    "9ca6e60ea32cb8b58d67305f78dd213dbdeb7e1bb4b7a4e4fb5c093183211537",
    "19a85316433270504a4c3c5916971129fe37adca81709c67d1274ee432b7307a",
    "299590b71b3684af2877a6bae97f32473e835d432a054dc455dbb34cbc15c0c9", "connect_error",
    {"phase": "transport_failure", "error_type": "connect_timeout_before_post", "event_type": "ipv4_preflight_http_401"},
    earlier=(_12203_WAVE12_FIRST,), replay_files={
        "attempt_00.replay_authorization.json": "fe3f21744a668027bcce37fff1337384e31257d028242f19d5154751a6bfe278",
        "attempt_00.replay.terminal.json": "54a9cf6654cba8bc2e38b49948ff022b061731daad02367c25959ad4b2a8886a",
        "attempt_01.replay_link.json": "dd8842bef86d5602e5e2d0d8eac25e844b07f05ad12bdcd8ee83feaea6ca7775",
    })
SCOPES[(12203, 13)] = replace(SCOPES[(12203, 12)], wave=13, cooldown_unix=1795701276.67948,
                               boundary_physical_attempts=1, archive=(
    *SCOPES[(12203, 12)].archive, _12203_WAVE12_PRE_POST_CONNECT,
))


SCOPES[(12203, 14)] = replace(SCOPES[(12203, 13)], wave=14, cooldown_field="not_before_unix",
                               cooldown_unix=1795702290.7997937, original_boundary_physical_attempts=1,
                               boundary_physical_attempts=2, archive=(
    *SCOPES[(12203, 13)].archive,
    Review(retry.bounded.RunIdentity("-12203-relay-retry-13",
        "bb8bb02bf7478865f6bd510c6667687b89c8044152a1af834288f4fba3125801",
        "68fc075c81f6c9408b99a9669a650136542f399b77573095f91d475373236055"),
        "094821a5d94e07fd86e2a799bc29905037d405f924048b116eea204a5c398962", "51559c40c09344f1a05f16bae486785f",
        "c52a75b11fe8b8315c73d6322434c70c32b5302ac627dbc7046188e2745bb6dd",
        "ccfef94b6a6b867c063d7e4907221ce70db3cc064e432da7dc73e46e0e859bab",
        "f5254ed9b334038058004aa91a1861b99b91d59ee25d2790890bbd3ac6b5a0be",
        "fe1c0ffa23209796269b9f47e1e5a5d281a59229b81132f999da226f5bef7f77", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 25,
         "provider_request_id": "637102d2-1bbe-40b4-952b-88f7fd4b3dd8", "model": "gpt-6.2-astra",
         "response_id": "resp_088293b7705ca974016b083d3a738887d0b2d4e01c94eb52cf",
         "conflicting_response_id": "resp_637102d21bbe40b4952b88f7fd4b3dd8", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"}),
))


SCOPES[(12203, 15)] = replace(SCOPES[(12203, 14)], wave=15, cooldown_field="failed_unix",
                               cooldown_unix=1795705935.3553925, original_boundary_physical_attempts=2,
                               boundary_physical_attempts=1, archive=(
    *SCOPES[(12203, 14)].archive,
    Review(retry.bounded.RunIdentity("-12203-relay-retry-14",
        "158dcfc081b4f75dc2aabf6296a1d07440fe3680ab278b70c5aa51adec633ef0",
        "443fba215aebbbc0e6170e6d2b9ab8b37c8298e795d2d3e7332333ce5f0e2adf"),
        "529d87bb7b93bcfea2b85466ecaafecaf802549c63ef9eb23cce67ea4ce0b17b", "fdb4baee7c5b4f0e90eb85782b464612",
        "dff0b37c70e9cbbb39d3b89d031f5e1529b329a7692b002a6caf15524c7bc093",
        "97ae0af3ca4b8fc33467947e299d99c8afff6d432874e8c31f1784dd83f6f380",
        "70fc3592a2dd242f1fa7d5a105ccd4ff965a181c2badc379a27bf5863d4e2f6e",
        "ba07dcbd07341c90a9e971277f49f4ce116e6c7c171b5b2167425c152c8cbc89", "timeout",
        {"http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
         "event_type": "response.output_item.added", "sse_event_count": 15, "last_sse_seconds": 153.12645345786586,
         "provider_request_id": "9309d0ca-9a4f-4459-a16b-9c2a037f642e", "model": "gpt-6.2-astra",
         "response_id": "resp_002c70521cd2c530016b0848517b9887d0a373425c8bb727f2"}),
))


SCOPES[(12203, 16)] = replace(SCOPES[(12203, 15)], wave=16, cooldown_field="not_before_unix",
                               cooldown_unix=1795707786.662386, original_boundary_physical_attempts=1,
                               boundary_physical_attempts=2, archive=(
    *SCOPES[(12203, 15)].archive,
    Review(retry.bounded.RunIdentity("-12203-relay-retry-15",
        "30941667c357c8f3e11342f7e6044629721d060a7b6b8c0e5d1933f0b16c75d1",
        "846fc429706ceff7454ecb653b171683262a989a2e664480c0954ef7d521eb30"),
        "3ea5066c06c0fe613d6b1d312734aab65e9f38f0224dbe04d10e99537ff50ee9", "4fdcc2c2a5b9469eb0e0eb0124b7ad0e",
        "4426c7dc986c68654f3ee8976fb1fe80a0db9821ecd5249e399c531ec13591a0",
        "be11cf9bedb53a7b00a71eb263cf98783714c0259e6e63c02429fc6868c8765e",
        "283c19e27fd67795e3a53aec8f012028aca447615fb2276d7735047727fffa74",
        "2258c99051d47806e1f6b54f48d1e7289fedcb77b1a1a865e7881db1f6ff43c6", "timeout",
        {"http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
         "event_type": "response.output_item.done", "sse_event_count": 8, "last_sse_seconds": 568.9974320828915,
         "provider_request_id": "2cf1bb57-c0c2-4841-adc8-36cfcb7c15b2", "model": "gpt-6.2-astra",
         "response_id": "resp_005ad517c01224af016b084de4458887d0a9fa543067dee748"}),
))


SCOPES[(12203, 17)] = replace(SCOPES[(12203, 16)], wave=17, cooldown_field="failed_unix",
                               cooldown_unix=1795709731.0657685, original_boundary_physical_attempts=2,
                               boundary_physical_attempts=1, archive=(
    *SCOPES[(12203, 16)].archive,
    Review(retry.bounded.RunIdentity("-12203-relay-retry-16",
        "a6bfa7d6fa6c82cdf000aa859699ea37fe80c970fc762b0ea22dba545d4a5aa7",
        "f49406db892e95b4e85f2d6fd5061238fd42b6fdf72501ffc167e47861bbc02b"),
        "6c6b7c124f80b82a41809003d7fd7b3b2dc6b8fb3bc3e738b1daaa3e24d1624c", "2b0e526441584843bac2a9a185a5f4fe",
        "c42744c3574b7c569850e860b0341ab331fd64cb2c222937f7f0d75197a10539",
        "763787be79a67319d1da98ba84644d7d12f9bc5e33478fea5f97cd30dc424678",
        "d408a77082c3af6bdca7dd6d253a2a263d280e9b8257f12d15f7d024e6192ad0",
        "e7b7ea9a15ec61d758bc1c7690b9504050d809dffc60e345e076b9d12584e116", "connection_interrupted",
        {"http_status": None, "phase": "transport_failure", "error_type": "ReadError",
         "event_type": "ipv4_preflight_http_401", "provider_request_id": None, "model": None,
         "response_id": None, "sse_event_count": None}),
))


SCOPES[(12203, 18)] = replace(SCOPES[(12203, 17)], wave=18, cooldown_field="not_before_unix",
                               cooldown_unix=1795711802.841558, original_boundary_physical_attempts=1,
                               boundary_physical_attempts=2, archive=(
    *SCOPES[(12203, 17)].archive,
    Review(retry.bounded.RunIdentity("-12203-relay-retry-17",
        "9d68d3cd05cdfb725c5cfbad68474154748bcff75f06b7cf40c9463980c8921a",
        "211b469f82f130d714bc3ccdaf8ab3ae3d116237303b81918bf13c6cb33419ba"),
        "b67d3712badb2793ef1d60eb765532df03c73018896561bebc491d58d7984b93", "3ef3bfee813c47acb717ba512634e87a",
        "2992c3d73abfc1851e774777200b3d9f8e5db41adaa7613281e31c9ccc88a57f",
        "914807619b3dcfa3d2ef9c04a6232a08e135708b9f314960a0d72a71e0135eed",
        "22ed13565ffdbaf4138e5d320e7f00f47439c3b255fa82ba4366f80a8896d289",
        "2dc92d87f3e6cc06b8cf9960c443170a9098a49d88b469bd3f5de183631e7fd2", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch",
         "event_type": "response.output_item.added", "sse_event_count": 25, "last_sse_seconds": 211.33261599997059,
         "provider_request_id": "020886eb-e25b-468a-98e8-a5b6da9b33f9", "model": "gpt-6.2-astra",
         "response_id": "resp_0eca49db69bee932016b08625d14ec87d28bb2cef15a58ae8f",
         "conflicting_response_id": "resp_020886ebe25b468a98e8a5b6da9b33f9", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"}),
))


_12203_WAVE18 = retry.bounded.RunIdentity("-12203-relay-retry-18",
    "63922d96bc95ece68bdb8b2789370d20966e9a6cfd60dde903f6c4d3c1f5a37f",
    "4539f6fe8b51e1595907e023adf0e12de638c7c94d433bc5f58ce6493879fa0a")
_12203_WAVE18_REQUEST = "311a66cacc49616d2d96466027ea69d5663e28190a7d0b8d6d089fb55264f26a"
_12203_WAVE18_FIRST = Review(_12203_WAVE18, _12203_WAVE18_REQUEST, "0d93c0e153a34a7b8463363771c657c1",
    "565d4b8e3d2779a85bd55be27ba770ddcf982eb29d58096aba2d84291091c02c",
    "fe7a3230fa1739c0e691e831fca0fcd26b240fc0fb5f994062580ea24db72221",
    "a0aa0b391806467a3b91e2e14b25e2244e0e1a77e46c3a310f7837cce36b34fd",
    "1c6e42505edc2485605ea01a2f6397ed41f2dec246a987aa1d736657224248bf", "response_id_mismatch",
    {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch",
     "event_type": "response.output_item.added", "sse_event_count": 41,
     "provider_request_id": "1f13c457-fd1c-4a60-b4c6-148b2254e541", "model": "gpt-6.2-astra",
     "response_id": "resp_0ab272bbaad6aeda016b0866f3d39087d283049a941ed46d88",
     "conflicting_response_id": "resp_1f13c457fd1c4a60b4c6148b2254e541", "conflicting_event_type": "response.failed",
     "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"})
SCOPES[(12203, 19)] = replace(SCOPES[(12203, 18)], wave=19, cooldown_field="failed_unix",
                               cooldown_unix=1795714917.1493309, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(12203, 18)].archive,
    Review(_12203_WAVE18, _12203_WAVE18_REQUEST, "d95e1a4f686a466ebff67c01c09bdd9f",
        "1729d2b6cb97884274cef553d3917ff24416da6e00a906a50fa99e90ebfe89c5",
        "ccaea91b793a3a7e32afe7d2410df226ca6c2522353446ff83487340ffcd5484",
        "dcaa7599809f0e2971402549917dfce8da3d65aa3c9a0561b20cbacc9dbee176",
        "e3aa3de2fff7e7de647f48119e9e559f9a7244dd31bcbe8eb447822fffe86360", "timeout",
        {"http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
         "event_type": "response.output_item.done", "sse_event_count": 40,
         "last_sse_seconds": 601.0424845840316, "last_sse_activity_seconds": 601.0424845840316,
         "elapsed_seconds": 1501.0500557499472, "provider_request_id": "0308e63a-cc9d-4443-ac78-ada08c120344",
         "model": "gpt-6.2-astra", "response_id": "resp_0a6f786002d26b85016b08698f43fc87d2977526038f2f2c8f"},
        earlier=(_12203_WAVE18_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "c6519b97bc46d29e6bb21a05b66fa34a9eea6194fad5b1516a6e25ce6e1f505a",
            "attempt_00.replay.terminal.json": "97066490432870b2c57a338195df9329d3f918966ec82e15df4faaec9f25e655",
            "attempt_01.replay_link.json": "1c91a704844194e69d091d9a5b3db1f3235d026765a2f5aa42807db2e6e71a7d",
        }),
))


_12203_WAVE19 = retry.bounded.RunIdentity("-12203-relay-retry-19",
    "49bb4ef1565ed0ae87249701e9079728d6945371a43ca230814d6d54b57f3f43",
    "77b2d1e58c7a53b7a410a1906222b495b752787328af7d83f7a024e30f29fc94")
_12203_WAVE19_REQUEST = "80240ab888f288dec5cbf7b4abec27cb48e7d47615902c72e3a6512f07079ad7"
_12203_WAVE19_FIRST = Review(_12203_WAVE19, _12203_WAVE19_REQUEST, "b57d782d8ffd45edb332d4dd612e0197",
    "ed465796ba7dda913c4b22f901b5f09e1e40a7f18c22c1a1b459ab89182f1964",
    "9a51721e86002bce279451db544d4f044dd97b38e00396e1213bd810e4285e80",
    "6c1b0b897e9505bb575d167458e220d116c80f7228cd173972750c2799ccf8f1",
    "7638c5edd0eaa28f3eb30ede5f51f738ce3e43e808e7281ffcfababa5fbcb264", "response_id_mismatch",
    {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 25,
     "provider_request_id": "cbae14ff-040c-429a-bdf1-fe2470c1cf75", "model": "gpt-6.2-astra",
     "response_id": "resp_0c340402ddb40e23016b087788efe487d1964b99018ce642ea",
     "conflicting_response_id": "resp_cbae14ff040c429abdf1fe2470c1cf75", "conflicting_event_type": "response.failed",
     "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"})
SCOPES[(12203, 20)] = replace(SCOPES[(12203, 19)], wave=20, cooldown_field="not_before_unix",
                               cooldown_unix=1795717610.025714, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(12203, 19)].archive,
    Review(_12203_WAVE19, _12203_WAVE19_REQUEST, "7fa11bb380bd4d409f4f85c69df814b3",
        "d4c2208ae015ce72d14b466b481f1f909ceab81b38b7099f8bccd676ff8a35f8",
        "1918e76accd66ae2e30b0c6556ba94601e9add98713d0a9230bee88246df82b0",
        "37c1c267b7aa48726c180d910d828f61475b243024a7a7e40fa349855a6661c4",
        "a9d63d20ec106348db4c2770be209270da92d99a837c789753fb851aa49d87cc", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 41,
         "provider_request_id": "94641705-dd6f-4d42-a2c7-f13b78fd24c4", "model": "gpt-6.2-astra",
         "response_id": "resp_0758128f78ca6ba4016b08791cd33087d2830e3fd378fe2c3f",
         "conflicting_response_id": "resp_94641705dd6f4d42a2c7f13b78fd24c4", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"},
        earlier=(_12203_WAVE19_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "2ea77975157a68fd1d1f609420b6b389f8769263be4dcaa03eb6f52003246b18",
            "attempt_00.replay.terminal.json": "5b716a384dde6aaec7d77c95f504547f701271ba66f760e0cee06cba0f4212a6",
            "attempt_01.replay_link.json": "a5a2c9081e7519df6901789588902cf128888c8e8761494a4d2ad97f04a4d4a1",
        }),
))


_12203_WAVE20 = retry.bounded.RunIdentity("-12203-relay-retry-20",
    "a3eec33906e719e0c25a5a070411640d5ff1b00a99738484a66b5fdaea833d55",
    "901f6ad61d59f658a401cd7acccd0612a3c15e3b4c789562603a50e9e4b64f16")
_12203_WAVE20_REQUEST = "ba765e1b6d6bd90cab93e314cbda5dd40463e06708c6123be8674c25c1ed4d5d"
_12203_WAVE20_FIRST = Review(_12203_WAVE20, _12203_WAVE20_REQUEST, "c234a16437b24f02aeb25f9880152e66",
    "dbb6e4cefa02fd600e4c3783736d00edc98a6c5e6dd626931c4955bdcaddd088",
    "3bd7f01a20c518f20317b3f81ede50c3cd2856a42a6f9d311241035128867830",
    "3411c110316f7facfb777f875535a93372a7490c9764c891f271d5e3d480b87f",
    "afe562ac94772cec10b1b369cef5397be1f924bbc5baa0dc449b3e6f0d74284d", "response_id_mismatch",
    {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 25,
     "provider_request_id": "776a7ecb-2be5-4110-b438-7f55f808e9b4", "model": "gpt-6.2-astra",
     "response_id": "resp_0ac7de51fdc06fbd016b087be8cd0887d1bcdf01665fa402f0",
     "conflicting_response_id": "resp_776a7ecb2be54110b4387f55f808e9b4", "conflicting_event_type": "response.failed",
     "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"})
SCOPES[(12203, 21)] = replace(SCOPES[(12203, 20)], wave=21, cooldown_field="not_before_unix",
                               cooldown_unix=1795718766.3791935, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(12203, 20)].archive,
    Review(_12203_WAVE20, _12203_WAVE20_REQUEST, "2d6754e26cc847dcb6c4dba057646ecd",
        "01f4db72e3d15371b541f444193332587d689fcdcba082d3fc4e676fef84a50e",
        "116b5bb4990ccc67e0d804c751ad2429e77b200908dc1c409389140d19e6c84f",
        "2e3f94479dc7028ed018fc82e7cf4964e4f02ec134887082a5758a3e14576cb3",
        "1c41810acd031369902531dd1a22b6bd3fcbe13a297cd2896940ba2f6efc777c", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch", "sse_event_count": 25,
         "provider_request_id": "9140242f-2d65-4f15-901a-f0d8fc0ef8e3", "model": "gpt-6.2-astra",
         "response_id": "resp_0515e086da96efaf016b087d952d9087d18f4843fb7407898e",
         "conflicting_response_id": "resp_9140242f2d654f15901af0d8fc0ef8e3", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error"},
        earlier=(_12203_WAVE20_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "0962610faf4ae8efad79e4477bd8ef16604ef4bac6b4fc20162a3b9dce3f4794",
            "attempt_00.replay.terminal.json": "7d641c0c581a91638a1d517ef9e8028bd4f826f19398bf79585d383f416b2520",
            "attempt_01.replay_link.json": "b15b845a350c55ced28d0d1f08c1f263828a7a72dfa8797031f0bec47d44c38a",
        }),
))


_12203_WAVE21 = retry.bounded.RunIdentity("-12203-relay-retry-21",
    "70b7fb578ac41d9aac9bc36d651057ef61fd234398c73b207d97dbc469649033",
    "1a8d3d7e6719e79441ca3e2fe0c01054ea378af1ca63446a7b6e39d42a51dce3")
_12203_WAVE21_REQUEST = "473e3e929c6548cff9b2f6391a35e7bec82570feb4ce781b770c5273c8b3fa31"
_12203_WAVE21_FIRST = Review(_12203_WAVE21, _12203_WAVE21_REQUEST, "0dd696b221124af1b6c7b8e9c2c801fd",
    "9c77ce93efb842b5df0ad2f6ae18a8655709f92aaff116afcf39850b786eb0ca",
    "074f3aa259ae089e25009f6ba1d109600b8f5c2486a77175e8599133ef0cb90e",
    "c90e7e2da595681f809b2fc97f7e89b0c65e974282f3106f7f3d93bca7c94534",
    "3ba3cb832181a96d61f240712af9182b94049b47cf635eb9bd24d0f7e98c32f6", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'sse_event_count': 41,
     'model': 'gpt-6.2-astra',
     'provider_request_id': '41c1ff16-69b6-4cd1-ad5b-91156e742776',
     'response_id': 'resp_0cc2fb77d38f22d5016b087ff3236887d2843e6fa79f320cf5',
     'conflicting_response_id': 'resp_41c1ff1669b64cd1ad5b91156e742776',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 197.95120687503368,
     'elapsed_seconds': 200.37845779187046,
     'sse_milestones': {'response.created': {'elapsed_seconds': 20.045916749862954, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 20.046707042027265,
                                                 'interval_seconds': 0.000790292164310813},
                        'response.output_item.added': {'elapsed_seconds': 20.04751633387059,
                                                       'interval_seconds': 0.0008092918433248997},
                        'response.output_item.done': {'elapsed_seconds': 29.884481333894655,
                                                      'interval_seconds': 9.836965000024065}}})

SCOPES[(12203, 22)] = replace(SCOPES[(12203, 21)], wave=22, cooldown_field="not_before_unix",
                              cooldown_unix=1795719758.7627635,
                              original_boundary_physical_attempts=2, archive=(
    *SCOPES[(12203, 21)].archive,
    Review(_12203_WAVE21, _12203_WAVE21_REQUEST, "53851a8fce95464fbadddeb5b692cf7a",
        "a7b34485b706fee8b1cf5111f2a640516f43b959a47dc368d8a03efd5418376e",
        "7c19370a27531331a705657cfa05b550f9c88f47404b2dd6bbb774bc64a93223",
        "fa5729da7f1920987643fcc497f0a1e47e3d056b5669cae0e29d8dfdf82ab24d",
        "147b8c428f6ba4666ae0c8155773f1bc9b972e81858f2a931ed34c3c42eb48ee", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'sse_event_count': 41,
         'model': 'gpt-6.2-astra',
         'provider_request_id': 'fd79d365-c0a9-476e-af80-267871d433b1',
         'response_id': 'resp_00cffacbbc637616016b08818304bc87d2baca34bef23ebeef',
         'conflicting_response_id': 'resp_fd79d365c0a9476eaf80267871d433b1',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 196.87378316605464,
         'elapsed_seconds': 199.54550095787272,
         'sse_milestones': {'response.created': {'elapsed_seconds': 19.58489670790732, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 19.587101500015706,
                                                     'interval_seconds': 0.002204792108386755},
                            'response.output_item.added': {'elapsed_seconds': 19.589333874871954,
                                                           'interval_seconds': 0.002232374856248498},
                            'response.output_item.done': {'elapsed_seconds': 28.865945207886398,
                                                          'interval_seconds': 9.276611333014444}}},
        earlier=(_12203_WAVE21_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "f0012e47ceef357718497542e59fc68cc6b00f598df51790680dffebe4bd1891",
            "attempt_00.replay.terminal.json": "4e034216f36a394bac49a2fe1b3631d5f4fdadfcf0cdc29a54bb9f470b2e1877",
            "attempt_01.replay_link.json": "50812e27152b546a898b76eacd26ec6805a324780433fcc993b938ab69a66422",
        }),
))


_12203_WAVE22 = retry.bounded.RunIdentity("-12203-relay-retry-22",
    "e885e39585f247f0a629c94d5e2ef77debdd441e66e2d3eb24f23eaa3b838686",
    "9c7ad4e88455e2d0daa75ef3ed9b90a93bf17d1484327715bd543608e6f70ed3")
_12203_WAVE22_REQUEST = "3ac31ab1eeb9cddc94cc7ce1962c1981d6ca79e23c7ee1dc7ca03b217bd1a2e7"
_12203_WAVE22_FIRST = Review(_12203_WAVE22, _12203_WAVE22_REQUEST, "5c8e1aa9a3e2468489d5e2f17c9cd061",
    "940bae9dbc20aff9334fd4b054ac20b8ad698b1cec672b05004320d8b2677959",
    "9662c638c5c230381125a3156e8d93659f90af3edf35a34706c1d0730bad5fe3",
    "8a18c7fa771db6f429a3e1cb27afdd9f6087f6ae3b2ff8a15e9e49a441292de8",
    "1eb0d0a637985363ceab37f471108955fb9880ae9e1f3e7fe0cc2d002238a6f9", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'sse_event_count': 41,
     'model': 'gpt-6.2-astra',
     'provider_request_id': '54d0c317-982f-4116-a993-5e7ef4bded28',
     'response_id': 'resp_0e914e0cefff57f6016b08887955a887d28cd393466c5a06b3',
     'conflicting_response_id': 'resp_54d0c317982f4116a9935e7ef4bded28',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 199.76960833300836,
     'elapsed_seconds': 202.8339555831626,
     'sse_milestones': {'response.created': {'elapsed_seconds': 22.611213167198002, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 22.61198191717267,
                                                 'interval_seconds': 0.000768749974668026},
                        'response.output_item.added': {'elapsed_seconds': 22.61282933317125,
                                                       'interval_seconds': 0.0008474159985780716},
                        'response.output_item.done': {'elapsed_seconds': 31.77621958311647,
                                                      'interval_seconds': 9.163390249945223}}})

SCOPES[(12203, 23)] = replace(SCOPES[(12203, 22)], wave=23, cooldown_field="not_before_unix",
                              cooldown_unix=1795721957.2408683, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(12203, 22)].archive,
    Review(_12203_WAVE22, _12203_WAVE22_REQUEST, "df7c651d002142829aea8c19d2be8af3",
        "7e3f33772b9ac8ea236ba9b41864f8c3d5114e4123063c384de32826205d2c4b",
        "4122a448f580ea72877de7680f72e3957d881069be8e3ae1f50ee61c46f6bb42",
        "db1ee53749d5ac595233d9c1efe14a91971e2cd3ca3f26ed7cf837a3fb655c77",
        "a4a35e2a988b993bc530f3c46778e70f6a14f7539272030129ad56e0ebec1a94", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'sse_event_count': 25,
         'model': 'gpt-6.2-astra',
         'provider_request_id': 'b39bc58f-4bf3-4457-b0ea-d92f4b8b2cf4',
         'response_id': 'resp_071b06bd17676762016b088a08fe4c87d18f1b3c7bac30eb66',
         'conflicting_response_id': 'resp_b39bc58f4bf34457b0ead92f4b8b2cf4',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 201.229527708143,
         'elapsed_seconds': 210.57383850007318,
         'sse_milestones': {'response.created': {'elapsed_seconds': 30.770994333084673, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 30.77340200007893,
                                                     'interval_seconds': 0.0024076669942587614},
                            'response.output_item.added': {'elapsed_seconds': 30.77498145820573,
                                                           'interval_seconds': 0.0015794581267982721},
                            'response.output_item.done': {'elapsed_seconds': 46.84271379117854,
                                                          'interval_seconds': 16.06773233297281}}},
        earlier=(_12203_WAVE22_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "e889574ff2dc535f931a57b27a3b1a51f4bb54b05efb30b06b68f3e35145edc3",
            "attempt_00.replay.terminal.json": "3c80dcf3735537b8389b91c9f1364309dce997eda2a2ab83a308b723c8a2d80a",
            "attempt_01.replay_link.json": "7abd6c22e2d4b485dd8bc02bb2efe3229d1964489b6f81c6747e94e3518d2f92",
        }),
))


_12203_WAVE23 = retry.bounded.RunIdentity("-12203-relay-retry-23",
    "924ba82ee83a5280b2d10daa5f750e60521a7c88a4375c22ebdd767d669fc730",
    "b95ee9dfd7dcbad3c63a85b996baa5ff5588956e94c4c11124eb4530bbc252e0")
_12203_WAVE23_REQUEST = "f5ab73402adf522c9661d384c0d0ece42bf9169f6d4515228e78c1663e1d0e4b"
_12203_WAVE23_FIRST = Review(_12203_WAVE23, _12203_WAVE23_REQUEST, "e7dc3a17644e4204bd63ee32ca8b819c",
    "adc59e603656e8e4bc32d043b6dd41d1300f9ecf6689c3e8ce381ed0f732e825",
    "db0b86745315d5d93a01f2e30e777d9bdf6657692ac3d2569b1ccabb8453a450",
    "f612246bc9f3fae685948de3d4e64bda72cd8d728b8dd97a781415c0696a6a33",
    "fb370efa92982734174b6a808c008ec26f957df3707134e992e0a9fea75ffde9", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'sse_event_count': 41,
     'model': 'gpt-6.2-astra',
     'provider_request_id': 'd501696a-53e2-4207-866a-59494e3246bd',
     'response_id': 'resp_0fb7ea475bde9e0a016b088d65bac887d19a785d6314d4b85d',
     'conflicting_response_id': 'resp_d501696a53e24207866a59494e3246bd',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 198.6531495419331,
     'elapsed_seconds': 201.35397354187444,
     'sse_milestones': {'response.created': {'elapsed_seconds': 21.281937750056386, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 21.28287749993615,
                                                 'interval_seconds': 0.0009397498797625303},
                        'response.output_item.added': {'elapsed_seconds': 21.283713083947077,
                                                       'interval_seconds': 0.0008355840109288692},
                        'response.output_item.done': {'elapsed_seconds': 30.520349584054202,
                                                      'interval_seconds': 9.236636500107124}}})

SCOPES[(12203, 24)] = replace(SCOPES[(12203, 23)], wave=24, cooldown_field="not_before_unix",
                              cooldown_unix=1795723202.4180803, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(12203, 23)].archive,
    Review(_12203_WAVE23, _12203_WAVE23_REQUEST, "d50d43f67cbb4f0f814baa9122654bea",
        "b4a2b6621932cb7662ea8f35bfece28e9759e9325eaa1df55640190177c3f32e",
        "f2d7c318bc1b86345dcc40c4b9919d4749b91def1028736182ff899b44f1925a",
        "88fd1e2c33ac0d6b3e3c47841cd60e1393c9fc6d8a709efeb72996a6d71a9547",
        "8391ae3f4fbc5aa4f697fab9de3b7d97cf889eb4ab2779ac0f25cdf7f8e6e53b", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'sse_event_count': 41,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '6da7f03c-5c15-41d4-bf62-c23f0364062f',
         'response_id': 'resp_09e3d709ff4cb791016b088ef0e73487d1900550222ebb0039',
         'conflicting_response_id': 'resp_6da7f03c5c1541d4bf62c23f0364062f',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 195.0682915840298,
         'elapsed_seconds': 197.92816583416425,
         'sse_milestones': {'response.created': {'elapsed_seconds': 17.92737291706726, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 17.929158250102773,
                                                     'interval_seconds': 0.0017853330355137587},
                            'response.output_item.added': {'elapsed_seconds': 17.931698334170505,
                                                           'interval_seconds': 0.0025400840677320957},
                            'response.output_item.done': {'elapsed_seconds': 27.289097334025428,
                                                          'interval_seconds': 9.357398999854922}}},
        earlier=(_12203_WAVE23_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "ead692bafdaee8808fb994e94e67656decfd617ec0235c6ed5d91967033dcad9",
            "attempt_00.replay.terminal.json": "0ab8e8984dca2f573059713760d30000c466c873ef881b2d26f48e03a2604752",
            "attempt_01.replay_link.json": "9ba2232f2cd895cbb2438d0d7c0130b58c6ec6430948814365dbca75e0d47565",
        }),
))


_12203_WAVE24 = retry.bounded.RunIdentity("-12203-relay-retry-24",
    "ebb25c05bda831640eecaf6f7029d04a725c5d0da898b442b179288907e6f791",
    "5720b107ac7a29a6a9dab47acbbf3627ea3e0ed278d1b330f4b7fbcaa102030b")
_12203_WAVE24_REQUEST = "e40f78bc0bb14345f3c96ff1706a31c393692a55f8452d4bca3fedc3cf3905bf"
_12203_WAVE24_FIRST = Review(_12203_WAVE24, _12203_WAVE24_REQUEST, "8d53189d20c74d69a5b9ebb3729ac285",
    "abbba70b470f41bb4b87af3d566da910a7eb6e9924dc8c02e30e163614a130c3",
    "5c307ddd4a9bd05c8e8e1f4e32932c25039462ea48cae069a0aacf40e67e76c0",
    "55a3705a916a01da5ceccc283776377339466e9a2d8c564aa1f2c7e2a8f61124",
    "f2af42497005087db65b61b88f3675f98ad956269e71da1f3ad595738bc83cdc", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'event_type': 'response.output_item.added',
     'sse_event_count': 25,
     'model': 'gpt-6.2-astra',
     'provider_request_id': '0a42f763-fd20-4e34-9e72-d8f334f3c373',
     'response_id': 'resp_04696cd966c4e493016b08912d46c887d2b62bd1b44534ef79',
     'conflicting_response_id': 'resp_0a42f763fd204e349e72d8f334f3c373',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 208.36902737501077,
     'last_sse_activity_seconds': 208.36902737501077,
     'elapsed_seconds': 212.08428316679783,
     'sse_milestones': {'response.created': {'elapsed_seconds': 31.960310666821897, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 31.961317833978683,
                                                 'interval_seconds': 0.0010071671567857265},
                        'response.output_item.added': {'elapsed_seconds': 31.96216283389367,
                                                       'interval_seconds': 0.0008449999149888754},
                        'response.output_item.done': {'elapsed_seconds': 48.32804175000638,
                                                      'interval_seconds': 16.365878916112706}}})

SCOPES[(12203, 25)] = replace(SCOPES[(12203, 24)], wave=25, cooldown_field="not_before_unix",
                              cooldown_unix=1795725106.1941266, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(12203, 24)].archive,
    Review(_12203_WAVE24, _12203_WAVE24_REQUEST, "b31f16e0b5204612add655f1995cc71e",
        "47c0a777186cbf3a4a2dc3c43170df1af08b124775d1108681eedbb381c2b0cc",
        "5b6988c729e2d94a4d17170acffaf00c075487588745606108d50f4f96a564c1",
        "47042d1223569116bad111564a194f3b417c38aa615ae4bd282bde263c121a9b",
        "5d969fe6ce45782e175f9f21e4e932be0f59a08fe7d376bcbc5d1e4c14505e68", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'event_type': 'response.output_item.added',
         'sse_event_count': 41,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '86033026-fd7b-492b-9a50-f37a64ece320',
         'response_id': 'resp_057f1d06095cf713016b089664e6d487d1800531f4e5cadf57',
         'conflicting_response_id': 'resp_86033026fd7b492b9a50f37a64ece320',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 197.96779104112647,
         'last_sse_activity_seconds': 197.96779104112647,
         'elapsed_seconds': 200.7695372500457,
         'sse_milestones': {'response.created': {'elapsed_seconds': 20.84528554114513, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 20.84716658294201,
                                                     'interval_seconds': 0.0018810417968779802},
                            'response.output_item.added': {'elapsed_seconds': 20.848750666016713,
                                                           'interval_seconds': 0.0015840830747038126},
                            'response.output_item.done': {'elapsed_seconds': 30.206162833143026,
                                                          'interval_seconds': 9.357412167126313}}},
        earlier=(_12203_WAVE24_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "9ad1c6af3e496846554c3a4a0792dac068834c20483ccefa4f4e8cc7f9cfb88d",
            "attempt_00.replay.terminal.json": "c01400967a6ec414b28399074db948d21386a2a78e6de9afeee4be5af1a601fc",
            "attempt_01.replay_link.json": "93e5218bb272f42a304a5ca2ff3f928b11648450259ce65c6142f2fdae3a5578",
        }),
))


_12203_WAVE25 = retry.bounded.RunIdentity("-12203-relay-retry-25",
    "2bbf4ef9605cb96197f5bec605bc48ff066aad987d630867511eaedbd812051f",
    "2e0b9389fc5e539d09a480bc499c4baa3eb8ced57c99c0d4260f1f01205762eb")
_12203_WAVE25_REQUEST = "97980a219599e17c10f77e905b59e62515bb21ac46ee0182faae2d6f959f0dcc"
_12203_WAVE25_FIRST = Review(_12203_WAVE25, _12203_WAVE25_REQUEST, "e6f1d4b3ee7145cc91dbd2970967f83a",
    "07d623855102fab8bf4252d55115957313cc185e734615fc3d61193e6c7ecf95",
    "afe709092017a41347f83b02677c5e9ce80bc2a610cd4cdbe03e158ce8efe14b",
    "21780831e37426a092e2b5aea1533fd4facc1a0bdee362f91612450e6f70a4c0",
    "f05f4dfe705a245a336d08fd2b6127bffd0a584cb71f9993991fe1f983148fc3", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'event_type': 'response.output_item.added',
     'sse_event_count': 41,
     'model': 'gpt-6.2-astra',
     'provider_request_id': '95025066-bc38-45d5-8d80-273a927529fc',
     'response_id': 'resp_025f01d1571ec249016b08990a70ec87d2a565b6539ef7b2af',
     'conflicting_response_id': 'resp_95025066bc3845d58d80273a927529fc',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 196.94270929088816,
     'last_sse_activity_seconds': 196.94270929088816,
     'elapsed_seconds': 200.10959616606124,
     'sse_milestones': {'response.created': {'elapsed_seconds': 20.015231000026688, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 20.016195791075006,
                                                 'interval_seconds': 0.0009647910483181477},
                        'response.output_item.added': {'elapsed_seconds': 20.017092790920287,
                                                       'interval_seconds': 0.0008969998452812433},
                        'response.output_item.done': {'elapsed_seconds': 29.148797624977306,
                                                      'interval_seconds': 9.131704834057018}}})

SCOPES[(12203, 26)] = replace(SCOPES[(12203, 25)], wave=26, cooldown_field="not_before_unix",
                              cooldown_unix=1795726193.246844, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(12203, 25)].archive,
    Review(_12203_WAVE25, _12203_WAVE25_REQUEST, "ed84d3db2fe14fb08ceeee420241e445",
        "12bd4313d1c87e06136ff48e6494b59ec0ef28997263d8dcee40578b0cb5c0ff",
        "f07a8646b7c797459c7bb905e0c09fe4450179272b117f0ad334d7a38186b467",
        "747c239aa8a093bc86a3776c6a76e084dc74666414df497052c973acb305a097",
        "98b3a95c6997af60b42cf3230547354c2075632719f9cc5af837bd65dbda7759", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'event_type': 'response.output_item.added',
         'sse_event_count': 25,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '4b176877-fef6-43a4-9976-a36ee210a947',
         'response_id': 'resp_0725e99f1a22f3b0016b089a9a8e9887d2a5d8c1aa0e810611',
         'conflicting_response_id': 'resp_4b176877fef643a49976a36ee210a947',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 204.1329087088816,
         'last_sse_activity_seconds': 204.1329087088816,
         'elapsed_seconds': 210.04968970897608,
         'sse_milestones': {'response.created': {'elapsed_seconds': 30.04070670902729, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 30.04346466693096,
                                                     'interval_seconds': 0.0027579579036682844},
                            'response.output_item.added': {'elapsed_seconds': 30.045470041921362,
                                                           'interval_seconds': 0.002005374990403652},
                            'response.output_item.done': {'elapsed_seconds': 47.67604554188438,
                                                          'interval_seconds': 17.630575499963015}}},
        earlier=(_12203_WAVE25_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "d7569b90c3b89161f41093704cfc1a9c56133f6d0db1c445846353b65c1cd6b5",
            "attempt_00.replay.terminal.json": "2830a89082370f6d311a2eb43a95f0475a92e6c427fa5a69b2666de7f4446bbe",
            "attempt_01.replay_link.json": "dda02620b82c8f2321f0b5d9f0141bce9e61bf36776537b598ec3d2087d20374",
        }),
))


_12203_WAVE26 = retry.bounded.RunIdentity("-12203-relay-retry-26",
    "670ed3f7438fe0e02ec0bdec3deb08f23cd5ed4841a2f56f19a61b745f483a2c",
    "92ec7ceefa9e7d81c9e7ffc4a8421e3f528eba5a8d396be92f57530f3c1c7e70")
_12203_WAVE26_REQUEST = "90740618254a45e9461f5ee948fc9209576ebfefae755c405a094905e26bf503"
_12203_WAVE26_FIRST = Review(_12203_WAVE26, _12203_WAVE26_REQUEST, "f1f7f01659bc42fd9b652cb05740bf2b",
    "73772f5964e83762bfdaa96807d2acbbf2327b9578d05af0d8116b825d8bfe5f",
    "2b85016e87707a5f65408de237bfdadc108a38264300554c39b5a18cebb8c1fb",
    "b297d490ae2ecd8e1999e8bd72d63001b33b4bda4516512daf7af12a274c81e9",
    "e39453563ef7e62a34e1648cb0ef20d507e4b33539ff808afa3efc106f6911fa", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'event_type': 'response.output_item.added',
     'sse_event_count': 25,
     'model': 'gpt-6.2-astra',
     'provider_request_id': '3300fd09-c20d-480e-9454-aba347a6af0b',
     'response_id': 'resp_0f8c1ef3bed38381016b089d27800087d1af72d48d47e95958',
     'conflicting_response_id': 'resp_3300fd09c20d480e9454aba347a6af0b',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 202.71854716609232,
     'last_sse_activity_seconds': 202.71854716609232,
     'elapsed_seconds': 209.4571649581194,
     'sse_milestones': {'response.created': {'elapsed_seconds': 29.274646833073348, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 29.275661958148703,
                                                 'interval_seconds': 0.0010151250753551722},
                        'response.output_item.added': {'elapsed_seconds': 29.27650783304125,
                                                       'interval_seconds': 0.0008458748925477266},
                        'response.output_item.done': {'elapsed_seconds': 45.8307475829497,
                                                      'interval_seconds': 16.554239749908447}}})

SCOPES[(12203, 27)] = replace(SCOPES[(12203, 26)], wave=27, cooldown_field="not_before_unix",
                              cooldown_unix=1795727260.780899, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(12203, 26)].archive,
    Review(_12203_WAVE26, _12203_WAVE26_REQUEST, "bfd2c04a4900472388c84af214f5e72d",
        "7f3c732815fddc7775169e5b19d1f89ccf0cf7cf98ce7e9b4c86693dd17f1fcb",
        "c6ee950870297765ed537da9fa50582e118eb317378236799cb954d1b8ad62ee",
        "6fd09a8819a1796998b05e14ac1d7a743bad6671fe46dd22e144943f80ee05c7",
        "c4beb80f40eab4afb15744a8371d14f043d2d2679a4f0233d1534a6db0b0c4ec", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'event_type': 'response.output_item.added',
         'sse_event_count': 25,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '0f30e10d-3317-47ca-8cc1-7a3c2e8bdd7a',
         'response_id': 'resp_022042e8bda5f6db016b089ec8496487d1a5709ffbdcd1a899',
         'conflicting_response_id': 'resp_0f30e10d331747ca8cc17a3c2e8bdd7a',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 196.61737745790742,
         'last_sse_activity_seconds': 196.61737745790742,
         'elapsed_seconds': 207.03115104185417,
         'sse_milestones': {'response.created': {'elapsed_seconds': 26.975886582862586, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 26.977500916924328,
                                                     'interval_seconds': 0.001614334061741829},
                            'response.output_item.added': {'elapsed_seconds': 26.978995874989778,
                                                           'interval_seconds': 0.0014949580654501915},
                            'response.output_item.done': {'elapsed_seconds': 41.70412016683258,
                                                          'interval_seconds': 14.725124291842803}}},
        earlier=(_12203_WAVE26_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "7ec65254bf7ba81d5c7f426ebaabad70ec272900cd44ea1d41b082c8aba509c0",
            "attempt_00.replay.terminal.json": "896ae33f5ffde7798213fd4b5b1cc097798388cb2ab8c73bbba7c1e32562ee58",
            "attempt_01.replay_link.json": "4c1942baf9fd5fd66e57e1f5520ed8348f055a08ef744600dae85dcd52b55cf1",
        }),
))


_12203_WAVE27 = retry.bounded.RunIdentity("-12203-relay-retry-27",
    "44fcaaafc7c8083456cb7fb20105ccb81f5b8b63eba0defd77d3c1660ee5131f",
    "a1b4a0e28d1758ebdfe2037d0895377706de93aa0eec8b071535e9121d6002c0")
_12203_WAVE27_REQUEST = "8dfa3871d1dd0ceb5228520eb0ae6745ac13eb164105f105e0fad3f414013cee"
_12203_WAVE27_FIRST = Review(_12203_WAVE27, _12203_WAVE27_REQUEST, "a2f3f4d289f348b287cc243e2735d113",
    "a73080e3e79d533ea8489353b204a779b564a2b57599c935c30b79d24e252e67",
    "adedafe461242a8a9b47e98ad7fefd85372249f65b6228b844100a7ef7d84584",
    "d106da71ec687bc7a581eee9ebe129fdacbe6dd955222639bf7069503f5128f3",
    "2941f20b17ebcdac7b2a07938adadafd08c50d32fec3c3422cdc1fec45dd76a3", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'event_type': 'response.output_item.added',
     'sse_event_count': 41,
     'model': 'gpt-6.2-astra',
     'provider_request_id': '18190c2d-2995-4fa9-a24f-f9571afda1b1',
     'response_id': 'resp_0d0edc845f350279016b08a1a96ce887d285aa5bda62c7ff0d',
     'conflicting_response_id': 'resp_18190c2d29954fa9a24ff9571afda1b1',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 195.55153283313848,
     'last_sse_activity_seconds': 195.55153283313848,
     'elapsed_seconds': 198.39667770802043,
     'sse_milestones': {'response.created': {'elapsed_seconds': 18.325522291939706, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 18.326417500153184,
                                                 'interval_seconds': 0.0008952082134783268},
                        'response.output_item.added': {'elapsed_seconds': 18.32734933309257,
                                                       'interval_seconds': 0.0009318329393863678},
                        'response.output_item.done': {'elapsed_seconds': 27.514876832952723,
                                                      'interval_seconds': 9.187527499860153}}})

SCOPES[(12203, 28)] = replace(SCOPES[(12203, 27)], wave=28, cooldown_field="not_before_unix",
                              cooldown_unix=1795728393.4999516, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(12203, 27)].archive,
    Review(_12203_WAVE27, _12203_WAVE27_REQUEST, "8c5b0a47fdc840b3977f30d25dbcb061",
        "d07d9a3699207ac0b2402f4ac5cd6c4f8ef2caf84d86cdae59e09ea682c3d3be",
        "3e5756fa875c45810525fe259aee5499533e4f6c03ceca2ce91992729dd85326",
        "ed4e43d6f8253eb7f0211d239fc0e756d13dfb0e85dcefeb3bb14140fcbaa384",
        "fa0099360fbf50c2c2995aa32a8b1105ab32ba4043a0a9edbe1d048bc320fc6e", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'event_type': 'response.output_item.added',
         'sse_event_count': 25,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '1bf68c2c-1f8a-4830-a217-e4d4c1b2415c',
         'response_id': 'resp_0b900cbd2f8e0992016b08a335f92c87d0b81b0703126df851',
         'conflicting_response_id': 'resp_1bf68c2c1f8a4830a217e4d4c1b2415c',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 200.91070891707204,
         'last_sse_activity_seconds': 200.91070891707204,
         'elapsed_seconds': 208.21483075013384,
         'sse_milestones': {'response.created': {'elapsed_seconds': 28.216866417089477, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 28.218607207993045,
                                                     'interval_seconds': 0.0017407909035682678},
                            'response.output_item.added': {'elapsed_seconds': 28.220885083079338,
                                                           'interval_seconds': 0.0022778750862926245},
                            'response.output_item.done': {'elapsed_seconds': 44.96939670806751,
                                                          'interval_seconds': 16.74851162498817}}},
        earlier=(_12203_WAVE27_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "bde1eaa4e28db61dafd378cea2fb55e1c5d6b46c7c701cb668543d7d177bce30",
            "attempt_00.replay.terminal.json": "0544c1e458145155eb17a622d88e3422ddddbf73cfc9a728a6f1dcacc2796675",
            "attempt_01.replay_link.json": "67421cb45910f3bd575661247d05506ace0c36fbf5d3e463179b6b3e40e45672",
        }),
))


_12203_WAVE28 = retry.bounded.RunIdentity("-12203-relay-retry-28",
    "8f38ecdc0aaec090d23eeb9ed52024b6fd2e7dcce325b0d2d88e5ab05b7042e8",
    "e9e083b3eb9c4cfd44f8cb3a35f9d78be8349fc2e95021e9ac0be44ae14f77e3")
_12203_WAVE28_REQUEST = "6f298cc3b55eef06e1a679ac588ee7bbb17d8e3e83c453d960f8e84b7bcfdbbc"
_12203_WAVE28_FIRST = Review(_12203_WAVE28, _12203_WAVE28_REQUEST, "28549f09fb7243bc8938f9fa3437e1e4",
    "2df9575e48f00a43357d7581a5b3c5793f74486cfbeacf313d87ddbdd4ea874b",
    "0f229a6de810cd0ce1bb57639ed6b6391f6d6c17cc80b037a3f57435e26434a0",
    "ab736073f1f8eec5fdde734e2acd9eea470703b75ac25b74ba1b50a1db5c7099",
    "82bbd84d7bad156c70daf9d5e173b298f10db47c20d096d1251d7a722eac6f94", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'event_type': 'response.output_item.added',
     'sse_event_count': 25,
     'model': 'gpt-6.2-astra',
     'provider_request_id': '1ca42010-dea2-4c02-bcee-59289dcbca7b',
     'response_id': 'resp_0114899e81c583e3016b08a5ba08dc87d196911ae93d08d7b1',
     'conflicting_response_id': 'resp_1ca42010dea24c02bcee59289dcbca7b',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 201.30618937523104,
     'last_sse_activity_seconds': 201.30618937523104,
     'elapsed_seconds': 207.19607854215428,
     'sse_milestones': {'response.created': {'elapsed_seconds': 27.074091750197113, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 27.074800667120144,
                                                 'interval_seconds': 0.0007089169230312109},
                        'response.output_item.added': {'elapsed_seconds': 27.075751584023237,
                                                       'interval_seconds': 0.0009509169030934572},
                        'response.output_item.done': {'elapsed_seconds': 43.836752334143966,
                                                      'interval_seconds': 16.76100075012073}}})

SCOPES[(12203, 29)] = replace(SCOPES[(12203, 28)], wave=29, cooldown_field="not_before_unix",
                              cooldown_unix=1795729434.5429068, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(12203, 28)].archive,
    Review(_12203_WAVE28, _12203_WAVE28_REQUEST, "dd515d6606684aa3bd332499ceaf456e",
        "245c1679de45a59a6d11c9d75c1582cb516cd8306b7cb08d8c7083e7d1789b8b",
        "8ba60b7b135cf0033512f918ecdb73436017d16a516e6be9f1b9d052a3c44b2f",
        "76a55f0d9304bb5465cfb70b50f4a6b22e7e0256533fc6c4290a8235bdef4ae5",
        "64c94ca0854fde776105d03d778ff07a95121ebb9a005f6cb35bf0aff47cd7c9", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'event_type': 'response.output_item.added',
         'sse_event_count': 41,
         'model': 'gpt-6.2-astra',
         'provider_request_id': 'c096a548-d766-4bae-8d94-0fb2162161b2',
         'response_id': 'resp_0254d74ea878b7aa016b08a74f571487d29731b1d45ba95dee',
         'conflicting_response_id': 'resp_c096a548d7664bae8d940fb2162161b2',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 194.4747688327916,
         'last_sse_activity_seconds': 194.4747688327916,
         'elapsed_seconds': 197.2414221658837,
         'sse_milestones': {'response.created': {'elapsed_seconds': 17.30073512485251, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 17.303228582954034,
                                                     'interval_seconds': 0.0024934581015259027},
                            'response.output_item.added': {'elapsed_seconds': 17.305725457845256,
                                                           'interval_seconds': 0.0024968748912215233},
                            'response.output_item.done': {'elapsed_seconds': 26.645467040827498,
                                                          'interval_seconds': 9.339741582982242}}},
        earlier=(_12203_WAVE28_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "9d47d05d474e86a53f9e5922a808d2120cdeaa1782342004427b5e1bd5a545ad",
            "attempt_00.replay.terminal.json": "1adc9d264a4db81e2b97624b0338d38414eef25daddd3de8fafc2256913887df",
            "attempt_01.replay_link.json": "64397057e0c7f74054abb21dd5421b1d8246a3a8a6375c2391f54dcdd2598990",
        }),
))


_12203_WAVE29 = retry.bounded.RunIdentity("-12203-relay-retry-29",
    "41f51726abea8f9f41e12c46dd8f3a85b735558b38245cc3d56db971e5408302",
    "17182020d6bcaed7fe7eb1cd074d19ca3ed55766497f4e5b50e2c6260cc4d016")
_12203_WAVE29_REQUEST = "6b7ab741692c51fb04b666d58213b786ae52fa74f722a7fe888886ce6786457d"
_12203_WAVE29_FIRST = Review(_12203_WAVE29, _12203_WAVE29_REQUEST, "420185dd70764c478a8ca7bd2b202726",
    "0a3dc3a4accdebc897b717768792b8823e9a3d61563d79ad781e292c016e11ac",
    "8073f6746f74cc4ca3a1a137f6bbcc0e9072865ef61cba3d0c9a946b71c7b1b4",
    "a74dda302023589875dbdfa30e8e83aa3d16bd8f02313e21d239fd3f51c45947",
    "28ae6f1f1fb8d361210b4cdc2bf7085fa91df33184bd95478dfa0ec95c5762eb", "response_id_mismatch",
    {'http_status': 200,
     'phase': 'transport_failure',
     'error_type': 'response_id_mismatch',
     'event_type': 'response.output_item.added',
     'sse_event_count': 25,
     'model': 'gpt-6.2-astra',
     'provider_request_id': 'd3458f51-8f90-4154-b743-428450d33fb2',
     'response_id': 'resp_04908cc8e396528a016b08aa10d8b087d1863ee4660bc19954',
     'conflicting_response_id': 'resp_d3458f518f904154b743428450d33fb2',
     'conflicting_event_type': 'response.failed',
     'response_id_matches_transport_alias': True,
     'provider_error_code': 'upstream_error',
     'last_sse_seconds': 194.78144849999808,
     'last_sse_activity_seconds': 194.78144849999808,
     'elapsed_seconds': 208.881252292078,
     'sse_milestones': {'response.created': {'elapsed_seconds': 28.730306667042896, 'interval_seconds': None},
                        'response.in_progress': {'elapsed_seconds': 28.731097584124655,
                                                 'interval_seconds': 0.0007909170817583799},
                        'response.output_item.added': {'elapsed_seconds': 28.731932459166273,
                                                       'interval_seconds': 0.0008348750416189432},
                        'response.output_item.done': {'elapsed_seconds': 43.21316108410247,
                                                      'interval_seconds': 14.481228624936193}}})

SCOPES[(12203, 30)] = replace(SCOPES[(12203, 29)], wave=30, cooldown_field="not_before_unix",
                              cooldown_unix=1795730553.2632422, original_boundary_physical_attempts=2, archive=(
    *SCOPES[(12203, 29)].archive,
    Review(_12203_WAVE29, _12203_WAVE29_REQUEST, "75901ef837ec4280aa8ba975239f2066",
        "1add75757e5e324368de91dff8344ab19278c56da9c1ef5e3d32c9602e33321b",
        "de156ca538750c9262129978410a1214ed37a8fa6293761f5f1345b1bf096ecd",
        "c77bb0c5102f86a3f4efecf0f24e3fa94bf47188d91915108e4483a537343279",
        "fc62b9466cf6db88e4ea4565b5a10e4212b507963898778e5c6b096e3f519c90", "response_id_mismatch",
        {'http_status': 200,
         'phase': 'transport_failure',
         'error_type': 'response_id_mismatch',
         'event_type': 'response.output_item.added',
         'sse_event_count': 25,
         'model': 'gpt-6.2-astra',
         'provider_request_id': '3b8e9b05-66cc-4365-b04e-8f04d0c0ed88',
         'response_id': 'resp_0c0db30a37dc2254016b08aba29c3887d1b1bd12e0607cdf6b',
         'conflicting_response_id': 'resp_3b8e9b0566cc4365b04e8f04d0c0ed88',
         'conflicting_event_type': 'response.failed',
         'response_id_matches_transport_alias': True,
         'provider_error_code': 'upstream_error',
         'last_sse_seconds': 202.90926029207185,
         'last_sse_activity_seconds': 202.90926029207185,
         'elapsed_seconds': 209.08399891713634,
         'sse_milestones': {'response.created': {'elapsed_seconds': 29.08444350003265, 'interval_seconds': None},
                            'response.in_progress': {'elapsed_seconds': 29.08614725014195,
                                                     'interval_seconds': 0.0017037501093000174},
                            'response.output_item.added': {'elapsed_seconds': 29.088339167181402,
                                                           'interval_seconds': 0.0021919170394539833},
                            'response.output_item.done': {'elapsed_seconds': 44.03598566702567,
                                                          'interval_seconds': 14.947646499844268}}},
        earlier=(_12203_WAVE29_FIRST,), replay_files={
            "attempt_00.replay_authorization.json": "5ba40274fb33f67bea87b04c55133fd89acde42da659b96be8b601194522c28f",
            "attempt_00.replay.terminal.json": "28865419dcc582fb1651fa7fd45ce810b2b1114ab2e90eba98f725566862bdc9",
            "attempt_01.replay_link.json": "9d81d29356e39f11896fa03828c86cb4ccc4cdba41ee1166f361b604ef3ff588",
        }),
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
SCOPES[(887, 3)] = replace(SCOPES[(887, 2)], wave=3, cooldown_unix=1795687017.5816553,
                           original_start_round=None, fitted_prefix_files=(), archive=(
    *SCOPES[(887, 2)].archive,
    Review(retry.bounded.RunIdentity("-887-relay-retry-02",
        "a0e19b519b4cd3c96038563fb470f93ab5b5ff9d943aee4215292a256ba50f3e",
        "80c19a60bda6db98a5ec2f7b98051408b027b1dec7889b11f89a7288c3537252"),
        "33dd5e0f75cc3e5b932126505a2a2aae1bfd76fbb9f77394b1d1cb94815a4520", "844b413de8424d998374e86a31535b48",
        "b12ee1189f1d36bd35f1eccbfd6a80caf8962908864f0ba7438feb092798372e",
        "6b3284b78b60617b0f45befb1913d21f33ff20859122f2d75749f9a13be80719",
        "4d8908c38af866b174465006e0d65bcacd47a6d1ba85af658f0c76c9c038634d",
        "c3a801b2c92eca99e825e30f523e0025d7065fc775e8a73853a066b803ca9b92", "error",
        {"http_status": 200, "phase": "transport_failure", "event_type": "error", "error_type": "error",
         "provider_error_code": "upstream_stream_read_error", "sse_event_count": 26,
         "provider_request_id": "1d9ab1ff-4896-4cf8-9c86-512896829522",
         "response_id": "resp_048c6e89491c1847016b0801da3a9087d0b9f92cea1ca055a9"}),
))

SCOPES[(887, 4)] = replace(SCOPES[(887, 3)], wave=4, cooldown_unix=1795687864.4677405, archive=(
    *SCOPES[(887, 3)].archive,
    Review(retry.bounded.RunIdentity("-887-relay-retry-03",
        "9535b64e738bfcfb1479437c59e32f80207b16b3e7aaf2eeecbba4ac7c7bc26f",
        "75879126368c97351371c13b9b952fef24e9836079097a9e6de4d5512b91dc39"),
        "89b698f474ca01c573d836049270aaba70cd41145b02ea0e3459f104f530cc50", "35c5eb82fb794d78a637d5c75e4be90e",
        "2faf1fd49ad5967e912896851df88254a30c14466506ad9db6c24d3b41c7ca43",
        "933c0890ab7376093cd62abdfb0637682a1088be8be3398a0e4fba3fea7dd0b4",
        "152fb89f5861ffe113431e3449036265517697845d2df6c3e3ef53f353cfeed5",
        "73070cee42f232570bbe4a30a2cf83630b050e35917ba4a207f7f432517b0eb7", "response_id_mismatch",
        {"http_status": 200, "phase": "transport_failure", "error_type": "response_id_mismatch",
         "sse_event_count": 41, "provider_request_id": "0898714b-833f-46f6-a9d3-1cd9cfd0cd2c",
         "response_id": "resp_0d4ca1588c4db38d016b0804ed1f2887d2af7e53be8e31df23",
         "conflicting_response_id": "resp_0898714b833f46f6a9d31cd9cfd0cd2c", "conflicting_event_type": "response.failed",
         "response_id_matches_transport_alias": True}),
))


_887_R9_CONTINUATION = retry.bounded.RunIdentity("-887-r9-continuation-01",
    "58de5fc9e6a56844cc720404a3ceba8f8e7c8445c4f25290f175748a9d0c9d4f",
    "d32b845eb0e8cbfbab28c523f82acba853c1e8df345e492033c2458a879658cc")
_887_R12_CHECKPOINT = "68f0414b95ed886540b989083c5e1c58bc96304fb3178b117d8cde85e8182603"
SCOPES[(887, 5)] = Scope(
    887, 12, 5, "4064b0a2694a7d36ece124937bbc5c47b2c5e4e77b22b8384a2af998c15c5b38", (
        Review(_887_R9_CONTINUATION,
            "a53987367a8eddc7f4b0bf731e290388b42e1ade9b0c5c3416a5344f5021b2f7", "75647f5eaad9426d8f7cc13ebfa490b8",
            "80e6838f4e2bcfbc61ebb99a80e4712f982fef48cad8abe06ee1ea33d33ef7d8",
            "49fb3999e75a5ccc343a10ecfef40c7b796f971847b111f282cf2b1e3f771514",
            "8cde8a42b2fbfbb4c3602142add3f0ebfcdda7e58f8e3b47c53f90a8502f0ce9",
            "9c84d32abf0ca52029e6cb144a47255577d57e99984ea33698faa83674c79c76", "timeout",
            {"http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
             "event_type": "response.output_text.delta", "sse_event_count": 207, "last_sse_seconds": 961.4464893341064,
             "provider_request_id": None, "model": "gpt-6.2-astra",
             "response_id": "resp_0aceb6badc01f076016b08246221d087d2a30ef3e67e38261f"}),
    ), "failed_unix", 1795697557.9016292, original_start_round=9, fitted_prefix_files=(
        ("boundary.json", "b68021e483cd762af998caaeac7a5f03ff151a8f7331213a851765f2c25a67c0"),
        ("round_09/source.json", "0b33cbe7d235940cd51ea5f0a934ec34805c9ce7b10e7c6bc9844b1daadd1cbd"),
        ("round_09/boundary_memory.json", "f7dc25bf194a2538139d1caa566df9a64ff98a967d268132152c74782e55c2c6"),
        ("round_09/memory.json", "a18263a6de3410b4b1841e861b8d836c9dbc9e29d123c1f7cf6c379eee65337b"),
        ("round_10/transition.json", "8608338a1c622370eca3ff3692e79c7ed8ebd57d003f701525cc4adfd8d596e6"),
        ("round_10/memory.json", "46196709e12b669d0ea5894ce50d41e3273e7e4b41eedc63f1577db2f7294638"),
        ("round_11/transition.json", "08f35a25536ac2a450eb7faa827f8c385ac8d1555c7c44a89b9fb4dacb097ca4"),
        ("round_11/memory.json", "1175f594a4c9dc046a50858f2353ef990e42062771d7bf20fe601e7487683d0f"),
        ("round_12/source.json", "fefcfca59556ff2a8623feacbcc5dda078a44f05ad006d9c5af712fa1117e3bb"),
        ("round_12/transition.json", "8bae92f10d48972c566ad18a4fd868c0ab904447daad26222401ab75f6664b05"),
        ("round_12/fit_000/fit_audit.json", "76d60c67bbe4034975ec2b7c250d4ee7b7767c3527859a3505f9cd8babf5eb3a"),
    ), required_sources=SCOPES[(2357, 5)].required_sources,
)


SCOPES[(887, 6)] = replace(SCOPES[(887, 5)], wave=6, cooldown_field="not_before_unix", cooldown_unix=1795701794.7327738,
                             original_start_round=None, original_boundary_physical_attempts=1, fitted_prefix_files=(),
                             boundary_physical_attempts=2, required_sources=tuple(
    (source, "aacbc199e896091d4395fc036cb60bc604d91ec326a79fcff26e157d9a73705f" if source == PROVIDER_SOURCE else sha)
    for source, sha in SAME_RESPONSE_RETRY_SOURCES
), archive=(
    *SCOPES[(887, 5)].archive,
    Review(retry.bounded.RunIdentity("-887-relay-retry-05",
        "56dadfb27df633863b7a8e31ce103446fba7ea2a3b0f13188ffea5a3f7af5daf",
        "d40358da6676fd0ce3d75ae507d6c3abe8443124ac8772479766ed2feb20b604"),
        "8eaac87cccc02053f168a8d32860d34ab8af33291832dd148e1a9bac0c9c8d4d", "f0fcbe7e61894a389e1f6f2715b0081c",
        "0c497d13b0c326e29b2e94dd6387308e2f4d7752ff979d8424d3b193a71e407b",
        "53db427b46fc66113949696a70c99b7377e1d49857df219c02d7b0ede4cad5ec",
        "f7cec4bdff72d7fc9fd789aa4ed117d066cb0accd86ec3386151fa85b1fd1cc7",
        "323deb41f369ff271b089ae7f894e7a9dac58f085bce7874326ad7ce5f447e43", "timeout",
        {"http_status": 200, "phase": "transport_failure", "error_type": "hard_wall_deadline",
         "event_type": "response.output_text.delta", "sse_event_count": 3063, "last_sse_seconds": 3017.041881625075,
         "provider_request_id": None, "model": "gpt-6.2-astra",
         "response_id": "resp_0f3cc35efc34af43016b082e2f53ac87d29e05725c60e2bff5"}),
))


SCOPES[(887, 7)] = replace(SCOPES[(887, 6)], wave=7, cooldown_field="failed_unix", cooldown_unix=1795706314.0892487,
                             original_start_round=12, original_boundary_physical_attempts=2,
                             boundary_physical_attempts=1, archive=(
    *SCOPES[(887, 6)].archive,
    Review(retry.bounded.RunIdentity("-887-relay-retry-06",
        "05df8c27277d8ef241e0e24eebee334b7e261f93a26f8687279135a0c795e7a6",
        "7c7b2c6a3b6c521568df409f0813b6378b3c72cee701f5972ef368e12383eb8e"),
        "a21b75b60154249e0a6f115269ec581e595cf674249f3993672a02a4bdc126ba", "bd3433a1d2924c038d11fcc28c9c50ec",
        "18c75f6cb93d97bbe50e24bf3845ad2c6a9b6a2d34aacc22f17da95e586c948a",
        "c3bfb08f768c2b36e9954cedb14c0ba83b7cd603cab002749665f97e2fdbb1bb",
        "b411f9727ce4d310961bb7acfbdabe0a5681aaa25a52b2cb71bfddbf5ad768c4",
        "d4db95abc9c0f37a85e94b7830c0920f74003169974372121c3cef4e2ee41b07", "timeout",
        {"http_status": 200, "phase": "transport_failure", "error_type": "hard_wall_deadline",
         "event_type": "response.output_item.done", "sse_event_count": 4060, "last_sse_seconds": 2876.7962896670215,
         "provider_request_id": None, "model": "gpt-6.2-astra",
         "response_id": "resp_0ed690650e4d4829016b083fd8387087d0ade644c2fa35138a"}),
))


SCOPES[(887, 8)] = replace(SCOPES[(887, 7)], wave=8, cooldown_field="not_before_unix", cooldown_unix=1795708892.5213063,
                             original_boundary_physical_attempts=1, boundary_physical_attempts=2, archive=(
    *SCOPES[(887, 7)].archive,
    Review(retry.bounded.RunIdentity("-887-relay-retry-07",
        "e1521cd0a4df8d040e307e72f2d446e0f5020368227c9d0ef7a0f0c4122abc94",
        "be8e4fd97acea0f28408a47f405dd2631e67a862c0294f6b5ed6d5272c747a68"),
        "0b7052c4d92bb87d7043d1b974da626a35330ce0f44ea3466b5f44ea07181f0c", "d820f33e0ffc4a96aec41a0bf9ef6afd",
        "9f67b64d9165d2e19686a0e2d8ee15e72de3a9822aa559661e46c33f05b727c6",
        "4433da0b23521da441117d9bb805a2126455a0d10e89747e6223a8797ec8395a",
        "c509d8b291b11b5951409ffebe50f67d03b9635cb4aafc2bad71df617bbd45f2",
        "b2adc0ee127762c715d227dd7676190be9e8c1c07f2f20d8aab8882c6b78c4d5", "timeout",
        {"http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
         "event_type": "response.output_text.delta", "sse_event_count": 1347, "last_sse_seconds": 348.2114551658742,
         "provider_request_id": "175117fa-25c0-40b8-b11a-23960a32a5bd", "model": "gpt-6.2-astra",
         "response_id": "resp_0c10cef05b83b84a016b085029790487d1b095a46533e7c77e"}),
))


SCOPES[(887, 9)] = replace(SCOPES[(887, 8)], wave=9, cooldown_field="failed_unix", cooldown_unix=1795713858.0125747,
                             original_boundary_physical_attempts=2, boundary_physical_attempts=1, archive=(
    *SCOPES[(887, 8)].archive,
    Review(retry.bounded.RunIdentity("-887-relay-retry-08",
        "dc53baf5ce4f7aa74bdfa300d38bee182735132ddac9294eab940f07ecd7157b",
        "dcf42f11bf0a7294b2b7e0e61dc5c2f27e03d3be1f26b1f379de5df68067a5e6"),
        "2677b92a0b9886f5e295317d7e8d42188a226b7ff53554c80606f85fb8252414", "083e46db66ab4d6d862e7b21e1d270e1",
        "106cc3576359c6113a6f5af8442eb097bd5a559b80b8fce330c63197eb30a6df",
        "9d796f0f9727a12acf840d7d9a7e4fdcdb9113278db4b006c3c88d436625d217",
        "d251850e10d8e66f795b77213ae0a11943c049817000a8080d4412db7b27d3fe",
        "44140b6b9f3e0358077181667c1b462eb772c0a8be6d8fc89bca84129a8efdfd", "timeout",
        {"http_status": 200, "phase": "transport_failure", "error_type": "hard_wall_deadline",
         "event_type": "response.output_text.delta", "sse_event_count": 4679, "last_sse_seconds": 3551.905509167118,
         "last_sse_activity_seconds": 3551.905509167118,
         "provider_request_id": "32b23c7e-dde0-42bb-92b9-f54b74090afa", "model": "gpt-6.2-astra",
         "response_id": "resp_091d3a6b10f6db84016b085d5f6be487d1832b50cb44fc0739"}),
))


SCOPES[(887, 10)] = Scope(
    887, 13, 10, "20b9b243194cbc330493d023e3f27928fea6d7f05ae8d900c33b413a9670dcb4", (
        Review(retry.bounded.RunIdentity("-887-relay-retry-09",
            "8408218609f6a2597f575872027d9063e755193e9a19ecc1b11c185596f9f9f2",
            "e3dfe88654c69a1bee57dff0ac61e9446b4369e9063abec6bba449b6afd08cd5"),
            "4cd434414bb2a004cf502820a697c108563791901d236f56da48554ed26a557e", "45aa34b1ce2e4057a50e125986247374",
            "d09ef244178f2ece349133c58f0132ee079b0d9ace936a35327996bce3f92ca7",
            "548edea79c0185f5a909e8360567be3eb3bbef173e228997f9375087f8558ff5",
            "a7a4b3c0e234c0ba382d6d83d76d194a3df457e21cce20a7b88ae1eb0f81d707",
            "b0f4ef098235ec9888f93138edece746ed17d4df6ce277b9a72b799ccde892bd", "timeout",
            {"http_status": 200, "phase": "transport_failure", "error_type": "sse_inactivity_timeout",
             "event_type": "response.output_item.added", "sse_event_count": 33, "last_sse_seconds": 698.2018592918757,
             "elapsed_seconds": 1598.217064499855, "provider_request_id": "1d4b6c50-a782-4beb-942c-428ab95712bc",
             "model": "gpt-6.2-astra", "response_id": "resp_0e4cc43d31cd7c17016b086ea8394087d286ecf2b81d61e187"}),
    ), "not_before_unix", 1795716988.4392984, original_start_round=12, fitted_prefix_files=(
        ("boundary.json", "f8e16e48fcbb21596d7f953b3486e2a5639cf745195959b6da881a8f3dbc8228"),
        ("round_12/source.json", "fefcfca59556ff2a8623feacbcc5dda078a44f05ad006d9c5af712fa1117e3bb"),
        ("round_12/boundary_memory.json", "1175f594a4c9dc046a50858f2353ef990e42062771d7bf20fe601e7487683d0f"),
        ("round_12/boundary_user_message.json", "fe4cb1ee67adc272f372a701901238617447c19b47b28d923bb8728f5f856f6e"),
        ("round_12/selection.json", "bdfff944548e2c28b3f787abfcb218fdb64bc9f48d1e83f8886bd29a82794a33"),
        ("round_12/batch_freeze.json", "5d7ce14dff50d459e5859e30f4c94b1fecf079830cf4a07b9c17935e8431dc30"),
        ("round_12/memory.json", "58763db2a3995f943fb84f7f9b32c937705b4606b1c43a988079e84132380410"),
        ("round_12/turn_00/attempt_00.terminal.json", "9713bcb1e5fe1ef5a5ebf14864ee56f1104a6558143efe541f42de1f117c3d92"),
        ("round_13/source.json", "242f308cc4fa10fbe86d31bde6e349f31c5745c2a8b47a3588468c92fb008be1"),
        ("round_13/transition.json", "4d4f93c6958478d309ff42560626118cf5629a3aa6323a5a069bcfa734a69f33"),
        ("round_13/fit_000/fit_audit.json", "11b20513827e0c9bc6c094557de4f2d2e0adfa38529ca482e44bb692855791cb"),
        ("round_13/fit_000/validation_metrics.json", "3858914983e81f6e16302d6ed82b7cbc48133f4e673b6b78a7e746ba1027b132"),
        ("round_13/authoritative_packet.json", "fc15e56dd9c646aa7155986e15343c4c5548529023a8a136b9fde024106b0c9c"),
    ), required_sources=SCOPES[(887, 9)].required_sources,
    boundary_physical_attempts=1, original_boundary_physical_attempts=1,
)


_887_R20_CONTINUATION = retry.bounded.RunIdentity(
    "-887-r20-continuation-01",
    "c5105cdaaeb08d1efac389ce50e30f9a4e063fb9346fd84d8b26d1a2d68ee195",
    "a546f4ee7398eb4f16dd0c4c0016f5e1b5028ba14963fdf6366f8eb9b96031f5",
)
_887_R32_CHECKPOINT = "ac23374d4dd7f7276053d0d333707ccc03862aae40c7c419db01ac40806100a7"
_887_R32_FAILED_STREAM = Review(
    _887_R20_CONTINUATION,
    "7f0007dfc6b13e98ec4cd2048924ffa5f252543cc39a53a9b6ed797a1e16beab",
    "bede371adb36462082aac5b39355d648",
    "b63239eb02e2c6d71743fc127e09f0dab6b39cd410ac291edf917cc1a8e7b91c",
    "f3472191db9b5db955e7c64b4eaebf81e82bbccc9cab0f0b099bdb46f92f8389",
    "9f4c44ba5702e45a78eac04a62b660fb7decc9c64e5ff89a1382d528db0cbe64",
    "201fbfcb589974fd6e1b22c507147578c0cb9f5d26a3e2e26d38a0a7cb6468c6",
    "response.failed",
    {"http_status": 200, "phase": "transport_failure", "error_type": "response.failed", "event_type": "response.failed",
     "sse_event_count": 9, "first_sse_seconds": 9.765055833850056, "last_sse_seconds": 24.46110837487504,
     "last_sse_activity_seconds": 9.773610417032614, "elapsed_seconds": 24.466253083897755,
     "model": "gpt-6.2-astra", "response_id": "resp_ft_8b4f0119e61b4184bd7b87a7dfbcda7a",
     "provider_error_code": "upstream_error", "provider_request_id": None,
     "sse_milestones": {
         "response.created": {"elapsed_seconds": 9.765055833850056, "interval_seconds": None},
         "response.in_progress": {"elapsed_seconds": 9.767424499848858, "interval_seconds": 0.002368665998801589},
         "response.output_item.added": {"elapsed_seconds": 9.769154584035277,
                                        "interval_seconds": 0.0017300841864198446},
         "response.output_text.delta": {"elapsed_seconds": 9.770635250024498,
                                        "interval_seconds": 0.0003777500241994858},
         "response.output_text.done": {"elapsed_seconds": 9.771671583876014,
                                       "interval_seconds": 0.0010363338515162468},
         "response.output_item.done": {"elapsed_seconds": 9.773610417032614,
                                       "interval_seconds": 0.0010443332139402628},
         "response.failed": {"elapsed_seconds": 24.46110837487504, "interval_seconds": 14.687497957842425},
     }},
)
SCOPES[(887, 11)] = Scope(
    887, 32, 11, "302b17f2672c1cb477425d7aaa8ec077e443c07f6b4015a4499fb54b5e5d9b65",
    (_887_R32_FAILED_STREAM,), "not_before_unix", 1795730965.8450844,
    original_start_round=20,
    fitted_prefix_files=(
        ("round_20/source.json", "79f91d8382922a05f29a239176e7e9e20b48756ee3a3813e77f152cb640d3ae6"),
        ("round_31/memory.json", "ec18fb358a4bbd94dc384ad0671f568e9be700ee1d379f8cc16920167057ca3c"),
        ("round_31/batch_freeze.json", "caccafc445a769b5ec83754f4094ff63a5e80656cdeb606060bfd81d8c7e8f01"),
        ("round_31/selection.json", "175cc1905010f05a91b2bb946337a6355d1c91eea2a233a6bf4f73c7ed9679f3"),
        ("round_32/source.json", "35b8b5f750460dd39c9fd98e70b1e9ae167f25889d438d2720c7d9745aa1e9d3"),
        ("round_32/transition.json", "6e03687d135eb8696e1dafae86766368f34a284c85cdf4d559ff34c18ec3f7ef"),
        ("round_32/fit_000/fit_audit.json", "ebf3cb221b13939bf50d25bab0e0e70120f803b2fa25914aee39bb3e0a8d5a25"),
        ("round_32/fit_000/validation_metrics.json", "05e57e272b50bf9542d1f1a6d33440cc3da731f0fb672f106b9aed1d9a4b3047"),
        ("round_32/authoritative_packet.json", "8c9099c4600386baae1435633754edccf14291839d59fa252d997d1431357fda"),
        ("round_32/turn_00/request.json", "b067a98ce415d93ba15c4e7045083d7863a8937b1db3a9622366f64478ad2fbb"),
    ),
    required_sources=(
        *QUEUE_SOURCES,
        (PROVIDER_SOURCE, "aacbc199e896091d4395fc036cb60bc604d91ec326a79fcff26e157d9a73705f"),
        ((STUDY / "a7_gpu/seed_runner.py").as_posix(),
         "19122c372294d5dd0f08d2ba4b1388b027ac1bacf65d84004c5eacf5b498e9f6"),
        ((STUDY.parent / "pure_llm_v3_20261124/pure_llm_v3/transport_manager.py").as_posix(),
         "2cdc0990f81012f040a73f8d5b7226fe1d35b934f477dc1654ee071126193a3c"),
    ),
)


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
                != (attempt.attempt_id, attempt.job_sha256, attempt.result_sha256, attempt.failure,
                    True, not attempt.pre_post_connect_failure)):
            raise PermissionError("a reviewed stopped transport receipt changed")
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
    for index, review in enumerate(reversed(scope.archive), 1):
        contract, approval = retry.bounded._archived(run, release, review.identity)
        require_pause = bool(records) or review.identity == _887_R20_CONTINUATION
        directory, active = retry._retired(run, scope, require_pause=require_pause)
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
        if index < len(scope.archive):
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


def _reviewed_boundary(root, directory, active, parent, approval, scope, cache):
    source_only_import = {
        (887, 12, 9, _887_R9_CONTINUATION.contract):
            ("dispatch_v13_887_r9_continuation", _887_R12_CHECKPOINT, 717),
        (6101, 6, 5, _6101_R5_CONTINUATION.contract):
            ("dispatch_v13_6101_r5_continuation", _6101_R6_CHECKPOINT, 525),
        (887, 32, 20, _887_R20_CONTINUATION.contract):
            ("dispatch_v13_887_r20_continuation", _887_R32_CHECKPOINT, 1357),
    }.get((scope.seed, scope.round, scope.original_start_round, parent.digest))
    if source_only_import is None:
        return retry._boundary(root, directory, active, parent, approval, scope, cache)
    dispatcher, checkpoint, labels = source_only_import
    descriptor = parent.profile["resume_boundaries"][str(scope.seed)]

    # These exact signed continuations imported state and memory before staging a model request.
    # Verify their original fit transitions and then pin the current archived body in construct.
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend import TrainingBackend

    prior_scope = seed_runner._scope(parent, approval, scope.seed, False)
    start = scope.original_start_round
    initial = seed_runner.round_directory(directory, start)
    state = seed_runner.FrozenState(**read(initial / "source.json"))
    memory = read(initial / "boundary_memory.json")
    if (parent.predecessor.get("dispatcher_module") != dispatcher
            or descriptor != {"round": start, "state_sha256": digest(asdict(state)), "memory_sha256": digest(memory)}
            or read(directory / "boundary.json") != {"formal_run": prior_scope, **descriptor}
            or (initial / "boundary_user_message.json").exists()):
        raise PermissionError(f"the exact R{start} continuation boundary structure changed")
    proof = {"initial_boundary": file_digest(directory / "boundary.json")}
    backend = TrainingBackend(root, cache, scope.seed)
    for index in range(start, scope.round):
        prior = seed_runner.round_directory(directory, index)
        following = seed_runner.round_directory(directory, index + 1)
        state, memory = seed_runner._successor(prior, state, following, prior_scope, parent.transport, parent, backend)
        proof[f"transition_{index + 1:02d}"] = file_digest(following / "transition.json")
    if ((state.seed, state.round, state.checkpoint, len(state.provenance["expected_observed"]))
            != (scope.seed, scope.round, checkpoint, labels) or asdict(state) != read(active / "source.json")):
        raise PermissionError(f"the reviewed R{scope.round} checkpoint differs from its verified L{labels} fit")
    proof.update(latest_source=file_digest(active / "source.json"), latest_memory=digest(memory))
    return (state, memory, scope.round), proof


def construct(root, failed_release, failed_run, run, scope):
    if run.name != scope.run_name or run.parent != failed_run.parent or root == failed_release:
        raise PermissionError("use the registered successor and an independent immutable source release")
    records = _archives(failed_run, failed_release, scope)
    last = records[0]
    parent, approval = last["contract"], last["approval"]
    header_deadline = scope.header_deadline_profile
    if HEADER_DEADLINE_FIELD in parent.profile and (
            type(parent.profile[HEADER_DEADLINE_FIELD]) is not int
            or parent.profile[HEADER_DEADLINE_FIELD] != header_deadline.get(HEADER_DEADLINE_FIELD)):
        raise PermissionError("the successor must explicitly preserve its parent's response headers deadline")
    old = parent.predecessor
    parent_start = parent.profile["resume_boundaries"][str(scope.seed)]["round"]
    cached_parent = scope.cached_origin is not None and scope.original_start_round is not None
    if cached_parent:
        remaining = parent.rounds - scope.round
        expected_budget = (scope.original_start_round, {"maximum_logical_requests": 1, "maximum_physical_attempts": 0},
                           {"first_logical_requests": remaining, "maximum_logical_requests": 3 * remaining,
                            "maximum_physical_attempts": 6 * remaining})
    elif scope.original_start_round is None:
        expected_budget = (scope.round, RECOVERY_BUDGET, scope.execution_budget(1))
    else:
        original_attempts = scope.original_boundary_physical_attempts
        original_execution = retry.execution_budget(parent.rounds, scope.original_start_round)
        original_execution["maximum_physical_attempts"] -= 2 - original_attempts
        expected_budget = (scope.original_start_round,
                           {"maximum_logical_requests": 1, "maximum_physical_attempts": original_attempts},
                           original_execution)
    if (last["terminal"].get(scope.cooldown_field) != scope.cooldown_unix
            or (parent_start, old["recovery_budget"], old["execution_budget"]) != expected_budget
            or any(file_digest(last["directory"] / name) != sha for name, sha in scope.fitted_prefix_files)):
        raise PermissionError("the retired physical budget or original cooldown clock differs from its record")
    if scope.cached_origin is not None and not cached_parent and old.get("reviewed_cached_import") != scope.cached_origin.review:
        raise PermissionError("the previous successor does not preserve the verified cached import evidence")
    resources = _cached_resources(failed_run, last["directory"], old, scope) if cached_parent else old
    cache = Path(old["cache_directory"]).resolve(strict=True)
    if file_digest(cache / "resource_receipt.json") != resources["public_resource_receipt_sha256"]:
        raise PermissionError("the original public training cache changed")
    boundary, prefix = _reviewed_boundary(failed_release, last["directory"], last["active"], parent, approval, scope, cache)
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
        "previous_prefix_files": prefix, "recovery_budget": scope.recovery_budget, "execution_budget": scope.budget,
        "relay_not_before_unix": scope.not_before_unix, "relay_cooldown_clock": scope.cooldown_field,
        "newly_authorized_boundary_physical_attempts": scope.boundary_physical_attempts, "reviewed_retry_chain": history,
        "public_resource_receipt_sha256": resources["public_resource_receipt_sha256"],
        "bridge_wait_seconds": resources["bridge_wait_seconds"],
    }
    proof.update({key: binding[key] for key in ("checkpoint_sha256", "packet_sha256", "wire_packet_sha256", "base_messages_sha256")})
    if scope.fitted_prefix_files:
        proof["reviewed_fitted_prefix_files"] = dict(scope.fitted_prefix_files)
    if cached_parent:
        proof.pop("cached_import")
        proof["reviewed_cached_import"] = scope.cached_origin.review
    for stale in ("authoritative_message_reconstruction", "inherited_boundary_budget"):
        proof.pop(stale, None)
    if host:
        proof["authoritative_message_reconstruction"] = host
    descriptor = {"round": index, "state_sha256": digest(asdict(state)),
                  "memory_sha256": digest(memory), "user_message_sha256": digest(user)}
    profile = {**parent.profile, "replay_eligible_from_round": scope.replay_start_round,
               "resume_boundaries": {str(scope.seed): descriptor},
               "sealed_relay_retry_seconds": retry.COOLDOWN_SECONDS, **header_deadline}
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
        parser.error("run and resume require the exact new authorization for the finite physical attempts")
    approval = verify_authorization(contract, read(args.authorization))
    if time() < scope.not_before_unix:
        raise PermissionError("the original 180-second relay cooldown is still in force")
    result = retry.execute(ROOT, failed_run, run, scope, contract, approval, boundary, cache, original,
                           resume=args.action == "resume")
    print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
