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
        parser.error("run and resume require the exact new authorization for the finite physical attempts")
    approval = verify_authorization(contract, read(args.authorization))
    if time() < scope.not_before_unix:
        raise PermissionError("the original 180-second relay cooldown is still in force")
    result = retry.execute(ROOT, failed_run, run, scope, contract, approval, boundary, cache, original,
                           resume=args.action == "resume")
    print(dumps(asdict(result)))


if __name__ == "__main__":
    main()
