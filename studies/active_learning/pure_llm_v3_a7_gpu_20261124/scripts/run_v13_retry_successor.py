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
