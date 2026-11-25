from dataclasses import asdict, replace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import dispatch_prompt_finish as finish


@pytest.mark.parametrize(("seed", "start"), ((157, 1), (887, 5), (2357, 1)))
def test_finish_scope_pins_seed_boundary_prompt_and_l1485(seed, start, monkeypatch):
    directory = f"/persistent/{finish.RUN_PREFIX}-{seed}"
    profile = {"arm": "A7", "memory": "M2", "device": "cuda", "initial_labels": 333,
               "target_labels": 1485, "batch_size": 32, "api_parallelism": 3, "store": False,
               "confirmation_seeds": list(finish.BOUNDARIES),
               "wire_representation": finish.FORMAL_WIRE_REPRESENTATION,
               "resume_boundaries": {str(seed): {"round": start}}}
    source = {path.as_posix(): "digest" for path in (finish.DISPATCH_SOURCE, finish.V5_DISPATCH_SOURCE)}
    info = {"contract": {"profile": profile, "run_directory": directory, "api_execution": finish.SSH_API,
                         "source_sha256": source}}
    monkeypatch.setattr(finish, "file_digest", lambda _path: "digest")
    monkeypatch.setattr(finish, "_approved_prompt_sha", lambda _info: finish.PROMPT_SHA256)
    assert finish._approved_scope(info, seed, directory) == finish.PROMPT_SHA256
    profile["resume_boundaries"][str(seed)]["round"] = start - 1
    with pytest.raises(PermissionError, match="V11 A7 trajectory"):
        finish._approved_scope(info, seed, directory)
    profile["resume_boundaries"][str(seed)]["round"] = start
    monkeypatch.setattr(finish, "_approved_prompt_sha", lambda _info: "different")
    with pytest.raises(PermissionError, match="improved V5 prompt"):
        finish._approved_scope(info, seed, directory)


def test_finish_transport_requires_the_direct_source_and_finite_policy(monkeypatch):
    pilot = TransportConfig("token4research", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh",
                            "https://token4research.cn/v1", 1800, 32768, transport_attempts=2, sdk_version="2.28.0")
    direct = replace(pilot, timeout_seconds=3600)
    profile = {"transport": asdict(direct), "request_timeouts": asdict(finish.TIMEOUTS)}
    info = {"contract": {"profile": profile,
                         "source_sha256": {finish.DIRECT_PROVIDER.as_posix(): finish.DIRECT_PROVIDER_SHA256}}}
    monkeypatch.setattr(finish, "file_digest", lambda _path: finish.DIRECT_PROVIDER_SHA256)
    monkeypatch.setattr(finish, "sdk_status", lambda _transport: {"matches": True})
    assert finish._approved_transport(info, pilot) == direct
    profile["request_timeouts"]["read_idle_seconds"] = 900
    with pytest.raises(PermissionError, match="timeouts"):
        finish._approved_transport(info, pilot)
    profile["request_timeouts"]["read_idle_seconds"] = 1800
    profile["transport"]["transport_attempts"] = 3
    with pytest.raises(PermissionError, match="finite request budget"):
        finish._approved_transport(info, pilot)


def test_finish_dispatch_cannot_send_after_round_35(monkeypatch):
    monkeypatch.setattr(finish.PromptDispatcher, "validate", lambda _self, job: (job, None, None))
    dispatcher = object.__new__(finish.FinishDispatcher)
    dispatcher.info = {"contract": {"profile": {"initial_labels": 333, "target_labels": 1485, "batch_size": 32}}}
    request = {"binding": {"round": 35}}
    assert dispatcher.validate(request)[0] == request
    request["binding"]["round"] = 36
    with pytest.raises(PermissionError, match="L1485"):
        dispatcher.validate(request)
