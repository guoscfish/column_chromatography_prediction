from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import dispatch_prompt_direct as direct


@pytest.mark.parametrize(("seed", "round_index"), ((157, 1), (887, 4), (2357, 1)))
def test_direct_attempt_keeps_exact_original_prompt_data_and_payload(seed, round_index):
    original = {
        "binding": {**dict.fromkeys(direct.DATA_BINDINGS, "unchanged"), "prompt_sha256": "approved"},
        "payload": {"input": [{"role": "system", "content": "approved prompt"}, {"role": "user", "content": "L365"}]},
        "request_id": "original",
        "payload_sha256": "same payload",
    }
    new = deepcopy(original)
    new["binding"].update(seed=seed, round=round_index)
    new.update(request_id="new", logical_index=0)
    dispatcher = SimpleNamespace(validate=lambda job: (job, None, {"attempt": 0}))
    assert direct._validate(dispatcher, new, original, seed, "approved") is new
    changed = deepcopy(new)
    changed["payload"]["input"][1]["content"] = "changed"
    with pytest.raises(PermissionError, match="exact original"):
        direct._validate(dispatcher, changed, original, seed, "approved")
    wrong_round = deepcopy(new)
    wrong_round["binding"]["round"] += 1
    with pytest.raises(PermissionError, match="exact original"):
        direct._validate(dispatcher, wrong_round, original, seed, "approved")
    replay_dispatcher = SimpleNamespace(validate=lambda job: (job, None, {"attempt": 1}))
    with pytest.raises(PermissionError, match="exact original"):
        direct._validate(replay_dispatcher, new, original, seed, "approved")


@pytest.mark.parametrize(("seed", "round_index"), ((157, 1), (887, 4), (2357, 1)))
def test_direct_transport_requires_one_attempt_and_verified_direct_source(seed, round_index, monkeypatch):
    old = TransportConfig("fixture", "fixture", "fixture", "xhigh", "https://example.invalid/v1", 1800, 32768,
                          logical_replies=3, transport_attempts=2)
    direct_config = replace(old, timeout_seconds=3600, transport_attempts=1, logical_replies=1)
    profile = {"transport": asdict(direct_config), "request_timeouts": asdict(direct.DIRECT_TIMEOUTS),
               "resume_boundaries": {str(seed): {"round": round_index}}}
    info = {"contract": {"profile": profile,
                         "source_sha256": {direct.DIRECT_PROVIDER.as_posix(): direct.DIRECT_PROVIDER_SHA256}}}
    monkeypatch.setattr(direct, "file_digest", lambda _path: direct.DIRECT_PROVIDER_SHA256)
    monkeypatch.setattr(direct, "sdk_status", lambda _transport: {"matches": True})
    assert direct._direct_transport(info, seed, old, old) == direct_config
    profile["transport"]["transport_attempts"] = 2
    with pytest.raises(PermissionError, match="single-attempt"):
        direct._direct_transport(info, seed, old, old)
    profile["transport"]["transport_attempts"] = 1
    monkeypatch.setattr(direct, "file_digest", lambda _path: "different source")
    with pytest.raises(PermissionError, match="direct HTTPS"):
        direct._direct_transport(info, seed, old, old)
