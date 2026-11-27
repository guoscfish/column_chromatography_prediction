from copy import deepcopy
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportStop
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_6101_r10_cached as cache


@pytest.fixture
def archive(monkeypatch):
    plan = {"choices": [{"candidate_id": f"C-{index:04d}", "reason": "original"} for index in range(32)]}
    final_text = f"\n{dumps(plan)}\n"
    message = {"type": "message", "role": "assistant", "status": "completed"}
    raw = {"id": cache.RESPONSE, "model": "gpt-6.2-astra", "status": "completed", "truncation": "disabled", "reasoning": {"effort": "xhigh"},
           "output": [{**message, "content": [{"type": "output_text", "text": " "}]},
                      *[{"type": "reasoning", "id": f"reasoning_{index}"} for index in range(7)],
                      {**message, "phase": "final_answer", "content": [{"type": "output_text", "text": final_text}]}],
           "usage": {key: cache.USAGE[key] for key in ("input_tokens", "output_tokens", "total_tokens")}}
    raw["usage"].update(input_tokens_details={"cached_tokens": 0}, output_tokens_details={"reasoning_tokens": 5517})
    transport = SimpleNamespace(provider="token4research", requested_model="gpt-6.2-astra", served_model="gpt-6.2-astra",
                                reasoning_effort="xhigh")
    lock = {"provider": "token4research", "requested_model": "gpt-6.2-astra", "served_model": "gpt-6.2-astra",
            "requested_reasoning_effort": "xhigh", "reported_reasoning_effort": "xhigh", "reported_model_version": None}
    capture = {"attempt_id": cache.ATTEMPT, "job_sha256": cache.JOB,
               "metadata": {"raw_response_sha256": digest(raw), "provider_request_id": cache.PROVIDER_REQUEST,
                            "model_version": None, "reported_reasoning_effort": "xhigh"}}
    for name, value in {"RAW": digest(raw), "PLACEHOLDER": digest(raw["output"][0]), "FINAL": digest(raw["output"][-1]),
                        "TEXT": digest(final_text), "CAPTURE": digest(capture), "PLAN": digest(plan),
                        "ORDER": digest([item["candidate_id"] for item in plan["choices"]])}.items():
        monkeypatch.setattr(cache, name, value)
    return raw, transport, lock, plan, final_text


def test_archived_final_is_used_byte_for_byte_with_original_identity_and_usage(archive, monkeypatch):
    raw, transport, lock, plan, text = archive
    before = deepcopy(raw)
    reply = cache._extract_final(raw, transport, lock)
    assert raw == before and reply.text == text
    assert (reply.response_id, reply.served_model, reply.usage) == (cache.RESPONSE, "gpt-6.2-astra", cache.USAGE)
    monkeypatch.setattr(cache, "_validation", lambda stage, value: SimpleNamespace(valid=value == plan, valid_choice_count=32))
    assert cache._verified_plan(None, reply) == plan
    monkeypatch.setattr(cache, "_validation", lambda stage, value: SimpleNamespace(valid=False, valid_choice_count=31))
    with pytest.raises(PermissionError, match="official A7 validator"):
        cache._verified_plan(None, reply)


@pytest.mark.parametrize("variant", ("substantive", "no_final", "duplicate_final", "extra_message", "non_assistant"))
def test_scoped_projection_refuses_other_phase_shapes(archive, monkeypatch, variant):
    raw, transport, lock, _, _ = archive
    if variant == "substantive":
        raw["output"][0]["content"][0]["text"] = "substantive"
    elif variant == "no_final":
        raw["output"][-1]["phase"] = None
    elif variant == "duplicate_final":
        raw["output"][5] = deepcopy(raw["output"][-1])
    elif variant == "extra_message":
        raw["output"][4] = deepcopy(raw["output"][0])
    else:
        raw["output"][0]["role"] = "user"
    for name, value in {"RAW": digest(raw), "PLACEHOLDER": digest(raw["output"][0]),
                        "FINAL": digest(raw["output"][-1])}.items():
        monkeypatch.setattr(cache, name, value)
    with pytest.raises(PermissionError, match="ASCII space placeholder"):
        cache._extract_final(raw, transport, lock)


def test_retired_fence_rejects_new_turn_or_freeze(tmp_path, monkeypatch):
    old = tmp_path / "old"
    current = old / "seed_6101/round_10"
    turn = current / "turn_00"
    turn.mkdir(parents=True)
    for name in ("started", "terminal"):
        (turn / f"attempt_00.{name}.json").write_text("{}")
    worker = old / "workers/seed_6101.json"
    worker.parent.mkdir()
    worker.write_text('{"status":"failed","pid":2000000000}')
    pause = current.parent / "pause.request"
    pause.write_text("fenced")
    monkeypatch.setattr(cache, "ARCHIVE_FILES", {str(pause.relative_to(old)): cache.file_digest(pause)})
    assert cache._fenced(old) == (current.parent, turn)
    (current / "batch_freeze.json").write_text("{}")
    with pytest.raises(PermissionError, match="pause fence"):
        cache._fenced(old)
    (current / "batch_freeze.json").unlink()
    (turn / "attempt_01.started.json").write_text("{}")
    with pytest.raises(PermissionError, match="pause fence"):
        cache._fenced(old)


def test_cached_scope_has_no_provider_and_rejects_bridge_evidence(tmp_path):
    assert cache.RECOVERY_BUDGET == {"maximum_logical_requests": 1, "maximum_physical_attempts": 0}
    assert (cache.SEED, cache.ROUND, cache.TARGET) == (6101, 10, 685)
    assert cache._virtual_evidence({"provider_post": False})["provider_dispatched"] is False
    with pytest.raises(TransportStop, match="no provider transport"):
        cache._no_provider()
    outbox = tmp_path / "bridge/outbox"
    outbox.mkdir(parents=True)
    (outbox / "unexpected.json").write_text("{}")
    with pytest.raises(PermissionError, match="zero-POST"):
        cache._require_no_bridge(tmp_path)


def test_readonly_split_receipt_has_signed_digest_and_only_initial_fit_roles(tmp_path):
    path = tmp_path / cache.SPLITS / "row_seed_6101.csv"
    path.parent.mkdir(parents=True)
    path.write_text("sample_id,role\nrow-a,l0\nrow-b,u0\nrow-c,validation\nrow-d,test\n")
    split = cache._split(tmp_path)
    assert split.l0_ids == ("row-a",) and split.u0_ids == ("row-b",)
    assert split.split_sha256 == cache.file_digest(path)
