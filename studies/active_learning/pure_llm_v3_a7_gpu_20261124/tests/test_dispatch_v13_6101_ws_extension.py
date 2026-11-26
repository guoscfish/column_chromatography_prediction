"""Checks for the signed 6101 WebSocket recovery boundary."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import dispatch_v13_6101_ws_extension as entry


@pytest.fixture
def staged(monkeypatch):
    payload = {"model": "gpt-6.2-astra", "reasoning": {"effort": "xhigh"},
               "input": [{"role": "user", "content": "synthetic fixture"}], "stream": True, "store": False}
    binding = {"seed": 6101, "round": 1, "prompt_sha256": "x", "checkpoint_sha256": "y",
               "a7_formal": {"contract_sha256": "new"}}
    original = {"request_id": "archived", "logical_index": 0, "payload": payload,
                "payload_sha256": digest(payload), "binding": binding}
    request = {**deepcopy(original), "request_id": "current"}
    request["binding"]["a7_formal"] = {"contract_sha256": "newer"}
    dispatcher = object.__new__(entry.WebSocketRecoveryDispatcher)
    dispatcher.original = original
    dispatcher.prompt_sha = binding["prompt_sha256"]
    started = {"attempt": 0, "attempt_id": "fresh-attempt"}
    monkeypatch.setattr(entry, "PAYLOAD_SHA", digest(payload))
    monkeypatch.setattr(entry.Dispatcher, "validate", lambda self, job: (request, "config", started))
    return SimpleNamespace(dispatcher=dispatcher, request=request, started=started)


def test_allows_only_the_exact_scientific_payload(staged):
    assert staged.dispatcher.validate({}) == (staged.request, "config", staged.started)
    staged.request["payload"]["input"][0]["content"] = "other input"
    staged.request["payload_sha256"] = digest(staged.request["payload"])
    with pytest.raises(PermissionError, match="exact archived scientific request"):
        staged.dispatcher.validate({})


@pytest.mark.parametrize("field,value", [("attempt", 1), ("attempt_id", entry.PREVIOUS_ATTEMPT)])
def test_rejects_an_extra_or_existing_physical_attempt(staged, field, value):
    staged.started[field] = value
    with pytest.raises(PermissionError, match="exact archived scientific request"):
        staged.dispatcher.validate({})


def test_refuses_a_changed_checkpoint_even_if_the_wire_matches(staged):
    staged.request["binding"]["checkpoint_sha256"] = "changed"
    with pytest.raises(PermissionError, match="exact archived scientific request"):
        staged.dispatcher.validate({})
