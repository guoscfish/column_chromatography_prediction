"""Ensure bounded recoveries cannot dispatch an extra or modified scientific request."""

from copy import deepcopy

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import dispatch_v13_bounded_sse_recovery as entry


@pytest.mark.parametrize("seed,round_index", [(157, 3), (887, 6), (2357, 2), (6101, 1)])
def test_one_exact_cached_decision_for_each_seed(monkeypatch, seed, round_index):
    payload = {"model": "gpt-6.2-astra", "stream": True, "input": {"synthetic": "example"}}
    binding = {"seed": seed, "round": round_index, "prompt_sha256": "p", "checkpoint_sha256": "c"}
    original = {"request_id": "old", "binding": binding, "payload": payload, "payload_sha256": digest(payload)}
    request = {**deepcopy(original), "request_id": "new", "logical_index": 0}
    started = {"attempt_id": "new-attempt", "attempt": 0}
    dispatcher = object.__new__(entry.BoundedRecoveryDispatcher)
    dispatcher.original = original
    dispatcher.scope = {"seed": seed, "round": round_index, "payload_sha256": original["payload_sha256"],
                        "previous_request_id": "spent-request", "previous_attempt": {"attempt_id": "spent-physical"}}
    dispatcher.prompt_sha = binding["prompt_sha256"]
    monkeypatch.setattr(entry.Dispatcher, "validate", lambda self, job: (request, None, started))
    assert dispatcher.validate({}) == (request, None, started)
    started["attempt"] = 1
    with pytest.raises(PermissionError, match="extra attempt"):
        dispatcher.validate({})
    started["attempt"] = 0
    request["payload"]["input"] = {"synthetic": "changed"}
    with pytest.raises(PermissionError, match="different scientific request"):
        dispatcher.validate({})
