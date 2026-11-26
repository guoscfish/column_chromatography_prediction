import pytest

from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import (
    dispatch_v13_157_relay_retry as mac,
    run_v13_157_relay_retry as gpu,
)


def test_only_one_new_157_slot_after_two_exhausted_attempts():
    assert len({attempt.attempt_id for attempt in gpu.PRIOR}) == 2
    assert gpu.RECOVERY_BUDGET == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert gpu.execution_budget() == {"first_logical_requests": 33, "maximum_logical_requests": 97,
                                      "maximum_physical_attempts": 193}
    assert gpu.NOT_BEFORE_UNIX - gpu.FAILED_UNIX == 180


def test_157_mac_requires_a_new_identity_after_the_second_failure_cooldown(monkeypatch):
    accepted = []
    monkeypatch.setattr(mac.RelayRetryDispatcher, "_check_boundary",
                        lambda self, request, started: accepted.append((request, started)))
    dispatcher = mac.AdditionalRelayDispatcher.__new__(mac.AdditionalRelayDispatcher)
    new = {"attempt": 0, "attempt_id": "a" * 32}
    monkeypatch.setattr(mac, "time", lambda: gpu.NOT_BEFORE_UNIX - 0.01)
    with pytest.raises(PermissionError, match="latest cooldown"):
        dispatcher._check_boundary({}, new)
    monkeypatch.setattr(mac, "time", lambda: gpu.NOT_BEFORE_UNIX)
    for bad in [{**new, "attempt": 1}, *[{**new, "attempt_id": p.attempt_id} for p in gpu.PRIOR]]:
        with pytest.raises(PermissionError, match="one distinct physical attempt"):
            dispatcher._check_boundary({}, bad)
    dispatcher._check_boundary({}, new)
    assert accepted == [({}, new)]
