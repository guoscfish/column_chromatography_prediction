import pytest

from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import (
    dispatch_v13_12203_relay_retry as mac,
    run_v13_12203_relay_retry as gpu,
)


def test_successor_inherits_only_the_undispatched_original_slot():
    assert gpu.RECOVERY_BUDGET == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert gpu.execution_budget() == {"first_logical_requests": 32, "maximum_logical_requests": 94,
                                      "maximum_physical_attempts": 187}
    assert gpu.execution_budget()["maximum_physical_attempts"] + 1 == gpu.retry.execution_budget(36, 4)[
        "maximum_physical_attempts"]
    assert gpu.NOT_BEFORE_UNIX - gpu.FAILED_UNIX == 180


def test_mac_rejects_early_or_extra_boundary_slots(monkeypatch):
    handled = []
    monkeypatch.setattr(mac.RelayRetryDispatcher, "_check_boundary",
                        lambda self, request, started: handled.append((request, started)))
    dispatcher = mac.InheritedRelayDispatcher.__new__(mac.InheritedRelayDispatcher)
    request = object()
    initial = {"attempt": 0}
    monkeypatch.setattr(mac, "time", lambda: gpu.NOT_BEFORE_UNIX - 1)
    with pytest.raises(PermissionError, match="single original unspent slot"):
        dispatcher._check_boundary(request, initial)
    monkeypatch.setattr(mac, "time", lambda: gpu.NOT_BEFORE_UNIX)
    with pytest.raises(PermissionError, match="single original unspent slot"):
        dispatcher._check_boundary(request, {"attempt": 1})
    dispatcher._check_boundary(request, initial)
    assert handled == [(request, initial)]
