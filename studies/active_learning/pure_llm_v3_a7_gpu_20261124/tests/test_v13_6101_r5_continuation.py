from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import dumps
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import dispatch_v13_6101_r5_continuation as mac
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_6101_r5_continuation as gpu


def test_handoff_requires_exited_l493_and_zero_bridge(tmp_path, monkeypatch):
    worker = tmp_path / "workers/seed_6101.json"
    worker.parent.mkdir()
    worker.write_text(dumps({"status": "initializing", "completed_rounds": 4, "active_labels": 461}))
    with pytest.raises(PermissionError, match="fully fitted L493"):
        gpu._require_complete(tmp_path)
    worker.write_text(dumps({"status": "complete", "completed_rounds": 5, "active_labels": 493}))
    checked = []
    monkeypatch.setattr(gpu.scheduler, "_assert_no_old_workers", lambda run, seeds: checked.append((run, seeds)))
    gpu._require_complete(tmp_path)
    assert checked == [(tmp_path, (6101,))]
    reservation = tmp_path / "bridge/outbox/job.json"
    reservation.parent.mkdir(parents=True)
    reservation.touch()
    with pytest.raises(PermissionError, match="zero-POST"):
        gpu._require_complete(tmp_path)


def test_terminal_only_cache_no_paid_calls_and_remaining_budget_is_finite():
    ordinary_transport = {"requested_model": "gpt-6.2-astra", "reasoning_effort": "xhigh", "sdk_version": "2.28.0",
                          "transport_attempts": 2, "logical_replies": 3, "wire_mode": "responses_stream_ipv4"}
    ordinary = {"target_labels": 1485, "transport": ordinary_transport, "confirmation_seeds": [6101]}
    descriptor = {"round": 5, "state_sha256": "exact fitted state", "memory_sha256": "exact fitted memory"}
    profile = gpu._original_profile(SimpleNamespace(profile=ordinary), descriptor)
    assert profile["transport"] == ordinary_transport
    assert profile["target_labels"] == 1485 and profile["resume_boundaries"] == {"6101": descriptor}
    assert profile["replay_eligible_from_round"] == 5 and profile["sealed_relay_retry_seconds"] == 180
    assert gpu.cached.BUDGET["maximum_physical_attempts"] == 0
    assert gpu.BOUNDARY_BUDGET == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert gpu.BUDGET == {"first_logical_requests": 31, "maximum_logical_requests": 1 + 30 * 3,
                          "maximum_physical_attempts": 2 + 30 * 3 * 2}


def test_mac_r5_cannot_reuse_l461_or_second_logical_slot():
    dispatcher = object.__new__(mac.ContinuationDispatcher)
    dispatcher.scope = {"checkpoint_sha256": "L493"}
    retries = []
    dispatcher._check_boundary_retry = lambda request, started: retries.append((request, started))
    request = {"binding": {"checkpoint_sha256": "L493"}, "logical_index": 0}
    dispatcher._check_boundary(request, {"attempt": 0})
    with pytest.raises(PermissionError, match="fitted checkpoint"):
        dispatcher._check_boundary({**request, "binding": {"checkpoint_sha256": "L461"}}, {"attempt": 0})
    with pytest.raises(PermissionError, match="one ordinary logical decision"):
        dispatcher._check_boundary({**request, "logical_index": 1}, {"attempt": 0})
    with pytest.raises(PermissionError, match="one ordinary logical decision"):
        dispatcher._check_boundary(request, {"attempt": 2})
    dispatcher._check_boundary(request, {"attempt": 1})
    assert retries == [(request, {"attempt": 1})]
