from copy import deepcopy
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import (
    dispatch_v13_887_r14_continuation as mac,
    run_v13_887_r14_continuation as gpu,
)


def test_restored_profile_replays_only_from_l781_and_preserves_the_regular_transport():
    inherited = {"target_labels": 1485, "transport": {"logical_replies": 3, "transport_attempts": 2},
                 "provider_capability": "gated_ipv4", "normalize_observed_reference_aliases": True,
                 "replay_eligible_from_round": 14, "resume_boundaries": {"887": {"round": 13}}}
    parent = SimpleNamespace(profile=inherited)
    expected = {"round": 14, "state_sha256": gpu.STATE_SHA, "memory_sha256": gpu.MEMORY_SHA}
    restored = gpu._original_profile(parent, expected)
    assert (restored["resume_boundaries"], restored["replay_eligible_from_round"]) == ({"887": expected}, 14)
    assert restored["transport"] is inherited["transport"] and restored["target_labels"] == 1485
    assert restored["sealed_relay_retry_seconds"] == 180 and restored["audited_future_transport_replay"]
    remaining = 36 - gpu.START
    assert gpu.BUDGET == {"first_logical_requests": remaining,
                          "maximum_logical_requests": 1 + (remaining - 1) * 3,
                          "maximum_physical_attempts": 2 + (remaining - 1) * 6}


def test_r14_requires_exact_rebuilt_packet_and_defers_second_attempt_to_existing_guard(monkeypatch):
    dispatcher = mac.ContinuationDispatcher.__new__(mac.ContinuationDispatcher)
    dispatcher.scope = {"r14_request": gpu.R14_REQUEST}
    accepted = {"logical_index": 0, "payload_sha256": gpu.R14_REQUEST["payload_sha256"],
                "binding": {k: v for k, v in gpu.R14_REQUEST.items() if k != "payload_sha256"}}
    p0 = {"attempt": 0, "attempt_id": "first"}
    p1 = {"attempt": 1, "attempt_id": "second"}
    recorded = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry", lambda req, slot: recorded.append((req, slot)))
    dispatcher._check_boundary(accepted, p0)
    assert not recorded
    dispatcher._check_boundary(accepted, p1)
    assert recorded == [(accepted, p1)]
    for field in accepted["binding"]:
        changed = deepcopy(accepted)
        changed["binding"][field] = "f" * 64
        with pytest.raises(PermissionError, match="exact fitted checkpoint"):
            dispatcher._check_boundary(changed, p0)
    for changed in ({**accepted, "logical_index": 1}, {**accepted, "payload_sha256": "e" * 64}):
        with pytest.raises(PermissionError, match="exact fitted checkpoint"):
            dispatcher._check_boundary(changed, p0)


def test_r14_dispatcher_rejects_all_prior_attempt_ids_and_fixed_repair(monkeypatch):
    dispatcher = mac.ContinuationDispatcher.__new__(mac.ContinuationDispatcher)
    dispatcher.scope = {"cached_local_attempt_id": gpu.LOCAL_ATTEMPT}
    valid = {"binding": {}, "logical_index": 0}
    monkeypatch.setattr(mac.RelayRetryDispatcher, "validate", lambda _, job: job)
    for identity in (gpu.repair.ATTEMPT, gpu.cached.ATTEMPT, gpu.LOCAL_ATTEMPT):
        with pytest.raises(PermissionError, match="cannot replay"):
            dispatcher.validate((valid, None, {"attempt_id": identity}))
    with pytest.raises(PermissionError, match="cannot replay"):
        dispatcher.validate(({**valid, "binding": {"fixed_reference_repair": {}}}, None, {"attempt_id": "fresh"}))
    job = (valid, None, {"attempt_id": "fresh"})
    assert dispatcher.validate(job) == job
