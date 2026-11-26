from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import (
    dispatch_v13_retry_successor as mac,
    run_v13_retry_successor as gpu,
)


def test_only_explicitly_reviewed_new_wave_can_reserve_its_one_current_slot():
    class Parser:
        def error(self, message):
            raise ValueError(message)

    scope = gpu.read_scope(Parser(), SimpleNamespace(seed=12203, wave=3))
    assert scope.budget == {"first_logical_requests": 32, "maximum_logical_requests": 94,
                            "maximum_physical_attempts": 187}
    assert scope.not_before_unix - scope.cooldown_unix == 180
    assert len(scope.archive) == 2
    assert scope.archive[-1].failure == "connection_interrupted"
    for receipt in scope.archive:
        assert all(mac._sha(value) for value in (receipt.identity.contract, receipt.identity.approval, receipt.request_id,
                                                 receipt.job_sha256, receipt.result_sha256, receipt.terminal_file_sha256,
                                                 receipt.progress_file_sha256))
    with pytest.raises(ValueError, match="no individually reviewed"):
        gpu.read_scope(Parser(), SimpleNamespace(seed=12203, wave=4))


@pytest.mark.parametrize("seed,rev_counts,physical_budget", [(12203, [1, 1], 187), (157, [2, 1], 193)])
def test_each_prior_identity_and_new_second_slot_are_rejected(monkeypatch, seed, rev_counts, physical_budget):
    scope = gpu.SCOPES[(seed, 3)]
    assert [len(wave.attempts) for wave in scope.archive] == rev_counts
    assert scope.budget["maximum_physical_attempts"] == physical_budget
    approved = []
    monkeypatch.setattr(mac.RelayRetryDispatcher, "_check_boundary",
                        lambda self, request, started: approved.append(started))
    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    new = {"attempt": 0, "attempt_id": "b" * 32}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="last cooldown"):
        dispatcher._check_boundary({}, new)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    retired = [{**new, "attempt_id": item.attempt_id} for wave in scope.archive for item in wave.attempts]
    for denied in [{**new, "attempt": 1}, *retired]:
        with pytest.raises(PermissionError, match="one distinct physical attempt"):
            dispatcher._check_boundary({}, denied)
    dispatcher._check_boundary({}, new)
    assert approved == [new]
