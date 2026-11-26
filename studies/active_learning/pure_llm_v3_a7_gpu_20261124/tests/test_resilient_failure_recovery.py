import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import once
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import (
    dispatch_v13_resilient_failure_recovery as mac,
    run_v13_resilient_failure_recovery as gpu,
)


@pytest.mark.parametrize("seed", tuple(gpu.FAILURES))
def test_recovery_binds_immediate_receipt_to_original_science(seed):
    failed = gpu.FAILURES[seed]
    payload = {"input": [{"role": "user", "content": "same candidate data"}], "stream": True}
    original = {"binding": {"checkpoint_sha256": "same", "a7_formal": "original"}, "request_id": "original",
                "logical_index": 0, "payload": payload, "payload_sha256": "same"}
    request = {**original, "request_id": failed.request_id, "binding": {**original["binding"], "a7_formal": "new",
                                                                      gpu.resilient.ALIAS: True}}
    receipt = {"attempt_id": failed.attempt_id, "job_sha256": failed.job_sha256, "result_sha256": failed.result_sha256,
               "error": failed.code, "provider_dispatched": failed.provider_dispatched,
               "ambiguous": failed.provider_dispatched}
    previous = {"previous_contract_sha256": "older", "previous_attempts": ["sealed"],
                "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}}
    proof = gpu._failure_scope(failed, previous, request, original, receipt)
    assert proof["previous_contract_sha256"] == failed.contract_sha256
    assert proof["previous_attempts"] == [receipt]
    assert proof["archived_previous"] == {"contract_sha256": "older", "attempts": ["sealed"]}
    assert proof["recovery_budget"] == previous["recovery_budget"]
    with pytest.raises(PermissionError):
        gpu._failure_scope(failed, previous, {**request, "payload": {**payload, "stream": False}}, original, receipt)
    with pytest.raises(PermissionError):
        gpu._failure_scope(failed, previous, request, original, {**receipt, "provider_dispatched": False if failed.provider_dispatched else True})


def test_gpu_only_extra_reservation_must_have_no_bridge_or_terminal(tmp_path):
    failed = gpu.FAILURES[157]
    run = tmp_path / "archive"
    active = run / "seed_157" / "round_03"
    turn = active / "turn_00"
    once(turn / "attempt_00.started.json", {})
    once(turn / "attempt_01.started.json", {"attempt": 1, "attempt_id": failed.aborted_attempt_id,
                                          "request_id": failed.request_id, "transport": "mac_bridge"})
    _, aborted = gpu._failed_turn(run, active, failed)
    assert aborted["attempt_id"] == failed.aborted_attempt_id and aborted["provider_dispatched"] is False
    once(run / "bridge" / "outbox" / f"{failed.aborted_attempt_id}.json", {})
    with pytest.raises(PermissionError, match="before its bridge submission"):
        gpu._failed_turn(run, active, failed)


def test_mac_no_post_requires_three_exact_receipts_and_no_started(tmp_path):
    failed = gpu.FAILURES[157]
    attempt = tmp_path / "attempts" / failed.attempt_id
    for window in range(1, 4):
        once(attempt / f"readiness_{window:02d}.json", {"attempt_id": failed.attempt_id,
            "job_sha256": failed.job_sha256, "window": window, "retry_unix": float(window)})
    mac._no_model_post(tmp_path, attempt, failed)
    once(attempt / "started.json", {})
    with pytest.raises(PermissionError, match="possible old model POST"):
        mac._no_model_post(tmp_path, attempt, failed)
