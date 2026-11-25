import json

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import file_digest, once
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import metrics


def test_auc_uses_raw_curve_without_smoothing_or_bridge():
    values = [{"active_labels": n, metrics.SCORE: v} for n, v in zip([333, 365, 397], [1.0, 0.5, 0.7])]
    assert metrics.aulc(values) == pytest.approx(0.675)
    assert metrics.aulc(values[:1]) is None
    with pytest.raises(ValueError, match="contiguous"):
        metrics.aulc(values[::2])


def test_snapshot_preserves_physical_logical_and_label_denominators(tmp_path):
    run = tmp_path
    seed = run / "seed_157"
    initial = seed / "initial"
    initial_metrics = initial / "anchor/validation_metrics.json"
    once(initial_metrics, {metrics.SCORE: 0.5, "training_seconds": 23.0})
    once(initial / "initial.json", {"fit_path": "anchor",
                                    "validation_metrics_sha256": file_digest(initial_metrics)})
    first = seed / "round_00/turn_00"
    once(first / "request.json", {})
    began = {"request_id": "mock-a", "attempt": 0, "attempt_id": "one"}
    once(first / "attempt_00.started.json", began)
    once(first / "attempt_00.terminal.json", {**began, "status": "retryable", "error": "http_429",
                                             "elapsed_seconds": 1.25})
    once(first / "attempt_01.started.json", {**began, "attempt": 1, "attempt_id": "two"})
    pending = metrics.snapshot(run)["seeds"][0]
    assert pending["latest_budget"] == 333
    assert pending["training_seconds_observed"] == 0
    assert pending["total_logical_requests_logged"] == 1
    assert pending["total_physical_attempts_started"] == 2
    assert pending["total_complete_logical_responses"] == 0
    assert pending["rounds"][0]["open_or_unreconciled_attempts"] == 1
    assert pending["transport_failures"] == {"http_429": 1}
    assert pending["first_response_legality"]["fraction"] is None


def test_local_repair_preservation_compares_full_locked_choices():
    locked = {"candidate_id": "C-1", "reason": "Reason with bound evidence",
              "references": [{"reference_id": "O-2"}]}
    changed = {"candidate_id": "C-2", "reason": "Original reason"}
    decision = {"state": "rejected"}
    answer = {"text": json.dumps({"choices": [locked, changed]})}
    summary = metrics._repair_metrics(
        [locked, {**changed, "reason": "Rewritten"}], {"decision": decision, "answer": answer},
        {"locked_choice_slots": [0, 1]},
    )
    assert summary == {"initial_locked_choices": 2, "retained_initial_legal_id_count": 2,
                       "retained_initial_legal_choice_count": 1}
