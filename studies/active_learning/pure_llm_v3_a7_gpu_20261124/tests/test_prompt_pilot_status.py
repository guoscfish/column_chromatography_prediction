import json

from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.prompt_pilot_status import (
    local_progress, pending_attempt,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def put_progress(spool, identity, round_index, **events):
    attempt = spool / "attempts" / identity
    job_sha = "6" * 64
    write_json(attempt / "started.json", {"attempt_id": identity, "job_sha256": job_sha})
    write_json(attempt / "provider_progress.json", {
        "attempt_id": identity, "job_sha256": job_sha, "seed": 887, "round": round_index,
        "logical_index": 0, "phase": "sse_event", **events,
    })
    return attempt, job_sha


def test_waiting_attempt_does_not_display_old_sealed_progress(tmp_path):
    spool, run = tmp_path / "mac", tmp_path / "gpu"
    old, pending = "1" * 32, "2" * 32
    put_progress(spool, old, 3, event_type="response.completed")
    write_json(spool / "results" / f"{old}.json", {"attempt_id": old, "job_sha256": "6" * 64})
    directory = run / "seed_887"
    write_json(directory / "round_04/turn_00/attempt_00.started.json", {"attempt_id": pending, "attempt": 0})

    active = pending_attempt(run, directory, 4)
    assert active["attempt_id"] == pending
    assert local_progress(spool, 887, active) == {}

    put_progress(spool, pending, 4, event_type="response.output_item.done", http_status=200)
    state = local_progress(spool, 887, active)
    assert state["http_status"] == 200
    assert state["first_output_seconds"] is None
    assert not state["complete_event_seen"]
    assert not state["sealed"]


def test_reports_matching_completion_and_delivery_separately(tmp_path):
    identity = "3" * 32
    active = {"attempt_id": identity, "round": 4, "turn": 0}
    attempt, job_sha = put_progress(tmp_path, identity, 4, event_type="response.completed", sse_milestones={
        "response.output_text.delta": {"elapsed_seconds": 640.5}, "response.completed": {"elapsed_seconds": 702},
    })
    write_json(attempt / "raw_provider_response.json", {})
    write_json(attempt / "provider_capture_receipt.json", {})
    result_sha = "7" * 64
    write_json(tmp_path / "results" / f"{identity}.json", {
        "attempt_id": identity, "job_sha256": job_sha, "result_sha256": result_sha,
    })
    saved = local_progress(tmp_path, 887, active)
    assert saved["first_output_seconds"] == 640.5
    assert saved["complete_event_seen"] and saved["raw_response_saved"] and saved["sealed"]
    assert not saved["delivered"]

    write_json(tmp_path / "delivered" / f"{identity}.json", {
        "attempt_id": identity, "result_sha256": result_sha,
    })
    assert local_progress(tmp_path, 887, active)["delivered"]


def test_completed_transport_can_still_fail_scientific_validation(tmp_path):
    identity = "4" * 32
    directory = tmp_path / "seed_887"
    turn = directory / "round_04/turn_00"
    write_json(turn / "attempt_00.started.json", {"attempt_id": identity, "attempt": 0})
    write_json(turn / "attempt_00.terminal.json", {"status": "completed"})
    write_json(turn / "decision.json", {"state": "rejected", "diagnosis": {"errors": [
        {"code": "unexpected_field", "path": "/choices/0/references/0"},
        {"code": "unexpected_field", "path": "/choices/1/references/0"},
        {"code": "multifactor_mismatch", "path": "/choices/2/references/0/relationship"},
    ]}})

    active = pending_attempt(tmp_path, directory, 4)
    assert active["status"] == "completed"
    assert active["selection"] == "rejected"
    assert active["selection_errors"] == {"unexpected_field": 2, "multifactor_mismatch": 1}
