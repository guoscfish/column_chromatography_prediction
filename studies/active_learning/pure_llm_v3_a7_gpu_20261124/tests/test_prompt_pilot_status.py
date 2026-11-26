import hashlib
import json

from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.prompt_pilot_status import (
    UNSTARTED_ANCHORS, local_progress, pending_attempt, remote_seed,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import prompt_pilot_status


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


def test_dispatch_wait_is_separate_from_provider_time(tmp_path, monkeypatch):
    identity = "5" * 32
    active = {
        "attempt_id": identity, "round": 4, "turn": 0,
        "reserved_unix": 1600000000, "status": "waiting_for_mac",
    }
    monkeypatch.setattr(prompt_pilot_status.time, "time", lambda: 1600000070)
    assert local_progress(tmp_path, 887, active, track_queue=True) == {
        "phase": "queued_for_dispatch", "dispatch_wait_seconds": 70,
    }

    attempt, job_sha = put_progress(tmp_path, identity, 4, elapsed_seconds=22, last_sse_interval_seconds=11)
    write_json(attempt / "started.json", {
        "attempt_id": identity, "job_sha256": job_sha, "started_utc": "2020-09-13T12:27:00+00:00",
    })
    state = local_progress(tmp_path, 887, active, track_queue=True)
    assert state["dispatch_wait_seconds"] == 20
    assert state["provider_elapsed_seconds"] == 50
    assert state["last_sse_interval_seconds"] == 11
    assert not state["complete_event_seen"]


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



def test_unstarted_seeds_show_only_verified_original_anchor(tmp_path, monkeypatch):
    runs = tmp_path / "purellm_a7/runs"
    checkpoint = (tmp_path / "formal_cuda/row4g_b32_20261124/trajectories"
                  / "random-6101/rounds/random/round_00/model/best.pt")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"original checkpoint")
    monkeypatch.setitem(UNSTARTED_ANCHORS, 6101, hashlib.sha256(checkpoint.read_bytes()).hexdigest())

    state = remote_seed(runs, 6101)
    assert (state["phase"], state["labels"], state["completed_rounds"]) == ("l333_anchor_verified", 333, 0)
    assert state["run"] is None and not state["worker_active"]

    checkpoint.write_bytes(b"changed checkpoint")
    state = remote_seed(runs, 6101)
    assert state["phase"] == "l333_anchor_unverified"
    assert state["labels"] is None and state["checkpoint_sha256"] is None


def test_2357_extension_preserves_parent_labels_and_other_seeds(tmp_path):
    cohort = tmp_path / prompt_pilot_status.SERIAL_IPV4
    parent = {
        "run_directory": str(cohort),
        "profile": {"confirmation_seeds": list(prompt_pilot_status.SEEDS), "provider_capability": "gated_ipv4"},
        "predecessor": {"dispatcher_module": "dispatch_prepared_ipv4",
                        "lanes": {"2357": {"start_round": 1, "checkpoint_sha256": "9" * 64}}},
    }
    parent_sha = hashlib.sha256(json.dumps(parent, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write_json(cohort / "formal_contract.json", parent)
    write_json(cohort / "authorization.json", {"contract_sha256": parent_sha})
    extension = tmp_path / prompt_pilot_status.IPV4_EXTENSION
    child = {
        "run_directory": str(extension), "profile": {"confirmation_seeds": [2357]},
        "predecessor": {"parent_contract_sha256": parent_sha,
                        "dispatcher_module": "dispatch_prepared_ipv4_recovery_2357"},
    }
    child_sha = hashlib.sha256(json.dumps(child, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write_json(extension / "formal_contract.json", child)
    write_json(extension / "authorization.json", {"contract_sha256": child_sha})

    status = remote_seed(tmp_path, 2357)
    assert (status["run"], status["labels"], status["target_labels"]) == (extension.name, 365, 1485)
    assert remote_seed(tmp_path, 887)["run"] == cohort.name

    continuation = tmp_path / prompt_pilot_status.IPV4_CONTINUATION
    follow_on = {
        "run_directory": str(continuation), "profile": {"confirmation_seeds": [2357]},
        "predecessor": {"parent_contract_sha256": parent_sha, "extension_contract_sha256": child_sha,
                        "dispatcher_module": "dispatch_prepared_ipv4_continue_2357"},
    }
    follow_on_sha = hashlib.sha256(json.dumps(follow_on, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write_json(continuation / "formal_contract.json", follow_on)
    write_json(continuation / "authorization.json", {"contract_sha256": follow_on_sha})
    assert remote_seed(tmp_path, 2357)["run"] == continuation.name

    r2_extension = tmp_path / prompt_pilot_status.IPV4_R2_EXTENSION
    r2 = {
        "run_directory": str(r2_extension), "profile": {"confirmation_seeds": [2357]},
        "predecessor": {"parent_contract_sha256": follow_on_sha,
                        "dispatcher_module": "dispatch_v13_2357_r2_sse_extension"},
    }
    r2_sha = hashlib.sha256(json.dumps(r2, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write_json(r2_extension / "formal_contract.json", r2)
    write_json(r2_extension / "authorization.json", {"contract_sha256": r2_sha})
    assert remote_seed(tmp_path, 2357)["run"] == r2_extension.name
