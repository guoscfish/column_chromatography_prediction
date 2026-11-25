"""Offline recovery decisions for a quiesced managed seed worker."""
from types import SimpleNamespace
import os

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import once
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import watch_transport_replay as watch


SEED, ROUND = 887, 8
CONTRACT = SimpleNamespace(transport=SimpleNamespace(logical_replies=3))


def plan(run, *, status="initializing"):
    return watch.resume_plan(run, run / f"seed_{SEED}", ROUND, {"seed": SEED, "status": status},
                             CONTRACT, {}, "fixture")


def turn_at(run, logical=0, *, attempt=None, result=None):
    turn = run / f"seed_{SEED}" / f"round_{ROUND:02d}" / f"turn_{logical:02d}"
    once(turn / "request.json", {"binding": {"seed": SEED, "round": ROUND},
                                 "logical_index": logical, "request_id": f"fixture-{logical}"})
    if attempt is not None:
        once(turn / f"attempt_{attempt:02d}.started.json", {"attempt_id": "a" * 32})
        if result is not None:
            once(turn / f"attempt_{attempt:02d}.terminal.json", result)
    return turn


def test_interrupted_boundary_and_unreserved_request(tmp_path):
    round_dir = tmp_path / f"seed_{SEED}" / f"round_{ROUND:02d}"
    assert plan(tmp_path)[0] == "stop"
    once(round_dir / "source.json", {"sealed": True})
    assert plan(tmp_path)[0] == "run"
    assert plan(tmp_path, status="failed")[0] == "stop"
    turn_at(tmp_path)
    assert plan(tmp_path)[1] == "resume the staged request before its first reservation"
    assert plan(tmp_path, status="failed")[0] == "stop"


@pytest.mark.parametrize("attempt", [0, 1])
def test_reuse_completed_receipt_only_after_interruption(tmp_path, attempt):
    turn = turn_at(tmp_path, logical=2, attempt=attempt, result={"status": "completed"})
    assert plan(tmp_path)[1] == "process the existing completed response"
    assert plan(tmp_path, status="failed")[0] == "stop"
    once(turn / "decision.json", {"state": "rejected"})
    assert plan(tmp_path)[:2] == ("stop", "logical response budget exhausted")


def test_ordinary_retry_resumes_with_budget_and_no_new_permit(tmp_path, monkeypatch):
    monkeypatch.setattr(watch, "authorize_ambiguous_replay", lambda *a, **kw: pytest.fail("unexpected alternate permit"))
    terminal = {"status": "retryable", "ambiguous": False, "circuit_open": False, "error": "http_429"}
    turn_at(tmp_path, attempt=0, result=terminal)
    assert plan(tmp_path) == ("run", "resume the already authorized ordinary second attempt", {})
    assert plan(tmp_path, status="failed")[0] == "stop"
    turn_at(tmp_path, attempt=1, result={"status": "stopped"})
    assert plan(tmp_path)[:2] == ("stop", "physical attempt budget exhausted")


def test_frozen_batch_and_reserved_courier_recover_without_replay(tmp_path):
    turn = turn_at(tmp_path, attempt=0)
    assert plan(tmp_path)[1] == "reattach to the original physical attempt"
    once(tmp_path / "bridge/outbox" / ("a" * 32 + ".json"), {})
    assert plan(tmp_path)[0] == "wait"
    assert plan(tmp_path, status="failed")[0] == "stop"
    once(turn.parent / "batch_freeze.json", {})
    assert plan(tmp_path, status="interrupted")[1] == "resume the already frozen batch"
    assert plan(tmp_path, status="failed")[0] == "stop"


def test_orphan_terminal_fails_closed(tmp_path):
    turn = turn_at(tmp_path)
    once(turn / "attempt_01.terminal.json", {"status": "completed"})
    with pytest.raises(PermissionError, match="no archived attempt reservation"):
        plan(tmp_path)


def test_initializing_marker_cannot_override_newer_failure_event(tmp_path):
    worker = tmp_path / "workers" / f"seed_{SEED}.json"
    directory = tmp_path / f"seed_{SEED}"
    once(worker, {"seed": SEED, "status": "initializing", "pid": 43})
    os.utime(worker, ns=(100_000_000_000, 100_000_000_000))
    old = directory / "events/old.json"
    once(old, {"status": "failed", "error_type": "ValueError"})
    os.utime(old, ns=(99_000_000_000, 99_000_000_000))
    assert watch._read_worker(tmp_path, directory, SEED)["status"] == "initializing"
    current = directory / "events/current.json"
    once(current, {"status": "failed", "error_type": "ValueError"})
    assert watch._read_worker(tmp_path, directory, SEED)["status"] == "failed"


def launcher_inputs():
    policy = {"launcher_sha256": "a" * 64, "transport_replacement_sha256": "b" * 64,
              "runs": {"/reviewed/run": "c" * 64}}
    policy_sha = watch.digest(policy)
    new = "d" * 64
    approval = {"schema": "a7_rolling_launcher_update_v1", "policy_sha256": policy_sha,
                "old_launcher_sha256": policy["launcher_sha256"], "new_launcher_sha256": new,
                "transport_replacement_sha256": policy["transport_replacement_sha256"], "runs": policy["runs"],
                "authorized_by": "fixture", "authorization_reference": "reviewed watcher only"}
    return policy, policy_sha, new, approval


def test_launcher_update_leaves_original_policy_usable_and_demands_exact_receipt():
    policy, policy_sha, new, approval = launcher_inputs()
    watch.verify_launcher(policy, policy_sha, policy["launcher_sha256"])
    watch.verify_launcher(policy, policy_sha, new, approval, watch.digest(approval))
    with pytest.raises(PermissionError, match="differs"):
        watch.verify_launcher(policy, policy_sha, new)
    with pytest.raises(PermissionError, match="both"):
        watch.verify_launcher(policy, policy_sha, new, approval=approval)
    with pytest.raises(PermissionError, match="both"):
        watch.verify_launcher(policy, policy_sha, new, approval_sha=watch.digest(approval))
    with pytest.raises(PermissionError, match="must pin"):
        watch.verify_launcher(policy, policy_sha, new, approval, "e" * 64)


@pytest.mark.parametrize("field,value", [
    ("runs", {"/unapproved/run": "c" * 64}),
    ("policy_sha256", "e" * 64),
    ("old_launcher_sha256", "e" * 64),
    ("new_launcher_sha256", "e" * 64),
    ("transport_replacement_sha256", "e" * 64),
    ("authorization_reference", ""),
    ("authorized_by", ""),
])
def test_launcher_update_rejects_altered_scope_even_with_matching_digest(field, value):
    policy, policy_sha, new, approval = launcher_inputs()
    altered = {**approval, field: value}
    with pytest.raises(PermissionError, match="must pin"):
        watch.verify_launcher(policy, policy_sha, new, altered, watch.digest(altered))


def test_same_durable_progress_cannot_restart_twice(tmp_path):
    directory = tmp_path / f"seed_{SEED}"
    active = directory / f"round_{ROUND:02d}"
    next_round = directory / f"round_{ROUND+1:02d}"
    once(active / "source.json", {"sealed": True})
    assert watch.reserve_resume(tmp_path, directory, ROUND, "first", {})
    once(next_round / "fit_000/fit_scope.json", {})
    once(next_round / "fit_001/fit_scope.json", {})
    once(tmp_path / "workers" / f"seed_{SEED}.json", {"new_pid": True})
    assert not watch.reserve_resume(tmp_path, directory, ROUND, "different reason", {})
    once(next_round / "fit_001/training_complete.json", {"sealed_optimizer": True})
    assert watch.reserve_resume(tmp_path, directory, ROUND, "recover sealed optimizer", {})
    assert not watch.reserve_resume(tmp_path, directory, ROUND, "same seal", {})
    assert len(list((tmp_path / "operator/rolling_resume_markers").glob("*.json"))) == 2


def test_new_mac_result_counts_as_durable_progress(tmp_path):
    directory = tmp_path / f"seed_{SEED}"
    turn_at(tmp_path, attempt=0)
    assert watch.reserve_resume(tmp_path, directory, ROUND, "original reservation", {})
    assert not watch.reserve_resume(tmp_path, directory, ROUND, "unchanged", {})
    once(tmp_path / "bridge/results" / ("a" * 32 + ".json"), {"returned": True})
    assert watch.reserve_resume(tmp_path, directory, ROUND, "existing returned result", {})
    assert not watch.reserve_resume(tmp_path, directory, ROUND, "unchanged", {})
