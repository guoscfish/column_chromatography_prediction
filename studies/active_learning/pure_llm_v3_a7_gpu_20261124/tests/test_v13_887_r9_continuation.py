from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import dumps
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import dispatch_v13_887_r9_continuation as mac
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_887_r9_continuation as gpu


def test_completed_correction_requires_all_l621_receipts(tmp_path, monkeypatch):
    worker = tmp_path / "workers/seed_887.json"
    worker.parent.mkdir()
    worker.write_text(dumps({"status": "bridge_pending", "completed_rounds": 8, "active_labels": 589}))
    with pytest.raises(PermissionError, match="fully fitted"):
        gpu._require_complete(tmp_path)
    worker.write_text(dumps({"status": "complete", "completed_rounds": 9, "active_labels": 621}))
    with pytest.raises(PermissionError, match="receipts"):
        gpu._require_complete(tmp_path)
    following = tmp_path / "seed_887/round_09"
    following.mkdir(parents=True)
    for name in ("source.json", "transition.json", "authorized_feedback.json", "fit_authorization.json"):
        (following / name).write_text("{}")
    checked = []
    monkeypatch.setattr(gpu.scheduler, "_assert_no_old_workers", lambda run, seeds: checked.append((run, seeds)))
    gpu._require_complete(tmp_path)
    assert checked == [(tmp_path, (887,))]


def test_normal_profile_retains_regular_recipe_with_finite_first_round():
    regular = {"target_labels": 1485, "normalize_observed_reference_aliases": True, "replay_eligible_from_round": 9,
               "confirmation_seeds": [887], "resume_boundaries": {"887": {"round": 8}},
               "transport": {"requested_model": "gpt-6.2-astra", "reasoning_effort": "xhigh", "sdk_version": "2.28.0",
                             "transport_attempts": 2, "logical_replies": 3}}
    descriptor = {"round": 9, "state_sha256": "fitted state", "memory_sha256": "fitted memory"}
    profile = gpu._original_profile(SimpleNamespace(profile=regular), descriptor)
    assert profile["transport"] == regular["transport"]
    assert profile["target_labels"] == 1485 and profile["resume_boundaries"] == {"887": descriptor}
    assert profile["replay_eligible_from_round"] == 9 and profile["sealed_relay_retry_seconds"] == 180
    assert "fixed_reference_repair" not in profile
    assert gpu.BOUNDARY_BUDGET == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert gpu.BUDGET == {"first_logical_requests": 27, "maximum_logical_requests": 1 + 26 * 3,
                          "maximum_physical_attempts": 2 + 26 * 3 * 2}


def test_second_physical_r9_attempt_requires_shared_exact_replay_guard():
    dispatcher = object.__new__(mac.ContinuationDispatcher)
    dispatcher.scope = {"checkpoint_sha256": "r9 fitted checkpoint"}
    calls = []
    dispatcher._check_boundary_retry = lambda request, started: calls.append((request, started))
    request = {"binding": {"checkpoint_sha256": "r9 fitted checkpoint"}, "logical_index": 0}
    dispatcher._check_boundary(request, {"attempt": 0})
    assert not calls
    with pytest.raises(PermissionError, match="one logical decision"):
        dispatcher._check_boundary({**request, "logical_index": 1}, {"attempt": 0})
    with pytest.raises(PermissionError, match="checkpoint"):
        dispatcher._check_boundary({**request, "binding": {"checkpoint_sha256": "older checkpoint"}}, {"attempt": 0})
    retry = {"attempt": 1}
    dispatcher._check_boundary(request, retry)
    assert calls == [(request, retry)]
