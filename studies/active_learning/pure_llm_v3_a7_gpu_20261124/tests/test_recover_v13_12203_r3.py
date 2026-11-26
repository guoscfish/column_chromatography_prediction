from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import once
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import recover_v13_12203_r3 as recover


def test_only_unused_slot1_and_reattach_or_completed_prefix(tmp_path):
    seed, active, turn = recover._paths(tmp_path)
    assert recover._inspect(tmp_path) == "waiting_for_original_terminal"
    once(turn / "attempt_00.terminal.json", {"status": "stopped"})
    assert recover._inspect(tmp_path) == "sealed_original_candidate"
    once(turn / "attempt_01.started.json", {"attempt": 1})
    with pytest.raises(PermissionError, match="already been reserved"):
        recover._inspect(tmp_path)
    once(turn / "attempt_00.replay_authorization.json", {"next_attempt": 1})
    assert recover._inspect(tmp_path) == "reattach_existing_replay"
    once(turn / "attempt_01.terminal.json", {"status": "stopped"})
    assert recover._inspect(tmp_path) == "physical_budget_exhausted"
    once(active / "batch_freeze.json", {"synthetic": "frozen"})
    assert recover._inspect(tmp_path) == "finish_frozen_R3"
    once(seed / "round_04/transition.json", {"synthetic": "fitted"})
    assert recover._inspect(tmp_path) == "R3_already_sealed_resume_normally"


def test_resume_does_not_reopen_r1_and_preserves_legacy_bridge_wait(tmp_path, monkeypatch):
    gpu = 0
    contract = SimpleNamespace(gpu_slots=(gpu,), rounds=36, transport=object(), lease_root=tmp_path)
    request = {"binding": {"seed": 12203, "round": 3}, "logical_index": 0}
    sent = []

    class Bridge:
        wait_seconds = 1

        def __call__(self, candidate, config, started):
            sent.append((candidate, config, started))

    bridge = Bridge()
    monkeypatch.setattr(recover.scheduler, "_provider_wiring", lambda *_: (None, {"managed_attempt": bridge}))

    def run_seed(**kwargs):
        assert bridge.wait_seconds == 18600
        options = kwargs["transport_options"]
        assert options["replay_authorizations"] == {recover.REQUEST: "r3 permit"}
        checked = options["managed_attempt"]
        with pytest.raises(recover.TransportStop, match="finished round"):
            checked({"binding": {"seed": 12203, "round": 1}}, contract.transport, {"attempt": 1})
        with pytest.raises(recover.TransportStop, match="only attempt 1"):
            checked(request, contract.transport, {"attempt": 0})
        checked(request, contract.transport, {"attempt": 1})
        checked({"binding": {"seed": 12203, "round": 4}, "logical_index": 0}, contract.transport, {"attempt": 0})
        return seed_runner.SeedProgress(12203, "complete", 36, 1485)

    monkeypatch.setattr(recover.seed_runner, "run_seed", run_seed)
    recover._resume(tmp_path / "worker.json", contract, {}, None, None, 18600, request, "r3 permit")
    assert [(candidate["binding"]["round"], start["attempt"]) for candidate, _, start in sent] == [(3, 1), (4, 0)]


def test_original_completion_is_preferred_to_retry(tmp_path):
    _, _, turn = recover._paths(tmp_path)
    once(turn / "attempt_00.terminal.json", {"status": "stopped"})
    once(turn / "attempt_00.reconciliation.json", {"decision": "import_completed"})
    assert recover._inspect(tmp_path) == "use_original_reconciliation"
