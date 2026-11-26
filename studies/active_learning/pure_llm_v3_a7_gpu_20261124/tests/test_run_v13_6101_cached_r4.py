from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_6101_cached_r4 as cached


def test_exact_mirrors_are_the_only_changes_to_all_choices(monkeypatch):
    stage = SimpleNamespace(context=SimpleNamespace(observed={"O-0001": {}}, candidates={"C-other": {}}, all_row_ids={}))
    choices = []
    for index in range(32):
        identity = f"C-{index:04d}"
        stage.context.all_row_ids[identity] = f"experiment-{index}"
        refs = [{"reference_kind": "observed", "reference_id": "O-0001", "observed_id": "O-0001"}]
        if index < 8:
            refs.append(dict(refs[0]))
        if index < 2:
            refs.append({"reference_kind": "candidate", "reference_id": "C-other", "candidate_id": "C-other"})
        choices.append({"candidate_id": identity, "reason": f"original reason {index}", "references": refs})
    plan = {"choices": choices, "hypothesis_updates": [{"original": True}]}
    normalized, paths = cached._project_mirrors(plan, stage)
    reasons = [item["reason"] for item in choices]
    ids = [item["candidate_id"] for item in choices]
    raw = {"id": "resp_test", "status": "completed", "obj": "fixture"}
    text = cached.dumps(plan)
    fixed = {"RAW": cached.digest(raw), "RESPONSE": raw["id"], "TEXT": cached.digest(text),
             "RAW_PLAN": cached.digest(plan), "PLAN": cached.digest(normalized), "PATHS": cached.digest(paths),
             "ORDER": cached.digest(ids), "REASONS": cached.digest(reasons),
             "FROZEN_ORDER": cached.digest([stage.context.all_row_ids[role] for role in ids])}
    for key, value in fixed.items():
        monkeypatch.setattr(cached, key, value)
    monkeypatch.setattr(cached, "response_output_text", lambda _: text)
    raw["output"] = []
    monkeypatch.setattr(cached, "RAW", cached.digest(raw))
    before = SimpleNamespace(valid=False, as_dict=lambda: {"valid": False, "errors": [{"code": "unexpected_field"}]})
    after = SimpleNamespace(valid=True, valid_choice_count=32)
    decision = {"state": "rejected", "diagnosis": before.as_dict()}
    monkeypatch.setattr(cached, "_validation", lambda _stage, candidate: before if candidate == plan else after)
    accepted, removed = cached._validated_projection(stage, raw, decision)
    assert accepted == normalized and removed == paths and len(removed) == 42
    assert [item["reason"] for item in accepted["choices"]] == reasons
    assert accepted["hypothesis_updates"] == plan["hypothesis_updates"]
    monkeypatch.setattr(cached, "_validation", lambda *_: SimpleNamespace(valid=False))
    failed = SimpleNamespace(valid=False, valid_choice_count=31)
    monkeypatch.setattr(cached, "_validation", lambda _stage, candidate: before if candidate == plan else failed)
    with pytest.raises(PermissionError, match="fully revalidated"):
        cached._validated_projection(stage, raw, decision)


def test_cached_fit_exposes_one_local_answer_and_no_bridge(monkeypatch, tmp_path):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import backend

    old, run = tmp_path / "old", tmp_path / "new"
    plan = {"choices": [{"candidate_id": "C-0001", "reason": "original"}]}
    archived = {"payload": {"input": [{"role": "system", "content": "system"}, {"role": "user", "content": "science"}]}}
    current = {**archived, "request_id": "signed local request"}
    reply = {"text": "exact old answer", "response_id": cached.RESPONSE, "served_model": "gpt-6.2-astra",
             "usage": {"input_tokens": 1000}, "status": "completed"}
    proof = {"cached_selection": {"provider_post": False}, "execution_budget": cached.BUDGET, "model_identity": {}}
    contract = SimpleNamespace(predecessor=proof, transport=object(), gpu_slots=(0,), lease_root="lease",
                               target_labels=493, record=lambda: {"formal": "local"})
    monkeypatch.setattr(cached, "PLAN", cached.digest(plan))
    monkeypatch.setattr(cached, "verify_authorization", lambda *_: None)
    monkeypatch.setattr(cached, "make_request", lambda *_: current)
    monkeypatch.setattr(cached, "_validation", lambda *_: SimpleNamespace(valid=True))
    monkeypatch.setattr(cached, "_sealed_parent", lambda *_, **__: None)
    monkeypatch.setattr(cached.scheduler, "_lock", lambda *_, **__: nullcontext())
    monkeypatch.setattr(cached.scheduler, "_assert_no_old_workers", lambda *_: None)
    monkeypatch.setattr(cached.scheduler, "_pause_requested", lambda *_: False)
    monkeypatch.setattr(cached.scheduler, "_provider_wiring", lambda *_: pytest.fail("constructed a provider"))
    monkeypatch.setattr(cached.seed_runner, "_scope", lambda *_: {})
    monkeypatch.setattr(cached.seed_runner, "_stage", lambda *_: SimpleNamespace(binding={}, messages=[]))
    monkeypatch.setattr(cached.seed_runner, "pin_boundary_user_message", lambda stage, *_: stage)
    monkeypatch.setattr(backend, "TrainingBackend", lambda *_: None)
    expected = cached.seed_runner.SeedProgress(6101, "complete", 5, 493)

    def run_seed(**options):
        local = options["transport_options"]["managed_attempt"]
        answer = local(current, contract.transport, {"attempt": 0})
        assert answer.usage == {} and answer.transport_evidence["provider_dispatched"] is False
        assert cached.parse_response(answer.text) == plan
        with pytest.raises(cached.TransportStop, match="exact archived decision"):
            local(current, contract.transport, {"attempt": 1})
        with pytest.raises(cached.TransportStop, match="no provider transport"):
            options["call"](current, contract.transport)
        return expected

    monkeypatch.setattr(cached.seed_runner, "run_seed", run_seed)
    result = cached.execute(tmp_path, old, run, contract, {}, (None, None, 4), None, archived, reply, plan, resume=False)
    assert result == expected and not (run / "bridge").exists()
