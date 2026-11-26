from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_6101_cached_r2 as cached


def _typed_plan():
    return {"choices": [{"candidate_id": "C-selected", "reason": "original explanation", "references": [
        {"reference_kind": "observed", "reference_id": "O-previous", "observed_id": "O-previous", "relationship": "exact"},
        {"reference_kind": "candidate", "reference_id": "C-compared", "candidate_id": "C-compared", "relationship": "distinct"},
    ]}]}


def _stage():
    return SimpleNamespace(context=SimpleNamespace(observed={"O-previous": {}}, candidates={"C-selected": {}, "C-compared": {}}))


def test_typed_mirror_removal_preserves_referenced_id_and_selection_order():
    original = _typed_plan()
    projected, paths = cached._project_mirrors(original, _stage())
    expected = deepcopy(original)
    del expected["choices"][0]["references"][0]["observed_id"]
    del expected["choices"][0]["references"][1]["candidate_id"]
    assert projected == expected
    assert paths == ["/choices/0/references/0/observed_id", "/choices/0/references/1/candidate_id"]
    assert original == _typed_plan()


@pytest.mark.parametrize(("index", "key", "value"), [
    (0, "observed_id", "O-foreign"),
    (0, "candidate_id", "C-selected"),
    (1, "candidate_id", "C-selected"),
    (1, "reference_kind", "observed"),
])
def test_import_stops_on_any_nonmirror_or_foreign_role(index, key, value):
    plan = _typed_plan()
    plan["choices"][0]["references"][index][key] = value
    with pytest.raises(PermissionError, match="authoritative reference ID"):
        cached._project_mirrors(plan, _stage())


def test_import_stops_if_the_canonical_candidate_reference_is_missing():
    plan = _typed_plan()
    stage = _stage()
    del stage.context.candidates["C-compared"]
    with pytest.raises(PermissionError, match="authoritative reference ID"):
        cached._project_mirrors(plan, stage)


def test_import_only_uses_local_receipt_and_denies_every_future_bridge_attempt(monkeypatch, tmp_path):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import backend

    stage = SimpleNamespace(binding={}, messages=[])
    run, old = tmp_path/"new", tmp_path/"old"
    archived = {"binding": {"seed": cached.SEED, "round": cached.ROUND}, "payload": {"input": [None, {"content": "pinned"}]}}
    frozen = _typed_plan()
    old_reply = {"text": "original", "response_id": cached.RESPONSE_ID, "served_model": "gpt-6.2-astra",
                 "usage": {"input_tokens": 10}, "status": "completed"}
    proof = {"cached_import": {"provider_post": False}, "bridge_wait_seconds": 7800,
             "previous_release_directory": "old-source", "model_identity": {}}
    contract = SimpleNamespace(predecessor=proof, transport=object(), rounds=36, gpu_slots=(0,), lease_root="lease",
                               record=lambda: {})
    bridge = lambda *_: pytest.fail("the cached round cannot reach the bridge")
    monkeypatch.setattr(cached, "PLAN_SHA", cached.digest(frozen))
    monkeypatch.setattr(cached, "make_request", lambda *_: archived)
    monkeypatch.setattr(cached, "_validation", lambda *_: SimpleNamespace(valid=True))
    monkeypatch.setattr(cached, "_old_receipt", lambda *_: None)
    monkeypatch.setattr(cached, "once", lambda *_: None)
    monkeypatch.setattr(cached, "read", lambda *_: {})
    monkeypatch.setattr(cached.scheduler, "_lock", lambda *_, **__: nullcontext())
    monkeypatch.setattr(cached.scheduler, "_assert_no_old_workers", lambda *_: None)
    monkeypatch.setattr(cached.scheduler, "_pause_requested", lambda *_: False)
    monkeypatch.setattr(cached.scheduler, "_atomic_status", lambda *_: None)
    monkeypatch.setattr(cached.scheduler, "_provider_wiring", lambda *_: (bridge, {"managed_attempt": bridge}))
    monkeypatch.setattr(cached.relay, "_retired", lambda *_, **__: (old, old))
    monkeypatch.setattr(cached.relay.bounded, "_archived", lambda *_: (object(), {}))
    monkeypatch.setattr(cached.seed_runner, "_scope", lambda *_: {})
    monkeypatch.setattr(cached.seed_runner, "_stage", lambda *_: stage)
    monkeypatch.setattr(cached.seed_runner, "pin_boundary_user_message", lambda *_: stage)
    monkeypatch.setattr(backend, "TrainingBackend", lambda *_: None)
    expected = cached.seed_runner.SeedProgress(cached.SEED, "complete", 3, 429)

    def run_seed(**options):
        local = options["transport_options"]["managed_attempt"]
        assert options["max_rounds"] == 3
        reply = local(archived, contract.transport, {"attempt": 0})
        assert reply.usage == {} and reply.transport_evidence["provider_dispatched"] is False
        assert cached.parse_response(reply.text) == cached.parse_response(cached.dumps(frozen))
        for request, candidate_attempt in ((archived, 1),
                ({**archived, "binding": {"seed": cached.SEED, "round": 3}}, 0)):
            with pytest.raises(cached.TransportStop, match="no provider transport"):
                local(request, contract.transport, {"attempt": candidate_attempt})
        return expected

    monkeypatch.setattr(cached.seed_runner, "run_seed", run_seed)
    result = cached.execute(tmp_path, run, old, contract, {}, (None, None, 2), None, archived, old_reply, frozen,
                            resume=False, import_only=True)
    assert result == expected
