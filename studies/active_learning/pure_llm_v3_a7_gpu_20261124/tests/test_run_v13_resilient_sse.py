from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_resilient_sse as resilient


def test_only_exact_new_alias_marker_may_differ_from_original_body():
    payload = {"input": [{"role": "user", "content": "ready"}], "stream": True, "model": "gpt-6.2-astra"}
    binding = {"seed": 157, "round": 3, "checkpoint_sha256": "old", "a7_formal": {"scope": "old"}}
    archived = {"payload": payload, "payload_sha256": "same", "binding": binding, "request_id": "old"}
    request = {**archived, "request_id": "new", "logical_index": 0,
               "binding": {**binding, "a7_formal": {"scope": "new"}, resilient.ALIAS: True}}
    assert resilient._same_science(request, archived)
    assert not resilient._same_science({**request, "binding": {**binding, resilient.ALIAS: False}}, archived)
    assert not resilient._same_science({**request, "binding": {**request["binding"], "checkpoint_sha256": "bad"}}, archived)
    assert not resilient._same_science({**request, "payload": {**payload, "stream": False}}, archived)


def test_one_recovery_slot_then_original_future_budgets():
    model = SimpleNamespace(logical_replies=3, transport_attempts=2)
    parent = SimpleNamespace(rounds=36, transport=model)
    expectations = {157: (33, 97, 193), 887: (30, 88, 175), 2357: (34, 100, 199),
                    6101: (35, 103, 205), 12203: (32, 94, 187)}
    for seed, lane in resilient.LANES.items():
        assert tuple(resilient._budget(parent, lane.round).values()) == expectations[seed]


def test_shared_runner_only_allows_one_recovery_dispatch_then_normal_rounds(tmp_path, monkeypatch):
    lane = resilient.LANES[12203]
    config = object()
    contract = SimpleNamespace(gpu_slots=(0,), rounds=36, transport=config,
        predecessor={"bridge_wait_seconds": 7800, "prior_attempts": [{"attempt_id": "retired"}], "previous_attempts": []})
    request = {"binding": {"seed": lane.seed, "round": lane.round}, "logical_index": 0, "request_id": "approved"}
    sent = []

    class Bridge:
        wait_seconds = 1

        def __call__(self, request, config, started):
            sent.append((request["binding"]["round"], started["attempt"]))

    bridge = Bridge()
    monkeypatch.setattr(resilient.bounded, "_retired", lambda *_: None)
    monkeypatch.setattr(resilient.scheduler, "_provider_wiring", lambda *_: (None, {"managed_attempt": bridge}))

    def run_seed(**kwargs):
        assert bridge.wait_seconds == 7800
        checked = kwargs["transport_options"]["managed_attempt"]
        rejected = {"request_id": request["request_id"], "attempt": 1, "attempt_id": "new"}
        with pytest.raises(resilient.TransportStop, match="only the exact first"):
            checked(request, config, rejected)
        with pytest.raises(resilient.TransportStop, match="only the exact first"):
            checked({**request, "logical_index": 1}, config, {**rejected, "attempt": 0})
        with pytest.raises(resilient.TransportStop, match="only the exact first"):
            checked(request, config, {**rejected, "attempt": 0, "attempt_id": "retired"})
        checked(request, config, {**rejected, "attempt": 0})
        checked({**request, "binding": {"seed": lane.seed, "round": lane.round + 1}}, config, rejected)
        return resilient.seed_runner.SeedProgress(lane.seed, "complete", 36, 1485)

    monkeypatch.setattr(resilient.seed_runner, "run_seed", run_seed)
    contract.lease_root = tmp_path
    result = resilient._run(tmp_path, tmp_path, tmp_path, tmp_path, lane, contract, {}, None, None,
                            {"payload": {"input": [{}, {"role": "user", "content": "ready"}]}}, request)
    assert result.active_labels == 1485
    assert sent == [(4, 0), (5, 1)]
