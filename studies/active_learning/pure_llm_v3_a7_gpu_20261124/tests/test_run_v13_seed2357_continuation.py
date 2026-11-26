from dataclasses import asdict, dataclass
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, file_digest, once
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.run_v13_seed2357_continuation import (
    _budget, _completion, _retirement,
)


def test_only_the_unspent_rounds_are_carried_forward():
    parent = SimpleNamespace(rounds=36, transport=SimpleNamespace(logical_replies=3, transport_attempts=2))
    assert _budget(parent, 2) == {"first_logical_requests": 34, "maximum_logical_requests": 102,
                                  "maximum_physical_attempts": 204}
    for round_index in (0, 1, 3):
        with pytest.raises(PermissionError, match="only a verified L397"):
            _budget(parent, round_index)


def test_completion_binds_checkpoint_memory_and_transition(tmp_path):
    @dataclass
    class State:
        seed: int = 2357
        round: int = 2
        checkpoint: str = "a" * 64

    contract = SimpleNamespace(digest="b" * 64)
    state = State()
    memory = {"rounds": {"0": {}, "1": {}}}
    transition = tmp_path / "transition.json"
    completion = tmp_path / "extension_completion.json"
    once(transition, {"synthetic": "cuda fit receipt"})
    budget = {"first_logical_requests": 34, "maximum_logical_requests": 102, "maximum_physical_attempts": 204}
    expected = {"contract_sha256": contract.digest, "seed": 2357, "round": 2, "active_labels": 397,
                "checkpoint_sha256": state.checkpoint, "state_sha256": digest(asdict(state)),
                "memory_sha256": digest(memory), "transition_sha256": file_digest(transition),
                "original_future_budget": budget}
    once(completion, expected)
    assert _completion(completion, contract, state, memory, transition, budget) == expected
    with pytest.raises(PermissionError, match="verified L397"):
        _completion(completion, contract, state, {"rounds": {"0": {}}}, transition, budget)
    transition.write_text('{"synthetic": "different fit receipt"}')
    with pytest.raises(PermissionError, match="verified L397"):
        _completion(completion, contract, state, memory, transition, budget)


def test_original_seed_must_remain_retired_before_continuation(tmp_path):
    with pytest.raises(PermissionError, match="pause only"):
        _retirement(tmp_path)
    seed = tmp_path / "seed_2357"
    seed.mkdir()
    marker = seed / "pause.request"
    marker.write_text("original lane retired")
    proof = _retirement(tmp_path)
    assert _retirement(tmp_path, expected=proof) == proof

    marker.write_text("different pause marker")
    with pytest.raises(PermissionError, match="marker changed"):
        _retirement(tmp_path, expected=proof)
    (seed / "round_01").mkdir()
    once(seed / "round_01" / "batch_freeze.json", {"unexpected": True})
    with pytest.raises(PermissionError, match="already has selection"):
        _retirement(tmp_path)
