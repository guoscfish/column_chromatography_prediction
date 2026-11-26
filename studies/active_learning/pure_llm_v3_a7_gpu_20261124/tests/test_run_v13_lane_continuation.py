from dataclasses import asdict, dataclass
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, file_digest, once
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_lane_continuation as continuation


@pytest.mark.parametrize(("seed", "labels", "future"), ((6101, 397, 34), (887, 557, 29)))
def test_only_unused_paid_budget_after_exact_fit(tmp_path, seed, labels, future):
    lane = continuation.extension.LANES[seed]
    parent = SimpleNamespace(rounds=36, transport=SimpleNamespace(logical_replies=3, transport_attempts=2))
    budget = continuation._budget(parent, lane, lane.round + 1)
    assert budget == {"first_logical_requests": future, "maximum_logical_requests": future * 3,
                      "maximum_physical_attempts": future * 3 * 2}
    with pytest.raises(PermissionError, match="next fitted round"):
        continuation._budget(parent, lane, lane.round)

    @dataclass
    class State:
        seed: int
        round: int
        checkpoint: str

    state = State(seed, lane.round + 1, "a" * 64)
    memory = {"rounds": {str(i): {} for i in range(state.round)}}
    one_shot = SimpleNamespace(digest="b" * 64, target_labels=labels)
    receipt = tmp_path / "transition.json"
    marker = tmp_path / "extension_completion.json"
    once(receipt, {"synthetic": "verified fit"})
    expected = {"contract_sha256": one_shot.digest, "seed": seed, "round": lane.round + 1,
                "active_labels": labels, "checkpoint_sha256": state.checkpoint, "state_sha256": digest(asdict(state)),
                "memory_sha256": digest(memory), "transition_sha256": file_digest(receipt),
                "original_future_budget": budget}
    once(marker, expected)
    assert continuation._completion(marker, one_shot, lane, state, memory, receipt, budget) == expected
    receipt.write_text('{"synthetic": "different fit"}')
    with pytest.raises(PermissionError, match="verified CUDA transition"):
        continuation._completion(marker, one_shot, lane, state, memory, receipt, budget)


def test_exact_one_shot_contract_and_authorization_must_both_match(tmp_path):
    record = {"synthetic": "one physical attempt"}
    contract = SimpleNamespace(digest=digest(record), record=lambda: record)
    approval = {"scope": "pure_llm_v3_a7_formal", "contract_sha256": contract.digest,
                "authorize_paid_api": True, "authorize_label_reveal": True, "authorize_cuda_training": True,
                "authorized_by": "offline fixture", "authorization_reference": "test scope"}
    once(tmp_path / "formal_contract.json", record)
    once(tmp_path / "authorization.json", approval)
    assert continuation._extension_approval(contract, tmp_path, contract.digest, digest(approval)) == approval
    with pytest.raises(PermissionError, match="differs"):
        continuation._extension_approval(contract, tmp_path, contract.digest, "d" * 64)
    with pytest.raises(PermissionError, match="exact released"):
        continuation._extension_approval(contract, tmp_path, "bad", digest(approval))



def test_only_887_continuation_carries_verified_host_serialization_proof():
    pair = {"archived_sha256": "a" * 64, "rebuilt_sha256": "b" * 64}
    signed = SimpleNamespace(predecessor={"authoritative_message_reconstruction": pair})
    assert continuation._reconstruction_proof(signed, continuation.extension.LANES[887]) == {
        "authoritative_message_reconstruction": pair,
    }
    assert continuation._reconstruction_proof(SimpleNamespace(predecessor={}), continuation.extension.LANES[6101]) == {}
    with pytest.raises(PermissionError, match="invalid host serialization"):
        continuation._reconstruction_proof(signed, continuation.extension.LANES[6101])
    pair["rebuilt_sha256"] = pair["archived_sha256"]
    with pytest.raises(PermissionError, match="invalid host serialization"):
        continuation._reconstruction_proof(signed, continuation.extension.LANES[887])


def test_887_reconstruction_requires_fixed_hashing_at_process_start():
    source = (
        "from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts"
        " import run_v13_lane_continuation as c" + chr(10) +
        "c._require_fixed_reconstruction(c.extension.LANES[887])"
    )
    for value, ok in (("0", True), ("1", False)):
        command = subprocess.run([sys.executable, "-B", "-c", source], capture_output=True,
                                 text=True, env={**os.environ, "PYTHONHASHSEED": value})
        assert (command.returncode == 0) is ok
        if not ok:
            assert "start this 887 continuation with PYTHONHASHSEED=0" in command.stderr
