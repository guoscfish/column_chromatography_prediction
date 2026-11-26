from dataclasses import asdict
import sys
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, once
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_relay_retry as retry


def test_only_registered_physical_budget_is_added():
    assert {seed: retry.execution_budget(36, lane.round) for seed, lane in retry.LANES.items()} == {
        157: {"first_logical_requests": 33, "maximum_logical_requests": 97, "maximum_physical_attempts": 194},
        2357: {"first_logical_requests": 32, "maximum_logical_requests": 94, "maximum_physical_attempts": 188},
        6101: {"first_logical_requests": 35, "maximum_logical_requests": 103, "maximum_physical_attempts": 206},
        12203: {"first_logical_requests": 32, "maximum_logical_requests": 94, "maximum_physical_attempts": 188},
    }


def test_retry_reproduces_exact_input_and_the_signed_host_only_reconstruction():
    request = {"binding": {"a7_formal": {"contract": "retired"}, retry.ALIAS: "prior", "checkpoint_sha256": "same"},
               "payload": {"input": [{"role": "user", "content": "signed bytes"}], "stream": True},
               "payload_sha256": "same", "logical_index": 0}
    new = {**request, "binding": {**request["binding"], "a7_formal": {"contract": "successor"}, retry.ALIAS: "new"}}
    reconstruction = {"previous_sha256": "prior", "reconstructed_sha256": "new"}
    assert retry._compare_science(new, request, reconstruction)
    assert not retry._compare_science(new, request)
    assert not retry._compare_science({**new, "logical_index": 1}, request, reconstruction)
    assert not retry._compare_science({**new, "payload": {**new["payload"], "stream": False}}, request, reconstruction)
    assert not retry._compare_science({**new, "binding": {**new["binding"], "checkpoint_sha256": "other"}},
                                      request, reconstruction)


def test_2357_import_uses_both_sealed_fits_after_previous_boundary(tmp_path, monkeypatch):
    lane = retry.LANES[2357]
    initial = tmp_path / "round_02"
    complete = tmp_path / "round_04"
    states = {index: retry.seed_runner.FrozenState(lane.seed, index, str(index), {}, [], [], {})
              for index in range(2, 5)}
    snapshots = {index: {"rounds": list(range(index))} for index in states}
    user = {"role": "user", "content": "exact archived boundary"}
    descriptor = {"round": 2, "state_sha256": digest(asdict(states[2])),
                  "memory_sha256": digest(snapshots[2]), "user_message_sha256": digest(user)}
    previous = SimpleNamespace(rounds=36, digest="retired", transport=None,
                               profile={"resume_boundaries": {str(lane.seed): descriptor}})
    approval = {"authorized_by": "operator"}
    once(initial / "source.json", asdict(states[2]))
    once(initial / "boundary_memory.json", snapshots[2])
    once(initial / "boundary_user_message.json", user)
    once(tmp_path / "boundary.json", {"formal_run": retry.seed_runner._scope(previous, approval, lane.seed, False),
                                     **descriptor})
    once(complete / "source.json", asdict(states[4]))
    steps = []

    def successor(_, state, follow, *__):
        index = state.round + 1
        steps.append(index)
        once(follow / "transition.json", {"round": index})
        return states[index], snapshots[index]

    monkeypatch.setitem(sys.modules, "studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.backend",
                        SimpleNamespace(TrainingBackend=lambda *_: object()))
    monkeypatch.setattr(retry.seed_runner, "_successor", successor)
    boundary, proof = retry._boundary(tmp_path, tmp_path, complete, previous, approval, lane, tmp_path)
    assert boundary == (states[4], snapshots[4], 4)
    assert steps == [3, 4]
    assert {"transition_03", "transition_04", "latest_source", "latest_memory"} <= proof.keys()
    with pytest.raises(PermissionError, match="latest completed signed fit"):
        retry._boundary(tmp_path, tmp_path, initial, previous, approval, lane, tmp_path)
