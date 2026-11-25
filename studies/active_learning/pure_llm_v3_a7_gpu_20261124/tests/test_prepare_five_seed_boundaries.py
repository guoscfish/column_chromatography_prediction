"""Check that five-seed preparation cannot replace an existing decision or anchor."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, once, read
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import prepare_five_seed_boundaries as prep


def initial_state(seed):
    return prep.seed_runner.FrozenState(seed, 0, "a" * 64, {"validation_test_records": 0}, [], [], {
        "expected_observed": ["l"], "expected_candidates": ["u"], "split_sha256": "split",
        "resource_receipt_sha256": "resources",
    })


class AnchorBackend:
    l0_ids = ("l",)
    u0_ids = ("u",)
    split_sha256 = "split"
    cache_receipt_sha256 = "resources"

    def initial_state(self, anchor, destination, *, allow_fresh_fit):
        assert anchor.name == "random-6101"
        assert allow_fresh_fit is False
        state = initial_state(6101)
        once(destination / "initial_state_receipt.json", {
            "checkpoint_sha256": state.checkpoint,
            "source": {"source_kind": "shared_numeric_L333_model_only"},
        })
        return state


def test_round_zero_reuses_pinned_anchor_and_empty_memory(tmp_path):
    state, snapshot, evidence = prep._anchor_boundary(6101, AnchorBackend(), tmp_path / "anchors", tmp_path, "a" * 64)
    record = prep._save_boundary(tmp_path, state, snapshot, tmp_path / "cache", evidence)
    assert record["descriptor"] == {"round": 0, "state_sha256": digest(asdict(state)),
                                     "memory_sha256": digest(snapshot)}
    assert record["checkpoint_sha256"] == "a" * 64
    assert snapshot == prep.MemoryManager(6101).snapshot()
    assert snapshot["rounds"] == {}
    assert read(tmp_path / "boundaries/seed_6101/state.json") == asdict(state)
    with pytest.raises(PermissionError, match="approved shared L333"):
        prep._anchor_boundary(6101, AnchorBackend(), tmp_path / "anchors", tmp_path, "b" * 64)


def test_failure_keeps_exact_evidence_but_completed_reply_requires_recovery(tmp_path, monkeypatch):
    seed, start, identity = 157, 1, "a" * 32
    contract = SimpleNamespace(digest="contract", seed_ids=(seed,), profile={"wire_representation": "wire"},
                               transport=SimpleNamespace(transport_attempts=2))
    approval = {"contract_sha256": contract.digest}
    scope = prep.seed_runner._scope(contract, approval, seed, False)
    request = {"binding": {"seed": seed, "round": start, "a7_formal": scope}, "logical_index": 0,
               "request_id": "request", "payload_sha256": "payload"}
    round_dir = tmp_path / "seed_157/round_01"
    turn = round_dir / "turn_00"
    started = {"attempt_id": identity}
    terminal = {"status": "stopped", "error": "timeout", "bridge_result_sha256": "result",
                "bridge_job_sha256": "job"}
    job = {"job_sha256": "job"}
    result = {"result_sha256": "result", "raw_response": None, "outcome": {
        "status": "failed", "failure": {"code": "timeout", "ambiguous": True},
    }}
    once(turn / "request.json", request)
    once(turn / "attempt_00.started.json", started)
    once(turn / "attempt_00.terminal.json", terminal)
    once(tmp_path / "bridge/outbox" / f"{identity}.json", job)
    result_path = tmp_path / "bridge/results" / f"{identity}.json"
    once(result_path, result)
    monkeypatch.setattr(prep, "verify_job", lambda *_args, **_kwargs: (request, contract.transport, started))
    monkeypatch.setattr(prep, "validate_result", lambda *_args: None)
    assert prep._retired_attempts(tmp_path, round_dir, contract, approval, start) == {identity: {
        "request_id": "request", "payload_sha256": "payload", "bridge_result_sha256": "result",
        "terminal_sha256": digest(terminal), "code": "timeout", "execution_status_unknown": True,
    }}
    result["outcome"] = {"status": "completed"}
    result_path.unlink()
    once(result_path, result)
    with pytest.raises(PermissionError, match="recover the existing V11 response"):
        prep._retired_attempts(tmp_path, round_dir, contract, approval, start)


def test_scientific_draft_uses_all_seeds_and_only_scientific_fields():
    profile = {"arm": "A7", "memory": "M2", "device": "cuda", "initial_labels": 333,
               "target_labels": 1485, "initial_pool": 2997, "validation": 416, "test": 417, "batch_size": 32,
               "requires_initial_qualification": False, "confirmation_seeds": [157],
               "resume_boundaries": {"157": {"round": 1}}, "store": False, "transport": {},
               "request_timeouts": {}, "wire_representation": {"pinned": True}}
    scientific = prep._scientific_profile(SimpleNamespace(profile=profile))
    assert scientific["confirmation_seeds"] == [157, 887, 2357, 6101, 12203]
    assert scientific["wire_representation"] == profile["wire_representation"]
    assert not {"transport", "store", "request_timeouts", "resume_boundaries"} & scientific.keys()
    with pytest.raises(PermissionError):
        prep._scientific_profile(SimpleNamespace(profile={**profile, "batch_size": 64}))


def test_fitted_archive_uses_fit_audit_and_separately_verified_resources(tmp_path):
    seed, start = 157, 1
    backend = AnchorBackend()
    acquired = [f"u{i}" for i in range(32)]
    backend.u0_ids = tuple(acquired + ["still-u"])
    state = prep.seed_runner.FrozenState(seed, start, "d" * 64, {"validation_test_records": 0}, [], [
        {"seed": seed, "round": 0, "method": "pure_llm_v3",
         "records": [{"candidate_id": item, "source": "LLM"} for item in acquired]},
    ], {"expected_observed": ["l", *acquired], "expected_candidates": ["still-u"],
        "split_sha256": "split", "fit_audit_sha256": "a" * 64})
    descriptor = {"round": start, "state_sha256": digest(asdict(state)), "memory_sha256": digest({})}
    contract = SimpleNamespace(digest="contract", profile={"resume_boundaries": {str(seed): descriptor}},
                               predecessor={"checkpoint_sha256": state.checkpoint},
                               batch_size=32, rounds=36, seed_ids=(seed,))
    approval = {"contract_sha256": contract.digest}
    directory = tmp_path / f"seed_{seed}"
    once(directory / "boundary.json", {"formal_run": prep.seed_runner._scope(contract, approval, seed, False),
                                        **descriptor})
    once(directory / "round_01/source.json", asdict(state))
    once(directory / "round_01/boundary_memory.json", {})
    _, _, evidence = prep._archive_boundary(tmp_path, seed, contract, approval, backend)
    assert evidence["fit_audit_sha256"] == "a" * 64
    assert evidence["public_resource_receipt_sha256"] == "resources"
    once(directory / "round_01/batch_freeze.json", {})
    with pytest.raises(PermissionError, match="finish already frozen"):
        prep._archive_boundary(tmp_path, seed, contract, approval, backend)
