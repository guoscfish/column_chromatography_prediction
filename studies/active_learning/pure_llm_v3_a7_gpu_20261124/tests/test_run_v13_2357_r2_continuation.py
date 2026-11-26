from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_STREAM_IPV4, TransportConfig,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_2357_r2_continuation as follow_on
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.run_prepared_five_seed import PreparedCohortContract


@dataclass(frozen=True)
class State:
    seed: int
    round: int
    checkpoint: str

    @property
    def provenance(self):
        return {"fit_audit_sha256": "a" * 64}


def test_only_verified_l429_releases_original_33_round_budget(tmp_path, monkeypatch):
    root, parent_release, extension_release = (tmp_path / name for name in ("new", "parent", "extension"))
    parent_run = tmp_path / "runs" / "parent"
    extension_run = parent_run.with_name("extension")
    run = parent_run.with_name(follow_on.RUN_NAME)
    state = State(2357, 2, "b" * 64)
    memory = {"rounds": {"0": {}, "1": {}}}
    completed = State(2357, 3, "c" * 64)
    completed_memory = {"rounds": {**memory["rounds"], "2": {}}}
    checkpoint = {"round": 2, "state_sha256": digest(asdict(state)), "memory_sha256": digest(memory),
                  "user_message_sha256": "d" * 64}

    old_transport = TransportConfig("token4research", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh",
                                    "https://token4research.cn/v1", 3600, 32768, transport_attempts=2,
                                    wire_mode=RESPONSES_STREAM_IPV4)
    short = replace(old_transport, logical_replies=1, transport_attempts=1)
    root.mkdir()
    (root / "science.txt").write_text("recorded science")
    sources = {"science.txt": file_digest(root / "science.txt")}
    for module in ("run_v13_2357_r2_continuation", follow_on.DISPATCHER, "run_v13_lane_continuation",
                   "run_v13_lane_extension", "dispatch_v13_lane_continuation", "dispatch_v13_lane_extension"):
        path = root / follow_on.STUDY / "scripts" / f"{module}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic signed module {module}")

    identity = {"served_model": "gpt-6.2-astra", "reasoning": "xhigh"}
    original_budget = {"first_logical_requests": 34, "maximum_logical_requests": 102, "maximum_physical_attempts": 204}
    expected = {"first_logical_requests": 33, "maximum_logical_requests": 99, "maximum_physical_attempts": 198}
    one_budget = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    parent = PreparedCohortContract(
        {"initial_labels": 333, "target_labels": 1485, "batch_size": 32, "confirmation_seeds": [2357]},
        old_transport, sources, {}, run_directory=str(parent_run), api_execution="mac_ssh_bridge",
        predecessor={"model_identity": identity, "execution_budget": original_budget},
    )
    one_shot = replace(parent, profile={
        **parent.profile, "target_labels": 429, "sse_inactivity_seconds": 900, "transport": asdict(short),
        "request_timeouts": asdict(follow_on.extension.TIMEOUTS),
        "resume_boundaries": {"2357": checkpoint},
    }, transport=short, run_directory=str(extension_run), predecessor={
        "model_identity": identity, "execution_budget": one_budget, "parent_seed_pause_sha256": "e" * 64,
        "bridge_wait_seconds": 7800,
    })
    approval = {"scope": "pure_llm_v3_a7_formal", "contract_sha256": one_shot.digest,
                "authorize_paid_api": True, "authorize_label_reveal": True, "authorize_cuda_training": True,
                "authorized_by": "synthetic", "authorization_reference": "offline test"}
    backend = SimpleNamespace(cache_dir=tmp_path / "public_cache", cache_receipt_sha256="f" * 64)
    boundary = (state, memory, 2)
    scope = follow_on.seed_runner._scope(one_shot, approval, 2357, False)
    seed = extension_run / "seed_2357"
    one_shot_round = seed / "round_02"
    transition = seed / "round_03/transition.json"
    once(extension_run / "formal_contract.json", one_shot.record())
    once(extension_run / "authorization.json", approval)
    once(extension_run / "provider_model_lock.json", identity)
    once(seed / "boundary.json", {"formal_run": scope, **checkpoint})
    once(one_shot_round / "source.json", asdict(state))
    once(one_shot_round / "boundary_memory.json", memory)
    once(transition, {"synthetic": "verified R2 CUDA fit"})
    marker_path = extension_run / "extension_completion.json"
    once(marker_path, {
        "contract_sha256": one_shot.digest, "seed": 2357, "round": 3, "active_labels": 429,
        "checkpoint_sha256": completed.checkpoint, "state_sha256": digest(asdict(completed)),
        "memory_sha256": digest(completed_memory), "transition_sha256": file_digest(transition),
        "original_future_budget": expected,
    })

    attempt = {"extension_attempt_id": "1" * 32, "extension_job_sha256": "2" * 64, "extension_result_sha256": "3" * 64}
    monkeypatch.setattr(follow_on.shared, "_retirement", lambda *_: "e" * 64)
    monkeypatch.setattr(follow_on.extension, "construct", lambda *_: (one_shot, boundary, backend, {"archived": True}))
    monkeypatch.setattr(follow_on.extension, "_original", lambda *_: (parent, {"synthetic": "approved"}, boundary, backend))
    monkeypatch.setattr(follow_on.extension, "_stage_request", lambda *_: {"archived": "exact"})
    monkeypatch.setattr(follow_on.shared, "_completed_attempt", lambda *_: attempt)
    monkeypatch.setattr(follow_on.seed_runner, "_successor", lambda *_: (completed, completed_memory))

    def construct():
        return follow_on.construct(
            root, parent_release, parent_run, extension_release, extension_run, run,
            expected_extension_contract=one_shot.digest, expected_extension_approval=digest(approval),
        )

    contract, restored, restored_backend = construct()
    assert restored == (completed, completed_memory, 3) and restored_backend == backend
    assert contract.record()["budget"] == expected
    assert contract.transport == old_transport
    assert contract.profile["sse_inactivity_seconds"] == 900
    assert contract.profile["request_timeouts"] == asdict(follow_on.extension.TIMEOUTS)
    assert contract.predecessor["bridge_wait_seconds"] == 7800
    assert contract.predecessor["checkpoint_sha256"] == completed.checkpoint
    assert contract.predecessor["extension_result_sha256"] == attempt["extension_result_sha256"]
    marker_path.unlink()
    with pytest.raises(FileNotFoundError, match="extension_completion"):
        construct()
