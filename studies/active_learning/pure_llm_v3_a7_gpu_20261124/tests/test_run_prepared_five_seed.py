"""Offline checks for exact five-seed transport approvals and persisted boundaries."""

from dataclasses import asdict

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, file_digest, once
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import RequestTimeouts
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_prepared_five_seed as runner


def route():
    timeout = RequestTimeouts()
    config = runner.TransportConfig("test-provider", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh", "https://example.org/v1",
                                    timeout.total_seconds, 32768, transport_attempts=2,
                                    wire_mode=runner.RESPONSES_BACKGROUND)
    return {"schema": "a7_route_policy_v1", "transport": asdict(config), "request_timeouts": asdict(timeout),
            "qualification": None, "dispatcher_module": "dispatch_test"}


def qualification_receipt(policy):
    transport = policy["transport"]
    route_identity = {key: transport[key] for key in (
        "provider", "base_url", "wire_mode", "requested_model", "reasoning_effort",
    )}
    return {**route_identity, "submission": {"response_id": "resp_same", "background": True},
            "retrieval": {"response_id": "resp_same", "status": "completed", "model": "gpt-6.2-astra",
                          "reasoning_effort": "xhigh"}}


def test_short_background_completion_is_enough_with_exact_same_id_retrieval(tmp_path):
    policy = route()
    receipt = tmp_path / "retrieved.json"
    recorded = qualification_receipt(policy)
    once(receipt, recorded)
    lock = {"provider": "test-provider", "requested_model": "gpt-6.2-astra", "served_model": "gpt-6.2-astra",
            "requested_reasoning_effort": "xhigh", "reported_reasoning_effort": "xhigh",
            "reported_model_version": None}
    qualified = {"receipt_path": str(receipt), "receipt_sha256": file_digest(receipt),
                 "long_context_completed": False, "same_id_retrieval_completed": True, "model_identity": lock}
    assert runner._provider_policy(policy)[3] is None
    assert runner._provider_policy({**policy, "qualification": qualified})[3] == qualified
    unqualified = {**qualified, "same_id_retrieval_completed": False}
    with pytest.raises(PermissionError, match="retrieval evidence"):
        runner._provider_policy({**policy, "qualification": unqualified})
    with pytest.raises(PermissionError, match="receipt is missing or changed"):
        runner._provider_policy({**policy, "qualification": {**qualified, "receipt_sha256": "a" * 64}})
    with pytest.raises(PermissionError, match="finite A7 budgets"):
        runner._provider_policy({**policy, "transport": {**policy["transport"], "transport_attempts": 3}})
    for index, bad in enumerate((
        {**recorded, "submission": {"response_id": "resp_same", "background": False}},
        {**recorded, "retrieval": {**recorded["retrieval"], "response_id": "resp_other"}},
        {**recorded, "retrieval": {**recorded["retrieval"], "status": "404"}},
        {**recorded, "retrieval": {**recorded["retrieval"], "model": "gpt-6.2-sol"}},
        {**recorded, "retrieval": {**recorded["retrieval"], "reasoning_effort": "high"}},
        {**recorded, "base_url": "https://example.com/v1"},
    )):
        wrong_receipt = tmp_path / f"wrong_{index}.json"
        once(wrong_receipt, bad)
        with pytest.raises(PermissionError):
            runner._provider_policy({**policy, "qualification": {
                **qualified, "receipt_path": str(wrong_receipt), "receipt_sha256": file_digest(wrong_receipt),
            }})


def test_astra_namespace_is_only_allowed_on_mantle(tmp_path):
    policy = route()
    namespaced = {**policy["transport"], "requested_model": "openai.gpt-6.2-astra"}
    with pytest.raises(PermissionError, match="model, reasoning"):
        runner._provider_policy({**policy, "transport": namespaced})
    namespaced["provider"] = "aws-bedrock-mantle"
    policy["transport"] = namespaced
    assert runner._provider_policy(policy)[0].served_model == "gpt-6.2-astra"
    for key in ("requested_model", "served_model"):
        with pytest.raises(PermissionError, match="model, reasoning"):
            runner._provider_policy({**policy, "transport": {**namespaced, key: "openai.gpt-6.2-astra-other"}})
    recorded = qualification_receipt(policy)
    receipt = tmp_path / "mantle.json"
    once(receipt, recorded)
    qualified = {"receipt_path": str(receipt), "receipt_sha256": file_digest(receipt),
                 "long_context_completed": False, "same_id_retrieval_completed": True,
                 "model_identity": {"provider": namespaced["provider"], "requested_model": namespaced["requested_model"],
                                    "served_model": namespaced["served_model"], "requested_reasoning_effort": "xhigh",
                                    "reported_reasoning_effort": "xhigh", "reported_model_version": None}}
    assert runner._provider_policy({**policy, "qualification": qualified})[0].requested_model == "openai.gpt-6.2-astra"
    with pytest.raises(PermissionError, match="reported a different"):
        runner._provider_policy({**policy, "qualification": qualified, "transport": {
            **namespaced, "served_model": "openai.gpt-6.2-astra",
        }})


def _fixture(tmp, monkeypatch):
    root, runs, preparation = tmp / "release", tmp / "runs", tmp / "preparation"
    scientific = {"arm": "A7", "memory": "M2", "device": "cuda", "initial_labels": 333, "target_labels": 1485,
                  "initial_pool": 2997, "validation": 416, "test": 417, "batch_size": 32,
                  "requires_initial_qualification": False, "confirmation_seeds": list(runner.SEEDS)}
    prompt = root / runner.seed_runner.PROMPT_PATH
    once(prompt, {"prompt": "fixed"})
    source = {runner.seed_runner.PROMPT_PATH.as_posix(): file_digest(prompt)}
    archived = {"profile": {**scientific, "transport": {**route()["transport"], "wire_mode": "responses_stream"}},
                "asset_sha256": {"asset": "meta"}, "source_sha256": source}
    archived_path = runs / f"{runner.prepared.V11_RUN_PREFIX}-157" / "formal_contract.json"
    once(archived_path, archived)
    archival_ids = {**runner.prepared.V11_CONTRACTS, 157: digest(archived)}
    monkeypatch.setattr(runner.prepared, "V11_CONTRACTS", archival_ids)
    for name in ("prepare_five_seed_boundaries", "run_prepared_five_seed", "dispatch_test"):
        once(root / runner.STUDY / "scripts" / f"{name}.py", {})
    for name in ("background_transport", "background_dispatcher"):
        once(root / runner.STUDY / "a7_gpu" / f"{name}.py", {})
    rows = {}
    for seed in runner.SEEDS:
        start = runner.prepared.V11_START.get(seed, 0)
        cache_root = runs / runner.V4_PILOT_RUN_NAME if start else preparation
        cache = cache_root / "public_graphs" / f"seed_{seed}"
        resource_path = cache / "resource_receipt.json"
        once(resource_path, {"seed": seed})
        resource = file_digest(resource_path)
        checkpoint = "a" * 64 if start else runner.L333_ANCHOR_SHA256[str(seed)]
        active_labels = 333 + 32 * start
        raw_state = runner.seed_runner.FrozenState(
            seed, start, checkpoint, {}, [], [], {"expected_observed": list(range(active_labels)),
                                                    "resource_receipt_sha256": resource},
        )
        memory = runner.MemoryManager(seed).snapshot()
        memory["rounds"] = {str(index): {"round": index, "checkpoint_sha256": "a" * 64, "plan_sha256": "b" * 64,
                              "batch_strategy": "", "feedback_interpretation": "", "batch_review": "",
                              "selected": [], "hypothesis_updates": []} for index in range(start)}
        memory["snapshot_sha256"] = digest({key: val for key, val in memory.items() if key != "snapshot_sha256"})
        evidence = {"source": "shared_cuda_l333_anchor"}
        if start:
            evidence = {"source": "sealed_v11", "run_directory": str(runs / f"{runner.prepared.V11_RUN_PREFIX}-{seed}"),
                        "contract_sha256": archival_ids[seed], "public_resource_receipt_sha256": resource}
        rows[str(seed)] = runner.prepared._save_boundary(preparation, raw_state, memory, cache, evidence)
    draft = {"schema": "a7_five_seed_boundary_draft_v1", "paid_requests_authorized": False, "gpu_slots": [0],
             "gpu_lease_root": str(runner.LEASE_ROOT), "initial_anchor_sha256_by_seed": runner.L333_ANCHOR_SHA256,
             "improved_prompt_sha256": runner.prepared.PROMPT_SHA256, "boundaries": rows,
             "scientific_profile": scientific, "asset_sha256": archived["asset_sha256"],
             "remaining_first_logical_requests": 173}
    draft_path, policy_path = preparation / "draft.json", preparation / "policy.json"
    once(draft_path, draft)
    once(policy_path, route())
    return root, draft_path, digest(draft), policy_path, runs / "a7-five-prepared"


def test_complete_candidate_contract_is_nonlaunching_until_specific_route_is_qualified(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    contract, draft = runner.construct(*args)
    assert contract.seed_ids == (157, 887, 2357, 6101, 12203)
    assert contract.target_labels == 1485
    assert contract.record()["budget"] == {"first_logical_requests": 173, "maximum_logical_requests": 519,
                                          "maximum_physical_attempts": 1038}
    assert contract.profile["store"] is False
    assert all((runner.STUDY / "a7_gpu" / f"{name}.py").as_posix() in contract.source_sha256
               for name in ("background_transport", "background_dispatcher"))
    assert not args[-1].exists()
    with pytest.raises(PermissionError, match="qualification receipt"):
        runner.launch(args[0], args[-1], contract, draft, args[1], {}, resume=False)
    assert not args[-1].exists()


def test_draft_rejects_changed_boundary_or_science(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    state = args[1].parent / "boundaries/seed_6101/state.json"
    state.write_text("{}")
    with pytest.raises((KeyError, PermissionError)):
        runner.construct(*args)
    (args[0] / runner.seed_runner.PROMPT_PATH).write_text("changed science")
    with pytest.raises(PermissionError, match="changes approved science"):
        runner.construct(*args)
