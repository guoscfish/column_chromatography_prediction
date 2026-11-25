from dataclasses import asdict, replace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    ProviderReply, TransportConfig, TransportFailure, TransportStop, make_request, reconcile, run_request,
)
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import read, once
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    PROFILE, PROFILE_V3, ROOT, SSH_API, RequestTimeouts, load_contract, request_timeouts, verify_authorization,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import http_failure, verify_reported_model
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.scheduler import _provider_wiring


def test_formal_authorization_is_bound_to_exact_run_directory(tmp_path):
    first = load_contract(ROOT, ROOT / PROFILE, gpu_slots=(0, 1), run_directory=tmp_path / "first")
    second = replace(first, run_directory=str(tmp_path / "second"))
    approval = {"scope": "pure_llm_v3_a7_formal", "contract_sha256": first.digest,
                "authorize_paid_api": True, "authorize_label_reveal": True, "authorize_cuda_training": True,
                "authorized_by": "mock authorization", "authorization_reference": "local test only"}
    assert verify_authorization(first, approval) == approval
    with pytest.raises(PermissionError):
        verify_authorization(second, approval)
    assert first.record()["budget"] == {"first_logical_requests": 180, "maximum_logical_requests": 540,
                                         "maximum_physical_attempts": 1080}


def test_new_timeouts_bind_to_successor_authority_and_bridge_budget(tmp_path):
    original = replace(load_contract(ROOT, ROOT / PROFILE), api_execution=SSH_API)
    policy = RequestTimeouts()
    profile = {**original.profile, "request_timeouts": asdict(policy)}
    candidate = replace(original, profile=profile)
    assert candidate.digest != original.digest
    assert request_timeouts(original.profile) is None and request_timeouts(candidate.profile) == policy
    assert policy.bridge_wait_seconds == 4200
    for label, record, wait in (("old", original, 1800), ("candidate", candidate, 4200)):
        _, wiring = _provider_wiring(tmp_path / label, record, {}, None, 157)
        assert wiring["managed_attempt"].wait_seconds == wait
    v3 = read(ROOT / PROFILE_V3)
    planned_file = tmp_path / "planned_successor.json"
    once(planned_file, {**v3, "transport": {**v3["transport"], "timeout_seconds": policy.total_seconds},
                        "request_timeouts": asdict(policy)})
    with pytest.raises(PermissionError, match="separately reviewed successor contract after v3 reconciliation"):
        load_contract(ROOT, planned_file, api_execution=SSH_API)
    mismatched = tmp_path / "mismatched_successor.json"
    once(mismatched, {**v3, "request_timeouts": asdict(policy)})
    with pytest.raises(ValueError, match="must match the transport audit configuration"):
        load_contract(ROOT, mismatched, api_execution=SSH_API)
    with pytest.raises(ValueError, match="must bind all of its durations"):
        request_timeouts({"request_timeouts": {"total_seconds": 1800}})
    with pytest.raises(ValueError, match="outside their supported bounds"):
        RequestTimeouts(read_idle_seconds=1900)


def test_provider_classification_does_not_assume_429_is_per_key():
    assert http_failure(429, {"retry-after": "15"}, {})[1] == "shared"
    failure, scope = http_failure(429, {"retry-after": "15", "x-ratelimit-scope": "api_key"}, {})
    assert (failure.code, failure.retry_after, scope) == ("http_429", 15, "credential")
    assert http_failure(413, {}, {})[0].code == "provider_context_rejected"
    assert http_failure(400, {}, {"error": {"param": "reasoning.effort"}})[0].code == "provider_reasoning_rejected"


def test_reported_model_version_and_effort_are_exact():
    profile = load_contract(ROOT, ROOT / PROFILE)
    reply = ProviderReply("{}", "mock_response", "gpt-6.2-sol", {})
    declared = {"model_version": "v202611", "reported_reasoning_effort": "ultra"}
    assert verify_reported_model(profile.transport, reply, declared)["reported_model_version"] == "v202611"
    with pytest.raises(TransportFailure, match="served_reasoning_effort_mismatch"):
        verify_reported_model(profile.transport, reply, {**declared, "reported_reasoning_effort": "xhigh"})
    with pytest.raises(TransportFailure, match="served_model_mismatch"):
        verify_reported_model(profile.transport, replace(reply, served_model="gpt-6.2-sol-other"), declared)


def test_manual_recovery_of_confirmed_403_keeps_one_remaining_attempt(tmp_path):
    config = TransportConfig(provider="mock", requested_model="mock", served_model="mock",
                             reasoning_effort="ultra", base_url="https://example.com", timeout_seconds=10,
                             max_output_tokens=10, transport_attempts=2)
    request = make_request({"seed": 157, "round": 0}, 0, [], config)
    calls = []
    def call(_request, _config):
        calls.append(True)
        if len(calls) == 1:
            raise TransportFailure("http_403")
        return ProviderReply("{}", "mock-recovered", "mock", {})
    for _ in range(2):
        with pytest.raises(TransportStop, match="http_403"):
            run_request(tmp_path, request, config, call)
    started = read(tmp_path / "turn_00/attempt_00.started.json")
    reconcile(tmp_path, request, 0, config=config, authorization={
        "request_id": request["request_id"], "attempt_id": started["attempt_id"],
        "decision": "confirmed_not_processed", "authorized_by": "fixture",
        "evidence": "mock provider rejected before execution"})
    assert run_request(tmp_path, request, config, call)["attempt"] == 1
    assert len(calls) == 2
