from dataclasses import replace
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import SCHEMA_VERSION
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_STREAM_IPV4, UNKNOWN_REPLAY_DECISION, TransportConfig, TransportFailure,
    _link_replay_attempt, authorize_ambiguous_replay, make_request, terminal_record,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import make_job, make_result
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_2357_r2_sse_extension as recovery


def _archive(parent, *, replay_really_posted=False):
    config = TransportConfig("token4research", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh",
                             "https://token4research.cn/v1", 3600, 32768, transport_attempts=2,
                             wire_mode=RESPONSES_STREAM_IPV4)
    packet = {"seed": 2357, "round": 2, "layout": "B", "compression": "lossless",
              "memory": {"mode": "M2"}, "selection_count": 32}
    messages = [{"role": "system", "content": "Synthetic A7 system message"},
                {"role": "user", "content": dumps(packet)}]
    approval = {"synthetic approval": True}
    scope = {"contract_sha256": "a" * 64, "authorization_sha256": digest(approval), "seed": 2357, "mock": False}
    binding = {
        "a7_formal": scope, "seed": 2357, "round": 2, "arm": "A7", "prompt_version": "v3",
        "schema_version": SCHEMA_VERSION, "packet_sha256": digest(packet),
        "prompt_sha256": digest(messages[0]["content"]), "base_messages_sha256": digest(messages),
        **{key: digest(key) for key in ("checkpoint_sha256", "state_sha256", "mapping_sha256", "split_sha256")},
    }
    request = make_request(binding, 0, messages, config)
    active = parent / "seed_2357/round_02"
    turn = active / "turn_00"
    once(turn / "request.json", request)
    contract = SimpleNamespace(digest=scope["contract_sha256"], transport=config, profile={"wire_representation": None})
    for index, attempt_id in enumerate(recovery.ATTEMPTS):
        original = index == 0
        dispatched = original or replay_really_posted
        error = "timeout" if original else "connect_error"
        started = {"request_id": request["request_id"], "attempt": index, "attempt_id": attempt_id,
                   "started_unix": 1, "transport": "mac_bridge"}
        job = make_job(request, config, started)
        failure = TransportFailure(error, ambiguous=original, retryable=not original)
        result = make_result(job, failure=failure, credential_id="key_01", provider_dispatched=dispatched)
        terminal = {**started, "status": "stopped", "error": error, "ambiguous": original,
                    "provider_dispatched": dispatched, "circuit_open": False, "rate_limit_count": 0, "not_before_unix": 1,
                    "bridge_job_sha256": job["job_sha256"], "bridge_result_sha256": result["result_sha256"]}
        once(turn / f"attempt_{index:02d}.started.json", started)
        once(turn / f"attempt_{index:02d}.terminal.json", terminal)
        for folder, record in (("outbox", job), ("results", result)):
            path = parent / "bridge" / folder / f"{attempt_id}.json"
            once(path, record)
            path.chmod(0o600)
        if original:
            permit = {"decision": UNKNOWN_REPLAY_DECISION, "request_id": request["request_id"],
                      "contract_sha256": contract.digest, "attempt_id": attempt_id,
                      "bridge_result_sha256": result["result_sha256"], "original_terminal_sha256": digest(terminal),
                      "next_attempt": 1, "accept_duplicate_billing_risk": True,
                      "authorized_by": "synthetic", "authorization_reference": "test"}
            permit_sha = digest(permit)
            authorize_ambiguous_replay(active, request, authorization=permit, config=config,
                                       expected_authorization_sha256=permit_sha)
        else:
            replay = terminal_record(turn, 0, replay_authorization_sha256=permit_sha)
            _link_replay_attempt(turn, started, replay, allow_create=True)
    return turn, request, contract, approval


def test_both_sealed_failures_preserve_exact_science_and_replay_proof(tmp_path):
    turn, request, contract, approval = _archive(tmp_path)
    prior, permit, link = recovery._prior_attempts(tmp_path, turn, request, contract, approval)
    assert all(len(value) == 64 for value in (permit, link))
    assert [(x["attempt_id"], x["provider_dispatched"], x["ambiguous"], x["error"]) for x in prior] == [
        (recovery.ATTEMPTS[0], True, True, "timeout"), (recovery.ATTEMPTS[1], False, False, "connect_error"),
    ]
    binding = {**request["binding"], "a7_formal": {**request["binding"]["a7_formal"], "contract_sha256": "e" * 64}}
    config = replace(contract.transport, logical_replies=1, transport_attempts=1)
    candidate = make_request(binding, 0, request["payload"]["input"], config)
    assert recovery.previous._same_science(candidate, request)
    assert recovery.SSE_INACTIVITY_SECONDS == 900
    assert (recovery.TIMEOUTS.read_idle_seconds, recovery.TIMEOUTS.total_seconds,
            recovery.TIMEOUTS.bridge_wait_seconds) == (1800, 3600, 7800)


def test_replay_that_posted_and_original_scientific_advance_are_disallowed(tmp_path):
    turn, request, contract, approval = _archive(tmp_path, replay_really_posted=True)
    with pytest.raises(PermissionError, match="preflight-only replay"):
        recovery._prior_attempts(tmp_path, turn, request, contract, approval)
    with pytest.raises(PermissionError, match="pause the original"):
        recovery._retirement(tmp_path)
    pause = turn.parent.parent / "pause.request"
    once(pause, {"seed": 2357})
    assert recovery._retirement(tmp_path)[1] == file_digest(pause)
    once(turn.parent / "batch_freeze.json", {"already selected": True})
    with pytest.raises(PermissionError, match="scientific progress"):
        recovery._retirement(tmp_path)


def test_only_provider_and_dispatcher_may_change(tmp_path):
    names = recovery.TRANSPORT_OVERLAYS | {"prompt.md"}
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original")
    sources = {name: file_digest(tmp_path / name) for name in names}
    contract = SimpleNamespace(source_sha256=sources)
    for name in recovery.TRANSPORT_OVERLAYS:
        (tmp_path / name).write_text("new SSE idle")
    for module in ("run_v13_2357_r2_sse_extension", recovery.DISPATCHER, "dispatch_prompt_direct"):
        (tmp_path / recovery.STUDY / "scripts").mkdir(parents=True, exist_ok=True)
        (tmp_path / recovery.STUDY / "scripts" / f"{module}.py").write_text("new signed entrypoint")
    updated = recovery._source_hashes(tmp_path, contract)
    assert {name for name in names if updated[name] != sources[name]} == recovery.TRANSPORT_OVERLAYS
    (tmp_path / "prompt.md").write_text("unauthorized change")
    with pytest.raises(PermissionError, match="only the new provider"):
        recovery._source_hashes(tmp_path, contract)
