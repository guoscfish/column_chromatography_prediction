"""Offline bridge validation and transport state-machine tests."""

from copy import deepcopy
from dataclasses import replace
import os
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import SCHEMA_VERSION
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_BACKGROUND, ProviderReply, TransportConfig, TransportFailure, TransportStop, make_request, round_lock,
    run_request, terminal_record,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import (
    BridgeCourier, BridgePending, make_job, make_result, read_bridge_json, validate_result, verify_job, verify_result,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import verify_reported_model
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import seed_runner


CONTRACT = "a" * 64
AUTHORIZATION = "b" * 64


@pytest.fixture
def setup_bridge(tmp_path):
    config = TransportConfig("test-provider", "test-model", "test-model", "ultra", "https://example.invalid/v1",
                             600, 32768, transport_attempts=2)
    packet = {"seed": 157, "round": 0, "layout": "B", "compression": "lossless",
              "memory": {"mode": "M2"}, "selection_count": 32}
    messages = [{"role": "system", "content": "Synthetic immutable A7 system message."},
                {"role": "user", "content": dumps(packet)}]
    binding = {
        "seed": 157, "round": 0, "arm": "A7", "prompt_version": "v3", "schema_version": SCHEMA_VERSION,
        **{field: digest(field) for field in ("checkpoint_sha256", "state_sha256", "mapping_sha256", "split_sha256")},
        "packet_sha256": digest(packet), "prompt_sha256": digest(messages[0]["content"]),
        "base_messages_sha256": digest(messages),
        "a7_formal": {"contract_sha256": CONTRACT, "authorization_sha256": AUTHORIZATION, "seed": 157, "mock": True},
    }
    request = make_request(binding, 0, messages, config)
    started = {"request_id": request["request_id"], "attempt": 0, "attempt_id": "c" * 32,
               "started_unix": 100.0, "transport": "mac_bridge"}
    bridge = BridgeCourier(tmp_path / "bridge", contract_sha256=CONTRACT, authorization_sha256=AUTHORIZATION,
                           wait_seconds=0)
    return config, request, started, bridge


def _payload(config, job, *, version="test-version-1"):
    reply = ProviderReply('{"unvalidated_experiment_plan": true}', "response-1", config.served_model,
                          {"input_tokens": 9, "output_tokens": 3}, provider_request_id="provider-req-1")
    raw = {"id": reply.response_id, "model": reply.served_model, "status": reply.status,
           "output": [{"type": "message", "content": [{"type": "output_text", "text": reply.text}]}]}
    identity = verify_reported_model(config, reply, {"model_version": version, "reported_reasoning_effort": "ultra"})
    return make_result(job, reply=reply, model_identity=identity, raw_response=raw,
                       credential_id="mock-slot-1", provider_dispatched=True, elapsed_seconds=2.1)


def _forbidden(*_args, **_kwargs):
    pytest.fail("GPU or direct transport called a provider during bridge test")


@pytest.fixture
def audited_live_bridge(setup_bridge):
    config, request, _, bridge = setup_bridge
    scope = {**request["binding"]["a7_formal"], "mock": False}
    messages = [*request["payload"]["input"], {"role": "user", "content": "Use the exact original repair context."}]
    request = make_request({**request["binding"], "a7_formal": scope}, 1, messages, config)
    run = bridge.directory.parent

    def permit(archived, terminal, candidate_config):
        return seed_runner._future_replay_authorization(run, scope, {"authorized_by": "offline fixture"},
                                                        archived, terminal, candidate_config)

    return run, config, request, bridge, permit


def test_future_audited_replay_preserves_exact_bridge_request_and_archives(audited_live_bridge):
    run, config, request, bridge, permit = audited_live_bridge
    directory = run / "seed_157" / "round_01"
    log = directory / "turn_01"
    calls = []
    original_files = {}

    def fake_mac(managed_request, managed_config, started):
        assert managed_request == request and managed_config == config
        calls.append(started["attempt_id"])
        job = make_job(managed_request, managed_config, started)
        name = f"{job['attempt_id']}.json"
        if started["attempt"] == 0:
            result = make_result(job, failure=TransportFailure("timeout", ambiguous=True),
                                 provider_dispatched=True, credential_id="mock-slot-1")
            original_files.update({bridge.outbox / name: job, bridge.results / name: result})
        else:
            assert started["attempt"] == 1 and calls[0] != calls[1]
            for path, artifact in original_files.items():
                assert read(path) == artifact
            grant = read(log / "attempt_00.replay_authorization.json")
            assert grant["bridge_result_sha256"] == read(bridge.results / f"{calls[0]}.json")["result_sha256"]
            assert grant["original_terminal_sha256"] == digest(terminal_record(log, 0))
            assert grant["accept_duplicate_billing_risk"] is True
            assert read(log / "attempt_01.replay_link.json")["replay_authorization_sha256"] == digest(grant)
            result = _payload(managed_config, job)
        once(bridge.outbox / name, job)
        once(bridge.results / name, result)
        return bridge(managed_request, managed_config, started)

    options = {"managed_attempt": fake_mac, "replay_authorization_factory": permit,
               "sleep": lambda _: None, "now": lambda: 0}
    completed = run_request(directory, request, config, _forbidden, **options)
    assert completed["attempt"] == 1
    assert (terminal_record(log, 0)["status"], terminal_record(log, 0)["ambiguous"]) == ("stopped", True)
    assert read(log / "attempt_00.replay.terminal.json")["prior_execution_and_billing"] == "unresolved"
    assert read(log / "request.json") == request
    assert len(calls) == 2
    old_bytes = {path: path.read_bytes() for path in (*original_files, log / "attempt_00.terminal.json")}
    assert run_request(directory, request, config, _forbidden, **{**options, "managed_attempt": _forbidden}) == completed
    with pytest.raises(TransportStop, match="timeout"):
        run_request(directory, request, config, _forbidden, managed_attempt=_forbidden)
    assert all(path.read_bytes() == value for path, value in old_bytes.items())
    assert not (log / "attempt_02.started.json").exists()


@pytest.mark.parametrize(("error", "dispatched", "raw", "expected"), [
    ("response_id_mismatch", True, None, TransportStop),
    ("timeout", False, None, TransportStop),
    ("timeout", True, {"status": "completed", "id": "captured"}, PermissionError),
])
def test_future_replay_does_not_resend_an_ineligible_bridge_failure(audited_live_bridge, error, dispatched, raw, expected):
    run, config, request, bridge, permit = audited_live_bridge
    directory = run / "seed_157" / "round_01"

    def fake_mac(managed_request, managed_config, started):
        assert started["attempt"] == 0
        job = make_job(managed_request, managed_config, started)
        result = make_result(job, failure=TransportFailure(error, ambiguous=True),
                             provider_dispatched=dispatched, credential_id="mock-slot-1", raw_response=raw)
        name = f"{job['attempt_id']}.json"
        once(bridge.outbox / name, job)
        once(bridge.results / name, result)
        return bridge(managed_request, managed_config, started)

    with pytest.raises(expected):
        run_request(directory, request, config, _forbidden, managed_attempt=fake_mac,
                    replay_authorization_factory=permit, now=lambda: 0, sleep=_forbidden)
    log = directory / "turn_01"
    assert not (log / "attempt_00.replay_authorization.json").exists()
    assert not (log / "attempt_01.started.json").exists()


def test_future_replay_does_not_authorize_a_pending_bridge_response(audited_live_bridge):
    run, config, request, bridge, _ = audited_live_bridge
    directory = run / "seed_157" / "round_01"
    with pytest.raises(BridgePending):
        run_request(directory, request, config, _forbidden, managed_attempt=bridge,
                    replay_authorization_factory=_forbidden, now=lambda: 0, sleep=_forbidden)
    log = directory / "turn_01"
    assert (log / "attempt_00.started.json").is_file()
    assert not (log / "attempt_00.terminal.json").exists()
    assert not (log / "attempt_00.replay_authorization.json").exists()
    assert not (log / "attempt_01.started.json").exists()


def test_formal_seed_cannot_inject_its_own_replay_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(seed_runner, "verify_authorization", lambda _contract, approval: approval)
    contract = SimpleNamespace(seed_ids=(157,), initial_labels=333, profile={}, rounds=36, anchor_root=None)
    backend = SimpleNamespace(seed=157, l0_ids=range(333), is_mock=False)
    run = tmp_path / "no-injected-replay"
    with pytest.raises(PermissionError, match="internally verified archive"):
        seed_runner.run_seed(root=tmp_path, run_directory=run, contract=contract, authorization={},
                             seed=157, backend=backend, call=_forbidden,
                             transport_options={"managed_attempt": _forbidden, "replay_authorization_factory": _forbidden})
    assert not run.exists()


def test_job_verifies_authorization_and_detects_payload_or_attempt_drift(setup_bridge):
    config, request, started, _ = setup_bridge
    job = make_job(request, config, started)
    assert verify_job(job, authorization_sha256=AUTHORIZATION, contract_sha256=CONTRACT) == (request, config, started)
    with pytest.raises(ValueError, match="approved A7 contract"):
        verify_job(job, authorization_sha256="e" * 64)
    for key in ("payload", "binding"):
        forged = deepcopy(job)
        forged["request"][key]["unexpected"] = False
        forged["job_sha256"] = digest({k: v for k, v in forged.items() if k != "job_sha256"})
        with pytest.raises(ValueError, match="request identity"):
            verify_job(forged)
    with pytest.raises(ValueError, match="unregistered bridge attempt"):
        make_job(request, config, {**started, "attempt": config.transport_attempts})


def test_receipt_correlates_attempt_and_provider_and_rejects_secret_headers(setup_bridge):
    config, request, started, _ = setup_bridge
    job = make_job(request, config, started)
    result = _payload(config, job)
    assert validate_result(result, job) == result
    reply = verify_result(result, job)
    assert reply.text == '{"unvalidated_experiment_plan": true}'  # No scientific acceptance by the bridge.
    assert reply.transport_evidence["credential_id"] == "mock-slot-1"
    different = make_job(request, config, {**started, "attempt_id": "e" * 32})
    with pytest.raises(ValueError, match="exact provider attempt"):
        verify_result(result, different)
    raw_change = deepcopy(result)
    raw_change["raw_response"]["model"] = "unknown-model"
    raw_change["raw_response_sha256"] = digest(raw_change["raw_response"])
    raw_change["result_sha256"] = digest({k: v for k, v in raw_change.items() if k != "result_sha256"})
    with pytest.raises(ValueError, match="raw provider response identity"):
        verify_result(raw_change, job)
    text_change = deepcopy(result)
    text_change["outcome"]["reply"]["text"] += "synthetic tamper"
    text_change["result_sha256"] = digest({k: v for k, v in text_change.items() if k != "result_sha256"})
    with pytest.raises(ValueError, match="extracted provider text differs"):
        verify_result(text_change, job)
    raw_change["raw_response"]["debug"] = {"authorization": "fixture-secret-only"}
    raw_change["raw_response_sha256"] = digest(raw_change["raw_response"])
    raw_change["result_sha256"] = digest({k: v for k, v in raw_change.items() if k != "result_sha256"})
    with pytest.raises(ValueError, match="credential or HTTP header"):
        validate_result(raw_change, job)
    failure = TransportFailure("served_model_mismatch", retryable=True)
    with pytest.raises(ValueError, match="classified provider failure"):
        make_result(job, failure=failure, credential_id="mock-slot-1", provider_dispatched=True)


def test_pending_gpu_attempt_reuses_job_and_completed_mac_receipt(setup_bridge, tmp_path):
    config, request, _, bridge = setup_bridge
    round_dir = tmp_path / "round_00"
    turn = round_dir / "turn_00"
    for _ in range(2):
        with round_lock(round_dir), pytest.raises(BridgePending):
            run_request(round_dir, request, config, _forbidden, managed_attempt=bridge)
    files = list(bridge.outbox.glob("*.json"))
    assert len(files) == 1
    job = read_bridge_json(files[0])
    original = read(turn / "attempt_00.started.json")
    assert job["started"] == original and not (turn / "attempt_00.terminal.json").exists()
    with round_lock(round_dir), pytest.raises(TransportStop, match="requires reconciliation"):
        run_request(round_dir, request, config, _forbidden)
    result = _payload(config, job)
    once(bridge.results / files[0].name, result)
    with round_lock(round_dir):
        completed = run_request(round_dir, request, config, _forbidden, managed_attempt=bridge)
    assert completed["attempt_id"] == original["attempt_id"]
    assert completed["bridge_job_sha256"] == job["job_sha256"]
    assert completed["bridge_result_sha256"] == result["result_sha256"]
    assert completed["provider_elapsed_seconds"] == 2.1
    assert read(bridge.model_lock) == result["model_identity"]
    with round_lock(round_dir):
        assert run_request(round_dir, request, config, _forbidden, managed_attempt=bridge) == completed
    assert not (turn / "attempt_01.started.json").exists()


def test_explicit_rejection_uses_next_reservation_but_ambiguity_never_resends(setup_bridge, tmp_path):
    config, request, _, bridge = setup_bridge
    round_dir = tmp_path / "round_00"
    turn = round_dir / "turn_00"
    clock = [100.0]

    def advance(seconds):
        clock[0] += seconds

    def invoke():
        with round_lock(round_dir):
            return run_request(round_dir, request, config, _forbidden, managed_attempt=bridge,
                               now=lambda: clock[0], sleep=advance, jitter=lambda: 0.0)

    with pytest.raises(BridgePending):
        invoke()
    first = read(turn / "attempt_00.started.json")
    first_name = f"{first['attempt_id']}.json"
    first_job = read_bridge_json(bridge.outbox / first_name)
    rejected = make_result(first_job, failure=TransportFailure("http_429", retryable=True, retry_after=1),
                           credential_id="mock-slot-1", provider_dispatched=True)
    once(bridge.results / first_name, rejected)
    with pytest.raises(BridgePending):
        invoke()
    second = read(turn / "attempt_01.started.json")
    second_name = f"{second['attempt_id']}.json"
    second_job = read_bridge_json(bridge.outbox / second_name)
    assert first["attempt_id"] != second["attempt_id"]
    assert first_job["request"] == second_job["request"] == request
    ambiguous = make_result(second_job, failure=TransportFailure("timeout", ambiguous=True),
                            credential_id="mock-slot-2", provider_dispatched=True)
    once(bridge.results / second_name, ambiguous)
    for _ in range(2):
        with pytest.raises(TransportStop, match="timeout"):
            invoke()
    assert len(list(bridge.outbox.glob("*.json"))) == 2
    assert read(turn / "attempt_01.terminal.json")["ambiguous"] is True


def test_courier_rejects_broad_permissions_and_existing_model_lock_drift(setup_bridge, tmp_path):
    config, request, started, bridge = setup_bridge
    job = make_job(request, config, started)
    result = _payload(config, job)
    path = bridge.results / f"{started['attempt_id']}.json"
    once(path, result)
    os.chmod(path, 0o644)
    with pytest.raises(PermissionError, match="owner-only regular file"):
        bridge(request, config, started)
    os.chmod(path, 0o600)
    once(bridge.model_lock, {**result["model_identity"], "reported_model_version": "different-model-version"})
    with pytest.raises(TransportFailure, match="provider_identity_drift"):
        bridge(request, config, started)
    foreign = tmp_path / "foreign"
    foreign.mkdir(mode=0o755)
    with pytest.raises(PermissionError, match="spool directory must be private"):
        BridgeCourier(foreign, contract_sha256=CONTRACT, authorization_sha256=AUTHORIZATION)



def test_background_lifecycle_flags_are_signed_into_the_bridge_job(setup_bridge):
    streaming, source, started, _ = setup_bridge
    background = replace(streaming, wire_mode=RESPONSES_BACKGROUND)
    request = make_request(source["binding"], 0, source["payload"]["input"], background)
    assert request["payload"] == {**source["payload"], "stream": False, "background": True}
    assert "background" not in source["payload"] and source["payload"]["stream"] is True
    assert request["payload_sha256"] != source["payload_sha256"]
    assert request["request_id"] != source["request_id"]

    started = {**started, "request_id": request["request_id"]}
    job = make_job(request, background, started)
    assert verify_job(job, authorization_sha256=AUTHORIZATION, contract_sha256=CONTRACT) == (request, background, started)
    forged = deepcopy(job)
    forged["request"]["payload"]["background"] = False
    forged["job_sha256"] = digest({key: value for key, value in forged.items() if key != "job_sha256"})
    with pytest.raises(ValueError, match="request identity"):
        verify_job(forged)
