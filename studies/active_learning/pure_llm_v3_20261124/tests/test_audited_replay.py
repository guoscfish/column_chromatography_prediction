import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    UNKNOWN_REPLAY_DECISION, ProviderReply, TransportConfig, TransportFailure, TransportStop,
    authorize_ambiguous_replay, make_request, reconcile, run_request, terminal_record,
)


@pytest.fixture
def unknown_attempt(tmp_path):
    config = TransportConfig("fixture", "mock", "mock", "xhigh", "https://example.invalid/v1", 30, 32,
                             transport_attempts=2)
    contract_sha = "c" * 64
    binding = {"seed": 887, "round": 3, "a7_formal": {"contract_sha256": contract_sha, "mock": False}}
    request = make_request(binding, 1, [{"role": "user", "content": "The same scientific repair."}], config)
    log = tmp_path / "turn_01"

    def first_failure(_request, _config, _started):
        error = TransportFailure("sse_frame_error", ambiguous=True)
        error.reply_identity = {"provider_dispatched": True, "bridge_result_sha256": "b" * 64}
        raise error

    with pytest.raises(TransportStop, match="sse_frame_error"):
        run_request(tmp_path, request, config, _no_dispatch, managed_attempt=first_failure, now=lambda: 0)
    previous = terminal_record(log, 0)
    authorization = {"decision": UNKNOWN_REPLAY_DECISION, "authorized_by": "offline fixture",
                     "authorization_reference": "approved second attempt despite unknown first execution",
                     "accept_duplicate_billing_risk": True, "contract_sha256": contract_sha,
                     "request_id": request["request_id"], "attempt_id": previous["attempt_id"],
                     "bridge_result_sha256": previous["bridge_result_sha256"],
                     "original_terminal_sha256": digest(previous), "next_attempt": 1}
    return tmp_path, request, config, log, authorization


def _no_dispatch(*_args, **_kwargs):
    pytest.fail("a provider attempt was sent without the exact replay permit")


def _authorize(case):
    directory, request, config, _, approval = case
    return authorize_ambiguous_replay(directory, request, authorization=approval, config=config,
                                      expected_authorization_sha256=digest(approval))


def test_reviewed_replay_keeps_the_original_request_and_unknown_receipt(unknown_attempt):
    directory, request, config, log, approval = unknown_attempt
    primary = log / "attempt_00.terminal.json"
    previous = primary.read_bytes()
    original_request = (log / "request.json").read_bytes()
    replay = _authorize(unknown_attempt)
    assert replay["prior_execution_and_billing"] == "unresolved"
    assert replay["next_attempt"] == 1
    assert terminal_record(log, 0)["ambiguous"] is True
    assert terminal_record(log, 0, replay_authorization_sha256=digest(approval)) == replay

    with pytest.raises(TransportStop, match="sse_frame_error"):
        run_request(directory, request, config, _no_dispatch, managed_attempt=_no_dispatch)
    with pytest.raises(PermissionError, match="trusted digest"):
        run_request(directory, request, config, _no_dispatch, managed_attempt=_no_dispatch,
                    replay_authorizations={request["request_id"]: "e" * 64})
    permit = {request["request_id"]: digest(approval)}
    with pytest.raises(PermissionError, match="two-attempt managed bridge"):
        run_request(directory, request, config, _no_dispatch, replay_authorizations=permit)
    assert not (log / "attempt_01.started.json").exists()

    new = []

    def second(managed_request, managed_config, started):
        new.append(started)
        assert managed_request == request and managed_config == config
        assert started["attempt"] == 1 and started["attempt_id"] != approval["attempt_id"]
        assert read(log / "attempt_01.replay_link.json") == {
            "request_id": request["request_id"], "attempt": 1, "attempt_id": started["attempt_id"],
            "original_attempt_id": approval["attempt_id"],
            "original_terminal_sha256": approval["original_terminal_sha256"],
            "replay_authorization_sha256": digest(approval),
        }
        return ProviderReply("new response", "resp_test", "mock", {"input_tokens": 10})

    finish = run_request(directory, request, config, _no_dispatch, managed_attempt=second,
                         replay_authorizations=permit)
    assert finish["attempt"] == 1 and finish == terminal_record(log, 1)
    assert len(new) == 1
    assert run_request(directory, request, config, _no_dispatch, managed_attempt=_no_dispatch,
                       replay_authorizations=permit) == finish
    assert _authorize(unknown_attempt) == replay
    assert primary.read_bytes() == previous and (log / "request.json").read_bytes() == original_request
    link = log / "attempt_01.replay_link.json"
    link.write_text("{}")
    with pytest.raises(TransportStop, match="attempt link mismatch"):
        run_request(directory, request, config, _no_dispatch, managed_attempt=_no_dispatch,
                    replay_authorizations=permit)


def test_interrupted_replay_reservation_recovers_the_same_attempt_link(unknown_attempt):
    directory, request, config, log, approval = unknown_attempt
    _authorize(unknown_attempt)
    reserved = {"request_id": request["request_id"], "attempt": 1, "attempt_id": "d" * 32,
                "started_unix": 1.0, "transport": "mac_bridge"}
    once(log / "attempt_01.started.json", reserved)

    def second(_request, _config, started):
        assert started == reserved
        assert read(log / "attempt_01.replay_link.json")["attempt_id"] == reserved["attempt_id"]
        return ProviderReply("new response", "resp_reserved", "mock", {"input_tokens": 10})

    finish = run_request(directory, request, config, _no_dispatch, managed_attempt=second,
                         replay_authorizations={request["request_id"]: digest(approval)})
    assert finish["attempt_id"] == reserved["attempt_id"]


def test_second_ambiguous_attempt_exhausts_the_replay_budget(unknown_attempt):
    directory, request, config, log, approval = unknown_attempt
    _authorize(unknown_attempt)
    permit = {request["request_id"]: digest(approval)}

    def second_failure(_request, _config, started):
        assert started["attempt"] == 1
        raise TransportFailure("connection_interrupted", ambiguous=True)

    with pytest.raises(TransportStop, match="connection_interrupted"):
        run_request(directory, request, config, _no_dispatch, managed_attempt=second_failure,
                    replay_authorizations=permit)
    with pytest.raises(TransportStop, match="connection_interrupted"):
        run_request(directory, request, config, _no_dispatch, managed_attempt=_no_dispatch,
                    replay_authorizations=permit)
    with pytest.raises(TransportStop, match="no physical attempt remains"):
        reconcile(directory, request, 1, config=config, authorization={
            **approval, "attempt_id": terminal_record(log, 1)["attempt_id"],
            "decision": "confirmed_not_processed", "evidence": "fixture confirmation",
        })
    assert not (log / "attempt_02.started.json").exists()


def test_audited_replay_rejects_missing_risk_acceptance_and_ordinary_reconciliation(unknown_attempt):
    directory, request, config, log, approval = unknown_attempt
    no_risk_ack = {**approval, "accept_duplicate_billing_risk": False}
    with pytest.raises(PermissionError, match="does not bind"):
        authorize_ambiguous_replay(directory, request, authorization=no_risk_ack, config=config,
                                   expected_authorization_sha256=digest(no_risk_ack))
    with pytest.raises(PermissionError, match="recovery requires"):
        reconcile(directory, request, 0, config=config, authorization=approval)
    assert not (log / "attempt_00.replay_authorization.json").exists()

    # The exact approval can finish staging after interruption between the two immutable writes.
    once(log / "attempt_00.replay_authorization.json", approval)
    with pytest.raises(TransportStop, match="conflicts with audited replay"):
        reconcile(directory, request, 0, config=config, authorization={
            **approval, "decision": "confirmed_not_processed", "evidence": "fixture",
        })
    grant = _authorize(unknown_attempt)
    assert read(log / "attempt_00.replay.terminal.json") == grant
