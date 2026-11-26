from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import (
    dispatch_v13_retry_successor as mac,
    run_v13_retry_successor as gpu,
)


def test_only_explicitly_reviewed_new_wave_can_reserve_its_one_current_slot():
    class Parser:
        def error(self, message):
            raise ValueError(message)

    scope = gpu.read_scope(Parser(), SimpleNamespace(seed=12203, wave=3))
    assert scope.budget == {"first_logical_requests": 32, "maximum_logical_requests": 94,
                            "maximum_physical_attempts": 187}
    assert scope.not_before_unix - scope.cooldown_unix == 180
    assert len(scope.archive) == 2
    assert scope.archive[-1].failure == "connection_interrupted"
    for receipt in scope.archive:
        assert all(mac._sha(value) for value in (receipt.identity.contract, receipt.identity.approval, receipt.request_id,
                                                 receipt.job_sha256, receipt.result_sha256, receipt.terminal_file_sha256,
                                                 receipt.progress_file_sha256))
    with pytest.raises(ValueError, match="no individually reviewed"):
        gpu.read_scope(Parser(), SimpleNamespace(seed=12203, wave=12))


@pytest.mark.parametrize("seed,wave,rev_counts,physical_budget", [
    (12203, 3, [1, 1], 187), (12203, 4, [1, 1, 1], 187), (12203, 5, [1, 1, 1, 1], 187),
    (12203, 6, [1, 1, 1, 1, 1], 187), (12203, 10, [2, 2, 2, 1], 181),
    (157, 3, [2, 1], 193), (157, 4, [2, 1, 1], 193), (157, 5, [2, 1, 1, 1], 193),
    (157, 6, [2, 1, 1, 1, 1], 193), (157, 7, [1], 187),
    (6101, 2, [2], 199), (6101, 3, [2, 1], 199), (6101, 5, [2, 1, 1, 1], 199), (2357, 2, [2], 187),
    (2357, 5, [2, 1, 2, 1], 187),
    (887, 1, [1], 163), (887, 3, [1, 1], 163), (887, 4, [1, 1, 1], 163), (887, 5, [1], 139),
])
def test_each_prior_identity_and_new_second_slot_are_rejected(monkeypatch, seed, wave, rev_counts, physical_budget):
    scope = gpu.SCOPES[(seed, wave)]
    assert [len(wave.attempts) for wave in scope.archive] == rev_counts
    assert scope.budget["maximum_physical_attempts"] == physical_budget
    approved = []
    monkeypatch.setattr(mac.RelayRetryDispatcher, "_check_boundary",
                        lambda self, request, started: approved.append(started))
    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    new = {"attempt": 0, "attempt_id": "b" * 32}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary({}, new)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    retired = [{**new, "attempt_id": item.attempt_id} for wave in scope.archive for item in wave.attempts]
    for denied in [{**new, "attempt": 1}, *retired]:
        with pytest.raises(PermissionError, match="signed physical attempts"):
            dispatcher._check_boundary({}, denied)
    dispatcher._check_boundary({}, new)
    assert approved == [new]


def test_6101_scope_pins_the_next_fitted_round_and_prior_replay_transcript():
    scope = gpu.SCOPES[(6101, 2)]
    assert (scope.original_start_round, scope.round) == (1, 2)
    assert set(dict(scope.fitted_prefix_files)) == {"round_02/source.json", "round_02/transition.json", "round_01/memory.json"}
    assert set(scope.archive[-1].replay_files) == {"attempt_00.replay_authorization.json",
                                                 "attempt_00.replay.terminal.json", "attempt_01.replay_link.json"}
    for item in scope.archive[-1].attempts:
        assert all(mac._sha(value) for value in (item.job_sha256, item.result_sha256,
                                                 item.terminal_file_sha256, item.progress_file_sha256))


def test_6101_wave03_retains_the_fitted_boundary_already_signed_by_wave02():
    previous = gpu.SCOPES[(6101, 2)]
    current = gpu.SCOPES[(6101, 3)]
    assert current.archive[:-1] == previous.archive
    assert (current.round, current.payload_sha256, current.budget) == (previous.round, previous.payload_sha256,
                                                                     previous.budget)
    assert current.original_start_round is None and current.fitted_prefix_files == ()
    assert current.required_sources == gpu.QUEUE_SOURCES
    assert current.not_before_unix == 1795686584.2017465
    latest = current.archive[-1]
    assert all(mac._sha(value) for value in (latest.identity.contract, latest.identity.approval, latest.job_sha256,
                                              latest.result_sha256, latest.terminal_file_sha256, latest.progress_file_sha256))


def test_wave04_preserves_157_prior_reviews_and_uses_the_latest_clock():
    previous = gpu.SCOPES[(157, 3)]
    current = gpu.SCOPES[(157, 4)]
    assert current.archive[:-1] == previous.archive
    assert current.round == previous.round == 3
    assert current.payload_sha256 == previous.payload_sha256
    assert current.budget == previous.budget
    assert current.cooldown_field == "not_before_unix"
    assert current.not_before_unix == 1795685629.213155
    latest = current.archive[-1]
    assert all(mac._sha(value) for value in (latest.identity.contract, latest.identity.approval, latest.job_sha256,
                                              latest.result_sha256, latest.terminal_file_sha256, latest.progress_file_sha256))


def test_12203_inactivity_review_requires_exact_transport_source_and_cooldown():
    previous = gpu.SCOPES[(12203, 3)]
    current = gpu.SCOPES[(12203, 4)]
    assert current.archive[:-1] == previous.archive
    assert current.round == previous.round == 4
    assert current.budget == previous.budget
    assert current.archive[-1].progress["error_type"] == "sse_inactivity_timeout"
    assert current.not_before_unix == 1795685888.5045788
    assert dict(current.required_sources)[(gpu.STUDY / "a7_gpu/dispatcher.py").as_posix()] == (
        "cde7ba080ba728b7600764d733cfab8a87bbe8e537840929bc1f47cd6c2cb3fc"
    )


def test_wave05_12203_extends_the_previous_archive_with_a_finished_alias_failure():
    previous = gpu.SCOPES[(12203, 4)]
    current = gpu.SCOPES[(12203, 5)]
    assert current.archive[:-1] == previous.archive
    assert current.required_sources == previous.required_sources
    assert (current.round, current.payload_sha256, current.budget) == (previous.round, previous.payload_sha256,
                                                                     previous.budget)
    assert current.not_before_unix == 1795686631.363517
    latest = current.archive[-1]
    assert latest.progress["conflicting_event_type"] == "response.failed"
    assert all(mac._sha(value) for value in (latest.identity.contract, latest.identity.approval, latest.job_sha256,
                                              latest.result_sha256, latest.terminal_file_sha256, latest.progress_file_sha256))


def test_wave05_157_preserves_the_prior_chain_and_requires_reviewed_dispatcher():
    previous = gpu.SCOPES[(157, 4)]
    current = gpu.SCOPES[(157, 5)]
    assert current.archive[:-1] == previous.archive
    assert (current.round, current.payload_sha256, current.budget) == (previous.round, previous.payload_sha256,
                                                                     previous.budget)
    assert current.not_before_unix == 1795686367.2452364
    assert current.required_sources == gpu.SCOPES[(12203, 4)].required_sources
    latest = current.archive[-1]
    assert all(mac._sha(value) for value in (latest.identity.contract, latest.identity.approval, latest.job_sha256,
                                              latest.result_sha256, latest.terminal_file_sha256, latest.progress_file_sha256))


def test_wave06_157_adds_one_exact_review_without_editing_the_previous_scope():
    previous = gpu.SCOPES[(157, 5)]
    current = gpu.SCOPES[(157, 6)]
    assert current.archive[:-1] == previous.archive
    assert current.required_sources == previous.required_sources
    assert (current.round, current.payload_sha256, current.budget) == (previous.round, previous.payload_sha256,
                                                                     previous.budget)
    assert current.not_before_unix == 1795687031.0807147
    latest = current.archive[-1]
    assert all(mac._sha(value) for value in (latest.identity.contract, latest.identity.approval, latest.job_sha256,
                                              latest.result_sha256, latest.terminal_file_sha256, latest.progress_file_sha256))


def test_887_zero_post_cached_round_must_match_the_signed_public_resource_ancestor(tmp_path, monkeypatch):
    scope = gpu.SCOPES[(887, 1)]
    assert (scope.original_start_round, scope.round) == (7, 8)
    assert scope.budget == {"first_logical_requests": 28, "maximum_logical_requests": 82,
                            "maximum_physical_attempts": 163}
    old_run = tmp_path / "cached"
    directory = old_run / "seed_887"
    cached = {"provider_post": False}
    receipt = old_run / "cached_response_import.json"
    once(receipt, cached)
    terminal = directory / "round_07/turn_00/attempt_00.terminal.json"
    once(terminal, {"status": "completed", "provider_dispatched": False, "attempt_id": "cached"})
    origin = replace(scope.cached_origin, proof_sha256=digest(cached), receipt_file_sha256=file_digest(receipt))
    scope = replace(scope, cached_origin=origin)
    source = {"cache_directory": "same", "model_identity": "same", "public_resource_receipt_sha256": origin.resource_sha256}
    old = {**source, "previous_run_directory": str(tmp_path / origin.ancestor.name),
           "previous_release_directory": str(tmp_path / "signed_release"), "previous_contract_sha256": origin.ancestor.contract,
           "previous_authorization_sha256": origin.ancestor.approval, "cached_import": cached}
    monkeypatch.setattr(gpu.retry.bounded, "_archived", lambda *args: (SimpleNamespace(predecessor=source), {}))
    assert gpu._cached_resources(old_run, directory, old, scope) == source
    terminal.write_text(dumps({"status": "completed", "provider_dispatched": True, "attempt_id": "cached"}))
    with pytest.raises(PermissionError, match="no-POST"):
        gpu._cached_resources(old_run, directory, old, scope)


def test_887_replacement_signs_relay_helpers_missing_from_cached_parent(tmp_path):
    original = gpu.SCOPES[(887, 1)]
    replacement = gpu.SCOPES[(887, 2)]
    assert replacement == replace(original, wave=2)
    assert replacement.run_name != original.run_name
    for path in gpu.SIGNED_RETRY_SOURCES:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(path.name)
    parent = SimpleNamespace(source_sha256={})
    sources = gpu._source_manifest(tmp_path, parent, replace(replacement, required_sources=()))
    for module in ("dispatch_v13_relay_retry", "run_v13_relay_retry"):
        path = gpu.STUDY / "scripts" / f"{module}.py"
        assert sources[path.as_posix()] == file_digest(tmp_path / path)
        assert path.as_posix() not in parent.source_sha256
    assert set(sources) == {path.as_posix() for path in gpu.SIGNED_RETRY_SOURCES}


def test_reviewed_upstream_read_errors_keep_the_exact_prior_body_and_budget():
    for seed, wave, not_before in ((887, 3, 1795687197.5816553), (12203, 6, 1795687215.0316763)):
        scope = gpu.SCOPES[(seed, wave)]
        prior = gpu.SCOPES[(seed, wave - 1)]
        assert scope.archive[:-1] == prior.archive
        assert (scope.round, scope.payload_sha256, scope.budget) == (prior.round, prior.payload_sha256, prior.budget)
        assert scope.required_sources == gpu.QUEUE_SOURCES
        assert scope.not_before_unix == not_before
        assert scope.archive[-1].failure == "error"
        assert scope.archive[-1].progress["provider_error_code"] == "upstream_stream_read_error"
    old = gpu.SCOPES[(887, 2)]
    new = gpu.SCOPES[(887, 3)]
    assert new.cached_origin == old.cached_origin
    assert new.cached_origin.review["provider_post"] is False
    assert new.original_start_round is None and new.fitted_prefix_files == ()


def test_887_wave04_preserves_cached_origin_and_only_extends_the_posted_archive():
    old = gpu.SCOPES[(887, 3)]
    scope = gpu.SCOPES[(887, 4)]
    assert scope.archive[:-1] == old.archive
    assert scope.cached_origin.review == old.cached_origin.review
    assert scope.cached_origin.review["provider_post"] is False
    assert scope.required_sources == gpu.QUEUE_SOURCES
    assert (scope.round, scope.payload_sha256, scope.budget) == (old.round, old.payload_sha256, old.budget)
    assert scope.not_before_unix == 1795688044.4677405
    last = scope.archive[-1]
    assert all(mac._sha(value) for value in (last.identity.contract, last.identity.approval, last.job_sha256,
                                            last.result_sha256, last.terminal_file_sha256, last.progress_file_sha256))


def test_6101_wave04_pins_unfinished_text_telemetry_without_resetting_the_fitted_boundary():
    scope = gpu.SCOPES[(6101, 4)]
    assert scope.payload_sha256 == gpu.SCOPES[(6101, 3)].payload_sha256
    assert scope.original_start_round is None and scope.round == 2
    assert [len(review.attempts) for review in scope.archive] == [2, 1, 1]
    assert scope.budget == {"first_logical_requests": 34, "maximum_logical_requests": 100, "maximum_physical_attempts": 200}
    last = scope.archive[-1]
    assert last.failure == "timeout" and last.progress["error_type"] == "hard_wall_deadline"
    assert last.progress["event_type"] == "response.output_text.delta"
    assert last.progress["sse_event_count"] == 3016


def test_6101_wave05_carries_only_the_remaining_slot_after_the_verified_frame_error():
    previous = gpu.SCOPES[(6101, 4)]
    scope = gpu.SCOPES[(6101, 5)]
    assert scope.payload_sha256 == previous.payload_sha256
    assert scope.original_start_round == scope.round == previous.round == 2
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.archive[:-1] == previous.archive
    last = scope.archive[-1]
    assert scope.boundary_physical_attempts + len(last.attempts) == previous.boundary_physical_attempts
    assert last.failure == "sse_frame_error" and last.progress["error_type"] == "ReadError"
    assert last.progress["event_type"] == "response.output_text.delta" and last.progress["sse_event_count"] == 252
    assert scope.not_before_unix == 1795692531.0271559
    assert scope.replay_start_round == 3


def test_6101_wave06_uses_the_exact_failed_response_at_the_same_fitted_boundary():
    previous = gpu.SCOPES[(6101, 5)]
    scope = gpu.SCOPES[(6101, 6)]
    assert scope.payload_sha256 == previous.payload_sha256
    assert scope.original_start_round is None and scope.round == scope.replay_start_round == previous.round == 2
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == {"first_logical_requests": 34, "maximum_logical_requests": 100, "maximum_physical_attempts": 200}
    assert scope.required_sources == gpu.QUEUE_SOURCES
    assert scope.archive[:-1] == previous.archive
    assert [len(review.attempts) for review in scope.archive] == [2, 1, 1, 1, 1]
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795693173.4806566
    last = scope.archive[-1]
    assert last.failure == last.progress["event_type"] == "response.failed"
    assert last.progress["http_status"] == 200 and last.progress["provider_error_code"] == "server_error"
    assert last.progress["sse_event_count"] == 41 and "first_text_seconds" not in last.progress


def test_2357_previous_partial_text_telemetry_is_bound_to_the_stopped_physical_attempt():
    scope = gpu.SCOPES[(2357, 2)]
    assert scope.round == scope.original_start_round == 4
    assert scope.not_before_unix == 1795687459.5865693
    assert scope.required_sources == gpu.QUEUE_SOURCES
    last = scope.archive[-1]
    assert len(last.attempts) == 2 and len(last.replay_files) == 3
    assert last.failure == "timeout" and last.progress["event_type"] == "response.output_text.delta"
    assert last.progress["error_type"] == "hard_wall_deadline"
    assert all(mac._sha(value) for value in (last.job_sha256, last.result_sha256, last.terminal_file_sha256,
                                            last.progress_file_sha256))


@pytest.mark.parametrize("seed,wave,expected_not_before", [
    (2357, 3, 1795688254.0102787), (2357, 4, 1795690643.3580766), (6101, 4, 1795690796.5505776),
    (2357, 6, 1795698146.9968047),
    (6101, 6, 1795693173.4806566),
    (12203, 8, 1795690919.9466105), (12203, 9, 1795693155.7197583), (12203, 11, 1795697031.8253946),
    (157, 8, 1795696780.5165217),
])
def test_conditional_second_attempt_requires_existing_delivered_failure_guard(monkeypatch, seed, wave, expected_not_before):
    previous = gpu.SCOPES[(seed, wave - 1)]
    scope = gpu.SCOPES[(seed, wave)]
    assert scope.archive[:-1] == previous.archive
    assert scope.execution_budget(previous.boundary_physical_attempts) == previous.budget
    assert scope.budget == scope.execution_budget(2)
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.replay_start_round == scope.round == previous.round
    assert scope.not_before_unix == expected_not_before
    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": scope.attempt_id}]}
    dispatcher.original = {"binding": {}, "logical_index": 0, "payload": {"same": True}, "request_id": "previous"}
    request = {**dispatcher.original, "request_id": "current"}
    second = {"attempt": 1, "attempt_id": "new-physical"}
    observed = []
    monkeypatch.setattr(mac.RelayRetryDispatcher, "_check_boundary_retry",
                        lambda self, call, started: observed.append((call, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, second)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, {"attempt": 0, "attempt_id": "first-physical"})
    assert not observed
    dispatcher._check_boundary(request, second)
    assert observed == [(request, second)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second, "attempt": 2})


def test_2357_wave04_keeps_both_delivered_failures_and_the_exact_timeout_evidence():
    scope = gpu.SCOPES[(2357, 4)]
    assert scope.payload_sha256 == gpu.SCOPES[(2357, 3)].payload_sha256
    assert scope.original_start_round == scope.round == 4
    assert scope.original_boundary_physical_attempts == 2
    assert [len(review.attempts) for review in scope.archive] == [2, 1, 2]
    first, last = scope.archive[-1].attempts
    assert first.request_id == last.request_id and first.identity == last.identity
    assert first.progress["conflicting_event_type"] == "response.failed"
    assert last.failure == "timeout"
    assert last.progress["error_type"] == "sse_inactivity_timeout"
    assert last.progress["sse_event_count"] == 14
    assert len(last.replay_files) == 3
    assert all(mac._sha(value) for value in last.replay_files.values())


def test_2357_wave05_keeps_only_the_unused_slot_and_pins_the_signed_transport_update():
    previous = gpu.SCOPES[(2357, 4)]
    scope = gpu.SCOPES[(2357, 5)]
    assert scope.payload_sha256 == previous.payload_sha256
    assert scope.original_start_round == scope.round == previous.round == 4
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.archive[:-1] == previous.archive
    last = scope.archive[-1]
    assert scope.boundary_physical_attempts + len(last.attempts) == previous.boundary_physical_attempts
    assert last.failure == "timeout" and last.progress["error_type"] == "hard_wall_deadline"
    assert last.progress["event_type"] == "response.output_text.delta" and last.progress["sse_event_count"] == 3170
    assert scope.not_before_unix == 1795694805.301756
    assert scope.replay_start_round == 5
    assert len(scope.required_sources) == len(gpu.QUEUE_SOURCES) + 2
    assert all(name in gpu.retry.OVERLAYS for name, _ in scope.required_sources)


def test_2357_wave06_pins_the_exact_delivered_inactivity_failure_and_stable_transport():
    previous = gpu.SCOPES[(2357, 5)]
    scope = gpu.SCOPES[(2357, 6)]
    assert scope.payload_sha256 == previous.payload_sha256
    assert scope.original_start_round is None and scope.round == previous.round == 4
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.archive[:-1] == previous.archive
    assert [len(review.attempts) for review in scope.archive] == [2, 1, 2, 1, 1]
    assert scope.budget == {"first_logical_requests": 32, "maximum_logical_requests": 94, "maximum_physical_attempts": 188}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795698146.9968047
    assert scope.replay_start_round == 4
    last = scope.archive[-1]
    assert last.failure == "timeout" and last.replay_files is None
    assert last.terminal_file_sha256 == "62945fd3e4bb2a9c4765cb9a15fce22cd2f2687748a42e55694f3efc5bb6427c"
    assert last.progress["error_type"] == "sse_inactivity_timeout"
    assert last.progress["event_type"] == "response.output_text.delta" and last.progress["sse_event_count"] == 831
    assert scope.required_sources == previous.required_sources == gpu.SAME_RESPONSE_RETRY_SOURCES
    assert all(file_digest(gpu.ROOT / name) == sha for name, sha in scope.required_sources)


def test_12203_wave10_keeps_one_unused_equivalent_after_inactivity():
    previous = gpu.SCOPES[(12203, 9)]
    scope = gpu.SCOPES[(12203, 10)]
    assert scope.payload_sha256 == previous.payload_sha256
    assert scope.original_start_round == scope.round == previous.round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.archive[:-1] == previous.archive
    last = scope.archive[-1]
    assert scope.boundary_physical_attempts + len(last.attempts) == previous.boundary_physical_attempts
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 181}
    assert scope.cooldown_field == "failed_unix" and scope.not_before_unix == 1795695858.1715395
    assert scope.replay_start_round == 6
    assert last.failure == "timeout" and last.replay_files is None
    assert last.progress["error_type"] == "sse_inactivity_timeout"
    assert last.progress["event_type"] == "response.output_item.done" and last.progress["sse_event_count"] == 28
    assert scope.required_sources == gpu.SCOPES[(2357, 5)].required_sources
    assert len(scope.required_sources) == len(gpu.QUEUE_SOURCES) + 2
    assert all(file_digest(gpu.ROOT / name) == sha for name, sha in scope.required_sources)


def test_12203_wave11_binds_the_exact_alias_failure_and_keeps_the_stable_transport():
    previous = gpu.SCOPES[(12203, 10)]
    scope = gpu.SCOPES[(12203, 11)]
    assert scope.payload_sha256 == previous.payload_sha256
    assert scope.original_start_round is None and scope.round == previous.round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert [len(review.attempts) for review in scope.archive] == [2, 2, 2, 1, 1]
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795697031.8253946
    assert scope.replay_start_round == 5
    last = scope.archive[-1]
    assert last.failure == "response_id_mismatch" and last.replay_files is None
    assert last.terminal_file_sha256 == "545be789c7eb182cc8cff91c3c3e3e88b777acc2395f2f63d520e12894ae9f56"
    assert last.progress["conflicting_event_type"] == "response.failed" and last.progress["sse_event_count"] == 25
    assert last.progress["response_id_matches_transport_alias"] is True
    assert scope.required_sources == previous.required_sources == gpu.SAME_RESPONSE_RETRY_SOURCES
    assert all(file_digest(gpu.ROOT / name) == sha for name, sha in scope.required_sources)


def test_12203_wave07_uses_the_fitted_next_boundary_with_a_conditional_retry():
    previous = gpu.SCOPES[(12203, 6)]
    scope = gpu.SCOPES[(12203, 7)]
    assert scope.original_start_round == previous.round == 4
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.round == 5 and scope.payload_sha256 != previous.payload_sha256
    assert scope.replay_start_round == 5
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182}
    assert scope.not_before_unix == 1795688861.139692
    assert set(dict(scope.fitted_prefix_files)) == {"round_05/source.json", "round_05/transition.json", "round_04/memory.json"}
    assert len(scope.archive[-1].attempts) == 2 and len(scope.archive[-1].replay_files) == 3
    assert all(mac._sha(value) for value in scope.archive[-1].replay_files.values())


@pytest.mark.parametrize("wave,chain,event_counts", [(8, [2, 2], [41, 23]), (9, [2, 2, 2], [24, 41])])
def test_12203_keeps_both_delivered_alias_failures_at_the_same_fitted_boundary(wave, chain, event_counts):
    previous = gpu.SCOPES[(12203, wave - 1)]
    scope = gpu.SCOPES[(12203, wave)]
    assert scope.payload_sha256 == previous.payload_sha256
    assert scope.original_start_round == scope.round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                             "maximum_physical_attempts": 182}
    assert [len(review.attempts) for review in scope.archive] == chain
    first, last = scope.archive[-1].attempts
    assert first.request_id == last.request_id and first.identity == last.identity
    assert all(item.failure == "response_id_mismatch" and item.progress["conflicting_event_type"] == "response.failed"
               for item in (first, last))
    assert [item.progress["sse_event_count"] for item in (first, last)] == event_counts
    assert len(last.replay_files) == 3
    assert all(mac._sha(value) for value in last.replay_files.values())


def test_157_wave07_carries_one_remaining_api_slot_after_the_zero_post_cached_fit():
    scope = gpu.SCOPES[(157, 7)]
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (3, 4, 5)
    assert scope.cached_origin.review["provider_post"] is False
    assert scope.cached_origin.ancestor.name.endswith("-157-relay-retry-06")
    assert scope.archive[0].identity.name.endswith("-157-cached-r3-01")
    fitted_prefix = dict(scope.fitted_prefix_files)
    assert "round_03/turn_00/attempt_00.terminal.json" in fitted_prefix
    assert "round_03/memory.json" in fitted_prefix and "round_04/transition.json" in fitted_prefix
    assert scope.boundary_physical_attempts + len(scope.archive[0].attempts) == 2
    assert scope.not_before_unix == 1795692680.964523
    assert scope.budget == {"first_logical_requests": 32, "maximum_logical_requests": 94, "maximum_physical_attempts": 187}
    assert scope.archive[0].failure == "timeout" and scope.archive[0].progress["sse_event_count"] == 1264


def test_157_wave08_keeps_the_reviewed_zero_post_origin_and_pins_the_new_transport_sources():
    previous = gpu.SCOPES[(157, 7)]
    scope = gpu.SCOPES[(157, 8)]
    assert scope.payload_sha256 == previous.payload_sha256
    assert scope.original_start_round is None and scope.round == previous.round == 4
    assert scope.cached_origin.review == previous.cached_origin.review
    assert [len(review.attempts) for review in scope.archive] == [1, 1]
    assert scope.budget == {"first_logical_requests": 32, "maximum_logical_requests": 94, "maximum_physical_attempts": 188}
    assert scope.cooldown_field == "not_before_unix"
    assert scope.required_sources == gpu.SCOPES[(2357, 5)].required_sources
    last = scope.archive[-1]
    assert last.failure == "timeout" and last.progress["error_type"] == "hard_wall_deadline"
    assert last.progress["event_type"] == "response.output_text.delta" and last.progress["sse_event_count"] == 2783


def test_887_r12_carries_the_one_unused_slot_and_the_exact_fitted_prefix():
    scope = gpu.SCOPES[(887, 5)]
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (9, 12, 13)
    assert scope.original_boundary_physical_attempts == 2
    assert scope.boundary_physical_attempts + len(scope.archive[0].attempts) == 2
    assert scope.cached_origin is None
    assert scope.archive[0].identity == gpu._887_R9_CONTINUATION
    assert scope.budget == {"first_logical_requests": 24, "maximum_logical_requests": 70, "maximum_physical_attempts": 139}
    assert scope.not_before_unix == 1795697737.9016292
    assert scope.required_sources == gpu.SCOPES[(2357, 5)].required_sources
    assert {f"round_{i:02d}/transition.json" for i in (10, 11, 12)} <= dict(scope.fitted_prefix_files).keys()


def test_single_reviewed_origin_does_not_require_an_untraversed_previous_pointer(tmp_path, monkeypatch):
    scope = gpu.SCOPES[(887, 5)]
    run, release = tmp_path / "old-run", tmp_path / "old-release"
    parent = SimpleNamespace(predecessor={"model_identity": {"served_model": "locked"}})
    monkeypatch.setattr(gpu.retry.bounded, "_archived", lambda actual_run, actual_release, identity:
                        (parent, {"id": identity.contract}) if (actual_run, actual_release) == (run, release) else pytest.fail())
    monkeypatch.setattr(gpu.retry, "_retired", lambda *args, **kwargs: (run, run))
    monkeypatch.setattr(gpu, "_evidence", lambda *args: ({}, [], {}))
    once(run / "provider_model_lock.json", parent.predecessor["model_identity"])
    assert [record["contract"] for record in gpu._archives(run, release, scope)] == [parent]


def test_exact_887_source_only_origin_replays_three_fit_verifications_and_rejects_inserted_user_message(tmp_path, monkeypatch):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import backend

    scope = gpu.SCOPES[(887, 5)]
    directory = tmp_path / "seed_887"
    initial = directory / "round_09"
    active = directory / "round_12"
    start = gpu.seed_runner.FrozenState(887, 9, "initial", {}, [], [], {"expected_observed": list(range(621))})
    memory = {"rounds": list(range(9))}
    descriptor = {"round": 9, "state_sha256": digest(asdict(start)), "memory_sha256": digest(memory)}
    formal = {"scope": "old"}
    parent = SimpleNamespace(digest=gpu._887_R9_CONTINUATION.contract, profile={"resume_boundaries": {"887": descriptor}},
                             predecessor={"dispatcher_module": "dispatch_v13_887_r9_continuation"}, transport=object())
    once(initial / "source.json", asdict(start))
    once(initial / "boundary_memory.json", memory)
    once(directory / "boundary.json", {"formal_run": formal, **descriptor})
    states = {index: replace(start, round=index, checkpoint=(gpu._887_R12_CHECKPOINT if index == 12 else str(index)),
                             provenance={"expected_observed": list(range(333 + 32 * index))}) for index in (10, 11, 12)}
    for index in (10, 11, 12):
        once(directory / f"round_{index:02d}" / "transition.json", {"round": index})
    once(active / "source.json", asdict(states[12]))
    reached = []
    monkeypatch.setattr(gpu.seed_runner, "_scope", lambda *_: formal)
    monkeypatch.setattr(backend, "TrainingBackend", lambda *_: object())

    def verify_fit(prior, state, following, trusted_scope, transport, contract, cached):
        reached.append(state.round + 1)
        assert prior.name == f"round_{state.round:02d}" and following.name == f"round_{reached[-1]:02d}"
        assert trusted_scope == formal and contract is parent and transport is parent.transport
        return states[reached[-1]], {"memory_round": reached[-1]}

    monkeypatch.setattr(gpu.seed_runner, "_successor", verify_fit)
    boundary, proof = gpu._reviewed_boundary(tmp_path, directory, active, parent, {}, scope, tmp_path)
    assert reached == [10, 11, 12] and boundary == (states[12], {"memory_round": 12}, 12)
    assert proof["transition_12"] == file_digest(active / "transition.json")
    once(initial / "boundary_user_message.json", {"content": "unbound"})
    with pytest.raises(PermissionError, match="R9 continuation boundary"):
        gpu._reviewed_boundary(tmp_path, directory, active, parent, {}, scope, tmp_path)
