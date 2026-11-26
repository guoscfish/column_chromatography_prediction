from dataclasses import replace
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
        gpu.read_scope(Parser(), SimpleNamespace(seed=12203, wave=6))


@pytest.mark.parametrize("seed,wave,rev_counts,physical_budget", [
    (12203, 3, [1, 1], 187), (12203, 4, [1, 1, 1], 187), (12203, 5, [1, 1, 1, 1], 187),
    (157, 3, [2, 1], 193), (157, 4, [2, 1, 1], 193), (157, 5, [2, 1, 1, 1], 193),
    (6101, 2, [2], 199), (6101, 3, [2, 1], 199), (887, 1, [1], 163),
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
    with pytest.raises(PermissionError, match="last cooldown"):
        dispatcher._check_boundary({}, new)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    retired = [{**new, "attempt_id": item.attempt_id} for wave in scope.archive for item in wave.attempts]
    for denied in [{**new, "attempt": 1}, *retired]:
        with pytest.raises(PermissionError, match="one distinct physical attempt"):
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
