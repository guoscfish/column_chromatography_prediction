from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportStop, is_sealed_relay_failure
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import (
    dispatch_v13_retry_successor as mac,
    run_v13_retry_successor as gpu,
)


def assert_prior_transport_atoms(scope):
    assert dict(scope.required_sources)[gpu.PROVIDER_SOURCE] == (
        "9b77e82920a303b9429957372d0943a1c729fecca3c80c00d86e0411a1c74bd0"
    )
    assert all(file_digest(gpu.ROOT / name) == sha for name, sha in scope.required_sources if name != gpu.PROVIDER_SOURCE)


def test_2357_header_deadline_is_opt_in_and_binds_runtime_source():
    old = gpu.SCOPES[(2357, 18)]
    sources = {**dict(old.required_sources), **gpu.HEADER_DEADLINE_SOURCES}
    review = replace(old, wave=19, response_headers_seconds=300, required_sources=tuple(sources.items()))
    assert old.header_deadline_profile == {}
    assert review.header_deadline_profile == {"response_headers_seconds": 300}
    for unsigned in (replace(review, seed=157), replace(review, wave=18),
                     replace(review, required_sources=old.required_sources), replace(review, response_headers_seconds=300.0)):
        with pytest.raises(PermissionError, match="reviewed lane and source pins"):
            _ = unsigned.header_deadline_profile


def test_2357_dispatcher_rejects_a_changed_or_unreviewed_header_deadline(monkeypatch, tmp_path):
    old = gpu.SCOPES[(2357, 18)]
    sources = {**dict(old.required_sources), **gpu.HEADER_DEADLINE_SOURCES}
    review = replace(old, wave=19, response_headers_seconds=300, required_sources=tuple(sources.items()))
    sources.update({path.as_posix(): file_digest(gpu.ROOT / path)
                    for path in (*mac.resilient.SOURCES, *gpu.SIGNED_RETRY_SOURCES)})
    monkeypatch.setattr(mac, "verify_local_source", lambda *args, **kwargs: None)
    prefix = {}
    hex_digest = "a" * 64

    def signed_info(lane):
        run = tmp_path / lane.run_name
        predecessor = lane.archive[-1]
        proof = {
            "dispatcher_module": gpu.DISPATCHER, "seed": lane.seed, "round": lane.round, "logical_index": 0,
            "previous_contract_sha256": predecessor.identity.contract,
            "previous_authorization_sha256": predecessor.identity.approval,
            "previous_run_directory": str(run.with_name(predecessor.identity.name)),
            "previous_request_id": predecessor.request_id, "payload_sha256": lane.payload_sha256,
            "previous_terminal_file_sha256": predecessor.terminal_file_sha256,
            "previous_transport_progress_file_sha256": predecessor.progress_file_sha256,
            "previous_transport_observation": predecessor.progress, "recovery_budget": lane.recovery_budget,
            "execution_budget": lane.budget, "bridge_wait_seconds": mac.resilient.TIMEOUTS.bridge_wait_seconds,
            "relay_not_before_unix": lane.not_before_unix, "relay_cooldown_clock": lane.cooldown_field,
            "newly_authorized_boundary_physical_attempts": lane.boundary_physical_attempts,
            "previous_prefix_files": prefix, "previous_prefix_sha256": digest(prefix),
            "reviewed_fitted_prefix_files": dict(lane.fitted_prefix_files),
            **{name: hex_digest for name in ("checkpoint_sha256", "packet_sha256", "wire_packet_sha256",
                                              "base_messages_sha256")},
        }
        profile = {
            "provider_capability": "gated_ipv4", "arm": "A7", "memory": "M2", "device": "cuda",
            "initial_labels": 333, "target_labels": 1485, "batch_size": 32, "api_parallelism": 1,
            "store": False, "confirmation_seeds": [lane.seed], "qualification_seed": lane.seed,
            "requires_initial_qualification": False, "wire_representation": mac.FORMAL_WIRE_REPRESENTATION,
            "sse_inactivity_seconds": 900, "ipv4_preflight_retry": True, "ipv4_readiness_windows": 3,
            "normalize_observed_reference_aliases": True, "audited_future_transport_replay": True,
            "replay_eligible_from_round": lane.replay_start_round, "sealed_relay_retry_seconds": gpu.retry.COOLDOWN_SECONDS,
            "resume_boundaries": {str(lane.seed): {"round": lane.round}},
            "request_timeouts": asdict(mac.resilient.TIMEOUTS), **lane.header_deadline_profile,
        }
        record = {"profile": profile, "predecessor": proof, "api_execution": mac.SSH_API,
                  "run_directory": str(run), "gpu_slots": [0], "budget": lane.budget,
                  "source_sha256": {**sources, **dict(lane.required_sources)}}
        return str(run), {"contract": record}

    run, info = signed_info(review)
    assert mac._scope(info, run, review) == info["contract"]["predecessor"]
    profile = info["contract"]["profile"]
    profile.pop(gpu.HEADER_DEADLINE_FIELD)
    with pytest.raises(PermissionError, match="reviewed archive"):
        mac._scope(info, run, review)
    for changed in (299, 300.0):
        profile[gpu.HEADER_DEADLINE_FIELD] = changed
        with pytest.raises(PermissionError, match="reviewed archive"):
            mac._scope(info, run, review)
    profile[gpu.HEADER_DEADLINE_FIELD] = 300
    info["contract"]["source_sha256"][gpu.PROVIDER_SOURCE] = hex_digest
    with pytest.raises(PermissionError, match="exact reviewed transport source"):
        mac._scope(info, run, review)

    run, info = signed_info(old)
    assert mac._scope(info, run, old)
    info["contract"]["profile"][gpu.HEADER_DEADLINE_FIELD] = 300
    with pytest.raises(PermissionError, match="reviewed archive"):
        mac._scope(info, run, old)

    data_scope = replace(review, reviewed_record_sha256="b" * 64, reviewed_checkpoint_sha256=hex_digest)
    run, info = signed_info(data_scope)
    proof = info["contract"]["predecessor"]
    proof[gpu.REVIEWED_RECORD_FIELD] = data_scope.reviewed_record_sha256
    assert mac._scope(info, run, data_scope) == proof
    with pytest.raises(PermissionError, match="reviewed archive"):
        mac._scope(info, run, review)
    for field in (gpu.REVIEWED_RECORD_FIELD, "checkpoint_sha256", "previous_authorization_sha256"):
        original = proof[field]
        proof[field] = "f" * 64
        with pytest.raises(PermissionError, match="reviewed archive"):
            mac._scope(info, run, data_scope)
        proof[field] = original


@pytest.fixture
def owned_recovery_record():
    seed = 157
    wave = max(wave for lane_seed, wave in gpu.SCOPES if lane_seed == seed)
    anchor = gpu.SCOPES[seed, wave]
    assert anchor.original_start_round == anchor.round and anchor.boundary_physical_attempts == 1
    trace = "0f0e0d0c-0004-0040-0002-000000000001"
    step = {key: format(index + 300, "064x") for index, key in enumerate(gpu.STEP_DIGESTS)}
    step.update(attempt_id="0" * 31 + "1", cooldown_unix=anchor.cooldown_unix + 301,
                provider_request_id=trace, response_id="resp_separate",
                conflicting_response_id="resp_" + trace.replace("-", ""), event_type="response.output_item.added")
    record = {"schema": gpu.REVIEWED_RECORD_SCHEMA, "seed": seed, "round": anchor.round, "anchor_wave": anchor.wave,
              "checkpoint_sha256": "f" * 64, "payload_sha256": anchor.payload_sha256, "steps": [step]}
    return anchor, record


def test_reviewed_data_only_scope_requires_external_sha_and_cannot_shadow(owned_recovery_record):
    anchor, record = owned_recovery_record
    sha = digest(record)
    scope = gpu._record_scope(record, sha, anchor.seed, anchor.wave + 1)
    assert scope.reviewed_record_sha256 == sha and scope.previous_reviewed_record_sha256 is None
    assert scope.archive[:-1] == anchor.archive and scope.archive[-1].receipt_provider_flags == (True, True)
    assert scope.boundary_physical_attempts == 1 and scope.required_sources == anchor.required_sources
    assert scope.not_before_unix == record["steps"][-1]["cooldown_unix"] + 180
    assert scope.replay_start_round == anchor.round + 1
    assert scope.archive[-1].progress["event_type"] == record["steps"][0]["event_type"]
    with pytest.raises(PermissionError, match="externally approved SHA"):
        gpu._record_scope({**record, "checkpoint_sha256": "e" * 64}, sha, anchor.seed, anchor.wave + 1)
    with pytest.raises(PermissionError, match="scientific payload"):
        changed = {**record, "payload_sha256": "e" * 64}
        gpu._record_scope(changed, digest(changed), anchor.seed, anchor.wave + 1)
    with pytest.raises(PermissionError, match="anchor"):
        changed = {**record, "anchor_wave": anchor.wave - 1}
        gpu._record_scope(changed, digest(changed), anchor.seed, anchor.wave)


def test_reviewed_record_next_parent_and_science_must_match(owned_recovery_record):
    anchor, initial = owned_recovery_record
    first = initial["steps"][0]
    second = {**first, "attempt_id": "0" * 31 + "2", "parent_contract_sha256": "9" * 64,
              "cooldown_unix": first["cooldown_unix"] + 301}
    record = {**initial, "steps": [first, second]}
    scope = gpu._record_scope(record, digest(record), anchor.seed, anchor.wave + 2)
    assert scope.previous_reviewed_record_sha256 == digest(initial)
    parent = {"dispatcher_module": gpu.DISPATCHER, gpu.REVIEWED_RECORD_FIELD: digest(initial)}
    request = {"payload_sha256": initial["payload_sha256"],
               "binding": {"checkpoint_sha256": initial["checkpoint_sha256"]}}
    identity = {"provider_status": "failed", "provider_request_id": second["provider_request_id"],
                "response_id": second["conflicting_response_id"], "served_model": "gpt-6.2-astra", "usage": {}}
    failure = {"reply_identity": identity, "code": "response_id_mismatch", "ambiguous": True, "retryable": False}
    result = {"provider_dispatched": True, "raw_response": None, "outcome": {"status": "failed", "failure": failure}}
    gpu._check_reviewed_parent(scope, parent, request, result)
    for wrong_parent, wrong_request in (
        ({**parent, gpu.REVIEWED_RECORD_FIELD: "f" * 64}, request),
        ({**parent, "dispatcher_module": "other"}, request),
        (parent, {**request, "payload_sha256": "a" * 64}),
        (parent, {**request, "binding": {"checkpoint_sha256": "b" * 64}}),
    ):
        with pytest.raises(PermissionError, match="exact parent, payload and checkpoint"):
            gpu._check_reviewed_parent(scope, wrong_parent, wrong_request, result)
    for wrong_identity in ({**identity, "provider_status": "completed"},
                           {**identity, "response_id": "resp_foreign"},
                           {**identity, "usage": {"input_tokens": 999}}):
        wrong = {**result, "outcome": {"status": "failed", "failure": {**failure, "reply_identity": wrong_identity}}}
        with pytest.raises(PermissionError, match="original owned upstream failure"):
            gpu._check_reviewed_parent(scope, parent, request, wrong)


def test_reviewed_record_denies_repeated_attempt_and_exhaustion(owned_recovery_record):
    anchor, initial = owned_recovery_record
    step = initial["steps"][0]
    repeated = {**initial, "steps": [step, step]}
    with pytest.raises(PermissionError, match="repeated"):
        gpu._record_scope(repeated, digest(repeated), anchor.seed, anchor.wave + 2)
    exhausted = {**initial, "steps": [step] * (gpu.MAX_REVIEWED_STEPS + 1)}
    with pytest.raises(PermissionError, match="finite successor chain"):
        gpu._record_scope(exhausted, digest(exhausted), anchor.seed, anchor.wave + len(exhausted["steps"]))
    for tampered in ({"conflicting_response_id": "resp_unowned"}, {"event_type": "response.output_text.done"},
                     {"event_type": "response.failed"}, {"event_type": "invented"}):
        foreign = {**initial, "steps": [{**step, **tampered}]}
        with pytest.raises(PermissionError, match="owned relay evidence"):
            gpu._record_scope(foreign, digest(foreign), anchor.seed, anchor.wave + 1)


def test_reviewed_activation_requires_exact_authority_and_original_immutable_record(tmp_path, owned_recovery_record):
    _, record = owned_recovery_record
    run = tmp_path / "fresh"
    approval = {"exact-approved-runtime": True}
    with pytest.raises(PermissionError, match="runtime authorization SHA"):
        gpu._activate_reviewed_record(run, record, approval, "f" * 64, resume=False)
    assert not run.exists()
    with pytest.raises(PermissionError, match="original immutable"):
        gpu._activate_reviewed_record(run, record, approval, digest(approval), resume=True)
    gpu._activate_reviewed_record(run, record, approval, digest(approval), resume=False)
    assert (run / gpu.REVIEWED_RECORD_FILE).is_file()
    with pytest.raises(PermissionError, match="already activated"):
        gpu._activate_reviewed_record(run, record, approval, digest(approval), resume=False)
    with pytest.raises(PermissionError, match="original immutable"):
        gpu._activate_reviewed_record(run, {**record, "round": 8}, approval, digest(approval), resume=True)
    gpu._activate_reviewed_record(run, record, approval, digest(approval), resume=True)



def test_reviewed_source_cannot_open_unrelated_overlay_or_change_frozen_launchers(
        tmp_path, monkeypatch, owned_recovery_record):
    anchor, record = owned_recovery_record
    scope = gpu._record_scope(record, digest(record), anchor.seed, anchor.wave + 1)
    provider = gpu.STUDY / "a7_gpu/provider.py"
    launcher = gpu.STUDY / "scripts/run_v13_retry_successor.py"
    monkeypatch.setattr(gpu, "SIGNED_RETRY_SOURCES", (launcher,))
    current = {provider: "new provider", launcher: "reviewed launcher"}
    for name, content in current.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    exact = {name.as_posix(): file_digest(tmp_path / name) for name in current}
    parent = SimpleNamespace(source_sha256={**exact, launcher.as_posix(): "b" * 64})
    scope = replace(scope, required_sources=())
    assert gpu._source_manifest(tmp_path, parent, scope) == exact
    with pytest.raises(PermissionError, match="outside the audited"):
        gpu._source_manifest(tmp_path, parent, replace(scope, previous_reviewed_record_sha256="c" * 64))
    parent.source_sha256[provider.as_posix()] = "a" * 64
    with pytest.raises(PermissionError, match="outside the audited"):
        gpu._source_manifest(tmp_path, parent, scope)


def test_reviewed_archive_rehashes_both_generations_and_denies_an_old_tamper(
        tmp_path, monkeypatch, owned_recovery_record):
    anchor, initial = owned_recovery_record
    original = initial["steps"][0]
    approval = {"approved": "first"}
    first = {**original, "parent_authorization_sha256": digest(approval)}
    second = {**first, "attempt_id": "0" * 31 + "2", "parent_contract_sha256": "a" * 64,
              "cooldown_unix": first["cooldown_unix"] + 301, "event_type": "response.output_item.done"}
    names = [retry_name(anchor.seed, wave) for wave in (anchor.wave, anchor.wave + 1)]
    release = tmp_path / "immutable"
    for index, (name, step) in enumerate(zip(names, (first, second), strict=True)):
        run = tmp_path / name
        directory = run / f"seed_{anchor.seed}"
        source = directory / f"round_{anchor.round:02d}" / "source.json"
        pause = directory / "pause.request"
        once(source, {"state": "same science"})
        once(pause, {"coordinator": index})
        once(run / "provider_model_lock.json", {"served_model": "gpt-6.2-astra"})
        step.update(source_file_sha256=file_digest(source), pause_file_sha256=file_digest(pause))
    record = {**initial, "steps": [first, second]}
    scope = gpu._record_scope(record, digest(record), anchor.seed, anchor.wave + 2)
    older, newer = scope.archive[-2:]
    assert newer.progress["event_type"] == "response.output_item.done"
    assert older.progress["event_type"] == "response.output_item.added"
    scope = replace(scope, archive=(older, newer))
    first_run, second_run = (tmp_path / name for name in names)
    model = {"served_model": "gpt-6.2-astra"}
    receipts = {first_run: [{"frozen": "prior"}], second_run: [{"frozen": "latest"}]}
    contracts = {
        first_run: SimpleNamespace(digest=older.identity.contract, predecessor={"model_identity": model}),
        second_run: SimpleNamespace(digest=newer.identity.contract, predecessor={
            "model_identity": model, "previous_contract_sha256": older.identity.contract,
            "previous_authorization_sha256": digest(approval), "previous_attempts": receipts[first_run],
            "previous_run_directory": str(first_run), "previous_release_directory": str(release),
        }),
    }

    def archived(run, frozen, identity):
        assert frozen == release and contracts[run].digest == identity.contract
        return contracts[run], approval

    def retired(run, scope, *, require_pause):
        directory = run / f"seed_{scope.seed}"
        return directory, directory / f"round_{scope.round:02d}"

    monkeypatch.setattr(gpu.retry.bounded, "_archived", archived)
    monkeypatch.setattr(gpu.retry, "_retired", retired)
    monkeypatch.setattr(gpu, "_evidence", lambda run, *_: ({"exact": "request"}, receipts[run], {}))
    monkeypatch.setattr(gpu.retry, "_compare_science", lambda current, previous, _: current == previous)
    assert len(gpu._archives(second_run, release, scope)) == 2
    old_pause = first_run / f"seed_{anchor.seed}" / "pause.request"
    old_pause.write_text("different signed predecessor")
    with pytest.raises(PermissionError, match="exact reviewed predecessor pause or source"):
        gpu._archives(second_run, release, scope)


def retry_name(seed, wave):
    return f"{gpu.retry.bounded.PREFIX}-{seed}-relay-retry-{wave:02d}"


def test_reviewed_start_lock_and_mac_boundary_gate_allow_one_after_cooldown(tmp_path, monkeypatch, owned_recovery_record):
    anchor, record = owned_recovery_record
    scope = gpu._record_scope(record, digest(record), anchor.seed, anchor.wave + 1)
    guard = tmp_path / "reviewed_recovery_activation.lock"
    approval = {"approved": True}
    with gpu.retry.scheduler._lock(guard, busy="duplicate reviewed startup"):
        gpu._activate_reviewed_record(tmp_path, record, approval, digest(approval), resume=False)
        with pytest.raises(TransportStop, match="duplicate reviewed startup"):
            with gpu.retry.scheduler._lock(guard, busy="duplicate reviewed startup"):
                pytest.fail("a competing startup cannot enter the activation window")
    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    checked = []
    monkeypatch.setattr(mac.RelayRetryDispatcher, "_check_boundary", lambda self, request, started: checked.append(started))
    attempt = {"attempt": 0, "attempt_id": "fresh"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.001)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary({}, attempt)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary({}, attempt)
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary({}, {**attempt, "attempt": 1})
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary({}, {**attempt, "attempt_id": scope.attempt_id})
    assert checked == [attempt]


def test_a_post_upload_header_timeout_never_authorizes_the_second_slot():
    identifier = "99bc486f-f525-4a96-8bbc-d77424cde734"
    observed = {"provider_request_id": identifier, "response_id": "resp_" + identifier.replace("-", ""),
                "provider_status": "failed", "served_model": "gpt-6.2-astra", "usage": {}}
    assert is_sealed_relay_failure({**observed, "error": "response_id_mismatch"})
    assert not is_sealed_relay_failure({**observed, "error": "response_headers_timeout"})


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
    unregistered_wave = max(wave for seed, wave in gpu.SCOPES if seed == 12203) + 1
    with pytest.raises(ValueError, match="no individually reviewed"):
        gpu.read_scope(Parser(), SimpleNamespace(seed=12203, wave=unregistered_wave))


@pytest.mark.parametrize("seed,wave,rev_counts,physical_budget", [
    (12203, 3, [1, 1], 187), (12203, 4, [1, 1, 1], 187), (12203, 5, [1, 1, 1, 1], 187),
    (12203, 6, [1, 1, 1, 1, 1], 187), (12203, 10, [2, 2, 2, 1], 181), (12203, 13, [2, 2, 2, 1, 1, 2, 2], 181),
    (12203, 15, [2, 2, 2, 1, 1, 2, 2, 1, 1], 181),
    (157, 3, [2, 1], 193), (157, 4, [2, 1, 1], 193), (157, 5, [2, 1, 1, 1], 193),
    (157, 6, [2, 1, 1, 1, 1], 193), (157, 7, [1], 187),
    (6101, 2, [2], 199), (6101, 3, [2, 1], 199), (6101, 5, [2, 1, 1, 1], 199), (2357, 2, [2], 187),
    (6101, 12, [2, 2, 1], 163),
    (2357, 5, [2, 1, 2, 1], 187), (2357, 7, [2, 1, 2, 1, 1, 1], 187),
    (887, 1, [1], 163), (887, 3, [1, 1], 163), (887, 4, [1, 1, 1], 163), (887, 5, [1], 139), (887, 7, [1, 1, 1], 139),
    (887, 11, [1], 19),
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


def test_6101_wave07_allows_one_round_three_attempt_after_the_exact_no_post_fit():
    scope = gpu.SCOPES[(6101, 7)]
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (2, 3, 4)
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 33, "maximum_logical_requests": 97, "maximum_physical_attempts": 193}
    assert scope.not_before_unix == 1795700566.0270398 and scope.cooldown_field == "failed_unix"
    assert scope.cached_origin.ancestor.name.endswith("-6101-relay-retry-06")
    assert scope.cached_origin.ancestor.contract == "3380abf0b3fbde69a4d4c7a9d8e64dc815cc64c258736088baa3041fff209023"
    assert scope.cached_origin.review["provider_post"] is False
    assert scope.cached_origin.receipt_file_sha256 == "55c1aaa458c9e105ce9d9b94ff7434ccfb347b69ea1b96a335175ecec713b244"
    assert dict(scope.fitted_prefix_files)["round_03/transition.json"] == (
        "8e5f9d4cc67593e008734a4ba72194970560a5d7d495ee4c507299391f2d9996"
    )
    assert scope.required_sources == gpu.SCOPES[(12203, 12)].required_sources
    assert all(file_digest(gpu.ROOT / name) == sha for name, sha in scope.required_sources)
    old, = scope.archive
    assert old.identity.name.endswith("-6101-cached-r2-01")
    assert old.attempt_id == "290284ba893c42f79a67a9b5bf88159b" and old.failure == "timeout"
    assert old.progress["error_type"] == "sse_inactivity_timeout" and old.progress["sse_event_count"] == 1092
    assert old.progress["event_type"] == "response.output_text.delta" and old.replay_files is None


def test_6101_wave08_preserves_the_cached_fit_and_the_hard_wall_failure():
    previous = gpu.SCOPES[(6101, 7)]
    scope = gpu.SCOPES[(6101, 8)]
    assert scope.round == scope.replay_start_round == 3
    assert scope.payload_sha256 == previous.payload_sha256
    assert scope.cached_origin == previous.cached_origin
    assert scope.archive[:-1] == previous.archive
    assert scope.not_before_unix == 1795704845.0379763
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == {"first_logical_requests": 33, "maximum_logical_requests": 97, "maximum_physical_attempts": 194}
    last = scope.archive[-1]
    assert last.attempt_id == "e674f38eefbb4e9fa50824f61f932400" and last.failure == "timeout"
    assert last.progress["error_type"] == "hard_wall_deadline" and last.progress["sse_event_count"] == 648


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
    (2357, 6, 1795698146.9968047), (2357, 11, 1795711758.937361), (2357, 12, 1795713361.660755),
    (6101, 6, 1795693173.4806566),
    (12203, 8, 1795690919.9466105), (12203, 9, 1795693155.7197583), (12203, 11, 1795697031.8253946),
    (887, 6, 1795701974.7327738), (887, 8, 1795709072.5213063),
    (12203, 12, 1795699275.9433548), (12203, 14, 1795702470.7997937), (157, 8, 1795696780.5165217),
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
    assert_prior_transport_atoms(scope)


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
    assert_prior_transport_atoms(scope)


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
    assert_prior_transport_atoms(scope)


def test_12203_wave12_keeps_both_exact_alias_failures_and_all_replay_links():
    previous = gpu.SCOPES[(12203, 11)]
    scope = gpu.SCOPES[(12203, 12)]
    assert scope.payload_sha256 == previous.payload_sha256
    assert scope.original_start_round == scope.round == previous.round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.archive[:-1] == previous.archive
    assert [len(review.attempts) for review in scope.archive] == [2, 2, 2, 1, 1, 2]
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182}
    assert scope.cooldown_field == "failed_unix" and scope.not_before_unix == 1795699275.9433548
    assert scope.replay_start_round == 5
    first, last = scope.archive[-1].attempts
    assert first.identity == last.identity and first.request_id == last.request_id
    assert first.terminal_file_sha256 == "903b43ba21fbaabc38bbb313d6d902975dc05737b100be0102f0d0fbe2699fa2"
    assert last.terminal_file_sha256 == "6ff9d185708909d691c4986010b25d0b8188aa096dfa21fb95ae78ddac08a874"
    for attempt in (first, last):
        assert attempt.failure == "response_id_mismatch"
        assert attempt.progress["conflicting_event_type"] == "response.failed"
        assert attempt.progress["response_id_matches_transport_alias"] is True and attempt.progress["sse_event_count"] == 25
    assert last.replay_files == {
        "attempt_00.replay_authorization.json": "0b19a39b3fe1948be32132803058b98d417a21420ae7ac08a3978dba61fa3517",
        "attempt_00.replay.terminal.json": "ba9dda32d3025c6434533d53a1a3de0b4ecf978758fe4c6fd6cfbf84e65bc3a2",
        "attempt_01.replay_link.json": "603fb01451eca42bee560998d456bddf67d1e50a0004dcf609ea31ad33b6fd29",
    }
    previous_sources = dict(previous.required_sources)
    assert previous_sources[gpu.PROVIDER_SOURCE] == "9b77e82920a303b9429957372d0943a1c729fecca3c80c00d86e0411a1c74bd0"
    previous_sources[gpu.PROVIDER_SOURCE] = "aacbc199e896091d4395fc036cb60bc604d91ec326a79fcff26e157d9a73705f"
    assert dict(scope.required_sources) == previous_sources
    assert all(file_digest(gpu.ROOT / name) == sha for name, sha in scope.required_sources)


def test_12203_wave13_binds_one_new_slot_to_the_exact_w12_pre_post_failure():
    previous = gpu.SCOPES[(12203, 12)]
    scope = gpu.SCOPES[(12203, 13)]
    assert scope.archive[:-1] == previous.archive
    assert scope.payload_sha256 == previous.payload_sha256 and scope.round == previous.round == 5
    assert scope.original_start_round == 5 and scope.original_boundary_physical_attempts == 2
    assert scope.replay_start_round == 6 and scope.not_before_unix == 1795701456.67948
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.required_sources == previous.required_sources
    first, last = scope.archive[-1].attempts
    assert first.progress["provider_error_code"] == "upstream_error"
    assert last.failure == "connect_error" and last.progress["error_type"] == "connect_timeout_before_post"
    assert last.terminal_file_sha256 == "19a85316433270504a4c3c5916971129fe37adca81709c67d1274ee432b7307a"
    assert last.pre_post_connect_failure is True and first.pre_post_connect_failure is False
    assert replace(last, attempt_id="a" * 32).pre_post_connect_failure is False
    assert replace(last, progress_file_sha256="a" * 64).pre_post_connect_failure is False
    assert all(not attempt.pre_post_connect_failure for wave in previous.archive for attempt in wave.attempts)


def test_12203_wave14_keeps_the_exact_scientific_input_after_w13_upstream_error():
    previous = gpu.SCOPES[(12203, 13)]
    scope = gpu.SCOPES[(12203, 14)]
    assert scope.archive[:-1] == previous.archive
    assert scope.payload_sha256 == previous.payload_sha256
    assert scope.original_start_round == scope.round == scope.replay_start_round == previous.round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795702470.7997937
    assert scope.required_sources == previous.required_sources
    old = scope.archive[-1]
    assert old.failure == "response_id_mismatch" and old.replay_files is None
    assert old.terminal_file_sha256 == "f5254ed9b334038058004aa91a1861b99b91d59ee25d2790890bbd3ac6b5a0be"
    assert old.progress["provider_error_code"] == "upstream_error" and old.progress["sse_event_count"] == 25
    assert old.progress["conflicting_event_type"] == "response.failed" and old.progress["response_id_matches_transport_alias"]
    assert not old.pre_post_connect_failure and scope.archive[-2].attempts[-1].pre_post_connect_failure


def test_12203_wave15_carries_only_the_unused_slot_after_the_sealed_idle_timeout():
    previous = gpu.SCOPES[(12203, 14)]
    scope = gpu.SCOPES[(12203, 15)]
    assert scope.archive[:-1] == previous.archive and scope.payload_sha256 == previous.payload_sha256
    assert scope.original_start_round == scope.round == previous.round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.boundary_physical_attempts + len(scope.archive[-1].attempts) == previous.boundary_physical_attempts
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 181}
    assert scope.required_sources == previous.required_sources
    assert scope.cooldown_field == "failed_unix" and scope.not_before_unix == 1795706115.3553925
    assert scope.replay_start_round == 6
    old = scope.archive[-1]
    assert old.replay_files is None and old.failure == "timeout"
    assert old.terminal_file_sha256 == "70fc3592a2dd242f1fa7d5a105ccd4ff965a181c2badc379a27bf5863d4e2f6e"
    assert old.progress["error_type"] == "sse_inactivity_timeout" and old.progress["sse_event_count"] == 15
    assert old.progress["event_type"] == "response.output_item.added"


def test_12203_wave16_preserves_the_exhausted_single_slot_and_nine_earlier_reviewed_waves(monkeypatch):
    previous = gpu.SCOPES[(12203, 15)]
    scope = gpu.SCOPES[(12203, 16)]
    assert scope.archive[:-1] == previous.archive and scope.payload_sha256 == previous.payload_sha256
    assert tuple(len(review.attempts) for review in scope.archive) == (2, 2, 2, 1, 1, 2, 2, 1, 1, 1)
    assert scope.original_start_round == scope.round == scope.replay_start_round == previous.round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert previous.budget["maximum_physical_attempts"] == 181
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.required_sources == previous.required_sources and scope.fitted_prefix_files == ()
    old = scope.archive[-1]
    assert old.identity.name.endswith("12203-relay-retry-15")
    assert old.attempt_id == "4fdcc2c2a5b9469eb0e0eb0124b7ad0e" and old.failure == "timeout"
    assert old.replay_files is None and not old.pre_post_connect_failure
    assert old.progress["http_status"] == 200 and old.progress["error_type"] == "sse_inactivity_timeout"
    assert old.progress["sse_event_count"] == 8 and old.progress["event_type"] == "response.output_item.done"
    assert old.terminal_file_sha256 == "283c19e27fd67795e3a53aec8f012028aca447615fb2276d7735047727fffa74"
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795707966.662386)

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    accepted = []
    monkeypatch.setattr(mac.RelayRetryDispatcher, "_check_boundary",
                        lambda _, request, reservation: accepted.append((request, reservation)))
    request = {"payload_sha256": scope.payload_sha256}
    new = {"attempt": 0, "attempt_id": "c" * 32}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - .01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, new)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    for reviewed in scope.archive:
        for attempt in reviewed.attempts:
            with pytest.raises(PermissionError, match="signed physical attempts"):
                dispatcher._check_boundary(request, {**new, "attempt_id": attempt.attempt_id})
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**new, "attempt": 2})
    dispatcher._check_boundary(request, new)
    assert accepted == [(request, new)]


def test_12203_wave17_carries_only_the_unused_post_slot_without_reusing_an_ambiguous_attempt(monkeypatch):
    previous = gpu.SCOPES[(12203, 16)]
    scope = gpu.SCOPES[(12203, 17)]
    old = scope.archive[-1]
    assert scope.archive[:-1] == previous.archive and scope.payload_sha256 == previous.payload_sha256
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (5, 5, 6)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.boundary_physical_attempts + len(old.attempts) == 2
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 181}
    assert scope.required_sources == previous.required_sources and scope.fitted_prefix_files == ()
    assert old.identity.name.endswith("12203-relay-retry-16")
    assert old.attempt_id == "2b0e526441584843bac2a9a185a5f4fe" and old.failure == "connection_interrupted"
    assert old.progress["error_type"] == "ReadError" and old.progress["http_status"] is None
    assert old.progress["sse_event_count"] is None and not old.pre_post_connect_failure and old.replay_files is None
    assert old.terminal_file_sha256 == "d408a77082c3af6bdca7dd6d253a2a263d280e9b8257f12d15f7d024e6192ad0"
    assert (scope.cooldown_field, scope.not_before_unix) == ("failed_unix", 1795709911.0657685)

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    admitted = []
    monkeypatch.setattr(mac.RelayRetryDispatcher, "_check_boundary",
                        lambda _, request, reservation: admitted.append((request, reservation)))
    request = {"payload_sha256": scope.payload_sha256}
    reservation = {"attempt": 0, "attempt_id": "e" * 32}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - .01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, reservation)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**reservation, "attempt_id": old.attempt_id})
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**reservation, "attempt": 1})
    dispatcher._check_boundary(request, reservation)
    assert admitted == [(request, reservation)]


def test_12203_wave18_requires_exact_old_alias_and_delivered_failure_guard_for_the_second_slot(monkeypatch):
    previous = gpu.SCOPES[(12203, 17)]
    scope = gpu.SCOPES[(12203, 18)]
    old = scope.archive[-1]
    assert scope.archive[:-1] == previous.archive and scope.payload_sha256 == previous.payload_sha256
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (5, 5, 5)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == len(old.attempts) == 1
    assert previous.budget["maximum_physical_attempts"] == 181
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.required_sources == previous.required_sources and scope.fitted_prefix_files == ()
    assert old.identity.name.endswith("12203-relay-retry-17")
    assert old.attempt_id == "3ef3bfee813c47acb717ba512634e87a" and old.failure == "response_id_mismatch"
    assert old.progress["provider_error_code"] == "upstream_error" and old.progress["sse_event_count"] == 25
    assert old.progress["conflicting_event_type"] == "response.failed" and old.progress["response_id_matches_transport_alias"]
    assert old.terminal_file_sha256 == "22ed13565ffdbaf4138e5d320e7f00f47439c3b255fa82ba4366f80a8896d289"
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795711982.841558)

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": old.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": old.request_id}
    checked = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry", lambda request, reservation: checked.append((request, reservation)))
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first = {"attempt": 0, "attempt_id": "a" * 32}
    second = {"attempt": 1, "attempt_id": "b" * 32}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - .01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**first, "attempt_id": old.attempt_id})
    dispatcher._check_boundary(request, first)
    assert not checked
    dispatcher._check_boundary(request, second)
    assert checked == [(request, second)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second, "attempt": 2})


@pytest.mark.parametrize("wave,eligible_unix,attempt_ids,terminal_shas,counts", [
    (31, 1795731691.2409678,
     ["3eb361a886824203a8eec5c54183b057", "b44af33229c94181a6f77130b68a42f0"],
     ["ba8c0852253f1e24e75bf15e08ac64288cf0e6590683836262c28f1674b314c8",
      "06a15c892c12fcbc9a2ab12b1aa01864e67f6b63975d5c324959f5da7a7552ba"], [41, 43]),
    (32, 1795732690.953343,
     ["852f0e2578c94c7a86bd715e199da17e", "4d9dc0c85cd24f71a191f0aaa0c6bacd"],
     ["57340e9d29010e856935b860b25261c7f87b15e525ad48dc3b0b765fd9b8637d",
      "39a81cc37b7911b7c04436ec5b4317786cf9500b119248c70f49d95b9eb35477"], [41, 41]),
])
def test_12203_preserves_both_owned_failures_and_p1_guard(
        monkeypatch, wave, eligible_unix, attempt_ids, terminal_shas, counts):
    previous = gpu.SCOPES[(12203, wave - 1)]
    scope = gpu.SCOPES[(12203, wave)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == (43 if wave == 31 else 45)
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", eligible_unix)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == attempt_ids
    assert [item.terminal_file_sha256 for item in (first, second)] == terminal_shas
    assert [item.progress["sse_event_count"] for item in (first, second)] == counts
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


@pytest.mark.parametrize("wave,total_archived,eligible_unix,attempt_ids,terminal_shas", [
    (35, 51, 1795735786.489421,
     ["881d112f46624bc49634c618350bdd0a", "a081770beebe417eabbe3079894a7991"],
     ["18abe2c55fb55d79e975f84fbf230f4b7040e2a6f0e188af8fee18e8cd4d33d1",
      "260d73d6121c87078a82dcc17424bc3b3548f9693b8f4f8f186b32474c61e4ab"]),
    (36, 53, 1795736915.1458318,
     ["c6395fd2ee1c4a099302f29cc33336bb", "ca104adc27b5428e87a33cce8ea78461"],
     ["43da785d03d8c5db5b83530c37616bd5e4b871b6d99e140e026348e738f09710",
      "05684bb58387335cbdb8ea438999ee8e441af3715812353134854524aad93c9c"]),
    (37, 55, 1795738859.9096358,
     ["8dd2bf5e7b534fe98d0e503bbe3a6aa9", "34d242e3acfd409694cb8b2ea726207b"],
     ["38d916218e10819c6c3d04103670866ababf2cf9c13c9830decdd27e152b6521",
      "1e03f1f66ef520b70c110e3bebde63c3bbbb569420f2e1a7089f720427e1092a"]),
    (38, 57, 1795740125.4333892,
     ["2c4ca26739bf403596f30160d71a2505", "9b912e715e87450496209abde65620c6"],
     ["94c2edb2ffacf88072b57655e9d7da5164d140b74932fc5fa899e43525539f34",
      "ca4e768c5132a659dc82eeba90beadfaf53665a9ce99d264402387b82570248c"]),
    (39, 59, 1795741459.7049484,
     ["c958c81f4b72442ca53ae1db01844d35", "fc2fbfeeafed46efa2950bf56d4ee748"],
     ["3f93f45109b384447886578dfd0fba1d54c263214b2d90959023c08de5deaaae",
      "61892e628880daad55719ffac6e5d932dd130c56b397d2f25efb3c73378dda63"]),
    (40, 61, 1795742501.963246,
     ["b4e28ce0bf7c4a0aaed2f58ebe209365", "910bb184c09847319f26780c0f13448c"],
     ["c076291c8bca9206692dc99e00e94136564e76532525b80a81d0d1e4056f3f22",
      "610ae07b66d4928c0967245b84932fc493fb9c69f55bb2025155d49320fe48fb"]),
    (41, 63, 1795743604.731306,
     ["1a7848a101724b8db4094c633dbdf00e", "9e7c53ff29514ab59fdc74d7b5f6ad22"],
     ["69bea35032b36ceb7299da6ed5e669caf2831aa350ff9be04755e7a0b93e66ed",
      "abc12cf5eed6b39dd9c224ca3f114d69215ff44de3a6fb91c055cb8fb03cb9d5"]),
    (42, 65, 1795744853.496817,
     ["dc480f90961d49aeb8ae0a179ed5afe7", "75dbecfd5c6f47ff95e19c7770d6fc3a"],
     ["0b5cfb91cbd0e80534cbed8ed8c0c99e34934c880b3538a2c4594eca9ba6579c",
      "fee6b5582bfd21b253514412719d89aae40b45c7a45df5185424386cdf186a96"]),
    (43, 67, 1795746019.1576593,
     ["c9071ea7ea6e4f409973d51b82615a52", "09811f9951de4c5da094ee9a0d5ac4f8"],
     ["f7cea80b169d02c004821670461ca1fffaa5893abce027af5442ec4bb1fc081b",
      "1f9107f8ce53ff698737cd89f441659a27e2b0f2046d3155d4bcdc50171f3a83"]),
    (44, 69, 1795747675.7502,
     ["f2051c29991c4408a3dc18805db95ef1", "69db88b2f0004b4ba2dfd8d2eb55e751"],
     ["b261f2a18cbf4b8401d49e4dc2e09b28c4e5113e18e5254f71f471fb266b020a",
      "88d1d3c64da619d827c34bac81f1107a00cc0b930a41f5fa1cec2df20299c81f"]),
])
def test_12203_preserves_both_latest_owned_failures_and_p1_guard(monkeypatch, wave, total_archived, eligible_unix,
                                                               attempt_ids, terminal_shas):
    previous = gpu.SCOPES[(12203, wave - 1)]
    scope = gpu.SCOPES[(12203, wave)]
    if wave == 38:
        assert scope.archive[-1].replay_files == {
            "attempt_00.replay_authorization.json": "a8d0a83c47d200e45d72a7c451c6e131b440c6db3da704d8cff646ebbb0fb0a7",
            "attempt_00.replay.terminal.json": "ecef5f421bbb045a34d71ad828b7aa67667da99278f6dd538a579b4918539046",
            "attempt_01.replay_link.json": "6a524dc25dc2447b1d2ca76bbcc7cc2df16caac9b267a5fd9d4c2c5548835025",
        }
    if wave == 39:
        assert scope.archive[-1].replay_files == {
            "attempt_00.replay_authorization.json": "c4bfacb425d64fdd9d51309f863cae1e024ee83a4df95073d75ad9e1df357ab4",
            "attempt_00.replay.terminal.json": "0a98bcdc30d4d4cb3853fa9204d55588e3878a18d293919ec4aa01f7ecccc60f",
            "attempt_01.replay_link.json": "8649f8f23140751caab5ef3c8c022403c24e4ec56b81da6fef70d5e4e108901d",
        }
    if wave == 40:
        assert scope.archive[-1].replay_files == {
            "attempt_00.replay_authorization.json": "bd5c8fccda285ee6b3bd455009a154ab4016d1c99c26c4449e602388905cdb1e",
            "attempt_00.replay.terminal.json": "7d4500cd81660e0b0aae23c8231fa100d0dcd58d6fc37ee26037ab42ccb2c10c",
            "attempt_01.replay_link.json": "e69531bc254d32c84ff7759183ca693ca3ff7e13dc39a4dcb51d203a4d49ddd8",
        }
    if wave == 41:
        assert scope.archive[-1].replay_files == {
            "attempt_00.replay_authorization.json": "6524e19de35bab01fb44ff9f90ec295b4d43673e97099554e69e449991d6f1c9",
            "attempt_00.replay.terminal.json": "69399361b218b4fff127515343a4478866993e084318f5461e1bcd009dc0d4bb",
            "attempt_01.replay_link.json": "066a6dc2af66290307ecfb417c0486ad4c45b82ea8c5973f689c43c7819bedf8",
        }
    if wave == 42:
        assert scope.archive[-1].replay_files == {
            "attempt_00.replay_authorization.json": "f945215fd633dc2092cbd2883772760928100a390eb555672c84db5ce87c105c",
            "attempt_00.replay.terminal.json": "f4678b63c42725600e4251434ea306e020c73411dcda80964eb00b6da353a8f2",
            "attempt_01.replay_link.json": "70e188e76b25f62945cf3ce42810c4e828948f0de7f153f24c621851d812a0e6",
        }
    if wave == 43:
        assert scope.archive[-1].replay_files == {
            "attempt_00.replay_authorization.json": "b9d20f7beb3348dfca197a296432a361b57cb948b1aaaf7195dc7cbf91841673",
            "attempt_00.replay.terminal.json": "125366a5dbed96afff3d1752154b8387d4217da18ca4c4a1de898dfc5979f3f8",
            "attempt_01.replay_link.json": "acb8477fa28f02f2a875cbca77b6af58f8ac42931d82302c12d86b7d226c00b0",
        }
    if wave == 44:
        assert scope.archive[-1].replay_files == {
            "attempt_00.replay_authorization.json": "9c5a673d9045220756e67385331b09dbb325b53010a6495d86818138103a6af0",
            "attempt_00.replay.terminal.json": "f8363d0bb201acf0ab628e2d847ce6ef3aeb77f49763aa340772148cf908ae31",
            "attempt_01.replay_link.json": "5f18bb0953c1c78213f257ca6e4e5a0b4653331949fef4620e8e1d684bbacd81",
        }
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == total_archived
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", eligible_unix)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == attempt_ids
    assert [item.terminal_file_sha256 for item in (first, second)] == terminal_shas
    expected_events = {39: [41, 25], 40: [25, 41], 41: [41, 25], 42: [41, 23], 43: [25, 15], 44: [19, 25]}.get(wave, [41, 41])
    assert [item.progress["sse_event_count"] for item in (first, second)] == expected_events
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_12203_wave52_preserves_wave51_failed_terminal_and_one_current_slot(monkeypatch):
    previous = gpu.SCOPES[(12203, 51)]
    scope = gpu.SCOPES[(12203, 52)]
    assert scope.archive[:-1] == previous.archive
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (5, 5, 6)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 181}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795756542.3095212)
    fitted = dict(scope.fitted_prefix_files)
    assert fitted["pause.request"] == "82ed4765593fb09095e73302f3e97d9dcf2edd41ac219aba403cbf0a44757088"
    assert fitted["round_05/turn_00/request.json"] == "c64156c33a9eeb36b1eafa03dcf9a3ad9cfaa9fd820d5120167b6b426117d5f5"
    review, = scope.archive[-1].attempts
    assert review.terminal_file_sha256 == "cafeaef6d7a9f255370c07cec2a2a3af70c3a5f30f24353dfa2a7f9abbf611f3"
    assert review.failure == "response_id_mismatch" and review.receipt_provider_flags == (True, True)
    assert review.progress["http_status"] == 200 and review.progress["sse_event_count"] == 27
    assert review.progress["provider_error_code"] == "upstream_error"
    assert review.progress["conflicting_event_type"] == "response.failed"
    assert review.progress["response_id_matches_transport_alias"] is True

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": review.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": review.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    started = {"attempt": 0, "attempt_id": "new-first"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, started)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, started)
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**started, "attempt": 1})


def test_12203_wave51_preserves_wave50_failed_terminal_and_one_current_slot(monkeypatch):
    previous = gpu.SCOPES[(12203, 50)]
    scope = gpu.SCOPES[(12203, 51)]
    assert scope.archive[:-1] == previous.archive
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (5, 5, 6)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 181}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795755559.6357415)
    fitted = dict(scope.fitted_prefix_files)
    assert fitted["pause.request"] == "b01c993bb2c4ba869264bb2228e87c7f79e1c1609c21f9723da3dd126b9b89a0"
    assert fitted["round_05/turn_00/request.json"] == "d13de70641ce7304b5cf570bf001cacf266bf6d06a5afbc3dccf6f1757b5131d"
    review, = scope.archive[-1].attempts
    assert review.terminal_file_sha256 == "a1386472328f91b09c25ff6edd83bc9f14e6e951ab8af04c67ab1e30acc03875"
    assert review.failure == "response_id_mismatch" and review.receipt_provider_flags == (True, True)
    assert review.progress["http_status"] == 200 and review.progress["sse_event_count"] == 41
    assert review.progress["provider_error_code"] == "upstream_error"
    assert review.progress["conflicting_event_type"] == "response.failed"
    assert review.progress["response_id_matches_transport_alias"] is True

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": review.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": review.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    started = {"attempt": 0, "attempt_id": "new-first"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, started)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, started)
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**started, "attempt": 1})


def test_12203_wave50_preserves_wave49_failed_terminal_and_one_current_slot(monkeypatch):
    previous = gpu.SCOPES[(12203, 49)]
    scope = gpu.SCOPES[(12203, 50)]
    assert scope.archive[:-1] == previous.archive
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (5, 5, 6)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 181}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795754821.0570178)
    fitted = dict(scope.fitted_prefix_files)
    assert fitted["pause.request"] == "2afc415a0c448367b4c90a5e3d343ae521a403c191f6432dc550ee92c1a17053"
    assert fitted["round_05/turn_00/request.json"] == "36d66dd0c4e677f73780183ae2004f26092a54a1697c64d9c7461d34b5bb968d"
    review, = scope.archive[-1].attempts
    assert review.terminal_file_sha256 == "8b0ced6595cd8cf07ad61687b516d1bdcb0cb4814a4a7b82ef5475cd700385bf"
    assert review.failure == "response_id_mismatch" and review.receipt_provider_flags == (True, True)
    assert review.progress["http_status"] == 200 and review.progress["sse_event_count"] == 41
    assert review.progress["provider_error_code"] == "upstream_error"
    assert review.progress["conflicting_event_type"] == "response.failed"
    assert review.progress["response_id_matches_transport_alias"] is True

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": review.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": review.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    started = {"attempt": 0, "attempt_id": "new-first"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, started)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, started)
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**started, "attempt": 1})


def test_12203_wave49_preserves_wave48_failed_terminal_and_one_current_slot(monkeypatch):
    previous = gpu.SCOPES[(12203, 48)]
    scope = gpu.SCOPES[(12203, 49)]
    assert scope.archive[:-1] == previous.archive
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (5, 5, 6)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 181}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795753994.9759812)
    fitted = dict(scope.fitted_prefix_files)
    assert fitted["pause.request"] == "5fcff6800488ad257974a5b2fd85708f493815aaca943330e0a305791230b26b"
    assert fitted["round_05/turn_00/request.json"] == "8e3866deb162b567a3259b27c2a387c90e7ade254368843c83157b1b9bfbf6ba"
    review, = scope.archive[-1].attempts
    assert review.terminal_file_sha256 == "4ce29a2d860e7dffc8e6317ae8b90ec1adf5b118f19dca3d23a27e84d72201b6"
    assert review.failure == "response_id_mismatch" and review.receipt_provider_flags == (True, True)
    assert review.progress["http_status"] == 200 and review.progress["sse_event_count"] == 41
    assert review.progress["provider_error_code"] == "upstream_error"
    assert review.progress["conflicting_event_type"] == "response.failed"
    assert review.progress["response_id_matches_transport_alias"] is True

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": review.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": review.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    started = {"attempt": 0, "attempt_id": "new-first"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, started)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, started)
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**started, "attempt": 1})


def test_12203_wave48_records_headers_without_sse_and_caps_new_boundary_slot(monkeypatch):
    previous = gpu.SCOPES[(12203, 47)]
    scope = gpu.SCOPES[(12203, 48)]
    assert scope.archive[:-1] == previous.archive
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (5, 5, 6)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 181}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795753327.3604734)
    fitted = dict(scope.fitted_prefix_files)
    assert fitted["pause.request"] == "ace8a9c06942d9e83f20c6f0f4d365ae75e9a57a7df45f626dd0e73ce9c26c08"
    assert fitted["round_05/turn_00/request.json"] == "c8ec0dd56488908930d6d65cd6d9b2e1702efc8cad377be92b7a7001cd5a92a5"
    review, = scope.archive[-1].attempts
    assert review.terminal_file_sha256 == "38e62cf8b7ca59d67a8c7fd8ec287ee3032d8cd5773b611141f15826b3bebb5d"
    assert review.failure == "timeout" and review.receipt_provider_flags == (True, True)
    assert review.progress["http_status"] == 200
    assert review.progress["error_type"] == "sse_inactivity_timeout"
    assert review.progress["network_phases"]["send_body"]["state"] == "complete"
    assert review.progress["network_phases"]["receive_headers"]["state"] == "complete"
    assert "response_id" not in review.progress and "sse_event_count" not in review.progress

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": review.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": review.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    started = {"attempt": 0, "attempt_id": "new-first"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, started)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, started)
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**started, "attempt": 1})


def test_12203_wave47_pins_fenced_wave46_and_authorizes_only_one_new_slot(monkeypatch):
    previous = gpu.SCOPES[(12203, 46)]
    scope = gpu.SCOPES[(12203, 47)]
    assert scope.archive[:-1] == previous.archive
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (5, 5, 6)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 181}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795751873.4475775)
    fitted = dict(scope.fitted_prefix_files)
    assert fitted["pause.request"] == "93e11d1075a9262d5d4db9a2896b4a51580621b6bbef118026f9c9ebe305a656"
    assert fitted["round_05/turn_00/request.json"] == "45b5f98bcad8dec075a1a72cf9e5f996ad9483189642ffc841301115ece5f3dc"
    review, = scope.archive[-1].attempts
    assert review.terminal_file_sha256 == "eea008dd5e061d7a4601920b0f3edef1a0eb9f720f89497b8a31f88d43c2acb8"
    assert review.failure == "response_id_mismatch" and review.receipt_provider_flags == (True, True)
    assert review.progress["http_status"] == 200 and review.progress["sse_event_count"] == 41
    assert review.progress["provider_error_code"] == "upstream_error"
    assert review.progress["conflicting_event_type"] == "response.failed"
    assert review.progress["response_id_matches_transport_alias"] is True

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": review.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": review.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    started = {"attempt": 0, "attempt_id": "new-first"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, started)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, started)
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**started, "attempt": 1})


def test_12203_wave46_replaces_only_the_unposted_slot_after_three_anonymous_checks(monkeypatch):
    previous = gpu.SCOPES[(12203, 45)]
    scope = gpu.SCOPES[(12203, 46)]
    assert scope.archive[:-1] == previous.archive
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (5, 5, 6)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 181}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795750879.3874757)
    review, = scope.archive[-1].attempts
    assert review.terminal_file_sha256 == "2e541c0b595b7016bbc9f59d4b7dbcc8b8496d198f41af59d5f427a684378e5e"
    assert review.failure == "connect_error" and review.receipt_provider_flags == (False, False)
    assert review.progress["error_type"] == "direct_ipv4_preflight_failed"
    assert review.progress["event_type"] == "ipv4_preflight_no_headers"
    assert tuple(name for name, _ in review.unposted_readiness_files) == (
        "readiness_01.json", "readiness_02.json", "readiness_03.json",
    )
    assert not gpu.replace(review, progress={**review.progress, "http_status": 200}).unposted_readiness_files
    assert not {"http_status", "sse_event_count", "provider_request_id", "response_id"} & review.progress.keys()
    assert not is_sealed_relay_failure({"error": review.failure})

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": review.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": review.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    started = {"attempt": 0, "attempt_id": "new-first"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, started)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, started)
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**started, "attempt": 1})


def test_12203_wave45_carries_only_the_unused_slot_after_idle_timeout(monkeypatch):
    previous = gpu.SCOPES[(12203, 44)]
    scope = gpu.SCOPES[(12203, 45)]
    assert scope.archive[:-1] == previous.archive
    review, = scope.archive[-1].attempts
    assert sum(len(item.attempts) for item in scope.archive) == 70
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (5, 5, 6)
    assert scope.boundary_physical_attempts + len(scope.archive[-1].attempts) == scope.original_boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == previous.header_deadline_profile == {}
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 181}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795750123.8644154)
    assert review.attempt_id == "60a4dc32674440019b48d753ac95d1e4"
    assert review.terminal_file_sha256 == "b6dc9b660099d9e603c2bcd512f7e5e63ecf73d240d0a318bc1b144abdb85db3"
    assert review.failure == "timeout" and not review.pre_post_connect_failure
    assert scope.archive[-1].replay_files is None
    progress = review.progress
    assert progress["http_status"] == 200 and progress["sse_event_count"] == 14
    assert progress["error_type"] == "sse_inactivity_timeout"
    assert 900 <= progress["elapsed_seconds"] - progress["last_sse_activity_seconds"] < 901
    assert not {"first_text_seconds", "complete_terminal_captured"}.intersection(progress)
    assert not is_sealed_relay_failure({"error": review.failure, "provider_request_id": progress["provider_request_id"]})

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": review.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": review.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first = {"attempt": 0, "attempt_id": "new-first"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first)
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**first, "attempt": 1})


def test_12203_wave34_preserves_both_owned_w33_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(12203, 33)]
    scope = gpu.SCOPES[(12203, 34)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 49
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795734737.44935)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["cac2a722c57549998cc42a219ce95288", "3c10b36f4fee4587aeb085be08211093"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "6369b7e46c0eb01cdcf5cb39fbc27a7ac3856505ca1bd199de01deaf37c21249",
        "520997df5973966ac63e1cf9a95417e6117aa1befb46591f769615eb1de8d518",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_12203_wave33_preserves_both_owned_w32_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(12203, 32)]
    scope = gpu.SCOPES[(12203, 33)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 47
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795733761.392024)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["51ccc1d35fb34c3c92688dfc4419ebb0", "a84c0a6e333241308f1e6b7488d1ce8c"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "c143de190893e4a0da5b9ba7440364bd89be5c6afc1e7a6dbd48bfa2dc42975d",
        "b9c2303c537b73b6bd7c3ba7aacc1b68394059e1070be7c0c6085ec50da6aa57",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_12203_wave30_preserves_both_owned_w29_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(12203, 29)]
    scope = gpu.SCOPES[(12203, 30)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 41
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795730733.2632422)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["420185dd70764c478a8ca7bd2b202726", "75901ef837ec4280aa8ba975239f2066"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "a74dda302023589875dbdfa30e8e83aa3d16bd8f02313e21d239fd3f51c45947",
        "c77bb0c5102f86a3f4efecf0f24e3fa94bf47188d91915108e4483a537343279",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [25, 25]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_12203_wave29_preserves_both_owned_w28_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(12203, 28)]
    scope = gpu.SCOPES[(12203, 29)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 39
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795729614.5429068)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["28549f09fb7243bc8938f9fa3437e1e4", "dd515d6606684aa3bd332499ceaf456e"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "ab736073f1f8eec5fdde734e2acd9eea470703b75ac25b74ba1b50a1db5c7099",
        "76a55f0d9304bb5465cfb70b50f4a6b22e7e0256533fc6c4290a8235bdef4ae5",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [25, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_12203_wave28_preserves_both_owned_w27_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(12203, 27)]
    scope = gpu.SCOPES[(12203, 28)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 37
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795728573.4999516)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["a2f3f4d289f348b287cc243e2735d113", "8c5b0a47fdc840b3977f30d25dbcb061"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "d106da71ec687bc7a581eee9ebe129fdacbe6dd955222639bf7069503f5128f3",
        "ed4e43d6f8253eb7f0211d239fc0e756d13dfb0e85dcefeb3bb14140fcbaa384",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 25]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_12203_wave27_preserves_both_owned_w26_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(12203, 26)]
    scope = gpu.SCOPES[(12203, 27)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 35
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795727440.780899)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["f1f7f01659bc42fd9b652cb05740bf2b", "bfd2c04a4900472388c84af214f5e72d"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "b297d490ae2ecd8e1999e8bd72d63001b33b4bda4516512daf7af12a274c81e9",
        "6fd09a8819a1796998b05e14ac1d7a743bad6671fe46dd22e144943f80ee05c7",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [25, 25]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_12203_wave26_preserves_both_owned_w25_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(12203, 25)]
    scope = gpu.SCOPES[(12203, 26)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 33
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795726373.246844)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["e6f1d4b3ee7145cc91dbd2970967f83a", "ed84d3db2fe14fb08ceeee420241e445"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "21780831e37426a092e2b5aea1533fd4facc1a0bdee362f91612450e6f70a4c0",
        "747c239aa8a093bc86a3776c6a76e084dc74666414df497052c973acb305a097",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 25]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_12203_wave25_preserves_both_owned_w24_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(12203, 24)]
    scope = gpu.SCOPES[(12203, 25)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 31
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795725286.1941266)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["8d53189d20c74d69a5b9ebb3729ac285", "b31f16e0b5204612add655f1995cc71e"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "55a3705a916a01da5ceccc283776377339466e9a2d8c564aa1f2c7e2a8f61124",
        "47042d1223569116bad111564a194f3b417c38aa615ae4bd282bde263c121a9b",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [25, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_12203_wave24_preserves_both_owned_w23_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(12203, 23)]
    scope = gpu.SCOPES[(12203, 24)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 29
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795723382.4180803)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["e7dc3a17644e4204bd63ee32ca8b819c", "d50d43f67cbb4f0f814baa9122654bea"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "f612246bc9f3fae685948de3d4e64bda72cd8d728b8dd97a781415c0696a6a33",
        "88fd1e2c33ac0d6b3e3c47841cd60e1393c9fc6d8a709efeb72996a6d71a9547",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_12203_wave23_preserves_both_owned_w22_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(12203, 22)]
    scope = gpu.SCOPES[(12203, 23)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 27
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795722137.2408683)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["5c8e1aa9a3e2468489d5e2f17c9cd061", "df7c651d002142829aea8c19d2be8af3"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "8a18c7fa771db6f429a3e1cb27afdd9f6087f6ae3b2ff8a15e9e49a441292de8",
        "db1ee53749d5ac595233d9c1efe14a91971e2cd3ca3f26ed7cf837a3fb655c77",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 25]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_12203_wave22_keeps_l493_and_bounds_retry_after_two_failures_without_text(monkeypatch):
    previous = gpu.SCOPES[(12203, 21)]
    scope = gpu.SCOPES[(12203, 22)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 25
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795719938.7627635)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == [
        "0dd696b221124af1b6c7b8e9c2c801fd", "53851a8fce95464fbadddeb5b692cf7a",
    ]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "c90e7e2da595681f809b2fc97f7e89b0c65e974282f3106f7f3d93bca7c94534",
        "fa5729da7f1920987643fcc497f0a1e47e3d056b5669cae0e29d8dfdf82ab24d",
    ]
    for item in (first, second):
        assert item.progress["http_status"] == 200 and item.progress["sse_event_count"] == 41
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": second.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_12203_wave21_preserves_r5_l493_and_reviews_the_two_owned_failures(monkeypatch):
    old = gpu.SCOPES[(12203, 20)]
    scope = gpu.SCOPES[(12203, 21)]
    assert scope.archive[:-1] == old.archive and sum(len(item.attempts) for item in scope.archive) == 23
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == old.boundary_physical_attempts == 2
    assert scope.payload_sha256 == old.payload_sha256 and scope.required_sources == old.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == old.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                         "maximum_physical_attempts": 182}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795718946.3791935
    review = scope.archive[-1]
    first, second = review.attempts
    assert first.identity == second.identity == gpu._12203_WAVE20 and first.request_id == second.request_id
    assert [item.progress["sse_event_count"] for item in review.attempts] == [25, 25]
    assert all(item.failure == "response_id_mismatch" and item.progress["provider_error_code"] == "upstream_error"
               and item.progress["response_id_matches_transport_alias"] for item in review.attempts)
    assert review.replay_files["attempt_01.replay_link.json"] == "b15b845a350c55ced28d0d1f08c1f263828a7a72dfa8797031f0bec47d44c38a"

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": first.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": first.request_id}
    checked = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry", lambda request, reservation: checked.append((request, reservation)))
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    p0 = {"attempt": 0, "attempt_id": "fresh-first"}
    p1 = {"attempt": 1, "attempt_id": "fresh-second"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - .01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, p0)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, p0)
    assert not checked
    dispatcher._check_boundary(request, p1)
    assert checked == [(request, p1)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**p1, "attempt": 2})


def test_12203_wave20_preserves_l493_and_reviews_both_exhausted_slots(monkeypatch):
    old = gpu.SCOPES[(12203, 19)]
    scope = gpu.SCOPES[(12203, 20)]
    assert scope.archive[:-1] == old.archive
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == old.boundary_physical_attempts == 2
    assert scope.payload_sha256 == old.payload_sha256 and scope.required_sources == old.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == old.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                         "maximum_physical_attempts": 182}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795717790.025714
    review = scope.archive[-1]
    first, second = review.attempts
    assert first.identity == second.identity == gpu._12203_WAVE19 and first.request_id == second.request_id
    assert [item.progress["sse_event_count"] for item in review.attempts] == [25, 41]
    assert all(item.failure == "response_id_mismatch" and item.progress["provider_error_code"] == "upstream_error"
               and item.progress["response_id_matches_transport_alias"] for item in review.attempts)
    assert review.replay_files["attempt_01.replay_link.json"] == "a5a2c9081e7519df6901789588902cf128888c8e8761494a4d2ad97f04a4d4a1"

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": first.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": first.request_id}
    checked = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry", lambda request, reservation: checked.append((request, reservation)))
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    next_first = {"attempt": 0, "attempt_id": "fresh-first"}
    next_second = {"attempt": 1, "attempt_id": "fresh-second"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - .01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, next_first)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, next_first)
    assert not checked
    dispatcher._check_boundary(request, next_second)
    assert checked == [(request, next_second)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**next_second, "attempt": 2})


def test_12203_wave19_pins_alias_and_idle_failures_and_gates_a_new_second_slot(monkeypatch):
    previous = gpu.SCOPES[(12203, 18)]
    scope = gpu.SCOPES[(12203, 19)]
    receipt = scope.archive[-1]
    first, second = receipt.attempts
    assert scope.archive[:-1] == previous.archive
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == len(receipt.attempts) == 2
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182}
    assert first.identity == second.identity == gpu._12203_WAVE18 and first.request_id == second.request_id
    assert [item.attempt_id for item in receipt.attempts] == [
        "0d93c0e153a34a7b8463363771c657c1", "d95e1a4f686a466ebff67c01c09bdd9f"]
    assert first.failure == "response_id_mismatch" and first.progress["sse_event_count"] == 41
    assert first.progress["provider_error_code"] == "upstream_error" and first.progress["response_id_matches_transport_alias"]
    assert second.failure == "timeout" and second.progress["error_type"] == "sse_inactivity_timeout"
    assert second.progress["sse_event_count"] == 40 and second.progress["event_type"] == "response.output_item.done"
    assert second.progress["elapsed_seconds"] - second.progress["last_sse_activity_seconds"] == pytest.approx(900, abs=.1)
    assert receipt.replay_files == {
        "attempt_00.replay_authorization.json": "c6519b97bc46d29e6bb21a05b66fa34a9eea6194fad5b1516a6e25ce6e1f505a",
        "attempt_00.replay.terminal.json": "97066490432870b2c57a338195df9329d3f918966ec82e15df4faaec9f25e655",
        "attempt_01.replay_link.json": "1c91a704844194e69d091d9a5b3db1f3235d026765a2f5aa42807db2e6e71a7d",
    }
    assert (scope.cooldown_field, scope.not_before_unix) == ("failed_unix", 1795715097.1493309)

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": first.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": first.request_id}
    checked = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry", lambda request, reservation: checked.append((request, reservation)))
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    next_first = {"attempt": 0, "attempt_id": "a" * 32}
    next_second = {"attempt": 1, "attempt_id": "b" * 32}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - .01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, next_first)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    for item in receipt.attempts:
        with pytest.raises(PermissionError, match="signed physical attempts"):
            dispatcher._check_boundary(request, {**next_first, "attempt_id": item.attempt_id})
    dispatcher._check_boundary(request, next_first)
    assert not checked
    dispatcher._check_boundary(request, next_second)
    assert checked == [(request, next_second)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**next_second, "attempt": 2})


def test_pre_post_trace_rejects_an_actual_http_send_or_response():
    trace = {
        "network_phases": {"dns_tcp": {"state": "complete"}, "tls": {"state": "failed"}},
        "first_phase_seconds": {"http_phase": 0.4, "request_dispatched": 25.0, "transport_failure": 55.0},
    }
    assert mac._pre_post_trace(trace)
    assert not mac._pre_post_trace({**trace, "http_status": 200})
    assert not mac._pre_post_trace({**trace, "first_phase_seconds": {"http_headers": 28.0}})
    assert not mac._pre_post_trace({**trace, "network_phases": {**trace["network_phases"], "send_body": {}}})
    assert not mac._pre_post_trace({**trace, "network_phases": {**trace["network_phases"], "tls": {"state": "complete"}}})


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


def test_157_wave09_preserves_the_three_cuda_fits_and_round_seven_payload():
    scope = gpu.SCOPES[(157, 9)]
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (4, 7, 7)
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170}
    assert scope.not_before_unix == 1795705081.8307962
    assert scope.required_sources == gpu.SCOPES[(6101, 8)].required_sources
    assert all(file_digest(gpu.ROOT / name) == sha for name, sha in scope.required_sources)
    files = dict(scope.fitted_prefix_files)
    assert {f"round_{i:02d}/transition.json" for i in range(5, 8)} <= files.keys()
    assert {f"round_{i:02d}/memory.json" for i in range(4, 7)} <= files.keys()
    assert scope.payload_sha256 == "a107c416d5d99771b3b14a88ee77b81c73378807da79cc849c3da05edae9e1e3"
    old, = scope.archive
    assert old.identity.name.endswith("-157-relay-retry-08")
    assert old.progress["error_type"] == "hard_wall_deadline" and old.progress["sse_event_count"] == 5937


def test_157_wave10_pins_the_same_l557_science_and_both_wave09_receipts(monkeypatch):
    previous = gpu.SCOPES[(157, 9)]
    scope = gpu.SCOPES[(157, 10)]
    assert scope.archive[:-1] == previous.archive
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (7, 7, 7)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.fitted_prefix_files == ()
    assert scope.required_sources == previous.required_sources and scope.recovery_budget == previous.recovery_budget
    assert scope.budget == previous.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85,
                                              "maximum_physical_attempts": 170}
    assert (scope.cooldown_field, scope.not_before_unix) == ("failed_unix", 1795707828.740078)
    review = scope.archive[-1]
    first, second = review.attempts
    assert first.identity == second.identity and first.request_id == second.request_id
    assert first.attempt_id == "ee1c6e1526ee4fd5b8985667fd5a7acd"
    assert first.failure == "response_id_mismatch" and first.progress["provider_error_code"] == "upstream_error"
    assert first.progress["sse_event_count"] == 25 and first.progress["response_id_matches_transport_alias"] is True
    assert second.attempt_id == "a36eb5dadf3848c480a6111e30fae29a"
    assert second.failure == "timeout" and second.progress["error_type"] == "sse_inactivity_timeout"
    assert second.progress["sse_event_count"] == 16 and second.progress["event_type"] == "response.output_item.done"
    assert review.replay_files["attempt_01.replay_link.json"] == (
        "58dfea77b6e9be2212170b4f3b9b7865c481480e594dc7623e46decfae43591a"
    )

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    admitted = []
    monkeypatch.setattr(mac.RelayRetryDispatcher, "_check_boundary",
                        lambda _, request, reservation: admitted.append((request, reservation)))
    request = {"payload_sha256": scope.payload_sha256}
    new = {"attempt": 0, "attempt_id": "c" * 32}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - .01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, new)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    for old in (first, second, *previous.archive[-1].attempts):
        with pytest.raises(PermissionError, match="signed physical attempts"):
            dispatcher._check_boundary(request, {**new, "attempt_id": old.attempt_id})
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**new, "attempt": 2})
    dispatcher._check_boundary(request, new)
    assert admitted == [(request, new)]


def test_157_wave12_retains_l557_and_adds_only_a_guarded_second_slot(monkeypatch):
    old = gpu.SCOPES[(157, 11)]
    scope = gpu.SCOPES[(157, 12)]
    assert scope.archive[:-1] == old.archive
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (7, 7, 7)
    assert scope.original_boundary_physical_attempts == old.boundary_physical_attempts == 1
    assert scope.payload_sha256 == old.payload_sha256 and scope.required_sources == old.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795713971.6597245
    receipt, = scope.archive[-1].attempts
    assert receipt.attempt_id == "1c721d7485b5425aa43fa84b40cb906c" and receipt.failure == "timeout"
    assert receipt.progress["error_type"] == "hard_wall_deadline" and receipt.progress["sse_event_count"] == 3544
    assert receipt.progress["last_sse_seconds"] == 3500.5466470001265
    assert receipt.progress["event_type"] == "response.output_text.delta" and "complete_terminal_captured" not in receipt.progress

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": receipt.attempt_id}]}
    dispatcher.original = {"binding": {}, "logical_index": 0, "payload": {"same": True}, "request_id": "old"}
    request = {**dispatcher.original, "request_id": "fresh"}
    guarded = []
    monkeypatch.setattr(mac.RelayRetryDispatcher, "_check_boundary_retry", lambda _, req, started: guarded.append((req, started)))
    first = {"attempt": 0, "attempt_id": "fresh-first"}
    second = {"attempt": 1, "attempt_id": "fresh-second"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first)
    dispatcher._check_boundary(request, second)
    assert guarded == [(request, second)]


@pytest.mark.parametrize("wave,eligible_unix,attempt_ids,terminal_shas", [
    (24, 1795731435.068868,
     ["afbcb8adb16847d097eb676509fe245a", "56dcb2c4d8384847872b8cda68e4de43"],
     ["83eccebb686f2e0a10d24956edf72a1c2a8caf2c61930c8c886b5ae037193309",
      "5a1c837c7398939e407815fee96cfedbd7e158400ff9fa3d466e8c5f772fb32c"]),
    (25, 1795732390.0013366,
     ["d4ddddd9d12a4410bc9da955510b954b", "3f49545be09a48caaa7e58eff1e59183"],
     ["591591be3f162772f976b62ea842fc53d4c440d570f234f35dfbb663168541ac",
      "e6bc3ce452e40ac7a2ee1204b44fd3743850b9e83524a64ce16e9c7a99c12cb5"]),
])
def test_157_preserves_both_owned_failures_and_p1_guard(monkeypatch, wave, eligible_unix, attempt_ids, terminal_shas):
    previous = gpu.SCOPES[(157, wave - 1)]
    scope = gpu.SCOPES[(157, wave)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == (29 if wave == 24 else 31)
    assert scope.original_start_round == scope.round == scope.replay_start_round == 7
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85,
                                              "maximum_physical_attempts": 170}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", eligible_unix)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == attempt_ids
    assert [item.terminal_file_sha256 for item in (first, second)] == terminal_shas
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


@pytest.mark.parametrize("wave,total_archived,eligible_unix,attempt_ids,terminal_shas,second_failure,expected_sse", [
    (29, 39, 1795736565.0167,
     ["3c03b8c5136c457a8b56706c46e48481", "c202018f7191414880907775b4ec2479"],
     ["ce671501df2d3688b771f89f9e7a3e4af096ffa70e7518ab04af094515c0e8b2",
      "c29d5feb98d2a687b810c7095f170703cbd311836d4c65b97c192b8a49548727"], "response_id_mismatch", [41, 41]),
    (30, 41, 1795738643.1450398,
     ["777d3b522a6447c0b6565c5f62c9762b", "53a1f934df2a401695fb4ea12e24f6cb"],
     ["ab143c4679e721dd1b9d923eb657070f4a1f16bac76058ceff54094cc455ee73",
      "642eba1fdd04da762dc73c29c8ae087471b344f1e9aa93768b94df9309426dee"], "response_id_mismatch", [41, 41]),
    (31, 43, 1795740329.9381309,
     ["1f856802fce4415a8bf718fbe73645dd", "0bb8a2c34e2c41d0a7d64c1b4474e37a"],
     ["ea260a269df18fdcb524587bef5c2b704c771f0f2f246e6745f0bd88a3061966",
      "c2ea039e916c8b3a1f8475896a3d0d9286808ab400a9f0e307e982df7443f1f7"], "sse_frame_error", [41, 33]),
    (32, 45, 1795741480.156861,
     ["4714a2a236364476b8a8429a97458f8d", "678f556c180c4ea4bd15605fa9ed3405"],
     ["e8373519e8299e337f3b12fee7c57f18ac7fa29e7297c3a1c7f96f1b60517b51",
      "fab6ec07bd38aee0ebc011ba07936d0fb43dfd2c1202006812707b059479d01b"], "response_id_mismatch", [41, 41]),
    (33, 47, 1795742579.7300103,
     ["be0356649b834158ada557ada8ed0942", "0cac35126289455a8c915a9fe8b801f5"],
     ["2e5ea06e4a450002038ac7ec9322353c8b387fc809bf932f388fb39d6a419cef",
      "09d18108221029c523b897d4a6468ff0146c7aa0e8aff04b4ea3381b7fb0b4ee"], "response_id_mismatch", [25, 41]),
    (34, 49, 1795743613.2174969,
     ["475190b05c444365a86dbb5c392049ab", "f7bf4a956cb14328ad35d68de916dfdb"],
     ["1d40cf77cf2ee269dd08ae8514c427d8de0a08457763b31923d3447b20f81c78",
      "c7e55a186779929d87048bf53c824ebf3cb983faee26e534b254974ae6ac0532"], "response_id_mismatch", [41, 23]),
    (37, 53, 1795746649.5255659,
     ["bd7a08f561004b19b9853bbcd01d0cc5", "d4d4475a91ff4c57b5218481fb7f225f"],
     ["7765ce8a28cb2122f1ec99085e1cf955818833332c1ac9718a7e18611937de13",
      "65edf8d1130a99a0f5cf233447cbd4ac81e697f113d2dc491575679db46914a9"], "response_id_mismatch", [41, 25]),
    (38, 55, 1795749546.8450897,
     ["e18055c94f9442d9b191102e4d72cf9b", "28353e2b81b5423cb41889ab0988126c"],
     ["eece665dc261af459c3923bc11a2bdb1e324125d970efd3c6678a1dae4f828a6",
      "2bfd0be30e444855bf99685d0ca163e2ad45e6b3b124427e8a7e9b230561a137"], "response_id_mismatch", [41, 41]),
])
def test_157_latest_scopes_preserve_reviewed_failures_and_p1_guard(monkeypatch, wave, total_archived, eligible_unix,
                                                               attempt_ids, terminal_shas, second_failure, expected_sse):
    previous = gpu.SCOPES[(157, wave - 1)]
    scope = gpu.SCOPES[(157, wave)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == total_archived
    assert scope.original_start_round == scope.round == scope.replay_start_round == 7
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85,
                                              "maximum_physical_attempts": 170}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", eligible_unix)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == attempt_ids
    assert [item.terminal_file_sha256 for item in (first, second)] == terminal_shas
    assert (first.failure, second.failure) == ("response_id_mismatch", second_failure)
    assert [item.progress["sse_event_count"] for item in (first, second)] == expected_sse
    for item in (first, second):
        assert item.progress["http_status"] == 200
        if item.failure == "sse_frame_error":
            assert item.progress["error_type"] == "ReadError"
            assert item.progress.get("response_id_matches_transport_alias") is None
            assert item.progress.get("provider_error_code") is None
            assert not is_sealed_relay_failure({"error": item.failure, "response_id": item.progress["response_id"]})
        else:
            assert item.progress["provider_error_code"] == "upstream_error"
            assert item.progress["response_id_matches_transport_alias"] is True
            assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_157_wave36_reviews_only_actual_wave35_slot_and_guards_new_second_slot(monkeypatch):
    previous = gpu.SCOPES[(157, 35)]
    scope = gpu.SCOPES[(157, 36)]
    assert scope.archive[:-1] == previous.archive
    review, = scope.archive[-1].attempts
    assert len(scope.archive) == 28 and sum(len(item.attempts) for item in scope.archive) == 51
    assert scope.original_start_round == scope.round == scope.replay_start_round == 7
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795745450.5783978)
    assert review.identity.contract == "fd7028bb24038fae4340bcdffb7d3c6aaedca8b3f71059f4d1084029bf3adcf4"
    assert review.terminal_file_sha256 == "7583ed6e4657e8359c75ea7a31b2e0537f0d9b6054437e752d6114559f85831b"
    trace = review.progress
    assert (trace["http_status"], trace["sse_event_count"], trace["provider_error_code"]) == (200, 25, "upstream_error")
    assert trace["conflicting_event_type"] == "response.failed" and trace["response_id_matches_transport_alias"] is True
    assert trace["conflicting_response_id"] == "resp_" + trace["provider_request_id"].replace("-", "")

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": review.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": review.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first = {"attempt": 0, "attempt_id": "new-first"}
    second = {"attempt": 1, "attempt_id": "new-second"}
    admitted = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry", lambda req, started: admitted.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first)
    assert admitted == []
    dispatcher._check_boundary(request, second)
    assert admitted == [(request, second)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second, "attempt": 2})


def test_157_wave39_carries_one_slot_after_owned_failure_and_unposted_readiness(tmp_path, monkeypatch):
    previous, scope = gpu.SCOPES[(157, 38)], gpu.SCOPES[(157, 39)]
    assert scope.archive[:-1] == previous.archive
    assert len(scope.archive) == 31 and sum(len(item.attempts) for item in scope.archive) == 57
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (7, 7, 8)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 169}
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.not_before_unix == 1795751251.8996296
    assert dict(scope.fitted_prefix_files) == {
        "pause.request": "fe7d847f1a688de23e34a3a96e1e998f014b4a0dab106021893453d68f5407f9",
        "round_07/source.json": "ac42a77b7b18963db4be8ca807aa68bc04636643cdb8b0f9d0226a3f515a1aee",
    }
    sent, stopped = scope.archive[-1].attempts
    assert sent.receipt_provider_flags == (True, True) and sent.progress["sse_event_count"] == 27
    assert sent.progress["provider_error_code"] == "upstream_error" and sent.progress["response_id_matches_transport_alias"]
    assert stopped.receipt_provider_flags == (False, False) and stopped.failure == "connect_error"
    assert replace(stopped, progress_file_sha256="a"*64).receipt_provider_flags == (True, True)
    assert stopped.progress["event_type"] == "ipv4_preflight_no_headers"
    assert [sent.terminal_file_sha256, stopped.terminal_file_sha256] == ["09429bd6fcc44118fa344d1ef48d1c77bff1e2ff1b8a081d0f5b65ff40890b75", "0de21da74dbea2e27e212a01796e39fd3332f44855c769b45120e492aee1e386"]
    assert stopped.replay_files == {
        "attempt_00.replay_authorization.json": "8cf5fef564ce5ef49b3fb68ed3d348b18950fac1e4058aa54b76749a8dfa1b3e",
        "attempt_00.replay.terminal.json": "9611bd57f964e04a44ab50440137d3cf5bc1cd7bc172f8f8be2435535975b556",
        "attempt_01.replay_link.json": "8a01fc808db62613c0d5a375ff05e25657bdc2fdf84e496567fb93f5eb8eeeb3",
    }
    for index, retry_unix in enumerate((1795750911.210913, 1795750966.365126, 1795751024.469307), 1):
        once(tmp_path / f"readiness_{index:02d}.json", {
            "attempt_id": stopped.attempt_id, "job_sha256": stopped.job_sha256, "retry_unix": retry_unix, "window": index,
        })
    expected = stopped.unposted_readiness_files
    assert len(expected) == 3 and mac._unposted_readiness(tmp_path, stopped.progress, expected)
    assert not mac._unposted_readiness(tmp_path, {**stopped.progress, "http_status": 200}, expected)
    once(tmp_path / "started.json", {})
    assert not mac._unposted_readiness(tmp_path, stopped.progress, expected)

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (sent, stopped)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": stopped.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first = {"attempt": 0, "attempt_id": "new-first"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first)
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**first, "attempt": 1})


def test_157_wave35_consumes_only_unused_slot_after_headers_without_sse(monkeypatch):
    previous = gpu.SCOPES[(157, 34)]
    scope = gpu.SCOPES[(157, 35)]
    assert scope.archive[:-1] == previous.archive
    review, = scope.archive[-1].attempts
    assert sum(len(item.attempts) for item in scope.archive) == 50
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (7, 7, 8)
    assert scope.boundary_physical_attempts + len(scope.archive[-1].attempts) == scope.original_boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 169}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795744861.3835022)
    assert review.failure == "timeout"
    assert review.terminal_file_sha256 == "6cf49c7056d20ce8e4f5817958205743ee30bf9062fbe4c02079888e9895e0b3"
    progress = review.progress
    assert progress["http_status"] == 200 and progress["error_type"] == "sse_inactivity_timeout"
    assert progress["network_phases"]["send_body"]["state"] == "complete"
    assert progress["first_phase_seconds"]["http_headers"] == 35.38578833406791
    assert not {"first_sse_seconds", "sse_event_count", "response_id", "first_text_seconds", "complete_terminal_captured"}.intersection(progress)
    assert not is_sealed_relay_failure({"error": review.failure, "provider_request_id": progress["provider_request_id"]})

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": review.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": review.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first = {"attempt": 0, "attempt_id": "new-first"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first)
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**first, "attempt": 1})


def test_157_wave28_preserves_both_owned_w27_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(157, 27)]
    scope = gpu.SCOPES[(157, 28)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 37
    assert scope.original_start_round == scope.round == scope.replay_start_round == 7
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85,
                                              "maximum_physical_attempts": 170}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795735518.7383046)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["00718d2c5eff4ebd91cb3057f1698b14", "4dab6aef3a0d463b8e479c56bde37874"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "4ea0599c0bcf2d2eec16bbde7b9662ed23d24504d5bcd4a580b8064a8b2a9f6b",
        "b2e1dd5d66d860d7d4085cb3f36600070c86b7a0046c88c2416a2e5bcbeacbf6",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_157_wave27_preserves_both_owned_w26_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(157, 26)]
    scope = gpu.SCOPES[(157, 27)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 35
    assert scope.original_start_round == scope.round == scope.replay_start_round == 7
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85,
                                              "maximum_physical_attempts": 170}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795734523.5643582)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["e2c4454234bb4fbf98c2bd9997833edf", "3571d6d09f704437a838c8bf2024f709"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "773dba409e12ae2f9b9f02af0e37d8eae620b685fc19c7846d3628cfb0ad381c",
        "b6688422ffbc826efd1a1657036a7eb7cad37bee7e79baa5dc868693c8d41d3c",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_157_wave26_preserves_both_owned_w25_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(157, 25)]
    scope = gpu.SCOPES[(157, 26)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 33
    assert scope.original_start_round == scope.round == scope.replay_start_round == 7
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85,
                                              "maximum_physical_attempts": 170}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795733574.8626847)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["efa33350473941da9ea4e7db87d73d11", "85ab8b7135e742a3abaa5f67101bc5d2"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "69f6dce3bddb2e3b2ad0237edf164e60bb64c733abaddf34cc790fa45972bdce",
        "1f6bcf4541f48b16b36d7956c6f6386126fddb43a9884e1d3da19a6ed00b7830",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_157_wave23_preserves_both_owned_w22_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(157, 22)]
    scope = gpu.SCOPES[(157, 23)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 27
    assert scope.original_start_round == scope.round == scope.replay_start_round == 7
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85,
                                              "maximum_physical_attempts": 170}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795730515.952652)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["2b9b653c5a98441eb5d17613841cbcc6", "33fabf638c754f0ca1655aeb45b3f763"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "17d93807928b22536fc71f443bb5833048a55597cde2ca74391a2aade757d744",
        "33b6bf126eba81acccbce695d4e0de5ae1da57612e1ccbf8dbdf48c6af3a6432",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [27, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_157_wave22_preserves_both_owned_w21_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(157, 21)]
    scope = gpu.SCOPES[(157, 22)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 25
    assert scope.original_start_round == scope.round == scope.replay_start_round == 7
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85,
                                              "maximum_physical_attempts": 170}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795729527.1188662)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["b41e1069da06490dba37e4cf783a19d5", "4bc6d117ceb145dba341f1e694f903e8"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "47f6405a933f740def543e2948d852b08eb1a7a1ae718a19051c8d60a730c5af",
        "3f5029b698dbbac214f7826a53111deb066b980c791abfbdcac9d0a87cc0fc00",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [25, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_157_wave21_preserves_both_owned_w20_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(157, 20)]
    scope = gpu.SCOPES[(157, 21)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 23
    assert scope.original_start_round == scope.round == scope.replay_start_round == 7
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85,
                                              "maximum_physical_attempts": 170}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795728380.9793127)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["aa56b198c9b04413ace9ca158b0fc34b", "7a9c1ecb0d024f189769ef2d2954374b"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "9082931239072337a8e46317dc0c8ae1cb19b47613cfbf12d62ffbfcc90ec047",
        "b1c6f17416f5ca38fc74e34747afdd360366becf7c838a23cb582262a5a08e9b",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 25]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_157_wave20_preserves_both_owned_w19_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(157, 19)]
    scope = gpu.SCOPES[(157, 20)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 21
    assert scope.original_start_round == scope.round == scope.replay_start_round == 7
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85,
                                              "maximum_physical_attempts": 170}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795727243.4760141)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["3eee29e09a504e3ca05604684fd42780", "c027d0bb376640a59a9127deff102363"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "4c58a5ef02f9d710f4f28e53f2bf214876a8dc1b6f52bef665487df5ce1c121e",
        "313bf93d351ae6ca5ed2fb177c76b517aef5ae9f02ae245b6851c790c9d8029a",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [25, 25]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_157_wave19_preserves_both_owned_w18_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(157, 18)]
    scope = gpu.SCOPES[(157, 19)]
    assert scope.archive[:-1] == previous.archive
    assert scope.original_start_round == scope.round == scope.replay_start_round == 7
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.required_sources == previous.required_sources and scope.header_deadline_profile == {}
    assert scope.payload_sha256 == previous.payload_sha256 == "a107c416d5d99771b3b14a88ee77b81c73378807da79cc849c3da05edae9e1e3"
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85,
                                              "maximum_physical_attempts": 170}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795725903.7298713)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id == "6e0635eb183a9921de26ff86560f2ad11d0dde26b82ee5315b31e8f911d5801b"
    assert [item.attempt_id for item in (first, second)] == ["dc7cfa5063e54558beb40e16fba7a0cc", "6e2250d2243c49b3888b1c92dd4ebcb1"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "5a1c97ba41c1dd3c3c5df2a699e53c8059264e70f5b4e3bf4dde0dcd209f731c",
        "616c2e1b13491406c543ddb968d66cb3f3dd919b9ed0ab51d802d0482d29adde",
    ]
    assert second.replay_files == {
            "attempt_00.replay_authorization.json": "44adc2554e9507e69a247c98de94a2cd785573fad4b6354c5f2ab214c473dae7",
            "attempt_00.replay.terminal.json": "46fe156552c86721fafd3db97c63372916e4a7c6bec6ea17700813808de3f8e7",
            "attempt_01.replay_link.json": "eff15bd38ffae5997f2f4bfbdc129854ec5a5d679df2fd7ee0e2075838ffdf41",
    }
    assert [item.progress["sse_event_count"] for item in (first, second)] == [25, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert not any("output_text" in event for event in item.progress["sse_milestones"])

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    physical0 = {"attempt": 0, "attempt_id": "new-first"}
    physical1 = {"attempt": 1, "attempt_id": "new-second"}
    guarded = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry", lambda req, started: guarded.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, physical0)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, physical0)
    dispatcher._check_boundary(request, physical1)
    assert guarded == [(request, physical1)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**physical1, "attempt": 2})
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**physical1, "attempt_id": second.attempt_id})


def test_157_wave18_preserves_both_owned_w17_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(157, 17)]
    scope = gpu.SCOPES[(157, 18)]
    assert scope.archive[:-1] == previous.archive
    assert [len(item.attempts) for item in scope.archive] == [1, 2, 1, 1, 2, 2, 2, 2, 2, 2]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 7
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85,
                                              "maximum_physical_attempts": 170}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795723851.023255)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["21368d1641eb45258b3c573d8e05c6c8", "9e607209d29c4dc0a5553f3f3cce9942"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "f085c4c7f362da41c9ecc9683153707ef4a7b6ecc3f488a1d2363d2f39364378",
        "8927aac5d427f40752ae4d3c50da49309df7317448d2292c8c20b92d859ae786",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [25, 39]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_157_wave17_preserves_both_owned_w16_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(157, 16)]
    scope = gpu.SCOPES[(157, 17)]
    assert scope.archive[:-1] == previous.archive
    assert [len(item.attempts) for item in scope.archive] == [1, 2, 1, 1, 2, 2, 2, 2, 2]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 7
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85,
                                              "maximum_physical_attempts": 170}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795722782.7178502)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["a428b0b0658443ef9b4d1e3de274a8ef", "39039b9e9017416984944f65a4eb37a8"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "d8d89a2ad5994beae4c1ebc18a5486a8096c8953cb4ed7e57380bafe6f0b75cb",
        "6bde11c304287cefe9b956f7deb61e6372aeaab1e4f8f70fcdcabc2273067766",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [25, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_157_wave16_keeps_l557_and_bounds_retry_after_two_failures_without_text(monkeypatch):
    previous = gpu.SCOPES[(157, 15)]
    scope = gpu.SCOPES[(157, 16)]
    assert scope.archive[:-1] == previous.archive
    assert [len(item.attempts) for item in scope.archive] == [1, 2, 1, 1, 2, 2, 2, 2]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 7
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85,
                                              "maximum_physical_attempts": 170}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795721741.325949)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == [
        "86e32af54c0e486285bb634fb55cbef8", "0753108b819c4adcbcb54938c6d5bdda",
    ]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "812fe89ce0153b3965857e329590ab95a05d667eec8ac22479697a91d216fbf1",
        "c4fd4fabef6dfd1c48ce3e3c796622d0cf327c98f4bcada47ba64f9258192b4e",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 25]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": second.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_157_wave15_keeps_l557_and_bounds_retry_after_two_failures_without_text(monkeypatch):
    previous = gpu.SCOPES[(157, 14)]
    scope = gpu.SCOPES[(157, 15)]
    assert scope.archive[:-1] == previous.archive
    assert [len(item.attempts) for item in scope.archive] == [1, 2, 1, 1, 2, 2, 2]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 7
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85,
                                              "maximum_physical_attempts": 170}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795719546.705628)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == [
        "2b15775e3fbf432bac822bf518b3f6fd", "140178c833074f75986a63a927d8ca00",
    ]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "c973eff85c8b6fe49414450082face5f86c9cf15db46f0aa357287d3705cc1eb",
        "96240c037c9f6683c7a041ca92cce890dd30b1629049101f6b30b442feb02adb",
    ]
    for item in (first, second):
        assert item.progress["http_status"] == 200 and item.progress["sse_event_count"] == 41
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": second.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_157_wave14_preserves_l557_and_requires_a_fresh_guarded_second_slot(monkeypatch):
    old = gpu.SCOPES[(157, 13)]
    scope = gpu.SCOPES[(157, 14)]
    assert scope.archive[:-1] == old.archive and [len(item.attempts) for item in scope.archive] == [1, 2, 1, 1, 2, 2]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 7
    assert scope.original_boundary_physical_attempts == old.boundary_physical_attempts == 2
    assert scope.payload_sha256 == old.payload_sha256 and scope.required_sources == old.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == old.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85,
                                         "maximum_physical_attempts": 170}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795718562.9140031
    review = scope.archive[-1]
    first, second = review.attempts
    assert first.identity == second.identity == gpu._157_WAVE13 and first.request_id == second.request_id
    assert [item.progress["sse_event_count"] for item in review.attempts] == [25, 41]
    assert all(item.failure == "response_id_mismatch" and item.progress["provider_error_code"] == "upstream_error"
               and item.progress["response_id_matches_transport_alias"] for item in review.attempts)
    assert review.replay_files["attempt_01.replay_link.json"] == "09fa2525b10dfb27e2a0cf712dae256571fb33389c7c7a5d405b8d6649adf4ae"

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": first.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": first.request_id}
    checked = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry", lambda request, reservation: checked.append((request, reservation)))
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    p0 = {"attempt": 0, "attempt_id": "fresh-first"}
    p1 = {"attempt": 1, "attempt_id": "fresh-second"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - .01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, p0)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, p0)
    assert not checked
    dispatcher._check_boundary(request, p1)
    assert checked == [(request, p1)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**p1, "attempt": 2})


def test_157_wave13_requires_both_l557_failures_and_gates_a_fresh_second_slot(monkeypatch):
    previous = gpu.SCOPES[(157, 12)]
    scope = gpu.SCOPES[(157, 13)]
    receipt = scope.archive[-1]
    first, second = receipt.attempts
    assert scope.archive[:-1] == previous.archive
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.fitted_prefix_files == previous.fitted_prefix_files and scope.original_start_round == 7
    assert scope.round == scope.replay_start_round == 7
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == len(receipt.attempts) == 2
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 170}
    assert first.identity == second.identity == gpu._157_WAVE12 and first.request_id == second.request_id
    assert [item.attempt_id for item in receipt.attempts] == [
        "8973b62b1ef34dbc92b9140dff4757cd", "3ea1f797475745898d5e73835cdf80ab"]
    assert first.failure == "response_id_mismatch" and first.progress["sse_event_count"] == 41
    assert first.progress["provider_error_code"] == "upstream_error" and first.progress["response_id_matches_transport_alias"]
    assert second.failure == "timeout" and second.progress["error_type"] == "sse_inactivity_timeout"
    assert second.progress["sse_event_count"] == 24 and second.progress["event_type"] == "response.output_item.done"
    assert second.progress["elapsed_seconds"] - second.progress["last_sse_activity_seconds"] == pytest.approx(900, abs=.1)
    assert receipt.replay_files == {
        "attempt_00.replay_authorization.json": "bbe0bcf9ca35132f2caa1af53b00212771775b6686833ab855f60a9a1e20e5ab",
        "attempt_00.replay.terminal.json": "8ca191ce30982e122f4c55c903dd40f7574d3c98c9d9dff03f35ca063e7177eb",
        "attempt_01.replay_link.json": "1b9229d17f2971f125b0063a2401d60700412466fac065ccf2300dafd1d8b5c6",
    }
    assert (scope.cooldown_field, scope.not_before_unix) == ("failed_unix", 1795717169.7967703)

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": first.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": first.request_id}
    checked = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry", lambda request, reservation: checked.append((request, reservation)))
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    next_first = {"attempt": 0, "attempt_id": "a" * 32}
    next_second = {"attempt": 1, "attempt_id": "b" * 32}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - .01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, next_first)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    for item in receipt.attempts:
        with pytest.raises(PermissionError, match="signed physical attempts"):
            dispatcher._check_boundary(request, {**next_first, "attempt_id": item.attempt_id})
    dispatcher._check_boundary(request, next_first)
    assert not checked
    dispatcher._check_boundary(request, next_second)
    assert checked == [(request, next_second)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**next_second, "attempt": 2})


def test_157_wave11_carries_only_the_remaining_slot_after_the_post_read_error():
    previous = gpu.SCOPES[(157, 10)]
    scope = gpu.SCOPES[(157, 11)]
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (7, 7, 8)
    assert scope.archive[:-1] == previous.archive and scope.payload_sha256 == previous.payload_sha256
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 169}
    assert scope.not_before_unix == 1795709712.7965784
    last = scope.archive[-1]
    assert len(last.attempts) == 1 and not last.pre_post_connect_failure
    assert last.attempt_id == "1403a2ad4fd640f38431fc5b41e8c63b"
    assert last.failure == "connection_interrupted" and last.progress["error_type"] == "ReadError"
    assert last.progress["event_type"] == "ipv4_preflight_http_401" and last.progress["http_status"] is None


def test_887_r32_retains_the_one_leftover_slot_and_pins_partial_failed_stream_custody():
    scope = gpu.SCOPES[(887, 11)]
    review, = scope.archive
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (20, 32, 33)
    assert scope.boundary_physical_attempts + len(review.attempts) == scope.original_boundary_physical_attempts == 2
    assert scope.budget == {"first_logical_requests": 4, "maximum_logical_requests": 10, "maximum_physical_attempts": 19}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.not_before_unix == 1795731145.8450844
    assert review.identity == gpu._887_R20_CONTINUATION and review.failure == "response.failed"
    progress = review.progress
    assert progress["http_status"] == 200 and progress["sse_event_count"] == 9
    assert progress["provider_error_code"] == "upstream_error" and progress["provider_request_id"] is None
    assert "response.output_text.delta" in progress["sse_milestones"]
    assert "response.output_text.done" in progress["sse_milestones"]
    assert "response.completed" not in progress["sse_milestones"]
    assert not is_sealed_relay_failure({"error": review.failure, "provider_status": "failed",
                                        "response_id": progress["response_id"], "provider_request_id": None})
    assert {"round_31/memory.json", "round_31/batch_freeze.json", "round_32/transition.json",
            "round_32/fit_000/fit_audit.json", "round_32/turn_00/request.json"} <= dict(scope.fitted_prefix_files).keys()
    parser = (gpu.STUDY.parent / "pure_llm_v3_20261124/pure_llm_v3/transport_manager.py").as_posix()
    assert dict(scope.required_sources)[parser] == "2cdc0990f81012f040a73f8d5b7226fe1d35b934f477dc1654ee071126193a3c"


def test_887_wave23_preserves_single_owned_wave22_and_reopens_guarded_second_slot(monkeypatch):
    previous = gpu.SCOPES[(887, 22)]
    scope = gpu.SCOPES[(887, 23)]
    assert scope.archive[:-1] == previous.archive
    assert len(scope.archive) == 12 and sum(len(item.attempts) for item in scope.archive) == 23
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (34, 34, 34)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == {"first_logical_requests": 2, "maximum_logical_requests": 4, "maximum_physical_attempts": 8}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795749090.3075333)
    assert dict(scope.fitted_prefix_files)["boundary.json"] == "6a13c7d3e2675dec9ff2f8b743a0d8fbb4af46295a303b1b1e16f1f3e5639889"
    assert dict(scope.fitted_prefix_files)["round_34/turn_00/request.json"] == "1ac029a766fb732630d79cfc114e0a4739ab2c9121a6610d1c3fbfb447d31f77"
    review, = scope.archive[-1].attempts
    assert review.terminal_file_sha256 == "92284a23412c374a7303a84de98f94a85acf90ec96f5a6f0dfe6ef38abfdb3f6"
    assert review.failure == "response_id_mismatch" and review.progress["provider_error_code"] == "upstream_error"
    assert review.progress["sse_event_count"] == 41
    assert review.progress["conflicting_event_type"] == "response.failed"
    assert review.progress["response_id_matches_transport_alias"] is True

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": review.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": review.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first = {"attempt": 0, "attempt_id": "new-first"}
    second = {"attempt": 1, "attempt_id": "new-second"}
    admitted = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry", lambda req, started: admitted.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first)
    assert admitted == []
    dispatcher._check_boundary(request, second)
    assert admitted == [(request, second)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second, "attempt": 2})


def test_887_wave22_reviews_ambiguous_second_attempt_and_limits_one_new_physical(monkeypatch):
    previous = gpu.SCOPES[(887, 21)]
    scope = gpu.SCOPES[(887, 22)]
    assert scope.archive[:-1] == previous.archive
    assert len(scope.archive) == 11 and sum(len(item.attempts) for item in scope.archive) == 22
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (34, 34, 35)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 2, "maximum_logical_requests": 4, "maximum_physical_attempts": 7}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795747698.3113158)
    assert dict(scope.fitted_prefix_files)["boundary.json"] == "459c5247566652b375f56e2922364a22a0ca87dac6d6f46c7e1842ea27aa445d"
    assert dict(scope.fitted_prefix_files)["round_34/turn_00/request.json"] == "4b806a46625517419cef2916f3301aa85b3bd309baee8048a910fb33466f433d"
    first, last = scope.archive[-1].attempts
    assert first.progress["provider_error_code"] == "upstream_error"
    assert last.terminal_file_sha256 == "77925f04bded91c702341525fa7e2890d989d1340524d72ea64f20394776bbba"
    assert last.failure == "timeout" and last.progress["error_type"] == "sse_inactivity_timeout"
    assert last.progress["sse_event_count"] == 25
    assert not {"provider_error_code", "conflicting_event_type", "response_id_matches_transport_alias",
                 "first_text_seconds", "complete_terminal_captured"} & last.progress.keys()

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": attempt.attempt_id} for attempt in (first, last)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": last.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    started = {"attempt": 0, "attempt_id": "new-first"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, started)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, started)
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**started, "attempt": 1})


@pytest.mark.parametrize("wave,total_archived,eligible_unix,request_sha,terminal_shas,expected_sse", [
    (15, 8, 1795738639.839541, "2d1f61fcf068314ecb7124edf95dba659480cb552b700dc73577ee0f6667f7bd",
     ["35cb0713357f835dc4cc0f1cb0cda33cdae0b29b29f4e141619d4b2f953482c5",
      "df4e812be59f9b2ae132bfba4e5cb33c8344bd603e30c2b8890c9e4f614b51d7"], [41, 41]),
    (16, 10, 1795739695.7711978, "b542064daf2261ba4b888ceeadbbc5fbf50f85c5da7f5be6dec725c75df4e7c9",
     ["1515b83fb718b479394c8acc2b8abef3942037f77bad7a7668b3442fb8ef4d9f",
      "9048ceda9accf27113f950918eb7a14c08a487ea9774ecc90bdebacfff5c57b6"], [39, 41]),
    (18, 14, 1795742290.9625022, "2fdc8b5ffa40ee7ca0bf63334a96a2f88a40ae28afed03bcced888bbd773432a",
     ["33484fe93a23ee15dc5cfaf8a6dcb48ea539d80d49aaa97ec9ff985d2190b1ca",
      "8e00a867a1934bf0cd91ca79c1dc72166021bb76cf096715584c36d6c84f32c9"], [41, 23]),
    (19, 16, 1795743289.2941442, "07df44f0431acbd58d47f4befd6b879513ca8e1ff912867e0c4c2bb84ba3af79",
     ["0c66e8083061daf4c7d333f8dc816edd8cfb4f0e3f16d700e3d98d900217c354",
      "bb175f9bf59fcac48923a201248a378a5bdbabdad3bff6734ff7096cf0657b34"], [41, 41]),
    (20, 18, 1795744552.392818, "eccf49870d9439c121a5cbebdbac0bda7fc9167a1acf0cd9a01251925051c9d9",
     ["6f329b4aa1dae2817470f56953744c4e0e7748be4ed644962cd6b79a8e421417",
      "b94a3f2797605ddc36781e739ec2a7599cd051901f33930af156557be9a68bc5"], [25, 41]),
    (21, 20, 1795745695.7466342, "c51f210de09b7e15ac65549a9921ee48e18c5a64d3caa850e80fac52ef8a0dac",
     ["21c6492f73dcf35ef888bd7182581fc0944e0749b3dd9f6325715611345441c9",
      "0b3aec9efe5657c07217088d2093038882cbd4a47649bf295e917652b870a671"], [41, 27]),
])
def test_887_latest_r34_scopes_keep_l1421_and_signed_guard_after_owned_failures(
        monkeypatch, wave, total_archived, eligible_unix, request_sha, terminal_shas, expected_sse):
    previous = gpu.SCOPES[(887, wave - 1)]
    scope = gpu.SCOPES[(887, wave)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == total_archived
    assert scope.original_start_round == scope.round == scope.replay_start_round == 34
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 == "eb9170128722472e627a9ae16fa23692c1fc19c3ed6156513bde1c98e0c45a91"
    assert scope.required_sources == previous.required_sources
    sources = {name.rsplit("/", 1)[-1]: value for name, value in scope.required_sources}
    assert sources["seed_runner.py"] == "417f41255d53d7794abcdc98c0a9258141d01c076df845bfa877013c84de1792"
    assert sources["transport_manager.py"] == "2cdc0990f81012f040a73f8d5b7226fe1d35b934f477dc1654ee071126193a3c"
    assert dict(scope.fitted_prefix_files)["round_34/turn_00/request.json"] == request_sha
    if wave == 18:
        assert dict(scope.fitted_prefix_files)["boundary.json"] == (
            "2b06c2cb393819ae9a6d09b8ae92825b2f3da5e4b202b9d75a294bbfed64f313"
        )
    if wave == 19:
        assert dict(scope.fitted_prefix_files)["boundary.json"] == (
            "c99c64ced68bae00dbad07c5a196ade4e1e9b1e59a5b414f4d862c3f12c08d45"
        )
    if wave == 20:
        assert dict(scope.fitted_prefix_files)["boundary.json"] == (
            "7db77ed950fe94e0a5f92687e5ccb44760a6d847c7cb4b1000d8545685437ac6"
        )
    if wave == 21:
        assert dict(scope.fitted_prefix_files)["boundary.json"] == (
            "4cb569159c5a6b1cbeb7a8e131bb8a70fd78e84983a3ae86d4ff706e848f0c7e"
        )
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == {"first_logical_requests": 2, "maximum_logical_requests": 4, "maximum_physical_attempts": 8}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", eligible_unix)
    first, second = scope.archive[-1].attempts
    assert [item.terminal_file_sha256 for item in (first, second)] == terminal_shas
    assert [item.progress["sse_event_count"] for item in (first, second)] == expected_sse
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    guard_calls = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry", lambda req, attempt: guard_calls.append((req, attempt)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert guard_calls == []
    dispatcher._check_boundary(request, second_physical)
    assert guard_calls == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


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


@pytest.mark.parametrize("wave,require_pause", [(5, False), (11, True)])
def test_single_reviewed_origin_does_not_require_an_untraversed_previous_pointer(
        tmp_path, monkeypatch, wave, require_pause):
    scope = gpu.SCOPES[(887, wave)]
    run, release = tmp_path / "old-run", tmp_path / "old-release"
    parent = SimpleNamespace(predecessor={"model_identity": {"served_model": "locked"}})
    monkeypatch.setattr(gpu.retry.bounded, "_archived", lambda actual_run, actual_release, identity:
                        (parent, {"id": identity.contract}) if (actual_run, actual_release) == (run, release) else pytest.fail())
    fences = []
    monkeypatch.setattr(gpu.retry, "_retired", lambda *args, **kwargs: fences.append(kwargs["require_pause"]) or (run, run))
    monkeypatch.setattr(gpu, "_evidence", lambda *args: ({}, [], {}))
    once(run / "provider_model_lock.json", parent.predecessor["model_identity"])
    assert [record["contract"] for record in gpu._archives(run, release, scope)] == [parent]
    assert fences == [require_pause]


@pytest.mark.parametrize("seed,wave,identity,dispatcher,checkpoint", [
    (887, 5, gpu._887_R9_CONTINUATION, "dispatch_v13_887_r9_continuation", gpu._887_R12_CHECKPOINT),
    (887, 11, gpu._887_R20_CONTINUATION, "dispatch_v13_887_r20_continuation", gpu._887_R32_CHECKPOINT),
    (6101, 9, gpu._6101_R5_CONTINUATION, "dispatch_v13_6101_r5_continuation", gpu._6101_R6_CHECKPOINT),
])
def test_exact_source_only_origin_replays_each_fit_and_rejects_inserted_user_message(
        tmp_path, monkeypatch, seed, wave, identity, dispatcher, checkpoint):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import backend

    scope = gpu.SCOPES[(seed, wave)]
    first, final = scope.original_start_round, scope.round
    fitted_rounds = list(range(first + 1, final + 1))
    directory = tmp_path / f"seed_{seed}"
    initial = directory / f"round_{first:02d}"
    active = directory / f"round_{final:02d}"
    start = gpu.seed_runner.FrozenState(seed, first, "initial", {}, [], [],
                                       {"expected_observed": list(range(333 + 32 * first))})
    memory = {"rounds": list(range(first))}
    descriptor = {"round": first, "state_sha256": digest(asdict(start)), "memory_sha256": digest(memory)}
    formal = {"scope": "old"}
    parent = SimpleNamespace(digest=identity.contract, profile={"resume_boundaries": {str(seed): descriptor}},
                             predecessor={"dispatcher_module": dispatcher}, transport=object())
    once(initial / "source.json", asdict(start))
    once(initial / "boundary_memory.json", memory)
    once(directory / "boundary.json", {"formal_run": formal, **descriptor})
    states = {index: replace(start, round=index, checkpoint=(checkpoint if index == final else str(index)),
                             provenance={"expected_observed": list(range(333 + 32 * index))}) for index in fitted_rounds}
    for index in fitted_rounds:
        once(directory / f"round_{index:02d}" / "transition.json", {"round": index})
    once(active / "source.json", asdict(states[final]))
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
    assert reached == fitted_rounds and boundary == (states[final], {"memory_round": final}, final)
    assert proof[f"transition_{final:02d}"] == file_digest(active / "transition.json")
    once(initial / "boundary_user_message.json", {"content": "unbound"})
    with pytest.raises(PermissionError, match=f"R{first} continuation boundary"):
        gpu._reviewed_boundary(tmp_path, directory, active, parent, {}, scope, tmp_path)


def test_6101_wave11_binds_both_r8_pairs_and_new_guarded_budget(monkeypatch):
    old = gpu.SCOPES[(6101, 10)]
    scope = gpu.SCOPES[(6101, 11)]
    assert scope.archive[:-1] == old.archive and [len(item.attempts) for item in scope.archive] == [2, 2]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 8
    assert scope.original_boundary_physical_attempts == old.boundary_physical_attempts == 2
    assert scope.payload_sha256 == old.payload_sha256 and scope.required_sources == old.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == old.budget == {"first_logical_requests": 28, "maximum_logical_requests": 82,
                                         "maximum_physical_attempts": 164}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795715004.6630635
    assert "round_07/transition.json" in dict(old.fitted_prefix_files)
    assert "round_08/transition.json" in dict(old.fitted_prefix_files)
    first, second = scope.archive[-1].attempts
    assert first.identity == second.identity == gpu._6101_WAVE10 and first.request_id == second.request_id
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 25]
    assert all(item.progress["provider_error_code"] == "upstream_error" and item.progress["response_id_matches_transport_alias"]
               for item in (first, second))
    assert scope.archive[-1].replay_files["attempt_01.replay_link.json"] == (
        "aa1a28b08103f5353c6dfc0b0a256bf4079b917b693ca80ae74a9f40032f4436"
    )

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": second.attempt_id}]}
    dispatcher.original = {"binding": {}, "logical_index": 0, "payload": {"same": True}, "request_id": "old"}
    request = {**dispatcher.original, "request_id": "new"}
    seen = []
    monkeypatch.setattr(mac.RelayRetryDispatcher, "_check_boundary_retry", lambda _, req, started: seen.append((req, started)))
    retry = {"attempt": 1, "attempt_id": "new-second"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, retry)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, {"attempt": 0, "attempt_id": "new-first"})
    dispatcher._check_boundary(request, retry)
    assert seen == [(request, retry)]


@pytest.mark.parametrize("wave,eligible_unix,attempt_ids,terminal_shas", [
    (24, 1795731182.7983212,
     ["f02ae69a17f84c60948f9686d5e9c494", "390ae9ee72354d5dbeaa69559af7765c"],
     ["5e37437642c86d92ae986740a9d9b960526fdfb66357c308854bb6607f11004e",
      "9090345385a0b06099ecb705e2b030f95a7794879cbad377a197a94560b1a0ff"]),
    (25, 1795732175.9664137,
     ["07394a0e514f4d25b4af9676265111d0", "b4d19df55bc64adda0c7e946a472ded2"],
     ["7127ac8c06772478fc849f15e47b7a24723a9417acfe4c02b7177423d5e15b83",
      "4fc6c4ec55b423452946cebb494f6fe714c66f233811d12dad50f4acfd4b57a5"]),
    (27, 1795734534.2416294,
     ["2f0b0d68e2c746b29d99f3d995092673", "792e6c2a9c41483ab2ecdea69057b635"],
     ["9889f4c2c65df4ffd3c0ec07a94ca5c3bf5343a3a299a50f972068f3e60236ad",
      "6ba2c8bc2cfb91026ddacf70fc240e0db538d5d136dfc0e30f219732d2c554a2"]),
    (28, 1795735503.443264,
     ["c5f6778ed0f4440086cb94ee949de396", "bbb02e67ce07472fbeda336256431039"],
     ["c08b0ebe90c475d96ea48d4753f99b803cf7ceed310d51b7adecd362785ae16e",
      "ae92f952b59484fd9065446c7eaa2d95e9d5a1e86c6365b5ecde59d320e96441"]),
])
def test_6101_preserves_both_owned_failures_and_p1_guard(monkeypatch, wave, eligible_unix, attempt_ids, terminal_shas):
    previous = gpu.SCOPES[(6101, wave - 1)]
    scope = gpu.SCOPES[(6101, wave)]
    assert scope.archive[:-1] == previous.archive
    assert [len(item.attempts) for item in scope.archive] == [2, 2, 1, 1, 2, 2, 2, 2, 1, 1] + [2] * (wave - 19)
    assert scope.original_start_round == scope.round == scope.replay_start_round == 8
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 28, "maximum_logical_requests": 82,
                                              "maximum_physical_attempts": 164}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", eligible_unix)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == attempt_ids
    assert [item.terminal_file_sha256 for item in (first, second)] == terminal_shas
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_6101_wave26_preserves_both_owned_w25_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(6101, 25)]
    scope = gpu.SCOPES[(6101, 26)]
    assert scope.archive[:-1] == previous.archive
    assert [len(item.attempts) for item in scope.archive] == [2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 8
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 28, "maximum_logical_requests": 82,
                                              "maximum_physical_attempts": 164}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795733263.9176679)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["4316b74f72db4b33a21bc1829c71fc89", "cccb5a1806a948929b171a8d429bda3e"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "d05f4ecf35feb3e742ed8e307a64a51afdf0d838a66c69071bbc0fd64df0c299",
        "29c511950282e9c37e8b81240e0415e7938d38afb47ea81720ef8819ec6216ee",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_6101_wave23_preserves_both_owned_w22_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(6101, 22)]
    scope = gpu.SCOPES[(6101, 23)]
    assert scope.archive[:-1] == previous.archive
    assert [len(item.attempts) for item in scope.archive] == [2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 8
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 28, "maximum_logical_requests": 82,
                                              "maximum_physical_attempts": 164}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795730260.231831)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["d2275b39cbea4e1ab69ed8c951bf489d", "d6530498d3dc436e8b05b2dbd49f74cd"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "6d294a04c3b4599a7e47e966ff0dfd719963167fb4d9030248578cdbbef5e6a8",
        "b8a3797b80079c6d75de4fd998461a7ace2c0ccfbb3f034a10379d6c6a2b75a8",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 25]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_6101_wave22_preserves_both_owned_w21_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(6101, 21)]
    scope = gpu.SCOPES[(6101, 22)]
    assert scope.archive[:-1] == previous.archive
    assert [len(item.attempts) for item in scope.archive] == [2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 8
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 28, "maximum_logical_requests": 82,
                                              "maximum_physical_attempts": 164}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795729309.1526127)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["ed0e9772e418412fbb46cf4c8adad047", "d61ca71a6ba34c72815eda95a4902d22"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "dc2fa0c5acf9e6b4b67d63fec16d54598cbfa88eb8876c8248df336e0623f4b9",
        "2eaec0f27a87734206b9ecf09953df388a03e8fecf9de3938d428aa9353f56a1",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_6101_wave21_preserves_both_owned_w20_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(6101, 20)]
    scope = gpu.SCOPES[(6101, 21)]
    assert scope.archive[:-1] == previous.archive
    assert [len(item.attempts) for item in scope.archive] == [2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 8
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 28, "maximum_logical_requests": 82,
                                              "maximum_physical_attempts": 164}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795728250.404925)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["94ff77fc975a486bbc6123e0eb12a505", "7d19e415a09c4786858a353cab95ee30"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "76fa8a43fc6de6b13deb5c86eacf9f5ea464fd7e10bea1fbf3fb651b5d47592d",
        "479980cf2f24c2f29b490a0874ace6e807a41e4c3fbf026e4421c77cdf50c7d8",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_6101_wave20_preserves_both_owned_w19_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(6101, 19)]
    scope = gpu.SCOPES[(6101, 20)]
    assert scope.archive[:-1] == previous.archive
    assert [len(item.attempts) for item in scope.archive] == [2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 8
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 28, "maximum_logical_requests": 82,
                                              "maximum_physical_attempts": 164}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795727232.2344866)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["ab403f7d4a1641118e5f70e701ae33f3", "57d10f4bffd649b7aa2c50065d1efd78"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "8240975e329e0bc59daf21b1bda05287e71f4b61dfaad635225da802ee56f6e1",
        "157563bb66d5aa09c884b71f2a2287585a147915426ba5fc541845a3121d0dab",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [25, 25]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_6101_wave19_keeps_original_transport_and_bounds_both_fresh_r8_slots(monkeypatch):
    previous = gpu.SCOPES[(6101, 18)]
    scope = gpu.SCOPES[(6101, 19)]
    assert scope.archive[:-1] == previous.archive
    assert [len(item.attempts) for item in scope.archive] == [2, 2, 1, 1, 2, 2, 2, 2, 1, 1]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 8
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == {}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == {"first_logical_requests": 28, "maximum_logical_requests": 82,
                            "maximum_physical_attempts": 164}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795725649.7751193)
    receipt, = scope.archive[-1].attempts
    assert receipt.attempt_id == "a25420736ed14156b62c45aa63147857"
    assert receipt.terminal_file_sha256 == "3b18be441d41efa746520a1c3734a79d36a3b9b0422fe85fd7ebf13f81a239a6"
    assert receipt.failure == "response_id_mismatch" and receipt.replay_files is None
    assert receipt.progress["http_status"] == 200 and receipt.progress["sse_event_count"] == 41
    assert receipt.progress["provider_error_code"] == "upstream_error"
    assert receipt.progress["conflicting_event_type"] == "response.failed"
    assert receipt.progress["response_id_matches_transport_alias"] is True

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": receipt.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": receipt.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    guarded = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry", lambda req, reservation: guarded.append((req, reservation)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert not guarded
    dispatcher._check_boundary(request, second_physical)
    assert guarded == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_6101_wave18_carries_only_the_unused_r8_slot_after_idle_timeout(monkeypatch):
    previous = gpu.SCOPES[(6101, 17)]
    scope = gpu.SCOPES[(6101, 18)]
    assert scope.archive[:-1] == previous.archive
    assert [len(item.attempts) for item in scope.archive] == [2, 2, 1, 1, 2, 2, 2, 2, 1]
    assert scope.original_start_round == scope.round == 8 and scope.replay_start_round == 9
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 28, "maximum_logical_requests": 82,
                            "maximum_physical_attempts": 163}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795725073.1359603)
    receipt, = scope.archive[-1].attempts
    assert receipt.attempt_id == "c93f82325eef449d965c6cd07d7370d1"
    assert receipt.terminal_file_sha256 == "e644e1eb54cb464a4e4f449fb078e805f55f38d5a19a554565c45990d9440b00"
    assert receipt.failure == "timeout" and receipt.replay_files is None
    assert receipt.progress["http_status"] == 200 and receipt.progress["sse_event_count"] == 23
    assert receipt.progress["error_type"] == "sse_inactivity_timeout"
    assert receipt.progress["elapsed_seconds"] - receipt.progress["last_sse_activity_seconds"] >= 900

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": receipt.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": receipt.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {"attempt": 1, "attempt_id": "not-authorized"})


def test_6101_wave17_preserves_both_owned_w16_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(6101, 16)]
    scope = gpu.SCOPES[(6101, 17)]
    assert scope.archive[:-1] == previous.archive
    assert [len(item.attempts) for item in scope.archive] == [2, 2, 1, 1, 2, 2, 2, 2]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 8
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 28, "maximum_logical_requests": 82,
                                              "maximum_physical_attempts": 164}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795723499.4104714)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["f63231e3fdbe466b9672a8d670113637", "c17bc04f17594d9e9acf4b4ce7357d51"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "c0468ae4cc19065d4f8f6fc8c0e92e62524bcaec62eb2302a8b5733c2372eded",
        "f37adbe6ee937f470dc7b0ea450be5d8922d3a34688134c6612953681786e1f6",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 25]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_6101_wave16_preserves_both_owned_w15_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(6101, 15)]
    scope = gpu.SCOPES[(6101, 16)]
    assert scope.archive[:-1] == previous.archive
    assert [len(item.attempts) for item in scope.archive] == [2, 2, 1, 1, 2, 2, 2]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 8
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 28, "maximum_logical_requests": 82,
                                              "maximum_physical_attempts": 164}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795722288.4976568)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["39b38386aaf54b998a6c652f004dbbdc", "c9dea397edc7441490ab906504560983"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "aeb856024cab2c5e5a79e17973fdf34ae8186d63705558c238fc32046e8bed45",
        "17ab86b5f20098b8227442fe27eff12973bddaee3a34f13511b8d3f269a520bf",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 25]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_6101_wave15_keeps_l589_and_bounds_retry_after_two_failures_without_text(monkeypatch):
    previous = gpu.SCOPES[(6101, 14)]
    scope = gpu.SCOPES[(6101, 15)]
    assert scope.archive[:-1] == previous.archive
    assert [len(item.attempts) for item in scope.archive] == [2, 2, 1, 1, 2, 2]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 8
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 28, "maximum_logical_requests": 82,
                                              "maximum_physical_attempts": 164}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795721303.602687)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == [
        "28a3cea9b05d46acb43d47c85fb1f019", "88e82c797f1647ef9bc5b30796f4ce44",
    ]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "d06cba12bacec23e9cb9119101094016068b0c917f3214163fe5f3b43ce4222d",
        "e575b6d466dd59367264e0199e0604f317fdc3d70434714517deec1048601994",
    ]
    for item in (first, second):
        assert item.progress["http_status"] == 200 and item.progress["sse_event_count"] == 41
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": second.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_6101_wave14_keeps_l589_and_bounds_retry_after_two_failures_without_text(monkeypatch):
    previous = gpu.SCOPES[(6101, 13)]
    scope = gpu.SCOPES[(6101, 14)]
    assert scope.archive[:-1] == previous.archive
    assert [len(item.attempts) for item in scope.archive] == [2, 2, 1, 1, 2]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 8
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 28, "maximum_logical_requests": 82,
                                              "maximum_physical_attempts": 164}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795719109.4587774)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == [
        "5838b6000a814025b2d1d852f34fdbec", "3c292dd317494e3c819dfd4d2d91e11f",
    ]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "643680653c93985cc863f6449c531baaf78c8a4dbcfd336b1af681c4e34f4310",
        "1f9916f971363cc387bbbe876cfc808b8392dfc1c43c3b589c12c5fbe81b7ab5",
    ]
    for item in (first, second):
        assert item.progress["http_status"] == 200 and item.progress["sse_event_count"] == 41
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": second.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_6101_wave13_keeps_l589_and_a_bounded_second_slot_after_exhaustion(monkeypatch):
    old = gpu.SCOPES[(6101, 12)]
    scope = gpu.SCOPES[(6101, 13)]
    assert scope.archive[:-1] == old.archive and [len(item.attempts) for item in scope.archive] == [2, 2, 1, 1]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 8
    assert scope.original_boundary_physical_attempts == old.boundary_physical_attempts == 1
    assert scope.payload_sha256 == old.payload_sha256 and scope.required_sources == old.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == {"first_logical_requests": 28, "maximum_logical_requests": 82, "maximum_physical_attempts": 164}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795718134.8001642
    review, = scope.archive[-1].attempts
    assert review.attempt_id == "4931381e0e68465ba95d5bde6d7250b5" and review.failure == "response_id_mismatch"
    assert review.progress["http_status"] == 200 and review.progress["sse_event_count"] == 41
    assert review.progress["provider_error_code"] == "upstream_error" and review.progress["response_id_matches_transport_alias"]

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": review.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": review.request_id}
    seen = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry", lambda request, started: seen.append((request, started)))
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    p0 = {"attempt": 0, "attempt_id": "new-first"}
    p1 = {"attempt": 1, "attempt_id": "new-second"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - .01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, p0)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, p0)
    assert not seen
    dispatcher._check_boundary(request, p1)
    assert seen == [(request, p1)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**p1, "attempt": 2})


def test_6101_wave12_carries_only_the_unused_r8_slot_after_the_inactivity_failure():
    previous = gpu.SCOPES[(6101, 11)]
    scope = gpu.SCOPES[(6101, 12)]
    receipt = scope.archive[-1]
    assert scope.archive[:-1] == previous.archive
    assert scope.original_start_round == scope.round == 8 and scope.replay_start_round == 9
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.boundary_physical_attempts == 2 - len(receipt.attempts) == 1
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 28, "maximum_logical_requests": 82,
                            "maximum_physical_attempts": 163}
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795717172.261229)
    assert (receipt.identity, receipt.attempt_id, receipt.failure) == (
        gpu._6101_WAVE11, "5b0bd7c2d18c4d19969336f0e21bc48b", "timeout",
    )
    assert not receipt.earlier and receipt.replay_files is None
    assert receipt.progress["http_status"] == 200 and receipt.progress["sse_event_count"] == 11
    assert receipt.progress["error_type"] == "sse_inactivity_timeout"


def test_6101_wave10_rechecks_two_cuda_fits_and_both_closed_r8_slots(monkeypatch):
    old = gpu.SCOPES[(6101, 9)]
    scope = gpu.SCOPES[(6101, 10)]
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (6, 8, 8)
    assert scope.original_boundary_physical_attempts == old.boundary_physical_attempts == 1
    assert scope.required_sources == old.required_sources
    prefix = dict(scope.fitted_prefix_files)
    assert prefix["round_07/transition.json"] == "56631c6ef5a417125f6def6f8a4492ec3dd4897b15d1c7e381b57ee66fe772a5"
    assert prefix["round_08/transition.json"] == "b857ed3cb6c60ad991758756502374a320e5e67b19b07ce31a636c69b57b99c8"
    assert prefix["round_06/boundary_user_message.json"] and prefix["round_08/authoritative_packet.json"]
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == {"first_logical_requests": 28, "maximum_logical_requests": 82, "maximum_physical_attempts": 164}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795713600.7855213
    review, = scope.archive
    first, second = review.attempts
    assert first.identity == second.identity == gpu._6101_WAVE09 and first.request_id == second.request_id
    assert [item.progress["sse_event_count"] for item in (first, second)] == [24, 25]
    assert all(item.progress["provider_error_code"] == "upstream_error" and item.progress["response_id_matches_transport_alias"]
               for item in (first, second))
    assert len(review.replay_files) == 3

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": review.attempt_id}]}
    dispatcher.original = {"binding": {}, "logical_index": 0, "payload": {"same": True}, "request_id": "previous"}
    request = {**dispatcher.original, "request_id": "fresh"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    guarded = []
    monkeypatch.setattr(mac.RelayRetryDispatcher, "_check_boundary_retry", lambda _, req, started: guarded.append((req, started)))
    second_attempt = {"attempt": 1, "attempt_id": "new-second"}
    dispatcher._check_boundary(request, {"attempt": 0, "attempt_id": "new-first"})
    dispatcher._check_boundary(request, second_attempt)
    assert guarded == [(request, second_attempt)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {"attempt": 2, "attempt_id": "new-third"})


def test_6101_wave09_binds_r5_fit_and_allows_only_one_remaining_r6_slot(monkeypatch):
    scope = gpu.SCOPES[(6101, 9)]
    assert scope.round == 6 and scope.original_start_round == 5 and scope.replay_start_round == 7
    assert scope.original_boundary_physical_attempts == 2
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 30, "maximum_logical_requests": 88, "maximum_physical_attempts": 175}
    assert scope.not_before_unix == 1795710450.8833244
    assert scope.required_sources == gpu.SCOPES[(6101, 8)].required_sources
    assert "round_06/transition.json" in dict(scope.fitted_prefix_files)
    receipt, = scope.archive[-1].attempts
    assert receipt.identity == gpu._6101_R5_CONTINUATION
    assert receipt.failure == "connection_interrupted" and receipt.pre_post_connect_failure is False
    assert receipt.progress["network_phases"]["send_body"]["state"] == "complete"
    assert receipt.progress["network_phases"]["receive_headers"]["state"] == "failed"
    assert {"http_status", "response_id", "provider_request_id"}.isdisjoint(receipt.progress)
    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    monkeypatch.setattr(mac.RelayRetryDispatcher, "_check_boundary", lambda *_: None)
    request = {"payload_sha256": scope.payload_sha256}
    fresh = {"attempt": 0, "attempt_id": "n" * 32}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, fresh)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, fresh)
    for rejected in ({**fresh, "attempt": 1}, {**fresh, "attempt_id": receipt.attempt_id}):
        with pytest.raises(PermissionError, match="signed physical attempts"):
            dispatcher._check_boundary(request, rejected)


def test_2357_wave12_pins_both_verified_relay_failures_at_l493():
    old = gpu.SCOPES[(2357, 11)]
    scope = gpu.SCOPES[(2357, 12)]
    assert scope.archive[:-1] == old.archive and [len(item.attempts) for item in scope.archive] == [2, 2, 1, 1, 2]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == old.boundary_physical_attempts == 2
    assert scope.payload_sha256 == old.payload_sha256 and scope.required_sources == old.required_sources
    assert scope.recovery_budget == old.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == old.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                         "maximum_physical_attempts": 182}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795713361.660755
    first, second = scope.archive[-1].attempts
    assert first.identity == second.identity == gpu._2357_WAVE11 and first.request_id == second.request_id
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 25]
    assert all(item.failure == "response_id_mismatch" and item.progress["http_status"] == 200
               and item.progress["provider_error_code"] == "upstream_error"
               and item.progress["response_id_matches_transport_alias"] for item in (first, second))
    assert scope.archive[-1].replay_files["attempt_01.replay_link.json"] == (
        "5717bfca88e6f0297cff08de37a963734045718a50fe705f4f801e95837f8d6f"
    )


def test_2357_wave15_keeps_exact_r5_and_reviews_both_closed_slots(monkeypatch):
    old = gpu.SCOPES[(2357, 14)]
    scope = gpu.SCOPES[(2357, 15)]
    assert scope.archive[:-1] == old.archive
    assert [len(item.attempts) for item in scope.archive] == [2, 2, 1, 1, 2, 2, 2, 2]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == old.boundary_physical_attempts == 2
    assert scope.payload_sha256 == old.payload_sha256 and scope.required_sources == old.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == old.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                         "maximum_physical_attempts": 182}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795718762.0906043
    review = scope.archive[-1]
    first, second = review.attempts
    assert first.identity == second.identity == gpu._2357_WAVE14 and first.request_id == second.request_id
    assert [item.progress["sse_event_count"] for item in review.attempts] == [23, 25]
    assert all(item.failure == "response_id_mismatch" and item.progress["provider_error_code"] == "upstream_error"
               and item.progress["response_id_matches_transport_alias"] for item in review.attempts)
    assert review.replay_files["attempt_01.replay_link.json"] == "811c37ed8a98533b809696e54486a7c3ad987c735eaa8dae3262068449769c42"

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": first.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": first.request_id}
    checked = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry", lambda request, reservation: checked.append((request, reservation)))
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    p0 = {"attempt": 0, "attempt_id": "fresh-first"}
    p1 = {"attempt": 1, "attempt_id": "fresh-second"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - .01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, p0)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, p0)
    assert not checked
    dispatcher._check_boundary(request, p1)
    assert checked == [(request, p1)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**p1, "attempt": 2})


@pytest.mark.parametrize("wave,eligible_unix,attempt_ids,terminal_shas,counts", [
    (24, 1795730986.2878697,
     ["c08afd5215ad43f7bea313fd70b0b32c", "188beb8c58bb4483b724f55ae7b8a4a7"],
     ["e8cc17354c9aa2ee79a3ad97e62562718f770815388dbc0e77d99d0a619803a9",
      "cbb859d09251fcaacff360a360b59f5b4ae25224cdf26b3a97245ba550dfbcac"], [25, 41]),
    (25, 1795732017.8086386,
     ["fd16a4a08e484174979bb12299314010", "c938c9841320450b888100333327d8ee"],
     ["b82f75d3e6c1368c049acb4e183efb7e5da5f2ab18a775976b62fbfddc47c4b9",
      "3caa6ad484a5a891f317add9738133d6938f9e8a32e5d3c775f020e06a653108"], [41, 41]),
])
def test_2357_preserves_header_deadline_and_exact_owned_failure_pair(
        monkeypatch, wave, eligible_unix, attempt_ids, terminal_shas, counts):
    previous = gpu.SCOPES[(2357, wave - 1)]
    scope = gpu.SCOPES[(2357, wave)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == (32 if wave == 24 else 34)
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == previous.header_deadline_profile == {"response_headers_seconds": 300}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", eligible_unix)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == attempt_ids
    assert [item.terminal_file_sha256 for item in (first, second)] == terminal_shas
    assert [item.progress["sse_event_count"] for item in (first, second)] == counts
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


@pytest.mark.parametrize("wave,total_archived,eligible_unix,attempt_ids,terminal_shas", [
    (29, 42, 1795736317.8906271,
     ["54682c14ff5a48dd904957b6b36272be", "a23d5a96ea9942d983bdb76b11234c9e"],
     ["5e1dbae6dd15933e0ec02f7d7ea37bc305281e62b0213f1243ed3e92c81edafe",
      "c8d52e745c6b9ec40ba06f47392e9b545aeca9f61fd41f6199e52f7e51c462cf"]),
    (30, 44, 1795737316.2258618,
     ["9269d180774d4e9490a49687a1745253", "01f17f4d387a4b55b8df14c7cc20faf2"],
     ["ef303e5b6a43d2e6fcf4eb751b82dcfeb3a2d4d23b98747e75d631e84d55889f",
      "76a2c187a3c2a854ea6ece2cc37dec4af206344f2b3a9d4d06b218e318f6986f"]),
])
def test_2357_latest_scopes_preserve_header_deadline_and_owned_failures(monkeypatch, wave, total_archived, eligible_unix,
                                                                     attempt_ids, terminal_shas):
    previous = gpu.SCOPES[(2357, wave - 1)]
    scope = gpu.SCOPES[(2357, wave)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == total_archived
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == previous.header_deadline_profile == {"response_headers_seconds": 300}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", eligible_unix)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == attempt_ids
    assert [item.terminal_file_sha256 for item in (first, second)] == terminal_shas
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_2357_wave31_preserves_header_deadline_and_exact_owned_w30_failures(monkeypatch):
    previous = gpu.SCOPES[(2357, 30)]
    scope = gpu.SCOPES[(2357, 31)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 46
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == previous.header_deadline_profile == {"response_headers_seconds": 300}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795739518.5515773)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["0f819345a2d54c579c14ae79cb126521", "91de0b3336a94cfb989c583d5a53531b"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "d5ba190b1f6a94d3e8c3ccf17866cd69478ff47da6611a4e1c42ef0d796ba515",
        "80eaebf17f56c1a52060d5964af2b0704a6f07fb9cd9b4fc8c14bb77a5ff9142",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_2357_wave32_preserves_header_deadline_and_exact_owned_w31_failures(monkeypatch):
    previous = gpu.SCOPES[(2357, 31)]
    scope = gpu.SCOPES[(2357, 32)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 48
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == previous.header_deadline_profile == {"response_headers_seconds": 300}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795740470.5931585)
    assert scope.archive[-1].replay_files == {
        "attempt_00.replay_authorization.json": "746c59d09c1c384b26f04ca3f2e7abe1c31edc3c794213f2d75dda8411679a19",
        "attempt_00.replay.terminal.json": "624de908783546e25dd51ac53e923a9b1910172e46bffb57d247876e4d8351f8",
        "attempt_01.replay_link.json": "8d1c692a49bc3d707f82bd65570ec52bb5eddc78e57294de045d65ae0a615792",
    }
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["bd1e5040860e4882a2f170a92daf52d1", "7c500315072d4486a136a0420acf83fe"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "4302a6ac9b775ad26943fb08dbbf59d07de1f69c7b0684889a0a8c371468634e",
        "ad9780e4a1de6bbfd73fab3d755b7c760b81d6b257cf50bd0ee7ca0ad67f072f",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_2357_wave33_preserves_header_deadline_and_exact_owned_w32_failures(monkeypatch):
    previous = gpu.SCOPES[(2357, 32)]
    scope = gpu.SCOPES[(2357, 33)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 50
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == previous.header_deadline_profile == {"response_headers_seconds": 300}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795741753.9078317)
    assert scope.archive[-1].replay_files == {
        "attempt_00.replay_authorization.json": "84e1d3f482153d24143982e3f01c5c57daa5380f43b6fba4b9b3b0ba1553e4cb",
        "attempt_00.replay.terminal.json": "702bd222a288b8785b3efbefd7ce0f347685fc127872c831d3596a7cea86a0a7",
        "attempt_01.replay_link.json": "7b9c62794b41b9bb6c66035814ce8bc97d1efcd82aa710e2ed61a2c62d8fb783",
    }
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["751f16ab78dd4ba38bd6fa698155b210", "1ef5a247e15f47e6abb6ca2b05a9b0b5"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "3f0dfcfc5436b3910d729d79d32d7f2f3adad758786c401ce8884941ec624f10",
        "bcef9196238e92645204433f43b0a5993e2612106b48ea491fa03df620964fb3",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_2357_wave34_preserves_header_deadline_and_exact_owned_w33_failures(monkeypatch):
    previous = gpu.SCOPES[(2357, 33)]
    scope = gpu.SCOPES[(2357, 34)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 52
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == previous.header_deadline_profile == {"response_headers_seconds": 300}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795742775.9691432)
    assert scope.archive[-1].replay_files == {
        "attempt_00.replay_authorization.json": "af319a563529c944712160533c93154d2060949d56845d5a5cb879ac8b8dba99",
        "attempt_00.replay.terminal.json": "b5ef20af999c8e0bdba9e773d9970500a775c55d6ce7a22ae7b5f85d1519856b",
        "attempt_01.replay_link.json": "3ca1ae319152585fe92388b601e8b6df9f33b0676e8d507dcf7bfc699440f8eb",
    }
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["07bc264d90bb4032a4c2381b7574bf63", "e4eee70e367d4552a9f03e5deb2ddfdb"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "6875ef0825e6c141ebdb93711c522b1aee3ffad0e7b290085fe1b1de83b56699",
        "25527520318af897e4ff1b033aff690e4c8aaabfcd9cd5f481105f93b10439e2",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [25, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_2357_wave35_preserves_header_deadline_and_exact_owned_w34_failures(monkeypatch):
    previous = gpu.SCOPES[(2357, 34)]
    scope = gpu.SCOPES[(2357, 35)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 54
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == previous.header_deadline_profile == {"response_headers_seconds": 300}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795743842.364679)
    assert scope.archive[-1].replay_files == {
        "attempt_00.replay_authorization.json": "5f0ee9cd0d37d61a6ff54b2c40aa4075d0b821709ef0af5ea418083b984f4424",
        "attempt_00.replay.terminal.json": "51f38f53ce392e9e01ca4516262b3bb2d0fdd7d6c8b222b56870fc3f3925a44b",
        "attempt_01.replay_link.json": "c11ae946cb6e34256e20fa80d3d5e2a1ef93e4898c2752764afbb8e174d5c3a3",
    }
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["2d9f40d0ff3546cd90d1e466f808147e", "2cbb4890ad2f46a087087267abc309ca"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "0119f396389d332e9c07e95c6577c4bbc67cc19cf56a032721cbee3d18fb7f52",
        "f1b978b29aded3becb5d6b3934e54b304c4aa24a7d40480db42c0091bde8ea1c",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 25]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_2357_wave36_preserves_header_deadline_and_exact_owned_w35_failures(monkeypatch):
    previous = gpu.SCOPES[(2357, 35)]
    scope = gpu.SCOPES[(2357, 36)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 56
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == previous.header_deadline_profile == {"response_headers_seconds": 300}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795745056.081053)
    assert scope.archive[-1].replay_files == {
        "attempt_00.replay_authorization.json": "296b677cd5b8662b31bef7fe1c00ea8053a2150832cd1b0816d2fa200148450b",
        "attempt_00.replay.terminal.json": "5a98f08e9a41d763147d6c9a5e715ee0819dce8e1e62776b6eebc0e88a00c05b",
        "attempt_01.replay_link.json": "3c32960082bdf7353cbe4f0b19ef2ed219346877d3bd509edf136cbe861578a9",
    }
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["0326abdb4ce940f9b7f2f09451cf1fca", "88572a161b524c3993815616b364eb6e"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "ea25b96a996459168f286e8c041354a84ac855004e91b63924cfd1b77117ac83",
        "6edd66bc0464ba5583706b6bab27802af6f290cbaf3c93c89518f26c423c292a",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_2357_wave37_preserves_header_deadline_and_exact_owned_w36_failures(monkeypatch):
    previous = gpu.SCOPES[(2357, 36)]
    scope = gpu.SCOPES[(2357, 37)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 58
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == previous.header_deadline_profile == {"response_headers_seconds": 300}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795746453.528209)
    assert scope.archive[-1].replay_files == {
        "attempt_00.replay_authorization.json": "59783334d8becef24035e8c300cf672c6b6fd4fcbf1811f4f84971af105cd568",
        "attempt_00.replay.terminal.json": "613de00db05e70862c447f75df0b2a42f85e5f4daceff0caec83f04da00e1ea0",
        "attempt_01.replay_link.json": "155e65a10c803622a7bb7c8d41d33bb09d89c05a04912b6016a40fcf390d70d2",
    }
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["f5c6d0d5491d41a9a1382869471edb15", "a3ac66b2f6714980abddba846dabb4a9"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "7cd0ee0b2e465ef04626fdff73e9c0725e5f2e8292972610c15e4b7b18298a1e",
        "b1a14616d6e1270668d203d9f54848a17a8dc6f2c2f3e8f087e40eaccba6a06f",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [31, 29]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_2357_wave40_preserves_only_the_unposted_slot_and_exact_readiness(tmp_path, monkeypatch):
    previous, scope = gpu.SCOPES[(2357, 39)], gpu.SCOPES[(2357, 40)]
    assert scope.archive[:-1] == previous.archive
    review, = scope.archive[-1].attempts
    assert len(scope.archive) == 33 and sum(len(item.attempts) for item in scope.archive) == 61
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (5, 5, 6)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == previous.header_deadline_profile == {"response_headers_seconds": 300}
    assert scope.recovery_budget == previous.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 181}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795750824.562045)
    assert review.terminal_file_sha256 == "0bf8391aa8d8566ba85a8718faca017e453b3ef9791d18f6656ed7e67bd83265"
    assert review.receipt_provider_flags == (False, False) and review.failure == "connect_error"
    assert not review.pre_post_connect_failure and review.replay_files is None
    assert replace(review, job_sha256="a" * 64).receipt_provider_flags == (True, True)
    assert gpu._887_W23_P1.receipt_provider_flags == (False, False)
    assert gpu._12203_WAVE12_PRE_POST_CONNECT.receipt_provider_flags == (True, False)
    progress = review.progress
    assert progress["event_type"] == "ipv4_preflight_no_headers"
    assert all(phase["state"] == "failed" for phase in progress["network_phases"].values())
    assert not {"http_status", "sse_event_count", "provider_request_id", "response_id"}.intersection(progress)

    for index, retry_unix in enumerate((1795750429.859632, 1795750520.775499, 1795750609.924542), 1):
        once(tmp_path / f"readiness_{index:02d}.json", {
            "attempt_id": review.attempt_id, "job_sha256": review.job_sha256, "retry_unix": retry_unix, "window": index,
        })
    expected = review.unposted_readiness_files
    assert len(expected) == 3 and mac._unposted_readiness(tmp_path, progress, expected)
    assert not mac._unposted_readiness(tmp_path, {**progress, "sse_event_count": 1}, expected)
    assert not mac._unposted_readiness(tmp_path, {**progress, "first_phase_seconds": {"request_dispatched": 2}}, expected)
    started_marker = tmp_path / "started.json"
    once(started_marker, {})
    assert not mac._unposted_readiness(tmp_path, progress, expected)
    started_marker.unlink()
    (tmp_path / "readiness_03.json").unlink()
    assert not mac._unposted_readiness(tmp_path, progress, expected)

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": review.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": review.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first = {"attempt": 0, "attempt_id": "new-first"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first)
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**first, "attempt": 1})


def test_2357_wave39_adds_only_one_new_attempt_after_exhausted_single_timeout(monkeypatch):
    previous = gpu.SCOPES[(2357, 38)]
    scope = gpu.SCOPES[(2357, 39)]
    assert scope.archive[:-1] == previous.archive
    assert len(scope.archive) == 32 and sum(len(item.attempts) for item in scope.archive) == 60
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (5, 5, 6)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == previous.header_deadline_profile == {"response_headers_seconds": 300}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 181}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795750122.0460975)
    review, = scope.archive[-1].attempts
    assert review.terminal_file_sha256 == "52535424e39717f2e6ef4918dce68b018f99651d2ddc14b0f951f4185da364ae"
    assert scope.archive[-1].replay_files is None
    assert review.failure == "timeout" and review.progress["error_type"] == "sse_inactivity_timeout"
    assert review.progress["http_status"] == 200 and review.progress["sse_event_count"] == 7
    assert not {"provider_error_code", "conflicting_event_type", "response_id_matches_transport_alias",
                 "first_text_seconds", "complete_terminal_captured"} & review.progress.keys()
    assert not is_sealed_relay_failure({"error": review.failure})

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": review.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": review.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    started = {"attempt": 0, "attempt_id": "new-first"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, started)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, started)
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**started, "attempt": 1})


def test_2357_wave38_carries_only_the_unused_slot_after_uploaded_request_header_timeout(monkeypatch):
    previous = gpu.SCOPES[(2357, 37)]
    scope = gpu.SCOPES[(2357, 38)]
    assert scope.archive[:-1] == previous.archive
    review, = scope.archive[-1].attempts
    assert sum(len(item.attempts) for item in scope.archive) == 59
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (5, 5, 6)
    assert scope.boundary_physical_attempts + len(scope.archive[-1].attempts) == scope.original_boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == previous.header_deadline_profile == {"response_headers_seconds": 300}
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 181}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795747408.5560722)
    assert review.attempt_id == "9b40cff8b2104675b5957347194d5cb3"
    assert review.terminal_file_sha256 == "ef831488691798f95d3d26db17d28b44c6434a914d9520ad397f8f4e73d8b1de"
    assert review.failure == "response_headers_timeout" and not review.pre_post_connect_failure
    assert scope.archive[-1].replay_files is None
    progress = review.progress
    assert progress["error_type"] == "response_headers_timeout" and progress["event_type"] == "awaiting_response_headers"
    assert progress["network_phases"]["send_body"]["state"] == "complete"
    assert progress["network_phases"]["send_body"]["observed_seconds"] == 7.18993837479502
    assert progress["network_phases"]["receive_headers"]["state"] == "failed"
    assert progress["network_phases"]["receive_headers"]["duration_seconds"] >= 300
    assert not {"http_status", "sse_event_count", "provider_request_id", "response_id", "complete_terminal_captured"}.intersection(progress)
    assert not is_sealed_relay_failure({"error": review.failure})

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": review.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": review.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first = {"attempt": 0, "attempt_id": "new-first"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first)
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**first, "attempt": 1})


def test_2357_wave28_preserves_header_deadline_and_exact_owned_w27_failures(monkeypatch):
    previous = gpu.SCOPES[(2357, 27)]
    scope = gpu.SCOPES[(2357, 28)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 40
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == previous.header_deadline_profile == {"response_headers_seconds": 300}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795735310.6910114)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["b5becb27b73d441abd542893c7ebe1fb", "22c3be0ae6a043ae8767b47fc0cc4f10"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "4d2d3fceae85f1590b082031a1f96de5bc06238615881563af8e39e5a659325d",
        "d1b95c66936cca86341faafa872b278aab4c6b14aaaa7c11ec47e3efbf063616",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_2357_wave27_preserves_header_deadline_and_exact_owned_w26_failures(monkeypatch):
    previous = gpu.SCOPES[(2357, 26)]
    scope = gpu.SCOPES[(2357, 27)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 38
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == previous.header_deadline_profile == {"response_headers_seconds": 300}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795734273.7636313)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["0ccb81fe134c44f0bddd832be74b011a", "9be434ac07fb46c695498bce09632916"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "9aea8facae60767aca52e842147a17a5f3df3afc8103850edfadb2e4413a2591",
        "9755f371d1e4e2c6144e129cc4f94b93edd93c8e431b5d8dfc233694401a2ac8",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_2357_wave26_preserves_header_deadline_and_exact_owned_w25_failures(monkeypatch):
    previous = gpu.SCOPES[(2357, 25)]
    scope = gpu.SCOPES[(2357, 26)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 36
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == previous.header_deadline_profile == {"response_headers_seconds": 300}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795733165.5820577)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["54e6018e7f3f4a69a085dfc19068f24a", "097a914930474894a4fe5ca1de033dc1"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "3484e73db6357940fc2f7c2bb7d8346021edb9bcf5cc5c46404d8c311c4d720c",
        "4faaf9566190c9de4c1dad47ec76198d64c104a4a07898fd9c153c3d05a564d9",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_2357_wave23_preserves_header_deadline_and_exact_owned_w22_failures(monkeypatch):
    previous = gpu.SCOPES[(2357, 22)]
    scope = gpu.SCOPES[(2357, 23)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 30
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == previous.header_deadline_profile == {"response_headers_seconds": 300}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795729955.532721)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["7497055372d545b4baeb0c3ee3d6705d", "165867cccd6b46678fb7d6238adfb53e"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "e3a394aa2503e3510b084d9c0a7a0135d92924f3fb0de92ebd929c10dc62b777",
        "952bc63f4deacdfaad9f7f310ed2c593020ea10def642325ce7477c7100ed486",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 25]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_2357_wave22_preserves_header_deadline_and_exact_owned_w21_failures(monkeypatch):
    previous = gpu.SCOPES[(2357, 21)]
    scope = gpu.SCOPES[(2357, 22)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 28
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == previous.header_deadline_profile == {"response_headers_seconds": 300}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795728993.6545773)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["61ca8f0d3b024d18a30097d130d3dd52", "8c97ccc80ce94f7cba136bfc8af99824"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "29bd0e510d1b94e7a444b3438f19e3b687ca21d2705481ee887dfa9249c0b43e",
        "ff4ad99d1554d7c7cbf9392cb99f630b2fdfa0e7411ee5cbd09a5c8542931bc1",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [25, 25]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_2357_wave21_preserves_header_deadline_and_exact_owned_w20_failures(monkeypatch):
    previous = gpu.SCOPES[(2357, 20)]
    scope = gpu.SCOPES[(2357, 21)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 26
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == previous.header_deadline_profile == {"response_headers_seconds": 300}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795728051.4221308)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["d3d52223789440a9bfa650ce7df19d34", "5732ea68547d4b0798f601e648609dee"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "574059ad3a5a86f12d6b906bf9d80466db808f95fc80e44dadc7052995ca8b63",
        "ba5175cd5179686b82d0e387ed9ad5a902588f28c31d6c87683e13e2b72b6806",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_2357_wave20_preserves_header_deadline_and_exact_owned_w19_failures(monkeypatch):
    previous = gpu.SCOPES[(2357, 19)]
    scope = gpu.SCOPES[(2357, 20)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 24
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.header_deadline_profile == previous.header_deadline_profile == {"response_headers_seconds": 300}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795727020.8317)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["6babdfd158a0464fa5ed42e9a2fe50bd", "b7cb05a627f84af6bcf8315e3fc0a2c7"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "192af9f7a98e0ea3f758921145b97f73c84810a896f2b130bb6aa438a2ab1c63",
        "41ffc8a8d97689087fe32b7daf0b4f8e879f343b19863c65a32c1f58dcc54bf1",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [25, 25]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_2357_wave19_binds_w18_post_upload_timeout_and_a_guarded_same_input_resume():
    previous = gpu.SCOPES[(2357, 18)]
    scope = gpu.SCOPES[(2357, 19)]
    assert scope.archive[:-1] == previous.archive
    assert sum(len(item.attempts) for item in scope.archive) == 22
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795725838.6105614)
    assert scope.header_deadline_profile == {"response_headers_seconds": 300}
    assert dict(scope.required_sources) == {**dict(previous.required_sources), **gpu.HEADER_DEADLINE_SOURCES}
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "b9682c0ac55fe8ef819b8ccf415575404dd3f961468849cca35335f6fd0b45a7",
        "eb5ed8a5bcc4a0d830972f13f7a53db4169125805b099c9f3d16ba46a2bebfcb",
    ]
    assert first.progress["sse_event_count"] == 41 and first.progress["provider_error_code"] == "upstream_error"
    assert second.failure == "timeout" and second.progress["error_type"] == "ReadTimeout"
    phases = second.progress["network_phases"]
    assert phases["send_body"]["state"] == "complete" and phases["receive_headers"]["state"] == "failed"
    assert phases["receive_headers"]["duration_seconds"] >= 1800
    assert {"http_status", "response_id", "sse_event_count", "first_text_seconds"}.isdisjoint(second.progress)


def test_2357_wave18_preserves_both_owned_w17_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(2357, 17)]
    scope = gpu.SCOPES[(2357, 18)]
    assert scope.archive[:-1] == previous.archive
    assert [len(item.attempts) for item in scope.archive] == [2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795723270.9339292)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["90186ecb2e534a0c89e883969ad487d3", "749dd3bcaf984c778355742ea5a8937e"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "5504e96e4ad6699a39ce568b006b06c37aff849ea206b3fccea599f1f85cad62",
        "ed30f2de9cf8e0567446f8652306f0e22551efa4ebef49ecd36a4dbc71d722ed",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [41, 23]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_2357_wave17_preserves_both_owned_w16_failures_and_p1_guard(monkeypatch):
    previous = gpu.SCOPES[(2357, 16)]
    scope = gpu.SCOPES[(2357, 17)]
    assert scope.archive[:-1] == previous.archive
    assert [len(item.attempts) for item in scope.archive] == [2, 2, 1, 1, 2, 2, 2, 2, 2, 2]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795722006.3807504)
    first, second = scope.archive[-1].attempts
    assert first.request_id == second.request_id
    assert [item.attempt_id for item in (first, second)] == ["1cded70dd18448a4b1e4338f03c05c51", "1f7361cc146f48fbad269764e61a17b8"]
    assert [item.terminal_file_sha256 for item in (first, second)] == [
        "9e008bdaad6940eb8a612c335036c44e4369ccf227f3870ba0fe23996b1aa66f",
        "b990236af5b0b825d2d6d7e197f40a45bec236e944fbe2b8319919f35e862fe0",
    ]
    assert [item.progress["sse_event_count"] for item in (first, second)] == [23, 41]
    for item in (first, second):
        assert item.progress["http_status"] == 200
        assert item.progress["provider_error_code"] == "upstream_error"
        assert item.progress["response_id_matches_transport_alias"] is True
        assert item.progress["conflicting_event_type"] == "response.failed"
        assert set(item.progress["sse_milestones"]) == {
            "response.created", "response.in_progress", "response.output_item.added", "response.output_item.done",
        }

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in (first, second)]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": second.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    first_physical = {"attempt": 0, "attempt_id": "new-first"}
    second_physical = {"attempt": 1, "attempt_id": "new-second"}
    admitted_replays = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry",
                        lambda req, started: admitted_replays.append((req, started)))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, first_physical)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, first_physical)
    assert admitted_replays == []
    dispatcher._check_boundary(request, second_physical)
    assert admitted_replays == [(request, second_physical)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**second_physical, "attempt": 2})


def test_2357_wave16_retains_both_anchored_w15_failures_and_signed_retry_guard(monkeypatch):
    previous = gpu.SCOPES[(2357, 15)]
    scope = gpu.SCOPES[(2357, 16)]
    review = scope.archive[-1]
    assert scope.archive[:-1] == previous.archive
    assert [len(item.attempts) for item in scope.archive] == [2, 2, 1, 1, 2, 2, 2, 2, 2]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795719784.1234941
    assert [item.attempt_id for item in review.attempts] == [
        "8e36b8d104ca4862997060e83c02b4b9", "6b1a6ffe4864450abae7e02daed30584",
    ]
    assert all(item.identity == gpu._2357_WAVE15 and item.request_id == gpu._2357_WAVE15_REQUEST
               and item.failure == "response_id_mismatch" and item.progress["http_status"] == 200
               and item.progress["sse_event_count"] == 41 and item.progress["provider_error_code"] == "upstream_error"
               and item.progress["response_id_matches_transport_alias"] for item in review.attempts)
    assert review.replay_files["attempt_01.replay_link.json"] == (
        "a9fb5a334645c5debd1c7854b260dfca8489bd990d419fc7ab925e986d8afd74"
    )

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": item.attempt_id} for item in review.attempts]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": review.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    p0 = {"attempt": 0, "attempt_id": "new-first"}
    p1 = {"attempt": 1, "attempt_id": "new-second"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, p0)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, p0)
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**p1, "attempt": 2})
    observed = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry", lambda req, slot: observed.append((req, slot)))
    dispatcher._check_boundary(request, p1)
    assert observed == [(request, p1)]


def test_2357_wave14_preserves_l493_and_both_owned_upstream_errors(monkeypatch):
    old = gpu.SCOPES[(2357, 13)]
    scope = gpu.SCOPES[(2357, 14)]
    assert scope.archive[:-1] == old.archive and [len(item.attempts) for item in scope.archive] == [2, 2, 1, 1, 2, 2, 2]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == old.boundary_physical_attempts == 2
    assert scope.payload_sha256 == old.payload_sha256 and scope.required_sources == old.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == old.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                         "maximum_physical_attempts": 182}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795717386.3467476
    review = scope.archive[-1]
    first, second = review.attempts
    assert first.identity == second.identity == gpu._2357_WAVE13 and first.request_id == second.request_id
    assert [item.progress["sse_event_count"] for item in review.attempts] == [25, 41]
    assert all(item.failure == "response_id_mismatch" and item.progress["provider_error_code"] == "upstream_error"
               and item.progress["response_id_matches_transport_alias"] for item in review.attempts)
    assert review.replay_files["attempt_01.replay_link.json"] == "6a33bded3147f04c4ba4e1d25c41645ad0668b9a233956f8f2d483a424016d2c"

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": first.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": first.request_id}
    checked = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry", lambda request, reservation: checked.append((request, reservation)))
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    p0 = {"attempt": 0, "attempt_id": "fresh-first"}
    p1 = {"attempt": 1, "attempt_id": "fresh-second"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - .01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, p0)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, p0)
    assert not checked
    dispatcher._check_boundary(request, p1)
    assert checked == [(request, p1)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**p1, "attempt": 2})


def test_2357_wave13_requires_both_sealed_predecessor_slots_and_keeps_conditional_retry(monkeypatch):
    previous = gpu.SCOPES[(2357, 12)]
    scope = gpu.SCOPES[(2357, 13)]
    receipt = scope.archive[-1]
    first, second = receipt.attempts
    assert scope.archive[:-1] == previous.archive
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == len(receipt.attempts) == 2
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182}
    assert first.identity == second.identity == gpu._2357_WAVE12 and first.request_id == second.request_id
    assert [item.attempt_id for item in receipt.attempts] == [
        "60096c8a5cf849d6a5c3a06bd203691b", "938a1fbd8a66479f87505a2824ba1c45"]
    assert all(item.failure == "response_id_mismatch" and item.progress["http_status"] == 200
               and item.progress["sse_event_count"] == 23 and item.progress["provider_error_code"] == "upstream_error"
               and item.progress["response_id_matches_transport_alias"] for item in receipt.attempts)
    assert receipt.replay_files == {
        "attempt_00.replay_authorization.json": "70682c9eaded0a16258580f17f2e24d29381504f061e494456607fcd8c708cb4",
        "attempt_00.replay.terminal.json": "51e39e4bed4fb1c52d889d6264e2a6b3b339f0bf48f31cc02b35a2ba4a22ed18",
        "attempt_01.replay_link.json": "8352325daa3f308d30727c140285464dc4ab68dda2cebc8714169505e595cb25",
    }
    assert (scope.cooldown_field, scope.not_before_unix) == ("failed_unix", 1795714691.0751874)

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": first.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": first.request_id}
    checked = []
    monkeypatch.setattr(dispatcher, "_check_boundary_retry", lambda request, reservation: checked.append((request, reservation)))
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    next_first = {"attempt": 0, "attempt_id": "a" * 32}
    next_second = {"attempt": 1, "attempt_id": "b" * 32}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - .01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, next_first)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    for item in receipt.attempts:
        with pytest.raises(PermissionError, match="signed physical attempts"):
            dispatcher._check_boundary(request, {**next_first, "attempt_id": item.attempt_id})
    dispatcher._check_boundary(request, next_first)
    assert not checked
    dispatcher._check_boundary(request, next_second)
    assert checked == [(request, next_second)]
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**next_second, "attempt": 2})


def test_2357_wave11_preserves_l493_after_exhausting_the_last_slot():
    old = gpu.SCOPES[(2357, 10)]
    scope = gpu.SCOPES[(2357, 11)]
    receipt = scope.archive[-1]
    assert scope.archive[:-1] == old.archive and [len(item.attempts) for item in scope.archive] == [2, 2, 1, 1]
    assert scope.original_start_round == scope.round == scope.replay_start_round == 5
    assert scope.original_boundary_physical_attempts == old.boundary_physical_attempts == 1
    assert scope.payload_sha256 == old.payload_sha256 and scope.required_sources == old.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795711758.937361
    assert receipt.attempt_id == "d2a8dbcdfbf14301b3e91b626cdfa716" and receipt.failure == "timeout"
    assert receipt.progress["http_status"] == 200 and receipt.progress["error_type"] == "sse_inactivity_timeout"
    assert receipt.progress["sse_event_count"] == 17 and receipt.progress["event_type"] == "response.output_item.added"
    assert "first_text_seconds" not in receipt.progress and "complete_terminal_captured" not in receipt.progress


def test_2357_wave10_carries_one_slot_after_a_sent_request_without_headers(monkeypatch):
    old = gpu.SCOPES[(2357, 9)]
    scope = gpu.SCOPES[(2357, 10)]
    receipt = scope.archive[-1]
    assert scope.archive[:-1] == old.archive and [len(item.attempts) for item in scope.archive] == [2, 2, 1]
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (5, 5, 6)
    assert scope.payload_sha256 == old.payload_sha256 and scope.required_sources == old.required_sources
    assert scope.original_boundary_physical_attempts == old.boundary_physical_attempts == 2
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 181}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795709663.170326
    assert receipt.failure == "timeout" and receipt.progress["error_type"] == "ReadTimeout"
    assert receipt.pre_post_connect_failure is False and receipt.progress["network_phases"]["send_body"]["state"] == "complete"
    assert receipt.progress["network_phases"]["receive_headers"]["duration_seconds"] == 1800.0037664161064
    assert {"http_status", "response_id", "provider_request_id"}.isdisjoint(receipt.progress)

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    monkeypatch.setattr(mac.RelayRetryDispatcher, "_check_boundary", lambda *_: None)
    request = {"payload_sha256": scope.payload_sha256}
    fresh = {"attempt": 0, "attempt_id": "n" * 32}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, fresh)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, fresh)
    for rejected in ({**fresh, "attempt": 1}, {**fresh, "attempt_id": receipt.attempt_id}):
        with pytest.raises(PermissionError, match="signed physical attempts"):
            dispatcher._check_boundary(request, rejected)


def test_2357_wave09_preserves_l493_and_both_closed_wave08_attempts(monkeypatch):
    previous = gpu.SCOPES[(2357, 8)]
    scope = gpu.SCOPES[(2357, 9)]
    assert scope.archive[:-1] == previous.archive
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (5, 5, 5)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.payload_sha256 == previous.payload_sha256 and scope.fitted_prefix_files == ()
    assert scope.required_sources == previous.required_sources and scope.recovery_budget == previous.recovery_budget
    assert scope.budget == previous.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91,
                                              "maximum_physical_attempts": 182}
    assert (scope.cooldown_field, scope.not_before_unix) == ("failed_unix", 1795706668.8023124)
    first, second = scope.archive[-1].attempts
    assert first.identity == second.identity and first.request_id == second.request_id
    assert (first.attempt_id, second.attempt_id) == ("dd59937e8b7a49a085984f24ee0effb7", "f4b70a1204d74013a247a4d8653ec3e9")
    assert all(item.failure == "response_id_mismatch" and item.progress["provider_error_code"] == "upstream_error"
               and item.progress["sse_event_count"] == 25 and item.progress["response_id_matches_transport_alias"]
               for item in (first, second))
    assert scope.archive[-1].replay_files["attempt_01.replay_link.json"] == (
        "33b610278022a90ddea6d0ab36f042fed266d1a31611c70fee05bef7d4e6b1e1"
    )

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    admitted = []
    monkeypatch.setattr(mac.RelayRetryDispatcher, "_check_boundary",
                        lambda _, request, reservation: admitted.append((request, reservation)))
    request = {"payload_sha256": scope.payload_sha256}
    new = {"attempt": 0, "attempt_id": "c" * 32}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - .01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, new)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    for old in (first, second, *previous.archive[-1].attempts):
        with pytest.raises(PermissionError, match="signed physical attempts"):
            dispatcher._check_boundary(request, {**new, "attempt_id": old.attempt_id})
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**new, "attempt": 2})
    dispatcher._check_boundary(request, new)
    assert admitted == [(request, new)]


def test_2357_wave08_pins_the_new_l493_decision_and_both_closed_r5_slots(monkeypatch):
    previous = gpu.SCOPES[(2357, 7)]
    scope = gpu.SCOPES[(2357, 8)]
    assert scope.original_start_round == previous.round == 4
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.round == scope.replay_start_round == 5 and scope.payload_sha256 != previous.payload_sha256
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 182}
    assert scope.not_before_unix == 1795705060.1421165 and scope.cooldown_field == "failed_unix"
    assert scope.required_sources == previous.required_sources
    assert dict(scope.fitted_prefix_files)["round_05/transition.json"] == (
        "bf855d43455d4ae2e1d2ff7d50a17b94be5c1adff2782c47fa5011d12f84703a"
    )
    archived, = scope.archive
    first, second = archived.attempts
    assert first.request_id == second.request_id and first.identity == second.identity
    assert (first.progress["sse_event_count"], second.progress["sse_event_count"]) == (23, 25)
    assert all(attempt.progress["provider_error_code"] == "upstream_error" and not attempt.pre_post_connect_failure
               for attempt in archived.attempts)
    assert archived.replay_files["attempt_01.replay_link.json"] == (
        "969c43ad504e1c4bb2641ef163f0fa757513bcd0447a9d9e4cee144d69acc892"
    )

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    admitted = []
    monkeypatch.setattr(mac.RelayRetryDispatcher, "_check_boundary",
                        lambda _, request, reservation: admitted.append((request, reservation)))
    new = {"attempt": 1, "attempt_id": "a" * 32}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary({}, new)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary({}, {**new, "attempt_id": second.attempt_id})
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary({}, {**new, "attempt": 2})
    dispatcher._check_boundary({}, new)
    assert admitted == [({}, new)]


def test_2357_wave07_keeps_the_unused_physical_from_the_exact_idle_timeout():
    previous = gpu.SCOPES[(2357, 6)]
    scope = gpu.SCOPES[(2357, 7)]
    assert scope.payload_sha256 == previous.payload_sha256
    assert scope.original_start_round == scope.round == previous.round == 4
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.archive[:-1] == previous.archive
    last = scope.archive[-1]
    assert scope.boundary_physical_attempts + len(last.attempts) == previous.boundary_physical_attempts
    assert last.failure == "timeout" and last.progress["error_type"] == "sse_inactivity_timeout"
    assert last.progress["event_type"] == "response.output_text.delta" and last.progress["sse_event_count"] == 440
    assert scope.cooldown_field == "failed_unix" and scope.not_before_unix == 1795701162.1268728
    assert scope.replay_start_round == 5
    assert dict(scope.required_sources)[gpu.PROVIDER_SOURCE] == "aacbc199e896091d4395fc036cb60bc604d91ec326a79fcff26e157d9a73705f"
    assert {k:v for k,v in scope.required_sources if k != gpu.PROVIDER_SOURCE} == {
        k:v for k,v in previous.required_sources if k != gpu.PROVIDER_SOURCE}


def test_887_r12_wave07_carries_only_the_unused_slot_after_the_hard_deadline():
    previous = gpu.SCOPES[(887, 6)]
    scope = gpu.SCOPES[(887, 7)]
    assert scope.archive[:-1] == previous.archive and scope.payload_sha256 == previous.payload_sha256
    assert scope.original_start_round == scope.round == previous.round == 12
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.boundary_physical_attempts + len(scope.archive[-1].attempts) == previous.boundary_physical_attempts
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 24, "maximum_logical_requests": 70, "maximum_physical_attempts": 139}
    assert scope.required_sources == previous.required_sources
    assert scope.cooldown_field == "failed_unix" and scope.not_before_unix == 1795706494.0892487
    assert scope.replay_start_round == 13
    last = scope.archive[-1]
    assert last.replay_files is None and last.failure == "timeout"
    assert last.terminal_file_sha256 == "b411f9727ce4d310961bb7acfbdabe0a5681aaa25a52b2cb71bfddbf5ad768c4"
    assert last.progress["error_type"] == "hard_wall_deadline" and last.progress["sse_event_count"] == 4060
    assert last.progress["event_type"] == "response.output_item.done"


def test_887_r12_wave08_binds_single_exhausted_idle_slot_and_stable_transport():
    previous = gpu.SCOPES[(887, 7)]
    scope = gpu.SCOPES[(887, 8)]
    receipt = scope.archive[-1]
    assert scope.run_name.endswith("-887-relay-retry-08") and scope.round == 12
    assert scope.payload_sha256 == previous.payload_sha256
    assert scope.required_sources == previous.required_sources
    assert scope.original_start_round == 12 and scope.original_boundary_physical_attempts == 1
    assert [len(item.attempts) for item in scope.archive] == [1, 1, 1, 1]
    assert receipt.attempt_id == "d820f33e0ffc4a96aec41a0bf9ef6afd" and receipt.failure == "timeout"
    assert receipt.progress["http_status"] == 200 and receipt.progress["error_type"] == "sse_inactivity_timeout"
    assert receipt.progress["sse_event_count"] == 1347 and receipt.progress["last_sse_seconds"] == 348.2114551658742
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795709072.5213063
    assert scope.budget == {"first_logical_requests": 24, "maximum_logical_requests": 70, "maximum_physical_attempts": 140}


def test_887_wave10_keeps_verified_l749_and_only_the_unused_r13_slot(monkeypatch):
    old = gpu.SCOPES[(887, 9)]
    scope = gpu.SCOPES[(887, 10)]
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (12, 13, 14)
    assert scope.original_boundary_physical_attempts == old.boundary_physical_attempts == 1
    assert scope.required_sources == old.required_sources and scope.payload_sha256 != old.payload_sha256
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 23, "maximum_logical_requests": 67, "maximum_physical_attempts": 133}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795717168.4392984
    prefix = dict(scope.fitted_prefix_files)
    assert prefix["round_12/boundary_user_message.json"] and prefix["round_12/batch_freeze.json"]
    assert prefix["round_13/transition.json"] == "4d4f93c6958478d309ff42560626118cf5629a3aa6323a5a069bcfa734a69f33"
    review, = scope.archive
    receipt, = review.attempts
    assert receipt.attempt_id == "45aa34b1ce2e4057a50e125986247374" and receipt.failure == "timeout"
    assert receipt.progress["sse_event_count"] == 33 and receipt.progress["error_type"] == "sse_inactivity_timeout"
    assert receipt.progress["event_type"] == "response.output_item.added" and "first_text_seconds" not in receipt.progress

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    monkeypatch.setattr(mac.RelayRetryDispatcher, "_check_boundary", lambda *_: None)
    request = {"payload_sha256": scope.payload_sha256}
    fresh = {"attempt": 0, "attempt_id": "fresh"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, fresh)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, fresh)
    for rejected in ({**fresh, "attempt": 1}, {**fresh, "attempt_id": receipt.attempt_id}):
        with pytest.raises(PermissionError, match="signed physical attempts"):
            dispatcher._check_boundary(request, rejected)


def test_887_r12_wave09_retains_only_one_slot_after_the_partial_hard_wall(monkeypatch):
    previous = gpu.SCOPES[(887, 8)]
    scope = gpu.SCOPES[(887, 9)]
    old = scope.archive[-1]
    assert scope.archive[:-1] == previous.archive and scope.payload_sha256 == previous.payload_sha256
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (12, 12, 13)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.boundary_physical_attempts + len(old.attempts) == 2
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 24, "maximum_logical_requests": 70, "maximum_physical_attempts": 139}
    assert scope.required_sources == previous.required_sources and scope.fitted_prefix_files == previous.fitted_prefix_files
    assert old.identity.name.endswith("887-relay-retry-08")
    assert old.attempt_id == "083e46db66ab4d6d862e7b21e1d270e1" and old.failure == "timeout"
    assert old.progress["http_status"] == 200 and old.progress["error_type"] == "hard_wall_deadline"
    assert old.progress["event_type"] == "response.output_text.delta" and old.progress["sse_event_count"] == 4679
    assert old.progress["last_sse_seconds"] == old.progress["last_sse_activity_seconds"] == 3551.905509167118
    assert old.replay_files is None and not old.pre_post_connect_failure
    assert old.terminal_file_sha256 == "d251850e10d8e66f795b77213ae0a11943c049817000a8080d4412db7b27d3fe"
    assert (scope.cooldown_field, scope.not_before_unix) == ("failed_unix", 1795714038.0125747)

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    admitted = []
    monkeypatch.setattr(mac.RelayRetryDispatcher, "_check_boundary",
                        lambda _, request, reservation: admitted.append((request, reservation)))
    request = {"payload_sha256": scope.payload_sha256}
    reservation = {"attempt": 0, "attempt_id": "e" * 32}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - .01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, reservation)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**reservation, "attempt_id": old.attempt_id})
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**reservation, "attempt": 1})
    dispatcher._check_boundary(request, reservation)
    assert admitted == [(request, reservation)]


def test_887_r12_wave06_keeps_the_verified_r9_ancestry_and_delayed_second_slot():
    previous = gpu.SCOPES[(887, 5)]
    scope = gpu.SCOPES[(887, 6)]
    assert scope.payload_sha256 == previous.payload_sha256
    assert scope.round == previous.round == 12 and scope.original_start_round is None
    assert previous.original_start_round == 9 and dict(previous.fitted_prefix_files)["round_12/transition.json"]
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.archive[:-1] == previous.archive and [len(item.attempts) for item in scope.archive] == [1, 1]
    assert scope.budget == {"first_logical_requests": 24, "maximum_logical_requests": 70, "maximum_physical_attempts": 140}
    last = scope.archive[-1]
    assert last.failure == "timeout" and last.progress["error_type"] == "hard_wall_deadline"
    assert last.progress["event_type"] == "response.output_text.delta" and last.progress["sse_event_count"] == 3063
    assert scope.cooldown_field == "not_before_unix" and scope.replay_start_round == 12
    assert dict(scope.required_sources)[gpu.PROVIDER_SOURCE] == "aacbc199e896091d4395fc036cb60bc604d91ec326a79fcff26e157d9a73705f"
    assert {k:v for k,v in scope.required_sources if k != gpu.PROVIDER_SOURCE} == {
        k:v for k,v in previous.required_sources if k != gpu.PROVIDER_SOURCE}


def test_6101_r13_retry_29_preserves_exact_posted_timeout_and_one_slot():
    scope = gpu.SCOPES[6101, 29]
    old = scope.archive[0]
    assert (scope.seed, scope.round, scope.original_start_round, scope.replay_start_round) == (6101, 13, 11, 14)
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 23, "maximum_logical_requests": 67, "maximum_physical_attempts": 133}
    assert len(scope.archive) == len(old.attempts) == 1 and old.replay_files is None
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795747684.6208463
    assert (old.failure, old.progress["error_type"], old.progress["http_status"], old.progress["sse_event_count"]) == (
        "timeout", "sse_inactivity_timeout", 200, 25,
    )
    assert not any(key.startswith("response.output_text") for key in old.progress["sse_milestones"])
    assert old.progress["elapsed_seconds"] - old.progress["last_sse_activity_seconds"] >= 900
    prefix = dict(scope.fitted_prefix_files)
    assert prefix["pause.request"].startswith("c2f1d007")
    assert prefix["round_11/batch_freeze.json"].startswith("362e2fa0")
    assert prefix["round_12/batch_freeze.json"].startswith("f4118d45")
    assert prefix["round_13/transition.json"].startswith("cf89a2ae")
    assert prefix["round_13/turn_00/request.json"].startswith("2b3b7e22")
    required = dict(scope.required_sources)
    assert required[(gpu.STUDY / "a7_gpu/seed_runner.py").as_posix()].startswith("8368fc31")
    assert required[(gpu.STUDY.parent / "pure_llm_v3_20261124/pure_llm_v3/transport_manager.py").as_posix()].startswith("2cdc0990")


def test_6101_r13_retry_29_normal_override_only_applies_to_exact_retired_r11_scope():
    from dataclasses import replace
    from types import SimpleNamespace

    scope = gpu.SCOPES[6101, 29]
    parent = SimpleNamespace(digest=gpu._6101_R11_CONTINUATION.contract, profile={
        "full_imported_boundary_round": 11, "replay_eligible_from_round": 11, "sealed_relay_retry_seconds": 180,
    }, predecessor={"dispatcher_module": "dispatch_v13_6101_r11_continuation"},
        transport=SimpleNamespace(logical_replies=3, transport_attempts=2))
    verify = gpu._reviewed_normal_parent_budget
    assert verify(parent, scope) == {
        "round": 11,
        "recovery_budget": {"maximum_logical_requests": 3, "maximum_physical_attempts": 2},
        "execution_budget": {"first_logical_requests": 25, "maximum_logical_requests": 75, "maximum_physical_attempts": 150},
    }
    for other in (replace(scope, boundary_physical_attempts=2), replace(scope, archive=()),
                  replace(scope, wave=30), replace(scope, original_start_round=12)):
        with pytest.raises(PermissionError):
            verify(parent, other)
    parent.profile["full_imported_boundary_round"] = True
    with pytest.raises(PermissionError):
        verify(parent, scope)


def test_6101_r13_retry_30_retains_both_closed_scopes_and_uses_bounded_new_window():
    old, scope = gpu.SCOPES[6101, 29], gpu.SCOPES[6101, 30]
    assert (scope.seed, scope.round, scope.original_start_round, scope.replay_start_round) == (6101, 13, 13, 13)
    assert scope.payload_sha256 == old.payload_sha256
    assert scope.original_boundary_physical_attempts == old.boundary_physical_attempts == 1
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
    assert scope.budget == {"first_logical_requests": 23, "maximum_logical_requests": 67, "maximum_physical_attempts": 134}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795749146.02625
    assert scope.archive[:-1] == old.archive and len(scope.archive[-1].attempts) == 1
    receipt = scope.archive[-1]
    assert receipt.identity == gpu._6101_R13_W29 and receipt.replay_files is None
    assert receipt.terminal_file_sha256.startswith("8c8286ae")
    assert receipt.progress["conflicting_event_type"] == "response.failed" and receipt.progress["response_id_matches_transport_alias"] is True
    assert receipt.progress["provider_error_code"] == "upstream_error" and receipt.progress["sse_event_count"] == 41
    assert not any(k.startswith("response.output_text") for k in receipt.progress["sse_milestones"])
    assert scope.required_sources == old.required_sources
    prefix = dict(scope.fitted_prefix_files)
    assert prefix["pause.request"].startswith("ab678d9a")
    assert prefix["round_13/source.json"].startswith("2d038696") and prefix["round_13/boundary_memory.json"].startswith("08735226")
    assert prefix["round_13/boundary_user_message.json"].startswith("414abd1b")
    assert prefix["round_13/turn_00/request.json"].startswith("afcd2bcd")


def test_6101_r13_retry_30_requires_narrow_delayed_second_slot(monkeypatch, tmp_path):

    scope = gpu.SCOPES[6101, 30]
    dispatcher = object.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.path = tmp_path
    observed = []
    monkeypatch.setattr(mac.RelayRetryDispatcher, "_check_boundary", lambda self, request, started: observed.append(started["attempt"]))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 1)
    with pytest.raises(PermissionError):
        dispatcher._check_boundary({}, {"attempt": 0, "attempt_id": "9" * 32})
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    for started in ({"attempt": 2, "attempt_id": "9" * 32},
                    {"attempt": 0, "attempt_id": gpu._6101_R13_P0.attempt_id},
                    {"attempt": 0, "attempt_id": scope.archive[-1].attempt_id}):
        with pytest.raises(PermissionError):
            dispatcher._check_boundary({}, started)
    dispatcher._check_boundary({}, {"attempt": 0, "attempt_id": "9" * 32})
    dispatcher._check_boundary({}, {"attempt": 1, "attempt_id": "8" * 32})
    assert observed == [0, 1]

    parent = SimpleNamespace(digest=gpu._6101_R13_W29.contract, profile={})
    assert gpu._reviewed_normal_parent_budget(parent, scope) is None


def test_887_wave24_reopens_only_one_slot_after_proven_preflight_failure():
    earlier, scope = gpu.SCOPES[887, 23], gpu.SCOPES[887, 24]
    assert scope.archive[:-1] == earlier.archive and scope.payload_sha256 == earlier.payload_sha256
    assert scope.original_start_round == scope.round == 34
    assert scope.original_boundary_physical_attempts == earlier.boundary_physical_attempts == 2
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 2, "maximum_logical_requests": 4, "maximum_physical_attempts": 7}
    assert scope.replay_start_round == 35 and scope.not_before_unix == 1795750280.2314875
    posted, readiness = scope.archive[-1].attempts
    assert posted.receipt_provider_flags == (True, True) and posted.unposted_readiness_files == ()
    assert readiness.receipt_provider_flags == (False, False) and readiness.failure == "connect_error"
    assert readiness.progress["event_type"] == "ipv4_preflight_no_headers"
    assert len(readiness.unposted_readiness_files) == len(readiness.replay_files) == 3
    assert replace(readiness, progress_file_sha256="a"*64).receipt_provider_flags == (True, True)
    assert gpu._12203_WAVE12_PRE_POST_CONNECT.receipt_provider_flags == (True, False)
    assert dict(scope.fitted_prefix_files)["pause.request"].startswith("08f6ff14")


def test_887_wave24_preflight_requires_exact_receipts_and_no_authenticated_start(tmp_path):
    names = [name for name, _ in gpu._887_W23_P1.unposted_readiness_files]
    for name in names:
        once(tmp_path / name, {"name": name})
    expected = tuple((name, file_digest(tmp_path / name)) for name in names)
    progress = gpu._887_W23_P1.progress
    assert mac._unposted_readiness(tmp_path, progress, expected)
    assert not mac._unposted_readiness(tmp_path, {**progress, "http_status": 200}, expected)
    assert not mac._unposted_readiness(tmp_path, {**progress, "first_phase_seconds": {"request_dispatched": 2}}, expected)
    active = {**progress, "network_phases": {**progress["network_phases"], "send_body": {"state": "complete"}}}
    assert not mac._unposted_readiness(tmp_path, active, expected)
    for name in ("started.json", "raw_provider_response.json", "provider_capture_receipt.json", "readiness_04.json"):
        path = tmp_path / name
        once(path, {})
        assert not mac._unposted_readiness(tmp_path, progress, expected)
        path.unlink()
    (tmp_path / names[-1]).write_text("{}")
    assert not mac._unposted_readiness(tmp_path, progress, expected)


def test_6101_r13_retry_31_keeps_all_three_proofs_and_one_remaining_attempt():
    previous, scope = gpu.SCOPES[6101, 30], gpu.SCOPES[6101, 31]
    assert scope.archive[:-1] == previous.archive and scope.payload_sha256 == previous.payload_sha256
    assert len(scope.archive) == 3 and all(len(item.attempts) == 1 for item in scope.archive)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 2
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 23, "maximum_logical_requests": 67, "maximum_physical_attempts": 133}
    assert scope.original_start_round == scope.round == 13 and scope.replay_start_round == 14
    assert scope.not_before_unix == 1795750835.6582642 and scope.cooldown_field == "not_before_unix"
    receipt = scope.archive[-1]
    assert receipt.identity == gpu._6101_R13_W30 and receipt.replay_files is None
    assert receipt.receipt_provider_flags == (True, True) and receipt.failure == "timeout"
    assert receipt.progress["sse_event_count"] == 5
    assert receipt.progress["elapsed_seconds"] - receipt.progress["last_sse_activity_seconds"] >= 900
    assert not any(k.startswith("response.output_text") for k in receipt.progress["sse_milestones"])
    assert scope.required_sources == previous.required_sources
    assert dict(scope.fitted_prefix_files)["round_13/turn_00/request.json"].startswith("1fbeedf1")
    assert dict(scope.fitted_prefix_files)["pause.request"].startswith("cffa5fcb")


def test_6101_r13_retry_31_only_one_physical_and_no_prior_attempt_identity(monkeypatch):
    scope = gpu.SCOPES[6101, 31]
    dispatcher = object.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    observed = []
    monkeypatch.setattr(mac.RelayRetryDispatcher, "_check_boundary",
                        lambda self, request, started: observed.append(started["attempt"]))
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 1)
    with pytest.raises(PermissionError):
        dispatcher._check_boundary({}, {"attempt": 0, "attempt_id": "9" * 32})
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    for identity in (item.attempt_id for item in scope.archive):
        with pytest.raises(PermissionError):
            dispatcher._check_boundary({}, {"attempt": 0, "attempt_id": identity})
    with pytest.raises(PermissionError):
        dispatcher._check_boundary({}, {"attempt": 1, "attempt_id": "8" * 32})
    dispatcher._check_boundary({}, {"attempt": 0, "attempt_id": "9" * 32})
    assert observed == [0]


def test_887_wave25_preserves_unresolved_upload_and_prior_proven_no_post():
    previous, scope = gpu.SCOPES[887, 24], gpu.SCOPES[887, 25]
    assert scope.archive[:-1] == previous.archive and scope.payload_sha256 == previous.payload_sha256
    assert scope.original_start_round == scope.round == 34 and scope.replay_start_round == 35
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 2, "maximum_logical_requests": 4, "maximum_physical_attempts": 7}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795751015.668551
    assert scope.required_sources == previous.required_sources
    assert scope.archive[-2] == gpu._887_W23_P1 and scope.archive[-2].receipt_provider_flags == (False, False)
    unknown = scope.archive[-1]
    assert unknown.identity == gpu._887_W24 and unknown.receipt_provider_flags == (True, True)
    assert unknown.unposted_readiness_files == () and unknown.replay_files is None and len(unknown.attempts) == 1
    assert unknown.failure == "timeout" and unknown.progress["error_type"] == "WriteTimeout"
    phases = unknown.progress["network_phases"]
    assert phases["send_headers"]["state"] == "complete" and phases["send_body"]["state"] == "failed"
    assert not {"receive_headers", "receive_body"}.intersection(phases)
    assert dict(scope.fitted_prefix_files)["round_34/turn_00/request.json"].startswith("06c1dd02")
    assert dict(scope.fitted_prefix_files)["pause.request"].startswith("5ddee934")


def test_6101_r13_retry_32_replaces_one_definitely_unposted_readiness_slot():
    previous, scope = gpu.SCOPES[6101, 31], gpu.SCOPES[6101, 32]
    assert scope.archive[:-1] == previous.archive and scope.payload_sha256 == previous.payload_sha256
    assert (scope.round, scope.original_start_round, scope.replay_start_round) == (13, 13, 14)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 23, "maximum_logical_requests": 67, "maximum_physical_attempts": 133}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795751615.3338208
    failed = scope.archive[-1]
    assert failed == gpu._6101_R13_W31_P0 and failed.receipt_provider_flags == (False, False)
    assert len(failed.unposted_readiness_files) == 3 and failed.replay_files is None
    assert failed.failure == "connect_error" and failed.progress["event_type"] == "ipv4_preflight_no_headers"
    assert replace(failed, progress_file_sha256="f"*64).receipt_provider_flags == (True, True)
    assert scope.archive[-2].receipt_provider_flags == (True, True)
    assert gpu._887_W23_P1.receipt_provider_flags == (False, False)
    assert scope.required_sources == previous.required_sources
    assert dict(scope.fitted_prefix_files)["pause.request"].startswith("6d11178a")



def test_2357_wave41_retains_the_verified_no_post_slot(tmp_path):
    prior, scope = gpu.SCOPES[(2357, 40)], gpu.SCOPES[(2357, 41)]
    review = scope.archive[-1]
    assert scope.archive[:-1] == prior.archive
    assert scope.payload_sha256 == prior.payload_sha256
    assert scope.required_sources == prior.required_sources
    assert scope.header_deadline_profile == prior.header_deadline_profile
    assert scope.recovery_budget == prior.recovery_budget
    assert scope.budget == prior.budget
    assert scope.not_before_unix == 1795751622.286606
    assert review.receipt_provider_flags == (False, False)
    for window, retry_unix in enumerate([1795751297.630015,1795751354.0037591,1795751407.509011], 1):
        once(tmp_path / f"readiness_{window:02d}.json", {
            "attempt_id": review.attempt_id, "job_sha256": review.job_sha256,
            "retry_unix": retry_unix, "window": window,
        })
    assert mac._unposted_readiness(tmp_path, review.progress, review.unposted_readiness_files)
    once(tmp_path / "started.json", {})
    assert not mac._unposted_readiness(tmp_path, review.progress, review.unposted_readiness_files)


def test_887_wave26_keeps_owned_failure_and_preserves_one_new_slot():
    previous, scope = gpu.SCOPES[887, 25], gpu.SCOPES[887, 26]
    assert scope.payload_sha256 == previous.payload_sha256 and scope.archive[:-1] == previous.archive
    assert scope.round == scope.original_start_round == 34 and scope.replay_start_round == 35
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 2, "maximum_logical_requests": 4, "maximum_physical_attempts": 7}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795751908.9702716
    assert scope.required_sources == previous.required_sources
    assert scope.archive[-3] == gpu._887_W23_P1 and scope.archive[-3].receipt_provider_flags == (False, False)
    latest = scope.archive[-1]
    assert latest.identity == gpu._887_W25 and latest.receipt_provider_flags == (True, True)
    assert latest.replay_files is None and len(latest.attempts) == 1
    assert latest.failure == "response_id_mismatch" and latest.progress["provider_error_code"] == "upstream_error"
    assert latest.progress["http_status"] == 200 and latest.progress["sse_event_count"] == 25
    assert latest.progress["response_id_matches_transport_alias"] is True
    assert not any(key.startswith("response.output_text") for key in latest.progress["sse_milestones"])
    assert dict(scope.fitted_prefix_files)["round_34/turn_00/request.json"].startswith("ccb994d6")
    assert dict(scope.fitted_prefix_files)["pause.request"].startswith("7ee711f6")


def test_157_wave40_preserves_w39_owned_failure_and_allows_only_one_new_slot():
    earlier, scope = gpu.SCOPES[157, 39], gpu.SCOPES[157, 40]
    assert scope.payload_sha256 == earlier.payload_sha256 and scope.archive[:-1] == earlier.archive
    assert scope.round == scope.original_start_round == 7 and scope.replay_start_round == 8
    assert scope.original_boundary_physical_attempts == earlier.boundary_physical_attempts == 1
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 169}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795751920.7254832
    assert scope.required_sources == earlier.required_sources
    assert scope.archive[-2] == gpu._157_W38_P1 and scope.archive[-2].receipt_provider_flags == (False, False)
    latest = scope.archive[-1]
    assert latest.identity == gpu._157_W39 and latest.receipt_provider_flags == (True, True)
    assert latest.replay_files is None and len(latest.attempts) == 1
    assert latest.failure == "response_id_mismatch" and latest.progress["provider_error_code"] == "upstream_error"
    assert latest.progress["http_status"] == 200 and latest.progress["sse_event_count"] == 41
    assert latest.progress["response_id_matches_transport_alias"] is True
    assert not any(key.startswith("response.output_text") for key in latest.progress["sse_milestones"])
    assert dict(scope.fitted_prefix_files)["round_07/turn_00/request.json"].startswith("b50b37ef")
    assert dict(scope.fitted_prefix_files)["pause.request"].startswith("75ca23bf")



def test_157_wave41_keeps_one_slot_after_the_exact_stream_idle_timeout():
    earlier, scope = gpu.SCOPES[157, 40], gpu.SCOPES[157, 41]
    assert scope.payload_sha256 == earlier.payload_sha256 and scope.archive[:-1] == earlier.archive
    assert scope.round == scope.original_start_round == 7 and scope.replay_start_round == 8
    assert scope.original_boundary_physical_attempts == earlier.boundary_physical_attempts == 1
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 169}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795753390.6367888
    assert scope.required_sources == earlier.required_sources and scope.header_deadline_profile == {}
    latest = scope.archive[-1]
    assert latest.identity == gpu._157_W40 and latest.receipt_provider_flags == (True, True)
    assert latest.replay_files is None and len(latest.attempts) == 1
    assert latest.failure == "timeout" and latest.terminal_file_sha256 == "0590b3651dec712804c0b6b84ed652edb462521aa8f0aa467c4ad54187687043"
    assert latest.progress_file_sha256 == "3940d862d3c5d460960b362ead424bde4bc7eb5237797d25b4f89a87573d1c93"
    progress = latest.progress
    assert progress["http_status"] == 200 and progress["sse_event_count"] == 13
    assert progress["error_type"] == "sse_inactivity_timeout"
    assert progress["elapsed_seconds"] - progress["last_sse_activity_seconds"] == pytest.approx(900, abs=.1)
    assert all(not key.startswith("response.output_text") for key in progress["sse_milestones"])
    assert dict(scope.fitted_prefix_files)["round_07/source.json"] == "ac42a77b7b18963db4be8ca807aa68bc04636643cdb8b0f9d0226a3f515a1aee"
    assert dict(scope.fitted_prefix_files)["pause.request"] == "a9c151ea06ecd51f3a6c00e7765b126d70342c1227b7ee7351bb8b5156e02864"


def test_2357_wave42_retains_the_owned_failed_terminal():
    prior, scope = gpu.SCOPES[(2357, 41)], gpu.SCOPES[(2357, 42)]
    review = scope.archive[-1]
    assert scope.archive[:-1] == prior.archive
    assert scope.round == prior.round == 5
    assert scope.payload_sha256 == prior.payload_sha256
    assert scope.required_sources == prior.required_sources
    assert scope.recovery_budget == prior.recovery_budget
    assert scope.budget == prior.budget
    assert scope.not_before_unix == 1795752409.7153354
    assert review.receipt_provider_flags == (True, True)
    progress = review.progress
    assert progress["conflicting_event_type"] == "response.failed"
    assert progress["conflicting_response_id"] == "resp_" + progress["provider_request_id"].replace("-", "")
    assert progress["provider_error_code"] == "upstream_error"
    assert review.terminal_file_sha256 == "6f88a656da52fcdb7155499ac1c35e8ab3ce58704aedefe5e5842cc85147733b"


def test_6101_r13_retry_33_preserves_earlier_no_post_and_new_owned_failure():
    earlier, scope = gpu.SCOPES[6101, 32], gpu.SCOPES[6101, 33]
    assert scope.payload_sha256 == earlier.payload_sha256 and scope.archive[:-1] == earlier.archive
    assert scope.round == scope.original_start_round == 13 and scope.replay_start_round == 14
    assert scope.original_boundary_physical_attempts == earlier.boundary_physical_attempts == 1
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 23, "maximum_logical_requests": 67, "maximum_physical_attempts": 133}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795752394.0001726
    assert scope.required_sources == earlier.required_sources
    assert scope.archive[-2] == gpu._6101_R13_W31_P0 and scope.archive[-2].receipt_provider_flags == (False, False)
    latest = scope.archive[-1]
    assert latest.identity == gpu._6101_R13_W32 and latest.receipt_provider_flags == (True, True)
    assert latest.replay_files is None and len(latest.attempts) == 1
    assert latest.failure == "response_id_mismatch" and latest.progress["provider_error_code"] == "upstream_error"
    assert latest.progress["http_status"] == 200 and latest.progress["sse_event_count"] == 41
    assert latest.progress["response_id_matches_transport_alias"] is True
    assert not any(key.startswith("response.output_text") for key in latest.progress["sse_milestones"])
    assert dict(scope.fitted_prefix_files)["round_13/turn_00/request.json"].startswith("6871e345")
    assert dict(scope.fitted_prefix_files)["pause.request"].startswith("949bc2d1")


def test_887_wave27_binds_actual_idle_timeout_and_one_fresh_slot():
    earlier, scope = gpu.SCOPES[887, 26], gpu.SCOPES[887, 27]
    assert scope.archive[:-1] == earlier.archive and scope.payload_sha256 == earlier.payload_sha256
    assert scope.round == scope.original_start_round == 34 and scope.replay_start_round == 35
    assert scope.original_boundary_physical_attempts == earlier.boundary_physical_attempts == 1
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 2, "maximum_logical_requests": 4, "maximum_physical_attempts": 7}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795753387.6624951
    assert scope.required_sources == earlier.required_sources
    assert scope.archive[-4] == gpu._887_W23_P1 and scope.archive[-4].receipt_provider_flags == (False, False)
    latest = scope.archive[-1]
    assert latest.identity == gpu._887_W26 and latest.receipt_provider_flags == (True, True)
    assert latest.replay_files is None and len(latest.attempts) == 1
    assert latest.failure == "timeout" and latest.progress["error_type"] == "sse_inactivity_timeout"
    assert latest.progress["http_status"] == 200 and latest.progress["sse_event_count"] == 23
    assert latest.progress["elapsed_seconds"] - latest.progress["last_sse_activity_seconds"] >= 900
    assert not any(key.startswith("response.output_text") for key in latest.progress["sse_milestones"])
    assert dict(scope.fitted_prefix_files)["round_34/turn_00/request.json"].startswith("1084dac0")
    assert dict(scope.fitted_prefix_files)["pause.request"].startswith("55af7d77")


def test_6101_r13_retry_34_preserves_owned_failure_and_one_new_slot():
    earlier, scope = gpu.SCOPES[6101, 33], gpu.SCOPES[6101, 34]
    assert scope.payload_sha256 == earlier.payload_sha256 and scope.archive[:-1] == earlier.archive
    assert scope.round == scope.original_start_round == 13 and scope.replay_start_round == 14
    assert scope.original_boundary_physical_attempts == earlier.boundary_physical_attempts == 1
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 23, "maximum_logical_requests": 67, "maximum_physical_attempts": 133}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795753557.8435712
    assert scope.required_sources == earlier.required_sources
    assert scope.archive[-3] == gpu._6101_R13_W31_P0 and scope.archive[-3].receipt_provider_flags == (False, False)
    latest = scope.archive[-1]
    assert latest.identity == gpu._6101_R13_W33 and latest.receipt_provider_flags == (True, True)
    assert latest.replay_files is None and len(latest.attempts) == 1
    assert latest.failure == "response_id_mismatch" and latest.progress["provider_error_code"] == "upstream_error"
    assert latest.progress["http_status"] == 200 and latest.progress["sse_event_count"] == 28
    assert latest.progress["response_id_matches_transport_alias"] is True
    assert not any(key.startswith("response.output_text") for key in latest.progress["sse_milestones"])
    assert dict(scope.fitted_prefix_files)["round_13/turn_00/request.json"].startswith("f94fb1fc")
    assert dict(scope.fitted_prefix_files)["pause.request"].startswith("03fb85a9")



def test_2357_wave43_retains_the_owned_failed_terminal():
    prior, scope = gpu.SCOPES[(2357, 42)], gpu.SCOPES[(2357, 43)]
    review = scope.archive[-1]
    assert scope.archive[:-1] == prior.archive
    assert scope.round == prior.round == 5
    assert scope.payload_sha256 == prior.payload_sha256
    assert scope.required_sources == prior.required_sources
    assert scope.recovery_budget == prior.recovery_budget
    assert scope.budget == prior.budget
    assert scope.not_before_unix == 1795753609.308748
    assert review.receipt_provider_flags == (True, True)
    progress = review.progress
    assert progress["conflicting_event_type"] == "response.failed"
    assert progress["conflicting_response_id"] == "resp_" + progress["provider_request_id"].replace("-", "")
    assert progress["provider_error_code"] == "upstream_error"
    assert review.terminal_file_sha256 == "932afc2d09c42e08680955c9f03a3d6cef34a6404dbe2a372d2184cdb4e62a52"


def test_887_wave28_preserves_owned_failure_and_one_new_slot():
    earlier, scope = gpu.SCOPES[887, 27], gpu.SCOPES[887, 28]
    assert scope.archive[:-1] == earlier.archive and scope.payload_sha256 == earlier.payload_sha256
    assert scope.round == scope.original_start_round == 34 and scope.replay_start_round == 35
    assert scope.original_boundary_physical_attempts == earlier.boundary_physical_attempts == 1
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 2, "maximum_logical_requests": 4, "maximum_physical_attempts": 7}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795753981.4353123
    assert scope.required_sources == earlier.required_sources
    assert scope.archive[-5] == gpu._887_W23_P1 and scope.archive[-5].receipt_provider_flags == (False, False)
    latest = scope.archive[-1]
    assert latest.identity == gpu._887_W27 and latest.receipt_provider_flags == (True, True)
    assert latest.replay_files is None and len(latest.attempts) == 1
    assert latest.failure == "response_id_mismatch" and latest.progress["provider_error_code"] == "upstream_error"
    assert latest.progress["http_status"] == 200 and latest.progress["sse_event_count"] == 25
    assert latest.progress["response_id_matches_transport_alias"] is True
    assert not any(key.startswith("response.output_text") for key in latest.progress["sse_milestones"])
    assert dict(scope.fitted_prefix_files)["round_34/turn_00/request.json"].startswith("f377973e")
    assert dict(scope.fitted_prefix_files)["pause.request"].startswith("bc623526")



def test_157_wave45_preserves_the_owned_failure_with_one_signed_post():
    previous, scope = gpu.SCOPES[157, 44], gpu.SCOPES[157, 45]
    assert scope.payload_sha256 == previous.payload_sha256 and scope.archive[:-1] == previous.archive
    assert (scope.round, scope.original_start_round, scope.replay_start_round) == (7, 7, 8)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 169}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795756480.9419253
    assert scope.required_sources == previous.required_sources and scope.header_deadline_profile == {}
    latest = scope.archive[-1]
    assert latest == gpu._157_W44_P0 and latest.receipt_provider_flags == (True, True)
    assert latest.failure == "response_id_mismatch" and latest.replay_files is None and len(latest.attempts) == 1
    assert latest.terminal_file_sha256 == "402a59ef4d9c18182d6886224dc45d9e0eb43827f8a9dad155296d07b7adfe86"
    progress = latest.progress
    assert progress["http_status"] == 200 and progress["sse_event_count"] == 25
    assert progress["provider_error_code"] == "upstream_error" and progress["conflicting_event_type"] == "response.failed"
    assert progress["conflicting_response_id"] == "resp_" + progress["provider_request_id"].replace("-", "")
    assert not any(field.startswith("response.output_text") for field in progress["sse_milestones"])
    assert dict(scope.fitted_prefix_files)["round_07/turn_00/request.json"] == "c9cfd6c4e79e84763550a5b614f5ccb5f8c59b33049eeaa37cf934b54b186728"
    assert dict(scope.fitted_prefix_files)["pause.request"] == "925adb61654e1554408161b37996f8f40737e2b8effdd0735a14cd987fc76ce7"


def test_157_wave44_preserves_the_owned_failure_with_one_signed_post():
    previous, scope = gpu.SCOPES[157, 43], gpu.SCOPES[157, 44]
    assert scope.payload_sha256 == previous.payload_sha256 and scope.archive[:-1] == previous.archive
    assert (scope.round, scope.original_start_round, scope.replay_start_round) == (7, 7, 8)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 169}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795755539.9294367
    assert scope.required_sources == previous.required_sources and scope.header_deadline_profile == {}
    latest = scope.archive[-1]
    assert latest == gpu._157_W43_P0 and latest.receipt_provider_flags == (True, True)
    assert latest.failure == "response_id_mismatch" and latest.replay_files is None and len(latest.attempts) == 1
    assert latest.terminal_file_sha256 == "cdcf14d70b8f8030e99ba7f6a5c6080dd2d5cbcc0c50bdf99d327f40674bc94f"
    progress = latest.progress
    assert progress["http_status"] == 200 and progress["sse_event_count"] == 41
    assert progress["provider_error_code"] == "upstream_error" and progress["conflicting_event_type"] == "response.failed"
    assert progress["conflicting_response_id"] == "resp_" + progress["provider_request_id"].replace("-", "")
    assert not any(field.startswith("response.output_text") for field in progress["sse_milestones"])
    assert dict(scope.fitted_prefix_files)["round_07/turn_00/request.json"] == "477740baea039fa4c20b025b77b96ce2ebd7b30a59f2dea9a3392d0ed552534f"
    assert dict(scope.fitted_prefix_files)["pause.request"] == "c03eaf366f0170966c49f697d3f19017a93df1ab8f508c05e0fe0e15e79acbfd"


def test_157_wave43_preserves_the_owned_failure_with_one_signed_post():
    previous, scope = gpu.SCOPES[157, 42], gpu.SCOPES[157, 43]
    assert scope.payload_sha256 == previous.payload_sha256 and scope.archive[:-1] == previous.archive
    assert (scope.round, scope.original_start_round, scope.replay_start_round) == (7, 7, 8)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 29, "maximum_logical_requests": 85, "maximum_physical_attempts": 169}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795754729.0493004
    assert scope.required_sources == previous.required_sources and scope.header_deadline_profile == {}
    latest = scope.archive[-1]
    assert latest == gpu._157_W42_P0 and latest.receipt_provider_flags == (True, True)
    assert latest.failure == "response_id_mismatch" and latest.replay_files is None and len(latest.attempts) == 1
    assert latest.terminal_file_sha256 == "726c28488f66093d3c3ddd54fb4fa9f70141355e4fd2d410eab1667ee3f7b01b"
    progress = latest.progress
    assert progress["http_status"] == 200 and progress["sse_event_count"] == 41
    assert progress["provider_error_code"] == "upstream_error" and progress["conflicting_event_type"] == "response.failed"
    assert progress["conflicting_response_id"] == "resp_" + progress["provider_request_id"].replace("-", "")
    assert not any(field.startswith("response.output_text") for field in progress["sse_milestones"])
    assert dict(scope.fitted_prefix_files)["round_07/turn_00/request.json"] == "8207886970da8548df9cf36877c0276e6b9e098af756c266ed2d357ee3340981"
    assert dict(scope.fitted_prefix_files)["pause.request"] == "b4fdf22c1e0036e251a39829ac87f4998f36d164c57533bbf088fa4c18c716b8"


def test_157_wave42_retains_the_owned_failed_terminal():
    prior, scope = gpu.SCOPES[(157, 41)], gpu.SCOPES[(157, 42)]
    review = scope.archive[-1]
    assert scope.archive[:-1] == prior.archive
    assert scope.round == prior.round == 7
    assert scope.payload_sha256 == prior.payload_sha256
    assert scope.required_sources == prior.required_sources
    assert scope.recovery_budget == prior.recovery_budget
    assert scope.budget == prior.budget
    assert scope.not_before_unix == 1795754192.396091
    assert review.receipt_provider_flags == (True, True)
    progress = review.progress
    assert progress["conflicting_event_type"] == "response.failed"
    assert progress["conflicting_response_id"] == "resp_" + progress["provider_request_id"].replace("-", "")
    assert progress["provider_error_code"] == "upstream_error"
    assert review.terminal_file_sha256 == "07aa3bad880ef80e0f1e15d0fc9fbaa2992cfb17595e902dba89880ec5bfae4f"


def test_6101_r13_retry_35_keeps_archived_post_and_one_new_physical():
    earlier, scope = gpu.SCOPES[6101, 34], gpu.SCOPES[6101, 35]
    assert scope.payload_sha256 == earlier.payload_sha256 and scope.archive[:-1] == earlier.archive
    assert scope.round == scope.original_start_round == 13 and scope.replay_start_round == 14
    assert scope.original_boundary_physical_attempts == earlier.boundary_physical_attempts == 1
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 23, "maximum_logical_requests": 67, "maximum_physical_attempts": 133}
    assert scope.cooldown_field == "not_before_unix" and scope.not_before_unix == 1795754203.3355026
    assert scope.required_sources == earlier.required_sources
    assert scope.archive[-4] == gpu._6101_R13_W31_P0 and scope.archive[-4].receipt_provider_flags == (False, False)
    latest = scope.archive[-1]
    assert latest.identity == gpu._6101_R13_W34 and latest.receipt_provider_flags == (True, True)
    assert latest.replay_files is None and len(latest.attempts) == 1
    assert latest.failure == "response_id_mismatch" and latest.progress["provider_error_code"] == "upstream_error"
    assert latest.progress["http_status"] == 200 and latest.progress["sse_event_count"] == 41
    assert latest.progress["response_id_matches_transport_alias"] is True
    assert not any(key.startswith("response.output_text") for key in latest.progress["sse_milestones"])
    assert dict(scope.fitted_prefix_files)["round_13/turn_00/request.json"].startswith("5178903b")
    assert dict(scope.fitted_prefix_files)["pause.request"].startswith("932844c1")



def test_2357_wave44_retains_the_owned_failed_terminal():
    prior, scope = gpu.SCOPES[(2357, 43)], gpu.SCOPES[(2357, 44)]
    review = scope.archive[-1]
    assert scope.archive[:-1] == prior.archive
    assert scope.round == prior.round == 5
    assert scope.payload_sha256 == prior.payload_sha256
    assert scope.required_sources == prior.required_sources
    assert scope.recovery_budget == prior.recovery_budget
    assert scope.budget == prior.budget
    assert scope.not_before_unix == 1795754226.258519
    assert review.receipt_provider_flags == (True, True)
    progress = review.progress
    assert progress["conflicting_event_type"] == "response.failed"
    assert progress["conflicting_response_id"] == "resp_" + progress["provider_request_id"].replace("-", "")
    assert progress["provider_error_code"] == "upstream_error"
    assert review.terminal_file_sha256 == "fa08b953a7b03e709a278f97e5447e225bdda10e107a411d76aee4179c9822d6"



def test_887_wave29_retains_the_owned_failed_terminal():
    prior, scope = gpu.SCOPES[(887, 28)], gpu.SCOPES[(887, 29)]
    review = scope.archive[-1]
    assert scope.archive[:-1] == prior.archive
    assert scope.round == prior.round == 34
    assert scope.payload_sha256 == prior.payload_sha256
    assert scope.required_sources == prior.required_sources
    assert scope.recovery_budget == prior.recovery_budget
    assert scope.budget == prior.budget
    assert scope.not_before_unix == 1795754613.8271155
    assert review.receipt_provider_flags == (True, True)
    progress = review.progress
    assert progress["conflicting_event_type"] == "response.failed"
    assert progress["conflicting_response_id"] == "resp_" + progress["provider_request_id"].replace("-", "")
    assert progress["provider_error_code"] == "upstream_error"
    assert review.terminal_file_sha256 == "41a988aba61160b02917622030a15a79479292938f480e82eb21a9913a9fdb49"



def test_6101_wave38_preserves_wave37_failed_terminal_and_one_current_slot(monkeypatch):
    previous = gpu.SCOPES[(6101, 37)]
    scope = gpu.SCOPES[(6101, 38)]
    assert scope.archive[:-1] == previous.archive
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (13, 13, 14)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 23, "maximum_logical_requests": 67, "maximum_physical_attempts": 133}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795756052.3693388)
    fitted = dict(scope.fitted_prefix_files)
    assert fitted["pause.request"] == "5803d2fe446180841b23e9e3681189ee3b5053ce6f5f9a1e61653aaa23e4f76c"
    assert fitted["round_13/turn_00/request.json"] == "cbfb7cbe8f1e07ddf628196edc9bbabf1235bf6582811357c846d692dfb5282d"
    review, = scope.archive[-1].attempts
    assert review.terminal_file_sha256 == "63459a1be8461efe1be8a1b50a54ba411815ec683e2969f32c80d1b7e7cb9088"
    assert review.failure == "response_id_mismatch" and review.receipt_provider_flags == (True, True)
    assert review.progress["http_status"] == 200 and review.progress["sse_event_count"] == 25
    assert review.progress["provider_error_code"] == "upstream_error"
    assert review.progress["conflicting_event_type"] == "response.failed"
    assert review.progress["response_id_matches_transport_alias"] is True

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": review.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": review.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    started = {"attempt": 0, "attempt_id": "new-first"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, started)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, started)
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**started, "attempt": 1})


def test_6101_wave37_preserves_wave36_failed_terminal_and_one_current_slot(monkeypatch):
    previous = gpu.SCOPES[(6101, 36)]
    scope = gpu.SCOPES[(6101, 37)]
    assert scope.archive[:-1] == previous.archive
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (13, 13, 14)
    assert scope.original_boundary_physical_attempts == previous.boundary_physical_attempts == 1
    assert scope.payload_sha256 == previous.payload_sha256 and scope.required_sources == previous.required_sources
    assert scope.recovery_budget == {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    assert scope.budget == {"first_logical_requests": 23, "maximum_logical_requests": 67, "maximum_physical_attempts": 133}
    assert (scope.cooldown_field, scope.not_before_unix) == ("not_before_unix", 1795755371.6992474)
    fitted = dict(scope.fitted_prefix_files)
    assert fitted["pause.request"] == "4553984475065b48ea6cd953798afb56307fb89dc5797c05bd05bfd61ba683c3"
    assert fitted["round_13/turn_00/request.json"] == "38587c7608d4b9b3ee64b02dac060e30a41c3e7553a4eb869712124d1c438e03"
    review, = scope.archive[-1].attempts
    assert review.terminal_file_sha256 == "8bda568986db37f56565e70291d9ef475ef44dfbbd45bd4af23386604f9ea435"
    assert review.failure == "response_id_mismatch" and review.receipt_provider_flags == (True, True)
    assert review.progress["http_status"] == 200 and review.progress["sse_event_count"] == 41
    assert review.progress["provider_error_code"] == "upstream_error"
    assert review.progress["conflicting_event_type"] == "response.failed"
    assert review.progress["response_id_matches_transport_alias"] is True

    dispatcher = mac.SuccessorDispatcher.__new__(mac.SuccessorDispatcher)
    dispatcher.reviewed_scope = scope
    dispatcher.scope = {"previous_attempts": [{"attempt_id": review.attempt_id}]}
    dispatcher.original = {"binding": {}, "payload": {}, "request_id": review.request_id}
    request = {**dispatcher.original, "request_id": "fresh", "logical_index": 0}
    started = {"attempt": 0, "attempt_id": "new-first"}
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix - 0.01)
    with pytest.raises(PermissionError, match="cooldown"):
        dispatcher._check_boundary(request, started)
    monkeypatch.setattr(mac, "time", lambda: scope.not_before_unix)
    dispatcher._check_boundary(request, started)
    with pytest.raises(PermissionError, match="signed physical attempts"):
        dispatcher._check_boundary(request, {**started, "attempt": 1})


def test_6101_wave36_retains_the_owned_failed_terminal():
    prior, scope = gpu.SCOPES[(6101, 35)], gpu.SCOPES[(6101, 36)]
    review = scope.archive[-1]
    assert scope.archive[:-1] == prior.archive
    assert scope.round == prior.round == 13
    assert scope.payload_sha256 == prior.payload_sha256
    assert scope.required_sources == prior.required_sources
    assert scope.recovery_budget == prior.recovery_budget
    assert scope.budget == prior.budget
    assert scope.not_before_unix == 1795754760.4797416
    assert review.receipt_provider_flags == (True, True)
    progress = review.progress
    assert progress["conflicting_event_type"] == "response.failed"
    assert progress["conflicting_response_id"] == "resp_" + progress["provider_request_id"].replace("-", "")
    assert progress["provider_error_code"] == "upstream_error"
    assert review.terminal_file_sha256 == "99e4ed2c4f0f742727020e6d7dfbc669957530e7aaf128cae32bc84e16ccaf06"



def test_2357_wave45_retains_the_owned_failed_terminal():
    prior, scope = gpu.SCOPES[(2357, 44)], gpu.SCOPES[(2357, 45)]
    review = scope.archive[-1]
    assert scope.archive[:-1] == prior.archive
    assert scope.round == prior.round == 5
    assert scope.payload_sha256 == prior.payload_sha256
    assert scope.required_sources == prior.required_sources
    assert scope.recovery_budget == prior.recovery_budget
    assert scope.budget == prior.budget
    assert scope.not_before_unix == 1795754938.2905803
    assert review.receipt_provider_flags == (True, True)
    progress = review.progress
    assert progress["conflicting_event_type"] == "response.failed"
    assert progress["conflicting_response_id"] == "resp_" + progress["provider_request_id"].replace("-", "")
    assert progress["provider_error_code"] == "upstream_error"
    assert review.terminal_file_sha256 == "8e4e3babbb83dea152bd9d73abb8c38249d37870613b826f6f7e7fe024d8ed7b"



def test_2357_wave46_retains_the_owned_failed_terminal():
    prior, scope = gpu.SCOPES[(2357, 45)], gpu.SCOPES[(2357, 46)]
    review = scope.archive[-1]
    assert scope.archive[:-1] == prior.archive
    assert scope.round == prior.round == 5
    assert scope.payload_sha256 == prior.payload_sha256
    assert scope.required_sources == prior.required_sources
    assert scope.recovery_budget == prior.recovery_budget
    assert scope.budget == prior.budget
    assert scope.not_before_unix == 1795755644.5120244
    assert review.receipt_provider_flags == (True, True)
    progress = review.progress
    assert progress["conflicting_event_type"] == "response.failed"
    assert progress["conflicting_response_id"] == "resp_" + progress["provider_request_id"].replace("-", "")
    assert progress["provider_error_code"] == "upstream_error"
    assert review.terminal_file_sha256 == "3650c5ffe5ef6fca49c799333a9e08ee46e1e0a565cfdab3dd29f62e48834764"
