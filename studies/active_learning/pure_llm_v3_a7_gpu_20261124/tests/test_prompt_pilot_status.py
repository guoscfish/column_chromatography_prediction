import hashlib
import json

import pytest

from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.prompt_pilot_status import (
    UNSTARTED_ANCHORS, local_progress, pending_attempt, remote_seed,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import prompt_pilot_status


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def put_progress(spool, identity, round_index, **events):
    attempt = spool / "attempts" / identity
    job_sha = "6" * 64
    write_json(attempt / "started.json", {"attempt_id": identity, "job_sha256": job_sha})
    write_json(attempt / "provider_progress.json", {
        "attempt_id": identity, "job_sha256": job_sha, "seed": 887, "round": round_index,
        "logical_index": 0, "phase": "sse_event", **events,
    })
    return attempt, job_sha


def test_waiting_attempt_does_not_display_old_sealed_progress(tmp_path):
    spool, run = tmp_path / "mac", tmp_path / "gpu"
    old, pending = "1" * 32, "2" * 32
    put_progress(spool, old, 3, event_type="response.completed")
    write_json(spool / "results" / f"{old}.json", {"attempt_id": old, "job_sha256": "6" * 64})
    directory = run / "seed_887"
    write_json(directory / "round_04/turn_00/attempt_00.started.json", {"attempt_id": pending, "attempt": 0})

    active = pending_attempt(run, directory, 4)
    assert active["attempt_id"] == pending
    assert local_progress(spool, 887, active) == {}

    put_progress(spool, pending, 4, event_type="response.output_item.done", http_status=200)
    state = local_progress(spool, 887, active)
    assert state["http_status"] == 200
    assert state["first_output_seconds"] is None
    assert not state["complete_event_seen"]
    assert not state["sealed"]


def test_reports_matching_completion_and_delivery_separately(tmp_path):
    identity = "3" * 32
    active = {"attempt_id": identity, "round": 4, "turn": 0}
    attempt, job_sha = put_progress(tmp_path, identity, 4, event_type="response.completed", sse_milestones={
        "response.output_text.delta": {"elapsed_seconds": 640.5}, "response.completed": {"elapsed_seconds": 702},
    })
    write_json(attempt / "raw_provider_response.json", {})
    write_json(attempt / "provider_capture_receipt.json", {})
    result_sha = "7" * 64
    write_json(tmp_path / "results" / f"{identity}.json", {
        "attempt_id": identity, "job_sha256": job_sha, "result_sha256": result_sha,
    })
    saved = local_progress(tmp_path, 887, active)
    assert saved["first_output_seconds"] == 640.5
    assert saved["complete_event_seen"] and saved["raw_response_saved"] and saved["sealed"]
    assert not saved["delivered"]

    write_json(tmp_path / "delivered" / f"{identity}.json", {
        "attempt_id": identity, "result_sha256": result_sha,
    })
    assert local_progress(tmp_path, 887, active)["delivered"]


def test_dispatch_wait_is_separate_from_provider_time(tmp_path, monkeypatch):
    identity = "5" * 32
    active = {
        "attempt_id": identity, "round": 4, "turn": 0,
        "reserved_unix": 1600000000, "status": "waiting_for_mac",
    }
    monkeypatch.setattr(prompt_pilot_status.time, "time", lambda: 1600000070)
    assert local_progress(tmp_path, 887, active, track_queue=True) == {
        "phase": "queued_for_dispatch", "dispatch_wait_seconds": 70,
    }

    attempt, job_sha = put_progress(tmp_path, identity, 4, elapsed_seconds=22, last_sse_interval_seconds=11)
    write_json(attempt / "started.json", {
        "attempt_id": identity, "job_sha256": job_sha, "started_utc": "2020-09-13T12:27:00+00:00",
    })
    state = local_progress(tmp_path, 887, active, track_queue=True)
    assert state["dispatch_wait_seconds"] == 20
    assert state["provider_elapsed_seconds"] == 50
    assert state["last_sse_interval_seconds"] == 11
    assert not state["complete_event_seen"]


def test_completed_transport_can_still_fail_scientific_validation(tmp_path):
    identity = "4" * 32
    directory = tmp_path / "seed_887"
    turn = directory / "round_04/turn_00"
    write_json(turn / "attempt_00.started.json", {"attempt_id": identity, "attempt": 0})
    write_json(turn / "attempt_00.terminal.json", {"status": "completed"})
    write_json(turn / "decision.json", {"state": "rejected", "diagnosis": {"errors": [
        {"code": "unexpected_field", "path": "/choices/0/references/0"},
        {"code": "unexpected_field", "path": "/choices/1/references/0"},
        {"code": "multifactor_mismatch", "path": "/choices/2/references/0/relationship"},
    ]}})

    active = pending_attempt(tmp_path, directory, 4)
    assert active["status"] == "completed"
    assert active["selection"] == "rejected"
    assert active["selection_errors"] == {"unexpected_field": 2, "multifactor_mismatch": 1}


def test_failed_worker_reports_sealed_failure_instead_of_unsubmitted_reservation(tmp_path):
    directory = tmp_path / "seed_157"
    turn = directory / "round_03/turn_00"
    old, local_only = "6" * 32, "7" * 32
    write_json(turn / "attempt_00.started.json", {"attempt_id": old, "attempt": 0, "transport": "mac_bridge"})
    write_json(turn / "attempt_00.terminal.json", {"status": "stopped", "error": "connect_error", "ambiguous": False})
    write_json(tmp_path / "bridge/results" / f"{old}.json", {})
    write_json(turn / "attempt_01.started.json", {"attempt_id": local_only, "attempt": 1, "transport": "mac_bridge"})

    assert pending_attempt(tmp_path, directory, 3)["attempt_id"] == local_only
    active = pending_attempt(tmp_path, directory, 3, failed=True)
    assert (active["attempt_id"], active["status"], active["failure"], active["unsubmitted_reservations_skipped"]) == (
        old, "stopped", "connect_error", 1,
    )
    write_json(tmp_path / "bridge/outbox" / f"{local_only}.json", {})
    active = pending_attempt(tmp_path, directory, 3, failed=True)
    assert active["attempt_id"] == local_only and active["status"] == "waiting_for_mac"


def test_local_progress_exposes_only_sanitized_stream_error_fields(tmp_path):
    identity = "3" * 32
    active = {"attempt_id": identity, "round": 4, "turn": 0}
    put_progress(tmp_path, identity, 4, event_type="response.in_progress", error_type="APIError",
                 http_status=200, provider_error_status=503, provider_error_code="server_error",
                 provider_error_type="server_error", provider_error_param="sk-not-for-status",
                 conflicting_event_type="response.completed", conflicting_response_id="resp_fixture_alias",
                 response_id_matches_transport_alias=True, provider_error_message="private upstream body")
    provider = local_progress(tmp_path, 887, active)
    assert provider["http_status"] == 200 and provider["provider_error_status"] == 503
    assert provider["error_type"] == "APIError" and provider["provider_error_code"] == "server_error"
    assert provider["conflicting_event_type"] == "response.completed"
    assert provider["response_id_matches_transport_alias"] is True
    assert "provider_error_param" not in provider and "provider_error_message" not in provider


def test_unstarted_seeds_show_only_verified_original_anchor(tmp_path, monkeypatch):
    runs = tmp_path / "purellm_a7/runs"
    checkpoint = (tmp_path / "formal_cuda/row4g_b32_20261124/trajectories"
                  / "random-6101/rounds/random/round_00/model/best.pt")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"original checkpoint")
    monkeypatch.setitem(UNSTARTED_ANCHORS, 6101, hashlib.sha256(checkpoint.read_bytes()).hexdigest())

    state = remote_seed(runs, 6101)
    assert (state["phase"], state["labels"], state["completed_rounds"]) == ("l333_anchor_verified", 333, 0)
    assert state["run"] is None and not state["worker_active"]

    checkpoint.write_bytes(b"changed checkpoint")
    state = remote_seed(runs, 6101)
    assert state["phase"] == "l333_anchor_unverified"
    assert state["labels"] is None and state["checkpoint_sha256"] is None


def test_2357_extension_preserves_parent_labels_and_other_seeds(tmp_path):
    cohort = tmp_path / prompt_pilot_status.SERIAL_IPV4
    parent = {
        "run_directory": str(cohort),
        "profile": {"confirmation_seeds": list(prompt_pilot_status.SEEDS), "provider_capability": "gated_ipv4"},
        "predecessor": {"dispatcher_module": "dispatch_prepared_ipv4",
                        "lanes": {"2357": {"start_round": 1, "checkpoint_sha256": "9" * 64}}},
    }
    parent_sha = hashlib.sha256(json.dumps(parent, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write_json(cohort / "formal_contract.json", parent)
    write_json(cohort / "authorization.json", {"contract_sha256": parent_sha})
    extension = tmp_path / prompt_pilot_status.IPV4_EXTENSION
    child = {
        "run_directory": str(extension), "profile": {"confirmation_seeds": [2357]},
        "predecessor": {"parent_contract_sha256": parent_sha,
                        "dispatcher_module": "dispatch_prepared_ipv4_recovery_2357"},
    }
    child_sha = hashlib.sha256(json.dumps(child, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write_json(extension / "formal_contract.json", child)
    write_json(extension / "authorization.json", {"contract_sha256": child_sha})

    status = remote_seed(tmp_path, 2357)
    assert (status["run"], status["labels"], status["target_labels"]) == (extension.name, 365, 1485)
    assert remote_seed(tmp_path, 887)["run"] == cohort.name

    continuation = tmp_path / prompt_pilot_status.IPV4_CONTINUATION
    follow_on = {
        "run_directory": str(continuation), "profile": {"confirmation_seeds": [2357]},
        "predecessor": {"parent_contract_sha256": parent_sha, "extension_contract_sha256": child_sha,
                        "dispatcher_module": "dispatch_prepared_ipv4_continue_2357"},
    }
    follow_on_sha = hashlib.sha256(json.dumps(follow_on, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write_json(continuation / "formal_contract.json", follow_on)
    write_json(continuation / "authorization.json", {"contract_sha256": follow_on_sha})
    assert remote_seed(tmp_path, 2357)["run"] == continuation.name

    r2_extension = tmp_path / prompt_pilot_status.IPV4_R2_EXTENSION
    r2 = {
        "run_directory": str(r2_extension), "profile": {"confirmation_seeds": [2357]},
        "predecessor": {"parent_contract_sha256": follow_on_sha,
                        "dispatcher_module": "dispatch_v13_2357_r2_sse_extension"},
    }
    r2_sha = hashlib.sha256(json.dumps(r2, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write_json(r2_extension / "formal_contract.json", r2)
    write_json(r2_extension / "authorization.json", {"contract_sha256": r2_sha})
    assert remote_seed(tmp_path, 2357)["run"] == r2_extension.name

    r2_continuation = tmp_path / prompt_pilot_status.IPV4_R2_CONTINUATION
    next_record = {
        "run_directory": str(r2_continuation), "profile": {"confirmation_seeds": [2357]},
        "predecessor": {"parent_contract_sha256": follow_on_sha, "extension_contract_sha256": r2_sha,
                        "dispatcher_module": "dispatch_v13_2357_r2_continuation"},
    }
    next_sha = hashlib.sha256(json.dumps(next_record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write_json(r2_continuation / "formal_contract.json", next_record)
    write_json(r2_continuation / "authorization.json", {"contract_sha256": next_sha})
    assert remote_seed(tmp_path, 2357)["run"] == r2_continuation.name


@pytest.mark.parametrize("seed,index,segment,dispatcher", [
    (887, 6, "IPV4_JSON_EXTENSION", "dispatch_v13_887_json_extension"),
    (6101, 1, "IPV4_WS_EXTENSION", "dispatch_v13_6101_ws_extension"),
])
def test_tracks_transport_extension_only_from_its_exact_previous_run(tmp_path, seed, index, segment, dispatcher):
    cohort = tmp_path / prompt_pilot_status.SERIAL_IPV4
    streaming = cohort.with_name(f"{cohort.name}-{seed}-extension-01")
    whole_response = tmp_path / getattr(prompt_pilot_status, segment)

    def publish(run, seeds, predecessor):
        record = {"run_directory": str(run), "profile": {"confirmation_seeds": seeds,
                  "provider_capability": "gated_ipv4"}, "predecessor": predecessor}
        sha = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        write_json(run / "formal_contract.json", record)
        write_json(run / "authorization.json", {"contract_sha256": sha})
        return sha

    parent = publish(cohort, list(prompt_pilot_status.SEEDS), {
        "dispatcher_module": "dispatch_prepared_ipv4",
        "lanes": {str(seed): {"start_round": index, "checkpoint_sha256": "3" * 64}},
    })
    streaming_sha = publish(streaming, [seed], {"parent_contract_sha256": parent,
                                              "dispatcher_module": "dispatch_v13_lane_extension"})
    prior = {"parent_contract_sha256": parent, "previous_contract_sha256": "a" * 64,
             "dispatcher_module": dispatcher}
    publish(whole_response, [seed], prior)
    assert remote_seed(tmp_path, seed)["run"] == streaming.name

    last_sha = publish(whole_response, [seed], {**prior, "previous_contract_sha256": streaming_sha})
    current = remote_seed(tmp_path, seed)
    assert (current["run"], current["labels"], current["phase"]) == (whole_response.name, 333 + 32 * index, "waiting_to_start")
    bounded = cohort.with_name(f"{cohort.name}-{seed}-bounded-sse-01")
    bounded_prior = {"previous_contract_sha256": last_sha, "previous_run_directory": str(whole_response),
                     "dispatcher_module": "dispatch_v13_bounded_sse_recovery"}
    publish(bounded, [seed], bounded_prior)
    assert remote_seed(tmp_path, seed)["run"] == bounded.name


@pytest.mark.parametrize(("seed", "start", "labels"),
                         [(12203, 4, 461), (157, 3, 429), (6101, 1, 365), (887, 7, 557), (2357, 4, 461)])
def test_resilient_run_tracks_exact_signed_ancestor_and_launchd(tmp_path, monkeypatch, seed, start, labels):
    cohort = tmp_path / prompt_pilot_status.SERIAL_IPV4
    resilient = cohort.with_name(f"{cohort.name}-{seed}-resilient-sse-01")
    parent = {"run_directory": str(cohort), "profile": {"confirmation_seeds": list(prompt_pilot_status.SEEDS),
              "provider_capability": "gated_ipv4"},
              "predecessor": {"dispatcher_module": "dispatch_prepared_ipv4",
                              "lanes": {str(seed): {"start_round": start, "checkpoint_sha256": "8" * 64}}}}

    def publish(run, record, **approval):
        sha = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        write_json(run / "formal_contract.json", record)
        write_json(run / "authorization.json", {"contract_sha256": sha, **approval})
        return sha

    parent_sha = publish(cohort, parent)
    prior, prior_sha = cohort, parent_sha
    if seed == 2357:
        extension = tmp_path / prompt_pilot_status.IPV4_EXTENSION
        extension_sha = publish(extension, {
            "run_directory": str(extension), "profile": {"confirmation_seeds": [seed]},
            "predecessor": {"parent_contract_sha256": parent_sha, "dispatcher_module": "dispatch_prepared_ipv4_recovery_2357"},
        })
        prior = tmp_path / prompt_pilot_status.IPV4_CONTINUATION
        prior_sha = publish(prior, {
            "run_directory": str(prior), "profile": {"confirmation_seeds": [seed]},
            "predecessor": {"parent_contract_sha256": parent_sha, "extension_contract_sha256": extension_sha,
                            "dispatcher_module": "dispatch_prepared_ipv4_continue_2357"},
        })
    record = {
        "run_directory": str(resilient),
        "profile": {"confirmation_seeds": [seed], "target_labels": 1485, "ipv4_readiness_windows": 3,
                    "resume_boundaries": {str(seed): {"round": start}}},
        "predecessor": {
            "dispatcher_module": "dispatch_v13_resilient_sse", "seed": seed, "round": start,
            "parent_run_directory": str(prior), "parent_contract_sha256": prior_sha,
            "previous_run_directory": str(prior), "previous_contract_sha256": "a" * 64,
            "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 1},
        },
    }
    if seed in (6101, 2357):
        record["profile"]["sse_inactivity_seconds"] = 900
    if seed == 2357:
        record["profile"]["request_timeouts"] = {"total_seconds": 3600}
    flags = {"authorize_paid_api": True, "authorize_label_reveal": True, "authorize_cuda_training": True}
    publish(resilient, record, **flags)
    assert remote_seed(tmp_path, seed)["run"] == prior.name
    record["predecessor"]["previous_contract_sha256"] = prior_sha
    publish(resilient, record)
    assert remote_seed(tmp_path, seed)["run"] == prior.name
    exact = publish(resilient, record, **flags)

    row = remote_seed(tmp_path, seed)
    assert (row["run"], row["segment"], row["labels"], row["contract_sha256"]) == (
        resilient.name, "ipv4_resilient_sse", labels, exact,
    )
    assert row["recovery_round"] == start and row["readiness_windows"] == 3
    commands = []

    def launchctl(command):
        commands.append(command)
        return ["  pid = 1234"]

    monkeypatch.setattr(prompt_pilot_status, "command_lines", launchctl)
    assert prompt_pilot_status.dispatcher_pid(seed, row["segment"]) == 1234
    assert commands[0][-1].endswith(f"/com.purellm.a7-ipv4-v13-{seed}-resilient-sse-01")

    recovery = cohort.with_name(f"{cohort.name}-{seed}-resilient-failure-01")
    persisted_approval = prompt_pilot_status.read_json(resilient / "authorization.json")
    predecessor = {**record["predecessor"],
                   "dispatcher_module": "dispatch_v13_resilient_failure_recovery",
                   "previous_run_directory": str(resilient), "previous_contract_sha256": exact,
                   "previous_authorization_sha256": prompt_pilot_status.record_digest(persisted_approval)}
    new_record = {**record, "run_directory": str(recovery), "predecessor": predecessor}
    publish(recovery, {**new_record, "predecessor": {**predecessor, "previous_contract_sha256": parent_sha}}, **flags)
    assert remote_seed(tmp_path, seed)["run"] == resilient.name
    publish(recovery, {**new_record, "predecessor": {**predecessor, "previous_authorization_sha256": "a" * 64}}, **flags)
    assert remote_seed(tmp_path, seed)["run"] == resilient.name
    publish(recovery, new_record)
    assert remote_seed(tmp_path, seed)["run"] == resilient.name
    recovery_sha = publish(recovery, new_record, **flags)
    row = remote_seed(tmp_path, seed)
    assert (row["run"], row["segment"], row["labels"], row["contract_sha256"]) == (
        recovery.name, "ipv4_resilient_failure", labels, recovery_sha,
    )
    assert row["continuation"] and row["recovery_round"] == start and row["readiness_windows"] == 3
    assert prompt_pilot_status.dispatcher_pid(seed, row["segment"]) == 1234
    assert commands[-1][-1].endswith(f"/com.purellm.a7-ipv4-v13-{seed}-resilient-failure-01")

    if seed == 887:
        check_887_reviewed_cached_r8(tmp_path, monkeypatch, publish, flags, commands, recovery, recovery_sha, new_record)
        return
    relay = cohort.with_name(f"{cohort.name}-{seed}-relay-retry-01")
    relay_record = {
        **new_record, "run_directory": str(relay),
        "profile": {**new_record["profile"], "sealed_relay_retry_seconds": 180,
                    "audited_future_transport_replay": True, "replay_eligible_from_round": start},
        "predecessor": {**predecessor, "dispatcher_module": "dispatch_v13_relay_retry",
                        "previous_run_directory": str(recovery), "previous_contract_sha256": recovery_sha,
                        "previous_authorization_sha256": prompt_pilot_status.record_digest(
                            prompt_pilot_status.read_json(recovery / "authorization.json")),
                        "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}},
    }
    relay_sha = publish(relay, relay_record, **flags)
    row = remote_seed(tmp_path, seed)
    assert (row["run"], row["segment"], row["labels"], row["contract_sha256"]) == (
        relay.name, "ipv4_relay_retry", labels, relay_sha,
    )
    assert row["recovery_round"] == start and row["sealed_relay_retry_seconds"] == 180
    assert prompt_pilot_status.dispatcher_pid(seed, row["segment"]) == 1234
    assert commands[-1][-1].endswith(f"/com.purellm.a7-ipv4-v13-{seed}-relay-retry-01")

    relay_auth = prompt_pilot_status.record_digest(prompt_pilot_status.read_json(relay / "authorization.json"))
    if seed == 6101:
        check_6101_reviewed_wave02(tmp_path, monkeypatch, publish, flags, commands, relay, relay_sha, relay_auth,
                                  relay_record)
        return
    if seed == 2357:
        check_2357_reviewed_wave02(tmp_path, monkeypatch, publish, flags, commands, relay, relay_sha, relay_auth,
                                  relay_record)
        return
    monkeypatch.setattr(prompt_pilot_status, f"RELAY_{seed}_WAVE01_CONTRACT", relay_sha)
    monkeypatch.setattr(prompt_pilot_status, f"RELAY_{seed}_WAVE01_AUTH", relay_auth)
    deadline = getattr(prompt_pilot_status, f"RELAY_{seed}_NOT_BEFORE")
    monkeypatch.setattr(prompt_pilot_status.time, "time", lambda: deadline - 15)
    successor = cohort.with_name(f"{cohort.name}-{seed}-relay-retry-02")
    successor_proof = {**relay_record["predecessor"], "dispatcher_module": f"dispatch_v13_{seed}_relay_retry",
                       "previous_run_directory": str(relay), "previous_contract_sha256": relay_sha,
                       "previous_authorization_sha256": relay_auth, "relay_not_before_unix": deadline,
                       "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}}
    if seed == 157:
        successor_proof.update(
            prior_boundary_budget={"physical_authorized": 2, "physical_already_used": 2, "physical_remaining": 0},
            newly_authorized_boundary_physical_attempts=1,
            previous_request_id=prompt_pilot_status.RELAY_157_BOUNDARY_REQUEST,
            previous_terminal_file_sha256=["1" * 64, "2" * 64],
            previous_transport_progress_file_sha256=["3" * 64, "5" * 64],
            previous_attempts=[{"attempt": 0, "attempt_id": "4" * 32}, {"attempt": 1, "attempt_id": "5" * 32}],
            execution_budget={"first_logical_requests": 33, "maximum_logical_requests": 97,
                              "maximum_physical_attempts": 193},
        )
    else:
        successor_proof.update(
            inherited_boundary_budget={"physical_authorized": 2, "physical_already_used": 1, "physical_remaining": 1},
            previous_request_id="1" * 64, previous_terminal_file_sha256="2" * 64,
            previous_transport_progress_file_sha256="3" * 64,
            previous_attempts=[{"attempt": 0, "attempt_id": "4" * 32}],
        )
    successor_record = {**relay_record, "run_directory": str(successor), "predecessor": successor_proof,
                        "profile": {**relay_record["profile"], "replay_eligible_from_round": start + 1}}
    publish(successor, successor_record)
    assert remote_seed(tmp_path, seed)["run"] == relay.name
    publish(successor, {**successor_record, "predecessor": {**successor_proof,
            "previous_authorization_sha256": "f" * 64}}, **flags)
    assert remote_seed(tmp_path, seed)["run"] == relay.name
    publish(successor, {**successor_record, "predecessor": {**successor_proof,
            "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}}}, **flags)
    assert remote_seed(tmp_path, seed)["run"] == relay.name
    if seed == 157:
        publish(successor, {**successor_record, "predecessor": {**successor_proof,
                "newly_authorized_boundary_physical_attempts": True}}, **flags)
        assert remote_seed(tmp_path, seed)["run"] == relay.name
        publish(successor, {**successor_record, "predecessor": {**successor_proof,
                "prior_boundary_budget": {"physical_authorized": 2, "physical_already_used": 1,
                                          "physical_remaining": 1}}}, **flags)
        assert remote_seed(tmp_path, seed)["run"] == relay.name
    successor_sha = publish(successor, successor_record, **flags)
    row = remote_seed(tmp_path, seed)
    assert (row["run"], row["segment"], row["labels"], row["contract_sha256"]) == (
        successor.name, "ipv4_relay_retry_02", labels, successor_sha,
    )
    assert row["relay_not_before_unix"] == deadline and row["sealed_relay_retry_seconds"] == 180
    assert row["api"] == {"status": "relay_boundary_cooldown", "relay_retry_wait_seconds": 15}
    assert prompt_pilot_status.dispatcher_pid(seed, row["segment"]) == 1234
    assert commands[-1][-1].endswith(f"/com.purellm.a7-ipv4-v13-{seed}-relay-retry-02")

    successor_auth = prompt_pilot_status.record_digest(prompt_pilot_status.read_json(successor / "authorization.json"))
    monkeypatch.setattr(prompt_pilot_status, f"RELAY_{seed}_WAVE02_CONTRACT", successor_sha)
    monkeypatch.setattr(prompt_pilot_status, f"RELAY_{seed}_WAVE02_AUTH", successor_auth)
    stage03 = cohort.with_name(f"{cohort.name}-{seed}-relay-retry-03")
    exact = getattr(prompt_pilot_status, f"RELAY_{seed}_WAVE03_PROOF")
    terminal = {**getattr(prompt_pilot_status, f"RELAY_{seed}_WAVE02_ATTEMPT"), "terminal_sha256": "6" * 64}
    terminal_file = successor_proof["previous_terminal_file_sha256"]
    progress_file = successor_proof["previous_transport_progress_file_sha256"]
    if seed == 157:
        terminal_file, progress_file = terminal_file[-1], progress_file[-1]
    chain = [
        {"contract_sha256": relay_sha, "authorization_sha256": relay_auth, "run_directory": str(relay),
         "request_id": successor_proof["previous_request_id"],
         "terminal_file_sha256": terminal_file, "progress_file_sha256": progress_file,
         "previous_attempts": successor_proof["previous_attempts"]},
        {"contract_sha256": successor_sha, "authorization_sha256": successor_auth, "run_directory": str(successor),
         "request_id": exact["previous_request_id"], "terminal_file_sha256": exact["previous_terminal_file_sha256"],
         "progress_file_sha256": exact["previous_transport_progress_file_sha256"], "previous_attempts": [terminal]},
    ]
    proof03 = {**successor_proof, **exact, "dispatcher_module": "dispatch_v13_retry_successor",
               "previous_run_directory": str(successor), "previous_contract_sha256": successor_sha,
               "previous_authorization_sha256": successor_auth, "reviewed_retry_chain": chain,
               "previous_attempts": [terminal]}
    record03 = {**successor_record, "run_directory": str(stage03), "predecessor": proof03}
    monkeypatch.setattr(prompt_pilot_status.time, "time", lambda: exact["relay_not_before_unix"] - 20)
    publish(stage03, record03)
    assert remote_seed(tmp_path, seed)["run"] == successor.name
    for changes in (
        {"previous_authorization_sha256": "a" * 64},
        {"payload_sha256": "a" * 64},
        {"reviewed_retry_chain": list(reversed(chain))},
        {"reviewed_retry_chain": [{**chain[0], "progress_file_sha256": "a" * 64}, chain[1]]},
        {"reviewed_retry_chain": [{**chain[0], "previous_attempts": chain[0]["previous_attempts"][1:]}, chain[1]]},
        {"newly_authorized_boundary_physical_attempts": True},
        {"previous_attempts": [{**terminal, "result_sha256": "a" * 64}]},
    ):
        publish(stage03, {**record03, "predecessor": {**proof03, **changes}}, **flags)
        assert remote_seed(tmp_path, seed)["run"] == successor.name
    stage03_sha = publish(stage03, record03, **flags)
    row = remote_seed(tmp_path, seed)
    assert (row["run"], row["segment"], row["labels"], row["contract_sha256"]) == (
        stage03.name, "ipv4_relay_retry_03", labels, stage03_sha,
    )
    assert row["relay_not_before_unix"] == exact["relay_not_before_unix"] and row["sealed_relay_retry_seconds"] == 180
    assert row["api"] == {"status": "relay_boundary_cooldown", "relay_retry_wait_seconds": 20}
    assert prompt_pilot_status.dispatcher_pid(seed, row["segment"]) == 1234
    assert commands[-1][-1].endswith(f"/com.purellm.a7-ipv4-v13-{seed}-relay-retry-03")

    stage, stage_sha, record = stage03, stage03_sha, record03
    for wave in (4, 5, 6):
        stage, stage_sha, record, chain = check_reviewed_wave(
            tmp_path, monkeypatch, publish, flags, commands, stage, stage_sha, record, chain, seed, start, labels, wave,
        )


def check_reviewed_wave(runs, monkeypatch, publish, flags, commands, parent, parent_sha, parent_record, chain, seed, start,
                        labels, wave):
    parent_auth = prompt_pilot_status.record_digest(prompt_pilot_status.read_json(parent / "authorization.json"))
    run = parent.with_name(f"{prompt_pilot_status.SERIAL_IPV4}-{seed}-relay-retry-{wave:02d}")
    prior = f"RELAY_{seed}_WAVE{wave - 1:02d}"
    multiple = (seed, wave) == (2357, 4)
    physical = getattr(prompt_pilot_status, prior + "_ATTEMPTS") if multiple else (
        getattr(prompt_pilot_status, prior + "_ATTEMPT"),
    )
    turn = parent / f"seed_{seed}/round_{start:02d}/turn_00"
    terminal_shas, receipts = [], []
    for attempt in physical:
        archived = {**attempt, "status": "stopped"}
        terminal = turn / f"attempt_{attempt['attempt']:02d}.terminal.json"
        write_json(terminal, archived)
        terminal_shas.append(prompt_pilot_status.checksum(terminal))
        receipts.append({**attempt, "terminal_sha256": prompt_pilot_status.record_digest(archived)})
    if (seed, wave) in ((2357, 6), (2357, 7)):
        write_json(turn / "attempt_00.started.json", {"attempt": 0, "attempt_id": physical[0]["attempt_id"]})
    if multiple:
        replay_files = {}
        for name in prompt_pilot_status.RELAY_2357_WAVE03_REPLAY_FILES:
            write_json(turn / name, {"file": name})
            replay_files[name] = prompt_pilot_status.checksum(turn / name)
        monkeypatch.setattr(prompt_pilot_status, prior + "_TERMINALS", terminal_shas)
        monkeypatch.setattr(prompt_pilot_status, prior + "_REPLAY_FILES", replay_files)
    exact = {**getattr(prompt_pilot_status, f"RELAY_{seed}_WAVE{wave:02d}_PROOF"),
             "previous_terminal_file_sha256": prompt_pilot_status.checksum(terminal)}
    monkeypatch.setattr(prompt_pilot_status, f"RELAY_{seed}_WAVE{wave - 1:02d}_CONTRACT", parent_sha)
    monkeypatch.setattr(prompt_pilot_status, f"RELAY_{seed}_WAVE{wave - 1:02d}_AUTH", parent_auth)
    monkeypatch.setattr(prompt_pilot_status, f"RELAY_{seed}_WAVE{wave:02d}_PROOF", exact)
    monkeypatch.setattr(prompt_pilot_status.time, "time", lambda: exact["relay_not_before_unix"] - 25)
    last = {"contract_sha256": parent_sha, "authorization_sha256": parent_auth, "run_directory": str(parent),
            "request_id": exact["previous_request_id"], "terminal_file_sha256": exact["previous_terminal_file_sha256"],
            "progress_file_sha256": exact["previous_transport_progress_file_sha256"], "previous_attempts": receipts}
    proof = {**parent_record["predecessor"], **exact, "previous_run_directory": str(parent),
             "previous_contract_sha256": parent_sha, "previous_authorization_sha256": parent_auth,
             "previous_attempts": receipts, "reviewed_retry_chain": [*chain, last]}
    record = {**parent_record, "run_directory": str(run), "predecessor": proof, "budget": exact["execution_budget"]}
    if (seed, wave) in ((2357, 3), (2357, 6), (6101, 4), (6101, 6)):
        publish(run, record, **flags)
        assert remote_seed(runs, seed)["run"] == parent.name
        proof["recovery_budget"] = {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}
        publish(run, record, **flags)
        assert remote_seed(runs, seed)["run"] == parent.name
        record["profile"] = {**parent_record["profile"], "replay_eligible_from_round": start}
    if (seed, wave) in ((6101, 5), (2357, 5), (2357, 7)):
        proof["recovery_budget"] = {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
        record["profile"] = {**parent_record["profile"], "replay_eligible_from_round": start + 1}
    observation = getattr(prompt_pilot_status, f"RELAY_{seed}_WAVE{wave:02d}_OBSERVATION", None)
    if observation is not None:
        proof["previous_transport_observation"] = observation
        if (seed, wave) == (2357, 7):
            record["source_sha256"] = prompt_pilot_status.RELAY_12203_WAVE12_SOURCES
        else:
            record["source_sha256"] = (prompt_pilot_status.RELAY_2357_WAVE05_SOURCES if (seed, wave) in ((2357, 5), (2357, 6))
                                       else prompt_pilot_status.RELAY_RETRY_QUEUE_SOURCES)
        publish(run, {**record, "source_sha256": {}}, **flags)
        assert remote_seed(runs, seed)["run"] == parent.name
        publish(run, {**record, "predecessor": {**proof, "previous_transport_observation": {}}}, **flags)
        assert remote_seed(runs, seed)["run"] == parent.name
    publish(run, record)
    assert remote_seed(runs, seed)["run"] == parent.name
    invalid_proofs = [
        {"previous_authorization_sha256": "a" * 64}, {"reviewed_retry_chain": [last, *chain]},
        {"reviewed_retry_chain": [{**chain[0], "previous_attempts": chain[0]["previous_attempts"][1:]}, *chain[1:], last]},
        {"previous_attempts": [{**receipts[0], "terminal_sha256": "a" * 64}]},
    ]
    if seed == 887:
        invalid_proofs += [
            {"reviewed_cached_import": {**proof["reviewed_cached_import"], "provider_post": True}},
            {"cached_import": proof["reviewed_cached_import"]},
        ]
    if multiple:
        invalid_proofs += [{"previous_attempts": receipts[1:]}, {"previous_attempts": list(reversed(receipts))}]
    for bad in invalid_proofs:
        publish(run, {**record, "predecessor": {**proof, **bad}}, **flags)
        assert remote_seed(runs, seed)["run"] == parent.name
    current_sha = publish(run, record, **flags)
    if (seed, wave) in ((6101, 5), (6101, 6), (2357, 5), (2357, 6), (2357, 7)):
        monkeypatch.setattr(prompt_pilot_status, f"RELAY_{seed}_WAVE{wave:02d}_CONTRACT", current_sha)
        actual_auth = prompt_pilot_status.record_digest(prompt_pilot_status.read_json(run / "authorization.json"))
        monkeypatch.setattr(prompt_pilot_status, f"RELAY_{seed}_WAVE{wave:02d}_AUTH", actual_auth)
    row = remote_seed(runs, seed)
    assert (row["run"], row["segment"], row["labels"], row["contract_sha256"]) == (
        run.name, f"ipv4_relay_retry_{wave:02d}", labels, current_sha,
    )
    assert row["api"] == {"status": "relay_boundary_cooldown", "relay_retry_wait_seconds": 25}
    if (seed, wave) == (2357, 3):
        assert row["boundary_api"] == {
            "maximum_logical_requests": 1, "maximum_physical_attempts": 2, "replay_delay_seconds": 180,
        }
    elif (seed, wave) == (6101, 4):
        assert row["boundary_api"] == {
            "maximum_logical_requests": 1, "maximum_physical_attempts": 2, "replay_delay_seconds": 180,
            "physical_attempts_reserved": 0, "physical_attempts_posted_confirmed": 0, "physical_attempts_remaining": 2,
        }
        assert row["reviewed_previous_api"] == {
            "http_status": 200, "error_type": "hard_wall_deadline", "sse_event_count": 3016,
            "event_type": "response.output_text.delta", "last_sse_seconds": 3531.8199360000435,
            "complete_event_seen": False, "sse_inactivity_seconds": 900,
        }
        publish(run, {**record, "budget": parent_record["budget"]}, **flags)
        assert remote_seed(runs, seed)["run"] == parent.name
        publish(run, record, **flags)
    elif (seed, wave) == (6101, 5):
        assert row["boundary_api"] == {
            "maximum_logical_requests": 1, "maximum_physical_attempts": 1, "replay_delay_seconds": 180,
            "physical_attempts_reserved": 0, "physical_attempts_posted_confirmed": 0, "physical_attempts_remaining": 1,
        }
        assert row["reviewed_previous_api"] == {
            "http_status": 200, "error_type": "ReadError", "sse_event_count": 252,
            "event_type": "response.output_text.delta", "last_sse_seconds": 1201.01432175003,
            "complete_event_seen": False,
        }
        second = turn / "attempt_01.started.json"
        write_json(second, {"attempt": 1})
        assert remote_seed(runs, seed)["run"] == parent.name
        second.unlink()
    elif (seed, wave) == (6101, 6):
        assert row["boundary_api"]["physical_attempts_remaining"] == 2
        assert row["reviewed_previous_api"] == {
            "http_status": 200, "error_type": "response.failed", "provider_error_code": "server_error",
            "sse_event_count": 41, "event_type": "response.failed", "last_sse_seconds": 318.5360944159329,
            "complete_event_seen": False,
        }
    elif (seed, wave) == (2357, 5):
        assert row["boundary_api"]["physical_attempts_remaining"] == 1
        assert row["boundary_api"]["future_replay_from_round"] == 5
        assert row["reviewed_previous_api"] == {
            "http_status": 200, "error_type": "hard_wall_deadline", "sse_event_count": 3170,
            "event_type": "response.output_text.delta", "last_sse_seconds": 3181.4656675001606,
            "complete_event_seen": False, "hard_deadline_seconds": 3600,
        }
        second = turn / "attempt_01.started.json"
        write_json(second, {"attempt": 1})
        assert remote_seed(runs, seed)["run"] == parent.name
        second.unlink()
    elif (seed, wave) == (2357, 6):
        assert row["boundary_api"] == {
            "maximum_logical_requests": 1, "maximum_physical_attempts": 2, "replay_delay_seconds": 180,
            "physical_attempts_reserved": 0, "physical_attempts_posted_confirmed": 0, "physical_attempts_remaining": 2,
            "future_replay_from_round": 5, "future_replay_requires_verified_response_id": True,
            "replay_requires_verified_response_id": True,
        }
        assert row["reviewed_previous_api"] == {
            "http_status": 200, "error_type": "sse_inactivity_timeout", "sse_event_count": 831,
            "event_type": "response.output_text.delta", "last_sse_seconds": 1604.6706121659372,
            "complete_event_seen": False, "sse_inactivity_seconds": 900,
        }
        second = turn / "attempt_01.started.json"
        write_json(second, {"attempt": 1})
        assert remote_seed(runs, seed)["run"] == parent.name
        second.unlink()
    elif (seed, wave) == (2357, 7):
        assert row["boundary_api"] == {
            "maximum_logical_requests": 1, "maximum_physical_attempts": 1, "replay_delay_seconds": 180,
            "physical_attempts_reserved": 0, "physical_attempts_posted_confirmed": 0, "physical_attempts_remaining": 1,
            "future_replay_from_round": 5, "future_replay_requires_verified_response_id": True,
        }
        assert row["reviewed_previous_api"] == {
            "http_status": 200, "error_type": "sse_inactivity_timeout", "sse_event_count": 440,
            "event_type": "response.output_text.delta", "last_sse_seconds": 991.6895799168851,
            "complete_event_seen": False, "sse_inactivity_seconds": 900,
        }
        second = turn / "attempt_01.started.json"
        write_json(second, {"attempt": 1})
        assert remote_seed(runs, seed)["run"] == parent.name
        second.unlink()
    elif multiple:
        assert row["boundary_api"]["maximum_physical_attempts"] == 2
        assert row["boundary_api"]["physical_attempts_remaining"] == 2
        assert row["reviewed_previous_api"] == {
            "http_status": 200, "error_type": "sse_inactivity_timeout", "sse_event_count": 14,
            "event_type": "response.output_item.done", "last_sse_seconds": 289.73664454207756,
            "complete_event_seen": False, "sse_inactivity_seconds": 900,
        }
        link = turn / "attempt_01.replay_link.json"
        original_link = link.read_bytes()
        write_json(link, {"detached": True})
        assert remote_seed(runs, seed)["run"] == parent.name
        link.write_bytes(original_link)
    else:
        assert "boundary_api" not in row
    assert prompt_pilot_status.dispatcher_pid(seed, row["segment"]) == 1234
    assert commands[-1][-1].endswith(f"/com.purellm.a7-ipv4-v13-{seed}-relay-retry-{wave:02d}")
    original_terminal = terminal.read_bytes()
    write_json(terminal, {**archived, "status": "unknown"})
    assert remote_seed(runs, seed)["run"] == parent.name
    terminal.write_bytes(original_terminal)
    return run, current_sha, record, [*chain, last]


def check_887_reviewed_cached_r8(runs, monkeypatch, publish, flags, commands, old, old_sha, old_record):
    cached = runs / f"{prompt_pilot_status.SERIAL_IPV4}-887-cached-r7-01"
    retired = runs / f"{prompt_pilot_status.SERIAL_IPV4}-887-relay-retry-01"
    successor = runs / f"{prompt_pilot_status.SERIAL_IPV4}-887-relay-retry-02"
    cached_plan = {"provider_post": False, "normalized_plan_sha256": prompt_pilot_status.CACHED_R7_PLAN_SHA}
    old_auth = prompt_pilot_status.record_digest(prompt_pilot_status.read_json(old / "authorization.json"))
    cached_proof = {**old_record["predecessor"], "dispatcher_module": "dispatch_v13_887_cached_r7",
                    "previous_run_directory": str(old), "previous_contract_sha256": old_sha,
                    "previous_authorization_sha256": old_auth, "cached_import": cached_plan,
                    "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 0}}
    cached_profile = {**old_record["profile"], "audited_future_transport_replay": True, "replay_eligible_from_round": 8}
    cached_record = {**old_record, "run_directory": str(cached), "profile": cached_profile, "predecessor": cached_proof}
    cache_receipt = cached / "cached_response_import.json"
    write_json(cache_receipt, cached_plan)
    write_json(cached / "coordinator_generation.json", {"pid": 1234, "id": "a" * 32})
    cached_sha = publish(cached, cached_record, **flags)
    cached_auth = prompt_pilot_status.record_digest(prompt_pilot_status.read_json(cached / "authorization.json"))
    monkeypatch.setattr(prompt_pilot_status, "RELAY_887_CACHED_CONTRACT", cached_sha)
    monkeypatch.setattr(prompt_pilot_status, "RELAY_887_CACHED_AUTH", cached_auth)
    cached_review = {"proof_sha256": prompt_pilot_status.record_digest(cached_plan),
                     "receipt_file_sha256": prompt_pilot_status.checksum(cache_receipt), "provider_post": False,
                     "resource_contract_sha256": old_sha, "resource_authorization_sha256": old_auth}
    monkeypatch.setattr(prompt_pilot_status, "RELAY_887_CACHED_REVIEW", cached_review)

    directory = cached / "seed_887"
    current = directory / "round_08"
    source, transition = current / "source.json", current / "transition.json"
    metrics, memory = current / "fit_000/validation_metrics.json", directory / "round_07/memory.json"
    freeze = directory / "round_07/batch_freeze.json"
    cached_terminal = directory / "round_07/turn_00/attempt_00.terminal.json"
    local_attempt = "5" * 32
    write_json(source, {"round": 8})
    write_json(metrics, {"combined_normalized_rmse": 0.2})
    write_json(memory, {"hypotheses": []})
    write_json(freeze, {"mock": False})
    write_json(cached_terminal, {"status": "completed", "provider_dispatched": False, "attempt_id": local_attempt})
    write_json(transition, {"fit_path": "fit_000", "checkpoint_sha256": "c" * 64,
                            "source_sha256": prompt_pilot_status.checksum(source),
                            "validation_metrics_sha256": prompt_pilot_status.checksum(metrics)})
    fitted = {path.relative_to(directory).as_posix(): prompt_pilot_status.checksum(path)
              for path in (source, transition, memory, freeze, cached_terminal)}
    monkeypatch.setattr(prompt_pilot_status, "RELAY_887_FITTED_PREFIX", fitted)

    attempt = prompt_pilot_status.RELAY_887_R8_ATTEMPT
    archived = {**attempt, "status": "stopped"}
    terminal = current / "turn_00/attempt_00.terminal.json"
    write_json(terminal, archived)
    receipts = [{**attempt, "terminal_sha256": prompt_pilot_status.record_digest(archived)}]
    exact = {**prompt_pilot_status.RELAY_887_R8_PROOF, "previous_terminal_file_sha256": prompt_pilot_status.checksum(terminal)}
    monkeypatch.setattr(prompt_pilot_status, "RELAY_887_R8_PROOF", exact)
    monkeypatch.setattr(prompt_pilot_status.time, "time", lambda: exact["relay_not_before_unix"] - 30)
    chain = [{"contract_sha256": cached_sha, "authorization_sha256": cached_auth, "run_directory": str(cached),
              "request_id": exact["previous_request_id"], "terminal_file_sha256": exact["previous_terminal_file_sha256"],
              "progress_file_sha256": exact["previous_transport_progress_file_sha256"], "previous_attempts": receipts}]
    proof = {**exact, "dispatcher_module": "dispatch_v13_retry_successor", "seed": 887, "round": 8,
             "previous_run_directory": str(cached), "previous_contract_sha256": cached_sha,
             "previous_authorization_sha256": cached_auth, "previous_attempts": receipts,
             "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 1},
             "reviewed_fitted_prefix_files": fitted, "reviewed_retry_chain": chain, "reviewed_cached_import": cached_review,
             "previous_transport_observation": prompt_pilot_status.RELAY_887_R8_OBSERVATION,
             "public_resource_receipt_sha256": prompt_pilot_status.RELAY_887_RESOURCE_SHA}
    record = {**cached_record, "run_directory": str(successor), "predecessor": proof,
              "source_sha256": prompt_pilot_status.RELAY_887_R8_WAVE02_SOURCES,
              "profile": {**cached_profile, "resume_boundaries": {"887": {"round": 8}},
                          "replay_eligible_from_round": 9, "sealed_relay_retry_seconds": 180}}
    retired_sha = publish(retired, {**record, "run_directory": str(retired),
                                   "source_sha256": prompt_pilot_status.RELAY_887_R8_SOURCES}, **flags)
    retired_auth = prompt_pilot_status.record_digest(prompt_pilot_status.read_json(retired / "authorization.json"))
    assert remote_seed(runs, 887)["run"] == cached.name
    publish(successor, record)
    assert remote_seed(runs, 887)["segment"] == "ipv4_cached_r7"
    for bad in ({"reviewed_cached_import": {**cached_review, "provider_post": True}}, {"cached_import": cached_plan},
                {"reviewed_retry_chain": []}, {"previous_attempts": []}):
        publish(successor, {**record, "predecessor": {**proof, **bad}}, **flags)
        assert remote_seed(runs, 887)["run"] == cached.name
    publish(successor, {**record, "source_sha256": {}}, **flags)
    assert remote_seed(runs, 887)["run"] == cached.name
    missing_helper = {name: sha for name, sha in record["source_sha256"].items()
                      if not name.endswith("/dispatch_v13_relay_retry.py")}
    publish(successor, {**record, "source_sha256": missing_helper}, **flags)
    assert remote_seed(runs, 887)["run"] == cached.name
    publish(successor, {**record, "predecessor": {**proof, "previous_run_directory": str(retired),
                                                 "previous_contract_sha256": retired_sha,
                                                 "previous_authorization_sha256": retired_auth}}, **flags)
    assert remote_seed(runs, 887)["run"] == cached.name
    successor_sha = publish(successor, record, **flags)
    row = remote_seed(runs, 887)
    assert (row["run"], row["segment"], row["labels"], row["contract_sha256"]) == (
        successor.name, "ipv4_relay_retry_02", 589, successor_sha,
    )
    assert row["recovery_round"] == 8 and row["sealed_relay_retry_seconds"] == 180
    assert row["api"] == {"status": "relay_boundary_cooldown", "relay_retry_wait_seconds": 30}
    assert prompt_pilot_status.dispatcher_pid(887, row["segment"]) == 1234
    assert commands[-1][-1].endswith("/com.purellm.a7-ipv4-v13-887-relay-retry-02")
    stage03, stage03_sha, record03, chain03 = check_reviewed_wave(
        runs, monkeypatch, publish, flags, commands, successor, successor_sha, record, chain, 887, 8, 589, 3,
    )
    stage04, stage04_sha, record04, _ = check_reviewed_wave(
        runs, monkeypatch, publish, flags, commands, stage03, stage03_sha, record03, chain03, 887, 8, 589, 4,
    )
    repair_run = runs / prompt_pilot_status.REPAIR_887_RUN
    repair_record = {**record04, "run_directory": str(repair_run), "profile": {**record04["profile"], "target_labels": 621},
                     "predecessor": {**record04["predecessor"], "dispatcher_module": "dispatch_v13_887_r8_targeted_repair",
                                     "previous_run_directory": str(stage04), "previous_contract_sha256": stage04_sha,
                                     "prior_response_reposted": False}}
    repair_sha = publish(repair_run, repair_record, **flags)
    repair_auth = prompt_pilot_status.record_digest(prompt_pilot_status.read_json(repair_run / "authorization.json"))
    monkeypatch.setattr(prompt_pilot_status, "REPAIR_887_PARENT", stage04_sha)
    monkeypatch.setattr(prompt_pilot_status, "REPAIR_887_CONTRACT", repair_sha)
    assert remote_seed(runs, 887)["run"] == stage04.name
    monkeypatch.setattr(prompt_pilot_status, "REPAIR_887_AUTH", repair_auth)
    row = remote_seed(runs, 887)
    assert (row["run"], row["segment"], row["labels"]) == (repair_run.name, "ipv4_r8_targeted_repair", 589)
    assert "relay_not_before_unix" not in row
    assert prompt_pilot_status.dispatcher_pid(887, row["segment"]) == 1234
    assert commands[-1][-1].endswith("/com.purellm.a7-ipv4-v13-887-r8-targeted-repair-01")
    following = repair_run / "seed_887/round_09"
    write_json(following / "source.json", {"round": 9})
    write_json(following / "fit_000/validation_metrics.json", {"combined_normalized_rmse": 0.25})
    write_json(following / "transition.json", {
        "fit_path": "fit_000", "checkpoint_sha256": "c" * 64,
        "source_sha256": prompt_pilot_status.checksum(following / "source.json"),
        "validation_metrics_sha256": prompt_pilot_status.checksum(following / "fit_000/validation_metrics.json"),
    })
    row = remote_seed(runs, 887)
    assert (row["labels"], row["phase"]) == (621, "targeted_repair_round_complete")
    cached_run = runs / prompt_pilot_status.CACHED_887_R8_RUN
    continuation_run = runs / prompt_pilot_status.CONTINUATION_887_R9_RUN
    custody = {"provider_post": False, "projected_plan_sha256": prompt_pilot_status.CACHED_887_R8_PROJECTION}
    receipt = cached_run / "cached_correction.json"
    write_json(receipt, custody)
    cached_record = {**repair_record, "run_directory": str(cached_run),
                     "predecessor": {"dispatcher_module": "run_v13_887_r8_cached_correction", "previous_run_directory": str(repair_run),
                                     "previous_contract_sha256": repair_sha, "previous_authorization_sha256": repair_auth,
                                     "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 0},
                                     "cached_correction": custody}}
    cached_sha = publish(cached_run, cached_record, **flags)
    cached_auth = prompt_pilot_status.record_digest(prompt_pilot_status.read_json(cached_run / "authorization.json"))
    monkeypatch.setattr(prompt_pilot_status, "CACHED_887_R8_CONTRACT", cached_sha)
    monkeypatch.setattr(prompt_pilot_status, "CACHED_887_R8_AUTH", cached_auth)
    monkeypatch.setattr(prompt_pilot_status, "CACHED_887_R8_RECEIPT", prompt_pilot_status.checksum(receipt))
    for relative in ("source.json", "transition.json", "fit_000/validation_metrics.json"):
        write_json(cached_run / "seed_887/round_09" / relative, prompt_pilot_status.read_json(following / relative))
    cached_fit = cached_run / "seed_887/round_09"
    boundary = {"checkpoint_sha256": "c" * 64, "extension_source_sha256": prompt_pilot_status.checksum(cached_fit / "source.json"),
                "extension_transition_sha256": prompt_pilot_status.checksum(cached_fit / "transition.json")}
    monkeypatch.setattr(prompt_pilot_status, "CACHED_887_R8_BOUNDARY", boundary)
    row = remote_seed(runs, 887)
    assert (row["run"], row["labels"], row["phase"]) == (cached_run.name, 621, "cached_correction_round_complete")
    assert row["archived_correction_provider_post"] is False
    assert prompt_pilot_status.dispatcher_pid(887, row["segment"]) is None
    next_record = {"run_directory": str(continuation_run), "source_sha256": prompt_pilot_status.CONTINUATION_887_R9_SOURCES,
                   "budget": prompt_pilot_status.CONTINUATION_887_R9_BUDGET,
                   "profile": {**repair_record["profile"], "target_labels": 1485, "resume_boundaries": {"887": {"round": 9}},
                               "replay_eligible_from_round": 9, "sealed_relay_retry_seconds": 180,
                               "audited_future_transport_replay": True, "sse_inactivity_seconds": 900,
                               "request_timeouts": {"total_seconds": 3600},
                               "transport": {"requested_model": "gpt-6.2-astra", "reasoning_effort": "xhigh", "sdk_version": "2.28.0",
                                             "logical_replies": 3, "transport_attempts": 2}},
                   "predecessor": {**boundary, "dispatcher_module": "dispatch_v13_887_r9_continuation", "seed": 887, "round": 9,
                                   "extension_run_directory": str(cached_run), "extension_contract_sha256": cached_sha,
                                   "extension_authorization_sha256": cached_auth, "paid_correction_contract_sha256": repair_sha,
                                   "paid_correction_authorization_sha256": repair_auth, "cached_correction_file_sha256": prompt_pilot_status.CACHED_887_R8_RECEIPT,
                                   "extension_projected_plan_sha256": custody["projected_plan_sha256"],
                                   "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 2},
                                   "execution_budget": {"first_logical_requests": 27, "maximum_logical_requests": 79, "maximum_physical_attempts": 158}}}
    publish(continuation_run, next_record)
    assert remote_seed(runs, 887)["run"] == cached_run.name
    next_sha = publish(continuation_run, next_record, **flags)
    monkeypatch.setattr(prompt_pilot_status, "CONTINUATION_887_R9_CONTRACT", next_sha)
    assert remote_seed(runs, 887)["run"] == cached_run.name
    next_auth = prompt_pilot_status.record_digest(prompt_pilot_status.read_json(continuation_run / "authorization.json"))
    monkeypatch.setattr(prompt_pilot_status, "CONTINUATION_887_R9_AUTH", next_auth)
    row = remote_seed(runs, 887)
    assert (row["run"], row["labels"], row["contract_sha256"]) == (continuation_run.name, 621, next_sha)
    assert row["boundary_api"]["physical_attempts_remaining"] == 2
    assert prompt_pilot_status.dispatcher_pid(887, row["segment"]) == 1234
    assert commands[-1][-1].endswith("/com.purellm.a7-ipv4-v13-887-r9-continuation-01")
    check_887_r12_wave05(runs, monkeypatch, publish, flags, commands, continuation_run, next_sha, next_record)
    write_json(cached_fit / "source.json", {"round": 8})
    assert remote_seed(runs, 887)["run"] == cached_run.name
    write_json(cached / "bridge/outbox" / f"{local_attempt}.json", {"unexpected": "provider attempt"})
    assert remote_seed(runs, 887)["run"] == cached.name


def check_887_r12_wave05(runs, monkeypatch, publish, flags, commands, previous, previous_sha, previous_record):
    status = prompt_pilot_status
    directory = previous / "seed_887"
    for name in status.RELAY_887_R12_FITTED_FILES:
        write_json(directory / name, {"fixture": name})
    fitted = directory / "round_12"
    write_json(fitted / "source.json", {"checkpoint": status.RELAY_887_R12_CHECKPOINT})
    metrics = fitted / "fit_000/validation_metrics.json"
    write_json(metrics, {"combined_normalized_rmse": .4})
    write_json(fitted / "transition.json", {"checkpoint_sha256": status.RELAY_887_R12_CHECKPOINT, "fit_path": "fit_000",
                                            "source_sha256": status.checksum(fitted / "source.json"),
                                            "validation_metrics_sha256": status.checksum(metrics)})
    fitted_files = {name: status.checksum(directory / name) for name in status.RELAY_887_R12_FITTED_FILES}
    monkeypatch.setattr(status, "RELAY_887_R12_FITTED_FILES", fitted_files)
    turn = fitted / "turn_00"
    attempt = status.RELAY_887_R12_ATTEMPT
    terminal = {**attempt, "status": "stopped"}
    terminal_path = turn / "attempt_00.terminal.json"
    write_json(terminal_path, terminal)
    write_json(turn / "attempt_00.started.json", {"attempt": 0, "attempt_id": attempt["attempt_id"]})
    exact = {**status.RELAY_887_WAVE05_PROOF, "previous_terminal_file_sha256": status.checksum(terminal_path)}
    monkeypatch.setattr(status, "RELAY_887_WAVE05_PROOF", exact)
    receipt = {**attempt, "terminal_sha256": status.record_digest(terminal)}
    history = [status._relay_review(previous, previous_sha, status.CONTINUATION_887_R9_AUTH, exact, [receipt])]
    run = runs / f"{status.SERIAL_IPV4}-887-relay-retry-05"
    proof = {**exact, "dispatcher_module": "dispatch_v13_retry_successor", "previous_run_directory": str(previous),
             "previous_contract_sha256": previous_sha, "previous_authorization_sha256": status.CONTINUATION_887_R9_AUTH,
             "round": 12, "seed": 887, "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 1},
             "previous_transport_observation": status.RELAY_887_WAVE05_OBSERVATION, "previous_attempts": [receipt],
             "reviewed_retry_chain": history, "reviewed_fitted_prefix_files": fitted_files,
             "checkpoint_sha256": status.RELAY_887_R12_CHECKPOINT}
    record = {"run_directory": str(run), "source_sha256": status.RELAY_2357_WAVE05_SOURCES,
              "budget": exact["execution_budget"], "predecessor": proof,
              "profile": {**previous_record["profile"], "resume_boundaries": {"887": {"round": 12}},
                          "replay_eligible_from_round": 13}}
    sha = publish(run, record, **flags)
    monkeypatch.setattr(status, "RELAY_887_WAVE05_CONTRACT", sha)
    assert remote_seed(runs, 887)["run"] == previous.name
    authorization = status.record_digest(status.read_json(run / "authorization.json"))
    monkeypatch.setattr(status, "RELAY_887_WAVE05_AUTH", authorization)
    monkeypatch.setattr(status.time, "time", lambda: exact["relay_not_before_unix"] - 30)
    row = remote_seed(runs, 887)
    assert (row["run"], row["segment"], row["labels"], row["contract_sha256"]) == (run.name, "ipv4_relay_retry_05", 717, sha)
    assert row["checkpoint_sha256"] == status.RELAY_887_R12_CHECKPOINT
    assert row["archived_correction_provider_post"] is False
    assert row["api"] == {"status": "relay_boundary_cooldown", "relay_retry_wait_seconds": 30}
    assert row["reviewed_previous_api"] == {
        "http_status": 200, "error_type": "sse_inactivity_timeout", "sse_event_count": 207,
        "event_type": "response.output_text.delta", "last_sse_seconds": 961.4464893341064,
        "complete_event_seen": False, "sse_inactivity_seconds": 900,
    }
    assert row["boundary_api"]["maximum_physical_attempts"] == 1
    assert row["boundary_api"]["future_replay_from_round"] == 13
    assert row["boundary_api"]["future_replay_requires_verified_response_id"] is True
    assert status.dispatcher_pid(887, row["segment"]) == 1234
    assert commands[-1][-1].endswith("/com.purellm.a7-ipv4-v13-887-relay-retry-05")
    check_887_r12_wave06(runs, monkeypatch, publish, flags, commands, run, sha, record)
    write_json(turn / "attempt_01.started.json", {"attempt": 1})
    assert remote_seed(runs, 887)["run"] == previous.name
    (turn / "attempt_01.started.json").unlink()
    write_json(directory / "round_11/memory.json", {"changed": True})
    assert remote_seed(runs, 887)["run"] == previous.name


def check_887_r12_wave06(runs, monkeypatch, publish, flags, commands, previous, previous_sha, previous_record):
    status = prompt_pilot_status
    turn = previous / "seed_887/round_12/turn_00"
    attempt = status.RELAY_887_WAVE05_ATTEMPT
    terminal = {**attempt, "status": "stopped"}
    terminal_path = turn / "attempt_00.terminal.json"
    write_json(terminal_path, terminal)
    write_json(turn / "attempt_00.started.json", {"attempt": 0, "attempt_id": attempt["attempt_id"]})
    exact = {**status.RELAY_887_WAVE06_PROOF, "previous_terminal_file_sha256": status.checksum(terminal_path)}
    monkeypatch.setattr(status, "RELAY_887_WAVE06_PROOF", exact)
    receipt = {**attempt, "terminal_sha256": status.record_digest(terminal)}
    last = status._relay_review(previous, previous_sha, status.RELAY_887_WAVE05_AUTH, exact, [receipt])
    old_proof = previous_record["predecessor"]
    run = runs / f"{status.SERIAL_IPV4}-887-relay-retry-06"
    proof = {**old_proof, **exact, "previous_run_directory": str(previous), "previous_contract_sha256": previous_sha,
             "previous_authorization_sha256": status.RELAY_887_WAVE05_AUTH,
             "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 2},
             "previous_transport_observation": status.RELAY_887_WAVE06_OBSERVATION, "previous_attempts": [receipt],
             "reviewed_retry_chain": [*old_proof["reviewed_retry_chain"], last]}
    record = {**previous_record, "run_directory": str(run), "source_sha256": status.RELAY_12203_WAVE12_SOURCES,
              "budget": exact["execution_budget"], "predecessor": proof,
              "profile": {**previous_record["profile"], "replay_eligible_from_round": 12}}
    sha = publish(run, record, **flags)
    monkeypatch.setattr(status, "RELAY_887_WAVE06_CONTRACT", sha)
    assert remote_seed(runs, 887)["run"] == previous.name
    auth = status.record_digest(status.read_json(run / "authorization.json"))
    monkeypatch.setattr(status, "RELAY_887_WAVE06_AUTH", auth)
    monkeypatch.setattr(status.time, "time", lambda: exact["relay_not_before_unix"] - 30)
    row = remote_seed(runs, 887)
    assert (row["run"], row["segment"], row["labels"], row["contract_sha256"]) == (run.name, "ipv4_relay_retry_06", 717, sha)
    assert row["checkpoint_sha256"] == status.RELAY_887_R12_CHECKPOINT
    assert row["archived_correction_provider_post"] is False
    assert row["api"] == {"status": "relay_boundary_cooldown", "relay_retry_wait_seconds": 30}
    assert row["reviewed_previous_api"] == {
        "http_status": 200, "error_type": "hard_wall_deadline", "sse_event_count": 3063,
        "event_type": "response.output_text.delta", "last_sse_seconds": 3017.041881625075,
        "complete_event_seen": False, "hard_deadline_seconds": 3600,
    }
    assert row["boundary_api"]["maximum_physical_attempts"] == 2
    assert row["boundary_api"]["replay_requires_verified_response_id"] is True
    assert row["boundary_api"]["future_replay_from_round"] == 13
    assert status.dispatcher_pid(887, row["segment"]) == 1234
    assert commands[-1][-1].endswith("/com.purellm.a7-ipv4-v13-887-relay-retry-06")
    write_json(turn / "attempt_01.started.json", {"unexpected": True})
    assert remote_seed(runs, 887)["run"] == previous.name
    (turn / "attempt_01.started.json").unlink()
    check_887_r12_wave07(runs, monkeypatch, publish, flags, commands, run, sha, record)


def check_887_r12_wave07(runs, monkeypatch, publish, flags, commands, previous, previous_sha, previous_record):
    status = prompt_pilot_status
    turn = previous / "seed_887/round_12/turn_00"
    attempt = status.RELAY_887_WAVE06_ATTEMPT
    terminal = {**attempt, "status": "stopped"}
    terminal_path = turn / "attempt_00.terminal.json"
    write_json(terminal_path, terminal)
    write_json(turn / "attempt_00.started.json", {"attempt": 0, "attempt_id": attempt["attempt_id"]})
    exact = {**status.RELAY_887_WAVE07_PROOF, "previous_terminal_file_sha256": status.checksum(terminal_path)}
    monkeypatch.setattr(status, "RELAY_887_WAVE07_PROOF", exact)
    receipt = {**attempt, "terminal_sha256": status.record_digest(terminal)}
    last = status._relay_review(previous, previous_sha, status.RELAY_887_WAVE06_AUTH, exact, [receipt])
    original = previous_record["predecessor"]
    run = runs / f"{status.SERIAL_IPV4}-887-relay-retry-07"
    proof = {**original, **exact, "previous_run_directory": str(previous), "previous_contract_sha256": previous_sha,
             "previous_authorization_sha256": status.RELAY_887_WAVE06_AUTH,
             "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 1},
             "previous_transport_observation": status.RELAY_887_WAVE07_OBSERVATION, "previous_attempts": [receipt],
             "reviewed_retry_chain": [*original["reviewed_retry_chain"], last]}
    record = {**previous_record, "run_directory": str(run), "budget": exact["execution_budget"], "predecessor": proof,
              "profile": {**previous_record["profile"], "replay_eligible_from_round": 13}}
    sha = publish(run, record, **flags)
    monkeypatch.setattr(status, "RELAY_887_WAVE07_CONTRACT", sha)
    assert remote_seed(runs, 887)["run"] == previous.name
    auth = status.record_digest(status.read_json(run / "authorization.json"))
    monkeypatch.setattr(status, "RELAY_887_WAVE07_AUTH", auth)
    monkeypatch.setattr(status.time, "time", lambda: exact["relay_not_before_unix"] - 30)
    row = remote_seed(runs, 887)
    assert (row["run"], row["segment"], row["labels"], row["contract_sha256"]) == (run.name, "ipv4_relay_retry_07", 717, sha)
    assert row["checkpoint_sha256"] == status.RELAY_887_R12_CHECKPOINT
    assert row["archived_correction_provider_post"] is False
    assert row["api"] == {"status": "relay_boundary_cooldown", "relay_retry_wait_seconds": 30}
    assert row["reviewed_previous_api"] == {
        "http_status": 200, "error_type": "hard_wall_deadline", "sse_event_count": 4060,
        "event_type": "response.output_item.done", "last_sse_seconds": 2876.7962896670215,
        "complete_event_seen": False, "hard_deadline_seconds": 3600, "output_text_done_seconds": 2861.863800625084,
    }
    assert row["boundary_api"]["maximum_physical_attempts"] == 1
    assert row["boundary_api"]["future_replay_from_round"] == 13
    assert row["boundary_api"]["future_replay_requires_verified_response_id"] is True
    assert status.dispatcher_pid(887, row["segment"]) == 1234
    assert commands[-1][-1].endswith("/com.purellm.a7-ipv4-v13-887-relay-retry-07")
    write_json(turn / "attempt_01.started.json", {"unexpected": True})
    assert remote_seed(runs, 887)["run"] == previous.name
    (turn / "attempt_01.started.json").unlink()

    def publish_next(destination, candidate):
        digest = publish(destination, candidate, **flags)
        return digest, status.record_digest(status.read_json(destination / "authorization.json"))

    check_repeated_boundary(runs, monkeypatch, publish_next, 887, 8, run, sha, auth, record)


def check_6101_reviewed_wave02(runs, monkeypatch, publish, flags, commands, relay, relay_sha, relay_auth, relay_record):
    directory = relay / "seed_6101"
    current = directory / "round_02"
    source, transition = current / "source.json", current / "transition.json"
    metrics, memory = current / "fit_000/validation_metrics.json", directory / "round_01/memory.json"
    write_json(source, {"round": 2})
    write_json(metrics, {"combined_normalized_rmse": 0.25})
    write_json(memory, {"hypotheses": []})
    write_json(transition, {"fit_path": "fit_000", "checkpoint_sha256": "c" * 64,
                            "source_sha256": prompt_pilot_status.checksum(source),
                            "validation_metrics_sha256": prompt_pilot_status.checksum(metrics)})
    fitted = {path.relative_to(directory).as_posix(): prompt_pilot_status.checksum(path)
              for path in (source, transition, memory)}
    receipts, files = [], []
    for attempt in prompt_pilot_status.RELAY_6101_WAVE01_ATTEMPTS:
        terminal = {**attempt, "status": "stopped"}
        file = current / "turn_00" / f"attempt_{attempt['attempt']:02d}.terminal.json"
        write_json(file, terminal)
        receipts.append({**attempt, "terminal_sha256": prompt_pilot_status.record_digest(terminal)})
        files.append(file)
    terminal_shas = tuple(prompt_pilot_status.checksum(file) for file in files)
    exact = {**prompt_pilot_status.RELAY_6101_WAVE02_PROOF, "previous_terminal_file_sha256": terminal_shas[-1]}
    monkeypatch.setattr(prompt_pilot_status, "RELAY_6101_WAVE01_CONTRACT", relay_sha)
    monkeypatch.setattr(prompt_pilot_status, "RELAY_6101_WAVE01_AUTH", relay_auth)
    monkeypatch.setattr(prompt_pilot_status, "RELAY_6101_FITTED_PREFIX", fitted)
    monkeypatch.setattr(prompt_pilot_status, "RELAY_6101_WAVE01_TERMINALS", terminal_shas)
    monkeypatch.setattr(prompt_pilot_status, "RELAY_6101_WAVE02_PROOF", exact)
    monkeypatch.setattr(prompt_pilot_status.time, "time", lambda: exact["relay_not_before_unix"] - 20)
    successor = relay.with_name(f"{prompt_pilot_status.SERIAL_IPV4}-6101-relay-retry-02")
    chain = [{"contract_sha256": relay_sha, "authorization_sha256": relay_auth, "run_directory": str(relay),
              "request_id": exact["previous_request_id"], "terminal_file_sha256": terminal_shas[-1],
              "progress_file_sha256": exact["previous_transport_progress_file_sha256"], "previous_attempts": receipts}]
    proof = {**exact, "dispatcher_module": "dispatch_v13_retry_successor", "seed": 6101, "round": 2,
             "previous_run_directory": str(relay), "previous_contract_sha256": relay_sha,
             "previous_authorization_sha256": relay_auth, "previous_attempts": receipts,
             "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 1},
             "reviewed_fitted_prefix_files": fitted, "reviewed_retry_chain": chain}
    record = {**relay_record, "run_directory": str(successor), "predecessor": proof,
              "profile": {**relay_record["profile"], "resume_boundaries": {"6101": {"round": 2}},
                          "replay_eligible_from_round": 3}}
    publish(successor, record)
    assert remote_seed(runs, 6101)["run"] == relay.name
    for bad in ({"previous_attempts": receipts[1:]}, {"previous_attempts": list(reversed(receipts))},
                {"reviewed_fitted_prefix_files": {**fitted, "round_02/source.json": "a" * 64}},
                {"reviewed_retry_chain": [{**chain[0], "previous_attempts": receipts[:1]}]}):
        publish(successor, {**record, "predecessor": {**proof, **bad}}, **flags)
        assert remote_seed(runs, 6101)["run"] == relay.name
    publish(successor, {**record, "profile": {**record["profile"], "replay_eligible_from_round": 2}}, **flags)
    assert remote_seed(runs, 6101)["run"] == relay.name
    successor_sha = publish(successor, record, **flags)
    row = remote_seed(runs, 6101)
    assert (row["run"], row["segment"], row["labels"], row["contract_sha256"]) == (
        successor.name, "ipv4_relay_retry_02", 397, successor_sha,
    )
    assert row["recovery_round"] == 2 and row["sealed_relay_retry_seconds"] == 180
    assert row["api"] == {"status": "relay_boundary_cooldown", "relay_retry_wait_seconds": 20}
    assert prompt_pilot_status.dispatcher_pid(6101, row["segment"]) == 1234
    assert commands[-1][-1].endswith("/com.purellm.a7-ipv4-v13-6101-relay-retry-02")
    first_bytes = files[0].read_bytes()
    write_json(files[0], {"status": "unknown"})
    assert remote_seed(runs, 6101)["run"] == relay.name
    files[0].write_bytes(first_bytes)
    first_memory = memory.read_bytes()
    write_json(memory, {"hypotheses": ["changed after the reviewed fit"]})
    assert remote_seed(runs, 6101)["run"] == relay.name
    memory.write_bytes(first_memory)
    stage03, sha03, record03, chain03 = check_reviewed_wave(
        runs, monkeypatch, publish, flags, commands, successor, successor_sha, record, chain, 6101, 2, 397, 3,
    )
    stage04, sha04, record04, chain04 = check_reviewed_wave(
        runs, monkeypatch, publish, flags, commands, stage03, sha03, record03, chain03, 6101, 2, 397, 4,
    )
    stage05, sha05, record05, chain05 = check_reviewed_wave(
        runs, monkeypatch, publish, flags, commands, stage04, sha04, record04, chain04, 6101, 2, 397, 5,
    )
    check_reviewed_wave(runs, monkeypatch, publish, flags, commands, stage05, sha05, record05, chain05, 6101, 2, 397, 6)


def check_2357_reviewed_wave02(runs, monkeypatch, publish, flags, commands, relay, relay_sha, relay_auth, relay_record):
    directory = relay / "seed_2357"
    current = directory / "round_04"
    turn = current / "turn_00"
    fitted = {}
    for name in prompt_pilot_status.RELAY_2357_FITTED_PREFIX:
        path = directory / name
        write_json(path, {"name": name})
        fitted[name] = prompt_pilot_status.checksum(path)
    replay = {}
    for name in prompt_pilot_status.RELAY_2357_WAVE01_REPLAY_FILES:
        path = turn / name
        write_json(path, {"name": name})
        replay[name] = prompt_pilot_status.checksum(path)
    receipts, terminal_shas = [], []
    for attempt in prompt_pilot_status.RELAY_2357_WAVE01_ATTEMPTS:
        terminal = {**attempt, "status": "stopped"}
        path = turn / f"attempt_{attempt['attempt']:02d}.terminal.json"
        write_json(path, terminal)
        receipts.append({**attempt, "terminal_sha256": prompt_pilot_status.record_digest(terminal)})
        terminal_shas.append(prompt_pilot_status.checksum(path))
    exact = {**prompt_pilot_status.RELAY_2357_WAVE02_PROOF, "previous_terminal_file_sha256": terminal_shas[-1]}
    monkeypatch.setattr(prompt_pilot_status, "RELAY_2357_WAVE01_CONTRACT", relay_sha)
    monkeypatch.setattr(prompt_pilot_status, "RELAY_2357_WAVE01_AUTH", relay_auth)
    monkeypatch.setattr(prompt_pilot_status, "RELAY_2357_FITTED_PREFIX", fitted)
    monkeypatch.setattr(prompt_pilot_status, "RELAY_2357_WAVE01_TERMINALS", terminal_shas)
    monkeypatch.setattr(prompt_pilot_status, "RELAY_2357_WAVE01_REPLAY_FILES", replay)
    monkeypatch.setattr(prompt_pilot_status, "RELAY_2357_WAVE02_PROOF", exact)
    monkeypatch.setattr(prompt_pilot_status.time, "time", lambda: exact["relay_not_before_unix"] - 35)
    successor = relay.with_name(f"{prompt_pilot_status.SERIAL_IPV4}-2357-relay-retry-02")
    chain = [{"contract_sha256": relay_sha, "authorization_sha256": relay_auth, "run_directory": str(relay),
              "request_id": exact["previous_request_id"], "terminal_file_sha256": terminal_shas[-1],
              "progress_file_sha256": exact["previous_transport_progress_file_sha256"], "previous_attempts": receipts}]
    proof = {**exact, "dispatcher_module": "dispatch_v13_retry_successor", "seed": 2357, "round": 4,
             "previous_run_directory": str(relay), "previous_contract_sha256": relay_sha,
             "previous_authorization_sha256": relay_auth, "previous_attempts": receipts,
             "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 1},
             "reviewed_fitted_prefix_files": fitted, "reviewed_retry_chain": chain,
             "previous_transport_observation": prompt_pilot_status.RELAY_2357_WAVE02_OBSERVATION}
    record = {**relay_record, "run_directory": str(successor), "predecessor": proof,
              "source_sha256": prompt_pilot_status.RELAY_RETRY_QUEUE_SOURCES,
              "profile": {**relay_record["profile"], "replay_eligible_from_round": 5}}
    publish(successor, record)
    assert remote_seed(runs, 2357)["run"] == relay.name
    publish(successor, {**record, "source_sha256": {}}, **flags)
    assert remote_seed(runs, 2357)["run"] == relay.name
    for bad in ({"previous_attempts": receipts[1:]}, {"reviewed_retry_chain": []}, {"previous_transport_observation": {}}):
        publish(successor, {**record, "predecessor": {**proof, **bad}}, **flags)
        assert remote_seed(runs, 2357)["run"] == relay.name
    successor_sha = publish(successor, record, **flags)
    row = remote_seed(runs, 2357)
    assert (row["run"], row["segment"], row["labels"], row["contract_sha256"]) == (
        successor.name, "ipv4_relay_retry_02", 461, successor_sha,
    )
    assert row["api"] == {"status": "relay_boundary_cooldown", "relay_retry_wait_seconds": 35}
    assert row["reviewed_previous_api"] == {
        "http_status": 200, "error_type": "hard_wall_deadline", "sse_event_count": 5325,
        "event_type": "response.output_text.delta", "last_sse_seconds": 2957.5922856668476, "complete_event_seen": False,
    }
    assert prompt_pilot_status.dispatcher_pid(2357, row["segment"]) == 1234
    assert commands[-1][-1].endswith("/com.purellm.a7-ipv4-v13-2357-relay-retry-02")
    stage03, stage03_sha, record03, chain03 = check_reviewed_wave(
        runs, monkeypatch, publish, flags, commands, successor, successor_sha, record, chain, 2357, 4, 461, 3,
    )
    stage04, sha04, record04, chain04 = check_reviewed_wave(
        runs, monkeypatch, publish, flags, commands, stage03, stage03_sha, record03, chain03, 2357, 4, 461, 4,
    )
    stage05, sha05, record05, chain05 = check_reviewed_wave(
        runs, monkeypatch, publish, flags, commands, stage04, sha04, record04, chain04, 2357, 4, 461, 5,
    )
    stage06, sha06, record06, chain06 = check_reviewed_wave(
        runs, monkeypatch, publish, flags, commands, stage05, sha05, record05, chain05, 2357, 4, 461, 6,
    )
    check_reviewed_wave(runs, monkeypatch, publish, flags, commands, stage06, sha06, record06, chain06, 2357, 4, 461, 7)
    write_json(turn / "attempt_01.replay_link.json", {"detached": True})
    assert "reviewed_previous_api" not in remote_seed(runs, 2357)
    assert remote_seed(runs, 2357)["run"] == relay.name


@pytest.mark.parametrize("seed", (6101, 2357))
def test_reviewed_wave02_status_uses_the_registered_gpu_scope(seed):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_retry_successor

    scope = run_v13_retry_successor.SCOPES[(seed, 2)]
    previous, = scope.archive
    assert (scope.original_start_round, scope.round) == ((1, 2) if seed == 6101 else (4, 4))
    exact = f"RELAY_{seed}_"
    assert (getattr(prompt_pilot_status, exact + "WAVE01_CONTRACT"), getattr(prompt_pilot_status, exact + "WAVE01_AUTH")) == (
        previous.identity.contract, previous.identity.approval,
    )
    assert getattr(prompt_pilot_status, exact + "FITTED_PREFIX") == dict(scope.fitted_prefix_files)
    assert getattr(prompt_pilot_status, exact + "WAVE01_TERMINALS") == tuple(item.terminal_file_sha256 for item in previous.attempts)
    assert getattr(prompt_pilot_status, exact + "WAVE01_ATTEMPTS") == tuple(
        {"attempt": index, "attempt_id": item.attempt_id, "job_sha256": item.job_sha256, "result_sha256": item.result_sha256,
         "error": item.failure, "provider_dispatched": True, "ambiguous": True}
        for index, item in enumerate(previous.attempts)
    )
    assert getattr(prompt_pilot_status, exact + "WAVE02_PROOF") == {
        "logical_index": 0, "payload_sha256": scope.payload_sha256, "previous_request_id": previous.request_id,
        "previous_terminal_file_sha256": previous.terminal_file_sha256,
        "previous_transport_progress_file_sha256": previous.progress_file_sha256,
        "relay_cooldown_clock": scope.cooldown_field, "relay_not_before_unix": scope.not_before_unix,
        "newly_authorized_boundary_physical_attempts": 1, "execution_budget": scope.budget,
    }
    if seed == 2357:
        assert prompt_pilot_status.RELAY_2357_WAVE01_REPLAY_FILES == previous.replay_files
        assert prompt_pilot_status.RELAY_2357_WAVE02_OBSERVATION == previous.progress
        assert prompt_pilot_status.RELAY_RETRY_QUEUE_SOURCES == dict(scope.required_sources)


def test_887_cached_r8_status_uses_the_registered_gpu_scope():
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_retry_successor

    scope = run_v13_retry_successor.SCOPES[(887, 2)]
    previous, = scope.archive
    assert (scope.original_start_round, scope.round) == (7, 8)
    assert (prompt_pilot_status.RELAY_887_CACHED_CONTRACT, prompt_pilot_status.RELAY_887_CACHED_AUTH) == (
        previous.identity.contract, previous.identity.approval,
    )
    assert prompt_pilot_status.RELAY_887_CACHED_REVIEW == scope.cached_origin.review
    assert prompt_pilot_status.RELAY_887_RESOURCE_SHA == scope.cached_origin.resource_sha256
    assert prompt_pilot_status.RELAY_887_FITTED_PREFIX == dict(scope.fitted_prefix_files)
    assert prompt_pilot_status.RELAY_887_R8_SOURCES == dict(scope.required_sources)
    signed_sources = prompt_pilot_status.RELAY_887_R8_WAVE02_SOURCES
    assert set(signed_sources) == set(dict(scope.required_sources)) | {
        source.as_posix() for source in run_v13_retry_successor.SIGNED_RETRY_SOURCES
    }
    for source in run_v13_retry_successor.SIGNED_RETRY_SOURCES:
        # The two versioned launchers are pinned to the archived 887 release.
        if not source.stem.endswith("retry_successor"):
            assert signed_sources[source.as_posix()] == prompt_pilot_status.checksum(run_v13_retry_successor.ROOT / source)
    assert prompt_pilot_status.RELAY_887_R8_OBSERVATION == previous.progress
    assert prompt_pilot_status.RELAY_887_R8_ATTEMPT == {
        "attempt": 0, "attempt_id": previous.attempt_id, "job_sha256": previous.job_sha256,
        "result_sha256": previous.result_sha256, "error": previous.failure, "provider_dispatched": True, "ambiguous": True,
    }
    assert prompt_pilot_status.RELAY_887_R8_PROOF == {
        "logical_index": 0, "payload_sha256": scope.payload_sha256, "previous_request_id": previous.request_id,
        "previous_terminal_file_sha256": previous.terminal_file_sha256,
        "previous_transport_progress_file_sha256": previous.progress_file_sha256,
        "relay_cooldown_clock": scope.cooldown_field, "relay_not_before_unix": scope.not_before_unix,
        "newly_authorized_boundary_physical_attempts": 1, "execution_budget": scope.budget,
    }


def test_887_r12_relay_status_matches_reviewed_ordinary_prefix():
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_retry_successor

    status = prompt_pilot_status
    scope = run_v13_retry_successor.SCOPES[(887, 5)]
    prior, = scope.archive
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (9, 12, 13)
    assert (prior.identity.contract, prior.identity.approval) == (status.CONTINUATION_887_R9_CONTRACT, status.CONTINUATION_887_R9_AUTH)
    assert status.SHA.fullmatch(status.RELAY_887_WAVE05_CONTRACT)
    assert status.SHA.fullmatch(status.RELAY_887_WAVE05_AUTH)
    assert status.RELAY_887_R12_CHECKPOINT == run_v13_retry_successor._887_R12_CHECKPOINT
    assert status.RELAY_887_R12_FITTED_FILES == dict(scope.fitted_prefix_files)
    assert status.RELAY_2357_WAVE05_SOURCES == dict(scope.required_sources)
    assert status.RELAY_887_WAVE05_OBSERVATION == prior.progress
    assert status.RELAY_887_WAVE05_PROOF == {
        "logical_index": 0, "payload_sha256": scope.payload_sha256, "previous_request_id": prior.request_id,
        "previous_terminal_file_sha256": prior.terminal_file_sha256,
        "previous_transport_progress_file_sha256": prior.progress_file_sha256,
        "relay_cooldown_clock": scope.cooldown_field, "relay_not_before_unix": scope.not_before_unix,
        "newly_authorized_boundary_physical_attempts": scope.boundary_physical_attempts, "execution_budget": scope.budget,
    }
    assert status.RELAY_887_R12_ATTEMPT == {
        "attempt": 0, "attempt_id": prior.attempt_id, "job_sha256": prior.job_sha256, "result_sha256": prior.result_sha256,
        "error": prior.failure, "provider_dispatched": True, "ambiguous": True,
    }


@pytest.mark.parametrize(("seed", "wave", "archive_count"),
                         [(12203, 3, 2), (157, 3, 3), (157, 4, 4), (12203, 4, 3), (157, 5, 5), (12203, 5, 4),
                          (6101, 3, 3), (6101, 4, 4), (157, 6, 6), (887, 3, 2), (12203, 6, 5),
                          (887, 4, 3), (2357, 3, 3), (2357, 4, 5), (12203, 8, 4), (6101, 5, 5),
                          (6101, 6, 6), (12203, 9, 6), (2357, 5, 6), (12203, 10, 7), (157, 8, 2),
                          (12203, 11, 8), (2357, 6, 7), (12203, 12, 10), (2357, 7, 8), (12203, 13, 12), (887, 6, 2),
                          (12203, 14, 13), (12203, 15, 14), (887, 7, 3)])
def test_reviewed_wave_status_uses_the_registered_gpu_scope(seed, wave, archive_count):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_retry_successor

    scope = run_v13_retry_successor.SCOPES[(seed, wave)]
    previous = scope.archive[-1]
    assert sum(len(review.attempts) for review in scope.archive) == archive_count
    physical = 2 if (seed, wave) in ((2357, 3), (2357, 4), (2357, 6), (6101, 4), (6101, 6), (887, 6),
                                    (12203, 8), (12203, 9), (12203, 11), (12203, 12), (12203, 14), (157, 8)) else 1
    assert scope.boundary_physical_attempts == physical
    assert scope.replay_start_round == scope.round + (physical == 1)
    assert (getattr(prompt_pilot_status, f"RELAY_{seed}_WAVE{wave - 1:02d}_CONTRACT"),
            getattr(prompt_pilot_status, f"RELAY_{seed}_WAVE{wave - 1:02d}_AUTH")) == (
        previous.identity.contract, previous.identity.approval,
    )
    expected_proof = {
        "payload_sha256": scope.payload_sha256, "previous_request_id": previous.request_id,
        "previous_terminal_file_sha256": previous.terminal_file_sha256,
        "previous_transport_progress_file_sha256": previous.progress_file_sha256,
        "relay_cooldown_clock": scope.cooldown_field, "relay_not_before_unix": scope.not_before_unix,
        "newly_authorized_boundary_physical_attempts": physical, "execution_budget": scope.budget,
    }
    if seed in (6101, 887, 2357) or (seed, wave) == (157, 8) or (seed == 12203 and 8 <= wave <= 15):
        expected_proof["logical_index"] = 0
    assert getattr(prompt_pilot_status, f"RELAY_{seed}_WAVE{wave:02d}_PROOF") == expected_proof
    expected_attempts = tuple({"attempt": slot, "attempt_id": item.attempt_id, "job_sha256": item.job_sha256,
                               "result_sha256": item.result_sha256, "error": item.failure, "provider_dispatched": True,
                               "ambiguous": not item.pre_post_connect_failure} for slot, item in enumerate(previous.attempts))
    if (seed, wave) in ((2357, 4), (12203, 8), (12203, 9), (12203, 12), (12203, 13)):
        prior = f"RELAY_{seed}_WAVE{wave - 1:02d}"
        assert getattr(prompt_pilot_status, prior + "_ATTEMPTS") == expected_attempts
        assert getattr(prompt_pilot_status, prior + "_TERMINALS") == tuple(item.terminal_file_sha256 for item in previous.attempts)
        assert getattr(prompt_pilot_status, prior + "_REPLAY_FILES") == previous.replay_files
    else:
        assert getattr(prompt_pilot_status, f"RELAY_{seed}_WAVE{wave - 1:02d}_ATTEMPT") == expected_attempts[0]
    observation = getattr(prompt_pilot_status, f"RELAY_{seed}_WAVE{wave:02d}_OBSERVATION", None)
    if observation is not None:
        assert observation == previous.progress
        if (seed, wave) in ((2357, 7), (12203, 12), (12203, 13), (12203, 14), (12203, 15), (887, 6), (887, 7)):
            required = prompt_pilot_status.RELAY_12203_WAVE12_SOURCES
        elif (seed, wave) in ((2357, 5), (2357, 6), (12203, 10), (12203, 11), (157, 8)):
            required = prompt_pilot_status.RELAY_2357_WAVE05_SOURCES
        else:
            required = prompt_pilot_status.RELAY_RETRY_QUEUE_SOURCES
        assert required == dict(scope.required_sources)
    if seed == 887 and wave not in (6, 7):
        assert scope.cached_origin.review == prompt_pilot_status.RELAY_887_CACHED_REVIEW
    if (seed, wave) == (887, 6):
        assert prompt_pilot_status.SHA.fullmatch(prompt_pilot_status.RELAY_887_WAVE06_CONTRACT)
        assert prompt_pilot_status.SHA.fullmatch(prompt_pilot_status.RELAY_887_WAVE06_AUTH)
        assert scope.archive[0] == run_v13_retry_successor.SCOPES[(887, 5)].archive[0]
    if (seed, wave) == (887, 7):
        assert prompt_pilot_status.SHA.fullmatch(prompt_pilot_status.RELAY_887_WAVE07_CONTRACT)
        assert prompt_pilot_status.SHA.fullmatch(prompt_pilot_status.RELAY_887_WAVE07_AUTH)
        assert scope.archive[:-1] == run_v13_retry_successor.SCOPES[(887, 6)].archive
    if (seed, wave) == (157, 8):
        assert scope.cached_origin.review == prompt_pilot_status.RELAY_157_WAVE07_CACHED_REVIEW
        assert scope.archive[0] == run_v13_retry_successor.SCOPES[(157, 7)].archive[0]
    if (seed, wave) == (12203, 11):
        assert prompt_pilot_status.SHA.fullmatch(prompt_pilot_status.RELAY_12203_WAVE11_CONTRACT)
        assert prompt_pilot_status.SHA.fullmatch(prompt_pilot_status.RELAY_12203_WAVE11_AUTH)
    if (seed, wave) == (12203, 12):
        assert prompt_pilot_status.SHA.fullmatch(prompt_pilot_status.RELAY_12203_WAVE12_CONTRACT)
        assert prompt_pilot_status.SHA.fullmatch(prompt_pilot_status.RELAY_12203_WAVE12_AUTH)
        assert all(item.progress["sse_event_count"] == 25 and "provider_error_code" not in item.progress
                   for item in previous.attempts)
    if (seed, wave) == (12203, 13):
        assert prompt_pilot_status.SHA.fullmatch(prompt_pilot_status.RELAY_12203_WAVE13_CONTRACT)
        assert prompt_pilot_status.SHA.fullmatch(prompt_pilot_status.RELAY_12203_WAVE13_AUTH)
        first, second = previous.attempts
        assert first.progress == prompt_pilot_status.RELAY_12203_WAVE12_POSTED_OBSERVATION
        assert not first.pre_post_connect_failure and second.pre_post_connect_failure
        extra = set(prompt_pilot_status.RELAY_12203_WAVE13_SOURCES) - set(dict(scope.required_sources))
        assert {path.rsplit("/", 1)[-1] for path in extra} == {"dispatch_v13_retry_successor.py", "run_v13_retry_successor.py"}
        assert all(prompt_pilot_status.SHA.fullmatch(prompt_pilot_status.RELAY_12203_WAVE13_SOURCES[path]) for path in extra)
    if (seed, wave) == (12203, 14):
        assert prompt_pilot_status.SHA.fullmatch(prompt_pilot_status.RELAY_12203_WAVE14_CONTRACT)
        assert prompt_pilot_status.SHA.fullmatch(prompt_pilot_status.RELAY_12203_WAVE14_AUTH)
        assert scope.archive[-2].pre_post_connect_failure
        assert not previous.pre_post_connect_failure and previous.progress["provider_error_code"] == "upstream_error"
        extra = set(prompt_pilot_status.RELAY_12203_WAVE14_SOURCES) - set(dict(scope.required_sources))
        assert {path.rsplit("/", 1)[-1] for path in extra} == {"dispatch_v13_retry_successor.py", "run_v13_retry_successor.py"}
        assert all(prompt_pilot_status.SHA.fullmatch(prompt_pilot_status.RELAY_12203_WAVE14_SOURCES[path]) for path in extra)
    if (seed, wave) == (12203, 15):
        assert prompt_pilot_status.SHA.fullmatch(prompt_pilot_status.RELAY_12203_WAVE15_CONTRACT)
        assert prompt_pilot_status.SHA.fullmatch(prompt_pilot_status.RELAY_12203_WAVE15_AUTH)
        assert scope.archive[:-1] == run_v13_retry_successor.SCOPES[(12203, 14)].archive
        assert scope.original_boundary_physical_attempts == 2 and scope.boundary_physical_attempts == 1
        extra = set(prompt_pilot_status.RELAY_12203_WAVE15_SOURCES) - set(dict(scope.required_sources))
        assert {path.rsplit("/", 1)[-1] for path in extra} == {"dispatch_v13_retry_successor.py", "run_v13_retry_successor.py"}
    if (seed, wave) == (2357, 6):
        assert prompt_pilot_status.SHA.fullmatch(prompt_pilot_status.RELAY_2357_WAVE06_CONTRACT)
        assert prompt_pilot_status.SHA.fullmatch(prompt_pilot_status.RELAY_2357_WAVE06_AUTH)
    if (seed, wave) == (2357, 7):
        assert prompt_pilot_status.SHA.fullmatch(prompt_pilot_status.RELAY_2357_WAVE07_CONTRACT)
        assert prompt_pilot_status.SHA.fullmatch(prompt_pilot_status.RELAY_2357_WAVE07_AUTH)


@pytest.mark.parametrize("seed", prompt_pilot_status.SEEDS)
def test_each_resilient_failure_lane_requires_unchanged_seed_scope(tmp_path, seed):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.prompt_pilot_status import (
        FORMAL_APPROVAL_FLAGS, SERIAL_IPV4, record_digest, resilient_failure_contract,
    )

    previous = tmp_path / f"{SERIAL_IPV4}-{seed}-resilient-sse-01"
    recovery = tmp_path / f"{SERIAL_IPV4}-{seed}-resilient-failure-01"
    profile = {"confirmation_seeds": [seed], "target_labels": 1485, "resume_boundaries": {str(seed): {"round": 3}}}
    previous_record = {"run_directory": str(previous), "profile": profile}
    previous_sha = record_digest(previous_record)
    flags = dict.fromkeys(FORMAL_APPROVAL_FLAGS, True)
    previous_approval = {"contract_sha256": previous_sha, **flags}
    write_json(previous / "formal_contract.json", previous_record)
    write_json(previous / "authorization.json", previous_approval)
    record = {"run_directory": str(recovery), "profile": profile, "predecessor": {
        "dispatcher_module": "dispatch_v13_resilient_failure_recovery",
        "previous_run_directory": str(previous), "previous_contract_sha256": previous_sha,
        "previous_authorization_sha256": record_digest(previous_approval), "seed": seed, "round": 3,
        "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 1},
    }}
    sha = record_digest(record)
    write_json(recovery / "formal_contract.json", record)
    write_json(recovery / "authorization.json", {"contract_sha256": sha, **flags})
    assert resilient_failure_contract(tmp_path, seed, previous, previous_sha, 3) == (recovery, sha)

    record["profile"] = {**profile, "target_labels": 653}
    write_json(recovery / "formal_contract.json", record)
    write_json(recovery / "authorization.json", {"contract_sha256": record_digest(record), **flags})
    assert resilient_failure_contract(tmp_path, seed, previous, previous_sha, 3) is None


def test_cached_r7_requires_signed_predecessor_and_real_activation(tmp_path, monkeypatch):
    seed = 887
    prefix = prompt_pilot_status.SERIAL_IPV4
    previous = tmp_path / f"{prefix}-{seed}-resilient-failure-01"
    run = tmp_path / f"{prefix}-{seed}-cached-r7-01"
    digest = prompt_pilot_status.record_digest
    flags = dict.fromkeys(prompt_pilot_status.FORMAL_APPROVAL_FLAGS, True)
    policy = {"confirmation_seeds": [seed], "target_labels": 1485, "ipv4_readiness_windows": 3,
              "audited_future_transport_replay": True, "resume_boundaries": {str(seed): {"round": 7}},
              "replay_eligible_from_round": 8}
    prior_record = {"run_directory": str(previous), "profile": policy}
    prior_sha = digest(prior_record)
    prior_approval = {"contract_sha256": prior_sha, **flags}
    write_json(previous / "formal_contract.json", prior_record)
    write_json(previous / "authorization.json", prior_approval)
    cached = {"provider_post": False, "normalized_plan_sha256": prompt_pilot_status.CACHED_R7_PLAN_SHA}
    record = {"run_directory": str(run), "profile": policy, "predecessor": {
        "dispatcher_module": "dispatch_v13_887_cached_r7", "seed": seed, "round": 7,
        "previous_run_directory": str(previous), "previous_contract_sha256": prior_sha,
        "previous_authorization_sha256": digest(prior_approval), "cached_import": cached,
        "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 0},
    }}
    approved_sha = digest(record)
    write_json(run / "formal_contract.json", record)
    write_json(run / "authorization.json", {"contract_sha256": approved_sha, **flags})
    write_json(run / "cached_response_import.json", cached)
    assert prompt_pilot_status.cached_r7_contract(tmp_path, seed, previous, prior_sha) is None

    write_json(run / "coordinator_generation.json", {"pid": 1234, "id": "4" * 32})
    assert prompt_pilot_status.cached_r7_contract(tmp_path, seed, previous, prior_sha) == (run, approved_sha, 7, 3)
    assert prompt_pilot_status.cached_r7_contract(tmp_path, seed, previous, "2" * 64) is None
    commands = []
    monkeypatch.setattr(prompt_pilot_status, "command_lines", lambda command: commands.append(command) or [])
    prompt_pilot_status.dispatcher_pid(seed, "ipv4_cached_r7")
    assert commands[0][-1].endswith("/com.purellm.a7-ipv4-v13-887-cached-r7-01")


def test_cached_r3_requires_the_original_response_and_linked_zero_post_fit(tmp_path, monkeypatch):
    status = prompt_pilot_status
    previous = tmp_path / f"{status.SERIAL_IPV4}-157-relay-retry-06"
    run = tmp_path / f"{status.SERIAL_IPV4}-157-cached-r3-01"
    profile = {"confirmation_seeds": [157], "target_labels": 1485, "resume_boundaries": {"157": {"round": 3}},
               "audited_future_transport_replay": True, "sealed_relay_retry_seconds": 180,
               "replay_eligible_from_round": 4, "ipv4_readiness_windows": 3, "sse_inactivity_seconds": 900,
               "request_timeouts": {"total_seconds": 3600}}
    flags = dict.fromkeys(status.FORMAL_APPROVAL_FLAGS, True)
    old = {"run_directory": str(previous), "profile": profile}
    old_contract = status.record_digest(old)
    old_approval = {"contract_sha256": old_contract, **flags}
    old_auth = status.record_digest(old_approval)
    write_json(previous / "formal_contract.json", old)
    write_json(previous / "authorization.json", old_approval)
    old_turn = previous / "seed_157/round_03/turn_00"
    write_json(old_turn / "attempt_00.terminal.json", {"status": "completed"})
    write_json(old_turn / "decision.json", {"state": "rejected"})
    cached = {**status.CACHED_R3_157_REVIEW, "removed_paths": [f"/mock/{index}" for index in range(79)],
              "archived_terminal_file_sha256": status.checksum(old_turn / "attempt_00.terminal.json"),
              "archived_decision_file_sha256": status.checksum(old_turn / "decision.json")}
    attempt = cached["attempt_id"]
    write_json(previous / f"bridge/outbox/{attempt}.json", {"job_sha256": cached["job_sha256"]})
    write_json(previous / f"bridge/results/{attempt}.json", {"result_sha256": cached["result_sha256"],
                                                           "raw_response_sha256": cached["raw_response_sha256"]})
    monkeypatch.setattr(status, "CACHED_R3_157_OLD_CONTRACT", old_contract)
    monkeypatch.setattr(status, "CACHED_R3_157_OLD_AUTH", old_auth)
    monkeypatch.setattr(status, "CACHED_R3_157_REVIEW", {key: cached[key] for key in status.CACHED_R3_157_REVIEW})
    monkeypatch.setattr(status, "CACHED_R3_157_MIRRORS_SHA", status.record_digest(cached["removed_paths"]))
    proof = {"dispatcher_module": "dispatch_v13_157_cached_r3", "previous_run_directory": str(previous),
             "previous_contract_sha256": old_contract, "previous_authorization_sha256": old_auth, "seed": 157, "round": 3,
             "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 0},
             "execution_budget": status.CACHED_R3_157_BUDGET, "cached_import": cached}
    new = {"run_directory": str(run), "profile": profile, "predecessor": proof}
    contract = status.record_digest(new)
    approval = {"contract_sha256": contract, **flags}
    write_json(run / "formal_contract.json", new)
    write_json(run / "authorization.json", approval)
    write_json(run / "cached_response_import.json", cached)
    monkeypatch.setattr(status, "CACHED_R3_157_CONTRACT", contract)
    monkeypatch.setattr(status, "CACHED_R3_157_AUTH", status.record_digest(approval))
    monkeypatch.setattr(status, "CACHED_R3_157_IMPORT_FILE", status.checksum(run / "cached_response_import.json"))
    directory = run / "seed_157"
    local_attempt = "f" * 32
    terminal = {"attempt_id": local_attempt, "status": "completed", "provider_dispatched": False,
                "cached_import_sha256": status.record_digest(cached), "imported_from_response_id": status.CACHED_R3_157_PROVIDER_RESPONSE}
    write_json(directory / "round_03/turn_00/attempt_00.terminal.json", terminal)
    for path in ("round_03/batch_freeze.json", "round_03/selection.json", "round_03/memory.json"):
        write_json(directory / path, {})
    assert status.cached_r3_157_contract(tmp_path, 157, previous, old_contract) is None
    fitted = directory / "round_04"
    write_json(fitted / "source.json", {"checkpoint": status.CACHED_R3_157_CHECKPOINT})
    audit, metrics = fitted / "fit_000/fit_audit.json", fitted / "fit_000/validation_metrics.json"
    write_json(audit, {})
    write_json(metrics, {"combined_normalized_rmse": .5})
    write_json(fitted / "transition.json", {"checkpoint_sha256": status.CACHED_R3_157_CHECKPOINT, "fit_path": "fit_000",
                                             "source_sha256": status.checksum(fitted / "source.json"),
                                             "fit_audit_sha256": status.checksum(audit),
                                             "validation_metrics_sha256": status.checksum(metrics)})
    monkeypatch.setattr(status, "CACHED_R3_157_FITTED_FILES", {name: status.checksum(directory / name)
                                                              for name in status.CACHED_R3_157_FITTED_FILES})
    monkeypatch.setattr(status, "CACHED_R3_157_FIT_AUDIT", status.checksum(audit))
    monkeypatch.setattr(status, "CACHED_R3_157_METRICS", status.checksum(metrics))
    assert status.cached_r3_157_contract(tmp_path, 157, previous, old_contract) == (run, contract, 3, 3)
    next_run = tmp_path / f"{status.SERIAL_IPV4}-157-relay-retry-07"
    attempt = status.RELAY_157_WAVE07_PRIOR_ATTEMPT
    archived = {**attempt, "status": "stopped"}
    terminal_file = fitted / "turn_00/attempt_00.terminal.json"
    write_json(terminal_file, archived)
    exact = {**status.RELAY_157_WAVE07_PROOF, "previous_terminal_file_sha256": status.checksum(terminal_file)}
    review = {**status.RELAY_157_WAVE07_CACHED_REVIEW, "proof_sha256": status.record_digest(cached),
              "receipt_file_sha256": status.CACHED_R3_157_IMPORT_FILE}
    monkeypatch.setattr(status, "RELAY_157_WAVE07_PROOF", exact)
    monkeypatch.setattr(status, "RELAY_157_WAVE07_CACHED_REVIEW", review)
    receipts = [{**attempt, "terminal_sha256": status.record_digest(archived)}]
    chain = [{"contract_sha256": contract, "authorization_sha256": status.CACHED_R3_157_AUTH, "run_directory": str(run),
              "request_id": exact["previous_request_id"], "terminal_file_sha256": exact["previous_terminal_file_sha256"],
              "progress_file_sha256": exact["previous_transport_progress_file_sha256"], "previous_attempts": receipts}]
    next_record = {"run_directory": str(next_run), "budget": exact["execution_budget"], "source_sha256": status.RELAY_RETRY_QUEUE_SOURCES,
                   "profile": {**profile, "resume_boundaries": {"157": {"round": 4}}, "replay_eligible_from_round": 5},
                   "predecessor": {**exact, "dispatcher_module": "dispatch_v13_retry_successor", "previous_run_directory": str(run),
                                   "seed": 157, "round": 4, "previous_contract_sha256": contract,
                                   "previous_authorization_sha256": status.CACHED_R3_157_AUTH,
                                   "reviewed_fitted_prefix_files": status.CACHED_R3_157_FITTED_FILES, "reviewed_cached_import": review,
                                   "previous_transport_observation": status.RELAY_157_WAVE07_OBSERVATION,
                                   "public_resource_receipt_sha256": status.RELAY_157_WAVE07_RESOURCE_SHA,
                                   "reviewed_retry_chain": chain, "previous_attempts": receipts,
                                   "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}}}
    next_sha = status.record_digest(next_record)
    next_auth = {"contract_sha256": next_sha, **flags}
    write_json(next_run / "formal_contract.json", next_record)
    write_json(next_run / "authorization.json", next_auth)
    monkeypatch.setattr(status, "RELAY_157_WAVE07_CONTRACT", next_sha)
    assert status.relay_retry_07_157_contract(tmp_path, run, contract) is None
    monkeypatch.setattr(status, "RELAY_157_WAVE07_AUTH", status.record_digest(next_auth))
    assert status.relay_retry_07_157_contract(tmp_path, run, contract) == (next_run, next_sha, 4, 3, exact["relay_not_before_unix"])
    second_slot = terminal_file.with_name("attempt_01.started.json")
    write_json(second_slot, {"attempt": 1})
    assert status.relay_retry_07_157_contract(tmp_path, run, contract) is None
    second_slot.unlink()
    check_157_cache_wave08(tmp_path, monkeypatch, flags, previous, old_contract, next_run, next_sha, next_record)
    later = directory / "round_05"
    write_json(later / "source.json", {"checkpoint": "c" * 64})
    write_json(later / "fit_000/validation_metrics.json", {"combined_normalized_rmse": .4})
    write_json(later / "transition.json", {"checkpoint_sha256": "c" * 64, "fit_path": "fit_000",
                                            "source_sha256": status.checksum(later / "source.json"),
                                            "validation_metrics_sha256": status.checksum(later / "fit_000/validation_metrics.json")})
    assert status.latest_round(directory)[0] == 5
    assert status.cached_r3_157_contract(tmp_path, 157, previous, old_contract) == (run, contract, 3, 3)
    write_json(run / f"bridge/outbox/{local_attempt}.json", {"unexpected": "provider transport"})
    assert status.cached_r3_157_contract(tmp_path, 157, previous, old_contract) is None


def check_157_cache_wave08(runs, monkeypatch, flags, cache_parent, cache_parent_sha, previous, previous_sha, previous_record):
    status = prompt_pilot_status
    seed = 157
    cohort = runs / status.SERIAL_IPV4
    cohort_record = {"run_directory": str(cohort), "profile": {"confirmation_seeds": list(status.SEEDS),
                       "provider_capability": "gated_ipv4"}, "predecessor": {"dispatcher_module": "dispatch_prepared_ipv4"}}
    write_json(cohort / "formal_contract.json", cohort_record)
    write_json(cohort / "authorization.json", {"contract_sha256": status.record_digest(cohort_record), **flags})
    turn = previous / "seed_157/round_04/turn_00"
    attempt = status.RELAY_157_WAVE07_ATTEMPT
    terminal = {**attempt, "status": "stopped"}
    terminal_file = turn / "attempt_00.terminal.json"
    write_json(turn / "attempt_00.started.json", {"attempt": 0, "attempt_id": attempt["attempt_id"]})
    write_json(terminal_file, terminal)
    exact = {**status.RELAY_157_WAVE08_PROOF, "previous_terminal_file_sha256": status.checksum(terminal_file)}
    monkeypatch.setattr(status, "RELAY_157_WAVE08_PROOF", exact)
    receipt = {**attempt, "terminal_sha256": status.record_digest(terminal)}
    old_proof = previous_record["predecessor"]
    chain = [*old_proof["reviewed_retry_chain"], status._relay_review(previous, previous_sha,
                                                                  status.RELAY_157_WAVE07_AUTH, exact, [receipt])]
    run = runs / f"{status.SERIAL_IPV4}-157-relay-retry-08"
    proof = {**old_proof, **exact, "previous_run_directory": str(previous), "previous_contract_sha256": previous_sha,
             "previous_authorization_sha256": status.RELAY_157_WAVE07_AUTH,
             "previous_transport_observation": status.RELAY_157_WAVE08_OBSERVATION,
             "reviewed_retry_chain": chain, "previous_attempts": [receipt],
             "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}}
    record = {"run_directory": str(run), "budget": exact["execution_budget"], "source_sha256": status.RELAY_2357_WAVE05_SOURCES,
              "profile": {**previous_record["profile"], "replay_eligible_from_round": 4}, "predecessor": proof}
    sha = status.record_digest(record)
    authorization = {"contract_sha256": sha, **flags}
    write_json(run / "formal_contract.json", record)
    write_json(run / "authorization.json", authorization)
    monkeypatch.setattr(status, "RELAY_157_WAVE08_CONTRACT", sha)
    assert status.reviewed_relay_contract(runs, seed, 8, previous, previous_sha) is None
    monkeypatch.setattr(status, "RELAY_157_WAVE08_AUTH", status.record_digest(authorization))
    assert status.reviewed_relay_contract(runs, seed, 8, previous, previous_sha) == (run, sha, 4, 3, exact["relay_not_before_unix"])
    monkeypatch.setattr(status, "resilient_contract", lambda *_: (runs / "resilient", "resilient", 3, 3))
    monkeypatch.setattr(status, "resilient_failure_contract", lambda *_: (runs / "failed", "failed"))
    monkeypatch.setattr(status, "relay_retry_contract", lambda *_: (runs / "retry", "retry", 3, 3))
    monkeypatch.setattr(status, "relay_retry_02_contract", lambda *_: (runs / "wave02", "wave02", 3, 3, 10))
    contract = status.reviewed_relay_contract
    monkeypatch.setattr(status, "reviewed_relay_contract", lambda _runs, _seed, wave, *args: (
        contract(_runs, _seed, wave, *args) if wave == 8 else
        (cache_parent if wave == 6 else runs / f"wave{wave}", cache_parent_sha if wave == 6 else f"wave{wave}", 3, 3, 10)
        if wave <= 6 else None))
    monkeypatch.setattr(status.time, "time", lambda: exact["relay_not_before_unix"] - 30)
    row = remote_seed(runs, seed)
    assert (row["run"], row["segment"], row["labels"]) == (run.name, "ipv4_relay_retry_08", 461)
    assert row["api"] == {"status": "relay_boundary_cooldown", "relay_retry_wait_seconds": 30}
    assert row["cached_import"]["provider_dispatched"] is False
    assert row["reviewed_previous_api"] == {
        "http_status": 200, "error_type": "hard_wall_deadline", "sse_event_count": 2783,
        "event_type": "response.output_text.delta", "last_sse_seconds": 3561.158813457936,
        "complete_event_seen": False, "hard_deadline_seconds": 3600,
    }
    assert row["boundary_api"]["maximum_physical_attempts"] == 2
    assert row["boundary_api"]["replay_requires_verified_response_id"] is True
    assert row["boundary_api"]["future_replay_from_round"] == 5
    commands = []
    monkeypatch.setattr(status, "command_lines", lambda command: commands.append(command) or [])
    status.dispatcher_pid(seed, row["segment"])
    assert commands[0][-1].endswith("/com.purellm.a7-ipv4-v13-157-relay-retry-08")
    write_json(turn / "attempt_01.started.json", {"unexpected": True})
    assert remote_seed(runs, seed)["run"] == previous.name


def test_6101_cache_status_requires_the_exact_zero_post_l429_fit(tmp_path, monkeypatch):
    status = prompt_pilot_status
    seed = 6101
    cohort = tmp_path / status.SERIAL_IPV4
    old = tmp_path / f"{status.SERIAL_IPV4}-6101-relay-retry-06"
    run = tmp_path / f"{status.SERIAL_IPV4}-6101-cached-r2-01"
    flags = dict.fromkeys(status.FORMAL_APPROVAL_FLAGS, True)

    def publish(path, record):
        sha = status.record_digest(record)
        authorization = {"contract_sha256": sha, **flags}
        write_json(path / "formal_contract.json", record)
        write_json(path / "authorization.json", authorization)
        return sha, status.record_digest(authorization)

    publish(cohort, {"run_directory": str(cohort), "profile": {"confirmation_seeds": list(status.SEEDS),
                                                              "provider_capability": "gated_ipv4"},
                     "predecessor": {"dispatcher_module": "dispatch_prepared_ipv4"}})
    profile = {"confirmation_seeds": [seed], "target_labels": 1485, "resume_boundaries": {str(seed): {"round": 2}},
               "audited_future_transport_replay": True, "sealed_relay_retry_seconds": 180,
               "replay_eligible_from_round": 3, "ipv4_readiness_windows": 3, "sse_inactivity_seconds": 900,
               "request_timeouts": {"total_seconds": 3600},
               "transport": {"logical_replies": 3, "transport_attempts": 2}}
    old_sha, old_auth = publish(old, {"run_directory": str(old), "profile": profile})
    monkeypatch.setattr(status, "RELAY_6101_WAVE06_CONTRACT", old_sha)
    monkeypatch.setattr(status, "RELAY_6101_WAVE06_AUTH", old_auth)
    old_turn = old / "seed_6101/round_02/turn_00"
    write_json(old_turn / "attempt_00.terminal.json", {"status": "completed"})
    write_json(old_turn / "decision.json", {"state": "rejected"})
    review = {**status.CACHED_R2_6101_REVIEW,
              "archived_terminal_file_sha256": status.checksum(old_turn / "attempt_00.terminal.json"),
              "archived_decision_file_sha256": status.checksum(old_turn / "decision.json")}
    monkeypatch.setattr(status, "CACHED_R2_6101_REVIEW", review)
    attempt = review["attempt_id"]
    write_json(old / f"bridge/outbox/{attempt}.json", {"job_sha256": review["job_sha256"]})
    write_json(old / f"bridge/results/{attempt}.json", {"result_sha256": review["result_sha256"],
                                                       "raw_response_sha256": review["raw_response_sha256"],
                                                       "provider_dispatched": True})
    mirrors = [f"/choices/{index}/references/0/{role}_id" for role, count in (("observed", 59), ("candidate", 15))
               for index in range(count)]
    monkeypatch.setattr(status, "CACHED_R2_6101_MIRRORS_SHA", status.record_digest(mirrors))
    cached = {**review, "removed_paths": mirrors}
    proof = {"previous_run_directory": str(old), "previous_contract_sha256": old_sha,
             "previous_authorization_sha256": old_auth, "dispatcher_module": "dispatch_v13_6101_cached_r2",
             "seed": seed, "round": 2, "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 0},
             "execution_budget": status.CACHED_R2_6101_BUDGET, "cached_import": cached}
    sha, auth = publish(run, {"run_directory": str(run), "profile": profile, "predecessor": proof,
                              "budget": status.CACHED_R2_6101_BUDGET})
    monkeypatch.setattr(status, "CACHED_R2_6101_CONTRACT", sha)
    monkeypatch.setattr(status, "CACHED_R2_6101_AUTH", auth)
    write_json(run / "cached_response_import.json", cached)
    local_attempt = "a" * 32
    directory = run / "seed_6101"
    terminal = directory / "round_02/turn_00/attempt_00.terminal.json"
    write_json(terminal, {"attempt_id": local_attempt, "status": "completed", "provider_dispatched": False,
                          "usage": {}, "cached_import_sha256": status.record_digest(cached),
                          "imported_from_response_id": status.CACHED_R2_6101_RESPONSE})
    assert status.cached_r2_6101_contract(tmp_path, seed, old, old_sha) is None
    fitted = directory / "round_03"
    metric = fitted / "fit_000/validation_metrics.json"
    audit = fitted / "fit_000/fit_audit.json"
    source = fitted / "source.json"
    transition = fitted / "transition.json"
    write_json(metric, {"combined_normalized_rmse": .80893578})
    write_json(audit, {"device": "cuda"})
    write_json(source, {"checkpoint": status.CACHED_R2_6101_CHECKPOINT})
    write_json(transition, {"checkpoint_sha256": status.CACHED_R2_6101_CHECKPOINT, "fit_path": "fit_000",
                            "source_sha256": status.checksum(source), "fit_audit_sha256": status.checksum(audit),
                            "validation_metrics_sha256": status.checksum(metric)})
    monkeypatch.setattr(status, "CACHED_R2_6101_TRANSITION", status.checksum(transition))
    result = status.cached_r2_6101_contract(tmp_path, seed, old, old_sha)
    assert result[:4] == (run, sha, 2, 3)
    assert result[4] == {"round": 2, "fitted_labels": 429, "choice_count": 32, "removed_reference_fields": 74,
                         "removed_observed_reference_fields": 59, "removed_candidate_reference_fields": 15,
                         "provider_dispatched": False, "checkpoint_sha256": status.CACHED_R2_6101_CHECKPOINT,
                         "validation_nrmse": .80893578, "future_starts_at_round": 3,
                         "future_logical_replies": 3, "future_transport_attempts": 2}
    monkeypatch.setattr(status, "resilient_contract", lambda *_: (tmp_path / "resilient", "resilient", 2, 3))
    monkeypatch.setattr(status, "resilient_failure_contract", lambda *_: (tmp_path / "failed", "failed"))
    monkeypatch.setattr(status, "relay_retry_contract", lambda *_: (tmp_path / "retry", "retry", 2, 3))
    monkeypatch.setattr(status, "relay_retry_02_contract", lambda *_: (tmp_path / "wave02", "wave02", 2, 3, 10))
    reviewed = status.reviewed_relay_contract
    monkeypatch.setattr(status, "reviewed_relay_contract", lambda _runs, _seed, wave, *args: (
        reviewed(_runs, _seed, wave, *args) if wave == 8 else
        (old if wave == 6 else tmp_path / f"wave{wave}", old_sha if wave == 6 else f"wave{wave}", 2, 3, 10)
        if wave <= 6 else None))
    row = remote_seed(tmp_path, seed)
    assert (row["segment"], row["labels"], row["phase"]) == ("ipv4_cached_r2_6101", 429, "cached_import_complete")
    assert row["cached_import"] == result[4]
    assert row["validation_nrmse"] == .80893578
    commands = []
    monkeypatch.setattr(status, "command_lines", lambda command: commands.append(command) or [])
    status.dispatcher_pid(seed, row["segment"])
    assert commands[0][-1].endswith("/com.purellm.a7-ipv4-v13-6101-cached-r2-01")
    check_6101_cache_wave07(tmp_path, monkeypatch, publish, run, sha, profile, cached, result[4], commands)
    write_json(run / f"bridge/outbox/{local_attempt}.json", {"unexpected": "provider transport"})
    assert status.cached_r2_6101_contract(tmp_path, seed, old, old_sha) is None
    assert remote_seed(tmp_path, seed)["run"] == old.name


def check_6101_cache_wave07(runs, monkeypatch, publish, previous, previous_sha, previous_profile, cached, imported, commands):
    status = prompt_pilot_status
    directory = previous / "seed_6101"
    for name in status.RELAY_6101_WAVE07_FITTED_FILES:
        if not (directory / name).exists():
            write_json(directory / name, {"fixture": name})
    fitted = {name: status.checksum(directory / name) for name in status.RELAY_6101_WAVE07_FITTED_FILES}
    monkeypatch.setattr(status, "RELAY_6101_WAVE07_FITTED_FILES", fitted)
    cache_review = {**status.RELAY_6101_WAVE07_CACHED_REVIEW, "proof_sha256": status.record_digest(cached),
                    "receipt_file_sha256": status.checksum(previous / "cached_response_import.json")}
    monkeypatch.setattr(status, "RELAY_6101_WAVE07_CACHED_REVIEW", cache_review)
    turn = directory / "round_03/turn_00"
    attempt = status.RELAY_6101_WAVE07_PRIOR_ATTEMPT
    terminal = {**attempt, "status": "stopped"}
    terminal_file = turn / "attempt_00.terminal.json"
    write_json(turn / "attempt_00.started.json", {"attempt": 0, "attempt_id": attempt["attempt_id"]})
    write_json(terminal_file, terminal)
    exact = {**status.RELAY_6101_WAVE07_PROOF, "previous_terminal_file_sha256": status.checksum(terminal_file)}
    monkeypatch.setattr(status, "RELAY_6101_WAVE07_PROOF", exact)
    receipt = {**attempt, "terminal_sha256": status.record_digest(terminal)}
    run = runs / f"{status.SERIAL_IPV4}-6101-relay-retry-07"
    proof = {**exact, "dispatcher_module": "dispatch_v13_retry_successor", "previous_run_directory": str(previous),
             "previous_contract_sha256": previous_sha, "previous_authorization_sha256": status.CACHED_R2_6101_AUTH,
             "seed": 6101, "round": 3, "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 1},
             "reviewed_fitted_prefix_files": fitted, "checkpoint_sha256": status.CACHED_R2_6101_CHECKPOINT,
             "reviewed_cached_import": cache_review, "public_resource_receipt_sha256": status.RELAY_6101_WAVE07_RESOURCE_SHA,
             "previous_transport_observation": status.RELAY_6101_WAVE07_OBSERVATION, "previous_attempts": [receipt],
             "reviewed_retry_chain": [status._relay_review(previous, previous_sha, status.CACHED_R2_6101_AUTH, exact, [receipt])]}
    record = {"run_directory": str(run), "source_sha256": status.RELAY_12203_WAVE12_SOURCES,
              "profile": {**previous_profile, "resume_boundaries": {"6101": {"round": 3}}, "replay_eligible_from_round": 4},
              "predecessor": proof, "budget": exact["execution_budget"]}
    sha, auth = publish(run, record)
    monkeypatch.setattr(status, "RELAY_6101_WAVE07_CONTRACT", sha)
    assert remote_seed(runs, 6101)["run"] == previous.name
    monkeypatch.setattr(status, "RELAY_6101_WAVE07_AUTH", auth)
    monkeypatch.setattr(status.time, "time", lambda: exact["relay_not_before_unix"] - 25)
    row = remote_seed(runs, 6101)
    assert (row["run"], row["segment"], row["labels"], row["contract_sha256"]) == (run.name, "ipv4_relay_retry_07", 429, sha)
    assert row["api"] == {"status": "relay_boundary_cooldown", "relay_retry_wait_seconds": 25}
    assert row["cached_import"] == {**imported, "future_starts_at_round": 4}
    assert row["reviewed_previous_api"] == {
        "http_status": 200, "error_type": "sse_inactivity_timeout", "sse_event_count": 1092,
        "event_type": "response.output_text.delta", "last_sse_seconds": 1930.1032520001754,
        "complete_event_seen": False, "sse_inactivity_seconds": 900,
    }
    assert row["boundary_api"]["maximum_physical_attempts"] == 1
    assert row["boundary_api"]["future_replay_from_round"] == 4
    assert row["boundary_api"]["future_replay_requires_verified_response_id"] is True
    status.dispatcher_pid(6101, row["segment"])
    assert commands[-1][-1].endswith("/com.purellm.a7-ipv4-v13-6101-relay-retry-07")
    check_6101_cache_wave08(runs, monkeypatch, publish, run, sha, auth, record, imported, commands)
    write_json(turn / "attempt_01.started.json", {"unexpected": True})
    assert remote_seed(runs, 6101)["run"] == previous.name
    (turn / "attempt_01.started.json").unlink()
    write_json(directory / "round_02/selection.json", {"modified": True})
    assert remote_seed(runs, 6101)["run"] == previous.name


def check_6101_cache_wave08(runs, monkeypatch, publish, previous, previous_sha, previous_auth, previous_record,
                            imported, commands):
    status = prompt_pilot_status
    seed = 6101
    turn = previous / "seed_6101/round_03/turn_00"
    attempt = status.RELAY_6101_WAVE07_ATTEMPT
    terminal = {**attempt, "status": "stopped"}
    terminal_file = turn / "attempt_00.terminal.json"
    write_json(turn / "attempt_00.started.json", {"attempt": 0, "attempt_id": attempt["attempt_id"]})
    write_json(terminal_file, terminal)
    exact = {**status.RELAY_6101_WAVE08_PROOF, "previous_terminal_file_sha256": status.checksum(terminal_file)}
    monkeypatch.setattr(status, "RELAY_6101_WAVE08_PROOF", exact)
    receipt = {**attempt, "terminal_sha256": status.record_digest(terminal)}
    prior = previous_record["predecessor"]
    chain = [*prior["reviewed_retry_chain"], status._relay_review(previous, previous_sha, previous_auth, exact, [receipt])]
    run = runs / f"{status.SERIAL_IPV4}-6101-relay-retry-08"
    proof = {**prior, **exact, "previous_run_directory": str(previous), "previous_contract_sha256": previous_sha,
             "previous_authorization_sha256": previous_auth, "previous_transport_observation": status.RELAY_6101_WAVE08_OBSERVATION,
             "previous_attempts": [receipt], "reviewed_retry_chain": chain,
             "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}}
    record = {"run_directory": str(run), "source_sha256": status.RELAY_12203_WAVE12_SOURCES,
              "profile": {**previous_record["profile"], "replay_eligible_from_round": 3}, "predecessor": proof,
              "budget": exact["execution_budget"]}
    sha, auth = publish(run, record)
    monkeypatch.setattr(status, "RELAY_6101_WAVE08_CONTRACT", sha)
    assert remote_seed(runs, seed)["run"] == previous.name
    monkeypatch.setattr(status, "RELAY_6101_WAVE08_AUTH", auth)
    with monkeypatch.context() as wave_clock:
        wave_clock.setattr(status.time, "time", lambda: exact["relay_not_before_unix"] - 25)
        row = remote_seed(runs, seed)
        assert (row["run"], row["segment"], row["labels"], row["contract_sha256"]) == (run.name, "ipv4_relay_retry_08", 429, sha)
        assert row["api"] == {"status": "relay_boundary_cooldown", "relay_retry_wait_seconds": 25}
    assert row["cached_import"] == {**imported, "future_starts_at_round": 4}
    assert row["reviewed_previous_api"] == {
        "http_status": 200, "error_type": "hard_wall_deadline", "sse_event_count": 648,
        "event_type": "response.output_text.delta", "last_sse_seconds": 3314.1049271670636,
        "complete_event_seen": False, "hard_deadline_seconds": 3600,
    }
    assert row["boundary_api"]["maximum_physical_attempts"] == 2
    assert row["boundary_api"]["replay_requires_verified_response_id"] is True
    assert row["boundary_api"]["future_replay_from_round"] == 4
    status.dispatcher_pid(seed, row["segment"])
    assert commands[-1][-1].endswith("/com.purellm.a7-ipv4-v13-6101-relay-retry-08")
    broken = {**record, "predecessor": {**proof, "reviewed_retry_chain": chain[1:]}}
    broken_sha, broken_auth = publish(run, broken)
    with monkeypatch.context() as signed_fixture:
        signed_fixture.setattr(status, "RELAY_6101_WAVE08_CONTRACT", broken_sha)
        signed_fixture.setattr(status, "RELAY_6101_WAVE08_AUTH", broken_auth)
        assert remote_seed(runs, seed)["run"] == previous.name
    publish(run, record)
    unexpected_slot = turn / "attempt_01.started.json"
    write_json(unexpected_slot, {"unexpected": True})
    assert remote_seed(runs, seed)["run"] == previous.name
    unexpected_slot.unlink()


def test_6101_wave08_matches_both_reviewed_transport_attempts():
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.run_v13_retry_successor import SCOPES

    status = prompt_pilot_status
    scope = SCOPES[(6101, 8)]
    ancestor, prior = scope.archive
    assert ancestor == SCOPES[(6101, 7)].archive[0]
    assert tuple(len(review.attempts) for review in scope.archive) == (1, 1)
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (None, 3, 3)
    assert (prior.identity.contract, prior.identity.approval) == (status.RELAY_6101_WAVE07_CONTRACT, status.RELAY_6101_WAVE07_AUTH)
    assert status.RELAY_6101_WAVE07_CACHED_REVIEW == scope.cached_origin.review
    assert status.RELAY_12203_WAVE12_SOURCES == dict(scope.required_sources)
    assert status.RELAY_6101_WAVE08_OBSERVATION == prior.progress
    assert status.RELAY_6101_WAVE08_PROOF == {
        "logical_index": 0, "payload_sha256": scope.payload_sha256, "previous_request_id": prior.request_id,
        "previous_terminal_file_sha256": prior.terminal_file_sha256,
        "previous_transport_progress_file_sha256": prior.progress_file_sha256,
        "relay_cooldown_clock": scope.cooldown_field, "relay_not_before_unix": scope.not_before_unix,
        "newly_authorized_boundary_physical_attempts": scope.boundary_physical_attempts, "execution_budget": scope.budget,
    }
    assert status.RELAY_6101_WAVE07_ATTEMPT == {
        "attempt": 0, "attempt_id": prior.attempt_id, "job_sha256": prior.job_sha256, "result_sha256": prior.result_sha256,
        "error": prior.failure, "provider_dispatched": True, "ambiguous": True,
    }


def test_6101_wave07_matches_the_registered_exact_cached_fit():
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.run_v13_retry_successor import SCOPES

    status = prompt_pilot_status
    scope = SCOPES[(6101, 7)]
    prior, = scope.archive
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (2, 3, 4)
    assert (prior.identity.contract, prior.identity.approval) == (status.CACHED_R2_6101_CONTRACT, status.CACHED_R2_6101_AUTH)
    assert status.SHA.fullmatch(status.RELAY_6101_WAVE07_CONTRACT)
    assert status.SHA.fullmatch(status.RELAY_6101_WAVE07_AUTH)
    assert status.RELAY_6101_WAVE07_CACHED_REVIEW == scope.cached_origin.review
    assert status.RELAY_6101_WAVE07_RESOURCE_SHA == scope.cached_origin.resource_sha256
    assert status.RELAY_6101_WAVE07_FITTED_FILES == dict(scope.fitted_prefix_files)
    assert status.RELAY_12203_WAVE12_SOURCES == dict(scope.required_sources)
    assert status.RELAY_6101_WAVE07_OBSERVATION == prior.progress
    assert status.RELAY_6101_WAVE07_PROOF == {
        "logical_index": 0, "payload_sha256": scope.payload_sha256, "previous_request_id": prior.request_id,
        "previous_terminal_file_sha256": prior.terminal_file_sha256,
        "previous_transport_progress_file_sha256": prior.progress_file_sha256,
        "relay_cooldown_clock": scope.cooldown_field, "relay_not_before_unix": scope.not_before_unix,
        "newly_authorized_boundary_physical_attempts": scope.boundary_physical_attempts, "execution_budget": scope.budget,
    }
    assert status.RELAY_6101_WAVE07_PRIOR_ATTEMPT == {
        "attempt": 0, "attempt_id": prior.attempt_id, "job_sha256": prior.job_sha256, "result_sha256": prior.result_sha256,
        "error": prior.failure, "provider_dispatched": True, "ambiguous": True,
    }


def test_157_wave07_requires_the_exact_post_cache_producer_scope():
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.run_v13_retry_successor import SCOPES

    status = prompt_pilot_status
    scope = SCOPES[(157, 7)]
    review, = scope.archive
    assert len(review.attempts) == 1
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (3, 4, 5)
    assert (review.identity.contract, review.identity.approval) == (status.CACHED_R3_157_CONTRACT, status.CACHED_R3_157_AUTH)
    assert status.RELAY_157_WAVE07_CACHED_REVIEW == scope.cached_origin.review
    assert status.RELAY_157_WAVE07_RESOURCE_SHA == scope.cached_origin.resource_sha256
    assert status.RELAY_157_WAVE07_OBSERVATION == review.progress
    assert status.CACHED_R3_157_FITTED_FILES == dict(scope.fitted_prefix_files)
    assert status.RELAY_RETRY_QUEUE_SOURCES == dict(scope.required_sources)
    assert status.RELAY_157_WAVE07_PROOF == {
        "logical_index": 0, "payload_sha256": scope.payload_sha256, "previous_request_id": review.request_id,
        "previous_terminal_file_sha256": review.terminal_file_sha256, "previous_transport_progress_file_sha256": review.progress_file_sha256,
        "relay_cooldown_clock": scope.cooldown_field, "relay_not_before_unix": scope.not_before_unix,
        "newly_authorized_boundary_physical_attempts": scope.boundary_physical_attempts, "execution_budget": scope.budget,
    }
    assert status.RELAY_157_WAVE07_PRIOR_ATTEMPT == {
        "attempt": 0, "attempt_id": review.attempt_id, "job_sha256": review.job_sha256, "result_sha256": review.result_sha256,
        "error": review.failure, "provider_dispatched": True, "ambiguous": True,
    }


def test_12203_wave07_requires_the_fitted_predecessor_and_two_closed_attempts(tmp_path, monkeypatch, capsys):
    status = prompt_pilot_status
    previous = tmp_path / f"{status.SERIAL_IPV4}-12203-relay-retry-06"
    run = tmp_path / f"{status.SERIAL_IPV4}-12203-relay-retry-07"
    directory = previous / "seed_12203"
    flags = dict.fromkeys(status.FORMAL_APPROVAL_FLAGS, True)
    profile = {"confirmation_seeds": [12203], "target_labels": 1485, "resume_boundaries": {"12203": {"round": 4}},
               "audited_future_transport_replay": True, "sealed_relay_retry_seconds": 180,
               "replay_eligible_from_round": 5, "ipv4_readiness_windows": 3, "sse_inactivity_seconds": 900,
               "request_timeouts": {"total_seconds": 3600}}
    old = {"run_directory": str(previous), "profile": profile,
           "budget": {"first_logical_requests": 32, "maximum_logical_requests": 94, "maximum_physical_attempts": 187},
           "predecessor": {"recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}}}
    old_contract = status.record_digest(old)
    old_approval = {"contract_sha256": old_contract, **flags}
    old_auth = status.record_digest(old_approval)
    write_json(previous / "formal_contract.json", old)
    write_json(previous / "authorization.json", old_approval)
    monkeypatch.setattr(status, "RELAY_12203_WAVE06_CONTRACT", old_contract)
    monkeypatch.setattr(status, "RELAY_12203_WAVE06_AUTH", old_auth)
    fitted = directory / "round_05"
    write_json(fitted / "source.json", {"checkpoint": "d" * 64})
    write_json(fitted / "fit_000/validation_metrics.json", {"combined_normalized_rmse": .4})
    write_json(fitted / "transition.json", {"checkpoint_sha256": "d" * 64, "fit_path": "fit_000",
                                             "source_sha256": status.checksum(fitted / "source.json"),
                                             "validation_metrics_sha256": status.checksum(fitted / "fit_000/validation_metrics.json")})
    write_json(directory / "round_04/memory.json", {})
    prefix = {name: status.checksum(directory / name) for name in status.RELAY_12203_WAVE06_FITTED_PREFIX}
    monkeypatch.setattr(status, "RELAY_12203_WAVE06_FITTED_PREFIX", prefix)
    old_turn = fitted / "turn_00"
    replay = {}
    for name in status.RELAY_12203_WAVE06_REPLAY_FILES:
        write_json(old_turn / name, {"reviewed": name})
        replay[name] = status.checksum(old_turn / name)
    monkeypatch.setattr(status, "RELAY_12203_WAVE06_REPLAY_FILES", replay)
    receipts, terminal_shas = [], []
    for item in status.RELAY_12203_WAVE06_ATTEMPTS:
        path = old_turn / f"attempt_{item['attempt']:02d}.terminal.json"
        write_json(path, {**item, "status": "stopped"})
        receipts.append({**item, "terminal_sha256": status.record_digest(status.read_json(path))})
        terminal_shas.append(status.checksum(path))
    monkeypatch.setattr(status, "RELAY_12203_WAVE06_TERMINALS", tuple(terminal_shas))
    exact = {**status.RELAY_12203_WAVE07_PROOF, "previous_terminal_file_sha256": terminal_shas[-1]}
    monkeypatch.setattr(status, "RELAY_12203_WAVE07_PROOF", exact)
    proof = {**exact, "dispatcher_module": "dispatch_v13_retry_successor", "previous_run_directory": str(previous),
             "previous_contract_sha256": old_contract, "previous_authorization_sha256": old_auth, "seed": 12203, "round": 5,
             "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 2},
             "reviewed_fitted_prefix_files": prefix, "previous_transport_observation": status.RELAY_12203_WAVE07_OBSERVATION,
             "previous_attempts": receipts, "reviewed_retry_chain": [status._relay_review(previous, old_contract, old_auth, exact, receipts)]}
    current = {"run_directory": str(run), "profile": {**profile, "resume_boundaries": {"12203": {"round": 5}}},
               "predecessor": proof, "source_sha256": status.RELAY_RETRY_QUEUE_SOURCES, "budget": exact["execution_budget"]}
    contract = status.record_digest(current)
    approval = {"contract_sha256": contract, **flags}
    write_json(run / "formal_contract.json", current)
    write_json(run / "authorization.json", approval)
    monkeypatch.setattr(status, "RELAY_12203_WAVE07_CONTRACT", contract)
    monkeypatch.setattr(status, "RELAY_12203_WAVE07_AUTH", status.record_digest(approval))
    assert status.relay_retry_07_12203_contract(tmp_path, previous, old_contract) == (
        run, contract, 5, 3, exact["relay_not_before_unix"],
    )
    old_link = old_turn / "attempt_01.replay_link.json"
    original_link = old_link.read_bytes()
    write_json(old_link, {"modified": True})
    assert status.relay_retry_07_12203_contract(tmp_path, previous, old_contract) is None
    old_link.write_bytes(original_link)

    successor = tmp_path / f"{status.SERIAL_IPV4}-12203-relay-retry-08"
    turn = run / "seed_12203/round_05/turn_00"
    attempts, hashes = [], []
    for item in status.RELAY_12203_WAVE07_ATTEMPTS:
        path = turn / f"attempt_{item['attempt']:02d}.terminal.json"
        write_json(path, {**item, "status": "stopped"})
        attempts.append({**item, "terminal_sha256": status.record_digest(status.read_json(path))})
        hashes.append(status.checksum(path))
    replay_hashes = {}
    for name in status.RELAY_12203_WAVE07_REPLAY_FILES:
        write_json(turn / name, {"reviewed": name})
        replay_hashes[name] = status.checksum(turn / name)
    monkeypatch.setattr(status, "RELAY_12203_WAVE07_TERMINALS", tuple(hashes))
    monkeypatch.setattr(status, "RELAY_12203_WAVE07_REPLAY_FILES", replay_hashes)
    latest = {**status.RELAY_12203_WAVE08_PROOF, "previous_terminal_file_sha256": hashes[-1]}
    monkeypatch.setattr(status, "RELAY_12203_WAVE08_PROOF", latest)
    current_approval_sha = status.record_digest(approval)
    last = status._relay_review(run, contract, current_approval_sha, latest, attempts)
    successor_proof = {**proof, **latest, "previous_run_directory": str(run), "previous_contract_sha256": contract,
                       "previous_authorization_sha256": current_approval_sha, "previous_attempts": attempts,
                       "previous_transport_observation": status.RELAY_12203_WAVE08_OBSERVATION,
                       "reviewed_retry_chain": [*proof["reviewed_retry_chain"], last]}
    successor_record = {**current, "run_directory": str(successor), "predecessor": successor_proof}

    def publish(candidate, **authorization):
        sha = status.record_digest(candidate)
        write_json(successor / "formal_contract.json", candidate)
        write_json(successor / "authorization.json", {"contract_sha256": sha, **authorization})
        return sha

    def reviewed():
        return status.reviewed_relay_contract(tmp_path, 12203, 8, run, contract)

    publish(successor_record)
    assert reviewed() is None
    for bad in ({"reviewed_retry_chain": [last]}, {"previous_attempts": attempts[1:]},
                {"recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}},
                {"previous_transport_observation": {}}, {"relay_not_before_unix": latest["relay_not_before_unix"] - 1}):
        publish({**successor_record, "predecessor": {**successor_proof, **bad}}, **flags)
        assert reviewed() is None
    publish({**successor_record, "profile": {**current["profile"], "replay_eligible_from_round": 6}}, **flags)
    assert reviewed() is None
    successor_sha = publish(successor_record, **flags)
    assert reviewed() == (successor, successor_sha, 5, 3, latest["relay_not_before_unix"])
    commands = []
    monkeypatch.setattr(status, "command_lines", lambda command: commands.append(command) or ["  pid = 1234"])
    assert status.dispatcher_pid(12203, "ipv4_relay_retry_08") == 1234
    assert commands[-1][-1].endswith("/com.purellm.a7-ipv4-v13-12203-relay-retry-08")
    link = turn / "attempt_01.replay_link.json"
    original = link.read_bytes()
    write_json(link, {"modified": True})
    assert reviewed() is None
    link.write_bytes(original)

    after = tmp_path / f"{status.SERIAL_IPV4}-12203-relay-retry-09"
    wave08_turn = successor / "seed_12203/round_05/turn_00"
    prior_receipts, prior_shas = [], []
    for item in status.RELAY_12203_WAVE08_ATTEMPTS:
        path = wave08_turn / f"attempt_{item['attempt']:02d}.terminal.json"
        write_json(path, {**item, "status": "stopped"})
        prior_receipts.append({**item, "terminal_sha256": status.record_digest(status.read_json(path))})
        prior_shas.append(status.checksum(path))
    prior_replay = {}
    for name in status.RELAY_12203_WAVE08_REPLAY_FILES:
        write_json(wave08_turn / name, {"reviewed": name})
        prior_replay[name] = status.checksum(wave08_turn / name)
    monkeypatch.setattr(status, "RELAY_12203_WAVE08_TERMINALS", tuple(prior_shas))
    monkeypatch.setattr(status, "RELAY_12203_WAVE08_REPLAY_FILES", prior_replay)
    exact09 = {**status.RELAY_12203_WAVE09_PROOF, "previous_terminal_file_sha256": prior_shas[-1]}
    monkeypatch.setattr(status, "RELAY_12203_WAVE09_PROOF", exact09)
    wave08_auth = status.record_digest(status.read_json(successor / "authorization.json"))
    monkeypatch.setattr(status, "RELAY_12203_WAVE08_CONTRACT", successor_sha)
    monkeypatch.setattr(status, "RELAY_12203_WAVE08_AUTH", wave08_auth)
    prior = status._relay_review(successor, successor_sha, wave08_auth, exact09, prior_receipts)
    proof09 = {**successor_proof, **exact09, "previous_run_directory": str(successor), "previous_contract_sha256": successor_sha,
               "previous_authorization_sha256": wave08_auth, "previous_attempts": prior_receipts,
               "previous_transport_observation": status.RELAY_12203_WAVE09_OBSERVATION,
               "reviewed_retry_chain": [*successor_proof["reviewed_retry_chain"], prior]}
    record09 = {**successor_record, "run_directory": str(after), "predecessor": proof09}
    sha09 = status.record_digest(record09)
    approval09 = {"contract_sha256": sha09, **flags}
    write_json(after / "formal_contract.json", record09)
    write_json(after / "authorization.json", approval09)
    monkeypatch.setattr(status, "RELAY_12203_WAVE09_CONTRACT", sha09)
    assert status.reviewed_relay_contract(tmp_path, 12203, 9, successor, successor_sha) is None
    monkeypatch.setattr(status, "RELAY_12203_WAVE09_AUTH", status.record_digest(approval09))
    assert status.reviewed_relay_contract(tmp_path, 12203, 9, successor, successor_sha) == (
        after, sha09, 5, 3, exact09["relay_not_before_unix"],
    )
    assert status.dispatcher_pid(12203, "ipv4_relay_retry_09") == 1234
    assert commands[-1][-1].endswith("/com.purellm.a7-ipv4-v13-12203-relay-retry-09")
    old_replay_link = wave08_turn / "attempt_01.replay_link.json"
    old_replay_bytes = old_replay_link.read_bytes()
    write_json(old_replay_link, {"modified": True})
    assert status.reviewed_relay_contract(tmp_path, 12203, 9, successor, successor_sha) is None
    old_replay_link.write_bytes(old_replay_bytes)

    wave10 = tmp_path / f"{status.SERIAL_IPV4}-12203-relay-retry-10"
    wave09_turn = after / "seed_12203/round_05/turn_00"
    item09 = status.RELAY_12203_WAVE09_ATTEMPT
    terminal09 = {**item09, "status": "stopped"}
    terminal_path = wave09_turn / "attempt_00.terminal.json"
    write_json(terminal_path, terminal09)
    write_json(wave09_turn / "attempt_00.started.json", {"attempt": 0, "attempt_id": item09["attempt_id"]})
    receipts09 = [{**item09, "terminal_sha256": status.record_digest(terminal09)}]
    exact10 = {**status.RELAY_12203_WAVE10_PROOF, "previous_terminal_file_sha256": status.checksum(terminal_path)}
    monkeypatch.setattr(status, "RELAY_12203_WAVE10_PROOF", exact10)
    auth09 = status.record_digest(approval09)
    review09 = status._relay_review(after, sha09, auth09, exact10, receipts09)
    proof10 = {**proof09, **exact10, "previous_run_directory": str(after), "previous_contract_sha256": sha09,
               "previous_authorization_sha256": auth09, "previous_attempts": receipts09,
               "previous_transport_observation": status.RELAY_12203_WAVE10_OBSERVATION,
               "reviewed_retry_chain": [*proof09["reviewed_retry_chain"], review09],
               "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}}
    record10 = {**record09, "run_directory": str(wave10), "predecessor": proof10,
                "profile": {**record09["profile"], "replay_eligible_from_round": 6},
                "budget": exact10["execution_budget"], "source_sha256": status.RELAY_2357_WAVE05_SOURCES}

    def publish10(candidate):
        value = status.record_digest(candidate)
        approval = {"contract_sha256": value, **flags}
        write_json(wave10 / "formal_contract.json", candidate)
        write_json(wave10 / "authorization.json", approval)
        monkeypatch.setattr(status, "RELAY_12203_WAVE10_CONTRACT", value)
        monkeypatch.setattr(status, "RELAY_12203_WAVE10_AUTH", status.record_digest(approval))
        return value

    def reviewed10():
        return status.reviewed_relay_contract(tmp_path, 12203, 10, after, sha09)

    for invalid in ({**record10, "source_sha256": status.RELAY_RETRY_QUEUE_SOURCES},
                    {**record10, "profile": record09["profile"]},
                    {**record10, "predecessor": {**proof10, "reviewed_retry_chain": [review09]}},
                    {**record10, "budget": record09["budget"]}):
        publish10(invalid)
        assert reviewed10() is None
    sha10 = publish10(record10)
    write_json(wave10 / "authorization.json", {"contract_sha256": sha10, **flags, "authorized_by": "different"})
    assert reviewed10() is None
    publish10(record10)
    assert reviewed10() == (wave10, sha10, 5, 3, exact10["relay_not_before_unix"])
    assert status.dispatcher_pid(12203, "ipv4_relay_retry_10") == 1234
    assert commands[-1][-1].endswith("/com.purellm.a7-ipv4-v13-12203-relay-retry-10")
    write_json(wave09_turn / "attempt_01.started.json", {"attempt": 1})
    assert reviewed10() is None
    (wave09_turn / "attempt_01.started.json").unlink()
    check_12203_wave11(tmp_path, monkeypatch, capsys, flags, previous, old_contract, wave10, sha10, record10)


def check_12203_wave11(runs, monkeypatch, capsys, flags, fitted_run, fitted_sha, previous, previous_sha, previous_record):
    status = prompt_pilot_status
    seed = 12203
    cohort = runs / status.SERIAL_IPV4
    cohort_record = {"run_directory": str(cohort), "profile": {"confirmation_seeds": list(status.SEEDS),
                     "provider_capability": "gated_ipv4"}, "predecessor": {"dispatcher_module": "dispatch_prepared_ipv4"}}
    write_json(cohort / "formal_contract.json", cohort_record)
    write_json(cohort / "authorization.json", {"contract_sha256": status.record_digest(cohort_record), **flags})
    turn = previous / "seed_12203/round_05/turn_00"
    attempt = status.RELAY_12203_WAVE10_ATTEMPT
    terminal = {**attempt, "status": "stopped"}
    terminal_file = turn / "attempt_00.terminal.json"
    write_json(turn / "attempt_00.started.json", {"attempt": 0, "attempt_id": attempt["attempt_id"]})
    write_json(terminal_file, terminal)
    exact = {**status.RELAY_12203_WAVE11_PROOF, "previous_terminal_file_sha256": status.checksum(terminal_file)}
    monkeypatch.setattr(status, "RELAY_12203_WAVE11_PROOF", exact)
    receipt = {**attempt, "terminal_sha256": status.record_digest(terminal)}
    consumed = status._relay_review(previous, previous_sha, status.RELAY_12203_WAVE10_AUTH, exact, [receipt])
    old_proof = previous_record["predecessor"]
    run = runs / f"{status.SERIAL_IPV4}-12203-relay-retry-11"
    proof = {**old_proof, **exact, "previous_run_directory": str(previous), "previous_contract_sha256": previous_sha,
             "previous_authorization_sha256": status.RELAY_12203_WAVE10_AUTH, "previous_attempts": [receipt],
             "previous_transport_observation": status.RELAY_12203_WAVE11_OBSERVATION,
             "reviewed_retry_chain": [*old_proof["reviewed_retry_chain"], consumed],
             "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}}
    record = {**previous_record, "run_directory": str(run), "predecessor": proof, "budget": exact["execution_budget"],
              "profile": {**previous_record["profile"], "replay_eligible_from_round": 5}}

    def publish(candidate):
        sha = status.record_digest(candidate)
        authorization = {"contract_sha256": sha, **flags}
        write_json(run / "formal_contract.json", candidate)
        write_json(run / "authorization.json", authorization)
        monkeypatch.setattr(status, "RELAY_12203_WAVE11_CONTRACT", sha)
        monkeypatch.setattr(status, "RELAY_12203_WAVE11_AUTH", status.record_digest(authorization))
        return sha

    publish({**record, "predecessor": {**proof, "reviewed_retry_chain": [consumed]}})
    assert status.reviewed_relay_contract(runs, seed, 11, previous, previous_sha) is None
    sha = publish(record)
    assert status.reviewed_relay_contract(runs, seed, 11, previous, previous_sha) == (run, sha, 5, 3, exact["relay_not_before_unix"])
    monkeypatch.setattr(status, "resilient_contract", lambda *_: (runs / "resilient", "resilient", 3, 3))
    monkeypatch.setattr(status, "resilient_failure_contract", lambda *_: (runs / "failed", "failed"))
    monkeypatch.setattr(status, "relay_retry_contract", lambda *_: (runs / "retry", "retry", 3, 3))
    monkeypatch.setattr(status, "relay_retry_02_contract", lambda *_: (runs / "wave02", "wave02", 3, 3, 10))
    reviewed = status.reviewed_relay_contract
    monkeypatch.setattr(status, "reviewed_relay_contract", lambda _runs, _seed, wave, *args: (
        reviewed(_runs, _seed, wave, *args) if wave >= 7 else
        (fitted_run if wave == 6 else runs / f"wave{wave}", fitted_sha if wave == 6 else f"wave{wave}", 4, 3, 10)))
    monkeypatch.setattr(status.time, "time", lambda: exact["relay_not_before_unix"] - 30)
    row = remote_seed(runs, seed)
    assert (row["run"], row["segment"], row["labels"]) == (run.name, "ipv4_relay_retry_11", 493)
    assert row["api"] == {"status": "relay_boundary_cooldown", "relay_retry_wait_seconds": 30}
    assert row["reviewed_previous_api"] == {
        "http_status": 200, "error_type": "response_id_mismatch", "sse_event_count": 25, "event_type": "response.failed",
        "complete_event_seen": False, "response_id_matches_transport_alias": True,
    }
    assert row["boundary_api"]["maximum_physical_attempts"] == 2
    assert row["boundary_api"]["replay_requires_verified_response_id"] is True
    assert row["boundary_api"]["future_replay_from_round"] == 6
    commands = []
    monkeypatch.setattr(status, "command_lines", lambda command: commands.append(command) or [])
    status.dispatcher_pid(seed, row["segment"])
    assert commands[0][-1].endswith("/com.purellm.a7-ipv4-v13-12203-relay-retry-11")
    status.show({"seeds": [row], "mac": {"keys": [], "limits": {}, "queue": []}})
    printed = capsys.readouterr().out
    assert "SSE 25  事件 response.failed  未收到完整模型答复" in printed
    assert "传输别名一致" in printed
    check_12203_wave12(runs, monkeypatch, capsys, flags, run, sha, record)
    write_json(turn / "attempt_01.started.json", {"unexpected": True})
    assert remote_seed(runs, seed)["run"] == previous.name


def check_12203_wave12(runs, monkeypatch, capsys, flags, previous, previous_sha, previous_record):
    status = prompt_pilot_status
    turn = previous / "seed_12203/round_05/turn_00"
    snapshots, receipts, terminal_hashes = {}, [], []
    for attempt in status.RELAY_12203_WAVE11_ATTEMPTS:
        terminal = {**attempt, "status": "stopped"}
        filename = turn / f"attempt_{attempt['attempt']:02d}.terminal.json"
        write_json(filename, terminal)
        terminal_hashes.append(status.checksum(filename))
        receipts.append({**attempt, "terminal_sha256": status.record_digest(terminal)})
    for name in status.RELAY_12203_WAVE11_REPLAY_FILES:
        write_json(turn / name, {"verified": name})
        snapshots[name] = status.checksum(turn / name)
    monkeypatch.setattr(status, "RELAY_12203_WAVE11_TERMINALS", tuple(terminal_hashes))
    monkeypatch.setattr(status, "RELAY_12203_WAVE11_REPLAY_FILES", snapshots)
    exact = {**status.RELAY_12203_WAVE12_PROOF, "previous_terminal_file_sha256": terminal_hashes[-1]}
    monkeypatch.setattr(status, "RELAY_12203_WAVE12_PROOF", exact)
    history = status._relay_review(previous, previous_sha, status.RELAY_12203_WAVE11_AUTH, exact, receipts)
    old_proof = previous_record["predecessor"]
    run = runs / f"{status.SERIAL_IPV4}-12203-relay-retry-12"
    proof = {**old_proof, **exact, "previous_run_directory": str(previous), "previous_contract_sha256": previous_sha,
             "previous_authorization_sha256": status.RELAY_12203_WAVE11_AUTH,
             "previous_attempts": receipts, "previous_transport_observation": status.RELAY_12203_WAVE12_OBSERVATION,
             "reviewed_retry_chain": [*old_proof["reviewed_retry_chain"], history]}
    record = {**previous_record, "run_directory": str(run), "predecessor": proof,
              "source_sha256": status.RELAY_12203_WAVE12_SOURCES}

    def publish(candidate):
        sha = status.record_digest(candidate)
        authorization = {"contract_sha256": sha, **flags}
        write_json(run / "formal_contract.json", candidate)
        write_json(run / "authorization.json", authorization)
        monkeypatch.setattr(status, "RELAY_12203_WAVE12_CONTRACT", sha)
        monkeypatch.setattr(status, "RELAY_12203_WAVE12_AUTH", status.record_digest(authorization))
        return sha

    publish({**record, "source_sha256": status.RELAY_2357_WAVE05_SOURCES})
    assert remote_seed(runs, 12203)["run"] == previous.name
    sha = publish(record)
    monkeypatch.setattr(status.time, "time", lambda: exact["relay_not_before_unix"] - 30)
    row = remote_seed(runs, 12203)
    assert (row["run"], row["segment"], row["labels"], row["contract_sha256"]) == (run.name, "ipv4_relay_retry_12", 493, sha)
    assert row["api"] == {"status": "relay_boundary_cooldown", "relay_retry_wait_seconds": 30}
    assert row["reviewed_previous_api"] == {
        "http_status": 200, "error_type": "response_id_mismatch", "sse_event_count": 25, "event_type": "response.failed",
        "complete_event_seen": False, "response_id_matches_transport_alias": True, "archived_failure_count": 2,
    }
    assert row["boundary_api"]["maximum_physical_attempts"] == 2
    assert row["boundary_api"]["replay_requires_verified_response_id"] is True
    assert row["boundary_api"]["future_replay_from_round"] == 6
    commands = []
    monkeypatch.setattr(status, "command_lines", lambda command: commands.append(command) or [])
    status.dispatcher_pid(12203, row["segment"])
    assert commands[0][-1].endswith("/com.purellm.a7-ipv4-v13-12203-relay-retry-12")
    status.show({"seeds": [row], "mac": {"keys": [], "limits": {}, "queue": []}})
    assert "2 次提交均收到同类失败事件，原回执未保存具体错误码" in capsys.readouterr().out
    check_12203_wave13(runs, monkeypatch, capsys, flags, run, sha, record)
    write_json(turn / "attempt_01.replay_link.json", {"changed": True})
    assert remote_seed(runs, 12203)["run"] == previous.name


def check_12203_wave13(runs, monkeypatch, capsys, flags, previous, previous_sha, previous_record):
    status = prompt_pilot_status
    turn = previous / "seed_12203/round_05/turn_00"
    snapshots, receipts, terminal_hashes = {}, [], []
    for attempt in status.RELAY_12203_WAVE12_ATTEMPTS:
        slot = attempt["attempt"]
        terminal = {**attempt, "status": "stopped"}
        filename = turn / f"attempt_{slot:02d}.terminal.json"
        write_json(turn / f"attempt_{slot:02d}.started.json", {"attempt": slot, "attempt_id": attempt["attempt_id"]})
        write_json(filename, terminal)
        terminal_hashes.append(status.checksum(filename))
        receipts.append({**attempt, "terminal_sha256": status.record_digest(terminal)})
    for name in status.RELAY_12203_WAVE12_REPLAY_FILES:
        write_json(turn / name, {"verified": name})
        snapshots[name] = status.checksum(turn / name)
    monkeypatch.setattr(status, "RELAY_12203_WAVE12_TERMINALS", tuple(terminal_hashes))
    monkeypatch.setattr(status, "RELAY_12203_WAVE12_REPLAY_FILES", snapshots)
    exact = {**status.RELAY_12203_WAVE13_PROOF, "previous_terminal_file_sha256": terminal_hashes[-1]}
    monkeypatch.setattr(status, "RELAY_12203_WAVE13_PROOF", exact)
    history = status._relay_review(previous, previous_sha, status.RELAY_12203_WAVE12_AUTH, exact, receipts)
    old_proof = previous_record["predecessor"]
    run = runs / f"{status.SERIAL_IPV4}-12203-relay-retry-13"
    proof = {**old_proof, **exact, "previous_run_directory": str(previous), "previous_contract_sha256": previous_sha,
             "previous_authorization_sha256": status.RELAY_12203_WAVE12_AUTH,
             "previous_attempts": receipts, "previous_transport_observation": status.RELAY_12203_WAVE13_OBSERVATION,
             "reviewed_retry_chain": [*old_proof["reviewed_retry_chain"], history],
             "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}}
    record = {**previous_record, "run_directory": str(run), "predecessor": proof, "budget": exact["execution_budget"],
              "source_sha256": status.RELAY_12203_WAVE13_SOURCES,
              "profile": {**previous_record["profile"], "replay_eligible_from_round": 6}}

    def publish(candidate):
        sha = status.record_digest(candidate)
        authorization = {"contract_sha256": sha, **flags}
        write_json(run / "formal_contract.json", candidate)
        write_json(run / "authorization.json", authorization)
        monkeypatch.setattr(status, "RELAY_12203_WAVE13_CONTRACT", sha)
        monkeypatch.setattr(status, "RELAY_12203_WAVE13_AUTH", status.record_digest(authorization))
        return sha

    for invalid in ({**record, "source_sha256": status.RELAY_12203_WAVE12_SOURCES},
                    {**record, "predecessor": {**proof, "previous_attempts": [receipts[0], {**receipts[1], "ambiguous": True}]}}):
        publish(invalid)
        assert remote_seed(runs, 12203)["run"] == previous.name
    sha = publish(record)
    monkeypatch.setattr(status.time, "time", lambda: exact["relay_not_before_unix"] - 30)
    row = remote_seed(runs, 12203)
    assert (row["run"], row["segment"], row["labels"], row["contract_sha256"]) == (run.name, "ipv4_relay_retry_13", 493, sha)
    assert row["api"] == {"status": "relay_boundary_cooldown", "relay_retry_wait_seconds": 30}
    assert row["reviewed_previous_api"] == {
        "http_status": 200, "error_type": "response_id_mismatch", "sse_event_count": 23, "event_type": "response.failed",
        "complete_event_seen": False, "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error",
        "pre_post_failure": {"error_type": "connect_timeout_before_post", "network_phase": "tls", "model_post_observed": False},
    }
    assert row["boundary_api"]["maximum_physical_attempts"] == 1
    assert row["boundary_api"]["future_replay_from_round"] == 6
    commands = []
    monkeypatch.setattr(status, "command_lines", lambda command: commands.append(command) or [])
    status.dispatcher_pid(12203, row["segment"])
    assert commands[0][-1].endswith("/com.purellm.a7-ipv4-v13-12203-relay-retry-13")
    status.show({"seeds": [row], "mac": {"keys": [], "limits": {}, "queue": []}})
    printed = capsys.readouterr().out
    assert "upstream_error  SSE 23" in printed
    assert "TLS 握手时超时，未观察到模型 POST、HTTP 响应或 SSE" in printed
    check_12203_wave14(runs, monkeypatch, capsys, flags, run, sha, record)
    write_json(turn / "attempt_01.reconciliation.json", {"unexpected": True})
    assert remote_seed(runs, 12203)["run"] == previous.name


def check_12203_wave14(runs, monkeypatch, capsys, flags, previous, previous_sha, previous_record):
    status = prompt_pilot_status
    turn = previous / "seed_12203/round_05/turn_00"
    attempt = status.RELAY_12203_WAVE13_ATTEMPT
    terminal = {**attempt, "status": "stopped"}
    filename = turn / "attempt_00.terminal.json"
    write_json(turn / "attempt_00.started.json", {"attempt": 0, "attempt_id": attempt["attempt_id"]})
    write_json(filename, terminal)
    exact = {**status.RELAY_12203_WAVE14_PROOF, "previous_terminal_file_sha256": status.checksum(filename)}
    monkeypatch.setattr(status, "RELAY_12203_WAVE14_PROOF", exact)
    receipt = {**attempt, "terminal_sha256": status.record_digest(terminal)}
    last = status._relay_review(previous, previous_sha, status.RELAY_12203_WAVE13_AUTH, exact, [receipt])
    old_proof = previous_record["predecessor"]
    run = runs / f"{status.SERIAL_IPV4}-12203-relay-retry-14"
    proof = {**old_proof, **exact, "previous_run_directory": str(previous), "previous_contract_sha256": previous_sha,
             "previous_authorization_sha256": status.RELAY_12203_WAVE13_AUTH,
             "checkpoint_sha256": status.RELAY_12203_WAVE15_CHECKPOINT,
             "previous_attempts": [receipt], "previous_transport_observation": status.RELAY_12203_WAVE14_OBSERVATION,
             "reviewed_retry_chain": [*old_proof["reviewed_retry_chain"], last],
             "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}}
    record = {**previous_record, "run_directory": str(run), "predecessor": proof, "budget": exact["execution_budget"],
              "source_sha256": status.RELAY_12203_WAVE14_SOURCES,
              "profile": {**previous_record["profile"], "replay_eligible_from_round": 5}}

    def publish(candidate):
        sha = status.record_digest(candidate)
        authorization = {"contract_sha256": sha, **flags}
        write_json(run / "formal_contract.json", candidate)
        write_json(run / "authorization.json", authorization)
        monkeypatch.setattr(status, "RELAY_12203_WAVE14_CONTRACT", sha)
        monkeypatch.setattr(status, "RELAY_12203_WAVE14_AUTH", status.record_digest(authorization))
        return sha

    publish({**record, "source_sha256": status.RELAY_12203_WAVE13_SOURCES})
    assert remote_seed(runs, 12203)["run"] == previous.name
    sha = publish(record)
    monkeypatch.setattr(status.time, "time", lambda: exact["relay_not_before_unix"] - 30)
    row = remote_seed(runs, 12203)
    assert (row["run"], row["segment"], row["labels"], row["contract_sha256"]) == (run.name, "ipv4_relay_retry_14", 493, sha)
    assert row["api"] == {"status": "relay_boundary_cooldown", "relay_retry_wait_seconds": 30}
    assert row["reviewed_previous_api"] == {
        "http_status": 200, "error_type": "response_id_mismatch", "sse_event_count": 25, "event_type": "response.failed",
        "complete_event_seen": False, "response_id_matches_transport_alias": True, "provider_error_code": "upstream_error",
        "earlier_pre_post_failure": {"error_type": "connect_timeout_before_post", "network_phase": "tls",
                                     "model_post_observed": False, "wave": 12},
    }
    assert row["boundary_api"]["maximum_physical_attempts"] == 2
    assert row["boundary_api"]["replay_requires_verified_response_id"] is True
    assert row["boundary_api"]["future_replay_from_round"] == 6
    commands = []
    monkeypatch.setattr(status, "command_lines", lambda command: commands.append(command) or [])
    status.dispatcher_pid(12203, row["segment"])
    assert commands[0][-1].endswith("/com.purellm.a7-ipv4-v13-12203-relay-retry-14")
    status.show({"seeds": [row], "mac": {"keys": [], "limits": {}, "queue": []}})
    printed = capsys.readouterr().out
    assert "upstream_error  SSE 25" in printed
    assert "更早的 W12 的另一次尝试在 TLS 握手时超时" in printed
    check_12203_wave15(runs, monkeypatch, flags, run, sha, record)
    write_json(turn / "attempt_01.started.json", {"unexpected": True})
    assert remote_seed(runs, 12203)["run"] == previous.name


def check_12203_wave15(runs, monkeypatch, flags, previous, previous_sha, previous_record):
    status = prompt_pilot_status
    turn = previous / "seed_12203/round_05/turn_00"
    attempt = status.RELAY_12203_WAVE14_ATTEMPT
    terminal = {**attempt, "status": "stopped"}
    filename = turn / "attempt_00.terminal.json"
    write_json(turn / "attempt_00.started.json", {"attempt": 0, "attempt_id": attempt["attempt_id"]})
    write_json(filename, terminal)
    exact = {**status.RELAY_12203_WAVE15_PROOF, "previous_terminal_file_sha256": status.checksum(filename)}
    monkeypatch.setattr(status, "RELAY_12203_WAVE15_PROOF", exact)
    receipt = {**attempt, "terminal_sha256": status.record_digest(terminal)}
    last = status._relay_review(previous, previous_sha, status.RELAY_12203_WAVE14_AUTH, exact, [receipt])
    old_proof = previous_record["predecessor"]
    run = runs / f"{status.SERIAL_IPV4}-12203-relay-retry-15"
    proof = {**old_proof, **exact, "previous_run_directory": str(previous), "previous_contract_sha256": previous_sha,
             "previous_authorization_sha256": status.RELAY_12203_WAVE14_AUTH,
             "previous_attempts": [receipt], "previous_transport_observation": status.RELAY_12203_WAVE15_OBSERVATION,
             "reviewed_retry_chain": [*old_proof["reviewed_retry_chain"], last],
             "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}}
    record = {**previous_record, "run_directory": str(run), "predecessor": proof, "budget": exact["execution_budget"],
              "source_sha256": status.RELAY_12203_WAVE15_SOURCES,
              "profile": {**previous_record["profile"], "replay_eligible_from_round": 6}}

    def publish(candidate):
        sha = status.record_digest(candidate)
        authorization = {"contract_sha256": sha, **flags}
        write_json(run / "formal_contract.json", candidate)
        write_json(run / "authorization.json", authorization)
        monkeypatch.setattr(status, "RELAY_12203_WAVE15_CONTRACT", sha)
        monkeypatch.setattr(status, "RELAY_12203_WAVE15_AUTH", status.record_digest(authorization))
        return sha

    publish({**record, "predecessor": {**proof, "reviewed_retry_chain": proof["reviewed_retry_chain"][:-1]}})
    assert remote_seed(runs, 12203)["run"] == previous.name
    sha = publish(record)
    with monkeypatch.context() as clock:
        clock.setattr(status.time, "time", lambda: exact["relay_not_before_unix"] - 30)
        row = remote_seed(runs, 12203)
    assert (row["run"], row["segment"], row["labels"], row["contract_sha256"]) == (run.name, "ipv4_relay_retry_15", 493, sha)
    assert row["api"] == {"status": "relay_boundary_cooldown", "relay_retry_wait_seconds": 30}
    assert row["reviewed_previous_api"] == {
        "http_status": 200, "error_type": "sse_inactivity_timeout", "sse_event_count": 15,
        "event_type": "response.output_item.added", "last_sse_seconds": 153.12645345786586,
        "complete_event_seen": False, "sse_inactivity_seconds": 900,
    }
    assert row["boundary_api"]["maximum_physical_attempts"] == 1
    assert row["boundary_api"]["future_replay_from_round"] == 6
    assert row["boundary_api"]["future_replay_requires_verified_response_id"] is True
    commands = []
    monkeypatch.setattr(status, "command_lines", lambda command: commands.append(command) or [])
    status.dispatcher_pid(12203, row["segment"])
    assert commands[0][-1].endswith("/com.purellm.a7-ipv4-v13-12203-relay-retry-15")
    unexpected = turn / "attempt_01.started.json"
    write_json(unexpected, {"unexpected": True})
    assert remote_seed(runs, 12203)["run"] == previous.name
    unexpected.unlink()
    def publish_next(destination, candidate):
        new_sha = status.record_digest(candidate)
        approval = {"contract_sha256": new_sha, **flags}
        write_json(destination / "formal_contract.json", candidate)
        write_json(destination / "authorization.json", approval)
        return new_sha, status.record_digest(approval)

    check_repeated_boundary(runs, monkeypatch, publish_next, 12203, 16, run, sha, status.RELAY_12203_WAVE15_AUTH, record)


def test_12203_wave07_reviewed_trace_matches_the_producer():
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_retry_successor

    scope = run_v13_retry_successor.SCOPES[(12203, 7)]
    assert prompt_pilot_status.RELAY_12203_WAVE07_OBSERVATION == scope.archive[-1].progress


def test_boundary_attempt_counts_distinguish_reserved_from_confirmed_posted(tmp_path):
    run = tmp_path / "run"
    seed = run / "seed_12203"
    turn = seed / "round_05/turn_00"
    first, second = "a" * 32, "b" * 32
    write_json(turn / "attempt_00.started.json", {"attempt": 0, "attempt_id": first})
    write_json(turn / "attempt_01.started.json", {"attempt": 1, "attempt_id": second})
    write_json(run / f"bridge/outbox/{first}.json", {"attempt_id": first, "job_sha256": "f" * 64})
    write_json(run / f"bridge/results/{first}.json", {"attempt_id": first, "job_sha256": "f" * 64, "provider_dispatched": True})
    assert prompt_pilot_status.boundary_attempt_counts(run, seed, 5, 2) == {
        "physical_attempts_reserved": 2, "physical_attempts_posted_confirmed": 1, "physical_attempts_remaining": 0,
    }


def test_relay_countdown_uses_only_bound_failed_and_replay_receipts(tmp_path, monkeypatch):
    directory = tmp_path / "seed_157"
    turn = directory / "round_03/turn_00"
    contract, identity, request_id = "a" * 64, "b" * 32, "c" * 64
    started = {"attempt_id": identity, "request_id": request_id, "attempt": 0, "transport": "mac_bridge"}
    terminal = {**started, "status": "stopped", "ambiguous": True, "error": "http_524",
                "failed_unix": 120, "not_before_unix": 125, "bridge_result_sha256": "d" * 64}
    seal = prompt_pilot_status.record_digest(terminal)
    permit = {"attempt_id": identity, "request_id": request_id, "next_attempt": 1,
              "decision": "authorized_replay_unknown", "contract_sha256": contract,
              "bridge_result_sha256": terminal["bridge_result_sha256"], "original_terminal_sha256": seal,
              "accept_duplicate_billing_risk": True}
    replay = {**started, "status": "retryable", "ambiguous": False, "error": "authorized_replay_unknown",
              "next_attempt": 1, "sealed_relay_retry_seconds": 180, "not_before_unix": 300,
              "supersedes_sha256": seal, "replay_authorization_sha256": prompt_pilot_status.record_digest(permit)}
    for filename, payload in (("attempt_00.started.json", started), ("attempt_00.terminal.json", terminal),
                              ("attempt_00.replay_authorization.json", permit), ("attempt_00.replay.terminal.json", replay)):
        write_json(turn / filename, payload)
    monkeypatch.setattr(prompt_pilot_status.time, "time", lambda: 200)
    assert pending_attempt(tmp_path, directory, 3, relay_retry=180, contract=contract)["relay_retry_wait_seconds"] == 100
    assert "relay_retry_wait_seconds" not in pending_attempt(tmp_path, directory, 3, relay_retry=180, contract="e" * 64)
    write_json(turn / "attempt_00.replay.terminal.json", {**replay, "not_before_unix": 290})
    assert "relay_retry_wait_seconds" not in pending_attempt(tmp_path, directory, 3, relay_retry=180, contract=contract)
    write_json(turn / "attempt_00.replay.terminal.json", replay)
    write_json(turn / "attempt_01.started.json", {**started, "attempt": 1, "attempt_id": "e" * 32})
    assert "relay_retry_wait_seconds" not in pending_attempt(tmp_path, directory, 3, relay_retry=180, contract=contract)


def test_readiness_counts_before_post_and_reports_sealed_no_post(tmp_path, monkeypatch):
    identity, job_sha, result_sha = "a" * 32, "b" * 64, "c" * 64
    active = {"attempt_id": identity, "round": 6, "turn": 0, "reserved_unix": 100, "status": "waiting_for_mac"}
    attempt = tmp_path / "attempts" / identity
    write_json(tmp_path / "jobs" / f"{identity}.json", {
        "attempt_id": identity, "job_sha256": job_sha,
        "request": {"binding": {"seed": 887, "round": 6}, "logical_index": 0},
    })
    write_json(attempt / "readiness_01.json", {
        "attempt_id": identity, "job_sha256": job_sha, "window": 1, "retry_unix": 130,
    })
    monkeypatch.setattr(prompt_pilot_status.time, "time", lambda: 120)
    row = local_progress(tmp_path, 887, active, track_queue=True, readiness_windows=3)
    assert row["phase"] == "waiting_for_ipv4_route" and row["provider_dispatched"] is False
    assert row["readiness_windows_used"] == 1 and row["readiness_retry_seconds"] == 10
    assert "provider_elapsed_seconds" not in row

    write_json(tmp_path / "results" / f"{identity}.json", {
        "attempt_id": identity, "job_sha256": job_sha, "result_sha256": result_sha,
        "provider_dispatched": False, "outcome": {"status": "failed"},
    })
    write_json(tmp_path / "delivered" / f"{identity}.json", {"attempt_id": identity, "result_sha256": result_sha})
    row = local_progress(tmp_path, 887, active, track_queue=True, readiness_windows=3)
    assert row["phase"] == "transport_failure" and row["delivered"] and row["provider_dispatched"] is False
    assert "readiness_retry_seconds" not in row


def test_status_distinguishes_unauthenticated_readiness_live_model_sse_and_sealed_post(tmp_path, capsys):
    status = prompt_pilot_status
    identity, job_sha = "a" * 32, "6" * 64
    active = {"attempt_id": identity, "round": 6, "turn": 0, "status": "waiting_for_mac"}
    attempt = tmp_path / "attempts" / identity
    write_json(tmp_path / "jobs" / f"{identity}.json", {
        "attempt_id": identity, "job_sha256": job_sha,
        "request": {"binding": {"seed": 887, "round": 6}, "logical_index": 0},
    })
    write_json(attempt / "readiness_01.json", {
        "attempt_id": identity, "job_sha256": job_sha, "window": 1, "retry_unix": 130, "http_status": 401,
    })
    row = {"seed": 887, "labels": 525, "target_labels": 1485, "completed_rounds": 6,
           "target_rounds": 36, "phase": "waiting_for_mac", "worker_pid": None, "coordinator_pid": None,
           "run": None, "api": active}

    def rendered():
        row["provider"] = local_progress(tmp_path, 887, active, readiness_windows=3)
        status.show({"seeds": [row], "mac": {"keys": [], "limits": {}, "queue": []}})
        return capsys.readouterr().out

    assert "模型请求 未提交" in rendered()
    put_progress(tmp_path, identity, 6, phase="http_headers", http_status=200, sse_event_count=0)
    assert "模型请求 待确认" in rendered()
    put_progress(tmp_path, identity, 6, http_status=200, sse_event_count=3, event_type="response.in_progress")
    assert "模型请求 已收到 SSE，回执待封存" in rendered()
    assert "provider_dispatched" not in row["provider"]
    write_json(tmp_path / "results" / f"{identity}.json", {
        "attempt_id": identity, "job_sha256": job_sha, "result_sha256": "c" * 64, "provider_dispatched": True,
    })
    assert "模型请求 已提交" in rendered()


@pytest.mark.parametrize("seed,wave", [(157, 9), (2357, 8), (6101, 9), (6101, 10), (887, 10)])
def test_reviewed_relay_after_new_cuda_fit_requires_exact_fit_and_archive(tmp_path, monkeypatch, seed, wave):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.run_v13_retry_successor import SCOPES

    status = prompt_pilot_status
    scope = SCOPES[(seed, wave)]
    review, = scope.archive
    start = scope.round
    prefix = f"RELAY_{seed}_WAVE{wave:02d}"
    source_only = (seed, wave) == (6101, 9)
    previous_prefix = "CONTINUATION_6101_R5" if source_only else f"RELAY_{seed}_WAVE{wave - 1:02d}"
    proof = getattr(status, f"{prefix}_PROOF")
    fitted = getattr(status, f"{prefix}_FITTED_FILES")
    checkpoint = getattr(status, f"{prefix}_CHECKPOINT")
    observation = getattr(status, f"{prefix}_OBSERVATION")
    attempts = {(157, 9): (status.RELAY_157_WAVE08_ATTEMPT,), (2357, 8): status.RELAY_2357_WAVE07_ATTEMPTS,
                (6101, 9): (status.RELAY_6101_WAVE09_PRIOR_ATTEMPT,), (6101, 10): status.RELAY_6101_WAVE09_ATTEMPTS,
                (887, 10): (status.RELAY_887_WAVE09_ATTEMPT,)}[seed, wave]
    sources = getattr(status, f"{prefix}_SOURCES", status.RELAY_12203_WAVE12_SOURCES)
    assert fitted == dict(scope.fitted_prefix_files)
    assert {name: sources[name] for name, _ in scope.required_sources} == dict(scope.required_sources)
    assert observation == review.progress
    assert (getattr(status, f"{previous_prefix}_CONTRACT"), getattr(status, f"{previous_prefix}_AUTH")) == (
        review.identity.contract, review.identity.approval,
    )
    assert proof == {"logical_index": 0, "payload_sha256": scope.payload_sha256, "previous_request_id": review.request_id,
                     "previous_terminal_file_sha256": review.terminal_file_sha256,
                     "previous_transport_progress_file_sha256": review.progress_file_sha256,
                     "relay_cooldown_clock": scope.cooldown_field, "relay_not_before_unix": scope.not_before_unix,
                     "newly_authorized_boundary_physical_attempts": scope.boundary_physical_attempts,
                     "execution_budget": scope.budget}
    for index, (item, archived) in enumerate(zip(attempts, review.attempts, strict=True)):
        assert item == {"attempt": index, "attempt_id": archived.attempt_id, "job_sha256": archived.job_sha256,
                        "result_sha256": archived.result_sha256, "error": archived.failure,
                        "provider_dispatched": True, "ambiguous": True}
    assert status.SHA.fullmatch(getattr(status, f"{prefix}_CONTRACT"))
    assert status.SHA.fullmatch(getattr(status, f"{prefix}_AUTH"))

    previous = tmp_path / f"{status.SERIAL_IPV4}{review.identity.suffix}"
    run = tmp_path / scope.run_name
    directory = previous / f"seed_{seed}"
    stage = directory / f"round_{start:02d}"
    turn = stage / "turn_00"
    for name in fitted:
        write_json(directory / name, {"fixture": name})
    write_json(stage / "transition.json", {
        "checkpoint_sha256": checkpoint, "fit_path": "fit_000", "source_sha256": status.checksum(stage / "source.json"),
        "validation_metrics_sha256": status.checksum(stage / "fit_000/validation_metrics.json"),
    })
    fitted = {name: status.checksum(directory / name) for name in fitted}
    monkeypatch.setattr(status, f"{prefix}_FITTED_FILES", fitted)
    receipts, terminals = [], []
    for index, attempt in enumerate(attempts):
        terminal = {**attempt, "status": "stopped"}
        file = turn / f"attempt_{index:02d}.terminal.json"
        write_json(file, terminal)
        write_json(turn / f"attempt_{index:02d}.started.json", {"attempt": index, "attempt_id": attempt["attempt_id"]})
        receipts.append({**attempt, "terminal_sha256": status.record_digest(terminal)})
        terminals.append(status.checksum(file))
    proof = {**proof, "previous_terminal_file_sha256": terminals[-1]}
    monkeypatch.setattr(status, f"{prefix}_PROOF", proof)
    if len(attempts) == 2:
        assert dict(review.replay_files) == getattr(status, f"{previous_prefix}_REPLAY_FILES")
        assert tuple(a.terminal_file_sha256 for a in review.attempts) == getattr(status, f"{previous_prefix}_TERMINALS")
        for name in review.replay_files:
            write_json(turn / name, {"fixture": name})
        monkeypatch.setattr(status, f"{previous_prefix}_TERMINALS", terminals)
        monkeypatch.setattr(status, f"{previous_prefix}_REPLAY_FILES",
                            {name: status.checksum(turn / name) for name in review.replay_files})

    def publish(path, record):
        digest = status.record_digest(record)
        authorization = {"contract_sha256": digest, **dict.fromkeys(status.FORMAL_APPROVAL_FLAGS, True)}
        write_json(path / "formal_contract.json", record)
        write_json(path / "authorization.json", authorization)
        return digest, status.record_digest(authorization)

    old_profile = {"confirmation_seeds": [seed], "target_labels": 1485,
                   "resume_boundaries": {str(seed): {"round": scope.original_start_round}},
                   "audited_future_transport_replay": True, "sealed_relay_retry_seconds": 180,
                   "replay_eligible_from_round": {(157, 9): 4, (2357, 8): 5, (6101, 9): 5, (6101, 10): 7,
                                                  (887, 10): 13}[seed, wave],
                   "ipv4_readiness_windows": 3,
                   "sse_inactivity_seconds": 900, "request_timeouts": {"total_seconds": 3600}}
    old_budget = (status.CONTINUATION_6101_R5_BUDGET if source_only
                  else getattr(status, f"{previous_prefix}_PROOF")["execution_budget"])
    old_proof = {"recovery_budget": {"maximum_logical_requests": 1,
                                    "maximum_physical_attempts": scope.original_boundary_physical_attempts}}
    cached = {}
    if seed == 157:
        cached = {"reviewed_cached_import": status.RELAY_157_WAVE07_CACHED_REVIEW,
                  "public_resource_receipt_sha256": status.RELAY_157_WAVE07_RESOURCE_SHA}
    old_sha, old_auth = publish(previous, {"run_directory": str(previous), "profile": old_profile,
                                          "budget": old_budget, "predecessor": {**old_proof, **cached}})
    monkeypatch.setattr(status, f"{previous_prefix}_CONTRACT", old_sha)
    monkeypatch.setattr(status, f"{previous_prefix}_AUTH", old_auth)
    new_proof = {**proof, **cached, "dispatcher_module": "dispatch_v13_retry_successor", "seed": seed, "round": start,
                 "previous_run_directory": str(previous), "previous_contract_sha256": old_sha,
                 "previous_authorization_sha256": old_auth, "recovery_budget": scope.recovery_budget,
                 "reviewed_fitted_prefix_files": fitted, "checkpoint_sha256": checkpoint,
                 "previous_transport_observation": observation, "previous_attempts": receipts,
                 "reviewed_retry_chain": [status._relay_review(previous, old_sha, old_auth, proof, receipts)]}
    descriptor = {"round": start, "state_sha256": "1" * 64, "memory_sha256": "2" * 64, "user_message_sha256": "3" * 64}
    new_profile = {**old_profile, "resume_boundaries": {str(seed): descriptor},
                   "replay_eligible_from_round": scope.replay_start_round}
    signed = {"run_directory": str(run), "profile": new_profile, "budget": scope.budget,
              "source_sha256": sources, "predecessor": new_proof}
    digest, auth = publish(run, signed)
    monkeypatch.setattr(status, f"{prefix}_CONTRACT", digest)
    assert status.reviewed_relay_contract(tmp_path, seed, wave, previous, old_sha) is None
    monkeypatch.setattr(status, f"{prefix}_AUTH", auth)
    expected = run, digest, start, 3, scope.not_before_unix
    assert status.reviewed_relay_contract(tmp_path, seed, wave, previous, old_sha) == expected
    mutation = {2357: turn / "attempt_01.replay_link.json", 157: directory / "round_06/batch_freeze.json",
                6101: stage / "transition.json", 887: stage / "transition.json"}[seed]
    original = mutation.read_bytes()
    write_json(mutation, {"modified": True})
    assert status.reviewed_relay_contract(tmp_path, seed, wave, previous, old_sha) is None
    mutation.write_bytes(original)
    assert status.reviewed_relay_contract(tmp_path, seed, wave, previous, old_sha) == expected
    if source_only:
        unexpected = directory / "round_05/boundary_user_message.json"
        write_json(unexpected, {"added": True})
        assert status.reviewed_relay_contract(tmp_path, seed, wave, previous, old_sha) is None
        unexpected.unlink()
        write_json(turn / "attempt_01.started.json", {"unexpected": True})
        assert status.reviewed_relay_contract(tmp_path, seed, wave, previous, old_sha) is None
    elif (seed, wave) == (6101, 10):
        assert status.RELAY_6101_WAVE10_ARCHIVED_SSE_COUNTS == tuple(item.progress["sse_event_count"] for item in review.attempts)
        check_repeated_boundary(tmp_path, monkeypatch, publish, seed, wave + 1, run, digest, auth, signed)
        write_json(turn / "attempt_02.started.json", {"unexpected": True})
        assert status.reviewed_relay_contract(tmp_path, seed, wave, previous, old_sha) is None
    elif (seed, wave) == (887, 10):
        write_json(stage / "turn_01/attempt_00.started.json", {"unexpected": True})
        assert status.reviewed_relay_contract(tmp_path, seed, wave, previous, old_sha) is None
    else:
        check_repeated_boundary(tmp_path, monkeypatch, publish, seed, wave + 1, run, digest, auth, signed)


def check_repeated_boundary(runs, monkeypatch, publish, seed, wave, previous, previous_sha, previous_auth, previous_record):
    status = prompt_pilot_status
    prefix = f"RELAY_{seed}_WAVE{wave:02d}"
    previous_prefix = f"RELAY_{seed}_WAVE{wave - 1:02d}"
    known = getattr(status, f"{prefix}_PROOF")
    round_index = previous_record["profile"]["resume_boundaries"][str(seed)]["round"]
    turn = previous / f"seed_{seed}/round_{round_index:02d}/turn_00"
    receipts, terminals = [], []
    attempts = getattr(status, f"{previous_prefix}_ATTEMPTS", None)
    if attempts is None:
        attempts = (getattr(status, f"{previous_prefix}_ATTEMPT"),)
    for attempt in attempts:
        index = attempt["attempt"]
        terminal = {**attempt, "status": "stopped"}
        filename = turn / f"attempt_{index:02d}.terminal.json"
        write_json(turn / f"attempt_{index:02d}.started.json", {"attempt": index, "attempt_id": attempt["attempt_id"]})
        write_json(filename, terminal)
        terminals.append(status.checksum(filename))
        receipts.append({**attempt, "terminal_sha256": status.record_digest(terminal)})
    checked_file = turn / "attempt_00.terminal.json"
    if len(attempts) == 2:
        links = getattr(status, f"{previous_prefix}_REPLAY_FILES")
        for name in links:
            write_json(turn / name, {"fixture": name})
        monkeypatch.setattr(status, f"{previous_prefix}_TERMINALS", terminals)
        monkeypatch.setattr(status, f"{previous_prefix}_REPLAY_FILES", {name: status.checksum(turn / name) for name in links})
        checked_file = turn / "attempt_01.replay_link.json"
    exact = {**known, "previous_terminal_file_sha256": terminals[-1]}
    monkeypatch.setattr(status, f"{prefix}_PROOF", exact)
    old_proof = previous_record["predecessor"]
    new_review = status._relay_review(previous, previous_sha, previous_auth, exact, receipts)
    run = runs / f"{status.SERIAL_IPV4}-{seed}-relay-retry-{wave:02d}"
    physical = exact["newly_authorized_boundary_physical_attempts"]
    proof = {**old_proof, **exact, "previous_run_directory": str(previous), "previous_contract_sha256": previous_sha,
             "previous_authorization_sha256": previous_auth, "previous_attempts": receipts,
             "previous_transport_observation": getattr(status, f"{prefix}_OBSERVATION"),
             "reviewed_retry_chain": [*old_proof["reviewed_retry_chain"], new_review],
             "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": physical}}
    record = {**previous_record, "run_directory": str(run), "predecessor": proof, "budget": exact["execution_budget"],
              "source_sha256": getattr(status, f"{prefix}_SOURCES", previous_record["source_sha256"]),
              "profile": {**previous_record["profile"], "replay_eligible_from_round": round_index + (physical == 1)}}
    if (seed, wave) == (2357, 19):
        record["profile"]["response_headers_seconds"] = 300
    broken = {**record, "predecessor": {**proof, "reviewed_retry_chain": [new_review]}}
    broken_sha, broken_auth = publish(run, broken)
    monkeypatch.setattr(status, f"{prefix}_CONTRACT", broken_sha)
    monkeypatch.setattr(status, f"{prefix}_AUTH", broken_auth)
    assert status.reviewed_relay_contract(runs, seed, wave, previous, previous_sha) is None
    sha, auth = publish(run, record)
    monkeypatch.setattr(status, f"{prefix}_CONTRACT", sha)
    monkeypatch.setattr(status, f"{prefix}_AUTH", auth)
    expected = run, sha, round_index, 3, exact["relay_not_before_unix"]
    assert status.reviewed_relay_contract(runs, seed, wave, previous, previous_sha) == expected
    original = checked_file.read_bytes()
    write_json(checked_file, {"modified": True})
    assert status.reviewed_relay_contract(runs, seed, wave, previous, previous_sha) is None
    checked_file.write_bytes(original)
    assert status.reviewed_relay_contract(runs, seed, wave, previous, previous_sha) == expected
    if (seed, wave) in ((157, 10), (157, 11), (157, 12), (887, 8), (2357, 9), (2357, 10), (2357, 11), (2357, 12),
                        (157, 13), (157, 14), (157, 15), (157, 16), (157, 17), (157, 18),
                        (2357, 13), (2357, 14), (2357, 15), (2357, 16), (2357, 17), (2357, 18),
                        (6101, 11), (6101, 12), (6101, 13), (6101, 14), (6101, 15), (6101, 16), (6101, 17), (6101, 18),
                        (12203, 16), (12203, 17), (12203, 18), (12203, 19), (12203, 20), (12203, 21),
                        (12203, 22), (12203, 23), (12203, 24), (12203, 25), (12203, 26), (12203, 27), (12203, 28), (12203, 29), (12203, 30), (12203, 31), (12203, 32), (12203, 33), (12203, 34), (12203, 35), (12203, 36),
                        (157, 19), (2357, 19), (6101, 19), (157, 20), (2357, 20), (6101, 20),
                        (2357, 21), (6101, 21), (157, 21), (2357, 22), (6101, 22), (157, 22), (157, 23), (2357, 23), (6101, 23), (157, 24), (2357, 24), (6101, 24), (2357, 25), (6101, 25), (157, 25),
                        (157, 26), (2357, 26), (6101, 26), (157, 27), (2357, 27), (6101, 27), (157, 28), (157, 29), (157, 30), (2357, 28), (2357, 29), (2357, 30)):
        check_repeated_boundary(runs, monkeypatch, publish, seed, wave + 1, run, sha, auth, record)


def test_2357_wave09_monitor_pins_the_same_l493_request_and_four_archived_attempts():
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.run_v13_retry_successor import SCOPES

    status = prompt_pilot_status
    scope = SCOPES[(2357, 9)]
    previous = SCOPES[(2357, 8)]
    review = scope.archive[-1]
    assert scope.archive[:-1] == previous.archive
    assert tuple(len(item.attempts) for item in scope.archive) == (2, 2)
    assert (scope.original_start_round, scope.round, scope.replay_start_round) == (5, 5, 5)
    assert (review.identity.contract, review.identity.approval) == (status.RELAY_2357_WAVE08_CONTRACT, status.RELAY_2357_WAVE08_AUTH)
    assert status.SHA.fullmatch(status.RELAY_2357_WAVE09_CONTRACT)
    assert status.SHA.fullmatch(status.RELAY_2357_WAVE09_AUTH)
    assert status.RELAY_2357_WAVE09_OBSERVATION == review.progress
    assert status.RELAY_2357_WAVE08_TERMINALS == tuple(attempt.terminal_file_sha256 for attempt in review.attempts)
    assert status.RELAY_2357_WAVE08_REPLAY_FILES == review.replay_files
    assert status.RELAY_2357_WAVE08_FITTED_FILES == dict(previous.fitted_prefix_files)
    assert status.RELAY_12203_WAVE12_SOURCES == dict(scope.required_sources)
    assert status.RELAY_2357_WAVE09_PROOF == {
        "logical_index": 0, "payload_sha256": scope.payload_sha256, "previous_request_id": review.request_id,
        "previous_terminal_file_sha256": review.terminal_file_sha256,
        "previous_transport_progress_file_sha256": review.progress_file_sha256,
        "relay_cooldown_clock": scope.cooldown_field, "relay_not_before_unix": scope.not_before_unix,
        "newly_authorized_boundary_physical_attempts": scope.boundary_physical_attempts, "execution_budget": scope.budget,
    }


def test_6101_cached_r4_monitor_requires_both_old_receipts_and_verified_local_cuda_fit(tmp_path, monkeypatch):
    status = prompt_pilot_status
    seed = 6101
    previous = tmp_path / f"{status.SERIAL_IPV4}-6101-relay-retry-08"
    run = tmp_path / f"{status.SERIAL_IPV4}-6101-cached-r4-01"
    old_directory = previous / "seed_6101"
    current_directory = run / "seed_6101"
    archived = old_directory / "round_04"
    old_checkpoint = status.CACHED_R4_6101_PREVIOUS_CHECKPOINT
    old_profile = {"confirmation_seeds": [seed], "target_labels": 1485, "resume_boundaries": {str(seed): {"round": 3}},
                   "transport": {"logical_replies": 3, "transport_attempts": 2}, "ipv4_readiness_windows": 3,
                   "audited_future_transport_replay": True, "replay_eligible_from_round": 3, "sealed_relay_retry_seconds": 180}
    old_approval = dict.fromkeys(status.FORMAL_APPROVAL_FLAGS, True)

    def publish(directory, signed):
        approved = {**old_approval, "contract_sha256": status.record_digest(signed)}
        write_json(directory / "formal_contract.json", signed)
        write_json(directory / "authorization.json", approved)
        return approved["contract_sha256"], status.record_digest(approved)

    write_json(archived / "source.json", {"checkpoint": old_checkpoint})
    old_metrics = archived / "fit_000/validation_metrics.json"
    write_json(old_metrics, {"combined_normalized_rmse": .66})
    write_json(archived / "transition.json", {"checkpoint_sha256": old_checkpoint, "fit_path": "fit_000",
                                             "source_sha256": status.checksum(archived / "source.json"),
                                             "validation_metrics_sha256": status.checksum(old_metrics)})
    previous_sources = {"previous/workflow.py": "1" * 64}
    old_sha, old_auth = publish(previous, {"run_directory": str(previous), "profile": old_profile,
                                          "source_sha256": previous_sources})
    monkeypatch.setattr(status, "RELAY_6101_WAVE08_CONTRACT", old_sha)
    monkeypatch.setattr(status, "RELAY_6101_WAVE08_AUTH", old_auth)
    old_term = archived / "turn_00/attempt_00.terminal.json"
    correction = archived / "turn_01/attempt_00.terminal.json"
    decision = archived / "turn_00/decision.json"
    write_json(old_term, {"status": "completed", "provider_dispatched": True})
    write_json(correction, {"status": "stopped", "provider_dispatched": True})
    write_json(decision, {"state": "rejected"})
    write_json(old_directory / "pause.request", {"retired": True})
    plan = {"choices": [{"candidate_id": f"C-{i}", "reason": "original"} for i in range(32)]}
    batch = [f"row_{i}" for i in range(32)]
    review = {**status.CACHED_R4_6101_REVIEW, "archived_terminal_file_sha256": status.checksum(old_term),
              "archived_decision_file_sha256": status.checksum(decision), "normalized_plan_sha256": status.record_digest(plan),
              "frozen_order_sha256": status.record_digest(batch), "previous_seed_pause_file_sha256": status.checksum(old_directory / "pause.request")}
    correction_review = {**status.CACHED_R4_6101_FAILED_CORRECTION, "terminal_file_sha256": status.checksum(correction)}
    boundaries = {"round_04/source.json": status.checksum(archived / "source.json")}
    mirrors = [f"/choices/{i}/references/0/observed_id" for i in range(42)]
    monkeypatch.setattr(status, "CACHED_R4_6101_REVIEW", review)
    monkeypatch.setattr(status, "CACHED_R4_6101_FAILED_CORRECTION", correction_review)
    monkeypatch.setattr(status, "CACHED_R4_6101_BOUNDARY_FILES", boundaries)
    monkeypatch.setattr(status, "CACHED_R4_6101_MIRRORS_SHA", status.record_digest(mirrors))
    write_json(previous / f"bridge/outbox/{review['attempt_id']}.json", {"job_sha256": review["job_sha256"]})
    write_json(previous / f"bridge/results/{review['attempt_id']}.json", {
        "provider_dispatched": True, "result_sha256": review["result_sha256"], "raw_response_sha256": review["raw_response_sha256"],
    })
    correction_result = previous / f"bridge/results/{correction_review['attempt_id']}.json"
    write_json(correction_result, {"result_sha256": correction_review["result_sha256"], "raw_response_sha256": None})
    custody = {**review, "archived_boundary_files": boundaries, "removed_paths": mirrors, "failed_correction": correction_review}
    module = f"studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/{status.CACHED_R4_6101_MODULE}.py"
    child_profile = {"confirmation_seeds": [seed], "target_labels": 493, "resume_boundaries": {str(seed): {"round": 4}},
                     "transport": {"logical_replies": 1, "transport_attempts": 1}, "ipv4_readiness_windows": 3}
    proof = {"dispatcher_module": status.CACHED_R4_6101_MODULE, "seed": seed, "round": 4, "recovery_budget": {
        "maximum_logical_requests": 1, "maximum_physical_attempts": 0,
    }, "execution_budget": status.CACHED_R4_6101_BUDGET, "newly_authorized_boundary_physical_attempts": 0,
             "checkpoint_sha256": old_checkpoint, "previous_run_directory": str(previous), "previous_contract_sha256": old_sha,
             "previous_authorization_sha256": old_auth, "cached_selection": custody}
    sha, auth = publish(run, {"run_directory": str(run), "profile": child_profile, "predecessor": proof,
                               "budget": status.CACHED_R4_6101_BUDGET,
                               "source_sha256": {**previous_sources, module: status.CACHED_R4_6101_SOURCE}})
    monkeypatch.setattr(status, "CACHED_R4_6101_CONTRACT", sha)
    monkeypatch.setattr(status, "CACHED_R4_6101_AUTH", auth)
    write_json(run / "cached_response_import.json", custody)
    status_row = lambda: status.cached_r4_6101_contract(tmp_path, seed, previous, old_sha)
    assert status_row()[:3] == (run, sha, 4)
    assert status_row()[3]["fitted_labels"] is None
    assert status.dispatcher_pid(seed, "ipv4_cached_r4_6101") is None

    stage = current_directory / "round_04"
    turn = stage / "turn_00"
    fitted = current_directory / "round_05"
    fit = fitted / "fit_000"
    fit.mkdir(parents=True)
    (fit / "best.pt").write_bytes(b"checkpoint bytes")
    checkpoint = status.checksum(fit / "best.pt")
    monkeypatch.setattr(status, "CACHED_R4_6101_FITTED_CHECKPOINT", checkpoint)
    formal_run = {"contract_sha256": sha}
    write_json(turn / "attempt_00.terminal.json", {"attempt_id": "a" * 32, "status": "completed", "provider_dispatched": False, "usage": {},
               "cached_selection_sha256": status.record_digest(custody), "response_id": review["response_id"],
               "imported_from_response_id": review["response_id"], "request_id": "local-request"})
    write_json(turn / "decision.json", {"state": "legal"})
    write_json(stage / "selection.json", plan)
    write_json(stage / "memory.json", {})
    write_json(stage / "batch_freeze.json", {"mock": False, "label_access": False, "batch_ids": batch,
              "binding": {"a7_formal": formal_run, "seed": seed, "round": 4, "checkpoint_sha256": old_checkpoint},
              "accepted_logical_index": 0, "accepted_attempt": 0, "request_id": "local-request", "provider_response_id": review["response_id"],
              "selection_file_sha256": status.checksum(stage / "selection.json"), "memory_file_sha256": status.checksum(stage / "memory.json")})
    freeze_sha = status.checksum(stage / "batch_freeze.json")
    grant = {"scope": "reveal_and_fit_one_frozen_batch", "formal_run": formal_run, "device": "cuda", "selection_freeze_sha256": freeze_sha}
    write_json(fitted / "fit_authorization.json", grant)
    write_json(fitted / "authorized_feedback.json", {"formal_grant_sha256": status.record_digest(grant),
                                                     "selection_freeze_sha256": freeze_sha,
                                                     "records": [{"candidate_id": i} for i in batch]})
    write_json(fitted / "source.json", {"checkpoint": checkpoint, "round": 5})
    write_json(fit / "fit_audit.json", {"train_rows": 493, "checkpoint_sha256": checkpoint})
    write_json(fit / "fit_artifact_receipt.json", {"best.pt": checkpoint})
    write_json(fit / "validation_metrics.json", {"checkpoint_sha256": checkpoint, "combined_normalized_rmse": .46})
    write_json(fitted / "transition.json", {"formal_run": formal_run, "selection_freeze_sha256": freeze_sha, "fit_path": "fit_000",
               "memory_file_sha256": status.checksum(stage / "memory.json"), "checkpoint_sha256": checkpoint,
               "source_sha256": status.checksum(fitted / "source.json"), "feedback_sha256": status.checksum(fitted / "authorized_feedback.json"),
               "fit_audit_sha256": status.checksum(fit / "fit_audit.json"), "fit_artifact_receipt_sha256": status.checksum(fit / "fit_artifact_receipt.json"),
               "validation_metrics_sha256": status.checksum(fit / "validation_metrics.json")})
    monkeypatch.setattr(status, "CACHED_R4_6101_LOCAL_TERMINAL", status.checksum(turn / "attempt_00.terminal.json"))
    monkeypatch.setattr(status, "CACHED_R4_6101_TRANSITION", status.checksum(fitted / "transition.json"))
    assert status_row()[3] == {"round": 4, "fitted_labels": 493, "choice_count": 32, "removed_reference_fields": 42,
                               "provider_dispatched": False, "archived_failed_correction": "connection_interrupted",
                               "checkpoint_sha256": checkpoint, "validation_nrmse": .46}
    check_6101_r5_handoff(tmp_path, monkeypatch, publish, previous, run, sha, auth, status_row()[3])
    (fit / "best.pt").write_bytes(b"different checkpoint bytes")
    assert status_row()[3]["fitted_labels"] is None
    correction_result.unlink()
    assert status_row() is None


def check_6101_r5_handoff(runs, monkeypatch, publish, parent, cached, cached_sha, cached_auth, imported):
    status = prompt_pilot_status
    run = runs / f"{status.SERIAL_IPV4}-6101-r5-continuation-01"
    stage = cached / "seed_6101/round_04"
    following = cached / "seed_6101/round_05"
    transition = following / "transition.json"
    profile = status.read_json(parent / "formal_contract.json")["profile"]
    child = status.read_json(cached / "formal_contract.json")
    custody = child["predecessor"]["cached_selection"]
    handoff = {"cached_selection_file_sha256": status.checksum(cached / "cached_response_import.json"),
               "extension_source_sha256": status.checksum(following / "source.json"),
               "extension_fit_audit_sha256": status.checksum(following / "fit_000/fit_audit.json"),
               "extension_freeze_sha256": status.checksum(stage / "batch_freeze.json")}
    monkeypatch.setattr(status, "CONTINUATION_6101_R5_HANDOFF", handoff)
    write_json(cached / "workers/seed_6101.json", {"status": "complete", "completed_rounds": 5, "active_labels": 493})
    proof = {**handoff, "dispatcher_module": "dispatch_v13_6101_r5_continuation", "seed": 6101, "round": 5,
             "parent_run_directory": str(parent), "parent_contract_sha256": status.RELAY_6101_WAVE08_CONTRACT,
             "parent_authorization_sha256": status.RELAY_6101_WAVE08_AUTH,
             "parent_seed_pause_sha256": status.CACHED_R4_6101_REVIEW["previous_seed_pause_file_sha256"],
             "extension_run_directory": str(cached), "extension_contract_sha256": cached_sha,
             "extension_authorization_sha256": cached_auth, "cached_local_attempt_id": "a" * 32,
             "cached_local_terminal_sha256": status.CACHED_R4_6101_LOCAL_TERMINAL,
             "cached_selection_sha256": status.record_digest(custody), "reviewed_cached_selection": custody,
             "extension_transition_sha256": status.checksum(transition), "checkpoint_sha256": imported["checkpoint_sha256"],
             "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 2},
             "execution_budget": status.CONTINUATION_6101_R5_BUDGET}
    boundary = {"round": 5, "state_sha256": status.record_digest(status.read_json(following / "source.json")),
                "memory_sha256": status.record_digest(status.read_json(stage / "memory.json"))}
    record = {"run_directory": str(run), "profile": {**profile, "resume_boundaries": {"6101": boundary},
                                                      "replay_eligible_from_round": 5},
              "source_sha256": {**child["source_sha256"], **status.CONTINUATION_6101_R5_SOURCES},
              "predecessor": proof, "budget": status.CONTINUATION_6101_R5_BUDGET}
    sha, auth = publish(run, record)
    monkeypatch.setattr(status, "CONTINUATION_6101_R5_CONTRACT", sha)
    monkeypatch.setattr(status, "CONTINUATION_6101_R5_AUTH", auth)
    assert status.r5_after_6101_cached_r4(runs, cached, cached_sha, imported) == (run, sha, 5, 3)
    assert status.r5_after_6101_cached_r4(runs, cached, cached_sha, {**imported, "fitted_labels": None}) is None
    commands = []
    monkeypatch.setattr(status, "command_lines", lambda command: commands.append(command) or [])
    status.dispatcher_pid(6101, "ipv4_6101_r5_continuation")
    assert commands[0][-1].endswith("/com.purellm.a7-ipv4-v13-6101-r5-continuation-01")
    original = (stage / "batch_freeze.json").read_bytes()
    write_json(stage / "batch_freeze.json", {"changed": True})
    assert status.r5_after_6101_cached_r4(runs, cached, cached_sha, imported) is None
    (stage / "batch_freeze.json").write_bytes(original)


@pytest.mark.parametrize("seed,wave", [(887, 8), (887, 9), (157, 11), (157, 12), (2357, 10), (2357, 11), (2357, 12),
                                       (2357, 13), (2357, 14), (2357, 15), (6101, 11), (6101, 12), (6101, 13),
                                       (6101, 14), (157, 13), (157, 14), (12203, 17), (12203, 18), (12203, 19),
                                       (12203, 20), (12203, 21), (12203, 22), (12203, 23), (157, 15), (157, 16),
                                       (2357, 16), (2357, 17), (2357, 18), (6101, 15), (6101, 16), (6101, 17),
                                       (157, 17), (157, 18), (157, 19), (2357, 19), (6101, 18), (6101, 19),
                                       (12203, 24), (12203, 25), (12203, 26), (157, 20),
                                       (2357, 20), (6101, 20), (12203, 27), (157, 21), (2357, 21), (6101, 21), (12203, 28),
                                       (2357, 22), (6101, 22), (157, 22), (12203, 29), (2357, 23), (6101, 23), (157, 23), (12203, 30),
                                       (157, 24), (2357, 24), (6101, 24), (12203, 31),
                                       (157, 25), (2357, 25), (6101, 25), (12203, 32), (2357, 26), (6101, 26), (157, 26), (12203, 33),
                                       (157, 27), (2357, 27), (6101, 27), (12203, 34),
                                       (157, 28), (887, 13), (2357, 28), (6101, 28), (12203, 35),
                                       (157, 29), (157, 30), (887, 14), (887, 15), (2357, 29), (2357, 30), (12203, 36), (12203, 37), (2357, 31), (157, 31),
                                       (157, 32), (157, 33), (157, 34), (157, 35), (157, 36)])
def test_new_exact_reviewed_relay_pins_the_original_retired_receipt(seed, wave):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.run_v13_retry_successor import SCOPES

    status = prompt_pilot_status
    scope = SCOPES[(seed, wave)]
    review = scope.archive[-1]
    prefix = f"RELAY_{seed}_WAVE{wave:02d}"
    old = f"RELAY_{seed}_WAVE{wave - 1:02d}"
    assert (getattr(status, f"{old}_CONTRACT"), getattr(status, f"{old}_AUTH")) == (review.identity.contract, review.identity.approval)
    assert status.SHA.fullmatch(getattr(status, f"{prefix}_CONTRACT"))
    assert status.SHA.fullmatch(getattr(status, f"{prefix}_AUTH"))
    assert getattr(status, f"{prefix}_OBSERVATION") == review.progress
    assert getattr(status, f"{prefix}_PROOF") == {
        "logical_index": 0, "payload_sha256": scope.payload_sha256, "previous_request_id": review.request_id,
        "previous_terminal_file_sha256": review.terminal_file_sha256,
        "previous_transport_progress_file_sha256": review.progress_file_sha256,
        "relay_cooldown_clock": scope.cooldown_field, "relay_not_before_unix": scope.not_before_unix,
        "newly_authorized_boundary_physical_attempts": scope.boundary_physical_attempts, "execution_budget": scope.budget,
    }
    attempts = tuple({"attempt": index, "attempt_id": item.attempt_id, "job_sha256": item.job_sha256,
                      "result_sha256": item.result_sha256, "error": item.failure,
                      "provider_dispatched": True, "ambiguous": True} for index, item in enumerate(review.attempts))
    if len(attempts) == 1:
        assert getattr(status, f"{old}_ATTEMPT") == attempts[0]
    else:
        receipt_pin = f"{prefix}_PARENT" if (seed, wave) == (887, 13) else old
        assert getattr(status, f"{receipt_pin}_ATTEMPTS") == attempts
        assert getattr(status, f"{receipt_pin}_TERMINALS") == tuple(item.terminal_file_sha256 for item in review.attempts)
        assert getattr(status, f"{receipt_pin}_REPLAY_FILES") == review.replay_files
        assert getattr(status, f"{prefix}_ARCHIVED_SSE_COUNTS") == tuple(
            item.progress.get("sse_event_count", 0) for item in review.attempts
        )
    sources = getattr(status, f"{prefix}_SOURCES")
    assert {name: sources[name] for name, _ in scope.required_sources} == dict(scope.required_sources)
    if (seed, wave) == (887, 13):
        previous = SCOPES[(887, 12)].archive[-1]
        assert status.RELAY_887_WAVE12_TERMINALS == tuple(item.terminal_file_sha256 for item in previous.attempts)
        assert status.RELAY_887_WAVE12_REPLAY_FILES == previous.replay_files
        assert status.RELAY_887_WAVE13_FITTED_FILES == dict(scope.fitted_prefix_files)
    elif seed == 887 and wave in (14, 15):
        assert getattr(status, f"{prefix}_FITTED_FILES") == dict(scope.fitted_prefix_files)


@pytest.mark.parametrize("parent,seed,round_index", [
    ("887-relay-retry-21", 887, 34), ("887-relay-retry-22", 887, 34),
    ("2357-relay-retry-37", 2357, 5), ("6101-r11-continuation-01", 6101, 13), ("6101-relay-retry-29", 6101, 13),
    ("157-relay-retry-38", 157, 7),
    ("887-relay-retry-23", 887, 34),
    ("887-relay-retry-24", 887, 34),
    ("2357-relay-retry-38", 2357, 5),
    ("2357-relay-retry-39", 2357, 5),
    ("6101-relay-retry-30", 6101, 13),
    ("12203-relay-retry-44", 12203, 5),
    ("12203-relay-retry-45", 12203, 5),
])
def test_exact_residual_discovery_binds_producer_and_latest_receipt(monkeypatch, tmp_path, parent, seed, round_index):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.run_v13_retry_successor import SCOPES

    status = prompt_pilot_status
    pin = status.EXACT_REVIEWED_RELAYS[parent, seed]
    scope = SCOPES[seed, pin["wave"]]
    review = scope.archive[-1]
    assert pin["round"] == scope.round == round_index
    assert pin.get("physical_limits", (2, 1)) == (scope.original_boundary_physical_attempts,
                                                  scope.boundary_physical_attempts)
    assert pin["parent"] == (review.identity.contract, review.identity.approval)
    assert status.SHA.fullmatch(pin["current"][0]) and status.SHA.fullmatch(pin["current"][1])
    assert pin["terminal_files"] == tuple((attempt.terminal_file_sha256, attempt.failure) for attempt in review.attempts)
    assert pin["observation_sha256"] == status.record_digest(review.progress)
    assert pin.get("preflight_attempts", ()) == tuple(
        index for index, attempt in enumerate(review.attempts) if attempt.receipt_provider_flags == (False, False)
    )
    if "replay_files" in pin:
        assert pin["replay_files"] == review.replay_files
    previous = tmp_path / f"{status.SERIAL_IPV4}-{parent}"
    assert status._exact_reviewed_successor(tmp_path, previous, "0" * 64, seed) is None
    monkeypatch.setattr(status, "_signed_retry_record", lambda *args: None)
    assert status._exact_reviewed_successor(tmp_path, previous, pin["parent"][0], seed) is None


def test_pinned_preflight_rejects_tampered_terminal_or_authenticated_submission(tmp_path, monkeypatch):
    status = prompt_pilot_status
    previous = tmp_path.resolve() / "parent"
    directory = previous / "seed_12203"
    turn = directory / "round_05/turn_00"
    (turn).mkdir(parents=True)
    (directory / "pause.request").touch()
    write_json(previous / "workers/seed_12203.json", {"status": "failed", "pid": 4321})
    write_json(previous / "coordinator_generation.json", {"pid": 4321})
    monkeypatch.setattr(status, "live", lambda _pid: False)

    attempt, request_id = "a" * 32, "b" * 64
    payload = {"input": "exact"}
    binding = {key: "3" * 64 for key in ("checkpoint_sha256", "packet_sha256", "wire_packet_sha256", "base_messages_sha256")}
    request = {"request_id": request_id, "logical_index": 0, "payload": payload, "payload_sha256": status.record_digest(payload),
               "binding": binding}
    write_json(turn / "request.json", request)
    write_json(turn / "attempt_00.started.json", {"attempt": 0, "attempt_id": attempt})
    job = {"attempt_id": attempt, "request_id": request_id, "request": request}
    job["job_sha256"] = status.record_digest(job)
    write_json(previous / "bridge/outbox" / f"{attempt}.json", job)
    result = {"attempt_id": attempt, "request_id": request_id, "job_sha256": job["job_sha256"], "provider_dispatched": False,
              "raw_response": None, "outcome": {"status": "failed", "failure": {"code": "connect_error", "ambiguous": False,
                                                                              "retryable": True, "reply_identity": None}}}
    result["result_sha256"] = status.record_digest(result)
    result_path = previous / "bridge/results" / f"{attempt}.json"
    write_json(result_path, result)
    terminal = {"attempt": 0, "attempt_id": attempt, "request_id": request_id, "status": "stopped", "error": "connect_error",
                "provider_dispatched": False, "ambiguous": False, "not_before_unix": 100.0}
    terminal_path = turn / "attempt_00.terminal.json"
    write_json(terminal_path, terminal)
    proof = {**binding, "previous_request_id": request_id, "payload_sha256": request["payload_sha256"],
             "relay_cooldown_clock": "not_before_unix", "relay_not_before_unix": 280.0,
             "previous_terminal_file_sha256": status.checksum(terminal_path), "previous_attempts": [
                 {"attempt": 0, "attempt_id": attempt, "error": "connect_error", "provider_dispatched": False, "ambiguous": False,
                  "job_sha256": job["job_sha256"], "result_sha256": result["result_sha256"],
                  "terminal_sha256": status.record_digest(terminal)}]}
    exact = {"terminal_files": ((status.checksum(terminal_path), "connect_error"),), "preflight_attempts": (0,),
             "request_file_sha256": status.checksum(turn / "request.json")}
    assert status._pinned_relay_evidence(previous, 12203, 5, proof, exact)
    write_json(terminal_path, {**terminal, "provider_dispatched": True})
    assert not status._pinned_relay_evidence(previous, 12203, 5, proof, exact)
    write_json(terminal_path, terminal)
    write_json(result_path, {**result, "provider_dispatched": True})
    assert not status._pinned_relay_evidence(previous, 12203, 5, proof, exact)


def test_157_wave36_accepts_actual_single_attempt_in_wave34_lineage(tmp_path):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.run_v13_retry_successor import SCOPES

    old_chain = [{
        "run_directory": str(tmp_path / review.identity.name),
        "contract_sha256": review.identity.contract,
        "authorization_sha256": review.identity.approval,
        "previous_attempts": [{"attempt_id": attempt.attempt_id} for attempt in review.attempts],
    } for review in SCOPES[157, 36].archive[:-1]]
    old = {"reviewed_retry_chain": old_chain}
    assert prompt_pilot_status._reviewed_relay_prefix(tmp_path, 157, 36, old) == old_chain
    assert len(old_chain[-1]["previous_attempts"]) == 1
    old_chain[-1]["previous_attempts"].append({"attempt_id": "unexpected"})
    assert prompt_pilot_status._reviewed_relay_prefix(tmp_path, 157, 36, old) is None


def test_157_wave35_pins_one_ambiguous_http_header_timeout():
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.run_v13_retry_successor import SCOPES

    scope = SCOPES[157, 35]
    assert scope.round == 7
    assert scope.original_boundary_physical_attempts == 2
    assert scope.boundary_physical_attempts == 1
    assert scope.replay_start_round == 8
    assert scope.budget["maximum_physical_attempts"] == 169
    assert scope.not_before_unix == 1795744861.3835022
    assert prompt_pilot_status.RELAY_157_WAVE34_ATTEMPT["error"] == "timeout"
    observation = prompt_pilot_status.RELAY_157_WAVE35_OBSERVATION
    assert observation["http_status"] == 200
    assert observation["error_type"] == "sse_inactivity_timeout"
    assert observation["first_phase_seconds"]["http_headers"] == 35.38578833406791
    assert not {"response_id", "provider_error_code", "conflicting_event_type", "sse_event_count"} & observation.keys()


def test_887_r13_targeted_repair_requires_exact_retired_rejection_and_one_slot(tmp_path, monkeypatch):
    status = prompt_pilot_status
    previous = tmp_path / f"{status.SERIAL_IPV4}-887-relay-retry-10"
    run = tmp_path / status.REPAIR_887_R13_RUN
    stage = previous / "seed_887/round_13"
    write_json(stage / "turn_00/attempt_00.started.json", {"attempt": 0})
    files = {}
    for rel in status.REPAIR_887_R13_ORIGINAL_RESPONSE_CUSTODY:
        path = previous / rel
        write_json(path, {"fixture": rel})
        files[rel] = status.checksum(path)
    monkeypatch.setattr(status, "REPAIR_887_R13_ORIGINAL_RESPONSE_CUSTODY", files)

    def publish(directory, signed):
        sha = status.record_digest(signed)
        approval = {"contract_sha256": sha, **dict.fromkeys(status.FORMAL_APPROVAL_FLAGS, True)}
        write_json(directory / "formal_contract.json", signed)
        write_json(directory / "authorization.json", approval)
        return sha, status.record_digest(approval)

    boundary = {"887": {"round": 13}}
    old, old_auth = publish(previous, {"run_directory": str(previous), "profile": {"resume_boundaries": boundary}})
    monkeypatch.setattr(status, "RELAY_887_WAVE10_CONTRACT", old)
    monkeypatch.setattr(status, "RELAY_887_WAVE10_AUTH", old_auth)
    metadata = {"repair_index": 11}
    metadata_sha = status.record_digest(metadata)
    monkeypatch.setattr(status, "REPAIR_887_R13_METADATA_SHA", metadata_sha)
    write_json(run / "fixed_reference_repair.json", metadata)
    budget = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    signed = {"run_directory": str(run), "source_sha256": status.REPAIR_887_R13_SOURCES, "budget": budget,
              "profile": {"confirmation_seeds": [887], "target_labels": 781, "resume_boundaries": boundary,
                          "ipv4_readiness_windows": 3, "fixed_reference_repair": metadata,
                          "transport": {"logical_replies": 1, "transport_attempts": 1}},
              "predecessor": {"dispatcher_module": "dispatch_v13_887_r13_targeted_repair", "seed": 887, "round": 13,
                              "previous_run_directory": str(previous), "previous_contract_sha256": old,
                              "previous_authorization_sha256": old_auth, "prior_response_reposted": False,
                              "logical_index": 0, "fixed_reference_repair_sha256": metadata_sha,
                              "execution_budget": budget, "recovery_budget": {"maximum_logical_requests": 1,
                                                                            "maximum_physical_attempts": 1}}}
    sha, auth = publish(run, signed)
    monkeypatch.setattr(status, "REPAIR_887_R13_CONTRACT", sha)
    monkeypatch.setattr(status, "REPAIR_887_R13_AUTH", auth)
    assert status.r13_targeted_repair(tmp_path, previous, old) == (run, sha, 13, 3)
    decision = stage / "turn_00/decision.json"
    original = decision.read_bytes()
    write_json(decision, {"modified": True})
    assert status.r13_targeted_repair(tmp_path, previous, old) is None
    decision.write_bytes(original)
    changed = {**signed, "profile": {**signed["profile"], "transport": {"logical_replies": 1, "transport_attempts": 2}}}
    sha, auth = publish(run, changed)
    monkeypatch.setattr(status, "REPAIR_887_R13_CONTRACT", sha)
    monkeypatch.setattr(status, "REPAIR_887_R13_AUTH", auth)
    assert status.r13_targeted_repair(tmp_path, previous, old) is None


def test_887_r13_cached_repair_requires_real_l781_and_zero_bridge_posts(tmp_path, monkeypatch):
    status = prompt_pilot_status
    paid = tmp_path / status.REPAIR_887_R13_RUN
    run = tmp_path / status.CACHED_887_R13_RUN
    boundary = {"887": {"round": 13}}
    paid_profile = {"confirmation_seeds": [887], "target_labels": 781, "resume_boundaries": boundary}

    def publish(directory, signed):
        sha = status.record_digest(signed)
        approval = {"contract_sha256": sha, **dict.fromkeys(status.FORMAL_APPROVAL_FLAGS, True)}
        write_json(directory / "formal_contract.json", signed)
        write_json(directory / "authorization.json", approval)
        return sha, status.record_digest(approval)

    paid_sha, paid_auth = publish(paid, {"run_directory": str(paid), "profile": paid_profile})
    monkeypatch.setattr(status, "REPAIR_887_R13_CONTRACT", paid_sha)
    monkeypatch.setattr(status, "REPAIR_887_R13_AUTH", paid_auth)
    for key, directory in (("CACHED_887_R13_PAID_FILES", paid), ("CACHED_887_R13_FILES", run)):
        for name in getattr(status, key):
            write_json(directory / name, {"fixture": name})
    custody = {"provider_post": False, "projected_plan_sha256": status.CACHED_887_R13_PROJECTED}
    monkeypatch.setattr(status, "CACHED_887_R13_CUSTODY_SHA", status.record_digest(custody))
    write_json(run / "cached_correction.json", custody)
    write_json(run / "seed_887/round_13/turn_00/attempt_00.terminal.json",
               {"status": "completed", "provider_dispatched": False, "usage": {}, "attempt_id": status.CACHED_887_R13_ATTEMPT})
    for key, directory in (("CACHED_887_R13_PAID_FILES", paid), ("CACHED_887_R13_FILES", run)):
        monkeypatch.setattr(status, key, {name: status.checksum(directory / name) for name in getattr(status, key)})
    budget = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 0}
    signed = {"run_directory": str(run), "profile": paid_profile, "budget": budget,
              "predecessor": {"dispatcher_module": "run_v13_887_r13_cached_correction", "seed": 887, "round": 13,
                              "previous_run_directory": str(paid), "previous_contract_sha256": paid_sha,
                              "previous_authorization_sha256": paid_auth,
                              "execution_budget": budget, "cached_correction": custody,
                              "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 0}}}
    sha, auth = publish(run, signed)
    monkeypatch.setattr(status, "CACHED_887_R13_CONTRACT", sha)
    monkeypatch.setattr(status, "CACHED_887_R13_AUTH", auth)
    assert status.r13_cached_correction(tmp_path, paid, paid_sha) == (run, sha, 13, 0)
    outbox = run / "bridge/outbox" / "unexpected.json"
    write_json(outbox, {})
    assert status.r13_cached_correction(tmp_path, paid, paid_sha) is None
    outbox.unlink()
    (run / "seed_887/round_14/fit_000/best.pt").unlink()
    assert status.r13_cached_correction(tmp_path, paid, paid_sha) is None


def test_887_r20_monitor_requires_exact_cached_handoff_and_new_parser(tmp_path, monkeypatch):
    status = prompt_pilot_status
    previous = tmp_path / status.CACHED_887_R19_RUN
    run = tmp_path / status.CONTINUATION_887_R20_RUN
    h = dict(status.CACHED_887_R19_HANDOFF)
    custody = {"provider_post": False}
    h["cached_selection_sha256"] = status.record_digest(custody)
    monkeypatch.setattr(status, "CACHED_887_R19_HANDOFF", h)
    parser_path = "studies/active_learning/pure_llm_v3_20261124/pure_llm_v3/transport_manager.py"
    helper = "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_887_r19_cached.py"
    sources = {**status.CONTINUATION_887_R20_SOURCES, parser_path: status.CONTINUATION_887_R20_PARSER,
               helper: status.CONTINUATION_887_R20_CACHED_HELPER}
    proof = {"dispatcher_module": "dispatch_v13_887_r20_continuation", "seed": 887, "round": 20,
             "parent_run_directory": str(tmp_path / status.CONTINUATION_887_R14_RUN),
             "parent_contract_sha256": status.CONTINUATION_887_R14_CONTRACT,
             "parent_authorization_sha256": status.CONTINUATION_887_R14_AUTH, "extension_run_directory": str(previous),
             "checkpoint_sha256": h["checkpoint_sha256"], "r19_handoff": h, "cached_selection": custody,
             "cached_response_import_file_sha256": status.CACHED_887_R19_FILES["cached_response_import.json"],
             "r20_request": status.CONTINUATION_887_R20_REQUEST, "execution_budget": status.CONTINUATION_887_R20_BUDGET,
             "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 2}}
    boundary = {"round": 20, "state_sha256": h["state_sha256"], "memory_sha256": h["scientific_memory_sha256"]}
    profile = {"resume_boundaries": {"887": boundary}, "target_labels": 1485, "confirmation_seeds": [887],
               "replay_eligible_from_round": 20, "audited_future_transport_replay": True,
               "sealed_relay_retry_seconds": 180, "ipv4_readiness_windows": 3}
    signed = {"run_directory": str(run), "budget": status.CONTINUATION_887_R20_BUDGET,
              "profile": profile, "source_sha256": sources, "predecessor": proof}

    def publish(record):
        sha = status.record_digest(record)
        approval = {"contract_sha256": sha, **dict.fromkeys(status.FORMAL_APPROVAL_FLAGS, True)}
        write_json(run / "formal_contract.json", record)
        write_json(run / "authorization.json", approval)
        monkeypatch.setattr(status, "CONTINUATION_887_R20_CONTRACT", sha)
        monkeypatch.setattr(status, "CONTINUATION_887_R20_AUTH", status.record_digest(approval))
        return sha

    sha = publish(signed)
    assert status.r20_after_887_cached_correction(tmp_path, previous, h["contract_sha256"]) == (run, sha, 20, 3)
    wrong_parser = {**signed, "source_sha256": {**sources, parser_path: "f" * 64}}
    publish(wrong_parser)
    assert status.r20_after_887_cached_correction(tmp_path, previous, h["contract_sha256"]) is None
    wrong_handoff = {**signed, "predecessor": {**proof, "r19_handoff": {**h, "provider_post": True}}}
    publish(wrong_handoff)
    assert status.r20_after_887_cached_correction(tmp_path, previous, h["contract_sha256"]) is None


def test_887_r32_monitor_requires_exact_l1357_and_one_retired_attempt(tmp_path, monkeypatch):
    status = prompt_pilot_status
    previous = tmp_path / status.CONTINUATION_887_R20_RUN
    run = tmp_path / f"{status.SERIAL_IPV4}-887-relay-retry-11"
    seed = previous / "seed_887"
    current = seed / "round_32"
    turn = current / "turn_00"
    old_budget = status.CONTINUATION_887_R20_BUDGET
    budget = status.RELAY_887_WAVE11_PROOF["execution_budget"]
    descriptor = status.RELAY_887_WAVE11_DESCRIPTOR

    def publish(path, record):
        contract = status.record_digest(record)
        approval = {"contract_sha256": contract, **dict.fromkeys(status.FORMAL_APPROVAL_FLAGS, True)}
        write_json(path / "formal_contract.json", record)
        write_json(path / "authorization.json", approval)
        return contract, status.record_digest(approval)

    profile = {"resume_boundaries": {"887": {"round": 20}},
               "replay_eligible_from_round": 20, "audited_future_transport_replay": True,
               "sealed_relay_retry_seconds": 180, "ipv4_readiness_windows": 3, "sse_inactivity_seconds": 900,
               "confirmation_seeds": [887], "target_labels": 1485, "request_timeouts": {"total_seconds": 3600}}
    old, old_auth = publish(previous, {"run_directory": str(previous), "profile": profile, "budget": old_budget,
                                      "predecessor": {"recovery_budget": {"maximum_logical_requests": 1,
                                                                         "maximum_physical_attempts": 2}}})
    monkeypatch.setattr(status, "CONTINUATION_887_R20_CONTRACT", old)
    monkeypatch.setattr(status, "CONTINUATION_887_R20_AUTH", old_auth)
    for name in status.RELAY_887_WAVE11_FITTED_FILES:
        write_json(seed / name, {"file": name})
    metrics = current / "fit_000/validation_metrics.json"
    write_json(current / "transition.json", {
        "fit_path": "fit_000", "checkpoint_sha256": status.RELAY_887_WAVE11_CHECKPOINT,
        "source_sha256": status.checksum(current / "source.json"), "validation_metrics_sha256": status.checksum(metrics),
    })
    files = {name: status.checksum(seed / name) for name in status.RELAY_887_WAVE11_FITTED_FILES}
    monkeypatch.setattr(status, "RELAY_887_WAVE11_FITTED_FILES", files)
    prior = status.RELAY_887_WAVE11_PREVIOUS_ATTEMPT
    terminal = {**prior, "status": "stopped"}
    write_json(turn / "attempt_00.started.json", {"attempt": 0, "attempt_id": prior["attempt_id"]})
    write_json(turn / "attempt_00.terminal.json", terminal)
    exact = {**status.RELAY_887_WAVE11_PROOF,
             "previous_terminal_file_sha256": status.checksum(turn / "attempt_00.terminal.json")}
    monkeypatch.setattr(status, "RELAY_887_WAVE11_PROOF", exact)
    attempts = [{**prior, "terminal_sha256": status.record_digest(terminal)}]
    proof = {**exact, "dispatcher_module": "dispatch_v13_retry_successor", "seed": 887, "round": 32,
             "previous_run_directory": str(previous), "previous_contract_sha256": old,
             "previous_authorization_sha256": old_auth, "recovery_budget": {"maximum_logical_requests": 1,
                                                                            "maximum_physical_attempts": 1},
             "previous_attempts": attempts, "checkpoint_sha256": status.RELAY_887_WAVE11_CHECKPOINT,
             "previous_transport_observation": status.RELAY_887_WAVE11_OBSERVATION,
             "reviewed_fitted_prefix_files": files,
             "reviewed_retry_chain": [status._relay_review(previous, old, old_auth, exact, attempts)]}
    signed = {"run_directory": str(run), "budget": budget, "source_sha256": status.RELAY_887_WAVE11_SOURCES,
              "predecessor": proof, "profile": {**profile, "replay_eligible_from_round": 33,
                                                "resume_boundaries": {"887": descriptor}}}
    contract, auth = publish(run, signed)
    monkeypatch.setattr(status, "RELAY_887_WAVE11_CONTRACT", contract)
    monkeypatch.setattr(status, "RELAY_887_WAVE11_AUTH", auth)
    assert status.reviewed_relay_contract(tmp_path, 887, 11, previous, old) is None
    (seed / "pause.request").touch()
    result = run, contract, 32, 3, exact["relay_not_before_unix"]
    assert status.reviewed_relay_contract(tmp_path, 887, 11, previous, old) == result

    for unexpected in ("round_33/source.json", "round_32/batch_freeze.json", "round_32/turn_00/attempt_01.started.json"):
        path = seed / unexpected
        write_json(path, {})
        assert status.reviewed_relay_contract(tmp_path, 887, 11, previous, old) is None
        path.unlink()
        if unexpected.startswith("round_33/"):
            path.parent.rmdir()
    assert status.reviewed_relay_contract(tmp_path, 887, 11, previous, old) == result


def test_887_r32_targeted_monitor_requires_a_verified_l1389_fit(tmp_path, monkeypatch):
    status = prompt_pilot_status
    previous = tmp_path / f"{status.SERIAL_IPV4}-887-relay-retry-11"
    run = tmp_path / status.REPAIR_887_R32_RUN
    descriptor = {"887": status.RELAY_887_WAVE11_DESCRIPTOR}

    def publish(directory, signed):
        contract = status.record_digest(signed)
        approval = {"contract_sha256": contract, **dict.fromkeys(status.FORMAL_APPROVAL_FLAGS, True)}
        write_json(directory / "formal_contract.json", signed)
        write_json(directory / "authorization.json", approval)
        return contract, status.record_digest(approval)

    old, old_auth = publish(previous, {"run_directory": str(previous), "profile": {"resume_boundaries": descriptor}})
    monkeypatch.setattr(status, "RELAY_887_WAVE11_CONTRACT", old)
    monkeypatch.setattr(status, "RELAY_887_WAVE11_AUTH", old_auth)
    write_json(previous / "seed_887/round_32/turn_00/attempt_00.started.json", {"attempt": 0})
    original = {}
    for name in status.REPAIR_887_R32_ORIGINAL:
        write_json(previous / name, {"file": name})
        original[name] = status.checksum(previous / name)
    monkeypatch.setattr(status, "REPAIR_887_R32_ORIGINAL", original)

    metadata = {"original_selection_reposted": False}
    metadata_sha = status.record_digest(metadata)
    file = run / "fixed_reference_repair.json"
    write_json(file, metadata)
    monkeypatch.setattr(status, "REPAIR_887_R32_METADATA_SHA", metadata_sha)
    monkeypatch.setattr(status, "REPAIR_887_R32_METADATA_FILE", status.checksum(file))
    budget = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    profile = {"confirmation_seeds": [887], "target_labels": 1389, "resume_boundaries": descriptor,
               "ipv4_readiness_windows": 3, "fixed_reference_repair": metadata,
               "transport": {"logical_replies": 1, "transport_attempts": 1}}
    proof = {"dispatcher_module": "dispatch_v13_887_r32_targeted_repair", "seed": 887, "round": 32,
             "previous_run_directory": str(previous), "previous_contract_sha256": old,
             "previous_authorization_sha256": old_auth, "prior_response_reposted": False, "logical_index": 0,
             "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 1},
             "execution_budget": budget, "fixed_reference_repair_sha256": metadata_sha}
    signed = {"run_directory": str(run), "budget": budget, "profile": profile, "predecessor": proof,
              "source_sha256": status.REPAIR_887_R32_SOURCES}
    sha, auth = publish(run, signed)
    monkeypatch.setattr(status, "REPAIR_887_R32_CONTRACT", sha)
    monkeypatch.setattr(status, "REPAIR_887_R32_AUTH", auth)
    expected = run, sha, 32, 3
    assert status.r32_targeted_repair(tmp_path, previous, old) == expected

    directory = run / "seed_887"
    for name in status.REPAIR_887_R32_FITTED:
        write_json(directory / name, {"file": name})
    active = directory / "round_33"
    transition = active / "transition.json"
    write_json(transition, {
        "fit_path": "fit_000", "checkpoint_sha256": status.REPAIR_887_R32_CHECKPOINT,
        "source_sha256": status.checksum(active / "source.json"),
        "validation_metrics_sha256": status.checksum(active / "fit_000/validation_metrics.json"),
    })
    assert status.r32_targeted_repair(tmp_path, previous, old) is None
    fit = {name: status.checksum(directory / name) for name in status.REPAIR_887_R32_FITTED}
    monkeypatch.setattr(status, "REPAIR_887_R32_FITTED", fit)
    assert status.r32_targeted_repair(tmp_path, previous, old) == expected
    write_json(directory / "round_32/batch_freeze.json", {"changed": True})
    assert status.r32_targeted_repair(tmp_path, previous, old) is None


def test_887_r33_monitor_pins_normal_budgets_and_exact_l1389_handoff(tmp_path, monkeypatch):
    status = prompt_pilot_status
    targeted = tmp_path / status.REPAIR_887_R32_RUN
    run = tmp_path / status.CONTINUATION_887_R33_RUN
    directory = targeted / "seed_887/round_33"
    metrics = directory / "fit_000/validation_metrics.json"
    source = directory / "source.json"
    transfer = directory / "transition.json"
    write_json(source, {"round": 33})
    write_json(metrics, {"combined_normalized_rmse": 0.4})
    write_json(transfer, {"fit_path": "fit_000", "checkpoint_sha256": status.REPAIR_887_R32_CHECKPOINT,
                          "source_sha256": status.checksum(source), "validation_metrics_sha256": status.checksum(metrics)})

    def publish(location, signed):
        contract = status.record_digest(signed)
        approval = {"contract_sha256": contract, **dict.fromkeys(status.FORMAL_APPROVAL_FLAGS, True)}
        write_json(location / "formal_contract.json", signed)
        write_json(location / "authorization.json", approval)
        return contract, status.record_digest(approval)

    old, approval = publish(targeted, {"run_directory": str(targeted)})
    monkeypatch.setattr(status, "REPAIR_887_R32_CONTRACT", old)
    monkeypatch.setattr(status, "REPAIR_887_R32_AUTH", approval)
    handoff = {"fit_files": {str(transfer.relative_to(targeted)): status.checksum(transfer)}}
    request = {"payload_sha256": "a" * 64}
    monkeypatch.setattr(status, "CONTINUATION_887_R33_HANDOFF", status.record_digest(handoff))
    monkeypatch.setattr(status, "CONTINUATION_887_R33_REQUEST", status.record_digest(request))
    signed = {
        "run_directory": str(run), "budget": status.CONTINUATION_887_R33_BUDGET,
        "source_sha256": status.CONTINUATION_887_R33_SOURCES,
        "profile": {"resume_boundaries": {"887": status.CONTINUATION_887_R33_DESCRIPTOR},
                    "confirmation_seeds": [887], "target_labels": 1485, "full_imported_boundary_round": 33,
                    "replay_eligible_from_round": 33, "sealed_relay_retry_seconds": 180,
                    "audited_future_transport_replay": True, "ipv4_readiness_windows": 3,
                    "transport": {"logical_replies": 3, "transport_attempts": 2}},
        "predecessor": {"dispatcher_module": "dispatch_v13_887_r33_continuation", "seed": 887, "round": 33,
                        "previous_run_directory": str(targeted), "previous_contract_sha256": old,
                        "previous_authorization_sha256": approval,
                        "normal_contract_sha256": status.RELAY_887_WAVE11_CONTRACT,
                        "normal_authorization_sha256": status.RELAY_887_WAVE11_AUTH,
                        "checkpoint_sha256": status.REPAIR_887_R32_CHECKPOINT,
                        "r32_handoff": handoff, "r33_request": request,
                        "recovery_budget": status.CONTINUATION_887_R33_SCOPE,
                        "execution_budget": status.CONTINUATION_887_R33_BUDGET},
    }
    sha, auth = publish(run, signed)
    monkeypatch.setattr(status, "CONTINUATION_887_R33_CONTRACT", sha)
    monkeypatch.setattr(status, "CONTINUATION_887_R33_AUTH", auth)
    assert status.r33_after_887_targeted_repair(tmp_path, targeted, old) == (run, sha, 33, 3)
    original = transfer.read_bytes()
    write_json(transfer, {"changed": True})
    assert status.r33_after_887_targeted_repair(tmp_path, targeted, old) is None
    transfer.write_bytes(original)
    signed["predecessor"]["r33_request"] = {"payload_sha256": "b" * 64}
    wrong_sha, wrong_auth = publish(run, signed)
    monkeypatch.setattr(status, "CONTINUATION_887_R33_CONTRACT", wrong_sha)
    monkeypatch.setattr(status, "CONTINUATION_887_R33_AUTH", wrong_auth)
    assert status.r33_after_887_targeted_repair(tmp_path, targeted, old) is None


def test_887_r34_monitor_pins_the_normal_parent_and_both_terminal_receipts(tmp_path, monkeypatch):
    status = prompt_pilot_status
    previous = tmp_path / status.CONTINUATION_887_R33_RUN
    run = tmp_path / f"{status.SERIAL_IPV4}-887-relay-retry-12"
    directory = previous / "seed_887"
    active = directory / "round_34"
    turn = active / "turn_00"
    old_profile = {"resume_boundaries": {"887": {"round": 33}}, "full_imported_boundary_round": 33,
                   "replay_eligible_from_round": 33, "audited_future_transport_replay": True,
                   "sealed_relay_retry_seconds": 180, "ipv4_readiness_windows": 3, "sse_inactivity_seconds": 900,
                   "confirmation_seeds": [887], "target_labels": 1485, "request_timeouts": {"total_seconds": 3600}}
    old_budget = status.CONTINUATION_887_R33_BUDGET
    old_scope = status.CONTINUATION_887_R33_SCOPE

    def publish(path, record):
        digest = status.record_digest(record)
        authorization = {"contract_sha256": digest, **dict.fromkeys(status.FORMAL_APPROVAL_FLAGS, True)}
        write_json(path / "formal_contract.json", record)
        write_json(path / "authorization.json", authorization)
        return digest, status.record_digest(authorization)

    old, old_auth = publish(previous, {"run_directory": str(previous), "profile": old_profile,
                                      "budget": old_budget, "predecessor": {"recovery_budget": old_scope}})
    monkeypatch.setattr(status, "CONTINUATION_887_R33_CONTRACT", old)
    monkeypatch.setattr(status, "CONTINUATION_887_R33_AUTH", old_auth)
    for name in status.RELAY_887_WAVE12_FITTED_FILES:
        write_json(directory / name, {"file": name})
    write_json(active / "transition.json", {
        "fit_path": "fit_000", "checkpoint_sha256": status.RELAY_887_WAVE12_CHECKPOINT,
        "source_sha256": status.checksum(active / "source.json"),
        "validation_metrics_sha256": status.checksum(active / "fit_000/validation_metrics.json"),
    })
    files = {name: status.checksum(directory / name) for name in status.RELAY_887_WAVE12_FITTED_FILES}
    monkeypatch.setattr(status, "RELAY_887_WAVE12_FITTED_FILES", files)

    receipts, terminals = [], []
    for a in status.RELAY_887_WAVE12_PREVIOUS_ATTEMPTS:
        index = a["attempt"]
        terminal = {**a, "status": "stopped"}
        path = turn / f"attempt_{index:02d}.terminal.json"
        write_json(turn / f"attempt_{index:02d}.started.json", {"attempt_id": a["attempt_id"], "attempt": index})
        write_json(path, terminal)
        terminals.append(status.checksum(path))
        receipts.append({**a, "terminal_sha256": status.record_digest(terminal)})
    monkeypatch.setattr(status, "RELAY_887_WAVE12_TERMINALS", tuple(terminals))
    for name in status.RELAY_887_WAVE12_REPLAY_FILES:
        write_json(turn / name, {"file": name})
    replay = {name: status.checksum(turn / name) for name in status.RELAY_887_WAVE12_REPLAY_FILES}
    monkeypatch.setattr(status, "RELAY_887_WAVE12_REPLAY_FILES", replay)
    known = {**status.RELAY_887_WAVE12_PROOF, "previous_terminal_file_sha256": terminals[-1]}
    monkeypatch.setattr(status, "RELAY_887_WAVE12_PROOF", known)

    profile = {k: v for k, v in old_profile.items() if k != "full_imported_boundary_round"}
    profile.update(resume_boundaries={"887": status.RELAY_887_WAVE12_DESCRIPTOR}, replay_eligible_from_round=34)
    proof = {**known, "dispatcher_module": "dispatch_v13_retry_successor", "seed": 887, "round": 34,
             "previous_run_directory": str(previous), "previous_contract_sha256": old,
             "previous_authorization_sha256": old_auth, "recovery_budget": {"maximum_logical_requests": 1,
                                                                            "maximum_physical_attempts": 2},
             "previous_attempts": receipts, "checkpoint_sha256": status.RELAY_887_WAVE12_CHECKPOINT,
             "previous_transport_observation": status.RELAY_887_WAVE12_OBSERVATION,
             "reviewed_fitted_prefix_files": files,
             "reviewed_normal_parent_budget": {"round": 33, "recovery_budget": old_scope, "execution_budget": old_budget},
             "reviewed_retry_chain": [status._relay_review(previous, old, old_auth, known, receipts)]}
    record = {"run_directory": str(run), "budget": known["execution_budget"], "source_sha256": status.RELAY_887_WAVE12_SOURCES,
              "predecessor": proof, "profile": profile}
    sha, auth = publish(run, record)
    monkeypatch.setattr(status, "RELAY_887_WAVE12_CONTRACT", sha)
    monkeypatch.setattr(status, "RELAY_887_WAVE12_AUTH", auth)
    assert status.reviewed_relay_contract(tmp_path, 887, 12, previous, old) is None
    (directory / "pause.request").touch()
    assert status.reviewed_relay_contract(tmp_path, 887, 12, previous, old) == (
        run, sha, 34, 3, known["relay_not_before_unix"],
    )
    write_json(turn / "attempt_01.replay_link.json", {"changed": True})
    assert status.reviewed_relay_contract(tmp_path, 887, 12, previous, old) is None


def test_6101_live_repair_and_normal_monitor_require_original_and_fitted_receipts(tmp_path, monkeypatch):
    status = prompt_pilot_status
    normal = tmp_path / f"{status.SERIAL_IPV4}-6101-relay-retry-28"
    targeted = tmp_path / status.REPAIR_6101_R8_RUN
    resumed = tmp_path / status.CONTINUATION_6101_R9_RUN
    normal_boundary = {"6101": {"round": 8}}
    repair_budget = {"first_logical_requests": 1, "maximum_logical_requests": 1, "maximum_physical_attempts": 1}

    def publish(directory, record):
        record = {**record, "run_directory": str(directory)}
        sha = status.record_digest(record)
        authorization = {"contract_sha256": sha, **dict.fromkeys(status.FORMAL_APPROVAL_FLAGS, True)}
        write_json(directory / "formal_contract.json", record)
        write_json(directory / "authorization.json", authorization)
        return sha, status.record_digest(authorization)

    old, old_auth = publish(normal, {"profile": {"resume_boundaries": normal_boundary}})
    monkeypatch.setattr(status, "RELAY_6101_WAVE28_CONTRACT", old)
    monkeypatch.setattr(status, "RELAY_6101_WAVE28_AUTH", old_auth)
    for index in (0, 1):
        write_json(normal / f"seed_6101/round_08/turn_00/attempt_{index:02d}.started.json", {"attempt": index})
    decision = "seed_6101/round_08/turn_00/decision.json"
    write_json(normal / decision, {"accepted": False})
    monkeypatch.setattr(status, "REPAIR_6101_R8_PARENT_FILES", {decision: status.checksum(normal / decision)})
    metadata = {"fixture_scope": "two fixed references"}
    metadata_path = targeted / "fixed_reference_repair.json"
    write_json(metadata_path, metadata)
    monkeypatch.setattr(status, "REPAIR_6101_R8_METADATA", status.record_digest(metadata))
    monkeypatch.setattr(status, "REPAIR_6101_R8_METADATA_FILE", status.checksum(metadata_path))
    repair, repair_auth = publish(targeted, {
        "source_sha256": status.REPAIR_6101_R8_SOURCES, "budget": repair_budget,
        "profile": {"confirmation_seeds": [6101], "target_labels": 621, "resume_boundaries": normal_boundary,
                    "ipv4_readiness_windows": 3, "fixed_reference_repair": metadata,
                    "transport": {"logical_replies": 1, "transport_attempts": 1}},
        "predecessor": {"dispatcher_module": "dispatch_v13_6101_r8_targeted_repair", "seed": 6101, "round": 8,
                        "previous_run_directory": str(normal), "previous_contract_sha256": old,
                        "previous_authorization_sha256": old_auth, "prior_response_reposted": False,
                        "logical_index": 0, "fixed_reference_repair_sha256": status.REPAIR_6101_R8_METADATA,
                        "execution_budget": repair_budget,
                        "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}},
    })
    monkeypatch.setattr(status, "REPAIR_6101_R8_CONTRACT", repair)
    monkeypatch.setattr(status, "REPAIR_6101_R8_AUTH", repair_auth)
    assert status.r8_6101_targeted_repair(tmp_path, normal, old) == (targeted, repair, 8, 3)
    write_json(normal / decision, {"accepted": True})
    assert status.r8_6101_targeted_repair(tmp_path, normal, old) is None
    write_json(normal / decision, {"accepted": False})

    freeze = targeted / "seed_6101/round_08/batch_freeze.json"
    write_json(freeze, {"candidate_count": 32})
    monkeypatch.setattr(status, "REPAIR_6101_R8_FREEZE", status.checksum(freeze))
    fit = targeted / "seed_6101/round_09"
    source, metrics = fit / "source.json", fit / "fit_000/validation_metrics.json"
    write_json(source, {"round": 9})
    write_json(metrics, {"combined_normalized_rmse": 0.5})
    attempt_id = "a" * 32
    job = targeted / "bridge/outbox" / f"{attempt_id}.json"
    result = targeted / "bridge/results" / f"{attempt_id}.json"
    write_json(job, {"attempt_id": attempt_id})
    write_json(result, {"attempt_id": attempt_id, "complete": True})
    request = {"payload_sha256": "b" * 64}
    handoff = {"fit_files": {str(source.relative_to(targeted)): status.checksum(source)},
               "bridge": {"attempt_id": attempt_id, "provider_post": True,
                          "job_file_sha256": status.checksum(job), "result_file_sha256": status.checksum(result)}}
    monkeypatch.setattr(status, "CONTINUATION_6101_R9_HANDOFF", status.record_digest(handoff))
    monkeypatch.setattr(status, "CONTINUATION_6101_R9_REQUEST", status.record_digest(request))
    continued, continued_auth = publish(resumed, {
        "source_sha256": status.CONTINUATION_6101_R9_SOURCES, "budget": status.CONTINUATION_6101_R9_BUDGET,
        "profile": {"transport": {"logical_replies": 3, "transport_attempts": 2}, "target_labels": 1485,
                    "resume_boundaries": {"6101": status.CONTINUATION_6101_R9_DESCRIPTOR},
                    "confirmation_seeds": [6101], "full_imported_boundary_round": 9, "replay_eligible_from_round": 9,
                    "sealed_relay_retry_seconds": 180, "audited_future_transport_replay": True, "ipv4_readiness_windows": 3},
        "predecessor": {"dispatcher_module": "dispatch_v13_6101_r9_continuation", "seed": 6101, "round": 9,
                        "previous_run_directory": str(targeted), "previous_contract_sha256": repair,
                        "previous_authorization_sha256": repair_auth, "normal_contract_sha256": old,
                        "normal_authorization_sha256": old_auth, "normal_run_directory": str(normal),
                        "checkpoint_sha256": status.REPAIR_6101_R8_CHECKPOINT, "r8_handoff": handoff,
                        "r9_request": request, "execution_budget": status.CONTINUATION_6101_R9_BUDGET,
                        "recovery_budget": status.CONTINUATION_6101_R9_SCOPE},
    })
    monkeypatch.setattr(status, "CONTINUATION_6101_R9_CONTRACT", continued)
    monkeypatch.setattr(status, "CONTINUATION_6101_R9_AUTH", continued_auth)
    assert status.r9_after_6101_r8_repair(tmp_path, targeted, repair) is None
    write_json(fit / "transition.json", {"fit_path": "fit_000", "checkpoint_sha256": status.REPAIR_6101_R8_CHECKPOINT,
                                         "source_sha256": status.checksum(source),
                                         "validation_metrics_sha256": status.checksum(metrics)})
    assert status.r9_after_6101_r8_repair(tmp_path, targeted, repair) == (resumed, continued, 9, 3)
    write_json(result, {"attempt_id": attempt_id, "complete": False})
    assert status.r9_after_6101_r8_repair(tmp_path, targeted, repair) is None


def test_relay_discovery_reads_only_private_canonical_launcher_arguments(tmp_path):
    import plistlib

    status = prompt_pilot_status
    private = tmp_path.resolve()
    directory = private / "launchd"
    directory.mkdir(mode=0o700)
    label = "com.purellm.a7-ipv4-v13-887-relay-retry-16"
    suffix = f"{status.SERIAL_IPV4}-887-relay-retry-16"
    args = ["/usr/bin/caffeinate", "-i", "/private/runtime/bin/python", "-B", "-u", "-m",
            "studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.dispatch_v13_retry_successor", "--execute-paid",
            "--seed", "887", "--wave", "16",
            "--remote-root", str(status.RUNS.parent / "releases" / ("ipv4-v13-887-relay-retry-16-" + "e" * 64)),
            "--remote-run-dir", str(status.RUNS / suffix), "--contract-sha256", "a" * 64, "--authorization-sha256", "b" * 64]
    path = directory / (label + ".plist")
    path.write_bytes(plistlib.dumps({"Label": label, "ProgramArguments": args, "EnvironmentVariables": {"ignored": "secret"}}))
    path.chmod(0o600)
    assert status.discover_relay_pins(private, "eit-hpc") == [f"887:16:{'a'*64}:{'b'*64}"]
    assert status.discover_relay_pins(private, "another-host") == []
    path.chmod(0o622)
    assert status.discover_relay_pins(private, "eit-hpc") == []
    path.chmod(0o600)
    original = path.read_bytes()
    injected = list(args)
    injected[args.index("--remote-run-dir") + 1] = str(status.RUNS / ".." / suffix)
    path.write_bytes(plistlib.dumps({"Label": label, "ProgramArguments": injected}))
    assert status.discover_relay_pins(private, "eit-hpc") == []
    path.unlink()
    target = private / "shadow.plist"
    target.write_bytes(original)
    path.symlink_to(target)
    assert status.discover_relay_pins(private, "eit-hpc") == []
    assert status._relay_pin(f"887:16:{'a'*64}:aaa;touch") is None


def test_owned_upstream_failure_requires_exact_failure_event_and_transport_alias():
    provider = {
        "http_status": 200,
        "error_type": "response_id_mismatch",
        "conflicting_event_type": "response.failed",
        "response_id_matches_transport_alias": True,
        "provider_error_code": "upstream_error",
    }
    assert prompt_pilot_status.owned_upstream_failure(provider)
    assert not prompt_pilot_status.owned_upstream_failure({**provider, "response_id_matches_transport_alias": False})
    assert not prompt_pilot_status.owned_upstream_failure({**provider, "conflicting_event_type": "response.completed"})
    assert not prompt_pilot_status.owned_upstream_failure({**provider, "error_type": "sse_inactivity_timeout"})


def test_live_dispatch_requires_matching_loaded_arguments_and_expected_pins(tmp_path, monkeypatch):
    status = prompt_pilot_status
    private = tmp_path.resolve() / "purellm"
    directory = private / "launchd"
    directory.mkdir(parents=True, mode=0o700)
    label = "com.purellm.a7-ipv4-v13-6101-relay-retry-32"
    path = directory / f"{label}.plist"
    contract, authorization, package = "a" * 64, "b" * 64, "c" * 64
    run = status.RUNS / f"{status.SERIAL_IPV4}-6101-relay-retry-32"
    release = status.RUNS.parent / "releases" / f"ipv4-v13-6101-relay-retry-32-{package}"
    args = ["/usr/bin/caffeinate", "-i", "/trusted/python", "-B", "-u", "-m",
            "studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.dispatch_v13_retry_successor",
            "--execute-paid", "--seed", "6101", "--wave", "32", "--remote-root", str(release),
            "--remote-run-dir", str(run), "--contract-sha256", contract, "--authorization-sha256", authorization]
    path.write_bytes(status.plistlib.dumps({"Label": label, "ProgramArguments": args}))
    path.chmod(0o600)
    loaded = [f"path = {path}", "state = running", "pid = 420", "arguments = {", *args, "}"]
    monkeypatch.setattr(status, "command_lines", lambda command: [f"420\t0\t{label}"] if command[-1] == "list" else loaded)
    running = status._running_relays()
    assert running == {label: 420}
    expected = f"6101:32:{contract}:{authorization}:{package}"
    assert status.discover_relay_pins(private, "eit-hpc", running=running, expected=(contract, authorization)) == [expected]
    assert status._live_dispatch_pin(expected) == (6101, 32, contract, authorization, package)
    assert status._live_dispatch_pin(expected + ";touch") is None
    assert status.discover_relay_pins(private, "eit-hpc", running=running, expected=(authorization, contract)) == []
    assert status.discover_relay_pins(private, "eit-hpc", running={}) == []
    loaded[3] = "arguments = unknown"
    assert status.discover_relay_pins(private, "eit-hpc", running=running) == []
    loaded[3] = "arguments = {"
    loaded[-2] = "d" * 64
    assert status.discover_relay_pins(private, "eit-hpc", running=running) == []
    assert status.discover_relay_pins(private, "eit-hpc") == [f"6101:32:{contract}:{authorization}"]


def test_live_dispatch_accepts_reviewed_launch_only_with_local_exact_private_pair(tmp_path, monkeypatch):
    status = prompt_pilot_status
    private = tmp_path.resolve() / "purellm"
    records = private / "reviewed_recovery"
    records.mkdir(parents=True, mode=0o700)
    (private / "launchd").mkdir(mode=0o700)
    label = "com.purellm.a7-ipv4-v13-6101-relay-retry-32"
    plist = private / "launchd" / f"{label}.plist"
    contract, authorization, package = "a" * 64, "b" * 64, "c" * 64
    run = status.RUNS / f"{status.SERIAL_IPV4}-6101-relay-retry-32"
    release = status.RUNS.parent / "releases" / f"ipv4-v13-6101-relay-retry-32-{package}"
    review = {"schema": status.REVIEWED_RECOVERY_SCHEMA, "seed": 6101}
    digest = status.record_digest(review)
    path = records / f"6101-32-{digest}.json"
    write_json(path, review)
    path.chmod(0o600)
    args = ["/usr/bin/caffeinate", "-i", "/trusted/python", "-B", "-u", "-m",
            "studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.dispatch_v13_retry_successor",
            "--execute-paid", "--seed", "6101", "--wave", "32", "--remote-root", str(release),
            "--remote-run-dir", str(run), "--contract-sha256", contract, "--authorization-sha256", authorization]

    def discovered(flags, *, live=True):
        arguments = [*args, *flags]
        plist.write_bytes(status.plistlib.dumps({"Label": label, "ProgramArguments": arguments}))
        plist.chmod(0o600)
        loaded = [f"path = {plist}", "state = running", "pid = 420", "arguments = {", *arguments, "}"]
        monkeypatch.setattr(status, "command_lines", lambda *_: loaded)
        return status.discover_relay_pins(private, "eit-hpc", running={label: 420} if live else None)

    flags = ["--reviewed-record", str(path), "--reviewed-record-sha256", digest]
    pin = f"6101:32:{contract}:{authorization}:{package}:32:{digest}"
    assert discovered(flags) == [pin]
    assert status._live_dispatch_pin(pin) == (6101, 32, contract, authorization, package, 32, digest)
    assert status._live_dispatch_pin(pin + ":oops") is None
    assert discovered(flags, live=False) == []
    assert discovered(flags[:2]) == [] and discovered(flags[2:]) == []
    assert discovered(["--reviewed-record", "../borrowed.json", *flags[2:]]) == []
    path.chmod(0o644)
    assert discovered(flags) == []
    path.chmod(0o600)
    write_json(path, {**review, "seed": 887})
    assert discovered(flags) == []
    path.rename(path.with_suffix(".outside"))
    path.symlink_to(path.with_suffix(".outside"))
    assert discovered(flags) == []


def test_live_dispatch_keeps_scientific_status_and_checks_gpfs_manifest(tmp_path, monkeypatch):
    status = prompt_pilot_status
    runs = tmp_path.resolve() / "runs"
    run = runs / f"{status.SERIAL_IPV4}-6101-relay-retry-32"
    package = "c" * 64
    release = runs.parent / "releases" / f"ipv4-v13-6101-relay-retry-32-{package}"
    manifests = {}
    for module in ("dispatch_v13_retry_successor", "run_v13_retry_successor"):
        name = f"studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/{module}.py"
        source = release / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("source\n")
        manifests[name] = status.checksum(source)
    record = {"run_directory": str(run), "source_sha256": manifests, "api_execution": "mac_ssh_dispatch_v1",
              "profile": {"confirmation_seeds": [6101], "resume_boundaries": {"6101": {"round": 13}}},
              "predecessor": {"dispatcher_module": "dispatch_v13_retry_successor", "seed": 6101, "round": 13}}
    contract = status.record_digest(record)
    approval = {"contract_sha256": contract, **dict.fromkeys(status.FORMAL_APPROVAL_FLAGS, True)}
    authorization = status.record_digest(approval)
    write_json(run / "formal_contract.json", record)
    write_json(run / "authorization.json", approval)
    write_json(run / "workers/seed_6101.json", {"status": "initializing"})
    write_json(run / "coordinator_generation.json", {"pid": 4321})
    monkeypatch.setattr(status, "live", lambda process, **options: process == 4321 and options.get("run") == run)
    monkeypatch.setattr(status, "command_lines", lambda *_args: [])
    expected_seeds = [{"seed": seed, "labels": 333} for seed in status.SEEDS]
    monkeypatch.setattr(status, "remote_seed", lambda _runs, seed: next(row for row in expected_seeds if row["seed"] == seed))

    request_id, attempt_id = "d" * 64, "e" * 32
    started = {"attempt_id": attempt_id, "request_id": request_id, "attempt": 0, "started_unix": 500.0}
    ledger = run / "seed_6101/round_13/turn_00/attempt_00.started.json"
    write_json(ledger, started)
    payload = {"input": "bound"}
    request = {"request_id": request_id, "logical_index": 0, "payload": payload, "payload_sha256": status.record_digest(payload),
               "binding": {"seed": 6101, "round": 13, "a7_formal": {"contract_sha256": contract,
                           "authorization_sha256": authorization, "seed": 6101, "mock": False}}}
    job = {"attempt_id": attempt_id, "request_id": request_id, "request": request, "started": started}
    job["job_sha256"] = status.record_digest(job)
    write_json(run / "bridge/outbox" / f"{attempt_id}.json", job)
    pin = (6101, 32, contract, authorization, package)
    view = status.remote_report(runs, live_dispatch_pins=[pin])
    assert view["seeds"] == expected_seeds and "live_dispatch" not in status.remote_report(runs)
    live, = view["live_dispatch"]
    assert live["coordinator_pid"] == 4321 and "labels" not in live and "checkpoint_sha256" not in live
    assert live["api"]["attempt_id"] == attempt_id and live["api"]["status"] == "waiting_for_mac"
    assert not status.live_dispatch_report(runs, [(6101, 32, contract, "0" * 64, package)])
    result = run / "bridge/results" / f"{attempt_id}.json"
    write_json(result, {"attempt_id": attempt_id})
    assert status.live_dispatch_report(runs, [pin])[0]["api"] == {}
    parent_run = run.with_name(f"{status.SERIAL_IPV4}-6101-relay-retry-31")
    parent = {"run_directory": str(parent_run), "predecessor": {}}
    parent_contract = status.record_digest(parent)
    parent_approval = {"contract_sha256": parent_contract, **dict.fromkeys(status.FORMAL_APPROVAL_FLAGS, True)}
    parent_auth = status.record_digest(parent_approval)
    write_json(parent_run / "formal_contract.json", parent)
    write_json(parent_run / "authorization.json", parent_approval)
    payload_sha, checkpoint_sha = "f" * 64, "a" * 64
    archive = {"schema": status.REVIEWED_RECOVERY_SCHEMA, "seed": 6101, "round": 13, "anchor_wave": 31,
               "payload_sha256": payload_sha, "checkpoint_sha256": checkpoint_sha,
               "steps": [{"parent_contract_sha256": parent_contract, "parent_authorization_sha256": parent_auth,
                          "request_id": request_id}]}
    review_sha = status.record_digest(archive)
    record["predecessor"].update({
        status.REVIEWED_RECOVERY_FIELD: review_sha,
        "previous_run_directory": str(parent_run), "previous_contract_sha256": parent_contract,
        "previous_authorization_sha256": parent_auth, "previous_request_id": request_id,
        "payload_sha256": payload_sha, "checkpoint_sha256": checkpoint_sha,
        "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 1},
    })
    contract = status.record_digest(record)
    approval["contract_sha256"] = contract
    authorization = status.record_digest(approval)
    write_json(run / "formal_contract.json", record)
    write_json(run / "authorization.json", approval)
    reviewed = run / status.REVIEWED_RECOVERY_FILE
    write_json(reviewed, archive)
    reviewed_pin = (6101, 32, contract, authorization, package, 32, review_sha)
    view = status.remote_report(runs, live_dispatch_pins=[reviewed_pin])
    assert view["seeds"] == expected_seeds
    live, = view["live_dispatch"]
    assert live[status.REVIEWED_RECOVERY_FIELD] == review_sha and "checkpoint_sha256" not in live
    assert not status.live_dispatch_report(runs, [reviewed_pin[:5]])
    assert not status.live_dispatch_report(runs, [(*reviewed_pin[:-1], "0" * 64)])
    write_json(reviewed, {**archive, "round": 14})
    assert not status.live_dispatch_report(runs, [reviewed_pin])
    write_json(reviewed, archive)
    reviewed.rename(reviewed.with_suffix(".outside"))
    reviewed.symlink_to(reviewed.with_suffix(".outside"))
    assert not status.live_dispatch_report(runs, [reviewed_pin])
    reviewed.unlink()
    write_json(reviewed, archive)
    write_json(parent_run / "authorization.json", {**parent_approval, "authorize_paid_api": False})
    assert not status.live_dispatch_report(runs, [reviewed_pin])
    write_json(parent_run / "authorization.json", parent_approval)

    next_run = run.with_name(f"{status.SERIAL_IPV4}-6101-relay-retry-33")
    next_request = "b" * 64
    next_archive = {**archive, "steps": [*archive["steps"], {
        "parent_contract_sha256": contract, "parent_authorization_sha256": authorization, "request_id": next_request,
    }]}
    next_sha = status.record_digest(next_archive)
    next_record = {**record, "run_directory": str(next_run), "predecessor": {
        **record["predecessor"], status.REVIEWED_RECOVERY_FIELD: next_sha,
        "previous_run_directory": str(run), "previous_contract_sha256": contract,
        "previous_authorization_sha256": authorization, "previous_request_id": next_request,
    }}
    next_contract = status.record_digest(next_record)
    next_approval = {**approval, "contract_sha256": next_contract}
    write_json(next_run / "formal_contract.json", next_record)
    write_json(next_run / "authorization.json", next_approval)
    write_json(next_run / status.REVIEWED_RECOVERY_FILE, next_archive)
    next_pin = (6101, 33, next_contract, status.record_digest(next_approval), package, 32, next_sha)
    assert status.live_dispatch_report(runs, [next_pin])[0][status.REVIEWED_RECOVERY_FIELD] == next_sha
    assert not status.live_dispatch_report(runs, [(*next_pin[:5], 31, next_sha)])
    source.write_text("changed\n")
    assert not status.live_dispatch_report(runs, [reviewed_pin])
    assert not status.live_dispatch_report(runs, [next_pin])


def test_fixed_active_review_requires_response_owned_by_the_exact_request(tmp_path):
    status = prompt_pilot_status
    attempt = "a" * 32
    provider = "12345678-bbbb-cccc-dddd-000011112222"
    alias = "resp_12345678bbbbccccdddd000011112222"
    progress = {"http_status": 200, "provider_error_code": "upstream_error",
                "conflicting_event_type": "response.failed", "response_id_matches_transport_alias": True,
                "provider_request_id": provider, "conflicting_response_id": alias, "response_id": "resp_original"}
    identity = {"provider_request_id": provider, "provider_status": "failed", "response_id": alias,
                "served_model": "gpt-6.2-astra", "usage": {}}
    result = tmp_path / "bridge/results" / f"{attempt}.json"
    write_json(result, {"outcome": {"failure": {"reply_identity": identity}}})
    proof = {"previous_transport_observation": progress}
    assert status._fixed_owned_response(tmp_path, proof, {"attempt_id": attempt})
    progress["provider_request_id"] = "12345678-bbbb-cccc-dddd-000011112223"
    assert not status._fixed_owned_response(tmp_path, proof, {"attempt_id": attempt})
    progress["provider_request_id"] = provider
    progress.pop("response_id_matches_transport_alias")
    assert not status._fixed_owned_response(tmp_path, proof, {"attempt_id": attempt})
    assert status._fixed_owned_response(tmp_path, proof, {"attempt_id": attempt}, legacy_alias=True)
    progress["response_id_matches_transport_alias"] = False
    assert not status._fixed_owned_response(tmp_path, proof, {"attempt_id": attempt}, legacy_alias=True)
    progress["response_id_matches_transport_alias"] = True
    write_json(result, {"outcome": {"failure": {"reply_identity": {**identity, "usage": {"output_tokens": 1}}}}})
    assert not status._fixed_owned_response(tmp_path, proof, {"attempt_id": attempt})
    assert not status._fixed_owned_response(tmp_path, proof, {"attempt_id": "../foreign"})


def test_fixed_reviewed_bridge_requires_signed_contiguous_retired_receipts(tmp_path, monkeypatch):
    status = prompt_pilot_status
    seed, start, first_wave, wave = 2357, 5, 48, 50
    runs = tmp_path.resolve() / "runs"
    previous, reviewed, child = [runs / f"{status.SERIAL_IPV4}-{seed}-relay-retry-{index:02d}"
                                 for index in range(first_wave, wave + 1)]
    package, checkpoint, payload = "a" * 64, "b" * 64, "c" * 64
    release = runs.parent / "releases" / f"ipv4-v13-{seed}-relay-retry-{wave:02d}-{package}"
    sources = {}
    for name in ("run_v13_retry_successor", "dispatch_v13_retry_successor"):
        filename = f"studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/{name}.py"
        target = release / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("signed launcher\n")
        sources[filename] = status.checksum(target)
    physical_limit = {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
    budget = {"first_logical_requests": 31, "maximum_logical_requests": 91, "maximum_physical_attempts": 181}
    profile = {"confirmation_seeds": [seed], "target_labels": 1485, "resume_boundaries": {str(seed): {"round": start}},
               "ipv4_readiness_windows": 3, "replay_eligible_from_round": 6, "sealed_relay_retry_seconds": 180,
               "audited_future_transport_replay": True}
    common = {"dispatcher_module": "dispatch_v13_retry_successor", "seed": seed, "round": start, "logical_index": 0,
              "recovery_budget": physical_limit, "execution_budget": budget, "checkpoint_sha256": checkpoint,
              "payload_sha256": payload, "reviewed_retry_chain": [],
              **dict.fromkeys(("packet_sha256", "wire_packet_sha256", "base_messages_sha256", "model_identity",
                               "cache_directory", "public_resource_receipt_sha256", "bridge_wait_seconds"), "bound")}

    def publish(run, proof):
        record = {"run_directory": str(run), "profile": profile, "predecessor": proof, "budget": budget,
                  "source_sha256": sources}
        contract = status.record_digest(record)
        approval = {"contract_sha256": contract, **dict.fromkeys(status.FORMAL_APPROVAL_FLAGS, True)}
        write_json(run / "formal_contract.json", record)
        write_json(run / "authorization.json", approval)
        return contract, status.record_digest(approval)

    def successor_of(last, proof, signed, code):
        contract, auth = signed
        receipts = [{"error": code, "provider_dispatched": True}]
        result = {**common, "previous_contract_sha256": contract, "previous_authorization_sha256": auth,
                  "previous_run_directory": str(last), "previous_request_id": "d" * 64,
                  "previous_terminal_file_sha256": "e" * 64, "previous_transport_progress_file_sha256": "f" * 64,
                  "previous_attempts": receipts, "relay_not_before_unix": 830.0}
        result["reviewed_retry_chain"] = [*proof["reviewed_retry_chain"], status._relay_review(last, contract, auth, result, receipts)]
        return result

    anchor = publish(previous, common)
    reviewed_proof = successor_of(previous, common, anchor, "response_id_mismatch")
    reviewed_record = {"schema": status.REVIEWED_RECOVERY_SCHEMA, "seed": seed, "round": start, "anchor_wave": first_wave,
                       "payload_sha256": payload, "checkpoint_sha256": checkpoint,
                       "steps": [{"parent_contract_sha256": anchor[0], "parent_authorization_sha256": anchor[1],
                                  "request_id": reviewed_proof["previous_request_id"]}]}
    marker = status.record_digest(reviewed_record)
    reviewed_proof[status.REVIEWED_RECOVERY_FIELD] = marker
    review_pin = publish(reviewed, reviewed_proof)
    write_json(reviewed / status.REVIEWED_RECOVERY_FILE, reviewed_record)
    retired = []
    for member, pair, digest in ((previous, anchor, None), (reviewed, review_pin, marker)):
        directory = member / f"seed_{seed}"
        pause = directory / "pause.request"
        stage = directory / f"round_{start:02d}"
        source = stage / "source.json"
        write_json(pause, {"paused": True})
        write_json(source, {"checkpoint_sha256": checkpoint})
        retired.append({"contract_sha256": pair[0], status.REVIEWED_RECOVERY_FIELD: digest,
                        "pause_file_sha256": status.checksum(pause), "source_file_sha256": status.checksum(source)})
    proof = successor_of(reviewed, reviewed_proof, review_pin, "timeout")
    proof.update(retired_reviewed_recovery_records=retired, previous_transport_observation={
        "http_status": 200, "error_type": "sse_inactivity_timeout", "sse_event_count": 21,
    })
    current = publish(child, proof)
    pin = {"anchor_contract": anchor[0], "anchor_wave": first_wave, "wave": wave, "round": start,
           "retired_anchor_wave": first_wave, "current": current, "checkpoint": checkpoint,
           "payload": payload, "package": package}
    monkeypatch.setitem(status.EXACT_RETIRED_REVIEWED_BRIDGES, (previous.name, seed), pin)
    monkeypatch.setattr(status, "_pinned_relay_evidence", lambda member, *args: member in (previous, reviewed))
    successor, lineage = status._fixed_reviewed_bridge(runs, previous, anchor[0], seed)
    assert successor[:5] == (child, current[0], start, 3, 830.0) and lineage == (reviewed, previous)
    assert status._fixed_reviewed_bridge(runs, previous, current[0], seed) is None
    path = reviewed / status.REVIEWED_RECOVERY_FILE
    write_json(path, {**reviewed_record, "round": 6})
    assert status._fixed_reviewed_bridge(runs, previous, anchor[0], seed) is None
    write_json(path, reviewed_record)
    write_json(reviewed / f"seed_{seed}/pause.request", {"paused": False})
    assert status._fixed_reviewed_bridge(runs, previous, anchor[0], seed) is None


def test_relay_discovery_keeps_exact_parent_contract_science_and_fence(tmp_path, monkeypatch):
    status = prompt_pilot_status
    runs = tmp_path.resolve()
    previous = runs / f"{status.SERIAL_IPV4}-887-relay-retry-15"
    current = runs / f"{status.SERIAL_IPV4}-887-relay-retry-16"

    # The receipt check has its own fixture below. Keep this fixture focused on the authority chain.
    archived = []
    def verified(path, seed, round_index, proof, contract, auth):
        archived.append((path, seed, round_index, contract, auth))
        return True
    monkeypatch.setattr(status, "_owned_relay_retry_evidence", verified)

    def publish(run, record):
        signed = {**record, "run_directory": str(run)}
        sha = status.record_digest(signed)
        approval = {"contract_sha256": sha, **dict.fromkeys(status.FORMAL_APPROVAL_FLAGS, True)}
        write_json(run / "formal_contract.json", signed)
        write_json(run / "authorization.json", approval)
        return sha, status.record_digest(approval)

    profile = {"confirmation_seeds": [887], "target_labels": 1485, "ipv4_readiness_windows": 3,
               "audited_future_transport_replay": True, "sealed_relay_retry_seconds": 180, "replay_eligible_from_round": 34}
    immutable = dict.fromkeys(("checkpoint_sha256", "packet_sha256", "wire_packet_sha256", "base_messages_sha256",
                               "payload_sha256", "model_identity", "cache_directory", "public_resource_receipt_sha256",
                               "bridge_wait_seconds"), "bound")
    previous_proof = {**immutable, "round": 34, "dispatcher_module": "dispatch_v13_retry_successor",
                      "reviewed_retry_chain": [{"scope": "older signed history"}]}
    producer = "studies/active_learning/pure_llm_v3_a7_gpu_20261124/scripts/run_v13_retry_successor.py"
    inherited = dict.fromkeys(("api_execution", "gpu_slots", "asset_sha256", "anchor_root", "lease_root", "initial_policy"), "same")
    old, old_auth = publish(previous, {**inherited, "source_sha256": {producer: "a" * 64, "provider.py": "pinned"},
                                       "profile": profile, "predecessor": previous_proof})
    budget = {"first_logical_requests": 2, "maximum_logical_requests": 4, "maximum_physical_attempts": 8}
    proof = {**immutable, "round": 34, "seed": 887, "logical_index": 0, "dispatcher_module": "dispatch_v13_retry_successor",
             "previous_run_directory": str(previous), "previous_contract_sha256": old, "previous_authorization_sha256": old_auth,
             "newly_authorized_boundary_physical_attempts": 2, "relay_not_before_unix": 1720000000.5, "execution_budget": budget,
             "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 2},
             "previous_request_id": "request", "previous_terminal_file_sha256": "terminal",
             "previous_transport_progress_file_sha256": "progress", "previous_attempts": []}
    proof["reviewed_retry_chain"] = [*previous_proof["reviewed_retry_chain"],
                                     status._relay_review(previous, old, old_auth, proof, proof["previous_attempts"])]
    record = {**inherited, "profile": profile, "predecessor": proof, "budget": budget,
              "source_sha256": {producer: "b" * 64, "provider.py": "pinned"}}
    pin = publish(current, record)
    result = status.discovered_relay_successor(runs, previous, old, 887, 16, pin)
    assert result == (current, pin[0], 34, 3, proof["relay_not_before_unix"], proof)
    assert archived == [(previous, 887, 34, old, old_auth)]
    assert status.discovered_relay_successor(runs, previous, old, 887, 17, pin) is None
    assert status.discovered_relay_successor(runs, previous, old, 887, 16, (pin[0], "0" * 64)) is None
    changed = {**record, "profile": {**profile, "reasoning_effort": "low"}}
    assert status.discovered_relay_successor(runs, previous, old, 887, 16, publish(current, changed)) is None
    changed = {**record, "source_sha256": {**record["source_sha256"], "provider.py": "unapproved"}}
    assert status.discovered_relay_successor(runs, previous, old, 887, 16, publish(current, changed)) is None
    pin = publish(current, record)
    approval = current / "authorization.json"
    shadow = runs / "shadow_approval.json"
    approval.rename(shadow)
    approval.symlink_to(shadow)
    assert status.discovered_relay_successor(runs, previous, old, 887, 16, pin) is None


def test_relay_discovery_needs_retired_owned_provider_failure_and_unmodified_receipts(tmp_path):
    status = prompt_pilot_status
    previous = tmp_path.resolve()
    directory = previous / "seed_887"
    turn = directory / "round_34/turn_00"
    attempt = "a" * 32
    contract, auth, request_id = "b" * 64, "c" * 64, "d" * 64
    provider_request = "5758fe74-3467-48ca-bb60-eb3cf5751bfd"
    alias = "resp_" + provider_request.replace("-", "")
    payload = {"input": [{"role": "user", "content": "fixture"}]}
    request = {"request_id": request_id, "payload": payload, "payload_sha256": status.record_digest(payload),
               "logical_index": 0, "binding": {"a7_formal": {"contract_sha256": contract,
                                                            "authorization_sha256": auth, "seed": 887, "mock": False}}}
    write_json(turn / "request.json", request)
    write_json(turn / "attempt_00.started.json", {"attempt": 0, "attempt_id": attempt})
    write_json(previous / "coordinator_generation.json", {})
    write_json(previous / "workers/seed_887.json", {"status": "failed"})
    terminal_path = turn / "attempt_00.terminal.json"
    error = "response_id_mismatch"
    terminal = {"attempt_id": attempt, "request_id": request_id, "status": "stopped", "error": error,
                "provider_dispatched": True, "ambiguous": True, "failed_unix": 490.0, "not_before_unix": 500.0}
    write_json(terminal_path, terminal)
    job = {"attempt_id": attempt, "request_id": request_id, "request": request}
    job["job_sha256"] = status.record_digest(job)
    write_json(previous / "bridge/outbox" / f"{attempt}.json", job)
    identity = {"provider_request_id": provider_request, "response_id": alias, "provider_status": "failed", "usage": {}}
    result = {"attempt_id": attempt, "request_id": request_id, "job_sha256": job["job_sha256"],
              "provider_dispatched": True, "raw_response": None,
              "outcome": {"failure": {"code": error, "ambiguous": True, "reply_identity": identity}}}
    result["result_sha256"] = status.record_digest(result)
    result_path = previous / "bridge/results" / f"{attempt}.json"
    write_json(result_path, result)
    receipt = {"attempt": 0, "attempt_id": attempt, "terminal_sha256": status.record_digest(terminal),
               "job_sha256": job["job_sha256"], "result_sha256": result["result_sha256"], "error": error,
               "provider_dispatched": True, "ambiguous": True}
    observation = {"http_status": 200, "event_type": "response.output_item.done", "error_type": error,
                   "conflicting_event_type": "response.failed", "conflicting_response_id": alias,
                   "response_id_matches_transport_alias": True, "provider_request_id": provider_request,
                   "provider_error_code": "upstream_error"}
    proof = {"previous_attempts": [receipt], "previous_request_id": request_id, "payload_sha256": request["payload_sha256"],
             "previous_transport_observation": observation, "previous_terminal_file_sha256": status.checksum(terminal_path),
             "relay_cooldown_clock": "not_before_unix", "relay_not_before_unix": 680.0}
    assert not status._owned_relay_retry_evidence(previous, 887, 34, proof, contract, auth)
    (directory / "pause.request").touch()
    assert status._owned_relay_retry_evidence(previous, 887, 34, proof, contract, auth)
    assert not status._owned_relay_retry_evidence(previous, 887, 34, {**proof, "relay_not_before_unix": 679.0}, contract, auth)
    write_json(result_path, {**result, "raw_response": {"declared_complete": True}})
    assert not status._owned_relay_retry_evidence(previous, 887, 34, proof, contract, auth)


def test_6101_r10_cached_status_requires_l685_freeze_and_zero_new_posts(tmp_path, monkeypatch):
    status = prompt_pilot_status
    previous = tmp_path / status.CONTINUATION_6101_R9_RUN
    cached = tmp_path / status.CACHED_R10_6101_RUN
    selection = [f"C-{index:04d}" for index in range(32)]
    custody = {"provider_post": False, "response_id": "resp_exact",
               "ordered_candidate_ids_sha256": status.record_digest(selection),
               "original_failure": {"code": "invalid_response_message_phase"}}
    receipt = status.record_digest(custody)
    stage = cached / "seed_6101/round_10"
    fit = cached / "seed_6101/round_11"
    write_json(cached / "cached_response_import.json", custody)
    write_json(stage / "turn_00/attempt_00.terminal.json", {
        "status": "completed", "provider_dispatched": False, "cached_selection_sha256": receipt,
        "request_id": "exact_request", "imported_from_response_id": "resp_exact",
    })
    freeze = stage / "batch_freeze.json"
    frozen_ids = [f"physical_{index}" for index in range(32)]
    write_json(freeze, {"batch_ids": frozen_ids, "request_id": "exact_request"})
    write_json(stage / "selection.json", {"choices": [{"candidate_id": identity} for identity in selection]})
    write_json(cached / "workers/seed_6101.json", {"status": "complete", "completed_rounds": 11, "active_labels": 685})
    write_json(fit / "source.json", {"round": 11})
    write_json(fit / "fit_000/validation_metrics.json", {"combined_normalized_rmse": 0.47})

    proof = {"dispatcher_module": "run_v13_6101_r10_cached", "seed": 6101, "round": 10,
             "previous_run_directory": str(previous), "previous_contract_sha256": status.CONTINUATION_6101_R9_CONTRACT,
             "previous_authorization_sha256": status.CONTINUATION_6101_R9_AUTH, "checkpoint_sha256": status.CACHED_R10_6101_START,
             "execution_budget": status.CACHED_R10_6101_BUDGET, "cached_selection": custody,
             "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 0}}
    signed = {"budget": status.CACHED_R10_6101_BUDGET, "predecessor": proof,
              "profile": {"confirmation_seeds": [6101], "target_labels": 685, "resume_boundaries": {"6101": {"round": 10}}}}
    actual_latest = status.latest_round
    monkeypatch.setattr(status, "_signed_retry_record", lambda run, *ident: (signed, ident[-1]))
    monkeypatch.setattr(status, "latest_round",
                        lambda directory: (10, {"checkpoint_sha256": status.CACHED_R10_6101_START})
                        if directory == previous / "seed_6101" else actual_latest(directory))
    monkeypatch.setattr(status, "CACHED_R10_6101_RECEIPT", receipt)
    monkeypatch.setattr(status, "CACHED_R10_6101_FILES", {str(freeze.relative_to(cached)): status.checksum(freeze)})
    status_row = lambda: status.r10_cached_after_6101_r9(tmp_path, previous, status.CONTINUATION_6101_R9_CONTRACT)
    assert status_row() is None
    write_json(fit / "transition.json", {"fit_path": "fit_000", "checkpoint_sha256": status.CACHED_R10_6101_FIT,
                                         "source_sha256": status.checksum(fit / "source.json"),
                                         "validation_metrics_sha256": status.checksum(fit / "fit_000/validation_metrics.json")})
    assert status_row() == (cached, status.CACHED_R10_6101_CONTRACT, 10, {
        "round": 10, "fitted_labels": 685, "choice_count": 32, "removed_reference_fields": 0,
        "provider_dispatched": False, "archived_failed_correction": "invalid_response_message_phase",
    })
    unexpected = cached / "bridge/outbox/provider-post.json"
    write_json(unexpected, {"unexpected": True})
    assert status_row() is None
    unexpected.unlink()
    write_json(stage / "selection.json", {"choices": [{"candidate_id": identity} for identity in reversed(selection)]})
    assert status_row() is None


def test_6101_r11_monitor_requires_the_signed_cached_handoff_and_normal_budget(tmp_path, monkeypatch):
    status = prompt_pilot_status
    prior = tmp_path / status.CACHED_R10_6101_RUN
    normal = tmp_path / status.CONTINUATION_6101_R11_RUN
    handoff = {"provider_post": False, "checkpoint_sha256": status.CACHED_R10_6101_FIT}
    request = {"payload_sha256": "a" * 64}
    custody = {"provider_post": False}
    for key, value in (("CONTINUATION_6101_R11_HANDOFF", handoff), ("CONTINUATION_6101_R11_REQUEST", request),
                       ("CACHED_R10_6101_RECEIPT", custody)):
        monkeypatch.setattr(status, key, status.record_digest(value))
    profile = {"resume_boundaries": {"6101": status.CONTINUATION_6101_R11_DESCRIPTOR},
               "confirmation_seeds": [6101], "target_labels": 1485, "full_imported_boundary_round": 11,
               "replay_eligible_from_round": 11, "sealed_relay_retry_seconds": 180, "audited_future_transport_replay": True,
               "ipv4_readiness_windows": 3, "transport": {"logical_replies": 3, "transport_attempts": 2}}
    proof = {"dispatcher_module": "dispatch_v13_6101_r11_continuation", "seed": 6101, "round": 11,
             "previous_run_directory": str(prior), "previous_contract_sha256": status.CACHED_R10_6101_CONTRACT,
             "previous_authorization_sha256": status.CACHED_R10_6101_AUTH,
             "normal_run_directory": str(tmp_path / status.CONTINUATION_6101_R9_RUN),
             "normal_contract_sha256": status.CONTINUATION_6101_R9_CONTRACT, "normal_authorization_sha256": status.CONTINUATION_6101_R9_AUTH,
             "checkpoint_sha256": status.CACHED_R10_6101_FIT, "execution_budget": status.CONTINUATION_6101_R11_BUDGET,
             "recovery_budget": status.CONTINUATION_6101_R9_SCOPE, "r10_handoff": handoff,
             "r11_request": request, "cached_selection": custody,
             "cached_response_import_file_sha256": status.CACHED_R10_6101_FILES["cached_response_import.json"]}
    record = {"profile": profile, "predecessor": proof, "budget": status.CONTINUATION_6101_R11_BUDGET,
              "source_sha256": status.CONTINUATION_6101_R11_SOURCES}
    monkeypatch.setattr(status, "_signed_retry_record", lambda run, *ident: (record, ident[-1]))
    status_row = lambda: status.r11_6101_after_cached_r10(tmp_path, prior, status.CACHED_R10_6101_CONTRACT)
    assert status_row() == (normal, status.CONTINUATION_6101_R11_CONTRACT, 11, 3)
    profile["transport"] = {"logical_replies": 3, "transport_attempts": 1}
    assert status_row() is None
    profile["transport"] = {"logical_replies": 3, "transport_attempts": 2}
    proof["r10_handoff"] = {**handoff, "provider_post": True}
    assert status_row() is None
