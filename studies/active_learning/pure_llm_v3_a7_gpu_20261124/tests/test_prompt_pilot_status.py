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


@pytest.mark.parametrize("seed,wave", [(157, 9), (2357, 8)])
def test_reviewed_relay_after_new_cuda_fit_requires_exact_fit_and_archive(tmp_path, monkeypatch, seed, wave):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.run_v13_retry_successor import SCOPES

    status = prompt_pilot_status
    scope = SCOPES[(seed, wave)]
    review, = scope.archive
    start = scope.round
    prefix = f"RELAY_{seed}_WAVE{wave:02d}"
    previous_prefix = f"RELAY_{seed}_WAVE{wave - 1:02d}"
    proof = getattr(status, f"{prefix}_PROOF")
    fitted = getattr(status, f"{prefix}_FITTED_FILES")
    checkpoint = getattr(status, f"{prefix}_CHECKPOINT")
    observation = getattr(status, f"{prefix}_OBSERVATION")
    attempts = (status.RELAY_157_WAVE08_ATTEMPT,) if seed == 157 else status.RELAY_2357_WAVE07_ATTEMPTS
    sources = status.RELAY_12203_WAVE12_SOURCES
    assert fitted == dict(scope.fitted_prefix_files)
    assert sources == dict(scope.required_sources)
    assert observation == review.progress
    assert (getattr(status, f"{previous_prefix}_CONTRACT"), getattr(status, f"{previous_prefix}_AUTH")) == (
        review.identity.contract, review.identity.approval,
    )
    assert proof == {"logical_index": 0, "payload_sha256": scope.payload_sha256, "previous_request_id": review.request_id,
                     "previous_terminal_file_sha256": review.terminal_file_sha256,
                     "previous_transport_progress_file_sha256": review.progress_file_sha256,
                     "relay_cooldown_clock": scope.cooldown_field, "relay_not_before_unix": scope.not_before_unix,
                     "newly_authorized_boundary_physical_attempts": 2, "execution_budget": scope.budget}
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
    if seed == 2357:
        assert dict(review.replay_files) == status.RELAY_2357_WAVE07_REPLAY_FILES
        assert tuple(a.terminal_file_sha256 for a in review.attempts) == status.RELAY_2357_WAVE07_TERMINALS
        for name in review.replay_files:
            write_json(turn / name, {"fixture": name})
        monkeypatch.setattr(status, "RELAY_2357_WAVE07_TERMINALS", terminals)
        monkeypatch.setattr(status, "RELAY_2357_WAVE07_REPLAY_FILES",
                            {name: status.checksum(turn / name) for name in review.replay_files})

    def publish(path, record):
        digest = status.record_digest(record)
        authorization = {"contract_sha256": digest, **dict.fromkeys(status.FORMAL_APPROVAL_FLAGS, True)}
        write_json(path / "formal_contract.json", record)
        write_json(path / "authorization.json", authorization)
        return digest, status.record_digest(authorization)

    old_profile = {"confirmation_seeds": [seed], "target_labels": 1485, "resume_boundaries": {str(seed): {"round": 4}},
                   "audited_future_transport_replay": True, "sealed_relay_retry_seconds": 180,
                   "replay_eligible_from_round": 4 if seed == 157 else 5, "ipv4_readiness_windows": 3,
                   "sse_inactivity_seconds": 900, "request_timeouts": {"total_seconds": 3600}}
    old_budget = getattr(status, f"{previous_prefix}_PROOF")["execution_budget"]
    old_proof = {"recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 2 if seed == 157 else 1}}
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
    new_profile = {**old_profile, "resume_boundaries": {str(seed): descriptor}, "replay_eligible_from_round": start}
    signed = {"run_directory": str(run), "profile": new_profile, "budget": scope.budget,
              "source_sha256": sources, "predecessor": new_proof}
    digest, auth = publish(run, signed)
    monkeypatch.setattr(status, f"{prefix}_CONTRACT", digest)
    assert status.reviewed_relay_contract(tmp_path, seed, wave, previous, old_sha) is None
    monkeypatch.setattr(status, f"{prefix}_AUTH", auth)
    expected = run, digest, start, 3, scope.not_before_unix
    assert status.reviewed_relay_contract(tmp_path, seed, wave, previous, old_sha) == expected
    mutation = turn / "attempt_01.replay_link.json" if seed == 2357 else directory / "round_06/batch_freeze.json"
    original = mutation.read_bytes()
    write_json(mutation, {"modified": True})
    assert status.reviewed_relay_contract(tmp_path, seed, wave, previous, old_sha) is None
    mutation.write_bytes(original)
    assert status.reviewed_relay_contract(tmp_path, seed, wave, previous, old_sha) == expected
    if seed == 2357:
        check_2357_wave09(tmp_path, monkeypatch, publish, run, digest, auth, signed)


def check_2357_wave09(runs, monkeypatch, publish, previous, previous_sha, previous_auth, previous_record):
    status = prompt_pilot_status
    turn = previous / "seed_2357/round_05/turn_00"
    receipts, terminals = [], []
    for attempt in status.RELAY_2357_WAVE08_ATTEMPTS:
        index = attempt["attempt"]
        terminal = {**attempt, "status": "stopped"}
        filename = turn / f"attempt_{index:02d}.terminal.json"
        write_json(turn / f"attempt_{index:02d}.started.json", {"attempt": index, "attempt_id": attempt["attempt_id"]})
        write_json(filename, terminal)
        terminals.append(status.checksum(filename))
        receipts.append({**attempt, "terminal_sha256": status.record_digest(terminal)})
    for name in status.RELAY_2357_WAVE08_REPLAY_FILES:
        write_json(turn / name, {"fixture": name})
    monkeypatch.setattr(status, "RELAY_2357_WAVE08_TERMINALS", terminals)
    monkeypatch.setattr(status, "RELAY_2357_WAVE08_REPLAY_FILES",
                        {name: status.checksum(turn / name) for name in status.RELAY_2357_WAVE08_REPLAY_FILES})
    exact = {**status.RELAY_2357_WAVE09_PROOF, "previous_terminal_file_sha256": terminals[-1]}
    monkeypatch.setattr(status, "RELAY_2357_WAVE09_PROOF", exact)
    old_proof = previous_record["predecessor"]
    old_chain = old_proof["reviewed_retry_chain"]
    new_review = status._relay_review(previous, previous_sha, previous_auth, exact, receipts)
    run = runs / f"{status.SERIAL_IPV4}-2357-relay-retry-09"
    proof = {**old_proof, **exact, "previous_run_directory": str(previous), "previous_contract_sha256": previous_sha,
             "previous_authorization_sha256": previous_auth, "previous_attempts": receipts,
             "previous_transport_observation": status.RELAY_2357_WAVE09_OBSERVATION,
             "reviewed_retry_chain": [*old_chain, new_review]}
    record = {**previous_record, "run_directory": str(run), "predecessor": proof}
    broken = {**record, "predecessor": {**proof, "reviewed_retry_chain": [new_review]}}
    broken_sha, broken_auth = publish(run, broken)
    monkeypatch.setattr(status, "RELAY_2357_WAVE09_CONTRACT", broken_sha)
    monkeypatch.setattr(status, "RELAY_2357_WAVE09_AUTH", broken_auth)
    assert status.reviewed_relay_contract(runs, 2357, 9, previous, previous_sha) is None
    sha, auth = publish(run, record)
    monkeypatch.setattr(status, "RELAY_2357_WAVE09_CONTRACT", sha)
    monkeypatch.setattr(status, "RELAY_2357_WAVE09_AUTH", auth)
    expected = run, sha, 5, 3, exact["relay_not_before_unix"]
    assert status.reviewed_relay_contract(runs, 2357, 9, previous, previous_sha) == expected
    link = turn / "attempt_01.replay_link.json"
    original = link.read_bytes()
    write_json(link, {"modified": True})
    assert status.reviewed_relay_contract(runs, 2357, 9, previous, previous_sha) is None
    link.write_bytes(original)
    assert status.reviewed_relay_contract(runs, 2357, 9, previous, previous_sha) == expected


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
