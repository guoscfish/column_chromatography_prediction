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


@pytest.mark.parametrize(("seed", "start", "labels"), [(12203, 4, 461), (157, 3, 429)])
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
    record = {
        "run_directory": str(resilient),
        "profile": {"confirmation_seeds": [seed], "target_labels": 1485, "ipv4_readiness_windows": 3,
                    "resume_boundaries": {str(seed): {"round": start}}},
        "predecessor": {
            "dispatcher_module": "dispatch_v13_resilient_sse", "seed": seed, "round": start,
            "parent_run_directory": str(cohort), "parent_contract_sha256": parent_sha,
            "previous_run_directory": str(cohort), "previous_contract_sha256": "a" * 64,
            "recovery_budget": {"maximum_logical_requests": 1, "maximum_physical_attempts": 1},
        },
    }
    flags = {"authorize_paid_api": True, "authorize_label_reveal": True, "authorize_cuda_training": True}
    publish(resilient, record, **flags)
    assert remote_seed(tmp_path, seed)["run"] == cohort.name
    record["predecessor"]["previous_contract_sha256"] = parent_sha
    publish(resilient, record)
    assert remote_seed(tmp_path, seed)["run"] == cohort.name
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


@pytest.mark.parametrize(("seed", "archive_count"), [(12203, 2), (157, 3)])
def test_wave03_status_uses_the_registered_gpu_scope(seed, archive_count):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_retry_successor

    scope = run_v13_retry_successor.SCOPES[(seed, 3)]
    previous = scope.archive[-1]
    assert sum(len(review.attempts) for review in scope.archive) == archive_count
    assert (getattr(prompt_pilot_status, f"RELAY_{seed}_WAVE02_CONTRACT"),
            getattr(prompt_pilot_status, f"RELAY_{seed}_WAVE02_AUTH")) == (
        previous.identity.contract, previous.identity.approval,
    )
    assert getattr(prompt_pilot_status, f"RELAY_{seed}_WAVE03_PROOF") == {
        "payload_sha256": scope.payload_sha256, "previous_request_id": previous.request_id,
        "previous_terminal_file_sha256": previous.terminal_file_sha256,
        "previous_transport_progress_file_sha256": previous.progress_file_sha256,
        "relay_cooldown_clock": scope.cooldown_field, "relay_not_before_unix": scope.not_before_unix,
        "newly_authorized_boundary_physical_attempts": 1, "execution_budget": scope.budget,
    }
    assert getattr(prompt_pilot_status, f"RELAY_{seed}_WAVE02_ATTEMPT") == {
        "attempt": 0, "attempt_id": previous.attempt_id, "job_sha256": previous.job_sha256,
        "result_sha256": previous.result_sha256, "error": previous.failure, "provider_dispatched": True,
        "ambiguous": True,
    }


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
