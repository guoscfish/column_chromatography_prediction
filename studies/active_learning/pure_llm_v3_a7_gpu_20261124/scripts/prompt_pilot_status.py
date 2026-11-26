#!/usr/bin/env python3
"""Show A7 continuation progress, GPUs and the shared Mac Key Pool."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import time


ROOT = Path("/1/column_chromatography_prediction")
RUNS = ROOT / "purellm_a7/runs"
CONTINUATION = "a7-gpt62-astra-xhigh-ssh-v5-prompt-continuation"
FULL = "a7-gpt62-astra-xhigh-ssh-v6-prompt-full"
DIRECT = "a7-gpt62-astra-xhigh-ssh-v8-direct-one"
DIRECT_887 = "a7-gpt62-astra-xhigh-ssh-v9-direct-one"
REPAIR = "a7-gpt62-astra-xhigh-ssh-v10-targeted-repair"
DIRECT_FULL = "a7-gpt62-astra-xhigh-ssh-v11-direct-full"
BACKGROUND = "a7-gpt62-astra-xhigh-background-v12"
SERIAL_IPV4 = "a7-gpt62-astra-xhigh-ipv4-v13"
IPV4_EXTENSION = f"{SERIAL_IPV4}-2357-extension-01"
IPV4_CONTINUATION = f"{SERIAL_IPV4}-2357-continuation-01"
IPV4_R2_EXTENSION = f"{SERIAL_IPV4}-2357-r2-extension-01"
IPV4_R2_CONTINUATION = f"{SERIAL_IPV4}-2357-r2-continuation-01"
IPV4_JSON_EXTENSION = f"{SERIAL_IPV4}-887-json-extension-01"
IPV4_WS_EXTENSION = f"{SERIAL_IPV4}-6101-ws-extension-01"
IPV4_EXTENSION_DISPATCHERS = {
    157: "dispatch_v13_lane_extension",
    887: "dispatch_v13_lane_extension",
    2357: "dispatch_prepared_ipv4_recovery_2357",
    6101: "dispatch_v13_lane_extension",
}
IPV4_CONTINUATION_DISPATCHERS = {
    887: "dispatch_v13_lane_continuation",
    2357: "dispatch_prepared_ipv4_continue_2357",
    6101: "dispatch_v13_lane_continuation",
}
IPV4_EXTENSION_END_ROUNDS = {157: 4, 887: 7, 2357: 2, 6101: 2}
BOUNDED_SSE_END_ROUNDS = {157: 4, 887: 7, 2357: 3, 6101: 2}
RESILIENT_SEGMENTS = ("ipv4_resilient_sse", "ipv4_resilient_failure", "ipv4_relay_retry",
                      "ipv4_relay_retry_02", "ipv4_relay_retry_03", "ipv4_cached_r7")
FORMAL_APPROVAL_FLAGS = ("authorize_paid_api", "authorize_label_reveal", "authorize_cuda_training")
RELAY_RETRY_SECONDS = 180
RELAY_157_WAVE01_CONTRACT = "51022ae153326c4d3259ebb183632564184671955e3ca86646debdfc9615862d"
RELAY_157_WAVE01_AUTH = "0ffd0471f4a5ed444f5e67a3c0bbf0e956c7c2fcaaec49a7d64840fff6a27a45"
RELAY_157_NOT_BEFORE = 1795683853.804615
RELAY_157_BOUNDARY_REQUEST = "a39894954355f629d0fb102a9bf04086d6c4235a3d54a4e90350042bbea429a8"
RELAY_157_WAVE02_CONTRACT = "a277fb5b7a0806c2f8f69443ad4b05821659783b8fa65d0b9ba6f7c814f052cd"
RELAY_157_WAVE02_AUTH = "498d339f217f9b4e1d3d4b16cd1e4602048e3ad15715bca927fe1d6c41b797aa"
RELAY_157_WAVE03_PROOF = {
    "payload_sha256": "594ff66ced4fe668cdde6bcbdcc07e9f8cf49933a45accc89ef81a4c97e42107",
    "previous_request_id": "d8fed3a9c76a3a4bc6fa27bac6a705b202b8d90d9565606ed65b1783046feb52",
    "previous_terminal_file_sha256": "0dcc718d9aa7ca2c4b6df268e04e6644c96103d9acd17c90f058ba0631b472cf",
    "previous_transport_progress_file_sha256": "b0b8b6ad3a6366329afb407ed8f6025d26983e594389dd032946ce8976d8dd7c",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": 1795684354.7931683 + RELAY_RETRY_SECONDS,
    "newly_authorized_boundary_physical_attempts": 1,
    "execution_budget": {"first_logical_requests": 33, "maximum_logical_requests": 97, "maximum_physical_attempts": 193},
}
RELAY_157_WAVE02_ATTEMPT = {
    "attempt": 0, "attempt_id": "ddd2a7ca64a74a75a20d198d72501cb1",
    "job_sha256": "dabdd32ea8889eb86fd6ff2228ab66b8b6a3dc149a7087fbc839c2205bcdadd8",
    "result_sha256": "5f0b04be5af4d1476810de3335c077034074e0e101d93f443ee9a2af3a6f2283",
    "error": "sse_frame_error", "provider_dispatched": True, "ambiguous": True,
}
RELAY_12203_WAVE01_CONTRACT = "1f1f225de4f396e639ba97bd56b2ab145307e66626b316b115ba7c82beaf11f5"
RELAY_12203_WAVE01_AUTH = "77b8f85d33eed696a3ec19147804a00a963b37bea56b3fcdb2eb5d6dfbf0f16d"
RELAY_12203_NOT_BEFORE = 1795683358.4218476
RELAY_12203_WAVE02_CONTRACT = "f373999a7c88aae8134bf2db212f6c52c8c7d75aedc1779cf5d4daede4b7a86c"
RELAY_12203_WAVE02_AUTH = "fe8cbc69384fb5561d63a91a72173882462d70ddaa87796990e8da24caae7680"
RELAY_12203_WAVE03_NOT_BEFORE = 1795683909.169368 + RELAY_RETRY_SECONDS
RELAY_12203_WAVE03_PROOF = {
    "payload_sha256": "0bd98eeac5b9b9049b7e00d7e8e439d0d1c07e7bd419370440d0259dfc91e1f5",
    "previous_request_id": "3bd1d62bce670c46ad2b10584769a0bfdce2fa9d0d9256000a3b7bbdf1441b79",
    "previous_terminal_file_sha256": "d62cef08ea21bbb26fa238eb1994093c15dc79aeacac9948f5b8f3bbb7130099",
    "previous_transport_progress_file_sha256": "cbda54406f564c8869a77f1ad09de5ada161f9b03d02cabb621bc367009bcfbd",
    "relay_cooldown_clock": "not_before_unix",
    "relay_not_before_unix": RELAY_12203_WAVE03_NOT_BEFORE,
    "newly_authorized_boundary_physical_attempts": 1,
    "execution_budget": {"first_logical_requests": 32, "maximum_logical_requests": 94, "maximum_physical_attempts": 187},
}
RELAY_12203_WAVE02_ATTEMPT = {
    "attempt": 0, "attempt_id": "ff520a3b471840e68a1f9ed34bee41f1",
    "job_sha256": "0fafc1c0274a578b04eb5a21ac9a78981d72fefecfa313455d2b814a4f3516ad",
    "result_sha256": "898014f9ac17cf60b55317aae8e4dadb56243c9ea689e1883b2c1c8044327803",
    "error": "connection_interrupted", "provider_dispatched": True, "ambiguous": True,
}
RELAY_6101_WAVE01_CONTRACT = "978ef32b62d20f330b3fd9f4fa40afc9831c098d01a1a2ee013655ae9f18ab4d"
RELAY_6101_WAVE01_AUTH = "f8dc94f7d0c9c68b9023230675fcb1b476864ada0efc136123f991c88f1736f6"
RELAY_6101_FITTED_PREFIX = {
    "round_02/source.json": "c33ecb89a166f5f2d112f2bb434fab6529d4bf108fe0d4f84b28d3b7e39648b0",
    "round_02/transition.json": "90e14a3d1ab7fd8719c68b5ec86c3d3d580cf5c733a880bc6581f371995cf54c",
    "round_01/memory.json": "989f5e746a5398725bd4bcb37ee199fab8b1f13516e0e58c6d47b66365b07cec",
}
RELAY_6101_WAVE01_ATTEMPTS = (
    {"attempt": 0, "attempt_id": "70aeee7f486449bba0e3344bece900cc",
     "job_sha256": "0f7db8326ea1e093e10a9629c42fec235ef0f30a92583041b0d90722d7d93711",
     "result_sha256": "7ad3f71eba6ad06fc442e87d5cbc65e98f99038af24635e983f021c1c07f89c9",
     "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True},
    {"attempt": 1, "attempt_id": "2547ab46579b4341bc04f26550be98fc",
     "job_sha256": "c1da929e150505f240cf13a6b590f605a6b6dc438bd23893fed4640ee7f7c70d",
     "result_sha256": "755fedb86b30db88d74fb6c9eb45faeb4dd4c017f8b3ccf5d6c4e2bff5bfccfd",
     "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True},
)
RELAY_6101_WAVE01_TERMINALS = (
    "64a972a3d630c138d0b891eead861eb1ff71ee9c3d3bcab2289022dab6b2ad1d",
    "42e00584598175ae64e7af0a569f7402a7071d7b9c73a5ab373264a91b82512f",
)
RELAY_6101_WAVE02_PROOF = {
    "logical_index": 0,
    "payload_sha256": "f3f3dac061e9baca67095771305e0ea47827c20d35e7d1018c89059172533b1a",
    "previous_request_id": "2f30189c779807c27f260f361864c4a88e777f0677b6d1cdaa81247023803303",
    "previous_terminal_file_sha256": RELAY_6101_WAVE01_TERMINALS[-1],
    "previous_transport_progress_file_sha256": "7b5b71d36c42c24bf39ea4ecce27d11e7b3c8d1dbe2ee57f963c1d97fd82c1bd",
    "relay_cooldown_clock": "failed_unix",
    "relay_not_before_unix": 1795684592.3081465 + RELAY_RETRY_SECONDS,
    "newly_authorized_boundary_physical_attempts": 1,
    "execution_budget": {"first_logical_requests": 34, "maximum_logical_requests": 100, "maximum_physical_attempts": 199},
}
CACHED_R7_PLAN_SHA = "d8acfb0d43235068f932f18b7f09c08554e811060f04878fe1a5a9afe8386d47"
PILOT_ROUNDS = 10
TARGET_ROUNDS = 36
TARGET_LABELS = 1485
ORIGINAL = "a7-gpt62-astra-xhigh-ssh-v4-pilot-3x10"
FIRST_887 = "a7-gpt62-astra-xhigh-ssh-v5-prompt-pilot-887"
PILOT = "24f01791f4e6e16e9b644dc2165fe1557eeaa7030087df2b726485ad9878e410"
SEEDS = (157, 887, 2357, 6101, 12203)
UNSTARTED_ANCHORS = {
    6101: "6959c9417166245403dad173ca82347336de53a37d643591ca449b0cf2a929c4",
    12203: "e056a9924b408e8885e7fa619f0ce63a44b8b3b47a2126d57c1fdc80e3df428d",
}
KEYS = ("key_01", "key_02", "key_03")
SHA = re.compile(r"[a-f0-9]{64}\Z")
IDENTITY = re.compile(r"[a-f0-9]{32}\Z")
FIT = re.compile(r"fit_[0-9]{3}\Z")
ROUND = re.compile(r"round_([0-9]{2})\Z")
CODE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_.]{0,79}\Z")
PROVIDER_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}\Z")
PROVIDER_DIAGNOSTICS = ("error_type", "conflicting_event_type", "provider_error_code", "provider_error_type",
                        "provider_error_param", "provider_incomplete_reason")
PHASES = {"request_dispatched", "http_phase", "http_headers", "sse_event",
          "transport_completed", "transport_failure"}
TOKEN_FIELDS = ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "total_tokens")


def read_json(path):
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def checksum(path):
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        return None


def safe_code(value):
    return value if isinstance(value, str) and CODE.fullmatch(value) else None


def provider_token(value):
    if isinstance(value, str) and PROVIDER_ID.fullmatch(value) and not value.lower().startswith("sk-"):
        return value
    return None


def pid(value):
    return value if type(value) is int and value > 1 else None


def live(proc, *, coordinator=None, run=None):
    if proc is None:
        return False
    try:
        state = Path(f"/proc/{proc}/stat").read_text().rsplit(")", 1)[1].split()
        if state[0] == "Z" or coordinator is not None and int(state[1]) != coordinator:
            return False
        if run is None:
            return True
        argv = Path(f"/proc/{proc}/cmdline").read_bytes().split(b"\x00")
        return os.fsencode(run) in argv
    except (OSError, IndexError, ValueError):
        return False


def latest_round(directory):
    for path in sorted(directory.glob("round_[0-9][0-9]"), reverse=True):
        transition = read_json(path / "transition.json")
        fit = transition.get("fit_path")
        checkpoint = transition.get("checkpoint_sha256", "")
        if (not isinstance(fit, str) or not FIT.fullmatch(fit)
                or not isinstance(checkpoint, str) or not SHA.fullmatch(checkpoint)):
            continue
        evidence = {"source_sha256": path / "source.json",
                    "validation_metrics_sha256": path / fit / "validation_metrics.json"}
        if all(transition.get(field) and checksum(file) == transition[field] for field, file in evidence.items()):
            number = int(ROUND.fullmatch(path.name)[1])
            report = read_json(evidence["validation_metrics_sha256"])
            return number, {"checkpoint_sha256": checkpoint,
                            "validation_nrmse": report.get("combined_normalized_rmse")}
    if (directory / "initial/initial.json").is_file():
        return 0, {}
    return None, {}


def _relay_wait(turn, started, terminal, contract, delay):
    if (type(delay) is not int or delay != RELAY_RETRY_SECONDS or started.get("attempt") != 0
            or terminal.get("status") != "stopped" or terminal.get("ambiguous") is not True
            or (turn / "attempt_01.started.json").exists()
            or (turn / "attempt_00.recovery.terminal.json").exists()):
        return {}
    permit = read_json(turn / "attempt_00.replay_authorization.json")
    replay = read_json(turn / "attempt_00.replay.terminal.json")
    failed_unix = seconds(terminal.get("failed_unix"))
    prior_not_before = seconds(terminal.get("not_before_unix"))
    not_before = seconds(replay.get("not_before_unix"))
    terminal_sha = record_digest(terminal)
    if (failed_unix is None or prior_not_before is None or not_before is None
            or not_before != max(prior_not_before, failed_unix + delay)
            or replay.get("sealed_relay_retry_seconds") != delay
            or replay.get("status") != "retryable" or replay.get("ambiguous") is not False
            or replay.get("error") != "authorized_replay_unknown"
            or replay.get("supersedes_sha256") != terminal_sha
            or replay.get("replay_authorization_sha256") != record_digest(permit)
            or (replay.get("attempt_id"), replay.get("request_id"), replay.get("next_attempt"))
            != (started.get("attempt_id"), started.get("request_id"), 1)
            or (permit.get("attempt_id"), permit.get("request_id"), permit.get("next_attempt"))
            != (started.get("attempt_id"), started.get("request_id"), 1)
            or permit.get("contract_sha256") != contract
            or permit.get("decision") != "authorized_replay_unknown"
            or permit.get("original_terminal_sha256") != terminal_sha
            or permit.get("bridge_result_sha256") != terminal.get("bridge_result_sha256")
            or permit.get("accept_duplicate_billing_risk") is not True):
        return {}
    return {"relay_retry_wait_seconds": max(0, math.ceil(not_before - time.time()))}


def pending_attempt(run, directory, index, *, failed=False, relay_retry=None, contract=None):
    if index is None:
        return {}
    skipped = 0
    for path in sorted((directory / f"round_{index:02}").glob("turn_[0-9][0-9]/attempt_*.started.json"), reverse=True):
        started = read_json(path)
        identity = started.get("attempt_id", "")
        if not isinstance(identity, str) or not IDENTITY.fullmatch(identity):
            continue
        stem = path.name.removesuffix(".started.json")
        recovery = path.with_name(f"{stem}.recovery.terminal.json")
        terminal = path.with_name(f"{stem}.terminal.json")
        record = read_json(recovery) or read_json(terminal)
        returned = (run / "bridge/results" / f"{identity}.json").is_file()
        if (failed and started.get("transport") == "mac_bridge" and not returned
                and not recovery.exists() and not terminal.exists() and not any(
                    (run / "bridge" / folder / f"{identity}.json").exists() for folder in ("outbox", "claims")
                )):
            skipped += 1
            continue
        status = safe_code(record.get("status")) or ("returned_from_mac" if returned else "waiting_for_mac")
        decision = read_json(path.parent / "decision.json")
        diagnosis = decision.get("diagnosis") or {}
        errors = Counter(code for entry in diagnosis.get("errors", ()) if isinstance(entry, dict)
                         if (code := safe_code(entry.get("code"))) is not None)
        return {"status": status, "failure": safe_code(record.get("error")), "ambiguous": record.get("ambiguous") is True,
                "round": index, "turn": int(path.parent.name[-2:]), "attempt": started.get("attempt"),
                "reserved_unix": seconds(started.get("started_unix")),
                "attempt_id": identity, "returned_from_mac": returned,
                **({"unsubmitted_reservations_skipped": skipped} if skipped else {}),
                **_relay_wait(path.parent, started, record, contract, relay_retry),
                "selection": safe_code(decision.get("state")), "selection_errors": dict(errors)}
    return {"status": "failed_before_bridge", "unsubmitted_reservations_skipped": skipped} if skipped else {}


def record_digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                                     allow_nan=False).encode()).hexdigest()


def fallback_contract(run):
    record = read_json(run / "formal_contract.json")
    if record.get("run_directory") != str(run):
        return None
    actual = record_digest(record)
    approval = read_json(run / "authorization.json")
    return actual if approval.get("contract_sha256") == actual else None


def dispatch_contract(run, seed):
    binding = read_json(run / "operator/dispatch_binding.json")
    sha = binding.get("contract_sha256")
    if (binding.get("seed") == seed and binding.get("run") in (str(run), run.name)
            and isinstance(sha, str) and SHA.fullmatch(sha) and fallback_contract(run) == sha):
        return sha
    return None


def resilient_contract(runs, seed, previous, previous_contract, cohort_contract, follow_on_contract):
    run = runs / f"{SERIAL_IPV4}-{seed}-resilient-sse-01"
    record = read_json(run / "formal_contract.json")
    proof = record.get("predecessor") or {}
    profile = record.get("profile", {})
    parent = runs / (IPV4_CONTINUATION if seed == 2357 else SERIAL_IPV4)
    parent_contract = follow_on_contract if seed == 2357 else cohort_contract
    start = profile.get("resume_boundaries", {}).get(str(seed), {}).get("round")
    approval = read_json(run / "authorization.json")
    if (not previous_contract or not parent_contract
            or proof.get("dispatcher_module") != "dispatch_v13_resilient_sse"
            or proof.get("previous_run_directory") != str(previous)
            or proof.get("previous_contract_sha256") != previous_contract
            or proof.get("parent_run_directory") != str(parent)
            or proof.get("parent_contract_sha256") != parent_contract
            or proof.get("seed") != seed or proof.get("round") != start or type(start) is not int
            or not 0 <= start < TARGET_ROUNDS
            or proof.get("recovery_budget") != {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
            or profile.get("confirmation_seeds") != [seed] or profile.get("target_labels") != TARGET_LABELS
            or profile.get("ipv4_readiness_windows") != 3
            or any(approval.get(name) is not True for name in FORMAL_APPROVAL_FLAGS)):
        return None
    if contract := fallback_contract(run):
        return run, contract, start, profile["ipv4_readiness_windows"]
    return None


def resilient_failure_contract(runs, seed, previous, previous_contract, recovery_round):
    run = runs / f"{SERIAL_IPV4}-{seed}-resilient-failure-01"
    record = read_json(run / "formal_contract.json")
    proof = record.get("predecessor") or {}
    profile = record.get("profile", {})
    previous_record = read_json(previous / "formal_contract.json")
    previous_approval = read_json(previous / "authorization.json")
    approval = read_json(run / "authorization.json")
    if (previous.name != f"{SERIAL_IPV4}-{seed}-resilient-sse-01"
            or previous_approval.get("contract_sha256") != previous_contract
            or proof.get("dispatcher_module") != "dispatch_v13_resilient_failure_recovery"
            or proof.get("previous_run_directory") != str(previous)
            or proof.get("previous_contract_sha256") != previous_contract
            or proof.get("previous_authorization_sha256") != record_digest(previous_approval)
            or proof.get("seed") != seed or proof.get("round") != recovery_round
            or proof.get("recovery_budget") != {"maximum_logical_requests": 1, "maximum_physical_attempts": 1}
            or profile != previous_record.get("profile")
            or profile.get("confirmation_seeds") != [seed] or profile.get("target_labels") != TARGET_LABELS
            or any(approval.get(name) is not True for name in FORMAL_APPROVAL_FLAGS)):
        return None
    if contract := fallback_contract(run):
        return run, contract
    return None


def _signed_recovery_successor(run, seed, previous, previous_contract, dispatcher, physical_attempts, *,
                               previous_suffix="resilient-failure-01"):
    record = read_json(run / "formal_contract.json")
    proof = record.get("predecessor") or {}
    profile = record.get("profile", {})
    prior_record = read_json(previous / "formal_contract.json")
    prior_approval = read_json(previous / "authorization.json")
    approval = read_json(run / "authorization.json")
    boundary = profile.get("resume_boundaries", {}).get(str(seed), {})
    start = boundary.get("round")
    budget = proof.get("recovery_budget")
    if (previous.name != f"{SERIAL_IPV4}-{seed}-{previous_suffix}"
            or prior_approval.get("contract_sha256") != previous_contract
            or proof.get("dispatcher_module") != dispatcher
            or proof.get("previous_run_directory") != str(previous)
            or proof.get("previous_contract_sha256") != previous_contract
            or proof.get("previous_authorization_sha256") != record_digest(prior_approval)
            or proof.get("seed") != seed or proof.get("round") != start or type(start) is not int
            or not 0 <= start < TARGET_ROUNDS
            or budget != {"maximum_logical_requests": 1, "maximum_physical_attempts": physical_attempts}
            or any(type(value) is not int for value in budget.values())
            or profile.get("confirmation_seeds") != [seed] or profile.get("target_labels") != TARGET_LABELS
            or any(approval.get(name) is not True for name in FORMAL_APPROVAL_FLAGS)):
        return None
    prior_round, _ = latest_round(previous / f"seed_{seed}")
    if prior_round is None:
        prior_round = prior_record.get("profile", {}).get("resume_boundaries", {}).get(str(seed), {}).get("round")
    if prior_round != start:
        return None
    if contract := fallback_contract(run):
        return contract, start, profile, proof
    return None


def relay_retry_contract(runs, seed, previous, previous_contract):
    run = runs / f"{SERIAL_IPV4}-{seed}-relay-retry-01"
    if candidate := _signed_recovery_successor(run, seed, previous, previous_contract, "dispatch_v13_relay_retry", 2):
        contract, start, profile, _ = candidate
        if (type(profile.get("sealed_relay_retry_seconds")) is int
                and profile["sealed_relay_retry_seconds"] == RELAY_RETRY_SECONDS
                and profile.get("audited_future_transport_replay") is True
                and profile.get("replay_eligible_from_round") == start
                and profile.get("ipv4_readiness_windows") == 3):
            return run, contract, start, profile["ipv4_readiness_windows"]
    return None


def relay_retry_02_contract(runs, seed, previous, previous_contract):
    if seed == 6101:
        return relay_retry_02_6101_contract(runs, previous, previous_contract)
    if seed == 157:
        signed_contract, signed_approval = RELAY_157_WAVE01_CONTRACT, RELAY_157_WAVE01_AUTH
        deadline, boundary_round = RELAY_157_NOT_BEFORE, 3
        boundary_proof = {
            "prior_boundary_budget": {"physical_authorized": 2, "physical_already_used": 2, "physical_remaining": 0},
            "newly_authorized_boundary_physical_attempts": 1,
            "previous_request_id": RELAY_157_BOUNDARY_REQUEST,
            "execution_budget": {"first_logical_requests": 33, "maximum_logical_requests": 97,
                                 "maximum_physical_attempts": 193},
        }
    elif seed == 12203:
        signed_contract, signed_approval = RELAY_12203_WAVE01_CONTRACT, RELAY_12203_WAVE01_AUTH
        deadline, boundary_round = RELAY_12203_NOT_BEFORE, 4
        boundary_proof = {
            "inherited_boundary_budget": {"physical_authorized": 2, "physical_already_used": 1, "physical_remaining": 1},
        }
    else:
        return None
    if previous_contract != signed_contract:
        return None
    run = runs / f"{SERIAL_IPV4}-{seed}-relay-retry-02"
    candidate = _signed_recovery_successor(run, seed, previous, previous_contract,
                                           f"dispatch_v13_{seed}_relay_retry", 1, previous_suffix="relay-retry-01")
    if candidate is None:
        return None
    contract, start, profile, proof = candidate
    not_before = seconds(proof.get("relay_not_before_unix"))
    if (start == boundary_round and proof.get("previous_authorization_sha256") == signed_approval
            and record_digest({name: proof.get(name) for name in boundary_proof}) == record_digest(boundary_proof)
            and not_before == deadline and profile.get("audited_future_transport_replay") is True
            and profile.get("replay_eligible_from_round") == start + 1
            and type(profile.get("sealed_relay_retry_seconds")) is int
            and profile["sealed_relay_retry_seconds"] == RELAY_RETRY_SECONDS
            and profile.get("ipv4_readiness_windows") == 3):
        return run, contract, start, profile["ipv4_readiness_windows"], not_before
    return None


def relay_retry_02_6101_contract(runs, previous, previous_contract):
    seed = 6101
    if previous_contract != RELAY_6101_WAVE01_CONTRACT:
        return None
    run = runs / f"{SERIAL_IPV4}-{seed}-relay-retry-02"
    candidate = _signed_recovery_successor(run, seed, previous, previous_contract, "dispatch_v13_retry_successor", 1,
                                           previous_suffix="relay-retry-01")
    if candidate is None or fallback_contract(previous) != previous_contract:
        return None
    contract, start, profile, proof = candidate
    old_profile = read_json(previous / "formal_contract.json").get("profile", {})
    boundaries = profile.get("resume_boundaries", {})
    exact = {**RELAY_6101_WAVE02_PROOF, "previous_authorization_sha256": RELAY_6101_WAVE01_AUTH,
             "reviewed_fitted_prefix_files": RELAY_6101_FITTED_PREFIX}
    if (start != 2 or old_profile.get("resume_boundaries", {}).get(str(seed), {}).get("round") != 1
            or set(boundaries) != {str(seed)} or profile != {**old_profile, "resume_boundaries": boundaries,
                                                              "replay_eligible_from_round": 3}
            or profile.get("audited_future_transport_replay") is not True
            or type(profile.get("sealed_relay_retry_seconds")) is not int
            or profile["sealed_relay_retry_seconds"] != RELAY_RETRY_SECONDS or profile.get("ipv4_readiness_windows") != 3
            or record_digest({key: proof.get(key) for key in exact}) != record_digest(exact)):
        return None
    directory = previous / f"seed_{seed}"
    if any(checksum(directory / name) != sha for name, sha in RELAY_6101_FITTED_PREFIX.items()):
        return None
    receipts = proof.get("previous_attempts")
    if type(receipts) is not list or len(receipts) != len(RELAY_6101_WAVE01_ATTEMPTS):
        return None
    turn = directory / "round_02/turn_00"
    for receipt, expected, sha in zip(receipts, RELAY_6101_WAVE01_ATTEMPTS, RELAY_6101_WAVE01_TERMINALS, strict=True):
        terminal = turn / f"attempt_{expected['attempt']:02d}.terminal.json"
        if (type(receipt) is not dict or checksum(terminal) != sha
                or receipt.get("terminal_sha256") != record_digest(read_json(terminal))
                or record_digest({key: receipt.get(key) for key in expected}) != record_digest(expected)):
            return None
    chain = [{"contract_sha256": previous_contract, "authorization_sha256": RELAY_6101_WAVE01_AUTH,
              "run_directory": str(previous), "request_id": proof["previous_request_id"],
              "terminal_file_sha256": proof["previous_terminal_file_sha256"],
              "progress_file_sha256": proof["previous_transport_progress_file_sha256"], "previous_attempts": receipts}]
    if proof.get("reviewed_retry_chain") != chain:
        return None
    return run, contract, start, profile["ipv4_readiness_windows"], proof["relay_not_before_unix"]


def relay_retry_03_contract(runs, seed, previous, previous_contract):
    if seed == 157:
        wave01_contract, wave01_auth = RELAY_157_WAVE01_CONTRACT, RELAY_157_WAVE01_AUTH
        wave02_contract, wave02_auth = RELAY_157_WAVE02_CONTRACT, RELAY_157_WAVE02_AUTH
        wave03_proof, wave02_attempt = RELAY_157_WAVE03_PROOF, RELAY_157_WAVE02_ATTEMPT
        expected_round, wave01_count = 3, 2
    elif seed == 12203:
        wave01_contract, wave01_auth = RELAY_12203_WAVE01_CONTRACT, RELAY_12203_WAVE01_AUTH
        wave02_contract, wave02_auth = RELAY_12203_WAVE02_CONTRACT, RELAY_12203_WAVE02_AUTH
        wave03_proof, wave02_attempt = RELAY_12203_WAVE03_PROOF, RELAY_12203_WAVE02_ATTEMPT
        expected_round, wave01_count = 4, 1
    else:
        return None
    if previous_contract != wave02_contract:
        return None
    run = runs / f"{SERIAL_IPV4}-{seed}-relay-retry-03"
    candidate = _signed_recovery_successor(run, seed, previous, previous_contract, "dispatch_v13_retry_successor", 1,
                                           previous_suffix="relay-retry-02")
    if candidate is None or fallback_contract(previous) != previous_contract:
        return None
    contract, start, profile, proof = candidate
    previous_record = read_json(previous / "formal_contract.json")
    exact = {**wave03_proof, "previous_authorization_sha256": wave02_auth}
    if (start != expected_round or profile != previous_record.get("profile")
            or profile.get("replay_eligible_from_round") != start + 1
            or profile.get("audited_future_transport_replay") is not True
            or type(profile.get("sealed_relay_retry_seconds")) is not int
            or profile["sealed_relay_retry_seconds"] != RELAY_RETRY_SECONDS or profile.get("ipv4_readiness_windows") != 3
            or record_digest({key: proof.get(key) for key in exact}) != record_digest(exact)):
        return None
    previous_attempts = proof.get("previous_attempts")
    if type(previous_attempts) is not list or len(previous_attempts) != 1 or type(previous_attempts[0]) is not dict:
        return None
    attempt = previous_attempts[0]
    if (not isinstance(attempt.get("terminal_sha256"), str) or not SHA.fullmatch(attempt["terminal_sha256"])
            or record_digest({key: attempt.get(key) for key in wave02_attempt}) != record_digest(wave02_attempt)):
        return None
    old = previous_record.get("predecessor") or {}
    old_attempts = old.get("previous_attempts")
    terminals = old.get("previous_terminal_file_sha256")
    traces = old.get("previous_transport_progress_file_sha256")
    if wave01_count == 1:
        terminals, traces = [terminals], [traces]
    if (type(old_attempts) is not list or len(old_attempts) != wave01_count
            or any(type(item) is not dict or type(item.get("attempt")) is not int or item["attempt"] != index
                   for index, item in enumerate(old_attempts))
            or any(type(hashes) is not list or len(hashes) != wave01_count
                   or any(not isinstance(value, str) or not SHA.fullmatch(value) for value in hashes)
                   for hashes in (terminals, traces))):
        return None
    chain = [
        {"contract_sha256": wave01_contract, "authorization_sha256": wave01_auth,
         "run_directory": str(runs / f"{SERIAL_IPV4}-{seed}-relay-retry-01"),
         "request_id": old.get("previous_request_id"), "terminal_file_sha256": terminals[-1],
         "progress_file_sha256": traces[-1], "previous_attempts": old_attempts},
        {"contract_sha256": previous_contract, "authorization_sha256": wave02_auth,
         "run_directory": str(previous), "request_id": proof["previous_request_id"],
         "terminal_file_sha256": proof["previous_terminal_file_sha256"],
         "progress_file_sha256": proof["previous_transport_progress_file_sha256"], "previous_attempts": previous_attempts},
    ]
    if (not isinstance(chain[0]["request_id"], str) or not SHA.fullmatch(chain[0]["request_id"])
            or proof.get("reviewed_retry_chain") != chain):
        return None
    return run, contract, start, profile["ipv4_readiness_windows"], proof["relay_not_before_unix"]


def cached_r7_contract(runs, seed, previous, previous_contract):
    if seed != 887:
        return None
    run = runs / f"{SERIAL_IPV4}-887-cached-r7-01"
    if candidate := _signed_recovery_successor(run, seed, previous, previous_contract, "dispatch_v13_887_cached_r7", 0):
        contract, start, profile, proof = candidate
        cached = proof.get("cached_import") or {}
        generation = read_json(run / "coordinator_generation.json")
        if (start == 7 and profile.get("replay_eligible_from_round") == 8
                and profile.get("audited_future_transport_replay") is True
                and profile.get("ipv4_readiness_windows") == 3
                and cached.get("provider_post") is False and cached.get("normalized_plan_sha256") == CACHED_R7_PLAN_SHA
                and read_json(run / "cached_response_import.json") == cached
                and pid(generation.get("pid"))
                and isinstance(identifier := generation.get("id"), str) and IDENTITY.fullmatch(identifier)):
            return run, contract, start, profile["ipv4_readiness_windows"]
    return None


def remote_seed(runs, seed):
    cohort_contract = None
    for name, mode, dispatcher, capability in (
        (SERIAL_IPV4, "ipv4", "dispatch_prepared_ipv4", "gated_ipv4"),
        (BACKGROUND, "background", "dispatch_prepared_background", "verified_route"),
    ):
        cohort = runs / name
        if candidate := fallback_contract(cohort):
            record = read_json(cohort / "formal_contract.json")
            profile = record.get("profile", {})
            if (tuple(profile.get("confirmation_seeds", ())) == SEEDS
                    and profile.get("provider_capability") == capability
                    and (record.get("predecessor") or {}).get("dispatcher_module") == dispatcher):
                cohort_contract = candidate
                break
    direct_full = runs / f"{DIRECT_FULL}-{seed}"
    full = runs / f"{FULL}-{seed}"
    direct = runs / f"{DIRECT_887 if seed == 887 else DIRECT}-{seed}"
    continuation = runs / f"{CONTINUATION}-{seed}"
    fallback = runs / (FIRST_887 if seed == 887 else ORIGINAL)
    repair = runs / f"{REPAIR}-{seed}"
    direct_full_contract, full_contract, repair_contract, direct_contract, continuation_contract = (
        dispatch_contract(run, seed) for run in (direct_full, full, repair, direct, continuation))
    extension = runs / f"{SERIAL_IPV4}-{seed}-extension-01"
    extension_dispatcher = IPV4_EXTENSION_DISPATCHERS.get(seed)
    extension_record = read_json(extension / "formal_contract.json") if extension_dispatcher else {}
    extension_parent = extension_record.get("predecessor") or {}
    extension_contract = (
        fallback_contract(extension) if cohort_contract and mode == "ipv4"
        and extension_parent.get("parent_contract_sha256") == cohort_contract
        and extension_parent.get("dispatcher_module") == extension_dispatcher
        and extension_record.get("profile", {}).get("confirmation_seeds") == [seed] else None
    )
    follow_on = runs / f"{SERIAL_IPV4}-{seed}-continuation-01"
    follow_on_record = read_json(follow_on / "formal_contract.json") if extension_contract else {}
    follow_on_parent = follow_on_record.get("predecessor") or {}
    follow_on_contract = (
        fallback_contract(follow_on) if extension_contract
        and follow_on_parent.get("extension_contract_sha256") == extension_contract
        and follow_on_parent.get("parent_contract_sha256") == cohort_contract
        and follow_on_parent.get("dispatcher_module") == IPV4_CONTINUATION_DISPATCHERS.get(seed)
        and follow_on_record.get("profile", {}).get("confirmation_seeds") == [seed] else None
    )
    r2_extension = runs / IPV4_R2_EXTENSION
    r2_record = read_json(r2_extension / "formal_contract.json") if seed == 2357 and follow_on_contract else {}
    r2_parent = r2_record.get("predecessor") or {}
    r2_contract = (
        fallback_contract(r2_extension) if follow_on_contract
        and r2_parent.get("parent_contract_sha256") == follow_on_contract
        and r2_parent.get("dispatcher_module") == "dispatch_v13_2357_r2_sse_extension"
        and r2_record.get("profile", {}).get("confirmation_seeds") == [seed] else None
    )
    r2_continuation = runs / IPV4_R2_CONTINUATION
    r2_follow_record = read_json(r2_continuation / "formal_contract.json") if r2_contract else {}
    r2_follow_prior = r2_follow_record.get("predecessor") or {}
    r2_follow_contract = (
        fallback_contract(r2_continuation) if r2_contract
        and r2_follow_prior.get("parent_contract_sha256") == follow_on_contract
        and r2_follow_prior.get("extension_contract_sha256") == r2_contract
        and r2_follow_prior.get("dispatcher_module") == "dispatch_v13_2357_r2_continuation"
        and r2_follow_record.get("profile", {}).get("confirmation_seeds") == [seed] else None
    )
    extra_name, extra_dispatcher, extra_segment = {
        887: (IPV4_JSON_EXTENSION, "dispatch_v13_887_json_extension", "ipv4_json_extension"),
        6101: (IPV4_WS_EXTENSION, "dispatch_v13_6101_ws_extension", "ipv4_ws_extension"),
    }.get(seed, ("", "", ""))
    extra = runs / extra_name
    extra_record = read_json(extra / "formal_contract.json") if extra_name and extension_contract else {}
    extra_prior = extra_record.get("predecessor") or {}
    extra_contract = (
        fallback_contract(extra) if extension_contract
        and extra_prior.get("parent_contract_sha256") == cohort_contract
        and extra_prior.get("previous_contract_sha256") == extension_contract
        and extra_prior.get("dispatcher_module") == extra_dispatcher
        and extra_record.get("profile", {}).get("confirmation_seeds") == [seed] else None
    )
    if extra_contract:
        run, contract, segment = extra, extra_contract, extra_segment
        ancestors = (extension, cohort, direct_full, continuation, fallback)
    elif r2_follow_contract:
        run, contract, segment = r2_continuation, r2_follow_contract, "ipv4_r2_continuation"
        ancestors = (r2_extension, follow_on, extension, cohort, direct_full, continuation, fallback)
    elif r2_contract:
        run, contract, segment = r2_extension, r2_contract, "ipv4_r2_extension"
        ancestors = (follow_on, extension, cohort, direct_full, continuation, fallback)
    elif follow_on_contract:
        run, contract, segment = follow_on, follow_on_contract, "ipv4_continuation"
        ancestors = (extension, cohort, direct_full, continuation, fallback)
    elif extension_contract:
        run, contract, segment = extension, extension_contract, "ipv4_extension"
        ancestors = (cohort, direct_full, continuation, fallback)
    elif cohort_contract:
        run, contract, segment = cohort, cohort_contract, mode
        ancestors = tuple(path for path, approved in (
            (direct_full, direct_full_contract), (repair, repair_contract), (direct, direct_contract),
            (continuation, continuation_contract), (fallback, fallback_contract(fallback)),
        ) if approved)
    elif direct_full_contract:
        run, contract, segment = direct_full, direct_full_contract, "v11"
        ancestors = tuple(path for path, approved in (
            (repair, repair_contract), (direct, direct_contract), (continuation, continuation_contract),
            (fallback, fallback_contract(fallback)),
        ) if approved)
    elif full_contract:
        run, contract, segment = full, full_contract, "v6"
        ancestors = (repair, direct, continuation, fallback)
    elif repair_contract:
        run, contract, segment = repair, repair_contract, "repair"
        ancestors = (direct, continuation, fallback)
    elif direct_contract:
        run, contract, segment = direct, direct_contract, "direct"
        ancestors = (continuation, fallback)
    elif continuation_contract:
        run, contract, segment = continuation, continuation_contract, "v5"
        ancestors = (fallback,)
    else:
        run, contract, segment = fallback, fallback_contract(fallback), "historical"
        ancestors = ()
    bounded = runs / f"{SERIAL_IPV4}-{seed}-bounded-sse-01"
    bounded_record = read_json(bounded / "formal_contract.json") if contract and seed in BOUNDED_SSE_END_ROUNDS else {}
    bounded_prior = bounded_record.get("predecessor") or {}
    if (bounded_prior.get("previous_contract_sha256") == contract
            and bounded_prior.get("previous_run_directory") == str(run)
            and bounded_prior.get("dispatcher_module") == "dispatch_v13_bounded_sse_recovery"
            and bounded_record.get("profile", {}).get("confirmation_seeds") == [seed]
            and (bounded_contract := fallback_contract(bounded))):
        ancestors = (run, *ancestors)
        run, contract, segment = bounded, bounded_contract, "ipv4_bounded_sse"
    recovery_round, readiness_windows = None, 0
    relay_retry = relay_boundary_not_before = None
    if mode == "ipv4" and (resilient := resilient_contract(
        runs, seed, run, contract, cohort_contract, follow_on_contract,
    )):
        ancestors = (run, *ancestors)
        run, contract, recovery_round, readiness_windows = resilient
        segment = "ipv4_resilient_sse"
        if recovered := resilient_failure_contract(runs, seed, run, contract, recovery_round):
            ancestors = (run, *ancestors)
            run, contract = recovered
            segment = "ipv4_resilient_failure"
            if cached := cached_r7_contract(runs, seed, run, contract):
                ancestors = (run, *ancestors)
                run, contract, recovery_round, readiness_windows = cached
                segment = "ipv4_cached_r7"
            elif relay := relay_retry_contract(runs, seed, run, contract):
                ancestors = (run, *ancestors)
                run, contract, recovery_round, readiness_windows = relay
                segment = "ipv4_relay_retry"
                relay_retry = RELAY_RETRY_SECONDS
                if wave02 := relay_retry_02_contract(runs, seed, run, contract):
                    ancestors = (run, *ancestors)
                    run, contract, recovery_round, readiness_windows, relay_boundary_not_before = wave02
                    segment = "ipv4_relay_retry_02"
                    if wave03 := relay_retry_03_contract(runs, seed, run, contract):
                        ancestors = (run, *ancestors)
                        run, contract, recovery_round, readiness_windows, relay_boundary_not_before = wave03
                        segment = "ipv4_relay_retry_03"
    if seed in UNSTARTED_ANCHORS and not cohort_contract and not any(
        (path / f"seed_{seed}").exists() for path in (run, *ancestors)
    ):
        anchor = (runs.parent.parent / "formal_cuda/row4g_b32_20261124/trajectories"
                  / f"random-{seed}/rounds/random/round_00/model/best.pt")
        expected = UNSTARTED_ANCHORS[seed]
        verified = checksum(anchor) == expected
        return {"seed": seed, "run": None, "contract_sha256": None, "segment": "anchor",
                "continuation": False, "target_rounds": TARGET_ROUNDS, "target_labels": TARGET_LABELS,
                "completed_rounds": 0 if verified else None, "labels": 333 if verified else None,
                "phase": "l333_anchor_verified" if verified else "l333_anchor_unverified",
                "worker_pid": None, "worker_active": False, "worker_error": None, "coordinator_pid": None,
                "coordinator_active": False, "gpu": None, "api": {},
                "checkpoint_sha256": expected if verified else None}
    directory = run / f"seed_{seed}"
    latest, metrics = latest_round(directory)
    for ancestor in ancestors:
        if latest is not None:
            break
        latest, metrics = latest_round(ancestor / f"seed_{seed}")
    if cohort_contract and latest is None:
        lane = record["predecessor"].get("lanes", {}).get(str(seed), {})
        start, checkpoint = lane.get("start_round"), lane.get("checkpoint_sha256")
        if type(start) is int and 0 <= start < TARGET_ROUNDS and isinstance(checkpoint, str) and SHA.fullmatch(checkpoint):
            latest, metrics = start, {"checkpoint_sha256": checkpoint}
    worker = read_json(run / "workers" / f"seed_{seed}.json")
    coordinator = pid(read_json(run / "coordinator_generation.json").get("pid"))
    coordinator_live = live(coordinator, run=run)
    worker_pid = pid(worker.get("pid"))
    worker_live = coordinator_live and (worker_pid == coordinator or live(worker_pid, coordinator=coordinator))
    paused = (run / "pause.request").is_file() or (directory / "pause.request").is_file()
    gpu = read_json(run / "gpu" / f"seed_{seed}.json")
    phase = safe_code(worker.get("status")) or "waiting_to_start"
    api = pending_attempt(run, directory, latest,
                          failed=segment in RESILIENT_SEGMENTS and phase == "failed" and not worker_live,
                          relay_retry=relay_retry, contract=contract)
    if segment == "ipv4_cached_r7" and latest == recovery_round:
        if api.get("status") == "waiting_for_mac":
            api["status"] = "validating_cached_response"
        api["cached_import"] = True
    if (relay_boundary_not_before is not None and latest == recovery_round
            and not api and relay_boundary_not_before > time.time()):
        api = {"status": "relay_boundary_cooldown",
               "relay_retry_wait_seconds": max(0, math.ceil(relay_boundary_not_before - time.time()))}
    if latest == TARGET_ROUNDS:
        phase = "complete"
    elif segment in ("ipv4_extension", "ipv4_json_extension", "ipv4_ws_extension") and latest == IPV4_EXTENSION_END_ROUNDS[seed] and (
        run / "extension_completion.json"
    ).is_file():
        phase = "transport_extension_complete"
    elif segment == "ipv4_bounded_sse" and latest == BOUNDED_SSE_END_ROUNDS[seed] and (run / "extension_completion.json").is_file():
        phase = "transport_extension_complete"
    elif segment == "ipv4_r2_extension" and latest == 3 and (run / "extension_completion.json").is_file():
        phase = "transport_extension_complete"
    elif latest == PILOT_ROUNDS and segment not in ("v6", "v11", "background", "ipv4", "ipv4_continuation",
                                                     "ipv4_r2_continuation", *RESILIENT_SEGMENTS):
        phase = "awaiting_v6_handoff"
    elif paused:
        phase = "paused"
    elif latest == 5 and segment == "repair":
        phase = "targeted_repair_round_complete"
    elif latest == (5 if seed == 887 else 2) and segment == "direct":
        phase = "direct_round_complete"
    elif phase == "failed" and api.get("selection") in {"rejected", "malformed_complete_response", "patch_rejected"}:
        phase = "selection_rejected"
    elif phase in {"paused", "failed", "bridge_pending"}:
        pass
    elif worker_live and gpu.get("stage") in ("waiting", "cuda"):
        phase = f"gpu_{gpu['stage']}"
    elif worker_live and "relay_retry_wait_seconds" in api:
        phase = "relay_retry_cooldown" if api["relay_retry_wait_seconds"] else "awaiting_relay_retry"
    elif api.get("status") in {"waiting_for_mac", "returned_from_mac"}:
        phase = api["status"]
    elif worker_live and api.get("status") == "completed":
        phase = "selection_or_repair"
    elif not worker_live and phase == "initializing":
        phase = "worker_stopped"
    return {"seed": seed, "run": run.name, "contract_sha256": contract, "segment": segment,
            "continuation": segment in ("ipv4", "ipv4_extension", "ipv4_continuation", "ipv4_r2_extension",
                                        "ipv4_r2_continuation", "ipv4_json_extension", "ipv4_ws_extension",
                                        "ipv4_bounded_sse", *RESILIENT_SEGMENTS, "background", "v11", "v6", "v5",
                                        "direct", "repair"),
            **({"recovery_round": recovery_round, "readiness_windows": readiness_windows}
               if segment in RESILIENT_SEGMENTS else {}),
            **({"sealed_relay_retry_seconds": relay_retry} if relay_retry is not None else {}),
            **({"relay_not_before_unix": relay_boundary_not_before} if relay_boundary_not_before is not None else {}),
            "target_rounds": TARGET_ROUNDS, "target_labels": TARGET_LABELS,
            "completed_rounds": latest, "labels": 333 + latest * 32 if latest is not None else None,
            "phase": phase, "worker_pid": worker_pid, "worker_active": worker_live,
            "worker_error": safe_code(worker.get("error_type")), "coordinator_pid": coordinator,
            "coordinator_active": coordinator_live, "gpu": gpu.get("gpu"), "api": api, **metrics}


def command_lines(command):
    try:
        return subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True, timeout=8).splitlines()
    except (OSError, subprocess.SubprocessError):
        return None


def remote_report(runs=RUNS):
    return {"seeds": [remote_seed(runs, seed) for seed in SEEDS], "gpus": command_lines([
        "nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits",
    ]), "gpu_processes": command_lines([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits",
    ])}


def usage(data):
    return {key: data[key] for key in TOKEN_FIELDS if isinstance(data, dict) and type(data.get(key)) is int}


def seconds(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def mac_start_time(marker):
    try:
        parsed = datetime.fromisoformat(marker["started_utc"])
        return parsed.timestamp() if parsed.tzinfo is not None else None
    except (KeyError, TypeError, ValueError, OSError, OverflowError):
        return None


def route_readiness(spool, attempt, identity, seed, active, windows):
    """Read only receipts for the exact current reservation, before any model start."""
    if windows != 3:
        return {}
    job = read_json(spool / "jobs" / f"{identity}.json")
    job_sha = job.get("job_sha256")
    request = job.get("request") or {}
    binding = request.get("binding") or {}
    if ((job.get("attempt_id"), binding.get("seed"), binding.get("round"), request.get("logical_index"))
            != (identity, seed, active.get("round"), active.get("turn"))
            or not isinstance(job_sha, str) or not SHA.fullmatch(job_sha)):
        return {}
    files = sorted(attempt.glob("readiness_[0-9][0-9].json"))
    if not files or len(files) > windows:
        return {}
    for index, path in enumerate(files, 1):
        latest = read_json(path)
        if (path.name != f"readiness_{index:02d}.json"
                or (latest.get("attempt_id"), latest.get("job_sha256"), latest.get("window")) != (identity, job_sha, index)
                or seconds(latest.get("retry_unix")) is None):
            return {}
    ready = {"readiness_windows_used": len(files), "readiness_windows": windows}
    if (attempt / "started.json").exists():
        return ready
    result = read_json(spool / "results" / f"{identity}.json")
    ack = read_json(spool / "delivered" / f"{identity}.json")
    result_sha = result.get("result_sha256")
    if (result.get("attempt_id") == identity and result.get("job_sha256") == job_sha
            and result.get("provider_dispatched") is False and (result.get("outcome") or {}).get("status") == "failed"
            and isinstance(result_sha, str) and SHA.fullmatch(result_sha)):
        return {**ready, "phase": "transport_failure", "provider_dispatched": False, "sealed": True,
                "delivered": ack == {"attempt_id": identity, "result_sha256": result_sha}}
    retry = {"readiness_retry_seconds": max(0, round(latest["retry_unix"] - time.time()))} if len(files) < windows else {}
    return {**ready, **retry, "phase": "waiting_for_ipv4_route", "provider_dispatched": False}


def local_progress(spool, seed, active, *, track_queue=False, readiness_windows=0):
    """Report only the Mac attempt selected by the current GPFS ledger."""
    identity = active.get("attempt_id")
    if not isinstance(identity, str) or not IDENTITY.fullmatch(identity):
        return {}
    attempt = spool / "attempts" / identity
    path = attempt / "provider_progress.json"
    status = read_json(path)
    marker = read_json(attempt / "started.json")
    now = time.time()
    start = mac_start_time(marker) if marker.get("attempt_id") == identity else None
    readiness = route_readiness(spool, attempt, identity, seed, active, readiness_windows)
    timing = {}
    if track_queue and (reserved := seconds(active.get("reserved_unix"))) is not None:
        if start is not None:
            timing["dispatch_wait_seconds"] = seconds(start - reserved)
        elif not marker and active.get("status") == "waiting_for_mac":
            timing = {"phase": "queued_for_dispatch", "dispatch_wait_seconds": seconds(now - reserved)}
    if (status.get("seed") != seed or status.get("attempt_id") != identity or marker.get("attempt_id") != identity
            or not marker.get("job_sha256") or status.get("job_sha256") != marker["job_sha256"]
            or status.get("round") != active.get("round") or status.get("logical_index") != active.get("turn")
            or status.get("phase") not in PHASES):
        return {**timing, **readiness}
    try:
        updated = path.stat().st_mtime
    except OSError:
        return {**timing, **readiness}
    result = read_json(spool / "results" / f"{identity}.json")
    ack = read_json(spool / "delivered" / f"{identity}.json")
    sealed = result.get("attempt_id") == identity and result.get("job_sha256") == marker["job_sha256"]
    delivered = sealed and ack == {"attempt_id": identity, "result_sha256": result.get("result_sha256")}
    events = status.get("sse_milestones", {})
    if not isinstance(events, dict):
        events = {}
    first_text_event = events.get("response.output_text.delta")
    first_text = first_text_event.get("elapsed_seconds") if isinstance(first_text_event, dict) else None
    terminal_event = events.get("response.completed")
    terminal_time = terminal_event.get("elapsed_seconds") if isinstance(terminal_event, dict) else None
    tokens = {field: value for field in ("response_id", "conflicting_response_id", "provider_request_id",
                                         *PROVIDER_DIAGNOSTICS) if (value := provider_token(status.get(field))) is not None}
    diagnostics = {}
    provider_status = status.get("provider_error_status")
    if type(provider_status) is int and 400 <= provider_status <= 599:
        diagnostics["provider_error_status"] = provider_status
    alias = status.get("response_id_matches_transport_alias")
    if type(alias) is bool:
        diagnostics["response_id_matches_transport_alias"] = alias
    fields = ("round", "logical_index", "http_status", "elapsed_seconds", "sse_event_count")
    progress = {field: status[field] for field in fields if type(status.get(field)) in (int, float)}
    elapsed = seconds(status.get("elapsed_seconds"))
    if not sealed and status["phase"] not in ("transport_completed", "transport_failure"):
        if start is not None and (running := seconds(now - start)) is not None:
            elapsed = max(elapsed or 0, running)
    dispatch = {"provider_dispatched": result["provider_dispatched"]} if (
        sealed and type(result.get("provider_dispatched")) is bool
    ) else {}
    return {**timing, **readiness, **progress, **dispatch, "phase": status["phase"], "sealed": sealed, "delivered": delivered,
            "progress_age_seconds": max(0, round(now - updated)),
            "event_type": safe_code(status.get("event_type")), "usage": usage(status.get("usage")),
            **tokens, **diagnostics,
            "provider_elapsed_seconds": elapsed,
            "first_output_seconds": seconds(first_text),
            "last_sse_seconds": seconds(status.get("last_sse_seconds")),
            "last_sse_activity_seconds": seconds(status.get("last_sse_activity_seconds")),
            "last_sse_interval_seconds": seconds(status.get("last_sse_interval_seconds")),
            "complete_event_seen": (status.get("complete_terminal_captured") is True or "response.completed" in events
                                    or status.get("event_type") in {"response.completed", "background.completed"}),
            "complete_event_seconds": seconds(terminal_time),
            "raw_response_saved": (attempt / "raw_provider_response.json").is_file()
                                  and (attempt / "provider_capture_receipt.json").is_file()}


def dispatcher_pid(seed, segment):
    cohort_services = {"background": "a7-background-v12", "ipv4": "a7-ipv4-v13",
                       "ipv4_extension": f"a7-ipv4-v13-{seed}-extension-01",
                       "ipv4_continuation": f"a7-ipv4-v13-{seed}-continuation-01",
                       "ipv4_r2_extension": "a7-ipv4-v13-2357-r2-extension-01",
                       "ipv4_r2_continuation": "a7-ipv4-v13-2357-r2-continuation-01",
                       "ipv4_json_extension": "a7-ipv4-v13-887-json-extension-01",
                       "ipv4_ws_extension": "a7-ipv4-v13-6101-ws-extension-01",
                       "ipv4_bounded_sse": f"a7-ipv4-v13-{seed}-bounded-sse-01",
                       "ipv4_resilient_sse": f"a7-ipv4-v13-{seed}-resilient-sse-01",
                       "ipv4_resilient_failure": f"a7-ipv4-v13-{seed}-resilient-failure-01",
                       "ipv4_relay_retry": f"a7-ipv4-v13-{seed}-relay-retry-01",
                       "ipv4_relay_retry_02": f"a7-ipv4-v13-{seed}-relay-retry-02",
                       "ipv4_relay_retry_03": f"a7-ipv4-v13-{seed}-relay-retry-03",
                       "ipv4_cached_r7": "a7-ipv4-v13-887-cached-r7-01"}
    if segment in cohort_services:
        service = cohort_services[segment]
        label = f"gui/{os.getuid()}/com.purellm.{service}"
    else:
        suffix = {"v11": "v11-direct-full", "v6": "v6-full", "v5": "v5-continuation", "direct": ("v9-direct" if seed == 887 else "v8-direct"),
                  "repair": "v10-targeted-repair"}[segment]
        label = f"gui/{os.getuid()}/com.purellm.a7-{suffix}-{seed}"
    lines = command_lines(["/bin/launchctl", "print", label])
    for line in lines or ():
        if match := re.fullmatch(r"\s*pid = ([0-9]+)\s*", line):
            return pid(int(match[1]))
    return None


def mac_report(private, seeds):
    pool = private / "spool" / PILOT / "key_pool/key_pool.sqlite3"
    background = next((row for row in seeds if row.get("segment") == "background"), None)
    if background and isinstance(contract := background.get("contract_sha256"), str) and SHA.fullmatch(contract):
        pool = private / "spool" / contract / "key_pool/key_pool.sqlite3"
    keys, queue, limits = [], [], {}
    try:
        with sqlite3.connect(f"{pool.as_uri()}?mode=ro", uri=True, timeout=2) as conn:
            conn.row_factory = sqlite3.Row
            limits = dict(conn.execute("SELECT cap,shared_until,shared_disabled FROM settings").fetchone() or {})
            rows = conn.execute("SELECT seed,round,logical_index,state,key_id,dispatched,response_complete,outcome,usage FROM queue").fetchall()
            for record in conn.execute("SELECT id,present,cooldown_until,disabled_error FROM keys WHERE configured=1 ORDER BY id"):
                alias = record["id"]
                if alias not in KEYS:
                    continue
                state = "available"
                if record["disabled_error"] or not record["present"]:
                    state = "disabled"
                elif any(row["key_id"] == alias and row["state"] == "busy" for row in rows):
                    state = "busy"
                elif record["cooldown_until"] > time.time():
                    state = "cooldown"
                dispatched = [row for row in rows if row["key_id"] == alias and row["dispatched"]]
                used = {name: 0 for name in TOKEN_FIELDS}
                for row in dispatched:
                    for name, count in usage(json.loads(row["usage"] or "{}")).items():
                        used[name] += count
                keys.append({"id": alias, "state": state, "requests": len(dispatched),
                             "complete": sum(row["response_complete"] for row in dispatched),
                             "usage": {field: count for field, count in used.items() if count}})
            queue = [{field: row[field] for field in ("seed", "round", "logical_index", "state", "key_id")}
                     for row in rows if row["state"] in ("queued", "busy") and row["key_id"] in (None, *KEYS)]
    except (OSError, sqlite3.Error, ValueError):
        pass
    for row in seeds:
        contract = row.get("contract_sha256")
        if isinstance(contract, str) and SHA.fullmatch(contract):
            spool = private / "spool" / contract
            if read_json(spool / "run_binding.json").get("contract_sha256") == contract:
                row["provider"] = local_progress(
                    spool, row["seed"], row["api"],
                    track_queue=row["segment"] in ("ipv4", "ipv4_extension", "ipv4_continuation",
                                                   "ipv4_r2_extension", "ipv4_r2_continuation",
                                                   "ipv4_json_extension", "ipv4_ws_extension", "ipv4_bounded_sse",
                                                   *RESILIENT_SEGMENTS),
                    readiness_windows=row.get("readiness_windows", 0),
                )
        if row.get("continuation"):
            row["dispatcher_pid"] = dispatcher_pid(row["seed"], row["segment"])
    return {"keys": keys, "queue": queue, "limits": limits}


def duration_text(value):
    return f"{value:.1f} s" if seconds(value) is not None else "—"


def show(view):
    print(f"A7 status  {view.get('checked_at', '—')}")
    for row in view.get("seeds", ()):
        api = row["api"]
        provider = row.get("provider") or {}
        rounds = row["completed_rounds"] if row["completed_rounds"] is not None else "—"
        print(f"seed {row['seed']}  L{row['labels'] or '—'}/L{row['target_labels']}  "
              f"rounds {rounds}/{row['target_rounds']}  {row['phase']}  worker {row['worker_pid'] or '—'}  "
              f"coordinator {row['coordinator_pid'] or '—'}  Mac {row.get('dispatcher_pid') or '—'}  {row['run'] or '—'}")
        if api or provider:
            sealed = ("已回传" if provider.get("delivered") else "已封存待回传" if provider.get("sealed")
                      else "待封存" if provider else "未见当前回执")
            if provider.get("phase") == "queued_for_dispatch":
                sealed = "等待派发"
            elif provider.get("phase") == "waiting_for_ipv4_route":
                sealed = "预检中，尚未提交模型请求"
            print(f"  API {api.get('status', '—')}  {api.get('failure') or '—'}  "
                  f"Mac {provider.get('phase', '—')} ({sealed})  HTTP {provider.get('http_status', '—')}  "
                  f"SSE {provider.get('sse_event_count', '—')}  "
                  f"状态更新 {provider.get('progress_age_seconds', '—')} s 前  tokens {provider.get('usage') or '—'}")
            if skipped := api.get("unsubmitted_reservations_skipped"):
                print(f"  已略过 {skipped} 个未提交到 bridge 的本地预留")
            if api.get("cached_import"):
                print("  本轮读取已归档回答，无新 API 提交")
            if (relay_wait := api.get("relay_retry_wait_seconds")) is not None:
                print(f"  已授权的同请求重试最早 {relay_wait} s 后可派发")
            error_details = [f"{label} {provider[field]}" for field, label in (
                ("error_type", "传输异常"), ("provider_error_code", "SSE 错误码"),
                ("provider_error_status", "SSE 错误状态"), ("conflicting_event_type", "ID 变化事件"),
            ) if field in provider]
            if provider.get("response_id_matches_transport_alias") is True:
                error_details.append("变化 ID 与传输别名一致")
            if error_details:
                print("  " + "  ".join(error_details))
            if provider.get("readiness_windows_used"):
                retry = provider.get("readiness_retry_seconds")
                retry_text = f"  下个窗口最早 {retry} s 后" if retry is not None else ""
                sent = {True: "已提交", False: "未提交"}.get(provider.get("provider_dispatched"), "待确认")
                print(f"  IPv4 预检窗口 {provider['readiness_windows_used']}/{provider['readiness_windows']}"
                      f"  模型请求 {sent}{retry_text}")
            if provider and provider.get("phase") != "waiting_for_ipv4_route":
                terminal = ("—" if provider.get("phase") == "queued_for_dispatch"
                            else "已收到" if provider.get("complete_event_seen") else "未收到")
                elapsed = seconds(provider.get("provider_elapsed_seconds"))
                last_event = seconds(provider.get("last_sse_seconds"))
                last_activity = seconds(provider.get("last_sse_activity_seconds"))
                since_event = elapsed - last_event if elapsed is not None and last_event is not None else None
                since_activity = elapsed - last_activity if elapsed is not None and last_activity is not None else None
                inactivity_text = (f"  距上条有效事件 {duration_text(since_activity)}"
                                   if since_activity is not None else "")
                print(f"  派发前等待 {duration_text(provider.get('dispatch_wait_seconds'))}  "
                      f"API 已用时（含预检） {duration_text(elapsed)}  "
                      f"首段正文 {duration_text(provider.get('first_output_seconds'))}  "
                      f"最近两条 SSE 间隔 {duration_text(provider.get('last_sse_interval_seconds'))}  "
                      f"距上条 SSE {duration_text(since_event)}{inactivity_text}  "
                      f"完成终态 {terminal} {duration_text(provider.get('complete_event_seconds'))}")
            if api.get("selection"):
                print(f"  A7 校验 {api['selection']}  {api['selection_errors'] or '—'}")
    for gpu in view.get("gpus") or ():
        print(f"GPU {gpu} (MiB used, MiB total, utilization %)")
    for process in view.get("gpu_processes") or ():
        print(f"  GPU 计算进程 {process}")
    if view.get("gpus") is None:
        print("GPU 查询暂不可用")
    mac = view["mac"]
    print("共享 Key 池", ", ".join(f"{row['id']} {row['state']} ({row['complete']}/{row['requests']})"
                                   for row in mac["keys"]) or "暂不可用")
    limits = mac["limits"]
    cooldown = int(max(0, limits.get("shared_until", 0) - time.time()))
    if cooldown or limits.get("shared_disabled"):
        print(f"  共享冷却 {cooldown} 秒，已停用 {bool(limits.get('shared_disabled'))}")
    for row in mac["queue"]:
        print(f"  seed {row['seed']}  round {row['round']}  turn {row['logical_index']}  {row['state']}  {row['key_id'] or '—'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--ssh-host", default="eit-hpc")
    parser.add_argument("--watch", type=float, metavar="SECONDS", help="repeat the read-only snapshot")
    parser.add_argument("--remote", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.remote:
        print(json.dumps(remote_report()))
        return 0
    source = Path(__file__).read_text()
    command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=8",
               "--", args.ssh_host, f"{ROOT}/environment/venv/bin/python -B - --remote"]
    if args.watch is not None and not 3 <= args.watch <= 300:
        parser.error("watch interval must be between 3 and 300 seconds")
    while True:
        try:
            result = subprocess.run(command, input=source, capture_output=True, text=True, timeout=30, check=True)
            view = json.loads(result.stdout)
        except (OSError, ValueError, subprocess.SubprocessError):
            view = {"remote_unavailable": True}
        view["checked_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        view["mac"] = mac_report(Path.home() / ".config/purellm", view.get("seeds", ()))
        if args.json:
            print(json.dumps(view, ensure_ascii=False, indent=None if args.watch else 2), flush=True)
        else:
            show(view)
            print(flush=True)
        if args.watch is None:
            return int(view.get("remote_unavailable", False))
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            return 130



if __name__ == "__main__":
    raise SystemExit(main())
