from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    SEALED_RELAY_RETRY_SECONDS, TransportConfig, TransportStop, UNKNOWN_REPLAY_DECISION, make_request,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import (
    dispatch_v13_6101_r11_continuation as mac,
    run_v13_6101_r11_continuation as gpu,
)


@pytest.fixture
def stage():
    transport = TransportConfig("provider", "model", "model", "xhigh", "https://invalid.local/v1", 3600, 32768,
                                logical_replies=3, transport_attempts=2)
    base = [{"role": "system", "content": "ordinary A7"}, {"role": "user", "content": "complete input"}]
    binding = {field: "a" * 64 for field in gpu.normal.STAGE_FIELDS}
    binding.update(seed=6101, round=11, base_messages_sha256=digest(base))
    request = make_request(binding, 0, base, transport)
    return request, gpu.normal.request_identity(request), transport


def correction(stage, *, index=1, mode="patch"):
    request, _, transport = stage
    instruction = gpu.REPAIR_INSTRUCTIONS if mode == "patch" else gpu.REPLAN_INSTRUCTIONS
    messages = [*request["payload"]["input"], {"role": "user", "content": instruction + "\n" + gpu.dumps({"mode": mode})}]
    return make_request(request["binding"], index, messages, transport)


def test_first_round_keeps_complete_initial_input_and_existing_multiline_corrections(stage):
    first, identity, _ = stage
    gpu.check_first_round(first, identity)
    gpu.check_first_round(correction(stage), identity)
    gpu.check_first_round(correction(stage, index=2, mode="full_replan"), identity)


@pytest.mark.parametrize("change", ["first_payload", "checkpoint", "base_messages", "fourth_logical", "repair_guide", "mode"])
def test_first_round_rejects_foreign_science_or_correction(stage, change):
    first, identity, _ = stage
    request = correction(stage)
    if change == "first_payload":
        request = {**first, "payload_sha256": "b" * 64}
    if change == "checkpoint":
        request["binding"] = {**request["binding"], "checkpoint_sha256": "b" * 64}
    if change == "base_messages":
        request["payload"]["input"][1] = {"role": "user", "content": "truncated"}
    if change == "fourth_logical":
        request["logical_index"] = 3
    if change == "repair_guide":
        request["payload"]["input"][2]["content"] = "targeted R32 guide"
    if change == "mode":
        request["payload"]["input"][2]["content"] = gpu.REPAIR_INSTRUCTIONS + '\n{"mode":"full_replan"}'
    with pytest.raises(PermissionError):
        gpu.check_first_round(request, identity)


def test_correction_requires_completed_prior_and_rejects_post_freeze(stage, tmp_path):
    first, _, _ = stage
    new = correction(stage)
    prior = tmp_path / "turn_00"
    once(prior / "request.json", first)
    once(prior / "decision.json", {"state": "rejected"})
    with pytest.raises(PermissionError):
        gpu.normal._correction_ready(tmp_path, new)
    once(prior / "attempt_00.terminal.json", {"status": "completed", "request_id": first["request_id"]})
    gpu.normal._correction_ready(tmp_path, new)
    once(tmp_path / "batch_freeze.json", {})
    with pytest.raises(PermissionError):
        gpu.normal._correction_ready(tmp_path, new)


def test_real_replay_link_needs_same_logical_and_sealed_clock(stage, tmp_path):
    request = correction(stage)
    turn = tmp_path / "turn_01"
    scope = "1" * 64
    started0 = {"request_id": request["request_id"], "attempt_id": "b" * 32, "attempt": 0,
                "started_unix": 10, "transport": "mac_bridge"}
    failure = {**started0, "status": "stopped", "error": "response_id_mismatch", "ambiguous": True,
        "provider_dispatched": True, "circuit_open": False, "bridge_result_sha256": "2" * 64, "rate_limit_count": 0,
        "provider_status": "failed", "provider_request_id": "6cdab875-5e2f-4aae-ada2-bfeb8a7f587f",
        "response_id": "resp_6cdab8755e2f4aaeada2bfeb8a7f587f", "not_before_unix": 22, "failed_unix": 20}
    request["binding"] = {**request["binding"], "a7_formal": {"contract_sha256": scope, "mock": False}}
    authorization = {"decision": UNKNOWN_REPLAY_DECISION, "authorized_by": "test", "authorization_reference": "test",
        "accept_duplicate_billing_risk": True, "contract_sha256": scope, "request_id": request["request_id"],
        "attempt_id": started0["attempt_id"], "bridge_result_sha256": failure["bridge_result_sha256"],
        "original_terminal_sha256": digest(failure), "next_attempt": 1, "sealed_relay_retry_seconds": SEALED_RELAY_RETRY_SECONDS}
    once(turn / "request.json", request)
    once(turn / "attempt_00.started.json", started0)
    once(turn / "attempt_00.terminal.json", failure)
    from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import _seal_ambiguous_replay
    replay = _seal_ambiguous_replay(turn, request, authorization, digest(authorization))
    started1 = {**started0, "attempt_id": "c" * 32, "attempt": 1, "started_unix": 200}
    gpu.normal._link_replay_attempt(turn, started1, replay, allow_create=True)
    gpu.normal._check_second_slot(tmp_path, request, started1)
    with pytest.raises(TransportStop):
        gpu.normal._check_second_slot(tmp_path, request, {**started1, "started_unix": 199})
    with pytest.raises(FileNotFoundError):
        gpu.normal._check_second_slot(tmp_path, {**request, "logical_index": 0}, started1)


def test_mac_same_logical_physical_retry_does_not_reuse_complete_local_answer(stage, tmp_path, monkeypatch):
    request = correction(stage)
    _, identity, transport = stage
    dispatcher = object.__new__(mac.ContinuationDispatcher)
    dispatcher.path, dispatcher.transport = tmp_path, transport
    dispatcher.scope = {"r11_request": identity}
    old = {"attempt_id": "b" * 32, "provider_dispatched": True, "raw_response": None,
           "outcome": {"status": "failed", "failure": {"code": "response_id_mismatch", "retryable": False,
               "ambiguous": True, "reply_identity": {"usage": {}, "served_model": None, "provider_status": "failed",
               "provider_request_id": "6cdab875-5e2f-4aae-ada2-bfeb8a7f587f",
               "response_id": "resp_6cdab8755e2f4aaeada2bfeb8a7f587f"}}}}
    seen = []

    def read_prior(item, logical, physical):
        seen.append((logical, physical))
        if logical == 0 and physical == 0:
            return item, {"outcome": {"status": "completed"}, "raw_response": {"status": "completed"}}
        if logical == 1 and physical == 0:
            return item, old
        return None

    monkeypatch.setattr(dispatcher, "_prior", read_prior)
    started = {"attempt_id": "c" * 32, "attempt": 1}
    dispatcher._check_boundary(request, started)
    assert seen == [(0, 0), (0, 1), (1, 0)]
    old["raw_response"] = {"status": "completed"}
    with pytest.raises(PermissionError):
        dispatcher._check_boundary(request, started)


def test_r11_budget_expands_only_exact_normal_import():
    contract = SimpleNamespace(predecessor={"dispatcher_module": gpu.DISPATCHER, "execution_budget": gpu.BUDGET},
        profile={"full_imported_boundary_round": 11}, seed_ids=(6101,), rounds=36, target_labels=1485,
        transport=SimpleNamespace(logical_replies=3, transport_attempts=2))
    run = Path(gpu.RUN_NAME)
    guard = gpu.seed_runner._imported_boundary_reply_limit
    assert guard(contract, run, 6101, 11, audited=True, replay_start=11, relay_delay=180) == 3
    with pytest.raises(PermissionError):
        guard(contract, run, 6101, 11, audited=True, replay_start=12, relay_delay=180)
    with pytest.raises(PermissionError):
        guard(contract, run, 887, 11, audited=True, replay_start=11, relay_delay=180)
    contract.predecessor = {"dispatcher_module": "dispatch_v13_887_r20_continuation"}
    contract.profile = {}
    assert guard(contract, run, 887, 20, audited=True, replay_start=20, relay_delay=180) == 1



def test_signed_handoff_fails_closed_if_snapshot_or_zero_post_proof_drifts(stage):
    from copy import deepcopy
    _, identity, _ = stage
    actual = {**identity, "checkpoint_sha256": gpu.HANDOFF["checkpoint_sha256"]}
    pins = {"r10_handoff": dict(gpu.HANDOFF), "r11_request": actual, "cached_response_import_file_sha256": "d"*64}
    assert gpu.validate_pins(pins) == (gpu.HANDOFF, actual)
    for section, name, other in (
        ("r10_handoff", "provider_post", 0),
        ("r10_handoff", "local_attempt_id", gpu.cached.ATTEMPT),
        ("r10_handoff", "checkpoint_sha256", "e"*64),
        ("r11_request", "checkpoint_sha256", "f"*64),
    ):
        tampered = deepcopy(pins)
        tampered[section][name] = other
        with pytest.raises(PermissionError, match="pins"):
            gpu.validate_pins(tampered)
    with pytest.raises(PermissionError):
        gpu.validate_pins({})


def test_scoped_overlay_preserves_r9_budget_and_does_not_enable_another_run():
    contract = SimpleNamespace(predecessor={"dispatcher_module": gpu.normal.DISPATCHER,
        "execution_budget": gpu.normal.BUDGET}, profile={"full_imported_boundary_round": 9},
        seed_ids=(6101,), rounds=36, target_labels=1485, transport=SimpleNamespace(logical_replies=3, transport_attempts=2))
    guard = gpu.seed_runner._imported_boundary_reply_limit
    run = Path(gpu.normal.RUN_NAME)
    assert guard(contract, run, 6101, 9, audited=True, replay_start=9, relay_delay=180) == 3
    with pytest.raises(PermissionError):
        guard(contract, Path(gpu.RUN_NAME), 6101, 9, audited=True, replay_start=9, relay_delay=180)


def _apply_overlay(git, tmp_path, overlay):
    path = gpu.ROOT / gpu.STUDY / "operator/overlays" / overlay
    subprocess.run([git, "apply", "--unidiff-zero", "--check", str(path)], cwd=tmp_path, check=True)
    subprocess.run([git, "apply", "--unidiff-zero", str(path)], cwd=tmp_path, check=True)


def test_overlays_reproduce_exact_signed_r11_runner(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is needed to reproduce signed overlays")
    commit = "1793de14d76a57e5b1913e31204353f564835d33"
    archived = subprocess.run([git, "show", f"{commit}:{gpu.normal.SEED_RUNNER}"], cwd=gpu.ROOT, capture_output=True)
    if archived.returncode:
        pytest.skip("Historical W28 release source object is unavailable in this checkout")
    runner = tmp_path / gpu.normal.SEED_RUNNER
    runner.parent.mkdir(parents=True)
    runner.write_bytes(archived.stdout)
    assert file_digest(runner) == gpu.normal.SOURCE_BASE[gpu.normal.SEED_RUNNER]
    _apply_overlay(git, tmp_path, "6101-r9-normal-boundary-budget.patch")
    assert file_digest(runner) == gpu.normal.RUNNER_PATCH
    _apply_overlay(git, tmp_path, "6101-r11-normal-boundary-budget.patch")
    assert file_digest(runner) == gpu.RUNNER_PATCH


def test_reviewed_whitespace_overlay_uses_only_complete_final(tmp_path, monkeypatch):
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is needed to reproduce the signed whitespace overlay")
    parser = tmp_path / gpu.PARSER
    parser.parent.mkdir(parents=True)
    parser.write_bytes((gpu.ROOT / gpu.PARSER).read_bytes())
    assert file_digest(parser) == gpu.OLD_PARSER
    _apply_overlay(git, tmp_path, "887-r20-final-answer-whitespace.patch")
    assert file_digest(parser) == gpu.NEW_PARSER
    module_name = "studies.active_learning.pure_llm_v3_20261124.pure_llm_v3._tested_whitespace_transport"
    spec = spec_from_file_location(module_name, parser)
    module = module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    placeholder = {"type": "message", "status": "completed", "role": "assistant",
                   "content": [{"type": "output_text", "text": " "}]}
    final = {**placeholder, "phase": "final_answer", "content": [{"type": "output_text", "text": "exact final"}]}
    assert module.response_output_text([placeholder, {"type": "reasoning"}, final]) == "exact final"
    with pytest.raises(module.ResponseMessagePhaseError):
        module.response_output_text([placeholder, {**final, "phase": None}, final])
