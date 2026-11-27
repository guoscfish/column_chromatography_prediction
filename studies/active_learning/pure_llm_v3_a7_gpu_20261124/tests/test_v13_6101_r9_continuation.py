from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    SEALED_RELAY_RETRY_SECONDS, TransportConfig, TransportStop, UNKNOWN_REPLAY_DECISION, make_request,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import (
    dispatch_v13_6101_r9_continuation as mac,
    run_v13_6101_r9_continuation as gpu,
)


@pytest.fixture
def stage():
    transport = TransportConfig("provider", "model", "model", "xhigh", "https://invalid.local/v1", 3600, 32768,
                                logical_replies=3, transport_attempts=2)
    base = [{"role": "system", "content": "ordinary A7"}, {"role": "user", "content": "complete input"}]
    binding = {field: "a" * 64 for field in gpu.STAGE_FIELDS}
    binding.update(seed=6101, round=9, base_messages_sha256=digest(base))
    request = make_request(binding, 0, base, transport)
    return request, gpu.request_identity(request), transport


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
        request["payload"]["input"][2]["content"] = "targeted R8 guide"
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
        gpu._correction_ready(tmp_path, new)
    once(prior / "attempt_00.terminal.json", {"status": "completed", "request_id": first["request_id"]})
    gpu._correction_ready(tmp_path, new)
    once(tmp_path / "batch_freeze.json", {})
    with pytest.raises(PermissionError):
        gpu._correction_ready(tmp_path, new)


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
    gpu._link_replay_attempt(turn, started1, replay, allow_create=True)
    gpu._check_second_slot(tmp_path, request, started1)
    with pytest.raises(TransportStop):
        gpu._check_second_slot(tmp_path, request, {**started1, "started_unix": 199})
    with pytest.raises(FileNotFoundError):
        gpu._check_second_slot(tmp_path, {**request, "logical_index": 0}, started1)


def test_mac_same_logical_physical_retry_does_not_reuse_complete_local_answer(stage, tmp_path, monkeypatch):
    request = correction(stage)
    _, identity, transport = stage
    dispatcher = object.__new__(mac.ContinuationDispatcher)
    dispatcher.path, dispatcher.transport = tmp_path, transport
    dispatcher.scope = {"r9_request": identity}
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


def test_r9_budget_expands_only_exact_normal_import():
    contract = SimpleNamespace(predecessor={"dispatcher_module": gpu.DISPATCHER, "execution_budget": gpu.BUDGET},
        profile={"full_imported_boundary_round": 9}, seed_ids=(6101,), rounds=36, target_labels=1485,
        transport=SimpleNamespace(logical_replies=3, transport_attempts=2))
    run = Path(gpu.RUN_NAME)
    guard = gpu.seed_runner._imported_boundary_reply_limit
    assert guard(contract, run, 6101, 9, audited=True, replay_start=9, relay_delay=180) == 3
    with pytest.raises(PermissionError):
        guard(contract, run, 6101, 9, audited=True, replay_start=10, relay_delay=180)
    with pytest.raises(PermissionError):
        guard(contract, run, 887, 9, audited=True, replay_start=9, relay_delay=180)
    contract.predecessor = {"dispatcher_module": "dispatch_v13_887_r20_continuation"}
    contract.profile = {}
    assert guard(contract, run, 887, 20, audited=True, replay_start=20, relay_delay=180) == 1


def test_r9_cannot_borrow_the_r33_budget():
    contract = SimpleNamespace(
        predecessor={"dispatcher_module": gpu.DISPATCHER, "execution_budget": {
            "first_logical_requests": 3, "maximum_logical_requests": 9, "maximum_physical_attempts": 18,
        }},
        profile={"full_imported_boundary_round": 9}, seed_ids=(6101,), rounds=36, target_labels=1485,
        transport=SimpleNamespace(logical_replies=3, transport_attempts=2),
    )
    with pytest.raises(PermissionError, match="imported R9 normal-round budget"):
        gpu.seed_runner._imported_boundary_reply_limit(
            contract, Path(gpu.RUN_NAME), 6101, 9, audited=True, replay_start=9, relay_delay=180,
        )


def test_w28_overlay_reproduces_the_signed_r9_seed_runner(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is required to reproduce the historical W28 overlay")
    commit = "1793de14d76a57e5b1913e31204353f564835d33"
    archived = subprocess.run([git, "show", f"{commit}:{gpu.SEED_RUNNER}"], cwd=gpu.ROOT, capture_output=True)
    if archived.returncode:
        pytest.skip("The historical W28 seed-runner Git object is outside this checkout")
    runner = tmp_path / gpu.SEED_RUNNER
    runner.parent.mkdir(parents=True)
    runner.write_bytes(archived.stdout)
    assert file_digest(runner) == gpu.SOURCE_BASE[gpu.SEED_RUNNER]
    overlay = gpu.ROOT / gpu.STUDY / "operator/overlays/6101-r9-normal-boundary-budget.patch"
    subprocess.run([git, "apply", "--unidiff-zero", "--check", str(overlay)], cwd=tmp_path, check=True)
    subprocess.run([git, "apply", "--unidiff-zero", str(overlay)], cwd=tmp_path, check=True)
    assert file_digest(runner) == gpu.RUNNER_PATCH
