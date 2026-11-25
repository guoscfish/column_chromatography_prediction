"""The new authorization is separate even though the scientific inputs are the same."""

from copy import deepcopy
import fcntl

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import SCHEMA_VERSION
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportFailure, make_request
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import supersession
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import make_job, make_result
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.compact_payload import FORMAL_WIRE_REPRESENTATION
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    PROFILE, PROFILE_V3, PROFILE_V4, PROFILE_V4_PILOT, ROOT, SEEDS, SSH_API, V4_PILOT_RUN_NAME, V4_RUN_NAME,
    load_contract, request_timeouts, verify_authorization,
)


@pytest.fixture
def original_run(tmp_path, monkeypatch):
    original = tmp_path / "original"
    anchor, lease = tmp_path / "anchors", tmp_path / "lease"
    hashes = {}
    for seed in SEEDS:
        file = anchor / f"random-{seed}/rounds/random/round_00/model/best.pt"
        file.parent.mkdir(parents=True)
        file.write_text(f"offline shared checkpoint for {seed}")
        hashes[str(seed)] = file_digest(file)
    monkeypatch.setattr(supersession, "L333_ANCHOR_SHA256", hashes)
    direct = load_contract(ROOT, ROOT / PROFILE, anchor_root=anchor, lease_root=lease,
                           run_directory=original, gpu_slots=(0, 1))
    monkeypatch.setattr(supersession, "PREDECESSOR_SHA256", direct.digest)
    approval = {"scope": "pure_llm_v3_a7_formal", "contract_sha256": direct.digest,
                "authorized_by": "synthetic test", "authorization_reference": "no live provider",
                "authorize_paid_api": True, "authorize_label_reveal": True, "authorize_cuda_training": True}
    binding = {"seed": 157, "round": 0, "checkpoint_sha256": hashes["157"],
               "a7_formal": {"seed": 157, "mock": False, "contract_sha256": direct.digest,
                             "authorization_sha256": digest(approval)}}
    request = make_request(binding, 0, [{"role": "user", "content": "offline fixture"}], direct.transport)
    turn = original / "seed_157/round_00/turn_00"
    once(original / "formal_contract.json", direct.record())
    once(original / "authorization.json", approval)
    once(original / "pause.request", {"reason": "offline fixture"})
    once(turn / "request.json", request)
    for n in range(2):
        started = {"request_id": request["request_id"], "attempt": n, "attempt_id": f"{n+1:032x}"}
        once(turn / f"attempt_{n:02d}.started.json", started)
        once(turn / f"attempt_{n:02d}.terminal.json", {
            **started, "status": "retryable" if n == 0 else "stopped",
            "error": "connect_error", "ambiguous": False,
        })
    return direct, approval, original, turn


@pytest.fixture
def stopped_v2(original_run, tmp_path, monkeypatch):
    direct, original_approval, original, _ = original_run
    v2 = tmp_path / "a7-gpt62-ultra-ssh-v2"
    contract = load_contract(ROOT, ROOT / PROFILE, anchor_root=direct.anchor_root, lease_root=direct.lease_root,
                             run_directory=v2, gpu_slots=direct.gpu_slots, api_execution=SSH_API,
                             predecessor_run=original)
    monkeypatch.setattr(supersession, "V2_SHA256", contract.digest)
    approval, _, result = _stage_stopped_ssh(contract, original_approval, v2, error="provider_reasoning_rejected")
    monkeypatch.setattr(supersession, "V2_RESULT_SHA256", result["result_sha256"])
    return contract, approval, v2, original


def _stage_stopped_ssh(contract, original_approval, directory, *, error, ambiguous=False):
    approval = {**original_approval, "contract_sha256": contract.digest}
    scope = {"seed": 157, "mock": False, "contract_sha256": contract.digest,
             "authorization_sha256": digest(approval)}
    packet = {"seed": 157, "round": 0, "layout": "B", "compression": "lossless",
              "selection_count": 32, "memory": {"mode": "M2"}}
    messages = [{"role": "system", "content": "Synthetic A7 task."}, {"role": "user", "content": dumps(packet)}]
    binding = {"seed": 157, "round": 0, "arm": "A7", "schema_version": SCHEMA_VERSION, "prompt_version": "v3",
               "checkpoint_sha256": supersession.L333_ANCHOR_SHA256["157"], "packet_sha256": digest(packet),
               "base_messages_sha256": digest(messages), "prompt_sha256": digest(messages[0]["content"]),
               "mapping_sha256": "a" * 64, "state_sha256": "b" * 64, "split_sha256": "e" * 64, "a7_formal": scope}
    request = make_request(binding, 0, messages, contract.transport)
    attempt, dispatcher = ("3" if ambiguous else "1") * 32, "2" * 32
    started = {"request_id": request["request_id"], "attempt": 0, "attempt_id": attempt,
               "started_unix": 1.0, "transport": "mac_bridge"}
    job = make_job(request, contract.transport, started)
    result = make_result(job, failure=TransportFailure(error, ambiguous=ambiguous), credential_id="fixture-key",
                         provider_dispatched=True, elapsed_seconds=858 if ambiguous else None)
    once(directory / "formal_contract.json", contract.record())
    once(directory / "authorization.json", approval)
    if ambiguous:
        once(directory / "coordinator.lock", {})
        for seed in SEEDS:
            once(directory / "workers" / f"seed_{seed}.lock", {})
        once(directory / "workers/seed_157.json", {"seed": 157, "status": "failed",
                                                   "error_type": "TransportStop", "reason": "timeout"})
    else:
        once(directory / "pause.request", {"reason": "offline fixture"})
    turn = directory / "seed_157/round_00/turn_00"
    once(turn / "request.json", request)
    once(turn / "formal_request_authorization.json", {"scope": "paid_selection_only", "formal_run": scope,
                                                      "request_id": request["request_id"]})
    once(turn / "attempt_00.started.json", started)
    once(turn / "attempt_00.terminal.json", {
        **started, "status": "stopped", "error": error, "ambiguous": ambiguous,
        "bridge_job_sha256": job["job_sha256"], "bridge_result_sha256": result["result_sha256"],
        "provider_dispatched": True, "provider_elapsed_seconds": result["elapsed_seconds"],
        "credential_id": result["credential_id"],
    })
    once(directory / "bridge/dispatcher.json", {"dispatcher_id": dispatcher, "contract_sha256": contract.digest})
    once(directory / f"bridge/outbox/{attempt}.json", job)
    once(directory / f"bridge/claims/{attempt}.json", {"dispatcher_id": dispatcher, "attempt_id": attempt,
                                                     "job_sha256": job["job_sha256"]})
    once(directory / f"bridge/results/{attempt}.json", result)
    return approval, request, result


@pytest.fixture
def stopped_v3(stopped_v2, tmp_path, monkeypatch):
    previous, approval, v2, original = stopped_v2
    v3 = tmp_path / "a7-gpt62-max-ssh-v3"
    contract = load_contract(ROOT, ROOT / PROFILE_V3, anchor_root=previous.anchor_root,
                             lease_root=previous.lease_root, run_directory=v3, gpu_slots=previous.gpu_slots,
                             api_execution=SSH_API, predecessor_run=v2)
    approval, request, result = _stage_stopped_ssh(contract, approval, v3, error="timeout", ambiguous=True)
    monkeypatch.setattr(supersession, "V3_SHA256", contract.digest)
    monkeypatch.setattr(supersession, "V3_REQUEST_ID", request["request_id"])
    monkeypatch.setattr(supersession, "V3_ATTEMPT_ID", result["attempt_id"])
    monkeypatch.setattr(supersession, "V3_RESULT_SHA256", result["result_sha256"])
    return contract, approval, v3, v2, original


def test_separate_ssh_version_inherits_identical_science_only(original_run, tmp_path):
    direct, previous_approval, original, _ = original_run
    old_paths = {path: path.read_bytes() for path in original.rglob("*.json")}
    ssh = load_contract(ROOT, ROOT / PROFILE, anchor_root=direct.anchor_root, gpu_slots=(0, 1),
                        lease_root=direct.lease_root, run_directory=tmp_path / "bridge",
                        api_execution=SSH_API, predecessor_run=original)
    assert ssh.profile == direct.profile and ssh.asset_sha256 == direct.asset_sha256
    assert ssh.record()["budget"] == direct.record()["budget"]
    assert ssh.record()["api_execution"] == SSH_API
    assert ssh.predecessor["prior_connection_attempts"] == 2 and ssh.predecessor["scientific_decisions"] == 0
    assert not (tmp_path / "bridge").exists()
    assert all(path.read_bytes() == data for path, data in old_paths.items())
    with pytest.raises(PermissionError):
        verify_authorization(ssh, previous_approval)


def test_provider_receipt_or_science_cannot_disappear_from_history(original_run):
    _, _, original, turn = original_run
    terminal = turn / "attempt_01.terminal.json"
    evidence = read(terminal)
    changed = deepcopy(evidence)
    changed["response_id"] = "do-not-duplicate"
    import json

    terminal.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="execution or response evidence"):
        supersession.inspect_predecessor(original)
    terminal.write_text(json.dumps(evidence))
    once(original / "seed_157/round_00/batch_freeze.json", {"test_only": True})
    with pytest.raises(ValueError, match="scientific decision"):
        supersession.inspect_predecessor(original)


def test_v3_changes_only_reasoning_and_binds_all_five_shared_checkpoints(stopped_v2, tmp_path):
    previous, previous_approval, v2, original = stopped_v2
    effort = read(ROOT / PROFILE_V3)["transport"]["reasoning_effort"]
    v3_path = tmp_path / f"a7-gpt62-{effort}-ssh-v3"
    kwargs = {"anchor_root": previous.anchor_root, "gpu_slots": previous.gpu_slots, "lease_root": previous.lease_root,
              "run_directory": v3_path, "api_execution": SSH_API, "predecessor_run": v2}
    history = {path: path.read_bytes() for run in (original, v2) for path in run.rglob("*.json")}
    contract = load_contract(ROOT, ROOT / PROFILE_V3, **kwargs)
    assert contract.profile == {**previous.profile, "transport": {**previous.profile["transport"],
                                                                   "reasoning_effort": effort}}
    old_request = read(v2 / "seed_157/round_00/turn_00/request.json")
    new_request = make_request(old_request["binding"], 0, old_request["payload"]["input"], contract.transport)
    assert new_request["payload"] == {**old_request["payload"], "reasoning": {"effort": "max"}}
    assert contract.record()["budget"] == previous.record()["budget"]
    assert contract.predecessor["original_contract_sha256"] == previous.predecessor["contract_sha256"]
    assert contract.predecessor["prior_reasoning_rejections"] == 1
    assert contract.predecessor["initial_checkpoint_sha256_by_seed"] == supersession.L333_ANCHOR_SHA256
    assert not v3_path.exists() and all(path.read_bytes() == data for path, data in history.items())
    with pytest.raises(PermissionError):
        verify_authorization(contract, previous_approval)
    with pytest.raises(ValueError, match="approved v2 predecessor"):
        load_contract(ROOT, ROOT / PROFILE_V3, **{**kwargs, "predecessor_run": original})
    checkpoint = tmp_path / "anchors/random-887/rounds/random/round_00/model/best.pt"
    checkpoint.write_text("drifted checkpoint")
    with pytest.raises(ValueError, match="CUDA checkpoint differs for seed 887"):
        load_contract(ROOT, ROOT / PROFILE_V3, **kwargs)


def test_v3_rejects_a_v2_scientific_decision(stopped_v2):
    _, _, v2, _ = stopped_v2
    once(v2 / "seed_157/round_00/turn_00/decision.json", {"test_only": True})
    with pytest.raises(ValueError, match="scientific decision"):
        supersession.inspect_predecessor(v2)


def test_v3_cannot_fall_back_to_xhigh(tmp_path):
    profile = read(ROOT / PROFILE_V3)
    profile["transport"]["reasoning_effort"] = "xhigh"
    unqualified = tmp_path / PROFILE_V3.name
    once(unqualified, profile)
    with pytest.raises(ValueError, match="reasoning and budget are fixed"):
        load_contract(ROOT, unqualified)


def _v4_inputs(stopped_v3, tmp_path):
    previous, _, v3, _, _ = stopped_v3
    return {"anchor_root": previous.anchor_root, "gpu_slots": previous.gpu_slots, "lease_root": previous.lease_root,
            "run_directory": tmp_path / V4_RUN_NAME, "api_execution": SSH_API, "predecessor_run": v3}


def test_v4_binds_exact_science_model_and_unresolved_stopped_v3(stopped_v3, tmp_path):
    previous, approval, v3, v2, original = stopped_v3
    before = {path: path.read_bytes() for run in (original, v2, v3) for path in run.rglob("*") if path.is_file()}
    args = _v4_inputs(stopped_v3, tmp_path)
    v4 = load_contract(ROOT, ROOT / PROFILE_V4, **args)
    assert (v4.transport.requested_model, v4.transport.served_model, v4.transport.reasoning_effort) == (
        "gpt-6.2-astra", "gpt-6.2-astra", "xhigh")
    assert v4.profile["wire_representation"] == FORMAL_WIRE_REPRESENTATION
    assert request_timeouts(v4.profile).bridge_wait_seconds == 4200
    assert (v4.asset_sha256, v4.anchor_root, v4.gpu_slots, v4.lease_root, v4.record()["budget"]) == (
        previous.asset_sha256, previous.anchor_root, previous.gpu_slots, previous.lease_root, previous.record()["budget"])
    predecessor = v4.predecessor
    assert predecessor["contract_sha256"] == previous.digest
    assert predecessor["initial_checkpoint_sha256_by_seed"] == supersession.L333_ANCHOR_SHA256
    assert predecessor["prior_ambiguous_timeouts"] == 1 and predecessor["complete_provider_responses"] == 0
    assert predecessor["provider_execution_and_billing"] == "unresolved"
    assert not predecessor["provider_billing_independently_verified"]
    assert predecessor["evidence_sha256"]["workers/seed_157.json"] == file_digest(v3 / "workers/seed_157.json")
    old_request = read(v3 / "seed_157/round_00/turn_00/request.json")
    new_request = make_request(old_request["binding"], 0, old_request["payload"]["input"], v4.transport)
    assert new_request["request_id"] != old_request["request_id"]
    assert new_request["payload_sha256"] != old_request["payload_sha256"]
    assert v4.digest != previous.digest and not args["run_directory"].exists()
    assert all(path.read_bytes() == data for path, data in before.items())
    with pytest.raises(PermissionError):
        verify_authorization(v4, approval)
    new_approval = {**approval, "contract_sha256": v4.digest, "authorization_reference": "unresolved v3 test only"}
    assert verify_authorization(v4, new_approval) == new_approval
    for change in ({"run_directory": tmp_path / "wrong-v4-run"}, {"predecessor_run": v2}):
        with pytest.raises(ValueError, match="audited v3 predecessor and its own registered run name"):
            load_contract(ROOT, ROOT / PROFILE_V4, **{**args, **change})


def test_v4_refuses_active_or_changed_v3(stopped_v3, tmp_path):
    _, _, v3, _, _ = stopped_v3
    args = _v4_inputs(stopped_v3, tmp_path)
    with (v3 / "coordinator.lock").open("r+") as active:
        fcntl.flock(active, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="coordinator or a seed worker is still active"):
            load_contract(ROOT, ROOT / PROFILE_V4, **args)
    terminal = v3 / "seed_157/round_00/turn_00/attempt_00.terminal.json"
    terminal.write_text(dumps({**read(terminal), "ambiguous": False}))
    with pytest.raises(ValueError, match="timeout is detached from its immutable bridge evidence"):
        load_contract(ROOT, ROOT / PROFILE_V4, **args)


@pytest.mark.parametrize(("field", "value"), (("requested_model", "gpt-6.2-sol"),
                                              ("served_model", "gpt-6.2-astra-other"),
                                              ("reasoning_effort", "max"), ("sdk_version", "2.29.0")))
def test_v4_profile_refuses_different_model_effort_or_sdk(tmp_path, field, value):
    profile = read(ROOT / PROFILE_V4)
    profile["transport"][field] = value
    unqualified = tmp_path / PROFILE_V4.name
    once(unqualified, profile)
    with pytest.raises(ValueError, match="reasoning and budget are fixed"):
        load_contract(ROOT, unqualified)


def test_pilot_scope_caps_a_single_gpu_three_seed_run(stopped_v3, tmp_path):
    full = load_contract(ROOT, ROOT / PROFILE_V4, **_v4_inputs(stopped_v3, tmp_path))
    args = {**_v4_inputs(stopped_v3, tmp_path), "run_directory": tmp_path / V4_PILOT_RUN_NAME, "gpu_slots": (0,)}
    pilot = load_contract(ROOT, ROOT / PROFILE_V4_PILOT, **args)
    assert pilot.seed_ids == (157, 887, 2357)
    assert (pilot.initial_labels, pilot.target_labels, pilot.batch_size, pilot.rounds) == (333, 653, 32, 10)
    assert pilot.record()["budget"] == {"first_logical_requests": 30, "maximum_logical_requests": 90,
                                       "maximum_physical_attempts": 180}
    assert not pilot.requires_initial_qualification and full.requires_initial_qualification
    assert full.seed_ids == SEEDS and full.gpu_slots == (0, 1) and full.rounds == 36
    assert pilot.record()["gpu_slots"] == [0]
    assert (pilot.transport, pilot.asset_sha256, pilot.anchor_root, pilot.lease_root, pilot.predecessor) == (
        full.transport, full.asset_sha256, full.anchor_root, full.lease_root, full.predecessor)
    assert set(pilot.predecessor["initial_checkpoint_sha256_by_seed"]) == {str(seed) for seed in SEEDS}
    assert pilot.digest != full.digest
    with pytest.raises(ValueError, match="requires physical GPU slot 0 alone"):
        load_contract(ROOT, ROOT / PROFILE_V4_PILOT, **{**args, "gpu_slots": (0, 1)})
    with pytest.raises(ValueError, match="own registered run name"):
        load_contract(ROOT, ROOT / PROFILE_V4_PILOT, **{**args, "run_directory": tmp_path / V4_RUN_NAME})
    changed = tmp_path / PROFILE_V4_PILOT.name
    once(changed, {**read(ROOT / PROFILE_V4_PILOT), "target_labels": 685})
    with pytest.raises(ValueError, match="differs from its registered values"):
        load_contract(ROOT, changed, **args)
