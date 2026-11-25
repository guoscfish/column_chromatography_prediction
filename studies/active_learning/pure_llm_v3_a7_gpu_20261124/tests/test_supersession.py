"""The new authorization is separate even though the scientific inputs are the same."""

from copy import deepcopy

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import SCHEMA_VERSION
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportFailure, make_request
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import supersession
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import make_job, make_result
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    PROFILE, PROFILE_V3, ROOT, SEEDS, SSH_API, load_contract, verify_authorization,
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
    attempt, dispatcher = "1" * 32, "2" * 32
    started = {"request_id": request["request_id"], "attempt": 0, "attempt_id": attempt,
               "started_unix": 1.0, "transport": "mac_bridge"}
    job = make_job(request, contract.transport, started)
    result = make_result(job, failure=TransportFailure("provider_reasoning_rejected"), credential_id="fixture-key",
                         provider_dispatched=True)
    monkeypatch.setattr(supersession, "V2_RESULT_SHA256", result["result_sha256"])
    once(v2 / "formal_contract.json", contract.record())
    once(v2 / "authorization.json", approval)
    once(v2 / "pause.request", {"reason": "offline fixture"})
    turn = v2 / "seed_157/round_00/turn_00"
    once(turn / "request.json", request)
    once(turn / "formal_request_authorization.json", {"scope": "paid_selection_only", "formal_run": scope,
                                                      "request_id": request["request_id"]})
    once(turn / "attempt_00.started.json", started)
    once(turn / "attempt_00.terminal.json", {
        **started, "status": "stopped", "error": "provider_reasoning_rejected", "ambiguous": False,
        "bridge_job_sha256": job["job_sha256"], "bridge_result_sha256": result["result_sha256"],
        "provider_dispatched": True,
    })
    once(v2 / "bridge/dispatcher.json", {"dispatcher_id": dispatcher, "contract_sha256": contract.digest})
    once(v2 / f"bridge/outbox/{attempt}.json", job)
    once(v2 / f"bridge/claims/{attempt}.json", {"dispatcher_id": dispatcher, "attempt_id": attempt,
                                              "job_sha256": job["job_sha256"]})
    once(v2 / f"bridge/results/{attempt}.json", result)
    return contract, approval, v2, original


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
