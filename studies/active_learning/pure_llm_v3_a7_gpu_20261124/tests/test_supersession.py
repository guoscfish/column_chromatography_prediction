"""The new authorization is separate even though the scientific inputs are the same."""

from copy import deepcopy

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import make_request
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import supersession
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import (
    PROFILE, ROOT, SSH_API, load_contract, verify_authorization,
)


@pytest.fixture
def original_run(tmp_path, monkeypatch):
    original = tmp_path / "original"
    anchor, lease = tmp_path / "anchors", tmp_path / "lease"
    direct = load_contract(ROOT, ROOT / PROFILE, anchor_root=anchor, lease_root=lease,
                           run_directory=original, gpu_slots=(0, 1))
    monkeypatch.setattr(supersession, "PREDECESSOR_SHA256", direct.digest)
    approval = {"scope": "pure_llm_v3_a7_formal", "contract_sha256": direct.digest,
                "authorized_by": "synthetic test", "authorization_reference": "no live provider",
                "authorize_paid_api": True, "authorize_label_reveal": True, "authorize_cuda_training": True}
    binding = {"seed": 157, "round": 0, "checkpoint_sha256": "3" * 64,
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
