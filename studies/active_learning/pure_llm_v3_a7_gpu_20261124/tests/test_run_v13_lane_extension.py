from dataclasses import replace
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import SCHEMA_VERSION
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_STREAM_IPV4, TransportConfig, TransportFailure, make_request,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import make_job, make_result
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_lane_extension as extension
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.run_v13_lane_extension import (
    LANES, TIMEOUTS, _retired_attempts, _uncommitted, previous,
)


def _original(lane):
    config = TransportConfig("token4research", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh",
                             "https://token4research.cn/v1", 3600, 32768, transport_attempts=2,
                             wire_mode=RESPONSES_STREAM_IPV4)
    packet = {"seed": lane.seed, "round": lane.round, "layout": "B", "compression": "lossless",
              "memory": {"mode": "M2"}, "selection_count": 32}
    messages = [{"role": "system", "content": "Synthetic A7 system message"},
                {"role": "user", "content": dumps(packet)}]
    approval = {"synthetic approval": True}
    scope = {"contract_sha256": "a" * 64, "authorization_sha256": digest(approval),
             "seed": lane.seed, "mock": False}
    binding = {
        "a7_formal": scope, "seed": lane.seed, "round": lane.round, "arm": "A7", "prompt_version": "v3",
        "schema_version": SCHEMA_VERSION, "packet_sha256": digest(packet),
        "prompt_sha256": digest(messages[0]["content"]), "base_messages_sha256": digest(messages),
        **{key: digest(key) for key in ("checkpoint_sha256", "state_sha256", "mapping_sha256", "split_sha256")},
    }
    return make_request(binding, 0, messages, config), config, approval


def _archive(parent, lane, *, preflight_posted=False):
    original, config, approval = _original(lane)
    turn = parent / f"seed_{lane.seed}/round_{lane.round:02d}/turn_00"
    contract = SimpleNamespace(digest=original["binding"]["a7_formal"]["contract_sha256"],
                               transport=config, profile={"wire_representation": None})
    for index, attempt_id in enumerate(lane.attempt_ids):
        posted = bool(index)
        dispatched = posted or preflight_posted
        error = lane.posted_error if posted else "connect_error"
        started = {"request_id": original["request_id"], "attempt": index, "attempt_id": attempt_id,
                   "started_unix": 1, "transport": "mac_bridge"}
        job = make_job(original, config, started)
        failure = TransportFailure(error, ambiguous=posted, retryable=not posted)
        result = make_result(job, failure=failure, credential_id="key_01", provider_dispatched=dispatched)
        terminal = {**started, "status": "stopped" if posted else "retryable", "error": error,
                    "ambiguous": posted, "provider_dispatched": dispatched, "circuit_open": False,
                    "bridge_job_sha256": job["job_sha256"], "bridge_result_sha256": result["result_sha256"]}
        once(turn / f"attempt_{index:02d}.started.json", started)
        once(turn / f"attempt_{index:02d}.terminal.json", terminal)
        for folder, record in (("outbox", job), ("results", result)):
            path = parent / "bridge" / folder / f"{attempt_id}.json"
            once(path, record)
            path.chmod(0o600)
    return turn, original, contract, approval


@pytest.mark.parametrize("seed", sorted(LANES))
def test_exact_preflight_and_posted_receipts_and_shorter_transport(tmp_path, seed):
    lane = LANES[seed]
    turn, request, contract, approval = _archive(tmp_path, lane)
    attempts = _retired_attempts(tmp_path, turn, lane, request, contract, approval)
    assert [(item["attempt_id"], item["provider_dispatched"], item["ambiguous"], item["error"]) for item in attempts] == [
        (lane.attempt_ids[0], False, False, "connect_error"), (lane.attempt_ids[1], True, True, lane.posted_error),
    ]
    old_binding = request["binding"]
    extended_binding = {**old_binding, "a7_formal": {**old_binding["a7_formal"], "contract_sha256": "b" * 64}}
    shortened = replace(contract.transport, logical_replies=1, transport_attempts=1, timeout_seconds=TIMEOUTS.total_seconds)
    candidate = make_request(extended_binding, 0, request["payload"]["input"], shortened)
    assert previous._same_science(candidate, request)
    assert [TIMEOUTS.connect_seconds, TIMEOUTS.write_seconds, TIMEOUTS.read_idle_seconds,
            TIMEOUTS.total_seconds] == [30, 120, 900, 1800]
    assert 36 - lane.round - 1 == {6101: 34, 887: 29}[seed]


def test_rejects_false_preflight_claim_and_original_advance(tmp_path):
    lane = LANES[6101]
    turn, request, contract, approval = _archive(tmp_path, lane, preflight_posted=True)
    with pytest.raises(PermissionError, match="sealed preflight"):
        _retired_attempts(tmp_path, turn, lane, request, contract, approval)
    with pytest.raises(PermissionError, match="pause just this original"):
        _uncommitted(tmp_path, lane, require_paused=True)
    once(turn.parent.parent / "pause.request", {"seed": lane.seed})
    _uncommitted(tmp_path, lane, require_paused=True)
    once(turn.parent / "batch_freeze.json", {"already selected": True})
    with pytest.raises(PermissionError, match="already selected"):
        _uncommitted(tmp_path, lane, require_paused=True)



def test_bridge_wait_covers_legacy_queue_without_lengthening_provider_deadline(tmp_path, monkeypatch):
    lane = LANES[6101]
    request, config, _ = _original(lane)
    managed = SimpleNamespace(wait_seconds=TIMEOUTS.bridge_wait_seconds)
    contract = SimpleNamespace(gpu_slots=(0,), lease_root=tmp_path,
                               predecessor={"bridge_wait_seconds": extension.BRIDGE_WAIT_SECONDS})
    monkeypatch.setattr(extension.scheduler, "_provider_wiring",
                        lambda *_: (None, {"managed_attempt": managed}))

    def fake_seed(**_kwargs):
        assert managed.wait_seconds == config.timeout_seconds + TIMEOUTS.total_seconds + TIMEOUTS.bridge_margin_seconds
        assert managed.wait_seconds == 6000 and TIMEOUTS.total_seconds == 1800
        return extension.seed_runner.SeedProgress(lane.seed, "paused", lane.round, 365)

    monkeypatch.setattr(extension.seed_runner, "run_seed", fake_seed)
    result = extension._run(tmp_path, tmp_path, tmp_path, lane, contract, {}, None, None, request, request)
    assert result.status == "paused"


def test_only_887_can_rebind_one_derived_message_hash_with_identical_payload():
    lane = LANES[887]
    base, config, _ = _original(lane)
    field = extension.AUTHORITATIVE_MESSAGE_HASH
    old = {**base["binding"], field: "a" * 64}
    rebuilt = {**old, field: "b" * 64}
    messages = base["payload"]["input"]
    original = make_request(old, 0, messages, config)
    regenerated = make_request(rebuilt, 0, messages, config)
    proof = extension._reconstruction(lane, original, regenerated, config)["authoritative_message_reconstruction"]

    new_binding = {**rebuilt, "a7_formal": {**rebuilt["a7_formal"], "contract_sha256": "e" * 64}}
    new_config = replace(config, logical_replies=1, transport_attempts=1, timeout_seconds=1800)
    candidate = make_request(new_binding, 0, messages, new_config)
    assert extension._same_science(candidate, original, proof)
    assert not extension._same_science(candidate, original, None)
    assert not extension._same_science(candidate, original, {**proof, "rebuilt_sha256": "d" * 64})
    assert extension._reconstruction(LANES[6101], original, original, config) == {}
    with pytest.raises(PermissionError, match="pinned host serialization"):
        extension._reconstruction(LANES[6101], original, regenerated, config)
    with pytest.raises(PermissionError, match="pinned host serialization"):
        extension._reconstruction(lane, original, make_request({**rebuilt, "packet_sha256": "f" * 64},
                                                               0, messages, config), config)
