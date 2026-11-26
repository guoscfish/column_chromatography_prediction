from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
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
    timing, profile, bridge_wait = extension._lane_timing(lane)
    authorized = replace(contract.transport, logical_replies=1, transport_attempts=1, timeout_seconds=timing.total_seconds)
    candidate = make_request(extended_binding, 0, request["payload"]["input"], authorized)
    assert previous._same_science(candidate, request)
    assert (timing.connect_seconds, timing.write_seconds) == (30, 120)
    if seed == 157:
        assert (timing.read_idle_seconds, timing.total_seconds, bridge_wait) == (1800, 3600, 7800)
        assert profile == {"sse_inactivity_seconds": 900}
    else:
        assert (timing.read_idle_seconds, timing.total_seconds, bridge_wait) == (900, 1800, 6000)
        assert profile == {}
    assert 36 - lane.round - 1 == {157: 32, 6101: 34, 887: 29}[seed]


def test_157_requires_second_terminal_record(tmp_path):
    lane = LANES[157]
    turn, request, contract, approval = _archive(tmp_path, lane)
    (turn / "attempt_01.terminal.json").unlink()
    with pytest.raises(FileNotFoundError):
        _retired_attempts(tmp_path, turn, lane, request, contract, approval)


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


def test_157_boundary_authenticates_both_v13_cuda_transitions(tmp_path, monkeypatch):
    from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import backend

    @dataclass(frozen=True)
    class State:
        round: int
        checkpoint: str

    lane = LANES[157]
    parent, preparation, cache = (tmp_path / name for name in ("parent", "prep", "cache"))
    origin = preparation / "boundaries" / f"seed_{lane.seed}"
    origin.mkdir(parents=True)
    once(cache / "resource_receipt.json", {"fixture": "shared cache"})
    state = State(1, "1" * 64)
    memory = {"rounds": [0]}
    descriptor = {"round": 1, "state_sha256": digest(asdict(state)), "memory_sha256": digest(memory)}
    row = {"descriptor": descriptor, "cache_directory": str(cache), "boundary_directory": str(origin),
           "evidence": {"public_resource_receipt_sha256": file_digest(cache / "resource_receipt.json")}}
    draft = {"boundaries": {str(lane.seed): row}}
    once(preparation / "five_seed_boundary_draft.json", draft)
    contract = SimpleNamespace(digest="d" * 64, transport=object(),
                               profile={"resume_boundaries": {str(lane.seed): descriptor}},
                               predecessor={
                                   "prepared_draft_sha256": digest(draft),
                                   "lanes": {str(lane.seed): {**row, "start_round": 1, "checkpoint_sha256": state.checkpoint}},
                               })
    approval = {"fixture": True}
    scope = extension.seed_runner._scope(contract, approval, lane.seed, False)
    seed = parent / f"seed_{lane.seed}"
    once(seed / "boundary.json", {"formal_run": scope, **descriptor})
    once(seed / "round_01/source.json", asdict(state))
    once(seed / "round_01/boundary_memory.json", memory)
    once(seed / "round_03/transition.json", {"round": 3, "checkpoint": "3" * 64})
    calls = []

    def verify_fit(before, current, after, supplied_scope, *_):
        assert supplied_scope == scope
        calls.append((before.name, after.name))
        expected = extension.read(after / "transition.json")
        assert expected["round"] == current.round + 1
        return State(expected["round"], expected["checkpoint"]), {"rounds": list(range(expected["round"]))}

    monkeypatch.setattr(extension, "read_boundary", lambda *_: (state, memory, 1))
    monkeypatch.setattr(backend, "TrainingBackend", lambda *_: "fixture CUDA backend")
    monkeypatch.setattr(extension.seed_runner, "_successor", verify_fit)
    with pytest.raises(FileNotFoundError):
        extension._boundary(tmp_path, parent, lane, contract, approval)
    assert calls == [("round_01", "round_02")]
    once(seed / "round_02/transition.json", {"round": 2, "checkpoint": "2" * 64})
    calls.clear()
    manifest, _, fitted, history = extension._boundary(tmp_path, parent, lane, contract, approval)
    assert calls == [("round_01", "round_02"), ("round_02", "round_03")]
    assert manifest == seed / "round_03" and fitted == State(3, "3" * 64) and history == {"rounds": [0, 1, 2]}



def test_157_only_overlays_transport_sources_after_verifying_original(tmp_path):
    original = tmp_path / "original"
    release = tmp_path / "release"
    science = "studies/active_learning/scientific_prompt.md"
    filenames = extension.SSE_TRANSPORT_OVERLAYS | {science}
    for root in (original, release):
        for name in filenames:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"original {name}")
    expected = {name: file_digest(original / name) for name in filenames}
    assert extension._source_hashes(release, original, expected, LANES[6101]) == expected
    with pytest.raises(PermissionError, match="precisely the approved transport"):
        extension._source_hashes(release, original, expected, LANES[157])

    for name in extension.SSE_TRANSPORT_OVERLAYS:
        (release / name).write_text(f"new inactivity {name}")
    signed = extension._source_hashes(release, original, expected, LANES[157])
    assert {name for name in signed if signed[name] != expected[name]} == extension.SSE_TRANSPORT_OVERLAYS
    assert signed == {name: file_digest(release / name) for name in filenames}

    with pytest.raises(PermissionError, match="precisely the approved transport"):
        extension._source_hashes(release, original, expected, LANES[6101])
    (release / science).write_text("unapproved draft")
    with pytest.raises(PermissionError, match="precisely the approved transport"):
        extension._source_hashes(release, original, expected, LANES[157])
    (original / next(iter(extension.SSE_TRANSPORT_OVERLAYS))).write_text("tampered original transport")
    with pytest.raises(PermissionError, match="original V13 release changed"):
        extension._source_hashes(release, original, expected, LANES[157])
