from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, file_digest, once
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_bounded_sse_recovery as recovery


@pytest.mark.parametrize("seed", sorted(recovery.LANES))
def test_only_the_archived_json_stream_flag_may_differ(seed):
    lane = recovery.LANES[seed]
    payload = {"input": [{"role": "system", "content": "scientific selection"}], "stream": True,
               "store": False, "model": "gpt-6.2-astra", "reasoning": {"effort": "xhigh"}}
    binding = {"checkpoint_sha256": "checkpoint", "packet_sha256": "packet", "a7_formal": {"scope": "before"}}
    original = {"payload": payload, "payload_sha256": digest(payload), "binding": binding}
    previous_payload = {**payload, "stream": lane.previous_wire != "responses_json_ipv4"}
    previous = {"payload": previous_payload, "payload_sha256": digest(previous_payload),
                "binding": {**binding, "a7_formal": {"scope": "after"}}}
    assert recovery._same_payload(original, previous, lane)
    altered = {**previous_payload, "input": [{"role": "system", "content": "different guidance"}]}
    assert not recovery._same_payload(original, {**previous, "payload": altered, "payload_sha256": digest(altered)}, lane)
    assert not recovery._same_payload(original, {**previous, "binding": {**binding, "checkpoint_sha256": "other"}}, lane)


def test_old_slot_and_recovered_terminal_prevent_a_bounded_attempt(tmp_path):
    for index in (0, 1):
        once(tmp_path / "turn_00" / f"attempt_{index:02d}.started.json", {"attempt": index})
    assert recovery._turn(tmp_path, 2).name == "turn_00"
    with pytest.raises(PermissionError, match="different reservations"):
        recovery._turn(tmp_path, 1)
    once(tmp_path / "turn_00/attempt_01.reconciliation.json", {"decision": "import_completed"})
    with pytest.raises(PermissionError, match="recovered completion"):
        recovery._turn(tmp_path, 2)


def test_only_transport_sources_may_change(tmp_path):
    names = recovery.TRANSPORT_SOURCES | {"A7_prompt.md"}
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("old")
    original = SimpleNamespace(source_sha256={name: file_digest(tmp_path / name) for name in names})
    for name in recovery.TRANSPORT_SOURCES:
        (tmp_path / name).write_text("new signed SSE transport")
    for module in ("run_v13_bounded_sse_recovery", recovery.DISPATCHER, "run_v13_lane_extension",
                   "run_v13_seed2357_extension", "dispatch_prompt_direct", "dispatch_prompt_pilot", "dispatch_prompt_successor", "dispatch_v13_lane_extension"):
        path = tmp_path / recovery.STUDY / "scripts" / f"{module}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("new entry point")
    sources = recovery._sources(tmp_path, original)
    assert {name for name in names if sources[name] != original.source_sha256[name]} == recovery.TRANSPORT_SOURCES
    assert recovery.TIMEOUTS.bridge_wait_seconds == 7800
    (tmp_path / "A7_prompt.md").write_text("other science")
    with pytest.raises(PermissionError, match="outside the authorized transport"):
        recovery._sources(tmp_path, original)


def test_sparse_archive_only_inherits_a_missing_identically_signed_source(tmp_path):
    original, previous = (tmp_path / directory for directory in ("original", "previous"))
    name = "application/QGeoGNN.py"
    (original / name).parent.mkdir(parents=True)
    (original / name).write_text("exact original")
    expected = file_digest(original / name)
    inherited = original, {name: expected}
    assert recovery._archived_source(previous, name, expected, inherited) == expected
    with pytest.raises(FileNotFoundError):
        recovery._archived_source(previous, name, "different digest", inherited)
    with pytest.raises(FileNotFoundError):
        recovery._archived_source(previous, name, expected, (original, {}))
    (previous / name).parent.mkdir(parents=True)
    (previous / name).write_text("mismatched existing source")
    assert recovery._archived_source(previous, name, expected, inherited) != expected
    (previous / name).unlink()
    (previous / name).symlink_to("missing-target")
    with pytest.raises(FileNotFoundError):
        recovery._archived_source(previous, name, expected, inherited)
