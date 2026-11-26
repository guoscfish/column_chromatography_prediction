from dataclasses import replace
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import SCHEMA_VERSION
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_STREAM_IPV4, RESPONSES_WEBSOCKET_IPV4, TransportConfig, TransportFailure, make_request,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import make_job, make_result
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_6101_ws_extension as extension


def _request():
    config = TransportConfig("token4research", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh",
                             "https://token4research.cn/v1", 1800, 32768, transport_attempts=1, logical_replies=1,
                             wire_mode=RESPONSES_STREAM_IPV4)
    packet = {"seed": 6101, "round": 1, "layout": "B", "compression": "lossless",
              "memory": {"mode": "M2"}, "selection_count": 32}
    messages = [{"role": "system", "content": "Synthetic A7 system message"},
                {"role": "user", "content": dumps(packet)}]
    approval = {"synthetic approval": True}
    scope = {"contract_sha256": "a" * 64, "authorization_sha256": digest(approval), "seed": 6101, "mock": False}
    binding = {
        "a7_formal": scope, "seed": 6101, "round": 1, "arm": "A7", "prompt_version": "v3",
        "schema_version": SCHEMA_VERSION, "packet_sha256": digest(packet),
        "prompt_sha256": digest(messages[0]["content"]), "base_messages_sha256": digest(messages),
        **{key: digest(key) for key in ("checkpoint_sha256", "state_sha256", "mapping_sha256", "split_sha256")},
    }
    return make_request(binding, 0, messages, config), config, approval


def test_signed_websocket_body_stays_identical():
    original, config, _ = _request()
    lane = replace(extension.LANE, payload_sha256=original["payload_sha256"])
    ws = replace(config, wire_mode=RESPONSES_WEBSOCKET_IPV4, timeout_seconds=3600)
    extension._same_wire(original, ws, lane)
    binding = {**original["binding"], "a7_formal": {**original["binding"]["a7_formal"], "contract_sha256": "b" * 64}}
    new = make_request(binding, 0, original["payload"]["input"], ws)
    assert extension.earlier.previous._same_science(new, original)
    assert new["payload"]["stream"] is True and new["payload"]["store"] is False
    assert extension.INACTIVITY_SECONDS == 900
    assert extension.WEBSOCKETS_VERSION == "15.0.1"
    assert (extension.TIMEOUTS.read_idle_seconds, extension.TIMEOUTS.total_seconds,
            extension.TIMEOUTS.bridge_wait_seconds) == (1800, 3600, 7800)
    with pytest.raises(PermissionError, match="retain the entire"):
        extension._same_wire(original, replace(ws, max_output_tokens=4096), lane)


def test_preceding_sse_failure_must_be_exact_and_unreconciled(tmp_path):
    request, config, approval = _request()
    lane = replace(extension.LANE, payload_sha256=request["payload_sha256"])
    active = tmp_path / "seed_6101" / "round_01"
    turn = active / "turn_00"
    scope = request["binding"]["a7_formal"]
    contract = SimpleNamespace(digest=scope["contract_sha256"], transport=config, profile={"wire_representation": None})
    started = {"request_id": request["request_id"], "attempt": 0, "attempt_id": "d" * 32,
               "started_unix": 1, "transport": "mac_bridge"}
    job = make_job(request, config, started)
    result = make_result(job, failure=TransportFailure("response_id_mismatch", ambiguous=True), credential_id="key_01",
                         provider_dispatched=True)
    terminal = {**started, "status": "stopped", "error": "response_id_mismatch", "ambiguous": True,
                "provider_dispatched": True, "circuit_open": False,
                "bridge_job_sha256": job["job_sha256"], "bridge_result_sha256": result["result_sha256"]}
    spec = extension.FailedOneShot(contract.digest, digest(approval), started["attempt_id"], request["request_id"],
                                  job["job_sha256"], result["result_sha256"])
    once(turn / "request.json", request)
    once(turn / "attempt_00.started.json", started)
    once(turn / "attempt_00.terminal.json", terminal)
    once(turn / "formal_request_authorization.json", seed_runner._selection_grant(scope, request["request_id"]))
    for folder, value in (("outbox", job), ("results", result)):
        path = tmp_path / "bridge" / folder / f"{started['attempt_id']}.json"
        once(path, value)
        path.chmod(0o600)
    assert extension._previous_attempt(tmp_path, active, lane, contract, approval, request, spec) == {
        "attempt": 0, "attempt_id": spec.attempt_id, "job_sha256": spec.job_sha256,
        "result_sha256": spec.result_sha256, "terminal_sha256": digest(terminal),
        "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True,
    }
    once(turn / "attempt_00.reconciliation.json", {"decision": "synthetic later completion"})
    with pytest.raises(PermissionError, match="unreconciled"):
        extension._previous_attempt(tmp_path, active, lane, contract, approval, request, spec)


def test_release_overlays_websocket_transport_only(tmp_path):
    names = extension.OVERLAYS | {"A7_prompt.md"}
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("old")
    previous = SimpleNamespace(source_sha256={name: file_digest(tmp_path / name) for name in names})
    for name in extension.OVERLAYS:
        (tmp_path / name).write_text("new WS route")
    for module in ("run_v13_6101_ws_extension", extension.DISPATCHER):
        path = tmp_path / extension.STUDY / "scripts" / f"{module}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("new signed entry point")
    result = extension._sources(tmp_path, previous)
    assert {name for name in names if result[name] != previous.source_sha256[name]} == extension.OVERLAYS
    (tmp_path / "A7_prompt.md").write_text("changed prompt")
    with pytest.raises(PermissionError, match="only the signed WebSocket"):
        extension._sources(tmp_path, previous)
