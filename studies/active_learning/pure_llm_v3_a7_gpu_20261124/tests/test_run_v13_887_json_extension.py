from dataclasses import replace
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import SCHEMA_VERSION
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_JSON_IPV4, RESPONSES_STREAM_IPV4, TransportConfig, TransportFailure, make_request,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import seed_runner
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import make_job, make_result
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import run_v13_887_json_extension as extension


def _fixture():
    config = TransportConfig("token4research", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh",
                             "https://token4research.cn/v1", 1800, 32768, transport_attempts=1, logical_replies=1,
                             wire_mode=RESPONSES_STREAM_IPV4)
    packet = {"seed": 887, "round": 6, "layout": "B", "compression": "lossless",
              "memory": {"mode": "M2"}, "selection_count": 32}
    messages = [{"role": "system", "content": "Synthetic A7 system message"},
                {"role": "user", "content": dumps(packet)}]
    approval = {"synthetic approval": True}
    scope = {"contract_sha256": extension.PREVIOUS_CONTRACT, "authorization_sha256": digest(approval),
             "seed": 887, "mock": False}
    binding = {
        "a7_formal": scope, "seed": 887, "round": 6, "arm": "A7", "prompt_version": "v3",
        "schema_version": SCHEMA_VERSION, "packet_sha256": digest(packet),
        "prompt_sha256": digest(messages[0]["content"]), "base_messages_sha256": digest(messages),
        **{key: digest(key) for key in ("checkpoint_sha256", "state_sha256", "mapping_sha256", "split_sha256")},
    }
    return make_request(binding, 0, messages, config), config, approval


def test_same_science_can_change_only_stream_flag(monkeypatch):
    original, config, _ = _fixture()
    json_config = replace(config, wire_mode=RESPONSES_JSON_IPV4)
    payload = {**original["payload"], "stream": False}
    monkeypatch.setattr(extension, "JSON_PAYLOAD", digest(payload))
    binding = {**original["binding"], "a7_formal": {**original["binding"]["a7_formal"], "contract_sha256": "a" * 64}}
    request = make_request(binding, 0, payload["input"], json_config)
    assert extension._same_json_science(request, original, json_config)
    tampered = make_request({**binding, "checkpoint_sha256": "z" * 64}, 0, payload["input"], json_config)
    assert not extension._same_json_science(tampered, original, json_config)
    with pytest.raises(PermissionError, match="only the stream flag"):
        extension._json_payload(original, replace(json_config, max_output_tokens=4096))
    assert (extension.TIMEOUTS.read_idle_seconds, extension.TIMEOUTS.total_seconds,
            extension.BRIDGE_WAIT_SECONDS) == (1800, 1800, 6000)


def test_previous_cross_id_failure_cannot_be_replaced_by_other_receipt(tmp_path, monkeypatch):
    request, config, approval = _fixture()
    active = tmp_path / "seed_887" / "round_06"
    turn = active / "turn_00"
    seed = request["binding"]["a7_formal"]
    contract = SimpleNamespace(digest=seed["contract_sha256"], transport=config, profile={"wire_representation": None})
    started = {"request_id": request["request_id"], "attempt": 0, "attempt_id": extension.PREVIOUS_ATTEMPT,
               "started_unix": 1, "transport": "mac_bridge"}
    job = make_job(request, config, started)
    result = make_result(job, failure=TransportFailure("response_id_mismatch", ambiguous=True), credential_id="key_01",
                         provider_dispatched=True)
    terminal = {**started, "status": "stopped", "error": "response_id_mismatch", "ambiguous": True,
                "provider_dispatched": True, "circuit_open": False,
                "bridge_job_sha256": job["job_sha256"], "bridge_result_sha256": result["result_sha256"]}
    monkeypatch.setattr(extension, "LANE", replace(extension.LANE, payload_sha256=request["payload_sha256"]))
    monkeypatch.setattr(extension, "PREVIOUS_REQUEST", request["request_id"])
    monkeypatch.setattr(extension, "PREVIOUS_JOB", job["job_sha256"])
    monkeypatch.setattr(extension, "PREVIOUS_RESULT", result["result_sha256"])
    once(turn / "request.json", request)
    once(turn / "attempt_00.started.json", started)
    once(turn / "attempt_00.terminal.json", terminal)
    once(turn / "formal_request_authorization.json", seed_runner._selection_grant(seed, request["request_id"]))
    for folder, value in (("outbox", job), ("results", result)):
        path = tmp_path / "bridge" / folder / f"{started['attempt_id']}.json"
        once(path, value)
        path.chmod(0o600)
    assert extension._previous_attempt(tmp_path, active, contract, approval, request) == {
        "attempt": 0, "attempt_id": started["attempt_id"], "job_sha256": job["job_sha256"],
        "result_sha256": result["result_sha256"], "terminal_sha256": digest(terminal),
        "error": "response_id_mismatch", "provider_dispatched": True, "ambiguous": True,
    }
    once(turn / "attempt_00.reconciliation.json", {"decision": "synthetic recovered answer"})
    with pytest.raises(PermissionError, match="unreconciled"):
        extension._previous_attempt(tmp_path, active, contract, approval, request)


def test_new_release_only_replaces_json_overlays(tmp_path):
    names = extension.OVERLAYS | {"scientific_prompt.md"}
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("old")
    prior = SimpleNamespace(source_sha256={name: file_digest(tmp_path / name) for name in names})
    for name in extension.OVERLAYS:
        (tmp_path / name).write_text("new JSON capability")
    for module in ("run_v13_887_json_extension", extension.DISPATCHER):
        path = tmp_path / extension.STUDY / "scripts" / f"{module}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("signed new entry")
    sources = extension._sources(tmp_path, prior)
    assert {name for name in names if sources[name] != prior.source_sha256[name]} == extension.OVERLAYS
    (tmp_path / "scientific_prompt.md").write_text("unapproved")
    with pytest.raises(PermissionError, match="only the JSON provider"):
        extension._sources(tmp_path, prior)
