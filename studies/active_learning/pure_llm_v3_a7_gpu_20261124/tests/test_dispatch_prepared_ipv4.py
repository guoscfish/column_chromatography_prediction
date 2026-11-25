"""Offline checks for the signed Mac IPv4 entry point."""

from copy import deepcopy
from dataclasses import asdict, replace
import sys

import pytest

from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import RequestTimeouts
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import dispatch_prepared_ipv4 as dispatcher


@pytest.fixture
def approved(monkeypatch):
    pilot = dispatcher.TransportConfig(
        "token4research", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh", "https://token4research.cn/v1",
        1800, 32768, transport_attempts=2, sdk_version="2.28.0",
    )
    timing = RequestTimeouts(read_idle_seconds=1800, total_seconds=3600)
    transport = replace(pilot, timeout_seconds=timing.total_seconds, wire_mode=dispatcher.RESPONSES_STREAM_IPV4)
    identity = {"provider": "token4research", "requested_model": "gpt-6.2-astra", "served_model": "gpt-6.2-astra",
                "requested_reasoning_effort": "xhigh", "reported_reasoning_effort": "xhigh",
                "reported_model_version": None}
    info = {"contract": {
        "run_directory": "/a7-ipv4-cohort", "api_execution": dispatcher.SSH_API,
        "predecessor": {"dispatcher_module": dispatcher.MODULE, "provider_qualification": {"model_identity": identity}},
        "profile": {"provider_capability": "gated_ipv4", "arm": "A7", "memory": "M2", "device": "cuda",
                    "initial_labels": 333, "target_labels": 1485, "batch_size": 32, "api_parallelism": 1,
                    "store": False, "confirmation_seeds": list(dispatcher.SEEDS),
                    "wire_representation": dispatcher.FORMAL_WIRE_REPRESENTATION,
                    "resume_boundaries": {str(seed): {} for seed in dispatcher.SEEDS},
                    "request_timeouts": asdict(timing), "transport": asdict(transport)},
        "source_sha256": {path.as_posix(): "a" * 64 for path in dispatcher.SOURCES},
    }}
    monkeypatch.setattr(dispatcher, "sdk_status", lambda _: {"matches": True})
    monkeypatch.setattr(dispatcher, "verify_local_source", lambda *_, **__: None)
    monkeypatch.setattr(dispatcher, "file_digest", lambda _: "a" * 64)
    monkeypatch.setattr(dispatcher, "_approved_prompt_sha", lambda _: "prompt")
    return info, pilot, transport, identity


def test_ipv4_scope_requires_exact_serial_route_sources_and_model(approved):
    info, pilot, transport, identity = approved
    assert dispatcher._approved(info, "/a7-ipv4-cohort", pilot) == (transport, identity, "prompt")
    for changes in (
        {"provider_capability": "draft"}, {"api_parallelism": 3}, {"api_parallelism": True},
        {"store": True}, {"confirmation_seeds": [157]}, {"fixed_reference_repair": {}},
        {"transport": {**asdict(transport), "wire_mode": "responses_stream"}},
        {"transport": {**asdict(transport), "transport_attempts": 3}},
    ):
        bad = deepcopy(info)
        bad["contract"]["profile"].update(changes)
        with pytest.raises(PermissionError):
            dispatcher._approved(bad, "/a7-ipv4-cohort", pilot)
    for source in dispatcher.SOURCES:
        bad = deepcopy(info)
        del bad["contract"]["source_sha256"][source.as_posix()]
        with pytest.raises(PermissionError, match="source differs"):
            dispatcher._approved(bad, "/a7-ipv4-cohort", pilot)
    bad = deepcopy(info)
    bad["contract"]["predecessor"]["provider_qualification"]["model_identity"]["served_model"] = "gpt-6.2-sol"
    with pytest.raises(PermissionError, match="model lock differs"):
        dispatcher._approved(bad, "/a7-ipv4-cohort", pilot)


def test_command_requires_paid_flag_before_ssh(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "dispatch_prepared_ipv4", "--remote-root", "/staged", "--remote-run-dir", "/a7-ipv4-cohort",
        "--contract-sha256", "a" * 64, "--authorization-sha256", "b" * 64,
        "--pilot-spool", "/pilot", "--spool-dir", "/new-spool",
    ])
    monkeypatch.setattr(dispatcher, "SSHCourier", lambda *_: pytest.fail("opened SSH without paid flag"))
    with pytest.raises(SystemExit) as error:
        dispatcher.main()
    assert error.value.code == 2
