"""Offline checks for the scheduling addendum and custody of the V13 dispatcher."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, file_digest, once
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import dispatch_prepared_ipv4_parallel as parallel


CONTRACT = "a" * 64
AUTHORIZATION = "b" * 64
DISPATCHER = "c" * 32


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(parallel.os, "umask", lambda _: None)
    spools = tmp_path / ".config/purellm/spool"
    spool = spools / CONTRACT
    for directory in parallel.JOURNAL_DIRS:
        (spool / directory).mkdir(mode=0o700, parents=True)
    spool.chmod(0o700)
    once(spool / "run_binding.json", {"contract_sha256": CONTRACT, "authorization_sha256": AUTHORIZATION, "mock": False})
    once(spool / "dispatcher_identity.json", {"dispatcher_id": DISPATCHER})
    once(spool / "provider_model_lock.json", {"model": "unchanged"})
    approval = {"schema": parallel.SCHEMA, "contract_sha256": CONTRACT, "authorization_sha256": AUTHORIZATION,
                "launcher_sha256": file_digest(Path(parallel.__file__)), "dispatcher_id": DISPATCHER,
                "previous_api_parallelism": 1, "api_parallelism": 3, "authorize_paid_api_scheduling": True,
                "authorized_by": "fixture", "authorization_reference": "fixture approval"}
    remote = "/example/a7-ipv4-v13"

    def invoke(schedule, supplied_sha=None):
        path = tmp_path / "schedule.json"
        once(path, schedule)
        monkeypatch.setattr(sys, "argv", [
            "dispatch_prepared_ipv4_parallel", "--execute-paid", "--remote-root", "/release",
            "--remote-run-dir", remote, "--contract-sha256", CONTRACT,
            "--authorization-sha256", AUTHORIZATION, "--schedule-authorization", str(path),
            "--schedule-authorization-sha256", supplied_sha or digest(schedule),
        ])
        parallel.main()

    return spool, remote, approval, invoke


@pytest.mark.parametrize("pool_cap", [2, 3])
def test_uses_original_spool_pool_and_dispatcher_at_the_effective_cap(configured, monkeypatch, pool_cap):
    spool, remote, approval, invoke = configured
    info, transport, original_transport = object(), object(), object()
    pilot = spool.parent / parallel.PILOT_CONTRACT_SHA256
    pool = SimpleNamespace(directory=pilot / "key_pool", config=SimpleNamespace(parallelism=pool_cap))
    courier = SimpleNamespace(info=lambda: info)
    dispatcher = SimpleNamespace(dispatcher_id=DISPATCHER)

    def verify(actual, *, expected_contract, expected_auth):
        assert (actual, expected_contract, expected_auth) == (info, CONTRACT, AUTHORIZATION)
        return info

    def original_pool(path):
        assert path == pilot
        return pool, None, original_transport

    def original_approval(actual, run, previous):
        assert (actual, run, previous) == (info, remote, original_transport)
        return transport, {"model": "unchanged"}, "sealed-prompt"

    def shared_pool(directory, config):
        assert (directory, config) == (pool.directory, pool.config)
        return pool

    def original_dispatcher(path, **options):
        assert path == spool
        assert options == {"info": info, "contract_sha256": CONTRACT, "transport": transport, "mock": False, "pool": pool}
        return dispatcher

    def execute(actual, endpoint, *, watch, workers):
        assert (actual, endpoint, watch, workers) == (dispatcher, courier, True, pool_cap)
        assert dispatcher.prompt_sha == "sealed-prompt"
        return {"started": True}

    monkeypatch.setattr(parallel, "SSHCourier", lambda *_: courier)
    monkeypatch.setattr(parallel, "verify_info", verify)
    monkeypatch.setattr(parallel, "_pool", original_pool)
    monkeypatch.setattr(parallel, "_approved", original_approval)
    monkeypatch.setattr(parallel, "SharedKeyPool", shared_pool)
    monkeypatch.setattr(parallel, "PromptDispatcher", original_dispatcher)
    monkeypatch.setattr(parallel, "run_dispatcher", execute)
    invoke(approval)


@pytest.mark.parametrize("changes", [
    {"contract_sha256": "d" * 64}, {"authorization_sha256": "d" * 64}, {"dispatcher_id": "d" * 32},
    {"launcher_sha256": "d" * 64}, {"previous_api_parallelism": True}, {"api_parallelism": 2},
    {"authorize_paid_api_scheduling": False}, {},
])
def test_invalid_schedule_is_refused_before_ssh(configured, monkeypatch, changes):
    _, _, approval, invoke = configured
    monkeypatch.setattr(parallel, "SSHCourier", lambda *_: pytest.fail("invalid authorization opened SSH"))
    with pytest.raises(PermissionError):
        invoke({**approval, **changes}, None if changes else "d" * 64)


def test_missing_spool_custody_is_refused_before_ssh(configured, monkeypatch):
    spool, _, approval, invoke = configured
    (spool / "attempts").rmdir()
    monkeypatch.setattr(parallel, "SSHCourier", lambda *_: pytest.fail("lost spool opened SSH"))
    with pytest.raises(FileNotFoundError):
        invoke(approval)
