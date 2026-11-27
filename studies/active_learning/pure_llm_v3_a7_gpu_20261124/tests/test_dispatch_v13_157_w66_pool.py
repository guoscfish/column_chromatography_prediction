from dataclasses import replace
import fcntl

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_STREAM_IPV4,
    TransportConfig,
    TransportFailure,
    TransportStop,
    make_request,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.key_pool import LEASE, KeyPool, PoolConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts import dispatch_v13_157_w66_pool as w66


@pytest.fixture
def scope(tmp_path, monkeypatch):
    key = tmp_path / "test-key"
    key.touch(mode=0o600)
    t = TransportConfig("token4research", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh",
                        "https://token4research.cn/v1", 3600, 32768,
                        logical_replies=3, transport_attempts=2, wire_mode=RESPONSES_STREAM_IPV4, sdk_version="2.28.0")
    binding = {
        "seed": 157, "round": 7,
        "a7_formal": {"contract_sha256": "1" * 64, "authorization_sha256": "2" * 64, "mock": False},
    }
    request = make_request(binding, 0, [{"role": "user", "content": "test packet"}], t)
    monkeypatch.setattr(w66, "PAYLOAD_SHA256", request["payload_sha256"])
    now = [w66.EXPIRES_UNIX - 120]
    ordinary = KeyPool(tmp_path / "pool", PoolConfig(1, {"key_01": key}), clock=lambda: now[0])
    ordinary.prepare()
    with ordinary.tx() as db:
        db.execute("UPDATE settings SET shared_disabled=1,shared_error=?", (w66.LATCH,))

    def permit():
        return w66.W66SinglePermitPool(
            ordinary.directory, ordinary.config, clock=lambda: now[0],
            contract_sha256=binding["a7_formal"]["contract_sha256"], payload_sha256=request["payload_sha256"],
            expires_unix=w66.EXPIRES_UNIX,
        )

    reserved = permit()
    reserved.bind_request(request)
    return ordinary, reserved, permit, request, t, now


def test_only_one_possible_post_survives_no_post_preflight_and_restart(scope):
    ordinary, permit, reopen, request, transport, _ = scope
    with pytest.raises(TransportStop, match="shared scope disabled"), ordinary.attempt(request, transport):
        pytest.fail("ordinary dispatcher ignored the shared latch")
    with permit.attempt(request, transport):
        lease = LEASE.get()
        lease.outcome("connect_error")
        assert ordinary.status(readonly=True)["limits"]["shared_disabled"] == 1
    with permit.attempt(request, transport):
        lease = LEASE.get()
        lease.dispatched()
        with pytest.raises(TransportFailure, match="operator_w66_post_not_permitted"):
            lease.dispatched()
        lease.outcome("completed", response_complete=True)
    reopened = reopen()
    reopened.bind_request(request)
    with pytest.raises(TransportStop, match="already consumed"), reopened.attempt(request, transport):
        pytest.fail("operator permit reopened after use")
    with ordinary.tx() as db:
        marker = db.execute("SELECT * FROM w66_operator_post WHERE one=1").fetchone()
        assert marker["ticket"] == lease.ticket
        assert marker["payload_sha256"] == request["payload_sha256"]
        assert db.execute("SELECT sum(dispatched) FROM queue").fetchone()[0] == 1
        assert db.execute("SELECT shared_disabled,shared_error FROM settings").fetchone()[:] == (1, w66.LATCH)


@pytest.mark.parametrize("identity", ("wrong_seed", "wrong_round", "wrong_contract", "wrong_payload", "wrong_effort"))
def test_request_binding_fail_closed(scope, identity):
    _, permit, _, request, transport, _ = scope
    binding = dict(request["binding"])
    payload = list(request["payload"]["input"])
    if identity == "wrong_seed":
        binding["seed"] = 6101
    elif identity == "wrong_round":
        binding["round"] = 8
    elif identity == "wrong_contract":
        binding["a7_formal"] = {**binding["a7_formal"], "contract_sha256": "f" * 64}
    elif identity == "wrong_payload":
        payload = [{"role": "user", "content": "another packet"}]
    else:
        transport = replace(transport, reasoning_effort="medium")
    with pytest.raises(PermissionError):
        permit.bind_request(make_request(binding, 0, payload, transport))


def test_usable_only_with_bound_id_and_matching_latch(scope):
    ordinary, permit, reopen, request, transport, _ = scope
    with pytest.raises(PermissionError, match="not authenticated"), reopen().attempt(request, transport):
        pytest.fail("an unbound permit acquired a credential")
    with ordinary.tx() as db:
        db.execute("UPDATE settings SET shared_error='quota_exhausted'")
    with pytest.raises(TransportStop, match="exact shared latch"), permit.attempt(request, transport):
        pytest.fail("a different shared stop was bypassed")


def test_expired_after_admission_remains_strictly_no_post(scope):
    ordinary, permit, _, request, transport, now = scope
    with permit.attempt(request, transport):
        now[0] = w66.EXPIRES_UNIX
        with pytest.raises(TransportFailure, match="operator_w66_post_not_permitted"):
            LEASE.get().dispatched()
    with ordinary.tx() as db:
        assert db.execute("SELECT sum(dispatched) FROM queue").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM w66_operator_post").fetchone()[0] == 0
        assert db.execute("SELECT shared_disabled FROM settings").fetchone()[0] == 1


@pytest.mark.parametrize("table", ("settings", "keys"))
def test_preserves_shared_and_credential_cooldowns(scope, table):
    ordinary, permit, _, request, transport, now = scope
    sql = "UPDATE settings SET shared_until=?" if table == "settings" else "UPDATE keys SET cooldown_until=?"
    with ordinary.tx() as db:
        db.execute(sql, (now[0] + 100,))
    with pytest.raises(TransportStop, match="cooldown"), permit.attempt(request, transport):
        pytest.fail("an existing credential or shared cooldown was bypassed")


def test_other_active_ticket_prevents_admission(scope):
    ordinary, permit, _, request, transport, now = scope
    with (ordinary.tickets / "other").open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with ordinary.tx() as db:
            db.execute("INSERT INTO queue(ticket,seed,round,logical_index,request_id,state,queued) "
                       "VALUES('other',6101,13,0,'other','queued',?)", (now[0],))
        with pytest.raises(TransportStop, match="otherwise idle shared pool"), permit.attempt(request, transport):
            pytest.fail("a scope was admitted ahead of other live work")
