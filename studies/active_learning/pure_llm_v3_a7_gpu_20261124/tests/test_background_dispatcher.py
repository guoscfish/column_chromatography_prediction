"""Mock background bridge delivery through a restarted Mac dispatcher."""

from contextlib import contextmanager
from dataclasses import asdict, replace
from types import SimpleNamespace

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import RESPONSES_BACKGROUND
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu import ssh_courier
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.background_dispatcher import BackgroundDispatcher
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.formal_contract import RequestTimeouts
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.key_pool import KeyPool, PoolConfig
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.tests.test_background_transport import FakeClient, RESPONSE_ID, raw_response
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.tests.test_distributed_a7 import make_sample_job
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.tests.test_seed_runner import formal  # noqa: F401


def test_background_restart_retrieves_and_delivers_original_attempt(tmp_path, formal):
    base, approval = formal
    transport = replace(base.transport, wire_mode=RESPONSES_BACKGROUND)
    contract = replace(base, transport=transport, profile={
        **base.profile, "transport": asdict(transport), "request_timeouts": asdict(RequestTimeouts()),
    })
    approval = {**approval, "contract_sha256": contract.digest}
    info = {"schema": ssh_courier.SCHEMA, "contract_sha256": contract.digest,
            "authorization_sha256": digest(approval), "contract": contract.record(), "mock": False, "paused": False}
    job, _, started = make_sample_job((contract, approval), mock=False)
    alias = "key_01"
    key = tmp_path / "key"
    key.write_text("fixture-only-secret")
    key.chmod(0o600)
    pool = KeyPool(tmp_path / "api", PoolConfig(1, {alias: key}))
    pool.prepare()
    raw = {"model": transport.served_model, "reasoning": {"effort": transport.reasoning_effort}}
    submitter = FakeClient(post=raw_response(**raw))
    retriever = FakeClient(gets=[raw_response("completed", **raw)])
    courier = SimpleNamespace(info=lambda: info, claim=lambda *_: job)

    def dispatcher(client_instance):
        @contextmanager
        def client(key_value):
            assert key_value == "fixture-only-secret"
            yield client_instance

        return BackgroundDispatcher(tmp_path / "spool", info=info, contract_sha256=contract.digest,
                                    transport=transport, mock=False, pool=pool, client_for_key=client)

    mac = dispatcher(submitter)
    identity = mac.accept(job)
    mac._submit(courier, identity, {alias})
    assert identity == started["attempt_id"]
    assert [verb for verb, _ in submitter.calls] == ["POST"]
    assert mac.pending() == [] and mac.undelivered() == []

    reopened = dispatcher(retriever)
    active, = reopened._active()
    reopened._poll_accepted(*active)
    assert retriever.calls == [("GET", RESPONSE_ID)]
    assert reopened._active() == [] and reopened.undelivered() == [identity]
    result = reopened.result(identity)
    assert result["raw_response"] == raw_response("completed", **raw)
    assert result["credential_id"] == alias and result["provider_dispatched"]
    assert result["outcome"]["reply"]["response_id"] == RESPONSE_ID
    assert read(active[0] / "provider_capture_receipt.json")["job_sha256"] == job["job_sha256"]
    with pool.tx() as db:
        count, = db.execute("SELECT SUM(dispatched) FROM queue").fetchone()
    assert count == 1
