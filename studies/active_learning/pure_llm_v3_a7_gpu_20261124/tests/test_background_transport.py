from dataclasses import replace
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, read
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.background_transport import (
    BackgroundBinding, BackgroundLedger,
)

MODEL = "registered-model"
EFFORT = "xhigh"
KEY = "local-caller-only-synthetic-secret"
ALIAS = "key_02"
RESPONSE_ID = "resp_same_job"
PAYLOAD = {"model": MODEL, "reasoning": {"effort": EFFORT}, "input": "synthetic exercise",
           "background": True, "stream": False, "store": True}
BINDING = BackgroundBinding("a" * 64, "b" * 64, digest(PAYLOAD), ALIAS, MODEL, EFFORT)


class HTTPError(Exception):
    def __init__(self, status):
        self.status_code = status
        self.request_id = "request_error"


def raw_response(status="queued", **changes):
    raw = {"id": RESPONSE_ID, "model": MODEL, "status": status, "background": True,
           "reasoning": {"effort": EFFORT}}
    if status == "completed":
        raw.update(output=[{"type": "message", "content": [{"type": "output_text", "text": "answer"}]}],
                   usage={"input_tokens": 20, "output_tokens": 9, "total_tokens": 29})
    return {**raw, **changes}


class FakeClient:
    def __init__(self, post=None, gets=(), *, retries=0, redirect=False):
        self.api_key = KEY
        self.max_retries = retries
        self._client = SimpleNamespace(follow_redirects=redirect)
        self.post = post
        self.gets = iter(gets)
        self.calls = []
        api = SimpleNamespace(create=self.create, retrieve=self.retrieve)
        self.responses = SimpleNamespace(with_raw_response=api)

    @staticmethod
    def packet(raw):
        if isinstance(raw, BaseException):
            raise raw
        response = SimpleNamespace(model_dump=lambda **kwargs: raw)
        return SimpleNamespace(parse=lambda: response, headers={"x-request-id": "request_test"})

    def create(self, **kwargs):
        self.calls.append(("POST", kwargs))
        return self.packet(self.post)

    def retrieve(self, response_id):
        self.calls.append(("GET", response_id))
        return self.packet(next(self.gets))


def open_ledger(tmp_path, *, binding=BINDING, payload=PAYLOAD):
    return BackgroundLedger(tmp_path / "background", binding=binding, payload=payload)


def test_restart_uses_only_get_and_exact_original_identity(tmp_path):
    ledger = open_ledger(tmp_path)
    submitter = FakeClient(post=raw_response())
    asserted = ledger.submit(submitter, credential_alias=ALIAS, allow_post=True)
    assert asserted["response_id"] == RESPONSE_ID and asserted["state"] == "queued"
    post = {"X-Client-Request-Id": BINDING.request_sha256}
    assert submitter.calls == [("POST", {**PAYLOAD, "extra_headers": post})]

    fresh = open_ledger(tmp_path)
    retriever = FakeClient(gets=(HTTPError(503), raw_response("in_progress"), raw_response("completed")))
    with pytest.raises(PermissionError, match="original credential"):
        fresh.poll(retriever, credential_alias="key_01")
    assert fresh.poll(retriever, credential_alias=ALIAS)["last_http_status"] == 503
    assert fresh.submit(retriever, credential_alias=ALIAS, allow_post=True)["post_reserved"]
    assert fresh.poll(retriever, credential_alias=ALIAS)["state"] == "in_progress"
    assert fresh.poll(retriever, credential_alias=ALIAS)["state"] == "completed"
    assert fresh.poll(retriever, credential_alias=ALIAS)["state"] == "completed"
    assert fresh.completed() == raw_response("completed")
    assert retriever.calls == [("GET", RESPONSE_ID)] * 3


def test_unverified_synchronous_response_cannot_be_retrieved_or_used(tmp_path):
    ledger = open_ledger(tmp_path)
    client = FakeClient(post=raw_response("completed", background=False))
    status = ledger.submit(client, credential_alias=ALIAS, allow_post=True)
    assert status["state"] == "submission_unverified"
    assert status["last_error"] == "provider_did_not_affirm_background"
    assert status["observed_background"] is False
    assert ledger.poll(client, credential_alias=ALIAS) == status
    assert ledger.completed() is None
    assert len(client.calls) == 1


def test_unknown_response_keeps_the_original_post_reserved(tmp_path):
    client = FakeClient(post=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        open_ledger(tmp_path).submit(client, credential_alias=ALIAS, allow_post=True)
    restored = open_ledger(tmp_path)
    unknown = restored.submit(client, credential_alias=ALIAS, allow_post=True)
    assert restored.poll(client, credential_alias=ALIAS) == unknown
    assert unknown["state"] == "submission_unknown"
    assert len(client.calls) == 1


def test_reject_wrong_signed_wire_and_hidden_post_retries(tmp_path):
    mutated = {**PAYLOAD, "input": "different exercise"}
    with pytest.raises(PermissionError, match="payload differs"):
        open_ledger(tmp_path, payload=mutated)
    ledger = open_ledger(tmp_path)
    for client in (FakeClient(retries=2), FakeClient(redirect=True)):
        with pytest.raises(PermissionError, match="implicit retries and HTTP redirects"):
            ledger.submit(client, credential_alias=ALIAS, allow_post=True)
        assert client.calls == []
    assert not ledger.status()["post_reserved"]
    with pytest.raises(ValueError, match="immutable receipt differs"):
        open_ledger(tmp_path, binding=replace(BINDING, request_sha256="c" * 64))


def test_wrong_id_model_effort_or_credentials_never_becomes_usable(tmp_path):
    ledger = open_ledger(tmp_path)
    complete_with_key = raw_response("completed", output=[{
        "type": "message", "content": [{"type": "output_text", "text": KEY}]
    }])
    get_replies = (raw_response("completed", id="resp_different"),
                   raw_response("completed", model="wrong-model"),
                   raw_response("completed", reasoning={"effort": "low"}),
                   complete_with_key)
    client = FakeClient(post=raw_response(), gets=get_replies)
    ledger.submit(client, credential_alias=ALIAS, allow_post=True)
    for expected in ("response_id_mismatch", "served_model_or_effort_mismatch",
                     "served_model_or_effort_mismatch", "unsafe_response_evidence"):
        assert ledger.poll(client, credential_alias=ALIAS)["last_error"] == expected
        assert ledger.completed() is None
    assert not any(KEY in file.read_text() for file in ledger.path.rglob("*.json"))


def test_fails_closed_on_tampered_completion(tmp_path):
    ledger = open_ledger(tmp_path)
    client = FakeClient(post=raw_response(), gets=(raw_response("completed"),))
    ledger.submit(client, credential_alias=ALIAS, allow_post=True)
    ledger.poll(client, credential_alias=ALIAS)
    path = ledger.path / "retrieval_terminal.json"
    corrupted = read(path)
    corrupted["raw_response"]["usage"]["input_tokens"] += 1
    path.write_text(dumps(corrupted))
    with pytest.raises(PermissionError, match="integrity verification"):
        ledger.completed()


def test_known_served_alias_is_exact_and_network_hook_counts_methods(tmp_path):
    routed_model = "registry.registered-model"
    payload = {**PAYLOAD, "model": routed_model}
    binding = replace(BINDING, model=routed_model, wire_sha256=digest(payload), served_model=MODEL)
    ledger = open_ledger(tmp_path, binding=binding, payload=payload)
    missing_text = raw_response("completed", output=[])
    client = FakeClient(post=raw_response(), gets=(missing_text, raw_response("completed")))
    methods = []
    args = {"credential_alias": ALIAS, "on_dispatch": methods.append}

    assert ledger.submit(client, allow_post=True, **args)["model"] == MODEL
    ledger.submit(client, allow_post=True, **args)
    assert ledger.poll(client, **args)["last_error"] == "missing_completed_output"
    assert ledger.completed() is None
    assert ledger.poll(client, **args)["state"] == "completed"
    assert ledger.poll(client, **args)["state"] == "completed"
    assert methods == ["POST", "GET", "GET"]
