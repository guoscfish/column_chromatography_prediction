"""Only the exact audited upstream code may enter existing network retries."""
import importlib.util
from pathlib import Path

import httpx
import pytest


@pytest.fixture
def m(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/studies/resume_nocw_upstream_20261114.py"
    spec = importlib.util.spec_from_file_location("upstream_recovery_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    monkeypatch.setattr(module, "PLAN", plan)
    monkeypatch.setattr(module, "APPROVAL", tmp_path / "unapproved.json")
    return module


class ProtocolError(RuntimeError):
    pass


def test_only_upstream_error_is_retried_and_receipted(m, tmp_path):
    def fail(*args):
        raise ProtocolError("provider terminal failure code: upstream_error")
    with pytest.raises(httpx.RemoteProtocolError, match="upstream_error"):
        m.retry_upstream(fail, ProtocolError, tmp_path, 0, 4, "messages", "config")
    receipt = m.m.read(tmp_path / "upstream_retry_policy_00_04.json")
    assert receipt["billing_status"] == "unknown"
    assert receipt["provider_code"] == "upstream_error"


@pytest.mark.parametrize("message", ["wrong model", "provider HTTP 401", "provider terminal failure code: invalid_api_key", "provider terminal event: response.incomplete"])
def test_fatal_identity_auth_and_incomplete_guards_are_retained(m, tmp_path, message):
    def fail(*args):
        raise ProtocolError(message)
    with pytest.raises(ProtocolError, match=message):
        m.retry_upstream(fail, ProtocolError, tmp_path, 0, 4)
    assert not (tmp_path / "upstream_retry_policy_00_04.json").exists()


def test_success_is_returned_unchanged_and_request_arguments_are_preserved(m, tmp_path):
    value = object()
    def complete(*args):
        assert args == ("messages", "config", tmp_path, 2, 5)
        return value
    assert m.retry_upstream(complete, ProtocolError, tmp_path, 2, 5, "messages", "config") is value


def test_missing_approval_cannot_start_a_worker(m, monkeypatch):
    monkeypatch.setattr(m.m, "read", lambda p: {"recovery_script_sha256": "hash", "test_sha256": "hash"})
    monkeypatch.setattr(m.m, "sha", lambda p: "hash")
    with pytest.raises(PermissionError, match="pending"):
        m.require_approval()
