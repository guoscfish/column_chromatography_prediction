"""Explicit credential-scoped Responses clients on the frozen A7 provider contract."""
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
import re
import time

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    ProviderReply, TransportFailure, check_request, collect_completed, retry_http,
)

from .credentials import api_key, key_text
from .key_pool import LEASE

TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def sdk_status(config):
    try:
        actual = version("openai")
    except PackageNotFoundError:
        actual = None
    return {"expected": config.sdk_version, "installed": actual, "matches": actual == config.sdk_version}


def http_failure(status, headers, body):
    """Only explicit credential scope permits another key to bypass a cooldown."""
    details = body.get("error", body) if isinstance(body, dict) else {}
    details = details if isinstance(details, dict) else {}
    value = headers.get("x-ratelimit-scope") or details.get("rate_limit_scope")
    scope = "credential" if value in ("credential", "api_key", "key") else "shared"
    code = details.get("code")
    if code in ("insufficient_quota", "billing_hard_limit_reached", "insufficient_balance"):
        return TransportFailure("quota_exhausted"), scope
    if status in (400, 404, 413, 422):
        if status == 413 or code in ("context_length_exceeded", "context_window_exceeded", "input_too_long"):
            return TransportFailure("provider_context_rejected"), "shared"
        if code in ("model_not_found", "unsupported_model") or details.get("param") == "model":
            return TransportFailure("provider_model_rejected"), "shared"
        if details.get("param") in ("reasoning", "reasoning.effort", "reasoning_effort"):
            return TransportFailure("provider_reasoning_rejected"), "shared"
    return retry_http(status, headers), scope


def _collect(stream, deadline, capture_response=None, capture_receipt=None):
    declared = {}
    terminal = None

    def watch():
        nonlocal terminal
        for event in stream:
            if event.type == "response.completed":
                terminal = event.response
                extras = getattr(terminal, "model_extra", {}) or {}
                reasoning = getattr(terminal, "reasoning", None)
                declared.update(model_version=extras.get("model_version"),
                                reported_reasoning_effort=getattr(reasoning, "effort", None))
            yield event

    reply = collect_completed(watch(), deadline=deadline)
    headers = stream.response.headers
    if declared.get("model_version") is None:
        declared["model_version"] = headers.get("x-model-version")
    reply = replace(reply, provider_request_id=reply.provider_request_id or headers.get("x-request-id"))
    if capture_response is not None:
        raw = terminal.model_dump(mode="json", exclude_unset=True)
        if capture_receipt is not None:
            capture_receipt({"raw_response_sha256": digest(raw),
                             "provider_request_id": reply.provider_request_id, **declared})
        capture_response(raw)
    return reply, declared


def verify_reported_model(config, reply, declared):
    if reply.served_model != config.served_model:
        raise TransportFailure("served_model_mismatch")
    if declared["reported_reasoning_effort"] not in (None, config.reasoning_effort):
        raise TransportFailure("served_reasoning_effort_mismatch")
    version_id = declared["model_version"]
    if version_id is not None and (not isinstance(version_id, str) or not TOKEN.fullmatch(version_id)):
        raise TransportFailure("invalid_reported_model_version")
    return {"provider": config.provider, "requested_model": config.requested_model,
            "served_model": reply.served_model, "reported_model_version": version_id,
            "requested_reasoning_effort": config.reasoning_effort,
            "reported_reasoning_effort": declared["reported_reasoning_effort"]}


def _lock_model(lock_path, config, reply, declared):
    lock = verify_reported_model(config, reply, declared)
    try:
        once(lock_path, lock)
    except ValueError:
        raise TransportFailure("provider_identity_drift") from None
    return lock


def recover_completed_response(raw, config, *, model_lock, metadata=None):
    """Recover the sole previously completed provider response using durable local evidence."""
    if not isinstance(raw, dict) or raw.get("status") != "completed":
        raise ValueError("local raw response does not prove completion")
    reasoning = raw.get("reasoning") or {}
    if not isinstance(reasoning, dict):
        raise ValueError("invalid provider reasoning evidence")
    declared = {"model_version": raw.get("model_version"),
                "reported_reasoning_effort": reasoning.get("effort")}
    provider_request_id = None
    if metadata is not None:
        if (
            type(metadata) is not dict or set(metadata) != {
                "raw_response_sha256", "provider_request_id", "model_version", "reported_reasoning_effort"
            }
            or metadata["raw_response_sha256"] != digest(raw)
            or any(declared[k] is not None and declared[k] != metadata[k] for k in declared)
            or metadata["provider_request_id"] is not None and not isinstance(metadata["provider_request_id"], str)
        ):
            raise ValueError("local recovery metadata does not match the raw provider response")
        provider_request_id = metadata["provider_request_id"]
        declared = {key: metadata[key] for key in declared}
    output = raw.get("output")
    if not isinstance(output, list):
        raise ValueError("invalid raw provider output")
    answer = "".join(block["text"] for item in output if item["type"] == "message"
                     for block in item["content"] if block["type"] == "output_text")
    measured = raw.get("usage") or {}
    if not isinstance(measured, dict):
        raise ValueError("invalid raw provider token usage")
    usage = {k: measured[k] for k in ("input_tokens", "output_tokens", "total_tokens")
             if measured.get(k) is not None}
    for block, field in (("input_tokens_details", "cached_tokens"), ("output_tokens_details", "reasoning_tokens")):
        count = (measured.get(block) or {}).get(field)
        if count is not None:
            usage[field] = count
    if (not isinstance(raw.get("id"), str) or not raw["id"] or not answer.strip()
            or any(type(v) is not int or v < 0 for v in usage.values())):
        raise ValueError("raw provider completion is missing identity, answer, or consistent usage")
    reply = ProviderReply(answer, raw["id"], raw["model"], usage, "completed", provider_request_id)
    return reply, _lock_model(model_lock, config, reply, declared)


def call_responses(request, config, *, model_lock, capture_response=None, capture_receipt=None):
    """Only the authorized runner can call this with an acquired physical lease."""
    import httpx
    import openai

    lease = LEASE.get()
    if lease is None:
        raise TransportFailure("missing_physical_api_lease")
    check_request(request, request["binding"], config)
    if not sdk_status(config)["matches"]:
        lease.outcome("sdk_version_mismatch")
        raise TransportFailure("sdk_version_mismatch")
    reply = None
    scope = "shared"
    try:
        # A client gets one immutable key, no mutation of the process-wide environment.
        if lease.path is None:
            with api_key() as (value, _):
                return _call_with_key(value, request, config, model_lock, lease, capture_response, capture_receipt)
        with key_text(lease.path) as value:
            return _call_with_key(value, request, config, model_lock, lease, capture_response, capture_receipt)
    except (PermissionError, FileNotFoundError):
        failure = TransportFailure("credential_file_error")
    except openai.APIStatusError as error:
        failure, scope = http_failure(error.status_code, error.response.headers, error.body)
    except (openai.APITimeoutError, httpx.TimeoutException):
        failure = TransportFailure("timeout", ambiguous=True)
    except openai.APIConnectionError as error:
        failure = (TransportFailure("connect_error", retryable=True) if isinstance(error.__cause__, httpx.ConnectError)
                   else TransportFailure("connection_interrupted", ambiguous=True))
    except (ValueError, httpx.TransportError):
        failure = TransportFailure("sse_frame_error", ambiguous=True)
    except openai.OpenAIError:
        failure = TransportFailure("sdk_error", ambiguous=True)
    except _ReturnedResponseError as error:
        reply, failure = error.reply, error.failure
    except TransportFailure as error:
        failure = error
    lease.outcome(failure.code, scope=scope, retry_after=failure.retry_after,
                  usage=reply.usage if reply else None, response_complete=reply is not None)
    if reply is not None:
        # The legacy receipt writes the returned identity before its exact-model validation.
        failure.reply_identity = {"response_id": reply.response_id, "served_model": reply.served_model,
                                  "usage": reply.usage, "provider_request_id": reply.provider_request_id,
                                  "provider_status": reply.status}
    raise failure from None


class _ReturnedResponseError(Exception):
    def __init__(self, reply, failure):
        self.reply = reply
        self.failure = failure


def _call_with_key(key, request, config, model_lock, lease, capture_response, capture_receipt):
    import openai

    deadline = time.monotonic() + config.timeout_seconds
    with openai.OpenAI(api_key=key, base_url=config.base_url, timeout=config.timeout_seconds,
                       max_retries=0) as client:
        lease.dispatched()
        with client.responses.create(
            **request["payload"], extra_headers={"X-Client-Request-Id": request["request_id"]}
        ) as stream:
            reply, declared = _collect(stream, deadline, capture_response, capture_receipt)
    try:
        if not reply.response_id or reply.status != "completed":
            raise TransportFailure("incomplete_response", ambiguous=True)
        _lock_model(model_lock, config, reply, declared)
        if type(reply.text) is not str or not reply.text.strip():
            raise TransportFailure("empty_response")
        if any(type(v) is not int or v < 0 for v in reply.usage.values()):
            raise TransportFailure("invalid_usage")
    except TransportFailure as error:
        raise _ReturnedResponseError(reply, error) from None
    lease.outcome("completed", usage=reply.usage, response_complete=True)
    return reply
