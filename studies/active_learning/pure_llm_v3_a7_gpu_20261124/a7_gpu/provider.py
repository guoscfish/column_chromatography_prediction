"""Explicit credential-scoped Responses clients on the frozen A7 provider contract."""
import asyncio
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from ipaddress import ip_address
import re
import time

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_STREAM, RESPONSES_STREAM_IPV4, ProviderReply, TransportFailure, check_request, collect_completed, retry_http,
)

from .credentials import api_key, key_text
from .key_pool import LEASE

TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
LIFECYCLE_EVENTS = {
    "response.created", "response.queued", "response.in_progress", "response.completed",
    "response.incomplete", "response.failed", "error",
}
PROGRESS_MILESTONES = LIFECYCLE_EVENTS | {
    "response.output_item.added", "response.output_item.done", "response.output_text.delta", "response.output_text.done",
}
SSE_START_EVENTS = {"response.created", "response.queued", "response.in_progress"}
SSE_DELTA_EVENTS = {
    "response.output_text.delta", "response.reasoning_text.delta",
    "response.reasoning_summary_text.delta", "response.refusal.delta",
}
SSE_OUTPUT_EVENTS = {
    "response.output_item.added", "response.output_item.done", "response.output_text.done",
    "response.content_part.added", "response.content_part.done", "response.reasoning_text.done",
    "response.reasoning_summary_part.added", "response.reasoning_summary_part.done",
    "response.reasoning_summary_text.done", "response.refusal.done",
}
PROGRESS_INTERVAL_SECONDS = 5
DIRECT_IPV4_PREFLIGHT_SECONDS = 8
NETWORK_PHASES = {"connect_tcp": "dns_tcp", "start_tls": "tls", "send_request_headers": "send_headers",
                  "send_request_body": "send_body", "receive_response_headers": "receive_headers",
                  "direct_ipv4_preflight": "direct_ipv4_preflight",
                  "direct_ipv4_http11": "direct_ipv4_http11"}


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


def _progress_token(value):
    if isinstance(value, str) and TOKEN.fullmatch(value):
        return value
    return None


def _headers_progress(response, report):
    details = {"http_status": response.status_code}
    provider_request_id = _progress_token(response.headers.get("x-request-id"))
    if provider_request_id is not None:
        details["provider_request_id"] = provider_request_id
    report("http_headers", **details)


class _EventDeadline(TransportFailure):
    def __init__(self):
        super().__init__("timeout", ambiguous=True)


class _WallDeadline(TransportFailure):
    def __init__(self):
        super().__init__("timeout", ambiguous=True)


class _SSEInactivityDeadline(TransportFailure):
    def __init__(self):
        super().__init__("timeout", ambiguous=True)


class _PrePostConnectTimeout(TransportFailure):
    def __init__(self):
        super().__init__("connect_error", retryable=True)


class _IPv4PreflightFailure(TransportFailure):
    def __init__(self, retry=None):
        super().__init__(retry.code if retry else "connect_error", retryable=True,
                         retry_after=retry.retry_after if retry else None)


class _StreamEvidence:
    def __init__(self, response, deadline, report, *, activity=None):
        self.headers = response.headers
        self.deadline = deadline
        self.report = report
        self.activity = activity
        self.sse_activity_at = None
        self.response_id = None
        self.reported_events = set()
        self.event_count = 0
        self.last_event = None
        self.last_report = None
        if report is not None:
            _headers_progress(response, report)

    def _report_event(self, received, event, response, interval):
        kind = event.type
        error = event if kind == "error" else getattr(response, "error", None)
        details = {field: value for field, raw in (
            ("event_type", kind), ("response_id", getattr(response, "id", None)),
            ("model", getattr(response, "model", None)),
            ("provider_error_code", getattr(error, "code", None)),
            ("provider_error_param", getattr(error, "param", None)),
            ("provider_incomplete_reason", getattr(getattr(response, "incomplete_details", None), "reason", None)),
        ) if (value := _progress_token(raw)) is not None}
        if interval is not None:
            details["sse_interval_seconds"] = interval
        self.report("sse_event", observed_at=received, sse_activity_at=self.sse_activity_at,
                    sse_event_count=self.event_count, **details)

    def observe(self, event):
        received = time.monotonic()
        kind = event.type
        response = getattr(event, "response", None) if kind in LIFECYCLE_EVENTS else None
        response_id = getattr(response, "id", None)
        if isinstance(response_id, str) and response_id:
            if self.response_id is not None and response_id != self.response_id:
                if self.report is not None:
                    identities = {field: token for field, value in (
                        ("response_id", self.response_id), ("conflicting_response_id", response_id),
                    ) if (token := _progress_token(value)) is not None}
                    self.report("sse_event", observed_at=received, error_type="response_id_mismatch", **identities)
                raise TransportFailure("response_id_mismatch", ambiguous=True)
            self.response_id = response_id
        if self.activity is not None and self.activity.observe(event):
            self.sse_activity_at = received
        if self.report is not None:
            self.event_count += 1
            interval = None if self.last_event is None else received - self.last_event[0]
            self.last_event = received, event, response, interval
            milestone = kind in PROGRESS_MILESTONES and kind not in self.reported_events
            if milestone or self.last_report is None or received - self.last_report >= PROGRESS_INTERVAL_SECONDS:
                self._report_event(*self.last_event)
                self.last_report = received
                if milestone:
                    self.reported_events.add(kind)
        if self.deadline is not None and received > self.deadline:
            raise _EventDeadline()
        if kind in {"response.incomplete", "response.failed", "error"}:
            raise TransportFailure(kind, ambiguous=True)
        if kind == "response.completed":
            if not isinstance(response_id, str) or not response_id or getattr(response, "status", None) != "completed":
                raise TransportFailure("incomplete_response", ambiguous=True)
            return True
        return False

    def flush(self):
        if self.last_event is not None and self.last_event[0] != self.last_report:
            self._report_event(*self.last_event)

    def complete(self, event, capture_response, capture_receipt):
        terminal = event.response
        extras = getattr(terminal, "model_extra", {}) or {}
        reasoning = getattr(terminal, "reasoning", None)
        model_version = extras.get("model_version")
        declared = {"model_version": self.headers.get("x-model-version") if model_version is None else model_version,
                    "reported_reasoning_effort": getattr(reasoning, "effort", None)}
        reply = collect_completed((event,))
        reply = replace(reply, provider_request_id=reply.provider_request_id or self.headers.get("x-request-id"))
        if capture_response is not None:
            raw = terminal.model_dump(mode="json", exclude_unset=True)
            if capture_receipt is not None:
                capture_receipt({"raw_response_sha256": digest(raw),
                                 "provider_request_id": reply.provider_request_id, **declared})
            capture_response(raw)
        if self.report is not None:
            self.report("transport_completed", usage=reply.usage)
        return reply, declared


def _collect(stream, deadline, capture_response=None, capture_receipt=None, *, report=None):
    evidence = _StreamEvidence(stream.response, deadline, report)
    try:
        for event in stream:
            if evidence.observe(event):
                # A definitive completion does not need [DONE] or a clean EOF.
                return evidence.complete(event, capture_response, capture_receipt)
    finally:
        evidence.flush()
    raise TransportFailure("sse_missing_completion", ambiguous=True)


class _SSEActivity:
    """Recognize progress in decoded model frames independently of socket traffic."""

    def __init__(self):
        self.sequence = -1
        self.markers = set()

    def observe(self, event):
        kind = event.type
        sequence = getattr(event, "sequence_number", None)
        has_sequence = type(sequence) is int and sequence >= 0
        if has_sequence and sequence <= self.sequence:
            return False
        marker = None
        if kind in SSE_START_EVENTS and getattr(getattr(event, "response", None), "id", None):
            marker = (kind,)
        elif kind in SSE_OUTPUT_EVENTS:
            item_id = getattr(event, "item_id", None) or getattr(getattr(event, "item", None), "id", None)
            indices = (getattr(event, field, None) for field in ("output_index", "content_index", "summary_index"))
            marker = (kind, _progress_token(item_id), *(index if type(index) is int else None for index in indices))
            if all(value is None for value in marker[1:]):
                return False
        elif kind not in SSE_DELTA_EVENTS or not isinstance(getattr(event, "delta", None), str) or not event.delta:
            return False
        if marker is not None:
            if marker in self.markers:
                return False
            self.markers.add(marker)
        if has_sequence:
            self.sequence = sequence
        return True


async def _collect_async(stream, deadline, capture_response, capture_receipt, report, *,
                         sse_inactivity_seconds=None):
    activity = _SSEActivity() if sse_inactivity_seconds is not None else None
    evidence = _StreamEvidence(stream.response, deadline, report, activity=activity)
    clock = asyncio.get_running_loop().time
    inactivity = asyncio.timeout(sse_inactivity_seconds)
    try:
        async with inactivity:
            async for event in stream:
                received = clock()
                if activity is not None and received >= inactivity.when():
                    raise _SSEInactivityDeadline()
                previous_activity = evidence.sse_activity_at
                if evidence.observe(event):
                    return evidence.complete(event, capture_response, capture_receipt)
                if evidence.sse_activity_at != previous_activity:
                    inactivity.reschedule(received + sse_inactivity_seconds)
    except TimeoutError as error:
        if not inactivity.expired():
            raise
        raise _SSEInactivityDeadline() from error
    finally:
        evidence.flush()
    raise TransportFailure("sse_missing_completion", ambiguous=True)


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


def call_responses(request, config, *, model_lock, capture_response=None, capture_receipt=None, progress=None,
                   timeouts=None, sse_inactivity_seconds=None):
    """Only the authorized runner can call this with an acquired physical lease."""
    import httpx
    import openai

    lease = LEASE.get()
    if lease is None:
        raise TransportFailure("missing_physical_api_lease")
    check_request(request, request["binding"], config)
    if config.wire_mode not in (RESPONSES_STREAM, RESPONSES_STREAM_IPV4):
        raise TransportFailure("background_dispatcher_required")
    if config.wire_mode == RESPONSES_STREAM_IPV4 and timeouts is None:
        raise TransportFailure("direct_ipv4_timeouts_required")
    if sse_inactivity_seconds is not None and (
        timeouts is None or type(sse_inactivity_seconds) not in (float, int)
        or not 0 < sse_inactivity_seconds <= timeouts.total_seconds
    ):
        raise ValueError("SSE inactivity requires a positive finite timeout within the asynchronous total deadline")
    if not sdk_status(config)["matches"]:
        lease.outcome("sdk_version_mismatch")
        raise TransportFailure("sdk_version_mismatch")
    reply = None
    scope = "shared"
    started = time.monotonic()
    last_sse_activity_at = None

    def report(phase, *, observed_at=None, sse_activity_at=None, **fields):
        nonlocal last_sse_activity_at
        if progress is not None:
            try:
                elapsed = (time.monotonic() if observed_at is None else observed_at) - started
                if sse_activity_at is not None:
                    last_sse_activity_at = sse_activity_at
                if last_sse_activity_at is not None and phase in ("sse_event", "transport_failure"):
                    fields["sse_activity_seconds"] = last_sse_activity_at - started
                progress({"phase": phase, "elapsed_seconds": elapsed, **fields})
            except OSError:
                # Status is observational. An unavailable status file must not drop the response.
                pass

    error_type = None
    try:
        # A client gets one immutable key, no mutation of the process-wide environment.
        if lease.path is None:
            with api_key() as (value, _):
                return _call_with_key(value, request, config, model_lock, lease, capture_response, capture_receipt,
                                      report if progress is not None else None, timeouts, sse_inactivity_seconds)
        with key_text(lease.path) as value:
            return _call_with_key(value, request, config, model_lock, lease, capture_response, capture_receipt,
                                  report if progress is not None else None, timeouts, sse_inactivity_seconds)
    except (PermissionError, FileNotFoundError):
        failure = TransportFailure("credential_file_error")
    except openai.APIStatusError as error:
        failure, scope = http_failure(error.status_code, error.response.headers, error.body)
    except _EventDeadline as error:
        failure, error_type = error, "event_deadline_exceeded"
    except _WallDeadline as error:
        failure, error_type = error, "hard_wall_deadline"
    except _SSEInactivityDeadline as error:
        failure, error_type = error, "sse_inactivity_timeout"
    except _IPv4PreflightFailure as error:
        failure, error_type = error, "direct_ipv4_preflight_failed"
    except _PrePostConnectTimeout as error:
        failure, error_type = error, "connect_timeout_before_post"
    except (openai.APITimeoutError, httpx.TimeoutException) as error:
        failure = TransportFailure("timeout", ambiguous=True)
        error_type = type(error.__cause__ or error).__name__
    except openai.APIConnectionError as error:
        failure = (TransportFailure("connect_error", retryable=True) if isinstance(error.__cause__, httpx.ConnectError)
                   else TransportFailure("connection_interrupted", ambiguous=True))
        error_type = type(error.__cause__ or error).__name__
    except (ValueError, httpx.TransportError) as error:
        failure = TransportFailure("sse_frame_error", ambiguous=True)
        error_type = type(error).__name__
    except openai.OpenAIError as error:
        failure = TransportFailure("sdk_error", ambiguous=True)
        error_type = type(error).__name__
    except _ReturnedResponseError as error:
        reply, failure = error.reply, error.failure
    except TransportFailure as error:
        failure = error
    report("transport_failure", error_type=error_type or failure.code)
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


class _TransmissionEvidence:
    def __init__(self):
        self.post_may_have_started = False
        self.failed_connection = None

    def proves_pre_post_timeout(self, cause):
        return (not self.post_may_have_started and self.failed_connection is not None
                and (cause is self.failed_connection or cause.__cause__ is self.failed_connection))


def _trace_request(report, transmission):
    async def attach(request):
        if request.method != "POST":
            return
        started = {}
        observed_proxy = None
        direct_tls = False

        async def trace(name, info):
            nonlocal observed_proxy, direct_tls
            source, stage, state = name.rsplit(".", 2)
            phase = NETWORK_PHASES.get(stage)
            if phase is None:
                return
            if source in ("proxy", "socks"):
                observed_proxy = True
            observed = time.monotonic()
            step = source, stage
            exchange = stage in ("send_request_headers", "send_request_body", "receive_response_headers")
            if exchange and state in ("complete", "failed") and step not in started:
                transmission.post_may_have_started = True
            if state == "started":
                started[step] = observed
                if stage in ("connect_tcp", "start_tls"):
                    transmission.failed_connection = None
                if exchange:
                    method = getattr(info.get("request"), "method", None)
                    # The proxy CONNECT cannot transmit the provider POST. Treat missing method
                    # evidence conservatively once a send or response callback begins.
                    if method != b"CONNECT":
                        transmission.post_may_have_started = True
                    if stage == "send_request_headers":
                        if method == b"CONNECT":
                            observed_proxy = True
                        elif method == b"POST" and direct_tls and observed_proxy is None:
                            observed_proxy = False
            elif state in ("complete", "failed") and step in started:
                if (state == "failed" and stage in ("connect_tcp", "start_tls")
                        and source in ("connection", "proxy", "socks")):
                    transmission.failed_connection = info.get("exception")
                details = {"network_phase": phase, "network_state": state,
                           "phase_duration_seconds": observed - started.pop(step)}
                if stage == "start_tls" and source == "connection" and state == "complete":
                    direct_tls = True
                if observed_proxy is not None:
                    details["explicit_proxy"] = observed_proxy
                if report is not None:
                    report("http_phase", observed_at=observed, **details)

        request.extensions["trace"] = trace
    return attach


def _direct_ipv4_http11(response):
    network = response.extensions.get("network_stream")
    peer = network.get_extra_info("server_addr") if network is not None else None
    return bool(peer and ip_address(peer[0]).version == 4 and response.http_version == "HTTP/1.1")


async def _preflight_ipv4(http, base_url, report):
    """Check the exact gateway over a credential-free request before reserving a model POST."""
    import httpx

    started = time.monotonic()
    status = None
    passed = False
    try:
        async with asyncio.timeout(DIRECT_IPV4_PREFLIGHT_SECONDS):
            response = await http.get(f"{base_url.rstrip('/')}/models")
            status = response.status_code
            if status not in (200, 401) or not _direct_ipv4_http11(response):
                retry = retry_http(status, response.headers) if status == 429 else None
                raise _IPv4PreflightFailure(retry)
            passed = True
    except (TimeoutError, httpx.HTTPError, ValueError) as error:
        raise _IPv4PreflightFailure() from error
    finally:
        # The route probe must not introduce a cookie to the scientific request.
        http.cookies.clear()
        if report is not None:
            report("http_phase", network_phase="direct_ipv4_preflight",
                   network_state="complete" if passed else "failed",
                   phase_duration_seconds=time.monotonic() - started, explicit_proxy=False,
                   event_type=f"ipv4_preflight_http_{status}" if status is not None else "ipv4_preflight_no_headers")


async def _hard_wall_call(key, request, config, lease, capture_response, capture_receipt, report, timeouts,
                          sse_inactivity_seconds):
    import httpx
    import openai

    sdk_timeout = httpx.Timeout(connect=timeouts.connect_seconds, pool=timeouts.connect_seconds,
                                write=timeouts.write_seconds, read=timeouts.read_idle_seconds)
    transmission = _TransmissionEvidence()
    event_hooks = {"request": [_trace_request(report, transmission)]}
    ipv4_direct = config.wire_mode == RESPONSES_STREAM_IPV4
    client_options = {}
    if ipv4_direct:
        client_options = {
            "trust_env": False,
            # Binding an IPv4 source constrains the resolver to A records. The HTTPS
            # URL still supplies the original hostname for SNI and certificate checks.
            "transport": httpx.AsyncHTTPTransport(local_address="0.0.0.0", verify=True,
                                                 http1=True, http2=False, trust_env=False, retries=0),
        }
    reply = None
    started = time.monotonic()
    deadline = started + timeouts.total_seconds
    alarm = asyncio.timeout(timeouts.total_seconds)
    submission_reserved = False
    try:
        async with alarm:
            async with httpx.AsyncClient(timeout=sdk_timeout, follow_redirects=False, event_hooks=event_hooks,
                                         **client_options) as http:
                if ipv4_direct:
                    await _preflight_ipv4(http, config.base_url, report)
                async with openai.AsyncOpenAI(api_key=key, base_url=config.base_url, timeout=sdk_timeout,
                                              http_client=http, max_retries=0) as client:
                    lease.dispatched()
                    submission_reserved = True
                    if report is not None:
                        report("request_dispatched")
                    stream = await client.responses.create(
                        **request["payload"], extra_headers={"X-Client-Request-Id": request["request_id"]})
                    transmission.post_may_have_started = True
                    async with stream:
                        if ipv4_direct:
                            response = stream.response
                            if not _direct_ipv4_http11(response):
                                raise TransportFailure("direct_ipv4_route_mismatch", ambiguous=True)
                            if report is not None:
                                report("http_phase", network_phase="direct_ipv4_http11", network_state="complete",
                                       phase_duration_seconds=time.monotonic() - started, explicit_proxy=False)
                        reply = await _collect_async(stream, deadline, capture_response, capture_receipt, report,
                                                     sse_inactivity_seconds=sse_inactivity_seconds)
    except TimeoutError as error:
        if not alarm.expired():
            raise
        if ipv4_direct and not submission_reserved:
            raise _IPv4PreflightFailure() from error
        if reply is None:
            raise _WallDeadline() from error
        if report is not None:
            report("sse_event", event_type="close_after_completed", error_type="hard_wall_deadline")
    except (httpx.TransportError, openai.APIConnectionError) as error:
        if reply is None:
            cause = error.__cause__ if isinstance(error, openai.APITimeoutError) else error
            if isinstance(cause, httpx.ConnectTimeout) and transmission.proves_pre_post_timeout(cause):
                raise _PrePostConnectTimeout() from error
            raise
        if report is not None:
            report("sse_event", event_type="close_after_completed", error_type=type(error).__name__)
    return reply


def _run_hard_wall(*args):
    # asyncio.run drains its default executor at exit. A cancelled OS DNS lookup may still be
    # running in that executor, so close this private loop without waiting for the resolver thread.
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_hard_wall_call(*args))
    finally:
        try:
            # A terminal event ends the iterator early. Finalize suspended generators within a
            # bounded local cleanup allowance without draining the default resolver executor.
            loop.run_until_complete(asyncio.wait_for(loop.shutdown_asyncgens(), timeout=.1))
        except TimeoutError:
            pass
        finally:
            loop.close()


def _call_with_key(key, request, config, model_lock, lease, capture_response, capture_receipt, report, timeouts,
                   sse_inactivity_seconds):
    import httpx
    import openai

    deadline = time.monotonic() + config.timeout_seconds

    def private_report(phase, **fields):
        report(phase, **{name: value for name, value in fields.items() if not isinstance(value, str) or key not in value})

    reply = None
    try:
        if timeouts is not None:
            reply, declared = _run_hard_wall(key, request, config, lease, capture_response, capture_receipt,
                                             private_report if report is not None else None, timeouts,
                                             sse_inactivity_seconds)
        else:
            with openai.OpenAI(api_key=key, base_url=config.base_url, timeout=config.timeout_seconds,
                               max_retries=0) as client:
                lease.dispatched()
                if report is not None:
                    private_report("request_dispatched")
                with client.responses.create(
                    **request["payload"], extra_headers={"X-Client-Request-Id": request["request_id"]}
                ) as stream:
                    reply, declared = _collect(stream, deadline, capture_response, capture_receipt,
                                               report=private_report if report is not None else None)
    except openai.APIStatusError as error:
        if report is not None:
            _headers_progress(error.response, private_report)
        raise
    except (httpx.TransportError, openai.APIConnectionError) as error:
        if reply is None:
            raise
        # Only stream/client cleanup ran after the response was verified and captured.
        if report is not None:
            private_report("sse_event", event_type="close_after_completed", error_type=type(error).__name__)
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
