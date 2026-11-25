"""Bound provider requests, immutable receipts and bounded transport retries."""
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import fcntl
import math
import random
import time
from urllib.parse import urlparse
from uuid import uuid4

from .records import digest, once, read


class TransportStop(RuntimeError):
    pass


class TransportFailure(Exception):
    def __init__(self, code, *, retryable=False, retry_after=None, ambiguous=False):
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.retry_after = retry_after
        self.ambiguous = ambiguous


@dataclass(frozen=True)
class TransportConfig:
    provider: str
    requested_model: str
    served_model: str
    reasoning_effort: str
    base_url: str
    timeout_seconds: float
    max_output_tokens: int
    transport_attempts: int = 3
    logical_replies: int = 3
    base_delay_seconds: float = 2
    max_delay_seconds: float = 60
    rate_limit_threshold: int = 2
    wire_mode: str = "responses_stream"
    sdk_version: str = "offline"

    def __post_init__(self):
        endpoint = urlparse(self.base_url)
        if (endpoint.scheme != "https" or not endpoint.hostname or endpoint.username or endpoint.password
                or endpoint.query or endpoint.fragment or self.wire_mode != "responses_stream"):
            raise ValueError("unsupported Responses endpoint")
        budgets = (self.max_output_tokens, self.transport_attempts, self.logical_replies, self.rate_limit_threshold)
        if any(type(v) is not int or v < 1 for v in budgets) or not (
            self.transport_attempts <= 10 and self.logical_replies <= 10
        ):
            raise ValueError("invalid finite request budget")
        timings = (self.timeout_seconds, self.base_delay_seconds, self.max_delay_seconds)
        if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in timings):
            raise ValueError("invalid transport timings")
        if self.base_delay_seconds > self.max_delay_seconds:
            raise ValueError("backoff base exceeds ceiling")
        if any(type(v) is not str or not v for v in (
            self.provider, self.requested_model, self.served_model, self.reasoning_effort, self.sdk_version
        )):
            raise ValueError("incomplete provider identity")


@dataclass(frozen=True)
class ProviderReply:
    text: str
    response_id: str
    served_model: str
    usage: dict
    status: str = "completed"
    provider_request_id: str | None = None
    transport_evidence: dict | None = None


def wire_payload(messages, config):
    return {"model": config.requested_model, "reasoning": {"effort": config.reasoning_effort},
            "input": messages, "max_output_tokens": config.max_output_tokens, "stream": True, "store": False}


def make_request(binding, logical_index, messages, config):
    payload = wire_payload(messages, config)
    identity = {"binding": binding, "logical_index": logical_index, "configuration": asdict(config),
                "payload_sha256": digest(payload)}
    return {**identity, "request_id": digest(identity), "payload": payload}


def check_request(request, binding, config):
    if request != make_request(binding, request["logical_index"], request["payload"]["input"], config):
        raise ValueError("request identity, packet, prompt or transport contract changed")
    if not 0 <= request["logical_index"] < config.logical_replies:
        raise TransportStop("logical reply budget exhausted")


def _validate_reply(reply, config):
    if reply.served_model != config.served_model:
        raise TransportFailure("served_model_mismatch")
    if not reply.response_id:
        raise TransportFailure("missing_response_identity", ambiguous=True)
    if reply.status != "completed":
        raise TransportFailure("incomplete_response", ambiguous=True)
    if type(reply.text) is not str or not reply.text.strip():
        raise TransportFailure("empty_response")
    if any(type(v) is not int or v < 0 for v in reply.usage.values()):
        raise TransportFailure("invalid_usage")


@contextmanager
def round_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "round.lock").open("a") as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise TransportStop("another worker owns this round") from None
        yield


def _retry_after(value):
    if value is None:
        return 0.0
    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            seconds = (date - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError):
            return 0.0
    return max(0.0, seconds) if math.isfinite(seconds) else 0.0


def retry_http(status, headers):
    if status == 429:
        return TransportFailure("http_429", retryable=True, retry_after=_retry_after(headers.get("retry-after")))
    if status in (401, 403):
        return TransportFailure(f"http_{status}")
    return TransportFailure(f"http_{status}", ambiguous=status >= 500)


def _may_reconcile_previous(previous, decision):
    if previous["status"] != "stopped":
        return False
    if previous.get("ambiguous"):
        return True
    rejected_before_execution = {"http_401", "http_403", "http_429", "credential_file_error"}
    return decision == "confirmed_not_processed" and previous.get("error") in rejected_before_execution


def terminal_record(log, index):
    standard = log / f"attempt_{index:02d}.terminal.json"
    recovery = log / f"attempt_{index:02d}.recovery.terminal.json"
    if recovery.exists():
        old = read(standard)
        saved = read(recovery)
        authorization = read(log / f"attempt_{index:02d}.reconciliation.json")
        if (not _may_reconcile_previous(old, authorization["decision"])
                or saved["supersedes_sha256"] != digest(old)
                or saved["reconciliation_sha256"] != digest(authorization)):
            raise TransportStop("reconciled receipt mismatch")
        return saved
    saved = read(standard)
    if "reconciliation_sha256" in saved:
        if saved["reconciliation_sha256"] != digest(read(log / f"attempt_{index:02d}.reconciliation.json")):
            raise TransportStop("reconciled receipt mismatch")
    return saved


def run_request(directory, request, config, call, *, sleep=time.sleep, jitter=random.random, now=time.time,
                attempt_slot=None, managed_attempt=None):
    """Run direct attempts or recover an immutable managed attempt without resending it here."""
    check_request(request, request["binding"], config)
    if attempt_slot is not None and managed_attempt is not None:
        raise ValueError("a managed attempt cannot use a direct provider slot")
    log = directory / f"turn_{request['logical_index']:02d}"
    once(log / "request.json", request)
    rates = 0
    request_id = request["request_id"]
    for index in range(config.transport_attempts):
        started_path = log / f"attempt_{index:02d}.started.json"
        terminal_path = log / f"attempt_{index:02d}.terminal.json"
        started = None
        if started_path.exists():
            started = read(started_path)
            if (started["request_id"], started["attempt"]) != (request_id, index):
                raise TransportStop("started receipt mismatch")
            if not terminal_path.exists() and (
                managed_attempt is None or started.get("transport") != "mac_bridge"
            ):
                raise TransportStop("ambiguous interrupted request requires reconciliation")
        elif terminal_path.exists():
            raise TransportStop("terminal receipt has no started receipt")
        if not terminal_path.exists():
            scope = attempt_slot(request, config) if attempt_slot else nullcontext()
            # Direct callers acquire a provider slot here. A managed attempt has
            # a durable GPU reservation before it joins the Mac provider queue.
            with scope as credential_id:
                if started is None:
                    started = {"request_id": request_id, "attempt": index, "attempt_id": uuid4().hex,
                               "started_unix": now()}
                    if managed_attempt is not None:
                        started["transport"] = "mac_bridge"
                    if credential_id is not None:
                        started["credential_id"] = credential_id
                    once(started_path, started)
                start = time.monotonic()
                evidence = {}
                try:
                    reply = (managed_attempt(request, config, started) if managed_attempt is not None
                             else call(request, config))
                    evidence = {"response_id": reply.response_id, "served_model": reply.served_model,
                                "usage": reply.usage, "provider_request_id": reply.provider_request_id,
                                "provider_status": reply.status}
                    if managed_attempt is not None:
                        evidence.update(reply.transport_evidence or {})
                    _validate_reply(reply, config)
                except TransportFailure as error:
                    evidence.update(getattr(error, "reply_identity", None) or {})
                    delay = max(error.retry_after or 0, min(config.max_delay_seconds,
                        config.base_delay_seconds * 2 ** index * (1 + jitter() * 0.25)))
                    if error.code == "http_429":
                        rates += 1
                    can_retry = (
                        error.retryable and not error.ambiguous and delay <= config.max_delay_seconds
                        and rates < config.rate_limit_threshold and index + 1 < config.transport_attempts
                    )
                    saved = {**started, **evidence, "status": "retryable" if can_retry else "stopped",
                             "error": error.code, "ambiguous": error.ambiguous,
                             "retry_after_seconds": error.retry_after, "rate_limit_count": rates,
                             "circuit_open": rates >= config.rate_limit_threshold or delay > config.max_delay_seconds,
                             "not_before_unix": now() + delay,
                             "elapsed_seconds": time.monotonic() - start}
                else:
                    saved = {**started, **evidence, "status": "completed", "text": reply.text,
                             "answer_sha256": digest(reply.text), "elapsed_seconds": time.monotonic() - start}
                once(terminal_path, saved)
        saved = terminal_record(log, index)
        if any(saved.get(k) != started[k] for k in started):
            raise TransportStop("terminal receipt mismatch")
        if saved["status"] == "completed":
            if saved["answer_sha256"] != digest(saved["text"]):
                raise TransportStop("completed response digest mismatch")
            _validate_reply(ProviderReply(saved["text"], saved["response_id"], saved["served_model"],
                                          saved["usage"], saved["provider_status"]), config)
            return saved
        if saved["status"] != "retryable" or saved["ambiguous"]:
            raise TransportStop(saved["error"])
        rates = saved["rate_limit_count"]
        delay = saved["not_before_unix"] - now()
        next_started = log / f"attempt_{index + 1:02d}.started.json"
        if delay > 0 and not next_started.exists():
            sleep(delay)
    raise TransportStop("transport attempt budget exhausted")


def reconcile(directory, request, attempt, *, authorization, config, reply=None):
    """Import an independently recovered completion or record verified nonexecution."""
    check_request(request, request["binding"], config)
    if not 0 <= attempt < config.transport_attempts:
        raise ValueError("invalid attempt index")
    log = directory / f"turn_{request['logical_index']:02d}"
    with round_lock(directory):
        started = read(log / f"attempt_{attempt:02d}.started.json")
        terminal = log / f"attempt_{attempt:02d}.terminal.json"
        previous = read(terminal) if terminal.exists() else None
        recovered = log / f"attempt_{attempt:02d}.recovery.terminal.json"
        if (
            started["request_id"] != request["request_id"] or started["attempt"] != attempt
            or recovered.exists() or previous is not None and (
                not _may_reconcile_previous(previous, authorization.get("decision"))
                or any(previous.get(k) != v for k, v in started.items())
            )
        ):
            raise TransportStop("attempt is not awaiting reconciliation")
        expected = {"request_id": started["request_id"], "attempt_id": started["attempt_id"],
                    "decision": "import_completed" if reply else "confirmed_not_processed"}
        if any(authorization.get(k) != v for k, v in expected.items()) or not all(
            isinstance(authorization.get(k), str) and authorization[k].strip()
            for k in ("authorized_by", "evidence")
        ):
            raise PermissionError("recovery requires request-bound human authorization and evidence")
        if not reply and attempt + 1 >= config.transport_attempts:
            raise TransportStop("no physical attempt remains in this logical request")
        if reply:
            _validate_reply(reply, config)
            saved = {**started, "status": "completed", "text": reply.text, "answer_sha256": digest(reply.text),
               "response_id": reply.response_id, "served_model": reply.served_model, "usage": reply.usage,
               "provider_request_id": reply.provider_request_id, "provider_status": reply.status}
        else:
            saved = {**started, "status": "retryable", "error": "confirmed_not_processed",
                     "ambiguous": False, "rate_limit_count": sum(
                         read(path).get("error") == "http_429" for path in log.glob("*.terminal.json")
                     ), "not_before_unix": time.time()}
        once(log / f"attempt_{attempt:02d}.reconciliation.json", authorization)
        reconciled = {**saved, "reconciliation_sha256": digest(authorization), "elapsed_seconds": None}
        if previous is None:
            once(terminal, reconciled)
        else:
            once(recovered, {**reconciled, "supersedes_sha256": digest(previous)})


def collect_completed(events, *, deadline=None):
    """Consume the terminal Responses event, never decide from text deltas."""
    final = None
    for event in events:
        if deadline is not None and time.monotonic() > deadline:
            raise TransportFailure("timeout", ambiguous=True)
        if event.type == "response.completed":
            if final is not None:
                raise TransportFailure("duplicate_completion", ambiguous=True)
            final = event.response
        elif event.type in {"response.incomplete", "response.failed", "error"}:
            raise TransportFailure(event.type, ambiguous=True)
    if final is None:
        raise TransportFailure("sse_missing_completion", ambiguous=True)
    details = final.usage
    usage = {key: value for key in ("input_tokens", "output_tokens", "total_tokens")
             if (value := getattr(details, key, None)) is not None}
    for name, key in (("input_tokens_details", "cached_tokens"), ("output_tokens_details", "reasoning_tokens")):
        if (value := getattr(getattr(details, name, None), key, None)) is not None:
            usage[key] = value
    return ProviderReply(final.output_text, final.id, final.model, usage, final.status,
                         getattr(final, "_request_id", None))


def responses_call(request, config):
    # Imported only on an explicitly authorized paid execution path.
    from importlib.metadata import version
    import openai
    import httpx

    if config.sdk_version != version("openai"):
        raise TransportFailure("sdk_version_mismatch")
    try:
        deadline = time.monotonic() + config.timeout_seconds
        with openai.OpenAI(base_url=config.base_url, timeout=config.timeout_seconds, max_retries=0) as client:
            with client.responses.create(**request["payload"],
                    extra_headers={"X-Client-Request-Id": request["request_id"]}) as stream:
                reply = collect_completed(stream, deadline=deadline)
                if not reply.provider_request_id:
                    reply = ProviderReply(reply.text, reply.response_id, reply.served_model, reply.usage,
                                          reply.status, stream.response.headers.get("x-request-id"))
                return reply
    except openai.APIStatusError as error:
        raise retry_http(error.status_code, error.response.headers) from None
    except (openai.APITimeoutError, httpx.TimeoutException):
        raise TransportFailure("timeout", ambiguous=True) from None
    except openai.APIConnectionError as error:
        if isinstance(error.__cause__, httpx.ConnectError):
            raise TransportFailure("connect_error", retryable=True) from None
        raise TransportFailure("connection_interrupted", ambiguous=True) from None
    except (ValueError, httpx.TransportError):
        raise TransportFailure("sse_frame_error", ambiguous=True) from None
    except openai.OpenAIError:
        raise TransportFailure("sdk_error", ambiguous=True) from None
