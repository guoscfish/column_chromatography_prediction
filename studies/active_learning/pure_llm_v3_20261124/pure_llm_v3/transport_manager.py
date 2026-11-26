"""Bound provider requests, immutable receipts and bounded transport retries."""
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import fcntl
import math
import random
import re
import time
from urllib.parse import urlparse
from uuid import uuid4

from .records import digest, once, read


class TransportStop(RuntimeError):
    pass


RESPONSES_STREAM = "responses_stream"
RESPONSES_STREAM_IPV4 = "responses_stream_ipv4"
RESPONSES_JSON_IPV4 = "responses_json_ipv4"
RESPONSES_WEBSOCKET_IPV4 = "responses_websocket_ipv4"
RESPONSES_BACKGROUND = "responses_background"


UNKNOWN_REPLAY_DECISION = "authorized_replay_unknown"
REPLAY_AUTH_FIELDS = {"decision", "authorized_by", "authorization_reference", "accept_duplicate_billing_risk",
                      "contract_sha256", "request_id", "attempt_id", "bridge_result_sha256",
                      "original_terminal_sha256", "next_attempt"}


class TransportFailure(Exception):
    def __init__(self, code, *, retryable=False, retry_after=None, ambiguous=False):
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.retry_after = retry_after
        self.ambiguous = ambiguous


class ResponseMessagePhaseError(TransportFailure):
    def __init__(self):
        super().__init__("invalid_response_message_phase")


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
    wire_mode: str = RESPONSES_STREAM
    sdk_version: str = "offline"

    def __post_init__(self):
        endpoint = urlparse(self.base_url)
        if (endpoint.scheme != "https" or not endpoint.hostname or endpoint.username or endpoint.password
                or endpoint.query or endpoint.fragment
                or self.wire_mode not in (RESPONSES_STREAM, RESPONSES_STREAM_IPV4, RESPONSES_JSON_IPV4,
                                          RESPONSES_WEBSOCKET_IPV4, RESPONSES_BACKGROUND)):
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
    background = config.wire_mode == RESPONSES_BACKGROUND
    payload = {"model": config.requested_model, "reasoning": {"effort": config.reasoning_effort},
               "input": messages, "max_output_tokens": config.max_output_tokens,
               "stream": not background and config.wire_mode != RESPONSES_JSON_IPV4, "store": False}
    if background:
        payload["background"] = True
    return payload


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


def _replay_paths(log):
    return log / "attempt_00.replay_authorization.json", log / "attempt_00.replay.terminal.json"


def _unknown_replay_record(request, started, previous, authorization):
    scope = request["binding"].get("a7_formal", {})
    if (scope.get("mock") is not False or started.get("transport") != "mac_bridge"
            or started.get("request_id") != request["request_id"] or started.get("attempt") != 0
            or not re.fullmatch(r"[0-9a-f]{32}", started.get("attempt_id", ""))
            or any(previous.get(key) != value for key, value in started.items())
            or previous.get("status") != "stopped" or previous.get("ambiguous") is not True
            or previous.get("provider_dispatched") is not True or previous.get("circuit_open") is not False
            or not re.fullmatch(r"[0-9a-f]{64}", previous.get("bridge_result_sha256", ""))
            or previous.get("provider_status") == "completed" or "answer_sha256" in previous):
        raise TransportStop("audited replay requires the original stopped ambiguous bridge attempt")
    expected = {"decision": UNKNOWN_REPLAY_DECISION, "request_id": request["request_id"],
                "contract_sha256": scope.get("contract_sha256"), "attempt_id": started["attempt_id"],
                "bridge_result_sha256": previous["bridge_result_sha256"],
                "original_terminal_sha256": digest(previous), "next_attempt": 1}
    if (type(authorization) is not dict or set(authorization) != REPLAY_AUTH_FIELDS
            or any(authorization[key] != value for key, value in expected.items())
            or type(authorization["next_attempt"]) is not int
            or not re.fullmatch(r"[0-9a-f]{64}", authorization["contract_sha256"] or "")
            or authorization["accept_duplicate_billing_risk"] is not True
            or any(not isinstance(authorization[key], str) or not authorization[key].strip()
                   for key in ("authorized_by", "authorization_reference"))):
        raise PermissionError("audited replay authorization does not bind the original unknown request")
    return {**started, "status": "retryable", "error": UNKNOWN_REPLAY_DECISION, "ambiguous": False,
            "prior_execution_and_billing": "unresolved", "next_attempt": 1, "elapsed_seconds": None,
            "rate_limit_count": previous["rate_limit_count"], "not_before_unix": previous["not_before_unix"],
            "supersedes_sha256": authorization["original_terminal_sha256"],
            "replay_authorization_sha256": digest(authorization)}


def _check_replay_digest(authorization, expected_sha256):
    if not isinstance(expected_sha256, str) or digest(authorization) != expected_sha256:
        raise PermissionError("audited replay authorization differs from the trusted digest")


def _link_replay_attempt(log, started, replay, *, allow_create):
    if (started.get("request_id") != replay["request_id"] or started.get("transport") != "mac_bridge"
            or started.get("attempt") != replay["next_attempt"] or started.get("attempt_id") == replay["attempt_id"]):
        raise TransportStop("audited replay attempt does not match its permit")
    path = log / "attempt_01.replay_link.json"
    expected = {"request_id": started["request_id"], "attempt": started["attempt"],
                "attempt_id": started["attempt_id"], "original_attempt_id": replay["attempt_id"],
                "original_terminal_sha256": replay["supersedes_sha256"],
                "replay_authorization_sha256": replay["replay_authorization_sha256"]}
    if not path.exists() and allow_create:
        once(path, expected)
    if not path.is_file() or read(path) != expected:
        raise TransportStop("audited replay attempt link mismatch")


def terminal_record(log, index, *, replay_authorization_sha256=None):
    standard = log / f"attempt_{index:02d}.terminal.json"
    recovery = log / f"attempt_{index:02d}.recovery.terminal.json"
    if replay_authorization_sha256 is not None:
        if index != 0 or recovery.exists() or (log / "attempt_00.reconciliation.json").exists():
            raise TransportStop("audited replay conflicts with ordinary reconciliation")
        authorization_path, receipt_path = _replay_paths(log)
        authorization = read(authorization_path)
        _check_replay_digest(authorization, replay_authorization_sha256)
        expected = _unknown_replay_record(read(log / "request.json"), read(log / "attempt_00.started.json"),
                                          read(standard), authorization)
        if read(receipt_path) != expected:
            raise TransportStop("audited replay receipt mismatch")
        return expected
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
                attempt_slot=None, managed_attempt=None, replay_authorizations=None, replay_authorization_factory=None,
                attempt_limit=None):
    """Run direct attempts or recover an immutable managed attempt without resending it here."""
    check_request(request, request["binding"], config)
    limit = config.transport_attempts if attempt_limit is None else attempt_limit
    if type(limit) is not int or not 1 <= limit <= config.transport_attempts:
        raise ValueError("physical attempt limit must lie inside the registered transport budget")
    if attempt_slot is not None and managed_attempt is not None:
        raise ValueError("a managed attempt cannot use a direct provider slot")
    if replay_authorization_factory is not None and (
        not callable(replay_authorization_factory) or managed_attempt is None or limit != 2
    ):
        raise PermissionError("automatic audited replay requires the two-slot managed bridge")
    log = directory / f"turn_{request['logical_index']:02d}"
    once(log / "request.json", request)
    rates = 0
    request_id = request["request_id"]
    replay_sha256 = (replay_authorizations or {}).get(request_id)
    replay = None
    if replay_sha256 is not None:
        if managed_attempt is None or limit != 2:
            raise PermissionError("audited replay needs the original two-attempt managed bridge")
        replay = terminal_record(log, 0, replay_authorization_sha256=replay_sha256)
    for index in range(limit):
        started_path = log / f"attempt_{index:02d}.started.json"
        terminal_path = log / f"attempt_{index:02d}.terminal.json"
        started = None
        replaying = index == 1 and replay is not None
        if started_path.exists():
            started = read(started_path)
            if (started["request_id"], started["attempt"]) != (request_id, index):
                raise TransportStop("started receipt mismatch")
            if not terminal_path.exists() and (
                managed_attempt is None or started.get("transport") != "mac_bridge"
            ):
                raise TransportStop("ambiguous interrupted request requires reconciliation")
            if replaying:
                _link_replay_attempt(log, started, replay, allow_create=not terminal_path.exists())
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
                    if replaying:
                        _link_replay_attempt(log, started, replay, allow_create=True)
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
                        and rates < config.rate_limit_threshold and index + 1 < limit
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
        saved = replay if index == 0 and replay is not None else terminal_record(log, index)
        if any(saved.get(k) != started[k] for k in started):
            raise TransportStop("terminal receipt mismatch")
        if saved["status"] == "completed":
            if saved["answer_sha256"] != digest(saved["text"]):
                raise TransportStop("completed response digest mismatch")
            _validate_reply(ProviderReply(saved["text"], saved["response_id"], saved["served_model"],
                                          saved["usage"], saved["provider_status"]), config)
            return saved
        if index == 0 and replay is None and saved["ambiguous"] and replay_authorization_factory is not None:
            authorization = replay_authorization_factory(request, saved, config)
            if authorization is not None:
                replay = _seal_ambiguous_replay(log, request, authorization, digest(authorization))
                saved = replay
        if saved["status"] != "retryable" or saved["ambiguous"]:
            raise TransportStop(saved["error"])
        rates = saved["rate_limit_count"]
        delay = saved["not_before_unix"] - now()
        next_started = log / f"attempt_{index + 1:02d}.started.json"
        if delay > 0 and not next_started.exists():
            sleep(delay)
    raise TransportStop("transport attempt budget exhausted")


def _seal_ambiguous_replay(log, request, authorization, expected_authorization_sha256):
    authorization_path, receipt_path = _replay_paths(log)
    if (log / "attempt_00.reconciliation.json").exists() or (log / "attempt_00.recovery.terminal.json").exists():
        raise TransportStop("audited replay conflicts with ordinary reconciliation")
    if request != read(log / "request.json"):
        raise TransportStop("audited replay request differs from its original archive")
    if receipt_path.exists():
        return terminal_record(log, 0, replay_authorization_sha256=expected_authorization_sha256)
    if any((log / f"attempt_01.{kind}.json").exists() for kind in ("started", "terminal")):
        raise TransportStop("audited replay requires an unused second physical attempt")
    receipt = _unknown_replay_record(request, read(log / "attempt_00.started.json"),
                                     read(log / "attempt_00.terminal.json"), authorization)
    once(authorization_path, authorization)
    once(receipt_path, receipt)
    return receipt


def authorize_ambiguous_replay(directory, request, *, authorization, config, expected_authorization_sha256):
    """Seal the single remaining attempt under a separate, trusted risk authorization."""
    check_request(request, request["binding"], config)
    _check_replay_digest(authorization, expected_authorization_sha256)
    if config.transport_attempts != 2:
        raise PermissionError("audited replay needs the original two-attempt managed bridge")
    log = directory / f"turn_{request['logical_index']:02d}"
    with round_lock(directory):
        return _seal_ambiguous_replay(log, request, authorization, expected_authorization_sha256)


def reconcile(directory, request, attempt, *, authorization, config, reply=None):
    """Import an independently recovered completion or record verified nonexecution."""
    check_request(request, request["binding"], config)
    if not 0 <= attempt < config.transport_attempts:
        raise ValueError("invalid attempt index")
    log = directory / f"turn_{request['logical_index']:02d}"
    with round_lock(directory):
        if attempt == 0 and any(path.exists() for path in _replay_paths(log)):
            raise TransportStop("ordinary reconciliation conflicts with audited replay")
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


def _response_field(item, field):
    return item.get(field) if isinstance(item, dict) else getattr(item, field, None)


def response_output_text(output):
    """Read the final answer from either SDK output items or their archived JSON objects."""
    if not isinstance(output, list):
        raise ValueError("invalid provider output list")
    messages = [item for item in output if _response_field(item, "type") == "message"]
    phases = [_response_field(item, "phase") for item in messages]
    phase_aware = any(phase is not None for phase in phases)
    if phase_aware:
        finals = [item for item, phase in zip(messages, phases, strict=True) if phase == "final_answer"]
        if (len(finals) != 1 or any(phase not in ("commentary", "final_answer") for phase in phases)
                or any(_response_field(item, "role") != "assistant" for item in messages)
                or _response_field(finals[0], "status") != "completed"):
            raise ResponseMessagePhaseError()
        messages = finals
    text = []
    for item in messages:
        content = _response_field(item, "content")
        if not isinstance(content, list):
            raise ValueError("invalid provider message content")
        for part in content:
            if _response_field(part, "type") == "output_text":
                value = _response_field(part, "text")
                if not isinstance(value, str):
                    raise ValueError("invalid provider text")
                text.append(value)
    answer = "".join(text)
    if phase_aware and not answer.strip():
        raise ResponseMessagePhaseError()
    return answer


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
    return ProviderReply(response_output_text(final.output), final.id, final.model, usage, final.status,
                         getattr(final, "_request_id", None))


def responses_call(request, config):
    if config.wire_mode == RESPONSES_JSON_IPV4:
        raise TransportFailure("json_ipv4_dispatcher_required")
    if config.wire_mode == RESPONSES_WEBSOCKET_IPV4:
        raise TransportFailure("websocket_ipv4_dispatcher_required")
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
