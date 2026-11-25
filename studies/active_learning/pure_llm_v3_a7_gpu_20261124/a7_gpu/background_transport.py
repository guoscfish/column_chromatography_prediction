"""Durable, single-submission custody for an explicitly approved background Responses client."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import time
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, once

from .bridge_protocol import MAX_ENVELOPE_BYTES, _check_raw, read_bridge_json
from .dispatcher import _exclusive, _private_directory
from .key_pool import KEY_ID
from .provider import TOKEN, http_failure
from .ssh_courier import SHA256

SCHEMA = "a7_durable_background_transport_v1"
RESPONSE_ID = re.compile(r"resp_[A-Za-z0-9_-]{1,123}")
CREDENTIAL_TEXT = re.compile(r"(?i)\b(?:bearer [A-Za-z0-9_.-]{12,}|sk-[A-Za-z0-9_-]{12,})")
PENDING = ("queued", "in_progress")
TERMINAL = ("completed", "incomplete", "failed", "cancelled")


@dataclass(frozen=True)
class BackgroundBinding:
    """Caller-verified authorization, scientific request and credential identities."""

    authorization_sha256: str
    request_sha256: str
    wire_sha256: str
    credential_alias: str
    model: str
    reasoning_effort: str
    served_model: str | None = None

    def __post_init__(self):
        fields = ((SHA256, self.authorization_sha256), (SHA256, self.request_sha256), (SHA256, self.wire_sha256),
                  (KEY_ID, self.credential_alias), (TOKEN, self.model), (TOKEN, self.reasoning_effort))
        if (any(not isinstance(value, str) or not pattern.fullmatch(value) for pattern, value in fields)
                or self.served_model is not None and (
                    not isinstance(self.served_model, str) or not TOKEN.fullmatch(self.served_model))):
            raise ValueError("invalid durable background binding")

    @property
    def expected_served_model(self):
        return self.model if self.served_model is None else self.served_model


class BackgroundProtocolError(ValueError):
    """A provider reply did not prove its expected status or custody."""


def _token(value, secret):
    if (isinstance(value, str) and TOKEN.fullmatch(value) and not value.lower().startswith("sk-")
            and (not secret or secret not in value)):
        return value
    return None


def _response_fields(raw, *, expected_id, initial):
    if not isinstance(raw, dict) or raw.get("status") not in (*PENDING, *TERMINAL):
        raise BackgroundProtocolError("invalid_response_status")
    identity = raw.get("id")
    if (not isinstance(identity, str) or not RESPONSE_ID.fullmatch(identity)
            or expected_id is not None and identity != expected_id):
        raise BackgroundProtocolError("response_id_mismatch")
    # The original POST needs a positive acknowledgement. Subsequent GETs may omit the
    # field if the original accepted response established background execution.
    background = raw.get("background")
    if background is not True and (initial or background is not None):
        raise BackgroundProtocolError("provider_did_not_affirm_background")
    return identity, raw["status"]


def _validate_response(raw, binding, *, expected_id=None, initial=False, secret=None):
    _check_raw(raw)
    encoded = dumps(raw)
    if (len(encoded.encode()) >= MAX_ENVELOPE_BYTES - 4096 or CREDENTIAL_TEXT.search(encoded)
            or secret and secret in encoded):
        raise BackgroundProtocolError("unsafe_response_evidence")
    identity, status = _response_fields(raw, expected_id=expected_id, initial=initial)
    reasoning = raw.get("reasoning")
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    if (raw.get("model") != binding.expected_served_model or reasoning is not None and not isinstance(reasoning, dict)
            or effort not in (None, binding.reasoning_effort)):
        raise BackgroundProtocolError("served_model_or_effort_mismatch")
    if status == "completed":
        output = raw.get("output")
        if not isinstance(output, list) or any(not isinstance(item, dict) for item in output):
            raise BackgroundProtocolError("invalid_completed_output")
        messages = [item.get("content") for item in output if item.get("type") == "message"]
        if (any(not isinstance(content, list) or any(not isinstance(block, dict) for block in content)
                for content in messages)
                or not any(block.get("type") == "output_text" and isinstance(block.get("text"), str)
                           and block["text"].strip() for content in messages for block in content)):
            raise BackgroundProtocolError("missing_completed_output")
        usage = raw.get("usage")
        if usage is not None and (not isinstance(usage, dict) or any(
            type(usage[name]) is not int or usage[name] < 0
            for name in ("input_tokens", "output_tokens", "total_tokens") if name in usage
        )):
            raise BackgroundProtocolError("invalid_completed_usage")
    return {"response_id": identity, "status": status, "model": binding.expected_served_model,
            "reasoning_effort": effort,
            "background": raw.get("background")}


def _read_optional(path):
    return read_bridge_json(path) if path.exists() else None


class BackgroundLedger:
    """Seal one paid POST and reattach to the accepted response with same-ID GETs."""

    def __init__(self, path, *, binding, payload):
        if (not isinstance(binding, BackgroundBinding) or type(payload) is not dict
                or digest(payload) != binding.wire_sha256 or payload.get("model") != binding.model
                or payload.get("reasoning") != {"effort": binding.reasoning_effort}
                or payload.get("background") is not True or payload.get("stream") is not False
                or type(payload.get("store")) is not bool):
            raise PermissionError("payload differs from the approved background transport binding")
        self.path = Path(path).expanduser().absolute()
        self.binding = binding
        self._payload = deepcopy(payload)
        _private_directory(self.path, create=True)
        scope = {"schema": SCHEMA, "directory": str(self.path), "binding": asdict(binding)}
        self.scope_sha256 = digest(scope)
        once(self.path / "scope.json", scope)

    def _read(self, name):
        saved = _read_optional(self.path / name)
        if saved is not None and (type(saved) is not dict or saved.get("scope_sha256") != self.scope_sha256):
            raise PermissionError("background receipt belongs to a different scope")
        return saved

    def _write(self, name, receipt):
        once(self.path / name, {"scope_sha256": self.scope_sha256, **receipt})

    def _valid_receipt(self, saved, *, response_id=None, initial=False):
        raw = saved["raw_response"]
        summary = _validate_response(raw, self.binding, expected_id=response_id, initial=initial)
        if (digest(raw) != saved["raw_response_sha256"] or summary != saved["response"]):
            raise PermissionError("background response receipt failed integrity verification")
        return summary

    def _credential(self, alias):
        if alias != self.binding.credential_alias:
            raise PermissionError("background transport must use the original credential alias")

    @staticmethod
    def _check_client(client):
        retries = getattr(client, "max_retries", None)
        http = getattr(client, "_client", None)
        if type(retries) is not int or retries != 0 or getattr(http, "follow_redirects", None) is not False:
            raise PermissionError("background SDK client must disable implicit retries and HTTP redirects")

    def _submission(self):
        submitted = self._read("submission.json")
        if submitted is not None:
            intent = self._read("intent.json")
            if intent is None or intent.get("credential_alias") != self.binding.credential_alias:
                raise PermissionError("accepted background response lacks its original intent")
        return submitted

    def _call(self, client, *, response_id=None, on_dispatch=None):
        secret = getattr(client, "api_key", None)
        if not isinstance(secret, str):
            secret = None
        raw = None
        request_id = None
        try:
            api = client.responses.with_raw_response
            if on_dispatch is not None:
                on_dispatch("POST" if response_id is None else "GET")
            http = (api.create(**deepcopy(self._payload), extra_headers={
                "X-Client-Request-Id": self.binding.request_sha256,
            }) if response_id is None else api.retrieve(response_id))
            request_id = _token(http.headers.get("x-request-id"), secret)
            raw = http.parse().model_dump(mode="json", exclude_unset=True)
            summary = _validate_response(raw, self.binding, expected_id=response_id,
                                         initial=response_id is None, secret=secret)
        except Exception as error:
            http_status = getattr(error, "status_code", None)
            evidence = {"error": (str(error) if isinstance(error, BackgroundProtocolError) else
                                  _token(type(error).__name__, secret) or "unknown_transport_error"),
                        "provider_request_id": request_id or _token(getattr(error, "request_id", None), secret)}
            if type(http_status) is int and 100 <= http_status <= 599:
                evidence["http_status"] = http_status
                headers = getattr(getattr(error, "response", None), "headers", {})
                failure, scope = http_failure(http_status, headers, getattr(error, "body", None))
                evidence.update(failure_code=failure.code, rate_limit_scope=scope,
                                retry_after=failure.retry_after)
            if isinstance(raw, dict):
                identity = _token(raw.get("id"), secret)
                if identity and RESPONSE_ID.fullmatch(identity):
                    evidence["observed_response_id"] = identity
                if type(raw.get("background")) is bool:
                    evidence["observed_background"] = raw["background"]
            return evidence
        return {"response": summary, "provider_request_id": request_id, "raw_response_sha256": digest(raw),
                "raw_response": raw}

    def submit(self, client, *, credential_alias, allow_post=False, on_dispatch=None):
        """Reserve and send the only POST. Invoke the dispatch hook only for a live call."""
        self._credential(credential_alias)
        if allow_post is not True:
            raise PermissionError("background submission requires the caller's exact paid authorization")
        with _exclusive(self.path / "operation.lock"):
            if self._read("intent.json") is None:
                if any((self.path / name).exists() for name in
                       ("submission.json", "submission_error.json", "retrieval_terminal.json", "polls")):
                    raise PermissionError("background submission custody is missing its original intent")
                self._check_client(client)
                self._write("intent.json", {"credential_alias": credential_alias, "created_ns": time.time_ns()})
                receipt = self._call(client, on_dispatch=on_dispatch)
                record = "submission.json" if "response" in receipt else "submission_error.json"
                self._write(record, receipt)
            return self.status()

    def poll(self, client, *, credential_alias, on_dispatch=None):
        """Send at most one GET for the accepted response using its original credential."""
        self._credential(credential_alias)
        with _exclusive(self.path / "operation.lock"):
            submitted = self._submission()
            if submitted is None:
                return self.status()
            initial = self._valid_receipt(submitted, initial=True)
            if self._read("retrieval_terminal.json") is None and initial["status"] in PENDING:
                self._check_client(client)
                receipt = self._call(client, response_id=initial["response_id"], on_dispatch=on_dispatch)
                response = receipt.get("response")
                if response and response["status"] in TERMINAL:
                    self._write("retrieval_terminal.json", receipt)
                else:
                    _private_directory(self.path / "polls", create=True)
                    identity = f"{time.time_ns():020d}-{uuid4().hex}"
                    self._write(f"polls/{identity}.json", receipt)
            return self.status()

    def _final_receipt(self, submitted, initial):
        terminal = self._read("retrieval_terminal.json")
        if terminal is not None:
            accepted = self._valid_receipt(terminal, response_id=initial["response_id"])
            if accepted["status"] not in TERMINAL:
                raise PermissionError("retrieval terminal receipt contains a nonterminal response")
            return terminal
        if initial["status"] in TERMINAL:
            return submitted
        return None

    def completed(self):
        """Return the verified completed raw artifact to the integrating bridge."""
        submitted = self._submission()
        if submitted is None:
            return None
        initial = self._valid_receipt(submitted, initial=True)
        terminal = self._final_receipt(submitted, initial)
        return terminal["raw_response"] if terminal and terminal["response"]["status"] == "completed" else None

    def status(self):
        """Return identifiers, protocol state and errors with no raw output or credentials."""
        intent = self._read("intent.json")
        output = {"scope_sha256": self.scope_sha256, "credential_alias": self.binding.credential_alias,
                  "post_reserved": intent is not None, "state": "planned" if intent is None else "submission_unknown"}
        submitted = self._submission()
        if submitted is None:
            failure = self._read("submission_error.json")
            if failure is not None:
                output.update(state="submission_unverified", last_error=failure["error"])
                for field in ("http_status", "provider_request_id", "observed_background", "observed_response_id",
                              "failure_code", "rate_limit_scope", "retry_after"):
                    if field in failure:
                        output[field] = failure[field]
            return output
        initial = self._valid_receipt(submitted, initial=True)
        terminal = self._final_receipt(submitted, initial)
        if terminal is not None:
            output.update(terminal["response"], state=terminal["response"]["status"],
                          provider_request_id=terminal["provider_request_id"])
            return output
        output.update(initial, state=initial["status"], provider_request_id=submitted["provider_request_id"])
        last = max((self.path / "polls").glob("*.json"), key=lambda path: path.name, default=None)
        if last is not None:
            latest = self._read(last.relative_to(self.path))
            if "error" in latest:
                output.update(last_error=latest["error"], last_http_status=latest.get("http_status"),
                              last_failure_code=latest.get("failure_code"), last_rate_limit_scope=latest.get("rate_limit_scope"),
                              last_retry_after=latest.get("retry_after"))
            else:
                response = self._valid_receipt(latest, response_id=initial["response_id"])
                output.update(response, state=response["status"])
        return output
