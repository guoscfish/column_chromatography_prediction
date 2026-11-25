"""Immutable GPU-to-Mac work items and response receipts over authenticated SSH."""

import math
import os
from pathlib import Path
import re
import stat
import time

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import SCHEMA_VERSION, parse_response
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    ProviderReply, TransportConfig, TransportFailure, TransportStop, check_request,
)

from .compact_payload import FORMAT_SUFFIX, WIDTHS, validate_wire_representation
from .provider import verify_reported_model


JOB_VERSION = "purellm-mac-bridge-job-v1"
RESULT_VERSION = "purellm-mac-bridge-result-v1"
MAX_ENVELOPE_BYTES = 16 * 1024 * 1024
ATTEMPT_ID = re.compile(r"[0-9a-f]{32}")
SHA256 = re.compile(r"[0-9a-f]{64}")
CREDENTIAL_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,31}")
FAILURE_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,79}")
RETRYABLE_CODES = {"connect_error", "http_429"}
JOB_FIELDS = {"version", "attempt_id", "request_id", "request", "started", "job_sha256"}
STARTED_FIELDS = {"request_id", "attempt", "attempt_id", "started_unix", "transport"}
RESULT_FIELDS = {
    "version", "attempt_id", "request_id", "job_sha256", "credential_id", "provider_dispatched",
    "elapsed_seconds", "model_identity", "outcome", "raw_response", "raw_response_sha256", "result_sha256",
}
REPLY_FIELDS = {"text", "response_id", "served_model", "usage", "status", "provider_request_id"}
FAILURE_FIELDS = {"code", "retryable", "retry_after", "ambiguous", "reply_identity"}
REPLY_IDENTITY_FIELDS = {"response_id", "served_model", "usage", "provider_request_id", "provider_status"}
SENSITIVE_FIELDS = {
    "apikey", "authorization", "clientsecret", "password", "accesstoken", "refreshtoken",
    "proxyauthorization", "signaturesecret", "token4researchapikey",
}
_UNSPECIFIED_WIRE = object()


class BridgePending(TransportStop):
    """Remote worker can reattach to the same job after controller delay or pause."""


def _matches(pattern, value):
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _duration(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _validate_started(request, config, started):
    if (type(started) is not dict or set(started) != STARTED_FIELDS
            or started["request_id"] != request["request_id"]
            or started["transport"] != "mac_bridge"
            or not _matches(ATTEMPT_ID, started["attempt_id"])
            or type(started["attempt"]) is not int or not 0 <= started["attempt"] < config.transport_attempts
            or not _duration(started["started_unix"])):
        raise ValueError("unregistered bridge attempt identity")


def _validate_binding(request, authorization_sha256, contract_sha256):
    binding = request["binding"]
    scope = binding["a7_formal"]
    messages = request["payload"]["input"]
    roles = ["system", "user"] + (["user"] if request["logical_index"] else [])
    if (type(scope) is not dict
            or set(scope) != {"contract_sha256", "authorization_sha256", "seed", "mock"}
            or (binding.get("arm"), binding.get("schema_version"), binding.get("prompt_version"))
            != ("A7", SCHEMA_VERSION, "v3") or type(binding.get("seed")) is not int
            or scope["seed"] != binding["seed"] or type(binding.get("round")) is not int or binding["round"] < 0
            or type(scope["mock"]) is not bool or type(request["logical_index"]) is not int
            or not all(_matches(SHA256, binding.get(key)) for key in (
                "checkpoint_sha256", "packet_sha256", "mapping_sha256", "state_sha256",
                "base_messages_sha256", "prompt_sha256", "split_sha256",
            ))
            or not all(_matches(SHA256, scope.get(key)) for key in ("authorization_sha256", "contract_sha256"))
            or authorization_sha256 is not None and scope["authorization_sha256"] != authorization_sha256
            or contract_sha256 is not None and scope["contract_sha256"] != contract_sha256):
        raise ValueError("bridge job does not belong to the approved A7 contract, seed and round")
    if (type(messages) is not list or len(messages) != len(roles)
            or any(type(message) is not dict or set(message) != {"role", "content"}
                   or message["role"] != role or not isinstance(message["content"], str)
                   for message, role in zip(messages, roles, strict=True))
            or digest(messages[:2]) != binding["base_messages_sha256"]):
        raise ValueError("bridge request messages differ from the staged base prompt")
    packet = parse_response(messages[1]["content"])
    if type(packet) is not dict:
        raise ValueError("bridge packet differs from the staged A7/M2 input")
    prompt, packet_sha, compression = messages[0]["content"], binding["packet_sha256"], "lossless"
    if "wire_representation" in binding:
        profile = validate_wire_representation(binding["wire_representation"])
        if (not prompt.endswith(FORMAT_SUFFIX) or not _matches(SHA256, binding.get("wire_packet_sha256"))
                or not _matches(SHA256, binding.get("authoritative_base_messages_sha256"))
                or packet.get("wire_schema") != profile["schema"] or packet.get("source_compression") != "lossless"
                or type(packet.get("feature_precision")) is not int
                or packet["feature_precision"] != profile["feature_precision"] or packet.get("crossing_guard") is not True
                or type(packet.get("derived")) is not dict
                or any(set(fields) & set(WIDTHS) for fields in packet["derived"].values())):
            raise ValueError("bridge packet differs from the approved compact wire representation")
        prompt = prompt.removesuffix(FORMAT_SUFFIX)
        packet_sha, compression = binding["wire_packet_sha256"], profile["compression"]
    if (digest(prompt) != binding["prompt_sha256"] or digest(packet) != packet_sha
            or (packet.get("seed"), packet.get("round"), packet.get("layout"), packet.get("compression"))
            != (binding["seed"], binding["round"], "B", compression)
            or packet.get("selection_count") != 32 or type(packet.get("memory")) is not dict
            or packet["memory"].get("mode") != "M2"):
        raise ValueError("bridge packet differs from the staged A7/M2 input")


def make_job(request, config, started):
    """Bind the exact payload, checkpoint, authorization and reserved attempt."""
    check_request(request, request["binding"], config)
    _validate_started(request, config, started)
    _validate_binding(request, None, None)
    payload = {"version": JOB_VERSION, "attempt_id": started["attempt_id"],
               "request_id": request["request_id"], "request": request, "started": started}
    return {**payload, "job_sha256": digest(payload)}


def verify_job(job, *, authorization_sha256=None, contract_sha256=None, wire_representation=_UNSPECIFIED_WIRE):
    if type(job) is not dict or set(job) != JOB_FIELDS or job["version"] != JOB_VERSION:
        raise ValueError("unexpected bridge job envelope")
    request, started = job["request"], job["started"]
    config = TransportConfig(**request["configuration"])
    if job != make_job(request, config, started):
        raise ValueError("bridge job digest or identities drifted")
    _validate_binding(request, authorization_sha256, contract_sha256)
    if wire_representation is not _UNSPECIFIED_WIRE and request["binding"].get("wire_representation") != wire_representation:
        raise ValueError("bridge job differs from the contract's wire representation")
    return request, config, started


def _check_raw(value):
    """Reject credentials or HTTP header containers anywhere in raw response evidence."""
    if value is None:
        return
    if type(value) is not dict:
        raise ValueError("raw provider response must be a JSON object")
    pending = [value]
    while pending:
        obj = pending.pop()
        if isinstance(obj, dict):
            for key, child in obj.items():
                normalized = re.sub(r"[^a-z0-9]", "", key.lower())
                if normalized in SENSITIVE_FIELDS or normalized in ("headers", "requestheaders"):
                    raise ValueError("unexpected credential or HTTP header field in bridge evidence")
                pending.append(child)
        elif isinstance(obj, list):
            pending.extend(obj)


def make_result(job, *, reply=None, failure=None, credential_id=None, model_identity=None,
                provider_dispatched=None, elapsed_seconds=None, raw_response=None):
    """Serialize one provider outcome without accepting or editing a selection."""
    if (reply is None) == (failure is None):
        raise ValueError("one response or failure is required")
    if reply is not None:
        outcome = {"status": "completed", "reply": {name: getattr(reply, name) for name in REPLY_FIELDS}}
    else:
        outcome = {"status": "failed", "failure": {
            "code": failure.code, "retryable": failure.retryable, "retry_after": failure.retry_after,
            "ambiguous": failure.ambiguous, "reply_identity": getattr(failure, "reply_identity", None),
        }}
    record = {
        "version": RESULT_VERSION, "attempt_id": job["attempt_id"], "request_id": job["request_id"],
        "job_sha256": job["job_sha256"], "credential_id": credential_id,
        "provider_dispatched": reply is not None if provider_dispatched is None else provider_dispatched,
        "elapsed_seconds": elapsed_seconds, "model_identity": model_identity, "outcome": outcome,
        "raw_response": raw_response, "raw_response_sha256": digest(raw_response) if raw_response is not None else None,
    }
    result = {**record, "result_sha256": digest(record)}
    validate_result(result, job)
    return result


def _check_usage(usage):
    if type(usage) is not dict or any(type(value) is not int or value < 0 for value in usage.values()):
        raise ValueError("invalid bridge provider usage evidence")


def _optional_text(value):
    return value is None or type(value) is str and bool(value.strip())


def _check_model_identity(config, reply, identity):
    if type(identity) is not dict:
        raise ValueError("completed bridge result is missing model identity evidence")
    declared = {"model_version": identity.get("reported_model_version"),
                "reported_reasoning_effort": identity.get("reported_reasoning_effort")}
    try:
        recomputed = verify_reported_model(config, reply, declared)
    except TransportFailure as error:
        raise ValueError("completed bridge response differs from its approved provider model") from error
    if identity != recomputed:
        raise ValueError("bridge provider model evidence differs from the exact declared identity")


def _check_completed_raw(raw, reply, identity):
    if (raw.get("id"), raw.get("model"), raw.get("status")) != (
        reply["response_id"], reply["served_model"], reply["status"]
    ):
        raise ValueError("raw provider response identity differs from the extracted answer")
    if type(raw.get("output")) is not list:
        raise ValueError("raw provider response is missing its output list")
    text_parts = []
    for item in raw["output"]:
        if not isinstance(item, dict):
            raise ValueError("invalid raw provider output item")
        if item.get("type") != "message":
            continue
        if type(item.get("content")) is not list:
            raise ValueError("invalid raw provider message")
        for part in item["content"]:
            if not isinstance(part, dict):
                raise ValueError("invalid raw provider output content")
            if part.get("type") == "output_text":
                if not isinstance(part.get("text"), str):
                    raise ValueError("invalid raw provider text")
                text_parts.append(part["text"])
    if "".join(text_parts) != reply["text"]:
        raise ValueError("extracted provider text differs from raw provider output")
    usage = raw.get("usage")
    if usage is not None and (
        type(usage) is not dict
        or any(value is not None and reply["usage"].get(key) != value for key, value in usage.items()
               if key in {"input_tokens", "output_tokens", "total_tokens"})
    ):
        raise ValueError("extracted provider usage differs from raw provider output")
    if raw.get("model_version") is not None and raw["model_version"] != identity["reported_model_version"]:
        raise ValueError("reported bridge model version differs from raw provider output")


def validate_result(result, job):
    """Validate both completed and failed envelopes without dispatch or side effects."""
    request, config, started = verify_job(job)
    if (type(result) is not dict or set(result) != RESULT_FIELDS or result["version"] != RESULT_VERSION
            or (result["attempt_id"], result["request_id"], result["job_sha256"])
            != (started["attempt_id"], request["request_id"], job["job_sha256"])
            or result["result_sha256"] != digest({key: value for key, value in result.items() if key != "result_sha256"})
            or type(result["provider_dispatched"]) is not bool
            or result["credential_id"] is not None and not _matches(CREDENTIAL_ID, result["credential_id"])
            or result["provider_dispatched"] and result["credential_id"] is None and not request["binding"]["a7_formal"]["mock"]
            or result["elapsed_seconds"] is not None and not _duration(result["elapsed_seconds"])):
        raise ValueError("bridge response is not bound to this exact provider attempt")
    raw = result["raw_response"]
    _check_raw(raw)
    if result["raw_response_sha256"] != (digest(raw) if raw is not None else None):
        raise ValueError("raw provider response digest drifted")
    outcome = result["outcome"]
    if type(outcome) is not dict:
        raise ValueError("unexpected bridge outcome")
    if outcome.get("status") == "completed":
        reply = outcome.get("reply")
        if (set(outcome) != {"status", "reply"} or type(reply) is not dict or set(reply) != REPLY_FIELDS
                or not result["provider_dispatched"] or reply["status"] != "completed"
                or raw is None and not request["binding"]["a7_formal"]["mock"]
                or not isinstance(reply["text"], str) or not reply["text"].strip()
                or not isinstance(reply["response_id"], str) or not reply["response_id"].strip()
                or not _optional_text(reply["provider_request_id"])):
            raise ValueError("invalid completed provider response on bridge")
        _check_usage(reply["usage"])
        _check_model_identity(config, ProviderReply(**reply), result["model_identity"])
        if raw is not None:
            _check_completed_raw(raw, reply, result["model_identity"])
    elif outcome.get("status") == "failed":
        failure = outcome.get("failure")
        if (set(outcome) != {"status", "failure"} or type(failure) is not dict or set(failure) != FAILURE_FIELDS
                or result["model_identity"] is not None or not _matches(FAILURE_CODE, failure["code"])
                or type(failure["retryable"]) is not bool or type(failure["ambiguous"]) is not bool
                or failure["retryable"] and (failure["ambiguous"] or failure["code"] not in RETRYABLE_CODES)
                or failure["retry_after"] is not None and not _duration(failure["retry_after"])):
            raise ValueError("invalid classified provider failure on bridge")
        identity = failure["reply_identity"]
        if identity is not None:
            if (type(identity) is not dict or set(identity) != REPLY_IDENTITY_FIELDS
                    or not all(_optional_text(identity[field]) for field in REPLY_IDENTITY_FIELDS - {"usage"})):
                raise ValueError("invalid failed provider response identity evidence")
            _check_usage(identity["usage"])
    else:
        raise ValueError("unrecognized bridge outcome state")
    return result


def verify_result(result, job):
    """Return the original provider text or raise the original classified failure."""
    validate_result(result, job)
    evidence = {"bridge_result_sha256": result["result_sha256"], "bridge_job_sha256": result["job_sha256"],
                "provider_dispatched": result["provider_dispatched"],
                "provider_elapsed_seconds": result["elapsed_seconds"]}
    if result["credential_id"] is not None:
        evidence["credential_id"] = result["credential_id"]
    outcome = result["outcome"]
    if outcome["status"] == "completed":
        return ProviderReply(**outcome["reply"], transport_evidence=evidence)
    item = outcome["failure"]
    failure = TransportFailure(item["code"], retryable=item["retryable"], retry_after=item["retry_after"],
                               ambiguous=item["ambiguous"])
    failure.reply_identity = {**(item["reply_identity"] or {}), **evidence}
    raise failure


def private_directory(path):
    """Create or check a same-owner, non-symlink, owner-only spool directory."""
    path = Path(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
        raise PermissionError("bridge spool directory must be private and owned by the runner")
    return path


def read_bridge_json(path):
    """Read bounded owner-only bridge JSON without following symlinks."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
                or metadata.st_mode & 0o077 or metadata.st_size > MAX_ENVELOPE_BYTES):
            raise PermissionError("bridge envelope must be a bounded owner-only regular file")
        data = stream.read(MAX_ENVELOPE_BYTES + 1)
    if len(data) > MAX_ENVELOPE_BYTES:
        raise ValueError("bridge envelope exceeds the configured byte limit")
    return parse_response(data.decode())


class BridgeCourier:
    """GPU-side handoff. The Mac is the only provider caller."""

    def __init__(self, directory, *, authorization_sha256, contract_sha256, model_lock=None,
                 pause=None, wait_seconds=900, poll_seconds=0.25, wire_representation=None):
        if (not _matches(SHA256, authorization_sha256) or not _matches(SHA256, contract_sha256)
                or not _duration(wait_seconds) or not _duration(poll_seconds) or poll_seconds == 0):
            raise ValueError("bridge courier needs exact approval hashes and finite wait settings")
        self.directory = private_directory(directory)
        self.outbox = private_directory(self.directory / "outbox")
        self.results = private_directory(self.directory / "results")
        self.authorization_sha256 = authorization_sha256
        self.contract_sha256 = contract_sha256
        self.wire_representation = (None if wire_representation is None
                                    else dict(validate_wire_representation(wire_representation)))
        self.model_lock = Path(model_lock) if model_lock is not None else self.directory.parent / "provider_model_lock.json"
        self.pause = pause
        self.wait_seconds = wait_seconds
        self.poll_seconds = poll_seconds

    def __call__(self, request, config, started):
        job = make_job(request, config, started)
        verify_job(job, authorization_sha256=self.authorization_sha256, contract_sha256=self.contract_sha256,
                   wire_representation=self.wire_representation)
        name = f"{job['attempt_id']}.json"
        deadline = time.monotonic() + self.wait_seconds
        if self.pause is not None and self.pause():
            raise BridgePending("formal run paused before bridge dispatch")
        outbox_path = self.outbox / name
        if outbox_path.exists() and read_bridge_json(outbox_path) != job:
            raise ValueError("existing bridge job differs from its reserved attempt")
        once(outbox_path, job)
        while True:
            if self.pause is not None and self.pause():
                raise BridgePending("formal run paused while waiting for the existing Mac attempt")
            try:
                result = read_bridge_json(self.results / name)
            except FileNotFoundError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BridgePending("existing Mac attempt is pending without a verified terminal receipt") from None
                time.sleep(min(remaining, self.poll_seconds))
                continue
            reply = verify_result(result, job)
            try:
                once(self.model_lock, result["model_identity"])
            except ValueError:
                failure = TransportFailure("provider_identity_drift")
                failure.reply_identity = {
                    "response_id": reply.response_id, "served_model": reply.served_model, "usage": reply.usage,
                    "provider_request_id": reply.provider_request_id, "provider_status": reply.status,
                    **reply.transport_evidence,
                }
                raise failure from None
            return reply
