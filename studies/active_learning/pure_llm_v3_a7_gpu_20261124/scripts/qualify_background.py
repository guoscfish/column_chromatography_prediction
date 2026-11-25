"""Qualify one synthetic Token4Research background submission and resume by response ID."""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import time
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportStop
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import _check_raw, read_bridge_json
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.credentials import key_text
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.dispatcher import _exclusive, _private_directory
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.key_pool import LEASE
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.provider import TOKEN, http_failure, sdk_status

from .dispatch_prompt_pilot import SharedKeyPool
from .dispatch_prompt_successor import PILOT_CONTRACT_SHA256, _pool

MODEL = "gpt-6.2-astra"
EFFORT = "xhigh"
BASE_URL = "https://token4research.cn/v1"
SDK_VERSION = "2.28.0"
RESPONSE_ID = re.compile(r"resp_[A-Za-z0-9_-]{1,123}")
PENDING = ("queued", "in_progress")
STATES = (*PENDING, "completed", "incomplete", "failed", "cancelled")
ANSWER = {"candidate_id": "SYN-A", "difference": 3}
PAYLOAD = {
    "model": MODEL,
    "reasoning": {"effort": EFFORT},
    "input": [
        {"role": "system", "content": "Answer this synthetic exercise with one JSON object only."},
        {"role": "user", "content": (
            "The invented table has SYN-A score 8 and SYN-B score 5. Select the higher score. "
            "Return keys candidate_id (text) and difference (integer, higher minus lower)."
        )},
    ],
    "max_output_tokens": 4096,
    "background": True,
    "store": False,
    "stream": False,
}


class PinnedPollPool(SharedKeyPool):
    """Use the original credential for retrieval without taking a second pool or bypassing FIFO."""

    def __init__(self, pool, credential_id):
        super().__init__(pool.directory, pool.config, clock=pool.clock, poll=pool.poll)
        self.credential_id = credential_id

    def _grant(self, driver, ticket, max_wait):
        intended = driver.execute("SELECT * FROM keys WHERE id=?", (self.credential_id,)).fetchone()
        if (intended is None or not intended["configured"] or not intended["present"]
                or intended["disabled_error"] is not None or intended["cooldown_until"] - self.clock() > max_wait):
            raise TransportStop("original credential unavailable for retrieval")
        selected = super()._grant(driver, ticket, max_wait)
        if selected is None or selected == self.credential_id:
            return selected
        busy = driver.execute("SELECT 1 FROM queue WHERE state='busy' AND key_id=?",
                              (self.credential_id,)).fetchone()
        # This is inside the same pool transaction as the ordinary grant. A provisional grant
        # can switch to the exact available key without increasing global or key occupancy.
        if busy is None and intended["cooldown_until"] <= self.clock():
            driver.execute("UPDATE queue SET key_id=? WHERE ticket=?", (self.credential_id, ticket))
            return self.credential_id
        driver.execute("UPDATE queue SET state='queued',key_id=NULL,started=NULL WHERE ticket=?", (ticket,))
        return None


def contract(ledger):
    return {"schema": "a7_synthetic_background_qualification_v1", "provider": "token4research",
            "base_url": BASE_URL, "sdk_version": SDK_VERSION, "ledger": str(ledger),
            "launcher_sha256": file_digest(Path(__file__)), "payload": PAYLOAD,
            "payload_sha256": digest(PAYLOAD), "maximum_paid_posts": 1,
            "poll_same_credential": True, "research_data": False}


def build_plan(ledger):
    record = contract(ledger)
    return {"contract": record, "contract_sha256": digest(record)}


def plan(ledger):
    expected = build_plan(ledger)
    once(ledger / "plan.json", expected)
    return expected


def read_plan(ledger):
    pinned = read_bridge_json(ledger / "plan.json")
    if pinned != build_plan(ledger):
        raise PermissionError("qualification plan does not match this ledger and launcher")
    return pinned


def _token(value, key):
    return value if isinstance(value, str) and TOKEN.fullmatch(value) and key not in value else None


def summarize_response(raw, provider_request_id, key, *, expected_id=None):
    _check_raw(raw)
    if key in dumps(raw):
        raise ValueError("sensitive value found in provider response")
    identity, status = raw.get("id"), raw.get("status")
    if (not isinstance(identity, str) or not RESPONSE_ID.fullmatch(identity)
            or expected_id is not None and identity != expected_id or status not in STATES):
        raise ValueError("missing or inconsistent background response identity")
    reasoning = raw.get("reasoning")
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    model = raw.get("model")
    if model != MODEL or effort not in (None, EFFORT):
        raise ValueError("provider response differs from the registered model or effort")
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    summary = {"response_id": identity, "status": status, "model": model, "reasoning_effort": effort,
               "background": raw.get("background"), "store": raw.get("store"),
               "provider_request_id": _token(provider_request_id, key),
               "usage": {name: usage[name] for name in ("input_tokens", "output_tokens", "total_tokens")
                         if type(usage.get(name)) is int and usage[name] >= 0}}
    for name, field in (("input_tokens_details", "cached_tokens"), ("output_tokens_details", "reasoning_tokens")):
        details = usage.get(name)
        if isinstance(details, dict) and type(details.get(field)) is int and details[field] >= 0:
            summary["usage"][field] = details[field]
    if status == "completed":
        parts = [block["text"] for item in raw.get("output") or []
                 if item.get("type") == "message" for block in item.get("content") or []
                 if block.get("type") == "output_text" and isinstance(block.get("text"), str)]
        try:
            answer = json.loads("".join(parts))
        except ValueError:
            answer = None
        summary["synthetic_answer_ok"] = (type(answer) is dict and answer == ANSWER
                                          and type(answer["difference"]) is int)
    return summary


def _error_receipt(error, key, sdk):
    if isinstance(error, sdk.APIStatusError):
        failure, scope = http_failure(error.status_code, error.response.headers, error.body)
        details = error.body if isinstance(error.body, dict) else {}
        details = details.get("error", details)
        details = details if isinstance(details, dict) else {}
        safe = {"http_status": error.status_code, "error": failure.code,
                "provider_code": _token(details.get("code"), key), "provider_param": _token(details.get("param"), key),
                "provider_request_id": _token(error.response.headers.get("x-request-id"), key)}
        # Preserve real quota/auth/429 handling. A canary-only model or background rejection
        # cannot disable the production pool's otherwise separate synchronous contract.
        shared_codes = {"http_401", "http_429", "quota_exhausted"}
        return safe, (failure.code if failure.code in shared_codes else "qualification_" + failure.code), scope, failure.retry_after
    return {"error": type(error).__name__}, "qualification_" + type(error).__name__, "shared", None


@contextmanager
def _client(key, limit):
    import httpx
    import openai

    timeout = httpx.Timeout(limit, connect=min(limit, 15), write=min(limit, 30), pool=min(limit, 15))
    with httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False) as http:
        with openai.OpenAI(api_key=key, base_url=BASE_URL, timeout=timeout,
                           http_client=http, max_retries=0) as client:
            yield client, openai


def _shared_pool(pilot_spool):
    pool, _, config = _pool(pilot_spool)
    if (config.base_url != BASE_URL or config.provider != "token4research"
            or config.sdk_version != SDK_VERSION or not sdk_status(config)["matches"]):
        raise PermissionError("original credential pool or installed SDK differs from the canary scope")
    shared = SharedKeyPool(pool.directory, pool.config)
    shared.prepare()
    return shared, replace(config, max_delay_seconds=min(config.max_delay_seconds, 30))


def _run_http(pool, config, pinned, action, publish, *, response_id=None):
    request = {"binding": {"seed": -1, "round": -1}, "logical_index": 0,
               "request_id": pinned["contract_sha256"]}
    deadline = time.monotonic() + (5 if response_id else 60)
    with pool.attempt(request, config, pause=lambda: time.monotonic() >= deadline) as credential_id:
        lease = LEASE.get()
        if lease.path is None:
            raise PermissionError("canary needs the private configured credential file")
        with key_text(lease.path) as key:
            with _client(key, 30 if response_id else 90) as (client, sdk):
                action(credential_id)
                lease.dispatched()
                try:
                    responses = client.responses.with_raw_response
                    network = (responses.retrieve(response_id) if response_id else responses.create(
                        **PAYLOAD, extra_headers={"X-Client-Request-Id": pinned["contract_sha256"]}))
                    raw = network.parse().model_dump(mode="json", exclude_unset=True)
                    summary = summarize_response(raw, network.headers.get("x-request-id"), key, expected_id=response_id)
                except (sdk.OpenAIError, ValueError) as error:
                    result, kind, scope, retry_after = _error_receipt(error, key, sdk)
                    publish(result)
                    lease.outcome(kind, scope=scope, retry_after=retry_after)
                    return result
                publish({"summary": summary, "raw_response": raw})
                completed = summary["status"] == "completed"
                lease.outcome("completed", usage=summary["usage"] if completed else None, response_complete=completed)
                return summary


def submit(ledger, pool, config, *, paid):
    if not paid:
        raise PermissionError("submit needs explicit --authorize-one-paid-post")
    pinned = read_plan(ledger)
    with _exclusive(ledger / "operation.lock"):
        started = ledger / "submission_started.json"
        if started.exists():
            return status(ledger)
        # The marker precedes all possible wire activity and is never cleared on an unknown outcome.
        def start(credential_id):
            once(started, {"contract_sha256": pinned["contract_sha256"], "credential_id": credential_id,
                           "authorized_one_paid_post": True, "time_unix": time.time()})

        _run_http(pool, config, pinned, start, lambda receipt: once(ledger / "submission.json", receipt))
        return status(ledger)


def _load_optional(path):
    return read_bridge_json(path) if path.is_file() else None


def status(ledger):
    pinned = read_plan(ledger)
    started = _load_optional(ledger / "submission_started.json")
    submission = _load_optional(ledger / "submission.json")
    terminal = _load_optional(ledger / "retrieval_terminal.json")
    latest = max((ledger / "polls").glob("*.json"), key=lambda p: p.name, default=None)
    pending = _load_optional(latest) if latest else None
    details = terminal or pending or submission
    phase = ("retrieval_terminal" if terminal is not None else "planned" if started is None
             else "submission_unknown" if submission is None else
             "retrieval_pending" if "summary" in submission else "submission_failed")
    summary = {"contract_sha256": pinned["contract_sha256"], "paid_post_reserved": started is not None,
               "credential_id": started["credential_id"] if started else None, "phase": phase,
               "submission": submission.get("summary", submission) if submission else None,
               "result": details.get("summary", details) if details else None,
               "same_id_retrieval_completed": terminal is not None and terminal["summary"]["status"] == "completed"}
    return summary


def poll(ledger, pool, config):
    pinned = read_plan(ledger)
    with _exclusive(ledger / "operation.lock"):
        submission = _load_optional(ledger / "submission.json")
        started = _load_optional(ledger / "submission_started.json")
        if (_load_optional(ledger / "retrieval_terminal.json") is not None
                or submission is None or started is None or "summary" not in submission):
            return status(ledger)
        submitted = submission["summary"]
        if not RESPONSE_ID.fullmatch(submitted["response_id"]):
            raise PermissionError("retrieval needs the exact submitted response ID")
        if started["contract_sha256"] != pinned["contract_sha256"]:
            raise PermissionError("retrieval custody differs from the original submission")
        pinned_pool = PinnedPollPool(pool, started["credential_id"])

        def publish(receipt):
            summary = receipt.get("summary")
            path = (ledger / "retrieval_terminal.json" if summary and summary["status"] not in PENDING
                    else ledger / "polls" / f"{time.time_ns():020d}-{uuid4().hex}.json")
            once(path, receipt)

        _run_http(pinned_pool, config, pinned, lambda _: None, publish, response_id=submitted["response_id"])
        return status(ledger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "submit", "poll", "status"))
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--pilot-spool", type=Path, default=Path.home() / ".config/purellm/spool" / PILOT_CONTRACT_SHA256)
    parser.add_argument("--authorize-one-paid-post", action="store_true")
    args = parser.parse_args()
    if (args.action == "submit") != args.authorize_one_paid_post:
        parser.error("only submit requires --authorize-one-paid-post")
    ledger = args.ledger.expanduser().absolute()
    os.umask(0o077)
    _private_directory(ledger, create=args.action == "plan")
    if args.action == "plan":
        print(dumps(plan(ledger), sort_keys=True))
    elif args.action == "status":
        print(dumps(status(ledger), sort_keys=True))
    else:
        pool, config = _shared_pool(args.pilot_spool.expanduser())
        output = submit(ledger, pool, config, paid=True) if args.action == "submit" else poll(ledger, pool, config)
        print(dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
