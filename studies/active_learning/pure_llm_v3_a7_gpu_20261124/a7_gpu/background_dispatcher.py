"""Dispatch explicitly qualified background jobs and retrieve accepted jobs independently of SSH."""

from contextlib import contextmanager
import time

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_BACKGROUND, TransportFailure, TransportStop,
)

from .background_transport import BackgroundBinding, BackgroundLedger, PENDING
from .bridge_protocol import make_result, read_bridge_json
from .credentials import key_text
from .dispatcher import Dispatcher, _exclusive, _private_directory, _transport_progress, deliver, pull
from .key_pool import LEASE, KeyPool
from .provider import recover_completed_response
from .ssh_courier import ATTEMPT_ID, CourierUnavailable


class _SelectivePool(KeyPool):
    """Apply credential affinity inside the existing locked global pool transaction."""

    def __init__(self, pool, *, allowed):
        super().__init__(pool.directory, pool.config, clock=pool.clock, poll=pool.poll)
        self.allowed = frozenset(allowed)

    def _grant(self, driver, ticket, max_wait):
        selected = super()._grant(driver, ticket, max_wait)
        if selected is None or selected in self.allowed:
            return selected
        now = self.clock()
        candidates = driver.execute("""SELECT keys.id FROM keys WHERE keys.configured=1 AND keys.present=1
            AND keys.disabled_error IS NULL AND keys.cooldown_until <= ?
            AND NOT EXISTS (SELECT 1 FROM queue WHERE state='busy' AND key_id=keys.id)
            ORDER BY COALESCE((SELECT MAX(started) FROM queue WHERE key_id=keys.id),0), keys.id""", (now,))
        replacement = next((row["id"] for row in candidates if row["id"] in self.allowed), None)
        if replacement:
            driver.execute("UPDATE queue SET key_id=? WHERE ticket=?", (replacement, ticket))
        else:
            driver.execute("UPDATE queue SET state='queued',key_id=NULL,started=NULL WHERE ticket=?", (ticket,))
        return replacement


class BackgroundDispatcher(Dispatcher):
    def __init__(self, *args, client_for_key, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mock or self.pool is None or self.transport.wire_mode != RESPONSES_BACKGROUND:
            raise PermissionError("background dispatch needs a separately authorized paid transport and key pool")
        if self.timeouts is None:
            raise PermissionError("background dispatch needs bound HTTP timeouts")
        self.client_for_key = client_for_key

    def _ledger(self, attempt, job, credential):
        request = job["request"]
        contract = self.info["authorization_sha256"]
        config = self.transport
        binding = BackgroundBinding(contract, request["request_id"], request["payload_sha256"], credential,
                                    config.requested_model, config.reasoning_effort, served_model=config.served_model)
        return BackgroundLedger(attempt / "background", binding=binding, payload=request["payload"])

    @contextmanager
    def _client(self, lease):
        if lease.path is None:
            raise PermissionError("background request needs a private credential file")
        with key_text(lease.path) as key:
            with self.client_for_key(key) as client:
                yield client

    def _active(self):
        active = []
        for marker in sorted((self.path / "attempts").glob("*/started.json")):
            identity = marker.parent.name
            if not ATTEMPT_ID.fullmatch(identity):
                raise ValueError("invalid background attempt directory")
            if (self.path / "results" / f"{identity}.json").is_file():
                continue
            record = read_bridge_json(marker)
            job, _, _ = self._job(identity)
            if record["attempt_id"] != identity or record["job_sha256"] != job["job_sha256"]:
                raise PermissionError("background attempt reservation belongs to another job")
            active.append((marker.parent, record, job))
        return active

    def _progress(self, attempt, job, record, state, *, usage=None):
        elapsed = max(0, time.time() - record["submitted_unix"])
        phase = "transport_completed" if state["state"] == "completed" else "http_headers"
        _transport_progress(attempt, job)({
            "phase": phase, "elapsed_seconds": elapsed,
            "response_id": state.get("response_id"), "event_type": f"background.{state['state']}",
            "model": state.get("model"), "error_type": state.get("last_error"),
            "http_status": state.get("last_http_status", state.get("http_status")), "usage": usage,
        })

    def _seal(self, attempt, job, record, ledger, state):
        phase = state["state"]
        if phase in PENDING:
            self._progress(attempt, job, record, state)
            return
        result_path = self.path / "results" / f"{attempt.name}.json"
        if phase == "completed":
            raw = ledger.completed()
            metadata = {"raw_response_sha256": digest(raw), "provider_request_id": state["provider_request_id"],
                        "model_version": raw.get("model_version"), "reported_reasoning_effort": state.get("reasoning_effort")}
            reply, identity = recover_completed_response(raw, self.transport,
                model_lock=self.path / "provider_model_lock.json", metadata=metadata)
            once(attempt / "provider_capture_receipt.json", {
                "attempt_id": attempt.name, "job_sha256": job["job_sha256"], "metadata": metadata,
            })
            once(attempt / "raw_provider_response.json", raw)
            self._progress(attempt, job, record, state, usage=reply.usage)
            result = make_result(job, reply=reply, credential_id=record["credential_id"], model_identity=identity,
                                 provider_dispatched=True, raw_response=raw)
        else:
            unknown = phase in {"planned", "submission_unknown", "submission_unverified"}
            if unknown and not (self.path / "submit_hold.json").exists():
                # An unacknowledged POST may still be running upstream. Freeze new submissions
                # while allowing retrieval and delivery of every other already accepted response.
                once(self.path / "submit_hold.json", {"attempt_id": attempt.name, "state": phase})
            self._progress(attempt, job, record, state)
            code = "background_submission_unknown" if unknown else f"background_{phase}"
            result = make_result(job, failure=TransportFailure(code, ambiguous=unknown),
                                 credential_id=record["credential_id"], provider_dispatched=state["post_reserved"])
        once(result_path, result)

    def _poll_accepted(self, attempt, record, job):
        alias = record["credential_id"]
        ledger = self._ledger(attempt, job, alias)
        state = ledger.status()
        if state["state"] in PENDING:
            pool = _SelectivePool(self.pool, allowed=[alias])
            expiry = time.monotonic() + 1
            request = job["request"]
            try:
                with pool.attempt(request, self.transport, pause=lambda: time.monotonic() >= expiry):
                    lease = LEASE.get()
                    with self._client(lease) as client:
                        state = ledger.poll(client, credential_alias=alias)
                    code = state.get("last_failure_code")
                    if code in {"http_401", "http_403", "http_429", "quota_exhausted"}:
                        lease.outcome(code, scope=state.get("last_rate_limit_scope") or "shared",
                                      retry_after=state.get("last_retry_after"))
            except TransportStop:
                return
        self._seal(attempt, job, record, ledger, state)

    def _submit(self, courier, identity, allowed):
        job, request, _ = self._job(identity)
        pool = _SelectivePool(self.pool, allowed=allowed)
        expiry = time.monotonic() + 1
        try:
            with pool.attempt(request, self.transport, pause=lambda: time.monotonic() >= expiry) as alias:
                check = self.check_remote(courier.info())
                if check["paused"]:
                    return
                claimed = courier.claim(identity, self.dispatcher_id)
                if self.validate(claimed)[0] != request or claimed != job:
                    raise PermissionError("claimed background job differs from the local archived job")
                attempt = self.path / "attempts" / identity
                _private_directory(attempt, create=True)
                record = {"attempt_id": identity, "job_sha256": job["job_sha256"],
                          "credential_id": alias, "submitted_unix": time.time()}
                lease = LEASE.get()
                with self._client(lease) as client:
                    ledger = self._ledger(attempt, job, alias)
                    once(attempt / "started.json", record)
                    state = ledger.submit(client, credential_alias=alias, allow_post=True,
                                          on_dispatch=lambda _: lease.dispatched())
                code = state.get("failure_code")
                if code in {"http_401", "http_403", "http_429", "quota_exhausted"}:
                    lease.outcome(code, scope=state.get("rate_limit_scope") or "shared",
                                  retry_after=state.get("retry_after"))
                self._seal(attempt, job, record, ledger, state)
        except TransportStop:
            return

    def watch(self, courier, *, poll_seconds=8):
        if not 3 <= poll_seconds <= 60:
            raise ValueError("invalid background polling interval")
        with _exclusive(self.path / "dispatcher.lock"):
            self.pool.prepare()
            while True:
                # GETs remain safe while SSH is unavailable or a remote pause blocks new POSTs.
                for attempt, record, job in self._active():
                    self._poll_accepted(attempt, record, job)
                try:
                    deliver(self, courier)
                    paused, outstanding = pull(self, courier, self.check_remote(courier.info()))
                    if not paused and not (self.path / "submit_hold.json").exists():
                        active = self._active()
                        if len(active) < self.pool.config.parallelism:
                            reserved = {record["credential_id"] for _, record, _ in active}
                            allowed = set(self.pool.config.keys) - reserved
                            identity = next((key for key in self.pending() if key in outstanding), None)
                            if allowed and identity is not None:
                                self._submit(courier, identity, allowed)
                except CourierUnavailable:
                    pass
                time.sleep(poll_seconds)
