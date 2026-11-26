"""Mac-side transport dispatcher for GPU-authored A7 request attempts."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import math
import os
from pathlib import Path
import re
import stat
from tempfile import NamedTemporaryFile
import time
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    ProviderReply, TransportFailure, TransportStop,
)

from .bridge_protocol import _check_raw, make_result, private_directory, read_bridge_json, validate_result, verify_job
from .compact_payload import validate_wire_representation
from .formal_contract import OFFLINE, PROFILE, ROOT, STUDY, _profile, request_timeouts
from .key_pool import LEASE, KeyPool, load_pool
from .provider import NETWORK_PHASES, PROGRESS_MILESTONES, call_responses, recover_completed_response, sdk_status
from .ssh_courier import ATTEMPT_ID, GPU_PYTHON, SHA256, CourierUnavailable, SSHCourier, verify_info, verify_list

PROGRESS_FILENAME = "provider_progress.json"
PROGRESS_PHASES = {"request_dispatched", "http_phase", "http_headers", "sse_event", "transport_completed",
                   "transport_failure"}
PROGRESS_TOKENS = ("provider_request_id", "response_id", "conflicting_response_id", "model", "event_type", "error_type",
                   "provider_error_code", "provider_error_param", "provider_incomplete_reason")
PROGRESS_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
USAGE_FIELDS = ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "total_tokens")


class RemotePaused(RuntimeError):
    """The authenticated GPU coordinator currently disallows new attempts."""


def stamp():
    return datetime.now(timezone.utc).isoformat()


def _private_directory(path, *, create=False):
    if create:
        return private_directory(path)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise PermissionError("Mac bridge spool directories must be private and owned by the current user")


@contextmanager
def _exclusive(path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as handle:
        info = os.fstat(handle.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) & 0o077):
            raise PermissionError("unsafe bridge dispatcher lock")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("a bridge dispatcher already owns this local spool") from None
        yield


def _lease_dispatched(lease):
    with lease.pool.tx() as driver:
        record = driver.execute("SELECT dispatched FROM queue WHERE ticket=?", (lease.ticket,)).fetchone()
    if record is None:
        raise RuntimeError("the active credential lease is missing from the local ledger")
    return bool(record["dispatched"])


def _valid_elapsed(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _transport_progress(attempt, job):
    """Keep the latest phase and earlier identity milestones in a private spool record."""
    request = job["request"]
    binding = request["binding"]
    snapshot = {"attempt_id": attempt.name, "job_sha256": job["job_sha256"],
                "request_id": request["request_id"], "seed": binding["seed"], "round": binding["round"],
                "logical_index": request["logical_index"], "first_phase_seconds": {}}

    def save(event):
        phase = event.get("phase")
        elapsed = event.get("elapsed_seconds")
        if phase not in PROGRESS_PHASES or not _valid_elapsed(elapsed):
            return
        snapshot.update(phase=phase, elapsed_seconds=elapsed, updated_utc=stamp())
        snapshot["first_phase_seconds"].setdefault(phase, elapsed)
        activity = event.get("sse_activity_seconds")
        if _valid_elapsed(activity) and activity <= elapsed:
            snapshot["last_sse_activity_seconds"] = activity
        status = event.get("http_status")
        if type(status) is int and 100 <= status <= 599:
            snapshot["http_status"] = status
        for field in PROGRESS_TOKENS:
            token = event.get(field)
            if isinstance(token, str) and PROGRESS_TOKEN.fullmatch(token) and not token.lower().startswith("sk-"):
                snapshot[field] = token
        if phase == "http_phase":
            network_phase, state = event.get("network_phase"), event.get("network_state")
            duration = event.get("phase_duration_seconds")
            if network_phase in NETWORK_PHASES.values() and state in ("complete", "failed") and _valid_elapsed(duration):
                snapshot.setdefault("network_phases", {})[network_phase] = {
                    "state": state, "duration_seconds": duration, "observed_seconds": elapsed,
                }
            if type(event.get("explicit_proxy")) is bool:
                snapshot["explicit_proxy"] = event["explicit_proxy"]
        if phase == "sse_event":
            count = event.get("sse_event_count")
            if type(count) is int and count > 0:
                snapshot.setdefault("first_sse_seconds", elapsed)
                snapshot.update(last_sse_seconds=elapsed, sse_event_count=count)
                interval = event.get("sse_interval_seconds")
                if _valid_elapsed(interval):
                    snapshot["last_sse_interval_seconds"] = interval
                kind = event.get("event_type")
                if kind in PROGRESS_MILESTONES:
                    snapshot.setdefault("sse_milestones", {}).setdefault(kind, {
                        "elapsed_seconds": elapsed, "interval_seconds": interval if _valid_elapsed(interval) else None,
                    })
        if phase == "transport_completed":
            snapshot["complete_terminal_captured"] = True
            usage = event.get("usage")
            if type(usage) is dict:
                snapshot["usage"] = {field: usage[field] for field in USAGE_FIELDS
                                     if type(usage.get(field)) is int and usage[field] >= 0}
        with NamedTemporaryFile(dir=attempt, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write((dumps(snapshot, sort_keys=True) + "\n").encode())
                stream.flush()
                os.fsync(stream.fileno())
                os.replace(temporary, attempt / PROGRESS_FILENAME)
                dir_fd = os.open(attempt, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            finally:
                temporary.unlink(missing_ok=True)

    return save


def _recent_provider_progress(path, started, results, *, limit=6):
    observed = []
    for identity in started:
        attempt = path / "attempts" / identity
        try:
            _private_directory(attempt)
            progress = attempt / PROGRESS_FILENAME
            updated = progress.stat().st_mtime_ns
            observed.append((identity not in results, updated, identity, progress))
        except OSError:
            continue
    view = []
    for open_attempt, _, identity, path in sorted(observed, reverse=True)[:limit]:
        try:
            progress = read_bridge_json(path)
            marker = read_bridge_json(path.parent / "started.json")
        except (OSError, ValueError):
            continue
        if (type(progress) is dict and type(marker) is dict and progress.get("attempt_id") == identity
                and marker.get("attempt_id") == identity and progress.get("job_sha256") == marker.get("job_sha256")):
            view.append({**progress, "sealed": not open_attempt})
    return view


def verify_local_source(info, root=ROOT):
    """Bind paid execution to the same approved transport code as the GPFS endpoint."""
    needed = [STUDY / "a7_gpu" / name for name in (
        "bridge_protocol.py", "credentials.py", "dispatcher.py", "formal_contract.py",
        "key_pool.py", "provider.py", "ssh_courier.py",
    )]
    needed.append(OFFLINE / "pure_llm_v3" / "transport_manager.py")
    if info["contract"].get("profile", {}).get("wire_representation") is not None:
        needed.append(STUDY / "a7_gpu" / "compact_payload.py")
    approved = info["contract"].get("source_sha256", {})
    for path in needed:
        if approved.get(path.as_posix()) != file_digest(Path(root) / path):
            raise PermissionError(f"local paid transport source differs from the formal contract: {path.name}")


class Dispatcher:
    """One Mac spool per exact authenticated run. No scientific operations are available."""

    def __init__(self, spool, *, info, contract_sha256, transport, mock=True, pool=None, call=None,
                 sse_inactivity_seconds=None):
        self.path = Path(spool).expanduser().absolute()
        if type(mock) is not bool or (mock and pool is not None) or (not mock and call is not None):
            raise ValueError("mock transport and paid credential leasing must remain separate")
        self.info = verify_info(info, expected_contract=contract_sha256, expected_mock=mock)
        profile = self.info["contract"].get("profile", {})
        if profile.get("transport") != asdict(transport):
            raise PermissionError("remote formal transport differs from the reviewed local A7 profile")
        self.transport = transport
        self.timeouts = request_timeouts(profile)
        contracted_inactivity = profile.get("sse_inactivity_seconds")
        if contracted_inactivity is not None and sse_inactivity_seconds not in (None, contracted_inactivity):
            raise PermissionError("the dispatcher inactivity policy differs from the formal contract")
        if sse_inactivity_seconds is None:
            sse_inactivity_seconds = contracted_inactivity
        if sse_inactivity_seconds is not None and (
            self.timeouts is None or not _valid_elapsed(sse_inactivity_seconds)
            or not 0 < sse_inactivity_seconds < self.timeouts.total_seconds
        ):
            raise ValueError("SSE inactivity must be positive and shorter than the async request deadline")
        self.sse_inactivity_seconds = sse_inactivity_seconds
        representation = profile.get("wire_representation")
        self.wire_representation = None if representation is None else dict(validate_wire_representation(representation))
        self.mock = mock
        self.pool = pool
        self.call = call
        _private_directory(self.path, create=True)
        for folder in ("jobs", "slots", "attempts", "results", "delivered"):
            _private_directory(self.path / folder, create=True)
        once(self.path / "run_binding.json", {
            "contract_sha256": contract_sha256,
            "authorization_sha256": self.info["authorization_sha256"],
            "mock": mock,
        })
        with _exclusive(self.path / "identity.lock"):
            identity = self.path / "dispatcher_identity.json"
            if not identity.exists():
                once(identity, {"dispatcher_id": uuid4().hex})
            record = read_bridge_json(identity)
            if (type(record) is not dict or set(record) != {"dispatcher_id"}
                    or not isinstance(record["dispatcher_id"], str)
                    or not ATTEMPT_ID.fullmatch(record["dispatcher_id"])):
                raise ValueError("invalid durable Mac dispatcher identity")
            self.dispatcher_id = record["dispatcher_id"]

    def check_remote(self, info):
        return verify_info(info, expected_contract=self.info["contract_sha256"],
                           expected_auth=self.info["authorization_sha256"], expected_mock=self.mock)

    def validate(self, job):
        request, config, started = verify_job(
            job, contract_sha256=self.info["contract_sha256"],
            authorization_sha256=self.info["authorization_sha256"],
            wire_representation=self.wire_representation,
        )
        if config != self.transport or request["binding"]["a7_formal"]["mock"] is not self.mock:
            raise PermissionError("bridge attempt uses foreign transport or mock mode")
        return request, config, started

    def accept(self, job, *, attempt_id=None):
        request, _, started = self.validate(job)
        identity = started["attempt_id"]
        if not ATTEMPT_ID.fullmatch(identity) or attempt_id is not None and identity != attempt_id:
            raise ValueError("retrieved job does not match its remote attempt identifier")
        binding = request["binding"]
        slot = (f"{binding['seed']}-{binding['round']}-{request['logical_index']}-{started['attempt']}.json")
        once(self.path / "slots" / slot, {"attempt_id": identity, "job_sha256": job["job_sha256"]})
        once(self.path / "jobs" / f"{identity}.json", job)
        return identity

    def pending(self):
        jobs = []
        for path in (self.path / "jobs").glob("*.json"):
            identity = path.stem
            if not ATTEMPT_ID.fullmatch(identity):
                raise ValueError("invalid local job filename")
            if (self.path / "results" / path.name).exists():
                continue
            if (self.path / "attempts" / identity / "started.json").exists():
                continue
            jobs.append((path.stat().st_mtime_ns, identity))
        return [identity for _, identity in sorted(jobs)]

    def _job(self, attempt_id):
        if not ATTEMPT_ID.fullmatch(attempt_id):
            raise ValueError("invalid local attempt identifier")
        job = read_bridge_json(self.path / "jobs" / f"{attempt_id}.json")
        request, config, started = self.validate(job)
        if started["attempt_id"] != attempt_id:
            raise ValueError("local job filename differs from submitted attempt")
        return job, request, config

    def _model_lock(self, reply):
        if not self.mock:
            return read(self.path / "provider_model_lock.json")
        return {
            "provider": self.transport.provider, "requested_model": self.transport.requested_model,
            "served_model": reply.served_model, "reported_model_version": None,
            "requested_reasoning_effort": self.transport.reasoning_effort,
            "reported_reasoning_effort": None,
        }

    def dispatch(self, attempt_id, *, before_call, pause=None):
        """Consume at most one approved physical attempt. Unknown completion is never retried."""
        job, request, config = self._job(attempt_id)
        attempt = self.path / "attempts" / attempt_id
        result_path = self.path / "results" / f"{attempt_id}.json"
        _private_directory(attempt, create=True)
        with _exclusive(attempt / "claim.lock"):
            if result_path.exists():
                self.result(attempt_id)
                return "sealed"
            if (attempt / "started.json").exists():
                return "open_or_ambiguous"
            if not self.mock and self.pool is None:
                raise PermissionError("no initialized Mac credential pool for this authorized live run")
            if self.mock and self.call is None:
                raise ValueError("an explicit mock response source is required")
            slot = nullcontext(None) if self.mock else self.pool.attempt(request, config, pause=pause)
            try:
                with slot as credential_id:
                    # A slow wait for a credential must not outlive remote permission.
                    before_call(attempt_id, job)
                    lease = LEASE.get() if not self.mock else None
                    once(attempt / "started.json", {
                        "attempt_id": attempt_id,
                        "job_sha256": job["job_sha256"],
                        "credential_id": credential_id,
                        "credential_lease_ticket": lease.ticket if lease else None,
                        "started_utc": stamp(),
                    })
                    raw_path = attempt / "raw_provider_response.json"
                    capture_path = attempt / "provider_capture_receipt.json"

                    def capture_receipt(metadata):
                        _check_raw(metadata)
                        once(capture_path, {"attempt_id": attempt_id, "job_sha256": job["job_sha256"],
                                            "metadata": metadata})

                    def capture(raw):
                        if type(raw) is not dict:
                            raise ValueError("provider response capture expected a JSON object")
                        _check_raw(raw)
                        once(raw_path, raw)

                    start = time.monotonic()
                    reply = failure = model_identity = None
                    try:
                        if self.mock:
                            reply = self.call(request, config)
                        else:
                            inactivity = ({"sse_inactivity_seconds": self.sse_inactivity_seconds}
                                          if self.sse_inactivity_seconds is not None else {})
                            reply = call_responses(request, config,
                                                   model_lock=self.path / "provider_model_lock.json",
                                                   capture_response=capture, capture_receipt=capture_receipt,
                                                   progress=_transport_progress(attempt, job), timeouts=self.timeouts,
                                                   **inactivity)
                        model_identity = self._model_lock(reply)
                    except TransportFailure as error:
                        failure = error
                    raw = read(raw_path) if raw_path.exists() else None
                    if raw is not None and read_bridge_json(capture_path)["metadata"].get(
                        "raw_response_sha256"
                    ) != digest(raw):
                        raise ValueError("captured provider response checksum drifted")
                    if not self.mock and reply is not None and raw is None:
                        raise RuntimeError("completed provider attempt has no captured raw response")
                    result = make_result(job, reply=reply, failure=failure, credential_id=credential_id,
                                         model_identity=model_identity,
                                         provider_dispatched=True if self.mock else _lease_dispatched(lease),
                                         elapsed_seconds=time.monotonic() - start, raw_response=raw)
                    once(result_path, result)
            except TransportStop:
                if (attempt / "started.json").exists():
                    # The local marker is conservative once provider execution was possible.
                    raise
                if pause is not None and pause():
                    raise RemotePaused("remote bridge paused while waiting for a credential") from None
                # No provider request was dispatched. The GPU keeps the budget decision.
                before_call(attempt_id, job)
                failure = TransportFailure("dispatcher_credential_unavailable")
                once(result_path, make_result(job, failure=failure, provider_dispatched=False))
        return "sealed"

    def recover_captured(self, *, exclude=()):
        """Seal verified captured completions without a provider call or a new Key lease."""
        total = 0
        for path in sorted((self.path / "attempts").glob("*/raw_provider_response.json")):
            attempt = path.parent
            attempt_id = attempt.name
            if attempt_id in exclude:
                continue
            _private_directory(attempt)
            result_path = self.path / "results" / f"{attempt_id}.json"
            capture = attempt / "provider_capture_receipt.json"
            with _exclusive(attempt / "claim.lock"):
                if result_path.exists() or not capture.exists():
                    # Unbound old or partial capture evidence is not enough to infer completion.
                    continue
                job, _, config = self._job(attempt_id)
                started = read_bridge_json(attempt / "started.json")
                receipt = read_bridge_json(capture)
                expected = {"attempt_id": attempt_id, "job_sha256": job["job_sha256"]}
                if (set(receipt) != {*expected, "metadata"}
                        or any(receipt.get(key) != value or started.get(key) != value
                               for key, value in expected.items())):
                    raise ValueError("captured provider response belongs to another attempt")
                raw = read_bridge_json(path)
                _check_raw(raw)
                reply, identity = recover_completed_response(raw, config,
                    model_lock=self.path / "provider_model_lock.json", metadata=receipt["metadata"])
                result = make_result(job, reply=reply, credential_id=started["credential_id"],
                                     model_identity=identity, provider_dispatched=True, raw_response=raw)
                once(result_path, result)
                total += 1
        return total

    def result(self, attempt_id):
        job, _, _ = self._job(attempt_id)
        result = read_bridge_json(self.path / "results" / f"{attempt_id}.json")
        validate_result(result, job)
        return result

    def undelivered(self):
        return [path.stem for path in sorted((self.path / "results").glob("*.json"))
                if not (self.path / "delivered" / path.name).exists()]

    def mark_delivered(self, attempt_id, result, ack):
        if ack != {"attempt_id": attempt_id, "result_sha256": result["result_sha256"]}:
            raise ValueError("GPU did not acknowledge the exact sealed result")
        once(self.path / "delivered" / f"{attempt_id}.json", ack)


def spool_status(path):
    path = Path(path).expanduser().absolute()
    if not path.exists():
        return {"initialized": False}
    _private_directory(path)
    binding = path / "run_binding.json"
    started = {file.parent.name for file in (path / "attempts").glob("*/started.json")}
    results = {file.stem for file in (path / "results").glob("*.json")}
    return {"initialized": binding.is_file(), "run": read_bridge_json(binding) if binding.is_file() else None,
            "jobs": sum(1 for _ in (path / "jobs").glob("*.json")), "sealed_results": len(results),
            "delivered_results": sum(1 for _ in (path / "delivered").glob("*.json")),
            "open_or_ambiguous_count": len(started - results),
            "provider_progress": _recent_provider_progress(path, started, results),
            "key_pool": KeyPool(path / "key_pool").status(readonly=True)}


def pull(dispatcher, courier, info):
    inventory = verify_list(courier.list(), info)
    if info["paused"] or inventory["paused"]:
        return True, []
    for identity in inventory["attempt_ids"]:
        if not (dispatcher.path / "jobs" / f"{identity}.json").exists():
            dispatcher.accept(courier.get(identity), attempt_id=identity)
    return False, inventory["attempt_ids"]


def deliver(dispatcher, courier):
    for attempt_id in dispatcher.undelivered():
        result = dispatcher.result(attempt_id)
        dispatcher.mark_delivered(attempt_id, result, courier.put(attempt_id, dispatcher.dispatcher_id, result))


def run_dispatcher(dispatcher, courier, *, watch, stage_only=False, poll=2, workers=3):
    from threading import Event

    if type(workers) is not int or not 1 <= workers <= 3 or not 0.05 <= poll <= 60:
        raise ValueError("dispatcher concurrency or poll delay is outside the supported range")
    blocked = Event()
    blocked.set()

    def permission(attempt_id, job):
        if dispatcher.check_remote(courier.info())["paused"]:
            raise RemotePaused("remote bridge is paused")
        fresh = courier.claim(attempt_id, dispatcher.dispatcher_id)
        dispatcher.validate(fresh)
        if fresh != job:
            raise ValueError("the authenticated claim returned a different provider job")

    with _exclusive(dispatcher.path / "dispatcher.lock"):
        if stage_only:
            pull(dispatcher, courier, dispatcher.check_remote(courier.info()))
            return spool_status(dispatcher.path)
        if dispatcher.pool is not None:
            dispatcher.pool.prepare()
        running = {}
        totals = Counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            while True:
                for identity, future in list(running.items()):
                    if not future.done():
                        continue
                    del running[identity]
                    try:
                        totals[future.result()] += 1
                    except (RemotePaused, CourierUnavailable):
                        totals["deferred"] += 1
                try:
                    info = dispatcher.check_remote(courier.info())
                    totals["recovered_from_capture"] += dispatcher.recover_captured(exclude=running)
                    deliver(dispatcher, courier)
                    paused, outstanding = pull(dispatcher, courier, info)
                    if paused:
                        blocked.set()
                    else:
                        blocked.clear()
                        for identity in dispatcher.pending():
                            if len(running) >= workers:
                                break
                            if identity in outstanding and identity not in running:
                                running[identity] = executor.submit(
                                    dispatcher.dispatch, identity, before_call=permission, pause=blocked.is_set,
                                )
                except CourierUnavailable:
                    blocked.set()
                    if not watch:
                        raise
                    totals["ssh_exchanges_deferred"] += 1
                if not watch:
                    for future in running.values():
                        try:
                            totals[future.result()] += 1
                        except (RemotePaused, CourierUnavailable):
                            totals["deferred"] += 1
                    dispatcher.check_remote(courier.info())
                    deliver(dispatcher, courier)
                    return {**spool_status(dispatcher.path), "round": dict(totals)}
                time.sleep(poll)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "stage", "run"))
    parser.add_argument("--spool-dir", type=Path, required=True)
    parser.add_argument("--ssh-host", default="eit-hpc")
    parser.add_argument("--remote-run-dir")
    parser.add_argument("--remote-root")
    parser.add_argument("--remote-python", default=GPU_PYTHON)
    parser.add_argument("--contract-sha")
    parser.add_argument("--config", type=Path, default=ROOT / PROFILE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute-paid", action="store_true")
    mode.add_argument("--mock-reply", type=Path)
    parser.add_argument("--key-pool-config", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll", type=float, default=2)
    args = parser.parse_args()
    if args.action == "status":
        print(dumps(spool_status(args.spool_dir)))
        return
    if (not args.remote_run_dir or not args.remote_root or not args.contract_sha
            or not SHA256.fullmatch(args.contract_sha)):
        parser.error("an exact contract SHA, remote root and remote run directory are required")
    if args.action == "run" and not (args.execute_paid or args.mock_reply):
        parser.error("run needs explicit --execute-paid or a mock reply file")
    if args.watch and args.action != "run":
        parser.error("--watch is only supported by run")
    courier = SSHCourier(args.ssh_host, args.remote_root, args.remote_run_dir, args.remote_python)
    info = courier.info()
    # Stage may pull a live job without permission to issue a provider request.
    expected_mock = bool(args.mock_reply) if args.action == "run" else info.get("mock")
    verify_info(info, expected_contract=args.contract_sha, expected_mock=expected_mock)
    if args.action == "run" and args.execute_paid:
        verify_local_source(info)
    profile, config = _profile(args.config)
    remote_profile = info["contract"].get("profile", {})
    if request_timeouts(profile) != request_timeouts(remote_profile):
        raise PermissionError("remote and local A7 timeout policies differ")
    if profile.get("wire_representation") != remote_profile.get("wire_representation"):
        raise PermissionError("remote and local A7 wire representations differ")
    mock_reply = ProviderReply(**read(args.mock_reply)) if args.mock_reply else None
    dispatcher = Dispatcher(args.spool_dir, info=info, contract_sha256=args.contract_sha, transport=config,
                            mock=expected_mock, call=(lambda *_: mock_reply) if mock_reply else None)
    workers = 3
    if args.action == "run" and args.execute_paid:
        if not sdk_status(config)["matches"]:
            raise PermissionError("local OpenAI SDK version differs from the approved A7 profile")
        pool_config = load_pool(args.key_pool_config)
        dispatcher.pool = KeyPool(dispatcher.path / "key_pool", pool_config)
        workers = pool_config.parallelism
    result = run_dispatcher(dispatcher, courier, watch=args.watch, stage_only=args.action == "stage",
                            poll=args.poll, workers=workers)
    print(dumps(result))


if __name__ == "__main__":
    main()
