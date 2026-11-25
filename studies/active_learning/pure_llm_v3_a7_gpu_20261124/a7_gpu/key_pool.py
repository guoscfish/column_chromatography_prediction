"""Cross-process FIFO API dispatch with per-key POSIX leases and a shared limit."""
from contextlib import closing, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from uuid import uuid4

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import TransportStop

from .credentials import credential_status, private_file_status, private_text

POOL_ENV = "PURE_LLM_KEY_POOL_CONFIG"
LEASE = ContextVar("purellm_credential_lease", default=None)
GLOBAL_STOPS = {
    "provider_context_rejected", "provider_model_rejected", "provider_reasoning_rejected",
    "served_model_mismatch", "served_reasoning_effort_mismatch", "provider_identity_drift",
    "invalid_reported_model_version",
}
KEY_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,31}")


@dataclass(frozen=True)
class PoolConfig:
    parallelism: int
    keys: dict[str, Path | None]

    def public(self):
        return {"max_concurrent_requests": self.parallelism, "keys": sorted(self.keys)}

    def status(self):
        return {key: credential_status() if path is None else private_file_status(path)
                for key, path in self.keys.items()}


def pool_path(path=None):
    return Path(path or os.environ.get(POOL_ENV) or Path.home() / ".config/purellm/key_pool.json").expanduser()


def load_pool(path=None, *, single=False, require_usable=True):
    if single:
        if path is not None:
            raise ValueError("single credential compatibility mode conflicts with an explicit pool")
        pool = PoolConfig(1, {"single": None})
    else:
        path = pool_path(path)
        with private_text(path) as contents:
            config = json.loads(contents)
        if type(config) is not dict or set(config) != {"max_concurrent_requests", "keys"}:
            raise ValueError("credential pool configuration has unrecognized fields")
        cap = config["max_concurrent_requests"]
        if type(cap) is not int or not 1 <= cap <= 3 or not isinstance(config["keys"], list) or not config["keys"]:
            raise ValueError("credential pool needs at least one key and a shared cap of one to three")
        keys = {}
        identities = set()
        paths = set()
        for item in config["keys"]:
            if (type(item) is not dict or set(item) != {"id", "path", "max_concurrent_requests"}
                    or not isinstance(item["id"], str) or not KEY_ID.fullmatch(item["id"])
                    or item["id"] in keys or type(item["max_concurrent_requests"]) is not int
                    or item["max_concurrent_requests"] != 1 or not isinstance(item["path"], str)
                    or not item["path"].strip()):
                raise ValueError("credential entries require unique IDs, private files and one request per key")
            declared = Path(item["path"]).expanduser()
            resolved = Path(os.path.abspath(path.parent / declared))
            if resolved in paths:
                raise ValueError("credential paths must be distinct")
            paths.add(resolved)
            status = private_file_status(resolved)
            if status["permissions_ok"]:
                info = resolved.lstat()
                inode = info.st_dev, info.st_ino
                if inode in identities:
                    raise ValueError("credential entries cannot refer to the same file")
                identities.add(inode)
            keys[item["id"]] = resolved
        pool = PoolConfig(min(cap, len(keys)), keys)
    if require_usable and not any(
        status["present"] and status["permissions_ok"] is not False for status in pool.status().values()
    ):
        raise PermissionError("no configured provider credential is present and private")
    return pool


@dataclass
class Lease:
    pool: "KeyPool"
    ticket: str
    credential_id: str
    path: Path | None

    def dispatched(self):
        self.pool.record_dispatch(self.ticket)

    def outcome(self, kind, *, scope="shared", retry_after=None, usage=None, response_complete=False):
        self.pool.record_outcome(self.ticket, kind, scope=scope, retry_after=retry_after, usage=usage,
                                 response_complete=response_complete)


class KeyPool:
    """Persist queue and evidence in SQLite; release occupancy automatically on process death."""

    def __init__(self, directory, config=None, *, clock=time.time, poll=0.1):
        self.directory = Path(directory)
        self.database = self.directory / "key_pool.sqlite3"
        self.tickets = self.directory / "tickets"
        self.config = config
        self.clock = clock
        self.poll = poll

    @contextmanager
    def tx(self):
        driver = sqlite3.connect(self.database, timeout=30, isolation_level=None)
        try:
            driver.row_factory = sqlite3.Row
            driver.execute("PRAGMA busy_timeout=30000")
            driver.execute("BEGIN IMMEDIATE")
            try:
                yield driver
            except BaseException:
                driver.rollback()
                raise
            else:
                driver.commit()
        finally:
            driver.close()

    def prepare(self):
        """Coordinator calls this while no other coordinator owns the run."""
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.tickets.mkdir(exist_ok=True, mode=0o700)
        with self.tx() as driver:
            driver.execute("""CREATE TABLE IF NOT EXISTS settings
                (one INTEGER PRIMARY KEY, cap INTEGER NOT NULL, shared_until REAL NOT NULL DEFAULT 0,
                 shared_error TEXT, shared_disabled INTEGER NOT NULL DEFAULT 0)""")
            driver.execute("""CREATE TABLE IF NOT EXISTS keys
                (id TEXT PRIMARY KEY, configured INTEGER NOT NULL, present INTEGER NOT NULL,
                 cooldown_until REAL NOT NULL DEFAULT 0, disabled_error TEXT, recent_error TEXT)""")
            driver.execute("""CREATE TABLE IF NOT EXISTS queue
                (position INTEGER PRIMARY KEY, ticket TEXT UNIQUE, seed INTEGER, round INTEGER,
                 logical_index INTEGER, request_id TEXT, state TEXT, key_id TEXT, queued REAL,
                 started REAL, ended REAL, dispatched INTEGER DEFAULT 0,
                 response_complete INTEGER DEFAULT 0, outcome TEXT, usage TEXT)""")
            self._prune(driver)
            if driver.execute("SELECT 1 FROM queue WHERE state IN ('queued','busy') LIMIT 1").fetchone():
                raise TransportStop("active credential leases prevent pool reconfiguration")
            driver.execute("""INSERT INTO settings(one,cap) VALUES(1,?)
                ON CONFLICT(one) DO UPDATE SET cap=excluded.cap""", (self.config.parallelism,))
            driver.execute("UPDATE keys SET configured=0, present=0")
            for key, status in self.config.status().items():
                present = status["present"] and status["permissions_ok"] is not False
                driver.execute("""INSERT INTO keys(id,configured,present) VALUES(?,1,?)
                    ON CONFLICT(id) DO UPDATE SET configured=1,present=excluded.present""", (key, present))

    def _prune(self, driver):
        for row in driver.execute("SELECT ticket FROM queue WHERE state IN ('queued','busy')").fetchall():
            # A live worker owns its ticket flock from before enqueue until receipt commit.
            path = self.tickets / row["ticket"]
            with path.open("a") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                driver.execute("""UPDATE queue SET state='abandoned',ended=?,outcome=COALESCE(outcome,
                    CASE WHEN dispatched THEN 'ambiguous_worker_exit' ELSE 'worker_exit_before_dispatch' END)
                    WHERE ticket=?""", (self.clock(), row["ticket"]))

    def _grant(self, driver, ticket, max_wait):
        self._prune(driver)
        setting = driver.execute("SELECT * FROM settings WHERE one=1").fetchone()
        now = self.clock()
        if setting["shared_disabled"]:
            raise TransportStop("provider shared scope disabled")
        if setting["shared_until"] - now > max_wait:
            raise TransportStop("shared provider cooldown exceeds configured wait")
        if setting["shared_until"] > now:
            return None
        rows = driver.execute("""SELECT keys.* FROM keys WHERE configured=1 AND present=1
            AND disabled_error IS NULL ORDER BY COALESCE(
                (SELECT MAX(queue.started) FROM queue WHERE queue.key_id=keys.id),0), id""").fetchall()
        if not rows:
            raise TransportStop("all provider credentials are unavailable")
        if all(row["cooldown_until"] - now > max_wait for row in rows):
            raise TransportStop("all provider credentials are in extended cooldown")
        waiting = driver.execute("SELECT ticket FROM queue WHERE state='queued' ORDER BY position LIMIT 1").fetchone()
        busy = {row["key_id"] for row in driver.execute("SELECT key_id FROM queue WHERE state='busy'")}
        if waiting["ticket"] != ticket or len(busy) >= setting["cap"]:
            return None
        for row in rows:
            if row["id"] not in busy and row["cooldown_until"] <= now:
                driver.execute("UPDATE queue SET state='busy',key_id=?,started=? WHERE ticket=?",
                               (row["id"], now, ticket))
                return row["id"]
        return None

    @contextmanager
    def attempt(self, request, transport, *, pause=None):
        """Take and release one lease for one physical attempt, including corrections."""
        ticket = uuid4().hex
        path = self.tickets / ticket
        binding = request["binding"]
        with path.open("x") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                with self.tx() as driver:
                    driver.execute("""INSERT INTO queue(ticket,seed,round,logical_index,request_id,state,queued)
                        VALUES(?,?,?,?,?,'queued',?)""",
                        (ticket, binding["seed"], binding["round"], request["logical_index"],
                         request["request_id"], self.clock()))
                while True:
                    if pause and pause():
                        raise TransportStop("formal run paused while waiting for provider capacity")
                    with self.tx() as driver:
                        key_id = self._grant(driver, ticket, transport.max_delay_seconds)
                    if key_id is not None:
                        break
                    time.sleep(self.poll)
                lease = Lease(self, ticket, key_id, self.config.keys[key_id])
                token = LEASE.set(lease)
                try:
                    yield key_id
                finally:
                    LEASE.reset(token)
            finally:
                with self.tx() as driver:
                    driver.execute("""UPDATE queue SET state='ended',ended=?,outcome=COALESCE(outcome,
                        CASE WHEN dispatched THEN 'ambiguous_worker_exit' ELSE 'not_dispatched' END)
                        WHERE ticket=?""", (self.clock(), ticket))
                path.unlink(missing_ok=True)

    def record_dispatch(self, ticket):
        with self.tx() as driver:
            if driver.execute("""UPDATE queue SET dispatched=1 WHERE ticket=? AND state='busy'
                              AND dispatched=0""", (ticket,)).rowcount != 1:
                raise TransportStop("provider lease has already sent its one physical request")

    def record_outcome(self, ticket, kind, *, scope, retry_after, usage, response_complete):
        with self.tx() as driver:
            row = driver.execute("SELECT key_id,dispatched,outcome FROM queue WHERE ticket=? AND state='busy'",
                                 (ticket,)).fetchone()
            if row is None or row["outcome"] is not None:
                raise TransportStop("provider lease outcome is not open")
            driver.execute("UPDATE queue SET outcome=?,usage=?,response_complete=? WHERE ticket=?",
                           (kind, json.dumps(usage) if usage is not None else None, response_complete, ticket))
            if kind != "completed":
                driver.execute("UPDATE keys SET recent_error=? WHERE id=?", (kind, row["key_id"]))
            if kind == "http_429":
                until = self.clock() + max(retry_after or 0, 4)
                if scope == "credential":
                    driver.execute("UPDATE keys SET cooldown_until=max(cooldown_until,?) WHERE id=?",
                                   (until, row["key_id"]))
                else:
                    driver.execute("UPDATE settings SET shared_until=max(shared_until,?),shared_error=? WHERE one=1",
                                   (until, kind))
            elif kind in GLOBAL_STOPS or kind == "quota_exhausted" and scope != "credential":
                driver.execute("UPDATE settings SET shared_disabled=1,shared_error=? WHERE one=1", (kind,))
            elif kind in ("http_401", "http_403", "credential_file_error", "quota_exhausted"):
                driver.execute("UPDATE keys SET disabled_error=? WHERE id=?", (kind, row["key_id"]))

    @contextmanager
    def _status_connection(self, readonly):
        if readonly:
            uri = self.database.absolute().as_uri() + "?mode=ro"
            with closing(sqlite3.connect(uri, uri=True, timeout=30)) as driver:
                driver.row_factory = sqlite3.Row
                driver.execute("BEGIN")
                yield driver
            return
        with self.tx() as driver:
            self._prune(driver)
            yield driver

    def status(self, *, readonly=False):
        if not self.database.exists():
            return {"initialized": False}
        with self._status_connection(readonly) as driver:
            if readonly and driver.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='settings'"
            ).fetchone() is None:
                return {"initialized": False}
            limits = dict(driver.execute("SELECT * FROM settings WHERE one=1").fetchone())
            rows = [dict(r) for r in driver.execute("SELECT * FROM queue ORDER BY position")]
            busy = {row["key_id"] for row in rows if row["state"] == "busy"}
            keys = [dict(r) for r in driver.execute("SELECT * FROM keys WHERE configured=1 ORDER BY id")]
        now = self.clock()
        for key in keys:
            attempts = [row for row in rows if row["key_id"] == key["id"] and row["dispatched"]]
            states = {"disabled": bool(key["disabled_error"] or not key["present"]), "busy": key["id"] in busy,
                      "cooldown": key["cooldown_until"] > now}
            key["status"] = next((state for state, enabled in states.items() if enabled), "available")
            key["requests"] = len(attempts)
            key["completed"] = sum(row["response_complete"] for row in attempts)
            key["failures"] = sum(row["outcome"] not in (None, "completed") for row in attempts)
            key["corrective_requests"] = sum(row["logical_index"] > 0 for row in attempts)
            key["usage"] = {}
            for attempt in attempts:
                for name, count in json.loads(attempt["usage"] or "{}").items():
                    key["usage"][name] = key["usage"].get(name, 0) + count
        queue_fields = ("seed", "round", "logical_index", "state", "key_id", "queued")
        return {"initialized": True, "limits": limits, "keys": keys,
                "queue": [{k: row[k] for k in queue_fields} for row in rows if row["state"] in ("queued", "busy")]}
