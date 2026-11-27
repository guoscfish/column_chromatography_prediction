"""One signed W67 POST using the latched shared Key Pool."""
from contextlib import contextmanager
from threading import Lock

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_STREAM_IPV4,
    TransportConfig,
    TransportFailure,
    TransportStop,
    check_request,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.ssh_courier import SHA256

from .dispatch_prompt_pilot import SharedKeyPool
from .dispatch_v13_157_w66_pool import _W66GrantView

PAYLOAD_SHA256 = "a107c416d5d99771b3b14a88ee77b81c73378807da79cc849c3da05edae9e1e3"
EXPIRES_UNIX = 1795783500
LATCH = "served_reasoning_effort_mismatch"
EXPECTED_TRANSPORT = ("token4research", "https://token4research.cn/v1", "gpt-6.2-astra", "gpt-6.2-astra",
                      "xhigh", RESPONSES_STREAM_IPV4, "2.28.0")
SETTINGS_QUERY = "SELECT * FROM settings WHERE one=1"


W66_CONTRACT = "21efdac44631d67e886559c873cde4706a292c647f9d3ae808ac08a6075174f0"
W66_REQUEST = "f8eb40a3e8681b7fa88f9e5e9610d1505a6da0880b2209daa97c4391d992c674"
W66_TICKET = "ce06590e1770477ebb89574eee0c32c1"
W66_CONSUMED = 1795775558.959481


def _require_w66_receipt(driver):
    if driver.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='w66_operator_post'").fetchone() is None:
        raise TransportStop("the W67 permit requires the preserved W66 POST journal")
    prior = driver.execute("SELECT * FROM w66_operator_post WHERE one=1").fetchone()
    expected_prior = (W66_CONTRACT, W66_REQUEST, PAYLOAD_SHA256, W66_TICKET, W66_CONSUMED)
    if prior is None or tuple(prior[key] for key in (
        "contract_sha256", "request_id", "payload_sha256", "ticket", "consumed_unix",
    )) != expected_prior:
        raise TransportStop("the exact consumed W66 permit is missing or changed")
    row = driver.execute("SELECT * FROM queue WHERE ticket=?", (W66_TICKET,)).fetchone()
    expected = (157, 7, 0, W66_REQUEST, "ended", 1, 0, "response_id_mismatch", None)
    if row is None or tuple(row[key] for key in (
        "seed", "round", "logical_index", "request_id", "state", "dispatched",
        "response_complete", "outcome", "usage",
    )) != expected:
        raise TransportStop("the exact prior W66 dispatched and failed queue record changed")


def verify_w66_receipt(pool):
    """Read the completed W66 bookkeeping without changing the shared database."""
    with pool._status_connection(readonly=True) as driver:
        _require_w66_receipt(driver)


class W67SinglePermitPool(SharedKeyPool):
    """Keep the shared latch set and admit only the verified W67 bridge job."""

    def __init__(self, directory, config, *, contract_sha256, payload_sha256, expires_unix,
                 request_id=None, **settings):
        super().__init__(directory, config, **settings)
        if (not isinstance(contract_sha256, str) or not SHA256.fullmatch(contract_sha256)
                or contract_sha256 == W66_CONTRACT or payload_sha256 != PAYLOAD_SHA256
                or type(expires_unix) is not int or expires_unix != EXPIRES_UNIX
                or request_id is not None and (not isinstance(request_id, str) or not SHA256.fullmatch(request_id))):
            raise PermissionError("the W67 permit differs from its registered signed limits")
        self.contract_sha256 = contract_sha256
        self.payload_sha256 = payload_sha256
        self.expires_unix = expires_unix
        self.request_id = request_id
        self._request_lock = Lock()

    def _check_request(self, request, transport):
        binding = request["binding"]
        formal = binding["a7_formal"]
        if ((binding["seed"], binding["round"], request["logical_index"]) != (157, 7, 0)
                or request["request_id"] == W66_REQUEST
                or formal["contract_sha256"] != self.contract_sha256 or formal["mock"] is not False
                or request["payload_sha256"] != self.payload_sha256 or digest(request["payload"]) != self.payload_sha256
                or (transport.provider, transport.base_url, transport.requested_model, transport.served_model,
                    transport.reasoning_effort, transport.wire_mode, transport.sdk_version) != EXPECTED_TRANSPORT):
            raise PermissionError("the W67 lease binds the complete reviewed R7 scientific request and transport")
        check_request(request, binding, transport)

    def bind_request(self, request):
        """Called by the authenticated dispatcher after ordinary job validation."""
        self._check_request(request, TransportConfig(**request["configuration"]))
        with self._request_lock:
            if self.request_id not in (None, request["request_id"]):
                raise PermissionError("the W67 lease is already bound to a different request")
            self.request_id = request["request_id"]

    @contextmanager
    def attempt(self, request, transport, *, pause=None):
        self._check_request(request, transport)
        if self.request_id != request["request_id"]:
            raise PermissionError("the W67 bridge job was not authenticated and bound")
        with super().attempt(request, transport, pause=pause) as credential_id:
            yield credential_id

    def _check_admission(self, driver, ticket, state):
        self._prune(driver)
        now = self.clock()
        settings = driver.execute(SETTINGS_QUERY).fetchone()
        if (now >= self.expires_unix or settings is None or settings["shared_disabled"] != 1
                or settings["shared_error"] != LATCH or settings["cap"] != self.config.parallelism):
            raise TransportStop("the W67 permit expired or its exact shared latch is unavailable")
        queue = driver.execute(
            "SELECT * FROM queue WHERE state IN ('queued','busy') ORDER BY position",
        ).fetchall()
        if (len(queue) != 1 or
                (queue[0]["ticket"], queue[0]["state"], queue[0]["seed"], queue[0]["round"],
                 queue[0]["logical_index"], queue[0]["request_id"], queue[0]["dispatched"])
                != (ticket, state, 157, 7, 0, self.request_id, 0)):
            raise TransportStop("the W67 permit requires an otherwise idle shared pool and its exact authorized ticket")
        _require_w66_receipt(driver)
        driver.execute("""CREATE TABLE IF NOT EXISTS w67_operator_post
            (one INTEGER PRIMARY KEY CHECK(one=1), contract_sha256 TEXT NOT NULL, request_id TEXT NOT NULL,
             payload_sha256 TEXT NOT NULL, ticket TEXT NOT NULL, consumed_unix REAL NOT NULL)""")
        if driver.execute("SELECT 1 FROM w67_operator_post WHERE one=1").fetchone() is not None:
            raise TransportStop("the W67 physical POST permit was already consumed")
        return now, settings, queue[0]

    def _grant(self, driver, ticket, max_wait):
        self._check_admission(driver, ticket, "queued")
        return super()._grant(_W66GrantView(driver), ticket, max_wait)

    def record_dispatch(self, ticket):
        """Consume the permit atomically with the existing pre-POST journal marker."""
        try:
            with self.tx() as driver:
                now, settings, row = self._check_admission(driver, ticket, "busy")
                key = driver.execute(
                    "SELECT * FROM keys WHERE id=? AND configured=1 AND present=1 AND disabled_error IS NULL",
                    (row["key_id"],),
                ).fetchone()
                if (settings["shared_until"] > now or key is None or key["cooldown_until"] > now
                        or row["key_id"] not in self.config.keys):
                    raise TransportStop("a shared or credential cooldown prevents the W67 POST")
                driver.execute(
                    "INSERT INTO w67_operator_post VALUES(1,?,?,?,?,?)",
                    (self.contract_sha256, self.request_id, self.payload_sha256, ticket, now),
                )
                if driver.execute(
                    "UPDATE queue SET dispatched=1 WHERE ticket=? AND state='busy' AND dispatched=0",
                    (ticket,),
                ).rowcount != 1:
                    raise TransportStop("the W67 physical lease has already started a provider request")
        except TransportStop as error:
            # The provider translates this known pre-POST refusal into a sealed no-dispatch result.
            raise TransportFailure("operator_w67_post_not_permitted") from error
