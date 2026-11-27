"""One signed W66 POST using the latched shared Key Pool."""
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

PAYLOAD_SHA256 = "a107c416d5d99771b3b14a88ee77b81c73378807da79cc849c3da05edae9e1e3"
EXPIRES_UNIX = 1795782600
LATCH = "served_reasoning_effort_mismatch"
EXPECTED_TRANSPORT = ("token4research", "https://token4research.cn/v1", "gpt-6.2-astra", "gpt-6.2-astra",
                      "xhigh", RESPONSES_STREAM_IPV4, "2.28.0")
SETTINGS_QUERY = "SELECT * FROM settings WHERE one=1"


class _W66GrantView:
    """Expose the exception only to the ordinary grant query in this transaction."""

    def __init__(self, connection):
        self.connection = connection

    def execute(self, statement, parameters=()):
        if statement == SETTINGS_QUERY:
            return self.connection.execute(
                "SELECT one, cap, shared_until, shared_error, 0 AS shared_disabled "
                "FROM settings WHERE one=1 AND shared_disabled=1 AND shared_error=?",
                (LATCH,),
            )
        return self.connection.execute(statement, parameters)


class W66SinglePermitPool(SharedKeyPool):
    """Keep the shared latch set and admit only the verified W66 bridge job."""

    def __init__(self, directory, config, *, contract_sha256, payload_sha256, expires_unix,
                 request_id=None, **settings):
        super().__init__(directory, config, **settings)
        if (not isinstance(contract_sha256, str) or not SHA256.fullmatch(contract_sha256)
                or payload_sha256 != PAYLOAD_SHA256
                or type(expires_unix) is not int or expires_unix != EXPIRES_UNIX
                or request_id is not None and (not isinstance(request_id, str) or not SHA256.fullmatch(request_id))):
            raise PermissionError("the W66 permit differs from its registered signed limits")
        self.contract_sha256 = contract_sha256
        self.payload_sha256 = payload_sha256
        self.expires_unix = expires_unix
        self.request_id = request_id
        self._request_lock = Lock()

    def _check_request(self, request, transport):
        binding = request["binding"]
        formal = binding["a7_formal"]
        if ((binding["seed"], binding["round"], request["logical_index"]) != (157, 7, 0)
                or formal["contract_sha256"] != self.contract_sha256 or formal["mock"] is not False
                or request["payload_sha256"] != self.payload_sha256 or digest(request["payload"]) != self.payload_sha256
                or (transport.provider, transport.base_url, transport.requested_model, transport.served_model,
                    transport.reasoning_effort, transport.wire_mode, transport.sdk_version) != EXPECTED_TRANSPORT):
            raise PermissionError("the W66 lease binds the complete reviewed R7 scientific request and transport")
        check_request(request, binding, transport)

    def bind_request(self, request):
        """Called by the authenticated dispatcher after ordinary job validation."""
        self._check_request(request, TransportConfig(**request["configuration"]))
        with self._request_lock:
            if self.request_id not in (None, request["request_id"]):
                raise PermissionError("the W66 lease is already bound to a different request")
            self.request_id = request["request_id"]

    @contextmanager
    def attempt(self, request, transport, *, pause=None):
        self._check_request(request, transport)
        if self.request_id != request["request_id"]:
            raise PermissionError("the W66 bridge job was not authenticated and bound")
        with super().attempt(request, transport, pause=pause) as credential_id:
            yield credential_id

    def _check_admission(self, driver, ticket, state):
        self._prune(driver)
        now = self.clock()
        settings = driver.execute(SETTINGS_QUERY).fetchone()
        if (now >= self.expires_unix or settings is None or settings["shared_disabled"] != 1
                or settings["shared_error"] != LATCH or settings["cap"] != self.config.parallelism):
            raise TransportStop("the W66 permit expired or its exact shared latch is unavailable")
        queue = driver.execute(
            "SELECT * FROM queue WHERE state IN ('queued','busy') ORDER BY position",
        ).fetchall()
        if (len(queue) != 1 or
                (queue[0]["ticket"], queue[0]["state"], queue[0]["seed"], queue[0]["round"],
                 queue[0]["logical_index"], queue[0]["request_id"], queue[0]["dispatched"])
                != (ticket, state, 157, 7, 0, self.request_id, 0)):
            raise TransportStop("the W66 permit requires an otherwise idle shared pool and its exact authorized ticket")
        driver.execute("""CREATE TABLE IF NOT EXISTS w66_operator_post
            (one INTEGER PRIMARY KEY CHECK(one=1), contract_sha256 TEXT NOT NULL, request_id TEXT NOT NULL,
             payload_sha256 TEXT NOT NULL, ticket TEXT NOT NULL, consumed_unix REAL NOT NULL)""")
        if driver.execute("SELECT 1 FROM w66_operator_post WHERE one=1").fetchone() is not None:
            raise TransportStop("the W66 physical POST permit was already consumed")
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
                    raise TransportStop("a shared or credential cooldown prevents the W66 POST")
                driver.execute(
                    "INSERT INTO w66_operator_post VALUES(1,?,?,?,?,?)",
                    (self.contract_sha256, self.request_id, self.payload_sha256, ticket, now),
                )
                if driver.execute(
                    "UPDATE queue SET dispatched=1 WHERE ticket=? AND state='busy' AND dispatched=0",
                    (ticket,),
                ).rowcount != 1:
                    raise TransportStop("the W66 physical lease has already started a provider request")
        except TransportStop as error:
            # The provider translates this known pre-POST refusal into a sealed no-dispatch result.
            raise TransportFailure("operator_w66_post_not_permitted") from error
