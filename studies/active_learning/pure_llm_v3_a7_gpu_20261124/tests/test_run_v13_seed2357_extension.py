from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, file_digest, once
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.response_schema import SCHEMA_VERSION
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    RESPONSES_STREAM_IPV4, TransportConfig, TransportFailure, TransportStop, make_request, run_request,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.a7_gpu.bridge_protocol import (
    BridgePending, make_job, make_result,
)
from studies.active_learning.pure_llm_v3_a7_gpu_20261124.scripts.run_v13_seed2357_extension import (
    _fitted_cache, _one_attempt, _prior_attempts, _same_science,
)


@pytest.fixture
def original_request():
    config = TransportConfig("token4research", "gpt-6.2-astra", "gpt-6.2-astra", "xhigh",
                             "https://token4research.cn/v1", 3600, 32768, transport_attempts=2,
                             wire_mode=RESPONSES_STREAM_IPV4)
    packet = {"seed": 2357, "round": 1, "layout": "B", "compression": "lossless",
              "memory": {"mode": "M2"}, "selection_count": 32}
    messages = [{"role": "system", "content": "Synthetic A7 system message"},
                {"role": "user", "content": dumps(packet)}]
    approval = {"test approval": True}
    scope = {"contract_sha256": "a" * 64, "authorization_sha256": digest(approval),
             "seed": 2357, "mock": False}
    binding = {
        "a7_formal": scope, "seed": 2357, "round": 1, "arm": "A7", "prompt_version": "v3",
        "schema_version": SCHEMA_VERSION, "packet_sha256": digest(packet),
        "prompt_sha256": digest(messages[0]["content"]), "base_messages_sha256": digest(messages),
        **{key: digest(key) for key in ("checkpoint_sha256", "state_sha256", "mapping_sha256", "split_sha256")},
    }
    return make_request(binding, 0, messages, config), config, approval


def test_two_preserved_receipts_and_no_recovery(tmp_path, original_request):
    original, config, approval = original_request
    parent = tmp_path / "v13"
    turn = parent / "seed_2357/round_01/turn_00"
    contract = SimpleNamespace(digest=original["binding"]["a7_formal"]["contract_sha256"],
                               transport=config, profile={"wire_representation": None})
    evidence = []
    for index, attempt_id in enumerate(("b" * 32, "c" * 32)):
        started = {"request_id": original["request_id"], "attempt": index, "attempt_id": attempt_id,
                   "started_unix": 1, "transport": "mac_bridge"}
        job = make_job(original, config, started)
        result = make_result(job, failure=TransportFailure("timeout", ambiguous=True), credential_id="key_01",
                             provider_dispatched=True)
        terminal = {**started, "status": "stopped", "error": "timeout", "ambiguous": True,
                    "provider_dispatched": True, "circuit_open": False,
                    "bridge_job_sha256": job["job_sha256"], "bridge_result_sha256": result["result_sha256"]}
        once(turn / f"attempt_{index:02d}.started.json", started)
        once(turn / f"attempt_{index:02d}.terminal.json", terminal)
        for folder, record in (("outbox", job), ("results", result)):
            path = parent / "bridge" / folder / f"{attempt_id}.json"
            once(path, record)
            path.chmod(0o600)
        evidence.append({"attempt": index, "attempt_id": attempt_id, "job_sha256": job["job_sha256"],
                         "result_sha256": result["result_sha256"], "terminal_sha256": digest(terminal)})

    assert _prior_attempts(parent, turn, original, contract, approval) == evidence
    once(turn / "attempt_01.reconciliation.json", {"decision": "import_completed"})
    with pytest.raises(PermissionError, match="unreconciled"):
        _prior_attempts(parent, turn, original, contract, approval)


def test_exact_payload_one_reservation_and_resume(tmp_path, original_request):
    original, old_config, _ = original_request
    config = replace(old_config, transport_attempts=1, logical_replies=1)
    binding = deepcopy(original["binding"])
    binding["a7_formal"]["contract_sha256"] = "e" * 64
    request = make_request(binding, 0, original["payload"]["input"], config)
    assert _same_science(request, original)
    assert not _same_science(original, original)
    changed = deepcopy(request)
    changed["binding"]["checkpoint_sha256"] = "f" * 64
    assert not _same_science(changed, original)

    seen = []
    prior = [{"attempt_id": "b" * 32}, {"attempt_id": "c" * 32}]

    def mocked(candidate, transport, started):
        assert _one_attempt(request, config, prior, candidate, transport, started)
        for changed_start in ({**started, "attempt": 1}, {**started, "attempt_id": prior[0]["attempt_id"]}):
            assert not _one_attempt(request, config, prior, candidate, transport, changed_start)
        seen.append(started["attempt_id"])
        if len(seen) == 1:
            raise BridgePending("existing local result not returned yet")
        raise TransportFailure("timeout", ambiguous=True)

    def forbidden_direct(*_args):
        pytest.fail("extension must only use the managed bridge")

    with pytest.raises(BridgePending):
        run_request(tmp_path, request, config, forbidden_direct, managed_attempt=mocked)
    for _ in range(2):
        with pytest.raises(TransportStop, match="timeout"):
            run_request(tmp_path, request, config, forbidden_direct, managed_attempt=mocked)
    assert len(seen) == 2 and seen[0] == seen[1]
    assert not (tmp_path / "turn_00" / "attempt_01.started.json").exists()



def test_fitted_cache_uses_signed_draft_when_state_has_no_anchor_resource_digest(tmp_path):
    preparation = tmp_path / "preparation"
    prepared = preparation / "boundaries" / "seed_2357"
    cache = tmp_path / "public_graphs" / "seed_2357"
    receipt = cache / "resource_receipt.json"
    once(receipt, {"synthetic": "registered cache"})
    state = {"checkpoint": "a" * 64, "provenance": {"fit_audit_sha256": "b" * 64}}
    descriptor = {"round": 1, "state_sha256": digest(state), "memory_sha256": "c" * 64}
    lane = {"cache_directory": str(cache)}
    draft = {"boundaries": {"2357": {
        **lane, "descriptor": descriptor, "boundary_directory": str(prepared),
        "checkpoint_sha256": state["checkpoint"],
        "evidence": {"source": "sealed_v11", "contract_sha256": "d" * 64,
                     "fit_audit_sha256": state["provenance"]["fit_audit_sha256"],
                     "public_resource_receipt_sha256": file_digest(receipt)},
    }}}
    once(preparation / "five_seed_boundary_draft.json", draft)
    contract = SimpleNamespace(predecessor={"prepared_draft_sha256": digest(draft),
                                            "prior_v11_contract_sha256_by_seed": {"2357": "d" * 64}})

    assert _fitted_cache(contract, lane, prepared, descriptor, state) == cache
    receipt.write_text('{"synthetic": "changed cache"}')
    with pytest.raises(PermissionError, match="signed fitted boundary"):
        _fitted_cache(contract, lane, prepared, descriptor, state)
