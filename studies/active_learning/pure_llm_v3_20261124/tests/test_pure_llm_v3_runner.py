from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.context_builder import decode_context, encode_context, read_legacy_cards
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.frozen_state import load_historical
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.offline_examples import mock_plan
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import digest, dumps, once, read
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.runner import (
    ROOT, PROTOCOL_PATH, _authorize, _validation, select_stage, stage_state,
)
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.repair_protocol import build_repair_request
from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.transport_manager import (
    ProviderReply, TransportConfig, TransportFailure, TransportStop, collect_completed, make_request,
    reconcile, retry_http, round_lock, run_request,
)


@pytest.fixture(scope="module")
def stages():
    states = [load_historical(ROOT, 157, j) for j in range(2)]
    return [stage_state(ROOT, s) for s in states]


@pytest.fixture
def config():
    return replace(TransportConfig(**read(ROOT / PROTOCOL_PATH)["transport"]), provider="mock",
                   sdk_version="offline")


def test_full_pool_layouts_role_migration_and_metadata(stages):
    first, later = stages
    assert (len(first.context.candidates), len(first.context.observed)) == (2997, 333)
    assert (len(later.context.candidates), len(later.context.observed)) == (2965, 365)
    for stage in stages:
        context = stage.context
        for layout in "AB":
            for compress in (False, True):
                packet = encode_context(context, layout=layout, compress=compress)
                assert decode_context(packet) == {"candidates": context.candidates, "observed": context.observed}
        assert {r["conditions"]["flow"] for r in context.candidates.values()} >= {"5", "10"}
        assert any(r["conditions"]["purity_raw"] == "AR" for r in context.candidates.values())
        assert all("V1_ml" not in r and "true_V1_ml" not in r for r in context.candidates.values())
    newly_observed = later.context.observed.keys() - first.context.observed.keys()
    assert len(newly_observed) == 32 and len(later.packet["memory"]["latest_feedback"]) == 32
    for alias in newly_observed:
        candidate = "C" + alias[1:]
        assert first.context.all_row_ids[candidate] == later.context.all_row_ids[alias]
        assert candidate not in later.context.candidates
    assert list(encode_context(first.context, layout="A")["candidates"]["rows"][0])[0] == next(
        iter(first.context.candidates)
    )


def test_unregistered_historical_column_and_foreign_observation_rejected():
    state = load_historical(ROOT, 157, 0)
    packet = deepcopy(state.packet)
    packet["candidates"]["columns"].append("true_V1_ml")
    for row in packet["candidates"]["rows"]:
        row.append(999)
    with pytest.raises(ValueError, match="table schema"):
        read_legacy_cards(replace(state, packet=packet))
    packet = deepcopy(state.packet)
    packet["observed"]["rows"][0][0] = "foreign-observation"
    with pytest.raises(ValueError, match="role IDs"):
        read_legacy_cards(replace(state, packet=packet))


def test_legacy_reference_and_prompt_only_ablation_preserve_exact_data_message():
    state = load_historical(ROOT, 157, 0)
    original, revised = [stage_state(ROOT, state, arm=a) for a in ("A0", "A1")]
    assert original.messages == state.messages
    assert original.messages[0] != revised.messages[0]
    assert original.messages[1] == revised.messages[1]
    assert revised.packet["packet_sha256"] == original.packet["packet_sha256"]


def test_transport_with_429_then_completion_and_idempotence(tmp_path, config):
    calls, delays = [], []
    request = make_request({"seed": 157}, 0, [], config)
    def call(_request, _config):
        calls.append(1)
        if len(calls) == 1:
            raise retry_http(429, {"retry-after": "5"})
        return ProviderReply("{}", "response-1", config.served_model, {"input_tokens": 8})
    opts = {"sleep": delays.append, "now": lambda: 12, "jitter": lambda: 0}
    for _ in range(2):
        assert run_request(tmp_path, request, config, call, **opts)["response_id"] == "response-1"
    assert len(calls) == 2
    assert delays == [5]
    assert len(list(tmp_path.glob("*/attempt_*.started.json"))) == 2


@pytest.mark.parametrize("kind", ["http_403", "timeout", "sse_frame_error", "empty_response",
                                  "served_model_mismatch", "incomplete_response"])
def test_permanent_or_ambiguous_loss_stops_without_retransmission(tmp_path, config, kind):
    calls = []
    request = make_request({"seed": 157}, 0, [], config)
    def call(_request, _config):
        calls.append(1)
        if kind in {"http_403", "timeout", "sse_frame_error"}:
            raise TransportFailure(kind, ambiguous=kind != "http_403")
        return ProviderReply("" if kind == "empty_response" else "{}",
            "resp-1", "incorrect" if kind == "served_model_mismatch" else config.served_model, {},
            "incomplete" if kind == "incomplete_response" else "completed")
    for _ in range(2):
        with pytest.raises(TransportStop, match=kind):
            run_request(tmp_path, request, config, call)
    assert len(calls) == 1


def test_429_circuit_and_retry_after_ceiling(tmp_path, config):
    for delay, expected in ((1, 2), (61, 1)):
        subdir = tmp_path / str(delay)
        req = make_request({"seed": 157}, 0, [], config)
        def call(_request, _config):
            raise retry_http(429, {"retry-after": str(delay)})
        with pytest.raises(TransportStop, match="http_429"):
            run_request(subdir, req, config, call, sleep=lambda _: None)
        terminal = sorted(subdir.glob("*/attempt_*.terminal.json"))
        assert len(terminal) == expected
        assert read(terminal[-1])["circuit_open"]


def test_incomplete_sse_has_no_answer_and_receipts_reconcile(tmp_path, config):
    with pytest.raises(TransportFailure, match="sse_missing_completion"):
        collect_completed([SimpleNamespace(type="response.output_text.delta")])
    binding = {"seed": 157}
    request = make_request(binding, 0, [], config)
    def interrupt(_request, _config):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run_request(tmp_path, request, config, interrupt)
    with pytest.raises(TransportStop, match="ambiguous"):
        run_request(tmp_path, request, config, interrupt)
    started = read(tmp_path / "turn_00/attempt_00.started.json")
    approval = {"request_id": request["request_id"], "attempt_id": started["attempt_id"],
                "decision": "import_completed", "authorized_by": "offline fixture",
                "evidence": "synthetic externally recovered terminal"}
    reconcile(tmp_path, request, 0, authorization=approval, config=config,
              reply=ProviderReply("{}", "recovered", config.served_model, {}))
    assert run_request(tmp_path, request, config, interrupt)["response_id"] == "recovered"
    foreign = make_request({"seed": 887}, 0, [], config)
    with pytest.raises(ValueError, match="immutable receipt"):
        run_request(tmp_path, foreign, config, interrupt)


def test_runner_select_repair_freeze_and_refuse_modified_packet(stages, tmp_path, config):
    stage = stages[1]
    plan = mock_plan(stage)
    initial = deepcopy(plan)
    initial["choices"][7]["candidate_id"] = "C-9999"
    initial["choices"][7]["reason"] = "EXTERNAL_REJECTED_PROSE cw_distance_percentile: 0.91"
    plan["choices"][5]["reason"] += " Model-authored LCMD theory mention."
    initial["choices"][5]["reason"] = plan["choices"][5]["reason"]
    contract = build_repair_request(initial, validation=_validation(stage, initial), binding=stage.binding)
    patch = {"mode": "patch", "repair_id": contract["repair_id"], "reason": "fixture replacement",
             "replace_choices": [{"slot": 7, "choice": plan["choices"][7]}], "replace_fields": {},
             "append_choices": [], "replace_hypothesis_updates": []}
    calls = []
    def call(request, cfg):
        index = request["logical_index"]
        calls.append(index)
        assert "EXTERNAL_REJECTED_PROSE" not in dumps(request["payload"])
        return ProviderReply(dumps((initial, patch)[index]), f"reply-{index}", cfg.served_model, {})
    result = select_stage(stage, tmp_path, config, call, mock=True)
    assert len(set(result["batch_ids"])) == 32 and result["label_access"] is False
    assert read(tmp_path / "selection.json") == plan
    assert select_stage(stage, tmp_path, config, call, mock=True) == result
    assert calls == [0, 1]
    tampered = {**stage.packet, "candidate_count": 1}
    with pytest.raises(ValueError, match="binding drift"):
        select_stage(replace(stage, packet=tampered), tmp_path, config, call, mock=True)
    with pytest.raises(PermissionError):
        _authorize(None, stage, config)


def test_independent_retry_and_logical_reply_budgets(tmp_path, stages, config):
    stage = stages[0]
    config = replace(config, logical_replies=2)
    good = mock_plan(stage)
    calls = []
    def call(request, cfg):
        logical = request["logical_index"]
        calls.append(logical)
        if len(calls) == 1:
            raise retry_http(429, {})
        return ProviderReply("bad JSON" if logical == 0 else dumps(good),
                             f"answer-{logical}", cfg.served_model, {})
    select_stage(stage, tmp_path, config, call, mock=True, retry_options={"sleep": lambda _: None})
    assert calls == [0, 0, 1]
    assert len(list(tmp_path.glob("*/decision.json"))) == 2


def test_nested_round_lock_rejects_concurrent_worker(tmp_path):
    with round_lock(tmp_path):
        with pytest.raises(TransportStop, match="another worker"):
            with round_lock(tmp_path):
                pytest.fail("concurrent lock unexpectedly acquired")


def test_recorded_timeout_can_be_reconciled_without_overwriting_or_retrying(tmp_path, config):
    request = make_request({"seed": 157}, 0, [], config)
    def timeout(_request, _config):
        raise TransportFailure("timeout", ambiguous=True)
    with pytest.raises(TransportStop, match="timeout"):
        run_request(tmp_path, request, config, timeout)
    terminal = tmp_path / "turn_00/attempt_00.terminal.json"
    before = terminal.read_bytes()
    approval = {"request_id": request["request_id"], "attempt_id": read(terminal)["attempt_id"],
                "decision": "import_completed", "authorized_by": "offline fixture", "evidence": "mock evidence"}
    with pytest.raises(PermissionError):
        reconcile(tmp_path, request, 0, authorization={**approval, "attempt_id": "unbound"},
                  config=config, reply=ProviderReply("{}", "recovered", config.served_model, {}))
    reconcile(tmp_path, request, 0, authorization=approval, config=config,
              reply=ProviderReply("{}", "recovered", config.served_model, {}))
    assert terminal.read_bytes() == before
    assert run_request(tmp_path, request, config, timeout)["response_id"] == "recovered"


def test_mock_freeze_cannot_enter_future_label_stage(stages, tmp_path, config):
    from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.trajectory import _accepted

    stage = stages[0]
    plan = mock_plan(stage)
    select_stage(stage, tmp_path, config,
                 lambda req, cfg: ProviderReply(dumps(plan), "mock-completed", cfg.served_model, {}),
                 mock=True)
    with pytest.raises(PermissionError, match="live V3 selection"):
        _accepted(tmp_path, 157, 0)


def test_legacy_control_validates_actual_frozen_plans_at_both_pilot_budgets():
    pytest.importorskip("rdkit")
    for round_index in (0, 6):
        state = load_historical(ROOT, 157, round_index)
        historical_plan = read(ROOT / state.provenance["source_directory"] / "selection.json")
        for arm in ("A0", "A1"):
            result = _validation(stage_state(ROOT, state, arm=arm), historical_plan)
            assert result.valid, result.as_dict()
            assert result.valid_choice_count == 32
