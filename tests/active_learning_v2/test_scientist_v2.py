"""Synthetic scientific boundary tests; no network or canonical hidden labels."""
import copy
import json

import numpy as np
import pytest

from src.qgeognn_al.active_learning_v2 import scientist_study as study
from src.qgeognn_al.active_learning_v2.dialog_bridge import once, read
from src.qgeognn_al.active_learning_v2.protocol import ids_hash, stable_hash
from src.qgeognn_al.active_learning_v2.scientist_selector import (
    ReadOnlyCatalog, SYSTEM_PROMPT, scientific_memory, validate_selection)
from src.qgeognn_al.active_learning_v2.scientist_transport import (
    audit_cli, recover_cli_rollout, run_selector, settings)


def row(i, observed=False):
    value = {"id": str(i), "smiles": "CCO" if i % 2 else "CCN",
             "conditions": {"PE/EA": i % 5, "Density g/ml": 1.0, "V/ul": 5,
                            "loading solvent": "EA", "Volume of loading solvent/ul": 200}}
    if observed:
        value.update(V1_ml=1.0, V2_ml=2.0)
    else:
        value.update(pred_V1_ml=float(i), pred_V2_ml=float(i+3), cw_distance_percentile=i/100)
        value.update({f"pred_{t}_{q}_ml": float(i+j) for t in ("V1", "V2")
                      for j, q in enumerate(("q10", "q50", "q90"))})
    return value


def raw(free=True):
    return {"candidates": [row(i) for i in range(17, 97)], "observed": [row(0, True)],
            "pending": [] if free else [row(i) for i in range(1, 17)], "salt": "synthetic:157:free:0"}


def packet(count=32, r=0):
    p = {"selection_count": count, "round": r, "seed": 157, "method": study.METHODS[count == 32]}
    p["packet_sha256"] = stable_hash(p)
    return p


def selection(cat, p):
    return {"type": "selection", "packet_sha256": p["packet_sha256"], "batch_strategy": "local contrasts",
            "choices": [{"id": i, "reason": "test condition response", "hypothesis_id": None}
                        for i in sorted(cat.viewed)[:p["selection_count"]]], "hypotheses": [],
            "feedback_interpretation": "" if p["round"] == 0 else "The prior hypothesis was weakened.",
            "batch_rationale": "These related experiments test a response curve that may generalize."}


def view(cat):
    cat.initial_cards()
    for offset in (0, 24, 48):
        cat.query("search_candidates", {"offset": offset})


def test_unbiased_order_pagination_and_pending_references():
    data = raw(False)
    a = ReadOnlyCatalog(**data)
    data["candidates"].reverse()
    b = ReadOnlyCatalog(**data)
    assert a.initial_cards() == b.initial_cards()
    assert a.query("search_candidates", {}) == b.query("search_candidates", {})
    first = a.query("search_candidates", {"limit": 24})
    second = a.query("search_candidates", {"offset": first["next_offset"]})
    assert not {r["id"] for r in first["matches"]} & {r["id"] for r in second["matches"]}
    assert first["total_matches"] == 80
    found = a.query("search_candidates", {"same_molecule_as": "1", "similar_to": "1"})["matches"]
    assert found and all(r["smiles"] == "CCO" and r["similarity"] == 1 for r in found)
    assert a.query("get_pending", {"ids": ["1"]})["matches"][0]["id"] == "1"


@pytest.mark.parametrize("sort_by", ["pred_V1_ml", "pred_V2_ml", "cw_distance_percentile", "MW_g_mol",
    "V1_quantile_width_ml", "V2_quantile_width_ml", "molecule_novelty", "condition_novelty",
    "max_similarity_to_observed", "max_similarity_to_pending"])
def test_numerical_evidence_sorting(sort_by):
    c = ReadOnlyCatalog(**raw())
    rows = c.query("search_candidates", {"sort_by": sort_by, "order": "asc"})["matches"]
    values = [r["descriptors"].get(sort_by, r.get(sort_by)) for r in rows]
    assert values == sorted(values)


def test_residual_queries_only_for_observed_and_missing_last():
    data = raw()
    data["observed"].append({**row(2, True), "premeasurement_error_V1_ml": -7,
                              "premeasurement_error_V2_ml": 5})
    c = ReadOnlyCatalog(**data)
    assert c.query("search_observed", {"sort_by": "historical_abs_error_ml"})["matches"][0]["id"] == "2"
    with pytest.raises(ValueError, match="observed-only"):
        c.query("search_candidates", {"sort_by": "historical_abs_error_ml"})
    assert c.query("search_candidates", {"signal": "MW_g_mol", "min_signal": 0})["matches"]
    assert c.query("search_candidates", {"conditions": {"PE/EA": 100}})["matches"] == []


@pytest.mark.parametrize("field", ["V1_ml", "V2_ml"])
def test_measured_endpoint_sort_and_filter_are_observed_only(field):
    data = raw(False)
    data["observed"].append({**row(-2, True), field: 9.0})
    c = ReadOnlyCatalog(**data)
    found = c.query("search_observed", {"sort_by": field, "order": "desc"})["matches"]
    assert [r["id"] for r in found] == ["-2", "0"]
    assert c.query("search_observed", {"signal": field, "min_signal": 8})["returned_count"] == 1
    for operation in ("search_candidates", "search_pending"):
        for args in ({"sort_by": field}, {"signal": field}):
            with pytest.raises(ValueError, match="observed-only"):
                c.query(operation, args)


@pytest.mark.parametrize("field", ["V1_ml", "true_V1_ml", "test_error", "secret"])
def test_unobserved_field_allowlist(field):
    data = raw()
    data["candidates"][0][field] = 99
    with pytest.raises(ValueError, match="unauthorized"):
        ReadOnlyCatalog(**data)


def test_foreign_ids_overlap_budget_and_response_copy():
    data = raw(False)
    c = ReadOnlyCatalog(**data, view_budget=21, query_budget=2)
    initial = c.initial_cards()
    initial[0]["conditions"]["PE/EA"] = "modified"
    assert "modified" not in [r["conditions"]["PE/EA"] for r in c.candidates.values()]
    c.query("search_candidates", {"offset": 24})
    assert len(c.viewed) == 21
    assert c.query("get_observed", {"ids": ["unknown-other-seed", "17", "1"]})["matches"] == []
    with pytest.raises(ValueError, match="budget"):
        c.query("search_candidates", {})
    data["pending"].append(row(0))
    with pytest.raises(ValueError, match="overlapping"):
        ReadOnlyCatalog(**data)


@pytest.mark.parametrize("count", [16, 32])
def test_free_and_hybrid_selection_without_molecule_quota(count):
    c = ReadOnlyCatalog(**raw(count == 32))
    view(c)
    p = packet(count)
    result = selection(c, p)
    assert len(validate_selection(result, c, p)["choices"]) == count
    for change in (lambda v: v["choices"].pop(),
                   lambda v: v["choices"][0].update(id="0"),
                   lambda v: v["choices"][0].update(hypothesis_id="missing"),
                   lambda v: v.update(packet_sha256="foreign"),
                   lambda v: v.update(batch_rationale="")):
        invalid = copy.deepcopy(result)
        change(invalid)
        with pytest.raises(ValueError):
            validate_selection(invalid, c, p)


def test_hypotheses_and_memory_are_trajectory_local_and_bounded():
    c = ReadOnlyCatalog(**raw())
    view(c)
    p = packet(r=1)
    result = selection(c, p)
    h = {"id": "h", "content": "possible loading effect", "basis": "mixed",
         "supporting_observed_ids": ["0"], "contradicting_observed_ids": [],
         "confidence": "low", "status": "unresolved", "why_it_matters_for_model_learning": "response curve",
         "next_discriminating_question": "Does loading alter the curve?"}
    result["hypotheses"] = [h]
    validate_selection(result, c, p)
    h["supporting_observed_ids"] = ["17"]
    with pytest.raises(ValueError, match="unobserved"):
        validate_selection(result, c, p)
    histories = [{"seed": 157, "method": "free", "round": r, "hypotheses": [],
                  "feedback_interpretation": "verbatim LLM reasoning", "records": [
                      {"candidate_id": str(r*32+i), "error_V1_ml": i, "error_V2_ml": -i}
                      for i in range(32)]} for r in range(6)]
    m = scientific_memory(histories, 157, "free")
    assert len(m["recent_outcomes"]) == 32 and len(m["high_error_measured_examples"]) == 8
    assert len(m["previous_selected_ids"]) == 192
    assert m["previous_feedback_interpretation"] == "verbatim LLM reasoning"
    with pytest.raises(ValueError, match="foreign"):
        scientific_memory(histories, 6101, "free")


def mocked_call(p):
    def call(messages, config):
        answer = {"type": "query", "queries": [
            {"operation": "search_candidates", "args": {"offset": offset}} for offset in (0, 24)]}
        if len(messages) > 2:
            results = json.loads(messages[-1]["content"])["query_results"]
            ids = [r["id"] for res in results for r in res["matches"]]
            answer = {"type": "selection", "packet_sha256": p["packet_sha256"], "batch_strategy": "local contrasts",
                      "choices": [{"id": i, "reason": "information", "hypothesis_id": None} for i in ids[:p["selection_count"]]],
                      "hypotheses": [], "feedback_interpretation": "" if p["round"] == 0 else "updated",
                      "batch_rationale": "Contrasts may reveal systematic errors."}
        return json.dumps(answer), {"mock": True, "native_tool_calls": 0}
    return call


def test_transport_replay_and_no_automatic_retry(tmp_path):
    p = packet()
    c = ReadOnlyCatalog(**raw())
    c.initial_cards()
    first = run_selector(p, c, tmp_path, settings(), call=mocked_call(p))
    other = ReadOnlyCatalog(**raw())
    other.initial_cards()
    def fail(*args):
        pytest.fail("resume must not call model")
    assert run_selector(p, other, tmp_path, settings(), call=fail) == first
    assert c.queries == other.queries
    with pytest.raises(RuntimeError, match="drift"):
        run_selector({**p, "seed": 6101}, other, tmp_path, settings(), call=fail)
    empty = tmp_path / "interrupted"
    empty.mkdir()
    once(empty / "turn_00.started.json", {"request_sha256": "anything"})
    with pytest.raises(RuntimeError, match="interrupted|started CLI"):
        run_selector(p, c, empty, settings(), call=fail)


def test_cli_audit_rejects_native_tools_unknown_items_and_wrong_model():
    events = [{"type": "item.completed", "item": {"type": "agent_message", "text": "{}"}},
              {"type": "turn.completed"}]
    rollouts = [{"type": "turn_context", "payload": {"model": "gpt-5.5", "effort": "high"}},
                {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                 "phase": "final", "content": [{"text": "{}"}]}}]
    assert audit_cli(events, rollouts, "gpt-5.5", "high")[0] == "{}"
    for kind in ("function_call", "custom_tool_call", "web_search_call", "future_unknown_item"):
        with pytest.raises(RuntimeError, match="native|unknown"):
            audit_cli(events, rollouts + [{"type": "response_item", "payload": {"type": kind}}], "gpt-5.5", "high")
    with pytest.raises(RuntimeError, match="native"):
        audit_cli(events + [{"type": "item.started", "item": {"type": "command_execution"}}], rollouts, "gpt-5.5", "high")
    with pytest.raises(RuntimeError, match="provenance"):
        audit_cli(events, rollouts, "wrong", "high")


def test_cli_audit_accepts_codex_cli_jsonl_event_casing():
    events = [{"type": "item_completed", "item": {"type": "AgentMessage", "text": "{}"}},
              {"type": "task_complete"}]
    rollouts = [{"type": "turn_context", "payload": {"model": "gpt-6-sol", "effort": "high"}},
                {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                 "phase": "final_answer", "content": [{"text": "{}"}]}}]
    assert audit_cli(events, rollouts, "gpt-6-sol", "high")[0] == "{}"
    for kind in ("FunctionCall", "CustomToolCall", "WebSearchCall"):
        bad = [{"type": "item_completed", "item": {"type": kind}}, {"type": "task_complete"}]
        with pytest.raises(RuntimeError, match="native|unknown"):
            audit_cli(bad, rollouts, "gpt-6-sol", "high")


def test_cli_ignored_settings_warning_is_not_a_native_call():
    warning = ("Codex is ignoring 2 unrecognized configuration settings. Check for typos or deprecated settings.\n"
               "  user (/tmp/config.toml): `disable_response_storage` is ignored.\n"
               "  user (/tmp/config.toml): `mcp_servers.computer-use.type` is ignored.")
    events = [{"type": "item.completed", "item": {"type": "error", "message": warning}},
              {"type": "item.completed", "item": {"type": "agent_message", "text": "{}"}},
              {"type": "turn.completed"}]
    rows = [{"type": "turn_context", "payload": {"model": "gpt-6-sol", "effort": "high"}},
            {"type": "response_item", "payload": {"type": "message", "role": "assistant",
             "phase": "final_answer", "content": [{"text": "{}"}]}}]
    assert audit_cli(events, rows, "gpt-6-sol", "high")[1]["configuration_warnings"] == [warning]
    for error in ("API request failed", warning + "\nadditional error", warning.replace("disable_response_storage", "unknown")):
        events[0]["item"]["message"] = error
        with pytest.raises(RuntimeError, match="native|unknown"):
            audit_cli(events, rows, "gpt-6-sol", "high")
    events.pop(0)
    events[0]["item"] = {"type": "AgentMessage", "content": [{"text": "{}"}]}
    assert audit_cli(events, rows, "gpt-6-sol", "high")[0] == "{}"


@pytest.mark.parametrize("kind,field,bound,value", [
    ("signal", "molecule_novelty", "min_signal", 1),
    ("descriptor", "HBA", "min_value", 1),
])
def test_nested_filters_match_flat_filters_and_preserve_request(kind, field, bound, value):
    a, b = ReadOnlyCatalog(**raw(True)), ReadOnlyCatalog(**raw(True))
    nested = {kind: {"name": field, bound: value}, "limit": 24}
    flat = {kind: field, bound: value, "limit": 24}
    assert a.query("search_candidates", nested) == b.query("search_candidates", flat)
    assert a.queries[0]["args"] == nested
    with pytest.raises(ValueError, match="conflicting"):
        a.query("search_candidates", {**nested, bound: value+1})
    with pytest.raises(ValueError, match="invalid nested"):
        a.query("search_candidates", {kind: {"name": field, "unknown": 1}})
    with pytest.raises(ValueError, match="observed-only"):
        a.query("search_candidates", {"signal": {"name": "V1_ml", "min_signal": 0}})


def test_recover_completed_cli_rollout_only_for_exact_request(tmp_path):
    message = [{"role": "system", "content": "scientific prompt"},
               {"role": "user", "content": "exact packet"}]
    final = json.dumps({"type": "query", "queries": [{"operation": "search_candidates", "args": {}}]})
    rows = [{"type": "turn_context", "payload": {"model": "gpt-6-sol", "effort": "high"}},
            {"type": "response_item", "payload": {"type": "message", "role": "user",
             "content": [{"text": json.dumps(message, ensure_ascii=False)}]}},
            {"type": "event_msg", "payload": {"type": "item_completed", "item": {"type": "UserMessage"}}},
            {"type": "event_msg", "payload": {"type": "item_completed", "item": {"type": "Reasoning"}}},
            {"type": "response_item", "payload": {"type": "reasoning"}},
            {"type": "response_item", "payload": {"id": "final-id", "type": "message", "role": "assistant",
             "phase": "final_answer", "content": [{"text": final}]}},
            {"type": "event_msg", "payload": {"type": "item_completed", "item": {"type": "AgentMessage", "id": "final-id"}}},
            {"type": "event_msg", "payload": {"type": "task_complete"}}]
    session = tmp_path / "sessions/one.jsonl"
    session.parent.mkdir()
    session.write_text("\n".join(json.dumps(row) for row in rows))
    unrelated = [rows[0], {"type": "response_item", "payload": {"type": "function_call"}}]
    (session.parent / "unrelated.jsonl").write_text("\n".join(json.dumps(row) for row in unrelated))
    session.write_text("\n".join(json.dumps(row) for row in rows + unrelated))
    config = settings(model="gpt-6-sol")
    answer, provenance = recover_cli_rollout(message, config, started_at=0, codex_home=tmp_path)
    assert answer == final and provenance["recovered_after_cli_event_parser_failure"]
    foreign = [{"role": "system", "content": "different"}, *message[1:]]
    with pytest.raises(RuntimeError, match="exactly one"):
        recover_cli_rollout(foreign, config, started_at=0, codex_home=tmp_path)
    rows.insert(-2, {"type": "response_item", "payload": {"type": "function_call"}})
    session.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(RuntimeError, match="native"):
        recover_cli_rollout(message, config, started_at=0, codex_home=tmp_path)


@pytest.mark.parametrize("method", study.METHODS)
def test_freeze_before_reveal_idempotence_and_tamper(tmp_path, monkeypatch, method):
    monkeypatch.setattr(study, "STUDY", tmp_path)
    monkeypatch.setattr(study, "stage", lambda *args: None)
    p = packet(16 if method == study.METHODS[0] else 32)
    d = study.selection_directory(157, method, 0)
    d.mkdir(parents=True)
    once(tmp_path / "protocol.json", {"transport": settings()})
    once(d / "input.json", p)
    once(d / "catalog.json", raw(method == study.METHODS[1]))
    once(d / "contract.json", {"U_t_ids": [str(i) for i in range(1, 97)]})
    def selector(p, c, directory, config):
        return run_selector(p, c, directory, config, call=mocked_call(p))
    monkeypatch.setattr(study, "run_selector", selector)
    study.select(157, method, 0)
    monkeypatch.setattr(study, "run_selector", run_selector)
    frozen = study.audit_batch(d)
    assert len(set(frozen["batch_ids"])) == 32 and not frozen["new_batch_labels_revealed"]
    assert len(frozen["pending_ids"]) == (16 if method == study.METHODS[0] else 0)
    assert study.select(157, method, 0)["status"] == "ALREADY_FROZEN"
    assert not (d / "feedback.json").exists()
    with (d / "selection.json").open("a") as f:
        f.write(" ")
    with pytest.raises(RuntimeError, match="drift"):
        study.audit_batch(d)


def test_report_barrier_precedes_truth_access(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "STUDY", tmp_path)
    monkeypatch.setattr(study, "validate", lambda: None)
    def forbidden(*args):
        pytest.fail("incomplete study cannot reach a label store")
    monkeypatch.setattr(study, "Context", forbidden)
    with pytest.raises(FileNotFoundError):
        study.report()


def test_advance_reveals_only_frozen_batch_then_one_fit(tmp_path, monkeypatch):
    method = study.METHODS[1]
    monkeypatch.setattr(study, "STUDY", tmp_path)
    monkeypatch.setattr(study, "validate", lambda: None)
    monkeypatch.setattr(study, "stage", lambda *args: None)
    monkeypatch.setattr(study, "BUDGETS", (1, 33, 65, 97, 129, 161, 193))
    p = packet()
    d = study.selection_directory(157, method, 0)
    d.mkdir(parents=True)
    once(tmp_path / "protocol.json", {"transport": settings()})
    once(d / "input.json", p)
    once(d / "catalog.json", raw())
    once(d / "contract.json", {"U_t_ids": [str(i) for i in range(17, 97)]})
    monkeypatch.setattr(study, "run_selector", lambda p, c, directory, config:
                        run_selector(p, c, directory, config, call=mocked_call(p)))
    study.select(157, method, 0)
    monkeypatch.setattr(study, "run_selector", run_selector)
    calls = []
    class Store:
        def __init__(self):
            self.audit = []
        def freeze_acquisitions(self, ids):
            self.frozen = ids
        def reveal(self, ids, purpose):
            if purpose == "after_acquisition_fit":
                assert ids == self.frozen == read(d / "batch_freeze.json")["batch_ids"]
                assert (d / "batch_freeze.json").exists()
            calls.append(purpose)
            self.audit.append({"purpose": purpose, "requested_ids_hash": ids_hash(ids), "acquisitions_frozen": True})
            return np.array([[1.123, 2.789]]*len(ids), dtype=np.float32)
    class Context:
        seed = 157
        roles = {"l0": np.array([0]), "u0": np.arange(17, 97), "validation": np.array([100])}
        l0_truth = np.array([[1., 2.]], dtype=np.float32)
        runtime = tmp_path / "runtime/seed_157"
        def __init__(self, seed):
            pass
        def ids(self, indices):
            return [str(i) for i in indices]
        def new_store(self):
            return Store()
        def fit(self, method, r, labeled, truth, runtime):
            assert len(labeled) == len(truth) == 33 and r == 1
            calls.append("fit")
    monkeypatch.setattr(study, "Context", Context)
    monkeypatch.setattr(study, "prediction_freeze", lambda *args: None)
    assert study.advance(157, method)["active_label_count"] == 33
    assert calls == ["after_acquisition_fit", "initial_fit", "fit"]
    feedback = read(d / "feedback.json")
    assert all(r["source"] == "LLM" for r in feedback["records"])
    _, _, _, history = study.state(Context(157), method, 1)
    assert len(history) == 1
    calls.clear()
    study.advance(157, method)
    assert calls == ["initial_fit", "fit"]
    receipt = read(d / "label_access_receipt.json")
    receipt["audit"]["acquisitions_frozen"] = False
    (d / "label_access_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="receipt"):
        study.state(Context(157), method, 1)


def test_prompt_objective_and_no_hidden_strategy_quota():
    assert "Numerical signals are evidence, not mandatory rules" in SYSTEM_PROMPT
    assert "The goal is NOT to optimize chromatographic performance" in SYSTEM_PROMPT
    assert "These are outcomes of experiments you previously selected" in SYSTEM_PROMPT
    assert "V2 - V1 is the elution" in SYSTEM_PROMPT
