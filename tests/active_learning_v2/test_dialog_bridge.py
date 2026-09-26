"""Synthetic relay tests, not scientific experiment results."""
import json
import pytest
from src.qgeognn_al.active_learning_v2 import dialog_bridge as bridge
from src.qgeognn_al.active_learning_v2.full_pool_selector import ReadOnlyCatalog


def raw(final, extra=None):
    rows = [{"type": "turn_context", "payload": {"model": "gpt-6-sol", "effort": "high"}},
            {"type": "response_item", "payload": {"type": "message", "role": "assistant",
             "phase": "final_answer", "content": [{"text": json.dumps(final)}]}}]
    if extra:
        rows.append(extra)
    return ("\n".join(json.dumps(row) for row in rows)+"\n").encode()


def test_native_calls_and_model_drift_rejected():
    assert bridge.audit_bytes(raw({}))["native_tool_calls"] == 0
    for kind in ("function_call", "custom_tool_call", "web_search_call"):
        with pytest.raises(RuntimeError, match="native tool"):
            bridge.audit_bytes(raw({}, {"type": "response_item", "payload": {"type": kind}}))
    with pytest.raises(RuntimeError, match="model"):
        bridge.audit_bytes(raw({}).replace(b'gpt-6-sol', b'other-model'))


def test_relay_resume_acceptance_and_original_evidence(tmp_path, monkeypatch):
    def card(i, observed=False):
        row = {"id": str(i), "smiles": "CC", "conditions": {}}
        row.update({"V1_ml": 1, "V2_ml": 2} if observed else {"pred_V1_ml": 1, "pred_V2_ml": 2})
        return row
    cat = ReadOnlyCatalog([card(i) for i in range(17, 161)], [card(0, True)],
                          [card(i) for i in range(1, 17)], [str(i) for i in range(17, 145)])
    packet = {"seed": 157, "method": "LLM", "round": 0, "packet_sha256": "synthetic"}
    bridge.once(tmp_path / "input.json", packet)
    bridge.export_catalog(tmp_path, cat)
    bridge.once(tmp_path / "dialog_binding.json", {"thread_id": "synthetic", "start_final_count": 0})
    rollout = tmp_path / "rollout.jsonl"
    monkeypatch.setattr(bridge, "rollout_path", lambda _: rollout)
    rollout.write_bytes(raw({"type": "query", "queries": [{"operation": "get_candidates", "args": {"ids": ["145"]}}]}))
    first = bridge.process(tmp_path)
    assert bridge.process(tmp_path) == first
    assert len(list(tmp_path.glob("relay_*.json"))) == 1
    choice = {"type": "selection", "packet_sha256": "synthetic", "choices": [
        {"id": str(i), "reason": "synthetic contrast"} for i in range(145, 161)], "hypotheses": []}
    with rollout.open('ab') as handle:
        handle.write(raw(choice))
    assert bridge.process(tmp_path)["outside_128"] == 16
    assert bridge.process(tmp_path)["status"] == "ALREADY_ACCEPTED"
    assert bridge.accepted_selection(tmp_path, bridge.load_catalog(tmp_path), packet)["choices"] == choice["choices"]
    with pytest.raises(RuntimeError, match="binding"):
        bridge.accepted_selection(tmp_path, bridge.load_catalog(tmp_path), dict(packet, packet_sha256="other"))
    rollout.write_bytes(rollout.read_bytes().replace(b'synthetic contrast', b'changed contrast!!'))
    with pytest.raises(RuntimeError, match="original"):
        bridge.accepted_selection(tmp_path, bridge.load_catalog(tmp_path), packet)
