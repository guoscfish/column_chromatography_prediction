"""Recovery keeps immutable evidence and budgets across a killed process."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest
from test_scientist_v3_2 import setup, query, plan, config
from src.qgeognn_al.active_learning_v3_2 import transport
from src.qgeognn_al.active_learning_v3_2.artifacts import read, verify_audit

SCRIPTS = Path(__file__).resolve().parents[2]/'scripts/studies'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('process_recovery', SCRIPTS/'resume_qgeognn_v3_2_interrupted.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def interrupted(directory, monkeypatch, attempts=3, errors=0):
    c, p, ledger = setup()
    count = 0
    monkeypatch.setattr(transport.time, 'sleep', lambda _: None)
    def call(*_):
        nonlocal count
        count += 1
        if errors and count <= errors:
            return 'malformed', {}
        if not errors and count <= 2:
            return json.dumps(query({'offset': (count-1)*24})), {}
        before = errors or 2
        if count-before < attempts:
            raise ConnectionError()
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        transport.run_selector(p, c, ledger, directory, config(), call=call)
    return p, ledger


def test_dry_replay_changes_no_artifacts_or_calls(tmp_path, monkeypatch):
    p, ledger = interrupted(tmp_path, monkeypatch)
    before = {f.name: f.read_bytes() for f in tmp_path.iterdir()}
    c, _, _ = setup()
    result = recovery.recover_selector(tmp_path, p, c, ledger, config(), call=lambda *_: pytest.fail('network during dry replay'))
    assert result['remaining_attempts'] == 1
    assert result['accepted_queries'] == 2
    assert before == {f.name: f.read_bytes() for f in tmp_path.iterdir()}


def test_resume_preserves_prefix_counts_and_uses_only_remaining_attempt(tmp_path, monkeypatch):
    p, ledger = interrupted(tmp_path, monkeypatch)
    prefix = (tmp_path/'audit.jsonl').read_bytes()
    receipts = {f.name: f.read_bytes() for f in tmp_path.glob('turn_*.json')}
    c, _, _ = setup()
    calls = []
    def call(*_):
        calls.append(1)
        return json.dumps(plan(c, p)), {}
    result, _ = recovery.recover_selector(tmp_path, p, c, ledger, config(), execute=True, call=call)
    assert len(result['choices']) == 32 and len(calls) == 1
    assert (tmp_path/'audit.jsonl').read_bytes().startswith(prefix)
    assert all((tmp_path/name).read_bytes() == value for name, value in receipts.items())
    audit = verify_audit(tmp_path/'audit.jsonl')
    assert len([e for e in audit if e['event_type']=='query']) == 2
    assert [e['payload']['attempt'] for e in audit if e['event_type']=='transport_attempt' and e['turn']==2] == [1,2,3,4]
    assert read(tmp_path/'interaction_summary.json')['total_llm_calls'] == 3
    assert read(tmp_path/'turn_02.json')['provenance']['transport_attempts'] == 4


def test_last_retry_failure_stops_without_fifth_attempt(tmp_path, monkeypatch):
    p, ledger = interrupted(tmp_path, monkeypatch)
    c, _, _ = setup()
    calls = []
    def call(*_):
        calls.append(1)
        raise ConnectionError()
    with pytest.raises(RuntimeError, match='STOP: transport failure'):
        recovery.recover_selector(tmp_path, p, c, ledger, config(), execute=True, call=call)
    assert calls == [1]
    assert read(tmp_path/'failure_02.json')['attempts'] == 4
    assert not (tmp_path/'selection.json').exists()


def test_repair_budget_is_not_reset(tmp_path, monkeypatch):
    p, ledger = interrupted(tmp_path, monkeypatch, attempts=1, errors=4)
    c, _, _ = setup()
    with pytest.raises(RuntimeError, match='repair budget exceeded'):
        recovery.recover_selector(tmp_path, p, c, ledger, config(), execute=True, call=lambda *_: ('still malformed', {}))
    events = verify_audit(tmp_path/'audit.jsonl')
    assert sum(e['event_type']=='response_validation_error' for e in events) == 5
    assert not (tmp_path/'selection.json').exists()


def test_fourth_inflight_attempt_cannot_be_retried(tmp_path, monkeypatch):
    p, ledger = interrupted(tmp_path, monkeypatch, attempts=4)
    c, _, _ = setup()
    with pytest.raises(RuntimeError, match='no transport attempts remain'):
        recovery.recover_selector(tmp_path, p, c, ledger, config(), execute=True, call=lambda *_: pytest.fail('extra retry'))


def test_replay_detects_response_artifact_drift_before_network(tmp_path, monkeypatch):
    p, ledger = interrupted(tmp_path, monkeypatch)
    value = read(tmp_path/'turn_00.json')
    value['answer'] = '{}'
    (tmp_path/'turn_00.json').write_text(json.dumps(value))
    c, _, _ = setup()
    with pytest.raises(RuntimeError, match='replay mismatch'):
        recovery.recover_selector(tmp_path, p, c, ledger, config(), execute=True, call=lambda *_: pytest.fail('network before verification'))
