#!/usr/bin/env python3
"""Explicit process-interruption recovery; replay frozen responses, never reset budgets.

The frozen selector is executed unchanged in a temporary directory. Every historic
audit event, write-once artifact and outgoing request must match before networking
is enabled. The original audit prefix stays byte-for-byte intact. An in-flight
request with unknown outcome consumes its attempt; only remaining retries may run.
This does not recover terminal errors, exhausted repairs, or protocol drift.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas
from src.qgeognn_al.active_learning_v3_2 import study, protocol, transport
from src.qgeognn_al.active_learning_v3_2.artifacts import Audit, digest, file_hash, once, read, verify_audit, operation_lock
from run_qgeognn_v3_2_loop import summarize_validation


class ReplayComplete(Exception):
    pass


class ReplayAudit:
    def __init__(self, directory, packet):
        self.events = verify_audit(Path(directory)/'audit.jsonl')
        self.cursor = 0
        self.writer = Audit(directory, packet)
        self.live = False

    def append(self, event_type, payload, *, turn, request_sha256):
        actual = dict(event_type=event_type, payload=payload, turn=turn, request_sha256=request_sha256)
        if self.cursor < len(self.events):
            expected = self.events[self.cursor]
            if any(expected[k] != v for k, v in actual.items()):
                raise RuntimeError('recovery audit replay mismatch')
            self.cursor += 1
        elif self.live:
            self.writer.append(**actual)
        else:
            raise RuntimeError('recovery reached an unexpected audit boundary')

    def replay_transport(self):
        while self.cursor < len(self.events) and self.events[self.cursor]['event_type'].startswith('transport_'):
            e = self.events[self.cursor]
            self.append(e['event_type'], e['payload'], turn=e['turn'], request_sha256=e['request_sha256'])


def recover_selector(directory, packet, catalog, ledger, config, *, execute=False, call=None, guard=None):
    directory = Path(directory)
    audit = ReplayAudit(directory, packet)
    events = audit.events
    if not events or events[-1]['event_type'] != 'transport_attempt':
        raise RuntimeError('recovery requires an unfinished transport attempt')
    if any((directory/name).exists() for name in ('batch_freeze.json', 'selection.json', 'recovery_started.json')):
        raise RuntimeError('recovery already used or selection already exists')
    if any(e['event_type'] in ('transport_failure', 'model_call_budget_exhausted', 'context_budget_failure') for e in events):
        raise RuntimeError('terminal failure cannot be recovered')
    pending = events[-1]
    pending_turn = pending['turn']
    consumed = pending['payload']['attempt']
    maximum = config['retry_policy']['max_retries']+1
    if consumed >= maximum:
        raise RuntimeError('no transport attempts remain after interruption')
    if (directory/f'turn_{pending_turn:02d}.json').exists():
        raise RuntimeError('interrupted call already has a response')
    baseline = {p.name: file_hash(p) for p in directory.iterdir() if p.is_file() and p.name != '.operation.lock'}
    original_retry = transport.call_with_retries
    call = call or transport.responses_call
    replayed = []

    def relay(actual_call, messages, cfg, scratch, actual_audit, turn, request_hash, actual_guard=None):
        if turn < pending_turn:
            receipt = read(directory/f'turn_{turn:02d}.json')
            if request_hash != receipt['request_sha256']:
                raise RuntimeError('recovery request hash mismatch')
            audit.replay_transport()
            replayed.append(turn)
            return receipt['answer'], receipt['provenance']
        if turn > pending_turn:
            return original_retry(actual_call, messages, cfg, scratch, actual_audit, turn, request_hash, actual_guard)
        if request_hash != pending['request_sha256'] or replayed != list(range(pending_turn)):
            raise RuntimeError('recovery pending request or completed turn mismatch')
        audit.replay_transport()
        if audit.cursor != len(events):
            raise RuntimeError('recovery did not consume the complete original audit')
        for name, expected in baseline.items():
            if file_hash(directory/name) != expected:
                raise RuntimeError('interrupted artifacts changed during replay')
        if not execute:
            raise ReplayComplete()
        if actual_guard:
            actual_guard()
        once(directory/'recovery_started.json', {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'recovery_script_sha256': file_hash(__file__), 'original_artifact_hashes': baseline,
            'pending_turn': turn, 'consumed_transport_attempts': consumed,
            'remaining_transport_attempts': maximum-consumed, 'request_sha256': request_hash,
            'completed_responses_reused': replayed, 'scientific_budgets_reset': False,
            'unknown_inflight_response_discarded': True,
        })
        audit.live = True
        interrupted = {'attempt': consumed, 'outcome': 'unknown_process_interruption',
                       'counts_against_transport_budget': True}
        once(directory/f'transport_attempt_{turn:02d}_{consumed:02d}.json', interrupted)
        audit.append('transport_attempt_interrupted', interrupted, turn=turn, request_sha256=request_hash)
        for attempt in range(consumed+1, maximum+1):
            if actual_guard:
                actual_guard()
            audit.append('transport_attempt', {'attempt': attempt, 'max_attempts': maximum}, turn=turn, request_sha256=request_hash)
            try:
                answer, provenance = actual_call(messages, cfg)
            except Exception as error:
                detail = transport.transport_error(error)
                retry = detail['retryable'] and attempt < maximum
                delay = cfg['retry_policy']['backoff_seconds'][attempt-1] if retry else 0
                receipt = {**transport.failure_receipt(error, request_hash, cfg, attempts=attempt),
                           'will_retry': retry, 'retry_delay_seconds': delay}
                transport.once(scratch/f'transport_attempt_{turn:02d}_{attempt:02d}.json', receipt)
                audit.append('transport_attempt_failed', receipt, turn=turn, request_sha256=request_hash)
                if not retry:
                    transport.once(scratch/f'failure_{turn:02d}.json', receipt)
                    audit.append('transport_failure', receipt, turn=turn, request_sha256=request_hash)
                    raise RuntimeError('STOP: transport failure; see sanitized failure receipt') from None
                transport.time.sleep(delay)
            else:
                audit.append('transport_attempt_succeeded', {'attempt': attempt}, turn=turn, request_sha256=request_hash)
                return answer, {**provenance, 'transport_attempts': attempt, 'transport_retries': attempt-1}

    with tempfile.TemporaryDirectory(prefix='scientist-exact-replay-') as tmp:
        scratch = Path(tmp)
        def mirrored_once(path, value):
            destination = directory/Path(path).relative_to(scratch)
            if destination.exists() and read(destination) != value:
                raise RuntimeError('recovery immutable artifact mismatch: '+destination.name)
            if audit.live:
                once(destination, value)
            elif not destination.exists():
                raise RuntimeError('recovery replay artifact missing: '+destination.name)
            once(path, value)
        with patch.object(transport, 'Audit', lambda *_: audit), patch.object(transport, 'once', mirrored_once), patch.object(transport, 'call_with_retries', relay):
            try:
                return transport.run_selector(packet, catalog, ledger, scratch, config, call=call, guard=guard)
            except ReplayComplete:
                return {'status': 'EXACT_REPLAY_VERIFIED', 'completed_responses': len(replayed),
                        'pending_turn': pending_turn, 'consumed_attempts': consumed,
                        'remaining_attempts': maximum-consumed, 'new_model_calls': 0,
                        'accepted_queries': len(catalog.queries), 'invalid_queries': len(catalog.invalid_queries)}


def recover(root, *, execute=False):
    root = Path(root)
    seed, method = 157, study.METHODS[0]
    with operation_lock(root):
        protocol.validate(root)
        current = study.state(root, seed, method)
        if current['next_action'] != 'STOP_failed_or_interrupted_selector':
            raise RuntimeError('study is not at an interrupted selection')
        r = current['round']
        d = study.directory(root, seed, method, r)
        packet, raw, ledger = (read(d/name) for name in ('input.json', 'catalog.json', 'ledger_before.json'))
        catalog = study.ReadOnlyCatalog(**raw, ledger=ledger)
        catalog.initial_cards()
        result = recover_selector(d, packet, catalog, ledger, read(root/'protocol.json')['transport'],
                                  execute=execute, guard=lambda: protocol.validate(root))
        if not execute:
            return result
        selection, merged = result
        return study.freeze_batch(root, seed, method, r, selection, merged, catalog)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, default=study.STUDY)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(recover(args.study_dir)))
        return 0
    execution = args.study_dir/'execution/process_recovery'
    with operation_lock(execution):
        if (execution/'started.json').exists():
            raise RuntimeError('explicit recovery already started')
        verified = recover(args.study_dir)
        once(execution/'started.json', {'verified_replay': verified, 'script_sha256': file_hash(__file__),
             'protocol_sha256': file_hash(args.study_dir/'protocol.json'),
             'timestamp': datetime.now(timezone.utc).isoformat(), 'authorization': 'User requested continuing the six-round run.'})
        os.environ['SCIENTIST_API_KEY'] = (Path.home()/'.config/qgeognn-scientist/token4research.api-key').read_text().strip()
        outcome = {'target_labels': 525}
        try:
            print(json.dumps(recover(args.study_dir, execute=True)), flush=True)
            for _ in range(18):
                protocol.validate(args.study_dir)
                state = study.state(args.study_dir, 157, study.METHODS[0])
                action, r = state['next_action'], state['round']
                if action == 'complete_no_test_evaluation':
                    outcome.update(status='COMPLETED', completed_rounds=r, active_label_count=state['active_label_count'])
                    break
                if action not in ('stage', 'select', 'advance'):
                    raise RuntimeError('canonical state does not permit continuation')
                print(json.dumps({'action': action, 'round': r, 'timestamp': datetime.now(timezone.utc).isoformat()}), flush=True)
                result = getattr(study, action)(157, study.METHODS[0], r, root=args.study_dir)
                print(json.dumps(result), flush=True)
                if action == 'advance':
                    from src.qgeognn_al.active_learning_v3_2.report import summarize
                    summarize(args.study_dir, round_index=r)
            else:
                raise RuntimeError('controller transition bound exhausted')
        except Exception as error:
            allowed = ('STOP: response validation repair budget exceeded', 'STOP: query validation repair budget exceeded',
                       'STOP: transport failure; see sanitized failure receipt')
            outcome.update(status='STOPPED', error_type=type(error).__name__,
                           diagnostic=str(error) if str(error) in allowed else 'See immutable phase artifacts; no automatic recovery.')
        finally:
            os.environ.pop('SCIENTIST_API_KEY', None)
        outcome['timestamp'] = datetime.now(timezone.utc).isoformat()
        once(execution/'result.json', outcome)
        summarize_validation(args.study_dir, 157, study.METHODS[0])
        print(json.dumps(outcome), flush=True)
        return 0 if outcome['status'] == 'COMPLETED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
