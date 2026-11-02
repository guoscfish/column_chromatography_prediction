#!/usr/bin/env python3
"""Run the existing frozen Scientist protocol to its registered label budget."""
import argparse
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas
from src.qgeognn_al.active_learning_v3_2 import study, protocol
from src.qgeognn_al.active_learning_v3_2.artifacts import file_hash, once, read, operation_lock


def now():
    return datetime.now(timezone.utc).isoformat()


def run_loop(root, seed, method, *, api=study, validate=protocol.validate):
    root = Path(root)
    execution = root/'execution/continuous_loop'
    with operation_lock(execution):
        if (execution/'started.json').exists():
            raise RuntimeError('continuous loop already started; inspect its outcome before any restart')
        registered = read(root/'protocol.json')
        budgets = registered['budgets']
        validate(root)
        initial = api.state(root, seed, method)
        once(execution/'started.json', {'started_at': now(), 'seed': seed, 'method': method,
             'start_round': initial['round'], 'start_labels': initial['active_label_count'],
             'registered_budgets': budgets, 'target_labels': budgets[-1],
             'protocol_sha256': file_hash(root/'protocol.json'),
             'controller_sha256': file_hash(Path(__file__)), 'strategy_changed': False})
        def event(value):
            value = {'timestamp': now(), **value}
            with (execution/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(value, ensure_ascii=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            print(json.dumps(value, ensure_ascii=False), flush=True)
        outcome = {'seed': seed, 'method': method, 'target_labels': budgets[-1]}
        try:
            for _ in range(3*(len(budgets)-1)+1):
                validate(root)
                current = api.state(root, seed, method)
                r, count, action = (current[k] for k in ('round', 'active_label_count', 'next_action'))
                if action == 'complete_no_test_evaluation':
                    if r != len(budgets)-1 or count != budgets[-1]:
                        raise RuntimeError('terminal budget mismatch')
                    outcome.update(status='COMPLETED', completed_rounds=r, active_label_count=count,
                                   test_evaluation=False)
                    break
                if action not in ('stage', 'select', 'advance'):
                    raise RuntimeError('canonical state does not permit automatic continuation')
                if not 0 <= r < len(budgets)-1 or count != budgets[r]:
                    raise RuntimeError('round budget mismatch')
                event({'status': 'ACTION_STARTED', 'round': r, 'active_label_count': count, 'action': action})
                result = getattr(api, action)(seed, method, r, root=root)
                after = api.state(root, seed, method)
                if all(after[k] == current[k] for k in ('round', 'active_label_count', 'next_action')):
                    raise RuntimeError('action made no canonical progress')
                if action == 'advance' and api is study:
                    from src.qgeognn_al.active_learning_v3_2.report import summarize
                    summarize(root, round_index=r)
                event({'status': 'ACTION_COMPLETED', 'round': r, 'action': action, 'result': result,
                       'active_label_count': after['active_label_count'], 'next_action': after['next_action']})
            else:
                raise RuntimeError('controller transition bound exhausted')
        except Exception as error:
            known = ('query budget exhausted', 'model call budget exhausted; no selection accepted',
                     'STOP: transport failure; see sanitized failure receipt',
                     'canonical state does not permit automatic continuation', 'round budget mismatch',
                     'terminal budget mismatch', 'action made no canonical progress',
                     'controller transition bound exhausted')
            outcome.update(status='STOPPED', error_type=type(error).__name__,
                           diagnostic=str(error) if str(error) in known else
                           'See immutable phase artifacts; no source repair or automatic operation restart.')
        outcome['completed_at'] = now()
        once(execution/'result.json', outcome)
        event(outcome)
        return outcome


def summarize_validation(root, seed, method):
    """Read already stored fit metrics only; never open labels or feed scores to the selector."""
    root = Path(root)
    frozen = read(root/'protocol.json')
    baseline = read(frozen['initial'][str(seed)]['checkpoint_audit'])
    fits = [(0, baseline)]
    for r in range(len(frozen['budgets'])-1):
        completion = root/f'selections/seed_{seed}/{method}/round_{r:02d}/training_complete.json'
        if not completion.exists():
            break
        trained = read(completion)
        fits.append((r+1, read(Path(trained['checkpoint']).parent/'fit_audit.json')))
    points = []
    for r, fit in fits:
        if (fit['validation_ids_hash'] != baseline['validation_ids_hash'] or
                fit['initialization_hash'] != baseline['initialization_hash'] or
                fit['train_rows'] != frozen['budgets'][r] or
                fit['test_labels_used_for_fit_or_checkpoint_selection'] != 0):
            raise RuntimeError('learning curve fit comparability mismatch')
        points.append({'completed_rounds': r, 'training_labels': fit['train_rows'],
                       'validation_combined_normalized_rmse': fit['best_validation_combined_normalized_rmse']})
    first, last = (points[i]['validation_combined_normalized_rmse'] for i in (0, -1))
    result = {'seed': seed, 'method': method, 'metric_source': 'existing fit audits; fixed checkpoint validation set',
              'lower_is_better': True, 'points': points,
              'relative_reduction_from_L333_percent': 100*(first-last)/first if first else None,
              'test_truth_reads': 0, 'selector_receives_these_metrics': False,
              'limitation': 'One seed; checkpoint validation scores are not an independent test or a matched acquisition-strategy comparison.'}
    target = root/'execution/continuous_loop'
    once(target/'validation_learning_curve.json', result)
    lines = ['# Scientist V3.2 validation learning curve', '',
             '| Completed rounds | Training labels | Validation normalized RMSE |',
             '| --- | --- | --- |']
    lines += [f"| {p['completed_rounds']} | {p['training_labels']} | {p['validation_combined_normalized_rmse']:.6f} |" for p in points]
    lines += ['', result['limitation'], '', 'No test truth was read. Metrics were not supplied to the selector.']
    with (target/'validation_learning_curve.md').open('x') as stream:
        stream.write('\n'.join(lines)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, default=study.STUDY)
    parser.add_argument('--seed', type=int, choices=study.SEEDS, required=True)
    parser.add_argument('--method', choices=study.METHODS, required=True)
    parser.add_argument('--api-key-file', type=Path,
                        default=Path.home()/'.config/qgeognn-scientist/token4research.api-key')
    args = parser.parse_args()
    key = args.api_key_file.read_text().strip()
    if not key or len(key.splitlines()) != 1:
        parser.error('API key file must contain one nonempty line')
    os.environ['SCIENTIST_API_KEY'] = key
    try:
        result = run_loop(args.study_dir, args.seed, args.method)
    finally:
        os.environ.pop('SCIENTIST_API_KEY', None)
    summarize_validation(args.study_dir, args.seed, args.method)
    return 0 if result['status'] == 'COMPLETED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
