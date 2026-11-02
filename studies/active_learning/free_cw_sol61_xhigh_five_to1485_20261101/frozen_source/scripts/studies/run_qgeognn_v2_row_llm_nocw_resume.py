#!/usr/bin/env python3
"""Append-only, user-authorized correction continuation for two stopped NoCW rounds."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.studies import run_qgeognn_v2_row_llm_nocw_stream as stream

base, retry, s = stream.base, stream.retry, stream.s
STUDY = stream.STUDY
SCRIPT = Path(__file__).resolve()
TEST = s.ROOT / 'tests/active_learning_v2/test_nocw_correction_resume.py'
POLICY = STUDY / 'correction_resume_amendment_20261030_v2.json'
GRANTS = {887: 3, 6101: 8}  # Seed -> failed acquisition round; not a global budget increase.
ORIGINAL_RUN_SELECTION = s.run_selection


def prepare():
    valid = stream.validate()
    if POLICY.exists():
        return validate()
    grants = []
    for seed, r in GRANTS.items():
        if retry.seed_busy(seed) or stream.unresolved(seed):
            raise RuntimeError('stopped seed has a worker or unresolved request')
        progress = s.read(STUDY / f'progress/seed_{seed}.json')
        if progress.get('phase') != 'FAILED_CLOSED' or progress.get('reason') != 'selection remained invalid after bounded corrections; fail closed':
            raise RuntimeError('unexpected failure; correction grant does not apply')
        directory = s.selection_directory(seed, 'llm32_nocw', r)
        if any((directory / name).exists() for name in ('selection.json', 'batch_freeze.json', 'feedback.json')):
            raise RuntimeError('failed round already accepted or revealed')
        if sorted(p.name for p in directory.glob('rejected_*.json')) != ['rejected_00.json', 'rejected_01.json', 'rejected_02.json']:
            raise RuntimeError('grant requires exactly the existing three rejected answers')
        prefix = {str(p.relative_to(STUDY)): s.sha256_file(p) for p in directory.iterdir() if p.is_file()}
        grants.append({'seed': seed, 'round': r, 'previous_answer_limit': 3, 'additional_answers': 3,
                       'new_answer_limit': 6, 'preserved_prefix': prefix})
    record = {'version': 'nocw_two_round_correction_resume_v1', 'authorized_utc': retry.now(),
              'user_authorization': '帮我继续跑（在已说明两seed纠错耗尽及此前每seed追加3次方案后）',
              'base_protocol_sha256': valid['protocol_sha256'],
              'stream_transport_amendment_sha256': s.sha256_file(stream.POLICY),
              'grants': grants, 'future_round_answer_limit': 3,
              'preserve_all_existing_requests_attempts_started_answers_and_fits': True,
              'model_endpoint_effort_key_mapping_candidate_pool_validation': 'unchanged',
              'network_retry': 'existing persistent retry amendment; no new probes',
              'code_hashes': {str(SCRIPT.relative_to(s.ROOT)): s.sha256_file(SCRIPT),
                              str(TEST.relative_to(s.ROOT)): s.sha256_file(TEST)}}
    draft = STUDY / 'correction_resume_amendment_20261030.json'
    if draft.exists():
        record['supersedes_unlaunched_draft_sha256'] = s.sha256_file(draft)
        record['prelaunch_revision'] = 'CLI validate initializes five-seed context; draft retained, no calls launched under draft'
    base.once(POLICY, record)
    return record


def validate():
    result = stream.validate()
    policy = s.read(POLICY)
    if (policy['base_protocol_sha256'] != result['protocol_sha256'] or
            policy['stream_transport_amendment_sha256'] != s.sha256_file(stream.POLICY)):
        raise RuntimeError('continuation bound to different frozen protocol')
    if {(g['seed'], g['round'], g['previous_answer_limit'], g['new_answer_limit']) for g in policy['grants']} != {(887, 3, 3, 6), (6101, 8, 3, 6)}:
        raise RuntimeError('correction grant drift')
    for name, digest in policy['code_hashes'].items():
        if s.sha256_file(s.ROOT / name) != digest:
            raise RuntimeError('continuation source drift')
    for grant in policy['grants']:
        for name, digest in grant['preserved_prefix'].items():
            if s.sha256_file(STUDY / name) != digest:
                raise RuntimeError('original failed-round ledger changed')
    return result


def activate(seed=None):
    stream.activate(seed)
    s.validate = validate
    s.run_selection = run_selection


def answer_limit(directory, config):
    limit = config['selection_attempts']
    if limit != 3:
        raise RuntimeError('frozen answer limit changed')
    for grant in s.read(POLICY)['grants']:
        expected = STUDY / f"selections/seed_{grant['seed']}/llm32_nocw/round_{grant['round']:02d}"
        if Path(directory).resolve() == expected.resolve():
            return grant['new_answer_limit']
    return limit


def run_selection(packet, catalog, directory, variant, config, call):
    directory = Path(directory)
    limit = answer_limit(directory, config)
    if limit == config['selection_attempts']:
        return ORIGINAL_RUN_SELECTION(packet, catalog, directory, variant, config, call)
    # Same serialization and strict validation as the frozen implementation.
    # Only the loop bound on the two explicitly authorized rounds differs.
    if any(packet.get(k) != v for k, v in s.encode_catalog(catalog).items()):
        raise ValueError('packet differs from complete catalog')
    if (packet.get('selection_count') != 32 or catalog.pending or
            packet.get('candidate_count') != len(catalog.candidates) or
            packet.get('observed_count') != len(catalog.observed) or
            packet.get('active_label_count') != len(catalog.observed)):
        raise ValueError('full Free32 packet count mismatch')
    if packet['packet_sha256'] != s.stable_hash({k: v for k, v in packet.items() if k != 'packet_sha256'}):
        raise ValueError('packet binding mismatch')
    messages = [{'role': 'system', 'content': s.prompt(variant)},
                {'role': 'user', 'content': json.dumps(packet, ensure_ascii=False, separators=(',', ':'), allow_nan=False)}]
    for turn in range(limit):
        if variant == 'no_cw':
            s.audit_no_cw(messages)
        s.once(directory / f'request_{turn:02d}.json', {'messages': messages, 'config': config})
        saved = s.request_with_retries(messages, config, directory, turn, call)
        s.once(directory / f'turn_{turn:02d}.json', saved)
        try:
            result = s.validate_plan(json.loads(saved['answer']), catalog, packet, variant)
        except (ValueError, TypeError, KeyError) as error:
            correction = {'error': str(error)[:1000], 'instruction':
                'Return corrected final JSON from the SAME complete pool. Exactly 32 distinct candidate IDs. '
                'Measured evidence belongs only in typed observed-ID lists. No queries or tools.'}
            s.once(directory / f'rejected_{turn:02d}.json', correction)
            messages += [{'role': 'assistant', 'content': saved['answer']},
                         {'role': 'user', 'content': json.dumps(correction)}]
            continue
        s.once(directory / 'selection.json', result)
        return result
    raise RuntimeError('selection remained invalid after authorized six-answer limit; fail closed')


def resume_round(seed):
    """Audit and skip completed fits; never train or reselect an existing prefix."""
    for r in range(len(s.BUDGETS)-1):
        directory = s.selection_directory(seed, 'llm32_nocw', r)
        next_dir = STUDY / f'runtime/seed_{seed}/llm32_nocw/round_{r+1:02d}'
        frozen = next_dir / 'prediction_freeze.json'
        if not frozen.exists():
            return r
        s.audit_batch(directory)
        fit = s.read(next_dir / 'model/fit_audit.json')
        freeze = s.read(frozen)
        if (fit['train_rows'] != s.BUDGETS[r+1] or fit['checkpoint_sha256'] != freeze['checkpoint_sha256'] or
                s.sha256_file(Path(freeze['checkpoint_path'])) != freeze['checkpoint_sha256'] or
                s.sha256_file(Path(freeze['prediction_path'])) != freeze['prediction_sha256'] or
                fit['test_labels_used_for_fit_or_checkpoint_selection'] != 0):
            raise RuntimeError('completed fit lineage mismatch before resume')
        feedback = s.read(directory / 'feedback.json')
        if feedback['batch_freeze_sha256'] != s.sha256_file(directory / 'batch_freeze.json'):
            raise RuntimeError('completed feedback lineage mismatch')
    return len(s.BUDGETS)-1


def worker(seed):
    activate(seed)
    validate()
    s.require_authorization()
    base.credential_audit()
    os.environ.pop('SCIENTIST_API_KEY', None)
    os.environ['SCIENTIST_API_KEY_FILE'] = str(base.KEY_FILES[seed])
    base.torch.set_num_threads(2)
    with base.BASE_EXCLUSIVE_LOCK(STUDY / f'locks/worker_seed_{seed}.lock'):
        try:
            if stream.unresolved(seed):
                raise RuntimeError('unresolved provider request requires reconciliation')
            first = resume_round(seed)
            base.once(STUDY / f'progress/resume_prefix_seed_{seed}.json', {
                'seed': seed, 'resume_round': first, 'completed_budget': s.BUDGETS[first],
                'completed_fits_audited_and_skipped': first, 'new_model_calls_during_audit': 0,
                'amendment_sha256': s.sha256_file(POLICY)})
            for r in range(first, len(s.BUDGETS)-1):
                base.status_record(seed, 'SELECTING', round=r, arm='llm32_nocw', current_budget=s.BUDGETS[r])
                s.select(seed, 'llm32_nocw', r, allow_paid_llm=True, allow_feature_compute=r > 0)
                base.status_record(seed, 'ADVANCING', round=r, arm='llm32_nocw', next_budget=s.BUDGETS[r+1])
                s.advance(seed, 'llm32_nocw', r, allow_reveal=True, allow_training=True)
            base.status_record(seed, 'COMPLETE', final_budget=1005, arms=['llm32_nocw'])
        except BaseException as error:
            base.status_record(seed, 'FAILED_CLOSED', error_type=type(error).__name__, reason=str(error)[:700])
            raise


def supervise():
    activate()
    s.require_authorization()
    children = {}
    with base.BASE_EXCLUSIVE_LOCK(STUDY / 'locks/stream_supervisor.lock'):
        with base.BASE_EXCLUSIVE_LOCK(STUDY / 'locks/persistent_supervisor.lock'):
            last_validation = 0
            while True:
                if time.monotonic() - last_validation >= 15:
                    validate()
                    last_validation = time.monotonic()
                rows = []
                for seed in base.SEEDS:
                    progress = s.read(STUDY / f'progress/seed_{seed}.json')
                    row = {k: progress.get(k) for k in ('seed', 'pid', 'round', 'phase', 'current_budget')}
                    busy = retry.seed_busy(seed)
                    if progress['phase'] == 'COMPLETE':
                        row['supervision'] = 'COMPLETE'
                    elif seed in children:
                        child = children[seed]
                        row.update(child_pid=child.pid, supervision='STREAM_WORKER_RUNNING' if child.poll() is None else 'NEEDS_RECONCILIATION', exit_code=child.returncode)
                    elif busy:
                        # Existing workers and their inflight requests are untouched.
                        info = stream.process_info(progress['pid'])
                        if (str(stream.SCRIPT) not in info or f'--seed {seed}' not in info or
                                ' worker ' not in info):
                            raise RuntimeError('existing worker process identity mismatch')
                        row['supervision'] = 'STREAM_WORKER_RUNNING'
                    elif seed in GRANTS and not (STUDY / f'progress/resume_worker_seed_{seed}.json').exists() and not stream.unresolved(seed):
                        if progress['phase'] != 'FAILED_CLOSED':
                            raise RuntimeError('unexpected state for authorized stopped seed')
                        env = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
                        env.pop('SCIENTIST_API_KEY', None)
                        with (STUDY / f'progress/seed_{seed}_resume.log').open('ab') as output:
                            child = subprocess.Popen([sys.executable, str(SCRIPT), 'worker', '--seed', str(seed), '--allow-paid-llm', '--allow-training', '--allow-label-reveal'],
                                                     cwd=s.ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                        children[seed] = child
                        base.once(STUDY / f'progress/resume_worker_seed_{seed}.json', {'seed': seed, 'pid': child.pid, 'started_utc': retry.now(), 'amendment_sha256': s.sha256_file(POLICY)})
                        row.update(child_pid=child.pid, supervision='STREAM_WORKER_DISPATCHED')
                    else:
                        row['supervision'] = 'NEEDS_RECONCILIATION'
                    rows.append(row)
                base.atomic_json(STUDY / 'progress/stream_supervisor.json', {'pid': os.getpid(), 'updated_utc': retry.now(), 'workers': rows,
                    'amendment_sha256': s.sha256_file(stream.POLICY), 'correction_resume_amendment_sha256': s.sha256_file(POLICY)})
                if all(r['supervision'] == 'COMPLETE' for r in rows):
                    return {'status': 'ALL_FIVE_COMPLETE'}
                time.sleep(2)


def launch():
    activate()
    validate()
    s.require_authorization()
    base.credential_audit()
    with base.BASE_EXCLUSIVE_LOCK(STUDY / 'locks/correction_resume_launch.lock'):
        if (STUDY / 'progress/correction_resume_launch.json').exists():
            raise RuntimeError('continuation already launched; inspect, do not duplicate')
        for seed in GRANTS:
            if retry.seed_busy(seed) or stream.unresolved(seed):
                raise RuntimeError('stopped seed has an active or ambiguous request')
        previous = s.read(STUDY / 'progress/stream_supervisor.json')
        pid = previous['pid']
        info = stream.process_info(pid)
        if str(stream.SCRIPT) not in info or ' supervise ' not in info:
            raise RuntimeError('original supervisor identity mismatch')
        base.once(STUDY / 'progress/supervisor_before_correction_resume.json', previous)
        os.kill(pid, signal.SIGTERM)  # Controller only; detached workers are preserved.
        for _ in range(50):
            try:
                info = stream.process_info(pid)
                if info.startswith('Z'):
                    break
            except subprocess.CalledProcessError:
                break
            time.sleep(.1)
        else:
            raise RuntimeError('old controller did not exit; no replacement launched')
        with (STUDY / 'progress/correction_resume_supervisor.log').open('ab') as output:
            child = subprocess.Popen([sys.executable, str(SCRIPT), 'supervise', '--allow-paid-llm', '--allow-training', '--allow-label-reveal'],
                                     cwd=s.ROOT, env=dict(os.environ), stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        record = {'pid': child.pid, 'started_utc': retry.now(), 'old_controller_pid': pid,
                  'resumed_seeds': list(GRANTS), 'inflight_workers_interrupted': False,
                  'amendment_sha256': s.sha256_file(POLICY)}
        base.once(STUDY / 'progress/correction_resume_launch.json', record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'validate', 'launch', 'supervise', 'worker', 'status'))
    parser.add_argument('--seed', type=int, choices=tuple(GRANTS))
    for flag in ('allow-paid-llm', 'allow-training', 'allow-label-reveal'):
        parser.add_argument('--' + flag, action='store_true')
    args = parser.parse_args()
    if args.action in ('launch', 'supervise', 'worker') and not (args.allow_paid_llm and args.allow_training and args.allow_label_reveal):
        parser.error('all explicit cost/training/reveal flags required')
    if args.action == 'worker':
        if args.seed is None:
            parser.error('worker requires seed')
        result = worker(args.seed)
    elif args.action == 'status':
        result = stream.status()
    else:
        if args.action in ('prepare', 'validate'):
            stream.activate()
        result = globals()[args.action]()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
