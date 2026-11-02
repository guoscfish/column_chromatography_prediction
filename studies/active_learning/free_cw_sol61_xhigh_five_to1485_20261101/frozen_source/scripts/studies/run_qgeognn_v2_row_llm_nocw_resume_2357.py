#!/usr/bin/env python3
"""Append the authorized seed2357 round10 grant without editing frozen adapters."""
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
from scripts.studies import run_qgeognn_v2_row_llm_nocw_resume as prior

stream, base, retry, s = prior.stream, prior.base, prior.retry, prior.s
STUDY = prior.STUDY
SCRIPT = Path(__file__).resolve()
TEST = s.ROOT / 'tests/active_learning_v2/test_nocw_resume_2357.py'
POLICY = STUDY / 'correction_resume_seed2357_round10_20261030.json'
BASE_ANSWER_LIMIT = prior.answer_limit
SEED, ROUND = 2357, 10


def failed_directory():
    return STUDY / f'selections/seed_{SEED}/llm32_nocw/round_{ROUND:02d}'


def safe_stopped_seed():
    if retry.seed_busy(SEED) or stream.unresolved(SEED):
        raise RuntimeError('seed2357 has an active or ambiguous request')
    progress = s.read(STUDY / f'progress/seed_{SEED}.json')
    if (progress.get('phase') != 'FAILED_CLOSED' or
            progress.get('reason') != 'selection remained invalid after bounded corrections; fail closed'):
        raise RuntimeError('grant requires the recorded three-answer selection failure')
    directory = failed_directory()
    if any((directory / n).exists() for n in ('selection.json', 'batch_freeze.json', 'feedback.json')):
        raise RuntimeError('failed round already accepted or revealed')
    if sorted(p.name for p in directory.glob('rejected_*.json')) != ['rejected_00.json', 'rejected_01.json', 'rejected_02.json']:
        raise RuntimeError('grant requires exactly three original rejected answers')


def prepare():
    stream.activate()
    valid = prior.validate()
    if POLICY.exists():
        return validate()
    safe_stopped_seed()
    prefix = {str(p.relative_to(STUDY)): s.sha256_file(p)
              for p in sorted(failed_directory().iterdir()) if p.is_file()}
    record = {
        'version': 'nocw_seed2357_round10_append_three_v1', 'authorized_utc': retry.now(),
        'user_authorization': '仅此轮追加 3 个答案并续跑（seed2357 round10）',
        'base_protocol_sha256': valid['protocol_sha256'],
        'previous_correction_amendment_sha256': s.sha256_file(prior.POLICY),
        'stream_transport_amendment_sha256': s.sha256_file(stream.POLICY),
        'grant': {'seed': SEED, 'round': ROUND, 'previous_answer_limit': 3,
                  'additional_answers': 3, 'new_answer_limit': 6, 'preserved_prefix': prefix},
        'previous_887_and_6101_grants_preserved': True, 'future_round_answer_limit': 3,
        'preserve_all_requests_attempts_started_answers_failures_and_fits': True,
        'model_endpoint_effort_key_mapping_candidate_pool_validation': 'unchanged',
        'no_additional_probes': True,
        'code_hashes': {str(p.relative_to(s.ROOT)): s.sha256_file(p) for p in (SCRIPT, TEST)},
    }
    base.once(POLICY, record)
    return record


def validate():
    result = prior.validate()
    policy = s.read(POLICY)
    grant = policy['grant']
    if (policy['base_protocol_sha256'] != result['protocol_sha256'] or
            policy['previous_correction_amendment_sha256'] != s.sha256_file(prior.POLICY) or
            policy['stream_transport_amendment_sha256'] != s.sha256_file(stream.POLICY)):
        raise RuntimeError('grant protocol or previous amendment drift')
    if ((grant['seed'], grant['round'], grant['previous_answer_limit'], grant['additional_answers'],
         grant['new_answer_limit']) != (SEED, ROUND, 3, 3, 6) or policy['future_round_answer_limit'] != 3):
        raise RuntimeError('seed2357 correction grant drift')
    for name, digest in policy['code_hashes'].items():
        if s.sha256_file(s.ROOT / name) != digest:
            raise RuntimeError('seed2357 adapter source drift')
    for name, digest in grant['preserved_prefix'].items():
        if s.sha256_file(STUDY / name) != digest:
            raise RuntimeError('original seed2357 failed-round prefix changed')
    return result


def answer_limit(directory, config):
    if config['selection_attempts'] != 3:
        raise RuntimeError('frozen answer limit changed')
    if Path(directory).resolve() == failed_directory().resolve():
        return s.read(POLICY)['grant']['new_answer_limit']
    return BASE_ANSWER_LIMIT(directory, config)


def activate(seed=None):
    prior.activate(seed)
    # Reuse the already audited request serialization, replay, and strict validator.
    # The previous two grants keep their original policy and source bytes.
    prior.answer_limit = answer_limit
    s.validate = validate


def worker():
    activate(SEED)
    validate()
    s.require_authorization()
    base.credential_audit()
    os.environ.pop('SCIENTIST_API_KEY', None)
    os.environ['SCIENTIST_API_KEY_FILE'] = str(base.KEY_FILES[SEED])
    base.torch.set_num_threads(2)
    with base.BASE_EXCLUSIVE_LOCK(STUDY / f'locks/worker_seed_{SEED}.lock'):
        try:
            if stream.unresolved(SEED):
                raise RuntimeError('unresolved provider request requires reconciliation')
            first = prior.resume_round(SEED)
            if first < ROUND:
                raise RuntimeError('completed seed2357 prefix is missing')
            base.once(STUDY / f'progress/resume2357_prefix_seed_{SEED}.json', {
                'seed': SEED, 'resume_round': first, 'completed_budget': s.BUDGETS[first],
                'completed_fits_audited_and_skipped': first, 'new_model_calls_during_audit': 0,
                'amendment_sha256': s.sha256_file(POLICY)})
            for r in range(first, len(s.BUDGETS)-1):
                base.status_record(SEED, 'SELECTING', round=r, arm='llm32_nocw', current_budget=s.BUDGETS[r])
                s.select(SEED, 'llm32_nocw', r, allow_paid_llm=True, allow_feature_compute=r > 0)
                base.status_record(SEED, 'ADVANCING', round=r, arm='llm32_nocw', next_budget=s.BUDGETS[r+1])
                s.advance(SEED, 'llm32_nocw', r, allow_reveal=True, allow_training=True)
            base.status_record(SEED, 'COMPLETE', final_budget=1005, arms=['llm32_nocw'])
        except BaseException as error:
            base.status_record(SEED, 'FAILED_CLOSED', error_type=type(error).__name__, reason=str(error)[:700])
            raise


def existing_worker_identity(seed, pid):
    info = stream.process_info(pid)
    allowed = (stream.SCRIPT, prior.SCRIPT, SCRIPT)
    if not (any(str(p) in info for p in allowed) and ' worker ' in info and f'--seed {seed}' in info):
        raise RuntimeError('existing worker process identity mismatch')


def supervise():
    activate()
    validate()
    s.require_authorization()
    child = None
    receipt = STUDY / f'progress/resume2357_worker_seed_{SEED}.json'
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
                    elif seed == SEED and child is not None:
                        row.update(child_pid=child.pid, supervision='STREAM_WORKER_RUNNING' if child.poll() is None else 'NEEDS_RECONCILIATION', exit_code=child.returncode)
                    elif busy:
                        existing_worker_identity(seed, progress['pid'])
                        row['supervision'] = 'STREAM_WORKER_RUNNING'
                    elif seed == SEED and not receipt.exists():
                        safe_stopped_seed()
                        env = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
                        env.pop('SCIENTIST_API_KEY', None)
                        with (STUDY / f'progress/seed_{SEED}_resume2357.log').open('ab') as output:
                            child = subprocess.Popen([sys.executable, str(SCRIPT), 'worker', '--seed', str(SEED), '--allow-paid-llm', '--allow-training', '--allow-label-reveal'], cwd=s.ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                        base.once(receipt, {'seed': SEED, 'pid': child.pid, 'started_utc': retry.now(), 'amendment_sha256': s.sha256_file(POLICY)})
                        row.update(child_pid=child.pid, supervision='STREAM_WORKER_DISPATCHED')
                    else:
                        row['supervision'] = 'NEEDS_RECONCILIATION'
                    rows.append(row)
                base.atomic_json(STUDY / 'progress/stream_supervisor.json', {
                    'pid': os.getpid(), 'updated_utc': retry.now(), 'workers': rows,
                    'amendment_sha256': s.sha256_file(stream.POLICY),
                    'correction_resume_amendment_sha256': s.sha256_file(prior.POLICY),
                    'seed2357_correction_amendment_sha256': s.sha256_file(POLICY)})
                if all(r['supervision'] == 'COMPLETE' for r in rows):
                    return {'status': 'ALL_FIVE_COMPLETE'}
                time.sleep(2)


def launch():
    activate()
    validate()
    s.require_authorization()
    base.credential_audit()
    with base.BASE_EXCLUSIVE_LOCK(STUDY / 'locks/resume2357_launch.lock'):
        launch_file = STUDY / 'progress/correction_resume2357_launch.json'
        if launch_file.exists():
            raise RuntimeError('seed2357 continuation already launched; inspect, do not duplicate')
        safe_stopped_seed()
        previous = s.read(STUDY / 'progress/stream_supervisor.json')
        pid = previous['pid']
        info = stream.process_info(pid)
        if str(prior.SCRIPT) not in info or ' supervise ' not in info:
            raise RuntimeError('previous resume supervisor identity mismatch')
        base.once(STUDY / 'progress/supervisor_before_resume2357.json', previous)
        os.kill(pid, signal.SIGTERM)  # Controller only. All detached worker sessions survive.
        for _ in range(50):
            try:
                if stream.process_info(pid).startswith('Z'):
                    break
            except subprocess.CalledProcessError:
                break
            time.sleep(.1)
        else:
            raise RuntimeError('old controller did not exit; no replacement launched')
        safe_stopped_seed()
        with (STUDY / 'progress/resume2357_supervisor.log').open('ab') as output:
            child = subprocess.Popen([sys.executable, str(SCRIPT), 'supervise', '--allow-paid-llm', '--allow-training', '--allow-label-reveal'], cwd=s.ROOT, env=dict(os.environ), stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        record = {'pid': child.pid, 'started_utc': retry.now(), 'old_controller_pid': pid,
                  'resumed_seeds': [SEED], 'inflight_workers_interrupted': False,
                  'amendment_sha256': s.sha256_file(POLICY)}
        base.once(launch_file, record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'validate', 'launch', 'supervise', 'worker', 'status'))
    parser.add_argument('--seed', type=int, choices=(SEED,))
    for flag in ('allow-paid-llm', 'allow-training', 'allow-label-reveal'):
        parser.add_argument('--' + flag, action='store_true')
    args = parser.parse_args()
    if args.action in ('launch', 'supervise', 'worker') and not (args.allow_paid_llm and args.allow_training and args.allow_label_reveal):
        parser.error('all explicit cost/training/reveal flags required')
    if args.action == 'worker' and args.seed != SEED:
        parser.error('worker requires --seed 2357')
    if args.action in ('prepare', 'validate'):
        stream.activate()
    result = stream.status() if args.action == 'status' else globals()[args.action]()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
