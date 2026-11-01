#!/usr/bin/env python3
"""User-authorized continuous NoCW corrections; preserve every frozen predecessor."""
from __future__ import annotations

import argparse
from itertools import count
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.studies import run_qgeognn_v2_row_llm_nocw_resume_6101_r20 as predecessor
from scripts.studies import run_qgeognn_v2_row_llm_nocw_unbounded as prepared_draft

prior, stream, base, retry, s = (predecessor.prior, predecessor.stream, predecessor.base,
                                predecessor.retry, predecessor.s)
STUDY = predecessor.STUDY
SCRIPT = Path(__file__).resolve()
TEST = s.ROOT / 'tests/active_learning_v2/test_nocw_unbounded_v2.py'
POLICY = STUDY / 'unbounded_correction_amendment_20261031_v2.json'
RESUME = {157: 17, 887: 13, 2357: 15}
FAILURES = {
    157: ('15200fcaf0db3a1af5007d174a96b51de186b9ee7fea21674045745711a4ccc8', 45),
    887: ('1ea245b5eca56cf2400eae92f60ecd5c90ae4dc3de020eb45cfd1a6e59268059', 24),
    2357: ('6af971d2fd8cb59b2b3f775bcd8cca95e65b0e77f5245f8234d0a7b3f6aab029', 27),
}
HASH_HINT_ROUNDS = {(6101, 11), (887, 9), (6101, 20)}
packet_hash_hint = predecessor.packet_hash_hint


def directory_for(seed, round_index):
    return STUDY / f'selections/seed_{seed}/llm32_nocw/round_{round_index:02d}'


def completed_trajectories():
    expected = {
        6101: ('4b68fc4bdfa2d47227583f5a91eddbc8c7b2853f036248abfd18e33bbaae2107',
               '01ab076e82bf8ddfcc47a36834be016910c1ba832037fb9d6105eb892d6c08bb'),
        12203: ('f6845d3ade87e4a748c1b99258b5df23b80f44b9742d00d511c76151333a0f68',
                '7ba0e1167d95df4d7ce8e20860a8b88031e9d65bf60db8fb3ee7e3697bd3a392'),
    }
    result = {}
    for seed, (receipt_hash, trajectory_hash) in expected.items():
        receipt = STUDY / f'progress/milestone_seed_{seed}_L1005.json'
        if s.sha256_file(receipt) != receipt_hash:
            raise RuntimeError('completed seed milestone receipt drift')
        record = s.read(receipt)
        trajectory = STUDY / record['trajectory_freeze_file']
        if s.sha256_file(trajectory) != trajectory_hash:
            raise RuntimeError('completed seed trajectory drift')
        for name, digest in s.read(trajectory)['files'].items():
            if s.sha256_file(STUDY / name) != digest:
                raise RuntimeError('completed seed frozen artifact drift')
        if (s.read(STUDY / f'progress/seed_{seed}.json')['phase'] != 'COMPLETE' or
                stream.unresolved(seed) or retry.seed_busy(seed)):
            raise RuntimeError('completed seed is not safely finished')
        result[str(seed)] = {'milestone_receipt_sha256': receipt_hash,
                             'trajectory_freeze_sha256': trajectory_hash,
                             'recollect_or_retrain': False}
    return result


def original_prefix(seed):
    digest, size = FAILURES[seed]
    audit = STUDY / f'progress/failure_seed_{seed}_round{RESUME[seed]:02d}_audit.json'
    if s.sha256_file(audit) != digest:
        raise RuntimeError('original failure audit drift')
    prefix = s.read(audit)['preserved_prefix']
    if len(prefix) != size:
        raise RuntimeError('original failed-round prefix count drift')
    for name, expected in prefix.items():
        if s.sha256_file(STUDY / name) != expected:
            raise RuntimeError('original failed-round evidence changed')
    return prefix


def safe_stopped_seed(seed):
    if seed not in RESUME or retry.seed_busy(seed) or stream.unresolved(seed):
        raise RuntimeError('target seed has an active or ambiguous request')
    progress = s.read(STUDY / f'progress/seed_{seed}.json')
    if (progress.get('phase') != 'FAILED_CLOSED' or progress.get('reason') !=
            'selection remained invalid after bounded corrections; fail closed'):
        raise RuntimeError('continuation requires the recorded selection failure')
    original_prefix(seed)
    directory = directory_for(seed, RESUME[seed])
    if any((directory / name).exists() for name in
           ('selection.json', 'batch_freeze.json', 'feedback.json', 'label_access_receipt.json',
            'request_03.json', 'turn_03.json')):
        raise RuntimeError('failed round already resumed or revealed')
    if sorted(p.name for p in directory.glob('rejected_*.json')) != [
            'rejected_00.json', 'rejected_01.json', 'rejected_02.json']:
        raise RuntimeError('continuation requires exactly the three original rejected answers')


def prepare():
    stream.activate()
    valid = predecessor.validate()
    prepared_draft.validate()
    if POLICY.exists():
        return validate()
    for seed in RESUME:
        safe_stopped_seed(seed)
    record = {
        'version': 'nocw_unbounded_correction_v2', 'authorized_utc': retry.now(),
        'user_authorization': '明确授权五个seed持续付费重试，无次数或费用上限',
        'authorization_question': '是否明确授权本聊天五个 NoCW seed（157、887、2357、6101、12203）的所有剩余轮次持续调用付费 LLM，网络重试和纠错答案均不设次数或费用上限，直到通过严格校验或达到 L1005？',
        'authorization_source': 'Human async reply request_user_input_async/call_423212c91545496a879bf0fa814abd92/0',
        'previous_auto_review_block': 'retained; explicit scope/cost confirmation now received',
        'supersedes_unlaunched_draft_sha256': s.sha256_file(prepared_draft.POLICY),
        'base_protocol_sha256': valid['protocol_sha256'],
        'previous_seed6101_round20_amendment_sha256': s.sha256_file(predecessor.POLICY),
        'maximum_correction_answers': None, 'maximum_transport_attempts': None,
        'scope': 'All five registered NoCW seeds, existing stopped rounds and every subsequent round to L1005',
        'resumed_rounds': {str(seed): {'round': rnd, 'failure_audit_sha256': FAILURES[seed][0],
                                     'preserved_prefix': original_prefix(seed)}
                           for seed, rnd in RESUME.items()},
        'hash_hint_rounds': [list(pair) for pair in sorted(HASH_HINT_ROUNDS)],
        'new_hash_hint': {'seed': 6101, 'round': 20, 'first_new_turn': 3,
                          'original_failure': 'wrong 64-character contents; not 63-character truncation'},
        'preserve_seven_previous_grants_and_three_hash_hints': True,
        'future_round_hash_hints': False,
        'strict_validation_full_pool_and_all_original_evidence': 'unchanged',
        'model_endpoint_effort_key_mapping_background_store': 'unchanged',
        'transport_backoff_seconds': [30, 60, 120, 240, 300],
        'ambiguous_started_identity_protocol_authentication_or_confirmed_quota': 'stop and reconcile',
        'worker_restart': 'dispatch each authorized stopped seed once; never restart an abnormal worker automatically',
        'completed_seeds': completed_trajectories(),
        'paid_probes': False,
        'code_hashes': {str(p.relative_to(s.ROOT)): s.sha256_file(p) for p in (SCRIPT, TEST)},
    }
    base.once(POLICY, record)
    return record


def validate():
    valid = predecessor.validate()
    prepared_draft.validate()
    policy = s.read(POLICY)
    if (policy['supersedes_unlaunched_draft_sha256'] != s.sha256_file(prepared_draft.POLICY) or
            policy['user_authorization'] != '明确授权五个seed持续付费重试，无次数或费用上限' or
            policy['completed_seeds'] != completed_trajectories()):
        raise RuntimeError('explicit authorization or completed trajectory drift')
    if (policy['base_protocol_sha256'] != valid['protocol_sha256'] or
            policy['previous_seed6101_round20_amendment_sha256'] != s.sha256_file(predecessor.POLICY)):
        raise RuntimeError('unbounded amendment predecessor drift')
    if (policy['maximum_correction_answers'] is not None or
            policy['maximum_transport_attempts'] is not None or
            {tuple(pair) for pair in policy['hash_hint_rounds']} != HASH_HINT_ROUNDS or
            policy['future_round_hash_hints'] is not False or
            set(policy['resumed_rounds']) != {str(seed) for seed in RESUME}):
        raise RuntimeError('continuous retry authorization scope drift')
    for seed, rnd in RESUME.items():
        item = policy['resumed_rounds'][str(seed)]
        if (item['round'] != rnd or item['failure_audit_sha256'] != FAILURES[seed][0] or
                item['preserved_prefix'] != original_prefix(seed)):
            raise RuntimeError('original failure binding changed')
    for name, digest in policy['code_hashes'].items():
        if s.sha256_file(s.ROOT / name) != digest:
            raise RuntimeError('continuous retry adapter source drift')
    return valid


def activate(seed=None):
    predecessor.activate(seed)
    s.validate = validate
    s.run_selection = run_selection


def selection_identity(directory):
    parts = Path(directory).resolve().relative_to(STUDY.resolve()).parts
    if len(parts) != 4 or parts[0] != 'selections' or parts[2] != 'llm32_nocw':
        raise ValueError('continuous corrections are restricted to registered NoCW rounds')
    seed, rnd = int(parts[1].removeprefix('seed_')), int(parts[3].removeprefix('round_'))
    if seed not in base.SEEDS or not 0 <= rnd < len(s.BUDGETS)-1 or (
            Path(directory).resolve() != directory_for(seed, rnd).resolve()):
        raise ValueError('continuous corrections are restricted to registered NoCW rounds')
    return seed, rnd


def run_selection(packet, catalog, directory, variant, config, call):
    directory = Path(directory)
    identity = selection_identity(directory)
    if variant != 'no_cw' or config['selection_attempts'] != 3:
        raise ValueError('frozen NoCW variant/config changed')
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
    # Replay every saved turn from zero without resetting or overwriting its ledger.
    # A saved valid answer returns before any additional request can be issued.
    for turn in count():
        if turn == 3 and identity in HASH_HINT_ROUNDS:
            messages.append({'role': 'user', 'content': packet_hash_hint(packet)})
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


def worker(seed):
    if seed not in RESUME:
        raise ValueError('completed seed is not authorized for redispatch')
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
            first = prior.resume_round(seed)
            if first != RESUME[seed]:
                raise RuntimeError('completed prefix differs from authorized continuation')
            base.once(STUDY / f'progress/resume_unbounded_v2_prefix_seed_{seed}.json', {
                'seed': seed, 'resume_round': first, 'completed_budget': s.BUDGETS[first],
                'completed_fits_audited_and_skipped': first, 'new_model_calls_during_audit': 0,
                'amendment_sha256': s.sha256_file(POLICY)})
            for rnd in range(first, len(s.BUDGETS)-1):
                base.status_record(seed, 'SELECTING', round=rnd, arm='llm32_nocw', current_budget=s.BUDGETS[rnd])
                s.select(seed, 'llm32_nocw', rnd, allow_paid_llm=True, allow_feature_compute=rnd > 0)
                base.status_record(seed, 'ADVANCING', round=rnd, arm='llm32_nocw', next_budget=s.BUDGETS[rnd+1])
                s.advance(seed, 'llm32_nocw', rnd, allow_reveal=True, allow_training=True)
            base.status_record(seed, 'COMPLETE', final_budget=1005, arms=['llm32_nocw'])
        except BaseException as error:
            base.status_record(seed, 'FAILED_CLOSED', error_type=type(error).__name__, reason=str(error)[:700])
            raise


def existing_worker_identity(seed, pid):
    if str(SCRIPT) in stream.process_info(pid):
        info = stream.process_info(pid)
        if ' worker ' in info and f'--seed {seed}' in info:
            return
        raise RuntimeError('existing worker process identity mismatch')
    predecessor.existing_worker_identity(seed, pid)


def supervise():
    activate()
    validate()
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
                    receipt = STUDY / f'progress/resume_unbounded_v2_worker_seed_{seed}.json'
                    busy = retry.seed_busy(seed)
                    if progress['phase'] == 'COMPLETE':
                        row['supervision'] = 'COMPLETE'
                    elif seed in children:
                        child = children[seed]
                        row.update(child_pid=child.pid, supervision='STREAM_WORKER_RUNNING' if child.poll() is None else 'NEEDS_RECONCILIATION', exit_code=child.returncode)
                    elif busy:
                        existing_worker_identity(seed, progress['pid'])
                        row['supervision'] = 'STREAM_WORKER_RUNNING'
                    elif seed in RESUME and not receipt.exists():
                        safe_stopped_seed(seed)
                        env = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
                        env.pop('SCIENTIST_API_KEY', None)
                        with (STUDY / f'progress/seed_{seed}_unbounded_v2.log').open('ab') as output:
                            child = subprocess.Popen([sys.executable, str(SCRIPT), 'worker', '--seed', str(seed), '--allow-paid-llm', '--allow-training', '--allow-label-reveal'], cwd=s.ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                        children[seed] = child
                        base.once(receipt, {'seed': seed, 'pid': child.pid, 'started_utc': retry.now(), 'amendment_sha256': s.sha256_file(POLICY)})
                        row.update(child_pid=child.pid, supervision='STREAM_WORKER_DISPATCHED')
                    else:
                        row['supervision'] = 'NEEDS_RECONCILIATION'
                    rows.append(row)
                base.atomic_json(STUDY / 'progress/stream_supervisor.json', {
                    'pid': os.getpid(), 'updated_utc': retry.now(), 'workers': rows,
                    'amendment_sha256': s.sha256_file(stream.POLICY),
                    'unbounded_correction_amendment_sha256': s.sha256_file(POLICY)})
                if all(row['supervision'] == 'COMPLETE' for row in rows):
                    return {'status': 'ALL_FIVE_COMPLETE'}
                time.sleep(2)


def launch():
    activate()
    validate()
    s.require_authorization()
    base.credential_audit()
    with base.BASE_EXCLUSIVE_LOCK(STUDY / 'locks/unbounded_v2_launch.lock'):
        receipt = STUDY / 'progress/correction_unbounded_v2_launch.json'
        if receipt.exists():
            raise RuntimeError('continuous continuation already launched; inspect, do not duplicate')
        for seed in RESUME:
            safe_stopped_seed(seed)
        previous = s.read(STUDY / 'progress/stream_supervisor.json')
        pid = previous['pid']
        info = stream.process_info(pid)
        if str(predecessor.SCRIPT) not in info or ' supervise ' not in info:
            raise RuntimeError('previous supervisor identity mismatch')
        base.once(STUDY / 'progress/supervisor_before_unbounded_v2.json', previous)
        os.kill(pid, signal.SIGTERM)  # Verified controller only; never signal a worker.
        for _ in range(50):
            try:
                if stream.process_info(pid).startswith('Z'):
                    break
            except subprocess.CalledProcessError:
                break
            time.sleep(.1)
        else:
            raise RuntimeError('old controller did not exit; no replacement launched')
        for seed in RESUME:
            safe_stopped_seed(seed)
        with (STUDY / 'progress/unbounded_v2_supervisor.log').open('ab') as output:
            child = subprocess.Popen([sys.executable, str(SCRIPT), 'supervise', '--allow-paid-llm', '--allow-training', '--allow-label-reveal'], cwd=s.ROOT, env=dict(os.environ), stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        record = {'pid': child.pid, 'started_utc': retry.now(), 'old_controller_pid': pid,
                  'resumed_seeds': list(RESUME), 'inflight_workers_interrupted': False,
                  'amendment_sha256': s.sha256_file(POLICY)}
        base.once(receipt, record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'validate', 'launch', 'supervise', 'worker', 'status'))
    parser.add_argument('--seed', type=int, choices=tuple(RESUME))
    for flag in ('allow-paid-llm', 'allow-training', 'allow-label-reveal'):
        parser.add_argument('--' + flag, action='store_true')
    args = parser.parse_args()
    if args.action in ('launch', 'supervise', 'worker') and not (args.allow_paid_llm and args.allow_training and args.allow_label_reveal):
        parser.error('all explicit cost/training/reveal flags required')
    if args.action == 'worker' and args.seed not in RESUME:
        parser.error('worker requires an authorized stopped seed')
    if args.action in ('prepare', 'validate'):
        stream.activate()
    if args.action == 'worker':
        result = worker(args.seed)
    else:
        result = stream.status() if args.action == 'status' else globals()[args.action]()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
