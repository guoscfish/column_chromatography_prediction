#!/usr/bin/env python3
"""Append the authorized five-seed NoCW L1005 prefixes through L1389.

The L1005 workers and all frozen predecessors remain untouched. A seed is
dispatched here only after its old trajectory is complete and its lock is free.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.studies import run_qgeognn_v2_row_llm_nocw_unbounded_v3 as legacy

base, retry, stream, s = legacy.base, legacy.retry, legacy.stream, legacy.s
SOURCE = legacy.STUDY
STUDY = SOURCE.parent / 'five_seed_nocw_to1389_20261101'
SCRIPT = Path(__file__).resolve()
TEST = s.ROOT / 'tests/active_learning_v2/test_nocw_to1389.py'
POLICY = STUDY / 'extension_to1389_amendment.json'
TARGET = 1389
FIRST_ROUND = 21
BUDGETS = tuple(range(333, TARGET + 1, 32))
ARM = 'llm32_nocw'
ORIGINAL_VALIDATE = retry.BASE_VALIDATE
ORIGINAL_PREDICTION_FREEZE = s.prediction_freeze
ORIGINAL_WRITE_STAGE = s.write_stage
ORIGINAL_AUDIT_BATCH = s.audit_batch
ORIGINAL_ONCE = base.BASE_ONCE
LEGACY_HASHES = {
    'persistent_retry_amendment.json': '2b3c1b62057a6015deb7871bda21cd498dd53e2dfeaceaa086208c777939fc60',
    'stream_transport_amendment.json': 'beb59766429506b118c125c306a006f518839628653c022455cbefc8cd16baeb',
    'correction_resume_amendment_20261030_v2.json': '2335406b512ec9bb963cc1ec4743727116fdede4f6901aed2546330243304375',
    'correction_resume_seed2357_round10_20261030.json': 'dd66a60a7fc93b51a021bfcfad7dbdbabdf2e87b7e03ea976bb765d4f3f67fab',
    'correction_resume_seed6101_round10_20261030.json': 'ed8894fda8d524d9c3895ccc0498cf7d61622a7d37a9ec35afea94b4fdd19446',
    'correction_resume_seed6101_round11_20261030.json': 'a390a5bf554e127d692147084889b2f7ca568c3d65bdfea8f238a06f49feb878',
    'correction_resume_seed887_round09_20261030.json': '9ca1f8072f8cc41778d27acd130b4e18767bd9051c256adbc4fe59cee711a8eb',
    'correction_resume_seed6101_round20_20261031.json': '357f587aeb707813ecfdf6002202121880479b8ac26d39077ef61e356f3498ae',
    'unbounded_correction_amendment_20261031.json': 'b358da1dfeaeb96ab63f58aa1142b746c144e0025e5911438537b10bf52787d3',
    'unbounded_correction_amendment_20261031_v2.json': '78ba62172ada375741dc0996a0b534982a0a1e46855f1275bde2c6dfb12237a8',
    'unbounded_correction_amendment_20261031_v3.json': '19c18c1810a2069339e06cedf56144494da1d8a4706bc97346a2be3269ec631f',
}


def original_validation():
    # The scientific source/config/prompt/environment pins still refer to the
    # unchanged L1005 protocol. The separate amendment extends its budget only.
    previous = s.STUDY
    s.STUDY = SOURCE
    try:
        return ORIGINAL_VALIDATE()
    finally:
        s.STUDY = previous


def validate():
    result = original_validation()
    p = s.read(POLICY)
    if (p['target_budget'] != TARGET or p['budgets'] != list(BUDGETS)
            or p['source_budget'] != 1005 or p['first_new_round'] != FIRST_ROUND
            or p['seeds'] != list(base.SEEDS) or p['arms'] != [ARM]
            or p['batch_size'] != 32 or p['maximum_correction_answers'] is not None
            or p['maximum_transport_attempts'] is not None
            or p['original_protocol_sha256'] != result['protocol_sha256']
            or p['legacy_amendment_hashes'] != LEGACY_HASHES
            or p['user_target_reply'] != 'L1389：保持每批32，停在1400以下'):
        raise RuntimeError('extension authorization or budget scope drift')
    for name, digest in LEGACY_HASHES.items():
        f = SOURCE / name
        if s.sha256_file(f) != digest:
            raise RuntimeError('frozen predecessor amendment drift')
        for path, expected in s.read(f).get('code_hashes', {}).items():
            if s.sha256_file(s.ROOT / path) != expected:
                raise RuntimeError('frozen predecessor source drift')
    for name, digest in p['copied_root_files'].items():
        if s.sha256_file(SOURCE / name) != digest or s.sha256_file(STUDY / name) != digest:
            raise RuntimeError('scientific prefix configuration drift')
    for name, digest in p['code_hashes'].items():
        if s.sha256_file(s.ROOT / name) != digest:
            raise RuntimeError('extension adapter source drift')
    return {**result, 'target_budget': TARGET, 'extension_amendment_sha256': s.sha256_file(POLICY)}


def require_authorization():
    result = validate()
    approval = s.read(STUDY / 'formal_authorization.json')
    if (not approval.get('formal_run_authorized') or approval['target_budget'] != TARGET
            or approval['seeds'] != list(base.SEEDS)
            or approval['extension_amendment_sha256'] != result['extension_amendment_sha256']
            or approval['protocol_sha256'] != result['protocol_sha256']):
        raise PermissionError('extension authorization does not bind this revision')


def activate(seed=None):
    legacy.activate(seed)
    # These are process-local runtime adapters, never edits to frozen files.
    base.STUDY = retry.STUDY = stream.STUDY = legacy.STUDY = s.STUDY = STUDY
    s.BUDGETS = BUDGETS
    s.validate, s.require_authorization = validate, require_authorization
    s.prediction_freeze = prediction_freeze
    s.write_stage, s.audit_batch = write_stage, audit_batch
    s.run_selection = legacy.run_selection


def prediction_freeze(context, method, rnd, labeled):
    if rnd <= FIRST_ROUND:
        directory = STUDY / f'runtime/seed_{context.seed}/{method}/round_{rnd:02d}'
        frozen = s.read(directory / 'prediction_freeze.json')
        if frozen['L_t_ids_hash'] != s.ids_hash(context.ids(labeled)) or frozen['test_truth_access_count'] != 0:
            raise RuntimeError('reused predictor label identity mismatch')
        for kind in ('checkpoint', 'prediction'):
            if s.sha256_file(Path(frozen[f'{kind}_path'])) != frozen[f'{kind}_sha256']:
                raise RuntimeError('reused predictor artifact drift')
        return Path(frozen['checkpoint_path']), directory
    return ORIGINAL_PREDICTION_FREEZE(context, method, rnd, labeled)


def write_stage(directory, raw, roles, source, seed, arm, rnd, history):
    if rnd < FIRST_ROUND:
        raise ValueError('the extension cannot stage or recollect a completed prefix')
    catalog, view, packet = s.input_view(raw, seed, arm, rnd, history)
    if ([x['id'] for x in view['observed']] != roles['l0']
            or [x['id'] for x in view['candidates']] != roles['u0']):
        raise RuntimeError('catalog/trajectory IDs disagree')
    directory.mkdir(parents=True, exist_ok=True)
    s.once(directory / 'catalog.json', view)
    s.once(directory / 'input.json', packet)
    s.once(directory / 'contract.json', {
        'seed': seed, 'method': arm, 'round': rnd, 'variant': s.ARMS[arm],
        'protocol_hash': validate()['protocol_sha256'],
        'extension_amendment_sha256': s.sha256_file(POLICY),
        'checkpoint_sha256': source['checkpoint_sha256'],
        'L_t_ids': roles['l0'], 'U_t_ids': roles['u0'],
        'input_sha256': s.sha256_file(directory / 'input.json'),
        'catalog_sha256': s.sha256_file(directory / 'catalog.json'), 'dry_run': False})
    return catalog, packet


def audit_batch(directory):
    c = s.read(Path(directory) / 'contract.json')
    if c['round'] >= FIRST_ROUND and c.get('extension_amendment_sha256') != s.sha256_file(POLICY):
        raise RuntimeError('extended batch does not bind its authorized amendment')
    # Original strict parser, full pool, saved request replay and batch hashes.
    return ORIGINAL_AUDIT_BATCH(directory)


def unresolved_at(root, seed):
    return [p for p in (root / f'selections/seed_{seed}/{ARM}').glob('round_*/attempt_*.started.json')
            if not p.with_name(p.name.replace('.started.json', '.json')).exists()]


def lock_busy(root, seed):
    import fcntl
    path = root / f'locks/worker_seed_{seed}.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as h:
        try:
            fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(h, fcntl.LOCK_UN)
    return False


def source_complete(seed):
    progress = s.read(SOURCE / f'progress/seed_{seed}.json')
    if progress.get('phase') != 'COMPLETE' or progress.get('final_budget') != 1005:
        return False
    # Short lock contention from another read-only auditor is not a failure.
    for i in range(3):
        if not lock_busy(SOURCE, seed):
            break
        if i == 2:
            return False
        time.sleep(.05)
    if unresolved_at(SOURCE, seed):
        raise RuntimeError('completed source contains an ambiguous request')
    f = SOURCE / f'runtime/seed_{seed}/{ARM}/trajectory_freeze.json'
    trajectory = s.read(f)
    ids = trajectory['selected_ids']
    if (trajectory['final_active_labels'] != 1005 or len(ids) != 672
            or len(set(ids)) != 672 or len(trajectory['files']) != 85):
        raise RuntimeError('source L1005 trajectory is incomplete')
    for name, digest in trajectory['files'].items():
        if s.sha256_file(SOURCE / name) != digest:
            raise RuntimeError('source L1005 frozen trajectory drift')
    return True


def verify_prefix(seed):
    p = s.read(STUDY / f'progress/import_seed_{seed}.json')
    if p['source_budget'] != 1005 or p['source_study'] != str(SOURCE):
        raise RuntimeError('imported prefix identity drift')
    for name, digest in p['files'].items():
        if s.sha256_file(SOURCE / name) != digest or s.sha256_file(STUDY / name) != digest:
            raise RuntimeError('imported immutable prefix drift')
    f = SOURCE / f'runtime/seed_{seed}/{ARM}/trajectory_freeze.json'
    if s.sha256_file(f) != p['source_trajectory_sha256']:
        raise RuntimeError('source trajectory binding drift')
    return p


@contextmanager
def offline_saved_answers():
    original = s.request_with_retries
    def saved(messages, config, directory, turn, call):
        path = Path(directory) / f'turn_{turn:02d}.json'
        if not path.exists():
            raise RuntimeError('offline audit refuses any new provider request')
        record = s.read(path)
        if (not record.get('success') or record['request_sha256'] != s.stable_hash({'messages': messages, 'config': config})
                or record['provenance']['served_model'] != s.read(STUDY / 'served_model_lock.json')['served_model']):
            raise RuntimeError('saved answer request/model binding drift')
        return record
    s.request_with_retries = saved
    try:
        yield
    finally:
        s.request_with_retries = original


def audit_fit(seed, rnd):
    d = STUDY / f'runtime/seed_{seed}/{ARM}/round_{rnd:02d}'
    frozen, fit = s.read(d / 'prediction_freeze.json'), s.read(d / 'model/fit_audit.json')
    initial = s.read(base.initial_paths(seed)['member0_audit'])
    if (fit['train_rows'] != BUDGETS[rnd] or fit['train_ids_hash'] != frozen['L_t_ids_hash']
            or fit['initialization_hash'] != initial['initialization_hash']
            or fit['validation_ids_hash'] != initial['validation_ids_hash']
            or fit['test_labels_used_for_fit_or_checkpoint_selection'] != 0
            or frozen['test_truth_access_count'] != 0):
        raise RuntimeError('reused fit budget/label/initialization lineage mismatch')
    for kind, filename in (('checkpoint','best.pt'), ('prediction','predictions.csv.gz')):
        expected = frozen[f'{kind}_sha256']
        if (fit[f'{kind}_sha256'] != expected or s.sha256_file(d / 'model' / filename) != expected
                or s.sha256_file(Path(frozen[f'{kind}_path'])) != expected):
            raise RuntimeError('reused fitted artifact drift')


def import_prefix(seed):
    receipt = STUDY / f'progress/import_seed_{seed}.json'
    if receipt.exists():
        return verify_prefix(seed)
    if not source_complete(seed):
        raise RuntimeError('source must finish L1005 before extension dispatch')
    if lock_busy(STUDY, seed) or unresolved_at(STUDY, seed):
        raise RuntimeError('extension seed is active or ambiguous')
    files = []
    selections = SOURCE / f'selections/seed_{seed}/{ARM}'
    for rnd in range(FIRST_ROUND):
        files.extend(p for p in (selections / f'round_{rnd:02d}').iterdir() if p.is_file())
    runtime = SOURCE / f'runtime/seed_{seed}'
    files.append(runtime / 'context.json')
    for rnd in range(FIRST_ROUND + 1):
        d = runtime / ARM / f'round_{rnd:02d}'
        files.append(d / 'prediction_freeze.json')
        if (d / 'model').exists():
            files.extend(p for p in (d / 'model').rglob('*') if p.is_file())
    manifest = {}
    for f in sorted(set(files)):
        name = str(f.relative_to(SOURCE))
        target = STUDY / name
        digest = s.sha256_file(f)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if s.sha256_file(target) != digest:
                raise RuntimeError('partial imported prefix differs from source')
        else:
            shutil.copy2(f, target)
        if s.sha256_file(target) != digest:
            raise RuntimeError('copied prefix checksum mismatch')
        manifest[name] = digest
    before = dict(manifest)
    with offline_saved_answers():
        for rnd in range(FIRST_ROUND):
            s.audit_batch(s.selection_directory(seed, ARM, rnd))
            audit_fit(seed, rnd + 1)
    for name, digest in before.items():
        if s.sha256_file(STUDY / name) != digest or s.sha256_file(SOURCE / name) != digest:
            raise RuntimeError('prefix audit mutated frozen evidence')
    record = {'seed': seed, 'imported_utc': retry.now(), 'source_study': str(SOURCE),
        'source_budget': 1005, 'first_new_round': FIRST_ROUND,
        'source_trajectory_sha256': s.sha256_file(runtime / ARM / 'trajectory_freeze.json'),
        'files': manifest, 'strict_saved_batch_replays': FIRST_ROUND,
        'completed_fits_reused': FIRST_ROUND, 'paid_calls_during_import': 0,
        'test_truth_reads': 0, 'extension_amendment_sha256': s.sha256_file(POLICY)}
    ORIGINAL_ONCE(receipt, record)
    return record


def prepare():
    # This action is offline and never modifies or signals the source workers.
    base.activate()
    valid = original_validation()
    for name, digest in LEGACY_HASHES.items():
        if s.sha256_file(SOURCE / name) != digest:
            raise RuntimeError('predecessor amendment drift')
    root_files = ('config.json', 'protocol.json', 'protocol.lock.json', 'served_model_lock.json')
    hashes = {}
    for name in root_files:
        f, target = SOURCE / name, STUDY / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(f, target)
        hashes[name] = s.sha256_file(f)
        if s.sha256_file(target) != hashes[name]:
            raise RuntimeError('extension root already differs from source')
    record = {
        'version': 'nocw_five_seed_extension_to1389_v1', 'authorized_utc': retry.now(),
        'user_request': '都跑到1400吧',
        'user_target_reply': 'L1389：保持每批32，停在1400以下',
        'target_reply_source': 'Human async reply request_user_input_async/call_c0b4e9b65a494668aa02ffb3e3ab737b/0',
        'continuing_retry_authorization': '明确授权五个seed持续付费重试，无次数或费用上限',
        'target_budget': TARGET, 'source_budget': 1005, 'first_new_round': FIRST_ROUND,
        'budgets': list(BUDGETS), 'seeds': list(base.SEEDS), 'arms': [ARM], 'batch_size': 32,
        'maximum_correction_answers': None, 'maximum_transport_attempts': None,
        'original_protocol_sha256': valid['protocol_sha256'],
        'legacy_amendment_hashes': LEGACY_HASHES, 'copied_root_files': hashes,
        'hash_hints': 'only the three previously authorized historical rounds; none in new rounds',
        'scientific_changes': 'target endpoint only; same strict full-pool NoCW acquisition, model, effort, independent keys and training',
        'handover': 'leave old workers running; import complete byte-identical L1005 prefixes after source lock release; dispatch each extension worker once',
        'frozen_source_changes': False, 'completed_prefix_retraining': False, 'paid_probes': False,
        'abnormal_worker_restart': False, 'test_truth_reads': 0,
        'code_hashes': {str(p.relative_to(s.ROOT)): s.sha256_file(p) for p in (SCRIPT, TEST)}}
    if POLICY.exists():
        old = s.read(POLICY)
        if any(old[k] != v for k,v in record.items() if k != 'authorized_utc'):
            raise RuntimeError('prepared extension policy drift')
    else:
        ORIGINAL_ONCE(POLICY, record)
    ORIGINAL_ONCE(STUDY / 'formal_authorization.json', {
        'formal_run_authorized': True, 'protocol_sha256': valid['protocol_sha256'],
        'extension_amendment_sha256': s.sha256_file(POLICY), 'target_budget': TARGET,
        'seeds': list(base.SEEDS), 'user_target_reply': record['user_target_reply']})
    activate()
    return validate()


def worker(seed):
    if seed not in base.SEEDS:
        raise ValueError('unregistered extension seed')
    activate(seed)
    require_authorization()
    base.credential_audit()
    os.environ.pop('SCIENTIST_API_KEY', None)
    os.environ['SCIENTIST_API_KEY_FILE'] = str(base.KEY_FILES[seed])
    base.torch.set_num_threads(2)
    with base.BASE_EXCLUSIVE_LOCK(STUDY / f'locks/worker_seed_{seed}.lock'):
        try:
            verify_prefix(seed)
            if not source_complete(seed):
                raise RuntimeError('source is no longer safely complete before extension start')
            if unresolved_at(STUDY, seed):
                raise RuntimeError('unresolved extension request requires reconciliation')
            for rnd in range(FIRST_ROUND, len(BUDGETS) - 1):
                d = STUDY / f'runtime/seed_{seed}/{ARM}/round_{rnd+1:02d}'
                if (d / 'prediction_freeze.json').exists():
                    raise RuntimeError('extension fit already exists; explicit reconciliation required')
                base.status_record(seed, 'SELECTING', round=rnd, arm=ARM, current_budget=BUDGETS[rnd])
                s.select(seed, ARM, rnd, allow_paid_llm=True, allow_feature_compute=True)
                base.status_record(seed, 'ADVANCING', round=rnd, arm=ARM, next_budget=BUDGETS[rnd+1])
                s.advance(seed, ARM, rnd, allow_reveal=True, allow_training=True)
                audit_fit(seed, rnd+1)
            base.status_record(seed, 'COMPLETE', final_budget=TARGET, arms=[ARM])
        except BaseException as error:
            base.status_record(seed, 'FAILED_CLOSED', error_type=type(error).__name__, reason=retry.safe_error(error))
            raise


def process_alive(pid):
    try:
        info = stream.process_info(pid)
        return bool(info) and not info.startswith('Z')
    except subprocess.CalledProcessError:
        return False


def supervise():
    activate()
    require_authorization()
    children = {}
    with base.BASE_EXCLUSIVE_LOCK(STUDY / 'locks/extension_supervisor.lock'):
        while True:
            validate()
            rows = []
            for seed in base.SEEDS:
                path = STUDY / f'progress/seed_{seed}.json'
                progress = s.read(path) if path.exists() else {}
                row = {'seed': seed, 'target_budget': TARGET, 'phase': progress.get('phase'), 'pid': progress.get('pid')}
                intent = STUDY / f'progress/dispatch_seed_{seed}.intent.json'
                if progress.get('phase') == 'COMPLETE':
                    if progress.get('final_budget') != TARGET:
                        raise RuntimeError('extension completion budget mismatch')
                    row['supervision'] = 'COMPLETE'
                elif seed in children:
                    child = children[seed]
                    row.update(pid=child.pid, supervision='EXTENSION_WORKER_RUNNING' if child.poll() is None else 'NEEDS_RECONCILIATION', exit_code=child.returncode)
                elif intent.exists():
                    row['supervision'] = 'NEEDS_RECONCILIATION'
                elif not source_complete(seed):
                    old = s.read(SOURCE / f'progress/seed_{seed}.json')
                    row.update(supervision='SOURCE_L1005_RUNNING' if lock_busy(SOURCE, seed) else 'SOURCE_NEEDS_RECONCILIATION', source_phase=old['phase'], source_pid=old.get('pid'))
                else:
                    import_prefix(seed)
                    if not source_complete(seed) or lock_busy(STUDY, seed) or unresolved_at(STUDY, seed):
                        raise RuntimeError('source/extension handover state changed')
                    ORIGINAL_ONCE(intent, {'seed': seed, 'created_utc': retry.now(), 'amendment_sha256': s.sha256_file(POLICY)})
                    env = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
                    env.pop('SCIENTIST_API_KEY', None)
                    with (STUDY / f'progress/seed_{seed}.log').open('ab') as output:
                        child = subprocess.Popen([sys.executable, str(SCRIPT), 'worker', '--seed', str(seed), '--allow-paid-llm', '--allow-training', '--allow-label-reveal'], cwd=s.ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                    children[seed] = child
                    ORIGINAL_ONCE(STUDY / f'progress/worker_seed_{seed}.json', {'seed': seed, 'pid': child.pid, 'started_utc': retry.now(), 'amendment_sha256': s.sha256_file(POLICY)})
                    row.update(pid=child.pid, supervision='EXTENSION_WORKER_DISPATCHED')
                rows.append(row)
            base.atomic_json(STUDY / 'progress/supervisor.json', {'pid': os.getpid(), 'updated_utc': retry.now(), 'target_budget': TARGET, 'workers': rows, 'amendment_sha256': s.sha256_file(POLICY)})
            if all(r['supervision'] == 'COMPLETE' for r in rows):
                return {'status': 'ALL_FIVE_COMPLETE', 'target_budget': TARGET}
            time.sleep(10)


def launch():
    activate()
    require_authorization()
    base.credential_audit()
    with base.BASE_EXCLUSIVE_LOCK(STUDY / 'locks/launch.lock'):
        path = STUDY / 'progress/launch.json'
        if path.exists():
            raise RuntimeError('extension launch already recorded; do not duplicate')
        if any((STUDY / f'progress/dispatch_seed_{seed}.intent.json').exists() for seed in base.SEEDS):
            raise RuntimeError('existing dispatch requires reconciliation')
        intent = STUDY / 'progress/launch.intent.json'
        ORIGINAL_ONCE(intent, {'created_utc': retry.now(), 'amendment_sha256': s.sha256_file(POLICY)})
        with (STUDY / 'progress/supervisor.log').open('ab') as output:
            child = subprocess.Popen([sys.executable, str(SCRIPT), 'supervise', '--allow-paid-llm', '--allow-training', '--allow-label-reveal'], cwd=s.ROOT, env=dict(os.environ), stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        record = {'pid': child.pid, 'started_utc': retry.now(), 'target_budget': TARGET,
            'source_workers_interrupted': False, 'seeds': list(base.SEEDS), 'amendment_sha256': s.sha256_file(POLICY)}
        ORIGINAL_ONCE(path, record)
        return record


def report():
    activate()
    require_authorization()
    for seed in base.SEEDS:
        p = s.read(STUDY / f'progress/seed_{seed}.json')
        t = s.read(STUDY / f'runtime/seed_{seed}/{ARM}/trajectory_freeze.json')
        if p.get('phase') != 'COMPLETE' or p.get('final_budget') != TARGET or t['final_active_labels'] != TARGET:
            raise RuntimeError('all five full L1389 trajectories must finish before report')
        if unresolved_at(STUDY, seed) or lock_busy(STUDY, seed):
            raise RuntimeError('final reporting requires released workers and no ambiguous requests')
        verify_prefix(seed)
        for name,digest in t['files'].items():
            if s.sha256_file(STUDY/name) != digest:
                raise RuntimeError('final trajectory hash mismatch')
    with offline_saved_answers():
        result = base.report(TARGET)
    attempts = []
    for seed in base.SEEDS:
        for f in (STUDY / f'selections/seed_{seed}/{ARM}').glob('round_*/attempt_*.json'):
            if re.fullmatch(r'attempt_\d+_\d+\.json', f.name):
                a = s.read(f)
                attempts.append({'seed': seed, 'path': str(f.relative_to(STUDY)), 'success': a['success'], 'usage': a.get('provenance',{}).get('usage'), 'elapsed_seconds': a['elapsed_seconds']})
    appendix = {'validation_report_sha256': s.sha256_file(STUDY/f'reports/validation_to{TARGET}.json'),
        'legacy_amendment_hashes': LEGACY_HASHES, 'extension_amendment_sha256': s.sha256_file(POLICY),
        'all_physical_attempts': attempts, 'test_truth_reads': 0}
    ORIGINAL_ONCE(STUDY/f'reports/validation_to{TARGET}_amendment_appendix.json', appendix)
    return {'status': 'VALIDATION_REPORT_COMPLETE', 'target_budget': TARGET, 'test_truth_reads': 0, 'attempt_count': len(attempts)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare','validate','import-prefix','launch','supervise','worker','report'))
    p.add_argument('--seed', type=int, choices=base.SEEDS)
    for flag in ('allow-paid-llm','allow-training','allow-label-reveal'):
        p.add_argument('--'+flag, action='store_true')
    a = p.parse_args()
    if a.action in ('launch','supervise','worker') and not (a.allow_paid_llm and a.allow_training and a.allow_label_reveal):
        p.error('all explicit cost/training/reveal flags required')
    if a.action in ('worker','import-prefix') and a.seed is None:
        p.error('seed required')
    if a.action != 'prepare':
        activate(a.seed)
    if a.action == 'import-prefix':
        require_authorization()
        record = import_prefix(a.seed)
        result = {'seed':a.seed,'source_budget':1005,'copied_files':len(record['files']),'strict_batch_replays':21,'paid_calls':0,'test_truth_reads':0}
    elif a.action == 'worker':
        result = worker(a.seed)
    else:
        result = globals()[a.action]()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
