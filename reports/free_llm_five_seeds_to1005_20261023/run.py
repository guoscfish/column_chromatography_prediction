"""Complete the traditional five-seed roster with three isolated Free LLM runs.

Scientific functions and prompts are imported unchanged. Only the registered
seeds, split source, budget schedule, and process-local I/O are extended here.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.qgeognn_al.active_learning_v2 import scientist_study as s
from src.qgeognn_al.active_learning_v2 import scientist_transport as transport
from src.qgeognn_al.active_learning_v2 import llm_screen
from src.qgeognn_al.active_learning_v2.benchmark_protocol import CONFIRMATION_SEEDS
from src.qgeognn_al.training.predictor import atomic_json

SEEDS = (887, 2357, 12203)
FREE = 'free_llm32_scientist_v2'
BUDGETS = tuple(range(333, 1006, 32))
REVISION = 'sol61_xhigh_free_five_seeds_to1005_20261023'
STUDY = s.ARCHIVE_STUDY / 'revisions' / REVISION
PREVIOUS = s.ARCHIVE_STUDY / 'revisions/sol_xhigh_hybrid_free_to1005_20261013'
CONFIG = s.read(PREVIOUS / 'extension_manifest.json')['new_transport']
KEY_ROOT = Path.home() / '.config/qgeognn-scientist'
KEYS = {887: KEY_ROOT/'dual-expert-813.api-key',
        2357: KEY_ROOT/'dual-expert-813-seed6101.api-key',
        12203: KEY_ROOT/'dual-expert-813.api-key'}


def now():
    return datetime.now().astimezone().isoformat()


def event(seed, action, **fields):
    with (OUT / f'events_seed_{seed}.jsonl').open('a') as handle:
        handle.write(json.dumps({'time': now(), 'seed': seed, 'pid': os.getpid(),
                                 'action': action, **fields}) + '\n')


def status(seed, phase, r, **fields):
    atomic_json(OUT/f'status_seed_{seed}.json', {
        'updated_at': now(), 'seed': seed, 'phase': phase, 'completed_rounds': r,
        'completed_labels': BUDGETS[r], 'target_labels': 1005, **fields})


@contextmanager
def lock(path):
    with Path(path).open('a+') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def training_slot():
    chosen = None
    while chosen is None:
        with lock(OUT/'training_allocation.lock'):
            for i in range(2):
                handle = (OUT/f'training_slot_{i}.lock').open('a+')
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                chosen = handle
                break
        if chosen is None:
            time.sleep(2)
    try:
        yield
    finally:
        fcntl.flock(chosen.fileno(), fcntl.LOCK_UN)
        chosen.close()


def directory(seed, method, r):
    if seed not in SEEDS or method != FREE or r not in range(21):
        raise ValueError('Unregistered Free LLM seed/method/round')
    return STUDY/f'selections/seed_{seed}/{method}/round_{r:02d}'


def validate():
    record = s.read(STUDY/'protocol.json')
    if record['code_hashes'] != s.code_hashes():
        raise RuntimeError('Frozen source drift')
    if record['runner_sha256'] != s.sha256_file(Path(__file__)):
        raise RuntimeError('Frozen runner drift')
    if record['prompt_sha256'] != s.stable_hash(s.SYSTEM_PROMPT):
        raise RuntimeError('Frozen prompt drift')
    if (STUDY/'selector_prompt.txt').read_text() != s.SYSTEM_PROMPT:
        raise RuntimeError('Prompt artifact drift')
    for path, digest in record['source_hashes'].items():
        if s.sha256_file(ROOT/path) != digest:
            raise RuntimeError(f'Frozen source artifact drift: {path}')
    return {'status': 'VALID', 'protocol_sha256': s.stable_hash(record)}


def activate():
    s.configure_revision(REVISION)
    s.SEEDS = SEEDS
    s.BUDGETS = BUDGETS
    # All five original row splits and L333 checkpoints exist in BASELINE.
    s.CW_SHORT = s.BASELINE
    s.selection_directory = directory
    s.validate = validate
    # The two old seeds had a precomputed CW bank at round zero. Recompute
    # exactly the same sketches for new seeds using the existing later-round
    # implementation; no historical directory or graph cache is modified.
    s.banks = lambda context, current, model, r, path: llm_screen.banks(
        context, current, model, max(1, r), path)
    validate()


def prepare():
    assert tuple(CONFIRMATION_SEEDS) == (157, 887, 2357, 6101, 12203)
    assert CONFIG['model'] == 'gpt-6.1-sol' and CONFIG['reasoning_effort'] == 'xhigh'
    STUDY.mkdir(parents=True, exist_ok=True)
    sources = {str(s.SOURCE_DATA.relative_to(ROOT)): s.sha256_file(s.SOURCE_DATA)}
    for seed in SEEDS:
        split = s.BASELINE/f'splits/row_seed_{seed}.csv'
        partition = s.pd.read_csv(split)
        s.validate_row_protocol(partition)
        paths = s.historical_round0_paths(seed)
        fit = s.read(paths['member0_audit'])
        assert fit['train_rows'] == 333
        assert fit['train_ids_hash'] == s.ids_hash(partition.loc[partition.role.eq('l0'), 'sample_id'].astype(str))
        assert fit['validation_ids_hash'] == s.ids_hash(partition.loc[partition.role.eq('validation'), 'sample_id'].astype(str))
        for kind in ('checkpoint', 'prediction'):
            source = paths['member0_model' if kind == 'checkpoint' else 'member0_predictions']
            assert s.sha256_file(source) == fit[kind+'_sha256']
        for path in [split, *(paths[k] for k in ('context', 'scrubbed_graphs', 'member0_model',
                                                'member0_predictions', 'member0_audit'))]:
            sources[str(path.relative_to(ROOT))] = s.sha256_file(path)
        dest = STUDY/f'splits/row_seed_{seed}.csv'
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            shutil.copy2(split, dest)
        assert s.sha256_file(split) == s.sha256_file(dest)
        sources[str(dest.relative_to(ROOT))] = s.sha256_file(dest)
    reused = []
    for seed in (157, 6101):
        path = PREVIOUS/f'{seed}_{FREE}_result.json'
        result = s.read(path)
        assert result['final_labels'] == 1005 and len(set(result['selected_ids'])) == 672
        sources[str(path.relative_to(ROOT))] = s.sha256_file(path)
        reused.append({'seed': seed, 'result': str(path), 'final_labels': 1005})
    previous_protocol = s.read(PREVIOUS/'protocol.json')
    # Confirm exact scientific implementation against the historical experiment.
    scientific_paths = ('scientist_study.py', 'scientist_selector.py', 'scientist_full_delivery.py',
                        'scientist_transport.py', 'short_sequential_runner.py', 'llm_screen.py',
                        'runner.py', 'benchmark_protocol.py', 'gradient_features.py')
    for name in scientific_paths:
        relative = 'src/qgeognn_al/active_learning_v2/' + name
        assert s.sha256_file(ROOT/relative) == previous_protocol['code_hashes'][relative], relative
    record = {
        'authorization_date': '2026-10-23', 'user_request': 'Complete Free LLM for the same five traditional seeds',
        'all_seeds': list(CONFIRMATION_SEEDS), 'seeds': list(SEEDS), 'methods': [FREE],
        'reused_completed_trajectories': reused, 'budgets': list(BUDGETS), 'batch_size': 32,
        'transport': CONFIG, 'training': s.TRAINING_CONFIG, 'model_call_budget': transport.FULL_SELECTION_ATTEMPTS,
        'candidate_delivery': 'all candidates and observed records; lossless tables; no retrieval',
        'selector_boundary': 'own L_t truth and U_t features/predictions only; no validation/test or other trajectories',
        'truth_barrier': 'freeze all 32 IDs and premeasurement predictions before reveal',
        'test_evaluation_performed': False, 'test_reveal_barrier': 'all three new trajectories frozen before separate test reporting',
        'historical_model_transitions_preserved': True, 'new_seed_model': 'gpt-6.1-sol xhigh from round zero',
        'training_concurrency': 2, 'training_threads_each': 2, 'max_calls_per_credential': 1,
        'prompt_sha256': s.stable_hash(s.SYSTEM_PROMPT), 'code_hashes': s.code_hashes(),
        'runner_sha256': s.sha256_file(Path(__file__)), 'source_hashes': sources,
        'scientific_implementation_matches_previous': list(scientific_paths),
    }
    s.once(STUDY/'protocol.json', record)
    (STUDY/'selector_prompt.txt').write_text(s.SYSTEM_PROMPT)
    activate()
    print(json.dumps({'status': 'PREPARED', 'new_seeds': SEEDS, 'reused_seeds': [157, 6101],
                      'target': 1005, 'scientific_source_matches_previous': True}), flush=True)


def client_for(proxy, timeout=600):
    import httpx
    return httpx.Client(proxy=proxy, trust_env=proxy is None, timeout=timeout)


def probe(key_file):
    failures = []
    for proxy in ('http://127.0.0.1:3066', None):
        try:
            with client_for(proxy, 20) as client:
                response = client.get(CONFIG['base_url']+'/models',
                                      headers={'Authorization': 'Bearer '+key_file.read_text().strip()})
                response.raise_for_status()
                assert any(x.get('id') == CONFIG['model'] for x in response.json().get('data', []))
            return {'status': 200, 'requested_model_listed': True, 'proxy': proxy}
        except Exception as exc:
            failures.append(type(exc).__name__)
    raise RuntimeError('Provider preflight failed: '+','.join(failures))


def responses_call(seed, messages, config):
    """Same Responses request; persist SSE progress and accept completed output only."""
    import httpx
    import openai
    route = s.read(OUT/'launch.json')['routes'][str(seed)]
    key = KEYS[seed].read_text().strip()
    lane = hashlib.sha256(key.encode()).hexdigest()[:16]
    with lock(OUT/f'credential_{lane}.lock'):
        if (OUT/f'quota_{lane}.json').exists():
            raise SystemExit(78)
        event(seed, 'request_started', model=config['model'], effort=config['reasoning_effort'])
        with client_for(route['proxy']) as client:
            with client.stream('POST', config['base_url']+'/responses',
                               headers={'Authorization': 'Bearer '+key},
                               json={'model': config['model'], 'input': messages,
                                     'reasoning': {'effort': config['reasoning_effort']},
                                     'tools': [], 'store': False, 'stream': True}) as response:
                if response.status_code != 200:
                    body = response.read().decode(errors='replace')
                    if any(x in body for x in ('DAILY_LIMIT_EXCEEDED', 'USAGE_LIMIT_EXCEEDED', 'insufficient_quota')):
                        atomic_json(OUT/f'quota_{lane}.json', {'time': now(), 'status': response.status_code})
                        raise SystemExit(78)
                    response.raise_for_status()
                completed = None
                last = time.monotonic()
                for line in response.iter_lines():
                    if time.monotonic()-last > 30:
                        event(seed, 'stream_progress')
                        last = time.monotonic()
                    if not line.startswith('data:') or line[5:].strip() == '[DONE]':
                        continue
                    data = json.loads(line[5:].strip())
                    kind = data.get('type')
                    if kind == 'response.completed':
                        completed = data['response']
                        break
                    if kind in ('error', 'response.failed', 'response.incomplete'):
                        error = data.get('error') or data.get('response', {}).get('error') or {}
                        code = str(error.get('code', ''))
                        if any(x in str(error) for x in ('DAILY_LIMIT_EXCEEDED', 'USAGE_LIMIT_EXCEEDED', 'insufficient_quota')):
                            atomic_json(OUT/f'quota_{lane}.json', {'time': now(), 'code': code})
                            raise SystemExit(78)
                        raise openai.APIConnectionError(message='Provider stream failure: '+code,
                                                       request=response.request)
        if not completed or completed.get('status') != 'completed':
            raise openai.APIConnectionError(message='Stream ended before response.completed',
                                           request=httpx.Request('POST', config['base_url']+'/responses'))
        model = completed['model']
        if model != config['model'] and not model.startswith(config['model']+'-'):
            raise RuntimeError('Provider model mismatch')
        if any(item['type'] not in ('message', 'reasoning') for item in completed['output']):
            raise RuntimeError('Native tool output invalidates selection')
        answer = ''.join(part['text'] for item in completed['output'] if item['type'] == 'message'
                         for part in item.get('content', []) if part['type'] == 'output_text')
        event(seed, 'response_complete', response_id=completed['id'], served_model=model)
        return answer, {'response_id': completed['id'], 'served_model': model,
                        'usage': completed.get('usage'), 'native_tool_calls': 0}


def finish(seed):
    context = s.Context(seed)
    labeled, _, _, _ = s.state(context, FREE, 21)
    assert len(labeled) == 1005
    rows, files = [], {}
    initial = s.read(s.historical_round0_paths(seed)['member0_audit'])
    for r, budget in enumerate(BUDGETS):
        path = STUDY/f'runtime/seed_{seed}/{FREE}/round_{r:02d}/prediction_freeze.json'
        frozen = s.read(path)
        for kind in ('checkpoint', 'prediction'):
            assert s.sha256_file(Path(frozen[kind+'_path'])) == frozen[kind+'_sha256']
        fit = s.read(Path(frozen['checkpoint_path']).parent/'fit_audit.json')
        assert fit['train_rows'] == budget and fit['initialization_hash'] == initial['initialization_hash']
        assert fit['validation_ids_hash'] == initial['validation_ids_hash']
        rows.append({'labels': budget, 'validation_nrmse': fit['best_validation_combined_normalized_rmse']})
        files[str(path.relative_to(STUDY))] = s.sha256_file(path)
    ids = []
    for r in range(21):
        path = directory(seed, FREE, r)
        batch = s.audit_batch(path)
        assert len(batch['batch_ids']) == 32 and not batch['pending_ids']
        ids.extend(batch['batch_ids'])
        for receipt in path.glob('turn_*.json'):
            if '.started.' in receipt.name:
                continue
            served = s.read(receipt)['provenance']['served_model']
            assert served == 'gpt-6.1-sol' or served.startswith('gpt-6.1-sol-')
        for name in ('batch_freeze.json', 'feedback.json', 'label_access_receipt.json'):
            files[str((path/name).relative_to(STUDY))] = s.sha256_file(path/name)
    assert len(ids) == len(set(ids)) == 672
    s.once(STUDY/f'{seed}_{FREE}_result.json', {'seed': seed, 'method': FREE, 'final_labels': 1005,
           'selected_ids': ids, 'validation_learning_curve': rows, 'files': files,
           'test_evaluation_performed': False})
    status(seed, 'COMPLETE', 21)


def work(seed):
    activate()
    transport.responses_call = lambda messages, config: responses_call(seed, messages, config)
    os.environ['SCIENTIST_API_KEY_FILE'] = str(KEYS[seed])
    os.environ.pop('SCIENTIST_API_KEY', None)
    with s.exclusive_lock(OUT/f'worker_seed_{seed}.lock'):
        try:
            for r in range(21):
                path = STUDY/f'runtime/seed_{seed}/{FREE}/round_{r+1:02d}/prediction_freeze.json'
                if path.exists():
                    s.state(s.Context(seed), FREE, r+1)
                    continue
                status(seed, 'SELECTING', r)
                event(seed, 'batch_started', from_labels=BUDGETS[r], to_labels=BUDGETS[r+1])
                with s.status_phase(f'Free seed{seed} select {BUDGETS[r]}->{BUDGETS[r+1]}'):
                    s.select(seed, FREE, r)
                status(seed, 'WAITING_FOR_TRAINING', r)
                with training_slot():
                    status(seed, 'TRAINING', r, training_labels=BUDGETS[r+1])
                    with s.status_phase(f'Free seed{seed} train {BUDGETS[r+1]}'):
                        s.advance(seed, FREE, r)
                event(seed, 'batch_complete', labels=BUDGETS[r+1])
            finish(seed)
        except BaseException as exc:
            traceback.print_exc()
            completed = max((int(p.parent.name.split('_')[1]) for p in
                            (STUDY/f'runtime/seed_{seed}/{FREE}').glob('round_*/prediction_freeze.json')), default=0)
            status(seed, 'STOPPED', completed, error_type=type(exc).__name__,
                   reason='QUOTA_EXHAUSTED' if isinstance(exc, SystemExit) and exc.code == 78 else 'See worker log')
            raise


def launch():
    prepare()
    with s.exclusive_lock(OUT/'launch.lock'):
        if (OUT/'launch.json').exists():
            raise RuntimeError('Launch already recorded; use --worker SEED to resume a stopped worker')
        routes = {str(seed): probe(KEYS[seed]) for seed in SEEDS}
        record = {'time': now(), 'all_seeds': list(CONFIRMATION_SEEDS), 'new_seeds': list(SEEDS),
                  'target_labels': 1005, 'model': CONFIG['model'], 'reasoning_effort': CONFIG['reasoning_effort'],
                  'routes': routes, 'workers': [], 'revision': str(STUDY)}
        atomic_json(OUT/'launch.json', record)
        env = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        for seed in SEEDS:
            log = OUT/f'worker_seed_{seed}.log'
            with log.open('ab') as handle:
                child = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker', str(seed)],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                    env=env, start_new_session=True)
            subprocess.Popen(['caffeinate', '-i', '-w', str(child.pid)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            record['workers'].append({'seed': seed, 'pid': child.pid, 'log': str(log)})
            atomic_json(OUT/'launch.json', record)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, choices=SEEDS)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.worker is not None:
        work(args.worker)
    elif args.launch:
        launch()
    elif args.prepare:
        prepare()
    else:
        activate()
        print(json.dumps(validate()))
