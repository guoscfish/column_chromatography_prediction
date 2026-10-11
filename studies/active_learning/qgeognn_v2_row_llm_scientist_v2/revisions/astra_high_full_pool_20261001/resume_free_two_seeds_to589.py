"""Authorized Free comparison at 589 labels; extend seed157 only, reuse seed6101."""
import os
import sys
from pathlib import Path
import resume_round6_transcription_fix as repair
import resume_6101_full_plan_compatible as compatible
original = compatible.original
s = original.s
METHOD = 'free_llm32_scientist_v2'
previous_directory = s.selection_directory
previous_selector = s.run_selector

def directory(seed, method, r):
    if seed == 157 and method == METHOD and r in range(8):
        return s.STUDY / f'selections/seed_157/{method}/round_{r:02d}'
    return previous_directory(seed, method, r)

def selector(packet, catalog, path, config, call=None):
    if Path(path) == repair.DIRECTORY:
        return repair.corrected_selector(packet, catalog, path, config, call=call)
    return previous_selector(packet, catalog, path, config, call=call)

s.selection_directory = directory
s.run_selector = selector
OUT = s.STUDY / 'extension_free_two_seeds_to589'

def audit(seed):
    labeled, _, _, _ = s.state(s.Context(seed), METHOD, 8)
    assert len(labeled) == 589
    ids = []
    for r in range(8):
        b = s.audit_batch(directory(seed, METHOD, r))
        assert len(b['batch_ids']) == 32 and not b['pending_ids']
        ids += b['batch_ids']
    assert len(ids) == len(set(ids)) == 256
    rows = [row for row in original.curve(seed, METHOD) if row['labels'] <= 589]
    assert len(rows) == 9 and rows[-1]['labels'] == 589
    return {'seed': seed, 'method': METHOD, 'labels': 589, 'unique_added_ids': 256,
            'validation_learning_curve': rows, 'test_evaluation_performed': False}

def main():
    with s.exclusive_lock(s.STUDY / 'execution.lock'):
        current, random = original.preflight()
        frozen = original.read(original.EXT / 'manifest.json')
        compatible.compatible_manifest(current, frozen, compatible.EXPECTED_ADDITIONS)
        for name, digest in compatible.EXPECTED_ADDITIONS.items():
            assert s.sha256_file(original.ROOT / name) == digest
        assert random == original.read(original.EXT / 'random157_reuse.json')
        old = s.STUDY / f'runtime/seed_157/{METHOD}/trajectory_freeze.json'
        old_hash = s.sha256_file(old)
        s.verify_files(s.STUDY, original.read(old)['files'])
        assert len(s.state(s.Context(157), METHOD, 6)[0]) == 525
        audit(6101)
        authorization = {'authorization': 'User requested both Free seeds to 589 labels on 2026-10-10',
            'target_labels': 589, 'extend_seed157_rounds': [6, 7], 'seed6101': 'reuse existing frozen 589 fit; already reached 653',
            'original_manifest_sha256': s.sha256_file(original.EXT / 'manifest.json'),
            'original_seed157_525_freeze_sha256': old_hash, 'runner_sha256': s.sha256_file(Path(__file__)),
            'historical_deviations': 'seed157 round6 extra response and authorized character repair; seed6101 round0 extra correction',
            'scientific_settings_changed': False}
        if '--check' in sys.argv:
            print('FREE589_PREFLIGHT_PASS', flush=True)
            return
        OUT.mkdir(exist_ok=True)
        original.once(OUT / 'authorization.json', authorization)
        s._progress({'action': 'free589_started', 'pid': os.getpid()})
        try:
            for r in (6, 7):
                pred = s.STUDY / f'runtime/seed_157/{METHOD}/round_{r+1:02d}/prediction_freeze.json'
                if pred.exists():
                    s.state(s.Context(157), METHOD, r+1)
                    continue
                details = {'seed': 157, 'method': METHOD, 'round': r, 'free589': True}
                with s.status_phase(f'Free589 seed157 {s.BUDGETS[r]}->{s.BUDGETS[r+1]} select', 30, **details):
                    s.select(157, METHOD, r)
                with s.status_phase(f'Free589 seed157 train {s.BUDGETS[r+1]}', 30, **details):
                    s.advance(157, METHOD, r)
            assert s.sha256_file(old) == old_hash
            assert s.sha256_file(original.EXT / 'manifest.json') == authorization['original_manifest_sha256']
            for seed in (157, 6101):
                original.once(OUT / f'{seed}_result.json', audit(seed))
            s._progress({'action': 'free589_completed', 'target_labels': 589})
        except Exception as exc:
            s._progress({'action': 'free589_stopped', 'error_type': type(exc).__name__, 'message': str(exc)[:1000]})
            raise

def wait_for_service():
    import time
    import httpx
    from src.qgeognn_al.active_learning_v2.scientist_transport import provider_key
    config = original.read(s.STUDY / 'protocol.json')['transport']
    failures = 0
    while True:
        try:
            response = httpx.get(config['base_url'].rstrip('/') + '/models',
                headers={'Authorization': 'Bearer ' + provider_key(config)}, timeout=20)
            if response.status_code == 200:
                print('AUTH_HTTP200; starting Free589', flush=True)
                return
            body = response.text
            if 'DAILY_LIMIT_EXCEEDED' in body or 'USAGE_LIMIT_EXCEEDED' in body:
                raise RuntimeError('QUOTA_EXHAUSTED; no retry')
            delay = max(900, int(response.headers.get('Retry-After', '900')))
            print(f'SERVICE_WAIT_HTTP_{response.status_code}', flush=True)
        except RuntimeError:
            raise
        except Exception as exc:
            delay = 900
            print('SERVICE_WAIT_' + type(exc).__name__, flush=True)
        failures += 1
        if failures >= 3:
            delay = max(delay, 3600)
        print(f'Service backoff {delay}s; no LLM calls', flush=True)
        time.sleep(delay)

if __name__ == '__main__':
    if '--wait-for-service' in sys.argv:
        wait_for_service()
    main()
