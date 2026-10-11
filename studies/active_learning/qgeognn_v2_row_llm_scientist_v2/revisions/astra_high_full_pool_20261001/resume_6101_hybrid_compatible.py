"""Resume only seed6101 hybrid; preserve the original manifest and science."""
import os
import sys
from pathlib import Path
import resume_813_hybrid_first as original

EXPECTED_ADDITIONS = {
    'src/qgeognn_al/active_learning_v2/cw_factorial.py': '2ea829d39c7b3dd73cdda94ee9a9ca4b0167c2e1bb070a9e1d5d3592eb30f448',
    'src/qgeognn_al/active_learning_v2/cw_factorial_diagnostics.py': 'bbc13d81f5c532185f642cca371c75b4cf54b71f414d93489ad563053160568a',
    'src/qgeognn_al/active_learning_v2/cw_factorial_study.py': '64a1a6f9237131a90767c86ca92605dbfc1f77bfa6a900a488f4c8353510521c',
    'src/qgeognn_al/active_learning_v2/feedback_ablation.py': '39be7fb0022e5d6a20bccd1dc6e9c53ebddb40db2918750a43acc368aa318b97',
    'src/qgeognn_al/active_learning_v2/feedback_ablation_adapter.py': '44361d0a444ee0126ea9f25fe8dee3a2a64b99c8d3b445154b39c6cd15a2e5f2',
}


def compatible_manifest(current, frozen, additions):
    old = set(frozen['unrelated_added_modules'])
    if set(current['unrelated_added_modules']) != old | set(additions):
        raise RuntimeError('Unexpected added or removed modules')
    normalized = dict(current)
    normalized['unrelated_added_modules'] = frozen['unrelated_added_modules']
    if normalized != frozen:
        raise RuntimeError('Original manifest field changed beyond recorded additions')


def main():
    s = original.s
    with s.exclusive_lock(s.STUDY / 'execution.lock'):
        current, random = original.preflight()
        manifest_path = original.EXT / 'manifest.json'
        frozen = original.read(manifest_path)
        compatible_manifest(current, frozen, EXPECTED_ADDITIONS)
        for name, digest in EXPECTED_ADDITIONS.items():
            if s.sha256_file(original.ROOT / name) != digest:
                raise RuntimeError(f'Additional module drift: {name}')
        if random != original.read(original.EXT / 'random157_reuse.json'):
            raise RuntimeError('Random reuse audit changed')
        audit = {
            'authorization': 'User requested continuing seed6101 hybrid after manifest mismatch diagnosis on 2026-10-09',
            'scope': {'seed': 6101, 'method': 'cw16_llm16_scientist_v2', 'target_labels': 813},
            'original_manifest_sha256': s.sha256_file(manifest_path),
            'added_module_sha256': EXPECTED_ADDITIONS,
            'runner_sha256': s.sha256_file(Path(__file__)),
            'scientific_settings_changed': False,
            'original_protocol_hashes_verified': True,
            'free_started': False,
        }
        if '--check' in sys.argv:
            print('COMPATIBILITY_PREFLIGHT_PASS; original manifest unchanged', flush=True)
            return
        original.once(original.EXT / 'seed6101_hybrid_compatibility_20261009.json', audit)
        seed, method = 6101, 'cw16_llm16_scientist_v2'
        s._progress({'action': 'extension813_started', 'pid': os.getpid(), 'schedule': 'seed6101_hybrid_only'})
        try:
            for r in range(15):
                d = original.selection_directory(seed, method, r)
                pred = s.STUDY / f'runtime/seed_{seed}/{method}/round_{r+1:02d}/prediction_freeze.json'
                if pred.exists() and (d / 'feedback.json').exists():
                    s.state(s.Context(seed), method, r + 1)
                    continue
                details = {'seed': seed, 'method': method, 'round': r, 'extension813': True}
                s._progress({'action': 'extension813_cycle_started', **details})
                with s.status_phase(f'813 extension seed={seed} {method} {s.BUDGETS[r]}->{s.BUDGETS[r+1]} select', 30, **details):
                    s.select(seed, method, r)
                with s.status_phase(f'813 extension seed={seed} {method} train {s.BUDGETS[r+1]}', 30, **details):
                    s.advance(seed, method, r)
                s._progress({'action': 'extension813_cycle_completed', **details})
            original.finish(seed, method)
            if s.sha256_file(manifest_path) != audit['original_manifest_sha256']:
                raise RuntimeError('Original manifest drift')
            for name, digest in frozen['prior_freezes'].items():
                if s.sha256_file(s.STUDY / name) != digest:
                    raise RuntimeError('Prior freeze drift')
                s.verify_files(s.STUDY, original.read(s.STUDY / name)['files'])
            s._progress({'action': 'extension813_seed6101_hybrid_completed'})
        except Exception as exc:
            s._progress({'action': 'extension813_stopped', 'error_type': type(exc).__name__, 'message': str(exc)[:1000]})
            raise


if __name__ == '__main__':
    main()
