"""Archive completed validation evidence without checkpoint payloads or credentials."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REV = ROOT / 'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/sol_xhigh_hybrid_free_to1005_20261013'
METHODS = ('cw16_llm16_scientist_v2', 'free_llm32_scientist_v2')

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def archive():
    completion = read(OUT / 'completion_audit.json')
    assert completion['status'] == 'ALL_FOUR_COMPLETE1005_VERIFIED'
    fits = {}
    for seed in (157, 6101):
        for method in METHODS:
            result = read(REV / f'{seed}_{method}_result.json')
            assert result['final_labels'] == 1005
            assert len(result['selected_ids']) == len(set(result['selected_ids'])) == 672
            for relative, expected in result['files'].items():
                assert sha(REV / relative) == expected, relative
            ids = []
            for index in range(21):
                batch = read(REV / f'selections/seed_{seed}/{method}/round_{index:02d}/batch_freeze.json')
                assert len(batch['batch_ids']) == len(set(batch['batch_ids'])) == 32
                assert len(batch['pending_ids']) == (16 if method.startswith('cw') else 0)
                ids.extend(batch['batch_ids'])
            assert ids == result['selected_ids']
            for index in range(22):
                freeze_path = REV / f'runtime/seed_{seed}/{method}/round_{index:02d}/prediction_freeze.json'
                freeze = read(freeze_path)
                for kind in ('checkpoint', 'prediction'):
                    assert sha(Path(freeze[kind + '_path'])) == freeze[kind + '_sha256']
                fit_path = Path(freeze['checkpoint_path']).parent / 'fit_audit.json'
                fit = read(fit_path)
                assert fit['train_rows'] == 333 + 32 * index
                assert fit['test_labels_used_for_fit_or_checkpoint_selection'] == 0
                fits[f'{seed}:{method}:{index:02d}'] = {
                    'prediction_freeze': freeze,
                    'prediction_freeze_sha256': sha(freeze_path),
                    'fit_audit': fit,
                    'fit_audit_sha256': sha(fit_path),
                }
    payload = {
        'status': 'VERIFIED_1005_VALIDATION_EVIDENCE',
        'test_evaluation_performed': False,
        'checkpoint_payloads_archived_in_git': False,
        'fits': fits,
    }
    (OUT / 'frozen_fit_audits.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(f'Archived {len(fits)} verified frozen fit audits; no credential or checkpoint payloads.')

if __name__ == '__main__':
    archive()
