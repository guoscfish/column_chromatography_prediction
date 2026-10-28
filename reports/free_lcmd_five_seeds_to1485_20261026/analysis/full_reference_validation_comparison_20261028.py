"""Read frozen validation metadata only; no predictor or selector is executed."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'studies/active_learning/qgeognn_v2_row_sequential_b32'
EXT = BASE.parent / 'qgeognn_v2_row_lcmd_to1485_20261026'
CURVES = ROOT / 'reports/free_llm_five_seeds_to1485_20261024/figures/validation_curves.json'
SEEDS = (157, 887, 2357, 6101, 12203)
FREE = 'free_llm32_scientist_v2'
curves = json.loads(CURVES.read_text())
rows, hashes = [], {}
for seed in SEEDS:
    directory = BASE / f'runtime/seed_{seed}/full_data_reference'
    audit_path = directory / 'model/fit_audit.json'
    freeze = json.loads((directory / 'pre_test_freeze.json').read_text())
    fit = json.loads(audit_path.read_text())
    digest = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    assert digest == freeze['files']['model/fit_audit.json']
    hashes[str(audit_path.relative_to(ROOT))] = digest
    lcmd_fit = json.loads((EXT / f'runtime/seed_{seed}/lcmd/round_36/model/fit_audit.json').read_text())
    assert fit['train_rows'] == 3330 and fit['validation_rows'] == 416
    assert freeze['test_truth_access_count'] == 0
    for key in ('initialization_hash', 'validation_ids_hash'):
        assert fit[key] == lcmd_fit[key]
    lcmd = {p['labels']: p['nrmse'] for p in curves[str(seed)]['lcmd']}
    free = {p['labels']: p['nrmse'] for p in curves[str(seed)][FREE]}
    rows.append(dict(seed=seed, full3330=fit['best_validation_combined_normalized_rmse'],
                     lcmd1005=lcmd[1005], lcmd1357=lcmd[1357], lcmd1485=lcmd[1485],
                     free1357=free[1357]))

means = {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key != 'seed'}
result = dict(snapshot=datetime.now().astimezone().isoformat(), validation_only=True,
              test_truth_accessed=False, full_training_labels=3330, validation_rows=416,
              rows=rows, means=means, full_reference_fit_audit_hashes=hashes,
              lcmd_gap_to_full=means['lcmd1485']-means['full3330'],
              lcmd_gain_1005_to1485=means['lcmd1005']-means['lcmd1485'],
              lcmd_gain_1357_to1485=means['lcmd1357']-means['lcmd1485'],
              free1357_gap_to_full=means['free1357']-means['full3330'])
(OUT / 'full_reference_validation_comparison_20261028.json').write_text(json.dumps(result, indent=2)+'\n')

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax in axes:
    for method, color, label in [('lcmd', '#159796', 'LCMD32'), (FREE, '#ad398c', 'Free LLM32')]:
        common = min(curves[str(s)][method][-1]['labels'] for s in SEEDS)
        x = [p['labels'] for p in curves[str(SEEDS[0])][method] if p['labels'] <= common]
        values = np.array([[p['nrmse'] for p in curves[str(s)][method] if p['labels'] <= common] for s in SEEDS])
        ax.plot(x, values.mean(axis=0), color=color, lw=2.3, label=label)
    ax.axhline(means['full3330'], color='#394357', ls='--', lw=1.7,
               label=f'Full training reference (3330): {means["full3330"]:.4f}')
    ax.grid(alpha=.17)
    ax.set_xlabel('Active training labels')
    ax.set_ylabel('Mean validation NRMSE (five seeds)')
axes[0].set(title='Verified learning curves', xlim=(333,1485))
axes[1].set(title='Late-budget detail', xlim=(1005,1485), ylim=(.315,.38))
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle('Active learning versus matched full-training reference', fontsize=15)
fig.text(.07,.015,'Same five splits and L0-fixed endpoint scales. Validation only; no test truth read.\n'
         'Free stops at the common verified budget; seeds 157/6101 retain Astra/high prefixes. Full-data fit is a reference, not a strict lower bound.',fontsize=8.5,color='#526174')
fig.tight_layout(rect=(0,.09,1,.95))
fig.savefig(OUT / 'full_reference_validation_comparison_20261028.png',dpi=180)
fig.savefig(OUT / 'full_reference_validation_comparison_20261028.pdf')
print(json.dumps(result['means']))
