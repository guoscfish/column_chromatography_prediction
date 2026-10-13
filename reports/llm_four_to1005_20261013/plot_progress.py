"""Validation-only comparison from frozen fits; never reads evaluation truth."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REV = ROOT / 'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/sol_xhigh_hybrid_free_to1005_20261013'
curves = json.loads((ROOT / 'reports/llm_hybrid_complete_20261010/validation_curves.json').read_text())
sources = {}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def audit_row(path):
    audit = json.loads(path.read_text())
    sources[str(path.relative_to(ROOT))] = sha(path)
    return {'labels': audit['train_rows'], 'nrmse': audit['best_validation_combined_normalized_rmse']}

for seed in ('157', '6101'):
    for method in ('cw16_llm16_scientist_v2', 'free_llm32_scientist_v2'):
        rows = []
        for path in sorted((REV / f'runtime/seed_{seed}/{method}').glob('round_*/prediction_freeze.json')):
            freeze = json.loads(path.read_text())
            for kind in ('checkpoint', 'prediction'):
                assert sha(Path(freeze[kind + '_path'])) == freeze[kind + '_sha256'], path
            rows.append(audit_row(Path(freeze['checkpoint_path']).parent / 'fit_audit.json'))
        curves[seed][method] = rows
    cw = {row['labels']: row for row in curves[seed]['center_width_lcmd']}
    for path in (ROOT / f'studies/active_learning/qgeognn_v2_row_cw_lcmd_to_1005/runtime/seed_{seed}/center_width_lcmd').glob('round_*/model/fit_audit.json'):
        row = audit_row(path)
        cw[row['labels']] = row
    curves[seed]['center_width_lcmd'] = [cw[n] for n in sorted(cw)]
    curves[seed]['random32'] = sorted((audit_row(path) for path in (ROOT / f'studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_{seed}/random').glob('round_*/model/fit_audit.json')), key=lambda row: row['labels'])

summary = {}
for seed, methods in curves.items():
    budget = min(rows[-1]['labels'] for rows in methods.values())
    summary[seed] = {'common_budget': budget, 'methods': {}}
    for method, rows in methods.items():
        shared = [row for row in rows if row['labels'] <= budget]
        assert [row['labels'] for row in shared] == list(range(333, budget + 1, 32)), (seed, method)
        x, y = [row['labels'] for row in shared], [row['nrmse'] for row in shared]
        area = np.trapezoid(y, x) if hasattr(np, 'trapezoid') else np.trapz(y, x)
        summary[seed]['methods'][method] = {'latest_labels': rows[-1]['labels'], 'latest_validation_nrmse': rows[-1]['nrmse'], 'common_budget_validation_nrmse': y[-1], 'normalized_validation_AULC': float(area / (x[-1] - x[0]))}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
colors = {'cw16_llm16_scientist_v2': '#007f88', 'free_llm32_scientist_v2': '#b43e83', 'center_width_lcmd': '#e58a19', 'random32': '#8997a5'}
labels = {'cw16_llm16_scientist_v2': 'CW16 + LLM16', 'free_llm32_scientist_v2': 'Free LLM32', 'center_width_lcmd': 'CW32', 'random32': 'Random32'}
for ax, seed in zip(axes, ('157', '6101')):
    for method, rows in curves[seed].items():
        x, y = [r['labels'] for r in rows], [r['nrmse'] for r in rows]
        ax.plot(x, y, color=colors[method], label=f"{labels[method]}: {x[-1]} labels, {y[-1]:.4f}", lw=2.5, marker='o' if 'scientist' in method else None, ms=4, ls='-' if 'scientist' in method else '--')
    ax.axvline(813, color='#444', ls=':', alpha=.65)
    switch = 525 if seed == '157' else 653
    ax.axvline(switch, color=colors['free_llm32_scientist_v2'], ls=':', alpha=.45)
    ax.text(switch + 5, .98, f'Free switched after {switch}', fontsize=9, color=colors['free_llm32_scientist_v2'])
    ax.set_title(f'Seed {seed}', fontsize=19, fontweight='bold')
    ax.set_xlim(325, 1020)
    ax.set_xticks([333, 429, 525, 621, 717, 813, 909, 1005])
    ax.set_xlabel('Active training labels')
    ax.grid(alpha=.15)
    ax.legend(fontsize=11, loc='upper right')
axes[0].set_ylabel('Validation NRMSE (lower is better)')
fig.suptitle('Hybrid and Free continuation to 1005 | verified completed fits', fontsize=22, fontweight='bold')
fig.text(.07, .045, 'Hybrid: Astra/high through 813, then Sol/xhigh. Free: Astra/high through 525 (seed157) / 653 (seed6101), then Sol/xhigh.\nExploratory two-seed validation results; no test evaluation. Historical authorized correction exceptions are retained. Lines end at frozen fits.', fontsize=11, color='#555')
fig.tight_layout(rect=[0, .11, 1, .93])
for ext in ('png', 'svg', 'pdf'):
    fig.savefig(OUT / f'validation_progress.{ext}', dpi=200, bbox_inches='tight')
for name, data in [('validation_curves.json', curves), ('same_budget_comparison.json', summary), ('source_hashes.json', sources)]:
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False))
