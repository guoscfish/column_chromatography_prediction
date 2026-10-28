"""Regenerate the completed validation comparison from archived curves only."""
import json
import hashlib
import copy
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BASELINE = ROOT / 'reports/llm_four_to1005_20261013'
curves = json.loads((BASELINE / 'validation_curves.json').read_text())
sources = json.loads((BASELINE / 'source_hashes.json').read_text())
for name, digest in sources.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
for name in ('validation_curves.json', 'source_hashes.json'):
    source = BASELINE / name
    sources[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
assert set(curves) == {'157', '6101'}
for methods in curves.values():
    for rows in methods.values():
        assert [row['labels'] for row in rows] == list(range(333, 1006, 32))

DUAL = 'free_llm32_dual_expert_v1'
for seed, revision in [('157', 'sol61_xhigh_seed157_from621_to813_20261014'), ('6101', 'sol61_xhigh_seed6101_to813_20261014')]:
    study = ROOT / 'studies/active_learning/qgeognn_v2_row_llm_dual_expert_v1/revisions' / revision
    rows = []
    for r in range(16):
        freeze = study / f'runtime/seed_{seed}/{DUAL}/round_{r:02d}/prediction_freeze.json'
        if not freeze.exists(): break
        data = json.loads(freeze.read_text())
        for kind in ('checkpoint', 'prediction'):
            artifact = Path(data[f'{kind}_path'])
            assert hashlib.sha256(artifact.read_bytes()).hexdigest() == data[f'{kind}_sha256']
            sources[str(artifact.relative_to(ROOT))] = data[f'{kind}_sha256']
        audit_path = Path(data['checkpoint_path']).parent / 'fit_audit.json'
        audit = json.loads(audit_path.read_text())
        assert audit['train_rows'] == 333 + 32*r
        rows.append({'labels': audit['train_rows'], 'nrmse': audit['best_validation_combined_normalized_rmse']})
        for source in (freeze, audit_path): sources[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    assert rows
    curves[seed][DUAL] = rows
(OUT / 'progress_plot_source_hashes.json').write_text(json.dumps(sources, indent=2)+'\n')

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
colors = {'cw16_llm16_scientist_v2': '#007f88', 'free_llm32_scientist_v2': '#b43e83', 'center_width_lcmd': '#e58a19', 'random32': '#8997a5', DUAL: '#2463eb'}
labels = {'cw16_llm16_scientist_v2': 'CW16 + LLM16', 'free_llm32_scientist_v2': 'Free LLM32', 'center_width_lcmd': 'CW32', 'random32': 'Random32', DUAL: 'Dual Expert'}
for ax, seed in zip(axes, ('157', '6101')):
    for method, rows in curves[seed].items():
        x, y = [r['labels'] for r in rows], [r['nrmse'] for r in rows]
        ax.plot(x, y, color=colors[method], label=f"{labels[method]}: {x[-1]} labels, {y[-1]:.4f}", lw=2.5, marker='D' if method == DUAL else ('o' if 'scientist' in method else None), ms=4, ls='-' if method == DUAL or 'scientist' in method else '--')
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
fig.suptitle('Hybrid, Free and Dual Expert | verified completed fits', fontsize=22, fontweight='bold')
fig.text(.07, .045, 'Hybrid/Free/controls: original completed curves through 1005. Dual Expert: blue diamonds, completed fits only.\nDual: seed157 Astra/high through397 then Sol6.1/xhigh; seed6101 Sol6.1/xhigh from333. Target813. Exploratory validation only; no test evaluation.', fontsize=11, color='#555')
fig.tight_layout(rect=[0, .11, 1, .93])
for ext in ('png', 'svg', 'pdf'):
    fig.savefig(OUT / f'validation_progress_with_dual.{ext}', dpi=200, bbox_inches='tight')
svg_path = OUT / 'validation_progress_with_dual.svg'
svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
for name, data in [('validation_curves_with_dual.json', curves), ('same_budget_comparison_with_dual.json', summary)]:
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False))
