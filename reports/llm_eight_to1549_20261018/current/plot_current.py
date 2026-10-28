"""Regenerate the completed validation comparison from archived curves only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
OUT = Path(__file__).resolve().parent
curves = json.loads((OUT / 'validation_curves.json').read_text())
assert set(curves) == {'157', '6101'}
for methods in curves.values():
    for rows in methods.values():
        assert [row['labels'] for row in rows] == list(range(333, rows[-1]['labels'] + 1, 32))

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
    ax.set_xlim(325, 1568)
    ax.set_xticks([333, 429, 525, 621, 717, 813, 909, 1005, 1101, 1197, 1293, 1421, 1549])
    ax.set_xlabel('Active training labels')
    ax.grid(alpha=.15)
    ax.legend(fontsize=11, loc='upper right')
axes[0].set_ylabel('Validation NRMSE (lower is better)')
fig.suptitle('Four strategies toward 1549 | completed fits; in progress', fontsize=22, fontweight='bold')
fig.text(.07, .045, 'Hybrid: Astra/high through 813, then Sol/xhigh. Free: Astra/high through 525 (seed157) / 653 (seed6101), then Sol/xhigh.\nExploratory two-seed validation results; no test evaluation. Authorized correction deviations: Free157 r5; Free6101 r0/r19; Hybrid6101 r21 (extra turn03). Lines end at frozen fits.', fontsize=11, color='#555')
fig.tight_layout(rect=[0, .11, 1, .93])
for ext in ('png', 'svg', 'pdf'):
    fig.savefig(OUT / f'validation_progress.{ext}', dpi=200, bbox_inches='tight')
svg_path = OUT / 'validation_progress.svg'
svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
for name, data in [('validation_curves.json', curves), ('same_budget_comparison.json', summary)]:
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False))
