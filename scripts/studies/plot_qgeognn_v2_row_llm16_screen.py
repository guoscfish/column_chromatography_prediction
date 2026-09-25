#!/usr/bin/env python3
"""Plot the completed, globally frozen CW16 + LLM16 development screen."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'studies/active_learning/qgeognn_v2_row_llm16_screen'


def main():
    if not (STUDY / 'global_pre_test_freeze.json').exists():
        raise RuntimeError('all trajectories must freeze before plotting results')
    curves = pd.read_csv(STUDY / 'results/learning_curves.csv')
    areas = pd.read_csv(STUDY / 'results/aulc.csv')
    styles = {'center_width_lcmd': ('CW32', '#555555', 'o'),
              'cw16_random16': ('CW16 + random16', '#0072B2', 's'),
              'cw16_llm16': ('CW16 + LLM16', '#009E73', '^')}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, seed in zip(axes, (157, 6101)):
        for method, (label, color, marker) in styles.items():
            rows = curves.loc[curves.seed.eq(seed) & curves.method.eq(method)].sort_values('budget')
            area = areas.loc[areas.seed.eq(seed) & areas.method.eq(method), 'AULC_333_429'].item()
            ax.plot(rows.budget, rows.combined_normalized_RMSE,
                    label=f'{label} (AULC {area:.4f})', color=color, marker=marker,
                    linewidth=1.8, markersize=5)
        ax.set(title=f'Development seed {seed}', xlabel='Labeled experimental records',
               xticks=[333, 365, 397, 429])
        ax.grid(axis='y', alpha=0.22)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8, loc='best', frameon=False)
    axes[0].set_ylabel('Combined NRMSE (lower is better)')
    fig.suptitle('Unchanged QGeoGNN: three-round row-split screen', fontsize=13)
    fig.tight_layout()
    target = STUDY / 'figures/learning_curves.png'
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(target)


if __name__ == '__main__':
    main()
