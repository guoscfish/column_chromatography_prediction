#!/usr/bin/env python3
"""Render one reporting-only slide-13 curve from a pinned main snapshot.

Run from any directory with an environment containing numpy and matplotlib:
  python reports/ppt_202610_al/plot_slide13_cw32_vs_cw16_llm16.py

Source: guoscfish/column_chromatography_prediction main, verified 2026-10-12,
commit 42ab12600a353bba4a063ef5d1b623ee2f7b976a. Snapshot captured at
2026-10-11T23:28:59+08:00. Reads only the committed snapshot; no repository
model/selector imports, training, LLM calls, checkpoint or test-label access.
Writes only the requested PNG and SVG. Missing/nonmatching nodes fail before
creating figures. No interpolation, smoothing or uncertainty bands.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REV = '42ab12600a353bba4a063ef5d1b623ee2f7b976a'
SOURCE = 'reports/active_learning_latest/snapshot.json'
OUTPUT = Path(__file__).resolve().parent / 'figures'
STEM = 'slide13_cw32_vs_cw16_llm16_333_813'
SEEDS = ('157', '6101')
METHODS = (
    ('CW32 (LCMD)', 'CW32（纯数值策略）', '#617A8B'),
    ('CW16 + LLM16', 'CW16+LLM16（LLM 混合选点）', '#8C0000'),
)
EXPECTED = np.arange(333, 814, 32)


def frozen_curves():
    raw = subprocess.check_output(['git', 'show', f'{REV}:{SOURCE}'], cwd=ROOT)
    snapshot = json.loads(raw)
    if snapshot.get('evaluation') != 'validation_only':
        raise ValueError('Snapshot evaluation is not validation_only; stop.')
    curves = {}
    for method, _, _ in METHODS:
        rows = []
        for seed in SEEDS:
            points = [p for p in snapshot['curves'][seed][method]
                      if 333 <= p['labels'] <= 813]
            budgets = np.asarray([p['labels'] for p in points])
            if not np.array_equal(budgets, EXPECTED):
                raise ValueError(f'{seed}/{method}: missing or inconsistent budgets {budgets.tolist()}; stop.')
            values = np.asarray([p['nrmse'] for p in points], dtype=np.float64)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f'{seed}/{method}: invalid validation metric; stop.')
            rows.append(values)
        curves[method] = np.stack(rows)
    return curves, hashlib.sha256(raw).hexdigest()


def chinese_font():
    for path in ('/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
                 '/Library/Fonts/Arial Unicode.ttf'):
        if Path(path).is_file():
            font_manager.fontManager.addfont(path)
            return font_manager.FontProperties(fname=path).get_name()
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ('Noto Sans CJK SC', 'Microsoft YaHei', 'SimHei', 'PingFang SC', 'Arial Unicode MS'):
        if name in available:
            return name
    raise RuntimeError('A Chinese-capable font is required; stop rather than render missing glyphs.')


def main():
    curves, source_hash = frozen_curves()  # All four curves validated first.
    plt.rcParams.update({
        'font.family': chinese_font(), 'font.size': 17,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'axes.unicode_minus': False,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#A5A5A5', 'axes.labelcolor': '#303030',
        'xtick.color': '#303030', 'ytick.color': '#303030',
        'svg.fonttype': 'path', 'svg.hashsalt': 'slide13_cw32_vs_cw16_llm16',
    })
    fig = plt.figure(figsize=(12, 8))  # 3:2, 3600 x 2400 at 300 dpi.
    ax = fig.add_axes([.135, .18, .81, .63])
    fig.text(.54, .948, 'CW32 与 CW16+LLM16 主动学习曲线',
             ha='center', va='center', fontsize=23, color='#303030')
    for method, label, color in METHODS:
        for values in curves[method]:
            ax.plot(EXPECTED, values, color=color, alpha=.30, linewidth=1.15,
                    linestyle='-', label='_nolegend_', zorder=2)
        ax.plot(EXPECTED, curves[method].mean(axis=0), color=color,
                linewidth=3.1, linestyle='-', label=label, zorder=3)
    if len(ax.lines) != 6:
        raise AssertionError('Exactly four seed lines and two mean lines are required.')
    # Separate legend area above the axes; it cannot cover any curve.
    fig.legend(*ax.get_legend_handles_labels(), loc='center',
               bbox_to_anchor=(.54, .875), ncol=2, frameon=False,
               fontsize=15.5, handlelength=2.5, columnspacing=1.6)
    ax.set_xlim(325, 821)
    ax.set_ylim(.285, 1.055)
    ax.set_xticks([333, 429, 525, 653, 813])
    ax.set_yticks(np.arange(.3, 1.01, .1))
    ax.tick_params(labelsize=16, length=5, width=.8, pad=7)
    ax.set_xlabel('当前已标注训练样本数', fontsize=19, labelpad=17)
    ax.set_ylabel('验证集 Combined NRMSE（越低越好）', fontsize=18, labelpad=16)
    ax.grid(False)
    fig.text(.54, .035, '浅色细线：seed 157 / 6101；粗实线：两 seed 逐预算算术均值',
             ha='center', fontsize=12, color='#707070')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / (STEM + '.png'), dpi=300)
    fig.savefig(OUTPUT / (STEM + '.svg'), metadata={'Date': None})
    plt.close(fig)
    print(f'Validated four aligned 16-point validation curves, 333–813; source commit {REV}; SHA256 {source_hash}')
    print(OUTPUT / (STEM + '.png'))
    print(OUTPUT / (STEM + '.svg'))


if __name__ == '__main__':
    main()
