"""Plot frozen validation results only, without test labels or experiment calls."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent/'figures'
OUT.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(OUT/'matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SEEDS = (157, 887, 2357, 6101, 12203)
METHODS = ('random', 'lcmd', 'hybrid', 'free_llm32_scientist_v2')
FREE = METHODS[-1]
BASE = ROOT/'studies/active_learning/qgeognn_v2_row_sequential_b32'
REVISIONS = ROOT/'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions'
OLD = REVISIONS/'sol_xhigh_hybrid_free_to1005_20261013'
NEW = REVISIONS/'sol61_xhigh_free_five_seeds_to1005_20261023'
COLORS = {'random': '#8C99A7', 'lcmd': '#148B88', 'hybrid': '#D89025', FREE: '#A63B83'}
NAMES = {'random': 'Random32', 'lcmd': 'LCMD32', 'hybrid': 'Traditional Hybrid32', FREE: 'Free LLM32'}
SOURCES = {}


def sha(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    value = h.hexdigest()
    SOURCES[str(path.relative_to(ROOT))] = value
    return value


def read(path):
    sha(path)
    return json.loads(Path(path).read_text())


def audited_fit(path, seed, budget, initial):
    audit = read(path)
    assert audit['train_rows'] == budget
    assert audit['initialization_hash'] == initial['initialization_hash']
    assert audit['validation_ids_hash'] == initial['validation_ids_hash']
    assert audit['test_labels_used_for_fit_or_checkpoint_selection'] == 0
    assert sha(path.parent/'best.pt') == audit['checkpoint_sha256']
    assert sha(path.parent/'predictions.csv.gz') == audit['prediction_sha256']
    return {'labels': budget, 'nrmse': audit['best_validation_combined_normalized_rmse']}


curves = {}
for seed in SEEDS:
    initial = read(BASE/f'runtime/seed_{seed}/lcmd/round_00/model/fit_audit.json')
    curves[str(seed)] = {}
    for method in METHODS[:-1]:
        rows = []
        for r in range(22):
            path = BASE/f'runtime/seed_{seed}/{method}/round_{r:02d}/model/fit_audit.json'
            rows.append(audited_fit(path, seed, 333+32*r, initial))
        curves[str(seed)][method] = rows
    owner = OLD if seed in (157, 6101) else NEW
    rows = []
    for path in sorted((owner/f'runtime/seed_{seed}/{FREE}').glob('round_*/prediction_freeze.json')):
        r = int(path.parent.name.split('_')[1])
        assert r == len(rows)
        frozen = read(path)
        assert frozen['test_truth_access_count'] == 0
        for kind in ('checkpoint', 'prediction'):
            assert sha(frozen[kind+'_path']) == frozen[kind+'_sha256']
        fit_path = Path(frozen['checkpoint_path']).parent/'fit_audit.json'
        rows.append(audited_fit(fit_path, seed, 333+32*r, initial))
    assert rows
    if rows[-1]['labels'] == 1005:
        result = read(owner/f'{seed}_{FREE}_result.json')
        assert result['final_labels'] == 1005 and len(rows) == 22
        assert len(result['selected_ids']) == len(set(result['selected_ids'])) == 672
        assert result['test_evaluation_performed'] is False
        for relative, expected in result['files'].items():
            assert sha(owner/relative) == expected, relative
        assert rows == [{'labels': x['labels'], 'nrmse': x['validation_nrmse']}
                        for x in result['validation_learning_curve']]
    else:
        state = read(Path(__file__).resolve().parent/f'status_seed_{seed}.json')
        assert rows[-1]['labels'] == state['completed_labels']
    curves[str(seed)][FREE] = rows

common = min(curves[str(seed)][FREE][-1]['labels'] for seed in SEEDS)
complete = common == 1005
budgets = np.arange(333, common+1, 32)
summary = {'snapshot_time': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
           'status': 'COMPLETE' if complete else 'IN_PROGRESS',
           'common_budget': common, 'all_seed_count': 5, 'test_evaluation_performed': False,
           'latest_free': {str(seed): curves[str(seed)][FREE][-1] for seed in SEEDS},
           'common_budget_methods': {},
           'model_note': '157/6101 retain historical Astra/high prefixes; 887/2357/12203 use gpt-6.1-sol/xhigh throughout.'}
mean_curves = {}
for method in METHODS:
    values = np.array([[x['nrmse'] for x in curves[str(seed)][method] if x['labels'] <= common]
                       for seed in SEEDS])
    assert values.shape == (5, len(budgets))
    mean_curves[method] = values.mean(axis=0)
    area = np.trapezoid(values, budgets, axis=1) if hasattr(np, 'trapezoid') else np.trapz(values, budgets, axis=1)
    summary['common_budget_methods'][method] = {
        'mean_validation_nrmse': float(values[:, -1].mean()),
        'sd_across_seeds': float(values[:, -1].std(ddof=1)),
        'per_seed_validation_nrmse': dict(zip(map(str, SEEDS), values[:, -1].tolist())),
        'mean_normalized_aulc': float((area/(common-333)).mean())}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': '#3D4652', 'text.color': '#223044',
                     'axes.titleweight': 'bold', 'svg.fonttype': 'none'})
fig, axes = plt.subplots(2, 3, figsize=(17.4, 10.2))
for index, (ax, seed) in enumerate(zip(axes.flat, SEEDS)):
    for method in METHODS:
        rows = curves[str(seed)][method]
        x = [r['labels'] for r in rows]
        y = [r['nrmse'] for r in rows]
        ax.plot(x, y, color=COLORS[method], lw=2.5 if method == FREE else 1.6,
                marker='o' if method == FREE else None, ms=3,
                alpha=1 if method == FREE else .85, zorder=4 if method == FREE else 2,
                label=NAMES[method])
    end = curves[str(seed)][FREE][-1]
    ax.scatter(end['labels'], end['nrmse'], s=56, color=COLORS[FREE], edgecolor='white', zorder=6)
    ax.set_title(f'{chr(65+index)}   Seed {seed}', loc='left', fontsize=13, pad=13)
    endpoint = f"Free: {end['labels']} labels  |  NRMSE {end['nrmse']:.3f}"
    ax.text(.98, .96, endpoint, ha='right', va='top', transform=ax.transAxes,
            color=COLORS[FREE], fontsize=10, fontweight='bold')
    if end['labels'] < 1005:
        ax.axvline(end['labels'], color=COLORS[FREE], alpha=.25, lw=1, ls=':')
        ax.text(.98, .885, 'Run in progress', ha='right', va='top',
                transform=ax.transAxes, fontsize=8.5, color='#687387')
    ax.set_xlim(325, 1020)
    ax.set_ylim(.18, 1.12)
    ax.set_xticks([333, 525, 717, 1005])
    ax.set_xlabel('Active training labels')
    ax.set_ylabel('Validation NRMSE')
    ax.grid(axis='y', alpha=.16)

ax = axes.flat[-1]
for method in METHODS:
    y = mean_curves[method]
    ax.plot(budgets, y, color=COLORS[method], lw=2.5 if method == FREE else 1.8,
            marker='o' if method == FREE else None, ms=3, label=NAMES[method])
    label_dy = {'lcmd': 14, 'hybrid': 2, FREE: -12}.get(method, 0) if complete else (6 if method == 'lcmd' else (-6 if method == 'hybrid' else 0))
    ax.annotate(f'{y[-1]:.3f}', (common, y[-1]), xytext=(7, label_dy), textcoords='offset points',
                va='center', color=COLORS[method], fontsize=10, fontweight='bold')
ax.set_title(f'F   Matched comparison: all 5 seeds', loc='left', fontsize=13, pad=13)
ax.text(.02, .965, f'Fixed n = 5 at every point; common endpoint = {common}',
        transform=ax.transAxes, va='top', fontsize=9, color='#687387')
ax.set_xlim(325, common+36)
ax.set_ylim(.25, 1.02)
ax.set_xticks([333, 525, 717, 1005] if complete else [333, 397, 461, 525, common])
ax.set_xlabel('Active training labels (shared range only)')
ax.set_ylabel('Mean validation NRMSE')
ax.grid(axis='y', alpha=.16)
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .955),
           ncol=4, frameon=False, fontsize=11, handlelength=3)
fig.suptitle('Free LLM vs. traditional methods | '+('five seeds completed to 1005' if complete else 'current frozen validation results'),
             fontsize=19, fontweight='bold', y=.995)
fig.text(.055, .035,
    'Lower is better. All curves use verified frozen fits and the same initialization and validation split within each seed.\n'
    'Historical seed 157/6101 retain Astra/high prefixes. Seeds 887/2357/12203 use Sol6.1/xhigh throughout.\n'
    f'Panel F compares the same five seeds at every budget (333–{common}); validation only, no test evaluation.',
    fontsize=10, color='#5D6878', linespacing=1.5)
fig.subplots_adjust(left=.055, right=.97, top=.875, bottom=.16, wspace=.28, hspace=.43)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT/f'current_five_seed_comparison.{ext}', dpi=190, facecolor='white')
plt.close(fig)

# A second view makes the variation among the five Free LLM trajectories clear.
fig, ax = plt.subplots(figsize=(11.6, 6.6))
seed_colors = dict(zip(SEEDS, ('#5276B7', '#D67835', '#20958C', '#8C6CB0', '#B74571')))
for seed in SEEDS:
    rows = curves[str(seed)][FREE]
    x, y = [r['labels'] for r in rows], [r['nrmse'] for r in rows]
    ax.plot(x, y, lw=2.4, color=seed_colors[seed], marker='o', ms=3.5,
            label=f"Seed {seed}: {x[-1]} labels, {y[-1]:.3f}")
    ax.scatter(x[-1], y[-1], s=60, color=seed_colors[seed], edgecolor='white', zorder=5)
ax.axvline(common, color='#778294', alpha=.65, ls=':', lw=1.1)
ax.text(.02, .97, f'Common budget: {common}', transform=ax.transAxes, va='top', fontsize=10, color='#687387')
ax.set_title('Free LLM32 | five-seed learning curves', loc='left', fontsize=18, pad=19)
ax.set_xlabel('Active training labels')
ax.set_ylabel('Validation NRMSE (lower is better)')
ax.set_xlim(325, 1020)
ax.set_ylim(.25, 1.12)
ax.set_xticks([333, 429, 525, 621, 717, 813, 909, 1005])
ax.grid(axis='y', alpha=.15)
ax.legend(loc='upper right', frameon=False, fontsize=10)
fig.text(.09, .04,
    ('All five seeds completed to 1005 labels; models, predictions and selection freezes verified.\n' if complete else 'Lines stop at verified completed fits.\n')+
    'Seeds 157/6101 retain historical Astra/high prefixes; the three new seeds use Sol6.1/xhigh throughout.',
    fontsize=9.5, color='#687387')
fig.tight_layout(rect=[0, .10, 1, 1])
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT/f'free_llm_five_seed_curves.{ext}', dpi=190, facecolor='white')
plt.close(fig)
for name, data in [('validation_curves.json', curves), ('current_summary.json', summary),
                   ('source_hashes.json', SOURCES)]:
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(summary, ensure_ascii=False), flush=True)
