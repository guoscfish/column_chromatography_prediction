"""Offline validation plots from completed, audited fits only."""
import csv
import hashlib
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', '/tmp/nocw_plot_matplotlib_20261101')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REV = ROOT / 'studies/active_learning/qgeognn_v2_row_llm_nocw_ablation/revisions'
SOURCE = REV / 'five_seed_nocw_to1005_20261029'
EXT = REV / 'five_seed_nocw_to1389_20261101'
BASE = ROOT / 'studies/active_learning/qgeognn_v2_row_sequential_b32'
LCMD = BASE.parent / 'qgeognn_v2_row_lcmd_to1485_20261026'
HIST = ROOT / 'reports/free_llm_five_seeds_to1485_20261024/figures'
SEEDS = (157, 887, 2357, 6101, 12203)
FREE = 'free_llm32_scientist_v2'
METHODS = ('random', 'lcmd', 'hybrid', FREE, 'nocw')
NAMES = {'random': 'Random32', 'lcmd': 'LCMD32', 'hybrid': 'Hybrid32', FREE: 'Free LLM32 (CW-aware)', 'nocw': 'NoCW LLM32'}
COLORS = {'random': '#98A2B3', 'lcmd': '#148B88', 'hybrid': '#D89025', FREE: '#8055B8', 'nocw': '#E34A33'}
SOURCES = {}


def sha(path):
    path = Path(path)
    with path.open('rb') as handle:
        value = hashlib.file_digest(handle, 'sha256').hexdigest()
    SOURCES[str(path.relative_to(ROOT))] = value
    return value


def read(path):
    sha(path)
    return json.loads(Path(path).read_text())


def fit_point(model, budget, initial, historical=False):
    audit_path = model / 'fit_audit.json'
    audit = read(audit_path)
    assert audit['train_rows'] == budget
    assert audit['initialization_hash'] == initial['initialization_hash']
    assert audit['validation_ids_hash'] == initial['validation_ids_hash']
    assert audit['test_labels_used_for_fit_or_checkpoint_selection'] == 0
    if historical:
        for path in (audit_path, model / 'best.pt', model / 'predictions.csv.gz'):
            assert sha(path) == historical_hashes[str(path.relative_to(ROOT))], path
    else:
        assert sha(model / 'best.pt') == audit['checkpoint_sha256']
        assert sha(model / 'predictions.csv.gz') == audit['prediction_sha256']
    history = model / 'history.csv'
    with history.open() as handle:
        chosen = [row for row in csv.DictReader(handle) if int(row['epoch']) == audit['best_epoch']]
    sha(history)
    assert len(chosen) == 1
    score = float(chosen[0]['validation_selection_score'])
    assert abs(score - audit['best_validation_combined_normalized_rmse']) < 1e-10
    return {'labels': budget, 'nrmse': score}, audit


spec = importlib.util.spec_from_file_location('historical_report', ROOT / 'scripts/report_free_lcmd.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
sha(Path(__file__))
sha(ROOT / 'scripts/report_free_lcmd.py')
historical_curves = read(HIST / 'validation_curves.json')
historical_hashes = read(HIST / 'source_hashes.json')
monitor = read(EXT / 'progress/monitor_state.json')
assert monitor['target_budget'] == 1389 and not monitor['active_issues']
assert monitor['amendment_sha256'] == '910939d995cb388d6c55dd74d8bb6282e58411e63d01a2283c0bcb6abbe43de4'
assert monitor['paid_calls_during_audit'] == monitor['test_truth_reads'] == 0
sha(EXT / 'extension_to1389_amendment.json')
snapshot = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
curves, status = {}, {}

for seed in SEEDS:
    key = str(seed)
    progress = monitor['seeds'][key]
    assert progress['process_alive'] and progress['lock_held']
    budget = progress['budget']
    initial = read(BASE / f'runtime/seed_{seed}/lcmd/round_00/model/fit_audit.json')
    methods = {}
    for method in METHODS[:-1]:
        rows = [row for row in historical_curves[key][method] if row['labels'] <= budget]
        assert [row['labels'] for row in rows] == list(range(333, rows[-1]['labels'] + 1, 32))
        for r, row in enumerate(rows):
            if method == FREE:
                frozen = read(old.owner(seed, r) / f'runtime/seed_{seed}/{FREE}/round_{r:02d}/prediction_freeze.json')
                assert frozen['test_truth_access_count'] == 0
                model = Path(frozen['checkpoint_path']).parent
            else:
                parent = LCMD if method == 'lcmd' and r >= 22 else BASE
                model = parent / f'runtime/seed_{seed}/{method}/round_{r:02d}/model'
            point, _ = fit_point(model, row['labels'], initial, historical=True)
            assert abs(point['nrmse'] - row['nrmse']) < 1e-10
        methods[method] = rows
    run = EXT if progress['mode'] == 'EXTENSION' else SOURCE
    if run == EXT:
        read(EXT / f'progress/import_seed_{seed}.json')
    rows, seen = [], set()
    for r in range((budget - 333) // 32 + 1):
        frozen = read(run / f'runtime/seed_{seed}/llm32_nocw/round_{r:02d}/prediction_freeze.json')
        assert frozen['test_truth_access_count'] == 0
        point, audit = fit_point(Path(frozen['checkpoint_path']).parent, 333 + 32*r, initial)
        assert frozen['checkpoint_sha256'] == audit['checkpoint_sha256']
        assert frozen['prediction_sha256'] == audit['prediction_sha256']
        assert frozen['L_t_ids_hash'] == audit['train_ids_hash']
        if r:
            selection = run / f'selections/seed_{seed}/llm32_nocw/round_{r-1:02d}'
            batch_path = selection / 'batch_freeze.json'
            batch = read(batch_path)
            ids = batch['batch_ids']
            assert len(ids) == len(set(ids)) == 32 and not seen.intersection(ids)
            assert not batch['pending_ids']
            seen.update(ids)
            for name, digest in batch['artifact_hashes'].items():
                assert sha(selection / name) == digest
            feedback = read(selection / 'feedback.json')
            receipt = read(selection / 'label_access_receipt.json')
            assert feedback['batch_freeze_sha256'] == receipt['batch_freeze_sha256'] == sha(batch_path)
        rows.append(point)
    assert rows[-1]['labels'] == budget and len(seen) == budget - 333
    methods['nocw'] = rows
    curves[key] = methods
    status[key] = {'completed_labels': budget, 'completed_batches': len(rows)-1,
                   'validation_nrmse': rows[-1]['nrmse'], 'mode': progress['mode'],
                   'process_alive': True, 'lock_held': True,
                   'retry_phase': progress['retry_phase'], 'error_type': progress['error_type']}

common = min(row['completed_labels'] for row in status.values())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'none'})
visible = [row['nrmse'] for methods in curves.values() for rows in methods.values() for row in rows]
ylim = (min(.22, min(visible)-.035), max(1.06, max(visible)+.035))


def draw(ax, subset, cap, band=False):
    for method in METHODS:
        stop = min(cap, min(curves[str(seed)][method][-1]['labels'] for seed in subset))
        x = np.arange(333, stop+1, 32)
        values = np.array([[row['nrmse'] for row in curves[str(seed)][method] if row['labels'] <= stop] for seed in subset])
        assert values.shape == (len(subset), len(x))
        mean = values.mean(axis=0)
        ax.plot(x, mean, color=COLORS[method], lw=2.8 if method == 'nocw' else 1.6,
                marker='o' if method == 'nocw' else None, ms=3.6,
                label=NAMES[method], zorder=6 if method == 'nocw' else 3)
        if band and len(subset) > 1 and method in ('nocw', FREE):
            sd = values.std(axis=0, ddof=1)
            ax.fill_between(x, mean-sd, mean+sd, color=COLORS[method], alpha=.09)
    ax.grid(axis='y', alpha=.18)
    ax.set_xlabel('Active training labels')
    ax.set_ylabel('Validation NRMSE (lower is better)')
    ax.set_ylim(*ylim)


fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.5))
for ax, seed in zip(axes.flat, SEEDS):
    last = curves[str(seed)]['nocw'][-1]
    draw(ax, (seed,), last['labels'])
    ax.set_title(f'Seed {seed} | NoCW L{last["labels"]} | {status[str(seed)]["completed_batches"]}/33 batches',
                 loc='left', fontsize=11, weight='bold')
    ax.annotate(f'L{last["labels"]}\n{last["nrmse"]:.3f}', (last['labels'], last['nrmse']),
                xytext=(9, 12), textcoords='offset points', fontsize=9, color=COLORS['nocw'],
                bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':.8, 'pad':1.5})
    ax.axvline(1389, ls=':', lw=1.1, color='#667085', alpha=.7)
    ax.text(1389, ylim[1]-.025, 'target', ha='right', va='top', fontsize=8, color='#667085')
    ax.set_xlim(318, 1410)
    ax.set_xticks([333, 653, 973, 1197, 1389])
ax = axes.flat[-1]
draw(ax, SEEDS, common, band=True)
ax.set_title(f'Matched 5-seed mean through L{common}', loc='left', fontsize=11, weight='bold')
ax.set_xlim(318, common+20)
ax.set_xticks([333, 461, 589, 717, 845, common])
ax.text(.035, .035, 'All methods use the same five seeds and budgets.\nShading: +/- 1 sample SD for the LLM methods.',
        transform=ax.transAxes, fontsize=8.5, color='#475467')
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .95), ncol=5, frameon=False)
fig.suptitle('NoCW LLM32 | latest completed validation fits | target L1389', fontsize=18, weight='bold', y=.991)
fig.text(.055, .028,
         'Only completed frozen fits are plotted. Individual panels show available references up to that seed\'s completed budget; Random/Hybrid end at L1005.\n'
         'Historical Free LLM reference includes Astra/high prefixes for seeds 157/6101; NoCW uses gpt-6.1-sol/xhigh throughout. Descriptive historical comparison.\n'
         f'Validation only; no test evaluation, new training, or model calls. Snapshot: {snapshot}',
         fontsize=9, color='#475467', linespacing=1.6)
fig.subplots_adjust(left=.055, right=.98, top=.86, bottom=.135, wspace=.27, hspace=.36)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'nocw_latest_comparison.{ext}', dpi=180, facecolor='white')
plt.close(fig)

fig, ax = plt.subplots(figsize=(11.4, 6.6))
seed_colors = ('#2563EB', '#D97706', '#059669', '#9333EA', '#DB2777')
for seed, color in zip(SEEDS, seed_colors):
    rows = curves[str(seed)]['nocw']
    ax.plot([r['labels'] for r in rows], [r['nrmse'] for r in rows], color=color,
            marker='o', ms=3.7, lw=1.9, label=f'Seed {seed} - L{rows[-1]["labels"]}')
ax.axvline(1389, color='#667085', ls=':', lw=1.2)
ax.set(xlim=(318,1410), ylim=ylim, xlabel='Active training labels', ylabel='Validation NRMSE (lower is better)',
       title='NoCW LLM32 | five seeds | completed frozen fits')
ax.set_xticks([333, 493, 653, 813, 973, 1133, 1293, 1389])
ax.grid(axis='y', alpha=.18)
ax.legend(frameon=False, ncol=2, loc='upper right')
ax.text(1389, .60, 'L1389 target', ha='right', va='center', color='#667085', fontsize=9)
fig.text(.08,.025, f'Each seed ends at its latest completed fit. Validation only; no extrapolation. Snapshot: {snapshot}', fontsize=9, color='#475467')
fig.subplots_adjust(left=.09, right=.98, top=.91, bottom=.13)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'nocw_five_seed_curves.{ext}', dpi=180, facecolor='white')
plt.close(fig)

summary = {'snapshot_time': snapshot, 'audited_at_utc': monitor['checked_utc'], 'target_budget':1389,
           'status':status, 'common_five_seed_budget':common,
           'metric':'saved best-epoch validation_selection_score',
           'test_truth_reads':0, 'model_calls':0, 'new_fits_performed':0,
           'matched_means':{method:float(np.mean([next(row['nrmse'] for row in curves[str(seed)][method] if row['labels']==common) for seed in SEEDS])) for method in METHODS}}
for name, data in (('validation_curves.json',curves), ('summary.json',summary), ('source_audit.json',SOURCES)):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
