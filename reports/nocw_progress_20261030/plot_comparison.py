"""Offline progress plots from frozen fits; no model calls or label evaluation."""
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
os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

P = ROOT / 'studies/active_learning/qgeognn_v2_row_llm_nocw_ablation/revisions/five_seed_nocw_to1005_20261029'
H = ROOT / 'reports/free_llm_five_seeds_to1485_20261024/figures'
BASE = ROOT / 'studies/active_learning/qgeognn_v2_row_sequential_b32'
SEEDS = (157, 887, 2357, 6101, 12203)
FREE = 'free_llm32_scientist_v2'
METHODS = ('random', 'lcmd', 'hybrid', FREE, 'nocw')
NAMES = {'random': 'Random32', 'lcmd': 'LCMD32', 'hybrid': 'Hybrid32', FREE: 'Free LLM32 (CW-aware)', 'nocw': 'NoCW LLM32'}
COLORS = {'random': '#98A2B3', 'lcmd': '#148B88', 'hybrid': '#D89025', FREE: '#8055B8', 'nocw': '#E34A33'}
SOURCES = {}


def digest(path):
    path = Path(path)
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    SOURCES[str(path.relative_to(ROOT))] = value
    return value


def read(path):
    digest(path)
    return json.loads(Path(path).read_text())


spec = importlib.util.spec_from_file_location('historical_report', ROOT / 'scripts/report_free_lcmd.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
digest(Path(__file__))
curves = read(H / 'validation_curves.json')
historical_hashes = read(H / 'source_hashes.json')
monitor = read(P / 'progress/monitor_state.json')
snapshot = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
status = {}


def check_model(model, seed, budget, initial, historical=False):
    audit_path = model / 'fit_audit.json'
    audit = read(audit_path)
    if historical:
        assert SOURCES[str(audit_path.relative_to(ROOT))] == historical_hashes[str(audit_path.relative_to(ROOT))]
    assert audit['train_rows'] == budget
    assert audit['initialization_hash'] == initial['initialization_hash']
    assert audit['validation_ids_hash'] == initial['validation_ids_hash']
    assert audit['test_labels_used_for_fit_or_checkpoint_selection'] == 0
    assert digest(model / 'best.pt') == audit['checkpoint_sha256']
    with (model / 'history.csv').open() as h:
        chosen = [r for r in csv.DictReader(h) if int(r['epoch']) == audit['best_epoch']]
    digest(model / 'history.csv')
    assert len(chosen) == 1
    score = float(chosen[0]['validation_selection_score'])
    assert abs(score - audit['best_validation_combined_normalized_rmse']) < 1e-10
    return {'labels': budget, 'nrmse': score}, audit


for seed in SEEDS:
    key = str(seed)
    initial = read(BASE / f'runtime/seed_{seed}/lcmd/round_00/model/fit_audit.json')
    for method in METHODS[:-1]:
        rows = [r for r in curves[key][method] if r['labels'] <= 1005]
        assert [r['labels'] for r in rows] == list(range(333, 1006, 32))
        for r, row in enumerate(rows):
            if method == FREE:
                freeze = read(old.owner(seed, r) / f'runtime/seed_{seed}/{FREE}/round_{r:02d}/prediction_freeze.json')
                assert freeze['test_truth_access_count'] == 0
                model = Path(freeze['checkpoint_path']).parent
            else:
                model = BASE / f'runtime/seed_{seed}/{method}/round_{r:02d}/model'
            point, audit = check_model(model, seed, row['labels'], initial, historical=True)
            assert abs(point['nrmse'] - row['nrmse']) < 1e-10
        curves[key][method] = rows
    rows = []
    seen_ids = set()
    root = P / f'runtime/seed_{seed}/llm32_nocw'
    for r, directory in enumerate(sorted(root.glob('round_*'))):
        freeze_path = directory / 'prediction_freeze.json'
        if not freeze_path.exists():
            break
        assert directory.name == f'round_{r:02d}'
        freeze = read(freeze_path)
        assert freeze['test_truth_access_count'] == 0
        model = Path(freeze['checkpoint_path']).parent
        point, audit = check_model(model, seed, 333 + 32*r, initial)
        assert audit['checkpoint_sha256'] == freeze['checkpoint_sha256']
        assert audit['train_ids_hash'] == freeze['L_t_ids_hash']
        if r:
            selection = P / f'selections/seed_{seed}/llm32_nocw/round_{r-1:02d}'
            batch = read(selection / 'batch_freeze.json')
            assert len(batch['batch_ids']) == len(set(batch['batch_ids'])) == 32
            assert not seen_ids.intersection(batch['batch_ids'])
            seen_ids.update(batch['batch_ids'])
            for name, value in batch['artifact_hashes'].items():
                assert digest(selection / name) == value
            feedback = read(selection / 'feedback.json')
            assert feedback['batch_freeze_sha256'] == digest(selection / 'batch_freeze.json')
        rows.append(point)
    assert rows and len(seen_ids) == 32*(len(rows)-1)
    curves[key]['nocw'] = rows
    progress = next(x for x in monitor['last_observed_state'] if x['seed'] == seed)
    assert rows[-1]['labels'] == progress['completed_budget']
    status[key] = {'completed_labels': rows[-1]['labels'], 'nrmse': rows[-1]['nrmse'],
                   'frozen_batches': len(rows)-1, 'phase': progress['phase'],
                   'process_alive': progress['process_alive'],
                   'last_error_type': (progress.get('retry') or {}).get('last_error_type')}


plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'none'})


def draw(ax, subset, cap, band=False):
    for method in METHODS:
        stop = min(cap, min(curves[str(seed)][method][-1]['labels'] for seed in subset))
        x = np.arange(333, stop+1, 32)
        values = np.array([[row['nrmse'] for row in curves[str(seed)][method] if row['labels'] <= stop] for seed in subset])
        mean = values.mean(axis=0)
        ax.plot(x, mean, color=COLORS[method], lw=2.8 if method == 'nocw' else 1.7,
                marker='o' if method == 'nocw' else None, ms=3.7, label=NAMES[method], zorder=5 if method == 'nocw' else 3)
        if band and len(subset) > 1 and method in ('nocw', FREE):
            sd = values.std(axis=0, ddof=1)
            ax.fill_between(x, mean-sd, mean+sd, color=COLORS[method], alpha=.09)
    ax.grid(axis='y', alpha=.18)
    ax.set_xlabel('Active training labels')
    ax.set_ylabel('Validation NRMSE (lower is better)')


active = tuple(s for s in SEEDS if status[str(s)]['process_alive'])
common5 = min(status[str(s)]['completed_labels'] for s in SEEDS)
# Fixed cohort from the first snapshot; membership does not depend on scores or liveness.
continuation = (157, 2357, 12203)
common_continuation = min(status[str(s)]['completed_labels'] for s in continuation)
fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.2))
for ax, seed in zip(axes.flat, SEEDS):
    draw(ax, (seed,), 1005)
    last = curves[str(seed)]['nocw'][-1]
    phase = status[str(seed)]['phase']
    ax.set_title(f'Seed {seed} | NoCW L{last["labels"]} | {phase.lower()}', loc='left', fontsize=11, weight='bold')
    ax.annotate(f'L{last["labels"]}\n{last["nrmse"]:.3f}', (last['labels'], last['nrmse']), xytext=(8, 10), textcoords='offset points', color=COLORS['nocw'], fontsize=9)
    ax.set_xlim(325, 1020)
    ax.set_ylim(.22, 1.06)
    ax.set_xticks([333, 429, 653, 813, 1005])
ax = axes.flat[-1]
draw(ax, SEEDS, common5, band=True)
ax.set_title(f'Matched 5-seed mean through L{common5}', loc='left', fontsize=11, weight='bold')
ax.text(.03, .03, 'Seeds: ' + ', '.join(map(str, SEEDS)) + '\nSame seeds and budgets for every method', transform=ax.transAxes, fontsize=9, color='#475467')
ax.set_xlim(325, common5+12)
ax.set_xticks(list(range(333, common5+1, 64)))
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .95), ncol=5, frameon=False)
fig.suptitle('NoCW LLM32 vs existing methods | frozen validation progress', fontsize=19, weight='bold', y=.993)
fig.text(.055, .025, 'NoCW lines end at completed frozen fits; no extrapolation or replacement of stopped seeds. Shading: mean +/- SD for the two LLM methods.\n'
         'Historical Free LLM reference: seeds 157/6101 retain Astra/high prefixes; NoCW uses gpt-6.1-sol/xhigh throughout. Descriptive comparison, not a newly paired causal result.\n'
         f'Validation only; no test evaluation or selector feedback from these plots. Snapshot: {snapshot}', fontsize=9, color='#475467', linespacing=1.6)
fig.subplots_adjust(left=.055, right=.98, top=.865, bottom=.135, wspace=.27, hspace=.36)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'nocw_all_methods_per_seed.{ext}', dpi=170, facecolor='white')
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7))
for ax, subset, cap, title in [(axes[0], SEEDS, common5, f'All 5 seeds | common completed range to L{common5}'),
                               (axes[1], continuation, common_continuation, f'Fixed 3-seed cohort | common completed range to L{common_continuation}')]:
    draw(ax, subset, cap, band=False)
    ax.set_title(title, loc='left', fontsize=11, weight='bold')
    ax.set_xlim(329, cap+7)
    ticks = list(np.arange(333, cap+1, 32 if cap == common5 else 64))
    ax.set_xticks(sorted(set(ticks + [cap])))
    ax.text(.03, .03, 'Seeds: ' + ', '.join(map(str, subset)), transform=ax.transAxes, fontsize=9, color='#475467')
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .93), ncol=5, frameon=False, fontsize=9)
fig.suptitle('NoCW progress | comparison at identical seeds and label budgets', fontsize=16, weight='bold', y=.995)
fig.text(.06, .025, 'Each panel keeps a fixed seed set for all methods. The original 3-seed continuation cohort is a subset of the full 5-seed result.\n'
         'Historical Free includes Astra/high prefixes on 157/6101. Validation only; lower is better. No extrapolation.\n'
         f'Snapshot: {snapshot}', fontsize=9, color='#475467', linespacing=1.5)
fig.subplots_adjust(left=.065, right=.98, top=.80, bottom=.23, wspace=.25)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'nocw_matched_means.{ext}', dpi=180, facecolor='white')
plt.close(fig)

summary = {'snapshot_time': snapshot, 'status': status, 'common_all_five_budget': common5,
           'active_seeds': active, 'continuation_seed_subset': continuation,
           'continuation_subset_common_budget': common_continuation,
           'metric': 'validation_selection_score at the saved best epoch',
           'test_evaluation_performed': False, 'model_calls': 0,
           'comparison': 'Historical control; 157/6101 Free have Astra/high prefixes; NoCW Sol6.1/xhigh throughout',
           'matched_endpoints': {}}
for name, subset, budget in [('all_five', SEEDS, common5), ('continuation_subset', continuation, common_continuation)]:
    summary['matched_endpoints'][name] = {'seeds': subset, 'budget': budget, 'means': {
        method: float(np.mean([next(x['nrmse'] for x in curves[str(s)][method] if x['labels'] == budget) for s in subset])) for method in METHODS}}
for name, data in [('validation_curves.json', curves), ('summary.json', summary), ('source_audit.json', SOURCES)]:
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
