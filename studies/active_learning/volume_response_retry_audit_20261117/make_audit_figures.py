#!/usr/bin/env python3
"""Plot existing frozen inference/retry tables only; never read control-fit results."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/audit_scientific_figures_mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
METHODS = ['historical_free_mixed_model', 'new_free_cw_sol61_xhigh', 'nocw']
LABELS = ['Historical CW-Free', 'New CW-Free', 'NoCW-Free']
BLUE, ORANGE, GRAY = '#0072B2', '#D55E00', '#AAB1B8'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'audit_figures')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {}
    outputs = []

    def table(name, columns=None):
        path = HERE / name
        raw = path.read_bytes()
        assert raw == path.read_bytes(), ('Changing figure input', name)
        inputs[name] = {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        return pd.read_csv(io.BytesIO(raw), usecols=columns)

    def save(fig, name):
        for extension in ('png', 'pdf'):
            path = args.output_dir / f'{name}.{extension}'
            fig.savefig(path, dpi=190, facecolor='white')
            outputs.append({'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        plt.close(fig)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'axes.titleweight': 'normal'})

    # Each small dot is one campaign mean, not an independent condition pair.
    response = table('frozen_response_seed_budget.csv', ['seed', 'budget', 'axis', 'target', 'pairs',
                                                        'observed_abs_mean', 'predicted_abs_mean'])
    assert sorted(response.seed.unique()) == [157, 887, 2357, 6101, 12203]
    assert sorted(response.budget.unique()) == [525, 1005]
    for budget, factor, count in [(525, 'V/ul', 464), (1005, 'V/ul', 1393), (525, 'PE/EA', 1277), (1005, 'PE/EA', 4054)]:
        assert int(response[(response.budget == budget) & (response.axis == factor) & (response.target == 'V1')].pairs.sum()) == count
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 7.8))
    fig.subplots_adjust(left=.085, right=.98, top=.83, bottom=.16, hspace=.47, wspace=.27)
    for row, (factor, factor_label) in enumerate([('V/ul', 'Sample volume'), ('PE/EA', 'PE/EA eluent')]):
        for col, target in enumerate(('V1', 'V2')):
            ax = axes[row, col]
            for metric, label, offset, color, marker in [
                ('observed_abs_mean', 'Observed', -.10, BLUE, 'o'),
                ('predicted_abs_mean', 'Predicted q50', .10, ORANGE, 's')]:
                means = []
                for x, budget in enumerate((525, 1005)):
                    data = response[(response.axis == factor) & (response.target == target) & (response.budget == budget)].sort_values('seed')
                    assert len(data) == 5
                    values = data[metric].to_numpy(float)
                    assert np.isfinite(values).all() and np.all(values > 0)
                    mean = values.mean()
                    means.append(mean)
                    xx = x + offset
                    ax.plot([xx, xx], [values.min(), values.max()], color=color, alpha=.65, lw=1.4)
                    ax.scatter(xx + np.linspace(-.027, .027, 5), values, color=color, alpha=.47, s=23, marker=marker, zorder=3)
                    ax.scatter([xx], [mean], color=color, marker=marker, s=85, edgecolor='white', lw=.8,
                               zorder=5, label=label if x == 0 else None)
                    ax.annotate(f'{mean:.3g}', (xx, mean), xytext=(-7 if offset < 0 else 7, 8),
                                textcoords='offset points', ha='right' if offset < 0 else 'left', color=color, fontsize=9)
            ax.set_xticks([0, 1], ['L525', 'L1005'])
            ax.set_xlim(-.45, 1.45)
            ax.set_title(f'{factor_label} · {target}', pad=9)
            if row == 0:
                ax.set_yscale('log')
                ax.set_ylim(.0004, 8)
                ax.set_ylabel('Absolute response (mL, log scale)')
            else:
                ax.set_ylim(bottom=0)
                ax.set_ylabel('Absolute response (mL)')
            ax.grid(axis='y', alpha=.17)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(.5, .885), ncol=2)
    fig.suptitle('Original frozen models: observed and predicted condition responses', fontsize=14, y=.965)
    fig.text(.5, .91, 'Five seed means weighted equally · known training histories (in-sample) · no refitting for this figure',
             ha='center', fontsize=9.5)
    fig.text(.085, .065, 'Large markers: equal-seed means; small markers: individual seed means; whiskers: seed min–max (not confidence intervals).\n'
             'Pair events at L525 / L1005: sample volume 464 / 1393; PE/EA 1277 / 4054, each scored for V1 and V2.\n'
             'Multiple pairs from a series are correlated. Volume panels use a log scale; these are not held-out control-fit scores.',
             fontsize=8.5, va='bottom')
    save(fig, 'a2_frozen_condition_amplitudes')

    # The manually accepted historical batch remains in the full-540 denominator.
    failure = table('prefix_by_first_failure.csv')
    batches = table('retry_case_batches.csv', ['method', 'seed', 'round', 'prefix_final',
                                             'first_failure_category', 'accepted_receipt_exact_matches'])
    assert len(batches) == int(failure.batches.sum()) == 540
    assert int(failure.prefix_batches.sum()) == 127
    assert int((batches.accepted_receipt_exact_matches == 1).sum()) == 539
    recomputed = batches.groupby(['method', 'first_failure_category']).agg(
        batches=('prefix_final', 'size'), prefix_batches=('prefix_final', 'sum')).sort_index()
    reference = failure.set_index(['method', 'first_failure_category'])[['batches', 'prefix_batches']].sort_index()
    assert recomputed.index.equals(reference.index)
    assert np.array_equal(recomputed.to_numpy(), reference.to_numpy())
    manual = batches[(batches.method == METHODS[0]) & (batches.seed == 157) & (batches['round'] == 5)]
    assert len(manual) == 1 and not bool(manual.prefix_final.iloc[0]) and int(manual.accepted_receipt_exact_matches.iloc[0]) == 0
    categories = ['none', 'candidate_identity', 'packet_binding', 'evidence_membership', 'field_whitelist', 'selection_count_or_structure']
    category_labels = ['No recorded first failure', 'Candidate ID legality', 'Packet binding',
                       'Evidence membership', 'Field whitelist', 'Count / structure']
    aggregate = failure.groupby('first_failure_category')[['batches', 'prefix_batches']].sum().reindex(categories)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.0), gridspec_kw={'width_ratios': [1.05, 1]})
    fig.subplots_adjust(left=.17, right=.96, top=.81, bottom=.30, wspace=.50)
    y = np.arange(len(categories))
    prefix = aggregate.prefix_batches.to_numpy(int)
    other = aggregate.batches.to_numpy(int) - prefix
    axes[0].barh(y, prefix, color=ORANGE, label='Final exact ordered prefix')
    axes[0].barh(y, other, left=prefix, color=BLUE, alpha=.72, label='Final non-prefix')
    for i, (n, p) in enumerate(zip(aggregate.batches, prefix)):
        axes[0].text(n + 4, i, f'{p}/{int(n)} prefix', va='center', fontsize=8.5)
    axes[0].set_yticks(y, category_labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 345)
    axes[0].set_xlabel('Accepted batches')
    axes[0].set_title('Counts by first failure category', fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, loc='upper center', bbox_to_anchor=(.5, .91), ncol=2)
    rates = np.full((len(categories), 3), np.nan)
    counts = {}
    for r in failure.itertuples():
        i, j = categories.index(r.first_failure_category), METHODS.index(r.method)
        rates[i, j] = r.prefix_batches / r.batches
        counts[i, j] = (int(r.prefix_batches), int(r.batches))
    cmap = plt.get_cmap('Oranges').copy()
    cmap.set_bad('#E6E8EB')
    im = axes[1].imshow(np.ma.masked_invalid(rates), vmin=0, vmax=1, cmap=cmap, aspect='auto')
    for i in range(len(categories)):
        for j in range(3):
            text = f'{counts[i, j][0]}/{counts[i, j][1]}' if (i, j) in counts else 'No cases'
            axes[1].text(j, i, text, ha='center', va='center', fontsize=9)
    axes[1].set_xticks(range(3), ['Historical\nCW-Free', 'New\nCW-Free', 'NoCW-Free'])
    axes[1].set_yticks(y, [])
    axes[1].set_title('Prefix fraction; cells show prefix / batches', fontsize=11)
    colorbar = fig.colorbar(im, ax=axes[1], fraction=.045, pad=.03)
    colorbar.set_label('Fraction', fontsize=9)
    fig.suptitle('First failure and final display-prefix selection', fontsize=14, y=.96)
    fig.text(.17, .065, 'N=540 accepted Free32 trajectory batches; 127 end as the exact ordered first 32 displayed candidate rows.\n'
             'Historical seed157 round5 is a manually accepted non-prefix exception: it remains in the candidate-ID denominator (61);\n'
             '539 batches have a matching saved accepted-response receipt. A recorded association is not a causal error effect.',
             fontsize=8.5, va='bottom')
    save(fig, 'c2_first_failure_and_prefix')

    repair = table('local_repair_summary.csv').set_index('method').loc[METHODS]
    cases = table('local_repair_cases.csv', ['method', 'original_proposal_n', 'original_unique_legal_n',
                                          'preserved_unique_legal_n', 'historical_final_retained_original_legal_n',
                                          'vacant_slots', 'science_review_needed', 'local_draft_passes_historical_gate'])
    assert len(cases) == int(repair.cases.sum()) == 127
    assert int(repair.original_legal.sum()) == int(repair.retained.sum()) == 3634
    assert int(repair.vacancies.sum()) == int(cases.vacant_slots.sum()) == 430
    assert int(repair.historical_retained.sum()) == 440
    assert (cases.original_proposal_n == 32).all() and int(cases.local_draft_passes_historical_gate.sum()) == 0
    assert int(cases.science_review_needed.sum()) == 27
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9), gridspec_kw={'width_ratios': [1.15, 1]})
    fig.subplots_adjust(left=.075, right=.98, top=.79, bottom=.28, wspace=.30)
    x = np.arange(3)
    history_fraction = repair.historical_retained / repair.original_legal * 100
    local_fraction = repair.retained / repair.original_legal * 100
    axes[0].bar(x - .18, history_fraction, width=.32, color=ORANGE, label='Historical final selection')
    axes[0].bar(x + .18, local_fraction, width=.32, color=BLUE, label='Local repair draft')
    for j, row in enumerate(repair.itertuples()):
        axes[0].text(j - .18, history_fraction.iloc[j] + 3, f'{row.historical_retained}/{row.original_legal}',
                     ha='center', fontsize=8)
        axes[0].text(j + .18, 103, f'{row.retained}/{row.original_legal}', ha='center', fontsize=8)
    axes[0].set_ylim(0, 114)
    axes[0].set_xticks(x, [f'{label}\n(n={int(repair.loc[method, "cases"])} cases)' for method, label in zip(METHODS, LABELS)], fontsize=8.5)
    axes[0].set_ylabel('Original legal ID events retained (%)')
    axes[0].set_title('Preservation of the original legal choices', fontsize=11)
    axes[0].legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(.52, 1.27))
    counts = cases.vacant_slots.value_counts().sort_index()
    axes[1].bar(counts.index, counts.values, color=[ORANGE if v == 0 else GRAY for v in counts.index], edgecolor='white')
    for slots, n in counts.items():
        axes[1].text(slots, n + .7, f'{n}*' if slots == 0 else str(n), ha='center', fontsize=9)
    axes[1].set_xticks(counts.index)
    axes[1].set_ylim(0, counts.max() * 1.22)
    axes[1].set_xlabel('ID slots still requiring a replacement')
    axes[1].set_ylabel('Prefix cases')
    axes[1].set_title('Gaps left explicitly open by local repair', fontsize=11)
    fig.suptitle('Offline local repair: ID preservation is not scientific acceptance', fontsize=14, y=.96)
    fig.text(.075, .065, 'Subset: 127 prefix cases, each with a parseable 32-choice original plan (4064 choice events).\n'
             'Local drafts retain 3634/3634 original legal ID events; historical finals retain 440/3634. Drafts leave 430 missing slots.\n'
             '*Both zero-gap cases still need scientific evidence review. Overall, 27/127 need evidence review and 0/127 drafts pass the full gate.\n'
             'No API calls, automatic filling, historical writes, or downstream training are represented here.',
             fontsize=8.3, va='bottom')
    save(fig, 'c4_local_repair_preservation_and_gaps')

    manifest = {'status': 'PASS', 'inputs': list(inputs.values()), 'outputs': outputs,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'new_fits': 0, 'new_model_inference': 0, 'new_llm_calls': 0,
        'fit_result_files_read': [], 'manual_case_policy': 'Keep historical seed157 round5 in full-540 category denominator, explicitly annotate; exclude from prefix-only 127 cohort',
        'a2_aggregation': 'Mean within each seed, then equal weight to five seeds. Whiskers are min/max, not confidence intervals.'}
    (args.output_dir / 'figure_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'figures': 3, 'png_pdf_files': len(outputs), 'input_tables': list(inputs)}))


if __name__ == '__main__':
    main()
