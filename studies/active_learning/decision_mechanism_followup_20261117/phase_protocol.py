#!/usr/bin/env python3
"""Audit saved validation phases and screening protocols without models or truth."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = ROOT / 'studies/active_learning/llm_decision_behavior_audit_20261116'
SCREEN = ROOT / 'studies/active_learning/qgeognn_v2_row_llm16_screen'
REVISIONS = ROOT / 'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions'
PAIR = ROOT / 'studies/active_learning/nocw_vs_cw_lcmd_five_to1485_20261112'
SEEDS = (157, 6101)
METHODS = ('free16_lcmd16', 'cw_gradient_lcmd32')
PHASES = ((333, 525), (525, 813), (813, 1005), (1005, 1453))


def integrate(points, start, end):
    """Integrate only exact adjacent observed points; never bridge missing budgets."""
    xs = list(range(start, end + 1, 32))
    if end <= start or xs[-1] != end or any(x not in points for x in xs):
        raise ValueError('Missing exact 32-label grid')
    if any(not math.isfinite(points[x]) or points[x] < 0 for x in xs):
        raise ValueError('Invalid loss')
    return sum(16. * (points[a] + points[b]) for a, b in zip(xs, xs[1:]))


def segment(points, start, end):
    """Return normalized raw-curve AULC, with lower values better."""
    return integrate(points, start, end) / (end - start)


def omission_sensitivity(differences, start=333, end=1453):
    """Omit actual adjacent trapezoids; do not interpolate across the removed gap."""
    xs = list(range(start, end + 1, 32))
    total = sum(16. * (differences[a] + differences[b]) for a, b in zip(xs, xs[1:]))
    rows = []
    for i in range(1, len(xs) - 1):
        removed = 16. * (differences[xs[i - 1]] + 2 * differences[xs[i]] + differences[xs[i + 1]])
        rows.append({'center_budget': xs[i], 'removed_from': xs[i - 1], 'removed_to': xs[i + 1],
                     'removed_width': 64, 'removed_delta_area': removed,
                     'remaining_observed_interval_delta_AULC': (total - removed) / (end - start - 64)})
    return rows


def planned_cost(target, nseeds=3, batch=32, initial=333):
    """Count effective accepted batches and scratch fits, excluding reused initial fits."""
    if target < initial or (target - initial) % batch:
        raise ValueError('Target is not an exact full-batch horizon')
    batches = (target - initial) // batch
    return {'target_L': target, 'new_seeds': nseeds, 'batches_per_seed': batches,
            'effective_llm_decisions': nseeds * batches, 'new_qgeognn_fits': nseeds * batches,
            'newly_acquired_row_events': nseeds * batches * batch,
            'llm_chosen_row_events': nseeds * batches * (batch // 2),
            'numerical_chosen_row_events': nseeds * batches * (batch // 2),
            'reusable_initial_fits': nseeds, 'evaluation_points_including_initial': nseeds * (batches + 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curves', type=Path, default=PRIOR / 'row_curves.csv')
    parser.add_argument('--output', type=Path, default=HERE)
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {}

    def read(path):
        path = path.resolve()
        raw = path.read_bytes()
        assert raw == path.read_bytes(), ('Changing input', path)
        sources[str(path.relative_to(ROOT))] = {'path': str(path.relative_to(ROOT)),
            'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        return raw

    def js(path):
        return json.loads(read(path))

    def table(path):
        return list(csv.DictReader(io.StringIO(read(path).decode())))

    def save(name, rows):
        assert rows
        with (args.output / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    raw_curves = table(args.curves)
    curves = {(m, s): {int(r['budget']): float(r['nrmse']) for r in raw_curves
                      if r['method'] == m and int(r['seed']) == s and int(r['budget']) <= 1453}
              for m in METHODS for s in SEEDS}
    for values in curves.values():
        assert sorted(values) == list(range(333, 1454, 32))
    fit_rows = table(ROOT / 'studies/active_learning/comprehensive_history_diagnostics_20261112/tables/round_metrics.csv')
    fits = {(r['method'], int(r['seed']), int(r['budget'])): r for r in fit_rows}
    training = []
    for (method, seed), values in curves.items():
        for budget, loss in values.items():
            row = fits[method, seed, budget]
            fit = js(ROOT / row['source'])
            assert abs(fit['best_validation_combined_normalized_rmse'] - loss) < 1e-11
            assert fit['train_rows'] == budget and fit['test_labels_used_for_fit_or_checkpoint_selection'] == 0
            training.append({'method': method, 'seed': seed, 'budget': budget,
                             'training_seconds': fit['training_seconds'], 'fit_audit': row['source'],
                             'checkpoint_sha256': fit['checkpoint_sha256'],
                             'initialization_hash': fit['initialization_hash'],
                             'validation_ids_hash': fit['validation_ids_hash']})
    save('phase_fit_metadata.csv', training)
    for seed in SEEDS:
        baseline = [r for r in training if r['seed'] == seed and r['budget'] == 333]
        assert len({(r['checkpoint_sha256'], r['initialization_hash'], r['validation_ids_hash']) for r in baseline}) == 1
    phases = []
    summary = []
    intervals = []
    omissions = []
    phase_omissions = []
    weights = []
    for seed in SEEDS:
        h, c = (curves[m, seed] for m in METHODS)
        delta = {x: h[x] - c[x] for x in h}
        total_area = integrate(h, 333, 1453) - integrate(c, 333, 1453)
        for start, end in PHASES:
            xs = list(range(start, end + 1, 32))
            ha, ca = segment(h, start, end), segment(c, start, end)
            dh = [h[b] - h[a] for a, b in zip(xs, xs[1:])]
            dc = [c[b] - c[a] for a, b in zip(xs, xs[1:])]
            phases.append({'seed': seed, 'phase': f'{start}-{end}', 'L_start': start, 'L_end': end,
                'intervals': len(xs) - 1, 'hybrid_AULC': ha, 'cw_AULC': ca,
                'delta_AULC_hybrid_minus_cw': ha - ca, 'relative_AULC_delta': (ha - ca) / ca,
                'hybrid_start': h[start], 'cw_start': c[start], 'start_delta': delta[start],
                'hybrid_endpoint': h[end], 'cw_endpoint': c[end], 'endpoint_delta': delta[end],
                'contribution_to_full_delta_AULC': (ha - ca) * (end - start) / 1120.,
                'hybrid_better_evaluation_points': sum(h[x] < c[x] for x in xs),
                'evaluation_points': len(xs),
                'hybrid_total_variation': sum(map(abs, dh)), 'cw_total_variation': sum(map(abs, dc)),
                'hybrid_max_abs_step': max(map(abs, dh)), 'cw_max_abs_step': max(map(abs, dc)),
                'hybrid_upward_steps': sum(v > 0 for v in dh), 'cw_upward_steps': sum(v > 0 for v in dc)})
            removed = (ha - ca) * (end - start)
            phase_omissions.append({'seed': seed, 'omitted_phase': f'{start}-{end}',
                'remaining_width': 1120 - (end - start),
                'remaining_observed_interval_delta_AULC': (total_area - removed) / (1120 - end + start)})
        for start in range(333, 1453, 32):
            end = start + 32
            intervals.append({'seed': seed, 'L_start': start, 'L_end': end,
                'hybrid_gain': h[start] - h[end], 'cw_gain': c[start] - c[end],
                'delta_area': 16. * (delta[start] + delta[end]),
                'contribution_to_full_delta_AULC': 16. * (delta[start] + delta[end]) / 1120.,
                'hybrid_start': h[start], 'hybrid_end': h[end], 'cw_start': c[start], 'cw_end': c[end]})
        for i, budget in enumerate(sorted(h)):
            w = 16. if i in (0, len(h) - 1) else 32.
            weights.append({'seed': seed, 'budget': budget, 'delta_loss': delta[budget],
                'trapezoid_weight': w, 'contribution_to_full_delta_AULC': w * delta[budget] / 1120.})
        for row in omission_sensitivity(delta):
            omissions.append({'seed': seed, **row})
        summary.append({'seed': seed, 'L_start': 333, 'L_end': 1453,
            'hybrid_AULC': segment(h, 333, 1453), 'cw_AULC': segment(c, 333, 1453),
            'delta_AULC': total_area / 1120., 'endpoint_delta': delta[1453],
            'hybrid_better_points_excluding_initial': sum(h[x] < c[x] for x in h if x > 333),
            'postinitial_points': 35,
            'omission_delta_min': min(r['remaining_observed_interval_delta_AULC'] for r in omissions if r['seed'] == seed),
            'omission_delta_max': max(r['remaining_observed_interval_delta_AULC'] for r in omissions if r['seed'] == seed)})
    save('phase_performance.csv', phases)
    save('phase_overall.csv', summary)
    save('phase_interval_influence.csv', intervals)
    save('phase_point_influence.csv', weights)
    save('phase_local_omission_sensitivity.csv', omissions)
    save('phase_omission_sensitivity.csv', phase_omissions)

    # Recover early-screen validation summaries without reading frozen test predictions.
    early = []
    for seed in SEEDS:
        for method in ('cw16_llm16', 'cw16_random16'):
            runtime = SCREEN / f'runtime/seed_{seed}/{method}'
            for rnd in range(4):
                freeze = js(runtime / f'round_{rnd:02d}/prediction_freeze.json')
                audit_path = Path(freeze['checkpoint_path']).with_name('fit_audit.json')
                audit = js(audit_path)
                assert audit['checkpoint_sha256'] == freeze['checkpoint_sha256']
                assert audit['test_labels_used_for_fit_or_checkpoint_selection'] == 0
                assert audit['train_rows'] == 333 + 32 * rnd
                base = js(ROOT / fits['cw_gradient_lcmd32', seed, 333]['source'])
                assert audit['initialization_hash'] == base['initialization_hash']
                assert audit['validation_ids_hash'] == base['validation_ids_hash']
                if rnd == 0:
                    assert audit['checkpoint_sha256'] == base['checkpoint_sha256']
                early.append({'cohort': 'early_shortlist_screen', 'method': method, 'seed': seed,
                    'budget': 333 + 32 * rnd, 'validation_nrmse': audit['best_validation_combined_normalized_rmse'],
                    'fit_audit': str(audit_path.relative_to(ROOT)), 'checkpoint_sha256': audit['checkpoint_sha256']})
    save('early_screen_validation_curves.csv', early)
    comparison = []
    for seed in SEEDS:
        for method in ('cw16_llm16', 'cw16_random16'):
            p = {r['budget']: r['validation_nrmse'] for r in early if r['method'] == method and r['seed'] == seed}
            c = curves['cw_gradient_lcmd32', seed]
            comparison.append({'cohort': 'early_shortlist_screen_validation_recovered', 'method': method, 'seed': seed,
                'L_start': 333, 'L_end': 429, 'AULC': segment(p, 333, 429), 'endpoint': p[429],
                'cw_AULC': segment(c, 333, 429), 'cw_endpoint': c[429],
                'delta_AULC': segment(p, 333, 429) - segment(c, 333, 429), 'endpoint_delta': p[429] - c[429]})
        h, c = (curves[m, seed] for m in METHODS)
        comparison.append({'cohort': 'long_fullpool_prefix_validation', 'method': 'free16_lcmd16', 'seed': seed,
            'L_start': 333, 'L_end': 429, 'AULC': segment(h, 333, 429), 'endpoint': h[429],
            'cw_AULC': segment(c, 333, 429), 'cw_endpoint': c[429],
            'delta_AULC': segment(h, 333, 429) - segment(c, 333, 429), 'endpoint_delta': h[429] - c[429]})
    save('early_long_validation_comparison.csv', comparison)

    packet_checks = []
    for seed in SEEDS:
        early_packet = js(SCREEN / f'selections/seed_{seed}/cw16_llm16/round_00/packet.json')
        longdir = REVISIONS / f'sol_xhigh_hybrid_free_to1005_20261013/selections/seed_{seed}/cw16_llm16_scientist_v2/round_00'
        long_packet = js(longdir / 'input.json')
        long_freeze = js(longdir / 'batch_freeze.json')
        long_contract = js(longdir / 'contract.json')
        basefit = js(ROOT / fits['cw_gradient_lcmd32', seed, 333]['source'])
        assert long_contract['checkpoint_sha256'] == basefit['checkpoint_sha256']
        assert len(long_contract['L_t_ids']) == 333 and len(long_contract['U_t_ids']) == 2997
        pending = [r['id'] for r in early_packet['pending_CW16']]
        assert pending == long_freeze['pending_ids']
        assert len(early_packet['candidates']) == 128
        assert len(long_packet['candidates']['rows']) + len(long_packet['pending']['rows']) == 2997
        packet_checks.append({'seed': seed, 'initial_L333_checkpoint_match': True,
            'same_ordered_numeric_first16': True, 'early_eligible_candidate_count': len(early_packet['candidates']),
            'long_eligible_candidate_count': len(long_packet['candidates']['rows']),
            'long_pending_count': len(long_packet['pending']['rows']),
            'early_observed_outcomes_supplied': early_packet['observed_outcome_values_in_packet'],
            'long_observed_record_count': len(long_packet['observed']['rows']),
            'long_delivery': long_packet['delivery'],
            'early_quantiles_available': False, 'long_quantiles_available': True})
    save('hybrid_protocol_packet_checks.csv', packet_checks)

    # Audit only paths and metadata for reusable initial states and unavailable hybrid prefixes.
    reuse = []
    for seed in (887, 2357, 12203):
        directory = PAIR / f'cw_lcmd/runtime/seed_{seed}/center_width_lcmd/round_00'
        freeze = js(directory / 'round_freeze.json')
        fit = js(Path(freeze['checkpoint_path']).with_name('fit_audit.json'))
        bank = directory / 'acquisition_artifacts/current_center_width_gradient_features.npz'
        sidecar = js(bank.with_suffix('.npz.contract.json'))
        prior = list(REVISIONS.glob(f'*/runtime/seed_{seed}/cw16_llm16_scientist_v2/round_*/prediction_freeze.json'))
        assert not prior
        reuse.append({'seed': seed, 'initial_L': 333, 'initial_checkpoint_path': freeze['checkpoint_path'],
            'initial_checkpoint_exists': Path(freeze['checkpoint_path']).is_file(),
            'initial_checkpoint_sha256': fit['checkpoint_sha256'], 'initial_fit_rows': fit['train_rows'],
            'initialization_hash': fit['initialization_hash'], 'validation_ids_hash': fit['validation_ids_hash'],
            'cw_bank_path': str(bank.relative_to(ROOT)), 'cw_bank_exists': bank.is_file(),
            'cw_bank_sha256': sidecar['sha256'], 'new_hash_of_bank_computed': False,
            'existing_long_hybrid_saved_fits_in_revisions': len(prior),
            'reuse_boundary': 'L333 state only; later pure-strategy fits are not interchangeable with new hybrid histories'})
    save('hybrid_new_seed_reusable_states.csv', reuse)
    costs = [planned_cost(target) for target in (525, 813, 1005, 1453, 1485)]
    empirical = [r['training_seconds'] for r in training if r['method'] == 'free16_lcmd16' and r['budget'] > 333]
    for row in costs:
        row['historical_hybrid_fit_seconds_median'] = statistics.median(empirical)
        row['serial_fit_hours_at_historical_median'] = row['new_qgeognn_fits'] * statistics.median(empirical) / 3600.
        row['pricing_status'] = 'provider price, correction attempts and future walltime unknown; no monetary quote'
    save('hybrid_expansion_costs.csv', costs)
    for p in [SCREEN / 'FINAL_REPORT.md', SCREEN / 'README.md', SCREEN / 'protocol.json', SCREEN / 'prompt.txt',
              REVISIONS / 'sol_xhigh_hybrid_free_to1005_20261013/protocol.json',
              REVISIONS / 'sol_xhigh_hybrid_free_to1005_20261013/extension_manifest.json',
              ROOT / 'src/qgeognn_al/active_learning_v2/llm_screen.py',
              ROOT / 'src/qgeognn_al/active_learning_v2/scientist_full_delivery.py',
              ROOT / 'src/qgeognn_al/active_learning_v2/scientist_selector.py',
              ROOT / 'src/qgeognn_al/active_learning_v2/scientist_study.py']:
        read(p)
    read(Path(__file__))
    (args.output / 'phase_source_manifest.json').write_text(json.dumps({'inputs': list(sources.values()),
        'row_evaluation': 'saved validation summaries only', 'test_predictions_opened': False,
        'test_truth_opened': False, 'new_training': 0, 'new_llm_calls': 0,
        'stages_prespecified_for_this_task': [list(v) for v in PHASES],
        'short_screen_report_role': 'existing published gate/test aggregates only; no new test evaluation',
        'curves_rows_used_long': 144, 'curves_rows_recovered_early': 16}, ensure_ascii=False, indent=2) + '\n')
    if args.plots:
        import os
        os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/phase_protocol_mpl')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        for ax, feature, title in zip(axes, ('delta_AULC_hybrid_minus_cw', 'endpoint_delta'), ('Stage AULC difference', 'Stage endpoint difference')):
            for seed, marker in zip(SEEDS, ('o', 's')):
                part = [r for r in phases if r['seed'] == seed]
                ax.plot(range(4), [r[feature] for r in part], marker=marker, label=f'seed {seed}')
            ax.axhline(0, color='.4', lw=.8)
            ax.set_xticks(range(4), ['333–525', '525–813', '813–1005', '1005–1453'], rotation=20)
            ax.set(title=title, ylabel='Hybrid − CW-LCMD (lower is better)', xlabel='Observed label window')
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', alpha=.2)
        axes[0].legend(frameon=False)
        figures = args.output / 'figures'
        figures.mkdir(exist_ok=True)
        fig.savefig(figures / 'hybrid_phase_differences.png', dpi=180)
        fig.savefig(figures / 'hybrid_phase_differences.pdf')
        plt.close(fig)
    print(json.dumps({'phase_rows': len(phases), 'summary': summary, 'cost_rows': len(costs)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
