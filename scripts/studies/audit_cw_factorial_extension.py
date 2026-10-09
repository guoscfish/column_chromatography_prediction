#!/usr/bin/env python
"""Read-only scientific verification plus compact extension delivery records."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from src.qgeognn_al.active_learning_v2 import cw_factorial_extension as ex
from src.qgeognn_al.active_learning_v2.protocol import ids_hash, stable_hash
from src.qgeognn_al.artifacts import sha256_file as sha
from src.qgeognn_al.training.predictor import atomic_json


def main():
    p = ex.verify()
    s = ex.STUDY
    read = ex.base.read
    amended = (ex.base.PARENT / 'qgeognn_v2_row_cw_selector_extension_to_1005/WHITENING_STOP_AMENDMENT.json').exists()
    arms = ('cw_maxdet', 'cw_ivr') if amended else tuple(ex.ARMS)
    expected_fits = len(arms)*8
    ledger = read(s / 'fit_ledger.json')
    assert len(ledger) == expected_fits and all(v['status'] == 'completed' for v in ledger.values())
    checks, costs = [], []
    original = read(ex.base.STUDY / 'protocol.json')
    for seed in ex.SEEDS:
        for arm in arms:
            root = s / f'runtime/seed_{seed}/{arm}'
            anchor = ex.source_dir(seed, arm, 6)
            prior = read(anchor / 'state.json')
            previous_checkpoint = anchor / 'model/best.pt'
            selected = list(prior['selected_all'])
            for r in ex.NEW_ROUNDS:
                before = root / f'round_{r-1:02d}'
                d = root / f'round_{r:02d}'
                f = read(before / 'batch_freeze.json')
                a = read(d / 'model/fit_audit.json')
                state = read(d / 'state.json')
                item = ledger[f'{seed}/{arm}/{r}']
                assert sha(d / 'model/fit_audit.json') == item['audit_sha256']
                assert sha(d / 'model/best.pt') == a['checkpoint_sha256'] == state['checkpoint_sha256']
                assert sha(d / 'model/predictions.csv.gz') == a['prediction_sha256'] == state['prediction_sha256']
                assert a['initialization_hash'] == original['contexts'][str(seed)]['initialization_hash']
                assert a['test_labels_used_for_fit_or_checkpoint_selection'] == 0
                assert a['test_graph_label_state'] == 'zero_sentinel'
                assert f['contract']['protocol_hash'] == stable_hash(p)
                assert f['contract']['checkpoint_sha256'] == sha(previous_checkpoint)
                ids, l, u = f['ids'], f['contract']['ordered_l'], f['contract']['ordered_u']
                positions = f['positions']
                assert len(ids) == len(set(ids)) == 32 and set(ids) <= set(u) and not set(ids) & set(l)
                assert [u[i] for i in positions] == ids
                assert len(l) == len(prior['labeled']) == 333 + 32*(r-1)
                assert len(set(l)) == len(l) and not set(l) & set(u)
                if r > 7:
                    assert l == previous_l and u == previous_u
                assert sha(before / 'premeasurement.csv.gz') == f['premeasurement_sha256']
                assert pd.read_csv(before / 'premeasurement.csv.gz').sample_id.astype(str).tolist() == ids
                selected += ids
                assert state['selected_all'] == selected and len(set(selected)) == len(selected)
                assert state['labeled'] == prior['labeled'] + [prior['unlabeled'][i] for i in positions]
                assert state['unlabeled'] == np.delete(prior['unlabeled'], positions).tolist()
                previous_l = l + ids
                previous_u = [v for v in u if v not in set(ids)]
                assert a['train_ids_hash'] == ids_hash(previous_l)
                assert a['train_rows'] == len(state['labeled']) == 333 + 32*r
                checks.append({'seed': seed, 'arm': arm, 'round': r,
                    'checkpoint_sha256': a['checkpoint_sha256'], 'prediction_sha256': a['prediction_sha256'],
                    'audit_sha256': item['audit_sha256'], 'batch_freeze_sha256': sha(before / 'batch_freeze.json'),
                    'state_sha256': sha(d / 'state.json'), 'verified': True})
                costs.append({'seed': seed, 'arm': arm, 'round': r, 'budget': a['train_rows'],
                    'training_seconds': a['training_seconds'], 'epochs_run': a['epochs_run'],
                    'best_epoch': a['best_epoch'], 'acquisition_seconds': read(before / 'selection_audit.json')['seconds']})
                prior = state
                previous_checkpoint = d / 'model/best.pt'
            trajectory = read(root / 'trajectory_freeze.json')
            assert trajectory['all_selected_ids'] == selected and trajectory['budget'] == 653
            assert trajectory['test_truth_access_count'] == 0
            assert sha(root / 'trajectory_freeze.json') == read(s / 'GLOBAL_FREEZE.json')['trajectories'][f'{seed}/{arm}']
    for entry in read(s / 'REUSE_MANIFEST.json')['entries']:
        checkpoint = ex.base.ROOT / entry['checkpoint']
        assert sha(checkpoint) == entry['checkpoint_sha256']
        assert sha(checkpoint.with_name('fit_audit.json')) == entry['audit_sha256']
        assert sha(checkpoint.with_name('predictions.csv.gz')) == entry['prediction_sha256']
    assert read(s / 'RUN_STATE.json')['completed_fits'] == expected_fits
    curves = pd.read_csv(s / 'validation_curves.csv')
    summary = pd.read_csv(s / 'validation_summary.csv')
    assert len(curves) == 112+(len(arms)+5)*8 and len(summary) == (len(arms)+5)*2
    assert not curves.duplicated(['seed', 'arm', 'budget']).any()
    assert all(g.budget.tolist() == list(range(333, 654, 32)) for (_, arm), g in curves.groupby(['seed', 'arm']) if arm in (*arms, 'raw_lcmd','raw_maxdet','raw_ivr','cw_lcmd','random'))
    pd.testing.assert_frame_equal(curves[curves.budget <= 525].reset_index(drop=True),
                                  pd.read_csv(ex.base.STUDY / 'validation_curves.csv'))
    for row in summary.itertuples():
        g = curves[(curves.seed == row.seed) & (curves.arm == row.arm)]
        late = g[g.budget >= 525]
        assert np.isclose(row.AULC525_653, np.trapezoid(late.combined_normalized_RMSE, late.budget)/128)
        assert np.isclose(row.AULC333_653, np.trapezoid(g.combined_normalized_RMSE, g.budget)/320)
    cost = pd.DataFrame(costs)
    cost.to_csv(s / 'resource_costs.csv', index=False)
    curves[curves.budget == 653].to_csv(s / 'endpoint_metrics.csv', index=False)
    differences = []
    for seed in ex.SEEDS:
        rows = summary[summary.seed == seed].set_index('arm')
        for method, reference in [('cw_lcmd', 'raw_lcmd'), ('cw_maxdet', 'raw_maxdet'),
                                  ('cw_ivr', 'raw_ivr'), ('whitening_lcmd', 'cw_lcmd')]:
            if method not in rows.index:
                continue
            a, b = rows.loc[method], rows.loc[reference]
            differences.append({'seed': seed, 'method': method, 'reference': reference,
                'late_AULC_delta': float(a.AULC525_653 - b.AULC525_653),
                'late_relative_improvement_percent': float(100*(b.AULC525_653-a.AULC525_653)/b.AULC525_653),
                'full_AULC_delta': float(a.AULC333_653-b.AULC333_653),
                'NRMSE653_delta': float(a.NRMSE653-b.NRMSE653)})
    paired = pd.DataFrame(differences)
    paired.to_csv(s / 'paired_seed_differences.csv', index=False)
    decision = {'status': 'completed', 'evidence_class': 'post-development exploration',
        'primary_metric': 'validation AULC525-653', 'new_fits': expected_fits, 'reused_control_points': 40,
        'paid_llm_calls': 0, 'new_test_reveals': 0, 'original_confirmation_gate_changed': False,
        'next_action': 'stop training at 653; no automatic seed confirmation or LLM requests',
        'training_seconds': float(cost.training_seconds.sum()),
        'acquisition_seconds': float(cost.acquisition_seconds.sum()),
        'paired_effects': differences}
    atomic_json(s / 'decision.json', decision)
    audit = {'status': 'completed', 'verified_fits': len(checks), 'test_reveals': 0,
        'protocol_sha256': sha(s / 'protocol.json'), 'checks': checks,
        'report_code_sha256': sha(Path(__file__)),
        'validation_curves_sha256': sha(s / 'validation_curves.csv')}
    atomic_json(s / 'DELIVERY_AUDIT.json', audit)
    lines = ['# Four-round CW extension results', '',
        f'Completed{expected_fits}/{expected_fits} new fits: {len(arms)} arms × two seeds × four batches (525→653). Forty historical control points reused. Whitening stopped at525 by user request.' if amended else 'Completed24/24 new fits;40 control points reused.', '',
        'This extension was authorized after inspecting333–525. Its frozen descriptive primary metric is validation AULC525–653; it is exploratory and cannot reverse the original confirmation gate or establish significance with two seeds.', '',
        '| Seed | Arm | AULC525–653 | AULC333–653 | NRMSE653 |',
        '|---|---|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.seed} | {r.arm} | {r.AULC525_653:.6f} | {r.AULC333_653:.6f} | {r.NRMSE653:.6f} |')
    lines += ['', '## Paired late-window effects', '',
        'Positive relative improvement favors the method; negative AULC/endpoint deltas favor the method. Selector-specific normalization and finite CountSketch effects remain part of these trajectories.', '',
        '| Seed | Method / reference | Late AULC delta | Relative improvement | Endpoint delta |',
        '|---|---|---:|---:|---:|']
    for r in paired.itertuples():
        lines.append(f'| {r.seed} | {r.method} / {r.reference} | {r.late_AULC_delta:+.6f} | {r.late_relative_improvement_percent:+.2f}% | {r.NRMSE653_delta:+.6f} |')
    lines += ['', 'Per-target RMSE/MAE/R² are in endpoint_metrics.csv; all eleven-point curves in validation_curves.csv and validation_curves.png. Raw values and per-seed effects take precedence over pooled rankings.', '',
        f"Training wall time: {decision['training_seconds']:.1f}s; acquisition: {decision['acquisition_seconds']:.1f}s. Exact costs and epochs: resource_costs.csv. Delivery audit verifies all authorized fits, unchanged initialization, own ordered label trajectories,32 unique IDs per batch and premeasurement prediction hashes.", '',
        '## Reproduction and stopping point', '', 'No further training is needed. Verification/report commands:', '',
        '```sh', 'KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_cw_factorial_extension.py report',
        'KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/audit_cw_factorial_extension.py', '```', '',
        'The original no-new-selector-candidate gate remains unchanged. Any future new-seed study needs its own rationale, fixed candidate/control set and untouched-seed audit; P2 live LLM ablations remain unrun with zero paid budget. Historical test exposure must not be described as a new blind test.']
    (s / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in decision.items() if k != 'paired_effects'}))


if __name__ == '__main__':
    main()
