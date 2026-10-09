#!/usr/bin/env python
"""Verify the final1005 matrix and emit validation-only descriptive evidence."""
from pathlib import Path
import sys
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from src.qgeognn_al.active_learning_v2 import cw_factorial_to1005 as study
from src.qgeognn_al.active_learning_v2.protocol import ids_hash, stable_hash
from src.qgeognn_al.artifacts import sha256_file as sha
from src.qgeognn_al.training.predictor import atomic_json


def main():
    p = study.verify()
    s, read = study.STUDY, study.base.read
    anchors = read(s/'ANCHORS.json')
    assert anchors['protocol_hash'] == stable_hash(p)
    assert anchors['source_global_freeze_sha256'] == sha(study.previous.STUDY/'GLOBAL_FREEZE.json')
    initial = read(study.base.STUDY/'protocol.json')['contexts']
    checks, costs = [], []
    for seed in study.SEEDS:
        for arm in study.ARMS:
            root = s/f'runtime/seed_{seed}/{arm}'
            anchor = study.previous.STUDY/f'runtime/seed_{seed}/{arm}/round_10'
            bound = next(x for x in anchors['anchors'] if x['seed'] == seed and x['arm'] == arm)
            assert sha(anchor/'state.json') == bound['state_sha256']
            assert sha(anchor/'model/best.pt') == bound['checkpoint_sha256']
            state = read(anchor/'state.json')
            checkpoint = anchor/'model/best.pt'
            ledger = read(root/'fit_ledger.json')
            assert set(ledger) == {str(r) for r in range(11,22)}
            selected = list(state['selected_all'])
            expected_l = expected_u = None
            for r in range(11,22):
                before = root/f'round_{r-1:02d}'
                d = root/f'round_{r:02d}'
                a, f = read(d/'model/fit_audit.json'), read(before/'batch_freeze.json')
                assert ledger[str(r)]['status'] == 'completed'
                assert sha(d/'model/fit_audit.json') == ledger[str(r)]['audit_sha256']
                assert sha(d/'model/best.pt') == a['checkpoint_sha256']
                assert sha(d/'model/predictions.csv.gz') == a['prediction_sha256']
                assert a['initialization_hash'] == initial[str(seed)]['initialization_hash']
                assert a['test_labels_used_for_fit_or_checkpoint_selection'] == 0
                assert a['test_graph_label_state'] == 'zero_sentinel'
                assert f['contract']['checkpoint_sha256'] == sha(checkpoint)
                assert f['contract']['protocol_hash'] == stable_hash(p)
                ids, l, u = f['ids'], f['contract']['ordered_l'], f['contract']['ordered_u']
                positions = f['positions']
                assert len(ids) == len(set(ids)) == 32 and set(ids) <= set(u) and not set(ids) & set(l)
                assert [u[i] for i in positions] == ids
                assert len(l) == len(state['labeled']) == 333+32*(r-1)
                if expected_l is not None:
                    assert l == expected_l and u == expected_u
                assert sha(before/'premeasurement.csv.gz') == f['premeasurement_sha256']
                assert pd.read_csv(before/'premeasurement.csv.gz').sample_id.astype(str).tolist() == ids
                new = read(d/'state.json')
                selected += ids
                assert new['selected_all'] == selected and len(set(selected)) == len(selected)
                assert new['labeled'] == state['labeled'] + [state['unlabeled'][i] for i in positions]
                assert new['unlabeled'] == np.delete(state['unlabeled'],positions).tolist()
                expected_l, expected_u = l+ids, [v for v in u if v not in set(ids)]
                assert a['train_rows'] == 333+32*r == len(new['labeled'])
                assert a['train_ids_hash'] == ids_hash(expected_l)
                assert new['checkpoint_sha256'] == a['checkpoint_sha256']
                assert new['prediction_sha256'] == a['prediction_sha256']
                checks.append({'seed':seed,'arm':arm,'round':r,'checkpoint_sha256':a['checkpoint_sha256'],
                    'audit_sha256':sha(d/'model/fit_audit.json'),'prediction_sha256':a['prediction_sha256'],
                    'batch_freeze_sha256':sha(before/'batch_freeze.json'),'state_sha256':sha(d/'state.json')})
                costs.append({'seed':seed,'arm':arm,'round':r,'budget':a['train_rows'],
                    'training_seconds':a['training_seconds'],'epochs_run':a['epochs_run'],'best_epoch':a['best_epoch'],
                    'acquisition_seconds':read(before/'selection_audit.json')['seconds']})
                state, checkpoint = new, d/'model/best.pt'
            t = read(root/'trajectory_freeze.json')
            assert t['budget'] == 1005 and t['all_selected_ids'] == selected and t['test_truth_access_count'] == 0
            assert sha(root/'trajectory_freeze.json') == read(s/'GLOBAL_FREEZE.json')['trajectories'][f'{seed}/{arm}']
    for entry in read(s/'REUSE_MANIFEST.json')['entries']:
        checkpoint = study.base.ROOT/entry['checkpoint']
        assert sha(checkpoint) == entry['checkpoint_sha256']
        assert sha(checkpoint.with_name('fit_audit.json')) == entry['audit_sha256']
        assert sha(checkpoint.with_name('predictions.csv.gz')) == entry['prediction_sha256']
    curves, summary = pd.read_csv(s/'validation_curves.csv'), pd.read_csv(s/'validation_summary.csv')
    assert len(curves) == 352 and len(summary) == 16 and len(checks) == 66
    assert not curves.duplicated(['seed','arm','budget']).any()
    assert all(g.budget.tolist() == list(range(333,1006,32)) for _,g in curves.groupby(['seed','arm']))
    pd.testing.assert_frame_equal(curves[curves.budget<=653].reset_index(drop=True),pd.read_csv(study.previous.STUDY/'validation_curves.csv'))
    for row in summary.itertuples():
        g = curves[(curves.seed==row.seed)&(curves.arm==row.arm)]
        for start in (333,525,653):
            w = g[g.budget>=start]
            assert np.isclose(getattr(row,f'AULC{start}_1005'),np.trapezoid(w.combined_normalized_RMSE,w.budget)/(1005-start))
    assert read(s/'RUN_STATE.json')['completed_fits'] == 66
    cost = pd.DataFrame(costs)
    cost.to_csv(s/'resource_costs_653_1005.csv',index=False)
    allcost = pd.concat([pd.read_csv(study.previous.STUDY/'resource_costs.csv'),cost],ignore_index=True)
    assert len(allcost) == 90
    allcost.to_csv(s/'resource_costs_525_1005.csv',index=False)
    differences = []
    for seed in study.SEEDS:
        rows = summary[summary.seed==seed].set_index('arm')
        for method, reference in [('cw_lcmd','raw_lcmd'),('cw_maxdet','raw_maxdet'),('cw_ivr','raw_ivr'),('whitening_lcmd','cw_lcmd')]:
            a,b = rows.loc[method],rows.loc[reference]
            differences.append({'seed':seed,'method':method,'reference':reference,
                'AULC525_1005_delta':float(a.AULC525_1005-b.AULC525_1005),
                'relative_improvement_percent':float(100*(b.AULC525_1005-a.AULC525_1005)/b.AULC525_1005),
                'AULC333_1005_delta':float(a.AULC333_1005-b.AULC333_1005),'NRMSE1005_delta':float(a.NRMSE1005-b.NRMSE1005)})
    paired = pd.DataFrame(differences)
    paired.to_csv(s/'paired_seed_differences.csv',index=False)
    decision = {'status':'completed','evidence_class':'post-development exploration','new_fits653_1005':66,
        'new_fits525_1005':90,'total_new_fits_including_original':126,'paid_llm_calls':0,'new_test_reveals':0,
        'primary_metric':'normalized validation AULC525-1005','original_confirmation_gate_changed':False,
        'next_action':'stop training at1005; no automatic new-seed confirmation or paid LLM calls',
        'training_seconds525_1005':float(allcost.training_seconds.sum()),
        'acquisition_seconds525_1005':float(allcost.acquisition_seconds.sum()),'effects':differences}
    atomic_json(s/'decision.json',decision)
    atomic_json(s/'DELIVERY_AUDIT.json',{'status':'completed','verified_fits':66,'checks':checks,'protocol_sha256':sha(s/'protocol.json'),
        'validation_curves_sha256':sha(s/'validation_curves.csv'),'audit_code_sha256':sha(Path(__file__))})
    lines = ['# CW factorial: continuation to1005', '',
        'Completed all16 curves through1005: eight arms × two development seeds,22 budget points each. Three new arms required90 extra fits after525 (24 through653,66 thereafter);150 additional historical control points were audited and reused. Including the original36 fits,126 fits were added across this work. No paid LLM calls or new test reveals.', '',
        'The user extended the budget after seeing the333–525 development results. The frozen extension metric is validation AULC525–1005. This is post-development exploration; the original AULC333–525 and failed new-selector confirmation gate remain unchanged. Two seeds do not establish statistical significance.', '',
        '| Seed | Arm | AULC525–1005 | AULC333–1005 | NRMSE1005 |', '|---|---|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.seed} | {r.arm} | {r.AULC525_1005:.6f} | {r.AULC333_1005:.6f} | {r.NRMSE1005:.6f} |')
    lines += ['', '## Paired effects', '', 'Positive relative improvement favors the method; negative deltas favor the method.', '',
        '| Seed | Method / reference | AULC525–1005 delta | Relative improvement | NRMSE1005 delta |','|---|---|---:|---:|---:|']
    for r in paired.itertuples():
        lines.append(f'| {r.seed} | {r.method} / {r.reference} | {r.AULC525_1005_delta:+.6f} | {r.relative_improvement_percent:+.2f}% | {r.NRMSE1005_delta:+.6f} |')
    lines += ['', 'V1/V2 RMSE, MAE and R²: endpoint_metrics.csv. Each seed curve: validation_curves.csv/PNG. Supplemental AULC653–1005: validation_summary.csv. Selector normalization and finite sketch behavior remain part of the comparison; cross-selector differences are not exclusively objective-function effects.', '',
        f"Additional525–1005 training wall-seconds summed over fits: {decision['training_seconds525_1005']:.1f}; acquisition: {decision['acquisition_seconds525_1005']:.1f}. Two independent fits may run concurrently after653; summed wall time is not elapsed makespan. See resource_costs_525_1005.csv.", '',
        '## Stop and recover', '', 'All training stops at1005. Reporting and verification only:', '', '```sh',
        'KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/run_cw_factorial_to1005.py report',
        'KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n fish python scripts/studies/audit_cw_factorial_to1005.py','```','',
        'P2 paid independent ablations remain unrun. No new-seed confirmation is automatically authorized by a favorable extension endpoint. Any next study needs a fixed candidate/control set and independently audited unused seeds. Historical exposed test data is not a new blind test.']
    (s/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in decision.items() if k!='effects'}))


if __name__=='__main__':
    main()
