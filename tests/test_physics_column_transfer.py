"""Leakage, physical units, model identity and blind-evaluation contracts."""
import copy
import json
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
import numpy as np
import pandas as pd
import pytest
import torch
from src.qgeognn_al.transfer.column_physics import packing_mass,context_matrix,TrainingNormalizer
from src.qgeognn_al.transfer.column_conditioned import ColumnConditionedQGeoGNN
from scripts.studies import run_physics_column_transfer as run


def frame():
    return pd.DataFrame({'sample_id':['source','target'],'column_specs':['Silica-CS 4g','Silica-CS 4g+4g'],'Flow mL/min':[5.,10.],'Density g/ml':[1.,1.2],'V/ul':[100.,100.],'Volume of loading solvent/ul':[200.,400.]})


def test_units_and_specs():
    assert [packing_mass(s) for s in ['Silica-CS 4g','Silica-CS 4g+4g','Silica-CS 25g','Silica-CS 40g']]==[4,8,25,40]
    np.testing.assert_allclose(context_matrix(frame()),[[4,5,25,50],[8,10,15,50]])
    for bad in [None,'unknown','Silica-CS 0g','Silica-CS 25g diameter 2']:
        with pytest.raises(ValueError): packing_mass(bad)
    with pytest.raises(ValueError): context_matrix(frame().drop(columns='Flow mL/min'))


def test_training_only_normalization():
    f=frame();n=TrainingNormalizer().fit(f,['source','target'])
    original=n.mean.copy();test=f.copy();test['Flow mL/min']=100000
    n.transform(test);np.testing.assert_array_equal(original,n.mean)
    with pytest.raises(ValueError): TrainingNormalizer().fit(pd.concat([f,test.assign(sample_id=['test1','test2'])]),['source','target'])
    with pytest.raises(ValueError): TrainingNormalizer().fit(f,['source'])


def test_ledgers_and_balanced_sampling():
    assert len(run.expected_contexts())==120
    for c,p,s,b in run.expected_contexts():
        usage=run.old.ledger(c,p,s,b)
        assert usage['other_target_column_labels_used']==0
        assert not set(usage['gradient_train'])&set(usage['test'])
        si,ti=run.balanced_indices(s,1,3330,len(usage['gradient_train']))
        assert len(si)==len(ti)==len(set(si))==len(set(ti))


def test_single_head_and_zero_known_context():
    torch.set_num_threads(1)
    source=run.load_predictor_checkpoint(run.old.SOURCE)
    original={k:v.clone() for k,v in source.state_dict().items()}
    usage=run.old.ledger('8g','row',run.SEEDS[0],30)
    pre=torch.load(run.old.SOURCE,weights_only=False)['preprocessing']
    cache=torch.load(run.old.SOURCE_GRAPH_CACHE,weights_only=False)
    cache.update(torch.load(run.old.OLD/'data_audit/graph_cache_8g_only.pt',weights_only=False))
    ids=usage['gradient_train'][:3]
    graphs=run.old.read_graphs(run.target_path('8g'),ids,cache,pre)
    f=run.metadata(run.target_path('8g'),ids);norm=TrainingNormalizer().fit(f,ids)
    run.attach(graphs,f,norm);a,b=run.batch(graphs)
    raw=ColumnConditionedQGeoGNN(source).eval();normalized=ColumnConditionedQGeoGNN(source,True).eval()
    with torch.no_grad():
        expected=source(a,b);torch.testing.assert_close(raw(a,b),expected,rtol=0,atol=0)
        torch.testing.assert_close(normalized(a,b)*4,expected,rtol=0,atol=0)
        a.column_context.fill_(100);torch.testing.assert_close(raw(a,b),expected,rtol=0,atol=0)
        raw.column_branch[-1].weight.fill_(.001)
        assert not torch.equal(raw(a,b),expected)
    assert sum(isinstance(m,torch.nn.Linear) and m.out_features==6 for m in raw.modules())==1
    assert sum(p.numel() for p in raw.parameters())==461208
    for k,v in source.state_dict().items(): torch.testing.assert_close(v,original[k],rtol=0,atol=0)
    raw.training_joint()
    assert all(not m.training for m in raw.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm))


def test_evaluation_requires_global_freeze(tmp_path,monkeypatch):
    from scripts.studies import evaluate_physics_column_transfer as evaluate
    monkeypatch.setattr(evaluate,'OUT',tmp_path)
    monkeypatch.setattr(run,'prepare',lambda:None)
    with pytest.raises(FileNotFoundError): evaluate.validate_freeze()
    run.atomic_json(tmp_path/'all_predictions_frozen.json',{'contexts':119,'files':{}})
    with pytest.raises(RuntimeError): evaluate.validate_freeze()


def test_joint_fit_and_resume_without_test_or_donors(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    usage=run.old.ledger('8g','row',run.SEEDS[0],30)
    pre=torch.load(run.old.SOURCE,weights_only=False)['preprocessing']
    target,source,source_ids,_,_=run.old.load_fitting_inputs(usage,pre)
    sf=run.metadata(run.old.SOURCE_DATA,source_ids)
    tf={r:run.metadata(run.target_path('8g'),usage[r]) for r in ['gradient_train','validation']}
    norm=TrainingNormalizer().fit(pd.concat([sf,tf['gradient_train']],ignore_index=True),source_ids+usage['gradient_train'])
    run.attach(source,sf,norm)
    for r in tf:run.attach(target[r],tf[r],norm)
    monkeypatch.setattr(run,'STUDY',tmp_path)
    monkeypatch.setitem(run.CONFIG,'maximum_epochs',2)
    monkeypatch.setattr(run,'selected_truth',lambda *args:pytest.fail('fit tried to read extra labels'))
    contract={'column':'8g','protocol':'row','budget':30,'target_train_ids':usage['gradient_train'],'target_validation_ids':usage['validation']}
    for method in run.NEURAL:
        result=run.fit(method,17,target,source,contract,pre['target_scales'])
        assert result['source_draws']==44 and result['balanced_batches'] and result['shared_head']
        assert result==run.fit(method,17,target,source,contract,pre['target_scales'])
        model=ColumnConditionedQGeoGNN(run.load_predictor_checkpoint(run.old.SOURCE),method.startswith('mass_normalized'))
        model.load_state_dict(torch.load(tmp_path/'runtime'/run.context_path('8g','row',17,30)/method/'best.pt',weights_only=False)['model'])
        assert np.isfinite(run.predict(model,target['validation'])).all()


def test_evaluation_does_not_read_truth_without_freeze(monkeypatch):
    from scripts.studies import evaluate_physics_column_transfer as ev
    def deny(): raise RuntimeError('incomplete')
    monkeypatch.setattr(ev,'validate_freeze',deny)
    monkeypatch.setattr(run,'selected_truth',lambda *args:pytest.fail('premature test read'))
    with pytest.raises(RuntimeError):ev.evaluate()


def test_nonfinite_context_rejected():
    f=frame();f.loc[0,'Flow mL/min']=np.nan
    with pytest.raises(ValueError):context_matrix(f)
    f=frame();f.loc[1,'Flow mL/min']=0
    with pytest.raises(ValueError):context_matrix(f)


def test_completed_artifacts_when_present():
    path=run.STUDY/'all_predictions_frozen.json'
    if not path.exists():pytest.skip('formal run has not reached global freeze')
    from scripts.studies.evaluate_physics_column_transfer import validate_freeze
    freeze=validate_freeze()
    assert len(freeze['files'])==120
    for name in freeze['files']:
        output=(run.STUDY/name).parent
        pred=pd.read_csv(output/'predictions_blind.csv.gz').set_index('sample_id')
        assert np.isfinite(pred.to_numpy()).all()
        usage=json.loads((output/'label_usage.json').read_text())
        assert list(pred.index)==usage['test']
        assert set(usage['normalization_ids'])==set(usage['source_train_ids']+usage['gradient_train'])
        assert not set(usage['normalization_ids'])&set(usage['test']+usage['validation'])
        assert usage['other_target_column_labels_used']==0
        fits=json.loads((output/'fit_audit.json').read_text())
        for method,fit in fits.items():
            runtime=run.STUDY/'runtime'/output.relative_to(run.STUDY/'contexts')/method
            assert run.sha256_file(runtime/'best.pt')==fit['checkpoint_sha256']
            history=pd.read_csv(runtime/'history.csv')
            assert list(history.epoch)==list(range(1,fit['epochs_run']+1))
            best=history.loc[history.validation_score.idxmin()]
            assert int(best.epoch)==fit['best_epoch']
            assert best.validation_score==pytest.approx(fit['validation_score'],rel=1e-12)
            assert (history.source_rows==history.target_rows).all()
            assert fit['source_draws']==int(history.source_rows.sum())
            assert fit['shared_head'] and fit['balanced_batches'] and fit['fit_success']
    event=json.loads((run.STUDY/'test_evaluation_started.json').read_text())
    for name in freeze['files']:
        record=json.loads((run.STUDY/name).read_text())
        assert record['frozen_at_unix']<event['unix_time']


def test_success_gate_rejects_output_tradeoff_and_requires_replication():
    from scripts.studies.evaluate_physics_column_transfer import paired,decision
    rows=[]
    for c in run.COLUMNS:
        for protocol in ['row','compound']:
            for seed in range(5):
                for m in [*run.BASELINES,*run.METHODS]:
                    rows.append(dict(column=c,protocol=protocol,seed=seed,method=m,combined_normalized_rmse=.8 if m in run.METHODS else 1.,V1_rmse=1.2 if m in run.METHODS else 1.,V2_rmse=.5 if m in run.METHODS else 1.))
    q=paired(pd.DataFrame(rows),'combined_normalized_rmse','budget100')
    assert not q.material.any()
    assert not q.no_output_regression.any()
    q['endpoint']='aulc';q['material']=True
    only=q.loc[q.column.eq('25g')&q.protocol.eq('compound')]
    assert not decision(only)['gates']['raw_column_conditioned']['aulc']['replicated']
    two=q.loc[q.column.isin(['25g','40g'])&q.protocol.eq('compound')]
    assert decision(two)['gates']['raw_column_conditioned']['aulc']['replicated']
