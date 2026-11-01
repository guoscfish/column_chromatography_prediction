"""Offline budget extension and immutable-prefix handover checks."""
from contextlib import nullcontext
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def m(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / 'scripts/studies/run_qgeognn_v2_row_llm_nocw_to1389.py'
    spec = importlib.util.spec_from_file_location('nocw_extension_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'SOURCE', tmp_path / 'source')
    monkeypatch.setattr(module, 'STUDY', tmp_path / 'extension')
    monkeypatch.setattr(module, 'POLICY', module.STUDY / 'policy.json')
    module.STUDY.mkdir()
    module.POLICY.write_text('{}')
    return module


def test_exact_authorized_grid_and_twelve_extra_batches(m):
    assert m.BUDGETS[21] == 1005 and m.BUDGETS[-1] == 1389
    assert len(m.BUDGETS) == 34 and len(m.BUDGETS[22:]) == 12
    assert set(b-a for a,b in zip(m.BUDGETS,m.BUDGETS[1:])) == {32}
    assert 1400 not in m.BUDGETS and 1421 not in m.BUDGETS


@pytest.mark.parametrize('phase,budget', [('SELECTING',None),('ADVANCING',None),('FAILED_CLOSED',None),('COMPLETE',909)])
def test_source_must_finish_before_any_import_or_dispatch(m, phase, budget):
    p=m.SOURCE/'progress';p.mkdir(parents=True)
    (p/'seed_157.json').write_text(json.dumps({'phase':phase,'final_budget':budget}))
    assert not m.source_complete(157)
    with pytest.raises(RuntimeError,match='finish L1005'):
        m.import_prefix(157)
    assert not (m.STUDY/'selections').exists()


def test_busy_completed_source_is_deferred_without_interruption(m,monkeypatch):
    p=m.SOURCE/'progress';p.mkdir(parents=True)
    (p/'seed_6101.json').write_text(json.dumps({'phase':'COMPLETE','final_budget':1005}))
    monkeypatch.setattr(m,'lock_busy',lambda *a:True)
    monkeypatch.setattr(m.time,'sleep',lambda *a:None)
    monkeypatch.setattr(m.os,'kill',lambda *a:pytest.fail('old worker signalled'))
    assert not m.source_complete(6101)


def test_ambiguous_source_cannot_be_imported(m,monkeypatch):
    p=m.SOURCE/'progress';p.mkdir(parents=True)
    (p/'seed_6101.json').write_text(json.dumps({'phase':'COMPLETE','final_budget':1005}))
    monkeypatch.setattr(m,'lock_busy',lambda *a:False)
    monkeypatch.setattr(m,'unresolved_at',lambda *a:['started'])
    with pytest.raises(RuntimeError,match='ambiguous'):
        m.source_complete(6101)


@pytest.mark.parametrize('seed,rnd',[(157,21),(887,32),(2357,25),(6101,21),(12203,32)])
def test_new_rounds_keep_unlimited_strict_corrections_and_no_hash_hint(m,monkeypatch,seed,rnd):
    monkeypatch.setattr(m.legacy,'STUDY',m.STUDY)
    monkeypatch.setattr(m.s,'BUDGETS',m.BUDGETS)
    monkeypatch.setattr(m.s,'prompt',lambda v:'same scientific prompt')
    monkeypatch.setattr(m.s,'audit_no_cw',lambda x:None)
    encoded={'candidates':{'columns':['id'],'rows':[[str(i)] for i in range(32)]}}
    monkeypatch.setattr(m.s,'encode_catalog',lambda c:encoded)
    catalog=SimpleNamespace(pending=[],candidates=list(range(32)),observed=list(range(m.BUDGETS[rnd])))
    packet={**encoded,'selection_count':32,'candidate_count':32,'observed_count':len(catalog.observed),'active_label_count':len(catalog.observed)}
    packet['packet_sha256']=m.s.stable_hash(packet)
    config={'selection_attempts':3,'model':'gpt-6.1-sol','reasoning_effort':'xhigh'}
    calls=[]
    def saved(messages,c,directory,turn,call):
        calls.append(turn)
        assert c==config and json.loads(messages[1]['content'])==packet
        assert all(msg['content'] != m.legacy.packet_hash_hint(packet) for msg in messages)
        return {'answer':json.dumps({'valid':turn==101})}
    def check(result,*args):
        if not result['valid']:raise ValueError('distinct legal supplied IDs and exact packet hash required')
        return result
    monkeypatch.setattr(m.s,'request_with_retries',saved)
    monkeypatch.setattr(m.s,'validate_plan',check)
    d=m.STUDY/f'selections/seed_{seed}/{m.ARM}/round_{rnd:02d}'
    assert m.legacy.run_selection(packet,catalog,d,'no_cw',config,None)['valid']
    assert calls==list(range(102)) and (d/'request_101.json').exists()
    assert not (d/'request_102.json').exists()


def test_offline_audit_never_falls_through_to_a_missing_paid_request(m,monkeypatch):
    monkeypatch.setattr(m.s,'request_with_retries',lambda *a:pytest.fail('paid request'))
    with m.offline_saved_answers():
        with pytest.raises(RuntimeError,match='refuses any new provider'):
            m.s.request_with_retries([],{},m.STUDY,0,None)


def test_offline_saved_answer_still_binds_request_and_served_model(m):
    (m.STUDY/'served_model_lock.json').write_text(json.dumps({'served_model':'locked'}))
    (m.STUDY/'turn_00.json').write_text(json.dumps({'success':True,'request_sha256':'wrong','provenance':{'served_model':'locked'}}))
    with m.offline_saved_answers():
        with pytest.raises(RuntimeError,match='binding drift'):
            m.s.request_with_retries([],{},m.STUDY,0,None)


def test_prefix_fit_reuse_does_not_rewrite_absolute_paths_or_retrain(m,monkeypatch):
    source=m.SOURCE/'old.pt';source.parent.mkdir();source.write_bytes(b'checkpoint')
    predictions=m.SOURCE/'old.csv.gz';predictions.write_bytes(b'predictions')
    d=m.STUDY/f'runtime/seed_6101/{m.ARM}/round_21';d.mkdir(parents=True)
    ids=['a','b'];f={'checkpoint_path':str(source),'checkpoint_sha256':m.s.sha256_file(source),
        'prediction_path':str(predictions),'prediction_sha256':m.s.sha256_file(predictions),
        'L_t_ids_hash':m.s.ids_hash(ids),'test_truth_access_count':0}
    p=d/'prediction_freeze.json';p.write_text(json.dumps(f));before=p.read_bytes()
    monkeypatch.setattr(m,'ORIGINAL_PREDICTION_FREEZE',lambda *a:pytest.fail('prefix refit/refreeze'))
    ctx=SimpleNamespace(seed=6101,ids=lambda labeled:ids)
    checkpoint,runtime=m.prediction_freeze(ctx,m.ARM,21,ids)
    assert checkpoint==source and runtime==d and p.read_bytes()==before


def test_extended_batch_requires_its_new_amendment_binding(m,monkeypatch):
    d=m.STUDY/'batch';d.mkdir();p=d/'contract.json'
    p.write_text(json.dumps({'round':21,'extension_amendment_sha256':'wrong'}))
    monkeypatch.setattr(m,'ORIGINAL_AUDIT_BATCH',lambda *a:pytest.fail('unbound batch accepted'))
    with pytest.raises(RuntimeError,match='authorized amendment'):
        m.audit_batch(d)


@pytest.mark.parametrize('seed',[157,887,2357,6101,12203])
def test_worker_runs_only_twelve_new_batches_and_reuses_all_twenty_one_prefix_fits(m,monkeypatch,seed):
    monkeypatch.setattr(m,'activate',lambda *a:None)
    monkeypatch.setattr(m,'require_authorization',lambda:None)
    monkeypatch.setattr(m,'verify_prefix',lambda *a:None)
    monkeypatch.setattr(m,'source_complete',lambda *a:True)
    monkeypatch.setattr(m,'unresolved_at',lambda *a:[])
    monkeypatch.setattr(m,'audit_fit',lambda *a:None)
    monkeypatch.setattr(m.base,'credential_audit',lambda:{})
    monkeypatch.setattr(m.base,'BASE_EXCLUSIVE_LOCK',lambda p:nullcontext())
    monkeypatch.setattr(m.base,'KEY_FILES',{seed:m.STUDY/'mock-key'})
    statuses=[];calls=[]
    monkeypatch.setattr(m.base,'status_record',lambda seed,phase,**kw:statuses.append((phase,kw)))
    monkeypatch.setattr(m.s,'select',lambda seed,arm,rnd,**kw:calls.append(('select',seed,arm,rnd)))
    monkeypatch.setattr(m.s,'advance',lambda seed,arm,rnd,**kw:calls.append(('fit',seed,arm,rnd)))
    m.worker(seed)
    assert calls==[(kind,seed,m.ARM,rnd) for rnd in range(21,33) for kind in ('select','fit')]
    assert statuses[-1]==('COMPLETE',{'final_budget':1389,'arms':[m.ARM]})


def test_existing_dispatch_intent_blocks_abnormal_worker_restart(m,monkeypatch):
    p=m.STUDY/'progress';p.mkdir()
    for seed in m.base.SEEDS:(p/f'dispatch_seed_{seed}.intent.json').write_text('{}')
    monkeypatch.setattr(m,'activate',lambda:None)
    monkeypatch.setattr(m,'require_authorization',lambda:None)
    monkeypatch.setattr(m,'validate',lambda:None)
    monkeypatch.setattr(m.base,'BASE_EXCLUSIVE_LOCK',lambda p:nullcontext())
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**kw:pytest.fail('abnormal worker restarted'))
    def stop(*args):raise LookupError('one iteration')
    monkeypatch.setattr(m.time,'sleep',stop)
    with pytest.raises(LookupError):m.supervise()
    rows=json.loads((p/'supervisor.json').read_text())['workers']
    assert [r['supervision'] for r in rows]==['NEEDS_RECONCILIATION']*5


def test_source_workers_are_left_running_without_early_extension_calls(m,monkeypatch):
    p=m.STUDY/'progress';p.mkdir()
    q=m.SOURCE/'progress';q.mkdir(parents=True)
    for seed in m.base.SEEDS:(q/f'seed_{seed}.json').write_text(json.dumps({'phase':'SELECTING','pid':seed}))
    monkeypatch.setattr(m,'activate',lambda:None)
    monkeypatch.setattr(m,'require_authorization',lambda:None)
    monkeypatch.setattr(m,'validate',lambda:None)
    monkeypatch.setattr(m,'source_complete',lambda seed:False)
    monkeypatch.setattr(m,'lock_busy',lambda *a:True)
    monkeypatch.setattr(m.base,'BASE_EXCLUSIVE_LOCK',lambda p:nullcontext())
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**kw:pytest.fail('duplicate source worker'))
    monkeypatch.setattr(m.os,'kill',lambda *a:pytest.fail('source worker interrupted'))
    def stop(*args):raise LookupError('one iteration')
    monkeypatch.setattr(m.time,'sleep',stop)
    with pytest.raises(LookupError):m.supervise()
    assert all(r['supervision']=='SOURCE_L1005_RUNNING' for r in json.loads((p/'supervisor.json').read_text())['workers'])


def test_final_report_is_blocked_until_all_full_l1389_trajectories_exist(m,monkeypatch):
    monkeypatch.setattr(m,'activate',lambda:None)
    monkeypatch.setattr(m,'require_authorization',lambda:None)
    p=m.STUDY/'progress';p.mkdir()
    (p/'seed_157.json').write_text(json.dumps({'phase':'COMPLETE','final_budget':1005}))
    q=m.STUDY/f'runtime/seed_157/{m.ARM}';q.mkdir(parents=True)
    (q/'trajectory_freeze.json').write_text(json.dumps({'final_active_labels':1005}))
    monkeypatch.setattr(m.base,'report',lambda *a:pytest.fail('premature final report'))
    with pytest.raises(RuntimeError,match='full L1389'):m.report()
