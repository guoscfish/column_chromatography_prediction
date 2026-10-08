#!/usr/bin/env python
"""Verify delivered study artifacts; never trains, reveals labels or calls LLMs."""
from pathlib import Path
import json
import hashlib
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
S=ROOT/'studies/active_learning/qgeognn_v2_row_cw_selector_factorial'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=read(S/'protocol.json'); ledger=read(S/'fit_ledger.json')
checks=[]
for key,item in ledger.items():
    if item['status']!='completed':continue
    seed,arm,r=key.split('/');r=int(r)
    root=S/f'runtime/seed_{seed}/{arm}'
    d=root/f'round_{r:02d}/model';a=read(d/'fit_audit.json')
    assert sha(d/'fit_audit.json')==item['audit_sha256']
    assert sha(d/'best.pt')==a['checkpoint_sha256']
    assert sha(d/'predictions.csv.gz')==a['prediction_sha256']
    assert a['initialization_hash']==p['contexts'][seed]['initialization_hash']
    assert a['train_rows']==333+32*r
    assert a['test_labels_used_for_fit_or_checkpoint_selection']==0
    before=root/f'round_{r-1:02d}'
    f=read(before/'batch_freeze.json')
    ids=f['ids'];l=f['contract']['ordered_l'];u=f['contract']['ordered_u']
    assert len(ids)==32 and len(set(ids))==32 and set(ids)<=set(u) and not set(ids)&set(l)
    assert [u[i] for i in f['positions']]==ids
    assert sha(before/'premeasurement.csv.gz')==f['premeasurement_sha256']
    assert pd.read_csv(before/'premeasurement.csv.gz').sample_id.astype(str).tolist()==ids
    checks.append({'key':key,'audit_sha256':sha(d/'fit_audit.json'),'batch_freeze_sha256':sha(before/'batch_freeze.json'),
                   'checkpoint_sha256':a['checkpoint_sha256'],'prediction_sha256':a['prediction_sha256'],'all_checks_passed':True})
assert len(ledger)<=36
for name,digest in read(S/'execution_code_freeze.json').items():assert sha(ROOT/name)==digest,name
for name,digest in read(S/'supplemental_code_freeze.json').items():assert sha(ROOT/name)==digest,name
historical=ROOT/'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/astra_high_full_pool_20261001'
for name,digest in read(historical/'protocol.json')['code_hashes'].items():assert sha(ROOT/name)==digest,name
result={'status':'completed' if len(checks)==36 else 'partial_verified','new_fit_count':len(checks),'test_reveal_count':0,'paid_llm_calls':0,'fits':checks}
(S/'DELIVERY_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
if len(checks)==36:
    assert read(S/'RUN_STATE.json')['P1a']['status']=='completed'
    assert read(S/'RUN_STATE.json')['P1b']['status']=='completed'
    manifest={'status':'complete_global_matrix_predictions_frozen','new_test_reveals':0,'budgets':p['budgets'],
              'new_fits':checks,'controls':read(S/'REUSE_MANIFEST.json')['entries'],
              'protocol_sha256':sha(S/'protocol.json'),'validation_curves_sha256':sha(S/'validation_curves.csv')}
    path=S/'GLOBAL_MATRIX_FREEZE.json'
    if path.exists():assert read(path)==manifest
    else:path.write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'status':result['status'],'verified_fits':len(checks),'protected_historical_source_unchanged':True}))
