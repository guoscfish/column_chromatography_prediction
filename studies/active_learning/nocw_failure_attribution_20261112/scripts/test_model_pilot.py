#!/usr/bin/env python3
"""Meaningful pilot boundary checks; model forward only, never an optimizer step."""
import importlib.util,json,sys,tempfile
from pathlib import Path
import torch
P=Path(__file__).resolve().parent/'model_pilot.py'
spec=importlib.util.spec_from_file_location('pilot',P);pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
sys.path.insert(0,str(pilot.ROOT/'src'))
def main():
    # Same-condition closure and overlapping pairs must be held out as a unit.
    meta={}
    for i,v,pea in [('a',50,'2/1'),('b',100,'2/1'),('c',100,'2/1'),('d',100,'5/1')]:
        meta[i]={'smiles':'CCO','conditions':dict(zip(pilot.FIELDS,[pea,1.0,v,'DCM',300]))}
    gs=pilot.components(meta,list(meta));assert len(gs)==1 and gs[0]['ids']==list(meta)
    assert set(e['axis'] for e in gs[0]['pair_edges'])=={'V/ul','PE/EA'}
    assert pilot.norm('2/1','PE/EA')==pilot.norm('4/2','PE/EA')
    assert pilot.norm(34.01360544,'V/ul')!=pilot.norm(34.01,'V/ul')
    # Zero residual: exact same initial function. Nonzero residual: explicitly
    # molecule dependent, bounded gain, no additional learned parameter.
    torch.manual_seed(2);p=torch.randn(8,128);r=torch.zeros_like(p)
    assert torch.equal(pilot.interaction_fusion(p,r),p)
    r=torch.randn_like(p);delta=pilot.interaction_fusion(p,r)-p
    assert torch.all(delta.abs()<=2*r.abs()+1e-6)
    # Frozen actual graph forward checks full model state/init, not only algebra.
    from qgeognn_al.schemas.conditions import ConditionNormalization
    from qgeognn_al.training.predictor import loader_pair,seed_everything
    torch.set_num_threads(2)
    runtime=pilot.ROOT/'studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_157'
    ctx=json.loads((runtime/'context.json').read_text())['contract'];norm=ConditionNormalization(**ctx['normalization'])
    atom,angle=torch.load(runtime/'scrubbed_graphs.pt',map_location='cpu',weights_only=False)
    assert all(torch.count_nonzero(g.y).item()==0 for g in atom)
    seed_everything(2000160);baseline=pilot.build_pilot_model(norm,'baseline').eval()
    seed_everything(2000160);interaction=pilot.build_pilot_model(norm,'interaction').eval()
    assert set(baseline.state_dict())==set(interaction.state_dict())
    assert all(torch.equal(v,interaction.state_dict()[k]) for k,v in baseline.state_dict().items())
    for a,b in zip(*loader_pair(atom,angle,[0,1,2],3)):
        with torch.no_grad():assert torch.equal(baseline(a,b),interaction(a,b))
    assert sum(p.numel() for p in baseline.parameters())==sum(p.numel() for p in interaction.parameters())==458952
    # The label boundary must reject a test role before opening a source file.
    fake={str(i):{'role':'validation'} for i in range(416)};fake['0']['role']='test'
    try:pilot.restricted_validation_truth(list(fake),fake,Path('/no/such/source'))
    except AssertionError:pass
    else:raise AssertionError('test role passed validation allowlist')
    # Check qualified H groups against initial and training exact condition keys.
    q=json.loads((pilot.OUT/'verification/pilot_qualification.json').read_text())
    assert len(q['states'])==4 and all(s['qualification'] for s in q['states'])
    assert sum(4 for s in q['states'] if s['qualification'])==16
    slots={seed:{(s['seed'],s['available_budget'],a,r) for s in q['states'] if s['qualification'] and s['seed']==seed for a in ['baseline','interaction'] for r in [0,1]} for seed in [157,887]}
    assert len(slots[157])==len(slots[887])==8 and not slots[157]&slots[887] and len(slots[157]|slots[887])==16
    import lzma
    snapshot=json.loads(lzma.decompress(pilot.SNAP.read_bytes()));metadata=pilot.feature_metadata()
    for s in q['states']:
        keys={pilot.key(metadata[i]) for i in s['H_ids']}
        valkeys={pilot.key(metadata[i]) for i,r in snapshot['partitions'][str(s['seed'])].items() if r['role']=='validation'}
        assert not keys&valkeys
        assert not keys&{pilot.key(metadata[i]) for i in s['train_ids']}
        assert not keys&{pilot.key(metadata[i]) for i in s['L0_ids']}
    result={'status':'PASS','tests':['shared endpoints and duplicate closure','exact ratios and decimal precision','zero-residual exact equality',
      'bounded multiplicative term','full 458952-parameter paired initialization','full graph forward identical at initialization',
      'test role refused before source access','H versus validation/train/L333 recorded-key disjointness','two worker slot sets disjoint, 8+8=16','fixed H qualification yields at most16fits'],
      'successful_fits':0,'optimizer_steps':0,'test_targets_read':0,'candidate_targets_read':0}
    pilot.dump(pilot.OUT/'verification/pilot_preflight_tests.json',result);print(json.dumps(result))
if __name__=='__main__':main()
