#!/usr/bin/env python3
"""NoCW fixed-label model pilot. Metadata qualification is separate from fitting.

Never imports an AL runner or selector. Training remains gated by an explicit
frozen C-gate/root authorization file and the maximum successful-fit ledger.
"""
from __future__ import annotations
from pathlib import Path
import argparse,collections,csv,hashlib,json,lzma,math,os,sys
from decimal import Decimal
from fractions import Fraction

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[2]
BASE=ROOT/'studies/active_learning/qgeognn_v2_row_llm_nocw_ablation/revisions/five_seed_nocw_to1005_20261029'
SNAP=ROOT/'studies/active_learning/nocw_free_selection_diagnostics_20261101/snapshot_inputs.json.xz'
FIELDS=['PE/EA','Density g/ml','V/ul','loading solvent','Volume of loading solvent/ul']
RULE=OUT/'verification/pilot_metadata_rule.json'
SOURCES={}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):
    p=Path(p);b=p.read_bytes();SOURCES[str(p.relative_to(ROOT))]={'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)};return b
def js(p):return json.loads(read(p))
def stable(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def dump(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write_once(p,v):
    p=Path(p)
    if p.exists():
        if json.loads(p.read_text())!=v:raise RuntimeError(f'fixed artifact differs: {p}')
    else:dump(p,v)
def csvout(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def norm(v,axis):
    if axis=='PE/EA':
        a,b=str(v).split('/');return str(Fraction(Decimal(a))/Fraction(Decimal(b)))
    if axis=='loading solvent':return str(v)
    return str(Decimal(str(v)).normalize())
def key(meta):return (meta['smiles'],*[norm(meta['conditions'][k],k) for k in FIELDS])
def state_contract(seed,budget):
    r=(budget-333)//32
    for rev in ['five_seed_nocw_to1005_20261029','five_seed_nocw_to1389_20261101']:
        p=BASE.parent/rev/f'selections/seed_{seed}/llm32_nocw/round_{r:02d}/contract.json'
        if p.exists():return p,js(p)
    # Completion state has no next accepted proposal. Derive L from previous
    # premeasurement contract plus its already accepted and frozen final batch.
    p=BASE/f'selections/seed_{seed}/llm32_nocw/round_{r-1:02d}/contract.json';d=js(p)
    fr=js(p.parent/'batch_freeze.json');ids=d['L_t_ids']+fr['batch_ids']
    assert len(ids)==budget and len(set(ids))==budget
    return p,{'seed':seed,'round':r,'L_t_ids':ids,'U_t_ids':[i for i in d['U_t_ids'] if i not in set(fr['batch_ids'])],
      'derived_terminal_state':True,'previous_contract':str(p.relative_to(ROOT))}
def components(meta,eligible):
    parent={i:i for i in eligible}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    mols=collections.defaultdict(list)
    for i in eligible:mols[meta[i]['smiles']].append(i)
    edges=[]
    for sm,ids in sorted(mols.items()):
        for a_pos,a in enumerate(sorted(ids)):
            ka=key(meta[a])
            for b in sorted(ids)[a_pos+1:]:
                kb=key(meta[b]);dif=[i for i,(x,y) in enumerate(zip(ka[1:],kb[1:])) if x!=y]
                if len(dif)<=1:
                    union(a,b)
                    if len(dif)==1:edges.append({'a':a,'b':b,'axis':FIELDS[dif[0]],'smiles':sm})
    groups=collections.defaultdict(list)
    for i in eligible:groups[find(i)].append(i)
    result=[]
    for ids in groups.values():
        ids=sorted(ids);es=[e for e in edges if e['a'] in set(ids) and e['b'] in set(ids)]
        if not es:continue
        v=any(e['axis']=='V/ul' for e in es)
        result.append({'group_id':stable(ids)[:20],'ids':ids,'molecule':meta[ids[0]]['smiles'],'volume':v,'axes':sorted({e['axis'] for e in es}),'pair_edges':es})
    return sorted(result,key=lambda g:(not g['volume'],stable(g['ids'])))
def feature_metadata():
    """Read exactly identity and covariate columns, never response columns."""
    import pandas as pd
    source=ROOT/'experiments/e0_4g_baseline/canonical_4g.csv'
    fields=['sample_id','canonical_smiles']+FIELDS
    frame=pd.read_csv(source,usecols=fields,dtype=str,keep_default_na=False)
    assert set(frame.columns)==set(fields) and len(frame)==4163 and frame.sample_id.is_unique
    SOURCES[str(source.relative_to(ROOT))]={'path':str(source.relative_to(ROOT)),'sha256':sha(source),'bytes':source.stat().st_size,
      'role':'X-only metadata read; exact usecols sample_id/canonical_smiles/five conditions, dtype=str; no response columns loaded'}
    return {r['sample_id']:{'id':r['sample_id'],'smiles':r['canonical_smiles'],'conditions':{k:r[k] for k in FIELDS}} for r in frame.to_dict('records')}
def metadata_qualification():
    rule=js(RULE);snapshot=json.loads(lzma.decompress(read(SNAP)));meta=feature_metadata()
    assert all(set(m)=={'id','smiles','conditions'} for m in meta.values()),'metadata must contain no response fields'
    all_states=[];group_rows=[];h_rows=[]
    for seed in rule['outer_seeds']:
        l0=set(snapshot['initial'][str(seed)]);assert len(l0)==333
        l0keys={key(meta[i]) for i in l0}
        valkeys={key(meta[i]) for i,v in snapshot['partitions'][str(seed)].items() if v['role']=='validation'}
        for budget in rule['available_label_budgets']:
            cp,c=state_contract(seed,budget);l=list(c['L_t_ids']);assert len(l)==len(set(l))==budget and l0<=set(l)
            assert not any(snapshot['partitions'][str(seed)][i]['role'] not in ['l0','u0'] for i in l)
            eligible=[i for i in l if i not in l0 and key(meta[i]) not in l0keys and key(meta[i]) not in valkeys]
            gs=components(meta,eligible);volume=[g for g in gs if g['volume']]
            qualified=len(gs)>=rule['minimum_qualified_components'] and len(volume)>=rule['minimum_volume_components'] and len({g['molecule'] for g in volume})>=rule['minimum_volume_molecules']
            selected=gs[:10] if qualified else []
            H=sorted({i for g in selected for i in g['ids']});train=[i for i in l if i not in set(H)]
            assert not(set(H)&l0) and not({key(meta[i]) for i in H}&l0keys)
            assert not({key(meta[i]) for i in H}&{key(meta[i]) for i in train})
            assert not({key(meta[i]) for i in H}&valkeys),'H/validation recorded-condition leakage'
            record={'seed':seed,'available_budget':budget,'qualification':qualified,'eligible_post_l0_rows':len(eligible),
              'qualified_components':len(gs),'volume_components':len(volume),'volume_molecules':len({g['molecule'] for g in volume}),
              'H_components':len(selected),'H_rows':len(H),'actual_train_rows':len(train),'H_pairs':sum(len(g['pair_edges']) for g in selected),
              'H_volume_pairs':sum(sum(e['axis']=='V/ul' for e in g['pair_edges']) for g in selected),
              'H_ids':H,'train_ids':train,'L_ids':l,'L0_ids':sorted(l0),'groups':selected,'contract_source':str(cp.relative_to(ROOT)),
              'reason': 'eligible' if qualified else 'fails prespecified ten-component and/or two-volume-molecule qualification; no criterion reduced'}
            all_states.append(record)
            for g in gs:
                group_rows.append({'seed':seed,'available_budget':budget,'group_id':g['group_id'],'molecule_sha256':stable(g['molecule']),
                  'rows':len(g['ids']),'strict_pair_edges':len(g['pair_edges']),'has_volume_pair':g['volume'],'axes':'|'.join(g['axes']),
                  'in_H':g in selected,'qualification':qualified})
            for i in H:h_rows.append({'seed':seed,'available_budget':budget,'sample_id':i,'molecule_sha256':stable(meta[i]['smiles']),
              **meta[i]['conditions'],'role':'held_out_already_observed; metadata-only selection'})
    payload={'status':'METADATA_ONLY_QUALIFIED','rule_sha256':sha(RULE),'states':all_states,'successful_fits_performed':0,
      'responses_used_to_choose_H':False,'successful_fit_cap':16,'rule_fixed_before_counting':True}
    write_once(OUT/'verification/pilot_qualification.json',payload)
    csvout(OUT/'tables/pilot_state_qualification.csv',[{k:v for k,v in s.items() if k not in ['H_ids','train_ids','L_ids','L0_ids','groups']} for s in all_states])
    csvout(OUT/'tables/pilot_metadata_groups.csv',group_rows);csvout(OUT/'tables/pilot_H_metadata.csv',h_rows)
    dump(OUT/'verification/pilot_metadata_inputs.json',list(SOURCES.values()))
    print(json.dumps([{k:s[k] for k in ['seed','available_budget','qualification','qualified_components','volume_components','volume_molecules','H_rows','actual_train_rows']} for s in all_states]))

def freeze_protocol():
    """Seal the one model design and H identities before reading fit targets."""
    cg=js(OUT/'verification/C_gate.json')
    assert str(cg.get('status','')).startswith('PASS'), 'C gate is not passed'
    release=js(OUT/'verification/pilot_root_release.json')
    assert release.get('training_authorized') is True and release.get('successful_fit_cap')==16
    q=js(OUT/'verification/pilot_qualification.json');assert q['rule_sha256']==sha(RULE)
    data_sources={e['path']:e['sha256'] for e in js(OUT/'verification/pilot_metadata_inputs.json')}
    for state in q['states']:
        cp,c=state_contract(state['seed'],state['available_budget'])
        for p in [cp,cp.parent/'input.json']:
            data_sources[str(p.relative_to(ROOT))]=sha(p)
        if c.get('derived_terminal_state'):
            p=cp.parent/'feedback.json';data_sources[str(p.relative_to(ROOT))]=sha(p)
    for seed in [157,887]:
        ctx=ROOT/f'studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_{seed}/context.json'
        for p in [ctx,ctx.parent/'scrubbed_graphs.pt']:
            data_sources[str(p.relative_to(ROOT))]=sha(p)
    design={'version':'nocw_condition_interaction_pilot_v1','single_change':'z=p+r+r*tanh(nonaffine_layer_norm(p))',
      'new_parameters':0,'total_parameters_per_arm':458952,'outer_seeds':[157,887],'available_budgets':[525,1005],
      'training_replicates':[0,1],'maximum_successful_fits':16,
      'initialization_seed':'outer_seed+2000003+replicate*1000033','training_rng_seed':'outer_seed+3000017+replicate*1000033',
      'training':{'optimizer':'Adam','learning_rate':.001,'weight_decay':0.,'batch_size':2048,'maximum_epochs':1000,'patience':100,'cpu_threads':2},
      'execution_schedule':{'maximum_concurrent_workers':2,'outer_seed_per_worker':[157,887],'slots_per_worker':8,'maximum_total_CPU_threads':4},
      'checkpoint_selection':'original 416 validation combined NRMSE; never H',
      'labels_for_H_selection':False,'training_label_source':'same-state NoCW observed + accepted terminal feedback',
      'validation_label_source':'416 explicit allowed validation IDs; streaming canonical target store, no other numeric targets',
      'holdout_qualification_sha256':sha(OUT/'verification/pilot_qualification.json'),
      'metadata_rule_sha256':sha(RULE),'script_sha256':sha(Path(__file__)),
      'protocol_sha256':sha(OUT/'MODEL_PILOT_PROTOCOL.md'),'C_gate_sha256':sha(OUT/'verification/C_gate.json'),
      'root_release_sha256':sha(OUT/'verification/pilot_root_release.json'),
      'data_sources':data_sources,
      'source_hashes':{name:sha(ROOT/name) for name in [
        'src/qgeognn_al/models/qgeognn_v2.py','src/qgeognn_al/schemas/conditions.py',
        'src/qgeognn_al/training/predictor.py','src/qgeognn_al/evaluation/point.py']}}
    write_once(OUT/'verification/pilot_protocol_freeze.json',design)
    print('FROZEN',stable(design))

def check_freeze():
    d=js(OUT/'verification/pilot_protocol_freeze.json')
    for p,keyname in [(Path(__file__),'script_sha256'),(OUT/'MODEL_PILOT_PROTOCOL.md','protocol_sha256'),
      (RULE,'metadata_rule_sha256'),(OUT/'verification/pilot_qualification.json','holdout_qualification_sha256'),
      (OUT/'verification/C_gate.json','C_gate_sha256'),(OUT/'verification/pilot_root_release.json','root_release_sha256')]:
        assert sha(p)==d[keyname],f'protocol drift: {p}'
    for path,digest in d['source_hashes'].items():assert sha(ROOT/path)==digest,path
    for path,digest in d['data_sources'].items():assert sha(ROOT/path)==digest,('fixed data drift',path)
    return d

def initial_observed_truth(seed,budget,allow):
    """Read only already observed current-L labels from authorized frozen packets."""
    cp,c=state_contract(seed,budget);inp=js(cp.parent/'input.json')
    assert sha(cp.parent/'input.json')==c.get('input_sha256',js(cp)['input_sha256'])
    assert inp['validation_test_records']==0
    cols=inp['observed']['columns'];ids_idx=cols.index('id');v1=cols.index('V1_ml');v2=cols.index('V2_ml')
    rows=inp['observed']['rows'];expected_pre=set(js(cp)['L_t_ids']);allowed=set(allow);observed={}
    assert expected_pre<=allowed
    for r in rows:
        rid=str(r[ids_idx])
        assert rid in expected_pre and rid in allowed and rid not in observed,'non-authorized or duplicate observed ID'
        observed[rid]=(float(r[v1]),float(r[v2]))
    assert set(observed)==expected_pre
    if c.get('derived_terminal_state'):
        feedback=js(cp.parent/'feedback.json')
        # Field structure is asserted explicitly below; no raw source fallback.
        frecs=feedback.get('records',feedback.get('observed',feedback.get('outcomes')))
        if frecs is None:frecs=feedback.get('selected_outcomes')
        if frecs is None:raise ValueError('Unsupported accepted feedback schema; inspect metadata, do not read another label source')
        for r in frecs:
            rid=str(r.get('id',r.get('sample_id',r.get('candidate_id'))))
            assert rid in allowed and rid not in observed,'non-authorized or duplicate accepted feedback ID'
            actual=r.get('observed',r.get('actual',r))
            observed[rid]=(float(actual.get('V1_ml',actual.get('true_V1_ml',actual.get('V1')))),float(actual.get('V2_ml',actual.get('true_V2_ml',actual.get('V2')))))
    assert set(observed)==set(allow)
    assert all(math.isfinite(x) for v in observed.values() for x in v)
    return observed

def restricted_validation_truth(ids,roles,source):
    """Only ID-authorized validation target strings are converted into numbers."""
    ids=list(ids);assert len(ids)==416 and len(set(ids))==416
    assert all(roles[i]['role']=='validation' for i in ids)
    wanted=set(ids);found={};parsed=[]
    # Canonical CSV streaming is intentionally not delegated to a whole-frame
    # pandas reader. Non-allowlisted target strings are never numericized.
    SOURCES[str(source.relative_to(ROOT))]={'path':str(source.relative_to(ROOT)),'sha256':sha(source),'bytes':source.stat().st_size,
      'role':'restricted streaming target store; 416 validation IDs only'}
    with source.open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rid=str(row['sample_id'])
            if rid in wanted:
                assert roles[rid]['role']=='validation'
                found[rid]=(float(row['V1_ml']),float(row['V2_ml']));parsed.append(rid)
    assert set(found)==wanted and len(parsed)==416
    return found,{'purpose':'fixed validation checkpointing','allowed_rows':416,'numeric_target_rows_parsed':416,
      'roles':['validation'],'allowed_ids_sha256':stable(sorted(ids)),'test_rows_numericized':0,'unobserved_U_rows_numericized':0}

def interaction_fusion(p,r):
    import torch
    from torch.nn import functional as F
    return p+r+r*torch.tanh(F.layer_norm(p,(p.shape[1],),weight=None,bias=None,eps=1e-5))

def build_pilot_model(norm,arm):
    from qgeognn_al.models import build_predictor,QGeoGNNV2
    from torch_geometric.nn import global_add_pool
    if arm not in ['baseline','interaction']:raise ValueError('Only the two prespecified model arms are allowed')
    class Interaction(QGeoGNNV2):
        def extract_representation(self,a,b):
            p=global_add_pool(self.backbone(a,b),a.batch)
            r,_=self.condition_branch(a)
            return interaction_fusion(p,r)
    model=build_predictor(norm)
    if arm=='interaction':model.__class__=Interaction
    return model

def fit_one(state,arm,rep,design):
    """One scratch fit with the frozen original objective and training settings."""
    import numpy as np
    import torch
    sys.path.insert(0,str(ROOT/'src'))
    from qgeognn_al.schemas.conditions import ConditionNormalization
    from qgeognn_al.training.predictor import seed_everything,loader_pair,predict,target_loss
    from qgeognn_al.evaluation.point import point_metrics
    seed=int(state['seed']);budget=int(state['available_budget']);runtime=OUT/f'runtime/seed_{seed}/L{budget}/{arm}/rep_{rep}'
    runtime.mkdir(parents=True,exist_ok=True);auditpath=runtime/'fit_audit.json'
    init_seed=seed+2000003+rep*1000033;train_seed=seed+3000017+rep*1000033
    contract={'design_sha256':stable(design),'seed':seed,'available_budget':budget,'arm':arm,'replicate':rep,
      'initialization_seed':init_seed,'training_seed':train_seed,'train_ids_sha256':stable(state['train_ids']),
      'H_ids_sha256':stable(state['H_ids']),'actual_train_rows':state['actual_train_rows']}
    if auditpath.exists():
        audit=js(auditpath);assert audit['fit_contract']==contract
        assert sha(runtime/'best.pt')==audit['checkpoint_sha256']
        return audit
    # Any partial run must be inspected; never silently restart/retry a fit.
    if (runtime/'fit_intent.json').exists():raise RuntimeError(f'Incomplete fit requires explicit review: {runtime}')
    write_once(runtime/'fit_intent.json',contract)
    snapshot=json.loads(lzma.decompress(read(SNAP)));roles=snapshot['partitions'][str(seed)]
    contextpath=ROOT/f'studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_{seed}/context.json'
    context=js(contextpath)['contract'];norm=ConditionNormalization(**context['normalization']);prep=context['preprocessing']
    assert context['source_sha256']==design['data_sources']['experiments/e0_4g_baseline/canonical_4g.csv']
    graphpath=contextpath.parent/'scrubbed_graphs.pt'
    SOURCES[str(graphpath.relative_to(ROOT))]={'path':str(graphpath.relative_to(ROOT)),'sha256':sha(graphpath),'bytes':graphpath.stat().st_size,'role':'all labels zero sentinel'}
    atom,angle=torch.load(graphpath,map_location='cpu',weights_only=False)
    assert len(atom)==len(angle)==4163
    assert all(torch.count_nonzero(g.y).item()==0 for g in atom),'graph cache has non-sentinel targets'
    index={i:int(v['canonical_index']) for i,v in roles.items()}
    assert all(int(atom[index[i]].canonical_position.item())==index[i] for i in state['L_ids'])
    truth=initial_observed_truth(seed,budget,state['L_ids'])
    val_ids=sorted(i for i,v in roles.items() if v['role']=='validation')
    val,vreceipt=restricted_validation_truth(val_ids,roles,ROOT/'experiments/e0_4g_baseline/canonical_4g.csv')
    train_ids=state['train_ids'];H=state['H_ids']
    assert not(set(train_ids)&set(H)) and not(set(state['L_ids'])&set(val_ids))
    # Only training and validation graphs receive labels. H stays hidden from
    # all training primitives, including their graph targets.
    working=list(atom)
    for i in train_ids+val_ids:
        pos=index[i];g=atom[pos].clone();g.y=torch.tensor([truth[i] if i in truth else val[i]],dtype=torch.float32);working[pos]=g
    assert all(torch.count_nonzero(working[index[i]].y).item()==0 for i in H)
    assert all(torch.count_nonzero(working[index[i]].y).item()==0 for i,v in roles.items() if v['role']=='test')
    torch.set_num_threads(2);seed_everything(init_seed);model=build_pilot_model(norm,arm)
    assert sum(p.numel() for p in model.parameters())==458952
    init_bytes=b''.join(k.encode()+v.detach().cpu().numpy().tobytes() for k,v in model.state_dict().items())
    init_hash=hashlib.sha256(init_bytes).hexdigest()
    # Both definitions agree when the existing residual is zero initialized.
    assert torch.count_nonzero(model.condition_branch.output.weight).item()==0
    assert torch.count_nonzero(model.condition_branch.output.bias).item()==0
    seed_everything(train_seed)
    cfg=design['training'];optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    train_idx=np.asarray([index[i] for i in train_ids],int);val_idx=np.asarray([index[i] for i in val_ids],int)
    H_idx=np.asarray([index[i] for i in H],int)
    best=float('inf');best_epoch=None;stale=0;history=[]
    import time
    start=time.perf_counter()
    for epoch in range(1,cfg['maximum_epochs']+1):
        model.train();order=np.random.default_rng(train_seed*10000+epoch).permutation(train_idx);losses=[]
        for a,b in zip(*loader_pair(working,angle,order,cfg['batch_size'])):
            pred=model(a,b);loss=target_loss(a.y[:,0],pred[:,:3])+target_loss(a.y[:,1],pred[:,3:])
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss; stop rather than change configuration')
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        vt,vp,_=predict(model,working,angle,val_idx);metrics=point_metrics(vt,vp,prep['target_scales']);score=metrics['combined_normalized_rmse']
        assert math.isfinite(score)
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_selection_score':score,**{f'validation_{k}':v for k,v in metrics.items()}})
        if score<best:
            best=score;best_epoch=epoch;stale=0
            torch.save({'model_state_dict':model.state_dict(),'pilot_arm':arm,'fit_contract':contract,'normalization':context['normalization'],
              'preprocessing':prep,'epoch':epoch,'validation_selection_score':score,'parameter_count':458952},runtime/'best.pt')
        else:stale+=1
        if epoch==1 or epoch%50==0:print(json.dumps({'seed':seed,'budget':budget,'arm':arm,'replicate':rep,'epoch':epoch,'best_epoch':best_epoch,'best_score':best}),flush=True)
        if stale>=cfg['patience']:break
    ck=torch.load(runtime/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model_state_dict']);model.eval()
    _,vp,_=predict(model,working,angle,val_idx);validation_metrics=point_metrics(np.asarray([val[i] for i in val_ids],np.float32),vp,prep['target_scales'])
    _,hp,_=predict(model,atom,angle,H_idx)
    Htruth=np.asarray([truth[i] for i in H],np.float32);Hmetrics=point_metrics(Htruth,hp,prep['target_scales'])
    hpred={i:hp[j] for j,i in enumerate(H)};pairrows=[]
    for group in state['groups']:
        for edge in group['pair_edges']:
            a,b=edge['a'],edge['b']
            axis=edge['axis']
            def axis_value(i):
                value=snapshot['metadata'][i]['conditions'][axis]
                if axis=='PE/EA':
                    x,y=str(value).split('/');return Fraction(Decimal(y))/(Fraction(Decimal(x))+Fraction(Decimal(y)))
                if axis=='loading solvent':return str(value)
                return Decimal(str(value))
            if axis_value(a)>axis_value(b):a,b=b,a
            for t,head in [('V1',1),('V2',4)]:
                k=0 if t=='V1' else 1;dy=truth[b][k]-truth[a][k];dp=float(hpred[b][head]-hpred[a][head])
                pairrows.append({'seed':seed,'available_budget':budget,'arm':arm,'replicate':rep,'group_id':group['group_id'],
                  'molecule_sha256':stable(group['molecule']),'axis':edge['axis'],'target':t,'a':a,'b':b,
                  'observed_difference_ml':dy,'predicted_difference_ml':dp,'signed_difference_error_ml':dp-dy,'absolute_difference_error_ml':abs(dp-dy),
                  'direction_match':int(np.sign(dy)==np.sign(dp)),'observed_abs_difference_ml':abs(dy),'predicted_abs_difference_ml':abs(dp)})
    csvout(runtime/'heldout_pairs.csv',pairrows)
    csvout(runtime/'history.csv',history)
    # Public summaries aggregate within group, avoiding pair multiplicity as
    # an implicit independent-sample count. Raw H pair responses stay local.
    grouped=collections.defaultdict(list)
    for r in pairrows:grouped[r['axis'],r['target'],r['group_id']].append(r)
    grouprows=[]
    for (axis,t,gid),rs in grouped.items():
        grouprows.append({'seed':seed,'available_budget':budget,'arm':arm,'replicate':rep,'group_id':gid,
           'molecule_sha256':rs[0]['molecule_sha256'],'axis':axis,'target':t,'pair_count':len(rs),
           **{k:float(np.mean([r[k] for r in rs])) for k in ['signed_difference_error_ml','absolute_difference_error_ml','direction_match','observed_abs_difference_ml','predicted_abs_difference_ml']}})
    csvout(runtime/'heldout_group_metrics.csv',grouprows)
    receipt={'validation':vreceipt,'training':{'rows':len(train_ids),'ids_sha256':stable(sorted(train_ids)),'source':'same-state observed accepted labels'},
       'heldout':{'rows':len(H),'ids_sha256':stable(sorted(H)),'source':'same-state observed labels','role':'final evaluation only; never checkpoint'},
       'test_numeric_targets_read':0,'unacquired_numeric_targets_read':0,'all_graphs_zero_before_assignment':True,'test_and_H_graphs_zero_during_fit':True}
    audit={'status':'SUCCESS','fit_contract':contract,'initialization_state_sha256':init_hash,'parameter_count':458952,
      'best_epoch':best_epoch,'epochs_run':len(history),'training_seconds':time.perf_counter()-start,'actual_train_rows':len(train_ids),
      'H_rows':len(H),'validation_rows':416,'validation_metrics':validation_metrics,'H_point_metrics':Hmetrics,
      'checkpoint_sha256':sha(runtime/'best.pt'),'history_sha256':sha(runtime/'history.csv'),
      'H_pair_file_sha256':sha(runtime/'heldout_pairs.csv'),'label_access':receipt,'new_AL_selections':0,'paid_LLM_calls':0}
    dump(auditpath,audit)
    return audit

def execute(outer_seed):
    design=check_freeze();q=js(OUT/'verification/pilot_qualification.json')
    assert outer_seed in [157,887],'Worker must own exactly one prespecified outer seed'
    completed=[]
    for state in q['states']:
        if not state['qualification'] or state['seed']!=outer_seed:continue
        for rep in [0,1]:
            for arm in ['baseline','interaction']:
                existing=list((OUT/'runtime').glob('seed_*/L*/*/rep_*/fit_audit.json'))
                current=OUT/f"runtime/seed_{state['seed']}/L{state['available_budget']}/{arm}/rep_{rep}/fit_audit.json"
                assert current.exists() or len(existing)<16,'Successful fit cap reached'
                audit=fit_one(state,arm,rep,design);completed.append(audit)
                assert len(completed)<=8
                dump(OUT/f'verification/pilot_execution_seed_{outer_seed}.json',{'status':'IN_PROGRESS' if len(completed)<8 else 'COMPLETE',
                   'outer_seed':outer_seed,'successful_fits_in_worker':len(completed),'successful_fit_cap_global':16,
                   'last':audit['fit_contract'],'new_LLM_calls':0})
    dump(OUT/f'verification/pilot_training_inputs_seed_{outer_seed}.json',list(SOURCES.values()))
    print('WORKER COMPLETE',outer_seed,'successful fits',len(completed),flush=True)

def summarize_execution():
    design=check_freeze();q=js(OUT/'verification/pilot_qualification.json')
    expected={(s['seed'],s['available_budget'],a,r) for s in q['states'] if s['qualification'] for a in ['baseline','interaction'] for r in [0,1]}
    audit_paths=sorted((OUT/'runtime').glob('seed_*/L*/*/rep_*/fit_audit.json'))
    assert len(audit_paths)<=16,'Global successful-fit limit exceeded'
    completed=[js(p) for p in audit_paths];actual=set()
    for a in completed:
        c=a['fit_contract'];slot=(c['seed'],c['available_budget'],c['arm'],c['replicate'])
        assert slot in expected and slot not in actual and c['design_sha256']==stable(design)
        actual.add(slot)
        rt=OUT/f"runtime/seed_{c['seed']}/L{c['available_budget']}/{c['arm']}/rep_{c['replicate']}"
        assert sha(rt/'best.pt')==a['checkpoint_sha256'] and sha(rt/'history.csv')==a['history_sha256']
        assert sha(rt/'heldout_pairs.csv')==a['H_pair_file_sha256']
        assert a['label_access']['test_numeric_targets_read']==a['label_access']['unacquired_numeric_targets_read']==0
    assert actual==expected,{'missing_successful_slots':sorted(expected-actual)}
    # Verify exact arm initialization pairing and export only aggregate metrics.
    records=[];groups=[]
    for a in completed:
        c=a['fit_contract'];records.append({**c,'best_epoch':a['best_epoch'],'epochs_run':a['epochs_run'],'training_seconds':a['training_seconds'],
          'H_rows':a['H_rows'],'validation_rows':a['validation_rows'],'initialization_sha256':a['initialization_state_sha256'],
          **{f'validation_{k}':v for k,v in a['validation_metrics'].items()},**{f'H_{k}':v for k,v in a['H_point_metrics'].items()}})
        p=OUT/f"runtime/seed_{c['seed']}/L{c['available_budget']}/{c['arm']}/rep_{c['replicate']}/heldout_group_metrics.csv"
        with p.open() as f:groups.extend(list(csv.DictReader(f)))
    for seed in [157,887]:
        for b in [525,1005]:
            for rep in [0,1]:
                x=[r for r in records if r['seed']==seed and r['available_budget']==b and r['replicate']==rep]
                if x:assert len(x)==2 and len({r['initialization_sha256'] for r in x})==1
    csvout(OUT/'tables/pilot_fit_results.csv',records);csvout(OUT/'tables/pilot_H_group_metrics.csv',groups)
    inputs={}
    for seed in [157,887]:
        worker=js(OUT/f'verification/pilot_execution_seed_{seed}.json');assert worker['status']=='COMPLETE' and worker['successful_fits_in_worker']<=8
        for x in js(OUT/f'verification/pilot_training_inputs_seed_{seed}.json'):
            if x['path'] in inputs:assert inputs[x['path']]['sha256']==x['sha256']
            else:inputs[x['path']]=x
    dump(OUT/'verification/pilot_training_inputs.json',list(inputs.values()))
    dump(OUT/'verification/pilot_execution_status.json',{'status':'COMPLETE','successful_fits':len(completed),'successful_fit_cap':16,
      'new_LLM_calls':0,'new_AL_selections':0,'test_truth_reads':0,'unacquired_truth_reads':0,'paired_initialization_verified':True,
      'inputs_frozen_before_fitting':True,'unique_model_changes_tested':1,'validation_used_to_select_architecture':False})
    print('COMPLETE successful fits',len(completed),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['qualify','freeze','execute','summarize']);ap.add_argument('--outer-seed',type=int,choices=[157,887]);args=ap.parse_args()
    if args.phase=='qualify':metadata_qualification()
    elif args.phase=='freeze':freeze_protocol()
    elif args.phase=='execute':execute(args.outer_seed)
    elif args.phase=='summarize':summarize_execution()
if __name__=='__main__':main()
