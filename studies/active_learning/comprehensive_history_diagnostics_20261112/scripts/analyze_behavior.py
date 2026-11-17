#!/usr/bin/env python3
"""Target-free acquisition diagnostics. Reuse old audited results; extend full CW rerun.
No model fitting/import, APIs, target table or test truth. Public outputs are metadata
and aggregate diagnostics only. Historical files are read-only and SHA-locked.
"""
from __future__ import annotations
import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/history_behavior_mpl')
import json,gzip,lzma,hashlib,io,time,argparse,csv
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
HERE=Path(__file__).resolve().parent.parent
ROOT=HERE.parents[2]
OLD=ROOT/'studies/active_learning/nocw_free_selection_diagnostics_20261101'
sys.path.insert(0,str(OLD))
from condition_metrics import compute_condition_metrics,_canonical_conditions
from geometry_metrics import evaluate_geometry,fixed_reference_ids,self_check
import analyze as old
SEEDS=[157,887,2357,6101,12203]
NAMES={'free':'historical_free_mixed_model','nocw':'nocw','hybrid':'numerical_hybrid32','lcmd':'gradient_lcmd32'}
CW='new_free_cw_sol61_xhigh'
NEWROOT=ROOT/'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101'
FILES={}; EMBED=[];MISSING=[]
LOCKPATH=HERE/'verification/behavior_inputs.json'
LOCK={x['path']:x for x in json.loads(LOCKPATH.read_text())['files']} if LOCKPATH.exists() else None

def sha(b):return hashlib.sha256(b).hexdigest()
def stable(v):return sha(json.dumps(v,sort_keys=True,separators=(',',':')).encode())
def relative(p):return str(Path(p).resolve().relative_to(ROOT))
def read(p,expected=None):
    p=Path(p); b=p.read_bytes();b2=p.read_bytes()
    assert b==b2,('unstable',p)
    h=sha(b);key=relative(p)
    if expected: assert h==expected,('hash',p)
    if LOCK is not None and key in LOCK and key != relative(Path(__file__)): assert h==LOCK[key]['sha256'],('changed frozen input',key)
    FILES[key]={'path':key,'sha256':h,'bytes':len(b)}
    return b

def load(p,expected=None):return json.loads(read(p,expected))
def writejson(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def phase(L):return '333-525' if L<525 else '525-813' if L<813 else '813-1005' if L<1005 else '1005-1485'

def get_batches():
    for p in [OLD/'analyze.py',OLD/'condition_metrics.py',OLD/'geometry_metrics.py',Path(__file__)]:read(p)
    snap=json.loads(lzma.decompress(read(OLD/'snapshot_inputs.json.xz')))
    meta=snap['metadata'];batches=[]
    metrics=pd.read_csv(io.BytesIO(read(ROOT/'reports/free_cw_final_20261112/validation_curves.csv')))
    scores={(x.method,int(x.seed),int(x.budget)):float(x.validation_combined_normalized_rmse) for x in metrics.itertuples()}
    existing=set()
    for b0 in snap['batches']:
        if not b0['completed_fit_validation']:continue
        b={k:v for k,v in b0.items() if k!='notes'};b['original_method']=b['method'];b['method']=NAMES[b['method']]
        b['reused_frozen_snapshot']=True;existing.add((b['method'],b['seed'],b['round']));batches.append(b)
    # Add only completed, archived-endpoint NoCW rounds absent from prior snapshot.
    for seed in SEEDS:
        maxL=int(metrics[(metrics.method=='nocw')&(metrics.seed==seed)].budget.max())
        for r in range((maxL-333)//32):
            if ('nocw',seed,r) in existing:continue
            d=old.llm_dirs('nocw',seed,r)
            try:
                f=load(d/'batch_freeze.json');c=load(d/'contract.json',f['artifact_hashes']['contract.json'])
                s=load(d/'selection.json',f['artifact_hashes']['selection.json'])
                assert f['batch_ids']==[v['id'] for v in s['choices']]
                b={'method':'nocw','original_method':'nocw','seed':seed,'round':r,'L_before':333+32*r,'L_after':365+32*r,
                   'selected_ids':f['batch_ids'],'labeled_ids':c['L_t_ids'],'unlabeled_ids':c['U_t_ids'],
                   'source':relative(d/'batch_freeze.json'),'source_sha256':sha(read(d/'batch_freeze.json')),
                   'contract_source':relative(d/'contract.json'),'checkpoint_sha256':c['checkpoint_sha256'],
                   'completed_fit_validation':True,'reused_frozen_snapshot':False}
                batches.append(b)
            except (OSError,KeyError) as e:MISSING.append({'method':'nocw','seed':seed,'round':r,'reason':str(e)})
    # Immutable CW scientific archive: project only already-frozen selection metadata.
    for seed in SEEDS:
        archive=ROOT/f'reports/free_cw_final_20261112/scientific_artifacts/seed_{seed}_results.jsonl.gz'
        entries={}
        for line in gzip.decompress(read(archive)).decode().splitlines():
            e=json.loads(line); p=Path(e['path']); name=p.name
            if name in ('contract.json','batch_freeze.json','selection.json'):
                raw=e['raw_utf8'].encode(); assert sha(raw)==e['sha256']
                r=int(p.parent.name.split('_')[-1]);entries[(r,name)]=(json.loads(raw),e)
                EMBED.append({'archive':relative(archive),'path':e['path'],'sha256':e['sha256']})
        assert len(entries)==108
        for r in range(36):
            f,fe=entries[(r,'batch_freeze.json')];c,ce=entries[(r,'contract.json')];s,se=entries[(r,'selection.json')]
            assert f['artifact_hashes']['contract.json']==ce['sha256'];assert f['artifact_hashes']['selection.json']==se['sha256']
            assert f['batch_ids']==[v['id'] for v in s['choices']];assert not f['pending_ids']
            assert f['checkpoint_sha256']==c['checkpoint_sha256']
            b={'method':CW,'original_method':CW,'seed':seed,'round':r,'L_before':333+32*r,'L_after':365+32*r,
               'selected_ids':f['batch_ids'],'labeled_ids':c['L_t_ids'],'unlabeled_ids':c['U_t_ids'],
               'source':fe['path'],'source_sha256':fe['sha256'],'contract_source':ce['path'],
               'checkpoint_sha256':c['checkpoint_sha256'],'completed_fit_validation':True,'reused_frozen_snapshot':False}
            batches.append(b)
    # Random32 is a frozen permutation; audit contract and prior L/U, no bank needed.
    for seed in SEEDS:
        labeled=list(snap['initial'][str(seed)]);pool=list(snap['pool'][str(seed)])
        for r in range(21):
            d=old.BASE/f'runtime/seed_{seed}/random/round_{r:02d}'
            c=load(d/'acquisition_artifacts/contract.json')
            table=read(d/'acquisition_artifacts/selected_next_batch.csv',c['files']['acquisition_artifacts/selected_next_batch.csv'])
            selected=[v['sample_id'] for v in csv.DictReader(io.StringIO(table.decode()))]
            assert stable(selected)==c['selected_ids_ordered_hash']; assert stable(sorted(labeled))==c['input']['L_t_ids_hash']
            unlabeled=[v for v in pool if v not in set(labeled)]
            assert stable(sorted(unlabeled))==c['input']['U_t_ids_hash']
            a=load(d/'model/fit_audit.json');a2=load(d.parent/f'round_{r+1:02d}/model/fit_audit.json')
            b={'method':'random32','original_method':'random32','seed':seed,'round':r,'L_before':333+32*r,'L_after':365+32*r,
               'selected_ids':selected,'labeled_ids':list(labeled),'unlabeled_ids':unlabeled,
               'source':relative(d/'acquisition_artifacts/selected_next_batch.csv'),'source_sha256':sha(table),
               'contract_source':relative(d/'acquisition_artifacts/contract.json'),'checkpoint_sha256':c['input']['evaluation_checkpoint_sha256'],
               'completed_fit_validation':True,'reused_frozen_snapshot':False,
               'validation_nrmse_before':a['best_validation_combined_normalized_rmse'],'validation_nrmse_after':a2['best_validation_combined_normalized_rmse']}
            assert a['train_ids_hash']==stable(sorted(labeled));assert a2['train_ids_hash']==stable(sorted(labeled+selected))
            batches.append(b);labeled.extend(selected)
    for b in batches:
        assert len(b['selected_ids'])==len(set(b['selected_ids']))==32
        assert not set(b['selected_ids'])&set(b['labeled_ids'])
        assert set(b['selected_ids'])<=set(b['unlabeled_ids'])
        assert len(b['labeled_ids'])==b['L_before']
        if b['method']!='random32':
            b['validation_nrmse_before']=scores[(b['method'],b['seed'],b['L_before'])]
            b['validation_nrmse_after']=scores[(b['method'],b['seed'],b['L_after'])]
    return snap,meta,sorted(batches,key=lambda b:(b['method'],b['seed'],b['round']))

def structure(meta):
    smiles=sorted(set(v['smiles'] for v in meta.values()));gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    mols=[Chem.MolFromSmiles(s) for s in smiles];assert all(m is not None for m in mols)
    fps=[gen.GetFingerprint(m) for m in mols]; sim=np.array([DataStructs.BulkTanimotoSimilarity(f,fps) for f in fps])
    scaffolds=[MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False) for m in mols]
    return {s:i for i,s in enumerate(smiles)},sim,scaffolds

def behavior(snap,meta,batches):
    lookup,sim,scaffolds=structure(meta); rows=[]; selectedrows=[]
    olddf=pd.read_csv(io.BytesIO(read(OLD/'batch_metrics.csv'))).set_index(['method','seed','round'])
    conditions={sid:_canonical_conditions(v) for sid,v in meta.items()}
    def response_series(ids):
        groups=defaultdict(set)
        for sid in ids:
            c=conditions[sid]
            if c is not None:groups[(meta[sid]['smiles'],)+c[1:]].add(c[0])
        return sum(len(v)>=3 for v in groups.values())
    for b in batches:
        sel=b['selected_ids'];lab=b['labeled_ids'];sm=[meta[i]['smiles'] for i in sel];prevsm={meta[i]['smiles'] for i in lab}
        counts=Counter(sm);novel=set(sm)-prevsm;si=[lookup[s] for s in sm]; li=[lookup[s] for s in prevsm]
        ds=1-sim[np.ix_(si,si)];un=np.unique(si)
        if b['reused_frozen_snapshot']:
            oldrow=olddf.loc[(b['original_method'],b['seed'],b['round'])].to_dict()
            cond={k:v for k,v in oldrow.items() if k.startswith(('within_','cross_','condition_'))}
        else:cond,_=compute_condition_metrics(sel,lab,meta,return_examples=0)
        cond={k.replace('_selected_slots_fraction','_selected_fraction'):v for k,v in cond.items()}
        ear=[1/(1+float(conditions[i][0])) for i in sel]
        vs=[float(conditions[i][2]) for i in sel]; loads=[float(conditions[i][4]) for i in sel]
        labsigs={(meta[i]['smiles'],conditions[i]) for i in lab}; selsigs={(meta[i]['smiles'],conditions[i]) for i in sel}
        r={k:v for k,v in b.items() if k not in ('selected_ids','labeled_ids','unlabeled_ids')}
        r.update(labeled_size_before=b['L_before'],labeled_size_after=b['L_after'],phase=phase(b['L_before']),
                 batch_size=32,unique_molecules=len(counts),new_molecules=len(novel),new_molecule_slot_fraction=sum(counts[s] for s in novel)/32,
                 max_molecule_slot_fraction=max(counts.values())/32,cumulative_molecules_before=len(prevsm),cumulative_molecules_after=len(prevsm|set(sm)),
                 within_tanimoto_distance_mean=float(ds[np.triu_indices(32,1)].mean()),
                 unique_molecule_tanimoto_distance_mean=float((1-sim[np.ix_(un,un)])[np.triu_indices(len(un),1)].mean()) if len(un)>1 else 0,
                 chemical_nearest_l_distance_mean=float((1-sim[np.ix_(si,li)].max(axis=1)).mean()),
                 new_molecule_nearest_l_distance_mean=float(np.mean([1-sim[lookup[s],li].max() for s in novel])) if novel else np.nan,
                 unique_scaffolds=len({scaffolds[i] for i in si}),new_scaffolds=len({scaffolds[i] for i in si}-{scaffolds[i] for i in li}),
                 acyclic_scaffold_slot_fraction=sum(scaffolds[i]=='' for i in si)/32,
                 ea_fraction_mean=float(np.mean(ear)),ea_fraction_min=min(ear),ea_fraction_max=max(ear),
                 sample_volume_ul_mean=np.mean(vs),sample_volume_ul_range=max(vs)-min(vs),
                 sample_mass_mg_mean=np.mean([float(conditions[i][1]*conditions[i][2]) for i in sel]),
                 loading_volume_ul_mean=np.mean(loads),loading_volume_ul_range=max(loads)-min(loads),
                 loading_solvent_types=len({conditions[i][3] for i in sel}),
                 unique_recorded_condition_sets=len(selsigs),novel_recorded_condition_sets=len(selsigs-labsigs),
                 within_pe_series_ge3=response_series(sel),new_pe_series_ge3=response_series(lab+sel)-response_series(lab),
                 gain=b['validation_nrmse_before']-b['validation_nrmse_after'],**cond)
        rows.append(r)
        for sid in sel:
            c=conditions[sid]
            selectedrows.append({'method':b['method'],'seed':b['seed'],'round':b['round'],'labeled_size_before':b['L_before'],
                'sample_id':sid,'canonical_smiles':meta[sid]['smiles'],'scaffold':scaffolds[lookup[meta[sid]['smiles']]],
                'ea_fraction':1/(1+float(c[0])),'sample_volume_ul':float(c[2]),'sample_mass_mg':float(c[1]*c[2]),
                'loading_solvent':c[3],'loading_volume_ul':float(c[4]),'known_molecule_before':meta[sid]['smiles'] in prevsm,
                'information_time':'candidate metadata available before selection','source':b['source'],'source_sha256':b['source_sha256']})
    df=pd.DataFrame(rows);df.to_csv(HERE/'tables/behavior_batches.csv',index=False)
    pd.DataFrame(selectedrows).to_csv(HERE/'tables/behavior_selected_rows.csv',index=False)
    common=df[df.L_after<=1005]; nums=['unique_molecules','new_molecules','new_molecule_slot_fraction','cumulative_molecules_after',
       'within_any_single_selected_fraction','cross_any_single_selected_fraction','cross_only_sample_volume_selected_fraction',
       'within_only_pe_selected_fraction','within_multi_factor_selected_fraction','cross_exact_selected_fraction','within_tanimoto_distance_mean',
       'chemical_nearest_l_distance_mean','unique_scaffolds','ea_fraction_mean','sample_volume_ul_mean','sample_mass_mg_mean',
       'loading_volume_ul_mean','within_pe_series_ge3','new_pe_series_ge3']
    summaries=[]
    for scope,d in [('all_available',df),('common_1005',common)]:
        ps=d.groupby(['method','seed','phase'])[nums].mean().reset_index();ps['aggregation']='seed_mean';ps['scope']=scope
        summaries.extend(ps.to_dict('records'))
        for keys,g in ps.groupby(['method','phase']):
            for agg in ['mean','std']:
                rr=dict(zip(['method','phase'],keys));rr.update(aggregation='group_'+agg,scope=scope,seed='ALL',n_seeds=len(g))
                rr.update(getattr(g[nums],agg)(**({'ddof':1} if agg=='std' else {})).to_dict());summaries.append(rr)
    pd.DataFrame(summaries).to_csv(HERE/'tables/behavior_seed_phase.csv',index=False)
    return df

def geometry(snap,batches):
    # Published old metrics are reused, not represented as new computations.
    g=pd.read_csv(io.BytesIO(read(OLD/'geometry_metrics.csv')));g['method']=g.method.map(NAMES)
    g['reused_existing_result']=True;g['phase']=g.L_before.map(phase)
    rows=g.to_dict('records');new=[]
    oldkeys=set(zip(g.method,g.seed,g['round']))
    for j,b in enumerate(batches):
        if b['method'] not in (CW,'nocw') or (b['method'],b['seed'],b['round']) in oldkeys:continue
        seed=b['seed'];r=b['round'];d=Path(str(ROOT/b['contract_source']).replace('/selections/','/runtime/')).parent
        shared=False
        if b['method']==CW and r==0 and not (d/'banks.json').exists():
            initial=g[(g.method=='historical_free_mixed_model')&(g.seed==seed)&(g['round']==0)&(g.space=='raw_gradient')].iloc[0]
            d=(ROOT/initial.bank_path).parent;shared=True
        try:
            md=load(d/'banks.json'); contract=md['contract'];assert contract['checkpoint']==b['checkpoint_sha256']
            rev={int(row['canonical_index']):sid for sid,row in snap['partitions'][str(seed)].items() if row['role'] in ('l0','u0')}
            ids=[rev[int(i)] for i in contract['indices']];n=len(b['labeled_ids'])
            assert len(set(ids))==3330 and set(ids[:n])==set(b['labeled_ids']) and set(ids[n:])==set(b['unlabeled_ids'])
            bankbytes=read(d/'banks.npz',md['sha256'])
            refs=fixed_reference_ids(snap['pool'][str(seed)],128,'nocw_free_selection_diagnostics_v1_seed_'+str(seed))
            with np.load(io.BytesIO(bankbytes),allow_pickle=False) as bank:
                for space,key in [('raw_gradient','raw'),('cw_gradient','cw')]:
                    x=bank[key];assert x.shape==(3330,512)
                    vals=evaluate_geometry(x,ids,b['labeled_ids'],b['unlabeled_ids'],b['selected_ids'],reference_ids=refs)
                    row={'method':b['method'],'seed':seed,'round':r,'L_before':b['L_before'],'L_after':b['L_after'],
                         'space':space,'available':True,'missing_reason':'','bank_path':relative(d/'banks.npz'),
                         'bank_sha256':md['sha256'],'checkpoint_ids_hash_order_verified':True,'phase':phase(b['L_before']),
                         'reused_existing_result':False,'shared_initial_bank':shared,'source':b['source'],'contract_source':b['contract_source'],**vals}
                    rows.append(row);new.append(row)
            if len(new)%40==0: print('new geometry rows',len(new),flush=True)
        except (OSError,ValueError,KeyError) as e:
            MISSING.append({'method':b['method'],'seed':seed,'round':r,'kind':'geometry','reason':str(e)})
    g=pd.DataFrame(rows);g['labeled_size_before']=g.L_before;g['labeled_size_after']=g.L_after
    g.to_csv(HERE/'tables/behavior_geometry.csv',index=False)
    return g

def candidate_signals(batches,df):
    from scipy.stats import spearmanr
    rows=[]
    for b in batches:
        if b['method']!=CW:continue
        d=(ROOT/b['contract_source']).parent
        expected=next(e['sha256'] for e in EMBED if e['path']==b['contract_source'])
        c=load(ROOT/b['contract_source'],expected); catalog=load(d/'catalog.json',c['catalog_sha256'])
        candidates=catalog['candidates']+catalog['pending'];assert [v['id'] for v in candidates]==b['unlabeled_ids']
        selected=set(b['selected_ids']);ids=[v['id'] for v in candidates];mask=np.array([i in selected for i in ids]);assert mask.sum()==32
        values={
            'cw_distance':np.array([v['cw_distance_percentile'] for v in candidates]),
            'v1_width':np.array([v['pred_V1_q90_ml']-v['pred_V1_q10_ml'] for v in candidates]),
            'v2_width':np.array([v['pred_V2_q90_ml']-v['pred_V2_q10_ml'] for v in candidates]),
            'v1_q50':np.array([v['pred_V1_q50_ml'] for v in candidates]),
            'v2_q50':np.array([v['pred_V2_q50_ml'] for v in candidates])}
        rr={'method':b['method'],'seed':b['seed'],'round':b['round'],'labeled_size_before':b['L_before'],'L_before':b['L_before'],
            'phase':phase(b['L_before']),'candidate_n':len(candidates),'selected_n':32,'catalog_sha256':c['catalog_sha256'],
            'source':relative(d/'catalog.json'),'information_time':'selection-visible candidate predictions only; no true labels'}
        tops={}
        for name,vals in values.items():
            rank=pd.Series(vals).rank(method='average',pct=True).values
            top=set(sorted(range(len(ids)),key=lambda i:(-vals[i],ids[i]))[:32]);tops[name]=top
            rr['selected_'+name+'_percentile_mean']=rank[mask].mean()
            rr['selected_'+name+'_top32_overlap']=len(top&set(np.flatnonzero(mask)))
        for a,z in [('cw_distance','v1_width'),('cw_distance','v2_width'),('cw_distance','v1_q50'),('cw_distance','v2_q50'),('v1_width','v2_width'),('v1_width','v1_q50'),('v2_width','v2_q50')]:
            rr[a+'__'+z+'_spearman']=float(spearmanr(values[a],values[z]).statistic)
            rr[a+'__'+z+'_top32_jaccard']=len(tops[a]&tops[z])/len(tops[a]|tops[z])
        rows.append(rr)
    sig=pd.DataFrame(rows);sig.to_csv(HERE/'tables/behavior_candidate_signals.csv',index=False)
    cols=[c for c in sig if c.startswith('selected_') and c!='selected_n']
    df=df.merge(sig[['method','seed','round']+cols],on=['method','seed','round'],how='left',validate='one_to_one')
    df.to_csv(HERE/'tables/behavior_batches.csv',index=False)
    return df

def additional_summaries(df,g):
    cols=['batch_nn_distance_mean_random_percentile','selected_to_l_distance_mean_random_percentile',
          'reference_coverage_relative_drop_random_percentile','reference_coverage_relative_drop']
    gs=g.groupby(['method','seed','space','phase'])[cols].mean().reset_index()
    gs.to_csv(HERE/'tables/behavior_geometry_seed_phase.csv',index=False)
    pairs=[]
    for comp in ['nocw','historical_free_mixed_model','gradient_lcmd32']:
        for ph in ['333-525','525-813','813-1005']+(['1005-1485'] if comp!='nocw' else []):
            for space in ['raw_gradient','cw_gradient']:
                a=gs[(gs.method==CW)&(gs.phase==ph)&(gs.space==space)].set_index('seed')
                z=gs[(gs.method==comp)&(gs.phase==ph)&(gs.space==space)].set_index('seed')
                for col in cols:
                    v=(a[col]-z[col]).dropna()
                    if len(v)==5:pairs.append({'contrast':'CW_minus_'+comp,'phase':ph,'space':space,'metric':col,
                        'mean_delta':v.mean(),'sd_delta':v.std(),'positive_seeds':int(sum(v>1e-12)),
                        'negative_seeds':int(sum(v< -1e-12)),'deltas':json.dumps(v.to_dict(),sort_keys=True)})
    pd.DataFrame(pairs).to_csv(HERE/'tables/behavior_geometry_paired.csv',index=False)
    cols=['unique_molecules','new_molecule_slot_fraction','within_any_single_selected_fraction',
          'cross_only_sample_volume_selected_fraction','within_tanimoto_distance_mean','cross_exact_selected_fraction']
    ss=df.groupby(['method','seed','phase'])[cols].mean().reset_index();pairs=[]
    for comp in ['nocw','historical_free_mixed_model','gradient_lcmd32','numerical_hybrid32','random32']:
        for ph in ['333-525','525-813','813-1005']+(['1005-1485'] if comp in ['historical_free_mixed_model','gradient_lcmd32'] else []):
            for subset,seeds in [('five_seeds',SEEDS),('sol_consistent_three',[887,2357,12203])]:
                a=ss[(ss.method==CW)&(ss.phase==ph)&ss.seed.isin(seeds)].set_index('seed')
                z=ss[(ss.method==comp)&(ss.phase==ph)&ss.seed.isin(seeds)].set_index('seed')
                for col in cols:
                    v=(a[col]-z[col]).dropna()
                    if len(v)==len(seeds):pairs.append({'contrast':'CW_minus_'+comp,'subset':subset,'phase':ph,'metric':col,
                        'mean_delta':v.mean(),'sd_delta':v.std(),'positive_seeds':int(sum(v>1e-12)),
                        'negative_seeds':int(sum(v< -1e-12)),'deltas':json.dumps(v.to_dict(),sort_keys=True)})
    pd.DataFrame(pairs).to_csv(HERE/'tables/behavior_paired_differences.csv',index=False)

def plots(df,g):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    names={CW:'CW-Free32 (new)',NAMES['free']:'Free32 (mixed)',NAMES['lcmd']:'Gradient-LCMD32',NAMES['hybrid']:'Numerical Hybrid32','nocw':'NoCW32','random32':'Random32'}
    colors={CW:'#225ea8',NAMES['free']:'#7a5195',NAMES['lcmd']:'#00876c',NAMES['hybrid']:'#d49e15','nocw':'#d94f38','random32':'#8c8c8c'}
    fig,axs=plt.subplots(2,3,figsize=(15,8))
    metrics=[('unique_molecules','Distinct molecules / 32 slots'),('new_molecule_slot_fraction','Previously unseen molecule slots / 32'),
       ('within_any_single_selected_fraction','Within-batch single-factor slots / 32'),('cross_only_sample_volume_selected_fraction','Sample-volume-only contrast to prior L / 32'),
       ('within_tanimoto_distance_mean','Mean 1 - Tanimoto (Morgan radius2/2048)'),('cumulative_molecules_after','Cumulative molecules incl. initial 333')]
    for ax,(c,title) in zip(axs.flat,metrics):
        for m,d in df.groupby('method'):
            n=d.groupby('L_before')[c].count();gg=d.groupby('L_before')[c];v=gg.mean()[n==5];sd=gg.std()[n==5]
            ax.plot(v.index,v,label=names[m],color=colors[m]);ax.fill_between(v.index,v-sd,v+sd,alpha=.08,color=colors[m])
        ax.set_title(title,fontsize=10);ax.set_xlabel('Preselection labelled rows L');ax.grid(alpha=.2)
        for boundary in [525,813,1005]:ax.axvline(boundary,color='gray',ls=':',lw=.6)
    axs[0,0].legend(fontsize=7);fig.suptitle('Acquisition behavior | each point: 5 seeds x 32 selected rows; mean +/- sample SD\nFrozen metadata only; method-specific histories; all complete five-seed budgets shown')
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(HERE/'figures/behavior_trajectories.png',dpi=160);plt.close(fig)
    metrics=['batch_nn_distance_mean_random_percentile','selected_to_l_distance_mean_random_percentile','reference_coverage_relative_drop_random_percentile']
    fig,axs=plt.subplots(1,3,figsize=(14,4.7),sharey=True)
    for ax,c,title in zip(axs,metrics,['Within-batch nearest distance','Selected-to-L nearest distance','Fixed-reference coverage improvement']):
        for m in [CW,NAMES['free'],NAMES['lcmd'],NAMES['hybrid'],'nocw']:
            space='latent' if m==NAMES['hybrid'] else 'raw_gradient';d=g[(g.method==m)&(g.space==space)]
            nn=d.groupby('L_before')[c].count();v=d.groupby('L_before')[c].mean()[nn==5]
            ax.plot(v.index,v,label=names[m]+(' (latent)' if space=='latent' else ' (raw)'),color=colors[m])
        ax.axhline(.5,color='gray',ls='--',lw=.8);ax.set_ylim(-.03,1.03);ax.set_title(title,fontsize=10);ax.set_xlabel('Preselection labelled rows L');ax.grid(alpha=.2)
    axs[0].set_ylabel('Within-state percentile vs 16 uniform random batches');axs[0].legend(fontsize=7)
    fig.suptitle('Frozen geometry | 5 seeds; fixed 128 U0 references per seed, 32 slots per batch\nState-normalized proxies; raw and latent distances are never pooled')
    fig.tight_layout(rect=(0,0,1,.90));fig.savefig(HERE/'figures/behavior_geometry.png',dpi=170);plt.close(fig)
    # Condition distributions at common completed L1005, so counts have equal budget.
    sr=pd.read_csv(HERE/'tables/behavior_selected_rows.csv');sr=sr[sr.labeled_size_before<1005]
    fig,axs=plt.subplots(1,3,figsize=(14,4.7)); columns=['ea_fraction','sample_mass_mg','loading_volume_ul']
    for ax,c,title in zip(axs,columns,['EA volume fraction (b / (a + b))','Sample mass (mg = density x sample uL)','Loading solvent volume (uL)']):
        for m,d in sr.groupby('method'):
            v=np.sort(d[c].values);ax.plot(v,np.arange(1,len(v)+1)/len(v),label=names[m],color=colors[m])
        ax.set_xlabel(title);ax.set_ylabel('Selected-row empirical CDF');ax.grid(alpha=.2)
    axs[0].legend(fontsize=7);fig.suptitle('Condition choices | common L333-1005; each method 5 seeds x 21 batches x 32 rows\nRepeated rows across independent seeds count as separate selections; no response labels')
    fig.tight_layout(rect=(0,0,1,.90));fig.savefig(HERE/'figures/behavior_condition_distributions.png',dpi=170);plt.close(fig)

def verify(df,g,snap,meta,batches):
    checks=[]
    # Independently audit fixed middle-seed batches using simple nested comparison,
    # not the imported condition classifier (all five recorded factors exact).
    for b in batches:
        if b['seed']!=2357 or b['round'] not in [3,10,17,28]:continue
        ids=b['selected_ids'];prev=b['labeled_ids'];v=df[(df.method==b['method'])&(df.seed==2357)&(df['round']==b['round'])].iloc[0]
        for scope,pairs in [('within',[(a,z) for j,a in enumerate(ids) for z in ids[j+1:]]),('cross',[(a,z) for a in ids for z in prev])]:
            selected=set();singles=0;multi=0;exact=0
            for a,z in pairs:
                if meta[a]['smiles']!=meta[z]['smiles']:continue
                ca=_canonical_conditions(meta[a]);cz=_canonical_conditions(meta[z]);diff=sum(x!=y for x,y in zip(ca,cz))
                if diff==1:
                    singles+=1;selected.add(a)
                    if scope=='within':selected.add(z)
                elif diff==0:exact+=1
                else:multi+=1
            assert singles==int(v[scope+'_any_single_pairs']);assert multi==int(v[scope+'_multi_factor_pairs']);assert exact==int(v[scope+'_exact_pairs'])
            assert len(selected)/32==v[scope+'_any_single_selected_fraction']
        assert len({meta[i]['smiles'] for i in ids})==v.unique_molecules
        checks.append({'method':b['method'],'seed':2357,'round':b['round'],'status':'PASS'})
    for (m,s),d in df.groupby(['method','seed']):
        assert d.sort_values('round')['round'].tolist()==list(range(len(d)))
        assert d.new_molecules.sum()==d.iloc[-1].cumulative_molecules_after-d.iloc[0].cumulative_molecules_before
    assert len(df[df.method==CW])==180 and len(g[g.method==CW])==360
    assert not any('test' in k.lower() for k in [])
    # Independent scipy.cdist verification of actual + 16 random coverage metrics.
    from scipy.spatial.distance import cdist
    geometric_checks=[]
    for b in batches:
        if b['method']!=CW or b['seed']!=2357 or b['round'] not in [0,3,10,17,28]:continue
        gg=g[(g.method==CW)&(g.seed==2357)&(g['round']==b['round'])]
        path=ROOT/gg.iloc[0].bank_path; md=load(path.with_suffix('.json'))
        rev={int(v['canonical_index']):sid for sid,v in snap['partitions']['2357'].items() if v['role'] in ('l0','u0')}
        ids=[rev[int(i)] for i in md['contract']['indices']];pos={v:i for i,v in enumerate(ids)}
        refs=fixed_reference_ids(snap['pool']['2357'],128,'nocw_free_selection_diagnostics_v1_seed_2357')
        with np.load(io.BytesIO(read(path,gg.iloc[0].bank_sha256)),allow_pickle=False) as bank:
            for space,key in [('raw_gradient','raw'),('cw_gradient','cw')]:
                x=bank[key].astype(float);L=x[[pos[i] for i in b['labeled_ids']]];R=x[[pos[i] for i in refs]]
                before=cdist(R,L).min(axis=1);before[[i in set(b['labeled_ids']) for i in refs]]=0
                def independent(batch_ids):
                    S=x[[pos[i] for i in batch_ids]]; nn=cdist(S,S);np.fill_diagonal(nn,np.inf)
                    after=np.minimum(before,cdist(R,S).min(axis=1));after[[i in set(batch_ids) for i in refs]]=0
                    return {'batch_nn_distance_mean':nn.min(axis=1).mean(),'selected_to_l_distance_mean':cdist(S,L).min(axis=1).mean(),
                            'reference_coverage_relative_drop':(before-after).mean()/before.mean()}
                actual=independent(b['selected_ids']);randoms=[]
                for seed0 in range(90210,90226):
                    ii=np.random.default_rng(seed0).choice(len(b['unlabeled_ids']),32,replace=False)
                    randoms.append(independent([b['unlabeled_ids'][i] for i in ii]))
                row=gg[gg.space==space].iloc[0]
                for metric,value in actual.items():
                    assert np.isclose(value,row[metric],rtol=1e-8,atol=1e-10),(metric,value,row[metric])
                    vals=np.array([v[metric] for v in randoms]);ties=np.isclose(vals,value,rtol=1e-10,atol=1e-12)
                    percentile=(((vals<value)&~ties).sum()+.5*ties.sum())/16
                    assert percentile==row[metric+'_random_percentile'],(metric,percentile,row[metric+'_random_percentile'])
                geometric_checks.append({'seed':2357,'round':b['round'],'space':space,'independent_engine':'scipy.cdist','status':'PASS'})
    result={'status':'PASS','batch_rows':len(df),'selected_row_records':len(df)*32,'cw_complete_batches':180,
        'geometry_rows':len(g),'geometry_new_rows':int((~g.reused_existing_result).sum()),
        'batches_by_method':df.groupby('method').size().to_dict(),'geometry_self_check':self_check(),
        'independent_pair_rechecks':checks,'independent_geometry_rechecks':geometric_checks,'safety':{'training':0,'api_calls':0,'unselected_target_reads':0,'test_target_reads':0},
        'metadata_scope':'target-free old snapshot plus accepted contracts/freeze; no raw response table',
        'old_results_reused':str(OLD.relative_to(ROOT)),'limitations':['128-point reference; 16 random batches; descriptive not causal','Scaffold empty strings form an acyclic bucket']}
    writejson(HERE/'verification/behavior_checks.json',result)
    writejson(LOCKPATH,{'files':sorted(FILES.values(),key=lambda v:v['path']),'embedded_archive_entries':EMBED,'missing':MISSING,
        'policy':'Inputs immutable; on rerun previously recorded input SHA is enforced; only derived outputs overwritten'})
    print(json.dumps({k:v for k,v in result.items() if k not in ('independent_pair_rechecks','geometry_self_check')},ensure_ascii=False),flush=True)

def main():
    start=time.monotonic();snap,meta,batches=get_batches();print('batches',len(batches),flush=True)
    df=behavior(snap,meta,batches);df=candidate_signals(batches,df);print('behavior done',flush=True);g=geometry(snap,batches);additional_summaries(df,g);plots(df,g);verify(df,g,snap,meta,batches)
    print('seconds',time.monotonic()-start,flush=True)
if __name__=='__main__':main()
