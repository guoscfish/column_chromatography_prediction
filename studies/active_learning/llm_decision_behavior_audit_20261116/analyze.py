#!/usr/bin/env python3
"""Compute per-state metrics from a frozen label-free snapshot, without models."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse, hashlib, json, lzma, sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from metrics import rarity,random_batches,percentile,align_gain,aulc

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'studies/active_learning/nocw_free_selection_diagnostics_20261101'))
from condition_metrics import _canonical_conditions
from geometry_metrics import fixed_reference_ids

def phase(L):
    return 'early' if L<525 else 'middle' if L<813 else 'late' if L<1005 else 'extension'

def coding(values):
    names=sorted(set(values),key=str);lookup={v:i for i,v in enumerate(names)}
    return np.array([lookup[v] for v in values],dtype=np.int16),len(names)

def build_spaces(data):
    ids=sorted(data['metadata']);pos={v:i for i,v in enumerate(ids)}
    meta=data['metadata'];cond=[_canonical_conditions(meta[v]) for v in ids]
    assert all(v is not None for v in cond)
    smiles=[meta[v]['smiles'] for v in ids];molcode,nmol=coding(smiles)
    discrete=[molcode];cards=[nmol]
    for j in range(5):
        c,n=coding([v[j] for v in cond]);discrete.append(c);cards.append(n)
    codes=np.array(discrete).T
    # Numeric scales are fixed from candidate metadata, never response labels.
    numeric=np.array([[1/(1+float(v[0])),float(v[1]),float(v[2]),float(v[4])] for v in cond])
    lo=numeric.min(0);span=numeric.max(0)-lo;span[span==0]=1
    norm=(numeric-lo)/span
    condition=(cdist(norm,norm,metric='cityblock')+(codes[:,4,None]!=codes[None,:,4]))/5
    ham=np.zeros(condition.shape,np.float32)
    for j in range(6):ham+=(codes[:,j,None]!=codes[None,:,j])/6
    names=sorted(set(smiles));gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    fps=[gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in names]
    molecular=np.array([1-np.asarray(DataStructs.BulkTanimotoSimilarity(f,fps)) for f in fps],np.float32)
    struct=molecular[molcode[:,None],molcode[None,:]]
    return ids,pos,codes,cards,norm,condition.astype(np.float32),ham,struct,lo,span

def describe_batch(s,h,u,codes,cards,norm,ham,condition,struct,near_h,near_c,near_s,ref,prior_ref,binned,rare,rngb,prev):
    mol=codes[:,0];seen=set(mol[h]);sm=mol[s]
    same=(mol[s,None]==mol[h][None,:]);dc=condition[np.ix_(s,h)].copy();dc[~same]=np.inf
    same_near=dc.min(1);valid=np.isfinite(same_near)
    counts=(codes[s,None,1:]!=codes[h][None,:,1:]).sum(2)
    single=np.any(same&(counts==1),axis=1)
    equal=np.any(same&(counts==0),axis=1)
    within=condition[np.ix_(s,s)].copy();np.fill_diagonal(within,np.inf)
    ms=struct[np.ix_(s,s)]
    joint=.5*condition[np.ix_(ref,s)]+.5*struct[np.ix_(ref,s)]
    after=np.minimum(prior_ref,joint.min(1))
    before_c=condition[np.ix_(ref,h)].min(1)
    after_c=np.minimum(before_c,condition[np.ix_(ref,s)].min(1))
    before_s=struct[np.ix_(ref,h)].min(1)
    after_s=np.minimum(before_s,struct[np.ix_(ref,s)].min(1))
    histconditions={tuple(v[1:]) for v in codes[h]};states={tuple(v) for v in codes[h]}
    stats=dict(hamming_novelty=near_h[s].mean(),condition_novelty=near_c[s].mean(),
        structure_novelty=near_s[s].mean(),same_molecule_condition_novelty=same_near[valid].mean() if valid.any() else np.nan,
        same_molecule_available_fraction=valid.mean(),single_factor_history_fraction=single.mean(),exact_condition_history_fraction=equal.mean(),
        new_molecules=len(set(sm)-seen),new_molecule_slot_fraction=np.mean([v not in seen for v in sm]),
        unique_molecules=len(set(sm)),max_molecule_fraction=max(np.bincount(sm))/len(s),
        cumulative_molecules_before=len(seen),cumulative_molecules_after=len(seen|set(sm)),
        new_condition_combinations=len({tuple(v[1:]) for v in codes[s]}-histconditions),
        new_molecule_condition_states=len({tuple(v) for v in codes[s]}-states),
        batch_condition_nn=within.min(1).mean(),batch_structure_pair_distance=ms[np.triu_indices(len(s),1)].mean(),
        joint_coverage_gain=(prior_ref-after).mean(),condition_coverage_gain=(before_c-after_c).mean(),
        structure_coverage_gain=(before_s-after_s).mean(),rarity=rare[s].mean(),
        rarity_molecule=rarity(codes[:,:1],h,cards[:1])[s].mean(),
        rarity_condition=rarity(codes[:,1:],h,cards[1:])[s].mean(),
        rarity_paper_normalized=rarity(codes,h,cards,normalized=True)[s].mean(),
        distance_previous_batch=ham[np.ix_(s,prev[-1])].min(1).mean() if prev else np.nan,
        distance_recent3_batches=ham[np.ix_(s,np.concatenate(prev[-3:]))].min(1).mean() if prev else np.nan)
    squared=cdist(norm[s],norm[h],metric='sqeuclidean')+(codes[s,4,None]!=codes[h,4][None,:])
    stats['condition_novelty_rms']=np.sqrt(squared/5).min(1).mean()
    stats['nearest_history_equal_factors']=6*(1-near_h[s].mean())
    # Distinct continuous changes retain their magnitude in the condition view.
    for n,rc in binned.items():stats[f'rarity_bins{n}']=rc[s].mean()
    random=dict(hamming_novelty=near_h[rngb].mean(1),condition_novelty=near_c[rngb].mean(1),
        structure_novelty=near_s[rngb].mean(1),rarity=rare[rngb].mean(1))
    # Reference-by-random-batch blocks avoid a pool squared allocation per state.
    alljoint=.5*condition[ref]+.5*struct[ref]
    random['joint_coverage_gain']=(prior_ref[None,:]-np.minimum(prior_ref[None,:],alljoint[:,rngb].min(2).T)).mean(1)
    random['new_molecules']=np.array([len(set(mol[v])-seen) for v in rngb])
    for n,rc in binned.items():random[f'rarity_bins{n}']=rc[rngb].mean(1)
    for k,values in random.items():
        stats[k+'_random_mean']=values.mean();stats[k+'_random_sd']=values.std(ddof=1)
        stats[k+'_random_delta']=stats[k]-values.mean();stats[k+'_random_percentile']=percentile(stats[k],values)
        stats[k+'_random_mcse']=values.std(ddof=1)/np.sqrt(len(values))
        stats[k+'_random_delta_first100']=stats[k]-values[:100].mean()
    return stats,same_near,single

def main(config):
    cfg=json.loads(config.read_text());data=json.loads(lzma.decompress((OUT/cfg['snapshot']).read_bytes()))
    ids,pos,codes,cards,norm,condition,ham,struct,lo,span=build_spaces(data)
    curves=pd.read_csv(OUT/'row_curves.csv');lookup={(m,int(s)):dict(zip(g.budget,g.nrmse)) for (m,s),g in curves.groupby(['method','seed'])}
    states=data['states'];out=[];candidates=[];binned_codes={}
    for n in cfg['continuous_bins']:
        b=np.minimum((norm*n).astype(int),n-1)
        binned_codes[n]=np.column_stack([codes[:,0],b,codes[:,4]])
    for (method,seed),group in pd.DataFrame(states).groupby(['method','seed'],sort=True):
        refids=fixed_reference_ids(data['pool'][str(seed)],cfg['reference_size'],'nocw_free_selection_diagnostics_v1_seed_'+str(seed))
        ref=np.array([pos[v] for v in refids]);previous=[]
        first=group.sort_values('round').iloc[0];h0=np.array([pos[v] for v in first.labeled_ids])
        domain=np.array([pos[v] for v in data['initial'][str(seed)]+data['pool'][str(seed)]])
        seed_cards=[len(np.unique(codes[domain,j])) for j in range(codes.shape[1])]
        cumulative_history=set(h0)
        near_h=ham[:,h0].min(1);near_c=condition[:,h0].min(1);near_s=struct[:,h0].min(1)
        for state in group.sort_values('round').to_dict('records'):
            h=np.array([pos[v] for v in state['labeled_ids']]);u=np.array([pos[v] for v in state['unlabeled_ids']]);s=np.array([pos[v] for v in state['selected_ids']])
            assert set(h)==cumulative_history,('history transition',method,seed,state['round'])
            r=state['round'];L=state['L_before'];
            rare=rarity(codes,h,seed_cards)
            br={n:rarity(b,h,[len(np.unique(b[domain,j])) for j in range(b.shape[1])]) for n,b in binned_codes.items()}
            random=random_batches(u,cfg['random_batches'],len(s),cfg['random_seed']+int(seed)*100+r)
            before=(.5*condition[np.ix_(ref,h)]+.5*struct[np.ix_(ref,h)]).min(1)
            stats,same,single=describe_batch(s,h,u,codes,seed_cards,norm,ham,condition,struct,near_h,near_c,near_s,ref,before,br,rare,random,previous)
            scores=lookup[(method,int(seed))];gain=align_gain(scores,L,L+32)
            row=dict(method=method,seed=int(seed),round=r,L_before=L,L_after=L+32,phase=phase(L),batch_size=len(s),
                history_n=len(h),pool_n=len(u),random_batches=cfg['random_batches'],reference_n=len(ref),
                validation_nrmse_before=scores[L],validation_nrmse_after=scores[L+32],next_gain=gain,
                source=state['source'],**stats)
            out.append(row)
            for j,idx in enumerate(s):
                candidates.append(dict(method=method,seed=int(seed),round=r,sample_id=ids[idx],
                    hamming_novelty=float(near_h[idx]),condition_novelty=float(near_c[idx]),structure_novelty=float(near_s[idx]),
                    same_molecule_condition_novelty=float(same[j]) if np.isfinite(same[j]) else np.nan,
                    rarity=float(rare[idx]),single_factor_history=bool(single[j]),
                    hamming_pool_percentile=float((np.sum(near_h[u]<near_h[idx])+.5*np.sum(near_h[u]==near_h[idx]))/len(u)),
                    new_molecule=bool(codes[idx,0] not in set(codes[h,0]))))
            near_h=np.minimum(near_h,ham[:,s].min(1));near_c=np.minimum(near_c,condition[:,s].min(1));near_s=np.minimum(near_s,struct[:,s].min(1));previous.append(s)
            cumulative_history.update(s)
        print(method,seed,len(group),flush=True)
    df=pd.DataFrame(out);df.to_csv(OUT/'batch_metrics.csv',index=False)
    candidate_frame=pd.DataFrame(candidates)
    candidate_frame.to_csv(OUT/'candidate_metrics.csv',index=False)
    try:
        candidate_frame.to_parquet(OUT/'candidate_metrics.parquet',index=False)
    except ImportError:
        (OUT/'PARQUET_UNAVAILABLE.txt').write_text('No installed pyarrow/fastparquet engine. Lossless candidate_metrics.csv is delivered instead.\n')
    assert len(candidates)==32*len(states) and len(df)==len(states)
    commonseeds=cfg['common_seeds'];groups=df[df.seed.isin(commonseeds)].groupby(['method','seed']).L_after.max()
    common=int(groups.min());summaries=[]
    numeric=[c for c in df.select_dtypes('number').columns if c not in ['seed','round','L_before','L_after']]
    for name,g in [('A_common',df[df.seed.isin(commonseeds)&(df.L_after<=common)]),('all_available',df)]:
        ss=g.groupby(['method','seed'])[numeric].mean().reset_index();ss['scope']=name;summaries.append(ss)
    pd.concat(summaries).to_csv(OUT/'seed_metrics.csv',index=False)
    measures=[]
    for (m,s),g in curves.groupby(['method','seed']):
        for end in sorted(set([1005,common,1485,int(g.budget.max())])):
            v=g[g.budget<=end].sort_values('budget')
            if v.budget.max()!=end:continue
            measures.append(dict(method=m,seed=int(s),L_start=333,L_end=end,AULC=aulc(v.budget,v.nrmse),endpoint=v.nrmse.iloc[-1],n_points=len(v)))
    pd.DataFrame(measures).to_csv(OUT/'performance_summary.csv',index=False)
    (OUT/'behavior_verification.json').write_text(json.dumps(dict(status='PASS',states=len(states),selected_events=len(candidates),
        candidate_metadata_coverage=1.,common_seeds=commonseeds,common_end=common,random_batches=cfg['random_batches'],
        fixed_scales=dict(fields=['EA_fraction','density','sample_volume','loading_volume'],minimum=lo.tolist(),span=span.tolist(),
            metadata_rule='shared 4162-row target-free metadata min/max; no labels; rarity domain is per-seed L0 union U0',normalized_max=norm.max(0).tolist()),
        high_value_history='not defined prospectively; not computed',flow_rate='not recorded in five-factor catalog',
        structural_representation='static Morgan radius2/2048bit Tanimoto',new_training=0,new_label_queries=0),indent=2))
    print('PASS',len(df),len(candidates),'common',common,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=OUT/'config.json');main(p.parse_args().config)
