#!/usr/bin/env python3
"""Frozen-model condition diagnostics, never trains or opens a raw label table.

Observed deltas come only from accepted CW feedback. Numerical probes use
metadata-selected real candidate pairs and zero-sentinel graph caches.
"""
from __future__ import annotations
from pathlib import Path
import argparse, collections, gzip, hashlib, itertools, json, os, sys
from fractions import Fraction
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/nocw_condition_mpl')
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[4]; OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OLD=ROOT/'studies/active_learning/comprehensive_history_diagnostics_20261112'
CW=ROOT/'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101'
ARCH=ROOT/'reports/free_cw_final_20261112/scientific_artifacts'
NOCW=ROOT/'studies/active_learning/qgeognn_v2_row_llm_nocw_ablation/revisions/five_seed_nocw_to1005_20261029'
BASE=ROOT/'studies/active_learning/qgeognn_v2_row_sequential_b32'
SEEDS=[157,887,2357,6101,12203]
AXES=['PE/EA','Density g/ml','V/ul','loading solvent','Volume of loading solvent/ul']
LABELS={'PE/EA':'EA fraction','Density g/ml':'Density','V/ul':'Sample volume','loading solvent':'Loading solvent','Volume of loading solvent/ul':'Loading-solvent volume'}
INPUTS={}
def digest(b):return hashlib.sha256(b).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def source(p,role):
 p=Path(p);p=p if p.is_absolute() else ROOT/p
 b=p.read_bytes();r=str(p.relative_to(ROOT));INPUTS[r]={'path':r,'sha256':digest(b),'bytes':len(b),'role':role};return b
def register(p,role):
 p=Path(p);INPUTS[str(p.relative_to(ROOT))]={'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size,'role':role}
def js(p,role='metadata_or_source'):return json.loads(source(p,role))
def phase(r):return ['333-525','525-813','813-1005','1005-1485'][0 if r<6 else 1 if r<15 else 2 if r<21 else 3]
def molkey(s):return digest(s.encode())[:16]
def val(c,k):
 if k=='PE/EA':
  a,b=map(Fraction,c[k].split('/'));return b/(a+b)
 if k=='loading solvent':return str(c[k])
 return Fraction(str(c[k]))
def ckey(c,excluded=None):return tuple((k,str(val(c,k))) for k in AXES if k!=excluded)
def pair_candidates(records,axis):
 groups=collections.defaultdict(list)
 for a in records:groups[a['smiles'],ckey(a['conditions'],axis)].append(a)
 pairs=[]
 for group in groups.values():
  for a,b in itertools.combinations(group,2):
   va,vb=val(a['conditions'],axis),val(b['conditions'],axis)
   if va==vb:continue
   if va>vb:a,b=b,a
   pairs.append((a,b))
 return sorted(pairs,key=lambda ab:digest((ab[0]['id']+'|'+ab[1]['id']).encode()))
def csv(name,rows):
 d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);d.to_csv(OUT/'tables'/name,index=False,float_format='%.17g');return d
def dump(name,obj):(OUT/'verification'/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def observed_pairs():
 """Extend old aggregate diagnostics with signed, molecule-balanced, robust views."""
 pairs=[];duplicate_groups=[];archives=0
 for seed in SEEDS:
  ap=ARCH/f'seed_{seed}_results.jsonl.gz';register(ap,'published_selected_feedback_archive');arc={}
  with gzip.open(ap,'rt') as f:
   for line in f:
    z=json.loads(line);name=Path(z['path']).name
    if name in ['contract.json','feedback.json','batch_freeze.json']:
     assert digest(z['raw_utf8'].encode())==z['sha256'];arc[z['path']]=json.loads(z['raw_utf8']);archives+=1
  observed={}
  for r in range(36):
   path=CW/f'selections/seed_{seed}/free_cw_sol61_xhigh_rerun32/round_{r:02d}'
   rel=str(path.relative_to(ROOT));contract=arc[rel+'/contract.json'];fb=arc[rel+'/feedback.json'];freeze=arc[rel+'/batch_freeze.json']
   raw=source(path/'catalog.json','preselection_X_and_legally_observed_L');assert digest(raw)==contract['catalog_sha256']
   cat=json.loads(raw);cand={x['id']:x for x in cat['candidates']+cat['pending']}
   assert set(cand)==set(contract['U_t_ids'])
   assert set(v['id'] for v in cat['observed'])==set(contract['L_t_ids'])
   assert [v['candidate_id'] for v in fb['records']]==freeze['batch_ids']
   rows=[]
   for x in cat['observed']:observed[x['id']]={'id':x['id'],'smiles':x['smiles'],'conditions':x['conditions'],'y':[x['V1_ml'],x['V2_ml']]}
   for f,z in zip(fb['records'],freeze['premeasurement_predictions']):
    i=f['candidate_id'];c=cand[i];assert i not in set(contract['L_t_ids'])
    pred=[f['pred_V1_ml'],f['pred_V2_ml']];truth=[f['true_V1_ml'],f['true_V2_ml']]
    assert all(pred[t]==z[f'pred_V{t+1}_ml']==c[f'pred_V{t+1}_q50_ml'] for t in range(2))
    a={'id':i,'smiles':c['smiles'],'conditions':c['conditions'],'pred':pred,'y':truth};rows.append(a);observed[i]=a
   for axis in AXES:
    for a,b in pair_candidates(rows,axis):
     assert a['smiles']==b['smiles'] and sum(val(a['conditions'],k)!=val(b['conditions'],k) for k in AXES)==1
     for t in range(2):
      dy=b['y'][t]-a['y'][t];dp=b['pred'][t]-a['pred'][t]
      pairs.append({'seed':seed,'round':r,'phase':phase(r),'axis':axis,'target':f'V{t+1}',
       'molecule_key':molkey(a['smiles']),'pair_key':digest('|'.join(sorted([a['id'],b['id']])).encode())[:16],
       'row_a':a['id'],'row_b':b['id'],'axis_low':str(val(a['conditions'],axis)),'axis_high':str(val(b['conditions'],axis)),
       'ea_fraction':float(val(a['conditions'],'PE/EA')),'loading_solvent':a['conditions']['loading solvent'],
       'actual_signed_delta_ml':dy,'pred_signed_delta_ml':dp,'signed_error_pred_minus_actual_ml':dp-dy,
       'absolute_difference_error_ml':abs(dp-dy),'actual_abs_delta_ml':abs(dy),'pred_abs_delta_ml':abs(dp),
       'direction_agreement':int(np.sign(dy)==np.sign(dp)),'pred_flat_lt_0p1_ml':int(abs(dp)<.1),
       'under_amplitude':int(abs(dp)<abs(dy)),
       'amplitude_ratio':abs(dp)/abs(dy) if abs(dy)>1e-12 else np.nan,
       'acquisition_batch_same':True,'lab_batch_known':False,'future_label_feature':False})
  groups=collections.defaultdict(list)
  for a in observed.values():groups[a['smiles'],ckey(a['conditions'])].append(a)
  for (sm,key),g in groups.items():
   if len(g)<2:continue
   for t in range(2):
    y=np.array([a['y'][t] for a in g]);duplicate_groups.append({'seed':seed,'target':f'V{t+1}','molecule_key':molkey(sm),'n_rows':len(g),
     'recorded_group_key':digest(str((sm,key)).encode())[:16],'range_ml':float(np.ptp(y)),'sd_ml':float(np.std(y,ddof=1))})
 p=csv('condition_pair_signed_events.csv',pairs);assert len(p)==1786 and len(p[p.axis.eq('V/ul')])==330
 # Summaries balance molecule first, then seed. Pair-sharing dependence retained.
 fields=['actual_signed_delta_ml','pred_signed_delta_ml','signed_error_pred_minus_actual_ml','absolute_difference_error_ml',
  'actual_abs_delta_ml','pred_abs_delta_ml','direction_agreement','pred_flat_lt_0p1_ml','under_amplitude','amplitude_ratio']
 bymol=p.groupby(['seed','phase','axis','target','molecule_key'])[fields].mean().reset_index()
 counts=p.groupby(['seed','phase','axis','target','molecule_key']).size().rename('pair_events').reset_index();bymol=bymol.merge(counts)
 csv('condition_pair_molecule_phase.csv',bymol)
 seedphase=bymol.groupby(['seed','phase','axis','target'])[fields].mean().reset_index()
 csv('condition_pair_seed_phase.csv',seedphase)
 bymolall=p.groupby(['seed','axis','target','molecule_key'])[fields].mean().reset_index()
 seedall=bymolall.groupby(['seed','axis','target'])[fields].mean().reset_index()
 csv('condition_pair_seed_balanced.csv',seedall)
 sg=seedall.groupby(['axis','target'])[fields]
 csv('condition_pair_summary.csv',sg.mean().add_suffix('_mean').join(sg.std().add_suffix('_std')).reset_index())
 # Robustness: per-molecule medians; trim largest observed delta within seed only AFTER fixed pair assembly.
 robust=[]
 for (s,axis,t),g in p.groupby(['seed','axis','target']):
  m=g.groupby('molecule_key')[fields].mean();med=g.groupby('molecule_key')[fields].median()
  cutoff=g.actual_abs_delta_ml.quantile(.9);trim=g[g.actual_abs_delta_ml<=cutoff]
  for kind,z in [('molecule_equal',m),('molecule_median',med),('trim_top10pct_observed_delta_diagnostic',trim.groupby('molecule_key')[fields].mean())]:
   for f in fields:robust.append({'seed':s,'axis':axis,'target':t,'method':kind,'metric':f,'value':z[f].mean(),'n_molecules':len(z),'n_pair_events':len(trim) if 'trim' in kind else len(g)})
  # Leave one molecule out range shows domination without changing any selection/model.
  for f in ['actual_abs_delta_ml','pred_abs_delta_ml','absolute_difference_error_ml','direction_agreement']:
   vals=m[f].to_numpy();loo=[np.delete(vals,i).mean() for i in range(len(vals))] if len(vals)>1 else []
   robust.append({'seed':s,'axis':axis,'target':t,'method':'leave_one_molecule_out_min','metric':f,'value':min(loo) if loo else np.nan,'n_molecules':len(vals),'n_pair_events':len(g)})
   robust.append({'seed':s,'axis':axis,'target':t,'method':'leave_one_molecule_out_max','metric':f,'value':max(loo) if loo else np.nan,'n_molecules':len(vals),'n_pair_events':len(g)})
 csv('condition_pair_robustness.csv',robust)
 csv('condition_duplicate_groups.csv',duplicate_groups)
 # Record independent degrees of freedom and metadata distributions.
 comp=[]
 for (s,axis),g in p[p.target.eq('V1')].groupby(['seed','axis']):
  parents={x:x for x in set(g.row_a)|set(g.row_b)}
  def find(x):
   while parents[x]!=x:parents[x]=parents[parents[x]];x=parents[x]
   return x
  for a,b in zip(g.row_a,g.row_b):parents[find(a)]=find(b)
  comp.append({'seed':s,'axis':axis,'pair_events':len(g),'unique_row_pairs':g.pair_key.nunique(),'unique_rows':len(parents),
   'connected_row_components':len(set(find(i) for i in parents)),'molecules':g.molecule_key.nunique(),
   'largest_molecule_event_fraction':g.molecule_key.value_counts().max()/len(g)})
 csv('condition_pair_dependence.csv',comp)
 csv('condition_pair_metadata_distribution.csv',p.groupby(['seed','axis','target','loading_solvent','ea_fraction']).agg(n_pairs=('round','size'),n_molecules=('molecule_key','nunique'),actual_abs_delta_ml=('actual_abs_delta_ml','mean'),pred_abs_delta_ml=('pred_abs_delta_ml','mean')).reset_index())
 plot_pairs(p,seedphase)
 return {'archive_records_verified':archives,'pair_endpoint_records':len(p),'pair_events':len(p)//2,'sample_volume_pair_events':len(p[p.axis.eq('V/ul')])//2,'observed_source':'matched accepted selected feedback only'}

def plot_pairs(p,sp):
 fig,axs=plt.subplots(1,2,figsize=(11,4.4),sharex=True)
 for ax,t in zip(axs,['V1','V2']):
  z=p[p.axis.eq('V/ul')&p.target.eq(t)]
  for s,g in z.groupby('seed'):ax.scatter(g.actual_signed_delta_ml,g.pred_signed_delta_ml,s=13,alpha=.65,label=str(s))
  lim=max(abs(z.actual_signed_delta_ml).max(),abs(z.pred_signed_delta_ml).max())*1.03
  ax.plot([-lim,lim],[-lim,lim],'k--',lw=.8);ax.axhline(0,color='grey',lw=.6);ax.axvline(0,color='grey',lw=.6)
  ax.set(xlabel='Observed signed delta (mL)',ylabel='Predicted signed delta (mL)',title=t+' sample-volume pairs');ax.legend(fontsize=7)
 fig.text(.02,.01,'165 same-acquisition-batch pairs, 93 molecules, five seeds. Laboratory batch unknown; pairs share rows. Source: condition_pair_signed_events.csv.',fontsize=8)
 fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(OUT/'figures/condition_signed_response.png',dpi=180);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(11,4.2))
 for ax,t in zip(axs,['V1','V2']):
  z=sp[sp.axis.eq('V/ul')&sp.target.eq(t)]
  for s,g in z.groupby('seed'):
   ph=['333-525','525-813','813-1005','1005-1485'];g=g.set_index('phase').reindex(ph)
   ax.plot(range(4),g.actual_abs_delta_ml,'o-',label=str(s)+' observed',alpha=.65)
   ax.plot(range(4),g.pred_abs_delta_ml,'x--',alpha=.65,color=ax.lines[-1].get_color())
  ax.set_xticks(range(4),ph,rotation=15);ax.set_title(t);ax.set_ylabel('Molecule-balanced mean |delta| (mL)');ax.legend(fontsize=6,ncol=2)
 fig.text(.02,.01,'Within each seed/phase, equal-weight molecules rather than pair count. Missing phases remain gaps. Solid=observed; dashed=prediction.',fontsize=8)
 fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(OUT/'figures/condition_stage_response.png',dpi=180);plt.close(fig)

def nocw_observed_pairs():
 """New direct NoCW replication at the five-seed common completed horizon."""
 rows=[]
 for seed in SEEDS:
  pp=BASE/f'splits/row_seed_{seed}.csv';source(pp,'NoCW_role_ID_mapping_no_response_columns');partition=pd.read_csv(pp)
  allowed=set(partition[partition.role.isin(['l0','u0'])].sample_id)
  for r in range(21):
   d=NOCW/f'selections/seed_{seed}/llm32_nocw/round_{r:02d}'
   contract=js(d/'contract.json','NoCW_preselection_contract');freeze=js(d/'batch_freeze.json','NoCW_accepted_batch_freeze')
   raw=source(d/'catalog.json','NoCW_preselection_X_and_authorized_observed_L');assert digest(raw)==contract['catalog_sha256']
   cat=json.loads(raw);fb=js(d/'feedback.json','NoCW_selected_accepted_feedback')
   cand={x['id']:x for x in cat['candidates']+cat['pending']}
   assert set(cand)==set(contract['U_t_ids']) and set(x['id'] for x in cat['observed'])==set(contract['L_t_ids'])
   assert (set(contract['L_t_ids'])|set(contract['U_t_ids']))==allowed and len(contract['L_t_ids'])==333+32*r
   assert [f['candidate_id'] for f in fb['records']]==freeze['batch_ids'] and not set(freeze['batch_ids'])&set(contract['L_t_ids'])
   chosen=[]
   for f in fb['records']:
    c=cand[f['candidate_id']];chosen.append({'id':f['candidate_id'],'smiles':c['smiles'],'conditions':c['conditions'],
      'pred':[f['pred_V1_ml'],f['pred_V2_ml']],'y':[f['true_V1_ml'],f['true_V2_ml']]})
    assert f['pred_V1_ml']==c['pred_V1_q50_ml'] and f['pred_V2_ml']==c['pred_V2_q50_ml']
   for axis in AXES:
    for a,b in pair_candidates(chosen,axis):
     for t in range(2):
      dy=b['y'][t]-a['y'][t];dp=b['pred'][t]-a['pred'][t]
      rows.append({'seed':seed,'round':r,'phase':phase(r),'axis':axis,'target':f'V{t+1}',
       'molecule_key':molkey(a['smiles']),'pair_key':digest('|'.join(sorted([a['id'],b['id']])).encode())[:16],
       'row_a':a['id'],'row_b':b['id'],'actual_signed_delta_ml':dy,'pred_signed_delta_ml':dp,
       'actual_abs_delta_ml':abs(dy),'pred_abs_delta_ml':abs(dp),'signed_error_pred_minus_actual_ml':dp-dy,
       'absolute_difference_error_ml':abs(dp-dy),'direction_agreement':int(np.sign(dy)==np.sign(dp)),
       'pred_flat_lt_0p1_ml':int(abs(dp)<.1),'under_amplitude':int(abs(dp)<abs(dy)),
       'amplitude_ratio':abs(dp)/abs(dy) if abs(dy)>1e-12 else np.nan,
       'acquisition_batch_same':True,'lab_batch_known':False})
 p=csv('condition_nocw_pair_signed_events.csv',rows)
 fields=['actual_signed_delta_ml','pred_signed_delta_ml','actual_abs_delta_ml','pred_abs_delta_ml','signed_error_pred_minus_actual_ml',
  'absolute_difference_error_ml','direction_agreement','pred_flat_lt_0p1_ml','under_amplitude','amplitude_ratio']
 m=p.groupby(['seed','phase','axis','target','molecule_key'])[fields].mean().reset_index();csv('condition_nocw_pair_molecule_phase.csv',m)
 csv('condition_nocw_pair_seed_phase.csv',m.groupby(['seed','phase','axis','target'])[fields].mean().reset_index())
 m=p.groupby(['seed','axis','target','molecule_key'])[fields].mean().reset_index();csv('condition_nocw_pair_seed_balanced.csv',m.groupby(['seed','axis','target'])[fields].mean().reset_index())
 counts=p.groupby(['axis','target']).agg(pair_events=('seed','size'),unique_row_pairs=('pair_key','nunique'),molecules=('molecule_key','nunique'),seeds=('seed','nunique')).reset_index()
 csv('condition_nocw_pair_counts.csv',counts)
 robust=[]
 for (s,axis,t),g in p.groupby(['seed','axis','target']):
  m=g.groupby('molecule_key')[fields].mean()
  trim=g[g.actual_abs_delta_ml<=g.actual_abs_delta_ml.quantile(.9)].groupby('molecule_key')[fields].mean()
  for kind,z in [('molecule_equal',m),('trim_top10pct_observed_delta_diagnostic',trim)]:
   for f in fields:robust.append({'seed':s,'axis':axis,'target':t,'method':kind,'metric':f,'value':z[f].mean(),'n_molecules':len(z)})
  for f in ['actual_abs_delta_ml','pred_abs_delta_ml','absolute_difference_error_ml']:
   vals=m[f].to_numpy();loo=[np.delete(vals,i).mean() for i in range(len(vals))] if len(vals)>1 else []
   for kind,v in [('min',min(loo) if loo else np.nan),('max',max(loo) if loo else np.nan)]:
    robust.append({'seed':s,'axis':axis,'target':t,'method':'leave_one_molecule_out_'+kind,'metric':f,'value':v,'n_molecules':len(vals)})
 csv('condition_nocw_pair_robustness.csv',robust)
 return {'nocw_accepted_batches_checked':105,'nocw_pair_events':len(p)//2,'nocw_sample_volume_pair_events':int(p.axis.eq('V/ul').sum())//2}

def frozen_probes():
 import torch
 from torch_geometric.nn import global_add_pool
 from qgeognn_al.models import load_predictor_checkpoint
 from qgeognn_al.training.predictor import loader_pair
 from qgeognn_al.data import eluent_descriptor,minmax_transform
 torch.set_num_threads(2)
 results=[];gradients=[];encoded=[];paramchecks=[]
 for seed in [157,887]:
  catp=NOCW/f'selections/seed_{seed}/llm32_nocw/round_00/catalog.json';cat=js(catp,'NoCW_initial_outer_train_X_and_L333_only')
  records=cat['candidates']+cat['pending']+cat['observed'];assert len(records)==3330 and len({a['id'] for a in records})==3330
  partp=BASE/f'splits/row_seed_{seed}.csv';source(partp,'role_ID_mapping_no_response_columns');part=pd.read_csv(partp)
  assert not any(c in part for c in ['V1_ml','V2_ml','t1','t2'])
  role=dict(zip(part.sample_id,part.role));indices=dict(zip(part.sample_id,part.canonical_index))
  assert set(a['id'] for a in records)==set(part[part.role.isin(['l0','u0'])].sample_id)
  gp=BASE/f'runtime/seed_{seed}/scrubbed_graphs.pt';register(gp,'zero_sentinel_graph_inputs');atom,angle=torch.load(gp,weights_only=False)
  assert all(torch.count_nonzero(a.y).item()==0 for a in atom),'graph cache not zero sentinel'
  state_records={a['id']:a for a in records}
  # Fixed pair set from existing outer-train X. All pairs are real rows and selected independent of labels/gain.
  selected_pairs=[]
  for axis in AXES:
   used=set();chosen=[]
   for a,b in pair_candidates(records,axis):
    if a['id'] in used or b['id'] in used:continue
    chosen.append((axis,a,b));used.update([a['id'],b['id']])
    if len(chosen)==32:break
   selected_pairs+=chosen
  # Software derivatives: deterministic rows, one per molecule, at actual observed candidate coordinates.
  first={}
  for a in sorted(records,key=lambda a:digest(a['id'].encode())):first.setdefault(a['smiles'],a)
  gradrows=list(first.values())[:32]
  for budget in [525,1005]:
   r=(budget-333)//32;fp=NOCW/f'runtime/seed_{seed}/llm32_nocw/round_{r:02d}/prediction_freeze.json';fr=js(fp,'NoCW_state_prediction_freeze')
   ck=Path(fr['checkpoint_path']);register(ck,'frozen_NoCW_model_weights');assert sha(ck)==fr['checkpoint_sha256']
   model=load_predictor_checkpoint(ck);original={k:v.detach().clone() for k,v in model.state_dict().items()}
   model.eval();scale=model.condition_branch.loading_range.detach().cpu().numpy();normal=model.condition_branch.normalization
   cp=js(BASE/f'runtime/seed_{seed}/context.json','frozen_X_scaler_and_L333_scales')['contract']
   # Exact reconstruction checks prove V/density/volume/solvent were not lost in the cached input.
   all_ids=sorted(set([x['id'] for _,a,b in selected_pairs for x in [a,b]]+[a['id'] for a in gradrows]))
   ix=[int(indices[i]) for i in all_ids]
   aa,bb=next(iter(loader_pair(atom,angle,ix,2048)[0])),next(iter(loader_pair(atom,angle,ix,2048)[1]))
   with torch.no_grad():
    typed=model.condition_branch.typed_inputs(aa)
    nodes=model.backbone(aa,bb);pooled=global_add_pool(nodes,aa.batch)
    residual,hidden=model.condition_branch(aa);fusion=pooled+residual;out=model(aa,bb)
   ordmap={i:j for j,i in enumerate(all_ids)}
   for i in all_ids:
    a=state_records[i];co=a['conditions'];j=ordmap[i]
    expected_amount=(float(co['Density g/ml'])*float(co['V/ul'])-normal.loading_amount_min)/(normal.loading_amount_max-normal.loading_amount_min)
    expected_vol=(float(co['Volume of loading solvent/ul'])-normal.loading_volume_min)/(normal.loading_volume_max-normal.loading_volume_min)
    elu=minmax_transform(eluent_descriptor(co['PE/EA']),cp['preprocessing']['scaler']['eluent'])
    assert np.allclose(typed[j,:4].numpy(),[elu[4],elu[5],expected_amount,expected_vol],rtol=2e-6,atol=1e-6)
    encoded.append({'seed':seed,'budget':budget,'row_id':i,'molecule_key':molkey(a['smiles']),
     'sample_volume_ul':float(co['V/ul']),'density_g_ml':float(co['Density g/ml']),
     'mass_mg':float(co['Density g/ml'])*float(co['V/ul']),'mass_normalized':float(typed[j,2]),
     'loading_volume_ul':float(co['Volume of loading solvent/ul']),'loading_volume_normalized':float(typed[j,3]),
     'input_roundtrip_check':True})
   for axis,a,b in selected_pairs:
    ja,jb=ordmap[a['id']],ordmap[b['id']];dc=typed[jb]-typed[ja];dr=residual[jb]-residual[ja];dp=pooled[jb]-pooled[ja];df=fusion[jb]-fusion[ja]
    denom=float(val(b['conditions'],axis)-val(a['conditions'],axis)) if axis!='loading solvent' else np.nan
    for t in [0,1]:
     do=float(out[jb,t*3+1]-out[ja,t*3+1]);results.append({'seed':seed,'budget':budget,'axis':axis,'target':f'V{t+1}',
       'row_a':a['id'],'row_b':b['id'],'molecule_key':molkey(a['smiles']),
       'condition_input_l2':float(torch.linalg.vector_norm(dc)),'condition_hidden_l2':float(torch.linalg.vector_norm(hidden[jb]-hidden[ja])),
       'condition_residual_l2':float(torch.linalg.vector_norm(dr)),'backbone_pool_l2':float(torch.linalg.vector_norm(dp)),
       'fused_l2':float(torch.linalg.vector_norm(df)),'output_signed_delta_ml':do,'output_abs_delta_ml':abs(do),
       'output_signed_secant_per_axis_unit':do/denom if np.isfinite(denom) else np.nan,
       'domain':'actual_outer_train_X_pairs','any_label_read':False})
   # Local derivatives wrt graph-level typed continuous entries, with graph representation held fixed.
   # They are software directional derivatives at existing X, not response labels or arbitrary new experiments.
   gix=[int(indices[a['id']]) for a in gradrows]
   ga,gb=next(iter(loader_pair(atom,angle,gix,2048)[0])),next(iter(loader_pair(atom,angle,gix,2048)[1]))
   with torch.no_grad():
    z=model.condition_branch.typed_inputs(ga).detach();po=global_add_pool(model.backbone(ga,gb),ga.batch).detach()
   z.requires_grad_(True);hid=model.condition_branch.activation(model.condition_branch.hidden(z));res=model.condition_branch.output(hid)
   raw=model.head(po+res)
   for target in [0,1]:
    jac=torch.autograd.grad(raw[:,target*3+1].sum(),z,retain_graph=True)[0]
    for j,a in enumerate(gradrows):
     co=a['conditions'];g=jac[j].detach().numpy();v=float(co['V/ul']);density=float(co['Density g/ml'])
     gradients.append({'seed':seed,'budget':budget,'row_id':a['id'],'molecule_key':molkey(a['smiles']),'target':f'V{target+1}',
      'd_output_d_normalized_mass':float(g[2]),'d_output_d_mass_mg':float(g[2]/scale[0]),
      'd_output_d_sample_volume_ul_at_fixed_density':float(g[2]*density/scale[0]),
      'd_output_d_density_at_fixed_volume':float(g[2]*v/scale[0]),
      'd_output_d_normalized_loading_volume':float(g[3]),'d_output_d_loading_volume_ul':float(g[3]/scale[1]),
      'd_output_d_normalized_eluent_hba_completion_path':float(g[0]),
      'd_output_d_normalized_eluent_logp_completion_path':float(g[1]),
      'derivative_scope':'local_software_partial_at_real_X; PE backbone path not included',
      'q50_active':bool(raw[j,target*3+1]>0)})
   assert all(torch.equal(v,original[k]) for k,v in model.state_dict().items()),'frozen parameters changed'
   paramchecks.append({'seed':seed,'budget':budget,'checkpoint_sha256':fr['checkpoint_sha256'],'weights_unchanged':True,
    'real_candidate_pairs':len(selected_pairs),'derivative_rows':len(gradrows),'zero_sentinel_graphs':len(atom)})
 c=csv('condition_frozen_pair_probes.csv',results);g=csv('condition_frozen_local_gradients.csv',gradients);csv('condition_input_roundtrip.csv',encoded)
 sumcols=['condition_input_l2','condition_hidden_l2','condition_residual_l2','backbone_pool_l2','fused_l2','output_abs_delta_ml']
 sg=c.groupby(['seed','budget','axis','target'])[sumcols]
 csv('condition_frozen_probe_summary.csv',sg.mean().add_suffix('_mean').join(sg.median().add_suffix('_median')).join(sg.max().add_suffix('_max')).reset_index())
 csv('condition_gradient_summary.csv',g.groupby(['seed','budget','target']).agg(n_rows=('row_id','size'),
  mean_abs_dq_dmass=('d_output_d_mass_mg',lambda x:abs(x).mean()),mean_abs_dq_dsampleV=('d_output_d_sample_volume_ul_at_fixed_density',lambda x:abs(x).mean()),
  mean_abs_dq_dloadingV=('d_output_d_loading_volume_ul',lambda x:abs(x).mean()),
  max_abs_dq_dmass=('d_output_d_mass_mg',lambda x:abs(x).max()),nonzero_mass_derivative=('d_output_d_mass_mg',lambda x:np.mean(abs(x)>1e-8))).reset_index())
 # Original source unchanged: forwards/gradients only, no optimizer is constructed.
 dump('condition_frozen_model_checks.json',{'status':'PASS','states':paramchecks,'pair_selection':'fixed hash, at most32 disjoint actual row pairs per axis/seed; same set at both budgets',
  'derivative_selection':'fixed hash, first32 distinct molecules per seed; real X coordinates','test_truth_reads':0,'training_calls':0,'llm_calls':0})
 return {'frozen_state_count':len(paramchecks),'legal_pair_target_records':len(c),'local_gradient_target_records':len(g),'all_weights_unchanged':True,'all_input_roundtrips_pass':True}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pairs-only',action='store_true');ap.add_argument('--probes-only',action='store_true');args=ap.parse_args()
 for d in ['tables','figures','verification']:(OUT/d).mkdir(exist_ok=True)
 # source trace is deliberately restricted to explicit Python modules.
 trace=['src/qgeognn_al/data.py','src/qgeognn_al/input_schema.py','src/qgeognn_al/schemas/conditions.py','src/qgeognn_al/models/qgeognn_v2.py',
 'src/qgeognn_al/active_learning_v2/runner.py','src/qgeognn_al/active_learning_v2/benchmark_protocol.py','src/qgeognn_al/training/predictor.py','application/QGeoGNN.py']
 for p in trace:source(p,'condition_source_semantics_only')
 checks={}
 if not args.probes_only:
  checks.update(observed_pairs());checks.update(nocw_observed_pairs())
 if not args.pairs_only:checks.update(frozen_probes())
 dump('condition_input_manifest.json',{'sources':list(INPUTS.values()),'label_scope':'accepted selected feedback / matching prior L only; no original truth CSVs','source_search_scope':'explicit Python files only'})
 dump('condition_checks.json',{'status':'PASS','checks':checks,'training_calls':0,'llm_calls':0,'test_truth_reads':0,
  'meaning':'Input correctness and frozen response diagnostics, not identification of irreducible noise or causal AL gain'})
 print(json.dumps(checks))
if __name__=='__main__':main()
