#!/usr/bin/env python3
"""Independent scalar checks of paired deltas and analytic condition Jacobians."""
from pathlib import Path
import collections,gzip,hashlib,json,math,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 pairs=pd.read_csv(OUT/'tables/condition_pair_signed_events.csv')
 # This independent check uses direct feedback differences, no production pair helper.
 sampled=pairs[pairs.seed.eq(2357)&pairs['round'].isin([0,3,10,17,28])]
 truth={};pred={};checked=0;maximum=0.
 ap=ROOT/'reports/free_cw_final_20261112/scientific_artifacts/seed_2357_results.jsonl.gz'
 with gzip.open(ap,'rt') as f:
  for line in f:
   o=json.loads(line)
   if not o['path'].endswith('/feedback.json'):continue
   r=int(Path(o['path']).parent.name.split('_')[1])
   if r not in [0,3,10,17,28]:continue
   assert hashlib.sha256(o['raw_utf8'].encode()).hexdigest()==o['sha256']
   d=json.loads(o['raw_utf8'])
   for x in d['records']:
    for t in ['V1','V2']:
     truth[r,x['candidate_id'],t]=float(x[f'true_{t}_ml']);pred[r,x['candidate_id'],t]=float(x[f'pred_{t}_ml'])
 for a in sampled.itertuples():
  actual=truth[a.round,a.row_b,a.target]-truth[a.round,a.row_a,a.target]
  predicted=pred[a.round,a.row_b,a.target]-pred[a.round,a.row_a,a.target]
  for v,w in [(actual,a.actual_signed_delta_ml),(predicted,a.pred_signed_delta_ml),
   (predicted-actual,a.signed_error_pred_minus_actual_ml),(abs(predicted-actual),a.absolute_difference_error_ml)]:
   err=abs(v-w);maximum=max(maximum,err);assert err<1e-10;checked+=1
 # Reproduce OLD absolute phase summaries: extension changes orientation, not prior arithmetic.
 old=pd.read_csv(ROOT/'studies/active_learning/comprehensive_history_diagnostics_20261112/tables/model_same_batch_single_factor_response.csv')
 means=pairs.groupby(['seed','phase','axis','target']).agg(actual_abs_delta_ml=('actual_abs_delta_ml','mean'),
  pred_abs_delta_ml=('pred_abs_delta_ml','mean'),difference_error_ml=('absolute_difference_error_ml','mean')).reset_index()
 merged=old.merge(means,on=['seed','phase','axis','target'],suffixes=('_old','_new'),validate='one_to_one');assert len(merged)==len(old)
 prior_max=0.
 for k in ['actual_abs_delta_ml','pred_abs_delta_ml','difference_error_ml']:
  prior_max=max(prior_max,float(np.max(abs(merged[k+'_old']-merged[k+'_new']))))
 assert prior_max<1e-10
 # Independent Jacobian algebra: Dhead * Wout * Dhidden * Whidden, evaluated only at legitimate real X.
 import torch
 from torch_geometric.nn import global_add_pool
 from qgeognn_al.models import load_predictor_checkpoint
 from qgeognn_al.training.predictor import loader_pair
 torch.set_num_threads(2)
 grad=pd.read_csv(OUT/'tables/condition_frozen_local_gradients.csv');maxjac=0.;jacchecks=0
 for (seed,budget),g in grad.groupby(['seed','budget']):
  fr=ROOT/f'studies/active_learning/qgeognn_v2_row_llm_nocw_ablation/revisions/five_seed_nocw_to1005_20261029/runtime/seed_{seed}/llm32_nocw/round_{(budget-333)//32:02d}/prediction_freeze.json'
  ck=json.loads(fr.read_text());m=load_predictor_checkpoint(Path(ck['checkpoint_path']));m.eval()
  atom,angle=torch.load(ROOT/f'studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_{seed}/scrubbed_graphs.pt',weights_only=False)
  assert all(torch.count_nonzero(a.y).item()==0 for a in atom)
  part=pd.read_csv(ROOT/f'studies/active_learning/qgeognn_v2_row_sequential_b32/splits/row_seed_{seed}.csv')
  mp=dict(zip(part.sample_id,part.canonical_index));ids=sorted(g.row_id.unique());ix=[int(mp[i]) for i in ids]
  assert set(part.set_index('sample_id').loc[ids].role)<= {'l0','u0'}
  a=next(iter(loader_pair(atom,angle,ix,2048)[0]));b=next(iter(loader_pair(atom,angle,ix,2048)[1]))
  with torch.no_grad():
   z=m.condition_branch.typed_inputs(a);hiddenpre=m.condition_branch.hidden(z);hiddenmask=(hiddenpre>0).to(z.dtype)
   pooled=global_add_pool(m.backbone(a,b),a.batch);res=m.condition_branch.output(torch.relu(hiddenpre));raw=m.head[0](pooled+res)
   whead=m.head[0].weight;wout=m.condition_branch.output.weight;whid=m.condition_branch.hidden.weight
   for j,i in enumerate(ids):
    jac=(whead@wout*hiddenmask[j][None,:])@whid;jac=jac*(raw[j]>0)[:,None]
    for target,k in [('V1',1),('V2',4)]:
     row=g[(g.row_id==i)&(g.target==target)].iloc[0]
     expected=[jac[k,2].item(),jac[k,3].item()]
     actual=[row.d_output_d_normalized_mass,row.d_output_d_normalized_loading_volume]
     for x,y in zip(expected,actual):maxjac=max(maxjac,abs(x-y));assert abs(x-y)<1e-6;jacchecks+=1
 # Boundary cases for exact pair metadata rule, including multi-axis exclusion and PE fraction direction.
 import importlib.util
 sp=importlib.util.spec_from_file_location('cond_production',OUT/'scripts/condition_mechanism.py');mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
 c={'PE/EA':'5/1','Density g/ml':1.,'V/ul':50.,'loading solvent':'PE','Volume of loading solvent/ul':200}
 a={'id':'a','smiles':'CC','conditions':c};b={'id':'b','smiles':'CC','conditions':{**c,'V/ul':100.}}
 d={'id':'d','smiles':'CC','conditions':{**c,'V/ul':100.,'PE/EA':'10/1'}}
 assert len(mod.pair_candidates([a,b],'V/ul'))==1 and not mod.pair_candidates([a,d],'V/ul')
 assert mod.val({'PE/EA':'50/1'},'PE/EA')<mod.val({'PE/EA':'5/1'},'PE/EA')
 result={'status':'PASS','fixed_sample':'seed2357 rounds0,3,10,17,28 all available pair endpoint records',
  'direct_feedback_scalar_checks':checked,'max_scalar_error':maximum,'old_phase_absolute_means_checked':len(merged),'max_old_summary_error':prior_max,
  'analytic_chain_rule_checks':jacchecks,'max_autodiff_vs_analytic_jacobian_error':maxjac,
  'metadata_boundaries':'single-axis included; multi-axis excluded; EA fraction direction tested',
  'test_truth_reads':0,'training_calls':0,'source_outputs':{n:h(OUT/'tables'/n) for n in ['condition_pair_signed_events.csv','condition_frozen_local_gradients.csv']}}
 (OUT/'verification/condition_independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
