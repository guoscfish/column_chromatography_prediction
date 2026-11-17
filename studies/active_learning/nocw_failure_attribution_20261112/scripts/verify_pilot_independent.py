#!/usr/bin/env python3
"""Independent fixed-pilot verification. No optimizer, fit, selector, or raw truth reader.

Partial mode audits completed slots while workers run. Complete mode additionally
cross-checks final public paired/group summaries. Optional H forwards evaluate
already frozen weights against already authorized held-out observations.
"""
from pathlib import Path
import argparse,collections,hashlib,json,lzma,math,random,sys
from decimal import Decimal
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd

def js(p):return json.loads(Path(p).read_text())
def stable(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
FIELDS=['PE/EA','Density g/ml','V/ul','loading solvent','Volume of loading solvent/ul']
def normalize(v,k):
 if k=='PE/EA':
  a,b=str(v).split('/');return Fraction(Decimal(b))/(Fraction(Decimal(a))+Fraction(Decimal(b)))
 return str(v) if k=='loading solvent' else Decimal(str(v))
def key(meta):return (meta['smiles'],*[normalize(meta['conditions'][k],k) for k in FIELDS])
def state_truth(state):
 """Only current legally-observed packet/accepted-feedback labels, with explicit ID allowlist."""
 cp=ROOT/state['contract_source'];contract=js(cp);packet=js(cp.parent/'input.json')
 assert sha(cp.parent/'input.json')==contract['input_sha256'] and packet['validation_test_records']==0
 allowed=set(state['L_ids']);prior=set(contract['L_t_ids']);assert prior<=allowed
 cols=packet['observed']['columns'];idpos=cols.index('id');v1=cols.index('V1_ml');v2=cols.index('V2_ml');truth={}
 assert {r[idpos] for r in packet['observed']['rows']}==prior
 for row in packet['observed']['rows']:
  i=str(row[idpos]);assert i in allowed;truth[i]=(float(row[v1]),float(row[v2]))
 if prior!=allowed:
  freeze=js(cp.parent/'batch_freeze.json');feedback=js(cp.parent/'feedback.json')
  assert prior.isdisjoint(freeze['batch_ids']) and prior|set(freeze['batch_ids'])==allowed
  assert [r['candidate_id'] for r in feedback['records']]==freeze['batch_ids']
  for r in feedback['records']:
   i=r['candidate_id'];assert i in allowed and i not in truth;truth[i]=(float(r['true_V1_ml']),float(r['true_V2_ml']))
 assert set(truth)==allowed
 return truth
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--complete',action='store_true');ap.add_argument('--forward-H',action='store_true');args=ap.parse_args()
 design=js(OUT/'verification/pilot_protocol_freeze.json');qual=js(OUT/'verification/pilot_qualification.json')
 assert design['maximum_successful_fits']==16 and design['new_parameters']==0 and design['total_parameters_per_arm']==458952
 assert design['outer_seeds']==[157,887] and design['available_budgets']==[525,1005] and design['training_replicates']==[0,1]
 fixed=[('scripts/model_pilot.py','script_sha256'),('MODEL_PILOT_PROTOCOL.md','protocol_sha256'),
  ('verification/pilot_metadata_rule.json','metadata_rule_sha256'),('verification/pilot_qualification.json','holdout_qualification_sha256'),
  ('verification/C_gate.json','C_gate_sha256'),('verification/pilot_root_release.json','root_release_sha256')]
 for p,k in fixed:assert sha(OUT/p)==design[k],p
 for p,h in design['source_hashes'].items():assert sha(ROOT/p)==h,p
 for p,h in design['data_sources'].items():assert sha(ROOT/p)==h,p
 expected={(s['seed'],s['available_budget'],a,r) for s in qual['states'] if s['qualification'] for a in ['baseline','interaction'] for r in [0,1]}
 assert len(expected)==16
 # Only identity and covariate columns are parsed. No canonical response column is requested.
 feature_path=ROOT/'experiments/e0_4g_baseline/canonical_4g.csv'
 features=pd.read_csv(feature_path,usecols=['sample_id','canonical_smiles']+FIELDS,dtype=str,keep_default_na=False)
 assert set(features.columns)=={'sample_id','canonical_smiles',*FIELDS}
 meta={r['sample_id']:{'smiles':r['canonical_smiles'],'conditions':{k:r[k] for k in FIELDS}} for r in features.to_dict('records')}
 snap=json.loads(lzma.decompress((ROOT/'studies/active_learning/nocw_free_selection_diagnostics_20261101/snapshot_inputs.json.xz').read_bytes()))
 states={(s['seed'],s['available_budget']):s for s in qual['states']};isolation=[];truths={}
 for (seed,budget),s in states.items():
  L,H,T,L0=map(set,[s['L_ids'],s['H_ids'],s['train_ids'],s['L0_ids']]);role=snap['partitions'][str(seed)]
  V={i for i,r in role.items() if r['role']=='validation'};test={i for i,r in role.items() if r['role']=='test'}
  assert len(L)==budget and L==T|H and T.isdisjoint(H) and L0<=T
  assert not L&(V|test) and not H&L0 and len(T)==s['actual_train_rows'] and len(H)==s['H_rows']
  assert all(role[i]['role'] in ['l0','u0'] for i in L)
  assert not {key(meta[i]) for i in H}&{key(meta[i]) for i in T|V|L0}
  assert len(s['groups'])==10 and len({g['molecule'] for g in s['groups']})==10
  group_rows=set()
  for g in s['groups']:
   assert set(g['ids'])<=H and group_rows.isdisjoint(g['ids']);group_rows.update(g['ids'])
   for e in g['pair_edges']:
    a,b=e['a'],e['b'];assert a in g['ids'] and b in g['ids']
    assert meta[a]['smiles']==meta[b]['smiles']==g['molecule']
    changed=[k for k in FIELDS if normalize(meta[a]['conditions'][k],k)!=normalize(meta[b]['conditions'][k],k)]
    assert changed==[e['axis']]
  assert group_rows==H
  context=js(ROOT/f'studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_{seed}/context.json')['contract']
  prep=context['preprocessing'];assert prep['target_scale_fit_role']=='l0_labels_only' and prep['target_scale_fit_rows']==333
  assert prep['target_scale_fit_ids_hash']==stable(sorted(L0))
  truths[seed,budget]=state_truth(s)
  isolation.append({'seed':seed,'available_budget':budget,'actual_train_rows':len(T),'H_rows':len(H),'H_groups':10,
   'H_L333_train_validation_recorded_key_disjoint':True,'H_molecules_unseen_in_train':len({meta[i]['smiles'] for i in H}-{meta[i]['smiles'] for i in T})})
 paths=sorted((OUT/'runtime').glob('seed_*/L*/*/rep_*/fit_audit.json'));assert len(paths)<=16
 if args.complete:assert len(paths)==16,('Incomplete',len(paths))
 audits=[];slots=set();valerr=0.;Hmetricerr=0.;historyerr=0.;pairerr=0.;grouperr=0.;predforwarderr=0.;scalar_count=0;groups_count=0
 rawgroups=[];rawpair_cache={}
 for p in paths:
  audit=js(p);c=audit['fit_contract'];slot=c['seed'],c['available_budget'],c['arm'],c['replicate'];assert slot in expected and slot not in slots;slots.add(slot);audits.append(audit)
  assert c['design_sha256']==stable(design) and audit['parameter_count']==458952 and audit['status']=='SUCCESS'
  assert c['initialization_seed']==c['seed']+2000003+c['replicate']*1000033
  assert c['training_seed']==c['seed']+3000017+c['replicate']*1000033
  s=states[c['seed'],c['available_budget']];assert c['actual_train_rows']==s['actual_train_rows']
  assert c['train_ids_sha256']==stable(s['train_ids']) and c['H_ids_sha256']==stable(s['H_ids'])
  label=audit['label_access'];assert label['test_numeric_targets_read']==label['unacquired_numeric_targets_read']==0
  assert label['heldout']['rows']==len(s['H_ids']) and label['training']['rows']==len(s['train_ids'])
  assert label['all_graphs_zero_before_assignment'] and label['test_and_H_graphs_zero_during_fit']
  assert label['validation']['numeric_target_rows_parsed']==416 and label['validation']['roles']==['validation']
  roles=snap['partitions'][str(c['seed'])];valid_ids=sorted(i for i,r in roles.items() if r['role']=='validation')
  assert label['validation']['allowed_ids_sha256']==stable(valid_ids)
  assert label['training']['ids_sha256']==stable(sorted(s['train_ids'])) and label['heldout']['ids_sha256']==stable(sorted(s['H_ids']))
  rt=p.parent
  for f,k in [('best.pt','checkpoint_sha256'),('history.csv','history_sha256'),('heldout_pairs.csv','H_pair_file_sha256')]:assert sha(rt/f)==audit[k]
  scales=js(ROOT/f"studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_{c['seed']}/context.json")['contract']['preprocessing']['target_scales']
  for pop,store in [('validation',audit['validation_metrics']),('H',audit['H_point_metrics'])]:
   score=math.sqrt(((store['V1_rmse']/scales['V1'])**2+(store['V2_rmse']/scales['V2'])**2)/2)
   er=abs(score-store['combined_normalized_rmse']);assert er<1e-12
   if pop=='validation':valerr=max(valerr,er)
   else:Hmetricerr=max(Hmetricerr,er)
  history=pd.read_csv(rt/'history.csv',float_precision='round_trip');assert len(history)==audit['epochs_run']<=1000
  assert list(history.epoch)==list(range(1,len(history)+1))
  for r in history.itertuples():
   calc=math.sqrt(((r.validation_V1_rmse/scales['V1'])**2+(r.validation_V2_rmse/scales['V2'])**2)/2)
   historyerr=max(historyerr,abs(calc-r.validation_selection_score));assert abs(calc-r.validation_selection_score)<1e-12
  best=int(history.loc[history.validation_selection_score.idxmin(),'epoch']);assert best==audit['best_epoch']
  assert abs(history.validation_selection_score.min()-audit['validation_metrics']['combined_normalized_rmse'])<1e-12
  pr=pd.read_csv(rt/'heldout_pairs.csv',float_precision='round_trip');gr=pd.read_csv(rt/'heldout_group_metrics.csv',float_precision='round_trip')
  assert len(pr)==2*sum(len(g['pair_edges']) for g in s['groups']);rawpair_cache[slot]=pr
  assert set(pr.a)|set(pr.b)<=set(s['H_ids']) and not pr.duplicated(['group_id','axis','target','a','b']).any()
  true=truths[c['seed'],c['available_budget']]
  for r in pr.itertuples():
   t=0 if r.target=='V1' else 1;dy=true[r.b][t]-true[r.a][t];delta=r.predicted_difference_ml
   for x,y in [(dy,r.observed_difference_ml),(delta-dy,r.signed_difference_error_ml),(abs(delta-dy),r.absolute_difference_error_ml),
               (abs(dy),r.observed_abs_difference_ml),(abs(delta),r.predicted_abs_difference_ml)]:
    er=abs(x-y);pairerr=max(pairerr,er);assert er<1e-10;scalar_count+=1
   assert r.direction_match==int(np.sign(dy)==np.sign(delta))
  metriccols=['signed_difference_error_ml','absolute_difference_error_ml','direction_match','observed_abs_difference_ml','predicted_abs_difference_ml']
  for keyg,g in pr.groupby(['axis','target','group_id']):
   row=gr[(gr.axis==keyg[0])&(gr.target==keyg[1])&(gr.group_id==keyg[2])];assert len(row)==1;row=row.iloc[0]
   assert row.pair_count==len(g)
   for k in metriccols:
    er=abs(g[k].mean()-row[k]);grouperr=max(grouperr,er);assert er<1e-10
   groups_count+=1
  rawgroups.extend(gr.to_dict('records'))
 # Both arms must share a whole initialized state and target/row identities.
 initpairs=0
 for seed,budget in states:
  for rep in [0,1]:
   a=[x for x in audits if x['fit_contract']['seed']==seed and x['fit_contract']['available_budget']==budget and x['fit_contract']['replicate']==rep]
   if len(a)==2:assert a[0]['initialization_state_sha256']==a[1]['initialization_state_sha256'];initpairs+=1
 if args.complete:assert slots==expected and initpairs==8
 fresh_init_count=verify_fresh_initializations(audits)
 # Frozen forwards are deferred until fitting is complete to avoid resource contention.
 if args.forward_H:
  assert args.complete,'Use frozen forward after complete fit ledger only'
  predforwarderr=verify_H_forwards(audits,states,truths,rawpair_cache)
 summary_checks=verify_public_summaries(audits,rawgroups) if args.complete else {'status':'not_checked_until_complete'}
 result={'status':'PASS' if args.complete else 'PARTIAL_COMPLETED_FITS_VERIFIED','successful_fits_checked':len(audits),'successful_fit_cap':16,
  'distinct_slots':len(slots),'initialization_arm_pairs':initpairs,'frozen_design_and_source_hashes_unchanged':True,
  'fresh_initialized_models_reconstructed_without_fitting':fresh_init_count,
  'state_isolation_checks':isolation,'validation_formula_max_error':valerr,'H_point_formula_max_error':Hmetricerr,
  'all_history_formula_max_error':historyerr,'pair_scalar_checks':scalar_count,'pair_scalar_max_error':pairerr,
  'H_groups_reaggregated':groups_count,'H_group_max_error':grouperr,'frozen_H_forward_max_error':predforwarderr,
  'frozen_H_forwards_performed':args.forward_H,'public_summaries':summary_checks,
  'canonical_truth_columns_numericized_by_this_check':0,'observed_truth_source':'state L allowlist; accepted terminal feedback where needed',
  'test_truth_reads':0,'unacquired_truth_reads':0,'training_calls':0,'llm_calls':0,
  'limitations':['Hashes/metrics verify execution and arithmetic, not hypothesis causality','Two outer seeds and two training replicates are not16 independent studies','H contains both seen and unseen molecules; cross-budget H differs'],
  'verification_script_sha256':sha(Path(__file__))}
 name='pilot_independent_verification.json' if args.complete else 'pilot_independent_partial_verification.json'
 (OUT/'verification'/name).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))

def verify_fresh_initializations(audits):
 """Independently reconstruct initial parameters; this creates no optimizer or fit."""
 import torch
 from qgeognn_al.models import build_predictor
 from qgeognn_al.schemas.conditions import ConditionNormalization
 torch.set_num_threads(2);checked={}
 for audit in audits:
  c=audit['fit_contract'];k=c['seed'],c['replicate']
  if k not in checked:
   seed=c['initialization_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
   ctx=js(ROOT/f"studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_{c['seed']}/context.json")['contract']
   model=build_predictor(ConditionNormalization(**ctx['normalization']))
   assert sum(p.numel() for p in model.parameters())==458952
   assert torch.count_nonzero(model.condition_branch.output.weight).item()==0 and torch.count_nonzero(model.condition_branch.output.bias).item()==0
   payload=b''.join(name.encode()+value.detach().cpu().numpy().tobytes() for name,value in model.state_dict().items())
   checked[k]=hashlib.sha256(payload).hexdigest()
  assert checked[k]==audit['initialization_state_sha256']
 return len(checked)

def verify_H_forwards(audits,states,truths,rawpair_cache):
 import torch
 from torch_geometric.nn import global_add_pool
 from torch.nn import functional as F
 from qgeognn_al.models import build_predictor
 from qgeognn_al.schemas.conditions import ConditionNormalization
 from qgeognn_al.training.predictor import loader_pair
 torch.set_num_threads(2);maximum=0.;graphs={}
 for seed in [157,887]:
  graphs[seed]=torch.load(ROOT/f'studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_{seed}/scrubbed_graphs.pt',weights_only=False)
  assert all(torch.count_nonzero(a.y).item()==0 for a in graphs[seed][0])
 for audit in audits:
  c=audit['fit_contract'];slot=c['seed'],c['available_budget'],c['arm'],c['replicate'];seed,budget,arm,rep=slot
  state=states[seed,budget];H=state['H_ids'];true=truths[seed,budget]
  ctx=js(ROOT/f'studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_{seed}/context.json')['contract']
  model=build_predictor(ConditionNormalization(**ctx['normalization']));ck=torch.load(OUT/f'runtime/seed_{seed}/L{budget}/{arm}/rep_{rep}/best.pt',map_location='cpu',weights_only=False)
  model.load_state_dict(ck['model_state_dict']);model.eval()
  assert ck['fit_contract']==c and ck['epoch']==audit['best_epoch'] and ck['parameter_count']==458952
  partition=pd.read_csv(ROOT/f'studies/active_learning/qgeognn_v2_row_sequential_b32/splits/row_seed_{seed}.csv');ix=dict(zip(partition.sample_id,partition.canonical_index))
  atom,angle=graphs[seed];positions=[int(ix[i]) for i in H]
  a=next(iter(loader_pair(atom,angle,positions,2048)[0]));b=next(iter(loader_pair(atom,angle,positions,2048)[1]))
  with torch.no_grad():
   p=global_add_pool(model.backbone(a,b),a.batch);r,_=model.condition_branch(a)
   z=p+r if arm=='baseline' else p+r+r*torch.tanh(F.layer_norm(p,(128,),weight=None,bias=None,eps=1e-5))
   prediction=torch.clamp(model.head(z),0,1e8).cpu().numpy()
  values={i:prediction[j] for j,i in enumerate(H)}
  for row in rawpair_cache[slot].itertuples():
   q=1 if row.target=='V1' else 4;d=float(values[row.b][q]-values[row.a][q]);er=abs(d-row.predicted_difference_ml);maximum=max(maximum,er);assert er<1e-5
  for target,q in [('V1',1),('V2',4)]:
   t=0 if target=='V1' else 1;y=np.asarray([true[i][t] for i in H],np.float32);e=prediction[:,q]-y
   for k,v in [('mae',float(np.abs(e).mean())),('rmse',float(np.sqrt(np.mean(e**2))))]:
    er=abs(v-audit['H_point_metrics'][target+'_'+k]);maximum=max(maximum,er);assert er<1e-5
 return maximum

def verify_public_summaries(audits,groups):
 fits=pd.read_csv(OUT/'tables/pilot_fit_results.csv',float_precision='round_trip');assert len(fits)==len(audits)==16
 paired=pd.read_csv(OUT/'tables/pilot_paired_validation.csv',float_precision='round_trip');assert len(paired)==8
 maximum=0.
 for r in paired.itertuples():
  a=[x for x in audits if x['fit_contract']['seed']==r.seed and x['fit_contract']['available_budget']==r.available_budget and x['fit_contract']['replicate']==r.replicate]
  b=next(x for x in a if x['fit_contract']['arm']=='baseline');i=next(x for x in a if x['fit_contract']['arm']=='interaction')
  for k in ['combined_normalized_rmse','V1_rmse','V1_mae','V1_r2','V2_rmse','V2_mae','V2_r2']:
   v=i['validation_metrics'][k]-b['validation_metrics'][k];w=getattr(r,'validation_'+k+'_delta_interaction_minus_baseline');maximum=max(maximum,abs(v-w));assert abs(v-w)<1e-12
 gm=pd.DataFrame(groups);cond=pd.read_csv(OUT/'tables/pilot_condition_metrics.csv',float_precision='round_trip')
 keys=['seed','available_budget','arm','replicate','axis','target'];checked=0
 for keyg,g in gm.groupby(keys):
  mask=np.ones(len(cond),bool)
  for k,v in zip(keys,keyg):mask&=cond[k].eq(v)
  row=cond[mask];assert len(row)==1;row=row.iloc[0]
  for out,src in [('group_mean_abs_difference_error_ml','absolute_difference_error_ml'),('group_mean_signed_difference_error_ml','signed_difference_error_ml'),
   ('group_mean_direction_match','direction_match'),('group_mean_observed_abs_difference_ml','observed_abs_difference_ml'),('group_mean_predicted_abs_difference_ml','predicted_abs_difference_ml')]:
   er=abs(g[src].mean()-row[out]);maximum=max(maximum,er);assert er<1e-10;checked+=1
 joint=verify_joint_aggregation(audits,groups)
 return {'status':'PASS','paired_validation_rows':len(paired),'condition_scalar_checks':checked,'max_error':maximum,
  'joint_state_outer_aggregation':joint}

def verify_joint_aggregation(audits,groups):
 """Rebuild Q/state/outer effects with scalar loops, separately from pandas summaries."""
 slots={}
 for a in audits:
  c=a['fit_contract'];k=(c['seed'],c['available_budget'],c['arm'],c['replicate']);assert k not in slots;slots[k]=a
 assert len(slots)==16
 grouped=collections.defaultdict(list)
 for r in groups:
  k=(int(r['seed']),int(r['available_budget']),r['arm'],int(r['replicate']),r['axis'],r['target'])
  grouped[k].append(r)
 scales={s:js(ROOT/f'studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_{s}/context.json')['contract']['preprocessing']['target_scales'] for s in [157,887]}
 Q={}
 for seed,budget,arm,rep in slots:
  errors={}
  for t in ['V1','V2']:
   rows=grouped[seed,budget,arm,rep,'V/ul',t]
   assert len({r['group_id'] for r in rows})==len(rows)>0
   assert len({r['molecule_sha256'] for r in rows})==len(rows),'This pilot assumes one component per molecule'
   # Independent equal-component mean; no pair multiplicity weight.
   errors[t]=math.fsum(float(r['absolute_difference_error_ml']) for r in rows)/len(rows)
  Q[seed,budget,arm,rep]=.5*(errors['V1']/scales[seed]['V1']+errors['V2']/scales[seed]['V2'])
 states=[]
 for seed in [157,887]:
  for budget in [525,1005]:
   b=[slots[seed,budget,'baseline',r]['validation_metrics']['combined_normalized_rmse'] for r in [0,1]]
   x=[slots[seed,budget,'interaction',r]['validation_metrics']['combined_normalized_rmse'] for r in [0,1]]
   mb,mx=math.fsum(b)/2,math.fsum(x)/2;diff=[x[i]-b[i] for i in range(2)]
   qb=math.fsum(Q[seed,budget,'baseline',r] for r in [0,1])/2
   qx=math.fsum(Q[seed,budget,'interaction',r] for r in [0,1])/2
   states.append({'seed':seed,'available_budget':budget,'actual_train_rows':slots[seed,budget,'baseline',0]['actual_train_rows'],
    'baseline_nrmse_mean':mb,'baseline_nrmse_sd':math.sqrt(math.fsum((v-mb)**2 for v in b)),
    'interaction_nrmse_mean':mx,'interaction_nrmse_sd':math.sqrt(math.fsum((v-mx)**2 for v in x)),
    'paired_delta_mean':math.fsum(diff)/2,'paired_delta_min':min(diff),'paired_delta_max':max(diff),
    'baseline_Q':qb,'interaction_Q':qx,'Q_delta_interaction_minus_baseline':qx-qb,'row_relative_change':mx/mb-1})
 public_state=pd.read_csv(OUT/'tables/pilot_state_summary.csv',float_precision='round_trip')
 public_joint=pd.read_csv(OUT/'tables/pilot_joint_interpretation.csv',float_precision='round_trip')
 assert len(public_state)==len(public_joint)==4
 maximum=0.;scalar_checks=0
 state_fields=['actual_train_rows','baseline_nrmse_mean','baseline_nrmse_sd','interaction_nrmse_mean','interaction_nrmse_sd','paired_delta_mean','paired_delta_min','paired_delta_max']
 joint_fields=['baseline_Q','interaction_Q','Q_delta_interaction_minus_baseline','row_relative_change']
 for s in states:
  for frame,fields in [(public_state,state_fields),(public_joint,state_fields+joint_fields)]:
   row=frame[(frame.seed==s['seed'])&(frame.available_budget==s['available_budget'])];assert len(row)==1;row=row.iloc[0]
   for k in fields:
    er=abs(float(row[k])-s[k]);maximum=max(maximum,er);scalar_checks+=1;assert er<1e-12,(s['seed'],s['available_budget'],k,er)
 outer=[]
 for seed in [157,887]:
  g=[s for s in states if s['seed']==seed]
  outer.append({'seed':seed,'row_delta':math.fsum(s['paired_delta_mean'] for s in g)/2,
   'Q_delta':math.fsum(s['Q_delta_interaction_minus_baseline'] for s in g)/2})
 row_pass=(all(s['row_delta']<0 for s in outer) and sum(s['paired_delta_mean']<0 for s in states)>=3 and all(s['row_relative_change']<=.01 for s in states))
 Q_pass=all(s['Q_delta']<0 for s in outer) and sum(s['Q_delta_interaction_minus_baseline']<0 for s in states)>=3
 expected_status='STABLE_JOINT_SUPPORT' if row_pass and Q_pass else 'NO_STABLE_JOINT_SUPPORT_FOR_THIS_SINGLE_CHANGE'
 verdict=js(OUT/'verification/pilot_joint_interpretation.json')
 assert verdict['status']==expected_status and verdict['row_criterion_pass']==row_pass and verdict['volume_Q_criterion_pass']==Q_pass
 assert verdict['row_improved_states']==sum(s['paired_delta_mean']<0 for s in states)
 assert verdict['Q_improved_states']==sum(s['Q_delta_interaction_minus_baseline']<0 for s in states)
 assert len(verdict['outer_seed_effects'])==2
 for s in outer:
  row=next(x for x in verdict['outer_seed_effects'] if x['seed']==s['seed'])
  for k in ['row_delta','Q_delta']:
   er=abs(float(row[k])-s[k]);maximum=max(maximum,er);scalar_checks+=1;assert er<1e-12
 plan=js(OUT/'verification/pilot_interpretation_plan.json')
 assert 'NOT_PREREGISTERED' in plan['status'] and 'after training started' in plan['timing_disclosure']
 return {'status':'PASS','Q_fit_slots':len(Q),'state_rows':len(states),'outer_rows':len(outer),'aggregate_scalar_checks':scalar_checks,
  'max_aggregate_error':maximum,'independent_joint_status':expected_status,'row_criterion_pass':row_pass,'condition_Q_criterion_pass':Q_pass,
  'interpretation_rule_scope':'Descriptive rule written after initial intervention validation results; verified arithmetic, not preregistered efficacy inference',
  'outer_effects_independently_recomputed':outer}
if __name__=='__main__':main()
