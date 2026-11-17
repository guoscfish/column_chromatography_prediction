#!/usr/bin/env python3
"""Read-only E1/E2 packet-boundary checks and finite offline mocks.

No LLM transport, acquisition optimizer, label store, optimizer or runner import.
Perturbed packets live in memory only and are never historical experiment truth.
"""
from pathlib import Path
import copy,hashlib,json,math,collections,argparse
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[2]
BASE=ROOT/'studies/active_learning/qgeognn_v2_row_llm_nocw_ablation/revisions'
SOURCES={}
def stable(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def read(p):
 b=p.read_bytes();SOURCES[str(p.relative_to(ROOT))]={'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)};return json.loads(b)
def decode(p,name):return {r[0]:dict(zip(p[name]['columns'],r)) for r in p[name]['rows']}
def encode(p,name,records):p[name]['rows']=[[r.get(k) for k in p[name]['columns']] for r in records.values()]
def owner(seed,r):
 rev='five_seed_nocw_to1005_20261029' if r<21 else 'five_seed_nocw_to1389_20261101'
 return BASE/rev/f'selections/seed_{seed}/llm32_nocw/round_{r:02d}'
def assert_nocw(p):
 assert p['validation_test_records']==0
 assert p['condition_columns']==['PE/EA','Density g/ml','V/ul','loading solvent','Volume of loading solvent/ul']
 assert not any('cw_' in k.lower() for k in p['candidates']['columns'])
 assert not set(p['candidates']['columns'])&{'V1_ml','V2_ml','true_V1_ml','true_V2_ml'}
 c,o=decode(p,'candidates'),decode(p,'observed')
 assert not set(c)&set(o) and len(c)==p['candidate_count'] and len(o)==p['active_label_count']
 assert all(len(r)==len(p['candidates']['columns']) for r in p['candidates']['rows'])
 return c,o
def common_memory(p):
 """Uniform typed memory for all E1 arms; no old prose that leaks real outcomes."""
 x=copy.deepcopy(p)
 x['memory']={'previous_hypotheses':[],'previous_feedback_interpretation':'',
  'previous_batch_rationale':'','previous_selected_ids':list(p['memory']['previous_selected_ids']),
  'recent_outcomes':[],'high_error_measured_examples':[],
  'memory_policy':'Controlled typed feedback only. No inherited free-text scientific claims in any arm.'}
 return x
def error(y,pred):return None if y is None or pred is None else float(pred)-float(y)
def update_derived(rec):
 e=[]
 for target in ['V1','V2']:
  v=error(rec.get(target+'_ml'),rec.get('premeasurement_prediction_'+target+'_ml'))
  rec['premeasurement_error_'+target+'_ml']=v
  if v is not None:e.append(abs(v))
 # Preserve the repository's historical error convention: max over endpoints.
 rec['historical_abs_error_ml']=max(e) if e else None
def visible_evidence_ids(packet):
 return {i for i,r in decode(packet,'observed').items() if r.get('V1_ml') is not None and r.get('V2_ml') is not None}
def validate_visible_evidence(packet,ids):
 assert set(ids)<=visible_evidence_ids(packet),'evidence row has no visible measured response'
 return True
def feedback_arm(packet,l0,arm,permutation_seed):
 assert arm in ['real','hidden','permuted']
 x=common_memory(packet);_,obs=assert_nocw(x);post=set(obs)-set(l0)
 assert set(l0)<=set(obs)
 mapping={i:i for i in post}
 if arm=='permuted':
  groups=collections.defaultdict(list)
  for i in sorted(post):
   r=obs[i];groups[r['molecule_id'],str(r['conditions'][3])].append(i)
  for key,ids in sorted(groups.items()):
   # Group by molecule + loading solvent, then two predicted-response rank bands.
   # These are premeasurement predictions, not revealed truth used to choose groups.
   ids=sorted(ids,key=lambda i:(float(obs[i].get('premeasurement_prediction_V2_ml') or 0),i))
   bands=[ids] if len(ids)<4 else [ids[:len(ids)//2],ids[len(ids)//2:]]
   for band in bands:
    if len(band)<2:continue
    band=sorted(band,key=lambda i:stable([permutation_seed,packet['seed'],packet['round'],i]))
    for j,i in enumerate(band):mapping[i]=band[(j+1)%len(band)]
 original=copy.deepcopy(obs)
 for i in post:
  for target in ['V1','V2']:
   obs[i][target+'_ml']=None if arm=='hidden' else original[mapping[i]][target+'_ml']
  update_derived(obs[i])
 # Initial response records are retained identically, including their history columns.
 assert all(obs[i]==original[i] for i in l0)
 recent=[i for i in packet['memory']['previous_selected_ids'] if i in obs]
 def fact(i):
  return {k:obs[i].get(k) for k in ['id','molecule_id','conditions','V1_ml','V2_ml',
    'premeasurement_prediction_V1_ml','premeasurement_prediction_V2_ml',
    'premeasurement_error_V1_ml','premeasurement_error_V2_ml']}
 x['memory']['recent_outcomes']=[fact(i) for i in recent]
 ranked=sorted((i for i in post if obs[i].get('historical_abs_error_ml') is not None),
  key=lambda i:(-obs[i]['historical_abs_error_ml'],i))[:8]
 x['memory']['high_error_measured_examples']=[fact(i) for i in ranked]
 encode(x,'observed',obs)
 x['packet_sha256']=stable({k:v for k,v in x.items() if k!='packet_sha256'})
 assert_nocw(x)
 assert x['candidates']==packet['candidates'] and x['molecules']==packet['molecules']
 assert set(mapping)==post and set(mapping.values())==post
 # Only aggregate perturbation diagnostics are exported; no synthetic response rows.
 return x,{'post_L333_rows':len(post),'permuted_rows':sum(i!=j for i,j in mapping.items()),
  'hidden_response_rows':len(post) if arm=='hidden' else 0,'initial_responses_unchanged':True,
  'all_feedback_source_ids_time_legal':set(mapping.values())<=post,'typed_memory_common_policy':True,
  'memory_prose_removed_in_all_E1_arms':True}
def residual_mae(p):
 _,o=assert_nocw(p);es=[]
 for r in o.values():
  vals=[abs(r[k]) for k in ['premeasurement_error_V1_ml','premeasurement_error_V2_ml'] if r.get(k) is not None]
  if vals:es.append(sum(vals)/len(vals))
 return sum(es)/len(es) if es else None
def joint_checks(original,arms):
 c,o=assert_nocw(original)
 for name,p in arms.items():
  cc,oo=assert_nocw(p);assert set(c)==set(cc) and set(o)==set(oo)
  for i in o:
   assert all(o[i][k]==oo[i][k] for k in ['id','molecule_id','conditions'])
  assert p['candidates']==original['candidates'] and p['molecules']==original['molecules']
 return True
def mock_accept(packet,archived_selection):
 """Replay an already accepted ID set; never propose a new acquisition."""
 c,o=assert_nocw(packet);ids=[x['id'] for x in archived_selection['choices']]
 assert len(ids)==len(set(ids))==32 and set(ids)<=set(c) and not set(ids)&set(o)
 return {'response_source':'offline replay of historical accepted IDs; no new selection',
  'packet_sha256':packet['packet_sha256'],'selected_ids_sha256':stable(ids),'count':len(ids),
  'validation_scope':'choices IDs only; historical prose/hypotheses are not revalidated',
  'full_scientific_response_validation':False}
def main():
 rule_path=OUT/'verification/next_al_precheck_plan.json'
 rule={'status':'FIXED_BEFORE_PERTURBATION_DIAGNOSTICS','states':[[s,l] for s in [157,887] for l in [525,1005]],
  'E1_arms':['real','hidden','permuted'],'permutation_seed':20261112,
  'permutation':'Joint V1/V2 cyclic permutation of postL333 known outcomes within molecule + loading-solvent groups, split into two premeasurement V2 rank bands if >=4 rows. SHA order, fixed shift1. L333 unaltered. No response-based retry.',
  'all_E1_memory':'Uniform typed reconstruction, old scientific prose absent in every arm to prevent truthful-memory contradictions',
  'contradiction_gate':'If permutation/real historical prediction MAE ratio >1.5, flag the shuffled arm for explicit contradiction review; do not reroll until passing. This is engineering eligibility, not a chemical theorem.',
  'E2':'All NoCW candidate packets unchanged. Geometry is an extra post-proposal diagnostic; identical identity/condition feedback for critic and placebo.',
  'mock_only':True,'new_acquisition':False,'llm_calls':0,'training_calls':0}
 if rule_path.exists():assert read(rule_path)==rule
 else:rule_path.write_text(json.dumps(rule,ensure_ascii=False,indent=2)+'\n')
 rows=[];checks=[]
 for seed,L in rule['states']:
  rnd=(L-333)//32;d=owner(seed,rnd);packet=read(d/'input.json');contract=read(d/'contract.json')
  sel=read(d/'selection.json') if (d/'selection.json').exists() else None
  assert SOURCES[str((d/'input.json').relative_to(ROOT))]['sha256']==contract['input_sha256']
  assert packet['active_label_count']==L and packet['round']==rnd
  c,o=assert_nocw(packet);assert set(o)==set(contract['L_t_ids']) and set(c)==set(contract['U_t_ids'])
  l0packet=read(owner(seed,0)/'input.json');l0=set(decode(l0packet,'observed'))
  arms={};audit={}
  for arm in rule['E1_arms']:arms[arm],audit[arm]=feedback_arm(packet,l0,arm,rule['permutation_seed'])
  joint_checks(packet,arms)
  assert visible_evidence_ids(arms['hidden'])==l0
  assert visible_evidence_ids(arms['real'])==set(o)
  validate_visible_evidence(arms['hidden'],l0)
  hidden_id=next(iter(set(o)-l0))
  try:validate_visible_evidence(arms['hidden'],[hidden_id])
  except AssertionError:pass
  else:raise AssertionError('hidden outcome accepted as measured scientific evidence')
  real_error=residual_mae(arms['real']);perm_error=residual_mae(arms['permuted'])
  ratio=perm_error/real_error if real_error else None
  for arm,p in arms.items():
   replay=mock_accept(p,sel) if sel is not None else {'response_source':'input-only check: no accepted selection at this staged state; response replay skipped','selected_ids_sha256':None}
   rows.append({'seed':seed,'available_budget':L,'round':rnd,'arm':arm,'checkpoint_sha256':contract['checkpoint_sha256'],
     'candidate_payload_sha256':stable(p['candidates']),'input_sha256':stable(p),'original_memory_sha256':stable(packet['memory']),
     **audit[arm],'historical_prediction_mae_ml':residual_mae(p),
     'permuted_to_real_mae_ratio':ratio if arm=='permuted' else None,
     'obvious_mismatch_review_required':bool(ratio and ratio>1.5) if arm=='permuted' else False,
     'mock_response_source':replay['response_source'],'mock_selected_ids_sha256':replay['selected_ids_sha256'],
     'full_scientific_response_validation':False,'visible_measured_evidence_rows':len(visible_evidence_ids(p))})
  # E2 four-arm mock uses one archived accepted proposal and a typed condition ledger.
  # It verifies sameness and a permitted geometry attachment, not critic efficacy.
  condition_ledger={'selected_ids_sha256':stable([x['id'] for x in sel['choices']]) if sel is not None else None,'identity_check':'historical accepted U IDs when available; no proposal invented',
    'condition_axes':'derived from existing metadata only; identical in critic/placebo'}
  base={'packet':packet,'shared_condition_feedback':condition_ledger,'extra_geometry':None}
  e2={arm:copy.deepcopy(base) for arm in ['original','placebo','critic','raw_lcmd_mock']}
  e2['critic']['extra_geometry']={'kind':'mock only','checkpoint_sha256':contract['checkpoint_sha256'],
    'metrics':['distance_to_L','batch_nearest_neighbor','random_reference_percentile'],
    'aggregate_total_acquisition_score':None,'real_measurement_not_included':True}
  assert len({stable(v['packet']) for v in e2.values()})==1
  assert e2['critic']['shared_condition_feedback']==e2['placebo']['shared_condition_feedback']
  assert e2['original']['extra_geometry'] is None and e2['placebo']['extra_geometry'] is None
  checks.append({'seed':seed,'L':L,'checkpoint_sha256':contract['checkpoint_sha256'],'E1_shared_identity_X_checks':True,
   'E2_same_NoCW_candidate_packet':True,'E2_critic_only_geometry_attachment':True,'E2_shared_condition_feedback':True,
   'new_raw_lcmd_batch_computed':False,'response_replay_available':sel is not None,'all_replayed_IDs_are_historical_accepted':True})
 # Boundary tests use copies of authorized packets and deliberately invalid metadata.
 broken=copy.deepcopy(packet);broken['candidates']['columns'].append('true_V1_ml')
 try:assert_nocw(broken)
 except AssertionError:pass
 else:raise AssertionError('candidate truth field accepted')
 broken=copy.deepcopy(packet);broken['validation_test_records']=1
 try:assert_nocw(broken)
 except AssertionError:pass
 else:raise AssertionError('validation/test packet accepted')
 import csv
 with (OUT/'tables/next_al_precheck.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (OUT/'verification/next_al_precheck.json').write_text(json.dumps({'status':'PASS_MOCK_AND_BOUNDARY_ONLY','states':checks,
  'E1_permutation_warning_states':sum(x['obvious_mismatch_review_required'] for x in rows),
  'new_llm_calls':0,'new_selections':0,'new_fits':0,'private_packets_exported':False,
  'does_not_establish':'LLM response changes, model noise, critic benefit or feedback causality'},indent=2)+'\n')
 (OUT/'verification/next_al_inputs.json').write_text(json.dumps(list(SOURCES.values()),indent=2)+'\n')
 print(json.dumps({'status':'PASS_MOCK_ONLY','states':len(checks),'E1_payload_variants':len(rows),'permutation_warning_states':sum(x['obvious_mismatch_review_required'] for x in rows)}))
if __name__=='__main__':main()
