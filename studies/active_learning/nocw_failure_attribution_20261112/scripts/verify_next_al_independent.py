#!/usr/bin/env python3
"""Read-only independent field checks of E1 transformations, no new proposal."""
from pathlib import Path
import importlib.util,json,hashlib,math
OUT=Path(__file__).resolve().parents[1]
def main():
 sp=importlib.util.spec_from_file_location('next_pre',OUT/'scripts/next_al_precheck.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 checks=[]
 for seed in [157,887]:
  for L in [525,1005]:
   d=m.owner(seed,(L-333)//32);p=json.loads((d/'input.json').read_text());c=json.loads((d/'contract.json').read_text());p0=json.loads((m.owner(seed,0)/'input.json').read_text());l0=set(m.decode(p0,'observed'))
   assert hashlib.sha256((d/'input.json').read_bytes()).hexdigest()==c['input_sha256']
   original=m.decode(p,'observed');assert set(original)==set(c['L_t_ids']);arms={}
   for arm in ['real','hidden','permuted']:arms[arm],_=m.feedback_arm(p,l0,arm,20261112)
   for arm,z in arms.items():
    assert z['candidates']==p['candidates'] and z['molecules']==p['molecules']
    obs=m.decode(z,'observed');assert set(obs)==set(c['L_t_ids'])
    assert all(obs[i]==original[i] for i in l0)
    if arm=='hidden':
     for i in set(obs)-l0:
      assert all(obs[i][k] is None for k in ['V1_ml','V2_ml','historical_abs_error_ml','premeasurement_error_V1_ml','premeasurement_error_V2_ml'])
     assert not z['memory']['high_error_measured_examples']
    assert not z['memory']['previous_hypotheses'] and not z['memory']['previous_feedback_interpretation'] and not z['memory']['previous_batch_rationale']
    if arm=='real':
     for i in set(obs)-l0:
      assert math.isclose(obs[i]['historical_abs_error_ml'],max(abs(obs[i]['premeasurement_error_V1_ml']),abs(obs[i]['premeasurement_error_V2_ml'])),rel_tol=0,abs_tol=1e-12)
   assert m.visible_evidence_ids(arms['hidden'])==l0
   m.validate_visible_evidence(arms['hidden'],l0)
   try:m.validate_visible_evidence(arms['hidden'],[next(iter(set(original)-l0))])
   except AssertionError:pass
   else:raise AssertionError('Hidden numeric feedback accepted as visible scientific evidence')
   def outcomes(z):return sorted((v['V1_ml'],v['V2_ml']) for i,v in m.decode(z,'observed').items() if i not in l0)
   assert outcomes(arms['real'])==outcomes(arms['permuted'])
   es=[];post=set(original)-l0
   for arm in ['real','permuted']:
    obs=m.decode(arms[arm],'observed');es.append(sum(sum(abs(obs[i][t+'_ml']-obs[i]['premeasurement_prediction_'+t+'_ml']) for t in ['V1','V2'])/2 for i in post)/len(post))
   ratio=es[1]/es[0]
   checks.append({'seed':seed,'L':L,'full_X_and_L0_identical':True,'hidden_response_and_derived_fields_removed':True,
     'old_prose_absent_all_arms':True,'joint_postL333_outcome_multiset_preserved':True,'hidden_numeric_evidence_rejected':True,
     'historical_abs_error_max_convention_checked':True,'independent_permutation_error_ratio':ratio,'warning':ratio>1.5})
 result={'status':'PASS_DATA_TRANSFORM_ONLY','states':checks,'no_api':True,'no_training':True,'no_new_selection':True,
  'scope':'independent packet-field checks; not scientific efficacy','source_script_sha256':hashlib.sha256((OUT/'scripts/next_al_precheck.py').read_bytes()).hexdigest()}
 (OUT/'verification/next_al_independent_data_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
