#!/usr/bin/env python3
"""Independent scalar/group checks of the fixed 16-fit pilot; no training."""
from pathlib import Path
import collections,csv,hashlib,json,math
OUT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readcsv(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def main():
 protocol=json.loads((OUT/'verification/pilot_protocol_freeze.json').read_text())
 q=json.loads((OUT/'verification/pilot_qualification.json').read_text())
 expected={(s['seed'],s['available_budget'],arm,rep) for s in q['states'] if s['qualification'] for arm in ['baseline','interaction'] for rep in [0,1]}
 audits=list((OUT/'runtime').glob('seed_*/L*/*/rep_*/fit_audit.json'));assert len(audits)==len(expected)<=16
 maxerr=0.;checks=0;slots=set();init=collections.defaultdict(set)
 for p in audits:
  a=json.loads(p.read_text());c=a['fit_contract'];slot=(c['seed'],c['available_budget'],c['arm'],c['replicate'])
  assert slot in expected and slot not in slots;slots.add(slot);init[c['seed'],c['available_budget'],c['replicate']].add(a['initialization_state_sha256'])
  assert a['parameter_count']==458952 and a['validation_rows']==416
  assert a['label_access']['test_numeric_targets_read']==a['label_access']['unacquired_numeric_targets_read']==0
  assert a['label_access']['test_and_H_graphs_zero_during_fit']
  h=readcsv(p.parent/'history.csv');assert len(h)==a['epochs_run']<=1000
  best=min(h,key=lambda r:float(r['validation_selection_score']))
  assert int(best['epoch'])==a['best_epoch']
  maxerr=max(maxerr,abs(float(best['validation_selection_score'])-a['validation_metrics']['combined_normalized_rmse']))
  assert sha(p.parent/'best.pt')==a['checkpoint_sha256'] and sha(p.parent/'history.csv')==a['history_sha256']
  pairs=readcsv(p.parent/'heldout_pairs.csv');groups=readcsv(p.parent/'heldout_group_metrics.csv')
  by=collections.defaultdict(list)
  for r in pairs:
   dy=float(r['observed_difference_ml']);dp=float(r['predicted_difference_ml']);e=dp-dy
   for col,want in [('signed_difference_error_ml',e),('absolute_difference_error_ml',abs(e)),('observed_abs_difference_ml',abs(dy)),('predicted_abs_difference_ml',abs(dp))]:
    maxerr=max(maxerr,abs(float(r[col])-want));checks+=1
   want=int((dy>0)-(dy<0)==(dp>0)-(dp<0));assert want==int(r['direction_match'])
   by[r['group_id'],r['axis'],r['target']].append(r)
  assert len(by)==len(groups)
  for r in groups:
   rs=by[r['group_id'],r['axis'],r['target']];assert len(rs)==int(r['pair_count'])
   for col in ['signed_difference_error_ml','absolute_difference_error_ml','direction_match','observed_abs_difference_ml','predicted_abs_difference_ml']:
    want=math.fsum(float(x[col]) for x in rs)/len(rs);maxerr=max(maxerr,abs(float(r[col])-want));checks+=1
 assert slots==expected and all(len(v)==1 for v in init.values())
 assert maxerr<1e-10,maxerr
 qual={ (s['seed'],s['available_budget']):s for s in q['states'] }
 result={'status':'PASS','successful_fits':len(audits),'prespecified_slots_exactly_complete':True,'all_paired_initialization_equal':True,
  'H_group_arithmetic_checks':checks,'max_scalar_difference':maxerr,'min_validation_epoch_independently_verified':True,
  'actual_train_rows':{f'{s}/{b}':v['actual_train_rows'] for (s,b),v in qual.items()},
  'test_truth_reads':0,'new_training_runs':0,'read_scope':'local already-observed H pair outputs, history/audit, no raw target store',
  'scientific_limits':'same pilot data, independent arithmetic check only; not an independent efficacy study'}
 (OUT/'verification/pilot_independent_results_check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
