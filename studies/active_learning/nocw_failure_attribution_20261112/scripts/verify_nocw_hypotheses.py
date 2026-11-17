#!/usr/bin/env python3
"""Independent graph/gain checks using explicit loops and scalar least squares."""
from pathlib import Path
import json,hashlib,itertools
from decimal import Decimal
from fractions import Fraction
import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.stats import rankdata
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[2];OLD=OUT.parent/'comprehensive_history_diagnostics_20261112'
def load(n):return pd.read_csv(OUT/'tables'/n,float_precision='round_trip')
def main():
 b=load('nocw_batches.csv');h=load('nocw_hypotheses.csv');pairs=load('nocw_test_pairs.csv');cases=load('nocw_gain_cases.csv');corr=load('nocw_science_gain_correlations.csv')
 source=pd.read_csv(OLD/'tables/behavior_batches.csv');fixed=[];max_gainerr=0.;pair_count=0
 for seed in [157,887,2357,6101,12203]:
  for r in [0,5,14,20]:
   row=source[(source.method=='nocw')&(source.seed==seed)&(source['round']==r)].iloc[0];d=(ROOT/row.source).parent
   packet=json.loads((d/'input.json').read_text());selection=json.loads((d/'selection.json').read_text())
   def decode(n):
    rows={}
    for values in packet[n]['rows']:
     x=dict(zip(packet[n]['columns'],values));rows[x['id']]=x
    return rows
   obs=decode('observed');cand=decode('candidates');expected=0
   def key(x):
    c=dict(zip(packet['condition_columns'],x['conditions']));return [Fraction(c['PE/EA']),Decimal(str(c['Density g/ml'])),Decimal(str(c['V/ul'])),c['loading solvent'],Decimal(str(c['Volume of loading solvent/ul']))]
   def single(a,c):return a['molecule_id']==c['molecule_id'] and sum(x!=y for x,y in zip(key(a),key(c)))==1
   for hyp in selection['hypotheses']:
    ids=[c['id'] for c in selection['choices'] if c['hypothesis_id']==hyp['id']]
    evidence=set(hyp['supporting_observed_ids']+hyp['contradicting_observed_ids'])
    w=sum(single(cand[a],cand[c]) for a,c in itertools.combinations(ids,2))
    x=sum(single(cand[a],obs[c]) for a in ids for c in evidence)
    sr=h[(h.seed==seed)&(h['round']==r)&(h.hypothesis_id==hyp['id'])].iloc[0]
    assert int(sr.within_linked_strict_single_pairs)==w and int(sr.vs_cited_evidence_strict_single_pairs)==x
    expected+=w+x
   actual=len(pairs[(pairs.seed==seed)&(pairs['round']==r)]);assert actual==expected
   fixed.append({'seed':seed,'round':r,'pairs':actual,'hypotheses':len(selection['hypotheses'])});pair_count+=actual
 for x in cases.to_dict('records'):
  traj=b[b.seed==x['seed']].set_index('round');r=int(x['round'])
  for horizon in [2,3]:
   future=list(range(r+1,r+horizon+1));available=all(i in traj.index for i in future)
   assert bool(x[f'next_{horizon}_right_censored'])==(not available)
   if available:
    ref=float(traj.loc[r,'nrmse_before']-traj.loc[r+horizon,'nrmse_after'])
    max_gainerr=max(max_gainerr,abs(ref-x[f'current_plus_next_{horizon}_gain']))
 assert max_gainerr<1e-12
 max_partial=0.;checked=0
 for x in corr.to_dict('records'):
  g=b[(b.seed==x['seed'])&b.primary].dropna(subset=[x['feature']]);rk=lambda v:rankdata(np.round(v.to_numpy(dtype=float),12))
  if g[x['feature']].nunique()<2:continue
  for include,column in [(True,'partial_budget_phase_before'),(False,'partial_budget_phase_only')]:
   if not np.isfinite(x[column]):continue
   cov=[np.ones(len(g)),rk(g.labeled_size_before)]
   if include:cov.append(rk(g.nrmse_before))
   cov.extend(pd.get_dummies(g.phase,drop_first=True,dtype=float).to_numpy().T)
   q=null_space(np.asarray(cov));a=q.T@rk(g[x['feature']]);c=q.T@rk(g.gain)
   value=float(a@c/np.sqrt((a@a)*(c@c)))
   max_partial=max(max_partial,abs(value-x[column]));checked+=1
 assert max_partial<1e-10
 # Compare the exact prior-59 cohort with its already published audit, not a changed denominator.
 caps={157:14,887:7,2357:13,6101:11,12203:14};old=b[b.apply(lambda x:x['round']<caps[x.seed],axis=1)]
 assert len(old)==59 and int(old.first_complete_legal.sum())==33 and int(old.complete_response_n.sum())==99
 result={'status':'PASS','fixed_grid_batches':len(fixed),'independently_rebuilt_pairs':pair_count,'fixed_grid':fixed,
  'cumulative_gain_max_telescoping_error':max_gainerr,'partial_coefficients_recomputed':checked,'max_nullspace_correlation_error':max_partial,
  'prior_59_batch_audit_exactly_reproduced':{'first_complete_legal':33,'complete_responses':99},
  'checks_do_not_establish':'semantic hypothesis truth, causal learning benefit or independence of shared-row pairs'}
 (OUT/'verification/nocw_independent_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
