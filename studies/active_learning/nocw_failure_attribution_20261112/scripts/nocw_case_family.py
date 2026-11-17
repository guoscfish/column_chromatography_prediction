#!/usr/bin/env python3
"""Metadata-only Murcko concentration on the already fixed 45 NoCW case positions."""
from pathlib import Path
import collections,hashlib,itertools,json,lzma,math
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[2]
OLD=OUT.parent/'comprehensive_history_diagnostics_20261112';SOURCES={}
def read(p,role,expected=None):
 p=Path(p);p=p if p.is_absolute() else ROOT/p
 assert 'qualification_dataset_manifest' not in str(p) and '/dataset/' not in str(p)
 b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
 if expected:assert h==expected,(p,'source drift')
 SOURCES[str(p.relative_to(ROOT))]={'path':str(p.relative_to(ROOT)),'sha256':h,'bytes':len(b),'role':role}
 return b
def frame(p,columns):
 import io
 return pd.read_csv(io.BytesIO(read(p,'pre-existing case identifiers or target-free selected metadata')),usecols=columns,keep_default_na=False)
def concentration(scaffolds):
 c=collections.Counter(scaffolds);n=sum(c.values());assert n>0
 dominant=min(c,key=lambda s:(-c[s],s))
 return {'n':n,'unique':len(c),'max_share':max(c.values())/n,'hhi':sum((v/n)**2 for v in c.values()),
   'dominant':dominant or '[acyclic]','phenyl_share':c['c1ccccc1']/n,'acyclic_share':c['']/n}
def expected_sample_hhi(pop_counts,n):
 N=sum(pop_counts.values());assert 1<=n<=N and N>1
 same=sum(k*(k-1) for k in pop_counts.values())/(N*(N-1))
 return 1/n+(n-1)/n*same
def main():
 rule={'cases':'exactly the already frozen 45 seed x phase x gain-category positions; no case added/dropped',
  'groups':'high_gain, minimum_gain_case, near_zero; phase and seed equally weighted',
  'family':'existing RDKit Murcko, includeChirality=False; preserve empty acyclic bucket',
  'metrics':['max slot share','HHI=sum(scaffold slot fraction squared)','same-U reference','exact expected HHI of uniform without-replacement size32 sample'],
  'temporal_reference':'up to three immediately preceding real accepted batches of the same seed/method; no future rows',
  'label_access':'none; source metadata has id/smiles/conditions only; gain values not loaded or used',
  'purpose':'descriptive family concentration check only; no acquisition rule/model/H change'}
 plan=OUT/'verification/nocw_case_family_plan.json'
 if plan.exists():assert json.loads(plan.read_text())==rule
 else:plan.write_text(json.dumps(rule,ensure_ascii=False,indent=2)+'\n')
 read(plan,'metadata-only supplementary plan')
 cases=frame(OUT/'tables/nocw_gain_cases.csv',['method','seed','round','phase','case_category'])
 batches=frame(OUT/'tables/nocw_batches.csv',['seed','round','selection_source'])
 selected=frame(OLD/'tables/behavior_selected_rows.csv',['method','seed','round','sample_id','canonical_smiles','scaffold'])
 nocw=selected[selected.method=='nocw'].copy();lookup={}
 for smile,g in selected.groupby('canonical_smiles'):
  assert g.scaffold.nunique()==1
  lookup[smile]=g.scaffold.iloc[0]
 assert len(lookup)==217
 snapshot=ROOT/'studies/active_learning/nocw_free_selection_diagnostics_20261101/snapshot_inputs.json.xz'
 snap=json.loads(lzma.decompress(read(snapshot,'existing audited target-free metadata snapshot')));meta=snap['metadata']
 assert all(set(x)=={'id','smiles','conditions'} for x in meta.values())
 for x in nocw.to_dict('records'):assert meta[x['sample_id']]['smiles']==x['canonical_smiles'] and lookup[x['canonical_smiles']]==x['scaffold']
 locks=json.loads(read(OUT/'verification/nocw_inputs.json','accepted contract identity/SHA lock'))
 lock={e['path']:e['sha256'] for e in locks['files']}
 rows=[];contracts={}
 for c in cases.to_dict('records'):
  seed,r=int(c['seed']),int(c['round']);z=nocw[(nocw.seed==seed)&(nocw['round']==r)];assert len(z)==32
  if (seed,r) not in contracts:
   source=batches[(batches.seed==seed)&(batches['round']==r)].selection_source.iloc[0];p=(ROOT/source).parent/'contract.json'
   path=str(p.relative_to(ROOT));ct=json.loads(read(p,'metadata-only same-state L/U contract',lock[path]));contracts[seed,r]=ct
  ct=contracts[seed,r];U=ct['U_t_ids'];assert len(U)==2997-32*r and len(set(U))==len(U) and set(z.sample_id)<=set(U)
  Usc=[lookup[meta[i]['smiles']] for i in U];batch=concentration(list(z.scaffold));pool=concentration(Usc)
  expected=expected_sample_hhi(collections.Counter(Usc),32)
  prior=nocw[(nocw.seed==seed)&(nocw['round']>=max(0,r-3))&(nocw['round']<r)]
  assert len(prior)==32*min(3,r)
  prior_sc=set(prior.scaffold);seen=float(z.scaffold.isin(prior_sc).mean()) if len(prior) else np.nan
  U_seen=sum(s in prior_sc for s in Usc)/len(Usc) if len(prior) else np.nan
  prior_top=concentration(list(prior.scaffold))['dominant'] if len(prior) else None
  prior_key='' if prior_top=='[acyclic]' else prior_top
  rec={**c,'candidate_U_n':len(U),'batch_unique_scaffolds':batch['unique'],'batch_max_slot_share':batch['max_share'],'batch_hhi':batch['hhi'],
   'U_unique_scaffolds':pool['unique'],'U_max_slot_share':pool['max_share'],'U_hhi':pool['hhi'],
   'batch_minus_U_max_share':batch['max_share']-pool['max_share'],'batch_minus_U_hhi':batch['hhi']-pool['hhi'],
   'same_U_expected_random32_hhi':expected,'batch_hhi_minus_expected_random32':batch['hhi']-expected,
   'batch_dominant_scaffold':batch['dominant'],'U_dominant_scaffold':pool['dominant'],
   'batch_phenyl_core_share':batch['phenyl_share'],'U_phenyl_core_share':pool['phenyl_share'],
   'batch_acyclic_share':batch['acyclic_share'],'U_acyclic_share':pool['acyclic_share'],
   'prior_actual_batch_count':min(3,r),'prior3_dominant_scaffold':prior_top,
   'batch_slot_fraction_seen_prior3_scaffolds':seen,'U_slot_fraction_seen_prior3_scaffolds':U_seen,
   'batch_minus_U_prior3_scaffold_seen_fraction':seen-U_seen,
   'batch_prior3_dominant_scaffold_share':float((z.scaffold==prior_key).mean()) if prior_top is not None else np.nan,
   'U_prior3_dominant_scaffold_share':sum(s==prior_key for s in Usc)/len(Usc) if prior_top is not None else np.nan,
   'contract_source':str((ROOT/batches[(batches.seed==seed)&(batches['round']==r)].selection_source.iloc[0]).parent.relative_to(ROOT))+'/contract.json'}
  assert 0<rec['batch_hhi']<=1 and 0<rec['U_hhi']<=1
  rows.append(rec)
 d=pd.DataFrame(rows);assert len(d)==45
 assert d.groupby(['case_category','seed','phase']).size().eq(1).all()
 d.to_csv(OUT/'tables/nocw_case_family.csv',index=False,float_format='%.17g')
 metrics=['batch_max_slot_share','batch_hhi','U_max_slot_share','U_hhi','batch_minus_U_max_share','batch_minus_U_hhi','same_U_expected_random32_hhi',
  'batch_hhi_minus_expected_random32','batch_phenyl_core_share','U_phenyl_core_share','batch_acyclic_share','U_acyclic_share',
  'batch_slot_fraction_seen_prior3_scaffolds','U_slot_fraction_seen_prior3_scaffolds','batch_minus_U_prior3_scaffold_seen_fraction']
 summaries=[]
 for category,g in d.groupby('case_category'):
  seed_mean=g.groupby('seed')[metrics].mean();rec={'case_category':category,'seed_phase_positions':len(g),'outer_seeds':len(seed_mean),'phases_per_seed':3,
    'prior3_defined_positions':int(g.prior_actual_batch_count.gt(0).sum())}
  for metric in metrics:rec[metric+'_mean']=float(seed_mean[metric].mean());rec[metric+'_seed_sd']=float(seed_mean[metric].std(ddof=1))
  summaries.append(rec)
 pd.DataFrame(summaries).to_csv(OUT/'tables/nocw_case_family_summary.csv',index=False,float_format='%.17g')
 paired=[]
 for seed,g in d.groupby('seed'):
  high=g[g.case_category=='high_gain'].set_index('phase');low=g[g.case_category=='minimum_gain_case'].set_index('phase');near=g[g.case_category=='near_zero'].set_index('phase')
  rec={'seed':int(seed)}
  for m in ['batch_max_slot_share','batch_hhi','batch_hhi_minus_expected_random32','batch_minus_U_prior3_scaffold_seen_fraction']:
   rec['minimum_minus_high_'+m]=float((low[m]-high[m]).mean());rec['near_zero_minus_high_'+m]=float((near[m]-high[m]).mean())
  paired.append(rec)
 # Independent finite-population identity on an exhaustive toy case.
 toy=list('aaaabb');enumerated=np.mean([concentration([toy[i] for i in ids])['hhi'] for ids in itertools.combinations(range(6),3)])
 analytic=expected_sample_hhi(collections.Counter(toy),3);assert abs(enumerated-analytic)<1e-14
 result={'status':'PASS','case_positions':45,'unique_case_batches':d[['seed','round']].drop_duplicates().shape[0],
  'reused_RDKit_molecules':217,'all_scaffold_metadata_matches_old_selected_table':True,'same_state_U_contracts':len(contracts),
  'metadata_has_no_response_fields':True,'actual_previous_batches_only':True,'balanced_seed_phase_groups':True,
  'finite_population_HHI_exhaustive_toy_check':{'enumerated':enumerated,'analytic':analytic},'paired_seed_differences':paired,
  'limits':['Murcko phenyl core collapses many chemically distinct functional-group families','Empty Murcko string pools all acyclic molecules','Raw pool HHI is not the expectation of a size32 sample; analytic expectation is given separately','45 outcome-selected positions are descriptive, with five dependent seed trajectories; no causal or new-rule claim'],
  'new_labels_read':0,'test_truth_reads':0,'new_fits':0,'new_llm_calls':0}
 (OUT/'verification/nocw_case_family_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 (OUT/'verification/nocw_case_family_inputs.json').write_text(json.dumps(list(SOURCES.values()),ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':'PASS','case_positions':45,'contracts':len(contracts),'paired_seed_differences':paired},ensure_ascii=False))
if __name__=='__main__':main()
