#!/usr/bin/env python3
"""Supplementary completed numerical trajectories, using fit metadata only."""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[2]
sources=[]
def read(p):
 b=p.read_bytes();sources.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)});return json.loads(b)
def main():
 rows=[]
 names=['qgeognn_v2_row_maxdet_b32','qgeognn_v2_row_fused_maxdet_b32','qgeognn_v2_row_fused_maxdet_a080_b32_screen','qgeognn_v2_row_cw_ivr_portfolio_b32']
 for name in names:
  study=ROOT/'studies/active_learning'/name
  for f in sorted(study.glob('runtime/seed_*/*/round_*/model/fit_audit.json')):
   a=read(f);seed=int(f.parts[-5].split('_')[1]);method=f.parts[-4];r=int(f.parts[-3].split('_')[1])
   assert a['train_rows']==333+32*r and a['test_labels_used_for_fit_or_checkpoint_selection']==0
   rows.append({'method':method,'seed':seed,'budget':a['train_rows'],'nrmse':a['best_validation_combined_normalized_rmse'],
    'initialization_hash':a['initialization_hash'],'validation_ids_hash':a['validation_ids_hash'],'source':str(f.relative_to(ROOT))})
 df=pd.DataFrame(rows);base=pd.read_csv(OUT/'tables/round_metrics.csv')
 for (m,s),g in df.groupby(['method','seed']):
  b=base[(base.method=='gradient_lcmd32')&(base.seed==s)&(base.budget==333)].iloc[0]
  assert set(g.initialization_hash)=={b.initialization_hash} and set(g.validation_ids_hash)=={b.validation_ids_hash}
  rows.append({'method':m,'seed':s,'budget':333,'nrmse':b.nrmse,'initialization_hash':b.initialization_hash,'validation_ids_hash':b.validation_ids_hash,'source':b.source})
 df=pd.DataFrame(rows).sort_values(['method','seed','budget']);df.to_csv(OUT/'tables/additional_round_metrics.csv',index=False)
 reg=[];paired=[]
 for m,g in df.groupby('method'):
  common=int(g.groupby('seed').budget.max().min())
  for seed,h in g.groupby('seed'):
   assert np.all(np.diff(h.budget)==32)
   z=h[h.budget<=common];auc=float(np.trapezoid(z.nrmse,z.budget)/(common-333))
   reg.append({'method':m,'seed':seed,'observed_endpoint':int(h.budget.max()),'method_common_budget':common,'common_aulc':auc,'common_endpoint_nrmse':z.nrmse.iloc[-1]})
   for comp in ['random32','gradient_lcmd32','numerical_hybrid32','new_free_cw_sol61_xhigh']:
    b=base[(base.method==comp)&(base.seed==seed)&(base.budget<=common)].sort_values('budget')
    assert list(b.budget)==list(z.budget)
    paired.append({'method':m,'comparator':comp,'seed':seed,'budget':common,'aulc_delta':auc-float(np.trapezoid(b.nrmse,b.budget)/(common-333)),'endpoint_delta':z.nrmse.iloc[-1]-b.nrmse.iloc[-1]})
 pd.DataFrame(reg).to_csv(OUT/'tables/additional_run_registry.csv',index=False)
 pd.DataFrame(paired).to_csv(OUT/'tables/additional_paired_effects.csv',index=False)
 inventory=[]
 for p in sorted((ROOT/'studies/active_learning').iterdir()):
  if p.is_dir() and p!=OUT:
   readme=p/'README.md'; inventory.append({'study':p.name,'readme_available':readme.exists(),'readme_sha256':hashlib.sha256(readme.read_bytes()).hexdigest() if readme.exists() else '',
    'in_main_eight_method_analysis':p.name in ['qgeognn_v2_row_sequential_b32','qgeognn_v2_row_llm_scientist_v2','qgeognn_v2_row_llm_nocw_ablation','free_cw_sol61_xhigh_five_to1485_20261101'],
    'note':'directory inventory only; inclusion/causal status requires protocol review'})
 pd.DataFrame(inventory).to_csv(OUT/'tables/study_directory_inventory.csv',index=False)
 (OUT/'verification/additional_input_manifest.json').write_text(json.dumps(sources,indent=2)+'\n')
 (OUT/'verification/additional_verification.json').write_text(json.dumps({'status':'PASS','methods':df.method.nunique(),'runs':len(reg),'points':len(df),'gap_check':True,'initialization_validation_identity':True,'new_fits':0,'test_files_opened':0},indent=2)+'\n')
 print('PASS additional',len(df),'points',len(reg),'runs')
if __name__=='__main__':main()
