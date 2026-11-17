#!/usr/bin/env python3
"""Seal/check read-only analysis provenance and new deliverables. No experiment imports."""
from pathlib import Path
import argparse,hashlib,json,re,subprocess,datetime,sys
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[2]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def js(p):return json.loads(Path(p).read_text())
def rel(p):return str(Path(p).relative_to(ROOT))
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def collect():
 sources={}; embedded=[]
 specs=[('learning_input_manifest.json',None),('additional_input_manifest.json',None),
   ('behavior_inputs.json','files'),('reasoning_input_manifest.json','files'),('model_input_manifest.json','inputs'),('gain_input_manifest.json',None)]
 for name,key in specs:
  data=js(OUT/'verification'/name);entries=data if key is None else data[key]
  for item in entries:
   path=item['path'];p=Path(path);p=p if p.is_absolute() else ROOT/p
   if 'qualification_dataset_manifest.csv' in str(p):raise AssertionError('Excluded incident file must never be an analysis input')
   keyp=rel(p)
   if keyp in sources:assert sources[keyp]['sha256']==item['sha256'],('source version conflict',keyp)
   else:sources[keyp]={'path':keyp,'sha256':item['sha256'],'role':item.get('role','historical metadata / selected feedback / frozen geometry'),'module_manifests':[]}
   sources[keyp]['module_manifests'].append(name)
  if name=='behavior_inputs.json':embedded+=data['embedded_archive_entries']
 extra=['src/qgeognn_al/active_learning_v2/benchmark_protocol.py','src/qgeognn_al/active_learning_v2/sequential_runner.py',
   'src/qgeognn_al/evaluation/point.py','studies/active_learning/nocw_free_selection_diagnostics_20261101/PROTOCOL_AUDIT.md',
   'reports/free_cw_final_20261112/README.md','reports/free_cw_final_20261112/REPORT.md','reports/free_cw_final_20261112/snapshot.json',
   'reports/free_cw_final_20261112/final_verification_20261112.json','reports/five_methods_cw_gradient_20261103/README.md']
 for path in extra:
  if path not in sources:sources[path]={'path':path,'sha256':sha(ROOT/path),'role':'protocol/source/reused report','module_manifests':['root qualitative review']}
 # Hash only allowed sources. Do not open data files outside the module source lists.
 for item in sources.values():
  p=ROOT/item['path'];assert p.is_file(),('missing fixed input',item['path'])
  assert sha(p)==item['sha256'],('input drift',item['path'])
  item['bytes']=p.stat().st_size;item['available_at_seal']=True
  item['derived_in_this_study']=p.is_relative_to(OUT)
 return list(sorted(sources.values(),key=lambda x:x['path'])),embedded
def links_and_structure():
 required=['README.md','MASTER_REPORT.md','EXPERIMENT_INVENTORY.md','PROTOCOL_AND_FAIRNESS_AUDIT.md','LEARNING_DYNAMICS.md',
  'ACQUISITION_BEHAVIOR.md','ACQUISITION_VS_LEARNING_GAIN.md','SCIENTIFIC_REASONING_AUDIT.md','CHEMICAL_ERROR_AND_DATA_AUDIT.md',
  'NUMERICAL_SIGNAL_AUDIT.md','STABILITY_AND_COST_AUDIT.md','RESEARCH_DIRECTIONS.md','NEXT_STAGE_DECISION.md','MISSING_EVIDENCE.md']
 for n in required:assert (OUT/n).is_file(),n
 bad=[];local_links=0
 for p in OUT.glob('*.md'):
  for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   if target.startswith(('http:','https:','#')):continue
   target=target.split('#')[0]
   if not target:continue
   local_links+=1;q=p.parent/target
   if not q.exists() and q.name not in ['INPUT_MANIFEST.json','final_verification.json']:bad.append({'document':p.name,'target':target})
 assert not bad,bad
 return {'required_reports':len(required),'local_links_checked':local_links,'broken_links':bad}
def edge_checks():
 import importlib.util
 import pandas as pd
 spec=importlib.util.spec_from_file_location('history_learning',OUT/'scripts/analyze_learning.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 a=mod.area(pd.DataFrame({'budget':[333,365,397],'nrmse':[1.,.2,.5]}));assert abs(a-.475)<1e-14
 assert abs(a-.4)>.01,'must not silently use best-so-far'
 for bad in [pd.DataFrame({'budget':[333],'nrmse':[1.]}),pd.DataFrame({'budget':[333,397],'nrmse':[1.,.2]})]:
  try:mod.area(bad)
  except AssertionError:pass
  else:raise AssertionError('incomplete curve accepted')
 d=pd.read_csv(OUT/'tables/round_metrics.csv');st=pd.read_csv(OUT/'tables/stage_metrics.csv')
 assert len(d)==1088 and d.groupby(['method','seed']).ngroups==34
 assert len(pd.read_csv(OUT/'tables/additional_round_metrics.csv'))==238
 assert len(pd.read_csv(OUT/'tables/gain_linked_batches.csv'))==873
 assert len(pd.read_csv(OUT/'tables/reasoning_hypotheses.csv'))==871
 assert len(pd.read_csv(OUT/'tables/model_batch_metrics.csv'))==180
 return {'nonmonotone_area':a,'best_so_far_rejected':True,'single_point_and_missing_round_rejected':True,
  'main_curve_points':1088,'additional_curve_points':238,'linked_batches':873,'hypotheses':871,'model_batches':180}
def worktree():
 start=js(OUT/'verification/repository_start.json');changed=[];deleted=[]
 for path,old in start['preexisting_changes'].items():
  p=ROOT/path
  if not p.exists():deleted.append(path)
  elif sha(p)!=old['sha256']:changed.append(path)
 expected=['reports/llm_eight_to1549_20261018/live_status.json',
 'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/sol_xhigh_all_strategies_to1549_20261018/recovery_monitor_status.json']
 unexpected=[x for x in changed if x not in expected]
 assert not deleted and not unexpected,{'deleted':deleted,'unexpected':unexpected}
 status=subprocess.check_output(['git','status','--porcelain=v1','-uall'],cwd=ROOT,text=True)
 outside=[]
 for line in status.splitlines():
  path=line[3:]
  if not path.startswith(rel(OUT)+'/') and path not in start['preexisting_changes']:outside.append(path)
 assert not outside,('new outside changes',outside)
 return {'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
  'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'new_analysis_untracked':True,
  'preexisting_changed_untracked_files_checked':len(start['preexisting_changes']),'unchanged_preexisting_files':len(start['preexisting_changes'])-len(changed),
  'live_monitor_files_changed_since_start':changed,'unexpected_source_mutations':unexpected,'preexisting_deleted':deleted,'new_changes_outside_study':outside,
  'committed':False,'pushed':False,'note':'Known pre-existing monitoring scripts remain active; their two live status files are not frozen inputs or analysis writes.'}
def outputs():
 exclude={'INPUT_MANIFEST.json','verification/final_verification.json','verification/output_manifest.json'}
 return [{'path':str(p.relative_to(OUT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.rglob('*'))
   if p.is_file() and str(p.relative_to(OUT)) not in exclude and '__pycache__' not in p.parts]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
 if args.check:
  manifest=js(OUT/'INPUT_MANIFEST.json');fail=[]
  for e in manifest['sources']:
   p=ROOT/e['path']
   if not p.exists() or sha(p)!=e['sha256']:fail.append(e['path'])
  for e in js(OUT/'verification/output_manifest.json'):
   p=OUT/e['path']
   if not p.exists() or sha(p)!=e['sha256']:fail.append(str(p.relative_to(ROOT)))
  assert not fail,fail
  print(json.dumps({'status':'SEALED_INPUTS_AND_OUTPUTS_MATCH','inputs':len(manifest['sources']),'outputs':len(js(OUT/'verification/output_manifest.json'))}))
  return
 sources,embedded=collect();structure=links_and_structure();edge=edge_checks();git=worktree()
 reviews={}
 for n in ['independent_learning_review.json','independent_gain_review.json','independent_synthesis_review.json','model_synthesis_review.json','model_independent_verification.json','behavior_checks.json']:
  data=js(OUT/'verification'/n);reviews[n]=data.get('status','recorded')
  assert str(data.get('status','')).startswith('PASS'),(n,data.get('status'))
 now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
 manifest={'created_shanghai':now,'repository_head':git['head'],'remote_main_checked':'44be2c0070219d71b2560604f8234a390cc14304',
  'policy':'Fixed historical inputs; no hidden candidate truth or test truth in analysis dataflow. See disclosed incidental search event.',
  'sources':sources,'embedded_entries_additional_index':embedded,'full_embedded_audits':'module verification archives retain full checks',
  'incident_excluded':'verification/label_access_incident.json','new_training_runs':0,'new_paid_llm_requests':0,'new_selections':0,
  'session_wide_zero_label_file_contact_claim':False}
 dump(OUT/'INPUT_MANIFEST.json',manifest)
 out=outputs();dump(OUT/'verification/output_manifest.json',out)
 result={'status':'PASS_WITH_DISCLOSED_SCOPE_DEVIATION','completed_shanghai':now,'source_files_checked':len(sources),
 'source_bytes':sum(x['bytes'] for x in sources),'output_files':len(out),'output_bytes':sum(x['bytes'] for x in out),
 'figures_png':len(list((OUT/'figures').glob('*.png'))),'structure':structure,'boundary_checks':edge,'independent_reviews':reviews,
 'git':git,'scientific_limits':'observational histories, five correlated seeds, selected-only residuals, not causal validation',
 'analysis_scripts_test_truth_files_opened':0,'session_wide_zero_label_file_contact_claim':False,
 'scope_deviation':{'incident':'verification/label_access_incident.json','incidental_manifest_rows':3,'used_in_analysis':False},
 'new_paid_llm_requests':0,'new_training_runs':0,'new_selections':0,'historical_sources_modified_by_analysis':False,
 'checks_not_claimed':'Hash/identity/arithmetic checks do not establish chemical truth, causal efficacy, independent validation or provider billing.'}
 dump(OUT/'verification/final_verification.json',result)
 print(json.dumps({k:result[k] for k in ['status','source_files_checked','output_files','figures_png','git']},ensure_ascii=False))
if __name__=='__main__':main()
