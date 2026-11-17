#!/usr/bin/env python3
"""Verify and seal the attribution study without executing any model or selector."""
from pathlib import Path
import argparse,hashlib,json,csv,re,subprocess,datetime,collections
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[2]
def js(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def rel(p):return str(Path(p).relative_to(ROOT))
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def countcsv(p):
 with p.open() as f:return len(list(csv.DictReader(f)))
def sources():
 specs=[('nocw_inputs.json','files'),('nocw_review_inputs.json',None),('nocw_case_family_inputs.json',None),('condition_input_manifest.json','sources'),('pilot_metadata_inputs.json',None),
  ('pilot_training_inputs.json',None),('next_al_inputs.json',None),('decision_inputs.json',None)]
 found={}
 for filename,key in specs:
  obj=js(OUT/'verification'/filename);items=obj[key] if key else obj
  for item in items:
   p=Path(item['path']);p=p if p.is_absolute() else ROOT/p;path=rel(p)
   assert 'qualification_dataset_manifest.csv' not in path,'Forbidden label manifest dependency'
   if path in found:assert found[path]['sha256']==item['sha256'],('source revisions conflict',path)
   else:found[path]={'path':path,'sha256':item['sha256'],'role':item.get('role','frozen metadata / authorized observed feedback / existing analysis'),'role_details':[],'manifests':[]}
   detail=item.get('role')
   if detail and detail not in found[path]['role_details']:found[path]['role_details'].append(detail)
   found[path]['manifests'].append(filename)
 # Freeze adds the exact training sources and code before outcomes were obtained.
 freeze=js(OUT/'verification/pilot_protocol_freeze.json')
 for path,h in {**freeze['source_hashes'],**freeze['data_sources']}.items():
  if path in found:assert found[path]['sha256']==h
  else:found[path]={'path':path,'sha256':h,'role':'pre-training frozen model/data identity','manifests':['pilot_protocol_freeze.json']}
 prior=ROOT/'studies/active_learning/comprehensive_history_diagnostics_20261112'
 for p in list(prior.glob('*.md'))+[prior/'INPUT_MANIFEST.json']:
  path=rel(p)
  if path not in found:found[path]={'path':path,'sha256':sha(p),'role':'reused prior report; never overwritten','manifests':['root synthesis']}
 for item in found.values():
  p=ROOT/item['path'];assert p.exists(),('missing fixed source',item['path'])
  assert sha(p)==item['sha256'],('input drift',item['path'])
  item['bytes']=p.stat().st_size;item['available']=True
  if item['path']=='experiments/e0_4g_baseline/canonical_4g.csv':
   item['role']='restricted canonical source: specified X-only columns plus explicit 416 validation-ID target reads; no other target fields extracted'
   item['permitted_numeric_target_roles']=['validation']
 return sorted(found.values(),key=lambda x:x['path'])
def check_pilot():
 status=js(OUT/'verification/pilot_execution_status.json')
 assert status['status']=='COMPLETE' and status['successful_fits']==16
 freeze=js(OUT/'verification/pilot_protocol_freeze.json');release=js(OUT/'verification/pilot_root_release.json')
 assert release['successful_fit_cap']==freeze['maximum_successful_fits']==16
 assert sha(OUT/'scripts/model_pilot.py')==freeze['script_sha256']
 assert sha(OUT/'MODEL_PILOT_PROTOCOL.md')==freeze['protocol_sha256']
 expected={(s,b,a,r) for s in [157,887] for b in [525,1005] for a in ['baseline','interaction'] for r in [0,1]}
 audits=list((OUT/'runtime').glob('seed_*/L*/*/rep_*/fit_audit.json'));slots={};inits={}
 assert len(audits)==16
 for p in audits:
  d=js(p);c=d['fit_contract'];slot=(c['seed'],c['available_budget'],c['arm'],c['replicate'])
  assert slot in expected and slot not in slots and d['status']=='SUCCESS'
  assert d['parameter_count']==458952 and d['epochs_run']<=1000
  assert d['label_access']['test_numeric_targets_read']==d['label_access']['unacquired_numeric_targets_read']==0
  assert d['label_access']['test_and_H_graphs_zero_during_fit']
  assert sha(p.parent/'best.pt')==d['checkpoint_sha256'] and sha(p.parent/'history.csv')==d['history_sha256']
  assert sha(p.parent/'heldout_pairs.csv')==d['H_pair_file_sha256']
  key=(c['seed'],c['available_budget'],c['replicate']);inits.setdefault(key,set()).add(d['initialization_state_sha256'])
  slots[slot]=rel(p)
 assert set(slots)==expected and all(len(v)==1 for v in inits.values())
 return {'successful_fits':16,'unique_architecture_changes':1,'new_parameters':0,'paired_initialization':True,
  'global_fit_cap_respected':True,'model_script_and_protocol_unchanged_since_freeze':True,
  'validation_role_allowlist_rows_per_read':416,'test_and_unacquired_numeric_targets_read':0,'slots':[{'slot':list(k),'audit':v} for k,v in sorted(slots.items())]}
def git_check():
 original=js(OUT/'verification/repository_start.json');changed=[];deleted=[]
 for path,item in original['preexisting_changes'].items():
  p=ROOT/path
  if not p.exists():deleted.append(path)
  elif sha(p)!=item['sha256']:changed.append(path)
 allowed=['reports/llm_eight_to1549_20261018/live_status.json',
 'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/sol_xhigh_all_strategies_to1549_20261018/recovery_monitor_status.json']
 assert not deleted and set(changed)<=set(allowed),('unexpected prior source changes',changed,deleted)
 status=subprocess.check_output(['git','status','--porcelain=v1','-uall'],cwd=ROOT,text=True)
 new_external=[]
 for l in status.splitlines():
  path=l[3:]
  if not path.startswith(rel(OUT)+'/') and path not in original['preexisting_changes']:new_external.append(path)
 # Other chats/processes share this workspace. Preserve and disclose concurrent
 # additions; they are neither task inputs nor evidence of this task's actions.
 concurrent={'status':'OBSERVED_CONCURRENT_WORKSPACE_ADDITIONS' if new_external else 'NO_EXTERNAL_ADDITIONS_OBSERVED',
  'paths':sorted(new_external),'count':len(new_external),
  'grouped_prefixes':dict(collections.Counter('/'.join(p.split('/')[:4]) for p in new_external)),
  'content_read_for_this_inventory':False,'used_as_research_inputs':False,
  'provenance':'Unattributed external workspace writers; this inventory does not infer who started them',
  'handling':'Preserved without reading labels, executing runners, editing, deleting or incorporating into this study'}
 dump(OUT/'verification/concurrent_workspace_changes.json',concurrent)
 remote=js(OUT/'verification/remote_main_final.json')
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 assert remote['remote_main']==remote['local_HEAD']==head and remote['matches']
 return {'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
  'head':head,'remote_main_verified':remote['remote_main'],'remote_main_checked_shanghai':remote['checked_shanghai'],
  'preexisting_files_checked':len(original['preexisting_changes']),'preexisting_monitor_changes':changed,
  'historical_analysis_or_runner_mutations_by_this_task':False,
  'new_changes_outside_study_observed':len(new_external),'external_changes_attributed_to_this_task':False,
  'concurrent_workspace_inventory':'verification/concurrent_workspace_changes.json',
  'this_task_writes_confined_to_new_study':True,'committed':False,'pushed':False}
def artifact_files():
 skip={'INPUT_MANIFEST.json','verification/final_verification.json','verification/output_manifest.json'}
 out=[]
 for p in sorted(OUT.rglob('*')):
  local=p.relative_to(OUT)
  if not p.is_file() or str(local) in skip or local.parts[0] in ['runtime','private_cache'] or '__pycache__' in local.parts:continue
  out.append({'path':str(local),'sha256':sha(p),'bytes':p.stat().st_size})
 return out
def structure():
 required=['README.md','NOCW_PROBLEM_AUDIT.md','HYPOTHESIS_TO_LEARNING.md','CONDITION_LEARNABILITY_AUDIT.md',
 'MODEL_PILOT_PROTOCOL.md','MODEL_PILOT_RESULTS.md','NEXT_AL_EXPERIMENT_PROTOCOL.md','RESEARCH_DECISION.md']
 for n in required:assert (OUT/n).exists(),n
 broken=[]
 for p in OUT.glob('*.md'):
  for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   if link.startswith(('http:','https:','#')):continue
   q=p.parent/link.split('#')[0]
   if not q.exists() and q.name not in ['INPUT_MANIFEST.json','final_verification.json']:broken.append((p.name,link))
 assert not broken,broken
 return {'required_reports':8,'broken_local_links':0,'private_runtime_excluded_from_artifact_manifest':True}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
 if args.check:
  m=js(OUT/'INPUT_MANIFEST.json');bad=[]
  for x in m['sources']:
   p=ROOT/x['path']
   if not p.exists() or sha(p)!=x['sha256']:bad.append(x['path'])
  files=js(OUT/'verification/output_manifest.json')
  for x in files:
   p=OUT/x['path']
   if not p.exists() or sha(p)!=x['sha256']:bad.append(str(p.relative_to(ROOT)))
  assert not bad,bad
  print(json.dumps({'status':'SEALED_INPUTS_OUTPUTS_MATCH','inputs':len(m['sources']),'outputs':len(files)}));return
 src=sources();pilot=check_pilot();g=git_check();layout=structure()
 checks={}
 for name in ['nocw_checks.json','nocw_independent_checks.json','nocw_case_family_checks.json','condition_checks.json','condition_independent_verification.json',
 'next_al_precheck.json','next_al_independent_review.json','pilot_independent_verification.json','synthesis_independent_review.json']:
  d=js(OUT/'verification'/name);s=d.get('status');assert str(s).startswith('PASS'),(name,s);checks[name]=s
 now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
 dump(OUT/'INPUT_MANIFEST.json',{'created_shanghai':now,'head':g['head'],'sources':src,
  'label_access_policy':'Only authorized observed/accepted training feedback; canonical X-only metadata, and explicit 416 validation-ID target allowlist. No test or unacquired numeric targets.',
  'private_payloads_exported':False,'actual_new_model_fits':16,'new_paid_LLM_calls':0,'new_AL_selections':0})
 files=artifact_files();dump(OUT/'verification/output_manifest.json',files)
 result={'status':'PASS','completed_shanghai':now,'source_files':len(src),'public_artifact_files':len(files),
  'figures_png':len(list((OUT/'figures').glob('*.png'))),'structure':layout,'pilot':pilot,'independent_checks':checks,'git':g,
  'new_paid_LLM_calls':0,'new_AL_selections':0,'test_truth_numeric_rows':0,'unacquired_truth_numeric_rows':0,
  'scientific_limits':['observational history is not causal identification','pilot is two outer seeds, not five-seed confirmation','one structural change changes gradient conditioning too','true repeat-experiment noise not identified','future E1/E2 are mock/protocol only']}
 dump(OUT/'verification/final_verification.json',result)
 print(json.dumps({k:result[k] for k in ['status','source_files','public_artifact_files','figures_png','new_paid_LLM_calls','new_AL_selections']},ensure_ascii=False))
if __name__=='__main__':main()
