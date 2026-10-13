from pathlib import Path
import json,hashlib,re
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
NEW=ROOT/'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/free_sol_xhigh_continuation_20261012'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(NEW/'transition_manifest.json');OLD=Path(manifest['source_revision'])
assert sha(OLD/'protocol.json')==manifest['original_protocol_sha256']
assert sha(OLD/'extension_authorized_to813/manifest.json')==manifest['original_manifest_sha256']
for name,h in read(OLD/'protocol.json')['code_hashes'].items():assert sha(ROOT/name)==h,name
for name,h in manifest['independent_modules_sha256'].items():assert sha(ROOT/name)==h,name
for name,h in manifest['inherited_files'].items():assert sha(NEW/name)==h,name
summary={};frozen={}
for seed in (157,6101):
 result=read(NEW/f'{seed}_result.json');assert result['final_labels']==813
 selected=[];files={};curve=[]
 for r in range(15):
  d=NEW/f'selections/seed_{seed}/free_llm32_scientist_v2/round_{r:02d}';b=read(d/'batch_freeze.json');raw=read(d/'catalog.json')
  assert not b['pending_ids'] and len(b['batch_ids'])==len(set(b['batch_ids']))==32
  assert set(b['batch_ids'])<={x['id'] for x in raw['candidates']}
  for name,h in b['artifact_hashes'].items():assert sha(d/name)==h
  assert read(d/'feedback.json')['batch_freeze_sha256']==sha(d/'batch_freeze.json')
  assert read(d/'label_access_receipt.json')['audit']['purpose']=='after_acquisition_fit'
  selected+=b['batch_ids']
  for name in ('batch_freeze.json','feedback.json','label_access_receipt.json'):files[str((d/name).relative_to(NEW))]=sha(d/name)
  if 333+32*r>=manifest['start_labels'][str(seed)]:
   config=read(d/'model_transition.json');assert config['model']=='gpt-6.1-sol' and config['reasoning_effort']=='xhigh'
   for f in d.glob('turn_*.json'):
    if not re.fullmatch(r'turn_\d\d.json',f.name):continue
    model=read(f)['provenance']['served_model'];assert model=='gpt-6.1-sol' or model.startswith('gpt-6.1-sol-')
 assert len(selected)==len(set(selected))==480 and selected==result['selected_ids']
 for r in range(16):
  f=NEW/f'runtime/seed_{seed}/free_llm32_scientist_v2/round_{r:02d}/prediction_freeze.json';d=read(f)
  assert d['test_truth_access_count']==0
  for kind in ('checkpoint','prediction'):assert sha(Path(d[kind+'_path']))==d[kind+'_sha256']
  a=read(Path(d['checkpoint_path']).parent/'fit_audit.json');assert a['train_rows']==333+32*r
  curve.append({'labels':a['train_rows'],'nrmse':a['best_validation_combined_normalized_rmse']});files[str(f.relative_to(NEW))]=sha(f)
 area=sum((a['nrmse']+b['nrmse'])*.5*(b['labels']-a['labels']) for a,b in zip(curve,curve[1:]))/480
 summary[str(seed)]={'labels':813,'batches':15,'unique_added_ids':480,'final_validation_nrmse':curve[-1]['nrmse'],'normalized_validation_aulc_333_813':area,'switch_after_labels':manifest['start_labels'][str(seed)]}
 record={'seed':seed,'method':'free_llm32_scientist_v2','final_active_labels':813,'selected_ids':selected,'files':files,'validation_learning_curve':curve,'test_evaluation_performed':False,'classification':'Astra/high prefix, Sol/xhigh continuation; exploratory two-seed validation'}
 path=NEW/f'{seed}_free_trajectory_freeze.json'
 if path.exists():assert read(path)==record
 else:path.write_text(json.dumps(record,indent=2)+'\n')
 frozen[str(seed)]=sha(path)
summary['audit']={'all_original_source_hashes_verified':True,'original_manifest_unchanged':True,'all_inherited_prefix_hashes_verified':True,'all_checkpoint_prediction_hashes_verified':True,'trajectory_freeze_sha256':frozen,'test_evaluation_performed':False,'historical_deviations':'seed157 round5 fourth response plus authorized character repair; seed6101 round0 fourth response','model_transition':'seed157 after525; seed6101 after653; not pureSol trajectories'}
(OUT/'completion_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
