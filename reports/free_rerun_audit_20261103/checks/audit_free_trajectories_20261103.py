import json,importlib.util,statistics,hashlib
from pathlib import Path
s=importlib.util.spec_from_file_location('m','scripts/studies/report_free_cw_sol61_five_20261101.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def rel(p):return str(p.relative_to(m.ROOT))
def read(p):return json.loads(p.read_text())
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def record(method,seed,r):
 model,freeze=m.model_directory(method,seed,r)
 a=read(model/'fit_audit.json')
 row={'r':r,'L':333+32*r,'model':rel(model),'metric':a['best_validation_combined_normalized_rmse'],**{k:a[k] for k in ['train_ids_hash','initialization_seed','initialization_hash','validation_ids_hash','checkpoint_sha256','fit_contract_hash','best_epoch']}}
 if r:
  p=model.parents[4]/f'selections/seed_{seed}'/model.parents[1].name/f'round_{r-1:02d}'
  f=read(p/'batch_freeze.json');row['batch_path']=rel(p);row['ids']=f['batch_ids'];row['turns']=[]
  row['artifact_hash_errors']=[n for n,v in f['artifact_hashes'].items() if h(p/n)!=v]
  for n in f['artifact_hashes']:
   if n.startswith('turn_') and n.endswith('.json') and 'started' not in n:
    t=read(p/n);pr=t.get('provenance',{});u=pr.get('usage',{})
    row['turns'].append({'name':n,**{k:pr.get(k) for k in ['served_model','requested_model','reasoning_effort','response_id']},'input_tokens':u.get('input_tokens'),'output_tokens':u.get('output_tokens'),'source':pr.get('source')})
 return row
rows=[];summary=[]
for seed in m.SEEDS:
 pair={method:[record(method,seed,r) for r in range(16)] for method in m.METHODS[:2]}
 rows.append({'seed':seed,'pair':pair})
 new,old=(pair[k] for k in m.METHODS[:2]);cum_n=set();cum_o=set();ovs=[]
 for n,o in zip(new[1:],old[1:]):
  ns,os=set(n['ids']),set(o['ids']);cum_n|=ns;cum_o|=os
  ovs.append({'round':n['r'],'L':n['L'],'overlap':len(ns&os),'ordered_equal':n['ids']==o['ids'],'set_equal':ns==os,'cum_overlap':len(cum_n&cum_o),'checkpoint_equal':n['checkpoint_sha256']==o['checkpoint_sha256'],'metric_old':o['metric'],'metric_new':n['metric'],'delta':n['metric']-o['metric']})
 diff=next(x for x in ovs if not x['set_equal']);metric_diff=next(x for x in ovs if abs(x['delta'])>1e-12)
 sm={'seed':seed,'initial_identical_checkpoint':new[0]['checkpoint_sha256']==old[0]['checkpoint_sha256'],'all_init_hashes_match':all(n['initialization_hash']==o['initialization_hash'] for n,o in zip(new,old)),'all_validation_ids_match':all(n['validation_ids_hash']==o['validation_ids_hash'] for n,o in zip(new,old)),'first_batch_divergence':diff,'first_metric_divergence':metric_diff,'r1_overlap':ovs[0]['overlap'],'cum_L813_overlap_new480':len(cum_n&cum_o),'L813_old':old[-1]['metric'],'L813_new':new[-1]['metric'],'L813_delta':new[-1]['metric']-old[-1]['metric'],'aulc_old':sum((a['metric']+b['metric'])*.5 for a,b in zip(old,old[1:]))/15,'aulc_new':sum((a['metric']+b['metric'])*.5 for a,b in zip(new,new[1:]))/15,'overlaps':ovs,'old_served_sequence':[x['turns'][-1].get('served_model') for x in old[1:]],'new_served_sequence':[x['turns'][-1].get('served_model') for x in new[1:]]}
 summary.append(sm)
 print(json.dumps({k:v for k,v in sm.items() if k!='overlaps'},ensure_ascii=False))
Path('/tmp/free_trajectory_audit_20261103.json').write_text(json.dumps({'summary':summary,'rows':rows},ensure_ascii=False,indent=2))
print('BUDGET MEANS')
for r in [0,1,2,3,6,7,10,11,15]:
 ns=[x['pair'][m.METHODS[0]][r]['metric'] for x in rows];os=[x['pair'][m.METHODS[1]][r]['metric'] for x in rows]; print(333+32*r,statistics.mean(os),statistics.mean(ns),statistics.mean(ns)-statistics.mean(os))
