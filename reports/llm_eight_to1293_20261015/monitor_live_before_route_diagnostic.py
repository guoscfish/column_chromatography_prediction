import json,time,subprocess,datetime
from pathlib import Path
ROOT=Path('/Users/fish/Documents/GitHub/column_chromatography_prediction');P=ROOT/'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/sol_xhigh_all_strategies_to1293_20261015';OUT=ROOT/'reports/llm_eight_to1293_20261015';OUT.mkdir(exist_ok=True)
METHODS=('cw16_llm16_scientist_v2','free_llm32_scientist_v2','center_width_lcmd','random32');last=None;printed=0
while True:
 events=[]
 for line in (P/'execution_progress.jsonl').read_text().splitlines():
  try:events.append(json.loads(line))
  except ValueError:pass
 active={}
 for e in events:
  if e['action']=='LLM_request_started':active[e['pid']]=e
  elif e['action']=='LLM_request_finished':active.pop(e['pid'],None)
 processes={}
 for line in subprocess.check_output(['ps','-axo','pid,stat,args']).decode().splitlines():
  fields=line.strip().split(None,2)
  if len(fields)==3 and fields[2].startswith('/Users/fish/miniforge3/envs/fish/bin/python') and str(P) in fields[2] and ('--worker' in fields[2] or 'resume_hybrid6101_round21_extra.py' in fields[2]) and not fields[1].startswith('Z'):processes[int(fields[0])]=fields[2]
 rows=[]
 for seed in (157,6101):
  for method in METHODS:
   fs=sorted((P/f'runtime/seed_{seed}/{method}').glob('round_*/prediction_freeze.json'));budget=1165;value=None
   if fs:
    f=json.loads(fs[-1].read_text());a=json.loads((Path(f['checkpoint_path']).parent/'fit_audit.json').read_text());budget=a['train_rows'];value=a['best_validation_combined_normalized_rmse']
   worker=next((e for e in reversed(events) if e['action']=='worker_started' and e['seed']==seed and e['method']==method),None)
   recent=[e for e in events if worker and e.get('seed')==seed and e.get('method')==method and e['time']>=worker['time'] and e['action'] in ('training_started','training_completed','training_queued','select_started','control_select_started','requires_decision','transport_retry')];phase='等待'
   e=recent[-1] if recent else None;pid=worker['pid'] if worker else None
   if budget==1293:phase='已冻结1293'
   elif not pid or pid not in processes:phase='进程未运行'
   elif pid in active:phase='LLM请求中 '+str(int(time.time()-active[pid]['time']))+'秒'
   elif e:
    phase={'training_started':'训练'+str(333+32*(e['round']+1)),'training_completed':'准备下一轮','training_queued':'等待训练资源','select_started':'准备选样/等待本seed请求锁','control_select_started':'准备选样/等待计算资源','requires_decision':'需要决定','transport_retry':'传输重试'}.get(e['action'],'等待')
   rows.append({'seed':seed,'method':method,'labels':budget,'validation_nrmse':value,'phase':phase,'pid':pid})
 stamp=datetime.datetime.now().astimezone().isoformat();result={'updated_at':stamp,'rows':rows,'quota_seeds':[seed for seed in (157,6101) if (P/f'quota_seed_{seed}.json').exists()],'all_complete':(P/'result.json').exists()};(OUT/'live_status.json').write_text(json.dumps(result,indent=2)+'\n')
 signature=tuple((r['seed'],r['method'],r['labels'],r['phase'].split(' ')[0]) for r in rows)
 if signature!=last or time.time()-printed>=60:print(stamp,[(r['seed'],r['method'],r['labels'],r['phase']) for r in rows],flush=True);last=signature;printed=time.time()
 if result['all_complete']:break
 time.sleep(10)
