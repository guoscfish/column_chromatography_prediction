from pathlib import Path
import json,hashlib,collections
R=Path('/Users/fish/Documents/GitHub/column_chromatography_prediction');P=R/'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101';ARM='free_cw_sol61_xhigh_rerun32'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();stable=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest();ids_hash=lambda x:stable(sorted(map(str,x)))
errors=[];seeds=[];responses={};configs=collections.Counter();served=collections.Counter();req=collections.Counter();effort=collections.Counter();snap=read(P/'source_snapshot.json');sourcechecks=0
for n,h in snap['frozen_hashes'].items():
 sourcechecks+=1
 if sha(P/'frozen_source'/n)!=h:errors.append(['frozen_source',n])
authchecks=0
for n,root in [('persistent_20261102_v2/authorization.json',R),('transport_recovery_20261102/authorization.json',P),('transport_recovery_20261102/controller_authorization.json',P),('http403_recovery_20261102/authorization.json',P),('immediate_retry_20261102/authorization.json',P)]:
 a=read(P/n)
 for f,h in a['code_hashes'].items():
  authchecks+=1
  if sha(root/f)!=h:errors.append(['authorization_code',n,f])
 for f,h in a.get('preserved_fatal_receipts',{}).items():
  authchecks+=1
  if sha(P/f)!=h:errors.append(['preserved_receipt',f])
counts=collections.Counter()
for seed in [157,887,2357,6101,12203]:
 sd=P/f'selections/seed_{seed}/{ARM}';rs=sorted(sd.glob('round_*/batch_freeze.json'));history=[];seen=set();totalturns=0;rejects=0;recovered=0;tokens=[];rounddata=[]
 for fr in rs:
  counts['accepted_batches']+=1;d=fr.parent;round=int(d.name.split('_')[1]);f=read(fr);c=read(d/'contract.json');packet=read(d/'input.json');sel=read(d/'selection.json')
  def error(*s):errors.append([seed,round,*s])
  for n,h in f['artifact_hashes'].items():
   counts['frozen_artifact_refs']+=1
   if sha(d/n)!=h:error('frozen_artifact_drift',n)
  if len(f['batch_ids'])!=32 or len(set(f['batch_ids']))!=32 or set(f['batch_ids'])&seen:error('duplicate_batch')
  seen.update(f['batch_ids'])
  if set(f['batch_ids'])-set(c['U_t_ids']) or set(f['batch_ids'])&set(c['L_t_ids']):error('illegal_pool')
  if len(c['L_t_ids'])!=333+32*round or len(c['U_t_ids'])!=2997-32*round:error('budget')
  if packet['packet_sha256']!=stable({k:v for k,v in packet.items() if k!='packet_sha256'}):error('packet_hash')
  if packet['candidate_count']!=len(packet['candidates']['rows']) or packet['observed_count']!=len(packet['observed']['rows']):error('truncated_table')
  records=[r for h in history for r in h['records']];last=history[-1] if history else {};high=sorted(records,key=lambda r:(-max(abs(r['error_V1_ml']),abs(r['error_V2_ml'])),r['candidate_id']))[:8]
  expected={'previous_hypotheses':last.get('hypotheses',[]),'previous_feedback_interpretation':last.get('feedback_interpretation',''),'previous_batch_rationale':last.get('batch_rationale',''),'previous_selected_ids':[r['candidate_id'] for r in records],'recent_outcomes':last.get('records',[]),'high_error_measured_examples':high,'memory_policy':'verbatim previous LLM analysis; exact latest 32 and top-8 absolute-error records; older records queryable'}
  if expected!=packet['memory']:error('memory_chain')
  counts['memory_chain_checks']+=1
  accepted=[];turnfiles=sorted(d.glob('turn_*.json'))
  for t in turnfiles:
   totalturns+=1;counts['successful_logical_turns']+=1;a=read(t);turn=t.stem.split('_')[1];rq=read(d/f'request_{turn}.json')
   if a['request_sha256']!=stable(rq):error('request_hash',t.name)
   if json.loads(rq['messages'][1]['content'])!=packet:error('request_missing_packet',t.name)
   if len(rq['messages'])!=2+2*int(turn):error('correction_transcript_length',t.name)
   configs[stable(rq['config'])]+=1;pro=a['provenance'];served[pro.get('served_model')]+=1;req[pro.get('requested_model')]+=1;effort[pro.get('reasoning_effort')]+=1
   rid=pro.get('response_id')
   if rid in responses:error('response_id_reused_across_turns',t.name,responses[rid])
   responses[rid]=str(t.relative_to(P));tokens.append(pro.get('usage',{}).get('input_tokens',0))
   if pro.get('mock') or pro.get('native_tool_calls',0)!=0:error('mock_tools')
   try:
    if json.loads(a['answer'])==sel:accepted.append(t.name)
   except ValueError:pass
   recovered+=bool(a.get('recovered_local_completion'))
  if len(accepted)!=1:error('accepted_turn_count',len(accepted))
  rejects+=len(list(d.glob('rejected_*.json')))
  fb=read(d/'feedback.json')
  if fb['batch_freeze_sha256']!=sha(fr):error('feedback_freeze')
  if (fb['seed'],fb['method'],fb['round'])!=(seed,ARM,round):error('foreign_feedback')
  history.append(fb)
  md=P/f'runtime/seed_{seed}/{ARM}/round_{round+1:02d}';marker=P/f'progress/completed_seed_{seed}_round_{round:02d}.json'
  if not marker.exists():error('no_completion');continue
  m=read(marker);counts['completed_fit_markers']+=1
  for n,h in m['files'].items():
   counts['completion_artifact_refs']+=1
   if sha(P/n)!=h:error('completion_drift',n)
  audit=read(md/'model/fit_audit.json');pred=read(md/'prediction_freeze.json')
  if sha(md/'model/best.pt')!=audit['checkpoint_sha256']:error('checkpoint_hash')
  if sha(md/'model/predictions.csv.gz')!=audit['prediction_sha256']:error('prediction_hash')
  if any(pred[k]!=audit[k] for k in ('checkpoint_sha256','prediction_sha256','initialization_hash')):error('freeze_fit_binding')
  if audit['test_labels_used_for_fit_or_checkpoint_selection'] or pred['test_truth_access_count']:error('test_truth_access')
  if audit['train_rows']!=365+32*round or audit['initialization_seed']!=seed+2000003:error('training_budget_or_init')
  initial=read(P/f'offline_checks/initial_state_seed_{seed}.json')
  if audit['initialization_hash']!=initial['initialization_hash']:error('initialization_hash')
  rounddata.append({'round':round,'accepted_turn':accepted,'successful_turns':len(turnfiles),'memory_records':len(records),'input_sha256':c['input_sha256'],'completed_utc':m['completed_utc']})
 seeds.append({'seed':seed,'accepted':len(rs),'complete_rounds':len(rounddata),'successful_turns':totalturns,'rejected_logical_answers':rejects,'recovered_local_completion_turns':recovered,'selected_duplicate_count':32*len(rs)-len(seen),'input_tokens_range':[min(tokens),max(tokens)],'rounds':rounddata})
summary={'source_hash_checks':sourcechecks,'authorization_hash_checks':authchecks,'counts':dict(counts),'served_models':dict(served),'requested_models':dict(req),'reasoning_efforts':dict(effort),'config_hashes':dict(configs),'errors':errors,'seeds':seeds}
Path('/tmp/cw_runtime_science_audit_20261103.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='seeds'},indent=2));print(json.dumps([{k:v for k,v in s.items() if k!='rounds'} for s in seeds],indent=2))
