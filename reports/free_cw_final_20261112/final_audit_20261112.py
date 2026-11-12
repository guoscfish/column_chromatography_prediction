"""Final offline audit: saved artifacts only; no provider, fitting or truth access."""
from pathlib import Path
import collections, csv, datetime, hashlib, html, importlib.util, json, math, re, statistics

ROOT=Path('/Users/fish/Documents/GitHub/column_chromatography_prediction')
S=ROOT/'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101'
OUT=S/'reports'
SEEDS=(157,887,2357,6101,12203)
ARM='free_cw_sol61_xhigh_rerun32'
CURRENT='new_free_cw_sol61_xhigh'
MILESTONES=(525,813,1005,1389,1485)
cache={}
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    p=Path(p).resolve()
    if p not in cache: cache[p]=hashlib.sha256(p.read_bytes()).hexdigest()
    return cache[p]
def stable(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def ids_hash(x): return stable(sorted(map(str,x)))
def check(x,message):
    if not x: raise RuntimeError(message)
def near(a,b): return math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-12)
def summary(v): return {'mean':statistics.mean(v),'seed_std_ddof1':statistics.stdev(v)}
def area(points):
    return math.fsum((float(b['budget'])-float(a['budget']))*(float(a['validation_selection_score'])+float(b['validation_selection_score']))/2 for a,b in zip(points,points[1:]))/(int(points[-1]['budget'])-int(points[0]['budget']))

protocol=stable(read(S/'protocol.json'))
served=read(S/'served_model_lock.json')['served_model']
check(served=='gpt-6.1-sol','served model lock drift')
seed_results=[]; attempt_count=0; accepted=0; completed=0; success_turns=0
fatal401=[]; errors=collections.Counter(); receipt_sources=set()
for seed in SEEDS:
    initial=read(S/f'offline_checks/initial_state_seed_{seed}.json')
    labeled=list(initial['roles']['l0']); remaining=set(initial['roles']['u0'])
    check(len(labeled)==333 and len(set(labeled))==333,'initial IDs')
    test=set(initial['roles']['test']); validation=set(initial['roles']['validation'])
    check(not (set(labeled)|remaining)&(test|validation),'role overlap')
    runtime=S/f'runtime/seed_{seed}/{ARM}'
    trajectory=read(runtime/'trajectory_freeze.json')
    check(trajectory['final_active_labels']==1485 and trajectory['seed']==seed and trajectory['method']==ARM,'trajectory identity')
    check(len(trajectory['files'])==145,'trajectory file coverage')
    for path,digest in trajectory['files'].items(): check(sha(S/path)==digest,'trajectory hash '+path)
    chosen_all=[]; starts_count=0; batch_rows=[]
    for r in range(36):
        d=S/f'selections/seed_{seed}/{ARM}/round_{r:02d}'
        marker=read(S/f'progress/completed_seed_{seed}_round_{r:02d}.json')
        check((marker['seed'],marker['round'],marker['budget'])==(seed,r,365+32*r),'completion identity')
        for path,digest in marker['files'].items(): check(sha(S/path)==digest,'completion hash '+path)
        frozen=read(d/'batch_freeze.json'); contract=read(d/'contract.json')
        for name,digest in frozen['artifact_hashes'].items(): check(sha(d/name)==digest,'batch binding '+str(d/name))
        batch=frozen['batch_ids']
        check(len(batch)==len(set(batch))==32 and set(batch)<=remaining and not set(batch)&set(labeled),'batch IDs')
        check((contract['seed'],contract['method'],contract['round'],contract['variant'])==(seed,ARM,r,'cw_aware'),'contract identity')
        check(contract['protocol_hash']==frozen['protocol_hash']==protocol and not contract['dry_run'] and not frozen['dry_run'],'protocol/dry run')
        check(not frozen['pending_ids'] and frozen['new_batch_labels_revealed'] is False,'premeasurement freeze')
        check(contract['L_t_ids']==labeled and set(contract['U_t_ids'])==remaining,'trajectory prefix')
        check(contract['input_sha256']==sha(d/'input.json') and contract['catalog_sha256']==sha(d/'catalog.json'),'full input/catalog binding')
        packet=read(d/'input.json'); catalog=read(d/'catalog.json'); selection=read(d/'selection.json')
        check(packet['packet_sha256']==stable({k:v for k,v in packet.items() if k!='packet_sha256'}),'packet hash')
        check(packet['selection_count']==32 and packet['active_label_count']==len(labeled) and packet['observed_count']==len(labeled),'packet label count')
        check(packet['candidate_count']==len(remaining)==len(catalog['candidates']) and packet['delivery']=='all_candidates_one_request','complete pool')
        check(packet['validation_test_records']==0 and not catalog['pending'],'packet barrier')
        check(len(packet['candidates']['rows'])==len(remaining) and len(packet['observed']['rows'])==len(labeled),'full packet rows')
        check({row['id'] for row in catalog['candidates']}==remaining,'catalog identity')
        check(selection['packet_sha256']==packet['packet_sha256'] and [x['id'] for x in selection['choices']]==batch,'accepted plan identity')
        prev_freeze=read(runtime/f'round_{r:02d}/prediction_freeze.json')
        check(contract['checkpoint_sha256']==frozen['checkpoint_sha256']==prev_freeze['checkpoint_sha256'],'selection predictor binding')
        accepted_turns=[]; requests={}; request_hashes={}
        for p in sorted(d.glob('request_*.json')):
            req=read(p); turn=int(p.stem.split('_')[1]); requests[turn]=req; request_hashes[turn]=stable(req)
            check(req['config']['model']=='gpt-6.1-sol' and req['config']['reasoning_effort']=='xhigh','request model/effort')
            check(not req['config'].get('tools') and req['config']['store'] is False,'request tools/store')
            check(req['messages'][1]['role']=='user' and json.loads(req['messages'][1]['content'])==packet,'untruncated request packet')
            check(len(req['messages'])==2+2*turn,'full correction message history')
        for p in sorted(d.glob('turn_*.json')):
            turn=int(p.stem.split('_')[1]); x=read(p); prov=x['provenance']
            check(x['success'] and prov.get('served_model')==served and not prov.get('mock') and isinstance(prov.get('response_id'),str),'success response identity')
            check(prov.get('requested_model')=='gpt-6.1-sol' and prov.get('reasoning_effort')=='xhigh','response provenance model/effort')
            check(x['request_sha256']==request_hashes[turn],'response request binding')
            if json.loads(x['answer'])==selection: accepted_turns.append(turn)
            success_turns+=1
        check(accepted_turns,'no accepted provider answer')
        starts=list(d.glob('attempt_*.started.json')); starts_count+=len(starts)
        for p in starts:
            receipt=p.with_name(p.name.replace('.started.json','.json'))
            check(receipt.exists(),'unresolved provider request '+str(p))
            started=read(p); x=read(receipt); turn=int(p.name.split('_')[1])
            check(started['request_sha256']==x['request_sha256']==request_hashes[turn],'attempt request drift')
            receipt_sources.add(str(receipt.relative_to(S)))
            if not x['success']:
                errors[(x.get('error_type'),x.get('http_status'))]+=1
                if x.get('http_status')==401 or re.search(r'\b401\b',x.get('reason','')): fatal401.append(str(receipt.relative_to(S)))
        feedback=read(d/'feedback.json'); receipt=read(d/'label_access_receipt.json'); audit=receipt['audit']
        check(feedback['batch_freeze_sha256']==receipt['batch_freeze_sha256']==sha(d/'batch_freeze.json'),'label reveal freeze binding')
        check(receipt['feedback_sha256']==stable(feedback),'feedback binding')
        check(audit['purpose']=='after_acquisition_fit' and audit['acquisitions_frozen'] and audit['requested_rows']==32 and audit['requested_ids_hash']==ids_hash(batch),'selected-label reveal receipt')
        check([x['candidate_id'] for x in feedback['records']]==batch,'feedback IDs')
        labeled+=batch; remaining-=set(batch); chosen_all+=batch
        fit=read(runtime/f'round_{r+1:02d}/model/fit_audit.json'); pred=read(runtime/f'round_{r+1:02d}/prediction_freeze.json')
        check(fit['train_rows']==len(labeled) and fit['train_ids_hash']==pred['L_t_ids_hash']==ids_hash(labeled),'training ID lineage')
        check(fit['initialization_hash']==initial['initialization_hash']==pred['initialization_hash'],'initialization')
        check(fit['test_labels_used_for_fit_or_checkpoint_selection']==pred['test_truth_access_count']==0,'test barrier')
        check(sha(Path(pred['checkpoint_path']))==pred['checkpoint_sha256']==fit['checkpoint_sha256'],'checkpoint bytes')
        check(sha(Path(pred['prediction_path']))==pred['prediction_sha256']==fit['prediction_sha256'],'prediction bytes')
        accepted+=1;completed+=1
        batch_rows.append({'round':r+1,'budget':len(labeled),'batch_freeze_sha256':sha(d/'batch_freeze.json'),'completion_sha256':sha(S/f'progress/completed_seed_{seed}_round_{r:02d}.json')})
    check(trajectory['selected_ids']==chosen_all and len(set(chosen_all))==1152,'full trajectory selections')
    attempt_count+=starts_count
    seed_results.append({'seed':seed,'rounds':36,'budget':1485,'selected_unique':1152,'transmission_attempts':starts_count,'trajectory_sha256':sha(runtime/'trajectory_freeze.json'),'last_completed_utc':marker['completed_utc'],'batches':batch_rows})
    print(json.dumps({'checked_seed':seed,'rounds':36,'attempts':starts_count,'hashed_files':len(cache)}),flush=True)

manual=read(S/'manual_resume_20261109/authorization.json')
# The current validation already pins all historical exceptions; retain an explicit final list.
check(set(fatal401)==set(manual['preserved_fatal_receipts']) and len(fatal401)==5,'historical HTTP401 receipt set mismatch')
for name,digest in manual['preserved_fatal_receipts'].items(): check(sha(S/name)==digest,'historical HTTP401 changed')
check(attempt_count==173511 and accepted==completed==180,'terminal counts')

# Re-read saved validation histories and checkpoint/prediction hashes using the report-only module.
spec=importlib.util.spec_from_file_location('offline_final_report',ROOT/'scripts/studies/report_free_cw_sol61_five_20261101.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
with (OUT/'validation_curves.csv').open() as f: curve_rows=list(csv.DictReader(f))
groups=collections.defaultdict(list)
for row in curve_rows: groups[row['method'],int(row['seed'])].append(row)
for key,points in groups.items():
    points.sort(key=lambda x:int(x['budget']))
    check([int(x['budget']) for x in points]==list(range(333,334+32*(len(points)-1),32)),'noncontiguous curve')
    for r,row in enumerate(points):
        saved=report.saved_point(key[0],key[1],r)
        check(saved is not None,'missing saved predictor')
        for k,v in saved.items():
            check((row[k]==str(v)) if isinstance(v,bool) or not isinstance(v,(float,int)) else near(row[k],v),'saved curve mismatch '+str(key)+' '+k)
        fit=read(ROOT/row['source'])
        check(fit['initialization_hash']==read(S/f'offline_checks/initial_state_seed_{key[1]}.json')['initialization_hash'],'comparison initialization mismatch')
        ref_initial=read(ROOT/groups[CURRENT,key[1]][0]['source'])
        check(fit['validation_ids_hash']==ref_initial['validation_ids_hash'],'comparison validation split mismatch')
    print(json.dumps({'checked_curve':key,'points':len(points)}),flush=True)

comparison=read(OUT/'comparison_summary.json')
check(comparison['test_truth_reads']==0 and comparison['selector_feedback'] is False,'report barrier')
check('Astra/high' in comparison['historical_free_model_mixing'] and 'mixed_model' in ''.join(report.METHODS),'historical mixed-model tag')
with (OUT/'paired_differences.csv').open() as f: paired=list(csv.DictReader(f))
paired_index={(r['comparator'],int(r['budget']),int(r['seed']),r['metric']):r for r in paired}
for row in comparison['comparisons']:
    comp,budget,metric=row['comparator'],row['budget'],row['metric'];new=[];ref=[];delta=[]
    fair=min(int(groups[m,seed][-1]['budget']) for m in (CURRENT,comp) for seed in SEEDS)
    check(budget<=fair and row['seeds']==list(SEEDS),'unfair comparison endpoint')
    for seed in SEEDS:
        a=[x for x in groups[CURRENT,seed] if int(x['budget'])<=budget];b=[x for x in groups[comp,seed] if int(x['budget'])<=budget]
        av=area(a) if metric=='raw_validation_AULC' else float(a[-1][metric]);bv=area(b) if metric=='raw_validation_AULC' else float(b[-1][metric])
        saved=paired_index[comp,budget,seed,metric]
        check(near(av,saved['new_free']) and near(bv,saved['reference']) and near(av-bv,saved['paired_delta_new_minus_reference']),'paired values')
        new.append(av);ref.append(bv);delta.append(av-bv)
    for name,values in [('new_free',new),('reference',ref),('paired_delta_new_minus_reference',delta)]:
        for k,v in summary(values).items():check(near(row[name][k],v),'paired mean/sample standard deviation')

milestones=[]
for budget in MILESTONES:
    x=read(OUT/f'milestone_L{budget}.json');check(x['budget']==budget and [r['seed'] for r in x['seeds']]==list(SEEDS),'milestone seeds')
    vals=[];areas=[]
    for row in x['seeds']:
        points=[p for p in groups[CURRENT,row['seed']] if int(p['budget'])<=budget]
        check(near(row['point']['validation_selection_score'],points[-1]['validation_selection_score']) and near(row['raw_validation_AULC'],area(points)),'milestone points/AULC')
        vals.append(float(points[-1]['validation_selection_score']));areas.append(area(points))
    for key,values in [('validation_selection_score',vals),('raw_validation_AULC',areas)]:
        for k,v in summary(values).items():check(near(x['summary'][key][k],v),'milestone summary')
    milestones.append({'budget':budget,'validation_selection_score':summary(vals),'raw_validation_AULC':summary(areas),'sha256':sha(OUT/f'milestone_L{budget}.json')})

costs=read(OUT/'costs.json')
check(costs['request_count']==attempt_count and costs['accepted_batches']==accepted and costs['completed_fits']==completed,'automatic cost report counts')
check({x['source'] for x in costs['requests']}==receipt_sources,'automatic costs retain every receipt')
check(all(x['requested_model']=='gpt-6.1-sol' and x['reasoning_effort']=='xhigh' and (not x['success'] or x['served_model']==served) for x in costs['requests']),'cost response model identity')
check(costs['currency_cost'] is None and costs['monetary_price_verified'] is False,'unverified money not invented')
fair_comparisons=[]
for comp in report.METHODS[1:]:
    rows=[x for x in comparison['comparisons'] if x['comparator']==comp];budget=max(x['budget'] for x in rows)
    fair_comparisons.append({'comparator':comp,'budget':budget,**{x['metric']:x for x in rows if x['budget']==budget and x['metric'] in ('raw_validation_AULC','validation_selection_score')}})

result={'status':'FINAL_OFFLINE_AUDIT_PASSED','checked_shanghai':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'accepted_batches':accepted,'completed_fits':completed,'transmission_attempts':attempt_count,'successful_response_turns':success_turns,
 'hashed_artifact_files':len(cache),'new_provider_requests':0,'new_training_runs':0,'test_truth_reads':0,'selector_feedback':False,
 'model':served,'reasoning_effort':'xhigh','terminal_seed_results':seed_results,'historical_http401_receipts':fatal401,
 'failure_receipt_counts':[{'error_type':k[0],'http_status':k[1],'count':v} for k,v in sorted(errors.items(),key=lambda t:str(t[0]))],
 'milestones':milestones,'fair_comparisons':fair_comparisons,'historical_free_model_mixing':comparison['historical_free_model_mixing'],
 'AULC':'normalized trapezoidal integral of every raw saved validation point; no smoothing, monotone envelope, or best-so-far',
 'metric_direction':'validation_selection_score is combined normalized validation RMSE; lower score and lower AULC are better',
 'limitations':comparison['causality'],'automatic_report_hashes':{p.name:sha(p) for p in OUT.glob('*') if p.is_file() and p.name in ('REPORT.md','comparison_summary.json','paired_differences.csv','validation_curves.csv','completion.csv','costs.json')}}
dest=OUT/'final_verification_20261112.json'
if dest.exists():
    previous=read(dest)
    check({k:v for k,v in previous.items() if k!='checked_shanghai'}=={k:v for k,v in result.items() if k!='checked_shanghai'},'existing final audit differs; preserved without overwrite')
else:
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:result[k] for k in ('status','accepted_batches','completed_fits','transmission_attempts','hashed_artifact_files')},ensure_ascii=False),flush=True)
