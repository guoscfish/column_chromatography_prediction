"""Frozen, read-only Level-1 history audit. No runner/model/label-store imports.

Default replays history_snapshot_manifest.json and fails on source drift. --snapshot
creates the explicit cohort once; --refresh-snapshot is deliberately unsupported.
Only completed, frozen batches enter rates; active rounds are inventory-only.
"""
from __future__ import annotations
import argparse, ast, collections, copy, csv, hashlib, json, math, re, statistics
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STUDIES = ROOT / 'studies/active_learning'
REV = STUDIES / 'qgeognn_v2_row_llm_scientist_v2/revisions'
NOCW = STUDIES / 'qgeognn_v2_row_llm_nocw_ablation/revisions/five_seed_nocw_to1005_20261029'
FREE = 'free_llm32_scientist_v2'
HYBRID = 'cw16_llm16_scientist_v2'
SEEDS = (157,887,2357,6101,12203)
OLD = ((21,'sol_xhigh_hybrid_free_to1005_20261013'),(26,'sol_xhigh_all_strategies_to1165_20261014'),(30,'sol_xhigh_all_strategies_to1293_20261015'),(34,'sol_xhigh_all_strategies_to1421_20261017'),(36,'sol_xhigh_all_strategies_to1549_20261018'))
NEW = ((21,'sol61_xhigh_free_five_seeds_to1005_20261023'),(36,'sol61_xhigh_free_five_seeds_to1485_20261024'))
AXES = ('PE/EA','Density g/ml','V/ul','loading solvent','Volume of loading solvent/ul')
SNAPSHOT = HERE / 'history_snapshot_manifest.json'


def rel(p): return str(Path(p).relative_to(ROOT))
def digest(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def load(p): return json.loads(Path(p).read_text())
def dump(name,value): (HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def stable_hash(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def free_owner(seed,r):
    if seed in {887:33,12203:24} and r+1>{887:33,12203:24}[seed]: return REV/'sol61_xhigh_free_to1485_extra3_20261026'
    return REV/next(n for end,n in (OLD if seed in (157,6101) else NEW) if r+1<=end)
def hybrid_owner(r): return REV/next(n for end,n in OLD if r+1<=end)


def capture_snapshot():
    if SNAPSHOT.exists(): raise RuntimeError('Snapshot already exists; use replay, never silently move cutoff')
    batches=[]; files={}; active=[]
    dirs=[(s,FREE,r,free_owner(s,r)/f'selections/seed_{s}/{FREE}/round_{r:02d}') for s in SEEDS for r in range(36)]
    # All actual frozen Hybrid history: 157 has 36 batches, 6101 has 35.
    for s in (157,6101):
        r=0
        while (hybrid_owner(r)/f'selections/seed_{s}/{HYBRID}/round_{r:02d}/batch_freeze.json').exists():
            dirs.append((s,HYBRID,r,hybrid_owner(r)/f'selections/seed_{s}/{HYBRID}/round_{r:02d}')); r+=1
            if r>=36: break
    for s in SEEDS:
        for d in sorted((NOCW/f'selections/seed_{s}/llm32_nocw').glob('round_*')):
            if (d/'batch_freeze.json').exists(): dirs.append((s,'llm32_nocw',int(d.name.split('_')[1]),d))
            else:
                # Inventory names only; no partial or active response contents are read.
                active.append({'seed':s,'method':'llm32_nocw','round':int(d.name.split('_')[1]),'directory':rel(d),'files_present':sorted(p.name for p in d.iterdir() if p.is_file()),'status':'excluded_unfrozen_inventory_only'})
    for s,m,r,d in dirs:
        frozen=load(d/'batch_freeze.json')
        if frozen.get('dry_run'): raise ValueError('mock batch in formal cohort')
        names=set(frozen['artifact_hashes'])|{'batch_freeze.json'}
        # V2 retry/rejection records were not covered by its batch freeze. Capture
        # completed immutable sidecars separately, explicitly weaker provenance.
        names|={str(p.relative_to(d)) for pattern in ('rejected*.json','authorized*.json','extra_correction*.json','retries/turn_*/attempt_*.json') for p in d.glob(pattern)}
        for name in sorted(names):
            p=d/name; h=digest(p)
            if name in frozen['artifact_hashes'] and h!=frozen['artifact_hashes'][name]: raise ValueError(f'frozen drift: {p}')
            files[rel(p)]={'sha256':h,'bytes':p.stat().st_size,'evidence':'batch_freeze_artifact' if name in frozen['artifact_hashes'] else 'batch_freeze' if name=='batch_freeze.json' else 'completed_snapshot_sidecar'}
        batches.append({'seed':s,'method':m,'round':r,'directory':rel(d),'batch_freeze_sha256':digest(d/'batch_freeze.json'),'source_files':sorted(rel(d/name) for name in names)})
    for p in (ROOT/'src/qgeognn_al/active_learning_v2/scientist_full_delivery.py', ROOT/'src/qgeognn_al/active_learning_v2/nocw_inputs.py', ROOT/'scripts/studies/run_qgeognn_v2_row_llm_nocw_five.py'):
        files[rel(p)]={'sha256':digest(p),'bytes':p.stat().st_size,'evidence':'validator_source'}
    snap={'schema_version':1,'captured_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'cohort':'all 180 complete Free batches; all actual 71 Hybrid batches; local NoCW batches frozen at capture','batches':batches,'files':files,'excluded_unfrozen_rounds':active,'truth_table_reads':0,'real_llm_calls':0,'new_training':0,'selection_resolver':'Free: original report_free_lcmd owner(seed, fit_round=round+1); Hybrid: OLD revision boundaries by fit round; inherited copies are counted once.'}
    dump(SNAPSHOT.name,snap)
    return snap


def frozen_validator():
    """Execute exact pure validator AST, without importing predictor/RDKit/runners."""
    ns={'copy':copy,'json':json,'re':re,'MAX_HYPOTHESES':8,'stable_hash':stable_hash}
    p=ROOT/'src/qgeognn_al/active_learning_v2/scientist_full_delivery.py'
    tree=ast.parse(p.read_text()); selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'decode_table','validate_direct'}]
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(p),'exec'),ns)
    p=ROOT/'src/qgeognn_al/active_learning_v2/nocw_inputs.py'; tree=ast.parse(p.read_text())
    wanted={'CW_PATTERN','HYPOTHESIS_FIELDS','PLAN_FIELDS'}
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'_only','audit_no_cw','validate_plan'} or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in wanted for t in n.targets)]
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(p),'exec'),ns)
    # Actual five-seed NoCW runner deliberately supersedes the paired wrapper.
    ns['views']=SimpleNamespace(**{k:ns[k] for k in ('_only','PLAN_FIELDS','HYPOTHESIS_FIELDS','audit_no_cw')})
    p=ROOT/'scripts/studies/run_qgeognn_v2_row_llm_nocw_five.py'
    tree=ast.parse(p.read_text()); fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_plan')
    fn.body=[n for n in fn.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns)
    ns['actual_nocw_validator_source_sha256']=digest(p)
    return ns


def parse_answer(receipt):
    try: return json.loads(receipt['answer']),''
    except (json.JSONDecodeError,KeyError,TypeError) as e: return None,type(e).__name__+': '+str(e)


def strict_answer_issue(receipt):
    def unique_pairs(pairs):
        result={}
        for key,value in pairs:
            if key in result: raise ValueError('duplicate JSON key: '+key)
            result[key]=value
        return result
    def finite_constant(value): raise ValueError('nonfinite JSON constant: '+value)
    def finite_float(value):
        result=float(value)
        if not math.isfinite(result): raise ValueError('nonfinite JSON numeric overflow: '+value)
        return result
    try: json.loads(receipt['answer'],object_pairs_hook=unique_pairs,parse_constant=finite_constant,parse_float=finite_float)
    except (ValueError,TypeError,KeyError) as e: return str(e)
    return ''


def reliable_choices(obj):
    # No regex ID extraction, no salvage of wrong structures, no code-fence strip.
    if not isinstance(obj,dict) or not isinstance(obj.get('choices'),list): return None
    if not all(isinstance(x,dict) and isinstance(x.get('id'),str) and isinstance(x.get('reason'),str) and 'hypothesis_id' in x for x in obj['choices']): return None
    return obj['choices']


def classify(error):
    t=error.lower()
    if any(x in t for x in ('jsondecode','expecting','invalid control character','unterminated','extra data')): return 'syntactic','json_parse'
    if 'wrong selection count' in t: return 'syntactic','count_or_choice_structure'
    if 'legal supplied candidate' in t: return 'constraint','candidate_identity'
    if 'unobserved evidence' in t: return 'constraint','unobserved_evidence'
    if 'packet binding' in t: return 'constraint','packet_binding'
    if 'duplicate hypothesis' in t: return 'constraint','duplicate_hypothesis_id'
    if 'hypothesis reference' in t: return 'constraint','hypothesis_reference'
    if 'row ids in prose' in t: return 'constraint','untyped_prose_evidence'
    if 'nocw visible' in t: return 'constraint','forbidden_cw_content'
    if 'memory size bound' in t: return 'constraint','plan_size_bound'
    return 'syntactic','schema_or_required_field'


def details(obj,cand,obs,n):
    choices=reliable_choices(obj); out={'candidate_choices_analyzable':choices is not None,'raw_choices_list_length':len(obj['choices']) if isinstance(obj,dict) and isinstance(obj.get('choices'),list) else None}
    raw=obj.get('choices') if isinstance(obj,dict) else None
    identity_scan=isinstance(raw,list) and all(isinstance(x,dict) and isinstance(x.get('id'),str) for x in raw)
    out['raw_identity_reference_scan_available']=identity_scan
    out['identity_scan_scope']='raw choice-object ID fields; this is not a recovered experimental batch when schema is invalid'
    if not identity_scan: return out
    ids=[x['id'] for x in raw]; counts=collections.Counter(ids)
    out.update({'choice_count':len(ids) if choices is not None else None,'raw_id_object_count':len(ids),'duplicate_ids':[i for i,c in counts.items() if c>1], 'duplicate_occurrences':sum(c-1 for c in counts.values()),'outside_candidate_ids':list(dict.fromkeys(i for i in ids if i not in cand)), 'already_observed_ids':list(dict.fromkeys(i for i in ids if i in obs)), 'unknown_ids':list(dict.fromkeys(i for i in ids if i not in cand|obs)), 'legal_unique_ids':list(dict.fromkeys(i for i in ids if i in cand)) if choices is not None else None, 'raw_legal_candidate_id_references':list(dict.fromkeys(i for i in ids if i in cand)), 'missing_count':max(0,n-len(ids)) if choices is not None else None, 'excess_count':max(0,len(ids)-n) if choices is not None else None})
    return out


def transport_type(record):
    t=' '.join(str(record.get(k,'')) for k in ('error_type','message','error_message','http_status','cause_type')).lower()
    if 'timeout' in t or 'timed out' in t: return 'request_timeout'
    if any(x in t for x in ('ratelimit','rate limit','rate_limit','server_error','apierror','httpstatuserror')) or isinstance(record.get('http_status'),int) and record['http_status']>=400 or re.search(r'(?:error code|status|http|code[=:]).{0,12}\b(?:401|403|429|500|502|503|504)\b',t): return 'api_or_http_failure'
    if any(x in t for x in ('stream','remoteprotocol','server disconnected','incomplete chunk')): return 'stream_or_protocol_interruption'
    if any(x in t for x in ('connect','network','ssl','dns')): return 'connection_failure'
    return 'transport_other_or_parser_recovery'


def behavior(ids,meta,observed):
    rows=[meta[i] for i in dict.fromkeys(ids) if i in meta]
    n=len(rows)
    if not n: return {}
    count=collections.Counter(x['smiles'] for x in rows)
    pairs=sum(c*(c-1)//2 for c in count.values())
    prior_by=collections.defaultdict(list)
    for x in observed: prior_by[x['smiles']].append(x)
    def key(x): return tuple(x['conditions'][a] for a in AXES)
    exact=0; strict=0; pe=0; tuple_count=collections.Counter((x['smiles'],key(x)) for x in rows)
    for x in rows:
        prior=prior_by[x['smiles']]; exact+=any(key(x)==key(p) for p in prior)
        other=prior+[p for p in rows if p['id']!=x['id'] and p['smiles']==x['smiles']]
        diffs=[tuple(a for a in AXES if x['conditions'][a]!=p['conditions'][a]) for p in other]
        strict+=any(len(v)==1 and v[0] in AXES[1:] for v in diffs) and not any(key(x)==key(p) for p in prior)
        pe+=any(v==(AXES[0],) for v in diffs) and not any(key(x)==key(p) for p in prior)
    out={'legal_metadata_n':n,'unique_molecules':len(count),'max_molecule_count':max(count.values()),'molecule_hhi':sum((v/n)**2 for v in count.values()),'same_molecule_pair_fraction':pairs/(n*(n-1)/2) if n>1 else 0,'exact_prior_tuple_repeat_fraction':exact/n,'strict_loading_contrast_L_or_batch_fraction':strict/n,'pe_contrast_L_or_batch_fraction':pe/n,'exact_batch_tuple_pairs':sum(c*(c-1)//2 for c in tuple_count.values())}
    for a in AXES: out['unique_condition_'+a]=len({str(x['conditions'][a]) for x in rows})
    for field in ('cw_distance_percentile','max_similarity_to_observed','condition_novelty'):
        v=[x[field] for x in rows if isinstance(x.get(field),(float,int)) and not isinstance(x[field],bool)]
        out['mean_'+field]=statistics.mean(v) if len(v)==n else None
    return out


def scalar_usage(receipt,name):
    v=(receipt.get('provenance') or {}).get('usage') or {}
    v=v.get(name)
    return v if isinstance(v,(float,int)) and not isinstance(v,bool) else None

def csv_write(name,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in rows: w.writerow({k:json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def audit(snap):
    for name,record in snap['files'].items():
        if digest(ROOT/name)!=record['sha256']: raise ValueError(f'snapshot source changed: {name}')
    validators=frozen_validator(); metrics=[]; failures=[]; repairs=[]; manifest=[]; response_metrics=[]; interventions=[]; correction_causes=[]; rejection_replay_checks=[]
    for b in snap['batches']:
        d=ROOT/b['directory']; frozen=load(d/'batch_freeze.json'); packet=load(d/'input.json'); catalog=load(d/'catalog.json'); contract=load(d/'contract.json'); final=load(d/'selection.json')
        cand={x['id']:x for x in catalog['candidates']}; obs={x['id']:x for x in catalog['observed']}; pending={x['id']:x for x in catalog['pending']}; meta=cand|obs|pending
        cat=SimpleNamespace(candidates=cand,observed=obs,pending=pending,references=meta)
        n=packet['selection_count']; variant='no_cw' if b['method']=='llm32_nocw' else None
        validate=(lambda obj:validators['validate_plan'](obj,cat,packet,'no_cw')) if variant else (lambda obj:validators['validate_direct'](obj,cat,packet))
        validate(final)
        final_ids=[x['id'] for x in final['choices']]
        assert len(frozen['batch_ids'])==len(set(frozen['batch_ids']))==32
        assert frozen['batch_ids']==list(pending)+final_ids or set(frozen['batch_ids'])==set(pending)|set(final_ids)
        assert set(frozen['batch_ids']) <= set(contract['U_t_ids']) and not set(frozen['batch_ids']) & set(contract['L_t_ids'])
        base={k:b[k] for k in ('method','seed','round')}; base['method_group']='NoCW' if variant else 'Hybrid' if b['method']==HYBRID else 'Free'
        turns=[]; transport=[]
        for name in b['source_files']:
            p=ROOT/name
            if re.fullmatch(r'turn_\d+\.json',p.name):
                if p.name not in frozen['artifact_hashes']: raise ValueError('unfrozen turn included')
                receipt=load(p); obj,parse_error=parse_answer(receipt); error=parse_error
                if not parse_error:
                    try: validate(obj)
                    except (ValueError,TypeError,KeyError) as e: error=str(e)
                strict_issue=strict_answer_issue(receipt)
                ids_detail=details(None if strict_issue else obj,set(cand),set(obs),n)
                turn=int(p.stem.split('_')[1]); served=receipt.get('provenance',{}).get('served_model')
                item={**base,'attempt':turn,'response_path':name,'packet_path':rel(d/'input.json'),'catalog_path':rel(d/'catalog.json'),'contract_path':rel(d/'contract.json'),'pool_path':rel(d/'input.json'),'observed_ids_path':rel(d/'contract.json'),'n_pick':n,'variant':variant,'validator_kind':'nocw' if variant else 'direct','response_format':'receipt.answer JSON string','accepted':obj==final,'original_validator_legal':not bool(error),'batch_freeze_path':rel(d/'batch_freeze.json'),'frozen_sha256':b['batch_freeze_sha256'],'response_sha256':snap['files'][name]['sha256'],'completion':'frozen_artifact','served_model':served,'validation_error':error,'strict_parse_issue':strict_issue,'ids_detail':ids_detail}
                manifest.append(item); turns.append((turn,receipt,obj,error,item))
                response_metrics.append({**base,'attempt':turn,'served_model':served,'legal':not bool(error),'elapsed_seconds':receipt.get('elapsed_seconds'),'input_tokens':scalar_usage(receipt,'input_tokens'),'output_tokens':scalar_usage(receipt,'output_tokens'),'total_tokens':scalar_usage(receipt,'total_tokens'),'cost':None,'response_path':name})
                if error:
                    category,subtype=classify(error)
                    failure={**base,'event_id':f"{base['method_group']}:{b['seed']}:{b['round']}:{turn}:llm",'event_kind':'llm_validation_failure','attempt':turn,'transport_attempt':None,'failure_class':category,'primary_failure_type':subtype,'error':error,'served_model':served,'complete_response':True,'frozen_response':True,'source_path':name,'evidence':'batch_freeze_artifact','elapsed_seconds':receipt.get('elapsed_seconds'),'input_tokens':scalar_usage(receipt,'input_tokens'),'output_tokens':scalar_usage(receipt,'output_tokens'),'cost':None,**ids_detail}
                    rejection_paths=[d/f'rejected_turn_{turn:02d}.json',d/f'rejected_{turn:02d}.json']
                    saved_rejections=[q for q in rejection_paths if rel(q) in snap['files']]
                    if saved_rejections:
                        saved_error=load(saved_rejections[0]).get('error')
                        failure['historical_rejection_path']=rel(saved_rejections[0]); failure['historical_error_matches_replay']=saved_error==error[:1000]
                        rejection_replay_checks.append({'response_path':name,'rejection_path':rel(saved_rejections[0]),'matches':saved_error==error[:1000],'historical_error':saved_error,'replay_error':error})
                    failures.append(failure)
            elif re.fullmatch(r'attempt_\d+_\d+\.json',p.name) or re.fullmatch(r'attempt_\d+\.json',p.name) and 'retries/' in name:
                t=load(p)
                # NoCW successes duplicate turn_XX; never count a second model call.
                if t.get('success') is True: continue
                if '.started.' in p.name: continue
                match=re.search(r'turn_(\d+)',name)
                turn=int(match.group(1)) if match else int(p.stem.split('_')[1])
                attempt=int(p.stem.split('_')[-1]); transport.append(t)
                failures.append({**base,'event_id':f"{base['method_group']}:{b['seed']}:{b['round']}:{turn}:transport:{attempt}",'event_kind':'transport_failure_record','attempt':turn,'transport_attempt':attempt,'failure_class':'transport','primary_failure_type':transport_type(t),'error':' '.join(str(t.get(k,'')) for k in ('error_type','message','error_message')),'served_model':None,'complete_response':False,'frozen_response':False,'source_path':name,'evidence':snap['files'][name]['evidence'],'elapsed_seconds':t.get('elapsed_seconds'),'input_tokens':None,'output_tokens':None,'cost':None,'candidate_choices_analyzable':False})
        turns.sort(key=lambda x:x[0]); assert turns
        for previous,current in zip(turns,turns[1:]):
            category,subtype=classify(previous[3])
            correction_causes.append({**base,'trigger_turn':previous[0],'correction_turn':current[0],'trigger_response_path':previous[4]['response_path'],'correction_response_path':current[4]['response_path'],'trigger_failure_class':category,'trigger_primary_failure_type':subtype,'trigger_error':previous[3],'cause_group':'format_or_schema' if category=='syntactic' else 'candidate_identity' if subtype=='candidate_identity' else 'evidence_or_other_constraints','next_served_model':current[1].get('provenance',{}).get('served_model'),'next_input_tokens':scalar_usage(current[1],'input_tokens'),'next_output_tokens':scalar_usage(current[1],'output_tokens'),'next_elapsed_seconds':current[1].get('elapsed_seconds'),'next_cost':None})
        accepted=[x for x in turns if x[2]==final]; rejected=[x for x in turns if x[3]]
        correction_calls=max(0,len(turns)-1); final_raw_legal=bool(accepted)
        models=list(dict.fromkeys(x[1].get('provenance',{}).get('served_model') or 'unknown' for x in turns))
        times=[x[1].get('elapsed_seconds') for x in turns]+[x.get('elapsed_seconds') for x in transport]
        metric={**base,'labels_before':len(contract['L_t_ids']),'candidate_count':len(cand),'pool_U_count':len(contract['U_t_ids']),'input_packet_bytes':snap['files'][rel(d/'input.json')]['bytes'],'llm_selection_count':n,'executed_batch_count':32,'served_models':';'.join(models),'first_served_model':turns[0][1].get('provenance',{}).get('served_model'),'final_served_model':accepted[-1][1].get('provenance',{}).get('served_model') if accepted else None,'completed_llm_responses':len(turns),'first_output_legal':not bool(turns[0][3]),'final_raw_output_matches_frozen_selection':final_raw_legal,'final_frozen_batch_legal':True,'llm_correction_requests':correction_calls,'extra_requests_due_format_or_candidate_identity':sum(classify(x[3])[0]=='syntactic' or classify(x[3])[1]=='candidate_identity' for x in turns[:-1] if x[3]),'extra_requests_due_evidence_or_other_constraints':sum(classify(x[3])[0]=='constraint' and classify(x[3])[1]!='candidate_identity' for x in turns[:-1] if x[3]),'invalid_complete_outputs':len(rejected),'syntactic_failures':sum(classify(x[3])[0]=='syntactic' for x in rejected),'constraint_failures':sum(classify(x[3])[0]=='constraint' for x in rejected),'transport_failure_records':len(transport),'observed_provider_request_records':len(turns)+len(transport),'provider_request_count_interpretation':'completed responses + recorded transport failures; legacy recovery records may not represent distinct billed requests','accepted_raw_response_count':len(accepted),'historical_authorized_transcription_repair':not final_raw_legal,'elapsed_seconds_complete':all(isinstance(x,(int,float)) for x in times),'elapsed_seconds_total':sum(times) if all(isinstance(x,(int,float)) for x in times) else None,'elapsed_seconds_observed_subtotal':sum(x for x in times if isinstance(x,(int,float))) if any(isinstance(x,(int,float)) for x in times) else None,'elapsed_seconds_records_available':sum(isinstance(x,(int,float)) for x in times),'cost':None,'batch_freeze_path':rel(d/'batch_freeze.json')}
        for field in ('input_tokens','output_tokens','total_tokens'):
            vals=[scalar_usage(x[1],field) for x in turns]
            metric[field+'_complete_responses_subtotal']=sum(x for x in vals if x is not None)
            metric[field+'_available_response_count']=sum(x is not None for x in vals)
            metric[field+'_all_provider_requests']=sum(vals) if not transport and all(x is not None for x in vals) else None
            metric[field+'_extra_correction_subtotal']=sum(x for x in vals[1:] if x is not None)
        metrics.append(metric)
        if correction_calls or not final_raw_legal:
            first=turns[0]; choices=None if first[4]['strict_parse_issue'] else reliable_choices(first[2]); result={**base,'first_response_path':first[4]['response_path'],'final_selection_path':rel(d/'selection.json'),'first_response_legal':not bool(first[3]),'completed_llm_responses':len(turns),'correction_requests':correction_calls,'final_raw_output_matches_selection':final_raw_legal,'comparison_analyzable':choices is not None,'nonanalyzable_reason':None if choices is not None else 'First complete answer is not strict parseable JSON with a list of string-ID choice objects; no IDs recovered from prose.','scientific_intent_change':'not inferred; ID/reason/metadata comparisons are descriptive only','geometric_redundancy_change':None,'geometric_redundancy_note':'not recomputed here; no trajectory bank was read by this audit'}
            if choices is not None:
                initial_ids=[x['id'] for x in choices]; initial_legal=list(dict.fromkeys(i for i in initial_ids if i in cand)); first_set=set(initial_ids); final_set=set(final_ids)
                retained=[i for i in dict.fromkeys(initial_ids) if i in final_set]
                result.update({'initial_choice_count':len(initial_ids),'initial_unique_legal_candidate_count':len(initial_legal),'initial_valid_ids':initial_legal,'retained_ids':retained,'retained_legal_count':len(set(initial_legal)&final_set),'deleted_or_replaced_ids':[i for i in dict.fromkeys(initial_ids) if i not in final_set],'added_final_ids':[i for i in final_ids if i not in first_set],'selection_jaccard':len(first_set&final_set)/len(first_set|final_set) if first_set|final_set else 1.,'all_legal_initial_choices_replaced':bool(initial_legal) and not bool(set(initial_legal)&final_set),'full_plan_response_regenerated':True,'regenerated_batch_selection_changed':first_set!=final_set,'reason_exactly_preserved_retained_count':sum(next(x['reason'] for x in final['choices'] if x['id']==i)==next(x.get('reason') for x in choices if x['id']==i) for i in retained),'initial_legal_behavior':behavior(initial_legal,meta,catalog['observed']),'final_behavior':behavior(final_ids,meta,catalog['observed']),'initial_metadata_denominator':len(initial_legal),'final_metadata_denominator':len(final_ids),'metadata_comparison_caution':'first proposal uses only unique legal candidates; unequal subsets/counts can change concentration fractions independent of scientific intent'})
                for stage,ids in [('first_legal',initial_legal),('final',final_ids)]:
                    for k,v in behavior(ids,meta,catalog['observed']).items(): result[f'{stage}_{k}']=v
            repairs.append(result)
        if correction_calls>=3 or not final_raw_legal or b['method']=='llm32_nocw' and (b['seed'],b['round']) in {(887,3),(6101,8),(2357,10),(6101,10)}:
            interventions.append({**base,'complete_responses':len(turns),'response_turn_indices':[x[0] for x in turns],'final_batch_frozen':True,'final_raw_accepted':final_raw_legal,'calls_beyond_original_three':max(0,len(turns)-3),'manual_transcription_repair':not final_raw_legal,'authorization_sidecars':[x for x in b['source_files'] if 'authorized' in x or 'extra_correction' in x]})
    csv_write('failure_events.csv',failures); csv_write('reliability_metrics.csv',metrics); csv_write('repair_trajectory_analysis.csv',repairs); csv_write('completed_response_costs.csv',response_metrics); csv_write('correction_request_causes.csv',correction_causes)
    with (HERE/'response_manifest.jsonl').open('w') as f:
        for x in manifest: f.write(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n')
    groups=[]
    for group in ('Free','Hybrid','NoCW'):
        rows=[x for x in metrics if x['method_group']==group]; event=[x for x in failures if x['method_group']==group]
        groups.append({'method_group':group,'frozen_batches':len(rows),'per_seed_batches':dict(collections.Counter(str(x['seed']) for x in rows)),'completed_llm_responses':sum(x['completed_llm_responses'] for x in rows),'first_output_legal_numerator':sum(x['first_output_legal'] for x in rows),'first_output_legal_denominator':len(rows),'first_output_legal_rate':sum(x['first_output_legal'] for x in rows)/len(rows),'frozen_batch_legal_numerator':sum(x['final_frozen_batch_legal'] for x in rows),'frozen_batch_legal_denominator':len(rows),'final_raw_legal_numerator':sum(x['final_raw_output_matches_frozen_selection'] for x in rows),'rounds_needing_correction':sum(x['llm_correction_requests']>0 for x in rows),'llm_correction_requests':sum(x['llm_correction_requests'] for x in rows),'mean_corrections_per_frozen_batch':statistics.mean(x['llm_correction_requests'] for x in rows),'invalid_complete_outputs':sum(x['invalid_complete_outputs'] for x in rows),'transport_failure_records':sum(x['transport_failure_records'] for x in rows),'failure_class_record_counts':dict(collections.Counter(x['failure_class'] for x in event)),'primary_failure_type_counts':dict(collections.Counter(x['primary_failure_type'] for x in event)),'cost':None})
    diagnostic_counts=collections.Counter()
    for e in failures:
        if e['event_kind']=='llm_validation_failure':
            for key in ('duplicate_ids','already_observed_ids','unknown_ids','outside_candidate_ids'):
                if e.get(key): diagnostic_counts[key]+=1
    summary={'snapshot_captured_at':snap['captured_at'],'source_file_count':len(snap['files']),'batches':len(metrics),'complete_responses':len(manifest),'failure_event_rows':len(failures),'corrected_rounds':len(repairs),'repair_comparison_analyzable_rounds':sum(x['comparison_analyzable'] for x in repairs),'groups':groups,'candidate_error_diagnostic_response_counts_overlapping':dict(diagnostic_counts),'interventions':interventions,'excluded_unfrozen_rounds':snap['excluded_unfrozen_rounds'],'served_model_complete_response_counts':dict(collections.Counter(x['served_model'] or 'unknown' for x in manifest)),'tokens_cost_note':'Provider usage for complete responses only; transport failures may have unknown usage. No token price or billed cost estimated. Blank costs are missing, not zero.','selection_legality_note':'Conditional on completed frozen cohort, so final legality is a freeze audit, not eventual success probability for ongoing/never-frozen requests. One historical authorized transcription repair is distinct from raw LLM success.','primary_taxonomy_note':'primary_failure_type is the first error raised by original validator (AST-exact replay); overlapping duplicate/out-of-pool/measured-ID details are separate diagnostic fields. Scientific behavior is not an execution failure.','transport_note':'NoCW attempt success mirrors turn and is counted once. V2 retries can include CLI recovery/parsing records, so recorded request failures are not asserted to be distinct billed model requests.','actual_nocw_validator_source_sha256':validators['actual_nocw_validator_source_sha256'],'actual_nocw_validator_source':'scripts/studies/run_qgeognn_v2_row_llm_nocw_five.py:74 (paired prose-ID prohibition is not applied)','real_llm_calls':0,'new_training':0,'truth_table_reads':0}
    summary['correction_request_cause_counts']=dict(collections.Counter(x['cause_group'] for x in correction_causes))
    summary['correction_request_primary_type_counts']=dict(collections.Counter(x['trigger_primary_failure_type'] for x in correction_causes))
    summary['correction_request_cost_by_cause']=[{'cause_group':group,'extra_completed_requests':len(rr),'input_tokens_subtotal':sum(x['next_input_tokens'] for x in rr if x['next_input_tokens'] is not None),'output_tokens_subtotal':sum(x['next_output_tokens'] for x in rr if x['next_output_tokens'] is not None),'usage_available_requests':sum(x['next_input_tokens'] is not None and x['next_output_tokens'] is not None for x in rr),'cost':None} for group in ('format_or_schema','candidate_identity','evidence_or_other_constraints') for rr in [[x for x in correction_causes if x['cause_group']==group]]]
    summary['historical_rejection_replay']={'saved_rejection_count':len(rejection_replay_checks),'matching_first_error_count':sum(x['matches'] for x in rejection_replay_checks),'mismatches':[x for x in rejection_replay_checks if not x['matches']]}
    summary['strict_parse_issue_response_count']=sum(bool(x['strict_parse_issue']) for x in manifest)
    summary['complete_responses_with_analyzable_choices']=sum(x['ids_detail']['candidate_choices_analyzable'] for x in manifest)
    summary['extra_requests_due_format_or_candidate_identity']=sum(x['extra_requests_due_format_or_candidate_identity'] for x in metrics)
    summary['extra_requests_due_evidence_or_other_constraints']=sum(x['extra_requests_due_evidence_or_other_constraints'] for x in metrics)
    summary['cost_availability_by_method']=[]
    summary['per_model_complete_response_reliability']=[]
    for group in ('Free','Hybrid','NoCW'):
        rr=[x for x in response_metrics if x['method_group']==group]
        ff=[x for x in failures if x['method_group']==group and x['event_kind']=='transport_failure_record']
        summary['cost_availability_by_method'].append({'method_group':group,'completed_responses':len(rr),'input_tokens_available_responses':sum(x['input_tokens'] is not None for x in rr),'output_tokens_available_responses':sum(x['output_tokens'] is not None for x in rr),'input_tokens_complete_response_subtotal':sum(x['input_tokens'] for x in rr if x['input_tokens'] is not None),'output_tokens_complete_response_subtotal':sum(x['output_tokens'] for x in rr if x['output_tokens'] is not None),'elapsed_seconds_available_complete_responses':sum(x['elapsed_seconds'] is not None for x in rr),'elapsed_seconds_available_transport_failures':sum(x.get('elapsed_seconds') is not None for x in ff),'elapsed_seconds_completed_responses_subtotal':sum(x['elapsed_seconds'] for x in rr if isinstance(x['elapsed_seconds'],(float,int))) if any(isinstance(x['elapsed_seconds'],(float,int)) for x in rr) else None,'elapsed_seconds_transport_failure_subtotal':sum(x['elapsed_seconds'] for x in ff if isinstance(x.get('elapsed_seconds'),(float,int))) if any(isinstance(x.get('elapsed_seconds'),(float,int)) for x in ff) else None,'cost':None})
        for model in sorted({x['served_model'] or 'unknown' for x in rr}):
            sub=[x for x in rr if (x['served_model'] or 'unknown')==model]; first=[x for x in sub if x['attempt']==0]
            summary['per_model_complete_response_reliability'].append({'method_group':group,'served_model':model,'complete_response_legal_numerator':sum(x['legal'] for x in sub),'complete_response_denominator':len(sub),'first_output_legal_numerator':sum(x['legal'] for x in first),'first_output_denominator':len(first),'note':'Versions are confounded with seeds, budgets and history; no causal model comparison.'})
    scale=[]
    for group in ('Free','Hybrid','NoCW'):
        for lo,hi in ((333,525),(525,813),(813,1005),(1005,1486)):
            rr=[x for x in metrics if x['method_group']==group and lo<=x['labels_before']<hi]
            if rr: scale.append({'method_group':group,'labels_before_band':f'{lo}-{hi-1}','frozen_batch_denominator':len(rr),'first_output_failures':sum(not x['first_output_legal'] for x in rr),'first_output_failure_fraction':sum(not x['first_output_legal'] for x in rr)/len(rr),'mean_labels_before':statistics.mean(x['labels_before'] for x in rr),'mean_candidate_count':statistics.mean(x['candidate_count'] for x in rr),'mean_input_packet_bytes':statistics.mean(x['input_packet_bytes'] for x in rr),'mean_llm_correction_requests':statistics.mean(x['llm_correction_requests'] for x in rr),'transport_failure_records':sum(x['transport_failure_records'] for x in rr),'note':'Descriptive only: L grows while U shrinks, packet/history changes and model version/budget confound these trends.'})
    csv_write('input_scale_reliability.csv',scale)
    summary['input_scale_descriptive_bands']=scale
    summary['repair_behavior_summary']=[]
    for group in ('Free','Hybrid','NoCW'):
        rr=[x for x in repairs if x['method_group']==group and x['comparison_analyzable']]
        if not rr: continue
        item={'method_group':group,'analyzable_corrected_rounds':len(rr),'mean_initial_legal_candidate_count':statistics.mean(x['initial_unique_legal_candidate_count'] for x in rr),'mean_retained_legal_count':statistics.mean(x['retained_legal_count'] for x in rr),'mean_selection_jaccard':statistics.mean(x['selection_jaccard'] for x in rr),'zero_legal_candidate_overlap_rounds':sum(x['all_legal_initial_choices_replaced'] for x in rr),'mean_retained_reason_exact_fraction':statistics.mean(x['reason_exactly_preserved_retained_count']/len(x['retained_ids']) for x in rr if x['retained_ids']),'caution':'All first-proposal behavior uses unique legal subset; n can differ from final. Does not establish intent preservation or quality.'}
        for feature in ('molecule_hhi','same_molecule_pair_fraction','strict_loading_contrast_L_or_batch_fraction','pe_contrast_L_or_batch_fraction'):
            available=[x for x in rr if x.get('first_legal_'+feature) is not None and x.get('final_'+feature) is not None]
            item[feature+'_increase_rounds']=sum(x['final_'+feature]>x['first_legal_'+feature]+1e-12 for x in available)
            item[feature+'_decrease_rounds']=sum(x['final_'+feature]<x['first_legal_'+feature]-1e-12 for x in available)
            item[feature+'_unchanged_rounds']=len(available)-item[feature+'_increase_rounds']-item[feature+'_decrease_rounds']
        summary['repair_behavior_summary'].append(item)
    summary['candidate_error_diagnostic_strict_choice_schema_response_counts_overlapping']={key:sum(bool(x['ids_detail'].get(key)) and x['ids_detail']['candidate_choices_analyzable'] for x in manifest) for key in ('duplicate_ids','already_observed_ids','unknown_ids','outside_candidate_ids')}
    summary['candidate_error_raw_scan_note']='Raw ID-object scans do not recover malformed scientific plans; hypothesis-shaped entries in choices may be among raw out-of-pool references. Strict-schema counts are separately given.'
    summary['duplicate_id_responses_in_frozen_complete_history']=sum(bool(x['ids_detail'].get('duplicate_ids')) for x in manifest)
    summary['duplicate_id_interpretation']='Zero actual duplicates in all 462 complete frozen answers; generic historical validator messages conflate duplicate and illegal IDs. This does not rule out duplicates in excluded/unavailable history.'
    dump('history_audit_summary.json',summary)
    dump('history_integrity_verification.json',{'all_snapshot_source_hashes_match':True,'verified_source_files':len(snap['files']),'unique_batch_keys':len({(x['method'],x['seed'],x['round']) for x in metrics}),'batches':len(metrics),'every_frozen_batch_32_unique_legal_ids':True,'every_final_selection_passes_current_frozen_validator':True,'historical_raw_final_response_mismatch_batches':[x for x in interventions if x['manual_transcription_repair']],'no_source_mutations':True})
    notes=['# 历史可靠性审计说明',f"固定本地截止：{snap['captured_at']}。仅分析 capture 时 batch_freeze 已存在且 SHA 通过的批次。",'','| 方法 | 冻结批次 | 完整LLM回答 | 首次合法 | 需纠错轮 | 纠错请求 | 非法完整回答 | 传输失败记录 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for x in groups: notes.append(f"| {x['method_group']} | {x['frozen_batches']} | {x['completed_llm_responses']} | {x['first_output_legal_numerator']}/{x['first_output_legal_denominator']} ({x['first_output_legal_rate']:.2%}) | {x['rounds_needing_correction']} | {x['llm_correction_requests']} | {x['invalid_complete_outputs']} | {x['transport_failure_records']} |")
    notes+=['','分母与限制：Free每seed36批（333→1485）；Hybrid157有36批、6101有35批（333→1485/1453），完整权威冻结尾端均保留；NoCW按本地快照冻结批次数逐seed见JSON。冻结合法率只说明已经冻结的批次满足执行约束，不能推断进行中/未冻结轮的最终成功率。','', 'Free157 round05历史授权ID抄写修正使selection通过但无原始turn与selection完全一致；必须与LLM最终原始合法率分开。原始无唯一匹配的修正不在本次重做。','', 'V2 transport retries未必纳入旧batch_freeze，已逐文件锁定为completed_snapshot_sidecar；若为CLI recovery/parser失败不保证产生一个新的付费请求。NoCW成功attempt与turn为同一次回答，绝不重复计数；started/stream中间内容不作为答案。','', 'token以已保存provenance.usage为实测子计；失败网络请求可能缺token，费用均留空。旧V2普遍无耗时，NoCW可用elapsed_seconds；不估算计费或墙钟成本。','', '纠错前后只对严格JSON且choices为字符串ID对象列表的完整回答比较。合法候选子集与最终计划的分子/条件指标保留各自分母，不自动判定科学意图改变或科学质量优劣；几何冗余未在该脚本重算。','', '首错分类来自原始冻结validator纯函数AST的离线执行；候选错误细分扫描raw choices中明确的ID字段，允许同一回答同时有重复、已测和未知ID；schema不合法时该扫描不恢复实验批次。NoCW依照实际five-seed runner覆盖版：字段白名单和CW内容约束；未运行paired协议的额外prose ID禁令不适用于该历史。','', '本次新增LLM调用、模型训练、未观测真值表读取、test读取均为0。现有observed/catalog里的已观测信息仅用于合法证据和条件关系；未读取预测真值文件。','', '复现：`python studies/active_learning/verifiable_scientific_agent_20261030/audit_history.py`。首次快照：`--snapshot`，已有manifest禁止静默刷新。']
    notes += ['', '补充核验：462份完整回答全部为原json.loads可接受JSON，原validator首错153/153与历史rejected.error一致。严格JSON歧义检查（重复键、NaN/Infinity、浮点溢出）无历史触发。只有460份满足可可靠恢复choice实验对象的schema；另外2份仍保留原始字段扫描，不恢复实验批次。', '', '实际重复ID为0/462；历史\"distinct legal IDs\"报错是合并错误信息，不构成发现重复的证据。原validator首错：candidate_identity114、unobserved_evidence28、hypothesis_reference4、packet_binding2、count_or_choice_structure5。Raw字段扫描117份含池外引用、63份含已测ID、66份含未知引用，分类可重叠；严格choice-schema子集对应116/63/65，全部重复0。', '', '152个追加纠错请求已在correction_request_causes.csv逐条绑定上一失败回答：113个由candidate identity、5个由count/schema、34个由证据/其他约束触发。对应完整回答额外输入tokens合计70,029,763，输出1,381,906；费用缺失。网络失败token未知，不包含在上述完整回答子计。', '', '四个NoCW授权例外887r3、6101r8、2357r10、6101r10均在第4份完整回答后冻结，实际各用1个超出原三答案上限的回答，未用到第五/第六份。', '', '2357r10：turn00有32个结构完整choices、30个唯一合法候选、2个未知ID，无重复；turn01仍有1个未知ID；turn02的choices数组37元素混入5个hypothesis形状对象，不可作为确定实验批次，也不自动摘取前32；turn03合法接受32。其最终32与最初30个合法ID无交集，说明完整重新输出可能显著改变选择，不能由此认定科学质量变差。', '', '114个纠错轮的首次完整提案均可按严格choice结构分析；Free/Hybrid/NoCW分别12/1/8轮初始合法候选与最终无交集。分子集中和loading对照在部分轮增加、更多轮减少，均为不同合法子集分母的描述；不证明科学意图保持或损失。', '', '按输入规模的描述分箱见input_scale_reliability.csv；候选U随L增长而下降，输入packet字节在约1.7MB的窄区间，版本/预算/历史共变，不能把错误因果归于长上下文或比较移除CW的因果效果。', '', 'NoCW完成回答99与传输失败330均有elapsed，记录请求耗时合计132,159.710684秒=36.711小时，59个已冻结批次均摊37.333分钟；是跨并发worker累加的请求时间，不含backoff/人工等待，不是墙钟历时。Free/Hybrid elapsed全部缺失，留NA。']
    (HERE/'HISTORY_AUDIT_NOTES.md').write_text('\n'.join(notes)+'\n')
    print(json.dumps({'batches':len(metrics),'complete_responses':len(manifest),'groups':groups,'corrected_rounds':len(repairs),'failure_events':len(failures)},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',action='store_true');a=p.parse_args()
    snap=capture_snapshot() if a.snapshot else load(SNAPSHOT)
    audit(snap)
