from pathlib import Path
import ast,collections,hashlib,json
ROOT=Path('/Users/fish/Documents/GitHub/column_chromatography_prediction');S=ROOT/'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101';H=ROOT/'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions';F=S/'frozen_source';seeds=[157,887,2357,6101,12203]
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
stable=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
sysmod=ast.parse((F/'src/qgeognn_al/active_learning_v2/scientist_selector.py').read_text());sysval=next(ast.literal_eval(n.value) for n in sysmod.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SYSTEM_PROMPT' for t in n.targets))
fullmod=ast.parse((F/'src/qgeognn_al/active_learning_v2/scientist_full_delivery.py').read_text());nodes=[n for n in fullmod.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DIRECT_PROMPT' for t in n.targets)) or (isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='DIRECT_PROMPT')];ns={'SYSTEM_PROMPT':sysval};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<prompt-constant>','exec'),ns);oldprompt=ns['DIRECT_PROMPT']
result={'no_api_calls':True,'no_training':True,'no_unselected_labels_or_test_truth_read':True,'source_pins':[],'round0_inputs':[],'rejection_errors':[],'fit_pairs':[],'source_notes':[]}
for rev in ['astra_high_full_pool_20261001','sol_xhigh_hybrid_free_to1005_20261013','sol_xhigh_all_strategies_to1165_20261014','sol_xhigh_all_strategies_to1293_20261015','sol_xhigh_all_strategies_to1421_20261017','sol_xhigh_all_strategies_to1549_20261018','sol61_xhigh_free_five_seeds_to1005_20261023','sol61_xhigh_free_five_seeds_to1485_20261024']:
 x=read(H/rev/'protocol.json');matches=[];diff=[]
 for n,h in x['code_hashes'].items():
  if (F/n).exists() and sha(F/n)==h:matches.append(n)
  else:diff.append(n)
 result['source_pins'].append({'revision':rev,'matches':len(matches),'differences':diff,'historical_prompt_hash_verified':stable(oldprompt)==x.get('prompt_sha256')})
for seed in seeds:
 rev='astra_high_full_pool_20261001' if seed in (157,6101) else 'sol61_xhigh_free_five_seeds_to1005_20261023';op=H/rev/f'selections/seed_{seed}/free_llm32_scientist_v2/round_00';np=S/f'selections/seed_{seed}/free_cw_sol61_xhigh_rerun32/round_00';a=read(op/'input.json');b=read(np/'input.json');req=read(np/'request_00.json');om=[{'role':'system','content':oldprompt},{'role':'user','content':json.dumps(a,ensure_ascii=False,separators=(',',':'),allow_nan=False)}]
 result['round0_inputs'].append({'seed':seed,'json_semantics_and_packet_hash_equal':a==b,'historical_user_string_reconstruction_hash_matches_receipt':stable({'messages':om,'config':read(H/rev/'protocol.json')['transport']})==read(op/'turn_00.json')['request_sha256'],'actual_request_user_semantics_equal':json.loads(req['messages'][1]['content'])==a,'actual_user_string_equal':req['messages'][1]['content']==om[1]['content'],'actual_system_string_equal':req['messages'][0]['content']==oldprompt,'system_difference_only_44_character_pending_sentence':req['messages'][0]['content']==oldprompt.replace('Pending-CW outcomes are identified separately.\n',''),'old_system_characters':len(oldprompt),'new_system_characters':len(req['messages'][0]['content']),'old_user_characters':len(om[1]['content']),'new_user_characters':len(req['messages'][1]['content'])})
rejs=collections.Counter();whitelist=[];usages=[];selectedusages=[]
for d in (S/'selections').glob('seed_*/*/round_*'):
 selected=read(d/'selection.json') if (d/'selection.json').exists() else None
 for p in d.glob('rejected_*.json'):
  x=read(p);err=x.get('error','');cat=('unobserved_evidence' if 'cites unobserved evidence' in err else err);rejs[cat]+=1
  if 'unauthorized' in err:whitelist.append(str(p.relative_to(ROOT)))
 for p in d.glob('turn_*.json'):
  x=read(p);u=x.get('provenance',{}).get('usage') or {};ot=u.get('output_tokens')
  if ot is not None:
   item={'output_tokens':ot,'reasoning_tokens':(u.get('output_tokens_details') or {}).get('reasoning_tokens'),'path':str(p.relative_to(ROOT))};usages.append(item)
   try:equal=json.loads(x.get('answer',''))==selected
   except ValueError:equal=False
   if selected is not None and equal:selectedusages.append(item)
result['rejection_errors']=dict(rejs);result['whitelist_rejections']=whitelist;result['completed_answer_usage']={'count':len(usages),'max':max(usages,key=lambda x:x['output_tokens']),'at_or_above_32768':sum(x['output_tokens']>=32768 for x in usages)};result['accepted_answer_usage']={'count':len(selectedusages),'max':max(selectedusages,key=lambda x:x['output_tokens']),'at_or_above_32768':sum(x['output_tokens']>=32768 for x in selectedusages)}
# Check metadata only; no raw prediction CSV, label source, or checkpoint truth tensor is read.
import torch
OLD=[(21,'sol_xhigh_hybrid_free_to1005_20261013'),(26,'sol_xhigh_all_strategies_to1165_20261014'),(30,'sol_xhigh_all_strategies_to1293_20261015'),(34,'sol_xhigh_all_strategies_to1421_20261017'),(36,'sol_xhigh_all_strategies_to1549_20261018')];NEW=[(21,'sol61_xhigh_free_five_seeds_to1005_20261023'),(36,'sol61_xhigh_free_five_seeds_to1485_20261024')]
for seed in seeds:
 for marker in sorted((S/'progress').glob(f'completed_seed_{seed}_round_*.json')):
  r=int(marker.stem.split('_')[-1])+1;rev=next(n for end,n in (OLD if seed in (157,6101) else NEW) if r<=end);nd=S/f'runtime/seed_{seed}/free_cw_sol61_xhigh_rerun32/round_{r:02d}/model';od=H/rev/f'runtime/seed_{seed}/free_llm32_scientist_v2/round_{r:02d}/model';na=read(nd/'fit_audit.json');oa=read(od/'fit_audit.json');nc=torch.load(nd/'best.pt',map_location='cpu',weights_only=False);oc=torch.load(od/'best.pt',map_location='cpu',weights_only=False)
  fields=['initialization_seed','initialization_hash','validation_ids_hash','validation_rows','prediction_rows','model_variant','parameter_count','test_graph_label_state','test_labels_used_for_fit_or_checkpoint_selection'];md=['schema_version','model_variant','input_schema','input_schema_hash','output_order','normalization','preprocessing','training_config','parameter_count'];diff=[k for k in fields if na.get(k)!=oa.get(k)];mdiff=[k for k in md if nc.get(k)!=oc.get(k)]
  result['fit_pairs'].append({'seed':seed,'round':r,'budget':333+32*r,'historical_owner':rev,'same_budget':na['train_rows']==oa['train_rows']==333+32*r,'audit_protocol_differences':diff,'checkpoint_metadata_differences':mdiff,'new_checkpoint_bound':sha(nd/'best.pt')==na['checkpoint_sha256'],'old_checkpoint_bound':sha(od/'best.pt')==oa['checkpoint_sha256'],'identical_training_id_hash':na['train_ids_hash']==oa['train_ids_hash'],'identical_checkpoint_bytes':na['checkpoint_sha256']==oa['checkpoint_sha256'],'target_scales':nc['preprocessing']['target_scales']})
result['fit_summary']={'pairs':len(result['fit_pairs']),'bad_budget':sum(not x['same_budget'] for x in result['fit_pairs']),'audit_protocol_differences':sum(bool(x['audit_protocol_differences']) for x in result['fit_pairs']),'checkpoint_metadata_differences':sum(bool(x['checkpoint_metadata_differences']) for x in result['fit_pairs']),'broken_bindings':sum(not x['new_checkpoint_bound'] or not x['old_checkpoint_bound'] for x in result['fit_pairs']),'identical_training_id_hash':[(x['seed'],x['round']) for x in result['fit_pairs'] if x['identical_training_id_hash']]}
Path('/tmp/free_code_protocol_audit_20261103.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['fit_pairs']},ensure_ascii=False,indent=2))
