#!/usr/bin/env python3
"""NoCW scientific-chain audit. Only frozen packets and accepted feedback; no runners."""
from __future__ import annotations
import ast,copy,collections,csv,hashlib,itertools,json,math,re
from pathlib import Path
from types import SimpleNamespace
from decimal import Decimal
from fractions import Fraction
import numpy as np
import pandas as pd
from scipy.stats import rankdata

OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[2]
OLD=ROOT/'studies/active_learning/comprehensive_history_diagnostics_20261112'
SEEDS=(157,887,2357,6101,12203);MANIFEST=[]
CONDITIONS=['PE/EA','Density g/ml','V/ul','loading solvent','Volume of loading solvent/ul']
ID=re.compile(r'(?<![0-9a-f])[0-9a-f]{20}(?![0-9a-f])')
TOPICS={'eluent':r'PE/EA|eluent|mobile.phase|solvent.strength|ratio',
 'sample_loading':r'load|sample.volume|sample.amount|injection',
 'structure_transfer':r'similar|transfer|analog|scaffold|structur|isomer|functional.group',
 'model_failure':r'predict|model|quantile|uncertaint|bias|error',
 'nonlinearity':r'non.monoton|nonlinear|threshold|transition|anomal|inversion'}
REV=re.compile(r'correct|revis|retract|weaken|contradict|not support|no longer|misattribut|mistaken|overgeneral|not universal',re.I)
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p,role,expected=None):
 p=Path(p);p=p if p.is_absolute() else ROOT/p
 assert 'qualification_dataset_manifest' not in str(p) and '/dataset/' not in str(p)
 before=p.stat();b=p.read_bytes();after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 h=sha(b)
 if expected:assert h==expected,(p,'hash drift')
 MANIFEST.append({'path':str(p.relative_to(ROOT)),'sha256':h,'bytes':len(b),'role':role})
 return b
def js(p,role,expected=None):return json.loads(read(p,role,expected))
def frame(p):
 import io
 return pd.read_csv(io.BytesIO(read(p,'previously audited derived data')),float_precision='round_trip')
def table(name,rows):
 d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
 for c in d.columns:
  if d[c].map(lambda x:isinstance(x,(list,dict))).any():d[c]=d[c].map(lambda x:json.dumps(x,ensure_ascii=False) if isinstance(x,(list,dict)) else x)
 d.to_csv(OUT/'tables'/name,index=False,float_format='%.17g');return d
def dump(name,data):(OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def phase(r):return '333-525' if r<6 else '525-813' if r<15 else '813-1005' if r<21 else 'post1005'
def decoded(packet,name):
 out={}
 for vals in packet[name]['rows']:
  d=dict(zip(packet[name]['columns'],vals));d['conditions']=dict(zip(packet['condition_columns'],d['conditions']))
  d['smiles']=packet['molecules'][d['molecule_id']]['smiles'];out[d['id']]=d
 return out
def condition_key(d):
 c=d['conditions'];r=Fraction(c['PE/EA']);return (r,*[Decimal(str(c[k])) if k!='loading solvent' else c[k] for k in CONDITIONS[1:]])
def same_axes(a,b):
 if a['smiles']!=b['smiles']:return None
 return [k for k,x,y in zip(CONDITIONS,condition_key(a),condition_key(b)) if x!=y]
def tok(s):return set(re.findall(r'[a-z0-9]+',s.lower()))
def similarity(a,b):
 a,b=tok(a),tok(b);return len(a&b)/len(a|b) if a|b else 1.
def ranks(v):return rankdata(np.round(np.asarray(v,dtype=float),12))
def corr(x,y):
 if len(x)<6 or len(set(x))<2:return np.nan
 a,b=ranks(x),ranks(y)
 if np.std(a)==0 or np.std(b)==0:return np.nan
 return float(np.corrcoef(a,b)[0,1])
def partial(g,col,before=True):
 if len(g)<12 or g[col].nunique()<2:return np.nan
 cov=[np.ones(len(g)),ranks(g.labeled_size_before)]
 if before:cov.append(ranks(g.nrmse_before))
 cov.append(pd.get_dummies(g.phase,drop_first=True,dtype=float).to_numpy());c=np.column_stack(cov)
 x=ranks(g[col]);y=ranks(g.gain);x-=c@np.linalg.lstsq(c,x,rcond=None)[0];y-=c@np.linalg.lstsq(c,y,rcond=None)[0]
 return float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-10 and np.std(y)>1e-10 else np.nan
def validator():
 ns={'copy':copy,'json':json,'re':re,'MAX_HYPOTHESES':8,
     'stable_hash':lambda v:sha(json.dumps(v,sort_keys=True,separators=(',',':')).encode())}
 p=ROOT/'src/qgeognn_al/active_learning_v2/scientist_full_delivery.py';tree=ast.parse(read(p,'pure validator source').decode())
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('decode_table','validate_direct')]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
 p=ROOT/'src/qgeognn_al/active_learning_v2/nocw_inputs.py';tree=ast.parse(read(p,'pure validator schema source').decode())
 names={'CW_PATTERN','PLAN_FIELDS','HYPOTHESIS_FIELDS'}
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_only','audit_no_cw') or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
 ns['views']=SimpleNamespace(**{k:ns[k] for k in ['PLAN_FIELDS','HYPOTHESIS_FIELDS','_only','audit_no_cw']})
 p=ROOT/'scripts/studies/run_qgeognn_v2_row_llm_nocw_five.py';tree=ast.parse(read(p,'actual historical NoCW wrapper validator').decode())
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_plan');fn.body=[n for n in fn.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns)
 return ns['validate_plan']
def errclass(s):
 for a,b in [('legal supplied','candidate_identity'),('unobserved evidence','unobserved_evidence'),('hypothesis reference','hypothesis_reference'),('packet binding','packet_binding'),('NoCW visible','forbidden_cw_content'),('selection count','selection_shape')]:
  if a in s:return b
 return 'schema_or_json'
def main():
 plan=js(OUT/'verification/nocw_analysis_plan.json','prospective descriptive case rule')
 prior=OUT/'verification/nocw_inputs.json'
 if prior.exists():
  for e in json.loads(prior.read_text())['files']:
   p=ROOT/e['path'];assert sha(p.read_bytes())==e['sha256'],('frozen input drift',p)
 gain=frame(OLD/'tables/batch_gains.csv');behavior=frame(OLD/'tables/behavior_batches.csv');geo=frame(OLD/'tables/behavior_geometry.csv')
 original=behavior[behavior.method=='nocw'].sort_values(['seed','round']);primary=original[original['round']<21]
 assert len(primary)==105 and len(original)==123
 v=validator();bs=[];hs=[];es=[];ps=[];rs=[];rep=[];flags=[];samples=[];chains={};selected_science=[];lineages=[]
 for seed,g in original.groupby('seed'):
  previous=None;acquired={};previous_chosen=set()
  for rec in g.to_dict('records'):
   r=int(rec['round']);directory=(ROOT/rec['source']).parent
   fr=js(directory/'batch_freeze.json','accepted frozen batch',rec['source_sha256']);ct=js(directory/'contract.json','preselection L/U contract',fr['artifact_hashes']['contract.json'])
   sel=js(directory/'selection.json','accepted scientific selection',fr['artifact_hashes']['selection.json'])
   packet=js(directory/'input.json','hash-bound preselection packet',fr['artifact_hashes']['input.json']);cat=js(directory/'catalog.json','hash-bound preselection catalog',fr['artifact_hashes']['catalog.json'])
   feedback=js(directory/'feedback.json','accepted selected feedback')
   assert feedback['batch_freeze_sha256']==sha((directory/'batch_freeze.json').read_bytes())
   assert ct['input_sha256']==sha((directory/'input.json').read_bytes()) and ct['catalog_sha256']==sha((directory/'catalog.json').read_bytes())
   assert packet['validation_test_records']==0
   obs,cand=decoded(packet,'observed'),decoded(packet,'candidates');L,U=set(ct['L_t_ids']),set(ct['U_t_ids']);chosen={c['id'] for c in sel['choices']}
   assert set(obs)==L and set(cand)==U and len(L)==333+32*r
   assert chosen<=U and not chosen&L and len(chosen)==32
   assert not set(packet['candidates']['columns'])&{'V1_ml','V2_ml','true_V1_ml','true_V2_ml'}
   for memory_name in ['recent_outcomes','high_error_measured_examples']:
    for record in packet['memory'][memory_name]:
     assert record['candidate_id'] in L,'future row in scientific memory'
     for t in ['V1','V2']:
      assert math.isclose(record[f'true_{t}_ml'],obs[record['candidate_id']][f'{t}_ml'],abs_tol=1e-8),'memory/observed response mismatch'
   assert chosen==set(fr['batch_ids'])=={x['candidate_id'] for x in feedback['records']}
   fb={x['candidate_id']:x for x in feedback['records']}
   for ident in chosen:
    assert all(math.isclose(cand[ident][f'pred_{t}_ml'],fb[ident][f'pred_{t}_ml'],abs_tol=1e-9) for t in ['V1','V2'])
   cobj=SimpleNamespace(candidates={x['id']:x for x in cat['candidates']},observed={x['id']:x for x in cat['observed']});v(sel,cobj,packet,'no_cw')
   if r==0:acquired.update({i:-1 for i in L})
   assert set(acquired)==L
   if previous:
    assert packet['memory']['previous_hypotheses']==previous['selection']['hypotheses']
    assert {x['candidate_id'] for x in packet['memory']['recent_outcomes']}==previous_chosen
   common={'method':'nocw','seed':int(seed),'round':r,'labeled_size_before':333+32*r,'phase':phase(r),'primary':r<21,
      'selection_source':str((directory/'selection.json').relative_to(ROOT))}
   support=[];contra=[];hrows=[];linked_pairs=[]
   for hi,h in enumerate(sel['hypotheses']):
    ev=h['supporting_observed_ids']+h['contradicting_observed_ids'];support+=h['supporting_observed_ids'];contra+=h['contradicting_observed_ids']
    assert set(ev)<=L
    linked=[c['id'] for c in sel['choices'] if c.get('hypothesis_id')==h['id']]
    within=[];cross=[]
    for a,b in itertools.combinations(linked,2):
     axes=same_axes(cand[a],cand[b])
     if axes and len(axes)==1:within.append((a,b,axes[0]))
    for a in linked:
     for b in dict.fromkeys(ev):
      axes=same_axes(cand[a],obs[b])
      if axes and len(axes)==1:cross.append((a,b,axes[0]))
    for a,b,axis in within+cross:
     ar=cand[a];br=cand[b] if b in chosen else obs[b];aa=fb[a];bb=fb[b] if b in chosen else br
     pair={'pair_kind':'same_batch' if b in chosen else 'vs_cited_preselection_observed','id_a':a,'id_b':b,'axis':axis,
       'molecule_key':sha(ar['smiles'].encode())[:16],'molecule_id':ar['molecule_id'],'conditions_a':ar['conditions'],'conditions_b':br['conditions']}
     for t in ['V1','V2']:
      y1=aa[f'true_{t}_ml'];y2=bb[f'true_{t}_ml'] if b in chosen else bb[f'{t}_ml']
      pair[f'actual_delta_{t}_ml']=y2-y1
      pair[f'pred_delta_{t}_ml']=(cand[b][f'pred_{t}_ml']-ar[f'pred_{t}_ml']) if b in chosen else None
     ps.append({**common,'hypothesis_id':h['id'],**pair});linked_pairs.append(pair)
    text=h['content']+' '+h['next_discriminating_question'];scope=re.findall(r'\bm\d+\b',text)
    best=max(previous['selection']['hypotheses'],key=lambda z:similarity(z['content'],h['content'])) if previous and previous['selection']['hypotheses'] else None
    sim=similarity(best['content'],h['content']) if best else None
    repeated=bool(previous and any(z['content']==h['content'] for z in previous['selection']['hypotheses']))
    row={**common,'hypothesis_index':hi,'hypothesis_id':h['id'],'content':h['content'],'question':h['next_discriminating_question'],
      'basis':h['basis'],'status':h['status'],'confidence':h['confidence'],'linked_choices_n':len(linked),
      'support_n':len(h['supporting_observed_ids']),'contradiction_n':len(h['contradicting_observed_ids']),
      'explicit_molecule_tokens_n':len(set(scope)),'linked_molecules_n':len({cand[i]['smiles'] for i in linked}),
      'within_linked_strict_single_pairs':len(within),'vs_cited_evidence_strict_single_pairs':len(cross),
      'has_real_condition_test_structure':bool(within or cross),'exact_text_repeat':repeated,
      'previous_candidate_similarity':sim,'lexical_revision_not_scientific_truth':bool(REV.search(text)),
      'topic_tags':[k for k,p in TOPICS.items() if re.search(p,text,re.I)],
      'interpretation':'condition-test graph is structural evidence, not semantic entailment or causal experiment'}
    hs.append(row);hrows.append(row)
    if best and sim>=.55:
     lineages.append({**common,'previous_hypothesis_id':best['id'],'current_hypothesis_id':h['id'],'text_jaccard':sim,
      'exact_repeat':best['content']==h['content'],'same_evidence_sets':set(best['supporting_observed_ids'])==set(h['supporting_observed_ids']) and set(best['contradicting_observed_ids'])==set(h['contradicting_observed_ids']),
      'new_evidence_n':len(set(ev)&previous_chosen),'previous_status':best['status'],'current_status':h['status'],
      'previous_confidence':best['confidence'],'current_confidence':h['confidence']})
    for kind,ids in [('support',h['supporting_observed_ids']),('contradiction',h['contradicting_observed_ids'])]:
     for ident in ids:es.append({**common,'hypothesis_id':h['id'],'kind':kind,'id':ident,'molecule_id':obs[ident]['molecule_id'],
       'molecule_key':sha(obs[ident]['smiles'].encode())[:16],'preselection_legal':True,'acquired_round':acquired[ident],
       'previous_batch':ident in previous_chosen})
   mismatch=0;proseunknown=0
   for c in sel['choices']:
    molecules=re.findall(r'\bm\d+\b',c['reason'])
    if molecules and molecules[0]!=cand[c['id']]['molecule_id']:
     mismatch+=1;flags.append({**common,'flag':'first_molecule_token_mismatch','id':c['id'],'actual_molecule':cand[c['id']]['molecule_id'],
       'claimed_first_molecule':molecules[0],'text':c['reason'],'not_auto_error':True})
   fields=[('batch_rationale',sel['batch_rationale']),('feedback_interpretation',sel['feedback_interpretation'])]+[(f'choice:{c["id"]}',c['reason']) for c in sel['choices']]+[(f'hypothesis:{h["id"]}',h['content']+' '+h['next_discriminating_question']) for h in sel['hypotheses']]
   for field,text in fields:
    for ident in ID.findall(text):
     if ident not in L|U:
      proseunknown+=1;flags.append({**common,'flag':'unknown_prose_id','id':ident,'field':field,'text':text,'not_auto_error':True})
   turns=[]
   for name in sorted(fr['artifact_hashes']):
    if not re.fullmatch(r'turn_\d+\.json',name):continue
    receipt=js(directory/name,'frozen complete model response',fr['artifact_hashes'][name]);obj=None;error=''
    try:obj=json.loads(receipt['answer']);v(obj,cobj,packet,'no_cw')
    except (ValueError,KeyError,TypeError) as e:error=str(e)
    choices=obj.get('choices') if isinstance(obj,dict) else None;reliable=isinstance(choices,list) and all(isinstance(c,dict) and isinstance(c.get('id'),str) for c in choices)
    ids={c['id'] for c in choices} if reliable else set();legal=ids&U
    hypot=obj.get('hypotheses',[]) if isinstance(obj,dict) else [];hypot=hypot if isinstance(hypot,list) else []
    n_ev=sum(len(h.get(k,[])) for h in hypot if isinstance(h,dict) for k in ['supporting_observed_ids','contradicting_observed_ids'] if isinstance(h.get(k),list))
    out={**common,'turn_file':name,'original_validator_legal':not error,'first_error':errclass(error) if error else '',
      'reliable_choice_structure':reliable,'legal_unique_ids':len(legal),'hypothesis_n':len(hypot),'evidence_n':n_ev,
      'matches_final':obj==sel,'served_model':(receipt.get('provenance') or {}).get('model','')}
    rs.append(out);turns.append((out,obj,ids,legal))
   assert turns and any(z[0]['matches_final'] for z in turns)
   if len(turns)>1:
    first=turns[0];rep.append({**common,'responses':len(turns),'first_legal_ids':len(first[3]),'first_legal_retained':len(first[3]&chosen),
      'all_legal_replaced':bool(first[3]) and not bool(first[3]&chosen),'set_changed':first[2]!=chosen,
      'first_evidence_n':first[0]['evidence_n'],'final_evidence_n':len(support)+len(contra),
      'first_hypothesis_n':first[0]['hypothesis_n'],'final_hypothesis_n':len(sel['hypotheses'])})
   cited=set(support+contra);count=len(support)+len(contra);mols=collections.Counter(obs[i]['molecule_id'] for i in support+contra)
   common.update({'hypothesis_n':len(hrows),'condition_test_hypothesis_n':sum(h['has_real_condition_test_structure'] for h in hrows),
      'condition_test_fraction':sum(h['has_real_condition_test_structure'] for h in hrows)/len(hrows) if hrows else np.nan,
      'without_linked_test_n':sum(not h['has_real_condition_test_structure'] for h in hrows),'linked_choice_fraction':sum(c.get('hypothesis_id') is not None for c in sel['choices'])/32,
      'support_reference_n':len(support),'contradiction_reference_n':len(contra),'contradiction_hypothesis_n':sum(h['contradiction_n']>0 for h in hrows),
      'exact_repeat_n':sum(h['exact_text_repeat'] for h in hrows),'evidence_hhi':sum((v/count)**2 for v in mols.values()) if count else np.nan,
      'previous_batch_reference_fraction':sum(i in previous_chosen for i in support+contra)/count if count else np.nan,
      'previous_batch_evidence_n':len(cited&previous_chosen),'feedback_revision_keyword':bool(REV.search(sel['feedback_interpretation'])),
      'first_molecule_mismatch_flag_n':mismatch,'unknown_prose_reference_n':proseunknown,
      'complete_response_n':len(turns),'first_complete_legal':turns[0][0]['original_validator_legal'],
      'within_linked_strict_pairs':sum(h['within_linked_strict_single_pairs'] for h in hrows),
      'cited_evidence_strict_pairs':sum(h['vs_cited_evidence_strict_single_pairs'] for h in hrows)})
   bs.append(common);chains[int(seed),r]={'selection':sel,'packet':packet,'observed':obs,'candidates':cand,'feedback':fb,'hypotheses':hrows,'pairs':linked_pairs}
   previous=chains[int(seed),r];previous_chosen=chosen;acquired.update({i:r for i in chosen})
 # Add explicit next-step evidence-use graph, never equating ID reuse with semantic acceptance.
 for h in hs:
  seed,r=h['seed'],h['round'];next_=chains.get((seed,r+1));curr=chains[seed,r];hid=h['hypothesis_id']
  associated={c['id'] for c in curr['selection']['choices'] if c.get('hypothesis_id')==hid}
  if next_:
   nh=next_['selection']['hypotheses'];h['next_cites_linked_selected_n']=sum(len(set(z['supporting_observed_ids']+z['contradicting_observed_ids'])&associated) for z in nh)
   h['next_has_counterevidence_for_linked_n']=sum(bool(set(z['contradicting_observed_ids'])&associated) for z in nh)
   h['next_candidate_text_similarity']=max([similarity(h['content'],z['content']) for z in nh],default=0)
   h['next_available']=True
  else:h.update(next_cites_linked_selected_n=None,next_has_counterevidence_for_linked_n=None,next_candidate_text_similarity=None,next_available=False)
 bdf=pd.DataFrame(bs);g=gain[gain.method=='nocw'];bdf=bdf.merge(g.drop(columns=['phase']),on=['method','seed','round','labeled_size_before'],validate='one_to_one')
 assert len(bdf)==123 and int(bdf.primary.sum())==105
 table('nocw_batches.csv',bdf);table('nocw_hypotheses.csv',hs);table('nocw_evidence.csv',es);table('nocw_test_pairs.csv',ps);table('nocw_responses.csv',rs);table('nocw_repairs.csv',rep);table('nocw_text_flags.csv',flags);table('nocw_candidate_lineages.csv',lineages)
 cases=[]
 for (seed,ph),z in bdf[bdf.primary].groupby(['seed','phase']):
  for category,col,absolute,ascending in [('high_gain','gain',False,False),('minimum_gain_case','gain',False,True),('near_zero','gain',True,True)]:
   z=z.copy();z['sort_value']=z[col].abs() if absolute else z[col];row=z.sort_values(['sort_value','round'],ascending=[ascending,True]).iloc[0]
   d={k:row[k] for k in ['method','seed','round','labeled_size_before','phase','gain','nrmse_before','nrmse_after','hypothesis_n','condition_test_fraction','first_molecule_mismatch_flag_n','complete_response_n']};d['case_category']=category
   d['actual_gain_class']='worsened' if row['gain']<-.001 else 'near_zero' if abs(row['gain'])<=.001 else 'improved'
   traj=g[(g.seed==seed)].set_index('round')
   for horizon in [2,3]:
    needed=list(range(int(row['round']),int(row['round'])+horizon));available=all(x in traj.index for x in needed)
    d[f'gain_{horizon}_batch_cumulative']=sum(traj.loc[j,'gain'] for j in needed) if available else None
    d[f'gain_{horizon}_right_censored']=not available
    future=list(range(int(row['round'])+1,int(row['round'])+horizon+1));available_future=all(x in traj.index for x in future)
    d[f'next_{horizon}_batches_gain_only']=sum(traj.loc[j,'gain'] for j in future) if available_future else None
    d[f'current_plus_next_{horizon}_gain']=float(row['gain'])+sum(traj.loc[j,'gain'] for j in future) if available_future else None
    d[f'next_{horizon}_right_censored']=not available_future
   cases.append(d)
 cdf=table('nocw_gain_cases.csv',cases)
 # Explanatory sample records are bounded scientific excerpts, never full private packets.
 target={(int(c['seed']),int(c['round'])) for c in cases}|{(s,r) for s in SEEDS for r in [0,5,14,20]}
 for seed,r in sorted(target):
  chain=chains[seed,r];sel=chain['selection'];first=sel['hypotheses'][0] if sel['hypotheses'] else None
  linked={c['id'] for c in sel['choices'] if first and c.get('hypothesis_id')==first['id']}
  pairs=[p for p in ps if p['seed']==seed and p['round']==r and first and p['hypothesis_id']==first['id']]
  nxt=chains.get((seed,r+1));next_h=nxt['selection']['hypotheses'] if nxt else []
  samples.append({'seed':seed,'round':r,'phase':phase(r),'sample_rule':'gain case plus fixed r0/r5/r14/r20; first hypothesis only',
   'gain':float(bdf[(bdf.seed==seed)&(bdf['round']==r)].gain.iloc[0]),'first_hypothesis':first,
   'linked_choice_reasons':[c for c in sel['choices'] if c['id'] in linked],
   'first_metadata_sorted_test_pair':sorted(pairs,key=lambda p:(p['axis'],p['id_a'],p['id_b']))[0] if pairs else None,
   'next_hypotheses_citing_linked':[z for z in next_h if set(z['supporting_observed_ids']+z['contradicting_observed_ids'])&linked],
   'next_feedback_interpretation':nxt['selection']['feedback_interpretation'] if nxt else None,
   'feedback_interpretation':sel['feedback_interpretation']})
 dump('tables/nocw_fixed_case_excerpts.json',samples)
 features=['condition_test_fraction','without_linked_test_n','contradiction_hypothesis_n','exact_repeat_n','evidence_hhi','previous_batch_reference_fraction','first_molecule_mismatch_flag_n','complete_response_n']
 correlations=[]
 for seed,z in bdf[bdf.primary].groupby('seed'):
  for f in features:
   zf=z.dropna(subset=[f]);correlations.append({'seed':seed,'feature':f,'n':len(zf),'spearman':corr(zf[f],zf.gain),
    'partial_budget_phase_before':partial(zf,f),'partial_budget_phase_only':partial(zf,f,False)})
 table('nocw_science_gain_correlations.csv',correlations)
 # Merge selected-case behavior/geometry without recomputing old diagnostics.
 behavior_cols=['method','seed','round','unique_molecules','new_molecule_slot_fraction','max_molecule_slot_fraction','within_any_single_selected_fraction','cross_only_sample_volume_selected_fraction','within_exact_selected_fraction']
 cp=cdf.merge(behavior[behavior.method=='nocw'][behavior_cols],on=['method','seed','round'],validate='many_to_one')
 raw=geo[(geo.method=='nocw')&(geo.space=='raw_gradient')][['method','seed','round','reference_coverage_relative_drop_random_percentile','batch_nn_distance_mean_random_percentile','selected_to_l_distance_mean_random_percentile']]
 cp=cp.merge(raw,on=['method','seed','round'],validate='many_to_one');table('nocw_gain_cases_with_behavior.csv',cp)
 # Sanity checks include actual validator failures, not implementation-mirroring fixtures.
 one=chains[157,0];sel=copy.deepcopy(one['selection']);pkt=one['packet'];cobj=SimpleNamespace(candidates=one['candidates'],observed=one['observed'])
 sel['hypotheses'][0]['supporting_observed_ids']=[next(iter(one['candidates']))]
 try:v(sel,cobj,pkt,'no_cw')
 except ValueError as e:assert 'unobserved' in str(e)
 else:raise AssertionError('future candidate used as evidence')
 assert len({(x['seed'],x['round']) for x in bs if x['primary']})==105
 assert all(e['acquired_round']<e['round'] for e in es)
 assert all(len(c['selection']['choices'])==32 for c in chains.values())
 summary={'primary_batches':105,'extended_batches':len(bs)-105,'primary_hypotheses':sum(h['primary'] for h in hs),'all_hypotheses':len(hs),
  'primary_references':sum(e['primary'] for e in es),'references_outside_preselection_L':0,
  'primary_complete_responses':sum(r['primary'] for r in rs),'primary_first_complete_legal':sum(x['first_complete_legal'] for x in bs if x['primary']),
  'primary_condition_test_hypotheses':sum(h['primary'] and h['has_real_condition_test_structure'] for h in hs),
  'primary_hypotheses_without_linked_test':sum(h['primary'] and not h['has_real_condition_test_structure'] for h in hs),
  'primary_exact_repeat':sum(h['primary'] and h['exact_text_repeat'] for h in hs),
  'primary_molecule_flags':sum(x['first_molecule_mismatch_flag_n'] for x in bs if x['primary']),
  'primary_unknown_prose_ids':sum(x['unknown_prose_reference_n'] for x in bs if x['primary']),
  'primary_pair_events':sum(p['primary'] for p in ps),'case_rows':len(cases),'unique_case_batches':len(target),
  'sources_unique':len({e['path'] for e in MANIFEST}),'new_training':0,'new_paid_llm':0,'test_truth_reads':0,
  'limits':'105 batch histories, five correlated seeds. Structural pairs and lexical flags do not prove falsifiability, correctness or learning efficacy.'}
 dump('verification/nocw_summary.json',summary);dump('verification/nocw_inputs.json',{'files':list({e['path']:e for e in MANIFEST}.values()),'raw_private_packets_exported':False,'label_boundary':'only same-trajectory preselection observed and accepted feedback'})
 dump('verification/nocw_checks.json',{'status':'PASS','L_U_exact':True,'all_accepted_ids_frozen':True,'local_artifact_hashes_checked':True,
  'selected_prediction_feedback_aligned':True,'chronological_evidence':True,'raw_candidate_truth_columns_absent':True,'validator_candidate_evidence_negative_control':True,
  'actual_no_cw_wrapper_rules_ast_only':True,'complete_primary_105':True,'scientific_memory_records_in_L_and_truth_matches_observed':True,'case_rule_frozen_before_gain_loading':True,'new_llm_calls':0,'new_training_runs':0,'test_files_opened':0})
 print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
