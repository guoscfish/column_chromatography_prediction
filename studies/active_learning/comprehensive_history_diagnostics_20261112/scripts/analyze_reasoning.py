#!/usr/bin/env python3
"""Offline scientific-memory audit of 180 frozen CW-Free batches.

Reads archived accepted artifacts plus only their hash-bound input/catalog/turn
files. Never imports runners, label stores, training or model clients. No raw
packets, raw model answers, or observed truth tables are copied to outputs.
"""
from __future__ import annotations
import ast, collections, copy, csv, gzip, hashlib, json, math, re, statistics
from pathlib import Path
from types import SimpleNamespace

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[2]
STUDY = ROOT / 'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101'
ARCHIVE = ROOT / 'reports/free_cw_final_20261112/scientific_artifacts'
SEEDS = (157,887,2357,6101,12203)
METHOD = 'new_free_cw_sol61_xhigh'
ID = re.compile(r'(?<![0-9a-f])[0-9a-f]{20}(?![0-9a-f])')
HEX_NEAR_ID = re.compile(r'(?<![0-9a-f])[0-9a-f]{18,22}(?![0-9a-f])')
CATEGORIES = {
 'PE_EA_retention': r'ratio|PE/EA|eluent|mobile.phase|solvent.strength',
 'loading_effect': r'load|sample.volume|injection|solvent.volume',
 'V1_V2_asymmetry': r'asymmetr|selectiv.{0,20}(V1|V2)|onset.{0,45}finish|V1.{0,60}V2|finish.{0,45}onset|broadening',
 'structure_condition_interaction': r'structure.specific|family.specific|structure.condition|interact|functional.group|molecular.structure',
 'structure_similarity_transfer': r'similar|transfer|neighbo|analog|related.{0,30}(molecul|famil|structur)',
 'nonlinearity_threshold_anomaly': r'nonlinear|threshold|abrupt|non.monoton|outlier|anomal|transition|discontin',
 'model_error_uncertainty': r'quantile|uncertaint|width|calibrat|prediction.error|model.{0,30}(error|bias|miss)|underpredict|overpredict|underestimat|overestimat',
 'ordinary_vs_tail': r'tail|extrem|ordinary|early.retention|low.retention|common.{0,20}(row|region|regime)',
}
REVISIONS = re.compile(r'correct|revis|retract|weaken|contradict|not support|no longer|rather than|misattribut|mistaken|overgeneral',re.I)

def sha(raw): return hashlib.sha256(raw).hexdigest()
def relative(p): return str(p.relative_to(ROOT))
def phase(r): return '333–525' if r<6 else '525–813' if r<15 else '813–1005' if r<21 else '1005–1485'
def dump(name,data):
 (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def csv_write(name,rows):
 if not rows: return
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with (OUT/'tables'/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
  for row in rows: w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
def decode(packet,name): return {r[0]:dict(zip(packet[name]['columns'],r)) for r in packet[name]['rows']}
def tokens(text): return set(re.findall(r'[a-z0-9]+',text.lower()))
def similarity(a,b):
 a,b=tokens(a),tokens(b)
 return len(a&b)/len(a|b) if a|b else 1
def histogram_similarity(a,b):
 return sum(min(a[k]/sum(a.values()),b[k]/sum(b.values())) for k in a.keys()|b.keys()) if a and b else None

def pure_validator(manifest):
 ns={'copy':copy,'MAX_HYPOTHESES':8}
 paths=[STUDY/'frozen_source/src/qgeognn_al/active_learning_v2/scientist_full_delivery.py',
        STUDY/'frozen_source/src/qgeognn_al/active_learning_v2/nocw_inputs.py',
        STUDY/'frozen_source/scripts/studies/run_free_cw_sol61_five_20261101.py']
 for p in paths: manifest.append({'path':relative(p),'sha256':sha(p.read_bytes()),'role':'pure_validator_source'})
 tree=ast.parse(paths[0].read_text())
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('decode_table','validate_direct')]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(paths[0]),'exec'),ns)
 tree=ast.parse(paths[1].read_text()); wanted={'PLAN_FIELDS','HYPOTHESIS_FIELDS'}
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_only' or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in wanted for t in n.targets)]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(paths[1]),'exec'),ns)
 ns['views']=SimpleNamespace(**{k:ns[k] for k in ('_only','PLAN_FIELDS','HYPOTHESIS_FIELDS')})
 tree=ast.parse(paths[2].read_text()); fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_plan')
 fn.body=[n for n in fn.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(paths[2]),'exec'),ns)
 return ns['validate_plan']

def error_class(msg):
 for pattern,category in [('legal supplied candidate','candidate_identity'),('unobserved evidence','unobserved_evidence'),('hypothesis reference','hypothesis_reference'),('packet binding','packet_binding'),('choice structure','choice_structure'),('selection count','choice_structure'),('memory size bound','plan_size_bound')]:
  if pattern in msg:return category
 return 'schema_or_json' if msg else ''

def main():
 # Replay is pinned to the first recorded snapshot. Archived members are
 # authenticated through their archive; private dependencies must remain exact.
 previous_manifest=OUT/'verification/reasoning_input_manifest.json'
 if previous_manifest.exists():
  frozen=json.loads(previous_manifest.read_text())
  checked=set()
  for entry in frozen['files']:
   if entry['role']=='archive_member' or entry['path'] in checked:continue
   p=ROOT/entry['path'];assert p.is_file(),f'missing frozen dependency: {p}'
   assert sha(p.read_bytes())==entry['sha256'],f'frozen input drift: {p}'
   checked.add(entry['path'])
 manifest=[]; batches=[]; hypotheses=[]; refs=[]; responses=[]; repairs=[]; text_flags=[]; samples=[]; missing=[]; lineages=[]
 validator=pure_validator(manifest)
 verified_archive_records=0; packet_memory_checks=0
 for seed in SEEDS:
  archive=ARCHIVE/f'seed_{seed}_results.jsonl.gz'
  manifest.append({'path':relative(archive),'sha256':sha(archive.read_bytes()),'role':'published_scientific_archive'})
  archive_records={}
  with gzip.open(archive,'rt') as f:
   for line in f:
    x=json.loads(line); raw=x['raw_utf8'].encode(); assert sha(raw)==x['sha256'], x['path']; verified_archive_records+=1
    if x['path'].endswith(('selection.json','contract.json','feedback.json','batch_freeze.json')):
     archive_records[x['path']]=(json.loads(raw),x['sha256'])
  prior_h=[]; prior_selected=set(); acquired_at={}; prev_ref_mols=collections.Counter()
  for r in range(36):
   directory=STUDY/f'selections/seed_{seed}/free_cw_sol61_xhigh_rerun32/round_{r:02d}'
   def archived(name):
    p=relative(directory/name); d,h=archive_records[p]
    manifest.append({'path':p,'sha256':h,'role':'archive_member','archive':relative(archive)})
    return d
   selection=archived('selection.json'); contract=archived('contract.json'); feedback=archived('feedback.json'); freeze=archived('batch_freeze.json')
   L,U=set(contract['L_t_ids']),set(contract['U_t_ids']); chosen={c['id'] for c in selection['choices']}
   assert len(L)==333+32*r and len(chosen)==32 and chosen<=U and not chosen&L
   assert chosen==set(freeze['batch_ids'])=={x['candidate_id'] for x in feedback['records']}
   assert set(contract['L_t_ids'])==set(acquired_at) if r else True
   if r==0:acquired_at.update({i:-1 for i in L})
   def local(name):
    p=directory/name
    if not p.exists():missing.append({'path':relative(p),'reason':'missing_local_hash_bound_artifact'}); return None
    raw=p.read_bytes(); h=sha(raw); assert h==freeze['artifact_hashes'][name],p
    manifest.append({'path':relative(p),'sha256':h,'role':'private_local_hash_bound_input' if name in ('input.json','catalog.json') else 'frozen_response'})
    return json.loads(raw)
   packet=local('input.json'); catalog=local('catalog.json')
   assert packet is not None and catalog is not None, 'Local frozen packets required; do not substitute states.'
   assert packet['validation_test_records']==0
   obs,cand=decode(packet,'observed'),decode(packet,'candidates'); allrows={**obs,**cand}
   assert set(obs)==L and set(cand)==U
   assert not set(packet['candidates']['columns'])&{'V1_ml','V2_ml','true_V1_ml','true_V2_ml'}
   assert all(set(x)<=L for h in selection['hypotheses'] for x in (h['supporting_observed_ids'],h['contradicting_observed_ids']))
   cat=SimpleNamespace(candidates={x['id']:x for x in catalog['candidates']},observed={x['id']:x for x in catalog['observed']})
   validator(selection,cat,packet,'cw_aware')
   if r:
    assert packet['memory']['previous_hypotheses']==prior_h
    assert {x['candidate_id'] for x in packet['memory']['recent_outcomes']}==prior_selected
    packet_memory_checks+=1
   common={'method':METHOD,'seed':seed,'round':r,'labeled_size_before':333+32*r,'phase':phase(r),'selection_source':relative(directory/'selection.json')}
   support=[]; against=[]; ref_mols=collections.Counter(); ref_counts=collections.Counter(); h_current=[]
   for hindex,h in enumerate(selection['hypotheses']):
    tags=[key for key,pattern in CATEGORIES.items() if re.search(pattern,h['content'],re.I)]
    sup=h['supporting_observed_ids']; con=h['contradicting_observed_ids']; support+=sup; against+=con
    for kind,ids in [('support',sup),('contradiction',con)]:
     for ident in ids:
      row=obs[ident]; ref_mols[row['molecule_id']]+=1; ref_counts[ident]+=1
      refs.append({**common,'hypothesis_id':h['id'],'reference_kind':kind,'observed_id':ident,'molecule_id':row['molecule_id'],'in_preselection_L':True,'acquired_round':acquired_at[ident],'lag_rounds':r-acquired_at[ident] if acquired_at[ident]>=0 else None,'previous_batch':ident in prior_selected,'initial_L333':acquired_at[ident]==-1})
    evidence=sup+con; recent=sum(i in prior_selected for i in evidence)
    matched=max(prior_h,key=lambda p:similarity(p['content'],h['content'])) if prior_h else None
    sim=similarity(matched['content'],h['content']) if matched else None
    row={**common,'hypothesis_index':hindex,'hypothesis_id':h['id'],'content':h['content'],'basis':h['basis'],'confidence':h['confidence'],'status':h['status'],'categories':tags,'support_n':len(sup),'contradiction_n':len(con),'evidence_molecules':len({obs[i]['molecule_id'] for i in evidence}),'previous_batch_evidence_n':recent,'linked_choices_n':sum(c.get('hypothesis_id')==h['id'] for c in selection['choices']),'has_discriminating_question':bool(h['next_discriminating_question'].strip()),'question':h['next_discriminating_question'],'previous_best_text_jaccard':sim,'previous_best_id':matched['id'] if matched else None,'identical_content_to_previous':any(h['content']==p['content'] for p in prior_h),'same_id_previous_content_jaccard':next((similarity(p['content'],h['content']) for p in prior_h if p['id']==h['id']),None)}
    hypotheses.append(row); h_current.append(row)
    if matched and sim>=0.55:
     lineages.append({**common,'previous_id':matched['id'],'current_id':h['id'],'jaccard':sim,'previous_confidence':matched['confidence'],'current_confidence':h['confidence'],'previous_status':matched['status'],'current_status':h['status'],'exact_content_repeat':matched['content']==h['content'],'same_evidence_sets':set(matched['supporting_observed_ids'])==set(sup) and set(matched['contradicting_observed_ids'])==set(con),'previous_batch_evidence_n':recent,'interpretation':'lexical_candidate_link_not_verified_semantic_identity'})
   # Prose references are permitted by actual validator; audit provenance separately.
   fields=[('feedback_interpretation',selection['feedback_interpretation']),('batch_rationale',selection['batch_rationale'])]+[(f'choice:{c["id"]}',c['reason']) for c in selection['choices']]+[(f'hypothesis:{h["id"]}',h['content']+' '+h['why_it_matters_for_model_learning']+' '+h['next_discriminating_question']) for h in selection['hypotheses']]
   prose_counts=collections.Counter(); field_latest=[]
   for field,text in fields:
    for token in HEX_NEAR_ID.findall(text):
     if len(token)!=20:text_flags.append({**common,'flag':'malformed_length_prose_reference','field':field,'identifier':token,'text':text,'manual_error_status':'unresolved_reference_no_autocorrection'})
    for ident in ID.findall(text):
     origin='observed' if ident in L else 'same_batch_candidate' if ident in chosen else 'other_candidate' if ident in U else 'unknown'
     prose_counts[origin]+=1
     if field=='feedback_interpretation' and ident in prior_selected:field_latest.append(ident)
     if origin=='unknown':text_flags.append({**common,'flag':'unknown_20hex_prose_reference','field':field,'identifier':ident,'text':text,'manual_error_status':'requires_semantic_review'})
   # Candidate identity claim flags: first molecule mention mismatches current ID.
   # Flags are NOT automatically scientific errors: reference comparisons can be valid.
   mismatch_n=0
   for c in selection['choices']:
    ms=re.findall(r'\bm\d+\b',c['reason'])
    if ms and ms[0]!=cand[c['id']]['molecule_id']:
     mismatch_n+=1; text_flags.append({**common,'flag':'first_molecule_token_differs_selected','field':f'choice:{c["id"]}','identifier':c['id'],'text':c['reason'],'claimed_first_molecule':ms[0],'actual_molecule':cand[c['id']]['molecule_id'],'manual_error_status':'requires_semantic_review'})
   turns=[]
   for name in sorted(freeze['artifact_hashes']):
    if re.fullmatch(r'turn_\d+\.json',name):
     receipt=local(name)
     if receipt is None:continue
     err=''; obj=None
     try:
      obj=json.loads(receipt['answer']);validator(obj,cat,packet,'cw_aware')
     except (ValueError,TypeError,KeyError) as e:err=str(e)
     choice_structure=isinstance(obj,dict) and isinstance(obj.get('choices'),list) and all(isinstance(c,dict) and isinstance(c.get('id'),str) and isinstance(c.get('reason'),str) for c in obj['choices'])
     ids=[c['id'] for c in obj['choices']] if choice_structure else []
     legal_ids=set(ids)&U
     raw_h=obj.get('hypotheses',[]) if isinstance(obj,dict) else []
     raw_h=raw_h if isinstance(raw_h,list) else []
     raw_h=[h for h in raw_h if isinstance(h,dict)]
     evidence_n=sum(len(h.get(k,[])) for h in raw_h for k in ('supporting_observed_ids','contradicting_observed_ids') if isinstance(h.get(k),list))
     record={**common,'turn_file':name,'valid_original_rules':not err,'first_rejection_category':error_class(err),'choice_structure_reliable':choice_structure,'choice_n':len(ids) if choice_structure else None,'legal_unique_choice_n':len(legal_ids) if choice_structure else None,'outside_candidate_n':len(set(ids)-U) if choice_structure else None,'same_batch_duplicate_n':len(ids)-len(set(ids)) if choice_structure else None,'matches_accepted_selection':obj==selection,'served_model':(receipt.get('provenance') or {}).get('model',''),'hypothesis_n':len(raw_h),'structured_evidence_n':evidence_n}
     responses.append(record);turns.append((record,obj,legal_ids))
   assert turns and any(x[0]['matches_accepted_selection'] for x in turns)
   first=turns[0]; corrected=len(turns)>1
   if corrected:
    repairs.append({**common,'complete_response_n':len(turns),'first_valid':first[0]['valid_original_rules'],'first_choice_structure_reliable':first[0]['choice_structure_reliable'],'first_legal_n':len(first[2]),'first_legal_retained_n':len(first[2]&chosen),'first_legal_all_replaced':bool(first[2]) and not bool(first[2]&chosen),'final_selection_set_changed':set(c['id'] for c in first[1].get('choices',[]) if isinstance(c,dict) and isinstance(c.get('id'),str))!=chosen if isinstance(first[1],dict) else None,'first_hypothesis_n':first[0]['hypothesis_n'],'final_hypothesis_n':len(selection['hypotheses']),'first_structured_evidence_n':first[0]['structured_evidence_n'],'final_structured_evidence_n':len(support)+len(against)})
   count=len(support)+len(against); top8={x['candidate_id'] for x in packet['memory']['high_error_measured_examples']}
   batches.append({**common,'hypothesis_n':len(selection['hypotheses']),'high_confidence_n':sum(h['confidence']=='high' for h in selection['hypotheses']),'supported_n':sum(h['status']=='supported' for h in selection['hypotheses']),'weakened_rejected_n':sum(h['status'] in ('weakened','rejected') for h in selection['hypotheses']),'contradiction_hypothesis_n':sum(bool(h['contradicting_observed_ids']) for h in selection['hypotheses']),'support_reference_n':len(support),'contradiction_reference_n':len(against),'unique_evidence_rows':len(ref_counts),'unique_evidence_molecules':len(ref_mols),'evidence_molecule_hhi':sum((n/count)**2 for n in ref_mols.values()) if count else None,'evidence_top_molecule_fraction':max(ref_mols.values())/count if count else None,'previous_batch_evidence_fraction':sum(i in prior_selected for i in support+against)/count if count else None,'memory_top8_evidence_fraction':sum(i in top8 for i in support+against)/count if count else None,'initial_L333_evidence_fraction':sum(acquired_at[i]==-1 for i in support+against)/count if count else None,'feedback_cites_previous_batch':bool(field_latest),'feedback_previous_batch_unique_ids':len(set(field_latest)),'feedback_revision_keyword':bool(REVISIONS.search(selection['feedback_interpretation'])),'exact_hypothesis_repeat_n':sum(x['identical_content_to_previous'] for x in h_current),'prose_observed_references':prose_counts['observed'],'prose_same_batch_references':prose_counts['same_batch_candidate'],'prose_other_candidate_references':prose_counts['other_candidate'],'prose_unknown_references':prose_counts['unknown'],'first_molecule_mismatch_flag_n':mismatch_n,'complete_response_n':len(turns),'first_complete_response_legal':first[0]['valid_original_rules'],'candidate_identity_failures':sum(x[0]['first_rejection_category']=='candidate_identity' for x in turns),'unobserved_evidence_failures':sum(x[0]['first_rejection_category']=='unobserved_evidence' for x in turns),'reference_molecule_overlap_previous':histogram_similarity(ref_mols,prev_ref_mols),**{f'category_{key}_n':sum(key in h['categories'] for h in h_current) for key in CATEGORIES}})
   # Fixed stratified sample selected before inspecting semantic outcomes.
   if r in (0,5,14,20,35) or seed==887 and r in (9,10,11):
    samples.append({**common,'sampling_rule':'every seed rounds 0,5,14,20,35 plus prespecified 887 r9–11','hypotheses':[{'id':h['id'],'content':h['content'],'confidence':h['confidence'],'status':h['status'],'supporting_observed_ids':h['supporting_observed_ids'],'contradicting_observed_ids':h['contradicting_observed_ids']} for h in selection['hypotheses']],'feedback_interpretation':selection['feedback_interpretation'],'batch_rationale':selection['batch_rationale']})
   acquired_at.update({i:r for i in chosen}); prior_selected=chosen;prior_h=selection['hypotheses'];prev_ref_mols=ref_mols
 csv_write('reasoning_batches.csv',batches);csv_write('reasoning_hypotheses.csv',hypotheses);csv_write('reasoning_evidence_references.csv',refs)
 csv_write('reasoning_responses.csv',responses);csv_write('reasoning_repairs.csv',repairs);csv_write('reasoning_text_flags.csv',text_flags);csv_write('reasoning_candidate_lineages.csv',lineages)
 dump('tables/reasoning_fixed_samples.json',samples)
 # Seed, stage and overall summaries keep correlated rounds visibly grouped.
 summaries=[]
 for seed in (*SEEDS,'all'):
  for ph in ('333–525','525–813','813–1005','1005–1485','all'):
   bs=[b for b in batches if (seed=='all' or b['seed']==seed) and (ph=='all' or b['phase']==ph)]
   hs=[h for h in hypotheses if (seed=='all' or h['seed']==seed) and (ph=='all' or h['phase']==ph)]
   summaries.append({'seed':seed,'phase':ph,'batch_n':len(bs),'hypothesis_n':len(hs),'high_confidence_fraction':sum(h['confidence']=='high' for h in hs)/len(hs),'supported_fraction':sum(h['status']=='supported' for h in hs)/len(hs),'with_contradiction_fraction':sum(h['contradiction_n']>0 for h in hs)/len(hs),'high_no_contradiction_n':sum(h['confidence']=='high' and h['contradiction_n']==0 for h in hs),'exact_repeat_fraction':sum(h['identical_content_to_previous'] for h in hs)/len(hs),'first_legal_n':sum(b['first_complete_response_legal'] for b in bs),'mean_evidence_hhi':statistics.mean(b['evidence_molecule_hhi'] for b in bs if b['evidence_molecule_hhi'] is not None),'mean_memory_top8_evidence_fraction':statistics.mean(b['memory_top8_evidence_fraction'] for b in bs if b['memory_top8_evidence_fraction'] is not None),'mean_previous_batch_evidence_fraction':statistics.mean(b['previous_batch_evidence_fraction'] for b in bs if b['previous_batch_evidence_fraction'] is not None),'feedback_cites_previous_batch_n':sum(b['feedback_cites_previous_batch'] for b in bs),**{f'category_{k}_n':sum(k in h['categories'] for h in hs) for k in CATEGORIES}})
 csv_write('reasoning_summary.csv',summaries)
 summary={'accepted_batches':len(batches),'hypotheses':len(hypotheses),'structured_reference_occurrences':len(refs),'structured_references_outside_preselection_L':0,'confidence_counts':dict(collections.Counter(h['confidence'] for h in hypotheses)),'status_counts':dict(collections.Counter(h['status'] for h in hypotheses)),'basis_counts':dict(collections.Counter(h['basis'] for h in hypotheses)),'category_counts':{k:sum(k in h['categories'] for h in hypotheses) for k in CATEGORIES},'complete_responses':len(responses),'first_complete_legal':sum(b['first_complete_response_legal'] for b in batches),'response_failure_categories':dict(collections.Counter(x['first_rejection_category'] for x in responses if not x['valid_original_rules'])),'correction_batches':len(repairs),'first_legal_ids':sum(x['first_legal_n'] for x in repairs),'first_legal_ids_retained':sum(x['first_legal_retained_n'] for x in repairs),'first_legal_all_replaced_batches':sum(x['first_legal_all_replaced'] for x in repairs),'changed_final_sets':sum(x['final_selection_set_changed'] for x in repairs),'unknown_prose_reference_occurrences':sum(b['prose_unknown_references'] for b in batches),'first_molecule_mismatch_flags':sum(b['first_molecule_mismatch_flag_n'] for b in batches),'text_jaccard_candidate_links':len(lineages),'lineage_confidence_down':sum({'low':0,'medium':1,'high':2}[x['current_confidence']]<{'low':0,'medium':1,'high':2}[x['previous_confidence']] for x in lineages),'lineage_confidence_up':sum({'low':0,'medium':1,'high':2}[x['current_confidence']]>{'low':0,'medium':1,'high':2}[x['previous_confidence']] for x in lineages),'lineage_same_content_and_evidence':sum(x['exact_content_repeat'] and x['same_evidence_sets'] for x in lineages),'fixed_sample_batches':len(samples),'verified_archive_records':verified_archive_records,'verified_consecutive_packet_memory_links':packet_memory_checks,'missing':missing,'truth_scope':'only already observed L rows and accepted selected feedback; candidate tables checked for absence of truth columns; no test rows'}
 dump('verification/reasoning_summary.json',summary)
 dump('verification/reasoning_input_manifest.json',{'files':manifest,'scope':'accepted archived artifacts and hash-bound local packets/complete responses','raw_private_packets_exported':False})
 # Small contract checks target leakage and chronology, not implementation mirrors.
 assert len(batches)==180 and len({(b['seed'],b['round']) for b in batches})==180
 assert len(refs)==sum(b['support_reference_n']+b['contradiction_reference_n'] for b in batches)
 assert sum(x['first_complete_response_legal'] for x in batches)==sum(x['valid_original_rules'] for x in responses if x['turn_file']=='turn_00.json')
 assert all(x['lag_rounds'] is None or x['lag_rounds']>=1 for x in refs)
 assert all(x['first_legal_retained_n']<=x['first_legal_n'] for x in repairs)
 dump('verification/reasoning_checks.json',{'accepted_cohort':True,'all_archived_hashes_verified':True,'local_hashes_checked_against_batch_freeze':True,'preselection_L_and_U_exact_match':True,'selected_feedback_exact_match':True,'chronological_evidence_only':True,'memory_previous_hypotheses_and_recent_outcomes_exact':packet_memory_checks,'final_selections_pass_frozen_validator':180,'no_test_truth_no_candidate_truth_no_training_no_api':True,'classification_limits':'regex topic tags and lexical lineages are descriptions, not semantic truth or scientific improvement'})
 print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
