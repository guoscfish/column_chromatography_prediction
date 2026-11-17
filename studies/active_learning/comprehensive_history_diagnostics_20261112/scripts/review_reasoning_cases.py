#!/usr/bin/env python3
"""Deterministic case extraction plus explicitly manual flag adjudication.

The manual rules below were written after reading all 38 molecule-token flags.
They do not provide exhaustive semantic detection or a semantic error rate.
"""
from pathlib import Path
import collections,csv,hashlib,json

OUT=Path(__file__).resolve().parents[1]; ROOT=OUT.parents[2]
STUDY=ROOT/'studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101'
def read_csv(name):return list(csv.DictReader((OUT/'tables'/name).open()))
def write_csv(name,rows):
 keys=list(dict.fromkeys(k for x in rows for k in x))
 with (OUT/'tables'/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
 rows=read_csv('reasoning_text_flags.csv');counter=collections.Counter()
 for x in rows:
  s,r=int(x['seed']),int(x['round'])
  if x['flag']=='first_molecule_token_differs_selected':
   if (s,r) in [(157,7),(887,18),(6101,6)]:
    verdict='legitimate_cross_molecule_reference_not_identity_error'
   elif (s,r)==(12203,21):
    verdict='false_matched_loading_contrast_cross_molecule_and_multiple_conditions'
   else: verdict='confirmed_selected_molecule_misattribution'
  elif x['flag']=='unknown_20hex_prose_reference':verdict='untraceable_reference_no_guess_correction'
  else:verdict='malformed_identifier_no_guess_correction'
  x['manual_adjudication']=verdict;counter[verdict]+=1
 write_csv('reasoning_flag_adjudications.csv',rows)
 manifest=json.loads((OUT/'verification/reasoning_input_manifest.json').read_text())
 hashes={x['path']:x['sha256'] for x in manifest['files']}
 cases=[]
 requests=[(887,9,{'m125','m116','m72','m114'}),(887,10,{'m125','m117'}),(887,11,{'m125','m117'}),(6101,34,{'m179'}),(2357,19,{'m72'})]
 for seed,r,mols in requests:
  directory=STUDY/f'selections/seed_{seed}/free_cw_sol61_xhigh_rerun32/round_{r:02d}'
  def load(name):
   p=directory/name;raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==hashes[str(p.relative_to(ROOT))]
   return json.loads(raw)
  p,s,f=load('input.json'),load('selection.json'),load('feedback.json')
  meta={row[0]:dict(zip(p['candidates']['columns'],row)) for row in p['candidates']['rows']}; fb={x['candidate_id']:x for x in f['records']}
  for c in s['choices']:
   row=meta[c['id']]
   if row['molecule_id'] not in mols:continue
   q=fb[c['id']]; cond=row['conditions']
   cases.append({'seed':seed,'round':r,'id':c['id'],'molecule_id':row['molecule_id'],'PE_EA':cond[0],'density_g_ml':cond[1],'sample_ul':cond[2],'loading_solvent':cond[3],'loading_solvent_ul':cond[4],'pred_V1_ml':q['pred_V1_ml'],'pred_V2_ml':q['pred_V2_ml'],'observed_after_selection_V1_ml':q['true_V1_ml'],'observed_after_selection_V2_ml':q['true_V2_ml'],'evidence_time':'metadata and predictions before selection; observed V1/V2 after selected batch freeze','source':str((directory/'feedback.json').relative_to(ROOT))})
 write_csv('reasoning_case_evidence.csv',cases)
 # All 25 prespecified grid samples, plus the requested 887/9–11 chain, were read.
 # These notes describe review scope; they are not full-plan correctness grades.
 notes={
 (157,0):'Broad tail statement; cited provenance passes. No causal learning benefit established.',
 (157,5):'Scaffold-specific narrative; strong-condition wording is insufficient to define physical direction.',
 (157,14):'Explicit contrary IDs exist; broad graph-specific claim remains weakly falsifiable.',
 (157,20):'Explicit alternative noise explanations and failed interpolation predictions; no universal loading mechanism claimed.',
 (157,35):'Endpoint-specific counterexamples recognized; one selected molecular identity error in other choice.',
 (887,0):'General structure/solvent interaction claim; causal benefit to retraining untested.',
 (887,5):'Corrects three prior molecule assignments and 0/1 solvent-strength wording; exact observed-table lookup confirms identities.',
 (887,9):'m116 selection mislabeled m125, producing a false within-molecule solvent-volume contrast.',
 (887,10):'Acknowledges identity error, lowers loading-rule confidence, selects genuine m125 and m117 discriminating contrasts.',
 (887,11):'Targets continuing m125 surface and m117 ratio curve, but final structured evidence arrays all empty after correction.',
 (887,14):'Identifies more prior identity errors and describes conditional counterexamples; not all numeric prose manually verified.',
 (887,20):'Weakened high-loading-tail hypothesis and counterexample IDs explicit; alternative to width-only ranking.',
 (887,35):'Generic response-curve narrative; repeated style does not quantify semantic progress.',
 (2357,0):'General nonlinear-eluent hypothesis; another selected choice mislabels m41 as m160.',
 (2357,5):'Supporting and contrary references present; strength terminology ambiguous; malformed prose IDs present.',
 (2357,14):'High/low strength terms are inconsistent with PE/EA direction in described curve; claims need typed conditions.',
 (2357,20):'Corrects solvent-strength terminology and cites verified fixed-loading m72 ratio pair from r19.',
 (2357,35):'Explicitly reduces broad mechanism claims and revisits older attribution; no continuation after final batch.',
 (6101,0):'Hypothesis about learning value exceeds what cited observed labels can establish.',
 (6101,5):'Empty accepted hypothesis list after correction; informative prose remains but typed claim/evidence chain absent.',
 (6101,14):'Empty accepted hypothesis list after correction despite detailed feedback prose.',
 (6101,20):'Separates repeat variability, graph-dependent error direction and confounded loading packages.',
 (6101,35):'m179 rejection is about magnitude; observed V2 reversal remains positive. Rewritten content/status subject ambiguous.',
 (12203,0):'Broad late regime claim with valid provenance; causal learning benefit untested.',
 (12203,5):'Scaffold-specific transitions; molecule m209 incorrectly assigned to selected m143 choice.',
 (12203,14):'Hypothesis says late at stronger eluent while feedback points to higher PE/EA; physical terminology inverted.',
 (12203,20):'Rejects universal endpoint/loading correction; cited provenance passes.',
 (12203,35):'Final text refers to no labels after earlier same-round draft; cannot equate this with absent previous-round feedback.'}
 samples=json.loads((OUT/'tables/reasoning_fixed_samples.json').read_text())
 write_csv('reasoning_fixed_sample_review.csv',[{'seed':x['seed'],'round':x['round'],'sampling_rule':x['sampling_rule'],'hypothesis_n':len(x['hypotheses']),'scope':'first hypothesis plus feedback/rationale; flagged choices reviewed separately','review_note':notes[(x['seed'],x['round'])],'source':x['selection_source']} for x in samples])
 summary={'adjudication_counts':dict(counter),'all_flagged_texts_read':len(rows),'grid_sample_batches':25,'prespecified_case_batches':3,'case_selected_records':len(cases),'statistical_scope':'identity count is a lower bound from exact m-token detector, not an exhaustive semantic error rate','reviewer':'Codex independent offline semantic inspection; not blinded human inter-rater validation'}
 (OUT/'verification/reasoning_manual_review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
