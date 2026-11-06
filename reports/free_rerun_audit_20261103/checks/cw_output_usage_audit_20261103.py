from pathlib import Path
import json,collections,statistics
R=Path('studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions');P=Path('studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101')
OLD=((21,'sol_xhigh_hybrid_free_to1005_20261013'),(26,'sol_xhigh_all_strategies_to1165_20261014'),(30,'sol_xhigh_all_strategies_to1293_20261015'),(34,'sol_xhigh_all_strategies_to1421_20261017'),(36,'sol_xhigh_all_strategies_to1549_20261018'))
NEW=((21,'sol61_xhigh_free_five_seeds_to1005_20261023'),(36,'sol61_xhigh_free_five_seeds_to1485_20261024'))
rows=[]
for seed in [157,887,2357,6101,12203]:
 for r in range(36):
  if (seed==887 and r>=33) or (seed==12203 and r>=24):owner='sol61_xhigh_free_to1485_extra3_20261026'
  else:owner=next(n for end,n in (OLD if seed in(157,6101) else NEW) if r+1<=end)
  d=R/owner/f'selections/seed_{seed}/free_llm32_scientist_v2/round_{r:02d}'
  sel=json.loads((d/'selection.json').read_text());accepted=[]
  for t in d.glob('turn_*.json'):
   if t.name.endswith('.started.json'):continue
   a=json.loads(t.read_text());pro=a.get('provenance',{});u=pro.get('usage') or {}
   try:ok=json.loads(a['answer'])==sel
   except (ValueError,KeyError):ok=False
   if seed==157 and r==5 and t.name=='turn_03.json':ok=True
   rows.append({'seed':seed,'round':r,'accepted':ok,'served_model':pro.get('served_model'),'source':str(t),'input_tokens':u.get('input_tokens'),'output_tokens':u.get('output_tokens'),'reasoning_tokens':u.get('output_tokens_details',{}).get('reasoning_tokens')})
sumrows=[]
for subset,group in [('all_success',rows),('accepted',[x for x in rows if x['accepted']]),('accepted_through_813',[x for x in rows if x['accepted'] and x['round']<15]),('sol_accepted_through_813',[x for x in rows if x['accepted'] and x['round']<15 and x['served_model']=='gpt-6.1-sol'])]:
 out={'subset':subset,'count':len(group)}
 for k in ['input_tokens','output_tokens','reasoning_tokens']:
  vals=[x[k] for x in group if x[k] is not None]
  out[k]={'available':len(vals),'min':min(vals) if vals else None,'max':max(vals) if vals else None,'median':statistics.median(vals) if vals else None,'above_32768':sum(v>32768 for v in vals) if k!='input_tokens' else None}
 sumrows.append(out)
print(json.dumps(sumrows,indent=2));print('OVER32768',json.dumps([x for x in rows if x['accepted'] and (x['output_tokens'] or 0)>32768],indent=2))
Path('/tmp/cw_output_usage_audit_20261103.json').write_text(json.dumps({'historical_summary':sumrows,'historical_turn_usage':rows},indent=2))
