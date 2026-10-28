"""Plot only verified completed validation fits; never starts a study."""
import json, hashlib
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sources={}
def read(p):
 p=Path(p);raw=p.read_bytes();sources[str(p.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
def verify(p,digest):
 p=Path(p);actual=hashlib.sha256(p.read_bytes()).hexdigest();assert actual==digest,str(p);sources[str(p.relative_to(ROOT))]=actual

def fit(freeze_path):
 f=read(freeze_path)
 for k in ('checkpoint','prediction'):verify(f[k+'_path'],f[k+'_sha256'])
 a=read(Path(f['checkpoint_path']).parent/'fit_audit.json')
 assert a['test_labels_used_for_fit_or_checkpoint_selection']==0
 return {'labels':a['train_rows'],'nrmse':a['best_validation_combined_normalized_rmse']},a
base=ROOT/'reports/llm_four_to1005_20261013'
for p,h in read(base/'source_hashes.json').items():verify(ROOT/p,h)
curves=read(base/'validation_curves.json')
for report in ('llm_eight_to1165_20261014','llm_eight_to1293_20261015','llm_eight_to1421_20261017'):
 p=ROOT/'reports'/report
 archive=read(p/'completion_audit.json')
 for key,entry in archive['frozen_fit_audits'].items():
  seed,method,r=key.split(':');f=entry['prediction_freeze'];fp=Path(f['checkpoint_path']).parent.parent/'prediction_freeze.json'
  verify(fp,entry['prediction_freeze_sha256'])
  ap=Path(f['checkpoint_path']).parent/'fit_audit.json';verify(ap,entry['fit_audit_sha256'])
  row,a=fit(fp);assert row['labels']==333+32*int(r)
  rows=curves[seed][method];assert row['labels']==rows[-1]['labels']+32
  rows.append(row)
 assert curves==read(p/'validation_curves.json'),report
latest=ROOT/'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/sol_xhigh_all_strategies_to1549_20261018'
for seed,methods in curves.items():
 for method,rows in methods.items():
  for fp in sorted((latest/f'runtime/seed_{seed}/{method}').glob('round_*/prediction_freeze.json')):
   row,a=fit(fp)
   assert row['labels']==rows[-1]['labels']+32
   rows.append(row)
DUAL='free_llm32_dual_expert_v1'
for seed,revision in [('157','sol61_xhigh_seed157_from621_to813_20261014'),('6101','sol61_xhigh_seed6101_to813_20261014')]:
 rows=[]
 for r in range(16):
  fp=ROOT/f'studies/active_learning/qgeognn_v2_row_llm_dual_expert_v1/revisions/{revision}/runtime/seed_{seed}/{DUAL}/round_{r:02d}/prediction_freeze.json'
  if not fp.exists():break
  row,a=fit(fp);assert row['labels']==333+32*r;rows.append(row)
 continuation=ROOT/f'studies/active_learning/qgeognn_v2_row_llm_dual_expert_v1/revisions/sol61_xhigh_seed{seed}_from813_to1005_20261021'
 for r in range(16,22):
  fp=continuation/f'runtime/seed_{seed}/{DUAL}/round_{r:02d}/prediction_freeze.json'
  if not fp.exists():break
  row,a=fit(fp)
  assert row['labels']==333+32*r and row['labels']==rows[-1]['labels']+32
  rows.append(row)
 curves[seed][DUAL]=rows
stamp=datetime.now().astimezone().isoformat()
summary={seed:{method:rows[-1] for method,rows in methods.items()} for seed,methods in curves.items()}
for name,data in [('validation_curves.json',curves),('source_hashes.json',sources),('snapshot.json',{'updated_at':stamp,'latest':summary,'test_evaluation_performed':False})]:
 (OUT/name).write_text(json.dumps(data,indent=2)+'\n')
colors={'cw16_llm16_scientist_v2':'#007f88','free_llm32_scientist_v2':'#b43e83','center_width_lcmd':'#e58a19','random32':'#8997a5',DUAL:'#2463eb'}
names=dict(zip(colors,['CW16 + LLM16','Free LLM32','CW32','Random32','Dual Expert']))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(20,8),sharey=True)
xmax=max(row['labels'] for methods in curves.values() for rows in methods.values() for row in rows)
for ax,seed in zip(axes,('157','6101')):
 for method,rows in curves[seed].items():
  x=[r['labels'] for r in rows];y=[r['nrmse'] for r in rows]
  ax.plot(x,y,color=colors[method],lw=2.5,marker='D' if method==DUAL else ('o' if 'scientist' in method else None),ms=4,ls='-' if method==DUAL or 'scientist' in method else '--',label=f'{names[method]}: {x[-1]} labels, {y[-1]:.4f}')
 ax.axvline(813,color='#444',ls=':',alpha=.65)
 switch=525 if seed=='157' else 653
 ax.axvline(switch,color=colors['free_llm32_scientist_v2'],ls=':',alpha=.45)
 ax.text(switch+5,.98,f'Free switched after {switch}',fontsize=9,color=colors['free_llm32_scientist_v2'])
 ax.set_title(f'Seed {seed}',fontsize=19,fontweight='bold');ax.set_xlim(325,xmax+20)
 ticks=list(range(333,xmax+1,128))
 if xmax-ticks[-1]>48:ticks.append(xmax)
 ax.set_xticks(ticks);ax.set_xlabel('Active training labels');ax.grid(alpha=.15);ax.legend(fontsize=11,loc='upper right')
axes[0].set_ylabel('Validation NRMSE (lower is better)')
fig.suptitle('Hybrid, Free and Dual Expert | latest verified completed fits',fontsize=22,fontweight='bold')
fig.text(.07,.055,'Hybrid: Astra/high through 813, then Sol/xhigh. Free: Astra/high through 525 (seed157) / 653 (seed6101), then Sol/xhigh.\nDual: seed157 Astra/high through 397, then Sol6.1/xhigh; seed6101 Sol6.1/xhigh from 333. Blue diamonds end at completed fits.\nExploratory two-seed validation; no test evaluation. Historical authorized correction deviations retained. Snapshot: '+stamp[:19],fontsize=10.5,color='#555')
fig.tight_layout(rect=[0,.14,1,.93])
for ext in ('png','svg','pdf'):fig.savefig(OUT/f'comparison_latest.{ext}',dpi=200,bbox_inches='tight')
print(json.dumps({'updated_at':stamp,'latest':summary,'verified_source_files':len(sources)},indent=2))
