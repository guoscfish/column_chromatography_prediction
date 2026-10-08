from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
BASE=ROOT/'studies/active_learning';REV=BASE/'qgeognn_v2_row_llm_scientist_v2/revisions/astra_high_full_pool_20261001'
sources={}
def read(p):
 p=Path(p);sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def audit(p):
 a=read(p);return {'labels':a['train_rows'],'nrmse':a['best_validation_combined_normalized_rmse']}
curves={};reuse=read(REV/'control_reuse.json')['entries']
for seed in [157,6101]:
 c={}
 for m in ['cw16_llm16_scientist_v2','free_llm32_scientist_v2']:
  rows=[]
  for p in sorted((REV/f'runtime/seed_{seed}/{m}').glob('round_*/prediction_freeze.json')):
   f=read(p);rows.append(audit(Path(f['checkpoint_path']).parent/'fit_audit.json'))
  c[m]=rows
 end=max(r['labels'] for rows in c.values() for r in rows)
 for m in ['center_width_lcmd','random32']:
  rows=[audit((ROOT/e['checkpoint_path']).parent/'fit_audit.json') for e in reuse if e['seed']==seed and e['method']==m and e['budget']<=end]
  for b in range(557,end+1,32):
   r=(b-333)//32
   if m=='random32':p=BASE/f'qgeognn_v2_row_sequential_b32/runtime/seed_{seed}/random/round_{r:02d}/model/fit_audit.json'
   else:p=BASE/f'qgeognn_v2_row_cw_lcmd_to_653/runtime/seed_{seed}/center_width_lcmd/round_{r:02d}/model/fit_audit.json'
   if p.exists():rows.append(audit(p))
  c[m]=sorted(rows,key=lambda r:r['labels'])
 curves[str(seed)]=c
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
styles={'cw16_llm16_scientist_v2':('CW16 + LLM16','#087f8c','-'), 'free_llm32_scientist_v2':('Free LLM32','#7652ad','-'),'center_width_lcmd':('CW32','#de8a26','--'),'random32':('Random32','#8b96a2','--')}
fig,axes=plt.subplots(1,2,figsize=(13.5,5.5),sharey=True)
for ax,(seed,c) in zip(axes,curves.items()):
 for m,(label,color,ls) in styles.items():
  rows=c[m];x=[r['labels'] for r in rows];y=[r['nrmse'] for r in rows]
  ax.plot(x,y,label=label,color=color,linestyle=ls,marker='o',markersize=4,linewidth=2)
  if m in ('cw16_llm16_scientist_v2','free_llm32_scientist_v2'):
   ax.annotate(f'{y[-1]:.4f}',(x[-1],y[-1]),xytext=(6,-6 if m.startswith('cw') else 5),textcoords='offset points',color=color,fontweight='bold')
 end=max(r['labels'] for rows in c.values() for r in rows)
 ax.set_title(f'Seed {seed} | completed through {end} labels',fontweight='bold')
 ax.set_xticks(list(range(333,end+1,32)));ax.tick_params(axis='x',rotation=45);ax.set_xlim(326,end+24)
 ax.set_xlabel('Active training labels');ax.grid(alpha=.18);ax.legend(fontsize=9,loc='upper right' if seed=='157' else 'lower left')
axes[0].set_ylabel('Validation NRMSE (lower is better)')
fig.suptitle('LLM active learning: latest completed results',fontsize=17,fontweight='bold')
fig.text(.06,.015,'Validation only; curves stop at completed fits. Target: 813. Paused at provider daily limit.\nFree deviations: seed157 round6 extra correction + ID transcription repair; seed6101 round1 one extra correction. Extensions are exploratory.',fontsize=9,color='#555555')
fig.tight_layout(rect=[0,.10,1,.94])
for ext in ['png','svg']:fig.savefig(OUT/f'latest_validation.{ext}',dpi=190,bbox_inches='tight')
(OUT/'validation_curves.json').write_text(json.dumps(curves,indent=2)+'\n');(OUT/'source_hashes.json').write_text(json.dumps(sources,indent=2)+'\n')
print(json.dumps({s:{m:r[-1] for m,r in c.items()} for s,c in curves.items()},indent=2))
