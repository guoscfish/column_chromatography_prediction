from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
curves=json.loads((ROOT/'reports/llm_hybrid_complete_20261010/validation_curves.json').read_text())
NEW=ROOT/'studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/free_sol_xhigh_continuation_20261012'
starts={'157':525,'6101':653};sources={}
for seed in curves:
 rows=[]
 for f in sorted((NEW/f'runtime/seed_{seed}/free_llm32_scientist_v2').glob('round_*/prediction_freeze.json')):
  d=json.loads(f.read_text())
  for k in ('checkpoint','prediction'):
   assert hashlib.sha256(Path(d[k+'_path']).read_bytes()).hexdigest()==d[k+'_sha256']
  p=Path(d['checkpoint_path']).parent/'fit_audit.json';a=json.loads(p.read_text())
  rows.append({'labels':a['train_rows'],'nrmse':a['best_validation_combined_normalized_rmse']});sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 curves[seed]['free_llm32_scientist_v2']=rows
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(18,7),sharey=True)
for ax,seed in zip(axes,['157','6101']):
 c=curves[seed]
 for m,label,color in [('cw16_llm16_scientist_v2','CW16 + LLM16 (Astra/high)','#007f88'),('center_width_lcmd','CW32','#e58a19'),('random32','Random32','#8b96a2')]:
  rows=c[m];ax.plot([r['labels'] for r in rows],[r['nrmse'] for r in rows],label=label,color=color,lw=2,ls='-' if m.startswith('cw16') else '--')
 rows=c['free_llm32_scientist_v2'];before=[r for r in rows if r['labels']<=starts[seed]];after=[r for r in rows if r['labels']>=starts[seed]]
 ax.plot([r['labels'] for r in before],[r['nrmse'] for r in before],label='Free: Astra/high',color='#7952b3',lw=3,marker='o',ms=5)
 if len(after)>1:ax.plot([r['labels'] for r in after],[r['nrmse'] for r in after],label='Free: Sol/xhigh continuation',color='#cc3979',lw=3.5,marker='o',ms=6)
 ax.axvline(starts[seed],color='#cc3979',ls=':',alpha=.75)
 ax.text(starts[seed]-6,1.035,f'Switch after {starts[seed]}',ha='right',fontsize=10,color='#a12d60')
 last=rows[-1];ax.annotate(f"Free {last['labels']}: {last['nrmse']:.4f}",(last['labels'],last['nrmse']),xytext=(8,12),textcoords='offset points',color='#cc3979' if len(after)>1 else '#7952b3',fontweight='bold')
 ax.set_title(f'Seed {seed}',fontsize=19,fontweight='bold');ax.set_xlim(326,890);ax.set_xticks([333,397,461,525,589,653,717,781,813]);ax.tick_params(axis='x',rotation=40);ax.grid(alpha=.15);ax.legend(fontsize=10,loc='upper right');ax.set_xlabel('Active training labels')
axes[0].set_ylabel('Validation NRMSE (lower is better)')
fig.suptitle('Free model switch: completed validation fits | 2026-10-13',fontsize=21,fontweight='bold')
fig.text(.065,.025,'Free switched after 525 labels for seed157 and after 653 for seed6101. Curves end at verified frozen fits.\nExploratory, validation-only; no paired Astra continuation at the same later budgets. Historical authorized correction deviations apply.',fontsize=11,color='#555')
fig.tight_layout(rect=[0,.1,1,.93])
for ext in ('png','svg','pdf'):fig.savefig(OUT/f'model_switch_validation.{ext}',dpi=200,bbox_inches='tight')
(OUT/'validation_curves.json').write_text(json.dumps(curves,indent=2)+'\n');(OUT/'source_hashes.json').write_text(json.dumps(sources,indent=2)+'\n')
print('Latest seed157 relative change', (after[-1]['nrmse'] if False else curves['157']['free_llm32_scientist_v2'][-1]['nrmse'])/curves['157']['free_llm32_scientist_v2'][6]['nrmse']-1)
