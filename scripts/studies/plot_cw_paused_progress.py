#!/usr/bin/env python
"""Plot only completed validation scores; no model fitting or label access."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'studies/active_learning'
BASE=P/'qgeognn_v2_row_cw_selector_factorial'
FIRST=P/'qgeognn_v2_row_cw_selector_extension_to_653'
LAST=P/'qgeognn_v2_row_cw_selector_extension_to_1005'
OUT=LAST/'paused_results'
OUT.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=pd.read_csv(BASE/'validation_curves.csv')[['seed','arm','budget','combined_normalized_RMSE','checkpoint_sha256']].to_dict('records')
for s in (FIRST,LAST):
    for entry in read(s/'REUSE_MANIFEST.json')['entries']:
        checkpoint=ROOT/entry['checkpoint']; a=read(checkpoint.with_name('fit_audit.json'))
        assert sha(checkpoint)==entry['checkpoint_sha256']==a['checkpoint_sha256']
        rows.append({'seed':entry['seed'],'arm':entry['arm'],'budget':entry['budget'],
            'combined_normalized_RMSE':a['best_validation_combined_normalized_rmse'],'checkpoint_sha256':a['checkpoint_sha256']})
    for seed in (157,6101):
        for arm in ('cw_maxdet','cw_ivr','whitening_lcmd'):
            for path in (s/f'runtime/seed_{seed}/{arm}').glob('round_*/model/fit_audit.json'):
                state=path.parent.parent/'state.json'
                if not state.exists():continue
                a=read(path)
                assert sha(path.with_name('best.pt'))==a['checkpoint_sha256']==read(state)['checkpoint_sha256']
                rows.append({'seed':seed,'arm':arm,'budget':a['train_rows'],
                    'combined_normalized_RMSE':a['best_validation_combined_normalized_rmse'],'checkpoint_sha256':a['checkpoint_sha256']})
df=pd.DataFrame(rows).sort_values(['seed','arm','budget'])
assert not df.duplicated(['seed','arm','budget']).any()
df.to_csv(OUT/'completed_validation_curves.csv',index=False)
styles={
'raw_lcmd':('Raw-LCMD','#2674bf','--'), 'cw_lcmd':('CW-LCMD','#2674bf','-'),
'raw_maxdet':('Raw-MaxDet','#dc7f20','--'), 'cw_maxdet':('CW-MaxDet','#dc7f20','-'),
'raw_ivr':('Raw-IVR','#248d68','--'), 'cw_ivr':('CW-IVR','#248d68','-'),
'whitening_lcmd':('Whitening-LCMD','#8b52b5','-.'), 'random':('Random','#8b8b8b',':')}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
for end,name,title in [(653,'validation_to653.png','Completed validation curves: 333–653'),(1005,'validation_full_progress.png','Completed validation curves: actual stopping points')]:
    fig,axes=plt.subplots(1,2,figsize=(14,5.8),sharey=True)
    for ax,seed in zip(axes,(157,6101)):
        for arm,(label,color,style) in styles.items():
            g=df[(df.seed==seed)&(df.arm==arm)&(df.budget<=end)]
            ax.plot(g.budget,g.combined_normalized_RMSE,label=label,color=color,linestyle=style,
                linewidth=2 if arm.startswith('cw_') else 1.5,marker='o',markersize=3.5,alpha=.95)
            if arm in ('cw_maxdet','cw_ivr','whitening_lcmd'):
                ax.scatter(g.budget.iloc[-1],g.combined_normalized_RMSE.iloc[-1],s=52,facecolor='white',edgecolor=color,zorder=5)
        ax.axvline(525,color='#222222',linestyle=':',linewidth=1,alpha=.55)
        ax.set(title=f'Seed {seed}',xlabel='Number of labeled samples',xlim=(323,end+12))
        ax.set_xticks(sorted(set(list(range(333,end+1,64))+[end])))
        ax.grid(alpha=.17)
    axes[0].set_ylabel('Validation NRMSE (lower is better)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,.04),frameon=False)
    fig.suptitle(title,fontsize=15,y=.98)
    fig.text(.5,.005,'Training paused by user. Hollow markers = current endpoint of new arms. No extrapolation; no new test evaluation.',ha='center',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.16,1,.94))
    fig.savefig(OUT/name,dpi=180)
    plt.close(fig)
summary=df.groupby(['seed','arm']).tail(1)[['seed','arm','budget','combined_normalized_RMSE']]
summary.to_csv(OUT/'completed_endpoints.csv',index=False)
print(summary.to_string(index=False))
