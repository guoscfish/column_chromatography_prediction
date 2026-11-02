#!/usr/bin/env python3
"""Plot a fixed published validation snapshot; never load labels or run workers."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(folder):
    snapshot=json.loads((folder/'snapshot.json').read_text())
    rows=list(csv.DictReader((folder/'validation_curves.csv').open()))
    seeds=[r['seed'] for r in snapshot['seeds']]
    methods=['new_free_cw_sol61_xhigh','historical_free_mixed_model','nocw','numerical_hybrid32','gradient_lcmd32']
    labels=['Current Free (Sol/xhigh)','Historical Free (mixed model)','NoCW','Numerical Hybrid','Gradient LCMD']
    colors=['#087f8c','#778091','#c27732','#9274b0','#43885b']
    by={(m,s):sorted([r for r in rows if r['method']==m and int(r['seed'])==s],key=lambda r:int(r['budget'])) for m in methods for s in seeds}
    common=min(int(by[m,s][-1]['budget']) for m in methods for s in seeds)
    xs=sorted({int(r['budget']) for r in by[methods[0],seeds[0]] if int(r['budget'])<=common})
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for m,label,color in zip(methods,labels,colors):
        ys=[statistics.mean(float(next(r for r in by[m,s] if int(r['budget'])==x)['validation_selection_score']) for s in seeds) for x in xs]
        axes[0].plot(xs,ys,'o-',label=label,color=color,lw=1.8,ms=3)
    axes[0].set(title=f'Fair five-seed comparison through L{common}',xlabel='Labeled training rows',ylabel='Mean normalized validation RMSE\n(lower is better)',xticks=xs)
    axes[0].legend(frameon=False,fontsize=8);axes[0].grid(alpha=.15)
    for seed in seeds:
        curve=by[methods[0],seed]
        axes[1].plot([int(r['budget']) for r in curve],[float(r['validation_selection_score']) for r in curve],'o-',label=str(seed),ms=3)
    axes[1].set(title='Current Free: completed fits by seed',xlabel='Labeled training rows',ylabel='Normalized validation RMSE (lower is better)')
    axes[1].legend(title='Seed',frameon=False,fontsize=8);axes[1].grid(alpha=.15)
    fig.text(.5,.025,'Raw nonmonotone validation curves. Accepted batches awaiting a fit are excluded. Historical Free retains mixed-model prefixes.',ha='center',fontsize=8,color='#555555');fig.tight_layout(rect=[0,.07,1,1])
    for ext in ['png','svg','pdf']:fig.savefig(folder/f'validation_comparison.{ext}',bbox_inches='tight')
    report=folder/'REPORT.md';text=report.read_text();image='![Validation curves](validation_comparison.png)'
    if image not in text:report.write_text(text+'\n'+image+'\n')
    manifest=json.loads((folder/'snapshot_manifest.json').read_text());manifest['files']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(folder.iterdir()) if f.is_file() and f.name!='snapshot_manifest.json'};(folder/'snapshot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--snapshot',type=Path,required=True);plot(parser.parse_args().snapshot)
