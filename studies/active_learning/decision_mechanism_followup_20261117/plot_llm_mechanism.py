#!/usr/bin/env python3
"""Plot the LLM mechanism diagnostic from locked CSVs without recomputing tables."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/decision_llm_figure_mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
METHODS=['historical_free_mixed_model','new_free_cw_sol61_xhigh','nocw']
LABELS=['Historical CW-Free','New CW-Free','NoCW-Free']
COLORS=['#9467ad','#0085a1','#db6900']


def frozen_csv(path):
    raw=path.read_bytes();assert raw==path.read_bytes(),('unstable plot input',str(path))
    return pd.read_csv(io.BytesIO(raw)),{'path':str(path.resolve().relative_to(ROOT)),
        'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def main(args):
    prefix,source1=frozen_csv(args.input_dir/'llm_display_prefix_batches.csv')
    summary,source2=frozen_csv(args.input_dir/'llm_initial_adjustment_summary.csv')
    assert len(prefix)==540 and all(prefix.groupby('method').size().reindex(METHODS)==180)
    assert int(prefix.set_first32.sum())==127
    assert not summary.duplicated(['method','feature','adjustment']).any()
    assert (summary.states==5).all()
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.1),layout='constrained')
    for j,(method,label,color) in enumerate(zip(METHODS,LABELS,COLORS)):
        group=prefix[prefix.method==method]
        fraction=float(group.set_first32.mean()*100)
        axes[0].bar(j,fraction,color=color,alpha=.7)
        seed_rates=group.groupby('seed').set_first32.mean()*100
        axes[0].scatter(j+np.linspace(-.12,.12,len(seed_rates)),seed_rates,c=color,s=25,edgecolors='white',linewidth=.5)
        axes[0].text(j,fraction+1.2,f'{int(group.set_first32.sum())}/{len(group)}',ha='center',fontsize=10)
        stages=['raw','q50_polynomial','q50_molecule_conditions']
        values=[float(summary[(summary.method==method)&(summary.feature=='U_proxy')&
                              (summary.adjustment==stage)].mean_selected_excess.iloc[0]) for stage in stages]
        axes[1].plot(range(3),values,'o-',color=color,label=label,lw=1.5,markersize=5)
    axes[0].set_xticks(range(3),LABELS,rotation=12)
    axes[0].set_ylabel('Accepted exact display-prefix batches (%)')
    axes[0].set_title('Correction-linked prefix behavior\nSmall points: five seed rates')
    axes[0].set_ylim(0,48)
    axes[1].set_xticks(range(3),['Raw','q50 controls','q50 + molecule\n+ condition controls'])
    axes[1].set_ylabel('Selected U-proxy excess over legal pool')
    axes[1].axhline(0,c='gray',ls=':',lw=.8)
    axes[1].set_title('Shared initial states: score composition\nDescriptive projection, not information value')
    axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Two mechanisms to resolve before another full LLM rollout',fontsize=12)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for extension in ['png','pdf','svg']:
        # Omit varying creation dates from vector metadata.
        metadata={'CreationDate':None,'ModDate':None} if extension=='pdf' else {'Date':None} if extension=='svg' else None
        fig.savefig(args.output_dir/f'llm_correction_and_response_scale.{extension}',dpi=190,bbox_inches='tight',metadata=metadata)
    plt.close(fig)
    for source in [source1,source2]:
        assert hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest()==source['sha256']
    manifest={'status':'PASS','sources':[source1,source2],
              'entrypoint':str(Path(__file__).relative_to(ROOT)),
              'analysis_tables_recomputed':False,'analysis_table_writes':0,
              'prefix_batches':540,'exact_prefix_batches':127,
              'interpretation':'Display-prefix frequency and within-state score projection; no causal or information-value claim.'}
    (OUT/'llm_plot_inputs.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'PASS','figures':3,'analysis_tables_recomputed':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-dir',type=Path,default=OUT)
    parser.add_argument('--output-dir',type=Path,default=OUT/'figures')
    main(parser.parse_args())
