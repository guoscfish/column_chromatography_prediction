#!/usr/bin/env python3
"""Static scientific figures from derived, non-packet audit tables."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]
def rows(name):return list(csv.DictReader((OUT/'tables'/name).open()))
def main():
 batches=rows('reasoning_batches.csv');summaries=rows('reasoning_summary.csv');evidence=rows('reasoning_case_evidence.csv')
 phases=['333–525','525–813','813–1005','1005–1485'];seeds=[157,887,2357,6101,12203]
 fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
 specs=[('evidence_molecule_hhi','Evidence molecule concentration (HHI)'),('memory_top8_evidence_fraction','Fraction of references to memory top-8 errors'),('previous_batch_evidence_fraction','Fraction of references to previous batch')]
 for ax,(field,title) in zip(axes,specs):
  for seed in seeds:
   means=[]
   for ph in phases:
    a=[float(x[field]) for x in batches if int(x['seed'])==seed and x['phase']==ph and x[field]!=''];means.append(np.mean(a))
   ax.plot(range(4),means,'o-',label=str(seed),alpha=.75,linewidth=1.2)
  ax.set_xticks(range(4),phases,rotation=15);ax.set_xlabel('Label-budget phase');ax.set_title(title,fontsize=10);ax.set_ylim(bottom=0);ax.grid(alpha=.2)
 axes[0].legend(title='Seed',fontsize=8)
 fig.suptitle('CW-Free: cited evidence does not increasingly concentrate on top-error memory',fontsize=12)
 fig.savefig(OUT/'figures/reasoning_memory_dynamics.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
 ss=[next(x for x in summaries if x['seed']=='all' and x['phase']==p) for p in phases]
 for field,label in [('high_confidence_fraction','High confidence'),('supported_fraction','Status supported'),('with_contradiction_fraction','Has contrary evidence')]:
  axes[0].plot(range(4),[float(x[field]) for x in ss],'o-',label=label)
 axes[0].set_xticks(range(4),phases,rotation=15);axes[0].set_ylim(0,1);axes[0].legend(fontsize=8);axes[0].set_title('Self-reported states are not calibrated truth');axes[0].set_ylabel('Fraction of hypotheses')
 for s in seeds:
  bs=[x for x in batches if int(x['seed'])==s]
  axes[1].bar(str(s),sum(x['first_complete_response_legal']=='True' for x in bs)/36)
 axes[1].set_ylim(0,1);axes[1].set_ylabel('First complete answer legal fraction');axes[1].set_xlabel('Seed (36 batches each)');axes[1].set_title('Original frozen validator, n=180 batches')
 case=[x for x in evidence if int(x['seed'])==887 and int(x['round'])==10 and x['molecule_id']=='m117']
 order=sorted(case,key=lambda x:float(x['PE_EA'].split('/')[0]))
 for head,color in [('V1','C0'),('V2','C1')]:
  axes[2].plot([5,10],[float(x['pred_'+head+'_ml']) for x in order],'o--',color=color,label=head+' predicted before r10')
  axes[2].plot([5,10],[float(x['observed_after_selection_'+head+'_ml']) for x in order],'s-',color=color,label=head+' observed after r10')
 axes[2].set_xticks([5,10],['5/1','10/1']);axes[2].set_xlabel('PE/EA (higher ratio = lower EA fraction)');axes[2].set_ylabel('Elution volume (mL)');axes[2].set_title('Fixed-loading m117 test: seed 887, r10');axes[2].legend(fontsize=7)
 for ax in axes:ax.grid(alpha=.2)
 fig.savefig(OUT/'figures/reasoning_states_and_discriminating_case.png',dpi=180);plt.close(fig)
 # Captions have explicit denominators and semantic limits beyond plot titles.
 captions={
  'reasoning_memory_dynamics.png':{'source':'tables/reasoning_batches.csv; hash-bound preselection memory and hypothesis references','methods':['new_free_cw_sol61_xhigh'],'seeds':seeds,'batch_n':180,'nonempty_evidence_batches_by_phase':[27,39,29,72],'definition':'per-batch citation-occurrence fractions or molecule HHI, then mean within seed/phase; empty evidence batches excluded. r0 has empty prior selected/top8 memory and contributes 0 where evidence exists. HHI is unitless. Repeated citation occurrences retained.','limits':'Citations are not attention, scientific correctness, or actual acquisition preference; no causal memory benefit inferred.'},
  'reasoning_states_and_discriminating_case.png':{'source':'tables/reasoning_summary.csv, reasoning_batches.csv, reasoning_case_evidence.csv','hypothesis_n_by_phase':[143,197,134,397],'batch_n':180,'case_rows':2,'case_condition':'m117; 150 uL sample, DCM 600 uL; only PE/EA varies','units':'hypothesis/legality fractions unitless; volumes mL','limits':'Confidence/status are LLM self-reports. The 2-row case tests recorded response direction; no independent experimental replicate or causal AL gain.'}}
 (OUT/'verification/reasoning_figure_captions.json').write_text(json.dumps(captions,indent=2,ensure_ascii=False)+'\n')

if __name__=='__main__':main()
