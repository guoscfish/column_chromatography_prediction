#!/usr/bin/env python3
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/nocw_attribution_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parents[1]
def main():
 b=pd.read_csv(OUT/'tables/nocw_batches.csv');b=b[b.primary]
 p=pd.read_csv(OUT/'tables/nocw_test_pairs.csv');m=p[(p.seed==157)&(p['round']==0)&(p.molecule_id=='m39')&(p.pair_kind=='same_batch')].iloc[0]
 h=pd.read_csv(OUT/'tables/nocw_hypotheses.csv');h=h[h.primary]
 fig,axs=plt.subplots(1,3,figsize=(14,4.8));fig.subplots_adjust(left=.06,right=.98,bottom=.24,top=.83,wspace=.35)
 axs[0].bar(['Linked strict\ncondition test','No such graph\nidentified'],[h.has_real_condition_test_structure.sum(),(~h.has_real_condition_test_structure).sum()],color=['#287f8b','#bd9c83']);axs[0].set_ylabel('Hypotheses (n=527)');axs[0].set_title('Structural test proxy, not truth')
 for seed,g in b.groupby('seed'):axs[1].scatter(g.condition_test_fraction,g.gain,s=19,alpha=.65,label=str(seed))
 axs[1].axhline(0,c='grey',lw=.7);axs[1].set(xlabel='Fraction with linked condition-test graph',ylabel='Next fit gain (NRMSE before - after)');axs[1].legend(title='Outer seed',fontsize=7);axs[1].set_title('99 evaluable of 105 sequential batches')
 # Orient the m39 contrast as 5/1 -> 20/1, regardless of frozen choice ordering.
 ca=__import__('json').loads(m.conditions_a);cb=__import__('json').loads(m.conditions_b);sign=1 if ca['PE/EA']=='5/1' else -1
 x=np.arange(2);axs[2].bar(x-.17,[sign*m.pred_delta_V1_ml,sign*m.pred_delta_V2_ml],width=.34,label='Frozen prediction',color='#5384ad');axs[2].bar(x+.17,[sign*m.actual_delta_V1_ml,sign*m.actual_delta_V2_ml],width=.34,label='Selected observation',color='#dd8954')
 axs[2].set_xticks(x,['V1','V2']);axs[2].axhline(0,c='grey',lw=.7);axs[2].set_ylabel('20/1 minus 5/1 (mL)');axs[2].set_title('m39 discriminating case: seed157 r0');axs[2].legend(fontsize=8)
 fig.suptitle('NoCW: scientific test structure and learning value are separate outcomes',fontsize=14)
 fig.text(.02,.02,'Sources: nocw_hypotheses/batches/test_pairs.csv; five seeds, L333-1005. 305/527 hypotheses have a linked single-factor graph;\nabsence does not exclude valid cross-molecule tests. Six batches with no hypotheses are not plotted in the scatter. The right case uses\nthe same molecule, sample amount and loading conditions; responses revealed after selection. No causal acquisition benefit is identified.',fontsize=8)
 for ext in ['png','pdf']:fig.savefig(OUT/'figures'/f'nocw_science_and_gain.{ext}',dpi=175,bbox_inches='tight')
 plt.close(fig)
if __name__=='__main__':main()
