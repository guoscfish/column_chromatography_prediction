#!/usr/bin/env python3
"""Academic figures from audited derived tables; no interpolation or truth reads."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/behavior_audit_mpl')
from pathlib import Path
import textwrap,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent;FIG=OUT/'FIGURES'
METHODS=['historical_free_mixed_model','new_free_cw_sol61_xhigh','nocw','cw_gradient_lcmd32','free16_lcmd16']
LABELS=['Historical CW-Free','New CW-Free','NoCW-Free','CW-LCMD32','CW-Free16 + LCMD16']
COLORS=['#9063A9','#007F9B','#D55E00','#3478BD','#B79B22']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
    'axes.titleweight':'medium','pdf.fonttype':42,'svg.fonttype':'none','axes.grid':False})

def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(FIG/f'{name}.{ext}',dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def legend(ax):
    ax.legend(fontsize=8,frameon=False)

def main():
    FIG.mkdir(exist_ok=True);d=pd.read_csv(OUT/'batch_metrics.csv')
    a=d[d.seed.isin([157,6101])&(d.L_after<=1453)]
    def series(ax,g,f,random=False):
        for m,label,color in zip(METHODS,LABELS,COLORS):
            x=g[g.method==m];means=x.groupby('L_before')[f].mean()
            ax.plot(means.index,means,color=color,label=label,lw=1.6)
            if random:
                rr=x.groupby('L_before')[f+'_random_mean'].mean()
                ax.plot(rr.index,rr,color=color,lw=.8,ls='--',alpha=.5)
        ax.set_xlabel('Labels before acquisition');ax.grid(axis='y',alpha=.15)
    fields=[('hamming_novelty_random_delta','Hamming novelty − random'),
            ('single_factor_history_fraction','Single-factor historical control fraction'),
            ('joint_coverage_gain_random_delta','Joint coverage gain − random'),
            ('new_molecule_slot_fraction','Fraction selecting unseen molecules')]
    fig,axes=plt.subplots(2,3,figsize=(14,7.2),layout='constrained')
    for ax,(f,title) in zip(axes.flat,fields):
        for i,(m,label,color) in enumerate(zip(METHODS,LABELS,COLORS)):
            v=a[a.method==m].groupby('seed')[f].mean()
            ax.scatter(np.full(len(v),i)+np.linspace(-.08,.08,len(v)),v,color=color,s=35)
            ax.plot([i-.18,i+.18],[v.mean()]*2,color=color,lw=2)
        ax.set_title(title);ax.set_xticks(range(5),['Hist','NewCW','NoCW','CW-LCMD','16+16']);ax.grid(axis='y',alpha=.18)
    scores=pd.read_csv(OUT/'candidate_score_percentiles.csv')
    scores=scores[scores.seed.isin([157,6101])&(scores.L_after<=1005)]
    for ax,f,title in zip(axes.flat[4:],['G_cw_distance','U_quantile_width'],['G percentile · core L333–1005','U proxy percentile · core L333–1005']):
        for i,(m,label,color) in enumerate(zip(METHODS,LABELS,COLORS)):
            v=scores[(scores.method==m)&(scores.feature==f)].groupby('seed').selected_percentile_mean.mean()
            if not len(v):ax.text(i,.51,'missing',rotation=90,ha='center',fontsize=8);continue
            ax.scatter(np.full(len(v),i)+np.linspace(-.08,.08,len(v)),v,color=color,s=35)
            ax.plot([i-.18,i+.18],[v.mean()]*2,color=color,lw=2)
        ax.axhline(.5,color='gray',ls=':',lw=.8);ax.set_xticks(range(5),['Hist','NewCW','NoCW','CW-LCMD','16+16']);ax.set_title(title);ax.grid(axis='y',alpha=.18)
    fig.suptitle('Figure 1 · Behavior overview | seeds 157 / 6101, L333–1453\nEach point is a seed mean; bars are the two-seed mean',fontsize=13)
    save(fig,'figure1_behavior_overview')
    fig,axes=plt.subplots(2,2,figsize=(12,7.7),layout='constrained')
    for ax,f,title,rand in zip(axes.flat,['hamming_novelty','hamming_novelty_random_delta','condition_novelty','structure_novelty'],
            ['Discrete novelty (dashed: same-state random)','Discrete novelty − same-state random','Fixed normalized condition novelty','Static molecular Tanimoto novelty'],[True,False,True,True]):
        series(ax,a,f,rand);ax.set_title(title)
        if 'delta' in f:ax.axhline(0,c='grey',ls=':',lw=.8)
    legend(axes[0,0]);fig.suptitle('Figure 2 · Context locality changes with budget | two common seeds',fontsize=13)
    save(fig,'figure2_context_stickiness')
    pref=OUT/'behavior_preferences.csv'
    if pref.exists():
        p=pd.read_csv(pref);print('PREFERENCE_COLUMNS',p.columns.tolist(),flush=True)
        # Robustly select the declared common-space model rather than mixing scopes.
        scopes=[s for s in p.scope.unique() if 'common' in s.lower()]
        scope=scopes[0] if scopes else p.scope.iloc[0]
        q=p[(p.scope==scope)&(p.aggregation=='pooled')]
        if q.empty:q=p[(p.scope==scope)&(p.aggregation=='point')]
        if q.empty:q=p[p.scope==scope]
        # Single-feature coefficients carry the least collinearity ambiguity.
        q=q[q.model==q.feature]
        z=q.groupby(['method','feature']).coefficient.mean().unstack().reindex(METHODS)
        fig,axs=plt.subplots(1,3,figsize=(17,5.5),layout='constrained',gridspec_kw={'width_ratios':[1,1,1.25]})
        for ax,model,xfeature,yfeature,title in [(axs[0],'G_cw_distance+D_hamming','G_cw_distance','D_hamming','Shared G + D decomposition'),
            (axs[1],'G_cw_distance+U_quantile_width','G_cw_distance','U_quantile_width','G + U proxy (LLM/Hybrid states)')]:
            subset=p[(p.scope=='common_1005')&(p.model==model)]
            pooled=subset[subset.aggregation=='pooled'].pivot(index='method',columns='feature',values='coefficient')
            for m,label,color in zip(METHODS,LABELS,COLORS):
                if m not in pooled.index:continue
                ax.scatter(pooled.loc[m,xfeature],pooled.loc[m,yfeature],color=color,s=65,label=label)
                seeds=subset[(subset.method==m)&(subset.aggregation=='campaign')].pivot(index='seed',columns='feature',values='coefficient')
                ax.scatter(seeds[xfeature],seeds[yfeature],color=color,s=18,alpha=.35)
            ax.axhline(0,c='gray',lw=.7,ls=':');ax.axvline(0,c='gray',lw=.7,ls=':')
            ax.set(title=title,xlabel='βG (CW distance percentile)',ylabel='βD (novelty)' if ax is axs[0] else 'βU (quantile-width proxy)')
        legend(axs[0]);ax=axs[2]
        im=ax.imshow(z.to_numpy(),cmap='RdBu_r',vmin=-4.5,vmax=4.5,aspect='auto')
        ax.set_xticks(range(len(z.columns)),['C coverage','D novelty','G CW distance','U width']);ax.set_yticks(range(5),LABELS)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                v=z.iloc[i,j];ax.text(j,i,f'{v:.2f}' if np.isfinite(v) else 'missing',ha='center',va='center',color='white' if np.isfinite(v) and abs(v)>3 else 'black')
        fig.colorbar(im,ax=ax,label='Behavior coefficient per common z-scale unit')
        ax.set_title('Single-feature coefficients\nMissing pure-numerical U retained')
        fig.suptitle('Figure 3 · Project-specific behavior decomposition | core L333–1005, two seeds\nLarge points: pooled fit; small points: individual campaigns | proxy coefficients, not paper V/U',fontsize=12)
        save(fig,'figure3_behavior_preferences')
    fig,axes=plt.subplots(2,3,figsize=(12,7.5),layout='constrained')
    for j,(f,title) in enumerate([('hamming_novelty','Novelty'),('rarity','Rarity (raw −log q)'),('joint_coverage_gain','Fixed-reference coverage gain')]):
        series(axes[0,j],a,f);axes[0,j].set_title(title+' · absolute')
        series(axes[1,j],a,f+'_random_delta');axes[1,j].set_title(title+' · minus random');axes[1,j].axhline(0,c='gray',ls=':',lw=.7)
    legend(axes[0,0]);fig.suptitle('Figure 4 · Exploration proxies | 500 random batches per actual state',fontsize=13)
    save(fig,'figure4_exploration')
    curves=pd.read_csv(OUT/'row_curves.csv');g=pd.read_csv(OUT/'gain_associations.csv')
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    for m,label,color in zip(METHODS,LABELS,COLORS):
        x=curves[(curves.method==m)&curves.seed.isin([157,6101])&(curves.budget<=1453)].groupby('budget').nrmse.mean()
        axes[0].plot(x.index,x,color=color,label=label,lw=1.8)
    axes[0].set(title='Raw saved Row validation curves',xlabel='Labels L',ylabel='NRMSE ↓');legend(axes[0])
    for m,label,color in zip(METHODS,LABELS,COLORS):
        x=d[d.method==m];axes[1].scatter(x.joint_coverage_gain_random_delta,x.next_gain,color=color,s=7,alpha=.25)
    axes[1].set(title='Each point is a historical batch',xlabel='Joint coverage gain − random',ylabel='Next-fit loss improvement ↑')
    axes[1].axhline(0,c='grey',lw=.7);axes[1].axvline(0,c='grey',lw=.7)
    z=g[g.scope=='five_seed_all_methods'];ys=np.arange(len(z))
    axes[2].errorbar(z.mean_seed_partial_rho,ys,xerr=[z.mean_seed_partial_rho-z.cluster_bootstrap_low,z.cluster_bootstrap_high-z.mean_seed_partial_rho],fmt='o',color='#26576F',capsize=2)
    axes[2].set_yticks(ys,[f.replace('_random_delta','').replace('_',' ') for f in z.feature],fontsize=8)
    axes[2].set(title='Seed-block descriptive partial ranks',xlabel='Mean partial rank correlation')
    axes[2].axvline(0,c='grey',ls=':',lw=.8)
    fig.suptitle('Figure 5 · Behavior and subsequent Row performance | association, not intervention',fontsize=13)
    save(fig,'figure5_row_performance')
    hybrid=pd.read_csv(OUT/'hybrid_attribution.csv')
    gradient=pd.read_csv(OUT/'hybrid_gradient_attribution.csv')
    fig,grid=plt.subplots(2,2,figsize=(12,8),layout='constrained');axes=grid.ravel()
    for seed,color,offset in [(157,'#236F91',-.07),(6101,'#C26532',.07)]:
        x=hybrid[hybrid.seed==seed]
        vals=[x.llm_cw_distance_percentile_mean.mean(),x.numerical_cw_distance_percentile_mean.mean()]
        axes[0].plot(np.array([0,1])+offset,vals,'o-',color=color,label=f'seed {seed}')
        vals=[x.joint_llm_order_average.mean(),x.joint_numerical_order_average.mean()]
        axes[1].plot(np.array([0,1])+offset,vals,'o-',color=color)
        vals=[x.llm_same_molecule_single_factor_history_fraction.mean(),x.numerical_same_molecule_single_factor_history_fraction.mean()]
        axes[2].plot(np.array([0,1])+offset,vals,'o-',color=color)
        gx=gradient[(gradient.seed==seed)&(gradient.space=='cw_gradient')]
        vals=[gx.llm_order_average_relative.mean(),gx.numerical_order_average_relative.mean()]
        axes[3].plot(np.array([0,1])+offset,vals,'o-',color=color)
    for ax in axes:ax.set_xticks([0,1],['LLM16','CW-LCMD16']);ax.grid(axis='y',alpha=.15)
    axes[0].set_title('Current CW-distance percentile');axes[1].set_title('Order-averaged joint coverage attribution');axes[2].set_title('Historical single-factor controls')
    axes[3].set_title('Order-averaged CW-gradient coverage attribution');axes[3].set_ylabel('Relative to current reference distance')
    legend(axes[0]);fig.suptitle('Figure 6 · Hybrid complementarity | 71 source-verified batches; no causal learning attribution',fontsize=12)
    save(fig,'figure6_hybrid')
    fig,axes=plt.subplots(2,3,figsize=(13,7.5),layout='constrained')
    for j,seeds in enumerate([[157,6101],[887,2357,12203],[157,887,2357,6101,12203]]):
        for m,label,color in zip(METHODS[:3],LABELS[:3],COLORS[:3]):
            a1=d[(d.method==m)&d.seed.isin(seeds)]
            z=a1.groupby('L_before').hamming_novelty_random_delta.mean();axes[0,j].plot(z.index,z,color=color,label=label)
            z=curves[(curves.method==m)&curves.seed.isin(seeds)].groupby('budget').nrmse.mean();axes[1,j].plot(z.index,z,color=color)
        axes[0,j].set_title(['Mixed-model historical prefix (2 seeds)','Sol/xhigh from start (3 seeds)','All 5 seeds (descriptive)'][j]);axes[0,j].axhline(0,c='grey',ls=':',lw=.7)
        axes[0,j].set_ylabel('Novelty − same-state random');axes[1,j].set_ylabel('Raw Row NRMSE ↓');axes[1,j].set_xlabel('Labels L')
    legend(axes[0,0]);fig.suptitle('Figure 7 · Three LLM trajectories | same labels do not imply same frozen state',fontsize=13)
    save(fig,'figure7_llm_trajectories')
    cases=OUT/'decision_cases.csv'
    if cases.exists():
        c=pd.read_csv(cases);print('CASE_COLUMNS',c.columns.tolist(),flush=True)
        fig,axes=plt.subplots(4,2,figsize=(14,11),layout='constrained')
        llm_methods=[m for m in METHODS if m!='cw_gradient_lcmd32']
        for row,m in enumerate(llm_methods):
            g=c[c.method==m].sort_values('hamming_novelty_random_delta')
            for col,record in enumerate([g.iloc[0],g.iloc[-1]]):
                ax=axes[row,col];ax.set_facecolor('#F3F6F9');ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_visible(False)
                ax.text(.025,.93,f'{dict(zip(METHODS,LABELS))[m]} | seed {record.seed} | L{record.L_before}',
                    transform=ax.transAxes,fontsize=10,weight='bold',va='top')
                level='Low corrected novelty' if col==0 else 'High corrected novelty'
                ax.text(.025,.77,f'{level}: batch32 percentile {record.hamming_novelty_random_percentile:.3f}\n'
                    f'Batch32 joint coverage percentile {record.joint_coverage_gain_random_percentile:.3f} | {record.stage}',
                    transform=ax.transAxes,fontsize=9,color='#286280',va='top')
                excerpt=str(record.rationale_excerpt)[:250]
                ax.text(.025,.49,'\n'.join(textwrap.wrap(excerpt,76))+'…',transform=ax.transAxes,fontsize=8.6,va='top',linespacing=1.35)
                chosen=str(record.selected_ids_excerpt).split('|')[0]
                if m=='free16_lcmd16':
                    match=re.match(r'([0-9a-f]{20}):',str(record.candidate_reason_excerpt))
                    if match:chosen=match.group(1)
                ax.text(.025,.05,'Accepted LLM ID: '+chosen,transform=ax.transAxes,fontsize=8,color='#66717C')
        fig.suptitle('Figure 8 · Visible rationale and numerical action: examples and counterexamples\n'
            'Hybrid: LLM16 rationale with final batch32 metrics. Full 32 cases and IDs in decision_cases.csv.',fontsize=12)
        save(fig,'figure8_decision_cases')
    print('Figures',len(list(FIG.glob('*.png'))),flush=True)

if __name__=='__main__':main()
