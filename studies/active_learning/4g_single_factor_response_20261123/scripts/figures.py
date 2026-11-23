"""Static scientific figures, in PNG and vector PDF."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run import AXES, OUT

LABELS = ['EA fraction','Sample load','Loading-solvent volume','Flow rate','Loading-solvent type']
COLORS = ('#176c82','#bc5521')


def save(figure, name):
    for suffix in ('png','pdf'):
        figure.savefig(OUT/'figures'/f'{name}.{suffix}', dpi=260, bbox_inches='tight')
    plt.close(figure)


def figures(pairs, levels, model_pairs, summary, global_metrics):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for axis, name in zip(AXES[:2], ('eluent','sample_load')):
        data = levels[levels.scope.eq('coherent') & levels.axis.eq(axis)]
        ordered = data[['series','canonical_smiles','n_levels']].drop_duplicates().sort_values(['n_levels','series'], ascending=[False,True])
        chosen = ordered.drop_duplicates('canonical_smiles').head(4)
        fig, axs = plt.subplots(2,2,figsize=(8,5),layout='constrained')
        for ax, entry in zip(axs.flat,chosen.itertuples()):
            rows = data[data.series.eq(entry.series)].sort_values('level')
            for endpoint, color in zip(('V1','V2'), COLORS):
                ax.plot(rows.level.astype(float),rows[endpoint],'.-',color=color,label=endpoint)
            ax.set(title=f'CAS {rows.CAS.iloc[0]}  |  {entry.series[:8]}', xlabel='EA fraction' if axis=='ea' else 'Sample load (µL)', ylabel='Volume (mL)')
        axs.flat[0].legend(); fig.suptitle('Observed responses | most levels, one series per molecule')
        save(fig, name)
    fig, axs = plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    data = pairs[pairs.scope.eq('coherent')]
    for ax,target in zip(axs,('V1','V2')):
        groups = [data[data.axis.eq(axis)][f'd_{target}'].to_numpy() for axis in AXES]
        ax.boxplot(groups,orientation='horizontal',tick_labels=LABELS,showfliers=False)
        ax.axvline(0,color='0.5',lw=1); ax.set(xlabel=f'Observed Δ{target} (mL)',title='Recorded condition means; pair distributions')
    save(fig,'physical_distributions')
    data = model_pairs[(model_pairs.cohort=='qualification_row') & (model_pairs.seed==42) & (model_pairs.role=='validation')]
    fig, axs = plt.subplots(2,2,figsize=(8,7),layout='constrained')
    for row, axis in enumerate(AXES[:2]):
        sub = data[data.axis.eq(axis)]
        for col, endpoint in enumerate(('V1','V2')):
            ax = axs[row,col]
            a,p = sub[f'd_{endpoint}'],sub[f'd_p_{endpoint}']
            ax.scatter(a,p,s=17,alpha=.65,color=COLORS[col]); span = max(1,abs(a).max(),abs(p).max())
            ax.plot([-span,span],[-span,span],'--',color='0.5',lw=1)
            ax.set(xlabel=f'True Δ{endpoint} (mL)',ylabel=f'Predicted Δ{endpoint} (mL)',title=f'{LABELS[row]} | n={len(sub)}')
    fig.suptitle('Qualification row seed 42 | validation only'); save(fig,'truth_vs_prediction')
    frozen = summary[(summary.cohort=='al_new_free') & (summary.threshold==.5)]
    qual = summary[(summary.cohort=='qualification_row') & (summary.seed==42) & (summary.role=='validation') & (summary.threshold==.5)]
    fig, ax = plt.subplots(figsize=(9,4.5),layout='constrained')
    for end, offset, color in zip(('V1','V2'),(-.13,.13),COLORS):
        for index,axis in enumerate(AXES):
            result = qual[(qual.axis==axis)&(qual.target==end)]
            if len(result):
                val = result.iloc[0]
                ax.scatter(index+offset,val.difference_mae,color=color,label=end if index==0 else None)
                ax.vlines(index+offset,val.mae_lo,val.mae_hi,color=color)
    ax.set(xticks=range(len(AXES)),xticklabels=LABELS,ylabel='Molecule-weighted Δ MAE (mL)',title='Qualification row seed 42 | validation; blanks = no pairs'); ax.legend()
    save(fig,'local_error_by_axis')
    fig, axs = plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for ax,endpoint in zip(axs,('V1','V2')):
        for seed,block in frozen[(frozen.axis=='V/ul')&(frozen.target==endpoint)].groupby('seed'):
            ax.plot(block.budget,block.amplitude_ratio,'o-',label=seed)
        ax.axhline(1,color='0.5',lw=1,ls='--'); ax.set(xlabel='Training labels',ylabel='Predicted / true mean |Δ|',title=f'Sample load | {endpoint}')
    axs[1].legend(title='Seed'); fig.suptitle('Same validation IDs and contrasts within each seed'); save(fig,'matched_budget_compression')
    subset = model_pairs[(model_pairs.cohort=='al_new_free')&(model_pairs.seed==157)&(model_pairs.budget==525)&(model_pairs.axis=='V/ul')].copy()
    subset['score'] = subset.d_V1.abs()
    anomaly = subset[subset.d_p_V1.abs().lt(.1)].nlargest(4,'score')
    fig, axs = plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for ax,target in zip(axs,('V1','V2')):
        idx = np.arange(len(anomaly)); ax.bar(idx-.18, anomaly[f'd_{target}'],.36,label='Observed',color=COLORS[0]); ax.bar(idx+.18,anomaly[f'd_p_{target}'],.36,label='Frozen q50',color=COLORS[1]);
        ax.set(xticks=idx,xticklabels=anomaly.series.str[:8],ylabel=f'Δ{target} (mL)'); ax.legend()
    fig.suptitle('Post hoc exceptions: top |true ΔV1| among flat predictions | seed 157 L525'); save(fig,'flat_prediction_examples')
    both = frozen[frozen.axis.eq('V/ul')].merge(global_metrics,on=['cohort','seed','budget','role'])
    fig, axs = plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for ax,endpoint in zip(axs,('V1','V2')):
        for seed,block in both[both.target.eq(endpoint)].groupby('seed'):
            ax.plot(block.combined_normalized_rmse,block.difference_mae,'o-',label=seed)
        ax.set(xlabel='Global validation NRMSE',ylabel=f'Local Δ{endpoint} MAE (mL)',title='Sample load | each line one seed')
    axs[1].legend(title='Seed'); save(fig,'global_vs_local')
