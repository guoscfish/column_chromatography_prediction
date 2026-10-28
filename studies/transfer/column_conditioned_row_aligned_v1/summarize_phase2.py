"""All-five-seed internal reporting only; no fitting or outer-label access."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_phase2 as r

RED,BLUE='#8C0000','#607D8B'
METRICS=['V1_rmse','V2_rmse','V1_mae','V2_mae','Center_rmse','Width_rmse']


def table(frame,columns,digits=4):
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in frame[columns].itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.{digits}f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)+'\n'


def compute_statistics(results,tails):
    target=results.loc[results.column.isin(r.TARGETS)].copy()
    if len(target)!=100 or target.duplicated(['outer_seed','inner_fold','arm','column']).any():
        raise RuntimeError('exactly 50 fits / 100 target observations required')
    paired=[]
    for (seed,c),frame in target.groupby(['outer_seed','column']):
        a=frame.loc[frame.arm.eq('A1')].set_index('inner_fold').sort_index()
        b=frame.loc[frame.arm.eq('A2')].set_index('inner_fold').loc[a.index]
        for fold in a.index:
            row=dict(outer_seed=int(seed),inner_fold=int(fold),column=c)
            for metric in ['combined_normalized_rmse',*METRICS]:
                row['A1_'+metric]=float(a.loc[fold,metric]);row['A2_'+metric]=float(b.loc[fold,metric])
                row[metric+'_relative_gain']=1-row['A2_'+metric]/row['A1_'+metric]
            row['A2_wins']=row['combined_normalized_rmse_relative_gain']>0
            paired.append(row)
    paired=pd.DataFrame(paired)
    seed_methods=[]
    for (seed,c,arm),frame in target.groupby(['outer_seed','column','arm']):
        row=dict(outer_seed=int(seed),column=c,arm=arm,mean_nrmse=float(frame.combined_normalized_rmse.mean()),
                 fold_sd_nrmse=float(frame.combined_normalized_rmse.std(ddof=1)))
        for metric in METRICS:
            row['mean_'+metric]=float(frame[metric].mean())
            if metric.endswith('_rmse'):
                row['pooled_OOF_'+metric]=float(np.sqrt(np.average(frame[metric]**2,weights=frame.inner_validation_count)))
        for endpoint in ['V1','V2','Center','Width']:
            row[endpoint+'_max_abs_error']=float(frame[endpoint+'_max_abs_error'].max())
        seed_methods.append(row)
    seed_methods=pd.DataFrame(seed_methods)
    seed_pairs=[]
    for (seed,c),frame in seed_methods.groupby(['outer_seed','column']):
        ab=frame.set_index('arm');folds=paired.loc[paired.outer_seed.eq(seed)&paired.column.eq(c)]
        row=dict(outer_seed=int(seed),column=c,A1_mean_nrmse=float(ab.loc['A1','mean_nrmse']),
            A2_mean_nrmse=float(ab.loc['A2','mean_nrmse']),
            mean_paired_fold_gain_percent=float(folds.combined_normalized_rmse_relative_gain.mean()*100),
            gain_of_means_percent=float((1-ab.loc['A2','mean_nrmse']/ab.loc['A1','mean_nrmse'])*100),
            fold_wins=int(folds.A2_wins.sum()),seed_win=bool(ab.loc['A2','mean_nrmse']<ab.loc['A1','mean_nrmse']))
        for metric in ['V1_rmse','V2_rmse','Center_rmse','Width_rmse']:
            row[metric+'_pooled_gain_percent']=float((1-ab.loc['A2','pooled_OOF_'+metric]/ab.loc['A1','pooled_OOF_'+metric])*100)
        for endpoint in ['V1','V2','Center','Width']:
            row[endpoint+'_max_error_gain_percent']=float((1-ab.loc['A2',endpoint+'_max_abs_error']/ab.loc['A1',endpoint+'_max_abs_error'])*100)
        seed_pairs.append(row)
    seed_pairs=pd.DataFrame(seed_pairs)
    summary=[]
    for c in r.TARGETS:
        cp=paired.loc[paired.column.eq(c)];sp=seed_pairs.loc[seed_pairs.column.eq(c)]
        for arm in r.old.ARMS:
            frame=target.loc[target.column.eq(c)&target.arm.eq(arm)]
            means=seed_methods.loc[seed_methods.column.eq(c)&seed_methods.arm.eq(arm)]
            row=dict(column=c,arm=arm,mean_nrmse=float(frame.combined_normalized_rmse.mean()),
                fold_sd_nrmse=float(frame.combined_normalized_rmse.std(ddof=1)),
                seed_mean_sd_nrmse=float(means.mean_nrmse.std(ddof=1)),
                min_fold_nrmse=float(frame.combined_normalized_rmse.min()),max_fold_nrmse=float(frame.combined_normalized_rmse.max()),
                mean_paired_fold_gain_percent=float(cp.combined_normalized_rmse_relative_gain.mean()*100),
                gain_of_means_percent=float((1-sp.A2_mean_nrmse.mean()/sp.A1_mean_nrmse.mean())*100),
                seed_wins=int(sp.seed_win.sum()),fold_wins=int(cp.A2_wins.sum()),seeds=5,folds=25)
            for metric in METRICS:
                row['mean_'+metric]=float(frame[metric].mean());row['fold_sd_'+metric]=float(frame[metric].std(ddof=1))
            summary.append(row)
    summary=pd.DataFrame(summary)
    pooled=tails.loc[tails.column.isin(r.TARGETS)].groupby(['outer_seed','column','arm','endpoint'])[['tail_sse','tail_count','total_sse']].sum().reset_index()
    pooled['tail_rmse']=np.sqrt(pooled.tail_sse/pooled.tail_count)
    pooled['tail_sse_fraction']=pooled.tail_sse/pooled.total_sse
    tail_pairs=[]
    for (seed,c,e),frame in pooled.groupby(['outer_seed','column','endpoint']):
        ab=frame.set_index('arm')
        tail_pairs.append(dict(outer_seed=int(seed),column=c,endpoint=e,
            A1_tail_rmse=float(ab.loc['A1','tail_rmse']),A2_tail_rmse=float(ab.loc['A2','tail_rmse']),
            tail_gain_percent=float((1-ab.loc['A2','tail_rmse']/ab.loc['A1','tail_rmse'])*100),
            tail_count=int(ab.loc['A1','tail_count'])))
    tail_pairs=pd.DataFrame(tail_pairs)
    tail_summary=pooled.groupby(['column','arm','endpoint']).tail_rmse.agg(mean_tail_rmse='mean',seed_sd_tail_rmse='std').reset_index()
    return paired,seed_methods,seed_pairs,summary,pooled,tail_pairs,tail_summary


def epoch_statistics(results):
    group=pd.read_csv(r.old.STUDY/'ROW_INNER_RESULTS.csv')
    epoch_rows=[]
    for cv,frame in [('Group-CV',group),('Row-CV',results)]:
        for row in frame.loc[frame.column.eq('25g')].itertuples():
            epoch_rows.append(dict(cv=cv,outer_seed=int(row.outer_seed),inner_fold=int(row.inner_fold),arm=row.arm,
                best_epoch=int(row.best_epoch),epochs_run=int(row.epochs_run),
                best_at_cap=int(row.best_epoch)==500,training_hit_cap=int(row.epochs_run)==500))
    epochs=pd.DataFrame(epoch_rows)
    selections=[]
    for (cv,seed,arm),frame in epochs.groupby(['cv','outer_seed','arm']):
        frame=frame.sort_values('inner_fold')
        selections.append(dict(cv=cv,outer_seed=int(seed),arm=arm,
            five_best_epochs=', '.join(map(str,frame.best_epoch.tolist())),median_epoch=int(np.floor(frame.best_epoch.median()+.5)),
            mean_best_epoch=float(frame.best_epoch.mean()),fold_sd_best_epoch=float(frame.best_epoch.std(ddof=1)),
            best_at_cap=int(frame.best_at_cap.sum()),training_hit_cap=int(frame.training_hit_cap.sum())))
    selections=pd.DataFrame(selections)
    comparisons=[]
    for seed in r.SEEDS:
        sub=selections.loc[selections.outer_seed.eq(seed)].set_index(['cv','arm'])
        row={'outer_seed':seed}
        for arm in r.old.ARMS:
            a,b=sub.loc[('Group-CV',arm)],sub.loc[('Row-CV',arm)]
            row[arm+'_group_median']=int(a.median_epoch);row[arm+'_row_median']=int(b.median_epoch)
            row[arm+'_median_increase']=int(b.median_epoch-a.median_epoch)
            row[arm+'_group_cap_folds']=int(a.best_at_cap);row[arm+'_row_cap_folds']=int(b.best_at_cap)
        row['group_A2_minus_A1']=row['A2_group_median']-row['A1_group_median']
        row['row_A2_minus_A1']=row['A2_row_median']-row['A1_row_median']
        comparisons.append(row)
    return epochs,selections,pd.DataFrame(comparisons)


def summarize():
    preservation=r.verify_phase1()
    provenance=json.loads((r.STUDY/'PHASE2_EXECUTION_PROVENANCE.json').read_text())
    for rel,expected in provenance['files'].items():
        if r.old.sha(r.ROOT/rel)!=expected:
            raise RuntimeError(f'Phase 2 execution code changed: {rel}')
    frames={}
    for seed in r.SEEDS:
        out=r.PHASE2/f'seed_{seed}'
        receipt=json.loads((out/'COMPLETE_AUDIT.json').read_text())
        if not receipt['all_passed'] or receipt['models']!=10:
            raise RuntimeError('incomplete seed audit')
        for name,expected in receipt['files'].items():
            if r.old.sha(out/name)!=expected:raise RuntimeError('audited seed artifact changed')
            frames.setdefault(name,[]).append(pd.read_csv(out/name))
    frames={name:pd.concat(parts,ignore_index=True) for name,parts in frames.items()}
    frames['TRAINING_HISTORY.csv']['source_validation_labels_available']=frames['TRAINING_HISTORY.csv'].outer_seed.eq(r.p.PILOT_SEED)
    results,tails=frames['INNER_RESULTS.csv'],frames['TAIL_METRICS.csv']
    stats=compute_statistics(results,tails)
    paired,seed_methods,seed_pairs,summary,pooled,tail_pairs,tail_summary=stats
    epochs,selections,epoch_comparison=epoch_statistics(results)
    candidate_epochs=selections.loc[selections.cv.eq('Row-CV')].copy()
    candidate_epochs['final_epoch_selected']=candidate_epochs.median_epoch
    candidate_epochs['refit_executed']=False
    for name,frame in {**frames,'PAIRED_COMPARISONS.csv':paired,'SEED_RESULTS.csv':seed_methods,
        'SEED_PAIRED_COMPARISONS.csv':seed_pairs,'INNER_SUMMARY.csv':summary,'POOLED_TAIL_METRICS.csv':pooled,
        'TAIL_PAIRED_COMPARISONS.csv':tail_pairs,'TAIL_SUMMARY.csv':tail_summary,
        'BEST_EPOCH_DISTRIBUTION.csv':epochs,'EPOCH_SELECTION.csv':candidate_epochs,
        'GROUP_VS_ROW_EPOCH_SELECTION.csv':selections,
        'EPOCH_SELECTION_COMPARISON.csv':epoch_comparison}.items():
        frame.to_csv(r.STUDY/name,index=False)
    figures(seed_pairs,tail_pairs,epochs,frames['TRAINING_HISTORY.csv'],results)
    write_report(summary,seed_pairs,seed_methods,tail_pairs,tail_summary,selections,epoch_comparison,epochs,frames,preservation)
    outputs={str(path.relative_to(r.STUDY)):r.old.sha(path) for path in r.STUDY.iterdir()
             if path.is_file() and path.suffix in ['.csv','.md','.png','.py']}
    outputs.update({str(path.relative_to(r.STUDY)):r.old.sha(path) for path in (r.STUDY/'learning_curves').glob('*.png')})
    manifest={'phase':'PHASE_2_COMPLETE','reused_fits':10,'new_fits':40,'total_fits':50,
        'base_protocol_sha256':r.old.sha(r.STUDY/'protocol.json'),
        'phase2_protocol_sha256':r.old.sha(r.STUDY/'PHASE2_PROTOCOL.json'),
        'phase2_provenance_sha256':r.old.sha(r.STUDY/'PHASE2_EXECUTION_PROVENANCE.json'),
        'source_validation_labels_parsed_in_phase2':False,'outer_labels_parsed':False,'full_data_refit_executed':False,
        'outer_scoring_executed':False,'preservation':preservation,'files':outputs,
        'report_code_sha256':r.old.sha(Path(__file__))}
    r.save(r.STUDY/'PHASE2_REPORT_MANIFEST.json',manifest)
    r.save(r.STUDY/'REPORT_MANIFEST.json',manifest)
    replay=frames['CHECKPOINT_REPLAY.csv']
    r.save(r.STUDY/'CHECKPOINT_REPLAY.json',{'contexts_checked':len(replay),'target_prediction_rows':int(replay.rows.sum()),
        'max_abs_difference_ml':float(replay.max_abs_difference_ml.max()),'all_passed':True,
        'replay_inputs_zero_label':True,'metric_audit_labels':'authorized gradient_train only'})
    r.save(r.STUDY/'HISTORICAL_PRESERVATION_CHECK.json',{'files_checked':preservation['historical_files'],
        'all_unchanged':True,'full_data_refit_executed':False,'outer_evaluation_executed':False,
        'other_seeds_executed':True,'phase1_runtime_unchanged':True})
    print('PHASE 2 COMPLETE: 50 audited fits (10 reused + 40 new); no refit or outer scoring.',flush=True)


def figures(seed_pairs,tail_pairs,epochs,history,results):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(14,5.3))
    order=list(r.SEEDS)
    for ax,c in zip(axes,r.TARGETS):
        frame=seed_pairs.loc[seed_pairs.column.eq(c)].set_index('outer_seed').loc[order]
        x=np.arange(5)
        ax.bar(x-.18,frame.mean_paired_fold_gain_percent,.34,color=RED,label='A: mean paired-fold gain')
        ax.bar(x+.18,frame.gain_of_means_percent,.34,color=BLUE,label='B: gain of means')
        ax.axhline(0,color='#222222',lw=1.4)
        ax.set_xticks(x,[str(seed) for seed in order],rotation=22,ha='right')
        ax.set_title(f'{c} | seed wins {int(frame.seed_win.sum())}/5, fold wins {int(frame.fold_wins.sum())}/25')
        ax.set_ylabel('A2 relative NRMSE improvement (%)\npositive = A2 better')
        ax.grid(axis='y',alpha=.2);ax.legend(fontsize=9)
        lo=min(0.,float(frame.mean_paired_fold_gain_percent.min()),float(frame.gain_of_means_percent.min()))
        hi=max(0.,float(frame.mean_paired_fold_gain_percent.max()),float(frame.gain_of_means_percent.max()))
        span=max(1.,hi-lo);ax.set_ylim(lo-.18*span,hi+.18*span)
        for j,val in enumerate(frame.mean_paired_fold_gain_percent):
            ax.annotate(f'{val:+.2f}',(j-.18,val),xytext=(0,4 if val>=0 else -12),textcoords='offset points',ha='center',fontsize=9)
    fig.suptitle('Five-seed Row-CV: both gain definitions retained | development diagnostics')
    fig.text(.5,.015,'A = mean(1 - A2_fold/A1_fold); B = 1 - mean(A2)/mean(A1). Shared records: folds/seeds are not independent experiments.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.94));fig.savefig(r.STUDY/'five_seed_row_cv_paired_gains.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(16,6))
    for ax,arm in zip(axes,r.old.ARMS):
        for j,seed in enumerate(order):
            for offset,cv,color in [(-.18,'Group-CV',BLUE),(.18,'Row-CV',RED)]:
                vals=epochs.loc[epochs.arm.eq(arm)&epochs.outer_seed.eq(seed)&epochs.cv.eq(cv),'best_epoch'].to_numpy()
                pos=j+offset
                box=ax.boxplot([vals],positions=[pos],widths=.29,patch_artist=True,manage_ticks=False,
                    medianprops={'color':color,'linewidth':2},boxprops={'facecolor':color,'alpha':.22,'edgecolor':color})
                ax.scatter(pos+np.linspace(-.055,.055,len(vals)),vals,color=color,s=24,zorder=4)
                cap=int((vals==500).sum())
                if cap:
                    ax.scatter([pos],[500],marker='^',s=95,facecolor=color,edgecolor='black',zorder=6)
                    ax.text(pos,518,f'{cap} cap',color=color,ha='center',fontsize=8)
        ax.axhline(500,color='#333333',ls='--',lw=1,label='500 epoch ceiling')
        ax.set_ylim(0,555);ax.set_xticks(np.arange(5),[str(seed) for seed in order],rotation=22,ha='right')
        ax.set_title(f'{arm}: 25 best epochs per CV; triangles mark ceiling selections')
        ax.set_ylabel('Best epoch (one actual optimizer step/epoch)');ax.grid(axis='y',alpha=.2)
        ax.plot([],[],color=BLUE,marker='o',label='Historical Group-CV');ax.plot([],[],color=RED,marker='o',label='Row-CV')
        ax.legend(fontsize=9,loc='lower right')
    fig.suptitle('Group-CV vs Row-CV training selection | different validation identities; absolute errors are not compared')
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(r.STUDY/'group_vs_row_cv_best_epochs.png',dpi=170);plt.close(fig)

    labels=['V1','V2','Center','Width','Tail V1','Tail V2','Max V1','Max V2']
    matrices=[]
    for c in r.TARGETS:
        rows=[]
        for seed in order:
            row=seed_pairs.loc[seed_pairs.column.eq(c)&seed_pairs.outer_seed.eq(seed)].iloc[0]
            tail=tail_pairs.loc[tail_pairs.column.eq(c)&tail_pairs.outer_seed.eq(seed)].set_index('endpoint')
            rows.append([row[e+'_rmse_pooled_gain_percent'] for e in ['V1','V2','Center','Width']]
                +[float(tail.loc[e,'tail_gain_percent']) for e in ['V1','V2']]
                +[row[e+'_max_error_gain_percent'] for e in ['V1','V2']])
        matrices.append(np.asarray(rows))
    limit=max(5.,float(np.max(np.abs(matrices))))
    cmap=LinearSegmentedColormap.from_list('pku_blue_diverging',[BLUE,'#FFFFFF',RED])
    fig,axes=plt.subplots(1,2,figsize=(17,5.6))
    for ax,c,values in zip(axes,r.TARGETS,matrices):
        shown=ax.imshow(values,cmap=cmap,norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),aspect='auto')
        ax.set_xticks(np.arange(len(labels)),labels,rotation=30,ha='right')
        ax.set_yticks(np.arange(5),[str(seed) for seed in order]);ax.set_title(c)
        for i in range(5):
            for j in range(len(labels)):
                ax.text(j,i,f'{values[i,j]:+.1f}%',ha='center',va='center',color='white' if abs(values[i,j])>limit*.63 else '#222222',fontsize=9)
    fig.suptitle('All seeds: OOF endpoint / Center / Width / tail / maximum absolute error changes')
    fig.subplots_adjust(left=.08,right=.87,bottom=.2,top=.87,wspace=.25)
    color_axis=fig.add_axes([.91,.22,.014,.61])
    fig.colorbar(shown,cax=color_axis,label='A2 improvement (%) | negative = deterioration')
    fig.text(.5,.02,'RMSE from pooled OOF SSE per seed; tail thresholds from outer gradient_train only. Max = maximum absolute endpoint error across OOF folds.',ha='center',fontsize=9)
    fig.savefig(r.STUDY/'a2_center_width_tail_diagnostics.png',dpi=180);plt.close(fig)

    out=r.STUDY/'learning_curves';out.mkdir(exist_ok=True)
    for seed in order:
        fig,axes=plt.subplots(5,3,figsize=(15,17))
        for fold in range(1,6):
            for arm,color in [('A1',BLUE),('A2',RED)]:
                h=history.loc[history.outer_seed.eq(seed)&history.inner_fold.eq(fold)&history.arm.eq(arm)]
                best=int(results.loc[results.outer_seed.eq(seed)&results.inner_fold.eq(fold)&results.arm.eq(arm),'best_epoch'].iloc[0])
                axes[fold-1,0].plot(h.epoch,h.joint_validation_score,color=color,label=f'{arm}, best={best}')
                axes[fold-1,0].scatter([best],h.loc[h.epoch.eq(best),'joint_validation_score'],color=color,s=20)
                for j,c in enumerate(r.TARGETS,1):
                    axes[fold-1,j].plot(h.epoch,h[f'train_loss_{c}'],color=color,label=arm+' train',alpha=.8)
                    axes[fold-1,j].plot(h.epoch,h[f'validation_loss_{c}'],color=color,ls='--',label=arm+' valid')
                    axes[fold-1,j].set_yscale('log')
                for j in range(3):axes[fold-1,j].axvline(best,color=color,ls=':',alpha=.7)
            axes[fold-1,0].set_ylabel(f'Fold {fold}\nJoint NRMSE')
            for j,title in enumerate(['Joint selector','25g raw quantile loss','40g raw quantile loss']):
                axes[fold-1,j].set_title(title);axes[fold-1,j].legend(fontsize=8)
                axes[fold-1,j].set_xlabel('Epoch / actual optimizer steps');axes[fold-1,j].grid(alpha=.2)
        fig.suptitle(f'Row-CV seed {seed}: full curves; selected epochs marked | train mixed BN, validation eval BN; 4g subsampled')
        fig.tight_layout(rect=(0,0,1,.965));fig.savefig(out/f'row_cv_seed_{seed}.png',dpi=130);plt.close(fig)


def write_report(summary,seed_pairs,seed_methods,tail_pairs,tail_summary,selections,epoch_comparison,epochs,frames,preservation):
    overall=summary.loc[summary.arm.eq('A1')].set_index('column')
    consistent=all(overall.loc[c,'seed_wins']>=4 and overall.loc[c,'mean_paired_fold_gain_percent']>0
                   and overall.loc[c,'gain_of_means_percent']>0 for c in r.TARGETS)
    positive=all(overall.loc[c,'mean_paired_fold_gain_percent']>0 and overall.loc[c,'seed_wins']>=3
                 and overall.loc[c,'gain_of_means_percent']>0 for c in r.TARGETS)
    tail_overall=[]
    for (c,e),sub in tail_summary.groupby(['column','endpoint']):
        ab=sub.set_index('arm');pairs=tail_pairs.loc[tail_pairs.column.eq(c)&tail_pairs.endpoint.eq(e)]
        tail_overall.append(dict(column=c,endpoint=e,A1_mean_seed_tail_rmse=float(ab.loc['A1','mean_tail_rmse']),
            A2_mean_seed_tail_rmse=float(ab.loc['A2','mean_tail_rmse']),
            gain_of_seed_tail_means_percent=float((1-ab.loc['A2','mean_tail_rmse']/ab.loc['A1','mean_tail_rmse'])*100),
            A2_deteriorating_seeds=int((pairs.tail_gain_percent<0).sum())))
    tail_overall=pd.DataFrame(tail_overall);tail_overall.to_csv(r.STUDY/'TAIL_OVERALL_COMPARISONS.csv',index=False)
    profile=seed_methods.groupby(['column','arm']).mean(numeric_only=True).reset_index()
    profiles=[]
    for c in r.TARGETS:
        ab=profile.loc[profile.column.eq(c)].set_index('arm')
        profiles.append(dict(column=c,**{e+'_pooled_RMSE_gain_percent':float((1-ab.loc['A2','pooled_OOF_'+e+'_rmse']/ab.loc['A1','pooled_OOF_'+e+'_rmse'])*100) for e in ['V1','V2','Center','Width']}))
    profile_gains=pd.DataFrame(profiles);profile_gains.to_csv(r.STUDY/'COMPONENT_OVERALL_COMPARISONS.csv',index=False)
    max_tradeoffs=[]
    for row in seed_pairs.itertuples():
        for e in ['V1','V2','Center','Width']:
            max_gain=float(getattr(row,e+'_max_error_gain_percent'))
            max_tradeoffs.append(dict(outer_seed=row.outer_seed,column=row.column,endpoint=e,
                mean_nrmse_improved=bool(row.seed_win),max_error_gain_percent=max_gain,
                mean_improves_but_max_worsens=bool(row.seed_win and max_gain<0)))
    max_tradeoffs=pd.DataFrame(max_tradeoffs);max_tradeoffs.to_csv(r.STUDY/'MAX_ERROR_TRADEOFFS.csv',index=False)
    tradeoff_cases=max_tradeoffs.loc[max_tradeoffs.mean_improves_but_max_worsens]
    group=pd.read_csv(r.old.STUDY/'ROW_INNER_RESULTS.csv');group_comparisons=[]
    for cv,frame in [('Group-CV',group),('Row-CV',frames['INNER_RESULTS.csv'])]:
        for c in r.TARGETS:
            sub=frame.loc[frame.column.eq(c)]
            a=sub.loc[sub.arm.eq('A1')].set_index(['outer_seed','inner_fold']).sort_index()
            b=sub.loc[sub.arm.eq('A2')].set_index(['outer_seed','inner_fold']).loc[a.index]
            gains=1-b.combined_normalized_rmse/a.combined_normalized_rmse
            seed_a=a.combined_normalized_rmse.groupby(level=0).mean();seed_b=b.combined_normalized_rmse.groupby(level=0).mean()
            group_comparisons.append(dict(cv=cv,column=c,mean_paired_fold_gain_percent=float(gains.mean()*100),
                gain_of_means_percent=float((1-b.combined_normalized_rmse.mean()/a.combined_normalized_rmse.mean())*100),
                seed_wins=int((seed_b<seed_a).sum()),fold_wins=int((gains>0).sum())))
    group_comparisons=pd.DataFrame(group_comparisons);group_comparisons.to_csv(r.STUDY/'GROUP_VS_ROW_GAIN_SUMMARY.csv',index=False)
    caps=epochs.loc[epochs.cv.eq('Row-CV')&epochs.best_at_cap,['outer_seed','inner_fold','arm','best_epoch']]
    run_caps=epochs.loc[epochs.cv.eq('Row-CV')&epochs.training_hit_cap]
    caps.to_csv(r.STUDY/'BEST_EPOCH_CAP_CASES.csv',index=False)
    tail_negative=int((tail_pairs.tail_gain_percent<0).sum())
    gain_text='；'.join(f"{c}: mean paired-fold gain {overall.loc[c,'mean_paired_fold_gain_percent']:+.3f}%，gain of means {overall.loc[c,'gain_of_means_percent']:+.3f}%，seed wins {int(overall.loc[c,'seed_wins'])}/5，fold wins {int(overall.loc[c,'fold_wins'])}/25" for c in r.TARGETS)
    if consistent:
        finding='两个柱的平均收益在多数 seed 重复出现，支持 Row-CV 下的内部稳定性信号；稳定性仍须结合每柱端点、tail 和最大误差负结果理解。'
    elif positive:
        finding='两个柱的整体平均均改善，但 seed/折和误差剖面存在不一致；当前证据支持有限的内部收益，不能称为两柱所有输出均稳定改善。'
    else:
        finding='至少一个目标柱缺乏跨 seed 一致的收益，当前不支持将 A2 概括为稳定优于 A1 的统一方法；必须分别保留两柱结果。'
    if positive:
        candidate='A2 可保留为后续开发性配对 refit 的候选，同时保留 A1 对照；当前不能将 A2 认定为唯一最终模型。建议在用户另行确认后，按已冻结每 seed/方法 epoch 中位数做 A1/A2 FULL-data refit，先保存并冻结全部盲预测，再另行进行一次 DEVELOPMENTAL CONFIRMATION 评分。tail/最大误差及 500 cap 问题须保留，不能通过改 epoch 上限、网络或 selector 消除负结果。'
    else:
        candidate='当前内部证据不支持将 A2 选为覆盖两柱的统一最终模型候选，也不建议以提高预测性能为理由立即 FULL refit。若研究目的是诊断选择机制，可在另行确认后进行固定协议的开发性配对 refit；不得用已曝光外层测试来挑模型或 epoch。'
    inc=int((epoch_comparison.A2_median_increase>0).sum())
    greater=int((epoch_comparison.row_A2_minus_A1>0).sum())
    replay=frames['CHECKPOINT_REPLAY.csv']
    target_history=frames['TRAINING_HISTORY.csv']
    curves='\n'.join(f'- [seed {seed} 全部五折学习曲线](learning_curves/row_cv_seed_{seed}.png)' for seed in r.SEEDS)
    report=f'''# A1/A2 Row-aligned Phase 2：完整五 seed 内部验证

日期：2026-10-13（Asia/Shanghai）。**已完成50次拟合：复用Phase1的10次，新增四个seed的40次。** 本研究为开发性方法诊断；Phase1及历史Row外层结果已被观察，不是独立外部确认。没有读取外层validation/test端点，没有FULL refit、新外层评分、优化器干预或超参数搜索。原Phase1报告和来源记录见 [phase1_archive/REPORT.md](phase1_archive/REPORT.md)。

## 五 seed 主要结论

{gain_text}。

{finding}

候选与稳定性的判断为描述性研究建议，依据已披露的两柱均值、seed/fold wins及误差剖面；不是新增预注册通过gate，也不用于调整训练规则或选择已曝光外层测试上的最佳模型。

每折同一个验证集合用于扫描并选择epoch，再报告该最优checkpoint的验证指标，因此数值包含模型选择的乐观性；这些是内部选择诊断，不是独立泛化误差估计。

A = mean(1−A2_fold/A1_fold)，B = 1−mean(A2)/mean(A1)。下表SD为25折描述性标准差(ddof=1)，另列五个seed均值的描述性SD。fold和seed共享大量数据，不是独立实验；没有独立样本显著性检验或“统计显著”的宣称。

{table(summary,['column','arm','mean_nrmse','fold_sd_nrmse','seed_mean_sd_nrmse','mean_paired_fold_gain_percent','gain_of_means_percent','seed_wins','fold_wins'])}
每个seed的真实结果如下，两个gain定义分别列出；seed win按A2五折均值低于A1判定。

{table(seed_pairs,['column','outer_seed','A1_mean_nrmse','A2_mean_nrmse','mean_paired_fold_gain_percent','gain_of_means_percent','fold_wins','seed_win'])}
## V1/V2、Center/Width 与 tail

下表是25折指标算术均值，RMSE/MAE单位mL；离散程度和每seed数值见 INNER_SUMMARY.csv、SEED_RESULTS.csv。各seed的OOF pooled RMSE先按SSE/记录数汇总，再跨seed平均；不把重复seed中的记录计为额外独立样本。

{table(summary,['column','arm','mean_V1_rmse','mean_V2_rmse','mean_V1_mae','mean_V2_mae','mean_Center_rmse','mean_Width_rmse'])}
五seed pooled RMSE均值的相对收益（正值为A2改善）：

{table(profile_gains,list(profile_gains.columns))}
25g的pooled改善主要来自V1/Center，Width收益较小；40g的Width收益高于Center。需要保留一个输出代价：40g V2的25折平均MAE由10.7715增至10.8180 mL（恶化约0.43%），虽然其RMSE改善。平均RMSE降低不能替代所有输出误差均改善的判断。
每seed尾部均使用该seed outer gradient_train端点80%分位。先合并五折tail SSE/count后开根号，再比较方法；尾部不选择checkpoint。

{table(tail_overall,list(tail_overall.columns))}
全部seed尾部结果，包含Phase1已观察到的25g V1 tail约3.05%退化，不删除、不改阈值。

{table(tail_pairs,['column','outer_seed','endpoint','A1_tail_rmse','A2_tail_rmse','tail_gain_percent','tail_count'])}
共有 {tail_negative}/20 个“seed×柱×endpoint”tail比较出现A2退化。历史2%保护线仅是描述性参照，本研究没有事后引入正式通过gate。最大绝对误差保存在SEED_RESULTS.csv；平均NRMSE改善而最大误差增大的组合见MAX_ERROR_TRADEOFFS.csv，计 {int(max_tradeoffs.mean_improves_but_max_worsens.sum())}/40 个“seed×柱×V1/V2/Center/Width”组合。新增最大误差诊断不参与选择模型或epoch。

以下列出这些均值改善但最大误差退化的具体组合，负gain表示最大绝对误差增大：

{table(tradeoff_cases,['column','outer_seed','endpoint','max_error_gain_percent'])}

## 训练选择机制与收敛边界

{table(epoch_comparison,['outer_seed','A1_group_median','A1_row_median','A2_group_median','A2_row_median','A2_median_increase','group_A2_minus_A1','row_A2_minus_A1','A2_row_cap_folds'])}
A2的中位数在 {inc}/5 个seed高于历史Group-CV；Row-CV中A2中位数大于A1的seed数为 {greater}/5。完整五折best epoch和中位数如下。Group-CV及Row-CV的验证身份不同，fold编号不能跨协议当作身份配对，不比较其绝对NRMSE来判断协议优劣。

{table(selections,['cv','outer_seed','arm','five_best_epochs','median_epoch','best_at_cap','training_hit_cap'])}
Row-CV共 {len(caps)}/50 次拟合的best epoch选在500上限，其中A1 {int((caps.arm=='A1').sum())}/25、A2 {int((caps.arm=='A2').sum())}/25。此为收敛边界/右截断，不能声称这些折已充分收敛；没有自行延长上限。实际optimizer steps完整记录在FIT_AUDIT.csv与TRAINING_HISTORY.csv，所有context为 {','.join(map(str,sorted(frames['FIT_AUDIT.csv'].steps_per_epoch.unique())))} step/epoch。4g每epoch只抽相同配对数量，并非遍历3330条源数据。

另有 {len(run_caps)}/50 次训练实际运行到500（A1 {int((run_caps.arm=='A1').sum())}/25、A2 {int((run_caps.arm=='A2').sum())}/25）。即使best点位于500之前，部分晚期最优的曲线也受最大训练预算约束；“训练到上限”与“最佳点在上限”分别记录，不能混为同一计数。

以下只比较各CV内部的A2/A1相对收益与训练选择；原Group-CV冻结结果没有重新生成。

{table(group_comparisons,list(group_comparisons.columns))}
新Row-CV的五seed总体配对相对收益并未高于历史Group-CV；观察到的是验证任务和训练选择时长改变，而非已经证明某协议更优或已经修复旧外层失败。A2虽在五seed均比Group-CV选得更晚，在新Row-CV中仍有4/5 seed的中位数小于A1。
## 图表与完整曲线

![五seed配对收益](five_seed_row_cv_paired_gains.png)

![Group和Row的epoch选择](group_vs_row_cv_best_epochs.png)

![全部seed的误差剖面](a2_center_width_tail_diagnostics.png)

{curves}

TRAINING_HISTORY.csv保留每折每个实际epoch。train raw loss使用混合任务训练BN和配对抽样，validation loss使用eval BN固定验证集，不能视为完全同分布的损失估计；非有限预测、梯度和目标验证loss检查全部通过。晚期验证波动/回升与达到500上限均如实保留，不通过额外调参解释或消除。

## 实现、数据边界与下一步

Phase1原runtime {preservation['runtime_files']} 个文件和 {preservation['archived_files']} 个归档根文件逐字节不变；历史研究 {preservation['historical_files']} 个文件SHA256全部不变。每seed十次拟合均通过五折身份覆盖、A1/A2初始化/训练/验证身份/尺度/batch draw配对、最早joint argmin、真实step计数、冻结参数和BN状态检查。{len(replay)}个目标上下文、{int(replay.rows.sum())}条保存预测的零标签重放最大绝对差为 {replay.max_abs_difference_ml.max():.3g} mL。

Phase2仅解析源3330条source_train标签；源validation只用零标签输入检查有限性，不报告其误差或loss。此诊断权限收紧由用户明确要求，原优化代码仅跳过源验证标签诊断；三步合成测试中A1/A2权重、目标历史和best epoch逐位一致。Phase1已有源验证指标保留为历史数据，并明确其只属于Phase1。目标outer validation/test不构建训练/验证图，其真实端点未解析；feature读取全部使用显式白名单，排除t1/t2与verified times。INPUT_OVERLAP_AUDIT.csv/INPUT_DUPLICATE_COUNTS.csv披露不同ID相同可见输入和跨折重合；相同输入不是sample_id泄漏。物理独立批次/重复测量元数据仍未知，结论仅支持记录级Row插值。

{candidate}

不论是否后续refit，已反复曝光的Row外层测试不能提供独立确认；可靠泛化证据仍需未曝光独立数据。**Phase2在此停止，等待下一步确认。**

本阶段没有外层新评分，因此不能推断原40g Row外层退化已经修复；更长的内层epoch或较低的内部误差本身不构成这一结论。
'''
    (r.STUDY/'REPORT.md').write_text(report)
    audit=f'''# Implementation audit：完整五seed Phase 2

## 冻结与范围

PHASE2_PROTOCOL.json引用原冻结protocol.json和phase1_archive，并记录Phase1与历史Row外层结果已曝光。科学设置未变：sample_id排序、KFold5/shuffle/seed42、同一4g源checkpoint、原A1/A2结构、FiLM16与blocks3/4、Adam原LR/weight_decay、raw_quantile、混合late BN、500/patience80、等权两柱Combined NRMSE、最早argmin与五折中位数。新代码不含refit、外层score、PCGrad或搜索入口。

唯一执行诊断差异是遵照本轮用户权限停止解析源validation标签。test_phase2.py以AST比较确认fit代码仅新增source-validation诊断跳过分支；A1/A2三步合成更新、目标历史和best epoch与Phase1逐位一致。source_train人口及所有目标训练/选择完全不变。数据loader使源validation truth为NaN、图y为零，防止误将占位值记为观察指标；源validation误差/loss在Phase2缺失并明确标记。

## 逐seed核验

phase2/seed_*/COMPLETE_AUDIT.json保存每seed完整检查和CSV哈希。共50模型、100目标记录；Phase1十模型从原receipt复用，40新增模型均在完成后审计。ROW_SPLIT_MANIFEST包含全部真实折身份，每目标ID在本seed五折中恰一次验证；源train和diagnostic身份固定。对每折比较两个arm的训练/验证hash、数量、归一化尺度、训练seed、batchsize和steps，全部一致；共同epoch的batch_draws_sha256完全相同，共同初始状态digest一致。

独立读取best.pt核对best_epoch、源authority、冻结参数和early BN buffers、late BN num_batches_tracked增量；best是joint历史最早argmin，optimizer_steps是实际累加，每个checkpoint最佳步数与best_epoch×steps一致。重算获准inner验证指标与保存CSV、best history一致；零标签模型重放 {len(replay)}个上下文、{int(replay.rows.sum())}条预测，最大差 {replay.max_abs_difference_ml.max():.3g} mL。目标训练/验证预测、训练梯度、train和target validation损失均有限。

## 权限、重复输入和保存

PHASE2_LABEL_AUTHORIZATION固定每seed gradient_train与源source_train ID；每新fit的label_access.json记录实际读取请求hash。源validation/test、目标outer validation/test真实标签均无请求。只按特征白名单解析CSV，原features文件冗余时间列未解析。输入digest含实际图与条件张量，排除y/身份；不同ID重复输入保留当前协议并统计，没有自行去重或重新分组。跨柱task ID不同意味着骨干输入重合不等于完整模型输入相同。缺少明确物理实验批次/replicate元数据，不将共享记录的五seed或25折当独立物理重复。

原Phase1 {preservation['runtime_files']}个runtime文件、{preservation['archived_files']}个归档根文件，以及历史研究{preservation['historical_files']}个文件哈希保持不变。原Phase1 protocol.json、训练/测试代码与provenance不改动；PROTOCOL.md按用户要求追加阶段授权/曝光/诊断边界披露。根目录聚合结果更新为50拟合，Phase1原表保存在phase1_archive，原单seed负结果保留。

## 限制与执行记录

没有发现训练协议漂移或非有限预测/错误参数范围/BN故障。19项执行与权限测试通过（TEST_RESULTS_PHASE2.txt），统计定义的独立算术检查见REPORTING_TEST_RESULTS.txt。每seed的收敛上限和所有tail/max负结果见REPORT。没有提高500上限或改变epoch选择。完整曲线保存在learning_curves/，全部历史在TRAINING_HISTORY.csv。

复现已完成检查/汇总：fish Python运行summarize_phase2.py；Phase2运行入口run_phase2.py只接受四个新seed并按receipt复用已完成fit，拒绝拟合pilot。该入口没有其他科学动作。报告生成不训练模型；诊断重放不会读取外层标签。**已完成Phase2并停止，未执行FULL-data refit或outer评分。**
'''
    (r.STUDY/'IMPLEMENTATION_AUDIT.md').write_text(audit)
    (r.STUDY/'README.md').write_text('''# A1/A2 Row-aligned v1：Phase 2 完成

完整五seed内部Row-CV：50模型（Phase1复用10，Phase2新增40）。没有FULL refit、outer scoring或额外方法。研究是开发性诊断；Phase1与历史Row外层结果已经曝光。

- PROTOCOL.md：原设计与Phase2授权/曝光/标签权限补充。
- PHASE2_PROTOCOL.json、PHASE2_EXECUTION_PROVENANCE.json：本阶段冻结authority，引用原protocol.json。
- REPORT.md、IMPLEMENTATION_AUDIT.md：完整结果、限制及审计。
- INNER_RESULTS.csv、PAIRED_COMPARISONS.csv、TRAINING_HISTORY.csv：全50模型聚合。
- INNER_SUMMARY.csv、SEED_RESULTS.csv、SEED_PAIRED_COMPARISONS.csv：两种gain、描述性SD、每seed结果。
- EPOCH_SELECTION.csv：仅新Row-CV的10项未来候选epoch，refit_executed=False。
- GROUP_VS_ROW_EPOCH_SELECTION.csv、EPOCH_SELECTION_COMPARISON.csv：历史/新CV选择诊断，不能混作主实验预算。
- TAIL_PAIRED_COMPARISONS.csv、MAX_ERROR_TRADEOFFS.csv：全部负结果与最大误差诊断。
- five_seed_row_cv_paired_gains.png、group_vs_row_cv_best_epochs.png、a2_center_width_tail_diagnostics.png：三张分析图，北大红/蓝灰配色。
- learning_curves/：各seed全部五折完整训练/验证曲线。
- phase1_archive/：按字节保留原43个根产物及110个runtime文件的来源hash；原pilot runtime未改动。
- phase2/seed_*/COMPLETE_AUDIT.json：每seed十模型的完整性与重放收据。

在仓库根目录使用已验证fish Python。仅重新汇总与核验（不训练）：

```sh
KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python studies/transfer/column_conditioned_row_aligned_v1/summarize_phase2.py
```

run_phase2.py的完整入口按receipt复用已完成的四个seed，拒绝拟合pilot；没有refit或outer score路径。旧run_phase1.py/code/protocol/provenance作为冻结来源保留，不重跑Phase1。原protocol.json中PHASE_1_ONLY的文字属于原冻结记录，Phase2扩展由PHASE2_PROTOCOL及PROTOCOL.md补充定义。

本轮源标签权限仅source_train，源validation指标仅在已有Phase1记录中保留；Phase2不解析其标签、不报告零占位损失。目标仅各seed的gradient_train标签，features严格usecols白名单。19项执行/边界测试及3项独立统计算术测试通过。

Phase2已停止。后续FULL refit与外层评分均需用户另行确认；已曝光外层测试只能用于DEVELOPMENTAL CONFIRMATION，独立泛化需要未曝光数据。
''')
    r.save(r.STUDY/'PHASE2_DECISION.json',{'internal_candidate_supported':positive,'majority_seed_consistency':consistent,
        'A2_median_increased_seeds':inc,'A2_row_median_exceeds_A1_seeds':greater,'best_epoch_cap_fits':len(caps),
        'tail_deteriorating_seed_column_endpoint_cases':tail_negative,'mean_improved_max_worsened_cases':int(max_tradeoffs.mean_improves_but_max_worsens.sum()),
        'interpretation':'developmental internal diagnosis; no independent external efficacy confirmation',
        'next_action_requires_confirmation':True,'full_refit_executed':False,'outer_scoring_executed':False})

if __name__=='__main__':
    summarize()
