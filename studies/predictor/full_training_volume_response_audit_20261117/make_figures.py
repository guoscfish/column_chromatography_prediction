#!/usr/bin/env python3
"""Three scientific figures with fixed metadata pool and explicit exposure scopes."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def md(frame,digits=4):
    def fmt(v):return f'{v:.{digits}f}' if isinstance(v,(float,np.floating)) else str(v)
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(map(fmt,row))+' |' for row in frame.itertuples(index=False,name=None))

def main():
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    full=pd.read_csv(OUT/'formal_response_per_model.csv')
    volume=full[(full.axis=='V/ul')&(full.actual_threshold_ml==.5)&full.exposure.isin(['train_train','validation_validation'])]
    fig,axes=plt.subplots(1,2,figsize=(12.8,4.5))
    for ax,target in zip(axes,['V1','V2']):
        g=volume[volume.target==target].set_index(['row_seed','exposure']).reindex([(s,e) for s in [42,525,1101] for e in ['train_train','validation_validation']])
        x=np.arange(6)
        ax.bar(x-.17,g.observed_abs_mean,.34,color='#90a4b8',label='Observed')
        bars=ax.bar(x+.17,g.predicted_abs_mean,.34,color='#c46720',label='Predicted q50')
        for bar,v in zip(bars,g.predicted_abs_mean):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.03,f'{v:.3f}',ha='center',fontsize=8)
        ax.set_xticks(x,['42\nTrain','42\nValidation','525\nTrain','525\nValidation','1101\nTrain','1101\nValidation'])
        ax.set_ylabel('Mean absolute volume response (mL)');ax.set_title(target);ax.grid(axis='y',alpha=.15)
    axes[0].legend(frameon=False,loc='upper right');fig.suptitle('Qualified Row models: volume responses in their own frozen splits')
    fig.text(.02,.015,'Train pairs: 2771 / 2767 / 2730; validation: 48 / 44 / 48. Related pairs are descriptive, not independent replications.',fontsize=9)
    fig.tight_layout(rect=[0,.055,1,.94]);fig.savefig(figures/'train_validation_volume_response.png',dpi=180);fig.savefig(figures/'train_validation_volume_response.pdf');plt.close(fig)
    matched=pd.read_csv(OUT/'matched_seed_metrics.csv');matched=matched[(matched.actual_threshold_ml==.5)&(matched.axis=='V/ul')]
    cols=['observed_abs_mean','predicted_abs_mean','difference_mae','flat_fraction','direction_accuracy']
    per_formal=matched.groupby(['formal_seed','exposure','method','target'])[cols].mean().reset_index()
    equal=per_formal.groupby(['exposure','method','target'])[cols].mean().reset_index()
    equal.to_csv(OUT/'matched_plot_equal_models.csv',index=False)
    methods=['AL_L525','AL_L1005','AL_L1485','Fully_trained'];colors=['#94a3b8','#537aa0','#457867','#b65728']
    train=equal[equal.exposure=='both_source_train']
    fig,axes=plt.subplots(2,2,figsize=(10.7,7))
    for j,target in enumerate(['V1','V2']):
        g=train[train.target==target].set_index('method').reindex(methods)
        axes[0,j].bar(range(4),g.predicted_abs_mean,color=colors)
        axes[0,j].axhline(float(g.observed_abs_mean.iloc[0]),color='black',ls='--',lw=1,label='Observed mean')
        axes[0,j].set_title(target);axes[0,j].set_ylabel('Absolute response (mL)');axes[0,j].legend(frameon=False)
        axes[1,j].bar(range(4),g.difference_mae,color=colors);axes[1,j].set_ylabel('Difference MAE (mL), lower is better')
        for i,v in enumerate(g.difference_mae):axes[1,j].text(i,v+.025,f'{v:.3f}',ha='center',fontsize=9)
        for ax in axes[:,j]:ax.set_xticks(range(4),['L525','L1005','L1485','Fully trained']);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Exactly matched AL pairs; both endpoints are formal source-train')
    fig.text(.02,.015,'Mean over AL contexts within each formal model, then three formal models equally. Crossed seeds are not randomized paired replications.\nMixed and both-validation strata are reported separately; the latter has only six comparison rows.',fontsize=9)
    fig.tight_layout(rect=[0,.07,1,.95]);fig.savefig(figures/'matched_al_volume_response.png',dpi=180);fig.savefig(figures/'matched_al_volume_response.pdf');plt.close(fig)
    curves=pd.read_csv(OUT/'controlled_volume_curves.csv');pool=json.loads((OUT/'CURVE_POOL.json').read_text())['cases']
    observed=pd.read_csv(OUT/'curve_authorized_observations.csv')
    supported=curves[curves.locally_supported]
    spans=supported.groupby(['case_id','method','model_seed','target']).prediction_ml.agg(lambda a:a.max()-a.min()).reset_index(name='span_ml')
    spans.to_csv(OUT/'curve_response_spans.csv',index=False)
    case_stats=[]
    for case in pool:
        g=spans[spans.case_id==case['case_id']];f=g[g.method=='Fully_trained'].groupby('model_seed').span_ml.mean()
        a=float(g[g.method=='AL_L1005'].span_ml.mean());full_curves=supported[(supported.case_id==case['case_id'])&(supported.method=='Fully_trained')]
        level_spread=float(full_curves.groupby(['volume_ul','target']).prediction_ml.agg(lambda z:z.max()-z.min()).max())
        case_stats.append(dict(case_id=case['case_id'],al_seed=case['al_seed'],molecule=case['molecule'],full_max_endpoint_mean_span_ml=float(f.max()),
            full_min_endpoint_mean_span_ml=float(f.min()),formal_response_span_spread_ml=float(f.max()-f.min()),formal_max_level_spread_ml=level_spread,
            al_L1005_endpoint_mean_span_ml=a,recovery_in_at_least_one_formal=bool(f.max()>=1 and f.max()>=5*a),all_three_formal_flat=bool(f.max()<.1),formal_response_spread_above_1ml=bool(f.max()-f.min()>1)))
    stats=pd.DataFrame(case_stats).sort_values('case_id');stats.to_csv(OUT/'curve_case_metrics.csv',index=False)
    chosen=[];selection=[]
    for category,field in [('Partial recovery','recovery_in_at_least_one_formal'),('Flat responses','all_three_formal_flat'),('Seed disagreement','formal_response_spread_above_1ml')]:
        candidates=stats[stats[field]&~stats.case_id.isin(chosen)]
        qualifies=not candidates.empty
        row=(candidates if qualifies else stats[~stats.case_id.isin(chosen)]).iloc[0]
        chosen.append(row.case_id);selection.append(dict(category=category,case_id=row.case_id,criterion_met=qualifies,
            rule='First hash in the frozen pool satisfying the declared threshold; if absent use first remaining metadata hash and explicitly label absent'))
    dump(OUT/'CURVE_EXAMPLE_SELECTION.json',dict(selections=selection,pool_size=len(pool),post_prediction_selection_within_fixed_pool=True,new_cases_added_after_results=0))
    fig,axes=plt.subplots(2,3,figsize=(13.2,7))
    palette={'AL_L525':'#a7b3c1','AL_L1005':'#416b94','AL_L1485':'#467e67','Full_42':'#d07b27','Full_525':'#b94350','Full_1101':'#765293'}
    for j,selection_item in enumerate(selection):
        case=next(v for v in pool if v['case_id']==selection_item['case_id'])
        for i,target in enumerate(['V1','V2']):
            ax=axes[i,j];g=curves[(curves.case_id==case['case_id'])&(curves.target==target)]
            for (method,seed),v in g.groupby(['method','model_seed']):
                label=f'Full_{seed}' if method=='Fully_trained' else method;color=palette[label]
                ax.plot(v.volume_ul,v.prediction_ml,color=color,ls=':',lw=1,alpha=.7)
                supported_v=v[v.locally_supported];ax.plot(supported_v.volume_ul,supported_v.prediction_ml,color=color,marker='o',ms=3,label=label)
                outside=v[~v.locally_supported];ax.scatter(outside.volume_ul,outside.prediction_ml,color=color,marker='x',s=24)
            truth=observed[(observed.case_id==case['case_id'])&(observed.target==target)]
            ax.scatter(truth.volume_ul,truth.value_ml,color='black',marker='D',s=18,label='Authorized observations',zorder=5)
            ax.set_xlabel('Sample volume (uL)');ax.set_ylabel(target+' retention volume (mL)');ax.grid(alpha=.15)
            if i==0:ax.set_title(selection_item['category']+('' if selection_item['criterion_met'] else ' (criterion absent)')+f"\nAL seed {case['al_seed']}; case {case['case_id']}",fontsize=9)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,fontsize=9)
    fig.suptitle('Metadata-fixed volume curves; model seeds have different training histories')
    fig.text(.015,.09,'Solid circles: inside the recorded series range. Dotted segments / crosses: local extrapolation, not error-scored.\nObserved labels are authorized and already in the corresponding AL H525; no unobserved/test labels queried.',fontsize=9)
    fig.tight_layout(rect=[0,.15,1,.94]);fig.savefig(figures/'controlled_volume_examples.png',dpi=180);fig.savefig(figures/'controlled_volume_examples.pdf');plt.close(fig)
    content='# 受控体积曲线与案例\n\n'+md(stats[['case_id','al_seed','full_max_endpoint_mean_span_ml','full_min_endpoint_mean_span_ml','formal_response_span_spread_ml','al_L1005_endpoint_mean_span_ml','recovery_in_at_least_one_formal','all_three_formal_flat']])
    content+='\n\n15个series在新推理前按metadata hash固定，每AL seed最多3个，全部原观察端点对三个正式模型均非test且不触及隔离ID；没有按预测好坏扩展案例池。源条件只改变V/ul，通过原图构建器同步更新Density×V；没有修改加载溶剂体积。请求网格30/60/90/120/150µL在全域合法范围内，超出该series实际最小/最大体积者另标局部外推，仅作敏感度图，不计算真实误差。\n\n图中从固定池按预定规则选择部分恢复、全三seed平坦、seed响应差异案例，具体选择见`CURVE_EXAMPLE_SELECTION.json`。恢复只指至少一个正式seed，不冒充三seed都恢复；若没有全三seed平坦或其他类型，明确标criterion absent，使用固定池剩余最小hash作展示，不补选外部案例。\n\n'+md(pd.DataFrame(selection))
    content+='\n\n`controlled_volume_curves.csv`保留全部15个案例，不只保留画图的三个。span统计只用局部支持范围内的共同网格点，且区分响应span差异和绝对预测水平差异。真实值仅叠加已授权、并已在对应AL H525中的记录；模型与AL的曝光并不相同，曲线不能证明物理因果、LLM增量价值或泛化误差。'
    (OUT/'MODEL_RESPONSE_EXAMPLES.md').write_text(content+'\n')
    inputs=[OUT/n for n in ['formal_response_per_model.csv','matched_seed_metrics.csv','controlled_volume_curves.csv','CURVE_POOL.json','curve_authorized_observations.csv']]
    dump(OUT/'FIGURE_MANIFEST.json',dict(sources=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in inputs+[Path(__file__)]],figures=3,new_fits=0,API_calls=0))

if __name__=='__main__':main()
