"""Describe frozen refits and numerical checks without observed outer endpoints."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_phase3a as s


def table(frame):
    def cell(value):
        return str(value).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |',
                      '| '+' | '.join(['---']*len(frame.columns))+' |',
                      *['| '+' | '.join(cell(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)]])


def summarize():
    verified=s.verify_freeze()
    freeze=s.read_json(s.STUDY/'PREDICTION_FREEZE_MANIFEST.json')
    manifest=pd.read_csv(s.STUDY/'FINAL_REFIT_MANIFEST.csv')
    history=pd.read_csv(s.STUDY/'FINAL_TRAINING_HISTORY.csv')
    audit=s.read_json(s.STUDY/'PHASE3A_INPUT_AUDIT.json')
    baseline=s.read_json(s.STUDY/'PHASE3A_BASELINE_COMPATIBILITY.json')
    rows,losses,shape_checks,crossing_records=[],[],[],[]
    for seed in s.SEEDS:
        pair=manifest.loc[manifest.outer_seed.eq(seed)].set_index('arm')
        rows.append({'seed':seed,'A1 epochs':int(pair.loc['A1','fixed_epochs']),'A1 steps':int(pair.loc['A1','optimizer_steps']),
                     'A2 epochs':int(pair.loc['A2','fixed_epochs']),'A2 steps':int(pair.loc['A2','optimizer_steps']),
                     'source_train':int(pair.loc['A1','4g_count']),'25g train':int(pair.loc['A1','25g_count']),
                     '40g train':int(pair.loc['A1','40g_count'])})
        for arm in s.old.ARMS:
            run=s.FINAL/f'seed_{seed}/{arm}'
            fit=s.read_json(run/'fit.json')
            access=s.read_json(run/'label_access.json')
            ids=pd.read_csv(run/'training_identities.csv',dtype={'sample_id':str})
            s.require(len(access)==3,'label access receipt count mismatch')
            for item in access:
                c=item['column'];ordered=ids.loc[ids.column.eq(c),'sample_id'].tolist()
                s.require(item['authorized_ids']==ordered and item['count']==len(ordered)
                          and item['ids_sha256']==s.old.digest(sorted(ordered))
                          and item['outer_response_cells_parsed']==0
                          and item['source_validation_response_cells_parsed']==0,'label permission receipt mismatch')
            part=history.loc[history.outer_seed.eq(seed)&history.arm.eq(arm)].sort_values('epoch')
            for c in s.old.TASKS:
                values=part[f'train_loss_{c}'].to_numpy(float)
                s.require(np.isfinite(values).all(),'nonfinite training history')
                losses.append({'outer_seed':seed,'arm':arm,'column':c,'first_epoch_loss':values[0],
                               'final_epoch_loss':values[-1],'peak_epoch_loss':values.max(),
                               'peak_epoch':int(part.iloc[int(np.argmax(values))].epoch),'last_20_mean':values[-20:].mean(),
                               'last_20_sd':values[-20:].std(ddof=1),'all_finite':True})
            for name,check in fit['prediction_checks'].items():
                c,role=name.split('_')[:2]
                shape_checks.append({'outer_seed':seed,'arm':arm,'column':c,'role':role,'rows':check['rows'],
                                     'V2_below_V1_q10':check['V2_below_V1_by_quantile'][0],
                                     'V2_below_V1_q50':check['V2_below_V1_by_quantile'][1],
                                     'V2_below_V1_q90':check['V2_below_V1_by_quantile'][2],
                                     'V1_quantile_crossing_rows':check['V1_quantile_crossing_rows'],
                                     'V2_quantile_crossing_rows':check['V2_quantile_crossing_rows'],
                                     'all_outputs_finite':check['all_six_outputs_finite']})
                if check['V1_quantile_crossing_rows'] or check['V2_quantile_crossing_rows']:
                    blind=pd.read_csv(run/name,usecols=['sample_id',*s.old.QUANTILES],dtype={'sample_id':str})
                    for endpoint,columns in [('V1',s.old.QUANTILES[:3]),('V2',s.old.QUANTILES[3:])]:
                        bad=np.any(np.diff(blind[columns].to_numpy(),axis=1)<0,axis=1)
                        crossing_records.extend({'outer_seed':seed,'arm':arm,'column':c,'role':role,'endpoint':endpoint,
                                                 **item} for item in blind.loc[bad,['sample_id',*columns]].to_dict('records'))
    loss_frame,shape_frame=pd.DataFrame(losses),pd.DataFrame(shape_checks)
    loss_frame.to_csv(s.STUDY/'FINAL_TRAINING_DIAGNOSTICS.csv',index=False)
    shape_frame.to_csv(s.STUDY/'BLIND_PREDICTION_SHAPE_AUDIT.csv',index=False)
    pd.DataFrame(crossing_records).to_csv(s.STUDY/'BLIND_QUANTILE_CROSSINGS.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(5,3,figsize=(13,13),sharex=False,constrained_layout=True)
    colors={'A1':'#607D8B','A2':'#8C0000'}
    for i,seed in enumerate(s.SEEDS):
        for j,c in enumerate(s.old.TASKS):
            ax=axes[i,j]
            for arm in s.old.ARMS:
                part=history.loc[history.outer_seed.eq(seed)&history.arm.eq(arm)]
                ax.plot(part.epoch,part[f'train_loss_{c}'],color=colors[arm],lw=1,label=arm)
            ax.set(title=f'{seed} | {c}',xlabel='Fixed-budget epoch',ylabel='Training raw quantile loss')
            ax.grid(alpha=.18);ax.legend(frameon=False)
    fig.suptitle('FULL Row refits: training losses only; no outer response access',fontsize=14)
    fig.savefig(s.STUDY/'final_refit_training_curves.png',dpi=170)
    plt.close(fig)
    crossings=int(shape_frame[['V1_quantile_crossing_rows','V2_quantile_crossing_rows']].to_numpy().sum())
    below=int(shape_frame[['V2_below_V1_q10','V2_below_V1_q50','V2_below_V1_q90']].to_numpy().sum())
    mismatch_text='\n'.join(f"- `{item['path']}`：冻结期望 `{item['expected_sha256']}`，当前 `{item['actual_sha256']}`。" for item in baseline['frozen_provenance_mismatches'])
    text=f'''# Phase 3A：FULL-data 重训与盲预测冻结

日期：2026-10-14（Asia/Shanghai）。**10/10 次最终重训完成，40/40 个盲预测文件已冻结。未解析任何外层 validation/test 真实响应，未调用 outer score，未进入 Phase 3B。**

## 执行前核验与固定预算

只读复核全部50次Row-CV拟合、100个目标柱结果、每seed/arm五个joint best_epoch、原最早argmin、checkpoint、BN buffers、冻结参数、真实步数和确定性折身份。原Phase1的43个归档根文件、110个runtime文件，历史研究1069文件，以及Phase2的56个报告产物哈希均保持不变。详见 PHASE3A_INPUT_AUDIT.json；Phase1/2原报告仍保留，不回写。

8项Phase3A测试通过（TEST_RESULTS_PHASE3A.txt）：A1/A2三步合成数据更新与原fixed_epochs refit逐位一致，特征白名单、train-only标签请求、完全无标签重放、FULL身份授权、非法joint预算及预测schema/有限性守卫均通过。首次测试中修正了pandas测试夹具的浮点类型声明；发生在真实训练前，不改变任何训练设置。

每个arm使用自己五折joint best_epoch的round-half-up中位数，从同一个4g源checkpoint重新构建。实际optimizer.step逐次累加；以下全为1 step/epoch。两个arm的共同初始张量与共同epoch样本身份调度完全一致，A1/A2允许不同预算。这测量完整Row-CV选择加FULL refit流程，不是equal-step FiLM模块的净效应。

{table(pd.DataFrame(rows))}

全部五折epoch、最终预算、实际步数、每柱标签身份hash、模型文件SHA256及包含BN buffers的状态张量SHA256见 FINAL_REFIT_MANIFEST.csv、各run/fit.json。总计 **{int(manifest.optimizer_steps.sum())} optimizer steps**，FINAL_TRAINING_HISTORY.csv含{len(history)}个实际epoch。

## 数据与模型边界

FULL严格等于原Row split中全部合格目标gradient_train，包含原pool，外层validation/test均排除；没有将完整原始CSV作为监督人口。4g仅解析固定3330条source_train标签。目标仍为408/456条过滤人口，逐seed实际train数量见上表。label_access.json保存实际请求身份与数量，training_identities.csv逐arm保存全部标签身份；源validation标签也未解析。

输入仅解析原七列白名单，排除t1/t2、verified times和其他冗余终点。标签reader先读取sample_id，然后通过skiprows仅解析获准训练行的V1_ml/V2_ml。外层输入图的y全为零，独立truth数组全为NaN；checkpoint重放loader完全不读取标签。文件SHA256需要读取原CSV字节，这不等于解析外层响应；本报告的“未访问终点”指未解析、使用或评分任何外层真实终点。

直接复用冻结原模型、Adam、pretrained/new学习率、weight decay、raw_quantile损失、mixed late BN、frozen-prefix cache、balanced-task batcher、FiLM16维与block3/4作用位置。固定refit不接收validation，不运行early stopping；patience80仍属于冻结内层选择，不能用它提前停止最终预算。未新增PCGrad、calibration head、搜索或equal-step候选。

## 完整状态、盲预测与冻结验证

最终checkpoint位于 `runtime/final_row_aligned/seed_<seed>/<A1|A2>/final.pt`；每个checkpoint完整保存model_state_dict及BN running_mean/running_var/num_batches_tracked。各run分别保存25g/40g的validation/test六输出盲预测CSV，没有真实响应列。源checkpoint SHA256为 `{s.old.SOURCE_SHA}`，原split SHA256为 `{s.old.SPLIT_SHA}`。

全部40个文件的列顺序、身份顺序、记录数、有限性和原输出截断均通过检查，共 **{freeze['prediction_rows']} 条预测记录**（两个arm、多seed重复身份，非独立实验数）。十个checkpoint重新加载后在**零标签输入**上重放40个上下文，最大绝对差 **{freeze['replay_max_abs_difference_ml']:.3g} mL**；最终状态tensor digest、文件digest、固定预算和BN步数完全一致。见 FINAL_CHECKPOINT_REPLAY.csv。

所有最终权重、预测、训练身份、label receipt及汇总文件在 PREDICTION_FREEZE_MANIFEST.json 中hash锁定；清单通过immutable write写入，冻结文件设置只读。再次读取与hash验证通过；完成后的入口只验证冻结，不重复训练或改写预测。任何后续不一致须停止，不能根据评分修补已有文件。

## 数值与收敛限制

没有非有限loss/gradient/预测、checkpoint不一致、冻结参数或BN故障。训练损失完整曲线见下图，首轮/末轮和末20轮均值/SD见 FINAL_TRAINING_DIAGNOSTICS.csv。固定预算全部小于500，没有提高上限；固定预算完成不等于已证明充分收敛。保留pilot A2内层一折best_epoch=500的右截断边界，不据此延长最终预算。

训练loss并非所有任务都单调下降：A1五个seed的4g source训练loss均先升后降，峰值约432–468，位于46–62 epoch，最终约24–58；详细峰值/epoch已加入训练诊断表。源任务与目标任务处于同一等权原始loss及混合BN优化中，这里只描述观察，不据此改变权重、BN或训练预算，也不利用外层响应诊断其原因。

![FULL refit train curves](final_refit_training_curves.png)

原输出实现仅有ReLU和eval clamp[0,1e8]，并不保证每行分位数单调或V2≥V1；本阶段没有新增排序、投影或校准。盲输出诊断中，V1/V2分位数交叉计 **{crossings} 个端点-记录组合**，V2低于V1计 **{below} 个分位点-记录组合**；重复seed/arm记录分别计数，明细见 BLIND_PREDICTION_SHAPE_AUDIT.csv。这些是已有输出结构的可见属性，不涉及读取真实响应，也不参与模型或epoch选择。

三条交叉均发生于seed1425370602的25g test V1：A1一条，A2两条，都是q50大于q90。逐条身份和原始预测见 BLIND_QUANTILE_CROSSINGS.csv。保留这些负诊断，不对冻结预测排序或修补。

## 后续评价与基线异常

EVALUATION_PLAN.md在拟合前锁定，主要test评价按柱分别报告五seed均值、paired-seed gain与gain of means、seed wins、V1/V2 RMSE/MAE、Center/Width、train-80% tail与最大误差。validation单列次要描述，不与test合并。全部方法必须使用同一FULL gradient_train尺度与同一test身份。FiLM/A1、多任务/简单校准、均值/尾部代价是三个不同问题。

历史基线审查只读元数据、冻结哈希与sample_id，未读响应或评分表。Scale与P0/P1共{baseline['reference_contexts_verified']}个柱×seed×method上下文通过FULL Row身份/冻结检查；后两者使用Group-CV，是不同选择工作流参考，不能归因于单一结构。Affine没有找到兼容FULL冻结证据，paper-style等旧outer-validation选择工作流不进入主要同协议对照。

HIER的目标角色和两柱训练union身份相同，但其冻结provenance存在以下差异，因此**排除HIER直接对照，不修复或重建历史产物**：

{mismatch_text}

审查结论和具体预测hash保存于 PHASE3A_BASELINE_COMPATIBILITY.json，后续须再次核验。此异常不涉及A1/A2被冻结训练依据；A1/A2主比较的Phase3B前置条件已满足。

## 科学解释与停止

历史Row外层及Phase1/2结果已曝光，本研究为开发性方法诊断，未来外层结果不能称独立外部确认。五seed共享历史数据，不当作五次独立物理重复，不做不合理的独立样本显著性检验。Phase2的25g V1 tail平均恶化约1.98%、40g V2 MAE微增及最大误差负结果均继续保留；本阶段盲预测不能判断这些问题是否改善。

**Phase3A完成并停止。** 已满足执行预定A1/A2 Phase3B的技术前置条件，但未执行任何新外层评分，仍等待用户下一步确认；不能据此选择唯一最终arm或宣称旧40g外层退化已修复。
'''
    (s.STUDY/'PHASE3A_REPORT.md').write_text(text)
    names=['PHASE3A_REPORT.md','PREDICTION_FREEZE_MANIFEST.json','FINAL_TRAINING_DIAGNOSTICS.csv','BLIND_PREDICTION_SHAPE_AUDIT.csv','BLIND_QUANTILE_CROSSINGS.csv',
           'final_refit_training_curves.png','PHASE3A_BASELINE_COMPATIBILITY.json','PHASE3A_BASELINE_AUDIT_LOG.txt',
           'TEST_RESULTS_PHASE3A.txt','PHASE3A_EXECUTION_LOG.txt','audit_phase3a_baselines.py','summarize_phase3a.py']
    s.old.write_json(s.STUDY/'PHASE3A_REPORT_MANIFEST.json',{'phase':'PHASE3A_COMPLETE_STOPPED','freeze_verified':verified,
                     'files':{name:s.old.sha(s.STUDY/name) for name in names},'outer_score_executed':False,
                     'baseline_HIER_excluded_due_to_hash_mismatch':bool(baseline['frozen_provenance_mismatches'])},immutable=True)
    print(verified)
    print('Report complete; stopped before Phase 3B.')


if __name__=='__main__':
    summarize()
