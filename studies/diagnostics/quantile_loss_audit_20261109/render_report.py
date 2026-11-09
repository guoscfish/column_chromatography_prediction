"""Render one scientific report from measured aggregate CSVs and frozen evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
BASE='16eca74769f664bc378a58ccbf380a806723f587'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(folder,repository):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(folder);repository=Path(repository)
    components=pd.read_csv(folder/'loss_components.csv');competition=pd.read_csv(folder/'gradient_competition.csv');quantiles=pd.read_csv(folder/'quantile_metrics.csv');signals=pd.read_csv(folder/'signal_diagnostics.csv');incremental=pd.read_csv(folder/'incremental.csv');summary=pd.read_csv(folder/'seed_summary.csv')
    seeds=sorted(components.seed.unique());derived=[]
    for seed in seeds:
        losses=components[(components.seed==seed)&(components.component=='total')&(components.parameter_group=='backbone')].set_index('target').mean_loss
        train=quantiles[(quantiles.seed==seed)&(quantiles.split=='train')].set_index('target');valid=quantiles[(quantiles.seed==seed)&(quantiles.split=='validation')].set_index('target')
        norm=train.normalized_q50_mse
        row={'status':'NEW_OFFLINE_MEASURED','seed':seed,'budget':333,'data_role':'row','V1_raw_total_loss':losses.V1,'V2_raw_total_loss':losses.V2,'V2_raw_loss_share':losses.V2/losses.sum(),'V2_normalized_train_squared_error_share':norm.V2/norm.sum()}
        for target in ['V1','V2']:
            row.update({target+'_training_sigma':train.loc[target,'training_sigma'],target+'_validation_rmse':valid.loc[target,'q50_rmse'],target+'_validation_nrmse':valid.loc[target,'normalized_q50_rmse'],target+'_normalized_training_mse':norm[target]})
        row['validation_combined_nrmse']=np.sqrt(.5*sum(valid.loc[t,'normalized_q50_rmse']**2 for t in ['V1','V2']))
        derived.append(row)
    derived=pd.DataFrame(derived);derived.to_csv(folder/'loss_scale_contributions.csv',index=False)
    evidence_paths=['src/qgeognn_al/training/predictor.py','src/qgeognn_al/models/qgeognn_v2.py','src/qgeognn_al/uncertainty/quantiles.py','studies/predictor/final_4g_qualification/README.md','experiments/e0_3c_loss_controls/README.md','studies/predictor/performance_regression_audit/README.md','src/qgeognn_al/evaluation/point.py']
    evidence={};blobs={}
    for path in evidence_paths:
        body=subprocess.check_output(['git','--no-optional-locks','-C',str(repository),'show',BASE+':'+path]);evidence[path]={'commit':BASE,'sha256':hashlib.sha256(body).hexdigest()};blobs[path]=body.decode()
    historic=[]
    section=blobs['studies/predictor/final_4g_qualification/README.md'].split('## Test aggregate across three seeds',1)[1]
    for line in section.splitlines():
        if line.startswith('| row |') or line.startswith('| compound |'):
            v=[x.strip() for x in line.strip('|').split('|')]
            historic.append({'status':'HISTORICAL_MEASURED','evidence':'final_4g_qualification QUANTILE_AUDIT','data_role':v[0],'target':v[1],'split':'historically_disclosed_test','seeds':'42;525;1101','budget':'full_source_training','any_crossing_mean':float(v[2]),'central80_coverage_mean':float(v[4]),'width_error_spearman_mean':float(v[8]),'top20_enrichment_mean':float(v[10]),'comparison_limit':'Different seed cohort, full training budget, split role and preprocessing scale population; not current AL round results'})
    pd.DataFrame(historic).to_csv(folder/'historical_reuse.csv',index=False)
    statuses=[{'task':'A0 qualification quantile rates/coverage/enrichment','status':'HISTORICAL_MEASURED','scope':'3 seeds x row/compound qualification; historically disclosed test','reason':'Reuse published aggregate; no repeat forward or test-label access'},
              {'task':'A0 8g loss controls','status':'HISTORICAL_MEASURED','scope':'4g to 8g transfer fine tuning','reason':'Not controlled evidence about 4g active learning loss weights'},
              {'task':'A0 R0/R1 ladder','status':'HISTORICAL_MEASURED','scope':'Several training knobs changed jointly','reason':'Cannot isolate 1:0.5 versus 1:1 causal effect in 4g AL'},
              {'task':'A1 checkpoint loss and grouped gradients','status':'NEW_OFFLINE_MEASURED','scope':'5 row seeds at L333, authorized L0','reason':'Eval-mode frozen-state derivatives, no optimizer update'},
              {'task':'A1 actual Adam-update competition and training-trajectory dominance','status':'NOT_MEASURED','scope':'Optimizer state and full historical training trajectories','reason':'Checkpoint gradient geometry is not actual optimizer history; no retraining authorized'},
              {'task':'A2 raw quantiles and separate inference sorting','status':'NEW_OFFLINE_MEASURED','scope':'5 row seeds L333 train and validation','reason':'No sorting before raw crossing/coverage measurement'},
              {'task':'A3 width error ranking and feasible-signal controls','status':'NEW_OFFLINE_MEASURED','scope':'5 row seeds L333 validation; grouped fixed-fold ridge','reason':'Controls both q50, latent distance, full-network q50 gradient norms and conditions'},
              {'task':'A3 same-seed K3 ensemble control','status':'NOT_MEASURED','scope':'No matching same-seed K3 ensemble frozen outputs in selected snapshot','reason':'Different outer seeds must not be pooled to fabricate an ensemble'},
              {'task':'A1-A3 budgets L525/L1005/L1485','status':'NOT_MEASURED','scope':'Current five AL seeds at later budgets','reason':'No additional role-verified immutable snapshot prepared; no new training for symmetry'},
              {'task':'A1-A3 compound AL diagnosis','status':'NOT_MEASURED','scope':'Current AL compound cohort','reason':'Selected current AL protocol is row; historical compound qualification is only boundary evidence'},
              {'task':'U width/crossing without labels','status':'NOT_MEASURED','scope':'Current candidate U','reason':'Optional descriptive branch not needed for current training/validation gap; no U truth accessed'},
              {'task':'Quantile width improves learning efficiency/AULC','status':'HYPOTHESIS','scope':'Future preregistered matched-seed AL experiment','reason':'Offline correlation or enrichment cannot establish AULC gains'},
              {'task':'Changing target loss weight resolves gradient imbalance','status':'HYPOTHESIS','scope':'Future controlled 4g weight study','reason':'No weight changes or causal experiment performed'}]
    pd.DataFrame(statuses).to_csv(folder/'MEASUREMENT_STATUS.csv',index=False)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    x=np.arange(len(seeds));colors=['#147d92','#cc6b3f']
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    axes[0,0].bar(x-.17,derived.V1_raw_total_loss,.34,label='V1',color=colors[0]);axes[0,0].bar(x+.17,derived.V2_raw_total_loss,.34,label='V2',color=colors[1]);axes[0,0].set_title('Raw frozen-checkpoint training loss');axes[0,0].set_ylabel('Mean target loss');axes[0,0].legend()
    axes[0,1].bar(x,derived.V2_raw_loss_share,.35,label='V2 raw loss share',color=colors[1]);axes[0,1].scatter(x,derived.V2_normalized_train_squared_error_share,label='V2 normalized MSE share',color=colors[0],zorder=3);axes[0,1].set_ylim(0,1);axes[0,1].set_title('Different quantities: raw loss vs normalized error');axes[0,1].legend(fontsize=8)
    for group,color in zip(['backbone','condition_branch'],colors):
        rows=competition[competition.parameter_group==group].set_index('seed').loc[seeds]
        axes[1,0].plot(x,rows.V2_to_V1_gradient_norm,'o-',label=group,color=color)
        axes[1,1].plot(x,rows.cosine,'o-',label=group,color=color)
    axes[1,0].axhline(1,color='gray',ls='--');axes[1,0].set_title('V2/V1 gradient L2 ratio');axes[1,0].legend(fontsize=8)
    axes[1,1].axhline(0,color='gray',ls='--');axes[1,1].set_ylim(-1,1);axes[1,1].set_title('V1/V2 shared-parameter gradient cosine')
    for ax in axes.flat:ax.set_xticks(x,[str(s) for s in seeds]);ax.set_xlabel('Outer seed');ax.grid(axis='y',alpha=.15)
    fig.suptitle('L333, frozen eval mode; no optimizer steps or training-trajectory inference');fig.tight_layout();fig.savefig(folder/'loss_and_gradients.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7));validation=quantiles[quantiles.split=='validation']
    for target,color,offset in zip(['V1','V2'],colors,[-.15,.15]):
        rows=validation[validation.target==target].set_index('seed').loc[seeds]
        axes[0,0].plot(x,rows.central80_coverage,'o-',label=target,color=color)
        axes[0,1].plot(x,rows.any_crossing_rate,'o-',label=target,color=color)
        rank=signals[(signals.record_type=='error_ranking')&(signals.target==target)&(signals.signal=='signed_width')].set_index('seed').loc[seeds]
        axes[1,0].plot(x,rank.spearman,'o-',label=target,color=color)
        inc=incremental[incremental.target==target].set_index('seed').loc[seeds]
        axes[1,1].bar(x+offset,inc.rmse_delta_with_minus_base,.3,label=target,color=color)
    axes[0,0].axhline(.8,ls='--',color='gray',label='Nominal 80%');axes[0,0].set_ylim(0,1);axes[0,0].set_title('Raw q10/q90 validation coverage');axes[0,0].legend(fontsize=8)
    axes[0,1].set_title('Raw any-quantile crossing rate');axes[0,1].legend()
    axes[1,0].set_title('Signed width / absolute q50 error Spearman');axes[1,0].axhline(0,color='gray',ls='--');axes[1,0].legend()
    axes[1,1].axhline(0,color='gray',ls='--');axes[1,1].set_title('Width addition: OOF RMSE delta (lower is better)');axes[1,1].legend()
    for ax in axes.flat:ax.set_xticks(x,[str(s) for s in seeds]);ax.set_xlabel('Outer seed');ax.grid(axis='y',alpha=.15)
    fig.suptitle('Current five-seed row validation, L333; retrospective diagnosis only');fig.tight_layout();fig.savefig(folder/'quantiles_and_incremental.png');plt.close(fig)
    def md(frame,cols,formatters=None):
        text=['| '+' | '.join(cols)+' |','| '+' | '.join('---' for _ in cols)+' |']
        for _,row in frame.iterrows():
            vals=[]
            for col in cols:
                v=row[col];vals.append(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v))
            text.append('| '+' | '.join(vals)+' |')
        return '\n'.join(text)
    back=competition[competition.parameter_group=='backbone'];cond=competition[competition.parameter_group=='condition_branch'];deltas=summary[summary.metric=='rmse_delta_with_minus_base']
    qview=validation[['seed','target','q10_gt_q50_rate','q50_gt_q90_rate','negative_width_rate','central80_coverage','signed_width_mean','signed_width_median']]
    gradview=competition[['seed','parameter_group','V2_to_V1_gradient_norm','cosine','gradient_conflict']]
    text=f'''# 4g QGeoGNN-V2 分位数与损失离线诊断

科学状态：`NEW_OFFLINE_MEASURED`。本报告只分析既有训练快照；没有改模型、loss、训练配置或AL方法，没有optimizer step、LLM请求、新选点或标签揭示。

## 证据边界与真正缺口

已阅读并按固定提交 `{BASE}` 复核 A0 指定的 training/predictor.py、独立六输出 model、uncertainty/quantiles.py、最终4g资格报告QUANTILE_AUDIT、8g loss-controls、R0/R1 regression ladder以及 point_metrics。文件SHA见 `INPUT_AND_CODE_HASHES.json` 的 `historical_evidence`。

`HISTORICAL_MEASURED`：资格报告已测过 seeds42/525/1101 × row/compound 的交叉、覆盖、宽度-误差相关性和top20富集，已披露test aggregate只在 `historical_reuse.csv` 重用，未重复前向或读取其test标签。row历史V1/V2覆盖均值约0.722/0.691，compound约0.638/0.463；不能直接套到当前AL回合。

8g对照研究是在4g pretrained→8g微调背景比较0.5/1/SD归一化，3seed×2split，不能作为纯4g AL的权重因果证据。R0→R1还同时改变shuffle、weight decay与checkpoint选择目标，也不能隔离loss权重。当前loss仍是每目标q10 pinball + q50 MSE + q90 pinball + 两个soft crossing，V1/V2等权。q50并非严格pinball中位数；其统计解释是MSE目标下的中心预测，不能宣称已校准条件中位数。

本次填补缺口：当前五个AL outer seeds 157/887/2357/6101/12203、L333、row、每seed333个已授权L0训练行和416个validation行。只读checkpoint/partition/纯结构graphcache均匹配已有冻结SHA。canonical CSV按sample_id与角色先过滤，再转换目标字符串；U0/test目标转换数均为0。输出只含aggregate，不保存单行真实标签/预测到Git。

这些模型只用了L333标签，covariate scaler来自冻结outer-train X、target sigma来自L0；资格报告则是不同3seed、全source训练预算且披露的是test。训练状态、划分、角色和尺度估计人口均不同，不作直接优劣归因。L525/L1005/L1485、当前compound AL和same-seed K3ensemble控制为 `NOT_MEASURED`。完整逐项状态见 `MEASUREMENT_STATUS.csv`。

## A1：raw loss、归一化误差与梯度贡献

`NEW_OFFLINE_MEASURED`：对固定checkpoint保持eval模式，在全部L0行分块前向/反向，按行数加权累积每目标每组件梯度，测共享backbone与condition分支；所有模型参数/缓冲哈希前后相同。loss组件及分组L2见 `loss_components.csv`；raw loss份额、train-only sigma、validation RMSE/NRMSE及normalized squared贡献见 `loss_scale_contributions.csv`。CSV的mean_loss在两个参数组重复用于对齐，汇总只取一组，不对这两份scalar叠加。两目标未经尺度标准化训练，而validation combined NRMSE为 `sqrt(0.5*((RMSE1/sigma1)^2+(RMSE2/sigma2)^2))`，二者不是同一统计量。

{md(derived,['seed','V1_raw_total_loss','V2_raw_total_loss','V2_raw_loss_share','V2_normalized_train_squared_error_share','validation_combined_nrmse'])}

{md(gradview,['seed','parameter_group','V2_to_V1_gradient_norm','cosine','gradient_conflict'])}

这五个冻结eval状态中，backbone的V2/V1梯度范数比都大于1（{back.V2_to_V1_gradient_norm.min():.3f}–{back.V2_to_V1_gradient_norm.max():.3f}），平均约{back.V2_to_V1_gradient_norm.mean():.3f}；backbone余弦均为正。condition分支更异质：{int(cond.gradient_conflict.sum())}/5个seed余弦为负，且并非每seed都由V2拥有更大梯度范数。原始V2标签/RMSE较大不能单独推出这些结论，结论来自实际导数测量。

`NOT_MEASURED`：实际Adam更新竞争和整个训练轨迹谁“主导优化”。本次没有优化器momentum/逐epoch状态回放，也未执行参数更新；eval梯度与train-mode BatchNorm导数可能不同。CSV的gradient-conflict与对合并梯度的内积只是方向诊断，不是Adam更新或因果权重实验证明。改变V2权重是否有益仍为 `HYPOTHESIS`，本次不改权重。

![损失和梯度](loss_and_gradients.png)

## A2：原始分位数可靠性

以下全部在排序或后处理之前从原始六输出计算；严格 `>` 与精确数值相等 `==` 分开记录，负宽度保持有符号。q10/q90 pinball、q50误差、mean/median signed与clipped width、三个精确相等率均在 `quantile_metrics.csv`。

{md(qview,['seed','target','q10_gt_q50_rate','q50_gt_q90_rate','negative_width_rate','central80_coverage','signed_width_mean','signed_width_median'])}

当前validation的80%名义区间覆盖波动很大：V1为{validation[validation.target=='V1'].central80_coverage.min():.3f}–{validation[validation.target=='V1'].central80_coverage.max():.3f}，V2为{validation[validation.target=='V2'].central80_coverage.min():.3f}–{validation[validation.target=='V2'].central80_coverage.max():.3f}，不能笼统称为已校准80%区间。按L0响应四分位阈值与事后validation误差四分位分组的覆盖见 `coverage_groups.csv`；后者是描述性难度分层，不能用于策略选择或盲确认。

`sorting_*` 字段单独比较推理排序前后q50变动率、平均绝对变动和MAE/RMSE。排序会替换交叉行的中心输出，它只是后处理，不是模型内部硬约束；任何排序结果都未覆盖原始交叉率、误差或历史冻结报告。

## A3：宽度的增量困难样本信息

分别保存并按L0 sigma标准化 `w1=V1_q90-V1_q10`、`w2=V2_q90-V2_q10`，保留截零版本和crossing标志。`signal_diagnostics.csv` 含 signed/clipped width 对标准化绝对q50误差的Spearman、Top20% error enrichment、AUSE，以及与latent到L0距离、完整网络归一化q50梯度范数、9D条件覆盖距离和q50值的相关性/Top20重叠。Top20大小固定ceil(20%×n)，平局按原输入顺序。AUSE明确使用0..0.9十个固定剔除比例的归一化retained-MAE差面积。

增量检验使用固定 `SHA256(canonical_smiles) mod 5` 分组折，化学组不跨折；每折标准化只在辅助模型训练折拟合，ridge固定为1，不调参。对每个seed/target在validation内部做事后OOF困难度预测。baseline包含两目标q50/sigma、latent distance、两目标完整网络q50梯度norm及9D条件信号/距离，再加入signed/clipped widths和crossing flags。**这里的“留出”只针对辅助ridge；validation之前已用于选择原GNN checkpoint，不是独立盲确认数据。**

{md(incremental[['seed','target','baseline_oof_rmse','with_width_oof_rmse','rmse_delta_with_minus_base']],['seed','target','baseline_oof_rmse','with_width_oof_rmse','rmse_delta_with_minus_base'])}

{md(deltas,['target','mean','lower95','upper95'])}

在这组固定辅助模型和可获得控制信号下，加入宽度未改善困难度预测，平均OOF RMSE反而增大：V1 {float(deltas[deltas.target=='V1']['mean'].iloc[0]):+.5f}、V2 {float(deltas[deltas.target=='V2']['mean'].iloc[0]):+.5f}。这只是当前L333/row/固定ridge规格下没有观察到增量收益；不能证明宽度在所有非线性模型、预算或策略下都无信息。

`NOT_MEASURED`：匹配的same-seed、same-budget K=3 ensemble disagreement控制。没有将不同outerseed模型混合成伪ensemble，因此不能声称已经控制所有已知AL信号。bootstrap按5个seed聚合而非把所有行当独立样本；区间仅描述同一复用数据集和固定预算下的seed变异，不是新数据总体或学习效率的置信区间。

单独记录 `V2_q90-V1_q10` 的离线相关性。它等于 `(V2_q50-V1_q50)+(V2_q90-V2_q50)+(V1_q50-V1_q10)`，同时含洗脱过程中心跨度和两端边际；不是纯epistemic uncertainty，也不是 `(V2-V1)` 的一个有效分位数，不能据此解释80% coverage。

![分位数和增量诊断](quantiles_and_incremental.png)

## 可复现性与结论边界

代码、输入manifest、checkpoint/partition/graphcache来源及SHA、每seed参数/缓冲state前后哈希在 `INPUT_AND_CODE_HASHES.json`。原始授权标签、checkpoint和图仅放在独立只读私有scratch快照，不进入本报告Git目录。本次真实测量实际使用独立Git快照 `4ee480378b0293f1985e3ee4a09df2c78f77901e`。另外逐文件证明14个科学依赖文件与main祖先 `16eca74769f664bc378a58ccbf380a806723f587` 原字节完全一致，证书见 `main_ancestor_recovery`。因此 main-only 复现可从16eca恢复相同科学代码，不依赖B分支或其可编辑API。浅克隆若缺少固定历史对象，须先显式取回16eca；本程序不会自动fetch/pull或改工作树。只读恢复输入的独立演练已通过，未为此重跑诊断。运行示例：

```sh
python prepare_inputs.py --repository /path/to/main/repository --destination /private/tmp/fresh_quantile_snapshot\nKMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \\
python run_audit.py --snapshot-manifest /private/tmp/fresh_quantile_snapshot/data/MANIFEST.json \\
  --legacy-source /private/tmp/fresh_quantile_snapshot/code --output /path/to/diagnosis
python render_report.py --repository /path/to/original/repository --output /path/to/diagnosis
```

新核心诊断的9项合成安全/数值测试通过。**没有执行QGeoGNN训练、付费LLM、U/test目标读取、真实选点接受或标签揭示。**

`HYPOTHESIS`：宽度能否改善AL AULC或达到给定误差所需标签数，仍须等当前CW-Free结束并获新授权后开展预注册、匹配seed/budget的正式对照。当前offline富集、OOF或梯度事实均不构成学习效率或loss权重因果结论。
'''
    (folder/'SCIENCE_AUDIT.md').write_text(text)
    p=folder/'INPUT_AND_CODE_HASHES.json';provenance=json.loads(p.read_text());provenance['historical_evidence']=evidence;provenance['code_hashes']={q.name:sha(q) for q in HERE.glob('*.py')};provenance['artifacts']={q.name:sha(q) for q in folder.iterdir() if q.is_file() and q.name!='INPUT_AND_CODE_HASHES.json'};p.write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'report':str(folder/'SCIENCE_AUDIT.md'),'figures':2,'measured_seeds':len(seeds),'budgets':[333],'diagnostic_only':True},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repository',type=Path,required=True);p.add_argument('--output',type=Path,default=HERE);args=p.parse_args();render(args.output,args.repository)
