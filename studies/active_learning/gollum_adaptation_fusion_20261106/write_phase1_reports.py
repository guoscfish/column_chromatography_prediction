"""Write Chinese reports from actual immutable numerical receipts."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from data_boundary import HERE


def table(rows, headers):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def fmt(x,d=4):
    return f'{x:.{d}f}'


def main():
    s=HERE/'summary'
    a=pd.read_csv(s/'phase1_aggregate.csv');f=pd.read_csv(s/'phase1_fits.csv')
    costs=json.loads((s/'phase1_cost.json').read_text());gates=json.loads((s/'phase1_gates.json').read_text())
    split=[]
    for p in sorted((HERE/'phase1').glob('*_identity.json')):
        d=json.loads(p.read_text());split.append([d['state']['seed'],d['state']['budget'],*[len(d['roles'][r]) for r in ('fit','calibration','check')]])
    method_names={'frozen':'Frozen','original':'A40','converged':'B200','regularized':'C Ridge','noise_floor':'D Noise floor','staged':'E Staged'}
    lines=['# Phase 1 受控训练消融结果','',
           '本报告来自已执行的真实缓存与训练，不含模拟效果。首轮固定2开发seed×L333/L813、2端点、Projection init17/29；全部失败和未收敛结果保留。主矩阵200次拟合，另补8次同拆分Frozen Fusion；8次全L_t数值复跑单列。', '',
           '## 合法拆分与解释范围','',table(split,['Seed','预算状态','GP fit行','calibration行','check行']), '',
           '按分子hash固定分组，不是旧Row validation任务的同分布复现。Qwen-only的check分子完全未进入GP拟合/校准；Qwen预训练未使用本任务标签。Fusion/GNN/Raw/CW的历史GNN见过整个L_t，因此内部check仅为冻结监督表示后的GP核验；不能当成端到端无泄漏外样本证据。416行validation参与过GNN checkpoint选型，仅作开发评价。PCA/GP缩放仅fit子集，calibration只拟合观测方差temperature，check从不反馈调参。', '',
           '## 分组核验：原始概率输出与校准分开','',
           '表中先平均初始化/端点/预算，再等权平均seed。nRMSE=RMSE/历史L0端点std；NLPD为原mL单位密度评分。Coverage和标准化RMS不能取代区间宽度与NLPD。temperature只改变观测预测区间，不改均值/原始GP选点。', '']
    for rep in ('qwen','fusion'):
        rows=[]
        for method in method_names:
            for cal in ('raw','temperature'):
                d=a.loc[(a.representation==rep)&(a.method==method)&(a.role=='check')&(a.calibration==cal)]
                if len(d):
                    r=d.iloc[0];rows.append([method_names[method],cal,fmt(r.normalized_rmse_L0),fmt(r.nlpd),fmt(100*r.coverage_90,1)+'%',fmt(100*r.coverage_95,1)+'%',fmt(r.standardized_error_rms,2)])
        lines.extend(['### '+rep,'',table(rows,['方法','概率输出','nRMSE','NLPD','90%覆盖','95%覆盖','标准化RMS']),''])
    lines+=['## 区间宽度按端点保留','']
    m=pd.read_csv(s/'phase1_metrics.csv')
    width=m.loc[(m.role=='check')&m.representation.isin(['qwen','fusion'])].groupby(['representation','method','calibration','endpoint'],as_index=False)[['mean_interval_width_90','mean_interval_width_95']].mean()
    width.to_csv(s/'phase1_interval_width_by_endpoint.csv',index=False)
    lines+=['所有方法的90/95区间宽度分别见 [逐端点宽度](summary/phase1_interval_width_by_endpoint.csv)，不把V1/V2的mL宽度混成单一量。校准前后保持同均值，宽度乘sqrt(alpha)；巨大alpha是方差尺度失配的证据，不能证明信息排序正确或GP假设成立。','',
            '## 优化行为','']
    rows=[]
    for (rep,method),d in f.loc[f.representation.isin(['qwen','fusion'])].groupby(['representation','method']):
        rows.append([rep,method_names[method],len(d),fmt(100*d.optimizer_success.mean(),1)+'%',f'{d.noise_signal_ratio.median():.3e}',fmt(d.condition_number.median(),1),fmt(d.temperature.median(),2),fmt(d.training_seconds.sum(),1)])
    lines+=[table(rows,['表示','方法','有限fit','optimizer success','noise/signal中位','条件数中位','alpha中位','训练秒数合计']), '',
            'loss_history和objective_evaluations是函数求值记录，包含未接受的线搜索点，不应称作每次接受迭代的单调学习曲线。B200只是增加预算的候选，不能凭名称宣称收敛。E的success是表示冻结后核超参数阶段的状态，第一阶段停止状态另存，不能冒称整个联合目标已收敛。gradient-qualified门槛同时检查有界投影梯度<1e-3；完整梯度、MLL、边界和迭代在 [fit明细](summary/phase1_fits.csv)。全部数值求解float64，jitter与noise分开记录。', '',
            '## 种子、预算与最差情形','',
            '逐seed×预算×初始化×端点评分见 [完整metrics](summary/phase1_metrics.csv)；[seed配对差](summary/phase1_paired_seed_differences.csv) 保留B–E相对A40的差，不挑最优初始化。下表为每个seed平均两预算、端点、初始化后的原始check结果。', '']
    seedmeans=pd.read_csv(s/'phase1_seed_means.csv')
    rows=[]
    for _,r in seedmeans.loc[(seedmeans.role=='check')&(seedmeans.calibration=='raw')&seedmeans.representation.isin(['qwen','fusion'])].iterrows():
        rows.append([int(r.data_seed),r.representation,method_names[r.method],fmt(r.normalized_rmse_L0),fmt(r.nlpd),fmt(100*r.coverage_90,1)+'%'])
    lines+=[table(rows,['Seed','表示','方法','nRMSE','NLPD','90%覆盖']), '',
            '只有两个开发seed，不能称最终跨seed性能，也不将4状态当4独立统计重复。最差seed和预算必须与整体均值一起读；未进行显著性检验。', '',
            '## 采集排序与文本鲁棒性','']
    stability=pd.read_csv(s/'phase1_acquisition_stability.csv')
    rows=[]
    for (rep,method),d in stability.loc[stability.comparison_kind=='initialization'].groupby(['representation','method']):
        rows.append([rep,method_names[method],fmt(d.variance_rank_spearman.mean(),3),fmt(d.maxvar_top32_overlap.mean(),3),fmt(d.conditional_ivr_top32_overlap.mean(),3),fmt(d.batch_information_gain_top32_overlap.mean(),3)])
    lines+=[table(rows,['表示','方法','init方差秩ρ','MaxVar重叠','IVR重叠','BatchIG重叠']), '',
            '重叠为同状态同池32点交集/32，不是Jaccard。全部候选是原冻结U_t中同一256点子池。跨seed的L_t、checkpoint、候选池不同，不能把跨seed差异当作仅训练随机性效应；本表init17/29隔离GP初始化。', '',
            '模板A→B同一模型前向，未用B调参；均值变化、方差秩ρ与三种采集重叠在 [模板稳定性](summary/phase1_template_stability.csv)。[采集覆盖](summary/phase1_acquisition_coverage.csv) 保留分子数、全部五条件取值覆盖、批内相关、信息量及成本。方差绝对失准与相对排序分别报告；排序稳定本身不证明对未来误差/下游AULC有价值。', '',
            '## 开发validation与基线','']
    rows=[]
    for _,r in a.loc[(a.role=='development_validation')&(a.calibration=='raw')].iterrows():
        rows.append([r.representation,method_names[r.method],fmt(r.normalized_rmse_L0),fmt(r.nlpd),fmt(100*r.coverage_90,1)+'%'])
    lines+=[table(rows,['表示','方法','nRMSE','NLPD','90%覆盖']), '',
            '这张表只评价GP未拟合的416行，但历史GNN用过该集选checkpoint。仅展示冻结设置下的开发结果，不按它重新选正则/噪声强度。Structured/Raw/CW/Frozen GNN与语言模型使用相同GP fit子集；与旧全L_t报告不能直接做因果方法对比。', '',
            '## 成本与门槛','',f"实际保存有限fit {costs['finite_fits']}，失败fit {costs['failed_fits']}。训练累计 {costs['fit_seconds']:.1f} 秒；含逐fit数值与采集诊断 {costs['total_fit_plus_diagnostics_seconds']:.1f} 秒。墙钟与补充复跑成本各自见completion JSON；测试工程fit不计入科学矩阵。", '',
            table([[g['representation'],method_names[g['method']],fmt(g['optimizer_success_rate'],3),fmt(g['gradient_qualified_rate'],3),g['coverage_gate'],g['standardized_rms_gate'],g['nlpd_vs_frozen_qwen_gate'],g['pass']] for g in gates],['表示','方法','success率','梯度资格率','覆盖门槛','RMS门槛','NLPD门槛','通过']), '',
            'Fusion Noise-floor在条件性GP check上的原始覆盖/NLPD明显改善，但16/16仍未报告优化收敛，且GNN已见过这些check标签，不能据此宣布可信外样本概率专家。扩展门槛在训练前冻结，完整判据见 [PROTOCOL.md](PROTOCOL.md)。未通过时停止5seed×4budget扩展，不靠temperature、挑预算或删失败来取得资格。此次没有正式QGeoGNN AL轨迹、没有新模型权重下载、没有测试真值访问。', '',
            '![真实诊断图](summary/phase1_diagnostics.png)','']
    if (HERE/'test_receipt.json').exists():
        receipt=json.loads((HERE/'test_receipt.json').read_text())
        lines.append(f"最终工程测试：整目录{receipt['passed']}项全部通过（{receipt['pytest_seconds']}秒），见 `test_results.txt` 与 `test_receipt.json`。这不替代科学概率门槛。")
    (HERE/'PHASE1_RESULTS.md').write_text('\n'.join(lines))
    full=[]
    for p in sorted((HERE/'full_l').glob('*.json')):
        d=json.loads(p.read_text());aa=d['fit_audit'];mm=d['development_validation']
        full.append([d['budget_state'],d['representation'],method_names[d['method']],fmt(aa['marginal_log_likelihood'],2),fmt(d['training_rmse'],4),fmt(mm['rmse'],3),fmt(mm['nlpd'],2),fmt(100*mm['coverage_90'],1)+'%',fmt(aa['noise_variance'],6),fmt(aa['covariance_condition_number'],1)])
    diagnosis=['# Phase 1：过度自信与训练问题诊断','',
               '旧目标梯度、训练/推理核一致性均经真实优化闭包数值差分核验，未发现导致欠覆盖的具体数学实现Bug。关键根因证据指向高容量监督Projection在经验贝叶斯MLL下的过拟合、noise下界吸附和长度尺度耦合；40步不足以收敛是事实，但增加迭代本身没有解决概率失配。不能把这些证据归咎于小型LLM或“GP天然不适合”。', '',
               '## 历史证据','',
               '旧720端点fit中Projection360/360在40步上限停止，0次optimizer success；185次超参数触界。Qwen Projection观测90%覆盖22.7804%，Fusion21.7949%；Qwen Frozen为95.4267%。Qwen noise variance中位从68.0772降至0.05158 mL²，Fusion从2.10438降至0.05116。边界、数据身份与准确分母见 [AUDIT.md](AUDIT.md) 和 [机器审计](audit_results.json)。', '',
               '## 原样重放与单因素加预算','',
               'seed157、L333/L813、V1、init17、同完整L_t、Qwen及旧式Fusion：A40预测与封存历史在1e-6容差内一致；B只增加maxiter至200，不改核/标准化/初始化/噪声边界。下表全部是开发validation，不能当独立确认。', '',
               table(full,['L状态','表示','方法','训练MLL','训练RMSE','开发RMSE','开发NLPD','90%覆盖','noise mL²','条件数']), '',
               '200步继续提高训练证据并几乎插值训练响应，却没有恢复外样本概率质量，4/4组合NLPD更差，3/4组合覆盖下降（L333 Fusion覆盖略升但仍只有11.1%）。noise仍能落在标准化exp(-9)下界；多个ell达到exp(5)=148.413。MLL提升不等于泛化或概率校准改善；B200仍未充分收敛，不将其当“真全局最优”。', '',
               '## 几何、可辨识性与数值条件','',
               '保存训练对距离在Projection初始化/最终的比值分位数、有效距离distance/ell、W/b范数、压缩对比例及核条件数。上述8次全L_t复跑没有pair距离降到初始化的十分之一以下，距离比中位反而>1；不能声称“Projection整体压缩了所有距离”。较大ell使有效核距离变小、协方差病态，且学习的表示可拟合标签差异，这与整体几何缩小是不同机制。', '',
               '若直接将核输入z和ell同比缩放，Matérn完全不变；ELU的负分支不是齐次函数，因此W,b同比缩放不是所有样本上的严格对称。这里应称权重与ell尺度耦合/弱可辨识。表中条件数可到1e9级；所有这些fit的Cholesky仍成功，不能把统计欠覆盖归因于Cholesky已经数值失败。训练/后验jitter单独记账，不把jitter当响应噪声。', '',
               '本轮Phase1保留输入/源码哈希、初始化、完整loss/gradient诊断及预测/采集产物，未为每个fit额外序列化完整W/b。重建这些Projection后验需按冻结代码确定性重新拟合；不能只从W范数恢复。Phase3则可直接从已存标量参数精确重建。', '',
               '潜变量方差与观测方差始终分开：`v_obs = v_latent + noise`。旧报告及本轮覆盖评价均用v_obs。SciPy接收Torch float64的真实value/gradient，训练特征与查询特征使用同一训练侧均值/global RMS、同一W/b/ELU与distance维数归一。', '',
               '## 受控修正及校准','',
               '主消融把GP fit、校准、核验按分子固定分离，比较A40、B200、Gaussian权重正则、预设noise floor、表示冻结后再拟核超参数。完整量化见 [PHASE1_RESULTS.md](PHASE1_RESULTS.md)，没有使用416行反复选强度。variance temperature只在单独calibration估计；其对check覆盖、宽度、NLPD的影响同时保留。它可以诊断并修正观测方差的整体尺度，但不能自动给出一致的潜函数联合协方差或可信IVR/BatchIG。', '',
               '此处机制证据不唯一识别所有因果来源：异方差/模型失配、分子组分布变化、checkpoint选择和未积分Projection参数不确定性都可能影响覆盖。实验支持否定“40步预算足以训练”及“只需更多迭代就会校准”的说法，不支持在无进一步对照时认定唯一根因。', '',
               '## 概率资格与仅排序信号','',
               '绝对校准与候选排序分别评分。即使经过单独校准后覆盖改善，也不能将raw GP称作已校准概率专家。MaxVar/IVR/BatchIG在同一合法子池真实计算并保留初始化/文本敏感性；排序稳定和自身信息增益都不能解释为QGeoGNN AULC改善。按照冻结门槛决定扩展和下一阶段，详见 [NEXT_STAGE_DECISION.md](NEXT_STAGE_DECISION.md)。','']
    (HERE/'PHASE1_GP_DIAGNOSIS.md').write_text('\n'.join(diagnosis))

if __name__=='__main__':
    main()
