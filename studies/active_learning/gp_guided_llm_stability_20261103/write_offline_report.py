"""Render Chinese reporting tables from the measured, frozen offline results."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent


def main():
    metrics=json.loads((HERE/'results/metrics.json').read_text())
    verification=json.loads((HERE/'results/verification.json').read_text())
    response=[];risk=[]
    for row in metrics:
        if row['model']!='gp':continue
        baseline=next(x for x in metrics if x['seed']==row['seed'] and x['space']==row['space'] and x['target']==row['target'] and x['model']!='gp')
        if row['target'].startswith('response'):
            response.append(f"| {row['seed']} | {row['space']} | {row['target'][-2:]} | {row['rmse']:.3f} / {baseline['rmse']:.3f} | {row['nlpd']:.3f} / {baseline['nlpd']:.3f} | {row['coverage_90']:.1%} |")
        else:
            risk.append(f"| {row['seed']} | {row['space']} | {row['spearman']:.3f} | {row['rmse']:.3f} / {baseline['rmse']:.3f} | {row['nlpd']:.3f} / {baseline['nlpd']:.3f} | {row['coverage_90']:.1%} |")
    report="""# GP 实际离线验证报告

日期：2026-11-03，Asia/Shanghai。**Phase 0 已完成；正式 GP 引导与付费 Phase 1 为 NO-GO。** 固定旧 checkpoint 的风险排序出现可继续研究的信号，但响应 GP 明显欠覆盖，且当前模型状态适用性没有验证。下述结果不是 LLM 主动学习效率或稳定性改善的证据。

## 1. 实际数据与权限

只使用 seed157/6101 已存在的历史 Raw/CW bank、初始 catalog/contract，以及新 CW-Free 已完成的前6批反馈。每 seed 的初始333响应只用于训练；随后192个已观测响应按批次先后用于响应 GP forward 核验。风险 GP 用固定 r00 预测构造误差，前96点拟合、后96点核验。两个 seed 均为已有开发 seed，划分来自同源数据；逐行数量不能当作独立重复数量。

来源冻结在 [input_manifest.json](input_manifest.json)：45个文件双读一致，并核对上游 bank、contract、batch freeze 和已观测 receipt。反馈 receipt 的 `feedback_sha256` 是生产 `stable_hash` 的 canonical JSON 摘要；batch freeze、catalog 和缓存则使用原始字节 SHA256，两者明确区分。首次读取时发现该约定差异后修正适配器，没有修改历史文件或绕过哈希。

初始 bank 来自旧 Free 历史，但与新 CW-Free L333 checkpoint 和3330行有序 L/U **精确一致**，因此不是用后期 bank 代替历史表示。原生当轮 residual 的不同 checkpoint 已逐行登记，不进入固定 r00 风险拟合。

本次付费 LLM=0，QGeoGNN训练=0，新标签揭示=0，test真值读取=0，checkpoint权重加载=0，原始全量数据表读取=0；只运行单线程 CPU 的小规模 GP。没有调用正式 AL runner、label-store、生产 transport，也没有修改运行中的实验目录。

## 2. 预设、计算和筛查

核、L0缩放、拆分、两个 seed、64随机批次和开发筛查标准在 [configs/offline_plan.json](configs/offline_plan.json) 首次评分前保存并绑定哈希。每种表示各拟合两个独立响应 GP 和一个风险 GP；幅度/噪声只按训练似然估计，不使用评价标签选参数。风险 Gaussian 目标是 `h=log(1+两端点标准化平方误差均值)`，不是已校准的错误概率。

52次 GP 拟合全部优化成功、无对数参数边界命中、实际 Cholesky jitter 为0。一次哈希重放的数值分析耗时约5秒（不含原始文件审计、测试）；这不是在线服务延迟或大规模运行成本估计。数值计算成功与概率模型可信分开判定。

开发筛查要求各 seed、各端点 NLPD 不劣于训练侧常量 Gaussian、90%覆盖偏差不超过0.10；风险 rho至少0.2且NLPD优于训练常量。它们是本次开发的早期停止筛查，不是正式确认阈值。当前 response 四个 seed×表示组合均未过；风险四个组合过此弱筛查，但状态迁移仍未证实。

## 3. 响应 GP：均值有信号，概率覆盖不合格

下表每行192个 forward 评分点。RMSE 单位是固定 L333 端点标准差；NLPD 在同一标准化空间，越低越好。“GP / 常量”基准的常量均值和方差只来自 L0。名义覆盖为90%。

| Seed | 表示 | 目标 | RMSE：GP / 常量 | NLPD：GP / 常量 | GP实际90%覆盖 |
|---|---|---|---:|---:|---:|
"""+'\n'.join(response)+"""

GP相对弱常量基准降低了均值误差；这没有比较 QGeoGNN 重训性能。名义90%区间实际覆盖61.5%–74.5%，欠覆盖15.5–28.5个百分点。seed6101的V2在两种表示下，NLPD还劣于常量基准。不能把这些方差称为可靠的响应概率、当前 QGeoGNN 风险或安全性保证。

完整50/80/90/95%覆盖、区间宽度、标准化误差RMS和PIT直方计数保存在 [results/metrics.json](results/metrics.json)。没有为改善覆盖而在这192点评分上回调噪声或阈值。暂未计算CRPS、分子/条件/轨迹block置信区间和尾部条件覆盖，因此不是正式概率放行。

## 4. 风险 GP：固定 r00 的有限正信号

每行训练96点、核验96点。RMSE/NLPD在 h 空间；不能与上表端点误差横比。常量分布只由前96点确定。

| Seed | 表示 | 风险排序Spearman | h-RMSE：GP / 常量 | h-NLPD：GP / 常量 | h实际90%覆盖 |
|---|---|---:|---:|---:|---:|
"""+'\n'.join(risk)+"""

四个组合rho为0.461–0.556，NLPD均优于训练常量；这是固定旧模型的误差排序和概率评分线索。不能据此宣称已学到整个候选池的真实风险：这些标签来自原LLM轨迹的选点，存在选择偏倚；样本少、同分子条件相关、无独立block确认。seed6101的90%区间覆盖96.9%也体现偏保守，不能只挑覆盖较好的seed。

该风险模型直到前96点观测后才能拟合；此时正式轨迹的 QGeoGNN 已更新。因此本次没有一个“当前模型、当前时点、经独立校准”的可部署风险模型。后期checkpoint迁移、噪声异质性与候选分布漂移必须单独验证。

## 5. 与既有 IVR 及批次信息量的关系

在2997个初始候选上，响应 GP 潜变量方差与历史单位固定核条件方差的Spearman为0.821–0.870，与固定核首步IVR分数为0.791–0.848。GP均值与IVR分数也相关但不相同。详见 [results/signal_comparison.json](results/signal_comparison.json)。这些高相关表明共同梯度几何占了很大部分；不证明新增概率信号有下游价值，也不把单位噪声重新解释成已校准不确定性。

对首批历史Free32与64个预设随机32点批次，逐seed、表示、端点用联合后验协方差计算 `I(y_B;f_B|D_L0)`。历史批次在全部8个组合中都高于64随机批次的最大值：历史26.27–32.39 nats，随机中位15.89–18.37 nats。详见 [results/batch_information.json](results/batch_information.json)。两个端点单列，不作未经验证的联合汇总。

**这反驳了在这些状态下先验假设“Free32必然低信息”的做法。** 该结果只属于尚未充分校准的GP surrogate；有限随机参考也不提供总体最优性或真实QGeoGNN学习收益。控制器阈值不能依据这些批次事后设定为恰好触发，更不能因未触发而改用有利指标。

## 6. 控制器和输入检查：真实数据与 mock 分开

Controller默认阈值全部为null，`deployable=false`。它使用完整batch的联合IG、全部496对后验相关、冻结假设下的严格单因素结构豁免。缺合格风险、同实验域参考、阈值冻结或数值有效性时输出不可判定；没有全局覆盖硬下界或人为综合权重。

合成mock保留原始32点提案、诊断/触发、是否请求修正、完整32点修订、最终ID及理由差异。toy阈值仅用于测试流程，不是经校准的安全条件，mock的分数改变不构成研究收益。日志见 [results/mock_controller/mock_trace.json](results/mock_controller/mock_trace.json)。真实冻结输入的字节/协议和三臂差异检查另见 [results/real_frozen_input_check.json](results/real_frozen_input_check.json)；该检查只证明离线envelope构造一致，不是一次真实LLM请求或可部署协议认证。

数学/权限测试覆盖直接条件分布、joint与边际IG区别、条件链式法则、非法协方差拒绝、noise/jitter分离、历史IVR第一步对齐、评估标签扰动不改变先前响应或风险拟合、白名单、输入漂移、局部豁免和纠错轨迹。测试结果见 [results/test_results.txt](results/test_results.txt)，完整重放校验见 [results/reproduction_verification.json](results/reproduction_verification.json)。

## 7. 局限与下一步

本次只试低成本线性核、同方差Gaussian似然、经验均值和plug-in超参数；没有证据表明增加复杂核一定有效。响应校准失败可能包含尾部选择、异方差、表示失配及参数不确定性。Raw/CW的差异未形成稳定优胜，不能择一个好看组合宣称成功。

下一项最小工作仍是**无付费的概率校准补证**：在尚未用于本次拟合/核验的已观测历史block上，冻结适用checkpoint、训练/校准/核验拆分，比较简单预设噪声/尺度校准与原模型；逐端点报告适当评分、覆盖与宽度，并验证固定旧模型风险如何迁移或不能迁移。先写方案再看这些block结果，不以本次后96点反复调到通过。

未执行Phase1–3，未产生A/B/C学习曲线。不能回答数值信息或约束是否提高平均AULC、跨seed标准差、尾部性能或成本。当前最有价值的结论是：**有继续离线研究的机制依据，但可信GP不确定性与状态适用性仍是付费决策实验的前置缺口。**
"""
    (HERE/'OFFLINE_REPORT.md').write_text(report)


if __name__=='__main__':main()
