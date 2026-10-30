# Free LLM32 / LCMD32 五 seed 机制诊断协议

日期：2026-10-29（Asia/Shanghai）。性质：已完成轨迹的探索性、观察性分析；不是新增策略试验或事前注册。研究对象仅为 4g QGeoGNN-V2 Row split。

## 1. 固定范围与证据边界

- seeds = 157、887、2357、6101、12203；每法 37 个冻结验证拟合点（L333–1485，步长32），36批选择，每批32；配对单位为 seed。
- 独立输出本目录，不覆盖历史结果。只读冻结 checkpoint、梯度 bank、选择、catalog、已揭示 feedback 和元数据。预测 CSV 仅在完整性审计中按字节哈希，不加载其中的候选或 test 真值。
- 不训练、不调用 LLM、不重跑选择、不重新揭示标签、不读取新 test 真值、不执行梯度重建。缺项为 NA 并注明原因；禁止由性能结果反推选择。
- validation 是开发性轨迹，不能由其诊断声称独立泛化或机制因果性。所有假设、相关系数、阈值均为描述性；不以180批为180独立样本，不开展显著性筛选。
- 本地 HEAD 与远端 `git ls-remote origin refs/heads/main` 已核为 `fbf4efa9efe251a22c3865d15718e149a38eeb7d`。源谱系包括原 raw gradient `8081e865`、scientist-v2 `1ec395f8`、全候选反馈 `93609bd9`、CW 分支与后续冻结提交；详见审计 JSON。
- Free157/6101 的历史 Astra/high 前缀与其他 Sol/xhigh 轨迹分层报告，不能将版本差异解释为策略差异。历史授权纠正、ID抄写修正、887 r33/12203 r24 amendment 保留并披露。

## 2. 来源审计

`evidence_audit.py` 仅导入现有报告模块的只读 `collect(require_complete=True)`，不调用报告主程序；复核每次冻结模型/预测的哈希、初始化、划分、预算、每批独立32 ID、反馈链与冻结先于揭示的 receipt，逐点与已提交曲线快照比对。该 collector 还校验了已有 Random/Hybrid 前缀，仅用于源文件完整性，不将其纳入本次方法比较。

`runtime_source_hashes.json` 保存逐文件来源 SHA256；`runtime_integrity_audit.json` 保存独立复核。几何/行为模块记录自己实际读取的文件与缺失状态。运行依赖本地 gitignored runtime；新克隆可读派生CSV/图表，不能据此宣称复现 runtime 审计。

## 3. 学习曲线

沿用冻结 L0-only 归一化：每个 seed 初始333标签的 V1/V2 population std（ddof=0）固定为 s1、s2，NRMSE = sqrt(((RMSE1/s1)^2+(RMSE2/s2)^2)/2)。不重选分母。不同seed按原实验定义比较，分母并非全训练集或验证集标准差。

normalized validation AULC[a,b] = trapezoid(NRMSE versus labels)/(b-a)，数值越低越好，不额外除以初始误差。阶段为333–525、525–653、653–813、813–1005、1005–1197、1197–1485，加全程333–1485及问题驱动的653–1005。

- 绝对误差：同预算 E(n)；阶段改善：E(a)-E(b)，正数表示改善；速度：改善/(b-a)，另以每100标签表达；相对改善：(E(a)-E(b))/E(a)。
- 波动：相邻冻结点 E(n)-E(n+32)、恶化次数/幅度、total variation；保留全部非单调点，不用单调包络代替原曲线。
- 交叉：逐seed的Free-LCMD差值改变符号的相邻预算区间，线性插值仅作位置提示；持久领先和偶发交叉区分，不把初始相等算领先。
- 描述性阈值 = 0.50、0.45、0.40、0.35、0.30。报告首个实际拟合预算，不插值阈值标签数；未达到为 right-censored@1485；另区分偶次达到与持续达到。
- 最佳验证值及标签预算来自已冻结37点，属于事后最优，不能替代固定预算比较。
- 配对差一律明确方向：误差/AULC中 Free−LCMD 负为Free较优；改善量中 Free−LCMD 正为Free降得更多。汇总先按seed计算，再报告均值、样本SD、5个配对值及获胜数。
- 653–1005 的改善差恒等于期初误差差减去期末误差差。对全量参考 R_s 另报 E(n)-R_s 及阶段消除剩余误差比例；分母不为正时该比例NA。参考是既有3330标签 validation fit，可能被部分标签轨迹超越，不是不可突破的误差下界，不作因果校正。

## 4. 几何行为

只使用该方法该seed该轮冻结bank和当时L/U索引。LCMD真实采集空间为原始 q50 full-network 512D CountSketch，不是 CW-LCMD；其源副本与git8081e865逐字节一致。Free raw与CW来自自己冻结状态，CW为L0缩放center/width输出梯度；不混入其他transform分支。

- 选中样本到L最近欧氏距离以及它在U中的midrank百分位（ties平均），分布保存分位数/高分位占比；Free还比对当时可见catalog的CW percentile。
- 批内最近邻与两两距离、冗余、覆盖变化与有效秩以模块的明确公式记录。覆盖收益在固定的事前U上比较加批次前后nearest-center距离，以相对下降或本轮归一化距离表达；标出是否包含本批自身归零的机械贡献。
- raw/CW距离绝对值只在各自轮次空间内解释，跨轮比较使用池内rank、相对覆盖收益等尺度无关指标。各轨迹的模型和L不同；相同标签预算并非相同模型状态。
- LCMD若没有冻结CW bank，则CW指标NA；不将512D raw bank当作可精确恢复另一个sketch的原始输出梯度。无确认不重算。
- 后续误差改善取该批后一次冻结拟合的验证差。相关性先按seed/方法/阶段展示，并查看去预算趋势或阶段中心化结果；不由相关推断信息增益。

## 5. 结构、条件与反馈

使用真实 batch ID 和无标签 catalog 元数据映射，不使用LLM理由推断行为。分子标识沿用catalog canonical SMILES；Morgan radius2/2048bit Tanimoto，与selector定义一致。不同SMILES表示、骨架定义及家族代理的限制另注。

- 每批unique分子数、重复行比例、选前已观察分子/新分子比例、同分子多条件数量与PE/EA跨度；条件补全以选择前L定义，完整SMILES+五条件tuple已见过的独立ID重复不算新点补全。matched contrast单列并可包含重复。区分已见分子补条件和一次批内新分子多条件。
- PE/EA转为EA分数 EA/(PE+EA)；loading amount = density(g/ml)×V(µl)，单位mg，沿用selector派生量。统计跨度、分组和同分子条件变化。若loading只有单一水平，不能声称发现loading学习机制。
- 结构多样性、近邻相似与局部家族聚集和简单重复并列，不能将同分子必然相似直接解释为无用冗余。
- 历史残差只来自该轨迹选择前已揭示批次的采集时 prediction−measurement。初始L333无out-of-sample采集残差；不拿当前拟合残差补上。高残差集合、条件邻近定义由行为模块固定并记录；r0无历史则NA。
- 当前批feedback只能用于事后检验本轮假设，或进入下一轮历史。禁止把本轮获取标签后的误差用作本轮事前特征。

## 6. 文本与信息依赖

分析冻结 choices/reasons、hypotheses、batch_rationale、feedback_interpretation，分字段、轮次/阶段/模型版本统计CW、quantile width、similarity、化学先验、实测/残差、局部条件曲线、多样性、失败修复。关键词命中是可复现代理，不是人工语义真值；提示词本身提及某概念不算LLM主动引用。

将文本代理与真实CW rank、结构/条件重复、历史失败邻近配对查看，并抽查具体选择ID与批后响应。实际匹配只对叙述字符串values，不含schema键、ID或hash；模型修复不用泛化的fail/correct，避免逻辑纠正误判。引用率不能识别信息被利用的因果收益，尤其H5必须由以后NoCW验证。No-feedback/No-chemistry也需匹配输入、预算、模型版本与冻结状态，避免同时改变多项信息。

有限matched-condition探针固定seed157/887、r0/10/20/30（L333/653/973/1293），只读已存在raw/CW bank，以同分子且只改一个条件的行对直接差分范数计算小距离（避免norm-dot消减），按当前bank centered RMS定标。PE/EA、sample amount、loading volume的实际扰动范围/物理单位不等同，因此距离倍数不是单位导数或真实响应因果效应。探针在主诊断提出问题后增加，属探索性，不冒称事前注册，也不外推全五seed全部状态。

## 7. 执行与自检

运行方式、依赖与输出见 `README.md`。轻量测试覆盖指标恒等式、censoring、rank/coverage/effective-rank、选择/历史时间边界与合并行数；不触发任何训练入口。绘图为标准matplotlib静态科学图，图题/脚注注明 validation、5 seeds、seed均值/SD或配对口径；几何和选择指标注明来自对应冻结训练池而非validation样本。

最终停止在诊断报告与三个优先验证实验建议，不自动启动实验。
