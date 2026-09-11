# 柱迁移学习改进方案：以降低绝对误差和稳定跨柱收益为目标

审阅日期：2026-09-11。代码基点：`main@3a226d1`。本文是基于论文、代码和历史结果形成的研究方案，不是已完成的新实验，也不是性能保证。本轮只增加方案文档，未修改模型、重训网络或改变原有数据划分。

**建议主线：保留 QGeoGNN-V2 与已验证的 Center/Width 迁移基线，优先验证“按柱选择、可回退的预测修正”，再验证“已有 4g 实测记录提供的源域锚”。** 前者针对已经观察到的 25g/40g 负迁移，后者针对源模型误差经跨柱映射传播的问题。新的神经结构、物理参数扩展和主动学习暂列后续。

**1. 论文究竟提供了什么基准**

已提取所给 PDF 全部 15 页文字，并核对与模型、迁移、数据及实验设置有关的完整页面，重点为 PDF 第 6–10、12–14 页。论文正文页码比 PDF 页码小 1。

- 任务输出是开始洗脱体积 V1 和结束洗脱体积 V2；W=V2−V1 是洗脱区间宽度，不能直接解释为色谱峰标准差或理论塔板相关参数。
- 论文报告 4g 有 4,684 条记录、218 个化合物；8g/25g/40g 分别为 582/568/531 条记录、90/80/81 个化合物。目标柱化合物均在其 4g 化合物范围内。
- 论文方法采用随机 80/10/10 训练、验证、测试划分；迁移学习率为 1e-4，沿用预训练知识并调整模型。Figure 4 的迁移 R² 为 8g：0.759/0.752，25g：0.747/0.840，40g：0.826/0.824。论文未提供与当前仓库相同的 compound holdout 结果。
- 原文的迁移描述、公开代码和当前重构并不完全一致。公开代码对 target 全体数据的归一化、全参数优化、V1:V2=1:0.5 损失权重、未实际 step 的 scheduler、关闭 column-info 等问题，已在历史审计中逐项核对。当前 paper-style 也不能自动等同于作者原模型。[R1]

因此，正确的提升目标是：**在同一测试样本、同一标签预算及明确输入条件下超过可运行的论文式对照和当前强基线。** Figure 4 用于背景定位；拿另一套划分的 R² 超过图中数字，不能单独证明方法优于原文。原论文的正式出版信息见 [Chem 论文页面](https://www.sciencedirect.com/science/article/pii/S2451929425001883)。

**2. 历史已经排除了哪些低价值重复**

| 提交／阶段 | 实际得到的证据 | 对新方案的约束 |
|---|---|---|
| `a21554e`–`a0844fa` | 条件输入修正、无效参数裁剪、独立 V2 及 4g qualification；当前模型 458,952 参数，128 维池化后表示 | 不以重造 backbone 为默认起点 |
| `61f20c9`、`6e10498` | 跨柱低标签比较；scale、affine、identity shrinkage 很强；单调非线性校准没有稳定优势 | 不把线性输出校准或小 spline 换名当新方法 |
| `9225cc2` | EA 与残差有结构，但受控条件缩放没有稳定跨柱实质收益 | 相关性不等于新模型一定提升 |
| `76be41f` | 浅层、全量微调和 source replay 已试，未稳定超过强校准 | “解冻更多层”“加 replay”不是尚未探索的主要方向 |
| `bc0289a`、`19e1264` | mass、flow、单位质量 loading 等已试；几何常量缺少单位和测量来源 | 不直接堆砌所谓物理特征 |
| `cf21903`、`6e8104f` 的论文核对 | 重构与 release-code-aligned 诊断均不能完整恢复作者 pipeline；对齐代码后，25g V1改善而40g反而变差 | 不把论文差距全部归因于 epoch、seed 或一个代码开关 |
| `02fd01b`、`9667020` | magnitude、width、loading solvent、center→width 等小扩展大多不能稳定兼顾两端和两柱 | 不继续无界组合这些标量项 |
| `92f4a84` | 六系数 M3 Center/Width 超过 Conditional EA：compound 总体 NRMSE 改善约3.84%，9/10配对获益 | Center/Width 是值得保留的工作基础 |
| `8cb739d` | 25g+40g hierarchical Center/Width 再比 M3 改善约3.66%，8/10配对获益，四端点平均RMSE均改善，平均MAE改善约5.83% | 共享低维结构已经有效；默认加入8g没有增益 |
| `8cb739d`、`6e8104f` 的 latent 实验 | 小 adapter、Direct Ridge/PLS、PCA16、joint FULL128 已探索；latent 增量主要益40g、伤25g | 新假设应解释柱间异质性，而非继续扩大维数 |
| `3a226d1` | 修复新 HIER_CW_V2 的单位恢复；拆 C/W λ 和 endpoint objective 仍无稳定增量 | 新方案不能再以“拆开 C/W 正则”作为核心创新 |

历史范围和数值以原始报告为准。[R2][R3][R4][R5] 根 README 仍主要指向低标签 benchmark，`docs/NEXT_STAGE_DECISION.md` 首段仍停留在 M3，因此不能仅阅读导航页判断最新模型状态。

特别说明：**`OLD_HIER_REFERENCE` 本身没有此次发现的单位恢复 bug。** 出问题的是后来 `HIER_CW_V2` 的新 endpoint 实现。旧 HIER 仍是有效且强的参照；“corrected”命名并不意味着旧 HIER 的结果应被废弃。[R6]

**3. 当前值得超过的数字**

下表来自同一 filtered FULL-data、target-compound 划分、五 seeds 的均值，已重新读取 `matched_metrics.csv` 聚合核对。RMSE/MAE 单位均为 mL。FULL 指所有合格的非 validation/test 行参与训练；不包含测试集。[R7]

| 柱 | 方法 | V1 RMSE | V1 MAE | V1 R² | V2 RMSE | V2 MAE | V2 R² |
|---|---|---:|---:|---:|---:|---:|---:|
|25g|paper_style_current_v2|8.041|5.836|0.366|10.959|8.338|0.646|
|25g|OLD_HIER_REFERENCE|6.556|4.463|0.574|8.770|6.667|0.771|
|25g|HIER shared λ corrected|6.595|4.467|0.570|8.978|6.748|0.761|
|25g|corrected FULL128|6.988|4.792|0.513|9.322|7.050|0.743|
|40g|paper_style_current_v2|16.240|11.001|0.650|22.083|15.389|0.647|
|40g|OLD_HIER_REFERENCE|15.426|9.944|0.685|20.113|13.351|0.710|
|40g|HIER shared λ corrected|15.421|9.948|0.685|20.080|13.310|0.711|
|40g|corrected FULL128|15.150|9.688|0.696|19.548|12.937|0.726|

重要判断是：**已有方法确实提高了指标，但一个共同的复杂修正无法稳定服务两个柱。** FULL128 相对 corrected endpoint HIER 在 40g compound 的综合 NRMSE 改善约2.19%、5/5获益；25g 则恶化约4.39%、0/5获益。若参照 OLD_HIER，25g compound 的平均综合 NRMSE 恶化约6.56%。前两项是逐seed相对变化的均值，6.56%是平均NRMSE之比；参照模型和聚合方式均不同，不能混用。[R8]

25g 用旧 HIER、40g 用 FULL128 的事后拼接，只是提出假设的线索；本轮没有把这种测试集选法当作已经有效的新模型。

**4. “同时提高三个指标”的准确含义**

目标是 R² 上升，RMSE、MAE 下降。对完全相同的某个测试集和某个输出，有：

\[
R^2=1-\frac{\mathrm{RMSE}^2}{\frac1n\sum_i(y_i-\bar y)^2}.
\]

所以固定样本后，降低 RMSE 就会提高该端点的 R²；MAE 则需要另外检查。例如同一测试集原 R²=0.70，RMSE降低10%，则新 R²=1−0.9²×0.30=0.757。这是指标恒等式，不是本项目的性能预测。跨 seed 的平均 RMSE 和平均 R²也必须各自实际计算。

当前两个数据范围不能混排：25g reader-valid 490行、filtered 408行；40g为529行、filtered456行。过滤去除了约16.7%/13.8%的有效记录，也缩小标签方差。历史同一套冻结预测，仅改变测试子集就出现25g V1 RMSE 13.007→7.659、R² 0.864→0.405。这不是模型同时变好又变差，而是评价总体变了。[R9]

先在冻结 filtered 总体上定位增量，再在预先定义的无体积阈值总体上报告适用范围。按真实 V1/V2 阈值限定的总体是有条件的回顾性评估，部署时通常无法提前知道真实体积；不能用它代替全范围表现。禁止为了指标重新筛选测试大误差样本。

**5. 优先实验 A：按柱收缩、允许回退的预测修正**

这是当前最值得先实施的低成本计算实验。

代码发现：`hierarchical_cw_corrected.py` 的 latent 只追加一套共享128维列，C/W各有一套系数，没有柱专属 latent 系数。`run_hier_cw_semantic_repair.py` 的选模评分把25g/40g行合并，用 pooled train 的两个 endpoint 标准差归一化；没有按柱分别归一化、等权汇总。40g绝对误差较大时，更容易主导选择。实际 latent λ 网格为1/10/100/1000；关闭 latent 的 `None` 只用于数值嵌套测试，没有作为普通回退候选。[R10][R11]

上述事实与40g受益、25g受损相容，但尚不能断言已经找到了全部原因。

建议先固定两个预测器：H 为 shared-λ corrected HIER，J 为 corrected FULL128。以每柱一个系数做：

\[
\hat{\boldsymbol V}_c
=\hat{\boldsymbol V}_{H,c}
+\alpha_c(\hat{\boldsymbol V}_{J,c}-\hat{\boldsymbol V}_{H,c}),
\quad\alpha_c\in\{0,0.25,0.5,0.75,1\}.
\]

这里 V 是(V1,V2)二元输出；同一柱的两个端点共用α，先限制自由度。α=0精确回退H，α=1还原J。若两个基模型均满足0≤V1≤V2，凸组合也保持该约束。差分包含joint模型引起的全部预测变化，不能把它解释成只隔离了latent的物理作用。

同时测试按柱平衡的拟合与选择目标。对每个训练折，用该柱、该端点的训练标准差 s(c,e) 构造：

\[
L=\frac12\sum_{c\in\{25g,40g\}}\frac12\sum_{e\in\{V1,V2\}}
\frac1{n_c}\sum_{i\in c}\left(\frac{y_{ie}-\hat y_{ie}}{s(c,e)}\right)^2+\Omega.
\]

内层选择按四个“柱×端点”分别计算 normalized RMSE，再等权平均；MAE设置逐端点保护条件。这里只改变迁移拟合和选模，4g源模型的归一化与checkpoint保持冻结。最终仍报告原始mL误差和历史综合评分，以便判断目标改变的影响。

最小消融矩阵如下；避免一次混入新网络、新特征和新损失。

| 对照／候选 | 改动 | 回答的问题 |
|---|---|---|
|参考 R0|保留 OLD_HIER 的有效历史配方|相对现有强基线是否真有增益|
|参考 R1|shared-λ corrected HIER|保证有稳定回退点|
|参考 R2|原选择规则的 corrected FULL128|复核现有25g/40g方向|
|A1|FULL128仅改逐柱平衡拟合、评分|是否主要来自大柱主导目标|
|A2|R1与R2之间按柱选择α|仅增加回退是否有效|
|A3|R1与A1之间按柱选择α|两项改动是否互补|

α不根据旧测试表手工设为25g=0、40g=1。所有选择均来自outer train内的OOF预测：生成某折OOF预测时，基模型及其超参数必须只使用该折之外的数据；若还需选基模型超参数，则在折内再划训练/选择子折。不能把全outer-train已经调好的模型产生的in-sample残差冒充OOF证据。

主分析沿用5个outer seeds、两个柱、compound/row两种评价，共20个测试context。compound与row分别仅使用各自protocol、seed的outer-train进行选择，两个协议的标签与OOF选择不得交叉使用；compound作为主结论。冻结全部预测后统一评分。最终留多少目标标签、供体标签，各方法逐项记录。历史数据只能提供开发性评价。

如果A系列失败，不能直接得出“加更大的MLP就好”。只有训练内证据仍支持不同柱的残差方向不同，才单独测试“共享latent系数+柱专属偏移”的一个有限扩展，并加入偏移归零的强收缩对照。它与已经试过的“C/W分λ”是不同假设。

**预期定位：A主要争取小而稳定的增益、控制负迁移。当前证据不支持它能把所有R²直接推到0.9。** 若25g仅回退而不改善，就应如实称“避免退化”，尚未实现两柱全面提升。

**6. 优先实验 B：4g 实测锚 + 跨柱差分学习**

这一方向的信息增量更明确，适用于已有4g实验记录的化合物。

目前HIER等方法将源模型预测作为输入。若迁移映射接近质量比缩放，源预测误差的一部分可能经6.25或10的尺度放大；实际放大量由拟合映射决定，不能把这两个倍数当成已测误差分解结果。既然大部分目标分子已有4g记录，就应检验实测信息是否比仅压缩进模型权重更有用。

本次用冻结source row-seed-42的**3,330条train记录**做了纯字段匹配，未用source validation/test构建匹配库，也未拟合新模型：

| 在4g源训练集中可获得的信息 | 25g filtered，408行 | 40g filtered，456行 |
|---|---:|---:|
|同canonical SMILES|389（95.3%）|456（100%）|
|同SMILES、同PE/EA|365（89.5%）|436（95.6%）|
|再要求相同loading solvent|365（89.5%）|436（95.6%）|
|再要求相同loading amount与loading solvent volume|254（62.3%）|372（81.6%）|
|再要求相同flow|0|0|

最后一行符合跨柱流量不同的事实，不意味着前面不能作跨柱配对。前面的匹配也不保证实验批次、填料状态或单位质量负载相同。匹配字段是元数据，不是“新模型已可达到的准确率”。来源与校验和见第11节。

建议按以下次序构建输入锚：

1. 同分子、同EA、同loading条件时，用合法source-train记录的4g实测值构造锚；重复记录保留原始身份，先估计重复差异，再决定如何汇总。
2. 无完全匹配时，先回退原4g预测。只有source-train内部留出条件实验支持插值有效，才加入同分子、同loading solvent的EA保留曲线；不盲目做最近邻跨条件替代。
3. 可实现为原source预测加一个收缩的实测残差修正：`a(x)=f4(x)+w(x)·estimated_source_residual(x)`。w由匹配可靠度和训练内验证控制；无支持时w=0。拟合source残差时，必须在每条source记录自己的条件上计算f4。
4. 将a的Center/Width送入相同HIER迁移结构，与使用f4预测作为锚的版本比较。首轮保持迁移模型、目标标签及正则选择规则相同，只替换锚。A、B分别验证后才考虑组合。

这与历史source replay不同：replay通过4g损失约束训练参数；本方案在预测时明确使用该化合物可获得的4g实测信息。在审阅的当前迁移主线中，没有看到这种实测源锚作为目标推断输入的对照。

最低限度要报告三个版本：原预测锚、仅精确匹配实测锚、通过源域验证后才启用的插值/收缩锚。各版本都覆盖全部测试记录；匹配缺失必须回退，不能只报容易匹配的子集。另报告匹配/未匹配两组表现，定位收益来源。

**应用前提必须写清：有4g历史记录的已知分子换柱。** 若部署只有SMILES而没有该分子的4g记录，则自动回退原预测器。此设置不能与完全不提供实测锚的新分子模型混称；4g实验成本即便本项目已支付，也要披露。若提出真正source-unseen任务，测试分子的4g标签不能暗中进入锚库。

这一“基准预测+学习差分”的思路与[Δ-learning原始工作](https://arxiv.org/abs/1503.04987)在方法形式上相近；跨领域类比不提供本项目效果保证。与本领域更接近的[TLC–CC关系研究](https://www.nature.com/articles/s41467-025-56136-x)说明色谱实测锚值得检验，但不能将其公式或收益直接套到这里。

**7. 验证集和任务定义必须一起处理**

当前FULL compound每个context的目标训练量约为25g 306–334行、40g355–367行，验证集只有6–7行、1个化合物。[R9] 这会使神经网络early stopping高度依赖单个分子的表现。因此后续神经对照若重做，应先用训练内分组CV确定训练长度/设置，再用全部允许训练行重训；不先扩大网络。

同时，当前compound只保证**单柱内部**train/validation/test的分子不重叠。最新联合模型直接拼接两个柱各自的gradient_train。重新检查split manifest得到：五seeds累计，25g测试430行中188行、40g测试441行中250行，其分子在另一柱训练集中出现。相同测试行标签并未直接用于训练，但这属于有供体监督的目标柱留出；不能叫所有柱都未见的新分子。[R11][R12]

建议明确保留两种任务：

- **应用任务：已知4g分子迁移到目标柱。** 与论文场景接近。允许事先定义的源域信息；供体列是否可包含相同分子必须固定，并计入标签成本。旧frozen splits用于历史配对。
- **更严格的泛化任务：在所有目标柱上同时留出同一组分子。** 按canonical SMILES联动25g/40g的outer roles；所有比较方法使用相同角色。4g仍可作为已知源域。若进一步要求源域也未见，则需要全域联动划分并重训source，或独立新分子确认集。

不能把更严格划分的指标变低叫作算法退步，也不能因原文没有该任务就放弃报告。两类任务回答的是不同的问题。

重新划分已经看过的数据仍是开发性证据。所有方案、超参数网格、回退规则和成功标准应在新评分前固定；独立新批次才提供更强确认。嵌套选择的必要性可参照[小样本交叉验证方法研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC3994246/)。

**8. 损失与数据质量：有必要控制，但不应包装成已解决的瓶颈**

`adaptation.py:109`及`training/predictor.py:75`显示，名为q50的中心输出已经用平方误差训练，q10/q90才用pinball，还叠加crossing惩罚。[R13] 因此“把q50的MAE换成MSE即可提高R²”不符合当前代码；也不能仅凭q50名称就断定它是条件中位数。

若A/B之外仍需一个神经训练控制，建议限定为：固定相同backbone、训练范围、初始化与分组CV，比较原混合quantile loss和仅中心输出的标准化MSE；检查辅助分位数梯度是否妨碍点预测。该实验尚不能预期为正；旧8g损失尺度对照已经显示改权重的点预测收益很小。[R14] Huber、MAE混合或log-volume可能改善部分样本，也可能损失尾部RMSE，不作为默认“全指标提升”配方。

数据方面优先核查25g的79条invalid记录及有效数据中的t1>t2标记，追溯原始UV记录、采样索引到时间和体积的换算、标注规则及重复实验。[R15] 无法确认的标签做质量标记和独立敏感性分析；不按模型残差大小自动删除。当前无阈值大柱误差具有显著尾部贡献，但这不能直接证明尾部全是噪声。[R16]

如能补实验，先做小规模重复性验证：例如6个覆盖主要EA/保留体积区间的化合物×2个EA条件×2个柱×2次独立运行，共48次。化合物选择用训练信息和预定覆盖原则，保留批次/柱批号；这是粗略估计噪声和批次效应的试验设计，不是足够支撑所有物理机制的样本量结论。

若需要进一步扩展输入，TLC Rf、实测hold-up/void volume、柱批次与实际填料参数，比重复使用未经核验的dia/len/den更值得优先验证。mass×flow交叉实验用于分离混杂，不能仅凭每柱一个流量就拟合普适流速规律。[R17]

**9. 成功标准与停止条件**

建议先约定“有价值增量”和“完成全面提升”两级标准，避免把一个平均分数当成全部成功。

| 级别 | 预设标准建议 |
|---|---|
|值得保留继续验证|相对有效HIER参考，四个柱×端点的平均相对RMSE改善≥3%；每柱综合RMSE至少4/5 seeds获益；任何端点平均RMSE/MAE不得恶化超过2%|
|满足本轮全面提升目标|两个柱的V1/V2平均RMSE、MAE全部下降，平均R²全部上升；四端点平均相对RMSE和MAE改善均≥5%；逐柱稳定性仍须通过|
|可以写成较强论文结论|在独立新批次／预先留出的确认总体上复现，报告配对不确定性，并清楚区分与论文原图和可运行对照的比较|

这些3%/5%是建议的研究决策阈值，不是已知性能预测或普适显著性标准。正式执行前应根据实验容忍误差固定。若只有40g受益，结论可以是柱特异改进，但不能声称两个柱全面提升。

以OLD_HIER历史均值为尺度参考，若每端点RMSE都下降5%，对应25g约6.23/8.33 mL、40g约14.65/19.11 mL；同样MAE下降5%对应约4.24/6.33、9.45/12.68 mL。这些仅是旧总体上的规划目标，不能移植为新划分应达到的阈值。

报告每seed/每柱/每端点的R²、RMSE、MAE以及均值、标准差、配对差值；补充按化合物聚合误差、训练定义的EA/高低体积分层、错误区间顺序比例及绝对误差分位数。置信区间按化合物/实验批次聚类估计；不同seed的重复测试行不能当成彼此独立的新样本堆叠。5个seed的win count是稳定性描述，不等同统计显著。

若训练内看不到可重复的改进，停止该候选，保留强基线并记录负结果；不在观察测试后扩大网格、换seed、切换指标或删样本。如果A/B均未提供增量，再依据随机学习曲线判断是否值得补标签，而不是直接启动主动学习。主动学习针对给定标签预算的效率，不会凭空提高已经用完同一训练池的FULL-data指标。

**10. 实施顺序与代码落点**

| 顺序 | 工作 | 主要交付物 | 进入下一步的条件 |
|---|---|---|---|
|S0|锁定输入、有效HIER参考、任务语义和逐方法标签账本；统一分组CV入口|可复核的protocol、split与source hash、现有基线再现表|原预测/数据身份能准确还原|
|S1|完成A1/A2/A3有限消融|OOF预测、按柱α、拟合目标对照、20 contexts配对报告|至少有候选满足预设保留条件，或明确否定该假设|
|S2|先实现B的精确实测锚与缺失回退，再视源域验证决定是否插值|锚覆盖表、source标签使用账本、三个锚版本对照|源域条件验证可靠，且目标CV不显示系统退化|
|S3|对保留方法做联动target-compound划分及独立批次确认|按信息可用性分开的最终比较表|不靠供体同分子或旧test选择解释全部收益|
|S4|有需要再做固定预算随机学习曲线和一个point-loss控制|标签收益曲线／训练目标消融|已知瓶颈支持该方向|

S1和S2是两个独立假设，S1失败不妨碍执行有明确输入依据的S2。两者分别验证之后才增加组合候选；不把所有改动一次打包，导致无法解释收益。

现有可复用入口为 `QGeoGNNV2.extract_representation`、`hierarchical_cw_corrected.py`、`full_data.py` 和 `run_hier_cw_semantic_repair.py`。[R10][R11][R12][R18] 未来可新增 `selective_column_transfer.py`、`source_measurement_anchor.py` 及各自study runner；这些名字是拟议文件，目前未创建。

工程检查应覆盖：α=0/1的数值还原、分组互斥、source锚不读取禁止角色标签、修改test truth不能改变已拟合模型、逐折scaler来源、样本ID对齐、有限输出及物理顺序。只验证数值与数据契约，不能用单元测试代替科学效果评价。

当前运行时在`build_predictor`中限定CPU；首轮应复用现成128维表示和小规模线性拟合，不应把迁移到GPU或新架构加入同一实验。计划没有对运行时长作未经测量的承诺。[R18]

**11. 本次新算的审计如何复核**

实测锚覆盖率的计算：从source `row_seed_42.csv`取`split=train`的sample IDs，与`canonical_4g.csv`按ID相交；对每个目标filtered feature行检查字段元组是否存在于该source-train子集。数值字段转float后比较，其余用原字符串；SMILES已是canonical值。只检验可获得的条件支持，不进行标签拟合。

供体overlap的计算：在`full_data_baseline_finalization/split_manifest.csv`内，固定protocol=compound及outer_seed，取目标柱test行SMILES，检查其是否在另一柱gradient_train的SMILES集合中。统计按seed报告后汇总；430/441是五seed测试行计数之和，并非独立实验样本数。

| 审计输入 | SHA256 |
|---|---|
|4g canonical|`6cbe81311177ae178d10ff75a8f07c41c99a18648ec6104d2ffedca6e912d51c`|
|source row-seed-42 split|`a8a4a322cd9c75253b1a4ed8c5c91ededed74a933df2db2685015c8b3e008008`|
|25g filtered features|`3733df28f6c14d21563e02d8fc04a37ce4838b17a46303986b6da65719ab07c9`|
|40g filtered features|`e64e09686ebb52b9d1bd276354318d565b15dc7b3910dd65dc60174436ee5720`|
|latest matched_metrics|`9413c2e0cb65c038a9c42745d8198ed59bc7ef2e9efe3ba2f98273a0311d5e4f`|

本次没有训练A/B候选，也没有利用测试误差拟合α、权重或实测锚。历史测试结果用于提出研究假设，后续收益仍待验证。

**12. 证据索引**

- [R1：论文pipeline差异与对齐诊断][R1]
- [R2：跨柱残差诊断][R2]
- [R3：source replay与微调结果][R3]
- [R4：M3 FULL-data baseline][R4]
- [R5：hierarchical结构与latent容量比较][R5]
- [R6：新旧HIER单位语义审计][R6]
- [R7：最新同协议指标][R7]
- [R8：corrected FULL128逐柱收益][R8]
- [R9：过滤总体、FULL标签量、单分子验证集][R9]
- [R10：共享latent设计与归一化实现][R10]
- [R11：联合训练、内层评分和latent候选选择][R11]
- [R12：单柱compound互斥检查][R12]
- [R13：当前实际损失][R13]
- [R14：历史8g损失尺度控制][R14]
- [R15：数据质量与source分子重叠审计][R15]
- [R16：无阈值匹配RMSE及尾部结果][R16]
- [R17：物理元数据可辨识性][R17]
- [R18：当前预测器、128维表示及CPU运行契约][R18]

[R1]: /Users/fish/Documents/GitHub/column_chromatography_prediction/studies/transfer/paper_transfer_gap_decomposition/paper_pipeline_preprocessing_comparison.md
[R2]: /Users/fish/Documents/GitHub/column_chromatography_prediction/studies/transfer/residual_diagnostics/RESULT_INTERPRETATION.md
[R3]: /Users/fish/Documents/GitHub/column_chromatography_prediction/studies/transfer/source_anchored_shared_transfer/RESULT_INTERPRETATION.md
[R4]: /Users/fish/Documents/GitHub/column_chromatography_prediction/studies/transfer/full_data_baseline_finalization/FINAL_REPORT.md
[R5]: /Users/fish/Documents/GitHub/column_chromatography_prediction/studies/transfer/full_data_architecture_headroom/FINAL_REPORT.md
[R6]: /Users/fish/Documents/GitHub/column_chromatography_prediction/studies/transfer/hier_cw_semantic_repair/IMPLEMENTATION_AUDIT.md
[R7]: /Users/fish/Documents/GitHub/column_chromatography_prediction/studies/transfer/hier_cw_semantic_repair/FINAL_REPORT.md
[R8]: /Users/fish/Documents/GitHub/column_chromatography_prediction/studies/transfer/hier_cw_semantic_repair/joint_full128_reaudit.md
[R9]: /Users/fish/Documents/GitHub/column_chromatography_prediction/studies/transfer/filtered_full_data_benchmark/FINAL_REPORT.md
[R10]: /Users/fish/Documents/GitHub/column_chromatography_prediction/src/qgeognn_al/transfer/hierarchical_cw_corrected.py:101
[R11]: /Users/fish/Documents/GitHub/column_chromatography_prediction/scripts/studies/run_hier_cw_semantic_repair.py:85
[R12]: /Users/fish/Documents/GitHub/column_chromatography_prediction/src/qgeognn_al/transfer/full_data.py:161
[R13]: /Users/fish/Documents/GitHub/column_chromatography_prediction/src/qgeognn_al/transfer/adaptation.py:109
[R14]: /Users/fish/Documents/GitHub/column_chromatography_prediction/experiments/e0_3c_loss_controls/README.md
[R15]: /Users/fish/Documents/GitHub/column_chromatography_prediction/studies/transfer/cross_column/data_audit/DATA_AUDIT.md
[R16]: /Users/fish/Documents/GitHub/column_chromatography_prediction/studies/transfer/matched_rmse_benchmark/MATCHED_RMSE_REPORT.md
[R17]: /Users/fish/Documents/GitHub/column_chromatography_prediction/studies/transfer/physical_metadata_identifiability_audit/FINAL_REPORT.md
[R18]: /Users/fish/Documents/GitHub/column_chromatography_prediction/src/qgeognn_al/models/qgeognn_v2.py:55
