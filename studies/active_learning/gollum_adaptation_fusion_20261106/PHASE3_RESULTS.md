# Phase 3：加性核实际开发实验结果

本轮已完成32次真实 exact GP fit：seeds157/887 × L333/L813 × V1/V2 × 4方法。全部保留，32次获得 SciPy bounded optimizer success，0数值失败。加性核相对于 GNN-only 的均值收益很小，绝对概率质量仍明显欠覆盖；没有证据支持升级为可信概率专家。训练学到的语言权重总体很小，且等义文本改变明显影响候选方差与部分采集排序。这里仍使用旧冻结Qwen，不是T5或LoRA。

## 口径与数据边界

所有方法使用相同当前 L_t、历史 GNN checkpoint、开发 validation416和合法候选256；只读取已封存缓存内的当前训练/开发响应，无test或未来U标签查询，不训练新GNN或新语言编码器，不启动闭环AL。每个state/cache/checkpoint/行顺序、LLM缓存均重新校验SHA256；候选顺序与旧研究完全相同。详见`phase3/input_audits.json`及每状态audit。

416行validation此前参与GNN checkpoint selection，只能构成开发性证据；没有使用它选择λ或其他超参数。核标量只由当前 L_t exact MLL学习。新实验的单位核幅度与参数边界预先固定，见设计和`phase3/config.json`。每种方法只有初始化17；冻结核不含随机projection，但多起点可能产生其他局部极值，因此不能称为跨训练初始化稳定。两个seed、两个预算和两个端点具有相关性，8个端点结果不是8个独立统计重复。

## 预测和概率结果

下表为同一矩阵8个端点的描述性均值。nRMSE按历史L0各端点标准差归一化；RMSE、MAE、区间宽度单位为ml。NLPD越低越好。区间是含独立observation noise的预测区间。

| 方法 | nRMSE | RMSE | MAE | NLPD | 90%覆盖 | 95%覆盖 | 90%区间宽度 | 标准化残差RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GNN λ=0 | 0.554262 | 6.0251 | 3.0644 | 23.1411 | 55.08% | 59.65% | 3.4562 | 6.0605 |
| 加性 fixed λ=1 | 0.553027 | 6.0114 | 3.0448 | 22.3858 | 55.32% | 60.31% | 3.4236 | 5.9766 |
| 加性 learned λ | 0.553637 | 6.0180 | 3.0534 | 23.3548 | 55.02% | 60.10% | 3.4222 | 6.0931 |
| GNN双核容量对照 | 0.553392 | 6.0179 | 3.0614 | 23.0819 | 54.87% | 59.71% | 3.4059 | 6.0447 |

固定λ的nRMSE平均改善约0.00124（相对约0.22%）；learned λ改善约0.000625（相对约0.11%），却使平均NLPD变差0.214。容量对照本身也能取得类似量级的均值收益，因此这些微小变化不能证明语言先验提供稳定新增科学信息。覆盖率约55%而目标是90%，标准化残差RMS约6；有界优化成功没有带来概率校准。

保留每seed/预算、端点平均后的相对GNN配对差：

| Seed | 预算 | fixed1 ΔnRMSE | fixed1 ΔNLPD | learned ΔnRMSE | learned ΔNLPD | 容量对照 ΔnRMSE |
|---|---:|---:|---:|---:|---:|---:|
| 157 | 333 | -0.001278 | -2.169669 | -0.000173 | +0.185505 | 0.000000 |
| 157 | 813 | -0.000889 | -0.417777 | -0.000414 | +0.604830 | 0.000000 |
| 887 | 333 | -0.000421 | -0.406715 | -0.000105 | +0.026672 | -0.003482 |
| 887 | 813 | -0.002354 | -0.026837 | -0.001810 | +0.037678 | 0.000000 |

learned λ在四个配对状态中NLPD均略变差；固定λ的NLPD变化幅度主要来自seed157/L333。最差seed/预算为157/L333：GNN nRMSE0.859833、NLPD41.119976、90%覆盖44.95%；seed157/L813的90%覆盖只有34.74%（fixed1还降到33.77%）。完整端点与95%区间、宽度保存在`metrics.csv`、`predictions.csv`、`paired_differences.csv`；`per_seed_budget.csv`与`per_budget.csv`保留分层结果。

## λ、尺度和数值条件

learned λ逐状态/端点为：

| Seed | 预算 | V1 λ | V2 λ |
|---|---:|---:|---:|
| 157 | 333 | 0.004294 | 0.002813 |
| 157 | 813 | 0.014923 | 0.008139 |
| 887 | 333 | 0.035709 | 0.002258 |
| 887 | 813 | 0.028973 | 0.000123 |

8个λ均值0.012154；seed887/L813/V2命中exp(-9)下界。λ=0单独保留且严格嵌套GNN-only；learned分支不能取精确零。固定λ=1时，7/8个语言长度尺度命中exp(5)=148.413上界，其语言分量在当前尺度上趋近常量核，不能把非零固定权重当成模型利用了语言结构。learned λ平均训练MLL为-787.4266，高于GNN-only的-790.1979，但开发NLPD更差，再次说明优化训练MLL本身不足以证明概率可靠。

32次都在11–37次迭代内结束，未触及预置200迭代上限。全部 numerical jitter为0、无Cholesky失败。观测协方差condition范围约6.90e4–4.91e7；最大值来自learned fusion。按方法noise variance均值分别为GNN1.12675、fixed1 1.04187、learned1.06837、容量control0.94997 ml²；这些值和signal/latent尺度、每fit noise ratio完整保留在`fits.json/csv`。

“32/32成功”严格指SciPy在边界或函数变化准则下的终止状态，不代表全局最优或所有无约束梯度小于gtol。语言长度尺度上界的无约束梯度较大，说明当前约束下进一步增加语言平滑度可能继续改善训练目标；这里按冻结边界保留结果，没有根据validation扩边重跑。所有loss、梯度范数、迭代、消息和boundary索引均保存。

## 与旧Early Fusion比较

同4状态的历史结果只读复用，没有重跑旧Projection：

| 历史对照 | nRMSE | NLPD | 90%覆盖 | optimizer success |
|---|---:|---:|---:|---:|
| Frozen GNN Matérn | 0.554262 | 23.14108 | 55.08% | 8/8 |
| Frozen Qwen Matérn | 0.85598 | 3.65842 | 94.89% | 8/8 |
| Frozen PCA+concat Matérn | 0.55158 | 21.19820 | 56.13% | 8/8 |
| concat Projection init17 | 0.55502 | 577.45146 | 14.24% | 0/8 |
| concat Projection init29 | 0.55326 | 630.82093 | 13.04% | 0/8 |
| concat Projection init43 | 0.55251 | 611.92834 | 13.79% | 0/8 |

当前加性核比旧40迭代Projection的概率质量好得多，但不优于旧Frozen concat；不能称“加性优于简单拼接”。旧Frozen Qwen覆盖较高伴随约36ml的90%区间宽度和更差均值预测，不能只按覆盖率认定更优。新加性核使用原始block RMS，旧concat采用train-only PCA64，且Projection训练预算/容量不同；上述是历史参照，不是唯一改变融合算子的因果实验。来源与SHA256见`historical_references.csv`和`historical_source_hashes.json`。

## 离线采集：绝对校准与相对排序分开

相同候选池上，相对GNN-only的平均候选latent variance Spearman分别为fixed1 0.9602、learned0.9800、容量control0.9667。各方法的batch32集合重叠：

| 对照GNN | MaxVar | Conditional IVR | Greedy BatchIG |
|---|---:|---:|---:|
| fixed1 | 89.84% | 84.38% | 89.06% |
| learned | 92.97% | 90.62% | 92.97% |
| 容量control | 96.88% | 96.88% | 97.27% |

这说明同一文本模板下大部分探索排序与GNN接近，绝对欠覆盖并未导致所有排序完全不同；也说明语言模块带来的候选改变较少。批内后验绝对相关均值约0.055–0.079，平均覆盖约26.25–27.88个分子和31.50–31.75个精确条件组合。三种采集的所有ID、conditional trace、模型内信息量、分子覆盖及精确同分子条件匹配都保留在`acquisitions.json`、`selection_diversity.csv`。不能跨模型把更高GP信息增益解释为更高真实实验价值；本轮无候选真值或下游QGeoGNN收益验证。

模板鲁棒性使用A训练、B仅推断，平均结果：

| 方法 | 方差秩ρ | MaxVar重叠 | IVR重叠 | BatchIG重叠 | 均值绝对变化ml | latent方差A→B均值ml² | 方差绝对变化ml² |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed1 | 0.0677 | 31.64% | 41.41% | 84.77% | 0.5863 | 0.3160→5.8347 | 5.5187 |
| learned | 0.4027 | 53.13% | 39.45% | 91.02% | 0.3103 | 0.2773→2.8812 | 2.6038 |
| GNN与容量control | 1.0000 | 100% | 100% | 100% | 0 | 完全相同 | 0 |

这里不只是近乎相同方差间的微小秩交换：绝对方差均值也明显扩大，A/B使用相同冻结训练变换，没有B拟合。MaxVar/IVR对措辞较敏感，BatchIG的批集合则较稳定；三者不可混同。该结果指向旧Qwen表示及跨模板分布位移问题，不能外推到尚未运行的T5/LoRA。仍需检查适配后的模板鲁棒性，不能因单模板的ρ≈0.98就认定语言选点稳定。

## 资源、测试与可复现性

本机现有chromatography环境、CPU单线程、float64。32fit合计55.13秒（含condition计算），A采集合计3.66秒、B采集合计2.77秒；这些是分项计算时间，不含全部文件哈希/I/O。按方法fit时间为GNN11.81、fixed1 13.33、learned16.91、容量control13.09秒。额外32个冻结posterior精确重建与B诊断耗时16.58秒，不优化参数。无模型下载、依赖安装、付费API或新编码器forward。

`test_kernel_fusion.py`共12项通过（7.10秒），覆盖λ0与旧GNN严格嵌套、单位对角PSD、五个MLL标量有限差分、latent/observation分离、容量control语言独立、训练专属变换、非法输入、重复行、state/feature身份、test索引拒绝、cache哈希篡改拒绝、posterior恢复一致及seed元数据回归。`phase3/reproduction.json`确认32个posterior按冻结参数重建后，416行均值/方差与已存结果差异不超过1e-10；无重拟合。

发现并修复一处本轮结果元数据错误：合并fit audit时优化初始化seed17覆盖了outer trajectory seed。模型数值、预测、指标、采集身份不受影响；现将outer seed与optimizer_seed分开。`metadata_repair.json`记录全部32条原始/修复SHA，新增回归测试；后续B诊断补充的前后SHA保存在`reproduction.json`。未静默改历史研究或按表现删结果。

可重运行命令（项目根目录执行；runner复用已完成条目，不覆盖旧研究）：

```bash
env KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/fish/miniforge3/envs/chromatography/bin/python studies/active_learning/gollum_adaptation_fusion_20261106/run_phase3.py
env KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/fish/miniforge3/envs/chromatography/bin/python studies/active_learning/gollum_adaptation_fusion_20261106/reproduce_phase3.py
env KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/fish/miniforge3/envs/chromatography/bin/python -m pytest -q studies/active_learning/gollum_adaptation_fusion_20261106/test_kernel_fusion.py
```

决定：本轮加性核实现和数值测试通过，科学概率门槛未通过。对“比Frozen concat更稳定/更优”“语言核有可靠增量”“可直接进入正式QGeoGNN AL”均为No-Go。保留候选排序探索的条件性用途，但需先改善表示/模板鲁棒性并在合法训练外数据验证概率或排名信息价值。Residual为Not Yet Testable，Factorized仅设计；不自动增加融合结构或启动AL。
