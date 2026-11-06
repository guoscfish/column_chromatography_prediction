# 表示、GP 与离线采集的数学定义

本文件随 PROTOCOL 在效果评分前固定。符号：当前合法标注集 L、开发评价集 V、无标签诊断候选 C⊂U；q 表示 V1/V2 端点，d 为表示维度。所有标签相关学习只用 L。以下 GP 不表示 QGeoGNN 风险概率；也不默认概率校准成立。

## 表示

Structured 使用已有 RDKit Morgan radius2分子指纹与实际记录条件。PE/EA=a/b 转换为 EA fraction=b/(a+b)的同时保留原始配比文本；数值条件由 L 定标，loading solvent one-hot，不将类别数字当有序距离。密度 g/mL×样品体积 μL 的数值单位恰为 mg；样品 loading volume 与 loading solvent volume 是两个条件。

GNN Hidden 为 `h(x)=sum_nodes backbone(x)+condition_residual(x)`，位于六输出 head之前，128维；冻结checkpoint并eval，不更新458,952个 QGeoGNN 参数。Raw 为 `CountSketch([∂(q50_V1/s1)/∂w, ∂(q50_V2/s2)/∂w])`，512维；CW 将输出坐标换成 `[(V1+V2)/(2sC),(V2−V1)/sW]` 再用同一既有 linear-output 梯度提取接口。s1/s2/sC/sW只由该seed初始L333响应决定。Raw与CW是参数敏感性，不等于隐藏层表示；不得把Raw CountSketch结果线性后处理冒称CW，因为两端点的 sketch 分块映射不同。

语言表示使用已有预训练 Qwen2 decoder base model，移除不需要的 LM output head。每token最后层 hidden为 H∈R^(T×1024)，attention mask为 a；`e(x)=Σ_t a_t H_t / Σ_t a_t`。模型fp32 CPU eval，全部 requires_grad=False、inference_mode；均值池化及结果为float64。plain text直接 tokenize，不使用chat template、不generate、无dropout、不截断、不L2 normalize。T5接口若未来已有合法本地权重，则仅load T5EncoderModel，不调用decoder；本次T5不可用。

两个文本模板保留同一canonical SMILES、silica gel、4g柱、原PE/EA、原density、原样品体积、原loading solvent、原加载溶剂体积及单位，A为字段列表，B为等义句式。派生mass由Decimal乘法得到。只接受feature白名单，缺失/非有限数值、预测或标签字段拒绝。SMILES-only保留分子并去掉全部实验条件；同分子不同条件向量相同是此基线的已知限制。

固定表示统一 `x=(e−μ_L)/s_L`，`s_L=sqrt(mean_ij(e_ij−μ_Lj)^2)`。Fusion中分别按L_t处理GNN和LLM，再各取训练侧PCA64，拼接128维。容量控制GNN原隐藏128→64与Fusion128→64同有128×64+64=8,256个参数；Qwen1024→64有65,600个参数，不能与前者声称参数数目相同。Projection为 `z=ELU(xW+b)`，输入维数真实推断，输出64。

## 精确 GP

每端点独立，`y_std=(y−mean_L y)/std_L y`；固定零标准化均值，相当于原尺度 mean_L y。固定核或Projection核：

`r(x,x')=sqrt(||z(x)−z(x')||²/d_z)/ell`，

`k_52=a²(1+sqrt(5)r+5r²/3) exp(−sqrt(5)r)`，

`k_linear=a² z(x)^T z(x')/d_z`，`C=K_LL+sigma_n² I`。

`NLL=0.5(y−m)^T C⁻¹(y−m)+Σ log diag(chol(C))+n/2 log(2π)`。

Cholesky与solve计算该式，未显式求逆。最小化NLL联合拟合ell、a²、sigma_n²及可选W/b；固定projection初始化seed分别独立拟合，禁止由评价分数择seed。训练边际似然是优化目标，不是外样本校准评分。独立端点不估计V1/V2联合响应协方差。

`μ_B=m+K_BL solve(C,y_L−m)`，

`Σ_B=K_BB−K_BL solve(C,K_LB)`，

`v_latent=diag(Σ_B)`，`v_observation=v_latent+sigma_n²`。

NLPD和预测区间使用observational variance。`NLPD_i=0.5[log(2πv_obs,i)+(y_i−μ_i)²/v_obs,i]`，90%/95%区间分别为 `μ±1.64485362695 sqrt(v_obs)`、`μ±1.95996398454 sqrt(v_obs)`。目标标准化后将均值、variance及noise还原至mL/mL²，原尺度MLL包含目标变换Jacobian。逐状态记录latent/observation variance，不能混用。

数值jitter沿用Phase0稳定Cholesky并逐拟合报告；jitter只辅助求解，不作为观测噪声。微小浮点误差之外的非PSD拒绝。经验超参数为plug-in，不积分其不确定性；同方差noise可吸收表示失配，这可能造成校准问题。

## 距离、条件差分和文本扰动

同一状态先按训练侧尺度变换，再报告欧氏距离及按维数平均距离。严格单条件对(A,B)只来自开发评价集，`E_contrast,q=abs((y_Bq−y_Aq)−(μ_Bq−μ_Aq))`；同时保存QGeoGNN的同式。只比较小距离不充分，必须结合真实差分、差分误差和重复条件/分子依赖。单轴变化的幅度和单位不相同，因此不同轴距离比不是物理敏感度导数。

模板B只替换输入，不重新拟合GP或调整预测方差；保存相同实验两文本的距离、最近邻重叠及预测/方差/采集变化。均值扰动mL与归一化扰动并报，避免表示坐标比例影响解释。smiles-only的同分子ties固定ID打破。

## 离线数值采集

给定当前L后，Σ表示C候选潜变量联合协方差，noise为σ²。MaxVar按diag(Σ)排序。已选B之后条件化：`Σ^B=Σ−Σ_:B solve(Σ_BB+σ²I,Σ_B:)`。贪心IVR单步 `gain_i=Σ_Ri²之和/[|R|(Σ_ii+σ²)]`，R为冻结均匀256候选reference；每选择一步更新联合协方差，不能只拿首步分数Top32假装conditional batch选择。

Batch Information Gain为 `IG(B|L)=0.5 logdet(I+Σ_BB/σ²)`，Cholesky计算，单位nats。等价的贪心边际增益为 `0.5 log(1+var(f_i|L,y_B)/σ²)`。保留非对角相关；单点信息之和仅可作忽略冗余的上界诊断。两个独立端点逐端点报告；若求和明确其独立假设。该信息量是指定GP内 y_B 对 f_B 的互信息，不是对QGeoGNN重训误差下降的保证。

## 与GOLLuM的关系

[GOLLuM正式论文](https://www.nature.com/articles/s42256-026-01283-z)将语言表示与GP边际似然耦合；PLLM为冻结语言向量后的可学习投影，φ变体另外训练语言模型。本次只做PLLM启发的Projection，不做LoRA/encoder fine-tuning；本地Qwen与论文T5也不同。其[官方PLLM配置](https://github.com/schwallergroup/gollum/blob/main/configs/pllm.yaml)使用平均pooling、64维Projection及Matérn-5/2，目标为单点EI优化；本任务目标是响应学习，采用MaxVar/IVR/BatchIG离线诊断，不能照搬其发现最优值指标为AULC证据。

GP基本条件分布参考[GPML第2章与第5章](https://gaussianprocess.org/gpml/chapters/)。本次方法及概率结论仍以仓库真实执行为准。
