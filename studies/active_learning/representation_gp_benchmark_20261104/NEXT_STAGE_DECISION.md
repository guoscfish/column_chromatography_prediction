# 下一阶段研究裁决

日期：2026-11-04。**NO_GO_FOR_FORMAL_LLM_GP_ACTIVE_LEARNING**。本次停止于离线研究，不启动正式主动学习。

本次完成完整20状态开发矩阵，共720次端点GP拟合。所有结论来自已有LCMD轨迹、L_t训练与既有validation开发评价；四预算与五seed相关，validation参与过QGeoGNN checkpoint selection，不能称独立测试或AL确认。

当前证据支持继续研究可验证的数值表示与概率模型，但不足以把本地LLM+GP升级为正式选样方法。该判定针对本次明确冻结的Qwen方案；不等于否定T5、其他encoder或整个GOLLuM思路。

## 主要依据

1. 固定Qwen conditions的主误差为0.8776，Structured为0.7663，GNN hidden为0.5053。本地语言先验是否增加信息，应看这组非语言对照，而非只比SMILES-only。
2. Qwen Projection相对Frozen的配对主误差变化为-0.2333。适配后主误差为0.6443，低于Structured的0.7663、仍高于GNN的0.5053；这两个方向均在五个seed均值上一致。但其90%观测区间覆盖为22.8%、NLPD=86.1771；Frozen对应95.4%、3.6576。均值收益不能抵消概率可靠性失败。
3. GNN fixed的90%覆盖也只有68.6%，说明问题不限于语言模型。用监督训练后的表示拟合GP，得到的条件后验没有自动包含整个QGeoGNN预测器的泛化误差；这是与结果一致的机制解释，尚非唯一因果诊断。
4. Fusion Projection相对参数匹配GNN Projection的主误差变化为-0.0053；Fusion自身90%覆盖为21.8%。两个小头均8,256参数，但Fusion多了train-only PCA与block scaling，不能把全部均值差唯一归于语言新增信息。
5. 360/720个有限拟合未在冻结优化预算内收敛；全部按规则保留，没有按validation增加迭代或重启。模型内信息量和采集排序差异不能弥补这一可靠性缺口。
6. 等义模板改写的V1/V2 MaxVar Top32重叠为9.4%/9.7%。冻结计算的重复性与表述鲁棒性是不同要求；本次只核验conditions A/B，未把模板B作为调参集。

## 八个研究问题的回答

**1. LLM是否比传统化学与条件编码提供更多价值？** 当前固定本地Qwen的比较见上述主误差及配对seed表。完整数值支持的收益与负结果都保留；本次没有T5结果，不能外推所有语言模型。SMILES加入条件有帮助也只证明它利用了一部分实验信息，未证明优于结构化编码。

**2. Hidden与Gradient各有什么优势和不足？** Hidden直接继承任务监督，128维成本低，均值预测信息强；它的GP覆盖仍不足。Raw/CW描述全参数输出敏感性、512维且可精确绑定缓存，适合审计覆盖与相似性，但不是响应预测值或可靠不确定性。上样量方向的压缩必须与真实差分及预测差分误差共同解释；加载溶剂体积仅3对/2分子，无法作稳定结论。

**3. MLL学习Projection是否带来可重复未见行收益？** 三初始化全部平均后，Qwen Projection相对Frozen在五个seed均值上均改善；适配后也均优于Structured、仍均逊于GNN。这是可重复的开发均值收益，但严重欠覆盖和未收敛阻止“更可靠概率模型”的结论；不能以训练NLL低或挑单初始化证明可复现。

**4. 复杂表示是否过度自信？** 本次结果支持该风险：Projection观测方差与真实误差不匹配、NLPD和覆盖恶化。noise是surrogate条件似然参数，不能直接宣称为真实实验测量噪声。没有事后用validation缩放方差。

**5. Fusion收益是否超过容量解释？** 已做同参数数目的小头控制，仍有PCA/scaling与输入组成差异；需共同查看Frozen Fusion、参数匹配Projection及seed方向。当前概率结果不足以将Fusion升级为可靠互补性证据。

**6. 是否足以进入LLM+GP正式AL？** 不足；当前为No-Go，未计算任何新AL学习曲线或AULC。

**7. 最小且最有价值的后续核验？** 优先修复和核验概率模型，而非先跑昂贵AL：固定既有早期checkpoint及当时L，仅使用已授权、历史上已观测且模型未拟合的后续实验，按时间严格区分拟合、校准与最终核验；先锁定一个保守Projection/噪声方案并与Structured/GNN同协议比较。不能用GNN已监督的L内留出来冒充端到端外样本校准。采集偏倚和validation曝光仍须披露。随后若另获明确资源许可，再补真正T5-base冻结对照；不在本次下载。

**8. 如何服务最终学习效率与稳定性？** 让实验身份、表示、核、噪声、预测方差与batch条件信息量都可复算，并先通过外样本概率和文本稳定性门槛；通过后才检验同初始状态、同预算的数值batch AL。数值可审计是研究工具优势，本次尚未证明实验标签效率提高。

## 本次停止边界

未调用付费生成式API、未下载模型、未新训QGeoGNN、未读取本状态test真值或未来U响应、未改现有Free/NoCW/CW-Free轨迹。数据来源、权限和工程日志字段修复见[AUDIT.md](AUDIT.md)、[execution_audit.json](execution_audit.json)与[summary/summary.json](summary/summary.json)。原始拟合日志保留；状态seed由冻结目录恢复，模型初始化seed另存，数值预测未改。

完整证据：[表示](REPRESENTATION_REPORT.md)、[校准](GP_CALIBRATION_REPORT.md)、[容量与融合](COMPARISON_REPORT.md)、[未来AL设计](AL_FEASIBILITY.md)。本裁决机器汇总SHA256：`c6eff9e4004835036fff1b454bdf3b9f7222124fc47e05c03eb95ce0d0613b9a`。
