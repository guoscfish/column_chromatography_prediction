# 后续主动学习可行性：只设计、不执行

本次完成完整20状态开发矩阵，共720次端点GP拟合。所有结论来自已有LCMD轨迹、L_t训练与既有validation开发评价；四预算与五seed相关，validation参与过QGeoGNN checkpoint selection，不能称独立测试或AL确认。

**结论：No-Go，暂不启动正式LLM Embedding + GP主动学习。** 原因不是只看谁RMSE最小，而是本次Projection概率欠覆盖、模板采集不稳定、有限优化预算未收敛及开发评价曝光。详细裁决见[NEXT_STAGE_DECISION.md](NEXT_STAGE_DECISION.md)。

## 最小验证顺序

1. 先用既有早期冻结checkpoint与其未拟合、但历史已合法观测的实验作严格prequential开发核验；拟合/校准/最终核验按时间冻结，不用最终核验调噪声、核或Projection。只改一个预先指定的正则化/噪声问题，保留Structured、GNN与相同优化预算。该步骤也需要单独明确数据范围，不由本脚本自动执行。
2. 只有概率评分、覆盖-宽度、初始化与文本扰动都支持时，再补真正T5-base对照。T5当前缺本地权重，官方单份safetensors约892MB、Apache-2.0；资源与下载须另获明确许可。Qwen负结果不能代替这一步。
3. 只有通过上述验证的Frozen、Projection或Fusion才进入单独预注册的AL比较；不因本模型内BatchIG高而直接放行。

## 未来实验臂

| 实验臂 | 目的 |
| --- | --- |
| LCMD32 / Numerical Hybrid32 | 当前强数值参考，冻结其真实历史实现 |
| Frozen LLM + GP + batch acquisition | 检验固定语言表示 |
| Task-adapted LLM + GP + 同batch acquisition | 检验MLL适配是否有标签效率增益 |
| 经离线筛查的GNN+LLM Fusion + GP | 检验同容量控制后的互补性 |
| 已有Free LLM32历史参考 | 对照显式生成式决策的成本与可靠性；执行时期差异必须披露 |

GP-MaxVar、条件IVR和Batch Information Gain与预测学习目标一致；不直接照搬面向目标值最大化的EI。未来应使用相同初始L333、同合法候选、同标签预算、QGeoGNN scratch初始化/重训与checkpoint选择规则。各采集函数分别成臂，不把不同表示、核、容量和batch rule一起换后归因单因素。

未来两端点应共选一个32点batch，而非各选32点。可预先锁定独立端点GP下的等权归一化MaxVar `0.5×Σ_q Var(f_q)/s_L0,q²`；IVR使用相同两目标归一化平均方差缩减，BatchIG在明确的端点独立假设下相加。每加一点都对两个端点的共享batch同时条件化。该规则只作为未来预注册建议；本次按端点分别做诊断，没有运行上述合并选样，也未验证端点独立假设。

正式主指标为预冻结raw学习曲线的归一化梯形AULC；次指标为终点误差、跨seed方差、最差seed、标签及计算成本、合法运行成功率。各方法不能用不同完成端点或best-so-far曲线冒充公平比较。未来必须保留独立标签权限与prediction/batch freeze。

本次仅在同256个无标签合法U子池上比较数值采集排序、Top32交集、条件化批次与冗余；没有获取这些离线所选实验的响应，没有训练新QGeoGNN，没有新AL轨迹或AULC结果。不同GP的自评信息量不是共同真实收益标尺。
