# 当前模型与数值科学合同

当前实现是独立 **QGeoGNN-V2**，不是失败的Clean结构方案。标准资格配置有458,952个有效参数，原始工程等价与资格来源见 [PREDICTOR](../../research/PREDICTOR.md)。原论文/发布代码/历史Legacy差异的完整审计原文位于 [固定历史快照](https://github.com/guoscfish/column_chromatography_prediction/blob/973ccaf8cb436e129ba3facc79d42853263923ce/docs/model/PAPER_CODE_CONTRACT_AUDIT.md)；本页只说明稳定树现有合同，不重写原科学结论。

- 模型定义：[predictor/model.py](../../src/qgeognn_al/predictor/model.py)；图与原编码器仍真实依赖 `application/`，本次没有迁移或删除其数学实现。
- 输入单位、类别与8D条件补齐分支见 [INPUT_SCHEMA](INPUT_SCHEMA.md)。9D诊断矩阵不等于8D模型输入，column ID未被新增为特征。
- 原始输出固定为 `V1_q10,V1_q50,V1_q90,V2_q10,V2_q50,V2_q90`。每目标loss仍为q10 pinball + q50 MSE + q90 pinball + 两个soft crossing；资格协议V1/V2等权。不存在模型内硬非交叉约束。
- q50是MSE学习的中心预测，不能直接叫校准条件中位数。q10/q90宽度、交叉或跨目标跨度不自动构成epistemic UQ；原始交叉必须在排序前统计。
- source训练只用source-train拟合scaler、条件统计与target sigma。AL covariate normalization用冻结outer-train X，target sigma/CW只用已授权L0标签；candidate真值不得用于尺度或表示。
- 原loss未按target sigma标准化；评价Combined NRMSE为 `sqrt(0.5*((RMSE1/sigma1)^2+(RMSE2/sigma2)^2))`，不可把raw loss份额与该误差贡献混同。AULC必须注明预算网格、数据角色与归一化。
- checkpoint/图/表示绑定输入schema、初始化/训练来源、数据/划分、scaler与有序ID/hash。恢复前需匹配这些身份；同名文件不能代替精确SHA。

已验证的范围扩展为固定Row划分/初始化、已有checkpoint推理、数值核心与真实录制LLM选择层。真实Row训练/完整packet/受控provider路径已实现，完整多轮训练、K3、LLM与新NRMSE/AULC仍未资格化，详见 [能力矩阵](../CAPABILITY_MATRIX.md) 和 [等价报告](../EQUIVALENCE_REPORT.md)。正式科学切换仍需独立验收与授权。
