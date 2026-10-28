# Implementation audit：完整五seed Phase 2

## 冻结与范围

PHASE2_PROTOCOL.json引用原冻结protocol.json和phase1_archive，并记录Phase1与历史Row外层结果已曝光。科学设置未变：sample_id排序、KFold5/shuffle/seed42、同一4g源checkpoint、原A1/A2结构、FiLM16与blocks3/4、Adam原LR/weight_decay、raw_quantile、混合late BN、500/patience80、等权两柱Combined NRMSE、最早argmin与五折中位数。新代码不含refit、外层score、PCGrad或搜索入口。

唯一执行诊断差异是遵照本轮用户权限停止解析源validation标签。test_phase2.py以AST比较确认fit代码仅新增source-validation诊断跳过分支；A1/A2三步合成更新、目标历史和best epoch与Phase1逐位一致。source_train人口及所有目标训练/选择完全不变。数据loader使源validation truth为NaN、图y为零，防止误将占位值记为观察指标；源validation误差/loss在Phase2缺失并明确标记。

## 逐seed核验

phase2/seed_*/COMPLETE_AUDIT.json保存每seed完整检查和CSV哈希。共50模型、100目标记录；Phase1十模型从原receipt复用，40新增模型均在完成后审计。ROW_SPLIT_MANIFEST包含全部真实折身份，每目标ID在本seed五折中恰一次验证；源train和diagnostic身份固定。对每折比较两个arm的训练/验证hash、数量、归一化尺度、训练seed、batchsize和steps，全部一致；共同epoch的batch_draws_sha256完全相同，共同初始状态digest一致。

独立读取best.pt核对best_epoch、源authority、冻结参数和early BN buffers、late BN num_batches_tracked增量；best是joint历史最早argmin，optimizer_steps是实际累加，每个checkpoint最佳步数与best_epoch×steps一致。重算获准inner验证指标与保存CSV、best history一致；零标签模型重放 100个上下文、6788条预测，最大差 2.84e-14 mL。目标训练/验证预测、训练梯度、train和target validation损失均有限。

## 权限、重复输入和保存

PHASE2_LABEL_AUTHORIZATION固定每seed gradient_train与源source_train ID；每新fit的label_access.json记录实际读取请求hash。源validation/test、目标outer validation/test真实标签均无请求。只按特征白名单解析CSV，原features文件冗余时间列未解析。输入digest含实际图与条件张量，排除y/身份；不同ID重复输入保留当前协议并统计，没有自行去重或重新分组。跨柱task ID不同意味着骨干输入重合不等于完整模型输入相同。缺少明确物理实验批次/replicate元数据，不将共享记录的五seed或25折当独立物理重复。

原Phase1 110个runtime文件、43个归档根文件，以及历史研究1069个文件哈希保持不变。原Phase1 protocol.json、训练/测试代码与provenance不改动；PROTOCOL.md按用户要求追加阶段授权/曝光/诊断边界披露。根目录聚合结果更新为50拟合，Phase1原表保存在phase1_archive，原单seed负结果保留。

## 限制与执行记录

没有发现训练协议漂移或非有限预测/错误参数范围/BN故障。19项执行与权限测试通过（TEST_RESULTS_PHASE2.txt），统计定义的独立算术检查见REPORTING_TEST_RESULTS.txt。每seed的收敛上限和所有tail/max负结果见REPORT。没有提高500上限或改变epoch选择。完整曲线保存在learning_curves/，全部历史在TRAINING_HISTORY.csv。

复现已完成检查/汇总：fish Python运行summarize_phase2.py；Phase2运行入口run_phase2.py只接受四个新seed并按receipt复用已完成fit，拒绝拟合pilot。该入口没有其他科学动作。报告生成不训练模型；诊断重放不会读取外层标签。**已完成Phase2并停止，未执行FULL-data refit或outer评分。**
