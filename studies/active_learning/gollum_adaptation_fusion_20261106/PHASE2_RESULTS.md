# Phase 2 实际执行结果

本阶段完成官方方法对齐、模型/资源审计、模块化实现和本地工程测试；**没有执行T5或Qwen3真实任务训练**。原因是目标权重不在已检查本地缓存，用户未授权新增权重下载；PEFT/GPyTorch/BoTorch亦缺失。Phase1科学门槛仍独立适用，工程预飞不能绕过该门槛。

## 实际执行矩阵

| 对照 | 真预训练模型加载 | 梯度/数值预飞 | 色谱训练/外样本指标 |
|---|---|---|---|
| Frozen T5+GP | 离线尝试失败 | 未执行真T5；toy结构通过 | 未执行 |
| Frozen T5+Projection+GP | 缺权重 | 未执行真T5；toy结构通过 | 未执行 |
| T5+LoRA+GP | 缺权重 | 未执行真T5；原生rsLoRA toy通过 | 未执行 |
| T5+LoRA+Projection+GP | 缺权重 | 未执行真T5；原生rsLoRA toy通过 | 未执行 |
| Qwen3-Embedding-0.6B | 离线尝试失败 | 未执行真模型 | 未执行 |
| 旧Qwen1.5 | 已存在可复用缓存 | 本阶段不重复提取旧表示 | Phase1/Phase3复用，不能当成T5对照 |

`t5_offline_preflight.json` 和 `qwen3_offline_preflight.json` 保存了实际OSError、固定revision、离线标志、峰值RSS与依赖/缓存回执。异常文本是库统一提示，不能据其“couldn't connect”措辞认为发生了模型联网请求；本次加载强制local_files_only和offline。每次缺缓存退出约7秒，峰值RSS约424MiB，属于导入和缓存查找成本，不是模型forward成本。

## Toy 工程测试范围

`phase2/preflight.py --toy` 在随机小型网络上，使用8行人工token/标签、4种结构×初始化17/29，每次3步全批Exact MLL。每步使用同一完整8×8协方差，训练标签在训练集内部标准化；编码器float32、kernel/Cholesky/MLL float64。没有调用任何真实色谱标签。所有8次都通过：

- LoRA、Projection（需要时）和GP参数梯度有限、有非零分量且发生更新。
- 冻结base参数的字节哈希不变且没有梯度。
- Cholesky info每步为0，协方差包含独立noise和固定jitter。
- 每步在backward之后、optimizer.step之前执行梯度裁剪。

初始B=0时LoRA A第一步梯度为0符合链式法则，不能错误要求每个adapter参数从第一步都有非零梯度；第三步A/B均有有限非零梯度。toy将LoRA dropout设0以固定数值检查；真实默认配置为官方0.2。预飞状态始终标记 `step_limit_preflight_not_convergence`。

`phase2/tests/test_phase2.py` 有7项测试：padding/pooling及padding梯度，官方顶部层选择/zero-B恒等性，含重复输入的feature gradcheck与MultivariateNormal概率等价，LoRA B/Projection/GP lengthscale/noise中心有限差分，四结构完整梯度流，输入标签拒绝/两个体积字段/身份哈希，离线标志。实际测试回执见 `phase2/test_results.txt`、`phase2/test_receipt.json`。数值有限差分只验证梯度，不证明预训练表示、任务拟合或校准有效。

## 尚不能回答的研究问题

目前无法判断T5是否优于旧Qwen、Frozen T5是否有增量、Projection与LoRA各自贡献、二者交互收益、概率质量/文本鲁棒性是否改善，或微调成本是否值得。没有RMSE、NLPD、coverage或AULC数值可报告；toy loss不得用于这些比较。也不能声称语言先验优于GNN Hidden，或具备进入真实QGeoGNN闭环AL的资格。

此外当前工程GP与Phase1/旧Qwen的距离、mean、noise/jitter、目标std自由度不同，尚未形成公平性能比较器。正式Pilot前需统一euclidean或RMS距离、可学习或固定mean、noise/jitter及ddof，并冻结是否采用官方ScaleToBounds。本轮Phase1主实验发现更长拟合预算仍可能noise触下界且MLL改善伴随外样本NLPD恶化；这一前置门槛未通过时，Phase2保持工程准备状态。

模板A/B及SMILES-only接口复用旧研究的特征白名单renderer，保留PE/EA、Density、V/ul、loading solvent、loading solvent volume；其中V/ul是样品体积，另一个是加载溶剂体积。模板B仅作为未来冻结后的鲁棒性核验，不用于训练/调参。真模型的tokenization、embedding变化、GP均值/方差排序和MaxVar Top32目前未测。

## 恢复条件与命令

先获T5指定revision约0.9GB权重/必要tokenizer文件授权，并满足Phase1训练门槛；不需要先下载更大的第二模型。随后按 `PHASE2_MODEL_AND_RESOURCE_PLAN.md` 的只读本地模型命令执行4文本3步真模型预飞，验证模型加载键、目标模块、EOS/pooling、梯度及实际RSS/时间。通过后才能冻结真实Pilot的完整数据/初始化/模板/概率评分协议。当前恢复命令只实现工程预飞；真实色谱Pilot尚需承接主研究数据协议，不能把缺失runner写成已可直接完成的性能复现。

当前工程测试可重放：

```bash
export PYTHON_BIN=/Users/fish/miniforge3/envs/chromatography/bin/python
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON_BIN" -m pytest -q studies/active_learning/gollum_adaptation_fusion_20261106/phase2/tests/test_phase2.py
"$PYTHON_BIN" studies/active_learning/gollum_adaptation_fusion_20261106/phase2/preflight.py --toy --steps 3 --output studies/active_learning/gollum_adaptation_fusion_20261106/phase2/toy_gradient_flow_rerun.json
```
