# Implementation audit：Phase 1

## 证据边界和冻结

旧模型、训练脚本及其六个依赖代码的 SHA256 与历史 EXECUTION_PROVENANCE 一致；新入口通过导入使用原模型、loss、batcher、prediction 和 point metrics，没有改动其 GroupKFold 语义。所有写操作限本新目录。历史 study 1069 个文件全部哈希不变；source/data/graph cache/split 哈希与旧 protocol 一致。外层 test 与 validation 只读取身份/获准 feature 白名单，没有标签请求、训练图、epoch 选择或评分。LABEL_ACCESS_AUTHORIZATION.json 固定允许的 ID；PREFLIGHT.json 保存元数据 header 审计。

## 测试与真实拟合

test_protocol.py：14 passed。覆盖真实五个 seed 的 outer 身份隔离（只对试点拟合）、每折两柱完整性、稳定排序、固定 seed 重现、不同 seed 改变身份、跨柱共享 ID 拒绝、允许分子重合、授权读取跳过 forbidden 标签、真实目标 loader 白名单/标签请求/outer graphs 排除、共同初始权重、图输入 digest 与合成标签下原训练入口一步 checkpoint 完全一致。

FIT_AUDIT.csv 覆盖十个完成拟合。A1/A2 同折训练/验证 IDs、scales、counts、training seed、batch size、steps_per_epoch 一致；所有共同 epoch 的 batch_draws_sha256 一致。best_epoch 是严格最早 joint argmin，保存模型恢复包含 BN 的完整 state_dict。frozen 参数逐张量与初始状态相同；early BN 统计不变，late BN num_batches_tracked 增量等于 best checkpoint 实际步数。真实 optimizer_steps 是执行 optimizer.step 累加，而不是推测完整源数据遍历次数。

verify_inner_predictions.py 的零标签重放检查 20 个目标上下文、1368 条预测，最大绝对差 2.84e-14 mL。此独立检查只解析输入白名单、身份及保存的盲预测，没有新增标签读取。CHECKPOINT_REPLAY.csv/json 保存逐上下文证据。

每折 source、25g、40g 的训练损失和验证损失有限，预测/梯度有限；A1 trainable/total=130615/460500，A2=139367/469252。FiLM 仍是 embedding_dim16、blocks3/4、零初始化；pretrained/new LR、Adam wd、raw_quantile、混合 BN、patience80/max500 不变。新增诊断不改变目标或参数更新；合成等价性测试对此提供直接执行证据。

## 数据独立性

ROW_SPLIT_MANIFEST.csv 包含每个实际 fold 的真实身份。目标 OOF 每 ID 恰一次验证，两个 arm 预测 ID 相同。INPUT_OVERLAP_AUDIT 按 heldout 方向记录同柱及跨柱分子、身份、图输入重合；跨柱 full_model_input_overlap_rows=0，因为 task identity 不同。输入 digest 包含实际 atom/bond/angle 图和原条件张量，排除 y、canonical_position 和缓存派生量；不是把原 CSV 全列当模型输入。

CSV header 没有明确 independent batch/replicate ID；source_row_1based 不是实验批次证据。重复输入行可能来自不同实验、重复测量或数据处理，当前不能判定，保留 Row 记录级估计并披露边界。新 runner 若遇到共享 sample_id 立即拒绝，而不是静默按列拆到两侧。

## 结果限制与停止

未发现新的训练错误。十折模型全部按原 recipe 运行；收益、tail/Center/Width 和 cap/early stop 详见 REPORT。单 seed 不足以证明选择偏好、跨 seed 稳定性或外层泛化。后续四 seed/refit/score 不在入口中，也未执行。新增文件不覆盖历史冻结证据。

复现（在仓库根目录）：`KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python studies/transfer/column_conditioned_row_aligned_v1/run_phase1.py`；完成拟合按 receipt 校验复用。仅重新报告：同一 Python 运行 summarize_phase1.py。测试：`python -m pytest -q studies/transfer/column_conditioned_row_aligned_v1/test_protocol.py`。上述 Python 为已验证 fish 环境，系统/base Python 缺 torch，因此没有安装或修改环境。
