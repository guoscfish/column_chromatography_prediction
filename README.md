# QGeoGNN 统一主动学习科研框架

柱色谱 QGeoGNN-V2 Row 模型、表示、数值选点、独立 LLM profiles 和唯一轮次状态机。模型与 loss 保持原数学；真实训练、付费请求及新标签揭示都需要单独授权。

**当前不是 FULL READY。** 已验证确定性输入/初始化、已有模型推理、数值核心、30个真实录制选择回合，以及五seed两轮真实K3冻结checkpoint回归；完整重新训练及在线科研尚未资格化。逐项状态见 [能力矩阵](docs/CAPABILITY_MATRIX.md) 与 [等价报告](docs/EQUIVALENCE_REPORT.md)。main 实验独立保护，未切换或合并。

## 五分钟开始

在独立 Python 环境，从仓库根目录执行：

```sh
python -m pip install -r requirements-core.txt
python -m scripts.run_al validate --config configs/offline_synthetic.json
python -m scripts.run_al dry_run --config configs/offline_synthetic.json
python -m scripts.run_al --config configs/offline_synthetic.json --output /private/tmp/qgeognn_demo
python -m scripts.run_al resume --config configs/offline_synthetic.json --output /private/tmp/qgeognn_demo
python -m scripts.run_al report --output /private/tmp/qgeognn_demo
```

首次输出目录须全新。示例预算2→6，合成fit不优化权重，所有标签均为自有合成值。

运行真实已录制的 LLM 选择回放，无私有资源、网络或provider：

```sh
python -m pip install -r requirements-replay.txt
python -m scripts.run_al offline_replay --fixture tests/fixtures/llm_recorded_replay.json.gz
python -m scripts.run_al offline_replay --fixture tests/fixtures/llm_recorded_replay.json.gz --profile cw16_llm16_scientist_v2
```

内置8回合覆盖Screen两个seed各三轮、Scientist V2首轮与反馈纠错轮。外部只读私有快照另验证CW/NoCW五seed各两轮及Scientist第二seed两轮，共30独立回合；只证明相应选择/反馈层，不代替重新训练。

外部快照保持在Git之外，取得明确授权并配置独立只读manifest后：

```sh
python -m scripts.run_al offline_replay --fixture /snapshot/private_agent_manifest.json
export QGEOGNN_PRIVATE_AGENT_REPLAY_MANIFEST=/snapshot/cw_nocw_manifest.json:/snapshot/scientist_manifest.json
export QGEOGNN_K3_REGRESSION_MANIFEST=/snapshot/k3/MANIFEST.json
```

```sh
python -m pip install -r requirements-test.txt
python -m pytest -m unit
# 外部授权fixture与固定Git对象准备见等价报告
python -m pytest -m frozen_regression
python -m pytest -m historical_archive
```

## 真实科研入口

[运行与回放指南](docs/ARCHITECTURE.md) 说明独立只读资源、schema3配置、授权和按freeze绑定的标签收据。准备后使用同一入口：

```sh
python -m scripts.run_al validate --config /path/to/row_config.json
python -m scripts.run_al dry_run --config /path/to/row_config.json
python -m scripts.run_al authorized_real_run --config /path/to/row_config.json --authorization /path/to/explicit_grant.json --output /path/to/new_runtime
python -m scripts.run_al resume --config /path/to/row_config.json --authorization /path/to/explicit_grant.json --output /path/to/new_runtime
```

上两条不训练、不加载标签、不检查secret。后两条可能训练、调用provider并消费选中标签，只有在另获明确授权时运行；本轮没有执行它们。provider依赖另见 `requirements-provider.txt`。未准备标签时停在持久化freeze，可补充精确收据后resume；模糊调用不自动重复。

- 模型、图与训练：[predictor](src/qgeognn_al/predictor/)，仍依赖 [application](application/)。
- 数值方法：[profiles](src/qgeognn_al/al/numerical_profiles.py)；LLM方法：[profiles](src/qgeognn_al/al/agents/profiles.py)。Free全池、Screen128、Scientist及NoCW各自保留版本身份。
- 科学定义与数据边界：[代码合同](docs/model/CODE_CONTRACT.md)、[输入合同](docs/model/INPUT_SCHEMA.md)、[真值消费登记](configs/data_consumption_register.json)。
- 既有权威结果：[research](research/README.md)；其历史评价不作为新框架重训的结果。

## 固定历史恢复

[来源映射](docs/LEGACY_TO_NEW_CAPABILITY_MATRIX.md) 固定main `b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28`；重构前提交固定为 `33f0ab46020c9175b23aff9741d143cb4a869e9a`。两者均已有远端annotated tag，恢复方式见 [archive manifest](configs/archive_manifest.json) 与 [工具说明](tools/README.md)。完整旧研究仍由 `archive/framework-v2-pretrim-20261111` / `973ccaf8cb436e129ba3facc79d42853263923ce` 读取。

```sh
git fetch origin tag archive/framework-v3-main-source-20261112
git fetch origin tag archive/framework-v3-pre-migration-20261112
git fetch origin tag archive/framework-v2-pretrim-20261111
python tools/history_restore.py --path studies/active_learning/qgeognn_v2_ivr_mechanism_audit --destination /private/tmp/qgeognn_history --output /private/tmp/qgeognn_restore.json
```

Git只能恢复tracked字节；ignored/runtime、私有checkpoint、未提交请求、secret及旧环境未因此备份。本轮没有删除这些资源或向main写入。
