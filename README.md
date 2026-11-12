# QGeoGNN 主动学习科研框架 V2

面向柱色谱点预测、表示、数值选点和受控 Agent 的单一稳定 API。日常只维护 `predictor / al / infra`；权威科研摘要从 [research](research/README.md) 查找，完整历史统一通过远端 `archive/framework-v2-pretrim-20261111`（固定提交 `973ccaf8cb436e129ba3facc79d42853263923ce`）恢复。

**本分支目前仅验证合成 runner + 数值核心 + 已有模型推理，生产真实训练 / LLM / 真实 K3 仍未资格化。** 原 `main` 的 CW-Free 与其运行目录独立保护，未在本轮合并或替换。

## 五分钟开始

在独立 Python 环境安装所需依赖，从仓库根目录执行：

```sh
python -m pip install -r requirements-core.txt
python -m scripts.run_al --config configs/offline_synthetic.json --output /private/tmp/qgeognn_al_demo
# 同一代码身份和配置恢复，不重复已经持久化的回合
python -m scripts.run_al --config configs/offline_synthetic.json --output /private/tmp/qgeognn_al_demo
```

首次输出目录须全新，勿指向 main。示例预算2→6，完成两轮合成 fit→validate→feedback→select→freeze→reveal；fit 是 fixture，不优化权重，标签为明确合成值。它不读取真实候选标签或调用 provider。

- **模型**：[predictor](src/qgeognn_al/predictor/)，图/编码器仍依赖 [application](application/)。使用模型推理前另准备 `requirements-predictor.txt` 与授权、hash匹配的 checkpoint / 图缓存；Git 归档不替代私有资源。
- **组合方法**：[架构与API示例](docs/ARCHITECTURE.md) 展示授权 `FeatureSnapshot` → `select`；表示与acquisition分开，方法身份见 [registry](configs/method_registry.json)。
- **科研依据**：[预测器资格](research/PREDICTOR.md)、[五seed Free/LCMD](research/FIVE_SEED_AL.md)、[CW/NoCW边界](research/CW_NOCW.md)、[GP/IVR重要负结果](research/NEGATIVE_RESULTS.md)。
- **数据与权限**：[保护登记](configs/protected_artifacts.json)、[真值消费](configs/data_consumption_register.json)；已使用的test或validation不是新的盲确认集。
- **验收范围**：[Framework V2验收](docs/FRAMEWORK_V2_ACCEPTANCE.md)；当前状态见 [NEXT_STAGE_DECISION](docs/NEXT_STAGE_DECISION.md)。

测试按能力划分，使用测试依赖环境：

```sh
python -m pip install -r requirements-test.txt
python -m pytest -m unit
python -m pytest -m frozen_regression
python -m pytest -m historical_archive
```

后两类需要明确的历史Git对象或授权只读fixture。缺失时按 `BLOCKED` 报告，不能把 synthetic pass 当作真实 checkpoint 或完整历史实验复现。

## 查阅或恢复历史

完整历史的唯一主入口是 [固定973ccaf8快照](https://github.com/guoscfish/column_chromatography_prediction/tree/973ccaf8cb436e129ba3facc79d42853263923ce)。例如只恢复原IVR报告到新独立目录：

```sh
git fetch origin tag archive/framework-v2-pretrim-20261111
mkdir -p /private/tmp/qgeognn_history
git archive archive/framework-v2-pretrim-20261111 studies/active_learning/qgeognn_v2_ivr_mechanism_audit/FINAL_REPORT.md | tar -x -C /private/tmp/qgeognn_history
```

更完整的受控源码恢复见 [历史来源配置](configs/archive_manifest.json) 与 [工具说明](tools/README.md)。旧研究的相对链接、Prompt、冻结账本和源码应在原历史版本中解释。tracked Git字节可恢复，不代表 ignored checkpoint、原始runtime、私有请求、凭据或原环境已备份。不要向正在运行的main覆盖历史文件。
