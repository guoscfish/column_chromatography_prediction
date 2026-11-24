# PureLLM-V3 · 4g Row B32

独立的科学决策 baseline。使用冻结的 4g Row 状态，完整提供允许的候选与观测，分别核对候选、分子、条件和证据。历史方法、模型源码、受保护数据和基准配置保持原状。

[中文总报告](MASTER_REPORT.md) · [当前实现审计](CURRENT_IMPLEMENTATION_AUDIT.md) · [字段审计](DATA_FIELD_AUDIT.md) · [历史调用审计](HISTORICAL_LLM_AUDIT.md) · [离线验证](OFFLINE_TEST_REPORT.md) · [下一阶段](NEXT_STAGE_DECISION.md)

## 离线复现

在仓库根目录运行。示例和静态 stage 使用历史请求、分割和既有观测。选择器不会打开真值存储。

```sh
study=studies/active_learning/pure_llm_v3_20261124
pkg=studies.active_learning.pure_llm_v3_20261124.pure_llm_v3

python -m pytest -q "$study/tests"
python -m "$pkg.offline_examples" --out /tmp/pure-llm-v3-examples
python -m "$pkg.runner" stage --seed 157 --round 0 --arm A7 --out /tmp/pure-llm-v3-r00
python -m "$pkg.runner" stage --seed 157 --round 1 --arm A7 --out /tmp/pure-llm-v3-r01
python "$study/verify_artifacts.py"
```

[初轮完整请求](examples/example_round0_request.json) · [后轮完整请求](examples/example_round1_request.json) · [合成响应](examples/example_response.json) · [局部修复请求](examples/example_local_repair_request.json) · [合成补丁](examples/example_local_repair_response.json) · [合成记忆更新](examples/example_memory_update.json)

源代码、测试及工具都在研究目录下面。旧流程的某些 runner 会递归计算 `src`、`scripts` 或 `tests` 的目录哈希，这一布局避免单纯添加 V3 触发既有协议清单变化。复算命令与记录见各审计报告。若只克隆 Git 而无历史忽略目录的完整文件，可查看审计支持的派生数据，真实状态 staging 仍需可核验的原始冻结工件。

## 下一阶段 · 同状态试验

先把 30 个状态条件完整写入临时运行目录。下列循环只 stage，使用 [pilot 配置](pilot_config.json)。在 Linux/GPU 主机也可用同一命令，选择阶段不依赖 GPU。

```sh
study=studies/active_learning/pure_llm_v3_20261124
pkg=studies.active_learning.pure_llm_v3_20261124.pure_llm_v3
pilot=/tmp/pure-llm-v3-pilot

for seed in 157 887 2357 6101 12203; do
  for r in 0 6; do
    for arm in A0 A1 A6; do
      conda run -n fish python -m "$pkg.runner" stage --seed "$seed" --round "$r" \
        --arm "$arm" --config "$study/pilot_config.json" --out "$pilot/$seed/$r/$arm"
    done
  done
done
```

获得真实使用授权后，给每次选择准备审阅过的 JSON，字段为 `scope=paid_selection_only`、`request_id`（取自对应 staged_request.json）、`authorized_by`、`authorization_reference`。在授权环境配置 API 凭据。示例命令只运行一个状态。

```sh
conda run -n fish python -m "$pkg.runner" select --seed 157 --round 0 --arm A6 \
  --config "$study/pilot_config.json" --out "$pilot/157/0/A6" \
  --authorization "$pilot/157/0/A6/approved-selection.json"
```

后续连续试验从共享 L333 开始，为每 seed 和 M0/M1/M2 单独建持久目录。 `runner stage/select --trajectory <目录> --memory-mode <M0|M1|M2>` 使用同一固定配置，正式选择输出写入该目录的 `round_00`、`round_01` 等。非零轮读取前一接受批次绑定的持久记忆、反馈和 fit 产物。独立正式训练入口 `trajectory` 只在获得单批精确授权后读取该批标签，后续静态选点 pilot 不需调用。授权 JSON 需有 `scope=reveal_and_fit_one_frozen_batch`、冻结文件字节 SHA `selection_freeze_sha256`、固定 `device`、`authorized_by` 和 `authorization_reference`。

```sh
# 以下命令属于另行授权的后续连续试验。
trajectory=/persistent/pure-llm-v3/seed-157-M2
CUDA_VISIBLE_DEVICES=0 conda run -n fish python -m "$pkg.trajectory" \
  --trajectory "$trajectory" --seed 157 --round 0 --device cuda \
  --authorization "$trajectory/approved-reveal-and-fit-r00.json"
```

证据见 [Prompt](PROMPT_DESIGN.md)、[上下文](CONTEXT_DESIGN.md)、[身份](IDENTITY_PROTOCOL.md)、[证据](SCIENTIFIC_EVIDENCE_PROTOCOL.md)、[记忆](MEMORY_DESIGN.md)、[纠错](REPAIR_PROTOCOL.md)、[传输](TRANSPORT_PROTOCOL.md)、[纯度](BASELINE_PURITY_AUDIT.md) 和 [对照预注册草案](REAL_LLM_PILOT_PROTOCOL.md)。

本阶段真实付费 API 请求、新增标签揭示、正式模型训练均为 **0**。
