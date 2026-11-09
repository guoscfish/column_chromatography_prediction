# 第二期清理验证

本报告仅覆盖本期候选源码与明确列出的冻结输入。第一期 105 项测试记录保留在历史提交，没有为了本期计数重新运行。分位数/损失诊断另在 main 的 `fcb51852c8bb6e38403fe95a302287a0a1779d47`，见 [SCIENCE_AUDIT.md](SCIENCE_AUDIT.md)。

## 实际结果

在旧目录已经从实际候选 linked worktree 删除之后运行 `tests/phase2`：**91 passed，0 failed，0 skipped**。原始输出见 [final_tests.log](validation/final_tests.log) 和 [final_tests.xml](validation/final_tests.xml)。已有依赖发出 deprecation/版本告警，未屏蔽或当作数值失败。main 诊断的独立 9 项测试通过，其 XML 随 main 的诊断提交版本化。

| 测试组 | 项数 | 验证范围 |
| --- | ---: | --- |
| 表示、选样和代理输入 | 41 | 五个数值核心、表示、3 类合成边界、固定 reference/trajectory、真实冻结输入及 Free 共用合法性契约 |
| Predictor 迁移 | 8 | 原函数/模型 AST、真实图缓存与五个 checkpoint 的六输出和 latent |
| RoundWorkflow | 21 | 五方法多回合、freeze/reveal 顺序、receipt 恢复、模糊重试阻断、跨 seed/配置/标签身份和状态链攻击 |
| CLI | 2 | 两轮合成示例及 main/非合成执行限制 |
| AL 预处理 | 1 | outer-train X、L0-only target sigma 与旧实现相等 |
| 评价 | 3 | 原 NRMSE、标签转换、AULC 数值及冻结预算网格 |
| 历史恢复 | 5 | 固定 Git 身份、whole-src hash、环境版本、篡改与不安全恢复目标 |
| 运行隔离 | 4 | source/identity/main 保护和限定诊断提交例外 |
| 迁移审计 | 3 | 依赖类型、恢复证明和稳定引用退出 |
| 布局与资源 | 3 | 三层包、一 runner、无旧源码导入、只读快照路径/链接/哈希 |

### 真实模型与缓存

明确授权的五 seed 157/887/2357/6101/12203、L333 checkpoint 和纯结构图缓存来自独立只读私有快照，逐一核对 SHA，不从新 API 动态加载 live 源码。checkpoint 加载后六输出和 hidden 128 与不可变旧科学代码逐值相等（max absolute difference=0，模型参数/缓冲不变）。每 seed 完整重建 749 个授权 train/validation 图，并在固定 16 行上对照 x、edge、attrs、y 和 canonical position；没有比较或转换 U/test 真值。证据见 [predictor_equivalence.json](validation/predictor_equivalence.json)。

表示回归另用每 seed 的 8 个 L0 与 8 个合法 U 的无标签输入，验证 latent、q50 全网络梯度 CountSketch 与 CW 变换；原/新逐值相同，模型状态和 `.grad` 未改变。真实缓存不随 Git 分发，未提供授权 manifest 的新环境应报告相应测试 SKIP/BLOCKED，不能用纯合成测试替代。

### 算法与历史回合

[algorithm_equivalence.json](validation/algorithm_equivalence.json) 保存 33 条记录，包含选中 ID/顺序、trace 摘要和数值一致性。LCMD、MaxDet、IVR、Hybrid 覆盖 ordinary、ties、rank-deficient 三种边界；random 覆盖明确 seed 和历史三轮固定轨迹。原 LCMD 全零距离的空 cluster 异常行为仍保留，没有悄悄“修正”科学定义。

真实 seed157 的三个保存回合分别为 L333→365、L525→557、L973→1005；各用其自己的 3330×512 当前模型特征缓存和不同 checkpoint。新/旧 LCMD 的 32 个选中 ID 与排序完全匹配已有冻结结果；原/新进程内 trace 完全相同，对历史 CSV 浮点格式采用预设 rtol=1e-12、atol=1e-8。输出记录绑定各自 feature/checkpoint SHA 与当前 L 身份。MaxDet/IVR 在同一真实缓存上验证的是反事实数值核心等价，没有新建正式选点、冻结或标签揭示。

CW-LCMD 的 CW 梯度、线性变换和 LCMD 分别对旧实现回归，不表示已重训复现整条 CW AL 轨迹。IVR 验证完整原 universe 的固定 ID 顺序与逐轮 current-fit 对齐；fused 使用当前 L 的块 RMS。random 区分每轮 choice 与原 U0 一次 permutation 切片，末批不足额也有回归。

真实同 seed/同 L 的 K3 快照不可获得：Hybrid 纯数值核经过旧/新对照，真实 checkpoint 测试中的三成员信号明确来自同一模型的合成重复，只验证 tied-rank 工程边界。它不证明真实 K3、成员初始化/checkpoint 来源、Hybrid 完整重训练或 AULC 等价。生产后端缺失 supplier 即失败关闭。

### 生命周期与历史 LLM 协议

五个方法走完统一合成 fit→validate→feedback→select→freeze→reveal→下一轮 fit，最终预算 6、三次 fit fixture。回执落盘后状态写入失败可恢复且不重复 callback；只有 started 而无 receipt 的拟合或揭示必须人工处理，不能自动重复。跨 seed、标签/validation/backend 漂移，以及重算哈希后向状态注入 U 标签的攻击均在 callback 前失败。

`QGeoGNNBackend` 实际调用迁移后的训练/推理实现，但本次禁止新科学训练，因此完整生产 retrain 和 checkpoint 选择路径为 **NOT_MEASURED**，不是数值等价结论。CLI 只开放合成 fixture，没有真实科学标签或 provider 入口。

Free/CW/NoCW/V3/V3.2 都是 registry 的历史协议。旧 Free 的候选输入合法性与新共用契约对照通过；通用已提供计划可以通过状态机冻结。它们不证明旧数据包、提示词、随机 LLM 输出、重试或传输等价。没有新的 LLM adapter，没有付费请求。

## 退出证明与历史恢复

实际扫描 stable Python 174→38，旧版本目录 3→0，旧 v2/v3/v3_2 文件数 81/11/12→0。退役 374 路径=368 Python+6 其他文本；恢复源码物理行数 74,288（含注释空行）。9 个实际 representation/acquisition 算法文件；加 2 个 base 契约为 11 个非 init 模块。非 init 纯 re-export 层 1→0；精确规范化顶层函数 AST 重复组 80→0、额外副本 105→0。静态 parse error 与 stable→retired 运行引用均为 0。这不是语义重复/任意反射/外部消费者不存在的完备证明。

退出前对 374 个实际文件逐一比对 `DELETION_PROOF.csv` 的原 SHA；全部可从固定 `4ee480...` 和 blob 恢复。退出只发生在候选树。`INVENTORY.csv`、`MIGRATION_MAP.csv` 和带类型依赖图由最终实际工作树重新生成，历史 frozen_source 引用明确标记 `HISTORICAL_ONLY_RESTORE_REF`，不算作日常 stable 闭包。

两个真实 detached worktree 的恢复已执行，结果见 [history_scientific_restore.json](history_scientific_restore.json)、[history_phase1_restore.json](history_phase1_restore.json)：科学来源验证 263 源文件/165 whole-src-application/166 benchmark hashes；一期来源验证 375/177/178；各 7 项包版本匹配。未执行旧 controller、provider、数据加载或训练。恢复来源与命令见 [NEW_ARCHITECTURE.md](NEW_ARCHITECTURE.md)。

## main 与 CW-Free 保护

[GATE0.json](GATE0.json) 是开始时的只读快照，不能当作最新 seed 进度。最终 [RUNTIME_PROTECTION.json](RUNTIME_PROTECTION.json) 通过：main 仅从 1b2a2ca8 前进到获授权的诊断提交 fcb51852，净变更全为指定诊断目录的 23 个新增版本化文件；566 个既有科学/控制源码、受保护 frozen 文件和运行目录身份不变，main staging 为空，候选 HEAD/index/源码路径独立。没有改写旧冻结记录以适应新包布局。

最终只读进程核对中 supervisor PID90623 仍在，启动时间为 2026-11-09 16:51:24，与 Gate0 相符。worker 会自然完成和替换，不能声称最初所有 PID 或 runtime 字节均保持不变。本任务未发送信号、执行恢复控制器或重启实验；main 的其他 runtime 修改和新增工件由原实验继续产生，没有被本任务纳入提交。

## 复现入口

测试环境使用现有 `/Users/fish/miniforge3/envs/fish/bin/python`；依赖版本记录在 requirements 与历史恢复报告，本次未安装依赖。候选根目录执行：

```sh
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/column_phase2_mpl \
QGEOGNN_HISTORY_REPOSITORY="$PWD" \
QGEOGNN_REGRESSION_MANIFEST=/private/tmp/column_phase2_science_snapshots/REGRESSION_MANIFEST.json \
QGEOGNN_READONLY_REGRESSION_MANIFEST=/private/tmp/column_phase2_science_snapshots/MANIFEST.json \
QGEOGNN_FROZEN_ACQUISITION_MANIFEST=/private/tmp/column_phase2_frozen_acquisition_inputs/MANIFEST.json \
QGEOGNN_EQUIVALENCE_OUTPUT=/private/tmp/algorithm_equivalence.json \
python -m pytest tests/phase2 -q --junitxml=/private/tmp/phase2_tests.xml
```

上述私有路径仅描述本次授权输入位置，不能在新机器缺失时回退读取 live 标签。运行保护另需显式收据：

```sh
python tools/verify_runtime_isolation.py \
  --allowed-main-diagnostics-commit fcb51852c8bb6e38403fe95a302287a0a1779d47 \
  --output /private/tmp/runtime_protection.json
```

本分支未合并到 main。本期新提交尚未推送；main 正式切换、真实生产训练、K3 和 LLM 协议验证继续受 CW-Free 完成及新授权约束。
