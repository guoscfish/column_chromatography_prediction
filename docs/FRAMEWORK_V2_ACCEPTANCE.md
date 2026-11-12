# Framework V2 验收

完成日期：2026-11-12；任务及归档日期：2026-11-11。范围仅 `codex/repository-architecture-20261109`，未合并main、未新开项目分支。

**本分支目前仅验证合成 runner + 数值核心 + 已有模型推理，生产真实训练/LLM/K3 仍未资格化。** 生产训练、provider端到端与真实K3均为 `NOT_VERIFIED`；合成fit不优化权重。

## 完整历史与可恢复边界

远端仓库：`https://github.com/guoscfish/column_chromatography_prediction.git`。

- 已推送 annotated tag：`archive/framework-v2-pretrim-20261111`。
- tag对象：`698e64fd0e41b5c4af0c42c2f44876df8ca356c8`。
- peeled commit：`973ccaf8cb436e129ba3facc79d42853263923ce`。
- `git ls-remote`身份相符；独立HTTPS partial clone完整tree的 **22,294** 个path/blob身份相符。Git原生lazy-fetch真实恢复 **20** 个样本，逐一比对原blob、字节数和SHA256并设为只读。覆盖GP/IVR负结果、Free/LCMD五seed协议/汇总/冻结账本、V3.2 Prompt及tracked响应谱系、e0基线、旧工程审计和科学源码；未执行旧controller。

恢复入口为 [archive_manifest](../configs/archive_manifest.json)、[工具说明](../tools/README.md) 与 [根快速开始](../README.md)。先显式 `git fetch origin tag archive/framework-v2-pretrim-20261111`，再以 `history_restore.py --path ... --destination <全新独立目录>` 恢复所需路径。恢复工具不隐式fetch、不导入旧源码，不向现有Git工作树写入。

**仅保证已跟踪Git字节可恢复。** V3.2原登记明确有runtime留在本地；ignored/未提交请求、私有checkpoint、凭据、外部数据和旧环境未由此备份，也未物理删除。不得把20份代表性样本恢复说成完整旧实验复跑。

Commit A：`052fd0aca976247348595f30109b405726d8c45b`，仅提交小型归档凭据。Commit B：实际整组退出、长期测试/入口/科研索引及本验收报告；代码分支本轮只提交本地，已推送的是归档tag。

## 当前树及精简规模

统计以归档commit与最终Git跟踪树为准；目录含各顶层目录自身，总计含仓库根。字节为跟踪文件完整checkout内容之和，不是Git压缩空间或运行目录大小；ignored缓存不计入，也未用git clean删除。

| 顶层 | 跟踪文件 before → after | 目录 before → after | 字节 before → after |
| --- | ---: | ---: | ---: |
| 根文件 | 7 → 7 | 1 → 1 | 6,758 → 8,443 |
| application/ | 4 → 4 | 1 → 1 | 287,213 → 287,213 |
| configs/ | 4 → 6 | 2 → 1 | 5,467 → 188,211 |
| dataset/ | 10 → 10 | 1 → 1 | 1,065,830 → 1,066,811 |
| docs/ | 38 → 6 | 8 → 2 | 3,511,876 → 30,410 |
| experiments/ | 769 → 47 | 183 → 16 | 56,547,282 → 28,366,568 |
| reports/ | 462 → 0 | 48 → 0 | 91,567,536 → 0 |
| research/ | 0 → 7 | 0 → 1 | 0 → 65,510 |
| scripts/ | 2 → 2 | 1 → 1 | 3,821 → 3,821 |
| src/ | 38 → 38 | 10 → 10 | 157,659 → 157,659 |
| studies/ | 20,938 → 0 | 2,318 → 0 | 1,338,874,936 → 0 |
| tests/ | 14 → 17 | 3 → 2 | 95,444 → 98,851 |
| tools/ | 8 → 4 | 1 → 1 | 110,770 → 33,882 |
| 总计 | 22,294 → 148 | 2,577 → 37 | 1,492,234,592 → 30,307,379 |

重构linked worktree已取消自身sparse设置，148份跟踪文件实际全部展开，skip-worktree为0；不依赖隐藏的历史目录。

当前仅维护：`src/qgeognn_al/{predictor,al,infra}`、仍有真实依赖的 `application/`、`dataset/`、`configs/`、`scripts/`、长期 `tests/`、3种日常 `tools/`、精简 `docs/`/`research/`，以及下述最小 `experiments/` 资源。

实际退出原路径 **22,176个跟踪文件，1,462,288,973字节**；另有少量新摘要/测试/工具及迁移合同。`studies`的 **20,938/20,938** 文件、全部2,318目录退出当前Git树；`reports`的462文件和28个研究报告组全部退出。旧路径仍可在固定归档恢复；这不是sparse隐藏或改名搬进archive目录。

## 整组退出与必要例外

`studies/`四个一级组全部退出：`active_learning/`、`predictor/`、`i0_predictor_semantic_audit/`、`track_a_4g_al/`。为明确历史边界，以下为全部43个AL组和8个predictor组（位于原 `studies/`，均无当前跟踪文件）：

```text
active_learning/free_cw_sol61_xhigh_five_to1485_20261101/
active_learning/free_lcmd_mechanism_diagnostics_20261029/
active_learning/gp_guided_llm_stability_20261103/
active_learning/hybrid_loading_cap_audit_20261029/
active_learning/nocw_free_selection_diagnostics_20261101/
active_learning/qgeognn_v2_batch_adaptivity/
active_learning/qgeognn_v2_efficiency_review/
active_learning/qgeognn_v2_ivr_mechanism_audit/
active_learning/qgeognn_v2_row_center_width_performance/
active_learning/qgeognn_v2_row_cw_ivr_portfolio_b32/
active_learning/qgeognn_v2_row_cw_lcmd_to_1005/
active_learning/qgeognn_v2_row_cw_lcmd_to_525/
active_learning/qgeognn_v2_row_cw_lcmd_to_653/
active_learning/qgeognn_v2_row_cw_selector_extension_to_1005/
active_learning/qgeognn_v2_row_cw_selector_extension_to_653/
active_learning/qgeognn_v2_row_cw_selector_factorial/
active_learning/qgeognn_v2_row_fused_maxdet_a080_b32_screen/
active_learning/qgeognn_v2_row_fused_maxdet_b32/
active_learning/qgeognn_v2_row_hybrid_extension/
active_learning/qgeognn_v2_row_innovation_screen/
active_learning/qgeognn_v2_row_kernel_ivr_b32/
active_learning/qgeognn_v2_row_lcmd/
active_learning/qgeognn_v2_row_lcmd_to1485_20261026/
active_learning/qgeognn_v2_row_lcmd_to_ivr_b32/
active_learning/qgeognn_v2_row_llm16_screen/
active_learning/qgeognn_v2_row_llm_dialog_feedback_v1/
active_learning/qgeognn_v2_row_llm_dual_expert_v1/
active_learning/qgeognn_v2_row_llm_full_pool_feedback_v1/
active_learning/qgeognn_v2_row_llm_nocw_ablation/
active_learning/qgeognn_v2_row_llm_scientist_v2/
active_learning/qgeognn_v2_row_llm_scientist_v3/
active_learning/qgeognn_v2_row_llm_scientist_v3_1/
active_learning/qgeognn_v2_row_llm_scientist_v3_2/
active_learning/qgeognn_v2_row_master_report/
active_learning/qgeognn_v2_row_maxdet_b32/
active_learning/qgeognn_v2_row_sequential_b32/
active_learning/qgeognn_v2_row_short_sequential_b32/
active_learning/qgeognn_v2_row_small_batch_benchmark/
active_learning/qgeognn_v2_same_state_branching/
active_learning/qgeognn_v2_same_state_branching_cw_hybrid/
active_learning/representation_gp_benchmark_20261104/
active_learning/scientist_v3_responses_2026-09-29/
active_learning/verifiable_scientific_agent_20261030/
predictor/4g_source_benchmark/
predictor/clean_4g_baseline_qualification/
predictor/clean_qgeognn/
predictor/final_4g_qualification/
predictor/final_v2_engineering/
predictor/historical/
predictor/performance_regression_audit/
predictor/r2_pruned_requalification/
```

`reports/`全部28个直接组退出：

```text
active_learning_latest/
active_learning_status_20261005/
cw_retry_evidence_20261106_1445/
dual_expert_two_seeds_to1005_20261021/
dual_expert_two_seeds_to813_20261014/
five_methods_cw_gradient_20261103/
free_cw_progress_20261102_1633/
free_cw_progress_20261103/
free_first_vs_second_20261103/
free_first_vs_second_20261106_1434/
free_lcmd_five_seeds_to1485_20261026/
free_llm_five_seeds_to1005_20261023/
free_llm_five_seeds_to1485_20261024/
free_rerun_audit_20261103/
latest_five_strategy_comparison_20261018/
llm_all_strategies_20261010/
llm_eight_to1165_20261014/
llm_eight_to1293_20261015/
llm_eight_to1421_20261017/
llm_eight_to1549_20261018/
llm_four_to1005_20261013/
llm_free_complete_20261013/
llm_hybrid_complete_20261010/
llm_model_switch_20261012/
llm_progress_20261006/
llm_progress_20261008/
nocw_progress_20261030/
nocw_progress_to1389_20261101/
```

`experiments/`以下14组整体退出，其余7组只保留必要依赖：

```text
4g_error_landscape/
d04_conformer_selection/
d45_oracle_marginal_utility/
d46_oracle_utility_reliability/
e0_3b_controls/
e0_3c_loss_controls/
e2_4g_active_learning/
e2_4g_compound_active_learning/
e2_4g_compound_preflight/
e2_compound_failure_audit/
e2_random_smoke/
e4_a2a_low_budget_formal/
e4_active_transfer_preregistration/
reproductions/
```

保留 **47文件 / 28,366,568字节**，文件模式及blob均与归档完全一致：

| 原实验组 | 保留文件 | 必要原因 |
| --- | ---: | --- |
| e0_4g_baseline | 15 | 原始4g基线、checkpoint、scaler、graph cache与身份/权限来源 |
| e1_signal_qualification | 10 | 两份源数据成员与来源身份 |
| d28_al_engineering | 13 | 12份权威partition及manifest |
| data_audit | 4 | 原数据语义、来源和问题登记 |
| e4_a2a_low_budget_formal | 3 | 登记的低预算partition |
| e0_8g_transfer | 1 | 现有8g图缓存 |
| g0_3_threshold_sensitivity | 1 | 权威8g canonical数据 |

保留登记中的12个受保护路径及其成员；稳定加载器仍要求显式输入，不把旧实验目录变成新的隐式运行默认值。`dataset/`9份CSV原blob全部不变，仅README更新。

`docs/repository/`、`docs/research/`旧施工/审计组整体退出，`docs/model/`只留当前输入合同并新增V2代码合同。3份原机器合同从docs原字节迁入 `configs/{protected_artifacts,data_consumption_register,historical_protocols}.json`，消费者更新为现用位置；历史registry中的绝对路径仅是旧元数据，当前恢复保护从本仓库worktree registry发现，不读原main。

`tools/{audit_research_artifacts,build_migration_inventory,verify_cleanup_release,verify_runtime_isolation}.py`退出；保留通用历史恢复、当前文档检查、原aggregate曲线绘图。`tests/phase2/`及`tests/document_cleanup/`的旧跟踪路径全部退出，核心断言迁至按功能命名的测试。`configs/examples/`旧绘图配置退出。

## 科研索引与科学身份

`research/`仅5份精炼Markdown和2份原字节aggregate，不复制逐轮selection、原始预测或请求。摘要均固定seed、split、预算、模型/训练来源及标签消费边界，完整协议和原始结论以973ccaf8归档为准：

- [预测器资格](../research/PREDICTOR.md)：独立V2最终3seed×row/compound，保留已暴露test评价的说明。
- [五seed Free/LCMD](../research/FIVE_SEED_AL.md)：validation、L1485及实际曲线；Free终点均值0.334085、LCMD0.347582，但Free AULC 0.439159劣于LCMD 0.417049。模型前缀/授权例外不隐去，不宣称干净的模型因果对照。
- [CW/NoCW](../research/CW_NOCW.md)：CW-Free只指向main固定b1a6ed7a快照，不推断live完成进度；NoCW固定Nov1共同L973，不称L1389已完成。
- [负结果](../research/NEGATIVE_RESULTS.md)：GP20state/720fit为开发阶段NO_GO；IVR机制25state为无标签STOP_BEFORE_GATE_C。IVR sequential的417行test最终评价与validation曲线口径分开，未混为统一最终结果。

只读取已有公开aggregate与明确授权的L0/validation冻结fixture；本轮没有新揭示真实U/test真值，没有正式训练或付费LLM请求。

## 行为保持与长期验收

`src/`38文件、`application/`4文件、`scripts/`2文件，以及 `configs/method_registry.json`、`configs/offline_synthetic.json` 的Git blob与清理前逐一相同，模型/loss/预处理/选样核心公式、Prompt和正式结果未改。旧科学/工作流测试124个assert AST完整保留；旧一次性清理测试及施工脚本由归档承担。新增测试直接检查公共行为和独立参考，不要求机械维持旧测试数。

精简后标准自动发现 `python -m pytest -q`：**116 passed，0 skipped**，分为66 unit、37 historical_archive、13 frozen_regression（2026-11-11实测）。分类marker为Python标识符 `unit`、`historical_archive`、`frozen_regression`。

| 能力 | 实际证据 |
| --- | --- |
| Random/LCMD/IVR/MaxDet/Hybrid | 独立数值参考、排序、tie、reference、seed/随机profile、奇异输入与合法性检查；33条数值回归记录身份/顺序/trace一致 |
| 表示/模型 | latent、gradient、center-width、fused；预处理、六输出布局/损失接口、权限与维度保护 |
| 冻结真实checkpoint | 5seed（157/887/2357/6101/12203）L333，各16行授权L0/validation；原始六输出及hidden最大绝对差0，权重精确，图重建bitwise一致；梯度/表示在冻结fixture下回归 |
| 历史真实AL | seed157三个LCMD批次L333→365、L525→557、L973→1005，全部32个ID及顺序一致；不执行新一轮真实AL |
| 工作流/Agent | freeze前禁止reveal、标签边界、fit/validate/feedback/冻结/恢复、身份/hash篡改拒绝、计划合法化与候选包，不含真实provider |
| 日常工具/资源 | 三份迁移合同、47资源blob、固定历史恢复；当前文档导航；真实5seed590点aggregate原序列绘图，无平滑/外推或输入修改 |

真实fixture只使用此前已授权的 `/private/tmp/column_phase2_science_snapshots/` 与 `/private/tmp/column_phase2_frozen_acquisition_inputs/`，从未借用live main。私有fixture缺失时另行验证为 **13 BLOCKED skips**；没有把缺失当作通过。历史回归显式读取固定Git源码或纯AST参考，不自动网络fetch，不作为无私有unit依赖。

## 独立clean clone验收

**PASS（2026-11-12）**。用 `git clone --no-local --no-hardlinks --depth=1 --single-branch --no-tags` 从本地仓库的文件URL独立克隆已提交精简树；此时无需把代码分支推远端。没有 alternates、共享/硬链接对象、sparse配置或原工作树文件。未取 archive tag；`PYTHONPATH` 与全部私有fixture环境变量清空，禁用用户site包，历史fixture指向不存在目录。

- 标准 `python -m pytest -m unit -q`：**66 passed，0 skipped**；包含独立数学参考、权限失败保护、模型接口、工作流、CLI、机器合同、文档和工具测试。
- 冻结回归无fixture：**13 BLOCKED skips**，没有从main借用资源。
- 模型/AL/工作流/工具的 `__file__` 全在新clone；全部当前JSON合同可加载。
- CLI首次运行与resume：两个回合，预算2→6，3次合成fit、2次freeze/reveal；两次结果及全部持久化文件的SHA256相同。真实训练、真实标签和付费请求均为0。
- 当前相对链接通过；浅克隆不存在的固定历史对象明确标为未核验，没有隐式fetch。远端历史可取回的证明来自前述独立HTTPS归档clone。

独立克隆证据：`/private/tmp/framework_v2_20261111/CLEAN_CLONE_FINAL.json`。测试对象以除本验收报告外的路径/模式/blob身份摘要绑定：`4a7a5ca043f9fbfbc2cad3d1279439e88d1177527e4c81e6912e15bbb4965085`。最终报告补充不改变所验收代码、配置、资源或测试；最终提交的独立复核另记 `CLEAN_CLONE_FINAL_RECHECK.json`。

## main保护与证据边界

原main仍为 `b1a6ed7a7bdca05487eb13ddd7b62c8a70870b28`。只读核对623份原源码、13份固定控制/来源身份文件的SHA和inode、main index均未变，main暂存为空；原运行产生的未提交状态未stage/覆盖。不以worker PID或进度永远不变为条件；本任务没有向main写入、发信号、停止/启动恢复控制器。原登记的10个ignored文件路径均保留，不承诺运行缓存字节永不自然变化。

原项目分支仍只有main、迁移学习 `codex/archive-transfer-20261028` 和本重构分支。历史恢复目的地为独立普通目录，没有新增注册worktree。没有merge/rebase/reset/clean main。

大量证据仅在 `/private/tmp/framework_v2_20261111/`：`ARCHIVE_VERIFICATION.json`、`ARCHIVE_TREE_CHECK.json`、`TRIM_RESULT.json`、`MAIN_FINAL_CHECK.json`、`FINAL_TREE_STATS.json`、`CLEAN_CLONE_FINAL.json` 及 `tests/after_trim_{all,unit}.xml`、冻结模型/算法回归JSON；不将它们复制到日常docs。临时文件可能由系统清理，远端固定历史身份和必要小manifest在Git保留。本报告不宣称私有资源、旧环境或生产能力因此完整可用。
