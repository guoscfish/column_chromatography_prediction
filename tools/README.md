# 离线工具

状态：`CURRENT`。这些工具处理库存、历史恢复、运行保护或已有曲线，不是科学实验 runner。候选唯一回合入口 [scripts/run_al.py](../scripts/run_al.py) 目前只开放合成生命周期，见 [系统架构](../docs/ARCHITECTURE.md)。

| 工具 | 用途 | 写入与边界 |
| --- | --- | --- |
| [build_migration_inventory.py](build_migration_inventory.py) | 二期源码库存、多类型依赖图、删除证明与结构计数 | 读取固定基线和候选，不读真值/checkpoint。默认输出为二期验收目录，复查应写独立临时目录，不覆盖旧收据。 |
| [history_restore.py](history_restore.py) | 固定 scientific_source / phase1_candidate 的源码恢复、hash与环境核验 | 新建 source-only detached worktree；目标须不存在且在 main/candidate 之外，不展开数据或执行旧 controller/provider/training。 |
| [verify_runtime_isolation.py](verify_runtime_isolation.py) | 二期 Gate0 的 source/frozen 身份与工作树隔离 | 固定二期验收工具，绑定 architecture 分支及限定的诊断提交收据。不能将新一轮 main 的合法推进判成清理修改；本轮保护须单独固定 baseline。 |
| [plot_saved_validation.py](plot_saved_validation.py) | 保存的 validation snapshot / rerun-audit 离线绘图 | 向配置指定 snapshot_directory 写图；rerun-audit 还核对已有模型与历史。使用独立授权快照，不随意重绘运行中报告目录。 |
| [audit_research_artifacts.py](audit_research_artifacts.py) | 本轮逐文件研究产物、保护边界与类型引用审查 | 只生成计划；`--compress`将大清单压缩为确定性gzip。没有自动删除；闭合候选组不能替代实际删除子集的依赖复核。 |
| [check_documentation_links.py](check_documentation_links.py) | 全量Markdown相对链接、anchor与固定Git来源 | 区分Git稀疏目标和实际缺失；当前导航失败即非零退出；历史原文独立解析且不改写。 |
| [verify_cleanup_release.py](verify_cleanup_release.py) | 本轮main保护、科学代码不变、实际删除集合与远端恢复收据复查 | 绑定本轮main和原重构分支；核对固定独立fetch收据，本工具本身不重新联网、不删文件。 |

本轮新文档/产物审计工具和收据从 [工程审计导航](../docs/repository/README.md) 查找。上述四个既有工具用途独立，不作为重复历史脚本删除。

## 复核命令

从候选根目录将源码库存写到独立目录：

```sh
python tools/build_migration_inventory.py --repository "$PWD" --candidate "$PWD" \
  --source-commit 4ee480378b0293f1985e3ee4a09df2c78f77901e \
  --output /private/tmp/column_migration_recheck
```

此工具针对 Python 迁移，不是研究产物删除授权，不修改原冻结清单。

恢复固定科学来源到新独立目录：

```sh
python tools/history_restore.py --protocol scientific_source \
  --destination /private/tmp/column_history_source \
  --output /private/tmp/column_history_verification.json
```

`--protocol phase1_candidate` 恢复一期候选。固定来源见 [HISTORICAL_PROTOCOLS.json](../docs/repository/phase2/HISTORICAL_PROTOCOLS.json)，更早研究使用自己的 BASE_COMMIT。源码恢复不等于被忽略的 runtime、真实 checkpoint、完整请求或凭据已备份；远端来源与本地独有 ref 分别核查。

工具不安装或升级运行中环境。不要为了文档清理运行旧恢复控制器、改锁、重训练、接受新选点或发起付费 LLM。
