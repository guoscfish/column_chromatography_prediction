# 原仓库重构分支与运行目录隔离

最终仓库：`/Users/fish/Documents/GitHub/column_chromatography_prediction`。

| 用途 | 工作目录 | 分支 / 提交 |
| --- | --- | --- |
| 既有 CW-Free 实验 | `/Users/fish/Documents/GitHub/column_chromatography_prediction` | main，`16eca74769f664bc378a58ccbf380a806723f587` |
| 工程重构 | `/Users/fish/.codex/worktrees/repository-architecture-20261109/column_chromatography_prediction` | `codex/repository-architecture-20261109` |

两个工作目录属于同一个 Git 仓库，common Git directory 为原仓库的 `.git`；各自的工作文件、HEAD 和 index 独立。工作树是同一项目的分支视图，不是新的 clone 或 GitHub 项目。

用户最后确认沿用原仓库分支后，已将原候选的 `0c537fe7`、`ddf36aa1` 原提交迁入该分支。原目录当时检出重构分支，但 HEAD/main/重构分支均为同一基线SHA。先仅将 HEAD 符号归属恢复 main，未执行 checkout/switch，也未改源码或 index；随后才推进已释放的重构分支引用。迁入前后原 HEAD/main SHA、272个 src/application/scripts 源文件与 index SHA-256 均相同。

不要在实验目录切换到重构分支：Git 分支本身不为进程隔离文件，checkout 会改变实验读取的路径。在 CW-Free 完成并通过独立切换决定前，继续在上面的重构工作树编辑和提交。

## 两个旧副本

`/Users/fish/Documents/GitHub/column_chromatography_refactor` 的所有已提交改动已迁入原仓库分支。

`/Users/fish/Documents/GitHub/column_chromatography_lab` 不是运行依赖。其完整提交 `71e50d7911a4dacf2a317810778cd4a99e67ca24` 保留于原仓库 `refs/archive/column_chromatography_lab-20261109`，包括基线锁、源码快照、登记表、脚本、测试和合成结果。因此可以在核验后删除两个旧副本，不会丢失已交付成果。清理实际结果见 `validation/branch_layout_transfer.json`。

需要恢复 lab 时，可从原仓库生成归档：

```sh
git -C /Users/fish/Documents/GitHub/column_chromatography_prediction archive \
  refs/archive/column_chromatography_lab-20261109 \
  --output=/tmp/column_chromatography_lab-20261109.tar
```

此 archival ref 保存在本地原 Git 仓库；未创建新的远端仓库。未来向其他仓库备份时，应明确包含该 ref。研究方法晋升规程保留在归档 README 中，本次 API 不要求恢复 lab。

## 验证

`python scripts/verify_refactor_integrity.py --output /tmp/column_branch_integrity.json` 检查独立工作树 HEAD/index、同一 Git common directory、原 main/源码/身份、263个冻结源码、28个历史兼容路径、保护对象可恢复性和2条获批退役。冻结模型与科研计算未因目录迁入而修改。最终迁入测试与完整性结果见 `validation/`。
