# 日常离线工具

仅保留固定历史恢复、当前文档检查与已保存曲线的可视化。模型和AL入口见 [架构](../docs/ARCHITECTURE.md)；`scripts/run_al.py` 当前只执行合成生命周期。

| 工具 | 长期用途 | 边界 |
| --- | --- | --- |
| [history_restore.py](history_restore.py) | 从 `configs/archive_manifest.json` 固定完整历史恢复明确选中的Git路径；旧 scientific_source/phase1_candidate 保留整套源码hash验证 | 不隐式fetch，不借live目录，不执行被恢复的controller/模型；不恢复ignored/私有工件或环境 |
| [check_documentation_links.py](check_documentation_links.py) | 检查当前Markdown相对目标、标题锚点与明确的pinned Git引用 | 不把失效相对链接自动解释成旧版本；本地缺归档对象标未核验，可用严格选项拒绝 |
| [plot_saved_validation.py](plot_saved_validation.py) | 对已公开aggregate `{seed: {method: [{labels, nrmse}]}}` 原序列作图 | 不平滑、平均或外推预算；方法模型前缀/预算可能不同，必须同时看来源说明 |

## 恢复历史原文

先显式取回唯一历史入口，再选择要恢复的路径：

```sh
git fetch origin tag archive/framework-v2-pretrim-20261111
python tools/history_restore.py \
  --path studies/active_learning/qgeognn_v2_ivr_mechanism_audit \
  --destination /private/tmp/column_ivr_history \
  --output /private/tmp/column_ivr_restore.json
```

可重复 `--path`。目标必须是新的独立目录，不能位于任何现用Git工作树中；文件恢复为只读。`--source-only` 只恢复历史源码/config/tests，不展开研究数据；`--protocol scientific_source` 或 `--protocol phase1_candidate` 使用 [历史科学源码合同](../configs/historical_protocols.json) 进行whole-source、旧benchmark hash与已安装包元数据核对。旧登记中的本机路径仅为历史元数据，工具从当前Git的worktree registry发现保护目录，在clean clone中不会读取原main。

这些都是 **tracked source/bytes restoration**，不代表能重新执行历史训练，也不证明 ignored checkpoint、LLM原始请求、凭据、外部资源已备份。工具不导入恢复出的代码；旧hash函数只在明确的legacy source profile下以已验证源码中的纯AST片段检查。

## 当前文档与曲线

```sh
python tools/check_documentation_links.py --output /private/tmp/column_links.json
python tools/check_documentation_links.py --require-archive-objects --output /private/tmp/column_links_strict.json
python -m tools.plot_saved_validation \
  --input research/free_lcmd_validation_curves.json \
  --output /private/tmp/column_saved_validation.png
```

曲线图需要已有的matplotlib依赖，见 `requirements-predictor.txt`。图只写入独立新文件，不修改报告/manifest，不哈希或加载checkpoint，不读取原始标签。返回输入SHA与实际点数，保留每个seed及method的原始预算序列；不同实验不能因同图显示就视为同一科学对照。

历史一次性库存、清理验证器及其测试均由固定archive读取，不作为未来运行依赖。
