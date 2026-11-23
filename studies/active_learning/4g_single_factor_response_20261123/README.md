# 4g 柱色谱单因素响应与 QGeoGNN 局部学习

从 [完整中文报告](REPORT.md) 开始；复查可见 [数据审计](DATA_AUDIT.md)、[模型响应](MODEL_RESPONSE.md)、[主动学习含义](AL_IMPLICATIONS.md) 和 [下一阶段决策](NEXT_STAGE_DECISION.md)。

本研究用正式 canonical、qualification 冻结 q50、New Free32 的既有冻结状态与授权反馈。物理分析使用固定 qualification row seed 42 的非 test 行；模型主分析只看对应验证集及原轮出来后才可见的反馈，不使用任意 AL 未揭示候选标签做模型评估，不做新拟合或 rollout。

**结论：** EA 对迁移位置的影响强而且较易学到，但不是统一线性响应；上样量主要改变窗口宽，而不是只平移窗口。对上样量，模型能改善全局绝对预测而局部响应仍很弱。LCMD 的优势不需要用“数据近似线性”解释。

## 复现

在仓库根目录、使用项目已有 `fish` 环境（Python、NumPy/pandas、SciPy、scikit-learn、RDKit、PyTorch/PyG、Matplotlib、pytest）：

```sh
D=studies/active_learning/4g_single_factor_response_20261123
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/single_factor_mpl"
python "$D/scripts/run.py"
python "$D/scripts/diagnostics.py"
python -m pytest -p no:cacheprovider -q "$D/scripts/test_response.py"
```

`run.py` 只写本研究的 `tables/`、`figures/` 和 `manifest.json`。历史 runtime 中的盲图、checkpoint 与 feedback 为本地冻结产物，新 clone 若缺这些产物不能完整复现；不要用别的状态补位。`diagnostics.py` 使用本轮表派生形状、配对交互和类别统计，并刷新清单。CSV 的 `low_ids/high_ids` 用 `|` 列出合并进该水平的所有重复行，`series` 是条件键哈希。PNG 与 PDF 同名。所有目录都是只读输入，唯本研究目录输出。
