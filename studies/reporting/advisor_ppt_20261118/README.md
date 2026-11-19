# 柱色谱主动学习导师汇报素材

2026-11-18 · 4g Row validation · 历史证据终审

从 [PPT_ASSETS.zip](PPT_ASSETS.zip) 取素材。[逐页讲法](PPT_FACT_SHEET.md) 给出 P01–P12 的建议排法。[逐图清单](FIGURE_MANIFEST.md) 标注每幅图的口径、适用结论、来源及局限。[指标核验](METRICS_AUDIT.md) 和 [统一指标 CSV](metrics_source.csv) 可供追查数字。

- `plots/`　9 张图，每张含 4000 × 2250 PNG、SVG 和矢量 PDF。
- `rendered_previews/`　经 Poppler 从交付 PDF 直接渲染的预览。
- `scripts/`　独立数值复算、绘图、绘图 QA、来源索引和打包。
- `sources/`　绘图使用的核验后表格、来源路径与 SHA256。打包包含这些派生表，复核原始证据还需本仓库的历史归档。
- [缺失与执行记录](MISSING_INPUTS.md)　远端与当前分支、活跃研究的处理、实验缺口和未通过项。

## 复现

先在原仓库用已有 Python 环境运行。基础绘图用 Matplotlib、NumPy、pandas 和 Pillow，PDF 预览用 Poppler。默认 macOS 中文字体为 STHeiti Medium。跨平台可设置 `ADVISOR_CJK_FONT` 指向具备简体中文字形的 TrueType 字体。在素材 ZIP 解压后，绘图脚本可直接用包内的 `sources/` 数据重画，原始审计脚本需在仓库的报告目录运行。F09 单独重绘用 `python -B scripts/audit_backup.py --plot-only`。

```bash
export MPLCONFIGDIR="$(mktemp -d)"
python -B scripts/audit_performance.py
python -B scripts/audit_selection_behavior.py
python -B scripts/audit_model_figures.py
python -B scripts/plot_performance.py
python -B scripts/plot_selection_behavior.py
python -B scripts/plot_model_figures.py
python -B scripts/audit_backup.py
python -B scripts/verify_assets.py
python -B scripts/package_assets.py
```

报告中的数据流仅读取许可范围内的历史已保存产物。本轮没有新增拟合、选点、LLM 调用、标签揭示、未来 U 真实响应或 test 真值访问。
