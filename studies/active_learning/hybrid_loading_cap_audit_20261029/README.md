# CW16+LLM16 / Free Loading机制复核

先读[REPORT.md](REPORT.md)。范围为4g、QGeoGNN-V2、Row split、seed157/6101、333→813，共15轮。阶段A完成；由于内部LLM的loading差异不稳定、CW组成效应及后段反例，按用户规定暂停阶段B。这里没有新策略闭环结果，没有正式cap预注册。

## 复现

从仓库根目录运行。已验证环境为`/Users/fish/miniforge3/envs/chromatography/bin/python`，需要NumPy2、pandas、SciPy、RDKit、matplotlib、pytest。无需LLM凭据、GPU或网络。所有脚本只写本目录派生产物，不运行训练或label store。

只使用本目录派生CSV重新组装与绘图：

```sh
MPLCONFIGDIR=/tmp/hybrid_loading_audit_mpl OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/Users/fish/miniforge3/envs/chromatography/bin/python studies/active_learning/hybrid_loading_cap_audit_20261029/assemble_audit.py
MPLCONFIGDIR=/tmp/hybrid_loading_audit_mpl OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/Users/fish/miniforge3/envs/chromatography/bin/python studies/active_learning/hybrid_loading_cap_audit_20261029/render_report.py
```

完整读取历史冻结来源（要求本机gitignored runtime齐全；缺失会报错，不自动训练/重建）：

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/Users/fish/miniforge3/envs/chromatography/bin/python studies/active_learning/hybrid_loading_cap_audit_20261029/provenance_analysis.py
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/Users/fish/miniforge3/envs/chromatography/bin/python studies/active_learning/hybrid_loading_cap_audit_20261029/selection_analysis.py
MPLCONFIGDIR=/tmp/hybrid_loading_audit_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/Users/fish/miniforge3/envs/chromatography/bin/python studies/active_learning/hybrid_loading_cap_audit_20261029/geometry_analysis.py
```

按上述顺序后再运行组装、报告；测试和最终产物哈希：

```sh
MPLCONFIGDIR=/tmp/hybrid_loading_audit_mpl OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/Users/fish/miniforge3/envs/chromatography/bin/python studies/active_learning/hybrid_loading_cap_audit_20261029/verify_audit.py
```

这属于离线历史分析重放，不是阶段B的mock选择dry-run。B实现、B约束单元测试和B dry-run均按反证门暂停，不能把分析测试冒称B已通过。

## 产物

| 文件 | 内容 |
| --- | --- |
| `REPORT.md` / `PROTOCOL.md` | 完整回答、逐seed/阶段表、口径和证据边界 |
| `loading_mechanism_findings.md` / `LOADING_CAP_DESIGN.md` | H1–H4支持/反证、B暂停、非注册K候选、后续最小干预规格及成本 |
| `PREFLIGHT_REPORT.md` / `verification.json` / `artifact_hashes.json` | 分析测试和哈希、明确未执行项 |
| `provenance_analysis.py` / `source_audit.json` / `source_hashes.json` / `PROVENANCE_FINDINGS.md` | 原owner、继承副本、128预算点/122唯一fit、120批、2209文件SHA、标签边界和源码谱系 |
| `validation_rounds.csv` / `validation_stages.csv` | 128个validation预算点、32个方法×seed×窗口指标 |
| `selection_analysis.py` / `behavior_profiles.csv` / `BEHAVIOR_FINDINGS.md` | 180条来源行为、旧V-only逐行对齐、四轴strict与宽loading、历史残差/理由审计 |
| `selection_point_flags.csv` / `loading_matched_contrasts.csv` | 4800条点来源视图及每个实际matched peer/axis/时间关系；没有未测量真值列 |
| `reason_behavior_audit.csv` / `reason_id_reference_audit.csv` | 文本代理、原句及冻结catalog引用边界；外分子类比不自动判错 |
| `geometry_analysis.py` / `geometry_profiles.csv` / `geometry_points.csv` / `geometry_audit.json` / `GEOMETRY_FINDINGS.md` | 120冻结状态、90raw+90CW、覆盖、距离分位、16来源同状态边际分解与缺项NA |
| `hybrid_selection_profiles.csv` | 180行整合表；key=seed,round,source；16来源的性能是整个Hybrid32结果 |
| `combined_seed_summary.csv` / `combined_stage_by_seed.csv` | 分母一致的逐seed/逐stage摘要 |
| `cap_behavior_sensitivity.csv` / `hypothetical_cap_binding_rates.csv` | K8/12的冻结行为约束强度，未用validation择优；不是主实验注册 |
| `test_*.py` | 41项轻量分析/突变/边界测试，另有5个unittest subtests |
| 四组PNG/SVG | loading与分子行为、几何分工、分阶段边际覆盖、validation曲线 |

所有比例按来源实际16/32分母；4800点视图有Hybrid子来源/all32重叠，独立轨迹选点是3840条。只有两个独立seed配对，不能作点级或轮次级伪重复显著性检验。新克隆缺runtime时，派生表复现不等于原始来源审计复现。
