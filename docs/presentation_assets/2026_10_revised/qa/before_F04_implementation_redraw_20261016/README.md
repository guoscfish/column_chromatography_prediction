# 柱色谱主动学习科研图表：2026年10月修订版

本目录包含 **9张方法示意图（F01-F09）与4张真实结果图（R01-R04）**。全部使用Python + Matplotlib绘制；中文说明优先，方法名称保留QGeoGNN、LCMD、IVR、MaxDet、LLM等缩写。字体按用户追加要求统一为**微软雅黑**，数学公式使用mathtext。

本次只新增本目录文件，未修改模型训练代码、模型参数、原始实验数据或既有报告。绘图程序不导入仓库训练/选样模块，不加载模型检查点，不读取原始数据标签，不进行新预测、测试集评价或LLM调用。

**2026-10-15配图重构**：参考真实论文中对科学对象与中间表示的组织方式，重画F01-F06、F08与F09的方法部分。使用分子骨架、抽象图编码器、预测分位区间、候选覆盖、空白数据表和实验批次代替大段文字流程框；F06直接用32个方格呈现配额，F09用拟议消融设计表呈现后续机制研究。中文优先、微软雅黑、1.25倍行距、纯白与红灰配色。四张结果图保留实际曲线与评价口径。另修正F08显示窗口：原横轴上限0.73截掉两个429标签起点状态的6个分支点，现按完整数据范围加边距，8个状态/24个分支全部可见；底层指标不变。上一轮PNG与源码保存在`qa/before_pictorial_redesign_20261015/`。

## 参考论文与采用方式

| 论文 | 参考位置 | 本次采用的表达方式 |
|---|---|---|
| [Xu等，Nature Communications，2023：QGeoGNN](https://www.nature.com/articles/s41467-023-38853-3) | 图1的科研场景拆解、图2的分子/图/特征/预测结构 | 以明确的科学对象组织流程，显示中间表示。本项目图形独立重画；原文为手性HPLC保留时间，本项目为柱色谱V₁/V₂体积，不混用输出定义。 |
| [Kusne等，Nature Communications，2020：CAMEO](https://www.nature.com/articles/s41467-020-19597-w) | 图1的决策、执行、分析与反馈闭环 | 用一条主路径和清楚的返回线表达迭代；把决策对象、观测数据与计划中的实验接口分开。保留本项目的实线/虚线语义。 |

以上为构图参考，未复制论文图像、数据、设备、机器人形象或原始配色。`src/scientific_primitives.py`包含本次独立绘制的可编辑矢量构件。分子是示意苯酚/苯骨架，不指代本研究选中的具体样本；抽象模型末端的两组三点对应V₁/V₂各自q10/q50/q90，内部节点和特征槽不表示实际层数或维度。空白表格没有虚构实验值，固定概念点集没有作为降维或真实样本分布使用。


## 快速查看与插图

- [前三张图拼接预览](figures/preview_F01_F03.png)
- [F04-F09预览](figures/preview_F04_F09.png)
- [四张结果图预览](figures/preview_R01_R04.png)
- [全部图预览](figures/preview_all.png)
- [13页矢量PDF审阅图集](figures/all_figures_review.pdf)
- [机器可读图片索引](figure_index.csv)

每张图均为 **16 × 9 inch**；PNG为 **4800 × 2700 / 300 dpi**。SVG保留`<text>`文本元素，使用`svg.fonttype='none'`，可在矢量编辑器中修改文字与几何元素；数学公式由可编辑文本片段组成。PDF嵌入字体子集。WPS/PowerPoint中优先插入SVG；如果软件的SVG文本兼容性不佳，可用PNG或PDF。

微软雅黑从本机PowerPoint的字体资源目录加载，没有复制或重新分发字体文件。编辑SVG的另一台电脑需安装微软雅黑以保持一致排版。

## 图片索引与PPT页码

**R01-R04的P11/P12/P14/P15按用户任务指定。方法图页码是建议插入位置**，因为任务未提供原始PPT；本次不修改PPT。仓库旧报告的fig10/fig11/fig12编号与这份新索引不同，数据来源不因页码变化而改变。

| 编号 | PPT页码 | 内容 | 可编辑SVG | PDF | PNG | 内容/数据依据 |
|---|---|---|---|---|---|---|
| F01 | P03（建议） | 从分离条件优化到训练实验选择 | [SVG](figures/F01_research_gap.svg) | [PDF](figures/F01_research_gap.pdf) | [PNG](figures/F01_research_gap.png) | 用户任务；共享模型基础，两个独立目标 |
| F02 | P04（建议） | 并列研究问题与统一评价 | [SVG](figures/F02_research_questions.svg) | [PDF](figures/F02_research_questions.pdf) | [PNG](figures/F02_research_questions.png) | 用户任务；RQ1/RQ2/RQ3并列 |
| F03 | P05（建议） | 主动学习闭环与未来在线接口 | [SVG](figures/F03_active_learning_framework.svg) | [PDF](figures/F03_active_learning_framework.pdf) | [PNG](figures/F03_active_learning_framework.png) | Scientist实现中的选样→冻结→揭示→重训生命周期 |
| F04 | P06（建议） | 表示与选样准则解耦 | [SVG](figures/F04_representation_acquisition.svg) | [PDF](figures/F04_representation_acquisition.pdf) | [PNG](figures/F04_representation_acquisition.png) | `reports/ppt_202610_al/METHODS_APPENDIX.md`及冻结方法源码 |
| F05 | P08（建议） | 化学信息、历史反馈与LLM推理 | [SVG](figures/F05_llm_feedback_reasoning.svg) | [PDF](figures/F05_llm_feedback_reasoning.pdf) | [PNG](figures/F05_llm_feedback_reasoning.png) | `scientist_study.py`、`scientist_selector.py` |
| F06 | P09（建议） | CW16+LLM16 / 自由LLM32 / 双专家 | [SVG](figures/F06_llm_selection_variants.svg) | [PDF](figures/F06_llm_selection_variants.pdf) | [PNG](figures/F06_llm_selection_variants.png) | `scientist_study.py`、`scientist_lifecycle.py`、`dual_expert_study.py` |
| F07 | P10（建议） | 真实线性标签轴与AULC区间 | [SVG](figures/F07_evaluation_protocol.svg) | [PDF](figures/F07_evaluation_protocol.pdf) | [PNG](figures/F07_evaluation_protocol.png) | 初始333、B=32；813/1005分别为15/21轮 |
| R01 | **P11** | Random / Raw-LCMD / CW-LCMD / 传统Hybrid曲线 | [SVG](figures/R01_numerical_learning_curves.svg) | [PDF](figures/R01_numerical_learning_curves.pdf) | [PNG](figures/R01_numerical_learning_curves.png) | 冻结snapshot及逐点fit receipt；333-1005 validation |
| R02 | **P12** | 三种准则下Raw/CW AULC配对 | [SVG](figures/R02_raw_cw_aulc_comparison.svg) | [PDF](figures/R02_raw_cw_aulc_comparison.pdf) | [PNG](figures/R02_raw_cw_aulc_comparison.png) | factorial extension原指标CSV；333-1005 validation |
| F08 | P13（建议） | 同状态+64标签分支与8状态配对AULC | [SVG](figures/F08_state_branching.svg) | [PDF](figures/F08_state_branching.pdf) | [PNG](figures/F08_state_branching.png) | `same_state_branching/results/branch_results.csv`冻结副本；历史test |
| R03 | **P14** | CW32与CW16+LLM16两seed曲线 | [SVG](figures/R03_cw32_vs_cw16_llm16.svg) | [PDF](figures/R03_cw32_vs_cw16_llm16.pdf) | [PNG](figures/R03_cw32_vs_cw16_llm16.png) | 2026-10-11 snapshot；333-813 validation；Astra/high |
| R04 | **P15** | 自由LLM32与相关对照 | [SVG](figures/R04_free_llm_learning_curves.svg) | [PDF](figures/R04_free_llm_learning_curves.pdf) | [PNG](figures/R04_free_llm_learning_curves.png) | Free原始fit audit + 2026-10-13报告；333-813 validation；标出模型切换 |
| F09 | P16（建议） | 稳健性、机制、真实验证的研究计划 | [SVG](figures/F09_research_roadmap.svg) | [PDF](figures/F09_research_roadmap.pdf) | [PNG](figures/F09_research_roadmap.png) | 用户任务；虚线表示计划，非完成进度 |

## 统一视觉规范

色值严格使用：北大红`#9A0001`、深红`#73161D`、辅助红`#D9A9AD`、极浅红`#F4E9EA`、深灰`#343434`、辅助灰`#777777`、浅灰`#C8C8C8`、纯白`#FFFFFF`。平面矩形、细线、点标记与直线/折线箭头构成图形；没有阴影、渐变、3D卡片、设备插画或装饰曲线。

主标题24pt；方法图主标签通常16-19pt，图形注解按信息层级使用10.5-14.5pt；多行说明统一1.25倍行距。中性边框0.8pt、重点边框1.0pt，流程箭头以1.0pt为主；红色主要用于研究重点、选中点和方法曲线。注释与正文分级，输入/输出模块统一对齐；结果图共享纵轴范围，后两个面板省略重复的纵轴数字。

R01的CW数值方法用红色突出；Random、Raw-LCMD与传统Hybrid使用灰色并结合不同标记/线型。R03/R04保持相同LLM实验线条语义：CW32灰色方点实线，CW16+LLM16深红圆点实线；R04增加北大红菱形的自由LLM和灰色虚线Random。单seed面板与均值面板使用相同颜色和点形。全部曲线只连接真实记录点，没有平滑、插值、估读或置信带。

## 方法解释与实际代码核对

- **F01**：两条路径共享QGeoGNN方法基础，但分别优化分离结果与模型的数据效率；没有跨路径的前后连接。
- **F03**：实线是已实现的历史数据离线AL模拟；实验反馈指冻结选样后从历史记录揭示真实标签。虚线只用于计划中的真实自动化柱色谱接口，并标注“未来在线验证”（Future online validation）。数值与LLM模块的配额/组合随方案而异。
- **F04**：Latent是128D隐层表示；Raw与CW是两种输出坐标下的参数梯度，再用CountSketch压缩到512D。CW定义为`C=(V1+V2)/2`、`W=V2-V1`，**不是用C、W两个标量充当512D特征，也不是训练了新模型头**。右侧每个点集为固定随机种子生成的24个概念点，不是实际实验数据。LCMD表示补充覆盖，MaxDet表示互补方向，IVR只给参考集代理方差缩减的数学表达，没有伪造方差热力图。
- **F05**：只有已标记实验提供实测V1/V2及残差；未标记候选和待测CW16只能提供特征、预测与覆盖信号。验证集/test信息不进入LLM选样包。“不确定性”是代理信号，不是精确QGeoGNN后验。“假设”是待检验的选样解释。
- **F06-A**：代码先用CW-LCMD选出16个pending ID，再把这些ID排除出LLM合法候选池；LLM可以看到pending实验的预测但不能看到其标签，最终批次为`pending CW16 + LLM16`。这有明确顺序依赖，不能画成彼此独立的两个16样本选择后任意去重。重复/非法ID会被验证拒绝，不会静默去重、随机补齐。
- **F06-B**：Free没有CW pending，也没有预留CW名额；CW距离仍可作为模型覆盖信号，不能因此说它执行了CW选点。
- **F06-C**：两个科学家的当前建议互相不可见，规划者获得两份完整建议并负责最终32个唯一合法ID。实际代码按化学科学家→ML科学家的顺序调用，**建议在信息上独立，但并非宣称调用并发**。没有固定专家配额、自动排名聚合或隐藏重排。三种方法都先冻结32个ID与测量前预测，再揭示标签。
- **F08**：横轴自动包含全部24个分支点并检查可见数量，记录于`qa/branch_plot_range.json`。真实表包含8状态/24分支，每分支两轮32标签。右图仅比较同一行状态内的三种策略；不把不同状态的绝对误差差异当成策略能力排名。short-AULC复算后LCMD/IVR/MaxDet胜出次数为4/4/0。
- **F09**：所有阶段及预期贡献都是后续计划，虚线不表示已完成。

## 指标与共同训练协议

Combined NRMSE不是两个normalized RMSE的算术平均，而是：

\[
e=\sqrt{\frac{1}{2}\left[\left(\frac{\mathrm{RMSE}_{V1}}{s_{V1}}\right)^2+\left(\frac{\mathrm{RMSE}_{V2}}{s_{V2}}\right)^2\right]}.
\]

尺度固定为各seed初始L333训练标签的population std，不从验证集/test重新估计：seed157的`sV1=6.502957820892334, sV2=13.158523559570312`；seed6101的`sV1=5.462151050567627, sV2=12.266972541809082`。

\[
\mathrm{AULC}_{a:b}=\frac{1}{b-a}\sum_i (n_{i+1}-n_i)\frac{e_i+e_{i+1}}{2}.
\]

NRMSE与AULC均越低越好；不同标签区间的AULC不能混合排名。F08的short-AULC为`(e0 + 2e32 + e64)/4`。两seed均值只是逐预算或逐seed摘要的算术平均，不构成置信区间。

R01-R04共同采用QGeoGNN-V2（458,952个有效参数）、Row划分、外层seed157/6101、outer train3330、固定validation416、test417、L0=333、AL batch32。各seed内已核对validation ID hash与初始化hash相同。

训练为同初始化的每轮scratch fit：Adam、lr0.001、weight_decay0、训练batch2048、最多1000 epoch、patience100、V1/V2 loss权重均为1、按validation Combined NRMSE选checkpoint，不使用test标签作训练或checkpoint选择。初始化seed为`outer_seed + 2,000,003`。实际早停epoch不同不等于最大训练预算不同。

## 每张结果图的来源、协议与边界

### R01：P11数值策略学习曲线

主要来源为`reports/ppt_202610_al/frozen_inputs/reports/active_learning_latest/snapshot.json`（captured_at：2026-10-11T23:28:59+08:00，原报告冻结commit：`42ab12600a353bba4a063ef5d1b623ee2f7b976a`），逐点来源由`reports/ppt_202610_al/overview_fit_receipts.csv`提供。新绘图表：[R01_learning_curves.csv](data/R01_learning_curves.csv)。每条方法每seed完整22点（333+32k，k=0…21）。所有点均与已完成fit audit核对。

Random与Raw-LCMD来自`qgeognn_v2_row_sequential_b32`；CW来自short sequential及CW-LCMD到525/653/1005的续段；传统Hybrid为K=3 ensemble不确定性门控 + latent farthest-first，**不是CW+LLM，也不是Raw/CW表示的简单融合**。传统Hybrid有额外ensemble拟合，图只匹配训练标签预算；CW的525后阶段为看到开发结果后的探索性延伸。

### R02：P12 Raw/CW AULC

原始指标表为`reports/ppt_202610_al/frozen_inputs/studies/active_learning/qgeognn_v2_row_cw_selector_extension_to_1005/validation_curves.csv`；逐点比对带split/batch标签的`reports/ppt_202610_al/fig11_validation_curves.csv`，再与`fig11_summary.csv`核对梯形AULC。新绘图摘要：[R02_aulc.csv](data/R02_aulc.csv)。柱高为两seed均值，空心圆为157、实心方点为6101，细线连接同seed配对。

| 准则 | Raw Gradient AULC | CW Gradient AULC |
|---|---:|---:|
| LCMD | 0.5186718485845807 | 0.5000286815055881 |
| IVR | 0.5424333915431866 | 0.5199212231465139 |
| MaxDet | 0.5542921723214300 | 0.5748327739532934 |

与用户提供的四位小数表一致，实际绘图和复算采用CSV全精度。LCMD不做额外RMS；MaxDet使用当前L的RMS；IVR使用固定L+U的RMS，因此不同准则之间还有尺度差异。原333-525为开发factorial，525-1005为探索性延伸；全区间展示不改写原阶段门槛。MaxDet两个seed的CW相对Raw方向不一致。

### R03：P14 CW32与CW16+LLM16

来源为同一2026-10-11 snapshot，截取**两seed共同完整的333-813区间**，每条方法每seed16点；新绘图表：[R03_learning_curves.csv](data/R03_learning_curves.csv)。CW16+LLM16使用`astra_high_full_pool_20261001`修订版的Astra/high；CW32是审计过的复用数值对照。共享QGeoGNN训练协议，后续各方法训练标签集合与权重不同。

### R04：P15自由LLM与对照

Free原始数值直接读取`reports/llm_free_complete_20261013/source_hashes.json`列出的32份已完成fit audit，逐个核对SHA与`validation_curves.json`，并复算`completion_audit.json`中的AULC。控制曲线采用R03相同冻结快照，另与2026-10-13报告逐点确认一致。新绘图表：[R04_learning_curves.csv](data/R04_learning_curves.csv)。

Free的协议来源为`studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/astra_high_full_pool_20261001/protocol.json`及`revisions/free_sol_xhigh_continuation_20261012/transition_manifest.json`。seed157在525标签后、seed6101在653标签后，从`gpt-6-astra/high`切换为`gpt-6.1-sol/xhigh`；对应首次受切换影响的模型评价点为557和685。图中虚线标的是**发起新选样时的切换预算**，并非在边界处重算旧模型。

两seed的混合前缀/续段属于探索性LLM模型切换轨迹，不是纯Sol轨迹。QGeoGNN训练与选样预算不变，但LLM决策模型改变，不能据此隔离LLM型号、化学知识或反馈的因果作用。均值面板也明确标出包含不同预算切换。历史格式偏离保留：seed157一轮有额外回复及人工授权字符修正；seed6101首轮有额外回复。未把这些偏离隐藏成严格无人工干预的确认实验。

### F08中的真实结果补充

来源为`reports/ppt_202610_al/frozen_inputs/studies/active_learning/qgeognn_v2_same_state_branching/results/branch_results.csv`，新表：[F08_branch_results.csv](data/F08_branch_results.csv)。两seed × LCMD/IVR两条来源轨迹 × 429/653两锚点，共8状态；每状态内同checkpoint/L/U启动三策略，每分支+64标签。

该表是原实验全部分支选样、拟合、test预测冻结之后揭示的**历史test**结果，本次只读取保存的数值表，不读test真值、不重算test预测。它与R01-R04的validation指标不作绝对数值混合排名。8个状态来自同一两seed历史轨迹，并非8个独立seed。

## 完成情况与数据缺口

**F01-F09与R01-R04全部可靠生成，没有因原始指标缺失而省略的图。** 368条绘图fit映射均找到本机原文件，SHA及指标核对通过；R02原曲线CSV可复算完整AULC；F08完整24分支表可复算8状态结果。没有从图片估读数据。

这是一套固定来源的科研汇报素材：R01-R03使用2026-10-11冻结快照，R04使用2026-10-13完成记录；不自动追随本机同时运行中的后续实验或把新的LLM版本续段追加到当前图。要更新区间/实验版本，需先重新核对协议再显式更新冻结输入。

## 目录与复现

```text
2026_10_revised/
  README.md
  figure_index.csv
  src/                 完整绘图、数据核对、PDF合并与检查源码
  figures/             13套SVG/PDF/PNG、4张拼接预览、13页审阅PDF
  data/                实际绘图表、逐点fit映射、来源清单与冻结输入
  qa/                  数据/布局/输出审计、PDF渲染检查
```

本机复现全部图（只读取本目录冻结表，写入本目录）：

```bash
/Users/fish/miniforge3/envs/fish/bin/python docs/presentation_assets/2026_10_revised/src/build.py
```

其他环境先安装`src/requirements.txt`，并安装合法的微软雅黑字体；用环境变量`MSYH_FONT_DIR`指定含`msyh.ttc`和`msyhbd.ttc`的目录，然后运行`python src/build.py`。PDF合并需要pypdf；本机绘图环境没有pypdf时，程序自动使用Codex内置Python，也可通过`--pdf-python`指定。

可分别运行`src/draw_methods.py --figures F01 F02 F03`、`src/draw_methods.py --figures F04 F05 F06 F07 F08 F09`、`src/draw_results.py`。对仓库原始指标做完整只读复核，运行`src/prepare_data.py`或`src/build.py --audit-sources`；冻结来源发生变化时会停止，不会静默替换旧实验。

数据追溯：`data/source_manifest.json`保存仓库原路径、冻结副本路径、SHA-256及字节数；`data/plotted_fit_receipts.csv`给出R01/R03/R04每点与原始fit audit的对应；`data/result_summary.csv`保存各seed与均值的AULC/终点。`qa/data_audit.json`记录预算网格、指标一致性、原始文件hash、validation身份与模型初始化一致性检查。

最终布局检查覆盖文字溢出、模块内文字边界和文字之间重叠；另用Poppler逐页渲染13份PDF检查字体、公式、图例、箭头、轴标签及留白。文件尺寸、SVG文字可编辑性、颜色集合与PDF字体嵌入检查保存在`qa/`。图集PDF按F01-F09、R01-R04顺序保留书签，方法示意与结果图区分明确。
