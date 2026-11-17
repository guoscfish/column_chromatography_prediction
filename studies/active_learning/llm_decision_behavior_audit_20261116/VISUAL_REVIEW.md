# 独立视觉与来源复核

状态：**通过**。本次只读检视图6、图8及其数据来源；除本审阅记录外未修改已锁定数据或绘图文件。

## 图8：理由与行动正反案例

使用 `view_image` 查看最终 `FIGURES/figure8_decision_cases.png`。四种含LLM策略各一行，左列低校正新颖性、右列高校正新颖性，共八张卡片。方法、seed、预算、随机百分位、阶段、理由摘录及接受ID均可读；未发现文字越界、遮挡或截断造成的数值歧义。理由摘录使用省略号，完整文本与32个阶段案例仍保存在 `decision_cases.csv`。

图中低/高两例来自每种方法的校正新颖性极值，不能据此估计发生率。标题标为 Visible rationale，副标题明确“Hybrid: LLM16 rationale with final batch32 metrics”，卡片百分位也标为batch32；其文本来源是最终接受的stated rationale，没有将文本称作内部推理或校准信念。新增层级标注清晰可读，未改变原数值、ID与正反例布局。高新颖性但联合覆盖不高的Historical与Hybrid例也保留，未把不同探索代理画成等价量。

独立加载 `decision_cases.csv` 和 `hybrid_batches.json`，按图中相同排序核验两张Hybrid卡片：

| seed | round | 选前L | 显示的LLM ID | 属于LLM16 | 属于数值16 | 新颖性随机百分位 | 联合覆盖随机百分位 |
|---|---|---|---|---|---|---|---|
| 157 | 13 | 749 | ddb0a18bcd9c5fc460c2 | 是 | 否 | 0.074 | 0.576 |
| 157 | 2 | 397 | 40fd02db79e028fe05a9 | 是 | 否 | 0.997 | 0.168 |

两ID都来自对应候选理由的首个LLM候选，已解决上一版可能误显示数值pending首ID的问题。图8中Hybrid随机百分位描述完整32点集合，理由及示例ID来自LLM16；不能将整批百分位当作LLM16的独立贡献。固定数值16后的LLM条件随机分析应查 `hybrid_conditional_random.csv`。

## 图6：同状态来源分工

最终 `FIGURES/figure6_hybrid.png` 的四个面板均可读，无图例或标题遮挡。CW距离分位、历史单因素对照、固定联合几何顺序平均归因、当前CW梯度几何相对归因与对应逐批表一致。联合几何贡献跨seed交叉，CW梯度贡献由数值半批主导；标题明确排除因果学习归因。没有将几何分工标作Row性能协同。

## 文字与算术边界复核

- 独立从93个原结构化答复、当轮contract及freeze重数：28个池外提议事件包含16个已测量ID、4个pending ID及8个不在当轮H/U内的未知ID字符串事件；共24个不同ID字符串。不能写成28个唯一候选或统一拒绝原因。
- 71批中LLM与数值半批新增分子分别累计69、29个campaign内首次出现事件。每批新增分子跨半批重叠为零，联合去重后98；exclusive列与各半新增列一致。这不是跨seed全球唯一分子计数。
- Hybrid顺序平均几何归因之和等于联合覆盖增量，不能据此分配学习性能贡献。报告对此边界已有说明。
- `hybrid_text_input_manifest.json` 所存 `states.json.xz` SHA与当前文件一致：`f761b8266fecfed31930de2d6e35340ef41c15a6d26c473c328b3b52f27fa489`。

本次审阅对应图像SHA256：

| 图像 | SHA256 |
|---|---|
| figure8_decision_cases.png | c1844fb0a28e57756a2aa49a28db45f0814e39301a949710aa6590cab84120f8 |
| figure6_hybrid.png | 02a6c44cca3cad2073294900c14d849bd8cc37baa50a882de369efda0974e7f6 |
