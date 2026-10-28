"""Scientific method diagrams, authored as flat 2D Matplotlib vectors."""
import argparse
import numpy as np
import pandas as pd
from style import *


def f01():
    fig, ax = canvas("F01", "从分离条件优化到训练实验选择")
    text(ax, 8, 7.61, "相同的QGeoGNN基础，不同的优化目标", 17, GRAY)
    centers = [4.1, 11.9]
    titles = ["分离条件优化", "训练数据采集优化"]
    subtitles = ["已有研究", "本研究"]
    paths = [
        ["已训练的QGeoGNN", "预测候选条件下的\n$V_1/V_2$ 分布",
         "计算分离指标 $S_p$", "选择更好的分离条件"],
        ["当前QGeoGNN + 候选实验池", "估计候选样本的学习价值",
         "选择下一批训练实验", "获取新标签并更新模型"],
    ]
    ax.plot([8, 8], [1.35, 7.12], color=LINE, lw=.8)
    for j, cx in enumerate(centers):
        text(ax, cx, 6.98, titles[j], 24, RED if j else INK, weight="bold")
        text(ax, cx, 6.55, subtitles[j], 16, RED if j else GRAY)
        ys = [5.22, 4.03, 2.84, 1.65]
        for i, (y, label) in enumerate(zip(ys, paths[j])):
            box(ax, cx-3.05, y, 6.1, .9, label, accent=bool(j), size=18)
            if i < 3:
                arrow(ax, (cx, y), (cx, ys[i+1]+.9), RED if j else GRAY)
        text(ax, cx, 1.01, ["优化分离结果", "提升模型数据效率"][j],
             22, RED if j else INK, weight="bold")
    footer(ax, "两条路径相互独立；共享模型基础，任务目标不同。")
    save(fig, "F01_research_gap")


def f02():
    fig, ax = canvas("F02", "科学问题与研究路线")
    box(ax, 2.3, 6.75, 11.4, .87, "面向柱色谱的数据高效学习",
        accent=True, size=22, weight="bold")
    centers = [3.1, 8, 12.9]
    for cx in centers:
        route(ax, [(8, 6.75), (8, 6.35), (cx, 6.35), (cx, 5.89)], LINE, lw=1.2)
    cards = [
        ("RQ1", "数值选样", "怎样衡量实验价值？", "样本表示\nLCMD / IVR / MaxDet"),
        ("RQ2", "状态依赖策略", "采样收益是否依赖模型状态？", "同状态分支实验"),
        ("RQ3", "反馈引导的LLM", "化学知识与实验反馈\n能否提供额外信息？", "CW+LLM / 自由LLM\n双专家LLM"),
    ]
    for cx, (rq, label, question, method) in zip(centers, cards):
        box(ax, cx-2.18, 2.56, 4.36, 3.34)
        text(ax, cx, 5.46, rq, 20, RED, weight="bold")
        text(ax, cx, 4.92, label, 18, INK, weight="bold")
        text(ax, cx, 4.10, question, 19)
        ax.plot([cx-1.7, cx+1.7], [3.48, 3.48], lw=.8, color=LINE)
        text(ax, cx, 2.99, method, 17, GRAY)
        route(ax, [(cx, 2.56), (cx, 2.09), (8, 2.09), (8, 1.66)], LINE, lw=1.2)
    box(ax, 2.3, .88, 11.4, .78, "NRMSE  |  AULC  |  达到目标所需标签数", accent=True, size=21)
    footer(ax, "三个问题并列服务于总目标，在相同标签预算下评价。")
    save(fig, "F02_research_questions")


def f03():
    fig, ax = canvas("F03", "数值策略与LLM辅助的主动学习闭环",
                     "当前已实现：基于历史数据的离线主动学习模拟")
    # Six stages remain on one horizontal axis; acquisition internals sit below.
    xs = [.65, 3.02, 5.45, 8.13, 10.87, 13.35]
    ws = [2.0, 2.04, 2.28, 2.32, 2.08, 2.0]
    labels = ["已标记数据集\n$L_t$", "QGeoGNN\n训练", "模型预测与\n样本表示",
              "选样策略与\n批次选择", "实验反馈", "更新数据集\n$L_{t+1}$"]
    for i, (x, w, label) in enumerate(zip(xs, ws, labels)):
        text(ax, x+.12, 5.85, f"{i+1:02d}", 14, RED if i==3 else GRAY, ha="left")
        box(ax, x, 4.64, w, .98, label, accent=i==3, size=17)
        if i < 5:
            arrow(ax, (x+w, 5.13), (xs[i+1], 5.13), RED if i in (2, 3) else GRAY)
    box(ax, 5.42, 6.62, 5.04, .62, "未标记候选池 $U_t$", size=19)
    route(ax, [(6.59, 6.62), (6.59, 6.16), (6.59, 5.62)], LINE)
    route(ax, [(9.29, 6.62), (9.29, 6.16), (9.29, 5.62)], LINE)

    # Explicit offline oracle and a distinct prospective real experiment source.
    text(ax, 11.91, 4.12, "从历史记录中揭示\n所选实验的真实标签", 14, GRAY)
    box(ax, 11.08, 6.49, 4.25, .81, "真实自动化柱色谱平台\n未来在线验证",
        size=15, dashed=True, color=GRAY)
    route(ax, [(11.91, 6.49), (11.91, 5.62)], GRAY, dashed=True)
    # Acquisition modules feed selection in parallel; no fixed combination rule.
    box(ax, 3.30, 1.54, 4.43, 1.79, accent=False)
    text(ax, 5.515, 2.93, "数值选样", 20, INK, weight="bold")
    text(ax, 5.515, 2.19, "梯度表示\nLCMD / IVR / MaxDet", 18)
    box(ax, 8.12, 1.54, 4.43, 1.79, accent=True)
    text(ax, 10.335, 2.93, "LLM辅助选样", 20, RED, weight="bold")
    text(ax, 10.335, 2.19, "化学信息 / 历史实验反馈\n模型状态 / 科学推理", 18)
    route(ax, [(5.515, 3.33), (5.515, 3.67), (8.76, 3.67), (8.76, 4.64)], GRAY)
    route(ax, [(10.335, 3.33), (10.335, 3.78), (9.82, 3.78), (9.82, 4.64)], RED)
    text(ax, 8, 1.15, "组合方式与样本配额随方案而异", 16, GRAY)
    route(ax, [(14.35, 4.64), (14.35, .81), (2.82, .81), (2.82, 4.19),
               (4.04, 4.19), (4.04, 4.64)], RED)
    text(ax, 7.56, .59, "标签更新后重新训练", 14, RED)
    footer(ax, "实线：已实现的离线闭环     虚线：计划接入的真实实验接口", y=.23, size=12)
    save(fig, "F03_active_learning_framework")


def small_space(fig, bounds, points, selected=(), labeled=(), vectors=False):
    ax = fig.add_axes(bounds)
    ax.scatter(points[:, 0], points[:, 1], s=26, facecolors=WHITE, edgecolors=LINE, linewidths=1.1)
    if labeled:
        p = points[list(labeled)]
        ax.scatter(p[:, 0], p[:, 1], s=35, marker="s", color=INK, zorder=3)
    if selected:
        p = points[list(selected)]
        ax.scatter(p[:, 0], p[:, 1], s=58, facecolors=RED, edgecolors=RED, zorder=4)
        if vectors:
            for q in p:
                arrow(ax, (0, 0), tuple(q), RED, lw=1.4)
    ax.set(xlim=(-1.12, 1.12), ylim=(-1.05, 1.1), xticks=[], yticks=[])
    ax.spines[["left", "bottom"]].set_color(LINE)
    return ax


def f04():
    fig, ax = canvas("F04", "数值主动学习：样本表示与选样准则解耦")
    text(ax, .85, 7.45, "A  样本表示", 22, INK, ha="left", weight="bold")
    text(ax, 7.65, 7.45, "B  选样准则", 22, INK, ha="left", weight="bold")
    ax.plot([7.1, 7.1], [1.04, 7.58], lw=.8, color=LINE)
    box(ax, 1.02, 6.15, 5.63, .81, "分子结构 + 实验条件", size=21)
    arrow(ax, (3.83, 6.15), (3.83, 5.73))
    box(ax, 2.08, 4.93, 3.5, .8, "QGeoGNN", accent=True, size=23)
    xs = [1.02, 3.03, 5.04]
    for x in xs:
        route(ax, [(3.83, 4.93), (3.83, 4.45), (x+.805, 4.45), (x+.805, 4.06)], LINE)
    labels = ["隐层表示\n128D", "Raw输出梯度\n压缩至512D", "CW输出梯度\n压缩至512D"]
    for i, (x, label) in enumerate(zip(xs, labels)):
        box(ax, x, 2.65, 1.61, 1.41, label, accent=i==2, size=16)
    text(ax, 3.83, 2.02, r"$C=(V_1+V_2)/2$    $W=V_2-V_1$", 19)
    text(ax, 3.83, 1.35, "CW特征来自模型输出对参数的梯度\n再由CountSketch压缩，512D不是两个标量", 16, GRAY)

    # Fixed synthetic coordinates solely explain feature-space selection geometry.
    rng = np.random.default_rng(104)
    points = np.vstack([rng.normal([-.58, -.3], [.15, .27], (12, 2)),
                        rng.normal([.45, .3], [.22, .30], (12, 2))])
    points = np.clip(points, -.92, .92)
    small_space(fig, [.485, .565, .185, .195], points, selected=(19,), labeled=(0, 3, 8))
    text(ax, 9.14, 7.0, "LCMD", 18, INK, weight="bold")
    text(ax, 12.9, 6.24, "补充覆盖不足的区域\n最大簇 → 最远候选", 18)
    legend_handles=[Line2D([],[],color=INK,linestyle="none",marker="s",mfc=INK,mec=INK,label="已标记"),
                    Line2D([],[],color=GRAY,linestyle="none",marker="o",mfc=WHITE,mec=LINE,label="候选"),
                    Line2D([],[],color=RED,linestyle="none",marker="o",mfc=RED,mec=RED,label="选中")]
    fig.legend(handles=legend_handles,loc="center",bbox_to_anchor=(12.9/16,5.41/9),ncol=3,
               frameon=False,fontsize=12,handlelength=.8,handletextpad=.3,columnspacing=.7)

    points2 = rng.uniform(-.92, .92, (24, 2))
    points2[3], points2[17] = [.91, .15], [.06, .92]
    small_space(fig, [.485, .318, .185, .185], points2, selected=(3, 17), vectors=True)
    text(ax, 9.14, 4.78, "MaxDet", 18, INK, weight="bold")
    text(ax, 12.9, 4.09, "选择互补的信息方向", 18)
    text(ax, 12.9, 3.46, r"$\max_S\ \log\det(A_{L\cup S})-\log\det(A_L)$", 15)

    small_space(fig, [.485, .085, .185, .173], points2, selected=(11,))
    text(ax, 9.14, 2.60, "IVR", 18, INK, weight="bold")
    text(ax, 12.9, 2.04, "降低参考集的平均代理方差", 17)
    text(ax, 12.9, 1.48, r"$\Delta_i=\frac{x_i^T\Sigma_L M\Sigma_L x_i}{1+x_i^T\Sigma_Lx_i}$", 19)
    text(ax, 12.9, .91, r"$R=L_t\cup U_t$，$M=X_R^TX_R/|R|$", 14, GRAY)
    footer(ax, "右侧点集均为概念示意（conceptual），不是色谱实验数据；IVR是线性核代理方差缩减。", y=.26, size=12)
    save(fig, "F04_representation_acquisition")


def f05():
    fig, ax = canvas("F05", "化学知识与实验反馈驱动的LLM选样",
                     "信息边界：真实 V₁/V₂ 仅来自已标记实验；未标记候选只能提供特征与预测")
    inputs = [
        (5.64, "A  化学信息", "分子结构\n官能团 / 洗脱条件", False),
        (3.59, "B  历史实验反馈", "实测 V₁/V₂ / 实验条件\n预测残差", True),
        (1.54, "C  当前模型状态", "模型预测 / 不确定性代理\n候选覆盖情况", False),
    ]
    for y, heading, body, accent in inputs:
        box(ax, .78, y, 4.31, 1.62, accent=accent)
        text(ax, 2.935, y+1.24, heading, 20, RED if accent else INK, weight="bold")
        text(ax, 2.935, y+.57, body, 18)
        arrow(ax, (5.09, y+.81), (6.45, 4.4), RED if accent else LINE, lw=1.4)
    box(ax, 6.45, 3.53, 3.10, 1.74, "LLM辅助\n实验推理", accent=True, size=23, weight="bold")
    for y, label in [(5.72, "候选选择建议"), (4.16, "选样理由"), (2.60, "下一批实验")]:
        box(ax, 11.12, y, 4.0, .92, label, size=20)
        arrow(ax, (9.55, 4.4), (11.12, y+.46), LINE)
    text(ax, 8, 1.75, "根据已观察反馈更新选样假设", 16, GRAY)
    cxs = [2.3, 6.0, 9.75, 13.5]
    for i, (cx, label) in enumerate(zip(cxs, ["观察", "假设", "建议实验", "新反馈"])):
        text(ax, cx, .93, label, 20, RED if i==1 else INK)
        if i < 3:
            arrow(ax, (cx+.68, .93), (cxs[i+1]-.90, .93), GRAY)
    footer(ax, "“假设”是待检验的选样解释，不表示已经发现新的化学规律。", y=.29)
    save(fig, "F05_llm_feedback_reasoning")


def f06():
    fig, ax = canvas("F06", "三种LLM选样机制：相同的32标签预算")
    box(ax, 2.63, 6.80, 10.74, .74, "共同输入：当前模型状态 + 候选实验池", size=21)
    centers = [3.03, 8, 12.98]
    for cx in centers:
        route(ax, [(8, 6.80), (8, 6.43), (cx, 6.43), (cx, 6.12)], LINE)
    for x in (5.48, 10.52):
        ax.plot([x, x], [.89, 6.12], color=LINE, lw=.8)
    for cx, title in zip(centers, ["A  CW16 + LLM16", "B  自由LLM32", "C  双专家LLM"]):
        text(ax, cx, 5.79, title, 20, RED, weight="bold")

    box(ax, 1.04, 4.54, 3.98, .70, "CW-LCMD先选16个", size=18)
    arrow(ax, (3.03, 4.54), (3.03, 3.97), RED)
    box(ax, 1.04, 2.81, 3.98, 1.16, "LLM补充16个\n候选池排除待测CW16\nCW16只有预测、没有标签", accent=True, size=16.5)
    arrow(ax, (3.03, 2.81), (3.03, 1.99), RED)

    box(ax, 6.05, 3.92, 3.90, 1.32, "LLM独立选择\n完整32个实验\n无CW预留配额", accent=True, size=18)
    arrow(ax, (8, 3.92), (8, 1.99), RED)

    box(ax, 10.86, 4.00, 1.95, 1.24, "化学科学家\n独立建议", size=17)
    box(ax, 13.15, 4.00, 1.95, 1.24, "机器学习\n科学家建议", size=17)
    arrow(ax, (11.835, 4.0), (12.48, 3.51), GRAY)
    arrow(ax, (14.125, 4.0), (13.48, 3.51), GRAY)
    box(ax, 11.00, 2.71, 3.96, .8, "科学规划者汇总并最终选样", accent=True, size=17)
    arrow(ax, (12.98, 2.71), (12.98, 1.99), RED)
    for cx, desc in zip(centers, ["16 CW + 16 LLM", "32 LLM", "规划者选择32个"]):
        box(ax, cx-1.98, 1.01, 3.96, .98, "Batch 32\n"+desc, accent=True, size=20, weight="bold")
    footer(ax, "每批32个唯一合法ID；先冻结选择，再揭示标签。双专家建议独立，最终配额由规划者决定。", y=.32, size=12)
    save(fig, "F06_llm_selection_variants")


def f07():
    fig, ax = canvas("F07", "主动学习评价协议与标签预算",
                     "横轴只计训练标签；初始333个，每轮增加32个，固定验证集标签另计")
    timeline = fig.add_axes([.071, .457, .85, .09])
    grid = np.arange(333, 1006, 32)
    timeline.set(xlim=(325, 1013), ylim=(-1, 1), yticks=[])
    timeline.spines[["left", "top", "right"]].set_visible(False)
    timeline.spines["bottom"].set_position(("data", 0))
    ticks = [333, 365, 397, 429, 461, 493, 525, 813, 1005]
    timeline.set_xticks(ticks)
    timeline.set_xticks(grid, minor=True)
    timeline.tick_params(axis="x", which="major", labelsize=15, length=10, pad=11)
    timeline.tick_params(axis="x", which="minor", length=5, color=LINE)
    timeline.scatter(ticks, np.zeros(len(ticks)), s=45,
                     color=[RED if n in (333, 813, 1005) else GRAY for n in ticks], zorder=3)
    def xx(n):
        return (0.071 + .85*(n-325)/(1013-325))*16
    start, mid, end = xx(333), xx(813), xx(1005)
    ax.plot([start, mid], [6.54, 6.54], color=RED, lw=2.0)
    for x in (start, mid):
        ax.plot([x,x], [6.43,6.65], color=RED, lw=1.6)
    text(ax, (start+mid)/2, 6.98, "LLM混合实验的一类评价区间：333-813", 19, RED)
    ax.plot([start, end], [5.64, 5.64], color=GRAY, lw=1.8)
    for x in (start,end):
        ax.plot([x,x], [5.53,5.75], color=GRAY, lw=1.3)
    text(ax, (start+end)/2, 6.03, "传统策略完整对照的一类评价区间：333-1005", 18, GRAY)
    text(ax, 14.37, 3.27, "累积标签数", 14, GRAY)
    text(ax, 8, 2.82, "AULC只在相同标签区间内比较", 21, RED, weight="bold")
    text(ax, 8, 2.18, r"$\mathrm{AULC}_{a:b}=\frac{1}{b-a}\sum_i(n_{i+1}-n_i)\frac{e_i+e_{i+1}}{2}$", 23)
    cxs = [2.25, 6.0, 9.75, 13.5]
    for i, (cx,label) in enumerate(zip(cxs, ["选样", "揭示标签", "重新训练", "评价"])):
        text(ax, cx, 1.14, label, 21)
        if i < 3:
            arrow(ax, (cx+.80,1.14), (cxs[i+1]-.9,1.14))
    footer(ax, "预算点使用真实线性坐标；813与1005分别对应15轮与21轮新增实验。", y=.32)
    save(fig, "F07_evaluation_protocol")


def f08():
    fig, ax = canvas("F08", "同状态分支：策略收益是否依赖模型状态？",
                     "8个起始状态 = 2个seed × 2条历史轨迹 × 2个锚点（429 / 653）")
    box(ax, .70, 3.39, 2.64, 2.32, "同一模型检查点\n+ 已标记池 $L_t$\n+ 未标记池 $U_t$", size=18)
    for y, name in [(6.0, "LCMD"), (4.55, "IVR"), (3.10, "MaxDet")]:
        route(ax, [(3.34,4.55), (3.75,4.55), (3.75,y), (4.16,y)], LINE)
        box(ax, 4.16, y-.45, 2.50, .9, name, accent=name=="IVR", size=21)
        text(ax, 5.41, y-.79, "2轮 × 32 = +64标签", 12, GRAY)
    text(ax, 3.69, 1.91, "各起始状态内部比较\n相同的新增标签预算", 18)
    # Historical postfreeze test metrics, never mixed with validation figures R01-R04.
    path = OUT / "data/F08_branch_results.csv"
    if not path.is_file():
        path = ROOT / "reports/ppt_202610_al/frozen_inputs/studies/active_learning/qgeognn_v2_same_state_branching/results/branch_results.csv"
    d = pd.read_csv(path)
    check = (d.NRMSE_anchor+2*d.NRMSE_plus32+d.NRMSE_plus64)/4
    if len(d)!=24 or not np.allclose(check, d.short_AULC, atol=1e-12):
        raise ValueError("Invalid historical branching table")
    plot = fig.add_axes([.605, .19, .338, .54])
    states = list(d.groupby(["seed","source_trajectory","anchor_budget"], sort=True))
    styles = {"gradient_lcmd": (INK,"o","LCMD"), "kernel_ivr": (RED,"s","IVR"),
              "gradient_maxdet": (GRAY,"^","MaxDet")}
    wins = {s:0 for s in styles}
    for i, (key, g) in enumerate(states):
        lo, hi = g.short_AULC.min(), g.short_AULC.max()
        plot.plot([lo,hi],[i,i], color=LINE, lw=1.0)
        wins[g.loc[g.short_AULC.idxmin(),"strategy"]]+=1
        for s,(color,marker,label) in styles.items():
            value = float(g.loc[g.strategy==s,"short_AULC"].iloc[0])
            plot.scatter(value,i,marker=marker,color=color,s=58,zorder=3,label=label if i==0 else None)
    assert wins=={"gradient_lcmd":4,"kernel_ivr":4,"gradient_maxdet":0}
    labels = [f"{k[0]} / {'LCMD' if k[1]=='gradient_lcmd' else 'IVR'} / {k[2]}" for k,_ in states]
    plot.set(yticks=range(8), yticklabels=labels, ylim=(7.65,-.65), xlabel="同状态短程AULC（越低越好）")
    plot.tick_params(labelsize=13)
    plot.set_xlim(.44,.73)
    plot.set_xticks([.45,.55,.65,.70])
    plot.legend(loc="lower center", bbox_to_anchor=(.5,1.02), ncol=3, frameon=False, fontsize=14)
    text(ax, 11.4, 7.19, "真实历史日志：各状态内配对比较", 19, INK, weight="bold")
    text(ax, 11.55, .82, "胜出次数：LCMD 4 / IVR 4 / MaxDet 0", 18, RED, weight="bold")
    footer(ax, "右图为全部分支冻结后揭示的历史test指标，仅作机制探索；不与验证集曲线直接排名。", y=.30, size=12)
    save(fig, "F08_state_branching")


def f09():
    fig, ax = canvas("F09", "后续研究路线：稳健性、机制与真实验证",
                     "以下均为研究计划，尚未作为完整确认实验完成")
    centers = [3.03, 8, 12.98]
    stages = [("I  稳健性", "增加随机种子\n加强数值策略基线", "可靠的性能收益"),
              ("II  信息机制", "移除历史反馈\n移除化学上下文\n打乱反馈与条件的对应", "可识别的信息价值"),
              ("III  真实验证", "泛化能力\n真实实验闭环", "实际应用能力")]
    for i,(cx,(heading,body,outcome)) in enumerate(zip(centers,stages)):
        box(ax, cx-2.14, 3.28, 4.28, 3.02, dashed=True)
        text(ax, cx, 5.79, heading, 24, RED, weight="bold")
        ax.plot([cx-1.66,cx+1.66],[5.28,5.28],color=LINE,lw=.8)
        text(ax, cx, 4.28, body, 20)
        arrow(ax, (cx,3.28),(cx,2.42), RED, dashed=True)
        text(ax, cx, 1.90, outcome, 22, RED, weight="bold")
        if i<2:
            arrow(ax, (cx+2.17,4.84), (centers[i+1]-2.17,4.84), GRAY, dashed=True)
            arrow(ax, (cx+1.66,1.90), (centers[i+1]-1.76,1.90), GRAY, dashed=True)
    footer(ax, "虚线框与箭头表示计划；预期贡献须由后续实验检验。", y=.43)
    save(fig, "F09_research_roadmap")


METHODS = {"F01": f01, "F02": f02, "F03": f03, "F04": f04, "F05": f05,
           "F06": f06, "F07": f07, "F08": f08, "F09": f09}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", nargs="+", default=list(METHODS))
    args = parser.parse_args()
    for code in args.figures:
        METHODS[code]()
    if set(("F01", "F02", "F03")).issubset(args.figures):
        contact_sheet(["F01_research_gap", "F02_research_questions", "F03_active_learning_framework"],
                      FIGURES / "preview_F01_F03.png", cols=1, thumb=(1600, 900))
    write_layout_audit()
