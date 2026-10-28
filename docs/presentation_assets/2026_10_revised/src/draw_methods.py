"""Scientific method diagrams, authored as flat 2D Matplotlib vectors."""
import argparse
import numpy as np
import pandas as pd
from style import *
from scientific_primitives import *


def f01():
    fig, ax = canvas("F01", "从分离条件优化到训练实验选择")
    text(ax,.80,7.60,"同一模型基础，两类决策目标",14,GRAY,ha="left")
    ax.plot([8,8],[1.02,7.37],color=LINE,lw=.6)
    panels=[(.98,GRAY,"已有研究","分离条件优化"),
            (8.75,RED,"本研究","训练数据采集优化")]
    for x,c,tag,title in panels:
        rule(ax,x,x+6.23,7.24,c,1.15)
        text(ax,x,6.92,tag,12.5,c,ha="left")
        text(ax,x+1.09,6.92,title,21,INK,ha="left",weight="bold")
    left=[("已训练的QGeoGNN","输入分子结构与候选条件"),
          ("预测 V₁ / V₂ 分布","用预测区间描述洗脱体积"),
          ("计算分离指标 $S_p$","评估候选条件下的分离表现"),
          ("选择更好的分离条件","面向当前混合物推荐实验条件")]
    right=[("当前模型 + 候选实验池","综合预测与候选覆盖信息"),
           ("估计样本的学习价值","评估哪些实验值得补充标签"),
           ("选择下一批训练实验","固定每轮32个唯一实验"),
           ("获取新标签并更新模型","扩充已标记集，进入下一轮")]
    ys=[5.88,4.52,3.16,1.80]
    for k,(x,c,_,_) in enumerate(panels):
        for i,(cy,(head,body)) in enumerate(zip(ys,left if k==0 else right)):
            step_number(ax,x+.10,cy,i+1,c)
            if i<3:
                arrow(ax,(x+.10,cy-.19),(x+.10,ys[i+1]+.19),LINE,lw=.75)
            text(ax,x+2.61,cy+.18,head,18.5,INK,ha="left",weight="bold")
            text(ax,x+2.61,cy-.27,body,13.5,GRAY,ha="left")
            if i<3: rule(ax,x+.47,x+6.23,cy-.64,LINE,.45)
    model(ax,2.40,5.88,w=1.52,h=.72,color=GRAY)
    intervals(ax,1.70,4.13,w=1.45,h=.78,color=GRAY)
    # Separation is represented geometrically; these are not measured traces.
    ax.plot([1.73,2.16],[3.16,3.16],color=GRAY,lw=2.0)
    ax.plot([2.69,3.24],[3.16,3.16],color=GRAY,lw=2.0)
    for xx in (1.73,2.16,2.69,3.24):
        ax.plot([xx,xx],[3.04,3.28],color=GRAY,lw=.75)
    arrow(ax,(2.18,3.16),(2.67,3.16),GRAY,lw=.85,style="<->")
    table(ax,1.72,1.32,1.43,.91,rows=3,cols=3,header=False)
    ax.add_patch(Rectangle((1.72,1.32),1.43,.91/3,fill=False,edgecolor=GRAY,lw=1.25,zorder=5))
    # Right path: candidate pool, feature-space coverage, actual Batch 32, update.
    model(ax,9.76,5.87,w=1.12,h=.6,color=RED)
    pool(ax,10.20,5.50,.65,.72,selected=(),color=RED)
    pool(ax,9.45,4.16,1.43,.71,selected=(2,7,12),labeled=(0,5),color=RED,axes=True)
    batch(ax,9.48,2.76,1.49,.83,red_count=32)
    table(ax,9.47,1.33,1.00,.89,rows=4,cols=3,highlight=(3,),header=False,color=RED)
    for x,c,goal in [(1.0,GRAY,"优化分离结果"),(8.76,RED,"提升模型数据效率")]:
        rule(ax,x,x+6.23,.99,c,.8)
        text(ax,x,.68,"目标",12,c,ha="left")
        text(ax,x+.69,.68,goal,20,c,ha="left",weight="bold")
    footer(ax,"图形均为方法示意：区间、点集与空白表格不代表实测结果；两条路径相互独立。",y=.23,size=10.5)
    save(fig,"F01_research_gap")


def f02():
    fig, ax = canvas("F02", "科学问题与研究路线")
    text(ax,8,7.53,"研究总目标",12.5,RED)
    text(ax,8,7.07,"面向柱色谱的数据高效学习",24,INK,weight="bold")
    ax.plot([8,8],[6.75,6.49],color=LINE,lw=.75)
    rule(ax,3.02,12.98,6.49,LINE,.75)
    cards=[(.97,"RQ1","数值选样","怎样衡量实验价值？"),
           (5.95,"RQ2","状态依赖策略","采样收益是否依赖模型状态？"),
           (10.93,"RQ3","反馈引导的LLM","化学知识与反馈\n能否提供额外信息？")]
    for x,rq,heading,q in cards:
        cx=x+2.05
        arrow(ax,(cx,6.49),(cx,6.21),GRAY,lw=.8)
        text(ax,x,5.96,rq,13,RED,ha="left",weight="bold")
        text(ax,x+.70,5.96,heading,20.5,INK,ha="left",weight="bold")
        text(ax,cx,5.35,q,15.5,INK)
        rule(ax,x,x+4.10,4.93,LINE,.55)
        rule(ax,x,x+4.10,2.74,LINE,.55)
    for x in (5.48,10.48):
        ax.plot([x,x],[2.08,6.17],color=LINE,lw=.5)
    matrix(ax,1.26,3.30,.85,1.03,rows=5,cols=4,color=RED)
    arrow(ax,(2.30,3.84),(2.81,3.84),GRAY,lw=.9)
    pool(ax,3.02,3.20,1.65,1.14,selected=(2,7,12),labeled=(0,5),axes=True)
    text(ax,1.71,2.99,"样本表示",12,GRAY)
    text(ax,3.85,2.99,"候选覆盖",12,GRAY)
    # All three policies leave exactly the same state; no outcome is invented.
    model(ax,6.67,3.84,w=1.02,h=.65,color=GRAY)
    ax.plot([7.25,7.79],[3.84,3.84],color=GRAY,lw=.8)
    ax.plot([7.79,7.79],[3.15,4.49],color=GRAY,lw=.8)
    for yy,label,c in [(4.49,"LCMD",RED),(3.82,"IVR",GRAY),(3.15,"MaxDet",GRAY)]:
        arrow(ax,(7.79,yy),(8.38,yy),c,lw=.9)
        ax.plot(8.52,yy,marker="o",ms=5.5,mfc=WHITE,mec=c,mew=1.0,color=c,linestyle="none")
        text(ax,8.80,yy,label,13.5,c,ha="left")
    text(ax,6.67,2.99,"同一模型状态",12,GRAY)
    # Molecular context and observed feedback enter a structured reasoning record.
    molecule(ax,11.62,4.20,.59,GRAY)
    table(ax,12.44,3.91,.92,.69,rows=3,cols=2,highlight=(2,),color=RED)
    text(ax,14.22,4.19,"模型\n状态",12,GRAY)
    for xx in (11.62,12.90,14.22):
        route(ax,[(xx,3.79),(xx,3.60),(12.90,3.60),(12.90,3.43)],LINE,lw=.7)
    ax.add_patch(Rectangle((11.84,2.98),2.13,.45,edgecolor=RED,facecolor=PALE,lw=.8))
    text(ax,12.905,3.205,"基于反馈的选样推理",12.3,RED)
    methods=[("表征 × 选样准则","LCMD / IVR / MaxDet"),
             ("同状态分支实验","固定起点，分别追加标签"),
             ("三种LLM机制","CW+LLM / 自由LLM / 双专家")]
    for (x,_,_,_),(head,body) in zip(cards,methods):
        text(ax,x+2.05,2.37,head,17,INK,weight="bold")
        text(ax,x+2.05,1.96,body,13.3,GRAY)
        ax.plot([x+2.05]*2,[1.67,1.39],color=LINE,lw=.7)
    rule(ax,3.02,12.98,1.39,LINE,.7)
    ax.add_patch(Rectangle((2.15,.66),11.70,.55,facecolor=PALE,edgecolor=RED,lw=.7))
    text(ax,8,.935,"统一评价    NRMSE   ·   AULC   ·   达到目标所需标签数",17,RED)
    footer(ax,"图内的点集、特征槽与表格为概念示意；三个问题并列，在相同标签预算下评价。",y=.24,size=10.5)
    save(fig,"F02_research_questions")


def f03():
    fig, ax = canvas("F03", "数值策略与LLM辅助的主动学习闭环",
                     "当前已实现：基于历史数据的离线主动学习模拟")
    # Candidate information is upstream of prediction and acquisition, never truth.
    ax.add_patch(Rectangle((4.91,6.74),5.32,.60,facecolor=WHITE,edgecolor=GRAY,lw=.8))
    text(ax,7.57,7.04,"未标记候选池 $U_t$   ·   分子结构与实验条件",14,INK)
    for xx in (6.33,8.72):
        arrow(ax,(xx,6.74),(xx,6.34),GRAY,lw=.8)
    box(ax,11.28,6.65,3.92,.75,"真实自动化柱色谱平台\n未来在线验证",size=13.5,dashed=True,color=GRAY)
    route(ax,[(11.40,6.65),(11.40,6.43),(12.45,6.43),
              (12.45,5.77),(11.96,5.77)],GRAY,dashed=True,lw=.8)
    cxs=[1.64,4.00,6.35,8.73,11.13,14.04]
    headings=["已标记数据集","QGeoGNN训练","预测与样本表示","选样与批次选择","实验反馈","更新数据集"]
    for i,(cx,head) in enumerate(zip(cxs,headings)):
        text(ax,cx,6.09,head,16,RED if i==3 else INK,weight="bold")
        text(ax,cx-.94,6.09,f"{i+1:02d}",10.5,RED if i==3 else GRAY,ha="right")
    table(ax,.85,4.80,1.57,1.08,rows=4,cols=3,labels=["结构","V₁","V₂"])
    model(ax,4.00,5.32,w=1.79,h=.85,color=GRAY)
    intervals(ax,5.49,5.23,w=1.76,h=.59,color=GRAY)
    matrix(ax,5.74,4.81,1.28,.26,rows=2,cols=7,color=GRAY)
    batch(ax,7.88,4.86,1.79,.80,red_count=32)
    text(ax,8.73,5.82,"$B_t$ · 32个实验",10.5,RED)
    table(ax,10.40,4.80,1.50,1.08,rows=4,cols=3,highlight=(2,3),labels=["ID","V₁","V₂"],color=RED)
    table(ax,13.22,4.80,1.65,1.08,rows=5,cols=3,highlight=(3,4),labels=["结构","V₁","V₂"],color=RED)
    for a,b,c in [(2.42,3.07,GRAY),(4.92,5.45,GRAY),(7.28,7.82,RED),
                  (9.65,10.35,RED),(11.93,13.17,GRAY)]:
        arrow(ax,(a,5.32),(b,5.32),c,lw=1.0)
    text(ax,1.64,4.42,"$L_t$：结构、条件与实测标签",12.2,GRAY)
    text(ax,4.00,4.42,"用当前已标记集训练",12.2,GRAY)
    text(ax,6.35,4.42,"V₁/V₂预测 + 表征",12.2,GRAY)
    text(ax,8.73,5.98,"",12)
    text(ax,11.13,4.34,"从历史记录揭示\n所选实验的真实标签",12.2,GRAY)
    text(ax,14.04,4.42,"$L_{t+1}=L_t\\cup B_t$",13.5,RED)
    # Expanded acquisition mechanisms: one selection step, different configurations.
    rule(ax,4.91,15.20,3.88,LINE,.6)
    text(ax,4.91,3.61,"选样机制",12,GRAY,ha="left")
    text(ax,14.99,3.61,"冻结32个唯一实验后，再揭示标签",11.5,RED,ha="right")
    route(ax,[(7.00,3.20),(7.00,3.37),(8.73,3.37),(8.73,4.78)],GRAY,lw=.9)
    route(ax,[(12.19,3.20),(12.19,3.37),(8.73,3.37),(8.73,4.78)],RED,lw=.9)
    # White header masks ensure that the vertical connector has its own clear lane.
    ax.add_patch(Rectangle((4.91,1.87),4.15,1.33,facecolor=WHITE,edgecolor=LINE,lw=.8,zorder=2))
    text(ax,5.18,2.94,"A  数值选样",16.5,INK,ha="left",weight="bold")
    matrix(ax,5.21,2.07,.74,.60,rows=4,cols=5,color=GRAY)
    arrow(ax,(6.09,2.39),(6.47,2.39),GRAY,lw=.8)
    text(ax,7.60,2.39,"梯度表示\nLCMD / IVR / MaxDet",13.3,INK)
    ax.add_patch(Rectangle((9.35,1.87),5.85,1.33,facecolor=WHITE,edgecolor=RED,lw=.8,zorder=2))
    text(ax,9.62,2.94,"B  LLM辅助选样",16.5,RED,ha="left",weight="bold")
    molecule(ax,9.98,2.36,.48,GRAY,oxygen=False)
    table(ax,10.51,2.05,.80,.58,rows=3,cols=2,highlight=(2,),color=RED)
    text(ax,13.05,2.39,"化学信息 / 已观察反馈\n模型状态 / 科学推理",13.3,INK)
    text(ax,10.10,1.58,"数值与LLM的组合方式及配额随方案而异",11.5,GRAY)
    # Return goes through whitespace, entering the training stage from its left.
    route(ax,[(14.93,4.85),(15.52,4.85),(15.52,1.12),(2.86,1.12),
              (2.86,5.00),(3.08,5.00)],RED,lw=1.05)
    text(ax,8.03,.80,"更新标签 → 重新训练 → 下一轮选样",13,RED)
    ax.plot([.85,1.23],[3.24,3.24],color=GRAY,lw=.9)
    text(ax,1.39,3.24,"已实现：离线模拟",11.5,GRAY,ha="left")
    ax.plot([.85,1.23],[2.76,2.76],color=GRAY,lw=.9,ls=(0,(4,3)))
    text(ax,1.39,2.76,"计划：在线验证",11.5,GRAY,ha="left")
    footer(ax,"图内区间、特征槽和空白表格均为概念示意；候选与待测批次不含真实标签。",y=.26,size=10.5)
    save(fig,"F03_active_learning_framework")


def small_space(fig, bounds, points, selected=(), labeled=(), vectors=False):
    ax = fig.add_axes(bounds)
    ax.scatter(points[:, 0], points[:, 1], s=25, facecolors=WHITE, edgecolors=GRAY, linewidths=.7)
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
    # Detailed implementation-derived panels live in a dedicated editable module.
    from draw_f04 import draw
    draw()


def f05():
    fig, ax = canvas("F05", "化学知识与实验反馈驱动的LLM选样",
                     "信息边界：真实 V₁/V₂ 仅来自已标记实验；未标记候选只能提供特征与预测")
    text(ax,.93,7.05,"输入证据",13,GRAY,ha="left")
    text(ax,7.93,7.05,"基于证据形成选样假设",13,GRAY)
    text(ax,13.17,7.05,"输出实验建议",13,GRAY)
    rows=[(6.12,"A  化学信息","分子结构 / 官能团\n实验条件与洗脱体系",GRAY),
          (4.35,"B  历史实验反馈","已观察的实测 V₁/V₂\n预测残差与对应实验条件",RED),
          (2.58,"C  当前模型状态","候选预测 / 不确定性代理\n已标记与候选样本覆盖",GRAY)]
    for cy,head,body,c in rows:
        text(ax,2.75,cy+.24,head,18,INK,ha="left",weight="bold")
        text(ax,2.75,cy-.34,body,13.5,GRAY,ha="left")
        rule(ax,.93,5.75,cy-.82,LINE,.5)
        ax.plot([5.77,6.10],[cy,cy],color=c,lw=.9)
    molecule(ax,1.62,6.03,.77,GRAY)
    table(ax,1.01,3.87,1.22,.99,rows=3,cols=2,highlight=(2,),labels=["实测","预测"],color=RED)
    pool(ax,1.03,2.20,1.20,.77,selected=(2,7,12),labeled=(0,5),axes=True)
    ax.plot([6.10,6.10],[2.58,6.12],color=LINE,lw=.85)
    arrow(ax,(6.10,4.35),(6.66,4.35),RED,lw=.95)
    # A structured reasoning record, avoiding anthropomorphic robot/chat artwork.
    ax.add_patch(Rectangle((6.66,3.16),3.01,2.52,facecolor=WHITE,edgecolor=RED,lw=.9))
    ax.add_patch(Rectangle((6.66,3.16),.055,2.52,facecolor=RED,edgecolor=RED,lw=.2))
    text(ax,8.17,6.15,"LLM辅助实验推理",19,RED,weight="bold")
    text(ax,8.17,5.35,"已观察证据",12.5,GRAY)
    rule(ax,7.00,9.30,5.04,LINE,.6)
    for i,(cy,label) in enumerate([(4.69,"归纳实验反馈"),(4.16,"提出可检验假设"),(3.63,"给出选样理由")]):
        step_number(ax,7.10,cy,i+1,RED)
        text(ax,7.43,cy,label,14,INK,ha="left")
    text(ax,8.17,2.70,"假设随新反馈修订",13.5,RED)
    route(ax,[(9.67,4.35),(10.44,4.35),(10.44,5.10),(11.57,5.10)],GRAY,lw=.95)
    text(ax,13.19,6.15,"选择建议与理由",19,INK,weight="bold")
    table(ax,11.57,4.55,3.23,1.10,rows=3,cols=2,labels=["唯一候选ID","选样依据"],color=RED)
    arrow(ax,(13.19,4.55),(13.19,3.69),RED,lw=.95)
    text(ax,13.48,4.14,"冻结批次",12.5,RED,ha="left")
    batch(ax,11.68,2.64,3.18,1.00)
    text(ax,13.19,2.16,"下一批32个实验",18,RED,weight="bold")
    rule(ax,.93,15.05,1.49,LINE,.6)
    for i,(cx,label) in enumerate(zip([2.25,6.0,9.75,13.5],["观察","假设","建议实验","新反馈"])):
        text(ax,cx,.96,label,19,RED if i==1 else INK)
        if i<3: arrow(ax,(cx+.75,.96),(cx+2.85,.96),GRAY,lw=.9)
    footer(ax,"分子、点集与空白表格为概念示意；“假设”须经后续实验检验，不表示已发现化学规律。",y=.27,size=11)
    save(fig,"F05_llm_feedback_reasoning")


def f06():
    fig, ax = canvas("F06", "三种LLM选样机制：相同的32标签预算")
    box(ax,2.68,6.98,10.64,.57,"共同输入：当前模型状态 + 候选实验池 + 已观察反馈",size=16,fill=WHITE)
    rule(ax,3.03,12.98,6.59,LINE,.7)
    ax.plot([8,8],[6.98,6.59],color=LINE,lw=.7)
    centers=[3.03,8,12.98]
    for cx in centers: arrow(ax,(cx,6.59),(cx,6.30),GRAY,lw=.8)
    for x in (5.48,10.52): ax.plot([x,x],[1.0,6.29],color=LINE,lw=.55)
    for cx,title in zip(centers,["A  CW16 + LLM16","B  自由LLM32","C  双专家LLM"]):
        text(ax,cx,6.03,title,19,RED,weight="bold")
    text(ax,3.03,5.38,"CW-LCMD先选16个",17,INK,weight="bold")
    batch(ax,1.66,4.61,2.84,.55,rows=2,cols=8,red_count=0)
    text(ax,3.03,4.19,"先从LLM可选池排除CW16",12.5,GRAY)
    arrow(ax,(3.03,3.96),(3.03,3.64),GRAY,lw=.9)
    box(ax,1.19,2.86,3.68,.77,"LLM补充另外16个",accent=True,size=17,fill=WHITE)
    text(ax,3.03,2.54,"待测CW16仅提供预测，无真实标签",11.5,GRAY)
    arrow(ax,(3.03,2.34),(3.03,2.10),RED,lw=.9)
    pool(ax,6.66,4.45,2.69,.84,selected=(2,7,12),axes=True)
    text(ax,8,5.58,"完整候选池",13,GRAY)
    arrow(ax,(8,4.28),(8,3.65),RED,lw=.9)
    box(ax,6.16,2.86,3.68,.77,"LLM自主选择32个",accent=True,size=17,fill=WHITE)
    text(ax,8,2.54,"无CW预留配额",12.5,GRAY)
    arrow(ax,(8,2.34),(8,2.10),RED,lw=.9)
    text(ax,11.82,5.51,"化学科学家",15,INK,weight="bold")
    text(ax,14.12,5.51,"机器学习科学家",15,INK,weight="bold")
    molecule(ax,11.82,4.72,.63,GRAY)
    matrix(ax,13.57,4.38,1.15,.64,rows=4,cols=5,color=GRAY)
    text(ax,11.82,4.00,"独立建议",12.5,GRAY)
    text(ax,14.12,4.00,"独立建议",12.5,GRAY)
    route(ax,[(11.82,3.77),(11.82,3.73),(12.12,3.73),(12.12,3.64)],GRAY,lw=.8)
    route(ax,[(14.12,3.77),(14.12,3.73),(13.84,3.73),(13.84,3.64)],GRAY,lw=.8)
    box(ax,11.14,2.86,3.68,.77,"科学规划者汇总并最终选样",accent=True,size=15.3,fill=WHITE)
    text(ax,12.98,2.54,"不预设两位专家的最终入选配额",11.5,GRAY)
    arrow(ax,(12.98,2.34),(12.98,2.10),RED,lw=.9)
    for i,cx in enumerate(centers):
        batch(ax,cx-1.37,1.18,2.84,.85,red_count=16 if i==0 else 32,red_start=16 if i==0 else 0)
        text(ax,cx,.83,["灰：CW16   红：LLM16","LLM选择32个","规划者确定32个唯一实验"][i],13.5,RED)
    footer(ax,"每批32个唯一合法ID；先冻结选择再揭示标签。双专家信息独立，代码按顺序调用。点集为示意。",y=.27,size=11)
    save(fig,"F06_llm_selection_variants")


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
    ax.plot([start, mid], [6.54, 6.54], color=RED, lw=1.2)
    for x in (start, mid):
        ax.plot([x,x], [6.43,6.65], color=RED, lw=1.1)
    text(ax, (start+mid)/2, 6.98, "LLM混合实验的一类评价区间：333-813", 17.5, RED)
    ax.plot([start, end], [5.64, 5.64], color=GRAY, lw=1.1)
    for x in (start,end):
        ax.plot([x,x], [5.53,5.75], color=GRAY, lw=1.0)
    text(ax, (start+end)/2, 6.03, "传统策略完整对照的一类评价区间：333-1005", 17, GRAY)
    text(ax, 14.37, 3.27, "累积标签数", 14, GRAY)
    text(ax, 8, 2.82, "AULC只在相同标签区间内比较", 19, RED, weight="bold")
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
    text(ax,2.0,6.40,"同一模型检查点",17,INK,weight="bold")
    model(ax,2.0,5.73,w=1.70,h=.71,color=GRAY)
    table(ax,.89,4.18,.91,.84,rows=4,cols=3,header=False)
    pool(ax,2.08,4.19,.95,.80,labeled=(),selected=(),axes=False)
    text(ax,1.34,3.80,"已标记池 $L_t$",12,GRAY)
    text(ax,2.56,3.80,"候选池 $U_t$",12,GRAY)
    rule(ax,.87,3.14,3.43,LINE,.6)
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
    value_min,value_max=float(d.short_AULC.min()),float(d.short_AULC.max())
    pad=(value_max-value_min)*.06
    plot.set_xlim(value_min-pad,value_max+pad)
    plot.set_xticks([.50,.60,.70,.80])
    limits=plot.get_xlim()
    visible=int(d.short_AULC.between(*limits).sum())
    assert visible==24,"Every historical branch must be visible on the axis"
    (QA/"branch_plot_range.json").write_text(json.dumps({
        "source":"data/F08_branch_results.csv","states":len(states),
        "branches":len(d),"visible_points":visible,"xlim":list(limits),
        "value_min":value_min,"value_max":value_max
    },indent=2)+"\n",encoding="utf-8")
    plot.legend(loc="lower center", bbox_to_anchor=(.5,1.02), ncol=3, frameon=False, fontsize=14)
    plot.axhline(3.5,color=LINE,lw=.5,zorder=0)
    text(ax, 11.4, 7.19, "真实历史日志：各状态内配对比较", 17.5, INK, weight="bold")
    text(ax, 11.55, .82, "胜出次数：LCMD 4 / IVR 4 / MaxDet 0", 16.5, RED, weight="bold")
    footer(ax, "右图为全部分支冻结后揭示的历史test指标，仅作机制探索；不与验证集曲线直接排名。", y=.30, size=12)
    save(fig, "F08_state_branching")


def f09():
    fig, ax = canvas("F09", "后续研究路线：稳健性、机制与真实验证",
                     "以下均为研究计划，尚未作为完整确认实验完成")
    centers=[3.03,8.00,12.98]
    for cx,title in zip(centers,["I  稳健性","II  信息机制","III  泛化与真实验证"]):
        box(ax,cx-2.14,2.67,4.28,4.34,dashed=True,fill=WHITE).set_zorder(0)
        text(ax,cx,6.55,title,19.5,RED,weight="bold")
        rule(ax,cx-1.75,cx+1.75,6.11,LINE,.6)
    # Planned replicates: repeated initializations, no performance trajectories.
    for cy in (5.61,5.00,4.39):
        graph(ax,1.69,cy,.38,GRAY)
        arrow(ax,(2.09,cy),(2.52,cy),GRAY,dashed=True,lw=.75)
        batch(ax,2.74,cy-.21,1.48,.44,rows=4,cols=8,red_count=32)
    text(ax,3.03,3.69,"增加随机种子\n加强数值策略基线",17,INK)
    # A planned ablation design table; checkmarks mean retained inputs, not success.
    colxs=[8.00,8.73,9.45]
    for xx,label in zip(colxs,["化学","反馈","对应"]): text(ax,xx,5.66,label,11.5,GRAY)
    ablations=[("完整信息",[1,1,1]),("移除反馈",[1,0,0]),("移除化学",[0,1,1]),("打乱对应",[1,1,0])]
    for i,(label,keep) in enumerate(ablations):
        yy=5.16-i*.48
        text(ax,6.28,yy,label,12,INK,ha="left")
        for xx,yes in zip(colxs,keep):
            if yes:
                ax.plot([xx-.075,xx-.012,xx+.105],[yy-.005,yy-.070,yy+.075],color=GRAY,lw=1.1)
            else:
                ax.plot([xx-.07,xx+.07],[yy-.07,yy+.07],color=RED,lw=1.0)
                ax.plot([xx-.07,xx+.07],[yy+.07,yy-.07],color=RED,lw=1.0)
    text(ax,8,3.12,"检验各类信息的增量贡献",15.5,INK)
    # Abstract transfer and laboratory interface, without invented equipment.
    molecule(ax,11.72,5.22,.62,GRAY)
    arrow(ax,(12.32,5.22),(12.77,5.22),GRAY,dashed=True,lw=.85)
    table(ax,12.97,4.84,1.50,.83,rows=3,cols=3,header=False,color=RED)
    text(ax,11.73,4.45,"新分子 / 新条件",11.5,GRAY)
    text(ax,13.73,4.45,"固定评价协议",11.5,GRAY)
    route(ax,[(13.72,4.16),(13.72,3.89),(12.98,3.89),(12.98,3.69)],GRAY,dashed=True,lw=.8)
    box(ax,11.40,2.99,3.16,.70,"真实平台上的在线闭环",dashed=True,size=15.5,fill=WHITE)
    outcomes=["可靠的性能收益","可识别的信息价值","实际应用能力"]
    for i,(cx,label) in enumerate(zip(centers,outcomes)):
        arrow(ax,(cx,2.67),(cx,2.12),RED,dashed=True,lw=.9)
        text(ax,cx,1.73,label,19,RED,weight="bold")
        if i<2: arrow(ax,(cx+2.18,4.82),(centers[i+1]-2.18,4.82),GRAY,dashed=True,lw=.8)
    text(ax,8,.91,"稳健复现 → 机制解释 → 外部验证",16.5,GRAY)
    footer(ax,"虚线表示计划；中间表格表示拟保留或移除的信息，不是消融结果；预期贡献须由实验检验。",y=.28,size=11)
    save(fig,"F09_research_roadmap")


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
