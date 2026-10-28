"""Real recorded result plots. Connecting lines are guides, never fitted curves."""
import argparse
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from style import *

DATA = OUT / "data"


def learning(code, title, methods, end, note, switches=False, stage=False):
    d = pd.read_csv(DATA / f"{code}_learning_curves.csv")
    fig, ax = canvas(code, title, f"Row划分 | 验证集416条 | seed 157 / 6101 | 每轮32标签 | 333-{end}标签")
    handles = [Line2D([],[],color=c,linestyle=ls,marker=marker,markersize=5,linewidth=2.4,label=label)
               for _,label,c,marker,ls in methods]
    fig.legend(handles=handles,loc="center",bbox_to_anchor=(.525,.803),ncol=len(methods),frameon=False,
               fontsize=15,handlelength=2.7,columnspacing=1.5)
    starts = [.079, .386, .693]
    for i, (left,seed) in enumerate(zip(starts,[157,6101,"mean"])):
        plot = fig.add_axes([left,.228,.253,.45])
        text(ax,(left+.253/2)*16,6.68, f"seed {seed}" if seed!="mean" else "两seed算术均值（n=2）", 20, weight="bold")
        for method,label,color,marker,ls in methods:
            if seed=="mean":
                g=d[d.method==method].groupby("labels",sort=True).nrmse.agg(["mean","count"])
                if not (g["count"]==2).all():
                    raise ValueError("Means require two observed values at every budget")
                x,y=g.index.to_numpy(),g["mean"].to_numpy()
            else:
                g=d[(d.seed==seed)&(d.method==method)].sort_values("labels")
                x,y=g.labels.to_numpy(),g.nrmse.to_numpy()
            if not np.array_equal(x,np.arange(333,end+1,32)):
                raise ValueError(f"Incomplete grid: {code}/{seed}/{method}")
            plot.plot(x,y,color=color,marker=marker,ls=ls,ms=3.8,lw=2.9 if seed=="mean" else 2.2)
        plot.set(xlim=(325,end+9),ylim=(.275,1.085))
        plot.set_xticks([333,525,813,1005] if end==1005 else [333,429,525,653,813])
        plot.set_yticks([.3,.5,.7,.9,1.1])
        plot.set_ylim(.275,1.085)
        plot.tick_params(labelsize=13,length=4,pad=6)
        plot.set_xlabel("累积训练标签数",fontsize=17,labelpad=12)
        if i==0:
            plot.set_ylabel("验证集Combined NRMSE（越低越好）",fontsize=16,labelpad=12)
        plot.grid(axis="y",color=LINE,lw=.6,alpha=.65)
        if stage:
            plot.axvline(525,color=LINE,lw=1.1,ls=(0,(4,4)),zorder=0)
            plot.text(544,1.025,"探索延伸",fontsize=11,color=GRAY)
        if switches and seed!="mean":
            boundary=525 if seed==157 else 653
            plot.axvline(boundary,color=GRAY,lw=1.1,ls=(0,(4,4)),zorder=0)
            plot.text(boundary+8,.985,"后续选样切换LLM",fontsize=9.5,color=GRAY)
            text(ax,(left+.253/2)*16,6.26,f"{boundary}标签后：Astra/high → Sol6.1/xhigh",12,GRAY)
        elif switches:
            text(ax,(left+.253/2)*16,6.26,"包含不同预算的LLM切换，属于探索性轨迹",12,GRAY)
    footer(ax,"左、中：真实单seed曲线；右：同预算两seed算术均值。点为已记录指标，连接线仅便于阅读。",y=.95,size=13)
    footer(ax,note,y=.48,size=12)
    names={"R01":"R01_numerical_learning_curves","R03":"R03_cw32_vs_cw16_llm16","R04":"R04_free_llm_learning_curves"}
    save(fig,names[code])


def r01():
    learning("R01","数值主动学习策略：相同标签预算下的学习曲线",[
        ("Random32","随机采样",GRAY,"o",(0,(3,3))),
        ("LCMD32","Raw梯度-LCMD",INK,"^","-"),
        ("CW32 (LCMD)","CW梯度-LCMD",RED,"s","-"),
        ("Hybrid traditional","传统Hybrid",GRAY,"v","-.")],1005,
        "传统Hybrid使用K=3 ensemble，计算量更大；相同标签预算不等于相同计算预算。CW的525后阶段为探索性延伸。",stage=True)


def r02():
    d=pd.read_csv(DATA/"R02_aulc.csv",dtype={"seed":str})
    fig,ax=canvas("R02","Raw与CW输出梯度：收益取决于具体选样准则",
                  "验证集AULC | 相同区间333-1005 | 两seed：157 / 6101 | 越低越好")
    plot=fig.add_axes([.087,.235,.615,.48])
    x=np.arange(3)
    width=.30
    for i,selector in enumerate(["lcmd","ivr","maxdet"]):
        vals={}
        for rep,offset,fill,edge in [("raw",-.19,LINE,GRAY),("cw",.19,PALE,RED)]:
            method=f"{rep}_{selector}"
            g=d[d.method==method]
            mean=float(g.loc[g.seed=="mean","AULC"].iloc[0])
            plot.bar(i+offset,mean,width,color=fill,edgecolor=edge,lw=1.4,zorder=2)
            plot.text(i+offset,mean/2,f"{mean:.4f}",ha="center",va="center",fontsize=17,color=RED if rep=="cw" else INK)
            vals[rep]=[float(g.loc[g.seed==s,"AULC"].iloc[0]) for s in ["157","6101"]]
        for j,(marker,fillstyle) in enumerate([("o",WHITE),("s",INK)]):
            plot.plot([i-.19,i+.19],[vals["raw"][j],vals["cw"][j]],color=GRAY,lw=1.05,zorder=3)
            plot.scatter([i-.19,i+.19],[vals["raw"][j],vals["cw"][j]],marker=marker,
                         s=62,edgecolors=INK,facecolors=fillstyle,lw=1.2,zorder=4)
    plot.set(xticks=x,xticklabels=["LCMD","IVR","MaxDet"],ylim=(0,.70),xlim=(-.65,2.65))
    plot.set_ylabel("验证集AULC（333-1005）",fontsize=19,labelpad=15)
    plot.set_yticks(np.arange(0,.71,.1))
    plot.tick_params(labelsize=18,length=5,pad=9)
    plot.grid(axis="y",color=LINE,lw=.6,zorder=0)
    handles=[Patch(facecolor=LINE,edgecolor=GRAY,label="Raw输出梯度"),
             Patch(facecolor=PALE,edgecolor=RED,label="CW输出梯度"),
             Line2D([],[],marker="o",mfc=WHITE,mec=INK,color=INK,linestyle="none",label="seed 157"),
             Line2D([],[],marker="s",mfc=INK,mec=INK,color=INK,linestyle="none",label="seed 6101")]
    fig.legend(handles=handles,loc="center",bbox_to_anchor=(.494,.80),ncol=4,frameon=False,fontsize=16)
    text(ax,12.77,6.07,"CW相对Raw的均值变化",17,weight="bold")
    for i,(selector,cnote) in enumerate([("lcmd","两seed方向一致"),("ivr","两seed方向一致"),("maxdet","两seed方向不一致")]):
        mean=d[(d.seed=="mean")&(d.method==f"cw_{selector}")].iloc[0].AULC
        ref=d[(d.seed=="mean")&(d.method==f"raw_{selector}")].iloc[0].AULC
        pct=(mean/ref-1)*100
        text(ax,12.77,5.18-i*1.30,f"{['LCMD','IVR','MaxDet'][i]}   {pct:+.1f}%",23,RED,weight="bold")
        text(ax,12.77,4.66-i*1.30,cnote,15,GRAY)
    footer(ax,"柱高：两seed算术均值；圆点/方点：单seed原值；连线：同seed的Raw-CW配对。",y=1.02,size=13)
    footer(ax,"不同选样准则还采用不同RMS尺度；本图不能把跨准则差异完全归因于目标函数。525后为探索性延伸。",y=.47,size=12)
    save(fig,"R02_raw_cw_aulc_comparison")


def r03():
    learning("R03","CW32与CW16+LLM16：两seed学习曲线",[
        ("CW32 (LCMD)","CW32（数值基线）",GRAY,"s","-"),
        ("CW16 + LLM16","CW16 + LLM16",DARKRED,"o","-")],813,
        "LLM使用Astra/high；读取2026-10-11冻结快照的完整333-813轨迹。两seed开发性验证，不使用置信带。")


def r04():
    learning("R04","自由LLM32与相关对照：探索性学习曲线",[
        ("Random32","随机采样",GRAY,"o",(0,(3,3))),
        ("CW32 (LCMD)","CW32",GRAY,"s","-"),
        ("CW16 + LLM16","CW16 + LLM16",DARKRED,"o","-"),
        ("Free LLM32","自由LLM32",RED,"D","-")],813,
        "虚线标记Free LLM的模型切换预算；QGeoGNN训练协议相同。Astra前缀与Sol后续不能作为纯Sol实验解释。",switches=True)


RESULTS={"R01":r01,"R02":r02,"R03":r03,"R04":r04}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--figures",nargs="+",default=list(RESULTS))
    args=parser.parse_args()
    for code in args.figures:
        RESULTS[code]()
    contact_sheet(["R01_numerical_learning_curves","R02_raw_cw_aulc_comparison",
                   "R03_cw32_vs_cw16_llm16","R04_free_llm_learning_curves"],
                  FIGURES/"preview_R01_R04.png",cols=2)
    write_layout_audit()
