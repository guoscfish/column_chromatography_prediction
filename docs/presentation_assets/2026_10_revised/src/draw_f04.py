"""F04: implementation-derived acquisition mechanisms, not arbitrary point picks."""
import numpy as np
from style import *
from scientific_primitives import molecule,model,matrix,table,rule
from f04_mechanisms import examples


def plot_area(fig,x,y,w,h,limits,aspect=True):
    ax=fig.add_axes([x/16,y/9,w/16,h/9])
    ax.set(xlim=limits[0],ylim=limits[1],xticks=[],yticks=[])
    if aspect: ax.set_aspect('equal',adjustable='box')
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color(LINE)
    ax.spines[['left','bottom']].set_linewidth(.6)
    return ax


def cluster_view(fig,x,data,after=False):
    a=plot_area(fig,x,3.67,1.88,1.25,(( -1.14,1.50),(-.73,.91)))
    train=np.array(data['train']); pool=np.array(data['pool'])
    selected=data['selected']; current=selected[1] if after else selected[0]
    centers=np.vstack([train,pool[selected[0]]]) if after else train
    state=data['after_first'] if after else data['initial']
    for i,p in enumerate(pool):
        if after and i==selected[0]: continue
        c=centers[state['center'][i]]
        a.plot([c[0],p[0]],[c[1],p[1]],color=RED if i==current else LINE,
               lw=1.1 if i==current else .6,zorder=1)
        a.plot(*p,marker='o',ms=5.0,mfc=RED if i==current else WHITE,
               mec=RED if i==current else GRAY,mew=.8,linestyle='none',color=GRAY,zorder=3)
    a.plot(train[:,0],train[:,1],marker='s',ms=5.5,mfc=INK,mec=INK,
           linestyle='none',color=INK,zorder=4)
    if after:
        p=pool[selected[0]]
        a.plot(*p,marker='s',ms=6,mfc=WHITE,mec=RED,mew=1.15,
               color=RED,linestyle='none',zorder=4)
    p=pool[current]
    a.annotate('②' if after else '①',p,xytext=(4,5),textcoords='offset points',
               color=RED,fontsize=11,ha='left')
    return a


def representation_strip(ax):
    text(ax,.82,7.29,'样本表示',14,INK,ha='left',weight='bold')
    molecule(ax,1.13,6.68,.38,GRAY,oxygen=False)
    table(ax,1.59,6.44,.58,.47,rows=3,cols=2,header=False)
    text(ax,1.49,6.20,'分子 + 条件',10.5,GRAY)
    arrow(ax,(2.24,6.68),(2.56,6.68),GRAY,lw=.8)
    model(ax,3.17,6.68,w=1.19,h=.56,color=GRAY)
    text(ax,3.17,7.29,'QGeoGNN',14,INK,weight='bold')
    route(ax,[(3.84,6.68),(4.02,6.68),(4.02,7.12),(4.24,7.12)],GRAY,lw=.7)
    route(ax,[(3.84,6.68),(4.02,6.68),(4.02,6.54),(4.24,6.54)],GRAY,lw=.7)
    box(ax,4.24,6.94,1.90,.36,'隐层表示 128D',size=12.5,fill=WHITE)
    box(ax,4.24,6.36,1.90,.36,'V₁ / V₂ 的 q50',size=12.5,fill=WHITE)
    route(ax,[(6.14,6.54),(6.38,6.54),(6.38,7.12),(6.62,7.12)],GRAY,lw=.7)
    arrow(ax,(6.14,6.54),(6.62,6.54),GRAY,lw=.7)
    box(ax,6.62,6.94,2.14,.36,'Raw输出坐标',size=12.5,fill=WHITE)
    box(ax,6.62,6.36,2.14,.36,'CW输出坐标',size=12.5,fill=WHITE)
    for yy in (7.12,6.54):
        route(ax,[(8.76,yy),(9.06,yy),(9.06,6.82),(9.32,6.82)],GRAY,lw=.7)
    box(ax,9.32,6.35,1.66,.94,'对参数求梯度\n拼接两路',size=12.5,fill=WHITE)
    arrow(ax,(10.98,6.82),(11.64,6.82),GRAY,lw=.8)
    matrix(ax,11.73,6.45,1.00,.54,rows=3,cols=6,color=RED)
    text(ax,12.23,7.29,'CountSketch',12,INK)
    text(ax,12.23,6.20,'512D',12,RED)
    ax.plot([13.08,13.08],[6.27,7.43],color=LINE,lw=.65)
    text(ax,14.22,6.84,'本轮特征固定\n逐点更新选样状态\n实验批次 B=32',12.5,INK)
    text(ax,7.40,6.10,r'$C=(V_1+V_2)/2,\ W=V_2-V_1$；Raw/CW尺度冻结于 $L_0$',10.5,GRAY)


def draw():
    d=examples()
    fig,ax=canvas('F04','数值主动学习：特征构造与采样准则的实际计算',
                  '二维构造算例直接运行原始选样函数；展开批次内的逐点选择、状态更新与重新评分')
    representation_strip(ax)
    rule(ax,.80,15.21,5.96,LINE,.75)
    for xx in (5.48,10.52):
        ax.plot([xx,xx],[1.01,5.79],color=LINE,lw=.55)
    for x,letter,title,sub in [(.89,'a','LCMD','按总平方距离选簇，再取簇内最远点'),
                              (5.91,'b','MaxDet','在当前已标记集条件下，贪心增加logdet'),
                              (10.94,'c','IVR','选择使全参考集平均代理方差下降最多的点')]:
        text(ax,x,5.64,letter,13,RED,ha='left',weight='bold')
        text(ax,x+.30,5.64,title,20,INK,ha='left',weight='bold')
        text(ax,x,5.25,sub,12.3,INK,ha='left')

    # LCMD: the three-point cluster wins over the six-point cluster, exactly as code.
    lc=d['lcmd']
    text(ax,1.89,5.00,'第一次选择',11.5,GRAY)
    text(ax,4.12,5.00,'加入①后重新分配',11.5,GRAY)
    cluster_view(fig,.95,lc)
    cluster_view(fig,3.18,lc,after=True)
    arrow(ax,(2.86,4.30),(3.14,4.30),RED,lw=.8)
    c=lc['initial']['cluster_score']
    text(ax,1.89,3.42,f'6点簇：{c[0]:.3f}',11.5,GRAY)
    text(ax,1.89,3.13,f'3点簇：{c[1]:.3f} → 先选①',11.5,RED)
    text(ax,4.12,3.42,'①成为新中心，改变候选归属',10.7,GRAY)
    text(ax,4.12,3.13,'重算簇分数 → 再选②',11.5,RED)
    text(ax,3.01,2.79,'■ 当前中心    ○ 候选    红色：选中 / 新增',10.5,GRAY)
    text(ax,3.01,2.31,r'$d_u=\min_{c\in\mathcal{C}}\|\phi_u-\phi_c\|^2$',15)
    text(ax,3.01,1.87,r'$j^*=\arg\max_j\sum_{u:c(u)=j}d_u$',15)
    text(ax,3.01,1.43,'选中点加入中心集，更新最近距离与归属',12.3,INK)
    text(ax,3.01,1.07,r'初始 $\mathcal{C}=L_t$；不额外做RMS或逐行归一化',11,GRAY)

    # MaxDet: condition on current L and show the actual score changes after Q.
    md=d['maxdet']; p=np.array(md['conditioned_pool'])
    text(ax,6.71,5.00,'经当前L条件化的坐标',10.8,GRAY)
    av=plot_area(fig,5.88,3.66,1.75,1.25,((-.07,.70),(-.06,.92)))
    for i,v in enumerate(p):
        arrow(av,(0,0),tuple(v),RED if i==1 else GRAY,lw=1.2 if i==1 else .75)
        av.plot(*v,marker='o',ms=4,mfc=RED if i==1 else WHITE,mec=RED if i==1 else GRAY,
                color=GRAY,linestyle='none',zorder=3)
        label_offset={0:(4,1),1:(5,4),2:(5,-5),3:(4,2),4:(4,2)}[i]
        av.annotate(md['names'][i],v,xytext=label_offset,textcoords='offset points',
                    fontsize=10,color=RED if i==1 else INK)
    text(ax,9.00,5.00,'候选的边际logdet增益',11,GRAY)
    ab=plot_area(fig,8.14,3.83,1.93,1.07,((0,.55),(-.50,4.60)),aspect=False)
    ab.set_yticks(range(5),md['names'],fontsize=10)
    ab.set_xticks([0,.25,.50],['0','0.25','0.50'],fontsize=9.5)
    ab.tick_params(axis='both',length=2,pad=2)
    for i,(before,after) in enumerate(zip(md['gain_before'],md['gain_after_first'])):
        ab.barh(i-.13,before,height=.21,color=LINE,edgecolor=LINE,lw=.3)
        if i!=md['selected'][0]:
            ab.barh(i+.13,after,height=.21,color=RED,edgecolor=RED,lw=.3)
        else:
            text(ab,.12,i+.15,'已选',9.5,RED,ha='left')
    ab.invert_yaxis()
    text(ax,8.03,3.40,'首选Q → 重新评分 → 次选P',12,RED)
    text(ax,8.03,3.12,'初始次高为R；加入Q后，P成为剩余最优',10.8,GRAY)
    text(ax,8.03,2.79,'灰：首选前     红：加入Q后',10.5,GRAY)
    text(ax,8.03,2.31,r'$A=I+X_L^TX_L+X_S^TX_S$',14.5)
    text(ax,8.03,1.87,r'$\Delta_i=\log(1+x_i^TA^{-1}x_i)$',15)
    text(ax,8.03,1.43,'每选1个点，秩一更新并重算剩余增益',12.3,INK)
    text(ax,8.03,1.07,r'$x_i=\phi_i/s_L$；整体RMS尺度 $s_L$ 仅由当前 $L_t$ 计算',11,GRAY)

    # IVR: all reference rows count equally; high local variance alone is not IVR.
    iv=d['ivr']; train=np.array(iv['train']); p=np.array(iv['pool'])
    text(ax,11.72,5.00,'孤立方向P与密集方向Q',10.8,GRAY)
    ap=plot_area(fig,10.89,3.66,1.72,1.25,((-.20,2.18),(-.22,3.12)))
    ap.plot(train[:,0],train[:,1],marker='s',ms=4.5,mfc=INK,mec=INK,color=INK,linestyle='none')
    for i,v in enumerate(p):
        ap.plot(*v,marker='o',ms=4.5 if i==1 else 3.8,mfc=RED if i==1 else WHITE,
                mec=RED if i==1 else GRAY,color=GRAY,linestyle='none',zorder=3)
        if i in (0,1):
            ap.annotate(iv['names'][i],v,xytext=(3,2),textcoords='offset points',fontsize=10,
                        color=RED if i==1 else INK)
    before=np.array(iv['variance_before']);after=np.array(iv['variance_after_first'])
    text(ax,13.96,5.00,f'R均值：{before.mean():.3f} → {after.mean():.3f}',11,RED)
    ar=plot_area(fig,12.92,3.83,2.09,1.07,((-.55,8.55),(0,3.0)),aspect=False)
    ar.bar(np.arange(9)-.16,before,width=.26,color=LINE,edgecolor=LINE,lw=.3)
    ar.bar(np.arange(9)+.16,after,width=.26,color=RED,edgecolor=RED,lw=.3)
    ar.set_xticks(range(9),['L','L',*iv['names']],fontsize=9.5)
    ar.set_yticks([0,1,2,3],['0','1','2','3'],fontsize=9.5)
    ar.tick_params(axis='both',length=2,pad=2)
    text(ax,13.02,3.40,f'局部方差：P {before[2]:.3f} > Q {before[3]:.3f}',11.4,GRAY)
    text(ax,13.02,3.12,f'IVR得分：P {iv["scores_before"][0]:.3f} < Q {iv["scores_before"][1]:.3f} → 先选Q',11.4,RED)
    text(ax,13.02,2.79,'均匀参考；更新C后次选P｜灰：选前，红：选Q后',10.5,GRAY)
    text(ax,13.02,2.28,r'$\mathrm{IVR}(i)=\frac{x_i^TCMCx_i}{1+x_i^TCx_i}$',16)
    text(ax,13.02,1.82,r'$C=(I+X_L^TX_L+X_S^TX_S)^{-1}$',12.7)
    text(ax,13.02,1.43,r'$M=X_R^TX_R/N,\quad R=L_t\cup U_t$',13)
    text(ax,13.02,1.07,'整体RMS来自R；单位先验 / 噪声；逐点更新C',11,GRAY)

    rule(ax,.80,15.21,.78,LINE,.6)
    footer(ax,'算例为二维构造特征，非真实实验数据或降维结果；点位是显式构造输入，分数与选序由未修改的原始函数计算。',y=.51,size=10.5)
    footer(ax,'实验使用512D梯度特征并逐点选满32个；IVR是固定特征线性核的代理方差缩减，不是QGeoGNN的精确后验。',y=.24,size=10.5)
    # Include tick labels in the page-specific check; shared style audits text
    # blocks and containers, but chart ticks also need room above the captions.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    ticks=[t for a in fig.axes for t in a.get_xticklabels()+a.get_yticklabels()
           if t.get_visible() and t.get_text()]
    blocks=[t for a in fig.axes for t in a.texts if t.get_visible() and t.get_text()]
    collisions=[]
    for tick in ticks:
        bt=tick.get_window_extent(renderer)
        for block in blocks:
            bb=block.get_window_extent(renderer)
            w=min(bt.x1,bb.x1)-max(bt.x0,bb.x0)
            h=min(bt.y1,bb.y1)-max(bt.y0,bb.y0)
            if w>2 and h>2:
                collisions.append({'tick':tick.get_text(),'text':block.get_text()})
    (QA/'F04_plot_label_audit.json').write_text(json.dumps({
        'tick_labels_checked':len(ticks),'text_blocks_checked':len(blocks),
        'overlap_issues':collisions},ensure_ascii=False,indent=2)+'\n')
    if collisions:
        raise ValueError('F04 chart tick/caption overlap: '+str(collisions))
    save(fig,'F04_representation_acquisition')


if __name__=='__main__':
    draw()
    write_layout_audit()
