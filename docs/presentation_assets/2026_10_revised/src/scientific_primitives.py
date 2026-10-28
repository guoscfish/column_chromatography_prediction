"""Small, semantic 2D drawings for method figures.

All marks are editable Matplotlib vectors. These drawings encode objects and
operations, never experimental measurements, performance, or feature values.
"""
import numpy as np
from matplotlib.patches import Circle, Polygon
from style import *


def rule(ax, x0, x1, y, color=LINE, lw=.6):
    ax.plot([x0, x1], [y, y], color=color, lw=lw)


def molecule(ax, cx, cy, scale=1., color=INK, oxygen=True):
    """Skeletal phenol illustration; no claim that it is an evaluated sample."""
    angles=np.arange(6)*np.pi/3+np.pi/6
    p=np.column_stack([np.cos(angles),np.sin(angles)])*.46*scale+[cx,cy]
    ax.add_patch(Polygon(p,closed=True,fill=False,edgecolor=color,lw=1.1,zorder=3))
    center=np.array([cx,cy])
    for i in (0,2,4):
        a,b=p[i],p[(i+1)%6]
        a=.79*a+.21*center; b=.79*b+.21*center
        ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=.8)
    if oxygen:
        a=p[1]; b=a+np.array([0,.30*scale])
        ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=1.1)
        text(ax,b[0],b[1]+.14*scale,"OH",max(8,12*scale),color)


def graph(ax, cx, cy, scale=1., color=GRAY):
    p=np.array([[-.63,-.12],[-.30,.38],[.25,.29],[.62,-.1],[.12,-.43],[-.31,-.37]])*scale+[cx,cy]
    for i,j in [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(1,5),(2,4)]:
        ax.plot(p[[i,j],0],p[[i,j],1],color=LINE,lw=.8,zorder=2)
    for i,(x,y) in enumerate(p):
        ax.add_patch(Circle((x,y),.075*scale,facecolor=color if i in (1,4) else WHITE,
                            edgecolor=color,lw=.9,zorder=3))


def model(ax, cx, cy, w=1.7, h=.72, color=GRAY):
    """Abstract encoder/readout; two groups of three output quantiles.

    Six endpoints correspond to V1/V2 q10/q50/q90. Internal nodes and feature
    slots do not specify the actual network's layer count or hidden dimension.
    """
    graph(ax,cx-w*.31,cy,scale=w*.34,color=color)
    arrow(ax,(cx-w*.045,cy),(cx+w*.11,cy),color,lw=.8).set_zorder(4)
    for i in range(4):
        ax.add_patch(Rectangle((cx+w*.13,cy-h*.38+i*h*.22),w*.13,h*.13,
                               edgecolor=color,facecolor=WHITE,lw=.75,zorder=3))
    for xx in (.40,.51):
        for yy in (-.24,0,.24):
            ax.plot([cx+w*.28,cx+w*xx],[cy,cy+yy*h],color=LINE,lw=.65)
            ax.add_patch(Circle((cx+w*xx,cy+yy*h),.027*w,facecolor=color,
                                edgecolor=color,lw=.5,zorder=3))


def table(ax, x, y, w, h, rows=4, cols=3, highlight=(), labels=None,
          color=GRAY, header=True):
    """Data schema with intentionally empty data cells, not fabricated values."""
    rh=h/(rows+int(header)); cw=w/cols
    ax.add_patch(Rectangle((x,y),w,h,facecolor=WHITE,edgecolor=LINE,lw=.75,zorder=2))
    if header:
        ax.add_patch(Rectangle((x,y+h-rh),w,rh,facecolor=PALE if color==RED else LINE,
                               edgecolor=LINE,lw=.5,zorder=3))
    for i in highlight:
        ax.add_patch(Rectangle((x,y+h-rh*(i+1+int(header))),w,rh,
                               facecolor=PALE,edgecolor=RED,lw=.65,zorder=3))
    for i in range(1,rows+int(header)):
        rule(ax,x,x+w,y+i*rh,LINE,.5)
    for j in range(1,cols):
        ax.plot([x+j*cw,x+j*cw],[y,y+h],color=LINE,lw=.5,zorder=4)
    if labels:
        for j,label in enumerate(labels):
            text(ax,x+(j+.5)*cw,y+h-rh*.5,label,10.5,color,zorder=5)
    else:
        for i in range(rows):
            yy=y+h-rh*(i+.5+int(header))
            for j in range(cols):
                ax.plot([x+(j+.24)*cw,x+(j+.72)*cw],[yy,yy],
                        color=RED if i in highlight else LINE,lw=.8,zorder=4)


def matrix(ax, x, y, w, h, rows=5, cols=5, color=GRAY):
    """Empty feature-vector slots; fills distinguish columns, not magnitudes."""
    for i in range(rows):
        for j in range(cols):
            ax.add_patch(Rectangle((x+j*w/cols,y+i*h/rows),w/cols*.72,h/rows*.68,
                                   facecolor=PALE if color==RED else WHITE,
                                   edgecolor=color if j==cols-1 else LINE,lw=.55,zorder=3))
    for xx in (x-.08,x+w+.025):
        ax.plot([xx,xx],[y-.05,y+h-.015],color=color,lw=.8)


POINTS=np.array([[.08,.12],[.23,.10],[.13,.35],[.34,.32],[.30,.58],[.11,.70],
                 [.40,.84],[.49,.52],[.63,.32],[.76,.15],[.88,.38],[.74,.62],
                 [.91,.78],[.65,.86],[.52,.14],[.90,.07],[.07,.94],[.45,.98]])


def pool(ax, x, y, w, h, selected=(), labeled=(), color=RED, axes=False):
    """Fixed conceptual coordinates, explicitly identified as such in figures."""
    p=POINTS*[w,h]+[x,y]
    if axes:
        ax.plot([x-.10,x-.10,x+w+.06],[y+h+.03,y-.08,y-.08],color=LINE,lw=.65)
    for i,(xx,yy) in enumerate(p):
        ax.plot(xx,yy,marker="s" if i in labeled else "o",ms=4.7 if i in selected else 3.9,
                mfc=color if i in selected else (INK if i in labeled else WHITE),
                mec=color if i in selected else GRAY,mew=.75,linestyle="none",color=GRAY,zorder=4)


def batch(ax, x, y, w, h, rows=4, cols=8, red_count=32, color=RED, red_start=0):
    """Each square denotes one experiment; a 4x8 grid is exactly Batch 32."""
    size=min(w/cols,h/rows)*.65
    for i in range(rows):
        for j in range(cols):
            c=color if red_start<=i*cols+j<red_start+red_count else GRAY
            ax.add_patch(Rectangle((x+j*w/cols,y+(rows-i-1)*h/rows),size,size,
                                   facecolor=PALE if c==RED else WHITE,
                                   edgecolor=c,lw=.7,zorder=3))


def intervals(ax, x, y, w=1.7, h=.8, color=GRAY, labels=True):
    """Two quantile-style intervals without an invented probability density."""
    for i,(a,b,m) in enumerate(((.05,.61,.28),(.39,.98,.71))):
        yy=y+h*(.78 if i==0 else .20)
        xx=x+(.25 if labels else 0)
        ww=w-(.25 if labels else 0)
        ax.plot([xx+a*ww,xx+b*ww],[yy,yy],color=color,lw=1.4,zorder=3)
        for q in (a,b):
            ax.plot([xx+q*ww]*2,[yy-.055,yy+.055],color=color,lw=.8,zorder=3)
        ax.plot(xx+m*ww,yy,marker="o",ms=4.5,mfc=WHITE,mec=color,mew=1.0,
                color=color,linestyle="none",zorder=4)
        if labels:
            text(ax,x+.10,yy,"V₁" if i==0 else "V₂",11,color)


def evidence(ax, x, y, w=2.4, h=1.0):
    """Observed evidence and a structured reasoning record, without fake prose."""
    molecule(ax,x+w*.18,y+h*.43,.55,GRAY)
    table(ax,x+w*.40,y+h*.12,w*.53,h*.76,rows=3,cols=2,highlight=(2,),color=RED)
    arrow(ax,(x+w*.28,y+h*.43),(x+w*.39,y+h*.43),GRAY,lw=.8)


def step_number(ax, x, y, n, color=GRAY):
    ax.add_patch(Circle((x,y),.14,facecolor=WHITE,edgecolor=color,lw=.8,zorder=5))
    text(ax,x,y,str(n),10,color,zorder=6)
