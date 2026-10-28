"""Shared Matplotlib style and editable-vector export; no training imports."""
from pathlib import Path
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.text import Text
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1]
FIGURES = OUT / "figures"
QA = OUT / "qa"
RED = "#9A0001"
DARKRED = "#73161D"
SOFTRED = "#D9A9AD"
PALE = "#F4E9EA"
INK = "#343434"
GRAY = "#777777"
LINE = "#C8C8C8"
WHITE = "#FFFFFF"
PALETTE = [RED, DARKRED, SOFTRED, PALE, INK, GRAY, LINE, WHITE]
AUDIT = []


def setup():
    # Use the installed Office fonts without copying or redistributing font files.
    dirs = [Path(os.environ.get("MSYH_FONT_DIR", "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts")),
            Path("/Applications/Microsoft Word.app/Contents/Resources/DFonts")]
    for directory in dirs:
        for name in ("msyh.ttc", "msyhbd.ttc"):
            path = directory / name
            if path.is_file():
                font_manager.fontManager.addfont(str(path))
    cjk = next((n for n in ("Microsoft YaHei", "Microsoft YaHei UI")
                if any(f.name == n for f in font_manager.fontManager.ttflist)), None)
    if cjk is None:
        raise RuntimeError("Microsoft YaHei is required. Set MSYH_FONT_DIR to the directory containing msyh.ttc/msyhbd.ttc.")
    plt.rcParams.update({
        "font.family": [cjk, "DejaVu Sans"], "font.sans-serif": [cjk, "DejaVu Sans"],
        "font.size": 16, "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": GRAY, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .8, "lines.linewidth": 2.0,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans", "figure.facecolor": WHITE,
        "axes.facecolor": WHITE, "savefig.facecolor": WHITE,
        "savefig.dpi": 300, "axes.unicode_minus": False,
    })
    FIGURES.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    return cjk


def canvas(code, title, subtitle="", size=(16, 9)):
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 16), ylim=(0, 9))
    ax.axis("off")
    text(ax, .72, 8.43, code, 14, RED, weight="bold", ha="left")
    text(ax, 1.48, 8.43, title, 24, INK, weight="bold", ha="left")
    ax.plot([.72, 15.28], [8.02, 8.02], color=LINE, lw=.6)
    if subtitle:
        text(ax, .75, 7.68, subtitle, 12.5, GRAY, ha="left")
    return fig, ax


def text(ax, x, y, value, size=18, color=INK, ha="center", va="center", weight="normal", **kw):
    return ax.text(x, y, value, fontsize=size, color=color, ha=ha, va=va,
                   fontweight=weight, linespacing=1.25, **kw)


def box(ax, x, y, w, h, value="", accent=False, size=18, fill=None,
        color=None, dashed=False, weight="normal"):
    ec = RED if accent else LINE
    patch = Rectangle((x, y), w, h, linewidth=1.0 if accent else .8, edgecolor=ec,
                      facecolor=fill or (PALE if accent else WHITE),
                      linestyle=(0, (5, 4)) if dashed else "solid", zorder=2)
    ax.add_patch(patch)
    if value:
        t = text(ax, x+w/2, y+h/2, value, size, color or INK, weight=weight)
        t._container = (x, y, w, h)
    return patch


def arrow(ax, start, end, color=GRAY, dashed=False, lw=1.0, style="-|>", **kw):
    a = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=10,
                        linewidth=lw, color=color, shrinkA=2.5, shrinkB=2.5,
                        linestyle=(0, (5, 4)) if dashed else "solid", zorder=1, **kw)
    ax.add_patch(a)
    return a


def route(ax, points, color=GRAY, dashed=False, lw=1.0):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=lw,
                linestyle=(0, (5, 4)) if dashed else "solid", zorder=1)
    arrow(ax, points[-2], points[-1], color=color, dashed=dashed, lw=lw)


def footer(ax, value, y=.42, size=13):
    text(ax, .75, y, value, min(size,12), GRAY, ha="left")


def save(fig, name):
    """Save identical uncropped 16:9 artboards; retain SVG text elements."""
    for t in fig.findobj(match=Text):
        t.set_linespacing(1.25)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems = []
    labels = []
    for ax in fig.axes:
        for t in list(ax.texts)+[ax.xaxis.label,ax.yaxis.label,ax.title]:
            if not t.get_visible() or not t.get_text():
                continue
            bb = t.get_window_extent(renderer)
            labels.append((t.get_text(), bb))
            if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.x1+1 or bb.y1 > fig.bbox.y1+1:
                problems.append({"text": t.get_text(), "issue": "artboard overflow"})
            if hasattr(t, "_container"):
                x, y, w, h = t._container
                p0, p1 = ax.transData.transform([(x, y), (x+w, y+h)])
                if bb.x0 < p0[0]+3 or bb.x1 > p1[0]-3 or bb.y0 < p0[1]+3 or bb.y1 > p1[1]-3:
                    problems.append({"text": t.get_text(), "issue": "box overflow"})
    for i,(a,ba) in enumerate(labels):
        for b,bb in labels[i+1:]:
            w,h=min(ba.x1,bb.x1)-max(ba.x0,bb.x0),min(ba.y1,bb.y1)-max(ba.y0,bb.y0)
            if w>2 and h>2:
                problems.append({"text":a,"other_text":b,"issue":"text overlap"})
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=300)
    AUDIT.append({"figure": name, "size_inches": list(fig.get_size_inches()),
                  "png_dpi": 300, "line_spacing":1.25, "layout_issues": problems})
    if problems:
        print(name, "LAYOUT:", problems)
    plt.close(fig)


def contact_sheet(names, target, cols=3, thumb=(960, 540)):
    rows = (len(names)+cols-1)//cols
    gap=24
    sheet = Image.new("RGB", (cols*thumb[0]+(cols+1)*gap, rows*thumb[1]+(rows+1)*gap), WHITE)
    for i, name in enumerate(names):
        img = Image.open(FIGURES / f"{name}.png").convert("RGB")
        img = ImageOps.contain(img, thumb, Image.Resampling.LANCZOS)
        x, y = gap+(i%cols)*(thumb[0]+gap), gap+(i//cols)*(thumb[1]+gap)
        sheet.paste(img, (x, y))
    sheet.save(target, dpi=(300, 300))


def write_layout_audit():
    path = QA / "layout_audit.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    new = {item["figure"]: item for item in old}
    new.update({item["figure"]: item for item in AUDIT})
    path.write_text(json.dumps(list(new.values()), indent=2, ensure_ascii=False), encoding="utf-8")


setup()
