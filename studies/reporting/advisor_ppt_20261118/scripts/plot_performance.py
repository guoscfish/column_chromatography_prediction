"""Render the performance figures using only the audited validation summaries."""
from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

REPORT = Path(__file__).resolve().parents[1]
DATA = REPORT / "sources"
PLOTS = REPORT / "plots"
RED = "#8C1D2B"
INK = "#182838"
MUTED = "#526171"
GRID = "#E3E8ED"
SEEDS = [157, 887, 2357, 6101, 12203]
METHODS = {
    "cw_gradient_lcmd32": ("CW-LCMD32", "#1A5485", "o"),
    "historical_free_mixed_model": ("Historical CW-Free32", "#AC4E11", "s"),
    "new_free_cw_sol61_xhigh": ("New CW-Free32", "#E07719", "D"),
    "nocw": ("NoCW-Free32", "#875344", "^"),
    "free16_lcmd16": ("CW16+LLM16", RED, "D"),
    "numerical_hybrid32": ("数值 Hybrid32", "#1A5485", "o"),
    "gradient_lcmd32": ("raw Gradient-LCMD32", "#3A79AA", "s"),
    "random32": ("Random32", "#666E77", "^"),
}
LONG = list(METHODS)[:4]
SHORT = ["numerical_hybrid32", "gradient_lcmd32", "new_free_cw_sol61_xhigh", "random32"]
HYBRID = ["cw_gradient_lcmd32", "free16_lcmd16"]


def configure():
    candidates = [
        os.environ.get("ADVISOR_CJK_FONT", ""),
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            font_manager.fontManager.addfont(candidate)
            family = font_manager.FontProperties(fname=candidate).get_name()
            break
    else:
        family = "Noto Sans CJK SC"
        font_manager.findfont(family, fallback_to_default=False)
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", family], "axes.unicode_minus": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "font.size": 15, "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": "#CCD5DD", "svg.fonttype": "path",
        "pdf.fonttype": 42, "savefig.facecolor": "white",
    })


def canvas(num, title, subtitle, source):
    fig = plt.figure(figsize=(16, 9))
    fig.text(.062, .937, num, color=RED, weight="bold", fontsize=16)
    fig.text(.124, .931, title, color=INK, weight="bold", fontsize=30)
    fig.text(.062, .871, subtitle, color=MUTED, fontsize=16)
    fig.add_artist(Line2D([.062, .947], [.836, .836], transform=fig.transFigure,
                          color=GRID, lw=1))
    fig.add_artist(Line2D([.062, .947], [.086, .086], transform=fig.transFigure,
                          color=GRID, lw=1))
    fig.text(.062, .056, source, color=MUTED, fontsize=11.5)
    return fig


def axis_plot(ax, budget_ticks, ylabel=False):
    ax.set_xticks(budget_ticks)
    ax.set_xlim(budget_ticks[0], budget_ticks[-1])
    ax.set_ylim(.23, 1.04)
    ax.set_yticks([.25, .40, .55, .70, .85, 1.00])
    ax.tick_params(length=0, pad=9, labelsize=13)
    ax.grid(axis="y", color=GRID, zorder=0)
    if ylabel:
        ax.set_ylabel("combined validation NRMSE", fontsize=15, labelpad=12)
    else:
        ax.tick_params(labelleft=False)


def save(fig, number, stem):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(PLOTS / f"{number}_{stem}.{ext}", dpi=250)
    plt.close(fig)


def long_curves(curves):
    fig = canvas("F01", "五条种子轨迹的长期学习表现",
                 "4g Row 验证集 · L333–1485 · 每法 5 seeds · 每 32 行一个原始检查点",
                 "源  L1485 五 seed 终表 / llm_decision_behavior row_curves.csv  |  "
                 "历史 Free 含混合模型配置，NoCW 为历史观察对照")
    grid = fig.add_gridspec(2, 2, left=.10, right=.945, bottom=.168, top=.766,
                           hspace=.48, wspace=.15)
    for index, method in enumerate(LONG):
        col = index % 2
        ax = fig.add_subplot(grid[index // 2, col])
        name, color, marker = METHODS[method]
        block = curves.loc[curves.method == method]
        matrix = block.pivot(index="budget", columns="seed", values="nrmse")
        means = matrix.mean(axis=1)
        sd = matrix.std(axis=1, ddof=1)
        ax.fill_between(matrix.index, means - sd, means + sd, color=color, alpha=.17, lw=0)
        ax.plot(matrix.index, means, color=color, lw=2.7, marker=marker, ms=3.1)
        axis_plot(ax, [333, 525, 813, 1005, 1229, 1485], ylabel=col == 0)
        ax.set_title(name, loc="left", color=color, weight="bold", fontsize=17, pad=10)
    fig.text(.50, .115, "累计训练标签数 L（行）   ·   实线为逐点均值，色带为 ±1 个 seed 样本标准差",
             ha="center", color=MUTED, fontsize=14)
    save(fig, "F01", "five_seed_long_learning_curves")


def paired_metrics(metrics, methods, short=False):
    if short:
        num, stem, endpoint = "F08", "five_seed_short_window_baselines", 1005
        title = "共同短窗内，数值基线的平均误差更低"
        subtitle = "L333–1005 · 5 个匹配 seeds · 4g Row 验证集 · 点按预算的原值积分"
        source = "源  comprehensive_history_diagnostics validation 汇总  |  数值 Hybrid32 为分歧筛选加 latent 覆盖，跨来源历史对照"
        limits = ((.31, .81), (.26, .76))
    else:
        num, stem, endpoint = "F02", "five_seed_auc_vs_endpoint", 1485
        title = "长期面积与终点误差给出不同排名"
        subtitle = "L333–1485 · 5 个匹配 seeds · 4g Row 验证集 · 两个指标均为越低越好"
        source = "源  L1485 五 seed 终表 / llm_decision_behavior row_curves.csv  |  配对保持同 seed，历史分支在采样后分叉"
        limits = ((.30, .58), (.23, .41))
    fig = canvas(num, title, subtitle, source)
    grid = fig.add_gridspec(1, 2, left=.101, right=.942, bottom=.231, top=.743, wspace=.23)
    for i, (metric, metric_title, bounds) in enumerate(zip(
            ("aulc", "endpoint"), ("raw AULC", f"NRMSE @ L{endpoint}"), limits)):
        ax = fig.add_subplot(grid[0, i])
        matrix = metrics.pivot(index="seed", columns="method", values=metric)[methods]
        x = np.arange(len(methods))
        for seed in matrix.index:
            ax.plot(x, matrix.loc[seed], color="#BDC6CF", linewidth=1.1, alpha=.85, zorder=2)
        for xpos, method in zip(x, methods):
            label, color, mark = METHODS[method]
            ys = matrix[method]
            ax.scatter(np.repeat(xpos, len(ys)), ys, color=color, s=47, marker=mark,
                       alpha=.85, edgecolor="white", linewidth=.5, zorder=3)
            mu = ys.mean()
            ax.scatter(xpos, mu, s=180, marker=mark, facecolor="white",
                       edgecolor=color, linewidth=2.6, zorder=4)
            ax.annotate(f"{mu:.3f}", (xpos, mu), xytext=(10, 0), textcoords="offset points",
                        va="center", color=color, weight="bold", fontsize=14,
                        bbox={"facecolor":"white", "edgecolor":"none", "pad":.9, "alpha":.92},
                        zorder=5)
        names = (["数值 Hybrid32", "raw Gradient-\nLCMD32", "New CW-\nFree32", "Random32"]
                 if short else
                 ["CW-LCMD32", "Historical\nCW-Free32", "New\nCW-Free32", "NoCW-Free32"])
        ax.set_xticks(x, names, fontsize=13)
        ax.set_xlim(-.38, 3.58)
        ax.set_ylim(bounds)
        ax.yaxis.set_major_locator(MaxNLocator(7, prune="both"))
        ax.tick_params(length=0, pad=9, labelsize=13)
        ax.grid(axis="y", color=GRID)
        ax.set_title(metric_title, loc="left", fontsize=21, weight="bold", pad=17)
    fig.text(.50, .116, "实心点为单 seed，灰线连接同一 seed，空心点为五 seed 均值",
             ha="center", color=MUTED, fontsize=14)
    save(fig, num, stem)


def hybrid_curves(curves, metrics):
    fig = canvas("F03", "16+16 融合的两条历史配对轨迹",
                 "4g Row 验证集 · seeds 157、6101 · 共同预算 L333–1453 · 每批新增 32 行",
                 "源  decision_mechanism_followup / hybrid_phase_differences  |  "
                 "融合 LLM 于选前 L813 切换 Astra/high → Sol/xhigh，分叉后状态不同")
    grid = fig.add_gridspec(1, 2, left=.10, right=.946, bottom=.267, top=.753, wspace=.125)
    for i, seed in enumerate([157, 6101]):
        ax = fig.add_subplot(grid[0, i])
        axis_plot(ax, [333, 525, 813, 1005, 1229, 1453], ylabel=i == 0)
        ax.set_title(f"Seed {seed}", loc="left", weight="bold", pad=12)
        for method in HYBRID:
            name, color, marker = METHODS[method]
            block = curves.loc[(curves.method == method) & (curves.seed == seed)]
            ax.plot(block.budget, block.nrmse, color=color, marker=marker, ms=3.8, lw=2.8)
        ax.axvline(813, color="#9BA9B5", linestyle=":", lw=1)
    fig.text(.50, .205, "累计训练标签数 L（行）", ha="center", color=MUTED, fontsize=15)
    for i, method in enumerate(HYBRID):
        name, color, marker = METHODS[method]
        block = metrics.loc[metrics.method == method]
        fig.text(.115 + .44 * i, .142, f"{name}   |   AULC {block.aulc.mean():.4f}   "
                 f"·   L1453 {block.endpoint.mean():.4f}", color=color, fontsize=16, weight="bold")
    save(fig, "F03", "hybrid16_lcmd16_two_seed")


def main():
    configure()
    curves = pd.read_csv(DATA / "performance_curves.csv").sort_values("budget")
    metrics = pd.read_csv(DATA / "performance_seed_metrics.csv")
    long_curves(curves[curves.window == "long_five_seed"])
    paired_metrics(metrics[metrics.window == "long_five_seed"], LONG)
    hybrid_curves(curves[curves.window == "hybrid_two_seed"],
                  metrics[metrics.window == "hybrid_two_seed"])
    paired_metrics(metrics[metrics.window == "short_five_seed"], SHORT, short=True)


if __name__ == "__main__":
    main()
