#!/usr/bin/env python3
"""Render frozen QGeoGNN sensitivity and fixed-label controls for a 16:9 slide."""

import csv
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "advisor_ppt_20261118_matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.lines import Line2D


REPORT = Path(__file__).resolve().parents[1]
SOURCES = REPORT / "sources"
PLOTS = REPORT / "plots"
RED = "#8C1D2B"
BLUE = "#245782"
INK = "#182638"
SECONDARY = "#4F6273"
GRID = "#E1E8EE"
PALE = "#FAF1F2"
SEEDS = (157, 887, 2357, 6101, 12203)
SEED_INK = ("#21537C", "#4291AA", "#53638D", "#648D77", "#786D9A")
ARMS = ("A_original", "B_amount_skip", "C_volume_skip")


def setup():
    requested = os.environ.get("ADVISOR_CJK_FONT")
    candidates = [
        requested,
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    font_path = next((Path(path) for path in candidates if path and Path(path).is_file()), None)
    if not font_path:
        raise RuntimeError("Set ADVISOR_CJK_FONT to a Chinese-capable font file")
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=font_path).get_name(),
        "font.size": 13,
        "mathtext.fontset": "dejavusans",
        "text.color": INK,
        "axes.labelcolor": SECONDARY,
        "axes.edgecolor": GRID,
        "xtick.color": SECONDARY,
        "ytick.color": SECONDARY,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "path",
    })


def read(name):
    with (SOURCES / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def canvas(title, subtitle):
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.add_artist(Line2D([.055, .089], [.938, .938], color=RED, lw=4, transform=fig.transFigure))
    fig.text(.055, .891, title, fontsize=31, color=INK, va="center")
    fig.text(.055, .837, subtitle, fontsize=15, color=SECONDARY, va="center")
    return fig


def finish(fig, stem, footer):
    fig.add_artist(Line2D([.055, .945], [.099, .099], color=GRID, lw=1, transform=fig.transFigure))
    for y, text in zip((.069, .041), footer):
        fig.text(.055, y, text, fontsize=10.5, color=SECONDARY, va="center")
    PLOTS.mkdir(exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(PLOTS / f"{stem}.{extension}", dpi=250, facecolor="white")
    plt.close(fig)


def panel_style(ax, grid_axis):
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", length=0, pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=.85)


def plot_sensitivity():
    fig = canvas(
        "冻结模型对样品体积的预测响应明显偏小",
        "原 QGeoGNN 的两个预算检查点  /  5 个 seed  /  同分子单因素对的真实与预测幅度",
    )
    summary = read("f06_frozen_response_equal_seed.csv")
    seed_rows = read("f06_frozen_response_seed.csv")
    series = [
        ("observed_abs_seed_mean", "observed_abs_mean", INK, "观测"),
        ("predicted_abs_seed_mean", "predicted_abs_mean", BLUE, "预测"),
    ]
    handles = [Line2D([], [], marker="o", ls="-", lw=1.2, color=color, markersize=8,
                      markeredgecolor="white", label=label) for _, _, color, label in series]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.949, .808), ncol=2,
               frameon=False, fontsize=14, handlelength=1.6, columnspacing=1.7)
    for x, axis, title, scale in (
        (.139, "V/ul", "样品体积", "对数刻度"),
        (.621, "PE/EA", "洗脱剂 PE/EA", "线性刻度"),
    ):
        ax = fig.add_axes((x, .237, .322, .470))
        panel_style(ax, "x")
        ax.set_title(f"{title}   ·   {scale}", fontsize=20, loc="left", pad=20)
        pos = [3.3, 2.3, 1.1, .1]
        sequence = [("525", "V1"), ("525", "V2"), ("1005", "V1"), ("1005", "V2")]
        ax.set_yticks(pos, [f"L{budget} · {target}" for budget, target in sequence], fontsize=12.5)
        ax.set_ylim(-.55, 3.95)
        ax.axhline(1.7, color=GRID, lw=1.0)
        if axis == "V/ul":
            ax.set_xscale("log")
            ax.set_xlim(.0004, 12)
            ax.set_xticks([.001, .01, .1, 1, 10], ["0.001", "0.01", "0.1", "1", "10"])
            ax.xaxis.set_minor_locator(ticker.NullLocator())
        else:
            ax.set_xlim(0, 37)
            ax.set_xticks([0, 10, 20, 30])
        for y, (budget, target) in zip(pos, sequence):
            row = next(r for r in summary if (r["budget"], r["axis"], r["target"]) == (budget, axis, target))
            observed, predicted = float(row[series[0][0]]), float(row[series[1][0]])
            ax.hlines(y, min(observed, predicted), max(observed, predicted), color=GRID, linewidth=2.0, zorder=1)
            matched = [r for r in seed_rows if (r["budget"], r["axis"], r["target"]) == (budget, axis, target)]
            for index, (mean_key, seed_key, color, _) in enumerate(series):
                point = float(row[mean_key])
                equal = [float(r[seed_key]) for r in matched]
                offsets = [.12, .15, .18, .21, .24]
                offset_sign = 1 if index == 0 else -1
                ax.hlines(y + offset_sign * .18, min(equal), max(equal), color=color, lw=1.0, alpha=.48, zorder=2)
                ax.scatter(equal, [y + offset_sign * value for value in offsets], s=19, color=color,
                           alpha=.58, edgecolors="white", linewidths=.25, zorder=3)
                ax.scatter(point, y, s=94, color=color, edgecolors="white", linewidths=1, zorder=4)
                decimals = 3 if axis == "V/ul" and index == 1 else 2
                ax.annotate(f"{point:.{decimals}f}", (point, y), xytext=(0, 21*offset_sign),
                            textcoords="offset points", color=color, ha="center",
                            va="bottom" if index == 0 else "top", fontsize=12)
        ax.set_xlabel("平均绝对响应幅度  |Δ|（mL）", fontsize=14, labelpad=14)
    fig.text(.055, .136, "L525 / L1005 体积对数 464 / 1,393，PE/EA 对数 1,277 / 4,054。每对记录 V1 与 V2。",
             fontsize=12.5, color=INK, va="center")
    finish(fig, "qgeognn_loading_volume_sensitivity", (
        "来源  Volume response retry audit 20261117 · A2。已观察训练历史 H。大点是五 seed 等权均值，小点与线段是逐 seed 均值及范围。",
        "各预算的 H 与配对构成随标签变化。两个面板的横轴刻度不同。seed 157 / 887 / 2357 / 6101 / 12203。",
    ))


def plot_controls():
    fig = canvas(
        "两种简单 Skip 均未通过预设门槛",
        "从头拟合 15 次  /  3 个模型 × 5 个配对 seed  /  同 seed 固定 L1005 历史标签与预定检查系列",
    )
    means = {r["kind"]: r for r in read("f07_control_equal_seed.csv")}
    rows = {(int(r["seed"]), r["kind"]): r for r in read("f07_control_seed.csv")}
    handles = [Line2D([], [], marker="o", ls="-", color=color, markersize=6, label=str(seed))
               for seed, color in zip(SEEDS, SEED_INK)]
    handles.append(Line2D([], [], marker="D", ls="none", markersize=7.5,
                          markerfacecolor=RED, markeredgecolor="white", label="五 seed 均值"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .810), ncol=6, title="配对 seed",
               frameon=False, fontsize=12.5, title_fontsize=12.5, handlelength=1.7, columnspacing=1.7)
    for left, metric, title, limits, ticks, fmt in (
        (.09, "difference_error_mean", "隔离系列差分 MAE（mL）", (2.35, 5.20), [2.5, 3.0, 3.5, 4.0, 4.5, 5.0], ".3f"),
        (.55, "combined_normalized_rmse", "4g Row validation combined NRMSE", (.330, .452),
         [.34, .36, .38, .40, .42, .44], ".5f"),
    ):
        ax = fig.add_axes((left, .322, .395, .363))
        panel_style(ax, "y")
        ax.set_title(title, loc="left", fontsize=20, pad=15)
        ax.set(xlim=(-.42, 2.42), ylim=limits, yticks=ticks)
        ax.set_xticks(range(3), ["原模型", "上样量 Skip", "独立体积 Skip"], fontsize=13)
        for seed, color in zip(SEEDS, SEED_INK):
            values = [float(rows[seed, arm][metric]) for arm in ARMS]
            ax.plot(range(3), values, color=color, alpha=.86, marker="o", ms=6.5,
                    lw=1.45, mec="white", mew=.55, zorder=2)
        point_values = [float(means[arm][metric]) for arm in ARMS]
        ax.plot(range(3), point_values, marker="D", ms=9, lw=2.1, color=RED, mec="white", mew=.75, zorder=3)
        for i, point in enumerate(point_values):
            local = metric == "difference_error_mean"
            ax.annotate(format(point, fmt), (i, point), xytext=(0, -12 if local else 12),
                        textcoords="offset points", color=RED, fontsize=13, ha="center",
                        va="top" if local else "bottom")
    gates = {r["comparison"]: r for r in read("f07_control_gates.csv")}
    b, c = gates["B-A"], gates["C-A"]
    row_b, row_c = (-100*float(g["row_relative_change_equal_seed_means"]) for g in (b, c))
    local_b, local_c = (-100*float(g["local_improvement_ratio_equal_seed_means"]) for g in (b, c))
    assert all(g["mechanism_reference_pass"] == g["row_relevance_reference_pass"] == "False" for g in (b, c))
    fig.text(.092, .215, f"局部 MAE 变差 {local_b:.2f}% / {local_c:.2f}%   ·   改善各 3/5 seed",
             fontsize=15, color=RED, va="center")
    fig.text(.550, .215, f"Row NRMSE 改善 {row_b:.2f}% / {row_c:.2f}%   ·   不变差各 3/5 seed",
             fontsize=15, color=RED, va="center")
    fig.text(.055, .143, "机制门槛需局部改善 ≥20%、至少 4/5 seed 改善且平均 Row 恶化 ≤1%。Row 门槛需平均改善 ≥1%、至少 4/5 seed 不变差。",
             fontsize=12, color=RED, va="center",
             bbox=dict(boxstyle="round,pad=.55,rounding_size=.25", facecolor=PALE, edgecolor="none"))
    finish(fig, "qgeognn_15fit_negative_control", (
        "来源  Volume response retry audit 20261117 · 三模型固定标签对照。局部每 seed 10 系列、共 68 对/端点，V1/V2 等权。",
        "Row validation 每次 416 行、使用冻结 L333 目标尺度。检查系列不参与训练或早停，其中部分条件与原 validation 关联。",
    ))


if __name__ == "__main__":
    setup()
    plot_sensitivity()
    plot_controls()
