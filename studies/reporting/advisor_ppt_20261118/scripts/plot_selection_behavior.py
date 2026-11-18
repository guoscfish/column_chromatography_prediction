"""Render F04–F05 from audited metadata-only plotting tables."""
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_performance import DATA, GRID, INK, METHODS, MUTED, RED, canvas, configure, save

BLUE = METHODS["cw_gradient_lcmd32"][1]
FREE = METHODS["new_free_cw_sol61_xhigh"][1]
ORANGE_DARK = METHODS["historical_free_mixed_model"][1]
GRAY_FILL = "#E8EDF1"
ARMS = (("lcmd_back16", BLUE), ("llm16", RED))
SEEDS = (157, 6101)


def replay_axes(ax, limit, ticks):
    ax.set_ylim(0, limit)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlim(-.54, 1.54)
    ax.tick_params(length=0, pad=10, labelsize=14)
    ax.grid(axis="y", color=GRID, linewidth=.8, zorder=0)


def hybrid_replay(summary, states):
    summary = summary.loc[summary.scope == "all"].set_index(["seed", "method_key"])
    fig = canvas(
        "F04", "同一 Hybrid 状态下，后 16 个名额的选择取舍",
        "71 个冻结状态 · 两种选法共用历史 H、候选池和 CW-LCMD 前 16 · 71/71 有序匹配",
        "源  decision_mechanism_followup_20261117 / HYBRID_COUNTERFACTUAL  |  同状态候选集合回放",
    )
    grid = fig.add_gridspec(
        1, 2, left=.100, right=.942, bottom=.225, top=.691, wspace=.27,
    )
    volume = fig.add_subplot(grid[0, 0])
    cw = fig.add_subplot(grid[0, 1])
    fig.legend(
        handles=[
            Patch(facecolor=BLUE, label="CW-LCMD 后 16（回放）"),
            Patch(facecolor=RED, label="Hybrid 的 LLM 后 16（实际）"),
        ],
        loc="center", bbox_to_anchor=(.5, .793), ncols=2, frameon=False, fontsize=15,
        handlelength=1.2, handleheight=.8, columnspacing=2.4,
    )

    volume.set_title("样品体积单因素对照的名额比例", loc="left", fontsize=22, pad=18)
    cw.set_title("CW 边际覆盖增量", loc="left", fontsize=22, pad=18)
    volume.set_ylabel("后 16 中有 H+前 16 锚点的比例", fontsize=15, labelpad=11)
    cw.set_ylabel("参考平均最近距离降幅（相对选前 H）", fontsize=15, labelpad=11)
    replay_axes(volume, 64, [0, 15, 30, 45, 60])
    replay_axes(cw, 16, [0, 4, 8, 12, 16])
    for ax in (volume, cw):
        ticks = [
            f"Seed {seed} · {summary.loc[(seed, 'llm16'), 'batches']} 批\n"
            f"L333–{summary.loc[(seed, 'llm16'), 'L_end']}"
            for seed in SEEDS
        ]
        ax.set_xticks([0, 1], ticks)
    for group_index, seed in enumerate(SEEDS):
        for displacement, (arm, color) in zip((-.18, .18), ARMS):
            rec = summary.loc[(seed, arm)]
            x = group_index + displacement
            share = 100 * rec.volume_hn_slots / rec.slots
            volume.bar(x, share, width=.32, color=color, zorder=3)
            volume.annotate(
                f"{share:.2f}%\n{rec.volume_hn_slots}/{rec.slots}", (x, share),
                xytext=(0, 8), textcoords="offset points", ha="center",
                color=color, fontsize=14, linespacing=1.35, fontweight="bold",
            )
            state_gains = 100 * states.loc[states.seed == seed, f"{arm}_cw_gain_relative_h"]
            jitter = .13 * np.sin(np.arange(len(state_gains)) * 2.3999632)
            cw.scatter(x + jitter, state_gains, s=20, color=color, alpha=.28, zorder=2,
                       lw=0, clip_on=False)
            cw.scatter(x, rec.cw_gain_pct, s=125, marker="D", facecolor="white", edgecolor=color,
                       linewidth=2.25, zorder=4)
            cw.annotate(
                f"{rec.cw_gain_pct:.2f}%", (x, rec.cw_gain_pct),
                xytext=(0, 10), textcoords="offset points", ha="center",
                color=color, fontsize=15, fontweight="bold", zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .88, "pad": 1.5},
            )
    both_seeds = summary.xs("llm16", level="method_key")
    more_volume = both_seeds.volume_hn_delta_positive.sum()
    lower_coverage = both_seeds.cw_delta_negative.sum()
    same_coverage = both_seeds.cw_delta_zero.sum()
    higher_coverage = both_seeds.cw_delta_positive.sum()
    fig.text(
        .062, .113,
        f"LLM 体积对照增加 {more_volume}/71 批 · CW 增量较低／持平／较高 "
        f"{lower_coverage}／{same_coverage}／{higher_coverage} 批 · 右侧小点为状态，菱形为 seed 均值",
        fontsize=14, color=MUTED,
    )
    save(fig, "F04", "hybrid_replay_71_batches")


def prefix_failure(summary):
    overall = summary.loc[summary.method_key == "all"].iloc[0]
    strict = overall.accepted_exact_receipt
    count = overall.prefix_batches
    total = overall.batches
    fig = canvas(
        "F05", "Free32 纠错后出现展示前缀，局部草稿可保留原选择",
        "5 seeds · L333–1485 · 3 种策略各 180 批 · 540 批含 1 次已授权人工 ID 修正（非前缀）",
        "源  volume_response_retry_audit_20261117 / PREFIX_FAILURE_ANALYSIS、LOCAL_REPAIR_DESIGN  |  C2 / C4",
    )
    left, right = .063, .535
    fig.add_artist(Line2D([.503, .503], [.170, .790], transform=fig.transFigure, lw=1, color=GRID))
    fig.text(left, .772, "最终顺序与展示前 32 行完全一致", fontsize=20)
    fig.text(right, .772, "原提议的选择保留与待补空额", fontsize=20)
    fig.text(left, .701, f"{count}/{total} 批   {count / total:.2%}", color=ORANGE_DARK,
             fontsize=28, fontweight="bold")
    fig.text(right, .705, f"{overall.local_repair_passes_gate}/{count} 份草稿通过完整校验",
             color=INK, fontsize=25, fontweight="bold")

    ax = fig.add_axes([left, .322, .386, .326])
    free_rows = summary.loc[summary.method_key != "all"]
    for position, (_, row) in enumerate(free_rows.iterrows()):
        y = 2 - position
        color = METHODS[row.method_key][1]
        fraction = row.prefix_batches / row.batches
        ax.barh(y - .18, .4, height=.14, color=GRAY_FILL)
        ax.barh(y - .18, fraction, height=.14, color=color)
        ax.text(0, y + .085, row.method, va="center", color=INK, fontsize=14)
        ax.text(fraction + .009, y - .18, f"{row.prefix_batches}/{row.batches}  {fraction:.1%}",
                va="center", color=color, fontsize=13.5, fontweight="bold")
    for tick in (0, .20, .40):
        ax.text(tick, -.45, f"{tick:.0%}", ha="left" if tick == 0 else "center",
                va="top", fontsize=11.5, color=MUTED)
    ax.set(xlim=(0, .435), ylim=(-.69, 2.46))
    ax.axis("off")
    fig.text(left, .259, f"{count} 批均先触发校验失败", fontsize=17, color=INK)
    fig.text(left, .219,
             f"首错  {overall.prefix_candidate_identity} 候选身份  /  "
             f"{overall.prefix_packet_binding} packet 绑定", fontsize=15, color=MUTED)
    fig.text(left, .171, f"仅计与原答精确匹配的记录，前缀率为 {count}/{strict}。", fontsize=13, color=MUTED)

    repair = fig.add_axes([right, .309, .407, .342])
    retention = overall.historical_retained / overall.original_legal
    preserved = overall.local_repair_preserved / overall.original_slots
    for y, title, fraction, n, denom in (
        (1.9, "历史最终保留 / 原合法 ID", retention, overall.historical_retained, overall.original_legal),
        (.76, "离线草稿保留 / 原始 32 名额", preserved, overall.local_repair_preserved, overall.original_slots),
    ):
        repair.text(0, y + .27, title, color=INK, fontsize=15)
        repair.text(1, y + .27, f"{n}/{denom}  {fraction:.2%}", ha="right",
                    color=ORANGE_DARK, fontsize=15, fontweight="bold")
        repair.barh(y, 1, height=.21, color=GRAY_FILL)
        repair.barh(y, fraction, height=.21, color=FREE)
    repair.set(xlim=(0, 1), ylim=(0, 2.45))
    repair.axis("off")
    fig.text(right, .272, f"{overall.local_repair_vacant} 个空额未填补   ·   "
             f"{overall.local_repair_science_review} 批待复核证据", fontsize=16)
    fig.text(right, .226, "计数单位为每批去重后的选择名额，跨批可重复。", fontsize=13, color=MUTED)

    fig.text(.062, .112, "历史记录与离线保真诊断 · 尚未检验局部纠错对合法率或 Row 学习收益的影响",
             fontsize=14, color=MUTED)
    save(fig, "F05", "free_prefix_and_repair")


def main():
    configure()
    hybrid_replay(pd.read_csv(DATA / "f04_replay_seed_summary.csv"),
                  pd.read_csv(DATA / "f04_replay_states.csv"))
    prefix_failure(pd.read_csv(DATA / "f05_prefix_repair_summary.csv"))


if __name__ == "__main__":
    main()
