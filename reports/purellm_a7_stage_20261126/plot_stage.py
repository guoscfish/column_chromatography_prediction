"""Render the paired PureLLM-A7/M2 and Hybrid validation checkpoints."""

import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MaxNLocator


SEEDS = (157, 887, 2357, 6101, 12203)
FIRST, LAST, BATCH = 333, 1485, 32
METRICS = (
    ("validation_nrmse_combined", "Combined NRMSE ↓"),
    ("validation_r2_v1", "V1 validation R² ↑"),
    ("validation_r2_v2", "V2 validation R² ↑"),
)
PALETTE = ("#0072B2", "#D55E00", "#00876A", "#AA4499", "#8A6B00")
MARKERS = ("o", "s", "^", "D", "v")
INK, MUTED, GRID = "#18263A", "#526175", "#DFE5EC"


def read_curves(path, *, complete):
    """Read only reported points, requiring the exact B32 budget series."""
    curves = defaultdict(dict)
    required = {"seed", "active_labels", *(metric for metric, _ in METRICS)}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        if not required.issubset(rows.fieldnames or ()):
            raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(required - set(rows.fieldnames or ())))}")
        for line, row in enumerate(rows, 2):
            try:
                seed, labels = int(row["seed"]), int(row["active_labels"])
                values = tuple(float(row[metric]) for metric, _ in METRICS)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path.name}:{line} has an invalid validation value") from error
            if (seed not in SEEDS or labels not in range(FIRST, LAST + 1, BATCH)
                    or labels in curves[seed] or not all(math.isfinite(value) for value in values)):
                raise ValueError(f"{path.name}:{line} has an invalid or repeated seed/budget checkpoint")
            curves[seed][labels] = values
    if set(curves) != set(SEEDS):
        raise ValueError(f"{path.name} must contain exactly the five paired seeds")
    for seed, points in curves.items():
        end = LAST if complete else max(points)
        if sorted(points) != list(range(FIRST, end + 1, BATCH)):
            raise ValueError(f"{path.name}: seed {seed} has missing validation checkpoints")
    return curves


def _plot_curve(ax, points, metric, color, marker, upper, *, purellm, overview):
    xs = sorted(labels for labels in points if labels <= upper)
    ys = [points[labels][metric] for labels in xs]
    if purellm:
        ax.plot(xs, ys, color=color, linewidth=1.9, marker=marker, markersize=3.5, zorder=3)
        ax.plot(xs[-1], ys[-1], color=color, marker=marker, markersize=7.2,
                markeredgecolor="white", markeredgewidth=0.9, zorder=4)
    else:
        ax.plot(xs, ys, color=color, linewidth=1.35, linestyle=(0, (4.1, 2.25)), alpha=0.79,
                marker=marker, markerfacecolor="white", markeredgewidth=0.8, markersize=3.6,
                markevery=6 if overview else 1, zorder=2)


def render(purellm, hybrid, output, *, snapshot=None):
    end_by_seed = {seed: max(purellm[seed]) for seed in SEEDS}
    stage_end = max(end_by_seed.values())
    if stage_end <= FIRST:
        raise ValueError("the PureLLM export does not include a fitted stage checkpoint")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK, "axes.edgecolor": GRID, "axes.titlecolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(3, 2, figsize=(13.6, 9.6), sharex="col", sharey="row",
                             gridspec_kw={"width_ratios": (1.34, 1)})
    fig.subplots_adjust(left=0.106, right=0.966, top=0.790, bottom=0.145, wspace=0.115, hspace=0.25)

    fig.text(0.106, 0.956, "PureLLM-A7/M2 vs Hybrid", fontsize=23, weight="bold", va="top")
    subtitle = "QGeoGNN validation on five matched seed splits  ·  Batch size 32  ·  Common L333 start"
    if snapshot:
        subtitle += f"  ·  {snapshot}"
    fig.text(0.106, 0.908, subtitle, fontsize=10.4, color=MUTED, va="top")
    method_keys = [Line2D([], [], color=INK, linewidth=1.9, marker="o", markersize=5, label="PureLLM-A7/M2"),
                   Line2D([], [], color=INK, linewidth=1.35, linestyle=(0, (4.1, 2.25)), marker="o",
                          markerfacecolor="white", markersize=5, label="Hybrid")]
    seed_keys = [Line2D([], [], linestyle="none", color=color, marker=marker, markersize=5.5, label=str(seed))
                 for seed, color, marker in zip(SEEDS, PALETTE, MARKERS, strict=True)]
    fig.legend(handles=method_keys, loc="upper left", bbox_to_anchor=(0.106, 0.882), ncol=2,
               frameon=False, borderaxespad=0, handlelength=2.4, columnspacing=1.65)
    fig.legend(handles=seed_keys, title="Seed", loc="upper right", bbox_to_anchor=(0.966, 0.883), ncol=5,
               frameon=False, borderaxespad=0, handlelength=0.95, columnspacing=1.25, fontsize=9.4, title_fontsize=8.9)

    full_ticks = list(range(FIRST, LAST + 1, 6 * BATCH))
    stage_step = BATCH * max(1, math.ceil((stage_end - FIRST) / (BATCH * 6)))
    stage_ticks = list(range(FIRST, stage_end + 1, stage_step))
    if stage_ticks[-1] != stage_end:
        if stage_end - stage_ticks[-1] < stage_step * 0.65:
            stage_ticks[-1] = stage_end
        else:
            stage_ticks.append(stage_end)

    for metric, (_, name) in enumerate(METRICS):
        left, right = axes[metric]
        left.set_ylabel(name, fontsize=11.5, labelpad=11, weight="semibold")
        if metric == 0:
            left.set_title(f"Stage detail through L{stage_end}", loc="left", fontsize=12, weight="semibold", pad=9)
            right.set_title("Full budget range through L1485", loc="left", fontsize=12, weight="semibold", pad=9)
        for col, ax in enumerate((left, right)):
            overview = col == 1
            upper = LAST if overview else stage_end
            for seed, color, marker in zip(SEEDS, PALETTE, MARKERS, strict=True):
                _plot_curve(ax, hybrid[seed], metric, color, marker, upper, purellm=False, overview=overview)
                _plot_curve(ax, purellm[seed], metric, color, marker, upper, purellm=True, overview=overview)
            if overview and stage_end < LAST:
                ax.axvspan(FIRST, stage_end, facecolor="#EEF4FA", alpha=0.72, zorder=0)
                ax.axvline(stage_end, color="#A8B8C8", alpha=0.67, linewidth=0.8, zorder=1)
            ax.set_xlim(FIRST - BATCH * 0.25, upper + BATCH * 0.25)
            ax.xaxis.set_major_locator(FixedLocator(full_ticks if overview else stage_ticks))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.margins(y=0.10)
            ax.grid(axis="y", color=GRID, linewidth=0.75)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", length=0, pad=6, labelsize=9.5)
            for position in ("top", "right", "left"):
                ax.spines[position].set_visible(False)
            if metric == len(METRICS) - 1:
                ax.set_xlabel("Labeled experiments", fontsize=10.5, labelpad=9)

    endpoints = "    ".join(f"{seed}  L{end_by_seed[seed]}" for seed in SEEDS)
    fig.text(0.106, 0.057, f"Last verified PureLLM checkpoints   {endpoints}", fontsize=10, weight="semibold")
    fig.text(0.106, 0.034, "Lines connect observed validation checkpoints. Hybrid has all 37 checkpoints per seed.",
             color=MUTED, fontsize=8.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=210, metadata={"Title": "PureLLM-A7/M2 vs Hybrid stage validation"})
    plt.close(fig)


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--purellm", type=Path, default=directory / "purellm_validation_curves.csv")
    parser.add_argument("--hybrid", type=Path, default=directory / "hybrid_validation_curves.csv")
    parser.add_argument("--output", type=Path, default=directory / "purellm_vs_hybrid_stage.png")
    parser.add_argument("--snapshot", help="Optional snapshot date or timestamp in the figure subtitle")
    args = parser.parse_args()
    render(read_curves(args.purellm, complete=False), read_curves(args.hybrid, complete=True),
           args.output, snapshot=args.snapshot)


if __name__ == "__main__":
    main()
