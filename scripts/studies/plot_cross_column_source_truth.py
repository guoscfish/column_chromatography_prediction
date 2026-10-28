#!/usr/bin/env python3
"""Redraw retained transfer predictions without rerunning models or scoring."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies/transfer/cross_column"
COLUMNS = ("8g", "25g", "40g")
DEFAULT_SEED = 769539383


def plot_source_truth(
    predictions: pd.DataFrame,
    output_dir: Path,
    *,
    protocol: str = "row",
    seed: int = DEFAULT_SEED,
    budget: int = 100,
) -> list[dict]:
    """Use separate data limits; preserve the original descriptive OLS lines.

    The full view includes every held-out point. The companion zoom uses the
    marginal 90th percentiles solely as display limits and labels omitted points.
    Neither view changes predictions, fits a transfer model, or evaluates it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    for column in COLUMNS:
        subset = predictions.loc[
            predictions.column.eq(column)
            & predictions.protocol.eq(protocol)
            & predictions.seed.eq(seed)
            & predictions.planned_budget.eq(budget)
        ]
        for target in ("V1", "V2"):
            x = subset[f"zero_shot_{target}"].to_numpy(dtype=float)
            y = subset[f"{target}_true"].to_numpy(dtype=float)
            if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f"{column} {target}: expected nonempty finite plotting data")
            if np.unique(x).size < 2 or np.square(x).sum() == 0:
                raise ValueError(f"{column} {target}: degenerate source predictions")
            panels.append({
                "column": column, "target": target, "x": x, "y": y,
                "scale": float((x * y).sum() / np.square(x).sum()),
                "affine": np.polyfit(x, y, 1),
            })

    audit = []
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 14, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    }):
        for zoom in (False, True):
            fig, axes = plt.subplots(3, 2, figsize=(11.5, 12.5))
            fig.subplots_adjust(left=.09, right=.98, bottom=.105, top=.89,
                                hspace=.36, wspace=.23)
            title = "Source-to-target transfer: central-range zoom" if zoom else "Source-to-target transfer: full data range"
            fig.suptitle(title, y=.975, fontsize=18, fontweight="semibold")
            fig.text(.5, .946, f"{protocol} split  |  seed {seed}  |  label budget {budget}",
                     ha="center", color="#555555", fontsize=11)
            legend = [
                Line2D([], [], marker="o", ls="", markersize=5, color="#394b59", label="Held-out observations"),
                Line2D([], [], ls="--", lw=1.4, color="#999999", label="y = x"),
                Line2D([], [], lw=2, color="#d08700", label="Scale-only (descriptive)"),
                Line2D([], [], lw=2, color="#087e8b", label="Affine (descriptive)"),
            ]
            fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .928),
                       ncol=4, frameon=False, fontsize=10, handlelength=2)
            for ax, panel in zip(axes.flat, panels):
                x, y = panel["x"], panel["y"]
                x_top = float(np.quantile(x, .9) if zoom else x.max())
                y_top = float(np.quantile(y, .9) if zoom else y.max())
                # Independent limits prevent target-volume units from stretching
                # the source axis and prevent extrapolated lines from autoscaling.
                x_lim = (min(0., float(x.min()) * 1.06), max(x_top * 1.06, 1.))
                y_lim = (min(0., float(y.min()) * 1.06), max(y_top * 1.06, 1.))
                ax.set_xlim(*x_lim)
                ax.set_ylim(*y_lim)
                xx = np.linspace(float(x.min()), min(float(x.max()), x_lim[1]), 250)
                ax.plot(xx, xx, "--", color="#999999", lw=1.4, zorder=1)
                ax.plot(xx, panel["scale"] * xx, color="#d08700", lw=2, zorder=2)
                slope, intercept = panel["affine"]
                ax.plot(xx, slope * xx + intercept, color="#087e8b", lw=2, zorder=2)
                ax.scatter(x, y, s=24, alpha=.62, color="#394b59",
                           edgecolors="white", linewidths=.35, zorder=3)
                visible = ((x >= x_lim[0]) & (x <= x_lim[1]) &
                           (y >= y_lim[0]) & (y <= y_lim[1]))
                if not zoom and not visible.all():
                    raise AssertionError("Full view must include every observation")
                count = int(visible.sum())
                annotation = f"n = {len(x)} (all shown)"
                if zoom:
                    annotation = f"Shown: {count}/{len(x)}\nOutside view: {len(x) - count}"
                ax.text(.035, .965, annotation, transform=ax.transAxes, va="top",
                        fontsize=10, color="#555555")
                ax.set(title=f"{panel['column']}  |  {panel['target']}",
                       xlabel="4g source q50 prediction (mL)",
                       ylabel="Target truth (mL)")
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
                ax.grid(alpha=.18, linewidth=.6)
                ax.set_axisbelow(True)
                ax.spines[["top", "right"]].set_visible(False)
                audit.append({
                    "view": "zoom" if zoom else "full", "column": panel["column"],
                    "target": panel["target"], "n": len(x), "shown": count,
                    "x_min": x_lim[0], "x_max": x_lim[1],
                    "y_min": y_lim[0], "y_max": y_lim[1],
                    "descriptive_scale": panel["scale"],
                    "descriptive_affine_slope": float(slope),
                    "descriptive_affine_intercept": float(intercept),
                })
            note = "Axes follow each panel's observed x and y ranges; all observations are included."
            if zoom:
                note = "Zoom limits: each axis's 90th percentile + 6% margin; outside-view counts are labeled."
            fig.text(.09, .047, note, fontsize=10, color="#555555")
            fig.text(.09, .027,
                     "Lines reproduce the original descriptive fits to held-out points; they are not trained calibration models.",
                     fontsize=9, color="#666666")
            stem = "source_prediction_vs_target_truth" + ("_zoom" if zoom else "")
            for extension in ("png", "pdf", "svg"):
                fig.savefig(output_dir / f"{stem}.{extension}", dpi=240,
                            facecolor="white")
            plt.close(fig)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--protocol", choices=("row", "compound"), default="row")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--budget", type=int, default=100)
    args = parser.parse_args()
    frames = []
    for column in COLUMNS:
        source = args.study / column / args.protocol / f"seed_{args.seed}" / f"budget_{args.budget}" / "predictions.csv.gz"
        frame = pd.read_csv(source)
        frame = frame.assign(column=column, protocol=args.protocol, seed=args.seed, planned_budget=args.budget)
        frames.append(frame)
    output = args.output_dir or args.study / "plots"
    audit = plot_source_truth(pd.concat(frames, ignore_index=True), output,
                              protocol=args.protocol, seed=args.seed, budget=args.budget)
    pd.DataFrame(audit).to_csv(output / "source_prediction_vs_target_truth_plot_audit.csv", index=False)
    for row in audit:
        print(f"{row['view']}: {row['column']} {row['target']} {row['shown']}/{row['n']} points; "
              f"x=[{row['x_min']:.2f}, {row['x_max']:.2f}], y=[{row['y_min']:.2f}, {row['y_max']:.2f}]")


if __name__ == "__main__":
    main()
