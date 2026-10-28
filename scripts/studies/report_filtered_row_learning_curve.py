#!/usr/bin/env python3
"""Render frozen filtered Row split B=100/200/300/FULL learning-curve means.

No fitting, checkpoints, or test labels are loaded by this plotting script.
Run from any directory with matplotlib and Pillow installed:
  python scripts/studies/report_filtered_row_learning_curve.py
Optional: --font /path/to/a/CJK/font.ttf
The original reporting runtime used matplotlib==3.9.4 and pillow==11.3.0.
Training/split reproducibility: scripts/studies/run_filtered_row_learning_curve.py
and scripts/studies/run_filtered_row_learning_curve_b200.py.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "studies/transfer/filtered_row_learning_curve_b200_20261013"
HERE = INPUTS / "figures"
COLUMNS = ("8g", "25g", "40g")
BUDGETS = ("100", "200", "300", "FULL")
METHODS = ("zero_shot", "scale_only", "affine", "conditional_EA", "paper_style_current_v2")
LABELS = ("直接预测\n（无目标训练标签）", "比例校准", "线性校准", "洗脱条件校准", "论文风格神经迁移")
STYLES = {
    "zero_shot": {"color": "#909090", "linestyle": (0, (5, 4)), "marker": None, "linewidth": 2.1},
    "scale_only": {"color": "#29485D", "linestyle": "-", "marker": "s", "linewidth": 2.3},
    "affine": {"color": "#859CB1", "linestyle": (0, (6, 3)), "marker": "D", "linewidth": 2.4},
    "conditional_EA": {"color": "#507E89", "linestyle": (0, (1, 2.2)), "marker": "^", "linewidth": 2.5},
    "paper_style_current_v2": {"color": "#8C0000", "linestyle": "-", "marker": "o", "linewidth": 2.9},
}
FOOTER = "Row split｜阈值过滤｜5 seeds 均值｜独立纵轴｜RMSE（mL，越低越好）"
NOTE = "B 包含训练与验证标签；直接预测不使用目标柱训练标签"
XLABEL = "目标柱标签预算 B"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows():
    audit = json.loads((INPUTS / "completion_audit.json").read_text())
    protocol = json.loads((INPUTS / "protocol.json").read_text())
    freeze = json.loads((INPUTS / "all_predictions_frozen.json").read_text())
    assert sha(INPUTS / "summary.csv") == audit["summary_sha256"]
    assert sha(INPUTS / "protocol.json") == audit["protocol_sha256"] == freeze["protocol_sha256"]
    assert sha(INPUTS / "all_predictions_frozen.json") == audit["global_freeze_sha256"]
    assert protocol["columns"] == list(COLUMNS) and protocol["methods"] == list(METHODS)
    assert protocol["protocols"] == ["row"] and protocol["budgets"] == [200]
    assert protocol["plotted_budgets"] == list(BUDGETS)
    assert len(set(protocol["outer_seeds"])) == 5 and not protocol["test_labels_used_for_fit_or_selection"]
    assert all(audit[name] for name in ("same_tests_all_budgets", "same_validation_all_budgets",
                                       "nested_train_prefixes", "zero_shot_identical_all_budgets",
                                       "full_results_reused_unchanged", "previous_results_reused_unchanged",
                                       "test_evaluation_after_global_freeze"))
    assert protocol["thresholds_ml"] == {"8g": {"V1": 60., "V2": 120.}, "25g": {"V1": 60., "V2": 120.},
                                          "40g": {"V1": 150., "V2": 200.}}
    with (INPUTS / "summary.csv").open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    rows = {}
    for row in raw:
        key = (row["column"], row["budget"], row["method"])
        assert key not in rows and row["protocol"] == "row" and int(row["n_seeds"]) == 5
        for field in ("actual_budget_mean", "V1_rmse_mean", "V2_rmse_mean"):
            row[field] = float(row[field])
            assert math.isfinite(row[field]) and row[field] >= 0
        rows[key] = row
    assert set(rows) == {(c, b, m) for c in COLUMNS for b in BUDGETS for m in METHODS}
    for column in COLUMNS:
        for budget in BUDGETS:
            assert len({rows[(column, budget, method)]["actual_budget_mean"] for method in METHODS}) == 1
            if budget != "FULL":
                assert rows[(column, budget, METHODS[0])]["actual_budget_mean"] == float(budget)
        assert rows[(column, "FULL", METHODS[0])]["actual_budget_mean"] > 300
    return rows, audit


def choose_font(explicit):
    required = set("".join(LABELS) + FOOTER + NOTE + XLABEL + "直接预测基线") - {"\n"}
    paths = ([Path(explicit)] if explicit else []) + [Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
             Path("/System/Library/Fonts/PingFang.ttc"),
             Path("/System/Library/Fonts/STHeiti Light.ttc"), Path("/Library/Fonts/Arial Unicode.ttf")]
    paths += [Path(f.fname) for f in font_manager.fontManager.ttflist
              if any(s in f.name for s in ("Noto Sans CJK", "Source Han Sans", "Microsoft YaHei"))]
    for path in paths:
        if path.exists() and all(ord(c) in FT2Font(str(path)).get_charmap() for c in required):
            font_manager.fontManager.addfont(str(path))
            return font_manager.FontProperties(fname=str(path)).get_name()
    raise RuntimeError("No complete CJK font found; supply --font.")


def detail_bounds(values):
    """Use a uniform 8% margin; clearly labeled local linear zoom, no jitter."""
    low, high = min(values), max(values)
    margin = max((high - low) * .08, .04)
    return max(0, low - margin), high + margin


def verify_layout(fig, axes, curves, rows, baseline_texts):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    visible_text = list(fig.texts) + list(fig.legends[0].get_texts()) + list(baseline_texts.values())
    assert len(fig.axes) == 6 and len(curves) == 24
    for row_index, target in enumerate(("V1", "V2")):
        for column_index, column in enumerate(COLUMNS):
            ax = axes[row_index, column_index]
            assert ax.get_yscale() == ax.get_xscale() == "linear"
            for method in METHODS[1:]:
                line = curves[(target, column, method)]
                xs = [rows[(column, budget, method)]["actual_budget_mean"] for budget in BUDGETS]
                ys = [rows[(column, budget, method)][f"{target}_rmse_mean"] for budget in BUDGETS]
                assert list(line.get_xdata()) == xs and list(line.get_ydata()) == ys
                assert max(ys) < ax.get_ylim()[1] and min(ys) > ax.get_ylim()[0]
                assert xs[0:3] == [100, 200, 300] and xs[3] > 300
            baseline_values = [rows[(column, b, "zero_shot")][f"{target}_rmse_mean"] for b in BUDGETS]
            assert max(baseline_values) - min(baseline_values) < 1e-12
            label = baseline_texts[(target, column)]
            assert label.get_text() == f"直接预测基线  {baseline_values[-1]:.2f} mL"
            assert label.get_window_extent(renderer).y0 > ax.bbox.y1
            assert len(ax.lines) == 4
            tick_boxes = [t.get_window_extent(renderer) for t in ax.get_xticklabels()]
            assert not any(a.overlaps(b) for i, a in enumerate(tick_boxes) for b in tick_boxes[i + 1:]), column
            visible_text += ax.get_xticklabels() + ax.get_yticklabels() + [ax.yaxis.label]
    for artist in visible_text:
        if not artist.get_text():
            continue
        assert "局部放大" not in artist.get_text()
        box = artist.get_window_extent(renderer)
        assert box.x0 >= fig.bbox.x0 and box.y0 >= fig.bbox.y0, artist.get_text()
        assert box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1, artist.get_text()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font")
    args = parser.parse_args()
    rows, audit = load_rows()
    cjk_font = choose_font(args.font)
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", cjk_font], "font.size": 18,
        "axes.unicode_minus": False, "svg.fonttype": "path",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.linewidth": .75, "text.color": "#273640", "axes.labelcolor": "#273640",
        "lines.solid_capstyle": "round", "lines.dash_capstyle": "round",
        "lines.solid_joinstyle": "round", "text.antialiased": True,
    })
    warnings.filterwarnings("error", message="Glyph .* missing from font")
    fig, axes = plt.subplots(2, 3, figsize=(21, 12), dpi=150)
    fig.subplots_adjust(left=.065, right=.98, bottom=.183, top=.80, wspace=.29, hspace=.51)
    handles = []
    for method, label in zip(METHODS[1:], LABELS[1:]):
        style = dict(STYLES[method])
        color = style["color"]
        handles.append(Line2D([0], [0], **style, markersize=8,
                             markerfacecolor="white" if method in ("scale_only", "affine") else color,
                             markeredgecolor=color, markeredgewidth=1.4, label=label))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .985), ncol=4,
               frameon=False, fontsize=20, columnspacing=2.2, handlelength=2.6, handletextpad=.75)
    curves, baseline_texts = {}, {}
    for row_index, target in enumerate(("V1", "V2")):
        for column_index, column in enumerate(COLUMNS):
            ax = axes[row_index, column_index]
            if row_index == 0:
                box = ax.get_position()
                fig.text((box.x0 + box.x1) / 2, .881, column,
                         ha="center", va="center", fontsize=29, fontweight="semibold", color="#273640")
            actual_full = rows[(column, "FULL", "zero_shot")]["actual_budget_mean"]
            x = [100., 200., 300., actual_full]
            span = actual_full - 100
            all_values = [rows[(column, b, m)][f"{target}_rmse_mean"] for b in BUDGETS for m in METHODS[1:]]
            lower, upper = detail_bounds(all_values)
            ax.set_ylim(lower, upper)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=4))
            ax.set_xlim(100 - span * .085, actual_full + span * .12)
            ax.set_xticks(x, ["100", "200", "300", f"Full\n{actual_full:g}"], fontsize=17)
            ax.tick_params(axis="x", length=0, pad=12, colors="#66737D")
            ax.tick_params(axis="y", length=0, pad=9, labelsize=19, colors="#66737D")
            ax.set_ylabel(f"{target} RMSE (mL)", fontsize=21, labelpad=15)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E8EDF0", linewidth=.85)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color("#C7D0D6")
            baseline = rows[(column, "FULL", "zero_shot")][f"{target}_rmse_mean"]
            baseline_texts[(target, column)] = ax.text(
                1, 1.055, f"直接预测基线  {baseline:.2f} mL", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=16, color="#77838B")
            for method in METHODS[1:]:
                values = [rows[(column, b, method)][f"{target}_rmse_mean"] for b in BUDGETS]
                style = dict(STYLES[method])
                color = style["color"]
                line, = ax.plot(x, values, **style, markersize=8,
                                markerfacecolor="white" if method in ("scale_only", "affine") else color,
                                markeredgecolor=color, markeredgewidth=1.4,
                                zorder=5 if method == "paper_style_current_v2" else 3)
                curves[(target, column, method)] = line
    fig.text(.5, .112, XLABEL, ha="center", va="center", fontsize=21, color="#33434F")
    fig.text(.5, .064, FOOTER, ha="center", va="center", fontsize=17, color="#66737D")
    fig.text(.5, .030, NOTE, ha="center", va="center", fontsize=14.5, color="#8A959D")
    verify_layout(fig, axes, curves, rows, baseline_texts)
    HERE.mkdir(exist_ok=True)
    stem = HERE / "learning_curve"
    description = ("Filtered Row split learning curves, V1/V2 test RMSE means over five fixed seeds. "
                   "B=100/200/300 counts train+validation labels; FULL x is mean actual eligible label count. "
                   "Identical validation/test IDs and nested train prefixes. "
                   "8g/25g thresholds V1<=60,V2<=120; 40g V1<=150,V2<=200 mL. "
                   "Zero-shot is an identical zero-target-training-label baseline at all contexts, shown "
                   "as an annotation outside each panel; no baseline line is drawn. "
                   "The four label-using methods use independent linear RMSE axes with visible numeric ticks and "
                   "a uniform 8% margin (minimum .04 mL). No RMSE point labels, value tables, data jitter "
                   "or offsets. Distinct colors, line styles and marker shapes make curves easier to read. "
                   "Paper-style neural transfer is project implementation, not original-paper results. "
                   "All new predictions froze before evaluation; previous results reused unchanged. "
                   "Historically examined tests: descriptive learning curve.")
    fig.savefig(stem.with_suffix(".png"), dpi=300,
                metadata={"Description": description, "Source CSV SHA256": audit["summary_sha256"]})
    fig.savefig(stem.with_suffix(".svg"), metadata={"Title": "Filtered Row split learning curves: B100/B200/B300/FULL",
                                                   "Description": description, "Date": "2026-10-13"})
    plt.close(fig)
    with Image.open(stem.with_suffix(".png")) as png:
        assert png.size == (6300, 3600) and min(png.info["dpi"]) >= 299.9
    print("Verified: 96 curve points plus six constant baseline labels read from frozen CSV; six panels; "
          "actual numeric budget x; no baseline plots or strategy point labels; "
          "complete Chinese glyphs and no text clipping/overlap; PNG 6300×3600 at 300 dpi; embedded SVG fonts.")


if __name__ == "__main__":
    main()
