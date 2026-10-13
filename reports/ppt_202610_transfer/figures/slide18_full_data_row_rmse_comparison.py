#!/usr/bin/env python3
"""Reporting-only rendering of the completed filtered FULL/Row split results.

Training is separate: scripts/studies/run_row_full_data_rmse_completion.py.
This script loads only frozen aggregate metrics, never models or test labels.
Colors, layout, fonts, method order and axis limits match the B=100 figure.
The paper-style arm is the project implementation, not original-paper results.
Zero-shot uses no target training labels in either comparison.

Install: python -m pip install matplotlib==3.9.4 pillow==11.3.0
Run from any directory; use --font /path/to/CJK/font.ttf if necessary.
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
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent / "frozen_inputs/studies/transfer/row_full_data_rmse_completion_20261013"
HASHES = {}  # Populated from the frozen completion audit after input validation.
COLUMNS = ("8g", "25g", "40g")
METHODS = ("zero_shot", "scale_only", "affine", "conditional_EA", "paper_style_current_v2")
LABELS = ("直接\n预测", "比例\n校准", "线性\n校准", "洗脱条件\n校准", "论文风格\n神经迁移")
FOOTER = "Row split｜Full data（阈值过滤）｜5 seeds 均值｜RMSE（mL，越低越好）"
COLORS = ("#8095A3", "#8C0000")
LIMITS = {"8g": (25, 5), "25g": (80, 20), "40g": (150, 30)}


def load_values():
    """Validate the frozen completion summary and five fixed seeds."""
    audit = json.loads((INPUTS / "completion_audit.json").read_text())
    protocol = json.loads((INPUTS / "protocol.json").read_text())
    freeze = json.loads((INPUTS / "all_predictions_frozen.json").read_text())
    actual = hashlib.sha256((INPUTS / "summary.csv").read_bytes()).hexdigest()
    if actual != audit["summary_sha256"]:
        raise ValueError("Frozen full-data CSV hash mismatch")
    assert hashlib.sha256((INPUTS / "protocol.json").read_bytes()).hexdigest() == freeze["protocol_sha256"]
    assert hashlib.sha256((INPUTS / "all_predictions_frozen.json").read_bytes()).hexdigest() == audit["global_freeze_sha256"]
    assert protocol["budget"] == "FULL" and protocol["protocols"] == ["row"]
    assert protocol["columns"] == list(COLUMNS) and protocol["methods"] == list(METHODS)
    assert len(set(protocol["outer_seeds"])) == 5
    assert protocol["test_labels_used_for_fit_or_selection"] is False
    assert audit["test_evaluation_after_global_freeze"] and audit["reused_rmse_matches_parent"]
    assert protocol["thresholds_ml"] == {
        "8g": {"V1": 60.0, "V2": 120.0}, "25g": {"V1": 60.0, "V2": 120.0},
        "40g": {"V1": 150.0, "V2": 200.0},
    }
    HASHES["summary.csv"] = actual
    with (INPUTS / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15 and {r["protocol"] for r in rows} == {"row"}
    values = {}
    for row in rows:
        key = (row["column"], row["method"])
        assert key not in values and int(row["n_seeds"]) == 5
        means = tuple(float(row[f"{target}_rmse_mean"]) for target in ("V1", "V2"))
        assert all(math.isfinite(v) and v >= 0 for v in means)
        values[key] = means
    assert set(values) == {(c, m) for c in COLUMNS for m in METHODS}
    return values


def choose_font(explicit):
    required = set("".join(LABELS) + FOOTER + "独立纵轴尺度") - {"\n"}
    candidates = ([Path(explicit)] if explicit else []) + [
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
    ]
    candidates += [Path(f.fname) for f in font_manager.fontManager.ttflist
                   if any(n in f.name for n in ("Noto Sans CJK", "Source Han Sans", "Microsoft YaHei"))]
    for path in candidates:
        if path.exists() and all(ord(c) in FT2Font(str(path)).get_charmap() for c in required):
            font_manager.fontManager.addfont(str(path))
            return font_manager.FontProperties(fname=str(path)).get_name()
    raise RuntimeError("No font covers all Chinese glyphs; supply --font with a CJK font.")


def check_layout(fig, axes, bar_labels, values):
    """Check bar data, labels, linear zero baselines, overlap and canvas clipping."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    all_text = list(fig.texts) + list(fig.legends[0].get_texts())
    for column, ax, labels in zip(COLUMNS, axes, bar_labels):
        assert ax.get_yscale() == "linear" and ax.get_ylim()[0] == 0
        assert len(ax.patches) == 10 and len(labels) == 10
        expected = [values[(column, m)][t] for t in range(2) for m in METHODS]
        for patch, label, value in zip(ax.patches, labels, expected):
            assert patch.get_y() == 0 and patch.get_height() == value
            assert label.get_text() == f"{value:.1f}"
            assert label.get_window_extent(renderer).y1 < ax.bbox.y1
        label_boxes = [t.get_window_extent(renderer) for t in labels]
        assert not any(a.overlaps(b) for i, a in enumerate(label_boxes) for b in label_boxes[i + 1:])
        ticks = ax.get_xticklabels()
        tick_boxes = [t.get_window_extent(renderer) for t in ticks]
        assert not any(a.overlaps(b) for i, a in enumerate(tick_boxes) for b in tick_boxes[i + 1:])
        all_text += labels + ticks + ax.get_yticklabels() + [ax.yaxis.label, ax.title]
        all_text += list(ax.texts)
    for artist in all_text:
        box = artist.get_window_extent(renderer)
        assert box.x0 >= canvas.x0 and box.y0 >= canvas.y0, artist.get_text()
        assert box.x1 <= canvas.x1 and box.y1 <= canvas.y1, artist.get_text()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", help="Path to a font with complete CJK coverage")
    args = parser.parse_args()
    values = load_values()
    plt.rcParams.update({
        "font.family": choose_font(args.font), "font.size": 20,
        "axes.unicode_minus": False, "svg.fonttype": "path",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.linewidth": 0.8,
    })
    warnings.filterwarnings("error", message="Glyph .* missing from font")
    fig, axes = plt.subplots(1, 3, figsize=(21, 10), dpi=150, sharey=False)
    fig.subplots_adjust(left=0.060, right=0.984, bottom=0.230, top=0.800, wspace=0.245)
    fig.legend(handles=[Patch(facecolor=c, label=f"V{i + 1} RMSE") for i, c in enumerate(COLORS)],
               loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2, frameon=False,
               fontsize=22, handlelength=1.4, handleheight=0.8, columnspacing=2.5)
    bar_labels = []
    for column, ax in zip(COLUMNS, axes):
        upper, step = LIMITS[column]
        ax.set_ylim(0, upper)
        ax.set_yticks(range(0, upper + 1, step))
        ax.set_xlim(-0.58, 4.58)
        ax.set_ylabel("RMSE (mL)", fontsize=20, labelpad=12)
        ax.set_title(column, fontsize=30, fontweight="bold", pad=43)
        ax.text(0.5, 1.045, "独立纵轴尺度", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=18, color="#666666")
        ax.set_xticks(range(5), LABELS, fontsize=18)
        ax.tick_params(axis="x", length=0, pad=15)
        ax.tick_params(axis="y", length=0, pad=8, labelsize=19, colors="#4D4D4D")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#A0A0A0")
        labels = []
        for target, color, offset in zip(range(2), COLORS, (-0.24, 0.24)):
            heights = [values[(column, method)][target] for method in METHODS]
            assert max(heights) < upper * 0.90
            bars = ax.bar([x + offset for x in range(5)], heights, width=0.42,
                          color=color, edgecolor="none")
            labels += ax.bar_label(bars, labels=[f"{v:.1f}" for v in heights],
                                   padding=7, fontsize=18, color="#303030")
        bar_labels.append(labels)
    fig.text(0.5, 0.060, FOOTER, ha="center", va="center", fontsize=20, color="#595959")
    check_layout(fig, axes, bar_labels, values)
    stem = HERE / "slide18_full_data_row_rmse_comparison"
    description = ("Row split; filtered FULL data; arithmetic means across 5 fixed seeds; "
                   "V1/V2 test RMSE in mL; zero-shot uses no target training labels; "
                   "paper_style_current_v2 is project implementation, not original-paper results. "
                   "Thresholds: 8g/25g V1<=60, V2<=120 mL; 40g V1<=150, V2<=200 mL. "
                   "FULL uses all filtered gradient-train rows, excludes validation/test. "
                   "Independent linear zero-based y axes match B100: 8g 0–25, 25g 0–80, 40g 0–150 mL. "
                   "Inherited tests have been examined previously: descriptive completion. "
                   "Source main: 42ab12600a353bba4a063ef5d1b623ee2f7b976a.")
    fig.savefig(stem.with_suffix(".png"), dpi=300,
                metadata={"Description": description, "Source CSV SHA256": HASHES["summary.csv"]})
    fig.savefig(stem.with_suffix(".svg"),
                metadata={"Title": "Row split: filtered Full data RMSE comparison", "Description": description,
                          "Date": "2026-10-13"})
    plt.close(fig)
    from PIL import Image
    with Image.open(stem.with_suffix(".png")) as png:
        assert png.size == (6300, 3000)
        assert min(png.info["dpi"]) >= 299.9
    for column in COLUMNS:
        print(column, " | ".join(f"{m}: {values[(column, m)][0]:.1f}/{values[(column, m)][1]:.1f}"
                                 for m in METHODS))
    print("Verified: 30 CSV-derived labels; Row split only; zero-based independent linear axes; "
          "no label overlaps or clipping; PNG 6300×3000 at 300 dpi; SVG glyphs embedded as paths.")


if __name__ == "__main__":
    main()
