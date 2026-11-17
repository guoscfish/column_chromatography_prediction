#!/usr/bin/env python3
"""Plot saved validation curves for NoCW, both Free runs and three controls."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (157, 6101)
INPUTS = (
    ROOT / "reports/five_methods_cw_gradient_20261103/validation_curves.csv",
    ROOT / "reports/free_cw_final_20261112/final_curves_20261112/validation_curves.csv",
)
METHODS = (
    ("nocw", "nocwfreellm", "#D95F02", "o", "-"),
    ("new_free_cw_sol61_xhigh", "freellm（新版 CW 重跑）", "#007C91", "o", "-"),
    ("historical_free_mixed_model", "freellm（历史版）", "#9958AD", "s", "--"),
    ("free16_lcmd16", "lcmd16+freellm16", "#C99A00", "^", "-"),
    ("cw_gradient_lcmd32", "lcmd32（CW-gradient）", "#2864B4", "D", "-"),
    ("numerical_hybrid32", "hybrid", "#408849", "v", "-"),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(out):
    if out.exists():
        raise RuntimeError("Choose a new output directory to preserve saved figures")
    wanted = {method for method, *_ in METHODS}
    points, source_hashes, identities = {}, {}, {}
    for source in INPUTS:
        source_hashes[str(source.relative_to(ROOT))] = sha(source)
        with source.open(newline="") as handle:
            for row in csv.DictReader(handle):
                method, seed = row["method"], int(row["seed"])
                if method not in wanted or seed not in SEEDS:
                    continue
                budget, value = int(row["budget"]), float(row["validation_nrmse"])
                audit_name = row.get("fit_audit") or row["source"]
                audit_path = ROOT / audit_name
                digest = sha(audit_path)
                if digest != row["fit_audit_sha256"]:
                    raise RuntimeError(f"Saved fit audit changed: {audit_name}")
                audit = json.loads(audit_path.read_text())
                assert audit["train_rows"] == budget
                assert audit["test_labels_used_for_fit_or_checkpoint_selection"] == 0
                assert math.isfinite(value) and value >= 0
                assert abs(value - audit["best_validation_combined_normalized_rmse"]) < 1e-12
                identity = (audit["initialization_hash"], audit["validation_ids_hash"])
                identities.setdefault(seed, identity)
                assert identities[seed] == identity, (method, seed, budget)
                source_hashes[audit_name] = digest
                point = {"method": method, "seed": seed, "budget": budget,
                         "validation_nrmse": value, "fit_audit": audit_name,
                         "fit_audit_sha256": digest}
                key = method, seed, budget
                if key in points:
                    assert points[key]["validation_nrmse"] == value
                    assert points[key]["fit_audit_sha256"] == digest
                points[key] = point

    curves, means = {}, []
    for method, *_ in METHODS:
        per_seed = {}
        for seed in SEEDS:
            group = sorted((p for p in points.values()
                            if p["method"] == method and p["seed"] == seed),
                           key=lambda p: p["budget"])
            assert group and [p["budget"] for p in group] == list(range(333, group[-1]["budget"] + 1, 32))
            per_seed[seed] = group
        end = min(group[-1]["budget"] for group in per_seed.values())
        group = []
        for budget in range(333, end + 1, 32):
            values = [points[method, seed, budget]["validation_nrmse"] for seed in SEEDS]
            group.append({"method": method, "budget": budget, "n": len(SEEDS),
                          "mean": statistics.mean(values),
                          "sample_sd": statistics.stdev(values)})
        curves[method] = group
        means.extend(group)
    for seed in SEEDS:
        assert len({points[method, seed, 333]["validation_nrmse"] for method in wanted}) == 1
    common = min(group[-1]["budget"] for group in curves.values())

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/six_methods_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    chinese = next((name for name in ("Arial Unicode MS", "PingFang SC", "Heiti TC")
                    if name in available), None)
    if chinese is None:
        raise RuntimeError("A Chinese font is required for the requested labels")
    plt.rcParams.update({"font.family": chinese, "font.size": 11,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    out.mkdir(parents=True)

    def draw(name, cap=None):
        fig, ax = plt.subplots(figsize=(13.5, 8.1))
        fig.patch.set_facecolor("white")
        fig.text(.085, .951, "主动学习曲线对比", fontsize=23, weight="bold", color="#172B42")
        fig.text(.085, .906, "共同 seed：157、6101  |  各点为两个 seed 的均值  |  验证集 NRMSE，越低越好",
                 fontsize=11.5, color="#58677A")
        for method, label, color, marker, linestyle in METHODS:
            group = [p for p in curves[method] if cap is None or p["budget"] <= cap]
            ax.plot([p["budget"] for p in group], [p["mean"] for p in group],
                    label=label, color=color, linewidth=2.25, linestyle=linestyle,
                    marker=marker, markersize=3.7, markeredgewidth=0, markevery=2,
                    zorder=3)
            ax.scatter(group[-1]["budget"], group[-1]["mean"], s=32, color=color,
                       edgecolor="white", linewidth=.7, zorder=4)
        stop = cap or max(group[-1]["budget"] for group in curves.values())
        ax.set(xlabel="已标注训练样本数 L", ylabel="验证集 NRMSE（越低越好）",
               xlim=(320, stop + 20), ylim=(.28, 1.05))
        ax.set_xticks([v for v in (333, 525, 653, 813, 1005, 1069, 1197, 1389, 1485) if v <= stop])
        ax.grid(axis="y", color="#E2E8EF", linewidth=.75)
        ax.tick_params(axis="both", length=3, color="#A9B4C0")
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#E5EAF0",
                  framealpha=.95, fontsize=10.8, borderpad=.85, labelspacing=.65)
        if cap is None:
            ax.axvline(common, color="#9BA7B4", linewidth=1, linestyle=":", zorder=1)
            ax.text(common + 10, .02, f"六条曲线的共同预算上限 L{common}",
                    transform=ax.get_xaxis_transform(), fontsize=9, color="#6B7889", va="bottom")
        note = ("各曲线保持相同两个 seed，画至二者共同完成的预算；保留原始波动，未平滑或外推。\n"
                "LCMD 均为 CW-gradient；历史 freellm、lcmd16+freellm16 保留历史混合模型轨迹。")
        if cap is not None:
            note = f"本图仅显示六条曲线均已完成的共同区间 L333–L{cap}。\n" + note.split("\n")[1]
        fig.text(.085, .055, note, fontsize=9.7, color="#58677A", linespacing=1.7)
        fig.subplots_adjust(left=.085, right=.978, top=.85, bottom=.17)
        for extension in ("png", "svg", "pdf"):
            fig.savefig(out / f"{name}.{extension}", dpi=220, facecolor="white")
        plt.close(fig)

    draw("six_methods")
    draw("six_methods_common_budget", common)
    rows = sorted(points.values(), key=lambda p: (p["method"], p["seed"], p["budget"]))
    write_csv(out / "validation_curves.csv", rows)
    write_csv(out / "mean_curves.csv", means)
    snapshot = {
        "captured_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "seeds": list(SEEDS), "common_budget": common,
        "methods": [{"method": method, "label": label,
                     "common_completed_budget": curves[method][-1]["budget"]}
                    for method, label, *_ in METHODS],
        "metric": "best_validation_combined_normalized_rmse",
        "lcmd_geometry": "center_width_gradient", "display": "fixed two-seed mean; no bands",
        "raw_nonmonotone_curves": True, "smoothing": False, "extrapolation": False,
        "new_model_calls": 0, "new_fits": 0, "test_truth_reads": 0,
        "source_hashes": source_hashes,
    }
    (out / "snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    (out / "README.md").write_text(
        "# Six learning curves, including both Free versions\n\n"
        "The requested dual-expert curve is omitted. All six curves use the same two seeds, 157 and 6101. "
        "Each point is the arithmetic mean of saved validation NRMSE; lower is better. "
        "Each curve ends at the last budget completed by both seeds. "
        "No smoothing, interpolation, extrapolation or best-so-far envelope is applied.\n\n"
        "Free (new) is the completed CW-visible Sol 6.1/xhigh rerun. Free (historical) and "
        "LCMD16+Free16 retain historical mixed-model trajectories. Both LCMD controls use "
        "center/width-gradient geometry, as in the prior five-method figure. "
        "These are descriptive validation comparisons.\n\n"
        "six_methods.png/pdf/svg show all available matched-seed points. "
        f"six_methods_common_budget.png/pdf/svg show the common L333–L{common} interval. "
        "validation_curves.csv retains individual seed values and fit-audit hashes; "
        "mean_curves.csv contains means and sample standard deviations (ddof=1). "
        "snapshot.json records input hashes, method identities and endpoints. "
        "Only saved summaries and fit audits were read; no training, model calls or test-truth reads.\n\n"
        "Reproduce with the original saved audits available locally:\n\n"
        "```sh\n/Users/fish/miniforge3/envs/fish/bin/python "
        "scripts/studies/plot_six_methods_both_free_20261112.py --output reports/<fresh-directory>\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in snapshot.items() if k != "source_hashes"}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output.resolve())
