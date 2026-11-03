#!/usr/bin/env python3
"""Plot five saved validation trajectories, with CW-gradient LCMD, on shared seeds."""
import argparse
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (157, 6101)
METHODS = (
    ("nocw", "NoCW", "#087E8B"),
    ("historical_free_mixed_model", "Original Free32*", "#9A4DAD"),
    ("free16_lcmd16", "Free16 + LCMD16*", "#DD7632"),
    ("cw_gradient_lcmd32", "LCMD32 (CW-gradient)", "#2873B8"),
    ("numerical_hybrid32", "Hybrid32 (numerical)", "#428A53"),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw(out):
    if out.exists():
        raise RuntimeError("Choose a new output directory to preserve snapshots")
    spec = importlib.util.spec_from_file_location(
        "saved_curve_locations", ROOT / "scripts/studies/report_free_cw_sol61_five_20261101.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    rows, sources, curves = [], {}, {}
    for seed in SEEDS:
        baseline = None
        for method, _, _ in METHODS:
            points = []
            for rnd in range(37):
                if method == "free16_lcmd16":
                    owner = report.FREV / next(name for end, name in report.OLD if rnd <= end)
                    freeze_path = owner / f"runtime/seed_{seed}/cw16_llm16_scientist_v2/round_{rnd:02d}/prediction_freeze.json"
                    if not freeze_path.exists():
                        break
                    freeze = json.loads(freeze_path.read_text())
                    model = Path(freeze["checkpoint_path"]).parent
                    sources[str(freeze_path.relative_to(ROOT))] = sha(freeze_path)
                elif method == "cw_gradient_lcmd32" and rnd > 0:
                    stages = ((3, "qgeognn_v2_row_short_sequential_b32"),
                              (6, "qgeognn_v2_row_cw_lcmd_to_525"),
                              (10, "qgeognn_v2_row_cw_lcmd_to_653"),
                              (21, "qgeognn_v2_row_cw_lcmd_to_1005"))
                    if rnd > stages[-1][0]:
                        break
                    owner = report.BASE.parent / next(name for end, name in stages if rnd <= end)
                    directory = owner / f"runtime/seed_{seed}/center_width_lcmd/round_{rnd:02d}"
                    freeze_path = directory / "round_freeze.json"
                    if not freeze_path.exists():
                        break
                    freeze = json.loads(freeze_path.read_text())
                    model = directory / "model"
                    sources[str(freeze_path.relative_to(ROOT))] = sha(freeze_path)
                else:
                    located = report.model_directory(
                        "gradient_lcmd32" if method == "cw_gradient_lcmd32" else method, seed, rnd)
                    if located is None:
                        break
                    model, freeze = located
                path = model / "fit_audit.json"
                audit = json.loads(path.read_text())
                budget = 333 + 32 * rnd
                assert audit["train_rows"] == budget, path
                assert audit["test_labels_used_for_fit_or_checkpoint_selection"] == 0, path
                identity = (audit["initialization_hash"], audit["validation_ids_hash"])
                if baseline is None:
                    baseline = identity
                assert identity == baseline, (method, seed, budget)
                if freeze:
                    assert freeze["test_truth_access_count"] == 0, path
                    for key in ("checkpoint_sha256", "prediction_sha256", "initialization_hash"):
                        if key != "initialization_hash" or key in freeze:
                            assert freeze[key] == audit[key], path
                value = audit["best_validation_combined_normalized_rmse"]
                assert math.isfinite(value) and value >= 0, path
                sources[str(path.relative_to(ROOT))] = sha(path)
                point = {"method": method, "seed": seed, "budget": budget,
                         "validation_nrmse": value, "fit_audit": str(path.relative_to(ROOT)),
                         "fit_audit_sha256": sha(path)}
                points.append(point)
                rows.append(point)
            assert len(points) > 1, (method, seed)
            curves[method, seed] = points

    out.mkdir(parents=True)
    with (out / "validation_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    common = min(points[-1]["budget"] for points in curves.values())
    means, stats = {}, []
    for method, _, _ in METHODS:
        end = min(curves[method, seed][-1]["budget"] for seed in SEEDS)
        aggregated = []
        for budget in range(333, end + 1, 32):
            values = [next(p["validation_nrmse"] for p in curves[method, seed]
                           if p["budget"] == budget) for seed in SEEDS]
            aggregated.append({"method": method, "budget": budget, "n": len(values),
                               "mean": statistics.mean(values), "sample_sd": statistics.stdev(values)})
        means[method] = aggregated
        stats.extend(aggregated)
    with (out / "mean_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    chinese = next((name for name in ("Arial Unicode MS", "PingFang SC", "Heiti TC")
                    if name in available), None)
    plt.rcParams.update({"font.family": chinese or "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(13, 7.7))
    fig.patch.set_facecolor("#FAFBFD")
    ax.set_facecolor("white")
    ax.axvspan(333, common, color="#E9EEF4", alpha=.32, zorder=0)
    ax.axvline(common, color="#8390A0", linestyle=(0, (4, 4)), linewidth=1.2)
    for method, label, color in METHODS:
        group = means[method]
        x = [p["budget"] for p in group]
        y = [p["mean"] for p in group]
        sd = [p["sample_sd"] for p in group]
        ax.fill_between(x, [v-s for v, s in zip(y, sd)],
                        [v+s for v, s in zip(y, sd)], color=color, alpha=.08, linewidth=0)
        ax.plot(x, y, color=color, label=f"{label}  (L{x[-1]})", linewidth=2.4,
                marker="o", markersize=3.3, markeredgewidth=0, zorder=3)
        ax.scatter(x[-1], y[-1], s=32, color=color, edgecolors="white", linewidths=.7, zorder=4)
    title = "五种选点方法的学习曲线（CW-gradient LCMD）" if chinese else "Five-method comparison: CW-gradient LCMD"
    xlabel = "已标注训练样本数 L" if chinese else "Labeled training samples (L)"
    ylabel = "验证集 NRMSE（越低越好）" if chinese else "Validation NRMSE (lower is better)"
    fig.text(.085, .955, title, fontsize=22, weight="bold", color="#182436")
    subtitle = ("共同 seed：157、6101  ·  实线：均值  ·  阴影：±1 个样本标准差（n=2）"
                if chinese else "Shared seeds: 157 and 6101 | Mean with ±1 sample SD (n=2)")
    fig.text(.085, .912, subtitle, fontsize=11.5, color="#536174")
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(325, 1505))
    ax.set_xticks([333, 525, 813, 1005, 1197, 1389, 1485])
    ax.grid(axis="y", color="#DEE4EC", linewidth=.7)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E3E8EF", fontsize=10,
              loc="upper right", ncol=1)
    ax.text(common + 8, .025, f"共同预算 L{common}" if chinese else f"Shared budget L{common}",
            transform=ax.get_xaxis_transform(), color="#667586", fontsize=9, va="bottom")
    note = ("各方法画到两个共同 seed 均已完成的预算；LCMD32 与 Free16＋LCMD16 中的 LCMD 均为 CW-gradient。\n"
            "* 原始 Free 与 Free16＋LCMD16 保留历史混合模型轨迹。使用原始非单调曲线，未平滑、未外推。"
            if chinese else "Each method stops at the budget completed by both seeds. Both LCMD arms use CW-gradient.\n"
            "* Historical mixed-model trajectories retained. Raw nonmonotone curves; no smoothing or extrapolation.")
    fig.text(.085, .052, note, fontsize=9.2, color="#536174", linespacing=1.6)
    fig.subplots_adjust(left=.085, right=.975, bottom=.17, top=.865)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"five_methods.{ext}", dpi=210, facecolor=fig.get_facecolor())
    plt.close(fig)
    snapshot = {"captured_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "seeds": list(SEEDS), "common_budget": common,
                "lcmd_geometry": "center_width_gradient",
                "per_method_common_endpoint": {m: means[m][-1]["budget"] for m, _, _ in METHODS},
                "per_seed_endpoint": {m: {str(s): curves[m, s][-1]["budget"] for s in SEEDS}
                                      for m, _, _ in METHODS},
                "metric": "best_validation_combined_normalized_rmse",
                "band": "sample standard deviation, ddof=1", "source_hashes": sources,
                "test_truth_reads": 0, "new_model_calls": 0, "new_fits": 0}
    (out / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    print(json.dumps({k: v for k, v in snapshot.items() if k != "source_hashes"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw(args.output.resolve())
