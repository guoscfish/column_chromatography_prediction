#!/usr/bin/env python3
"""Compare two saved Free32 runs without model calls, fitting, or truth reads."""
import argparse
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (157, 887, 2357, 6101, 12203)
RUNS = (("first", "historical_free_mixed_model"), ("second", "new_free_cw_sol61_xhigh"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw(out):
    if out.exists():
        raise RuntimeError("Use a new output directory to preserve snapshots")
    spec = importlib.util.spec_from_file_location(
        "saved_free_curves", ROOT / "scripts/studies/report_free_cw_sol61_five_20261101.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    captured = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    rows, curves, endpoint, hashes = [], {}, {}, {}
    for seed in SEEDS:
        markers = sorted((report.STUDY / "progress").glob(f"completed_seed_{seed}_round_*.json"))
        records = [json.loads(f.read_text()) for f in markers]
        assert [r["round"] for r in records] == list(range(len(records)))
        for path, receipt in zip(markers, records):
            hashes[str(path.relative_to(ROOT))] = sha(path)
            for name, digest in receipt["files"].items():
                assert sha(report.STUDY / name) == digest, name
        endpoint[seed] = 333 + 32 * len(records)
        identity = None
        for run, method in RUNS:
            points = []
            for rnd in range(len(records) + 1):
                point = report.saved_point(method, seed, rnd)
                assert point is not None, (run, seed, rnd)
                path = ROOT / point["source"]
                audit = json.loads(path.read_text())
                hashes[str(path.relative_to(ROOT))] = sha(path)
                current_identity = (audit["initialization_hash"], audit["validation_ids_hash"])
                if identity is None:
                    identity = current_identity
                assert identity == current_identity, (run, seed, rnd)
                value = point["validation_selection_score"]
                assert abs(value - audit["best_validation_combined_normalized_rmse"]) < 1e-12
                item = {"run": run, "seed": seed, "budget": point["budget"],
                        "validation_nrmse": value, "source": point["source"],
                        "fit_audit_sha256": sha(path), "checkpoint_sha256": point["checkpoint_sha256"],
                        "history_sha256": point["history_sha256"]}
                points.append(item)
                rows.append(item)
            curves[run, seed] = points
    common = min(endpoint.values())
    shared = {key: [p for p in value if p["budget"] <= common] for key, value in curves.items()}
    paired = []
    for seed in SEEDS:
        values = {}
        for run, _ in RUNS:
            p = shared[run, seed]
            area = sum((a["validation_nrmse"] + b["validation_nrmse"]) * .5 *
                       (b["budget"] - a["budget"]) for a, b in zip(p, p[1:])) / (common - 333)
            values[run] = {"endpoint_nrmse": p[-1]["validation_nrmse"], "raw_normalized_aulc": area}
        for metric in ("endpoint_nrmse", "raw_normalized_aulc"):
            paired.append({"seed": seed, "budget": common, "metric": metric,
                           "first": values["first"][metric], "second": values["second"][metric],
                           "second_minus_first": values["second"][metric] - values["first"][metric]})
    summary = {}
    for metric in ("endpoint_nrmse", "raw_normalized_aulc"):
        group = [p for p in paired if p["metric"] == metric]
        summary[metric] = {field: {"mean": statistics.mean(p[field] for p in group),
                                 "sample_sd": statistics.stdev(p[field] for p in group)}
                           for field in ("first", "second", "second_minus_first")}
        summary[metric]["relative_change_percent"] = 100 * (
            summary[metric]["second"]["mean"] / summary[metric]["first"]["mean"] - 1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib import font_manager
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    font = next((n for n in ("Arial Unicode MS", "PingFang SC", "Heiti TC") if n in fonts), "DejaVu Sans")
    plt.rcParams.update({"font.family": font, "font.size": 10, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    colors = {"first": "#9364A7", "second": "#087F8C"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.1))
    fig.patch.set_facecolor("#FAFBFD")
    fig.text(.065, .957, "Free LLM32：第二次 vs 第一次", fontsize=23, weight="bold", color="#182436")
    fig.text(.065, .921, f"五个相同 seed · 原始验证集 NRMSE（越低越好）· 快照 {captured[:16].replace('T', ' ')} 上海时间",
             fontsize=11, color="#536174")
    handles = [Line2D([0], [0], color=colors["first"], lw=2.3, ls="--", label="第一次 Free32（历史混合模型）"),
               Line2D([0], [0], color=colors["second"], lw=2.3, label="第二次 CW-visible Free32（Sol/xhigh）")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.058, .905), frameon=False, ncol=2, fontsize=11)
    ax = axes.flat[0]
    aggregates = []
    for run, _ in RUNS:
        x = list(range(333, common + 1, 32))
        y = [statistics.mean(next(p["validation_nrmse"] for p in shared[run, s] if p["budget"] == b)
                             for s in SEEDS) for b in x]
        sd = [statistics.stdev(next(p["validation_nrmse"] for p in shared[run, s] if p["budget"] == b)
                              for s in SEEDS) for b in x]
        ax.fill_between(x, [v-s for v, s in zip(y, sd)], [v+s for v, s in zip(y, sd)],
                        color=colors[run], alpha=.10, linewidth=0)
        ax.plot(x, y, color=colors[run], lw=2.4, ls="--" if run == "first" else "-",
                marker="o", ms=3, markeredgewidth=0)
        aggregates += [{"run": run, "budget": b, "n": 5, "mean": v, "sample_sd": s}
                       for b, v, s in zip(x, y, sd)]
    ax.set_title(f"五 seed 均值 ± 1 SD · 共同到 L{common}", loc="left", fontsize=12, pad=12)
    for ax, seed in zip(list(axes.flat)[1:], SEEDS):
        for run, _ in RUNS:
            points = curves[run, seed]
            ax.plot([p["budget"] for p in points], [p["validation_nrmse"] for p in points],
                    color=colors[run], lw=2.1, ls="--" if run == "first" else "-",
                    marker="o", ms=3, markeredgewidth=0)
        ax.set_title(f"Seed {seed} · 相同预算到 L{endpoint[seed]}", loc="left", fontsize=12, pad=12)
    for ax in axes.flat:
        ax.set_facecolor("white")
        ax.grid(axis="y", alpha=.22)
        ax.set_xlabel("已标注训练样本数 L")
        ax.set_ylabel("验证集 NRMSE")
        right = max(int(v) for v in ax.get_lines()[0].get_xdata())
        ticks = [333] + [b for b in (525, 653, 813, 1005) if 333 < b and right - b >= 64] + [right]
        ax.set_xticks(ticks)
        ax.set_xlim(325, right + 10)
    rmse, area = summary["endpoint_nrmse"], summary["raw_normalized_aulc"]
    fig.text(.065, .082,
             f"共同 L{common}：均值 NRMSE 第一次 {rmse['first']['mean']:.4f} → 第二次 {rmse['second']['mean']:.4f}"
             f"（{rmse['relative_change_percent']:+.1f}%）；原始曲线 AULC {area['first']['mean']:.4f} → {area['second']['mean']:.4f}"
             f"（{area['relative_change_percent']:+.1f}%）。", fontsize=11, color="#253449")
    fig.text(.065, .031,
             "各 seed 只画两次运行都已完成的预算；未完成训练不计入。曲线未平滑、未取历史最优。\n"
             "第一次 seed157／6101 含 Astra/high 前缀（至 L525／L653）；该对照不能单独识别 CW 的因果效应。",
             fontsize=9.3, color="#536174", linespacing=1.6)
    fig.subplots_adjust(left=.065, right=.98, top=.825, bottom=.17, hspace=.42, wspace=.25)
    out.mkdir(parents=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"free_first_vs_second.{ext}", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    write_csv(out / "validation_curves.csv", rows)
    write_csv(out / "five_seed_means.csv", aggregates)
    write_csv(out / "paired_differences.csv", paired)
    result = {"captured_shanghai": captured, "seeds": list(SEEDS), "common_budget": common,
              "per_seed_endpoint": endpoint, "summary": summary, "source_hashes": hashes,
              "band": "sample standard deviation, ddof=1", "test_truth_reads": 0,
              "new_model_calls": 0, "new_fits": 0, "selector_feedback": False}
    (out / "snapshot.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_hashes"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    draw(parser.parse_args().output.resolve())
