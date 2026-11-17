#!/usr/bin/env python3
"""Audit and plot a fully paired saved-validation milestone, without new fits."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies/active_learning/nocw_vs_cw_lcmd_five_to1485_20261112"
SEEDS = (157, 887, 2357, 6101, 12203)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def table(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(budget, output):
    if output.exists():
        raise RuntimeError("Use a fresh milestone directory")
    historical = {}
    reference = ROOT / "reports/five_methods_cw_gradient_20261103/validation_curves.csv"
    with reference.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "cw_gradient_lcmd32":
                historical[int(row["seed"]), int(row["budget"])] = row
    rows, points, sources = [], {}, {str(reference.relative_to(ROOT)): sha(reference)}
    for seed in SEEDS:
        initial = read(STUDY / f"nocw/runtime/seed_{seed}/llm32_nocw/round_00/prediction_freeze.json")
        baseline = read(Path(initial["checkpoint_path"]).with_name("fit_audit.json"))
        for method in ("nocw", "cw"):
            for rnd, count in enumerate(range(333, budget + 1, 32)):
                if method == "nocw":
                    receipt_path = STUDY / f"nocw/runtime/seed_{seed}/llm32_nocw/round_{rnd:02d}/prediction_freeze.json"
                    receipt = read(receipt_path)
                    assert receipt["test_truth_access_count"] == 0
                    audit_path = Path(receipt["checkpoint_path"]).with_name("fit_audit.json")
                    sources[str(receipt_path.relative_to(ROOT))] = sha(receipt_path)
                else:
                    receipt_path = STUDY / f"cw_lcmd/runtime/seed_{seed}/center_width_lcmd/round_{rnd:02d}/fit_receipt.json"
                    if receipt_path.exists():
                        receipt = read(receipt_path)
                        audit_path = Path(receipt["checkpoint_path"]).with_name("fit_audit.json")
                        assert sha(audit_path) == receipt["fit_audit_sha256"]
                        sources[str(receipt_path.relative_to(ROOT))] = sha(receipt_path)
                    else:
                        old = historical[seed, count]
                        audit_path = ROOT / old["fit_audit"]
                        assert sha(audit_path) == old["fit_audit_sha256"]
                audit = read(audit_path)
                assert audit["train_rows"] == count
                assert audit["test_labels_used_for_fit_or_checkpoint_selection"] == 0
                assert all(audit[key] == baseline[key] for key in ("initialization_hash", "validation_ids_hash"))
                assert sha(audit_path.with_name("best.pt")) == audit["checkpoint_sha256"]
                assert sha(audit_path.with_name("predictions.csv.gz")) == audit["prediction_sha256"]
                value = audit["best_validation_combined_normalized_rmse"]
                assert math.isfinite(value) and value >= 0
                digest = sha(audit_path)
                sources[str(audit_path.relative_to(ROOT))] = digest
                rows.append({"method": method, "seed": seed, "budget": count,
                             "validation_nrmse": value, "fit_audit": str(audit_path.relative_to(ROOT)),
                             "fit_audit_sha256": digest})
                points[method, seed, count] = value
    means, paired = [], []
    for method in ("nocw", "cw"):
        for count in range(333, budget + 1, 32):
            values = [points[method, seed, count] for seed in SEEDS]
            means.append({"method": method, "budget": count, "n": 5,
                          "mean": statistics.mean(values), "sample_sd": statistics.stdev(values)})
    for seed in SEEDS:
        row = {"seed": seed, "budget": budget}
        for method in ("nocw", "cw"):
            values = [points[method, seed, count] for count in range(333, budget + 1, 32)]
            row[f"{method}_endpoint"] = values[-1]
            row[f"{method}_raw_AULC"] = sum((a + b)*16 for a, b in zip(values, values[1:]))/(budget-333)
        row["endpoint_delta_nocw_minus_cw"] = row["nocw_endpoint"]-row["cw_endpoint"]
        row["AULC_delta_nocw_minus_cw"] = row["nocw_raw_AULC"]-row["cw_raw_AULC"]
        paired.append(row)
    existing = read(STUDY / "reports/paired_summary.json")["paired_milestones"]
    for row in paired:
        old = next(r for r in existing if r["seed"] == row["seed"] and r["budget"] == budget)
        for method in ("nocw", "cw"):
            assert abs(old[method]["endpoint"]-row[f"{method}_endpoint"]) < 1e-12
            assert abs(old[method]["raw_normalized_AULC"]-row[f"{method}_raw_AULC"]) < 1e-12
    summary = {"budget": budget, "seeds": list(SEEDS), "points_per_curve": (budget-333)//32+1,
               "metric": "saved best-validation combined normalized RMSE", "lower_is_better": True,
               "lcmd_geometry": "center_width_gradient", "validation_only": True,
               "test_truth_reads": 0, "new_model_calls": 0, "new_fits": 0,
               "captured_utc": datetime.now(timezone.utc).isoformat(),
               "raw_nonmonotone_curves": True, "paired": paired}
    for method in ("nocw", "cw"):
        summary[method] = {metric: {"mean": statistics.mean(r[f"{method}_{metric}"] for r in paired),
                                   "sample_sd": statistics.stdev(r[f"{method}_{metric}"] for r in paired)}
                           for metric in ("endpoint", "raw_AULC")}
    summary["cw_AULC_wins"] = sum(r["AULC_delta_nocw_minus_cw"] > 0 for r in paired)
    summary["cw_endpoint_wins"] = sum(r["endpoint_delta_nocw_minus_cw"] > 0 for r in paired)
    output.mkdir(parents=True)
    table(output / "validation_curves.csv", rows)
    table(output / "five_seed_means.csv", means)
    table(output / "paired_results.csv", paired)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (output / "source_hashes.json").write_text(json.dumps(sources, indent=2) + "\n")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/nocw_cw_paired_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial Unicode MS", "font.size": 11,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(12.4, 7.5))
    fig.text(.09, .945, f"NoCW LLM 与 LCMD32 · 共同预算 L{budget}", fontsize=22, weight="bold", color="#182B43")
    fig.text(.09, .895, "固定五个配对 seed  |  均值与 ±1 样本标准差  |  验证集 NRMSE，越低越好",
             fontsize=11.5, color="#536579")
    for method, label, color in (("nocw", "纯 LLM：NoCW", "#D95F02"), ("cw", "纯数值：CW-gradient LCMD32", "#2265AF")):
        group = [r for r in means if r["method"] == method]
        xs, ys = [r["budget"] for r in group], [r["mean"] for r in group]
        ax.fill_between(xs, [r["mean"]-r["sample_sd"] for r in group],
                        [r["mean"]+r["sample_sd"] for r in group], color=color, alpha=.10, linewidth=0)
        ax.plot(xs, ys, color=color, linewidth=2.5, marker="o", markersize=3.7, label=label)
    ax.set(xlabel="已标注训练样本数 L", ylabel="验证集 NRMSE（越低越好）", xlim=(320, budget+15))
    ax.set_xticks([x for x in (333, 525, 653, 813, 1005, 1197, 1389, 1485) if x <= budget])
    ax.grid(axis="y", color="#E2E8EF", linewidth=.7)
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    fig.text(.09, .105, f"原始曲线 AULC：NoCW {summary['nocw']['raw_AULC']['mean']:.5f}  |  LCMD32 {summary['cw']['raw_AULC']['mean']:.5f}；LCMD32 在 {summary['cw_AULC_wins']}/5 个 seed 更低。", fontsize=10.5, color="#42566D")
    scope = ("既有 Row 分割上的完整共同预算验证结果；仅使用已保存验证指标。" if budget == 1485
             else "既有 Row 分割上的阶段性验证结果；总目标 L1485 尚未全部完成。")
    fig.text(.09, .055, "seed：157、887、2357、6101、12203；保留所有原始波动，未平滑或取历史最优。\n" + scope, fontsize=9.5, color="#536579", linespacing=1.6)
    fig.subplots_adjust(left=.09, right=.98, top=.83, bottom=.22)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"nocw_vs_lcmd_L{budget}.{extension}", dpi=210, facecolor="white")
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in ("budget", "points_per_curve", "nocw", "cw", "cw_AULC_wins", "cw_endpoint_wins")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, choices=(1005, 1389, 1485), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.budget, args.output.resolve())
