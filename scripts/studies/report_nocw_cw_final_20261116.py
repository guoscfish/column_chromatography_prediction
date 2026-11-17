#!/usr/bin/env python3
"""Export fully audited five-seed results from saved validation metrics only."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.studies import audit_cw_trajectory_20261116 as cw
from scripts.studies import audit_nocw_trajectory_20261116 as nocw
from scripts.studies import report_nocw_cw_milestone_20261114 as milestone

STUDY, SEEDS = nocw.STUDY, nocw.SEEDS
TARGETS = (1005, 1389, 1485)
COLORS = {"nocw": "#D95F02", "cw": "#2265AF"}
LABELS = {"nocw": "NoCW LLM", "cw": "CW-gradient LCMD32"}


def require_full_saved_points():
    for seed in SEEDS:
        files = sorted((STUDY / f"nocw/runtime/seed_{seed}/llm32_nocw").glob("round_*/prediction_freeze.json"))
        rounds = [int(p.parent.name.split("_")[-1]) for p in files]
        nocw.require(rounds == list(range(37)), f"Final export refused: seed {seed} lacks 37 actual fitted points")


def paired_tables(points):
    paired = []
    for budget in TARGETS:
        for seed in SEEDS:
            row = {"seed": seed, "budget": budget, "aulc_start_budget": 333}
            for method in ("nocw", "cw"):
                values = [points[method, seed, x] for x in range(333, budget + 1, 32)]
                nocw.require(all(math.isfinite(x) and x >= 0 for x in values), "Invalid saved metric")
                area = sum((a + b) * 16 for a, b in zip(values, values[1:]))
                row[f"{method}_endpoint"] = values[-1]
                row[f"{method}_raw_trapezoid_area"] = area
                row[f"{method}_normalized_AULC"] = area / (budget - 333)
            for metric in ("endpoint", "normalized_AULC"):
                row[f"{metric}_delta_nocw_minus_cw"] = row[f"nocw_{metric}"] - row[f"cw_{metric}"]
            paired.append(row)
    aggregates = []
    for budget in TARGETS:
        rows = [r for r in paired if r["budget"] == budget]
        summary = {"budget": budget, "paired_seeds": 5}
        for method in ("nocw", "cw"):
            for metric in ("endpoint", "normalized_AULC"):
                values = [r[f"{method}_{metric}"] for r in rows]
                summary[f"{method}_{metric}_mean"] = statistics.mean(values)
                summary[f"{method}_{metric}_sample_sd"] = statistics.stdev(values)
        for metric in ("endpoint", "normalized_AULC"):
            values = [r[f"{metric}_delta_nocw_minus_cw"] for r in rows]
            summary[f"{metric}_paired_delta_mean"] = statistics.mean(values)
            summary[f"{metric}_paired_delta_sample_sd"] = statistics.stdev(values)
            summary[f"cw_{metric}_wins"] = sum(x > 0 for x in values)
            summary[f"nocw_{metric}_wins"] = sum(x < 0 for x in values)
            summary[f"{metric}_ties"] = sum(x == 0 for x in values)
        aggregates.append(summary)
    return paired, aggregates


def plot_results(points, paired, output):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/nocw_cw_paired_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial Unicode MS", "font.size": 10,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    last_budget = max(key[2] for key in points)
    budgets = list(range(333, last_budget + 1, 32))
    targets = sorted({row["budget"] for row in paired})

    def save(fig, name):
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{name}.{extension}", dpi=210, facecolor="white")
        plt.close(fig)

    def draw(ax, seed):
        for method in ("nocw", "cw"):
            ax.plot(budgets, [points[method, seed, x] for x in budgets],
                    color=COLORS[method], marker="o", markersize=2.5, linewidth=1.6, label=LABELS[method])
        ax.set(title=f"Seed {seed}", xlabel="Labeled training samples L", ylabel="Validation NRMSE ↓",
               xlim=(320, last_budget + 15))
        ax.set_xticks([x for x in (333, 653, 1005, 1389, 1485) if x <= last_budget])
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=.2)

    for seed in SEEDS:
        fig, ax = plt.subplots(figsize=(10, 5.6), layout="constrained")
        draw(ax, seed)
        ax.legend(frameon=False)
        fig.suptitle(f"Raw saved validation learning curves · {len(budgets)-1} × batch32 · L333–L{last_budget}", fontsize=13)
        save(fig, f"raw_curves_seed_{seed}")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), layout="constrained")
    for ax, seed in zip(axes.flat, SEEDS):
        draw(ax, seed)
    ax = axes.flat[-1]
    for method in ("nocw", "cw"):
        means = [statistics.mean(points[method, seed, x] for seed in SEEDS) for x in budgets]
        sd = [statistics.stdev(points[method, seed, x] for seed in SEEDS) for x in budgets]
        ax.plot(budgets, means, color=COLORS[method], linewidth=2, label=LABELS[method])
        ax.fill_between(budgets, [m-s for m, s in zip(means, sd)], [m+s for m, s in zip(means, sd)],
                        color=COLORS[method], alpha=.12)
    ax.set(title="Five-seed mean ± sample SD", xlabel="Labeled training samples L", ylabel="Validation NRMSE ↓")
    ax.set_xticks([x for x in (333, 653, 1005, 1389, 1485) if x <= last_budget])
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=.2)
    axes.flat[0].legend(frameon=False)
    fig.suptitle("NoCW LLM vs CW-gradient LCMD32 · raw non-monotonic curves · saved validation only", fontsize=16)
    save(fig, "five_seed_raw_learning_curves")
    for metric, title in (("endpoint", "Saved validation endpoint NRMSE ↓"),
                          ("normalized_AULC", "Raw-curve normalized trapezoid AULC ↓")):
        fig, axes = plt.subplots(1, len(targets), figsize=(4.3*len(targets), 5), layout="constrained", squeeze=False)
        for ax, budget in zip(axes.flat, targets):
            rows = [r for r in paired if r["budget"] == budget]
            for row in rows:
                ys = [row[f"{m}_{metric}"] for m in ("nocw", "cw")]
                ax.plot([0, 1], ys, color="#9AA6B2", linewidth=1, alpha=.65)
                for x, method in enumerate(("nocw", "cw")):
                    ax.scatter(x, ys[x], color=COLORS[method], s=35)
            all_values = [row[f"{method}_{metric}"] for row in rows for method in ("nocw", "cw")]
            gap = max((max(all_values)-min(all_values))*.065, .002)
            previous = -math.inf
            for row in sorted(rows, key=lambda r: r[f"cw_{metric}"]):
                actual = row[f"cw_{metric}"]
                label_y = max(actual, previous + gap)
                ax.annotate(str(row["seed"]), (1, actual), xytext=(1.13, label_y),
                            fontsize=8, va="center", arrowprops={"arrowstyle": "-", "color": "#9AA6B2", "lw": .6})
                previous = label_y
            ax.set(title=f"L{budget}", ylabel=title, xlim=(-.3, 1.5))
            ax.set_xticks([0, 1], ["NoCW", "CW-gradient LCMD32"], fontsize=9)
            ax.grid(axis="y", alpha=.2)
        fig.suptitle(title + " · five paired seeds", fontsize=14)
        save(fig, f"paired_{metric}_three_budgets")


def run(output):
    nocw.require(not output.exists(), "Use a fresh final output directory")
    require_full_saved_points()
    spec = importlib.util.spec_from_file_location("frozen_final_validation", ROOT / "scripts/studies/run_nocw_cw_paired_to1485_20261112.py")
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    protocol = frozen.validate()
    audits = {"nocw": [nocw.audit_seed(seed) for seed in SEEDS],
              "cw": [cw.audit_seed(seed) for seed in SEEDS]}
    # The existing exporter independently checks all 370 fit metrics and hashes,
    # and compares each final AULC/endpoint to the frozen script's derived report.
    milestone.run(1485, output)
    points = {}
    with (output / "validation_curves.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["method"], int(row["seed"]), int(row["budget"])
            nocw.require(key not in points, "Duplicate saved evaluation point")
            points[key] = float(row["validation_nrmse"])
    nocw.require(len(points) == 370, "Final export requires ten curves × 37 fitted points")
    paired, aggregates = paired_tables(points)
    derived = nocw.read(STUDY / "reports/paired_summary.json")["paired_milestones"]
    for row in paired:
        old = next(r for r in derived if r["seed"] == row["seed"] and r["budget"] == row["budget"])
        for method in ("nocw", "cw"):
            nocw.require(abs(old[method]["endpoint"] - row[f"{method}_endpoint"]) < 1e-12,
                         "Final endpoint differs from paired report")
            nocw.require(abs(old[method]["raw_normalized_AULC"] - row[f"{method}_normalized_AULC"]) < 1e-12,
                         "Final AULC differs from paired report")
    milestone.table(output / "paired_three_budgets.csv", paired)
    milestone.table(output / "paired_three_budgets_aggregate.csv", aggregates)
    (output / "ten_trajectory_audit.json").write_text(json.dumps({
        "checked_utc": datetime.now(timezone.utc).isoformat(), "scientific_protocol": protocol,
        "all_ten_complete": True, "methods": audits, "source_test_label_reads": 0,
        "new_model_calls": 0, "new_fits": 0}, ensure_ascii=False, indent=2) + "\n")
    plot_results(points, paired, output)
    sources = nocw.read(output / "source_hashes.json")
    for results in audits.values():
        for result in results:
            sources.update(result["source_hashes"])
    sources[str(Path(__file__).relative_to(ROOT))] = milestone.sha(Path(__file__))
    sources[str(Path(cw.__file__).relative_to(ROOT))] = milestone.sha(Path(cw.__file__))
    sources[str(Path(nocw.__file__).relative_to(ROOT))] = milestone.sha(Path(nocw.__file__))
    (output / "source_hashes.json").write_text(json.dumps(sources, indent=2) + "\n")
    summary = nocw.read(output / "summary.json")
    summary.update({"all_ten_complete": True, "total_saved_evaluation_points": 370,
                    "paired_three_budgets": paired, "three_budget_aggregates": aggregates,
                    "AULC_definition": "Trapezoid integration of raw validation NRMSE from L333 to each target, divided by target minus 333",
                    "smoothing": False, "cumulative_minimum": False})
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    manifest = {f.name: milestone.sha(f) for f in sorted(output.iterdir()) if f.is_file()}
    (output / "artifact_hashes.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"final_output": str(output), "saved_points": len(points), "paired_rows": len(paired),
                      "all_ten_complete": True}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())
