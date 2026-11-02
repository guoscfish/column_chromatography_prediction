#!/usr/bin/env python
"""Build the 4g row active-learning master report from frozen study outputs.

This script only reads existing CSV artifacts.  It never starts training or
modifies the source studies.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "studies/active_learning/qgeognn_v2_row_master_report"
FIGURES = MASTER / "figures"
RESULTS = MASTER / "results"
SEEDS = (157, 6101)
MAIN_METHODS = (
    "random",
    "lcmd",
    "hybrid",
    "gradient_maxdet",
    "kernel_ivr",
    "center_width_lcmd",
)
EXPLORATORY_METHODS = (
    "gradient_latent_fusion_maxdet",
    "gradient_latent_fusion_a080_maxdet",
    "cw_ivr_portfolio",
)
ALL_METHODS = MAIN_METHODS + EXPLORATORY_METHODS
DISPLAY = {
    "random": "Random",
    "lcmd": "LCMD",
    "hybrid": "Hybrid",
    "gradient_maxdet": "Gradient MaxDet",
    "kernel_ivr": "Kernel IVR",
    "center_width_lcmd": "Center-width LCMD",
    "gradient_latent_fusion_maxdet": "Latent-fusion MaxDet",
    "gradient_latent_fusion_a080_maxdet": "Latent-fusion a=0.80 MaxDet",
    "cw_ivr_portfolio": "CW-IVR portfolio",
}
SCOPE = {m: ("main" if m in MAIN_METHODS else "exploratory") for m in ALL_METHODS}
SOURCES = {
    "random": "qgeognn_v2_row_sequential_b32",
    "lcmd": "qgeognn_v2_row_sequential_b32",
    "hybrid": "qgeognn_v2_row_sequential_b32",
    "gradient_maxdet": "qgeognn_v2_row_maxdet_b32",
    "kernel_ivr": "qgeognn_v2_row_kernel_ivr_b32",
    "center_width_lcmd": "qgeognn_v2_row_cw_lcmd_to_1005",
    "gradient_latent_fusion_maxdet": "qgeognn_v2_row_fused_maxdet_b32",
    "gradient_latent_fusion_a080_maxdet": "qgeognn_v2_row_fused_maxdet_a080_b32_screen",
    "cw_ivr_portfolio": "qgeognn_v2_row_cw_ivr_portfolio_b32",
}
SOURCE_METHODS = {
    "random": "random",
    "lcmd": "lcmd",
    "hybrid": "hybrid",
    "gradient_maxdet": "gradient_maxdet",
    "kernel_ivr": "kernel_ivr",
    "center_width_lcmd": "center_width_lcmd",
    "gradient_latent_fusion_maxdet": "gradient_latent_fusion_maxdet",
    "gradient_latent_fusion_a080_maxdet": "gradient_latent_fusion_a080_maxdet",
    "cw_ivr_portfolio": "cw_ivr_portfolio",
}
INTERVALS = ((333, 429), (429, 525), (525, 653), (653, 1005), (333, 1005))
METRIC_COLUMNS = (
    "AULC_333_429",
    "AULC_429_525",
    "AULC_525_653",
    "AULC_653_1005",
    "AULC_333_1005",
    "NRMSE_1005",
    "V1_RMSE_1005",
    "V2_RMSE_1005",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def load_tracks() -> dict[str, dict[int, list[dict[str, float]]]]:
    tracks: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for method in ALL_METHODS:
        source = ROOT / "studies/active_learning" / SOURCES[method] / "results/learning_curve_metrics.csv"
        rows = read_csv(source)
        source_method = SOURCE_METHODS[method]
        selected = [r for r in rows if r.get("method") == source_method and int(float(r["outer_seed"])) in SEEDS]
        if not selected:
            raise RuntimeError(f"No rows found for {method} in {source}")
        for row in selected:
            seed = int(float(row["outer_seed"]))
            label = int(float(row["active_label_count"]))
            tracks[method][seed].append(
                {
                    "active_label_count": label,
                    "nrmse": number(row.get("combined_normalized_RMSE")),
                    "v1": number(row.get("V1_RMSE")),
                    "v2": number(row.get("V2_RMSE")),
                }
            )
        for seed in tracks[method]:
            tracks[method][seed].sort(key=lambda x: x["active_label_count"])
    return tracks


def normalized_auc(points: list[dict[str, float]], start: int, stop: int) -> float:
    """Mean y over [start, stop], using the same normalized trapezoid AULC convention."""
    chosen = [p for p in points if start <= p["active_label_count"] <= stop and math.isfinite(p["nrmse"])]
    if len(chosen) < 2 or chosen[0]["active_label_count"] != start or chosen[-1]["active_label_count"] != stop:
        return math.nan
    x = np.array([p["active_label_count"] for p in chosen], dtype=float)
    y = np.array([p["nrmse"] for p in chosen], dtype=float)
    area = np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)
    return float(area / (stop - start))


def endpoint(points: list[dict[str, float]], label: int, key: str) -> float:
    for point in points:
        if point["active_label_count"] == label:
            return point[key]
    return math.nan


def calculate_rows(tracks: dict[str, dict[int, list[dict[str, float]]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in ALL_METHODS:
        for seed in SEEDS:
            points = tracks[method].get(seed, [])
            max_label = max((p["active_label_count"] for p in points), default=0)
            row: dict[str, object] = {
                "record_type": "per_seed",
                "scope": SCOPE[method],
                "method": method,
                "display_name": DISPLAY[method],
                "outer_seed": seed,
                "source_study": SOURCES[method],
                "available_through": max_label or "",
            }
            for start, stop in INTERVALS:
                row[f"AULC_{start}_{stop}"] = normalized_auc(points, start, stop)
            row["NRMSE_1005"] = endpoint(points, 1005, "nrmse")
            row["V1_RMSE_1005"] = endpoint(points, 1005, "v1")
            row["V2_RMSE_1005"] = endpoint(points, 1005, "v2")
            rows.append(row)
    for method in ALL_METHODS:
        seed_rows = [r for r in rows if r["method"] == method]
        mean_row: dict[str, object] = {
            "record_type": "mean",
            "scope": SCOPE[method],
            "method": method,
            "display_name": DISPLAY[method],
            "outer_seed": "",
            "source_study": SOURCES[method],
            "available_through": max((int(r["available_through"]) for r in seed_rows if r["available_through"]), default=""),
        }
        for column in METRIC_COLUMNS:
            values = [float(r[column]) for r in seed_rows if isinstance(r[column], (int, float)) and math.isfinite(float(r[column]))]
            mean_row[column] = float(np.mean(values)) if values else math.nan
        rows.append(mean_row)
    return rows


def write_summary(rows: list[dict[str, object]]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    columns = ["record_type", "scope", "method", "display_name", "outer_seed", "source_study", "available_through", *METRIC_COLUMNS]
    with (RESULTS / "master_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, "") for key in columns}
            for key in METRIC_COLUMNS:
                if isinstance(out[key], float) and not math.isfinite(out[key]):
                    out[key] = ""
                elif isinstance(out[key], float):
                    out[key] = f"{out[key]:.10f}"
            writer.writerow(out)


def mean_value(rows: list[dict[str, object]], method: str, column: str) -> float:
    row = next(r for r in rows if r["record_type"] == "mean" and r["method"] == method)
    return float(row[column])


def plot_full(tracks: dict[str, dict[int, list[dict[str, float]]]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    colors = plt.get_cmap("tab10").colors
    for idx, method in enumerate(MAIN_METHODS):
        color = colors[idx]
        seed_points = [tracks[method][seed] for seed in SEEDS]
        x = np.array([p["active_label_count"] for p in seed_points[0]])
        ys = np.array([[p["nrmse"] for p in points] for points in seed_points])
        ax.plot(x, ys.mean(axis=0), lw=3.0, color=color, label=DISPLAY[method], zorder=3)
        for seed_idx, seed in enumerate(SEEDS):
            ax.plot(x, ys[seed_idx], lw=0.9, alpha=0.38, color=color, zorder=2)
    for budget in (429, 525, 653, 1005):
        ax.axvline(budget, color="#667085", lw=0.9, ls="--", alpha=0.65)
        ax.text(budget + 5, 0.99, str(budget), transform=ax.get_xaxis_transform(), fontsize=9, color="#475467")
    ax.set_title("4g row active learning: complete 333→1005 trajectories", fontsize=15, weight="bold")
    ax.set_xlabel("Active labels")
    ax.set_ylabel("Combined normalized RMSE (lower is better)")
    ax.set_xlim(325, 1015)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=2, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "master_full_333_1005.png", bbox_inches="tight")
    plt.close(fig)


def plot_stage_aulc(rows: list[dict[str, object]]) -> None:
    labels = ["333–429", "429–525", "525–653", "653–1005", "333–1005"]
    columns = [f"AULC_{a}_{b}" for a, b in INTERVALS]
    x = np.arange(len(labels))
    width = 0.13
    fig, ax = plt.subplots(figsize=(14, 7), dpi=180)
    colors = plt.get_cmap("tab10").colors
    for idx, method in enumerate(MAIN_METHODS):
        vals = [mean_value(rows, method, col) for col in columns]
        ax.bar(x + (idx - 2.5) * width, vals, width, label=DISPLAY[method], color=colors[idx])
    ax.set_title("4g row active learning: normalized AULC by stage", fontsize=15, weight="bold")
    ax.set_ylabel("Normalized AULC (lower is better)")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=3, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "master_stage_aulc.png", bbox_inches="tight")
    plt.close(fig)


def plot_endpoint(rows: list[dict[str, object]]) -> None:
    metrics = [("NRMSE_1005", "NRMSE @ 1005"), ("V1_RMSE_1005", "V1 RMSE @ 1005"), ("V2_RMSE_1005", "V2 RMSE @ 1005")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), dpi=180, sharex=True)
    for ax, (column, title) in zip(axes, metrics):
        vals = [mean_value(rows, method, column) for method in MAIN_METHODS]
        bars = ax.bar(np.arange(len(MAIN_METHODS)), vals, color=[plt.get_cmap("tab10").colors[i] for i in range(len(MAIN_METHODS))])
        ax.set_title(title, fontsize=11, weight="bold")
        ax.grid(axis="y", alpha=0.22)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
        ax.set_xticks(np.arange(len(MAIN_METHODS)), [DISPLAY[m] for m in MAIN_METHODS], rotation=45, ha="right")
        ax.set_ylabel("RMSE (lower is better)")
    fig.suptitle("4g row active learning: endpoint comparison at 1005 labels", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "master_endpoint_1005.png", bbox_inches="tight")
    plt.close(fig)


def plot_exploratory(tracks: dict[str, dict[int, list[dict[str, float]]]]) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    colors = plt.get_cmap("tab10").colors
    for idx, method in enumerate(ALL_METHODS):
        color = colors[idx % len(colors)]
        points_by_seed = tracks[method]
        usable = [points_by_seed[s] for s in SEEDS if any(p["active_label_count"] == 653 for p in points_by_seed.get(s, []))]
        if not usable:
            continue
        x = np.array([p["active_label_count"] for p in usable[0] if p["active_label_count"] <= 653])
        ys = np.array([[p["nrmse"] for p in points if p["active_label_count"] <= 653] for points in usable])
        ax.plot(x, ys.mean(axis=0), lw=2.5 if method in MAIN_METHODS else 2.1, color=color, label=DISPLAY[method])
        if method not in MAIN_METHODS:
            for values in ys:
                ax.plot(x, values, lw=0.8, alpha=0.3, color=color)
    for budget in (429, 525, 653):
        ax.axvline(budget, color="#667085", lw=0.9, ls="--", alpha=0.65)
        ax.text(budget + 5, 0.99, str(budget), transform=ax.get_xaxis_transform(), fontsize=9, color="#475467")
    ax.set_title("4g row active learning: exploratory methods through 653 labels", fontsize=15, weight="bold")
    ax.set_xlabel("Active labels")
    ax.set_ylabel("Combined normalized RMSE (lower is better)")
    ax.set_xlim(325, 663)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=2, frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "master_exploratory_333_653.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    MASTER.mkdir(parents=True, exist_ok=True)
    tracks = load_tracks()
    rows = calculate_rows(tracks)
    write_summary(rows)
    plot_full(tracks)
    plot_stage_aulc(rows)
    plot_endpoint(rows)
    plot_exploratory(tracks)
    print(f"Wrote {RESULTS / 'master_summary.csv'}")
    for name in ("master_full_333_1005.png", "master_stage_aulc.png", "master_endpoint_1005.png", "master_exploratory_333_653.png"):
        print(f"Wrote {FIGURES / name}")


if __name__ == "__main__":
    main()
