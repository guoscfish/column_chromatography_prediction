"""Residual/uncertainty relationships from completed study predictions only.

The five equal-count variance-rank bins use predicted observation variance and
ID tie-breaking only. No response value defines, tunes or changes bin edges.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/representation_gp_residual_matplotlib")
import numpy as np
import pandas as pd
from scipy.stats import norm

from summarize_results import clean, dump, export_csv, mean_groups, summarize_groups, LABELS

HERE = Path(__file__).resolve().parent
KEYS = ["seed", "budget", "representation", "kernel", "initialization", "endpoint"]
VALUES = ["z_rms", "z_mean", "mean_absolute_z", "coverage_90", "coverage_95", "mean_sigma_L0",
          "normalized_rmse", "normalized_mae", "minimum_sigma_L0", "maximum_sigma_L0"]
BIN_COUNT = 5


def identity(row):
    return tuple(row.get(key) for key in KEYS)


def read_local(path, parent):
    path, parent = Path(path), Path(parent).resolve()
    if path.resolve().parent != parent:
        raise PermissionError("residual diagnostic source is outside its declared study state")
    return json.loads(path.read_text())


def endpoint_bins(rows, scale, *, bin_count=BIN_COUNT):
    """Compute fixed-rank bins; returns diagnostics without altering predictions."""
    if not rows or bin_count != BIN_COUNT or not np.isfinite(scale) or scale <= 0:
        raise ValueError("nonempty predictions, positive L0 scale and exactly five bins required")
    ids = np.array([str(r["id"]) for r in rows])
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate evaluation ID within endpoint fit")
    truth = np.array([r["truth_ml"] for r in rows], dtype=np.float64)
    mean = np.array([r["mean_ml"] for r in rows], dtype=np.float64)
    variance = np.array([r["observation_variance_ml2"] for r in rows], dtype=np.float64)
    if not all(np.isfinite(x).all() for x in (truth, mean, variance)) or np.any(variance <= 0):
        raise ValueError("finite predictions/truth and strictly positive observation variance required")
    sigma = np.sqrt(variance)
    z = (truth - mean) / sigma
    error = (truth - mean) / scale
    order = np.lexsort((ids, variance))
    out, offset = [], 0
    for index, positions in enumerate(np.array_split(order, BIN_COUNT), start=1):
        if len(positions) == 0:
            continue
        out.append(dict(variance_rank_bin=index, n_observations=len(positions),
                        rank_fraction_lower=offset / len(rows), rank_fraction_upper=(offset + len(positions)) / len(rows),
                        member_ids_sha256=hashlib.sha256(json.dumps(ids[positions].tolist()).encode()).hexdigest(),
                        historical_L0_scale_ml=float(scale),
                        z_rms=float(np.sqrt(np.mean(z[positions]**2))), z_mean=float(np.mean(z[positions])),
                        mean_absolute_z=float(np.mean(np.abs(z[positions]))),
                        coverage_90=float(np.mean(np.abs(z[positions]) <= norm.ppf(.95))),
                        coverage_95=float(np.mean(np.abs(z[positions]) <= norm.ppf(.975))),
                        mean_sigma_L0=float(np.mean(sigma[positions] / scale)),
                        minimum_sigma_L0=float(np.min(sigma[positions] / scale)),
                        maximum_sigma_L0=float(np.max(sigma[positions] / scale)),
                        normalized_rmse=float(np.sqrt(np.mean(error[positions]**2))),
                        normalized_mae=float(np.mean(np.abs(error[positions])))))
        offset += len(positions)
    return out


def load_state_bins(study):
    """Read config, complete markers, metrics and predictions within this study."""
    study = Path(study).resolve()
    root = study / "results"
    config_path = study / "config.json"
    config = read_local(config_path, study) if config_path.exists() else {}
    completed, excluded, bins, sources = [], [], [], []
    for directory in sorted(root.glob("seed_*_L*")):
        if not directory.is_dir():
            continue
        if directory.resolve().parent != root.resolve():
            raise PermissionError("state source escapes this study's results")
        match = re.fullmatch(r"seed_(\d+)_L(\d+)", directory.name)
        if not match:
            continue
        if not (directory / "complete.json").exists():
            excluded.append(directory.name)
            continue
        state = int(match[1]), int(match[2])
        receipt = read_local(directory / "complete.json", directory)
        if (int(receipt["seed"]), int(receipt["budget"])) != state:
            raise ValueError("complete marker identity mismatch")
        if any(receipt.get(k, 0) for k in ("test_truth_access_count", "new_gnn_fits", "al_trajectories_started")):
            raise ValueError("state receipt declares a forbidden operation")
        metrics = read_local(directory / "metrics.json", directory)
        predictions = read_local(directory / "predictions.json", directory)
        if len(metrics) != receipt["endpoint_fits"]:
            raise ValueError("endpoint metric count differs from completed fit count")
        metric_map, scale_map = {}, {}
        for row in metrics:
            if (int(row["seed"]), int(row["budget"])) != state:
                raise ValueError("metric belongs to a different state")
            key = identity(row)
            if key in metric_map:
                raise ValueError("duplicate metric endpoint identity")
            metric_map[key] = row
            if row.get("rmse", 0) > 0 and row.get("normalized_rmse_L0", 0) > 0:
                scale_map.setdefault(row["endpoint"], []).append(row["rmse"] / row["normalized_rmse_L0"])
        scales = {}
        for endpoint, values in scale_map.items():
            if not np.allclose(values, values[0], rtol=1e-8, atol=1e-8):
                raise ValueError("result-only historical L0 scales disagree across methods")
            scales[endpoint] = float(values[0])
        grouped = {}
        for row in predictions:
            if (int(row["seed"]), int(row["budget"])) != state:
                raise ValueError("prediction belongs to a different state")
            if row["endpoint"] not in ("V1", "V2"):
                raise ValueError("unknown response endpoint")
            key = identity(row)
            if key not in metric_map:
                raise ValueError("prediction identity has no matching completed metric")
            grouped.setdefault(key, []).append(row)
        if set(grouped) != set(metric_map):
            raise ValueError("completed endpoint predictions are missing")
        for key, rows in grouped.items():
            if len(rows) != receipt.get("evaluation_rows", len(rows)):
                raise ValueError("prediction row count differs from state evaluation receipt")
            endpoint = key[-1]
            if endpoint not in scales:
                raise ValueError("no positive historical L0 scale can be recovered from completed metrics")
            for row in endpoint_bins(rows, scales[endpoint]):
                bins.append({**dict(zip(KEYS, key)), **row})
        completed.append(dict(seed=state[0], budget=state[1], endpoint_fits=receipt["endpoint_fits"]))
        for name in ("complete.json", "metrics.json", "predictions.json"):
            path = directory / name
            sources.append(dict(path=str(path.relative_to(study)),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    actual = {(s["seed"], s["budget"]) for s in completed}
    expected = {(int(s), int(b)) for s in config.get("seeds", []) for b in config.get("budgets", [])}
    coverage = dict(completed_states=len(completed), expected_states=len(expected) or None,
                    complete_matrix=bool(expected) and actual == expected, states=completed,
                    missing_states=[dict(seed=s, budget=b) for s,b in sorted(expected - actual)],
                    incomplete_directories_excluded=excluded,
                    endpoint_fits=sum(s["endpoint_fits"] for s in completed),
                    deduplication="read each completed seed-budget directory exactly once on every run")
    return pd.DataFrame(bins), coverage, sources


def aggregate_bins(detail):
    if detail.empty:
        return {name: pd.DataFrame() for name in ("residual_endpoint_bins", "residual_state_bins", "residual_seed_bins",
                                                 "residual_bin_summary", "residual_endpoint_summary")}
    by_endpoint = mean_groups(detail, ["seed", "budget", "representation", "kernel", "endpoint", "variance_rank_bin"], VALUES)
    by_state = mean_groups(by_endpoint, ["seed", "budget", "representation", "kernel", "variance_rank_bin"], VALUES)
    by_seed = mean_groups(by_state, ["seed", "representation", "kernel", "variance_rank_bin"], VALUES)
    overall = summarize_groups(by_seed, ["representation", "kernel", "variance_rank_bin"], VALUES)
    endpoint_seed = mean_groups(by_endpoint, ["seed", "representation", "kernel", "endpoint", "variance_rank_bin"], VALUES)
    endpoint_overall = summarize_groups(endpoint_seed, ["representation", "kernel", "endpoint", "variance_rank_bin"], VALUES)
    return dict(residual_endpoint_bins=detail, residual_state_bins=by_state, residual_seed_bins=by_seed,
                residual_bin_summary=overall, residual_endpoint_summary=endpoint_overall)


def plot_relationship(overall, coverage, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "svg.fonttype":"none",
                         "axes.spines.top":False, "axes.spines.right":False, "savefig.dpi":220})
    if overall.empty:
        return [], {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
    fig.subplots_adjust(left=.06, right=.98, bottom=.32, top=.88, wspace=.28)
    colors = dict(zip(LABELS, plt.get_cmap("tab10").colors))
    groups = list(overall.groupby(["representation", "kernel"]))
    for (rep, kernel), group in groups:
        if kernel == "linear" or (kernel == "projected" and rep not in ("gnn", "llm_conditions", "fusion")):
            continue
        q = group.sort_values("variance_rank_bin")
        label = LABELS.get(rep, rep) + (" + projection" if kernel == "projected" else "")
        style = dict(color=colors.get(rep), linestyle="--" if kernel == "projected" else "-", marker="o", markersize=4, label=label)
        axes[0].plot(q.mean_sigma_L0_mean, q.z_rms_mean, **style)
        axes[1].plot(q.mean_sigma_L0_mean, q.coverage_90_mean, **style)
        axes[2].plot(q.mean_sigma_L0_mean, q.normalized_rmse_mean, **style)
    axes[0].axhline(1., color=".35", linestyle=":", label="Calibrated z RMS = 1")
    axes[0].set(xlabel="Predicted observation SD / L0 SD", ylabel="Standardized residual RMS", title="Residual scale versus uncertainty")
    axes[1].axhline(.9, color=".35", linestyle=":")
    axes[1].set(xlabel="Predicted observation SD / L0 SD", ylabel="Empirical 90% coverage", ylim=(0,1.03), title="Coverage within variance-rank bins")
    axes[2].set(xlabel="Predicted observation SD / L0 SD", ylabel="RMSE / L0 SD", title="Prediction error versus uncertainty")
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=.2)
    if (overall.z_rms_mean > 10).any():
        axes[0].set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.53,.075), ncol=4,
                        frameon=False, fontsize=8, labelspacing=.6, columnspacing=1.5)
    fig.suptitle("Residual diagnostics: five equal-count bins ranked only by predicted variance", fontsize=11)
    footer = fig.text(.01,.012,f"Completed states: {coverage['completed_states']}/{coverage['expected_states'] or '?'}; initialization → endpoint → budget within seed → seed means",
                      va="bottom",fontsize=8,color=".35")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    footer_box, legend_box = footer.get_window_extent(renderer), legend.get_window_extent(renderer)
    if footer_box.overlaps(legend_box):
        raise ValueError("residual diagnostic footer/legend overlap")
    qa = dict(footer_legend_overlaps=0, footer_bounds_display=list(footer_box.bounds), legend_bounds_display=list(legend_box.bounds),
              z_axis_scale=axes[0].get_yscale(), ideal_standardized_residual_rms=1.)
    files = []
    for suffix in ("png", "svg"):
        name = "residual_uncertainty_relationship." + suffix
        fig.savefig(output / name,bbox_inches="tight",facecolor="white")
        files.append(name)
    plt.close(fig)
    return files, qa


def run(study_dir=HERE, output_dir=None, *, plots=True):
    study = Path(study_dir).resolve()
    output = Path(output_dir).resolve() if output_dir else study / "summary"
    if output == study / "results" or study / "results" in output.parents:
        raise ValueError("residual diagnostics cannot overwrite model results")
    output.mkdir(parents=True,exist_ok=True)
    detail, coverage, sources = load_state_bins(study)
    tables = aggregate_bins(detail)
    for name, frame in tables.items():
        export_csv(frame, output / (name + ".csv"))
    figures, layout = plot_relationship(tables["residual_bin_summary"],coverage,output) if plots else ([],{})
    result = dict(schema="representation-gp-residual-diagnostics-v1", coverage=coverage,
                  partial_results=not coverage["complete_matrix"], bin_count=BIN_COUNT,
                  bin_rule="per fitted endpoint: observation variance ascending; equal-count np.array_split; ID lexicographic tie-break; no y in bin definitions",
                  standardized_residual="z=(truth_ml-mean_ml)/sqrt(observation_variance_ml2)",
                  scale_source="historical L0 SD recovered from completed metrics rmse/normalized_rmse_L0; checked across methods",
                  aggregation="mean initializations, mean endpoints, mean available budgets within seed, equal-weight mean seeds; RMS values averaged after group calculation",
                  statistical_unit="development seed; repeated states and shared validation rows are not independent replicates",
                  interpretation="descriptive residual relationship diagnostic; no calibration adjustment, refitting or method selection",
                  bin_summary=tables["residual_bin_summary"].to_dict("records"),
                  endpoint_summary=tables["residual_endpoint_summary"].to_dict("records"),
                  source_files=sources, source_scope="this study/config.json and completed results state complete.json, metrics.json, predictions.json only",
                  original_data_reads=0, test_truth_reads=0, figures=figures, figure_layout_qa=layout)
    dump(output / "residual_diagnostics.json",result)
    return clean(result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir",type=Path,default=HERE)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--no-plots",action="store_true")
    args=parser.parse_args()
    result=run(args.study_dir,args.output_dir,plots=not args.no_plots)
    print(json.dumps(dict(completed_states=result["coverage"]["completed_states"],expected_states=result["coverage"]["expected_states"],
                          endpoint_fits=result["coverage"]["endpoint_fits"],figures=result["figures"])))


if __name__=="__main__":
    main()
