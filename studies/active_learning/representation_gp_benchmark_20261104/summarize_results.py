"""Summarize completed offline states only; never open original data or test truth.

Aggregation order: initialization -> endpoint -> budget within seed -> seed.
No confidence interval or independence claim treats the 20 states as repeats.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/representation_gp_summary_matplotlib")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
IDENTITY = ["seed", "budget", "representation", "kernel", "initialization", "endpoint"]
METRICS = ["normalized_rmse_L0", "rmse", "mae", "nlpd", "coverage_90", "coverage_95",
           "mean_interval_width_90", "mean_interval_width_95", "standardized_error_rms",
           "absolute_error_uncertainty_spearman", "fit_seconds", "inference_seconds"]
LABELS = {"structured": "Morgan + conditions", "gnn": "GNN hidden", "raw": "Raw gradient",
          "cw": "CW gradient", "llm_smiles": "LLM SMILES", "llm_conditions": "LLM + conditions",
          "fusion": "GNN + LLM fusion"}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def records(frame):
    return clean(frame.to_dict("records")) if len(frame) else []


def dump(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def export_csv(frame, path):
    out = frame.copy()
    for column in out:
        if out[column].map(lambda x: isinstance(x, (list, dict, tuple))).any():
            out[column] = out[column].map(lambda x: json.dumps(clean(x), ensure_ascii=False)
                                          if isinstance(x, (list, dict, tuple)) else x)
    out.to_csv(path, index=False)


def load_completed(study):
    """Only explicit study results and its optional frozen config are readable."""
    study = Path(study).resolve()
    config_path = study / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    root = study / "results"
    tables = {name: [] for name in ("metrics", "fits", "contrasts", "acquisitions", "robustness")}
    states, sources, ignored, repairs = [], [], [], []
    for directory in sorted(root.glob("seed_*_L*")):
        if not directory.is_dir():
            continue
        if directory.resolve().parent != root.resolve():
            raise ValueError("state directory points outside this study's results")
        match = re.fullmatch(r"seed_(\d+)_L(\d+)", directory.name)
        if not match:
            continue
        marker = directory / "complete.json"
        if not marker.exists():
            ignored.append(directory.name)
            continue
        receipt = json.loads(marker.read_text())
        state = (int(match[1]), int(match[2]))
        if (int(receipt["seed"]), int(receipt["budget"])) != state:
            raise ValueError("complete marker does not match its state directory")
        if any(receipt.get(key, 0) for key in ("test_truth_access_count", "new_gnn_fits", "al_trajectories_started")):
            raise ValueError("completed state declares a forbidden operation")
        local = {}
        for name in tables:
            path = directory / (name + ".json")
            if not path.exists():
                raise ValueError(f"completed state missing required {name}.json: {directory.name}")
            rows = json.loads(path.read_text())
            if not isinstance(rows, list):
                raise ValueError("result table must be a row list")
            if name == "fits":
                repaired = 0
                for row in rows:
                    # Compatibility for the old {**base, **model.audit} merge:
                    # audit.seed is the optimizer seed, whereas base.seed is the state.
                    expected_model_seed = row.get("initialization")
                    expected_model_seed = 17 if expected_model_seed is None else int(expected_model_seed)
                    row.setdefault("model_random_seed", row.get("seed"))
                    if int(row["seed"]) != state[0] and int(row["seed"]) == expected_model_seed:
                        row["seed"] = state[0]
                        repaired += 1
                if repaired:
                    repairs.append(dict(state=directory.name, table="fits", rows=repaired,
                                        fields="seed: preserve optimizer seed as model_random_seed; restore state seed from verified complete marker"))
            if any((int(row["seed"]), int(row["budget"])) != state for row in rows):
                raise ValueError("result row belongs to a different state")
            local[name] = rows
            sources.append(str(path.relative_to(study)))
        if len(local["fits"]) != receipt["endpoint_fits"] or len(local["metrics"]) != receipt["endpoint_fits"]:
            raise ValueError("completed state fit/metric count differs from complete marker")
        seen = set()
        for row in local["metrics"]:
            identity = tuple(row.get(key) for key in IDENTITY)
            if identity in seen:
                raise ValueError("duplicate endpoint metric in completed state")
            seen.add(identity)
        kernel_repairs = 0
        for row in local["fits"]:
            # Older runner dictionaries let audit.kernel overwrite 'projected'.
            row["actual_kernel"] = row.get("actual_kernel", row["kernel"])
            if row.get("trainable_projection", False):
                if row.get("initialization") not in config.get("projection_seeds", [17, 29, 43]) or row["actual_kernel"] != "matern52":
                    raise ValueError("projection fit metadata contradicts frozen initialization/kernel rule")
                if row["kernel"] != "projected":
                    kernel_repairs += 1
                row["kernel"] = "projected"
            elif row.get("initialization") is not None:
                raise ValueError("fixed representation fit unexpectedly declares projection initialization")
        if kernel_repairs:
            repairs.append(dict(state=directory.name, table="fits", rows=kernel_repairs,
                                fields="kernel: preserve actual_kernel; restore projected variant from trainable_projection and initialization"))
        fit_identities = [tuple(row.get(key) for key in IDENTITY) for row in local["fits"]]
        if len(set(fit_identities)) != len(fit_identities) or set(fit_identities) != seen:
            raise ValueError("fit identities do not match metric identities after explicit metadata repair")
        for name in tables:
            tables[name].extend(local[name])
        states.append({**receipt, "directory": directory.name})
        sources.append(str(marker.relative_to(study)))
    if len({(s["seed"], s["budget"]) for s in states}) != len(states):
        raise ValueError("repeated complete state")
    expected = [(int(s), int(b)) for s in config.get("seeds", []) for b in config.get("budgets", [])]
    actual = {(s["seed"], s["budget"]) for s in states}
    coverage = {"completed_states": len(states), "expected_states": len(expected) or None,
                "complete_matrix": bool(expected) and actual == set(expected),
                "states": [{"seed": s["seed"], "budget": s["budget"]} for s in states],
                "missing_states": [{"seed": s, "budget": b} for s, b in expected if (s, b) not in actual],
                "incomplete_directories_excluded": ignored, "deduplication": "one seed-budget directory; preflight/full share state",
                "endpoint_fits": sum(s["endpoint_fits"] for s in states),
                "receipt_identity_repairs": repairs}
    return config, tables, coverage, sources


def mean_groups(frame, keys, values):
    if frame.empty:
        return pd.DataFrame(columns=keys + values)
    columns = [x for x in values if x in frame]
    return frame.groupby(keys, dropna=False, sort=True)[columns].mean().reset_index()


def summarize_groups(seed_frame, groups, values):
    """Descriptive spread over seed-level means, with no pseudo-replication."""
    rows = []
    if seed_frame.empty:
        return pd.DataFrame()
    for key, group in seed_frame.groupby(groups, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key))
        row["seed_count"] = int(group.seed.nunique())
        for metric in values:
            if metric not in group:
                continue
            v = group[metric].dropna().to_numpy(float)
            row[metric + "_mean"] = float(v.mean()) if len(v) else None
            row[metric + "_seed_sd"] = float(v.std(ddof=1)) if len(v) > 1 else None
            row[metric + "_min_seed"] = float(v.min()) if len(v) else None
            row[metric + "_max_seed"] = float(v.max()) if len(v) else None
        rows.append(row)
    return pd.DataFrame(rows)


def metric_tables(raw):
    frame = pd.DataFrame(raw)
    if frame.empty:
        return {k: pd.DataFrame() for k in ("metrics", "state_endpoint_metrics", "state_metrics", "seed_metrics",
                                            "overall_metrics", "budget_metrics", "endpoint_budget_metrics")}
    values = [m for m in METRICS if m in frame]
    endpoint = mean_groups(frame, ["seed", "budget", "representation", "kernel", "endpoint"], values)
    state = mean_groups(endpoint, ["seed", "budget", "representation", "kernel"], values)
    seed = mean_groups(state, ["seed", "representation", "kernel"], values)
    counts = state.groupby(["seed", "representation", "kernel"]).size().rename("budgets_available").reset_index()
    seed = seed.merge(counts, on=["seed", "representation", "kernel"])
    overall = summarize_groups(seed, ["representation", "kernel"], values)
    budget = summarize_groups(state, ["budget", "representation", "kernel"], values)
    endpoint_budget = summarize_groups(endpoint, ["budget", "representation", "kernel", "endpoint"], values)
    return dict(metrics=frame, state_endpoint_metrics=endpoint, state_metrics=state, seed_metrics=seed,
                overall_metrics=overall, budget_metrics=budget, endpoint_budget_metrics=endpoint_budget)


def paired_tables(raw):
    frame = pd.DataFrame(raw)
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame(), []
    comparisons = [(f"{r}: projection - frozen", (r, "projected"), (r, "matern52"))
                   for r in ("gnn", "llm_conditions", "fusion")]
    comparisons += [("Fusion projected - GNN projected", ("fusion", "projected"), ("gnn", "projected")),
                    ("LLM conditions - structured", ("llm_conditions", "matern52"), ("structured", "matern52")),
                    ("LLM conditions - GNN", ("llm_conditions", "matern52"), ("gnn", "matern52")),
                    ("LLM conditions - SMILES", ("llm_conditions", "matern52"), ("llm_smiles", "matern52")),
                    ("Fusion frozen - GNN", ("fusion", "matern52"), ("gnn", "matern52")),
                    ("GNN - structured", ("gnn", "matern52"), ("structured", "matern52"))]
    comparisons += [(f"{r}: Matern - linear", (r, "matern52"), (r, "linear")) for r in ("raw", "cw")]
    states = []
    for label, candidate, control in comparisons:
        left = frame[(frame.representation == candidate[0]) & (frame.kernel == candidate[1])]
        right = frame[(frame.representation == control[0]) & (frame.kernel == control[1])]
        if left.empty or right.empty:
            continue
        on = ["seed", "budget", "endpoint"]
        if candidate[1] == control[1] == "projected":
            on += ["initialization"]
        joined = left.merge(right, on=on, suffixes=("_candidate", "_control"))
        values = [m for m in METRICS if m in frame]
        temporary = joined[["seed", "budget", "endpoint"]].copy()
        for m in values:
            temporary[m + "_delta"] = joined[m + "_candidate"] - joined[m + "_control"]
            temporary[m + "_candidate"] = joined[m + "_candidate"]
            temporary[m + "_control"] = joined[m + "_control"]
        cols = [x for x in temporary if x not in ("seed", "budget", "endpoint")]
        endpoint = mean_groups(temporary, ["seed", "budget", "endpoint"], cols)
        state = mean_groups(endpoint, ["seed", "budget"], cols)
        state["comparison"] = label
        state["initialization_policy"] = ("matched initialization" if candidate[1] == control[1] == "projected"
                                          else "initialization mean before endpoint mean")
        states.extend(records(state))
    state_frame = pd.DataFrame(states)
    if state_frame.empty:
        return state_frame, pd.DataFrame(), []
    vals = [x for x in state_frame if x.endswith(("_delta", "_candidate", "_control"))]
    seeds = mean_groups(state_frame, ["seed", "comparison"], vals)
    budget_counts = state_frame.groupby(["seed", "comparison"]).size().rename("paired_budget_count").reset_index()
    seeds = seeds.merge(budget_counts, on=["seed", "comparison"])
    summary = summarize_groups(seeds, ["comparison"], vals)
    for i, row in summary.iterrows():
        group = seeds[seeds.comparison.eq(row.comparison)]
        primary = "normalized_rmse_L0_delta"
        if primary in group:
            summary.loc[i, "primary_seed_win_fraction"] = float((group[primary] < -1e-12).mean())
            summary.loc[i, "primary_seed_tie_fraction"] = float((group[primary].abs() <= 1e-12).mean())
        summary.loc[i, "paired_seed_budget_count"] = int((state_frame.comparison == row.comparison).sum())
    return state_frame, seeds, records(summary)


def optimizer_summary(raw):
    frame = pd.DataFrame(raw)
    rows = []
    if frame.empty:
        return frame, rows
    frame["boundary_hit"] = [bool(r.get("boundary_hit", r.get("optimizer", {}).get("boundary_hit", False))) for r in raw]
    for key, group in frame.groupby(["representation", "kernel"], dropna=False):
        noise = group.noise_variance.astype(float)
        signal = group.signal_variance.astype(float)
        row = dict(representation=key[0], kernel=key[1], endpoint_fit_count=len(group),
                   nonconverged_fit_count=int((~group.optimizer_success.astype(bool)).sum()),
                   boundary_hit_count=int(group.boundary_hit.sum()),
                   nonzero_jitter_fit_count=int((group.numerical_jitter > 0).sum()),
                   maximum_numerical_jitter=float(group.numerical_jitter.max()),
                   noise_variance_median=float(noise.median()), signal_variance_median=float(signal.median()),
                   noise_signal_ratio_median=float((noise / signal).median()),
                   total_training_seconds=float(group.training_seconds.sum()),
                   training_seconds_median=float(group.training_seconds.median()),
                   negative_log_marginal_likelihood_median=float(group.negative_log_marginal_likelihood.median()),
                   count_interpretation="computational diagnostics; fits are not independent statistical replicates")
        if "lengthscale" in group and group.lengthscale.notna().any():
            row["lengthscale_median"] = float(group.lengthscale.dropna().median())
        if "projection_parameters" in group and group.projection_parameters.notna().any():
            row["projection_parameters"] = sorted(group.projection_parameters.dropna().astype(int).unique().tolist())
        rows.append(row)
    return frame, rows


def initialization_summary(raw):
    frame = pd.DataFrame(raw)
    if frame.empty:
        return pd.DataFrame(), []
    frame = frame[frame.kernel.eq("projected")]
    rows = []
    for key, group in frame.groupby(["seed", "budget", "representation", "endpoint"]):
        row = dict(zip(["seed", "budget", "representation", "endpoint"], key))
        row["initialization_count"] = int(group.initialization.nunique())
        for m in METRICS:
            if m in group:
                values = group[m].dropna().to_numpy(float)
                row[m + "_initialization_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
                row[m + "_initialization_range"] = float(np.ptp(values)) if len(values) else None
        rows.append(row)
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, []
    values = [c for c in detail if c.endswith(("_initialization_sd", "_initialization_range"))]
    endpoint_avg = mean_groups(detail, ["seed", "budget", "representation"], values)
    seed = mean_groups(endpoint_avg, ["seed", "representation"], values)
    return detail, records(summarize_groups(seed, ["representation"], values))


def _rank(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def contrast_summary(raw, raw_metrics=()):
    frame = pd.DataFrame(raw)
    if frame.empty:
        return frame, pd.DataFrame(), []
    frame["initialization"] = frame.initialization.fillna(-1)
    scale_candidates = {}
    for row in raw_metrics:
        if row.get("normalized_rmse_L0", 0) > 0 and row.get("rmse", 0) > 0:
            scale_candidates.setdefault((row["seed"], row["budget"], row["endpoint"]), []).append(row["rmse"] / row["normalized_rmse_L0"])
    scales = {}
    for key, values in scale_candidates.items():
        if not np.allclose(values, values[0], atol=1e-8, rtol=1e-8):
            raise ValueError("historical L0 scales inferred from result metric ratios disagree")
        scales[key] = float(values[0])
    frame["historical_L0_scale_ml"] = [scales.get((r["seed"], r["budget"], r["endpoint"]), np.nan) for r in raw]
    frame["absolute_true_delta_L0"] = frame.true_delta_ml.abs() / frame.historical_L0_scale_ml
    frame["contrast_absolute_error_L0"] = frame.gp_contrast_absolute_error_ml / frame.historical_L0_scale_ml
    projected = frame.kernel.eq("projected")
    if projected.any() and ("learned_distance_train_rms" not in frame or frame.loc[projected, "learned_distance_train_rms"].isna().any()):
        raise ValueError("projected scientific geometry requires actual learned kernel representation distances")
    frame["effective_distance_train_rms"] = frame.distance_train_rms
    if projected.any():
        frame.loc[projected, "effective_distance_train_rms"] = frame.loc[projected, "learned_distance_train_rms"]
    keys = ["seed", "budget", "representation", "kernel", "initialization", "endpoint", "axis"]
    rows = []
    for key, group in frame.groupby(keys):
        row = dict(zip(keys, key))
        distance_name = "learned_distance_train_rms" if key[3] == "projected" and "learned_distance_train_rms" in group and group.learned_distance_train_rms.notna().any() else "distance_train_rms"
        row.update(pair_count=len(group), gp_contrast_absolute_error_ml=float(group.gp_contrast_absolute_error_ml.mean()),
                   gnn_contrast_absolute_error_ml=float(group.gnn_contrast_absolute_error_ml.mean()),
                   mean_embedding_distance=float(group[distance_name].mean()),
                   distance_definition=distance_name,
                   mean_absolute_response_delta_L0=float(group.absolute_true_delta_L0.mean()),
                   mean_contrast_absolute_error_L0=float(group.contrast_absolute_error_L0.mean()),
                   compressed_pair_count=int((group[distance_name] < 1e-3).sum()),
                   compressed_pair_fraction=float((group[distance_name] < 1e-3).mean()),
                   compressed_mean_absolute_response_delta_L0=float(group.loc[group[distance_name] < 1e-3, "absolute_true_delta_L0"].mean()),
                   compressed_mean_contrast_absolute_error_L0=float(group.loc[group[distance_name] < 1e-3, "contrast_absolute_error_L0"].mean()),
                   mean_absolute_response_delta_ml=float(group.true_delta_ml.abs().mean()),
                   distance_response_rank_spearman=_rank(group[distance_name], group.true_delta_ml.abs()),
                   distance_contrast_error_rank_spearman=_rank(group[distance_name], group.gp_contrast_absolute_error_ml))
        rows.append(row)
    detail = pd.DataFrame(rows)
    values = ["gp_contrast_absolute_error_ml", "gnn_contrast_absolute_error_ml", "mean_embedding_distance",
              "mean_absolute_response_delta_ml", "distance_response_rank_spearman", "distance_contrast_error_rank_spearman",
              "mean_absolute_response_delta_L0", "mean_contrast_absolute_error_L0", "compressed_pair_count", "compressed_pair_fraction",
              "compressed_mean_absolute_response_delta_L0", "compressed_mean_contrast_absolute_error_L0"]
    by_endpoint = mean_groups(detail, ["seed", "budget", "representation", "kernel", "endpoint", "axis"], values)
    by_state = mean_groups(by_endpoint, ["seed", "budget", "representation", "kernel", "axis"], values)
    by_seed = mean_groups(by_state, ["seed", "representation", "kernel", "axis"], values)
    return frame, detail, records(summarize_groups(by_seed, ["representation", "kernel", "axis"], values))


def acquisition_summary(raw):
    batches, comparisons, excluded = [], [], []
    grouped = {}
    for r in raw:
        identity = {key: r.get(key) for key in IDENTITY}
        for name, method in r["methods"].items():
            batches.append({**identity, "acquisition": name, "batch_information": method["batch_information"],
                            "mean_absolute_batch_latent_correlation": method.get("mean_absolute_batch_latent_correlation"),
                            "selected_ids": method.get("selected_ids", []),
                            "selected_candidate_positions": method["selected_candidate_positions"]})
        grouped.setdefault((r["seed"], r["budget"], r["endpoint"]), []).append(r)
    score_names = {"maxvar": "latent_variance", "conditional_ivr": "initial_ivr_scores",
                   "batch_information_gain": "initial_information_scores"}
    for state, group in grouped.items():
        for a, b in combinations(group, 2):
            va, vb = (a["representation"], a["kernel"]), (b["representation"], b["kernel"])
            if va == vb:
                continue
            if a["kernel"] == b["kernel"] == "projected" and a.get("initialization") != b.get("initialization"):
                continue
            if va > vb:
                a, b, va, vb = b, a, vb, va
            pa, pb = a["candidate_positions"], b["candidate_positions"]
            if not a.get("candidate_ids") or not b.get("candidate_ids"):
                excluded.append({"seed": state[0], "budget": state[1], "reason": "candidate IDs unavailable"})
                continue
            ia = [a["candidate_ids"][i] for i in pa]
            ib = [b["candidate_ids"][i] for i in pb]
            if len(ia) != len(set(ia)) or len(ib) != len(set(ib)) or set(ia) != set(ib):
                raise ValueError("acquisition comparison requires identical unique candidate ID universe")
            order = np.array([ib.index(i) for i in ia], dtype=int)
            for method, score in score_names.items():
                sa, sb = np.asarray(a[score], float), np.asarray(b[score], float)[order]
                k = min(32, len(ia))
                ta, tb = set(np.argsort(-sa, kind="stable")[:k]), set(np.argsort(-sb, kind="stable")[:k])
                def selected_ids(r):
                    return {r["candidate_ids"][i] for i in r["methods"][method]["selected_candidate_positions"]}
                selected_a, selected_b = selected_ids(a), selected_ids(b)
                comparison = f"{va[0]}/{va[1]} vs {vb[0]}/{vb[1]}"
                comparisons.append({"seed": state[0], "budget": state[1], "endpoint": state[2],
                    "representation_a": va[0], "kernel_a": va[1], "representation_b": vb[0], "kernel_b": vb[1],
                    "initialization_a": a.get("initialization"), "initialization_b": b.get("initialization"),
                    "comparison": comparison, "acquisition": method, "candidate_count": len(ia),
                    "initial_score_rank_spearman": _rank(sa, sb),
                    "initial_score_top32_overlap": len(ta & tb) / k,
                    "selected_batch_overlap": len(selected_a & selected_b) / max(1, min(len(selected_a), len(selected_b))),
                    "rank_definition": "pre-selection marginal scores; IVR batch is sequentially conditioned"})
    batch_frame, comparison_frame = pd.DataFrame(batches), pd.DataFrame(comparisons)
    def hierarchical(frame, keys, values, endpoint=True):
        if frame.empty:
            return []
        e = mean_groups(frame, ["seed", "budget", "endpoint"] + keys, values) if endpoint else frame
        s = mean_groups(e, ["seed", "budget"] + keys, values)
        seeds = mean_groups(s, ["seed"] + keys, values)
        return records(summarize_groups(seeds, keys, values))
    batch_stats = hierarchical(batch_frame, ["representation", "kernel", "acquisition"],
                               ["batch_information", "mean_absolute_batch_latent_correlation"])
    rank_stats = hierarchical(comparison_frame, ["comparison", "acquisition", "representation_a", "kernel_a", "representation_b", "kernel_b"],
                              ["initial_score_rank_spearman", "initial_score_top32_overlap", "selected_batch_overlap"])
    return batch_frame, comparison_frame, {"batch_summary": batch_stats, "paired_rank_summary": rank_stats,
                                           "excluded_comparisons": excluded,
                                           "interpretation": "model-relative offline signal; information gain does not establish AL benefit"}


def robustness_summary(raw):
    frame = pd.DataFrame(raw)
    if frame.empty:
        return frame, []
    numeric = [c for c in frame if c not in ("seed", "budget") and pd.api.types.is_numeric_dtype(frame[c])]
    seed = mean_groups(frame, ["seed"], numeric)
    rows = {"seed_count": int(seed.seed.nunique())}
    for c in numeric:
        values = seed[c].dropna().to_numpy(float)
        rows[c + "_mean"] = float(values.mean()) if len(values) else None
        rows[c + "_seed_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
    return frame, [rows]


def make_figures(summary, tables, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "savefig.dpi": 220})
    files, layout_records = [], []
    def save(fig, name):
        coverage = summary["coverage"]
        # Reserve an in-canvas footer strip independently of the plot legend.
        engine = fig.get_layout_engine()
        if engine is not None:
            engine.set(rect=(0., .075, 1., .925))
        elif not fig.legends:
            fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, .20))
        footer = fig.text(.01, .012, f"Completed states: {coverage['completed_states']}/{coverage['expected_states'] or '?'}; descriptive development evidence",
                          fontsize=8, color="0.35", va="bottom")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        footer_box = footer.get_window_extent(renderer)
        legend_boxes = [legend.get_window_extent(renderer) for legend in fig.legends]
        overlaps = [i for i, box in enumerate(legend_boxes) if box.overlaps(footer_box)]
        if overlaps:
            raise ValueError(f"footer overlaps figure legend in {name}")
        labels = [axis.xaxis.label for axis in fig.axes]
        label_overlaps = [i for i, label in enumerate(labels) if label.get_visible() and label.get_text()
                          and label.get_window_extent(renderer).overlaps(footer_box)]
        if label_overlaps:
            raise ValueError(f"footer overlaps axis label in {name}")
        layout_records.append(dict(figure=name, figure_size_inches=list(fig.get_size_inches()),
                                   footer_bounds_display=list(footer_box.bounds),
                                   legend_bounds_display=[list(box.bounds) for box in legend_boxes],
                                   footer_legend_overlaps=len(overlaps), footer_axis_label_overlaps=len(label_overlaps)))
        for suffix in ("png", "svg"):
            path = output / f"{name}.{suffix}"
            fig.savefig(path, bbox_inches="tight", facecolor="white")
            files.append(path.name)
        plt.close(fig)
    def label(rep, kernel):
        name = LABELS.get(rep, rep)
        return name + (" + projection" if kernel == "projected" else " (linear)" if kernel == "linear" else "")
    budget = tables["budget_metrics"]
    if len(budget):
        variants = list(budget[["representation", "kernel"]].drop_duplicates().itertuples(index=False, name=None))
        colors = dict(zip(LABELS, plt.get_cmap("tab10").colors))
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
        # Four legend rows and the footer have separate, explicit bottom areas.
        fig.subplots_adjust(left=.08, right=.98, bottom=.26, top=.91, hspace=.45, wspace=.23)
        for ax, metric, title in zip(axes.flat, ["normalized_rmse_L0", "nlpd", "coverage_90", "mean_interval_width_90"],
                                   ["Normalized RMSE (lower is better)", "Predictive NLPD (lower is better)",
                                    "90% interval coverage", "Mean 90% interval width (mL)"]):
            for rep, kernel in variants:
                if kernel == "linear" or metric + "_mean" not in budget:
                    continue
                q = budget[(budget.representation == rep) & (budget.kernel == kernel)].sort_values("budget")
                ax.plot(q.budget, q[metric + "_mean"], marker="o", markersize=4,
                        linestyle="--" if kernel == "projected" else "-", color=colors.get(rep), label=label(rep, kernel))
            if metric == "coverage_90":
                ax.axhline(.9, color="0.35", linestyle=":", linewidth=1)
                ax.set_ylim(0, 1.03)
            ax.set(title=title, xlabel="Observed training experiments")
            ax.grid(alpha=.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.53, .065),
                   ncol=3, frameon=False, fontsize=8, columnspacing=1.5, labelspacing=.6)
        fig.suptitle(f"Development validation: {summary['coverage']['completed_states']} completed states; seed means")
        save(fig, "prediction_and_calibration")
    paired = tables["paired_seed_differences"]
    if len(paired):
        selected = paired[paired.comparison.str.contains("projection - frozen|Fusion projected", regex=True)]
        if len(selected):
            fig, ax = plt.subplots(figsize=(8, 4))
            names = selected.comparison.unique()
            for i, name in enumerate(names):
                group = selected[selected.comparison.eq(name)]
                ax.scatter(group.normalized_rmse_L0_delta, np.full(len(group), i), s=35, alpha=.8)
                ax.plot([group.normalized_rmse_L0_delta.mean()], [i], marker="D", color="black", markersize=7)
            ax.axvline(0, color="0.3", linestyle=":")
            ax.set(yticks=range(len(names)), yticklabels=names, xlabel="Paired normalized RMSE difference (candidate - control)",
                   title="Projection comparisons: each dot is one seed's mean over paired budgets")
            ax.grid(axis="x", alpha=.2)
            save(fig, "paired_projection_effects")
    overall = tables["overall_metrics"]
    if len(overall):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
        fig.subplots_adjust(left=.07, right=.98, bottom=.31, top=.9, wspace=.28)
        colors = dict(zip(LABELS, plt.get_cmap("tab10").colors))
        for _, r in overall.iterrows():
            if r.kernel == "linear":
                continue
            name = label(r.representation, r.kernel)
            color = colors.get(r.representation, "black")
            # Shared colors identify representations; open diamonds identify projection.
            style = dict(s=52, marker="D" if r.kernel == "projected" else "o",
                         facecolors="white" if r.kernel == "projected" else color,
                         edgecolors=color, linewidths=1.6, zorder=3)
            axes[0].scatter(r.mean_interval_width_90_mean, r.coverage_90_mean, label=name, **style)
            axes[1].scatter(r.normalized_rmse_L0_mean, r.nlpd_mean, **style)
        axes[0].axhline(.9, color="0.3", linestyle=":")
        axes[0].set(xlabel="Mean 90% interval width (mL)", ylabel="Empirical 90% coverage", title="Coverage and sharpness")
        axes[1].set(xlabel="Normalized RMSE", ylabel="NLPD", title="Point and probability prediction")
        for ax in axes:
            ax.grid(alpha=.15)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.53, .055), ncol=3,
                   frameon=False, fontsize=8, labelspacing=.6, columnspacing=1.5)
        save(fig, "calibration_tradeoffs")
    rank = pd.DataFrame(summary["acquisition"]["paired_rank_summary"])
    if len(rank):
        fixed = rank[(rank.kernel_a == "matern52") & (rank.kernel_b == "matern52")]
        names = sorted(set(fixed.representation_a) | set(fixed.representation_b))
        if names:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
            for ax, method in zip(axes, ("maxvar", "conditional_ivr")):
                matrix = np.full((len(names), len(names)), np.nan)
                np.fill_diagonal(matrix, 1.)
                for _, r in fixed[fixed.acquisition.eq(method)].iterrows():
                    i, j = names.index(r.representation_a), names.index(r.representation_b)
                    matrix[i, j] = matrix[j, i] = r.selected_batch_overlap_mean
                image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
                for i in range(len(names)):
                    for j in range(len(names)):
                        if np.isfinite(matrix[i, j]):
                            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", color="white" if matrix[i,j] < .65 else "black", fontsize=8)
                ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=[LABELS[n] for n in names],
                       yticklabels=[LABELS[n] for n in names], title=method.replace("_", " ") + ": batch overlap")
                ax.tick_params(axis="x", rotation=65)
            fig.colorbar(image, ax=axes, shrink=.7, label="Shared fraction of selected batch")
            save(fig, "offline_acquisition_overlap")
    points = tables.get("geometry_scatter_points", pd.DataFrame())
    if len(points):
        from matplotlib.colors import LogNorm
        reps = [r for r in ("structured", "gnn", "raw", "cw", "llm_conditions") if r in set(points.representation)]
        axes_names = [a for a in ("sample_volume", "pe") if a in set(points.axis)]
        if reps and axes_names:
            # Dedicated title rows keep condition names separate from axis labels.
            fig = plt.figure(figsize=(3.2 * len(reps), 3.4 * len(axes_names) + .5), constrained_layout=True)
            grid = fig.add_gridspec(2 * len(axes_names), len(reps) + 1,
                                   height_ratios=[value for _ in axes_names for value in (.12, 1.)],
                                   width_ratios=[1.] * len(reps) + [.055])
            axes = np.empty((len(axes_names), len(reps)), dtype=object)
            for i, axis_name in enumerate(axes_names):
                row_title = fig.add_subplot(grid[2 * i, :-1])
                row_title.set_axis_off()
                row_title.text(.5, .5, "Sample volume" if axis_name == "sample_volume" else "PE/EA",
                               ha="center", va="center", fontsize=11, fontweight="semibold")
                for j in range(len(reps)):
                    axes[i, j] = fig.add_subplot(grid[2 * i + 1, j])
            positive = points.contrast_absolute_error_L0.dropna().clip(lower=1e-6)
            norm = LogNorm(vmin=float(positive.min()), vmax=max(float(positive.max()), float(positive.min()) * 1.01))
            for i, axis_name in enumerate(axes_names):
                for j, rep in enumerate(reps):
                    group = points[(points.axis == axis_name) & (points.representation == rep)]
                    ax = axes[i, j]
                    image = ax.scatter(group.effective_distance_train_rms.clip(lower=1e-8),
                                       group.absolute_true_delta_L0.clip(lower=1e-6),
                                       c=group.contrast_absolute_error_L0.clip(lower=1e-6), cmap="magma", norm=norm,
                                       s=15, alpha=.65, edgecolors="none")
                    ax.set(xscale="log", yscale="log", title=LABELS.get(rep, rep),
                           xlabel="Distance / training RMS", ylabel="|Δresponse| / L0 SD" if j == 0 else "")
                    ax.axvline(1e-3, color="0.5", linestyle=":", linewidth=.8)
            colorbar_axis = fig.add_subplot(grid[:, -1])
            fig.colorbar(image, cax=colorbar_axis, label="Contrast absolute error / L0 SD")
            fig.suptitle(f"Held-out one-condition pairs at L{int(points.budget.max())}; repeated IDs across seeds collapsed by median")
            save(fig, "scientific_distance_response")
    dump(output / "figure_layout_qa.json", layout_records)
    return files


def summarize(study_dir=HERE, output_dir=None, *, plots=True):
    study = Path(study_dir).resolve()
    output = Path(output_dir).resolve() if output_dir else study / "summary"
    if output == study / "results" or (study / "results") in output.parents:
        raise ValueError("summary output cannot overwrite model result directories")
    output.mkdir(parents=True, exist_ok=True)
    config, raw, coverage, sources = load_completed(study)
    tables = metric_tables(raw["metrics"])
    pair_state, pair_seed, pairs = paired_tables(raw["metrics"])
    tables["paired_state_differences"], tables["paired_seed_differences"] = pair_state, pair_seed
    fits, optimizers = optimizer_summary(raw["fits"])
    tables["fits"] = fits
    init, init_stats = initialization_summary(raw["metrics"])
    tables["projection_initialization_variation"] = init
    contrast, contrast_state, contrast_stats = contrast_summary(raw["contrasts"], raw["metrics"])
    tables["contrasts"], tables["contrast_state_summary"] = contrast, contrast_state
    geometry = pd.DataFrame()
    if len(contrast):
        latest = int(contrast.budget.max())
        fixed = contrast[(contrast.budget == latest) & contrast.kernel.eq("matern52")]
        keys = ["budget", "representation", "kernel", "axis", "endpoint", "id_a", "id_b"]
        values = ["effective_distance_train_rms", "absolute_true_delta_L0", "contrast_absolute_error_L0"]
        geometry = fixed.groupby(keys, dropna=False)[values].median().reset_index()
        occurrences = fixed.groupby(keys, dropna=False).seed.nunique().rename("seed_occurrences").reset_index()
        geometry = geometry.merge(occurrences, on=keys)
    tables["geometry_scatter_points"] = geometry
    batches, ranks, acquisition = acquisition_summary(raw["acquisitions"])
    tables["acquisition_batches"], tables["acquisition_pairwise_ranks"] = batches, ranks
    robust, robust_stats = robustness_summary(raw["robustness"])
    tables["robustness"] = robust
    summary = dict(schema="representation-gp-summary-v1", coverage=coverage,
                   aggregation="mean initializations, mean endpoints, mean available budgets within each seed, equal-weight mean seeds",
                   statistical_unit="development seed; shared data and repeated budgets are not independent",
                   evaluation="previously used checkpoint-selection validation; development evidence only",
                   primary_metric=config.get("primary_metric", "normalized_rmse_L0"),
                   overall_metrics=records(tables["overall_metrics"]), budget_metrics=records(tables["budget_metrics"]),
                   endpoint_budget_metrics=records(tables["endpoint_budget_metrics"]), paired_comparisons=pairs,
                   state_metrics=records(tables["state_metrics"]), seed_metrics=records(tables["seed_metrics"]),
                   paired_seed_differences=records(pair_seed), robustness_states=records(robust),
                   optimizer=optimizers, projection_initialization=init_stats, contrast=contrast_stats,
                   acquisition=acquisition, text_robustness=robust_stats,
                   scientific_geometry=dict(compressed_distance_threshold=1e-3,
                       compressed_threshold_role="descriptive geometric diagnostic only; not a preregistered performance endpoint or method-selection rule",
                       scale_source="historical L0 scales recovered only from completed result rmse/normalized_rmse_L0; consistency checked across methods",
                       statistical_summary="within state: initialization mean then endpoint mean; budgets within seed; equal seed means",
                       scatter_budget=int(geometry.budget.max()) if len(geometry) else None,
                       scatter_deduplication="one representation/axis/endpoint/ID pair at latest completed budget; median across seeds",
                       scatter_plot_floors=dict(distance=1e-8, normalized_response=1e-6, normalized_error_color=1e-6)),
                   source_files=sources, test_truth_reads=0, partial_results=not coverage["complete_matrix"])
    for name, frame in tables.items():
        export_csv(frame, output / (name + ".csv"))
    summary["figures"] = make_figures(summary, tables, output) if plots and coverage["completed_states"] else []
    dump(output / "summary.json", summary)
    dump(output / "source_coverage.json", coverage)
    return clean(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=HERE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = summarize(args.study_dir, args.output_dir, plots=not args.no_plots)
    print(json.dumps({"completed_states": result["coverage"]["completed_states"], "partial_results": result["partial_results"],
                      "output": str(args.output_dir or args.study_dir / "summary")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
