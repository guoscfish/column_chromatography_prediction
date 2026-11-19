"""Post-freeze diagnostics of saved, legally observed L333 OOF predictions.

No model fit, calibration adjustment, threshold selection, or U/validation/test
target read is performed. Bootstrap intervals are descriptive sensitivity only.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import kurtosis, norm, spearmanr


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
OUT = HERE / "qualification/failure_analysis"
OUT.mkdir(parents=True, exist_ok=True)


def wilson(k, n, z=1.959963984540054):
    p = k / n
    center = (p + z*z / (2*n)) / (1 + z*z / n)
    radius = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return center - radius, center + radius


summary, conditional, parameters, joint, audits, coverage_curves = [], [], [], [], [], []
for directory in sorted((HERE / "qualification").glob("seed_*_*")):
    if not (directory / "complete.json").exists():
        continue
    receipt = json.loads((directory / "complete.json").read_text())
    if receipt["freeze_sha256"] != hashlib.sha256((HERE / "freeze.json").read_bytes()).hexdigest():
        raise RuntimeError("qualification freeze receipt drift")
    for filename, expected in receipt["files"].items():
        if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"qualification completed output drift: {directory / filename}")
    _, seed_string, family = directory.name.split("_", 2)
    seed = int(seed_string)
    frame = pd.read_csv(directory / "oof_predictions.csv", dtype={"sample_id": str})
    metrics = json.loads((directory / "metrics.json").read_text())
    identities = pd.read_csv(ROOT / f"studies/active_learning/qgeognn_v2_row_sequential_b32/splits/row_seed_{seed}.csv",
                             usecols=["sample_id", "canonical_smiles", "role"], dtype={"sample_id": str})
    identities = identities.loc[identities.role.eq("l0")]
    if not set(frame.sample_id) <= set(identities.sample_id):
        raise PermissionError("diagnostic prediction contains non-L333 row")
    frame = frame.merge(identities[["sample_id", "canonical_smiles"]], on="sample_id", validate="many_to_one")
    for endpoint, rows in frame.groupby("endpoint", sort=False):
        residual = (rows.truth_ml - rows.mean_ml).to_numpy()
        sigma = np.sqrt(rows.variance_ml2.to_numpy())
        z = residual / sigma
        abs_z = np.abs(z)
        for nominal in np.arange(.05, 1, .025):
            coverage_curves.append({"seed": seed, "family": family, "endpoint": endpoint,
                                    "nominal": nominal, "observed": np.mean(abs_z <= ndtri((1+nominal)/2))})
        covered80 = abs_z <= ndtri(.9)
        covered95 = abs_z <= ndtri(.975)
        count80, n = int(covered80.sum()), len(z)
        low, high = wilson(count80, n)
        grouped = pd.DataFrame({"molecule": rows.canonical_smiles.to_numpy(), "hit": covered80}).groupby("molecule").agg(hits=("hit", "sum"), rows=("hit", "size"))
        rng = np.random.default_rng(seed + 970023)
        choices = rng.integers(0, len(grouped), size=(2000, len(grouped)))
        bootstrap = grouped.hits.to_numpy()[choices].sum(1) / grouped.rows.to_numpy()[choices].sum(1)
        boot_low, boot_high = np.quantile(bootstrap, [.025, .975])
        order = np.argsort(-np.square(z), kind="stable")
        top_n = int(np.ceil(.01*n))
        q50, q80, q95, q99 = np.quantile(abs_z, [.5, .8, .95, .99])
        summary.append({"seed": seed, "family": family, "endpoint": endpoint, "rows": n,
                        "molecules": len(grouped), "coverage80_hits": count80, "coverage80": count80/n,
                        "coverage80_excess_vs_gate": count80/n - .9,
                        "rows_reclassified_needed_to_reach_gate": max(0, count80 - int(np.floor(.9*n))),
                        "coverage80_wilson95_iid_lower": low, "coverage80_wilson95_iid_upper": high,
                        "coverage80_cluster_bootstrap95_lower": boot_low, "coverage80_cluster_bootstrap95_upper": boot_high,
                        "coverage95": covered95.mean(), "z_mean": z.mean(), "z_rms": np.sqrt(np.mean(z*z)),
                        "z_excess_kurtosis": kurtosis(z, fisher=True, bias=False),
                        "abs_z_q50": q50, "abs_z_q80": q80, "abs_z_q95": q95, "abs_z_q99": q99,
                        "q80_width_multiplier_descriptive_only": q80 / ndtri(.9),
                        "q95_width_multiplier_descriptive_only": q95 / ndtri(.975),
                        "top_ceil1percent_rows": top_n, "top_ceil1percent_fraction_z_squared": (z[order[:top_n]]**2).sum() / (z*z).sum(),
                        "tail_fraction_abs_z_gt3": np.mean(abs_z > 3), "tail_fraction_abs_z_gt4": np.mean(abs_z > 4),
                        "max_abs_z": abs_z.max(), "sigma_max_over_min": sigma.max()/sigma.min(),
                        "sigma_abs_error_spearman": spearmanr(sigma, np.abs(residual)).statistic})
        for name, ordering in (("predicted_sigma", np.argsort(sigma, kind="stable")),
                               ("predicted_mean", np.argsort(rows.mean_ml.to_numpy(), kind="stable"))):
            for quartile, positions in enumerate(np.array_split(ordering, 4), 1):
                conditional.append({"seed": seed, "family": family, "endpoint": endpoint, "stratification": name,
                                    "quartile": quartile, "rows": len(positions),
                                    "coverage50": np.mean(abs_z[positions] <= ndtri(.75)),
                                    "coverage80": covered80[positions].mean(), "coverage95": covered95[positions].mean(),
                                    "z_rms": np.sqrt(np.mean(z[positions]**2)), "mean_sigma_ml": sigma[positions].mean(),
                                    "mean_prediction_ml": rows.mean_ml.to_numpy()[positions].mean(),
                                    "rmse_ml": np.sqrt(np.mean(residual[positions]**2))})
    for fold in metrics["folds"]:
        model = fold["model"]
        joint.append({"seed": seed, "family": family, "fold": fold["fold"],
                      **{k: fold[k] for k in ("full_joint_nll_per_scalar", "diagonal_nll_per_scalar", "block_diagonal_nll_per_scalar", "whitened_residual_rms", "minimum_predictive_eigenvalue", "mean_absolute_offdiagonal_predictive_correlation")},
                      "cross_row_joint_nll_difference": fold["full_joint_nll_per_scalar"] - fold["block_diagonal_nll_per_scalar"]})
        if "parameters" in model:
            for output, p in enumerate(model["parameters"], 1):
                parameters.append({"seed": seed, "family": family, "fold": fold["fold"], "endpoint": f"V{output}",
                                   "pca_retained_fraction": model["pca_retained_standardized_input_variance_fraction"],
                                   **{k: p[k] for k in ("noise_variance_scaled", "signal_variance", "lengthscale", "noise_bound_hit", "optimizer_success", "nit")}})
        else:
            parameters.append({"seed": seed, "family": family, "fold": fold["fold"], "endpoint": "joint",
                               "pca_retained_fraction": model["pca_retained_standardized_input_variance_fraction"],
                               "noise_variance_scaled": model["noise_multiplier"], "signal_variance": 1/model["prior_precision"],
                               "lengthscale": None, "noise_bound_hit": model["noise_bound_hit"],
                               "optimizer_success": model["optimizer_success"], "nit": model["nit"]})
    for name in ("complete.json", "metrics.json", "oof_predictions.csv"):
        path = directory / name
        audits.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

for name, data in (("residual_shape", summary), ("conditional_calibration", conditional),
                   ("fitted_parameters", parameters), ("joint_dependence", joint), ("input_hashes", audits),
                   ("coverage_curves", coverage_curves)):
    pd.DataFrame(data).to_csv(OUT / f"{name}.csv", index=False)

s, c, p, j = map(pd.DataFrame, (summary, conditional, parameters, joint))
stats = {"residual_shape_ranges": {name: [float(s[name].min()), float(s[name].max())]
                                  for name in ("z_excess_kurtosis", "top_ceil1percent_fraction_z_squared", "abs_z_q50", "abs_z_q80", "abs_z_q95", "max_abs_z")},
         "pca_retained_range": [p.pca_retained_fraction.min(), p.pca_retained_fraction.max()],
         "noise_bound_hits": int(p.noise_bound_hit.sum()), "optimizer_failures": int((~p.optimizer_success).sum()),
         "coverage80_failures_out_of40": int((s.coverage80 > .9).sum()),
         "coverage80_failures_iid_interval_includes90": int(((s.coverage80 > .9) & (s.coverage80_wilson95_iid_lower <= .9)).sum()),
         "coverage80_failures_cluster_interval_includes90": int(((s.coverage80 > .9) & (s.coverage80_cluster_bootstrap95_lower <= .9)).sum()),
         "coverage80_iid_interval_excludes_nominal80": int((s.coverage80_wilson95_iid_lower > .8).sum()),
         "coverage80_cluster_interval_excludes_nominal80": int((s.coverage80_cluster_bootstrap95_lower > .8).sum()),
         "conditional_sigma_coverage80_range": [float(c.loc[c.stratification.eq("predicted_sigma"), "coverage80"].min()), float(c.loc[c.stratification.eq("predicted_sigma"), "coverage80"].max())],
         "conditional_sigma_rms_range": [float(c.loc[c.stratification.eq("predicted_sigma"), "z_rms"].min()), float(c.loc[c.stratification.eq("predicted_sigma"), "z_rms"].max())],
         "gp_noise_scaled_range": [p.loc[p.family.str.startswith("gp_"), "noise_variance_scaled"].min(), p.loc[p.family.str.startswith("gp_"), "noise_variance_scaled"].max()],
         "gp_lengthscale_range": [p.loc[p.family.str.startswith("gp_"), "lengthscale"].min(), p.loc[p.family.str.startswith("gp_"), "lengthscale"].max()],
         "joint_cross_row_nll_range": [j.cross_row_joint_nll_difference.min(), j.cross_row_joint_nll_difference.max()],
         "gaussian_reference": {"abs_z_q50": norm.ppf(.75), "abs_z_q80": norm.ppf(.9), "abs_z_q95": norm.ppf(.975), "abs_z_q99": norm.ppf(.995)},
         "new_model_fits": 0, "future_U_labels_read": 0, "validation_labels_read": 0, "test_labels_read": 0,
         "bootstrap": "2000 molecule-cluster resamples of fixed OOF predictions; exploratory descriptive intervals, no model refitting"}
(OUT / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
curves = pd.DataFrame(coverage_curves)
for family, grouped in curves.groupby("family", sort=False):
    aggregate = grouped.groupby("nominal")["observed"].agg(["mean", "min", "max"])
    line, = axes[0].plot(aggregate.index, aggregate["mean"], label=family, linewidth=1.8)
    axes[0].fill_between(aggregate.index, aggregate["min"], aggregate["max"], color=line.get_color(), alpha=.10)
axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
axes[0].set(xlabel="Nominal central coverage", ylabel="Observed OOF coverage", title="Peaked residual centers; tail coverage differs", xlim=(0, 1), ylim=(0, 1))
axes[0].legend(fontsize=8, loc="lower right")
for i, (family, grouped) in enumerate(s.groupby("family", sort=False)):
    axes[1].scatter(np.full(len(grouped), i), grouped.top_ceil1percent_fraction_z_squared,
                    s=25, alpha=.7, label=family)
axes[1].set(xticks=range(4), xticklabels=list(s.groupby("family", sort=False).groups), ylabel="Fraction of standardized squared error",
            title="Largest four of 333 rows contribute 32–42%", ylim=(0, .5))
axes[1].tick_params(axis="x", rotation=25, labelsize=8)
fig.suptitle("L333 train-only OOF diagnostics: no additional model fits", fontsize=12)
fig.savefig(OUT / "probability_calibration_diagnosis.png", dpi=180)
plt.close(fig)
print(json.dumps(stats, indent=2))
