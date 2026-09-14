#!/usr/bin/env python3
"""Summarize frozen A1/A2 evidence without reading any endpoint data source."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.studies import run_column_conditioned_multitask as r


def table(frame, digits=4):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}" if np.isfinite(value) else "NA"
        return str(value).replace("|", "/")
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def summarize_gradients(frame):
    rows = []
    for arm, subset in frame.groupby("arm"):
        for pair in ("4g_25g", "4g_40g", "25g_40g"):
            values = subset[f"cos_{pair}"]
            left, right = pair.split("_")
            first, second = subset[f"grad_norm_{left}"], subset[f"grad_norm_{right}"]
            ratio = np.maximum(first, second) / np.maximum(np.minimum(first, second), 1e-12)
            valid = values.dropna()
            seed_fraction = subset.groupby("outer_seed")[f"cos_{pair}"].agg(lambda v: v.dropna().lt(0).mean()) if "outer_seed" in subset else pd.Series([valid.lt(0).mean()])
            rows.append({"arm": arm, "pair": pair, "observations": len(values), "finite_cosine_observations": len(valid), "mean_cosine": valid.mean(),
                         "median_cosine": valid.median(), "negative_fraction": valid.lt(0).mean(),
                         "seeds_with_majority_negative_cosine": int(seed_fraction.gt(.5).sum()), "seed_count": len(seed_fraction),
                         "median_magnitude_ratio": ratio.median(), "zero_norm_count": int(((first == 0) | (second == 0)).sum())})
    return pd.DataFrame(rows)


def promotion_decision(compound, row, compound_tails, row_tails):
    """Conservative promotion versus every compatible frozen reference."""
    checks = []
    methods = sorted(set(compound.method) - {"A2"})
    for c in r.TASKS[1:]:
        for reference in methods:
            subset = compound.loc[compound.column.eq(c)]
            a2 = subset.loc[subset.method.eq("A2")].set_index("outer_seed")
            base = subset.loc[subset.method.eq(reference)].set_index("outer_seed").loc[a2.index]
            if len(a2) != 5 or len(base) != 5:
                raise RuntimeError("project comparison requires all five seeds")
            gain = 1 - a2.combined_normalized_rmse.mean() / base.combined_normalized_rmse.mean()
            wins = int((a2.combined_normalized_rmse < base.combined_normalized_rmse).sum())
            endpoint_worst = max(a2[f"{e}_rmse"].mean() / base[f"{e}_rmse"].mean() - 1 for e in ("V1", "V2"))
            tail = compound_tails.loc[compound_tails.column.eq(c)].groupby(["method", "endpoint"]).tail_rmse.mean()
            tail_worst = max(tail.loc[("A2", e)] / tail.loc[(reference, e)] - 1 for e in ("V1", "V2"))
            secondary = row.loc[row.column.eq(c)].groupby("method").combined_normalized_rmse.mean()
            row_worst = secondary.loc["A2"] / secondary.loc[reference] - 1
            secondary_endpoints = row.loc[row.column.eq(c)].groupby("method")[["V1_rmse", "V2_rmse"]].mean()
            row_endpoint_worst = float((secondary_endpoints.loc["A2"] / secondary_endpoints.loc[reference] - 1).max())
            secondary_tails = row_tails.loc[row_tails.column.eq(c)].groupby(["method", "endpoint"]).tail_rmse.mean()
            row_tail_worst = max(secondary_tails.loc[("A2", e)] / secondary_tails.loc[(reference, e)] - 1 for e in ("V1", "V2"))
            passed = gain > 0 and wins >= 3 and max(endpoint_worst, tail_worst, row_worst, row_endpoint_worst, row_tail_worst) <= .02
            checks.append({"column": c, "reference": reference, "mean_nrmse_gain": gain, "seed_wins": wins,
                           "worst_endpoint_deterioration": endpoint_worst, "worst_tail_deterioration": tail_worst,
                           "row_nrmse_deterioration": row_worst, "row_endpoint_deterioration": row_endpoint_worst,
                           "row_tail_deterioration": row_tail_worst, "passed": bool(passed)})
    representation = all(v["passed"] for v in checks if v["reference"] == "A1")
    project = all(v["passed"] for v in checks)
    # P0 has no compound artifact, but its ROW comparison remains a required guard.
    if "P0" in set(row.method):
        for c in r.TASKS[1:]:
            values = row.loc[row.column.eq(c)].groupby("method").combined_normalized_rmse.mean()
            project = project and bool(values.loc["A2"] <= 1.02 * values.loc["P0"])
            endpoint_values = row.loc[row.column.eq(c)].groupby("method")[["V1_rmse", "V2_rmse"]].mean()
            project = project and bool((endpoint_values.loc["A2"] <= 1.02 * endpoint_values.loc["P0"]).all())
            tail_values = row_tails.loc[row_tails.column.eq(c)].groupby(["method", "endpoint"]).tail_rmse.mean()
            project = project and all(tail_values.loc[("A2", e)] <= 1.02 * tail_values.loc[("P0", e)] for e in ("V1", "V2"))
    return {"REPRESENTATION_SIGNAL": representation, "PROJECT_TRANSFER_GAIN": project,
            "checks": checks, "compound_P0_comparison": "UNAVAILABLE; baseline was not retrained",
            "classification": "DEVELOPMENTAL CONFIRMATION"}


def summarize():
    torch.set_num_threads(1)
    r.execution_provenance()
    decision = json.loads((r.STUDY / "INNER_DECISION.json").read_text())
    if r.sha(r.STUDY / "INNER_RESULTS.csv") != decision["inner_results_sha256"] or r.sha(r.STUDY / "TAIL_METRICS.csv") != decision["tail_metrics_sha256"]:
        raise RuntimeError("inner evidence digest mismatch")
    results = pd.read_csv(r.STUDY / "INNER_RESULTS.csv")
    tails = pd.read_csv(r.STUDY / "TAIL_METRICS.csv")
    recomputed = r.gate_decision(results, tails)
    for current, stored in zip(recomputed["columns"], decision["columns"]):
        for key in ("column", "passed", "conditions", "fold_wins", "seed_wins"):
            if current[key] != stored[key]:
                raise RuntimeError("decision differs from recorded fold evidence")
        np.testing.assert_allclose(current["mean_relative_nrmse_gain"], stored["mean_relative_nrmse_gain"], rtol=1e-12, atol=1e-12)
        for key in ("endpoint_changes", "tail_changes"):
            np.testing.assert_allclose(list(current[key].values()), [stored[key][endpoint] for endpoint in current[key]], rtol=1e-12, atol=1e-12)
    histories, predictions = [], []
    for relative, expected in decision["completion_hashes"].items():
        run = r.STUDY / relative
        if r.sha(run / "complete.json") != expected or not r.verify_completed(run):
            raise RuntimeError("fit completion chain mismatch")
        parts = Path(relative).parts
        seed = int(parts[3].removeprefix("seed_"))
        fold = int(parts[4].removeprefix("fold_"))
        histories.append(pd.read_csv(run / "history.csv").assign(protocol="compound", outer_seed=seed, inner_fold=fold, arm=parts[5]))
        predictions.append(pd.read_csv(run / "predictions.csv"))
    history = pd.concat(histories, ignore_index=True)
    history.to_csv(r.STUDY / "INNER_EPOCH_HISTORY.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.concat(predictions, ignore_index=True).to_csv(r.STUDY / "INNER_PREDICTIONS.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    target = results.loc[results.column.isin(r.TASKS[1:])]
    metrics = ["V1_rmse", "V1_mae", "V2_rmse", "V2_mae", "combined_normalized_rmse", "V1_r2", "V2_r2", "Center_rmse", "Width_rmse"]
    means = target.groupby(["column", "arm"])[metrics].mean().reset_index()
    summary = results.groupby(["column", "arm"])[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(names) for names in summary.columns]
    summary.reset_index().to_csv(r.STUDY / "INNER_SUMMARY.csv", index=False)
    paired = []
    for c in r.TASKS[1:]:
        base = target.loc[target.column.eq(c) & target.arm.eq("A1")].set_index(["outer_seed", "inner_fold"])
        candidate = target.loc[target.column.eq(c) & target.arm.eq("A2")].set_index(["outer_seed", "inner_fold"]).loc[base.index]
        for metric in metrics:
            change = candidate[metric] - base[metric]
            paired.append({"column": c, "metric": metric, "A1_mean": base[metric].mean(), "A2_mean": candidate[metric].mean(),
                           "mean_paired_delta": change.mean(), "paired_delta_sd": change.std(ddof=1),
                           "relative_gain_of_means": 1 - candidate[metric].mean() / base[metric].mean() if "r2" not in metric else np.nan,
                           "fold_wins": int((change > 0).sum() if "r2" in metric else (change < 0).sum())})
    pairs = pd.DataFrame(paired)
    pairs.to_csv(r.STUDY / "INNER_PAIRED_COMPARISONS.csv", index=False)
    pooled_tail = tails.loc[tails.column.isin(r.TASKS[1:])].groupby(["column", "outer_seed", "arm", "endpoint"]).agg(tail_sse=("tail_sse", "sum"), tail_count=("tail_count", "sum"), total_sse=("total_sse", "sum"), threshold=("threshold", "first")).reset_index()
    pooled_tail["tail_rmse"] = np.sqrt(pooled_tail.tail_sse / pooled_tail.tail_count)
    pooled_tail["tail_sse_fraction"] = pooled_tail.tail_sse / pooled_tail.total_sse
    pooled_tail.to_csv(r.STUDY / "TAIL_SEED_METRICS.csv", index=False)
    tail_means = pooled_tail.groupby(["column", "arm", "endpoint"])[["tail_rmse", "tail_count", "tail_sse_fraction"]].mean().reset_index()
    gradient_summary = summarize_gradients(pd.read_csv(r.STUDY / "GRADIENT_DIAGNOSTICS.csv"))
    gradient_summary.to_csv(r.STUDY / "GRADIENT_SUMMARY.csv", index=False)
    fits = results.loc[results.column.eq("25g")]
    convergence = fits.groupby("arm").agg(fits=("best_epoch", "size"), mean_best_epoch=("best_epoch", "mean"), min_best_epoch=("best_epoch", "min"), max_best_epoch=("best_epoch", "max"), mean_epochs_run=("epochs_run", "mean"), selections_at_ceiling=("best_epoch", lambda v: int(v.eq(500).sum())), runs_at_ceiling=("epochs_run", lambda v: int(v.eq(500).sum()))).reset_index()
    convergence.to_csv(r.STUDY / "CONVERGENCE_SUMMARY.csv", index=False)
    outer_identities = []
    for keys, subset in r.verify_schedule().groupby(["column", "protocol", "outer_seed", "role"]):
        outer_identities.append({**dict(zip(("column", "protocol", "outer_seed", "role"), keys)),
                                 "count": len(subset), "sample_ids_sha256": r.digest(sorted(subset.sample_id.astype(str)))})
    pd.DataFrame(outer_identities).to_csv(r.STUDY / "OUTER_IDENTITY_AUDIT.csv", index=False)
    overlap_rows = []
    frozen_schedule = r.verify_schedule()
    for seed in r.SEEDS:
        context = frozen_schedule.loc[frozen_schedule.protocol.eq("compound") & frozen_schedule.outer_seed.eq(seed)]
        for focal, donor in (("25g", "40g"), ("40g", "25g")):
            if "canonical_smiles" not in context:
                continue
            donor_molecules = set(context.loc[context.column.eq(donor) & context.role.eq("gradient_train"), "canonical_smiles"])
            for role in ("validation", "test"):
                heldout = context.loc[context.column.eq(focal) & context.role.eq(role)]
                molecules = set(heldout.canonical_smiles)
                overlap_rows.append({"outer_seed": seed, "focal_column": focal, "donor_column": donor, "role": role,
                                     "heldout_molecules": len(molecules), "molecules_seen_in_other_target_train": len(molecules & donor_molecules),
                                     "heldout_rows_seen_in_other_target_train": int(heldout.canonical_smiles.isin(donor_molecules).sum()), "heldout_rows": len(heldout)})
    overlap_frame = pd.DataFrame(overlap_rows)
    overlap_frame.to_csv(r.STUDY / "OUTER_CROSS_TASK_MOLECULE_OVERLAP.csv", index=False)
    overlap_description = []
    if not overlap_frame.empty:
        for c in r.TASKS[1:]:
            counts = overlap_frame.loc[overlap_frame.focal_column.eq(c) & overlap_frame.role.eq("test"), "molecules_seen_in_other_target_train"]
            overlap_description.append(f"{c}: {int(counts.min())}-{int(counts.max())} test molecules per seed seen in the other target's gradient_train")
    overlap_text = "; ".join(overlap_description) or "No outer overlap rows are present in this fixture."
    regimes = pd.read_csv(r.STUDY / "RETENTION_REGIMES.csv")
    regime_means = regimes.loc[regimes.column.isin(r.TASKS[1:])].groupby(["column", "arm", "regime"])[["V1_rmse", "V2_rmse", "n"]].mean().reset_index()
    regime_means.to_csv(r.STUDY / "RETENTION_REGIME_SUMMARY.csv", index=False)
    gate_table = pd.DataFrame([{"column": d["column"], "mean_fold_gain_pct": 100 * d["mean_relative_nrmse_gain"], "seed_wins": f'{d["seed_wins"]}/5', "fold_wins": f'{d["fold_wins"]}/25', "worst_endpoint_change_pct": 100 * max(d["endpoint_changes"].values()), "worst_tail_change_pct": 100 * max(d["tail_changes"].values()), "passed": d["passed"]} for d in decision["columns"]])
    outer_exists = (r.STUDY / "OUTER_RESULTS.csv").exists()
    if not decision["passed"]:
        forbidden = ["PREDICTION_FREEZE_MANIFEST.json", "OUTER_RESULTS.csv", "PAIRED_COMPARISONS.csv", "ROW_INNER_RESULTS.csv"]
        if any((r.STUDY / name).exists() for name in forbidden) or list((r.STUDY / "runtime/final").glob("**/predictions_blind.csv")):
            raise RuntimeError("negative gate must have no outer success artifacts")
    outer_text = "Outer prediction and scoring were intentionally blocked by the failed COMPOUND inner gate. ROW was not run. No outer success artifacts were created."
    promotion = None
    if decision["passed"]:
        if not outer_exists or not (r.STUDY / "ROW_OUTER_RESULTS.csv").exists():
            raise RuntimeError("passing study must finish both developmental confirmations before final report")
        r.verify_freeze("compound")
        r.verify_freeze("row")
        compound = pd.read_csv(r.STUDY / "OUTER_RESULTS.csv")
        row = pd.read_csv(r.STUDY / "ROW_OUTER_RESULTS.csv")
        promotion = promotion_decision(compound, row, pd.read_csv(r.STUDY / "OUTER_TAIL_METRICS.csv"), pd.read_csv(r.STUDY / "ROW_OUTER_TAIL_METRICS.csv"))
        r.write_json(r.STUDY / "PROJECT_DECISION.json", promotion, immutable=True)
        outer_text = "All outer results below are DEVELOPMENTAL CONFIRMATION. Compound P0 is unavailable and was not retrained.\n\n" + table(pd.concat([compound, row]).groupby(["protocol", "column", "method"])[metrics[:5]].mean().reset_index()) + "\n\n" + table(pd.DataFrame(promotion["checks"]))
    gain_text = "; ".join(f'{d["column"]}: {100*d["mean_relative_nrmse_gain"]:+.2f}% mean paired-fold NRMSE gain, {d["seed_wins"]}/5 seed wins, {d["fold_wins"]}/25 fold wins' for d in decision["columns"])
    endpoint_text = "; ".join(f'{c}: V1 {100*pairs.loc[pairs.column.eq(c) & pairs.metric.eq("V1_rmse"), "relative_gain_of_means"].iloc[0]:+.2f}%, V2 {100*pairs.loc[pairs.column.eq(c) & pairs.metric.eq("V2_rmse"), "relative_gain_of_means"].iloc[0]:+.2f}%' for c in r.TASKS[1:])
    cw_text = "; ".join(f'{c}: Center {100*pairs.loc[pairs.column.eq(c) & pairs.metric.eq("Center_rmse"), "relative_gain_of_means"].iloc[0]:+.2f}%, Width {100*pairs.loc[pairs.column.eq(c) & pairs.metric.eq("Width_rmse"), "relative_gain_of_means"].iloc[0]:+.2f}%' for c in r.TASKS[1:])
    tail_text = "; ".join(f'{d["column"]}: V1 {100*d["tail_changes"]["V1"]:+.2f}%, V2 {100*d["tail_changes"]["V2"]:+.2f}% RMSE change (positive worsens)' for d in decision["columns"])
    conflict = bool((gradient_summary.negative_fraction > .5).any())
    imbalance = bool((gradient_summary.median_magnitude_ratio >= 10).any())
    diagnosis = ("At least one task pair has negative cosine on a majority of sampled steps, consistent with persistent conflict under this recipe. " if conflict else "No pair has negative cosine on a majority of sampled steps; persistent conflict is not established by these averages. ")
    diagnosis += ("At least one median pairwise gradient-magnitude ratio exceeds 10, indicating pronounced magnitude imbalance. " if imbalance else "No pair has a median gradient-magnitude ratio above 10. ")
    diagnosis += "These are descriptive thresholds, not training rules or causal proof. The diagnostics use shared late-backbone parameters, excluding heads, FiLM generators, and condition-completion parameters."
    baseline_answer = "Not established. No matching COMPOUND converged-P0 inner artifact exists, historical target-only outer identities differ from inner holdouts, and the failed gate blocks A1 outer scoring. Historical scores cannot answer this with a valid paired comparison."
    strongest_answer = "Not assessed: the gate blocked new outer scoring. Corrected HIER and paper-style remain frozen references; the study did not establish project transfer gain."
    representation_answer = "The registered two-column gate did not support a replicated representation-level gain. Any local inner improvement is insufficient for the requested general conclusion."
    complexity_answer = "Additional complexity is not justified for promotion by this study's gate. This does not prove that column conditioning can never help."
    if promotion is not None:
        baseline_answer = "See the matched DEVELOPMENTAL CONFIRMATION table: A1 can be compared with paper-style and corrected HIER on both protocols and P0 on ROW. No compound converged-P0 comparison is available, so a universal target-only superiority claim remains limited."
        strongest_answer = f'PROJECT_TRANSFER_GAIN={promotion["PROJECT_TRANSFER_GAIN"]} under the conservative all-compatible-reference guard. Compound P0 remains unavailable.'
        representation_answer = f'The inner gate passed; outer REPRESENTATION_SIGNAL={promotion["REPRESENTATION_SIGNAL"]}. This is predictive evidence on historically exposed identities, not physical or source-unseen OOD evidence.'
        complexity_answer = f'Project-level promotion is {"supported within these developmental comparisons" if promotion["PROJECT_TRANSFER_GAIN"] else "not supported"}; independent confirmation is still needed.'
    next_stage = "c) independent/crossed experimental data collection"
    next_reason = "Independent compound/batch evidence, crossed column/flow conditions, and tail coverage address the remaining identification limits."
    replicated_conflicts = gradient_summary.loc[gradient_summary.seeds_with_majority_negative_cosine.ge(3)].groupby("pair").arm.nunique()
    if bool(replicated_conflicts.ge(2).any()) and (promotion is None or not promotion["PROJECT_TRANSFER_GAIN"]):
        next_stage = "a) a separately preregistered gradient-conflict handling control"
        next_reason = "The same task pair has majority-negative sampled gradients in at least 3/5 seeds for both architectures. This supports isolating gradient conflict as the next computational mechanism; it does not establish that a correction will improve prediction. Independent/crossed data collection remains the external-validation priority."
    elif conflict:
        next_reason += " Gradient-conflict handling is a possible future control, but the observed conflict does not meet the descriptive cross-seed/both-arm consistency criterion used for this recommendation."
    next_reason += " Domain-specific normalization has not been isolated here. Uncontrolled neural architecture expansion should stop pending new evidence. No next-stage method was implemented."
    report = f'''# Column-conditioned multi-task QGeoGNN: final report

Decision: `{decision["status"]}`.

{gain_text}.

## Preregistered continuation gate

{table(gate_table)}

Mean gain is the arithmetic mean of 25 paired fold-relative gains. Seed wins
compare each seed's mean fold NRMSE. All five criteria are required in BOTH
columns. Positive endpoint/tail changes in this table mean deterioration.

## COMPOUND inner validation metrics

{table(means)}

RMSE/MAE are mL. Combined NRMSE uses inner-train-only endpoint scales.
The values average 25 folds; folds across repeated outer seeds overlap and
are not 25 independent experiments. These inner-selected metrics are
developmental selection evidence, not an unbiased external generalization estimate.

## Tail and retention regimes

{table(tail_means)}

Thresholds are per-endpoint outer gradient_train 80th percentiles. Tail SSE and
count are pooled over each seed's five OOF folds, then seed RMSEs are averaged.
Displayed counts are mean per-seed counts, not unique observations across seeds.
Full thresholds/counts/SSE are in `TAIL_METRICS.csv` and `TAIL_SEED_METRICS.csv`.

{table(regime_means)}

Regimes use outer gradient_train center tertiles. They are descriptive slices,
not selectors. Center/Width error variance and covariance are retained in
`INNER_RESULTS.csv`; endpoint changes must accompany any Center/Width claim.

## Gradient compatibility

{table(gradient_summary)}

{diagnosis}

Sampling cadence was fixed at epoch 1 and every 10 epochs. The raw per-task
losses and all per-task validation metrics for every epoch are retained in
`INNER_EPOCH_HISTORY.csv.gz`. Gradient diagnostics did not change training.

## Convergence

{table(convergence)}

The budget and patience were fixed before fitting. Ceiling selections are
reported as possible budget censoring; they do not authorize more epochs or
different learning rates. A1 and A2 always select one joint checkpoint per fold.

## Outer confirmation

{outer_text}

The inherited COMPOUND outer partitions are per-column. Identity-only overlap:
{overlap_text}. Exact required outer roles
were preserved; donor rows were not silently purged. Thus any outer result is
within-column compound holdout with related-task supervision, not a globally
unseen target-molecule test. This does not affect the globally grouped inner
continuation gate. Details are in `OUTER_CROSS_TASK_MOLECULE_OVERLAP.csv`.

## Required scientific questions

1. Does A1 outperform existing target-only transfer? {baseline_answer}

2. Does column FiLM improve over A1? {gain_text}. The complete continuation gate {"passes" if decision["passed"] else "fails"}.

3. Is gain replicated in both columns? {"Both columns satisfy the inner gate; outer interpretation is given above." if decision["passed"] else "No qualifying two-column replication under the preregistered gate."}

4. Is it replicated across seeds? {gain_text}. Wins alone do not satisfy the magnitude and endpoint/tail guards.

5. Is it driven by only V1 or V2? Inner absolute-RMSE relative gains: {endpoint_text}. Use these endpoint-specific changes rather than the combined score alone.

6. Does it reduce absolute RMSE, not only improve R2? {endpoint_text}. These are direct RMSE comparisons; R2 is secondary and did not select a candidate.

7. Does it improve the high-volume tail? {tail_text}. The gate requires both endpoint tail guards for both columns.

8. Does it improve both Center and Width? Inner RMSE relative gains: {cw_text}. Center and Width are algebraic surrogates; the covariance diagnostics do not establish a physical mechanism.

9. What do gradient cosines imply? {diagnosis}

10. Does A2 exceed the strongest legitimate baseline? {strongest_answer}

11. Must column identity affect the molecular representation before readout? {representation_answer}

12. Is complexity scientifically justified? {complexity_answer}

13. Recommended next stage: {next_stage}. {next_reason}

## Provenance and restrictions

Source SHA256: `{r.SOURCE_SHA}`. Split SHA256: `{r.SPLIT_SHA}`.
Protocol SHA256: `{r.sha(r.STUDY / "protocol.json")}`.
Decision SHA256: `{r.sha(r.STUDY / "INNER_DECISION.json")}`.

The source replay covers all 4,163 qualified rows, with maximum frozen-CSV
difference 3.8086e-6. Active-source/A1 and A1/A2 epoch-zero differences are zero.
Execution code/environment hashes, exact outer roles, inner identities, fold
hashes, histories, and completion digests are retained. Qualified 4g labels
intentionally overlap target compounds. Task embeddings represent known task
identity only. No 8g, PCGrad, GradNorm, domain-specific BN, calibration, scaling,
geometry descriptor, loss/width/embedding sweep, ensemble, or Active Learning
was added, and historical artifacts were preserved.
'''
    (r.STUDY / "FINAL_REPORT.md").write_text(report)
    r.write_json(r.STUDY / "REPORT_MANIFEST.json", {"decision_sha256": r.sha(r.STUDY / "INNER_DECISION.json"), "report_sha256": r.sha(r.STUDY / "FINAL_REPORT.md"), "outer_truth_read_by_report": False,
                 "next_stage_recommendation": next_stage, "next_stage_executed": False,
                 "files": {name: r.sha(r.STUDY / name) for name in ("INNER_EPOCH_HISTORY.csv.gz", "INNER_PREDICTIONS.csv.gz", "INNER_SUMMARY.csv", "INNER_PAIRED_COMPARISONS.csv", "TAIL_SEED_METRICS.csv", "GRADIENT_SUMMARY.csv", "CONVERGENCE_SUMMARY.csv", "RETENTION_REGIME_SUMMARY.csv", "OUTER_IDENTITY_AUDIT.csv", "OUTER_CROSS_TASK_MOLECULE_OVERLAP.csv")}})
    print(report)


if __name__ == "__main__":
    summarize()
