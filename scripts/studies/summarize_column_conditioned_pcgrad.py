#!/usr/bin/env python3
"""Generate the terminal report for the controlled A2-PCGrad study."""
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

from scripts.studies import run_column_conditioned_pcgrad as run


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    values = frame.copy()
    for column in values.select_dtypes(include=[np.number]).columns:
        values[column] = values[column].map(lambda value: f"{value:.{digits}f}")
    header = "| " + " | ".join(values.columns) + " |"
    rule = "| " + " | ".join("---" for _ in values.columns) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in values.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def paired_summary(results: pd.DataFrame, tails: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in run.TASKS[1:]:
        candidate = results.loc[(results.column == column) & (results.method == run.METHOD)].set_index(["outer_seed", "inner_fold"])
        reference = results.loc[(results.column == column) & (results.method == run.REFERENCE_METHOD)].set_index(["outer_seed", "inner_fold"]).loc[candidate.index]
        nrmse_gain = 1 - candidate.combined_normalized_rmse / reference.combined_normalized_rmse
        row = {"protocol": "compound", "evidence": "INNER_CV", "column": column,
               "candidate": run.METHOD, "reference": run.REFERENCE_METHOD,
               "mean_paired_fold_nrmse_gain": float(nrmse_gain.mean()),
               "seed_wins": int((candidate.combined_normalized_rmse.groupby(level=0).mean() < reference.combined_normalized_rmse.groupby(level=0).mean()).sum()),
               "fold_wins": int((nrmse_gain > 0).sum())}
        for metric in ("V1_rmse", "V1_mae", "V2_rmse", "V2_mae", "Center_rmse", "Width_rmse"):
            row[f"{metric}_relative_gain"] = float(1 - candidate[metric].mean() / reference[metric].mean())
        for endpoint in ("V1", "V2"):
            selected = tails.loc[(tails.column == column) & (tails.endpoint == endpoint)]
            grouped = selected.groupby(["outer_seed", "method"])[["tail_sse", "tail_count"]].sum()
            rmses = np.sqrt(grouped.tail_sse / grouped.tail_count).unstack("method")
            row[f"tail_{endpoint}_rmse_relative_gain"] = float(1 - rmses[run.METHOD].mean() / rmses[run.REFERENCE_METHOD].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def gradient_pair_summary(raw: pd.DataFrame, projected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair in ("4g_25g", "4g_40g", "25g_40g"):
        for stage, frame in (("raw", raw), ("projected", projected)):
            values = frame[f"cos_{pair}"].to_numpy(float)
            rows.append({"pair": pair, "stage": stage, "observations": len(values),
                         "mean_cosine": float(values.mean()), "median_cosine": float(np.median(values)),
                         "negative_fraction": float((values < 0).mean()),
                         "negative_burden": float(np.maximum(-values, 0).mean())})
    return pd.DataFrame(rows)


def summarize() -> str:
    torch.set_num_threads(1)
    run.prepare()
    run.execution_provenance()
    decision = json.loads((run.STUDY / "INNER_DECISION.json").read_text())
    audit = json.loads((run.STUDY / "EVIDENCE_AUDIT.json").read_text())
    if decision["passed"] or audit["status"] != "PASS":
        raise RuntimeError("this terminal summarizer expects the audited failed COMPOUND gate")
    results = pd.read_csv(run.STUDY / "INNER_RESULTS.csv")
    tails = pd.read_csv(run.STUDY / "TAIL_METRICS.csv")
    raw = pd.read_csv(run.STUDY / "GRADIENT_RAW.csv")
    projected = pd.read_csv(run.STUDY / "GRADIENT_PROJECTED.csv")
    paired = paired_summary(results, tails)
    paired.to_csv(run.STUDY / "PAIRED_COMPARISONS.csv", index=False)
    target = results.loc[results.column.isin(run.TASKS[1:])]
    metrics = ("V1_rmse", "V1_mae", "V2_rmse", "V2_mae", "combined_normalized_rmse", "V1_r2", "V2_r2", "Center_rmse", "Width_rmse")
    mean_metrics = target.groupby(["column", "method"])[list(metrics)].mean().reset_index()
    gradients = gradient_pair_summary(raw, projected)
    gradients.to_csv(run.STUDY / "GRADIENT_PAIR_SUMMARY.csv", index=False)
    convergence = target.loc[target.method == run.METHOD].drop_duplicates(["outer_seed", "inner_fold"])[["best_epoch", "epochs_run"]]
    mechanism = decision["mechanism"]
    column_text = {}
    for row in paired.itertuples(index=False):
        column_text[row.column] = (
            f"{100 * row.mean_paired_fold_nrmse_gain:+.2f}% mean paired-fold NRMSE gain, "
            f"{row.seed_wins}/5 seed wins, {row.fold_wins}/25 fold wins; "
            f"V1/V2 RMSE gains {100 * row.V1_rmse_relative_gain:+.2f}%/{100 * row.V2_rmse_relative_gain:+.2f}%, "
            f"MAE gains {100 * row.V1_mae_relative_gain:+.2f}%/{100 * row.V2_mae_relative_gain:+.2f}%, "
            f"tail gains {100 * row.tail_V1_rmse_relative_gain:+.2f}%/{100 * row.tail_V2_rmse_relative_gain:+.2f}%"
        )
    report = f"""# Controlled shared-backbone PCGrad: final report

Terminal decision: `PCGRAD_REDUCES_CONFLICT_BUT_DOES_NOT_MATERIALLY_IMPROVE_TRANSFER`.

PCGrad materially changed the diagnosed mechanism but failed the preregistered
COMPOUND prediction gate. ROW and new outer developmental confirmation were
therefore intentionally not run, and no prediction-freeze or outer-result
artifact was created.

## Scope and evidence boundary

`A2_ADAM_FROZEN` is the completed historical A2 column-FiLM reference.
`A2_PCGRAD` is the sole new candidate and is architecture-identical. Only the
44 parameters in the completed shared late-backbone diagnostic tuple receive
projected gradients; every other trainable parameter receives the ordinary
mean-loss gradient. Twenty-five COMPOUND inner fits were completed on the exact
historical identities, scales, source, loss, batch schedule, and selector.

Independent evidence audit: `{audit['status']}`; fits verified
{audit['fits_verified']}, target fold observations {audit['target_fold_observations']},
prediction rows {audit['prediction_rows_verified']}, global target-molecule
train/validation overlap {audit['global_target_group_overlap']}.

## Mechanism result

{markdown_table(gradients)}

Across all pairs and diagnostic steps, mean cosine changed from
{mechanism['raw_mean_cosine']:.4f} to {mechanism['projected_mean_cosine']:.4f}
(increase {mechanism['mean_cosine_increase']:.4f}). Negative-cosine burden fell
{100 * mechanism['negative_cosine_burden_relative_reduction']:.2f}%, and the
negative fraction fell from {100 * mechanism['raw_negative_fraction']:.2f}% to
{100 * mechanism['projected_negative_fraction']:.2f}%. PCGrad triggered on
{100 * mechanism['projection_trigger_fraction']:.2f}% of directed attempts.
The projected/original mean-update norm ratio averaged
{mechanism['mean_projected_original_update_norm_ratio']:.3f}; the median raw
task-gradient magnitude ratio was {mechanism['raw_median_magnitude_ratio']:.3f},
below the preregistered severe-imbalance threshold of 10. Thus PCGrad genuinely
altered the measured conflict rather than acting as an inert control.

## COMPOUND inner prediction result

{markdown_table(mean_metrics)}

{markdown_table(paired[['column', 'mean_paired_fold_nrmse_gain', 'seed_wins', 'fold_wins', 'V1_rmse_relative_gain', 'V1_mae_relative_gain', 'V2_rmse_relative_gain', 'V2_mae_relative_gain', 'tail_V1_rmse_relative_gain', 'tail_V2_rmse_relative_gain']])}

Positive gain means A2-PCGrad is better. 25g: {column_text['25g']}.
40g: {column_text['40g']}. The 25g V2 tail deterioration is 2.99%, exceeding
the 2% guard. Neither column reaches the required 2% mean NRMSE improvement;
25g also misses the fold-win and tail guards. The mechanism checks all pass,
but both predictive column gates fail.

## Convergence

The 25 candidate fits selected mean best epoch {convergence.best_epoch.mean():.1f}
(range {int(convergence.best_epoch.min())}-{int(convergence.best_epoch.max())})
and ran a mean {convergence.epochs_run.mean():.1f} epochs (range
{int(convergence.epochs_run.min())}-{int(convergence.epochs_run.max())}).
No learning-rate, epoch-budget, loss, or other search was performed.

## Required interpretation

1. Did PCGrad materially reduce negative gradient cosine? Yes. Mean cosine rose by {mechanism['mean_cosine_increase']:.3f}, negative burden fell {100 * mechanism['negative_cosine_burden_relative_reduction']:.1f}%, and all preregistered mechanism checks passed.

2. Did raw gradient magnitude imbalance remain modest? Yes. The median raw maximum/minimum task-gradient norm ratio was {mechanism['raw_median_magnitude_ratio']:.2f}, well below 10; this study provides no GradNorm rationale.

3. Did reducing conflict improve inner generalization? No materially replicated improvement occurred. 25g worsened 0.37% in mean paired-fold NRMSE and 40g improved only 0.50%, both below the required 2%.

4. Did it reduce the inner-to-outer gain collapse? Not testable: the failed COMPOUND inner gate correctly blocked new outer prediction and scoring.

5. Did it repair the prior 40g ROW deterioration? Not testable: ROW was prohibited after COMPOUND failure. The mechanism result alone cannot establish ROW repair.

6. Did RMSE improve at the expense of MAE? No consistent trade occurred. On 25g, V1 RMSE and both MAEs improved slightly while V2 RMSE worsened; on 40g all four improved by less than 1%. None produced material NRMSE gain.

7. Did the high-volume tail improve? Not consistently. 25g V1 tail worsened 0.92% and V2 worsened 2.99%; 40g V1 improved 0.71% and V2 worsened 0.42%.

8. Did PCGrad improve both columns? No. 25g mean NRMSE worsened; 40g's 0.50% gain was sub-threshold.

9. Did it improve both V1 and V2? Only 40g showed small RMSE gains for both endpoints. 25g was mixed, with V2 RMSE worse.

10. Does the candidate now exceed A1 on ROW? Unknown and deliberately unscored; ROW was not authorized.

11. Does it exceed corrected HIER on COMPOUND? Unknown and deliberately unscored; outer confirmation was not authorized.

12. What limitation remains? Conflict was real and was almost eliminated without material prediction gain, so conflict alone is insufficient. The remaining evidence is more consistent with a combination of limited/partly confounded column context, data identifiability, and high-volume-tail scarcity; this controlled study cannot separate them causally.

13. Is another model-side experiment justified? No uncontrolled neural architecture or optimizer expansion is justified under the current dataset. Independent/crossed compound and batch data with intentional tail coverage should precede another model-side mechanism study. No next method was implemented.

## Provenance and terminal rule

Qualified source SHA256: `{run.base.SOURCE_SHA}`. Split SHA256:
`{run.base.SPLIT_SHA}`. Protocol SHA256: `{run.sha(run.STUDY / 'protocol.json')}`.
Decision SHA256: `{run.sha(run.STUDY / 'INNER_DECISION.json')}`.

The reference artifact hashes, architecture signature, trainable parameter
identities, PCGrad scope, exact inner identities, scales, task losses, raw and
projected diagnostics, and completion hashes are retained. Outer validation or
test truth was not read. No ROW fit, new outer prediction, post-result method,
GradNorm/CAGrad/NashMTL, architecture change, calibration, 8g, physical
descriptor, or Active Learning action was performed.
"""
    (run.STUDY / "FINAL_REPORT.md").write_text(report)
    terminal = {"terminal_decision": "PCGRAD_REDUCES_CONFLICT_BUT_DOES_NOT_MATERIALLY_IMPROVE_TRANSFER",
                "compound_inner_gate_passed": False, "mechanism_gate_passed": True,
                "row_run": False, "outer_run": False, "outer_truth_read": False,
                "additional_candidates_authorized": False,
                "recommendation": "STOP uncontrolled neural architecture and optimizer expansion under the current dataset; prioritize independent/crossed data and tail coverage"}
    run.write_json(run.STUDY / "PROJECT_DECISION.json", terminal, immutable=True)
    files = ("FINAL_REPORT.md", "PROJECT_DECISION.json", "PAIRED_COMPARISONS.csv", "GRADIENT_PAIR_SUMMARY.csv", "EVIDENCE_AUDIT.json")
    run.write_json(run.STUDY / "REPORT_MANIFEST.json", {"files": {name: run.sha(run.STUDY / name) for name in files},
                   "next_stage_executed": False}, immutable=True)
    print(report)
    return report


if __name__ == "__main__":
    summarize()
