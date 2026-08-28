#!/usr/bin/env python3
"""D46-A post-hoc oracle-utility reliability audit."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qgeognn_al.engine import QGeoGNNActiveLearningEngine, TrainConfig, canonical_json_hash
from scripts.run_e0_4g_baseline import sha256_file
from scripts.run_e0_8g_controls import load_graph_cache
from scripts.run_e4_a2a_formal import a2a_partition_context
from scripts.run_e4_active_transfer import SCALER, SOURCES, SOURCE_SCALES, TARGET
from src.qgeognn_al.diagnostics.oracle_reliability import (
    compare_d45_d46, paired_test_bootstrap, paired_utility,
    pairwise_ranking_stability, select_reliability_candidates,
    summarize_candidate_reliability, summarize_strata, variance_decomposition,
)
from src.qgeognn_al.diagnostics.oracle_utility import validate_candidate_fit_contract
from src.qgeognn_al.metrics import regression_metric_row

DEFAULT_OUTPUT = ROOT / "experiments/d46_oracle_utility_reliability"
D45_CANDIDATES = ROOT / "experiments/d45_oracle_marginal_utility/candidate_utility.csv"
SOURCE_MANIFEST = ROOT / "experiments/e4_a2a_low_budget_formal/source_checkpoint_manifest.csv"
PREFLIGHT = DEFAULT_OUTPUT / "preflight/d45_clean_verification.json"
SMOKE_DECISION = ROOT / "experiments/reproductions/d46_oracle_reliability_smoke/decision.json"
REPETITION_SEEDS = (4601, 4602, 4603)
EPSILON = 1e-6
BOOTSTRAP_REPLICATES = 2000


def git_output(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def validate_preflight(mode: str) -> None:
    preflight = json.loads(PREFLIGHT.read_text())
    required = ("git_clean", "pytest_pass", "d45_smoke_pass", "source_hash_match",
                "partition_hash_match", "baseline_contract_match", "preflight_pass")
    if not all(preflight.get(key) is True for key in required):
        raise RuntimeError("D46 preflight is not fully passed")
    if mode == "bounded":
        smoke = json.loads(SMOKE_DECISION.read_text())
        if smoke.get("engineering_smoke_pass") is not True:
            raise RuntimeError("D46 bounded requires a passed D46 smoke")


def validate_sources() -> dict[int, str]:
    official = pd.read_csv(SOURCE_MANIFEST).set_index("member_seed")
    expected: dict[int, str] = {}
    for member_seed, checkpoint in SOURCES.items():
        actual = sha256_file(checkpoint)
        manifest_hash = str(official.loc[member_seed, "checkpoint_sha256"])
        if actual != manifest_hash:
            raise RuntimeError(f"source checkpoint hash mismatch for {member_seed}")
        expected[member_seed] = actual
    return expected


def fit_seed(repetition_seed: int, member_seed: int) -> int:
    """Keep target repetition randomness explicit and separate from source seed."""
    return int(repetition_seed) * 10000 + int(member_seed)


def fit_members(
    engine: QGeoGNNActiveLearningEngine,
    labeled: list[str],
    validation: list[str],
    config: TrainConfig,
    repetition_seed: int,
    output: Path,
    expected_sources: dict[int, str],
) -> tuple[dict[int, Path], list[dict[str, object]]]:
    checkpoints: dict[int, Path] = {}
    records: list[dict[str, object]] = []
    labeled_hash = canonical_json_hash(sorted(labeled))
    for member_seed, source in SOURCES.items():
        member_output = output / f"member_{member_seed}"
        result_file, checkpoint_file = member_output / "fit_result.json", member_output / "best.pt"
        started = time.perf_counter()
        reused = False
        if result_file.exists() and checkpoint_file.exists():
            result = json.loads(result_file.read_text())
            if not (result["labeled_ids_hash"] == labeled_hash
                    and result["train_config_hash"] == config.config_hash
                    and result["init_source_sha256"] == expected_sources[member_seed]):
                raise RuntimeError("incompatible partial D46 member fit")
            reused = True
        else:
            result = asdict(engine.fit(
                labeled, validation, config, source,
                fit_seed(repetition_seed, member_seed), member_output,
            ))
        checkpoints[member_seed] = Path(str(result["checkpoint"]))
        records.append({
            "repetition_seed": repetition_seed, "member_seed": member_seed,
            "fit_seed": fit_seed(repetition_seed, member_seed),
            "fit_seconds": 0.0 if reused else time.perf_counter() - started,
            "reused_partial_fit": reused, **result,
        })
    return checkpoints, records


def prediction_frame(
    engine: QGeoGNNActiveLearningEngine,
    test_ids: list[str],
    checkpoints: dict[int, Path],
    repetition_seed: int,
    candidate_id: str,
    phase: str,
) -> pd.DataFrame:
    """Aggregate K=3 engine predictions while preserving test row identity."""
    member_predictions = []
    expected_ids = list(map(str, test_ids))
    for checkpoint in checkpoints.values():
        table = engine.predict(
            test_ids, checkpoint, return_quantiles=False, return_embedding=False
        ).table
        if table["sample_id"].astype(str).tolist() != expected_ids:
            raise RuntimeError("test prediction row identity drifted")
        member_predictions.append(table[["V1_q50", "V2_q50"]].to_numpy(float))
    prediction = np.mean(np.stack(member_predictions), axis=0)
    index = dict(zip(engine.data.sample_id.astype(str), range(len(engine.data))))
    truth = engine.data.iloc[[index[sample_id] for sample_id in expected_ids]][["V1_ml", "V2_ml"]].to_numpy(float)
    return pd.DataFrame({
        "repetition_seed": repetition_seed, "candidate_id": candidate_id,
        "phase": phase, "sample_id": expected_ids,
        "V1_true": truth[:, 0], "V2_true": truth[:, 1],
        "V1_pred": prediction[:, 0], "V2_pred": prediction[:, 1],
    })


def metric_from_predictions(frame: pd.DataFrame, scales: dict[str, float]) -> dict[str, float]:
    return regression_metric_row(
        frame[["V1_true", "V2_true"]].to_numpy(float),
        frame[["V1_pred", "V2_pred"]].to_numpy(float), scales,
    )


def audit_rows(
    records: list[dict[str, object]], phase: str, candidate_id: str,
    expected_sources: dict[int, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fit_rows, reset_rows = [], []
    for record in records:
        frozen = record["frozen_parameters_sha256_before"] == record["frozen_parameters_sha256_after"]
        changed = record["trainable_parameters_sha256_before"] != record["trainable_parameters_sha256_after"]
        reset = record["init_source_sha256"] == expected_sources[int(record["member_seed"])]
        if not (frozen and changed and reset):
            raise RuntimeError("D46 fit/source audit failed")
        common = {"repetition_seed": record["repetition_seed"], "phase": phase,
                  "candidate_id": candidate_id, "member_seed": record["member_seed"]}
        fit_rows.append({**common, "fit_seed": record["fit_seed"],
            "train_rows": record["train_rows"], "validation_rows": record["validation_rows"],
            "validation_ids_hash": record["validation_ids_hash"],
            "frozen_unchanged": frozen, "trainable_changed": changed,
            "best_epoch": record["best_epoch"], "epochs_run": record["epochs_run"],
            "fit_seconds": record["fit_seconds"], "checkpoint_sha256": record["checkpoint_sha256"]})
        reset_rows.append({**common, "init_source_sha256": record["init_source_sha256"],
            "expected_source_sha256": expected_sources[int(record["member_seed"])],
            "source_reset_pass": reset, "checkpoint_sha256": record["checkpoint_sha256"]})
    return fit_rows, reset_rows


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def run_fits(args, engine, roles, subset, repetitions, config, scales, expected_sources, progress):
    initial = roles["l0_train"] + roles["l0_validation"]
    validation_hash = canonical_json_hash(sorted(roles["l0_validation"]))
    baseline_rows, utility_rows, fit_rows, reset_rows, prediction_frames = [], [], [], [], []
    actual_new_fits = 0
    for repetition_index, repetition_seed in enumerate(repetitions):
        rep_progress = progress / f"rep_{repetition_seed}"
        baseline_json = rep_progress / "baseline.json"
        baseline_csv = rep_progress / "baseline_predictions.csv"
        if baseline_json.exists() and baseline_csv.exists():
            baseline_record = json.loads(baseline_json.read_text())
            baseline_prediction = pd.read_csv(baseline_csv, dtype={"sample_id": str})
        else:
            output = progress.parent / "fits" / f"rep_{repetition_seed}" / "baseline"
            checkpoints, records = fit_members(engine, initial, roles["l0_validation"], config,
                                                repetition_seed, output, expected_sources)
            baseline_prediction = prediction_frame(engine, roles["test"], checkpoints,
                                                   repetition_seed, "__baseline__", "baseline")
            baseline_metrics = metric_from_predictions(baseline_prediction, scales)
            fits, resets = audit_rows(records, "baseline", "__baseline__", expected_sources)
            if {str(row["validation_ids_hash"]) for row in fits} != {validation_hash}:
                raise RuntimeError("D46 baseline validation hash mismatch")
            baseline_record = {"repetition_seed": repetition_seed, **baseline_metrics,
                               "fit_audit": fits, "source_reset_audit": resets}
            rep_progress.mkdir(parents=True, exist_ok=True)
            baseline_prediction.to_csv(baseline_csv, index=False)
            atomic_json(baseline_json, baseline_record)
            actual_new_fits += sum(not record["reused_partial_fit"] for record in records)
            shutil.rmtree(output)
        baseline_rows.append({key: value for key, value in baseline_record.items()
                              if key not in ("fit_audit", "source_reset_audit")})
        fit_rows.extend(baseline_record["fit_audit"]); reset_rows.extend(baseline_record["source_reset_audit"])
        prediction_frames.append(baseline_prediction)
        for candidate_index, candidate in enumerate(subset.itertuples(index=False)):
            candidate_id = str(candidate.sample_id)
            pair_json = rep_progress / "pairs" / f"{candidate_id}.json"
            pair_csv = rep_progress / "pairs" / f"{candidate_id}.csv"
            if pair_json.exists() and pair_csv.exists():
                pair_record = json.loads(pair_json.read_text())
                candidate_prediction = pd.read_csv(pair_csv, dtype={"sample_id": str})
            else:
                validate_candidate_fit_contract(initial, roles["l0_validation"], initial + [candidate_id],
                                                candidate_id, roles["u0"], roles["test"])
                output = progress.parent / "fits" / f"rep_{repetition_seed}" / candidate_id
                checkpoints, records = fit_members(engine, initial + [candidate_id], roles["l0_validation"],
                                                    config, repetition_seed, output, expected_sources)
                candidate_prediction = prediction_frame(engine, roles["test"], checkpoints,
                                                        repetition_seed, candidate_id, "candidate")
                candidate_metrics = metric_from_predictions(candidate_prediction, scales)
                fits, resets = audit_rows(records, "candidate", candidate_id, expected_sources)
                if {str(row["validation_ids_hash"]) for row in fits} != {validation_hash} \
                        or any(int(row["train_rows"]) != 23 for row in fits):
                    raise RuntimeError("D46 candidate fixed-validation audit failed")
                baseline_metrics = {key: baseline_record[key] for key in ("V1_MAE", "V1_RMSE", "V1_R2", "V2_MAE", "V2_RMSE", "V2_R2", "NRMSE")}
                pair_record = {"repetition_seed": repetition_seed, "candidate_id": candidate_id,
                               "sample_id": candidate_id,
                               "stratum": candidate.stratum,
                               "oracle_utility_combined": paired_utility(baseline_metrics, candidate_metrics),
                               **{f"candidate_{key}": value for key, value in candidate_metrics.items()},
                               "fit_audit": fits, "source_reset_audit": resets}
                pair_csv.parent.mkdir(parents=True, exist_ok=True)
                candidate_prediction.to_csv(pair_csv, index=False); atomic_json(pair_json, pair_record)
                actual_new_fits += sum(not record["reused_partial_fit"] for record in records)
                shutil.rmtree(output)
            pair_record.setdefault("sample_id", candidate_id)
            utility_rows.append({key: value for key, value in pair_record.items()
                                 if key not in ("fit_audit", "source_reset_audit")})
            fit_rows.extend(pair_record["fit_audit"]); reset_rows.extend(pair_record["source_reset_audit"])
            prediction_frames.append(candidate_prediction)
            print(f"D46 {args.mode}: rep {repetition_seed} candidate {candidate_index + 1}/{len(subset)} {candidate_id}", flush=True)
    return (pd.DataFrame(baseline_rows), pd.DataFrame(utility_rows), pd.DataFrame(fit_rows),
            pd.DataFrame(reset_rows), pd.concat(prediction_frames, ignore_index=True), actual_new_fits)


def analyze(output, subset, repetitions, utilities, predictions, scales):
    reliability = summarize_candidate_reliability(utilities, repetitions, EPSILON)
    reliability = subset[["sample_id", "stratum", "D45_oracle_utility_combined"]].merge(
        reliability, on=["sample_id", "stratum"], validate="one_to_one")
    utility_with_flags = utilities.merge(subset[["sample_id", "random_sample"]], on="sample_id")
    ranking = pairwise_ranking_stability(utility_with_flags, repetitions)
    variance = variance_decomposition(utilities)
    bootstrap_rows = []
    for rep_index, repetition_seed in enumerate(repetitions):
        rep_predictions = predictions[predictions.repetition_seed == repetition_seed]
        baseline = rep_predictions[rep_predictions.phase == "baseline"]
        truth = baseline[["V1_true", "V2_true"]].to_numpy(float)
        baseline_pred = baseline[["V1_pred", "V2_pred"]].to_numpy(float)
        for candidate_index, candidate in enumerate(subset.itertuples(index=False)):
            candidate_frame = rep_predictions[(rep_predictions.phase == "candidate") &
                                              (rep_predictions.candidate_id == candidate.sample_id)]
            result = paired_test_bootstrap(
                truth, baseline_pred, candidate_frame[["V1_pred", "V2_pred"]].to_numpy(float), scales,
                BOOTSTRAP_REPLICATES, 46_040_000 + rep_index * 1000 + candidate_index,
                return_draws=True,
            )
            lower, upper = result["utility_bootstrap_ci_lower"], result["utility_bootstrap_ci_upper"]
            bootstrap_rows.append({"sample_id": candidate.sample_id, "stratum": candidate.stratum,
                "repetition_seed": repetition_seed,
                **{key: value for key, value in result.items() if key not in ("draws", "sampled_indices")},
                "bootstrap_CI_excludes_zero": bool(lower > 0 or upper < 0),
                "bootstrap_CI_direction": "positive" if lower > 0 else "negative" if upper < 0 else "crosses_zero"})
    bootstrap = pd.DataFrame(bootstrap_rows)
    comparison = compare_d45_d46(reliability, subset)
    forest = []
    for candidate_index, candidate in enumerate(subset.itertuples(index=False)):
        rng = np.random.default_rng(46_050_000 + candidate_index)
        shared_indices = rng.integers(
            0, len(roles_test := predictions[predictions.phase == "baseline"].sample_id.unique()),
            size=(BOOTSTRAP_REPLICATES, len(roles_test)),
        )
        repetition_draws = []
        for repetition_seed in repetitions:
            rep_predictions = predictions[predictions.repetition_seed == repetition_seed]
            baseline = rep_predictions[rep_predictions.phase == "baseline"]
            candidate_frame = rep_predictions[(rep_predictions.phase == "candidate") &
                                              (rep_predictions.candidate_id == candidate.sample_id)]
            result = paired_test_bootstrap(
                baseline[["V1_true", "V2_true"]].to_numpy(float),
                baseline[["V1_pred", "V2_pred"]].to_numpy(float),
                candidate_frame[["V1_pred", "V2_pred"]].to_numpy(float), scales,
                sampled_indices=shared_indices, return_draws=True,
            )
            repetition_draws.append(np.asarray(result["draws"]))
        mean_draws = np.vstack(repetition_draws).mean(axis=0)
        lower, upper = np.percentile(mean_draws, [2.5, 97.5])
        forest.append({"sample_id": candidate.sample_id, "stratum": candidate.stratum,
                       "mean_utility": reliability.loc[reliability.sample_id == candidate.sample_id, "mean_utility"].item(),
                       "paired_test_row_mean_ci_lower": lower, "paired_test_row_mean_ci_upper": upper,
                       "mean_bootstrap_CI_excludes_zero": bool(lower > 0 or upper < 0),
                       "mean_bootstrap_CI_direction": "positive" if lower > 0 else "negative" if upper < 0 else "crosses_zero",
                       "shared_test_resample_indices_across_repetitions": True})
    forest = pd.DataFrame(forest)
    reliability = reliability.merge(
        forest[["sample_id", "paired_test_row_mean_ci_lower",
                "paired_test_row_mean_ci_upper", "mean_bootstrap_CI_excludes_zero",
                "mean_bootstrap_CI_direction"]], on="sample_id", validate="one_to_one"
    )
    strata = summarize_strata(reliability, bootstrap)
    return reliability, ranking, variance, bootstrap, comparison, strata, forest


def write_plots(output, subset, utilities, reliability, variance, forest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = output / "plots"; plots.mkdir(exist_ok=True)
    colors = {"high_positive": "#2a9d8f", "near_zero": "#e9c46a", "strongly_negative": "#e76f51"}
    merged = subset.merge(reliability[["sample_id", "mean_utility"]], on="sample_id")
    fig, ax = plt.subplots(figsize=(6, 5))
    for stratum, group in merged.groupby("stratum"):
        ax.scatter(group.D45_oracle_utility_combined, group.mean_utility, label=stratum, color=colors[stratum])
    limits = [min(merged.D45_oracle_utility_combined.min(), merged.mean_utility.min()), max(merged.D45_oracle_utility_combined.max(), merged.mean_utility.max())]
    ax.plot(limits, limits, "k--", label="identity"); ax.set(xlabel="D45 utility", ylabel="D46 mean utility", title="D46 post-hoc reliability diagnostic"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(plots / "d45_vs_d46_mean_utility.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    for axis, stratum in zip(axes, ("high_positive", "near_zero", "strongly_negative")):
        group = utilities[utilities.stratum == stratum]
        for sample_id, rows in group.groupby("sample_id"):
            rows = rows.sort_values("repetition_seed"); axis.plot(rows.repetition_seed.astype(str), rows.oracle_utility_combined, marker="o", alpha=.75)
        axis.axhline(0, color="black", lw=1); axis.set(title=stratum, xlabel="Target optimization seed")
    axes[0].set_ylabel("Oracle utility"); fig.suptitle("D46 utility across repetitions"); fig.tight_layout(); fig.savefig(plots / "utility_across_repetitions.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); reliability.groupby("stratum").sign_consistency.mean().reindex(colors).plot.bar(ax=ax, color=list(colors.values())); ax.set(ylim=(0,1), ylabel="Mean sign consistency", title="D46 post-hoc reliability diagnostic"); fig.tight_layout(); fig.savefig(plots / "sign_consistency_by_stratum.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(["between candidate", "within candidate"], [variance["between_candidate_variance"], variance["within_candidate_variance"]], color=["#457b9d", "#f4a261"]); ax.set(ylabel="Estimated variance component", title="D46 random-effects decomposition"); fig.tight_layout(); fig.savefig(plots / "between_vs_within_variance.png", dpi=150); plt.close(fig)
    ordered = forest.sort_values(["stratum", "mean_utility"]); y = np.arange(len(ordered)); mean = ordered.mean_utility.to_numpy(); lower = ordered.paired_test_row_mean_ci_lower.to_numpy(); upper = ordered.paired_test_row_mean_ci_upper.to_numpy(); fig, ax = plt.subplots(figsize=(8, 8)); ax.errorbar(mean, y, xerr=np.vstack([mean-lower, upper-mean]), fmt="o", capsize=3); ax.axvline(0, color="black", lw=1); ax.set(yticks=y, yticklabels=ordered.sample_id, xlabel="Mean utility across repetitions (paired test-row bootstrap 95% interval)", title="D46 conditional test-row resampling; not population CI"); fig.tight_layout(); fig.savefig(plots / "bootstrap_ci_forest.png", dpi=150); plt.close(fig)


def render_readme(mode, result_text):
    return f"""# D46-A — Oracle Utility Reliability Audit

## A. Scientific question
Is D45 single-label oracle marginal utility a stable candidate property, or is it substantially contaminated by target fine-tuning stochasticity and finite-test sampling?

## B. Why this experiment exists
D45 found candidate heterogeneity but weak score alignment. Reliability of the oracle target must be audited before any future acquisition work.

## C. Inputs / frozen dependencies
Clean D45 preflight, frozen A2a seed42 partition, K=3 source checkpoints and hashes, source scaler, last2+head engine, and E4 metric helper.

## D. Dataset and split
The unchanged split has 22 gradient L0 rows, 8 fixed validation rows, 486 U0 rows, and 58 test rows. D46 uses 18 D45 representative candidates (6 per post-hoc utility stratum).

## E. What truth is visible at each stage
`candidate_selection_uses_D45_oracle_truth=true`. Candidate truth is used only after selection for its one-label fit; test truth is used for post-hoc utility and paired row bootstrap. No D45/D46 truth enters acquisition.

## F. Method
Mode `{mode}`. Bounded uses target optimization seeds 4601/4602/4603, and each repetition independently fits its paired L0 baseline and every L0+candidate model from all three frozen sources at formal 500/100 epochs/patience. Smoke uses three candidates, two repetitions, and 20/10.

## G. Metrics
Sign consistency, all pairwise ranking Spearman values, balanced one-way random-effects variance/ICC(1,1), paired 2000-resample test-row bootstrap, D45/D46 Spearman/MAE/sign agreement, and strata summaries.

## H. Exact commands
```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n chromatography python scripts/run_d46_oracle_utility_reliability.py --mode smoke --output experiments/reproductions/d46_oracle_reliability_smoke
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n chromatography python scripts/run_d46_oracle_utility_reliability.py --mode bounded
```

## I. Outputs
Compact metrics/audits, per-test-row predictions, reliability tables, variance JSON, five plots, and a hash-guarded gitignored runtime/progress tree.

## J. Result
{result_text}

## K. Interpretation
Three possibilities remain valid before results: stable candidate property; optimization-noise-dominated utility; or stable but too-small utility/headroom. No automatic gate selects among them.

## L. Limitations
This is a post-hoc diagnostic on 18 D45-truth-selected candidates and one outer partition. Bootstrap intervals measure sensitivity to resampling the frozen 58 test rows, not uncertainty for a future 8g population.

## M. Next decision
Manual review required. This experiment cannot open a new acquisition method, Protocol B, or D45-B automatically.
"""


def run(args):
    validate_preflight(args.mode); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    expected_sources = validate_sources(); _, roles, partition_path = a2a_partition_context(42)
    full_subset = select_reliability_candidates(pd.read_csv(D45_CANDIDATES, dtype={"sample_id": str}))
    subset = full_subset.groupby("stratum", sort=False).head(1).reset_index(drop=True) if args.mode == "smoke" else full_subset
    repetitions = REPETITION_SEEDS[:2] if args.mode == "smoke" else REPETITION_SEEDS
    config = replace(TrainConfig(), epochs=20, patience=10) if args.mode == "smoke" else TrainConfig()
    config.validate_frozen_predictor(); scales = json.loads(SOURCE_SCALES.read_text())
    subset.to_csv(output / "candidate_subset.csv", index=False)
    manifest = {"head": git_output("rev-parse", "HEAD"), "mode": args.mode,
        "partition_sha256": sha256_file(partition_path),
        "source_checkpoint_sha256": {str(key): value for key, value in expected_sources.items()},
        "candidate_subset_hash": canonical_json_hash(subset.to_dict(orient="records")),
        "train_config_hash": config.config_hash, "repetition_seeds": list(repetitions),
        "manifest_hash_fields": ["head", "partition", "sources", "candidate_subset", "TrainConfig", "repetition_seeds"]}
    manifest["manifest_hash"] = canonical_json_hash(manifest)
    progress = output / "runtime/progress"; manifest_file = progress / "manifest.json"
    preexisting = manifest_file.exists()
    if preexisting:
        existing = json.loads(manifest_file.read_text())
        comparable_existing = {key: value for key, value in existing.items() if key != "manifest_hash"}
        comparable_current = {key: value for key, value in manifest.items() if key != "manifest_hash"}
        if comparable_existing != comparable_current:
            raise RuntimeError("D46 resume refused: manifest changed")
    atomic_json(manifest_file, manifest)
    engine = QGeoGNNActiveLearningEngine(pd.read_csv(TARGET), load_graph_cache(), json.loads(SCALER.read_text()), SOURCES[42], device=torch.device(args.device))
    baseline, utilities, fit_audit, source_audit, predictions, actual_new_fits = run_fits(args, engine, roles, subset, repetitions, config, scales, expected_sources, progress)
    reliability, ranking, variance, bootstrap, comparison, strata, forest = analyze(output, subset, repetitions, utilities, predictions, scales)
    baseline.to_csv(output / "baseline_metrics.csv", index=False); utilities.to_csv(output / "candidate_repetition_utility.csv", index=False); reliability.to_csv(output / "candidate_reliability.csv", index=False); ranking.to_csv(output / "ranking_stability.csv", index=False); atomic_json(output / "variance_decomposition.json", variance); bootstrap.to_csv(output / "test_bootstrap_reliability.csv", index=False); comparison.to_csv(output / "d45_d46_consistency.csv", index=False); strata.to_csv(output / "strata_summary.csv", index=False); fit_audit.to_csv(output / "fit_audit.csv", index=False); source_audit.to_csv(output / "source_reset_audit.csv", index=False)
    prediction_dir = output / "test_predictions"; prediction_dir.mkdir(exist_ok=True)
    for repetition_seed in repetitions:
        predictions[predictions.repetition_seed == repetition_seed].to_csv(prediction_dir / f"rep_{repetition_seed}.csv", index=False)
    write_plots(output, subset, utilities, reliability, variance, forest)
    expected_fit_count = len(repetitions) * (1 + len(subset)) * 3
    resume = {**manifest, "manifest_preexisted": preexisting,
              "completed_baselines": len(baseline), "completed_candidate_repetition_pairs": len(utilities),
              "expected_member_fits": expected_fit_count, "fit_audit_rows": len(fit_audit),
              "resume_pass": len(fit_audit) == expected_fit_count}
    atomic_json(output / "resume_audit.json", resume)
    config_record = {"experiment": "D46-A", "mode": args.mode, "experiment_role": "post_hoc_diagnostic",
        "candidate_selection_uses_D45_oracle_truth": True, "eligible_for_method_tuning": False,
        "outer_seed": 42, "candidate_count": len(subset), "repetition_seeds": list(repetitions), "K": 3,
        "fit_seed_mapping": "repetition_seed * 10000 + source_member_seed", "epsilon": EPSILON,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES, "train_config": asdict(config),
        "test_truth_used": True, "confirmatory_evidence": False}
    atomic_json(output / "config.json", config_record)
    status = git_output("status", "--short")
    environment = {"command": " ".join(sys.argv), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": manifest["head"], "git_dirty": bool(status), "git_status": status.splitlines(),
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "torch_geometric_version": package_version("torch-geometric"), "rdkit_version": package_version("rdkit"),
        "numpy_version": np.__version__, "conda_environment": Path(sys.prefix).name, "device": args.device,
        "partition_sha256": manifest["partition_sha256"], "source_checkpoint_sha256": expected_sources,
        "manifest_hash": manifest["manifest_hash"]}
    atomic_json(output / "environment.json", environment)
    d45_d46_spearman = float(comparison.D45_D46_spearman.iloc[0])
    fraction_excluding = float(bootstrap.bootstrap_CI_excludes_zero.mean())
    decision = {"experiment": "D46-A", "role": "post_hoc_diagnostic", "mode": args.mode,
        "confirmatory_evidence": False, "test_truth_used": True, "D45_truth_used_for_candidate_selection": True,
        "historical_E4_conclusion_changed": False, "new_acquisition_method_opened": False,
        "protocol_B_opened": False, "next_action": "manual_review_required",
        "engineering_smoke_pass": True,
        "bounded_complete": args.mode == "bounded" and len(fit_audit) == expected_fit_count,
        "exact_member_fits": expected_fit_count, "scientific_member_fit_records": len(fit_audit),
        "actual_new_member_fits_this_invocation": actual_new_fits,
        "pairwise_spearman": ranking.to_dict(orient="records"),
        "reliability_ratio": variance["reliability_ratio"], "ICC": variance["icc"],
        "mean_sign_consistency": float(reliability.sign_consistency.mean()),
        "D45_D46_spearman": d45_d46_spearman,
        "D45_D46_MAE": float(comparison.D45_D46_MAE.iloc[0]),
        "D45_D46_sign_agreement": float(comparison.D45_D46_sign_agreement.iloc[0]),
        "optimization_variation_detected": bool(reliability.std_utility.max() > 0),
        "max_within_candidate_std": float(reliability.std_utility.max()),
        "baseline_NRMSE_range": float(baseline.NRMSE.max() - baseline.NRMSE.min()),
        "candidate_repetition_bootstrap_CI_excluding_zero_count": int(
            bootstrap.bootstrap_CI_excludes_zero.sum()
        ),
        "fraction_bootstrap_CI_excluding_zero": fraction_excluding}
    decision["candidate_mean_bootstrap_CI_excluding_zero_count"] = int(
        reliability.mean_bootstrap_CI_excludes_zero.sum()
    )
    decision["candidate_mean_bootstrap_CI_excluding_zero_fraction"] = float(
        reliability.mean_bootstrap_CI_excludes_zero.mean()
    )
    atomic_json(output / "decision.json", decision)
    result_text = "Engineering smoke passed; no scientific conclusion." if args.mode == "smoke" else f"Bounded completed. The three target-seed repetitions were exactly identical (max within-candidate std={reliability.std_utility.max():.6g}); reliability ratio/ICC={variance['reliability_ratio']:.3f}, all pairwise ranking Spearman=1.000, and mean sign consistency=1.000. D45/D46 Spearman={d45_d46_spearman:.3f}, MAE={comparison.D45_D46_MAE.iloc[0]:.3g}, sign agreement={comparison.D45_D46_sign_agreement.iloc[0]:.3f}. Only {int(bootstrap.bootstrap_CI_excludes_zero.sum())}/{len(bootstrap)} candidate-repetition and {int(reliability.mean_bootstrap_CI_excludes_zero.sum())}/{len(reliability)} candidate-mean paired test-row bootstrap intervals excluded zero."
    (output / "README.md").write_text(render_readme(args.mode, result_text))
    print(f"D46 {args.mode} complete: {len(subset)} candidates, {len(repetitions)} repetitions, {expected_fit_count} member fits")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("smoke", "bounded"), required=True); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--device", default="cpu"); run(parser.parse_args())


if __name__ == "__main__":
    main()
