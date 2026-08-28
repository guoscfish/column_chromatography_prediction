#!/usr/bin/env python3
"""Run D45 single-label oracle marginal-utility diagnostics."""

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
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qgeognn_al.acquisition import minimum_reference_distance, top_score_select
from src.qgeognn_al.engine import QGeoGNNActiveLearningEngine, TrainConfig, canonical_json_hash
from scripts.run_e0_4g_baseline import sha256_file
from scripts.run_e0_8g_controls import load_graph_cache
from scripts.run_e4_a2a_formal import a2a_partition_context
from scripts.run_e4_active_transfer import (
    SCALER, SOURCES, SOURCE_SCALES, TARGET, acquire, metric_row, score_round,
)
from src.qgeognn_al.diagnostics.oracle_utility import (
    CHALLENGE_TAGS, SCORE_COLUMNS, binary_auroc, bootstrap_spearman_ci,
    build_diagnostic_candidate_subset, compute_enrichment,
    compute_marginal_utility, source_reset_audit_rows,
    summarize_utility_distribution, validate_candidate_fit_contract,
)

DEFAULT_OUTPUT = ROOT / "experiments/d45_oracle_marginal_utility"
OUTER_SEED, DIAGNOSTIC_SEED, BOOTSTRAP_SEED = 42, 4501, 4502


def package_version(distribution: str) -> str:
    """Get a package version without importing an additional runtime."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def fit_oracle_members(
    engine: QGeoGNNActiveLearningEngine,
    labeled_ids: list[str],
    validation_ids: list[str],
    config: TrainConfig,
    output_dir: Path,
    reuse: bool,
) -> tuple[dict[int, Path], list[dict[str, object]]]:
    """Fit K=3 from source with candidate-independent stochastic seeds."""
    checkpoints: dict[int, Path] = {}
    records: list[dict[str, object]] = []
    expected_labeled_hash = canonical_json_hash(sorted(map(str, labeled_ids)))
    for member_seed, source in SOURCES.items():
        member_dir = output_dir / f"member_{member_seed}"
        result_path, checkpoint_path = member_dir / "fit_result.json", member_dir / "best.pt"
        started, reused = time.perf_counter(), False
        if reuse and result_path.exists() and checkpoint_path.exists():
            result = json.loads(result_path.read_text())
            if not (
                result.get("train_config_hash") == config.config_hash
                and result.get("labeled_ids_hash") == expected_labeled_hash
                and result.get("init_source_sha256") == sha256_file(source)
            ):
                raise ValueError(f"Incompatible resumable D45 fit: {member_dir}")
            reused = True
        else:
            result = asdict(
                engine.fit(
                    labeled_ids, validation_ids, config, source,
                    OUTER_SEED * 100000 + member_seed, member_dir,
                )
            )
        if result["init_source_sha256"] != sha256_file(source):
            raise RuntimeError("D45 fit did not reset from the frozen source")
        checkpoints[member_seed] = Path(str(result["checkpoint"]))
        records.append(
            {
                "member_seed": member_seed,
                "fit_seed": OUTER_SEED * 100000 + member_seed,
                "fit_seconds": 0.0 if reused else time.perf_counter() - started,
                "reused_completed_fit": reused,
                **result,
            }
        )
    return checkpoints, records


def normalized_metric(metric: dict[str, object], scales: dict[str, float]) -> dict[str, object]:
    """Expose E4 component NRMSEs while retaining its frozen metric row."""
    return {
        **metric,
        "V1_NRMSE": float(metric["V1_RMSE"]) / scales["V1"],
        "V2_NRMSE": float(metric["V2_RMSE"]) / scales["V2"],
    }


def fit_audit_rows(
    records: list[dict[str, object]], phase: str, candidate_id: str | None
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        frozen_unchanged = record["frozen_parameters_sha256_before"] == record["frozen_parameters_sha256_after"]
        trainable_changed = record["trainable_parameters_sha256_before"] != record["trainable_parameters_sha256_after"]
        row = {
            "phase": phase, "candidate_id": candidate_id,
            "member_seed": record["member_seed"], "fit_seed": record["fit_seed"],
            "train_rows": record["train_rows"], "validation_rows": record["validation_rows"],
            "validation_ids_hash": record["validation_ids_hash"],
            "frozen_unchanged": frozen_unchanged, "trainable_changed": trainable_changed,
            "best_epoch": record["best_epoch"], "epochs_run": record["epochs_run"],
            "fit_seconds": record["fit_seconds"],
            "reused_completed_fit": record["reused_completed_fit"],
            "checkpoint_sha256": record["checkpoint_sha256"],
        }
        if not frozen_unchanged or not trainable_changed:
            raise RuntimeError(f"D45 parameter audit failed: {row}")
        rows.append(row)
    return rows


def build_candidate_subset(engine, checkpoints, roles, scales, mode):
    """Score U0 and freeze a truth-blind diagnostic subset."""
    scores = score_round(engine, checkpoints, roles["l0_train"], roles["u0"], scales)
    coverage = minimum_reference_distance(scores["train_rep"], scores["pool_rep"])
    pool_scores = pd.DataFrame(
        {"sample_id": roles["u0"], "ensemble_score": scores["ensemble_score"],
         "quantile_width": scores["quantile_width"], "coverage_score": coverage}
    )
    challenge_ids = {
        "ensemble_top8": top_score_select(roles["u0"], scores["ensemble_score"], 8),
        "qwidth_top8": top_score_select(roles["u0"], scores["quantile_width"], 8),
        "coverage_top8": acquire("coverage", roles["u0"], scores, 8, DIAGNOSTIC_SEED)[0],
    }
    subset = build_diagnostic_candidate_subset(
        pool_scores, seed=DIAGNOSTIC_SEED,
        random_n=12 if mode == "smoke" else 48, challenge_ids=challenge_ids,
        include_challenge=mode == "bounded",
    )
    if mode == "smoke" and len(subset) != 12:
        raise RuntimeError("Smoke subset must contain exactly 12 candidates")
    audit = {
        **scores["representation_audit"],
        "candidate_construction_columns": ["sample_id", *SCORE_COLUMNS],
        "truth_columns_used": [], "diagnostic_seed": DIAGNOSTIC_SEED,
        "representative_count": int(subset["random_sample"].sum()),
        "exact_candidate_count": len(subset), "challenge_ids": challenge_ids,
    }
    return subset, audit


def write_statistics(output: Path, utilities: pd.DataFrame) -> dict[str, object]:
    """Write bounded-only statistics and plots."""
    representative = utilities[utilities["random_sample"]].copy()
    summaries = pd.DataFrame([
        {"target": target, "subset": "representative_random",
         **summarize_utility_distribution(representative[f"oracle_utility_{target}"])}
        for target in ("V1", "V2", "combined")
    ])
    summaries.to_csv(output / "utility_summary.csv", index=False)
    threshold = float(representative["oracle_utility_combined"].quantile(0.75))
    high = representative["oracle_utility_combined"].to_numpy(float) >= threshold
    alignment_rows, auroc_rows, enrichment_rows = [], [], []
    for index, score in enumerate(SCORE_COLUMNS):
        alignment_rows.append(
            {"score": score, "subset": "representative_random", "n": len(representative),
             **bootstrap_spearman_ci(representative[score], representative["oracle_utility_combined"], BOOTSTRAP_SEED + index)}
        )
        auroc_rows.append(
            {"score": score, "subset": "representative_random",
             "oracle_high_definition": "combined utility >= representative 75th percentile",
             "oracle_high_threshold": threshold, "n": len(representative),
             "positive_count": int(high.sum()), "AUROC": binary_auroc(representative[score], high)}
        )
        for fraction in (0.10, 0.20, 0.25):
            enrichment_rows.append(
                {"score": score, "subset": "representative_random",
                 **compute_enrichment(representative[score], high, fraction)}
            )
    alignment, auroc, enrichment = map(pd.DataFrame, (alignment_rows, auroc_rows, enrichment_rows))
    alignment.to_csv(output / "score_alignment.csv", index=False)
    auroc.to_csv(output / "auroc.csv", index=False)
    enrichment.to_csv(output / "enrichment.csv", index=False)
    challenge_rows = []
    for tag in CHALLENGE_TAGS:
        frame = utilities[utilities[tag]]
        challenge_rows.append(
            {"subset": tag, "n": len(frame),
             "mean_oracle_utility_combined": float(frame["oracle_utility_combined"].mean()),
             "median_oracle_utility_combined": float(frame["oracle_utility_combined"].median()),
             "positive_fraction_combined": float((frame["oracle_utility_combined"] > 0).mean()),
             "representative_mean": float(representative["oracle_utility_combined"].mean()),
             "representative_median": float(representative["oracle_utility_combined"].median()),
             "representative_positive_fraction": float((representative["oracle_utility_combined"] > 0).mean())}
        )
    challenge = pd.DataFrame(challenge_rows)
    challenge.to_csv(output / "challenge_summary.csv", index=False)
    write_plots(output, utilities, representative, alignment)
    return {
        "utility_summary": summaries.to_dict(orient="records"),
        "score_alignment": alignment.to_dict(orient="records"),
        "auroc": auroc.to_dict(orient="records"),
        "enrichment": enrichment.to_dict(orient="records"),
        "challenge_summary": challenge.to_dict(orient="records"),
        "oracle_high_threshold": threshold,
    }


def write_plots(output, utilities, representative, alignment):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output / "plots"; plots.mkdir(exist_ok=True)
    title = "D45 post-hoc diagnostic"
    fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(representative.oracle_utility_combined, bins=12, edgecolor="black"); ax.axvline(0, color="black"); ax.set(xlabel="Oracle utility (combined NRMSE reduction)", ylabel="Candidates", title=f"{title}: utility distribution"); fig.tight_layout(); fig.savefig(plots / "oracle_utility_distribution.png", dpi=150); plt.close(fig)
    for score, filename, label in (("ensemble_score", "ensemble_vs_oracle_utility.png", "Ensemble score"), ("quantile_width", "qwidth_vs_oracle_utility.png", "Quantile width"), ("coverage_score", "coverage_vs_oracle_utility.png", "Coverage distance")):
        fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter(representative[score], representative.oracle_utility_combined); ax.axhline(0, color="black"); ax.set(xlabel=label, ylabel="Oracle utility (combined NRMSE reduction)", title=title); fig.tight_layout(); fig.savefig(plots / filename, dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); positions = np.arange(len(alignment)); rho = alignment.rho.to_numpy(); lower = alignment.bootstrap_ci_lower.to_numpy(); upper = alignment.bootstrap_ci_upper.to_numpy(); ax.errorbar(positions, rho, yerr=np.vstack([rho-lower, upper-rho]), fmt="o", capsize=4); ax.axhline(0, color="black"); ax.set(xticks=positions, xticklabels=alignment.score, ylabel="Spearman rho (bootstrap 95% CI)", title=title); fig.tight_layout(); fig.savefig(plots / "score_spearman_summary.png", dpi=150); plt.close(fig)
    plot_frame = pd.concat([pd.DataFrame({"subset": "representative_random", "utility": representative.oracle_utility_combined}), *[pd.DataFrame({"subset": tag, "utility": utilities.loc[utilities[tag], "oracle_utility_combined"]}) for tag in CHALLENGE_TAGS]], ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 4)); plot_frame.boxplot(column="utility", by="subset", ax=ax, rot=20); fig.suptitle(""); ax.axhline(0, color="black"); ax.set(ylabel="Oracle utility (combined NRMSE reduction)", title=f"{title}: challenge vs random"); fig.tight_layout(); fig.savefig(plots / "challenge_vs_random_utility.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5)); ax.scatter(utilities.oracle_utility_V1, utilities.oracle_utility_V2); ax.axhline(0, color="black"); ax.axvline(0, color="black"); ax.set(xlabel="V1 oracle utility (NRMSE reduction)", ylabel="V2 oracle utility (NRMSE reduction)", title=title); fig.tight_layout(); fig.savefig(plots / "V1_vs_V2_oracle_utility.png", dpi=150); plt.close(fig)


def render_readme(mode: str, statistics: dict[str, object]) -> str:
    if mode == "bounded":
        combined = next(row for row in statistics["utility_summary"] if row["target"] == "combined")
        correlations = {row["score"]: row["rho"] for row in statistics["score_alignment"]}
        result_text = (
            f"Representative combined utility: median `{combined['median']:.6f}`, "
            f"IQR `{combined['IQR']:.6f}`, P10 `{combined['P10']:.6f}`, "
            f"P90 `{combined['P90']:.6f}`, positive fraction "
            f"`{combined['fraction_utility_gt_0']:.3f}`. Spearman rho: "
            f"Ensemble `{correlations['ensemble_score']:.3f}`, QWidth "
            f"`{correlations['quantile_width']:.3f}`, Coverage "
            f"`{correlations['coverage_score']:.3f}`; all bootstrap 95% CIs cross zero."
        )
    else:
        result_text = "Engineering smoke completed; its utilities are not scientific evidence."
    return f"""# D45 — Oracle Marginal Utility Audit

## A. Scientific question
At frozen seed-42 L0=30, do individual U0 labels have heterogeneous marginal training utility, and do current legal acquisition scores identify high-utility candidates?

## B. Why this experiment exists
E4 Protocol A and E4-A2a both returned null active-acquisition evidence. D45 diagnoses candidate utility; it does not open another method.

## C. Inputs / frozen dependencies
Frozen 8g target data, A2a seed-42 partition, K=3 source checkpoints (42/525/1101), 4g scaler, QGeoGNN engine, E4 scores, and E4 metrics.

## D. Dataset and split
`l0_train=22`, fixed `l0_validation=8`, `u0=486`, `test=58`; the partition is unchanged.

## E. What truth is visible at each stage
Subset construction sees only IDs and unlabeled Round0 scores. Candidate truth is revealed only for that candidate's gradient fit. Frozen test truth defines oracle utility. `test_truth_used_for_oracle_utility=true`.

## F. Method
Every member resets from its frozen 4g source with the same candidate-independent fit seed. A candidate adds exactly one U0 row to gradient training, never validation. Mode `{mode}` uses K=3. Smoke is 12 candidates at 20/10 epochs/patience and is engineering-only. Bounded uses random 48 plus deduplicated top-8 challenge candidates at formal 500/100.

## G. Metrics
Baseline-minus-after V1 NRMSE, V2 NRMSE, and frozen E4 combined NRMSE; bounded adds fixed-seed bootstrap Spearman 95% CIs, representative-subset AUROC/enrichment, and challenge summaries.

## H. Exact commands
```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n chromatography python scripts/run_d45_oracle_marginal_utility.py --mode smoke
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n chromatography python scripts/run_d45_oracle_marginal_utility.py --mode bounded
```

## I. Outputs
Config/environment/decision JSON, candidate/baseline/utility/audit CSVs, bounded analysis tables, and plots. Runtime checkpoints are gitignored and removed after compact extraction.

## J. Result
{result_text}

## K. Interpretation
`experiment_role=post_hoc_diagnostic`, `confirmatory_evidence=false`, `historical_E4_conclusion_changed=false`, and `eligible_for_direct_method_tuning=false`.

## L. Limitations
Bounded evidence is 48 representative rows from one outer seed. This test partition is consumed as oracle truth. Any D45-informed method needs a new preregistered confirmatory partition/protocol.

## M. Next decision
Manual review only. Do not automatically open D46, a new method, or a full oracle trajectory.
"""


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    _, roles, partition_path = a2a_partition_context(OUTER_SEED)
    role_sets = {key: set(values) for key, values in roles.items()}
    if tuple(map(len, (roles["l0_train"], roles["l0_validation"], roles["u0"], roles["test"]))) != (22, 8, 486, 58):
        raise RuntimeError("D45 frozen partition count violation")
    keys = list(roles)
    if any(role_sets[keys[i]] & role_sets[keys[j]] for i in range(4) for j in range(i + 1, 4)):
        raise RuntimeError("D45 frozen partition overlap")
    formal_config = TrainConfig()
    config = replace(formal_config, epochs=20, patience=10) if args.mode == "smoke" else formal_config
    config.validate_frozen_predictor()
    if args.mode == "bounded" and (config.epochs, config.patience) != (500, 100):
        raise RuntimeError("Bounded mode requires formal 500/100")
    scales = json.loads(SOURCE_SCALES.read_text())
    engine = QGeoGNNActiveLearningEngine(pd.read_csv(TARGET), load_graph_cache(), json.loads(SCALER.read_text()), SOURCES[42], device=torch.device(args.device))
    initial = roles["l0_train"] + roles["l0_validation"]
    runtime = output / "runtime" / args.mode; progress = runtime / "progress"; progress.mkdir(parents=True, exist_ok=True)
    checkpoints, baseline_records = fit_oracle_members(engine, initial, roles["l0_validation"], config, runtime / "baseline", args.resume)
    baseline = normalized_metric(metric_row(engine, roles["test"], checkpoints, scales, {"phase": "baseline"}), scales)
    pd.DataFrame([baseline]).to_csv(output / "baseline_metrics.csv", index=False)
    subset, construction_audit = build_candidate_subset(engine, checkpoints, roles, scales, args.mode)
    subset.to_csv(output / "candidate_subset.csv", index=False)
    (output / "candidate_construction_audit.json").write_text(json.dumps(construction_audit, indent=2))
    partition_hash = sha256_file(partition_path)
    resume_manifest = {
        "mode": args.mode,
        "config_hash": config.config_hash,
        "partition_sha256": partition_hash,
        "candidate_subset_hash": canonical_json_hash(subset.to_dict(orient="records")),
    }
    resume_manifest_path = progress / "resume_manifest.json"
    manifest_preexisted = resume_manifest_path.exists()
    if manifest_preexisted and json.loads(resume_manifest_path.read_text()) != resume_manifest:
        raise RuntimeError("D45 resume refused: frozen run manifest changed")
    resume_manifest_path.write_text(json.dumps(resume_manifest, indent=2))
    expected_sources = {seed: sha256_file(path) for seed, path in SOURCES.items()}
    fit_rows = fit_audit_rows(baseline_records, "baseline", None)
    reset_rows = source_reset_audit_rows(baseline_records, expected_sources, "baseline", None)
    validation_hash = canonical_json_hash(sorted(roles["l0_validation"]))
    if {str(record["validation_ids_hash"]) for record in baseline_records} != {validation_hash}:
        raise RuntimeError("Baseline fixed-validation audit failed")
    utility_path, fit_path, reset_path = (progress / name for name in ("candidate_utility.partial.csv", "fit_audit.partial.csv", "source_reset_audit.partial.csv"))
    completed = pd.read_csv(utility_path, dtype={"sample_id": str}) if args.resume and utility_path.exists() else pd.DataFrame()
    utility_rows = completed.to_dict(orient="records"); completed_ids = set(completed.sample_id.astype(str)) if len(completed) else set()
    if len(completed) and (completed.sample_id.duplicated().any() or not completed_ids <= set(subset.sample_id)):
        raise RuntimeError("D45 resume refused: completed candidates mismatch frozen subset")
    if args.resume and fit_path.exists(): fit_rows.extend(pd.read_csv(fit_path, dtype={"candidate_id": str}).to_dict(orient="records"))
    if args.resume and reset_path.exists(): reset_rows.extend(pd.read_csv(reset_path, dtype={"candidate_id": str}).to_dict(orient="records"))
    for candidate_index, candidate in enumerate(subset.itertuples(index=False), 1):
        candidate_id = str(candidate.sample_id)
        if candidate_id in completed_ids: continue
        validate_candidate_fit_contract(
            initial, roles["l0_validation"], initial + [candidate_id], candidate_id,
            roles["u0"], roles["test"],
        )
        candidate_dir = runtime / "candidates" / candidate_id
        candidate_checkpoints, records = fit_oracle_members(engine, initial + [candidate_id], roles["l0_validation"], config, candidate_dir, args.resume)
        after = normalized_metric(metric_row(engine, roles["test"], candidate_checkpoints, scales, {"phase": "candidate", "sample_id": candidate_id}), scales)
        utility_rows.append({**candidate._asdict(),
            "baseline_V1_NRMSE": baseline["V1_NRMSE"], "after_V1_NRMSE": after["V1_NRMSE"], "oracle_utility_V1": compute_marginal_utility(baseline["V1_NRMSE"], after["V1_NRMSE"]),
            "baseline_V2_NRMSE": baseline["V2_NRMSE"], "after_V2_NRMSE": after["V2_NRMSE"], "oracle_utility_V2": compute_marginal_utility(baseline["V2_NRMSE"], after["V2_NRMSE"]),
            "baseline_combined_NRMSE": baseline["NRMSE"], "after_combined_NRMSE": after["NRMSE"], "oracle_utility_combined": compute_marginal_utility(baseline["NRMSE"], after["NRMSE"])})
        fit_rows.extend(fit_audit_rows(records, "candidate", candidate_id)); reset_rows.extend(source_reset_audit_rows(records, expected_sources, "candidate", candidate_id))
        if {str(record["validation_ids_hash"]) for record in records} != {validation_hash} or any(int(record["train_rows"]) != 23 for record in records):
            raise RuntimeError("Candidate fixed-validation/gradient-row audit failed")
        pd.DataFrame(utility_rows).to_csv(utility_path, index=False)
        pd.DataFrame([row for row in fit_rows if row["phase"] == "candidate"]).to_csv(fit_path, index=False)
        pd.DataFrame([row for row in reset_rows if row["phase"] == "candidate"]).to_csv(reset_path, index=False)
        shutil.rmtree(candidate_dir)
        print(f"D45 {args.mode}: candidate {candidate_index}/{len(subset)} {candidate_id}", flush=True)
    utilities = pd.DataFrame(utility_rows).drop_duplicates("sample_id", keep="last")
    utilities = subset.merge(utilities.drop(columns=[column for column in subset if column in utilities and column != "sample_id"]), on="sample_id", how="left", validate="one_to_one")
    if len(utilities) != len(subset) or utilities.oracle_utility_combined.isna().any():
        raise RuntimeError("D45 candidate completion audit failed")
    utilities.to_csv(output / "candidate_utility.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(output / "fit_audit.csv", index=False); pd.DataFrame(reset_rows).to_csv(output / "source_reset_audit.csv", index=False)
    if not all(bool(row["source_reset_pass"]) for row in reset_rows): raise RuntimeError("D45 source-reset audit failed")
    statistics = write_statistics(output, utilities) if args.mode == "bounded" else {}
    (output / "resume_audit.json").write_text(json.dumps({
        **resume_manifest,
        "manifest_preexisted": manifest_preexisted,
        "completed_candidates_loaded": len(completed),
        "candidate_subset_deterministic": True,
        "completed_ids_subset_of_frozen_subset": True,
        "resume_pass": True,
    }, indent=2))
    status = git_output("status", "--short")
    config_record = {"experiment": "D45", "stage": "D45-A bounded oracle qualification", "mode": args.mode, "outer_seed": OUTER_SEED, "diagnostic_seed": DIAGNOSTIC_SEED, "bootstrap_seed": BOOTSTRAP_SEED, "candidate_count": len(subset), "K": 3, "l0_train": 22, "fixed_validation": 8, "u0": 486, "test": 58, "train_config": asdict(config), "config_hash": config.config_hash, "coverage_contract": construction_audit, "test_truth_used_for_oracle_utility": True, "experiment_role": "post_hoc_diagnostic", "confirmatory_evidence": False, "historical_E4_conclusion_changed": False, "eligible_for_direct_method_tuning": False}
    (output / "config.json").write_text(json.dumps(config_record, indent=2))
    environment = {"command": " ".join(sys.argv), "timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": git_output("rev-parse", "HEAD"), "git_dirty": bool(status), "git_status": status.splitlines(), "python_version": platform.python_version(), "torch_version": torch.__version__, "torch_geometric_version": package_version("torch-geometric"), "rdkit_version": package_version("rdkit"), "numpy_version": np.__version__, "device": args.device, "seed": OUTER_SEED, "source_checkpoint_sha256": expected_sources, "partition_sha256": partition_hash, "config_hash": config.config_hash, "conda_environment": Path(sys.prefix).name, "kmp_duplicate_lib_ok": bool(os.environ.get("KMP_DUPLICATE_LIB_OK"))}
    (output / "environment.json").write_text(json.dumps(environment, indent=2))
    descriptive = {}
    if args.mode == "bounded":
        combined = next(row for row in statistics["utility_summary"] if row["target"] == "combined")
        correlations = {row["score"]: row["rho"] for row in statistics["score_alignment"]}
        descriptive = {"utility_median": combined["median"], "utility_IQR": combined["IQR"],
                       "utility_p90_minus_p10": combined["P90"] - combined["P10"],
                       "ensemble_spearman": correlations["ensemble_score"],
                       "qwidth_spearman": correlations["quantile_width"],
                       "coverage_spearman": correlations["coverage_score"]}
    decision = {"experiment": "D45", "role": "post_hoc_diagnostic", "mode": args.mode, "historical_E4_conclusion_changed": False, "test_truth_used_for_oracle_utility": True, "confirmatory_evidence": False, "eligible_for_direct_method_tuning": False, "full_oracle_trajectory_run": False, "new_acquisition_method_opened": False, "exact_candidate_count": len(subset), "exact_fits": 3 * (1 + len(subset)), "actual_new_fits": int(sum(not bool(row["reused_completed_fit"]) for row in fit_rows)), "next_action": "manual_review_required", **descriptive, **statistics}
    (output / "decision.json").write_text(json.dumps(decision, indent=2)); (output / "README.md").write_text(render_readme(args.mode, statistics))
    shutil.rmtree(runtime / "baseline")
    print(f"D45 {args.mode} complete: {len(subset)} candidates, {decision['exact_fits']} K-member fits")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "bounded"), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
