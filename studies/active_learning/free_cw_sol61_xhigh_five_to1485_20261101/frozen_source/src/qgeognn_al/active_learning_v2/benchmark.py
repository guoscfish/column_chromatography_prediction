"""Phase-gated study orchestration, kept separate from acquisition and fitting."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import sha256_file
from ..resources import ROOT, SOURCE_DATA
from ..training.predictor import atomic_json
from .benchmark_protocol import (ALL_SEEDS, CONFIRMATION_SEEDS, DEVELOPMENT_SEEDS, STUDY,
                                 assert_formal_authorized, load_features)
from .benchmark_reporting import metric_row, plot_primary, write_primary_tables
from .benchmark_seed import SeedContext, freeze_seed
from .protocol import RestrictedLabelStore, make_row_protocol, stable_hash


def validate_freezes(root, seeds, batch_size, smoke):
    freezes = {}
    for seed in seeds:
        runtime = root / f"seed_{seed}"
        path = runtime / f"b{batch_size}/pre_test_freeze.json"
        if not path.exists():
            raise RuntimeError(f"seed {seed} is not frozen before test")
        freeze = json.loads(path.read_text())
        if (freeze["status"] != "FROZEN_BEFORE_TEST_TRUTH" or freeze["test_truth_access_count"] != 0
                or freeze["smoke"] != smoke or freeze["batch_size"] != batch_size or freeze["outer_seed"] != seed):
            raise RuntimeError("invalid pre-test freeze")
        for relative, digest in freeze["files"].items():
            if sha256_file(runtime / relative) != digest:
                raise RuntimeError(f"frozen artifact changed: seed={seed} {relative}")
        freezes[str(seed)] = {"sha256": sha256_file(path), **freeze}
    return freezes


def evaluate_frozen(*, source, partitions, runtime, results, seeds, batch_size, smoke=False):
    # This global barrier precedes the first test-label reveal across every registered seed.
    freezes = validate_freezes(runtime, seeds, batch_size, smoke)
    atomic_json(results / f"global_pre_test_freeze_b{batch_size}.json", {"seeds": freezes, "smoke": smoke})
    rows, accesses, fits = [], [], []
    for seed in seeds:
        directory = runtime / f"seed_{seed}"
        partition = partitions[seed]
        store = RestrictedLabelStore(source, partition)
        context = json.loads((directory / "context.json").read_text())
        if context["source_sha256"] != sha256_file(source):
            raise RuntimeError("evaluation target provenance mismatch")
        selection = pd.read_csv(directory / f"b{batch_size}/selected_batches.csv")
        store.freeze_acquisitions(selection.sample_id)
        store.freeze_predictions()
        test_ids = partition.loc[partition.role.eq("test"), "sample_id"].tolist()
        truth = store.reveal(test_ids, "final_test_evaluation")
        audit = pd.read_csv(directory / f"b{batch_size}/fit_audit.csv")
        fits.append(audit)
        for _, fit in audit.loc[~audit.arm.str.startswith("ensemble_member_")].iterrows():
            prediction = pd.read_csv(directory / fit.arm / "predictions.csv.gz")
            if prediction.sample_id.tolist() != test_ids:
                raise RuntimeError("evaluation prediction identity/order mismatch")
            row = {"outer_seed": seed, "batch_size": batch_size, "arm": fit.arm.rsplit("/", 1)[-1],
                   "cohort": "smoke" if smoke else "confirmation" if seed in CONFIRMATION_SEEDS else "development",
                   **{key: fit[key] for key in ("active_label_count", "active_label_fraction_of_outer_train",
                      "shared_validation_label_count", "total_observed_label_count", "total_observed_fraction_of_full_dataset")},
                   **metric_row(truth, prediction.drop(columns="sample_id").to_numpy(), context["preprocessing"]["target_scales"])}
            rows.append(row)
        accesses.extend({"outer_seed": seed, **row} for row in store.audit)
        before = pd.read_csv(directory / f"b{batch_size}/label_access_audit.csv")
        accesses.extend(before.assign(outer_seed=seed).to_dict("records"))
    results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(accesses).to_csv(results / f"label_access_audit_b{batch_size}.csv", index=False)
    joined = pd.concat(fits, ignore_index=True)
    joined.to_csv(results / f"fit_audit_b{batch_size}.csv", index=False)
    joined[["outer_seed", "arm", "training_seconds", "epochs_run", "best_epoch"]].to_csv(
        results / f"runtime_audit_b{batch_size}.csv", index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(results / f"all_metrics_b{batch_size}.csv", index=False)
    for name in ("input_redundancy_audit", "gradient_duplicate_audit", "gradient_mechanism_correlations",
                 "gradient_mechanism_selection_profiles", "initialization_hash_audit"):
        frames = [pd.read_csv(runtime / f"seed_{seed}" / f"b{batch_size}" / f"{name}.csv").assign(outer_seed=seed)
                  for seed in seeds]
        pd.concat(frames, ignore_index=True).to_csv(results / f"{name}_b{batch_size}.csv", index=False)
    return metrics


def load_partitions(study, seeds):
    manifest = json.loads((study / "splits/split_manifest.json").read_text())
    records = {r["outer_seed"]: r for r in manifest["splits"]}
    partitions = {}
    for seed in seeds:
        path = study / "splits" / f"row_seed_{seed}.csv"
        if sha256_file(path) != records[seed]["sha256"]:
            raise RuntimeError("frozen split drift")
        partitions[seed] = pd.read_csv(path)
    return partitions


def execute_primary(commit, study=STUDY):
    commit = assert_formal_authorized(commit, study)
    if (study / "primary_results_freeze.json").exists():
        raise RuntimeError("primary results already frozen; refusing to overwrite")
    partitions = load_partitions(study, ALL_SEEDS)
    runtime, results = study / "runtime", study / "formal_results"
    for seed in ALL_SEEDS:
        context = SeedContext(SOURCE_DATA, runtime / f"seed_{seed}", seed, partitions[seed])
        freeze_seed(context, 32)
        print(json.dumps({"primary_seed_frozen": seed, "test_truth_access_count": 0}), flush=True)
    metrics = evaluate_frozen(source=SOURCE_DATA, partitions=partitions, runtime=runtime, results=results,
                              seeds=ALL_SEEDS, batch_size=32)
    decisions = write_primary_tables(results, metrics,
        {"confirmation": CONFIRMATION_SEEDS, "development": DEVELOPMENT_SEEDS})
    atomic_json(study / "formal_decision.json", {"preregistration_commit": commit,
                "primary": decisions["confirmation"], "development_descriptive": decisions["development"]})
    plot_primary(results, study / "figures")
    paths = sorted(results.glob("*")) + [study / "formal_decision.json"]
    atomic_json(study / "primary_results_freeze.json", {
        "preregistration_commit": commit, "all_primary_seeds": list(ALL_SEEDS),
        "files": {str(p.relative_to(study)): sha256_file(p) for p in paths if p.is_file()}})


def execute_secondary(commit, study=STUDY):
    commit = assert_formal_authorized(commit, study)
    if not (study / "primary_results_freeze.json").exists():
        raise RuntimeError("secondary requires all primary results frozen")
    frozen = json.loads((study / "primary_results_freeze.json").read_text())
    if frozen["preregistration_commit"] != commit or frozen["all_primary_seeds"] != list(ALL_SEEDS):
        raise RuntimeError("secondary requires all primary results frozen")
    for relative, digest in frozen["files"].items():
        if sha256_file(study / relative) != digest:
            raise RuntimeError("primary result freeze drift")
    partitions = load_partitions(study, ALL_SEEDS)
    sensitivity = []
    for batch in (16, 64):
        for seed in ALL_SEEDS:
            context = SeedContext(SOURCE_DATA, study / "runtime" / f"seed_{seed}", seed, partitions[seed])
            freeze_seed(context, batch)
        metrics = evaluate_frozen(source=SOURCE_DATA, partitions=partitions, runtime=study / "runtime",
            results=study / "secondary_results", seeds=ALL_SEEDS, batch_size=batch)
        sensitivity.append(metrics)
    primary = pd.read_csv(study / "formal_results/all_metrics_b32.csv")
    primary = primary.loc[primary.arm.eq("lcmd") | primary.arm.str.startswith("random_control_")]
    sensitivity.append(primary)
    rows = []
    for (cohort, batch, seed), frame in pd.concat(sensitivity).groupby(["cohort", "batch_size", "outer_seed"]):
        lcmd = float(frame.loc[frame.arm.eq("lcmd"), "combined_normalized_RMSE"].iloc[0])
        random = frame.loc[frame.arm.str.startswith("random_control_"), "combined_normalized_RMSE"]
        rows.append({"cohort": cohort, "batch_size": batch, "outer_seed": seed, "LCMD_NRMSE": lcmd,
                     "Random_mean_NRMSE": random.mean(), "relative_improvement": 1 - lcmd / random.mean(),
                     "beats_random_median": bool(lcmd < random.median())})
    pd.DataFrame(rows).to_csv(study / "secondary_results/batch_size_sensitivity.csv", index=False)


def create_smoke_fixture(directory, rows=160):
    """Sample X deterministically; held-out smoke targets are synthetic before parsing."""
    directory.mkdir(parents=True, exist_ok=True)
    features = load_features()
    positions = np.linspace(0, len(features) - 1, rows, dtype=int)
    fixture = features.iloc[positions].reset_index(drop=True)
    partition = make_row_protocol(fixture, 29)
    wanted = set(fixture.sample_id)
    test_ids = set(partition.loc[partition.role.eq("test"), "sample_id"])
    records = {}
    with SOURCE_DATA.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["sample_id"] in wanted:
                if row["sample_id"] in test_ids:
                    row["V1_ml"], row["V2_ml"] = "17.0", "31.0"
                records[row["sample_id"]] = row
    frame = pd.DataFrame([records[s] for s in fixture.sample_id])
    path = directory / "fixture.csv"
    frame.to_csv(path, index=False)
    partition.to_csv(directory / "partition.csv", index=False)
    return path, partition


def smoke(directory):
    directory = Path(directory)
    if directory.resolve() == STUDY.resolve() or (directory / "protocol.json").exists():
        raise ValueError("smoke requires an isolated output directory")
    source, partition = create_smoke_fixture(directory)
    runtime, results = directory / "runtime", directory / "synthetic_test_results"
    freezes = {}
    for batch in (32, 16, 64):
        context = SeedContext(source, runtime / "seed_29", 29, partition, smoke=True)
        freezes[str(batch)] = freeze_seed(context, batch)
        metrics = evaluate_frozen(source=source, partitions={29: partition}, runtime=runtime, results=results,
                                  seeds=(29,), batch_size=batch, smoke=True)
        if batch == 32:
            write_primary_tables(results, metrics, {}, smoke=True)
    report = {"status": "PASS", "fixture_rows": len(partition), "seed": 29, "maximum_epochs": 2,
              "batches": [32, 16, 64], "primary_arms": 8, "secondary_arms_each": 6,
              "formal_test_truth_access_count": 0, "held_out_truth": "synthetic_17_31",
              "formal_benchmark_run": False, "formal_decision_generated": False,
              "code_contract_hash": stable_hash(context.contract["code_hashes"]),
              "freezes": {key: {k: f[k] for k in ("selected_ids_hash", "checkpoint_state_hash", "seconds")}
                          for key, f in freezes.items()}}
    atomic_json(directory / "smoke_report.json", report)
    return report
