"""Frozen CW-only performance experiment over existing development artifacts."""
from dataclasses import asdict
import ast
import csv
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
import torch

from ..artifacts import sha256_file
from ..models import build_predictor, load_predictor_checkpoint
from ..resources import ROOT, SOURCE_DATA, SOURCE_GRAPH_CACHE
from ..training.predictor import atomic_json, seed_everything
from .benchmark_protocol import DEVELOPMENT_SEEDS, package_versions, seed_config, sketch_seed, initialization_seed
from .benchmark_reporting import metric_row
from .benchmark_seed import SeedContext
from .gradient_features import state_dict_hash
from .gradient_transforms import center_width_transform
from .innovation_screen import _verified_receipt, reveal_l0_targets_only
from .lcmd import lcmd_tp_select
from .protocol import RestrictedLabelStore, ids_hash, stable_hash, validate_row_protocol
from ..schemas.conditions import ConditionNormalization

STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_center_width_performance"
BENCHMARK = Path("studies/active_learning/qgeognn_v2_row_small_batch_benchmark")
INNOVATION = Path("studies/active_learning/qgeognn_v2_row_innovation_screen")
METRICS = [f"{t}_{m}" for t in ("V1", "V2") for m in ("RMSE", "MAE", "R2")] + ["combined_normalized_RMSE"]


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_batch(ids, pool):
    values = list(ids)
    require(len(values) == len(set(values)) == 32, "batch must contain exactly 32 unique IDs")
    require(set(values) <= set(pool), "selected batch outside U0")


def verify_files(files):
    for path, digest in files.items():
        require(Path(path).is_file() and sha256_file(Path(path)) == digest, f"frozen artifact changed: {path}")


def code_files():
    return [*sorted((ROOT / "src/qgeognn_al").rglob("*.py")),
            *sorted((ROOT / "application").glob("*.py")),
            ROOT / "scripts/studies/run_qgeognn_v2_4g_cw_performance.py",
            *sorted((ROOT / "tests/active_learning_v2").glob("*.py"))]


def audited_seed(seed, repository, innovation):
    require(seed in DEVELOPMENT_SEEDS, "only development seeds are permitted")
    source = repository / BENCHMARK / "runtime" / f"seed_{seed}"
    split_path = ROOT / BENCHMARK / "splits" / f"row_seed_{seed}.csv"
    partition = pd.read_csv(split_path)
    validate_row_protocol(partition)
    require(set(partition.outer_seed) == {seed}, "seed identity mismatch")
    manifest = read_json(ROOT / BENCHMARK / "splits/split_manifest.json")
    entry = next(row for row in manifest["splits"] if row["outer_seed"] == seed)
    require(sha256_file(split_path) == entry["sha256"], "split hash mismatch")
    roles = {role: partition.loc[partition.role.eq(role), "canonical_index"].to_numpy(int)
             for role in ("l0", "u0", "validation", "test")}
    require([len(roles[r]) for r in roles] == [333, 2997, 416, 417], "split sizes changed")
    outer = np.r_[roles["l0"], roles["u0"]]
    ids = lambda indices: partition.iloc[indices].sample_id.astype(str).tolist()
    for role in roles:
        require(ids_hash(ids(roles[role])) == entry[f"{role}_ids_hash"], "role identity mismatch")

    context_path = source / "context.json"
    graph_path = source / "scrubbed_graphs.pt"
    raw_path = source / "gradient_features.npz"
    graph = _verified_receipt(graph_path)
    raw = _verified_receipt(raw_path)
    context = read_json(context_path)
    contract = graph["contract"]
    require(context["contract_hash"] == graph["contract_hash"] == raw["contract"]["context_hash"],
            "historical context mismatch")
    require(contract["partition"] == partition.to_dict("list"), "historical partition mismatch")
    require(contract["ordered_sample_ids"] == ids(np.arange(len(partition))), "historical identity order mismatch")
    require(contract["source_sha256"] == sha256_file(SOURCE_DATA), "source data mismatch")
    require(contract["graph_cache_sha256"] == sha256_file(SOURCE_GRAPH_CACHE), "graph cache mismatch")
    require(contract["packages"] == package_versions() and not contract["smoke"], "runtime package/protocol mismatch")
    require(contract["preprocessing"] == context["preprocessing"], "preprocessing context mismatch")
    # The innovation change adds transformed extractors; historical Raw code is unchanged.
    for relative, digest in contract["code_hashes"].items():
        if relative != "src/qgeognn_al/active_learning_v2/gradient_features.py":
            require(sha256_file(ROOT / relative) == digest, f"historical training code changed: {relative}")
    gradient_relative = "src/qgeognn_al/active_learning_v2/gradient_features.py"
    original = subprocess.check_output(["git", "show", f"a419ea75e7894004aef393cb3734a7ae5e1356a1:{gradient_relative}"],
                                       cwd=ROOT, text=True)
    current_functions = {node.name: ast.dump(node) for node in ast.parse((ROOT / gradient_relative).read_text()).body
                         if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    require(all(current_functions.get(node.name) == ast.dump(node) for node in ast.parse(original).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))), "historical Raw extractor definitions changed")
    freeze_path = source / "b32/pre_test_freeze.json"
    freeze = read_json(freeze_path)
    global_freeze = read_json(ROOT / BENCHMARK / "formal_results/global_pre_test_freeze_b32.json")["seeds"][str(seed)]
    require(sha256_file(freeze_path) == global_freeze["sha256"], "historical pre-test freeze mismatch")
    require(freeze["context_hash"] == graph["contract_hash"] and freeze["test_truth_access_count"] == 0
            and freeze["status"] == "FROZEN_BEFORE_TEST_TRUTH" and not freeze["smoke"], "historical freeze invalid")
    protected = [split_path, context_path, graph_path, raw_path, freeze_path,
                 graph_path.with_suffix(".pt.contract.json"), raw_path.with_suffix(".npz.contract.json")]
    historical_selections = source / "b32/selected_batches.csv"
    protected.append(historical_selections)
    for relative in ("context.json", "gradient_features.npz", "b32/selected_batches.csv"):
        require(sha256_file(source / relative) == freeze["files"][relative], "historical frozen input changed")
    selected = pd.read_csv(historical_selections)
    raw_ids = selected.loc[selected.arm.eq("lcmd")].sort_values("selection_order").sample_id.astype(str).tolist()
    validate_batch(raw_ids, ids(roles["u0"]))
    config = {**seed_config(seed), "split_sha256": stable_hash(partition.to_dict("list"))}
    normalization = ConditionNormalization(**contract["normalization"])
    seed_everything(initialization_seed(seed))
    init_hash = state_dict_hash(build_predictor(normalization))
    for arm, count in (("baseline_l0", 333), ("b32/lcmd", 365)):
        for name in ("best.pt", "predictions.csv.gz", "fit_audit.json"):
            path = source / arm / name
            require(sha256_file(path) == freeze["files"][f"{arm}/{name}"], "historical fit freeze mismatch")
            protected.append(path)
        audit = read_json(source / arm / "fit_audit.json")
        checkpoint = torch.load(source / arm / "best.pt", map_location="cpu", weights_only=False)
        require(checkpoint["training_config"] == config, "historical training protocol incompatibility")
        require(checkpoint["preprocessing"] == context["preprocessing"] and
                checkpoint["normalization"] == contract["normalization"], "checkpoint preprocessing mismatch")
        require(audit["initialization_hash"] == init_hash and audit["initialization_seed"] == initialization_seed(seed),
                "initialization mismatch")
        require(audit["train_rows"] == count and audit["validation_rows"] == 416 and audit["prediction_rows"] == 417,
                "historical label budget mismatch")
        expected_train_ids = ids(roles["l0"]) + (raw_ids if count == 365 else [])
        require(audit["train_ids_hash"] == ids_hash(expected_train_ids) and
                audit["validation_ids_hash"] == ids_hash(ids(roles["validation"])), "fit identity mismatch")
        require(audit["checkpoint_sha256"] == sha256_file(source / arm / "best.pt") and
                audit["prediction_sha256"] == sha256_file(source / arm / "predictions.csv.gz"), "fit content mismatch")
        require(audit["test_labels_used_for_fit_or_checkpoint_selection"] == 0, "test label leakage")
        require(audit["checkpoint_state_hash"] == state_dict_hash(load_predictor_checkpoint(source / arm / "best.pt")),
                "checkpoint semantic hash mismatch")
        predictions = pd.read_csv(source / arm / "predictions.csv.gz")
        require(predictions.sample_id.astype(str).tolist() == ids(roles["test"]), "prediction identities mismatch")
    require(raw["contract"]["checkpoint_sha256"] == sha256_file(source / "baseline_l0/best.pt"), "raw checkpoint mismatch")
    require(raw["contract"]["sample_ids"] == ids(outer) and raw["contract"]["sketch_seed"] == sketch_seed(seed)
            and raw["contract"]["dimension"] == 512, "raw sketch contract mismatch")
    scales = [context["preprocessing"]["target_scales"][t] for t in ("V1", "V2")]
    require(raw["contract"]["scales"] == scales, "raw scales mismatch")
    l0_truth, access = reveal_l0_targets_only(SOURCE_DATA, partition)
    require(np.array_equal(l0_truth.std(0), scales), "L0 evaluation scales mismatch")
    transform, transform_audit = center_width_transform(l0_truth)
    cw_path = innovation / INNOVATION / "runtime" / f"seed_{seed}/center_width_gradient_features.npz"
    cw = _verified_receipt(cw_path)
    require(cw["contract"] == {"outer_seed": seed, "checkpoint_sha256": raw["contract"]["checkpoint_sha256"],
            "ordered_sample_ids_hash": stable_hash(ids(outer)), "sketch_seed": sketch_seed(seed),
            "sketch_dimension": 512, "output_transform": transform.tolist()}, "CW cache contract mismatch")
    protected.extend([cw_path, cw_path.with_suffix(".npz.contract.json")])
    positions = {}
    for name, path in (("raw", raw_path), ("cw", cw_path)):
        with np.load(path) as cache:
            features = cache["features"]
            require(np.array_equal(cache["canonical_indices"], outer), "gradient identity order mismatch")
        require(features.shape == (3330, 512) and np.isfinite(features).all(), "gradient shape/content invalid")
        first = lcmd_tp_select(features[333:], features[:333], 32).selected_pool_positions
        second = lcmd_tp_select(features[333:], features[:333], 32).selected_pool_positions
        require(np.array_equal(first, second), "nondeterministic selection")
        positions[name] = roles["u0"][first]
        validate_batch(ids(positions[name]), ids(roles["u0"]))
    require(ids(positions["raw"]) == raw_ids, "Raw historical ordered B32 regression failed")
    innovation_selected = innovation / INNOVATION / "results/selected_batches.csv"
    existing = pd.read_csv(innovation_selected)
    historical_cw = existing.loc[existing.outer_seed.eq(seed) & existing.method.eq("center_width_lcmd")]
    require(historical_cw.sort_values("selection_order").sample_id.astype(str).tolist() == ids(positions["cw"]),
            "CW innovation selection regression failed")
    protected.append(innovation_selected)
    rows = [{"outer_seed": seed, "selection_order": rank, "canonical_index": int(index),
             "sample_id": sample_id} for rank, (index, sample_id) in enumerate(zip(positions["cw"], ids(positions["cw"])))]
    return rows, {"outer_seed": seed, "status": "PASS", "raw_ordered_ids_identical": True,
                  "cw_ordered_ids_identical": True, "initialization_hash": init_hash,
                  "transform": transform_audit, "label_access": list(access),
                  "source_files": {str(p): sha256_file(p) for p in protected}}


def preflight(repository, innovation, study=STUDY):
    torch.set_num_threads(2)
    study.mkdir(parents=True, exist_ok=True)
    require(not (study / "pre_test_freeze.json").exists(), "cannot overwrite a completed prediction freeze")
    primary_freeze_path = ROOT / BENCHMARK / "primary_results_freeze.json"
    primary = read_json(primary_freeze_path)
    archive_files = [primary_freeze_path]
    for relative in ("formal_results/all_metrics_b32.csv", "formal_results/global_pre_test_freeze_b32.json"):
        path = repository / BENCHMARK / relative
        require(sha256_file(path) == primary["files"][relative], "historical result archive freeze mismatch")
        archive_files.append(path)
    rows, audits = [], []
    for seed in DEVELOPMENT_SEEDS:
        print(json.dumps({"preflight_seed": seed}), flush=True)
        selected, audit = audited_seed(seed, repository.resolve(), innovation.resolve())
        rows.extend(selected)
        audits.append(audit)
    pd.DataFrame(rows).to_csv(study / "selected_batches.csv", index=False)
    atomic_json(study / "preflight_audit.json", {"seeds": audits, "test_truth_access_count": 0,
                "archive_files": {str(p): sha256_file(p) for p in archive_files},
                "full_data_reference": "NOT_COMPARABLE", "reason": "existing qualification splits use seeds 42/525/1101"})
    atomic_json(study / "environment.json", {"python": platform.python_version(), "platform": platform.platform(),
                "packages": package_versions(), "device": "cpu", "torch_threads": 2,
                "artifact_repository": str(repository.resolve()), "innovation_repository": str(innovation.resolve())})
    protected = [study / name for name in ("PROTOCOL.md", "selected_batches.csv", "preflight_audit.json", "environment.json")]
    protected += code_files()
    atomic_json(study / "acquisition_freeze.json", {"status": "FROZEN_BEFORE_SELECTED_LABEL_REVEAL",
                "seeds": list(DEVELOPMENT_SEEDS), "test_truth_access_count": 0,
                "files": {str(p): sha256_file(p) for p in protected}})
    print("PREFLIGHT_PASS: all five CW batches frozen; no selected or test truth read", flush=True)


def verify_preflight(study):
    freeze = read_json(study / "acquisition_freeze.json")
    require(freeze["seeds"] == list(DEVELOPMENT_SEEDS), "seed matrix mismatch")
    verify_files(freeze["files"])
    audit_record = read_json(study / "preflight_audit.json")
    verify_files(audit_record["archive_files"])
    for audit in audit_record["seeds"]:
        verify_files(audit["source_files"])


def decision(paired):
    require(set(paired.outer_seed) == set(DEVELOPMENT_SEEDS) and len(paired) == 5, "complete five-seed pairing required")
    require(np.isfinite(paired.select_dtypes(include=[np.number]).to_numpy()).all(), "nonfinite metrics")
    deltas = paired.CW_NRMSE - paired.Raw_NRMSE
    wins = int((deltas < 0).sum())
    endpoint_changes = {t: float(paired[f"CW_{t}_RMSE"].mean() / paired[f"Raw_{t}_RMSE"].mean() - 1)
                        for t in ("V1", "V2")}
    endpoint_ok = all(value <= 0.02 for value in endpoint_changes.values())
    result = "CENTER_WIDTH_NO_GAIN" if deltas.mean() >= 0 else (
        "CENTER_WIDTH_PROMISING" if wins >= 4 and endpoint_ok else "CENTER_WIDTH_UNSTABLE_SIGNAL")
    return {"decision": result, "wins": wins, "paired_deltas": deltas.tolist(),
            "mean_improvement": float(-deltas.mean()), "median_improvement": float(-deltas.median()),
            "paired_delta_std_ddof1": float(deltas.std(ddof=1)), "endpoint_fractional_change": endpoint_changes,
            "endpoint_guard_pass": endpoint_ok, "full_data_reference": "NOT_COMPARABLE",
            "target_aware_CW_next_stage_supported": result == "CENTER_WIDTH_PROMISING"}


def execute(repository, innovation, study=STUDY):
    verify_preflight(study)
    environment = read_json(study / "environment.json")
    require(environment["packages"] == package_versions() and environment["python"] == platform.python_version(),
            "execution environment changed")
    require(environment["artifact_repository"] == str(repository.resolve()) and
            environment["innovation_repository"] == str(innovation.resolve()), "source repository changed")
    tests = read_json(study / "tests.json")
    require(tests["status"] == "PASS" and tests["exit_code"] == 0, "passing preflight tests required")
    verify_files(tests["files"])
    selected = pd.read_csv(study / "selected_batches.csv")
    protected = {str(study / "acquisition_freeze.json"): sha256_file(study / "acquisition_freeze.json"),
                 str(study / "tests.json"): sha256_file(study / "tests.json")}
    for seed in DEVELOPMENT_SEEDS:
        print(json.dumps({"CW_fit_started": seed}), flush=True)
        partition = pd.read_csv(ROOT / BENCHMARK / "splits" / f"row_seed_{seed}.csv")
        context = SeedContext(SOURCE_DATA, study / "runtime" / f"seed_{seed}", seed, partition)
        source = repository / BENCHMARK / "runtime" / f"seed_{seed}"
        old_context = read_json(source / "scrubbed_graphs.pt.contract.json")["contract"]
        require(context.preprocessing == old_context["preprocessing"] and
                asdict(context.normalization) == old_context["normalization"], "new preprocessing incompatible")
        batch = selected.loc[selected.outer_seed.eq(seed)].sort_values("selection_order")
        validate_batch(batch.sample_id, context.ids(context.roles["u0"]))
        indices = batch.canonical_index.to_numpy(int)
        require(context.ids(indices) == batch.sample_id.tolist(), "selected index identity drift")
        context.store.freeze_acquisitions(batch.sample_id)
        truth = context.store.reveal(batch.sample_id, "after_acquisition_fit")
        fit = context.fit("center_width_lcmd", np.r_[context.roles["l0"], indices], np.vstack([context.l0_truth, truth]))
        old_audit = read_json(source / "b32/lcmd/fit_audit.json")
        require(fit["initialization_hash"] == old_audit["initialization_hash"], "evaluation initialization mismatch")
        atomic_json(context.runtime / "label_access.json", context.store.audit)
        for name in ("best.pt", "predictions.csv.gz", "fit_audit.json"):
            path = context.runtime / "center_width_lcmd" / name
            protected[str(path)] = sha256_file(path)
        for name in ("context.json", "label_access.json"):
            path = context.runtime / name
            protected[str(path)] = sha256_file(path)
        print(json.dumps({"CW_fit_complete": seed, "best_epoch": fit["best_epoch"], "epochs_run": fit["epochs_run"]}), flush=True)
    verify_preflight(study)
    verify_files(protected)
    freeze_path = study / "pre_test_freeze.json"
    record = {"status": "FROZEN_BEFORE_TEST_TRUTH", "seeds": list(DEVELOPMENT_SEEDS),
              "test_truth_access_count": 0, "files": protected}
    if freeze_path.exists():
        require(read_json(freeze_path) == record, "refusing changed prediction freeze")
    else:
        atomic_json(freeze_path, record)
    analyze(repository, study)


def analyze(repository, study):
    verify_preflight(study)
    freeze = read_json(study / "pre_test_freeze.json")
    require(freeze["seeds"] == list(DEVELOPMENT_SEEDS) and freeze["status"] == "FROZEN_BEFORE_TEST_TRUTH",
            "all-seed prediction barrier incomplete")
    verify_files(freeze["files"])
    selections = pd.read_csv(study / "selected_batches.csv")
    records, accesses = [], []
    # Parse archived metric values only for the two authorized arms and development seeds.
    archived = {}
    with (repository / BENCHMARK / "formal_results/all_metrics_b32.csv").open() as handle:
        for row in csv.DictReader(handle):
            if int(row["outer_seed"]) in DEVELOPMENT_SEEDS and row["arm"] in ("baseline_l0", "lcmd"):
                archived[(int(row["outer_seed"]), row["arm"])] = {m: float(row[m]) for m in METRICS}
    for seed in DEVELOPMENT_SEEDS:
        partition = pd.read_csv(ROOT / BENCHMARK / "splits" / f"row_seed_{seed}.csv")
        test_ids = partition.loc[partition.role.eq("test"), "sample_id"].tolist()
        store = RestrictedLabelStore(SOURCE_DATA, partition)
        store.freeze_acquisitions(selections.loc[selections.outer_seed.eq(seed), "sample_id"])
        store.freeze_predictions()
        truth = store.reveal(test_ids, "final_test_evaluation")
        accesses.extend({"outer_seed": seed, **item} for item in store.audit)
        source = repository / BENCHMARK / "runtime" / f"seed_{seed}"
        scales = read_json(source / "context.json")["preprocessing"]["target_scales"]
        paths = {"L0": source / "baseline_l0", "Raw": source / "b32/lcmd",
                 "CW": study / "runtime" / f"seed_{seed}/center_width_lcmd"}
        for method, path in paths.items():
            prediction = pd.read_csv(path / "predictions.csv.gz")
            require(prediction.sample_id.tolist() == test_ids, "evaluation prediction order mismatch")
            metrics = metric_row(truth, prediction.drop(columns="sample_id").to_numpy(), scales)
            if method in ("Raw", "L0"):
                previous = archived[(seed, "lcmd" if method == "Raw" else "baseline_l0")]
                require(all(np.isclose(metrics[m], previous[m], rtol=1e-12, atol=1e-12) for m in METRICS),
                        "historical metric reproduction failed")
            records.append({"outer_seed": seed, "method": method, **metrics})
    frame = pd.DataFrame(records)
    frame.to_csv(study / "per_seed_metrics.csv", index=False)
    pairs = []
    for seed, group in frame.groupby("outer_seed", sort=False):
        values = group.set_index("method")
        row = {"outer_seed": seed, **{f"{method}_NRMSE": values.loc[method, "combined_normalized_RMSE"]
                                      for method in ("L0", "Raw", "CW")}}
        row.update({f"{method}_{target}_RMSE": values.loc[method, f"{target}_RMSE"]
                    for method in ("Raw", "CW") for target in ("V1", "V2")})
        row["delta_CW_minus_Raw"] = row["CW_NRMSE"] - row["Raw_NRMSE"]
        row["improvement_Raw_minus_CW"] = -row["delta_CW_minus_Raw"]
        row["relative_improvement"] = row["improvement_Raw_minus_CW"] / row["Raw_NRMSE"]
        pairs.append(row)
    paired = pd.DataFrame(pairs)
    paired.to_csv(study / "paired_comparison.csv", index=False)
    aggregate = frame.groupby("method")[METRICS].agg(["mean", "median", "std"])
    aggregate.to_csv(study / "aggregate_metrics.csv")
    pd.DataFrame(accesses).to_csv(study / "test_label_access.csv", index=False)
    outcome = decision(paired)
    atomic_json(study / "decision.json", outcome)
    pd.DataFrame([{"outer_seed": s, "status": "NOT_COMPARABLE", "reason": "No matched full-data split/protocol"}
                  for s in DEVELOPMENT_SEEDS]).to_csv(study / "full_data_gap.csv", index=False)
    report(frame, paired, outcome, study)
    atomic_json(study / "artifact_manifest.json", {"status": "COMPLETE", "source_preflight": "preflight_audit.json",
                "runtime_prediction_freeze": "pre_test_freeze.json",
                "files": {str(p.relative_to(study)): sha256_file(p) for p in sorted(study.rglob("*"))
                          if p.is_file() and "runtime" not in p.relative_to(study).parts and p.name != "artifact_manifest.json"}})
    print(json.dumps(outcome), flush=True)


def report(frame, paired, outcome, study):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = study / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for _, row in paired.iterrows():
        ax.plot(["L0 (333)", "Raw (365)", "CW (365)"],
                [row.L0_NRMSE, row.Raw_NRMSE, row.CW_NRMSE], marker="o", label=str(int(row.outer_seed)))
    ax.set_ylabel("Combined NRMSE (lower is better)")
    ax.legend(title="Development seed")
    fig.tight_layout()
    fig.savefig(figures / "paired_nrmse.png", dpi=160)
    plt.close(fig)
    lines = ["# Center/Width Gradient-LCMD performance", "", f"Decision: `{outcome['decision']}`.", "",
             "| Seed | L0 | Raw B32 | CW B32 | CW - Raw |", "| --- | ---: | ---: | ---: | ---: |"]
    for _, r in paired.iterrows():
        lines.append(f"| {int(r.outer_seed)} | {r.L0_NRMSE:.9f} | {r.Raw_NRMSE:.9f} | {r.CW_NRMSE:.9f} | {r.delta_CW_minus_Raw:+.9f} |")
    lines += ["", f"CW wins {outcome['wins']}/5. Mean paired improvement (Raw-CW): {outcome['mean_improvement']:.9f}; "
              f"median: {outcome['median_improvement']:.9f}; paired delta sample std: {outcome['paired_delta_std_ddof1']:.9f}.", "",
              "| Method | NRMSE mean | median | sample std | V1 mean RMSE | V2 mean RMSE |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for method, group in frame.groupby("method"):
        e = group.combined_normalized_RMSE
        lines.append(f"| {method} | {e.mean():.9f} | {e.median():.9f} | {e.std(ddof=1):.9f} | {group.V1_RMSE.mean():.6f} | {group.V2_RMSE.mean():.6f} |")
    lines += ["", "Endpoint mean RMSE changes CW relative to Raw: " + ", ".join(
        f"{t} {v:+.2%}" for t, v in outcome["endpoint_fractional_change"].items()) + ".",
        "The frozen endpoint guard allows at most 2% deterioration on either endpoint.", "",
        "Full-data gap_closed: NOT_COMPARABLE. Existing final qualification uses seeds 42/525/1101 "
        "and full-train normalization, not these L0-normalized development splits. No new reference was trained.", "",
        "All five Raw and L0 metric vectors reproduce the historical archive within 1e-12. "
        "Only five CW scratch fits were run. Acquisition IDs and all predictions were frozen before test truth. "
        "All results are exploratory development evidence, not independent confirmation or a multi-round learning curve. "
        "The active budget is 365/3330 (10.96%); including shared validation it is 781/4163 (18.76%).", "",
        "## NEXT_STEP_RECOMMENDATION", ""]
    if outcome["decision"] == "CENTER_WIDTH_PROMISING":
        lines += ["This supports a development-stage one-step label-efficiency signal for CW output-aware gradient geometry. "
                  "Consider CW gradients with target-aware acquisition next; do not claim confirmed learning-curve superiority."]
    else:
        lines += ["This does not establish a reliable improvement in label efficiency from CW output-aware gradient geometry. "
                  "Do not add CW variants or advance CW plus Target-IVR on this evidence. Reassess target-aware acquisition "
                  "using the existing Raw gradient representation in a separately preregistered study."]
    (study / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
