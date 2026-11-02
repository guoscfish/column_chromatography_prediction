"""Matched exploratory IVR extension, with immutable controls and a test barrier."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from ..artifacts import sha256_file
from ..models import load_predictor_checkpoint
from ..resources import ROOT, SOURCE_DATA, SOURCE_GRAPH_CACHE
from ..training.predictor import atomic_json
from .benchmark_protocol import initialization_seed, sketch_seed
from .benchmark_reporting import metric_row
from .cache import seal_cache, verify_cache
from .gradient_features import extract_q50_gradient_sketches
from .ivr import conditional_batch_ivr
from .protocol import RestrictedLabelStore, ids_hash, stable_hash
from .runner import fit_from_same_initialization
from .sequential_acquisition import validate_trajectory_transition
from .sequential_protocol import (
    ACTIVE_LABEL_BUDGETS, CONFIRMATION_SEEDS, STUDY as BASELINE,
    label_budget, package_versions,
)
from .sequential_reporting import labels_to_target, normalized_aulc
from .sequential_runner import (
    SequentialSeedContext, _assert_protected, _partition, _verify_global_freeze,
    _write_json_once,
)


STUDY = ROOT / "studies/active_learning/qgeognn_v2_row_kernel_ivr_b32"
METHOD = "kernel_ivr"
ENTRY = ROOT / "scripts/studies/run_qgeognn_v2_row_kernel_ivr_b32.py"
METRIC = "combined_normalized_RMSE"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another worker owns {path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def hashes(paths) -> dict:
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(set(paths))}


def prepare(test_report: Path) -> dict:
    """Seal once, before any new training or outcome is available."""
    if (STUDY / "seal.json").exists():
        return validate_seal()
    report = ET.parse(test_report).getroot()
    suites = [report] if report.tag == "testsuite" else list(report.iter("testsuite"))
    if not suites or sum(int(s.get("tests", 0)) for s in suites) < 11 or any(
        int(s.get("failures", 0)) + int(s.get("errors", 0)) + int(s.get("skipped", 0)) for s in suites
    ):
        raise RuntimeError("passing, non-skipped preflight tests required")
    if list((STUDY / "runtime").glob("seed_*/round_*")):
        raise RuntimeError("cannot preregister after trajectory execution")
    _verify_global_freeze(BASELINE)
    baseline_paths = [BASELINE / "global_pre_test_freeze.json", BASELINE / "protocol.json",
                      BASELINE / "splits/split_manifest.json"]
    baseline_paths.extend((BASELINE / "results").glob("*.csv"))
    for seed in CONFIRMATION_SEEDS:
        old = BASELINE / "runtime" / f"seed_{seed}"
        context = read_json(old / "context.json")["contract"]
        _assert_protected(ROOT, context["code_hashes"])
        baseline_paths.extend([old / "context.json", old / "scaler.json"])
        for method in ("random", "lcmd"):
            baseline_paths.append(old / method / "fit_audit.csv")
            for round_index in range(22):
                directory = old / method / f"round_{round_index:02d}"
                baseline_paths.append(directory / "contract.json")
        baseline_paths.extend((BASELINE / "splits").glob(f"*{seed}*"))
        baseline_paths.append(old / "full_data_reference/pre_test_freeze.json")
    environment = {"python": sys.version, "executable": sys.executable,
                   "platform": platform.platform(), "packages": package_versions(),
                   "cpu_threads_per_fit": 2, "max_concurrent_seeds": 2}
    atomic_json(STUDY / "environment.json", environment)
    checks = []
    for seed in CONFIRMATION_SEEDS:
        with tempfile.TemporaryDirectory(prefix=f"ivr_preflight_{seed}_") as temporary:
            old_context = read_json(BASELINE / "runtime" / f"seed_{seed}/context.json")["contract"]
            context = SequentialSeedContext(seed, _partition(seed), Path(temporary), old_context["preregistration_commit"])
            if context.context_contract != old_context:
                raise RuntimeError("round-zero preflight context mismatch")
            model_dir, _ = _fit(context, 0, context.roles["l0"], context.l0_truth, Path(temporary))
            features, _, _ = _features(context, 0, model_dir, Path(temporary))
            result = conditional_batch_ivr(features, np.arange(333), np.arange(333, 3330), 32)
            repeat = conditional_batch_ivr(features, np.arange(333), np.arange(333, 3330), 32)
            if result["selected_candidate_positions"] != repeat["selected_candidate_positions"]:
                raise RuntimeError("round-zero preflight selection is not deterministic")
            checks.append({"seed": seed, "context_match": True, "deterministic_batch": True,
                           "selected_count": len(result["selected_candidate_positions"]), **result["audit"]})
            print(json.dumps({"engineering_preflight_seed": seed, "passed": True}), flush=True)
    atomic_json(STUDY / "engineering_preflight.json", {"test_truth_access_count": 0, "new_fits": 0, "checks": checks})
    test_copy = STUDY / "preflight_tests.xml"
    test_copy.write_bytes(test_report.read_bytes())
    paths = list((ROOT / "src/qgeognn_al").rglob("*.py")) + [ENTRY]
    paths += [ROOT / "application/QGeoGNN.py", ROOT / "application/utils.py"]
    paths += [ROOT / "tests/active_learning_v2/test_ivr.py",
              ROOT / "tests/active_learning_v2/test_ivr_study.py"]
    paths += [STUDY / name for name in ("config.json", "PROTOCOL.md", "environment.json", "preflight_tests.xml", "engineering_preflight.json")]
    seal = {"status": "SEALED_BEFORE_NEW_TRAINING", "created_at": now(),
            "evidence": "exploratory_matched_extension_on_previously_evaluated_cohort",
            "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": sha256_file(SOURCE_DATA), "graph_sha256": sha256_file(SOURCE_GRAPH_CACHE),
            "files": hashes(paths), "baseline_files": hashes(baseline_paths),
            "test_truth_access_count_new_strategy": 0}
    archive = STUDY / "pre_execution_seal_archive.json"
    if archive.exists():
        seal["pre_execution_dependency_audit_addendum"] = {
            "previous_seal_sha256": sha256_file(archive), "new_fits_before_addendum": 0,
            "reason": "include imported application model primitives in protected source hashes"}
    _write_json_once(STUDY / "seal.json", seal)
    atomic_json(STUDY / "decision.json", {"status": "SEALED_PENDING_EXECUTION", "effectiveness": "NOT_EVALUATED"})
    return {"status": seal["status"], "seal_sha256": sha256_file(STUDY / "seal.json")}


def validate_seal() -> dict:
    seal = read_json(STUDY / "seal.json")
    if seal["status"] != "SEALED_BEFORE_NEW_TRAINING":
        raise RuntimeError("invalid seal")
    _assert_protected(ROOT, seal["files"])
    _assert_protected(ROOT, seal["baseline_files"])
    if sha256_file(SOURCE_DATA) != seal["source_sha256"] or sha256_file(SOURCE_GRAPH_CACHE) != seal["graph_sha256"]:
        raise RuntimeError("source data changed")
    if package_versions() != read_json(STUDY / "environment.json")["packages"]:
        raise RuntimeError("scientific package versions changed")
    return seal


def verify_record(path: Path) -> dict:
    record = read_json(path)
    if record.get("test_truth_access_count") != 0:
        raise RuntimeError("record is not blind")
    if record.get("seal_sha256") != sha256_file(STUDY / "seal.json"):
        raise RuntimeError("record seal mismatch")
    _assert_protected(ROOT, record["files"])
    return record


def _fit(context, round_index, labeled, truth, directory):
    if round_index == 0:
        directory = BASELINE / "runtime" / f"seed_{context.seed}" / "lcmd/round_00/model"
        old = read_json(directory.parent / "input_contract.json")
        if old["training_config"] != context.training_config(0) or old["L_t_ids_hash"] != ids_hash(context.ids(labeled)):
            raise RuntimeError("round-zero training inputs do not match")
        audit = read_json(directory / "fit_audit.json")
        if audit["checkpoint_sha256"] != sha256_file(directory / "best.pt") or audit["prediction_sha256"] != sha256_file(directory / "predictions.csv.gz"):
            raise RuntimeError("round-zero artifacts changed")
        return directory, audit
    model_dir = directory / "model"
    expected_files = [model_dir / name for name in ("best.pt", "predictions.csv.gz", "fit_audit.json")]
    if any(p.exists() for p in expected_files) and not all(p.exists() for p in expected_files):
        # Interrupted scratch fits restart deterministically; preserve the partial attempt.
        destination = context.runtime / "incomplete_attempts" / f"round_{round_index:02d}_{time.time_ns()}"
        destination.parent.mkdir(exist_ok=True)
        model_dir.rename(destination)
    contract = {"study": STUDY.name, "seal_sha256": sha256_file(STUDY / "seal.json"),
                "context_hash": stable_hash(context.context_contract), "method": METHOD,
                "round": round_index, "member": 0,
                "train_sample_ids": context.ids(labeled),
                "validation_sample_ids": context.ids(context.roles["validation"]),
                "prediction_role": "test_zero_label_graphs"}
    audit = fit_from_same_initialization(
        atom_base=context.atom, angle=context.angle, normalization=context.normalization,
        preprocessing=context.preprocessing, train_indices=labeled, train_truth=truth,
        validation_indices=context.roles["validation"], validation_truth=context.validation_truth,
        prediction_indices=context.roles["test"], prediction_sample_ids=context.ids(context.roles["test"]),
        initialization_seed=initialization_seed(context.seed, 0), training_config=context.training_config(0),
        contract=contract, runtime=model_dir,
    )
    return model_dir, audit


def _features(context, round_index, model_dir, directory):
    if round_index == 0:
        artifacts = model_dir.parent / "acquisition_artifacts"
        path = artifacts / "current_gradient_features.npz"
        receipt = read_json(path.with_suffix(".npz.contract.json"))
        verify_cache(path, receipt["contract"])
        config = receipt["contract"]
        if (config["evaluation_checkpoint_sha256"] != sha256_file(model_dir / "best.pt") or
                config["sketch_seed"] != sketch_seed(context.seed) or
                config["endpoint_scales"] != context.preprocessing["target_scales"]):
            raise RuntimeError("round-zero gradient provenance mismatch")
        audit_path = artifacts / "gradient_audit.json"
    else:
        path = directory / "gradient_features.npz"
        audit_path = directory / "gradient_audit.json"
        config = {"seal_sha256": sha256_file(STUDY / "seal.json"),
                  "checkpoint_sha256": sha256_file(model_dir / "best.pt"),
                  "reference_indices": context.outer.tolist(), "dimension": 512,
                  "sketch_seed": sketch_seed(context.seed), "scales": context.preprocessing["target_scales"]}
        if not verify_cache(path, config):
            model = load_predictor_checkpoint(model_dir / "best.pt")
            result = extract_q50_gradient_sketches(
                model, context.atom, context.angle, context.outer,
                tuple(context.preprocessing["target_scales"][key] for key in ("V1", "V2")),
                dimension=512, sketch_seed=sketch_seed(context.seed),
            )
            np.savez_compressed(path, features=result.features, canonical_indices=context.outer)
            atomic_json(audit_path, result.audit)
            seal_cache(path, config)
    with np.load(path) as values:
        if not np.array_equal(values["canonical_indices"], context.outer):
            raise RuntimeError("reference universe/order changed")
        features = values["features"].copy()
    return features, read_json(audit_path), [path, path.with_suffix(".npz.contract.json"), audit_path]


def execute_seed(seed: int) -> dict:
    if seed not in CONFIRMATION_SEEDS:
        raise ValueError("seed is outside the frozen cohort")
    validate_seal()
    runtime = STUDY / "runtime" / f"seed_{seed}"
    with exclusive_lock(runtime / "worker.lock"):
        if (runtime / "trajectory_freeze.json").exists():
            return verify_record(runtime / "trajectory_freeze.json")
        if (STUDY / "global_pre_test_freeze.json").exists():
            raise RuntimeError("execution closed after global freeze")
        old_context = read_json(BASELINE / "runtime" / f"seed_{seed}/context.json")["contract"]
        # Reuse the historical feature/label preprocessing API under an exact context comparison.
        context = SequentialSeedContext(seed, _partition(seed), runtime / "context", old_context["preregistration_commit"])
        if context.context_contract != old_context:
            raise RuntimeError("new and baseline preprocessing contexts differ")
        store = context.new_method_store()
        labeled, unlabeled = context.roles["l0"].copy(), context.roles["u0"].copy()
        truth = context.l0_truth.copy()
        selected_all, freezes, fit_rows, acquisition_rows = [], [], [], []
        reference_position = {int(value): i for i, value in enumerate(context.outer)}
        seal_sha = sha256_file(STUDY / "seal.json")
        for round_index, budget in enumerate(ACTIVE_LABEL_BUDGETS):
            directory = runtime / f"round_{round_index:02d}"
            directory.mkdir(exist_ok=True)
            inputs = {"seed": seed, "round": round_index, "active_labels": budget,
                      "ordered_labeled_indices": labeled.tolist(), "ordered_unlabeled_indices": unlabeled.tolist(),
                      "ordered_labeled_ids_hash": stable_hash(context.ids(labeled)),
                      "ordered_unlabeled_ids_hash": stable_hash(context.ids(unlabeled)), "seal_sha256": seal_sha}
            _write_json_once(directory / "input.json", inputs)
            if (directory / "freeze.json").exists():
                record = verify_record(directory / "freeze.json")
                if record["input"] != inputs:
                    raise RuntimeError("resumed state differs from frozen trajectory")
                model_dir = ROOT / record["model_directory"]
                fit = read_json(model_dir / "fit_audit.json")
            else:
                validate_seal()
                print(json.dumps({"seed": seed, "round": round_index, "labels": budget, "stage": "fit", "at": now()}), flush=True)
                model_dir, fit = _fit(context, round_index, labeled, truth, directory)
                protected = [directory / "input.json"] + [model_dir / name for name in ("best.pt", "predictions.csv.gz", "fit_audit.json")]
                if round_index < 21:
                    print(json.dumps({"seed": seed, "round": round_index, "stage": "gradients_and_ivr", "at": now()}), flush=True)
                    features, gradient_audit, feature_paths = _features(context, round_index, model_dir, directory)
                    selection_path = directory / "selection.json"
                    if selection_path.exists():
                        selection = verify_record(selection_path)
                        if selection["input"] != inputs:
                            raise RuntimeError("resumed selection input changed")
                    else:
                        result = conditional_batch_ivr(
                            features, np.array([reference_position[int(i)] for i in labeled]),
                            np.array([reference_position[int(i)] for i in unlabeled]), 32)
                        selected = unlabeled[result["selected_candidate_positions"]]
                        selection = {"input": inputs, "seal_sha256": seal_sha,
                                     "test_truth_access_count": 0, "selected_indices": selected.tolist(),
                                     "selected_ids": context.ids(selected), "trace": result["trace"],
                                     "ivr_audit": result["audit"], "gradient_audit": gradient_audit,
                                     "reused_gradients": round_index == 0, "files": hashes(feature_paths)}
                        _write_json_once(selection_path, selection)
                    protected.extend(feature_paths + [selection_path])
                record = {"input": inputs, "status": "FROZEN_BEFORE_NEW_STRATEGY_TEST_REVEAL",
                          "seal_sha256": seal_sha, "test_truth_access_count": 0,
                          "model_directory": str(model_dir.relative_to(ROOT)), "fit_reused": round_index == 0,
                          "files": hashes(protected)}
                _write_json_once(directory / "freeze.json", record)
            baseline_init = read_json(BASELINE / "runtime" / f"seed_{seed}/lcmd/round_00/model/fit_audit.json")["initialization_hash"]
            if fit["initialization_hash"] != baseline_init or fit["train_rows"] != budget:
                raise RuntimeError("initialization or label budget mismatch")
            freezes.append(directory / "freeze.json")
            fit_rows.append({"outer_seed": seed, "method": METHOD, "round": round_index,
                             "reused": round_index == 0, **fit})
            atomic_json(runtime / "progress.json", {"seed": seed, "completed_round_points": round_index + 1,
                        "active_labels": budget, "test_truth_access_count": 0, "at": now()})
            print(json.dumps({"seed": seed, "round_frozen": round_index, "labels": budget, "at": now()}), flush=True)
            if round_index < 21:
                selection = verify_record(directory / "selection.json")
                selected = np.array(selection["selected_indices"], dtype=int)
                selected_ids = selection["selected_ids"]
                if context.ids(selected) != selected_ids:
                    raise RuntimeError("selected ID/index mismatch")
                next_labeled = np.r_[labeled, selected]
                selected_set = set(selected.tolist())
                next_unlabeled = np.array([i for i in unlabeled if i not in selected_set], dtype=int)
                validate_trajectory_transition(context.ids(labeled), context.ids(unlabeled), selected_ids,
                                               context.ids(next_labeled), context.ids(next_unlabeled))
                selected_all.extend(selected_ids)
                store.freeze_acquisitions(selected_all)
                truth = np.vstack([truth, store.reveal(selected_ids, "after_acquisition_fit")])
                labeled, unlabeled = next_labeled, next_unlabeled
                acquisition_rows.append({"outer_seed": seed, "round": round_index,
                    "gradient_seconds": selection["gradient_audit"]["elapsed_seconds"],
                    "gradient_reused": round_index == 0, "ivr_seconds": selection["ivr_audit"]["elapsed_seconds"]})
        pd.DataFrame(fit_rows).to_csv(runtime / "fit_audit.csv", index=False)
        pd.DataFrame(acquisition_rows).to_csv(runtime / "acquisition_runtime.csv", index=False)
        pd.DataFrame(store.audit).to_csv(runtime / "label_access_audit.csv", index=False)
        trajectory = {"seed": seed, "seal_sha256": seal_sha, "round_points": 22,
                      "status": "FROZEN_BEFORE_NEW_STRATEGY_TEST_REVEAL", "test_truth_access_count": 0,
                      "selected_ids_hash": stable_hash(selected_all), "final_active_labels": len(labeled),
                      "files": hashes(freezes + [runtime / name for name in
                                                ("fit_audit.csv", "acquisition_runtime.csv", "label_access_audit.csv")])}
        _write_json_once(runtime / "trajectory_freeze.json", trajectory)
        return trajectory


def global_freeze() -> dict:
    validate_seal()
    _verify_global_freeze(BASELINE)
    trajectories = []
    for seed in CONFIRMATION_SEEDS:
        runtime = STUDY / "runtime" / f"seed_{seed}"
        record = verify_record(runtime / "trajectory_freeze.json")
        if record["round_points"] != 22 or record["final_active_labels"] != 1005:
            raise RuntimeError("incomplete trajectory")
        previous_l, previous_u = None, None
        partition = _partition(seed)
        id_by_index = dict(zip(partition.canonical_index, partition.sample_id.astype(str)))
        for round_index, budget in enumerate(ACTIVE_LABEL_BUDGETS):
            directory = runtime / f"round_{round_index:02d}"
            row = verify_record(directory / "freeze.json")["input"]
            labeled = [id_by_index[i] for i in row["ordered_labeled_indices"]]
            unlabeled = [id_by_index[i] for i in row["ordered_unlabeled_indices"]]
            if len(labeled) != budget or row["round"] != round_index or row["seed"] != seed:
                raise RuntimeError("frozen round identity/budget mismatch")
            if round_index:
                selected = verify_record(runtime / f"round_{round_index - 1:02d}/selection.json")["selected_ids"]
                validate_trajectory_transition(previous_l, previous_u, selected, labeled, unlabeled)
            previous_l, previous_u = labeled, unlabeled
        trajectories.append(runtime / "trajectory_freeze.json")
    result = {"status": "ALL_110_POINTS_FROZEN_BEFORE_NEW_STRATEGY_TEST_REVEAL",
              "seal_sha256": sha256_file(STUDY / "seal.json"), "test_truth_access_count": 0,
              "files": hashes(trajectories)}
    _write_json_once(STUDY / "global_pre_test_freeze.json", result)
    return result


def run_all() -> dict:
    """Run at most two independent seed processes, then freeze and report once."""
    validate_seal()
    with exclusive_lock(STUDY / "runtime/scheduler.lock"):
        pending = list(CONFIRMATION_SEEDS)
        running = {}
        failure = None
        atomic_json(STUDY / "decision.json", {"status": "RUNNING", "effectiveness": "NOT_EVALUATED", "started_at": now()})
        while pending or running:
            while pending and len(running) < 2 and failure is None:
                seed = pending.pop(0)
                log_path = STUDY / "runtime" / f"seed_{seed}.log"
                log = log_path.open("a")
                env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                           MKL_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")
                child = subprocess.Popen([sys.executable, str(ENTRY), "--execute-seed", str(seed)],
                                         cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                running[seed] = (child, log)
            for seed, (child, log) in list(running.items()):
                code = child.poll()
                if code is not None:
                    log.close()
                    del running[seed]
                    if code != 0:
                        failure = {"seed": seed, "exit_code": code}
            atomic_json(STUDY / "runtime/scheduler_status.json",
                        {"pid": os.getpid(), "running": {str(s): p.pid for s, (p, _) in running.items()},
                         "pending": pending, "failure": failure, "updated_at": now()})
            if failure and not running:
                atomic_json(STUDY / "decision.json", {"status": "EXECUTION_FAILED", "failure": failure, "effectiveness": "NOT_EVALUATED"})
                raise RuntimeError(f"seed execution failed: {failure}")
            if pending or running:
                time.sleep(5)
        return report()


def target_crossings(labels, errors, target) -> dict:
    interpolated = labels_to_target(np.asarray(labels), np.asarray(errors), target)
    reached = np.flatnonzero(np.asarray(errors) <= target)
    sustained = [i for i in reached if np.all(np.asarray(errors)[i:] <= target)]
    return {"interpolated_labels": interpolated,
            "first_observed_labels": int(labels[reached[0]]) if len(reached) else None,
            "sustained_through_final_labels": int(labels[sustained[0]]) if sustained else None,
            "censored": interpolated is None}


def report() -> dict:
    global_freeze()
    rows, access, costs = [], [], []
    for seed in CONFIRMATION_SEEDS:
        runtime = STUDY / "runtime" / f"seed_{seed}"
        partition = _partition(seed)
        test_ids = partition.loc[partition.role.eq("test"), "sample_id"].astype(str).tolist()
        scales = read_json(runtime / "context/context.json")["contract"]["preprocessing"]["target_scales"]
        selected = [value for r in range(21) for value in read_json(runtime / f"round_{r:02d}/selection.json")["selected_ids"]]
        store = RestrictedLabelStore(SOURCE_DATA, partition)
        store.freeze_acquisitions(selected)
        store.freeze_predictions()
        truth = store.reveal(test_ids, "final_test_evaluation")
        access.extend({"outer_seed": seed, **row} for row in store.audit)
        for round_index, budget in enumerate(ACTIVE_LABEL_BUDGETS):
            frozen = read_json(runtime / f"round_{round_index:02d}/freeze.json")
            prediction = pd.read_csv(ROOT / frozen["model_directory"] / "predictions.csv.gz")
            if prediction.sample_id.astype(str).tolist() != test_ids:
                raise RuntimeError("prediction identity/order mismatch")
            rows.append({"outer_seed": seed, "method": METHOD, "round": round_index,
                         **label_budget(budget), **metric_row(truth, prediction.drop(columns="sample_id").to_numpy(float), scales)})
        for method in ("random", "lcmd", METHOD):
            base = runtime if method == METHOD else BASELINE / "runtime" / f"seed_{seed}" / method
            fits = pd.read_csv(base / "fit_audit.csv")
            if method == METHOD:
                acq = pd.read_csv(runtime / "acquisition_runtime.csv")
                gradient_s, ivr_s = float(acq.gradient_seconds.sum()), float(acq.ivr_seconds.sum())
                new_train_s = float(fits.loc[~fits.reused, "training_seconds"].sum())
                new_gradient_s = float(acq.loc[~acq.gradient_reused, "gradient_seconds"].sum())
            else:
                gradient_s = sum(read_json(base / f"round_{r:02d}/acquisition_artifacts/gradient_audit.json")["elapsed_seconds"] for r in range(21)) if method == "lcmd" else 0.
                ivr_s, new_train_s, new_gradient_s = 0., 0., 0.
            costs.append({"outer_seed": seed, "method": method, "fit_count": len(fits),
                          "epochs": int(fits.epochs_run.sum()), "training_seconds": float(fits.training_seconds.sum()),
                          "gradient_seconds": gradient_s, "ivr_seconds": ivr_s,
                          "new_training_seconds": new_train_s, "new_gradient_seconds": new_gradient_s})
    old = pd.read_csv(BASELINE / "results/learning_curve_metrics.csv")
    combined = pd.concat([old.loc[old.method.isin(["random", "lcmd"])], pd.DataFrame(rows)], ignore_index=True)
    full = pd.read_csv(BASELINE / "results/full_data_reference.csv")
    return write_report(combined, full, pd.DataFrame(costs), pd.DataFrame(access))


def write_report(curves, full, costs, access) -> dict:
    results = STUDY / "results"
    results.mkdir(exist_ok=True)
    expected = {(s, m, r) for s in CONFIRMATION_SEEDS for m in ("random", "lcmd", METHOD) for r in range(22)}
    if len(curves) != len(expected) or set(zip(curves.outer_seed, curves.method, curves["round"])) != expected:
        raise RuntimeError("incomplete comparison matrix")
    if not np.isfinite(curves[[METRIC, "V1_RMSE", "V2_RMSE"]].to_numpy()).all():
        raise RuntimeError("nonfinite evaluation")
    means = curves.groupby(["method", "active_label_count"])[[METRIC, "V1_RMSE", "V2_RMSE"]].mean().reset_index()
    aulc, targets = [], []
    for seed, seed_rows in curves.groupby("outer_seed"):
        initial = seed_rows.loc[seed_rows["round"].eq(0), METRIC].to_numpy()
        if not np.allclose(initial, initial[0], rtol=0, atol=1e-12):
            raise RuntimeError("initial predictions differ between methods")
        e0, ef = float(initial[0]), float(full.loc[full.outer_seed.eq(seed), METRIC].iloc[0])
        random_target = float(seed_rows.loc[seed_rows.method.eq("random") & seed_rows["round"].eq(21), METRIC].iloc[0])
        for method, values in seed_rows.groupby("method"):
            values = values.sort_values("active_label_count")
            labels, errors = values.active_label_count.to_numpy(), values[METRIC].to_numpy()
            _, area = normalized_aulc(labels, errors)
            aulc.append({"outer_seed": seed, "method": method, "normalized_AULC": area})
            for name, threshold in {"Random@1005": random_target, "N80": ef + .2 * (e0 - ef),
                                    "N90": ef + .1 * (e0 - ef), "N95": ef + .05 * (e0 - ef)}.items():
                targets.append({"scope": "seed", "outer_seed": seed, "method": method, "target": name,
                                "threshold": threshold, **target_crossings(labels, errors, threshold)})
    e0 = float(means.loc[means.method.eq(METHOD) & means.active_label_count.eq(333), METRIC].iloc[0])
    ef = float(full[METRIC].mean())
    random_target = float(means.loc[means.method.eq("random") & means.active_label_count.eq(1005), METRIC].iloc[0])
    for method, values in means.groupby("method"):
        values = values.sort_values("active_label_count")
        for name, threshold in {"Random@1005": random_target, "N80": ef + .2 * (e0 - ef),
                                "N90": ef + .1 * (e0 - ef), "N95": ef + .05 * (e0 - ef)}.items():
            targets.append({"scope": "cohort_mean", "outer_seed": None, "method": method, "target": name,
                            "threshold": threshold, **target_crossings(values.active_label_count.to_numpy(), values[METRIC].to_numpy(), threshold)})
    areas = pd.DataFrame(aulc)
    paired = areas.pivot(index="outer_seed", columns="method", values="normalized_AULC")
    for comparator in ("random", "lcmd"):
        paired[f"ivr_minus_{comparator}"] = paired[METHOD] - paired[comparator]
    avg_area = areas.groupby("method").normalized_AULC.mean().to_dict()
    target_frame = pd.DataFrame(targets)
    target_mean = target_frame.loc[target_frame.scope.eq("cohort_mean")]
    ivr_n80 = target_mean.loc[target_mean.method.eq(METHOD) & target_mean.target.eq("N80"), "first_observed_labels"].iloc[0]
    lcmd_n80 = target_mean.loc[target_mean.method.eq("lcmd") & target_mean.target.eq("N80"), "first_observed_labels"].iloc[0]
    endpoint_final = means.loc[means.active_label_count.eq(1005)].set_index("method")
    endpoint_ok = all(endpoint_final.loc[METHOD, k] <= 1.02 * endpoint_final.loc["lcmd", k] for k in ("V1_RMSE", "V2_RMSE"))
    wins = int((paired.ivr_minus_lcmd < 0).sum())
    gain = 1 - avg_area[METHOD] / avg_area["lcmd"]
    supported = bool(gain >= .02 and wins >= 4 and endpoint_ok and pd.notna(ivr_n80) and pd.notna(lcmd_n80) and ivr_n80 <= lcmd_n80)
    decision = {"status": "COMPLETED", "evidence": "exploratory_reused_cohort_no_independent_confirmation",
                "decision": "PROMISING_IVR_EXTENSION" if supported else "NO_CLEAR_IVR_IMPROVEMENT",
                "mean_normalized_AULC": avg_area, "relative_AULC_gain_vs_lcmd": gain,
                "paired_AULC_wins_vs_lcmd": wins, "endpoint_2pct_noninferiority": bool(endpoint_ok),
                "mean_curve_N80_not_later_than_lcmd": bool(pd.notna(ivr_n80) and pd.notna(lcmd_n80) and ivr_n80 <= lcmd_n80),
                "automatic_followup_authorized": False, "completed_at": now()}
    for name, frame in {"learning_curve_metrics": curves, "cohort_mean_learning_curve": means,
                        "normalized_aulc": areas, "paired_aulc": paired.reset_index(), "labels_to_target": target_frame,
                        "runtime": costs, "test_label_access_audit": access}.items():
        frame.to_csv(results / f"{name}.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, color in (("random", "#777777"), ("lcmd", "#167c80"), (METHOD, "#bd3f59")):
        frame = means.loc[means.method.eq(method)].sort_values("active_label_count")
        ax.plot(frame.active_label_count, frame[METRIC], label=method, color=color, marker=".")
    ax.axhline(ef, label="Full data", color="#222222", linestyle="--")
    ax.set(xlabel="Active labels (+416 shared validation)", ylabel="Combined normalized RMSE")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    figures = STUDY / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "nrmse_vs_active_labels.png", dpi=180)
    plt.close(fig)
    text = ["# Kernel-IVR matched sequential experiment", "", f"Decision: `{decision['decision']}`.", "",
            "Exploratory extension on the previously evaluated cohort; no independent confirmation.", "",
            "| Method | Mean AULC | NRMSE at 1005 |", "|---|---:|---:|"]
    for method in ("random", "lcmd", METHOD):
        text.append(f"| {method} | {avg_area[method]:.6f} | {endpoint_final.loc[method, METRIC]:.6f} |")
    text += ["", f"IVR paired AULC wins versus LCMD: {wins}/5; relative mean gain: {gain:.2%}.", "",
             "| Method | Target | Interpolated labels | First observed budget | Sustained through final |", "|---|---|---:|---:|---:|"]
    def display(value):
        return ">1005" if pd.isna(value) else f"{value:.2f}"
    for _, row in target_mean.iterrows():
        text.append(f"| {row.method} | {row.target} | {display(row.interpolated_labels)} | {display(row.first_observed_labels)} | {display(row.sustained_through_final_labels)} |")
    text += ["", "Targets use the cohort-mean curve; per-seed crossings are reported separately.",
             "Censored counts are never extrapolated. Random uses empirical first crossing, not a fixed 1005 target count.",
             "Validation adds 416 labels to every active budget. Overlapping splits prevent an independent-seed significance claim.",
             "", "## Cost", "", "Timing is descriptive because historical machine load differs.",
             "Runtime tables separate training, gradients, IVR, and newly incurred versus reused computation.",
             "The IVR arm reuses five round-zero models and gradient matrices; 105 new scratch fits were required.",
             "", "## Interpretation", "", "The scalar row-kernel surrogate is not an exact two-output Bayesian posterior.",
             "The frozen unit regularization is one hypothesis, not a tuned optimum. No automatic variants or budget extension follow this result."]
    (STUDY / "FINAL_REPORT.md").write_text("\n".join(text) + "\n")
    atomic_json(STUDY / "decision.json", decision)
    return decision
