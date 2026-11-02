"""Frozen matched-budget Static/OneShot versus historical Adaptive LCMD."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from ..artifacts import sha256_file
from ..resources import ROOT, SOURCE_DATA, SOURCE_GRAPH_CACHE
from ..training.predictor import atomic_json
from .benchmark_protocol import initialization_seed, sketch_seed
from .benchmark_reporting import metric_row
from .cache import verify_cache
from .efficiency_reporting import METRICS, evaluate, historical_costs, write_evaluation
from .ivr_study import exclusive_lock, hashes, read_json
from .lcmd import lcmd_tp_select
from .protocol import RestrictedLabelStore, ids_hash, stable_hash
from .runner import fit_from_same_initialization
from .sequential_protocol import CONFIRMATION_SEEDS, STUDY as BASELINE, label_budget, package_versions
from .sequential_runner import SequentialSeedContext, _assert_protected, _partition, _verify_global_freeze, _write_json_once


STUDY = ROOT / "studies/active_learning/qgeognn_v2_batch_adaptivity"
BUDGETS = tuple(range(333, 654, 32))


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_seal():
    seal = read_json(STUDY / "seal.json")
    _assert_protected(ROOT, seal["files"])
    _assert_protected(ROOT, seal["baseline_files"])
    if sha256_file(SOURCE_DATA) != seal["source_sha256"] or sha256_file(SOURCE_GRAPH_CACHE) != seal["graph_sha256"]:
        raise RuntimeError("dataset drift")
    if package_versions() != seal["packages"]:
        raise RuntimeError("dependency drift")
    return seal


def verify_record(path):
    record = read_json(path)
    if record["seal_sha256"] != sha256_file(STUDY / "seal.json") or record["test_truth_access_count"] != 0:
        raise RuntimeError("invalid pre-test record")
    _assert_protected(ROOT, record["files"])
    return record


def freeze(path, files, **fields):
    record = {"seal_sha256": sha256_file(STUDY / "seal.json"),
              "test_truth_access_count": 0, "files": hashes(files), **fields}
    _write_json_once(path, record)
    return record


def prepare(test_report):
    if (STUDY / "seal.json").exists():
        return validate_seal()
    if list((STUDY / "runtime").glob("seed_*")):
        raise RuntimeError("cannot seal after execution")
    xml = ET.parse(test_report).getroot()
    suites = [xml] if xml.tag == "testsuite" else list(xml.iter("testsuite"))
    if not suites or sum(int(s.get("tests", 0)) for s in suites) < 8 or any(
            int(s.get("failures", 0)) + int(s.get("errors", 0)) + int(s.get("skipped", 0)) for s in suites):
        raise RuntimeError("passing non-skipped tests required")
    _verify_global_freeze(BASELINE)
    config = read_json(STUDY / "config.json")
    if config["seeds"] != list(CONFIRMATION_SEEDS) or config["final_labels"] != BUDGETS[-1]:
        raise RuntimeError("configuration mismatch")
    baseline_files = [BASELINE / "global_pre_test_freeze.json", BASELINE / "protocol.json"]
    baseline_files += list((BASELINE / "results").glob("*.csv"))
    for seed in CONFIRMATION_SEEDS:
        runtime = BASELINE / "runtime" / f"seed_{seed}"
        baseline_files.extend([runtime / "context.json", runtime / "scaler.json"])
        baseline_files.extend((BASELINE / "splits").glob(f"*{seed}*"))
        for method in ("lcmd", "random"):
            baseline_files.append(runtime / method / "fit_audit.csv")
            for r in range(11):
                directory = runtime / method / f"round_{r:02d}"
                baseline_files += [directory / "contract.json", directory / "input_contract.json"]
                contract = read_json(directory / "contract.json")
                _assert_protected(directory, contract["files"])
        directory = runtime / "lcmd/round_00/acquisition_artifacts"
        baseline_files += [directory / name for name in ("contract.json", "current_gradient_features.npz",
                          "current_gradient_features.npz.contract.json", "gradient_audit.json")]
    copied_tests = STUDY / "preflight_tests.xml"
    copied_tests.write_bytes(Path(test_report).read_bytes())
    code = list((ROOT / "src/qgeognn_al").rglob("*.py"))
    code += [ROOT / "application/QGeoGNN.py", ROOT / "application/utils.py",
             ROOT / "scripts/studies/run_qgeognn_v2_batch_adaptivity.py"]
    code += [STUDY / "config.json", STUDY / "PROTOCOL.md", copied_tests]
    seal = {"status": "SEALED_BEFORE_ACQUISITION_AND_TRAINING", "created_at": now(),
            "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "files": hashes(code), "baseline_files": hashes(baseline_files), "packages": package_versions(),
            "source_sha256": sha256_file(SOURCE_DATA), "graph_sha256": sha256_file(SOURCE_GRAPH_CACHE)}
    _write_json_once(STUDY / "seal.json", seal)
    return {"status": seal["status"]}


def select_static(features, indices, l0, u0):
    if not np.array_equal(indices, np.r_[l0, u0]):
        raise RuntimeError("initial gradient bank order mismatch")
    result = lcmd_tp_select(features[len(l0):], features[:len(l0)], 320)
    return np.asarray(u0)[result.selected_pool_positions], result.trace


def select_all():
    validate_seal()
    if (STUDY / "acquisition_freeze.json").exists():
        return verify_record(STUDY / "acquisition_freeze.json")
    paths, checks = [], []
    for seed in CONFIRMATION_SEEDS:
        partition = _partition(seed)
        old = BASELINE / "runtime" / f"seed_{seed}/lcmd/round_00"
        bank = old / "acquisition_artifacts/current_gradient_features.npz"
        receipt = read_json(bank.with_suffix(".npz.contract.json"))
        if not verify_cache(bank, receipt["contract"]):
            raise RuntimeError("invalid initial gradient cache")
        initial = read_json(old / "input_contract.json")
        if (receipt["contract"]["evaluation_checkpoint_sha256"] != sha256_file(old / "model/best.pt") or
                receipt["contract"]["sketch_seed"] != sketch_seed(seed) or
                receipt["contract"]["endpoint_scales"] != initial["endpoint_scales"]):
            raise RuntimeError("gradient provenance mismatch")
        l0 = partition.loc[partition.role.eq("l0"), "canonical_index"].to_numpy(int)
        u0 = partition.loc[partition.role.eq("u0"), "canonical_index"].to_numpy(int)
        started = time.perf_counter()
        with np.load(bank) as cache:
            selected, trace = select_static(cache["features"], cache["canonical_indices"], l0, u0)
        elapsed = time.perf_counter() - started
        ids = partition.iloc[selected].sample_id.astype(str).tolist()
        historical = pd.read_csv(old / "acquisition_artifacts/selected_next_batch.csv").sample_id.astype(str).tolist()
        if ids[:32] != historical or len(set(ids)) != 320:
            raise RuntimeError("static first batch does not match historical LCMD")
        directory = STUDY / "selections" / f"seed_{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        selection = directory / "ordered_b320.csv"
        pd.DataFrame({"selection_order": range(320), "canonical_index": selected, "sample_id": ids}).to_csv(selection, index=False)
        pd.DataFrame(trace).to_csv(directory / "lcmd_trace.csv", index=False)
        paths += [selection, directory / "lcmd_trace.csv"]
        checks.append({"outer_seed": seed, "prefix32_exact_match": True, "selector_seconds": elapsed,
                       "selection_hash": stable_hash(ids)})
    return freeze(STUDY / "acquisition_freeze.json", paths, checks=checks, created_at=now())


def verify_reused_fit(context, r, labeled):
    directory = BASELINE / "runtime" / f"seed_{context.seed}/lcmd/round_{r:02d}"
    contract = read_json(directory / "contract.json")
    _assert_protected(directory, contract["files"])
    old = read_json(directory / "input_contract.json")
    if (old["ordered_L_t_hash"] != stable_hash(context.ids(labeled)) or
            old["training_config"] != context.training_config(0) or
            old["endpoint_scales"] != context.preprocessing["target_scales"] or
            old["model_initialization_seed"] != initialization_seed(context.seed, 0)):
        raise RuntimeError("reused fit input mismatch")
    return directory / "model", read_json(directory / "model/fit_audit.json")


def execute_seed(seed):
    if seed not in CONFIRMATION_SEEDS:
        raise ValueError("unknown seed")
    validate_seal()
    verify_record(STUDY / "acquisition_freeze.json")
    runtime = STUDY / "runtime" / f"seed_{seed}"
    with exclusive_lock(runtime / "worker.lock"):
        if (runtime / "trajectory_freeze.json").exists():
            return verify_record(runtime / "trajectory_freeze.json")
        if (STUDY / "global_pre_test_freeze.json").exists():
            raise RuntimeError("execution closed after global freeze")
        old_context = read_json(BASELINE / "runtime" / f"seed_{seed}/context.json")["contract"]
        context = SequentialSeedContext(seed, _partition(seed), runtime / "context", old_context["preregistration_commit"])
        if context.context_contract != old_context:
            raise RuntimeError("preprocessing context mismatch")
        sequence = pd.read_csv(STUDY / "selections" / f"seed_{seed}/ordered_b320.csv")
        selected = sequence.canonical_index.to_numpy(int)
        store = context.new_method_store()
        fits, paths = [], []
        initial_hash = read_json(BASELINE / "runtime" / f"seed_{seed}/lcmd/round_00/model/fit_audit.json")["initialization_hash"]
        for r, budget in enumerate(BUDGETS):
            labeled = np.r_[context.roles["l0"], selected[:32*r]]
            if len(labeled) != budget or len(set(labeled)) != budget:
                raise RuntimeError("invalid static prefix")
            directory = runtime / f"round_{r:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            if r <= 1:
                model, audit = verify_reused_fit(context, r, labeled)
            else:
                store.freeze_acquisitions(context.ids(selected[:32*r]))
                truth = np.vstack([context.l0_truth, store.reveal(context.ids(selected[:32*r]), "after_acquisition_fit")])
                model = directory / "model"
                expected = [model / name for name in ("best.pt", "predictions.csv.gz", "fit_audit.json")]
                if any(p.exists() for p in expected) and not all(p.exists() for p in expected):
                    destination = runtime / "incomplete_attempts" / f"round_{r:02d}_{time.time_ns()}"
                    destination.parent.mkdir(exist_ok=True)
                    model.rename(destination)
                audit = fit_from_same_initialization(
                    atom_base=context.atom, angle=context.angle, normalization=context.normalization,
                    preprocessing=context.preprocessing, train_indices=labeled, train_truth=truth,
                    validation_indices=context.roles["validation"], validation_truth=context.validation_truth,
                    prediction_indices=context.roles["test"], prediction_sample_ids=context.ids(context.roles["test"]),
                    initialization_seed=initialization_seed(seed, 0), training_config=context.training_config(0),
                    contract={"study": STUDY.name, "seal_sha256": sha256_file(STUDY / "seal.json"),
                              "acquisition_sha256": sha256_file(STUDY / "acquisition_freeze.json"),
                              "method": "static_lcmd", "round": r,
                              "train_sample_ids": context.ids(labeled),
                              "validation_sample_ids": context.ids(context.roles["validation"]),
                              "prediction_role": "test_zero_label_graphs"}, runtime=model)
            if audit["initialization_hash"] != initial_hash or audit["train_ids_hash"] != ids_hash(context.ids(labeled)):
                raise RuntimeError("fit initialization or training set mismatch")
            files = [model / name for name in ("best.pt", "predictions.csv.gz", "fit_audit.json")]
            freeze(directory / "freeze.json", files, outer_seed=seed, round=r, active_labels=budget,
                   ordered_train_ids=context.ids(labeled), model_directory=str(model.relative_to(ROOT)), reused=r<=1)
            paths += [directory / "freeze.json", *files]
            fits.append({"outer_seed": seed, "method": "static_lcmd", "round": r, "reused": r<=1, **audit})
            print(json.dumps({"seed": seed, "round": r, "active_labels": budget, "reused": r<=1,
                              "epochs": audit["epochs_run"], "fit_seconds": audit["training_seconds"]}), flush=True)
        pd.DataFrame(fits).to_csv(runtime / "fit_audit.csv", index=False)
        pd.DataFrame(store.audit).to_csv(runtime / "label_access_audit.csv", index=False)
        return freeze(runtime / "trajectory_freeze.json", paths + [runtime / "fit_audit.csv", runtime / "label_access_audit.csv"],
                      outer_seed=seed, points=len(BUDGETS), final_active_labels=653)


def finalize():
    validate_seal()
    verify_record(STUDY / "acquisition_freeze.json")
    paths = []
    for seed in CONFIRMATION_SEEDS:
        path = STUDY / "runtime" / f"seed_{seed}/trajectory_freeze.json"
        record = verify_record(path)
        if record["outer_seed"] != seed or record["points"] != 11 or record["final_active_labels"] != 653:
            raise RuntimeError("incomplete static trajectory")
        for r, budget in enumerate(BUDGETS):
            row = verify_record(path.parent / f"round_{r:02d}/freeze.json")
            if row["round"] != r or row["active_labels"] != budget:
                raise RuntimeError("incomplete budget schedule")
        paths.append(path)
    if (STUDY / "global_pre_test_freeze.json").exists():
        return verify_record(STUDY / "global_pre_test_freeze.json")
    return freeze(STUDY / "global_pre_test_freeze.json", paths, created_at=now(), points=55)


def reveal_and_report():
    finalize()
    rows, access, costs, overlap = [], [], [], []
    acquisition = verify_record(STUDY / "acquisition_freeze.json")
    for seed in CONFIRMATION_SEEDS:
        runtime = STUDY / "runtime" / f"seed_{seed}"
        partition = _partition(seed)
        test_ids = partition.loc[partition.role.eq("test"), "sample_id"].astype(str).tolist()
        selected = pd.read_csv(STUDY / "selections" / f"seed_{seed}/ordered_b320.csv").sample_id.astype(str).tolist()
        store = RestrictedLabelStore(SOURCE_DATA, partition)
        store.freeze_acquisitions(selected)
        store.freeze_predictions()
        truth = store.reveal(test_ids, "final_test_evaluation")
        access.extend({"outer_seed": seed, **row} for row in store.audit)
        scales = read_json(BASELINE / "runtime" / f"seed_{seed}/context.json")["contract"]["preprocessing"]["target_scales"]
        for r, budget in enumerate(BUDGETS):
            record = verify_record(runtime / f"round_{r:02d}/freeze.json")
            prediction = pd.read_csv(ROOT / record["model_directory"] / "predictions.csv.gz")
            if prediction.sample_id.astype(str).tolist() != test_ids:
                raise RuntimeError("test prediction order mismatch")
            rows.append({"outer_seed": seed, "method": "static_lcmd", "round": r, **label_budget(budget),
                         **metric_row(truth, prediction.drop(columns="sample_id").to_numpy(float), scales)})
            adaptive = np.load(BASELINE / "runtime" / f"seed_{seed}/lcmd/round_{r:02d}/labeled_ids.npy").astype(str).tolist()[333:]
            overlap.append({"outer_seed": seed, "active_label_count": budget,
                            "selected_intersection": len(set(selected[:32*r]) & set(adaptive)),
                            "selected_overlap_fraction": len(set(selected[:32*r]) & set(adaptive))/(32*r) if r else np.nan})
        fits = pd.read_csv(runtime / "fit_audit.csv")
        gradient = read_json(BASELINE / "runtime" / f"seed_{seed}/lcmd/round_00/acquisition_artifacts/gradient_audit.json")["elapsed_seconds"]
        selector = next(c["selector_seconds"] for c in acquisition["checks"] if c["outer_seed"] == seed)
        for method, subset in (("static_lcmd", fits), ("oneshot_b320", fits.loc[fits["round"].isin([0, 10])])):
            costs.append({"outer_seed": seed, "method": method, "fit_count": len(subset),
                          "epochs": int(subset.epochs_run.sum()), "optimizer_steps": int(subset.epochs_run.sum()),
                          "training_seconds": float(subset.training_seconds.sum()), "gradient_seconds": gradient,
                          "gradient_extractions": 1, "selector_seconds": selector, "extra_ensemble_fits": 0,
                          "ensemble_inference_seconds": 0.,
                          "new_fit_count": int((~subset.reused).sum()) if method == "static_lcmd" else 0,
                          "new_training_seconds": float(subset.loc[~subset.reused, "training_seconds"].sum()) if method == "static_lcmd" else 0.,
                          "cost_note": "OneShot final fit shared with Static; do not sum logical arms"})
    old = pd.read_csv(BASELINE / "results/learning_curve_metrics.csv")
    old = old.loc[old.method.isin(["random", "lcmd"]) & old.active_label_count.le(653)].copy()
    old.loc[old.method.eq("lcmd"), "method"] = "adaptive_lcmd"
    curves = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    full = pd.read_csv(BASELINE / "results/full_data_reference.csv")
    reused_cost = historical_costs(10, include_ivr=False)
    reused_cost = reused_cost.loc[reused_cost.method.isin(["random", "lcmd"])].copy()
    reused_cost.loc[reused_cost.method.eq("lcmd"), "method"] = "adaptive_lcmd"
    # The historical last round acquired the next batch for its longer trajectory.
    # That round elapsed time is not a valid 653-label runtime estimate.
    reused_cost["historical_round_elapsed_seconds"] = np.nan
    tables = evaluate(curves, full)
    tables["test_label_access_audit"] = pd.DataFrame(access)
    tables["selection_overlap"] = pd.DataFrame(overlap)
    endpoints = curves.loc[curves.active_label_count.eq(653)].copy()
    one_shot = endpoints.loc[endpoints.method.eq("static_lcmd")].copy()
    one_shot["method"] = "oneshot_b320"
    tables["endpoint_comparison"] = pd.concat([endpoints, one_shot], ignore_index=True)
    cost_frame = pd.concat([reused_cost, pd.DataFrame(costs)], ignore_index=True)
    write_evaluation(tables, cost_frame, STUDY)
    summary = tables["metric_summary"].set_index(["method", "metric"])
    def value(method, metric, field):
        return float(summary.loc[(method, metric), field])
    paired = endpoints.pivot(index="outer_seed", columns="method", values=METRICS[0])
    conditions = {
        "mean_NRMSE_improves": value("adaptive_lcmd", METRICS[0], "endpoint_mean") < value("static_lcmd", METRICS[0], "endpoint_mean"),
        "paired_NRMSE_wins_at_least_4": int((paired.adaptive_lcmd < paired.static_lcmd).sum()) >= 4,
        "V1_RMSE_guard": value("adaptive_lcmd", "V1_RMSE", "endpoint_mean") <= 1.02*value("static_lcmd", "V1_RMSE", "endpoint_mean"),
        "V2_RMSE_guard": value("adaptive_lcmd", "V2_RMSE", "endpoint_mean") <= 1.02*value("static_lcmd", "V2_RMSE", "endpoint_mean"),
        "V1_R2_nonlower": value("adaptive_lcmd", "V1_R2", "endpoint_mean") >= value("static_lcmd", "V1_R2", "endpoint_mean"),
        "V2_R2_nonlower": value("adaptive_lcmd", "V2_R2", "endpoint_mean") >= value("static_lcmd", "V2_R2", "endpoint_mean")}
    curve_conditions = {"NRMSE_AULC_improves": value("adaptive_lcmd", METRICS[0], "aulc_mean") < value("static_lcmd", METRICS[0], "aulc_mean"),
                        **{f"{m}_AULC_nonlower": value("adaptive_lcmd", m, "aulc_mean") >= value("static_lcmd", m, "aulc_mean") for m in ("V1_R2", "V2_R2")}}
    decision = {"evidence": "iterative_development", "endpoint_conditions": conditions,
                "curve_conditions": curve_conditions, "feedback_signal": all(conditions.values()),
                "curve_efficiency_signal": all(conditions.values()) and all(curve_conditions.values()),
                "paired_NRMSE_wins": int((paired.adaptive_lcmd < paired.static_lcmd).sum()),
                "oneshot_static_endpoint_identical_by_construction": True, "new_fits": 45}
    atomic_json(STUDY / "decision.json", decision)
    return decision


def run_all():
    select_all()
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        list(pool.map(execute_seed, CONFIRMATION_SEEDS))
    finalize()
    atomic_json(STUDY / "execution_time.json", {"completed_at": now(),
                "invocation_elapsed_seconds": time.perf_counter()-started,
                "includes_reused_artifact_verification": True, "max_workers": 2})
    return reveal_and_report()
