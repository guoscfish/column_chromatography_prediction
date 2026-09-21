"""Continuation starts at verified LCMD-L653; no historical rounds are trained."""

import time

import numpy as np
import pandas as pd

from ..models import load_predictor_checkpoint
from ..training.predictor import atomic_json
from . import lcmd_to_ivr_study as study
from .benchmark_protocol import initialization_seed, sketch_seed
from .cache import seal_cache, verify_cache
from .gradient_features import extract_q50_gradient_sketches
from .ivr import conditional_batch_ivr
from .runner import fit_from_same_initialization
from .strategy_policy import StrategyState


def freeze(path, files, **fields):
    record = {"seal_sha256": study.sha256_file(study.STUDY / "seal.json"),
              "test_truth_access_count": 0, "files": study.hashes(files), **fields}
    study._write_json_once(path, record)
    return record


def verify_record(path):
    record = study.read_json(path)
    if (record["seal_sha256"] != study.sha256_file(study.STUDY / "seal.json")
            or record["test_truth_access_count"] != 0):
        raise RuntimeError("invalid continuation seal/test barrier")
    study._assert_protected(study.ROOT, record["files"])
    return record


def select(features, outer, labeled, unlabeled, state):
    study.assert_partition(labeled, unlabeled, outer)
    if state.active_label_count != len(labeled) or state.candidate_pool_size != len(unlabeled):
        raise RuntimeError("policy state does not match L/U")
    if study.POLICY.select(state) != "kernel_ivr":
        raise RuntimeError("continuation cannot rerun the historical prefix")
    position = {int(i): j for j, i in enumerate(outer)}
    result = conditional_batch_ivr(features, np.array([position[i] for i in labeled]),
                                   np.array([position[i] for i in unlabeled]), 32)
    selected = [unlabeled[i] for i in result["selected_candidate_positions"]]
    study.advance(labeled, unlabeled, selected, outer)
    return selected, result


def fit(context, lineage, r, labeled, truth, directory):
    if r < study.switch_round():
        raise RuntimeError("fits before L653 are forbidden")
    if r == study.switch_round() and lineage["checkpoint_reused"]:
        model = study.ROOT / lineage["source_checkpoint_path"]
        return model.parent, study.read_json(model.parent / "fit_audit.json")
    model_dir = directory / "model"
    artifacts = [model_dir / n for n in ("best.pt", "predictions.csv.gz", "fit_audit.json")]
    if model_dir.exists() and any(p.exists() for p in artifacts) and not all(p.exists() for p in artifacts):
        archive = context.runtime.parent / "incomplete_attempts" / f"round_{r:02d}_{time.time_ns()}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        model_dir.rename(archive)
    contract = {"study": study.STUDY.name, "method": study.METHOD, "round": r, "member": 0,
                "seal_sha256": study.sha256_file(study.STUDY / "seal.json"),
                "context_hash": study.stable_hash(context.context_contract),
                "train_sample_ids": context.ids(labeled), "validation_sample_ids": context.ids(context.roles["validation"]),
                "prediction_role": "test_zero_label_graphs"}
    audit = fit_from_same_initialization(
        atom_base=context.atom, angle=context.angle, normalization=context.normalization,
        preprocessing=context.preprocessing, train_indices=labeled, train_truth=truth,
        validation_indices=context.roles["validation"], validation_truth=context.validation_truth,
        prediction_indices=context.roles["test"], prediction_sample_ids=context.ids(context.roles["test"]),
        initialization_seed=initialization_seed(context.seed, 0), training_config=context.training_config(0),
        contract=contract, runtime=model_dir)
    return model_dir, audit


def features(context, lineage, r, model_dir, directory):
    if r == study.switch_round() and lineage["gradient_reused"]:
        return study.source_features(lineage)
    path, audit_path = directory / "gradient_features.npz", directory / "gradient_audit.json"
    config = {"seal_sha256": study.sha256_file(study.STUDY / "seal.json"),
              "checkpoint_sha256": study.sha256_file(model_dir / "best.pt"),
              "ordered_indices": context.outer.tolist(), "dimension": 512,
              "sketch_seed": sketch_seed(context.seed), "scales": context.preprocessing["target_scales"]}
    if not verify_cache(path, config):
        result = extract_q50_gradient_sketches(
            load_predictor_checkpoint(model_dir / "best.pt"), context.atom, context.angle, context.outer,
            tuple(context.preprocessing["target_scales"][k] for k in ("V1", "V2")),
            dimension=512, sketch_seed=sketch_seed(context.seed))
        np.savez_compressed(path, features=result.features, canonical_indices=context.outer)
        atomic_json(audit_path, result.audit)
        seal_cache(path, config)
    with np.load(path) as bank:
        if not np.array_equal(bank["canonical_indices"], context.outer):
            raise RuntimeError("gradient universe/order mismatch")
        values = bank["features"].copy()
    return values, study.read_json(audit_path), [path, path.with_suffix(".npz.contract.json"), audit_path]


def execute_seed(seed):
    if seed not in study.CONFIRMATION_SEEDS:
        raise ValueError("seed outside the frozen five-seed cohort")
    study.validate_seal()
    runtime = study.STUDY / f"runtime/seed_{seed}"
    with study.exclusive_lock(runtime / "worker.lock"):
        if (runtime / "trajectory_freeze.json").exists():
            return verify_trajectory(seed)
        if (study.STUDY / "global_pre_test_freeze.json").exists():
            raise RuntimeError("execution closed after global freeze")
        lineage = study.read_json(study.STUDY / f"lineage/seed_{seed}.json")
        context = study.make_context(seed, runtime / "context")
        store, truth, acquired = study.restore_truth(context, lineage)
        labeled, unlabeled = lineage["labeled_indices"].copy(), lineage["unlabeled_indices"].copy()
        study._write_json_once(runtime / "transition_audit.json", lineage)
        fit_rows, acquisition_rows, records = [], [], []
        recent_validation = []
        expected_initialization = None
        start, final = study.switch_round(), len(study.ACTIVE_LABEL_BUDGETS) - 1
        for r in range(start, final + 1):
            budget = study.ACTIVE_LABEL_BUDGETS[r]
            study.assert_partition(labeled, unlabeled, context.outer.tolist())
            if len(labeled) != budget:
                raise RuntimeError("continuation budget mismatch")
            directory = runtime / f"round_{r:02d}"
            directory.mkdir(exist_ok=True)
            inputs = {"seed": seed, "round": r, "active_labels": budget,
                      "ordered_labeled_indices": labeled, "ordered_unlabeled_indices": unlabeled,
                      "ordered_L_ids_hash": study.stable_hash(context.ids(labeled)),
                      "ordered_U_ids_hash": study.stable_hash(context.ids(unlabeled)),
                      "seal_sha256": study.sha256_file(study.STUDY / "seal.json")}
            study._write_json_once(directory / "input.json", inputs)
            if (directory / "freeze.json").exists():
                record = verify_record(directory / "freeze.json")
                if record["input"] != inputs:
                    raise RuntimeError("resumed input state changed")
                model_dir = study.ROOT / record["model_directory"]
                fit_audit = study.read_json(model_dir / "fit_audit.json")
            else:
                model_dir, fit_audit = fit(context, lineage, r, labeled, truth, directory)
                protected = [directory / "input.json"] + [model_dir / n for n in ("best.pt", "predictions.csv.gz", "fit_audit.json")]
                if r < final:
                    bank, gradient_audit, paths = features(context, lineage, r, model_dir, directory)
                    selection_path = directory / "selection.json"
                    if verify_cache(selection_path, inputs):
                        selection = verify_record(selection_path)
                        if selection["input"] != inputs:
                            raise RuntimeError("resumed selection input mismatch")
                    else:
                        state = StrategyState(budget, len(unlabeled), tuple(recent_validation + [fit_audit["best_validation_combined_normalized_rmse"]]))
                        selected, result = select(bank, context.outer.tolist(), labeled, unlabeled, state)
                        selection = freeze(selection_path, paths, input=inputs, selected_indices=selected,
                                           selected_ids=context.ids(selected), trace=result["trace"], ivr_audit=result["audit"],
                                           gradient_audit=gradient_audit, gradient_reused=r == start and lineage["gradient_reused"])
                        seal_cache(selection_path, inputs)
                    protected += paths + [selection_path, selection_path.with_suffix(".json.contract.json")]
                record = freeze(directory / "freeze.json", protected, input=inputs,
                                model_directory=str(model_dir.relative_to(study.ROOT)),
                                fit_reused=r == start and lineage["checkpoint_reused"])
            if fit_audit["train_rows"] != budget or fit_audit["test_labels_used_for_fit_or_checkpoint_selection"] != 0:
                raise RuntimeError("fit label budget/access violation")
            if expected_initialization is None:
                expected_initialization = fit_audit["initialization_hash"]
            if fit_audit["initialization_hash"] != expected_initialization:
                raise RuntimeError("scratch initialization changed between rounds")
            recent_validation.append(fit_audit["best_validation_combined_normalized_rmse"])
            fit_rows.append({"outer_seed": seed, "method": study.METHOD, "round": r, "reused": record["fit_reused"], **fit_audit})
            records.append(directory / "freeze.json")
            if r < final:
                selection = verify_record(directory / "selection.json")
                selected = selection["selected_indices"]
                if context.ids(selected) != selection["selected_ids"]:
                    raise RuntimeError("selection ID/index mismatch")
                labeled, unlabeled = study.advance(labeled, unlabeled, selected, context.outer.tolist())
                acquired += selection["selected_ids"]
                store.freeze_acquisitions(acquired)
                truth = np.vstack([truth, store.reveal(selection["selected_ids"], "after_acquisition_fit")])
                acquisition_rows.append({"round": r, "gradient_seconds": selection["gradient_audit"]["elapsed_seconds"],
                                         "gradient_reused": selection["gradient_reused"],
                                         "ivr_seconds": selection["ivr_audit"]["elapsed_seconds"]})
            pd.DataFrame(store.audit).to_csv(runtime / "label_access_audit.csv", index=False)
            atomic_json(runtime / "progress.json", {"seed": seed, "round_frozen": r, "active_labels": budget,
                                                    "test_truth_access_count": 0, "at": study.now()})
            print(f"seed={seed} frozen_round={r} active_labels={budget}", flush=True)
        pd.DataFrame(fit_rows).to_csv(runtime / "fit_audit.csv", index=False)
        pd.DataFrame(acquisition_rows).to_csv(runtime / "acquisition_runtime.csv", index=False)
        records += [runtime / n for n in ("transition_audit.json", "fit_audit.csv", "acquisition_runtime.csv", "label_access_audit.csv")]
        freeze(runtime / "trajectory_freeze.json", records, seed=seed, final_active_labels=len(labeled),
               round_points=final - start + 1, status="CONTINUATION_FROZEN_BEFORE_TEST_REVEAL")
        return verify_trajectory(seed)


def verify_trajectory(seed):
    runtime = study.STUDY / f"runtime/seed_{seed}"
    trajectory = verify_record(runtime / "trajectory_freeze.json")
    lineage = study.read_json(study.STUDY / f"lineage/seed_{seed}.json")
    labeled, unlabeled = lineage["labeled_indices"], lineage["unlabeled_indices"]
    partition = study._partition(seed)
    ids = lambda indices: partition.iloc[indices].sample_id.astype(str).tolist()
    final = len(study.ACTIVE_LABEL_BUDGETS) - 1
    if trajectory["seed"] != seed or trajectory["round_points"] != final - study.switch_round() + 1:
        raise RuntimeError("incomplete continuation")
    for r in range(study.switch_round(), final + 1):
        directory = runtime / f"round_{r:02d}"
        record = verify_record(directory / "freeze.json")
        inputs = record["input"]
        if (inputs["ordered_labeled_indices"] != labeled or inputs["ordered_unlabeled_indices"] != unlabeled
                or inputs["round"] != r or inputs["seed"] != seed or inputs["active_labels"] != study.ACTIVE_LABEL_BUDGETS[r]
                or inputs["ordered_L_ids_hash"] != study.stable_hash(ids(labeled))
                or inputs["ordered_U_ids_hash"] != study.stable_hash(ids(unlabeled))):
            raise RuntimeError("frozen continuation state mismatch")
        if r < final:
            selection = verify_record(directory / "selection.json")
            verify_cache(directory / "selection.json", inputs)
            if selection["input"] != inputs or selection["selected_ids"] != ids(selection["selected_indices"]):
                raise RuntimeError("frozen selection state mismatch")
            labeled, unlabeled = study.advance(labeled, unlabeled, selection["selected_indices"], lineage["outer_indices"])
    if len(labeled) != 1005 or trajectory["final_active_labels"] != 1005:
        raise RuntimeError("incomplete final budget")
    return trajectory


def global_freeze():
    study.validate_seal()
    paths = []
    for seed in study.CONFIRMATION_SEEDS:
        verify_trajectory(seed)
        paths.append(study.STUDY / f"runtime/seed_{seed}/trajectory_freeze.json")
    return freeze(study.STUDY / "global_pre_test_freeze.json", paths,
                  status="ALL_FIVE_CONTINUATIONS_FROZEN_BEFORE_TEST_REVEAL")


def run_all():
    """Sequential seed execution, intentionally without automatic test reporting."""
    study.validate_seal()
    with study.exclusive_lock(study.STUDY / "runtime/scheduler.lock"):
        for seed in study.CONFIRMATION_SEEDS:
            execute_seed(seed)
        return global_freeze()


def selection_smoke():
    study.validate_seal()
    checks = []
    for seed in study.CONFIRMATION_SEEDS:
        lineage = study.read_json(study.STUDY / f"lineage/seed_{seed}.json")
        if not lineage["gradient_reused"]:
            checks.append({"seed": seed, "status": "NO_REUSABLE_GRADIENT", "new_fits": 0})
            continue
        bank, _, _ = study.source_features(lineage)
        args = (bank, lineage["outer_indices"], lineage["labeled_indices"], lineage["unlabeled_indices"],
                StrategyState(lineage["switch_active_labels"], len(lineage["unlabeled_indices"])))
        selected, result = select(*args)
        repeated, repeat_result = select(*args)
        if selected != repeated or result["trace"] != repeat_result["trace"]:
            raise RuntimeError("selection-only repeat is not deterministic")
        partition = study._partition(seed)
        checks.append({"seed": seed, "status": "PASSED", "source_round": lineage["source_round"],
                       "from_labels": 653, "to_labels": 685, "selected_indices": selected,
                       "selected_ids": partition.iloc[selected].sample_id.astype(str).tolist(),
                       "trace": result["trace"], "ivr_audit": result["audit"], "new_fits": 0,
                       "new_label_reveals": 0, "test_truth_access_count": 0})
    result = {"checks": checks, "new_fits": 0, "new_test_evaluations": 0,
              "seal_sha256": study.sha256_file(study.STUDY / "seal.json")}
    atomic_json(study.STUDY / "engineering_smoke.json", result)
    return result
