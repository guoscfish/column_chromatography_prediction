#!/usr/bin/env python3
"""Run the preregistered A2 shared-late-backbone PCGrad mechanism study."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.studies import run_column_conditioned_multitask as base
from src.qgeognn_al.transfer.pcgrad import (
    deterministic_pcgrad_seed,
    gradient_geometry,
    pcgrad_backward,
)

STUDY = ROOT / "studies/transfer/column_conditioned_pcgrad"
REFERENCE = ROOT / "studies/transfer/column_conditioned_multitask"
METHOD = "A2_PCGRAD"
REFERENCE_METHOD = "A2_ADAM_FROZEN"
SEEDS = base.SEEDS
TASKS = base.TASKS
TRAINING = dict(base.TRAINING)
REFERENCE_HASHES = {
    "protocol.json": "fb7595d661ccaf7a228453536f0f34c891900e76015c93d50a043ff5b63bf567",
    "INNER_RESULTS.csv": "b6c0fddd3a499627aba47b31e5666bb7e3b1f281c11b1074437f148486c85510",
    "TAIL_METRICS.csv": "647e2993f38286d46f4c0a3faa303fbbfb6ebdf95fc5d4a2ae3f248bf2814ece",
    "INNER_IDENTITIES.csv": "0df11ce90f60e92b7185d5a1066095755b2446cfc09140675f74ab005a3fed20",
    "ROW_INNER_RESULTS.csv": "38b875ce6fea1156435ab846e5677da9a51f43bc23eee757a9779af27f825b34",
    "ROW_TAIL_METRICS.csv": "1eca4eb75c39730bdc4899e32cd0725ffb526c1ea993a60597f29c6d024cbe35",
    "ROW_INNER_IDENTITIES.csv": "a231e4659beb4f731103bdfe28adba8f6e25ae2a1d34b14f6b9801d592dadac5",
    "OUTER_RESULTS.csv": "3c7fdd0ae650fb1e13a9409d965e973b4fb132d2ef1abcba95b9b74f75a1c8dc",
    "OUTER_TAIL_METRICS.csv": "a72efe4eb24fa0813b32ff6e5086da97c18c06d8c68f91f75445d04dabbc63ff",
    "ROW_OUTER_RESULTS.csv": "c7ff09246ee220b6efc73c523cc559dce4deae7482276aa7d85dc4de49d0e7b8",
    "ROW_OUTER_TAIL_METRICS.csv": "3378676ab2e85559b6bc8feae336fb38a06fdbfa67386cc2fef768662b6f163d",
}
REFERENCE_IMPLEMENTATION_SHA = "b7beb9dfe0382f97d2e3bec960b38c394b185c58be633d13cbe342c43924a46b"
PAIR_COLUMNS = ("cos_4g_25g", "cos_4g_40g", "cos_25g_40g")


def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value, *, immutable: bool = False) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if immutable and path.exists():
        if path.read_text() != encoded:
            raise RuntimeError(f"immutable artifact mismatch: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def reference_file(protocol: str, name: str) -> Path:
    return REFERENCE / (("ROW_" if protocol == "row" else "") + name)


def verify_reference() -> dict:
    for name, expected in REFERENCE_HASHES.items():
        if sha(REFERENCE / name) != expected:
            raise RuntimeError(f"frozen A2 reference changed: {name}")
    if sha(ROOT / "src/qgeognn_al/transfer/column_conditioned_multitask.py") != REFERENCE_IMPLEMENTATION_SHA:
        raise RuntimeError("completed A2 architecture implementation changed")
    protocol = json.loads((REFERENCE / "protocol.json").read_text())
    if protocol["source_sha256"] != base.SOURCE_SHA or protocol["split_manifest_sha256"] != base.SPLIT_SHA:
        raise RuntimeError("A2 reference source/split authority mismatch")
    if protocol["training"] != TRAINING or protocol["film"]["embedding_dim"] != 16 or protocol["film"]["layers"] != [3, 4]:
        raise RuntimeError("A2 reference training/architecture protocol mismatch")
    if protocol["training"]["loss"] != "raw_quantile" or protocol["inner_cv"]["folds"] != 5:
        raise RuntimeError("A2 reference loss/selection semantics mismatch")
    return protocol


def architecture_signature(model: torch.nn.Module) -> str:
    payload = [(name, list(tensor.shape), str(tensor.dtype)) for name, tensor in model.state_dict().items()]
    return digest(payload)


def trainable_signature(model: torch.nn.Module) -> tuple[str, tuple[str, ...]]:
    names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    return digest(names), names


def shared_signature(model: torch.nn.Module) -> tuple[str, tuple[str, ...]]:
    lookup = {id(parameter): name for name, parameter in model.named_parameters()}
    names = tuple(lookup[id(parameter)] for parameter in model.shared_gradient_parameters())
    return digest(names), names


def protocol_payload() -> dict:
    reference = verify_reference()
    torch.manual_seed(42)
    model = base.build_multitask_model(base.source_path(), base.ARMS["A2"])
    trainable_sha, trainable_names = trainable_signature(model)
    shared_sha, shared_names = shared_signature(model)
    return {
        "study": "A2_COLUMN_CONDITIONED_MULTITASK_CONTROLLED_SHARED_BACKBONE_PCGRAD",
        "reference": REFERENCE_METHOD,
        "candidate": METHOD,
        "sole_difference": "standard deterministic PCGrad replaces mean-loss gradients only on completed A2 shared_gradient_parameters authority",
        "source_sha256": base.SOURCE_SHA,
        "split_manifest_sha256": base.SPLIT_SHA,
        "reference_protocol_sha256": REFERENCE_HASHES["protocol.json"],
        "reference_artifact_sha256": REFERENCE_HASHES,
        "reference_implementation_sha256": REFERENCE_IMPLEMENTATION_SHA,
        "outer_seeds": list(SEEDS),
        "training": TRAINING,
        "architecture_signature_sha256": architecture_signature(model),
        "trainable_parameter_names_sha256": trainable_sha,
        "trainable_parameter_names": list(trainable_names),
        "pcgrad_parameter_names_sha256": shared_sha,
        "pcgrad_parameter_names": list(shared_names),
        "pcgrad": {
            "tasks": list(TASKS),
            "projection": "if dot(g_i,g_j)<0: g_i -= dot/(norm2+1e-12)*g_j",
            "other_task_gradients": "original unprojected",
            "aggregation": "arithmetic mean of projected task gradients",
            "permutation": "numpy Generator permutation from deterministic mix of outer_seed, inner_fold, epoch, optimization_step",
            "nonshared_gradient": "ordinary gradient of arithmetic mean task loss",
            "renormalization": False,
        },
        "inner": reference["inner_cv"],
        "tail_threshold": reference["tail_threshold"],
        "epoch_refit": reference["epoch_refit"],
        "compound_gate": {
            "mean_paired_fold_nrmse_gain_min": 0.02,
            "seed_wins_min": 3,
            "fold_wins_min": 13,
            "endpoint_rmse_deterioration_max": 0.02,
            "endpoint_mae_deterioration_max": 0.02,
            "tail_rmse_deterioration_max": 0.02,
            "mechanism_mean_cosine_increase_min": 0.05,
            "mechanism_negative_cosine_burden_relative_reduction_min": 0.25,
            "projection_trigger_fraction_min": 0.05,
            "required_columns": ["25g", "40g"],
        },
        "row_gate": {
            "mean_nrmse_deterioration_max": 0.02,
            "endpoint_rmse_deterioration_max": 0.02,
            "endpoint_mae_deterioration_max": 0.02,
        },
        "outer_truth_used_for_fit_or_selection": False,
        "outer_scoring": "only after both inner gates and one global compound+row prediction freeze; DEVELOPMENTAL CONFIRMATION",
        "no_extra_candidates": True,
    }


def prepare() -> dict:
    base.verify_schedule()
    STUDY.mkdir(parents=True, exist_ok=True)
    payload = protocol_payload()
    write_json(STUDY / "protocol.json", payload, immutable=True)
    split = STUDY / "split_manifest.csv"
    if not split.exists():
        split.write_bytes((base.PARENT / "split_manifest.csv").read_bytes())
    if sha(split) != base.SPLIT_SHA:
        raise RuntimeError("PCGrad split authority mismatch")
    return payload


def execution_provenance() -> str:
    paths = [Path(__file__).resolve(), ROOT / "src/qgeognn_al/transfer/pcgrad.py",
             ROOT / "src/qgeognn_al/transfer/column_conditioned_multitask.py",
             ROOT / "src/qgeognn_al/models/qgeognn_v2.py",
             ROOT / "src/qgeognn_al/transfer/adaptation.py"]
    payload = {"files": {str(path.relative_to(ROOT)): sha(path) for path in paths},
               "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
               "cpu_threads": torch.get_num_threads(), "protocol_sha256": sha(STUDY / "protocol.json")}
    write_json(STUDY / "EXECUTION_PROVENANCE.json", payload, immutable=True)
    return sha(STUDY / "EXECUTION_PROVENANCE.json")


def historical_frames(protocol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(reference_file(protocol, "INNER_RESULTS.csv"))
    results = results.loc[results.arm.eq("A2")].copy()
    results["method"] = REFERENCE_METHOD
    tails = pd.read_csv(reference_file(protocol, "TAIL_METRICS.csv"))
    tails = tails.loc[tails.arm.eq("A2")].copy()
    tails["method"] = REFERENCE_METHOD
    identities = pd.read_csv(reference_file(protocol, "INNER_IDENTITIES.csv")).drop_duplicates()
    return results, tails, identities


def assert_context_comparable(data, train, valid, protocol: str, seed: int, fold: int) -> None:
    reference_results, _, reference_identities = historical_frames(protocol)
    current = base.identity_records(data, train, valid, protocol, seed, fold)
    expected = reference_identities.loc[(reference_identities.outer_seed == seed) & (reference_identities.inner_fold == fold)]
    columns = ["protocol", "outer_seed", "inner_fold", "column", "sample_id", "canonical_smiles", "role"]
    left = current[columns].astype({"sample_id": str, "canonical_smiles": str}).sort_values(columns).reset_index(drop=True)
    right = expected[columns].astype({"sample_id": str, "canonical_smiles": str}).sort_values(columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)
    if set(left.loc[left.column.isin(TASKS[1:]) & left.role.eq("inner_train"), "canonical_smiles"]) & set(left.loc[left.column.isin(TASKS[1:]) & left.role.eq("inner_validation"), "canonical_smiles"]):
        raise RuntimeError("global target molecule leakage")
    reference = reference_results.loc[(reference_results.outer_seed == seed) & (reference_results.inner_fold == fold)]
    for column in TASKS:
        row = reference.loc[reference.column.eq(column)].iloc[0]
        actual = base.scales(data[column]["truth"][train[column]])
        np.testing.assert_allclose([actual["V1"], actual["V2"]], [row.scale_V1, row.scale_V2], rtol=0, atol=1e-12)


def verify_completed(run: Path) -> bool:
    complete = run / "complete.json"
    if not complete.exists():
        return False
    record = json.loads(complete.read_text())
    if record["protocol_sha256"] != sha(STUDY / "protocol.json") or record["execution_provenance_sha256"] != execution_provenance():
        raise RuntimeError("completed PCGrad fit provenance changed")
    for name, expected in record["files"].items():
        if sha(run / name) != expected:
            raise RuntimeError(f"completed PCGrad artifact changed: {run / name}")
    return True


def fit_model(data, train, valid, outer_seed: int, fold: int, run: Path, *, fixed_epochs: int | None = None):
    base.authorize_fit(data, train, valid)
    training_seed = int(outer_seed) if fixed_epochs is not None else int(outer_seed) + int(fold) * 1009
    torch.manual_seed(training_seed)
    model = base.build_multitask_model(base.source_path(), base.ARMS["A2"])
    protocol = json.loads((STUDY / "protocol.json").read_text())
    if architecture_signature(model) != protocol["architecture_signature_sha256"]:
        raise RuntimeError("candidate architecture differs from frozen A2")
    if trainable_signature(model)[0] != protocol["trainable_parameter_names_sha256"] or shared_signature(model)[0] != protocol["pcgrad_parameter_names_sha256"]:
        raise RuntimeError("candidate trainable/PCGrad parameter identity changed")
    optimizer = model.optimizer(pretrained_lr=TRAINING["pretrained_lr"], new_lr=TRAINING["new_lr"], weight_decay=TRAINING["weight_decay"])
    size = min(TRAINING["per_task_batch_cap"], max(len(train[column]) for column in TASKS[1:]))
    batcher = base.BalancedTaskBatcher(train, per_task_batch_size=size, seed=training_seed)
    metric_scales = {column: base.scales(data[column]["truth"][train[column]]) for column in TASKS}
    maximum = int(fixed_epochs or TRAINING["maximum_epochs"])
    history, raw_rows, projected_rows = [], [], []
    best, best_epoch, best_state = float("inf"), 0, None
    started = time.monotonic()
    for epoch in range(1, maximum + 1):
        base.set_training_mode(model)
        epoch_losses = {column: [] for column in TASKS}
        for step, indices in enumerate(batcher.epoch_batches(epoch)):
            atoms, angles, tasks = base.make_batch(data, indices)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(atoms, angles, tasks)
            losses = {column: base.target_loss_recipe(atoms.y[tasks.eq(index), 0], atoms.y[tasks.eq(index), 1], prediction[tasks.eq(index)], {}, recipe="raw_quantile") for index, column in enumerate(TASKS)}
            raw, projected, mechanism = pcgrad_backward(
                losses, model.shared_gradient_parameters(),
                permutation_seed=deterministic_pcgrad_seed(outer_seed, fold, epoch, step),
            )
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise RuntimeError("non-finite PCGrad optimizer gradient")
            if epoch == 1 or epoch % TRAINING["gradient_diagnostic_interval"] == 0:
                meta = {"protocol": data["25g"]["protocol"], "outer_seed": outer_seed, "inner_fold": fold,
                        "epoch": epoch, "step": step, "permutation_seed": deterministic_pcgrad_seed(outer_seed, fold, epoch, step)}
                raw_rows.append({**meta, **gradient_geometry(raw)})
                projected_rows.append({**meta, **gradient_geometry(projected),
                                       "directed_projection_attempts": mechanism.directed_projection_attempts,
                                       "directed_projections_triggered": mechanism.directed_projections_triggered,
                                       "projection_trigger_fraction": mechanism.projection_trigger_fraction,
                                       "projected_minus_original_update_norm": mechanism.projected_minus_original_update_norm,
                                       "projected_original_update_norm_ratio": mechanism.projected_original_update_norm_ratio})
            optimizer.step()
            for column in TASKS:
                epoch_losses[column].append(float(losses[column].detach()))
        row = {"epoch": epoch, **{f"train_loss_{column}": float(np.mean(values)) for column, values in epoch_losses.items()}}
        if valid is not None:
            for column in TASKS:
                values = base.point_metrics(data[column]["truth"][valid[column]], base.predict(model, data, column, valid[column]), metric_scales[column])
                row.update({f"{column}_{key}": value for key, value in values.items()})
            score = float(np.mean([row[f"{column}_combined_normalized_rmse"] for column in TASKS[1:]]))
            row["joint_validation_score"] = score
            if score < best:
                best, best_epoch, best_state = score, epoch, deepcopy(model.state_dict())
        history.append(row)
        if epoch == 1 or epoch % 50 == 0:
            print(json.dumps({"run": str(run.relative_to(STUDY)), "epoch": epoch, "best_epoch": best_epoch,
                              "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
        if valid is not None and epoch - best_epoch >= TRAINING["patience"]:
            break
    if valid is not None:
        if best_state is None:
            raise RuntimeError("no finite validation checkpoint")
        model.load_state_dict(best_state)
    else:
        best_epoch = maximum
    run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(run / "history.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(run / "gradient_raw.csv", index=False)
    pd.DataFrame(projected_rows).to_csv(run / "gradient_projected.csv", index=False)
    torch.save({"model_state_dict": model.state_dict(), "method": METHOD, "best_epoch": best_epoch,
                "source_sha256": base.SOURCE_SHA, "protocol_sha256": sha(STUDY / "protocol.json")}, run / "best.pt")
    fitted = {"best_epoch": best_epoch, "epochs_run": epoch, "runtime_seconds": time.monotonic() - started,
              "per_task_batch_size": size, "steps_per_epoch": batcher.steps_per_epoch,
              "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
              "total_parameters": sum(parameter.numel() for parameter in model.parameters())}
    return model, fitted, metric_scales


def load_data(protocol: str, seed: int):
    data = base.joint_data(protocol, seed)
    for value in data.values():
        value["protocol"] = protocol
    return data


def inner_context(protocol: str, seed: int) -> None:
    if protocol == "row":
        require_compound_gate()
    data = load_data(protocol, seed)
    provenance = execution_provenance()
    for fold, (train, valid) in enumerate(base.fold_schedules(data), 1):
        run = STUDY / f"runtime/inner/{protocol}/seed_{seed}/fold_{fold}/{METHOD}"
        if verify_completed(run):
            continue
        assert_context_comparable(data, train, valid, protocol, seed, fold)
        run.mkdir(parents=True, exist_ok=True)
        identities = base.identity_records(data, train, valid, protocol, seed, fold)
        identities.to_csv(run / "identities.csv", index=False)
        model, fitted, metric_scales = fit_model(data, train, valid, seed, fold, run)
        result_rows, tail_rows, prediction_rows = [], [], []
        for column in TASKS:
            prediction = base.predict(model, data, column, valid[column])
            values, tails, _ = base.metric_rows(data[column]["truth"][valid[column]], prediction, metric_scales[column], data[column]["tail"], data[column]["regimes"])
            meta = {"protocol": protocol, "outer_seed": seed, "inner_fold": fold, "method": METHOD, "column": column}
            result_rows.append({**meta, **fitted, **values,
                                "inner_train_count": len(train[column]), "inner_validation_count": len(valid[column]),
                                "inner_train_ids_sha256": digest(sorted(data[column]["frame"].iloc[train[column]].sample_id.tolist())),
                                "inner_validation_ids_sha256": digest(sorted(data[column]["frame"].iloc[valid[column]].sample_id.tolist())),
                                "scale_V1": metric_scales[column]["V1"], "scale_V2": metric_scales[column]["V2"]})
            tail_rows.extend([{**meta, **tail} for tail in tails])
            prediction_rows.extend([{**meta, "sample_id": str(data[column]["frame"].iloc[index].sample_id),
                                     **dict(zip(base.QUANTILES, map(float, values_)))} for index, values_ in zip(valid[column], prediction)])
        for filename, rows in (("results.csv", result_rows), ("tails.csv", tail_rows), ("predictions.csv", prediction_rows)):
            pd.DataFrame(rows).to_csv(run / filename, index=False)
        files = ("results.csv", "tails.csv", "predictions.csv", "identities.csv", "history.csv", "gradient_raw.csv", "gradient_projected.csv", "best.pt")
        write_json(run / "complete.json", {"protocol_sha256": sha(STUDY / "protocol.json"),
                   "execution_provenance_sha256": provenance, "files": {name: sha(run / name) for name in files}}, immutable=True)
        print(f"COMPLETED {protocol} seed={seed} fold={fold} {METHOD}", flush=True)


def mechanism_summary(raw: pd.DataFrame, projected: pd.DataFrame) -> dict:
    raw_cos = raw[list(PAIR_COLUMNS)].to_numpy(float).reshape(-1)
    projected_cos = projected[list(PAIR_COLUMNS)].to_numpy(float).reshape(-1)
    raw_burden = float(np.maximum(-raw_cos, 0).mean())
    projected_burden = float(np.maximum(-projected_cos, 0).mean())
    reduction = 1 - projected_burden / raw_burden if raw_burden > 0 else 0.0
    return {
        "observations": len(raw),
        "raw_mean_cosine": float(raw_cos.mean()),
        "projected_mean_cosine": float(projected_cos.mean()),
        "mean_cosine_increase": float(projected_cos.mean() - raw_cos.mean()),
        "raw_negative_fraction": float((raw_cos < 0).mean()),
        "projected_negative_fraction": float((projected_cos < 0).mean()),
        "raw_negative_cosine_burden": raw_burden,
        "projected_negative_cosine_burden": projected_burden,
        "negative_cosine_burden_relative_reduction": float(reduction),
        "projection_trigger_fraction": float(projected.directed_projections_triggered.sum() / projected.directed_projection_attempts.sum()),
        "mean_projected_minus_original_update_norm": float(projected.projected_minus_original_update_norm.mean()),
        "mean_projected_original_update_norm_ratio": float(projected.projected_original_update_norm_ratio.mean()),
        "raw_median_magnitude_ratio": float(np.median([max(row) / max(min(row), 1e-30) for row in raw[[f"grad_norm_{task}" for task in TASKS]].to_numpy(float)])),
    }


def compound_decision(results: pd.DataFrame, tails: pd.DataFrame, mechanism: dict) -> dict:
    rows = []
    for column in TASKS[1:]:
        current = results.loc[(results.column == column) & (results.method == METHOD)].set_index(["outer_seed", "inner_fold"])
        reference = results.loc[(results.column == column) & (results.method == REFERENCE_METHOD)].set_index(["outer_seed", "inner_fold"]).loc[current.index]
        gain = 1 - current.combined_normalized_rmse / reference.combined_normalized_rmse
        seed_current = current.combined_normalized_rmse.groupby(level=0).mean()
        seed_reference = reference.combined_normalized_rmse.groupby(level=0).mean()
        rmse_changes = {endpoint: float(current[f"{endpoint}_rmse"].mean() / reference[f"{endpoint}_rmse"].mean() - 1) for endpoint in ("V1", "V2")}
        mae_changes = {endpoint: float(current[f"{endpoint}_mae"].mean() / reference[f"{endpoint}_mae"].mean() - 1) for endpoint in ("V1", "V2")}
        tail_changes = {}
        for endpoint in ("V1", "V2"):
            selected = tails.loc[(tails.column == column) & (tails.endpoint == endpoint)]
            grouped = selected.groupby(["outer_seed", "method"])[["tail_sse", "tail_count"]].sum()
            rmse = np.sqrt(grouped.tail_sse / grouped.tail_count).unstack("method")
            tail_changes[endpoint] = float(rmse[METHOD].mean() / rmse[REFERENCE_METHOD].mean() - 1)
        conditions = {"mean_gain": float(gain.mean()) >= .02,
                      "seed_wins": int((seed_current < seed_reference).sum()) >= 3,
                      "fold_wins": int((gain > 0).sum()) >= 13,
                      "rmse_guard": max(rmse_changes.values()) <= .02,
                      "mae_guard": max(mae_changes.values()) <= .02,
                      "tail_guard": max(tail_changes.values()) <= .02}
        rows.append({"column": column, "mean_paired_fold_nrmse_gain": float(gain.mean()),
                     "seed_wins": int((seed_current < seed_reference).sum()), "fold_wins": int((gain > 0).sum()),
                     "endpoint_rmse_changes": rmse_changes, "endpoint_mae_changes": mae_changes,
                     "tail_rmse_changes": tail_changes, "conditions": conditions, "passed": all(conditions.values())})
    mechanism_conditions = {
        "mean_cosine_increase": mechanism["mean_cosine_increase"] >= .05,
        "negative_burden_reduction": mechanism["negative_cosine_burden_relative_reduction"] >= .25,
        "nontrivial_trigger": mechanism["projection_trigger_fraction"] >= .05,
    }
    passed = all(row["passed"] for row in rows) and all(mechanism_conditions.values())
    return {"status": "COMPOUND_INNER_PCGRAD_GATE_PASSED" if passed else "COMPOUND_INNER_PCGRAD_GATE_FAILED",
            "passed": passed, "columns": rows, "mechanism": mechanism,
            "mechanism_conditions": mechanism_conditions, "outer_truth_read": False,
            "row_authorized": passed, "outer_authorized": False, "additional_candidates_authorized": False}


def row_decision(results: pd.DataFrame) -> dict:
    rows = []
    for column in TASKS[1:]:
        current = results.loc[(results.column == column) & (results.method == METHOD)]
        reference = results.loc[(results.column == column) & (results.method == REFERENCE_METHOD)]
        changes = {"combined_normalized_rmse": float(current.combined_normalized_rmse.mean() / reference.combined_normalized_rmse.mean() - 1)}
        changes.update({f"{endpoint}_rmse": float(current[f"{endpoint}_rmse"].mean() / reference[f"{endpoint}_rmse"].mean() - 1) for endpoint in ("V1", "V2")})
        changes.update({f"{endpoint}_mae": float(current[f"{endpoint}_mae"].mean() / reference[f"{endpoint}_mae"].mean() - 1) for endpoint in ("V1", "V2")})
        passed = max(changes.values()) <= .02
        rows.append({"column": column, "changes": changes, "passed": passed})
    passed = all(row["passed"] for row in rows)
    return {"status": "ROW_INNER_PCGRAD_GATE_PASSED" if passed else "ROW_INNER_PCGRAD_GATE_FAILED",
            "passed": passed, "columns": rows, "outer_authorized": passed,
            "outer_truth_read": False, "additional_candidates_authorized": False}


def aggregate(protocol: str) -> dict:
    candidate_results, candidate_tails, raw_frames, projected_frames, identities, completions = [], [], [], [], [], {}
    for seed in SEEDS:
        for fold in range(1, 6):
            run = STUDY / f"runtime/inner/{protocol}/seed_{seed}/fold_{fold}/{METHOD}"
            if not verify_completed(run):
                raise RuntimeError(f"incomplete PCGrad inner context: {run}")
            completions[str(run.relative_to(STUDY))] = sha(run / "complete.json")
            candidate_results.append(pd.read_csv(run / "results.csv"))
            candidate_tails.append(pd.read_csv(run / "tails.csv"))
            raw_frames.append(pd.read_csv(run / "gradient_raw.csv"))
            projected_frames.append(pd.read_csv(run / "gradient_projected.csv"))
            identities.append(pd.read_csv(run / "identities.csv"))
    current_results = pd.concat(candidate_results, ignore_index=True)
    current_tails = pd.concat(candidate_tails, ignore_index=True)
    reference_results, reference_tails, _ = historical_frames(protocol)
    keep = [column for column in current_results.columns if column in reference_results.columns or column == "method"]
    reference_results = reference_results.rename(columns={"arm": "historical_arm"})
    reference_tails = reference_tails.rename(columns={"arm": "historical_arm"})
    combined_results = pd.concat([current_results, reference_results[current_results.columns]], ignore_index=True)
    combined_tails = pd.concat([current_tails, reference_tails[current_tails.columns]], ignore_index=True)
    raw = pd.concat(raw_frames, ignore_index=True)
    projected = pd.concat(projected_frames, ignore_index=True)
    mechanism = mechanism_summary(raw, projected)
    prefix = "" if protocol == "compound" else "ROW_"
    combined_results.to_csv(STUDY / f"{prefix}INNER_RESULTS.csv", index=False)
    combined_tails.to_csv(STUDY / f"{prefix}TAIL_METRICS.csv", index=False)
    raw.to_csv(STUDY / f"{prefix}GRADIENT_RAW.csv", index=False)
    projected.to_csv(STUDY / f"{prefix}GRADIENT_PROJECTED.csv", index=False)
    pd.DataFrame([mechanism]).to_csv(STUDY / f"{prefix}GRADIENT_MECHANISM_SUMMARY.csv", index=False)
    pd.concat(identities, ignore_index=True).to_csv(STUDY / f"{prefix}INNER_IDENTITIES.csv", index=False)
    current_results[["protocol", "outer_seed", "inner_fold", "method", "column", "best_epoch", "epochs_run",
                     "inner_train_ids_sha256", "inner_validation_ids_sha256", "scale_V1", "scale_V2"]].to_csv(STUDY / f"{prefix}run_manifest.csv", index=False)
    target = combined_results.loc[combined_results.column.isin(TASKS[1:])]
    left = target.loc[target.method == REFERENCE_METHOD].set_index(["outer_seed", "inner_fold", "column"])
    right = target.loc[target.method == METHOD].set_index(["outer_seed", "inner_fold", "column"]).loc[left.index]
    paired = right[["combined_normalized_rmse", "V1_rmse", "V1_mae", "V2_rmse", "V2_mae"]].add_suffix("_candidate").join(
        left[["combined_normalized_rmse", "V1_rmse", "V1_mae", "V2_rmse", "V2_mae"]].add_suffix("_reference"))
    for metric in ("combined_normalized_rmse", "V1_rmse", "V1_mae", "V2_rmse", "V2_mae"):
        paired[f"{metric}_relative_gain"] = 1 - paired[f"{metric}_candidate"] / paired[f"{metric}_reference"]
    paired.reset_index().to_csv(STUDY / f"{prefix}INNER_PAIRED_COMPARISONS.csv", index=False)
    decision = compound_decision(combined_results, combined_tails, mechanism) if protocol == "compound" else row_decision(combined_results)
    decision.update({"protocol_sha256": sha(STUDY / "protocol.json"),
                     "results_sha256": sha(STUDY / f"{prefix}INNER_RESULTS.csv"),
                     "tail_metrics_sha256": sha(STUDY / f"{prefix}TAIL_METRICS.csv"),
                     "completion_hashes": completions})
    write_json(STUDY / f"{prefix}INNER_DECISION.json", decision, immutable=True)
    return decision


def require_compound_gate() -> dict:
    path = STUDY / "INNER_DECISION.json"
    if not path.exists():
        raise RuntimeError("COMPOUND gate absent")
    decision = json.loads(path.read_text())
    if not decision["passed"] or decision["protocol_sha256"] != sha(STUDY / "protocol.json"):
        raise RuntimeError("COMPOUND PCGrad gate failed or changed")
    return decision


def require_all_inner_gates() -> None:
    require_compound_gate()
    path = STUDY / "ROW_INNER_DECISION.json"
    if not path.exists() or not json.loads(path.read_text())["passed"]:
        raise RuntimeError("ROW PCGrad gate failed or absent")


def final_context(protocol: str, seed: int) -> None:
    require_all_inner_gates()
    data = load_data(protocol, seed)
    prefix = "" if protocol == "compound" else "ROW_"
    selections = pd.read_csv(STUDY / f"{prefix}INNER_RESULTS.csv")
    chosen = selections.loc[(selections.outer_seed == seed) & (selections.method == METHOD) & (selections.column == "25g"), "best_epoch"]
    if len(chosen) != 5:
        raise RuntimeError("missing PCGrad fixed-epoch selections")
    epochs = int(np.floor(chosen.median() + .5))
    run = STUDY / f"runtime/final/{protocol}/seed_{seed}/{METHOD}"
    if verify_completed(run):
        return
    train = {column: data[column]["positions"]["gradient_train"] for column in TASKS}
    model, fitted, _ = fit_model(data, train, None, seed, 0, run, fixed_epochs=epochs)
    rows = []
    for column in TASKS[1:]:
        for role in ("validation", "test"):
            positions = data[column]["positions"][role]
            prediction = base.predict(model, data, column, positions)
            rows.extend([{"protocol": protocol, "column": column, "role": role,
                          "sample_id": str(data[column]["frame"].iloc[index].sample_id),
                          **dict(zip(base.QUANTILES, map(float, values)))} for index, values in zip(positions, prediction)])
    run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(run / "predictions_blind.csv", index=False)
    write_json(run / "fit.json", {**fitted, "outer_truth_read": False,
               "selection_sha256": sha(STUDY / f"{prefix}INNER_RESULTS.csv")})
    files = ("predictions_blind.csv", "best.pt", "fit.json", "history.csv", "gradient_raw.csv", "gradient_projected.csv")
    write_json(run / "complete.json", {"protocol_sha256": sha(STUDY / "protocol.json"),
               "execution_provenance_sha256": execution_provenance(), "files": {name: sha(run / name) for name in files}}, immutable=True)


def freeze() -> Path:
    require_all_inner_gates()
    files = {}
    for protocol in ("compound", "row"):
        for seed in SEEDS:
            run = STUDY / f"runtime/final/{protocol}/seed_{seed}/{METHOD}"
            if not verify_completed(run):
                raise RuntimeError("global freeze requires every COMPOUND and ROW blind prediction")
            for name in ("complete.json", "predictions_blind.csv"):
                relative = str((run / name).relative_to(STUDY))
                files[relative] = sha(run / name)
    path = STUDY / "PREDICTION_FREEZE_MANIFEST.json"
    write_json(path, {"status": "ALL_COMPOUND_AND_ROW_PREDICTIONS_FROZEN_BEFORE_OUTER_TRUTH",
               "protocol_sha256": sha(STUDY / "protocol.json"), "files": files}, immutable=True)
    return path


def verify_freeze() -> Path:
    path = STUDY / "PREDICTION_FREEZE_MANIFEST.json"
    if not path.exists():
        raise RuntimeError("outer truth unavailable before global prediction freeze")
    record = json.loads(path.read_text())
    if record["status"] != "ALL_COMPOUND_AND_ROW_PREDICTIONS_FROZEN_BEFORE_OUTER_TRUTH" or len(record["files"]) != 20:
        raise RuntimeError("incomplete global PCGrad prediction freeze")
    for relative, expected in record["files"].items():
        if sha(STUDY / relative) != expected:
            raise RuntimeError("frozen PCGrad prediction changed")
    return path


def score() -> None:
    freeze_path = verify_freeze()
    all_paired = []
    for protocol in ("compound", "row"):
        candidate_rows, candidate_tails = [], []
        schedule = base.verify_schedule()
        for seed in SEEDS:
            data = load_data(protocol, seed)
            prediction_file = STUDY / f"runtime/final/{protocol}/seed_{seed}/{METHOD}/predictions_blind.csv"
            prediction_frame = pd.read_csv(prediction_file)
            for column in TASKS[1:]:
                context = schedule.loc[(schedule.column == column) & (schedule.protocol == protocol) & (schedule.outer_seed == seed)]
                ids = context.loc[context.role.eq("test"), "sample_id"].astype(str).tolist()
                train_ids = context.loc[context.role.eq("gradient_train"), "sample_id"].astype(str).tolist()
                canonical = base.PARENT / f"filtered_canonical_{column}.csv"
                truth = base._read_authorized_truth(canonical, ids)
                train_truth = base._read_authorized_truth(canonical, train_ids)
                selected = prediction_frame.loc[(prediction_frame.column == column) & prediction_frame.role.eq("test")].copy()
                selected.sample_id = selected.sample_id.astype(str)
                if set(selected.sample_id) != set(ids) or not selected.sample_id.is_unique:
                    raise RuntimeError("PCGrad outer prediction identities changed")
                prediction = selected.set_index("sample_id").loc[ids, base.QUANTILES].to_numpy(float)
                values, tails, _ = base.metric_rows(truth, prediction, base.scales(train_truth), np.quantile(train_truth, .8, axis=0), np.quantile(train_truth.mean(axis=1), [1/3, 2/3]))
                meta = {"protocol": protocol, "outer_seed": seed, "column": column, "method": METHOD,
                        "classification": "DEVELOPMENTAL CONFIRMATION"}
                candidate_rows.append({**meta, **values})
                candidate_tails.extend([{**meta, **tail} for tail in tails])
        historical_results = pd.read_csv(reference_file(protocol, "OUTER_RESULTS.csv"))
        historical_tails = pd.read_csv(reference_file(protocol, "OUTER_TAIL_METRICS.csv"))
        historical_results.loc[historical_results.method.eq("A2"), "method"] = REFERENCE_METHOD
        historical_tails.loc[historical_tails.method.eq("A2"), "method"] = REFERENCE_METHOD
        results = pd.concat([pd.DataFrame(candidate_rows), historical_results], ignore_index=True)
        tails = pd.concat([pd.DataFrame(candidate_tails), historical_tails], ignore_index=True)
        prefix = "" if protocol == "compound" else "ROW_"
        results.to_csv(STUDY / f"{prefix}OUTER_RESULTS.csv", index=False)
        tails.to_csv(STUDY / f"{prefix}OUTER_TAIL_METRICS.csv", index=False)
        for column in TASKS[1:]:
            current = results.loc[(results.column == column) & (results.method == METHOD)].set_index("outer_seed")
            for reference_method in results.loc[results.column == column, "method"].unique():
                if reference_method == METHOD:
                    continue
                reference = results.loc[(results.column == column) & (results.method == reference_method)].set_index("outer_seed").loc[current.index]
                for seed in current.index:
                    row = {"protocol": protocol, "outer_seed": seed, "column": column,
                           "candidate": METHOD, "reference": reference_method}
                    for metric in ("combined_normalized_rmse", "V1_rmse", "V1_mae", "V2_rmse", "V2_mae", "Center_rmse", "Width_rmse"):
                        row[f"{metric}_relative_gain"] = 1 - current.loc[seed, metric] / reference.loc[seed, metric]
                    all_paired.append(row)
        if protocol == "compound":
            pd.DataFrame(all_paired).to_csv(STUDY / "PAIRED_COMPARISONS.csv", index=False)
    pd.DataFrame(all_paired).to_csv(STUDY / "PAIRED_COMPARISONS.csv", index=False)
    write_json(STUDY / "SCORE_MANIFEST.json", {"prediction_freeze_sha256": sha(freeze_path),
               "classification": "DEVELOPMENTAL CONFIRMATION",
               "files": {name: sha(STUDY / name) for name in ("OUTER_RESULTS.csv", "OUTER_TAIL_METRICS.csv", "ROW_OUTER_RESULTS.csv", "ROW_OUTER_TAIL_METRICS.csv", "PAIRED_COMPARISONS.csv")}}, immutable=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "inner", "aggregate", "final", "freeze", "score", "execute"), required=True)
    parser.add_argument("--protocol", choices=("compound", "row"), default="compound")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(TRAINING["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    prepare()
    execution_provenance()
    if args.action == "prepare":
        print(json.dumps(protocol_payload(), indent=2))
    elif args.action == "inner":
        if args.seed is None:
            parser.error("inner requires --seed")
        inner_context(args.protocol, args.seed)
    elif args.action == "aggregate":
        print(json.dumps(aggregate(args.protocol), indent=2))
    elif args.action == "final":
        if args.seed is None:
            parser.error("final requires --seed")
        final_context(args.protocol, args.seed)
    elif args.action == "freeze":
        freeze()
    elif args.action == "score":
        score()
    elif args.action == "execute":
        for seed in SEEDS:
            inner_context("compound", seed)
        decision = aggregate("compound")
        if not decision["passed"]:
            print(decision["status"], flush=True)
            return
        for seed in SEEDS:
            inner_context("row", seed)
        row = aggregate("row")
        if not row["passed"]:
            print(row["status"], flush=True)
            return
        for protocol in ("compound", "row"):
            for seed in SEEDS:
                final_context(protocol, seed)
        freeze()
        print("ALL BLIND PREDICTIONS FROZEN; run separate score action", flush=True)


if __name__ == "__main__":
    main()
