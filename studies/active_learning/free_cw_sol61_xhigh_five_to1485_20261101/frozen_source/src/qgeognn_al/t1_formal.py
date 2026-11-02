"""Pure planning, resume, metric, and analysis contracts for formal T1."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from .artifacts import sha256_file
from .engine import canonical_json_hash


FROZEN_OUTER_SEEDS = [769539383, 1425370602, 536279090, 2767143051, 1362771960]
FROZEN_BUDGETS = [30, 50, 70, 100]
FROZEN_METHODS = [
    "zero_shot", "affine", "condition_ridge_residual", "target_head_only",
    "last1_head", "current_last2_head",
]
FROZEN_NEURAL_MODES = {
    "target_head_only": "head_only",
    "last1_head": "last1_head",
    "current_last2_head": "last2_head",
}
FROZEN_SOURCE_MEMBERS = [42, 525, 1101]
EXPECTED_TRAINABLE = {"head_only": 774, "last1_head": 93454, "last2_head": 186134}
EXPECTED_TOTAL_PARAMETERS = 775476
REFERENCE_METHOD = "current_last2_head"


@dataclass(frozen=True)
class FormalFitSpec:
    outer_seed: int
    budget: int
    method: str
    mode: str
    source_member: int

    @property
    def run_key(self) -> str:
        return (
            f"seed_{self.outer_seed}/budget_{self.budget}/"
            f"{self.method}/member_{self.source_member}"
        )


def validate_frozen_formal_config(config: dict[str, Any]) -> None:
    expected = {
        "protocol": "row",
        "master_seed": 20260902,
        "outer_seed_count": 5,
        "outer_seeds": FROZEN_OUTER_SEEDS,
        "target_label_budgets": FROZEN_BUDGETS,
        "fixed_validation": 8,
        "source_members": FROZEN_SOURCE_MEMBERS,
        "primary_methods": FROZEN_METHODS,
        "neural_modes": FROZEN_NEURAL_MODES,
        "ridge_alpha_grid": [0.01, 0.1, 1.0, 10.0, 100.0],
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    training_expected = {
        "learning_rate": 1e-4, "weight_decay": 1e-5, "epochs": 500,
        "patience": 100, "batch_size": 2048,
        "quantile_parameterization": "monotonic_softplus",
        "checkpoint_selection": "validation_normalized_mse",
    }
    training = config.get("training", {})
    for key, value in training_expected.items():
        if training.get(key) != value:
            mismatches[f"training.{key}"] = {"expected": value, "actual": training.get(key)}
    optional = config.get("optional_diagnostics", {}).get("full_finetune", {})
    if optional.get("enabled") is not False:
        mismatches["optional_diagnostics.full_finetune.enabled"] = {
            "expected": False, "actual": optional.get("enabled")
        }
    if mismatches:
        raise RuntimeError(f"frozen T1 formal config changed: {mismatches}")


def build_formal_fit_plan(config: dict[str, Any]) -> list[FormalFitSpec]:
    validate_frozen_formal_config(config)
    plan = [
        FormalFitSpec(int(seed), int(budget), method, FROZEN_NEURAL_MODES[method], int(member))
        for seed in config["outer_seeds"]
        for budget in config["target_label_budgets"]
        for method in FROZEN_NEURAL_MODES
        for member in config["source_members"]
    ]
    keys = [item.run_key for item in plan]
    if len(keys) != len(set(keys)):
        raise RuntimeError("formal fit plan contains duplicate run keys")
    expected = (
        len(config["outer_seeds"]) * len(config["target_label_budgets"])
        * len(FROZEN_NEURAL_MODES) * len(config["source_members"])
    )
    if len(plan) != expected or expected != 180:
        raise RuntimeError(f"unexpected formal neural fit count: {len(plan)}")
    return plan


def formal_config_hash(config: dict[str, Any]) -> str:
    """Hash the full authorized run contract, including the authorization state."""
    return canonical_json_hash(config)


def expected_fit_contract(
    spec: FormalFitSpec,
    config: dict[str, Any],
    gradient_train_ids: Sequence[str],
    validation_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_key": spec.run_key,
        **asdict(spec),
        "formal_config_hash": formal_config_hash(config),
        "partition_sha256": config["partition_sha256"],
        "schedule_sha256": config["schedule_sha256"],
        "source_checkpoint_sha256": config["source_checkpoint_hashes"][str(spec.source_member)],
        "gradient_train_ids_hash": canonical_json_hash(sorted(map(str, gradient_train_ids))),
        "validation_ids_hash": canonical_json_hash(sorted(map(str, validation_ids))),
        "labeled_ids_hash": canonical_json_hash(sorted(map(str, [*gradient_train_ids, *validation_ids]))),
        "expected_trainable_parameters": EXPECTED_TRAINABLE[spec.mode],
        "expected_total_parameters": EXPECTED_TOTAL_PARAMETERS,
    }


def inspect_fit_runtime(fit_dir: Path, expected_contract: dict[str, Any]) -> dict[str, Any]:
    fit_dir = Path(fit_dir)
    required = {
        "checkpoint": fit_dir / "best.pt",
        "history": fit_dir / "history.csv",
        "fit_result": fit_dir / "fit_result.json",
        "formal_contract": fit_dir / "formal_contract.json",
    }
    present = {name: path.is_file() for name, path in required.items()}
    if not any(present.values()):
        return {"status": "missing", "reason": "no_runtime_files", "present": present}
    if not all(present.values()):
        return {"status": "partial", "reason": "required_runtime_file_missing", "present": present}
    try:
        contract = json.loads(required["formal_contract"].read_text())
        result = json.loads(required["fit_result"].read_text())
    except (OSError, json.JSONDecodeError) as error:
        return {"status": "stale", "reason": f"unreadable_metadata:{type(error).__name__}", "present": present}
    mismatch = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected_contract.items()
        if contract.get(key) != value
    }
    checkpoint_hash = sha256_file(required["checkpoint"])
    if result.get("checkpoint_sha256") != checkpoint_hash:
        mismatch["fit_result.checkpoint_sha256"] = {
            "expected": checkpoint_hash, "actual": result.get("checkpoint_sha256")
        }
    if contract.get("checkpoint_sha256") != checkpoint_hash:
        mismatch["formal_contract.checkpoint_sha256"] = {
            "expected": checkpoint_hash, "actual": contract.get("checkpoint_sha256")
        }
    if result.get("train_config_hash") != contract.get("adaptation_train_config_hash"):
        mismatch["fit_result.train_config_hash"] = {
            "expected": contract.get("adaptation_train_config_hash"),
            "actual": result.get("train_config_hash"),
        }
    if result.get("labeled_ids_hash") != expected_contract["labeled_ids_hash"]:
        mismatch["fit_result.labeled_ids_hash"] = {
            "expected": expected_contract["labeled_ids_hash"], "actual": result.get("labeled_ids_hash")
        }
    if result.get("validation_ids_hash") != expected_contract["validation_ids_hash"]:
        mismatch["fit_result.validation_ids_hash"] = {
            "expected": expected_contract["validation_ids_hash"], "actual": result.get("validation_ids_hash")
        }
    if int(result.get("trainable_parameters", -1)) != int(expected_contract["expected_trainable_parameters"]):
        mismatch["fit_result.trainable_parameters"] = {
            "expected": expected_contract["expected_trainable_parameters"],
            "actual": result.get("trainable_parameters"),
        }
    expected_total = int(expected_contract.get("expected_total_parameters", EXPECTED_TOTAL_PARAMETERS))
    if int(result.get("total_parameters", -1)) != expected_total:
        mismatch["fit_result.total_parameters"] = {
            "expected": expected_total, "actual": result.get("total_parameters")
        }
    if mismatch:
        return {"status": "stale", "reason": "contract_or_hash_mismatch", "mismatch": mismatch, "present": present}
    return {
        "status": "complete", "reason": "all_contracts_match", "present": present,
        "fit_result": result, "contract": contract,
    }


def quarantine_fit_runtime(fit_dir: Path, quarantine_root: Path, reason: str) -> Path | None:
    fit_dir = Path(fit_dir)
    if not fit_dir.exists():
        return None
    quarantine_root = Path(quarantine_root)
    quarantine_root.mkdir(parents=True, exist_ok=True)
    safe_key = "__".join(fit_dir.parts[-4:])
    suffix = 0
    while True:
        destination = quarantine_root / f"{safe_key}__{reason}__{suffix}"
        if not destination.exists():
            break
        suffix += 1
    shutil.move(str(fit_dir), str(destination))
    return destination


def write_fit_contract(
    fit_dir: Path,
    expected_contract: dict[str, Any],
    adaptation_train_config_hash: str,
) -> dict[str, Any]:
    fit_dir = Path(fit_dir)
    result = json.loads((fit_dir / "fit_result.json").read_text())
    contract = dict(expected_contract)
    contract.update({
        "adaptation_train_config_hash": adaptation_train_config_hash,
        "checkpoint_sha256": sha256_file(fit_dir / "best.pt"),
    })
    (fit_dir / "formal_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    return result


def execute_fit_plan(
    plan: Sequence[FormalFitSpec],
    runtime_root: Path,
    contract_factory: Callable[[FormalFitSpec], dict[str, Any]],
    fit_executor: Callable[[FormalFitSpec, Path, dict[str, Any]], None],
    max_same_config_retry: int = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run or reuse each neural fit independently with deterministic retry."""
    runtime_root = Path(runtime_root)
    quarantine = runtime_root / "quarantine"
    rows: list[dict[str, Any]] = []
    counters = {name: 0 for name in ("completed", "reused", "rerun", "failed", "initially_missing", "stale_or_config_mismatch", "partial")}
    keys = [spec.run_key for spec in plan]
    duplicate_count = len(keys) - len(set(keys))
    if duplicate_count:
        raise RuntimeError("duplicate formal fit run keys")
    for spec in plan:
        fit_dir = runtime_root / spec.run_key
        contract = contract_factory(spec)
        initial = inspect_fit_runtime(fit_dir, contract)
        row = {**asdict(spec), "run_key": spec.run_key, "initial_status": initial["status"], "attempts": 0}
        if initial["status"] == "complete":
            counters["completed"] += 1; counters["reused"] += 1
            row.update({"final_status": "complete", "action": "reused", "reason": initial["reason"]})
            rows.append(row)
            continue
        counters["initially_missing"] += int(initial["status"] == "missing")
        counters["partial"] += int(initial["status"] == "partial")
        counters["stale_or_config_mismatch"] += int(initial["status"] == "stale")
        if initial["status"] in {"partial", "stale"}:
            quarantine_fit_runtime(fit_dir, quarantine, initial["status"])
            counters["rerun"] += 1
        final = initial
        errors = []
        for attempt in range(max_same_config_retry + 1):
            row["attempts"] = attempt + 1
            try:
                fit_executor(spec, fit_dir, contract)
                final = inspect_fit_runtime(fit_dir, contract)
                if final["status"] == "complete":
                    break
                errors.append(f"post_fit_{final['status']}:{final['reason']}")
            except Exception as error:  # recorded and retried under the identical contract
                errors.append(f"{type(error).__name__}:{error}")
            if fit_dir.exists():
                quarantine_fit_runtime(fit_dir, quarantine, f"retry_{attempt}")
        if row["attempts"] > 1 and initial["status"] == "missing":
            counters["rerun"] += 1
        if final["status"] == "complete":
            counters["completed"] += 1
            row.update({"final_status": "complete", "action": "executed", "reason": "same_config_success"})
        else:
            counters["failed"] += 1
            failure_dir = runtime_root / "failures" / spec.run_key
            failure_dir.mkdir(parents=True, exist_ok=True)
            (failure_dir / "failure.json").write_text(json.dumps({
                "run_key": spec.run_key, "contract": contract, "errors": errors,
                "retry_policy": f"max_same_config_retry={max_same_config_retry}",
            }, indent=2) + "\n")
            row.update({"final_status": "failed", "action": "recorded_failure", "reason": "same_config_retry_exhausted"})
        row["errors"] = errors
        rows.append(row)
    expected = len(plan)
    audit = {
        "expected_neural_fits": expected,
        **counters,
        "executed": int(counters["completed"] - counters["reused"]),
        "missing": int(expected - counters["completed"] - counters["failed"]),
        "completed_plus_failed": int(counters["completed"] + counters["failed"]),
        "duplicate_run_keys": duplicate_count,
        "all_expected_fits_accounted_for": bool(counters["completed"] + counters["failed"] == expected),
        "unresolved_missing": int(expected - counters["completed"] - counters["failed"]),
        "max_same_config_retry": max_same_config_retry,
    }
    return audit, rows


def regression_metrics(truth: np.ndarray, prediction: np.ndarray, scales: Sequence[float]) -> dict[str, float]:
    truth = np.asarray(truth, dtype=float); prediction = np.asarray(prediction, dtype=float)
    scale = np.asarray(scales, dtype=float)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 2:
        raise ValueError("truth and prediction must be matching N x 2 arrays")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("formal evaluation arrays must be finite")
    result: dict[str, float] = {}
    rmses = []
    for index, target in enumerate(("V1", "V2")):
        residual = truth[:, index] - prediction[:, index]
        ss_tot = float(np.square(truth[:, index] - truth[:, index].mean()).sum())
        result[f"{target}_MAE"] = float(np.abs(residual).mean())
        result[f"{target}_RMSE"] = float(np.sqrt(np.square(residual).mean()))
        result[f"{target}_R2"] = float(1 - np.square(residual).sum() / ss_tot) if ss_tot else float("nan")
        rmses.append(result[f"{target}_RMSE"])
    result["combined_NRMSE"] = float(0.5 * (rmses[0] / scale[0] + rmses[1] / scale[1]))
    return result


def compute_aulc(metrics: pd.DataFrame, budgets: Sequence[int]) -> pd.DataFrame:
    required = {"outer_seed", "budget", "method", "combined_NRMSE"}
    if required - set(metrics.columns):
        raise ValueError(f"metrics missing columns: {sorted(required - set(metrics.columns))}")
    frozen_budgets = list(map(int, budgets))
    rows = []
    for (seed, method), group in metrics.groupby(["outer_seed", "method"], sort=False):
        ordered = group.sort_values("budget")
        actual = ordered.budget.astype(int).tolist()
        if actual != frozen_budgets:
            raise ValueError(f"incomplete AULC budget grid for {seed}/{method}: {actual}")
        values = ordered.combined_NRMSE.to_numpy(float)
        area = float(np.trapezoid(values, np.asarray(frozen_budgets, dtype=float)))
        rows.append({
            "outer_seed": int(seed), "method": method,
            "AULC_30_100": area,
            "mean_NRMSE_over_budget_interval": area / float(frozen_budgets[-1] - frozen_budgets[0]),
        })
    return pd.DataFrame(rows)


def paired_aulc_effects(aulc: pd.DataFrame, reference: str = REFERENCE_METHOD) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = aulc.pivot(index="outer_seed", columns="method", values="mean_NRMSE_over_budget_interval")
    if reference not in pivot:
        raise ValueError(f"missing AULC reference: {reference}")
    differences, summary = [], []
    for method in [column for column in pivot.columns if column != reference]:
        delta = pivot[method] - pivot[reference]
        differences.extend({
            "outer_seed": int(seed), "candidate": method, "reference": reference,
            "delta_normalized_AULC_candidate_minus_reference": float(value),
            "candidate_better": bool(value < 0),
        } for seed, value in delta.items())
        values = delta.to_numpy(float)
        mean, median = float(values.mean()), float(np.median(values))
        wins = int(np.sum(values < 0))
        passed = mean < 0 and median < 0 and wins >= 4
        summary.append({
            "candidate": method, "reference": reference,
            "mean_paired_delta_AULC": mean, "median_paired_delta_AULC": median,
            "std_paired_delta_AULC": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "win_count": wins, "outer_seed_count": len(values),
            "stable_low_label_improvement": bool(passed),
            "decision_rule": "mean<0 and median<0 and wins>=4/5",
        })
    return pd.DataFrame(differences), pd.DataFrame(summary)


def stability_gate(differences: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(differences, dtype=float)
    mean, median = float(values.mean()), float(np.median(values))
    wins = int(np.sum(values < 0))
    return {
        "mean_paired_delta_AULC": mean, "median_paired_delta_AULC": median,
        "win_count": wins, "outer_seed_count": len(values),
        "pass": bool(mean < 0 and median < 0 and wins >= 4),
    }


def summarize_methods_by_budget(metrics: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    selected = metrics.loc[metrics.method.isin(methods)].copy()
    rows = []
    for (budget, method), group in selected.groupby(["budget", "method"], sort=True):
        values = group.combined_NRMSE.to_numpy(float)
        rows.append({
            "budget": int(budget), "method": method, "outer_seed_count": len(values),
            "combined_NRMSE_mean": float(values.mean()),
            "combined_NRMSE_median": float(np.median(values)),
            "combined_NRMSE_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        })
    result = pd.DataFrame(rows)
    for budget, group in selected.groupby("budget"):
        pivot = group.pivot(index="outer_seed", columns="method", values="combined_NRMSE")
        winners = pivot.idxmin(axis=1).value_counts()
        for method in methods:
            mask = result.budget.eq(int(budget)) & result.method.eq(method)
            result.loc[mask, "best_by_seed_count"] = int(winners.get(method, 0))
            if REFERENCE_METHOD in pivot:
                delta = pivot[method] - pivot[REFERENCE_METHOD]
                result.loc[mask, "mean_paired_delta_vs_current_last2_head"] = float(delta.mean())
                result.loc[mask, "median_paired_delta_vs_current_last2_head"] = float(np.median(delta))
                result.loc[mask, "paired_differences_by_seed_vs_current_last2_head"] = json.dumps(
                    {str(int(seed)): float(value) for seed, value in delta.items()}, sort_keys=True
                )
    return result


def capacity_crossover_summary(capacity: pd.DataFrame) -> dict[str, Any]:
    mean_best, median_best, wins = {}, {}, {}
    for budget, group in capacity.groupby("budget"):
        budget_key = str(int(budget))
        mean_best[budget_key] = str(group.loc[group.combined_NRMSE_mean.idxmin(), "method"])
        median_best[budget_key] = str(group.loc[group.combined_NRMSE_median.idxmin(), "method"])
        wins[budget_key] = {
            str(row.method): int(row.best_by_seed_count) for row in group.itertuples()
        }
    return {
        "best_mean_method_per_budget": mean_best,
        "best_median_method_per_budget": median_best,
        "number_of_outer_seed_wins_per_method_per_budget": wins,
        "descriptive_capacity_crossover": bool(
            len(set(mean_best.values())) > 1 or len(set(median_best.values())) > 1
        ),
        "interpretation": "descriptive pattern only; not an automatic causal or significance claim",
    }


def completion_gate(
    metrics: pd.DataFrame,
    resume_audit: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    expected_contexts = len(config["outer_seeds"]) * len(config["target_label_budgets"]) * len(FROZEN_METHODS)
    unique_contexts = len(metrics[["outer_seed", "budget", "method"]].drop_duplicates())
    expected_fits = len(build_formal_fit_plan(config))
    completed = int(resume_audit.get("completed", 0))
    failed = int(resume_audit.get("failed", 0))
    missing = int(resume_audit.get("missing", 0))
    passed = (
        unique_contexts == expected_contexts and completed == expected_fits
        and failed == 0 and missing == 0
        and bool(resume_audit.get("all_expected_fits_accounted_for", False))
    )
    return {
        "expected_evaluation_contexts": expected_contexts,
        "observed_evaluation_contexts": unique_contexts,
        "expected_neural_fits": expected_fits,
        "completed_neural_fits": completed, "failed_neural_fits": failed,
        "missing_neural_fits": missing, "pass": bool(passed),
        "final_scientific_decision_allowed": bool(passed),
    }
