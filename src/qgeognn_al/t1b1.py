"""Frozen planning and analysis contracts for the T1b-1 capacity sweep."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .engine import GraphAdapterTrainConfig, canonical_json_hash
from .t1_formal import compute_aulc, stability_gate


OUTER_SEEDS = [769539383, 1425370602, 536279090, 2767143051, 1362771960]
BUDGETS = [30, 50, 70, 100]
WIDTHS = [8, 16, 32]
SOURCE_MEMBERS = [42, 525, 1101]
ADAPTER_METHODS = {f"graph_adapter_r{width}": width for width in WIDTHS}
CAPACITY_METHODS = [
    "target_head_only", *ADAPTER_METHODS, "last1_head", "current_last2_head",
]
PRIMARY_REFERENCE = "target_head_only"


@dataclass(frozen=True)
class AdapterFitSpec:
    outer_seed: int
    budget: int
    method: str
    bottleneck_width: int
    source_member: int

    @property
    def run_key(self) -> str:
        return (
            f"seed_{self.outer_seed}/budget_{self.budget}/"
            f"{self.method}/member_{self.source_member}"
        )


def validate_config(config: dict[str, Any]) -> None:
    expected = {
        "study": "T1b1_adapter_capacity",
        "protocol": "frozen_t1a_row",
        "outer_seeds": OUTER_SEEDS,
        "target_label_budgets": BUDGETS,
        "fixed_validation": 8,
        "source_members": SOURCE_MEMBERS,
        "adapter_type": "graph_level_residual",
        "activation": "relu",
        "up_projection_initialization": "zeros",
        "bottleneck_widths": WIDTHS,
        "adapter_methods": ADAPTER_METHODS,
        "primary_reference": PRIMARY_REFERENCE,
        "capacity_methods": CAPACITY_METHODS,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items() if config.get(key) != value
    }
    if not isinstance(config.get("formal_authorized"), bool):
        mismatches["formal_authorized"] = {"expected": "boolean", "actual": config.get("formal_authorized")}
    if config.get("scientific_status") not in {"engineering_preregistered", "ready_for_formal_authorization"}:
        mismatches["scientific_status"] = {
            "expected": ["engineering_preregistered", "ready_for_formal_authorization"],
            "actual": config.get("scientific_status"),
        }
    training = config.get("training", {})
    for key, value in {
        "learning_rate": 1e-4, "weight_decay": 1e-5, "epochs": 500,
        "patience": 100, "batch_size": 2048, "v1_weight": 1.0,
        "v2_weight": 1.0, "scaler_policy": "source_train",
        "quantile_parameterization": "monotonic_softplus",
        "checkpoint_selection": "validation_normalized_mse",
    }.items():
        if training.get(key) != value:
            mismatches[f"training.{key}"] = {"expected": value, "actual": training.get(key)}
    gate = config.get("stable_improvement_gate", {})
    if gate != {"mean_delta_lt": 0, "median_delta_lt": 0, "minimum_wins": 4, "outer_seed_count": 5}:
        mismatches["stable_improvement_gate"] = {"expected": "frozen 4/5 gate", "actual": gate}
    if mismatches:
        raise RuntimeError(f"frozen T1b-1 config changed: {mismatches}")


def build_fit_plan(config: dict[str, Any]) -> list[AdapterFitSpec]:
    validate_config(config)
    plan = [
        AdapterFitSpec(int(seed), int(budget), method, int(width), int(member))
        for seed in config["outer_seeds"]
        for budget in config["target_label_budgets"]
        for method, width in config["adapter_methods"].items()
        for member in config["source_members"]
    ]
    keys = [item.run_key for item in plan]
    if len(plan) != 180 or len(keys) != len(set(keys)):
        raise RuntimeError("T1b-1 formal plan must contain exactly 180 unique adapter fits")
    return plan


def adapter_config(spec: AdapterFitSpec, epochs: int = 500, patience: int = 100) -> GraphAdapterTrainConfig:
    return GraphAdapterTrainConfig(
        bottleneck_width=spec.bottleneck_width, epochs=epochs, patience=patience,
    )


def expected_contract(
    spec: AdapterFitSpec,
    config: dict[str, Any],
    gradient_train_ids: Sequence[str],
    validation_ids: Sequence[str],
) -> dict[str, Any]:
    parameters = config["parameter_counts"][spec.method]
    train_config = adapter_config(spec)
    return {
        "schema_version": 1,
        "study": "T1b1_adapter_capacity",
        "run_key": spec.run_key,
        **asdict(spec),
        "adapter_type": config["adapter_type"],
        "adapter_config_hash": canonical_json_hash({
            "adapter_type": config["adapter_type"],
            "input_dim": config["graph_representation_dim"],
            "bottleneck_width": spec.bottleneck_width,
            "activation": config["activation"],
            "up_projection_initialization": config["up_projection_initialization"],
        }),
        "adapter_parameter_count": parameters["adapter_parameters"],
        "formal_config_hash": canonical_json_hash(config),
        "partition_sha256": config["t1a_artifacts"]["partition_sha256"],
        "schedule_sha256": config["t1a_artifacts"]["schedule_sha256"],
        "source_checkpoint_sha256": config["source_checkpoint_hashes"][str(spec.source_member)],
        "gradient_train_ids_hash": canonical_json_hash(sorted(map(str, gradient_train_ids))),
        "validation_ids_hash": canonical_json_hash(sorted(map(str, validation_ids))),
        "labeled_ids_hash": canonical_json_hash(sorted(map(str, [*gradient_train_ids, *validation_ids]))),
        "expected_trainable_parameters": parameters["total_trainable_parameters"],
        "expected_total_parameters": parameters["total_model_parameters"],
        "adaptation_train_config_hash": train_config.config_hash,
    }


def paired_effects(aulc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = aulc.pivot(index="outer_seed", columns="method", values="mean_NRMSE_over_budget_interval")
    rows, summaries = [], []
    if PRIMARY_REFERENCE not in pivot:
        raise ValueError("missing frozen target_head_only reference")
    for method in ADAPTER_METHODS:
        if method not in pivot:
            raise ValueError(f"missing adapter method: {method}")
        delta = pivot[method] - pivot[PRIMARY_REFERENCE]
        gate = stability_gate(delta.to_numpy(float))
        rows.extend({
            "outer_seed": int(seed), "candidate": method, "reference": PRIMARY_REFERENCE,
            "delta_normalized_AULC_candidate_minus_head": float(value),
            "candidate_better": bool(value < 0),
        } for seed, value in delta.items())
        summaries.append({
            "candidate": method, "reference": PRIMARY_REFERENCE,
            **gate, "decision_rule": "mean<0 and median<0 and wins>=4/5",
        })
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def capacity_summaries(
    metrics: pd.DataFrame, parameter_counts: dict[str, int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = metrics.loc[metrics.method.isin(CAPACITY_METHODS)].copy()
    expected = len(OUTER_SEEDS) * len(BUDGETS) * len(CAPACITY_METHODS)
    if len(selected) != expected or len(selected[["outer_seed", "budget", "method"]].drop_duplicates()) != expected:
        raise ValueError("capacity metrics must contain the complete 5x4x6 grid")
    curve_rows = []
    for (budget, method), group in selected.groupby(["budget", "method"], sort=False):
        values = group.combined_NRMSE.to_numpy(float)
        winners = selected.loc[selected.budget.eq(budget)].pivot(
            index="outer_seed", columns="method", values="combined_NRMSE"
        ).idxmin(axis=1).value_counts()
        curve_rows.append({
            "method": method, "trainable_parameters": int(parameter_counts[method]),
            "budget": int(budget), "mean_NRMSE": float(values.mean()),
            "median_NRMSE": float(np.median(values)), "std_NRMSE": float(values.std(ddof=1)),
            "best_by_seed_count": int(winners.get(method, 0)),
        })
    aulc = compute_aulc(selected, BUDGETS)
    _, paired = paired_effects(aulc)
    aulc_rows = []
    for method, group in aulc.groupby("method", sort=False):
        values = group.mean_NRMSE_over_budget_interval.to_numpy(float)
        row = {
            "method": method, "trainable_parameters": int(parameter_counts[method]),
            "mean_normalized_AULC": float(values.mean()),
            "median_normalized_AULC": float(np.median(values)),
            "std": float(values.std(ddof=1)),
            "paired_mean_delta_vs_head": 0.0 if method == PRIMARY_REFERENCE else None,
            "paired_median_delta_vs_head": 0.0 if method == PRIMARY_REFERENCE else None,
            "wins_vs_head": 0 if method == PRIMARY_REFERENCE else None,
            "stable_gate_pass": False,
        }
        match = paired.loc[paired.candidate.eq(method)]
        if not match.empty:
            row.update({
                "paired_mean_delta_vs_head": float(match.mean_paired_delta_AULC.iloc[0]),
                "paired_median_delta_vs_head": float(match.median_paired_delta_AULC.iloc[0]),
                "wins_vs_head": int(match.win_count.iloc[0]),
                "stable_gate_pass": bool(match["pass"].iloc[0]),
            })
        aulc_rows.append(row)
    return pd.DataFrame(curve_rows), pd.DataFrame(aulc_rows)


def completion_gate(metrics: pd.DataFrame, resume_audit: dict[str, Any]) -> dict[str, Any]:
    contexts = len(metrics[["outer_seed", "budget", "method"]].drop_duplicates())
    passed = (
        contexts == 120 and int(resume_audit.get("completed", 0)) == 180
        and int(resume_audit.get("failed", 0)) == 0
        and int(resume_audit.get("missing", 0)) == 0
    )
    return {
        "expected_evaluation_contexts": 120,
        "observed_evaluation_contexts": contexts,
        "expected_new_adapter_fits": 180,
        "completed_new_adapter_fits": int(resume_audit.get("completed", 0)),
        "failed_new_adapter_fits": int(resume_audit.get("failed", 0)),
        "missing_new_adapter_fits": int(resume_audit.get("missing", 0)),
        "pass": bool(passed),
        "final_scientific_decision_allowed": bool(passed),
    }
