"""Reusable primitives for a filtered, full-data target-transfer study.

The functions in this module deliberately keep target-test labels outside the
fitting interfaces.  A runner may apply a pre-specified operational-domain
filter before it creates roles, but after that point only gradient-train and
validation identities can be supplied to a fit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from ..data import build_model_data, eluent_descriptor
from ..models import build_predictor
from ..training.predictor import atomic_json, seed_everything
from .adaptation import PaperStyleCurrentV2, attach_column_context, loader_pair, quantile_target_loss
from .conditional_scaling import fit_conditional
from .evaluation import absolute_error_metrics


TARGETS = ("V1", "V2")


def stable_hash(value: object) -> str:
    """Hash a JSON-compatible record in a platform-independent form."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def ea_fraction(values: Sequence[object]) -> np.ndarray:
    """Convert the repository's ``PE/EA`` strings to the EA volume fraction."""

    result: list[float] = []
    for value in values:
        try:
            pe, ea = (float(part) for part in str(value).split("/"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid PE/EA ratio: {value!r}") from exc
        if pe < 0 or ea < 0 or pe + ea <= 0:
            raise ValueError(f"invalid PE/EA ratio: {value!r}")
        result.append(ea / (pe + ea))
    return np.asarray(result, dtype=float)


def filter_operational_domain(
    data: pd.DataFrame,
    *,
    column: str,
    v1_limit_ml: float,
    v2_limit_ml: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a declared, non-data-selected operational domain.

    The returned audit retains every reader-compatible row and marks whether it
    was retained.  The thresholds are parameters rather than estimates, so no
    test outcome is used to choose them.
    """

    required = {"sample_id", "canonical_smiles", "V1_ml", "V2_ml"}
    missing = required - set(data.columns)
    _assert(not missing, f"operational filter missing columns: {sorted(missing)}")
    _assert(v1_limit_ml > 0 and v2_limit_ml > 0, "operational limits must be positive")
    frame = data.copy().reset_index(drop=True)
    v1 = pd.to_numeric(frame["V1_ml"], errors="raise")
    v2 = pd.to_numeric(frame["V2_ml"], errors="raise")
    _assert(np.isfinite(v1).all() and np.isfinite(v2).all(), "non-finite operational-domain label")
    keep = v1.le(float(v1_limit_ml)) & v2.le(float(v2_limit_ml))
    audit = frame.copy()
    audit["operational_keep"] = keep.to_numpy(bool)
    audit["filter_reason"] = np.where(
        keep,
        "within_legacy_operational_domain",
        np.where(v1.gt(v1_limit_ml) & v2.gt(v2_limit_ml), "V1_and_V2_above_limit",
                 np.where(v1.gt(v1_limit_ml), "V1_above_limit", "V2_above_limit")),
    )
    audit["column"] = str(column)
    audit["V1_limit_ml"] = float(v1_limit_ml)
    audit["V2_limit_ml"] = float(v2_limit_ml)
    filtered = frame.loc[keep].copy().reset_index(drop=True)
    _assert(filtered.sample_id.astype(str).is_unique, "filtered sample IDs must remain unique")
    return filtered, audit


def build_full_data_schedule(
    parent_schedule: pd.DataFrame,
    filtered_by_column: Mapping[str, pd.DataFrame],
    *,
    columns: Sequence[str],
    protocols: Sequence[str],
    seeds: Sequence[int],
    parent_budget: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Intersect the frozen outer schedule and expose all eligible train rows.

    ``pool`` in the historical low-label schedule represents allowed but
    unpurchased labels.  For this full-data benchmark it becomes
    ``gradient_train``; only the frozen validation and test identities retain
    their former roles.  This preserves the outer draw while avoiding the
    misleading claim that B=100 is "full" data.
    """

    required = {"column", "protocol", "outer_seed", "planned_budget", "sample_id", "canonical_smiles", "role"}
    missing = required - set(parent_schedule.columns)
    _assert(not missing, f"parent schedule missing columns: {sorted(missing)}")
    rows: list[pd.DataFrame] = []
    accounts: list[dict[str, object]] = []
    for column in columns:
        data = filtered_by_column[column]
        allowed = set(data.sample_id.astype(str))
        _assert(len(allowed) == len(data), f"duplicate filtered IDs for {column}")
        for protocol in protocols:
            for seed in seeds:
                parent = parent_schedule.loc[
                    parent_schedule.column.eq(column)
                    & parent_schedule.protocol.eq(protocol)
                    & parent_schedule.outer_seed.eq(int(seed))
                    & parent_schedule.planned_budget.eq(int(parent_budget))
                ].copy()
                _assert(not parent.empty, f"missing parent context {column}/{protocol}/{seed}")
                _assert(parent.sample_id.astype(str).is_unique, "parent context has duplicate IDs")
                retained = parent.loc[parent.sample_id.astype(str).isin(allowed)].copy()
                _assert(not retained.empty, f"filter removed every row in {column}/{protocol}/{seed}")
                retained["parent_role"] = retained["role"].astype(str)
                retained["role"] = np.where(
                    retained.parent_role.isin(("validation", "test")), retained.parent_role, "gradient_train"
                )
                retained["full_data_definition"] = "all_retained_non_validation_non_test_rows"
                retained["training_label_count"] = int(retained.role.eq("gradient_train").sum())
                retained["n_total_filtered"] = int(len(retained))
                retained["parent_budget"] = int(parent_budget)
                retained["outer_seed"] = int(seed)
                roles = {role: retained.loc[retained.role.eq(role), "sample_id"].astype(str).tolist()
                         for role in ("gradient_train", "validation", "test")}
                _assert(all(roles.values()), f"empty retained role in {column}/{protocol}/{seed}")
                _assert(not (set(roles["gradient_train"]) & set(roles["validation"])
                             or set(roles["gradient_train"]) & set(roles["test"])
                             or set(roles["validation"]) & set(roles["test"])), "role overlap")
                if protocol == "compound":
                    compound_sets = {
                        role: set(retained.loc[retained.role.eq(role), "canonical_smiles"].astype(str))
                        for role in ("gradient_train", "validation", "test")
                    }
                    _assert(not any(compound_sets[left] & compound_sets[right]
                                    for index, left in enumerate(compound_sets)
                                    for right in tuple(compound_sets)[:index]),
                            f"compound leakage in {column}/{protocol}/{seed}")
                accounts.append({
                    "column": column,
                    "protocol": protocol,
                    "seed": int(seed),
                    "parent_budget": int(parent_budget),
                    "total_filtered_rows": int(len(retained)),
                    "gradient_train_rows": len(roles["gradient_train"]),
                    "validation_rows": len(roles["validation"]),
                    "test_rows": len(roles["test"]),
                    "training_label_count": len(roles["gradient_train"]),
                    "unique_compounds_total": int(retained.canonical_smiles.nunique()),
                    "unique_compounds_gradient_train": int(retained.loc[retained.role.eq("gradient_train"), "canonical_smiles"].nunique()),
                    "unique_compounds_validation": int(retained.loc[retained.role.eq("validation"), "canonical_smiles"].nunique()),
                    "unique_compounds_test": int(retained.loc[retained.role.eq("test"), "canonical_smiles"].nunique()),
                    "compound_leakage_count": 0,
                })
                rows.append(retained)
    schedule = pd.concat(rows, ignore_index=True)
    accounting = pd.DataFrame(accounts).sort_values(["column", "protocol", "seed"]).reset_index(drop=True)
    return schedule, accounting


@dataclass(frozen=True)
class TargetOnlyConditionNormalization:
    """Target-train-only normalization for a randomly initialized V2 model.

    It is intentionally a separate type from the qualified source's
    ``ConditionNormalization``.  That avoids relabelling source normalization
    as target normalization and keeps target-only preprocessing auditable.
    ``TypedConditionCompletionBranch`` only needs this validated value object.
    """

    loading_amount_min: float
    loading_amount_max: float
    loading_volume_min: float
    loading_volume_max: float
    fit_dataset: str
    fit_role: str
    fit_row_count: int
    fit_ids_hash: str
    eluent_scaler_sha256: str

    def validate(self) -> None:
        if self.fit_dataset != "target_only" or self.fit_role != "gradient_train":
            raise ValueError("target-only normalization must use target gradient_train rows")
        if self.fit_row_count < 1 or len(self.fit_ids_hash) != 64:
            raise ValueError("target-only normalization requires an auditable train ledger")
        if self.loading_amount_max <= self.loading_amount_min:
            raise ValueError("target-only loading amount range must be positive")
        if self.loading_volume_max <= self.loading_volume_min:
            raise ValueError("target-only loading volume range must be positive")


def fit_target_only_preprocessing(
    train_frame: pd.DataFrame,
    graph_cache: Mapping[str, Mapping[str, object]],
    *,
    scaler_path: Path,
) -> tuple[TargetOnlyConditionNormalization, dict[str, object]]:
    """Fit every target-only normalizer using gradient-train rows only."""

    required = {"sample_id", "canonical_smiles", "PE/EA", "Density g/ml", "V/ul", "Volume of loading solvent/ul", "V1_ml", "V2_ml"}
    missing = required - set(train_frame.columns)
    _assert(not missing, f"target-only preprocessing missing columns: {sorted(missing)}")
    _assert(len(train_frame) > 1, "target-only preprocessing needs at least two train rows")
    descriptors = np.vstack([graph_cache[str(value)]["descriptor"] for value in train_frame.canonical_smiles]).astype(np.float32)
    eluents = np.vstack([eluent_descriptor(str(value)) for value in train_frame["PE/EA"]]).astype(np.float32)
    scaler = {
        "descriptor": {"min": descriptors.min(axis=0).tolist(), "max": descriptors.max(axis=0).tolist()},
        "eluent": {"min": eluents.min(axis=0).tolist(), "max": eluents.max(axis=0).tolist()},
    }
    atomic_json(scaler_path, scaler)
    ids = sorted(train_frame.sample_id.astype(str))
    amounts = train_frame["Density g/ml"].to_numpy(float) * train_frame["V/ul"].to_numpy(float)
    volumes = train_frame["Volume of loading solvent/ul"].to_numpy(float)
    normalization = TargetOnlyConditionNormalization(
        loading_amount_min=float(amounts.min()), loading_amount_max=float(amounts.max()),
        loading_volume_min=float(volumes.min()), loading_volume_max=float(volumes.max()),
        fit_dataset="target_only", fit_role="gradient_train", fit_row_count=int(len(train_frame)),
        fit_ids_hash=stable_hash(ids), eluent_scaler_sha256=sha256_file(scaler_path),
    )
    normalization.validate()
    scales = {target: float(train_frame[f"{target}_ml"].to_numpy(float).std(ddof=0)) for target in TARGETS}
    _assert(all(np.isfinite(value) and value > 0 for value in scales.values()), "target-only train scales must be positive")
    preprocessing: dict[str, object] = {
        "scaler": scaler,
        "target_scales": scales,
        "fit_role": "target_gradient_train_only",
        "fit_rows": int(len(train_frame)),
        "fit_ids_hash": stable_hash(ids),
        "validation_rows_used": 0,
        "test_rows_used": 0,
        "source_rows_used": 0,
        "condition_normalization": asdict(normalization),
    }
    return normalization, preprocessing


def make_label_scrubbed_graphs(
    feature_frame: pd.DataFrame,
    graph_cache: Mapping[str, Mapping[str, object]],
    scaler: Mapping[str, object],
) -> tuple[list[object], list[object]]:
    """Build model inputs with a hard zero label sentinel on every row."""

    scrubbed = feature_frame.copy().reset_index(drop=True)
    scrubbed["V1_ml"] = 0.0
    scrubbed["V2_ml"] = 0.0
    return build_model_data(scrubbed, dict(graph_cache), pd.DataFrame(), dict(scaler))


def _set_revealed_labels(atom: Sequence[object], indices: Sequence[int], truth: np.ndarray) -> list[object]:
    result = [value.clone() for value in atom]
    targets = np.asarray(truth, dtype=np.float32)
    _assert(targets.shape == (len(indices), 2), "revealed truth does not align with graph indices")
    _assert(np.isfinite(targets).all(), "revealed target label is non-finite")
    for position, value in zip(indices, targets):
        result[int(position)].y = torch.tensor([value], dtype=torch.float32)
    return result


def _point_prediction(model: torch.nn.Module, atom: Sequence[object], angle: Sequence[object], indices: Sequence[int], batch_size: int) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for atom_batch, angle_batch in zip(*loader_pair(atom, angle, indices, batch_size)):
            value = model(atom_batch, angle_batch)
            if value.ndim != 2 or value.shape[1] != 6:
                raise ValueError("target-only model did not return current six-output contract")
            output.append(value.detach().cpu().numpy())
    return np.vstack(output)[:, (1, 4)] if output else np.empty((0, 2), dtype=float)


def fit_target_only_full(
    *,
    feature_frame: pd.DataFrame,
    graph_cache: Mapping[str, Mapping[str, object]],
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    test_indices: Sequence[int],
    train_truth: np.ndarray,
    validation_truth: np.ndarray,
    seed: int,
    config: Mapping[str, object],
    runtime: Path,
    contract: Mapping[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    """Train a random-init current-V2 target-only ceiling without test labels.

    There is intentionally no ``test_truth`` argument.  The returned values
    are frozen point predictions in the supplied test-index order.
    """

    runtime = Path(runtime)
    prediction_path, audit_path = runtime / "predictions.csv.gz", runtime / "fit_audit.json"
    if prediction_path.exists() and audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("contract") != dict(contract) or audit.get("prediction_sha256") != sha256_file(prediction_path):
            raise RuntimeError(f"target-only runtime contract changed: {runtime}")
        stored = pd.read_csv(prediction_path)
        values = stored.loc[:, ["V1_pred", "V2_pred"]].to_numpy(float)
        _assert(len(values) == len(test_indices) and np.isfinite(values).all(), "invalid frozen target-only prediction")
        return values, audit

    runtime.mkdir(parents=True, exist_ok=True)
    train = np.asarray(train_indices, dtype=int)
    valid = np.asarray(validation_indices, dtype=int)
    test = np.asarray(test_indices, dtype=int)
    _assert(len(train) and len(valid) and len(test), "target-only roles must be non-empty")
    _assert(not (set(train) & set(valid) or set(train) & set(test) or set(valid) & set(test)), "target-only role overlap")
    train_frame = feature_frame.iloc[train].copy()
    train_frame[["V1_ml", "V2_ml"]] = np.asarray(train_truth, dtype=float)
    normalization, preprocessing = fit_target_only_preprocessing(train_frame, graph_cache, scaler_path=runtime / "scaler.json")
    atom_base, angle = make_label_scrubbed_graphs(feature_frame, graph_cache, preprocessing["scaler"])
    atom = _set_revealed_labels(atom_base, train, train_truth)
    atom = _set_revealed_labels(atom, valid, validation_truth)
    torch.set_num_threads(int(config["cpu_threads"]))
    seed_everything(int(seed))
    model = build_predictor(normalization)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    scales = preprocessing["target_scales"]
    best_score, best_epoch, stale = float("inf"), 0, 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, int(config["maximum_epochs"]) + 1):
        model.train()
        losses: list[float] = []
        order = np.random.default_rng(int(seed) * 10000 + epoch).permutation(train)
        for atom_batch, angle_batch in zip(*loader_pair(atom, angle, order, int(config["batch_size"]))):
            output = model(atom_batch, angle_batch)
            loss = (quantile_target_loss(atom_batch.y[:, 0], output[:, :3])
                    + quantile_target_loss(atom_batch.y[:, 1], output[:, 3:]))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite target-only training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        valid_point = _point_prediction(model, atom, angle, valid, int(config["batch_size"]))
        score = float(absolute_error_metrics(validation_truth, valid_point, scales)["combined_normalized_rmse"])
        if not math.isfinite(score):
            raise RuntimeError("non-finite target-only validation score")
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)), "validation_score": score})
        if score < best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= int(config["patience"]):
            break
    if best_state is None:
        raise RuntimeError("target-only fit produced no validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    prediction = _point_prediction(model, atom, angle, test, int(config["batch_size"]))
    _assert(np.isfinite(prediction).all(), "non-finite target-only test prediction")
    pd.DataFrame({"V1_pred": prediction[:, 0], "V2_pred": prediction[:, 1]}).to_csv(
        prediction_path, index=False, compression={"method": "gzip", "mtime": 0}
    )
    torch.save({"contract": dict(contract), "model_state_dict": best_state,
                "target_only_normalization": asdict(normalization)}, runtime / "best.pt")
    audit: dict[str, object] = {
        "method": "target_only_full", "contract": dict(contract), "best_epoch": int(best_epoch),
        "epochs_run": len(history), "validation_score": float(best_score), "history": history,
        "random_initialization": "direct_seeded_current_v2_construction", "source_checkpoint_loaded": False,
        "source_predictions_used": False, "source_replay_used": False,
        "preprocessing": preprocessing, "test_labels_used_for_fit_or_selection": 0,
        "test_graph_labels": "zero_sentinel", "prediction_sha256": sha256_file(prediction_path),
        "checkpoint_sha256": sha256_file(runtime / "best.pt"),
    }
    atomic_json(audit_path, audit)
    return prediction, audit


def fit_paper_style_current_v2_full(
    *,
    source_model: torch.nn.Module,
    base_atom: Sequence[object],
    angle: Sequence[object],
    contexts: np.ndarray,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    test_indices: Sequence[int],
    train_truth: np.ndarray,
    validation_truth: np.ndarray,
    source_scales: Mapping[str, float],
    seed: int,
    config: Mapping[str, object],
    runtime: Path,
    contract: Mapping[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    """Run the frozen shallow current-V2 paper-style adaptation.

    This mirrors the matched benchmark's scope and validation-only selection;
    test truth is deliberately absent from the signature.
    """

    runtime = Path(runtime)
    prediction_path, audit_path = runtime / "predictions.csv.gz", runtime / "fit_audit.json"
    if prediction_path.exists() and audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("contract") != dict(contract) or audit.get("prediction_sha256") != sha256_file(prediction_path):
            raise RuntimeError(f"paper-style runtime contract changed: {runtime}")
        stored = pd.read_csv(prediction_path)
        values = stored.loc[:, ["V1_pred", "V2_pred"]].to_numpy(float)
        _assert(len(values) == len(test_indices) and np.isfinite(values).all(), "invalid frozen paper-style prediction")
        return values, audit

    runtime.mkdir(parents=True, exist_ok=True)
    train, valid, test = (np.asarray(value, dtype=int) for value in (train_indices, validation_indices, test_indices))
    _assert(not (set(train) & set(valid) or set(train) & set(test) or set(valid) & set(test)), "paper-style role overlap")
    torch.set_num_threads(int(config["cpu_threads"]))
    seed_everything(int(seed))
    atom = attach_column_context([value.clone() for value in base_atom], contexts)
    atom = _set_revealed_labels(atom, train, train_truth)
    atom = _set_revealed_labels(atom, valid, validation_truth)
    model = PaperStyleCurrentV2(source_model, context_dim=int(config["context_dim"]), scope="shallow")
    _assert(not any(parameter.requires_grad for parameter in model.source_head.parameters()), "paper-style source head must stay frozen")
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
    )
    scale = {target: float(source_scales[target]) for target in TARGETS}
    best_score, best_epoch, stale = float("inf"), 0, 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, int(config["maximum_epochs"]) + 1):
        model.training_target()
        losses: list[float] = []
        order = np.random.default_rng(int(seed) * 10000 + epoch).permutation(train)
        for atom_batch, angle_batch in zip(*loader_pair(atom, angle, order, int(config["batch_size"]))):
            output = model(atom_batch, angle_batch)
            loss = (quantile_target_loss(atom_batch.y[:, 0], output[:, :3])
                    + quantile_target_loss(atom_batch.y[:, 1], output[:, 3:]))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite paper-style training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        valid_point = _point_prediction(model, atom, angle, valid, int(config["batch_size"]))
        score = float(absolute_error_metrics(validation_truth, valid_point, scale)["combined_normalized_rmse"])
        if not math.isfinite(score):
            raise RuntimeError("non-finite paper-style validation score")
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)), "validation_score": score})
        if score < best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= int(config["patience"]):
            break
    if best_state is None:
        raise RuntimeError("paper-style fit produced no validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    prediction = _point_prediction(model, atom, angle, test, int(config["batch_size"]))
    _assert(np.isfinite(prediction).all(), "non-finite paper-style test prediction")
    pd.DataFrame({"V1_pred": prediction[:, 0], "V2_pred": prediction[:, 1]}).to_csv(
        prediction_path, index=False, compression={"method": "gzip", "mtime": 0}
    )
    torch.save({"contract": dict(contract), "model_state_dict": best_state}, runtime / "best.pt")
    audit: dict[str, object] = {
        "method": "paper_style_current_v2", "contract": dict(contract), "best_epoch": int(best_epoch),
        "epochs_run": len(history), "validation_score": float(best_score), "history": history,
        "trainable_scope": "final message layer + condition branch + target head + zero-initialized column adapter; source head frozen",
        "test_labels_used_for_fit_or_selection": 0, "test_graph_labels": "zero_sentinel",
        "prediction_sha256": sha256_file(prediction_path), "checkpoint_sha256": sha256_file(runtime / "best.pt"),
    }
    atomic_json(audit_path, audit)
    return prediction, audit


def select_conditional_ea(
    *,
    train_truth: np.ndarray,
    train_source: np.ndarray,
    train_ea: np.ndarray,
    validation_truth: np.ndarray,
    validation_source: np.ndarray,
    validation_ea: np.ndarray,
    predict_source: np.ndarray,
    predict_ea: np.ndarray,
    source_scales: Sequence[float],
    mass_ratio: float,
    penalties: Sequence[float] = (0.0, 0.1, 1.0),
) -> tuple[np.ndarray, dict[str, object]]:
    """Select the historical EA interaction penalty using validation only."""

    scales = np.asarray(source_scales, dtype=float)
    candidates: list[dict[str, object]] = []
    for penalty in penalties:
        fit = fit_conditional(train_source, train_truth, train_ea, scales, mass_ratio, float(penalty), interaction=True)
        valid_prediction = fit.predict(validation_source, validation_ea)
        score = float(absolute_error_metrics(validation_truth, valid_prediction, scales)["combined_normalized_rmse"])
        candidates.append({"penalty": float(penalty), "validation_score": score, "audit": fit.audit()})
    selected = min(candidates, key=lambda row: (float(row["validation_score"]), float(row["penalty"])))
    fit = fit_conditional(train_source, train_truth, train_ea, scales, mass_ratio, float(selected["penalty"]), interaction=True)
    prediction = fit.predict(predict_source, predict_ea)
    _assert(np.isfinite(prediction).all(), "non-finite conditional-EA prediction")
    return prediction, {
        "fit_role": "gradient_train", "selection_role": "validation",
        "selection": "validation_only_minimum_source_normalized_rmse", "interaction": True,
        "source_q50": True, "mass_ratio": float(mass_ratio), "selected_penalty": float(selected["penalty"]),
        "candidates": candidates, "final_fit": fit.audit(),
    }


__all__ = [
    "TARGETS", "TargetOnlyConditionNormalization", "build_full_data_schedule", "ea_fraction",
    "filter_operational_domain", "fit_paper_style_current_v2_full", "fit_target_only_full",
    "make_label_scrubbed_graphs", "select_conditional_ea", "sha256_file", "stable_hash",
]
