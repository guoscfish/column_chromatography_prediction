"""Offline numerical batch diagnostics; no label reader or online transport."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _array_hash(value: np.ndarray) -> str:
    """Hash shape and canonical little-endian float64 bytes without large JSON copies."""
    digest = hashlib.sha256(json.dumps(list(value.shape)).encode())
    digest.update(np.ascontiguousarray(value, dtype="<f8").tobytes())
    return digest.hexdigest()


def _indices(values: Sequence[int], n: int) -> np.ndarray:
    values = list(values)
    if not values or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                         for v in values):
        raise ValueError("A nonempty sequence of integer candidate indices is required")
    result = np.asarray(values, dtype=np.int64)
    if np.any(result < 0) or np.any(result >= n) or len(set(values)) != len(values):
        raise ValueError("Numerical adapter expects unique, in-range, shared-validated indices")
    return result


def _covariance_and_noise(covariance: Any, noise: Any) -> tuple[np.ndarray, np.ndarray]:
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or not covariance.size:
        raise ValueError("Posterior latent covariance must be a nonempty square matrix")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Posterior covariance contains nonfinite values")
    scale = max(1.0, float(np.max(np.abs(covariance))))
    if np.max(np.abs(covariance - covariance.T)) > 1e-10 * scale:
        raise ValueError("Posterior covariance is not symmetric")
    covariance = (covariance + covariance.T) / 2.0
    if np.min(np.diag(covariance)) < -1e-10 * scale:
        raise ValueError("Posterior covariance has negative variances")
    noise = np.asarray(noise, dtype=np.float64)
    if noise.ndim == 0:
        noise = np.full(covariance.shape[0], float(noise), dtype=np.float64)
    if noise.shape != (covariance.shape[0],) or not np.all(np.isfinite(noise)) or np.any(noise <= 0):
        raise ValueError("Observation noise variances must be finite and strictly positive")
    return covariance, noise


def conditional_information_gain(posterior_covariance: Any, noise_variance: Any,
                                 batch_indices: Sequence[int]) -> dict[str, Any]:
    """Compute I(f_B; y_B | D_L) from the *joint* posterior latent covariance.

    Noise is observation variance in the same units as the latent covariance.
    Conditioning on observed labels must already have been done by the GP.
    """
    covariance, noise = _covariance_and_noise(posterior_covariance, noise_variance)
    idx = _indices(batch_indices, len(noise))
    selected = covariance[np.ix_(idx, idx)]
    eigenvalues = np.linalg.eigvalsh(selected)
    scale = max(1.0, float(np.max(np.abs(selected))))
    if eigenvalues.min() < -1e-10 * scale:
        raise ValueError("Selected posterior covariance is not positive semidefinite")
    inverse_std = 1.0 / np.sqrt(noise[idx])
    information_matrix = np.eye(len(idx), dtype=np.float64) + (
        inverse_std[:, None] * selected * inverse_std[None, :])
    numerical_scale = max(1.0, float(np.max(np.diag(information_matrix))))
    factor = None
    used_jitter = None
    for fraction in (0.0, 1e-12, 1e-10, 1e-8):
        jitter = fraction * numerical_scale
        try:
            factor = np.linalg.cholesky(information_matrix + np.eye(len(idx)) * jitter)
            used_jitter = jitter
            break
        except np.linalg.LinAlgError:
            continue
    if factor is None:
        raise ValueError("Information covariance Cholesky failed; no diagnostic is certifiable")
    value = float(np.sum(np.log(np.diag(factor))))
    if not np.isfinite(value) or value < -1e-10:
        raise ValueError("Information gain is nonfinite or negative")
    # This upper bound is reported to make the independence approximation visible.
    independent_upper = float(0.5 * np.log1p(np.maximum(np.diag(selected), 0) / noise[idx]).sum())
    return {"value_nats": max(0.0, value), "joint_logdet": True,
            "conditioning": "posterior_already_conditioned_on_observed_L",
            "independent_marginal_upper_bound_nats": independent_upper,
            "cholesky_jitter": used_jitter, "dtype": "float64",
            "recompute_payload": {"batch_latent_covariance": selected.tolist(),
                                  "batch_noise_variance": noise[idx].tolist()},
            "minimum_batch_covariance_eigenvalue": float(eigenvalues.min())}


def _structural_exemptions(metadata: Sequence[Mapping[str, Any]] | None,
                           claims: Sequence[Mapping[str, Any]] | None,
                           approved_hypothesis_ids: Sequence[str],
                           condition_fields: Sequence[str], n: int) -> tuple[set[tuple[int, int]], list[dict]]:
    """Verify exact single-factor structure; free-text rationales confer no exemption."""
    accepted: set[tuple[int, int]] = set()
    report = []
    if not claims:
        return accepted, report
    if metadata is None or len(metadata) != n:
        return accepted, [{"status": "indeterminate", "reason": "candidate_metadata_unavailable"}]
    ids = {row.get("id"): i for i, row in enumerate(metadata)}
    if len(ids) != n:
        raise ValueError("Candidate metadata IDs must be unique")
    for claim in claims:
        outcome = {"claim": copy.deepcopy(dict(claim)), "status": "rejected", "reason": None}
        pair = claim.get("candidate_ids", [])
        if len(pair) != 2 or pair[0] == pair[1] or any(v not in ids for v in pair):
            outcome["reason"] = "requires_two_distinct_catalog_candidates"
        elif not condition_fields:
            outcome.update(status="indeterminate", reason="condition_schema_not_frozen")
        elif claim.get("hypothesis_id") not in set(approved_hypothesis_ids):
            outcome["reason"] = "hypothesis_not_in_shared_validated_frozen_ledger"
        else:
            left, right = (metadata[ids[v]] for v in pair)
            lc, rc = left.get("conditions", {}), right.get("conditions", {})
            complete = set(lc) == set(rc) == set(condition_fields)
            if not complete:
                outcome.update(status="indeterminate", reason="incomplete_condition_metadata")
            elif not left.get("molecule_id") or left.get("molecule_id") != right.get("molecule_id"):
                outcome["reason"] = "molecule_mismatch"
            else:
                changed = [key for key in condition_fields if lc[key] != rc[key]]
                if changed != [claim.get("varied_factor")]:
                    outcome["reason"] = "not_an_exact_single_factor_contrast"
                else:
                    accepted.add(tuple(sorted((ids[pair[0]], ids[pair[1]]))))
                    outcome.update(status="verified_structural_only", reason="exact_single_factor_contrast")
        report.append(outcome)
    return accepted, report


def _geometry_diagnostics(geometry: Mapping[str, Any] | None, idx: np.ndarray,
                          n: int) -> dict[str, Any]:
    if geometry is None:
        return {"status": "unavailable", "enforced_global_coverage_minimum": None}
    candidates = np.asarray(geometry["candidate_features"], dtype=np.float64)
    observed = np.asarray(geometry["observed_features"], dtype=np.float64)
    if (candidates.ndim != 2 or observed.ndim != 2 or candidates.shape[0] != n or
            candidates.shape[1] != observed.shape[1] or not len(observed) or
            not np.all(np.isfinite(candidates)) or not np.all(np.isfinite(observed))):
        raise ValueError("Geometry requires finite candidate and observed feature matrices")
    chosen = candidates[idx]
    nearest = np.sqrt(np.maximum(((chosen[:, None, :] - observed[None, :, :]) ** 2).sum(axis=2), 0)).min(axis=1)
    pairwise = np.sqrt(np.maximum(((chosen[:, None, :] - chosen[None, :, :]) ** 2).sum(axis=2), 0))
    distances = pairwise[np.triu_indices(len(idx), 1)]
    return {"status": "descriptive_only", "representation_id": geometry.get("representation_id"),
            "state_sha256": geometry.get("state_sha256"),
            "nearest_observed_distance": nearest.tolist(),
            "batch_pairwise_distance_min": float(distances.min()) if distances.size else None,
            "batch_pairwise_distance_median": float(np.median(distances)) if distances.size else None,
            "enforced_global_coverage_minimum": None}


def diagnose_batch(posterior_covariance: Any, noise_variance: Any,
                   batch_indices: Sequence[int], *, config: Mapping[str, Any],
                   reference_batch_indices: Sequence[int] | None = None,
                   reference_scope: Mapping[str, Any] | None = None,
                   metadata: Sequence[Mapping[str, Any]] | None = None,
                   matched_control_claims: Sequence[Mapping[str, Any]] | None = None,
                   approved_hypothesis_ids: Sequence[str] = (),
                   risk_evidence: Mapping[str, Any] | None = None,
                   geometry: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate development probes without a weighted objective or candidate replacement."""
    covariance, noise = _covariance_and_noise(posterior_covariance, noise_variance)
    idx = _indices(batch_indices, len(noise))
    ig = conditional_information_gain(covariance, noise, idx)
    thresholds = config.get("thresholds", {})
    triggers, indeterminate = [], []
    if ig["cholesky_jitter"]:
        indeterminate.append("numerical_jitter_perturbs_information_gain")
    reference = None
    ratio = None
    if reference_batch_indices is not None:
        ref_idx = _indices(reference_batch_indices, len(noise))
        if len(ref_idx) != len(idx):
            raise ValueError("Frozen IG reference must have the same batch size")
        reference = conditional_information_gain(covariance, noise, ref_idx)
        if reference["cholesky_jitter"]:
            indeterminate.append("numerical_jitter_perturbs_reference_information_gain")
        if reference["value_nats"] > 1e-14:
            ratio = ig["value_nats"] / reference["value_nats"]
    lower = thresholds.get("minimum_reference_ig_ratio")
    if lower is not None:
        if not np.isfinite(lower) or not 0 <= lower <= 1:
            raise ValueError("Minimum reference IG ratio must be in [0, 1]")
        required_scope = ("batch_experiment_domain_id", "reference_experiment_domain_id",
                          "batch_design_structure_sha256", "reference_design_structure_sha256",
                          "freeze_manifest_sha256")
        scope_complete = reference_scope and all(reference_scope.get(k) for k in required_scope)
        if not scope_complete:
            indeterminate.append("ig_reference_scope_not_frozen")
        elif (reference_scope["batch_experiment_domain_id"] != reference_scope["reference_experiment_domain_id"] or
              reference_scope["batch_design_structure_sha256"] != reference_scope["reference_design_structure_sha256"]):
            indeterminate.append("ig_reference_experiment_domain_or_structure_mismatch")
        elif ratio is None:
            indeterminate.append("reference_information_gain_missing_or_zero")
        elif ratio < lower:
            triggers.append({"constraint": "minimum_reference_ig_ratio", "value": ratio,
                             "threshold": float(lower), "reason": "joint_conditional_ig_below_frozen_reference_fraction"})
    exempt, control_report = _structural_exemptions(metadata, matched_control_claims,
        approved_hypothesis_ids, config.get("condition_fields", []), len(noise))
    selected = covariance[np.ix_(idx, idx)]
    variances = np.maximum(np.diag(selected), 0)
    stdprod = np.sqrt(variances[:, None] * variances[None, :])
    correlation = np.divide(selected, stdprod, out=np.zeros_like(selected), where=stdprod > 0)
    rho = thresholds.get("absolute_posterior_correlation")
    flagged, exempted = [], []
    if rho is not None:
        if not np.isfinite(rho) or not 0 < rho <= 1:
            raise ValueError("Correlation probe threshold must be in (0, 1]")
        for a, b in zip(*np.triu_indices(len(idx), 1)):
            if abs(correlation[a, b]) >= rho:
                row = {"candidate_indices": [int(idx[a]), int(idx[b])],
                       "absolute_correlation": float(abs(correlation[a, b]))}
                (exempted if tuple(sorted((int(idx[a]), int(idx[b])))) in exempt else flagged).append(row)
        maximum = thresholds.get("maximum_unexempt_correlated_pairs")
        if maximum is not None:
            if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
                raise ValueError("Maximum correlated pair count must be a nonnegative integer")
            if len(flagged) > maximum:
                triggers.append({"constraint": "maximum_unexempt_correlated_pairs", "value": len(flagged),
                                 "threshold": maximum, "reason": "highly_correlated_pairs_without_verified_single_factor_structure"})
    elif thresholds.get("maximum_unexempt_correlated_pairs") is not None:
        indeterminate.append("correlation_threshold_not_calibrated")
    max_correlation = float(np.abs(correlation[np.triu_indices(len(idx), 1)]).max()) if len(idx) > 1 else None
    risk = {"status": "indeterminate", "reason": "calibrated_checkpoint_compatible_risk_evidence_required"}
    required_risk_fields = ("calibration_manifest_sha256", "checkpoint_sha256", "risk_scores", "blindspot_mask")
    if (risk_evidence and risk_evidence.get("calibration_status") == "qualified_on_frozen_development" and
            all(risk_evidence.get(k) is not None for k in required_risk_fields) and
            risk_evidence.get("checkpoint_sha256") == config.get("checkpoint_sha256") and
            config.get("checkpoint_sha256")):
        scores = np.asarray(risk_evidence["risk_scores"], dtype=np.float64)
        raw_mask = np.asarray(risk_evidence["blindspot_mask"])
        if scores.shape != (len(noise),) or not np.all(np.isfinite(scores)) or np.any(scores < 0) or raw_mask.shape != (len(noise),) or raw_mask.dtype.kind != "b":
            raise ValueError("Risk evidence requires finite aligned scores and a boolean blindspot mask")
        risk = {"status": "qualified_external_evidence", "selected_risk_scores": scores[idx].tolist(),
                "selected_blindspot_count": int(raw_mask[idx].sum()),
                "pool_blindspot_count": int(raw_mask.sum()),
                "calibration_manifest_sha256": risk_evidence["calibration_manifest_sha256"],
                "qualification_checked_by": "external_offline_GP_qualification_not_controller"}
    required = thresholds.get("minimum_blindspot_count")
    if required is not None:
        if isinstance(required, bool) or not isinstance(required, int) or required < 0 or required > len(idx):
            raise ValueError("Minimum blindspot count must be between zero and batch size")
        if risk["status"] == "indeterminate":
            indeterminate.append("blindspot_constraint_has_no_qualified_risk_evidence")
        elif risk["pool_blindspot_count"] < required:
            indeterminate.append("blindspot_constraint_infeasible_in_current_pool")
        elif risk["selected_blindspot_count"] < required:
            triggers.append({"constraint": "minimum_blindspot_count", "value": risk["selected_blindspot_count"],
                             "threshold": required, "reason": "qualified_risk_blindspot_underrepresented"})
    calibrated = config.get("calibration_status") == "frozen_development_calibration"
    if not calibrated:
        indeterminate.append("thresholds_not_frozen_after_development_calibration")
    if not any(value is not None for value in thresholds.values()):
        indeterminate.append("no_decision_quality_thresholds_are_active")
    return {"schema_version": 1, "scope": "offline_development_only", "deployable": False,
            "decision_status": "indeterminate" if indeterminate else ("revision_probe_triggered" if triggers else "development_probes_passed"),
            "threshold_calibration_status": config.get("calibration_status", "not_calibrated"),
            "config_sha256": stable_hash(config), "batch_indices": idx.tolist(),
            "input_fingerprints": {"array_encoding": "shape_json_then_C_order_float64_little_endian_bytes",
                                   "posterior_covariance_sha256": _array_hash(covariance),
                                   "noise_variance_sha256": _array_hash(noise),
                                   "candidate_metadata_sha256": stable_hash(metadata) if metadata is not None else None,
                                   "matched_control_claims_sha256": stable_hash(matched_control_claims)},
            "conditional_information_gain": ig, "reference_information_gain": reference,
            "reference_ig_ratio": ratio, "reference_comparison_scope": copy.deepcopy(reference_scope),
            "redundancy": {"maximum_absolute_posterior_correlation": max_correlation,
                           "unexempt_high_correlation_pairs": flagged, "exempt_structural_control_pairs": exempted,
                           "scanned_pair_count": len(idx) * (len(idx) - 1) // 2},
            "single_factor_structural_checks": control_report,
            "risk_blindspot": risk, "geometry": _geometry_diagnostics(geometry, idx, len(noise)),
            "triggered_constraints": triggers, "indeterminate_reasons": indeterminate}


def proposal_difference(before: Mapping[str, Any], after: Mapping[str, Any], reasons: Sequence[str]) -> dict[str, Any]:
    left = [row["id"] for row in before["choices"]]
    right = [row["id"] for row in after["choices"]]
    left_rows = {row["id"]: row for row in before["choices"]}
    right_rows = {row["id"]: row for row in after["choices"]}
    return {"removed_ids": [v for v in left if v not in right],
            "added_ids": [v for v in right if v not in left],
            "choice_level_changes": [{"id": key, "before": copy.deepcopy(left_rows.get(key)),
                                      "after": copy.deepcopy(right_rows.get(key)),
                                      "controller_trigger_reasons": list(reasons),
                                      "llm_choice_reason": right_rows.get(key, {}).get("reason")}
                                     for key in dict.fromkeys(left + right)
                                     if left_rows.get(key) != right_rows.get(key)],
            "retained_count": len(set(left) & set(right)), "order_changed": left != right,
            "constraint_feedback_reasons": list(reasons),
            "llm_revision_reason": after.get("revision_reason"),
            "before_sha256": stable_hash(before), "after_sha256": stable_hash(after)}


def run_mock_controller(initial_proposal: Mapping[str, Any], *, candidate_ids: Sequence[str],
                        shared_validate: Callable[[Mapping[str, Any]], Any],
                        diagnose: Callable[[Sequence[int]], dict[str, Any]],
                        revise: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
                        max_revisions: int = 1) -> dict[str, Any]:
    """Exercise the audit trail with an in-memory mock callback, never an API client.

    The caller must supply the same production validation hook for all three arms.
    No candidate is inserted or removed by the controller itself.
    """
    if max_revisions < 0 or not isinstance(max_revisions, int):
        raise ValueError("Maximum mock revisions must be a nonnegative integer")
    lookup = {value: i for i, value in enumerate(candidate_ids)}
    if len(lookup) != len(candidate_ids):
        raise ValueError("Frozen candidate catalog must have unique IDs")
    original = copy.deepcopy(dict(initial_proposal))
    current = copy.deepcopy(original)
    records, changes = [], []
    final_status = "unresolved"
    for attempt in range(max_revisions + 1):
        shared_validate(current)
        indices = [lookup[row["id"]] for row in current["choices"]]
        try:
            diagnostics = diagnose(indices)
        except (ValueError, np.linalg.LinAlgError) as error:
            diagnostics = {"decision_status": "indeterminate", "deployable": False,
                           "triggered_constraints": [], "indeterminate_reasons": ["numerical_diagnostic_failed"],
                           "error": {"class": type(error).__name__, "message": str(error)}}
        triggers = diagnostics["triggered_constraints"]
        requested = bool(triggers) and attempt < max_revisions
        feedback = {"type": "numerical_development_feedback", "diagnostics": diagnostics,
                    "triggered_constraints": triggers,
                    "instruction": "Return a complete batch; preserve justified exact single-factor controls. Explain each change. This is an offline mock."}
        records.append({"attempt": attempt, "raw_llm_proposal": copy.deepcopy(current),
                        "numerical_diagnostics": diagnostics, "triggered_constraints": triggers,
                        "revision_requested": requested, "feedback": feedback if requested else None})
        if not requested:
            final_status = "unresolved_constraint" if triggers else (
                "indeterminate" if diagnostics["indeterminate_reasons"] else "passed_mock_probes")
            break
        revised = copy.deepcopy(dict(revise(copy.deepcopy(current), copy.deepcopy(feedback))))
        # A revision is validated before it can become a recorded final proposal.
        shared_validate(revised)
        changes.append(proposal_difference(current, revised, [row["reason"] for row in triggers]))
        current = revised
    return {"schema_version": 1, "execution_mode": "mock_no_network", "deployable": False,
            "raw_original_proposal": original, "attempts": records, "proposal_changes": changes,
            "revision_requested": any(row["revision_requested"] for row in records),
            "final_selection": copy.deepcopy(current), "final_status": final_status,
            "original_to_final_difference": proposal_difference(original, current,
                [row["reason"] for attempt in records for row in attempt["triggered_constraints"]])}
