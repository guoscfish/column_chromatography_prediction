"""Pure offline Numerical Critic over explicitly supplied frozen, label-free blocks.

This module has no file, model, selector, network, or label access. It reports
existing geometric/condition diagnostics and never edits, ranks, or accepts a
plan. See NUMERICAL_CRITIC_INTERFACE.md for the provenance trust boundary.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np


CONDITIONS = ("PE/EA", "Density g/ml", "V/ul", "loading solvent", "Volume of loading solvent/ul")
KINDS = {
    "current_full_network_q50_gradient_CountSketch",
    "current_center_width_linear_output_gradient_CountSketch",
}


class FrozenInputError(ValueError):
    """Input has an inconsistent freeze, schema, time boundary, or legal batch."""


def payload_hash(payload: Any) -> str:
    """SHA256 of canonical UTF-8 JSON, not the original file-byte SHA256."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise FrozenInputError(message)


def _keys(value, required, optional=(), *, name):
    _check(isinstance(value, Mapping), f"{name}: expected object")
    _check(set(required) <= set(value) <= set(required) | set(optional),
           f"{name}: missing or unknown fields")


def _time(value, name):
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        _check(stamp.tzinfo is not None, f"{name}: timezone required")
        return stamp.astimezone(timezone.utc)
    except (TypeError, AttributeError, ValueError) as exc:
        raise FrozenInputError(f"{name}: invalid timezone-aware timestamp") from exc


def _digest(value, name):
    _check(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
           f"{name}: expected lowercase SHA256")


def _ids(value, name, *, nonempty=True):
    _check(isinstance(value, list) and (bool(value) or not nonempty), f"{name}: expected ID list")
    _check(all(isinstance(i, str) and bool(i) for i in value), f"{name}: invalid ID")
    _check(len(set(value)) == len(value), f"{name}: duplicate IDs")
    return value


def _number(value, name):
    _check(isinstance(value, (int, float)) and not isinstance(value, bool)
           and math.isfinite(value), f"{name}: expected finite number")


def _matrix(value, rows, columns, name):
    # JSON-number type checking also prevents silently coercing numeric strings.
    _check(isinstance(value, list) and len(value) == rows, f"{name}: row mismatch")
    _check(all(isinstance(r, list) and len(r) == columns for r in value), f"{name}: dimension mismatch")
    for row in value:
        for entry in row:
            _number(entry, name)
    return np.array(value, dtype=np.float64, copy=True)


def _block(block, name, *, state_hash, checkpoint, cutoff):
    _keys(block, ("source", "payload"), name=name)
    src = block["source"]
    _keys(src, ("source_id", "payload_sha256", "frozen_at", "state_sha256", "checkpoint_sha256"),
          name=f"{name}.source")
    _check(isinstance(src["source_id"], str) and bool(src["source_id"]), f"{name}: missing source identity")
    for key in ("payload_sha256", "state_sha256", "checkpoint_sha256"):
        _digest(src[key], f"{name}.{key}")
    try:
        digest = payload_hash(block["payload"])
    except (TypeError, ValueError) as exc:
        raise FrozenInputError(f"{name}: noncanonical or nonfinite JSON payload") from exc
    _check(src["payload_sha256"] == digest, f"{name}: payload hash mismatch")
    _check(src["state_sha256"] == state_hash, f"{name}: frozen state mismatch")
    _check(src["checkpoint_sha256"] == checkpoint, f"{name}: checkpoint mismatch")
    _check(_time(src["frozen_at"], name) <= cutoff, f"{name}: frozen after selection")
    return block["payload"]


def _validate(snapshot):
    _keys(snapshot, ("schema_version", "state", "metadata", "predictions", "gradient_spaces", "plan", "feedback"),
          name="snapshot")
    _check(snapshot["schema_version"] == "numerical-critic-v1", "unsupported schema version")
    _keys(snapshot["state"], ("source", "payload"), name="state")
    state = snapshot["state"]["payload"]
    _keys(state, ("seed", "round", "selection_at", "checkpoint_sha256", "pool_partition", "synthetic", "L_t_ids", "U_t_ids"),
          name="state.payload")
    _check(type(state["seed"]) is int and type(state["round"]) is int and state["round"] >= 0,
           "state: invalid seed or round")
    _check(type(state["synthetic"]) is bool, "state: synthetic must be boolean")
    _check(state["pool_partition"] == "train_pool", "state: only train_pool permitted")
    _digest(state["checkpoint_sha256"], "checkpoint_sha256")
    cutoff = _time(state["selection_at"], "selection_at")
    L, U = _ids(state["L_t_ids"], "L_t_ids"), _ids(state["U_t_ids"], "U_t_ids")
    _check(not set(L) & set(U), "L_t and U_t overlap")
    try:
        state_hash = payload_hash(state)
    except (TypeError, ValueError) as exc:
        raise FrozenInputError("state: noncanonical JSON payload") from exc
    params = dict(state_hash=state_hash, checkpoint=state["checkpoint_sha256"], cutoff=cutoff)
    _block(snapshot["state"], "state", **params)
    ordered = L + U

    plan = _block(snapshot["plan"], "plan", **params)
    _keys(plan, ("ids",), name="plan.payload")
    selected = _ids(plan["ids"], "plan.ids")
    _check(len(selected) == 32 and set(selected) <= set(U), "plan: expected 32 unique IDs in frozen U_t")

    meta = _block(snapshot["metadata"], "metadata", **params)
    _keys(meta, ("ids", "rows"), name="metadata.payload")
    _check(_ids(meta["ids"], "metadata.ids") == ordered, "metadata: expected ordered L_t + U_t")
    _check(isinstance(meta["rows"], list) and len(meta["rows"]) == len(ordered), "metadata: row mismatch")
    for identifier, row in zip(ordered, meta["rows"]):
        _keys(row, ("id", "molecule", "conditions"), name="metadata.row")
        _check(row["id"] == identifier and isinstance(row["molecule"], str) and bool(row["molecule"]),
               "metadata: ID or molecule mismatch")
        c = row["conditions"]
        _keys(c, CONDITIONS, name="conditions")
        _check(isinstance(c["loading solvent"], str) and bool(c["loading solvent"]), "conditions: invalid solvent")
        try:
            pe, ea = map(float, c["PE/EA"].split("/"))
            _check(math.isfinite(pe + ea) and min(pe, ea) >= 0 and pe + ea > 0, "conditions: invalid PE/EA")
        except (TypeError, AttributeError, ValueError) as exc:
            raise FrozenInputError("conditions: invalid PE/EA") from exc
        for axis in ("Density g/ml", "V/ul", "Volume of loading solvent/ul"):
            _number(c[axis], f"conditions.{axis}")
    metadata = dict(zip(ordered, meta["rows"]))

    pred = _block(snapshot["predictions"], "predictions", **params)
    _keys(pred, ("ids", "q50"), ("q10", "q90"), name="predictions.payload")
    _check(_ids(pred["ids"], "predictions.ids") == ordered, "predictions: ordered IDs mismatch")
    q50 = _matrix(pred["q50"], len(ordered), 2, "q50")
    _check(("q10" in pred) == ("q90" in pred), "predictions: q10/q90 must be paired")
    width = crossing = None
    if "q10" in pred:
        q10 = _matrix(pred["q10"], len(ordered), 2, "q10")
        q90 = _matrix(pred["q90"], len(ordered), 2, "q90")
        # Frozen QGeoGNN quantiles can cross. This is a model diagnostic, not a
        # malformed plan or a reason to reject an otherwise valid frozen state.
        crossing = (q10 > q50) | (q50 > q90)
        width = q90 - q10

    _check(isinstance(snapshot["gradient_spaces"], list) and bool(snapshot["gradient_spaces"]),
           "gradient_spaces: at least one existing bank required")
    spaces = {}
    for block in snapshot["gradient_spaces"]:
        bank = _block(block, "gradient_space", **params)
        _keys(bank, ("name", "kind", "ids", "features", "preprocessing"), name="gradient_space.payload")
        _check(isinstance(bank["name"], str) and bool(bank["name"]) and bank["name"] not in spaces,
               "gradient_space: duplicate or empty name")
        _check(bank["kind"] in KINDS or (state["synthetic"] and bank["kind"] == "mock_frozen_gradient"),
               "gradient_space: unsupported existing representation kind")
        _check(isinstance(bank["preprocessing"], str) and bool(bank["preprocessing"]), "gradient_space: preprocessing provenance missing")
        _check(_ids(bank["ids"], "gradient_space.ids") == ordered, "gradient_space: ordered IDs mismatch")
        f = bank["features"]
        _check(isinstance(f, list) and bool(f) and isinstance(f[0], list) and len(f[0]) > 0,
               "gradient_space: empty dimensions")
        spaces[bank["name"]] = (bank, _matrix(f, len(ordered), len(f[0]), "gradient_space.features"))

    feedback = _block(snapshot["feedback"], "feedback", **params)
    _check(isinstance(feedback, list), "feedback: expected list")
    evidence = set()
    feedback_freeze = _time(snapshot["feedback"]["source"]["frozen_at"], "feedback.frozen_at")
    for item in feedback:
        _keys(item, ("evidence_id", "id", "observed_at", "acquisition_round", "error_v1_ml", "error_v2_ml"), name="feedback.item")
        _check(isinstance(item["evidence_id"], str) and bool(item["evidence_id"]) and item["evidence_id"] not in evidence,
               "feedback: duplicate or missing evidence identity")
        evidence.add(item["evidence_id"])
        _check(item["id"] in L, "feedback: evidence outside frozen L_t")
        _check(_time(item["observed_at"], "feedback.observed_at") < cutoff
               and _time(item["observed_at"], "feedback.observed_at") <= feedback_freeze,
               "feedback: observation not available before selection/freeze")
        _check(type(item["acquisition_round"]) is int and -1 <= item["acquisition_round"] < state["round"],
               "feedback: current or future acquisition round")
        _check((item["error_v1_ml"] is None) == (item["error_v2_ml"] is None), "feedback: partial residual")
        if item["error_v1_ml"] is not None:
            _number(item["error_v1_ml"], "feedback.error_v1_ml")
            _number(item["error_v2_ml"], "feedback.error_v2_ml")
    return state, state_hash, selected, metadata, q50, width, crossing, spaces, feedback


def _sqdist(a, b):
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.maximum(np.square(a).sum(1)[:, None] + np.square(b).sum(1)[None, :] - 2 * a @ b.T, 0.)
    _check(bool(np.isfinite(value).all()), "gradient geometry: numerical overflow")
    return value


def _ratio(a, b):
    return float(a / b) if b else None


def _mean(a):
    return float(a.mean()) if len(a) else None


def _ranks(values):
    """Average tie percentile, identical to pandas rank(method='average', pct=True)."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse] / len(values)


def _effective_rank(features):
    x = features - features.mean(0, keepdims=True)
    gram = x @ x.T if len(x) <= x.shape[1] else x.T @ x
    eigen = np.maximum(np.linalg.eigvalsh(gram), 0)
    if not eigen.max() > 0:
        return 0., 0., 0.
    eigen = eigen[eigen > eigen.max() * 1e-12]
    singular = np.sqrt(eigen)
    p = singular / singular.sum()
    return float(np.exp(-(p * np.log(p)).sum())), float(eigen.sum() ** 2 / np.square(eigen).sum()), float(eigen.max() / eigen.sum())


def _geometry(features, n, positions, ids):
    """Metric definitions align with the 20261029 Free/Hybrid frozen audits."""
    pool = features[n:]
    p = np.array(positions, dtype=int)
    batch = pool[p]
    before_sq = _sqdist(pool, features[:n]).min(1)
    before = np.sqrt(before_sq)
    percentile = _ranks(before)
    pool_batch_sq = _sqdist(pool, batch)
    for k, position in enumerate(p):
        pool_batch_sq[position, k] = 0.
    after_sq = np.minimum(before_sq, pool_batch_sq.min(1))
    after = np.sqrt(after_sq)
    remaining = np.ones(len(pool), dtype=bool)
    remaining[p] = False
    # Direct pair differences preserve the tiny matched-condition directions.
    pairwise = np.linalg.norm(batch[:, None, :] - batch[None, :, :], axis=2)
    np.fill_diagonal(pairwise, np.inf)
    nn = pairwise.min(1)
    mean, sqmean = float(before.mean()), float(before_sq.mean())
    rms = float(np.sqrt(np.square(features - features.mean(0, keepdims=True)).sum(1).mean()))
    er, pr, top = _effective_rank(features)
    ber, bpr, btop = _effective_rank(batch)
    result = {
        "selected_to_L_distance_mean": float(before[p].mean()),
        "selected_to_L_distance_median": float(np.median(before[p])),
        "selected_to_L_distance_p10": float(np.quantile(before[p], .1)),
        "selected_to_L_distance_p90": float(np.quantile(before[p], .9)),
        "selected_distance_percentile_mean": float(percentile[p].mean()),
        "selected_distance_percentile_median": float(np.median(percentile[p])),
        "selected_distance_percentile_p10": float(np.quantile(percentile[p], .1)),
        "selected_distance_percentile_p90": float(np.quantile(percentile[p], .9)),
        "selected_top_decile_fraction": float((percentile[p] >= .9).mean()),
        "pool_preselection_coverage_distance_mean": mean,
        "bank_centered_feature_RMS": rms,
        "pool_coverage_mean_to_bank_RMS": _ratio(mean, rms),
        "coverage_distance_relative_drop": _ratio(mean - after.mean(), mean),
        "coverage_squared_distance_relative_drop": _ratio(sqmean - after_sq.mean(), sqmean),
        "coverage_gain_to_bank_RMS": _ratio(mean - after.mean(), rms),
        "remaining_U_coverage_distance_relative_drop": _ratio(before[remaining].mean() - after[remaining].mean(), before[remaining].mean()) if remaining.any() else None,
        "remaining_U_coverage_gain_to_bank_RMS": _ratio(before[remaining].mean() - after[remaining].mean(), rms) if remaining.any() else None,
        "batch_nn_distance_mean": float(nn.mean()),
        "batch_nn_to_pool_coverage_ratio": _ratio(nn.mean(), mean),
        "batch_nn_to_selected_L_distance_ratio": _ratio(nn.mean(), before[p].mean()),
        "batch_nn_closer_than_L_fraction": float((nn < before[p]).mean()),
        "batch_nn_less_than_half_L_fraction": float((nn < .5 * before[p]).mean()),
        "batch_exact_feature_duplicate_fraction": float(1 - len(np.unique(batch, axis=0)) / len(batch)),
        "centered_bank_effective_rank": er,
        "centered_bank_covariance_participation_rank": pr,
        "centered_bank_top_covariance_fraction": top,
        "centered_batch_effective_rank": ber,
        "centered_batch_covariance_participation_rank": bpr,
        "centered_batch_top_covariance_fraction": btop,
    }
    points = []
    current_sq = before_sq.copy()
    for k, (identifier, position) in enumerate(zip(ids, p)):
        next_sq = np.minimum(current_sq, pool_batch_sq[:, k])
        current, following = np.sqrt(current_sq), np.sqrt(next_sq)
        points.append({
            "id": identifier,
            "to_L_distance": float(before[position]),
            "distance_percentile": float(percentile[position]),
            "batch_nn_distance": float(nn[k]),
            "batch_nn_ids": [ids[j] for j in np.flatnonzero(pairwise[k] == nn[k])],
            "batch_nn_less_than_half_L": bool(nn[k] < .5 * before[position]),
            "exact_feature_peer_ids": [ids[j] for j in np.flatnonzero(pairwise[k] == 0)],
            "plan_order": k,
            "prefix_marginal_mean_distance_drop_full_U": float((current - following).mean()),
            "prefix_marginal_mean_distance_drop_common_remaining_U": _mean((current - following)[remaining]),
            "prefix_marginal_common_remaining_U_drop_to_bank_RMS": _ratio((current - following)[remaining].mean(), rms) if remaining.any() else None,
        })
        current_sq = next_sq
    return result, points, pairwise


def critique_frozen_plan(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Validate frozen inputs and return descriptive diagnostics, without action.

    Raises FrozenInputError before numerical diagnosis for invalid input. The
    input consists solely of explicit JSON-like blocks; no paths are followed.
    """
    state, state_hash, selected, meta, q50, width, crossing, spaces, feedback = _validate(snapshot)
    L, U = state["L_t_ids"], state["U_t_ids"]
    ordered = L + U
    index = dict(zip(ordered, range(len(ordered))))
    positions = [U.index(i) for i in selected]
    result = {
        "schema_version": "numerical-critic-result-v1",
        "state_sha256": state_hash,
        "seed": state["seed"], "round": state["round"], "synthetic": state["synthetic"],
        "selected_ids": list(selected),
        "scope": "static_descriptive_diagnostics_only",
        "source_receipts": {key: snapshot[key]["source"] for key in ("state", "metadata", "predictions", "plan", "feedback")},
        "gradient_source_receipts": [b["source"] for b in snapshot["gradient_spaces"]],
        "geometry": {}, "candidate_diagnostics": [], "candidate_pair_relations": [],
        "scientific_controls": [], "review_items": [],
        "prediction_diagnostics": {
            "quantiles_available": crossing is not None,
            "pool_quantile_crossing_count_v1_v2": crossing[len(L):].sum(0).tolist() if crossing is not None else None,
            "pool_negative_quantile_width_count_v1_v2": (width[len(L):] < 0).sum(0).tolist() if width is not None else None,
            "interpretation": "model_diagnostic_only_not_plan_illegality",
        },
        "limitations": [
            "No weighted utility score, automatic replacement, acceptance decision, or predicted learning gain.",
            "Low distance and same-molecule concentration do not establish ineffective experiments.",
            "Matched conditions establish a design structure, not a causal outcome or the LLM's intent.",
            "Checks establish consistency of supplied freeze receipts; source authenticity and train-pool membership require a trusted upstream freeze.",
        ],
    }
    point_geometry = {}
    pair_geometry = {}
    for name, (bank, features) in spaces.items():
        metrics, points, pairwise = _geometry(features, len(L), positions, selected)
        result["geometry"][name] = {"kind": bank["kind"], "dimension": features.shape[1],
                                    "preprocessing": bank["preprocessing"], "metrics": metrics}
        point_geometry[name] = {p["id"]: p for p in points}
        pair_geometry[name] = pairwise

    molecule_counts = Counter(meta[i]["molecule"] for i in selected)
    selected_set = set(selected)
    all_controls = {i: [] for i in selected}
    prior_repeats = {i: [] for i in selected}
    batch_repeats = {i: [] for i in selected}
    for identifier in selected:
        row = meta[identifier]
        for peer in ordered:
            if peer == identifier or (peer in selected_set and selected.index(peer) < selected.index(identifier)):
                continue
            if peer not in L and peer not in selected_set:
                continue
            if meta[peer]["molecule"] != row["molecule"]:
                continue
            changed = [axis for axis in CONDITIONS if row["conditions"][axis] != meta[peer]["conditions"][axis]]
            scope = "L_t" if peer in L else "batch"
            if not changed:
                (prior_repeats if scope == "L_t" else batch_repeats)[identifier].append(peer)
                if scope == "batch":
                    batch_repeats[peer].append(identifier)
            if len(changed) == 1:
                control = {"id": identifier, "peer_id": peer, "peer_scope": scope,
                           "varied_condition": changed[0], "fixed_conditions": [c for c in CONDITIONS if c != changed[0]],
                           "q50_delta_v1_ml": float(q50[index[identifier], 0] - q50[index[peer], 0]),
                           "q50_delta_v2_ml": float(q50[index[identifier], 1] - q50[index[peer], 1]),
                           "gradient_distances": {name: float(np.linalg.norm(features[index[identifier]] - features[index[peer]]))
                                                  for name, (_, features) in spaces.items()},
                           "interpretation": "strict_single_axis_design_structure_only"}
                result["scientific_controls"].append(control)
                all_controls[identifier].append(control)
                if scope == "batch":
                    all_controls[peer].append(control)

    for i, identifier in enumerate(selected):
        for j in range(i + 1, len(selected)):
            peer = selected[j]
            same = meta[identifier]["molecule"] == meta[peer]["molecule"]
            result["candidate_pair_relations"].append({
                "id": identifier, "peer_id": peer, "same_molecule": same,
                "changed_conditions": [c for c in CONDITIONS if meta[identifier]["conditions"][c] != meta[peer]["conditions"][c]],
                "exact_molecule_condition_repeat": same and meta[identifier]["conditions"] == meta[peer]["conditions"],
                "gradient_distances": {name: float(pair_geometry[name][i, j]) for name in spaces},
            })

    for identifier in selected:
        controls = all_controls[identifier]
        prior_axes = sorted({c["varied_condition"] for c in controls if c["peer_scope"] == "L_t"})
        batch_axes = sorted({c["varied_condition"] for c in controls if c["peer_scope"] == "batch"})
        point = {"id": identifier, "molecule": meta[identifier]["molecule"],
                 "same_molecule_rows_in_batch": molecule_counts[meta[identifier]["molecule"]],
                 "exact_condition_repeat_prior_ids": prior_repeats[identifier],
                 "exact_condition_repeat_batch_ids": batch_repeats[identifier],
                 "strict_matched_prior_axes": prior_axes, "strict_matched_batch_axes": batch_axes,
                 "new_condition_curve_extension_prior_axes": prior_axes if not prior_repeats[identifier] else [],
                 "q50_v1_v2_ml": q50[index[identifier]].tolist(),
                 "q90_minus_q10_v1_v2_ml": width[index[identifier]].tolist() if width is not None else None,
                 "quantile_crossing_v1_v2": crossing[index[identifier]].tolist() if crossing is not None else None,
                 "geometry": {name: point_geometry[name][identifier] for name in spaces}}
        result["candidate_diagnostics"].append(point)
        reasons = []
        if crossing is not None and crossing[index[identifier]].any():
            reasons.append("frozen_predictor_quantile_crossing_model_diagnostic")
        if prior_repeats[identifier]:
            reasons.append("previously_observed_molecule_condition_tuple_replicate")
        if batch_repeats[identifier]:
            reasons.append("same_batch_molecule_condition_tuple_replicate")
        for name in spaces:
            g = point_geometry[name][identifier]
            if g["exact_feature_peer_ids"]:
                reasons.append(f"{name}:exact_feature_peer")
            if g["batch_nn_less_than_half_L"]:
                reasons.append(f"{name}:batch_nearest_neighbor_less_than_half_distance_to_L")
        if reasons:
            result["review_items"].append({"id": identifier, "reasons": reasons,
                                           "strict_control_relations": controls,
                                           "action": "human_or_future_scientific_review_only"})

    residuals = [r for r in feedback if r["error_v1_ml"] is not None]
    # Same fixed top8 definition as the historical mechanism audit, not a cutoff
    # selected using validation error. Multiple receipts for one sample collapse.
    by_id = {}
    for row in residuals:
        old = by_id.get(row["id"])
        if old is None or _time(row["observed_at"], "observed_at") > _time(old["observed_at"], "observed_at"):
            by_id[row["id"]] = row
    high = sorted(by_id.values(), key=lambda r: (-max(abs(r["error_v1_ml"]), abs(r["error_v2_ml"])), r["id"]))[:8]
    high_molecules = {meta[r["id"]]["molecule"] for r in high}
    sel_fraction = sum(meta[i]["molecule"] in high_molecules for i in selected) / 32 if high else None
    pool_fraction = sum(meta[i]["molecule"] in high_molecules for i in U) / len(U) if high else None
    result["preselection_feedback_context"] = {
        "available_evidence_count": len(feedback), "historical_residual_sample_count": len(by_id),
        "high8_evidence_ids": [r["evidence_id"] for r in high],
        "high8_same_molecule_fraction": sel_fraction,
        "high8_same_molecule_pool_fraction": pool_fraction,
        "high8_same_molecule_enrichment": sel_fraction - pool_fraction if high else None,
    }
    result["condition_summary"] = {
        "unique_molecules": len(molecule_counts),
        "same_molecule_pairs": sum(n * (n - 1) // 2 for n in molecule_counts.values()),
        "max_rows_per_molecule": max(molecule_counts.values()),
        "exact_condition_repeat_prior_fraction": sum(bool(prior_repeats[i]) for i in selected) / 32,
        "within_batch_exact_condition_repeat_pairs": sum(len(v) for v in batch_repeats.values()) // 2,
        "strict_single_axis_control_pair_count": len(result["scientific_controls"]),
    }
    return result
