"""Read-only same-state geometry of the frozen first proposal and final plan.

Only identity/split metadata, frozen selection artifacts and existing gradient
banks are read. No predictor, gradient extractor, label store or model is loaded.
Run from the repository root with the chromatography Python.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AL = ROOT / "studies/active_learning"
REV = AL / "qgeognn_v2_row_llm_scientist_v2/revisions"
HASHES = {}


def rel(path):
    return str(Path(path).relative_to(ROOT))


def sha(path):
    key = rel(path)
    if key not in HASHES:
        with Path(path).open("rb") as handle:
            HASHES[key] = hashlib.file_digest(handle, "sha256").hexdigest()
    return HASHES[key]


def read(path):
    sha(path)
    return json.loads(Path(path).read_text())


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def strict_answer(text):
    def pairs(items):
        result = {}
        for key, value in items:
            assert key not in result, "ambiguous duplicate JSON key"
            result[key] = value
        return result
    def forbidden(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=forbidden)


def prior_hashes():
    result = {}
    for name in ("free_lcmd_mechanism_diagnostics_20261029/runtime_source_hashes.json",
                 "hybrid_loading_cap_audit_20261029/source_hashes.json"):
        for path, digest in read(AL / name).items():
            if path in result:
                assert result[path] == digest, f"Earlier audits disagree: {path}"
            result[path] = digest
    # NoCW's frozen initial-state protocol records all five split-file hashes.
    # Read only that metadata; do not open the prediction/label paths it records.
    protocol = read(AL / "qgeognn_v2_row_llm_nocw_ablation/revisions/five_seed_nocw_to1005_20261029/protocol.json")
    for initial in protocol["initial_states"]:
        for path, digest in initial["sources"].items():
            if "/splits/row_seed_" in path and path.endswith(".csv"):
                if path in result:
                    assert result[path] == digest
                result[path] = digest
    return result


def legacy_metrics():
    path = AL / "free_lcmd_mechanism_diagnostics_20261029/geometry_analysis.py"
    sha(path)
    functions = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in ("squared_distances", "effective_rank")]
    assert len(functions) == 2
    namespace = {"np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["squared_distances"], namespace["effective_rank"]


def split_identity(seed, previous):
    path = AL / f"qgeognn_v2_row_sequential_b32/splits/row_seed_{seed}.csv"
    assert rel(path) in previous and sha(path) == previous[rel(path)], "split differs from earlier audit"
    # The split file has no target columns. Only these three identity/role fields
    # are used; test/validation identities are excluded from the returned map.
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        assert set(rows.fieldnames) == {"sample_id", "canonical_smiles", "canonical_index", "role", "outer_seed", "study"}
        mapping = {row["sample_id"]: int(row["canonical_index"]) for row in rows if row["role"] in ("l0", "u0")}
    assert len(mapping) == 3330 and len(set(mapping.values())) == len(mapping)
    return mapping


def bank_for(directory, contract, expected, previous):
    seed, method, round_ = contract["seed"], contract["method"], contract["round"]
    own = Path(str(directory).replace("/selections/", "/runtime/")) / "banks.json"
    # Finite exact seed/method/round lookup covers banks inherited across revision
    # boundaries; never substitute another method or another round.
    paths = [own] + sorted(REV.glob(f"*/runtime/seed_{seed}/{method}/round_{round_:02d}/banks.json"))
    paths = list(dict.fromkeys(paths))
    inspected = []
    for path in paths:
        if not path.is_file():
            continue
        meta = read(path)
        checkpoint_match = meta["contract"].get("checkpoint") == contract["checkpoint_sha256"]
        order_match = meta["contract"].get("indices") == expected
        inspected.append({"metadata_path": rel(path), "checkpoint_matches": checkpoint_match,
                          "ordered_train_pool_indices_match": order_match})
        if not checkpoint_match or not order_match:
            continue
        npz = path.with_suffix(".npz")
        if not npz.is_file():
            continue
        assert sha(npz) == meta["sha256"], "bank bytes differ from own metadata"
        for p in (path, npz):
            if rel(p) in previous:
                assert sha(p) == previous[rel(p)], "bank differs from earlier audit"
        return npz, meta, inspected, {
            "bank_path": rel(npz), "bank_metadata_path": rel(path),
            "bank_sha256": sha(npz), "bank_metadata_sha256": sha(path),
            "bank_prior_audit_hash_available": rel(npz) in previous,
            "bank_metadata_prior_audit_hash_available": rel(path) in previous,
            "bank_same_selection_owner": path == own,
            "bank_checkpoint_verified": True, "bank_ordered_L_U_verified": True,
            "bank_contract_protocol_sha256": meta["contract"].get("protocol"),
        }
    return None, None, inspected, {"bank_missing_reason": "No same-method, same-round frozen bank with matching checkpoint and exact ordered L+U train-pool indices; no cross-method substitution or recomputation."}


def geometry(features, labeled_count, positions, squared_distances, effective_rank):
    batch = np.asarray(features[labeled_count:][positions], dtype=np.float64)
    x = np.asarray(features, dtype=np.float64)
    n = len(batch)
    assert x.ndim == 2 and np.isfinite(x).all() and n > 0
    # Only the selected subset is compared against L: avoid unnecessary full-U
    # coverage computations. Same original norm-dot definition and exact zeros.
    dL = np.sqrt(squared_distances(batch, x[:labeled_count]).min(1))
    rms = float(np.sqrt(np.square(x - x.mean(0, keepdims=True)).sum(1).mean()))
    er, pr, top = effective_rank(batch)
    result = {"selected_to_L_distance_mean": float(dL.mean()),
              "selected_to_L_distance_median": float(np.median(dL)),
              "selected_to_L_mean_to_bank_RMS": float(dL.mean()/rms) if rms else None,
              "centered_batch_effective_rank": er,
              "centered_batch_covariance_participation_rank": pr,
              "centered_batch_top_covariance_fraction": top,
              "batch_exact_feature_duplicate_fraction": float(1-len(np.unique(batch, axis=0))/n)}
    if n > 1:
        bb = squared_distances(batch, batch)
        np.fill_diagonal(bb, np.inf)
        nn = np.sqrt(bb.min(1))
        result.update({"batch_nn_distance_mean": float(nn.mean()),
                       "batch_nn_mean_to_bank_RMS": float(nn.mean()/rms) if rms else None,
                       "batch_nn_to_selected_L_distance_ratio": float(nn.mean()/dL.mean()) if dL.mean() else None,
                       "batch_nn_closer_than_L_fraction": float((nn < dL).mean()),
                       "batch_nn_less_than_half_L_fraction": float((nn < .5*dL).mean()),
                       "batch_nn_closer_than_L_count": int((nn < dL).sum()),
                       "batch_nn_less_than_half_L_count": int((nn < .5*dL).sum())})
    else:
        result.update({key: None for key in ("batch_nn_distance_mean", "batch_nn_mean_to_bank_RMS",
                       "batch_nn_to_selected_L_distance_ratio", "batch_nn_closer_than_L_fraction",
                       "batch_nn_less_than_half_L_fraction", "batch_nn_closer_than_L_count",
                       "batch_nn_less_than_half_L_count")})
    return result


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method_group"], row["view"], row["space"])].append(row)
    result = []
    for (method, view, space), group in sorted(grouped.items()):
        available = [r for r in group if r["comparison_available"]]
        entry = {"method_group": method, "view": view, "space": space, "round_denominator": len(group),
                 "available_comparisons": len(available), "missing_comparisons": len(group)-len(available),
                 "equal_subset_size_comparisons": sum(r["same_selected_count"] for r in available),
                 "first_unique_legal_count_mean": statistics.mean(r["first_count"] for r in available) if available else None,
                 "final_count_mean": statistics.mean(r["final_count"] for r in available) if available else None,
                 "metric_changes": {}, "equal_count_metric_changes": {}, "per_seed": []}
        metrics = sorted({key[6:] for row in group for key in row if key.startswith("delta_")})
        for name in metrics:
            values = [r["delta_"+name] for r in available if r.get("delta_"+name) is not None]
            entry["metric_changes"][name] = {"available_rounds": len(values),
                "increased": sum(v > 1e-12 for v in values), "decreased": sum(v < -1e-12 for v in values),
                "unchanged_at_1e_minus12": sum(abs(v) <= 1e-12 for v in values),
                "mean_delta": statistics.mean(values) if values else None,
                "median_delta": statistics.median(values) if values else None}
            equal = [r["delta_"+name] for r in available if r["same_selected_count"] and r.get("delta_"+name) is not None]
            entry["equal_count_metric_changes"][name] = {"available_rounds": len(equal),
                "increased": sum(v > 1e-12 for v in equal), "decreased": sum(v < -1e-12 for v in equal),
                "unchanged_at_1e_minus12": sum(abs(v) <= 1e-12 for v in equal),
                "median_delta": statistics.median(equal) if equal else None}
        for seed in sorted({r["seed"] for r in group}):
            sub = [r for r in available if r["seed"] == seed]
            entry["per_seed"].append({"seed": seed, "available_rounds": len(sub),
                "mean_delta_batch_nn_to_selected_L_distance_ratio": statistics.mean(r["delta_batch_nn_to_selected_L_distance_ratio"] for r in sub) if sub else None,
                "mean_delta_centered_batch_effective_rank": statistics.mean(r["delta_centered_batch_effective_rank"] for r in sub) if sub else None})
        result.append(entry)
    return result


def reference_parity(rows):
    """Verify final-plan metrics against already published geometry-only CSVs."""
    references = {}
    for study, group in (("free_lcmd_mechanism_diagnostics_20261029", "Free"),
                         ("hybrid_loading_cap_audit_20261029", "Hybrid")):
        path = AL / study / "geometry_profiles.csv"
        sha(path)
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if group == "Free" and row["method"] != "free_llm32_scientist_v2":
                    continue
                source = row.get("source")
                view = "pending_plus_llm" if source == "hybrid_all32" else "llm_component"
                if group == "Hybrid" and source not in ("hybrid_llm16", "hybrid_all32"):
                    continue
                references[(group, int(row["seed"]), int(row["round"]), view)] = row
    names = ("selected_to_L_distance_mean", "selected_to_L_distance_median", "centered_batch_effective_rank",
             "centered_batch_covariance_participation_rank", "centered_batch_top_covariance_fraction",
             "batch_exact_feature_duplicate_fraction", "batch_nn_distance_mean",
             "batch_nn_to_selected_L_distance_ratio", "batch_nn_closer_than_L_fraction",
             "batch_nn_less_than_half_L_fraction")
    pairs, checks = 0, 0
    max_absolute_difference = 0.
    for row in rows:
        key = (row["method_group"], row["seed"], row["round"], row["view"])
        if not row["comparison_available"] or key not in references:
            continue
        pairs += 1
        for name in names:
            expected = references[key][row["space"]+"_"+name]
            actual = row["final_"+name]
            if not expected:
                assert actual is None
            else:
                expected = float(expected)
                assert np.isclose(actual, expected, rtol=1e-10, atol=1e-10), (key, row["space"], name)
                max_absolute_difference = max(max_absolute_difference, abs(actual-expected))
            checks += 1
    return {"final_profile_space_rows_compared": pairs, "metric_value_checks": checks,
            "absolute_and_relative_tolerance": 1e-10, "maximum_absolute_difference": max_absolute_difference,
            "all_passed": True}


def main():
    started = time.perf_counter()
    sha(Path(__file__))
    repairs_path = HERE / "repair_trajectory_analysis.csv"
    sha(repairs_path)
    with repairs_path.open(newline="") as handle:
        repairs = list(csv.DictReader(handle))
    snapshot = read(HERE / "history_snapshot_manifest.json")
    previous = prior_hashes()
    square, rank = legacy_metrics()
    known = geometry(np.array([[0., 0.], [4., 0.], [1., 0.], [3., 0.]]), 2, [0, 1], square, rank)
    assert known["selected_to_L_distance_mean"] == 1. and known["batch_nn_distance_mean"] == 2.
    assert known["batch_nn_to_selected_L_distance_ratio"] == 2. and known["centered_batch_effective_rank"] == 1.
    assert known["batch_nn_less_than_half_L_fraction"] == 0.
    zero = geometry(np.zeros((4, 2)), 2, [0, 1], square, rank)
    assert zero["batch_nn_to_selected_L_distance_ratio"] is None and zero["centered_batch_effective_rank"] == 0.
    assert geometry(np.zeros((3, 2)), 2, [0], square, rank)["batch_nn_distance_mean"] is None
    splits = {seed: split_identity(seed, previous) for seed in sorted({int(r["seed"]) for r in repairs})}
    rows, bank_audits = [], []
    for row in repairs:
        seed, round_ = int(row["seed"]), int(row["round"])
        directory = ROOT / Path(row["final_selection_path"]).parent
        paths = {name: directory/name for name in ("contract.json", "batch_freeze.json", "selection.json")}
        paths["first_response"] = ROOT / row["first_response_path"]
        for path in paths.values():
            assert sha(path) == snapshot["files"][rel(path)]["sha256"], "audit snapshot drift"
        contract, frozen, final = (read(paths[name]) for name in ("contract.json", "batch_freeze.json", "selection.json"))
        assert contract["seed"] == seed and contract["round"] == round_ and contract["method"] == row["method"]
        L, U = contract["L_t_ids"], contract["U_t_ids"]
        assert len(set(L+U)) == len(L+U) and set(L+U) == set(splits[seed])
        expected = [splits[seed][i] for i in L+U]
        final_ids = [c["id"] for c in final["choices"]]
        first_ids = json.loads(row["initial_valid_ids"]) if row["comparison_analyzable"] == "True" else None
        assert first_ids is None or (len(set(first_ids)) == len(first_ids) and set(first_ids) <= set(U))
        pending = frozen["pending_ids"]
        if first_ids is not None:
            raw = strict_answer(read(paths["first_response"])["answer"])
            assert isinstance(raw, dict) and isinstance(raw["choices"], list)
            assert all(isinstance(c, dict) and isinstance(c.get("id"), str) and isinstance(c.get("reason"), str)
                       and "hypothesis_id" in c for c in raw["choices"])
            allowed = set(U) - set(pending)
            independently_legal = list(dict.fromkeys(c["id"] for c in raw["choices"] if c["id"] in allowed))
            assert independently_legal == first_ids, "derived first subset differs from frozen complete response"
        assert len(set(final_ids)) == len(final_ids) and set(final_ids) <= set(U)
        assert set(pending) | set(final_ids) == set(frozen["batch_ids"]) and not set(pending) & set(final_ids)
        bank, metadata, inspected, audit = bank_for(directory, contract, expected, previous)
        audit.update({"method": row["method"], "method_group": row["method_group"], "seed": seed, "round": round_,
                      "selection_directory": rel(directory), "checkpoint_sha256": contract["checkpoint_sha256"],
                      "L_ordered_ids_sha256": stable_hash(L), "U_ordered_ids_sha256": stable_hash(U),
                      "candidate_source_metadata_checked": inspected})
        bank_audits.append(audit)
        views = {"llm_component": (first_ids, final_ids)}
        if pending:
            views["pending_plus_llm"] = (pending+first_ids if first_ids is not None else None, pending+final_ids)
        index = {i: j for j, i in enumerate(U)}
        with np.load(bank, allow_pickle=False) if bank is not None else _empty_bank() as loaded:
            for space in ("raw", "cw"):
                features = loaded[space] if space in loaded else None
                if features is not None:
                    assert features.shape == (len(L)+len(U), 512) and np.isfinite(features).all()
                for view, (first, last) in views.items():
                    available = features is not None and first is not None and len(first) > 0
                    out = {"method": row["method"], "method_group": row["method_group"], "seed": seed, "round": round_,
                           "space": space, "view": view, "comparison_available": available,
                           "first_count": len(first) if first is not None else None, "final_count": len(last),
                           "same_selected_count": len(first) == len(last) if first is not None else None,
                           "same_selected_ID_set": set(first) == set(last) if first is not None else None,
                           "first_selected_ids_sha256": stable_hash(first) if first is not None else None,
                           "final_selected_ids_sha256": stable_hash(last),
                           "L_count": len(L), "U_count": len(U), "pending_count": len(pending),
                           "subset_policy": "unique legal first-proposal IDs only; no repair of illegal identities",
                           "comparison_limit": "same frozen state; subset size/composition differ; descriptive, not causal or learning utility",
                           "bank_path": audit.get("bank_path"), "bank_sha256": audit.get("bank_sha256"),
                           "checkpoint_sha256": contract["checkpoint_sha256"],
                           "missing_reason": "" if available else audit.get("bank_missing_reason", "unparseable/empty first legal subset or missing space")}
                    if available:
                        a = geometry(features, len(L), [index[i] for i in first], square, rank)
                        b = geometry(features, len(L), [index[i] for i in last], square, rank)
                        for name in a:
                            out["first_"+name], out["final_"+name] = a[name], b[name]
                            out["delta_"+name] = b[name]-a[name] if a[name] is not None and b[name] is not None else None
                    rows.append(out)
        if len(bank_audits) % 20 == 0:
            print(f"Compared {len(bank_audits)}/{len(repairs)} frozen correction rounds", flush=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (HERE/"repair_geometry_profiles.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    parity = reference_parity(rows)
    summary = {"schema_version": "repair_geometry_v1", "snapshot_captured_at": snapshot["captured_at"],
               "correction_rounds": len(repairs), "profile_rows": len(rows),
               "available_bank_rounds": sum("bank_path" in a for a in bank_audits),
               "missing_bank_rounds": sum("bank_path" not in a for a in bank_audits),
               "method_view_space_summary": summarize(rows), "bank_audits": bank_audits,
               "verification": {"known_1d_distance_and_ratio": True, "known_rank_one": True,
                                "zero_denominator_returns_null": True, "singleton_neighbor_returns_null": True,
                                "every_first_legal_subset_rederived_from_frozen_complete_answer": True,
                                "all_split_bank_state_and_artifact_checks_passed": True,
                                "existing_final_profile_metric_parity": parity},
               "elapsed_seconds": time.perf_counter()-started, "source_sha256": HASHES,
               "new_training_runs": 0, "new_llm_calls": 0, "new_gradient_extractions": 0,
               "prediction_table_reads": 0, "target_table_reads": 0, "test_truth_access_count": 0,
               "interpretation": "Same-state description only. Unequal legal subset sizes affect nearest-neighbor, fractions and rank. No p-values, causal repair effect, or real learning benefit is estimated."}
    (HERE/"repair_geometry_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: summary[key] for key in ("correction_rounds", "profile_rows", "available_bank_rounds", "missing_bank_rounds", "elapsed_seconds")}))


class _empty_bank(dict):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


if __name__ == "__main__":
    main()
