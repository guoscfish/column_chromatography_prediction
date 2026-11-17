"""Read-only audit of legacy metadata and feature identities; never read truths.

The script reads aggregate probability metrics, fit diagnostics, split identities,
and binary file hashes. It does not deserialize checkpoint weights or responses.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = HERE.parent / "representation_gp_benchmark_20261104"
READS = {}


def sha(path):
    """Hash twice and reject files that change during this read-only audit."""
    path = Path(path)
    def one():
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    first = one()
    if first != one():
        raise ValueError(f"Source changed while auditing: {path}")
    READS[str(path.relative_to(ROOT))] = first
    return first


def read_json(path):
    sha(path)
    return json.loads(Path(path).read_text())


def read_csv(path):
    sha(path)
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    return {"minimum": float(a.min()), "median": float(np.median(a)),
            "mean": float(a.mean()), "maximum": float(a.max())}


def initialization_key(value):
    return "" if value is None or value == "" else str(int(float(value)))


def main():
    for report in (
        ROOT / "docs/NEXT_STAGE_DECISION.md",
        ROOT / "reports/free_lcmd_five_seeds_to1485_20261026/REPORT.md",
        HERE.parent / "free_cw_sol61_xhigh_five_to1485_20261101/reports/REPORT.md",
        HERE.parent / "nocw_free_selection_diagnostics_20261101/REPORT.md",
        HERE.parent / "qgeognn_v2_row_sequential_b32/FINAL_REPORT.md",
        HERE.parent / "qgeognn_v2_row_kernel_ivr_b32/FINAL_REPORT.md",
        HERE.parent / "qgeognn_v2_ivr_mechanism_audit/FINAL_REPORT.md",
        HERE.parent / "gp_guided_llm_stability_20261103/OFFLINE_REPORT.md",
        OLD / "GP_CALIBRATION_REPORT.md", OLD / "COMPARISON_REPORT.md", OLD / "AUDIT.md"):
        sha(report)
    fits = read_csv(OLD / "summary/fits.csv")
    metrics = read_csv(OLD / "summary/metrics.csv")
    endpoint_metrics = read_csv(OLD / "summary/endpoint_budget_metrics.csv")
    sources = read_json(OLD / "representation_source_audit.json")
    manifest = read_json(OLD / "artifact_manifest.json")
    configuration = read_json(OLD / "config.json")
    checks, groups, parameter_boundaries = [], {}, []
    summary_keys = {(int(r["seed"]), int(r["budget"]), r["representation"], r["kernel"],
                     initialization_key(r["initialization"]), r["endpoint"]) for r in fits}
    if len(summary_keys) != len(fits):
        raise ValueError("Duplicate logical fit in legacy summary")
    for source in sources["states"]:
        data_seed, budget = source["seed"], source["budget"]
        name = f"seed_{data_seed}_L{budget}"
        receipt = read_json(OLD / "results" / name / "complete.json")
        raw_fits = read_json(OLD / "results" / name / "fits.json")
        cache_receipt = read_json(OLD / "cache" / (name + ".json"))
        cache_path = OLD / "cache" / (name + ".npz")
        identity_files = {key: sha(ROOT / source[key]) == source[key + "_sha256"]
                          for key in ("checkpoint", "raw_cache", "partition")}
        cache_match = sha(cache_path) == cache_receipt["cache_sha256"]
        state_match = cache_receipt["state"] == source
        # Only identity indices are loaded; train_y/validation_y are never accessed.
        with np.load(cache_path, allow_pickle=False) as archive:
            indices = archive["indices"]
            indices_match = hashlib.sha256(indices.tobytes()).hexdigest() == cache_receipt["indices_sha256"]
            array_names = list(archive.files)
        expected_rows = budget + 256 + 416
        per_fit_match = True
        for raw in raw_fits:
            init = initialization_key(raw["initialization"])
            reported_kind = "projected" if raw["trainable_projection"] else raw["kernel"]
            key = (data_seed, budget, raw["representation"], reported_kind, init, raw["endpoint"])
            if key not in summary_keys:
                per_fit_match = False
                continue
            summary = next(r for r in fits if
                           (int(r["seed"]), int(r["budget"]), r["representation"], r["kernel"],
                            initialization_key(r["initialization"]), r["endpoint"]) == key)
            for field in ("signal_variance", "noise_variance", "negative_log_marginal_likelihood"):
                per_fit_match &= float(summary[field]) == float(raw[field])
        checks.append({"seed": data_seed, "budget": budget, "complete": receipt,
            "identity_file_hashes_match": identity_files, "prepared_cache_hash_matches": cache_match,
            "prepared_state_identity_matches": state_match, "indices_hash_matches": indices_match,
            "indices_count": len(indices), "expected_count": expected_rows,
            "cache_array_names": array_names, "responses_loaded": False,
            "raw_fit_count": len(raw_fits), "summary_numerics_match": bool(per_fit_match),
            "raw_fit_seed_is_model_seed": all(r["seed"] in (17, 29, 43) for r in raw_fits),
            "zero_test_truth_access_declared": receipt["test_truth_access_count"] == 0,
            "validation_used_for_checkpoint_selection": source["validation_previously_used_for_checkpoint_selection"]})
    llm = read_json(OLD / "cache/llm.json")
    llm_hash_matches = sha(OLD / "cache/llm.npz") == llm["cache_sha256"]
    for rep, kind in sorted({(r["representation"], r["kernel"]) for r in fits}):
        rows = [r for r in fits if r["representation"] == rep and r["kernel"] == kind]
        scores = [r for r in metrics if r["representation"] == rep and r["kernel"] == kind]
        log_signal = np.log([float(r["signal_variance"]) / float(r["target_scale"])**2 for r in rows])
        log_noise = np.log([float(r["noise_variance"]) / float(r["target_scale"])**2 for r in rows])
        boundary = {"signal_lower": int(np.sum(np.abs(log_signal + 9) < 1e-4)),
                    "signal_upper": int(np.sum(np.abs(log_signal - 9) < 1e-4)),
                    "noise_lower": int(np.sum(np.abs(log_noise + 9) < 1e-4)),
                    "noise_upper": int(np.sum(np.abs(log_noise - 9) < 1e-4))}
        if kind != "linear":
            log_ell = np.log([float(r["lengthscale"]) for r in rows])
            boundary.update(lengthscale_lower=int(np.sum(np.abs(log_ell + 5) < 1e-4)),
                            lengthscale_upper=int(np.sum(np.abs(log_ell - 5) < 1e-4)))
        groups[rep + "/" + kind] = {
            "fit_count": len(rows), "optimizer_success_count": sum(r["optimizer_success"] == "True" for r in rows),
            "boundary_hit_count": sum(r["boundary_hit"] == "True" for r in rows),
            "parameter_boundary_counts": boundary,
            "optimizer_messages": dict(Counter(r["optimizer_message"] for r in rows)),
            "noise_variance_ml2": describe([float(r["noise_variance"]) for r in rows]),
            "noise_variance_standardized": describe(np.exp(log_noise)),
            "signal_variance_ml2": describe([float(r["signal_variance"]) for r in rows]),
            "noise_signal_ratio": describe([float(r["noise_variance"]) / float(r["signal_variance"]) for r in rows]),
            "coverage_90": describe([float(r["coverage_90"]) for r in scores]),
            "coverage_95": describe([float(r["coverage_95"]) for r in scores]),
            "mean_interval_width_90_ml": describe([float(r["mean_interval_width_90"]) for r in scores]),
            "mean_interval_width_95_ml": describe([float(r["mean_interval_width_95"]) for r in scores]),
            "standardized_error_rms": describe([float(r["standardized_error_rms"]) for r in scores]),
            "nlpd": describe([float(r["nlpd"]) for r in scores]),
            "normalized_rmse_L0": describe([float(r["normalized_rmse_L0"]) for r in scores]),
            "total_training_seconds": sum(float(r["training_seconds"]) for r in rows),
            "training_nll_ml": describe([float(r["negative_log_marginal_likelihood"]) for r in rows]),
            "evaluation_role": sorted({r["evaluation_role"] for r in scores}),
            "aggregation_note": "Equal numbers per data seed/budget/endpoint/init; descriptive mean is balanced, not independent replicates."}
        for r in rows:
            opt = json.loads(r["optimizer"])
            if kind == "projected" and opt["iterations"] != 40:
                raise ValueError("Unexpected projected fit iteration budget")
    projection_pairs = {}
    for representation in ("llm_conditions", "gnn", "fusion"):
        frozen = {(r["seed"], r["budget"], r["endpoint"]): r for r in fits
                  if r["representation"] == representation and r["kernel"] == "matern52"}
        projected = [r for r in fits if r["representation"] == representation and r["kernel"] == "projected"]
        nll_differences, noise_ratios = [], []
        for row in projected:
            base = frozen[row["seed"], row["budget"], row["endpoint"]]
            nll_differences.append(float(row["negative_log_marginal_likelihood"]) - float(base["negative_log_marginal_likelihood"]))
            noise_ratios.append(float(row["noise_variance"]) / float(base["noise_variance"]))
        projection_pairs[representation] = {"paired_endpoint_initializations": len(projected),
            "projected_minus_frozen_training_nll": describe(nll_differences),
            "training_nll_improved_count": sum(v < 0 for v in nll_differences),
            "projected_over_frozen_noise_ratio": describe(noise_ratios),
            "noise_reduced_count": sum(v < 1 for v in noise_ratios),
            "interpretation": "Three projected initializations share each frozen comparator; counts are descriptive, not independent replicates."}
    selected_manifest_checks = {}
    for name in ("gp_models.py", "run_benchmark.py", "config.json", "summary/fits.csv", "summary/metrics.csv", "summary/summary.json"):
        value = sha(OLD / name)
        entry = manifest["files"].get(name)
        selected_manifest_checks[name] = {"actual_sha256": value, "listed": entry is not None,
                                         "manifest_matches": bool(entry and entry["sha256"] == value)}
    gradient = read_json(HERE / "legacy_gradient_results.json")
    engineering = {name: read_json(HERE / name) for name in
                   ("controlled_gradient_results.json", "data_permission_results.json")
                   if (HERE / name).exists()}
    output = {
        "date": "2026-11-06", "scope": "Read-only legacy code/aggregate metrics/identity audit plus synthetic gradient probes",
        "configuration": configuration, "complete_state_count": len(checks),
        "endpoint_fit_count": len(fits), "projected_fit_count": sum(r["kernel"] == "projected" for r in fits),
        "projected_all_at_40_iterations": True,
        "all_final_jitter_zero": all(float(r["numerical_jitter"]) == 0 for r in fits),
        "all_training_jitter_zero": all(json.loads(r["optimizer"]).get("max_training_jitter_standardized", 0) == 0 for r in fits),
        "all_identity_checks_passed": all(all(r["identity_file_hashes_match"].values()) and
            r["prepared_cache_hash_matches"] and r["prepared_state_identity_matches"] and
            r["indices_hash_matches"] and r["summary_numerics_match"] and r["indices_count"] == r["expected_count"] for r in checks),
        "llm_cache_hash_matches": llm_hash_matches,
        "endpoint_budget_coverage_90_range": [min(float(r["coverage_90_mean"]) for r in endpoint_metrics),
                                                max(float(r["coverage_90_mean"]) for r in endpoint_metrics)],
        "endpoint_budget_coverage_below_90_count": sum(float(r["coverage_90_mean"]) < .9 for r in endpoint_metrics),
        "endpoint_budget_row_count": len(endpoint_metrics), "groups": groups,
        "projection_paired_training_evidence": projection_pairs,
        "state_identity_checks": checks, "selected_legacy_manifest_checks": selected_manifest_checks,
        "gradient_verification": gradient,
        "additional_engineering_checks": engineering,
        "metadata_correction": {"raw_fit_seed_field": "Optimizer seed, overwritten by **model.audit in legacy run_benchmark.py",
            "raw_fit_kernel_field": "Projected fits store matern52 after **model.audit overrides the variant name; trainable_projection identifies them.",
            "summary_csv": "Data seed restored from state directory and projected variant restored from trainable_projection; model_random_seed/actual_kernel retain original fields.",
            "mathematical_result_bug_detected": False},
        "evidence_limits": [
            "Current L_t checkpoints were trained on all L_t responses; internal GP train/cal/check splits are conditional representation diagnostics only.",
            "The 416 validation responses participated in QGeoGNN checkpoint selection and repeated method development.",
            "A strict forward test must use future already-revealed rows with features from the pre-acquisition checkpoint, plus immutable acquisition/feedback receipts.",
            "The legacy prepared cache contains only L_t, 256 current U covariates, and 416 development validation covariates; it cannot cover an arbitrary future block.",
            "Projection weights and per-iteration gradient norms were not saved in historical fit JSON; historical geometric collapse/condition numbers cannot be reconstructed without a replay.",
            "All fit/coverage aggregates are descriptive; five seeds share source rows and four budgets are nested."],
        "new_test_truth_access_count": 0, "new_gnn_fits": 0,
        "source_sha256": READS}
    (HERE / "audit_results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in ("complete_state_count", "endpoint_fit_count", "projected_fit_count", "all_identity_checks_passed", "llm_cache_hash_matches", "all_final_jitter_zero")}, indent=2))


if __name__ == "__main__":
    main()
