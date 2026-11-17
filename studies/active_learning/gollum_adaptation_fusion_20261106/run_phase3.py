"""Bounded development-only additive-kernel comparison using sealed caches."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = HERE.parent / "representation_gp_benchmark_20261104"
sys.path.append(str(OLD))
from representations import audit_lcmd_state, file_hash, stable_hash, FEATURE_COLUMNS
from diagnostics import fixed_candidate_positions, _conditions
from kernel_fusion import AdditiveKernelGP, acquisition_diagnostics, regression_metrics

PLAN = dict(schema="additive-kernel-pilot-v1", seeds=[157, 887], budgets=[333, 813],
            endpoints=["V1", "V2"], candidate_count=256, validation_rows=416, batch_size=32,
            methods=["gnn_lambda0", "additive_fixed1", "additive_learned", "gnn_capacity_control"],
            max_iter=200, initialization=17, lambda_bounds_log=[-9., 9.],
            log_signal_noise_bounds=[-9., 9.], log_lengthscale_bounds=[-5., 5.],
            training="exact float64 MLL on current L_t responses only; no projection or encoder optimization",
            evaluation="previously checkpoint-selected validation; development evidence only",
            feature_processing="separate L_t block-centered global RMS; no PCA",
            noise_regularization="none; compare frozen-kernel behavior independently of Phase1 projection",
            capacity_control="two GNN Matern52 components, learned lambda and two lengthscales; same five scalars as learned fusion",
            candidate_policy="reuse sealed 256 SHA256-ranked current U rows and IDs without response queries",
            test_truth_access_count=0, future_U_label_reads=0, new_gnn_fits=0, new_lm_forwards=0,
            model_downloads=0, dependency_installs=0, al_trajectories_started=0,
            expansion="none; only two development seeds and L333/L813; all failures retained")


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def fit_record(base, audit):
    """Keep the trajectory seed distinct from the GP initialization seed."""
    return {**audit, **base, "optimizer_seed": audit["seed"]}


def load_state(seed, budget):
    """Read sealed authorized responses only after proving exact row identity."""
    state = audit_lcmd_state(ROOT, seed, budget)
    key = f"seed_{seed}_L{budget}"
    cache = OLD / "cache" / (key + ".npz")
    receipt = cache.with_suffix(".json")
    meta = json.loads(receipt.read_text())
    if file_hash(cache) != meta["cache_sha256"] or state.audit != meta["state"]:
        raise ValueError("state feature cache integrity or checkpoint state drift")
    positions = fixed_candidate_positions(state.unlabeled_ids, 256, seed)
    candidate_ids = [state.unlabeled_ids[i] for i in positions]
    expected_indices = np.r_[state.labeled_indices, state.unlabeled_indices[positions], state.validation_indices]
    state.validate_indices(expected_indices)
    if candidate_ids != meta["candidate_ids"]:
        raise ValueError("candidate IDs differ from sealed old comparison")
    with np.load(cache, allow_pickle=False) as z:
        if not np.array_equal(z["indices"], expected_indices):
            raise ValueError("cache row order differs from L/current-U/development validation")
        if hashlib.sha256(z["indices"].tobytes()).hexdigest() != meta["indices_sha256"]:
            raise ValueError("cache row hash mismatch")
        data = {k: z[k] for k in ("indices", "gnn", "train_y", "validation_y")}
    if data["train_y"].shape != (budget, 2) or data["validation_y"].shape != (416, 2):
        raise ValueError("authorized response shape mismatch")
    label_audit = meta["label_access"]
    if label_audit["test_truth_access_count"] or label_audit["future_U_labels_read"]:
        raise PermissionError("old cache response authorization failed")
    llm_path = OLD / "cache/llm.npz"
    llm_receipt = llm_path.with_suffix(".json")
    lm = json.loads(llm_receipt.read_text())
    if file_hash(llm_path) != lm["cache_sha256"]:
        raise ValueError("language cache integrity failed")
    with np.load(llm_path, allow_pickle=False) as z:
        if len(np.unique(z["indices"])) != len(z["indices"]):
            raise ValueError("language cache has duplicate indices")
        lookup = {int(i): j for j, i in enumerate(z["indices"])}
        pos = [lookup[int(i)] for i in expected_indices]
        language = z["llm_conditions"][pos]
        template_b = z["llm_template_b"][pos]
    x = np.concatenate([data["gnn"], language], axis=1).astype(np.float64)
    xb = np.concatenate([data["gnn"], template_b], axis=1).astype(np.float64)
    if not np.isfinite(x).all() or not np.isfinite(xb).all():
        raise ValueError("nonfinite frozen feature")
    audit = dict(state=state.audit, state_audit_sha256=stable_hash(state.audit),
                 cache=str(cache.relative_to(ROOT)), cache_sha256=file_hash(cache),
                 cache_receipt_sha256=file_hash(receipt), llm_cache_sha256=file_hash(llm_path),
                 llm_receipt_sha256=file_hash(llm_receipt),
                 selected_llm_features_sha256=hashlib.sha256(language.astype("<f8").tobytes()).hexdigest(),
                 selected_gnn_features_sha256=hashlib.sha256(data["gnn"].astype("<f8").tobytes()).hexdigest(),
                 train_y_sha256=hashlib.sha256(data["train_y"].astype("<f8").tobytes()).hexdigest(),
                 validation_y_sha256=hashlib.sha256(data["validation_y"].astype("<f8").tobytes()).hexdigest(),
                 ordered_indices_sha256=meta["indices_sha256"], candidate_ids_sha256=stable_hash(candidate_ids),
                 candidate_ids=candidate_ids, language_model=lm["encoder"],
                 label_access="only sealed current L_t and previously authorized development validation", test_truth_access_count=0)
    return state, data, x, xb, audit


def selection_metadata(state, candidate_ids):
    # Restrict the parser to response-free columns; no response table is loaded.
    source = ROOT / "experiments/e0_4g_baseline/canonical_4g.csv"
    frame = pd.read_csv(source, usecols=list(FEATURE_COLUMNS), dtype={"sample_id": str})
    ids = set(state.labeled_ids + candidate_ids)
    subset = frame.loc[frame.sample_id.isin(ids)]
    fields = ("PE/EA", "Density g/ml", "V/ul", "loading solvent", "Volume of loading solvent/ul")
    meta = {str(r.sample_id): {"smiles": str(r.canonical_smiles),
            "conditions": {name: r[name] for name in fields}} for _, r in subset.iterrows()}
    if set(meta) != ids:
        raise ValueError("selection metadata missing legal IDs")
    return meta


def summarize():
    fits, metrics, predictions, acquisitions, failures, audits = [], [], [], [], [], []
    for path in sorted((HERE / "phase3/results").glob("*/*.json")):
        r = json.loads(path.read_text())
        if r.get("status") == "failed":
            failures.append(r)
        elif r.get("status") == "complete":
            fits.append(r["fit"])
            metrics.append(r["metrics"])
            predictions.extend(r["predictions"])
            acquisitions.append(r["acquisition"])
    for path in sorted((HERE / "phase3/audits").glob("*.json")):
        audits.append(json.loads(path.read_text()))
    out = HERE / "phase3"
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "predictions.csv", index=False)
    dump(out / "fits.json", fits)
    dump(out / "acquisitions.json", acquisitions)
    dump(out / "failures.json", failures)
    dump(out / "input_audits.json", audits)
    template_rows, diversity_rows = [], []
    for acq in acquisitions:
        template_rows.append({**{k: acq[k] for k in ("seed", "budget", "endpoint", "method")},
                              **{k: v for k, v in acq["template_B"].items() if not isinstance(v, (list, dict))}})
        for mode, selection in acq["methods"].items():
            diversity_rows.append({**{k: acq[k] for k in ("seed", "budget", "endpoint", "method")},
                                   "acquisition": mode,
                                   "mean_absolute_batch_latent_correlation": selection["mean_absolute_batch_latent_correlation"],
                                   "unique_molecules": selection["unique_molecules"],
                                   "unique_condition_combinations": selection["unique_condition_combinations"],
                                   "batch_information": selection["batch_information"]})
    pd.DataFrame(template_rows).to_csv(out / "template_robustness.csv", index=False)
    pd.DataFrame(diversity_rows).to_csv(out / "selection_diversity.csv", index=False)
    scalar_fits = [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in fits]
    pd.DataFrame(scalar_fits).to_csv(out / "fits.csv", index=False)
    comparison = []
    for base in acquisitions:
        if base["method"] != "gnn_lambda0":
            continue
        for other in acquisitions:
            if (other["seed"], other["budget"], other["endpoint"]) != (base["seed"], base["budget"], base["endpoint"]):
                continue
            if other["method"] == base["method"]:
                continue
            if other["candidate_ids"] != base["candidate_ids"]:
                raise ValueError("comparison candidate pool drift")
            row = {k: other[k] for k in ("seed", "budget", "endpoint", "method")}
            row["latent_variance_rank_spearman_vs_gnn"] = float(spearmanr(base["latent_variance"], other["latent_variance"]).statistic)
            for name in base["methods"]:
                a = set(base["methods"][name]["selected_ids"])
                b = set(other["methods"][name]["selected_ids"])
                row[name + "_top32_overlap_vs_gnn"] = len(a & b)/32
            comparison.append(row)
    pd.DataFrame(comparison).to_csv(out / "acquisition_stability.csv", index=False)
    if metrics:
        frame = pd.DataFrame(metrics)
        controls = frame.loc[frame.method.eq("gnn_lambda0")]
        paired = frame.merge(controls, on=["seed", "budget", "endpoint"], suffixes=("", "_gnn"))
        fields = ["rmse", "mae", "nlpd", "coverage_90", "coverage_95", "mean_interval_width_90", "mean_interval_width_95", "standardized_error_rms", "normalized_rmse_L0"]
        rows = []
        for _, r in paired.iterrows():
            row = {k: r[k] for k in ("seed", "budget", "endpoint", "method")}
            for field in fields:
                if field in r:
                    row["delta_"+field] = float(r[field]-r[field+"_gnn"])
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / "paired_differences.csv", index=False)
        numeric = [c for c in frame.select_dtypes(include=np.number).columns if c not in ("seed", "budget")]
        frame.groupby(["seed", "budget", "method"])[numeric].mean().reset_index().to_csv(out / "per_seed_budget.csv", index=False)
        frame.groupby(["budget", "method"])[numeric].mean().reset_index().to_csv(out / "per_budget.csv", index=False)
        frame.groupby(["method"])[numeric].mean().reset_index().to_csv(out / "aggregate.csv", index=False)
    cost = dict(endpoint_fits_completed=len(fits), endpoint_fits_failed=len(failures),
                fit_seconds=sum(r["training_seconds_including_condition_number"] for r in fits),
                acquisition_seconds=sum(r["acquisition_seconds"] for r in metrics),
                template_seconds=sum(r["template_seconds"] for r in metrics),
                max_iter=PLAN["max_iter"], new_gnn_fits=0, lm_forwards=0,
                test_truth_access_count=0, al_trajectories_started=0)
    dump(out / "cost.json", cost)
    historical = []
    historical_sources = {}
    for seed in PLAN["seeds"]:
        for budget in PLAN["budgets"]:
            for tag in ("gnn_matern52", "llm_conditions_matern52", "fusion_matern52",
                        "fusion_projected_init17", "fusion_projected_init29", "fusion_projected_init43"):
                path = OLD / "results" / f"seed_{seed}_L{budget}" / (tag+".json")
                r = json.loads(path.read_text())
                historical_sources[str(path.relative_to(ROOT))] = file_hash(path)
                for metric, fit in zip(r["metrics"], r["fits"]):
                    historical.append({**metric, "historical_tag": tag,
                                       "optimizer_success": fit["optimizer_success"],
                                       "noise_variance": fit["noise_variance"],
                                       "numerical_jitter": fit["numerical_jitter"],
                                       "max_iter": fit["max_iter"],
                                       "source": str(path.relative_to(ROOT)),
                                       "source_sha256": historical_sources[str(path.relative_to(ROOT))]})
    pd.DataFrame(historical).to_csv(out / "historical_references.csv", index=False)
    dump(out / "historical_source_hashes.json", historical_sources)
    manifest = {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"}
    manifest.update({str((HERE / name).relative_to(ROOT)): file_hash(HERE / name) for name in
                     ("kernel_fusion.py", "run_phase3.py", "reproduce_phase3.py", "test_kernel_fusion.py",
                      "PHASE3_FUSION_DESIGN.md", "PHASE3_RESULTS.md") if (HERE/name).exists()})
    dump(out / "artifact_hashes.json", manifest)
    print(json.dumps(cost), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    out = HERE / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != PLAN:
        raise ValueError("frozen phase3 protocol changed")
    if not config_path.exists():
        dump(config_path, PLAN)
    if args.summarize_only:
        summarize()
        return
    seeds = PLAN["seeds"][:1] if args.preflight else PLAN["seeds"]
    budgets = PLAN["budgets"][:1] if args.preflight else PLAN["budgets"]
    endpoints = [0] if args.preflight else [0, 1]
    for seed in seeds:
        for budget in budgets:
            state, data, x, xb, audit = load_state(seed, budget)
            key = f"seed_{seed}_L{budget}"
            dump(out / "audits" / (key + ".json"), audit)
            candidate_ids = audit["candidate_ids"]
            metadata = selection_metadata(state, candidate_ids)
            n, vs = budget, budget+256
            for j in endpoints:
                for method in PLAN["methods"]:
                    path = out / "results" / key / f"{method}_V{j+1}.json"
                    if path.exists():
                        continue
                    base = dict(seed=seed, budget=budget, endpoint=f"V{j+1}", method=method, initialization=17)
                    started = time.perf_counter()
                    try:
                        gp = AdditiveKernelGP(gnn_dimension=data["gnn"].shape[1],
                            lambda_mode={"gnn_lambda0": "zero", "additive_fixed1": "fixed"}.get(method, "learned"),
                            max_iter=PLAN["max_iter"], capacity_control=method == "gnn_capacity_control")
                        gp.fit(x[:n], data["train_y"][:, j])
                        mu, latent = gp.predict(x[vs:])
                        score = regression_metrics(data["validation_y"][:, j], mu, latent+gp.noise)
                        score["normalized_rmse_L0"] = score["rmse"]/state.audit["endpoint_scales"][f"V{j+1}"]
                        acq_started = time.perf_counter()
                        acq = acquisition_diagnostics(gp, x[n:vs], batch_size=32, candidate_limit=256, seed=seed)
                        for selection in acq["methods"].values():
                            selection["selected_ids"] = [candidate_ids[p] for p in selection["selected_candidate_positions"]]
                            selected = selection["selected_ids"]
                            selection["unique_molecules"] = len({metadata[sid]["smiles"] for sid in selected})
                            selection["unique_condition_combinations"] = len({_conditions._canonical_conditions(metadata[sid]) for sid in selected})
                            selection["condition_metrics"], _ = _conditions.compute_condition_metrics(selected, state.labeled_ids, metadata, return_examples=0)
                        acq_seconds = time.perf_counter()-acq_started
                        template_started = time.perf_counter()
                        other = acquisition_diagnostics(gp, xb[n:vs], batch_size=32, candidate_limit=256, seed=seed)
                        template = dict(policy="fit A on L_t; apply frozen transform and GP to B candidate text; no B fitting",
                                        variance_rank_spearman=float(spearmanr(acq["latent_variance"], other["latent_variance"]).statistic),
                                        mean_absolute_change_ml=float(np.mean(np.abs(np.array(acq["latent_mean"])-other["latent_mean"]))),
                                        latent_variance_mean_A=float(np.mean(acq["latent_variance"])),
                                        latent_variance_mean_B=float(np.mean(other["latent_variance"])),
                                        mean_absolute_variance_change_ml2=float(np.mean(np.abs(np.array(acq["latent_variance"])-other["latent_variance"]))),
                                        latent_mean=other["latent_mean"], latent_variance=other["latent_variance"],
                                        methods=other["methods"])
                        for name in acq["methods"]:
                            a = set(acq["methods"][name]["selected_candidate_positions"])
                            b = set(other["methods"][name]["selected_candidate_positions"])
                            template[name+"_top32_overlap"] = len(a & b)/32
                        template_seconds = time.perf_counter()-template_started
                        prediction = [{**base, "id": sid, "truth_ml": float(data["validation_y"][k,j]),
                            "mean_ml": float(mu[k]), "latent_variance_ml2": float(latent[k]),
                            "observation_variance_ml2": float(latent[k]+gp.noise)} for k, sid in enumerate(state.validation_ids)]
                        r = dict(status="complete", metrics={**base, **score, "acquisition_seconds": acq_seconds,
                                  "template_seconds": template_seconds, "total_seconds": time.perf_counter()-started,
                                  "evaluation_role": PLAN["evaluation"]}, fit=fit_record(base, gp.audit), predictions=prediction,
                                 acquisition={**base, **acq, "candidate_ids": candidate_ids,
                                              "candidate_ids_sha256": audit["candidate_ids_sha256"], "template_B": template},
                                 input_state_sha256=audit["state_audit_sha256"], config_sha256=file_hash(config_path))
                        dump(path, r)
                        print(key, method, f"V{j+1}", "seconds", round(time.perf_counter()-started, 2),
                              "converged", gp.audit["optimizer_success"], "lambda", round(gp.lambda_value, 6), flush=True)
                    except Exception as exc:
                        dump(path, dict(status="failed", **base, error_type=type(exc).__name__, error=str(exc),
                                        elapsed_seconds=time.perf_counter()-started, input_state_sha256=audit["state_audit_sha256"]))
                        print(key, method, f"V{j+1}", "FAILED", str(exc), flush=True)
    summarize()


if __name__ == "__main__":
    main()
