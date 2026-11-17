"""Execute frozen qualification, blind acquisition diagnostics, and formal AL."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import pickle
import resource
import time

from study_support import (HERE, ROOT, BASE, read, dump, digest, write_once, feature_matrix,
                           training_inputs, reference_positions, verify_freeze, freeze)
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from qgeognn_al.active_learning_v2.epig_models import (
    GaussianProcessSurrogate, BayesianLinearSurrogate, qualify_model,
)
from qgeognn_al.active_learning_v2.epig import (
    GaussianAcquisitionState, score_gaussian_acquisition, select_gaussian_batch,
)


def factory(family):
    config = read(HERE / "config.json")
    common = {"max_components": config["pca_max_components"], "maxiter": config["optimizer_maxiter"]}
    if family in ("gp_rbf", "gp_matern32"):
        return GaussianProcessSurrogate(kernel=family.removeprefix("gp_"),
                                        restarts=config["optimizer_restarts"], **common)
    if family in ("blr_independent", "blr_coregionalized"):
        return BayesianLinearSurrogate(coregionalized=family == "blr_coregionalized", **common)
    raise ValueError("unregistered model family")


def qualification():
    verify_freeze()
    config = read(HERE / "config.json")
    started = time.perf_counter()
    budget_path = HERE / "qualification/compute_budget.json"
    budget = read(budget_path) if budget_path.exists() else {"started_unix": time.time(), "attempts": []}
    dump(budget_path, budget)
    def reserve_factory(family, seed):
        if len(budget["attempts"]) >= config["compute_budget"]["qualification_fit_attempts_max_including_interruptions"]:
            raise RuntimeError("qualification fit-attempt budget exhausted; incomplete is not an efficacy result")
        if time.time() - budget["started_unix"] > config["compute_budget"]["qualification_wall_budget_seconds"]:
            raise RuntimeError("cumulative qualification wall budget exhausted")
        budget["attempts"].append({"seed": seed, "family": family, "started_unix": time.time()})
        dump(budget_path, budget)
        return factory(family)
    frame, x = feature_matrix()
    summaries = []
    for seed in config["seeds"]:
        _, _, indices, y = training_inputs(seed, feature_frame=frame)
        for family in config["probabilistic_families"]:
            directory = HERE / f"qualification/seed_{seed}_{family}"
            complete = directory / "complete.json"
            if complete.exists():
                receipt = read(complete)
                if receipt["freeze_sha256"] != digest(HERE / "freeze.json"):
                    raise RuntimeError("qualification freeze drift")
                for name, expected in receipt["files"].items():
                    if digest(directory / name) != expected:
                        raise RuntimeError("qualification output drift")
            else:
                if time.perf_counter() - started > config["compute_budget"]["qualification_wall_budget_seconds"]:
                    dump(HERE / "qualification/status.json", {"status": "BUDGET_LIMIT_INCOMPLETE", "completed": len(summaries)})
                    return
                print(json.dumps({"phase": "qualification", "seed": seed, "family": family}), flush=True)
                directory.mkdir(parents=True, exist_ok=True)
                t = time.perf_counter()
                result = qualify_model(lambda: reserve_factory(family, seed), x[indices], y,
                                       n_splits=config["oof_folds"], seed=seed + config["oof_seed_offset"],
                                       cache_directory=directory / "fold_cache", cache_contract={"family": family,
                                       "freeze_sha256": digest(HERE / "freeze.json")})
                arrays = {"canonical_indices": indices, "sample_ids": frame.iloc[indices].sample_id.to_numpy(str),
                          "authorized_L333_truth_ml": y, "oof_mean_ml": result.pop("oof_mean"),
                          "oof_covariance_blocks_ml2": result.pop("oof_covariance_blocks")}
                np.savez_compressed(directory / "oof.npz", **arrays)
                fold_position = np.empty(len(indices), dtype=int)
                for f in result["folds"]:
                    fold_position[f["held_out_positions"]] = f["fold"]
                raw = []
                for i, index in enumerate(indices):
                    for j, endpoint in enumerate(("V1", "V2")):
                        raw.append({"seed": seed, "family": family, "sample_id": arrays["sample_ids"][i],
                                    "canonical_index": int(index), "fold": int(fold_position[i]), "endpoint": endpoint,
                                    "truth_ml": y[i, j], "mean_ml": arrays["oof_mean_ml"][i, j],
                                    "variance_ml2": arrays["oof_covariance_blocks_ml2"][i, j, j]})
                pd.DataFrame(raw).to_csv(directory / "oof_predictions.csv", index=False)
                dump(directory / "metrics.json", result)
                write_once(complete, {"freeze_sha256": digest(HERE / "freeze.json"),
                                     "elapsed_seconds": time.perf_counter() - t,
                                     "auxiliary_model_fits": config["oof_folds"],
                                     "files": {n: digest(directory / n) for n in ("oof.npz", "metrics.json", "oof_predictions.csv")}})
            result = read(directory / "metrics.json")
            for j, endpoint in enumerate(("V1", "V2")):
                m = result["metrics"]
                summaries.append({"seed": seed, "family": family, "endpoint": endpoint,
                                  **{k: m[k][j] for k in ("rmse", "mae", "marginal_nll", "coverage50", "coverage80", "coverage95")},
                                  "standardized_residual_rms": np.sqrt(m["standardized_residual_second_moment"][j]),
                                  "joint_nlpd_degradation_per_scalar": sum(f["held_out_rows"] *
                                      (f["full_joint_nll_per_scalar"] - f["diagonal_nll_per_scalar"]) for f in result["folds"]) / len(indices),
                                  "finite_positive_covariance": all(f["minimum_predictive_eigenvalue"] > 0 for f in result["folds"]),
                                  "optimizer_failures": sum(sum(not p["optimizer_success"] for p in f["model"].get("parameters", [f["model"]]))
                                                            for f in result["folds"])})
    table = pd.DataFrame(summaries)
    table.to_csv(HERE / "qualification/seed_metrics.csv", index=False)
    gates = []
    by_seed = {}
    q = config["qualification"]
    for (seed, family), group in table.groupby(["seed", "family"], sort=False):
        reasons = []
        for endpoint, endpoint_group in group.groupby("endpoint"):
            for field, bounds in (("coverage80", q["coverage80_bounds"]), ("coverage95", q["coverage95_bounds"]),
                                  ("standardized_residual_rms", q["standardized_residual_rms_bounds"])):
                # RMS aggregates second moments, rather than averaging RMS values.
                value = (np.sqrt(np.mean(endpoint_group[field] ** 2)) if field.endswith("_rms")
                         else float(endpoint_group[field].mean()))
                if not bounds[0] <= value <= bounds[1]:
                    reasons.append(f"{endpoint} {field}={value:.6f} outside {bounds}")
        joint_difference = float(group.joint_nlpd_degradation_per_scalar.mean())
        if joint_difference > q["joint_nlpd_max_degradation_vs_diagonal_per_scalar"]:
            reasons.append(f"joint NLPD degradation per scalar {joint_difference:.6f} > .10")
        if not group.finite_positive_covariance.all():
            reasons.append("nonpositive predictive covariance")
        gates.append({"seed": int(seed), "family": family, "qualified": not reasons, "reasons": reasons,
                      "mean_marginal_nlpd": float(group.marginal_nll.mean()),
                      "joint_nlpd_degradation_per_scalar": joint_difference})
    for seed in config["seeds"]:
        local = [g for g in gates if g["seed"] == seed]
        gp = sorted([g for g in local if g["qualified"] and g["family"].startswith("gp_")],
                    key=lambda g: (g["mean_marginal_nlpd"], g["family"]))
        fallback = sorted([g for g in local if g["qualified"] and g["family"].startswith("blr_")],
                          key=lambda g: (g["mean_marginal_nlpd"], g["family"]))
        chosen = (gp or fallback)
        by_seed[str(seed)] = {"status": "QUALIFIED" if chosen else "NO_QUALIFIED_POSTERIOR",
                              "selected_family": chosen[0]["family"] if chosen else None,
                              "formal_arms": config["primary_arms"], "models": local}
    ready = all(d["selected_family"] is not None for d in by_seed.values())
    decision = {"status": "QUALIFIED" if ready else "INCOMPLETE_POSTERIOR_QUALIFICATION", "run_allowed": ready,
                "selected_family_by_seed": {s: d["selected_family"] for s,d in by_seed.items()}, "by_seed": by_seed,
                "freeze_sha256": digest(HERE / "freeze.json"), "models": gates,
                "new_QGeoGNN_fits": 0, "future_U_labels_read": 0, "validation_labels_read": 0,
                "test_labels_read": 0, "evidence": "train-only L333 OOF development qualification",
                "qualification_auxiliary_fits": 60, "fit_attempts": len(budget["attempts"]),
                "elapsed_seconds_this_invocation": time.perf_counter() - started}
    if (HERE / "qualification/decision.json").exists():
        decision["elapsed_seconds_this_invocation"] = read(HERE / "qualification/decision.json")["elapsed_seconds_this_invocation"]
    write_once(HERE / "qualification/decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False), flush=True)


def fitted_initial_model(seed, family, x, indices, y):
    directory = HERE / f"acquisition/models/seed_{seed}_{family}"
    receipt = directory / "complete.json"
    if receipt.exists():
        r = read(receipt)
        if r["freeze_sha256"] != digest(HERE / "freeze.json") or digest(directory / "model.pkl") != r["model_sha256"] or digest(directory / "fit_audit.json") != r["fit_audit_sha256"]:
            raise RuntimeError("acquisition model cache drift")
        with (directory / "model.pkl").open("rb") as f:
            return pickle.load(f)
    scales = y.astype(np.float32).std(axis=0, ddof=0).astype(float)
    model = factory(family).fit(x[indices], y / scales)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "model.pkl").open("wb") as f:
        pickle.dump(model, f)
    dump(directory / "fit_audit.json", {**model.audit_, "output_units": "original fixed L333 population standard deviations",
                                       "physical_scale_ml": scales.tolist(), "training_indices": indices.tolist()})
    write_once(receipt, {"freeze_sha256": digest(HERE / "freeze.json"), "model_sha256": digest(directory / "model.pkl"),
                         "fit_audit_sha256": digest(directory / "fit_audit.json")})
    return model


def overlap(a, b, k):
    return len(set(np.argsort(-a, kind="stable")[:k]) & set(np.argsort(-b, kind="stable")[:k])) / k


def fixed_batch_epig(posterior, candidates, references, chosen):
    state = GaussianAcquisitionState(posterior, candidates, references)
    lookup = {int(p): i for i, p in enumerate(candidates)}
    info = 0.
    for pos in chosen:
        scores = state.scores("epig")
        index = lookup[int(pos)]
        info += float(scores[index])
        state.condition(index)
    return info


def acquisition_diagnostics():
    verify_freeze()
    config = read(HERE / "config.json")
    decision = read(HERE / "qualification/decision.json")
    # If no GP qualified, inspect both kernels without calling either a reliable acquisition model.
    started = time.perf_counter()
    frame, x = feature_matrix()
    runs = []
    for seed in config["seeds"]:
        selected_family = decision["selected_family_by_seed"][str(seed)]
        gps = [selected_family] if selected_family and selected_family.startswith("gp_") else ["gp_rbf", "gp_matern32"]
        _, split, indices, y = training_inputs(seed, feature_frame=frame)
        outer = split.loc[split.role.isin(["l0", "u0"]), "canonical_index"].to_numpy(int)
        current_u = split.loc[split.role.eq("u0"), "canonical_index"].to_numpy(int)
        index_map = {int(v): i for i, v in enumerate(outer)}
        candidates = np.array([index_map[int(i)] for i in current_u], dtype=int)
        labeled = np.array([index_map[int(i)] for i in indices], dtype=int)
        row_ids = frame.iloc[outer].sample_id.to_numpy(str)
        for family in gps + ["blr_independent", "blr_coregionalized"]:
            model = fitted_initial_model(seed, family, x, indices, y)
            posterior = model.posterior(x[outer], row_ids=row_ids)
            # Every GP gets P1/P2/M sensitivity; alternative BLRs are prespecified at primary P1/M1024.
            settings = [(p, m) for p in ("P1", "P2") for m in config["targets"]["reference_sizes"]] if family.startswith("gp_") else [("P1", 1024)]
            for distribution, size in settings:
                directory = HERE / f"acquisition/seed_{seed}_{family}_{distribution}_M{size}"
                complete = directory / "complete.json"
                if complete.exists():
                    r = read(complete)
                    if r["freeze_sha256"] != digest(HERE / "freeze.json"):
                        raise RuntimeError("acquisition freeze drift")
                    for name, expected in r["files"].items():
                        if digest(directory / name) != expected:
                            raise RuntimeError("acquisition output drift")
                    runs.append(read(directory / "summary.json"))
                    continue
                if time.perf_counter() - started > config["compute_budget"]["acquisition_diagnostics_wall_budget_seconds"]:
                    dump(HERE / "acquisition/status.json", {"status": "BUDGET_LIMIT_INCOMPLETE", "completed_states": len(runs)})
                    return
                print(json.dumps({"phase": "acquisition", "seed": seed, "family": family, "target": distribution, "M": size}), flush=True)
                references = reference_positions(seed, distribution, np.arange(len(outer)) if distribution == "P1" else candidates,
                                                 row_ids, size)
                directory.mkdir(parents=True, exist_ok=True)
                t = time.perf_counter()
                selections = {}
                for objective, mode in (("epig", "conditional_greedy"), ("ivr", "conditional_greedy"), ("epig", "pointwise_topk")):
                    name = f"{objective}_{mode}"
                    selections[name] = select_gaussian_batch(posterior, candidates, references, objective=objective,
                                                             batch_mode=mode, labeled_positions=labeled)
                scores = pd.DataFrame({"sample_id": row_ids[candidates], "canonical_index": outer[candidates],
                                       "epig": selections["epig_conditional_greedy"]["initial_scores"],
                                       "ivr": selections["ivr_conditional_greedy"]["initial_scores"]})
                for endpoint in range(2):
                    scores[f"epig_V{endpoint+1}"] = score_gaussian_acquisition(posterior, candidates, references, output_indices=[endpoint])
                scores.to_csv(directory / "scores.csv", index=False)
                dump(directory / "batches.json", selections)
                for s in selections.values():
                    selected = s["selected_row_ids"]
                    if len(set(selected)) != 32 or not set(selected) <= set(row_ids[candidates]) or set(selected) & set(row_ids[labeled]):
                        raise RuntimeError("illegal acquisition IDs")
                e, v = scores.epig.to_numpy(), scores.ivr.to_numpy()
                epig_batch = selections["epig_conditional_greedy"]["selected_positions"]
                ivr_batch = selections["ivr_conditional_greedy"]["selected_positions"]
                point_batch = selections["epig_pointwise_topk"]["selected_positions"]
                summary = {"seed": seed, "family": family, "distribution": distribution, "M": len(references),
                           "posterior_qualified": next(g["qualified"] for g in decision["models"] if g["family"] == family and g["seed"] == seed),
                           "spearman_epig_ivr": float(spearmanr(e, v).statistic),
                           "top32_overlap_epig_ivr": overlap(e, v, 32), "top100_overlap_epig_ivr": overlap(e, v, 100),
                           "greedy_batch_overlap_epig_ivr": len(set(epig_batch) & set(ivr_batch)) / 32,
                           "greedy_pointwise_batch_overlap": len(set(epig_batch) & set(point_batch)) / 32,
                           "greedy_epig_batch_information_nats": selections["epig_conditional_greedy"]["audit"]["conditional_batch_information_nats"],
                           "pointwise_batch_information_nats": selections["epig_pointwise_topk"]["audit"]["conditional_batch_information_nats"],
                           "ivr_batch_information_nats": fixed_batch_epig(posterior, candidates, references, ivr_batch),
                           "endpoint_V1_V2_top32_overlap": overlap(scores.epig_V1.to_numpy(), scores.epig_V2.to_numpy(), 32),
                           "V1_fraction_singleton_information": float(scores.epig_V1.sum() / (scores.epig_V1.sum()+scores.epig_V2.sum())),
                           "seconds": time.perf_counter() - t, "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           "candidate_rows": len(candidates), "reference_positions": references.tolist(),
                           "labels_read_for_acquisition": 0, "new_gnn_fits": 0}
                dump(directory / "summary.json", summary)
                write_once(complete, {"freeze_sha256": digest(HERE / "freeze.json"),
                                      "files": {n: digest(directory / n) for n in ("scores.csv", "batches.json", "summary.json")}})
                runs.append(summary)
                gc.collect()
    pd.DataFrame([{k:v for k,v in r.items() if k != "reference_positions"} for r in runs]).to_csv(HERE / "acquisition/summary.csv", index=False)
    sensitivity = []
    for seed in config["seeds"]:
        selected_family = decision["selected_family_by_seed"][str(seed)]
        gps = [selected_family] if selected_family and selected_family.startswith("gp_") else ["gp_rbf", "gp_matern32"]
        for family in gps:
            for distribution in ("P1", "P2"):
                full = HERE / f"acquisition/seed_{seed}_{family}_{distribution}_M2048"
                baseline = pd.read_csv(full / "scores.csv")
                for size in (256, 1024):
                    other = pd.read_csv(HERE / f"acquisition/seed_{seed}_{family}_{distribution}_M{size}/scores.csv")
                    if baseline.sample_id.tolist() != other.sample_id.tolist():
                        raise RuntimeError("sensitivity candidate drift")
                    for objective in ("epig", "ivr"):
                        a, b = baseline[objective].to_numpy(), other[objective].to_numpy()
                        sensitivity.append({"seed": seed, "family": family, "distribution": distribution, "M": size,
                                            "reference_M": 2048, "objective": objective,
                                            "relative_score_rmse": float(np.linalg.norm(a-b) / np.linalg.norm(a)),
                                            "spearman": float(spearmanr(a, b).statistic),
                                            "top32_overlap": overlap(a,b,32), "top100_overlap": overlap(a,b,100)})
    pd.DataFrame(sensitivity).to_csv(HERE / "acquisition/reference_sensitivity.csv", index=False)
    dump(HERE / "acquisition/status.json", {"status": "COMPLETE", "completed_states": len(runs), "new_GNN_fits": 0})


def formal_callback(seed, arm, round_index, labeled_indices, unlabeled_indices, labeled_truth,
                    feature_matrix_X, feature_frame, target_scales, previous_checkpoint_path=None):
    decision = read(HERE / "qualification/decision.json")
    if decision["status"] != "QUALIFIED":
        raise RuntimeError("no qualified primary GP")
    outer = np.r_[np.asarray(labeled_indices, int), np.asarray(unlabeled_indices, int)]
    scales = np.asarray([target_scales[t] for t in ("V1", "V2")])
    model = factory(decision["selected_family_by_seed"][str(seed)]).fit(feature_matrix_X[labeled_indices], labeled_truth / scales)
    row_ids = feature_frame.iloc[outer].sample_id.to_numpy(str)
    posterior = model.posterior(feature_matrix_X[outer], row_ids=row_ids)
    candidates = np.arange(len(labeled_indices), len(outer))
    references = reference_positions(seed, "P1", np.arange(len(outer)), row_ids, 1024)
    objective = "epig" if arm == "posterior_epig_conditional_greedy32" else "ivr"
    result = select_gaussian_batch(posterior, candidates, references, objective=objective,
                                  labeled_positions=np.arange(len(labeled_indices)))
    return {"selected_ids": result["selected_row_ids"], "diagnostics": result,
            "probabilistic_model_audit": model.audit_, "units": "fixed L333 endpoint std"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "qualify", "acquisition", "formal"])
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
    elif args.phase == "qualify":
        qualification()
    elif args.phase == "acquisition":
        acquisition_diagnostics()
    else:
        from epig_formal_runner import run_formal
        run_formal(formal_callback)


if __name__ == "__main__":
    main()
