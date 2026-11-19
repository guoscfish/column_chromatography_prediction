"""Sealed, label-free single-output versus joint-output selection appendix.

This is a supplementary analysis declared after stage-one qualification. It
does not alter the scientific freeze, qualify models, fit a model, reveal a
label, or claim row-level learning efficacy. Only completed P1/M1024 states are
eligible; later completed states can be appended under the same analysis seal.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
import pickle
import resource
import time

from study_support import (HERE, read, dump, digest, write_once, feature_matrix,
                           partition, verify_freeze)
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from qgeognn_al.active_learning_v2.epig import (
    GaussianAcquisitionState, select_gaussian_batch,
)

APPENDIX = HERE / "supplementary_output_ablation"


def seal():
    """Separate post-qualification analysis seal; main freeze stays intact."""
    verify_freeze()
    record = {"schema": "epig-output-analysis-appendix-v1",
              "evidence_scope": "post-stage-one-qualification label-free frozen-state analysis only",
              "script_sha256": digest(__file__), "science_freeze_sha256": digest(HERE / "freeze.json"),
              "config_sha256": digest(HERE / "config.json"),
              "settings": {"target": "existing P1", "reference_rows": 1024, "batch_size": 32,
                           "outputs": [[0], [1]], "new_model_fits": 0, "new_label_reads": 0,
                           "wall_seconds_cumulative_max": 600,
                           "budget_action": "preserve completed states and stop; no outcome stopping"}}
    write_once(APPENDIX / "analysis_seal.json", record)
    return record


def joint_batch_information(posterior, references, selected):
    """Independent direct batch Schur, including two-output dependencies.

    Targets are future independent noisy replicas, even when also selected.
    One 64x64 noisy batch factorization suffices for all target blocks.
    """
    references = np.asarray(references, dtype=int)
    selected = np.asarray(selected, dtype=int)
    batch = posterior.cov(selected, selected, noisy=True)
    factor = cho_factor((batch + batch.T) / 2, lower=True)
    cross = posterior.cov(references, selected, noisy=False)
    solved = cho_solve(factor, cross.T).T
    q = posterior.n_outputs
    reduction = np.einsum("nib,njb->nij", cross.reshape(len(references), q, -1),
                          solved.reshape(len(references), q, -1))
    target = posterior.latent_diag(references) + posterior.noise_blocks(references)
    after = target - reduction
    after = (after + after.transpose(0, 2, 1)) / 2
    lt, la = np.linalg.cholesky(target), np.linalg.cholesky(after)
    information = (np.log(np.diagonal(lt, axis1=1, axis2=2)).sum(1) -
                   np.log(np.diagonal(la, axis1=1, axis2=2)).sum(1))
    if not np.isfinite(information).all() or information.min() < -1e-9:
        raise np.linalg.LinAlgError("invalid direct joint batch information")
    return float(information.mean())


def checked(path, expected):
    if not Path(path).exists() or digest(path) != expected:
        raise RuntimeError(f"appendix input hash mismatch: {path}")


def main():
    started = time.perf_counter()
    seal()
    budget_path = APPENDIX / "budget.json"
    budget = read(budget_path) if budget_path.exists() else {"consumed_seconds": 0., "runs": []}
    frame, x = feature_matrix()  # Legal X-only loader; never training_inputs().
    rows = []
    candidates_dirs = []
    for complete in sorted((HERE / "acquisition").glob("seed_*_P1_M1024/complete.json")):
        summary = read(complete.parent / "summary.json")
        if summary["distribution"] == "P1" and summary["M"] == 1024:
            candidates_dirs.append(complete.parent)
    for directory in candidates_dirs:
        if budget["consumed_seconds"] + time.perf_counter() - started >= 600:
            break
        source_receipt = read(directory / "complete.json")
        if source_receipt["freeze_sha256"] != digest(HERE / "freeze.json"):
            raise RuntimeError("source acquisition freeze differs")
        for name, expected in source_receipt["files"].items():
            checked(directory / name, expected)
        summary = read(directory / "summary.json")
        seed, family = int(summary["seed"]), summary["family"]
        model_directory = HERE / f"acquisition/models/seed_{seed}_{family}"
        model_receipt = read(model_directory / "complete.json")
        if model_receipt["freeze_sha256"] != digest(HERE / "freeze.json"):
            raise RuntimeError("fitted model freeze differs")
        checked(model_directory / "model.pkl", model_receipt["model_sha256"])
        output = APPENDIX / f"seed_{seed}_{family}"
        contract = {"analysis_seal_sha256": digest(APPENDIX / "analysis_seal.json"),
                    "source_acquisition_receipt_sha256": digest(directory / "complete.json"),
                    "model_receipt_sha256": digest(model_directory / "complete.json"),
                    "model_sha256": model_receipt["model_sha256"]}
        if (output / "complete.json").exists():
            receipt = read(output / "complete.json")
            if receipt["contract"] != contract:
                raise RuntimeError("completed appendix input drift")
            for name, expected in receipt["files"].items():
                checked(output / name, expected)
            rows.append(read(output / "summary.json"))
            continue
        t0 = time.perf_counter()
        split = partition(seed)
        outer = split.loc[split.role.isin(["l0", "u0"]), "canonical_index"].to_numpy(int)
        current_u = split.loc[split.role.eq("u0"), "canonical_index"].to_numpy(int)
        l0 = split.loc[split.role.eq("l0"), "canonical_index"].to_numpy(int)
        position = {int(index): j for j, index in enumerate(outer)}
        candidates = np.array([position[int(i)] for i in current_u], dtype=int)
        labeled = np.array([position[int(i)] for i in l0], dtype=int)
        references = np.asarray(summary["reference_positions"], dtype=int)
        if len(references) != 1024 or np.any(references < 0) or np.any(references >= len(outer)):
            raise ValueError("source target reference set is not valid P1/M1024")
        with (model_directory / "model.pkl").open("rb") as f:
            model = pickle.load(f)
        posterior = model.posterior(x[outer], row_ids=frame.iloc[outer].sample_id.to_numpy(str))
        source_batches = read(directory / "batches.json")
        joint_positions = np.asarray(source_batches["epig_conditional_greedy"]["selected_positions"], dtype=int)
        selections = {}
        for endpoint in range(2):
            selections[f"V{endpoint+1}_only"] = select_gaussian_batch(
                posterior, candidates, references, objective="epig", output_indices=[endpoint],
                batch_size=32, batch_mode="conditional_greedy", labeled_positions=labeled)
        joint_state = GaussianAcquisitionState(posterior, candidates, references)
        joint_initial = joint_state.scores("epig")
        del joint_state
        sum_initial = sum(np.asarray(result["initial_scores"]) for result in selections.values())
        source_scores = pd.read_csv(directory / "scores.csv")
        if source_scores.sample_id.tolist() != posterior.row_ids[candidates].tolist():
            raise RuntimeError("source candidate order drift")
        independent_error = float(np.max(np.abs(joint_initial - sum_initial)))
        recomputation_error = float(np.max(np.abs(joint_initial - source_scores.epig.to_numpy())))
        if family.startswith("gp_") and not np.allclose(joint_initial, sum_initial, rtol=1e-8, atol=1e-10):
            raise RuntimeError("independent GP joint MI differs from independent output sum")
        output.mkdir(parents=True, exist_ok=True)
        saved = {name: {k: v for k, v in result.items() if k != "initial_scores"}
                 for name, result in selections.items()}
        dump(output / "batches.json", saved)
        raw = source_scores[["sample_id", "canonical_index"]].copy()
        raw["joint_initial_epig"] = joint_initial
        raw["V1_initial_epig"] = selections["V1_only"]["initial_scores"]
        raw["V2_initial_epig"] = selections["V2_only"]["initial_scores"]
        raw.to_csv(output / "initial_scores.csv", index=False)
        result = {"seed": seed, "family": family, "distribution": "P1", "M": 1024,
                  "posterior_qualified": summary["posterior_qualified"],
                  "joint_vs_independent_sum_max_abs_score_error": independent_error,
                  "joint_score_recomputation_max_abs_error": recomputation_error,
                  "independent_sum_is_exact_model_identity": family.startswith("gp_"),
                  "V1_V2_greedy32_overlap": len(set(selections["V1_only"]["selected_positions"]) & set(selections["V2_only"]["selected_positions"])) / 32,
                  "joint_batch_information_nats_direct": joint_batch_information(posterior, references, joint_positions),
                  "joint_batch_information_nats_original": source_batches["epig_conditional_greedy"]["audit"]["conditional_batch_information_nats"]}
        for name, selection in selections.items():
            positions = selection["selected_positions"]
            result[f"{name}_joint32_overlap"] = len(set(positions) & set(joint_positions)) / 32
            result[f"{name}_batch_joint_information_nats"] = joint_batch_information(posterior, references, positions)
        if not np.isclose(result["joint_batch_information_nats_direct"], result["joint_batch_information_nats_original"], rtol=1e-8, atol=1e-9):
            raise RuntimeError("direct joint B32 information differs from source conditional chain")
        result.update({"seconds": time.perf_counter()-t0, "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       "new_model_fits": 0, "new_label_reads": 0,
                       "coregionalized_caveat": ("shared positive-definite B in latent and noise is removable by output whitening for full two-output MI; this is genuine but separable dependence" if family == "blr_coregionalized" else None)})
        dump(output / "summary.json", result)
        write_once(output / "complete.json", {"contract": contract,
            "files": {name: digest(output / name) for name in ("batches.json", "initial_scores.csv", "summary.json")}})
        rows.append(result)
        print(json.dumps({"seed":seed,"family":family,"seconds":result["seconds"]}),flush=True)
        del model, posterior, selections
        gc.collect()
    pd.DataFrame(rows).to_csv(APPENDIX / "summary.csv", index=False)
    elapsed = time.perf_counter()-started
    budget["consumed_seconds"] += elapsed
    budget["runs"].append({"finished_unix": time.time(), "seconds": elapsed, "completed_states": len(rows)})
    dump(budget_path,budget)
    dump(APPENDIX / "status.json", {"status": "COMPLETE_FOR_CURRENTLY_AVAILABLE_STATES" if len(rows)==len(candidates_dirs) else "BUDGET_LIMIT_PARTIAL",
                                     "completed_states":len(rows),"available_source_states":len(candidates_dirs),
                                     "consumed_seconds":budget["consumed_seconds"],"new_model_fits":0,"new_label_reads":0})


if __name__ == "__main__":
    main()
