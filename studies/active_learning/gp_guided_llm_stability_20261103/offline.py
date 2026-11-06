"""Read only historical train-pool artifacts; no runner, oracle, or API imports."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import lzma
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from gp import LinearGP, batch_information, fixed_kernel_signals, probability_metrics, rank_correlation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101"
GEOMETRY = ROOT / "studies/active_learning/nocw_free_selection_diagnostics_20261101/geometry_input_manifest.json.xz"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def upstream_stable_hash(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


class Reader:
    """Stable double reads, explicit paths, and immutable source hashes."""

    def __init__(self, lock=None):
        self.expected = None if lock is None else {f["path"]: f for f in lock["files"]}
        self.files = {}
        self.allowed = {GEOMETRY.resolve()}

    def allow_exact(self, paths):
        self.allowed.update(Path(p).resolve() for p in paths)

    def read(self, path, expected=None):
        path = Path(path).resolve()
        key = str(path.relative_to(ROOT))
        # Paths are enumerated from the fixed plan, never discovered by recursive data scans.
        if path not in self.allowed:
            raise ValueError("path outside offline read allowlist: " + key)
        if self.expected is not None and key not in self.expected:
            raise ValueError("source is not in frozen input manifest")
        before = path.stat()
        a = path.read_bytes()
        middle = path.stat()
        b = path.read_bytes()
        after = path.stat()
        signature = lambda s: (s.st_size, s.st_mtime_ns, s.st_ino)
        if a != b or signature(before) != signature(middle) or signature(middle) != signature(after):
            raise ValueError("source changed during read")
        digest = sha(a)
        if expected is not None and digest != expected:
            raise ValueError("upstream receipt hash mismatch: " + key)
        if self.expected is not None and digest != self.expected[key]["sha256"]:
            raise ValueError("frozen input drift: " + key)
        if key in self.files and digest != self.files[key]["sha256"]:
            raise ValueError("source changed across dependent reads")
        self.files[key] = {"path": key, "sha256": digest, "bytes": len(a),
                           "mtime_ns": before.st_mtime_ns, "stable_double_read": True}
        return a

    def json(self, path, expected=None):
        return json.loads(self.read(path, expected))


def load_seed(seed, plan, reader, geometry):
    selection = SOURCE / f"selections/seed_{seed}/free_cw_sol61_xhigh_rerun32"
    contract = reader.json(selection / "round_00/contract.json")
    if any(contract.get(k) != v for k,v in {"seed":seed,"round":0,"method":"free_cw_sol61_xhigh_rerun32",
                                           "variant":"cw_aware","dry_run":False}.items()):
        raise ValueError("initial contract identity or runtime status mismatch")
    catalog = reader.json(selection / "round_00/catalog.json", contract["catalog_sha256"])
    entry = next(e for e in geometry["batches"] if (e["method"],e["seed"],e["round"]) == ("free",seed,0))
    metadata = reader.json(ROOT / entry["metadata_path"], entry["metadata_sha256"])
    if metadata["sha256"] != entry["bank_sha256"]:
        raise ValueError("bank upstream manifest hash mismatch")
    if metadata["contract"]["checkpoint"] != contract["checkpoint_sha256"] or entry["state_checkpoint_sha256"] != contract["checkpoint_sha256"]:
        raise ValueError("historical bank does not match initial CW-Free checkpoint")
    if metadata["contract"]["indices"] != entry["canonical_indices"]:
        raise ValueError("canonical bank row mapping mismatch")
    ids = entry["row_ids"]
    l0, u0 = contract["L_t_ids"], contract["U_t_ids"]
    if len(l0) != 333 or len(ids) != 3330 or len(set(ids)) != len(ids) or ids != l0 + u0:
        raise ValueError("bank must have exact ordered initial L+U rows")
    if set(c["id"] for c in catalog["observed"]) != set(l0) or set(c["id"] for c in catalog["candidates"]) != set(u0):
        raise ValueError("catalog partition mismatch")
    for candidate in catalog["candidates"]:
        if any(k.startswith(("true_", "error_")) or k in ("V1_ml", "V2_ml") for k in candidate):
            raise ValueError("candidate target contamination")
    bank_bytes = reader.read(ROOT / entry["bank_path"], metadata["sha256"])
    with np.load(io.BytesIO(bank_bytes), allow_pickle=False) as bank:
        if set(bank.files) != {"raw", "cw"}:
            raise ValueError("unexpected feature bank fields")
        features = {name: bank[name].astype(np.float64) for name in plan["representations"]}
    observed = {c["id"]: [c["V1_ml"], c["V2_ml"]] for c in catalog["observed"]}
    initial_predictions = {c["id"]: [c["pred_V1_ml"], c["pred_V2_ml"]] for c in catalog["candidates"]}
    batches, provenance = [], []
    seen = set(l0)
    for r in plan["observed_rounds"]:
        directory = selection / f"round_{r:02d}"
        receipt = reader.json(directory / "label_access_receipt.json")
        audit = receipt["audit"]
        if not audit["acquisitions_frozen"] or audit["requested_rows"] != 32 or audit["roles"] != "u0":
            raise ValueError("missing already-observed acquisition receipt")
        frozen = reader.json(directory / "batch_freeze.json", receipt["batch_freeze_sha256"])
        # Production stores a canonical-JSON hash here, whereas the freeze uses a file-byte hash.
        feedback = reader.json(directory / "feedback.json")
        if upstream_stable_hash(feedback) != receipt["feedback_sha256"]:
            raise ValueError("feedback canonical-JSON receipt mismatch")
        for artifact in (frozen,feedback):
            if any(artifact.get(k) != v for k,v in {"seed":seed,"round":r,"method":"free_cw_sol61_xhigh_rerun32"}.items()):
                raise ValueError("acquisition state identity mismatch")
        if frozen.get("dry_run") is not False or frozen.get("pending_ids") != []:
            raise ValueError("only completed real Free32 proposals are permitted")
        if r == 0:
            for name in ("contract.json","catalog.json"):
                key=str((directory/name).relative_to(ROOT))
                if frozen["artifact_hashes"][name] != reader.files[key]["sha256"]:
                    raise ValueError("initial frozen artifact byte binding mismatch")
        if feedback["batch_freeze_sha256"] != receipt["batch_freeze_sha256"]:
            raise ValueError("feedback freeze binding mismatch")
        batch = frozen["batch_ids"]
        if audit["requested_ids_hash"] != upstream_stable_hash(sorted(batch)):
            raise ValueError("acquired ID receipt mismatch")
        if len(batch) != 32 or len(set(batch)) != 32 or seen.intersection(batch) or not set(batch) <= set(u0):
            raise ValueError("invalid historical acquisition partition")
        predictions = {c["candidate_id"]: c for c in frozen["premeasurement_predictions"]}
        rows = {c["candidate_id"]: c for c in feedback["records"]}
        if set(rows) != set(batch) or set(predictions) != set(batch):
            raise ValueError("feedback row mismatch")
        for sid in batch:
            row = rows[sid]
            actual = [row["true_V1_ml"], row["true_V2_ml"]]
            for j, endpoint in enumerate(("V1", "V2")):
                pred = row[f"pred_{endpoint}_ml"]
                if pred != predictions[sid][f"pred_{endpoint}_ml"] or not np.isclose(pred-actual[j], row[f"error_{endpoint}_ml"], atol=1e-6):
                    raise ValueError("premeasurement residual mismatch")
            observed[sid] = actual
            provenance.append({"seed": seed, "round": r, "candidate_id": sid,
                               "fixed_r0_checkpoint": contract["checkpoint_sha256"],
                               "native_prediction_checkpoint": frozen["checkpoint_sha256"],
                               "native_checkpoint_matches_fixed": frozen["checkpoint_sha256"] == contract["checkpoint_sha256"],
                               "target_role": "already_observed_u0_not_initial_training",
                               "native_residual_used_for_gp": False,
                               "fixed_r0_prediction_source": "round_00/catalog.json",
                               "feedback_source": str((directory / "feedback.json").relative_to(ROOT))})
        batches.append(batch)
        seen.update(batch)
    positions = {sid: i for i, sid in enumerate(ids)}
    acquired = sum(batches, [])
    if not all(np.isfinite(x).all() and x.shape == (3330,512) for x in features.values()):
        raise ValueError("invalid frozen representation")
    y0 = np.array([observed[sid] for sid in l0], dtype=np.float64)
    y = np.array([observed[sid] for sid in acquired], dtype=np.float64)
    pred = np.array([initial_predictions[sid] for sid in acquired], dtype=np.float64)
    return {"ids": ids, "l0": l0, "u0": u0, "features": features, "positions": positions,
            "batches": [[positions[sid] for sid in batch] for batch in batches], "y0": y0, "y": y,
            "pred": pred, "checkpoint": contract["checkpoint_sha256"], "provenance": provenance}


def evaluate_seed(seed, data, plan):
    results, predictions, fits, comparisons, information = [], [], [], [], []
    y0, y = data["y0"], data["y"]
    scales = y0.std(axis=0)
    means = y0.mean(axis=0)
    z0, z = (y0-means)/scales, (y-means)/scales
    loss = np.mean(((data["pred"] - y)/scales)**2, axis=1)
    risk_target = np.log1p(loss)
    acquired_positions = np.array(sum(data["batches"], []))
    risk_train = np.arange(96)
    risk_eval = np.arange(96,192)
    for space, raw in data["features"].items():
        # The feature normalization is learned from initial L only and remains fixed.
        rms = np.sqrt(np.mean(np.sum(raw[:333]**2, axis=1)))
        x = raw/rms
        fixed_variance, ivr_score = fixed_kernel_signals(raw,333)
        response_models = []
        for endpoint in range(2):
            all_mu, all_var, all_baseline_mu, all_baseline_var = [], [], [], []
            for r, batch in enumerate(data["batches"]):
                train_positions = np.r_[np.arange(333), acquired_positions[:32*r]]
                target = np.r_[z0[:,endpoint], z[:32*r,endpoint]]
                model = LinearGP().fit(x[train_positions], target, mean=0., bounds=tuple(plan["log_variance_bounds"]))
                fits.append({"seed":seed,"space":space,"target":f"response_V{endpoint+1}","evaluation_round":r, **model.audit})
                mu, latent = model.predict(x[batch])
                var = latent + model.noise
                all_mu.extend(mu); all_var.extend(var)
                all_baseline_mu.extend([0.]*32); all_baseline_var.extend([1.]*32)
                for k in range(32):
                    predictions.append({"seed":seed,"space":space,"target":f"response_V{endpoint+1}","round":r,
                                        "id":data["ids"][batch[k]], "target_standardized":float(z[32*r+k,endpoint]),
                                        "mean":float(mu[k]),"observation_variance":float(var[k]),
                                        "latent_variance":float(latent[k])})
                if r == 0:
                    response_models.append(model)
            results.append({"seed":seed,"space":space,"target":f"response_V{endpoint+1}","model":"gp",
                            **probability_metrics(z[:,endpoint], all_mu, all_var)})
            results.append({"seed":seed,"space":space,"target":f"response_V{endpoint+1}","model":"L0_constant",
                            **probability_metrics(z[:,endpoint],all_baseline_mu,all_baseline_var)})
        model = LinearGP().fit(x[acquired_positions[risk_train]], risk_target[risk_train],
                               bounds=tuple(plan["log_variance_bounds"]))
        fits.append({"seed":seed,"space":space,"target":"fixed_r0_log_risk","evaluation_round":"3..5",**model.audit})
        risk_mu, risk_latent = model.predict(x[acquired_positions[risk_eval]])
        risk_var = risk_latent+model.noise
        rho = rank_correlation(risk_mu, risk_target[risk_eval])
        results.append({"seed":seed,"space":space,"target":"fixed_r0_log_risk","model":"gp", "spearman":rho,
                        **probability_metrics(risk_target[risk_eval],risk_mu,risk_var)})
        baseline_mu = np.full(96,risk_target[risk_train].mean())
        baseline_var = np.full(96,risk_target[risk_train].var())
        results.append({"seed":seed,"space":space,"target":"fixed_r0_log_risk","model":"training_constant",
                        **probability_metrics(risk_target[risk_eval],baseline_mu,baseline_var)})
        for k,i in enumerate(risk_eval):
            predictions.append({"seed":seed,"space":space,"target":"fixed_r0_log_risk","round":int(i//32),
                                "id":data["ids"][acquired_positions[i]],"target_standardized":float(risk_target[i]),
                                "mean":float(risk_mu[k]),"observation_variance":float(risk_var[k]),
                                "latent_variance":float(risk_latent[k])})
        # Evaluate correlations at one fixed checkpoint, on X-only candidate pool.
        pool = x[333:]
        for endpoint, response in enumerate(response_models):
            gp_mean, gp_variance = response.predict(pool)
            comparisons.append({"seed":seed,"space":space,"endpoint":endpoint+1,
                "pool_rows":len(pool),"gp_variance_vs_unit_kernel_variance_spearman":rank_correlation(gp_variance,fixed_variance[333:]),
                "gp_variance_vs_fixed_ivr_score_spearman":rank_correlation(gp_variance,ivr_score[333:]),
                "gp_mean_vs_fixed_ivr_score_spearman":rank_correlation(gp_mean,ivr_score[333:]),
                "variance_vs_realized_fixed_r0_loss_spearman":rank_correlation(response.predict(x[acquired_positions])[1],loss),
                "fixed_kernel_unit_noise_is_calibrated":False})
            # A single posterior projection supports all predeclared batches without repeated N*N work.
            from scipy.linalg import solve_triangular
            cross = response.signal*(pool @ response.x.T)
            projected = solve_triangular(response.cholesky,cross.T,lower=True).T
            rng = np.random.default_rng(plan["random_batch_seed"]+seed)
            indices = [acquired_positions[:32]-333] + [rng.choice(len(pool),32,replace=False) for _ in range(plan["random_batch_count"])]
            values = []
            for ix in indices:
                covariance = response.signal*(pool[ix] @ pool[ix].T)-projected[ix] @ projected[ix].T
                covariance = (covariance+covariance.T)/2
                value,jitter = batch_information(covariance,response.noise)
                values.append(value)
            information.append({"seed":seed,"space":space,"endpoint":endpoint+1,
                                "historical_batch_cig":values[0],"random_cig_min":min(values[1:]),
                                "random_cig_median":float(np.median(values[1:])),"random_cig_max":max(values[1:]),
                                "historical_random_percentile":float(np.mean(np.array(values[1:])<=values[0])),
                                "random_batch_count":len(values)-1,"metric":"I(y_B; f_B | D_L0)",
                                "not_expected_QGeoGNN_error_reduction":True})
    return results,predictions,fits,comparisons,information


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--capture",action="store_true",help="Freeze inputs once; never overwrite a prior lock")
    args=parser.parse_args()
    plan_bytes=(HERE/"configs/offline_plan.json").read_bytes()
    plan=json.loads(plan_bytes)
    if (plan["observed_rounds"]!=list(range(6)) or plan["risk_training_rounds"]!=[0,1,2]
            or plan["risk_evaluation_rounds"]!=[3,4,5] or plan["batch_size"]!=32):
        raise ValueError("this frozen prototype supports only the declared six-batch chronological design")
    lock_path=HERE/"input_manifest.json"
    if args.capture and lock_path.exists():
        raise ValueError("input manifest already exists; use default hash-verified replay")
    lock=None if args.capture else json.loads(lock_path.read_text())
    if lock is not None and lock["offline_plan_sha256"]!=sha(plan_bytes):
        raise ValueError("offline plan drift")
    reader=Reader(lock)
    geometry=json.loads(lzma.decompress(reader.read(GEOMETRY)))
    for seed in plan["seeds"]:
        entry=next(e for e in geometry["batches"] if (e["method"],e["seed"],e["round"]) == ("free",seed,0))
        reader.allow_exact([ROOT/entry["metadata_path"],ROOT/entry["bank_path"]])
        base=SOURCE/f"selections/seed_{seed}/free_cw_sol61_xhigh_rerun32"
        reader.allow_exact([base/"round_00/catalog.json",base/"round_00/contract.json"])
        reader.allow_exact([base/f"round_{r:02d}"/name for r in plan["observed_rounds"]
                            for name in ("label_access_receipt.json","batch_freeze.json","feedback.json")])
    datasets={seed:load_seed(seed,plan,reader,geometry) for seed in plan["seeds"]}
    if args.capture:
        lock={"version":"phase0-input-v1","captured_at_utc":datetime.now(timezone.utc).isoformat(),
              "offline_plan_sha256":sha(plan_bytes), "seeds":plan["seeds"],
              "scope":"historically observed training pool only; no raw dataset or model prediction table reads",
              "files":list(reader.files.values())}
        dump(lock_path,lock)
    started=time.monotonic()
    all_results,all_predictions,all_fits,all_comparisons,all_information=[],[],[],[],[]
    for seed,data in datasets.items():
        print(f"Evaluating frozen seed {seed}",flush=True)
        for dest,values in zip((all_results,all_predictions,all_fits,all_comparisons,all_information),evaluate_seed(seed,data,plan)):
            dest.extend(values)
    output=HERE/"results";output.mkdir(exist_ok=True)
    dump(output/"metrics.json",all_results);dump(output/"fit_audits.json",all_fits)
    dump(output/"signal_comparison.json",all_comparisons);dump(output/"batch_information.json",all_information)
    write_csv(output/"predictions.csv",all_predictions)
    write_csv(output/"residual_provenance.csv",sum([d["provenance"] for d in datasets.values()],[]))
    checks=[]
    screen=plan["development_reliability_screen"]
    for space in plan["representations"]:
        for seed in plan["seeds"]:
            subset=[r for r in all_results if r["space"]==space and r["seed"]==seed]
            response_ok=True
            for target in ("response_V1","response_V2"):
                a=next(r for r in subset if r["target"]==target and r["model"]=="gp")
                b=next(r for r in subset if r["target"]==target and r["model"]=="L0_constant")
                response_ok &= a["nlpd"]<=b["nlpd"] and abs(a["coverage_90"]-.9)<=screen["response_90_coverage_absolute_gap_max_each_endpoint_seed"]
            a=next(r for r in subset if r["target"]=="fixed_r0_log_risk" and r["model"]=="gp")
            b=next(r for r in subset if r["target"]=="fixed_r0_log_risk" and r["model"]=="training_constant")
            risk_ok=a["spearman"] is not None and a["spearman"]>=screen["risk_spearman_min_each_seed"] and a["nlpd"]<b["nlpd"]
            checks.append({"seed":seed,"space":space,"response_screen_pass":bool(response_ok),"risk_screen_pass":bool(risk_ok),
                           "current_checkpoint_risk_transport_validated":False,"formal_guidance_eligible":False})
    verification={"scope":"Phase0 completed; Phase1 mock preparation only", "elapsed_seconds":time.monotonic()-started,
                  "files_verified":len(reader.files),"source_checks_passed":True,
                  "historical_seeds":plan["seeds"],"initial_labels_per_seed":333,
                  "response_forward_evaluation_per_seed":192,"risk_training_per_seed":96,"risk_evaluation_per_seed":96,
                  "screen":checks,"formal_GP_guidance_eligible":False,
                  "risk_current_checkpoint_transport":"not_validated",
                  "all_fits_successful":all(f["optimizer_success"] for f in all_fits),
                  "boundary_fit_count":sum(f["boundary_hit"] for f in all_fits),"fit_count":len(all_fits),
                  "maximum_jitter":max(f["numerical_jitter"] for f in all_fits),
                  "actions":{"paid_llm_calls":0,"QGeoGNN_training_runs":0,"new_labels_revealed":0,
                             "test_truth_reads":0,"checkpoint_loads":0,"raw_dataset_reads":0},
                  "source_manifest_sha256":sha(lock_path.read_bytes()),"offline_plan_sha256":sha(plan_bytes)}
    dump(output/"verification.json",verification)
    print(json.dumps(verification,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
