#!/usr/bin/env python3
"""Filtered FULL-data ROW audit: balanced FULL128 and train-only column shrinkage."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.studies.run_hier_cw_semantic_repair as hier
import scripts.studies.run_full_data_baseline_finalization as baseline
from src.qgeognn_al.transfer.full_data import sha256_file, stable_hash, ea_fraction
from src.qgeognn_al.transfer.evaluation import absolute_error_metrics
from src.qgeognn_al.transfer.hierarchical_cw_corrected import fit_corrected_joint_full128, fit_hierarchical_cw_endpoint_aligned_corrected

STUDY = ROOT / "studies/transfer/row_column_selective_latent_audit"
BASE = ROOT / "studies/transfer/full_data_baseline_finalization"
ARCH = ROOT / "studies/transfer/full_data_architecture_headroom"
COLS = ("25g", "40g")
SEEDS = baseline.SEEDS
RATIOS = {"25g": 6.25, "40g": 10.0}
LAMBDA_GRID = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)
LATENT_GRID = (1.0, 10.0, 100.0, 1000.0)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
METHODS = ("OLD_HIER_REFERENCE", "HIER_CW_SHARED_LAMBDA_CORRECTED", "CORRECTED_JOINT_FULL128",
           "BALANCED_JOINT_FULL128", "A2_HIER_SHRINK_ORIGINAL", "A3_HIER_SHRINK_BALANCED",
           "paper_style_current_v2")

def write_json(p, x): p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(x, indent=2, sort_keys=True) + "\n")
def write_csv(df, p): p.parent.mkdir(parents=True, exist_ok=True); df.to_csv(p, index=False)
def positions(frame, ids):
    lookup = {str(v): i for i, v in enumerate(frame.sample_id.astype(str))}
    return np.asarray([lookup[str(v)] for v in ids], dtype=int)
def folds(n, seed, k=5): return list(KFold(n_splits=min(k, n), shuffle=True, random_state=int(seed)).split(np.arange(n)))
def balanced_score(y, p, labels, scales):
    e = (y - p) / scales
    vals = [np.mean(e[np.asarray(labels) == c, j] ** 2) for c in COLS for j in range(2)]
    return float(np.sqrt(np.mean(vals)))
def fit_model(data, train, kind, seed, balanced=False, inner=True):
    """Select every lambda using only train and its inner folds, then refit train."""
    y, labels = data["truth"], data["labels"]
    endpoint_key = (0.0, 0.0)
    if kind == "J":
        _, endpoint_key, _ = fit_model(data, train, "H", seed + 313, balanced, inner)
    candidates = [(a, b) for a in LAMBDA_GRID for b in LAMBDA_GRID]
    if kind in ("H", "J"):
        if kind == "H": candidates = [(x, x) for x in LAMBDA_GRID]
    if kind == "J": candidates = [(a, b) for a in LATENT_GRID for b in LATENT_GRID]
    score_rows = []
    split_iter = folds(len(train), seed + 991, 3) if inner and len(train) >= 9 else [(np.arange(len(train)), np.arange(len(train)))]
    for key in candidates:
        vals = []
        for tr_rel, va_rel in split_iter:
            tr, va = np.asarray(train)[tr_rel], np.asarray(train)[va_rel]
            scales = np.array([np.std(y[tr][np.asarray(labels)[tr] == c, j]) for c in COLS for j in range(2)]).reshape(2,2)
            scales = np.where(scales < 1e-10, 1.0, scales)
            weights = None
            if balanced: weights = np.column_stack([1.0 / scales[np.asarray([COLS.index(v) for v in labels[tr]]), j] ** 2 for j in range(2)])
            if kind == "H":
                fit = fit_hierarchical_cw_endpoint_aligned_corrected(data["source"][tr], y[tr], data["ea"][tr], labels[tr], COLS, mass_ratios=RATIOS, base_penalty=.1, lambda_center_deviation=key[0], lambda_width_deviation=key[1], endpoint_scale=np.ones(2), row_weights=weights)
                pred = fit.predict(data["source"][va], data["ea"][va], labels[va])
            else:
                fit = fit_corrected_joint_full128(data["source"][tr], y[tr], data["ea"][tr], data["latent"][tr], labels[tr], COLS, mass_ratios=RATIOS, base_penalty=.1, lambda_center_deviation=endpoint_key[0], lambda_width_deviation=endpoint_key[1], lambda_center_latent=key[0], lambda_width_latent=key[1], endpoint_scale=np.ones(2), row_weights=weights)
                pred = fit.predict(data["source"][va], data["ea"][va], labels[va], data["latent"][va])
            vals.append(balanced_score(y[va], pred, labels[va], np.ones((len(va),2))))
        score_rows.append((float(np.mean(vals)), key))
    selected = min(score_rows, key=lambda x: (x[0], x[1]))[1]
    scales = np.array([np.std(y[train][np.asarray(labels)[train] == c, j]) for c in COLS for j in range(2)]).reshape(2,2)
    scales = np.where(scales < 1e-10, 1.0, scales)
    weights = None
    if balanced: weights = np.column_stack([1.0 / scales[np.asarray([COLS.index(v) for v in labels[train]]), j] ** 2 for j in range(2)])
    if kind == "H":
        fit = fit_hierarchical_cw_endpoint_aligned_corrected(data["source"][train], y[train], data["ea"][train], labels[train], COLS, mass_ratios=RATIOS, base_penalty=.1, lambda_center_deviation=selected[0], lambda_width_deviation=selected[1], endpoint_scale=np.ones(2), row_weights=weights)
    else:
        fit = fit_corrected_joint_full128(data["source"][train], y[train], data["ea"][train], data["latent"][train], labels[train], COLS, mass_ratios=RATIOS, base_penalty=.1, lambda_center_deviation=endpoint_key[0], lambda_width_deviation=endpoint_key[1], lambda_center_latent=selected[0], lambda_width_latent=selected[1], endpoint_scale=np.ones(2), row_weights=weights)
    return fit, selected, score_rows

def load_data(seed):
    features = {c: pd.read_csv(baseline._features_path(c)) for c in COLS}
    sources = {c: baseline._source_frame(c)[["V1_source", "V2_source"]].to_numpy(float) for c in COLS}
    latents = {c: np.load(ARCH / f"runtime/latent_{c}.npz", allow_pickle=False)["latent"] for c in COLS}
    ids, src, truth, ea, latent, labels = [], [], [], [], [], []
    for c in COLS:
        x = baseline._ids(c, "row", seed, "gradient_train"); ix = positions(features[c], x)
        ids += x; src.append(sources[c][ix]); truth.append(baseline._read_authorized_truth(baseline._canonical_path(c), x)); ea.append(ea_fraction(features[c]["PE/EA"])[ix]); latent.append(latents[c][ix]); labels += [c] * len(x)
    return {"ids": ids, "source": np.vstack(src), "truth": np.vstack(truth), "ea": np.concatenate(ea), "latent": np.vstack(latent), "labels": np.asarray(labels)}

def ref_predictions(c, seed, ids):
    p = pd.read_csv(hier._context_dir(c, "row", seed) / "predictions_blind.csv.gz").set_index("sample_id").loc[ids]
    return {m: p[[f"{m}_V1", f"{m}_V2"]].to_numpy(float) for m in ("OLD_HIER_REFERENCE", "HIER_CW_SHARED_LAMBDA_CORRECTED", "CORRECTED_JOINT_FULL128", "paper_style_current_v2")}

def test_arrays(c, seed):
    frame = pd.read_csv(baseline._features_path(c)); ids = baseline._ids(c, "row", seed, "test"); ix = positions(frame, ids)
    source = baseline._source_frame(c)[["V1_source", "V2_source"]].to_numpy(float)[ix]
    return ids, source, ea_fraction(frame["PE/EA"])[ix], np.load(ARCH / f"runtime/latent_{c}.npz", allow_pickle=False)["latent"][ix]

def run_fit():
    STUDY.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(BASE / "split_manifest.csv"); write_csv(split[split.protocol.eq("row")], STUDY / "split_manifest.csv")
    write_json(STUDY / "PROTOCOL.json", {"protocol":"filtered FULL-data ROW", "columns":list(COLS), "seeds":list(SEEDS), "alpha_grid":list(ALPHAS), "latent_grid":list(LATENT_GRID), "test_truth_used_for_fit":False, "source_checkpoint_sha256":baseline.SOURCE_SHA256, "parent_split_sha256":sha256_file(BASE / "split_manifest.csv")})
    alpha_rows, inner_rows, pred_files = [], [], []
    for seed in SEEDS:
        data = load_data(seed); n = len(data["truth"]); tr = np.arange(n)
        hfit, hkey, hgrid = fit_model(data, tr, "H", seed, False); jfit, jkey, jgrid = fit_model(data, tr, "J", seed, False); bfit, bkey, bgrid = fit_model(data, tr, "J", seed, True)
        # OOF predictions: each fold refits both bases and selects their hyperparameters inside that fold.
        oof = {"original": np.full((n,2), np.nan), "balanced": np.full((n,2), np.nan), "h": np.full((n,2), np.nan)}
        for fold, (a,b) in enumerate(folds(n, seed)):
            hf, hk, hg = fit_model(data, a, "H", seed + fold, False); jf, jk, jg = fit_model(data, a, "J", seed + fold, False); bf, bk, bg = fit_model(data, a, "J", seed + fold, True)
            oof["h"][b] = hf.predict(data["source"][b], data["ea"][b], data["labels"][b]); oof["original"][b] = jf.predict(data["source"][b], data["ea"][b], data["labels"][b], data["latent"][b]); oof["balanced"][b] = bf.predict(data["source"][b], data["ea"][b], data["labels"][b], data["latent"][b])
            inner_rows.append({"seed":seed,"fold":fold,"n_train":len(a),"n_valid":len(b),"H_lambda":hk[0],"J_latent_lambda_C":jk[0],"J_latent_lambda_W":jk[1],"BAL_latent_lambda_C":bk[0],"BAL_latent_lambda_W":bk[1]})
        test_preds = {}
        for c in COLS:
            ids, test_source, test_ea, test_latent = test_arrays(c, seed)
            base = ref_predictions(c, seed, ids); local = {k:v for k,v in base.items()}
            # final balanced fit prediction for test; original is already frozen from semantic-repair.
            local["BALANCED_JOINT_FULL128"] = bfit.predict(test_source, test_ea, np.repeat(c,len(ids)), test_latent)
            # OOF alpha objective, shared by both endpoints, column-specific and MAE-guarded.
            train_c = np.flatnonzero(data["labels"] == c); scales = np.std(data["truth"][train_c], axis=0); scales=np.where(scales<1e-10,1,scales); truth_o=data["truth"][train_c]
            for name, jpred in (("original",oof["original"]),("balanced",oof["balanced"])):
                hp, jp = oof["h"][train_c], jpred[train_c]; best=None
                h_mae=float(np.mean(np.abs(truth_o-hp)))
                for alpha in ALPHAS:
                    pp=hp+alpha*(jp-hp)
                    if not np.isfinite(pp).all(): raise RuntimeError("non-finite OOF prediction")
                    rm=float(np.sqrt(np.mean((truth_o-pp)**2))); mae=float(np.mean(np.abs(truth_o-pp)))
                    row={"seed":seed,"column":c,"candidate":name,"alpha":alpha,"balanced_rmse":rm,"mae":mae}; alpha_rows.append(row)
                    row["mae_guard_limit"] = 1.02 * h_mae
                    row["mae_guard_pass"] = bool(mae <= 1.02 * h_mae + 1e-12)
                    if row["mae_guard_pass"] and (best is None or (rm,mae,alpha)<(best["balanced_rmse"],best["mae"],best["alpha"])): best=row
                if best is None:
                    best=min([r for r in alpha_rows if r["seed"]==seed and r["column"]==c and r["candidate"]==name], key=lambda r:(r["mae"],r["alpha"]))
                local["A2_HIER_SHRINK_ORIGINAL" if name=="original" else "A3_HIER_SHRINK_BALANCED"] = local["HIER_CW_SHARED_LAMBDA_CORRECTED"] + best["alpha"]*(local["CORRECTED_JOINT_FULL128" if name=="original" else "BALANCED_JOINT_FULL128"]-local["HIER_CW_SHARED_LAMBDA_CORRECTED"])
            out=pd.DataFrame({"sample_id":ids});
            for m in METHODS:
                out[f"{m}_V1"] = local[m][:,0]; out[f"{m}_V2"] = local[m][:,1]
            path=STUDY/f"runtime/{c}/row/seed_{seed}/predictions_blind.csv.gz"; write_csv(out,path); pred_files.append(str(path.relative_to(ROOT)))
    write_csv(pd.DataFrame(alpha_rows), STUDY/"alpha_selection.csv"); write_csv(pd.DataFrame(inner_rows), STUDY/"inner_cv_selection.csv")
    manifest={"status":"FROZEN_BEFORE_TEST_EVALUATION","files":{p:sha256_file(ROOT/p) for p in pred_files},"source_checkpoint_sha256":baseline.SOURCE_SHA256,"split_manifest_sha256":sha256_file(STUDY/"split_manifest.csv")}; write_json(STUDY/"artifact_manifest.json",manifest)

def score():
    rows=[]; tail=[]
    for c in COLS:
      for seed in SEEDS:
        ids=baseline._ids(c,"row",seed,"test"); truth=baseline._read_authorized_truth(baseline._canonical_path(c),ids); f=pd.read_csv(STUDY/f"runtime/{c}/row/seed_{seed}/predictions_blind.csv.gz")
        for m in METHODS:
          p=f[[f"{m}_V1",f"{m}_V2"]].to_numpy(float); rows.append({"column":c,"protocol":"row","seed":seed,"method":m,**absolute_error_metrics(truth,p,baseline.SOURCE_SCALES)})
          feat=pd.read_csv(baseline._canonical_path(c)).set_index("sample_id").loc[ids]; vol=feat[["V1_ml","V2_ml"]].to_numpy(float)
          for j,e in enumerate(("V1","V2")):
            err=(truth[:,j]-p[:,j])**2; order=np.argsort(vol[:,j]); q=np.array_split(order,5)
            for k,g in enumerate(q,1): tail.append({"column":c,"seed":seed,"method":m,"endpoint":e,"retention_quintile":k,"n":len(g),"rmse":float(np.sqrt(np.mean((truth[g,j]-p[g,j])**2))),"mae":float(np.mean(np.abs(truth[g,j]-p[g,j]))),"mean_signed_error":float(np.mean(p[g,j]-truth[g,j])),"sse_share":float(err[g].sum()/err.sum()),"top10_retention_sse_share":float(err[order[:max(1,int(np.ceil(.10*len(order))))]].sum()/err.sum()),"top20_retention_sse_share":float(err[order[:max(1,int(np.ceil(.20*len(order))))]].sum()/err.sum())})
    metrics=pd.DataFrame(rows); write_csv(metrics,STUDY/"per_seed_metrics.csv"); write_csv(pd.DataFrame(tail),STUDY/"tail_sse_diagnostic.csv")
    summary=metrics.groupby(["column","method"])[["V1_rmse","V1_mae","V1_r2","V2_rmse","V2_mae","V2_r2","combined_normalized_rmse"]].agg(["mean","std"]).reset_index(); write_csv(summary,STUDY/"summary.csv")
    alpha=pd.read_csv(STUDY/"alpha_selection.csv"); selected=alpha.sort_values(["seed","column","candidate","balanced_rmse","mae","alpha"]).groupby(["seed","column","candidate"],as_index=False).first(); write_csv(selected,STUDY/"alpha_selection_selected.csv")
    report="# Final report: row column-selective latent audit\n\nThis study froze the filtered FULL-data ROW protocol and reused the source checkpoint, row identities, and corrected HIER/FULL128 implementations. A1 changes only the train-fold objective to equalize the four column×endpoint tasks. A2/A3 select one shared endpoint alpha per column from train-only OOF predictions. No compound split, Active Learning, backbone change, new physical feature, or test-guided choice was used.\n\n## Results\n\n"+summary.to_string(index=False)+"\n\n## Alpha selections\n\n"+selected.to_string(index=False)+"\n\nInterpretation must use the complete per-seed metrics and endpoint rows; no aggregate score is used to hide endpoint regressions. The retention-tail table reports filtered test strata and is descriptive only.\n"
    (STUDY/"FINAL_REPORT.md").write_text(report)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--stage",choices=("fit","score","all"),default="all"); a=ap.parse_args();
    if a.stage in ("fit","all"): run_fit()
    if a.stage in ("score","all"): score()
if __name__ == "__main__": main()
