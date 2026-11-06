"""Target-safe paired geometry and train-only fusion transforms."""
from __future__ import annotations

from collections import defaultdict
import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
_path = ROOT / "studies/active_learning/nocw_free_selection_diagnostics_20261101/condition_metrics.py"
_spec = importlib.util.spec_from_file_location("representation_condition_metrics", _path)
_conditions = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conditions)


def fixed_candidate_positions(ids, size=256, seed=0):
    """Hash-sort legal IDs without using predictions or response labels."""
    import hashlib
    return np.array(sorted(range(len(ids)), key=lambda i: (
        hashlib.sha256(f"representation_gp_v1:{seed}:{ids[i]}".encode()).digest(), str(ids[i])))[:size], dtype=int)


def strict_pairs(records):
    """Reuse the audited exact Decimal/Fraction condition semantics."""
    keys = [_conditions._canonical_conditions(r) for r in records]
    if any(k is None for k in keys):
        raise ValueError("invalid exact condition metadata")
    for axis in ("sample_volume", "pe", "loading_volume"):
        field = list(_conditions.FACTORS).index(axis)
        groups = defaultdict(list)
        for i, (r, key) in enumerate(zip(records, keys)):
            groups[(r["smiles"],) + key[:field] + key[field + 1:]].append(i)
        pairs = [(a, b) for group in groups.values() for a, b in combinations(group, 2)
                 if keys[a][field] != keys[b][field]]
        yield axis, np.asarray(pairs, dtype=int).reshape(-1, 2)


def contrast_rows(records, features, targets, means, qgnn, train_features):
    """Both members must be held out from the current GP fit by the caller."""
    scale = np.sqrt(np.mean(np.sum((train_features - train_features.mean(0)) ** 2, axis=1)))
    scale = max(float(scale), np.finfo(float).eps)
    rows = []
    for axis, pairs in strict_pairs(records):
        for a, b in pairs:
            distance = float(np.linalg.norm(features[a] - features[b]) / scale)
            for j in range(2):
                dy = float(targets[b, j] - targets[a, j])
                dp = float(means[b, j] - means[a, j])
                dq = float(qgnn[b, j] - qgnn[a, j])
                rows.append({"axis": axis, "id_a": records[a]["id"], "id_b": records[b]["id"],
                             "endpoint": f"V{j+1}", "distance_train_rms": distance,
                             "true_delta_ml": dy, "gp_delta_ml": dp, "gnn_delta_ml": dq,
                             "gp_contrast_absolute_error_ml": abs(dy-dp),
                             "gnn_contrast_absolute_error_ml": abs(dy-dq)})
    return rows


def train_pca(values, train_count, dimension=64):
    """Fit all scaling and PCA on L_t; retain no response inputs."""
    x = np.asarray(values, dtype=np.float64)
    train = x[:train_count]
    mean, std = train.mean(0), train.std(0)
    std = np.where(std > 1e-12, std, 1.)
    normalized = (x-mean)/std
    _, singular, vt = np.linalg.svd(normalized[:train_count], full_matrices=False)
    k = min(dimension, len(vt))
    result = normalized @ vt[:k].T
    if k < dimension:
        result = np.pad(result, ((0,0),(0,dimension-k)))
    return result, {"fit_rows": train_count, "output_dimension": dimension,
                    "retained_variance_fraction": float((singular[:k]**2).sum()/max((singular**2).sum(),1e-30)),
                    "fit_role": "L_t_X_only", "whiten": False}


def fusion_features(gnn, llm, train_count):
    a, aa = train_pca(gnn, train_count)
    b, ba = train_pca(llm, train_count)
    # Equal block RMS prevents arbitrary encoder units from controlling fusion.
    for block in (a, b):
        rms = np.sqrt(np.mean(np.sum(block[:train_count] ** 2, axis=1)))
        block /= max(float(rms), 1e-12)
    return np.concatenate([a,b],axis=1), {"gnn": aa, "llm": ba, "output_dimension": 128,
        "projection_parameters": 128*64+64, "block_normalization": "L_t_RMS"}


def template_diagnostics(a, b, train_count, candidate_slice, models_a, models_b=None):
    """Use the A-fitted model to isolate semantic-equivalent template changes."""
    from scipy.spatial.distance import cdist
    means = a[:train_count].mean(0)
    scale = np.sqrt(np.mean(np.sum((a[:train_count]-means)**2,axis=1)))
    ua, ub = a[candidate_slice], b[candidate_slice]
    da, db = cdist(ua,ua), cdist(ub,ub)
    np.fill_diagonal(da,np.inf); np.fill_diagonal(db,np.inf)
    k = min(10,len(ua)-1)
    na, nb = np.argsort(da,axis=1)[:,:k], np.argsort(db,axis=1)[:,:k]
    result = {"candidate_rows":len(ua), "template_distance_train_rms_mean":float(np.mean(np.linalg.norm(ua-ub,axis=1))/max(scale,1e-12)),
              "nearest_10_overlap_mean":float(np.mean([len(set(x)&set(y))/k for x,y in zip(na,nb)])),
              "model_policy":"fit_template_A_only_apply_A_and_B"}
    for j,model in enumerate(models_a):
        ma,va=model.predict(ua);mb,vb=model.predict(ub)
        rho = None if np.ptp(va)==0 or np.ptp(vb)==0 else float(spearmanr(va,vb).statistic)
        result[f"V{j+1}_mean_absolute_change_ml"]=float(np.mean(np.abs(ma-mb)))
        result[f"V{j+1}_variance_rank_spearman"]=rho
        result[f"V{j+1}_maxvar_top32_overlap"]=len(set(np.argsort(va)[-32:])&set(np.argsort(vb)[-32:]))/32
        result[f"V{j+1}_maxvar_absolute_change_mean"]=float(np.mean(np.abs(va-vb)))
    return result
