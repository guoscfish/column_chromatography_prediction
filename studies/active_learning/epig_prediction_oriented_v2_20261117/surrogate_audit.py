"""Bounded post-stage1/stage2 feature/probability development diagnostics.

Only feature columns and seed-local original L333 targets are read. New
configurations are sealed before the first label read/fit; old results remain
unchanged. Every endpoint optimization restart and model fit is accounted for.
"""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[key]="1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
os.environ.setdefault("MPLCONFIGDIR","/tmp/epig_v2_surrogate_mpl")
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtri
from scipy.stats import kurtosis, spearmanr
from qgeognn_al.active_learning_v2.benchmark_protocol import load_features, FEATURE_COLUMNS
from qgeognn_al.active_learning_v2.protocol import RestrictedLabelStore, ids_hash, validate_row_protocol
from qgeognn_al.active_learning_v2.epig_models import FeatureTransform, gaussian_probability_metrics
from qgeognn_al.resources import SOURCE_DATA, SOURCE_GRAPH_CACHE
from surrogate_v2 import StudyGP

OLD=ROOT/"studies/active_learning/epig_prediction_oriented_20261117"
BASE=ROOT/"studies/active_learning/qgeognn_v2_row_sequential_b32"
OUT=HERE/"surrogate_evidence"
SEEDS=(157,887,2357,6101,12203)
CONFIGS={"F0_PCA16_opt":{"feature":"F0","grouped":False,"max_components":16},
         "F1_PCA16_opt":{"feature":"F1","grouped":False,"max_components":16},
         "F2_PCA16_opt":{"feature":"F1","grouped":True,"max_components":16},
         "F2_full_opt":{"feature":"F1","grouped":True,"max_components":None}}


def read(p): return json.loads(Path(p).read_text())
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+".tmp");tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+"\n");tmp.replace(p)
def once(p,v):
    p=Path(p)
    if p.exists():
        if read(p)!=v: raise RuntimeError(f"immutable contract drift: {p}")
    else:dump(p,v)
def arrayhash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def freeze():
    protocol={"schema":"epig-v2-surrogate-diagnostic-1","study_type":"post-stage1/stage2 exploratory development study",
              "seeds":list(SEEDS),"folds":3,"fold_seed":"outer_seed+910007, same folds as v1",
              "configs":CONFIGS,"kernel":"matern32","maxiter":200,"restarts":3,
              "mean":"constant training-side empirical mean; existing BLR linear mean is audited, not changed here",
              "noise":"independent endpoint homoscedastic Gaussian; residual conditional structure diagnosed, not assumed fixed",
              "group_scaling":"after fold-local z standardization, molecular16/eluent6/solvent3/loading6 each weighted 1/sqrt(group_dim), then optional PCA",
              "label_access":"seed-local original L333 only; no future U, validation, or test labels",
              "model_selection":"none; all four finite variants reported, not adopted by primary AL",
              "budget":{"model_fits_max":60,"endpoint_optimization_starts_max":360,"active_seconds_max":1800,"automatic_retries":0,"threads":1},
              "condition_groups":"PE/EA EA fraction, density, sample-volume, loading-solvent-volume quartiles with duplicate cuts dropped; solvent categories; sigma quartiles; molecule with n>=4",
              "cross_sample_logscore":"each held-out row conditional on other held-out rows, both endpoints kept together; diagnostic only, L333 known labels; no future-label acquisition utility",
              "coverage_uncertainty":"descriptive Wilson95 intervals; OOF row dependence and outer-seed overlap preclude independent-binomial guarantees",
              "files":{str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),HERE/"surrogate_v2.py",ROOT/"src/qgeognn_al/active_learning_v2/epig_models.py")},
              "source_sha256":digest(SOURCE_DATA),"graph_cache_sha256":digest(SOURCE_GRAPH_CACHE),
              "original_freeze_sha256":digest(OLD/"freeze.json")}
    once(HERE/"surrogate_protocol.json",protocol)
    return protocol


def inputs():
    import torch
    from qgeognn_al.data import eluent_descriptor
    frame=load_features()
    if set(frame)!=set(FEATURE_COLUMNS):raise PermissionError("feature schema drift")
    graphs=torch.load(SOURCE_GRAPH_CACHE,weights_only=False)
    molecular=np.vstack([graphs[s]["descriptor"] for s in frame.canonical_smiles]).astype(float)
    eluents=np.vstack([eluent_descriptor(v) for v in frame["PE/EA"]]).astype(float)
    solvent=np.column_stack([frame["loading solvent"].eq(s).to_numpy(float) for s in ("PE","EA","DCM")])
    density=frame["Density g/ml"].to_numpy(float);volume=frame["V/ul"].to_numpy(float)
    amount=density*volume;loading=frame["Volume of loading solvent/ul"].to_numpy(float)
    x0=np.column_stack([molecular,eluents,solvent,amount,loading,np.log1p(amount),np.log1p(loading)])
    x1=np.column_stack([x0,density,volume])
    if x0.shape!=(4163,29) or x1.shape!=(4163,31) or not np.isfinite(x1).all():raise ValueError("feature drift")
    names=[f"molecular_{i}" for i in range(16)]+[f"eluent_{i}" for i in range(6)]+["solvent_PE","solvent_EA","solvent_DCM","amount_density_x_ul","loading_solvent_ul","log1p_amount","log1p_loading_solvent","density_g_per_ml","sample_volume_ul"]
    once(OUT/"features.json",{"F0_shape":list(x0.shape),"F1_shape":list(x1.shape),"F0_sha256":arrayhash(x0),"F1_sha256":arrayhash(x1),"F1_names":names,"labels_read":0})
    # Positive finite parsing is mandatory. Ratios are normalized to EA fraction.
    def fraction(value):
        parts=str(value).split("/")
        if len(parts)!=2:raise ValueError("invalid PE/EA")
        pe,ea=map(float,parts)
        if pe<0 or ea<0 or pe+ea<=0:raise ValueError("invalid ratio")
        return ea/(pe+ea)
    frame["EA_fraction"]=frame["PE/EA"].map(fraction)
    return frame,{"F0":x0,"F1":x1},names


def wilson(k,n):
    z=ndtri(.975);p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return float(c-h),float(c+h)


def nll(y,mean,cov):
    e=(y-mean).ravel();factor=cho_factor(cov,lower=True,check_finite=False)
    return float(.5*(e@cho_solve(factor,e,check_finite=False)+2*np.log(np.diag(factor[0])).sum()+len(e)*np.log(2*np.pi))/len(e))


def fold_diagnostics(y,mean,cov):
    n,q=y.shape;error=(y-mean).ravel();factor=cho_factor(cov,lower=True)
    precision=cho_solve(factor,np.eye(n*q));pe=precision@error
    conditional=[];unconditional=[]
    for i in range(n):
        sl=slice(i*q,(i+1)*q);p=precision[sl,sl];f=cho_factor(p,lower=True)
        conditional_error=cho_solve(f,pe[sl]);cond_cov=cho_solve(f,np.eye(q))
        conditional.append(nll(conditional_error[None,:],np.zeros((1,q)),cond_cov))
        unconditional.append(nll(y[i:i+1],mean[i:i+1],cov[sl,sl]))
    return {"full_joint_nll_per_scalar":nll(y,mean,cov),"diagonal_nll_per_scalar":nll(y,mean,np.diag(np.diag(cov))),
            "block_diagonal_nll_per_scalar":float(np.mean(unconditional)),
            "cross_sample_conditional_nll_per_scalar":float(np.mean(conditional)),
            "conditional_minus_unconditional":float(np.mean(conditional)-np.mean(unconditional)),
            "minimum_eigenvalue":float(np.linalg.eigvalsh(cov)[0]),"maximum_asymmetry":float(np.abs(cov-cov.T).max()),
            "conditional_row_nll":conditional,"unconditional_row_nll":unconditional}


def residual_diagnostics(y,mean,blocks,frame,seed,config):
    sigma=np.sqrt(np.diagonal(blocks,axis1=1,axis2=2));z=(y-mean)/sigma
    rows=[];groups=[]
    for j,endpoint in enumerate(("V1","V2")):
        covered80=np.abs(z[:,j])<=ndtri(.9);covered95=np.abs(z[:,j])<=ndtri(.975)
        k=int(covered80.sum());lo,hi=wilson(k,len(z))
        rows.append({"seed":seed,"config":config,"endpoint":endpoint,"rows":len(z),"coverage80_count":k,
                     "coverage80":k/len(z),"coverage80_wilson95_low":lo,"coverage80_wilson95_high":hi,
                     "distance_from_old_upper_count":k-299,"coverage95":float(covered95.mean()),
                     "residual_rms":float(np.sqrt(np.mean(z[:,j]**2))),"z_abs_max":float(np.abs(z[:,j]).max()),
                     "z_excess_kurtosis":float(kurtosis(z[:,j],fisher=True,bias=False)),
                     "largest_four_square_fraction":float(np.sort(z[:,j]**2)[-4:].sum()/(z[:,j]**2).sum()),
                     "sigma_error_spearman":float(spearmanr(sigma[:,j],np.abs(y[:,j]-mean[:,j])).statistic)})
        factors={"sigma_quartile":pd.qcut(pd.Series(sigma[:,j]),4,duplicates="drop").astype(str),
                 "solvent":frame["loading solvent"].astype(str).reset_index(drop=True)}
        for col in ("Density g/ml","V/ul","Volume of loading solvent/ul","EA_fraction"):
            factors[col]=pd.qcut(frame[col].reset_index(drop=True),4,duplicates="drop").astype(str)
        molecules=frame.canonical_smiles.reset_index(drop=True);counts=molecules.value_counts()
        factors["molecule_n4"]=molecules.where(molecules.map(counts)>=4,"excluded_n_lt4")
        for factor,values in factors.items():
            for value in sorted(set(values)):
                mask=np.asarray(values==value)
                if not mask.any():continue
                groups.append({"seed":seed,"config":config,"endpoint":endpoint,"factor":factor,"level":value,"rows":int(mask.sum()),
                               "coverage80":float(covered80[mask].mean()),"coverage95":float(covered95[mask].mean()),
                               "residual_rms":float(np.sqrt(np.mean(z[mask,j]**2))),"mean_sigma":float(sigma[mask,j].mean()),
                               "rmse_ml":float(np.sqrt(np.mean((y[mask,j]-mean[mask,j])**2)))})
    return rows,groups


def old_audit():
    parameters=[];folds=[]
    for path in sorted((OLD/"qualification").glob("seed_*/fold_cache/fold_*.json")):
        if "complete" in path.name:continue
        f=read(path);seed_family=path.parent.parent.name
        if "model" not in f:continue
        model=f["model"]
        folds.append({"state":seed_family,"fold":f["fold"],**{k:f[k] for k in ("full_joint_nll_per_scalar","block_diagonal_nll_per_scalar","diagonal_nll_per_scalar","minimum_predictive_eigenvalue")}})
        for endpoint,p in enumerate(model.get("parameters",[model])):
            parameters.append({"state":seed_family,"fold":f["fold"],"endpoint":endpoint,"pca_fraction":model.get("pca_retained_standardized_input_variance_fraction"),
                               **{k:p.get(k) for k in ("optimizer_success","optimizer_message","nll","nit","nfev","lengthscale","noise_variance_scaled","noise_bound_hit")}})
    pd.DataFrame(parameters).to_csv(OUT/"old_optimizer_parameters.csv",index=False)
    pd.DataFrame(folds).to_csv(OUT/"old_joint_dependence.csv",index=False)


def run():
    protocol=freeze();OUT.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    frame,features,names=inputs();old_audit()
    ledgerpath=OUT/"compute_ledger.json"
    ledger=read(ledgerpath) if ledgerpath.exists() else {"active_seconds":0.,"attempts":[],"model_fit_limit":60,"restart_limit":360}
    previous_active=ledger["active_seconds"]
    metricrows=[];residualrows=[];conditionalrows=[];foldrows=[];pcarows=[]
    for seed in SEEDS:
        partition=pd.read_csv(BASE/f"splits/row_seed_{seed}.csv",dtype={"sample_id":str});validate_row_protocol(partition)
        if partition.sample_id.tolist()!=frame.sample_id.tolist():raise RuntimeError("partition identity drift")
        indices=partition.loc[partition.role.eq("l0"),"canonical_index"].to_numpy(int)
        store=RestrictedLabelStore(SOURCE_DATA,partition)
        y=store.reveal(frame.iloc[indices].sample_id,"initial_fit").astype(float)
        once(OUT/f"label_access_seed_{seed}.json",{"seed":seed,"calls":store.audit,"L333_ids_hash":ids_hash(frame.iloc[indices].sample_id),"future_U_labels_read":0,"validation_labels_read":0,"test_labels_read":0})
        splits=np.array_split(np.random.default_rng(seed+910007).permutation(len(indices)),3)
        for config,options in CONFIGS.items():
            x=features[options["feature"]][indices];means=np.empty_like(y);blocks=np.empty((len(y),2,2))
            for fold,held in enumerate(splits):
                train=np.setdiff1d(np.arange(len(y)),held);directory=OUT/f"seed_{seed}/{config}/fold_{fold}"
                contract={"protocol_sha256":digest(HERE/"surrogate_protocol.json"),"seed":seed,"config":config,"fold":fold,"X_sha256":arrayhash(x),"y_sha256":arrayhash(y),"train":train.tolist(),"held":held.tolist()}
                once(directory/"contract.json",contract)
                complete=directory/"complete.json"
                if complete.exists():
                    receipt=read(complete)
                    if receipt["contract_sha256"]!=digest(directory/"contract.json") or any(digest(directory/p)!=v for p,v in receipt["files"].items()):raise RuntimeError("completed fit drift")
                    arrays=np.load(directory/"oof.npz");f=read(directory/"metrics.json");audit=read(directory/"fit_audit.json")
                    means[held]=arrays["mean"];blocks[held]=arrays["blocks"]
                else:
                    active=previous_active+time.monotonic()-started
                    if len(ledger["attempts"])>=60 or active>=1800:raise RuntimeError("bounded diagnostic fit/activity budget exhausted")
                    ledger["attempts"].append({"seed":seed,"config":config,"fold":fold,"status":"RUNNING","started_unix":time.time(),"endpoint_starts":6})
                    ledger["active_seconds"]=active;dump(ledgerpath,ledger)
                    model=StudyGP(grouped=options["grouped"],max_components=options["max_components"]).fit(x[train],y[train])
                    posterior=model.posterior(x[held],row_ids=frame.iloc[indices[held]].sample_id.to_numpy())
                    mean=posterior.mean;cov=posterior.cov(np.arange(len(held)),np.arange(len(held)),noisy=True)
                    block=posterior.latent_diag(np.arange(len(held)))+posterior.noise_cov
                    f=fold_diagnostics(y[held],mean,cov);audit=model.audit_
                    means[held]=mean;blocks[held]=block
                    np.savez_compressed(directory/"oof.npz",mean=mean,blocks=block,covariance=cov,held_positions=held)
                    dump(directory/"metrics.json",f);dump(directory/"fit_audit.json",audit)
                    if options["max_components"]==16:
                        pca=model.transform_
                        for j,name in enumerate(names[:x.shape[1]]):
                            pcarows.append({"seed":seed,"config":config,"fold":fold,"feature":name,"coordinate_projection_energy":float((pca.rotation[j]**2).sum()),"raw_training_std":float(x[train,j].std())})
                    dump(complete,{"contract_sha256":digest(directory/"contract.json"),"files":{p:digest(directory/p) for p in ("oof.npz","metrics.json","fit_audit.json")}})
                    ledger["attempts"][-1].update(status="COMPLETE",finished_unix=time.time(),fit_seconds=audit["fit_seconds"])
                    ledger["active_seconds"]=previous_active+time.monotonic()-started;dump(ledgerpath,ledger)
                    print(json.dumps({"completed":len(ledger["attempts"]),"seed":seed,"config":config,"fold":fold,"active_seconds":ledger["active_seconds"]}),flush=True)
                foldrows.append({"seed":seed,"config":config,"fold":fold,**{k:v for k,v in f.items() if not isinstance(v,list)},"fit_seconds":audit["fit_seconds"],"pca_fraction":audit["pca_retained_standardized_input_variance_fraction"]})
            metrics=gaussian_probability_metrics(y,means,blocks)
            for j,endpoint in enumerate(("V1","V2")):
                metricrows.append({"seed":seed,"config":config,"endpoint":endpoint,**{k:metrics[k][j] for k in ("rmse","mae","marginal_nll","coverage50","coverage80","coverage95")}})
            rr,cr=residual_diagnostics(y,means,blocks,frame.iloc[indices],seed,config);residualrows+=rr;conditionalrows+=cr
            np.savez_compressed(OUT/f"seed_{seed}/{config}/oof.npz",mean=means,blocks=blocks,canonical_indices=indices)
    for name,rows in (("oof_metrics",metricrows),("residual_shape",residualrows),("conditional_calibration",conditionalrows),("joint_conditional_logscore",foldrows),("pca_coordinate_retention",pcarows)):
        pd.DataFrame(rows).to_csv(OUT/f"{name}.csv",index=False)
    # Input-only aliasing and duplicate-condition audit. No response examination.
    conditioncols=["canonical_smiles","Density g/ml","V/ul","loading solvent","Volume of loading solvent/ul","EA_fraction"]
    inputstats={"rows":len(frame),"molecules":int(frame.canonical_smiles.nunique()),"PEEA_strings":sorted(frame["PE/EA"].unique().tolist()),
                "normalized_EA_levels":sorted(frame.EA_fraction.unique().tolist()),"rows_in_duplicate_exact_conditions":int(frame.duplicated(conditioncols,keep=False).sum()),
                "unique_exact_conditions":int(len(frame.drop_duplicates(conditioncols))),
                "units":{"density":"column declares g/mL","sample_volume":"uL","density_times_volume":"numerically mg if density truly g/mL; no conversion required for learned standardized coordinate","loading_solvent_volume":"uL","targets":"mL"},
                "ranges":{c:{"min":float(frame[c].min()),"max":float(frame[c].max()),"unique":int(frame[c].nunique())} for c in ("Density g/ml","V/ul","Volume of loading solvent/ul")}}
    dump(OUT/"input_structure.json",inputstats)
    ledger["active_seconds"]=previous_active+time.monotonic()-started;dump(ledgerpath,ledger)
    dump(OUT/"complete.json",{"status":"COMPLETE","model_fits":len(ledger["attempts"]),"endpoint_optimization_starts":sum(a["endpoint_starts"] for a in ledger["attempts"]),
                              "active_seconds":ledger["active_seconds"],"ru_maxrss_bytes_macos":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                              "protocol_sha256":digest(HERE/"surrogate_protocol.json"),"future_U_labels_read":0,"validation_labels_read":0,"test_labels_read":0,
                              "files":{p.name:digest(p) for p in OUT.glob("*.csv")}})


if __name__=="__main__":run()
