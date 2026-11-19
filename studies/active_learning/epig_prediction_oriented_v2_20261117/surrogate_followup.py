"""Frozen-encoder covariance feasibility and cached-fold ranking diagnostics.

The frozen encoder has seen every L333 label. Its BLR is explicitly a
conditional diagnostic, never honest OOF or a qualification/main-AL model.
The four feature rankings restore already-fitted fold-0 parameters; no
additional optimized or fitted model is concealed in those reconstructions.
"""
from __future__ import annotations
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):os.environ[k]="1"
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT/"src"))
import hashlib,json,pickle,time,resource
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.stats import spearmanr
from qgeognn_al.active_learning_v2.epig_models import BayesianLinearSurrogate,FeatureTransform,_prepare_y
from qgeognn_al.active_learning_v2.protocol import RestrictedLabelStore,ids_hash
from qgeognn_al.resources import SOURCE_DATA
from surrogate_v2 import StudyGP,GroupedTransform
from surrogate_audit import BASE,OLD,OUT,SEEDS,CONFIGS,read,dump,once,digest,arrayhash,inputs

DEST=OUT/"followup"


def seal():
    dependencies={}
    for seed in SEEDS:
        for p in (BASE/f"runtime/seed_{seed}/lcmd/round_00/model/best.pt",BASE/f"runtime/seed_{seed}/scrubbed_graphs.pt",BASE/f"runtime/seed_{seed}/scrubbed_graphs.pt.contract.json"):
            dependencies[str(p.relative_to(ROOT))]=digest(p)
    record={"schema":"epig-v2-frozen-encoder-feasibility-1","study_type":"post-stage1/stage2 exploratory development study",
            "seeds":list(SEEDS),"budget":{"new_model_fits":5,"active_seconds_max":300,"automatic_retries":0,"threads":1},
            "encoder":"original seed-local L333 QGeoGNN-V2 extract_representation128, supervised with all L333; conditional diagnostic, not honest OOF",
            "new_model":"independent Gaussian BayesianLinearSurrogate on frozen encoder representation, PCA16, maxiter60, all L333, target divided by fixed L333 ddof0 scales",
            "qualification_claim":False,"adopt_for_primary_AL":False,"future_U_labels_read":0,"validation_labels_read":0,"test_labels_read":0,
            "ranking":"P1 complete3330 X, original U2997 candidates, objectives EPIG/IVR/BALD, streaming128x128 covariance blocks; report scores and top32, no new GNN fit",
            "feature_ranking":"reconstruct each configuration's already-fitted fold0 (222 labels) posterior; same L222 and U2997/X3330 for all four feature configs; diagnostic scope differs from L333 main AL",
            "files":{str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),HERE/"surrogate_v2.py")},"dependencies":dependencies,
            "parent_surrogate_protocol_sha256":digest(HERE/"surrogate_protocol.json")}
    once(HERE/"surrogate_followup_protocol.json",record)
    return record


def restore(seed,config,x,y):
    """Restore a completed fold without invoking fit/minimize or new choices."""
    directory=OUT/f"seed_{seed}/{config}/fold_0"
    contract=read(directory/"contract.json");audit=read(directory/"fit_audit.json");receipt=read(directory/"complete.json")
    if receipt["contract_sha256"]!=digest(directory/"contract.json") or any(digest(directory/p)!=h for p,h in receipt["files"].items()):raise RuntimeError("fold restore hash drift")
    options=CONFIGS[config];train=np.asarray(contract["train"])
    if arrayhash(x)!=contract["X_sha256"] or arrayhash(y)!=contract["y_sha256"]:raise RuntimeError("fold restore input drift")
    model=StudyGP(grouped=options["grouped"],max_components=options["max_components"])
    z,model.scales_,model.center_=_prepare_y(y[train],None,None)
    model.transform_=(GroupedTransform if options["grouped"] else FeatureTransform)(options["max_components"]).fit(x[train])
    model.train_x_=model.transform_.transform(x[train]);model.n_outputs=2;model.models_=[]
    for j,p in enumerate(audit["parameters"]):
        theta=np.log([p["signal_variance"],p["lengthscale"],p["noise_variance_scaled"]])
        k=model._kernel(model.train_x_,model.train_x_,theta);k.flat[::len(k)+1]+=np.exp(theta[2])
        factor=cho_factor(k,lower=True,check_finite=False)
        model.models_.append({"theta":theta,"factor":factor,"alpha":cho_solve(factor,z[:,j],check_finite=False)})
    model.noise_cov_=np.diag([p["noise_variance_scaled"] for p in audit["parameters"]])*np.outer(model.scales_,model.scales_)
    # Restored values must reproduce already saved held-out distributions.
    saved=np.load(directory/"oof.npz");held=np.asarray(contract["held"])
    p=model.posterior(x[held],row_ids=held)
    np.testing.assert_allclose(p.mean,saved["mean"],rtol=2e-10,atol=2e-10)
    np.testing.assert_allclose(p.latent_diag(np.arange(len(held)))+p.noise_cov,saved["blocks"],rtol=2e-10,atol=2e-10)
    return model


def scores(posterior,candidates,scales=None):
    """Exact full-P1 independent-output pair MI without U x U storage."""
    n=len(posterior.row_ids);q=posterior.n_outputs;references=np.arange(n)
    variances=np.diagonal(posterior.latent_diag(references),axis1=1,axis2=2)
    noise=np.diagonal(posterior.noise_cov);labelvar=variances+noise
    scale=np.ones(q) if scales is None else np.asarray(scales)
    epig=np.zeros(len(candidates));ivr=np.zeros(len(candidates));started=time.perf_counter()
    for a in range(0,n,128):
        refs=references[a:a+128]
        for b in range(0,len(candidates),128):
            cand=candidates[b:b+128]
            blocks=posterior.cov(refs,cand,noisy=False).reshape(len(refs),q,len(cand),q)
            cross=np.stack([blocks[:,j,:,j] for j in range(q)],axis=-1)
            for i in range(q):
                for j in range(q):
                    if i!=j and np.any(blocks[:,i,:,j]!=0):raise ValueError("independent-output streaming assumption violated")
            rho2=cross*cross/(labelvar[refs,None,:]*labelvar[cand][None,:,:])
            if np.any(rho2>=1):raise np.linalg.LinAlgError("nonpositive pair Schur complement")
            epig[b:b+len(cand)]+=(-.5*np.log1p(-rho2)).sum(axis=(0,2))/n
            ivr[b:b+len(cand)]+=(cross*cross/labelvar[cand][None,:,:]/scale**2).sum(axis=(0,2))/n
    bald=.5*np.log1p(variances[candidates]/noise).sum(axis=1)
    return {"epig":epig,"ivr":ivr,"bald":bald},time.perf_counter()-started


def representation(seed,outer,frame):
    import torch
    from torch_geometric.loader import DataLoader
    from qgeognn_al.models import load_predictor_checkpoint
    torch.set_num_threads(1)
    base=BASE/f"runtime/seed_{seed}";modelpath=base/"lcmd/round_00/model/best.pt";cache=base/"scrubbed_graphs.pt"
    frozen=read(OLD/"freeze.json")
    if frozen["dependencies"][str(modelpath.relative_to(ROOT))]!=digest(modelpath):raise RuntimeError("L333 checkpoint differs from v1")
    record=read(cache.with_suffix(".pt.contract.json"))
    if digest(cache)!=record["sha256"]:raise RuntimeError("scrubbed cache byte hash drift")
    atom,angle=torch.load(cache,weights_only=False)
    if len(atom)!=4163 or len(angle)!=4163:raise RuntimeError("cache row count drift")
    for i,item in enumerate(atom):
        if torch.count_nonzero(item.y) or int(item.canonical_position.reshape(-1)[0])!=i:raise PermissionError("cache contains labels or misaligned indices")
    model=load_predictor_checkpoint(modelpath)
    if model.training or sum(p.numel() for p in model.parameters())!=458952:raise RuntimeError("encoder identity drift")
    values=[]
    with torch.no_grad():
        atoms=DataLoader([atom[i] for i in outer],batch_size=256,shuffle=False)
        angles=DataLoader([angle[i] for i in outer],batch_size=256,shuffle=False)
        for ab,ba in zip(atoms,angles):values.append(model.extract_representation(ab,ba).numpy())
    result=np.vstack(values).astype(float)
    if result.shape!=(3330,128) or not np.isfinite(result).all():raise RuntimeError("invalid representation")
    return result


def run():
    seal();DEST.mkdir(parents=True,exist_ok=True);started=time.perf_counter();frame,features,_=inputs()
    comparisons=[];summary=[]
    for seed in SEEDS:
        partition=pd.read_csv(BASE/f"splits/row_seed_{seed}.csv",dtype={"sample_id":str})
        l0=partition.loc[partition.role.eq("l0"),"canonical_index"].to_numpy(int)
        outer=partition.loc[partition.role.isin(["l0","u0"]),"canonical_index"].to_numpy(int)
        outerrole=partition.iloc[outer].role.to_numpy();lpos=np.flatnonzero(outerrole=="l0");candidates=np.flatnonzero(outerrole=="u0")
        store=RestrictedLabelStore(SOURCE_DATA,partition);y=store.reveal(frame.iloc[l0].sample_id,"initial_fit").astype(float)
        scales=y.astype(np.float32).std(axis=0,ddof=0).astype(float)
        once(DEST/f"labels_seed_{seed}.json",{"calls":store.audit,"future_U_labels_read":0,"validation_labels_read":0,"test_labels_read":0})
        model_scores={}
        for config,options in CONFIGS.items():
            directory=DEST/f"seed_{seed}/{config}";path=directory/"scores.npz"
            if path.exists():
                saved=np.load(path);values={key:saved[key] for key in ("epig","ivr","bald")}
            else:
                model=restore(seed,config,features[options["feature"]][l0],y)
                posterior=model.posterior(features[options["feature"]][outer],row_ids=frame.iloc[outer].sample_id.to_numpy())
                values,seconds=scores(posterior,candidates,scales=scales)
                directory.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**values,candidate_canonical_indices=outer[candidates])
                dump(directory/"receipt.json",{"scope":"same restored fold0 L222, original U2997, full X3330 P1","score_seconds":seconds,"scores_sha256":digest(path),"new_fits":0,"max_covariance_request_shape":[256,256]})
            model_scores[config]=values
        encoder_directory=DEST/f"seed_{seed}/frozen_encoder_blr";receipt=encoder_directory/"complete.json"
        if not receipt.exists():
            if time.perf_counter()-started>300:raise RuntimeError("300second followup active budget exhausted")
            encoder_directory.mkdir(parents=True,exist_ok=True)
            reps=representation(seed,outer,frame)
            # lpos follows canonical order, exactly matching l0/source row order.
            if not np.array_equal(outer[lpos],l0):raise RuntimeError("L333 representation alignment drift")
            once(encoder_directory/"attempt.json",{"seed":seed,"status":"STARTED","protocol_sha256":digest(HERE/"surrogate_followup_protocol.json")})
            model=BayesianLinearSurrogate(coregionalized=False,max_components=16,maxiter=60).fit(reps[lpos],y/scales)
            posterior=model.posterior(reps,row_ids=frame.iloc[outer].sample_id.to_numpy())
            values,seconds=scores(posterior,candidates)
            positions=np.arange(64);cov=posterior.cov(positions,positions,noisy=True)
            np.savez_compressed(encoder_directory/"scores.npz",**values,candidate_canonical_indices=outer[candidates])
            dump(encoder_directory/"fit_audit.json",model.audit_)
            dump(receipt,{"scope":"conditional on full-L333 supervised encoder; not honest OOF; no calibration evidence","new_fits":1,
                          "representation_shape":list(reps.shape),"representation_sha256":arrayhash(reps),"score_seconds":seconds,
                          "sample64_noisy_covariance_min_eigenvalue":float(np.linalg.eigvalsh(cov)[0]),"noise_min_eigenvalue":float(np.linalg.eigvalsh(posterior.noise_cov)[0]),
                          "scores_sha256":digest(encoder_directory/"scores.npz"),"fit_audit_sha256":digest(encoder_directory/"fit_audit.json")})
        saved=np.load(encoder_directory/"scores.npz");model_scores["frozen_encoder_blr"]={k:saved[k] for k in ("epig","ivr","bald")}
        # The original L333 F0 GP is an available frozen model, not refitted.
        original=OLD/f"acquisition/models/seed_{seed}_gp_matern32";r=read(original/"complete.json")
        if digest(original/"model.pkl")!=r["model_sha256"]:raise RuntimeError("v1 probability model hash drift")
        with (original/"model.pkl").open("rb") as handle:model=pickle.load(handle)
        posterior=model.posterior(features["F0"][outer],row_ids=frame.iloc[outer].sample_id.to_numpy())
        origpath=DEST/f"seed_{seed}/F0_original_L333/scores.npz"
        if origpath.exists():
            a=np.load(origpath);values={k:a[k] for k in ("epig","ivr","bald")}
        else:
            values,seconds=scores(posterior,candidates);origpath.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(origpath,**values,candidate_canonical_indices=outer[candidates])
            dump(origpath.with_suffix(".json"),{"new_fits":0,"source_model_sha256":digest(original/"model.pkl"),"score_seconds":seconds,"scores_sha256":digest(origpath)})
        model_scores["F0_original_L333"]=values
        for name,values in model_scores.items():
            for objective,v in values.items():
                nporder=np.argsort(-v,kind="stable")
                summary.append({"seed":seed,"config":name,"objective":objective,"score_mean":float(v.mean()),"score_max":float(v.max()),"top32_ids_hash":ids_hash(frame.iloc[outer[candidates[nporder[:32]]]].sample_id)})
        for a,b in (("F0_PCA16_opt","F1_PCA16_opt"),("F1_PCA16_opt","F2_PCA16_opt"),("F2_PCA16_opt","F2_full_opt"),("F0_original_L333","frozen_encoder_blr")):
            for obj in ("epig","ivr","bald"):
                av,bv=model_scores[a][obj],model_scores[b][obj];at=np.argsort(-av,kind="stable")[:32];bt=np.argsort(-bv,kind="stable")[:32]
                comparisons.append({"seed":seed,"comparison":a+"__vs__"+b,"objective":obj,"spearman":float(spearmanr(av,bv).statistic),"top32_overlap":len(set(at)&set(bt))/32,
                                    "same_training_labels":True,"same_frozen_encoder":False if "encoder" in b else None})
        print(json.dumps({"seed":seed,"complete":True,"elapsed_seconds":time.perf_counter()-started}),flush=True)
    pd.DataFrame(summary).to_csv(DEST/"score_summary.csv",index=False);pd.DataFrame(comparisons).to_csv(DEST/"ranking_comparisons.csv",index=False)
    dump(DEST/"complete.json",{"status":"COMPLETE","new_model_fits":5,"restored_fold_models_no_new_fit":20,"old_fullL333_models_reused":5,
                               "active_seconds":time.perf_counter()-started,"ru_maxrss_bytes_macos":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                               "future_U_labels_read":0,"validation_labels_read":0,"test_labels_read":0,
                               "protocol_sha256":digest(HERE/"surrogate_followup_protocol.json"),"outputs":{p.name:digest(p) for p in DEST.glob("*.csv")}})


if __name__=="__main__":run()
