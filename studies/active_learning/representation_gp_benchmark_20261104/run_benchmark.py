"""Run the preregistered frozen-checkpoint study without AL state transitions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/representation_gp_matplotlib")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import scipy
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diagnostics import fixed_candidate_positions, contrast_rows, fusion_features, template_diagnostics
from representations import (audit_lcmd_state, file_hash, load_model, load_scrubbed_graphs,
                             extract_hidden, extract_gradient, structured_features)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


def state_key(seed,budget):
    return f"seed_{seed}_L{budget}"


def safe_records(frame, indices, with_ids=True):
    result=[]
    fields=("PE/EA","Density g/ml","V/ul","loading solvent","Volume of loading solvent/ul")
    for i in indices:
        row=frame.iloc[int(i)]
        r={"smiles":str(row.canonical_smiles),"conditions":{f:row[f] for f in fields}}
        if with_ids:r["id"]=str(row.sample_id)
        result.append(r)
    return result


def authorized_labels(state,source):
    from qgeognn_al.active_learning_v2.protocol import RestrictedLabelStore
    if set(state.labeled_ids)&set(state.validation_ids):raise ValueError("evaluation overlaps training")
    if len(set(state.labeled_ids))!=state.budget or len(set(state.validation_ids))!=416:
        raise ValueError("duplicate or missing authorized label IDs")
    context=json.loads((state.root/"studies/active_learning/qgeognn_v2_row_sequential_b32/runtime"/f"seed_{state.seed}/context.json").read_text())
    if file_hash(source)!=context["contract"]["source_sha256"]:
        raise ValueError("canonical source differs from historical freeze")
    store=RestrictedLabelStore(source,state.partition)
    role=dict(zip(state.partition.sample_id.astype(str),state.partition.role))
    acquired=[sid for sid in state.labeled_ids if role[sid]=="u0"]
    store.freeze_acquisitions(acquired)
    train=store.reveal(state.labeled_ids,"after_acquisition_fit").astype(np.float64)
    validation=store.reveal(state.validation_ids,"initial_fit").astype(np.float64)
    l0_map={int(i):train[j] for j,i in enumerate(state.labeled_indices)}
    l0=np.array([l0_map[int(i)] for i in state.l0_indices])
    return train,validation,l0,{"calls":store.audit,"test_truth_access_count":0,
        "future_U_labels_read":0,"freeze_scope":"historical L_t only; no acquisition executed",
        "float32_legacy_store_then_float64":True}


def prepare_state(root,seed,budget,frame,source,graphs=None):
    import torch
    torch.set_num_threads(1)
    key=state_key(seed,budget)
    cache=HERE/"cache"/(key+".npz")
    receipt=HERE/"cache"/(key+".json")
    state=audit_lcmd_state(root,seed,budget)
    up=fixed_candidate_positions(state.unlabeled_ids,256,seed)
    ci=state.unlabeled_indices[up]
    indices=np.r_[state.labeled_indices,ci,state.validation_indices]
    if cache.exists() and receipt.exists():
        meta=json.loads(receipt.read_text())
        if meta["cache_sha256"]!=file_hash(cache) or meta["state"]!=state.audit or meta["source_sha256"]!=file_hash(source):
            raise ValueError("prepared state cache drift")
        with np.load(cache,allow_pickle=False) as z:data={k:z[k] for k in z.files}
        if not np.array_equal(data["indices"],indices) or data["train_y"].shape!=(budget,2) or data["validation_y"].shape!=(416,2):
            raise ValueError("prepared row/response mapping drift")
        return state,data,meta
    started=time.perf_counter()
    train,val,l0,label_audit=authorized_labels(state,source)
    atom,angle=load_scrubbed_graphs(root,seed) if graphs is None else graphs
    model=load_model(state)
    gnn,qpred,ha=extract_hidden(state,atom,angle,indices,model=model)
    raw,ra=extract_gradient(state,atom,angle,indices,kind="raw",model=model)
    def progress(p):
        if p["completed_rows"]%500==0:print(key,"CW",p["completed_rows"],"/",p["total_rows"],flush=True)
    cw,ca=extract_gradient(state,atom,angle,indices,kind="cw",model=model,l0_targets=l0,progress=progress)
    subset=frame.iloc[indices].reset_index(drop=True)
    structured,sa=structured_features(subset,np.arange(budget))
    data={"indices":indices,"train_y":train,"validation_y":val,"gnn":gnn,"gnn_q50":qpred,
          "raw":raw,"cw":cw,"structured":structured}
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,**data)
    meta={"state":state.audit,"indices_sha256":hashlib.sha256(indices.tobytes()).hexdigest(),
          "source_sha256":file_hash(source),"label_access":label_audit,
          "representations":{"gnn":ha,"raw":ra,"cw":ca,"structured":sa},
          "candidate_policy":"256 SHA256-ranked legal current U IDs; identical across representations",
          "candidate_ids":[state.unlabeled_ids[i] for i in up],
          "cache_sha256":file_hash(cache),"elapsed_seconds":time.perf_counter()-started}
    dump(receipt,meta)
    del model;gc.collect()
    print(key,"features_ready",round(meta["elapsed_seconds"],2),flush=True)
    return state,data,meta


def prepare_llm(frame,indices):
    """Each row is encoded once and reused across historical states."""
    from llm_embeddings import LocalFrozenTextEncoder,render_experiment_text
    path=HERE/"cache/llm.npz";receipt=path.with_suffix(".json")
    ordered=np.array(sorted(set(int(i) for i in indices)),dtype=int)
    previous=None
    if path.exists():
        meta=json.loads(receipt.read_text())
        if meta["cache_sha256"]!=file_hash(path):raise ValueError("LLM cache drift")
        with np.load(path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
        if set(ordered)<=set(data["indices"]):return data,meta
        previous=data
        ordered=np.array([i for i in ordered if i not in set(data["indices"])],dtype=int)
    encoder=LocalFrozenTextEncoder()
    records=safe_records(frame,ordered,with_ids=False)
    data={"indices":ordered};times={}
    for name,variant,template in [("llm_smiles","smiles_only","A"),("llm_conditions","conditions","A"),
                                  ("llm_template_b","conditions","B")]:
        texts=[render_experiment_text(r,variant,template) for r in records]
        # Identical SMILES-only strings share one pretrained forward pass.
        unique=list(dict.fromkeys(texts));mapping={s:i for i,s in enumerate(unique)}
        encoded=encoder.encode_texts(unique,batch_size=8)
        data[name]=encoded[[mapping[t] for t in texts]]
        times[name]=dict(encoder.metadata["last_encoding"])
        print("LLM",name,"encoded",len(unique),"unique texts",times[name]["seconds"],flush=True)
    path.parent.mkdir(exist_ok=True)
    if previous is not None:
        data={name:np.concatenate([previous[name],value],axis=0) for name,value in data.items()}
        times={"extension":times,"previous":meta["encoding"]}
    np.savez_compressed(path,**data)
    from huggingface_hub import snapshot_download
    from llm_embeddings import model_file_manifest,MODEL_ID
    local=snapshot_download(MODEL_ID,local_files_only=True)
    meta={"encoder":encoder.metadata,"encoding":times,"model_files":model_file_manifest(local),
          "cache_sha256":file_hash(path),"rows":len(data["indices"]),"no_download":True,"paid_api_calls":0}
    dump(receipt,meta)
    del encoder;gc.collect()
    return data,meta


def fit_state(state,data,frame,plan,llm=None):
    from gp_models import FixedRepresentationGP,ProjectedMaternGP,regression_metrics,acquisition_diagnostics
    output=HERE/"results"/state_key(state.seed,state.budget)
    if (output/"complete.json").exists():return
    output.mkdir(parents=True,exist_ok=True)
    n=state.budget;c=256;vstart=n+c
    reps={name:data[name] for name in ("structured","gnn","raw","cw")}
    if llm is not None:
        lookup={int(i):j for j,i in enumerate(llm["indices"])}
        pos=[lookup[int(i)] for i in data["indices"]]
        for name in ("llm_smiles","llm_conditions","llm_template_b"):reps[name]=llm[name][pos]
        reps["fusion"],fa=fusion_features(reps["gnn"],reps["llm_conditions"],n)
        dump(output/"fusion_transform.json",fa)
    specs=[(name,"matern52",None) for name in reps if name!="llm_template_b"]
    specs += [(name,"linear",None) for name in ("raw","cw")]
    for name in ("gnn","llm_conditions","fusion"):
        if name in reps:specs += [(name,"projected",s) for s in plan["projection_seeds"]]
    metrics=[];fits=[];contrasts=[];acquisitions=[];predictions=[];robust=[]
    records=safe_records(frame,state.validation_indices)
    for name,kind,init in specs:
        tag=f"{name}_{kind}"+(f"_init{init}" if init is not None else "")
        existing=output/(tag+".json")
        if existing.exists():
            r=json.loads(existing.read_text())
            metrics.extend(r["metrics"]);fits.extend(r["fits"]);contrasts.extend(r["contrasts"])
            acquisitions.extend(r["acquisitions"]);predictions.extend(r["predictions"]);robust.extend(r["robustness"])
            continue
        x=reps[name];models=[];means=[];variances=[];mr=[];fr=[];ar=[];pr=[]
        for j in range(2):
            model=(ProjectedMaternGP(projection_dim=64,max_iter=plan["max_iter"],seed=init)
                   if kind=="projected" else FixedRepresentationGP(kernel=kind,max_iter=plan["max_iter"],starts=1,seed=17))
            t=time.perf_counter();model.fit(x[:n],data["train_y"][:,j]);ft=time.perf_counter()-t
            t=time.perf_counter();mu,latent=model.predict(x[vstart:]);it=time.perf_counter()-t
            obs=latent+model.noise
            base={"seed":state.seed,"budget":n,"representation":name,"kernel":kind,"initialization":init,"endpoint":f"V{j+1}"}
            score=regression_metrics(data["validation_y"][:,j],mu,obs)
            score["normalized_rmse_L0"]=float(score["rmse"] / state.audit["endpoint_scales"][f"V{j+1}"])
            mr.append({**base,**score,"fit_seconds":ft,"inference_seconds":it,"evaluation_role":"previously_used_validation_development"})
            fr.append({**base,**model.audit})
            acq=acquisition_diagnostics(model,x[n:vstart],batch_size=32,candidate_limit=256,seed=state.seed)
            up=fixed_candidate_positions(state.unlabeled_ids,256,state.seed)
            candidate_ids=[state.unlabeled_ids[i] for i in up]
            for method in acq["methods"].values():
                method["selected_ids"]=[candidate_ids[i] for i in method["selected_candidate_positions"]]
            ar.append({**base,**acq,"candidate_ids":candidate_ids})
            for k,sid in enumerate(state.validation_ids):
                pr.append({**base,"id":sid,"truth_ml":float(data["validation_y"][k,j]),"mean_ml":float(mu[k]),
                           "latent_variance_ml2":float(latent[k]),"observation_variance_ml2":float(obs[k])})
            means.append(mu);variances.append(latent);models.append(model)
        cr=contrast_rows(records,x[vstart:],data["validation_y"],np.column_stack(means),data["gnn_q50"][vstart:],x[:n])
        if kind=="projected":
            vlookup={r["id"]:i for i,r in enumerate(records)}
            for j,model in enumerate(models):
                embedded=model.embedding(x)
                erms=max(float(np.sqrt(np.mean(np.sum((embedded[:n]-embedded[:n].mean(0))**2,axis=1)))),1e-12)
                for r in cr:
                    if r["endpoint"]==f"V{j+1}":
                        a,b=vlookup[r["id_a"]]+vstart,vlookup[r["id_b"]]+vstart
                        r["learned_distance_train_rms"]=float(np.linalg.norm(embedded[a]-embedded[b])/erms)
        cr=[{"seed":state.seed,"budget":n,"representation":name,"kernel":kind,"initialization":init,**r} for r in cr]
        rr=[]
        if name=="llm_conditions" and kind=="matern52":
            rr=[{"seed":state.seed,"budget":n,**template_diagnostics(x,reps["llm_template_b"],n,slice(n,vstart),models)}]
        r={"metrics":mr,"fits":fr,"contrasts":cr,"acquisitions":ar,"predictions":pr,"robustness":rr}
        dump(existing,r)
        for dest,key in [(metrics,"metrics"),(fits,"fits"),(contrasts,"contrasts"),(acquisitions,"acquisitions"),(predictions,"predictions"),(robust,"robustness")]:dest.extend(r[key])
        print(state_key(state.seed,n),tag,"fit_done",flush=True)
        del models;gc.collect()
    for filename,rows in [("metrics",metrics),("fits",fits),("contrasts",contrasts),("acquisitions",acquisitions),("predictions",predictions),("robustness",robust)]:
        dump(output/(filename+".json"),rows)
    dump(output/"complete.json",{"seed":state.seed,"budget":n,"model_variants":len(specs),"endpoint_fits":len(fits),
        "evaluation_rows":416,"test_truth_access_count":0,"new_gnn_fits":0,"al_trajectories_started":0})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=["audit","prepare","encode","preflight","full"],required=True)
    parser.add_argument("--project-root",type=Path,default=ROOT)
    args=parser.parse_args();root=args.project_root.resolve()
    from qgeognn_al.active_learning_v2.benchmark_protocol import load_features
    source=root/"experiments/e0_4g_baseline/canonical_4g.csv"
    frame=load_features(source)
    plan=json.loads((HERE/"config.json").read_text())
    states=[(s,b) for s in plan["seeds"] for b in plan["budgets"]]
    if args.phase=="audit":
        rows=[]
        for s,b in states:
            try:rows.append({"available":True,**audit_lcmd_state(root,s,b).audit})
            except Exception as e:rows.append({"seed":s,"budget":b,"available":False,"reason":str(e)})
        dump(HERE/"state_audit.json",rows);print("available",sum(r["available"] for r in rows),"/",len(rows));return
    if args.phase in {"preflight","prepare"}:states=[(s,b) for s in plan["preflight_seeds"] for b in plan["preflight_budgets"]]
    lock=HERE/"protocol_freeze.json"
    if not lock.exists():raise ValueError("protocol must be frozen before feature/label preparation")
    frozen=json.loads(lock.read_text())
    if frozen["config_sha256"]!=file_hash(HERE/"config.json"):raise ValueError("frozen configuration drift")
    started=time.monotonic();prepared=[]
    for s,b in states:
        if time.monotonic()-started>plan["wall_limit_seconds"]:raise RuntimeError("preregistered wall resource cap reached")
        state,data,meta=prepare_state(root,s,b,frame,source)
        prepared.append((state,data,meta))
    if args.phase=="prepare":return
    allindices=np.concatenate([d["indices"] for _,d,_ in prepared])
    llm,lmeta=prepare_llm(frame,allindices)
    if args.phase=="encode":return
    for state,data,meta in prepared:
        if time.monotonic()-started>plan["wall_limit_seconds"]:raise RuntimeError("preregistered wall resource cap reached")
        fit_state(state,data,frame,plan,llm)
    dump(HERE/(args.phase+"_completion.json"),{"finished_utc":datetime.now(timezone.utc).isoformat(),
        "states":[[s.seed,s.budget] for s,_,_ in prepared],"elapsed_seconds":time.monotonic()-started})


if __name__=="__main__":main()
