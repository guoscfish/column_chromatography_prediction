"""Frozen-X, frozen-posterior full-integral and coupled-reference experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import tempfile
import time

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key,"1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/"studies/active_learning/epig_prediction_oriented_20261117"
BASE=ROOT/"studies/active_learning/qgeognn_v2_row_sequential_b32"
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from acquisition_v2 import (IndependentStreamingCache,select_streaming_batch,
    batch_target_information,coupled_reference_positions,GaussianStateWithBALD)
from qgeognn_al.active_learning_v2.benchmark_protocol import load_features
from qgeognn_al.resources import SOURCE_DATA,SOURCE_GRAPH_CACHE

SEEDS=(157,887,2357,6101,12203)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");tmp.replace(path)

def features():
    """Exact original F0 reconstruction via an explicit feature-only CSV read."""
    import torch
    from qgeognn_al.data import eluent_descriptor
    old=read(OLD/"features.json")
    if digest(SOURCE_DATA)!=old["source_sha256"] or digest(SOURCE_GRAPH_CACHE)!=old["graph_cache_sha256"]:
        raise RuntimeError("source feature dependency drift")
    frame=load_features()
    graphs=torch.load(SOURCE_GRAPH_CACHE,weights_only=False)
    molecular=np.vstack([graphs[s]["descriptor"] for s in frame.canonical_smiles]).astype(float)
    eluents=np.vstack([eluent_descriptor(v) for v in frame["PE/EA"]]).astype(float)
    solvent=np.column_stack([frame["loading solvent"].eq(v).to_numpy(float) for v in ("PE","EA","DCM")])
    amount=frame["Density g/ml"].to_numpy(float)*frame["V/ul"].to_numpy(float)
    volume=frame["Volume of loading solvent/ul"].to_numpy(float)
    x=np.column_stack([molecular,eluents,solvent,amount,volume,np.log1p(amount),np.log1p(volume)])
    if hashlib.sha256(x.tobytes()).hexdigest()!=old["features_sha256"]:
        raise RuntimeError("reconstructed F0 feature bytes differ")
    return frame,x

def protocol():
    dependencies=[SOURCE_DATA,SOURCE_GRAPH_CACHE,OLD/"freeze.json",OLD/"features.json"]
    for seed in SEEDS:
        dependencies += [BASE/f"splits/row_seed_{seed}.csv"]
        dependencies += [OLD/f"acquisition/models/seed_{seed}_gp_matern32/{name}" for name in
                         ("model.pkl","fit_audit.json","complete.json")]
    return {"schema":"epig-v2-reference-audit-v1","nature":"post-stage1/stage2 exploratory development study",
        "seeds":list(SEEDS),"family":"gp_matern32","features":"unchanged F0/PCA16",
        "posterior":"reuse hash-verified original full L333 cache, no refit, no new truth read",
        "outputs":"original fixed L333 endpoint scales; independent future label replicates",
        "distributions":{"P1":"all 3330 outer-training X rows","P2":"all current 2997 U X rows"},
        "settings":["full","coupled_M256","coupled_M1024","coupled_M2048","legacy_M256","legacy_M1024","legacy_M2048"],
        "objectives":["epig","ivr","bald"],"batch_size":32,
        "coupling":"sha256(epig_v2_crn:seed:row_id), shared across P1/P2",
        "maximum_active_seconds":7200,"new_surrogate_fits":0,"new_gnn_fits":0,
        "memory":"disk-backed float64 independent-channel target-candidate cross cache; requests <=64x256 rows; no full covariance call",
        "comparisons":"score MAE/max/relativeL2, Spearman, pointwise Top32, conditional greedy B32 overlap, own-target and common full-P1 batch MI",
        "dependencies":{str(p.relative_to(ROOT)):digest(p) for p in dependencies},
        "code":{name:digest(HERE/name) for name in ("acquisition_v2.py","reference_audit.py","math_oracle.py")}}

def seal():
    path=HERE/"reference_protocol.json"
    record=protocol()
    if path.exists():
        if read(path)!=record:raise RuntimeError("frozen reference protocol/code/dependency drift")
    else:
        if (HERE/"reference/results.csv").exists():raise RuntimeError("cannot freeze after results")
        dump(path,record)
    return digest(path)

def batch_bald(p,candidates):
    state=GaussianStateWithBALD(p,candidates,np.array([0]),chunk_size=128)
    initial=state.scores("bald");trace=[]
    for step in range(32):
        scores=initial if step==0 else state.scores("bald")
        index=int(np.argmax(scores));trace.append({"step":step+1,"candidate_index":index,
            "row_position":int(candidates[index]),"conditional_score":float(scores[index])});state.condition(index)
    return {"objective":"bald","mode":"conditional_greedy","selected_positions":candidates[state.selected_indices].tolist(),
        "initial_scores":initial.tolist(),"trace":trace}

def overlap(a,b):return len(set(a)&set(b))/32

def run(seed_filter=None):
    freeze_sha=seal();frame,x=features();rows=[];started=time.perf_counter()
    out=HERE/"reference";out.mkdir(exist_ok=True)
    for seed in SEEDS:
        if seed_filter is not None and seed not in seed_filter:continue
        directory=out/f"seed_{seed}";directory.mkdir(exist_ok=True)
        complete=directory/"complete.json"
        if complete.exists():
            receipt=read(complete)
            if receipt["reference_protocol_sha256"]!=freeze_sha:raise RuntimeError("reference resume protocol drift")
            for name,expected in receipt["files"].items():
                if digest(directory/name)!=expected:raise RuntimeError("reference result hash drift")
            rows.extend(pd.read_csv(directory/"summary.csv").to_dict("records"));continue
        split=pd.read_csv(BASE/f"splits/row_seed_{seed}.csv",dtype={"sample_id":str})
        if split.sample_id.tolist()!=frame.sample_id.tolist():raise RuntimeError("split feature identity mismatch")
        outer=split.loc[split.role.isin(["l0","u0"]),"canonical_index"].to_numpy(int)
        candidates=np.flatnonzero(split.iloc[outer].role.eq("u0").to_numpy())
        ids=frame.iloc[outer].sample_id.to_numpy(str)
        modeldir=OLD/f"acquisition/models/seed_{seed}_gp_matern32"
        receipt=read(modeldir/"complete.json")
        if digest(modeldir/"model.pkl")!=receipt["model_sha256"] or digest(modeldir/"fit_audit.json")!=receipt["fit_audit_sha256"]:
            raise RuntimeError("cached posterior hash drift")
        with (modeldir/"model.pkl").open("rb") as f:model=pickle.load(f)
        p=model.posterior(x[outer],row_ids=ids)
        bald=batch_bald(p,candidates);all_results={};seedrows=[]
        with tempfile.TemporaryDirectory(prefix=f"epig_v2_seed{seed}_",dir="/tmp") as tmp:
            cache=IndependentStreamingCache(p,candidates,tmp)
            for dist in ("P1","P2"):
                population=np.arange(len(outer)) if dist=="P1" else candidates
                settings={"full":population}
                for m in (256,1024,2048):
                    settings[f"coupled_M{m}"]=coupled_reference_positions(seed,population,ids,m)
                    settings[f"legacy_M{m}"]=np.array(sorted(population.tolist(),key=lambda i:hashlib.sha256(
                        f"epig_v1:{seed}:{dist}:{ids[i]}".encode()).digest())[:m],dtype=int)
                # Full first: all finite-M comparisons are bound to the same exact full target.
                for setting,refs in settings.items():
                    if time.perf_counter()-started>7200:raise RuntimeError("predefined reference audit active budget exhausted")
                    print(json.dumps({"seed":seed,"distribution":dist,"reference":setting,"rows":len(refs)}),flush=True)
                    key=f"{dist}_{setting}"
                    path=directory/f"{key}.json"
                    if path.exists():
                        record=read(path)
                        if record["reference_protocol_sha256"]!=freeze_sha:raise RuntimeError("partial resume protocol drift")
                        results=record["acquisitions"]
                    else:
                        results={objective:select_streaming_batch(cache,refs,objective=objective) for objective in ("epig","ivr")}
                        results["bald"]=bald
                        for objective,result in results.items():
                            result["batch_MI_own_reference"]=batch_target_information(p,refs,result["selected_positions"])
                            result["batch_MI_full_own_target"]=batch_target_information(p,population,result["selected_positions"])
                            result["batch_MI_common_full_P1"]=batch_target_information(p,np.arange(len(outer)),result["selected_positions"])
                            point=candidates[np.argsort(-np.asarray(result["initial_scores"]),kind="stable")[:32]].tolist()
                            result["pointwise_positions"]=point
                            result["pointwise_batch_MI_full_own_target"]=batch_target_information(p,population,point)
                        dump(path,{"reference_protocol_sha256":freeze_sha,"references":refs.tolist(),"acquisitions":results})
                    all_results[key]=results
                    for objective,result in results.items():
                        full=all_results[f"{dist}_full"][objective]
                        scores=np.asarray(result["initial_scores"]);truth=np.asarray(full["initial_scores"])
                        delta=scores-truth
                        seedrows.append({"seed":seed,"distribution":dist,"reference":setting,"M":len(refs),"objective":objective,
                            "score_mae":float(np.mean(abs(delta))),"score_max_error":float(np.max(abs(delta))),
                            "score_relative_L2_error":float(np.linalg.norm(delta)/np.linalg.norm(truth)),
                            "spearman_vs_full":float(spearmanr(scores,truth).statistic),
                            "top32_overlap_vs_full":overlap(result["pointwise_positions"],full["pointwise_positions"]),
                            "greedy32_overlap_vs_full":overlap(result["selected_positions"],full["selected_positions"]),
                            "batch_MI_own_reference":result["batch_MI_own_reference"],
                            "batch_MI_full_own_target":result["batch_MI_full_own_target"],
                            "batch_MI_common_full_P1":result["batch_MI_common_full_P1"],
                            "pointwise_batch_MI_full_own_target":result["pointwise_batch_MI_full_own_target"],
                            "seconds":result.get("seconds",0.)})
            cross=[]
            for dist in ("P1","P2"):
                for setting in ("full","coupled_M256","coupled_M1024","coupled_M2048","legacy_M256","legacy_M1024","legacy_M2048"):
                    result=all_results[f"{dist}_{setting}"]
                    for a,b in (("epig","ivr"),("epig","bald"),("ivr","bald")):
                        cross.append({"seed":seed,"distribution":dist,"reference":setting,"comparison":f"{a}_vs_{b}",
                            "spearman":float(spearmanr(result[a]["initial_scores"],result[b]["initial_scores"]).statistic),
                            "top32_overlap":overlap(result[a]["pointwise_positions"],result[b]["pointwise_positions"]),
                            "greedy32_overlap":overlap(result[a]["selected_positions"],result[b]["selected_positions"]),
                            "first_batch_MI_full_P1":result[a]["batch_MI_common_full_P1"],"second_batch_MI_full_P1":result[b]["batch_MI_common_full_P1"]})
            for setting in ("full","coupled_M256","coupled_M1024","coupled_M2048","legacy_M256","legacy_M1024","legacy_M2048"):
                for objective in ("epig","ivr","bald"):
                    a=all_results[f"P1_{setting}"][objective];b=all_results[f"P2_{setting}"][objective]
                    cross.append({"seed":seed,"distribution":"P1_vs_P2","reference":setting,"comparison":objective,
                        "spearman":float(spearmanr(a["initial_scores"],b["initial_scores"]).statistic),
                        "top32_overlap":overlap(a["pointwise_positions"],b["pointwise_positions"]),
                        "greedy32_overlap":overlap(a["selected_positions"],b["selected_positions"]),
                        "first_batch_MI_full_P1":a["batch_MI_common_full_P1"],"second_batch_MI_full_P1":b["batch_MI_common_full_P1"]})
            pd.DataFrame(seedrows).to_csv(directory/"summary.csv",index=False)
            pd.DataFrame(cross).to_csv(directory/"comparisons.csv",index=False)
            dump(directory/"cost.json",{"cross_cache_bytes_on_disk":cache.bytes,"maximum_covariance_request_rows":cache.max_covariance_shape,
                "cache_build_seconds":cache.build_seconds,"process_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "labels_read":0,"new_fits":0})
        files=[p for p in directory.iterdir() if p.is_file() and p.name!="complete.json"]
        dump(complete,{"reference_protocol_sha256":freeze_sha,"files":{p.name:digest(p) for p in files}})
        rows.extend(seedrows)
        pd.DataFrame(rows).to_csv(out/"results.csv",index=False)
    costs=[read(p) for p in out.glob("seed_*/cost.json")]
    dump(out/"status.json",{"completed_seeds":[int(p.parent.name.split("_")[1]) for p in out.glob("seed_*/complete.json")],
        "reference_protocol_sha256":freeze_sha,"elapsed_this_invocation_seconds":time.perf_counter()-started,
        "process_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"label_access":"none; only feature CSV columns and prefit models",
        "future_U_labels_read":0,"test_labels_read":0,"costs":costs})

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--seeds",type=int,nargs="*");parser.add_argument("--freeze-only",action="store_true")
    args=parser.parse_args()
    print(json.dumps({"reference_protocol_sha256":seal()}),flush=True)
    if not args.freeze_only:run(args.seeds)
