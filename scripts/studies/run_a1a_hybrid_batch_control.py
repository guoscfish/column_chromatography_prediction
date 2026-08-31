#!/usr/bin/env python3
"""A1a one-step shared-shortlist Hybrid batch-diversity control."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qgeognn_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.qgeognn_al.acquisition import batch_distance_summary, build_joint_representation, hybrid_select
from src.qgeognn_al.artifacts import finalize_experiment, sha256_file
from src.qgeognn_al.data import condition_matrix, eluent_descriptor, load_combined_graph_cache
from src.qgeognn_al.engine import QGeoGNNActiveLearningEngine, SourceFreeTrainConfig
from src.qgeognn_al.metrics import regression_metric_row
from src.qgeognn_al.partitions import make_row_partition
from src.qgeognn_al.resources import ROOT, SOURCE_DATA, SOURCE_GRAPH_CACHE

STUDY = ROOT / "studies/track_a_4g_al/a1a_hybrid_batch_control"
OUTER_SEEDS = (17, 137, 941, 2027, 4099)
FORBIDDEN_OLD_SEEDS = {42, 525, 1101}


def environment_record() -> dict:
    versions={}
    for name in ["numpy","pandas","torch","torch-geometric","rdkit","scipy","scikit-learn"]:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    return {"python":platform.python_version(),"platform":platform.platform(),"packages":versions,"device":"cpu"}


def member_seeds(outer_seed: int, count: int = 3) -> tuple[int, ...]:
    return tuple(outer_seed * 100 + index for index in range(count))


def control_seed(outer_seed: int, draw_index: int) -> int:
    return int(outer_seed * 100_000 + 70_000 + draw_index)


def random_control_batches(shortlist: list[str], batch_size: int, outer_seed: int, draws: int) -> list[dict]:
    result=[]
    ordered=np.asarray(shortlist,dtype=str)
    for draw in range(draws):
        rng=np.random.default_rng(control_seed(outer_seed,draw))
        selected=ordered[rng.choice(len(ordered),size=batch_size,replace=False)].tolist()
        result.extend({"outer_seed":outer_seed,"control_draw_index":draw,"control_seed":control_seed(outer_seed,draw),"selection_order":rank,"sample_id":sample_id} for rank,sample_id in enumerate(selected))
    return result


def mechanism_gate(seed_summary: pd.DataFrame) -> dict:
    ordered=seed_summary.sort_values("outer_seed").copy()
    ordered["hybrid_minus_random_mean"]=ordered.hybrid_gain-ordered.random_gain_mean
    ordered["hybrid_minus_random_median"]=ordered.hybrid_gain-ordered.random_gain_median
    conditions={
        "hybrid_above_control_median_at_least_4_of_5": int((ordered.hybrid_gain>ordered.random_gain_median).sum())>=4,
        "mean_hybrid_minus_random_mean_positive": float(ordered.hybrid_minus_random_mean.mean())>0,
        "median_hybrid_minus_random_median_positive": float(ordered.hybrid_minus_random_median.median())>0,
        "hybrid_beats_at_least_8_controls_at_least_3_of_5": int((ordered.hybrid_beats_random_count>=8).sum())>=3,
    }
    passed=all(conditions.values())
    return {"diversity_mechanism_supported":passed,"conditions":conditions,"independent_replication_unit":"5 outer seeds, not 50 controls","claim_if_passed":"under a shared Round-0 uncertainty shortlist, farthest-first has stable one-step learning-advantage evidence","does_not_establish":"all Hybrid advantage is caused uniquely by diversity","A1b_eligible_for_manual_review":passed,"A1b_automatic_launch":False}


def make_scaler(data: pd.DataFrame, cache: dict, partition: pd.DataFrame) -> dict:
    positions=partition.loc[partition.role.eq("l0_train"),"canonical_index"].to_numpy(int)
    descriptors=np.vstack([cache[value]["descriptor"] for value in data.canonical_smiles]).astype(np.float32)
    eluents=np.vstack([eluent_descriptor(value) for value in data["PE/EA"]]).astype(np.float32)
    def fit(values: np.ndarray) -> dict: return {"min":values.min(axis=0).tolist(),"max":values.max(axis=0).tolist()}
    return {"fit_split":"a1a_fixed_l0_train","descriptor":fit(descriptors[positions]),"eluent":fit(eluents[positions])}


def ids_by_role(partition: pd.DataFrame) -> dict[str,list[str]]:
    return {role:partition.loc[partition.role.eq(role),"sample_id"].astype(str).tolist() for role in ("l0_train","l0_validation","u0","test")}


def ensemble_predict(engine: QGeoGNNActiveLearningEngine, ids: list[str], checkpoints: dict[int,Path], scales: dict[str,float], embedding: bool=False) -> tuple[pd.DataFrame,np.ndarray|None]:
    tables=[]; primary=None
    for index,(seed,path) in enumerate(checkpoints.items()):
        result=engine.predict(ids,path,return_quantiles=False,return_embedding=embedding and index==0)
        tables.append(result.table[["sample_id","V1_q50","V2_q50"]]);
        if index==0: primary=result.embeddings
    output=tables[0][["sample_id"]].copy(); v1=np.column_stack([x.V1_q50 for x in tables]); v2=np.column_stack([x.V2_q50 for x in tables])
    output["V1_prediction"],output["V2_prediction"]=v1.mean(1),v2.mean(1)
    output["ensemble_score"]=np.var(v1/scales["V1"],axis=1,ddof=1)+np.var(v2/scales["V2"],axis=1,ddof=1)
    return output,primary


def evaluate(engine: QGeoGNNActiveLearningEngine, ids: list[str], prediction: pd.DataFrame, scales: dict[str,float]) -> dict:
    index={value:i for i,value in enumerate(engine.data.sample_id.astype(str))}; positions=[index[value] for value in ids]
    truth=engine.data.iloc[positions][["V1_ml","V2_ml"]].to_numpy(float); pred=prediction[["V1_prediction","V2_prediction"]].to_numpy(float)
    return regression_metric_row(truth,pred,scales)


def fit_members(engine: QGeoGNNActiveLearningEngine, labeled: list[str], validation: list[str], config: SourceFreeTrainConfig, outer_seed: int, output: Path, arm: str) -> tuple[dict[int,Path],list[dict]]:
    checkpoints={}; records=[]
    for member_index,seed in enumerate(member_seeds(outer_seed)):
        result=engine.fit(labeled,validation,config,None,seed,output/f"member_{member_index}")
        checkpoints[seed]=Path(result.checkpoint)
        record=asdict(result); record.update({"outer_seed":outer_seed,"arm":arm,"member_index":member_index,"member_seed":seed})
        records.append(record)
    return checkpoints,records


def batch_metrics(data: pd.DataFrame, ids: list[str], shortlist: set[str], joint: np.ndarray, graph: np.ndarray, conditions: np.ndarray, pool_ids: list[str], arm: str, outer_seed: int, draw: int|None) -> dict:
    position={value:i for i,value in enumerate(pool_ids)}; selected=np.asarray([position[value] for value in ids]); rows=data.set_index("sample_id").loc[ids]
    counts=rows.canonical_smiles.value_counts(); proportions=counts.to_numpy()/len(ids)
    joint_mean,joint_min=batch_distance_summary(joint[selected]); graph_mean,graph_min=batch_distance_summary(graph[selected]); cond_mean,cond_min=batch_distance_summary(conditions[selected])
    return {"outer_seed":outer_seed,"arm":arm,"control_draw_index":draw,"batch_size":len(ids),"all_from_shared_shortlist":set(ids)<=shortlist,"unique_compounds":len(counts),"compound_hhi":float(np.square(proportions).sum()),"joint_mean_pairwise_distance":joint_mean,"joint_min_pairwise_distance":joint_min,"graph_mean_pairwise_distance":graph_mean,"graph_min_pairwise_distance":graph_min,"condition_mean_pairwise_distance":cond_mean,"condition_min_pairwise_distance":cond_min}


def run_seed(seed: int, config_payload: dict) -> dict:
    torch.set_num_threads(int(config_payload.get("torch_threads_per_worker",2)))
    runtime=STUDY/"runtime"/f"seed_{seed}"; runtime.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(SOURCE_DATA); cache=torch.load(SOURCE_GRAPH_CACHE,weights_only=False); partition=pd.read_csv(STUDY/"partitions"/f"a1a_row_seed_{seed}.csv")
    if sha256_file(STUDY/"partitions"/f"a1a_row_seed_{seed}.csv")!=config_payload["partition_hashes"][str(seed)]: raise RuntimeError("partition drift")
    roles=ids_by_role(partition); scaler=make_scaler(data,cache,partition); engine=QGeoGNNActiveLearningEngine(data,cache,scaler,ROOT/"experiments/e0_4g_baseline/checkpoints/best.pt",device=torch.device("cpu"))
    train_config=SourceFreeTrainConfig(**config_payload["train_config"]); l0=roles["l0_train"]+roles["l0_validation"]
    checkpoints,fit_records=fit_members(engine,l0,roles["l0_validation"],train_config,seed,runtime/"round0","round0")
    test_pred,_=ensemble_predict(engine,roles["test"],checkpoints,{t:max(float(data.set_index('sample_id').loc[roles['l0_train'],f'{t}_ml'].std(ddof=0)),1e-8) for t in ('V1','V2')})
    scales={t:max(float(data.set_index('sample_id').loc[roles['l0_train'],f'{t}_ml'].std(ddof=0)),1e-8) for t in ('V1','V2')}; baseline=evaluate(engine,roles["test"],test_pred,scales)["NRMSE"]
    pool,pool_h=ensemble_predict(engine,roles["u0"],checkpoints,scales,True); labeled_result=engine.predict(roles["l0_train"],next(iter(checkpoints.values())),return_quantiles=False,return_embedding=True)
    data_index={value:i for i,value in enumerate(data.sample_id.astype(str))}; lpos=np.asarray([data_index[x] for x in roles["l0_train"]]); ppos=np.asarray([data_index[x] for x in roles["u0"]])
    labeled_joint,pool_joint,_=build_joint_representation(labeled_result.embeddings,pool_h,condition_matrix(data,lpos),condition_matrix(data,ppos))
    hybrid,shortlist=hybrid_select(roles["u0"],pool.ensemble_score.to_numpy(),pool_joint,labeled_joint,int(config_payload["query_size"]),.25)
    controls=random_control_batches(shortlist,int(config_payload["query_size"]),seed,int(config_payload["random_control_draws"])); control_frame=pd.DataFrame(controls)
    shortlist_frame=pool[pool.sample_id.isin(shortlist)].copy(); shortlist_frame["outer_seed"]=seed; shortlist_frame["shortlist_rank"]=shortlist_frame.ensemble_score.rank(method="first",ascending=False).astype(int)
    hybrid_rows=[{"outer_seed":seed,"selection_order":i,"sample_id":value} for i,value in enumerate(hybrid)]
    arms=[("hybrid",None,hybrid)]+[(f"random_{draw}",draw,group.sort_values("selection_order").sample_id.tolist()) for draw,group in control_frame.groupby("control_draw_index")]
    batch_rows=[]; gain_rows=[]; initialization=[]
    graph_z=pool_joint[:,:pool_h.shape[1]]; condition_z=pool_joint[:,pool_h.shape[1]:]
    for arm,draw,selected in arms:
        arm_checkpoints,records=fit_members(engine,l0+selected,roles["l0_validation"],train_config,seed,runtime/arm,arm); fit_records.extend(records)
        after,_=ensemble_predict(engine,roles["test"],arm_checkpoints,scales); after_metric=evaluate(engine,roles["test"],after,scales)["NRMSE"]
        gain_rows.append({"outer_seed":seed,"arm":arm,"control_draw_index":draw,"baseline_NRMSE":baseline,"after_NRMSE":after_metric,"gain":baseline-after_metric})
        batch_rows.append(batch_metrics(data,selected,set(shortlist),pool_joint,graph_z,condition_z,roles["u0"],arm,seed,draw))
        for record in records: initialization.append({"outer_seed":seed,"arm":arm,"member_index":record["member_index"],"member_seed":record["member_seed"],"initial_parameter_hash":record["trainable_parameters_sha256_before"],"validation_ids_hash":record["validation_ids_hash"],"scaler_hash":record["scaler_hash"],"test_used_for_checkpoint_selection":False})
        print(json.dumps({"A1a_arm_complete":arm,"outer_seed":seed,"gain":baseline-after_metric}),flush=True)
    compact_fits=[{k:v for k,v in record.items() if k not in {"checkpoint","init_source_checkpoint"}} for record in fit_records]
    return {"seed":seed,"round0":{"outer_seed":seed,"baseline_NRMSE":baseline},"shortlist":shortlist_frame.to_dict("records"),"hybrid":hybrid_rows,"controls":controls,"batch":batch_rows,"gains":gain_rows,"fits":compact_fits,"initialization":initialization}


def prepare(config_path: Path) -> None:
    config=json.loads(config_path.read_text()); data=pd.read_csv(SOURCE_DATA,usecols=["sample_id","canonical_smiles"]); data["canonical_index"]=np.arange(len(data)); directory=STUDY/"partitions"; directory.mkdir(parents=True,exist_ok=True); hashes={}
    for seed in OUTER_SEEDS:
        partition=make_row_partition(data,seed,config["test_fraction"],config["l0_fraction"],config["validation_fraction"],"A1a","row")
        path=directory/f"a1a_row_seed_{seed}.csv"; partition.to_csv(path,index=False); hashes[str(seed)]=sha256_file(path)
    config["partition_hashes"]=hashes; config_path.write_text(json.dumps(config,indent=2)+"\n"); (STUDY/"environment.json").write_text(json.dumps(environment_record(),indent=2)+"\n")


def run(config_path: Path, workers: int) -> None:
    config=json.loads(config_path.read_text()); runtime=STUDY/"runtime"; runtime.mkdir(parents=True,exist_ok=True)
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(run_seed,seed,config):seed for seed in OUTER_SEEDS}
        for future in concurrent.futures.as_completed(futures): results.append(future.result())
    results=sorted(results,key=lambda item:item["seed"]); generated=runtime/"generated"; generated.mkdir()
    def write_csv(name,key): pd.DataFrame([row for result in results for row in result[key]]).to_csv(generated/name,index=False)
    pd.DataFrame([result["round0"] for result in results]).to_csv(generated/"round0_metrics.csv",index=False)
    for name,key in [("round0_shortlists.csv","shortlist"),("hybrid_batches.csv","hybrid"),("random_control_batches.csv","controls"),("batch_metrics.csv","batch"),("fit_audit.csv","fits"),("initialization_hash_audit.csv","initialization")]: write_csv(name,key)
    gains=pd.DataFrame([row for result in results for row in result["gains"]]); hybrid=gains[gains.arm.eq("hybrid")].set_index("outer_seed")
    summaries=[]
    for seed in OUTER_SEEDS:
        h=float(hybrid.loc[seed,"gain"]); random=gains[(gains.outer_seed.eq(seed))&gains.arm.str.startswith("random_")].gain.to_numpy()
        summaries.append({"outer_seed":seed,"baseline_NRMSE":float(hybrid.loc[seed,"baseline_NRMSE"]),"hybrid_after_NRMSE":float(hybrid.loc[seed,"after_NRMSE"]),"hybrid_gain":h,"random_gain_mean":random.mean(),"random_gain_median":np.median(random),"random_gain_std":random.std(ddof=1),"random_gain_min":random.min(),"random_gain_max":random.max(),"hybrid_minus_random_mean":h-random.mean(),"hybrid_minus_random_median":h-np.median(random),"hybrid_beats_random_count":int((h>random).sum()),"hybrid_percentile_among_controls":float((random<h).mean())})
    summary=pd.DataFrame(summaries); summary.to_csv(generated/"seed_summary.csv",index=False); gate=mechanism_gate(summary); (generated/"mechanism_gate.json").write_text(json.dumps(gate,indent=2)+"\n")
    audit=pd.read_csv(generated/"initialization_hash_audit.csv"); hash_consistent=bool(audit.groupby(["outer_seed","member_index"]).initial_parameter_hash.nunique().eq(1).all()); validation_consistent=bool(audit.groupby("outer_seed").validation_ids_hash.nunique().eq(1).all()); scaler_consistent=bool(audit.groupby("outer_seed").scaler_hash.nunique().eq(1).all())
    runtime_bytes=sum(path.stat().st_size for path in runtime.rglob("*") if path.is_file())
    decision={"study":"A1a","status":"COMPLETED_STOPPED","diversity_mechanism_supported":gate["diversity_mechanism_supported"],"A1b_eligible_for_manual_review":gate["A1b_eligible_for_manual_review"],"A1b_automatic_launch":False,"compound_run_started":False,"advanced_method_started":False,"independent_outer_seeds":5,"random_controls_per_seed":10,"initialization_hashes_consistent":hash_consistent,"validation_consistent":validation_consistent,"scaler_consistent":scaler_consistent,"S1_used_for_acquisition":False,"runtime_bytes_deleted_after_finalization":runtime_bytes}
    (generated/"decision.json").write_text(json.dumps(decision,indent=2)+"\n"); shutil.copy2(config_path,generated/"config.json"); shutil.copy2(STUDY/"environment.json",generated/"environment.json")
    (generated/"README.md").write_text("# A1a — Hybrid One-Step Batch-Diversity Control\n\nCompleted and stopped. Five independent row-level outer seeds compare frozen farthest-first Hybrid against ten random batches from each exact shared Round-0 uncertainty shortlist. Controls within a seed are not independent datasets. See seed_summary.csv and mechanism_gate.json.\n")
    plotdir=generated/"plots"; plotdir.mkdir();
    fig,ax=plt.subplots(figsize=(7,4)); ax.errorbar(summary.outer_seed,summary.random_gain_mean,yerr=summary.random_gain_std,fmt='o',label='random shortlist mean±SD'); ax.scatter(summary.outer_seed,summary.hybrid_gain,label='Hybrid'); ax.legend(); ax.set(xlabel='outer seed',ylabel='one-step gain'); fig.tight_layout(); fig.savefig(plotdir/"hybrid_vs_random_control_gain.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3.5)); ax.bar(summary.outer_seed.astype(str),summary.hybrid_percentile_among_controls); ax.set(ylim=(0,1),ylabel='Hybrid percentile'); fig.tight_layout(); fig.savefig(plotdir/"hybrid_control_percentile.png",dpi=160); plt.close(fig)
    batch=pd.read_csv(generated/"batch_metrics.csv").merge(gains[["outer_seed","arm","gain"]],on=["outer_seed","arm"]); fig,ax=plt.subplots(figsize=(6,4)); ax.scatter(batch.joint_mean_pairwise_distance,batch.gain,c=np.where(batch.arm.eq('hybrid'),1,0),cmap='coolwarm'); ax.set(xlabel='batch mean pairwise joint distance',ylabel='one-step gain'); fig.tight_layout(); fig.savefig(plotdir/"batch_diversity_vs_gain.png",dpi=160); plt.close(fig)
    keep=["README.md","config.json","environment.json","decision.json","round0_metrics.csv","round0_shortlists.csv","hybrid_batches.csv","random_control_batches.csv","batch_metrics.csv","seed_summary.csv","mechanism_gate.json","fit_audit.csv","initialization_hash_audit.csv",*[f"plots/{p.name}" for p in plotdir.iterdir()]]
    decision["retained_compact_bytes_before_manifest"]=sum((generated/path).stat().st_size for path in keep if (generated/path).is_file())
    (generated/"decision.json").write_text(json.dumps(decision,indent=2)+"\n")
    finalize_experiment(generated,STUDY,keep); shutil.rmtree(runtime)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--prepare",action="store_true"); parser.add_argument("--run",action="store_true"); parser.add_argument("--workers",type=int,default=5); parser.add_argument("--config",type=Path,default=STUDY/"config.json"); args=parser.parse_args();
    if args.prepare==args.run: parser.error("choose exactly one of --prepare/--run")
    prepare(args.config) if args.prepare else run(args.config,args.workers)
if __name__=="__main__": main()
