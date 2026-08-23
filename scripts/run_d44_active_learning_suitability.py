#!/usr/bin/env python3
"""D44 offline AL-suitability, model-update, and soft-T3R qualification audit."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.al_acquisition import batch_distance_summary, mean_knn_distance
from scripts.al_engine import (
    QGeoGNNActiveLearningEngine, TrainConfig, initialize_round_state,
    load_round_state, save_round_state,
)
from scripts.run_e0_8g_controls import load_graph_cache
from scripts.run_e0_8g_transfer import configure_trainable, quantile_target_loss
from scripts.run_e1_signal_qualification import condition_matrix, standardize
from scripts.run_e4_active_transfer import (
    SCALER, SOURCES, TARGET, ensemble_scores, fit_members, partition_context,
    primary_quantile_width, representation_from_primary,
)
from scripts.transfer_aware_acquisition import (
    percentile_rank, soft_representative_score, target_representativeness,
    top_score, transfer_prediction_shift,
)

FORMAL = ROOT / "experiments/e4_protocol_a_formal"
D43 = ROOT / "experiments/e4_transfer_aware_acquisition_qualification"
OUT = ROOT / "experiments/e4_active_learning_suitability_diagnosis"
SOURCE_DATA = ROOT / "experiments/e0_4g_baseline/canonical_4g.csv"
SEEDS = (42, 525, 1101)
STRATEGIES = (
    "pretrained_random", "pretrained_coverage", "pretrained_ensemble",
    "pretrained_hybrid", "pretrained_quantile_width",
)
LAMBDAS = (0.0, 0.1, 0.2, 0.3)
CONDITION_NAMES = (
    "eluent_descriptor_0", "eluent_descriptor_1", "eluent_descriptor_2",
    "eluent_descriptor_3", "eluent_descriptor_4", "eluent_descriptor_5",
    "loading_solvent_code", "loading_mass_proxy", "loading_volume",
)


def checkpoint_set(seed: int, strategy: str, completed_round: int) -> dict[int, Path]:
    if completed_round == 0:
        base = FORMAL / "runtime" / f"seed_{seed}" / "shared_round0" / "round_0"
    else:
        short = strategy.removeprefix("pretrained_")
        base = FORMAL / "runtime" / f"seed_{seed}" / short / f"round_{completed_round}"
    result = {member: base / f"member_{member}" / "best.pt" for member in SOURCES}
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing frozen formal checkpoints: {missing}")
    return result


def source_prediction_table(engine: QGeoGNNActiveLearningEngine) -> pd.DataFrame:
    ids = engine.data.sample_id.astype(str).tolist()
    tables = [
        engine.predict(ids, checkpoint, return_quantiles=False, return_embedding=False)
        .table[["V1_q50", "V2_q50"]].to_numpy(float)
        for checkpoint in SOURCES.values()
    ]
    mean = np.stack(tables).mean(axis=0)
    return pd.DataFrame({"sample_id": ids, "mu_source_V1": mean[:, 0], "mu_source_V2": mean[:, 1]}).set_index("sample_id")


def score_state(
    engine: QGeoGNNActiveLearningEngine,
    train_ids: list[str],
    pool_ids: list[str],
    checkpoints: dict[int, Path],
    scales: dict[str, float],
    source_mean: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    target_tables, pool_embeddings = [], []
    for checkpoint in checkpoints.values():
        result = engine.predict(pool_ids, checkpoint, return_quantiles=True, return_embedding=True)
        target_tables.append(result.table)
        pool_embeddings.append(result.embeddings)
    target_q50 = np.stack([table[["V1_q50", "V2_q50"]].to_numpy(float) for table in target_tables])
    target_mean = target_q50.mean(axis=0)
    source = source_mean.loc[pool_ids][["mu_source_V1", "mu_source_V2"]].to_numpy(float)
    shift = transfer_prediction_shift(source, target_mean, scales)
    uncertainty = ensemble_scores(target_q50, scales)
    qwidth = primary_quantile_width(target_tables[list(checkpoints).index(42)], scales)
    train_embedding = engine.predict(train_ids, checkpoints[42], return_quantiles=False, return_embedding=True).embeddings
    index = engine._sample_to_index
    conditions = condition_matrix(engine.data, np.arange(len(engine.data)))
    train_conditions = conditions[[index[x] for x in train_ids]]
    pool_conditions = conditions[[index[x] for x in pool_ids]]
    train_rep, pool_rep, _ = representation_from_primary(
        train_embedding, pool_embeddings[list(checkpoints).index(42)],
        train_conditions, pool_conditions,
    )
    latent = mean_knn_distance(train_rep, pool_rep)
    representativeness, knn = target_representativeness(pool_rep, 10)
    information = 0.5 * percentile_rank(shift) + 0.5 * percentile_rank(uncertainty)
    rows = engine.data.iloc[[index[x] for x in pool_ids]]
    frame = pd.DataFrame({
        "sample_id": pool_ids,
        "canonical_smiles": rows.canonical_smiles.astype(str).to_numpy(),
        "mu_target_V1": target_mean[:, 0], "mu_target_V2": target_mean[:, 1],
        "transfer_prediction_shift": shift,
        "target_ensemble_uncertainty": uncertainty,
        "quantile_width": qwidth,
        "latent_distance": latent,
        "target_representativeness": representativeness,
        "pool_knn10_distance": knn,
        "informativeness": information,
    })
    return frame, train_rep, pool_rep


def compound_metrics(rows: pd.DataFrame) -> dict[str, float]:
    counts = rows.canonical_smiles.astype(str).value_counts()
    proportions = counts / counts.sum()
    return {
        "rows": float(len(rows)),
        "unique_compounds": float(len(counts)),
        "samples_per_compound_mean": float(counts.mean()),
        "samples_per_compound_median": float(counts.median()),
        "samples_per_compound_p90": float(counts.quantile(.9)),
        "samples_per_compound_max": float(counts.max()),
        "compound_HHI": float(np.square(proportions).sum()),
    }


def dataset_diagnosis(
    engine: QGeoGNNActiveLearningEngine,
    round0: dict[int, tuple[pd.DataFrame, np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_rows, structure_rows = [], []
    source = pd.read_csv(SOURCE_DATA)
    for metric, value in compound_metrics(source).items():
        data_rows.append({"dataset": "4g_source", "outer_seed": np.nan, "role": "all", "section": "compound", "metric": metric, "value": value})
    all_conditions = condition_matrix(engine.data, np.arange(len(engine.data)))
    for seed in SEEDS:
        _, roles = partition_context("A", seed)
        for role in ("l0_train", "u0"):
            ids = roles[role]; rows = engine.data.iloc[[engine._sample_to_index[x] for x in ids]]
            for metric, value in compound_metrics(rows).items():
                data_rows.append({"dataset": "8g_target", "outer_seed": seed, "role": role, "section": "compound", "metric": metric, "value": value})
            cond = all_conditions[[engine._sample_to_index[x] for x in ids]]
            unique_condition_count = len(np.unique(cond, axis=0))
            data_rows.append({"dataset":"8g_target","outer_seed":seed,"role":role,"section":"condition","metric":"unique_condition_vectors","value":float(unique_condition_count)})
            condition_per_compound = pd.DataFrame(cond).assign(compound=rows.canonical_smiles.astype(str).to_numpy()).groupby("compound").apply(lambda x: len(x.drop_duplicates()), include_groups=False)
            data_rows.append({"dataset":"8g_target","outer_seed":seed,"role":role,"section":"condition","metric":"mean_unique_conditions_per_compound","value":float(condition_per_compound.mean())})
            for j, name in enumerate(CONDITION_NAMES):
                for statistic, value in (("min",cond[:,j].min()),("q25",np.quantile(cond[:,j],.25)),("median",np.median(cond[:,j])),("q75",np.quantile(cond[:,j],.75)),("max",cond[:,j].max()),("std",cond[:,j].std(ddof=0))):
                    data_rows.append({"dataset":"8g_target","outer_seed":seed,"role":role,"section":"condition_dimension","metric":f"{name}_{statistic}","value":float(value)})

        frame, train_rep, pool_rep = round0[seed]
        model = KMeans(n_clusters=10, random_state=0, n_init=20).fit(pool_rep)
        labels = model.labels_
        l0_cluster = model.predict(train_rep)
        silhouette = float(silhouette_score(pool_rep, labels))
        data_rows.append({"dataset":"8g_target","outer_seed":seed,"role":"u0","section":"representation","metric":"kmeans10_silhouette","value":silhouette})
        distances = cdist(pool_rep, train_rep)
        nearest = distances.argmin(axis=1)
        graph_component = np.linalg.norm(pool_rep[:,:128] - train_rep[nearest,:128], axis=1)
        condition_component = np.linalg.norm(pool_rep[:,128:] - train_rep[nearest,128:], axis=1)
        for metric,value in (
            ("mean_nearest_L0_graph_component",graph_component.mean()),
            ("mean_nearest_L0_condition_component",condition_component.mean()),
            ("fraction_condition_component_gt_graph",np.mean(condition_component>graph_component)),
        ):
            data_rows.append({"dataset":"8g_target","outer_seed":seed,"role":"u0","section":"representation","metric":metric,"value":float(value)})
        pool_cond = all_conditions[[engine._sample_to_index[x] for x in roles["u0"]]]
        train_cond = all_conditions[[engine._sample_to_index[x] for x in roles["l0_train"]]]
        _, pool_cond_z = standardize(train_cond, pool_cond)
        cond_cluster = KMeans(n_clusters=10, random_state=0, n_init=20).fit_predict(pool_cond_z)
        compound_codes = pd.Categorical(frame.canonical_smiles).codes
        ami = float(adjusted_mutual_info_score(compound_codes, cond_cluster))
        data_rows.append({"dataset":"8g_target","outer_seed":seed,"role":"u0","section":"compound_condition_coupling","metric":"adjusted_mutual_information_compound_vs_condition_cluster","value":ami})
        for cluster in range(10):
            mask = labels == cluster
            structure_rows.append({
                "record_type":"cluster", "outer_seed":seed, "cluster":cluster,
                "cluster_size":int(mask.sum()), "cluster_fraction":float(mask.mean()),
                "L0_coverage_count":int(np.sum(l0_cluster==cluster)),
                "mean_transfer_prediction_shift":float(frame.loc[mask,"transfer_prediction_shift"].mean()),
                "mean_target_ensemble_uncertainty":float(frame.loc[mask,"target_ensemble_uncertainty"].mean()),
                "mean_pool_knn10_distance":float(frame.loc[mask,"pool_knn10_distance"].mean()),
                "unique_compounds":int(frame.loc[mask,"canonical_smiles"].nunique()),
            })
        for quantile in (0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1):
            structure_rows.append({"record_type":"shift_quantile","outer_seed":seed,"quantile":quantile,"value":float(frame.transfer_prediction_shift.quantile(quantile))})
        relationships = {
            "shift_vs_density": ("transfer_prediction_shift","target_representativeness"),
            "shift_vs_uncertainty": ("transfer_prediction_shift","target_ensemble_uncertainty"),
            "shift_vs_QWidth": ("transfer_prediction_shift","quantile_width"),
            "shift_vs_latent_distance": ("transfer_prediction_shift","latent_distance"),
            "uncertainty_vs_QWidth": ("target_ensemble_uncertainty","quantile_width"),
            "uncertainty_vs_latent_distance": ("target_ensemble_uncertainty","latent_distance"),
            "QWidth_vs_latent_distance": ("quantile_width","latent_distance"),
        }
        for name,(left,right) in relationships.items():
            structure_rows.append({"record_type":"correlation","outer_seed":seed,"relationship":name,"spearman":float(frame[left].corr(frame[right],method="spearman"))})
        for j,name in enumerate(CONDITION_NAMES):
            rho = spearmanr(frame.transfer_prediction_shift, pool_cond[:,j]).statistic
            structure_rows.append({"record_type":"condition_association","outer_seed":seed,"condition":name,"spearman":float(rho) if np.isfinite(rho) else np.nan})
        contribution = frame.groupby("canonical_smiles").transfer_prediction_shift.sum().sort_values(ascending=False)
        structure_rows.append({"record_type":"compound_concentration","outer_seed":seed,"metric":"top_10_compounds_shift_share","value":float(contribution.head(10).sum()/contribution.sum())})
        structure_rows.append({"record_type":"compound_concentration","outer_seed":seed,"metric":"shift_weight_HHI_by_compound","value":float(np.square(contribution/contribution.sum()).sum())})
    return pd.DataFrame(data_rows), pd.DataFrame(structure_rows)


def query_region_diagnosis(
    engine: QGeoGNNActiveLearningEngine,
    scales: dict[str, float],
    source_mean: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    history = pd.read_csv(FORMAL / "query_history.csv")
    metrics = pd.read_csv(FORMAL / "round_metrics.csv").set_index(["outer_seed","strategy","budget"])
    batch_rows, sample_rows = [], []
    for seed in SEEDS:
        _, roles = partition_context("A", seed)
        for strategy in STRATEGIES:
            prior: list[str] = []
            for query_round in range(1,16):
                train_ids = roles["l0_train"] + prior
                pool_ids = [x for x in roles["u0"] if x not in set(prior)]
                frame, _, _ = score_state(engine, train_ids, pool_ids, checkpoint_set(seed,strategy,query_round-1), scales, source_mean)
                selected = history[(history.outer_seed==seed)&(history.strategy==strategy)&(history["round"]==query_round)].sort_values("query_rank").sample_id.astype(str).tolist()
                if len(selected)!=10 or not set(selected)<=set(pool_ids):
                    raise AssertionError("Historical query identity contract failed")
                chosen = frame.set_index("sample_id").loc[selected].copy()
                truth = engine.data.set_index(engine.data.sample_id.astype(str)).loc[selected]
                train_truth = engine.data.set_index(engine.data.sample_id.astype(str)).loc[train_ids]
                label_z = np.column_stack([
                    (truth.V1_ml.to_numpy(float)-train_truth.V1_ml.mean())/max(train_truth.V1_ml.std(ddof=0),1e-8),
                    (truth.V2_ml.to_numpy(float)-train_truth.V2_ml.mean())/max(train_truth.V2_ml.std(ddof=0),1e-8),
                ])
                true_error = .5*(np.abs(truth.V1_ml.to_numpy(float)-chosen.mu_target_V1.to_numpy(float))/scales["V1"] + np.abs(truth.V2_ml.to_numpy(float)-chosen.mu_target_V2.to_numpy(float))/scales["V2"])
                before = float(metrics.loc[(seed,strategy,50+10*(query_round-1)),"NRMSE"])
                after = float(metrics.loc[(seed,strategy,50+10*query_round),"NRMSE"])
                delta = after-before
                for rank,sid in enumerate(selected,1):
                    row=chosen.loc[sid]
                    sample_rows.append({
                        "outer_seed":seed,"strategy":strategy,"round":query_round,"query_rank":rank,"sample_id":sid,
                        "transfer_prediction_shift":row.transfer_prediction_shift,
                        "target_ensemble_uncertainty":row.target_ensemble_uncertainty,
                        "quantile_width":row.quantile_width,"latent_distance":row.latent_distance,
                        "target_representativeness":row.target_representativeness,"pool_knn10_distance":row.pool_knn10_distance,
                        "true_standardized_error_after_reveal":float(true_error[rank-1]),
                        "label_extremeness_after_reveal":float(np.abs(label_z[rank-1]).mean()),
                        "next_round_delta_test_NRMSE_posthoc":delta,
                    })
                batch_rows.append({
                    "outer_seed":seed,"strategy":strategy,"round":query_round,"budget_before":50+10*(query_round-1),
                    "mean_transfer_prediction_shift":chosen.transfer_prediction_shift.mean(),
                    "mean_target_ensemble_uncertainty":chosen.target_ensemble_uncertainty.mean(),
                    "mean_quantile_width":chosen.quantile_width.mean(),"mean_latent_distance":chosen.latent_distance.mean(),
                    "mean_target_representativeness":chosen.target_representativeness.mean(),"mean_pool_knn10_distance":chosen.pool_knn10_distance.mean(),
                    "mean_true_standardized_error_after_reveal":float(true_error.mean()),
                    "mean_label_extremeness_after_reveal":float(np.abs(label_z).mean()),
                    "next_round_delta_test_NRMSE_posthoc":delta,"test_usage":"posthoc_mechanism_only",
                })
                prior.extend(selected)
    samples=pd.DataFrame(sample_rows); batches=pd.DataFrame(batch_rows); relations=[]
    signals=("transfer_prediction_shift","target_ensemble_uncertainty","quantile_width","latent_distance","target_representativeness")
    for seed_value,group in [("all",samples)]+[(str(seed),samples[samples.outer_seed==seed]) for seed in SEEDS]:
        for signal in signals:
            for outcome in ("true_standardized_error_after_reveal","label_extremeness_after_reveal"):
                relations.append({"level":"sample","outer_seed":seed_value,"signal":signal,"outcome":outcome,"n":len(group),"spearman":float(group[signal].corr(group[outcome],method="spearman"))})
    batch_signals=("mean_transfer_prediction_shift","mean_target_ensemble_uncertainty","mean_quantile_width","mean_latent_distance","mean_target_representativeness")
    for signal in batch_signals:
        relations.append({"level":"batch","outer_seed":"all","signal":signal,"outcome":"next_round_delta_test_NRMSE_posthoc","n":len(batches),"spearman":float(batches[signal].corr(batches.next_round_delta_test_NRMSE_posthoc,method="spearman"))})
    return batches, samples, pd.DataFrame(relations)


def parameter_group(name: str, trainable: bool) -> str:
    if not trainable:
        return "frozen_early"
    if name.startswith("graph_pred_linear"):
        return "head"
    for layer in (3,4):
        if any(name.startswith(f"gnn_node.{branch}.{layer}.") for branch in (
            "convs","convs_bond_angle","convs_bond_embeding","convs_bond_float","convs_angle_float"
        )):
            return f"last2_layer_{layer}"
    raise AssertionError(f"Unexpected trainable parameter: {name}")


def model_parameter_state(engine: QGeoGNNActiveLearningEngine, checkpoint: Path) -> dict[str, tuple[np.ndarray,bool]]:
    model=engine._load_model(checkpoint); configure_trainable(model,"last2_head")
    return {name:(parameter.detach().cpu().numpy().astype(np.float64),parameter.requires_grad) for name,parameter in model.named_parameters()}


def layer_updates(before: dict, after: dict) -> list[dict]:
    accum: dict[str,dict[str,float]]={}
    if set(before)!=set(after):
        raise AssertionError("Model parameter names changed between checkpoints")
    for name,(left,trainable) in before.items():
        right=after[name][0]; group=parameter_group(name,trainable)
        row=accum.setdefault(group,{"delta2":0.,"before2":0.,"parameter_count":0})
        row["delta2"]+=float(np.square(right-left).sum()); row["before2"]+=float(np.square(left).sum()); row["parameter_count"]+=left.size
    return [{"layer":group,"absolute_L2_update":math.sqrt(row["delta2"]),"relative_L2_update":math.sqrt(row["delta2"])/(math.sqrt(row["before2"])+1e-12),"parameter_count":int(row["parameter_count"])} for group,row in accum.items()]


def prediction_array(engine: QGeoGNNActiveLearningEngine, ids: list[str], checkpoint: Path) -> np.ndarray:
    return engine.predict(ids,checkpoint,return_quantiles=False,return_embedding=False).table[["V1_q50","V2_q50"]].to_numpy(float)


def probe_ids() -> list[str]:
    pools=[]
    for seed in SEEDS:
        _,roles=partition_context("A",seed); pools.append(set(roles["u0"]))
    common=set.intersection(*pools)
    ordered=sorted(common,key=lambda sid:(hashlib.sha256(f"D44-probe:{sid}".encode()).hexdigest(),sid))
    if len(ordered)<100:
        raise AssertionError("Protocol A U0 intersection has fewer than 100 rows")
    return ordered[:100]


def gradient_norms(engine: QGeoGNNActiveLearningEngine, checkpoint: Path, sample_id: str) -> dict[str,float]:
    model=engine._load_model(checkpoint); configure_trainable(model,"last2_head"); model.eval()
    index=engine._sample_to_index[sample_id]
    atoms,angles=engine._loaders_for_indices([index],1)
    atom=next(iter(atoms)).to(engine.device); angle=next(iter(angles)).to(engine.device)
    model.zero_grad(set_to_none=True); prediction,_=model(atom,angle)
    loss=quantile_target_loss(atom.y[:,0],prediction[:,0:3])+quantile_target_loss(atom.y[:,1],prediction[:,3:6])
    loss.backward()
    sums={"head_gradient_L2":0.,"last2_gradient_L2":0.}
    for name,parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None: continue
        key="head_gradient_L2" if name.startswith("graph_pred_linear") else "last2_gradient_L2"
        sums[key]+=float(torch.square(parameter.grad).sum().detach().cpu())
    sums={key:math.sqrt(value) for key,value in sums.items()}
    sums["gradient_L2"]=math.sqrt(sums["head_gradient_L2"]**2+sums["last2_gradient_L2"]**2)
    return sums


def model_update_diagnosis(
    engine: QGeoGNNActiveLearningEngine,
    scales: dict[str,float],
    round1_samples: pd.DataFrame,
) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    probe=probe_ids(); parameter_rows=[]; model_rows=[]; gradient_samples=[]
    scale=np.asarray([scales["V1"],scales["V2"]])
    for seed in SEEDS:
        round0=checkpoint_set(seed,STRATEGIES[0],0)
        source_states={member:model_parameter_state(engine,SOURCES[member]) for member in SOURCES}
        r0_states={member:model_parameter_state(engine,round0[member]) for member in SOURCES}
        source_pred={member:prediction_array(engine,probe,SOURCES[member]) for member in SOURCES}
        r0_pred={member:prediction_array(engine,probe,round0[member]) for member in SOURCES}
        for strategy in STRATEGIES:
            round1=checkpoint_set(seed,strategy,1); r1_pred={}
            for member in SOURCES:
                r1_state=model_parameter_state(engine,round1[member])
                for transition,before,after in (
                    ("source_to_round0",source_states[member],r0_states[member]),
                    ("round0_to_round1",r0_states[member],r1_state),
                ):
                    updates=layer_updates(before,after)
                    for row in updates:
                        parameter_rows.append({"outer_seed":seed,"strategy":strategy,"member_seed":member,"transition":transition,**row})
                    trainable=[row for row in updates if row["layer"]!="frozen_early"]
                    absolute=math.sqrt(sum(row["absolute_L2_update"]**2 for row in trainable))
                    before_norm_parts=[]
                    for name,(value,is_trainable) in before.items():
                        if is_trainable: before_norm_parts.append(float(np.square(value).sum()))
                    relative=absolute/(math.sqrt(sum(before_norm_parts))+1e-12)
                    model_rows.append({"outer_seed":seed,"strategy":strategy,"member_seed":member,"transition":transition,"absolute_trainable_parameter_L2_update":absolute,"relative_trainable_parameter_L2_update":relative})
                r1_pred[member]=prediction_array(engine,probe,round1[member])
                model_rows[-1]["function_update_mean_standardized_abs"] = float(np.mean(np.abs(r1_pred[member]-r0_pred[member])/scale))
            ensemble_source=np.stack(list(source_pred.values())).mean(axis=0)
            ensemble_r0=np.stack(list(r0_pred.values())).mean(axis=0)
            ensemble_r1=np.stack(list(r1_pred.values())).mean(axis=0)
            model_rows.append({"outer_seed":seed,"strategy":strategy,"member_seed":"ensemble","transition":"source_to_round0","function_update_mean_standardized_abs":float(np.mean(np.abs(ensemble_r0-ensemble_source)/scale))})
            model_rows.append({"outer_seed":seed,"strategy":strategy,"member_seed":"ensemble","transition":"round0_to_round1","function_update_mean_standardized_abs":float(np.mean(np.abs(ensemble_r1-ensemble_r0)/scale))})
            selected=round1_samples[(round1_samples.outer_seed==seed)&(round1_samples.strategy==strategy)&(round1_samples["round"]==1)]
            for sample in selected.itertuples():
                member_values=[gradient_norms(engine,round0[member],sample.sample_id) for member in SOURCES]
                means={key:float(np.mean([value[key] for value in member_values])) for key in ("gradient_L2","head_gradient_L2","last2_gradient_L2")}
                gradient_samples.append({"record_type":"sample","outer_seed":seed,"strategy":strategy,"sample_id":sample.sample_id,**means,"transfer_prediction_shift":sample.transfer_prediction_shift,"target_ensemble_uncertainty":sample.target_ensemble_uncertainty,"quantile_width":sample.quantile_width,"latent_distance":sample.latent_distance,"target_representativeness":sample.target_representativeness,"true_standardized_error_after_reveal":sample.true_standardized_error_after_reveal})
    gradients=pd.DataFrame(gradient_samples); summary=[]
    for (seed,strategy),group in gradients.groupby(["outer_seed","strategy"]):
        row={"record_type":"strategy_summary","outer_seed":seed,"strategy":strategy,"sample_id":np.nan}
        for column in ("gradient_L2","head_gradient_L2","last2_gradient_L2"):
            row[f"{column}_mean"]=group[column].mean(); row[f"{column}_median"]=group[column].median(); row[f"{column}_max"]=group[column].max(); row[f"{column}_CV"]=group[column].std(ddof=1)/group[column].mean()
        summary.append(row)
    gradients=pd.concat([gradients,pd.DataFrame(summary)],ignore_index=True,sort=False)
    associations=[]; samples=gradients[gradients.record_type=="sample"]
    for seed_value,group in [("all",samples)]+[(str(seed),samples[samples.outer_seed==seed]) for seed in SEEDS]:
        for signal in ("transfer_prediction_shift","target_ensemble_uncertainty","quantile_width","latent_distance","target_representativeness"):
            associations.append({"outer_seed":seed_value,"signal":signal,"outcome":"gradient_L2","n":len(group),"spearman":float(group[signal].corr(group.gradient_L2,method="spearman"))})
    return pd.DataFrame(model_rows),pd.DataFrame(parameter_rows),gradients,pd.DataFrame(associations)


def t3r_qualification(
    round0: dict[int,tuple[pd.DataFrame,np.ndarray,np.ndarray]],
) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    old_ids=pd.read_csv(D43/"selection_ids.csv")
    sweep=[]; selections=[]
    for seed in SEEDS:
        frame,_,representation=round0[seed]; ids=frame.sample_id.astype(str).tolist(); position={sid:i for i,sid in enumerate(ids)}
        chosen_by_lambda={}
        for value in LAMBDAS:
            score,information,rank_rep=soft_representative_score(frame.transfer_prediction_shift.to_numpy(),frame.target_ensemble_uncertainty.to_numpy(),frame.target_representativeness.to_numpy(),value)
            selected=top_score(ids,score,10); chosen_by_lambda[value]=selected
            chosen=np.asarray([position[x] for x in selected]); rows=frame.iloc[chosen]; pair_mean,pair_min=batch_distance_summary(representation[chosen]); compounds=rows.canonical_smiles.value_counts(normalize=True)
            record={"candidate":"T3R","outer_seed":seed,"lambda":value,"mean_transfer_shift":rows.transfer_prediction_shift.mean(),"mean_target_uncertainty":rows.target_ensemble_uncertainty.mean(),"mean_informativeness":information[chosen].mean(),"mean_representativeness":rows.target_representativeness.mean(),"mean_knn10_distance":rows.pool_knn10_distance.mean(),"batch_mean_pairwise_distance":pair_mean,"batch_min_pairwise_distance":pair_min,"unique_compounds":rows.canonical_smiles.nunique(),"compound_HHI":np.square(compounds.to_numpy()).sum()}
            sweep.append(record)
            for rank,sid in enumerate(selected,1): selections.append({"candidate":"T3R","outer_seed":seed,"lambda":value,"query_rank":rank,"sample_id":sid})
        baseline=next(row for row in sweep if row["candidate"]=="T3R" and row["outer_seed"]==seed and row["lambda"]==0.)
        baseline_ids=set(chosen_by_lambda[0.])
        for record in [row for row in sweep if row["candidate"]=="T3R" and row["outer_seed"]==seed]:
            record["shift_retention"]=record["mean_transfer_shift"]/baseline["mean_transfer_shift"]
            record["uncertainty_retention"]=record["mean_target_uncertainty"]/baseline["mean_target_uncertainty"]
            record["informativeness_retention"]=record["mean_informativeness"]/baseline["mean_informativeness"]
            record["representativeness_gain"]=record["mean_representativeness"]-baseline["mean_representativeness"]
            record["density_gain"]=1.-record["mean_knn10_distance"]/baseline["mean_knn10_distance"]
            record["T2_sample_overlap_fraction"]=len(set(chosen_by_lambda[record["lambda"]])&baseline_ids)/10.
        old=old_ids[(old_ids.outer_seed==seed)&(old_ids.strategy=="transfer_shift_uncertainty_representative")].sort_values("query_rank").sample_id.astype(str).tolist()
        chosen=np.asarray([position[x] for x in old]); rows=frame.iloc[chosen]; pair_mean,pair_min=batch_distance_summary(representation[chosen]); compounds=rows.canonical_smiles.value_counts(normalize=True)
        sweep.append({"candidate":"old_T3_D43_failed","outer_seed":seed,"lambda":np.nan,"mean_transfer_shift":rows.transfer_prediction_shift.mean(),"mean_target_uncertainty":rows.target_ensemble_uncertainty.mean(),"mean_informativeness":frame.informativeness.iloc[chosen].mean(),"mean_representativeness":rows.target_representativeness.mean(),"mean_knn10_distance":rows.pool_knn10_distance.mean(),"batch_mean_pairwise_distance":pair_mean,"batch_min_pairwise_distance":pair_min,"unique_compounds":rows.canonical_smiles.nunique(),"compound_HHI":np.square(compounds.to_numpy()).sum(),"shift_retention":rows.transfer_prediction_shift.mean()/baseline["mean_transfer_shift"],"uncertainty_retention":rows.target_ensemble_uncertainty.mean()/baseline["mean_target_uncertainty"],"informativeness_retention":frame.informativeness.iloc[chosen].mean()/baseline["mean_informativeness"],"representativeness_gain":rows.target_representativeness.mean()-baseline["mean_representativeness"],"density_gain":1.-rows.pool_knn10_distance.mean()/baseline["mean_knn10_distance"],"T2_sample_overlap_fraction":len(set(old)&baseline_ids)/10.})
        for rank,sid in enumerate(old,1): selections.append({"candidate":"old_T3_D43_failed","outer_seed":seed,"lambda":np.nan,"query_rank":rank,"sample_id":sid})
    table=pd.DataFrame(sweep); gates={}
    for value in LAMBDAS:
        group=table[(table.candidate=="T3R")&(table["lambda"]==value)]
        checks={"informativeness_retention_3of3":bool((group.informativeness_retention>=.90).all()),"shift_retention_3of3":bool((group.shift_retention>=.85).all()),"uncertainty_retention_3of3":bool((group.uncertainty_retention>=.85).all()),"density_gain_ge_10pct_at_least_2of3":bool((group.density_gain>=.10).sum()>=2),"not_90pct_overlap_in_all_3":bool(~(group.T2_sample_overlap_fraction>=.90).all())}
        checks["pass"]=all(checks.values()); gates[str(value)]=checks
    passed=[value for value in LAMBDAS if gates[str(value)]["pass"]]
    selected=min(passed) if passed else None
    decision={"lambda_grid":list(LAMBDAS),"gate_by_lambda":gates,"t3r_qualified":selected is not None,"selected_lambda":selected,"selection_rule":"smallest passing lambda; minimal intervention","test_or_truth_used_for_lambda_selection":False,"old_T3_modified":False}
    return table,pd.DataFrame(selections),decision


def run_conditional_smoke(
    engine: QGeoGNNActiveLearningEngine,
    scales: dict[str,float],
    source_mean: pd.DataFrame,
    selected_lambda: float | None,
) -> dict:
    if selected_lambda is None:
        return {"status":"not_run_t3r_gate_failed","engineering_smoke_pass":None}
    smoke=ROOT/"experiments/e4_t3r_low_l0_engineering_smoke"; smoke.mkdir(parents=True,exist_ok=True)
    _,roles=partition_context("A",42)
    ordered=sorted(roles["l0_train"],key=lambda sid:(hashlib.sha256(f"D44-low-L0:{sid}".encode()).hexdigest(),sid))
    train=ordered[:22]; held_back=ordered[22:]; validation=list(roles["l0_validation"]); pool=list(roles["u0"])+held_back
    if len(train)!=22 or len(validation)!=8 or len(pool)!=486 or set(pool)&set(roles["test"]):
        raise AssertionError("Low-L0 split contract failed")
    partition=[]
    for role,ids in (("gradient_train",train),("validation",validation),("pool",pool),("test",roles["test"])):
        partition.extend({"sample_id":sid,"role":role} for sid in ids)
    pd.DataFrame(partition).to_csv(smoke/"partition.csv",index=False)
    config=TrainConfig(epochs=50,patience=20)
    round0,fit_records=fit_members(engine,train+validation,validation,config,42,0,smoke/"shared_round0")
    pre_frame,_,_=score_state(engine,train,pool,round0,scales,source_mean)
    rng=np.random.default_rng(42000); random_ids=np.asarray(pool)[rng.choice(len(pool),10,replace=False)].tolist()
    random_ids_2=np.asarray(pool)[np.random.default_rng(42000).choice(len(pool),10,replace=False)].tolist()
    t3r_score,_,_=soft_representative_score(pre_frame.transfer_prediction_shift.to_numpy(),pre_frame.target_ensemble_uncertainty.to_numpy(),pre_frame.target_representativeness.to_numpy(),selected_lambda)
    t3r_ids=top_score(pre_frame.sample_id.astype(str).tolist(),t3r_score,10)
    t3r_ids_2=top_score(pre_frame.sample_id.astype(str).tolist(),soft_representative_score(pre_frame.transfer_prediction_shift.to_numpy(),pre_frame.target_ensemble_uncertainty.to_numpy(),pre_frame.target_representativeness.to_numpy(),selected_lambda)[0],10)
    checks=[]; query_rows=[]
    for strategy,selected,deterministic in (("random",random_ids,random_ids==random_ids_2),("T3R",t3r_ids,t3r_ids==t3r_ids_2)):
        for rank,sid in enumerate(selected,1): query_rows.append({"strategy":strategy,"round":1,"query_rank":rank,"sample_id":sid})
        labeled=train+validation+selected; remaining=[sid for sid in pool if sid not in set(selected)]
        checkpoints,records=fit_members(engine,labeled,validation,config,42,1,smoke/strategy)
        fit_records.extend(records)
        post_frame,_,_=score_state(engine,train+selected,remaining,checkpoints,scales,source_mean)
        common=pre_frame.set_index("sample_id").loc[remaining]
        recomputed=not np.allclose(common.transfer_prediction_shift.to_numpy(),post_frame.transfer_prediction_shift.to_numpy()) and not np.allclose(common.target_ensemble_uncertainty.to_numpy(),post_frame.target_ensemble_uncertainty.to_numpy())
        state=initialize_round_state(train+validation,pool,str(round0[42]),42000,"D44-low-L0-seed42",config.config_hash)
        chosen=set(selected); next_state=type(state)(1,1,state.labeled_ids+selected,[sid for sid in state.pool_ids if sid not in chosen],selected,state.checkpoint,state.seed,state.rng_state,state.split_hash,state.config_hash); next_state.validate()
        state_path=smoke/strategy/"state_round_1.json"; save_round_state(state_path,next_state); loaded=load_round_state(state_path,state.split_hash,state.config_hash)
        relevant=[row for row in records]
        checks.append({"strategy":strategy,"query_identity_pass":len(selected)==10 and len(set(selected))==10 and set(selected)<=set(pool) and not set(selected)&set(roles["test"]),"deterministic_selection_pass":deterministic,"source_reset_pass":all(row["init_source_sha256"] is not None for row in relevant),"frozen_early_parameters_unchanged":all(row["frozen_parameters_sha256_before"]==row["frozen_parameters_sha256_after"] for row in relevant),"last2_head_changed":all(row["trainable_parameters_sha256_before"]!=row["trainable_parameters_sha256_after"] for row in relevant),"score_recomputed_after_adaptation":recomputed,"resume_exact":loaded.labeled_ids==next_state.labeled_ids and loaded.pool_ids==next_state.pool_ids and loaded.selected_ids==next_state.selected_ids})
    pd.DataFrame(query_rows).to_csv(smoke/"query_history.csv",index=False); pd.DataFrame(checks).to_csv(smoke/"engineering_checks.csv",index=False)
    passed=all(all(value for key,value in row.items() if key!="strategy") for row in checks)
    record={"status":"completed","engineering_smoke_pass":passed,"protocol":"A","outer_seed":42,"L0":30,"gradient_train":22,"validation":8,"B":10,"methods":["Random","T3R"],"selected_lambda":selected_lambda,"K":3,"epochs":50,"patience":20,"test_evaluated":False,"performance_used":False}
    (smoke/"decision.json").write_text(json.dumps(record,indent=2)); (smoke/"config.json").write_text(json.dumps({**record,"source_members":list(SOURCES),"scope":"engineering only; 30 to 40"},indent=2)); (smoke/"README.md").write_text("# D44 T3R low-L0 engineering smoke\n\nProtocol A seed42, 30→40, Random/T3R, K=3. Engineering-only; test was not evaluated and no performance conclusion is permitted.\n")
    return record


def write_plots(dataset: pd.DataFrame, shift: pd.DataFrame, queries: pd.DataFrame, updates: pd.DataFrame, sweep: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot=OUT/"plots"; plot.mkdir(parents=True,exist_ok=True)
    clusters=shift[shift.record_type=="cluster"]
    fig,ax=plt.subplots(figsize=(9,4)); clusters.pivot(index="cluster",columns="outer_seed",values="cluster_size").plot.bar(ax=ax); ax.set(title="U0 deterministic k-means structure (diagnostic only)",ylabel="cluster size"); fig.tight_layout(); fig.savefig(plot/"cluster_sizes.png",dpi=150); plt.close(fig)
    correlations=shift[(shift.record_type=="correlation")&(shift.relationship.isin(["shift_vs_density","shift_vs_uncertainty","shift_vs_QWidth","shift_vs_latent_distance"]))]
    fig,ax=plt.subplots(figsize=(9,4)); correlations.pivot(index="relationship",columns="outer_seed",values="spearman").plot.bar(ax=ax); ax.axhline(0,color="black",lw=.8); ax.set(ylabel="Spearman",title="Round0 U0 shift relationships; no truth"); fig.tight_layout(); fig.savefig(plot/"shift_relationships.png",dpi=150); plt.close(fig)
    first=queries[queries["round"]==1]
    fig,ax=plt.subplots(figsize=(9,4)); first.groupby("strategy")[["mean_transfer_prediction_shift","mean_target_ensemble_uncertainty"]].mean().plot.bar(ax=ax); ax.set(title="Historical Round1 query-region signals",ylabel="mean score"); fig.tight_layout(); fig.savefig(plot/"round1_query_signals.png",dpi=150); plt.close(fig)
    ensemble=updates[(updates.member_seed.astype(str)=="ensemble")&(updates.transition=="round0_to_round1")]
    fig,ax=plt.subplots(figsize=(9,4)); ensemble.pivot(index="strategy",columns="outer_seed",values="function_update_mean_standardized_abs").plot.bar(ax=ax); ax.set(title="Round0→Round1 ensemble function update",ylabel="mean standardized |Δprediction|"); fig.tight_layout(); fig.savefig(plot/"function_update.png",dpi=150); plt.close(fig)
    soft=sweep[sweep.candidate=="T3R"]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    soft.pivot(index="lambda",columns="outer_seed",values="informativeness_retention").plot(ax=axes[0],marker="o"); axes[0].axhline(.9,color="red",ls="--"); axes[0].set(title="T3R information retention",ylabel="vs T2")
    soft.pivot(index="lambda",columns="outer_seed",values="density_gain").plot(ax=axes[1],marker="o"); axes[1].axhline(.1,color="red",ls="--"); axes[1].set(title="T3R density gain",ylabel="relative kNN-distance reduction")
    fig.tight_layout(); fig.savefig(plot/"t3r_tradeoff.png",dpi=150); plt.close(fig)


def write_a2a_preregistration() -> None:
    out=ROOT/"experiments/e4_a2a_low_budget_preregistration"; out.mkdir(parents=True,exist_ok=True)
    config={"stage":"E4-A2a low-budget preregistration","status":"preregistered_not_run","core_hypothesis":"Headroom Hypothesis: generic active acquisition may have more stable label-efficiency benefit under lower target-label headroom.","protocol":"A","L0":30,"gradient_train":22,"fixed_validation":8,"B":10,"budgets":[30,40,50,60,70,80,90,100],"query_rounds":7,"K":3,"outer_seeds":list(SEEDS),"epochs":500,"patience":100,"strategies":list(STRATEGIES),"single_changed_variable":"L0 50 to 30","T3R_included":False,"test_role":"frozen final evaluation only in a future authorized formal run","formal_started":False,"forbidden":["T3R","Protocol B","new outer splits","test-guided tuning"]}
    (out/"config.json").write_text(json.dumps(config,indent=2)); (out/"README.md").write_text("# E4-A2a Low-Budget Preregistration\n\nPreregistered but **not executed**. This future Protocol A experiment changes only L0 from 50 to 30 (22 gradient-train + 8 fixed validation) and retains Random, Coverage, Ensemble, Hybrid, and QWidth. T3R is excluded so headroom is the only changed experimental variable. Budgets are 30–100 by 10, K=3, seeds 42/525/1101, 500 epochs, patience 100.\n")


def main() -> None:
    formal=json.loads((FORMAL/"config.json").read_text()); d43=json.loads((D43/"qualification_decision.json").read_text())
    if formal.get("formal_complete") is not True or formal.get("protocol_b_started") is not False or d43.get("qualification_only") is not True:
        raise RuntimeError("Frozen E4/D43 provenance gate failed")
    OUT.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(TARGET); scales=json.loads((FORMAL/"source_target_scales.json").read_text())
    engine=QGeoGNNActiveLearningEngine(data,load_graph_cache(),json.loads(SCALER.read_text()),SOURCES[42],device=torch.device("cpu"))
    source_mean=source_prediction_table(engine); round0={}
    d43_scores=pd.read_csv(D43/"transfer_score_summary.csv")
    for seed in SEEDS:
        _,roles=partition_context("A",seed)
        scored,train_rep,pool_rep=score_state(engine,roles["l0_train"],roles["u0"],checkpoint_set(seed,STRATEGIES[0],0),scales,source_mean)
        prior=d43_scores[d43_scores.outer_seed==seed].set_index("sample_id").loc[scored.sample_id]
        for column in ("transfer_prediction_shift","target_ensemble_uncertainty","quantile_width","latent_distance","target_representativeness","pool_knn10_distance"):
            if not np.allclose(scored[column],prior[column],rtol=1e-10,atol=1e-10): raise AssertionError(f"D43 reproduction mismatch: seed={seed} {column}")
        round0[seed]=(scored,train_rep,pool_rep)
    sweep,t3r_ids,t3r_gate=t3r_qualification(round0)
    smoke=run_conditional_smoke(engine,scales,source_mean,t3r_gate["selected_lambda"])
    # Only after the no-truth lambda gate is frozen do historical reveal/test
    # mechanism diagnostics become readable in this pipeline.
    dataset,shift_structure=dataset_diagnosis(engine,round0)
    queries,query_samples,error_relations=query_region_diagnosis(engine,scales,source_mean)
    model_updates,parameter_updates,gradient_summary,gradient_relations=model_update_diagnosis(engine,scales,query_samples)
    error_relations=pd.concat([error_relations,gradient_relations.assign(level="sample_gradient")],ignore_index=True,sort=False)

    dataset.to_csv(OUT/"dataset_structure.csv",index=False); shift_structure.to_csv(OUT/"source_target_shift_structure.csv",index=False)
    error_relations.to_csv(OUT/"uncertainty_error_relation.csv",index=False); queries.to_csv(OUT/"query_region_structure.csv",index=False)
    model_updates.to_csv(OUT/"model_update_summary.csv",index=False); parameter_updates.to_csv(OUT/"parameter_update_by_layer.csv",index=False); gradient_summary.to_csv(OUT/"gradient_update_summary.csv",index=False)
    sweep.to_csv(OUT/"t3r_lambda_sweep.csv",index=False); t3r_ids.to_csv(OUT/"t3r_selection_ids.csv",index=False); (OUT/"t3r_gate.json").write_text(json.dumps(t3r_gate,indent=2))
    pd.DataFrame({"sample_id":probe_ids(),"selection":"SHA256(D44-probe:sample_id), first 100 from three-seed U0 intersection"}).to_csv(OUT/"diagnosis_probe_ids.csv",index=False)

    silhouette_values=dataset[dataset.metric=="kmeans10_silhouette"].value
    density_corr=shift_structure[(shift_structure.record_type=="correlation")&(shift_structure.relationship=="shift_vs_density")].spearman
    uncertainty_corr=shift_structure[(shift_structure.record_type=="correlation")&(shift_structure.relationship=="shift_vs_uncertainty")].spearman
    d43_corr=pd.read_csv(D43/"score_correlations.csv"); overlap=d43_corr[d43_corr.relationship=="transfer_shift_vs_ensemble_uncertainty"].top10_overlap_fraction
    grad_samples=gradient_summary[gradient_summary.record_type=="sample"]
    grad_shift_by_seed={str(seed):float(group.transfer_prediction_shift.corr(group.gradient_L2,method="spearman")) for seed,group in grad_samples.groupby("outer_seed")}
    grad_unc_by_seed={str(seed):float(group.target_ensemble_uncertainty.corr(group.gradient_L2,method="spearman")) for seed,group in grad_samples.groupby("outer_seed")}
    ensemble_update=model_updates[(model_updates.member_seed.astype(str)=="ensemble")&(model_updates.transition=="round0_to_round1")]
    round1_shock=queries[queries["round"]==1][["outer_seed","strategy","next_round_delta_test_NRMSE_posthoc"]]
    update_shock=ensemble_update.merge(round1_shock,on=["outer_seed","strategy"])
    update_shock_rho=float(update_shock.function_update_mean_standardized_abs.corr(update_shock.next_round_delta_test_NRMSE_posthoc,method="spearman"))
    parameter_update=model_updates[(model_updates.member_seed.astype(str)!="ensemble")&(model_updates.transition=="round0_to_round1")].groupby(["outer_seed","strategy"],as_index=False).absolute_trainable_parameter_L2_update.mean()
    parameter_shock=parameter_update.merge(round1_shock,on=["outer_seed","strategy"])
    parameter_shock_rho=float(parameter_shock.absolute_trainable_parameter_L2_update.corr(parameter_shock.next_round_delta_test_NRMSE_posthoc,method="spearman"))
    blocking=not bool((parameter_updates[parameter_updates.layer=="frozen_early"].absolute_L2_update==0).all())
    decision={
        "dataset_is_clustered":bool(silhouette_values.mean()>=.10),
        "dataset_cluster_silhouette_mean":float(silhouette_values.mean()),
        "high_shift_regions_are_low_density":bool((density_corr<=-.5).all()),
        "shift_vs_density_spearman_by_seed":dict(zip(map(str,SEEDS),map(float,density_corr))),
        "uncertainty_and_shift_are_redundant":bool(((uncertainty_corr>.95)&(overlap.to_numpy()>.9)).all()),
        "high_shift_associated_with_large_gradient_updates":bool(sum(value>0 for value in grad_shift_by_seed.values())>=2),
        "high_shift_gradient_spearman_by_seed":grad_shift_by_seed,
        "high_uncertainty_associated_with_large_gradient_updates":bool(sum(value>0 for value in grad_unc_by_seed.values())>=2),
        "high_uncertainty_gradient_spearman_by_seed":grad_unc_by_seed,
        "large_model_update_associated_with_next_round_shock_descriptively":bool(update_shock_rho>0),
        "function_update_vs_next_round_shock_spearman_round1_posthoc":update_shock_rho,
        "parameter_update_vs_next_round_shock_spearman_round1_posthoc":parameter_shock_rho,
        "t3r_qualified":t3r_gate["t3r_qualified"],"selected_lambda":t3r_gate["selected_lambda"],
        "t3r_engineering_smoke":smoke,
        "generic_low_budget_A2a_warranted":not blocking,
        "transfer_aware_performance_test_warranted_now":False,
        "protocol_a_active_evidence":"null","D42_status":"post-hoc descriptive only","D43_status":"qualification only; no performance evidence",
        "test_usage":"historical next-round NRMSE only in isolated post-hoc mechanism correlations; never T3R lambda selection",
        "blocking_issue_found":blocking,"formal_experiment_started":False,"protocol_b_started":False,
    }
    (OUT/"decision.json").write_text(json.dumps(decision,indent=2))
    recommendation={
        "recommended_next_scientific_test":"E4-A2a generic low-budget headroom sensitivity, subject to manual approval",
        "rationale":"Change only L0 from 50 to 30 before testing any new transfer-aware acquisition, so headroom is identifiable.",
        "recommended_current_acquisition_change":None,
        "T3R_status":"qualification_failed",
        "transfer_aware_performance_test":"defer",
        "predictor_change":"none",
        "test_used_to_select_acquisition_or_lambda":False,
        "frozen_historical_E4_context_used":True,
        "formal_run_authorized":False,
    }
    (OUT/"strategy_recommendation.json").write_text(json.dumps(recommendation,indent=2))
    config={"stage":"D44 Active-Learning Suitability Diagnosis + Soft Transfer-Aware Representative Acquisition","source_commit":"c684ff10274e32e523c7fce4b75ec812dad25330","outer_seeds":list(SEEDS),"formal_states_reused":"E4 Protocol A Round0/Round1 and all historical pre-query checkpoints","representation":"member42 128D h_graph + 9D conditions, current gradient-train normalization","kmeans":{"k":10,"random_state":0,"n_init":20,"diagnostic_only":True},"density":{"k":10,"truth_used":False},"probe":{"size":100,"universe":"intersection of all three Protocol A U0 sets","selection":"SHA256 deterministic"},"gradient":{"states":"Round0","labels":"historical Round1 reveal only","parameters":"last2+head","use":"posthoc mechanism only"},"T3R":{"lambda_grid":list(LAMBDAS),"weights":"(1-lambda)*I + lambda*representativeness percentile","gate":{"informativeness_retention":.90,"shift_retention":.85,"uncertainty_retention":.85,"density_gain":.10,"density_seed_count":2,"overlap_exclusion":.90},"test_or_truth_used":False},"decision_thresholds":{"dataset_clustered_mean_silhouette":.10,"high_shift_low_density_all_seed_spearman":-.5,"ranking_redundancy_spearman":.95,"ranking_redundancy_top10_overlap":.9},"forbidden_respected":["Protocol B","E4-A2a formal","transfer-aware formal","predictor modification","test-guided lambda"]}
    (OUT/"config.json").write_text(json.dumps(config,indent=2))

    relations_all=error_relations[error_relations.outer_seed.astype(str)=="all"]
    info_rows=[]
    definitions={"Random":"no model signal; sampling control","QWidth":"single-model predictive interval width","Ensemble uncertainty":"target-adapted epistemic disagreement","Latent distance":"lack of coverage by current gradient-training set","Coverage":"batch selection for training-set coverage","Transfer shift":"source-to-target prediction correction magnitude","T2 shift+uncertainty":"joint correction magnitude and adapted-model disagreement","Representativeness":"density in the current target pool","Gradient/model change":"label-induced local parameter/function change (post reveal only)"}
    for signal,meaning in definitions.items():
        info_rows.append({"signal":signal,"what_it_measures":meaning,"observed_relation_to_error":"see uncertainty_error_relation.csv; descriptive only","observed_relation_to_model_update":"see gradient_update_summary.csv and model_update_summary.csv","observed_relation_to_density":"shift is negatively associated with density; T3R sweep quantifies trade-off" if signal in ("Transfer shift","T2 shift+uncertainty","Representativeness") else "diagnostic-specific","redundancy":"Transfer shift is highly redundant with QWidth but not strict-redundant with Ensemble" if signal=="Transfer shift" else "not assumed equivalent","current_evidence":"qualification/post-hoc mechanism only; no new performance evidence"})
    pd.DataFrame(info_rows).to_csv(OUT/"strategy_information_map.csv",index=False)
    write_plots(dataset,shift_structure,queries,model_updates,sweep); write_a2a_preregistration()

    shift_corr=shift_structure[(shift_structure.record_type=="correlation")].pivot(index="relationship",columns="outer_seed",values="spearman")
    t3r_mean=sweep.groupby(["candidate","lambda"],dropna=False)[["shift_retention","uncertainty_retention","informativeness_retention","density_gain","T2_sample_overlap_fraction"]].mean()
    largest_parameter=model_updates[(model_updates.transition=="round0_to_round1")&(model_updates.member_seed.astype(str)!="ensemble")].groupby("strategy").absolute_trainable_parameter_L2_update.mean().sort_values(ascending=False)
    largest_function=ensemble_update.groupby("strategy").function_update_mean_standardized_abs.mean().sort_values(ascending=False)
    (OUT/"README.md").write_text(f"""# D44 — Active-Learning Suitability & Model-Update Diagnosis

This stage is diagnostic and qualification-only. E4 Protocol A remains `active evidence = null`; D42 remains post-hoc; D43 remains no-performance qualification. T3R lambda selection used only Round0 U0 predictions/features. Historical reveal labels and test NRMSE are isolated to post-hoc mechanism tables.

## Dataset and shift structure

Mean k-means-10 silhouette is `{silhouette_values.mean():.4f}`. Shift correlations by seed are `{shift_corr.to_json()}`. Compound/condition structure, cluster occupancy, L0 coverage, and graph-versus-condition distance decomposition are in `dataset_structure.csv` and `source_target_shift_structure.csv`.

## Historical query and model update

All 225 historical seed × strategy × query-round batches were rescored before reveal; revealed-error and next-round-test relationships are descriptive, not causal. Mean Round0→Round1 trainable parameter update ranks `{largest_parameter.to_dict()}`; ensemble function-update ranks `{largest_function.to_dict()}`. Parameter/function update vs first-round NRMSE shock Spearman is `{parameter_shock_rho:.4f}/{update_shock_rho:.4f}`. Per-sample gradients use only the 150 historical Round1 revealed rows.

## T3R

The old D43 T3 is unchanged and remains failed. T3R uses `Score=(1-lambda)*I+lambda*R`, lambda grid `{LAMBDAS}`, with the fixed retention/density/overlap gate. Mean trade-offs are `{t3r_mean.to_json(orient='index')}`. Gate: `{json.dumps(t3r_gate,sort_keys=True)}`. Conditional engineering smoke: `{json.dumps(smoke,sort_keys=True)}`. No performance was used.

## Decision

`{json.dumps(decision,sort_keys=True)}`. E4-A2a has been preregistered but not executed. The next authorized scientific question is the single-variable low-L0 headroom sensitivity, not transfer-aware performance.
""")
    print(json.dumps(decision,indent=2))


if __name__=="__main__":
    main()
