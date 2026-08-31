#!/usr/bin/env python3
"""S1 exploratory source-to-target structural shift audit."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qgeognn_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.qgeognn_al.artifacts import finalize_experiment, sha256_file
from src.qgeognn_al.data import (
    CONDITION_FEATURE_NAMES, apply_standardizer, condition_matrix,
    fit_standardizer, load_combined_graph_cache,
)
from src.qgeognn_al.engine import QGeoGNNActiveLearningEngine
from src.qgeognn_al.resources import (
    ROOT, SOURCE_CHECKPOINTS, SOURCE_DATA, SOURCE_GRAPH_CACHE, SOURCE_SCALER,
    TARGET_DATA, TARGET_GRAPH_CACHE, verified_source_checkpoints,
)

STUDY = ROOT / "studies/track_b_transfer/s1_source_target_shift"
LABEL_COLUMNS = {"V1_ml", "V2_ml"}


def make_partition(frame: pd.DataFrame, seed: int, analysis_fraction: float) -> pd.DataFrame:
    """Create a compound split from identity columns only."""
    identities = frame[["sample_id", "canonical_smiles"]].copy()
    compounds = np.array(sorted(identities["canonical_smiles"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    shuffled = compounds[rng.permutation(len(compounds))]
    analysis_count = int(round(len(compounds) * analysis_fraction))
    analysis = set(shuffled[:analysis_count])
    identities["role"] = np.where(identities["canonical_smiles"].isin(analysis), "analysis", "reserved")
    identities["split_seed"] = int(seed)
    return identities


def load_analysis_truth(path: Path, partition: pd.DataFrame) -> pd.DataFrame:
    """Parse label cells only for analysis rows; reserved label cells are skipped."""
    analysis_ids = set(partition.loc[partition.role.eq("analysis"), "sample_id"].astype(str))
    id_only = pd.read_csv(path, usecols=["sample_id"])
    skiprows = [index + 1 for index, sample_id in enumerate(id_only["sample_id"].astype(str)) if sample_id not in analysis_ids]
    truth = pd.read_csv(path, usecols=["sample_id", "V1_ml", "V2_ml"], skiprows=skiprows)
    if set(truth["sample_id"].astype(str)) != analysis_ids:
        raise RuntimeError("analysis truth identity mismatch")
    return truth


def environment_record() -> dict:
    names = ["numpy", "pandas", "scipy", "scikit-learn", "torch", "torch-geometric", "rdkit"]
    versions = {}
    for name in names:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions, "device": "cpu"}


def prepare(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    feature_columns = [column for column in pd.read_csv(TARGET_DATA, nrows=0).columns if column not in LABEL_COLUMNS]
    features = pd.read_csv(TARGET_DATA, usecols=feature_columns)
    partition = make_partition(features, int(config["partition_seed"]), float(config["analysis_compound_fraction"]))
    STUDY.mkdir(parents=True, exist_ok=True)
    partition.to_csv(STUDY / "s1_partition.csv", index=False)
    records = verified_source_checkpoints()
    pd.DataFrame(records).to_csv(STUDY / "source_checkpoint_manifest.csv", index=False)
    config["partition_sha256"] = sha256_file(STUDY / "s1_partition.csv")
    config["source_checkpoint_hashes"] = {str(row["source_seed"]): row["sha256"] for row in records}
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    (STUDY / "environment.json").write_text(json.dumps(environment_record(), indent=2) + "\n")


def safe_ratio(target: np.ndarray, source: np.ndarray, floor: float) -> np.ndarray:
    result = np.full(len(source), np.nan)
    valid = np.abs(source) >= floor
    result[valid] = target[valid] / source[valid]
    return result


def variance_decomposition(values: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    frame = pd.DataFrame({"value": values, "group": groups})
    overall = float(frame.value.mean())
    grouped = frame.groupby("group").value
    counts, means = grouped.size(), grouped.mean()
    between = float(np.sum(counts * np.square(means - overall)) / len(frame))
    within = float(sum(np.square(group.value - group.value.mean()).sum() for _, group in frame.groupby("group")) / len(frame))
    total = float(np.var(values, ddof=0))
    return {"between_compound_variance": between, "within_compound_variance": within, "total_variance": total, "between_total_ratio": between / total if total else float("nan")}


def combined_nrmse(truth: np.ndarray, prediction: np.ndarray, scales: np.ndarray) -> float:
    rmse = np.sqrt(np.mean(np.square(truth - prediction), axis=0))
    return float(np.mean(rmse / scales))


def correction_cv(frame: pd.DataFrame, condition_features: np.ndarray, folds: int, inner_folds: int, alphas: list[float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth = frame[["V1_ml", "V2_ml"]].to_numpy(float)
    source = frame[["source_V1_mean", "source_V2_mean"]].to_numpy(float)
    groups = frame["canonical_smiles"].astype(str).to_numpy()
    scales = np.maximum(truth.std(axis=0, ddof=0), 1e-8)
    outer_rows, predictions = [], []
    for fold, (train, valid) in enumerate(GroupKFold(n_splits=folds).split(truth, groups=groups)):
        train_groups = groups[train]
        candidates = []
        for alpha in alphas:
            scores = []
            for inner_train, inner_valid in GroupKFold(n_splits=inner_folds).split(train, groups=train_groups):
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                x_train = np.column_stack([source[train][inner_train], condition_features[train][inner_train]])
                x_valid = np.column_stack([source[train][inner_valid], condition_features[train][inner_valid]])
                model.fit(x_train, truth[train][inner_train] - source[train][inner_train])
                pred = source[train][inner_valid] + model.predict(x_valid)
                scores.append(combined_nrmse(truth[train][inner_valid], pred, scales))
            candidates.append((float(np.mean(scores)), alpha))
        alpha = min(candidates)[1]
        offset = (truth[train] - source[train]).mean(axis=0)
        affine = [LinearRegression().fit(source[np.ix_(train, [target])], truth[train, target]) for target in range(2)]
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        ridge.fit(np.column_stack([source[train], condition_features[train]]), truth[train] - source[train])
        model_predictions = {
            "S0_zero_shot": source[valid],
            "S1_global_offset": source[valid] + offset,
            "S2_affine": np.column_stack([affine[target].predict(source[np.ix_(valid, [target])]) for target in range(2)]),
            "S3_condition_ridge": source[valid] + ridge.predict(np.column_stack([source[valid], condition_features[valid]])),
        }
        for name, pred in model_predictions.items():
            metric = combined_nrmse(truth[valid], pred, scales)
            residual = truth[valid] - pred
            outer_rows.append({"fold": fold, "model": name, "selected_alpha": alpha if name == "S3_condition_ridge" else np.nan, "V1_RMSE": float(np.sqrt(np.mean(residual[:, 0] ** 2))), "V2_RMSE": float(np.sqrt(np.mean(residual[:, 1] ** 2))), "combined_NRMSE": metric, "validation_compounds": len(set(groups[valid])), "train_validation_compound_overlap": len(set(groups[train]) & set(groups[valid]))})
            predictions.extend({"sample_id": frame.iloc[row].sample_id, "fold": fold, "model": name, "V1_prediction": pred[pos, 0], "V2_prediction": pred[pos, 1]} for pos, row in enumerate(valid))
    cv = pd.DataFrame(outer_rows)
    baseline = cv.loc[cv.model.eq("S0_zero_shot"), "combined_NRMSE"].mean()
    summary = cv.groupby("model", as_index=False).agg(folds=("fold", "nunique"), V1_RMSE_mean=("V1_RMSE", "mean"), V2_RMSE_mean=("V2_RMSE", "mean"), combined_NRMSE_mean=("combined_NRMSE", "mean"), combined_NRMSE_std=("combined_NRMSE", "std"))
    summary["improvement_vs_zero_shot"] = baseline - summary["combined_NRMSE_mean"]
    return cv, pd.DataFrame(predictions), summary


def run(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    if sha256_file(STUDY / "s1_partition.csv") != config["partition_sha256"]:
        raise RuntimeError("partition changed after preregistration")
    verified_source_checkpoints()
    feature_columns = [column for column in pd.read_csv(TARGET_DATA, nrows=0).columns if column not in LABEL_COLUMNS]
    target = pd.read_csv(TARGET_DATA, usecols=feature_columns)
    source = pd.read_csv(SOURCE_DATA)
    partition = pd.read_csv(STUDY / "s1_partition.csv")
    truth = load_analysis_truth(TARGET_DATA, partition)
    runtime = STUDY / "runtime/generated"
    if runtime.exists(): shutil.rmtree(runtime)
    runtime.mkdir(parents=True)
    cache = load_combined_graph_cache(SOURCE_GRAPH_CACHE, TARGET_GRAPH_CACHE)
    scaler = json.loads(SOURCE_SCALER.read_text())
    engine = QGeoGNNActiveLearningEngine(target.assign(V1_ml=0.0, V2_ml=0.0), cache, scaler, SOURCE_CHECKPOINTS[42], device=torch.device("cpu"))
    member_tables = []
    for seed, checkpoint in SOURCE_CHECKPOINTS.items():
        table = engine.predict(target.sample_id.astype(str).tolist(), checkpoint, return_embedding=False).table
        member_tables.append(table[["sample_id", "V1_q50", "V2_q50"]].rename(columns={"V1_q50": f"source_{seed}_V1", "V2_q50": f"source_{seed}_V2"}))
    predictions = member_tables[0]
    for table in member_tables[1:]: predictions = predictions.merge(table, on="sample_id", validate="one_to_one")
    for target_name in ("V1", "V2"):
        columns = [f"source_{seed}_{target_name}" for seed in SOURCE_CHECKPOINTS]
        predictions[f"source_{target_name}_mean"] = predictions[columns].mean(axis=1)
        predictions[f"source_{target_name}_std"] = predictions[columns].std(axis=1, ddof=1)
    predictions.to_csv(runtime / "zero_shot_predictions.csv", index=False)
    analysis = target.merge(partition.loc[partition.role.eq("analysis"), ["sample_id", "role"]], on="sample_id").merge(predictions, on="sample_id").merge(truth, on="sample_id", validate="one_to_one")
    source_conditions = condition_matrix(source)
    target_conditions = condition_matrix(target)
    mean, scale = fit_standardizer(source_conditions)
    source_z, target_z = apply_standardizer(source_conditions, mean, scale), apply_standardizer(target_conditions, mean, scale)
    global_distance = np.sqrt(((target_z[:, None, :] - source_z[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    source_by_compound = {compound: np.flatnonzero(source.canonical_smiles.astype(str).eq(compound)) for compound in source.canonical_smiles.astype(str).unique()}
    same_distance = []
    source_counts = []
    for idx, compound in enumerate(target.canonical_smiles.astype(str)):
        positions = source_by_compound.get(compound, np.array([], dtype=int)); source_counts.append(len(positions))
        same_distance.append(float(np.sqrt(((source_z[positions] - target_z[idx]) ** 2).sum(axis=1)).min()) if len(positions) else np.nan)
    distance_frame = pd.DataFrame({"sample_id": target.sample_id, "same_compound_condition_distance": same_distance, "global_condition_distance": global_distance})
    analysis = analysis.merge(distance_frame, on="sample_id")
    analysis_positions = target.reset_index().set_index("sample_id").loc[analysis.sample_id, "index"].to_numpy()
    analysis_conditions = target_z[analysis_positions]
    for target_name in ("V1", "V2"):
        analysis[f"residual_{target_name}"] = analysis[f"{target_name}_ml"] - analysis[f"source_{target_name}_mean"]
        analysis[f"ratio_{target_name}"] = safe_ratio(analysis[f"{target_name}_ml"].to_numpy(), analysis[f"source_{target_name}_mean"].to_numpy(), float(config["ratio_source_floor_ml"]))
    overlap = pd.DataFrame({"canonical_smiles": target.canonical_smiles, "sample_id": target.sample_id, "source_row_count": source_counts}).sort_values(["canonical_smiles", "sample_id"])
    overlap.to_csv(runtime / "compound_overlap.csv", index=False)
    unique_target = target.canonical_smiles.nunique(); overlapping = sum(compound in source_by_compound for compound in target.canonical_smiles.unique())
    (runtime / "overlap_summary.json").write_text(json.dumps({"target_rows": len(target), "target_unique_compounds": int(unique_target), "overlapping_unique_compounds": int(overlapping), "target_only_unique_compounds": int(unique_target - overlapping), "analysis_rows": len(analysis), "reserved_rows": int((partition.role == "reserved").sum()), "standardization_reference": "4g source only"}, indent=2) + "\n")
    condition_rows = []
    for index, name in enumerate(CONDITION_FEATURE_NAMES):
        condition_rows.append({"feature": name, "source_mean": float(source_conditions[:, index].mean()), "source_std": float(source_conditions[:, index].std()), "target_mean": float(target_conditions[:, index].mean()), "target_std": float(target_conditions[:, index].std()), "standardized_mean_shift": float(target_z[:, index].mean()), "standardized_std_ratio": float(target_z[:, index].std())})
    pd.DataFrame(condition_rows).to_csv(runtime / "condition_shift.csv", index=False)
    summary_rows = []
    for target_name in ("V1", "V2"):
        residual = analysis[f"residual_{target_name}"].to_numpy()
        ratio = analysis[f"ratio_{target_name}"].dropna().to_numpy()
        summary_rows.append({"target": target_name, "count": len(residual), "mean": residual.mean(), "median": np.median(residual), "std": residual.std(), "IQR": np.quantile(residual, .75)-np.quantile(residual, .25), "P10": np.quantile(residual, .1), "P90": np.quantile(residual, .9), "MAE": np.abs(residual).mean(), "RMSE": np.sqrt(np.mean(residual**2)), "ratio_valid_count": len(ratio), "ratio_median": np.median(ratio)})
    pd.DataFrame(summary_rows).to_csv(runtime / "residual_summary.csv", index=False)
    zero_rows = []
    for seed in [*SOURCE_CHECKPOINTS, "mean"]:
        pred_cols = [f"source_{seed}_V1", f"source_{seed}_V2"] if seed != "mean" else ["source_V1_mean", "source_V2_mean"]
        error = analysis[["V1_ml", "V2_ml"]].to_numpy() - analysis[pred_cols].to_numpy()
        zero_rows.append({"member": seed, "V1_RMSE": np.sqrt(np.mean(error[:,0]**2)), "V2_RMSE": np.sqrt(np.mean(error[:,1]**2)), "V1_MAE": np.abs(error[:,0]).mean(), "V2_MAE": np.abs(error[:,1]).mean()})
    pd.DataFrame(zero_rows).to_csv(runtime / "zero_shot_summary.csv", index=False)
    association_features = {"source_prediction": None, "PE_EA_log_ratio": np.log((analysis["PE/EA"].str.split('/').str[0].astype(float)+1e-6)/(analysis["PE/EA"].str.split('/').str[1].astype(float)+1e-6)), "flow": analysis["Flow mL/min"], "loading_solvent": analysis["loading solvent"].map({"PE":0,"EA":1,"DCM":2}), "loading_amount": analysis["Density g/ml"]*analysis["V/ul"], "loading_solvent_volume": analysis["Volume of loading solvent/ul"], "same_compound_condition_distance": analysis["same_compound_condition_distance"], "global_condition_distance": analysis["global_condition_distance"]}
    association_rows = []
    for target_name in ("V1", "V2"):
        features = dict(association_features); features["source_prediction"] = analysis[f"source_{target_name}_mean"]; features["ensemble_uncertainty"] = analysis[f"source_{target_name}_std"]
        y = analysis[f"residual_{target_name}"]
        for name, values in features.items():
            valid = pd.notna(values) & pd.notna(y); x = np.asarray(values)[valid]; yy = y.to_numpy()[valid]
            association_rows.append({"target": target_name, "feature": name, "n": len(x), "pearson": pearsonr(x, yy).statistic if len(x)>2 and np.std(x)>0 else np.nan, "spearman": spearmanr(x, yy).statistic if len(x)>2 and np.std(x)>0 else np.nan, "interpretation": "descriptive_association_only"})
    pd.DataFrame(association_rows).to_csv(runtime / "residual_feature_associations.csv", index=False)
    compound_summary = analysis.groupby("canonical_smiles").agg(count=("sample_id","size"), V1_residual_mean=("residual_V1","mean"), V1_residual_std=("residual_V1","std"), V2_residual_mean=("residual_V2","mean"), V2_residual_std=("residual_V2","std")).reset_index()
    compound_summary.to_csv(runtime / "compound_residual_summary.csv", index=False)
    decomposition = {target_name: variance_decomposition(analysis[f"residual_{target_name}"].to_numpy(), analysis.canonical_smiles.to_numpy()) for target_name in ("V1", "V2")}
    decomposition["residual_V1_V2_pearson"] = float(pearsonr(analysis.residual_V1, analysis.residual_V2).statistic); decomposition["residual_V1_V2_spearman"] = float(spearmanr(analysis.residual_V1, analysis.residual_V2).statistic); decomposition["meaning"] = "compound-level residual clustering, not training reliability"
    (runtime / "residual_variance_decomposition.json").write_text(json.dumps(decomposition, indent=2)+"\n")
    cv, cv_predictions, cv_summary = correction_cv(analysis, analysis_conditions, int(config["outer_group_folds"]), int(config["inner_group_folds"]), config["ridge_alpha_grid"])
    cv.to_csv(runtime / "simple_correction_cv.csv", index=False); cv_summary.to_csv(runtime / "simple_correction_summary.csv", index=False); cv_predictions.to_csv(runtime / "simple_correction_predictions.csv", index=False)
    plot_dir = runtime / "plots"; plot_dir.mkdir()
    fig, axes = plt.subplots(1,2,figsize=(9,4));
    for i,t in enumerate(("V1","V2")): axes[i].scatter(analysis[f"source_{t}_mean"], analysis[f"{t}_ml"], s=10, alpha=.5); axes[i].set(xlabel=f"source {t}",ylabel=f"target {t}")
    fig.tight_layout(); fig.savefig(plot_dir/"zero_shot_vs_target.png",dpi=160); plt.close(fig)
    for t in ("V1","V2"):
        fig,ax=plt.subplots(figsize=(5,3.5)); ax.hist(analysis[f"residual_{t}"],bins=30); ax.set(xlabel=f"{t} residual",ylabel="rows"); fig.tight_layout(); fig.savefig(plot_dir/f"residual_distribution_{t.lower()}.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,3.5)); ax.scatter(analysis.global_condition_distance, np.sqrt(analysis.residual_V1**2+analysis.residual_V2**2),s=10,alpha=.5); ax.set(xlabel="global condition distance",ylabel="residual norm"); fig.tight_layout(); fig.savefig(plot_dir/"residual_vs_condition_distance.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3.5)); compound_summary.plot.scatter(x="V1_residual_std",y="V2_residual_std",s=compound_summary["count"]*3,ax=ax); fig.tight_layout(); fig.savefig(plot_dir/"compound_residual_variation.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3.5)); cv.boxplot(column="combined_NRMSE",by="model",ax=ax,rot=15); fig.suptitle(""); fig.tight_layout(); fig.savefig(plot_dir/"simple_correction_group_cv.png",dpi=160); plt.close(fig)
    best = cv_summary.sort_values("combined_NRMSE_mean").iloc[0]
    decision = {"study":"S1","role":"exploratory_shift_audit","confirmatory":False,"target_truth_used":True,"reserved_truth_used":False,"new_model_selected":False,"T1_started":False,"active_learning_started":False,"best_descriptive_simple_model":best["model"],"manual_recommendation_required":True}
    (runtime/"decision.json").write_text(json.dumps(decision,indent=2)+"\n"); shutil.copy2(config_path,runtime/"config.json"); shutil.copy2(STUDY/"environment.json",runtime/"environment.json")
    (runtime/"README.md").write_text("# S1 — Source-to-Target Structural Shift Audit\n\nExploratory, hypothesis-generating audit using frozen source inference. All residual and correction analyses use only the compound-held-out partition's `analysis` role; reserved truth remains unconsumed. GroupKFold is by compound. Results are descriptive and require manual T1 design review.\n")
    keep = ["README.md","config.json","environment.json","decision.json","overlap_summary.json","compound_overlap.csv","condition_shift.csv","zero_shot_summary.csv","residual_summary.csv","residual_feature_associations.csv","compound_residual_summary.csv","residual_variance_decomposition.json","simple_correction_cv.csv","simple_correction_summary.csv",*[f"plots/{p.name}" for p in plot_dir.iterdir()]]
    finalize_experiment(runtime, STUDY, keep)
    shutil.rmtree(STUDY/"runtime")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--prepare",action="store_true"); parser.add_argument("--run",action="store_true"); parser.add_argument("--config",type=Path,default=STUDY/"config.json"); args=parser.parse_args()
    if args.prepare == args.run: parser.error("choose exactly one of --prepare/--run")
    prepare(args.config) if args.prepare else run(args.config)


if __name__ == "__main__": main()
