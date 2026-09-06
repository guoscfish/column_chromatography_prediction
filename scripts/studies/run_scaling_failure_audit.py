#!/usr/bin/env python3
"""Training-only direction audit, followed by separately authorized descriptive test tables."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, Lipinski

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.qgeognn_al.artifacts import sha256_file
from src.qgeognn_al.resources import SOURCE_DATA
from src.qgeognn_al.transfer.calibration import fit_affine, fit_scale_only
from src.qgeognn_al.transfer.scaling_audit import (
    BIN_NAMES, CONDITIONS, DESCRIPTORS, correlation, match_key, neighborhood_consistency,
    out_of_fold_calibration, partial_rank, safe_ratio, source_bins, standardize_condition_contrast,
)

STUDY = ROOT/"studies/transfer/scaling_failure_audit"
OLD = ROOT/"studies/transfer/cross_column"
PREVIOUS = ROOT/"studies/transfer/residual_diagnostics"
COLUMNS = ("8g", "25g", "40g")
SCALES = json.loads((PREVIOUS/"all_predictions_frozen.json").read_text())["source_scales"]
FEATURES = ["sample_id", "canonical_smiles", "PE/EA", "loading solvent", "Density g/ml", "V/ul",
            "Volume of loading solvent/ul", "Flow mL/min"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def selected_truth(path, ids):
    ids = list(ids)
    identities = pd.read_csv(path, usecols=["sample_id"])
    allowed = set(ids)
    skip = [i+1 for i, sid in enumerate(identities.sample_id) if sid not in allowed]
    frame = pd.read_csv(path, usecols=["sample_id", "V1_ml", "V2_ml"], skiprows=skip).set_index("sample_id")
    if set(frame.index) != allowed:
        raise RuntimeError("label identity mismatch")
    return frame.loc[ids, ["V1_ml", "V2_ml"]].to_numpy(float)


def target_path(column):
    return OLD/"data_audit"/f"canonical_{column}.csv"


def lock_protocol():
    previous = json.loads((PREVIOUS/"protocol.json").read_text())
    for name, digest in previous["protected_hashes"].items():
        if sha256_file(ROOT/name) != digest:
            raise RuntimeError(f"baseline drift: {name}")
    files = [Path(__file__), ROOT/"src/qgeognn_al/transfer/scaling_audit.py", STUDY/"PREREGISTRATION.md",
             ROOT/"docs/research/CROSS_COLUMN_TRANSFER_STATUS.md", SOURCE_DATA,
             ROOT/"studies/predictor/final_4g_qualification/splits/row_seed_42.csv"]
    config = {"base_commit": "6e10498", "inputs": previous["protected_hashes"],
              "audit_files": {str(p.relative_to(ROOT)): sha256_file(p) for p in files},
              "ratio_floor_ml": .5, "rdkit": rdBase.rdkitVersion,
              "direction_evidence": "budget100 gradient_train only; compound out-of-fold residuals",
              "test_selection": False, "predictor_retraining": False}
    path = STUDY/"protocol.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("audit contract changed")
    write_json(path, config)
    return config


def prepare_features():
    lock_protocol()
    source = pd.read_csv(SOURCE_DATA, usecols=FEATURES)
    split = pd.read_csv(ROOT/"studies/predictor/final_4g_qualification/splits/row_seed_42.csv")
    train_ids = set(split.loc[split.split.eq("train"), "sample_id"])
    # Source-train labels are existing baseline resources, never source-validation/test labels.
    source_train_ids = sorted(train_ids & set(source.sample_id))
    source_y = pd.DataFrame(selected_truth(SOURCE_DATA, source_train_ids), index=source_train_ids, columns=["V1", "V2"])
    maps = {mode: defaultdict(list) for mode in ("exact", "relaxed")}
    for row in source.to_dict("records"):
        for mode in maps:
            maps[mode][match_key(row, relaxed=mode == "relaxed")].append(row["sample_id"])
    source_compounds = set(source.canonical_smiles)
    train_compounds = set(source.loc[source.sample_id.isin(train_ids), "canonical_smiles"])
    frames, pair_rows, descriptor_rows = {}, [], []
    for column in COLUMNS:
        data = pd.read_csv(target_path(column), usecols=FEATURES)
        cache = PREVIOUS/"runtime"/f"source_{column}.csv"
        if not cache.exists():
            from scripts.studies.run_next_transfer_diagnostics import contract, source_predictions
            source_predictions(contract())
        meta = json.loads(cache.with_suffix(".json").read_text())
        if sha256_file(cache) != meta["sha256"]:
            raise RuntimeError("source prediction cache changed")
        predicted = pd.read_csv(cache, usecols=["sample_id", "source_V1", "source_V2"])
        data = data.merge(predicted, on="sample_id", validate="one_to_one")
        data["EA_fraction"] = data["PE/EA"].map(lambda s: float(s.split("/")[1])/sum(map(float, s.split("/"))))
        data["solvent_DCM"] = data["loading solvent"].eq("DCM").astype(float)
        data["loading_ul"] = data["V/ul"]
        data["amount_density_ul"] = data["Density g/ml"]*data["V/ul"]
        data["loading_volume_ul"] = data["Volume of loading solvent/ul"]
        details = []
        for row in data.to_dict("records"):
            item = {"sample_id": row["sample_id"], "column": column}
            for mode in maps:
                matches = maps[mode].get(match_key(row, relaxed=mode == "relaxed"), [])
                eligible = [sid for sid in matches if sid in train_ids]
                item[f"{mode}_source_ids"] = json.dumps(matches)
                item[f"{mode}_source_train_ids"] = json.dumps(eligible)
                item[f"{mode}_matches"] = len(matches)
                item[f"{mode}_train_matches"] = len(eligible)
                for target in ("V1", "V2"):
                    values = source_y.loc[eligible, target]
                    item[f"{mode}_source_train_{target}"] = values.mean() if eligible else np.nan
                    item[f"{mode}_source_train_{target}_std"] = values.std(ddof=1) if len(values)>1 else np.nan
            item["pair_status"] = ("exact_paired" if item["exact_matches"] else
                                   "same_compound_condition_different" if row["canonical_smiles"] in source_compounds else "source_absent")
            item["relaxed_pair_status"] = ("relaxed_paired" if item["relaxed_matches"] else
                                           "same_compound_condition_different" if row["canonical_smiles"] in source_compounds else "source_absent")
            item["source_train_compound_seen"] = row["canonical_smiles"] in train_compounds
            details.append(item)
        detail = pd.DataFrame(details)
        pair_rows.extend(details)
        frames[column] = data.merge(detail.drop(columns="column"), on="sample_id", validate="one_to_one")
    molecules = sorted(set(pd.concat(frames.values()).canonical_smiles))
    for smiles in molecules:
        mol = Chem.MolFromSmiles(smiles)
        values = (Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol), Lipinski.NumHDonors(mol),
                  Lipinski.NumHAcceptors(mol), Lipinski.NumRotatableBonds(mol), Lipinski.RingCount(mol))
        descriptor_rows.append({"canonical_smiles": smiles, **dict(zip(DESCRIPTORS, values))})
    descriptors = pd.DataFrame(descriptor_rows)
    descriptors.to_csv(STUDY/"molecular_descriptors.csv", index=False)
    for column in COLUMNS:
        frames[column] = frames[column].merge(descriptors, on="canonical_smiles", validate="many_to_one").set_index("sample_id")
        frames[column].to_csv(STUDY/f"features_{column}.csv")
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(STUDY/"pair_identity_audit.csv", index=False)
    summary = []
    for column, group in pairs.groupby("column"):
        summary.append({"column": column, "rows": len(group), "exact_rows": int(group.exact_matches.gt(0).sum()),
                        "relaxed_rows": int(group.relaxed_matches.gt(0).sum()),
                        "exact_source_train_rows": int(group.exact_train_matches.gt(0).sum()),
                        "relaxed_source_train_rows": int(group.relaxed_train_matches.gt(0).sum()),
                        "ambiguous_exact_source_rows": int(group.exact_matches.gt(1).sum()),
                        "source_absent_rows": int(group.pair_status.eq("source_absent").sum())})
    pd.DataFrame(summary).to_csv(STUDY/"pair_coverage.csv", index=False)
    return frames


def detailed_rows(frame, truth, scale_prediction, affine_prediction, train_source, fold_ids=None):
    rows = []
    for j, target in enumerate(("V1", "V2")):
        part = frame.reset_index().copy()
        part["target"] = target
        part["truth"] = truth[:, j]
        part["source"] = part[f"source_{target}"]
        part["ratio"] = safe_ratio(truth[:, j], part.source.to_numpy())
        part["scale_prediction"] = scale_prediction[:, j]
        part["affine_prediction"] = affine_prediction[:, j]
        part["scale_residual"] = truth[:, j]-scale_prediction[:, j]
        part["affine_residual"] = truth[:, j]-affine_prediction[:, j]
        part["source_bin"], _ = source_bins(part.source.to_numpy(), train_source[:, j])
        part["source_error_anchor"] = part[f"exact_source_train_{target}"]-part.source
        if fold_ids is not None:
            part["oof_fold"] = fold_ids
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def slice_stats(frame):
    ratio = frame.ratio.dropna()
    record = {"rows": len(frame), "compounds": frame.canonical_smiles.nunique(), "ratio_defined_rows": len(ratio),
              "ratio_median": ratio.median(), "ratio_q25": ratio.quantile(.25), "ratio_q75": ratio.quantile(.75),
              "ratio_mean": ratio.mean(), "ratio_std": ratio.std(ddof=1)}
    for method in ("scale", "affine"):
        error = frame[f"{method}_residual"].to_numpy()
        total = np.square(frame.truth-frame.truth.mean()).sum()
        record.update({f"{method}_rmse": np.sqrt(np.square(error).mean()), f"{method}_mae": np.abs(error).mean(),
                       f"{method}_signed_mean": error.mean(), f"{method}_signed_median": np.median(error),
                       f"{method}_r2": 1-np.square(error).sum()/total if total>0 else np.nan,
                       f"{method}_nrmse": np.sqrt(np.square(error).mean())/SCALES[frame.target.iloc[0]],
                       f"{method}_sse": np.square(error).sum()})
    return record


def diagnostics(detail, keys):
    associations, molecules, neighbors = [], [], []
    for target, frame in detail.groupby("target"):
        base = {**keys, "target": target}
        for feature in ("source", *CONDITIONS):
            adjusted = partial_rank(frame[feature], frame.ratio, frame.source) if feature != "source" else np.nan
            contrast = standardize_condition_contrast(frame, feature) if feature != "source" else {}
            associations.append({**base, "feature": feature, "ratio_spearman": correlation(frame[feature], frame.ratio),
                "scale_residual_spearman": correlation(frame[feature], frame.scale_residual),
                "ratio_partial_source_rho": adjusted,
                "ratio_partial_source_compound_rho": partial_rank(frame[feature], frame.ratio, frame.source, frame.canonical_smiles)
                if feature != "source" else np.nan, **contrast})
        halves = []
        for smiles, group in frame.groupby("canonical_smiles"):
            residual = group.scale_residual
            info = {**base, "canonical_smiles": smiles, "rows": len(group), "conditions": group.EA_fraction.nunique(),
                    "ratio_mean": group.ratio.mean(), "ratio_median": group.ratio.median(), "ratio_std": group.ratio.std(ddof=1),
                    "residual_mean": residual.mean(), "residual_variance": residual.var(ddof=1),
                    "sign_consistency": max((residual>0).mean(), (residual<0).mean()),
                    **{name: group[name].iloc[0] for name in DESCRIPTORS}}
            molecules.append(info)
            if len(group)>=4 and group.EA_fraction.nunique()>=3:
                lower = group.loc[group.EA_fraction<=group.EA_fraction.median(), "ratio"]
                upper = group.loc[group.EA_fraction>group.EA_fraction.median(), "ratio"]
                if lower.notna().sum()>=2 and upper.notna().sum()>=2:
                    halves.append((lower.mean(), upper.mean()))
        molecule_frame = pd.DataFrame([m for m in molecules if m["target"] == target])
        neighbor = neighborhood_consistency(molecule_frame, int(keys["seed"]))
        consistency = correlation(*np.asarray(halves).T) if len(halves)>=5 else np.nan
        neighbor.update({**base, "condition_halves_rho": consistency, "condition_halves_compounds": len(halves),
                         "source_train_exact_coverage": frame.exact_train_matches.gt(0).mean(),
                         "source_error_partial_rho": partial_rank(frame.source_error_anchor, frame.scale_residual, frame.source)})
        neighbors.append(neighbor)
        for feature in DESCRIPTORS:
            associations.append({**base, "feature": feature, "ratio_spearman": correlation(molecule_frame[feature], molecule_frame.ratio_mean),
                                  "scale_residual_spearman": correlation(molecule_frame[feature], molecule_frame.residual_mean)})
    return associations, molecules, neighbors


def write_slices(detail, prefix):
    keys = ["column", "protocol", "seed", "target"]
    rows, interactions, tail = [], [], []
    for key, frame in detail.groupby(keys):
        base = dict(zip(keys, key))
        total = np.square(frame.scale_residual).sum()
        for dimension in ("source_bin", "PE/EA", "loading solvent", "loading_ul", "amount_density_ul",
                          "loading_volume_ul", "pair_status", "relaxed_pair_status"):
            for level, group in frame.groupby(dimension):
                rows.append({**base, "dimension": dimension, "level": str(level), **slice_stats(group)})
        for feature in CONDITIONS:
            work = frame.copy()
            work["condition_level"] = (work[feature]>.5).astype(int) if feature == "solvent_DCM" else (work[feature]>work[feature].median()).astype(int)
            for (source_bin, condition_level), group in work.groupby(["source_bin", "condition_level"]):
                interactions.append({**base, "feature": feature, "source_bin": source_bin,
                                     "condition_level": condition_level, **slice_stats(group)})
        extreme = frame.loc[frame.source_bin.eq("extreme_tail")]
        rest = frame.loc[~frame.source_bin.eq("extreme_tail")]
        tail.append({**base, "rows": len(frame), "tail_rows": len(extreme),
                     "tail_row_fraction": len(extreme)/len(frame),
                     "tail_sse_fraction": np.square(extreme.scale_residual).sum()/total if total>0 else np.nan,
                     "tail_rmse": np.sqrt(np.square(extreme.scale_residual).mean()),
                     "rest_rmse": np.sqrt(np.square(rest.scale_residual).mean()),
                     "tail_signed_mean": extreme.scale_residual.mean(), "rest_signed_mean": rest.scale_residual.mean()})
    pd.DataFrame(rows).to_csv(STUDY/f"{prefix}_slices.csv", index=False)
    pd.DataFrame(interactions).to_csv(STUDY/f"{prefix}_source_condition_interactions.csv", index=False)
    pd.DataFrame(tail).to_csv(STUDY/f"{prefix}_tail.csv", index=False)


def screen_directions(associations, neighbors):
    evidence = []
    for (column, protocol, target, feature), group in associations.loc[associations.feature.isin(CONDITIONS)].groupby(
            ["column", "protocol", "target", "feature"]):
        for sign in (1, -1):
            ranks = group.ratio_partial_source_rho*sign >= .3
            contrasts = group.relative_standardized_contrast*sign >= .15
            evidence.append({"direction": "CONDITIONAL_SCALING", "column": column, "protocol": protocol, "target": target,
                             "feature": feature, "sign": sign, "rank_seeds": int(ranks.sum()), "contrast_seeds": int(contrasts.sum()),
                             "passes": bool(ranks.sum()>=4 and contrasts.sum()>=4)})
    for (column, protocol, target), group in neighbors.groupby(["column", "protocol", "target"]):
        geometry = (group.neighbor_rho>=.3) & (group.permutation_p<=.1)
        halves = group.condition_halves_rho>=.3
        evidence.append({"direction": "MOLECULE_DEPENDENT_SCALING", "column": column, "protocol": protocol,
                         "target": target, "feature": "descriptor_neighborhood", "sign": 1,
                         "rank_seeds": int(geometry.sum()), "contrast_seeds": int(halves.sum()),
                         "passes": bool(geometry.sum()>=4 and halves.sum()>=4)})
        if column == "8g":
            signal = (group.source_train_exact_coverage>=.5) & (group.source_error_partial_rho>=.3)
            evidence.append({"direction": "PAIRED_DELTA_LEARNING", "column": column, "protocol": protocol,
                             "target": target, "feature": "source_error_anchor", "sign": 1,
                             "rank_seeds": int(signal.sum()), "contrast_seeds": int(signal.sum()), "passes": bool(signal.sum()>=4)})
    evidence = pd.DataFrame(evidence)
    evidence.to_csv(STUDY/"training_direction_evidence.csv", index=False)
    qualified = []
    for (direction, feature, target, sign), group in evidence.loc[evidence.passes].groupby(["direction", "feature", "target", "sign"]):
        both = group.groupby("column").protocol.nunique().eq(2).any()
        enough = both if direction == "PAIRED_DELTA_LEARNING" else (both or group.column.nunique()>=2)
        if enough:
            qualified.append({"direction": direction, "feature": feature, "target": target, "sign": int(sign),
                              "supporting_contexts": len(group), "columns": sorted(group.column.unique())})
    order = {"CONDITIONAL_SCALING": 0, "MOLECULE_DEPENDENT_SCALING": 1, "PAIRED_DELTA_LEARNING": 2}
    qualified.sort(key=lambda r: (-r["supporting_contexts"], order[r["direction"]], r["feature"], r["target"]))
    selected = list(dict.fromkeys(r["direction"] for r in qualified))[:2]
    result = {"selected_directions": selected, "qualifying_evidence": qualified,
              "decision": "STRUCTURED_SCALING_FAILURE_IDENTIFIED" if selected else "NO_STRUCTURED_SCALING_FAILURE_IDENTIFIED",
              "model_direction_uses_target_test_labels": False,
              "scope": "predeclared training-only screen; failure to pass is not proof of no structure"}
    write_json(STUDY/"training_direction_decision.json", result)
    return result


def train_audit():
    frames = prepare_features()
    schedule = pd.read_csv(OLD/"splits/schedule_manifest.csv")
    details, associations, molecules, neighbors, scales = [], [], [], [], []
    for (column, protocol, seed, budget), context in schedule.groupby(["column", "protocol", "outer_seed", "planned_budget"]):
        ids = sorted(context.loc[context.role.eq("gradient_train"), "sample_id"])
        frame = frames[column].loc[ids]
        source = frame[["source_V1", "source_V2"]].to_numpy(float)
        truth = selected_truth(target_path(column), ids)
        fit = fit_scale_only(truth, source, source)
        keys = {"column": column, "protocol": protocol, "seed": int(seed), "budget": int(budget)}
        for j, target in enumerate(("V1", "V2")):
            estimates = []
            for compound in frame.canonical_smiles.unique():
                keep = frame.canonical_smiles.ne(compound).to_numpy()
                estimates.append(float(fit_scale_only(truth[keep], source[keep], source).coefficients[j, 0]))
            upper = source[:, j]>=np.quantile(source[:, j], .9)
            truncated = fit_scale_only(truth[~upper], source[~upper], source).coefficients[j, 0]
            scales.append({**keys, "target": target, "scale": fit.coefficients[j, 0], "train_rows": len(ids),
                "compounds": frame.canonical_smiles.nunique(), "q50_min": source[:, j].min(), "q50_max": source[:, j].max(),
                "q50_q90": np.quantile(source[:, j], .9), "upper10_x2_leverage": np.square(source[upper, j]).sum()/np.square(source[:, j]).sum(),
                "without_upper10_scale": truncated, "without_upper10_relative_change": truncated/fit.coefficients[j, 0]-1,
                "leave_compound_min_scale": min(estimates), "leave_compound_max_scale": max(estimates),
                "leave_compound_max_relative_change": max(abs(np.array(estimates)/fit.coefficients[j, 0]-1))})
        if budget != 100:
            continue
        scale_prediction, affine_prediction, folds = out_of_fold_calibration(source, truth, frame.canonical_smiles.to_numpy())
        detail = detailed_rows(frame, truth, scale_prediction, affine_prediction, source, folds)
        for key, value in keys.items():
            detail[key] = value
        details.append(detail)
        a, m, n = diagnostics(detail, keys)
        associations.extend(a); molecules.extend(m); neighbors.extend(n)
        print(f"training audit {column}/{protocol}/{seed}: {len(ids)} rows", flush=True)
    detail = pd.concat(details, ignore_index=True)
    detail.to_csv(STUDY/"training_oof_rows.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    write_slices(detail, "training")
    associations, neighbors = pd.DataFrame(associations), pd.DataFrame(neighbors)
    associations.to_csv(STUDY/"training_associations.csv", index=False)
    neighbors.to_csv(STUDY/"training_molecule_consistency.csv", index=False)
    pd.DataFrame(molecules).to_csv(STUDY/"training_molecule_summary.csv", index=False)
    scales = pd.DataFrame(scales)
    scales.to_csv(STUDY/"scale_stability_by_seed.csv", index=False)
    summary = scales.groupby(["column", "protocol", "target", "budget"]).scale.agg(["mean", "std", "median", "min", "max"]).reset_index()
    summary["coefficient_cv"] = summary["std"]/summary["mean"]
    summary.to_csv(STUDY/"scale_stability_summary.csv", index=False)
    print(json.dumps(screen_directions(associations, neighbors), indent=2), flush=True)
    names = ["training_direction_decision.json", "training_direction_evidence.csv", "training_associations.csv",
             "training_molecule_consistency.csv", "training_oof_rows.csv.gz", "protocol.json"]
    write_json(STUDY/"training_audit_frozen.json", {"files": {name: sha256_file(STUDY/name) for name in names},
                                                "test_evaluation_started": False, "focal_contexts": 30})


def descriptive_test_audit():
    lock_protocol()
    model_path = STUDY/"model_protocol.json"
    if not model_path.exists():
        raise RuntimeError("freeze model/stop decision before reading test truth")
    model_protocol = json.loads(model_path.read_text())
    if model_protocol["selected_directions"] and not (STUDY/"model_predictions_frozen.json").exists():
        raise RuntimeError("all new predictions must freeze before test audit")
    frames = {c: pd.read_csv(STUDY/f"features_{c}.csv").set_index("sample_id") for c in COLUMNS}
    schedule = pd.read_csv(OLD/"splits/schedule_manifest.csv")
    details = []
    for (column, protocol, seed), context in schedule.loc[schedule.planned_budget.eq(100)].groupby(["column", "protocol", "outer_seed"]):
        ids = sorted(context.loc[context.role.eq("test"), "sample_id"])
        train_ids = sorted(context.loc[context.role.eq("gradient_train"), "sample_id"])
        frame = frames[column].loc[ids]
        old_path = OLD/column/protocol/f"seed_{seed}"/"budget_100"/"predictions.csv.gz"
        completion = json.loads((old_path.parent/"completion.json").read_text())
        if sha256_file(old_path) != completion["files"]["predictions.csv.gz"]:
            raise RuntimeError("historical predictions changed")
        previous = pd.read_csv(old_path).set_index("sample_id").loc[ids]
        truth = selected_truth(target_path(column), ids)
        detail = detailed_rows(frame, truth, previous[["scale_only_V1", "scale_only_V2"]].to_numpy(),
                              previous[["affine_V1", "affine_V2"]].to_numpy(),
                              frames[column].loc[train_ids, ["source_V1", "source_V2"]].to_numpy())
        for key, value in {"column": column, "protocol": protocol, "seed": seed}.items():
            detail[key] = value
        details.append(detail)
    all_rows = pd.concat(details, ignore_index=True)
    all_rows.to_csv(STUDY/"test_descriptive_rows.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    write_slices(all_rows, "test_descriptive")
    a, m, n = [], [], []
    for (column, protocol, seed), group in all_rows.groupby(["column", "protocol", "seed"]):
        ra, rm, rn = diagnostics(group, {"column": column, "protocol": protocol, "seed": int(seed), "budget": 100})
        a.extend(ra); m.extend(rm); n.extend(rn)
    pd.DataFrame(a).to_csv(STUDY/"test_descriptive_associations.csv", index=False)
    pd.DataFrame(m).to_csv(STUDY/"test_descriptive_molecules.csv", index=False)
    pd.DataFrame(n).to_csv(STUDY/"test_descriptive_molecule_consistency.csv", index=False)
    write_json(STUDY/"test_descriptive_audit.json", {"rows": len(all_rows), "contexts": 30,
              "selection_changed": False, "model_protocol_sha256": sha256_file(model_path)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-audit", action="store_true")
    parser.add_argument("--test-descriptive", action="store_true")
    args = parser.parse_args()
    if args.train_audit:
        train_audit()
    elif args.test_descriptive:
        descriptive_test_audit()
    else:
        prepare_features()
