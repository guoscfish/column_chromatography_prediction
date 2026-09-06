#!/usr/bin/env python3
"""The single direction supported by training evidence; no post-test iteration."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.studies.run_scaling_failure_audit import (
    COLUMNS, OLD, PREVIOUS, SCALES, STUDY, lock_protocol, selected_truth, target_path, write_json,
)
from src.qgeognn_al.artifacts import sha256_file
from src.qgeognn_al.evaluation.point import point_metrics
from src.qgeognn_al.transfer.calibration import fit_affine, fit_scale_only
from src.qgeognn_al.transfer.conditional_scaling import fit_conditional
from src.qgeognn_al.transfer.residual_diagnostics import validation_score

PENALTIES = (0., .1, 1.)
MASS = {"8g": 2., "25g": 6.25, "40g": 10.}
BASELINES = ("scale_only", "affine", "local_identity_shrinkage", "target_head_only")
METHODS = ("conditional_EA", "additive_EA_control", "conditional_policy", "additive_policy")
POINT_NAMES = ("V1_r2", "V1_rmse", "V1_mae", "V2_r2", "V2_rmse", "V2_mae", "normalized_rmse", "combined_normalized_rmse")


def prepare():
    lock_protocol()
    freeze = json.loads((STUDY/"training_audit_frozen.json").read_text())
    for name, digest in freeze["files"].items():
        if sha256_file(STUDY/name) != digest:
            raise RuntimeError("training direction evidence changed")
    decision = json.loads((STUDY/"training_direction_decision.json").read_text())
    if decision["selected_directions"] != ["CONDITIONAL_SCALING"]:
        raise RuntimeError("this fixed model only implements the supported conditional direction")
    paths = [Path(__file__), ROOT/"src/qgeognn_al/transfer/conditional_scaling.py", STUDY/"MODEL_PREREGISTRATION.md",
             STUDY/"training_audit_frozen.json", *[STUDY/f"features_{c}.csv" for c in COLUMNS]]
    config = {"selected_directions": decision["selected_directions"], "penalties": list(PENALTIES),
              "methods": list(METHODS), "baselines": list(BASELINES), "feature": "EA_fraction",
              "contexts": 120, "fit_labels": "gradient_train", "selection_labels": "validation",
              "test_tuning": False, "model_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}}
    path = STUDY/"model_protocol.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("model protocol changed")
    write_json(path, config)
    return config


def metric(truth, prediction):
    six = np.column_stack([prediction[:, 0]]*3+[prediction[:, 1]]*3)
    result = point_metrics(truth, six, SCALES)
    result["normalized_rmse"] = .5*sum(result[f"{t}_rmse"]/SCALES[t] for t in ("V1", "V2"))
    return result


def fit_all():
    config = prepare()
    scale_vector = np.array([SCALES[t] for t in ("V1", "V2")])
    features = {c: pd.read_csv(STUDY/f"features_{c}.csv").set_index("sample_id") for c in COLUMNS}
    schedule = pd.read_csv(OLD/"splits/schedule_manifest.csv")
    manifest = {}
    for (column, protocol, seed, budget), context in schedule.groupby(["column", "protocol", "outer_seed", "planned_budget"]):
        output = STUDY/"models"/column/protocol/f"seed_{seed}"/f"budget_{budget}"
        output.mkdir(parents=True, exist_ok=True)
        done = output/"frozen.json"
        if done.exists():
            for name, digest in json.loads(done.read_text())["files"].items():
                if sha256_file(output/name) != digest:
                    raise RuntimeError("completed model context changed")
            manifest[str(done.relative_to(STUDY))] = sha256_file(done)
            continue
        ids = {role: sorted(context.loc[context.role.eq(role), "sample_id"]) for role in ("gradient_train", "validation", "test")}
        frame = features[column]
        x = {role: frame.loc[index, ["source_V1", "source_V2"]].to_numpy() for role, index in ids.items()}
        ea = {role: frame.loc[index, "EA_fraction"].to_numpy() for role, index in ids.items()}
        truth = {role: selected_truth(target_path(column), ids[role]) for role in ("gradient_train", "validation")}
        train, valid, test = "gradient_train", "validation", "test"
        predictions, valid_predictions = {}, {}
        for name, fitter in (("scale_only", fit_scale_only), ("affine", fit_affine)):
            predictions[name] = fitter(truth[train], x[train], x[test]).prediction
            valid_predictions[name] = fitter(truth[train], x[train], x[valid]).prediction
        previous = PREVIOUS/"contexts"/column/protocol/f"seed_{seed}"/f"budget_{budget}"
        old_frozen = json.loads((previous/"frozen.json").read_text())
        if sha256_file(previous/"fit_audit.json") != old_frozen["files"]["fit_audit.json"]:
            raise RuntimeError("local shrinkage control changed")
        coefficient = np.asarray(json.loads((previous/"fit_audit.json").read_text())["local_identity_shrinkage"]["coefficients"])[COLUMNS.index(column)]
        for role, container in ((valid, valid_predictions), (test, predictions)):
            container["local_identity_shrinkage"] = (x[role]/scale_vector*coefficient[:, 0]+coefficient[:, 1])*MASS[column]*scale_vector
        scores = [{"method": name, "penalty": None, "validation_score": validation_score(truth[valid], pred, scale_vector)}
                  for name, pred in valid_predictions.items()]
        fits = {}
        for name, interaction in (("conditional_EA", True), ("additive_EA_control", False)):
            models = [fit_conditional(x[train], truth[train], ea[train], scale_vector, MASS[column], penalty,
                                      interaction=interaction) for penalty in PENALTIES]
            values = [validation_score(truth[valid], model.predict(x[valid], ea[valid]), scale_vector) for model in models]
            best = min(range(len(values)), key=lambda i: (values[i], -PENALTIES[i]))
            predictions[name] = models[best].predict(x[test], ea[test])
            valid_predictions[name] = models[best].predict(x[valid], ea[valid])
            fits[name] = {"selected_penalty": PENALTIES[best], "candidates": [model.audit() for model in models]}
            scores.extend({"method": name, "penalty": penalty, "validation_score": value} for penalty, value in zip(PENALTIES, values))
        choices = {}
        for policy, addition in (("conditional_policy", "conditional_EA"), ("additive_policy", "additive_EA_control")):
            candidates = ["scale_only", "affine", "local_identity_shrinkage", addition]
            best = min(candidates, key=lambda name: validation_score(truth[valid], valid_predictions[name], scale_vector))
            predictions[policy] = predictions[best]
            choices[policy] = best
        table = pd.DataFrame({"sample_id": ids[test]})
        for name, value in predictions.items():
            if not np.isfinite(value).all():
                raise RuntimeError("nonfinite model prediction")
            table[[f"{name}_V1", f"{name}_V2"]] = value
        table.to_csv(output/"predictions_blind.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        write_json(output/"fit_audit.json", {"fits": fits, "validation_candidates": scores, "policy_choices": choices})
        write_json(output/"label_usage.json", {"column": column, "protocol": protocol, "seed": int(seed), "budget": int(budget),
                   "actual_budget": int(context.actual_budget.iloc[0]), **ids,
                   "donor_target_labels_used": 0, "paired_source_label_features_used": 0, "test_labels_used_for_fit_or_selection": 0})
        write_json(done, {"files": {name: sha256_file(output/name) for name in ["predictions_blind.csv.gz", "fit_audit.json", "label_usage.json"]}})
        manifest[str(done.relative_to(STUDY))] = sha256_file(done)
        if budget == 100:
            print(f"conditional fits frozen: {column}/{protocol}/{seed}", flush=True)
    if len(manifest) != config["contexts"]:
        raise RuntimeError("incomplete fit phase")
    write_json(STUDY/"model_predictions_frozen.json", {"contexts": len(manifest), "files": manifest,
               "protocol_sha256": sha256_file(STUDY/"model_protocol.json"), "target_test_evaluation_started": False})


def paired_scores(frame, value_column):
    rows = []
    for (column, protocol), group in frame.groupby(["column", "protocol"]):
        pivot = group.pivot(index="seed", columns="method", values=value_column)
        for method in METHODS:
            for reference in (*BASELINES, "additive_policy"):
                if method == reference:
                    continue
                delta = pivot[method]-pivot[reference]
                gain = 1-pivot[method].mean()/pivot[reference].mean()
                rows.append({"column": column, "protocol": protocol, "method": method, "reference": reference,
                    "relative_gain": gain, "mean_delta": delta.mean(), "median_delta": delta.median(), "std_delta": delta.std(ddof=1),
                    "wins": int((delta<0).sum()), "seeds": len(delta),
                    "material": bool(gain>=.05 and delta.median()<0 and (delta<0).sum()>=4)})
    return pd.DataFrame(rows)


def replicated_contexts(pairs, point=None):
    contexts = []
    refs = ["scale_only", "affine", "local_identity_shrinkage", "additive_policy"]
    for (column, protocol), group in pairs.loc[pairs.method.eq("conditional_policy")].groupby(["column", "protocol"]):
        records = group.set_index("reference")
        passed = bool(records.loc[refs, "material"].all())
        if point is not None:
            means = point.loc[point.column.eq(column) & point.protocol.eq(protocol)].groupby("method")[["V1_rmse", "V2_rmse"]].mean()
            passed &= bool((means.loc[refs] >= means.loc["conditional_policy"]).all().all())
        if passed:
            contexts.append((column, protocol))
    paired_context = any((column, "row") in contexts and (column, "compound") in contexts for column in COLUMNS)
    compound_columns = [column for column, protocol in contexts if protocol == "compound"]
    for column in compound_columns.copy():
        row = pairs.loc[pairs.column.eq(column) & pairs.protocol.eq("row") & pairs.method.eq("conditional_policy")].set_index("reference")
        if (row.loc[refs, "relative_gain"] < -.05).any():
            compound_columns.remove(column)
    return bool(paired_context or len(compound_columns)>=2), contexts


def evaluate():
    prepare()
    freeze = json.loads((STUDY/"model_predictions_frozen.json").read_text())
    if freeze["contexts"] != 120 or freeze["protocol_sha256"] != sha256_file(STUDY/"model_protocol.json"):
        raise RuntimeError("all predictions must freeze under the exact model protocol")
    for relative, digest in freeze["files"].items():
        done = STUDY/relative
        if sha256_file(done) != digest:
            raise RuntimeError("freeze manifest changed")
        for name, expected in json.loads(done.read_text())["files"].items():
            if sha256_file(done.parent/name) != expected:
                raise RuntimeError("fit/predictions changed")
    rows, selections, usage_rows = [], [], []
    for relative in freeze["files"]:
        output = (STUDY/relative).parent
        usage = json.loads((output/"label_usage.json").read_text())
        keys = {k: usage[k] for k in ("column", "protocol", "seed", "budget", "actual_budget")}
        column, protocol, seed, budget = [keys[k] for k in ("column", "protocol", "seed", "budget")]
        table = pd.read_csv(output/"predictions_blind.csv.gz").set_index("sample_id")
        truth = selected_truth(target_path(column), table.index.tolist())
        old = OLD/column/protocol/f"seed_{seed}"/f"budget_{budget}"
        old_completion = json.loads((old/"completion.json").read_text())
        if sha256_file(old/"predictions.csv.gz") != old_completion["files"]["predictions.csv.gz"]:
            raise RuntimeError("original baseline changed")
        baseline = pd.read_csv(old/"predictions.csv.gz").set_index("sample_id").loc[table.index]
        previous = PREVIOUS/"contexts"/column/protocol/f"seed_{seed}"/f"budget_{budget}"
        previous_freeze = json.loads((previous/"frozen.json").read_text())
        if sha256_file(previous/"predictions_blind.csv.gz") != previous_freeze["files"]["predictions_blind.csv.gz"]:
            raise RuntimeError("frozen local control predictions changed")
        local = pd.read_csv(previous/"predictions_blind.csv.gz").set_index("sample_id").loc[table.index]
        for name in BASELINES:
            columns = [f"{name}_V1", f"{name}_V2"]
            prediction = local[columns].to_numpy() if name == "local_identity_shrinkage" else baseline[columns].to_numpy()
            if name != "target_head_only":
                np.testing.assert_allclose(table[columns], prediction, rtol=3e-5, atol=1e-4)
            rows.append({**keys, "method": name, **metric(truth, prediction)})
        for name in METHODS:
            rows.append({**keys, "method": name, **metric(truth, table[[f"{name}_V1", f"{name}_V2"]].to_numpy())})
        audit = json.loads((output/"fit_audit.json").read_text())
        selections.append({**keys, **audit["policy_choices"], **{f"{m}_penalty": v["selected_penalty"] for m, v in audit["fits"].items()}})
        usage_rows.append({**keys, "gradient_train_rows": len(usage["gradient_train"]), "validation_rows": len(usage["validation"]),
                           "test_rows": len(usage["test"]), "test_labels_used_for_fit_or_selection": 0})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(STUDY/"model_all_metrics.csv", index=False)
    metrics.groupby(["column", "protocol", "budget", "method"])[list(POINT_NAMES)].agg(["mean", "std", "median", "min", "max"]).to_csv(STUDY/"model_aggregate_metrics.csv")
    aulc = []
    for (column, protocol, seed, method), group in metrics.groupby(["column", "protocol", "seed", "method"]):
        group = group.sort_values("budget")
        actual = group.actual_budget.to_numpy()
        aulc.append({"column": column, "protocol": protocol, "seed": seed, "method": method,
                     "normalized_aulc": np.trapezoid(group.normalized_rmse, group.budget)/70,
                     "actual_budget_aulc": np.trapezoid(group.normalized_rmse, actual)/(actual[-1]-actual[0])})
    aulc = pd.DataFrame(aulc)
    aulc.to_csv(STUDY/"model_aulc_by_seed.csv", index=False)
    aulc.groupby(["column", "protocol", "method"]).normalized_aulc.agg(["mean", "std", "median", "min", "max"]).to_csv(STUDY/"model_aulc_summary.csv")
    pairs = paired_scores(aulc, "normalized_aulc")
    pairs.to_csv(STUDY/"model_paired_aulc.csv", index=False)
    b100 = metrics.loc[metrics.budget.eq(100)]
    accuracy = paired_scores(b100, "normalized_rmse")
    accuracy.to_csv(STUDY/"model_paired_budget100.csv", index=False)
    efficiency_pass, efficiency_contexts = replicated_contexts(pairs)
    accuracy_pass, accuracy_contexts = replicated_contexts(accuracy, b100)
    result = {"direction": "CONDITIONAL_SCALING", "training_structure_identified": True,
              "label_efficiency_material": efficiency_pass, "high_budget_accuracy_material": accuracy_pass,
              "efficiency_qualifying_contexts": efficiency_contexts, "accuracy_qualifying_contexts": accuracy_contexts,
              "decision": "MATERIAL_GAIN" if efficiency_pass or accuracy_pass else "STRUCTURED_FAILURE_BUT_NO_MATERIAL_MODEL_GAIN",
              "gain_types": (["LABEL_EFFICIENCY_GAIN"] if efficiency_pass else [])+(["ACCURACY_GAIN"] if accuracy_pass else []),
              "new_model_directions_run": 1, "test_based_method_additions": 0, "predictor_modified": False,
              "paired_delta_run": False, "molecule_dependent_run": False,
              "scope": "one EA-dependent slope family; evidence does not establish causal condition effects or close other hypotheses"}
    write_json(STUDY/"model_decision.json", result)
    pd.DataFrame(selections).to_csv(STUDY/"model_selection.csv", index=False)
    pd.DataFrame(usage_rows).to_csv(STUDY/"model_label_usage.csv", index=False)
    write_json(STUDY/"model_execution_audit.json", {"contexts": 120, "metric_rows": len(metrics), "failures": 0,
               "all_predictions_frozen_before_test": True, "protocol_sha256": sha256_file(STUDY/"model_protocol.json")})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.fit:
        fit_all()
    elif args.evaluate:
        evaluate()
    else:
        prepare()
