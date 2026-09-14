#!/usr/bin/env python3
"""Independently verify completed A2-PCGrad COMPOUND inner evidence."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.studies import run_column_conditioned_pcgrad as run


def audit() -> dict:
    torch.set_num_threads(1)
    run.prepare()
    run.execution_provenance()
    run.verify_reference()
    decision = json.loads((run.STUDY / "INNER_DECISION.json").read_text())
    if decision["passed"] or len(decision["completion_hashes"]) != 25:
        raise RuntimeError("audit expects a complete failed COMPOUND gate")
    maximum_metric_error = maximum_tail_error = 0.0
    prediction_rows = target_observations = 0
    global_overlap = 0
    source_state = torch.load(run.base.source_path(), map_location="cpu", weights_only=False)["model_state_dict"]
    adapted = ("head.", "condition_branch.", "backbone.convs.3.", "backbone.convs.4.",
               "backbone.convs_bond_angle.3.", "backbone.convs_bond_float.3.",
               "backbone.convs_bond_embeding.3.", "backbone.convs_angle_float.3.")
    frozen_names = [name for name in source_state if not name.startswith(adapted)]
    for seed in run.SEEDS:
        data = run.load_data("compound", seed)
        for fold, (train, valid) in enumerate(run.base.fold_schedules(data), 1):
            run.assert_context_comparable(data, train, valid, "compound", seed, fold)
            train_groups = {str(data[column]["frame"].iloc[index].canonical_smiles)
                            for column in run.TASKS[1:] for index in train[column]}
            valid_groups = {str(data[column]["frame"].iloc[index].canonical_smiles)
                            for column in run.TASKS[1:] for index in valid[column]}
            global_overlap += len(train_groups & valid_groups)
            path = run.STUDY / f"runtime/inner/compound/seed_{seed}/fold_{fold}/{run.METHOD}"
            relative = str(path.relative_to(run.STUDY))
            if run.sha(path / "complete.json") != decision["completion_hashes"][relative] or not run.verify_completed(path):
                raise RuntimeError("completion hash mismatch")
            checkpoint = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
            model = run.base.build_multitask_model(run.base.source_path(), run.base.ARMS["A2"])
            model.load_state_dict(checkpoint["model_state_dict"])
            protocol = json.loads((run.STUDY / "protocol.json").read_text())
            assert run.architecture_signature(model) == protocol["architecture_signature_sha256"]
            assert run.trainable_signature(model)[0] == protocol["trainable_parameter_names_sha256"]
            assert run.shared_signature(model)[0] == protocol["pcgrad_parameter_names_sha256"]
            for name in frozen_names:
                assert torch.equal(checkpoint["model_state_dict"][name], source_state[name]), name
            recorded_results = pd.read_csv(path / "results.csv")
            recorded_tails = pd.read_csv(path / "tails.csv")
            predictions = pd.read_csv(path / "predictions.csv")
            for column in run.TASKS:
                row = recorded_results.loc[recorded_results.column == column].iloc[0]
                ids = data[column]["frame"].iloc[valid[column]].sample_id.astype(str).tolist()
                prediction = predictions.loc[predictions.column == column].copy()
                prediction.sample_id = prediction.sample_id.astype(str)
                points = prediction.set_index("sample_id").loc[ids, run.base.QUANTILES].to_numpy(float)
                truth = data[column]["truth"][valid[column]]
                scales = run.base.scales(data[column]["truth"][train[column]])
                values, tails, _ = run.base.metric_rows(truth, points, scales, data[column]["tail"], data[column]["regimes"])
                for metric in ("V1_rmse", "V1_mae", "V2_rmse", "V2_mae", "combined_normalized_rmse", "Center_rmse", "Width_rmse"):
                    difference = abs(float(values[metric]) - float(row[metric]))
                    maximum_metric_error = max(maximum_metric_error, difference)
                    np.testing.assert_allclose(values[metric], row[metric], rtol=1e-10, atol=1e-10)
                for tail in tails:
                    recorded = recorded_tails.loc[(recorded_tails.column == column) & (recorded_tails.endpoint == tail["endpoint"])].iloc[0]
                    for metric in ("threshold", "tail_sse", "total_sse", "tail_rmse"):
                        difference = abs(float(tail[metric]) - float(recorded[metric]))
                        maximum_tail_error = max(maximum_tail_error, difference)
                        np.testing.assert_allclose(tail[metric], recorded[metric], rtol=1e-10, atol=1e-10)
                prediction_rows += len(points)
                if column in run.TASKS[1:]:
                    target_observations += 1
    recomputed = run.compound_decision(pd.read_csv(run.STUDY / "INNER_RESULTS.csv"),
                                       pd.read_csv(run.STUDY / "TAIL_METRICS.csv"),
                                       run.mechanism_summary(pd.read_csv(run.STUDY / "GRADIENT_RAW.csv"),
                                                             pd.read_csv(run.STUDY / "GRADIENT_PROJECTED.csv")))
    assert not recomputed["passed"] and recomputed["status"] == decision["status"]
    prohibited = ("ROW_INNER_RESULTS.csv", "ROW_INNER_DECISION.json", "PREDICTION_FREEZE_MANIFEST.json",
                  "OUTER_RESULTS.csv", "ROW_OUTER_RESULTS.csv")
    if any((run.STUDY / name).exists() for name in prohibited):
        raise RuntimeError("failed gate must block ROW and outer artifacts")
    result = {"status": "PASS", "fits_verified": 25, "target_fold_observations": target_observations,
              "prediction_rows_verified": prediction_rows, "global_target_group_overlap": global_overlap,
              "frozen_source_tensors_verified_per_fit": len(frozen_names),
              "max_metric_abs_difference": maximum_metric_error,
              "max_tail_abs_difference": maximum_tail_error,
              "outer_validation_or_test_truth_read": False,
              "row_and_outer_actions_blocked": True,
              "inner_decision_sha256": run.sha(run.STUDY / "INNER_DECISION.json")}
    run.write_json(run.STUDY / "EVIDENCE_AUDIT.json", result, immutable=True)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    audit()
