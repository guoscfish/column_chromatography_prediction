#!/usr/bin/env python3
"""Independently rescore completed inner predictions and verify identity roles."""
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
from scripts.studies import run_column_conditioned_multitask as r


def audit():
    torch.set_num_threads(1)
    r.execution_provenance()
    schedule = r.verify_schedule()
    source_roles = pd.read_csv(r.QUALIFIED / "splits/row_seed_42.csv")
    decision = json.loads((r.STUDY / "INNER_DECISION.json").read_text())
    if len(decision["completion_hashes"]) != 50:
        raise RuntimeError("audit requires all 50 fits")
    max_metric_error, max_tail_error = 0., 0.
    observations, prediction_count = 0, 0
    source_state = torch.load(r.source_path(), map_location="cpu", weights_only=False)["model_state_dict"]
    adapted_prefixes = ("head.", "condition_branch.", "backbone.convs.3.", "backbone.convs.4.",
                        "backbone.convs_bond_angle.3.", "backbone.convs_bond_float.3.",
                        "backbone.convs_bond_embeding.3.", "backbone.convs_angle_float.3.")
    frozen_names = [name for name in source_state if not name.startswith(adapted_prefixes)]
    for relative, expected in decision["completion_hashes"].items():
        run = r.STUDY / relative
        assert r.sha(run / "complete.json") == expected
        assert r.verify_completed(run)
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        for name in frozen_names:
            assert torch.equal(checkpoint["model_state_dict"][name], source_state[name]), name
        values = pd.read_csv(run / "results.csv")
        predictions = pd.read_csv(run / "predictions.csv")
        identities = pd.read_csv(run / "identities.csv")
        tails = pd.read_csv(run / "tails.csv")
        assert values.best_epoch.nunique() == 1
        targets = identities.loc[identities.column.isin(r.TASKS[1:])]
        assert not set(targets.loc[targets.role.eq("inner_train"), "canonical_smiles"]) & set(targets.loc[targets.role.eq("inner_validation"), "canonical_smiles"])
        for row in values.itertuples(index=False):
            c, seed = row.column, row.outer_seed
            current = identities.loc[identities.column.eq(c)]
            train_ids = current.loc[current.role.eq("inner_train"), "sample_id"].astype(str).tolist()
            valid_ids = current.loc[current.role.eq("inner_validation"), "sample_id"].astype(str).tolist()
            if c == "4g":
                canonical = r.SOURCE_DATA
                outer_train = source_roles.loc[source_roles.split.eq("train"), "sample_id"].astype(str).tolist()
                assert set(train_ids) == set(outer_train)
                assert set(valid_ids) == set(source_roles.loc[source_roles.split.eq("validation"), "sample_id"].astype(str))
            else:
                canonical = r.PARENT / f"filtered_canonical_{c}.csv"
                roles = schedule.loc[schedule.column.eq(c) & schedule.protocol.eq("compound") & schedule.outer_seed.eq(seed)]
                outer_train = roles.loc[roles.role.eq("gradient_train"), "sample_id"].astype(str).tolist()
                assert set(train_ids) | set(valid_ids) == set(outer_train)
                assert not set(train_ids) & set(valid_ids)
            assert r.digest(sorted(train_ids)) == row.inner_train_ids_sha256
            assert r.digest(sorted(valid_ids)) == row.inner_validation_ids_sha256
            pred = predictions.loc[predictions.column.eq(c)]
            assert pred.sample_id.is_unique and set(pred.sample_id) == set(valid_ids)
            truth = r._read_authorized_truth(canonical, valid_ids)
            training_truth = r._read_authorized_truth(canonical, train_ids)
            outer_training_truth = r._read_authorized_truth(canonical, outer_train)
            points = pred.set_index("sample_id").loc[valid_ids, r.QUANTILES].to_numpy(float)
            scale_values = np.std(training_truth, axis=0, ddof=0)
            error = points[:, [1, 4]] - truth
            rmses = np.sqrt(np.mean(error ** 2, axis=0))
            maes = np.mean(np.abs(error), axis=0)
            independent = {"V1_rmse": rmses[0], "V2_rmse": rmses[1], "V1_mae": maes[0], "V2_mae": maes[1],
                           "combined_normalized_rmse": np.sqrt(np.mean((rmses / scale_values) ** 2)),
                           "Center_rmse": np.sqrt(np.mean(error.mean(axis=1) ** 2)),
                           "Width_rmse": np.sqrt(np.mean((error[:, 1] - error[:, 0]) ** 2))}
            for name, value in independent.items():
                difference = abs(float(value) - float(getattr(row, name)))
                max_metric_error = max(max_metric_error, difference)
                np.testing.assert_allclose(value, getattr(row, name), rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(scale_values, [row.scale_V1, row.scale_V2], rtol=1e-12, atol=1e-12)
            threshold = np.quantile(outer_training_truth, .8, axis=0)
            for j, endpoint in enumerate(("V1", "V2")):
                recorded = tails.loc[tails.column.eq(c) & tails.endpoint.eq(endpoint)].iloc[0]
                mask = truth[:, j] >= threshold[j]
                sse = np.square(error[mask, j]).sum()
                assert int(mask.sum()) == int(recorded.tail_count)
                np.testing.assert_allclose(threshold[j], recorded.threshold, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(sse, recorded.tail_sse, rtol=1e-10, atol=1e-10)
                max_tail_error = max(max_tail_error, abs(float(sse) - float(recorded.tail_sse)))
            observations += 1
            prediction_count += len(points)
    if not decision["passed"]:
        assert not (r.STUDY / "PREDICTION_FREEZE_MANIFEST.json").exists()
        assert not (r.STUDY / "OUTER_RESULTS.csv").exists()
        assert not list((r.STUDY / "runtime/final").glob("**/predictions_blind.csv"))
    result = {"status": "PASS", "fits_verified": 50, "task_fold_observations": observations,
              "prediction_rows_verified": prediction_count, "max_metric_abs_difference": max_metric_error,
              "max_tail_sse_abs_difference": max_tail_error, "global_target_group_overlap": 0,
              "frozen_source_tensors_verified_per_fit": len(frozen_names),
              "outer_target_validation_or_test_truth_read": False, "inner_decision_sha256": r.sha(r.STUDY / "INNER_DECISION.json")}
    r.write_json(r.STUDY / "EVIDENCE_AUDIT.json", result, immutable=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    audit()
