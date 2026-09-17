import json

import numpy as np
import pandas as pd
import pytest

from src.qgeognn_al.active_learning_v2 import center_width_performance as cw
from src.qgeognn_al.active_learning_v2.benchmark import create_smoke_fixture
from src.qgeognn_al.active_learning_v2.benchmark_seed import SeedContext
from src.qgeognn_al.active_learning_v2.gradient_features import extract_linear_output_gradient_sketches_many
from src.qgeognn_al.active_learning_v2.gradient_transforms import center_width_transform
from src.qgeognn_al.active_learning_v2.lcmd import lcmd_tp_select
from src.qgeognn_al.active_learning_v2.protocol import make_row_protocol
from src.qgeognn_al.artifacts import sha256_file
from src.qgeognn_al.models import load_predictor_checkpoint


def test_cw_hidden_truth_mutation_leaves_actual_gradient_and_selection_unchanged(tmp_path):
    source, _ = create_smoke_fixture(tmp_path / "fixture", rows=160)
    data = pd.read_csv(source)
    partition = make_row_protocol(data, 73)
    hidden = partition.role.isin(["u0", "test"])
    mutated = data.copy()
    mutated.loc[hidden, ["V1_ml", "V2_ml"]] = 1e12
    changed = tmp_path / "changed.csv"
    mutated.to_csv(changed, index=False)
    results = []
    for name, path in (("original", source), ("changed", changed)):
        context = SeedContext(path, tmp_path / name, 73, partition, smoke=True)
        fit = context.fit("baseline_l0", context.roles["l0"], context.l0_truth)
        transform, _ = center_width_transform(context.l0_truth)
        model = load_predictor_checkpoint(context.runtime / "baseline_l0/best.pt")
        features = extract_linear_output_gradient_sketches_many(
            model, context.atom, context.angle, context.outer, {"cw": transform},
            dimension=512, sketch_seed=cw.sketch_seed(73))["cw"].features
        n = len(context.roles["l0"])
        positions = lcmd_tp_select(features[n:], features[:n], 32).selected_pool_positions
        ids = context.ids(context.roles["u0"][positions])
        cw.validate_batch(ids, context.ids(context.roles["u0"]))
        assert all(item["purpose"] == "initial_fit" for item in context.store.audit)
        results.append((fit["checkpoint_state_hash"], features, ids))
    assert results[0][0] == results[1][0]
    np.testing.assert_array_equal(results[0][1], results[1][1])
    assert results[0][2] == results[1][2]


@pytest.mark.parametrize("ids,pool", [(list(range(31)), range(40)),
                                       ([0] * 32, range(40)), (list(range(32)), range(31))])
def test_invalid_batch_rejected(ids, pool):
    with pytest.raises(RuntimeError):
        cw.validate_batch(ids, pool)


def test_acquisition_and_prediction_freezes_reject_tampering_before_truth(tmp_path, monkeypatch):
    artifact = tmp_path / "predictions.csv"
    artifact.write_text("frozen predictions")
    files = {str(artifact): sha256_file(artifact)}
    cw.verify_files(files)
    artifact.write_text("tampered")
    with pytest.raises(RuntimeError, match="frozen artifact changed"):
        cw.verify_files(files)
    (tmp_path / "pre_test_freeze.json").write_text(json.dumps({
        "seeds": list(cw.DEVELOPMENT_SEEDS), "status": "FROZEN_BEFORE_TEST_TRUTH", "files": files}))
    monkeypatch.setattr(cw, "verify_preflight", lambda study: None)
    monkeypatch.setattr(cw, "RestrictedLabelStore", lambda *args: pytest.fail("truth store constructed before barrier"))
    with pytest.raises(RuntimeError, match="frozen artifact changed"):
        cw.analyze(tmp_path, tmp_path)


def test_execution_requires_acquisition_freeze_before_selected_truth(tmp_path):
    with pytest.raises(FileNotFoundError):
        cw.execute(tmp_path, tmp_path, study=tmp_path)


def test_decision_rule_complete_cohort_wins_mean_and_endpoint_guard():
    paired = pd.DataFrame({"outer_seed": cw.DEVELOPMENT_SEEDS, "Raw_NRMSE": 1.,
                           "CW_NRMSE": [0.9, 0.9, 0.9, 0.9, 1.1],
                           "Raw_V1_RMSE": 10., "Raw_V2_RMSE": 20.,
                           "CW_V1_RMSE": 10.1, "CW_V2_RMSE": 19.})
    assert cw.decision(paired)["decision"] == "CENTER_WIDTH_PROMISING"
    paired.loc[0, "CW_NRMSE"] = 1.01
    assert cw.decision(paired)["decision"] == "CENTER_WIDTH_UNSTABLE_SIGNAL"
    paired["CW_NRMSE"] = 0.9
    paired["CW_V1_RMSE"] = 10.21
    assert cw.decision(paired)["decision"] == "CENTER_WIDTH_UNSTABLE_SIGNAL"
    paired["CW_NRMSE"] = 1.0
    assert cw.decision(paired)["decision"] == "CENTER_WIDTH_NO_GAIN"
    with pytest.raises(RuntimeError, match="complete five-seed"):
        cw.decision(paired.iloc[:-1])


def test_non_development_seed_rejected_before_artifact_read(tmp_path):
    with pytest.raises(RuntimeError, match="only development"):
        cw.audited_seed(29, tmp_path, tmp_path)
