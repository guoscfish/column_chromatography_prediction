import inspect
import json

import numpy as np
import pandas as pd
import torch

from scripts.studies import run_cross_column_transfer as study
from src.qgeognn_al.artifacts import sha256_file
from src.qgeognn_al.transfer.calibration import (
    fit_affine,
    fit_affine_condition_residual,
    fit_scale_only,
    mass_ratio_prediction,
)


def test_scale_affine_and_mass_ratio_definitions():
    source = np.array([[1., 2.], [2., 4.], [3., 6.]])
    truth = source * [2., 3.] + [1., -2.]
    target = np.array([[4., 8.]])
    affine = fit_affine(truth, source, target)
    np.testing.assert_allclose(affine.prediction, target * [2., 3.] + [1., -2.])
    np.testing.assert_allclose(affine.coefficients, [[2., 1.], [3., -2.]])
    scaled_truth = source * [2., 3.]
    np.testing.assert_allclose(fit_scale_only(scaled_truth, source, target).prediction, target * [2., 3.])
    np.testing.assert_allclose(mass_ratio_prediction(target, 8.), target * 2.)


def test_condition_residual_selection_is_group_nested_and_finite():
    source = np.column_stack([np.arange(1., 13.), np.arange(2., 26., 2.)])
    conditions = np.column_stack([np.tile([0., 1., 2.], 4), np.linspace(-1., 1., 12)])
    truth = source * [1.5, 2.] + [2., -3.] + conditions @ np.array([[.4, -.2], [.1, .3]])
    groups = np.repeat(["a", "b", "c", "d"], 3)
    fit, alpha, policy = fit_affine_condition_residual(
        truth, source, conditions, groups, source[:2], conditions[:2], [.1, 1., 10.], np.ones(2)
    )
    assert fit.prediction.shape == (2, 2)
    assert np.isfinite(fit.prediction).all()
    assert alpha in {.1, 1., 10.}
    assert policy == "groupkfold_4_gradient_train_only_refit_affine"
    implementation = inspect.getsource(fit_affine_condition_residual)
    assert implementation.index("for inner_train, inner_valid") < implementation.index("inner_affine_train = fit_affine")


def test_target_group_isolation_and_nested_budgets():
    schedule = pd.read_csv(study.STUDY / "splits/schedule_manifest.csv")
    for _, context in schedule.loc[schedule.protocol.eq("compound")].groupby(
        ["column", "outer_seed", "planned_budget"]
    ):
        sets = {role: set(context.loc[context.role.eq(role), "canonical_smiles"]) for role in ("gradient_train", "validation", "test")}
        assert not sets["gradient_train"] & sets["validation"]
        assert not sets["gradient_train"] & sets["test"]
        assert not sets["validation"] & sets["test"]
    for _, contexts in schedule.loc[schedule.protocol.eq("compound")].groupby(["column", "outer_seed"]):
        previous = set()
        for budget in study.BUDGETS:
            current = set(contexts.loc[(contexts.planned_budget.eq(budget)) & contexts.role.eq("gradient_train"), "sample_id"])
            assert previous <= current
            previous = current


def test_source_overlap_audit_and_ood_claim_are_explicit():
    for column in study.COLUMNS:
        audit = json.loads((study.STUDY / "data_audit" / f"audit_{column}.json").read_text())
        overlap = pd.read_csv(study.STUDY / "data_audit" / f"compound_overlap_{column}.csv")
        assert len(overlap) == audit["unique_canonical_compounds"]
        assert int((~overlap.present_in_fixed_4g_source_train).sum()) == audit["target_compounds_absent_from_fixed_source_train"]
        assert audit["source_unseen_ood_status"] == "SOURCE_UNSEEN_MOLECULE_OOD_NOT_ESTIMABLE_WITH_CURRENT_DATA"


def test_truth_access_and_protocol_hash_locking():
    implementation = inspect.getsource(study.run_context)
    assert 'feature_columns = [name for name in pd.read_csv(data_path, nrows=0).columns' in implementation
    assert implementation.index("for method, mode in neural.items()") < implementation.index("test_truth_table = load_selected_truth")
    assert implementation.index("test_truth_table = load_selected_truth") > implementation.index("train_adaptation(")
    protocol = json.loads((study.STUDY / "protocol.json").read_text())
    assert sha256_file(study.STUDY / "splits/schedule_manifest.csv") == protocol["schedule_sha256"]
    for column in study.COLUMNS:
        assert sha256_file(study.canonical_path(column)) == protocol["canonical_sha256"][column]
    assert protocol["target_threshold"] is None
    assert protocol["active_learning_executed"] is False


def test_split_reproducibility_without_reading_labels():
    implementation = inspect.getsource(study.make_splits)
    assert "V1_ml" not in implementation and "V2_ml" not in implementation
    before = sha256_file(study.STUDY / "splits/schedule_manifest.csv")
    study.make_splits()
    assert sha256_file(study.STUDY / "splits/schedule_manifest.csv") == before


def test_cached_head_representation_is_forward_equivalent():
    from src.qgeognn_al.data import build_model_data
    from src.qgeognn_al.models import load_predictor_checkpoint
    from src.qgeognn_al.training.predictor import loader_pair

    data = pd.read_csv(study.canonical_path("25g")).iloc[:2].copy()
    model = load_predictor_checkpoint(study.SOURCE)
    preprocessing = torch.load(study.SOURCE, map_location="cpu", weights_only=False)["preprocessing"]
    atom, angle = build_model_data(data, study.combined_cache("25g"), pd.DataFrame(), preprocessing["scaler"])
    atom_batch, angle_batch = next(zip(*loader_pair(atom, angle, [0, 1], 2048)))
    model.eval()
    with torch.no_grad():
        regular = model(atom_batch, angle_batch)
        cached = model.head(model.extract_representation(atom_batch, angle_batch))
    torch.testing.assert_close(regular, cached, rtol=0, atol=0)


def test_all_formal_contexts_complete_and_hash_locked():
    completions = list(study.STUDY.glob("*/**/completion.json"))
    assert len(completions) == 120
    for completion_path in completions:
        completion = json.loads(completion_path.read_text())
        for name, digest in completion["files"].items():
            assert sha256_file(completion_path.parent / name) == digest
        usage = json.loads((completion_path.parent / "label_usage.json").read_text())
        assert usage["actual_revealed_rows"] == usage["gradient_train_rows"] + usage["validation_rows"]
        assert usage["test_rows_used_for_fit"] == 0
        assert usage["test_rows_used_for_checkpoint_selection"] == 0
        assert usage["target_rows_used_for_preprocessing_fit"] == 0
        if completion["contract"]["split_protocol"] == "compound":
            assert usage["compound_isolation"] is True
    execution = json.loads((study.STUDY / "execution_audit.json").read_text())
    assert execution["failed"] == 0 and len(execution["contexts"]) == 120
    decision = json.loads((study.STUDY / "decision.json").read_text())
    assert decision["decision"] == "LOW_DIMENSIONAL_COLUMN_CALIBRATION_SUPPORTED"
    assert decision["primary_route"] == "ACTIVE_CALIBRATION"
    assert decision["active_learning_executed"] is False


def test_cross_column_graph_caches_are_protected():
    protected = {item["path"] for item in json.loads((study.ROOT / "docs/PROTECTED_ARTIFACTS.json").read_text())["artifacts"]}
    for column in study.COLUMNS:
        path = study.graph_path(column)
        assert str(path.relative_to(study.ROOT)) in protected
