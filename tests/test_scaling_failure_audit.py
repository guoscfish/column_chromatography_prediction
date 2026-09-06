import numpy as np
import pandas as pd
import hashlib
import json
from pathlib import Path

from src.qgeognn_al.transfer.scaling_audit import (
    match_key, out_of_fold_calibration, partial_rank, safe_ratio, source_bins,
    standardize_condition_contrast,
)
from src.qgeognn_al.transfer.conditional_scaling import fit_conditional


def test_pairing_is_exact_and_relaxes_only_flow():
    row = {"canonical_smiles": "CCO", "PE/EA": "2/1", "loading solvent": "PE",
           "Density g/ml": .8, "V/ul": 100, "Volume of loading solvent/ul": 300, "Flow mL/min": 10}
    assert match_key(row) == match_key({**row, "PE/EA": "4/2", "V/ul": 100.0})
    other = {**row, "Flow mL/min": 15}
    assert match_key(row) != match_key(other)
    assert match_key(row, relaxed=True) == match_key(other, relaxed=True)
    # Equal density*volume is deliberately not enough to manufacture a pair.
    assert match_key(row, relaxed=True) != match_key({**row, "Density g/ml": .4, "V/ul": 200}, relaxed=True)
    assert match_key(row) != match_key({**row, "Volume of loading solvent/ul": 300.001})


def test_ratio_floor_preserves_errors_and_bins_use_training_thresholds():
    source = np.array([0., .1, .5, 2.])
    ratio = safe_ratio(np.ones(4), source)
    assert np.isnan(ratio[:2]).all()
    np.testing.assert_allclose(ratio[2:], [2., .5])
    _, first = source_bins([1., 2.], np.arange(10.))
    bins, second = source_bins([1e9, -1e9], np.arange(10.))
    np.testing.assert_array_equal(first, second)
    assert bins.tolist() == ["extreme_tail", "low"]


def test_group_cross_fitting_does_not_fit_held_compound_labels():
    source = np.column_stack([np.arange(1., 21.), np.arange(2., 42., 2.)])
    groups = np.repeat(np.arange(5), 4)
    truth = source * 2
    p1, _, folds = out_of_fold_calibration(source, truth, groups)
    changed = truth.copy()
    changed[groups == 0] += 1000
    p2, _, folds2 = out_of_fold_calibration(source, changed, groups)
    np.testing.assert_array_equal(folds, folds2)
    np.testing.assert_allclose(p1[groups == 0], p2[groups == 0])
    for group in np.unique(groups):
        assert len(np.unique(folds[groups == group])) == 1


def test_constant_or_confounded_conditions_are_not_called_zero_effect():
    x = np.linspace(0, 1, 20)
    assert np.isnan(partial_rank(x, x**2, x))
    assert np.isnan(partial_rank(np.ones(20), x, x))
    frame = pd.DataFrame({"ratio": np.arange(1., 9.), "source_bin": ["low"]*8,
                          "canonical_smiles": list("abcdabcd"), "solvent_DCM": [0]*4+[1]*4})
    result = standardize_condition_contrast(frame, "solvent_DCM")
    assert result["common_source_bins"] == 1
    assert np.isnan(result["relative_standardized_contrast"])


def test_conditional_model_changes_slope_not_only_additive_intercept():
    rng = np.random.default_rng(91)
    x = rng.uniform(1., 20., (200, 2))
    ea = rng.uniform(0., 1., 200)
    truth = x*(2+ea[:, None])+3
    fit = fit_conditional(x, truth, ea, [7., 16.], 2., 0.)
    additive = fit_conditional(x, truth, ea, [7., 16.], 2., 0., interaction=False)
    np.testing.assert_allclose(fit.predict(x, ea), truth, atol=1e-10)
    assert np.sqrt(np.square(additive.predict(x, ea)-truth).mean()) > 1.
    test = np.array([[2., 3.], [4., 6.]])
    difference = fit.predict(test, [1., 1.])-fit.predict(test, [0., 0.])
    np.testing.assert_allclose(difference[1], difference[0]*2, atol=1e-10)
    before = fit.audit()
    fit.predict(test*1e5, [0., 1.])
    assert fit.audit() == before


def test_conditional_and_additive_families_recover_same_affine_at_constant_condition():
    x = np.column_stack([np.arange(1., 21.), np.arange(2., 42., 2.)])
    y = x*[2., 3.]+[1., 2.]
    for interaction in (False, True):
        fit = fit_conditional(x, y, np.ones(20)*.5, [7., 16.], 2., 0., interaction=interaction)
        np.testing.assert_allclose(fit.predict(x, np.ones(20)*.5), y, atol=1e-10)


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/transfer/scaling_failure_audit"


def read_json(path):
    return json.loads(path.read_text())


def assert_hashes(base, mapping):
    for name, expected in mapping.items():
        assert hashlib.sha256((base / name).read_bytes()).hexdigest() == expected, name


def test_completed_artifacts_preserve_preregistration_and_prediction_freezes():
    protocol = read_json(STUDY / "protocol.json")
    # Runtime checkpoints may be absent in a fresh clone; all available inputs
    # and every tracked contract still have to match their original hashes.
    assert_hashes(ROOT, {k: v for k, v in protocol["inputs"].items() if (ROOT / k).exists()})
    assert_hashes(ROOT, protocol["audit_files"])
    assert_hashes(STUDY, read_json(STUDY / "training_audit_frozen.json")["files"])
    model = read_json(STUDY / "model_protocol.json")
    assert model["selected_directions"] == ["CONDITIONAL_SCALING"]
    assert not model["test_tuning"]
    assert_hashes(ROOT, model["model_hashes"])
    frozen = read_json(STUDY / "model_predictions_frozen.json")
    assert len(frozen["files"]) == frozen["contexts"] == 120
    assert_hashes(STUDY, frozen["files"])
    for name in frozen["files"]:
        path = STUDY / name
        assert_hashes(path.parent, read_json(path)["files"])


def test_completed_models_use_exact_frozen_roles_and_equal_label_budgets():
    schedule = pd.read_csv(ROOT / "studies/transfer/cross_column/splits/schedule_manifest.csv")
    groups = schedule.groupby(["column", "protocol", "outer_seed", "planned_budget"])
    for path in sorted((STUDY / "models").glob("*/*/seed_*/budget_*/label_usage.json")):
        usage = read_json(path)
        context = groups.get_group((usage["column"], usage["protocol"], usage["seed"], usage["budget"]))
        roles = ("gradient_train", "validation", "test")
        for role in roles:
            assert usage[role] == sorted(context.loc[context.role.eq(role), "sample_id"])
        train, valid, test = (set(usage[role]) for role in roles)
        assert not (train & valid or train & test or valid & test)
        assert len(train) + len(valid) == usage["actual_budget"]
        for field in ("donor_target_labels_used", "paired_source_label_features_used", "test_labels_used_for_fit_or_selection"):
            assert usage[field] == 0
        predictions = pd.read_csv(path.parent / "predictions_blind.csv.gz")
        assert predictions.sample_id.tolist() == usage["test"]
        assert np.isfinite(predictions.drop(columns="sample_id").to_numpy()).all()
    metrics = pd.read_csv(STUDY / "model_all_metrics.csv")
    assert len(metrics) == 960
    assert metrics.groupby(["column", "protocol", "seed", "budget"]).size().eq(8).all()
    assert np.isfinite(metrics.filter(regex="r2|rmse|mae").to_numpy()).all()


def test_pair_anchor_truth_is_restricted_to_source_training_rows():
    split = pd.read_csv(ROOT / "studies/predictor/final_4g_qualification/splits/row_seed_42.csv")
    source_train = set(split.loc[split.split.eq("train"), "sample_id"])
    pairs = pd.read_csv(STUDY / "pair_identity_audit.csv")
    assert len(pairs) == 574 + 490 + 529
    for row in pairs.to_dict("records"):
        for kind in ("exact", "relaxed"):
            all_ids = set(json.loads(row[f"{kind}_source_ids"]))
            eligible = set(json.loads(row[f"{kind}_source_train_ids"]))
            assert eligible == all_ids & source_train
            assert len(all_ids) == row[f"{kind}_matches"]
            assert len(eligible) == row[f"{kind}_train_matches"]
            if not eligible:
                assert np.isnan(row[f"{kind}_source_train_V1"])
                assert np.isnan(row[f"{kind}_source_train_V2"])
        assert set(json.loads(row["exact_source_ids"])) <= set(json.loads(row["relaxed_source_ids"]))
