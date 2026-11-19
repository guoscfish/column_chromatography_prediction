"""Identity and pre-projection gates; all synthetic targets are test fixtures."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / "data_boundary.py"
SPEC = importlib.util.spec_from_file_location("formal_data_boundary", PATH)
db = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(db)


def covariates():
    base = {"canonical_smiles": "CCO", "Density g/ml": 1.0, "V/ul": 30.0,
            "loading solvent": "PE", "Volume of loading solvent/ul": 200, "PE/EA": "10/1"}
    return pd.DataFrame([dict(base, sample_id="a"), dict(base, sample_id="b", **{"V/ul": 150.0}),
                         dict(base, sample_id="c", **{"V/ul": 150.0}),
                         dict(base, sample_id="d", **{"Volume of loading solvent/ul": 300}),
                         dict(base, sample_id="e", **{"PE/EA": "5/1"})])


def test_exact_single_factor_and_replicate_identity():
    pairs, series = db.strict_pairs(covariates())
    volume = pairs[pairs.axis.eq("V/ul")]
    assert len(volume) == 2
    assert {frozenset(x) for x in zip(volume.low_id, volume.high_id)} == {frozenset(("a", "b")), frozenset(("a", "c"))}
    assert volume.coordinate_delta.tolist() == [120.0, 120.0]
    assert volume.series_id.nunique() == 1
    assert len(pairs[pairs.axis.eq("PE/EA")]) == 1
    assert set(pairs.low_id) | set(pairs.high_id) == {"a", "b", "c", "e"}
    assert series[series.axis.eq("V/ul")].iloc[0].n_rows == 3


def test_pair_identity_stable_under_row_order_and_numeric_format():
    a, _ = db.strict_pairs(covariates())
    b, _ = db.strict_pairs(covariates().iloc[::-1])
    assert a.sort_values("pair_id").to_dict("records") == b.sort_values("pair_id").to_dict("records")
    assert db.canonical_value(30) == db.canonical_value(30.0)
    assert db.axis_order("PE/EA", "10/1") < db.axis_order("PE/EA", "5/1")


def test_exposure_and_global_preview_quarantine():
    q = sorted(db.QUARANTINED_IDS)[0]
    pairs = pd.DataFrame({"low_id": ["a", "a", "b", q], "high_id": ["b", "c", "d", "a"]})
    split = pd.DataFrame({"row_seed": [42] * 5, "sample_id": ["a", "b", "c", "d", q],
                          "source_role": ["train", "train", "validation", "test", "train"]})
    result = db.annotate_pairs(pairs, split)
    assert result.exposure.tolist() == ["train_train", "train_validation", "source_test_quarantined", "train_train"]
    assert result.scoring_allowed.tolist() == [True, True, False, False]


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    source = tmp_path / "synthetic.csv"
    source.write_text("sample_id,V1_ml,V2_ml\na,1.25,2.5\nb,3.25,4.5\nt,DO_NOT_PARSE,DO_NOT_PARSE\n")
    q = sorted(db.QUARANTINED_IDS)[0]
    part = pd.DataFrame({"sample_id": ["a", "b", "t", q], "source_role": ["train", "validation", "test", "train"]})
    monkeypatch.setattr(db, "SOURCE", source)
    monkeypatch.setattr(db, "load_metadata", lambda: pd.DataFrame())
    monkeypatch.setattr(db, "load_formal_split", lambda seed, metadata: part)
    monkeypatch.setattr(db, "validate_label_roles", lambda train, validation, test, planned_budget: {
        "gradient_train": train, "validation": validation, "test": test})
    return db.FormalLabelBoundary(42)


def test_official_store_projects_requested_train_and_validation_only(boundary):
    train = boundary.reveal(["a"], "train")
    val = boundary.reveal(["b"], "validation")
    assert np.array_equal(train[["V1_ml", "V2_ml"]], np.array([[1.25, 2.5]], np.float32))
    assert np.array_equal(val[["V1_ml", "V2_ml"]], np.array([[3.25, 4.5]], np.float32))
    assert [receipt["roles"] for receipt in boundary.audit] == ["l0", "validation"]
    assert all(receipt["test_targets_projected"] == 0 for receipt in boundary.audit)


def test_reject_test_unknown_crossrole_and_duplicate_before_store(boundary, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("rejected request reached the official target store")
    monkeypatch.setattr(boundary._store, "reveal", forbidden)
    for ids, role, error in [(["t"], "test", PermissionError), (["t"], "train", PermissionError),
                             (["unknown"], "train", PermissionError), (["b"], "train", PermissionError),
                             (["a", "b"], "train", PermissionError), (["a", "a"], "train", ValueError),
                             ([], "train", ValueError), ([sorted(db.QUARANTINED_IDS)[0]], "train", PermissionError)]:
        with pytest.raises(error):
            boundary.reveal(ids, role)
    assert boundary.audit == []


def test_pair_gate_rejects_entire_request_before_any_projection(boundary, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("mixed authorized/quarantined pair request partially projected targets")
    monkeypatch.setattr(boundary._store, "reveal", forbidden)
    for endpoint in ["t", "unknown", sorted(db.QUARANTINED_IDS)[0]]:
        pairs = pd.DataFrame({"low_id": ["a", "a"], "high_id": ["b", endpoint]})
        with pytest.raises(PermissionError):
            boundary.labels_for_pairs(pairs)


def test_real_formal_split_identity_counts_without_targets():
    metadata = db.load_metadata()
    assert len(metadata) == metadata.sample_id.nunique() == 4163
    assert metadata.global_quarantine.sum() == 3
    assert not {"V1_ml", "V2_ml"} & set(metadata)
    for seed in db.SEEDS:
        split = db.load_formal_split(seed, metadata)
        assert split.source_role.value_counts().to_dict() == db.EXPECTED_COUNTS
        assert split.sample_id.tolist() == metadata.sample_id.tolist()
