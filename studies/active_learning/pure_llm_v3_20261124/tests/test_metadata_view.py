from collections import defaultdict
from pathlib import Path

import pytest

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3 import metadata_view as mv


ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def metadata():
    return mv.load_4g_metadata(ROOT)


def test_metadata_loader_uses_explicit_label_free_projections(monkeypatch):
    calls = []
    read_csv = mv.pd.read_csv

    def watched_read_csv(path, *, usecols, **kwargs):
        calls.append((Path(path).name, tuple(usecols)))
        assert not set(usecols) & {"t1", "t2", "verified t1", "verified t2", "V1_ml", "V2_ml",
                                   "true_V1_ml", "true_V2_ml", "quality_flags", "decision_reason"}
        return read_csv(path, usecols=usecols, **kwargs)

    monkeypatch.setattr(mv.pd, "read_csv", watched_read_csv)
    records = mv.load_4g_metadata(ROOT)
    assert calls == [("dataset_4g.csv", mv.RAW_COLUMNS), ("canonical_4g.csv", mv.CANONICAL_COLUMNS)]
    assert len(records) == len({row.source_row_1based for row in records.values()}) == 4163
    assert len({row.canonical_smiles for row in records.values()}) == 217
    assert "34.01360544" in {row.public_conditions()["sample_volume"] for row in records.values()}


def test_all_frozen_roles_covered_in_formal_seeds(metadata):
    for seed in mv.CONFIRMATION_SEEDS:
        roles = mv.load_frozen_split(ROOT, seed, metadata)
        assert {role: len(ids) for role, ids in roles.items()} == mv.ROW_COUNTS
        assert len({identity for ids in roles.values() for identity in ids}) == len(metadata)
        assert set(roles["l0"] + roles["u0"]) <= metadata.keys()


def test_purity_raw_identity_and_role_safe_public_fields(metadata):
    assert mv.purity_value("AR") == ("grade", None)
    assert mv.purity_value("") == mv.purity_value("N/A") == ("missing", None)
    assert mv.purity_value("98") == ("ambiguous_numeric", None)
    assert mv.purity_value("98%") == ("numeric_percent", 0.98)
    assert mv.purity_value("0.9800") == ("numeric_fraction", 0.98)
    assert mv.ea_fraction("20/1") < mv.ea_fraction("10/1")
    by_structure = defaultdict(set)
    for row in metadata.values():
        by_structure[row.canonical_smiles].add(row.cas)
        public = row.public_conditions()
        assert tuple(public) == mv.PUBLIC_CONDITIONS
        assert public["purity_raw"] == row.purity_raw and public["flow"] == row.flow
        if row.purity_raw == "AR":
            assert public["purity_fraction"] is None
    assert sum(len(reagents) > 1 for reagents in by_structure.values()) == 3


def test_full_condition_identity_splits_legacy_matches_and_preserves_repeats(metadata):
    legacy, complete = defaultdict(list), defaultdict(list)
    for row in metadata.values():
        legacy[row.identity_key(mv.LEGACY_AXES)].append(row)
        complete[row.identity_key()].append(row)
    differently_flowing = next(group for group in legacy.values()
                               if len({row.flow for row in group}) > 1)
    left = differently_flowing[0]
    right = next(row for row in differently_flowing if row.flow != left.flow)
    assert not mv.condition_changes(left.public_conditions(), right.public_conditions(), mv.LEGACY_AXES)
    assert "flow" in mv.condition_changes(left.public_conditions(), right.public_conditions())
    repeat_groups = [group for group in complete.values() if len(group) > 1]
    assert len(repeat_groups) == 31
    assert all(group[0].sample_id != group[1].sample_id for group in repeat_groups)
    left = metadata["0ccbd28a4414f97dfad8"].public_conditions()
    right = metadata["1a55639d058c08a1a06e"].public_conditions()
    assert mv.condition_changes(left, right, mv.LEGACY_AXES) == ("PE/EA",)
    assert mv.condition_changes(left, right) == ("PE/EA", "flow")
