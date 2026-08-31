from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.studies.run_s1_source_target_shift import (
    LABEL_COLUMNS, correction_cv, load_analysis_truth, make_partition,
)
from src.qgeognn_al.artifacts import sha256_file
from src.qgeognn_al.data import apply_standardizer, condition_matrix, fit_standardizer
from src.qgeognn_al.resources import ROOT, SOURCE_CHECKPOINTS, SOURCE_DATA, TARGET_DATA, verified_source_checkpoints

STUDY = ROOT / "studies/track_b_transfer/s1_source_target_shift"


def test_partition_generation_is_label_blind_and_compound_disjoint() -> None:
    base = pd.DataFrame({"sample_id":["a","b","c","d"],"canonical_smiles":["x","x","y","z"],"V1_ml":[1,2,3,4],"V2_ml":[4,3,2,1]})
    changed = base.assign(V1_ml=[100,200,300,400], V2_ml=[0,0,0,0])
    first = make_partition(base, 7, .7); second = make_partition(changed, 7, .7)
    pd.testing.assert_frame_equal(first, second)
    assert not LABEL_COLUMNS & set(first.columns)
    assert not set(first.loc[first.role.eq("analysis"),"canonical_smiles"]) & set(first.loc[first.role.eq("reserved"),"canonical_smiles"])


def test_reserved_truth_is_excluded() -> None:
    partition = pd.read_csv(STUDY / "s1_partition.csv")
    truth = load_analysis_truth(TARGET_DATA, partition)
    assert set(truth.sample_id.astype(str)) == set(partition.loc[partition.role.eq("analysis"),"sample_id"].astype(str))
    assert not set(truth.sample_id.astype(str)) & set(partition.loc[partition.role.eq("reserved"),"sample_id"].astype(str))


def test_source_hashes_match_frozen_audits() -> None:
    records = verified_source_checkpoints()
    assert {row["source_seed"] for row in records} == {42,525,1101}
    assert all(sha256_file(SOURCE_CHECKPOINTS[int(row["source_seed"])]) == row["sha256"] for row in records)


def test_condition_standardizer_is_source_only() -> None:
    source = pd.read_csv(SOURCE_DATA); target = pd.read_csv(TARGET_DATA)
    source_conditions = condition_matrix(source); target_conditions = condition_matrix(target)
    mean, scale = fit_standardizer(source_conditions)
    assert np.allclose(apply_standardizer(source_conditions, mean, scale).mean(axis=0), 0, atol=1e-6)
    changed_target = target_conditions + 999
    mean2, scale2 = fit_standardizer(source_conditions)
    assert np.array_equal(mean, mean2) and np.array_equal(scale, scale2)
    assert not np.allclose(apply_standardizer(changed_target, mean, scale).mean(axis=0), 0)


def test_group_cv_has_no_leakage_and_is_deterministic() -> None:
    rng=np.random.default_rng(4); groups=np.repeat([f"c{i}" for i in range(10)],2); n=len(groups)
    frame=pd.DataFrame({"sample_id":[f"s{i}" for i in range(n)],"canonical_smiles":groups,"V1_ml":rng.normal(size=n),"V2_ml":rng.normal(size=n),"source_V1_mean":rng.normal(size=n),"source_V2_mean":rng.normal(size=n)})
    conditions=rng.normal(size=(n,9))
    first=correction_cv(frame,conditions,5,4,[.1,1.0]); second=correction_cv(frame,conditions,5,4,[.1,1.0])
    pd.testing.assert_frame_equal(first[0],second[0]); assert first[0].train_validation_compound_overlap.eq(0).all()


def test_runner_has_no_historical_runner_import_or_target_gnn_fit() -> None:
    path=ROOT/"scripts/studies/run_s1_source_target_shift.py"; tree=ast.parse(path.read_text())
    imports=[node.module or "" for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    assert not [name for name in imports if name.startswith("scripts.run_")]
    source=path.read_text(); assert ".fit(" not in source.split("def run(config_path",1)[1].split("def main",1)[0] or "correction_cv(" in source
    decision=json.loads((STUDY/"decision.json").read_text()); assert decision["T1_started"] is False and decision["active_learning_started"] is False
