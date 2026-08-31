from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.studies.run_a1a_hybrid_batch_control import (
    FORBIDDEN_OLD_SEEDS, OUTER_SEEDS, control_seed, mechanism_gate,
    random_control_batches,
)
from src.qgeognn_al.partitions import make_row_partition
from src.qgeognn_al.resources import ROOT, SOURCE_DATA

STUDY=ROOT/"studies/track_a_4g_al/a1a_hybrid_batch_control"


def test_exact_new_outer_seeds() -> None:
    assert OUTER_SEEDS==(17,137,941,2027,4099)
    assert not set(OUTER_SEEDS)&FORBIDDEN_OLD_SEEDS


def test_partition_generation_truth_blind_and_authoritative_counts() -> None:
    full=pd.read_csv(SOURCE_DATA); full["canonical_index"]=np.arange(len(full)); identities=full[["sample_id","canonical_index","canonical_smiles"]]
    changed=full.assign(V1_ml=999,V2_ml=-999)
    for seed in OUTER_SEEDS:
        expected=make_row_partition(identities,seed,.1,.1,.15,"A1a","row")
        changed_partition=make_row_partition(changed,seed,.1,.1,.15,"A1a","row")
        pd.testing.assert_frame_equal(expected,changed_partition)
        actual=pd.read_csv(STUDY/"partitions"/f"a1a_row_seed_{seed}.csv")
        pd.testing.assert_frame_equal(expected,actual)
        assert actual.role.value_counts().to_dict()=={"u0":3371,"test":417,"l0_train":318,"l0_validation":57}
        role_sets=[set(actual.loc[actual.role.eq(role),"sample_id"]) for role in ("u0","test","l0_train","l0_validation")]
        assert all(not role_sets[i]&role_sets[j] for i in range(4) for j in range(i+1,4))


def test_partition_rule_regresses_to_authoritative_e2() -> None:
    identities=pd.read_csv(SOURCE_DATA); identities["canonical_index"]=np.arange(len(identities)); identities=identities[["sample_id","canonical_index","canonical_smiles"]]
    for seed in (42,525,1101):
        expected=pd.read_csv(ROOT/f"experiments/d28_al_engineering/partitions/e2_4g_row_seed_{seed}.csv")
        actual=make_row_partition(identities,seed,.1,.1,.15,"e2_4g","row")
        pd.testing.assert_frame_equal(actual,expected)


def test_random_controls_are_deterministic_unique_and_shortlist_only() -> None:
    shortlist=[f"s{i}" for i in range(100)]
    first=pd.DataFrame(random_control_batches(shortlist,25,17,10)); second=pd.DataFrame(random_control_batches(shortlist,25,17,10))
    pd.testing.assert_frame_equal(first,second)
    assert first.groupby("control_draw_index").sample_id.nunique().eq(25).all()
    assert set(first.sample_id)<=set(shortlist)
    assert all(control_seed(17,i)!=control_seed(17,j) for i in range(10) for j in range(i+1,10))


def test_mechanism_gate_is_deterministic_and_does_not_launch_a1b() -> None:
    frame=pd.DataFrame({"outer_seed":OUTER_SEEDS,"hybrid_gain":[2,2,2,2,0],"random_gain_mean":[1]*5,"random_gain_median":[1]*5,"hybrid_beats_random_count":[9,9,9,9,0]})
    first=mechanism_gate(frame); second=mechanism_gate(frame.copy())
    assert first==second and first["diversity_mechanism_supported"] is True
    assert first["A1b_automatic_launch"] is False


def test_runner_import_boundary_and_no_s1_coupling() -> None:
    path=ROOT/"scripts/studies/run_a1a_hybrid_batch_control.py"; tree=ast.parse(path.read_text())
    imports=[node.module or "" for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    assert not [name for name in imports if name.startswith("scripts.run_")]
    assert "s1_source_target_shift" not in path.read_text().lower()


def test_completed_outputs_contract_if_present() -> None:
    summary_path=STUDY/"seed_summary.csv"
    if not summary_path.exists(): return
    summary=pd.read_csv(summary_path); assert set(summary.outer_seed)==set(OUTER_SEEDS)
    shortlist=pd.read_csv(STUDY/"round0_shortlists.csv"); hybrid=pd.read_csv(STUDY/"hybrid_batches.csv"); controls=pd.read_csv(STUDY/"random_control_batches.csv")
    for seed in OUTER_SEEDS:
        pool=set(shortlist.loc[shortlist.outer_seed.eq(seed),"sample_id"]); h=hybrid[hybrid.outer_seed.eq(seed)]; c=controls[controls.outer_seed.eq(seed)]
        assert len(h)==25 and h.sample_id.nunique()==25 and set(h.sample_id)<=pool
        assert c.groupby("control_draw_index").size().eq(25).all() and set(c.sample_id)<=pool
    audit=pd.read_csv(STUDY/"initialization_hash_audit.csv")
    assert audit.groupby(["outer_seed","member_index"]).initial_parameter_hash.nunique().eq(1).all()
    assert audit.groupby("outer_seed").validation_ids_hash.nunique().eq(1).all()
    assert audit.groupby("outer_seed").scaler_hash.nunique().eq(1).all()
    assert not audit.test_used_for_checkpoint_selection.any()
    decision=json.loads((STUDY/"decision.json").read_text()); assert decision["A1b_automatic_launch"] is False and decision["S1_used_for_acquisition"] is False
