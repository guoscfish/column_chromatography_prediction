"""Read-only final verification of real receipts; never fit or reveal labels."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

from study_support import HERE, ROOT, read, digest, dump, partition, verify_freeze, ids_hash
import numpy as np
import pandas as pd


def main():
    verify_freeze()
    config = read(HERE / "config.json")
    test = read(HERE / "test_receipt.json")
    assert test["xml_sha256"] == digest(HERE / "preflight_tests.xml")
    suites = list(ET.parse(HERE / "preflight_tests.xml").getroot().iter("testsuite"))
    assert sum(int(s.get("tests",0)) for s in suites) == 69
    assert sum(int(s.get("errors",0))+int(s.get("failures",0)) for s in suites) == 0
    assert re.search(r"^40 passed(?:,| in)", (HERE/"legacy_regression_tests.log").read_text(), re.MULTILINE)
    qualification_count = 0
    fold_count = 0
    acquisition_count = 0
    files = {}
    def protect(path):
        files[str(Path(path).relative_to(HERE))] = digest(path)
    for seed in config["seeds"]:
        split = partition(seed)
        roles = {role: set(split.loc[split.role.eq(role), "sample_id"].astype(str))
                 for role in ("l0", "u0", "validation", "test")}
        access_path = HERE/f"label_access/qualification_seed_{seed}.json"
        access = read(access_path)
        assert access["seed"]==seed and access["training_rows"]==333
        assert access["validation_labels_read"]==access["future_U_labels_read"]==access["test_labels_read"]==0
        assert access["training_ids_hash"]==ids_hash(roles["l0"]) and len(access["calls"])==1
        call = access["calls"][0]
        assert call["purpose"]=="initial_fit" and call["roles"]=="l0" and call["requested_rows"]==333
        assert call["requested_ids_hash"]==ids_hash(roles["l0"])
        protect(access_path)
        outer = split.loc[split.role.isin(["l0","u0"])]
        local_u = set(np.flatnonzero(outer.role.eq("u0").to_numpy()))
        for family in config["probabilistic_families"]:
            directory = HERE / f"qualification/seed_{seed}_{family}"
            receipt = read(directory / "complete.json")
            assert receipt["freeze_sha256"] == digest(HERE / "freeze.json")
            for name, expected in receipt["files"].items():
                assert digest(directory/name) == expected
                protect(directory/name)
            m = read(directory / "metrics.json")
            held = []
            for fold in m["folds"]:
                assert not set(fold["train_positions"]) & set(fold["held_out_positions"])
                assert set(fold["train_positions"]) | set(fold["held_out_positions"]) == set(range(333))
                assert fold["minimum_predictive_eigenvalue"] > 0
                held.extend(fold["held_out_positions"])
                fold_count += 1
            assert sorted(held) == list(range(333))
            with np.load(directory/"oof.npz",allow_pickle=False) as data:
                assert set(data["sample_ids"]) == roles["l0"]
                assert data["oof_mean_ml"].shape == (333,2)
            protect(directory / "complete.json")
            qualification_count += 1
        selected_family = read(HERE/"qualification/decision.json")["selected_family_by_seed"][str(seed)]
        gp_families = [selected_family] if selected_family and selected_family.startswith("gp_") else ["gp_rbf","gp_matern32"]
        settings = [(f,p,m) for f in gp_families for p in ("P1","P2") for m in (256,1024,2048)]
        settings += [(f,"P1",1024) for f in ("blr_independent","blr_coregionalized")]
        for family, distribution, size in settings:
            directory = HERE/f"acquisition/seed_{seed}_{family}_{distribution}_M{size}"
            assert (directory/"complete.json").exists(), f"incomplete acquisition state: {directory.name}"
            receipt = read(directory/"complete.json")
            assert receipt["freeze_sha256"] == digest(HERE / "freeze.json")
            for name, expected in receipt["files"].items():
                assert digest(directory/name) == expected
                protect(directory/name)
            scores = pd.read_csv(directory/"scores.csv",dtype={"sample_id":str})
            assert len(scores)==2997 and scores.sample_id.is_unique and set(scores.sample_id)==roles["u0"]
            assert np.isfinite(scores.drop(columns=["sample_id"]).to_numpy(float)).all()
            for method, result in read(directory/"batches.json").items():
                chosen = result["selected_row_ids"]
                assert len(chosen)==len(set(chosen))==32 and set(chosen)<=roles["u0"]
                assert not set(chosen)&(roles["l0"]|roles["validation"]|roles["test"])
                assert result["audit"]["target_label_semantics"]=="independent_future_replicate"
                assert len(result["trace"])==32
            summary = read(directory/"summary.json")
            assert (summary["seed"],summary["family"],summary["distribution"],summary["M"])==(seed,family,distribution,size)
            assert summary["labels_read_for_acquisition"]==0 and summary["new_gnn_fits"]==0
            refs = set(summary["reference_positions"])
            assert len(refs)==size and refs<=set(range(3330))
            if distribution=="P2":
                assert refs<=local_u
            protect(directory/"complete.json")
            acquisition_count += 1
    assert qualification_count==20 and fold_count==60
    assert read(HERE/"acquisition/status.json")["status"]=="COMPLETE"
    assert acquisition_count==58
    attempts = list((HERE/"qualification").glob("seed_*/fold_cache/attempt_*.json"))
    assert len(attempts)==60
    assert len(read(HERE/"qualification/compute_budget.json")["attempts"])==60
    assert not list((HERE/"runtime").glob("seed_*/posterior_*/round_*/model/best.pt"))
    assert read(HERE/"formal_status.json")["new_fit_attempts"]==0
    stage2=HERE/"stage2_nested_calibration"
    sys.path.insert(0,str(stage2))
    import run_stage2
    run_stage2.verify()
    v2tests=read(stage2/"test_receipt.json")
    assert v2tests["exit_code"]==0 and v2tests["tests"]==17
    assert v2tests["xml_sha256"]==digest(stage2/"preflight_tests.xml")
    assert len(read(stage2/"compute_budget.json")["attempts"])==240
    assert len(list((stage2/"nested_cache").rglob("attempt_*.json")))==240
    stage2_candidates=0
    for receipt_path in (stage2/"qualification").glob("seed_*/complete.json"):
        receipt=read(receipt_path)
        assert receipt["freeze_sha256"]==digest(stage2/"freeze.json")
        for name,h in receipt["files"].items():
            assert digest(receipt_path.parent/name)==h
            protect(receipt_path.parent/name)
        protect(receipt_path)
        stage2_candidates+=1
    assert stage2_candidates==60
    assert not read(stage2/"qualification/decision.json")["run_allowed"]
    assert read(stage2/"formal_status.json")["new_neural_fit_attempts"]==0
    assert not (stage2/"runtime").exists()
    for path in (stage2/"freeze.json",stage2/"config.json",stage2/"qualification/decision.json",
                 stage2/"formal_status.json",stage2/"compute_budget.json",stage2/"test_receipt.json",
                 stage2/"preflight_tests.xml"):
        protect(path)
    for path in (HERE/"freeze.json",HERE/"config.json",HERE/"qualification/decision.json",HERE/"formal_status.json",
                 HERE/"acquisition/status.json",HERE/"acquisition/summary.csv",HERE/"acquisition/reference_sensitivity.csv",
                 HERE/"test_receipt.json",HERE/"preflight_tests.xml",HERE/"legacy_regression_tests.log",Path(__file__)):
        protect(path)
    result = {"status":"PASS", "science_freeze_sha256":digest(HERE/"freeze.json"),
              "new_tests_passed":86,"stage1_tests_passed":69,"stage2_tests_passed":17,"legacy_regression_tests_passed":40,
              "qualification_seed_family_states":qualification_count,"qualification_completed_fits":fold_count,
              "qualification_fit_attempts":len(attempts),"acquisition_completed_states":acquisition_count,
              "stage2_nested_fits":240,"stage2_candidates_checked":stage2_candidates,"total_qualification_fits":300,
              "batches_checked":acquisition_count*3,"unique_candidates_per_state":2997,"batch_size":32,
              "future_U_labels_revealed":0,"surrogate_validation_labels":0,"test_labels_read":0,
              "new_QGeoGNN_fits":0,"new_paid_LLM_API_calls":0,
              "formal_efficacy":"NOT_EVALUATED; qualification hold is not an efficacy NO-GO",
              "files":files}
    dump(HERE/"VERIFICATION.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="files"},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
