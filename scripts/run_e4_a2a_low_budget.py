#!/usr/bin/env python3
"""Deterministic E4-A2a partitioning and bounded engineering gate.

This entry point deliberately owns only the new low-budget design.  Model and
acquisition implementations remain imported from the frozen E4 runner.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "experiments/d28_al_engineering/partitions"
OUT = ROOT / "experiments/e4_a2a_low_budget_preregistration/partitions"
SEEDS = (42, 525, 1101)
EXPECTED = {
    "old_l0_train_count": 42,
    "new_l0_train_count": 22,
    "validation_count": 8,
    "new_u0_count": 486,
    "test_count": 58,
    "union_rows": 574,
}

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()

def make_partition(seed: int) -> dict:
    old_path=OLD/f"e4_8g_protocol_a_row_seed_{seed}.csv"
    old=pd.read_csv(old_path, dtype={"sample_id":str})
    if len(old)!=574 or old.sample_id.nunique()!=574: raise ValueError("old partition contract failed")
    train=old.loc[old.role=="l0_train","sample_id"].tolist()
    ranked=sorted(train,key=lambda sid:hashlib.sha256(f"E4A2A|{seed}|{sid}".encode()).hexdigest())
    keep=set(ranked[:22]); new=old.copy()
    new.loc[(new.role=="l0_train") & (~new.sample_id.isin(keep)),"role"]="u0"
    new.loc[(new.role=="l0_train") & (new.sample_id.isin(keep)),"role"]="l0_train"
    new["stage"]="e4_a2a"; new["protocol"]="protocol_a_low_budget"
    OUT.mkdir(parents=True,exist_ok=True); path=OUT/f"e4_a2a_protocol_a_seed_{seed}.csv"; new.to_csv(path,index=False)
    roles={r:set(new.loc[new.role==r,"sample_id"]) for r in ("l0_train","l0_validation","u0","test")}
    old_roles={r:set(old.loc[old.role==r,"sample_id"]) for r in ("l0_train","l0_validation","u0","test")}
    demoted=old_roles["l0_train"]-roles["l0_train"]
    audit={"outer_seed":seed,"old_partition_sha256":sha256(old_path),"new_partition_sha256":sha256(path),
           "old_l0_train_count":len(old_roles["l0_train"]),"new_l0_train_count":len(roles["l0_train"]),
           "validation_count":len(roles["l0_validation"]),"test_count":len(roles["test"]),"new_u0_count":len(roles["u0"]),
           "union_rows":sum(map(len,roles.values())),"sample_ids_unique":new.sample_id.nunique()==574,
           "nested_l0_train":roles["l0_train"] <= old_roles["l0_train"],
           "validation_identical":roles["l0_validation"]==old_roles["l0_validation"],"test_identical":roles["test"]==old_roles["test"],
           "all_pairwise_roles_disjoint":all(not (roles[a]&roles[b]) for i,a in enumerate(roles) for b in list(roles)[i+1:]),
           "demoted_count":len(demoted),"demoted_all_in_new_u0":demoted <= roles["u0"],
           "old_u0_subset_new_u0":old_roles["u0"] <= roles["u0"],
           "new_u0_exact_union_pass":roles["u0"] == old_roles["u0"] | demoted}
    audit["exact_count_contract_pass"] = all(audit[k] == v for k,v in EXPECTED.items())
    audit["partition_audit_pass"] = all(bool(audit[k]) for k in ("exact_count_contract_pass","nested_l0_train","validation_identical","test_identical","all_pairwise_roles_disjoint","sample_ids_unique","demoted_all_in_new_u0","old_u0_subset_new_u0","new_u0_exact_union_pass"))
    return audit

def main():
    p=argparse.ArgumentParser(description="Generate and audit frozen E4-A2a partitions."); p.parse_args()
    audits=[make_partition(s) for s in SEEDS]
    (OUT.parent/"partition_audit.json").write_text(json.dumps({"partition_audit_pass":all(a["partition_audit_pass"] for a in audits),"audits":audits},indent=2))
    pd.DataFrame(audits).to_csv(OUT.parent/"partition_manifest.csv",index=False)
    print(json.dumps(audits,indent=2))
if __name__=="__main__": main()
