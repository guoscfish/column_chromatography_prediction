#!/usr/bin/env python3
"""Bounded E4 preregistration preflight; never trains a model."""
from pathlib import Path
import hashlib, json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PART = ROOT / "experiments/d28_al_engineering/partitions"
OUT = ROOT / "experiments/e4_active_transfer_preregistration"
SEEDS = (42, 525, 1101)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def main():
    rows = []
    failures = []
    for protocol, stem in (("A", "e4_8g_protocol_a_row_seed"), ("B", "e4_8g_protocol_b_compound_seed")):
        for seed in SEEDS:
            p = PART / f"{stem}_{seed}.csv"
            df = pd.read_csv(p)
            counts = df.role.value_counts().to_dict()
            if counts.get("l0_train") != 42 or counts.get("l0_validation") != 8:
                failures.append(f"{protocol}/{seed}: L0 must be 42 train + 8 validation")
            test = set(df.loc[df.role == "test", "sample_id"])
            l0 = set(df.loc[df.role.isin(["l0_train", "l0_validation", "l0"]), "sample_id"])
            if test & l0: failures.append(f"{protocol}/{seed}: test-L0 overlap")
            if protocol == "B":
                test_compounds = set(df.loc[df.role == "test", "canonical_smiles"])
                train_compounds = set(df.loc[df.role != "test", "canonical_smiles"])
                if test_compounds & train_compounds: failures.append(f"B/{seed}: compound leakage")
            rows.append({"protocol": protocol, "seed": seed, "path": str(p.relative_to(ROOT)), "rows": len(df), "role_counts": json.dumps(counts, sort_keys=True), "unique_sample_ids": df.sample_id.nunique(), "unique_test_compounds": df.loc[df.role == "test", "canonical_smiles"].nunique(), "sha256": sha256(p), "ok": not failures})
    pd.DataFrame(rows).to_csv(OUT / "partition_audit.csv", index=False)
    source = []
    source_paths = {42: ROOT / "experiments/e0_4g_baseline/checkpoints/best.pt", 525: ROOT / "experiments/e1_signal_qualification/source_members/member_seed_525/checkpoints/best.pt", 1101: ROOT / "experiments/e1_signal_qualification/source_members/member_seed_1101/checkpoints/best.pt"}
    for seed in (42, 525, 1101):
        p = source_paths[seed]
        source.append({"source_seed": seed, "path": str(p.relative_to(ROOT)), "exists": p.exists(), "sha256": sha256(p) if p.exists() else None})
    distinct = len({x["sha256"] for x in source if x["sha256"]}) == 3
    guard = {"L0": 50, "B": 10, "rounds": 15, "K": 3, "budgets": list(range(50, 201, 10)), "formal_training_started": False}
    decision = {"partition_failures": failures, "source_members": source, "source_hashes_distinct": distinct, "budget_guard": guard, "test_isolation_checked": not failures, "formal_training_started": False, "preflight_pass": not failures and distinct}
    (OUT / "preflight_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    if failures or not distinct: raise SystemExit(1)

if __name__ == "__main__": main()
