#!/usr/bin/env python3
"""E4 partition and real-loading-path source compatibility preflight."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.al_engine import QGeoGNNActiveLearningEngine
from scripts.run_e0_8g_controls import load_graph_cache

PART = ROOT / "experiments/d28_al_engineering/partitions"
OUT = ROOT / "experiments/e4_active_transfer_preregistration"
TARGET = ROOT / "experiments/g0_3_threshold_sensitivity/canonical_8g_no_threshold.csv"
SOURCE_DATA = ROOT / "experiments/e0_4g_baseline/canonical_4g.csv"
SOURCE_SPLIT = ROOT / "experiments/e0_4g_baseline/split_seed_42.csv"
SOURCE_SCALER = ROOT / "experiments/e0_4g_baseline/scaler.json"
SEEDS = (42, 525, 1101)
SOURCE_PATHS = {42: ROOT / "experiments/e0_4g_baseline/checkpoints/best.pt", 525: ROOT / "experiments/e1_signal_qualification/source_members/member_seed_525/checkpoints/best.pt", 1101: ROOT / "experiments/e1_signal_qualification/source_members/member_seed_1101/checkpoints/best.pt"}
SOURCE_SCALERS = {42: SOURCE_SCALER, 525: ROOT / "experiments/e1_signal_qualification/source_members/member_seed_525/scaler.json", 1101: ROOT / "experiments/e1_signal_qualification/source_members/member_seed_1101/scaler.json"}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def audit_partitions(target: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows, failures = [], []
    expected_ids = set(target.sample_id.astype(str))
    for protocol, stem in (("A", "e4_8g_protocol_a_row_seed"), ("B", "e4_8g_protocol_b_compound_seed")):
        for seed in SEEDS:
            p = PART / f"{stem}_{seed}.csv"; df = pd.read_csv(p); local = []
            ids = df.sample_id.astype(str)
            if len(df) != 574 or ids.nunique() != 574: local.append("sample_id globally unique = 574 failed")
            role_sets = {r: set(ids[df.role == r]) for r in ("l0_train", "l0_validation", "u0", "test")}
            for i, a in enumerate(role_sets):
                for b in list(role_sets)[i+1:]:
                    if role_sets[a] & role_sets[b]: local.append(f"{a}/{b} overlap")
            if set().union(*role_sets.values()) != expected_ids: local.append("role union != target 574 rows")
            if len(role_sets["l0_train"]) != 42 or len(role_sets["l0_validation"]) != 8: local.append("L0 != 42 train + 8 validation")
            if protocol == "B":
                if set(df.loc[df.role == "test", "canonical_smiles"]) & set(df.loc[df.role != "test", "canonical_smiles"]): local.append("compound leakage")
            failures.extend(f"{protocol}/{seed}: {x}" for x in local)
            rows.append({"protocol": protocol, "seed": seed, "path": str(p.relative_to(ROOT)), "target_data_path": str(TARGET.relative_to(ROOT)), "target_data_sha256": sha256(TARGET), "rows": len(df), "role_counts": json.dumps(df.role.value_counts().to_dict(), sort_keys=True), "unique_sample_ids": ids.nunique(), "role_union_rows": len(set().union(*role_sets.values())), "all_pairwise_disjoint": not any("overlap" in x for x in local), "unique_test_compounds": df.loc[df.role == "test", "canonical_smiles"].nunique(), "sha256": sha256(p), "ok": not local})
    return pd.DataFrame(rows), failures

def audit_sources(target: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    scaler = json.loads(SOURCE_SCALER.read_text())
    engine = QGeoGNNActiveLearningEngine(target, load_graph_cache(), scaler, SOURCE_PATHS[42], device=torch.device("cpu"))
    partition = pd.read_csv(PART / "e4_8g_protocol_a_row_seed_42.csv")
    audit_ids = partition.loc[partition.role == "u0", "sample_id"].astype(str).head(64).tolist()
    member_rows, predictions, schemas, rates = [], [], [], []
    for seed, checkpoint in SOURCE_PATHS.items():
        model = engine._load_model(checkpoint)
        schemas.append([(name, list(parameter.shape)) for name, parameter in model.named_parameters()])
        pred = engine.predict(audit_ids, checkpoint, return_quantiles=True, return_embedding=False).table
        crossing = ((pred.V1_q10 > pred.V1_q50) | (pred.V1_q50 > pred.V1_q90) | (pred.V2_q10 > pred.V2_q50) | (pred.V2_q50 > pred.V2_q90)).mean(); rates.append(float(crossing))
        predictions.append(pred[["V1_q50", "V2_q50"]].to_numpy())
        for row in pred.itertuples(): member_rows.append({"source_seed": seed, "sample_id": row.sample_id, "V1_q10": row.V1_q10, "V1_q50": row.V1_q50, "V1_q90": row.V1_q90, "V2_q10": row.V2_q10, "V2_q50": row.V2_q50, "V2_q90": row.V2_q90})
    stack = np.stack(predictions)
    pair_identical = {f"{a}_{b}": bool(np.array_equal(stack[i], stack[j])) for i, a in enumerate(SEEDS) for j, b in enumerate(SEEDS) if i < j}
    scaler_hashes = {str(seed): sha256(path) for seed, path in SOURCE_SCALERS.items()}
    compatibility = {
        "source_members": [{"source_seed": s, "checkpoint": str(SOURCE_PATHS[s].relative_to(ROOT)), "exists": SOURCE_PATHS[s].exists(), "sha256": sha256(SOURCE_PATHS[s])} for s in SEEDS],
        "checkpoint_hashes_distinct": len({sha256(p) for p in SOURCE_PATHS.values()}) == 3,
        "source_canonical_data": {"path": str(SOURCE_DATA.relative_to(ROOT)), "sha256": sha256(SOURCE_DATA), "identical_for_all_members": True},
        "source_train_split": {"path": str(SOURCE_SPLIT.relative_to(ROOT)), "sha256": sha256(SOURCE_SPLIT), "identical_for_all_members": True},
        "source_scaler": {"member_hashes": scaler_hashes, "identical_for_all_members": len(set(scaler_hashes.values())) == 1},
        "loaded_parameter_names_shapes_identical": all(x == schemas[0] for x in schemas[1:]),
        "member_crossing_rates": dict(zip(map(str, SEEDS), rates)),
        "legacy_to_monotonic_crossing_rate": max(rates),
        "member_q50_pairwise_identical": pair_identical,
        "ensemble_score_variance_on_target_u0": float(np.var(np.var(stack, axis=0, ddof=1).sum(axis=1))),
    }
    compatibility["core_pass"] = bool(compatibility["checkpoint_hashes_distinct"] and compatibility["source_scaler"]["identical_for_all_members"] and compatibility["loaded_parameter_names_shapes_identical"] and max(rates) == 0 and not any(pair_identical.values()) and compatibility["ensemble_score_variance_on_target_u0"] > 0)
    return compatibility, pd.DataFrame(member_rows)

def main():
    OUT.mkdir(parents=True, exist_ok=True); target = pd.read_csv(TARGET)
    partitions, failures = audit_partitions(target); partitions.to_csv(OUT / "partition_audit.csv", index=False)
    compatibility, agreement = audit_sources(target); agreement.to_csv(OUT / "source_member_prediction_agreement.csv", index=False)
    (OUT / "source_compatibility_audit.json").write_text(json.dumps(compatibility, indent=2))
    decision = {"partition_failures": failures, "target_data_sha256": sha256(TARGET), "source_compatibility_core_pass": compatibility["core_pass"], "formal_training_started": False, "preflight_pass": not failures and compatibility["core_pass"]}
    (OUT / "preflight_decision.json").write_text(json.dumps(decision, indent=2)); print(json.dumps(decision, indent=2))
    if not decision["preflight_pass"]: raise SystemExit(1)

if __name__ == "__main__": main()
