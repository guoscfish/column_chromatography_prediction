#!/usr/bin/env python3
"""Build a run inventory from saved validation summaries and fit metadata only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = "studies/active_learning/comprehensive_history_diagnostics_20261112"
PAIRED = "studies/active_learning/nocw_vs_cw_lcmd_five_to1485_20261112"
FINAL = PAIRED + "/reports/final_five_seed_L1485_20261116"
NEW = "studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101"
METHODS = {
    "historical_free_mixed_model": "Historical CW-Free32",
    "new_free_cw_sol61_xhigh": "New CW-Free32",
    "nocw": "NoCW-Free32",
    "cw_gradient_lcmd32": "CW-gradient LCMD32",
    "free16_lcmd16": "CW-Free16 + CW-LCMD16",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {}

    def read(path):
        path = Path(path)
        path = path if path.is_absolute() else ROOT / path
        raw = path.read_bytes()
        assert raw == path.read_bytes(), ("Input changed while reading", path)
        sources[str(path.relative_to(ROOT))] = {
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "role": "saved validation summary, protocol or fit metadata; no truth table",
        }
        return raw

    def table(path):
        return list(csv.DictReader(io.StringIO(read(path).decode())))

    def js(path):
        return json.loads(read(path))

    rows = [r for r in table(OLD + "/tables/round_metrics.csv") if r["method"] in METHODS and r["method"] != "nocw"]
    acceptance = js(PAIRED + "/reports/final_export_acceptance_20261116.json")
    artifacts = js(FINAL + "/artifact_hashes.json")
    assert sources[FINAL + "/artifact_hashes.json"]["sha256"] == acceptance["final_artifact_manifest_sha256"]
    final = table(FINAL + "/validation_curves.csv")
    assert sources[FINAL + "/validation_curves.csv"]["sha256"] == artifacts["validation_curves.csv"]
    # Replace overlapping CW points with the exact saved-source ledger and add three seeds.
    keys = {(r["seed"], r["budget"]) for r in final if r["method"] == "cw"}
    rows = [r for r in rows if not (r["method"] == "cw_gradient_lcmd32" and (r["seed"], r["budget"]) in keys)]
    for r in final:
        rows.append({"method": "nocw" if r["method"] == "nocw" else "cw_gradient_lcmd32",
                     "seed": r["seed"], "budget": r["budget"], "nrmse": r["validation_nrmse"],
                     "source": r["fit_audit"], "fit_audit_sha256": r["fit_audit_sha256"]})
    by_run = {}
    for row in rows:
        by_run.setdefault((row["method"], int(row["seed"])), []).append(row)

    entries = []
    initial = {}
    for (method, seed), curve in sorted(by_run.items()):
        curve.sort(key=lambda r: int(r["budget"]))
        budgets = [int(r["budget"]) for r in curve]
        assert budgets == list(range(333, max(budgets) + 1, 32)), (method, seed, budgets)
        fit0 = js(curve[0]["source"])
        fit = js(curve[-1]["source"])
        for row, audit in ((curve[0], fit0), (curve[-1], fit)):
            assert audit["train_rows"] == int(row["budget"])
            assert abs(audit["best_validation_combined_normalized_rmse"] - float(row["nrmse"])) < 1e-11
            assert audit["test_labels_used_for_fit_or_checkpoint_selection"] == 0
            expected = row.get("fit_audit_sha256")
            if expected:
                assert sources[row["source"]]["sha256"] == expected
        ident = (fit0["initialization_hash"], fit0["validation_ids_hash"], fit0["train_ids_hash"], fit0["checkpoint_sha256"])
        if seed in initial:
            assert ident == initial[seed], ("Initial state differs", method, seed)
        initial[seed] = ident
        roots = sorted({str(Path(r["source"]).parents[2]) for r in curve})
        if method == "new_free_cw_sol61_xhigh":
            model = "gpt-6.1-sol/xhigh throughout; served-model lock"
            selection = f"{NEW}/selections/seed_{seed}/free_cw_sol61_xhigh_rerun32"
            prompt = "CW prompt SHA256 992df34be8c3822d2a2f0449e0911181c66cf4661bd52842221a72ddebd38089"
            notes = "Independent trajectory; finite 8-answer x 4-transport correction bound; original hybrid wording retained"
        elif method == "nocw":
            model = "gpt-6.1-sol/xhigh; served-model lock"
            selection = f"{PAIRED}/nocw/selections/seed_{seed}/llm32_nocw"
            prompt = "NoCW prompt SHA256 61d1f3cbd6b1b0b94c3df0a55282c91890cf14e35924448577c5618fea1adf4e"
            notes = "Original five-seed NoCW trajectory continued, not restarted; unbounded-retry amendments and transport/history-replay repairs"
        elif method == "historical_free_mixed_model":
            model = "gpt-6.1-sol/xhigh from L333"
            if seed == 157:
                model = "Astra/high r0-4; r5 accepted served-model unknown (authorized ID transcription); Sol/xhigh from L525"
            elif seed == 6101:
                model = "Astra/high r0-9; Sol/xhigh from L653"
            selection = "Multiple revision roots; see source_runtime_chain and historical frozen snapshots"
            prompt = "Historical full-pool scientific-plan v2 with explicit CW; prompt/config prefix differs in mixed-model seeds"
            notes = "Historical trajectory kept separate from New CW-Free; retries/transcription exceptions documented in existing audit"
        elif method == "free16_lcmd16":
            model = "Astra/high prefix through L813 transition; Sol/xhigh continuation"
            selection = "Multiple revision roots; cw16_llm16_scientist_v2 selection freezes"
            prompt = "Full-pool scientist v2; 16 numerical pending experiments visible, 16 LLM supplements"
            notes = "CW LCMD first; LLM supplements exclude pending; source attribution requires freezes, not choice order"
        else:
            model = "not applicable"
            selection = f"{PAIRED}/cw_lcmd/runtime/seed_{seed}/center_width_lcmd; historical extension chain for seed157/6101 L1517-L1549"
            prompt = "not applicable"
            notes = "CW-gradient LCMD, not raw-gradient LCMD. Five seeds through L1485; seeds157/6101 also through L1549"
        entries.append({
            "strategy": METHODS[method], "method": method, "seed": seed,
            "run_id": f"{method}:seed_{seed}", "batch_size": 32,
            "initial_L": 333, "final_L": max(budgets), "fitted_points": len(curve),
            "accepted_batches": (max(budgets) - 333) // 32,
            "source_runtime_chain": ";".join(roots), "selection_records": selection,
            "source_curve": (FINAL + "/validation_curves.csv" + (";" + OLD + "/tables/round_metrics.csv" if max(budgets) > 1485 else "")) if method in ("nocw", "cw_gradient_lcmd32") else OLD + "/tables/round_metrics.csv",
            "data": "4g; 4163 rows; outer train3330; L0=333; U0=2997; validation416; test417",
            "split": f"Row seed={seed}; no test truth read",
            "initialization_hash": ident[0], "validation_ids_hash": ident[1],
            "initial_train_ids_hash": ident[2], "initial_checkpoint_sha256": ident[3],
            "model": "QGeoGNN-V2;458952 parameters;scratch member0;Adam lr.001;batch2048;max1000 epochs;patience100",
            "rng": "initialization=seed+2000003; training=seed+3000017;CPU2threads",
            "endpoint_fit_audit": curve[-1]["source"], "endpoint_fit_audit_sha256": sources[curve[-1]["source"]]["sha256"],
            "endpoint_checkpoint_sha256": fit["checkpoint_sha256"],
            "llm_config": model, "prompt": prompt,
            "explicit_cw_visible": method not in ("nocw", "cw_gradient_lcmd32"),
            "gradient_source": "current-state q50 Center/Width full-network CountSketch512; L333 fixed C/W scales" if method != "nocw" else "CW hidden from LLM; cached raw/CW banks may exist for audit",
            "legal_pool": "actual U_t only; exclude all L_t and pending IDs; unique row IDs; no molecule/condition quotas",
            "proposal_vs_acceptance": "Final accepted freezes used; failed proposals are not acquired rows",
            "state_availability": "Saved checkpoint metadata, candidate catalogs/banks (per-state coverage audited separately); no re-extraction",
            "notes": notes,
        })
    target = args.output / "RUN_INVENTORY.csv"
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    metadata = [
        NEW + "/PROTOCOL.md", NEW + "/SCHEMA_DIFF.md", NEW + "/protocol.json",
        PAIRED + "/protocol.json", PAIRED + "/nocw/config.json", PAIRED + "/nocw/protocol.json",
        PAIRED + "/nocw/served_model_lock.json", PAIRED + "/reports/final_export_acceptance_20261116.json",
        OLD + "/PROTOCOL_AND_FAIRNESS_AUDIT.md", OLD + "/EXPERIMENT_INVENTORY.md",
        "studies/active_learning/nocw_free_selection_diagnostics_20261101/PROTOCOL_AUDIT.md",
        "src/qgeognn_al/active_learning_v2/gradient_transforms.py",
        "src/qgeognn_al/active_learning_v2/gradient_features.py", "src/qgeognn_al/active_learning_v2/lcmd.py",
        "src/qgeognn_al/active_learning_v2/scientist_selector.py", "src/qgeognn_al/active_learning_v2/scientist_full_delivery.py",
        "src/qgeognn_al/active_learning_v2/benchmark_protocol.py", "src/qgeognn_al/active_learning_v2/scientist_lifecycle.py",
        "src/qgeognn_al/active_learning_v2/nocw_inputs.py",
    ]
    for path in metadata:
        read(path)
    (args.output / "protocol_inputs.json").write_text(json.dumps({
        "files": [{"path": path, **value} for path, value in sorted(sources.items())],
        "campaigns": len(entries), "same_seed_initial_state_hashes_match": True,
        "uses_saved_summary_and_fit_metadata_only": True,
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"campaigns": len(entries), "runs_by_method": {m: sum(e["method"] == m for e in entries) for m in METHODS},
                      "initial_state_hashes_match": True, "inventory": str(target)}))


if __name__ == "__main__":
    main()
