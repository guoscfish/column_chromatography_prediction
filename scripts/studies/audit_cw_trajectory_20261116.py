#!/usr/bin/env python3
"""Recheck all saved CW acquisition transitions and fits, without source labels."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.studies import audit_nocw_trajectory_20261116 as nocw
from scripts.studies import run_nocw_cw_paired_to1485_20261112 as frozen_comparison

ROOT, STUDY, SEEDS = nocw.ROOT, nocw.STUDY, nocw.SEEDS


def audit_seed(seed):
    sources = {}

    def digest(path):
        path = path.resolve()
        key = str(path.relative_to(ROOT))
        if key not in sources:
            with path.open("rb") as handle:
                sources[key] = hashlib.file_digest(handle, "sha256").hexdigest()
        return sources[key]

    def read(path):
        digest(path)
        return nocw.read(path)

    def table(path):
        digest(path)
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    reference = ROOT / "reports/five_methods_cw_gradient_20261103/validation_curves.csv"
    historical = {int(r["budget"]): r for r in table(reference)
                  if r["method"] == "cw_gradient_lcmd32" and int(r["seed"]) == seed}
    initial = read(STUDY / f"nocw/selections/seed_{seed}/llm32_nocw/round_00/contract.json")
    baseline_freeze = read(STUDY / f"nocw/runtime/seed_{seed}/llm32_nocw/round_00/prediction_freeze.json")
    baseline = read(Path(baseline_freeze["checkpoint_path"]).with_name("fit_audit.json"))
    states, freezes, points = [], [], []
    for rnd in range(37):
        budget = 333 + 32 * rnd
        directory = STUDY / f"cw_lcmd/runtime/seed_{seed}/center_width_lcmd/round_{rnd:02d}"
        receipt_path = directory / "fit_receipt.json"
        if not receipt_path.exists():
            row = historical[budget]
            # Extension anchors reuse the preceding checkpoint but freeze a new
            # outgoing batch. Prefer the latest saved anchor over the fit's home.
            candidates = [ROOT / f"studies/active_learning/{study}/runtime/seed_{seed}/center_width_lcmd/round_{rnd:02d}"
                          for study in ("qgeognn_v2_row_cw_lcmd_to_1005", "qgeognn_v2_row_cw_lcmd_to_653",
                                        "qgeognn_v2_row_cw_lcmd_to_525",
                                        "qgeognn_v2_row_short_sequential_b32")]
            directory = next(d for d in candidates if (d / "round_freeze.json").exists())
        frozen = read(directory / "round_freeze.json")
        checkpoint = Path(frozen["checkpoint_path"])
        audit_path = checkpoint.with_name("fit_audit.json")
        audit = read(audit_path)
        nocw.require(digest(checkpoint) == frozen["checkpoint_sha256"] == audit["checkpoint_sha256"],
                     "CW checkpoint hash drift")
        nocw.require(digest(Path(frozen["prediction_path"])) == frozen["prediction_sha256"] == audit["prediction_sha256"],
                     "CW prediction hash drift")
        nocw.require(frozen["test_truth_access_count"] == audit["test_labels_used_for_fit_or_checkpoint_selection"] == 0,
                     "CW test-truth access")
        nocw.require(audit["train_rows"] == budget, "CW fit budget drift")
        for key in ("initialization_hash", "validation_ids_hash"):
            nocw.require(audit[key] == baseline[key], "CW/NoCW paired baseline differs: " + key)
        if receipt_path.exists():
            receipt = read(receipt_path)
            nocw.require(receipt["fit_audit_sha256"] == digest(audit_path), "CW fit receipt hash drift")
            nocw.require(receipt["train_rows"] == budget, "CW receipt budget drift")
            nocw.require(receipt["validation_nrmse"] == audit["best_validation_combined_normalized_rmse"],
                         "CW receipt metric drift")
        else:
            nocw.require(digest(audit_path) == historical[budget]["fit_audit_sha256"], "Imported CW fit hash drift")
        state = table(directory / "state.csv")
        labeled = [r["sample_id"] for r in state if r["role"] == "labeled"]
        unlabeled = [r["sample_id"] for r in state if r["role"] == "unlabeled"]
        nocw.require(len(labeled) == len(set(labeled)) == budget, "CW labeled budget or uniqueness drift")
        nocw.require(len(unlabeled) == len(set(unlabeled)) and not set(labeled).intersection(unlabeled),
                     "CW pool identity overlap")
        nocw.require(audit["train_ids_hash"] == nocw.ids_hash(labeled), "CW fitted IDs differ from saved state")
        if "input" in frozen:
            contract = read(directory / "input_contract.json")
            nocw.require(contract == frozen["input"], "CW frozen input differs from saved contract")
            nocw.require(contract["L_t_ids_hash"] == nocw.ids_hash(labeled) and
                         contract["U_t_ids_hash"] == nocw.ids_hash(unlabeled), "CW contract/state ID hash drift")
        states.append((labeled, unlabeled, directory))
        freezes.append(frozen)
        points.append({"round": rnd, "budget": budget,
                       "validation_nrmse": audit["best_validation_combined_normalized_rmse"],
                       "fit_audit": str(audit_path.relative_to(ROOT)), "fit_audit_sha256": digest(audit_path)})
    nocw.require(set(states[0][0]) == set(initial["L_t_ids"]) and
                 set(states[0][1]) == set(initial["U_t_ids"]), "Paired CW/NoCW initial training pool differs")
    additions, batches = [], []
    for rnd in range(36):
        labeled, unlabeled, directory = states[rnd]
        next_l, next_u, _ = states[rnd + 1]
        selected = next_l[len(labeled):]
        nocw.require(next_l[:len(labeled)] == labeled, "CW labeled order/transition drift")
        nocw.require(len(selected) == len(set(selected)) == 32 and set(selected).issubset(unlabeled),
                     "CW batch must contain 32 unique legal new IDs")
        nocw.require(not set(selected).intersection(labeled), "CW acquired ID repeated")
        nocw.require(next_u == [x for x in unlabeled if x not in set(selected)], "CW unlabeled transition drift")
        frozen = freezes[rnd]
        if "selected_next_batch" in frozen:
            nocw.require(frozen["selected_next_batch"] == selected, "CW selected batch differs from next fit")
            receipt = read(directory / "label_access_receipt.json")
            nocw.require(receipt["batch_ids"] == selected and
                         receipt["round_freeze_sha256"] == digest(directory / "round_freeze.json"),
                         "CW label-reveal freeze lineage drift")
            access = receipt["audit"]
            nocw.require(access["requested_rows"] == 32 and access["roles"] == "u0" and
                         access["requested_ids_hash"] == nocw.ids_hash(selected) and access["acquisitions_frozen"] is True,
                         "CW saved label-reveal scope drift")
        else:
            nocw.require(frozen["outgoing_selected_ids_hash"] == nocw.stable(selected),
                         f"Historical CW outgoing selection hash drift: seed={seed}, round={rnd}")
            nocw.require(freezes[rnd + 1].get("input", {}).get("incoming_selected_ids_hash", nocw.stable(selected)) == nocw.stable(selected),
                         "Historical CW incoming selection hash drift")
        additions.extend(selected)
        batches.append({"round": rnd, "batch_size": 32,
                        "round_freeze_sha256": digest(directory / "round_freeze.json")})
    nocw.require(len(additions) == len(set(additions)) == 1152, "CW total acquired-ID uniqueness drift")
    nocw.require(not set(additions).intersection(states[0][0]), "CW additions overlap initial training IDs")
    nocw.require(freezes[-1].get("selected_next_batch") == [], "CW final freeze has an outgoing batch")
    return {"seed": seed, "status": "VERIFIED_COMPLETE", "completed_rounds": 36,
            "evaluation_points": 37, "final_fitted_budget": 1485, "new_unique_training_ids": 1152,
            "paired_initialization_and_validation_match": True, "test_truth_reads": 0,
            "raw_saved_validation_points": points, "verified_acquisition_batches": batches,
            "source_hashes": sources}


def run(output):
    nocw.require(not output.exists(), "Use a fresh audit path")
    protocol = frozen_comparison.validate()
    results = [audit_seed(seed) for seed in SEEDS]
    record = {"checked_utc": datetime.now(timezone.utc).isoformat(), "method": "CW-gradient LCMD32",
              "scientific_protocol_validation": protocol,
              "all_requested_trajectories_complete": True, "new_model_calls": 0, "new_fits": 0,
              "test_truth_reads": 0, "per_seed": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"audit": str(output), "seeds_verified": len(results), "status": "VERIFIED_COMPLETE"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())
