#!/usr/bin/env python3
"""Audit saved NoCW trajectories without loading a label store or model.

The default requires all 36 acquisitions and 37 completed fits. --allow-prefix
audits an immutable fitted prefix and never treats a pending batch as a fit.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies/active_learning/nocw_vs_cw_lcmd_five_to1485_20261112"
SEEDS = (157, 887, 2357, 6101, 12203)
ARM = "llm32_nocw"


def read(path):
    return json.loads(path.read_text())


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def ids_hash(values):
    return stable(sorted(map(str, values)))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def audit_seed(seed, allow_prefix=False):
    sources = {}

    def digest(path):
        path = path.resolve()
        key = str(path.relative_to(ROOT))
        if key not in sources:
            with path.open("rb") as handle:
                sources[key] = hashlib.file_digest(handle, "sha256").hexdigest()
        return sources[key]

    runtime = STUDY / f"nocw/runtime/seed_{seed}/{ARM}"
    selections = STUDY / f"nocw/selections/seed_{seed}/{ARM}"
    receipts = sorted(runtime.glob("round_*/prediction_freeze.json"))
    rounds = [int(f.parent.name.split("_")[-1]) for f in receipts]
    require(rounds and rounds == list(range(rounds[-1] + 1)), "Missing or unordered fitted evaluation points")
    last = rounds[-1]
    require(last <= 36 and (allow_prefix or last == 36), "Full completion requires 37 saved fitted points")
    initial_contract = read(selections / "round_00/contract.json")
    labeled = list(map(str, initial_contract["L_t_ids"]))
    unlabeled = list(map(str, initial_contract["U_t_ids"]))
    initial_ids, additions = set(labeled), []
    require(len(labeled) == len(initial_ids) == 333, "Initial L333 identity drift")
    require(len(unlabeled) == len(set(unlabeled)) and not initial_ids.intersection(unlabeled), "Initial pool overlap")
    baseline = read(Path(read(receipts[0])["checkpoint_path"]).with_name("fit_audit.json"))
    cw_receipt = STUDY / f"cw_lcmd/runtime/seed_{seed}/center_width_lcmd/round_00/fit_receipt.json"
    if cw_receipt.exists():
        value = read(cw_receipt)
        cw_audit = Path(value["checkpoint_path"]).with_name("fit_audit.json")
        require(digest(cw_audit) == value["fit_audit_sha256"], "Paired CW audit drift")
        digest(cw_receipt)
    else:
        reference = ROOT / "reports/five_methods_cw_gradient_20261103/validation_curves.csv"
        digest(reference)
        with reference.open(newline="") as handle:
            old = next(row for row in csv.DictReader(handle) if row["method"] == "cw_gradient_lcmd32"
                       and int(row["seed"]) == seed and int(row["budget"]) == 333)
        cw_audit = ROOT / old["fit_audit"]
        require(digest(cw_audit) == old["fit_audit_sha256"], "Imported paired CW audit drift")
    cw = read(cw_audit)
    for key in ("initialization_hash", "validation_ids_hash"):
        require(cw[key] == baseline[key], "Same-seed CW/NoCW baseline mismatch: " + key)
    points, batches = [], []
    for rnd, receipt_path in enumerate(receipts):
        frozen = read(receipt_path)
        checkpoint = Path(frozen["checkpoint_path"])
        audit_path = checkpoint.with_name("fit_audit.json")
        fit = read(audit_path)
        digest(receipt_path)
        digest(audit_path)
        require(len(labeled) == len(set(labeled)) == 333 + 32 * rnd, "Training budget/uniqueness drift")
        require(fit["train_rows"] == len(labeled), "Fit budget differs from frozen acquisition history")
        require(fit["train_ids_hash"] == frozen["L_t_ids_hash"] == ids_hash(labeled), "Fit training IDs differ")
        require(frozen["test_truth_access_count"] == fit["test_labels_used_for_fit_or_checkpoint_selection"] == 0,
                "Test-truth access in fit or predictor")
        for key in ("initialization_hash", "validation_ids_hash"):
            require(fit[key] == baseline[key], "Fit initialization/validation drift")
        require(digest(checkpoint) == frozen["checkpoint_sha256"] == fit["checkpoint_sha256"], "Checkpoint hash drift")
        require(digest(Path(frozen["prediction_path"])) == frozen["prediction_sha256"] == fit["prediction_sha256"],
                "Prediction hash drift")
        metric = fit["best_validation_combined_normalized_rmse"]
        require(math.isfinite(metric) and metric >= 0, "Invalid saved validation metric")
        points.append({"round": rnd, "budget": len(labeled), "validation_nrmse": metric,
                       "fit_audit": str(audit_path.relative_to(ROOT)), "fit_audit_sha256": digest(audit_path)})
        if rnd == last:
            break
        directory = selections / f"round_{rnd:02d}"
        contract = read(directory / "contract.json")
        batch = read(directory / "batch_freeze.json")
        selected = list(map(str, batch["batch_ids"]))
        require(contract["seed"] == batch["seed"] == seed and contract["round"] == batch["round"] == rnd,
                "Batch seed/round identity drift")
        require(contract["method"] == batch["method"] == ARM, "Acquisition method drift")
        require(list(map(str, contract["L_t_ids"])) == labeled and list(map(str, contract["U_t_ids"])) == unlabeled,
                "L/U trajectory transition differs")
        require(len(selected) == len(set(selected)) == 32 and set(selected).issubset(unlabeled), "Invalid unique batch32")
        require(not set(selected).intersection(labeled), "Repeated training IDs")
        require(contract["checkpoint_sha256"] == batch["checkpoint_sha256"] == frozen["checkpoint_sha256"],
                "Selection predictor differs from saved fitted predictor")
        require(batch["new_batch_labels_revealed"] is False and batch["dry_run"] is False, "Selection freeze scope drift")
        for name, expected in batch["artifact_hashes"].items():
            require(digest(directory / name) == expected, "Frozen selection artifact drift: " + name)
        receipt = read(directory / "label_access_receipt.json")
        feedback = read(directory / "feedback.json")
        batch_hash = digest(directory / "batch_freeze.json")
        require(receipt["batch_freeze_sha256"] == feedback["batch_freeze_sha256"] == batch_hash, "Label-reveal lineage drift")
        require(receipt["feedback_sha256"] == stable(feedback), "Semantic feedback hash drift")
        access = receipt["audit"]
        require(access["requested_rows"] == 32 and access["requested_ids_hash"] == ids_hash(selected)
                and access["roles"] == "u0" and access["acquisitions_frozen"] is True, "Label-access scope drift")
        require(len(feedback["records"]) == 32 and {str(r["candidate_id"]) for r in feedback["records"]} == set(selected),
                "Observed feedback IDs differ from selected IDs")
        digest(directory / "label_access_receipt.json")
        digest(directory / "feedback.json")
        additions.extend(selected)
        labeled += selected
        unlabeled = [x for x in unlabeled if x not in set(selected)]
        batches.append({"round": rnd, "new_unique_training_ids": 32, "batch_freeze_sha256": batch_hash})
    require(len(additions) == len(set(additions)) == 32 * last and not initial_ids.intersection(additions),
            "Global acquired-ID uniqueness drift")
    if last == 36:
        saved_batch_rounds = sorted(int(f.parent.name.split("_")[-1]) for f in selections.glob("round_*/batch_freeze.json"))
        require(saved_batch_rounds == list(range(36)), "Completed trajectory must have exactly36 acquisition freezes")
        require(len(points) == 37 and len(additions) == 1152 and len(labeled) == 1485, "Incomplete final trajectory")
    return {"seed": seed, "status": "VERIFIED_COMPLETE" if last == 36 else "VERIFIED_FITTED_PREFIX",
            "complete": last == 36, "completed_rounds": last, "evaluation_points": len(points),
            "final_fitted_budget": 333 + 32 * last, "new_unique_training_ids": len(additions),
            "paired_initialization_and_validation_match": True, "test_truth_reads": 0,
            "raw_saved_validation_points": points, "verified_acquisition_batches": batches,
            "source_hashes": sources}


def run(seeds, output, allow_prefix=False):
    require(not output.exists(), "Use a new audit output path")
    spec = importlib.util.spec_from_file_location("frozen_comparison_audit", ROOT / "scripts/studies/run_nocw_cw_paired_to1485_20261112.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol = module.validate()
    results = [audit_seed(seed, allow_prefix) for seed in seeds]
    record = {"checked_utc": datetime.now(timezone.utc).isoformat(), "scientific_protocol_validation": protocol,
              "all_requested_trajectories_complete": all(r["complete"] for r in results), "seeds": list(seeds),
              "new_model_calls": 0, "new_fits": 0, "test_truth_reads": 0, "per_seed": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"audit": str(output), "complete": record["all_requested_trajectories_complete"],
                      "per_seed": [{k: r[k] for k in ("seed", "status", "completed_rounds", "evaluation_points",
                                                       "final_fitted_budget", "new_unique_training_ids")} for r in results]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-prefix", action="store_true")
    args = parser.parse_args()
    run(args.seed or SEEDS, args.output.resolve(), args.allow_prefix)
