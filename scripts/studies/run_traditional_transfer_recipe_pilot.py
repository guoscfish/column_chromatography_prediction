"""Controlled validation-only runner for the traditional transfer recipe pilot.

The default action is ``--dry-run``.  Fitting requires an explicit data/config
adapter supplied by the project; this runner deliberately has no test-scoring
path so a pilot cannot silently consume test labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHODS = {
    "P0": {"stage": "direct_shallow", "bn_policy": "current", "optimizer": "adam"},
    "P1": {"stage": "lp_then_shallow", "bn_policy": "current", "optimizer": "adam"},
    "P2": {"stage": "lp_then_shallow", "bn_policy": "source_stats", "optimizer": "adam"},
    "P3": {"stage": "lp_then_shallow", "bn_policy": "source_stats", "optimizer": "lbfgs"},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--column", default="25g")
    parser.add_argument("--protocol", default="row")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=sorted(METHODS), action="append")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--run", action="store_true", default=False)
    parser.add_argument("--maximum-epochs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("studies/transfer/traditional_transfer_improvement"))
    args = parser.parse_args()
    selected = args.method or list(METHODS)
    if args.protocol != "row" or args.column != "25g":
        raise SystemExit("pilot is frozen to column=25g and protocol=row")
    if args.run and args.dry_run:
        raise SystemExit("choose one of --dry-run or --run")
    manifest = {"status": "DRY_RUN" if args.dry_run or not args.run else "RUNNING", "column": args.column,
                "protocol": args.protocol, "seed": args.seed,
                "methods": {name: METHODS[name] for name in selected}, "test_scoring": False}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "pilot_artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.run:
        print(json.dumps(manifest, indent=2))
        return 0
    # The adapter is deliberately local and frozen to the audited 25g row data.
    import torch
    from src.qgeognn_al.data import build_model_data
    from src.qgeognn_al.models import load_predictor_checkpoint
    from src.qgeognn_al.transfer.adaptation import fit_target_scales, train_staged_target_adaptation, train_target_adaptation
    root = ROOT
    canonical_path = root / "studies/transfer/cross_column/data_audit/canonical_25g.csv"
    cache_path = root / "studies/transfer/cross_column/data_audit/graph_cache_25g_only.pt"
    source_checkpoint = root / "studies/predictor/final_4g_qualification/runtime/row/seed_42/best.pt"
    scaler = json.loads((root / "experiments/e0_4g_baseline/scaler.json").read_text())
    canonical = pd.read_csv(canonical_path).reset_index(drop=True)
    cache = torch.load(cache_path, weights_only=False)
    available = canonical[canonical["canonical_smiles"].isin(cache)].reset_index(drop=True)
    if len(available) < 10:
        raise RuntimeError("25g graph cache has insufficient rows for a pilot")
    canonical = available
    split_rng = np.random.default_rng(args.seed)
    order = split_rng.permutation(len(canonical))
    n_train = max(1, int(len(order) * 0.8)); n_valid = max(1, int(len(order) * 0.1))
    train_idx = order[:n_train].tolist(); valid_idx = order[n_train:n_train+n_valid].tolist()
    test_idx = order[n_train+n_valid:].tolist()
    split = {"train": train_idx, "valid": valid_idx, "test": test_idx}
    atom_data, angle_data = build_model_data(canonical, cache, None, scaler)
    scales = fit_target_scales(atom_data, train_idx)
    preprocessing = {"target_scales": scales, "fit_role": "target_train", "source_scaler": "experiments/e0_4g_baseline/scaler.json"}
    results = {}
    started = time.time()
    for name in selected:
        model = load_predictor_checkpoint(source_checkpoint)
        spec = METHODS[name]
        config = {"maximum_epochs": args.maximum_epochs, "patience": max(1, args.maximum_epochs), "batch_size": 2048,
                  "bn_policy": spec["bn_policy"], "optimizer": spec["optimizer"], "lbfgs_max_iter": 3}
        if spec["stage"] == "direct_shallow":
            fit = train_target_adaptation(model, atom_data, angle_data, train_idx, valid_idx, preprocessing,
                                          mode="last2", seed=args.seed, config=config)
            payload = fit.as_dict()
        else:
            staged = train_staged_target_adaptation(model, atom_data, angle_data, train_idx, valid_idx, preprocessing,
                                                     stage_b_mode="last2", seed=args.seed, stage_a_config=config, stage_b_config=config)
            payload = staged.as_dict()
        results[name] = payload
    manifest.update({"status": "COMPLETED_VALIDATION_ONLY", "rows_available_in_graph_cache": len(canonical), "split_counts": {k: len(v) for k,v in split.items()},
                     "split_indices": split, "target_scales": scales, "results": results,
                     "test_indices_recorded_only": True, "source_checkpoint": str(source_checkpoint.relative_to(root)),
                     "source_checkpoint_sha256": hashlib.sha256(source_checkpoint.read_bytes()).hexdigest(),
                     "maximum_epochs": args.maximum_epochs, "wall_time_seconds": time.time() - started,
                     "pilot_selection_basis": "validation_score_only", "test_truth_used": False})
    (args.output / "pilot_artifact_manifest.json").write_text(json.dumps(manifest, indent=2, default=float) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
