#!/usr/bin/env python3
"""Gate the standalone predictor against the trained, audited R2-pruned model."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.studies import run_r2_pruned_requalification as old
from src.qgeognn_al.historical.conversion import convert_pruned_to_standalone
from src.qgeognn_al.models import build_predictor, load_predictor_checkpoint, predictor_checkpoint
from src.qgeognn_al.schemas.conditions import ConditionNormalization
from src.qgeognn_al.training.predictor import atomic_json, point_metrics, predict

STUDY = ROOT / "studies/predictor/final_v2_engineering"


def main():
    data, split, indices, atom, angle, cn, norm, scales, batches = old.inputs()
    norm = ConditionNormalization(**asdict(norm))
    pruned = old.mapped_model(old.REFERENCE, cn, norm, torch.device("cpu"))
    checkpoint_path = old.STUDY / "runtime/R2_PRUNED/best.pt"
    expected = json.loads((old.STUDY / "results/R2_PRUNED/checkpoint_metadata.json").read_text())["sha256"]
    assert old.r2.sha256_file(checkpoint_path) == expected
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pruned.load_state_dict(payload["model_state_dict"])
    standalone = build_predictor(norm)
    state, conversion = convert_pruned_to_standalone(payload["model_state_dict"], standalone)
    standalone.load_state_dict(state)
    by_split = {}
    for role, positions in indices.items():
        old_metrics, truth, original, _ = old.r2.evaluate(pruned, old.REFERENCE, data, atom, angle, positions, cn, scales, torch.device("cpu"))
        truth, prediction, _ = predict(standalone, atom, angle, positions)
        new_metrics = point_metrics(truth, prediction, scales)
        diff = np.abs(original - prediction)
        by_split[role] = {"rows": len(positions), "six_output_max_abs_difference": diff.max(0).tolist(),
                          "max_abs_difference": float(diff.max()), "original_metrics": old_metrics,
                          "standalone_metrics": new_metrics,
                          "metric_deltas": {k: float(new_metrics[k]-old_metrics[k]) for k in new_metrics if k != "all_outputs_finite"}}
    # Freshly reload after gradient audit so BN statistics never modify migration evidence.
    standalone.train()
    seen, counts = set(), []
    for a, b in batches:
        standalone.zero_grad(set_to_none=True)
        standalone(a, b).sum().backward()
        names = {n for n, p in standalone.named_parameters() if p.grad is not None}
        seen |= names
        counts.append(sum(p.numel() for n, p in standalone.named_parameters() if n in names))
    rows = [{"name": n, "count": p.numel(), "requires_grad": p.requires_grad, "gradient_present": n in seen}
            for n, p in standalone.named_parameters()]
    total = sum(r["count"] for r in rows)
    reachable = sum(r["count"] for r in rows if r["gradient_present"])
    reachability = {"nominal_parameters": total, "requires_grad_parameters": sum(r["count"] for r in rows if r["requires_grad"]),
                    "gradient_bearing_parameters": reachable, "forward_unreachable_trainable_parameters": total-reachable,
                    "batches": counts, "parameters": rows, "status": "PASS" if total == reachable == 458952 else "FAIL"}
    maximum = max(r["max_abs_difference"] for r in by_split.values())
    metric_max = max(abs(v) for r in by_split.values() for v in r["metric_deltas"].values())
    gate = {"status": "PASS" if maximum <= 1e-6 and metric_max <= 1e-6 and reachability["status"] == "PASS" else "FAIL",
            "max_abs_difference": maximum, "metric_max_abs_difference": metric_max, "tolerance": 1e-6,
            "fixture_rows": len(data), "source_checkpoint_sha256": expected, "splits": by_split,
            "conversion": conversion, "direct_construction": True, "legacy_model_constructed_by_active_api": False}
    atomic_json(STUDY / "equivalence_audit.json", gate)
    atomic_json(STUDY / "reachability_audit.json", reachability)
    if gate["status"] != "PASS":
        raise RuntimeError("standalone equivalence failed; qualification/transfer forbidden")
    normalization = json.loads((old.STUDY / "results/R2_PRUNED/normalization.json").read_text())
    preprocessing = {"scaler": normalization["molecular_and_eluent_minmax"], "target_scales": scales,
                     "fit_role": "source_train", "fit_rows": 3330, "test_rows_used": 0, "target_rows_used": 0}
    standalone.load_state_dict(state)
    (STUDY / "runtime").mkdir(exist_ok=True)
    mapped_path = STUDY / "runtime/migrated.pt"
    torch.save(predictor_checkpoint(standalone, preprocessing=preprocessing, training_config={"role": "engineering_mapping_only"},
               provenance={"source_checkpoint_sha256": expected}), mapped_path)
    reloaded = load_predictor_checkpoint(mapped_path)
    assert all(torch.equal(v, reloaded.state_dict()[k]) for k, v in state.items())
    print(json.dumps({"equivalence": gate["status"], "max_difference": maximum, "parameters": total, "unreachable": total-reachable}), flush=True)


if __name__ == "__main__":
    main()
