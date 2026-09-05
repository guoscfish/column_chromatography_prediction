#!/usr/bin/env python3
"""Frozen R2 reachability, equivalence gates, and controlled pruning retrain."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.studies import run_point_predictor_regression_audit as r2

STUDY = ROOT / "studies/predictor/r2_pruned_requalification"
REFERENCE = "R2_CONDITION_COMPLETE_V2"


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in state.items():
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def mapped_model(variant, clean_norm, norm, device):
    from src.qgeognn_al.condition_complete_v2_pruned import (
        PrunedConditionCompleteQGeoGNNV2, convert_r2_state_dict_to_pruned,
    )
    canonical = r2.build_variant(REFERENCE, clean_norm, norm, device)
    rng = torch.get_rng_state().clone()
    pruned = PrunedConditionCompleteQGeoGNNV2(canonical)
    converted, report = convert_r2_state_dict_to_pruned(canonical.state_dict(), canonical, pruned)
    pruned.load_state_dict(converted, strict=True)
    if not torch.equal(torch.get_rng_state(), rng):
        raise RuntimeError("pruned construction unexpectedly consumed RNG")
    return pruned


def compare(original, pruned, data, atom, angle, indices, clean_norm, scales):
    result = {}
    for role, positions in indices.items():
        left, truth, prediction, ids = r2.evaluate(original, REFERENCE, data, atom, angle, positions, clean_norm, scales, torch.device("cpu"))
        right, _, other, other_ids = r2.evaluate(pruned, REFERENCE, data, atom, angle, positions, clean_norm, scales, torch.device("cpu"))
        assert np.array_equal(ids, other_ids)
        diff = np.abs(prediction - other)
        metric_diff = {k: abs(left[k] - right[k]) for k in left if k != "all_outputs_finite"}
        result[role] = {"rows": len(ids), "max_abs_difference": float(diff.max()),
                        "six_output_max_abs_difference": diff.max(axis=0).tolist(),
                        "original_metrics": left, "pruned_metrics": right,
                        "metric_absolute_differences": metric_diff,
                        "status": "PASS" if np.isfinite(diff).all() and diff.max() <= 1e-6 and max(metric_diff.values()) <= 1e-6 else "FAIL"}
    return result


def qualify(data, indices, atom, angle, clean_norm, norm, scales, batches, before, *, train=True):
    from src.qgeognn_al.condition_complete_v2_pruned import (
        PrunedConditionCompleteQGeoGNNV2, convert_r2_state_dict_to_pruned, MODEL_VARIANT,
    )
    canonical = r2.build_variant(REFERENCE, clean_norm, norm, torch.device("cpu"))
    rng = torch.get_rng_state().clone()
    pruned = PrunedConditionCompleteQGeoGNNV2(canonical)
    converted, conversion = convert_r2_state_dict_to_pruned(canonical.state_dict(), canonical, pruned)
    pruned.load_state_dict(converted, strict=True)
    initialization = {"seed": 42, "canonical_variant": REFERENCE,
                      "initialization_mapping_hash": state_hash(converted),
                      "retained_initial_values_bitwise_equal": conversion["retained_values_bitwise_equal"],
                      "rng_state_preserved": torch.equal(rng, torch.get_rng_state()),
                      "retained_parameter_order_equal": [n for n, _ in canonical.named_parameters() if n in dict(pruned.named_parameters())] == [n for n, _ in pruned.named_parameters()],
                      "canonical_initialization_hash": state_hash(canonical.state_dict()), "status": "PASS"}
    assert all(initialization[k] for k in ("retained_initial_values_bitwise_equal", "rng_state_preserved", "retained_parameter_order_equal"))
    r2.atomic_json(STUDY / "initialization_audit.json", initialization)
    random_gate = compare(canonical, pruned, data, atom, angle, indices, clean_norm, scales)
    checkpoint = r2.RUNTIME / REFERENCE / "best.pt"
    metadata = json.loads((r2.RESULTS / REFERENCE / "checkpoint_metadata.json").read_text())
    assert r2.sha256_file(checkpoint) == metadata["sha256"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    canonical.load_state_dict(payload["model_state_dict"], strict=True)
    converted, conversion = convert_r2_state_dict_to_pruned(payload["model_state_dict"], canonical, pruned)
    pruned.load_state_dict(converted, strict=True)
    conversion["source_checkpoint_sha256"] = r2.sha256_file(checkpoint)
    r2.atomic_json(STUDY / "state_dict_conversion_audit.json", conversion)
    trained_gate = compare(canonical, pruned, data, atom, angle, indices, clean_norm, scales)
    after = reachability(pruned, batches)
    r2.atomic_json(STUDY / "R2_REACHABILITY_AFTER.json", after)
    # Check the same loss/backward/Adam update as formal training on a real batch.
    canonical = r2.build_variant(REFERENCE, clean_norm, norm, torch.device("cpu"))
    pruned = mapped_model(REFERENCE, clean_norm, norm, torch.device("cpu"))
    optimizers = [torch.optim.Adam(m.parameters(), lr=.001, weight_decay=0) for m in (canonical, pruned)]
    step_diffs = []
    for a, b in batches[:2]:
        for m, optimizer in zip((canonical, pruned), optimizers):
            m.train()
            optimizer.zero_grad()
            p = m(a, b)[0]
            loss = r2.quantile_target_loss(a.y[:, 0], p[:, :3]) + r2.quantile_target_loss(a.y[:, 1], p[:, 3:])
            loss.backward()
            optimizer.step()
        step_diffs.append(max(float((canonical.state_dict()[k] - v).abs().max()) for k, v in pruned.state_dict().items()))
    gate = {"P0_random_init": random_gate, "P1_trained_checkpoint": trained_gate,
            "output_order": ["V1_q10", "V1_q50", "V1_q90", "V2_q10", "V2_q50", "V2_q90"],
            "tolerance": 1e-6, "fixture_rows": len(data),
            "fixture_coverage": {
                "molecules": int(data["canonical_smiles"].nunique()),
                "loading_solvents": sorted(data["loading solvent"].unique().tolist()),
                "loading_mass_values": int((data["Density g/ml"] * data["V/ul"]).nunique()),
                "loading_solvent_volume_values": int(data["Volume of loading solvent/ul"].nunique()),
                "eluent_compositions": sorted(data["PE/EA"].unique().tolist()),
            },
            "two_adam_step_retained_state_max_differences": step_diffs,
            "status": "PASS" if all(row["status"] == "PASS" for section in (random_gate, trained_gate) for row in section.values()) and after["forward_unreachable_trainable_parameters"] == 0 and max(step_diffs) <= 1e-6 else "FUNCTION_PRESERVING_PRUNING_FAILED"}
    r2.atomic_json(STUDY / "function_equivalence_audit.json", gate)
    if gate["status"] != "PASS":
        r2.atomic_json(STUDY / "decision.json", {"status": gate["status"], "training_executed": False})
        raise RuntimeError("FUNCTION_PRESERVING_PRUNING_FAILED; formal training forbidden")
    config = r2.variant_config(REFERENCE)
    config.update(scientific_role="IMPLEMENTATION_CLEANUP_AND_CONTROLLED_RETRAIN", variant="R2_PRUNED",
                  architecture=MODEL_VARIANT, initialization_mapping_hash=initialization["initialization_mapping_hash"],
                  active_learning=False, transfer=False, uq=False)
    config.pop("config_hash")
    config["config_hash"] = r2.stable_hash(config)
    r2.atomic_json(STUDY / "protocol.json", config)
    r2.atomic_json(STUDY / "environment.json", {"python": platform.python_version(), "torch": torch.__version__,
                   "numpy": np.__version__, "device": "cpu", "num_threads": torch.get_num_threads(),
                   "KMP_DUPLICATE_LIB_OK": os.environ.get("KMP_DUPLICATE_LIB_OK"),
                   "note": "fish environment matches historical R2 torch 2.10.0; existing repository OpenMP workaround"})
    if not train:
        print("Function-preserving gates PASS; gates-only invocation", flush=True)
        return None
    print("Function-preserving gates PASS; starting controlled retrain", flush=True)
    def progress(record):
        if record["epoch"] % 10 == 0 or record["epoch"] == 1:
            r2.atomic_json(STUDY / "runtime/progress.json", record)
            print(json.dumps(record), flush=True)
    def factory(variant, cn, vn, device):
        model = mapped_model(variant, cn, vn, device)
        assert state_hash(model.state_dict()) == initialization["initialization_mapping_hash"]
        return model
    return r2.run_variant(REFERENCE, torch.device("cpu"), model_factory=factory,
                          results_dir=STUDY / "results", runtime_dir=STUDY / "runtime",
                          config_override=config, run_name="R2_PRUNED", progress_callback=progress)


def static_reason(name):
    if name.startswith("legacy_model.NN_descriptor."):
        return "GINGraphPooling/ConditionComplete representations never call NN_descriptor"
    node = "legacy_model.gnn_node."
    if name.startswith(tuple(node + x + "." for x in ("batch_norms", "batch_norms_ba"))):
        return "outer BatchNorm unused in geometry-enhanced forward; internal GIN MLP BatchNorm retained"
    if name.startswith(node + "bond_angle_encoder."):
        return "initial bond_angle_encoder registered but never called"
    if name.startswith(tuple(node + x + ".4." for x in (
        "convs_bond_angle", "convs_bond_embeding", "convs_bond_float", "convs_angle_float"
    ))):
        return "terminal edge update computed after final node update; returned edge representation discarded by R2"
    return None


def reachability(model, batches):
    seen = set()
    per_batch = []
    calls = set()
    handles = [m.register_forward_hook(lambda m, a, o, n=n: calls.add(n))
               for n, m in model.named_modules()]
    model.train()
    for atom, angle in batches:
        model.zero_grad(set_to_none=True)
        output = model(atom, angle)[0]
        output.sum().backward()
        present = {n for n, p in model.named_parameters() if p.grad is not None}
        seen.update(present)
        per_batch.append({"graphs": int(atom.num_graphs), "gradient_parameter_count":
                          sum(p.numel() for n, p in model.named_parameters() if n in present)})
    for h in handles:
        h.remove()
    records = []
    for name, p in model.named_parameters():
        reason = static_reason(name)
        reachable = name in seen
        if not reachable and reason is None:
            raise RuntimeError(f"unexplained unreachable parameter: {name}")
        if reachable and reason:
            raise RuntimeError(f"static dead classification contradicted: {name}")
        records.append({"name": name, "module": name.rsplit(".", 1)[0],
                        "parameter_count": p.numel(), "requires_grad": p.requires_grad,
                        "gradient_present": reachable, "output_reachable": reachable,
                        "module_executed": name.rsplit(".", 1)[0] in calls,
                        "deletion_eligibility": bool(reason and not reachable),
                        "static_trace": reason or "retained prediction dependency confirmed by autograd"})
    modules = {}
    for record in records:
        key = record["module"]
        row = modules.setdefault(key, {"parameter_count": 0, "requires_grad": True,
                    "gradient_present": record["gradient_present"], "output_reachable": record["output_reachable"],
                    "deletion_eligibility": record["deletion_eligibility"], "static_trace": record["static_trace"]})
        row["parameter_count"] += record["parameter_count"]
    return {"nominal_parameters": sum(p.numel() for p in model.parameters()),
            "requires_grad_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "gradient_bearing_parameters": sum(p.numel() for n, p in model.named_parameters() if n in seen),
            "forward_unreachable_trainable_parameters": sum(p.numel() for n, p in model.named_parameters()
                                                            if p.requires_grad and n not in seen),
            "gradient_definition": "grad is not None; zero-valued gradients remain reachable",
            "static_trace_sources": {
                "application/QGeoGNN.py": {"sha256": r2.sha256_file(ROOT / "application/QGeoGNN.py"),
                    "geometry_forward_lines": [1169, 1207], "outer_descriptor_lines": [1296, 1317]},
                "src/qgeognn_al/condition_complete_v2.py": {
                    "sha256": r2.sha256_file(ROOT / "src/qgeognn_al/condition_complete_v2.py"),
                    "trace": "representations discards edge return; sums nodes, adds typed completion; forward applies unchanged head"},
            },
            "batches": per_batch, "parameters": records, "modules": modules}


def inputs():
    data, split, indices = r2.load_frozen_inputs()
    atom, angle, scaler, match, clean_norm, norm, scales = r2.fit_inputs(data, split, indices)
    # Three real, disjoint batches spanning the complete domain (no optimizer steps).
    batches = []
    for positions in np.array_split(np.arange(len(data)), 3):
        batches.extend(zip(*r2.loader_pair(atom, angle, positions)))
    return data, split, indices, atom, angle, clean_norm, norm, scales, batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--gates-only", action="store_true")
    args = parser.parse_args()
    data, split, indices, atom, angle, clean_norm, norm, scales, batches = inputs()
    model = r2.build_variant(REFERENCE, clean_norm, norm, torch.device("cpu"))
    before = reachability(model, batches)
    r2.atomic_json(STUDY / "R2_REACHABILITY_BEFORE.json", before)
    print(json.dumps({k: v for k, v in before.items() if isinstance(v, int)}), flush=True)
    if not args.audit_only:
        summary = qualify(data, indices, atom, angle, clean_norm, norm, scales, batches, before, train=not args.gates_only)
        if summary is not None:
            r2.atomic_json(STUDY / "results/run_summary.json", summary)


if __name__ == "__main__":
    main()
