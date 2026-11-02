"""Selection-only historical identity checks. Never fits or reveals test targets."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import time

import numpy as np
import pandas as pd
import torch

from . import same_state_branching_cw_hybrid as s
from .coverage import extract_representations
from .sequential_acquisition import v2_hybrid_select_current
from .runner import predict_outputs


AUDIT_TABLES = (
    "cw_selection_regression_audit.csv", "hybrid_selection_regression_audit.csv",
    "kernel_ivr_selection_regression_audit.csv", "gradient_maxdet_selection_regression_audit.csv",
    "historical_hybrid_artifact_inventory.csv", "selector_identity_and_overlap_audit.csv",
    "gradient_transform_audit.csv", "anchor_state_diagnostics_audit.csv",
)


@contextmanager
def audit_firewall():
    """Fail closed at the training and label-store boundaries; persist real accesses."""
    accesses = []
    original = s.RestrictedLabelStore.reveal
    def reveal(store, ids, purpose):
        ids = list(ids)
        roles = {store.roles.get(value) for value in ids}
        if purpose != "initial_fit" or not roles <= {"l0", "validation"}:
            raise PermissionError("selector audit permits only L0/validation labels")
        result = original(store, ids, purpose)
        accesses.append(dict(store.audit[-1]))
        return result
    def no_fit(*args, **kwargs):
        raise RuntimeError("model training is forbidden during selector audit")
    with patch.object(s.RestrictedLabelStore, "reveal", reveal), \
         patch.object(s, "fit_from_same_initialization", no_fit), \
         patch.object(torch.optim.Adam, "step", no_fit):
        yield accesses


def _row(seed, budget, strategy, historical, proposed, checkpoint, features, **extra):
    return {"seed": seed, "budget": budget, "strategy": strategy,
            "historical_selected_ids_hash": s.stable_hash(historical),
            "new_selected_ids_hash": s.stable_hash(proposed),
            "intersection_count": len(set(historical) & set(proposed)),
            "exact_match": historical == proposed,
            "gradient_bank_hash": s.array_hash(features),
            "checkpoint_sha256": s.sha256_file(checkpoint),
            "test_truth_access_count": 0, **extra}


def _save(name, rows, *, require_exact=True):
    frame = pd.DataFrame(rows)
    path = s.STUDY / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    if require_exact and (frame.empty or not frame.exact_match.all()):
        raise RuntimeError(f"historical ordered selection identity failed: {path}")
    return frame


def cw_regression(lineages, protected):
    rows = []
    for lineage in lineages:
        if lineage["source_trajectory"] != "center_width_lcmd":
            continue
        seed, budget = lineage["seed"], lineage["anchor_budget"]
        continuation = "qgeognn_v2_row_cw_lcmd_to_525" if budget == 429 else "qgeognn_v2_row_cw_lcmd_to_1005"
        directory = s.STUDY.parent / continuation / f"runtime/seed_{seed}/center_width_lcmd/round_{s._round_for_budget(budget):02d}"
        artifacts = directory / "acquisition_artifacts"
        old_path = artifacts / "current_center_width_gradient_features.npz"
        receipt_path = old_path.with_suffix(".npz.contract.json")
        receipt = s.read_json(receipt_path)
        if receipt["sha256"] != s.sha256_file(old_path):
            raise RuntimeError("historical CW bank failed checksum")
        contract = receipt["contract"]
        checkpoint = s.ROOT / lineage["checkpoint_path"]
        current = lineage["labeled_indices"] + lineage["unlabeled_indices"]
        if (contract["checkpoint_sha256"] != s.sha256_file(checkpoint)
                or contract["ordered_current_indices"] != current
                or contract["sketch_seed"] != s.sketch_seed(seed)):
            raise RuntimeError("CW continuation is not the exact anchor state")
        banks = s.load_anchor_banks(lineage)
        lpos, upos = s._positions(lineage)
        new = banks["center_width"][np.r_[lpos, upos]]
        new_contract = s.read_json((s.ROOT / lineage["cw_gradient_path"]).with_suffix(".npz.contract.json"))["contract"]
        if new_contract["transform_audit"] != contract["transform"]:
            raise RuntimeError("historical CW L0 transform mismatch")
        with np.load(old_path) as values:
            if not np.array_equal(values["canonical_indices"], current):
                raise RuntimeError("historical CW order mismatch")
            identical_bank = np.array_equal(new, values["features"])
            max_error = float(np.max(np.abs(new - values["features"])))
        selected_path = artifacts / "selected_next_batch.csv"
        historical = pd.read_csv(selected_path).sample_id.astype(str).tolist()
        acquisition = s.read_json(artifacts / "contract.json")
        if acquisition["selected_table_sha256"] != s.sha256_file(selected_path):
            raise RuntimeError("historical CW selected batch checksum changed")
        proposed = s.select_batch("center_width_lcmd", banks, lineage)["selected_ids"]
        rows.append(_row(seed, budget, "center_width_lcmd", historical, proposed, checkpoint, new,
                         transform_hash=s.stable_hash(contract["transform"]), gradient_array_exact_match=identical_bank,
                         gradient_max_abs_error=max_error, historical_selection_path=str(selected_path.relative_to(s.ROOT))))
        protected.extend([old_path, receipt_path, selected_path, artifacts / "contract.json"])
    return _save("cw_selection_regression_audit.csv", rows)


def hybrid_regression(lineages, protected):
    """Reproduce source Hybrid only; this does not make it an executable primary arm."""
    rows, inventory = [], []
    for lineage in lineages:
        seed, budget, source = lineage["seed"], lineage["anchor_budget"], lineage["source_trajectory"]
        directory, _ = s._source_paths(source, seed, budget)
        artifacts = directory / "acquisition_artifacts"
        paths = [artifacts / f"ensemble_member_{member}" / "best.pt" for member in (1, 2)]
        inventory.append({"seed": seed, "budget": budget, "source_trajectory": source,
                          "member_1_checkpoint_available": paths[0].exists(),
                          "member_2_checkpoint_available": paths[1].exists(),
                          "required_extra_fits_at_anchor": sum(not path.exists() for path in paths)})
        if source != "hybrid":
            continue
        checkpoint = s.ROOT / lineage["checkpoint_path"]
        acquisition_contract = s.read_json(artifacts / "contract.json")
        for relative, digest in acquisition_contract["files"].items():
            if s.sha256_file(directory / relative) != digest:
                raise RuntimeError("Hybrid acquisition artifact checksum changed")
        source_input = acquisition_contract["input"]
        if (source_input["evaluation_checkpoint_sha256"] != s.sha256_file(checkpoint)
                or source_input["L_t_ids_hash"] != lineage["L_set_hash"]
                or source_input["U_t_ids_hash"] != lineage["U_set_hash"]):
            raise RuntimeError("Hybrid source checkpoint mismatch")
        current = lineage["labeled_indices"] + lineage["unlabeled_indices"]
        atom, angle = torch.load(s.HYBRID_STUDY / f"runtime/seed_{seed}/scrubbed_graphs.pt", weights_only=False)
        model = s.load_predictor_checkpoint(checkpoint)
        representations = extract_representations(model, atom, angle, current)
        representation_path = artifacts / "current_representations.npz"
        receipt = s.read_json(representation_path.with_suffix(".npz.contract.json"))
        if not s.verify_cache(representation_path, receipt["contract"]):
            raise RuntimeError("historical Hybrid representation checksum failed")
        with np.load(representation_path) as values:
            if not np.array_equal(values["canonical_indices"], current):
                raise RuntimeError("Hybrid representation ordering mismatch")
            latent_exact = np.array_equal(representations, values["features"])
        pred0, order = predict_outputs(model, atom, angle, lineage["unlabeled_indices"])
        if list(order) != lineage["unlabeled_indices"]:
            raise RuntimeError("Hybrid member0 prediction order mismatch")
        predictions = [pred0]
        for member, path in zip((1, 2), paths):
            audit_path = path.parent / "fit_audit.json"
            audit = s.read_json(audit_path)
            if (audit["train_rows"] != budget or audit["checkpoint_sha256"] != s.sha256_file(path)
                    or audit["initialization_seed"] != s.initialization_seed(seed, member)
                    or audit["test_labels_used_for_fit_or_checkpoint_selection"] != 0):
                raise RuntimeError("Hybrid ensemble checkpoint provenance failed")
            # Reuse the historical stored float64 CSV exactly as its selector did.
            prediction_path = path.parent / "predictions.csv.gz"
            table = pd.read_csv(prediction_path)
            if table.sample_id.astype(str).tolist() != lineage["unlabeled_ids"]:
                raise RuntimeError("Hybrid member prediction identity mismatch")
            predictions.append(table.drop(columns="sample_id").to_numpy(float))
            protected.extend([path, audit_path, prediction_path])
        context = s.read_json(s.HYBRID_STUDY / f"runtime/seed_{seed}/context.json")["contract"]
        scales = tuple(context["preprocessing"]["target_scales"][key] for key in ("V1", "V2"))
        result = v2_hybrid_select_current(representations, np.stack(predictions), budget, scales, s.BATCH_SIZE)
        proposed = np.asarray(lineage["unlabeled_ids"])[result["selected_pool_positions"]].tolist()
        selection_path = artifacts / "selected_next_batch.csv"
        historical = pd.read_csv(selection_path).sample_id.astype(str).tolist()
        if s.stable_hash(historical) != acquisition_contract["selected_ids_ordered_hash"]:
            raise RuntimeError("Hybrid historical selected order changed")
        rows.append(_row(seed, budget, "historical_hybrid_exact_source_only", historical, proposed, checkpoint,
                         representations, gradient_bank_hash=None, latent_bank_hash=s.array_hash(representations),
                         latent_array_exact_match=latent_exact,
                         shortlist_size=result["shortlist_size"], historical_selection_path=str(selection_path.relative_to(s.ROOT))))
        protected.extend([representation_path, representation_path.with_suffix(".npz.contract.json"), selection_path,
                          artifacts / "contract.json", directory / "contract.json"])
    _save("historical_hybrid_artifact_inventory.csv", inventory, require_exact=False)
    return _save("hybrid_selection_regression_audit.csv", rows)


def ordinary_regression(strategy, protected):
    """Fresh checkpoint gradients and branch API versus frozen IVR/MaxDet batches."""
    rows = []
    for seed in s.SEEDS:
        budget, round_index = 429, 3
        base = s.IVR_STUDY if strategy == "kernel_ivr" else s.STUDY.parent / "qgeognn_v2_row_maxdet_b32"
        directory = base / f"runtime/seed_{seed}" / (f"round_{round_index:02d}" if strategy == "kernel_ivr" else f"gradient_maxdet/round_{round_index:02d}")
        if strategy == "kernel_ivr":
            inputs = s.read_json(directory / "input.json")
            labeled, unlabeled = inputs["ordered_labeled_indices"], inputs["ordered_unlabeled_indices"]
            old_path = directory / "gradient_features.npz"
            historical_path = directory / "selection.json"
            historical = s.read_json(historical_path)["selected_ids"]
            receipt_path = old_path.with_suffix(".npz.contract.json")
            contract = s.read_json(receipt_path)["contract"]
            if not s.verify_cache(old_path, contract):
                raise RuntimeError("historical IVR cache checksum failed")
            s.assert_hashes(s.read_json(directory / "freeze.json")["files"])
            protected.extend([directory / "input.json", directory / "freeze.json"])
        else:
            state = pd.read_csv(directory / "state.csv")
            labeled = state.loc[state.role.eq("labeled"), "canonical_index"].astype(int).tolist()
            unlabeled = state.loc[state.role.eq("unlabeled"), "canonical_index"].astype(int).tolist()
            old_path = directory / "acquisition/current_gradient_features.npz"
            receipt_path = directory / "acquisition/current_gradient_features.contract.json"
            contract = s.read_json(receipt_path)
            historical_path = directory / "acquisition/selected_next_batch.csv"
            historical = pd.read_csv(historical_path).sample_id.astype(str).tolist()
            frozen = s.read_json(directory / "round_freeze.json")
            acquisition = s.read_json(directory / "acquisition/contract.json")
            if (s.stable_hash(historical) != acquisition["selected_ids_ordered_hash"]
                    or s.sha256_file(historical_path) != acquisition["selected_table_sha256"]
                    or contract["checkpoint_sha256"] != frozen["checkpoint_sha256"]):
                raise RuntimeError("MaxDet historical selection provenance mismatch")
            protected.extend([directory / "state.csv", directory / "round_freeze.json", directory / "acquisition/contract.json"])
        checkpoint = directory / "model/best.pt"
        if contract["checkpoint_sha256"] != s.sha256_file(checkpoint):
            raise RuntimeError("historical ordinary checkpoint mismatch")
        partition = s._partition(seed)
        outer = partition.loc[partition.role.eq("l0"), "canonical_index"].astype(int).tolist() + partition.loc[partition.role.eq("u0"), "canonical_index"].astype(int).tolist()
        lids, uids = s._id_lists(partition, labeled, unlabeled)
        state = {"outer_indices": outer, "labeled_indices": labeled, "unlabeled_indices": unlabeled,
                 "labeled_ids": lids, "unlabeled_ids": uids}
        context_path = s.HYBRID_STUDY / f"runtime/seed_{seed}/context.json"
        context = s.read_json(context_path)["contract"]
        preprocessing = context["preprocessing"]
        _, _, _ = s._materialize_anchor_gradient(seed, strategy, budget, checkpoint.parent, outer, preprocessing)
        fresh_path = s.STUDY / f"runtime/anchor_features/seed_{seed}_{strategy}_{budget}.npz"
        with np.load(fresh_path) as values:
            fresh = values["features"].copy()
        with np.load(old_path) as values:
            positions = {int(value): i for i, value in enumerate(values["canonical_indices"])}
            old = values["features"][[positions[i] for i in outer]]
        proposed = s.select_batch(strategy, {"ordinary_q50": fresh}, state)["selected_ids"]
        rows.append(_row(seed, budget, strategy, historical, proposed, checkpoint, fresh,
                         gradient_array_exact_match=np.array_equal(fresh, old),
                         gradient_max_abs_error=float(np.max(np.abs(fresh-old))),
                         transform_hash=s.stable_hash(preprocessing["target_scales"]),
                         historical_selection_path=str(historical_path.relative_to(s.ROOT))))
        protected.extend([checkpoint, checkpoint.parent / "fit_audit.json", old_path, receipt_path,
                          historical_path, context_path, fresh_path, fresh_path.with_suffix(".npz.contract.json")])
    return _save(f"{strategy}_selection_regression_audit.csv", rows)


def run():
    if (s.STUDY / "seal.json").exists():
        s.validate_seal()
        return s.read_json(s.STUDY / "selector_audit.json")
    torch.set_num_threads(2)
    started = time.perf_counter()
    with audit_firewall() as accesses:
        lineages = []
        for seed in s.SEEDS:
            for source in s.SOURCES:
                for budget in s.ANCHOR_BUDGETS:
                    lineage = s.inspect_anchor(seed, source, budget)
                    s.write_json_once(s._lineage_path(seed, source, budget), lineage)
                    lineages.append(lineage)
        protected = []
        print("CW regression", flush=True)
        cw = cw_regression(lineages, protected)
        print("Hybrid source identity regression", flush=True)
        hybrid = hybrid_regression(lineages, protected)
        for strategy in ("kernel_ivr", "gradient_maxdet"):
            print(strategy, "regression", flush=True)
            ordinary_regression(strategy, protected)
        overlaps, diagnostics, proposals = [], [], []
        for lineage in lineages:
            selected, diagnostic = s._anchor_selections(lineage, persist=False)
            repeated, _ = s._anchor_selections(lineage, persist=False)
            if selected != repeated:
                raise RuntimeError("primary selector proposals not deterministic")
            diagnostics.append(diagnostic)
            proposals.append({"seed": lineage["seed"], "source_trajectory": lineage["source_trajectory"],
                              "budget": lineage["anchor_budget"], "selections": selected})
            frame = cw if lineage["source_trajectory"] == "center_width_lcmd" else hybrid
            row = frame.loc[(frame.seed == lineage["seed"]) & (frame.budget == lineage["anchor_budget"])].iloc[0]
            historical = pd.read_csv(s.ROOT / row.historical_selection_path).sample_id.astype(str).tolist()
            for strategy, proposal in selected.items():
                intersection = len(set(historical) & set(proposal["selected_ids"]))
                overlaps.append({"seed": lineage["seed"], "budget": lineage["anchor_budget"],
                                 "source_trajectory": lineage["source_trajectory"], "proposal_strategy": strategy,
                                 "checkpoint_sha256": lineage["checkpoint_sha256"],
                                 "ordered_L_hash": lineage["ordered_L_hash"], "ordered_U_hash": lineage["ordered_U_hash"],
                                 "historical_selected_ids_hash": s.stable_hash(historical),
                                 "same_name_exact_selector_ids_hash": row.new_selected_ids_hash,
                                 "same_name_exact_match": bool(row.exact_match),
                                 "proposal_ids_hash": s.stable_hash(proposal["selected_ids"]),
                                 "intersection_count": intersection, "jaccard": intersection / (64-intersection),
                                 "ordered_proposal_equals_source": historical == proposal["selected_ids"]})

        _save("selector_identity_and_overlap_audit.csv", overlaps, require_exact=False)
        _save("anchor_state_diagnostics_audit.csv", diagnostics, require_exact=False)
        _save("state_diagnostics.csv", diagnostics, require_exact=False)
        s.atomic_json(s.STUDY / "results/anchor_selection_proposals.json", {"anchors": proposals})
        transform_rows = []
        for lineage in lineages:
            for key, representation in (("gradient_path", "ordinary_q50"), ("cw_gradient_path", "L0_center_width")):
                path = s.ROOT / lineage[key]
                contract = s.read_json(path.with_suffix(".npz.contract.json"))["contract"]
                audit = s.read_json(path.with_name(path.stem + "_audit.json"))
                transform_rows.append({"seed": lineage["seed"], "source": lineage["source_trajectory"],
                                       "budget": lineage["anchor_budget"], "representation": representation,
                                       "checkpoint_sha256": contract["checkpoint_sha256"],
                                       "transform_hash": s.stable_hash(contract.get("transform_audit", contract["endpoint_scales"])),
                                       "sketch_seed": contract["sketch_seed"], "dimension": contract["dimension"],
                                       "sketch_mapping_sha256": audit["sketch_mapping_sha256"],
                                       "test_truth_access_count": 0})
        _save("gradient_transform_audit.csv", transform_rows, require_exact=False)
        s.atomic_json(s.STUDY / "results/selector_audit_label_access.json", {"accesses": accesses, "test_truth_access_count": 0})
        source_files = s.hashes(protected)
        for lineage in lineages:
            source_files.update(lineage["source_files"])
        record = {"status": "PASSED_EXACT_ORDERED_SELECTOR_AUDIT", "new_fits": 0, "test_truth_access_count": 0,
                  "elapsed_seconds": time.perf_counter()-started, "strategies": list(s.STRATEGIES),
                  "cw_exact_states": len(cw), "hybrid_exact_source_states": len(hybrid),
                  "ivr_exact_states": len(s.SEEDS), "maxdet_exact_states": len(s.SEEDS),
                  "source_files": source_files,
                  "files": s.hashes([Path(__file__), Path(s.__file__), *[s.STUDY / "results" / name for name in AUDIT_TABLES],
                                    s.STUDY / "results/anchor_selection_proposals.json",
                                    s.STUDY / "results/selector_audit_label_access.json"])}
        s.atomic_json(s.STUDY / "selector_audit.json", record)
    return record
