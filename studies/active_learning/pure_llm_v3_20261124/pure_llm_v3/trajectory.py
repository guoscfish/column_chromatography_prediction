"""Separately authorized benchmark handoff from a frozen V3 decision to the next state."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import gzip
import math
import csv
from pathlib import Path

from .context_builder import QUANTILES, TARGETS, read_legacy_cards
from .frozen_state import FrozenState, SEEDS, SPLITS, load_historical, verified_split
from .records import digest, file_digest, once, read
from .transport_manager import round_lock, terminal_record


def round_directory(directory, round_index):
    return directory / f"round_{round_index:02d}"


def load_transition(root, directory, seed, round_index):
    if seed not in SEEDS or not 0 <= round_index <= 36:
        raise ValueError("unregistered V3 trajectory seed or round")
    if round_index == 0:
        return load_historical(root, seed, 0), None
    previous = round_directory(directory, round_index - 1)
    freeze = _accepted(previous, seed, round_index - 1)
    next_state = round_directory(directory, round_index)
    state_file = next_state / "source.json"
    receipt = read(next_state / "transition.json")
    if (file_digest(state_file) != receipt["state_file_sha256"]
            or file_digest(previous / "batch_freeze.json") != receipt["selection_freeze_sha256"]
            or receipt["memory_file_sha256"] != freeze["memory_file_sha256"]):
        raise ValueError("next-state transition lineage changed")
    state = FrozenState(**read(state_file))
    if (state.seed, state.round, state.checkpoint) != (seed, round_index, receipt["checkpoint_sha256"]):
        raise ValueError("next-state transition points to the wrong checkpoint or round")
    feedback_file = next_state / "authorized_feedback.json"
    l0, u0, split_sha = verified_split(root, seed)
    after_l = set(state.provenance["expected_observed"])
    after_u = set(state.provenance["expected_candidates"])
    acquired = [record["candidate_id"] for entry in state.history for record in entry["records"]]
    if (len(state.history) != round_index or len(acquired) != 32 * round_index
            or len(set(acquired)) != len(acquired) or not set(acquired) <= u0
            or after_l != l0 | set(acquired) or after_u != u0 - set(acquired)
            or state.provenance["split_sha256"] != split_sha
            or file_digest(feedback_file) != receipt["feedback_file_sha256"]
            or read(feedback_file) != state.history[-1] or acquired[-32:] != freeze["batch_ids"]):
        raise ValueError("next-state feedback or L/U lineage changed")
    fit = next_state / receipt["fit_path"]
    if (file_digest(fit / "best.pt") != state.checkpoint
            or file_digest(fit / "predictions.csv.gz") != receipt["prediction_sha256"]
            or file_digest(fit / "fit_audit.json") != state.provenance["fit_audit_sha256"]):
        raise ValueError("next-state fit artifact digest changed")
    return state, read(previous / "memory.json")


def _accepted(previous, seed, round_index):
    frozen = read(previous / "batch_freeze.json")
    if (
        frozen["mock"] or frozen["label_access"] is not False
        or frozen["binding"]["seed"] != seed or frozen["binding"]["round"] != round_index
        or frozen["binding"]["arm"] not in ("A6", "A7")
        or file_digest(previous / "selection.json") != frozen["selection_file_sha256"]
        or file_digest(previous / "memory.json") != frozen["memory_file_sha256"]
        or len(frozen["batch_ids"]) != 32 or len(set(frozen["batch_ids"])) != 32
    ):
        raise PermissionError("only an intact live V3 selection may trigger a transition")
    turn = previous / f"turn_{frozen['accepted_logical_index']:02d}"
    request = read(turn / "request.json")
    terminal = terminal_record(turn, frozen["accepted_attempt"])
    authorization = read(previous / "authorization.json")
    if (request["request_id"] != frozen["request_id"] or request["binding"] != frozen["binding"]
            or terminal["request_id"] != request["request_id"] or terminal["status"] != "completed"
            or terminal["response_id"] != frozen["provider_response_id"]
            or terminal["answer_sha256"] != digest(terminal["text"])
            or authorization["request_id"] != read(previous / "turn_00/request.json")["request_id"]
            or authorization["scope"] != "paid_selection_only"):
        raise ValueError("freeze is not backed by the recorded accepted provider response and authorization")
    return frozen


def _build_feedback(stage, frozen, truth_by_id, audit):
    selections = {stage.context.all_row_ids[c["candidate_id"]]: c for c in frozen["plan"]["choices"]}
    candidate_rows = {stage.context.all_row_ids[key]: card for key, card in stage.context.candidates.items()}
    records = []
    for sample_id in frozen["batch_ids"]:
        choice, card = selections[sample_id], candidate_rows[sample_id]
        record = {"candidate_id": sample_id, "source": "LLM", "reason": choice["reason"],
                  "hypothesis_id": choice["hypothesis_id"]}
        for index, target in enumerate(TARGETS):
            pred = card[f"pred_{target}_q50_ml"]
            truth = float(truth_by_id[sample_id][index])
            record.update({f"pred_{target}_ml": pred, f"true_{target}_ml": truth, f"error_{target}_ml": pred-truth})
        records.append(record)
    return {"seed": stage.context.seed, "round": stage.context.round, "method": "pure_llm_v3",
            "checkpoint_sha256": stage.context.checkpoint, "records": records, "label_store_audit": audit}


def _next_snapshot(state, feedback, fit, fit_dir, next_l, next_u):
    # Reuse the existing QGeoGNN scientific card enrichment for descriptors and novelty only.
    from src.qgeognn_al.active_learning_v2.scientist_full_delivery import encode_catalog
    from src.qgeognn_al.active_learning_v2.scientist_selector import ReadOnlyCatalog

    old_candidates, old_observed = read_legacy_cards(state)
    facts = {key: {k: row[k] for k in ("id", "smiles", "conditions")}
             for key, row in {**old_candidates, **old_observed}.items()}
    with gzip.open(fit_dir / "predictions.csv.gz", "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["sample_id", *(f"{t}_{q}" for t in TARGETS for q in ("q10", "q50", "q90"))]:
            raise ValueError("prediction output has unregistered fields")
        predictions = list(reader)
    if [row["sample_id"] for row in predictions] != next_u:
        raise ValueError("prediction artifact is not bound to the remaining ordered U")
    candidates = []
    for row in predictions:
        card = {**facts[row["sample_id"]], **{k: float(row[k[5:-3]]) for k in QUANTILES}}
        for t in TARGETS:
            card[f"pred_{t}_ml"] = card[f"pred_{t}_q50_ml"]
        candidates.append(card)
    history = [*state.history, feedback]
    acquired = {r["candidate_id"]: r for h in history for r in h["records"]}
    observed = []
    for key in next_l:
        card = dict(facts[key])
        if key in acquired:
            prior = acquired[key]
            for t in TARGETS:
                card.update({f"{t}_ml": prior[f"true_{t}_ml"],
                             f"premeasurement_prediction_{t}_ml": prior[f"pred_{t}_ml"],
                             f"premeasurement_error_{t}_ml": prior[f"error_{t}_ml"]})
            card["record_kind"] = "acquired_with_premeasurement_prediction"
        else:
            card.update({k: old_observed[key][k] for k in ("V1_ml", "V2_ml", "record_kind")})
        observed.append(card)
    new_round = state.round + 1
    catalog = ReadOnlyCatalog(candidates, observed, [], salt=f"pure-llm-v3:{state.seed}:{new_round}")
    packet = {"seed": state.seed, "round": new_round, "study_version": "pure_llm_v3",
              "active_label_count": len(next_l), "selection_count": 32,
              "objective": "overall Row V1/V2 predictive accuracy after retraining",
              "validation_test_records": 0, "delivery": "all_candidates_one_request",
              "candidate_count": len(next_u), "observed_count": len(next_l),
              **encode_catalog(catalog),
              "memory": {"previous_hypotheses": [], "previous_feedback_interpretation": "",
              "previous_batch_rationale": "", "previous_selected_ids": list(acquired),
              "recent_outcomes": [], "high_error_measured_examples": [], "memory_policy": "V3 host memory"}}
    packet["packet_sha256"] = digest(packet)
    provenance = {"split_sha256": state.provenance["split_sha256"],
                  "expected_observed": sorted(next_l), "expected_candidates": sorted(next_u),
                  "fit_audit_sha256": file_digest(fit_dir / "fit_audit.json")}
    return FrozenState(state.seed, new_round, fit["checkpoint_sha256"], packet, [], history, provenance)


def _fit(root, seed, directory, state, stage, frozen, *, device, fit_dir, feedback_file):
    # Imports and label-store construction are isolated from the entire selection path.
    import numpy as np
    import pandas as pd
    import torch

    from src.qgeognn_al.active_learning_v2.benchmark_protocol import initialization_seed, load_features, seed_config
    from src.qgeognn_al.active_learning_v2.maxdet_study import historical_round0_paths
    from src.qgeognn_al.active_learning_v2.protocol import RestrictedLabelStore, validate_row_protocol
    from src.qgeognn_al.active_learning_v2.runner import fit_from_same_initialization
    from src.qgeognn_al.resources import SOURCE_DATA
    from src.qgeognn_al.schemas.conditions import ConditionNormalization

    l0, u0, split_sha = verified_split(root, seed)
    partition = pd.read_csv(root / SPLITS / f"row_seed_{seed}.csv")
    data = load_features()
    validate_row_protocol(partition)
    if data.sample_id.tolist() != partition.sample_id.tolist() or split_sha != state.provenance["split_sha256"]:
        raise ValueError("frozen feature/split identity drift")
    current_observed, current_u = (set(state.provenance[f"expected_{k}"]) for k in ("observed", "candidates"))
    batch = frozen["batch_ids"]
    if (not l0 <= current_observed or not set(batch) <= current_u or current_observed & current_u
            or (current_observed | current_u) != l0 | u0):
        raise ValueError("frozen batch is outside its authorized U")
    prior_order = [row["candidate_id"] for h in state.history for row in h["records"]]
    initial_order = partition.loc[partition.role.eq("l0"), "sample_id"].tolist()
    if set(initial_order + prior_order) != current_observed:
        raise ValueError("training order does not match observed history")
    next_l = initial_order + prior_order + batch
    next_u = [key for key in partition.sample_id if key in current_u and key not in batch]
    validation = partition.loc[partition.role.eq("validation"), "sample_id"].tolist()
    index = dict(zip(partition.sample_id, partition.canonical_index, strict=True))
    paths = historical_round0_paths(seed)
    graph_path = paths["scrubbed_graphs"]
    source_contract = read(paths["context"])
    seal = read(graph_path.with_suffix(".pt.contract.json"))
    contract = source_contract["contract"]
    if (seal["contract"] != contract or source_contract["contract_hash"] != digest(contract)
            or seal["contract_hash"] != digest(contract) or seal["sha256"] != file_digest(graph_path)
            or contract["outer_seed"] != seed or contract["partition_hash"] != digest(partition.to_dict("list"))
            or contract["source_sha256"] != file_digest(SOURCE_DATA)):
        raise ValueError("historical scrubbed graph or predictor contract drift")
    atom, angle = torch.load(graph_path, map_location="cpu", weights_only=False)
    if any(torch.count_nonzero(item.y) for item in atom):
        raise ValueError("selector training graph base must be label-scrubbed")
    torch.set_num_threads(2)
    device_scope = nullcontext()
    if device == "cuda":
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        from scripts.studies.run_qgeognn_cuda_three_methods import cuda_adapter, configure_device, frozen_inputs
        frozen_inputs()
        device_scope = cuda_adapter(configure_device())
    fit_config = {**seed_config(seed), "split_sha256": digest(partition.to_dict("list"))}
    # No labels are allocated until artifacts, exact configuration and CUDA compatibility pass.
    store = RestrictedLabelStore(SOURCE_DATA, partition)
    store.freeze_acquisitions(prior_order + batch)
    if feedback_file.exists():
        feedback = read(feedback_file)
        audit = feedback["label_store_audit"]
        if ([r["candidate_id"] for r in feedback["records"]] != batch
                or audit["requested_ids_hash"] != digest(sorted(batch)) or audit["requested_rows"] != 32
                or audit["purpose"] != "after_acquisition_fit" or not audit["acquisitions_frozen"]
                or (feedback["seed"], feedback["round"], feedback["method"]) != (seed, state.round, "pure_llm_v3")
                or feedback["checkpoint_sha256"] != stage.context.checkpoint
                or feedback["selection_freeze_sha256"] != frozen["freeze_sha256"]):
            raise ValueError("previously recorded measurements do not match this freeze")
        truth = {row["candidate_id"]: [row[f"true_{target}_ml"] for target in TARGETS]
                 for row in feedback["records"]}
        expected = _build_feedback(stage, frozen, truth, audit)
        if (any(not math.isfinite(value) for values in truth.values() for value in values)
                or expected["records"] != feedback["records"]):
            raise ValueError("recorded acquisition predictions or signed errors drifted")
    else:
        revealed = store.reveal(batch, "after_acquisition_fit")
        feedback = _build_feedback(stage, frozen, dict(zip(batch, revealed, strict=True)), store.audit[-1])
        feedback["selection_freeze_sha256"] = frozen["freeze_sha256"]
        once(feedback_file, feedback)
    fit_truth = {
        row["candidate_id"]: [row[f"true_{target}_ml"] for target in TARGETS]
        for item in (*state.history, feedback) for row in item["records"]
    }
    l0_truth = store.reveal(initial_order, "initial_fit")
    by_id = dict(zip(initial_order, l0_truth, strict=True))
    by_id.update(fit_truth)
    val_truth = store.reveal(validation, "initial_fit")
    once(directory / "fit_initial_label_audit.json", store.audit[-2:])
    train_contract = {"method": "pure_llm_v3", "round": state.round + 1, "outer_seed": seed,
                      "device": device, "train_sample_ids": next_l, "validation_sample_ids": validation,
                      "selection_freeze_sha256": frozen["freeze_sha256"],
                      "prediction_role": "current_U_t", "test_truth_access_count": 0}
    with device_scope:
        fit = fit_from_same_initialization(atom_base=atom, angle=angle,
            normalization=ConditionNormalization(**contract["normalization"]),
            preprocessing=contract["preprocessing"], train_indices=[index[key] for key in next_l],
            train_truth=np.asarray([by_id[key] for key in next_l], dtype=np.float32),
            validation_indices=[index[key] for key in validation], validation_truth=val_truth,
            prediction_indices=[index[key] for key in next_u], prediction_sample_ids=next_u,
            initialization_seed=initialization_seed(seed), training_config=fit_config,
            contract=train_contract, runtime=fit_dir)
    if fit["initialization_hash"] != read(paths["member0_audit"])["initialization_hash"]:
        raise ValueError("fixed L333 initialization drift")
    return _next_snapshot(state, feedback, fit, fit_dir, next_l, next_u), fit


def advance(root, directory, seed, round_index, *, device, authorization):
    from .runner import _validation, stage_state

    if seed not in SEEDS or not 0 <= round_index < 36 or device not in ("cpu", "cuda"):
        raise ValueError("unregistered fit contract")
    selection = round_directory(directory, round_index)
    launch = read(directory / "trajectory.json")
    expected = {"scope": "reveal_and_fit_one_frozen_batch", "device": device,
                "selection_freeze_sha256": file_digest(selection / "batch_freeze.json")}
    if (launch["seed"] != seed or not isinstance(authorization, dict)
            or any(authorization.get(k) != v for k, v in expected.items())
            or not authorization.get("authorized_by") or not authorization.get("authorization_reference")):
        raise PermissionError("advance requires a human-authorized exact freeze and device")
    next_dir = round_directory(directory, round_index + 1)
    with round_lock(next_dir):
        once(directory / "fit_device.json", {"seed": seed, "device": device})
        if (next_dir / "transition.json").exists():
            return load_transition(root, directory, seed, round_index + 1)[0]
        state, snapshot = load_transition(root, directory, seed, round_index)
        stage = stage_state(root, state, arm=launch["arm"], memory_mode=launch["memory_mode"], snapshot=snapshot)
        frozen = _accepted(selection, seed, round_index)
        if frozen["binding"] != stage.binding:
            raise ValueError("frozen selection candidate state cannot be reconstructed")
        plan = read(selection / "selection.json")
        if not _validation(stage, plan).valid:
            raise ValueError("frozen proposal failed current full validation")
        ids = [stage.context.all_row_ids[c["candidate_id"]] for c in plan["choices"]]
        if ids != frozen["batch_ids"]:
            raise ValueError("freeze differs from selected IDs")
        frozen = {**frozen, "freeze_sha256": expected["selection_freeze_sha256"],
                  "plan": plan}
        once(next_dir / "fit_authorization.json", authorization)
        fits = sorted(next_dir.glob("fit_[0-9][0-9][0-9]"))
        complete = [path for path in fits if (path / "fit_audit.json").exists()]
        fit_dir = complete[-1] if complete else next_dir / f"fit_{len(fits):03d}"
        result, fit = _fit(root, seed, next_dir, state, stage, frozen, device=device, fit_dir=fit_dir,
                           feedback_file=next_dir / "authorized_feedback.json")
        once(next_dir / "source.json", asdict(result))
        once(next_dir / "transition.json", {**expected,
            "state_file_sha256": file_digest(next_dir / "source.json"), "fit_path": fit_dir.name,
            "feedback_file_sha256": file_digest(next_dir / "authorized_feedback.json"),
            "checkpoint_sha256": fit["checkpoint_sha256"], "prediction_sha256": fit["prediction_sha256"],
            "memory_file_sha256": frozen["memory_file_sha256"]})
        return result


def main():
    from .runner import ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    state = advance(ROOT, args.trajectory, args.seed, args.round, device=args.device,
                    authorization=read(args.authorization))
    print(f"FROZEN_NEXT_STATE seed={state.seed} round={state.round} checkpoint={state.checkpoint}")


if __name__ == "__main__":
    main()
