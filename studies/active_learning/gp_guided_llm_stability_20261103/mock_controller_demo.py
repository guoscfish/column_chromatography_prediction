"""Write a reproducible synthetic 32-candidate mock trace without labels or APIs."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from arm_adapter import (MockTransport, add_mock_controller_feedback, assert_frozen_input_invariants,
                         build_arm_inputs, freeze_shared_inputs)
from controller import diagnose_batch, run_mock_controller, stable_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results/mock_controller")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_ids = [f"synthetic_{i:02d}" for i in range(40)]
    metadata = [{"id": value, "molecule_id": f"synthetic_molecule_{i}",
                 "conditions": {"loading": 1, "solvent": .5, "column": 4}}
                for i, value in enumerate(candidate_ids)]
    covariance = np.eye(40, dtype=np.float64)
    covariance[0, 1] = covariance[1, 0] = .999
    cfg = json.loads((Path(__file__).resolve().parent / "configs/controller_development.json").read_text())
    cfg["thresholds"].update(absolute_posterior_correlation=.99, maximum_unexempt_correlated_pairs=0)
    cfg["fixture_notice"] = "Synthetic unit fixture only; thresholds are not calibrated research policy."
    shared = freeze_shared_inputs(checkpoint_sha256=stable_hash("synthetic_checkpoint_descriptor_no_model_file"),
        labeled_ids=["synthetic_observed_0"], candidate_ids=candidate_ids,
        scientific_memory={"hypotheses": [], "synthetic_fixture": True},
        candidate_input=[{"id": value, "frozen_CW_percentile": i / 39} for i, value in enumerate(candidate_ids)],
        candidate_metadata=metadata, baseline_packet={"selection_count": 32, "synthetic_fixture": True},
        selector_protocol={"protocol": "mock hook only; replace with identical production validator in A/B/C"},
        llm_configuration={"model": "mock_no_llm", "parameters": {}, "budget": 0},
        tool_contract={"operations": [], "network": False})
    gp = {"synthetic_fixture": True, "latent_posterior_variance": np.diag(covariance).tolist(),
          "covariance_sha256": stable_hash(covariance.tolist()), "noise_variance": 1.0,
          "reliability_status": "synthetic_not_research_evidence"}
    inputs = build_arm_inputs(shared, gp)
    initial = {"choices": [{"id": value, "reason": "Synthetic initial mock proposal."}
                            for value in candidate_ids[:32]]}
    revised = copy.deepcopy(initial)
    revised["choices"][1] = {"id": candidate_ids[32], "reason": "Mock removes a high-correlation non-control pair."}
    revised["revision_reason"] = "Synthetic response to the exact numerical pair diagnostic."
    transports = {"A": MockTransport([initial]), "B": MockTransport([initial]), "C": MockTransport([initial, revised])}
    initial_proposals = {arm: transports[arm].respond(inputs[arm]) for arm in ("A", "B", "C")}

    def shared_mock_validate(proposal):
        # The mock demonstrates hook identity only; production validation is not reimplemented.
        if len(proposal["choices"]) != 32:
            raise ValueError("Mock fixture must contain a complete 32-choice proposal")

    for arm in ("A", "B", "C"):
        shared_mock_validate(initial_proposals[arm])

    def revise(proposal, feedback):
        nonlocal inputs
        inputs = add_mock_controller_feedback(inputs, feedback)
        return transports["C"].respond(inputs["C"])

    trace = run_mock_controller(initial_proposals["C"], candidate_ids=candidate_ids,
        shared_validate=shared_mock_validate, revise=revise,
        diagnose=lambda idx: diagnose_batch(covariance, 1.0, idx, config=cfg,
                                            reference_batch_indices=list(range(8, 40)), metadata=metadata))
    report = {"synthetic_fixture": True, "real_GP_reliability_evidence": False,
              "paid_LLM_calls": 0, "new_labels_revealed": 0,
              "initial_proposals": initial_proposals, "trace": trace,
              "mock_transport_calls": {arm: transport.calls for arm, transport in transports.items()}}
    frozen_check = assert_frozen_input_invariants(inputs)
    frozen_check.update(synthetic_fixture=True, initial_B_C_model_input_identical=(
        transports["B"].calls[0]["input_sha256"] == transports["C"].calls[0]["input_sha256"]))
    trace_path = args.output_dir / "mock_trace.json"
    check_path = args.output_dir / "frozen_input_check.json"
    trace_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    check_path.write_text(json.dumps(frozen_check, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"synthetic_fixture": True, "trace": str(trace_path), "frozen_input_check": str(check_path),
                      "final_status": trace["final_status"], "revision_requested": trace["revision_requested"],
                      "initial_IG_nats": trace["attempts"][0]["numerical_diagnostics"]["conditional_information_gain"]["value_nats"],
                      "final_IG_nats": trace["attempts"][-1]["numerical_diagnostics"]["conditional_information_gain"]["value_nats"]},
                     indent=2))


if __name__ == "__main__":
    main()
