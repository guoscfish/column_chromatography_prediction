#!/usr/bin/env python3
"""Audit actual completed dialog trajectories without opening test labels."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from src.qgeognn_al.active_learning_v2.dialog_study import STUDY, SEEDS, METHODS, read, validate
from src.qgeognn_al.active_learning_v2.dialog_bridge import load_catalog, accepted_selection


def main():
    validate()
    rows, task_ids = [], set()
    task_ids_by_seed = {seed: set() for seed in SEEDS}
    previous_turn_count_by_task = {}
    for seed in SEEDS:
        split = pd.read_csv(STUDY / f"splits/row_seed_{seed}.csv")
        forbidden = set(split.loc[~split.role.isin(["l0", "u0"]), "sample_id"].astype(str))
        trajectory = read(STUDY / f"runtime/seed_{seed}/{METHODS[1]}/trajectory_freeze.json")
        assert trajectory["final_active_labels"] == 525
        for round_index in range(6):
            directory = STUDY / f"selections/seed_{seed}/{METHODS[1]}/round_{round_index:02d}"
            catalog = load_catalog(directory)
            packet, contract = read(directory / "input.json"), read(directory / "contract.json")
            accepted_selection(directory, catalog, packet)
            accepted = read(directory / "dialog_acceptance.json")
            task_ids.add(accepted["thread_id"])
            task_ids_by_seed[seed].add(accepted["thread_id"])
            assert set(catalog.observed) == set(contract["L_t_ids"])
            assert set(catalog.candidates) == set(contract["U_t_ids"]) - set(catalog.pending)
            assert not forbidden & (set(catalog.observed) | set(catalog.candidates) | set(catalog.pending))
            assert all("V1_ml" not in row and "V2_ml" not in row for row in list(catalog.candidates.values())+list(catalog.pending.values()))
            frozen = read(directory / "batch_freeze.json")
            assert len(frozen["batch_ids"]) == len(set(frozen["batch_ids"])) == 32
            assert frozen["batch_ids"] == trajectory["selected_ids"][round_index*32:(round_index+1)*32]
            predictions = {row["candidate_id"]: row for row in frozen["premeasurement_predictions"]}
            feedback = read(directory / "feedback.json")["records"]
            assert [row["candidate_id"] for row in feedback] == frozen["batch_ids"]
            for record in feedback:
                pred = predictions[record["candidate_id"]]
                for target in ("V1", "V2"):
                    assert pred[f"pred_{target}_ml"] == record[f"pred_{target}_ml"]
                # Prediction and truth values are persisted from float32 arrays;
                # recomputing the subtraction in Python can differ by one ULP.
                assert abs(record[f"error_{target}_ml"] - (pred[f"pred_{target}_ml"]-record[f"true_{target}_ml"])) < 1e-5
            audit = read(directory / "dialog_audit.json")
            previous = previous_turn_count_by_task.get(accepted["thread_id"], 0)
            calls = len(audit["turns"]) - previous
            previous_turn_count_by_task[accepted["thread_id"]] = len(audit["turns"])
            assert calls <= 16 and len(accepted["queries"]) <= 12 and len(accepted["viewed_candidate_ids"]) <= 240
            rows.append({"seed": seed, "round": round_index, "model_turns": calls,
                         "labels_before": len(catalog.observed), "batch_size": 32, "legal": True,
                         "forbidden_records_exposed": 0, "unobserved_responses_exposed": 0,
                         "native_tool_calls": audit["native_tool_calls"], "premeasurement_errors_verified": 32})
    # Seed 157 was explicitly recovered in a new independent task after the
    # original selector task hit a 401 before round five; no labels were
    # revealed during the handoff. Seed 6101 stayed on its original task.
    assert len(task_ids_by_seed[157]) == 2 and len(task_ids_by_seed[6101]) == 1
    (STUDY / "execution_boundary_audit.json").write_text(json.dumps(rows, indent=2)+"\n")
    print(json.dumps({"status": "PASS", "rounds": len(rows), "independent_tasks": len(task_ids),
                      "task_ids_by_seed": {str(seed): sorted(values) for seed, values in task_ids_by_seed.items()},
                      "feedback_rows_verified": sum(row["premeasurement_errors_verified"] for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
