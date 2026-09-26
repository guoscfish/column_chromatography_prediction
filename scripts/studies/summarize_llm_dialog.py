#!/usr/bin/env python3
"""Post-evaluation diagnostics; never used by the experiment selector."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from src.qgeognn_al.active_learning_v2.dialog_study import STUDY, SEEDS, METHODS, CONDITION_FIELDS, read
from src.qgeognn_al.active_learning_v2.benchmark_protocol import load_features


def main():
    if not (STUDY / "global_pre_test_freeze.json").exists():
        raise RuntimeError("only run after global freeze and unified evaluation")
    result = STUDY / "results"
    curves = pd.read_csv(result / "learning_curves.csv")
    curves.drop(columns="seed").groupby(["method", "budget"]).mean().to_csv(result / "mean_learning_curves.csv")
    features = load_features().set_index("sample_id")
    source_rows, contrasts, calls = [], [], []
    for seed in SEEDS:
        for method in METHODS:
            for round_index in range(6):
                directory = STUDY / f"selections/seed_{seed}/{method}/round_{round_index:02d}"
                before = read(directory / "observed_records.json")["records"]
                known_smiles = {row["smiles"] for row in before}
                feedback = read(directory / "feedback.json")["records"]
                enriched = []
                for row in feedback:
                    feature = features.loc[row["candidate_id"]]
                    enriched.append({**row, "smiles": str(feature.canonical_smiles),
                                     "conditions": {key: feature[key] for key in CONDITION_FIELDS}})
                for source in sorted({row["source"] for row in enriched}):
                    chosen = [row for row in enriched if row["source"] == source]
                    smiles = [row["smiles"] for row in chosen]
                    mols = [Chem.MolFromSmiles(value) for value in smiles]
                    row = {"seed": seed, "method": method, "round": round_index, "source": source,
                           "selected_rows": len(chosen), "new_molecule_rows": sum(value not in known_smiles for value in smiles),
                           "known_molecule_rows": sum(value in known_smiles for value in smiles),
                           "distinct_molecules": len(set(smiles)),
                           "within_source_same_molecule_pairs": sum(smiles.count(value)*(smiles.count(value)-1)//2 for value in set(smiles)),
                           "selected_distinct_PE_EA": len({value["conditions"]["PE/EA"] for value in chosen})}
                    for name, function in [("MW", Descriptors.MolWt), ("LogP", Descriptors.MolLogP), ("TPSA", Descriptors.TPSA)]:
                        values = [function(mol) for mol in mols]
                        row.update({f"{name}_min": min(values), f"{name}_max": max(values), f"{name}_mean": sum(values)/len(values)})
                    source_rows.append(row)
                for row in enriched:
                    if row["source"] != "LLM":
                        continue
                    for known in before:
                        if known["smiles"] != row["smiles"]:
                            continue
                        differences = [field for field in CONDITION_FIELDS if known["conditions"][field] != row["conditions"][field]]
                        if len(differences) != 1:
                            continue
                        field = differences[0]
                        contrasts.append({"seed": seed, "round": round_index, "selected_id": row["candidate_id"],
                                          "observed_id": known["id"], "smiles": row["smiles"], "changed_field": field,
                                          "observed_setting": known["conditions"][field], "selected_setting": row["conditions"][field],
                                          "observed_V1_ml": known["V1_ml"], "observed_V2_ml": known["V2_ml"],
                                          "new_V1_ml": row["true_V1_ml"], "new_V2_ml": row["true_V2_ml"],
                                          "pred_V1_ml": row["pred_V1_ml"], "pred_V2_ml": row["pred_V2_ml"],
                                          "reason": row["reason"]})
                if method == METHODS[1]:
                    accepted = read(directory / "dialog_acceptance.json")
                    previous = read(directory / "dialog_binding.json")["usage_before"]
                    call = {"seed": seed, "round": round_index, "model": accepted["model"], "effort": accepted["effort"],
                            "assistant_answers": accepted["assistant_answers_this_round"], "queries": len(accepted["queries"]),
                            "viewed_candidates": len(accepted["viewed_candidate_ids"]), "native_tool_calls": accepted["native_tool_calls"],
                            "rejected_outputs": len(list(directory.glob("rejected_selection_*.json")))}
                    for field in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens"):
                        call[field] = accepted["usage_cumulative"].get(field, 0) - previous.get(field, 0)
                    calls.append(call)
    pd.DataFrame(source_rows).to_csv(result / "source_coverage.csv", index=False)
    pd.DataFrame(contrasts).to_csv(result / "matched_condition_contrasts.csv", index=False)
    pd.DataFrame(calls).to_csv(result / "dialog_calls.csv", index=False)
    print(json.dumps({"source_rows": len(source_rows), "matched_condition_contrasts": len(contrasts),
                      "dialog_rounds": len(calls)}, indent=2))


if __name__ == "__main__":
    main()
