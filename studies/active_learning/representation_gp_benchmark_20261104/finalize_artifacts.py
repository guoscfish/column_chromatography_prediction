"""Export actual predictions and measured costs, without original label access."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def encoding_jobs(value,prefix=""):
    if isinstance(value,dict):
        if {"count","tokens","seconds"}<=set(value):
            yield {"encoding":prefix,**value}
        else:
            for name,child in value.items():yield from encoding_jobs(child,prefix+"/"+name)


def main():
    states=sorted(p.parent for p in (HERE/"results").glob("seed_*_L*/complete.json"))
    target=HERE/"summary/predictions.csv.gz"
    count=0
    with gzip.open(target,"wt",newline="",encoding="utf-8",compresslevel=6) as stream:
        writer=None
        for state in states:
            for row in json.loads((state/"predictions.json").read_text()):
                if writer is None:
                    writer=csv.DictWriter(stream,fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row);count+=1
    llm=json.loads((HERE/"cache/llm.json").read_text())
    metadata=[json.loads(p.read_text()) for p in sorted((HERE/"cache").glob("seed_*.json"))]
    feature={"states":len(metadata),"frozen_experiment_rows_union":llm["rows"],
        "feature_state_row_count":sum(r["representations"]["gnn"]["shape"][0] for r in metadata),
        "raw_cached_rows_reused":sum(r["representations"]["raw"]["historical_rows_reused"] for r in metadata),
        "raw_validation_rows_computed":sum(r["representations"]["raw"]["computed_rows"] for r in metadata),
        "cw_rows_computed":sum(r["representations"]["cw"]["computed_rows"] for r in metadata),
        "feature_preparation_seconds_total":sum(r["elapsed_seconds"] for r in metadata),
        "new_QGeoGNN_fits":0,"test_feature_indices_requested_per_state":0,"test_truth_requests":0,
        "data_policy":"current-state L_t and validation responses only; U embeddings are label free"}
    (HERE/"feature_preparation_audit.json").write_text(json.dumps(feature,indent=2)+"\n")
    summary=json.loads((HERE/"summary/summary.json").read_text())
    jobs=list(encoding_jobs(llm["encoding"]))
    phase_times={}
    for phase in ("preflight","full"):
        path=HERE/(phase+"_completion.json")
        if path.exists():phase_times[phase]=json.loads(path.read_text())["elapsed_seconds"]
    costs={"scope":"measured local offline execution; no paid API/model download/new GNN training",
        "states_completed":len(states),"prediction_rows":count,
        "prediction_file":"summary/predictions.csv.gz",
        "prediction_sha256":hashlib.sha256(target.read_bytes()).hexdigest(),
        "phase_wall_seconds":phase_times,"phase_wall_seconds_total":sum(phase_times.values()),
        "feature_preparation":feature,"language_model":llm["encoder"]["model"],
        "llm_cached_experiment_rows":llm["rows"],"language_encoding_jobs":jobs,
        "language_encoding_seconds_total":sum(j["seconds"] for j in jobs),
        "language_tokens_processed":sum(j["tokens"] for j in jobs),
        "encoding_count_interpretation":"text-deduplicated within each immutable preflight/extension chunk; repeated SMILES text across chunks may be computed twice, earlier row vectors preserved",
        "gp_training_seconds_total":sum(r["total_training_seconds"] for r in summary["optimizer"]),
        "gp_cost_breakdown":summary["optimizer"],"new_al_trajectories":0,"new_lab_measurements":0,
        "paid_generation_api_calls":0,"model_downloads":0}
    (HERE/"summary/costs.json").write_text(json.dumps(costs,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"completed_states":len(states),"exported_predictions":count,"measured_phase_seconds":phase_times},indent=2))


if __name__=="__main__":main()
