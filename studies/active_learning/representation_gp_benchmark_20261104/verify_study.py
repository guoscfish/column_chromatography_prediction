"""Verify saved research receipts without requesting any source response labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    plan=json.loads((HERE/"config.json").read_text())
    freeze=json.loads((HERE/"protocol_freeze.json").read_text())
    if sha(HERE/"config.json")!=freeze["config_sha256"]:raise ValueError("configuration drift")
    expected=dict(freeze["code_sha256"])
    amendments=HERE/"engineering_amendments.json"
    if amendments.exists():
        for a in json.loads(amendments.read_text()):expected.update(a["updated_code_sha256"])
    if any(sha(HERE/name)!=value for name,value in expected.items()):raise ValueError("unrecorded code drift")
    states=[];fit_count=0;metric_count=0
    from representations import audit_lcmd_state
    for seed in plan["seeds"]:
        for budget in plan["budgets"]:
            key=f"seed_{seed}_L{budget}"
            output=HERE/"results"/key
            if not (output/"complete.json").exists():continue
            marker=json.loads((output/"complete.json").read_text())
            metrics=json.loads((output/"metrics.json").read_text())
            fits=json.loads((output/"fits.json").read_text())
            acq=json.loads((output/"acquisitions.json").read_text())
            meta=json.loads((HERE/"cache"/(key+".json")).read_text())
            if audit_lcmd_state(HERE.parents[2],seed,budget).audit!=meta["state"]:
                raise ValueError("historical checkpoint/L/U/cache identity changed after extraction")
            if marker["endpoint_fits"]!=len(fits) or len(metrics)!=len(fits) or len(acq)!=len(fits):
                raise ValueError("incomplete paired state outputs")
            if len(fits)!=36:raise ValueError("missing preregistered model arm")
            if any(f["training_rows"]!=budget for f in fits):raise ValueError("wrong training budget")
            if any(m["n"]!=416 or m["evaluation_role"]!="previously_used_validation_development" for m in metrics):
                raise ValueError("evaluation role drift")
            if any(f["dtype"]!="float64" for f in fits):raise ValueError("wrong GP numeric dtype")
            if any(m["test_truth_access_count"]!=0 for m in [meta["state"],meta["label_access"],marker]):
                raise ValueError("test access occurred")
            if len(meta["label_access"]["calls"])!=2 or meta["label_access"]["future_U_labels_read"]!=0:
                raise ValueError("unexpected response requests")
            if sha(HERE/"cache"/(key+".npz"))!=meta["cache_sha256"]:raise ValueError("feature cache drift")
            frozen_candidates=set(meta["candidate_ids"])
            if len(frozen_candidates)!=256:raise ValueError("candidate subset changed")
            for row in acq:
                if row["candidate_ids"]!=meta["candidate_ids"]:raise ValueError("unpaired acquisition candidates")
                for method in row["methods"].values():
                    ids=method["selected_ids"]
                    if len(ids)!=32 or len(set(ids))!=32 or not set(ids)<=frozen_candidates:
                        raise ValueError("illegal offline diagnostic batch")
            states.append([seed,budget]);fit_count+=len(fits);metric_count+=len(metrics)
    llm=json.loads((HERE/"cache/llm.json").read_text())
    if sha(HERE/"cache/llm.npz")!=llm["cache_sha256"]:raise ValueError("LLM cache drift")
    initial=json.loads((HERE/"llm_preflight_receipt.json").read_text())
    if initial["model_files"]!=llm["model_files"] or initial["encoder"]["model_commit"]!=llm["encoder"]["model_commit"]:
        raise ValueError("encoder/tokenizer changed between preflight and full")
    import numpy as np
    vectors=json.loads((HERE/"llm_preflight_vectors.json").read_text())
    with np.load(HERE/"cache/llm.npz",allow_pickle=False) as data:
        for name,digest in vectors["arrays"].items():
            if hashlib.sha256(data[name][:vectors["rows"]].tobytes()).hexdigest()!=digest:
                raise ValueError("incremental LLM cache changed an earlier frozen vector")
    audit=json.loads((HERE/"execution_audit.json").read_text())
    root=HERE.parents[2]
    if any(sha(root/name)!=digest for name,digest in audit["protected_reused_source_sha256"].items()):
        raise ValueError("protected production or previous study code changed")
    from summarize_results import load_completed
    _,_,summary_coverage,_=load_completed(HERE)
    result={"verified_states":states,"complete_matrix":len(states)==20,"endpoint_fits":fit_count,
        "metric_rows":metric_count,"test_truth_access_count":0,"future_U_label_requests":0,
        "new_gnn_training":0,"new_al_trajectories":0,"paid_api_calls":0,"model_downloads":0,
        "evaluation_status":"development evidence; shared validation used in checkpoint selection",
        "model":llm["encoder"]["model"],"core_code_hashes_verified":True,
        "encoder_tokenizer_identity_stable":True,"preflight_vector_prefix_bitwise_unchanged":True,
        "protected_production_sources_unchanged":True,
        "fit_log_metadata_repairs":summary_coverage.get("receipt_identity_repairs",[])}
    (HERE/"verification.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    return result


if __name__=="__main__":print(json.dumps(verify(),ensure_ascii=False,indent=2))
