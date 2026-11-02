#!/usr/bin/env python3
"""Read-only saved-validation comparisons; no source responses or selector imports."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ENTRY_ROOT = Path(__file__).resolve().parents[2]
ROOT = ENTRY_ROOT.parents[3] if ENTRY_ROOT.name == "frozen_source" else ENTRY_ROOT
STUDY = ROOT / "studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101"
OUT = STUDY / "reports"
SEEDS = (157,887,2357,6101,12203)
ARM = "free_cw_sol61_xhigh_rerun32"
BASE = ROOT / "studies/active_learning/qgeognn_v2_row_sequential_b32"
FREV = BASE.parent / "qgeognn_v2_row_llm_scientist_v2/revisions"
NREV = BASE.parent / "qgeognn_v2_row_llm_nocw_ablation/revisions"
OLD = ((21,"sol_xhigh_hybrid_free_to1005_20261013"),(26,"sol_xhigh_all_strategies_to1165_20261014"),
       (30,"sol_xhigh_all_strategies_to1293_20261015"),(34,"sol_xhigh_all_strategies_to1421_20261017"),
       (36,"sol_xhigh_all_strategies_to1549_20261018"))
NEW = ((21,"sol61_xhigh_free_five_seeds_to1005_20261023"),(36,"sol61_xhigh_free_five_seeds_to1485_20261024"))
METHODS = ("new_free_cw_sol61_xhigh", "historical_free_mixed_model", "nocw", "numerical_hybrid32", "gradient_lcmd32")


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atomic(p, value):
    p.parent.mkdir(parents=True,exist_ok=True)
    temporary = p.with_name(p.name+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    temporary.replace(p)


def table(p, rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    temporary = p.with_name(p.name+".tmp")
    with temporary.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(p)


def model_directory(method, seed, r):
    if method == METHODS[0]:
        if r == 0:
            source = read(STUDY / f"offline_checks/initial_state_seed_{seed}.json")
            return Path(source["checkpoint_path"]).parent, source["NoCW_prediction_freeze"]
        if not (STUDY / f"progress/completed_seed_{seed}_round_{r-1:02d}.json").exists():
            return None
        directory = STUDY / f"runtime/seed_{seed}/{ARM}/round_{r:02d}"
    elif method == METHODS[1]:
        if (seed == 887 and r>33) or (seed==12203 and r>24):
            owner = FREV / "sol61_xhigh_free_to1485_extra3_20261026"
        else:
            owner = FREV / next(name for end,name in (OLD if seed in (157,6101) else NEW) if r<=end)
        directory = owner / f"runtime/seed_{seed}/free_llm32_scientist_v2/round_{r:02d}"
    elif method == "nocw":
        owner = NREV / ("five_seed_nocw_to1005_20261029" if r<=21 else "five_seed_nocw_to1389_20261101")
        directory = owner / f"runtime/seed_{seed}/llm32_nocw/round_{r:02d}"
    else:
        if method == "numerical_hybrid32" and r>21:
            return None
        owner = BASE if r<=21 else BASE.parent / "qgeognn_v2_row_lcmd_to1485_20261026"
        name = "hybrid" if method == "numerical_hybrid32" else "lcmd"
        directory = owner / f"runtime/seed_{seed}/{name}/round_{r:02d}"
        model = directory / "model"
        return (model,None) if (model/"fit_audit.json").exists() else None
    p = directory / "prediction_freeze.json"
    if not p.exists():
        return None
    frozen = read(p)
    return Path(frozen["checkpoint_path"]).parent, frozen


def saved_point(method, seed, r):
    located = model_directory(method,seed,r)
    if located is None:
        return None
    model,frozen = located
    audit = read(model/"fit_audit.json")
    if audit["train_rows"] != 333+32*r or audit["test_labels_used_for_fit_or_checkpoint_selection"] != 0:
        raise RuntimeError("saved fit budget/label barrier mismatch")
    # Current historical fits are immutable once audit/freeze exists.
    if sha(model/"best.pt") != audit["checkpoint_sha256"] or sha(model/"predictions.csv.gz") != audit["prediction_sha256"]:
        raise RuntimeError("saved predictor hash drift")
    if frozen and (frozen["test_truth_access_count"] != 0 or any(
        frozen[k] != audit[k] for k in ("checkpoint_sha256","prediction_sha256","initialization_hash"))):
        raise RuntimeError("prediction freeze mismatch")
    with (model/"history.csv").open(newline="") as handle:
        chosen = [row for row in csv.DictReader(handle) if int(row["epoch"])==audit["best_epoch"]]
    if len(chosen)!=1:
        raise RuntimeError("chosen validation epoch missing")
    metrics = {k:(v=="True" if v in ("True","False") else float(v))
               for k,v in chosen[0].items() if k.startswith("validation_")}
    if not all(math.isfinite(v) for v in metrics.values()):
        raise RuntimeError("nonfinite saved metric")
    return {"method":method,"seed":seed,"budget":333+32*r,**metrics,
        "checkpoint_sha256":audit["checkpoint_sha256"],"history_sha256":sha(model/"history.csv"),
        "source":str((model/"fit_audit.json").relative_to(ROOT))}


def aulc(points):
    if len(points)<2:
        return None
    return sum((a["validation_selection_score"]+b["validation_selection_score"])*.5*(b["budget"]-a["budget"])
        for a,b in zip(points,points[1:]))/(points[-1]["budget"]-points[0]["budget"])


def summarize(values):
    return {"mean":statistics.mean(values),"seed_std_ddof1":statistics.stdev(values) if len(values)>1 else None}


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    curves, by, completion = [], {}, []
    for method in METHODS:
        for seed in SEEDS:
            points=[]
            for r in range(37 if method!="nocw" else 34):
                point=saved_point(method,seed,r)
                if point is None:
                    break
                points.append(point)
            curves.extend(points)
            by[method,seed]=points
            completion.append({"method":method,"seed":seed,"completed_budget":points[-1]["budget"] if points else None,
                "completed_batches":max(0,len(points)-1),"raw_validation_AULC":aulc(points)})
    table(OUT/"validation_curves.csv",curves)
    table(OUT/"completion.csv",completion)
    paired, summaries = [], []
    for other in METHODS[1:]:
        common=min(min(len(by[METHODS[0],seed]),len(by[other,seed])) for seed in SEEDS)
        if common<2:
            continue
        endpoints=sorted({common,*[(b-333)//32+1 for b in (525,813,1005,1389,1485) if (b-333)//32+1<=common]})
        for endpoint in endpoints:
            budget=333+32*(endpoint-1)
            fixed_metrics=[k for k,v in by[METHODS[0],SEEDS[0]][endpoint-1].items()
                           if k.startswith("validation_") and not isinstance(v,bool)]
            for seed in SEEDS:
                new,old=by[METHODS[0],seed][:endpoint],by[other,seed][:endpoint]
                values={"raw_validation_AULC":(aulc(new),aulc(old))}
                values.update({k:(new[-1][k],old[-1][k]) for k in fixed_metrics})
                for metric,(a,b) in values.items():
                    paired.append({"comparator":other,"budget":budget,"seed":seed,"metric":metric,
                        "new_free":a,"reference":b,"paired_delta_new_minus_reference":a-b})
            for metric in ["raw_validation_AULC",*fixed_metrics]:
                group=[row for row in paired if row["comparator"]==other and row["metric"]==metric and row["budget"]==budget]
                summaries.append({"comparator":other,"budget":budget,"metric":metric,"seeds":list(SEEDS),
                    "new_free":summarize([row["new_free"] for row in group]),
                    "reference":summarize([row["reference"] for row in group]),
                    "paired_delta_new_minus_reference":summarize([row["paired_delta_new_minus_reference"] for row in group])})
    table(OUT/"paired_differences.csv",paired)
    atomic(OUT/"comparison_summary.json",{"updated_utc":datetime.now(timezone.utc).isoformat(),
        "historical_free_model_mixing":"157: Astra/high prefix through L525; 6101: through L653; one historical unknown accepted ID-copy record; 887/2357/12203 Sol/xhigh from L333",
        "comparisons":summaries,"budget_rule":"all five seeds jointly complete, independently for each comparator",
        "AULC":"normalized trapezoid of complete raw nonmonotone curve; no best-so-far",
        "causality":"declared model/effort unified; API randomness, runtime and invisible backend changes remain; no independent CW causal claim",
        "test_truth_reads":0,"selector_feedback":False})
    for budget in (525,813,1005,1389,1485):
        n=(budget-333)//32+1
        if all(len(by[METHODS[0],seed])>=n for seed in SEEDS):
            path=OUT/f"milestone_L{budget}.json"
            result={"budget":budget,"seeds":[{"seed":seed,"point":by[METHODS[0],seed][n-1],
                "raw_validation_AULC":aulc(by[METHODS[0],seed][:n])} for seed in SEEDS]}
            result["summary"]={key:summarize([row["point"][key] for row in result["seeds"]])
                for key,value in result["seeds"][0]["point"].items() if key.startswith("validation_") and not isinstance(value,bool)}
            result["summary"]["raw_validation_AULC"]=summarize([row["raw_validation_AULC"] for row in result["seeds"]])
            if path.exists() and read(path)!=result:
                raise RuntimeError("frozen milestone report drift")
            atomic(path,result)
    costs=[]
    for seed in SEEDS:
        for path in (STUDY/f"selections/seed_{seed}/{ARM}").glob("round_*/attempt_*.json"):
            if not __import__("re").fullmatch(r"attempt_\d+_\d+\.json",path.name):
                continue
            record=read(path)
            provenance=record.get("provenance",{})
            parts=path.stem.split("_")
            costs.append({"seed":seed,"round":int(path.parent.name.split("_")[1]),"answer_turn":int(parts[1]),
                "transport_attempt":int(parts[2]),"success":record["success"],"elapsed_seconds":record["elapsed_seconds"],
                "response_id":provenance.get("response_id"),"request_id":provenance.get("request_id"),
                "requested_model":"gpt-6.1-sol","reasoning_effort":"xhigh","served_model":provenance.get("served_model"),
                "usage":provenance.get("usage"),"ended_utc":record.get("ended_utc"),"source":str(path.relative_to(STUDY))})
    atomic(OUT/"costs.json",{"requests":costs,"request_count":len(costs),
        "transport_retries":sum(r["transport_attempt"]>0 for r in costs),
        "correction_requests":sum(r["answer_turn"]>0 for r in costs),
        "accepted_batches":len(list((STUDY/"selections").glob("seed_*/*/round_*/batch_freeze.json"))),
        "completed_fits":sum(row["completed_batches"] for row in completion if row["method"]==METHODS[0]),
        "monetary_price_verified":False,"currency_cost":None,"usage_retained_without_invented_price":True})
    lines=["# New CW-visible Free32 rerun", "", "Target: five seeds × 36 accepted batches, L333 → L1485, gpt-6.1-sol / xhigh.","",
        "| Method | Seed | Completed L | Raw validation AULC |","|---|---:|---:|---:|"]
    for row in completion:
        lines.append(f"| {row['method']} | {row['seed']} | {row['completed_budget']} | {row['raw_validation_AULC']} |")
    lines += ["", "Per-budget metrics: validation_curves.csv. Paired differences and seed means/std: paired_differences.csv and comparison_summary.json.",
        "Milestones are written only after all five new seeds complete that budget. Historical Free retains its mixed-model mark.",
        "Comparisons use each comparator's five-seed common budget. API randomness, dates and invisible backend changes remain; these are not strict causal estimates of CW.",
        "No report, score or learning curve is supplied to selection or scientific memory. No test labels are read."]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":"REPORT_UPDATED","new_free_completed_budgets":[row["completed_budget"] for row in completion if row["method"]==METHODS[0]],
        "comparisons":len(summaries),"real_requests_recorded":len(costs)}))


if __name__ == "__main__":
    run()
