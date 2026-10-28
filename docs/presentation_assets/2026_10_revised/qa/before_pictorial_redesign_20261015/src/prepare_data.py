"""Read saved metrics and protocols; freeze provenance without evaluating models.

No dataset labels, checkpoints, predictions, training modules or LLM calls are used.
Run this explicitly to re-audit repository sources. Plotting uses the saved tables.
"""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from style import ROOT, OUT, QA

DATA = OUT / "data"
SOURCES = DATA / "frozen_inputs"
SEEDS = (157, 6101)
MANIFEST = []
RECEIPTS = []
CHECKS = []
SNAPSHOT = "reports/ppt_202610_al/frozen_inputs/reports/active_learning_latest/snapshot.json"
PPT = "reports/ppt_202610_al/"
FREE = "studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/free_sol_xhigh_continuation_20261012/"
ASTRA = "studies/active_learning/qgeognn_v2_row_llm_scientist_v2/revisions/astra_high_full_pool_20261001/"


def check(name, condition):
    CHECKS.append({"check": name, "passed": bool(condition)})
    if not condition:
        raise ValueError(name)


def read(rel, alias):
    path = ROOT / rel
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    target = SOURCES / alias
    if target.is_file():
        check("frozen_source_unchanged/"+alias, target.read_bytes()==raw)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    MANIFEST.append({"repository_path": rel, "frozen_copy": str(target.relative_to(OUT)),
                     "sha256": digest, "bytes": len(raw)})
    return raw


def js(rel, alias):
    return json.loads(read(rel, alias))


def csv(rel, alias):
    return pd.read_csv(io.BytesIO(read(rel, alias)))


def table(frame, name):
    frame.to_csv(DATA / name, index=False, float_format="%.17g")


def add_receipt(path, record, digest, figure, seed, method, labels, nrmse):
    check(f"fit_metric/{figure}/{seed}/{method}/{labels}",
          abs(record["best_validation_combined_normalized_rmse"]-nrmse)<1e-12)
    check("no_test_labels/"+path, record["test_labels_used_for_fit_or_checkpoint_selection"]==0)
    exists = (ROOT/path).is_file()
    if exists:
        raw = read(path, "fit_receipts/"+digest+".json")
        check("retained_fit_hash/"+path, hashlib.sha256(raw).hexdigest()==digest)
    RECEIPTS.append({"figure": figure, "seed":seed, "method":method, "labels":labels,
                     "source_path":path, "sha256":digest, "original_retained":exists,
                     **record})
    return path


def curves_from_snapshot(snap, figure, methods, end, overview):
    rows = []
    for seed in SEEDS:
        for method in methods:
            pts = [p for p in snap["curves"][str(seed)][method] if p["labels"]<=end]
            check(f"budget_grid/{figure}/{seed}/{method}",
                  [p["labels"] for p in pts]==list(range(333,end+1,32)))
            for p in pts:
                labels,value=p["labels"],p["nrmse"]
                if method=="CW16 + LLM16" and labels!=333:
                    matches=[(path,a) for path,a in snap["fit_audits"].items()
                             if f"/seed_{seed}/cw16_llm16_scientist_v2/" in path and a["train_rows"]==labels]
                else:
                    target = "Random32" if labels==333 else method
                    match = overview[(overview.seed==seed)&(overview.Method==target)&(overview.labels==labels)]
                    matches=[(str(r.source_path),snap["fit_audits"][str(r.source_path)])
                             for r in match.itertuples(index=False)]
                check(f"completed_fit/{figure}/{seed}/{method}/{labels}",bool(matches))
                path,a=matches[0]
                add_receipt(path,a,snap["source_hashes"][path],figure,seed,method,labels,value)
                rows.append({"seed":seed,"method":method,"labels":labels,"nrmse":value,
                             "split":"validation","batch_size":32,"source_path":path})
    return pd.DataFrame(rows)


def summary(frame, figure):
    rows=[]
    for (seed,method),g in frame.groupby(["seed","method"],sort=False):
        g=g.sort_values("labels")
        x,y=g.labels.to_numpy(),g.nrmse.to_numpy()
        check(f"finite/{figure}/{seed}/{method}",np.isfinite(y).all() and (y>=0).all())
        area=float(np.trapezoid(y,x)/(x[-1]-x[0]))
        rows.append({"figure":figure,"seed":str(seed),"method":method,"start":int(x[0]),
                     "end":int(x[-1]),"AULC":area,"endpoint":float(y[-1]),"split":"validation"})
    result=pd.DataFrame(rows)
    means=[]
    for method,g in result.groupby("method",sort=False):
        check(f"two_seeds/{figure}/{method}",len(g)==2)
        means.append({**g.iloc[0].to_dict(),"seed":"mean","AULC":g.AULC.mean(),"endpoint":g.endpoint.mean()})
    return pd.concat([result,pd.DataFrame(means)],ignore_index=True)


def main():
    DATA.mkdir(parents=True,exist_ok=True)
    SOURCES.mkdir(parents=True,exist_ok=True)
    QA.mkdir(parents=True,exist_ok=True)
    snap=js(SNAPSHOT,"snapshot_20261011.json")
    check("snapshot_validation_only",snap["evaluation"]=="validation_only")
    overview=csv(PPT+"overview_fit_receipts.csv","overview_fit_receipts.csv")
    protocol=js(PPT+"frozen_inputs/studies/active_learning/qgeognn_v2_row_cw_selector_factorial/protocol.json",
                "factorial_protocol.json")
    scientist=js(ASTRA+"protocol.json","scientist_astra_protocol.json")
    transition=js(FREE+"transition_manifest.json","free_transition_manifest.json")
    completion=js("reports/llm_free_complete_20261013/completion_audit.json","free_completion_audit.json")
    free_report=js("reports/llm_free_complete_20261013/validation_curves.json","free_validation_curves_20261013.json")
    free_hashes=js("reports/llm_free_complete_20261013/source_hashes.json","free_source_hashes.json")
    check("matched_predictor_training_protocol",scientist["training"]==protocol["training"])
    check("free_training_protocol_unchanged",transition["training_prompt_selection_budget_changed"] is False)
    read(PPT+"METHODS_APPENDIX.md","methods_appendix.md")
    read("src/qgeognn_al/active_learning_v2/scientist_study.py","scientist_study.py")
    read("src/qgeognn_al/active_learning_v2/scientist_selector.py","scientist_selector.py")
    read("src/qgeognn_al/active_learning_v2/dual_expert_study.py","dual_expert_study.py")
    read("src/qgeognn_al/active_learning_v2/scientist_lifecycle.py","scientist_lifecycle.py")

    r01=curves_from_snapshot(snap,"R01",["Random32","LCMD32","CW32 (LCMD)","Hybrid traditional"],1005,overview)
    r03=curves_from_snapshot(snap,"R03",["CW32 (LCMD)","CW16 + LLM16"],813,overview)
    table(r01,"R01_learning_curves.csv")
    table(r03,"R03_learning_curves.csv")
    summaries=[summary(r01,"R01"),summary(r03,"R03")]

    # R02: full precision from the saved CSV, not the rounded user table.
    extension=csv(PPT+"frozen_inputs/studies/active_learning/qgeognn_v2_row_cw_selector_extension_to_1005/validation_curves.csv",
                  "factorial_extension_validation_curves.csv")
    r02=extension[extension.arm.isin(["raw_lcmd","cw_lcmd","raw_ivr","cw_ivr","raw_maxdet","cw_maxdet"])].copy()
    tagged=csv(PPT+"fig11_validation_curves.csv","factorial_tagged_validation_curves.csv")
    for row in r02.itertuples(index=False):
        counterpart=tagged[(tagged.seed==row.seed)&(tagged.arm==row.arm)&(tagged.budget==row.budget)]
        check(f"R02_validation_metadata/{row.seed}/{row.arm}/{row.budget}",len(counterpart)==1 and
              counterpart.iloc[0].split=="validation" and counterpart.iloc[0].batch_size==32 and
              abs(counterpart.iloc[0].combined_normalized_RMSE-row.combined_normalized_RMSE)<1e-12)
    r02["split"]="validation"
    r02["batch_size"]=32
    r02=r02.rename(columns={"arm":"method","budget":"labels","combined_normalized_RMSE":"nrmse"})
    r02=r02[["seed","method","labels","nrmse","split","batch_size"]]
    for (seed,method),g in r02.groupby(["seed","method"]):
        check(f"R02_grid/{seed}/{method}",g.sort_values("labels").labels.tolist()==list(range(333,1006,32)))
    check("R02_split",set(r02.split)=={"validation"})
    r02s=summary(r02,"R02")
    historical=csv(PPT+"fig11_summary.csv","factorial_aulc_summary.csv")
    for row in r02s.itertuples(index=False):
        expected=historical[(historical.seed.astype(str)==row.seed)&(historical.arm==row.method)].iloc[0].AULC333_1005
        check(f"R02_AULC/{row.seed}/{row.method}",abs(row.AULC-expected)<1e-12)
    table(r02,"R02_learning_curves.csv")
    table(r02s,"R02_aulc.csv")
    summaries.append(r02s)

    # R04 controls come from the same audited snapshot; Free is reconstructed
    # exclusively from completed fit receipts with a published source hash.
    controls=curves_from_snapshot(snap,"R04",["Random32","CW32 (LCMD)","CW16 + LLM16"],813,overview)
    free_rows=[]
    for seed in SEEDS:
        boundary=int(transition["start_labels"][str(seed)])
        for budget in range(333,814,32):
            matches=[]
            for path,digest in free_hashes.items():
                if f"/seed_{seed}/" not in path:
                    continue
                if budget!=333 and "/free_llm32_scientist_v2/" not in path:
                    continue
                a=json.loads((ROOT/path).read_bytes())
                if a["train_rows"]==budget:
                    matches.append((path,digest,a))
            check(f"Free_completed_fit/{seed}/{budget}",len(matches)==1)
            path,digest,a=matches[0]
            value=a["best_validation_combined_normalized_rmse"]
            add_receipt(path,a,digest,"R04",seed,"Free LLM32",budget,value)
            p=next(p for p in free_report[str(seed)]["free_llm32_scientist_v2"] if p["labels"]==budget)
            check(f"Free_report_agrees/{seed}/{budget}",abs(value-p["nrmse"])<1e-12)
            free_rows.append({"seed":seed,"method":"Free LLM32","labels":budget,"nrmse":value,
                              "split":"validation","batch_size":32,"source_path":path,
                              "llm_for_preceding_batch":("shared initial fit" if budget==333 else
                                 "gpt-6-astra/high" if budget<=boundary else "gpt-6.1-sol/xhigh")})
    r04=pd.concat([controls,pd.DataFrame(free_rows)],ignore_index=True)
    table(r04,"R04_learning_curves.csv")
    r04s=summary(r04,"R04")
    for seed in SEEDS:
        got=r04s[(r04s.seed==str(seed))&(r04s.method=="Free LLM32")].iloc[0]
        check(f"Free_completion_AULC/{seed}",abs(got.AULC-completion[str(seed)]["normalized_validation_aulc_333_813"])<1e-12)
    summaries.append(r04s)
    # Equality checks prevent silently changing the reused controls across R03/R04.
    for seed in SEEDS:
        for old,new in [("center_width_lcmd","CW32 (LCMD)"),("random32","Random32"),("cw16_llm16_scientist_v2","CW16 + LLM16")]:
            reported=[p["nrmse"] for p in free_report[str(seed)][old] if p["labels"]<=813]
            actual=controls[(controls.seed==seed)&(controls.method==new)].sort_values("labels").nrmse.tolist()
            check(f"R04_reused_control_agrees/{seed}/{new}",np.allclose(reported,actual,rtol=0,atol=1e-12))

    branch=csv(PPT+"frozen_inputs/studies/active_learning/qgeognn_v2_same_state_branching/results/branch_results.csv",
               "same_state_branch_results.csv")
    check("branch_24_records",len(branch)==24)
    check("branch_AULC_recomputed",np.allclose((branch.NRMSE_anchor+2*branch.NRMSE_plus32+branch.NRMSE_plus64)/4,
                                               branch.short_AULC,rtol=0,atol=1e-12))
    table(branch,"F08_branch_results.csv")
    fits=pd.DataFrame(RECEIPTS)
    for seed,g in fits.groupby("seed"):
        check(f"shared_validation_identity/{seed}",g.validation_ids_hash.nunique()==1)
        check(f"validation_416/{seed}",set(g.validation_rows)=={416})
        check(f"shared_initialization/{seed}",g.initialization_hash.nunique()==1 and set(g.initialization_seed)=={seed+2000003})
        check(f"QGeoGNN_V2_identity/{seed}",set(g.model_variant)=={"qgeognn_v2"} and set(g.parameter_count)=={458952})
    table(fits,"plotted_fit_receipts.csv")
    table(pd.concat(summaries,ignore_index=True),"result_summary.csv")
    derived=[{"file":str(p.relative_to(OUT)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(DATA.glob("*.csv"))]
    (QA/"derived_data_manifest.json").write_text(json.dumps(derived,indent=2)+"\n",encoding="utf-8")
    # Deduplicate repeated retained receipt references without losing figure-point mapping.
    unique={r["repository_path"]:r for r in MANIFEST}
    (DATA/"source_manifest.json").write_text(json.dumps(list(unique.values()),indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    audit={"checks":CHECKS,"check_count":len(CHECKS),"all_passed":all(x["passed"] for x in CHECKS),
           "plotted_receipts":len(RECEIPTS),"original_receipts_missing":int((~fits.original_retained).sum()),
           "raw_dataset_labels_read":False,"models_or_checkpoints_loaded":False,"training_or_LLM_calls":False,
           "snapshot_capture":snap["captured_at"],"snapshot_source_commit":"42ab12600a353bba4a063ef5d1b623ee2f7b976a",
           "predictor_training":scientist["training"],"free_switch_after_labels":transition["start_labels"]}
    (QA/"data_audit.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"Data audit passed: {len(CHECKS)} checks, {len(RECEIPTS)} plotted fit receipts; missing originals={audit['original_receipts_missing']}")


if __name__=="__main__":
    main()
