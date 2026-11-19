"""Finite registered nested calibration, followed by the original neural pair."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import study_support as support
import numpy as np
import pandas as pd
from qgeognn_al.active_learning_v2.epig import select_gaussian_batch


def freeze():
    support.verify_freeze()
    if (HERE/"freeze.json").exists():
        return verify()
    assert not list((HERE/"qualification").glob("seed_*/complete.json"))
    assert not list((HERE/"nested_cache").rglob("attempt_*.json"))
    assert not list((HERE/"final_models").rglob("attempt_*.json"))
    if (HERE/"compute_budget.json").exists():
        assert not support.read(HERE/"compute_budget.json")["attempts"]
    receipt=support.read(HERE/"test_receipt.json")
    assert receipt["exit_code"]==0
    paths=[p for p in HERE.glob("*.py") if p.name!="report_stage2.py"]
    paths += [HERE/"config.json",HERE/"STAGE2_PROTOCOL.md"]
    paths += [HERE.parent/"stage2_formal_adapter.py", support.ROOT/"tests/active_learning_v2/test_epig_stage2_adapter.py"]
    record={"schema":"epig-stage2-freeze-v1","created_unix":time.time(),
            "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=support.ROOT,text=True).strip(),
            "stage1_freeze_sha256":support.digest(HERE.parent/"freeze.json"),
            "stage1_qualification_sha256":support.digest(HERE.parent/"qualification/decision.json"),
            "files":{str(p.relative_to(support.ROOT)):support.digest(p) for p in sorted(paths)},
            "test_receipt_sha256":support.digest(HERE/"test_receipt.json"),
            "data_hashes":{str(p.relative_to(support.ROOT)):support.digest(p) for p in (support.SOURCE_DATA,support.SOURCE_GRAPH_CACHE)},
            "packages":support.read(HERE.parent/"freeze.json")["packages"],
            "stage2_fits_before_seal":0,"new_neural_fits_before_seal":0,
            "registration_scope":"after stage1 OOF inspection; before any stage2 OOF or AL result"}
    support.write_once(HERE/"freeze.json",record)
    return record


def verify():
    support.verify_freeze()
    record=support.read(HERE/"freeze.json")
    assert record["stage1_freeze_sha256"]==support.digest(HERE.parent/"freeze.json")
    assert record["stage1_qualification_sha256"]==support.digest(HERE.parent/"qualification/decision.json")
    for p,h in {**record["files"],**record["data_hashes"]}.items():
        if support.digest(support.ROOT/p)!=h:
            raise RuntimeError(f"stage2 frozen dependency drift: {p}")
    assert record["packages"]=={n:importlib.metadata.version(n) for n in record["packages"]}
    return record


def reserve_fit(seed,family,phase):
    config=support.read(HERE/"config.json")
    path=HERE/"compute_budget.json"
    budget=support.read(path) if path.exists() else {"started_unix":time.time(),"attempts":[]}
    limit=config["compute_budget"]["qualification_fit_attempts_max_including_interruptions"]
    if len(budget["attempts"])>=limit:
        raise RuntimeError("stage2 cumulative 263-attempt budget exhausted")
    if time.time()-budget["started_unix"]>config["compute_budget"]["qualification_wall_budget_seconds"]:
        raise RuntimeError("stage2 cumulative 1800s qualification/finalization budget exhausted")
    budget["attempts"].append({"seed":seed,"family":family,"phase":phase,"started_unix":time.time()})
    support.dump(path,budget)


@contextmanager
def qualification_deadline():
    path=HERE/"compute_budget.json"
    budget=support.read(path) if path.exists() else {"started_unix":time.time(),"attempts":[]}
    support.dump(path,budget)
    seconds=1800-(time.time()-budget["started_unix"])
    if seconds<=0:
        raise RuntimeError("stage2 cumulative 1800s qualification deadline exhausted")
    old_handler=signal.getsignal(signal.SIGALRM)
    def expired(signum,frame):
        raise RuntimeError("stage2 cumulative 1800s qualification deadline reached")
    signal.signal(signal.SIGALRM,expired)
    old_timer=signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,*old_timer)
        signal.signal(signal.SIGALRM,old_handler)


def qualification():
    verify()
    from nested_calibration import run_nested_family
    config=support.read(HERE/"config.json")
    frame,x=support.feature_matrix()
    rows=[]
    for seed in config["seeds"]:
        _,_,indices,y=support.training_inputs(seed,feature_frame=frame)
        for family in config["probabilistic_families"]:
            print(json.dumps({"phase":"stage2_nested","seed":seed,"family":family}),flush=True)
            result=run_nested_family(seed,family,x[indices],y,
                cache_directory=HERE/f"nested_cache/seed_{seed}_{family}",
                cache_contract={"freeze_sha256":support.digest(HERE/"freeze.json"),"model_options":config["model_options"]},
                before_fit=lambda *a,**kw:reserve_fit(seed,family,"nested_qualification"))
            for scheme,candidate in result["candidates"].items():
                directory=HERE/f"qualification/seed_{seed}_{family}_{scheme}"
                complete=directory/"complete.json"
                if complete.exists():
                    receipt=support.read(complete)
                    assert receipt["freeze_sha256"]==support.digest(HERE/"freeze.json")
                    for n,h in receipt["files"].items():
                        assert support.digest(directory/n)==h
                    m=support.read(directory/"metrics.json")
                else:
                    directory.mkdir(parents=True,exist_ok=True)
                    m={k:v for k,v in candidate.items() if k not in ("oof_mean","oof_covariance_blocks")}
                    m["temperatures"]=result["temperatures"]
                    np.savez_compressed(directory/"oof.npz",canonical_indices=indices,
                        sample_ids=frame.iloc[indices].sample_id.to_numpy(str),authorized_L333_truth_ml=y,
                        oof_mean_ml=candidate["oof_mean"],oof_covariance_blocks_ml2=candidate["oof_covariance_blocks"])
                    support.dump(directory/"metrics.json",m)
                    support.write_once(complete,{"freeze_sha256":support.digest(HERE/"freeze.json"),
                        "files":{n:support.digest(directory/n) for n in ("metrics.json","oof.npz")}})
                metrics=m["metrics"]
                joint_difference=sum(f["held_out_rows"]*(f["full_joint_nll_per_scalar"]-f["diagonal_nll_per_scalar"])
                                     for f in m["folds"])/333
                for j,endpoint in enumerate(("V1","V2")):
                    rows.append({"seed":seed,"family":family,"calibration":scheme,"endpoint":endpoint,
                        **{k:metrics[k][j] for k in ("rmse","mae","marginal_nll","coverage50","coverage80","coverage95")},
                        "standardized_residual_rms":np.sqrt(metrics["standardized_residual_second_moment"][j]),
                        "joint_nlpd_degradation_per_scalar":joint_difference,
                        "finite_positive_covariance":all(f["minimum_predictive_eigenvalue"]>0 for f in m["folds"])})
    table=pd.DataFrame(rows)
    table.to_csv(HERE/"qualification/seed_metrics.csv",index=False)
    q=config["qualification"]
    gates=[]
    for (seed,family,scheme),group in table.groupby(["seed","family","calibration"],sort=False):
        reasons=[]
        for _,r in group.iterrows():
            for field,key in (("coverage80","coverage80_bounds"),("coverage95","coverage95_bounds"),("standardized_residual_rms","standardized_residual_rms_bounds")):
                if not q[key][0]<=r[field]<=q[key][1]:
                    reasons.append(f"{r.endpoint} {field}={r[field]:.6f} outside {q[key]}")
        joint=float(group.joint_nlpd_degradation_per_scalar.mean())
        if joint>q["joint_nlpd_max_degradation_vs_diagonal_per_scalar"]:
            reasons.append(f"joint NLPD degradation {joint:.6f} > .10 nat/scalar")
        if not group.finite_positive_covariance.all():
            reasons.append("nonpositive covariance")
        gates.append({"seed":int(seed),"family":family,"calibration":scheme,"qualified":not reasons,
                      "reasons":reasons,"mean_marginal_nlpd":float(group.marginal_nll.mean()),
                      "joint_nlpd_degradation_per_scalar":joint})
    by_seed={}
    for seed in config["seeds"]:
        local=[g for g in gates if g["seed"]==seed]
        eligible=sorted([g for g in local if g["qualified"]],key=lambda g:(not g["family"].startswith("gp_"),g["mean_marginal_nlpd"],g["family"],g["calibration"]))
        chosen=eligible[0] if eligible else None
        by_seed[str(seed)]={"selected_family":chosen["family"] if chosen else None,
                           "selected_calibration":chosen["calibration"] if chosen else None,"models":local}
    ready=all(d["selected_family"] for d in by_seed.values())
    decision={"status":"QUALIFIED" if ready else "INCOMPLETE_POSTERIOR_QUALIFICATION","run_allowed":ready,
              "selected_family_by_seed":{s:d["selected_family"] for s,d in by_seed.items()},
              "calibration_by_seed":{s:d["selected_calibration"] for s,d in by_seed.items()},
              "by_seed":by_seed,"models":gates,"freeze_sha256":support.digest(HERE/"freeze.json"),
              "completed_model_fits":240,"future_U_labels_read":0,"validation_labels_read":0,
              "test_labels_read":0,"new_neural_fits":0,"original_stage1_untouched":True}
    support.write_once(HERE/"qualification/decision.json",decision)
    print(json.dumps({k:v for k,v in decision.items() if k not in ("models","by_seed")},ensure_ascii=False),flush=True)


def finalize():
    verify()
    from nested_calibration import fit_calibrated_full
    decision=support.read(HERE/"qualification/decision.json")
    if not decision["run_allowed"]:
        raise RuntimeError("stage2 has no qualified full five-seed posterior; no final fit or AL")
    config=support.read(HERE/"config.json")
    frame,x=support.feature_matrix()
    completed=[]
    for seed in config["seeds"]:
        family=decision["selected_family_by_seed"][str(seed)]
        scheme=decision["calibration_by_seed"][str(seed)]
        _,_,idx,y=support.training_inputs(seed,feature_frame=frame)
        result=fit_calibrated_full(seed,family,scheme,x[idx],y,
            cache_directory=HERE/f"final_models/seed_{seed}",
            cache_contract={"freeze_sha256":support.digest(HERE/"freeze.json"),"model_options":config["model_options"]},
            before_fit=lambda *a,**kw:reserve_fit(seed,family,"full_L333_calibration"))
        record={"seed":seed,"family":family,"calibration":scheme,"width_temperature":np.asarray(result["temperature"]).tolist(),
                "scope":"fixed_L333_training_side_crossfit; unchanged through AL budget",
                "freeze_sha256":support.digest(HERE/"freeze.json")}
        support.write_once(HERE/f"final_models/seed_{seed}/temperature.json",record)
        completed.append(record)
    support.write_once(HERE/"finalization.json",{"status":"COMPLETE","seeds":completed,
        "qualification_sha256":support.digest(HERE/"qualification/decision.json"),
        "temperature_hashes":{str(seed):support.digest(HERE/f"final_models/seed_{seed}/temperature.json") for seed in config["seeds"]}})


def verify_finalization():
    final=support.read(HERE/"finalization.json")
    assert final["status"]=="COMPLETE" and final["qualification_sha256"]==support.digest(HERE/"qualification/decision.json")
    for seed,h in final["temperature_hashes"].items():
        assert support.digest(HERE/f"final_models/seed_{seed}/temperature.json")==h
    return final


def formal_callback(seed,arm,round_index,labeled_indices,unlabeled_indices,labeled_truth,feature_matrix_X,feature_frame,target_scales,previous_checkpoint_path=None):
    verify()
    from nested_calibration import TemperaturePosterior,model_factory
    decision=support.read(HERE/"qualification/decision.json")
    assert decision["run_allowed"] and verify_finalization()["status"]=="COMPLETE"
    family=decision["selected_family_by_seed"][str(seed)]
    temperature=support.read(HERE/f"final_models/seed_{seed}/temperature.json")["width_temperature"]
    scales=np.array([target_scales[t] for t in ("V1","V2")])
    ledger_path=HERE/"runtime/formal_aux_attempts.json"
    ledger=support.read(ledger_path) if ledger_path.exists() else {"attempts":[],"limit":210}
    if len(ledger["attempts"])>=210:
        raise RuntimeError("210 formal auxiliary model fit attempts exhausted")
    attempt={"seed":seed,"arm":arm,"source_round":round_index,"started_unix":time.time(),
             "L_ids_hash":support.ids_hash(feature_frame.iloc[labeled_indices].sample_id),"status":"RUNNING"}
    ledger["attempts"].append(attempt)
    support.dump(ledger_path,ledger)
    try:
        model=model_factory(family,{"model_options":support.read(HERE/"config.json")["model_options"]}).fit(feature_matrix_X[labeled_indices],labeled_truth/scales)
    except BaseException as error:
        attempt.update(status="FAILED",error=str(error),ended_unix=time.time())
        support.dump(ledger_path,ledger)
        raise
    attempt.update(status="COMPLETED",ended_unix=time.time(),model_audit=model.audit_)
    support.dump(ledger_path,ledger)
    outer=np.r_[labeled_indices,unlabeled_indices]
    ids=feature_frame.iloc[outer].sample_id.to_numpy(str)
    posterior=TemperaturePosterior(model.posterior(feature_matrix_X[outer],row_ids=ids),temperature)
    candidates=np.arange(len(labeled_indices),len(outer))
    refs=support.reference_positions(seed,"P1",np.arange(len(outer)),ids,1024)
    objective="epig" if arm=="posterior_epig_conditional_greedy32" else "ivr"
    result=select_gaussian_batch(posterior,candidates,refs,objective=objective,labeled_positions=np.arange(len(labeled_indices)))
    return {"selected_ids":result["selected_row_ids"],"diagnostics":result,
            "probabilistic_model_audit":{**model.audit_,"fixed_L333_width_temperature":temperature},
            "units":"original fixed L333 target standard deviations; calibrated covariance"}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("freeze","qualify","finalize","formal"))
    phase=parser.parse_args().phase
    from epig_formal_runner import exclusive_worker
    with exclusive_worker(HERE/"execution.lock"):
        if phase=="freeze":freeze()
        elif phase=="qualify":
            with qualification_deadline():qualification()
        elif phase=="finalize":
            with qualification_deadline():finalize()
        else:
            from stage2_formal_adapter import run_stage2_formal
            if support.read(HERE/"qualification/decision.json")["run_allowed"]:
                verify_finalization()
            run_stage2_formal(formal_callback,HERE,verify)


if __name__=="__main__":main()
