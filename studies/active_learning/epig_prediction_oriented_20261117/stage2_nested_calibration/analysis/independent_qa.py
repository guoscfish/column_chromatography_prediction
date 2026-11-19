"""Read-only numerical QA of completed stage-2 results; no fitting/imports."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri

HERE = Path(__file__).resolve().parents[1]
STAGE1 = HERE.parent
ROOT = STAGE1.parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_contract(array):
    array = np.ascontiguousarray(array)
    return {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def contract_sha(record):
    return hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()


def verify_receipt(directory, name="complete.json", freeze=None):
    receipt = read(directory/name)
    if freeze is not None:
        assert receipt["freeze_sha256"] == freeze
    for filename, expected in receipt["files"].items():
        assert sha(directory/filename) == expected, str(directory/filename)
    return receipt


def max_difference(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max([max_difference(a[k], b[k]) for k in a] + [0.])
    if isinstance(a, list):
        assert len(a) == len(b)
        return max([max_difference(x,y) for x,y in zip(a,b)] + [0.])
    if isinstance(a, (int,float)):
        return abs(float(a)-float(b))
    assert a == b
    return 0.


stage2_sha, stage1_sha = sha(HERE/"freeze.json"), sha(STAGE1/"freeze.json")
seal = read(HERE/"freeze.json")
assert seal["stage1_freeze_sha256"] == stage1_sha
assert seal["stage1_qualification_sha256"] == sha(STAGE1/"qualification/decision.json")
for relative, expected in seal["files"].items():
    assert sha(ROOT/relative) == expected
config = read(HERE/"config.json")
decision = read(HERE/"qualification/decision.json")
assert decision["freeze_sha256"] == stage2_sha
ledger = read(HERE/"compute_budget.json")
expected_states = {(int(seed), family) for seed in config["seeds"] for family in config["probabilistic_families"]}
reserved = Counter((int(a["seed"]),a["family"]) for a in ledger["attempts"])
assert set(reserved) == expected_states and set(reserved.values()) == {12}
assert len(ledger["attempts"]) == 240
assert {a["phase"] for a in ledger["attempts"]} == {"nested_qualification"}
identity_checks, boundaries, temperatures = [], [], []
local_count = 0
receipt_count = 0
extreme_nll_temperature = None
for seed, family in sorted(expected_states):
    first = STAGE1/f"qualification/seed_{seed}_{family}"
    verify_receipt(first, freeze=stage1_sha)
    stage2 = HERE/f"qualification/seed_{seed}_{family}_identity"
    verify_receipt(stage2, freeze=stage2_sha)
    first_metrics = read(first/"metrics.json")
    second_metrics = read(stage2/"metrics.json")
    with np.load(first/"oof.npz",allow_pickle=False) as source, np.load(stage2/"oof.npz",allow_pickle=False) as second:
        for field in ("sample_ids","canonical_indices","authorized_L333_truth_ml"):
            np.testing.assert_array_equal(source[field],second[field])
        y = source["authorized_L333_truth_ml"].copy()
        mean_diff = float(np.max(np.abs(source["oof_mean_ml"] - second["oof_mean_ml"])))
        covariance_diff = float(np.max(np.abs(source["oof_covariance_blocks_ml2"] - second["oof_covariance_blocks_ml2"])))
        np.testing.assert_allclose(source["oof_mean_ml"],second["oof_mean_ml"],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(source["oof_covariance_blocks_ml2"],second["oof_covariance_blocks_ml2"],rtol=1e-10,atol=1e-10)
    metric_diff = max_difference(first_metrics["metrics"],second_metrics["metrics"])
    assert metric_diff < 1e-10
    fold_differences = []
    for old,new in zip(first_metrics["folds"],second_metrics["folds"]):
        assert old["train_positions"] == new["train_positions"]
        assert old["held_out_positions"] == new["held_out_positions"]
        for field in ("full_joint_nll_per_scalar","diagonal_nll_per_scalar","block_diagonal_nll_per_scalar","minimum_predictive_eigenvalue","whitened_residual_rms"):
            fold_differences.append(abs(old[field]-new[field]))
    assert max(fold_differences) < 1e-9
    identity_checks.append({"seed":seed,"family":family,"mean_max_abs_difference_ml":mean_diff,
                            "covariance_max_abs_difference_ml2":covariance_diff,"metrics_max_abs_difference":metric_diff,
                            "fold_joint_metric_max_abs_difference":max(fold_differences)})
    cache = HERE/f"nested_cache/seed_{seed}_{family}"
    main_contract = read(cache/"contract.json")
    assert main_contract["y"] == array_contract(y)
    assert main_contract["X"] == read(first/"fold_cache/contract.json")["X"]
    assert main_contract["caller_contract"]["freeze_sha256"] == stage2_sha
    expected_outer = np.array_split(np.random.default_rng(seed+910007).permutation(333),3)
    local = list(cache.rglob("attempt_*.json"))
    assert len(local)==12
    assert all(read(path)["status"]=="COMPLETE" for path in local)
    local_count += len(local)
    for fold, held in enumerate(expected_outer):
        train = np.setdiff1d(np.arange(333),held)
        fold_dir = cache/f"fold_{fold}"
        assert verify_receipt(fold_dir)["contract_sha256"] == contract_sha(main_contract)
        receipt_count += 1
        evaluation = read(fold_dir/"evaluation.json")
        assert evaluation["train_positions"]==train.tolist() and evaluation["held_out_positions"]==held.tolist()
        inner_dir = fold_dir/"inner"
        inner_contract = read(inner_dir/"contract.json")
        assert inner_contract["y"] == array_contract(y[train])
        inner_seed = seed+910007+1009*(fold+1)
        assert inner_contract["seed"] == inner_seed
        outer_contract = read(fold_dir/"outer/contract.json")
        assert outer_contract["y"] == array_contract(y[train])
        assert verify_receipt(fold_dir/"outer")["contract_sha256"] == contract_sha(outer_contract)
        receipt_count += 1
        assert read(fold_dir/"outer/fit_audit.json")["n_train"]==222
        inner_mean, inner_cov = np.empty((222,2)), np.empty((222,2,2))
        expected_inner = np.array_split(np.random.default_rng(inner_seed).permutation(222),3)
        inner_held_all = []
        for inner_fold, inner_held in enumerate(expected_inner):
            inner_meta = read(inner_dir/f"fold_{inner_fold}.json")
            inner_train = np.setdiff1d(np.arange(222),inner_held)
            assert inner_meta["held_out_positions"]==inner_held.tolist()
            assert inner_meta["train_positions"]==inner_train.tolist()
            assert not set(train[inner_train]) & set(held)
            assert not set(train[inner_held]) & set(held)
            assert inner_meta["model"]["n_train"]==148
            assert verify_receipt(inner_dir,name=f"fold_{inner_fold}_complete.json")["contract_sha256"] == contract_sha(inner_contract)
            receipt_count += 1
            with np.load(inner_dir/f"fold_{inner_fold}.npz",allow_pickle=False) as arrays:
                inner_mean[inner_held] = arrays["mean"]
                inner_cov[inner_held] = arrays["covariance_blocks"]
            inner_held_all.extend(inner_held)
        assert sorted(inner_held_all)==list(range(222))
        z = (y[train]-inner_mean)/np.sqrt(np.diagonal(inner_cov,axis1=1,axis2=2))
        expected = {"identity":np.ones(2),"nll":np.sqrt(np.mean(z*z,axis=0)),
                    "central80":np.quantile(np.abs(z),.8,axis=0,method="linear")/ndtri(.9)}
        if extreme_nll_temperature is None or expected["nll"].max() > extreme_nll_temperature["width_temperature"]:
            output=int(np.argmax(expected["nll"]))
            squared=z[:,output]**2
            extreme_nll_temperature={"seed":seed,"family":family,"outer_fold":fold,"endpoint":f"V{output+1}",
                                     "width_temperature":float(expected["nll"][output]),
                                     "maximum_inner_abs_standardized_residual":float(np.abs(z[:,output]).max()),
                                     "largest_inner_residual_fraction_of_squared_sum":float(squared.max()/squared.sum()),
                                     "scope":"observed outer-training-side inner OOF only; not an unlabeled-pool oracle"}
        for scheme,value in expected.items():
            actual=np.asarray(evaluation["temperatures"][scheme])
            np.testing.assert_allclose(actual,value,rtol=1e-12,atol=1e-12)
            temperatures.append({"seed":seed,"family":family,"outer_fold":fold,"calibration":scheme,
                                 "V1_width":actual[0],"V2_width":actual[1],
                                 "independent_recomputation_max_abs_error":float(np.max(np.abs(actual-value)))})
            exported = HERE/f"qualification/seed_{seed}_{family}_{scheme}"
            verify_receipt(exported,freeze=stage2_sha)
            assert read(exported/"metrics.json")["folds"][fold]["temperature"]==actual.tolist()
        boundaries.append({"seed":seed,"family":family,"outer_fold":fold,"outer_train_rows":222,
                           "outer_held_rows":111,"inner_train_rows":148,"inner_held_rows":74,
                           "inner_seed":inner_seed,"outer_held_absent_from_temperature_inputs":True})
assert local_count==240
assert not list((HERE/"final_models").rglob("attempt_*.json"))
assert not list((HERE/"runtime").rglob("fit_receipt.json"))

table=pd.read_csv(HERE/"qualification/seed_metrics.csv")
schemes=[]
for scheme in ("identity","nll","central80"):
    states=[m for m in decision["models"] if m["calibration"]==scheme]
    causes=Counter()
    for model in states:
        for reason in model["reasons"]:
            causes[reason.split("=")[0] if "=" in reason else reason.split(" ")[0]]+=1
    rows=table.loc[table.calibration.eq(scheme)]
    schemes.append({"calibration":scheme,"candidate_states":len(states),"qualified_states":sum(m["qualified"] for m in states),
                    "failure_reason_counts":dict(causes),
                    "coverage80_range":[rows.coverage80.min(),rows.coverage80.max()],
                    "coverage95_range":[rows.coverage95.min(),rows.coverage95.max()],
                    "standardized_residual_rms_range":[rows.standardized_residual_rms.min(),rows.standardized_residual_rms.max()],
                    "mean_marginal_nlpd":rows.marginal_nll.mean()})
first_decision=read(STAGE1/"qualification/decision.json")
assert decision["selected_family_by_seed"]==first_decision["selected_family_by_seed"]
assert all(scheme in (None,"identity") for scheme in decision["calibration_by_seed"].values())
assert not decision["run_allowed"]
temperature_frame=pd.DataFrame(temperatures)
summary={"status":"PASS","stage1_freeze_sha256":stage1_sha,"stage2_freeze_sha256":stage2_sha,
         "stage2_scientific_code_hashes_verified":len(seal["files"]),
         "identity_states_checked":len(identity_checks),
         "identity_mean_max_abs_difference_ml":max(v["mean_max_abs_difference_ml"] for v in identity_checks),
         "identity_covariance_max_abs_difference_ml2":max(v["covariance_max_abs_difference_ml2"] for v in identity_checks),
         "identity_metrics_max_abs_difference":max(v["metrics_max_abs_difference"] for v in identity_checks),
         "identity_fold_joint_metrics_max_abs_difference":max(v["fold_joint_metric_max_abs_difference"] for v in identity_checks),
         "local_completed_fit_attempts":local_count,"global_reserved_fit_attempts":len(ledger["attempts"]),
         "global_and_local_per_seed_family_counts_match":True,"interrupted_or_incomplete_local_attempts":0,
         "cache_complete_receipts_verified":receipt_count,"outer_boundaries_checked":len(boundaries),
         "temperatures_recomputed":len(temperatures),
         "most_extreme_nll_temperature":extreme_nll_temperature,
         "temperature_max_abs_recomputation_error":temperature_frame.independent_recomputation_max_abs_error.max(),
         "temperature_width_ranges_by_scheme":{name:{"min":float(rows[["V1_width","V2_width"]].to_numpy().min()),
                                                        "max":float(rows[["V1_width","V2_width"]].to_numpy().max())}
                                                for name,rows in temperature_frame.groupby("calibration")},
         "schemes":schemes,"selected_family_by_seed":decision["selected_family_by_seed"],
         "selected_calibration_by_seed":decision["calibration_by_seed"],
         "formal_QGeoGNN_fits":0,"finalization_model_fits":0,"new_QA_model_fits":0,
         "new_QA_U_labels_read":0,"new_QA_validation_labels_read":0,"new_QA_test_labels_read":0,
         "formal_efficacy":"NOT_EVALUATED; finite stage-2 calibration completed, original AL objective remains incomplete",
         "audit_script_sha256":sha(__file__)}
for name,rows in (("identity_comparison",identity_checks),("outer_boundaries",boundaries),("temperature_recomputation",temperatures)):
    pd.DataFrame(rows).to_csv(HERE/f"independent_qa_{name}.csv",index=False)
(HERE/"INDEPENDENT_QA.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
