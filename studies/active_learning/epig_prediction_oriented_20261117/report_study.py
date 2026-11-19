"""Render evidence-bound EPIG reports; never fit or open source target data.

Only completed study-local OOF/acquisition receipts, formal fit receipts and
the audited historical validation tables are read. Missing runs remain missing.
Reporting provenance is separate from the frozen scientific decision seal.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/epig_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import t, spearmanr

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
SEEDS = (157, 887, 2357, 6101, 12203)
ARMS = ("posterior_epig_conditional_greedy32", "posterior_ivr_conditional_greedy32")
INPUTS: set[Path] = set()
OUTPUTS: set[Path] = set()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    INPUTS.add(path)
    return json.loads(path.read_text())


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    INPUTS.add(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".reporting.tmp")
    tmp.write_text(text.rstrip() + "\n")
    tmp.replace(path)
    OUTPUTS.add(path)


def csv(path, table):
    write(path, table.to_csv(index=False))


def md(table, columns=None):
    if table is None or table.empty:
        return "未执行 / 尚无完成的结果。"
    table = table.copy()
    if columns is not None:
        table = table.loc[:, [c for c in columns if c in table]]
    def format_value(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.6f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    columns = list(table)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"]*len(columns)) + " |"]
    rows += ["| " + " | ".join(format_value(v) for v in row) + " |" for row in table.itertuples(index=False, name=None)]
    return "\n".join(rows)


def figure(name, fig):
    path = FIG / name
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    OUTPUTS.add(path)
    return f"![{name}](figures/{name})"


def completed_directories(parent):
    result = []
    for receipt_path in sorted((HERE / parent).glob("seed_*/complete.json")):
        receipt = read_json(receipt_path)
        for filename, expected in receipt.get("files", {}).items():
            file = receipt_path.parent / filename
            if not file.exists() or sha(file) != expected:
                raise RuntimeError(f"completed evidence hash mismatch: {file}")
        result.append(receipt_path.parent)
    return result


def load_qualification():
    records = []
    arrays = []
    folds = []
    for directory in completed_directories("qualification"):
        result = read_json(directory / "metrics.json")
        suffix = directory.name.removeprefix("seed_")
        seed_text, family = suffix.split("_", 1)
        seed = int(seed_text)
        m = result["metrics"]
        for j, endpoint in enumerate(("V1", "V2")):
            records.append({"seed": seed, "family": family, "endpoint": endpoint,
                            **{key: m[key][j] for key in ("rmse", "mae", "marginal_nll", "coverage50", "coverage80", "coverage95")},
                            "standardized_residual_rms": float(np.sqrt(m["standardized_residual_second_moment"][j]))})
        for fold in result["folds"]:
            rows = fold["held_out_rows"]
            q = 2
            full = fold.get("full_joint_nll_per_scalar", fold.get("full_joint_nll_per_row", np.nan)/q)
            diagonal = fold.get("diagonal_nll_per_scalar", np.nan)
            folds.append({"seed": seed, "family": family, "fold": fold["fold"],
                          "held_out_rows": rows, "full_joint_nlpd_per_scalar": full,
                          "surrogate_fit_seconds": fold["model"]["fit_seconds"],
                          "diagonal_nlpd_per_scalar": diagonal,
                          "row_block_diagonal_nlpd_per_scalar": fold["block_diagonal_nll_per_scalar"],
                          "joint_minus_diagonal_nlpd_per_scalar": full-diagonal,
                          "cross_row_only_nlpd_difference": full-fold["block_diagonal_nll_per_scalar"],
                          "minimum_predictive_eigenvalue": fold["minimum_predictive_eigenvalue"],
                          "mean_absolute_offdiagonal_predictive_correlation": fold["mean_absolute_offdiagonal_predictive_correlation"]})
        npz = directory / "oof.npz"
        INPUTS.add(npz)
        with np.load(npz, allow_pickle=False) as data:
            arrays.append({"seed": seed, "family": family,
                           "truth": data["authorized_L333_truth_ml"].copy(),
                           "mean": data["oof_mean_ml"].copy(),
                           "cov": data["oof_covariance_blocks_ml2"].copy()})
    table = read_csv(HERE / "qualification/seed_metrics.csv")
    if table.empty:
        table = pd.DataFrame(records)
    return table, pd.DataFrame(folds), arrays


def probability_figures(arrays, folds):
    links = []
    if arrays:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True, sharey=True)
        probabilities = np.linspace(.1, .95, 18)
        for family in sorted({a["family"] for a in arrays}):
            chosen = [a for a in arrays if a["family"] == family]
            standardized = np.vstack([(a["truth"]-a["mean"])/np.sqrt(np.diagonal(a["cov"],axis1=1,axis2=2)) for a in chosen])
            for j, axis in enumerate(axes):
                coverage = [(np.abs(standardized[:,j]) <= ndtri((1+p)/2)).mean() for p in probabilities]
                axis.plot(probabilities, coverage, marker=".", label=family)
        for j, axis in enumerate(axes):
            axis.plot([0,1],[0,1],"k--",linewidth=1,label="Ideal")
            axis.set(title=f"V{j+1}: training-side OOF coverage",xlabel="Nominal central coverage",ylabel="Observed coverage",xlim=(0,1),ylim=(0,1))
            axis.grid(alpha=.2)
        axes[1].legend(fontsize=8)
        links.append(figure("probability_calibration.png",fig))
        fig, axes = plt.subplots(1,2,figsize=(10.5,4.2))
        for family in sorted({a["family"] for a in arrays}):
            chosen = [a for a in arrays if a["family"] == family]
            errors = np.vstack([a["truth"]-a["mean"] for a in chosen])
            sigmas = np.vstack([np.sqrt(np.diagonal(a["cov"],axis1=1,axis2=2)) for a in chosen])
            for j,axis in enumerate(axes):
                groups = np.array_split(np.argsort(sigmas[:,j],kind="stable"),4)
                values=[np.mean(np.abs(errors[g,j]/sigmas[g,j])<=ndtri(.9)) for g in groups]
                axis.plot(range(1,5),values,"o-",label=family)
        for j,axis in enumerate(axes):
            axis.axhline(.8,color="k",linestyle="--",linewidth=1)
            axis.set(title=f"V{j+1}: conditional coverage",xlabel="Predicted sigma quartile",ylabel="Observed 80% coverage",xticks=range(1,5),ylim=(0,1))
            axis.grid(alpha=.2)
        axes[1].legend(fontsize=8)
        links.append(figure("conditional_calibration.png",fig))
    if not folds.empty:
        fig,axis=plt.subplots(figsize=(9,4.3))
        groups=[]
        for (family,seed),group in folds.groupby(["family","seed"]):
            value=np.average(group.joint_minus_diagonal_nlpd_per_scalar,weights=group.held_out_rows)
            cross_row=np.average(group.cross_row_only_nlpd_difference,weights=group.held_out_rows)
            groups.append((family,seed,value,cross_row))
        families=sorted({x[0] for x in groups})
        for k,family in enumerate(families):
            values=[x[2] for x in groups if x[0]==family]
            axis.scatter(np.full(len(values),k-.06),values,s=30,color="tab:blue",label="Full vs scalar-independent" if k==0 else None)
            cross_values=[x[3] for x in groups if x[0]==family]
            axis.scatter(np.full(len(cross_values),k+.06),cross_values,s=30,marker="x",color="tab:orange",label="Full vs row-block (cross-row only)" if k==0 else None)
            axis.plot([k-.18,k+.18],[np.mean(values)]*2,color="k",linewidth=2)
        axis.axhline(.1,color="firebrick",linestyle="--",label="Preregistered maximum degradation")
        axis.axhline(0,color="grey",linewidth=.7)
        axis.set(xticks=range(len(families)),xticklabels=families,ylabel="Joint NLPD difference, per scalar",title="Held-out joint dependence: separate output and cross-row effects")
        axis.tick_params(axis="x",rotation=15)
        axis.legend(fontsize=8)
        axis.grid(axis="y",alpha=.2)
        links.append(figure("joint_covariance_nlpd.png",fig))
    return links


def load_acquisition():
    rows=[]
    for directory in completed_directories("acquisition"):
        summary=read_json(directory/"summary.json")
        rows.append({k:v for k,v in summary.items() if k!="reference_positions"})
    return pd.DataFrame(rows),read_csv(HERE/"acquisition/reference_sensitivity.csv")


def target_comparison(acq):
    """Compare P1/P2 only after both frozen states actually completed."""
    rows=[]
    if acq.empty:
        return pd.DataFrame()
    for (seed,family,size),group in acq.groupby(["seed","family","M"]):
        if set(group.distribution)!={"P1","P2"}:
            continue
        directories=[HERE/f"acquisition/seed_{seed}_{family}_{distribution}_M{size}" for distribution in ("P1","P2")]
        tables=[read_csv(d/"scores.csv") for d in directories]
        if tables[0].sample_id.tolist()!=tables[1].sample_id.tolist():
            raise RuntimeError("P1/P2 candidate ID alignment mismatch")
        batches=[read_json(d/"batches.json") for d in directories]
        for objective in ("epig","ivr"):
            a,b=(table[objective].to_numpy(float) for table in tables)
            selected=[set(batch[f"{objective}_conditional_greedy"]["selected_row_ids"]) for batch in batches]
            rows.append({"seed":seed,"family":family,"M":size,"objective":objective,
                         "spearman_P1_P2":float(spearmanr(a,b).statistic),
                         "relative_score_rmse_P2_vs_P1":float(np.linalg.norm(a-b)/np.linalg.norm(a)),
                         "top32_overlap_P1_P2":len(set(np.argsort(-a,kind="stable")[:32])&set(np.argsort(-b,kind="stable")[:32]))/32,
                         "greedy32_overlap_P1_P2":len(selected[0]&selected[1])/32})
    return pd.DataFrame(rows)


def load_output_ablation():
    directory=HERE/"supplementary_output_ablation"
    seal=read_json(directory/"analysis_seal.json",{})
    rows=[]
    for complete in sorted(directory.glob("seed_*/complete.json")):
        receipt=read_json(complete)
        if not seal or receipt["contract"]["analysis_seal_sha256"]!=sha(directory/"analysis_seal.json"):
            raise RuntimeError("output-ablation analysis seal mismatch")
        for name,expected in receipt["files"].items():
            path=complete.parent/name
            if not path.exists() or sha(path)!=expected:
                raise RuntimeError(f"completed output-ablation hash mismatch: {path}")
            INPUTS.add(path)
        rows.append(read_json(complete.parent/"summary.json"))
    return pd.DataFrame(rows),read_json(directory/"status.json",{})


def acquisition_figures(acq,sensitivity):
    links=[]
    if not acq.empty:
        chosen=acq.loc[acq.M.eq(1024)]
        if not chosen.empty:
            fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharey=True)
            groups=list(chosen.groupby(["family","distribution"],sort=True))
            labels=[f"{family}\n{distribution}" for (family,distribution),_ in groups]
            for j,field in enumerate(("top32_overlap_epig_ivr","greedy_batch_overlap_epig_ivr")):
                for k,(_,group) in enumerate(groups):
                    values=group[field].to_numpy()
                    axes[j].scatter(np.full(len(values),k),values,s=25)
                    axes[j].plot([k-.18,k+.18],[np.mean(values)]*2,color="k",linewidth=2)
                axes[j].set(xticks=range(len(labels)),xticklabels=labels,ylim=(-.03,1.03),ylabel="Selected fraction shared",title=("Pointwise Top-32: EPIG vs IVR" if j==0 else "Conditional B32: EPIG vs IVR"))
                axes[j].tick_params(axis="x",labelsize=7,rotation=20)
                axes[j].grid(axis="y",alpha=.2)
            links.append(figure("ranking_batch_overlap.png",fig))
            fig,axis=plt.subplots(figsize=(9,4.4))
            for k,(_,group) in enumerate(groups):
                ratio=group.greedy_epig_batch_information_nats/group.pointwise_batch_information_nats
                axis.scatter(np.full(len(ratio),k),ratio,s=30)
                axis.plot([k-.18,k+.18],[ratio.mean()]*2,color="k",linewidth=2)
            axis.axhline(1,color="grey",linestyle="--")
            axis.set(xticks=range(len(labels)),xticklabels=labels,ylabel="Conditional greedy MI / pointwise-batch MI",title="Batch redundancy under the stated Gaussian posterior")
            axis.tick_params(axis="x",rotation=20,labelsize=8)
            axis.grid(axis="y",alpha=.2)
            links.append(figure("batch_conditioning_information.png",fig))
    if not sensitivity.empty:
        fig,axes=plt.subplots(1,3,figsize=(13,4.2))
        for (family,distribution,objective),group in sensitivity.groupby(["family","distribution","objective"]):
            if objective!="epig":
                continue
            for axis,field in zip(axes,("relative_score_rmse","spearman","top32_overlap")):
                summary=group.groupby("M")[field].agg(["mean","std"])
                axis.errorbar(summary.index,summary["mean"],yerr=summary["std"].fillna(0),marker="o",capsize=3,label=f"{family}/{distribution}")
        for axis,title in zip(axes,("Score RMSE / norm at M2048","Rank Spearman vs M2048","Top-32 overlap vs M2048")):
            axis.set(xlabel="Reference rows M",title=title)
            axis.grid(alpha=.2)
        axes[-1].legend(fontsize=7)
        links.append(figure("reference_distribution_sensitivity.png",fig))
    return links


def formal_tables():
    rows=[]
    family_by_seed=read_json(HERE/"qualification/decision.json",{}).get("selected_family_by_seed",{})
    for path in sorted((HERE/"runtime").glob("seed_*/*/round_*/fit_receipt.json")):
        record=read_json(path)
        if record.get("arm") not in ARMS:
            continue
        rows.append({"seed":record["seed"],"method":record["arm"],"posterior_family":family_by_seed.get(str(record["seed"])),"budget":record["budget"],
                     "nrmse":record["validation"]["combined_nrmse"],
                     **{f"validation_{k}":record["validation"][k] for k in ("V1_RMSE","V2_RMSE","V1_MAE","V2_MAE")},
                     "reused":record["reused"],"training_seconds":record.get("training_seconds",0),
                     "checkpoint_sha256":record["checkpoint_sha256"]})
    curves=pd.DataFrame(rows)
    metrics=[]
    if not curves.empty:
        for (method,seed),group in curves.groupby(["method","seed"]):
            if group.budget.duplicated().any():
                raise RuntimeError("duplicate formal method/seed/budget")
            for start,end,stage in ((333,525,"early"),(525,813,"mid"),(813,1005,"late"),(333,1005,"primary_full"),(333,1485,"extension_full")):
                window=group.loc[group.budget.between(start,end)].sort_values("budget")
                expected=np.arange(start,end+1,32)
                if not np.array_equal(window.budget.to_numpy(),expected):
                    continue
                terminal=window.iloc[-1]
                metrics.append({"method":method,"seed":seed,"posterior_family":group.iloc[0].posterior_family,"stage":stage,"start_budget":start,"end_budget":end,
                                "raw_aulc":float(np.trapezoid(window.nrmse.to_numpy(),window.budget.to_numpy())/(end-start)),
                                "terminal_nrmse":float(terminal.nrmse),
                                **{k:terminal[k] for k in ("validation_V1_RMSE","validation_V2_RMSE","validation_V1_MAE","validation_V2_MAE")}})
    metrics=pd.DataFrame(metrics)
    pairs=[]
    if not metrics.empty:
        for stage,group in metrics.groupby("stage"):
            e=group.loc[group.method.eq(ARMS[0])].set_index("seed")
            v=group.loc[group.method.eq(ARMS[1])].set_index("seed")
            for seed in sorted(set(e.index)&set(v.index)):
                pairs.append({"seed":seed,"stage":stage,"delta_aulc_epig_minus_ivr":e.loc[seed,"raw_aulc"]-v.loc[seed,"raw_aulc"],
                              "delta_terminal_epig_minus_ivr":e.loc[seed,"terminal_nrmse"]-v.loc[seed,"terminal_nrmse"]})
    return curves,metrics,pd.DataFrame(pairs)


def paired_summary(pairs):
    rows=[]
    if pairs.empty:
        return pd.DataFrame()
    for stage,group in pairs.groupby("stage"):
        for metric in ("delta_aulc_epig_minus_ivr","delta_terminal_epig_minus_ivr"):
            values=group[metric].to_numpy(float)
            n=len(values)
            sd=np.std(values,ddof=1) if n>1 else np.nan
            half=t.ppf(.975,n-1)*sd/np.sqrt(n) if n>1 else np.nan
            perm=np.array([abs(np.mean(values*np.array(signs))) for signs in itertools.product((-1,1),repeat=n)])
            rows.append({"stage":stage,"metric":metric,"outer_seeds":n,"mean_delta":np.mean(values),"sample_sd":sd,
                         "t_interval_low":np.mean(values)-half,"t_interval_high":np.mean(values)+half,
                         "exact_sign_permutation_p":np.mean(perm>=abs(np.mean(values))-1e-14),
                         "five_seed_primary_complete":stage=="primary_full" and set(group.seed)==set(SEEDS)})
    return pd.DataFrame(rows)


def learning_figures(curves,baseline,pairs):
    links=[]
    if not baseline.empty:
        selected=("random32","gradient_lcmd32","historical_kernel_ivr32","new_free_cw_sol61_xhigh")
        fig,axes=plt.subplots(1,5,figsize=(16,3.7),sharex=True,sharey=True)
        for axis,seed in zip(axes,SEEDS):
            for method in selected:
                group=baseline.loc[baseline.method.eq(method)&baseline.seed.eq(seed)&baseline.budget.le(1005)].sort_values("budget")
                if not group.empty:
                    axis.plot(group.budget,group.nrmse,label=method,linewidth=1.1)
            axis.set(title=f"Seed {seed}",xlabel="Labels",ylabel="Validation combined NRMSE")
            axis.grid(alpha=.2)
        axes[-1].legend(fontsize=6,loc="upper right")
        links.append(figure("historical_validation_curves.png",fig))
        for method,method_group in baseline.groupby("method"):
            fig,axis=plt.subplots(figsize=(8,4.5))
            for seed,group in method_group.groupby("seed"):
                group=group.sort_values("budget")
                axis.plot(group.budget,group.nrmse,".-",markersize=2.5,label=f"Seed {seed}")
            axis.set(title=f"Historical development validation: {method}",xlabel="Labels",ylabel="Combined NRMSE")
            axis.legend(fontsize=8)
            axis.grid(alpha=.2)
            figure(f"historical_{method}_raw_curves.png",fig)
    if not curves.empty:
        fig,axes=plt.subplots(1,5,figsize=(16,3.7),sharex=True,sharey=True)
        for axis,seed in zip(axes,SEEDS):
            for method in ARMS:
                group=curves.loc[curves.method.eq(method)&curves.seed.eq(seed)].sort_values("budget")
                if not group.empty:
                    axis.plot(group.budget,group.nrmse,".-",label=method,markersize=3)
            axis.set(title=f"Seed {seed}",xlabel="Labels",ylabel="Validation combined NRMSE")
            axis.grid(alpha=.2)
        axes[-1].legend(fontsize=6)
        links.append(figure("formal_epig_ivr_raw_curves.png",fig))
    if not pairs.empty:
        chosen=pairs.loc[pairs.stage.eq("primary_full")]
        if not chosen.empty:
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            for axis,field in zip(axes,("delta_aulc_epig_minus_ivr","delta_terminal_epig_minus_ivr")):
                axis.axhline(0,color="k",linewidth=.8)
                axis.bar(chosen.seed.astype(str),chosen[field])
                axis.set(xlabel="Outer seed",ylabel="EPIG - same-posterior IVR",title=field.removeprefix("delta_").replace("_"," "))
            links.append(figure("formal_paired_epig_minus_ivr.png",fig))
    return links


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":110})
    config=read_json(HERE/"config.json",{})
    decision=read_json(HERE/"qualification/decision.json",{})
    stage2_decision=read_json(HERE/"stage2_nested_calibration/qualification/decision.json",{})
    stage2_budget=read_json(HERE/"stage2_nested_calibration/compute_budget.json",{})
    stage2_registered=(HERE/"stage2_nested_calibration/freeze.json").exists()
    stage2_started=bool(stage2_budget.get("attempts"))
    stage2_note=(f"第二阶段已按独立freeze实际执行，当前资格状态：{stage2_decision.get('status','执行中/尚无最终资格决定')}。实际结果与新formal状态以[stage2结果](stage2_nested_calibration/STAGE2_RESULTS.md)及[stage2决策](stage2_nested_calibration/STAGE2_NEXT_STAGE_DECISION.md)为准；下面保留stage1原资格结论，不把stage2结果倒写成stage1通过。" if stage2_started else "第二阶段已独立封存，待其真实执行结果。" if stage2_registered else "第二阶段尚未实施，仅以下有限nested calibration设计提案。")
    qtable,folds,arrays=load_qualification()
    acq,sensitivity=load_acquisition()
    targets=target_comparison(acq)
    output_ablation,output_status=load_output_ablation()
    baseline=read_csv(HERE/"baseline_validation_curves.csv")
    bmetrics=read_csv(HERE/"baseline_seed_metrics.csv")
    bsummary=read_csv(HERE/"baseline_summary.csv")
    curves,metrics,pairs=formal_tables()
    pair_summary=paired_summary(pairs)
    csv(HERE/"reported_joint_fold_metrics.csv",folds)
    csv(HERE/"formal_validation_curves.csv",curves)
    csv(HERE/"formal_seed_metrics.csv",metrics)
    csv(HERE/"formal_paired_differences.csv",pairs)
    csv(HERE/"formal_paired_summary.csv",pair_summary)
    csv(HERE/"reported_target_distribution_comparison.csv",targets)
    qfigs=probability_figures(arrays,folds)
    afigs=acquisition_figures(acq,sensitivity)
    if not targets.empty:
        fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharey=True)
        chosen=targets.loc[targets.objective.eq("epig")]
        for family,group in chosen.groupby("family"):
            for axis,field in zip(axes,("top32_overlap_P1_P2","greedy32_overlap_P1_P2")):
                summary=group.groupby("M")[field].agg(["mean","std"])
                axis.errorbar(summary.index,summary["mean"],yerr=summary["std"].fillna(0),marker="o",capsize=3,label=family)
        for axis,title in zip(axes,("EPIG pointwise Top-32: P1 vs P2","EPIG conditional B32: P1 vs P2")):
            axis.set(title=title,xlabel="Reference rows M",ylabel="Shared selected fraction",ylim=(-.03,1.03))
            axis.grid(alpha=.2)
        axes[-1].legend(fontsize=8)
        afigs.append(figure("target_distribution_overlap.png",fig))
    if not output_ablation.empty:
        fig,axes=plt.subplots(1,2,figsize=(11,4.4))
        groups=list(output_ablation.groupby("family",sort=True))
        labels=[family for family,_ in groups]
        for k,(_,group) in enumerate(groups):
            for field,offset,color,label in (("V1_only_joint32_overlap",-.08,"tab:blue","V1-only vs joint"),("V2_only_joint32_overlap",.08,"tab:orange","V2-only vs joint")):
                axes[0].scatter(np.full(len(group),k+offset),group[field],s=25,color=color,label=label if k==0 else None)
            for field,offset,color,label in (("V1_only_batch_joint_information_nats",-.08,"tab:blue","V1-only batch"),("V2_only_batch_joint_information_nats",.08,"tab:orange","V2-only batch")):
                ratio=group[field]/group.joint_batch_information_nats_direct
                axes[1].scatter(np.full(len(group),k+offset),ratio,s=25,color=color,label=label if k==0 else None)
        axes[0].set(title="Conditional B32: single-output vs joint",ylabel="Shared selected fraction",ylim=(-.03,1.03))
        axes[1].set(title="Same posterior: information about both outputs",ylabel="Single-output batch MI / joint-selected batch MI")
        axes[1].axhline(1,color="grey",linestyle="--")
        for axis in axes:
            axis.set(xticks=range(len(labels)),xticklabels=labels)
            axis.tick_params(axis="x",rotation=15,labelsize=8)
            axis.grid(axis="y",alpha=.2)
            axis.legend(fontsize=7)
        afigs.append(figure("single_vs_joint_output_ablation.png",fig))
    lfigs=learning_figures(curves,baseline,pairs)
    completed_families=len(arrays)
    qualification_complete=bool(decision)
    qstatus=decision.get("status","INCOMPLETE_NOT_YET_DECIDED")
    gpqualified=bool(decision.get("run_allowed",qstatus=="QUALIFIED"))
    full_primary=(not pairs.empty and set(pairs.loc[pairs.stage.eq("primary_full"),"seed"])==set(SEEDS))
    formal_new=int((~curves.reused.astype(bool)).sum()) if not curves.empty else 0
    models=pd.DataFrame([{**g,"reasons":"; ".join(g.get("reasons",[]))} for g in decision.get("models",[])])
    blocked=[seed for seed,item in decision.get("by_seed",{}).items() if item.get("selected_family") is None]
    reasons=[reason for item in decision.get("models",[]) for reason in item.get("reasons",[])]
    failure_detail=("所有失格条件均为80%区间过覆盖上界（大于0.90）；不是负方差、矩阵非正定或联合NLPD退化越界。部分值仅比门槛高1–6个样本比例；333行OOF的分辨率为1/333，折与行也并非完全独立。该预注册门槛必须原样执行，但这种轻微过覆盖不能表述为已证明联合相关失真或概率模型全面失效。" if reasons and all("coverage80" in reason for reason in reasons) else "具体失败维度逐项以上表为准，不从资格失败自动推断失效原因。")
    qagg=pd.DataFrame()
    if not qtable.empty:
        qagg=qtable.groupby(["family","endpoint"],as_index=False).agg(
            outer_seeds=("seed","nunique"),marginal_nlpd=("marginal_nll","mean"),coverage80=("coverage80","mean"),coverage95=("coverage95","mean"),
            standardized_residual_rms=("standardized_residual_rms",lambda x:np.sqrt(np.mean(x*x))),rmse_ml=("rmse","mean"),mae_ml=("mae","mean"))
    cost={"qualification_completed_seed_family_states":completed_families,
          "qualification_auxiliary_model_fits":completed_families*config.get("oof_folds",3),
          "acquisition_completed_states":len(acq),"new_completed_QGeoGNN_fits":formal_new,
          "acquisition_seconds_sum":float(acq.seconds.sum()) if not acq.empty else 0,
          "acquisition_peak_rss_bytes":int(acq.process_peak_rss_bytes.max()) if not acq.empty else 0,
          "formal_training_seconds_sum":float(curves.training_seconds.sum()) if not curves.empty else 0}
    cost["supplementary_output_ablation_completed_states"]=len(output_ablation)
    cost["supplementary_output_ablation_seconds_sum"]=float(output_ablation.seconds.sum()) if not output_ablation.empty else 0
    acquisition_fit_receipts=list((HERE/"acquisition/models").glob("seed_*/complete.json"))
    for path in acquisition_fit_receipts:
        read_json(path)
    cost["stage1_acquisition_surrogate_fits_completed"]=len(acquisition_fit_receipts)
    cost["stage2_nested_auxiliary_fit_attempts"]=len(stage2_budget.get("attempts",[]))
    cost["qualification_auxiliary_fit_seconds_sum"] = float(folds.surrogate_fit_seconds.sum()) if not folds.empty else 0
    acquisition_fit_audits = []
    for receipt_path in sorted((HERE/"acquisition/models").glob("*/complete.json")):
        receipt = read_json(receipt_path)
        audit_path = receipt_path.parent/"fit_audit.json"
        if sha(audit_path) != receipt["fit_audit_sha256"]:
            raise RuntimeError("acquisition model audit hash drift")
        acquisition_fit_audits.append(read_json(audit_path))
    cost["acquisition_auxiliary_model_fits"] = len(acquisition_fit_audits)
    cost["acquisition_auxiliary_fit_seconds_sum"] = sum(a["fit_seconds"] for a in acquisition_fit_audits)
    qbudget=read_json(HERE/"qualification/compute_budget.json",{})
    if qbudget:
        cost["qualification_fit_attempts_including_interruptions"]=len(qbudget.get("attempts",[]))
    ledger=read_json(HERE/"runtime/compute_budget.json",{})
    if ledger:
        cost["formal_attempts_in_budget_ledger"]=len(ledger.get("attempts",[]))
    write(HERE/"reported_compute_cost.json",json.dumps(cost,indent=2,ensure_ascii=False))
    scope="全部结果均为重复暴露历史 Row 开发队列上的 development evidence；不是独立确认。统计单位为 outer seed，五个 seed 的划分仍重叠，不能把 batch/OOF 行数当独立实验重复。"
    qtext=f"""# 概率模型资格验证

实际资格状态：**{qstatus}**。已完成 {completed_families}/20 个 seed×family 状态；每个完成状态进行 {config.get('oof_folds',3)} 折合法 L333 OOF。逐 seed 最终选择：`{json.dumps(decision.get('selected_family_by_seed',{}),ensure_ascii=False)}`。

本节为stage1原始资格。{stage2_note}

完整计算是否结束：{'是，20个状态/60次fold拟合均已完成；status的INCOMPLETE指五seed后验资格未齐，不是仍有漏跑模型。' if completed_families==20 else '否，仍有未完成状态。'}无合格本地后验的seed：{', '.join(blocked) or '无/尚未判定'}。

{scope}

## 数据边界与模型族

仅使用每个 seed 当前合法 L333 真值和预定义 response-independent 输入特征。每折的标准化、PCA、常数均值、核尺度与噪声由该折训练子集拟合；无全 L 监督 encoder 造成的 OOF 标签绕行。真实 OOF 预测与 mL² covariance 保存在 `qualification/seed_*/oof.npz` 和 `oof_predictions.csv`。不同 outer seed 的标签权限不交叉使用。

候选族为 RBF GP、Matérn-3/2 GP、独立 BLR 与 coupled matrix-normal BLR。GP 分别学习两个输出的幅度/长度尺度/噪声，是 empirical Bayes 的独立输出联合分布；BLR coupled 有明确跨输出 B，但先验/噪声共享 B，完整双输出 MI 对其 whitening 不敏感。不是 Deep Ensemble 的严格后验，也未从三个边缘分位数推断 joint covariance。

预注册门槛：每个 seed 分别仅用自己333个OOF行，要求每端点80% coverage ∈ {config.get('qualification',{}).get('coverage80_bounds')}，95% coverage ∈ {config.get('qualification',{}).get('coverage95_bounds')}，标准化残差 RMS ∈ {config.get('qualification',{}).get('standardized_residual_rms_bounds')}；held-out fold 全联合 NLPD 相对独立对角预测的平均差每 scalar ≤0.10，且 predictive covariance 正定。该seed合格GP按最低OOF NLPD选择；若无合格GP，才在该seed合格的预注册BLR中选择最低NLPD。**跨seed pooled只描述，不能影响该seed资格/选择**。任一seed无合格后验时，整个正式五seed比较不开始，不删除该seed。阈值冻结于资格结果揭示之前，不根据 AL 效果调整。

## 实际模型门槛

{md(models,['seed','family','qualified','mean_marginal_nlpd','joint_nlpd_degradation_per_scalar','reasons'])}

{failure_detail}

## 五 seed 端点描述性汇总（不作选择）

{md(qagg)}

逐 seed 原始数值见 `qualification/seed_metrics.csv`；每 fold 的跨样本联合评分见 `reported_joint_fold_metrics.csv`。80%/95% pooled coverage 与联合 NLPD 是互补诊断：边缘校准好不能保证跨样本相关正确。联合分布可直接给出有效 log-likelihood，但单一开发数据集的一组相关残差仍不足以证明真实 joint covariance 已识别。

{'资格已完成但至少一个seed没有合格本地后验，整个五seed正式门槛未满足；这支持概率模型问题类别 A 和证据不足 F，不支持 EPIG 目标无效。全部预注册候选族均应完整检验，不因某个模型先失败就取消其他族。' if qualification_complete and not gpqualified else '所有seed都有合格的本地后验（GP或明确记录的BLR备选），可在每seed相同概率后验与相同目标集下启动正式 EPIG/IVR 配对；不同seed实际模型族需分别报告，不能统一宣称五seed全是GP-EPIG。合格阈值不是校准的普遍证明。' if gpqualified else '资格尚未全部完成，不提前选择、否定或修改模型族。'}

## 概率校准与联合依赖图

{chr(10).join(qfigs) or '尚无完成的 OOF 产物，图未生成。'}

以上图的每个点/曲线来自真实 OOF 残差，不是模型自己采样出的校准效果；不同 seed pooled 图的行不是独立重复。若某族失格，原因限定于当前特征、均值、核、噪声和调参预算，不代表冻结 GNN 表征 GP、异方差代理或其他联合模型均无效。
"""
    write(HERE/"PROBABILISTIC_MODEL_QUALIFICATION.md",qtext)
    primary=acq.loc[acq.M.eq(1024)] if not acq.empty else acq
    ablation=f"""# 消融与机制证据

实际完成 {len(acq)} 个冻结 L333 acquisition 诊断状态；这些状态均**不揭示候选标签、不训练新的 QGeoGNN**。资格状态为 {qstatus}。每个状态的 `posterior_qualified` 是相应seed本地门槛；不合格状态只作数值/机制诊断，即使有其他seed合格也不能替它认证。通过资格的代理仍须接受真实 AL 终点评价。

## 同后验 EPIG 与 IVR、批量条件化和输出

{md(primary,['seed','family','distribution','M','posterior_qualified','spearman_epig_ivr','top32_overlap_epig_ivr','top100_overlap_epig_ivr','greedy_batch_overlap_epig_ivr','greedy_pointwise_batch_overlap','greedy_epig_batch_information_nats','pointwise_batch_information_nats','ivr_batch_information_nats','endpoint_V1_V2_top32_overlap','V1_fraction_singleton_information','seconds'])}

EPIG/IVR 使用相同 fitted posterior、相同参考集合和固定 L333 单位尺度，只改变 acquisition objective。Top-32 是初始单点排名重合，conditional B32 是条件化所选 batch 重合，两者不混用。MI 以 nats 记录；IVR 批次的 MI 也在同后验下独立条件化复核。贪心的代理 MI 优势不意味着 QGeoGNN 性能优势。

`epig_V1` / `epig_V2` 是边缘单任务诊断。独立输出 GP 的联合 MI 确实等于两边缘之和，但不能宣称有跨输出依赖；coupled BLR 另名单列。上表 V1 fraction 衡量初始候选总单点信息比例，不是唯一可加的 coupled 多输出归因，也不代表真实误差贡献。

### 资格揭示后的独立封存单输出批次补充分析

`supplementary_output_ablation.py` 属于**事后声明的analysis appendix**，独立seal绑定主科学freeze与代码hash，不修改stage1协议，也不借此选择主方法。状态：{output_status.get('status','尚未执行')}，已完成{len(output_ablation)}个实际P1/M1024状态。它从已校验hash的model.pkl构造相同posterior与相同refs，分别执行V1-only/V2-only conditional greedy B32，并对每个批次用直接64×64 noisy batch协方差/Schur重算完整双输出MI。新模型拟合数与新标签读取数均为0；原selected IDs与逐step trace保留在该appendix目录。

{md(output_ablation,['seed','family','posterior_qualified','V1_only_joint32_overlap','V2_only_joint32_overlap','V1_V2_greedy32_overlap','joint_batch_information_nats_direct','V1_only_batch_joint_information_nats','V2_only_batch_joint_information_nats','joint_vs_independent_sum_max_abs_score_error','joint_score_recomputation_max_abs_error','seconds'])}

对独立输出GP，`joint_vs_independent_sum_max_abs_score_error`是完整联合初始score与V1/V2独立score求和的独立复算误差；在该GP中等式来自输出独立假设。对coupled BLR，同先验/噪声共享B，使完整双输出MI在可逆whitening后与B无关，该特别可分离结构也可产生相同求和数值；真实非零cross-output covariance仍然存在。原始字段`independent_sum_is_exact_model_identity`仅标记是否执行GP独立恒等式断言，False不代表BLR数值不相等。独立BLR的共享空间结构也可使两输出排名相同。不能用这种排名不变否定更一般非可分离联合模型。贪心不是全局最优，某单输出批次完整联合MI偶尔高于joint-greedy也不矛盾。单输出批次的双输出MI差只是代理下的信息差，不是Row-level效果。

## P1/P2 与 M 敏感性

P1 从固定 outer-training row 输入抽样，P2 从合法当前 U 输入均匀抽样；均不使用 target 真值。M=256/1024/2048 采用预定义 nested hash ordering。`acquisition/reference_sensitivity.csv` 比较相同候选上的 M2048 score RMSE、Spearman 与 Top-32/100 重合。若尚无该表，不能把缺失画成稳定。reference error 是有限 M 相对 M2048 的经验差，不是相对未知真实积分的绝对误差。

{md(sensitivity)}

P1与P2在同seed/family/M上的实际比较（只有两边都完成才入表）：

{md(targets)}

{chr(10).join(afigs) or '尚无完成的 acquisition 诊断，图未生成。'}

## QGeoGNN 条件学习能力

现有冻结审计显示上样体积响应幅度压缩；这里只能将其作为候选预测器瓶颈，不是新 EPIG 失败的既定原因。未用未揭示 U 真值计算 oracle information。当前 acquisition 诊断不能识别数据在真实重训中是否被吸收。若改变预测器，必须预注册为第二研究因素，同时给 EPIG/IVR/LCMD 同一个新预测器。当前未完成这一独立预测器控制。

## 哪些消融有性能证据

以上为同状态决策与代理信息量消融，{'主 EPIG/IVR 已有五 seed 完整 L1005 性能配对，见 RESULTS.md。' if full_primary else '尚无五 seed 完整 L1005 EPIG/IVR 性能配对。'}P1/P2、Top32/greedy、单输出/多输出的完整五 seed AL 性能消融，只有确实存在相应冻结真实重训轨迹时才成立；本报告不把冻结状态的排名差异称为 AL 效果消融。
"""
    write(HERE/"ABLATION.md",ablation)
    primary_b=bsummary.loc[bsummary.end_budget.eq(1005)] if not bsummary.empty else bsummary
    seed_b=bmetrics.loc[bmetrics.end_budget.eq(1005)] if not bmetrics.empty else bmetrics
    results=f"""# 真实 Row-level 结果

**{'五 seed L333→L1005 主配对已完成。' if full_primary else '尚未完成五 seed L333→L1005 主 EPIG/同后验 IVR 配对，当前不能判定 EPIG 改善或不改善真实学习性能。'}** 当前本研究完成新的 QGeoGNN 拟合数为 {formal_new}；资格状态 {qstatus}。未执行的正式结果不以数学测试、单状态 MI 或历史曲线代替。

{stage2_note}

{scope}

## 新方法：真实已完成点与阶段

{md(metrics)}

逐点原始非单调曲线：`formal_validation_curves.csv`。窗口缺任一32-label预算点时不计算该完整窗口 AULC，不插补、不沿用别的方法checkpoint、不把best-so-far作为原始曲线。标准化 AULC 为 trapezoid(raw NRMSE,budget)/(end−start)，低者更优。V1/V2 MAE/RMSE 单位 mL。固定 L333 sigma 不随轮次改变。

## EPIG 减去同后验 IVR：逐 seed 配对

{md(pairs)}

{md(pair_summary)}

负差代表 EPIG 更优。early=333–525，mid=525–813，late=813–1005，primary_full=333–1005，extension_full=333–1485。若五 seed 主窗口未完整，任何已有阶段都只是进度快照，不构成提前否定或停止理由。区间为 outer-seed 配对 t 区间；五 seed 离散符号翻转双侧检验的分辨率有限，且划分重叠，不能将其解释为独立化学任务的总体因果保证。

## 已完成历史 validation 基线：仅作历史对照

这些是本次恢复审计的真实历史 validation 指标，**不是新 posterior-EPIG/posterior-IVR 的结果**。训练协议、split、初始化、L333尺度与checkpoint需逐项匹配后才可复用。正式辅助后验如有BLR备选，将逐seed注明真实family，不统一冒称GP。CW-gradient LCMD 仅两 seed；混合版本 historical Free 与最新 CW-Free 单列，不能包装成随机化因果消融。历史 Kernel-IVR 原 published final report 的 test 指标没有作为本表主结果。

{md(primary_b)}

{md(seed_b,['method','seed','end_budget','raw_aulc','terminal_nrmse','terminal_v1_mae','terminal_v2_mae','terminal_v1_rmse','terminal_v2_rmse'])}

全部525/813/1005/1485实际完整窗口见 `baseline_seed_metrics.csv`，所有原始预算点与hash见 `baseline_validation_curves.csv`。历史曲线末期的优势不能取代全程AULC；早期成本也不能掩盖长期终点收益。

{chr(10).join(lfigs) or '尚无可画的学习曲线。'}

`figures/historical_*_raw_curves.png` 为各历史方法的逐 seed 原始非单调曲线。没有正式 EPIG 结果时，不生成伪造的 EPIG 学习曲线/配对效果图。

## 实际计算成本

{md(pd.DataFrame([cost]))}

资格为 cheap auxiliary model fits，不计作 QGeoGNN fits。acquisition时间为完成状态的测量值之和，包含重复评分与条件化；不是神经训练时间。RSS 为进程生命周期峰值，不能把不同状态峰值相加。完整日志、complete receipts 与 `reported_compute_cost.json` 保留计数；未完成/失败attempt应结合原budget ledger单列，而非隐去。
"""
    write(HERE/"RESULTS.md",results)
    limitations=f"""# 适用范围与未决限制

{scope}

1. **数学正确性、概率可信度与真实学习收益不同。** float64解析矩阵、dense重算与规模测试证明实现一致，不证明真实 joint covariance。资格状态 {qstatus}，已完成 {completed_families}/20 状态。
2. **有限模型族。** 本次预定义 raw descriptor/condition 输入、train-fold PCA、常数均值、RBF/Matérn独立GP和线性matrix-normal代理。它们不是充分探索全部冻结 GNN 表征、异方差、Laplace、ensemble校准或nonseparable多输出结构。任一失败只能限定到该族及其预算。
3. **联合依赖识别有限。** OOF full-joint NLL 与对角NLL比较检查跨row相关是否伤害预测，但五fold队列重复暴露、残差异质性与均值失配都会影响它。边缘coverage合格不保证跨样本相关正确；mean失配也可能造成联合资格失败。
4. **同输入未来标签语义。** P1/P2采用独立噪声重复label；这与有限池“同一个潜在观测标签”的自信息语义不同。不能在未声明条件下互换。
5. **目标积分近似。** M2048不是精确真实p*，P1/P2是两个预注册输入假设；短M不稳定只意味着积分敏感，不足以否定EPIG。不得事后选择最有利分布作为主实验。
6. **状态与预测器匹配。** Cheap代理在合法L上拟合，不一定反映QGeoGNN当前可学习方向。体积响应压缩是已有development线索，不证明本次任何无收益主要由网络瓶颈造成。尚未完成匹配新预测器的因子控制。
7. **真实性能证据。** {'已完成五seed主配对，但长期结论仍须对应实际L1485完整轨迹。' if full_primary else '尚未完成五seed主配对；类别E（充分长期轨迹上的真实无收益）未被检验。不能用2–3轮、单seed、代理MI或覆盖率替代主AULC。'}L1005之外未运行时不宣称长期无效，也不挑最好窗口。
8. **五seed统计限制。** 种子是统计单位；fold、batch、行、重复RNG不是新增独立外层划分。t区间与符号翻转仅描述这组开发seed，独立确认须新冻结cohort/split或新实验数据。
9. **历史轨迹异质性。** 保存的旧test报告、两seed CW-LCMD、不同LLM模型/日期/授权例外不可混成严格同协议新对照；原始证据和局限在CODE_AND_EVIDENCE_AUDIT.md中保留。
10. **计算预算。** 资源、墙钟、哈希/标签状态故障只形成证据不足F；未执行结果不是负结果。资格/诊断与正式210次fit各有冻结预算、断点receipts；不自动调用付费LLM API。
11. **门槛与有限样本。** {failure_detail}不可因结果接近边界就事后放宽阈值；下一阶段若引入calibration，须另行冻结有限的校准候选、honest训练侧评估与预算，而非无限调到全部通过。

## 当前可支持的失败类别

{'A（当前预注册概率模型资格不足）与F（真实AL证据不足）；B/C/D/E只能作为有待区分的机制假说。' if qualification_complete and not gpqualified else '由实际资格与完整轨迹逐层判断；没有完整主轨迹时E尚不可判定。'}单点/条件化MI差可支持C类冗余诊断，但不能独立说明真实性能失败。排名很相似只能说明给定状态的目标函数差异小，不能普遍确认B。
"""
    write(HERE/"LIMITATIONS.md",limitations)
    q1=("解析实现与规模已通过；887/6101尚无通过冻结校准门槛的本地代理，五seed完整正式门槛未满足。矩阵正定性及平均联合NLPD门槛本身通过，不能称联合协方差已被证伪。其他seed的合格状态单独报告，不能跨seed认证。" if qualification_complete and not gpqualified else "解析实现与规模已通过，每个seed都有达到预注册资格的本地GP或BLR备选；这构成有限development资格，仍不是全分布校准证明。" if gpqualified else "解析实现与规模已通过；真实概率资格尚未完成，可靠性未定。")
    q2=(f"已获得{len(acq)}个同后验决策/MI诊断；M1024状态的EPIG/IVR平均pointwise Top32重合={primary.top32_overlap_epig_ivr.mean():.3f}，greedy批次重合={primary.greedy_batch_overlap_epig_ivr.mean():.3f}（实际已完成状态，非独立重复）。可有不同决策，但这不是新增真实预测价值的证据，且每状态资格须分别核对。" if not primary.empty else "尚无完成的同状态M1024 acquisition诊断，未知。")
    q3=("五seed主配对已完整，逐seed差值与区间见RESULTS；判断须同时考虑AULC、终点和early/mid/late，而非只选有利指标。" if full_primary else "未知：五seed完整L333→L1005主配对尚未完成，不能给Go/No-Go。")
    q4=("当前最强证据是预注册后验资格不足（A）及真实性能证据不足（F）；不能把尚未测量的性能称为未改善，预测器瓶颈D仍是候选解释。" if qualification_complete and not gpqualified else "若无收益，必须结合校准、同后验差异、batch冗余与预测器控制分层解释；当前不能直接归咎EPIG目标。")
    decision_text=f"""# 下一阶段决策与六个研究问题

当前状态：**{qstatus}；{'主五seedL1005配对完成' if full_primary else '主五seedL1005配对未完成'}**。这是证据范围声明，不是普遍否定EPIG。

本文件首先报告stage1的实际证据。{stage2_note}

1. **当前数据能否可靠计算EPIG？** {q1}
2. **EPIG信息是否超出现有IVR？** {q2}历史Kernel-IVR和新GP代理不同；只有同posterior IVR/EPIG配对才隔离objective因素。
3. **是否改善真实Row-level AULC与终点？** {q3}
4. **若未改善，最支持的原因？** {q4}
5. **是否值得作LLM数值辅助？** 当前不具备直接宣称有益的真实学习证据。可保留为待验证数值候选；只有概率资格、稳定决策和真实Row-level收益足够时才提出受控LLM辅助实验。本次未调用新的付费LLM API。
6. **仍不能确定什么？** 未执行的完整AL/目标分布性能消融、非可分离跨输出模型、GNN表征或异方差代理、匹配新预测器控制、长期L1485价值与独立确认均不能由当前诊断推定。

## 继续执行的依据

{'任一seed没有合格本地后验时，预注册完整正式门槛未满足，不能把不合格代理的210次昂贵重训当成已资格主实验，也不删掉失败seed。应先完整保存所有预注册族的资格/目标敏感性，再根据失败维度提出有限预算、预先冻结的下一代理族；该新资格设计须标明是结果后的第二阶段，不能伪装原预注册。当前研究结论为A/F，非EPIG NO-GO。' if qualification_complete and not gpqualified else '所有seed本地资格完成且通过，按冻结预算继续全部五seed两臂21轮；不能因早期/某seed不利提前停。完成L1005后按冻结extension_rule同时决定两臂L1485；任何预算/依赖阻碍须如实记录。' if gpqualified else '完成当前预注册的60次辅助资格拟合和全部候选族，不因早出现失格取消剩余比较。资格结果揭示前不调结果阈值。'}

## 有限第二阶段nested calibration：明确区分阶段

该设计明确写于stage1资格结果揭示之后，不改stage1的任何阈值、模型选择或已有轨迹，也不能称为原stage1预注册。{stage2_note}设计为先另立不可变stage2协议与hash，再一次性执行：保留现有四个模型族、三个outer OOF fold和原优化预算；每个outer训练fold内部再三折产生合法inner-OOF误差。只从inner-OOF拟合两个预定义校准候选——端点marginal Gaussian NLL温度与端点central80比例尺度——并保留identity作为控制。两种尺度可从同一inner预测计算，不重复拟合模型。outer held-out标签只用于最终资格评分，不用于估计该fold温度；五seed各自选择，不汇合其他seed真值。

若端点校准尺度为正对角D，则对所有latent/noise/cross blocks一致施加DΣD，保留正定性和原相关结构。该产物应另称**校准Gaussian联合代理**，不能声称仍是原GP先验/似然下未经修改的精确后验。温度仅改变幅度，不修复Gaussian的尖峰重尾、条件异方差、错误跨row相关或均值失配；中央80%覆盖即使改善，也不保证完整概率评分、95%尾部覆盖、相关结构或AL收益。小inner训练模型与outer模型样本量不同，校准迁移本身也需披露。

有限预算：四族×五seed×三outer×(三inner+一次outer拟合)=240次辅助拟合；若所有seed资格足够，再为各seed选定族五seed合计最多20次inner+全L333最终拟合，总计260次计划完成、最多263次含中断attempt、1800秒墙钟、逐fold缓存与attempt账本。模型族、两个校准候选、资格门槛、选择准则和预算均在stage2结果揭示前另行冻结，不用AL学习曲线挑温度。若仍无完整五seed合格后验，报告该有限校准族失败，不继续无限追加方法直到通过。是否启动正式210次QGeoGNN拟合，仍须满足新冻结资格与资源边界；**stage1门槛保持不变，实际stage2执行不属于原stage1结果**。

独立确认策略：目前五seed属于development，后续应在新冻结Row数据或新的独立实验批次预先指定选点策略、模型族、目标分布和主要指标；新测试真值只在全部方法与预算冻结后一次性评价。不得反复查看已有test来调整本轮代理。

{scope}
"""
    write(HERE/"NEXT_STAGE_DECISION.md",decision_text)
    manifest={"kind":"reporting_only_not_scientific_decision_seal","created_unix":time.time(),
              "script_sha256":sha(Path(__file__)),"source_inputs":{str(p.relative_to(HERE)):sha(p) for p in sorted(INPUTS)},
              "outputs":{str(p.relative_to(HERE)):sha(p) for p in sorted(OUTPUTS)},"qualification_status":qstatus,
              "five_seed_primary_complete":full_primary,"new_completed_qgeognn_fits":formal_new}
    write(HERE/"reporting_manifest.json",json.dumps(manifest,indent=2,ensure_ascii=False))
    print(json.dumps({"reports":6,"figures":len([p for p in OUTPUTS if p.suffix==".png"]),"qualification":qstatus,"formal_new_fits":formal_new},ensure_ascii=False))


if __name__=="__main__":
    main()
