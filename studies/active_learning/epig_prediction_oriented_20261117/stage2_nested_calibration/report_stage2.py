"""Report actual sealed nested-calibration evidence without fitting or labels.

Reads only completed stage-two OOF receipts, decision/budget/status metadata,
and saved formal validation receipts. No raw source table or label store is
opened. Missing candidates or trajectories stay unexecuted, never imputed.
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
from scipy.stats import t

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
FIG=HERE/"figures"
SEEDS=(157,887,2357,6101,12203)
SCHEMES=("identity","nll","central80")
ARMS=("posterior_epig_conditional_greedy32","posterior_ivr_conditional_greedy32")
INPUTS=set()
OUTPUTS=set()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path,default=None):
    path=Path(path)
    if not path.exists():
        return default
    INPUTS.add(path)
    return json.loads(path.read_text())


def table(path):
    path=Path(path)
    if not path.exists():
        return pd.DataFrame()
    INPUTS.add(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write(path,text):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".reporting.tmp")
    tmp.write_text(text.rstrip()+"\n")
    tmp.replace(path)
    OUTPUTS.add(path)


def csv(path,frame):
    write(path,frame.to_csv(index=False))


def md(frame,columns=None):
    if frame is None or frame.empty:
        return "尚无完成结果 / 未执行。"
    if columns is not None:
        frame=frame[[c for c in columns if c in frame]]
    def val(v):
        if isinstance(v,(list,dict)):
            return json.dumps(v,ensure_ascii=False).replace("|","\\|")
        if pd.isna(v):
            return "—"
        if isinstance(v,(float,np.floating)):
            return f"{v:.6f}"
        return str(v).replace("|","\\|").replace("\n"," ")
    out=["| "+" | ".join(frame.columns)+" |","| "+" | ".join(["---"]*len(frame.columns))+" |"]
    return "\n".join(out+["| "+" | ".join(val(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None)])


def figsave(name,fig):
    path=FIG/name
    fig.savefig(path,dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    OUTPUTS.add(path)
    return f"![{name}](figures/{name})"


def qualification_data():
    rows=[]
    arrays=[]
    temperatures=[]
    folds=[]
    for complete in sorted((HERE/"qualification").glob("seed_*/complete.json")):
        receipt=read(complete)
        for name,expected in receipt.get("files",{}).items():
            path=complete.parent/name
            if not path.exists() or digest(path)!=expected:
                raise RuntimeError(f"completed nested evidence hash mismatch: {path}")
        suffix=complete.parent.name.removeprefix("seed_")
        seed_text,remainder=suffix.split("_",1)
        family,scheme=remainder.rsplit("_",1)
        seed=int(seed_text)
        if scheme not in SCHEMES:
            raise ValueError("unknown nested calibration scheme")
        result=read(complete.parent/"metrics.json")
        metrics=result["metrics"]
        for j,endpoint in enumerate(("V1","V2")):
            rows.append({"seed":seed,"family":family,"calibration":scheme,"endpoint":endpoint,
                         **{field:metrics[field][j] for field in ("rmse","mae","marginal_nll","coverage50","coverage80","coverage95")},
                         "standardized_residual_rms":np.sqrt(metrics["standardized_residual_second_moment"][j])})
        for fold in result["folds"]:
            temperature=np.asarray(fold.get("temperature",[np.nan,np.nan]),dtype=float)
            for j,endpoint in enumerate(("V1","V2")):
                temperatures.append({"seed":seed,"family":family,"calibration":scheme,"fold":fold["fold"],"endpoint":endpoint,
                                     "width_multiplier_D":float(temperature[j]),"variance_multiplier_T":float(temperature[j]**2)})
            folds.append({"seed":seed,"family":family,"calibration":scheme,"fold":fold["fold"],"held_out_rows":fold["held_out_rows"],
                          "full_joint_nlpd_per_scalar":fold.get("full_joint_nll_per_scalar",np.nan),
                          "diagonal_nlpd_per_scalar":fold.get("diagonal_nll_per_scalar",np.nan),
                          "joint_minus_diagonal_nlpd_per_scalar":fold.get("full_joint_nll_per_scalar",np.nan)-fold.get("diagonal_nll_per_scalar",np.nan),
                          "minimum_predictive_eigenvalue":fold.get("minimum_predictive_eigenvalue",np.nan)})
        path=complete.parent/"oof.npz"
        INPUTS.add(path)
        with np.load(path,allow_pickle=False) as npz:
            def take(*names):
                for name in names:
                    if name in npz:
                        return npz[name].copy()
                raise KeyError(f"missing OOF array, expected {names}")
            arrays.append({"seed":seed,"family":family,"calibration":scheme,
                           "truth":take("authorized_L333_truth_ml","y_true","truth"),
                           "mean":take("oof_mean_ml","oof_mean"),
                           "cov":take("oof_covariance_blocks_ml2","oof_covariance_blocks")})
    saved=table(HERE/"qualification/seed_metrics.csv")
    return (saved if not saved.empty else pd.DataFrame(rows)),pd.DataFrame(temperatures),pd.DataFrame(folds),arrays


def graphics(arrays,temperatures,metrics,folds):
    links=[]
    if arrays:
        families=sorted({a["family"] for a in arrays})
        fig,axes=plt.subplots(2,len(families),figsize=(3.7*len(families),6.4),sharex=True,sharey=True,squeeze=False)
        probabilities=np.linspace(.1,.95,18)
        for k,family in enumerate(families):
            for scheme in SCHEMES:
                selected=[a for a in arrays if a["family"]==family and a["calibration"]==scheme]
                if not selected:
                    continue
                z=np.vstack([(a["truth"]-a["mean"])/np.sqrt(np.diagonal(a["cov"],axis1=1,axis2=2)) for a in selected])
                for j in range(2):
                    axes[j,k].plot(probabilities,[(np.abs(z[:,j])<=ndtri((1+p)/2)).mean() for p in probabilities],".-",label=scheme)
            for j in range(2):
                axes[j,k].plot([0,1],[0,1],"k--",linewidth=.8)
                axes[j,k].set(title=f"{family}: V{j+1}",xlabel="Nominal central coverage",ylabel="Observed OOF coverage",xlim=(0,1),ylim=(0,1))
                axes[j,k].grid(alpha=.2)
        axes[0,-1].legend(fontsize=8)
        links.append(figsave("nested_calibration_curves.png",fig))
    if not temperatures.empty:
        families=sorted(temperatures.family.unique())
        fig,axes=plt.subplots(1,len(families),figsize=(4*len(families),4.1),sharey=True,squeeze=False)
        for axis,family in zip(axes[0],families):
            for scheme in ("nll","central80"):
                for endpoint,linestyle in (("V1","-"),("V2","--")):
                    group=temperatures.loc[temperatures.family.eq(family)&temperatures.calibration.eq(scheme)&temperatures.endpoint.eq(endpoint)]
                    if group.empty:
                        continue
                    summary=group.groupby("seed").width_multiplier_D.agg(["mean","std"]).reindex(SEEDS)
                    axis.errorbar(range(5),summary["mean"],yerr=summary["std"].fillna(0),marker="o",linestyle=linestyle,capsize=2,label=f"{scheme}/{endpoint}")
            axis.axhline(1,color="grey",linewidth=.7)
            axis.set(title=family,xticks=range(5),xticklabels=SEEDS,ylabel="Width multiplier D (mean +/- fold SD)")
            axis.tick_params(axis="x",rotation=40,labelsize=7)
            axis.grid(axis="y",alpha=.2)
        axes[0,-1].legend(fontsize=7)
        links.append(figsave("nested_width_temperatures.png",fig))
    if not folds.empty:
        fig,axis=plt.subplots(figsize=(11,4.2))
        groupings=list(folds.groupby(["family","calibration"],sort=True))
        for k,(_,group) in enumerate(groupings):
            byseed=np.array([np.average(g.joint_minus_diagonal_nlpd_per_scalar,weights=g.held_out_rows) for _,g in group.groupby("seed")])
            axis.scatter(np.full(len(byseed),k),byseed,s=22)
            axis.plot([k-.17,k+.17],[byseed.mean()]*2,color="k",linewidth=1.5)
        axis.axhline(.1,color="firebrick",linestyle="--",label="Unchanged qualification limit")
        axis.axhline(0,color="grey",linewidth=.6)
        axis.set(title="Held-out full joint vs diagonal NLPD",ylabel="Joint - diagonal NLPD, per scalar",xticks=range(len(groupings)),xticklabels=[f"{f}\n{s}" for (f,s),_ in groupings])
        axis.tick_params(axis="x",rotation=30,labelsize=7)
        axis.legend(fontsize=8)
        axis.grid(axis="y",alpha=.2)
        links.append(figsave("nested_joint_nlpd.png",fig))
    return links


def formal_data():
    rows=[]
    decision=read(HERE/"qualification/decision.json",{})
    for path in sorted((HERE/"runtime").glob("seed_*/*/round_*/fit_receipt.json")):
        receipt=read(path)
        if receipt.get("arm") not in ARMS:
            continue
        rows.append({"seed":receipt["seed"],"method":receipt["arm"],"family":decision.get("selected_family_by_seed",{}).get(str(receipt["seed"])),
                     "calibration":decision.get("calibration_by_seed",{}).get(str(receipt["seed"])),"budget":receipt["budget"],"nrmse":receipt["validation"]["combined_nrmse"],
                     "reused":receipt["reused"],"training_seconds":receipt.get("training_seconds",0),
                     **{k:receipt["validation"][k] for k in ("V1_RMSE","V2_RMSE","V1_MAE","V2_MAE")},
                     "checkpoint_sha256":receipt["checkpoint_sha256"]})
    curves=pd.DataFrame(rows)
    metrics=[]
    if not curves.empty:
        for (method,seed),group in curves.groupby(["method","seed"]):
            if group.budget.duplicated().any():
                raise RuntimeError("duplicate stage-two formal budget")
            for start,end,stage in ((333,525,"early"),(525,813,"mid"),(813,1005,"late"),(333,1005,"primary_full"),(333,1485,"extension_full")):
                window=group.loc[group.budget.between(start,end)].sort_values("budget")
                if not np.array_equal(window.budget.to_numpy(),np.arange(start,end+1,32)):
                    continue
                last=window.iloc[-1]
                metrics.append({"seed":seed,"method":method,"family":last.family,"calibration":last.calibration,"stage":stage,"start_budget":start,"end_budget":end,
                                "raw_aulc":np.trapezoid(window.nrmse.to_numpy(),window.budget.to_numpy())/(end-start),"terminal_nrmse":last.nrmse,
                                **{k:last[k] for k in ("V1_RMSE","V2_RMSE","V1_MAE","V2_MAE")}})
    metrics=pd.DataFrame(metrics)
    pairs=[]
    if not metrics.empty:
        for stage,group in metrics.groupby("stage"):
            a=group.loc[group.method.eq(ARMS[0])].set_index("seed")
            b=group.loc[group.method.eq(ARMS[1])].set_index("seed")
            for seed in sorted(set(a.index)&set(b.index)):
                pairs.append({"seed":seed,"stage":stage,"delta_aulc_epig_minus_ivr":a.loc[seed,"raw_aulc"]-b.loc[seed,"raw_aulc"],
                              "delta_terminal_epig_minus_ivr":a.loc[seed,"terminal_nrmse"]-b.loc[seed,"terminal_nrmse"]})
    return curves,metrics,pd.DataFrame(pairs)


def paired_summary(pairs):
    rows=[]
    if pairs.empty:
        return pd.DataFrame()
    for stage,group in pairs.groupby("stage"):
        for field in ("delta_aulc_epig_minus_ivr","delta_terminal_epig_minus_ivr"):
            values=group[field].to_numpy(float)
            n=len(values)
            sd=np.std(values,ddof=1) if n>1 else np.nan
            half=t.ppf(.975,n-1)*sd/np.sqrt(n) if n>1 else np.nan
            permutation=np.array([abs(np.mean(values*np.asarray(signs))) for signs in itertools.product((-1,1),repeat=n)])
            rows.append({"stage":stage,"metric":field,"outer_seeds":n,"mean_delta":values.mean(),"sample_sd":sd,
                         "t95_low":values.mean()-half,"t95_high":values.mean()+half,
                         "exact_sign_permutation_p":np.mean(permutation>=abs(values.mean())-1e-14)})
    return pd.DataFrame(rows)


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    config=read(HERE/"config.json",{})
    decision=read(HERE/"qualification/decision.json",{})
    budget=read(HERE/"compute_budget.json",{})
    finalization=read(HERE/"finalization.json",{})
    formal_status=read(HERE/"formal_status.json",{})
    metrics,temperatures,folds,arrays=qualification_data()
    curves,formal,pairs=formal_data()
    pair_statistics=paired_summary(pairs)
    csv(HERE/"reported_width_temperatures.csv",temperatures)
    csv(HERE/"reported_joint_fold_metrics.csv",folds)
    csv(HERE/"formal_validation_curves.csv",curves)
    csv(HERE/"formal_seed_metrics.csv",formal)
    csv(HERE/"formal_paired_differences.csv",pairs)
    csv(HERE/"formal_paired_summary.csv",pair_statistics)
    links=graphics(arrays,temperatures,metrics,folds)
    if not curves.empty:
        fig,axes=plt.subplots(1,5,figsize=(16,3.6),sharex=True,sharey=True)
        for axis,seed in zip(axes,SEEDS):
            for arm in ARMS:
                group=curves.loc[curves.seed.eq(seed)&curves.method.eq(arm)].sort_values("budget")
                if not group.empty:
                    axis.plot(group.budget,group.nrmse,".-",markersize=3,label=arm)
            axis.set(title=f"Seed {seed}",xlabel="Labels",ylabel="Combined validation NRMSE")
            axis.grid(alpha=.2)
        axes[-1].legend(fontsize=6)
        links.append(figsave("stage2_formal_raw_curves.png",fig))
    if not pairs.empty:
        full_pairs=pairs.loc[pairs.stage.eq("primary_full")]
        if not full_pairs.empty:
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            for axis,field in zip(axes,("delta_aulc_epig_minus_ivr","delta_terminal_epig_minus_ivr")):
                axis.axhline(0,color="k",linewidth=.8)
                axis.bar(full_pairs.seed.astype(str),full_pairs[field])
                axis.set(title=field.replace("_"," "),xlabel="Outer seed",ylabel="EPIG - same-posterior IVR")
            links.append(figsave("stage2_formal_paired_effects.png",fig))
    gate_rows=[{**g,"reasons":"; ".join(g.get("reasons",[]))} for g in decision.get("models",[])]
    gates=pd.DataFrame(gate_rows)
    if not metrics.empty:
        if "calibration" not in metrics and "scheme" in metrics:
            metrics=metrics.rename(columns={"scheme":"calibration"})
        aggregate=metrics.groupby(["family","calibration","endpoint"],as_index=False).agg(
            outer_seeds=("seed","nunique"),coverage80=("coverage80","mean"),coverage95=("coverage95","mean"),
            marginal_nlpd=("marginal_nll","mean"),standardized_residual_rms=("standardized_residual_rms",lambda v:np.sqrt(np.mean(v*v))))
    else:
        aggregate=pd.DataFrame()
    failures=[]
    if not gates.empty:
        for scheme,group in gates.groupby("calibration",sort=False):
            local=metrics.loc[metrics.calibration.eq(scheme)]
            failures.append({"calibration":scheme,"candidate_states":len(group),"qualified_candidates":int(group.qualified.sum()),
                             "endpoint_coverage80_above_090":int((local.coverage80>.90).sum()),
                             "endpoint_coverage80_below_070":int((local.coverage80<.70).sum()),
                             "endpoint_coverage95_outside_bounds":int(((local.coverage95<.88)|(local.coverage95>.99)).sum()),
                             "endpoint_residual_RMS_above_150":int((local.standardized_residual_rms>1.50).sum()),
                             "endpoint_residual_RMS_below_070":int((local.standardized_residual_rms<.70).sum()),
                             "candidate_joint_NLPD_degradation_above_010":int((group.joint_nlpd_degradation_per_scalar>.10).sum())})
    failures=pd.DataFrame(failures)
    csv(HERE/"reported_failure_decomposition.csv",failures)
    old=table(PARENT/"qualification/seed_metrics.csv")
    identity_check=pd.DataFrame()
    if not old.empty and not metrics.empty:
        fields=["rmse","mae","marginal_nll","coverage80","coverage95","standardized_residual_rms"]
        merged=metrics.loc[metrics.calibration.eq("identity")].merge(old,on=["seed","family","endpoint"],suffixes=("_stage2","_stage1"))
        identity_check=merged[["seed","family","endpoint"]].copy()
        for field in fields:
            identity_check[f"delta_{field}"]=merged[f"{field}_stage2"]-merged[f"{field}_stage1"]
    csv(HERE/"reported_identity_vs_stage1.csv",identity_check)
    identity_max=(float(np.abs(identity_check.filter(like="delta_").to_numpy()).max()) if not identity_check.empty else None)
    chosen=pd.DataFrame([{"seed":seed,"family":decision.get("selected_family_by_seed",{}).get(str(seed)),
                          "calibration":decision.get("calibration_by_seed",{}).get(str(seed)),
                          "status":("QUALIFIED_LOCAL" if decision.get("selected_family_by_seed",{}).get(str(seed)) else "NO_LOCAL_QUALIFIED_POSTERIOR" if decision else "PENDING")} for seed in SEEDS])
    status=decision.get("status","NOT_YET_EXECUTED_OR_INCOMPLETE")
    ready=bool(decision.get("run_allowed",False))
    complete_states=len(arrays)
    newfits=int((~curves.reused.astype(bool)).sum()) if not curves.empty else 0
    full=not pairs.empty and set(pairs.loc[pairs.stage.eq("primary_full"),"seed"])==set(SEEDS)
    scope="本阶段由stage1已知结果触发，虽在新结果前另行冻结，仍属结果后的development研究；五个Row划分重叠，不是独立确认。"
    inference="同一已拟合family的正对角DΣD变换不改变Gaussian单点/条件化EPIG；它可改变trace-IVR的端点风险权重。校准改变选模资格或family时才可间接改变EPIG决策，不能把幅度校准直接说成EPIG获得新信息。"
    report=f"""# Nested covariance calibration：实际执行结果

资格状态：**{status}**。实际完成{complete_states}/60个seed×family×calibration候选；这些共享最多240次base辅助拟合，不是60个独立数据划分。{'所有seed资格齐全，可按新seal进入原QGeoGNN正式比较。' if ready else '完整五seed正式后验资格尚未齐全，不删除seed、不放宽原门槛。'}新完成QGeoGNN拟合{newfits}次；{'五seedL1005配对已完整。' if full else '五seedL1005配对未完整，真实效能仍未判定。'}

{scope}

## 方法与不变量

参见冻结的[STAGE2_PROTOCOL.md](STAGE2_PROTOCOL.md)与config。四族、三outer×(三inner+一outer)；temperature仅由outer训练侧inner-OOF产生，outer-held-out只评分。标准化、PCA、中心与base参数都在该fit训练子集学习。无候选U/validation/test监督，三个scheme共享同一outer base mean/covariance。

identity的D_width=1；nll的D_width=sqrt(mean(z²))，是每端点**marginal** Gaussian NLL最优尺度，并非full-joint NLL最优；central80的D_width=Q80(|z|)/Φ⁻¹(.9)。D_width是sigma/区间宽度倍率，其平方才是variance temperature；无任意bounds，非法/非正/非有限D明确失败。全部latent/noise/cross covariance同步DΣD，mean不变。{inference}

此代理称为nested校准Gaussian联合代理，不能声称仍是原GP似然/先验未经修改的精确后验。温度不改变Gaussian形状或相关系数，不能修复尖峰重尾、错均值、条件异方差或错误跨row依赖；80%中心覆盖改善不保证95%尾部、NLL、joint NLL或AL收益。

## 本地选择与完整门槛

{md(chosen)}

原门槛每seed/endpoint仍为80%coverage[.70,.90]、95%coverage[.88,.99]、标准化残差RMS[.70,1.50]；full-joint相对diagonal NLPD每scalar退化≤.10且covariance正定。只用本seed OOF，在合格GP family×calibration中优先最低marginal NLPD；无GP才选本seed合格BLR。跨seed pooled表只描述，不用于选择。

{md(gates,['seed','family','calibration','qualified','mean_marginal_nlpd','joint_nlpd_degradation_per_scalar','reasons'])}

## 独立失败维度复核：中央覆盖与尾部不能互相替代

{md(failures)}

上表candidate计数以seed×family×calibration为单位，endpoint计数每候选有两端点；同一候选可因多个门槛失败，不能将各列相加当独立失败次数。identity与stage1同seed/family/endpoint的RMSE、MAE、marginal NLPD、coverage80/95、标准化残差RMS最大绝对差：{identity_max if identity_max is not None else '尚未可比较'}，详细见`reported_identity_vs_stage1.csv`。这是复现控制，不是新数据确认。

{'实际identity只有3/20候选合格，NLL温度0/20、central80为0/20。NLL尺度主要仍未消除中央80%过覆盖；central80缩窄后中央门槛可改善，但20/20候选的V1标准化残差RMS仍超过1.5（部分同时95%覆盖不足或joint NLPD退化）。这支持“单一Gaussian幅度不能同时协调中央质量与尾部误差”的有限代理/校准失配解释，不能据此断言真实残差必属于某个特定重尾分布，也不是EPIG objective失败。' if len(failures)==3 and complete_states==60 and dict(zip(failures.calibration,failures.qualified_candidates))=={'identity':3,'nll':0,'central80':0} else '按实际完成候选逐维解释。中央coverage、尾部coverage、残差RMS和joint NLL可能相互牵制；有限温度校准未通过只限定该模型/校准族，不代表EPIG或更一般联合代理无效。'}

## 描述性OOF结果与实际温度

{md(aggregate)}

每seed原始表见`qualification/seed_metrics.csv`，每fold宽度和方差倍率见`reported_width_temperatures.csv`，full-joint评分见`reported_joint_fold_metrics.csv`。所有未完成候选保持缺失，不用其他seed或identity替代。

{chr(10).join(links) or '尚无完成的数值产物，图未生成。'}

## 计算、缓存与边界

新增辅助拟合attempt账本实际记录数：{len(budget.get('attempts',[]))}（精确阶段计数以`compute_budget.json`为准）；finalization状态：`{json.dumps(finalization,ensure_ascii=False)}`。资格240次加最终最多20次、总cap260次及1800秒预算不因效果好坏修改；中断attempt须按账本披露。阶段1没有保存full outer covariance，阶段2outer refit是明确记账的新联合评分计算，不冒充resume重复。

不调用付费LLM；真实神经训练仍是原模型、scratch、原训练配置/初始化/validation checkpoint selection和固定L333尺度。概率资格通过不等于主任务已完成。

## 真实Row-level轨迹

正式状态：`{json.dumps(formal_status,ensure_ascii=False)}`。

{md(formal)}

{md(pairs)}

{md(pair_statistics)}

原始非单调曲线见`formal_validation_curves.csv`。完整窗口缺任何预算点不计算该窗口AULC；不插值、不best-so-far。未完成五seed21轮主配对时不作效能Go/No-Go。L1485未运行时不宣称长期无效。配对负差代表EPIG更优；样本SD、t4区间与精确符号翻转检验仅描述五个重叠开发split，统计单位outer seed，batch/行/innerfold不增加n。历史Random/LCMD/Kernel-IVR与Free的审计曲线保留在[父研究RESULTS.md](../RESULTS.md)，不以异质历史轨迹补齐新方法。
"""
    write(HERE/"STAGE2_RESULTS.md",report)
    decision_report=f"""# 第二阶段证据决策

实际资格状态：**{status}**；完成{complete_states}/60个校准候选。{scope}

1. **能否可靠计算EPIG？** 数值实现已通过stage1矩阵/规模测试；stage2只检验有限Gaussian幅度校准资格。{'五seed本地资格均有模型，但仍是有限development资格。' if ready else '当前五seed完整资格尚未齐全，不能用可计算性代替可信概率资格。'}
2. **EPIG信息是否超出IVR？** {inference}同posterior真实学习对照才能识别objective的效用；stage1排名诊断不能直接回答新增预测价值。
3. **是否改善Row AULC/终点？** {'完整五seed主配对已有，逐seed表在STAGE2_RESULTS.md，须同时解读AULC/终点/早中晚。' if full else '尚无完整五seedL1005主配对，未知。'}温度资格改善不等于真实收益。
4. **若无收益的原因？** 依据实际资格、joint评分、批量冗余与预测器控制分别讨论；未执行的性能不能被称为无收益。Gaussian尖峰重尾/均值失配限制仍在，不能把失败归到EPIG目标。
5. **是否进入LLM辅助？** 仅有限概率资格/代理MI仍不充分。需真实Row学习收益与稳定性证据；本次未调用新付费API。
6. **仍未知？** 完整长期AL、独立cohort、非可分离多输出、异方差/重尾代理和匹配新预测器控制，均不从本温度试验推定。

{'按冻结预算执行全部五seed两臂21轮，不因某seed或短阶段不利提前停止；完成L1005后按原extension_rule共同决定L1485。' if ready else '保留完整候选和失败原因；不删seed，不事后改门槛，也不继续无限增加校准方法直到通过。若预算未完且候选仍在执行，继续完成冻结比较；若全部60状态完成仍不齐全，则本有限校准族为资格不足A/F，不是EPIG普遍NO-GO。'}

{md(failures)}

{'全部60候选已完整执行。校准没有新增合格候选：仅identity保留原3seed本地后验，887/6101仍不齐全。NLL与中央80%温度的不同失败维度见STAGE2_RESULTS，不把它缩写成EPIG失败。' if complete_states==60 and not ready else '当前完成范围以实际候选表为准。'}充分真实AL验证目标尚未完成时，必须在任何总体总结中显著披露；本阶段不能提供不存在的五seedEPIG AULC/终点表。

本阶段不是stage1原预注册的一部分；报告必须同时保留stage1与stage2，不隐藏最初资格失败或挑选有利窗口。真正独立确认仍需新冻结数据/实验批次。
"""
    write(HERE/"STAGE2_NEXT_STAGE_DECISION.md",decision_report)
    write(HERE/"README.md",f"""# Stage 2：nested calibration

{scope}

- [执行前协议](STAGE2_PROTOCOL.md)
- [真实资格、校准与Row结果](STAGE2_RESULTS.md)
- [证据决策与未决问题](STAGE2_NEXT_STAGE_DECISION.md)
- 科学配置/实现/输入由独立freeze/seal绑定；`reporting_manifest.json`仅绑定报告，不改变科学决定。

本报告可重复执行：`/Users/fish/miniforge3/envs/fish/bin/python {Path(__file__).resolve()}`。它只读取完成的OOF/fit receipts和metadata，不拟合、不揭示标签。实际状态：{status}。
""")
    manifest={"schema":"stage2-reporting-only-v1","created_unix":time.time(),"script_sha256":digest(__file__),
              "inputs":{str(p.relative_to(PARENT)):digest(p) for p in sorted(INPUTS)},
              "outputs":{str(p.relative_to(PARENT)):digest(p) for p in sorted(OUTPUTS)},
              "qualification_status":status,"completed_candidate_states":complete_states,"new_completed_QGeoGNN_fits":newfits}
    write(HERE/"reporting_manifest.json",json.dumps(manifest,indent=2,ensure_ascii=False))
    print(json.dumps({"qualification_status":status,"completed_candidates":complete_states,"formal_new_fits":newfits,"figures":len([p for p in OUTPUTS if p.suffix==".png"])}))


if __name__=="__main__":
    main()
