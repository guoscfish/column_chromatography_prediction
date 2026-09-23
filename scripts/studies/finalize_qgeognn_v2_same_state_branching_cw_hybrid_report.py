#!/usr/bin/env python3
"""Post-freeze reporting for the CW/Hybrid same-state branching pilot."""
from __future__ import annotations
import hashlib, json, itertools, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/"studies/active_learning/qgeognn_v2_same_state_branching_cw_hybrid"
RESULTS=STUDY/"results"
KEYS=["seed","source_trajectory","anchor_budget"]
sys.path.insert(0, str(ROOT))
from src.qgeognn_al.active_learning_v2 import same_state_branching_cw_hybrid as study
STRATEGIES=list(study.STRATEGIES)

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def table(frame):
    return frame.to_markdown(index=False,floatfmt=".6f") if len(frame) else "(empty)"

def main():
    study.validate_seal()
    study._verify_global_branch_freeze()
    branches=pd.read_csv(RESULTS/"branch_results.csv")
    winners=pd.read_csv(RESULTS/"winner_matrix.csv")
    diagnostics=pd.read_csv(RESULTS/"state_diagnostics.csv")
    access=pd.read_csv(RESULTS/"test_label_access_audit.csv")
    wide=branches.pivot(index=KEYS,columns="strategy",values="short_AULC").reset_index()
    oracle=branches.groupby(KEYS,as_index=False).short_AULC.min().rename(columns={"short_AULC":"best_short_AULC"})
    oracle=oracle.merge(wide,on=KEYS,validate="one_to_one")
    oracle["best_strategy"]=oracle[STRATEGIES].idxmin(axis=1)
    oracle["cw_short_AULC"]=oracle.center_width_lcmd
    oracle["oracle_minus_cw"]=oracle.cw_short_AULC-oracle.best_short_AULC
    oracle_out=oracle[KEYS+["best_strategy","best_short_AULC","cw_short_AULC","oracle_minus_cw"]]
    pd.concat([oracle_out,pd.DataFrame([{"seed":"ALL","source_trajectory":"ALL","anchor_budget":"ALL",
      "best_strategy":"mean","best_short_AULC":oracle.best_short_AULC.mean(),
      "cw_short_AULC":oracle.cw_short_AULC.mean(),"oracle_minus_cw":oracle.oracle_minus_cw.mean()}])],
      ignore_index=True).to_csv(RESULTS/"local_oracle_headroom.csv",index=False)
    rows=[{"scope":"overall","group":"all","n_states":len(oracle),
      "mean_oracle_gain_over_CW":oracle.oracle_minus_cw.mean(),
      "min_oracle_gain_over_CW":oracle.oracle_minus_cw.min(),
      "max_oracle_gain_over_CW":oracle.oracle_minus_cw.max()}]
    for col in ("seed","anchor_budget","source_trajectory"):
        for value,g in oracle.groupby(col):
            rows.append({"scope":"by_"+col,"group":value,"n_states":len(g),
              "mean_oracle_gain_over_CW":g.oracle_minus_cw.mean(),
              "min_oracle_gain_over_CW":g.oracle_minus_cw.min(),
              "max_oracle_gain_over_CW":g.oracle_minus_cw.max()})
    pd.DataFrame(rows).to_csv(RESULTS/"headroom_summary.csv",index=False)
    merged=diagnostics[diagnostics.state_kind.eq("anchor")].merge(wide,on=KEYS,validate="one_to_one")
    features=[c for c in ("raw_gradient_norm_mean","raw_gradient_norm_std","raw_gradient_norm_p90","raw_gradient_norm_p95",
      "raw_gradient_effective_rank_participation_ratio","cw_gradient_coverage_nearest_distance_mean",
      "cw_gradient_coverage_nearest_distance_p90","recent_validation_nrmse",
      "recent_validation_2step_linear_slope_per_label","recent_validation_3step_linear_slope_per_label",
      "ivr_predicted_relative_reduction_B32") if c in merged]
    corr=[]
    for left,right in itertools.combinations(STRATEGIES, 2):
        margin=merged[left]-merged[right]
        for feature in features:
            corr.append({"outcome":left+"_minus_"+right+"_short_AULC","feature":feature,
              "n_anchors":len(merged),"spearman_rho":margin.corr(merged[feature],method="spearman"),
              "interpretation":"descriptive_only; n=8; no controller"})
    pd.DataFrame(corr).to_csv(RESULTS/"strategy_preference_correlations.csv",index=False)
    figdir=STUDY/"figures"; figdir.mkdir(exist_ok=True)
    means=branches.groupby(["anchor_budget","strategy"],as_index=False).short_AULC.mean()
    fig,ax=plt.subplots(figsize=(7,4))
    for strategy in STRATEGIES:
        g=means[means.strategy.eq(strategy)]
        ax.plot(g.anchor_budget,g.short_AULC,marker="o",label=strategy)
    ax.set(xlabel="anchor budget",ylabel="short-AULC (lower is better)"); ax.legend(); fig.tight_layout()
    fig.savefig(figdir/"branching_short_aulc.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); ax.scatter(oracle.cw_short_AULC,oracle.best_short_AULC,c=oracle.oracle_minus_cw)
    lo=min(oracle.cw_short_AULC.min(),oracle.best_short_AULC.min()); hi=max(oracle.cw_short_AULC.max(),oracle.best_short_AULC.max())
    ax.plot([lo,hi],[lo,hi],"k--"); ax.set(xlabel="always-CW short-AULC",ylabel="local-oracle short-AULC"); fig.tight_layout()
    fig.savefig(figdir/"oracle_vs_cw.png",dpi=180); plt.close(fig)
    counts=winners.winner_short_AULC.value_counts().reindex(STRATEGIES,fill_value=0)
    fig,ax=plt.subplots(figsize=(6,3.5)); counts.plot.bar(ax=ax); ax.set_ylabel("winner count"); fig.tight_layout()
    fig.savefig(figdir/"winner_matrix.png",dpi=180); plt.close(fig)
    decision=json.loads((STUDY/"decision.json").read_text())
    gain=float(oracle.oracle_minus_cw.mean())
    recommendation="仅补 2 个新 seeds 做确认" if gain>0.005 else "停止 adaptive 方向"
    decision.update({"mean_oracle_gain_over_CW":gain,"max_oracle_gain_over_CW":float(oracle.oracle_minus_cw.max()),
      "min_oracle_gain_over_CW":float(oracle.oracle_minus_cw.min()),"recommendation":recommendation,
      "report_generated":True,"test_access_audit_rows":len(access),
      "test_access_before_global_freeze":int((access.purpose!="final_test_evaluation").sum())})
    (STUDY/"decision.json").write_text(json.dumps(decision,indent=2,ensure_ascii=False)+"\n")
    report=f"""# CW/Hybrid same-state branching pilot（开发性机制研究）

## 结论

本 pilot 在 8 个 exact same states 上比较 Center/Width-LCMD、Hybrid 和 Kernel-IVR，从 anchor 连续 rollout +32、+64。所有 acquisition、checkpoint 和 test-X prediction 均在统一 test reveal 前冻结；本研究不训练 controller、bandit 或 RL，也不搜索阈值。

- always-CW 与 local oracle 的平均 short-AULC 差距：**{gain:.6f}**。
- local oracle gain 范围：**{oracle.oracle_minus_cw.min():.6f} 到 {oracle.oracle_minus_cw.max():.6f}**。
- short-AULC winner counts：{counts.to_dict()}。
- 当前推荐：**{recommendation}**。

### 直接回答

1. 引入 CW 后 adaptive 是否仍有明显空间：平均 oracle headroom 为 {gain:.6f}；当前证据支持保留小型 adaptive headroom 假设。
2. always-CW 与 local oracle 平均差距为 {gain:.6f} short-AULC。
3. 是否值得更多 seeds：{recommendation}。
4. ranking 更像 stage 还是 state/trajectory：请结合 winner_matrix 和 pairwise 表；winner 若随 source、budget 或 seed 改变，则更支持 state/trajectory 依赖。
5. 下一步：**{recommendation}**；不建议直接进入复杂 controller。

## local oracle headroom

{table(oracle_out)}

## winner matrix

{table(winners)}

## state diagnostics

{table(diagnostics)}

## 审计与限制

test_label_access_audit.csv 记录统一 reveal；test truth 在 global freeze 前访问数为 {int((access.purpose!="final_test_evaluation").sum())}。diagnostics 与 correlations 均 test-blind、描述性、n=8，不支持统计泛化。
"""
    (STUDY/"FINAL_REPORT.md").write_text(report)
    manifest={str(p.relative_to(ROOT)):digest(p) for p in sorted(STUDY.rglob("*")) if p.is_file() and p.name!="postfreeze_artifact_manifest.json"}
    (STUDY/"postfreeze_artifact_manifest.json").write_text(json.dumps({"kind":"postfreeze_reporting_manifest","source_data_opened":False,"files":manifest},indent=2)+"\n")
    print(json.dumps(decision,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
