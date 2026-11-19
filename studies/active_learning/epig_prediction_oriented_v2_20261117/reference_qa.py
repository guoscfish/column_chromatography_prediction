"""Post-run independent receipts, legacy reproduction and reference figures."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR","/tmp/epig_v2_matplotlib")
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/"studies/active_learning/epig_prediction_oriented_20261117"
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run():
    base=HERE/"reference"
    report=base/"report";report.mkdir(exist_ok=True)
    chain=[];legacy=[];comparisons=[];summaries=[]
    for complete in sorted(base.glob("seed_*/complete.json")):
        directory=complete.parent;seed=int(directory.name.split("_")[1]);receipt=read(complete)
        for name,sha in receipt["files"].items():
            if digest(directory/name)!=sha:raise RuntimeError(f"hash drift: {directory/name}")
        comparisons.append(pd.read_csv(directory/"comparisons.csv"));summaries.append(pd.read_csv(directory/"summary.csv"))
        split=pd.read_csv(ROOT/f"studies/active_learning/qgeognn_v2_row_sequential_b32/splits/row_seed_{seed}.csv",dtype={"sample_id":str})
        universe=split[split.role.isin(["l0","u0"])].reset_index(drop=True)
        cand=universe[universe.role.eq("u0")]
        scores=pd.DataFrame({"sample_id":cand.sample_id.to_numpy(),"canonical_index":cand.canonical_index.to_numpy(),"posterior_position":cand.index.to_numpy()})
        for path in sorted(directory.glob("P*.json")):
            record=read(path);dist,setting=path.stem.split("_",1)
            for objective,result in record["acquisitions"].items():
                scores[f"{path.stem}_{objective}"]=result["initial_scores"]
                if objective=="epig":
                    actual=sum(t["conditional_score"] for t in result["trace"])
                    expected=result["batch_MI_own_reference"]
                    error=abs(actual-expected)
                    if error>1e-9:raise RuntimeError("batch MI chain rule failed")
                    chain.append({"seed":seed,"reference":path.stem,"chain_sum":actual,"direct_MI":expected,"absolute_error":error})
                if setting.startswith("legacy_M") and objective in {"epig","ivr"}:
                    old=OLD/f"acquisition/seed_{seed}_gp_matern32_{dist}_{setting.removeprefix('legacy_')}/batches.json"
                    if old.exists():
                        previous=read(old)[objective+"_conditional_greedy"]
                        error=float(np.max(abs(np.asarray(previous["initial_scores"])-result["initial_scores"])))
                        same=previous["selected_positions"]==result["selected_positions"]
                        if error>1e-10 or not same:raise RuntimeError("legacy reproduction failed")
                        legacy.append({"seed":seed,"reference":path.stem,"objective":objective,
                            "max_score_error":error,"same_batch_order":same})
        scores.to_csv(report/f"candidate_scores_seed_{seed}.csv.gz",index=False,compression="gzip")
    s=pd.concat(summaries,ignore_index=True);c=pd.concat(comparisons,ignore_index=True)
    s.to_csv(report/"reference_summary.csv",index=False);c.to_csv(report/"reference_comparisons.csv",index=False)
    pd.DataFrame(chain).to_csv(report/"batch_chain_rule_qa.csv",index=False)
    pd.DataFrame(legacy).to_csv(report/"legacy_reproduction_qa.csv",index=False)
    stats={"completed_seeds":sorted(s.seed.unique().tolist()),"reference_states":len(s)//3,
        "chain_rule_comparisons":len(chain),"chain_rule_max_absolute_error":max(r["absolute_error"] for r in chain),
        "legacy_reproduction_comparisons":len(legacy),"legacy_max_score_error":max(r["max_score_error"] for r in legacy),
        "legacy_batch_orders_identical":all(r["same_batch_order"] for r in legacy),"labels_read":0}
    (report/"independent_qa.json").write_text(json.dumps(stats,indent=2)+"\n")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for i,objective in enumerate(("epig","ivr")):
        for seed in sorted(s.seed.unique()):
            part=s[(s.distribution=="P1")&(s.objective==objective)&s.reference.str.startswith("coupled_")].sort_values("M")
            part=part[part.seed==seed]
            axes[0,i].plot(part.M,part.spearman_vs_full,"o-",label=str(seed))
            axes[1,i].plot(part.M,part.greedy32_overlap_vs_full,"o-")
        axes[0,i].set_title(objective.upper()+": shared reference ordering")
        axes[0,i].set_ylabel("Singleton Spearman vs full P1")
        axes[1,i].set_ylabel("Greedy B32 overlap vs full P1")
        axes[1,i].set_xlabel("Reference size M")
        for ax in axes[:,i]:ax.set_ylim(0,1.03);ax.set_xticks([256,1024,2048]);ax.grid(alpha=.2)
    axes[0,1].legend(title="Outer seed",fontsize=8)
    fig.savefig(report/"reference_convergence.png",dpi=180);plt.close(fig)
    selected=c[(c.distribution=="P1_vs_P2")&c.comparison.eq("epig")]
    settings=["legacy_M256","coupled_M256","legacy_M1024","coupled_M1024","legacy_M2048","coupled_M2048","full"]
    fig,ax=plt.subplots(figsize=(10,4.8),constrained_layout=True)
    for seed in sorted(s.seed.unique()):
        part=selected[selected.seed==seed].set_index("reference").loc[settings]
        ax.plot(range(len(settings)),part.greedy32_overlap,"o-",label=str(seed))
    ax.set_xticks(range(len(settings)),[v.replace("_","\n") for v in settings]);ax.set_ylim(0,1.03)
    ax.set_ylabel("P1/P2 EPIG greedy B32 overlap");ax.grid(alpha=.2);ax.legend(title="Outer seed")
    fig.savefig(report/"P1_P2_coupling.png",dpi=180);plt.close(fig)
    (report/"manifest.json").write_text(json.dumps({"reference_protocol_sha256":digest(HERE/"reference_protocol.json"),
        "files":{p.name:digest(p) for p in report.iterdir() if p.is_file() and p.name!="manifest.json"}},indent=2)+"\n")
    print(json.dumps(stats))

if __name__=="__main__":run()
