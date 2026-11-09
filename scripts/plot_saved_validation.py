#!/usr/bin/env python3
"""Render saved validation evidence via one config-driven entry point.

The two historical rendering bodies are consolidated here without changing
figure semantics. Inputs are saved aggregate validation evidence, never truth
or prediction tables. Rerun audits hash checkpoint bytes but never deserialize
or train a model. Run only against independent, authorized snapshots.
"""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_snapshot(folder):
    snapshot=json.loads((folder/'snapshot.json').read_text())
    with (folder/'validation_curves.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    seeds=[r['seed'] for r in snapshot['seeds']]
    methods=['new_free_cw_sol61_xhigh','historical_free_mixed_model','nocw','numerical_hybrid32','gradient_lcmd32']
    labels=['Current Free (Sol/xhigh)','Historical Free (mixed model)','NoCW','Numerical Hybrid','Gradient LCMD']
    colors=['#087f8c','#778091','#c27732','#9274b0','#43885b']
    by={(m,s):sorted([r for r in rows if r['method']==m and int(r['seed'])==s],key=lambda r:int(r['budget'])) for m in methods for s in seeds}
    common=min(int(by[m,s][-1]['budget']) for m in methods for s in seeds)
    xs=sorted({int(r['budget']) for r in by[methods[0],seeds[0]] if int(r['budget'])<=common})
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for m,label,color in zip(methods,labels,colors):
        ys=[statistics.mean(float(next(r for r in by[m,s] if int(r['budget'])==x)['validation_selection_score']) for s in seeds) for x in xs]
        axes[0].plot(xs,ys,'o-',label=label,color=color,lw=1.8,ms=3)
    axes[0].set(title=f'Fair five-seed comparison through L{common}',xlabel='Labeled training rows',ylabel='Mean normalized validation RMSE\n(lower is better)',xticks=xs)
    axes[0].legend(frameon=False,fontsize=8);axes[0].grid(alpha=.15)
    for seed in seeds:
        curve=by[methods[0],seed]
        axes[1].plot([int(r['budget']) for r in curve],[float(r['validation_selection_score']) for r in curve],'o-',label=str(seed),ms=3)
    axes[1].set(title='Current Free: completed fits by seed',xlabel='Labeled training rows',ylabel='Normalized validation RMSE (lower is better)')
    axes[1].legend(title='Seed',frameon=False,fontsize=8);axes[1].grid(alpha=.15)
    fig.text(.5,.025,'Raw nonmonotone validation curves. Accepted batches awaiting a fit are excluded. Historical Free retains mixed-model prefixes.',ha='center',fontsize=8,color='#555555');fig.tight_layout(rect=[0,.07,1,1])
    for ext in ['png','svg','pdf']:fig.savefig(folder/f'validation_comparison.{ext}',bbox_inches='tight')
    report=folder/'REPORT.md';text=report.read_text();image='![Validation curves](validation_comparison.png)'
    if image not in text:report.write_text(text+'\n'+image+'\n')
    manifest=json.loads((folder/'snapshot_manifest.json').read_text());manifest['files']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(folder.iterdir()) if f.is_file() and f.name!='snapshot_manifest.json'};(folder/'snapshot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def plot_rerun_audit(folder, repository_root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    evidence = json.loads((folder / "trajectory_audit.json").read_text())
    summary = evidence["summary"]
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    font = next((n for n in ("Arial Unicode MS", "PingFang SC", "Heiti TC") if n in fonts), "DejaVu Sans")
    plt.rcParams.update({"font.family": font, "font.size": 10, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ("#9364A7", "#087F8C")
    labels = [str(s["seed"]) for s in summary]
    ax = axes[0, 0]
    overlap = [s["r1_overlap"] for s in summary]
    bars = ax.bar(labels, overlap, color=colors[1], width=.6)
    ax.bar_label(bars, labels=[f"{n}/32" for n in overlap], padding=4)
    ax.set(ylim=(0, 37), ylabel="两次首批共同样本数", xlabel="Seed",
           title="A  从第一批就已分岔；12203 从第二批分岔")
    ax.text(.02, .91, "887、2357、12203 两次均为 Sol/xhigh", transform=ax.transAxes, color="#536174")

    ax = axes[0, 1]
    delta = [s["L813_delta"] for s in summary]
    bars = ax.bar(labels, delta, color=["#D07B53" if d > 0 else colors[1] for d in delta], width=.6)
    ax.bar_label(bars, labels=[f"{d:+.4f}" for d in delta], padding=4)
    ax.axhline(0, color="#536174", lw=.8)
    ax.set(ylim=(-.035, .19), ylabel="第二次 − 第一次 NRMSE", xlabel="Seed",
           title="B  L813 端点差：157 占总差值的 72.1%")

    pair = next(s for s in evidence["rows"] if s["seed"] == 157)["pair"]
    names = ("historical_free_mixed_model", "new_free_cw_sol61_xhigh")
    ax = axes[1, 0]
    for key, color, label, style in zip(names, colors, ("第一次", "第二次"), ("--", "-")):
        rows = pair[key]
        ax.plot([r["L"] for r in rows], [r["metric"] for r in rows], color=color,
                ls=style, marker="o", ms=3, lw=2, label=label)
    ax.axvline(525, color="#98A3B0", ls=":", lw=1)
    ax.text(532, .85, "历史运行此后改 Sol/xhigh", fontsize=9, color="#536174")
    ax.set(title="C  Seed157 原始曲线：后一个批次放大差距", xlabel="已标注样本数 L",
           ylabel="验证集 NRMSE", xticks=[333, 525, 653, 781, 813])
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for key, color, label, style in zip(names, colors, ("第一次", "第二次"), ("--", "-")):
        row = pair[key][15]
        model = repository_root / row["model"]
        audit = json.loads((model / "fit_audit.json").read_text())
        assert hashlib.sha256((model / "best.pt").read_bytes()).hexdigest() == audit["checkpoint_sha256"]
        with (model / "history.csv").open(newline="") as handle:
            history = list(csv.DictReader(handle))
        ax.plot([int(r["epoch"]) for r in history], [float(r["validation_selection_score"]) for r in history],
                color=color, ls=style, lw=1.2, alpha=.85, label=f"{label}：最佳 epoch {audit['best_epoch']}")
        ax.scatter(audit["best_epoch"], audit["best_validation_combined_normalized_rmse"], color=color, s=45, zorder=4)
    ax.set(title="D  同一早停规则，两套训练集的收敛路径不同", xlabel="训练 epoch（两次均从固定初始化重训）",
           ylabel="Seed157 · L813 验证集 NRMSE", ylim=(.35, 1.0))
    ax.legend(frameon=False, fontsize=9)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("Free LLM32 复查：差距沿着选点和重训轨迹积累", fontsize=20, weight="bold", y=.98)
    fig.text(.06, .025, "冻结到共同预算 L813；未平滑、未取历史最优。首批输入科学对象一致，实际提示串有微小差异；本图不能识别模型切换的因果效应。", fontsize=9, color="#536174")
    fig.subplots_adjust(left=.07, right=.97, bottom=.1, top=.9, hspace=.4, wspace=.25)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(folder / f"audit_evidence.{ext}", dpi=180, facecolor="white")
    plt.close(fig)



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    allowed = {"kind", "snapshot_directory", "repository_root"}
    if set(config) - allowed:
        raise ValueError("unknown plot configuration fields")
    folder = Path(config["snapshot_directory"]).resolve(strict=True)
    if config["kind"] == "validation-snapshot":
        plot_snapshot(folder)
    elif config["kind"] == "rerun-audit":
        plot_rerun_audit(folder, Path(config["repository_root"]).resolve(strict=True))
    else:
        raise ValueError("kind must be validation-snapshot or rerun-audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
