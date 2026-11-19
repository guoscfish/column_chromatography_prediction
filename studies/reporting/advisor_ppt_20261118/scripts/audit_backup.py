"""Aggregate published selected-batch diagnostics and record the archived gate receipts."""
import argparse
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_performance import DATA, REPORT, GRID, MUTED, RED, canvas, configure, save

REPO = REPORT.parents[2]
ACTIVE = Path("studies/active_learning")
HISTORY = ACTIVE / "comprehensive_history_diagnostics_20261112"
BRANCH = ACTIVE / "qgeognn_v2_same_state_branching"
EPIG = ACTIVE / "epig_prediction_oriented_20261117"
BATCH_COVERAGE = HISTORY / "tables/model_batch_metrics.csv"
SEED_COVERAGE = HISTORY / "tables/model_seed_phase_errors.csv"
PHASES = ["333-525", "525-813", "813-1005", "1005-1485"]
SEEDS = "157/887/2357/6101/12203"
COLUMNS = ["figure", "metric", "method", "value", "unit", "n", "seeds", "budget", "level",
           "split", "version", "frozen", "comparability", "limitation", "source"]


def load_json(path):
    return json.loads((REPO / path).read_text())


def records():
    batch = pd.read_csv(REPO / BATCH_COVERAGE, usecols=[
        "method", "seed", "round", "phase", "selected_n", *[
            f"selected_{endpoint}_{suffix}" for endpoint in ("v1", "v2")
            for suffix in ("coverage80_raw", "sorting_coverage80", "signed_width_mean_ml")]])
    batch = batch[batch.method == "new_free_cw_sol61_xhigh"]
    assert len(batch) == 180 and batch.selected_n.sum() == 5760
    assert batch.groupby("seed").size().eq(36).all()
    seed_phase = pd.read_csv(REPO / SEED_COVERAGE)
    plot_rows, metrics = [], []
    for endpoint in ("V1", "V2"):
        prefix = f"selected_{endpoint.lower()}_"
        for phase in PHASES:
            selected = batch[batch.phase == phase]
            by_seed = selected.groupby("seed")[prefix + "coverage80_raw"].mean()
            check = seed_phase[(seed_phase.target == endpoint) & (seed_phase.phase == phase)]
            assert np.allclose(by_seed, check.set_index("seed").loc[by_seed.index, "coverage80_raw"])
            plot_rows.append({
                "phase": phase, "target": endpoint, "coverage": by_seed.mean(),
                "seed_min": by_seed.min(), "seed_max": by_seed.max(),
                "events": int(selected.selected_n.sum()),
            })
        for suffix, unit in (("coverage80_raw", "proportion"), ("sorting_coverage80", "proportion"),
                             ("signed_width_mean_ml", "mL")):
            for phase in ["all", *PHASES]:
                selected = batch if phase == "all" else batch[batch.phase == phase]
                ys = selected[prefix + suffix]
                n = selected.selected_n
                budget = "L333–1485" if phase == "all" else "L" + phase
                metrics.append(["F09", f"{endpoint} {suffix}", "New CW-Free32",
                                np.average(ys, weights=n), unit, int(n.sum()), SEEDS,
                                budget, "4g 选中行事件", "historical selected feedback",
                                "20261112 comprehensive", "既有选前分位输出及选后反馈",
                                "相同端点与已选分布；每阶段各 seed 等权",
                                "选择分布有偏；名义边际区间；非 epistemic 校准", str(BATCH_COVERAGE)])
    pd.DataFrame(plot_rows).to_csv(DATA / "f09_selected_interval_coverage.csv", index=False)
    return plot_rows, metrics


def supplementary():
    validation_path = BRANCH / "results/validation_metrics.csv"
    counts_path = BRANCH / "results/winner_counts.csv"
    decision_path = BRANCH / "decision.json"
    # Validation metadata checks the execution count. Already public winner counts are a separate summary.
    validation = pd.read_csv(REPO / validation_path, usecols=[
        "seed", "source_trajectory", "anchor_budget", "strategy", "rollout_step"])
    anchor = ["seed", "source_trajectory", "anchor_budget"]
    n_anchors = len(validation[anchor].drop_duplicates())
    n_branches = len(validation[anchor + ["strategy"]].drop_duplicates())
    decision = load_json(decision_path)
    assert n_anchors == 8 and n_branches == decision["branch_count"] == 24
    assert len(validation) == decision["actual_new_fits"] == 48
    assert validation.groupby(anchor + ["strategy"]).rollout_step.apply(set).eq({1, 2}).all()
    assert not decision["controller_trained"]
    counts = pd.read_csv(REPO / counts_path)
    winners = counts[counts.endpoint == "short_AULC"].set_index("strategy")["count"].to_dict()
    assert list(sorted(winners.values())) == [4, 4] and sum(winners.values()) == n_anchors
    rows = []
    for metric, value, units in (("anchors", n_anchors, "states"), ("branches", n_branches, "branches"),
                                  ("historical scratch fits", len(validation), "fits")):
        rows.append(["P12", metric, "Same-state branching", value, units, n_anchors, "157/6101",
                     "L429/L653 +64", "4g Row", "validation fit metadata",
                     "20260922 completed", "historical global freeze", "common within anchor",
                     "仅开发状态", f"{validation_path} | {decision_path}"])
    for strategy in ("kernel_ivr", "gradient_lcmd", "gradient_maxdet"):
        rows.append(["P12", "short-AULC winners", strategy, winners.get(strategy, 0), "states",
                     n_anchors, "157/6101", "L429/L653 +64", "4g Row",
                     "historical published post-freeze test aggregate", "20260922 completed",
                     "historical global freeze", "同 anchor 短窗；汇总核对",
                     "引自公开归档计数；本轮未复算 test，不支持固定切换/controller",
                     str(counts_path)])
    stage1 = EPIG / "qualification/decision.json"
    stage2 = EPIG / "stage2_nested_calibration/qualification/decision.json"
    gate1 = EPIG / "formal_status.json"
    gate2 = EPIG / "stage2_nested_calibration/formal_status.json"
    for decision_file, gate_file, name, count_key in (
            (stage1, gate1, "stage 1", "qualification_auxiliary_fits"),
            (stage2, gate2, "stage 2", "completed_model_fits")):
        decision = load_json(decision_file)
        gate = load_json(gate_file)
        models = decision["selected_family_by_seed"]
        assert {int(seed) for seed, family in models.items() if not family} == {887, 6101}
        assert not decision["run_allowed"]
        assert gate.get("new_fit_attempts", gate.get("new_neural_fit_attempts")) == 0
        for metric, value, unit in (
                ("auxiliary qualification fits", decision[count_key], "fits"),
                ("qualified seeds", sum(bool(family) for family in models.values()), "seeds"),
                ("formal QGeoGNN fits attempted", 0, "fits")):
            rows.append(["P12", metric, f"EPIG {name}", value, unit, 5, SEEDS,
                         "L333 OOF；正式 L333–1005 未执行", "4g Row", "training-side OOF / formal receipt",
                         "20261117 非 main 分支归档", "原门槛/第二阶段独立封存",
                         "同 seed 资格；不能跨 seed 认证", "887/6101 未合格；真实效能未知",
                         f"{decision_file} | {gate_file}"])
    return rows


def plot(rows):
    configure()
    fig = canvas("F09", "已选样本的名义 80% 区间持续欠覆盖",
                 "New CW-Free32 · 5 seeds · 180 个历史批次 · 5,760 个选中行事件",
                 "源  comprehensive_history_diagnostics / model_batch_metrics.csv  |  "
                 "选前 q10–q90 对照选后已观测反馈；覆盖率仅描述已选分布")
    grid = fig.add_gridspec(1, 2, left=.103, right=.946, bottom=.244, top=.728, wspace=.16)
    for i, endpoint in enumerate(("V1", "V2")):
        ax = fig.add_subplot(grid[0, i])
        data = pd.DataFrame([row for row in rows if row["target"] == endpoint])
        y = 100 * data.coverage
        color = ("#E07719", "#AC4E11")[i]
        x = np.arange(4)
        ax.axhline(80, color=RED, linewidth=1.4, linestyle="--")
        ax.text(3.22, 82.5, "名义 80%", color=RED, fontsize=12, ha="right")
        ax.errorbar(x, y, yerr=[y - 100 * data.seed_min, 100 * data.seed_max - y],
                    color=color, marker="o", markersize=8, linewidth=0, elinewidth=2, capsize=6)
        for xpos, yp in zip(x, y):
            ax.annotate(f"{yp:.1f}%", (xpos, yp), xytext=(11, 0), textcoords="offset points",
                        va="center", color=color, fontsize=14, weight="bold")
        ax.set_ylim(20, 96)
        ax.set_xlim(-.34, 3.5)
        ax.set_yticks([20, 40, 60, 80])
        ax.set_xticks(x, [f"L{phase}" for phase in PHASES])
        ax.tick_params(length=0, pad=9, labelsize=13)
        ax.grid(axis="y", color=GRID, zorder=-1)
        ax.set_title(endpoint, loc="left", fontsize=21, weight="bold", pad=14)
        if i == 0:
            ax.set_ylabel("实际覆盖率（%）", labelpad=12)
    fig.text(.50, .126, "点为五 seed 均值，竖线为 seed 最小值到最大值 · 四阶段事件数为 960 / 1,440 / 960 / 2,400",
             ha="center", fontsize=14, color=MUTED)
    save(fig, "F09", "selected_quantile_interval_coverage")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true", help="redraw F09 from the packaged CSV")
    args = parser.parse_args()
    if args.plot_only:
        rows = pd.read_csv(DATA / "f09_selected_interval_coverage.csv").to_dict("records")
    else:
        rows, audit = records()
        pd.DataFrame(audit + supplementary(), columns=COLUMNS).to_csv(
            DATA / "backup_metrics_fragment.csv", index=False)
    plot(rows)
