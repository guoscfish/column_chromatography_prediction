"""Write Chinese reports from this study's machine summaries and audit only.

No source dataset, label store, prediction table, test truth, or model is opened.
Partial and complete summaries use exactly the same reporting logic.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
REPRESENTATIONS = {
    "structured": "Structured", "gnn": "GNN Hidden", "raw": "Raw Gradient",
    "cw": "CW Gradient", "llm_smiles": "Qwen SMILES", "llm_conditions": "Qwen Conditions",
    "fusion": "GNN+Qwen Fusion",
}
AXES = {"sample_volume": "样品上样体积", "pe": "PE/EA", "loading_volume": "加载溶剂体积"}
MAIN_COMPARISONS = ["LLM conditions - structured", "LLM conditions - GNN", "LLM conditions - SMILES",
                    "llm_conditions: projection - frozen", "gnn: projection - frozen",
                    "fusion: projection - frozen", "Fusion frozen - GNN", "Fusion projected - GNN projected"]


def number(value):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def fmt(value, digits=4):
    value = number(value)
    if value is None:
        return "NA"
    return f"{value:.2e}" if value != 0 and abs(value) < 10**(-digits) else f"{value:.{digits}f}"


def pct(value, digits=1):
    value = number(value)
    return "NA" if value is None else f"{100*value:.{digits}f}%"


def points(value):
    value = number(value)
    return "NA" if value is None else f"{100*value:+.1f} pp"


def mean(values):
    values = [v for v in map(number, values) if v is not None]
    return statistics.mean(values) if values else None


def label(row):
    result = REPRESENTATIONS.get(row.get("representation"), str(row.get("representation")))
    kind = row.get("kernel")
    return result + (" + Projection" if kind == "projected" else " / Linear" if kind == "linear" else " / Matérn")


def table(headers, rows):
    rows = list(rows)
    if not rows:
        return "尚无已完成结果；本节不填入预测或模拟数值。\n"
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                       "| " + " | ".join("---" for _ in headers) + " |"] +
                      ["| " + " | ".join(map(cell, row)) + " |" for row in rows]) + "\n"


class StudyReport:
    def __init__(self, study_dir=HERE, summary_dir=None):
        self.study = Path(study_dir).resolve()
        self.directory = Path(summary_dir).resolve() if summary_dir else self.study / "summary"
        if self.directory != self.study and self.study not in self.directory.parents:
            raise ValueError("Summary input must remain inside the selected study directory")
        self.summary_path = self.directory / "summary.json"
        self.summary = self._json(self.summary_path)
        if self.summary.get("schema") != "representation-gp-summary-v1":
            raise ValueError("Unsupported summary schema")
        if self.summary.get("test_truth_reads", 0):
            raise ValueError("Summary declares forbidden test-truth access")
        self.config = self._json(self.study / "config.json")
        self.support = self._json(self.study / "pair_support.json", optional=True)
        self.paired_state = self._csv("paired_state_differences.csv")
        self.paired_seed = self._csv("paired_seed_differences.csv")
        self.seed_metrics = self._csv("seed_metrics.csv")
        self.state_metrics = self._csv("state_metrics.csv")
        self.digest = hashlib.sha256(self.summary_path.read_bytes()).hexdigest()
        self.relative_summary = self.summary_path.relative_to(self.study).as_posix()

    def _safe(self, path):
        path = path.resolve()
        if self.study not in path.parents:
            raise ValueError("Report inputs cannot escape the study directory")
        return path

    def _json(self, path, optional=False):
        path = self._safe(path)
        if optional and not path.exists():
            return {}
        return json.loads(path.read_text())

    def _csv(self, filename):
        path = self._safe(self.directory / filename)
        if not path.exists():
            return []
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))

    def common(self, title):
        coverage = self.summary["coverage"]
        complete = coverage.get("complete_matrix", False)
        status = "完整预定开发矩阵" if complete else "部分完成的开发矩阵（partial）"
        lines = [f"# {title}\n", f"日期：2026-11-04。**当前为{status}**：已完成 "
                 f"{coverage['completed_states']}/{coverage.get('expected_states') or 'NA'} 个seed×预算状态，"
                 f"{coverage.get('endpoint_fits', 0)} 次端点GP拟合。只汇总有complete回执的真实结果，"
                 "不计未完成目录，不把预检重复计算为独立状态。\n",
                 "本地语言对照是 **Qwen1.5-0.5B-Chat的冻结decoder hidden state**，不是T5或语义embedding专用encoder。"
                 "T5缺本地权重，未下载；本次只研究Frozen/Projection思路，没有执行GOLLuM主变体PLLMφ的语言模型微调。\n",
                 "评价来自既有416行validation，它参与过QGeoGNN checkpoint selection，因此全部为开发证据。"
                 "五seed共享数据来源、四预算嵌套，20状态不能称20次统计独立重复。"
                 "主指标为两个端点RMSE/L0标准差的均值；Projection先平均17/29/43三初始化，再平均端点，"
                 "总体先在seed内平均已完成预算，再等权平均seed。±为seed样本SD，非置信区间。"
                 "缺项时不同seed的预算组成可能不同，必须结合下表解释；配对比较只用共同状态。\n"]
        states = defaultdict(list)
        for row in coverage.get("states", []):
            states[int(row["seed"])].append(int(row["budget"]))
        lines.append(table(["开发seed", "已完成预算", "计划预算"],
                           [[seed, ", ".join(map(str, sorted(states.get(seed, [])))) or "无",
                             ", ".join(map(str, self.config.get("budgets", [])))]
                            for seed in self.config.get("seeds", sorted(states))]))
        if not complete:
            missing = coverage.get("missing_states", [])
            text = "; ".join(f"{r['seed']}:L{r['budget']}" for r in missing)
            lines.append(f"未完成状态：{text or '见source_coverage.json'}。不得将本版结果写成五seed×四预算完整确认。\n")
        lines.append(f"数值来源：[机器汇总]({self.relative_summary})，SHA256 `{self.digest}`。"
                     "方法与权限见[PROTOCOL.md](PROTOCOL.md)、[METHOD.md](METHOD.md)、[AUDIT.md](AUDIT.md)。\n")
        return lines

    def pairs(self, selected=MAIN_COMPARISONS):
        lookup = {r["comparison"]: r for r in self.summary.get("paired_comparisons", [])}
        return [lookup[name] for name in selected if name in lookup]

    def pair_table(self, selected=MAIN_COMPARISONS):
        return table(["候选−对照", "配对seed/状态", "Δ主误差", "主误差下降seed", "ΔNLPD", "Δ90%覆盖"],
            [[r["comparison"], f"{r['seed_count']}/{int(r.get('paired_seed_budget_count', 0))}",
              fmt(r.get("normalized_rmse_L0_delta_mean")),
              f"{round(float(r.get('primary_seed_win_fraction', 0))*r['seed_count'])}/{r['seed_count']}",
              fmt(r.get("nlpd_delta_mean")), points(r.get("coverage_90_delta_mean"))]
             for r in self.pairs(selected)])

    def comparison_statement(self, name):
        rows = self.pairs([name])
        if not rows:
            return f"{name}：尚无完整配对结果，不能作效果判断。\n"
        row = rows[0]
        delta = number(row.get("normalized_rmse_L0_delta_mean"))
        direction = "下降" if delta is not None and delta < 0 else "上升" if delta is not None and delta > 0 else "未变或不可判定"
        n = int(row["seed_count"])
        wins = round(float(row.get("primary_seed_win_fraction", 0))*n)
        result = (f"{name}：已完成共同状态的seed均值主误差{direction}，Δ={fmt(delta)}，"
                  f"{wins}/{n} seed方向为下降；NLPD差={fmt(row.get('nlpd_delta_mean'))}。")
        pd = number(row.get("nlpd_delta_mean"))
        if delta is not None and pd is not None and delta < 0 <= pd:
            result += " **均值误差改善与概率评分恶化同时出现，不能称为更可靠的GP。**"
        elif delta is not None and pd is not None and delta >= 0:
            result += " 当前没有主指标平均收益；不能用个别预算或初始化替代该结论。"
            if pd < 0:
                result += " NLPD有所改善，需结合区间宽度判断是否主要来自更保守的方差；不把它等同均值优势。"
        else:
            result += " 概率结论还需共同检查逐端点覆盖、宽度和初始化/优化稳定性。"
        return result + "\n"

    def budget_primary(self):
        budgets = self.config.get("budgets", [])
        rows = self.summary.get("budget_metrics", [])
        variants = list(dict.fromkeys((r["representation"], r["kernel"]) for r in rows))
        lookup = {(r["representation"], r["kernel"], r["budget"]): r for r in rows}
        output = []
        for rep, kernel in variants:
            cells = [label({"representation": rep, "kernel": kernel})]
            for budget in budgets:
                r = lookup.get((rep, kernel, budget))
                cells.append("NA" if r is None else f"{fmt(r.get('normalized_rmse_L0_mean'))} ± "
                             f"{fmt(r.get('normalized_rmse_L0_seed_sd'))} (n={r['seed_count']})")
            output.append(cells)
        return table(["表示/核"] + [f"L{b}主误差" for b in budgets], output)

    def seed_pair_table(self):
        rows = [r for r in self.paired_seed if r.get("comparison") in MAIN_COMPARISONS]
        return table(["候选−对照", "seed", "共同预算数", "Δ主误差", "ΔNLPD", "Δ90%覆盖"],
                     [[r["comparison"], r["seed"], r.get("paired_budget_count", "NA"),
                       fmt(r.get("normalized_rmse_L0_delta")), fmt(r.get("nlpd_delta")),
                       points(r.get("coverage_90_delta"))] for r in rows])

    def budget_pair_table(self):
        groups = defaultdict(list)
        for row in self.paired_state:
            if row.get("comparison") in MAIN_COMPARISONS:
                groups[(row["comparison"], int(row["budget"]))].append(row)
        rows = []
        for (comparison, budget), group in sorted(groups.items()):
            values = [number(r.get("normalized_rmse_L0_delta")) for r in group]
            valid = [v for v in values if v is not None]
            rows.append([comparison, budget, fmt(mean(valid)), f"{sum(v < -1e-12 for v in valid)}/{len(valid)}",
                         fmt(mean(r.get("nlpd_delta") for r in group))])
        return table(["候选−对照", "预算", "Δ主误差均值", "下降seed/配对seed", "ΔNLPD均值"], rows)

    def representation_report(self):
        out = self.common("实验表示：响应预测、科学距离与文本鲁棒性")
        out += ["## 响应预测\n", self.budget_primary(),
                "按同seed同预算比较，而不是用全体最低RMSE命名胜者。表中Linear是Raw/CW核敏感性控制，"
                "它复用Phase0三起点，与Matérn单起点优化预算不同；两者差异不能完全归因核族。\n",
                self.pair_table(["LLM conditions - structured", "LLM conditions - GNN", "LLM conditions - SMILES", "GNN - structured"])]
        for name in ["LLM conditions - structured", "LLM conditions - GNN", "LLM conditions - SMILES"]:
            out.append(self.comparison_statement(name))
        out += ["Qwen SMILES-only使同分子不同实验条件具有相同向量，无法单独识别条件响应。"
                "Conditions改善或恶化必须相对这个对照和Structured/GNN一起解释；它不能仅由文本包含数字推定已学会数值含义。"
                "GNN Hidden保留任务监督信息，Raw/CW刻画参数敏感性，三者不是同一种表示。\n",
                "## 严格单条件观测对\n"]
        support = self.support.get("per_seed_axis", [])
        out.append(table(["seed", "变化轴", "不同观测对", "涉及分子"],
                         [[r["seed"], AXES.get(r["axis"], r["axis"]), r["strict_pair_count"],
                           r["participating_molecule_count"]] for r in support]))
        out.append("条件对全部来自validation，当前GP未用其响应拟合，但该validation曾用于QGeoGNN选checkpoint。"
                   "同一分子、其他实际记录条件完全一致，只有目标轴变化；不同预算复用同一validation，"
                   "对数不能乘四。加载溶剂体积只有极少数分子支持，零对seed为不可评，不能把缺失当零误差。\n")
        contrast = self.summary.get("contrast", [])
        out.append(table(["表示/核", "变化轴", "距离/训练RMS", "|真实Δ| mL", "GP差分误差 mL", "GNN差分误差 mL", "距离−|真实Δ|ρ", "seed数"],
            [[label(r), AXES.get(r["axis"], r["axis"]), fmt(r.get("mean_embedding_distance_mean")),
              fmt(r.get("mean_absolute_response_delta_ml_mean")), fmt(r.get("gp_contrast_absolute_error_ml_mean")),
              fmt(r.get("gnn_contrast_absolute_error_ml_mean")), fmt(r.get("distance_response_rank_spearman_mean")),
              r.get("seed_count", "NA")] for r in contrast]))
        out.append("差分误差为`|(y_B−y_A)−(pred_B−pred_A)|`。距离按训练侧centered向量RMS归一；"
                   "Projection行使用真实学习后空间。小距离若伴随很小真实差分，并不足以判表示失败；"
                   "小距离同时出现较大真实差分与较大差分误差，才支持未表达该响应方向的诊断。"
                   "表格同时给出这三项，未事后设一个有利的小距离阈值。不同轴的实际变化幅度/单位不同，"
                   "距离比不是物理导数，相关也不证明因果；row pairs与seed可能共享分子，ρ不带独立pair显著性。\n")
        for axis in ("sample_volume", "pe", "loading_volume"):
            chosen = [r for r in contrast if r.get("axis") == axis and r.get("kernel") == "matern52"
                      and r.get("representation") in {"raw", "cw", "gnn", "llm_conditions"}]
            if not chosen:
                continue
            statements = []
            for row in chosen:
                error = number(row.get("gp_contrast_absolute_error_ml_mean"))
                delta = number(row.get("mean_absolute_response_delta_ml_mean"))
                ratio = error / delta if error is not None and delta is not None and delta > 0 else None
                statements.append(f"{REPRESENTATIONS[row['representation']]}距离{fmt(row.get('mean_embedding_distance_mean'))}、"
                                  f"平均|真实Δ|={fmt(delta)} mL、差分误差={fmt(error)} mL"
                                  f"（两均值之比{fmt(ratio, 2)}）")
            out.append(f"{AXES[axis]}：" + "；".join(statements) + "。这个比值只把差分误差与该轴响应变化放在一起，"
                       "不是预注册的通过阈值；更大的embedding距离也可能伴随更差的真实差分预测。\n")
        out += ["## 等义模板扰动\n", "固定A模板拟合的Qwen Conditions GP直接输入B模板；没有重训或由B评分调参。"
                "两模板的原始数字、单位、分子及实验含义相同，变化只是措辞。以下是256个冻结合法U候选的平均描述；"
                "不读取候选响应。当前只检查固定Qwen Conditions GP，未完成Projection/Fusion模板稳健性验证。\n"]
        for row in self.summary.get("text_robustness", []):
            out.append(table(["seed数", "模板距离/训练RMS", "最近10邻居重叠", "V1均值变化 mL", "V2均值变化 mL", "V1方差ρ", "V2方差ρ", "V1 MaxVar32重叠", "V2 MaxVar32重叠"], [[
                row.get("seed_count"), fmt(row.get("template_distance_train_rms_mean_mean")),
                pct(row.get("nearest_10_overlap_mean_mean")), fmt(row.get("V1_mean_absolute_change_ml_mean")),
                fmt(row.get("V2_mean_absolute_change_ml_mean")), fmt(row.get("V1_variance_rank_spearman_mean")),
                fmt(row.get("V2_variance_rank_spearman_mean")), pct(row.get("V1_maxvar_top32_overlap_mean")),
                pct(row.get("V2_maxvar_top32_overlap_mean"))]]))
            out.append(table(["端点", "MaxVar采集分数绝对变化均值 mL²"],
                             [["V1", fmt(row.get("V1_maxvar_absolute_change_mean_mean"))],
                              ["V2", fmt(row.get("V2_maxvar_absolute_change_mean_mean"))]]))
        if not self.summary.get("text_robustness"):
            out.append("尚无已完成模板结果，不对稳定性作判断。\n")
        out.append("邻居或方差排序变化说明文本形式进入了数值决策链；即使输入语义等价，也不能默认采集稳定。"
                   "MaxVar重叠只是一种采集信号检查，未证明IVR/BatchIG或真实AL的模板鲁棒性。\n")
        out += self.figures(["prediction_and_calibration.png"])
        out.append("逐行已授权开发差分见summary/contrasts.csv；预算、端点及初始化明细见summary的metrics/state_metrics表。"
                   "本报告不构成真实新实验或主动学习效果证据。\n")
        return "\n".join(out)

    def calibration_report(self):
        out = self.common("GP概率预测与不确定性校准")
        out += ["## 均值与概率分开判定\n", self.pair_table(),
                "NLPD越小越好，90%/95%覆盖应与区间宽度共同检查；接近名义覆盖但区间极宽不代表有用。"
                "全部NLPD及宽度使用mL响应尺度，观测预测方差为latent variance+训练噪声；没有用latent区间冒充响应预测区间。"
                "本次未在validation上调方差或进行事后覆盖校准。\n"]
        for name in ["llm_conditions: projection - frozen", "fusion: projection - frozen", "Fusion projected - GNN projected"]:
            out.append(self.comparison_statement(name))
        rows = self.summary.get("endpoint_budget_metrics", [])
        low90 = sum(number(r.get("coverage_90_mean")) is not None and float(r["coverage_90_mean"]) < .9 for r in rows)
        total90 = sum(number(r.get("coverage_90_mean")) is not None for r in rows)
        cover = [number(r.get("coverage_90_mean")) for r in rows]
        cover = [v for v in cover if v is not None]
        if cover:
            out.append(f"当前预算×端点×表示/核的seed均值覆盖率范围为{pct(min(cover))}–{pct(max(cover))}，"
                       f"{low90}/{total90}条汇总行低于名义90%。这是描述性组均值计数，不是独立校准检验或新设通过门槛。\n")
        out += ["## 按预算和端点评分\n", table(["预算", "表示/核", "目标", "seed数", "RMSE mL", "NLPD", "90%覆盖", "95%覆盖", "90%宽 mL", "95%宽 mL", "标准化误差RMS"],
            [[r["budget"], label(r), r["endpoint"], r["seed_count"], fmt(r.get("rmse_mean")),
              fmt(r.get("nlpd_mean")), pct(r.get("coverage_90_mean")), pct(r.get("coverage_95_mean")),
              fmt(r.get("mean_interval_width_90_mean")), fmt(r.get("mean_interval_width_95_mean")),
              fmt(r.get("standardized_error_rms_mean"))] for r in rows])]
        out += ["## 优化、噪声及数值状态\n", table(["表示/核", "端点fits", "未收敛", "超参数边界", "非零jitter", "noise方差中位 mL²", "noise/signal中位", "lengthscale中位", "训练秒中位"],
            [[label(r), r["endpoint_fit_count"], r["nonconverged_fit_count"], r["boundary_hit_count"],
              r.get("nonzero_jitter_fit_count", "NA"), fmt(r.get("noise_variance_median")),
              fmt(r.get("noise_signal_ratio_median")), fmt(r.get("lengthscale_median")),
              fmt(r.get("training_seconds_median"), 3)] for r in self.summary.get("optimizer", [])])]
        nonconverged = sum(r.get("nonconverged_fit_count", 0) for r in self.summary.get("optimizer", []))
        fits = sum(r.get("endpoint_fit_count", 0) for r in self.summary.get("optimizer", []))
        out.append(f"{nonconverged}/{fits}次拟合未达到优化器收敛条件；按协议保留有限输出和失败标记，"
                   "没有按validation表现重启或延长。40步上限下不能声称边际似然已最优。"
                   "此处fit数用于计算诊断，不是统计独立重复；线性控制另用原Phase0三起点。\n")
        out.append("MLL联合学习可以让Projection压缩训练响应相关方向，也可能使训练noise趋小、外样本误差远超预测标准差。"
                   "较低训练NLL只能说明训练Gaussian模型拟合，不证明外样本区间可信。各预算训练样本数不同，"
                   "不直接横比未除样本数的训练NLL；完整loss、超参数、variance、jitter见results/fit与summary/fits.csv。"
                   "同方差noise吸收表示失配、超参数plug-in及未收敛均可能贡献过度自信，本次不能凭相关诊断把原因唯一归给某一项。\n")
        out.append("与Phase0的90%区间61.5%–74.5%欠覆盖相比，本次改变核、状态来源及评价分布，"
                   "不能把两报告的差异直接解释为已定位原因。真正核敏感性比较应看同状态Raw/CW的Matérn−Linear；"
                   "Projection比较还涉及容量和有限步优化。\n")
        out += self.residual_section()
        out += self.figures(["calibration_tradeoffs.png", "prediction_and_calibration.png"])
        out.append("没有独立训练/校准/最终核验分割，不授予部署概率校准资格。正式AL的适用checkpoint、候选分布、"
                   "噪声机制与状态转移仍需独立补证，不能由当前NLPD或均值误差单项放行。\n")
        return "\n".join(out)

    def residual_section(self):
        """Only combine residual diagnostics with an identical completed-state snapshot."""
        out = ["## 标准化残差与预测不确定性的关系\n"]
        path = self.directory / "residual_diagnostics.json"
        if not path.exists():
            return out + ["当前汇总目录没有residual_diagnostics.json，本节未填入残差分层结果。\n"]
        residual = self._json(path)
        if residual.get("schema") != "representation-gp-residual-diagnostics-v1":
            return out + ["残差诊断schema不可识别，本节跳过，不混入未核验结果。\n"]
        def identity(coverage):
            states = frozenset((int(r["seed"]), int(r["budget"])) for r in coverage.get("states", []))
            return (states, coverage.get("completed_states"), coverage.get("expected_states"),
                    coverage.get("endpoint_fits"), coverage.get("complete_matrix"))
        if identity(residual.get("coverage", {})) != identity(self.summary["coverage"]):
            return out + ["**残差诊断与主summary的已完成状态/拟合数覆盖不一致，本节明确跳过。** "
                          "须先按相同completed回执刷新两份汇总，不将不同快照的σ、覆盖或图混为当前结果。\n"]
        if residual.get("bin_count") != 5:
            return out + ["残差诊断未使用固定五分组，本节跳过以保留预定描述口径。\n"]
        relative = path.relative_to(self.study).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        out.append(f"[真实残差诊断]({relative})与本报告覆盖完全一致，SHA256 `{digest}`。"
                   "逐端点fit按观测预测方差升序、sample ID确定并列顺序后等数分五组；bin1是同一fit内最自信的20%行，"
                   "bin5为最不自信组。分组不使用真实响应，不重拟合、不调方差、不用本节决定方法保留。\n")
        out.append("标准化残差 `z=(y−μ)/sqrt(v_observation)`；若Gaussian预测无偏且尺度可靠，z的RMS应接近1，"
                   "90%/95%组内覆盖也应接近相应名义值。高方差与高误差相关只能支持相对排序，不能证明绝对尺度校准；"
                   "z偏置也会抬高RMS，故低覆盖不能仅凭相关性唯一归因噪声。σ/L0使不同响应量纲可比较。"
                   "先计算每组RMS，再依初始化→端点→seed内预算→seed平均；表格不是对全部行直接重算的单一RMS。\n")
        groups = defaultdict(list)
        for row in residual.get("bin_summary", []):
            groups[(row["representation"], row["kernel"])].append(row)
        def span(group, field, percentage=False):
            values = [number(r.get(field)) for r in group]
            values = [v for v in values if v is not None]
            if not values:
                return "NA"
            formatter = pct if percentage else fmt
            return f"{formatter(min(values))}–{formatter(max(values))}"
        rows = []
        for (rep, kernel), group in sorted(groups.items()):
            first = next((r for r in group if int(r["variance_rank_bin"]) == 1), {})
            rows.append([label({"representation": rep, "kernel": kernel}), len(group),
                         span(group, "mean_sigma_L0_mean"), span(group, "z_rms_mean"),
                         span(group, "coverage_90_mean", True), span(group, "coverage_95_mean", True),
                         fmt(first.get("z_rms_mean")), pct(first.get("coverage_90_mean")),
                         first.get("seed_count", "NA")])
        out.append(table(["表示/核", "实际bin数", "五bin σ/L0范围", "五bin zRMS范围", "五bin 90%覆盖范围", "五bin 95%覆盖范围", "最自信bin zRMS", "最自信bin 90%覆盖", "seed数"], rows))
        for (rep, kernel), group in sorted(groups.items()):
            if kernel != "projected":
                continue
            first = next((r for r in group if int(r["variance_rank_bin"]) == 1), {})
            z = number(first.get("z_rms_mean"))
            coverage = number(first.get("coverage_90_mean"))
            if z is not None and coverage is not None and z > 1 and coverage < .9:
                out.append(f"{label({'representation': rep, 'kernel': kernel})}：最自信bin仍有zRMS={fmt(z)}、"
                           f"90%覆盖={pct(coverage)}，与过度自信相容；不能把低预测σ当成已验证的低错误风险。"
                           "这属于相同开发快照的分层描述，未建立独立确认或唯一致因。\n")
        links = []
        for name in ("residual_endpoint_bins.csv", "residual_state_bins.csv", "residual_seed_bins.csv",
                     "residual_bin_summary.csv", "residual_endpoint_summary.csv"):
            target = self._safe(self.directory / name)
            if target.exists():
                links.append(f"[{name}]({target.relative_to(self.study).as_posix()})")
        if links:
            out.append("可复算分层明细：" + "、".join(links) + "。逐V1/V2结果避免端点平均掩盖差异。\n")
        name = "residual_uncertainty_relationship.png"
        target = self._safe(self.directory / name)
        if name in residual.get("figures", []) and target.exists():
            out.append(f"![真实开发分层残差与不确定性]({target.relative_to(self.study).as_posix()})\n")
        return out

    def comparison_report(self):
        out = self.common("GNN、语言表示、Projection与Fusion配对比较")
        out += ["## 语言表示是否提供额外预测信息\n", self.pair_table(),
                self.comparison_statement("LLM conditions - structured"),
                self.comparison_statement("LLM conditions - GNN"),
                "Structured是必要的非语言化学与条件对照；若它已优于本地Qwen，应保留该负结果。"
                "Qwen自身无优势不能外推为T5或所有语言模型均无价值；同样，局部有优势也不证明语言先验具有可迁移新增信息。\n",
                "## MLL训练Projection是否带来可重复的未见行收益\n"]
        for name in ["llm_conditions: projection - frozen", "gnn: projection - frozen", "fusion: projection - frozen"]:
            out.append(self.comparison_statement(name))
        out.append(table(["Projection表示", "seed数", "初始化间主误差SD均值", "初始化间主误差range均值", "初始化间NLPD SD均值"],
            [[REPRESENTATIONS.get(r["representation"], r["representation"]), r.get("seed_count"),
              fmt(r.get("normalized_rmse_L0_initialization_sd_mean")),
              fmt(r.get("normalized_rmse_L0_initialization_range_mean")),
              fmt(r.get("nlpd_initialization_sd_mean"))] for r in self.summary.get("projection_initialization", [])]))
        out.append("三初始化全部报告并先平均；未由validation选最优初始化。初始化SD与跨seed方向共同反映可重复性，"
                   "一个seed/一个初始化的最低RMSE不能替代整体配对。只有开发均值收益时，仍不能声称新独立数据上的复现，"
                   "更不能将MLL下降包装为概率可靠性。\n")
        out += ["## Fusion的收益是否超出容量\n", self.comparison_statement("Fusion frozen - GNN"),
                self.comparison_statement("Fusion projected - GNN projected"),
                "容量匹配对照为GNN原隐藏128→64与Fusion128→64，均8,256个投影参数；Fusion先用L_t covariates"
                "将GNN与Qwen分别标准化/PCA64并作block RMS，再拼128维。Qwen直接1024→64有65,600参数，"
                "不能与GNN直接投影称为容量相同。即使Fusion胜过参数匹配对照，PCA、block scaling与输入组成也变化，"
                "不能把全部增益唯一归于语言信息；需结合Frozen Fusion及相同PCA预处理的GNN对照进一步辨别。\n",
                "## 按预算与seed看方向\n", self.budget_primary(), self.budget_pair_table(), self.seed_pair_table(),
                "全局平均可能掩盖预算反向或少数seed主导。以上按预算配对、seed内预算均值及三初始化平均都固定展示；"
                "若partial缺不同预算，不能拿不平衡总体均值制造一致性。当前五seed为既有开发seed，不作独立外部确认p值。\n",
                "## 概率质量与离线采集是否跟随均值\n", self.acquisition_table(),
                "较好均值并不意味着覆盖、NLPD或batch稳定性更好。采集排序使用同256合法U的latent联合协方差；"
                "IVR/BatchIG逐步条件化，初始单点rank与最终batch重叠分别报告。信息量在各自GP内定义，"
                "不同模型的数值大小不能当真实实验学习收益的统一尺度。\n"]
        out += self.figures(["paired_projection_effects.png", "offline_acquisition_overlap.png"])
        out.append("本研究未启动AL闭环，无法宣称AULC、终点QGeoGNN误差或实验学习稳定性改善。"
                   "后续Go/No-Go及最小补证见[AL_FEASIBILITY.md](AL_FEASIBILITY.md)与[NEXT_STAGE_DECISION.md](NEXT_STAGE_DECISION.md)。\n")
        return "\n".join(out)

    def acquisition_table(self):
        rows = self.summary.get("acquisition", {}).get("paired_rank_summary", [])
        keep = {frozenset(("gnn", "llm_conditions")), frozenset(("gnn", "fusion")),
                frozenset(("structured", "llm_conditions"))}
        rows = [r for r in rows if frozenset((r.get("representation_a"), r.get("representation_b"))) in keep
                and r.get("kernel_a") == r.get("kernel_b") == "matern52"]
        return table(["表示对", "采集", "seed数", "初始分数ρ", "初始Top32重叠", "最终batch重叠"],
            [[r["comparison"], r["acquisition"], r["seed_count"], fmt(r.get("initial_score_rank_spearman_mean")),
              pct(r.get("initial_score_top32_overlap_mean")), pct(r.get("selected_batch_overlap_mean"))] for r in rows])

    def figures(self, names):
        output = []
        present = set(self.summary.get("figures", []))
        for name in names:
            if name in present:
                path = self._safe(self.directory / name)
                if path.exists():
                    output.append(f"![真实开发结果：{name}]({path.relative_to(self.study).as_posix()})\n")
        return output

    def write(self):
        reports = {"REPRESENTATION_REPORT.md": self.representation_report(),
                   "GP_CALIBRATION_REPORT.md": self.calibration_report(),
                   "COMPARISON_REPORT.md": self.comparison_report()}
        for filename, text in reports.items():
            target = self.study / filename
            temporary = target.with_suffix(".md.tmp")
            temporary.write_text(text)
            temporary.replace(target)
        return list(reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=HERE)
    parser.add_argument("--summary-dir", type=Path)
    parser.add_argument("--check", action="store_true", help="Render in memory without writing reports")
    args = parser.parse_args()
    report = StudyReport(args.study_dir, args.summary_dir)
    if args.check:
        texts = [report.representation_report(), report.calibration_report(), report.comparison_report()]
        print(json.dumps({"check": "passed", "reports": len(texts), "characters": list(map(len, texts)),
                          "partial_results": report.summary.get("partial_results")}, ensure_ascii=False))
    else:
        print(json.dumps({"reports": report.write(), "partial_results": report.summary.get("partial_results"),
                          "summary_sha256": report.digest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
