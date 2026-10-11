#!/usr/bin/env python3
"""Publish completed validation fits, or redraw a retained snapshot without runtime.

Reads fit audits and prediction-freeze receipts only. Never fits models, calls an
LLM, opens test truth or changes a study's sealed files.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "studies/active_learning"
SCIENTIST = BASE / "qgeognn_v2_row_llm_scientist_v2/revisions/astra_high_full_pool_20261001"
DUAL_REVISION = "sol61_xhigh_from429_evidence_repair_20261011"
SPEC = {
    "Random32": [("qgeognn_v2_row_sequential_b32", "random")],
    "LCMD32": [("qgeognn_v2_row_sequential_b32", "lcmd")],
    "Hybrid traditional": [("qgeognn_v2_row_sequential_b32", "hybrid")],
    "Gradient MaxDet": [("qgeognn_v2_row_maxdet_b32", "gradient_maxdet")],
    "U50 Gradient MaxDet": [("qgeognn_v2_row_maxdet_b32", "u50_gradient_maxdet")],
    "Fused MaxDet": [("qgeognn_v2_row_fused_maxdet_b32", "gradient_latent_fusion_maxdet")],
    "Fused MaxDet a=0.80": [("qgeognn_v2_row_fused_maxdet_a080_b32_screen", "gradient_latent_fusion_a080_maxdet")],
    "CW32 (LCMD)": [(name, "center_width_lcmd") for name in (
        "qgeognn_v2_row_short_sequential_b32", "qgeognn_v2_row_cw_lcmd_to_525",
        "qgeognn_v2_row_cw_lcmd_to_653", "qgeognn_v2_row_cw_lcmd_to_1005")],
    "Whitening LCMD": [("qgeognn_v2_row_cw_selector_factorial", "whitening_lcmd")],
    "CW-IVR": [(name, "cw_ivr") for name in (
        "qgeognn_v2_row_cw_selector_factorial", "qgeognn_v2_row_cw_selector_extension_to_653",
        "qgeognn_v2_row_cw_selector_extension_to_1005")],
    "CW-MaxDet": [(name, "cw_maxdet") for name in (
        "qgeognn_v2_row_cw_selector_factorial", "qgeognn_v2_row_cw_selector_extension_to_653",
        "qgeognn_v2_row_cw_selector_extension_to_1005")],
    "CW-IVR portfolio": [("qgeognn_v2_row_cw_ivr_portfolio_b32", "cw_ivr_portfolio")],
    "Direction LCMD": [("qgeognn_v2_row_short_sequential_b32", "direction_lcmd")],
    "CW16 + Random16": [("qgeognn_v2_row_llm_dialog_feedback_v1", "cw16_random16_full_pool")],
}
CORE = ["CW16 + LLM16", "Free LLM32", "Dual expert", "CW32 (LCMD)", "Random32"]
AUDIT_FIELDS = (
    "fit_contract_hash", "initialization_seed", "initialization_hash", "train_ids_hash",
    "validation_ids_hash", "train_rows", "validation_rows", "best_epoch", "epochs_run",
    "best_validation_combined_normalized_rmse", "checkpoint_sha256", "checkpoint_state_hash",
    "prediction_sha256", "training_seconds", "test_labels_used_for_fit_or_checkpoint_selection",
    "model_variant", "parameter_count",
)


def merge_point(rows, labels, value):
    if not math.isfinite(value) or value < 0:
        raise ValueError("Invalid validation metric")
    if labels in rows and abs(rows[labels] - value) > 1e-8:
        raise ValueError(f"Conflicting completed fits at {labels} labels")
    rows[labels] = value


def collect(dual_revision):
    sources, audits = {}, {}

    def read(path):
        data = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    def fit(path, freeze=None):
        audit = read(path)
        if audit["test_labels_used_for_fit_or_checkpoint_selection"] != 0:
            raise ValueError(f"Fit used test labels: {path}")
        if freeze and audit["checkpoint_sha256"] != freeze["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint audit disagrees with frozen prediction: {path}")
        audits[str(path.relative_to(ROOT))] = {k: audit[k] for k in AUDIT_FIELDS if k in audit}
        return audit["train_rows"], audit["best_validation_combined_normalized_rmse"]

    curves = {}
    for seed in (157, 6101):
        current = {}
        initial = fit(BASE / f"qgeognn_v2_row_sequential_b32/runtime/seed_{seed}/random/round_00/model/fit_audit.json")
        for name, studies in SPEC.items():
            rows = dict([initial])
            for study, method in studies:
                for path in sorted((BASE / study / f"runtime/seed_{seed}/{method}").glob("round_*/model/fit_audit.json")):
                    labels, value = fit(path)
                    if labels <= 1005:
                        merge_point(rows, labels, value)
            current[name] = rows
        for name, method, study in (
            ("CW16 + LLM16", "cw16_llm16_scientist_v2", SCIENTIST),
            ("Free LLM32", "free_llm32_scientist_v2", SCIENTIST),
            ("Dual expert", "free_llm32_dual_expert_v1",
             BASE / "qgeognn_v2_row_llm_dual_expert_v1/revisions" / dual_revision),
        ):
            rows = {}
            for path in sorted((study / f"runtime/seed_{seed}/{method}").glob("round_*/prediction_freeze.json")):
                freeze = read(path)
                merge_point(rows, *fit(Path(freeze["checkpoint_path"]).parent / "fit_audit.json", freeze))
            if rows:
                current[name] = rows
        curves[str(seed)] = {name: [{"labels": n, "nrmse": v} for n, v in sorted(rows.items())]
                             for name, rows in current.items()}
    status_path = BASE / "qgeognn_v2_row_llm_dual_expert_v1/revisions" / dual_revision / "execution_status.json"
    status = read(status_path) if status_path.exists() else {}
    return {"captured_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            "evaluation": "validation_only", "dual_revision": dual_revision,
            "dual_status": {k: status[k] for k in ("status", "latest_completed_labels", "target_labels") if k in status},
            "curves": curves, "source_hashes": sources, "fit_audits": audits}


def draw(snapshot, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "active-learning-snapshot"})
    for filename, names in (("llm_validation", CORE),
                            ("all_strategies_validation", list(SPEC) + CORE[:3])):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
        palette = {name: plt.get_cmap("tab20")(i / 20) for i, name in enumerate(list(SPEC) + CORE[:3])}
        palette.update({"CW16 + LLM16": "#087f8c", "Free LLM32": "#7652ad",
                        "Dual expert": "#c75259", "CW32 (LCMD)": "#de8a26", "Random32": "#8b96a2"})
        for ax, (seed, curves) in zip(axes, snapshot["curves"].items()):
            for name in names:
                if name not in curves:
                    continue
                rows = curves[name]
                ax.plot([r["labels"] for r in rows], [r["nrmse"] for r in rows],
                        label=name, color=palette[name], linewidth=2.4 if name in CORE[:3] else 1.3,
                        marker="o" if name in CORE[:3] else None, markersize=3)
            ax.set_title(f"Seed {seed}")
            ax.set_xlabel("Active training labels")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, ncol=2 if len(names) > 5 else 1)
        axes[0].set_ylabel("Validation NRMSE (lower is better)")
        fig.suptitle(f"Completed validation fits | {snapshot['captured_at'][:10]}")
        fig.text(.07, .02, "Curves stop at completed fits; no extrapolation. Exploratory row-split evidence.\n"
                 "Dual expert changes from astra/high to sol61/xhigh after 397 labels; no new test evaluation.", fontsize=9)
        fig.tight_layout(rect=[0, .09, 1, .94])
        for ext in ("png", "svg"):
            fig.savefig(output / f"{filename}.{ext}", dpi=170, metadata={"Date": None} if ext == "svg" else None)
        plt.close(fig)
    for path in output.glob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def report(snapshot, output):
    dual_status = snapshot.get("dual_status", {})
    dual_note = ("provider每日额度耗尽，493目标未完成，已停止新增调用；"
                 if dual_status.get("status") == "BLOCKED_PROVIDER_DAILY_LIMIT"
                 else "当前revision的计划终点为493；")
    lines = ["# 最新主动学习结果", "", f"快照时间：{snapshot['captured_at']}（Asia/Shanghai）。",
             "本页统一汇总已完成的 validation 拟合；运行中的选样或未完成训练不计作结果。", "",
             "![LLM comparison](llm_validation.png)", "",
             "## LLM 与匹配对照", "", "| Seed | 方法 | 已完成标签数 | 端点 validation NRMSE |",
             "| --- | --- | ---: | ---: |"]
    for seed, curves in snapshot["curves"].items():
        for name in CORE:
            if name in curves:
                row = curves[name][-1]
                lines.append(f"| {seed} | {name} | {row['labels']} | {row['nrmse']:.6f} |")
    lines += ["", "端点预算不同，不用上表直接排名。CW16+LLM16 的两个 seed 均到813；"
              "Free 的157/6101分别到525/653。双专家仅seed157，" + dual_note +
              "快照只报告已完成点；333→397使用astra/high，后续改用sol61/xhigh，并保留格式／证据修复记录。", "",
              "## 相同预算比较", "", "| Seed | 共同区间 | 方法 | 端点 NRMSE | Validation AULC |",
              "| --- | --- | --- | ---: | ---: |"]
    for seed, curves in snapshot["curves"].items():
        names = [name for name in CORE if name in curves]
        common = set.intersection(*[{r["labels"] for r in curves[name]} for name in names])
        end = max(common)
        for name in names:
            rows = [r for r in curves[name] if r["labels"] <= end]
            if [r["labels"] for r in rows] != list(range(333, end + 1, 32)):
                raise ValueError(f"Incomplete common-budget curve: {seed}/{name}")
            area = sum((a["nrmse"] + b["nrmse"]) / 2 * (b["labels"] - a["labels"])
                       for a, b in zip(rows, rows[1:])) / (end - 333) if end > 333 else None
            formatted = f"{area:.6f}" if area is not None else "—"
            lines.append(f"| {seed} | 333–{end} | {name} | {rows[-1]['nrmse']:.6f} | {formatted} |")
    lines += ["", "AULC为梯形积分除以该共同区间长度，越低越好。不同seed／区间之间不可直接排名。", "",
              "## 其他传统策略", "", "![All strategies](all_strategies_validation.png)", "",
              "CW-MaxDet/CW-IVR两seed均完成1005；Whitening按要求停525。"
              "原333–525确认门槛与1005探索性扩展分开解释。", "",
              "## 证据与复现", "", "[snapshot.json](snapshot.json)保留完整曲线、拟合摘要与原始文件SHA-256。"
              "checkpoint、完整请求和训练缓存仍在本地忽略的runtime目录；新克隆可直接读取或重绘快照。", "",
              "```bash", "python scripts/report_active_learning_snapshot.py --from-snapshot reports/active_learning_latest/snapshot.json",
              "```", "", "报告不打开测试真值、不训练、不调用LLM。两seed及历史修复后的曲线属于探索性开发证据，"
              "不能识别化学知识或额外模型调用的独立因果贡献。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/active_learning_latest")
    parser.add_argument("--from-snapshot", type=Path, help="redraw without local training runtime")
    parser.add_argument("--dual-revision", default=DUAL_REVISION)
    args = parser.parse_args()
    if Path(args.dual_revision).name != args.dual_revision:
        parser.error("--dual-revision must be a directory name")
    snapshot = json.loads(args.from_snapshot.read_text()) if args.from_snapshot else collect(args.dual_revision)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "snapshot.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
    draw(snapshot, args.output)
    report(snapshot, args.output)
    print(f"Saved {sum(len(v) for v in snapshot['curves'].values())} curves to {args.output}")


if __name__ == "__main__":
    main()
