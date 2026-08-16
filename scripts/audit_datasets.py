#!/usr/bin/env python3
"""Audit chromatography CSV datasets without mutating the source files."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd


DATASETS = {
    "4g": ("dataset_4g.csv", "Silica-CS 4g", 60, 120),
    "8g": ("dataset_8g.csv", "Silica-CS 4g+4g", 60, 120),
    "25g": ("dataset_25g.csv", "Silica-CS 25g", 60, 120),
    "40g": ("dataset_40g.csv", "Silica-CS 40g", 150, 200),
    "DCM": ("dataset_DCM.csv", "Silica-CS 4g", 60, 120),
    "C18": ("dataset_C18.csv", "C18", 60, 120),
    "NH2": ("dataset_NH2.csv", "NH2", 60, 120),
    "CN": ("dataset_CN.csv", "CN", 60, 120),
}

CORE_COLUMNS = [
    "CAS",
    "Density g/ml",
    "V/ul",
    "loading solvent",
    "Volume of loading solvent/ul",
    "PE/EA",
    "Flow mL/min",
    "column_specs",
    "t1",
    "t2",
    "smiles",
]

CONDITION_COLUMNS = [
    "PE/EA",
    "Flow mL/min",
    "loading solvent",
    "Volume of loading solvent/ul",
]

EXPERIMENT_KEY_COLUMNS = [
    "smiles",
    "PE/EA",
    "Flow mL/min",
    "Density g/ml",
    "V/ul",
    "loading solvent",
    "Volume of loading solvent/ul",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ratio_is_valid(value: object) -> bool:
    if pd.isna(value):
        return False
    try:
        left, right = str(value).strip().split("/")
        left_value = float(left)
        right_value = float(right)
        return left_value >= 0 and right_value >= 0 and (left_value + right_value) > 0
    except (TypeError, ValueError):
        return False


def parse_smiles_validity(values: pd.Series) -> tuple[pd.Series, str]:
    try:
        from rdkit import Chem
    except ImportError:
        return pd.Series([pd.NA] * len(values), index=values.index, dtype="boolean"), "rdkit_unavailable"

    def is_valid(value: object) -> bool:
        if pd.isna(value) or not str(value).strip():
            return False
        return Chem.MolFromSmiles(str(value)) is not None

    return values.map(is_valid).astype("boolean"), "rdkit_parse"


def unique_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(df[column].nunique(dropna=True))


def audit_dataset(
    dataset_id: str,
    path: Path,
    expected_spec: str,
    threshold_v1_ml: float,
    threshold_v2_ml: float,
) -> tuple[list[dict], dict, list[dict]]:
    df = pd.read_csv(path)
    missing_columns = [column for column in CORE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")

    source_hash = sha256_file(path)
    t1 = pd.to_numeric(df["t1"], errors="coerce")
    t2 = pd.to_numeric(df["t2"], errors="coerce")
    flow = pd.to_numeric(df["Flow mL/min"], errors="coerce")
    density = pd.to_numeric(df["Density g/ml"], errors="coerce")
    sample_volume = pd.to_numeric(df["V/ul"], errors="coerce")
    loading_volume = pd.to_numeric(df["Volume of loading solvent/ul"], errors="coerce")

    ratio_valid = df["PE/EA"].map(ratio_is_valid)
    smiles_valid, smiles_check = parse_smiles_validity(df["smiles"])
    spec_match = df["column_specs"].eq(expected_spec)
    core_not_missing = (
        df[["CAS", "loading solvent", "PE/EA", "column_specs", "smiles"]].notna().all(axis=1)
        & t1.notna()
        & t2.notna()
        & flow.notna()
        & density.notna()
        & sample_volume.notna()
        & loading_volume.notna()
    )
    # Reproduce the effective filters in utils.py and Construct_dataset_*.
    code_reader_mask = spec_match & t1.ne(-1) & t1.notna()
    volume_1_ml = t1 * flow / 1200.0
    volume_2_ml = t2 * flow / 1200.0
    qgeognn_threshold_mask = (
        code_reader_mask
        & volume_1_ml.le(threshold_v1_ml)
        & volume_2_ml.le(threshold_v2_ml)
    )

    stages = [
        (
            "raw_csv",
            pd.Series(True, index=df.index),
            "No filtering; source CSV as committed",
            "observed",
        ),
        (
            "code_reader_compatible",
            code_reader_mask,
            "column_specs matches; t1 != -1; t1 is not NaN",
            "reproduces current utils.py reader",
        ),
        (
            "current_qgeognn_effective",
            qgeognn_threshold_mask,
            f"reader-compatible and t1*flow/1200 <= {threshold_v1_ml:g} "
            f"and t2*flow/1200 <= {threshold_v2_ml:g}",
            "reproduces current Construct_dataset_* label thresholds",
        ),
    ]

    manifest_rows: list[dict] = []
    for order, (stage, mask, rule, note) in enumerate(stages):
        subset = df.loc[mask]
        output_rows = len(subset)
        manifest_rows.append(
            {
                "dataset_id": dataset_id,
                "stage_order": order,
                "stage": stage,
                "filter_rule": rule,
                "input_rows": len(df),
                "removed_from_raw": len(df) - output_rows,
                "output_rows": output_rows,
                "unique_cas": unique_count(subset, "CAS"),
                "unique_smiles": unique_count(subset, "smiles"),
                "unique_conditions": int(subset[CONDITION_COLUMNS].drop_duplicates().shape[0]),
                "source_file": str(path.as_posix()),
                "source_sha256": source_hash,
                "expected_column_spec": expected_spec,
                "threshold_v1_ml": threshold_v1_ml,
                "threshold_v2_ml": threshold_v2_ml,
                "status": "observed",
                "notes": note,
            }
        )

    exact_duplicates = df.duplicated(keep=False)
    repeated_experiment_keys = df.duplicated(EXPERIMENT_KEY_COLUMNS, keep=False)
    issue_details: list[dict] = []
    for index in df.index:
        issue_flags = []
        if not core_not_missing.at[index]:
            issue_flags.append("missing_core")
        if t1.at[index] < 0 or t2.at[index] < 0:
            issue_flags.append("negative_label")
        if pd.notna(t1.at[index]) and pd.notna(t2.at[index]) and t1.at[index] > t2.at[index]:
            issue_flags.append("t1_gt_t2")
        if not ratio_valid.at[index]:
            issue_flags.append("invalid_ratio")
        if pd.notna(smiles_valid.at[index]) and not bool(smiles_valid.at[index]):
            issue_flags.append("invalid_smiles")
        if not spec_match.at[index]:
            issue_flags.append("unexpected_column_spec")
        if exact_duplicates.at[index]:
            issue_flags.append("exact_duplicate")
        if repeated_experiment_keys.at[index]:
            issue_flags.append("repeated_experiment_key")
        if code_reader_mask.at[index] and not qgeognn_threshold_mask.at[index]:
            issue_flags.append("removed_by_qgeognn_threshold")
        if not issue_flags:
            continue

        row_payload = df.loc[index].fillna("<NA>").astype(str).to_dict()
        row_hash = hashlib.sha1(
            json.dumps(row_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:12]
        issue_details.append(
            {
                "dataset_id": dataset_id,
                "source_row_1based": int(index) + 2,
                "record_hash": row_hash,
                "CAS": df.at[index, "CAS"],
                "smiles": df.at[index, "smiles"],
                "column_specs": df.at[index, "column_specs"],
                "PE_EA": df.at[index, "PE/EA"],
                "flow_ml_min": flow.at[index],
                "t1_raw": t1.at[index],
                "t2_raw": t2.at[index],
                "V1_ml_current_formula": volume_1_ml.at[index],
                "V2_ml_current_formula": volume_2_ml.at[index],
                "issue_flags": ";".join(issue_flags),
                "current_reader_keeps": bool(code_reader_mask.at[index]),
                "current_qgeognn_keeps": bool(qgeognn_threshold_mask.at[index]),
            }
        )

    over_v1 = code_reader_mask & volume_1_ml.gt(threshold_v1_ml)
    over_v2 = code_reader_mask & volume_2_ml.gt(threshold_v2_ml)
    summary = {
        "dataset_id": dataset_id,
        "source_file": str(path.as_posix()),
        "source_sha256": source_hash,
        "source_rows": len(df),
        "source_columns": len(df.columns),
        "expected_column_spec": expected_spec,
        "threshold_v1_ml": threshold_v1_ml,
        "threshold_v2_ml": threshold_v2_ml,
        "observed_column_specs": " | ".join(sorted(map(str, df["column_specs"].dropna().unique()))),
        "unique_cas": unique_count(df, "CAS"),
        "unique_smiles": unique_count(df, "smiles"),
        "unique_conditions": int(df[CONDITION_COLUMNS].drop_duplicates().shape[0]),
        "missing_core_rows": int((~core_not_missing).sum()),
        "negative_label_rows": int((t1.lt(0) | t2.lt(0)).sum()),
        "t1_gt_t2_rows": int((t1.gt(t2) & t1.notna() & t2.notna()).sum()),
        "invalid_ratio_rows": int((~ratio_valid).sum()),
        "invalid_smiles_rows": int((smiles_valid.eq(False) & smiles_valid.notna()).sum()),
        "exact_duplicate_rows": int(exact_duplicates.sum()),
        "repeated_experiment_key_rows": int(repeated_experiment_keys.sum()),
        "code_reader_rows": int(code_reader_mask.sum()),
        "over_v1_threshold_rows": int(over_v1.sum()),
        "over_v2_threshold_rows": int(over_v2.sum()),
        "removed_by_threshold_union": int((code_reader_mask & ~qgeognn_threshold_mask).sum()),
        "current_qgeognn_rows": int(qgeognn_threshold_mask.sum()),
        "smiles_check": smiles_check,
    }
    return manifest_rows, summary, issue_details


def write_report(output_path: Path, summaries: list[dict]) -> None:
    summary_by_id = {row["dataset_id"]: row for row in summaries}
    eight = summary_by_id["8g"]
    lines = [
        "# QGeoGNN E0-1 数据口径审计报告",
        "",
        f"日期：{date.today().isoformat()}  ",
        "范围：当前仓库 `dataset/*.csv`；源文件只读，未改写。",
        "",
        "## 结论先行",
        "",
        "本项目以当前仓库 CSV 为唯一数据源，论文行数不作为实验口径或阻塞项。当前仓库 8g 的两个实际数字可以这样解释：",
        "",
        f"- **{eight['source_rows']}**：当前 `dataset_8g.csv` 的原始行数。",
        f"- **{eight['current_qgeognn_rows']}**：按现有 `Construct_dataset_8g()` 的阈值 `V1<=60 mL`、`V2<=120 mL` 过滤后的有效建模行数。",
        f"- 574→552 共排除 **{eight['removed_by_threshold_union']}** 行：V1 超阈值 {eight['over_v1_threshold_rows']} 行，V2 超阈值 {eight['over_v2_threshold_rows']} 行，两者有重叠。",
        "",
        "因此，**552 不是另一份 8g 原始数据，而是当前 574 行 CSV 经过模型构图阶段标签阈值过滤后的数量**。论文中的其他行数按已确认协议忽略。",
        "",
        "## 主数据集摘要",
        "",
        "| 数据集 | 原始行 | 当前 reader | 当前 QGeoGNN | 缺失核心 | 负标签 | t1>t2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset_id in ["4g", "8g", "25g", "40g"]:
        row = summary_by_id[dataset_id]
        lines.append(
            f"| {dataset_id} | {row['source_rows']} | {row['code_reader_rows']} | "
            f"{row['current_qgeognn_rows']} | "
            f"{row['missing_core_rows']} | {row['negative_label_rows']} | {row['t1_gt_t2_rows']} |"
        )

    lines.extend(
        [
            "",
            "## 当前规则的风险",
            "",
            "1. `utils.py` 只显式过滤 `t1==-1` 和 `t1` 缺失，没有对 `t2` 缺失/负值及 `t1>t2` 做统一处理。",
            "2. 60/120 mL 阈值在构图时静默删除记录，导致 reader 行数与实际建模行数不同。",
            "3. 阈值是按构图函数硬编码的：4g/8g/25g 为60/120 mL，40g为150/200 mL；即便40g使用更宽阈值，仍会排除73条reader记录。",
            "4. 重复实验已确认为真实重复测量并保留；严格评估时，同组记录应进入同一 split 以避免泄漏。",
            "",
            "## Gate 0 的当前决定与下一项",
            "",
            "- 基线按原代码保留 60/120 mL 阈值；阈值必要性放入后续消融，不把它表述为已验证的物理边界。",
            "- 重复实验全部保留，视为同一实验的重复测量。",
            "- 记录 `raw_csv`、`code_reader_compatible` 和 `current_qgeognn_effective` 三种现有代码口径。",
            "- 后续 split、scaler、图缓存都必须引用明确的 manifest stage 与 source hash。",
            "- 4g 过滤后的训练 CSV 已导出到 `experiments/e0_4g_baseline/canonical_4g.csv`，并附逐行决策表。",
            "- 4g 已补齐单位换算、RDKit/3D/图构建成功率及最终 canonical 口径；8g 将在 E0-3 开始时补齐。",
            "",
            "## 交付物",
            "",
            "- `data_manifest.csv`：逐数据集、逐过滤阶段的行数与哈希。",
            "- `dataset_summary.csv`：异常、阈值和重复键汇总。",
            "- `issue_details.csv`：需核查记录的源行号、问题标志和当前去留状态。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/data_audit"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_manifest: list[dict] = []
    all_summaries: list[dict] = []
    all_issues: list[dict] = []
    for dataset_id, (filename, expected_spec, threshold_v1_ml, threshold_v2_ml) in DATASETS.items():
        source_path = args.dataset_dir / filename
        manifest, summary, issues = audit_dataset(
            dataset_id,
            source_path,
            expected_spec,
            threshold_v1_ml,
            threshold_v2_ml,
        )
        all_manifest.extend(manifest)
        all_summaries.append(summary)
        all_issues.extend(issues)

    manifest_df = pd.DataFrame(all_manifest)
    summary_df = pd.DataFrame(all_summaries)
    issues_df = pd.DataFrame(all_issues)
    manifest_df.to_csv(args.output_dir / "data_manifest.csv", index=False)
    summary_df.to_csv(args.output_dir / "dataset_summary.csv", index=False)
    issues_df.to_csv(args.output_dir / "issue_details.csv", index=False)

    write_report(args.output_dir / "AUDIT_REPORT.md", all_summaries)

    print(json.dumps({
        "datasets": len(all_summaries),
        "manifest_rows": len(all_manifest),
        "issue_rows": len(all_issues),
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
