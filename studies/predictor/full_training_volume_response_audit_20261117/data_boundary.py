#!/usr/bin/env python3
"""Frozen Row split, covariate-only pairs, and permanently test-closed labels."""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qgeognn_al.active_learning_v2.protocol import RestrictedLabelStore, ids_hash
from qgeognn_al.transfer.protocol import assert_fit_ids_authorized, validate_label_roles

SEEDS = (42, 525, 1101)
SOURCE = ROOT / "experiments/e0_4g_baseline/canonical_4g.csv"
QUALIFICATION = ROOT / "studies/predictor/final_4g_qualification"
SOURCE_SHA256 = "6cbe81311177ae178d10ff75a8f07c41c99a18648ec6104d2ffedca6e912d51c"
SPLIT_SHA256 = {
    42: "a8a4a322cd9c75253b1a4ed8c5c91ededed74a933df2db2685015c8b3e008008",
    525: "58876dc5a9033a0584da964e1e52bd0fa5d1aeba8131410060cce947fe11d0a1",
    1101: "fb51a225dd2b802b4dbc2982fa45166a7c11962fef9add2015e6637a1a2a6bf1",
}
FEATURE_COLUMNS = (
    "sample_id", "canonical_smiles", "Density g/ml", "V/ul", "loading solvent",
    "Volume of loading solvent/ul", "PE/EA",
)
FACTORS = ("PE/EA", "Density g/ml", "V/ul", "loading solvent", "Volume of loading solvent/ul")
AXES = ("V/ul", "PE/EA")
EXPECTED_COUNTS = {"train": 3330, "validation": 416, "test": 417}
QUARANTINED_IDS = frozenset(("c32ed74eba9d7dfc4860", "9a478db27ef04d7bb59c", "bfe9ce431b45599fd5e6"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_metadata(source: Path = SOURCE) -> pd.DataFrame:
    """Read only model covariates, including source-test X, never target columns."""
    source = Path(source)
    if sha256_file(source) != SOURCE_SHA256:
        raise ValueError("canonical source SHA256 differs from the qualification protocol")
    frame = pd.read_csv(source, usecols=list(FEATURE_COLUMNS), dtype={"sample_id": str}).loc[:, FEATURE_COLUMNS]
    if len(frame) != 4163 or frame.sample_id.nunique() != 4163 or frame.isna().any().any():
        raise ValueError("canonical metadata must contain exactly 4163 complete unique IDs")
    if frame.canonical_smiles.nunique() != 217:
        raise ValueError("canonical molecule count differs from the qualification protocol")
    frame.insert(0, "canonical_index", np.arange(len(frame), dtype=int))
    frame["global_quarantine"] = frame.sample_id.isin(QUARANTINED_IDS)
    return frame


def load_formal_split(seed: int, metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    """Verify the original split bytes, complete identity set and row identity."""
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("only the three frozen formal Row seeds are authorized")
    path = QUALIFICATION / f"splits/row_seed_{seed}.csv"
    if sha256_file(path) != SPLIT_SHA256[seed]:
        raise ValueError("formal split SHA256 mismatch")
    frame = pd.read_csv(path, dtype={"sample_id": str})
    required = {"sample_id", "source_row", "canonical_smiles", "split", "seed", "split_mode"}
    if set(frame.columns) != required or frame.sample_id.duplicated().any():
        raise ValueError("formal split schema or ID uniqueness mismatch")
    if frame.split.value_counts().to_dict() != EXPECTED_COUNTS:
        raise ValueError("formal split role counts changed")
    if set(frame.seed) != {seed} or set(frame.split_mode) != {"row"}:
        raise ValueError("formal split seed/mode mismatch")
    metadata = load_metadata() if metadata is None else metadata
    if frame.sample_id.tolist() != metadata.sample_id.tolist():
        raise ValueError("canonical ID order differs from the frozen source split")
    if frame.canonical_smiles.tolist() != metadata.canonical_smiles.tolist():
        raise ValueError("canonical molecule identity differs from the source split")
    if frame.source_row.duplicated().any():
        raise ValueError("source_row must be unique")
    frame = frame.rename(columns={"split": "source_role", "seed": "row_seed"})
    frame.insert(0, "canonical_index", metadata.canonical_index.to_numpy())
    return frame


class FormalLabelBoundary:
    """Adapt frozen source-train to official L0 while keeping test permanently shut.

    This wrapper never exposes acquisition/prediction freezing, or the official
    final-test purpose. A separate instance is required for every formal seed.
    No label values are loaded by construction.
    """

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.metadata = load_metadata()
        self.split = load_formal_split(self.seed, self.metadata)
        self.source_roles = dict(zip(self.split.sample_id, self.split.source_role))
        grouped = {role: self.split.loc[self.split.source_role.eq(role), "sample_id"].tolist()
                   for role in EXPECTED_COUNTS}
        self.roles = validate_label_roles(grouped["train"], grouped["validation"], grouped["test"],
                                          planned_budget=3746)
        adapter = self.split[["sample_id"]].copy()
        adapter["role"] = self.split.source_role.map({"train": "l0", "validation": "validation", "test": "test"})
        self._store = RestrictedLabelStore(SOURCE, adapter)
        self.audit: list[dict[str, object]] = []

    def reveal(self, ids: Iterable[str], role: str) -> pd.DataFrame:
        """Project authorized train or validation targets only, in requested order."""
        requested = [str(value) for value in ids]
        if role not in ("train", "validation"):
            raise PermissionError("test and unknown label roles are permanently blocked in this audit")
        if set(requested) & QUARANTINED_IDS:
            raise PermissionError("globally quarantined preview IDs cannot be projected in any seed")
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("label requests must be nonempty and contain unique IDs")
        if any(self.source_roles.get(sid) != role for sid in requested):
            raise PermissionError("label request includes unknown IDs or IDs outside its declared source role")
        assert_fit_ids_authorized(requested, self.roles, allow_validation=(role == "validation"))
        values = self._store.reveal(requested, purpose="initial_fit")
        receipt = dict(self._store.audit[-1])
        receipt.update(row_seed=self.seed, source_role=role, source_sha256=SOURCE_SHA256,
                       split_sha256=SPLIT_SHA256[self.seed], adapter_train_role="l0",
                       test_targets_projected=0)
        self.audit.append(receipt)
        return pd.DataFrame({"sample_id": requested, "source_role": role,
                             "V1_ml": values[:, 0], "V2_ml": values[:, 1]})

    def labels_for_pairs(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Require every endpoint to be train/validation before any projection."""
        wanted = sorted(set(pairs.low_id.astype(str)) | set(pairs.high_id.astype(str)))
        if set(wanted) & QUARANTINED_IDS:
            raise PermissionError("pair set touches a globally quarantined preview ID")
        if any(self.source_roles.get(sid) not in ("train", "validation") for sid in wanted):
            raise PermissionError("pair set contains a test/unknown endpoint; quarantine it first")
        outputs = [self.reveal([sid for sid in wanted if self.source_roles[sid] == role], role)
                   for role in ("train", "validation")
                   if any(self.source_roles[sid] == role for sid in wanted)]
        return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(
            columns=["sample_id", "source_role", "V1_ml", "V2_ml"])


def canonical_value(value: object) -> str:
    """Normalize numeric formatting without a tolerance or scientific rounding."""
    if isinstance(value, (float, int, np.number)):
        number = Decimal(str(value))
        if not number.is_finite():
            raise ValueError("condition values must be finite")
        return format(number.normalize(), "f")
    return str(value)


def axis_order(axis: str, value: object) -> Fraction:
    if axis == "PE/EA":
        a, b = str(value).split("/")
        pe, ea = Fraction(a), Fraction(b)
        if pe < 0 or ea <= 0:
            raise ValueError("invalid PE/EA mixture")
        return ea / (pe + ea)
    return Fraction(str(value))


def strict_pairs(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enumerate all cross-level ID pairs; same-level replicates remain distinct.

    Series identity is molecule plus all fixed conditions; it does not depend
    on split, seed, endpoint IDs or labels. Pair identity is axis plus unordered
    stable endpoint IDs. PE/EA pairs are ordered by increasing EA fraction.
    """
    records = metadata.to_dict("records")
    pair_rows, series_rows = [], []
    for axis in AXES:
        fixed = ["canonical_smiles"] + [factor for factor in FACTORS if factor != axis]
        grouped: dict[tuple[str, ...], list[dict]] = {}
        for row in records:
            key = tuple(canonical_value(row[name]) for name in fixed)
            grouped.setdefault(key, []).append(row)
        for key, rows in sorted(grouped.items()):
            levels = sorted({canonical_value(row[axis]) for row in rows}, key=lambda value: axis_order(axis, value))
            if len(levels) < 2:
                continue
            series_id = digest([axis, list(zip(fixed, key))])
            molecule_id = digest(key[0])
            rows = sorted(rows, key=lambda row: (axis_order(axis, row[axis]), row["sample_id"]))
            series_rows.append(dict(axis=axis, series_id=series_id, molecule_id=molecule_id,
                                    canonical_smiles=key[0], n_rows=len(rows), n_levels=len(levels),
                                    levels="|".join(levels), ids="|".join(row["sample_id"] for row in rows),
                                    fixed_conditions=json.dumps(dict(zip(fixed[1:], key[1:])), sort_keys=True)))
            for a, b in itertools.combinations(rows, 2):
                if canonical_value(a[axis]) == canonical_value(b[axis]):
                    continue
                low, high = axis_order(axis, a[axis]), axis_order(axis, b[axis])
                if not low < high:
                    raise ValueError("distinct source levels unexpectedly have identical order coordinate")
                low_id, high_id = a["sample_id"], b["sample_id"]
                pair_rows.append(dict(axis=axis, pair_id=digest([axis, sorted([low_id, high_id])]),
                                      series_id=series_id, molecule_id=molecule_id, canonical_smiles=key[0],
                                      low_id=low_id, high_id=high_id,
                                      low_level=canonical_value(a[axis]), high_level=canonical_value(b[axis]),
                                      low_coordinate=float(low), high_coordinate=float(high),
                                      coordinate_delta=float(high-low),
                                      coordinate_unit="uL" if axis == "V/ul" else "EA_volume_fraction",
                                      fixed_conditions=json.dumps(dict(zip(fixed[1:], key[1:])), sort_keys=True)))
    pairs = pd.DataFrame(pair_rows)
    series = pd.DataFrame(series_rows)
    if pairs.pair_id.duplicated().any():
        raise ValueError("strict pair IDs must be unique")
    return pairs, series


def classify_exposure(low_role: str, high_role: str) -> str:
    values = {low_role, high_role}
    if "test" in values:
        return "source_test_quarantined"
    if values == {"train"}:
        return "train_train"
    if values == {"validation"}:
        return "validation_validation"
    if values == {"train", "validation"}:
        return "train_validation"
    return "other_quarantined"


def annotate_pairs(pairs: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    roles = dict(zip(split.sample_id, split.source_role))
    result = pairs.copy()
    result.insert(0, "row_seed", int(split.row_seed.iloc[0]))
    result["low_source_role"] = result.low_id.map(roles).fillna("other")
    result["high_source_role"] = result.high_id.map(roles).fillna("other")
    result["exposure"] = [classify_exposure(a, b) for a, b in
                          zip(result.low_source_role, result.high_source_role)]
    result["source_role_combination"] = ["|".join(sorted([a, b])) for a, b in
                                         zip(result.low_source_role, result.high_source_role)]
    result["label_authorized"] = ~result.exposure.str.endswith("quarantined")
    result["global_quarantine"] = result.low_id.isin(QUARANTINED_IDS) | result.high_id.isin(QUARANTINED_IDS)
    result["scoring_allowed"] = result.label_authorized & ~result.global_quarantine
    result["scoring_exclusion_reason"] = [
        "preview_ID_global_quarantine" if quarantined else "source_test_or_other_role" if not authorized else ""
        for quarantined, authorized in zip(result.global_quarantine, result.label_authorized)]
    return result


def summarize_coverage(pairs: pd.DataFrame, series: pd.DataFrame, splits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    output, series_output = [], []
    definitions = {row.series_id: row for row in series.itertuples()}
    for (seed, axis, exposure), group in pairs.groupby(["row_seed", "axis", "exposure"], sort=True):
        endpoints = set(group.low_id) | set(group.high_id)
        joined_roles = splits.loc[splits.row_seed.eq(seed)].set_index("sample_id").source_role.to_dict()
        for series_id, item in group.groupby("series_id", sort=True):
            definition = definitions[series_id]
            selected_ids = set(item.low_id) | set(item.high_id)
            all_ids = set(definition.ids.split("|"))
            all_levels = set(definition.levels.split("|"))
            present_levels = set(item.low_level) | set(item.high_level)
            series_output.append(dict(row_seed=seed, axis=axis, exposure=exposure,
                series_id=series_id, molecule_id=item.molecule_id.iloc[0],
                n_pairs=len(item), n_rows=len(selected_ids), n_levels=len(present_levels),
                n_domain_levels=len(all_levels), all_domain_levels_present=present_levels == all_levels,
                all_domain_replicates_present=selected_ids == all_ids,
                levels="|".join(sorted(present_levels, key=lambda value: axis_order(axis, value))),
                ids="|".join(sorted(selected_ids)),
                n_domain_train_rows=sum(joined_roles[sid] == "train" for sid in all_ids),
                n_domain_validation_rows=sum(joined_roles[sid] == "validation" for sid in all_ids),
                n_domain_test_rows=sum(joined_roles[sid] == "test" for sid in all_ids)))
        delta = group.coordinate_delta.to_numpy(float)
        output.append(dict(row_seed=seed, axis=axis, exposure=exposure, n_pairs=len(group),
            n_molecules=group.molecule_id.nunique(), n_series=group.series_id.nunique(),
            n_endpoint_rows=len(endpoints), n_levels=len(set(group.low_level) | set(group.high_level)),
            delta_min=float(delta.min()), delta_q25=float(np.quantile(delta, .25)),
            delta_median=float(np.median(delta)), delta_q75=float(np.quantile(delta, .75)),
            delta_max=float(delta.max()),
            coordinate_unit=group.coordinate_unit.iloc[0], labels_projected=0,
            n_global_quarantine_pairs=int(group.global_quarantine.sum()),
            n_scoring_pairs=int(group.scoring_allowed.sum()),
            n_scoring_molecules=group.loc[group.scoring_allowed, "molecule_id"].nunique(),
            n_scoring_series=group.loc[group.scoring_allowed, "series_id"].nunique(),
            n_scoring_endpoint_rows=len(set(group.loc[group.scoring_allowed, "low_id"]) |
                                        set(group.loc[group.scoring_allowed, "high_id"]))))
    summary = pd.DataFrame(output)
    details = pd.DataFrame(series_output)
    complete = details.groupby(["row_seed", "axis", "exposure"])[
        ["all_domain_levels_present", "all_domain_replicates_present"]].sum().reset_index().rename(
        columns={"all_domain_levels_present": "n_series_all_domain_levels",
                 "all_domain_replicates_present": "n_series_all_domain_replicates"})
    return summary.merge(complete, on=["row_seed", "axis", "exposure"], validate="one_to_one"), details


def duplicate_covariates(metadata: pd.DataFrame, splits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped: dict[tuple[str, ...], list[str]] = {}
    for row in metadata.to_dict("records"):
        key = tuple(canonical_value(row[name]) for name in ["canonical_smiles", *FACTORS])
        grouped.setdefault(key, []).append(row["sample_id"])
    groups, summary = [], []
    for seed, split in splits.groupby("row_seed"):
        roles = dict(zip(split.sample_id, split.source_role))
        for key, ids in grouped.items():
            if len(ids) < 2:
                continue
            counts = Counter(roles[sid] for sid in ids)
            groups.append(dict(row_seed=seed, duplicate_group_id=digest(list(key)), canonical_smiles=key[0],
                               n_rows=len(ids), excess_rows=len(ids)-1, ids="|".join(sorted(ids)),
                               n_train=counts["train"], n_validation=counts["validation"], n_test=counts["test"],
                               role_combination="|".join(sorted(counts)),
                               conditions=json.dumps(dict(zip(FACTORS, key[1:])), sort_keys=True)))
        for role in ("train", "validation", "test", "all_metadata"):
            counts = [sum(role == "all_metadata" or roles[sid] == role for sid in ids)
                      for ids in grouped.values()]
            summary.append(dict(row_seed=seed, source_role=role,
                                n_rows=sum(counts), n_unique_condition_groups=sum(n > 0 for n in counts),
                                n_duplicate_groups=sum(n > 1 for n in counts),
                                n_rows_in_duplicate_groups=sum(n for n in counts if n > 1),
                                n_excess_duplicate_rows=sum(max(0, n-1) for n in counts),
                                targets_used=False))
    return pd.DataFrame(groups), pd.DataFrame(summary)


def _write_report(summary: pd.DataFrame, duplicates: pd.DataFrame) -> None:
    legal = summary[summary.exposure.ne("source_test_quarantined")]
    cols = ["row_seed", "axis", "exposure", "n_pairs", "n_molecules", "n_series", "n_endpoint_rows",
            "n_global_quarantine_pairs", "n_scoring_pairs"]
    def table(frame: pd.DataFrame) -> str:
        rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
                "| " + " | ".join("---" for _ in frame.columns) + " |"]
        rows.extend("| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None))
        return "\n".join(rows)
    report = f"""# 正式 Row 模型的严格条件对覆盖与标签边界

三个模型各有 **3330 条 source-train、416 条 source-validation、417 条 source-test**，完整特征域为4163条、217个分子。Row seed 525 是模型随机种子，与 AL 标签预算 L525 无关。

## 无标签覆盖计数

{table(legal[cols])}

train_train 衡量已见端点上的拟合；validation_validation 才是两端均未用于梯度拟合的对照，仍属于原本用于 checkpoint 选择的 validation，不能称为全新独立测试。train_validation 单独标记为部分已见。所有涉及 source-test 的 pair 只保留 X 身份与隔离计数，不请求目标，也不参与评分。

“系列”固定 canonical SMILES 及另外四项条件，至少有两个变化因素水平；同一系列的多个 pair 与同分子的多个系列均不是独立科研重复。`n_series_all_domain_levels` 表示相应曝光子集包含该系列在4163条特征域出现过的全部水平；全部重复行也覆盖的更严格计数另列在CSV。它不表示潜在物理实验域已穷尽。

## 精确规则与稳定身份

仅改变样品体积 `V/ul` 或仅改变 `PE/EA`，其余四项条件与 canonical SMILES 必须精确相等，不设近似容差。样品体积和 `Volume of loading solvent/ul` 分别匹配。数值仅消除格式差异（例如30与30.0），不四舍五入；PE/EA固定条件使用源字符串精确匹配，排序按EA体积分数增加。体积以uL、PE/EA变化坐标以EA体积分数表示。没有把名义范围30–150内不存在的60/90水平当作已观察点。

`source_metadata.csv` 是完整4163条无目标特征；`formal_split_roles.csv` 给出三个原始split的12489条ID角色。`formal_pairs.csv` 的 pair_id=SHA256(axis,无序端点IDs)，跨模型保持相同，series_id由分子与固定条件决定。全部重复测量的跨水平ID组合都保留；同水平重复不冒充因素变化。`formal_pair_series.csv`、`formal_pair_level_counts.csv` 与 `formal_pair_delta_distribution.csv` 提供系列、水平、跨度和原始多重性。所有覆盖统计保留完整metadata域，但三条预览ID全局隔离，任何触碰它们的pair均`global_quarantine=True,scoring_allowed=False`；原split角色和3330训练数不改。`n_scoring_pairs`为真正允许下游评分的子集；`formal_role_counts.csv`另列每角色剔除后的可评分行数。

## 同条件重复与噪声边界

{table(duplicates[duplicates.source_role.isin(['train','validation','all_metadata'])])}

重复指所有已记录输入条件相同、sample_id不同。此表只依据协变量，不能判定是重复测量、独立实验批次、误录或未记录因素，也不凭无标签表宣称噪声大小。授权train/validation若比较重复目标离散程度，必须按各模型角色分别投影；含test的重复组仍隔离。

## 可复用的角色受限接口

历史正式资格验证runner直接读取全canonical表，未提供专用角色store；本轮不执行那个runner。`FormalLabelBoundary(seed)`先核验canonical字节SHA与冻结split字节SHA及逐ID/SMILES/行序，调用官方 `transfer.protocol.validate_label_roles`。独立适配器只把该模型原始train映射到官方 `RestrictedLabelStore` 的l0，validation/test保持原角色，不修改冻结split，也不重新划分。`reveal(ids, role='train'|'validation')` 先全局阻断三条预览ID、再校验请求完全属于声明角色，调用 `assert_fit_ids_authorized` 与官方 `RestrictedLabelStore.reveal(..., purpose='initial_fit')`；test、未知、重复和跨角色请求在打开目标store之前被拒绝。本接口不暴露freeze或final_test_evaluation解锁。`labels_for_pairs`在任何数值投影之前拒绝整个含test/未知/全局隔离端点的输入。

本脚本CLI只生成无标签覆盖表，**目标投影0行、模型计算0次、训练0次、LLM/API0次**。官方streaming store读取CSV字符串行后，只对请求授权ID的V1_ml/V2_ml转换float32；文件hash也经过所有字节。因此不声称目标文件字节从未被操作系统读取。下游调用会产生独立官方请求receipt；本覆盖生成器没有目标receipt。

## 文本查看边界例外

初次核查源manifest格式时，审阅命令误用了 `head -4 qualification_dataset_manifest.csv`，显示了头部3行目标文本。其中第一行是Row1101的test、第三行是Row525的test。此文本查看发生在正式loader建立之前；这些值未写入本轮表、未数值投影、未用于任何指标/模型/选择。不能声称本轮全过程test目标文本零可见。`data_boundary_verification.json`与`LABEL_BOUNDARY_INCIDENT.json`透明保存例外的身份及角色、零分析使用范围；不复制数值。**三个ID在所有seed全局隔离**，即使原角色为train也不再投影或评分，且接口在官方store访问前硬阻断。另一个agent初期sed命令把同一目标manifest列为第二文件，累计行号使其没有输出任何目标内容；记录为打开顺读、0额外预览ID，不等同新增数值评分。后续所有标签请求必须经过上述封闭接口。

## 复现与范围

```bash
/Users/fish/miniforge3/envs/chromatography/bin/python studies/predictor/full_training_volume_response_audit_20261117/data_boundary.py
/Users/fish/miniforge3/envs/chromatography/bin/python -m pytest -q studies/predictor/full_training_volume_response_audit_20261117/tests/test_data_boundary.py
```

覆盖表只回答哪些身份可比，不替代预测响应评价。AL共同对由独立模块按相同pair身份及每个正式seed的角色再分层；不同Row seed共享数据，不能当作独立随机种子配对实验。
"""
    (OUT / "PAIR_COVERAGE.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata()
    splits = pd.concat([load_formal_split(seed, metadata) for seed in SEEDS], ignore_index=True)
    splits["role_label_authorized"] = splits.source_role.isin(["train", "validation"])
    splits["global_quarantine"] = splits.sample_id.isin(QUARANTINED_IDS)
    splits["label_authorized"] = splits.role_label_authorized & ~splits.global_quarantine
    metadata.to_csv(OUT / "source_metadata.csv", index=False)
    splits.to_csv(OUT / "formal_split_roles.csv", index=False)
    base_pairs, series = strict_pairs(metadata)
    pairs = pd.concat([annotate_pairs(base_pairs, splits[splits.row_seed.eq(seed)]) for seed in SEEDS], ignore_index=True)
    summary, series_detail = summarize_coverage(pairs, series, splits)
    duplicate_groups, duplicate_summary = duplicate_covariates(metadata, splits)
    pairs.to_csv(OUT / "formal_pairs.csv", index=False)
    series.to_csv(OUT / "domain_response_series.csv", index=False)
    summary.to_csv(OUT / "formal_pair_coverage.csv", index=False)
    series_detail.to_csv(OUT / "formal_pair_series.csv", index=False)
    duplicate_groups.to_csv(OUT / "covariate_duplicate_groups.csv", index=False)
    duplicate_summary.to_csv(OUT / "covariate_duplicate_summary.csv", index=False)
    levels, deltas = [], []
    m = metadata.set_index("sample_id")
    for (seed, axis, exposure), group in pairs.groupby(["row_seed", "axis", "exposure"]):
        ids = set(group.low_id) | set(group.high_id)
        counts = Counter(canonical_value(m.loc[sid, axis]) for sid in ids)
        for level, count in sorted(counts.items(), key=lambda item: axis_order(axis, item[0])):
            levels.append(dict(row_seed=seed, axis=axis, exposure=exposure, level=level, n_endpoint_rows=count))
        for delta, count in group.coordinate_delta.value_counts().sort_index().items():
            deltas.append(dict(row_seed=seed, axis=axis, exposure=exposure, coordinate_delta=delta,
                               coordinate_unit=group.coordinate_unit.iloc[0], n_pairs=count))
    pd.DataFrame(levels).to_csv(OUT / "formal_pair_level_counts.csv", index=False)
    pd.DataFrame(deltas).to_csv(OUT / "formal_pair_delta_distribution.csv", index=False)
    role_counts = splits.groupby(["row_seed", "source_role"]).agg(n_rows=("sample_id", "size"),
                n_molecules=("canonical_smiles", "nunique"),
                n_global_quarantine_rows=("global_quarantine", "sum"),
                n_scoring_rows=("label_authorized", "sum")).reset_index()
    role_counts.to_csv(OUT / "formal_role_counts.csv", index=False)
    incident = dict(status="DISCLOSED_TEXT_INSPECTION_INCIDENT_GLOBAL_QUARANTINE", command="head -4 qualification_dataset_manifest.csv",
        source="studies/predictor/final_4g_qualification/splits/qualification_dataset_manifest.csv",
        visible_header_rows=1, visible_target_rows=3, displayed_lines=4,
        quarantined_ids=sorted(QUARANTINED_IDS),
        previewed_id_roles=[dict(sample_id=sid, row_seed=int(row.row_seed), source_role=row.source_role,
                                source_row=int(row.source_row))
                           for sid in sorted(QUARANTINED_IDS)
                           for row in splits.loc[splits.sample_id.eq(sid)].itertuples()],
        target_values_copied_into_artifacts=False, numeric_projection_rows=0,
        used_for_metrics_or_models_or_selection=False,
        affected_test_id_roles=[dict(sample_id="c32ed74eba9d7dfc4860", row_seed=1101, source_role="test"),
                                dict(sample_id="bfe9ce431b45599fd5e6", row_seed=525, source_role="test")],
        secondary_open_without_numeric_display=dict(agent="protocol", operation="sed with split CSV first and target manifest second; cumulative line numbering",
            additional_preview_ids=[], visible_target_rows=0, numeric_projection_rows=0,
            scope="target file opened/streamed; only the first split CSV lines were displayed"),
        remedy="All three previewed IDs are globally quarantined in every seed; role-gated label access and scored pairs reject them before numeric projection.")
    (OUT / "LABEL_BOUNDARY_INCIDENT.json").write_text(json.dumps(incident, indent=2) + "\n")
    input_paths = [SOURCE, QUALIFICATION / "protocol.json", QUALIFICATION / "splits/split_manifest.json",
        ROOT / "src/qgeognn_al/active_learning_v2/protocol.py", ROOT / "src/qgeognn_al/transfer/protocol.py",
        ROOT / "scripts/studies/run_final_4g_qualification.py", Path(__file__),
        *(QUALIFICATION / f"splits/row_seed_{seed}.csv" for seed in SEEDS)]
    inputs = [dict(path=str(path.relative_to(ROOT)), sha256=sha256_file(path), bytes=path.stat().st_size) for path in input_paths]
    manifest = dict(inputs=inputs, canonical_feature_columns=list(FEATURE_COLUMNS),
        target_access="CLI covariates only; target values projected zero; one initial text-inspection exception recorded separately")
    (OUT / "data_boundary_inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    checks = dict(status="PASS_WITH_EXPLICIT_TEXT_INSPECTION_EXCEPTION", seeds=list(SEEDS),
        domain_rows=len(metadata), domain_unique_ids=metadata.sample_id.nunique(), domain_molecules=metadata.canonical_smiles.nunique(),
        per_seed_role_counts=EXPECTED_COUNTS, domain_strict_pair_counts=base_pairs.groupby("axis").size().to_dict(),
        numeric_label_rows_projected_by_cli=0, test_numeric_label_rows_projected=0, test_targets_used_for_analysis=0,
        initial_text_inspection_exception=incident, model_inference_calls=0, model_fits=0, llm_api_calls=0,
        inputs_manifest_sha256=sha256_file(OUT / "data_boundary_inputs.json"))
    (OUT / "data_boundary_verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    _write_report(summary, duplicate_summary)
    print(json.dumps({"domain_rows": len(metadata), "pair_counts": checks["domain_strict_pair_counts"],
                      "numeric_label_rows_projected": 0}, sort_keys=True))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
