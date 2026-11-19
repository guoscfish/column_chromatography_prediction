#!/usr/bin/env python3
"""Recompute figures F06–F07 from saved observations and saved predictions."""

import csv
from collections import Counter, defaultdict
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
from io import BytesIO, StringIO
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np


REPORT = Path(__file__).resolve().parents[1]
ROOT = REPORT.parents[2]
VOLUME = ROOT / "studies/active_learning/volume_response_retry_audit_20261117"
FOLLOW = ROOT / "studies/active_learning/decision_mechanism_followup_20261117"
SOURCES = REPORT / "sources"
SEEDS = (157, 887, 2357, 6101, 12203)
ARMS = ("A_original", "B_amount_skip", "C_volume_skip")
TARGETS = ("V1", "V2")
PROVENANCE = {}
CHECKS = []


def read(path):
    blob = path.read_bytes()
    PROVENANCE[path.relative_to(ROOT).as_posix()] = sha256(blob).hexdigest()
    return blob


def rows(path):
    return list(csv.DictReader(StringIO(read(path).decode())))


def document(path):
    return json.loads(read(path))


def emit(name, data):
    with (SOURCES / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def compare(key, new, original, tolerance=2e-12):
    delta = max(abs(float(value) - float(original[field])) for field, value in new.items())
    if delta > tolerance:
        raise ValueError(f"{key}: error {delta:.6g} > {tolerance:.6g}")
    CHECKS.append({"key": key, "maximum_absolute_difference": delta})


def responses(records):
    actual, predicted = zip(*records)
    relevant = [(a, p) for a, p in records if abs(a) > 0.5]
    flat = sum(abs(p) < 0.1 for _, p in relevant)
    wrong = sum(abs(p) >= 0.1 and a * p < 0 for a, p in relevant)
    return dict(
        pairs=len(records), meaningful_pairs=len(relevant),
        observed_abs_mean=mean(map(abs, actual)),
        predicted_abs_mean=mean(map(abs, predicted)),
        difference_mae=mean(abs(a-p) for a, p in records),
        flat_fraction_meaningful=flat / len(relevant),
        wrong_nonflat_fraction_meaningful=wrong / len(relevant),
        correct_nonflat_fraction_meaningful=(len(relevant)-flat-wrong) / len(relevant),
    )


def frozen_history():
    groups = defaultdict(list)
    seen = defaultdict(set)
    hashes = defaultdict(set)
    file = VOLUME / "frozen_response_pairs.csv"
    for row in rows(file):
        seed, budget, axis, target = int(row["seed"]), int(row["budget"]), row["axis"], row["target"]
        pair = row["low_id"], row["high_id"]
        key = seed, budget, axis, target
        assert row["role"].startswith("in-sample known history")
        assert pair not in seen[key]
        seen[key].add(pair)
        hashes[seed, budget].add(row["checkpoint_sha256"])
        groups[key].append((float(row["actual_delta_ml"]), float(row["predicted_delta_ml"])))
    source_digest = document(VOLUME / "sensitivity_summary_manifest.json")["sources"]
    expected = {item["path"]: item["sha256"] for item in source_digest}
    assert PROVENANCE[file.relative_to(ROOT).as_posix()] == expected[file.relative_to(ROOT).as_posix()]
    assert len(hashes) == 10 and all(len(state) == 1 for state in hashes.values())
    source = {
        (int(r["seed"]), int(r["budget"]), r["axis"], r["target"]): r
        for r in rows(VOLUME / "frozen_response_seed_budget.csv")
    }
    assert groups.keys() == source.keys()
    seed_rows = []
    for (seed, budget, axis, target), records in sorted(groups.items()):
        alternate = "V2" if target == "V1" else "V1"
        assert seen[seed, budget, axis, target] == seen[seed, budget, axis, alternate]
        statistics = responses(records)
        compare(f"F06 data {seed}/{budget}/{axis}/{target}", statistics, source[seed, budget, axis, target])
        seed_rows.append(dict(seed=seed, budget=budget, axis=axis, target=target,
                              checkpoint_sha256=next(iter(hashes[seed, budget])), **statistics))
    aggregate = {(int(r["budget"]), r["axis"], r["target"]): r for r in rows(
        VOLUME / "frozen_response_equal_seed.csv"
    )}
    fields = {
        "observed_abs_seed_mean": "observed_abs_mean",
        "predicted_abs_seed_mean": "predicted_abs_mean",
        "difference_mae_seed_mean": "difference_mae",
        "flat_fraction_meaningful_seed_mean": "flat_fraction_meaningful",
        "correct_nonflat_seed_mean": "correct_nonflat_fraction_meaningful",
        "wrong_nonflat_seed_mean": "wrong_nonflat_fraction_meaningful",
    }
    summary = []
    for (budget, axis, target), reference in aggregate.items():
        matched = [r for r in seed_rows if (r["budget"], r["axis"], r["target"]) == (budget, axis, target)]
        assert {r["seed"] for r in matched} == set(SEEDS)
        statistics = {field: mean(r[metric] for r in matched) for field, metric in fields.items()}
        compare(f"F06 mean {budget}/{axis}/{target}", statistics, reference)
        summary.append(dict(budget=budget, axis=axis, target=target, seeds=len(matched),
                            pairs=sum(r["pairs"] for r in matched), **statistics))
    emit("f06_frozen_response_seed.csv", seed_rows)
    emit("f06_frozen_response_equal_seed.csv", summary)
    return summary


def selected_pairs():
    file = FOLLOW / "condition_response_pairs.csv"
    cohorts = defaultdict(list)
    endpoints = defaultdict(list)
    for row in rows(file):
        if (row["method"], row["source_half"], row["axis"]) != ("new_free_cw_sol61_xhigh", "all32", "V/ul"):
            continue
        assert row["information_time"] == "predictions before batch; outcomes only after this accepted batch freeze"
        seed, target = int(row["seed"]), row["target"]
        cohorts[seed, target].append((float(row["actual_delta_ml"]), float(row["predicted_delta_ml"])))
        endpoints[seed, row["round"], row["low_id"], row["high_id"]].append(target)
    assert len(endpoints) == 165 and all(sorted(e) == list(TARGETS) for e in endpoints.values())
    assert Counter(key[0] for key in endpoints) == dict(zip(SEEDS, (55, 21, 28, 27, 34)))
    output = [dict(seed=seed, target=target, **responses(records)) for (seed, target), records in sorted(cohorts.items())]
    summary = []
    fields = list(responses(cohorts[SEEDS[0], TARGETS[0]]))[2:]
    for target in TARGETS:
        matched = [r for r in output if r["target"] == target]
        for weighting in ("seed_equal", "pooled_pairs"):
            pooled = responses([pair for seed in SEEDS for pair in cohorts[seed, target]])
            average = {key: mean(r[key] for r in matched) if weighting == "seed_equal" else pooled[key] for key in fields}
            summary.append(dict(target=target, weighting=weighting, pairs=pooled["pairs"], seeds=len(matched), **average))
    emit("selected_165_volume_pairs_seed.csv", output)
    emit("selected_165_volume_pairs_summary.csv", summary)
    return summary


def coordinate(row):
    pe_ea = row["PE/EA"].strip().replace(":", "/")
    return (row["canonical_smiles"], Fraction(pe_ea), Decimal(row["Density g/ml"]),
            row["loading solvent"], Decimal(row["Volume of loading solvent/ul"]), Decimal(row["V/ul"]))


def fit_results():
    source_rows = {
        (int(r["seed"]), r["kind"]): r for r in rows(VOLUME / "control_seed_summary.csv")
    }
    source_pairs = {
        (int(r["seed"]), r["kind"], r["target"], r["low_id"], r["high_id"]): r
        for r in rows(VOLUME / "control_response_pairs.csv") if r["scope"] == "check" and r["axis"] == "V/ul"
    }
    source_robustness = {
        (int(r["seed"]), r["kind"]): r for r in rows(VOLUME / "control_robustness.csv")
    }
    prior_qa = document(VOLUME / "RESULTS_INDEPENDENT_QA.json")
    prior_source_hashes = {item["path"]: item["sha256"] for item in prior_qa["input_sources"]}
    plan = document(VOLUME / "ANALYSIS_PLAN.json")
    assert {(f["seed"], f["kind"]) for f in plan["fits"]} == {
        (seed, arm) for seed in SEEDS for arm in ARMS
    }
    summary, counts = [], []
    seen_pairs = set()
    for seed in SEEDS:
        base = VOLUME / "data" / f"seed_{seed}"
        identity = document(base / "identity.json")
        assert identity["unobserved_or_test_target_values_projected"] == 0
        assert identity["history_budget"] == 1005
        id_sets = {role: identity[f"{role}_ids"] for role in ("train", "check", "validation")}
        assert len(id_sets["validation"]) == 416
        assert set(id_sets["train"]).isdisjoint(id_sets["check"])
        assert set(id_sets["train"]) | set(id_sets["check"]) == set(identity["history_ids"])
        metadata = rows(base / "metadata.csv")
        assert PROVENANCE[(base / "metadata.csv").relative_to(ROOT).as_posix()] == identity["metadata_sha256"]
        assert [row["sample_id"] for row in metadata] == identity["ordered_ids"]
        relevant_ids = {sid for ids in id_sets.values() for sid in ids}
        coordinates = {row["sample_id"]: coordinate(row) for row in metadata if row["sample_id"] in relevant_ids}
        series = defaultdict(list)
        for sid in id_sets["check"]:
            series[coordinates[sid][:-1]].append(sid)
        assert len(series) == 10
        assert not (series.keys() & {coordinates[sid][:-1] for sid in id_sets["train"]})
        val_coordinates = {coordinates[sid] for sid in id_sets["validation"]}
        related = series.keys() & {row[:-1] for row in val_coordinates}
        pairs = [
            (low, high, key) for key, ids in series.items()
            for low, high in combinations(sorted(ids, key=lambda sid: coordinates[sid][-1]), 2)
            if coordinates[low][-1] != coordinates[high][-1]
        ]
        counts.append(dict(
            seed=seed, training_rows=len(id_sets["train"]), check_rows=len(id_sets["check"]),
            validation_rows=len(id_sets["validation"]), check_series=len(series), volume_pairs=len(pairs),
            validation_related_series=len(related),
            exact_condition_rows=sum(coordinates[sid] in val_coordinates for sid in id_sets["check"]),
        ))
        authorized_labels = read(base / "authorized_labels.npz")
        assert sha256(authorized_labels).hexdigest() == identity["authorized_labels_sha256"]
        with np.load(BytesIO(authorized_labels), allow_pickle=False) as data:
            # Access only the two frozen evaluation arrays. Training labels are unnecessary.
            truths = {scope: data[f"{scope}_y"] for scope in ("check", "validation")}
        actual = {}
        for scope, array in truths.items():
            assert array.shape == (len(id_sets[scope]), 2) and array.dtype == np.float32
            assert np.isfinite(array).all()
            actual[scope] = dict(zip(id_sets[scope], array))

        fit_identities = set()
        for arm in ARMS:
            fit = VOLUME / "fits" / f"seed_{seed}" / arm
            audit = document(fit / "fit_audit.json")
            assert audit["status"] == "COMPLETE" and (audit["seed"], audit["kind"]) == (seed, arm)
            assert audit["test_truth_used"] == audit["unobserved_targets_used"] == 0
            assert audit["holdout_used_for_early_stopping"] is False
            for scope in ("train", "check", "validation"):
                assert audit[f"{scope}_ids_hash"] == identity[f"{scope}_ids_hash"]
            fit_identities.add(tuple(audit[key] for key in (
                "common_initial_hash", "post_init_torch_rng_hash", "initialization_seed"
            )))
            predictions = {}
            for scope in ("check", "validation"):
                table = rows(fit / f"{scope}_predictions.csv")
                predictions[scope] = {row["sample_id"]: [float(row[f"{target}_q50"]) for target in TARGETS] for row in table}
                assert len(table) == len(predictions[scope]) and predictions[scope].keys() == actual[scope].keys()
            scales = identity["preprocessing"]["target_scales"]
            normalized_mse = [
                mean((p[i]-float(actual["validation"][sid][i]))**2 for sid, p in predictions["validation"].items())
                / float(scales[target])**2
                for i, target in enumerate(TARGETS)
            ]
            nrmse = math.sqrt(mean(normalized_mse))
            assert abs(nrmse - audit["best_validation_nrmse"]) < 1e-7
            endpoint_stats, series_equal, unrelated_stats = [], [], []
            for i, target in enumerate(TARGETS):
                differences, errors_by_series, unrelated = [], defaultdict(list), []
                for low, high, key in pairs:
                    a = float(actual["check"][high][i] - actual["check"][low][i])
                    p = predictions["check"][high][i] - predictions["check"][low][i]
                    pair_key = seed, arm, target, low, high
                    reference = source_pairs[pair_key]
                    assert (reference["validation_related_series"] == "True") == (key in related)
                    compare(f"F07 pair {seed}/{arm}/{target}/{low}/{high}",
                            dict(actual_delta_ml=a, predicted_delta_ml=p), reference)
                    seen_pairs.add(pair_key)
                    differences.append((a, p))
                    errors_by_series[key].append(abs(a - p))
                    if key not in related:
                        unrelated.append(abs(a - p))
                endpoint_stats.append(responses(differences))
                series_equal.append(mean(map(mean, errors_by_series.values())))
                unrelated_stats.append(mean(unrelated))
            statistics = dict(
                difference_error_mean=mean(r["difference_mae"] for r in endpoint_stats),
                observed_abs_mean=mean(r["observed_abs_mean"] for r in endpoint_stats),
                predicted_abs_mean=mean(r["predicted_abs_mean"] for r in endpoint_stats),
                flat_fraction_meaningful=mean(r["flat_fraction_meaningful"] for r in endpoint_stats),
                combined_normalized_rmse=nrmse,
            )
            compare(f"F07 seed/arm {seed}/{arm}", statistics, source_rows[seed, arm])
            robust = dict(difference_error_mean_series_equal=mean(series_equal),
                          difference_error_mean_validation_unrelated=mean(unrelated_stats))
            compare(f"F07 robustness {seed}/{arm}", robust, source_robustness[seed, arm])
            summary.append(dict(seed=seed, kind=arm, **statistics, **robust))
        assert len(fit_identities) == 1
    assert seen_pairs == source_pairs.keys() and len(seen_pairs) == 408
    assert sum(c["volume_pairs"] for c in counts) == 68
    for path, digest in PROVENANCE.items():
        if path in prior_source_hashes:
            assert digest == prior_source_hashes[path], path
    aggregates = {r["kind"]: r for r in rows(VOLUME / "control_equal_seed_summary.csv")}
    arm_rows = []
    for arm in ARMS:
        matched = [r for r in summary if r["kind"] == arm]
        statistics = {field: mean(r[field] for r in matched) for field in matched[0] if field not in ("seed", "kind")}
        compare(f"F07 mean {arm}", {f: v for f, v in statistics.items() if f in aggregates[arm]}, aggregates[arm])
        arm_rows.append(dict(kind=arm, seeds=len(matched), **statistics))
    emit("f07_control_seed.csv", summary)
    emit("f07_control_equal_seed.csv", arm_rows)
    emit("f07_check_denominators.csv", counts)
    gates = report_gates(plan, summary, arm_rows)
    return arm_rows, gates


def report_gates(plan, seed_rows, arm_rows):
    arm_mean = {r["kind"]: r for r in arm_rows}
    paired = {(r["seed"], r["kind"]): r for r in seed_rows}
    reference = {r["comparison"]: r for r in rows(VOLUME / "mechanism_gates.csv")}
    mechanism, row_gate = plan["mechanism_gate"], plan["row_relevance_gate"]
    result = []
    for first, second in ((ARMS[1], ARMS[0]), (ARMS[2], ARMS[0]), (ARMS[2], ARMS[1])):
        name = f"{first[0]}-{second[0]}"
        local, row = "difference_error_mean", "combined_normalized_rmse"
        local_mean_improvement = 1 - arm_mean[first][local] / arm_mean[second][local]
        row_change = arm_mean[first][row] / arm_mean[second][row] - 1
        local_seed_improvement = [1 - paired[seed, first][local] / paired[seed, second][local] for seed in SEEDS]
        row_seed_change = [paired[seed, first][row] / paired[seed, second][row] - 1 for seed in SEEDS]
        statistics = dict(
            local_improvement_ratio_equal_seed_means=local_mean_improvement,
            mean_paired_seed_relative_local_improvement=mean(local_seed_improvement),
            local_positive_seeds=sum(v > 0 for v in local_seed_improvement),
            row_relative_change_equal_seed_means=row_change, row_nonworse_seeds=sum(v <= 0 for v in row_seed_change),
            row_worse_over_one_percent_seeds=sum(v > 0.01 for v in row_seed_change),
        )
        compare(f"F07 gate {name}", statistics, reference[name])
        mechanism_pass = (
            local_mean_improvement >= mechanism["local_relative_improvement"]
            and statistics["local_positive_seeds"] >= mechanism["positive_outer_seeds"]
            and row_change <= mechanism["row_relative_worsening_max"]
        )
        row_pass = (
            -row_change >= row_gate["row_relative_improvement"]
            and statistics["row_nonworse_seeds"] >= row_gate["nonworse_outer_seeds"]
        )
        assert str(mechanism_pass) == reference[name]["mechanism_reference_pass"]
        assert str(row_pass) == reference[name]["row_relevance_reference_pass"]
        result.append(dict(comparison=name, **statistics, mechanism_reference_pass=mechanism_pass,
                           row_relevance_reference_pass=row_pass))
    emit("f07_control_gates.csv", result)
    return result


def metrics(history, selected, fits, gates):
    entries = []

    def entry(figure, metric, method, value, unit, n, budget, level, split, version, frozen,
              comparability, limitation, sources):
        entries.append(dict(
            figure=figure, metric=metric, method=method, value=value, unit=unit, n=n,
            seeds="157/887/2357/6101/12203", budget=budget, level=level, split=split,
            version=version, frozen=frozen, comparability=comparability, limitation=limitation,
            source=" | ".join(p.relative_to(ROOT).as_posix() for p in sources),
        ))

    for r in history:
        for key, label, unit in (
            ("observed_abs_seed_mean", "真实平均绝对变化", "mL"),
            ("predicted_abs_seed_mean", "预测平均绝对变化", "mL"),
            ("difference_mae_seed_mean", "差分 MAE", "mL"),
            ("flat_fraction_meaningful_seed_mean", "有意义真实变化中的近乎平坦比例", "fraction"),
        ):
            entry("F06", f"{label} {r['axis']} {r['target']}", "New CW-Free32 / 原 QGeoGNN",
                  r[key], unit, f"{r['pairs']} 同分子单因素对/端点", f"L{r['budget']}", "row pairs", "已观察训练 H",
                  "A2 十个旧冻结模型", "是", "同预算同因素同端点，五 seed 等权",
                  "训练内机制诊断；两个预算的 H 及 pair 构成不同；平坦阈值 |预测Δ|<0.1 mL、|真实Δ|>0.5 mL",
                  [VOLUME / "frozen_response_pairs.csv", VOLUME / "frozen_response_equal_seed.csv"])
    for r in selected:
        for key, label in (("observed_abs_mean", "真实"), ("predicted_abs_mean", "预测")):
            entry("F06", f"讲稿旁证 · 选中批次{label}平均绝对变化 {r['target']} ({r['weighting']})", "New CW-Free32", r[key],
                  "mL", "165 对/端点", "轨迹 L333–1485，选前 L333–1453", "row pairs", "已接受批次揭示反馈", "decision followup A",
                  "选前预测、接受后标签", "同批同分子单因素对比",
                  "选中样本事后分析，与 F06 冻结 H 诊断及 F07 隔离系列分别统计",
                  [FOLLOW / "condition_response_pairs.csv", FOLLOW / "MODEL_NUMERIC_REVIEW.md"])
    for r in fits:
        for key, label, unit, n, level, split in (
            ("difference_error_mean", "局部差分 MAE", "mL", "68 对/端点；每 seed 10 系列", "row pairs", "H1005 隔离检查"),
            ("combined_normalized_rmse", "4g Row combined NRMSE", "dimensionless", "416 行/拟合", "row", "原 validation"),
            ("observed_abs_mean", "真实平均绝对体积响应", "mL", "68 对/端点", "row pairs", "H1005 隔离检查"),
            ("predicted_abs_mean", "预测平均绝对体积响应", "mL", "68 对/端点", "row pairs", "H1005 隔离检查"),
            ("difference_error_mean_validation_unrelated", "剔除 validation 关联系列后局部差分 MAE",
             "mL", "59 对/端点", "row pairs", "H1005 隔离检查的敏感性子集"),
        ):
            entry("F07", label, r["kind"], r[key], unit, n, "L1005 固定历史训练标签，按预定系列扣除检查",
                  level, split, "volume response 三臂 × 五 seed", "冻结标签与已保存预测",
                  "同 seed 配对；五 seed 等权；Row 使用固定 L333 目标尺度",
                  "检查系列不参与训练或早停；有原 validation 条件关联；固定标签对照，无 AL 效能估计",
                  [VOLUME / "control_response_pairs.csv", VOLUME / "row_metrics.csv",
                   VOLUME / "RESULTS_INDEPENDENT_QA.json"])
    for r in gates:
        for key, label, unit in (
            ("local_improvement_ratio_equal_seed_means", "局部差分 MAE 相对改善", "fraction"),
            ("local_positive_seeds", "局部 MAE 改善 seed 数", "seed"),
            ("row_relative_change_equal_seed_means", "Row NRMSE 相对变化（正为恶化）", "fraction"),
            ("row_nonworse_seeds", "Row NRMSE 不变差 seed 数", "seed"),
            ("row_worse_over_one_percent_seeds", "Row NRMSE 变差超过 1% 的 seed 数", "seed"),
            ("mechanism_reference_pass", "机制参考门槛通过", "boolean"),
            ("row_relevance_reference_pass", "Row 参考门槛通过", "boolean"),
        ):
            entry("F07", label, r["comparison"], r[key], unit, "5 配对 seed", "L1005 固定历史训练标签",
                  "row pairs / row", "H1005 隔离检查 / 原 validation", "volume response 三臂 × 五 seed",
                  "门槛值预设；聚合公式见原说明", "同 seed、同标签配对", "工程参考门槛，无显著性推断",
                  [VOLUME / "ANALYSIS_PLAN.json", VOLUME / "mechanism_gates.csv",
                   VOLUME / "RESULTS_INDEPENDENT_QA.json"])
    emit("model_metrics_fragment.csv", entries)


def main():
    history = frozen_history()
    selected = selected_pairs()
    fits, gates = fit_results()
    metrics(history, selected, fits, gates)
    for path in [
        VOLUME / "audit_figures/a2_frozen_condition_amplitudes.pdf",
        VOLUME / "audit_figures/a2_frozen_condition_amplitudes.png",
        VOLUME / "figures/controls_paired_seeds.pdf",
        VOLUME / "figures/controls_paired_seeds.png",
        VOLUME / "figures/controls_response_amplitude.png",
        VOLUME / "MODEL_COMPARISON.md",
        VOLUME / "check_series_validation_overlap.csv",
        FOLLOW / "MODEL_LEARNABILITY.md",
        FOLLOW / "MODEL_NUMERIC_REVIEW.md",
    ]:
        read(path)
    provenance = {
        "scope": "F06/F07 frozen artifact audit. No inference, new fits or LLM calls. No test/future-label access.",
        "input_sha256": dict(sorted(PROVENANCE.items())),
        "reference_comparisons": CHECKS,
    }
    (SOURCES / "model_input_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+"\n")
    worst = max(c["maximum_absolute_difference"] for c in CHECKS)
    print(f"F06/F07: {len(CHECKS)} reference comparisons, max absolute difference {worst:.3g}")


if __name__ == "__main__":
    main()
