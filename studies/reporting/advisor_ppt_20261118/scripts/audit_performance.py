"""Export saved validation curves and independently recompute budget metrics."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median, stdev


OUT = Path(__file__).resolve().parent.parent
ROOT = OUT.parents[2]
AL = Path("studies/active_learning")
LLM = AL / "llm_decision_behavior_audit_20261116"
HISTORY = AL / "comprehensive_history_diagnostics_20261112"
FOLLOWUP = AL / "decision_mechanism_followup_20261117"
FINAL = (
    AL / "nocw_vs_cw_lcmd_five_to1485_20261112"
    / "reports/final_five_seed_L1485_20261116"
)
ROW_CURVES = LLM / "row_curves.csv"
ROUND_METRICS = HISTORY / "tables/round_metrics.csv"
SOURCE_FULL_CW = Path("reports/free_cw_final_20261112/validation_curves.csv")
SOURCE_SIX = Path("reports/six_methods_both_free_20261112/validation_curves.csv")
SOURCE_FIVE = FINAL / "validation_curves.csv"
FIVE = (157, 887, 2357, 6101, 12203)
TWO = (157, 6101)
CW = "cw_gradient_lcmd32"
HYBRID = "free16_lcmd16"
OLD = "historical_free_mixed_model"
NEW = "new_free_cw_sol61_xhigh"
NOCW = "nocw"
RAW = "gradient_lcmd32"
NUMERICAL = "numerical_hybrid32"
RANDOM = "random32"
WINDOWS = {
    "long_five_seed": (1485, FIVE, (CW, OLD, NEW, NOCW)),
    "hybrid_two_seed": (1453, TWO, (HYBRID, CW)),
    "short_five_seed": (1005, FIVE, (NUMERICAL, RAW, OLD, NEW, NOCW, RANDOM)),
}
FIGURES = {"long_five_seed": "F01/F02", "hybrid_two_seed": "F03", "short_five_seed": "F08"}
METHOD_LABELS = {
    CW: "CW-LCMD32",
    HYBRID: "CW16+LLM16",
    OLD: "Historical CW-Free32",
    NEW: "New CW-Free32",
    NOCW: "NoCW-Free32",
    RAW: "raw Gradient-LCMD32",
    NUMERICAL: "Numerical Hybrid32",
    RANDOM: "Random32",
}
VERSIONS = {
    CW: "center_width_lcmd，五seed封存版 20261116",
    HYBRID: "cw16_llm16_scientist_v2，20261112 历史长轨迹",
    OLD: "free_llm32_scientist_v2，20261112 历史混合配置",
    NEW: "free_cw_sol61_xhigh_rerun32，20261112 五seed完整快照",
    NOCW: "llm32_nocw，20261116 长窗封存 / 20261112 短窗快照",
    RAW: "lcmd，原始 V1/V2 q50 梯度几何，20261112 综合诊断",
    NUMERICAL: "hybrid，三初始化分歧筛前25%后 latent 覆盖，20261112 综合诊断",
    RANDOM: "random32，历史嵌套随机基线，20261112 综合诊断",
}
LIMITATIONS = {
    CW: "CW 梯度几何与 raw Gradient-LCMD 分开；后续训练状态分叉",
    HYBRID: "仅157/6101；Astra/high前缀，选前L813起Sol/xhigh；两seed历史轨迹，无配对重训",
    OLD: "157/6101含Astra/high前缀；157一次接受ID抄写修正的served identity未知",
    NEW: "Sol/xhigh独立重跑；日期、纠错机制及历史上下文与其他Free轨迹有差异",
    NOCW: "仅去除显式CW字段与说明，其他数值信息保留；历史纠错机制有差异",
    RAW: "历史raw q50梯度空间基线；与CW几何另列",
    NUMERICAL: "历史三成员分歧加latent覆盖；与CW16+LLM16另列",
    RANDOM: "五seed共同截止L1005；更长历史延伸只覆盖两seed",
}
PROVENANCE = set()


def read_csv(path):
    PROVENANCE.add(path)
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    PROVENANCE.add(path)
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def seal_check(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Archived input SHA256 differs: {path}")


def write_csv(name, fields, records):
    with (OUT / "sources" / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def key(row):
    return row["method"], int(row["seed"]), int(row["budget"])


def index(rows, key_fn=key):
    result = {}
    for row in rows:
        record_key = key_fn(row)
        if record_key in result:
            raise ValueError(f"Duplicate record: {record_key}")
        result[record_key] = row
    return result


def close(a, b):
    if not math.isclose(float(a), float(b), abs_tol=2e-13, rel_tol=0):
        raise ValueError(f"Validation archive disagreement: {a!r} vs {b!r}")
    return abs(float(a) - float(b))


def normalized_aulc(budgets, errors):
    areas = [
        (right - left) * (before + after) / 2
        for left, right, before, after in zip(budgets, budgets[1:], errors, errors[1:])
    ]
    return math.fsum(areas) / (budgets[-1] - budgets[0])


def threshold_record(window, method, seed, points, threshold):
    budgets = [int(point["budget"]) for point in points]
    at_target = [float(point["nrmse"]) <= threshold for point in points]
    first = next((budget for budget, hit in zip(budgets, at_target) if hit), None)
    triple = next(
        (start for start in range(len(budgets) - 2) if all(at_target[start : start + 3])),
        None,
    )
    return dict(
        window=window, method=method, seed=seed, threshold=threshold, observed_to=budgets[-1],
        first_budget=first, first_right_censored=int(first is None),
        three_consecutive_first_budget=budgets[triple] if triple is not None else None,
        three_consecutive_confirmation_budget=budgets[triple + 2] if triple is not None else None,
        three_consecutive_right_censored=int(triple is None),
    )


def source_for(window, method):
    if window == "short_five_seed" or window == "raw_reference_long":
        return str(ROUND_METRICS)
    if method in (CW, NOCW):
        return f"{ROW_CURVES} | {SOURCE_FIVE}"
    if method == HYBRID:
        return f"{ROW_CURVES} | {SOURCE_SIX}"
    return f"{ROW_CURVES} | {SOURCE_FULL_CW} | {SOURCE_SIX}"


def record(window, method, metric, value, n, seeds, budget, unit="dimensionless", extra="",
           source=None):
    figure = FIGURES.get(window, "F02/F08")
    return dict(
        figure=figure, metric=metric, method=METHOD_LABELS[method], value=value, unit=unit,
        n=n, seeds="/".join(map(str, seeds)), budget=budget, level="Row-level 4g",
        split="validation", version=VERSIONS[method], frozen="已归档/封存的验证记录",
        comparability="相同seed、起点、验证口径及共同预算；历史描述性配对",
        limitation=LIMITATIONS[method] + extra, source=source or source_for(window, method),
    )


def main():
    old_manifest = {item["path"]: item["sha256"] for item in read_json(LLM / "OUTPUT_MANIFEST.json")}
    history_manifest = {
        item["path"]: item["sha256"]
        for item in read_json(HISTORY / "verification/output_manifest.json")
    }
    final_manifest = read_json(FINAL / "artifact_hashes.json")
    phase_manifest = {
        item["path"]: item["sha256"] for item in read_json(FOLLOWUP / "OUTPUT_MANIFEST.json")
    }
    input_manifest = {
        item["path"]: item["sha256"] for item in read_json(LLM / "INPUT_MANIFEST.json")["sources"]
    }
    for path in (ROW_CURVES, LLM / "performance_summary.csv"):
        seal_check(path, old_manifest[path.name])
    for path in (ROUND_METRICS, HISTORY / "tables/stage_metrics.csv",
                 HISTORY / "tables/threshold_crossings.csv"):
        seal_check(path, history_manifest[str(path.relative_to(HISTORY))])
    for path in (SOURCE_FIVE, FINAL / "paired_results.csv"):
        seal_check(path, final_manifest[path.name])
    for path in (SOURCE_FULL_CW, SOURCE_SIX, SOURCE_FIVE):
        seal_check(path, input_manifest[str(path)])
    seal_check(FOLLOWUP / "phase_overall.csv", phase_manifest["phase_overall.csv"])

    archived = index(read_csv(ROW_CURVES))
    historical = index(read_csv(ROUND_METRICS))
    native = {
        str(SOURCE_FULL_CW): (index(read_csv(SOURCE_FULL_CW)), "validation_combined_normalized_rmse"),
        str(SOURCE_SIX): (index(read_csv(SOURCE_SIX)), "validation_nrmse"),
        str(SOURCE_FIVE): (index(read_csv(SOURCE_FIVE)), "validation_nrmse"),
    }
    original_metrics = {
        "row_curves": index(
            read_csv(LLM / "performance_summary.csv"),
            lambda row: (row["method"], int(row["seed"]), int(row["L_end"])),
        ),
        "round_metrics": index(
            [row for row in read_csv(HISTORY / "tables/stage_metrics.csv") if row["start"] == "333"],
            lambda row: (row["method"], int(row["seed"]), int(row["end"])),
        ),
    }
    old_thresholds = index(
        read_csv(HISTORY / "tables/threshold_crossings.csv"),
        lambda row: (row["method"], int(row["seed"]), float(row["threshold"])),
    )
    phase = index(read_csv(FOLLOWUP / "phase_overall.csv"), lambda row: int(row["seed"]))
    phase_scores = {
        HYBRID: "hybrid_AULC",
        CW: "cw_AULC",
    }
    final_metrics = index(read_csv(FINAL / "paired_results.csv"), lambda row: int(row["seed"]))
    errors_native, errors_historical, errors_integral, errors_summary = [], [], [], []
    ups = {}
    baseline_by_seed = {}
    curves, metrics, thresholds, pairs, summaries, metrics_fragment = [], [], [], [], [], []
    for window, (high, seeds, methods) in WINDOWS.items():
        budgets = tuple(range(333, high + 1, 32))
        for method in methods:
            group = []
            for seed in seeds:
                points = []
                for budget in budgets:
                    point_key = (method, seed, budget)
                    if window == "short_five_seed":
                        source_row = historical[point_key]
                        if point_key in archived:
                            errors_historical.append(close(source_row["nrmse"], archived[point_key]["nrmse"]))
                    else:
                        source_row = archived[point_key]
                        source_records, nrmse_key = native[source_row["source"]]
                        native_method = "cw" if method == CW and source_row["source"] == str(SOURCE_FIVE) else method
                        native_nrmse = source_records[native_method, seed, budget][nrmse_key]
                        errors_native.append(close(source_row["nrmse"], native_nrmse))
                        if point_key in historical:
                            errors_historical.append(close(source_row["nrmse"], historical[point_key]["nrmse"]))
                    points.append(dict(window=window, method=method, seed=seed, budget=budget,
                                       nrmse=source_row["nrmse"], source=source_row["source"]))
                errors = [float(point["nrmse"]) for point in points]
                if not all(math.isfinite(value) and value >= 0 for value in errors):
                    raise ValueError(f"Invalid archived NRMSE: {window}/{method}/{seed}")
                initial = baseline_by_seed.setdefault(seed, errors[0])
                close(initial, errors[0])
                # Independent equal-step identity weights the boundary points by a half.
                area = normalized_aulc(budgets, errors)
                equal_step = math.fsum((errors[0] / 2, *errors[1:-1], errors[-1] / 2)) / (len(errors) - 1)
                errors_integral.append(close(area, equal_step))
                which = "round_metrics" if window == "short_five_seed" else "row_curves"
                published = original_metrics[which][method, seed, high]
                errors_summary.append(close(area, published["aulc" if which == "round_metrics" else "AULC"]))
                errors_summary.append(close(errors[-1], published["endpoint_nrmse" if which == "round_metrics" else "endpoint"]))
                if window == "hybrid_two_seed":
                    errors_summary.append(close(area, phase[seed][phase_scores[method]]))
                if window == "long_five_seed" and method in (CW, NOCW):
                    prefix = "cw" if method == CW else "nocw"
                    errors_summary.append(close(area, final_metrics[seed][f"{prefix}_raw_AULC"]))
                    errors_summary.append(close(errors[-1], final_metrics[seed][f"{prefix}_endpoint"]))
                fit = dict(window=window, method=method, seed=seed, budget_start=333, budget_end=high,
                           n_points=len(points), aulc=area, endpoint=errors[-1])
                curves.extend(points)
                metrics.append(fit)
                group.append(fit)
                thresholds.append(threshold_record(window, method, seed, points, 0.4))
                ups[f"{window}/{method}/{seed}"] = sum(a < b for a, b in zip(errors, errors[1:]))
                metrics_fragment.append(record(window, method, "raw AULC by seed", area, 1,
                                               (seed,), f"L333–{high}"))
                metrics_fragment.append(record(window, method, "endpoint NRMSE by seed", errors[-1], 1,
                                               (seed,), f"L{high}"))
            aggregate = dict(window=window, method=method, budget_start=333, budget_end=high,
                             seed_n=len(group), n_points=sum(row["n_points"] for row in group))
            for field in ("aulc", "endpoint"):
                values = [row[field] for row in group]
                aggregate[f"{field}_mean"] = mean(values)
                aggregate[f"{field}_sample_sd"] = stdev(values)
                label = "raw AULC" if field == "aulc" else "endpoint validation NRMSE"
                budget = f"L333–{high}" if field == "aulc" else f"L{high}"
                metrics_fragment.append(record(window, method, f"{label} mean", mean(values),
                                               len(group), seeds, budget))
                metrics_fragment.append(record(window, method, f"{label} sample SD", stdev(values),
                                               len(group), seeds, budget, extra="；seed离散度，非置信区间"))
            summaries.append(aggregate)

    seed_metrics = index(metrics, lambda row: (row["window"], row["method"], row["seed"]))
    raw_long = []
    raw_budgets = list(range(333, 1486, 32))
    for seed in FIVE:
        points = [historical[RAW, seed, budget] for budget in raw_budgets]
        thresholds.append(threshold_record("raw_reference_long", RAW, seed, points, 0.4))
        errors = [float(point["nrmse"]) for point in points]
        area = normalized_aulc(raw_budgets, errors)
        published = original_metrics["round_metrics"][RAW, seed, 1485]
        errors_summary.append(close(area, published["aulc"]))
        errors_summary.append(close(errors[-1], published["endpoint_nrmse"]))
        new = seed_metrics["long_five_seed", NEW, seed]
        raw_long.append(dict(seed=seed, raw_aulc=area, raw_endpoint=errors[-1],
                             new_minus_raw_aulc=new["aulc"] - area,
                             new_minus_raw_endpoint=new["endpoint"] - errors[-1]))
    for metric, label, budget in (
        ("raw_aulc", "raw AULC mean (separate raw-gradient historical reference)", "L333–1485"),
        ("raw_endpoint", "endpoint validation NRMSE mean (separate raw-gradient historical reference)", "L1485"),
    ):
        metrics_fragment.append(record("raw_reference_long", RAW, label,
                                       mean(row[metric] for row in raw_long), len(FIVE), FIVE, budget))
    metrics_fragment.append(record(
        "raw_reference_long", NEW, "strictly lower seed count at L1485 vs separate raw Gradient-LCMD32",
        sum(row["new_minus_raw_endpoint"] < 0 for row in raw_long), len(FIVE), FIVE, "L1485", "seeds",
        extra="；对照为 raw 梯度 LCMD，与 F02 的 CW-LCMD 比较不同",
        source=f"{ROUND_METRICS} | {ROW_CURVES}",
    ))

    for window, reference, methods in (
        ("long_five_seed", CW, (OLD, NEW, NOCW)),
        ("hybrid_two_seed", CW, (HYBRID,)),
        ("short_five_seed", RAW, (NEW,)),
        ("short_five_seed", NUMERICAL, (NEW,)),
        ("short_five_seed", RANDOM, (NEW,)),
    ):
        high, seeds, _ = WINDOWS[window]
        for method in methods:
            pair_group = []
            for seed in seeds:
                treatment = seed_metrics[window, method, seed]
                control = seed_metrics[window, reference, seed]
                differences = dict(
                    window=window, method=method, reference=reference, seed=seed,
                    aulc_delta=treatment["aulc"] - control["aulc"],
                    endpoint_delta=treatment["endpoint"] - control["endpoint"],
                )
                pairs.append(differences)
                pair_group.append(differences)
            for metric, label, budget in (
                ("aulc_delta", "paired raw AULC delta vs", f"L333–{high}"),
                ("endpoint_delta", "paired endpoint NRMSE delta vs", f"L{high}"),
            ):
                source = " | ".join(dict.fromkeys(
                    f"{source_for(window, method)} | {source_for(window, reference)}".split(" | ")
                ))
                metrics_fragment.append(record(
                    window, method, f"{label} {METHOD_LABELS[reference]}", mean(row[metric] for row in pair_group),
                    len(seeds), seeds, budget, extra="；负差有利于所列方法；历史路径差，非因果效应",
                    source=source,
                ))
                metrics_fragment.append(record(
                    window, method, f"strictly lower seed count for {label} {METHOD_LABELS[reference]}",
                    sum(row[metric] < 0 for row in pair_group), len(seeds), seeds, budget, "seeds",
                    extra="；方向计数，非统计显著性检验", source=source,
                ))

    threshold_checks = 0
    for item in thresholds:
        old = old_thresholds.get((item["method"], item["seed"], item["threshold"]))
        if old is None:
            continue
        observed_to = item["observed_to"]
        for label, margin in (("first_budget", 0), ("three_consecutive_first_budget", 64)):
            old_value = old[label]
            if old_value and float(old_value) + margin <= min(float(old["observed_to"]), observed_to):
                if item[label] != int(float(old_value)):
                    raise ValueError(f"Archived threshold disagreement: {item}")
                threshold_checks += 1

    threshold_groups = defaultdict(list)
    for item in thresholds:
        threshold_groups[item["window"], item["method"]].append(item)
        seed = (item["seed"],)
        for label, title in (
            ("first_budget", "L at first observed NRMSE≤0.40"),
            ("three_consecutive_first_budget", "L at first of three consecutive NRMSE≤0.40"),
            ("three_consecutive_confirmation_budget", "L confirming three consecutive NRMSE≤0.40"),
        ):
            value = item[label]
            metrics_fragment.append(record(
                item["window"], item["method"], title, value if value is not None else "right_censored",
                1, seed, f"L333–{item['observed_to']}", "training row labels",
                extra="；历史诊断阈值；仅计实际检查点，未达标保留右删失",
            ))
    for (window, method), group in threshold_groups.items():
        for label, title in (
            ("first_budget", "observed first crossings NRMSE≤0.40"),
            ("three_consecutive_confirmation_budget", "observed three-consecutive confirmations NRMSE≤0.40"),
        ):
            values = [row[label] for row in group if row[label] is not None]
            seeds = tuple(row["seed"] for row in group)
            budget = f"L333–{group[0]['observed_to']}"
            metrics_fragment.append(record(window, method, title, len(values), len(group), seeds,
                                           budget, "seeds", extra="；历史诊断阈值；未达标保留右删失"))
            if len(values) == len(group):
                metrics_fragment.append(record(window, method, f"median labels among {title}", median(values),
                                               len(group), seeds, budget, "training row labels"))

    for name, rows in (
        ("performance_curves.csv", curves),
        ("performance_seed_metrics.csv", metrics),
        ("performance_summary.csv", summaries),
        ("performance_paired_differences.csv", pairs),
        ("performance_thresholds.csv", thresholds),
        ("performance_raw_long_reference.csv", raw_long),
        ("performance_metrics_fragment.csv", metrics_fragment),
    ):
        write_csv(name, list(rows[0]), rows)
    quality = {
        "scope": "Existing 4g row validation summaries only. No raw labels, candidate truth, tests, fits or acquisitions.",
        "grid": "Complete checkpoints from L333 in steps of 32; one value per method, seed and budget.",
        "window_points": {window: sum(row["window"] == window for row in curves) for window in WINDOWS},
        "native_archive_point_comparisons": len(errors_native),
        "native_archive_max_abs_difference": max(errors_native),
        "overlap_with_earlier_history_point_comparisons": len(errors_historical),
        "overlap_with_earlier_history_max_abs_difference": max(errors_historical),
        "published_summary_comparisons": len(errors_summary),
        "published_summary_max_abs_difference": max(errors_summary),
        "independent_integration_comparisons": len(errors_integral),
        "independent_integration_max_abs_difference": max(errors_integral),
        "published_threshold_value_checks": threshold_checks,
        "raw_increasing_steps": ups,
        "source_sha256": {str(path): digest(path) for path in sorted(PROVENANCE)},
    }
    (OUT / "sources" / "performance_qa.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
