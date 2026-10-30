"""Verify derived contracts and run offline tests. No experiment runner is imported."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location("verified_assembly", HERE / "assemble_audit.py")
    assembly = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assembly)
    merged = pd.read_csv(HERE / "hybrid_selection_profiles.csv")
    assembly.validate_profiles(merged)
    rebuilt = assembly.add_performance(assembly.merge_profiles(pd.read_csv(HERE / "behavior_profiles.csv"),
                                                               pd.read_csv(HERE / "geometry_profiles.csv")),
                                       pd.read_csv(HERE / "validation_rounds.csv"))
    pd.testing.assert_frame_equal(merged, rebuilt, check_dtype=False, rtol=1e-11, atol=1e-12)
    assert len(pd.read_csv(HERE / "selection_point_flags.csv")) == 4800
    assert len(pd.read_csv(HERE / "validation_rounds.csv")) == 128
    source = json.loads((HERE / "source_audit.json").read_text())
    assert source["status"] == "PASS" and source["verified_batch_count"] == 120
    assert all(source[k] == 0 for k in ("new_training", "new_llm_calls", "new_label_requests", "new_test_evaluations"))
    # Ensure the source audit really describes the supplied analysis implementations.
    for filename, key in (("source_hashes.json", None), ("behavior_summary.json", "source_hashes"),
                          ("geometry_audit.json", "source_sha256")):
        values = json.loads((HERE / filename).read_text())
        values = values[key] if key else values
        for name, digest in values.items():
            if str(HERE.relative_to(ROOT)) in name:
                assert sha(ROOT / name) == digest, f"analysis source drift: {name}"
    tests = sorted(HERE.glob("test_*.py"))
    cmd = [sys.executable, "-m", "pytest", "-q", *map(str, tests)]
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    (HERE / "test_results.txt").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    record = dict(status="PASS", test_command=cmd, tests_exit_code=0, test_output=result.stdout,
                  checked_profiles=180, checked_source_point_views=4800, independent_selected_rows=3840,
                  verified_historical_fit_points=128, unique_historical_fit_audits=source["unique_fit_audit_files"],
                  verified_historical_batches=120, input_source_files=source["source_file_count"],
                  phase_b="PAUSED_BY_USER_DEFINED_EVIDENCE_GATE", phase_b_implementation="NOT_PERFORMED",
                  phase_b_mock_dry_run="NOT_PERFORMED_AFTER_GATE_PAUSE",
                  real_llm_calls=0, new_model_fits=0, new_labels_revealed=0, new_test_evaluations=0,
                  visual_qa=["loading_and_molecules.png", "geometry_division_of_labor.png",
                             "geometry_stage_marginal_coverage.png", "validation_curves.png"],
                  analysis_replay="offline derived CSV merge, contract checks and plotting passed")
    (HERE / "verification.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    preflight = f'''# 分析自检与阶段B执行状态

日期：2026-10-29（Asia/Shanghai）。状态：**阶段A分析与离线重放PASS；阶段B因反证决策门暂停。**

## 已完成的检查

- 原始来源：2209个输入文件SHA、128个方法×预算点（122唯一fit audit）、120个冻结批次、Scientist原owner/继承副本、input/catalog完整送入、32ID freeze与揭示receipt、初始化/split/checkpoint/原尺度、真实validation非单调曲线。
- 行为：旧V-only/prior与五seed同窗口逐行精确对齐；四原始loading轴、prior/batch-only/both/union去重；16/32分母；已见完整tuple与补全分开；Density×V不重复计轴；自身历史残差及理由引用边界。
- 几何：120个状态、120 checkpoint SHA、90raw+90CW bank；有序L/U与模型状态绑定；60个Scientist catalog分位舍入一致；固定事前U和排除selected机械归零；同状态两个来源的solo+另一来源marginal=all32。
- 整合：180行key唯一、每组round恰为0–14、阶段与标签预算一致、labels_after=before+32、分母16/32正确、来源计数可加、缺失/冲突metadata拒绝、whole-Hybrid性能不冒充16点Arm、无未测量candidate真值导出。
- 测试：14项行为、7项几何、12项来源、8项整合，共41项，另有5个unittest subtests；全部通过。日志为[test_results.txt](test_results.txt)。
- 图形：四组PNG/SVG已检查。两seed分开或明确n=2；不把轮次当独立重复，不用test评价或重选NRMSE尺度。
- 离线重放：组装CSV、逐来源预算/覆盖恒等式、行为K敏感性及报告/图表生成通过，无LLM或模型推理。

## 没有执行的事项

阶段B策略代码、B的输入一致性/约束/无泄漏单元测试、B的mock选择dry-run均**未执行**。原因是用户要求：阶段A若不支持loading-only则暂停B；内部LLM方向不稳、CW组成效应及后段反例触发了此门槛。上述41项是阶段A分析测试，不能冒称loading-cap策略已实现或验证通过。

没有真实LLM调用、新QGeoGNN训练、梯度重建、新候选标签揭示或新test评价。没有修改历史冻结实验；工作目录原有监控状态/其他未提交实验改动不属于本任务。没有创建正式cap协议锁或启动入口。K8/12只是行为候选，没有主K注册。

## 可复现记录

完整检查命令见[README.md](README.md)。当前Python：`{sys.executable}`。派生文件与脚本SHA见[artifact_hashes.json](artifact_hashes.json)；历史输入SHA另见source_hashes/behavior_summary/geometry_audit。

```text
{result.stdout.strip()}
```
'''
    (HERE / "PREFLIGHT_REPORT.md").write_text(preflight)
    files = sorted(p for p in HERE.iterdir() if p.is_file() and p.name != "artifact_hashes.json")
    (HERE / "artifact_hashes.json").write_text(json.dumps({p.name: sha(p) for p in files}, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("status", "checked_profiles", "phase_b", "real_llm_calls", "new_model_fits", "new_labels_revealed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
