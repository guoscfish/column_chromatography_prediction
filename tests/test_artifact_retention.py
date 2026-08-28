from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_EXPERIMENTS = {
    "d04_conformer_selection", "d28_al_engineering", "d45_oracle_marginal_utility", "d46_oracle_utility_reliability",
    "data_audit", "e0_3b_controls", "e0_3c_loss_controls", "e0_4g_baseline", "e0_8g_transfer",
    "e1_signal_qualification", "e2_4g_active_learning", "e2_4g_compound_active_learning", "e2_4g_compound_preflight",
    "e2_compound_failure_audit", "e2_random_smoke", "e4_a2a_engineering_smoke", "e4_a2a_low_budget_formal",
    "e4_a2a_low_budget_preregistration", "e4_active_learning_suitability_diagnosis", "e4_active_transfer_preregistration",
    "e4_protocol_a_engineering_smoke", "e4_protocol_a_formal", "e4_protocol_a_headroom_audit",
    "e4_transfer_aware_acquisition_qualification", "g0_1_quantile_monotonicity", "g0_2_interval_calibration",
    "g0_3_threshold_sensitivity", "g0_4_paper_style_transfer", "g0_4_paper_style_transfer_random_init_diagnostic",
    "reproductions",
}


def tracked() -> list[str]:
    output = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return output.splitlines()


def test_no_tracked_partial_history_fit_or_runtime() -> None:
    paths = tracked()
    assert not [p for p in paths if p.startswith("experiments/") and p.endswith(".partial.csv")]
    assert not [p for p in paths if p.startswith("experiments/") and Path(p).name == "history.csv"]
    assert not [p for p in paths if p.startswith("experiments/") and Path(p).name == "fit_result.json"]
    assert not [p for p in paths if p.startswith("experiments/") and {"runtime", "progress"} & set(Path(p).parts)]


def test_all_tracked_checkpoints_are_protected() -> None:
    payload = json.loads((ROOT / "docs/PROTECTED_ARTIFACTS.json").read_text(encoding="utf-8"))
    protected = {item["path"] for item in payload["artifacts"]}
    checkpoints = {p for p in tracked() if Path(p).suffix.lower() in {".pt", ".pth", ".ckpt"}}
    assert checkpoints <= protected


def test_new_experiments_have_compact_record() -> None:
    required = {"README.md", "config.json", "environment.json", "decision.json"}
    for directory in (ROOT / "experiments").iterdir():
        if not directory.is_dir() or directory.name in LEGACY_EXPERIMENTS:
            continue
        assert required <= {path.name for path in directory.iterdir()}, directory


def test_no_identical_final_and_partial_pair() -> None:
    for partial in (ROOT / "experiments").rglob("*.partial.csv"):
        final = partial.with_name(partial.name.removesuffix(".partial.csv") + ".csv")
        assert not final.exists() or partial.read_bytes() != final.read_bytes()
