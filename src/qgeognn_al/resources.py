"""Canonical repository resources shared by current studies."""

from __future__ import annotations

import json
from pathlib import Path

from .artifacts import sha256_file

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATA = ROOT / "experiments/e0_4g_baseline/canonical_4g.csv"
TARGET_DATA = ROOT / "experiments/g0_3_threshold_sensitivity/canonical_8g_no_threshold.csv"
SOURCE_GRAPH_CACHE = ROOT / "experiments/e0_4g_baseline/graph_cache_4g.pt"
TARGET_GRAPH_CACHE = ROOT / "experiments/e0_8g_transfer/graph_cache_8g_only.pt"
SOURCE_SCALER = ROOT / "experiments/e0_4g_baseline/scaler.json"
SOURCE_COMPATIBILITY_AUDIT = ROOT / "experiments/e4_active_transfer_preregistration/source_compatibility_audit.json"
PROTECTED_ARTIFACTS = ROOT / "docs/PROTECTED_ARTIFACTS.json"
SOURCE_CHECKPOINTS = {
    42: ROOT / "experiments/e0_4g_baseline/checkpoints/best.pt",
    525: ROOT / "experiments/e1_signal_qualification/source_members/member_seed_525/checkpoints/best.pt",
    1101: ROOT / "experiments/e1_signal_qualification/source_members/member_seed_1101/checkpoints/best.pt",
}


def verified_source_checkpoints() -> list[dict[str, str | int]]:
    """Verify source paths against both protection policy and frozen E4 audit."""
    protected = {item["path"] for item in json.loads(PROTECTED_ARTIFACTS.read_text())["artifacts"]}
    audit = json.loads(SOURCE_COMPATIBILITY_AUDIT.read_text())
    expected = {int(item["source_seed"]): item for item in audit["source_members"]}
    records = []
    for seed, path in SOURCE_CHECKPOINTS.items():
        relative = str(path.relative_to(ROOT))
        digest = sha256_file(path)
        if relative not in protected:
            raise RuntimeError(f"source checkpoint is not protected: {relative}")
        if expected[seed]["checkpoint"] != relative or expected[seed]["sha256"] != digest:
            raise RuntimeError(f"source checkpoint audit mismatch: seed {seed}")
        records.append({"source_seed": seed, "checkpoint": relative, "sha256": digest})
    return records
