#!/usr/bin/env python3
"""Read-only snapshot of controller status and diagnostic logs; no experiment calls."""
import argparse, gzip, hashlib, json, re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101"
def export(out):
    if out.exists(): raise RuntimeError("Preserve prior snapshots; choose a new output")
    paths = sorted((STUDY / "progress").glob("*.json")) + sorted((STUDY / "progress").glob("*.log"))
    paths += [STUDY / "active_runtime.json", STUDY / "parallel_execution_amendment.json"]
    count = 0
    with gzip.open(out, "wt", encoding="utf-8") as dst:
        for path in paths:
            raw = path.read_bytes()
            if re.search(rb"sk-[A-Za-z0-9_-]{24,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----", raw):
                raise RuntimeError(f"Credential-like value in {path.relative_to(ROOT)}")
            dst.write(json.dumps({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest(), "raw_utf8": raw.decode("utf-8")}, ensure_ascii=False) + "\n")
            count += 1
    manifest = {"captured_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "files": count, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "bytes": out.stat().st_size, "sources_deleted": False}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True)
    export(ap.parse_args().output.resolve())
