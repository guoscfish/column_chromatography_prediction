#!/usr/bin/env python3
"""Export immutable retry evidence without API calls, training, or truth reads."""
import argparse, collections, gzip, hashlib, json, re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies/active_learning/free_cw_sol61_xhigh_five_to1485_20261101"
SEEDS = (157, 887, 2357, 6101, 12203)
SECRET = re.compile(rb"(?:sk-[A-Za-z0-9_-]{24,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)")
def export(out):
    out.mkdir(parents=True, exist_ok=False)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    manifest = {"captured_shanghai": stamp, "study": str(STUDY.relative_to(ROOT)),
                "source_files_deleted": False, "api_calls": 0, "training_calls": 0,
                "archive_format": "gzip JSONL: path, sha256, raw_utf8 (original bytes recoverable)", "seeds": {}}
    for seed in SEEDS:
        base = STUDY / f"selections/seed_{seed}/free_cw_sol61_xhigh_rerun32"
        paths = sorted(p for p in base.glob("round_*/*.json")
                       if p.name.startswith(("attempt_", "stream_", "turn_", "rejected_")))
        counts, errors, rejects, rounds = collections.Counter(), collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
        archive = out / f"seed_{seed}_receipts.jsonl.gz"
        with gzip.open(archive, "wt", encoding="utf-8", compresslevel=6) as dst:
            for p in paths:
                raw = p.read_bytes()
                if SECRET.search(raw):
                    raise RuntimeError(f"Credential-like value requires review: {p.relative_to(ROOT)}")
                try: d = json.loads(raw)
                except json.JSONDecodeError: continue  # live atomic write not yet complete
                dst.write(json.dumps({"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest(), "raw_utf8": raw.decode("utf-8")}, ensure_ascii=False) + "\n")
                kind = p.name.split("_")[0]
                if p.name.endswith(".started.json"): kind = "started"
                counts[kind] += 1
                if kind == "attempt":
                    result = "success" if d.get("success") else f"HTTP {d.get('http_status')} | {d.get('error_type')} | {d.get('error', d.get('reason', ''))}"
                    errors[result] += 1
                    rounds[p.parent.name][result] += 1
                if kind == "rejected": rejects[str(d.get("error", d.get("reason", "unknown")))] += 1
        manifest["seeds"][str(seed)] = {"archive": archive.name, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "bytes": archive.stat().st_size,
            "file_counts": dict(counts), "attempt_results": dict(errors), "logical_rejections": dict(rejects),
            "per_round_attempt_results": {k: dict(v) for k,v in rounds.items()}}
        print(seed, dict(counts), archive.stat().st_size, flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    lines = ["# CW Free32 retry evidence snapshot", "", f"Captured: {stamp}", "", "Each gzip JSONL row contains the original UTF-8 receipt and its SHA256. Source files remain in place for the running controller. Full request/input payloads and training arrays remain local; this export does not read labels or predictions.", "", "HTTP 429 does not identify whether the provider is enforcing quota, capacity, or another rate limit; response bodies were not preserved in those receipts. A successful transport answer is not an accepted batch. See logical_rejections separately.", "", "| Seed | Started | Terminal attempts | Rejected logical answers |", "|---|---:|---:|---:|"]
    for seed, item in manifest["seeds"].items():
        c = item["file_counts"]
        lines.append(f"| {seed} | {c.get('started',0)} | {c.get('attempt',0)} | {c.get('rejected',0)} |")
    (out / "README.md").write_text("\n".join(lines) + "\n")
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True)
    export(ap.parse_args().output.resolve())
