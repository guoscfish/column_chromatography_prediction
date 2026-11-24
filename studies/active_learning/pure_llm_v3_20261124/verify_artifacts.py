"""Verify the standalone study's generated reports, code, examples and tests."""
import argparse
import hashlib
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parent
MANIFEST = STUDY / "artifact_manifest.json"
EXCLUDED_DIRS = {"__pycache__", "runtime", ".pytest_cache"}


def inventory():
    files = {}
    for path in sorted(STUDY.rglob("*")):
        relative = path.relative_to(STUDY)
        if not path.is_file() or path == MANIFEST or EXCLUDED_DIRS.intersection(relative.parts):
            continue
        with path.open("rb") as stream:
            files[relative.as_posix()] = {"sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                                         "bytes": path.stat().st_size}
    return {"study": STUDY.name, "historical_head": "b6b03b9eb62ebc75b7e60a10fbe078492d7efd92",
            "preexisting_files_modified": [], "new_real_api_requests": 0, "new_label_reveals": 0,
            "formal_training_runs": 0, "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate after intentional study edits")
    args = parser.parse_args()
    actual = inventory()
    if args.write:
        MANIFEST.write_text(json.dumps(actual, indent=2, ensure_ascii=False) + "\n")
    else:
        expected = json.loads(MANIFEST.read_bytes())
        if actual != expected:
            keys = set(actual["files"]) | set(expected["files"])
            drift = sorted(key for key in keys if actual["files"].get(key) != expected["files"].get(key))
            raise SystemExit(f"manifest mismatch: {drift or 'study metadata'}")
    print(f"Verified {len(actual['files'])} files for {STUDY.name}")


if __name__ == "__main__":
    main()
