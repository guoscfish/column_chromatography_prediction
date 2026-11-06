"""Seal or verify the delivered snapshot; ignore regenerable local array caches."""
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def files():
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        relative=path.relative_to(HERE)
        if str(relative)=="artifact_manifest.json" or path.suffix in {".log",".tmp"}:
            continue
        if relative.parts[0]=="cache" and path.suffix==".npz":
            continue
        yield path,relative


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    current={str(relative):{"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size}
             for path,relative in files()}
    manifest=HERE/"artifact_manifest.json"
    if args.check:
        saved=json.loads(manifest.read_text())["files"]
        if saved!=current:raise ValueError("delivered artifact snapshot has changed")
        print("Verified",len(current),"delivered artifacts")
    else:
        result={"schema":"representation-gp-delivery-v1","scope":"actual research outputs and code; local NPZ hashes in cache receipts; logs excluded",
                "files":current}
        manifest.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
        print("Sealed",len(current),"delivered artifacts")


if __name__=="__main__":main()
