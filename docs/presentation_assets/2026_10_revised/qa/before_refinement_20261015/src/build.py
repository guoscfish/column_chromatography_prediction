"""Rebuild all scientific figures entirely from the frozen metric tables."""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from style import *
from draw_methods import METHODS
from draw_results import RESULTS

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--audit-sources",action="store_true",help="Read and re-audit original repository metric sources")
    parser.add_argument("--pdf-python",help="Python executable with pypdf, if not installed in this environment")
    args=parser.parse_args()
    if args.audit_sources:
        from prepare_data import main as prepare
        prepare()
    # Always check the frozen sources before using the saved derived tables.
    source_manifest=json.loads((OUT/"data/source_manifest.json").read_text(encoding="utf-8"))
    for item in source_manifest:
        digest=hashlib.sha256((OUT/item["frozen_copy"]).read_bytes()).hexdigest()
        if digest!=item["sha256"]:
            raise ValueError("Modified frozen input: "+item["frozen_copy"])
    for item in json.loads((QA/"derived_data_manifest.json").read_text()):
        if hashlib.sha256((OUT/item["file"]).read_bytes()).hexdigest()!=item["sha256"]:
            raise ValueError("Modified plotting table: "+item["file"])
    for draw in METHODS.values():
        draw()
    for draw in RESULTS.values():
        draw()
    write_layout_audit()
    issues=[a for a in AUDIT if a["layout_issues"]]
    if issues:
        raise ValueError("Resolve layout issues before packaging")
    names=[a["figure"] for a in AUDIT]
    contact_sheet(names[:3],FIGURES/"preview_F01_F03.png",cols=1,thumb=(1600,900))
    contact_sheet(names[3:9],FIGURES/"preview_F04_F09.png",cols=2)
    contact_sheet(names[9:],FIGURES/"preview_R01_R04.png",cols=2)
    contact_sheet(names,FIGURES/"preview_all.png",cols=3)
    pdf_python=args.pdf_python
    if not pdf_python:
        if importlib.util.find_spec("pypdf"):
            pdf_python=sys.executable
        else:
            bundled=Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
            if bundled.is_file():
                pdf_python=str(bundled)
    if not pdf_python:
        raise RuntimeError("Install pypdf or pass --pdf-python to create the review catalog")
    subprocess.run([pdf_python,str(OUT/"src/merge_pdfs.py"),str(FIGURES)],check=True)
    artifacts=[]
    for path in sorted(FIGURES.iterdir()):
        if path.is_file():
            artifacts.append({"file":str(path.relative_to(OUT)),"bytes":path.stat().st_size,
                              "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    (QA/"output_manifest.json").write_text(json.dumps(artifacts,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("Built 13 editable SVGs, 14 PDFs, 13 PNGs at 4800x2700 / 300dpi, and four preview sheets")

if __name__=="__main__":
    main()
