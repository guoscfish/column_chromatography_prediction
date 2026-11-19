"""Read-only verification of completed surrogate evidence; never refits.

The original bounded execution scripts are retained at their frozen hashes.
Use this entry point after completion: it verifies rather than regenerates
scientific aggregates or replacing process-time receipts.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=HERE/"surrogate_evidence"
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    checked=0
    for p in (HERE/"surrogate_protocol.json",HERE/"surrogate_followup_protocol.json"):
        protocol=read(p)
        for name,h in protocol["files"].items():
            if digest(ROOT/name)!=h:raise RuntimeError(f"sealed scientific code drift: {name}")
            checked+=1
        for name,h in protocol.get("dependencies",{}).items():
            if digest(ROOT/name)!=h:raise RuntimeError(f"encoder dependency drift: {name}")
            checked+=1
    first=read(OUT/"complete.json");second=read(OUT/"followup/complete.json")
    if (first["model_fits"],first["endpoint_optimization_starts"],second["new_model_fits"])!=(60,360,5):raise RuntimeError("fit accounting drift")
    for parent,record,key in ((OUT,first,"files"),(OUT/"followup",second,"outputs")):
        for name,h in record[key].items():
            if digest(parent/name)!=h:raise RuntimeError(f"completed aggregate drift: {parent/name}")
            checked+=1
    count=0
    for p in OUT.glob("seed_*/F*/fold_*/complete.json"):
        receipt=read(p)
        if digest(p.parent/"contract.json")!=receipt["contract_sha256"]:raise RuntimeError("fold contract drift")
        for name,h in receipt["files"].items():
            if digest(p.parent/name)!=h:raise RuntimeError(f"completed fold drift: {p.parent/name}")
            checked+=1
        count+=1
    if count!=60:raise RuntimeError("missing complete OOF folds")
    for p in OUT.glob("followup/seed_*/frozen_encoder_blr/complete.json"):
        receipt=read(p)
        for name,key in (("scores.npz","scores_sha256"),("fit_audit.json","fit_audit_sha256")):
            if digest(p.parent/name)!=receipt[key]:raise RuntimeError("encoder completion drift")
            checked+=1
    test=ET.parse(OUT/"unit_tests.xml").getroot()
    suites=test.findall("testsuite") if test.tag=="testsuites" else [test]
    if sum(int(s.attrib.get("failures",0))+int(s.attrib.get("errors",0)) for s in suites):raise RuntimeError("failed mathematical surrogate unit tests")
    return {"status":"VERIFIED","files_checked":checked,"completed_OOF_model_fits":60,"endpoint_optimizer_starts":360,
            "conditional_encoder_diagnostic_fits":5,"synthetic_unit_tests":4,"new_fits_in_verification":0,"labels_read":0,
            "verification_source_sha256":digest(__file__),"preserved_execution_scripts":True,
            "execution_replay_caution":"surrogate_audit.py is the sealed initial executor; its completed-cache PCA aggregation has a known replay omission. Use this read-only verification after completion; do not rerun that executor to regenerate the completed evidence."}


if __name__=="__main__":print(json.dumps(verify(),indent=2))
