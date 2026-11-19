#!/usr/bin/env python3
"""Independently reconcile the two covariate-only pair permission tables."""
from pathlib import Path
import json
import pandas as pd
from data_boundary import sha256_file

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def main():
    al_path, formal_path = OUT / "al_common_pair_exposure.csv", OUT / "formal_pairs.csv"
    al = pd.read_csv(al_path, dtype={"low_id": str, "high_id": str})
    formal = pd.read_csv(formal_path, dtype={"low_id": str, "high_id": str})
    lookup = {(int(row.row_seed), row.axis, frozenset((row.low_id, row.high_id))): row
              for row in formal.itertuples()}
    for row in al.itertuples():
        key = int(row.formal_row_seed), row.axis, frozenset((row.low_id, row.high_id))
        reference = lookup[key]
        if bool(row.allow_new_scoring) != bool(reference.scoring_allowed):
            raise ValueError("AL/formal pair scoring permission disagreement")
        if row.low_source_role != reference.low_source_role or row.high_source_role != reference.high_source_role:
            raise ValueError("AL/formal pair endpoint-role disagreement")
    record = dict(status="PASS", checked_AL_formal_pair_rows=len(al), identity_disagreements=0,
        role_disagreements=0, scoring_eligibility_disagreements=0, AL_allowed_rows=int(al.allow_new_scoring.sum()),
        inputs=[dict(path=str(path.relative_to(ROOT)), sha256=sha256_file(path), rows=len(frame))
                for path, frame in ((al_path, al), (formal_path, formal))],
        verifier_script_sha256=sha256_file(Path(__file__)), label_values_read_by_crosscheck=False)
    (OUT / "formal_AL_pair_permission_crosscheck.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: value for key, value in record.items() if key != "inputs"}))


if __name__ == "__main__":
    main()
