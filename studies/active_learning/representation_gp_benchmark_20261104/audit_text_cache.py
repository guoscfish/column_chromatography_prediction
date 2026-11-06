"""Check identical cached SMILES texts with explicit feature-only source access."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent


def main():
    frame=pd.read_csv(HERE.parents[2]/"experiments/e0_4g_baseline/canonical_4g.csv",usecols=["canonical_smiles"])
    with np.load(HERE/"cache/llm.npz",allow_pickle=False) as cache:
        indices=cache["indices"];values=cache["llm_smiles"]
        first={};differences=[]
        for i,index in enumerate(indices):
            text=str(frame.iloc[int(index),0])
            if text in first:
                differences.append(float(np.max(np.abs(values[i]-values[first[text]]))))
            else:first[text]=i
    result={"scope":"identical SMILES-only texts in immutable preflight/full encoding chunks; no response labels",
        "encoded_experiment_rows":len(indices),"unique_smiles":len(first),"duplicate_text_rows":len(differences),
        "maximum_absolute_coordinate_difference":float(max(differences,default=0)),
        "median_absolute_coordinate_difference":float(np.median(differences)) if differences else 0,
        "bitwise_equal_fraction":float(np.mean(np.array(differences)==0)) if differences else 1.,
        "interpretation":"measure actual cached equality without modifying embeddings; stored ID vectors remain immutable"}
    (HERE/"llm_text_cache_consistency.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
