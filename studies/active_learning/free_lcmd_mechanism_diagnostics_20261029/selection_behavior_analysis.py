"""Read frozen selections only; never import a runner, train, or request labels.

Run with the repository chromatography Python (RDKit is required). The only
response values read are initial/previously measured Free catalog records and
already frozen selected-batch feedback. LCMD residuals remain NA because its
receipts store hashes rather than response payloads.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SEEDS = (157, 887, 2357, 6101, 12203)
FREE = "free_llm32_scientist_v2"
CONDITIONS = ("PE/EA", "Density g/ml", "V/ul", "loading solvent", "Volume of loading solvent/ul")
STAGES = ((333, 525), (525, 653), (653, 813), (813, 1005), (1005, 1197), (1197, 1485))
TEXT_PATTERNS = {
    "cw": r"\bcw\b|center[ /_-]width|cw_distance|distance percentile",
    "quantile_width": r"quantile|q10|q90|p10|p90|prediction interval|uncertainty width",
    "similarity": r"similar|analog|analogue|tanimoto|fingerprint|structural neighbo",
    "chemistry": r"phenol|alcohol|amine|amide|ester|ketone|aromatic|hetero|hydrogen.bond|polarity|polar |halogen|carbonyl|nitrile|carboxy|nitro",
    "observed_feedback": r"measured|observed|residual|overpredict|underpredict|previous outcome|last batch|recent outcome|error",
    "local_condition_curve": r"same.molecule|condition.response|response.curve|interpolat|curvature|bracket|midpoint|dose.response|amount.response|loading.response|solvent.response|matched.*contrast",
    "batch_diversity": r"divers|redundan|duplicate|coverage|broad.*chemical|spread|complement",
    "failure_repair": r"overpredict|underpredict|residual|blind[ -]spot|systematic[^.]{0,50}error|(?:model|predictor)[^.]{0,50}fail|prediction[^.]{0,50}miss",
}
FAMILY_SMARTS = {
    "phenol": "[OX2H]-[c]", "alcohol": "[OX2H]-[CX4]",
    "ester": "[CX3](=O)[OX2H0][#6]", "amide": "[CX3](=O)[NX3]",
    "ketone": "[#6][CX3](=O)[#6]", "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "amine": "[NX3;!$(N-C=O)]", "nitrile": "[CX2]#N",
    "nitro": "[N+](=O)[O-]", "halogen": "[F,Cl,Br,I]",
}
SOURCES: dict[str, str] = {}


def sha(path):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    SOURCES[str(path.relative_to(ROOT))] = digest
    return digest


def read_json(path):
    sha(path)
    return json.loads(Path(path).read_text())


def read_csv(path):
    sha(path)
    return pd.read_csv(path)


def rel(path):
    return str(Path(path).relative_to(ROOT))


def condition_key(row, except_field=None):
    return tuple(row["conditions"][f] for f in CONDITIONS if f != except_field)


def ea_fraction(value):
    pe, ea = map(float, str(value).split("/"))
    return ea / (pe + ea)


def stage_name(budget):
    return next(f"{a}-{b}" for a, b in STAGES if a <= budget < b)


class Chemistry:
    def __init__(self, meta):
        self.smiles = sorted({r["smiles"] for r in meta.values()})
        self.pos = {s: i for i, s in enumerate(self.smiles)}
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        mols = [Chem.MolFromSmiles(s) for s in self.smiles]
        assert all(m is not None for m in mols), "Invalid frozen SMILES"
        fps = [generator.GetFingerprint(m) for m in mols]
        self.sim = np.array([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps])
        self.scaffolds = {}
        self.families = {}
        queries = {k: Chem.MolFromSmarts(v) for k, v in FAMILY_SMARTS.items()}
        for s, m in zip(self.smiles, mols):
            core = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
            # Empty Murcko cores do not make all acyclic molecules one family.
            self.scaffolds[s] = core or "acyclic:" + s
            self.families[s] = {k for k, q in queries.items() if m.HasSubstructMatch(q)}

    def positions(self, ids, meta):
        return np.array([self.pos[meta[i]["smiles"]] for i in ids], dtype=int)


def local_flags(row, by_smiles):
    """Only compare with records already observed before this batch."""
    previous = by_smiles.get(row["smiles"], [])
    out = {"known_molecule": bool(previous), "new_condition_for_known_molecule": False,
           "exact_condition_repeat": False}
    out["new_condition_for_known_molecule"] = bool(previous) and not any(condition_key(p) == condition_key(row) for p in previous)
    out["exact_condition_repeat"] = any(condition_key(p) == condition_key(row) for p in previous)
    for short, field in (("pe", "PE/EA"), ("amount", "V/ul"), ("loading_volume", "Volume of loading solvent/ul")):
        matched = [p for p in previous if condition_key(p, field) == condition_key(row, field)]
        convert = ea_fraction if field == "PE/EA" else float
        target = convert(row["conditions"][field])
        values = [convert(p["conditions"][field]) for p in matched]
        contrast = any(v != target for v in values)
        bracket = any(v < target for v in values) and any(v > target for v in values)
        out[short + "_matched_contrast"] = contrast
        # A new row ID can repeat an existing molecular-condition tuple. Such
        # repeats are contrasts/replicates, but cannot complete a missing point.
        out[short + "_curve_extension"] = contrast and not out["exact_condition_repeat"]
        out[short + "_interpolation"] = bracket and not out["exact_condition_repeat"]
    return out


def high_error_features(selected, pool, observed, meta, chemistry):
    # These are historical acquisition-time errors, not current-model residuals.
    errors = [r for r in observed if "premeasurement_error_V1_ml" in r]
    result = {"history_error_available_count": len(errors), "history_high8_count": 0}
    names = ("high8_same_molecule_fraction", "high8_same_molecule_pool_fraction", "high8_same_molecule_enrichment",
             "high8_max_morgan_similarity_mean", "high8_max_morgan_similarity_pool_mean", "high8_similarity_enrichment",
             "high8_condition_nearest_mean", "high8_same_molecule_condition_nearest_mean",
             "recent32_same_molecule_fraction", "history_high_error_threshold_ml")
    result.update({n: np.nan for n in names})
    if not errors:
        return result
    errors.sort(key=lambda r: (-max(abs(r["premeasurement_error_V1_ml"]), abs(r["premeasurement_error_V2_ml"])), r["id"]))
    high = errors[:8]
    high_ids = [r["id"] for r in high]
    high_sm = {r["smiles"] for r in high}
    sel_sm = [meta[i]["smiles"] for i in selected]
    pool_sm = [meta[i]["smiles"] for i in pool]
    hs = chemistry.positions(high_ids, meta)
    ss, ps = chemistry.positions(selected, meta), chemistry.positions(pool, meta)
    same, base = np.mean([s in high_sm for s in sel_sm]), np.mean([s in high_sm for s in pool_sm])
    ssim, psim = chemistry.sim[np.ix_(ss, hs)].max(1).mean(), chemistry.sim[np.ix_(ps, hs)].max(1).mean()
    def vector(row):
        c = row["conditions"]
        return np.array([ea_fraction(c["PE/EA"]), float(c["V/ul"])/150., float(c["Volume of loading solvent/ul"])/1000.])
    # Fixed physical scaling, three recorded conditions; loading solvent mismatch
    # adds a unit penalty. Density is excluded because mostly molecule-specific.
    distances, same_distances = [], []
    for i in selected:
        row = meta[i]
        d = [math.sqrt(float(np.square(vector(row)-vector(h)).sum()) + (row["conditions"]["loading solvent"] != h["conditions"]["loading solvent"])) for h in high]
        distances.append(min(d))
        matches = [v for v, h in zip(d, high) if row["smiles"] == h["smiles"]]
        if matches:
            same_distances.append(min(matches))
    result.update(history_high8_count=len(high), high8_same_molecule_fraction=same,
                  high8_same_molecule_pool_fraction=base, high8_same_molecule_enrichment=same-base,
                  high8_max_morgan_similarity_mean=ssim, high8_max_morgan_similarity_pool_mean=psim,
                  high8_similarity_enrichment=ssim-psim, high8_condition_nearest_mean=np.mean(distances),
                  high8_same_molecule_condition_nearest_mean=np.mean(same_distances) if same_distances else np.nan,
                  history_high_error_threshold_ml=min(max(abs(r["premeasurement_error_V1_ml"]),abs(r["premeasurement_error_V2_ml"])) for r in high))
    return result


def profile(selected, pool, observed_ids, meta, chemistry):
    by_smiles = collections.defaultdict(list)
    for i in observed_ids:
        by_smiles[meta[i]["smiles"]].append(meta[i])
    selected_rows = [meta[i] for i in selected]
    flags = [local_flags(r, by_smiles) for r in selected_rows]
    pool_flags = [local_flags(meta[i], by_smiles) for i in pool]
    sm = [r["smiles"] for r in selected_rows]
    counts = collections.Counter(sm)
    out = {"batch_size": len(selected), "observed_count": len(observed_ids), "candidate_pool_count": len(pool),
           "unique_molecules": len(counts), "new_molecule_count": len(set(sm)-set(by_smiles)),
           "same_molecule_pairs": sum(n*(n-1)//2 for n in counts.values()),
           "max_rows_per_molecule": max(counts.values()),
           "distinct_molecule_condition_tuples": len({(r["smiles"], condition_key(r)) for r in selected_rows})}
    for name in flags[0]:
        out[name + "_fraction"] = np.mean([r[name] for r in flags])
        out[name + "_pool_fraction"] = np.mean([r[name] for r in pool_flags])
        out[name + "_enrichment"] = out[name + "_fraction"] - out[name + "_pool_fraction"]
    out["new_molecule_row_fraction"] = 1-out["known_molecule_fraction"]
    out["within_batch_exact_condition_repeat_pairs"] = sum(n*(n-1)//2 for n in collections.Counter((r["smiles"],condition_key(r)) for r in selected_rows).values())
    out["within_batch_distinct_condition_same_molecule_pairs"] = out["same_molecule_pairs"] - out["within_batch_exact_condition_repeat_pairs"]
    for short, field in (("ea_fraction", "PE/EA"), ("amount_ul", "V/ul"), ("loading_volume_ul", "Volume of loading solvent/ul")):
        vals = np.array([ea_fraction(r["conditions"][field]) if field == "PE/EA" else float(r["conditions"][field]) for r in selected_rows])
        out.update({short+"_mean": vals.mean(), short+"_span": np.ptp(vals), short+"_unique_count":len(set(vals))})
    out["loading_mass_mg_mean"] = np.mean([float(r["conditions"]["Density g/ml"])*float(r["conditions"]["V/ul"]) for r in selected_rows])
    ss = chemistry.positions(selected, meta)
    sims = chemistry.sim[np.ix_(ss, ss)]
    out["batch_morgan_mean_similarity"] = sims[np.triu_indices(len(ss),1)].mean()
    nearest = sims.copy()
    np.fill_diagonal(nearest,-1)
    out["batch_morgan_nearest_similarity_mean"] = nearest.max(1).mean()
    unique = np.array([chemistry.pos[s] for s in sorted(set(sm))])
    out["unique_molecule_morgan_mean_similarity"] = chemistry.sim[np.ix_(unique,unique)][np.triu_indices(len(unique),1)].mean() if len(unique)>1 else np.nan
    obs = chemistry.positions(observed_ids, meta)
    out["nearest_observed_morgan_similarity_mean"] = chemistry.sim[np.ix_(ss,obs)].max(1).mean()
    scaffolds = collections.Counter(chemistry.scaffolds[s] for s in sm)
    out["unique_murcko_scaffolds"] = len(scaffolds)
    out["dominant_scaffold_fraction"] = max(scaffolds.values())/len(sm)
    out["same_scaffold_pair_fraction"] = sum(n*(n-1)//2 for n in scaffolds.values())/(len(sm)*(len(sm)-1)/2)
    for family in FAMILY_SMARTS:
        out["family_"+family+"_fraction"] = np.mean([family in chemistry.families[s] for s in sm])
    return out, flags


def narrative_text(value):
    """Narrative string values only: never schema keys or ID/reference fields."""
    if isinstance(value,str):
        return value
    if isinstance(value,list):
        return " ".join(narrative_text(v) for v in value)
    if isinstance(value,dict):
        return " ".join(narrative_text(v) for k,v in value.items()
                        if k != "id" and not k.endswith(("_id","_ids","_sha256")))
    return ""


def text_features(selection):
    # A mention is not an endorsement or evidence of causal dependence.
    text = narrative_text({k: selection.get(k) for k in ("choices","hypotheses","batch_strategy","batch_rationale","feedback_interpretation")})
    return {"text_"+k+"_mention": bool(re.search(v,text,re.I)) for k,v in TEXT_PATTERNS.items()}


def grouped_summary(df, cols):
    return df.groupby(["seed","method","stage"],as_index=False)[cols].mean()


def correlation_rows(frame, features, target, group_cols):
    rows = []
    groups = frame.groupby(group_cols) if group_cols else [((),frame)]
    for key, data in groups:
        key = key if isinstance(key,tuple) else (key,)
        for f in features:
            valid = data[[f,target]].dropna()
            rho = valid[f].rank().corr(valid[target].rank()) if len(valid)>2 and valid[f].nunique()>1 and valid[target].nunique()>1 else np.nan
            rows.append({**dict(zip(group_cols,key)),"feature":f,"target":target,"n_batches":len(valid),"spearman_rho":rho,
                         "inference":"within-trajectory descriptive; rounds not independent; no causal or p-value claim"})
    return rows


def adjusted_correlations(frame, features, target, group_cols, controls):
    """Residual correlation of ranks, descriptive sensitivity check only."""
    rows=[]
    for key,data in frame.groupby(group_cols):
        key=key if isinstance(key,tuple) else (key,)
        for feature in features:
            valid=data[[feature,target,*controls]].dropna()
            rho=np.nan
            if len(valid)>len(controls)+3 and valid[feature].nunique()>1:
                ranks=valid.rank()
                x=np.column_stack([np.ones(len(valid)),ranks[controls].to_numpy()])
                a=ranks[feature].to_numpy();b=ranks[target].to_numpy()
                ar=a-x@np.linalg.lstsq(x,a,rcond=None)[0]
                br=b-x@np.linalg.lstsq(x,b,rcond=None)[0]
                if np.std(ar)>1e-10 and np.std(br)>1e-10:
                    rho=np.corrcoef(ar,br)[0,1]
            rows.append({**dict(zip(group_cols,key)),"feature":feature,"target":target,
                         "controls":";".join(controls),"n_batches":len(valid),"partial_rank_correlation":rho,
                         "inference":"descriptive sensitivity to measured state; not causal, no independent-round inference"})
    return rows


def main():
    spec = importlib.util.spec_from_file_location("frozen_owner",ROOT/"scripts/report_free_lcmd.py")
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    sha(ROOT/"scripts/report_free_lcmd.py")
    for source in (Path(__file__), ROOT/"src/qgeognn_al/active_learning_v2/scientist_study.py",
                   ROOT/"src/qgeognn_al/active_learning_v2/scientist_selector.py",
                   ROOT/"src/qgeognn_al/active_learning_v2/scientist_full_delivery.py", ROOT/"src/qgeognn_al/schemas/conditions.py"):
        sha(source)
    curves = read_json(ROOT/"reports/free_llm_five_seeds_to1485_20261024/figures/validation_curves.json")
    meta = {}
    initial_ids = {}
    for seed in SEEDS:
        p=reader.owner(seed,1)/f"selections/seed_{seed}/{FREE}/round_00/catalog.json"
        raw=read_json(p)
        for row in raw["candidates"]+raw["pending"]+raw["observed"]:
            clean={k:row[k] for k in ("id","smiles","conditions")}
            assert row["id"] not in meta or meta[row["id"]] == clean
            meta[row["id"]]=clean
        initial_ids[seed]=[r["id"] for r in raw["observed"]]
        assert len(initial_ids[seed]) == 333
    chemistry=Chemistry(meta)
    profiles, records, text_rows, text_field_rows, case_studies=[],[],[],[],[]
    for seed in SEEDS:
        for method in (FREE,"lcmd"):
            labeled=list(initial_ids[seed])
            acquired=[]
            for r in range(36):
                budget=333+32*r
                base={"seed":seed,"method":method,"round":r,"labels_before":budget,"labels_after":budget+32,"stage":stage_name(budget)}
                if method == FREE:
                    directory=reader.owner(seed,r+1)/f"selections/seed_{seed}/{FREE}/round_{r:02d}"
                    frozen=read_json(directory/"batch_freeze.json")
                    selection=read_json(directory/"selection.json")
                    raw=read_json(directory/"catalog.json")
                    selected=frozen["batch_ids"]
                    assert selected == [c["id"] for c in selection["choices"]]
                    candidates={row["id"]:row for row in raw["candidates"]+raw["pending"]}
                    pool=list(candidates)
                    observed=raw["observed"]
                    assert set(labeled) == {row["id"] for row in observed}
                    for row in list(candidates.values())+observed:
                        assert {k:row[k] for k in ("id","smiles","conditions")} == meta[row["id"]]
                    feedback=read_json(directory/"feedback.json")
                    assert [row["candidate_id"] for row in feedback["records"]] == selected
                    assert feedback["round"] == r
                    turn_files=sorted(directory.glob("turn_*.json"))
                    served=[]
                    for p in turn_files:
                        turn=read_json(p)
                        answer=turn.get("answer")
                        try: answer=json.loads(answer) if isinstance(answer,str) else answer
                        except (TypeError,json.JSONDecodeError): continue
                        if isinstance(answer,dict) and answer.get("choices") == selection["choices"]:
                            served.append(turn.get("provenance",{}).get("served_model","unknown"))
                    base.update(selection_source=rel(directory/"batch_freeze.json"),catalog_source=rel(directory/"catalog.json"),
                                selection_sha256=sha(directory/"batch_freeze.json"),feedback_source=rel(directory/"feedback.json"),
                                selector_revision=directory.parents[3].name,served_model=";".join(sorted(set(served))) or "unknown")
                else:
                    directory=(reader.BASE if r<21 else reader.LCMD_EXTENSION)/f"runtime/seed_{seed}/lcmd/round_{r:02d}"
                    p=directory/"acquisition_artifacts/selected_next_batch.csv"
                    table=read_csv(p)
                    selected=table.sample_id.tolist()
                    # U0 from the matching seed's legal Free round-0 catalog;
                    # metadata only, identical split confirmed by integrity audit.
                    p0=reader.owner(seed,1)/f"selections/seed_{seed}/{FREE}/round_00/catalog.json"
                    raw0=read_json(p0)
                    pool0=[x["id"] for x in raw0["candidates"]+raw0["pending"]]
                    pool=[i for i in pool0 if i not in set(acquired)]
                    observed=[]
                    base.update(selection_source=rel(p),catalog_source=rel(p0),selection_sha256=sha(p),feedback_source="",
                                selector_revision=directory.parents[3].name,served_model="not applicable")
                    candidates={}
                assert len(selected) == len(set(selected)) == 32
                assert set(selected) <= set(pool) and not set(selected)&set(labeled)
                out,flags=profile(selected,pool,labeled,meta,chemistry)
                out.update(high_error_features(selected,pool,observed,meta,chemistry))
                out["history_error_status"]="own pre-selection observed acquisition-time errors" if method==FREE and r>0 else ("no acquired history at round 0" if r==0 else "NA: LCMD receipts contain hashes, no frozen residual payload; no truth reconstructed")
                if method==FREE:
                    tf=text_features(selection)
                    out.update(tf)
                    texts={**base,**tf,"hypothesis_count":len(selection.get("hypotheses",[])),"text_character_count":len(json.dumps(selection)),
                           "hypotheses":json.dumps(selection.get("hypotheses",[]),ensure_ascii=False),
                           "batch_rationale":selection.get("batch_rationale",""),"feedback_interpretation":selection.get("feedback_interpretation","")}
                    text_rows.append(texts)
                    for field in ("choices","hypotheses","batch_strategy","batch_rationale","feedback_interpretation"):
                        field_text=narrative_text(selection.get(field,""))
                        text_field_rows.append({**base,"text_field":field,
                                                **{"text_"+k+"_mention":bool(re.search(v,field_text,re.I)) for k,v in TEXT_PATTERNS.items()}})
                    out["selected_cw_percentile_mean"]=np.mean([candidates[i]["cw_distance_percentile"] for i in selected])
                    out["selected_cw_percentile_above_90_fraction"]=np.mean([candidates[i]["cw_distance_percentile"]>=.9 for i in selected])
                    for t in ("V1","V2"):
                        widths=lambda ids: np.array([candidates[i][f"pred_{t}_q90_ml"]-candidates[i][f"pred_{t}_q10_ml"] for i in ids])
                        ws=widths(selected); wp=widths(pool)
                        out[f"selected_{t}_quantile_width_mean_ml"]=ws.mean()
                        out[f"selected_{t}_quantile_width_pool_percentile_mean"]=np.mean([((wp<w).sum()+.5*(wp==w).sum())/len(wp) for w in ws])
                    recent_ids=[i for i in acquired[-32:]]
                    recent_sm={meta[i]["smiles"] for i in recent_ids}
                    out["recent32_same_molecule_fraction"]=np.mean([meta[i]["smiles"] in recent_sm for i in selected]) if r else np.nan
                    recent=[x for x in observed if x["id"] in set(recent_ids)]
                    recent_features=high_error_features(selected,pool,recent,meta,chemistry)
                    out.update({"recent_"+k:v for k,v in recent_features.items() if k.startswith("high8_")})
                    # Deterministically choose three concrete, modest audit cases.
                    if (seed,r) in ((887,0),(2357,10),(887,35)):
                        chosen_h=selection.get("hypotheses",[])[0]
                        choices=[x for x in selection["choices"] if x.get("hypothesis_id")==chosen_h["id"]][:3]
                        frecs={x["candidate_id"]:x for x in feedback["records"]}
                        case={**base,"hypothesis":chosen_h,"choices":[{**x,"metadata":meta[x["id"]],"revealed_feedback":frecs[x["id"]]} for x in choices],
                              "prior_measurements_for_case_molecules":[x for x in observed if x["smiles"] in {meta[c["id"]]["smiles"] for c in choices}],
                              "evidence_boundary":"Feedback values revealed after freeze; cases are audits, not prospective scientific hypothesis validation."}
                        if r<35:
                            nextdir=reader.owner(seed,r+2)/f"selections/seed_{seed}/{FREE}/round_{r+1:02d}"
                            following=read_json(nextdir/"selection.json")
                            case["following_selection_source"]=rel(nextdir/"selection.json")
                            case["following_feedback_interpretation"]=following.get("feedback_interpretation","")
                            case["following_choices_same_molecules"]=[c for c in following["choices"] if meta[c["id"]]["smiles"] in {meta[x["id"]]["smiles"] for x in choices}]
                        case_studies.append(case)
                else:
                    for name in ("selected_cw_percentile_mean","selected_cw_percentile_above_90_fraction"):
                        out[name]=np.nan
                ys=curves[str(seed)][method]
                assert ys[r]["labels"] == budget
                out["validation_nrmse_before"]=ys[r]["nrmse"]
                out["validation_nrmse_after"]=ys[r+1]["nrmse"]
                out["next_fit_validation_improvement"]=ys[r]["nrmse"]-ys[r+1]["nrmse"]
                profiles.append({**base,**out})
                for order,(i,fl) in enumerate(zip(selected,flags)):
                    row=meta[i]
                    record={**base,"selection_order":order,"sample_id":i,"smiles":row["smiles"],**row["conditions"],**fl,
                            "murcko_scaffold":chemistry.scaffolds[row["smiles"]],"chemical_families":";".join(sorted(chemistry.families[row["smiles"]]))}
                    if method==FREE:
                        record.update({k:v for k,v in candidates[i].items() if k not in ("id","smiles","conditions","descriptors")})
                    records.append(record)
                labeled.extend(selected)
                acquired.extend(selected)
            assert len(set(acquired))==1152 and len(labeled)==1485
    df=pd.DataFrame(profiles)
    selected_df=pd.DataFrame(records)
    assert len(df)==360 and len(selected_df)==11520
    df.to_csv(HERE/"behavior_profiles.csv",index=False)
    selected_df.to_csv(HERE/"selection_records.csv",index=False)
    pd.DataFrame(text_rows).to_csv(HERE/"llm_text_features.csv",index=False)
    pd.DataFrame(text_field_rows).to_csv(HERE/"text_field_mentions.csv",index=False)
    cols=["unique_molecules","same_molecule_pairs","known_molecule_fraction","new_molecule_row_fraction","pe_curve_extension_fraction","amount_curve_extension_fraction",
          "loading_volume_curve_extension_fraction","exact_condition_repeat_fraction","new_condition_for_known_molecule_fraction","pe_interpolation_fraction","amount_interpolation_fraction",
          "batch_morgan_mean_similarity","unique_molecule_morgan_mean_similarity","dominant_scaffold_fraction","high8_same_molecule_fraction","high8_same_molecule_enrichment",
          "high8_max_morgan_similarity_mean","high8_similarity_enrichment","recent32_same_molecule_fraction","selected_cw_percentile_mean","next_fit_validation_improvement"]
    stages=grouped_summary(df,cols)
    stages.to_csv(HERE/"behavior_stage_by_seed.csv",index=False)
    numeric=df.select_dtypes(include=[np.number,bool]).columns
    paired=[]
    for stage in [f"{a}-{b}" for a,b in STAGES]+["333-1485"]:
        sub=df if stage=="333-1485" else df[df.stage==stage]
        per=sub.groupby(["seed","method"])[list(numeric)].mean()
        for col in cols:
            free=per.xs(FREE,level="method")[col]
            lcmd=per.xs("lcmd",level="method")[col]
            delta=free-lcmd
            paired.append({"stage":stage,"metric":col,"free_mean_of_5_seed_means":free.mean(),"lcmd_mean_of_5_seed_means":lcmd.mean(),
                           "paired_difference_free_minus_lcmd":delta.mean(),"paired_difference_sd_across_seeds":delta.std(ddof=1),
                           "n_valid_seed_pairs":int(delta.notna().sum()),"free_greater_count":int((delta>0).sum())})
    pd.DataFrame(paired).to_csv(HERE/"behavior_paired_stage_comparison.csv",index=False)
    corr=correlation_rows(df,cols[:-1],"next_fit_validation_improvement",["seed","method"])
    pd.DataFrame(corr).to_csv(HERE/"behavior_improvement_correlations.csv",index=False)
    pd.DataFrame(adjusted_correlations(df,cols[:-1],"next_fit_validation_improvement",["seed","method"],
                                     ["labels_before","validation_nrmse_before"])).to_csv(HERE/"behavior_improvement_adjusted_correlations.csv",index=False)
    free=df[df.method==FREE].copy()
    textcorr=[]
    behavior_targets=[("cw","selected_cw_percentile_mean"),("quantile_width","selected_V2_quantile_width_pool_percentile_mean"),
                      ("similarity","batch_morgan_mean_similarity"),("chemistry","unique_molecule_morgan_mean_similarity"),
                      ("observed_feedback","high8_same_molecule_enrichment"),("local_condition_curve","pe_curve_extension_fraction"),
                      ("local_condition_curve","amount_curve_extension_fraction"),
                      ("batch_diversity","unique_molecules"),("failure_repair","high8_similarity_enrichment")]
    for text,target in behavior_targets:
        textcorr.extend(correlation_rows(free,["text_"+text+"_mention"],target,["seed"]))
    pd.DataFrame(textcorr).to_csv(HERE/"text_behavior_correlations.csv",index=False)
    textadjusted=[]
    for text,target in behavior_targets:
        textadjusted.extend(adjusted_correlations(free,["text_"+text+"_mention"],target,["seed"],["labels_before","validation_nrmse_before"]))
    pd.DataFrame(textadjusted).to_csv(HERE/"text_behavior_adjusted_correlations.csv",index=False)
    mentions=free.groupby(["seed","stage"])[["text_"+k+"_mention" for k in TEXT_PATTERNS]].mean().reset_index()
    mentions.to_csv(HERE/"text_mentions_by_seed_stage.csv",index=False)
    (HERE/"hypothesis_feedback_cases.json").write_text(json.dumps(case_studies,indent=2,ensure_ascii=False)+"\n")
    audit={"status":"PASS_360_BATCHES_11520_SELECTED_RECORDS", "batch_count":len(df),"selected_record_count":len(selected_df),
           "unique_metadata_ids":len(meta),"unique_molecules_in_eligible_metadata":len(chemistry.smiles),
           "id_mapping":"exact sample_id -> frozen catalog smiles + five experimental conditions, never positional guesses",
           "historical_truth_policy":"Only own Free catalog observed data and frozen selected Free feedback. No source response CSV, test truth, new label requests or predictions executed.",
           "lcmd_residual_limitation":"All LCMD historical residual features NA; frozen receipts only prove reveal hashes. No reconstruction from full source or foreign trajectories.",
           "morgan":{"radius":2,"bits":2048,"metric":"Tanimoto","unique_molecule_cache":True},
           "scaffold_caveat":"Exact Bemis-Murcko scaffold is only a family proxy; acyclic molecules are kept distinct. Functional groups may overlap.",
           "round_index":"0-based selection: 333+32*r labels before; evaluate improvement at the next fit",
           "text_caveat":"Narrative string values only, excluding schema keys and identifier/reference fields. Regex indicators may still misclassify semantic context, negations and self-disclaims. Constant indicators yield NA correlation. Mention does not imply causal contribution.",
           "statistics":"paired seed means (n=5); within-seed round correlations are descriptive only; no p-values",
           "source_hashes":SOURCES}
    (HERE/"behavior_data_audit.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False)+"\n")
    plot(df)
    print(json.dumps({k:audit[k] for k in ("status","batch_count","selected_record_count","unique_metadata_ids","unique_molecules_in_eligible_metadata")}))


def plot(df):
    os.environ.setdefault("MPLCONFIGDIR","/tmp/free_lcmd_diagnostics_mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors={FREE:"#a63b83","lcmd":"#148b88"}
    fig,axs=plt.subplots(2,3,figsize=(15,8.5))
    metrics=[("unique_molecules","Distinct molecules / 32 selected rows"),
             ("pe_curve_extension_fraction","Matched PE/EA extension fraction"),
             ("amount_curve_extension_fraction","Matched amount extension fraction"),
             ("exact_condition_repeat_fraction","Exact molecule-condition repeat fraction"),
             ("batch_morgan_mean_similarity","Within-batch Morgan Tanimoto mean"),
             ("new_molecule_row_fraction","Previously unobserved molecule fraction")]
    for ax,(metric,label) in zip(axs.flat,metrics):
        for method in (FREE,"lcmd"):
            sub=df[df.method==method]
            g=sub.groupby("labels_before")[metric]
            mean=g.mean(); sd=g.std()
            ax.plot(mean.index,mean.values,label="Free LLM32" if method==FREE else "LCMD32",color=colors[method])
            ax.fill_between(mean.index,mean-sd,mean+sd,alpha=.15,color=colors[method])
        ax.set_title(label,fontsize=10)
        ax.set_xlabel("Labels before frozen selection")
        ax.axvline(653,color="grey",ls=":",lw=.8);ax.axvline(1005,color="grey",ls=":",lw=.8)
    axs[0,0].legend(frameon=False)
    fig.suptitle("Frozen selection behavior, 5 paired seeds; mean ± across-seed SD\nNo test evaluation; validation trajectories used only for post-selection association",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.93))
    fig.savefig(HERE/"chemical_condition_behavior.png",dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,(metric,label) in zip(axs,[("high8_same_molecule_enrichment","High-error molecule selection minus pool fraction"),("high8_similarity_enrichment","High-error Morgan proximity minus pool mean")]):
        sub=df[df.method==FREE]
        for seed in SEEDS:
            g=sub[sub.seed==seed]
            ax.plot(g.labels_before,g[metric],alpha=.7,label=str(seed))
        ax.axhline(0,color="black",lw=.8)
        ax.set_title(label,fontsize=10); ax.set_xlabel("Labels before frozen selection")
    axs[0].legend(title="Seed",frameon=False,fontsize=8)
    fig.suptitle("Free only: pre-selection historical acquisition errors, top 8 records\nLCMD unavailable; historical errors are not current-model residuals",fontsize=11)
    fig.tight_layout(rect=(0,0,1,.9))
    fig.savefig(HERE/"historical_error_neighborhoods.png",dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
