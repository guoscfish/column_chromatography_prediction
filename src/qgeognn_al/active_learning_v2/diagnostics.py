"""Label-free input redundancy and gradient mechanisms; never select from audits."""

import json

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.stats import spearmanr

from ..artifacts import sha256_file
from ..data import condition_matrix
from ..models import load_predictor_checkpoint
from ..resources import SOURCE_DATA, SOURCE_GRAPH_CACHE
from ..training.predictor import atomic_json, loader_pair
from .benchmark_protocol import DEVELOPMENT_SEEDS, OLD_STUDY, STUDY, load_features
from .cache import array_hash
from .lcmd import _squared_distances, lcmd_tp_select
from .protocol import stable_hash
from .runner import make_label_scrubbed_graphs, predict_outputs


def input_identities(data, cache, atom, angle, indices):
    """Hash actual tensor inputs, with separate raw/source component identities.

    y, sample_id and canonical_position do not enter model equivalence.
    Raw density and volume are recorded separately because V2 consumes their product.
    """
    rows = []
    for i in indices:
        row, a, b = data.iloc[i], atom[i], angle[i]
        entry = cache[str(row.canonical_smiles)]
        tensor_hashes = {f"{name}.{key}": array_hash(getattr(graph, key).cpu().numpy())
                         for name, graph, keys in (("atom", a, ("x", "edge_index", "edge_attr")),
                                                   ("angle", b, ("edge_index", "edge_attr"))) for key in keys}
        raw = {k: str(row[k]) for k in data.columns if k != "sample_id"}
        rows.append({"canonical_index": int(i), "sample_id": str(row.sample_id),
                     "canonical_smiles": str(row.canonical_smiles),
                     "graph_hash": stable_hash({key: array_hash(value) for key, value in entry["graph"].items()}),
                     "descriptor_hash": array_hash(entry["descriptor"]), "raw_X_hash": stable_hash(raw),
                     "input_hash": stable_hash(tensor_hashes),
                     "PE_EA": row["PE/EA"], "loading_solvent": row["loading solvent"],
                     "density": row["Density g/ml"], "sample_volume": row["V/ul"],
                     "loading_volume": row["Volume of loading solvent/ul"]})
    return pd.DataFrame(rows)


def full_gradient_hash(model, atom, angle, position, scales):
    model.eval()
    a, b = next(zip(*loader_pair(atom, angle, [position], 1)))
    output = model(a, b)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    endpoints = []
    for endpoint, column in enumerate((1, 4)):
        grads = torch.autograd.grad(output[0, column] / scales[endpoint], parameters,
                                    retain_graph=endpoint == 0, allow_unused=True)
        endpoints.append(stable_hash([array_hash(np.zeros(p.numel(), dtype=np.float32) if g is None
                                                else g.detach().cpu().numpy().reshape(-1))
                                     for p, g in zip(parameters, grads)]))
    return stable_hash(endpoints)


def duplicate_audit(identities, gradients, l0_count, selections, raw_gradient=None):
    if len(identities) != len(gradients):
        raise ValueError("audit input/gradient identity count mismatch")
    _, groups = np.unique(gradients, axis=0, return_inverse=True)
    gradient_rows = []
    for group in np.flatnonzero(np.bincount(groups) > 1):
        subset = identities.iloc[np.flatnonzero(groups == group)]
        unique_inputs = subset.drop_duplicates("input_hash")
        count = len(unique_inputs)
        explanation = "exact_model_input_equivalence"
        raw_count = None
        if count > 1:
            if raw_gradient is None:
                explanation = "different_X_same_sketch_unresolved"
            else:
                raw_count = len({raw_gradient(int(i)) for i in unique_inputs.canonical_index})
                explanation = ("same_full_Jacobian_local_symmetry_or_saturation" if raw_count == 1
                               else "CountSketch_collision_between_distinct_full_Jacobians")
        gradient_rows.append({"gradient_group": int(group), "rows": len(subset),
                              "unique_model_inputs": count, "raw_X_variants": subset.raw_X_hash.nunique(),
                              **{f"unique_{column}": int(subset[column].nunique()) if column in subset else None
                                 for column in ("canonical_smiles", "graph_hash", "descriptor_hash")},
                              "explained_by_exact_X": count == 1, "unique_full_Jacobians": raw_count,
                              "explanation": explanation})
    counts = identities.input_hash.value_counts()
    l0_keys = set(identities.iloc[:l0_count].input_hash)
    keyed = identities.set_index("sample_id").input_hash
    input_rows = []
    for arm, selected in selections.groupby("arm", sort=False):
        seen = set(l0_keys)
        redundant = 0
        for key in keyed.loc[selected.sort_values("selection_order").sample_id]:
            redundant += key in seen
            seen.add(key)
        keys = keyed.loc[selected.sample_id]
        input_rows.append({"arm": arm, "outer_rows": len(identities), "unique_model_inputs": len(counts),
                           "exact_X_duplicate_groups": int((counts > 1).sum()),
                           "exact_X_excess_rows": int((counts - 1).sum()),
                           "gradient_duplicate_groups": len(gradient_rows),
                           "gradient_excess_rows": len(gradients) - len(np.unique(gradients, axis=0)),
                           "different_X_same_gradient_groups": sum(r["unique_model_inputs"] > 1 for r in gradient_rows),
                           "selected_rows": len(selected), "selected_redundant_to_L0_or_earlier_batch": redundant,
                           "selected_redundant_fraction": redundant / len(selected),
                           "selected_from_duplicate_group_fraction": float((keys.map(counts) > 1).mean())})
    columns = ["gradient_group", "rows", "unique_model_inputs", "raw_X_variants", "explained_by_exact_X",
               "unique_canonical_smiles", "unique_graph_hash", "unique_descriptor_hash",
               "unique_full_Jacobians", "explanation"]
    return pd.DataFrame(input_rows), pd.DataFrame(gradient_rows, columns=columns)


def mechanism_variables(data, indices, l0_count, gradients, predictions):
    rows = data.iloc[indices]
    molecules = [Chem.MolFromSmiles(s) for s in rows.canonical_smiles]
    conditions = condition_matrix(data, np.asarray(indices)).astype(float)
    scale = conditions.std(axis=0)
    scale[scale < 1e-8] = 1
    z = (conditions - conditions.mean(axis=0)) / scale
    result = pd.DataFrame({"sample_id": rows.sample_id.to_numpy(),
                           "gradient_norm": np.linalg.norm(gradients.astype(float), axis=1),
                           "atom_count": [m.GetNumAtoms() for m in molecules],
                           "bond_count": [m.GetNumBonds() for m in molecules],
                           "MolWt": [Descriptors.MolWt(m) for m in molecules],
                           "predicted_V1_q50": predictions[:, 1], "predicted_V2_q50": predictions[:, 4],
                           "density": rows["Density g/ml"].to_numpy(),
                           "sample_volume": rows["V/ul"].to_numpy(),
                           "nearest_l0_condition_distance": np.sqrt(_squared_distances(z, z[:l0_count]).min(axis=1)),
                           "nearest_l0_gradient_distance": np.sqrt(_squared_distances(gradients, gradients[:l0_count]).min(axis=1))})
    names = ("eluent_exact_mol_wt", "eluent_tpsa", "eluent_rotatable_bonds", "eluent_h_donors",
             "eluent_h_acceptors", "eluent_logp", "loading_solvent_code", "loading_amount", "loading_volume")
    for j, name in enumerate(names):
        result[name] = conditions[:, j]
    # Categorical solvent contrasts avoid interpreting arbitrary vocabulary codes as a physical rank.
    for code, name in enumerate(("PE", "EA", "DCM")):
        result[f"loading_solvent_is_{name}"] = (conditions[:, 6] == code).astype(int)
    return result


def mechanism_correlations(variables, l0_count):
    pool = variables.iloc[l0_count:]
    records = []
    for name in pool.columns:
        if name in ("sample_id", "gradient_norm", "loading_solvent_code"):
            continue
        constant = pool[name].nunique() < 2 or pool.gradient_norm.nunique() < 2
        rho, pvalue = (np.nan, np.nan) if constant else spearmanr(pool.gradient_norm, pool[name])
        records.append({"variable": name, "spearman_rho": rho, "descriptive_pvalue": pvalue,
                        "rows": len(pool), "population": "U0", "label_free": True,
                        "constant_variable": constant, "norm_definition": "512D_sketch_L2"})
    return pd.DataFrame(records)


def audit_context(context, gradients, selections, output):
    model = load_predictor_checkpoint(context.runtime / "baseline_l0/best.pt")
    n = len(context.roles["l0"])
    identities = input_identities(context.data, context.graph_cache, context.atom, context.angle, context.outer)
    scales = tuple(context.preprocessing["target_scales"][t] for t in ("V1", "V2"))
    redundant, duplicate = duplicate_audit(identities, gradients, n, selections,
        lambda i: full_gradient_hash(model, context.atom, context.angle, i, scales))
    predictions, order = predict_outputs(model, context.atom, context.angle, context.outer)
    if not np.array_equal(order, context.outer):
        raise RuntimeError("diagnostic prediction order mismatch")
    variables = mechanism_variables(context.data, context.outer, n, gradients, predictions)
    profiles = []
    for arm, rows in selections.groupby("arm"):
        profile = variables.set_index("sample_id").loc[rows.sample_id].mean(numeric_only=True).to_dict()
        profiles.append({"arm": arm, **profile})
    for name, frame in (("input_redundancy_audit", redundant), ("gradient_duplicate_audit", duplicate),
                        ("gradient_mechanism_correlations", mechanism_correlations(variables, n)),
                        ("gradient_mechanism_selection_profiles", pd.DataFrame(profiles))):
        frame.assign(outer_seed=context.seed).to_csv(output / f"{name}.csv", index=False)


def audit_existing_study(study=STUDY):
    """Read only the old L0 models, X, gradients and selections; no metric/test files."""
    torch.set_num_threads(2)
    data, cache = load_features(), torch.load(SOURCE_GRAPH_CACHE, weights_only=False)
    input_frames, gradient_frames, correlation_frames, profiles, regressions = [], [], [], [], []
    sources = {}
    for seed in DEVELOPMENT_SEEDS:
        runtime = OLD_STUDY / "runtime" / f"seed_{seed}"
        names = ["baseline_l0/best.pt", "gradient_features.npz", "gradient_feature_audit.json",
                 "preprocessing.json", "selected_batches.csv"]
        for name in names:
            path = runtime / name
            sources[str(path)] = sha256_file(path)
        audit = json.loads((runtime / "gradient_feature_audit.json").read_text())
        if audit["feature_file_sha256"] != sha256_file(runtime / "gradient_features.npz"):
            raise RuntimeError("old gradient artifact hash mismatch")
        if audit["contract"]["checkpoint_sha256"] != sha256_file(runtime / "baseline_l0/best.pt"):
            raise RuntimeError("old L0 checkpoint hash mismatch")
        pre = json.loads((runtime / "preprocessing.json").read_text())["preprocessing"]
        atom, angle = make_label_scrubbed_graphs(data, cache, pre["scaler"])
        with np.load(runtime / "gradient_features.npz") as stored:
            gradients, indices = stored["features"], stored["canonical_indices"]
        split = pd.read_csv(OLD_STUDY / "splits" / f"row_seed_{seed}.csv")
        l0_count = int(split.role.eq("l0").sum())
        expected = np.r_[split.loc[split.role.eq("l0"), "canonical_index"],
                         split.loc[split.role.eq("u0"), "canonical_index"]]
        if not np.array_equal(indices, expected):
            raise RuntimeError("old gradient canonical order mismatch")
        selections = pd.read_csv(runtime / "selected_batches.csv")
        selected = lcmd_tp_select(gradients[l0_count:], gradients[:l0_count], 333).selected_pool_positions
        selected_ids = data.iloc[indices[l0_count:][selected]].sample_id.tolist()
        prior_ids = selections.loc[selections.arm.eq("lcmd")].sort_values("selection_order").sample_id.tolist()
        if selected_ids != prior_ids:
            raise RuntimeError("frozen 333-label LCMD selection regression")
        regressions.append({"outer_seed": seed, "batch_size": 333, "selected_ids_identical": True,
                            "ordered_selected_ids_hash": stable_hash(selected_ids)})
        model = load_predictor_checkpoint(runtime / "baseline_l0/best.pt")
        identities = input_identities(data, cache, atom, angle, indices)
        scales = tuple(pre["target_scales"][t] for t in ("V1", "V2"))
        redundant, duplicates = duplicate_audit(identities, gradients, l0_count, selections,
            lambda i: full_gradient_hash(model, atom, angle, i, scales))
        prediction, order = predict_outputs(model, atom, angle, indices)
        if not np.array_equal(order, indices):
            raise RuntimeError("mechanism prediction order drift")
        variables = mechanism_variables(data, indices, l0_count, gradients, prediction)
        input_frames.append(redundant.assign(outer_seed=seed))
        gradient_frames.append(duplicates.assign(outer_seed=seed))
        correlation_frames.append(mechanism_correlations(variables, l0_count).assign(outer_seed=seed))
        for arm, rows in selections.groupby("arm"):
            profile = variables.set_index("sample_id").loc[rows.sample_id].mean(numeric_only=True).to_dict()
            profiles.append({"outer_seed": seed, "arm": arm, **profile})
    results = study / "results"
    results.mkdir(parents=True, exist_ok=True)
    for name, frames in (("input_redundancy_audit", input_frames), ("gradient_duplicate_audit", gradient_frames),
                         ("gradient_mechanism_correlations", correlation_frames)):
        pd.concat(frames, ignore_index=True).to_csv(results / f"{name}.csv", index=False)
    pd.DataFrame(profiles).to_csv(results / "gradient_mechanism_selection_profiles.csv", index=False)
    pd.DataFrame(regressions).to_csv(results / "lcmd_333_regression.csv", index=False)
    for path, digest in sources.items():
        from pathlib import Path
        if sha256_file(Path(path)) != digest:
            raise RuntimeError("retrospective audit modified its source")
    atomic_json(results / "retrospective_audit_provenance.json", {
        "source_data_sha256": sha256_file(SOURCE_DATA), "read_only_sources": sources,
        "test_truth_access_count": 0, "formal_test_metrics_read": False,
        "diagnostic_population": "old development outer-training X only",
    })
