"""Authorized full-catalog transforms transferred from the scientific protocol.

No file paths, graph objects, model handles or label loaders cross this surface.
Descriptor/novelty values and salted row order retain main's exact definitions.
"""
from __future__ import annotations
import copy
from ...infra.contracts import content_hash as stable_hash

CONDITIONS = ("PE/EA", "Density g/ml", "V/ul", "loading solvent", "Volume of loading solvent/ul")

DESCRIPTORS = ("MW_g_mol", "LogP", "TPSA_A2", "HBD", "HBA")

PREDICTIONS = tuple(f"pred_{t}_{q}_ml" for t in ("V1", "V2") for q in ("q10", "q50", "q90"))

OBSERVED_FIELDS = ("V1_ml", "V2_ml", "record_kind", "premeasurement_prediction_V1_ml",
                   "premeasurement_prediction_V2_ml", "premeasurement_error_V1_ml",
                   "premeasurement_error_V2_ml")

def clean_card(row, observed=False):
    allowed = {"id", "smiles", "conditions", "descriptors", "pred_V1_ml", "pred_V2_ml",
               "cw_distance_percentile", *PREDICTIONS}
    if observed:
        allowed.update(OBSERVED_FIELDS)
    if set(row) - allowed:
        raise ValueError(f"unauthorized catalog fields: {sorted(set(row) - allowed)}")
    if set(row["conditions"]) != set(CONDITIONS):
        raise ValueError("unexpected experimental conditions")
    result = copy.deepcopy(row)
    # Descriptors are always recomputed, never trusted as a second payload channel.
    result.pop("descriptors", None)
    return result

class FullCatalog:
    def __init__(self, candidates, observed, pending, *, salt):
        from rdkit import Chem, DataStructs
        from rdkit.Chem import Descriptors, rdFingerprintGenerator
        inputs = ([clean_card(row) for row in candidates],
                  [clean_card(row, True) for row in observed],
                  [clean_card(row) for row in pending])
        self.candidates, self.observed, self.pending = ({str(row['id']): row for row in rows} for rows in inputs)
        if any(len(mapping) != len(rows) for mapping, rows in zip((self.candidates,self.observed,self.pending), inputs)):
            raise ValueError('duplicate catalog IDs')
        groups = [set(self.candidates), set(self.observed), set(self.pending)]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError('candidate/observed/pending overlap')
        all_rows = [row for rows in inputs for row in rows]
        mols = {row['smiles']: Chem.MolFromSmiles(row['smiles']) for row in all_rows}
        if any(mol is None for mol in mols.values()):
            raise ValueError('invalid SMILES')
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.fps = {smiles: generator.GetFingerprint(mol) for smiles, mol in mols.items()}
        def descriptors(mol):
            return {'MW_g_mol': round(Descriptors.MolWt(mol), 2),
                    'LogP': round(Descriptors.MolLogP(mol), 2),
                    'TPSA_A2': round(Descriptors.TPSA(mol), 2),
                    'HBD': Descriptors.NumHDonors(mol), 'HBA': Descriptors.NumHAcceptors(mol)}
        for mapping in (self.candidates, self.observed, self.pending):
            for row in mapping.values():
                row['descriptors'] = descriptors(mols[row['smiles']])
        self.salt = salt
        self.references = {**self.candidates, **self.observed, **self.pending}
        obs_mols = {r['smiles'] for r in self.observed.values()}
        obs_conditions = {(r['smiles'], stable_hash(r['conditions'])) for r in self.observed.values()}
        obs_fps = [self.fps[s] for s in sorted(obs_mols)]
        pending_fps = [self.fps[s] for s in sorted({r['smiles'] for r in self.pending.values()})]
        similarity = {s: (max(DataStructs.BulkTanimotoSimilarity(fp, obs_fps), default=0.0),
                          max(DataStructs.BulkTanimotoSimilarity(fp, pending_fps), default=0.0))
                      for s, fp in self.fps.items()}
        for row in self.references.values():
            row['loading_amount_mg'] = float(row['conditions']['Density g/ml']) * float(row['conditions']['V/ul'])
            row['molecule_novelty'] = int(row['smiles'] not in obs_mols)
            row['condition_novelty'] = int((row['smiles'], stable_hash(row['conditions'])) not in obs_conditions)
            row['max_similarity_to_observed'], row['max_similarity_to_pending'] = similarity[row['smiles']]
            for target in ('V1','V2'):
                lo,hi = row.get(f'pred_{target}_q10_ml'),row.get(f'pred_{target}_q90_ml')
                if lo is not None and hi is not None:
                    row[f'{target}_quantile_width_ml'] = hi-lo
            if 'premeasurement_error_V1_ml' in row:
                row['historical_abs_error_ml'] = max(abs(row[f'premeasurement_error_{t}_ml']) for t in ('V1','V2'))

    def key(self, row):
        return stable_hash([self.salt, row['id']])

def encode_catalog(catalog):
    """Factor repeated molecular metadata without dropping any row or precision."""
    molecules = {}
    keys = {}
    for row in sorted(catalog.references.values(), key=lambda r: (r['smiles'], r['id'])):
        if row['smiles'] not in keys:
            key = f'm{len(keys)}'
            keys[row['smiles']] = key
            molecules[key] = {'smiles': row['smiles'], 'descriptors': row['descriptors']}
        elif molecules[keys[row['smiles']]]['descriptors'] != row['descriptors']:
            raise ValueError('inconsistent molecular metadata')
    tables = {}
    for name in ('candidates', 'observed', 'pending'):
        records = sorted(getattr(catalog, name).values(), key=catalog.key)
        fields = sorted(set().union(*(set(r) for r in records)) - {'id', 'smiles', 'descriptors', 'conditions'})
        tables[name] = {'columns': ['id', 'molecule_id', 'conditions', *fields],
                        'rows': [[r['id'], keys[r['smiles']], [r['conditions'][c] for c in CONDITIONS],
                                  *[r.get(f) for f in fields]] for r in records]}
    return {'condition_columns': list(CONDITIONS), 'molecules': molecules, **tables}

def decode_table(packet, name):
    rows = []
    for values in packet[name]['rows']:
        if len(values) != len(packet[name]['columns']):
            raise ValueError('table width mismatch')
        row = dict(zip(packet[name]['columns'], values))
        row.update(packet['molecules'][row.pop('molecule_id')])
        row['conditions'] = dict(zip(packet['condition_columns'], row['conditions']))
        rows.append(row)
    return rows

def scientific_memory(history, seed, method):
    if any((h["seed"], h["method"]) != (seed, method) for h in history):
        raise ValueError("foreign trajectory in scientific memory")
    last = history[-1] if history else {}
    records = [r for h in history for r in h["records"]]
    high_error = sorted(records, key=lambda r: (-max(abs(r["error_V1_ml"]), abs(r["error_V2_ml"])), r["candidate_id"]))[:8]
    return {"previous_hypotheses": last.get("hypotheses", []),
            "previous_feedback_interpretation": last.get("feedback_interpretation", ""),
            "previous_batch_rationale": last.get("batch_rationale", ""),
            "previous_selected_ids": [r["candidate_id"] for r in records],
            "recent_outcomes": last.get("records", []), "high_error_measured_examples": high_error,
            "memory_policy": "verbatim previous LLM analysis; exact latest 32 and top-8 absolute-error records; older records queryable"}
