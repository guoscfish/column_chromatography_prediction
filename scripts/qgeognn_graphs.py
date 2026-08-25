"""Deterministic conformer selection and QGeoGNN graph construction."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# Mordred 1.2.0 still imports NumPy's removed ``product`` alias.  Keep this
# compatibility shim at the legacy integration boundary; ``prod`` has the
# same behavior and no model or feature contract changes.
if not hasattr(np, "product"):
    np.product = np.prod  # type: ignore[attr-defined]


ROOT = Path(__file__).resolve().parents[1]
APPLICATION_DIR = ROOT / "application"
if str(APPLICATION_DIR) not in sys.path:
    sys.path.insert(0, str(APPLICATION_DIR))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qgeognn_matplotlib")
import QGeoGNN as qg  # noqa: E402


MORDRED_INDICES = [153, 278, 884, 885, 1273, 1594, 431, 1768, 1769, 1288, 1521]
CONFORMER_POLICIES = {"first_embedded", "lowest_energy"}


def _keep_only_conformer(mol: Chem.Mol, conformer_id: int) -> Chem.Mol:
    """Return a copy whose selected conformer is the only conformer and has id 0."""
    selected = Chem.Mol(mol)
    conformer = Chem.Conformer(mol.GetConformer(int(conformer_id)))
    selected.RemoveAllConformers()
    selected.AddConformer(conformer, assignId=True)
    return selected


def build_graph_and_descriptor(
    smiles: str,
    seed: int,
    conformer_policy: str = "first_embedded",
) -> tuple[dict, np.ndarray, dict]:
    """Build one deterministic graph and return auditable conformer metadata.

    Both policies embed and optimize the same ten conformers. ``first_embedded``
    reproduces the current code path; ``lowest_energy`` makes the optimized
    minimum-energy conformer the sole/default conformer before graph creation.
    """
    if conformer_policy not in CONFORMER_POLICIES:
        raise ValueError(f"unknown_conformer_policy:{conformer_policy}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("rdkit_parse_failed")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 1
    conformer_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=10, params=params))
    if not conformer_ids:
        raise ValueError("conformer_embedding_failed")
    if AllChem.MMFFHasAllMoleculeParams(mol):
        optimization = list(AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1))
        force_field = "MMFF94"
    else:
        optimization = list(AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=1))
        force_field = "UFF_fallback"
    if not optimization or len(optimization) != len(conformer_ids):
        raise ValueError("conformer_optimization_failed")

    energies = np.asarray([result[1] for result in optimization], dtype=np.float64)
    minimum_index = int(np.argmin(energies))
    selected_index = 0 if conformer_policy == "first_embedded" else minimum_index
    selected_id = int(conformer_ids[selected_index])
    if conformer_policy == "lowest_energy":
        mol = _keep_only_conformer(mol, selected_id)

    mol = Chem.RemoveHs(mol)
    graph = qg.mol_to_geognn_graph_data_MMFF3d(mol)
    if graph is None:
        raise ValueError("graph_construction_failed")
    mordred = np.asarray(qg.mord(mol), dtype=np.float64)
    if mordred.shape[0] <= max(MORDRED_INDICES):
        raise ValueError(f"mordred_dimension_{mordred.shape[0]}")
    selected = np.nan_to_num(
        mordred[MORDRED_INDICES], nan=0.0, posinf=0.0, neginf=0.0
    )
    attributes = np.array(
        [
            Descriptors.ExactMolWt(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.MolLogP(mol),
        ],
        dtype=np.float64,
    )
    descriptor = np.concatenate([attributes, selected]).astype(np.float32)
    metadata = {
        "conformer_policy": conformer_policy,
        "force_field": force_field,
        "embedded_conformers": len(conformer_ids),
        "selected_conformer_id": selected_id,
        "selected_energy": float(energies[selected_index]),
        "first_conformer_id": int(conformer_ids[0]),
        "first_energy": float(energies[0]),
        "first_is_minimum": bool(minimum_index == 0),
        "minimum_conformer_id": int(conformer_ids[minimum_index]),
        "minimum_energy": float(energies[minimum_index]),
        "energy_range": float(energies.max() - energies.min()),
        "selected_is_minimum": bool(selected_index == minimum_index),
        "nonconverged_conformers": int(sum(int(result[0]) != 0 for result in optimization)),
        "rdkit_seed": int(seed),
    }
    return graph, descriptor, metadata
