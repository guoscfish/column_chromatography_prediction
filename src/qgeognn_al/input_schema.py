"""Machine-readable input contract for the frozen legacy QGeoGNN variant.

This module documents existing behavior. Importing it does not patch or replace
the legacy graph construction or forward path.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


LEGACY_VARIANT = "legacy_qgeognn_clean_reproduction_v1"
CATEGORICAL_BOND_FEATURES = ("bond_dir", "bond_type", "is_in_ring")
CONTINUOUS_EDGE_FEATURES = (
    "bond_length",
    "eluent_exact_mol_wt",
    "eluent_tpsa",
    "eluent_rotatable_bonds",
    "eluent_h_donors",
    "eluent_h_acceptors",
    "eluent_logp",
    "loading_solvent_code",
    "loading_amount_density_x_volume",
    "loading_solvent_volume_ul",
)
LEGACY_BOND_FLOAT_NAMES = ("bond_length", "prop", "e", "m", "V_e")
LEGACY_CONSUMED_CONTINUOUS_POSITIONS = tuple(range(len(LEGACY_BOND_FLOAT_NAMES)))
LEGACY_IGNORED_CONTINUOUS_POSITIONS = tuple(
    range(len(LEGACY_BOND_FLOAT_NAMES), len(CONTINUOUS_EDGE_FEATURES))
)


def legacy_input_schema() -> dict[str, Any]:
    mappings = []
    for position, semantic_name in enumerate(CONTINUOUS_EDGE_FEATURES):
        consumed = position in LEGACY_CONSUMED_CONTINUOUS_POSITIONS
        mappings.append(
            {
                "continuous_position": position,
                "edge_attr_position": len(CATEGORICAL_BOND_FEATURES) + position,
                "constructed_semantic": semantic_name,
                "legacy_encoder_name": (
                    LEGACY_BOND_FLOAT_NAMES[position] if consumed else None
                ),
                "consumed_by_legacy_bond_float_rbf": consumed,
            }
        )
    return {
        "schema_version": 1,
        "model_variant": LEGACY_VARIANT,
        "categorical_bond_features": list(CATEGORICAL_BOND_FEATURES),
        "continuous_edge_features": list(CONTINUOUS_EDGE_FEATURES),
        "legacy_bond_float_names": list(LEGACY_BOND_FLOAT_NAMES),
        "constructed_continuous_feature_count": len(CONTINUOUS_EDGE_FEATURES),
        "consumed_continuous_feature_count": len(LEGACY_CONSUMED_CONTINUOUS_POSITIONS),
        "continuous_feature_mapping": mappings,
        "forward_behavior_changed": False,
    }


def input_schema_hash(schema: dict[str, Any] | None = None) -> str:
    payload = schema if schema is not None else legacy_input_schema()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
