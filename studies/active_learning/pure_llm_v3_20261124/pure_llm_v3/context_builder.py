"""Full-pool layout adapters with explicit per-role field boundaries."""
from dataclasses import dataclass
import copy
import math

from .identity_codec import IdentityCodec
from .records import digest, dumps

LEGACY_CONDITIONS = ("PE/EA", "Density g/ml", "V/ul", "loading solvent", "Volume of loading solvent/ul")
CONDITIONS = ("PE/EA", "density", "sample_volume", "loading_solvent", "loading_solvent_volume",
              "CAS", "name", "column_specs", "purity_raw", "purity_fraction", "purity_kind", "flow",
              "EA_fraction")
REAGENT_FIELDS = ("CAS", "name", "purity_raw", "purity_fraction", "purity_kind")
DESCRIPTORS = {"MW_g_mol", "LogP", "TPSA_A2", "HBD", "HBA"}
TARGETS = ("V1", "V2")
QUANTILES = {f"pred_{target}_{q}_ml" for target in TARGETS for q in ("q10", "q50", "q90")}
ALIASES = {f"pred_{target}_ml" for target in TARGETS}
WIDTHS = {f"{target}_quantile_width_ml" for target in TARGETS}
SIGNALS = {"condition_novelty", "molecule_novelty", "loading_amount_mg", "max_similarity_to_observed"}
PREMEASUREMENT = {f"premeasurement_{name}_{target}_ml" for target in TARGETS for name in ("prediction", "error")}
TRUTH = {"V1_ml", "V2_ml"}
CANDIDATE_FIELDS = QUANTILES | ALIASES | WIDTHS | SIGNALS | {"max_similarity_to_pending"}
OBSERVED_FIELDS = PREMEASUREMENT | TRUTH | SIGNALS | {"record_kind", "historical_abs_error_ml",
                                                      "max_similarity_to_pending"}
LEGACY_PACKET_FIELDS = {
    "study_version", "seed", "round", "active_label_count", "selection_count", "objective", "memory",
    "validation_test_records", "delivery", "candidate_count", "observed_count", "condition_columns",
    "molecules", "candidates", "observed", "pending", "method", "packet_sha256",
}
LEGACY_MEMORY_FIELDS = {
    "previous_hypotheses", "previous_feedback_interpretation", "previous_batch_rationale",
    "previous_selected_ids", "recent_outcomes", "high_error_measured_examples", "memory_policy",
}
LEGACY_HYPOTHESIS_FIELDS = {
    "id", "content", "basis", "supporting_observed_ids", "contradicting_observed_ids",
    "confidence", "status", "why_it_matters_for_model_learning", "next_discriminating_question",
}
LEGACY_FEEDBACK_FIELDS = {"candidate_id", "source", "pred_V1_ml", "pred_V2_ml", "true_V1_ml",
                          "true_V2_ml", "error_V1_ml", "error_V2_ml", "reason", "hypothesis_id"}


def _finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def _read_table(table, allowed):
    columns = table["columns"]
    if (
        set(table) != {"columns", "rows"} or len(columns) != len(set(columns))
        or set(columns) - (allowed | {"id", "molecule_id", "conditions"})
        or not {"id", "molecule_id", "conditions"} <= set(columns)
    ):
        raise ValueError("unexpected table schema")
    for row in table["rows"]:
        if len(row) != len(columns):
            raise ValueError("unexpected row width")
        yield {key: value for key, value in zip(columns, row, strict=True) if value is not None}


def _validate_legacy_memory(memory, observed, history):
    if set(memory) != LEGACY_MEMORY_FIELDS:
        raise ValueError("unexpected legacy memory fields")
    if any(type(memory[key]) is not str for key in ("previous_feedback_interpretation",
           "previous_batch_rationale", "memory_policy")):
        raise ValueError("unexpected legacy memory text")
    past = {r["candidate_id"]: r for h in history for r in h["records"]}
    if memory["previous_selected_ids"] != list(past):
        raise ValueError("legacy memory differs from recorded selection history")
    for table in ("recent_outcomes", "high_error_measured_examples"):
        for record in memory[table]:
            if set(record) != LEGACY_FEEDBACK_FIELDS or record != past.get(record["candidate_id"]):
                raise ValueError("unauthorized historical measurement in legacy memory")
    for hypothesis in memory["previous_hypotheses"]:
        if set(hypothesis) != LEGACY_HYPOTHESIS_FIELDS:
            raise ValueError("unexpected legacy hypothesis fields")
        for key in ("supporting_observed_ids", "contradicting_observed_ids"):
            if not isinstance(hypothesis[key], list) or not set(hypothesis[key]) <= observed:
                raise ValueError("foreign legacy evidence")
        if any(type(v) is not str for k, v in hypothesis.items() if not k.endswith("_observed_ids")):
            raise ValueError("unexpected legacy hypothesis value")


def read_legacy_cards(state):
    packet = state.packet
    packet_fields = set(packet)
    if (packet_fields - LEGACY_PACKET_FIELDS or LEGACY_PACKET_FIELDS - {"method"} - packet_fields
            or tuple(packet["condition_columns"]) != LEGACY_CONDITIONS):
        raise ValueError("unexpected historical packet schema")
    if list(_read_table(packet["pending"], set())):
        raise ValueError("historical packet has pending selections")
    molecules = packet["molecules"]
    for molecule in molecules.values():
        if (
            set(molecule) != {"smiles", "descriptors"} or type(molecule["smiles"]) is not str
            or set(molecule["descriptors"]) != DESCRIPTORS
            or not all(map(_finite, molecule["descriptors"].values()))
        ):
            raise ValueError("unexpected molecular dictionary")
    result = []
    for role, allowed, required in (
        ("candidates", CANDIDATE_FIELDS, QUANTILES | WIDTHS | SIGNALS),
        ("observed", OBSERVED_FIELDS, TRUTH | SIGNALS | {"record_kind"}),
    ):
        rows = {}
        for row in _read_table(packet[role], allowed):
            sample_id = row["id"]
            conditions = dict(zip(LEGACY_CONDITIONS, row.pop("conditions"), strict=True))
            if (any(not _finite(conditions[k]) for k in ("Density g/ml", "V/ul", "Volume of loading solvent/ul"))
                    or any(type(conditions[k]) is not str for k in ("PE/EA", "loading solvent"))):
                raise ValueError("unexpected historical condition value")
            molecule = molecules[row["molecule_id"]]
            values = {key: value for key, value in row.items() if key in allowed}
            if not required <= set(values):
                raise ValueError("incomplete historical record")
            if any(not _finite(v) for k, v in values.items() if k != "record_kind"):
                raise ValueError("invalid historical numeric value")
            if role == "observed" and values["record_kind"] not in (
                "initial_L333", "acquired_with_premeasurement_prediction"
            ):
                raise ValueError("unknown historical observation kind")
            if sample_id in rows:
                raise ValueError("duplicate historical row")
            rows[sample_id] = {**row, "conditions": conditions, **molecule}
        result.append(rows)
    candidates, observed = result
    if any(
        set(rows) != set(state.provenance[f"expected_{role}"])
        for role, rows in (("candidates", candidates), ("observed", observed))
    ) or (len(candidates), len(observed), len(observed)) != (
        packet["candidate_count"], packet["observed_count"], packet["active_label_count"]
    ):
        raise ValueError("packet role IDs or counts disagree with frozen lineage")
    _validate_legacy_memory(packet["memory"], set(observed), state.history)
    return candidates, observed


@dataclass(frozen=True)
class Context:
    seed: int
    round: int
    checkpoint: str
    candidates: dict
    observed: dict
    codec: IdentityCodec
    facts: dict
    legacy_molecules: dict
    legacy_memory: dict

    @property
    def all_row_ids(self):
        return self.codec.all_row_ids

    @property
    def mapping_digest(self):
        roles = self.all_row_ids
        legacy_molecules = {key: value["smiles"] for key, value in self.legacy_molecules.items()}
        return digest({"rows": self.codec.mapping_digest({roles[i] for i in self.observed}),
                       "legacy_molecules": legacy_molecules})


def make_context(state, metadata, *, identity="short", include_metadata=True, legacy_molecules=False):
    source_candidates, source_observed = read_legacy_cards(state)
    structures = {row["id"]: row["smiles"] for row in (*source_observed.values(), *source_candidates.values())}
    codec = IdentityCodec.from_pool(structures, identity)
    previous = {row["candidate_id"]: (row, history) for history in state.history for row in history["records"]}
    facts = {}
    def convert(rows, observed):
        result = {}
        for sample_id, source in rows.items():
            item = copy.deepcopy(source)
            row_meta = metadata[sample_id]
            if row_meta.canonical_smiles != item["smiles"]:
                raise ValueError("sidecar structure differs from frozen record")
            if include_metadata:
                conditions = row_meta.public_conditions()
                if set(conditions) != set(CONDITIONS):
                    raise ValueError("metadata sidecar fields drifted")
                item["conditions"] = conditions
            item["id"] = codec.row(sample_id, observed=observed)
            if not legacy_molecules:
                item["molecule_id"] = codec.molecules[item["smiles"]]
            if observed:
                if sample_id in previous:
                    prior, history = previous[sample_id]
                    item["acquisition_round"] = history["round"]
                    expected = {
                        **{f"{t}_ml": prior[f"true_{t}_ml"] for t in TARGETS},
                        **{f"premeasurement_prediction_{t}_ml": prior[f"pred_{t}_ml"] for t in TARGETS},
                        **{f"premeasurement_error_{t}_ml": prior[f"error_{t}_ml"] for t in TARGETS},
                        "record_kind": "acquired_with_premeasurement_prediction",
                    }
                    if any(item.get(k) != v for k, v in expected.items()):
                        raise ValueError("observation differs from verified historical feedback")
                    facts[sample_id] = {"choice_reason": prior["reason"], "hypothesis_id": prior["hypothesis_id"],
                        "acquisition_checkpoint_sha256": history["checkpoint_sha256"],
                        "acquisition_round": history["round"]}
                elif item["record_kind"] != "initial_L333" or set(item) & PREMEASUREMENT:
                    raise ValueError("observation has no acquisition lineage")
            result[item["id"]] = item
        return result
    return Context(state.seed, state.round, state.checkpoint, convert(source_candidates, False),
                   convert(source_observed, True), codec, facts, state.packet["molecules"], state.packet["memory"])


def _project(row, *, compress):
    row = {k: v for k, v in row.items() if k not in {"smiles", "descriptors"}}
    if compress:
        if row.pop("max_similarity_to_pending", 0) != 0:
            raise ValueError("cannot compress nonzero pending signal")
        for target in TARGETS:
            alias = f"pred_{target}_ml"
            if alias in row and row.pop(alias) != row[f"pred_{target}_q50_ml"]:
                raise ValueError("cannot compress unequal prediction alias")
    return row


def encode_context(context, *, layout="A", compress=True, memory=None):
    if layout not in ("A", "B"):
        raise ValueError("unknown layout")
    groups = {}
    columns = [key for key in CONDITIONS if key in next(iter(context.candidates.values()))["conditions"]]
    if not columns or set(columns) != set(next(iter(context.candidates.values()))["conditions"]):
        columns = list(LEGACY_CONDITIONS)
    constants = {}
    reagents = {}
    factor_reagents = compress and set(REAGENT_FIELDS) <= set(columns)
    if compress and "column_specs" in columns:
        specifications = {row["conditions"]["column_specs"]
                          for rows in (context.observed, context.candidates) for row in rows.values()}
        if len(specifications) == 1:
            constants["column_specs"] = specifications.pop()
    serialized_columns = [
        key for key in columns if key not in constants and (not factor_reagents or key not in REAGENT_FIELDS)
    ]
    if factor_reagents:
        serialized_columns.append("reagent_index")
    result = {"seed": context.seed, "round": context.round, "selection_count": 32, "layout": layout,
              "candidate_count": len(context.candidates), "observed_count": len(context.observed),
              "condition_columns": serialized_columns, "compression": "lossless" if compress else "none",
              "memory": memory or {}}
    if constants:
        result["condition_constants"] = constants
    tables = {}

    def pack_conditions(values):
        encoded = [values[k] for k in serialized_columns if k != "reagent_index"]
        if factor_reagents:
            reagent = tuple(values[k] for k in REAGENT_FIELDS)
            encoded.append(reagents.setdefault(reagent, len(reagents)))
        return encoded
    for role in ("observed", "candidates"):
        rows = getattr(context, role)
        projected = []
        for row in rows.values():
            m = {"smiles": row["smiles"], "descriptors": row["descriptors"]}
            if groups.setdefault(row["molecule_id"], m) != m:
                raise ValueError("molecular descriptor drift")
            if set(row["conditions"]) != set(columns):
                raise ValueError("inconsistent condition fields")
            projected.append(_project(row, compress=compress))
        fields = ["id", "molecule_id", "conditions", *sorted(
            set().union(*(set(r) for r in projected)) - {"id", "molecule_id", "conditions"}
        )]
        tables[role] = {"columns": fields, "rows": [
            [*[row[k] for k in fields[:2]], pack_conditions(row["conditions"]),
             *(row.get(k) for k in fields[3:])] for row in projected
        ]}
    if factor_reagents:
        result["reagent_columns"] = list(REAGENT_FIELDS)
        result["reagents"] = list(reagents)
    if layout == "A":
        result.update(tables, molecules=groups)
    else:
        result["table_columns"] = {
            role: [k for k in table["columns"] if k != "molecule_id"] for role, table in tables.items()
        }
        result["molecules"] = {key: {**groups[key], "observed": [], "candidates": []} for key in sorted(groups)}
        for role, table in tables.items():
            for sample_id, molecule_id, *values in table["rows"]:
                result["molecules"][molecule_id][role].append([sample_id, *values])
    if not compress:
        result["pending"] = {"columns": ["id", "molecule_id", "conditions"], "rows": []}
    # A serialization check protects against NaN, text/shape drift, and accidental omission.
    dumps(result)
    if decode_context(result) != {role: getattr(context, role) for role in tables}:
        raise ValueError("full-pool encode/decode mismatch")
    return result


def decode_context(packet):
    decoded = {}
    for role in ("observed", "candidates"):
        if packet["layout"] == "A":
            table = packet[role]
        elif packet["layout"] == "B":
            columns = list(packet["table_columns"][role])
            columns.insert(1, "molecule_id")
            table = {"columns": columns, "rows": [
                [sample_id, molecule_id, *values] for molecule_id, group in packet["molecules"].items()
                for sample_id, *values in group[role]
            ]}
        else:
            raise ValueError("unknown layout")
        allowed = (OBSERVED_FIELDS | {"acquisition_round"}) if role == "observed" else CANDIDATE_FIELDS
        rows = {}
        for row in _read_table(table, allowed):
            molecule = packet["molecules"][row["molecule_id"]]
            row.update({key: molecule[key] for key in ("smiles", "descriptors")})
            conditions = dict(zip(packet["condition_columns"], row["conditions"], strict=True))
            constants = packet.get("condition_constants", {})
            if constants.keys() & conditions.keys() or set(constants) - {"column_specs"}:
                raise ValueError("invalid global condition constants")
            conditions.update(constants)
            if "reagent_index" in conditions:
                index = conditions.pop("reagent_index")
                if (type(index) is not int or not 0 <= index < len(packet["reagents"])
                        or tuple(packet["reagent_columns"]) != REAGENT_FIELDS
                        or conditions.keys() & set(REAGENT_FIELDS)):
                    raise ValueError("invalid reagent dictionary reference")
                conditions.update(dict(zip(REAGENT_FIELDS, packet["reagents"][index], strict=True)))
            row["conditions"] = conditions
            if packet["compression"] == "lossless":
                row["max_similarity_to_pending"] = 0.0
                if role == "candidates":
                    for target in TARGETS:
                        row[f"pred_{target}_ml"] = row[f"pred_{target}_q50_ml"]
            if row["id"] in rows:
                raise ValueError("duplicate encoded row")
            rows[row["id"]] = row
        decoded[role] = rows
    return decoded
