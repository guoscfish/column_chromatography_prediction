"""A7 packet encodings and the opt-in formal grouped display."""

from copy import deepcopy
from itertools import combinations
import json
import math

from studies.active_learning.pure_llm_v3_20261124.pure_llm_v3.records import dumps


ROLES = ("observed", "candidates")
SCHEMA = "a7_grouped_tables_v1"
FORMAL_WIRE_REPRESENTATION = {
    "schema": SCHEMA,
    "compression": "rounded_display",
    "feature_precision": 3,
    "crossing_guard": True,
    "visible_widths": True,
    "preserve_full_precision_host_packet": True,
    "bind_authoritative_packet_and_transmitted_payload_sha256": True,
}
AUTHORITATIVE_PACKET_FILE = "authoritative_packet.json"
WIDTHS = ("V1_quantile_width_ml", "V2_quantile_width_ml")
STRUCTURE = {"table_columns", "condition_columns", "molecules"}
GROUP_SIGNALS = {"molecule_novelty", "max_similarity_to_observed"}
DESCRIPTOR_DIGITS = {"LogP": 3, "MW_g_mol": 3, "TPSA_A2": 2}

FORMAT_GUIDE = """
Wire format a7_grouped_tables_v1. Read every molecule group and every observed
and candidate row. Zip each row with table_columns[role]. Add the supplied
row_constants[role] and the group's role_group values zipped with
group_columns[role]. Zip descriptors with descriptor_columns. Conditions are
ordered by condition_columns. Treat numeric entries in
numeric_condition_strings as their original decimal text. Add EA_fraction
from ea_fraction_by_ratio using the exact PE/EA text when that map is supplied.
Apply condition_constants and the reagent lookup as usual. Density is g/mL,
sample and loading solvent volumes are µL, and Flow is mL/min. For a field
listed as derived, loading_amount_mg is density × sample_volume, and each
raw quantile width is q90 minus q10 of that target, including negative values.
Use supplied widths as reported, including negative values. Keep all IDs
exactly as printed. All molecular identities, conditions, measured targets
and memory text retain their recorded values. If feature_precision is
supplied, predictions, errors and selected numeric features are a rounded
display view; near ties can be induced by display precision. When supplied,
crossing_guard retains exact numbers where rounding would hide a crossing.
""".strip()
FORMAT_SUFFIX = "\n\n" + FORMAT_GUIDE


def validate_wire_representation(value):
    expected = FORMAL_WIRE_REPRESENTATION
    if (type(value) is not dict or set(value) != set(expected)
            or any(type(value[key]) is not type(known) or value[key] != known for key, known in expected.items())):
        raise ValueError("unregistered formal A7 wire representation")
    return value


def _raw(packet):
    if packet.get("layout") != "B" or set(packet["table_columns"]) != set(ROLES):
        raise ValueError("expected the complete A7 layout B packet")
    return {role: [(key, dict(zip(packet["table_columns"][role], row, strict=True)))
                   for key, group in packet["molecules"].items() for row in group[role]]
            for role in ROLES}


def _raw_numeric(value):
    """Emit numbers only when their ordinary decimal text recovers the raw condition."""
    try:
        result = int(value)
    except ValueError:
        try:
            result = float(value)
        except ValueError:
            return value
    return result if (type(result) is int or math.isfinite(result)) and str(result) == value else value


def _calculate(field, row, conditions):
    if field == "loading_amount_mg":
        return float(conditions["density"]) * float(conditions["sample_volume"])
    if field not in WIDTHS:
        raise ValueError("unregistered derived feature")
    target = field.split("_")[0]
    return float(row[f"pred_{target}_q90_ml"]) - float(row[f"pred_{target}_q10_ml"])


def _exact_formula(field, rows, condition_columns):
    try:
        return bool(rows) and all(dumps(row[field]) == dumps(_calculate(
            field, row, dict(zip(condition_columns, row["conditions"], strict=True)))) for _, row in rows)
    except (KeyError, TypeError, ValueError):
        return False


def _per_molecule(rows):
    groups = {}
    for key, row in rows:
        groups.setdefault(key, []).append(row)
    return groups


def compact_packet(packet, *, keep_widths=True):
    """Factor out exact repetitions from layout B without reading any source dataset."""
    original = _raw(packet)
    input_conditions = packet["condition_columns"]
    condition_maps = [dict(zip(input_conditions, row["conditions"], strict=True))
                      for role in ROLES for _, row in original[role]]
    ea_by_ratio = {}
    if {"PE/EA", "EA_fraction"} <= set(input_conditions):
        for conditions in condition_maps:
            ratio, fraction = conditions["PE/EA"], conditions["EA_fraction"]
            if type(ratio) is not str or dumps(ea_by_ratio.setdefault(ratio, fraction)) != dumps(fraction):
                ea_by_ratio = {}
                break
    output_conditions = [name for name in input_conditions if name != "EA_fraction" or not ea_by_ratio]
    numeric_strings = [name for name in output_conditions if condition_maps and
                       all(type(row[name]) is str for row in condition_maps) and
                       any(type(_raw_numeric(row[name])) is not str for row in condition_maps)]
    descriptors = sorted({field for group in packet["molecules"].values() for field in group["descriptors"]})
    descriptor_vectors = all(set(group["descriptors"]) == set(descriptors) for group in packet["molecules"].values())
    packed = {key: deepcopy(value) for key, value in packet.items() if key not in STRUCTURE}
    packed.update(wire_schema=SCHEMA, condition_columns=output_conditions,
                  original_condition_columns=list(input_conditions), original_table_columns=deepcopy(packet["table_columns"]),
                  numeric_condition_strings=numeric_strings, ea_fraction_by_ratio=ea_by_ratio,
                  descriptor_columns=descriptors if descriptor_vectors else [], molecules={},
                  row_constants={}, group_columns={}, table_columns={}, derived={})
    for key, group in packet["molecules"].items():
        values = [group["descriptors"][name] for name in descriptors] if descriptor_vectors else group["descriptors"]
        packed["molecules"][key] = {**{field: deepcopy(value) for field, value in group.items()
                                      if field not in (*ROLES, "descriptors")},
                                    "descriptors": deepcopy(values), **{role: [] for role in ROLES}}
    for role, rows in original.items():
        columns = list(packet["table_columns"][role])
        constants = {field: rows[0][1][field] for field in columns if field not in ("id", "conditions") and rows
                     and len({dumps(row[field]) for _, row in rows}) == 1}
        grouped = _per_molecule(rows)
        group_fields = [field for field in columns if field in GROUP_SIGNALS and field not in constants
                        and all(len({dumps(row[field]) for row in values}) == 1 for values in grouped.values())]
        derivable = ("loading_amount_mg",) if keep_widths else ("loading_amount_mg", *WIDTHS)
        derived = [field for field in derivable if field in columns and field not in constants
                   and _exact_formula(field, rows, input_conditions)]
        external = set(constants) | set(group_fields) | set(derived)
        packed["row_constants"][role], packed["group_columns"][role], packed["derived"][role] = constants, group_fields, derived
        packed["table_columns"][role] = [name for name in columns if name not in external]
        for key, values in grouped.items():
            group = packed["molecules"][key]
            if group_fields:
                group[f"{role}_group"] = [values[0][name] for name in group_fields]
            for row in values:
                conditions = dict(zip(input_conditions, row["conditions"], strict=True))
                row["conditions"] = [_raw_numeric(conditions[name]) if name in numeric_strings
                                     else conditions[name] for name in output_conditions]
                group[role].append([row[name] for name in packed["table_columns"][role]])
    return packed


def expand_packet(packed):
    """Restore names, types, metadata, and row order of the original layout B representation."""
    if packed.get("wire_schema") != SCHEMA:
        raise ValueError("unknown compact packet schema")
    internal = STRUCTURE | {"wire_schema", "original_condition_columns", "original_table_columns",
                            "numeric_condition_strings", "ea_fraction_by_ratio", "descriptor_columns",
                            "row_constants", "group_columns", "derived", "feature_precision",
                            "crossing_guard", "source_compression"}
    packet = {key: deepcopy(value) for key, value in packed.items() if key not in internal}
    if "source_compression" in packed:
        packet["compression"] = packed["source_compression"]
    packet["condition_columns"] = list(packed["original_condition_columns"])
    packet["table_columns"] = deepcopy(packed["original_table_columns"])
    packet["molecules"] = {}
    for key, group in packed["molecules"].items():
        expanded = {field: deepcopy(value) for field, value in group.items()
                    if field not in (*ROLES, *(f"{role}_group" for role in ROLES))}
        descriptors = group["descriptors"]
        if isinstance(descriptors, list):
            expanded["descriptors"] = dict(zip(packed["descriptor_columns"], descriptors, strict=True))
        for role in ROLES:
            expanded[role] = []
            inherited = {**packed["row_constants"][role],
                         **dict(zip(packed["group_columns"][role],
                                    group.get(f"{role}_group", []), strict=bool(group[role])))}
            for values in group[role]:
                row = {**dict(zip(packed["table_columns"][role], values, strict=True)), **inherited}
                conditions = dict(zip(packed["condition_columns"], row["conditions"], strict=True))
                for name in packed["numeric_condition_strings"]:
                    conditions[name] = str(conditions[name])
                if packed["ea_fraction_by_ratio"]:
                    conditions["EA_fraction"] = packed["ea_fraction_by_ratio"][conditions["PE/EA"]]
                row["conditions"] = [conditions[name] for name in packet["condition_columns"]]
                for name in packed["derived"][role]:
                    row[name] = _calculate(name, row, conditions)
                expanded[role].append([row[name] for name in packet["table_columns"][role]])
        packet["molecules"][key] = expanded
    return packet


def _condition_dictionary(packet):
    control = deepcopy(packet)
    table = {}
    for group in control["molecules"].values():
        for role in ROLES:
            index = control["table_columns"][role].index("conditions")
            for row in group[role]:
                row[index] = table.setdefault(dumps(row[index]), len(table))
    control["condition_dictionary"] = [json.loads(value) for value in table]
    return control


def grouped_table_text(packed):
    """Comparison format: a JSON metadata line followed by per-molecule TSV tables."""
    metadata = deepcopy(packed)
    blocks = []
    for key, group in metadata["molecules"].items():
        for role in ROLES:
            rows = group.pop(role)
            if rows:
                blocks.append(f"TABLE\t{dumps(key)}\t{role}")
                blocks.extend("\t".join(map(dumps, row)) for row in rows)
    return "\n".join([dumps(metadata), *blocks])


def read_grouped_table(text):
    metadata, *lines = text.splitlines()
    packed = json.loads(metadata)
    groups = packed["molecules"]
    for group in groups.values():
        group.update({role: [] for role in ROLES})
    current = None
    for line in lines:
        parts = line.split("\t")
        if parts[0] == "TABLE":
            _, key, role = parts
            current = groups[json.loads(key)][role]
        elif current is None:
            raise ValueError("row before table header")
        else:
            current.append([json.loads(value) for value in parts])
    return packed


def _crossing_exceptions(original, rounded, triplets):
    def hidden(left, right):
        return original[left] > original[right] and rounded[left] <= rounded[right]

    exceptions = {target for target, positions in triplets.items()
                  if any(hidden(left, right) for left, right in combinations(positions, 2))}
    if len(triplets) == 2 and hidden(triplets["V1"][1], triplets["V2"][1]):
        exceptions.update(triplets)
    return exceptions


def round_view(packet, decimals, *, preserve_crossings=False):
    """Candidate display view. Original conditions, truths, identities and memory are exact."""
    if decimals not in (2, 3):
        raise ValueError("only the audited two- and three-decimal views are supported")
    result = deepcopy(packet)
    for group in result["molecules"].values():
        for field, digits in DESCRIPTOR_DIGITS.items():
            if field in group["descriptors"]:
                group["descriptors"][field] = round(group["descriptors"][field], digits)
        for role in ROLES:
            columns = result["table_columns"][role]
            positions = [i for i, name in enumerate(columns) if name.startswith(
                ("pred_", "premeasurement_prediction_", "premeasurement_error_")) or name in (
                    "historical_abs_error_ml", "max_similarity_to_observed", "max_similarity_to_pending")]
            width_positions = {field: columns.index(field) for field in WIDTHS if field in columns}
            triplets = {}
            for target in ("V1", "V2"):
                names = [f"pred_{target}_{q}_ml" for q in ("q10", "q50", "q90")]
                if all(name in columns for name in names):
                    triplets[target] = [columns.index(name) for name in names]
            for row in group[role]:
                original = row.copy() if preserve_crossings else None
                for i in positions:
                    if type(row[i]) is float:
                        row[i] = round(row[i], decimals)
                guarded = _crossing_exceptions(original, row, triplets) if preserve_crossings else set()
                for target in guarded:
                    for i in triplets[target]:
                        row[i] = original[i]
                if width_positions:
                    values = dict(zip(columns, row, strict=True))
                    for field, i in width_positions.items():
                        row[i] = original[i] if field.split("_")[0] in guarded else round(
                            _calculate(field, values, {}), decimals)
    return result


def packet_messages(messages, compact, *, feature_precision=None, crossing_guard=False):
    if feature_precision is not None:
        compact = {**compact, "source_compression": compact["compression"], "compression": "rounded_display",
                   "feature_precision": feature_precision}
    if crossing_guard:
        compact = {**compact, "crossing_guard": True}
    return [{**messages[0], "content": messages[0]["content"] + FORMAT_SUFFIX},
            {**messages[1], "content": dumps(compact)}, *messages[2:]]


def formal_wire_messages(messages, packet, representation):
    """Encode the approved display view while retaining the full-precision packet on the host."""
    validate_wire_representation(representation)
    precision = representation["feature_precision"]
    guard = representation["crossing_guard"]
    displayed = round_view(packet, precision, preserve_crossings=guard)
    packed = compact_packet(displayed, keep_widths=representation["visible_widths"])
    if expand_packet(packed) != displayed:
        raise ValueError("compact A7 packet does not recover the complete display")
    return packet_messages(messages, packed, feature_precision=precision, crossing_guard=guard)


def candidate_formats(messages):
    """Return complete message values for comparable offline token estimates."""
    raw = json.loads(messages[1]["content"])
    factored = compact_packet(raw)
    dict_control = _condition_dictionary(raw)
    table = grouped_table_text(factored)
    baseline = deepcopy(messages)
    return {
        "existing_grouped_json": baseline,
        "global_condition_dictionary": [
            {**baseline[0], "content": baseline[0]["content"] + "\n\nEach conditions cell is a zero-based index into condition_dictionary."},
            {**baseline[1], "content": dumps(dict_control)}, *baseline[2:]],
        "grouped_lossless_visible_widths": packet_messages(messages, factored),
        "grouped_lossless_derived_widths": packet_messages(messages, compact_packet(raw, keep_widths=False)),
        "grouped_lossless_tsv": [
            {**baseline[0], "content": baseline[0]["content"] + "\n\n" + FORMAT_GUIDE +
             "\nThe first line is JSON metadata. TABLE lines open molecule and role blocks. Each subsequent row uses tabs between JSON cells."},
            {**baseline[1], "content": table}, *baseline[2:]],
        **{f"grouped_{digits}dp_visible_widths": packet_messages(messages, compact_packet(round_view(raw, digits)),
                                                               feature_precision=digits) for digits in (3, 2)},
        "grouped_3dp_crossing_guard": packet_messages(messages, compact_packet(
            round_view(raw, 3, preserve_crossings=True)), feature_precision=3, crossing_guard=True),
    }
