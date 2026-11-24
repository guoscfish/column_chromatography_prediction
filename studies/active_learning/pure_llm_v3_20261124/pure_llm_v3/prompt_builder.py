"""Independent versioned English prompts and canonical request messages."""
from pathlib import Path

from .records import dumps

STUDY_PATH = Path("studies/active_learning/pure_llm_v3_20261124")
PROMPTS = {"v3": "PROMPT_PURE_LLM_V3.md", "v3_legacy": "PROMPT_V3_LEGACY_SCHEMA_COMPATIBLE.md"}

METADATA_ADDENDUM = """The condition_columns also supply Flow (mL/min), raw Purity and its
numeric interpretation where available, CAS and reagent name. The label-free EA
fraction is b/(a+b) for PE/EA = a/b. Flow was used in converting time to volume.
Flow, Purity, CAS and name are not independent predictor inputs. A physical
difference can survive even when the predictor sees the same input. Use the full
recorded conditions before claiming a single-factor contrast or replicate. In
grouped delivery, each group's dictionary key is the molecule_id for its rows.
Add condition_constants when supplied. If a condition array contains reagent_index,
use it as a zero-based index into reagents and zip that entry with reagent_columns."""

REPAIR_INSTRUCTIONS = """Return one JSON patch for the supplied repair_id. A patch may replace
only the indicated choice or hypothesis slots and top-level fields. Supply a full
replacement choice or update for each changed slot. Keep valid choices, reasons,
and evidence locked. Use zero-based slot numbers. A replan request requires a brief
scientific reason and permits no edits. Reply with mode, repair_id, reason and,
for a patch, replace_choices [{slot, choice}], append_choices [choice],
replace_hypothesis_updates [{slot, update}], and replace_fields {}.
Only include valid IDs drawn directly from the original packet."""

REPLAN_INSTRUCTIONS = """Return a complete plan in the original plan schema from the
same full supplied state. A new complete plan has been requested. Reconsider the
joint design in light of the supplied structured diagnostics and validate each row
identity, condition anchor and evidence reference."""


def messages(root, prompt, packet, *, full_metadata=False, legacy_reference=None):
    if prompt == "historical":
        return legacy_reference
    system = (root / STUDY_PATH / PROMPTS[prompt]).read_text().strip()
    if prompt == "v3_legacy" and full_metadata:
        system += "\n\n" + METADATA_ADDENDUM
    user = (legacy_reference[1]["content"] if legacy_reference is not None else dumps(packet))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
