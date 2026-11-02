"""Read-only, trajectory-local catalog and stateless LLM selector.

The catalog is built from caller-supplied feature rows and *authorized* measured
rows.  It never opens the canonical target store, split files, or a repository
path.  Only the five named catalog operations are exposed to the model.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from ..training.predictor import atomic_json


PROMPT_VERSION = "full-pool-response-feedback-v1"
SYSTEM_PROMPT = """You select experiments to improve a 4g QGeoGNN V1/V2 predictor on a Row split.
Return exactly 16 distinct legal candidate IDs, each with a concise information-value reason.
The 16 pending CW experiments have no measured response. All 32 responses are revealed only
after the batch is frozen. Use only this trajectory's supplied measured records, feedback,
candidate catalog, and model predictions. Search before selecting. You may select any legal
candidate, including ones outside the 128 numeric reference IDs. Same-molecule condition
contrasts are welcome when informative; no new-molecule quota applies. Distinguish general
chemical prior, observed evidence, and untested inference. Relate reasons to possible model
generalization blind spots when justified; say when evidence is weak. V2-V1 is the elution
interval width in mL, not uncertainty. Predicted V1/V2 and measured V1/V2 are both in mL.
Initial L333 records have no out-of-sample prediction error. Never invent an executable ID.
Output JSON with packet_sha256 copied exactly from the supplied packet, choices,
hypotheses, and feedback_interpretation. choices is a list of 16 objects with id,
reason, and hypothesis_id (null allowed). hypotheses is a list of objects with id,
content, supporting_observed_ids, contradicting_observed_ids, evidence_strength
(weak/moderate/strong), and next_discriminating_question. Include updated assessments of
earlier hypotheses where relevant. feedback_interpretation explains what prior
measurement feedback changed in this round's search or choices; use an empty string
in round zero. No code, file, network, or external tool use is available.
"""

QUERY_BUDGET = 12
VIEW_BUDGET = 240
PER_QUERY_LIMIT = 24
MODEL_CALL_BUDGET = 16
INITIAL_VIEW_COUNT = 20


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _fingerprints(rows):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    smiles = {row["smiles"] for row in rows}
    mols = {value: Chem.MolFromSmiles(value) for value in smiles}
    if any(value is None for value in mols.values()):
        raise ValueError("invalid canonical SMILES in catalog")
    return mols, {value: generator.GetFingerprint(mol) for value, mol in mols.items()}


def _descriptors(mol):
    return {"MW_g_mol": round(Descriptors.MolWt(mol), 2),
            "LogP": round(Descriptors.MolLogP(mol), 2),
            "TPSA_A2": round(Descriptors.TPSA(mol), 2),
            "HBD": Descriptors.NumHDonors(mol),
            "HBA": Descriptors.NumHAcceptors(mol)}


class ReadOnlyCatalog:
    """Immutable search surface: candidates have no truth field, observed rows do."""

    def __init__(self, candidates: list[dict], observed: list[dict], pending: list[dict],
                 reference_ids: list[str], *, query_budget=QUERY_BUDGET,
                 view_budget=VIEW_BUDGET):
        self.candidates = {str(row["id"]): dict(row) for row in candidates}
        self.observed = {str(row["id"]): dict(row) for row in observed}
        self.pending = {str(row["id"]): dict(row) for row in pending}
        if len(self.candidates) != len(candidates) or len(self.observed) != len(observed):
            raise ValueError("duplicate catalog IDs")
        if set(self.candidates) & (set(self.observed) | set(self.pending)):
            raise ValueError("candidate/observed/pending overlap")
        if set(reference_ids) - set(self.candidates):
            raise ValueError("numeric reference contains illegal candidates")
        self.reference_ids = list(reference_ids)
        all_rows = candidates + observed + pending
        mols, self.fps = _fingerprints(all_rows)
        self.candidates = {key: {**row, "descriptors": _descriptors(mols[row["smiles"]])}
                           for key, row in self.candidates.items()}
        self.observed = {key: {**row, "descriptors": _descriptors(mols[row["smiles"]])}
                         for key, row in self.observed.items()}
        self.pending = {key: {**row, "descriptors": _descriptors(mols[row["smiles"]])}
                        for key, row in self.pending.items()}
        self.query_budget, self.view_budget = int(query_budget), int(view_budget)
        self.queries: list[dict] = []
        self.viewed: set[str] = set()

    def overview(self) -> dict:
        pool = list(self.candidates.values())
        fields = ("PE/EA", "Density g/ml", "V/ul", "loading solvent", "Volume of loading solvent/ul")
        summaries = {}
        for field in fields:
            values = [row["conditions"][field] for row in pool]
            numeric = [_number(value) for value in values]
            if all(value is not None for value in numeric):
                summaries[field] = {"min": min(numeric), "max": max(numeric)}
            else:
                summaries[field] = {"values": sorted(set(str(value) for value in values))[:30]}
        return {"eligible_count": len(pool), "eligible_distinct_molecules": len({r["smiles"] for r in pool}),
                "observed_count": len(self.observed), "pending_CW_count": len(self.pending),
                "numeric_reference_ids": self.reference_ids, "condition_summary": summaries,
                "query_budget": self.query_budget, "view_budget": self.view_budget,
                "per_query_limit": PER_QUERY_LIMIT}

    def initial_cards(self) -> list[dict]:
        ids = self.reference_ids[:INITIAL_VIEW_COUNT]
        self.viewed.update(ids)
        return [self.candidates[value] for value in ids]

    def _filtered(self, rows: dict[str, dict], args: dict) -> list[dict]:
        result = list(rows.values())
        if args.get("smiles"):
            query = str(args["smiles"])
            result = [row for row in result if query in row["smiles"]]
        if args.get("same_molecule_as"):
            ref = self.candidates.get(str(args["same_molecule_as"])) or self.observed.get(str(args["same_molecule_as"]))
            result = [row for row in result if ref and row["smiles"] == ref["smiles"]]
        if args.get("similar_to"):
            ref = self.candidates.get(str(args["similar_to"])) or self.observed.get(str(args["similar_to"]))
            if not ref:
                return []
            fp = self.fps[ref["smiles"]]
            result = sorted(result, key=lambda row: DataStructs.TanimotoSimilarity(fp, self.fps[row["smiles"]]), reverse=True)
            floor = _number(args.get("min_tanimoto")) or 0.0
            result = [row for row in result if DataStructs.TanimotoSimilarity(fp, self.fps[row["smiles"]]) >= floor]
        if args.get("conditions"):
            filters = args["conditions"]
            if not isinstance(filters, dict):
                raise ValueError("conditions must be an object")
            for field, wanted in filters.items():
                if field not in ("PE/EA", "Density g/ml", "V/ul", "loading solvent", "Volume of loading solvent/ul"):
                    raise ValueError("unknown condition field")
                result = [row for row in result if str(row["conditions"][field]) == str(wanted)]
        if args.get("descriptor"):
            field = str(args["descriptor"])
            if field not in ("MW_g_mol", "LogP", "TPSA_A2", "HBD", "HBA"):
                raise ValueError("unknown descriptor")
            low, high = _number(args.get("min_value")), _number(args.get("max_value"))
            result = [row for row in result if (low is None or row["descriptors"][field] >= low)
                      and (high is None or row["descriptors"][field] <= high)]
        return result

    def query(self, operation: str, args: dict) -> dict:
        if len(self.queries) >= self.query_budget:
            return {"error": "query_budget_exhausted", "remaining": 0}
        if not isinstance(args, dict):
            raise ValueError("query args must be object")
        limit = min(PER_QUERY_LIMIT, max(1, int(args.get("limit", PER_QUERY_LIMIT))))
        if operation == "get_candidates":
            ids = args.get("ids", [])
            if not isinstance(ids, list):
                raise ValueError("ids must be a list")
            found = [self.candidates[str(value)] for value in ids[:limit] if str(value) in self.candidates]
        elif operation == "search_candidates":
            found = self._filtered(self.candidates, args)[:limit]
        elif operation == "search_observed":
            found = self._filtered(self.observed, args)[:limit]
        elif operation == "get_observed":
            ids = args.get("ids", [])
            if not isinstance(ids, list):
                raise ValueError("ids must be a list")
            found = [self.observed[str(value)] for value in ids[:limit] if str(value) in self.observed]
        else:
            raise ValueError("unknown catalog operation")
        if operation.endswith("candidates"):
            new = [row for row in found if row["id"] not in self.viewed]
            available = max(0, self.view_budget - len(self.viewed))
            if len(new) > available:
                allowed = {row["id"] for row in new[:available]} | self.viewed
                found = [row for row in found if row["id"] in allowed]
            self.viewed.update(row["id"] for row in found)
        response = {"matches": found, "returned_count": len(found),
                    "remaining_queries": self.query_budget - len(self.queries) - 1,
                    "remaining_new_candidate_views": self.view_budget - len(self.viewed)}
        self.queries.append({"operation": operation, "arguments": args, "response": response})
        return response


def validate_selection(response: dict, catalog: ReadOnlyCatalog, *, packet_sha256=None,
                       require_feedback=False) -> dict:
    if not isinstance(response, dict) or not isinstance(response.get("choices"), list):
        raise ValueError("selector response must contain choices list")
    if packet_sha256 is not None and response.get("packet_sha256") != packet_sha256:
        raise ValueError("selector response is not bound to this packet")
    choices = response["choices"]
    ids = [choice.get("id") for choice in choices if isinstance(choice, dict)]
    if len(choices) != 16 or len(ids) != 16 or len(set(ids)) != 16 or not set(ids) <= set(catalog.candidates):
        raise ValueError("must select 16 distinct legal non-CW candidate IDs")
    if any(not isinstance(choice.get("reason"), str) or not choice["reason"].strip() for choice in choices):
        raise ValueError("each selection needs a reason")
    hypotheses = response.get("hypotheses", [])
    if not isinstance(hypotheses, list):
        raise ValueError("hypotheses must be a list")
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, dict) or not all(key in hypothesis for key in
                ("id", "content", "supporting_observed_ids", "contradicting_observed_ids", "evidence_strength", "next_discriminating_question")):
            raise ValueError("incomplete structured hypothesis")
        for key in ("supporting_observed_ids", "contradicting_observed_ids"):
            if not isinstance(hypothesis[key], list) or not set(hypothesis[key]) <= set(catalog.observed):
                raise ValueError("hypothesis cites unobserved ID")
        if hypothesis["evidence_strength"] not in ("weak", "moderate", "strong"):
            raise ValueError("invalid evidence strength")
    interpretation = response.get("feedback_interpretation", "")
    if not isinstance(interpretation, str) or (require_feedback and not interpretation.strip()):
        raise ValueError("missing subsequent-feedback interpretation")
    return {"choices": choices, "hypotheses": hypotheses,
            "feedback_interpretation": interpretation}


TOOL = {"type": "function", "name": "query_catalog",
        "description": "Search the eligible full pool or this trajectory's measured records. No hidden responses or filesystem access.",
        "parameters": {"type": "object", "properties": {
            "operation": {"type": "string", "enum": ["get_candidates", "search_candidates", "get_observed", "search_observed"]},
            "args": {"type": "object", "description": "Optional: ids, smiles, same_molecule_as, similar_to, min_tanimoto, conditions, descriptor, min_value, max_value, limit"}},
            "required": ["operation", "args"], "additionalProperties": False}}


def configured() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def run_openai_selector(packet: dict, catalog: ReadOnlyCatalog, directory: Path,
                        model: str, reasoning_effort: str, attempt: int = 0) -> dict:
    """One independent stateless Responses conversation; persist each model/tool turn."""
    if not configured():
        raise RuntimeError("OPENAI_API_KEY is absent; independent Responses API selector unavailable")
    from openai import OpenAI

    directory = Path(directory)
    transcript_path = directory / f"selector_attempt_{attempt:02d}.json"
    if transcript_path.exists():
        raise RuntimeError("selector transcript already exists; inspect before retry")
    client = OpenAI(max_retries=0, timeout=180)
    messages: list[Any] = [{"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user", "content": json.dumps(packet, ensure_ascii=False)}]
    transcript = {"model": model, "reasoning_effort": reasoning_effort,
                  "prompt_version": PROMPT_VERSION, "input": packet, "turns": [],
                  "query_log": [], "viewed_candidate_ids": sorted(catalog.viewed),
                  "usage": {"input_tokens": 0, "output_tokens": 0}}
    atomic_json(transcript_path, transcript)
    for turn in range(MODEL_CALL_BUDGET):
        allow_tools = len(catalog.queries) < catalog.query_budget and turn < MODEL_CALL_BUDGET - 1
        kwargs = dict(model=model, input=messages, reasoning={"effort": reasoning_effort},
                      tools=[TOOL] if allow_tools else [], store=False,
                      include=["reasoning.encrypted_content"])
        response = client.responses.create(**kwargs)
        output = [item.model_dump(mode="json") for item in response.output]
        usage = response.usage
        transcript["turns"].append({"response_id": response.id, "output": output,
                                    "usage": usage.model_dump(mode="json") if usage else None})
        if usage:
            transcript["usage"]["input_tokens"] += usage.input_tokens
            transcript["usage"]["output_tokens"] += usage.output_tokens
        calls = [item for item in response.output if item.type == "function_call"]
        if calls:
            if not allow_tools:
                transcript["error"] = "tool_call_after_budget"
                atomic_json(transcript_path, transcript)
                raise RuntimeError("selector exceeded model/tool budget")
            messages.extend(output)
            for call in calls:
                if call.name != "query_catalog":
                    raise RuntimeError("unregistered selector tool")
                try:
                    arguments = json.loads(call.arguments)
                    result = catalog.query(arguments["operation"], arguments["args"])
                except (ValueError, KeyError, TypeError) as exc:
                    result = {"error": str(exc)}
                messages.append({"type": "function_call_output", "call_id": call.call_id,
                                 "output": json.dumps(result, ensure_ascii=False)})
            transcript["query_log"] = catalog.queries
            transcript["viewed_candidate_ids"] = sorted(catalog.viewed)
            atomic_json(transcript_path, transcript)
            continue
        transcript["final_text"] = response.output_text
        transcript["query_log"] = catalog.queries
        transcript["viewed_candidate_ids"] = sorted(catalog.viewed)
        atomic_json(transcript_path, transcript)
        return json.loads(response.output_text)
    transcript["error"] = "model_call_budget_exhausted"
    atomic_json(transcript_path, transcript)
    raise RuntimeError("selector did not finish within fixed model-call budget")
