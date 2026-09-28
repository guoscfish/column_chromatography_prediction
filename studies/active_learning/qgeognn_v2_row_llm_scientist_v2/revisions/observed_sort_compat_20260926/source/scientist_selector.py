"""V2 scientific planning surface. No paths, label store, or predictor access."""
from __future__ import annotations

import copy
import math

from rdkit import DataStructs

from .full_pool_selector import ReadOnlyCatalog as V1Catalog, _number
from .protocol import stable_hash

VERSION = "llm_scientist_v2"
QUERY_BUDGET = 24
VIEW_BUDGET = 480
MODEL_CALL_BUDGET = 28
PER_QUERY_LIMIT = 24
MAX_HYPOTHESES = 8
CONDITIONS = ("PE/EA", "Density g/ml", "V/ul", "loading solvent", "Volume of loading solvent/ul")
DESCRIPTORS = ("MW_g_mol", "LogP", "TPSA_A2", "HBD", "HBA")
PREDICTIONS = tuple(f"pred_{t}_{q}_ml" for t in ("V1", "V2") for q in ("q10", "q50", "q90"))
SIGNALS = ("pred_V1_ml", "pred_V2_ml", "cw_distance_percentile",
           "V1_quantile_width_ml", "V2_quantile_width_ml", "molecule_novelty",
           "condition_novelty", "max_similarity_to_observed", "max_similarity_to_pending",
           "historical_abs_error_ml", "loading_amount_mg", "similarity", *PREDICTIONS)
OBSERVED_FIELDS = ("V1_ml", "V2_ml", "record_kind", "premeasurement_prediction_V1_ml",
                   "premeasurement_prediction_V2_ml", "premeasurement_error_V1_ml",
                   "premeasurement_error_V2_ml")

SYSTEM_PROMPT = """You are a scientific experimental planner / LLM scientist for active learning.

SCIENTIFIC SETTING
The data describe 4 g silica normal-phase flash column chromatography. Each experiment
is an organic molecule (canonical SMILES) together with its measured experimental
conditions: PE/EA, Density g/ml, V/ul, loading solvent, and Volume of loading solvent/ul.
PE means petroleum ether; EA means ethyl acetate. Silica is the normal-phase stationary
phase. PE/EA is the recorded PE/EA ratio (a/b); it changes mobile-phase elution strength.
Density and V/ul describe sample loading. loading_amount_mg is their product, matching
the predictor's density-times-volume loading feature, not a separately measured label.
Do not assume additional concentration, pH, flow, or column variables absent from the
catalog. Polarity, H-bond donors/acceptors, functional groups,
aromaticity, heteroatoms, acid/base character, steric/structural environment, and loading
conditions may affect retention. These are useful chemical priors, not absolute rules;
confront them with the supplied observations and acknowledge missing information.

PREDICTION TARGET AND OBJECTIVE
V1 is the eluent volume when the target compound starts eluting. V2 is the eluent volume
when it has substantially finished eluting. Both are in mL. V2 - V1 is the elution
interval width, NOT uncertainty. QGeoGNN predicts V1 and V2.
The goal is NOT to optimize chromatographic performance of the selected experiments
themselves. The goal is to choose experiments whose measured labels are expected to be
most useful for improving the retrained QGeoGNN's overall V1/V2 predictive accuracy on
the Row-distribution. After measurement, the predictor is retrained from the same
initialization. Do not seek maximal V1/V2 or directly optimize separation conditions.
Consider what labels could teach about uncovered relationships, model failure regions,
structure-condition-retention relationships, and patterns that generalize to other rows.

YOUR SCIENTIFIC JUDGMENT
You may actively use your chemistry/chromatography knowledge, including structural
analogies, silica interactions, solvent strength, same-molecule condition-response
curves, surprising predictions, related compounds behaving differently, and loading
effects. You may use your ML/active-learning knowledge, including exploration versus
exploitation, coverage, uncertainty, residuals, novelty, redundancy, batch diversity,
model sensitivity, information gain, hypothesis discrimination and learning curves.
These are possible considerations, not a checklist and not an acquisition formula.
Decide for yourself what evidence is most relevant in the current round.
Numerical signals are evidence, not mandatory rules. You may disagree with a numerical
heuristic when you have a scientifically plausible reason, but explain the reason and
distinguish it from observed evidence. Distinguish general chemical prior, observed
evidence, model prediction, and untested hypothesis; never present a prior as a finding.
Consider redundancy and diversity, but do not pursue diversity for its own sake.
Multiple experiments around the same molecule or chemical family are acceptable when
they form a scientifically meaningful local experiment, test a response curve, or
discriminate between competing hypotheses. No molecule cap, new-molecule quota, fixed
exploration ratio, uncertainty quota, or numerical-strategy quota applies.

EVIDENCE AND FEEDBACK
Only use this seed AND method's supplied trajectory. L_t records are measured;
candidates and pending experiments have predictions only. Pending experiments can be
used as references but cannot be selected again. Never access validation/test truth,
other seeds, other methods' outcomes, old research results, files, shell, web, or native
tools. All 32 IDs and premeasurement predictions must freeze before ANY new label reveal.
The required LLM count is packet.selection_count: 16 supplements to the pending 16 in
the hybrid arm, or all 32 with no pending experiments in the free arm.
Previous-round feedback: These are outcomes of experiments you previously selected.
Use them to update, support, weaken, or reject your previous hypotheses and to guide
what should be measured next. Pending-CW outcomes are identified separately.
Only recent raw feedback is repeated. Older measured records remain queryable.
Scientific memory contains your own previous interpretation and hypotheses, exact
high-error examples and previous selected IDs; it is not an algorithmic scientific
summary. Retain only useful hypotheses, including contradicted ones when informative.
Initial L333 observations have no out-of-sample error. Historical errors are the signed
prediction-minus-measurement errors at acquisition time, not current-model residuals.
Quantile widths are q90 minus q10 from the existing single predictor; they are not
ensemble epistemic uncertainty and need not be calibrated. CW distance is a percentile
of distance to labeled rows in the current center/width gradient feature space.
Molecule novelty is 1 for a SMILES absent from L_t; condition novelty is 1 when its
molecule-plus-condition tuple is absent. Similarity is Morgan-radius-2 Tanimoto.

READ-ONLY JSON RELAY
Every answer must be one JSON object, without Markdown or commentary. Do not invoke
native tools. To request data return {"type":"query","queries":[{"operation":
"search_candidates","args":{...}}]}. The host returns exact catalog records.
Operations: get_candidates, get_observed, get_pending (ids list); search_candidates,
search_observed, search_pending. References same_molecule_as and similar_to may identify
a candidate, observed, or pending row. Filters: smiles substring; conditions exact
field-value object; descriptor (MW_g_mol, LogP, TPSA_A2, HBD, HBA) with min_value/max_value;
similar_to with min_tanimoto; signal with min_signal/max_signal.
sort_by accepts those descriptors, pred_V1_ml, pred_V2_ml, pred_V1_q10_ml through
pred_V2_q90_ml, V1_quantile_width_ml, V2_quantile_width_ml, cw_distance_percentile,
molecule_novelty, condition_novelty, max_similarity_to_observed,
max_similarity_to_pending, loading_amount_mg, historical_abs_error_ml (observed only), or similarity
(requires similar_to). order is asc or desc. Missing signals sort last. Default search
order is a reproducible round-specific hash of IDs, independent of dataframe order;
similar_to defaults to decreasing similarity. offset enables paging; limit is at most
24. Returned total_matches describes the entire filtered set. No matches means empty
results, not permission to invent an experiment. At most 24 queries, 480 distinct
candidate views (including 20 initial cards), and 28 answers per round. Search before
selection. You can access any U_t candidate; there is no numerical shortlist.

FINAL SCIENTIFIC PLAN
Return {"type":"selection","packet_sha256":"copy exactly","batch_strategy":"...",
"choices":[{"id":"...","reason":"...","hypothesis_id":null}],"hypotheses":[],
"feedback_interpretation":"...","batch_rationale":"..."}.
Choose exactly selection_count distinct legal, viewed candidate IDs. Each reason
explains expected learning value with evidence IDs when available. hypothesis_id must
be null or reference a hypothesis in this answer. Zero to eight important hypotheses
are allowed; do not manufacture hypotheses. Each has id, content, basis
(chemical_prior/observed_data/model_behavior/mixed), supporting_observed_ids,
contradicting_observed_ids, confidence (low/medium/high), status
(proposed/supported/weakened/rejected/unresolved), why_it_matters_for_model_learning,
and next_discriminating_question. Cite only observed IDs as measured evidence.
feedback_interpretation explains how previous outcomes changed your reasoning; use an
empty string at round zero. batch_rationale must explain why the batch as a whole is
informative, the role of failure repair/exploration/condition-response experiments or
other considerations you chose, and why this may improve generalization across the
Row-distribution. No fixed composition or acquisition formula is required.
"""


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


class ReadOnlyCatalog(V1Catalog):
    def __init__(self, candidates, observed, pending, *, salt, query_budget=QUERY_BUDGET,
                 view_budget=VIEW_BUDGET):
        super().__init__([clean_card(r) for r in candidates],
                         [clean_card(r, True) for r in observed],
                         [clean_card(r) for r in pending], [],
                         query_budget=query_budget, view_budget=view_budget)
        if len(self.pending) != len(pending) or set(self.pending) & set(self.observed):
            raise ValueError("duplicate or overlapping pending IDs")
        self.salt = salt
        self.references = {**self.candidates, **self.observed, **self.pending}
        obs_mols = {r["smiles"] for r in self.observed.values()}
        obs_conditions = {(r["smiles"], stable_hash(r["conditions"])) for r in self.observed.values()}
        obs_fps = [self.fps[s] for s in sorted(obs_mols)]
        pending_fps = [self.fps[s] for s in sorted({r["smiles"] for r in self.pending.values()})]
        similarity = {}
        for smiles, fp in self.fps.items():
            similarity[smiles] = (
                max(DataStructs.BulkTanimotoSimilarity(fp, obs_fps), default=0.0),
                max(DataStructs.BulkTanimotoSimilarity(fp, pending_fps), default=0.0))
        for row in self.references.values():
            row["loading_amount_mg"] = float(row["conditions"]["Density g/ml"]) * float(row["conditions"]["V/ul"])
            row["molecule_novelty"] = int(row["smiles"] not in obs_mols)
            row["condition_novelty"] = int((row["smiles"], stable_hash(row["conditions"])) not in obs_conditions)
            row["max_similarity_to_observed"], row["max_similarity_to_pending"] = similarity[row["smiles"]]
            for target in ("V1", "V2"):
                lo, hi = row.get(f"pred_{target}_q10_ml"), row.get(f"pred_{target}_q90_ml")
                if lo is not None and hi is not None:
                    row[f"{target}_quantile_width_ml"] = hi - lo
            if "premeasurement_error_V1_ml" in row:
                row["historical_abs_error_ml"] = max(abs(row[f"premeasurement_error_{t}_ml"]) for t in ("V1", "V2"))

    def key(self, row):
        return stable_hash([self.salt, row["id"]])

    def initial_cards(self):
        rows = sorted(self.candidates.values(), key=self.key)[:min(20, self.view_budget)]
        self.viewed.update(r["id"] for r in rows)
        return copy.deepcopy(rows)

    def overview(self):
        result = super().overview()
        result.pop("numeric_reference_ids")
        result["default_order"] = "round_salted_ID_hash"
        return result

    def _filtered(self, rows, args):
        # V1's chemical filters are reused with a combined reference lookup, including pending.
        filtering = {k: v for k, v in args.items() if k not in ("same_molecule_as", "similar_to")}
        result = super()._filtered(rows, filtering)
        for name in ("same_molecule_as", "similar_to"):
            if name not in args:
                continue
            ref = self.references.get(str(args[name]))
            if ref is None:
                return []
            if name == "same_molecule_as":
                result = [r for r in result if r["smiles"] == ref["smiles"]]
            else:
                result = [{**r, "similarity": DataStructs.TanimotoSimilarity(
                    self.fps[ref["smiles"]], self.fps[r["smiles"]])} for r in result]
                result = [r for r in result if r["similarity"] >= float(args.get("min_tanimoto", 0))]
        if "signal" in args:
            name = args["signal"]
            self._check_signal(name, rows, args)
            lo, hi = float(args.get("min_signal", -math.inf)), float(args.get("max_signal", math.inf))
            def signal_value(row):
                return _number(row["descriptors"].get(name) if name in DESCRIPTORS else row.get(name))
            result = [r for r in result if signal_value(r) is not None and lo <= signal_value(r) <= hi]
        result.sort(key=self.key)
        field = args.get("sort_by", "similarity" if "similar_to" in args else None)
        if field:
            self._check_signal(field, rows, args)
            def value(row):
                return _number(row["descriptors"].get(field) if field in DESCRIPTORS else row.get(field))
            known = [r for r in result if value(r) is not None]
            missing = [r for r in result if value(r) is None]
            result = sorted(known, key=value, reverse=args.get("order", "desc") == "desc") + missing
        return result

    def _check_signal(self, field, rows, args):
        if field not in (*SIGNALS, *DESCRIPTORS):
            raise ValueError("unknown sort/signal field")
        if field == "historical_abs_error_ml" and rows is not self.observed:
            raise ValueError("historical error is observed-only")
        if field == "similarity" and "similar_to" not in args:
            raise ValueError("similarity requires similar_to")

    def query(self, operation, args):
        if len(self.queries) >= self.query_budget:
            raise ValueError("query budget exhausted")
        if not isinstance(args, dict) or set(args) - {"ids", "smiles", "same_molecule_as", "similar_to",
                "min_tanimoto", "conditions", "descriptor", "min_value", "max_value", "signal",
                "min_signal", "max_signal", "sort_by", "order", "offset", "limit"}:
            raise ValueError("unknown query arguments")
        if args.get("order", "desc") not in ("asc", "desc"):
            raise ValueError("order must be asc or desc")
        offset, limit = int(args.get("offset", 0)), int(args.get("limit", 24))
        if offset < 0 or not 1 <= limit <= 24:
            raise ValueError("invalid pagination")
        tables = {"candidates": self.candidates, "observed": self.observed, "pending": self.pending}
        parts = operation.split("_", 1)
        if len(parts) != 2 or parts[0] not in ("get", "search") or parts[1] not in tables:
            raise ValueError("unknown catalog operation")
        rows = tables[parts[1]]
        if parts[0] == "get":
            ids = args.get("ids", [])
            if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
                raise ValueError("ids must be a string list")
            found = [rows[i] for i in dict.fromkeys(ids) if i in rows]
        else:
            found = self._filtered(rows, args)
        total = len(found)
        found = found[offset:offset+limit]
        limited = False
        if rows is self.candidates:
            accepted = []
            for row in found:
                if row["id"] in self.viewed or len(self.viewed) < self.view_budget:
                    self.viewed.add(row["id"])
                    accepted.append(row)
                else:
                    limited = True
            found = accepted
        response = {"matches": copy.deepcopy(found), "total_matches": total, "returned_count": len(found),
                    "next_offset": offset+limit if offset+limit < total else None,
                    "view_budget_limited": limited, "remaining_queries": self.query_budget-len(self.queries)-1,
                    "remaining_new_candidate_views": self.view_budget-len(self.viewed)}
        self.queries.append(copy.deepcopy({"operation": operation, "args": args, "response": response}))
        return response


def validate_selection(response, catalog, packet):
    if response.get("type") != "selection" or response.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("selection packet binding mismatch")
    choices, hypotheses = response.get("choices"), response.get("hypotheses")
    if not isinstance(choices, list) or len(choices) != packet["selection_count"]:
        raise ValueError("wrong selection count")
    ids = [c.get("id") for c in choices if isinstance(c, dict)]
    if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(choices) or not set(ids) <= catalog.viewed:
        raise ValueError("must select distinct legal viewed IDs")
    if not set(ids) <= set(catalog.candidates) or not catalog.queries:
        raise ValueError("selection requires legal candidates and a query")
    if not isinstance(hypotheses, list) or len(hypotheses) > MAX_HYPOTHESES:
        raise ValueError("hypotheses must contain zero to eight entries")
    seen = set()
    for h in hypotheses:
        if not isinstance(h, dict):
            raise ValueError("hypothesis must be an object")
        for field in ("id", "content", "why_it_matters_for_model_learning", "next_discriminating_question"):
            if not isinstance(h.get(field), str) or not h[field].strip():
                raise ValueError("incomplete hypothesis")
        if h["id"] in seen:
            raise ValueError("duplicate hypothesis ID")
        seen.add(h["id"])
        if h.get("basis") not in ("chemical_prior", "observed_data", "model_behavior", "mixed"):
            raise ValueError("invalid hypothesis basis")
        if h.get("confidence") not in ("low", "medium", "high") or h.get("status") not in (
                "proposed", "supported", "weakened", "rejected", "unresolved"):
            raise ValueError("invalid hypothesis confidence/status")
        for field in ("supporting_observed_ids", "contradicting_observed_ids"):
            evidence = h.get(field)
            if not isinstance(evidence, list) or any(not isinstance(i, str) for i in evidence) or not set(evidence) <= set(catalog.observed):
                raise ValueError("hypothesis cites unobserved evidence")
    for c in choices:
        if not isinstance(c.get("reason"), str) or not c["reason"].strip():
            raise ValueError("each choice needs a reason")
        if "hypothesis_id" not in c or (c["hypothesis_id"] is not None and c["hypothesis_id"] not in seen):
            raise ValueError("invalid hypothesis reference")
    for field in ("batch_strategy", "batch_rationale", "feedback_interpretation"):
        if not isinstance(response.get(field), str) or (field != "feedback_interpretation" or packet["round"] > 0) and not response[field].strip():
            raise ValueError(f"missing {field}")
    if len(str(response)) > 30000:
        raise ValueError("scientific plan exceeds memory size bound")
    return copy.deepcopy(response)


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
