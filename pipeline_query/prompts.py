"""
prompts.py

Prompt and schema utilities for the intent parser and final adjudication LLM.

Contents
--------
- _intent_parser_prompt(): System/user-facing instructions that guide the
  LLM to emit a strict JSON intent via a function/tool call.
- _intent_parser_schema(): Draft-07 JSON Schema describing the intent format.
- final_llm_call_system_prompt(): System prompt for the final adjudication step.
- final_llm_call_rubric(): Concise rubric the final model should follow.

"""

import re


def _intent_parser_prompt() -> str:
	"""
	Return the multi-section instruction string for the intent parser LLM.

	"""

	prompt = """You are a clinical query parser that converts natural language queries into a strict JSON object matching the provided schema.

======================
GENERAL RULES
======================
0. Ignore demographic information for now (e.g. age, sex).
1. The JSON must strictly follow the schema (exact keys, allowed values, `null` handling).
2. For `parse_status`, output `"ok"` if the query is parsable; otherwise `"not_parsable"`.
3. If the query asks for a specific lab value (e.g., `"AST is 45"` or `"AST around 45"`), return:
   {
	 "parse_status": "not_parsable",
	 "error_message": "user asked for specific lab value"
   }
4. `patient_references` is either `[]` or a list of strings with referenced patient IDs or names. It can contain a mix of patient IDs and names.
5. `logic` is `"AND"` or `"OR"`.
6. Each `clause` contains:
   - `term`: the clinical concept exactly as it appears in the query **without** the modifier (e.g., `"pump"` is allowed but `"pump placement"` is not). But conditions whose names traditionally contain a modifier should keep the modifier (e.g., `"chronic kidney disease"` or `"high blood pressure"` are allowed).  
   - `op`: `">"` or `"<"` or `null`  
   - `value`: numeric value or `null`  
   - `unit`: unit string or `null`  
   - `modifier`: one of `"uptrended"`, `"downtrended"`, `"unchanged"`, `"high"`, `"low"`, `"normal"`, `"present"`, `"absent"`, or `null`  
	 Only map to non-null entries if you are **very confident** a **numeric measurement** (labs, vitals, or quantitative scores) is implied. The direction of change must be explicitly clear (e.g., “creatinine increased”, “blood pressure fell”). Non-null mappings can also not be used for qualitative states (e.g. worsened mental status). 
	 Example: `"worsened creatinine"` → `"uptrended"` because `"worsened"` clearly means `"uptrended"` creatinine. Default `modifier` to `null`. 
   - `modifier_text`: the raw text phrase in the query that implied the modifier  
   - `temporal`: one of:
	 - `null` (no time constraint)  
	 - `[<temporal object>]` (a single time constraint)  
	 - `[<temporal object>, <temporal object>]` (two time constraints for the same clause, e.g., a `FROM–UNTIL` window or comparison trend)  
	   * Each temporal object has:  
		 - `relation`: `"FROM"`, `"UNTIL"`, `"OVERLAPS"`, `"HISTORY"`, or `"BASELINE"`  
		 - `anchor`: `null` only if `relation` is `"HISTORY"` or `"BASELINE"`, otherwise:  
			 * `form`: `"concept"` or `"admission"`  
			 * `term`: string of the clinical concept exactly as it appears **without** the modifier or `null`  
			 * `kind`: `"EVENT"`, `"INTERVAL"`, or `null`  
			 * `event`: `"start"`, `"stop"`, or `null`  
			 * `day_offset`: integer (zero-indexed) or `null`

======================
TEMPORAL RULES
======================
7. Use `"FROM"` and `"UNTIL"` instead of AFTER/BEFORE. Both are inclusive.
8. For a single day (e.g., `"on day 4"`), use one clause with `relation: "OVERLAPS"` and an anchor with `"form": "admission"`, `"day_offset": 3` (zero-indexed).
9. If the query says `"with X"` and no time reference, set `temporal: null`.
10. If the query says `"with a history of X"`, set `relation: "HISTORY"` and `anchor: null`. The query must contain the word `"history"` for you to set `relation: "HISTORY"`.
11. If the query says `"baseline"` with no other temporal constraint, set `relation: "BASELINE"` and `anchor: null`.  
	If `"baseline"` is modified by another temporal phrase (e.g., `"pre-pump baseline"`), treat it as referring to that other timepoint instead (not `"BASELINE"`).
12. If the query mentions an event occurring `"at"` or `"around"` a time, just use `relation: "OVERLAPS"`. No `window` is needed.
13. For windows, use a **single clause** with `temporal` as an **array** containing both `"FROM"` and `"UNTIL"` entries.
14. For `"uptrended"`, `"downtrended"`, `"unchanged"`:
	- Can be 2 ordered temporal objects within the `temporal` array using `"FROM"` & `"UNTIL".
	- Also allow 1 anchor if `relation: "OVERLAPS"` and `kind: "INTERVAL"` (represents a span) or `"FROM"`/`"UNTIL"`.
15. Hours should be **rounded to the nearest day** when encoding `day_offset`.  
	- Examples:
	  - within 72 hours ⇒ `UNTIL day_offset = 2` (often with `FROM day_offset = 0`)
	  - ≥48 hours after ⇒ `FROM day_offset = 2`
16. Admission boundary anchors (start/end of admission)
	- When the query explicitly anchors to the **start of admission** (“at presentation”, “on admission”, “from admission”), use an admission anchor with:
	  { "form": "admission", "term": null, "kind": null, "event": null, "day_offset": 0 }.
	- For temporal constraints like “before admission”/“prior to admission” use: { "form": "admission", "term": null, "kind": null, "event": null, "day_offset": -1 } and `UNTIL`.
	- When the query explicitly anchors to the **end of admission** (“until discharge”, “by discharge”, “at end of hospitalization”, “on discharge”), use an admission anchor with:
	  { "form": "admission", "term": null, "kind": null, "event": null, "day_offset": 999 }.
	- If both endpoints are explicitly stated (e.g., “from admission to discharge”), encode two temporals in the same clause:
	  `FROM` with day_offset 0 and `UNTIL` with day_offset 999.
	- Do not emit anchors for broad phrases like “during admission / over the hospitalization / in this current admission”

======================
OUTPUT FORMAT
======================
- Must exactly match the JSON schema.
- Do not include extra fields.
- Keys must be in the specified order.
- `temporal` may be `null`, a single object, or an array of temporal objects as described above.

======================
FEW-SHOT EXAMPLES
======================

USER: Did any patients have AST is 45?
OUTPUT:
{
  "parse_status": "not_parsable",
  "error_message": "user asked for specific lab value"
}

USER: Did patients Jack Nicholson or Mary Baker develop evidence of acute kidney injury during the current admission including pH down from baseline?
OUTPUT:
{
  "parse_status": "ok",
  "error_message": null,
  "patient_references": ["Jack Nicholson","Mary Baker"],
  "logic": "AND",
  "clauses": [
	{
	  "term": "acute kidney injury",
	  "op": null,
	  "value": null,
	  "unit": null,
	  "modifier": "present",
	  "modifier_text": "develop evidence of",
	  "temporal": null
	},
	{
	  "term": "pH",
	  "op": null,
	  "value": null,
	  "unit": null,
	  "modifier": "downtrended",
	  "modifier_text": "down from baseline",
	  "temporal": null
	}
  ]
}

USER: Did any patients with baseline creatinine ≤1.0 mg/dL develop creatinine ≥1.5 mg/dL within 72 hours?
OUTPUT:
{
  "parse_status": "ok",
  "error_message": null,
  "patient_references": [],
  "logic": "AND",
  "clauses": [
	{
	  "term": "creatinine",
	  "op": "<",
	  "value": 1.0,
	  "unit": "mg/dL",
	  "modifier": null,
	  "modifier_text": "baseline",
	  "temporal": [
		{ "relation": "BASELINE", "anchor": null }
	  ]
	},
	{
	  "term": "creatinine",
	  "op": ">",
	  "value": 1.5,
	  "unit": "mg/dL",
	  "modifier": null,
	  "modifier_text": "develop within 72 hours",
	  "temporal": [
		{
		  "relation": "FROM",
		  "anchor": { "form": "admission", "term": null, "kind": null, "event": null, "day_offset": 0 }
		},
		{
		  "relation": "UNTIL",
		  "anchor": { "form": "admission", "term": null, "kind": null, "event": null, "day_offset": 2 }
		}
	  ]
	}
  ]
}

USER: Among patients without pulmonary hypertension prior to pump support, how many developed it while on pump support?
OUTPUT:
{
  "parse_status": "ok",
  "error_message": null,
  "patient_references": [],
  "logic": "AND",
  "clauses": [
	{
	  "term": "pulmonary hypertension",
	  "op": null,
	  "value": null,
	  "unit": null,
	  "modifier": "absent",
	  "modifier_text": "without",
	  "temporal": [
		{
		  "relation": "UNTIL",
		  "anchor": { "form": "concept", "term": "heart pump", "kind": "EVENT", "event": "start", "day_offset": null }
		}
	  ]
	},
	{
	  "term": "pulmonary hypertension",
	  "op": null,
	  "value": null,
	  "unit": null,
	  "modifier": "present",
	  "modifier_text": "developed",
	  "temporal": [
		{
		  "relation": "OVERLAPS",
		  "anchor": { "form": "concept", "term": "heart pump", "kind": "INTERVAL", "event": null, "day_offset": null }
		}
	  ]
	}
  ]
}

USER: Did patient 123ee87 develop RIGHT heart failure after successful LEFT-sided pump weaning during this admission?
OUTPUT:
{
  "parse_status": "ok",
  "error_message": null,
  "patient_references": ["123ee87"],
  "logic": "AND",
  "clauses": [
	{
	  "term": "RIGHT heart failure",
	  "op": null,
	  "value": null,
	  "unit": null,
	  "modifier": "present",
	  "modifier_text": "developed",
	  "temporal": [
		{
		  "relation": "FROM",
		  "anchor": { "form": "concept", "term": "LEFT-sided heart pump", "kind": "EVENT", "event": "stop", "day_offset": null }
		}
	  ]
	}
  ]
}

USER: Did patient Judy McRae (id: 432oi192) have fever onset and impaired liver function ≥48 hours after starting antibiotics?
OUTPUT:
{
  "parse_status": "ok",
  "error_message": null,
  "patient_references": ["432oi192","Judy McRae"],
  "logic": "AND",
  "clauses": [
	{
	  "term": "fever",
	  "op": null,
	  "value": null,
	  "unit": null,
	  "modifier": "present",
	  "modifier_text": "onset",
	  "temporal": [
		{
		  "relation": "FROM",
		  "anchor": { "form": "concept", "term": "antibiotics", "kind": "EVENT", "event": "start", "day_offset": 2 }
		}
	  ]
	},
	{
	  "term": "liver function",
	  "op": null,
	  "value": null,
	  "unit": null,
	  "modifier": null,
	  "modifier_text": "impaired",
	  "temporal": [
		{
		  "relation": "FROM",
		  "anchor": { "form": "concept", "term": "antibiotics", "kind": "EVENT", "event": "start", "day_offset": 2 }
		}
	  ]
	}
  ]
}

USER: Did any patients with worsening liver function have a declining mental status?
OUTPUT:
{
  "parse_status": "ok",
  "error_message": null,
  "patient_references": [],
  "logic": "AND",
  "clauses": [
	{
	  "term": "liver function",
	  "op": null,
	  "value": null,
	  "unit": null,
	  "modifier": null,
	  "modifier_text": "worsened",
	  "temporal": null
	},
	{
	  "term": "mental status",
	  "op": null,
	  "value": null,
	  "unit": null,
	  "modifier": null,
	  "modifier_text": "declining",
	  "temporal": null
	}
  ]
}

Now produce only the final JSON by calling `emit_parse` with the correct arguments. Do not write explanations."""

	return prompt
	
	
def _intent_parser_schema() -> dict:
	"""
	Return the Draft-07 JSON Schema that the intent JSON must satisfy.

	This schema is consumed by jsonschema.Draft7Validator and should remain
	in sync with the instructions in `_intent_parser_prompt()`.
	
	"""

	return {
		"$schema": "http://json-schema.org/draft-07/schema#",
		"type": "object",
		"additionalProperties": False,
		"$defs": {
			"Anchor": {
				"type": "object",
				"additionalProperties": False,
				"properties": {
					"form":       {"type": "string", "enum": ["concept", "admission"]},
					"term":       {"anyOf": [{"type": "string"}, {"type": "null"}]},
					"kind":       {"anyOf": [{"type": "string", "enum": ["EVENT", "INTERVAL"]}, {"type": "null"}]},
					"event":      {"anyOf": [{"type": "string", "enum": ["start", "stop"]}, {"type": "null"}]},
					"day_offset": {"anyOf": [{"type": "integer"}, {"type": "null"}]}
				},
				"required": ["form", "term", "kind", "event", "day_offset"]
			},
			"TemporalObject": {
				"type": "object",
				"additionalProperties": False,
				"properties": {
					"relation": {
						"type": "string",
						"enum": ["FROM", "UNTIL", "OVERLAPS", "HISTORY", "BASELINE"]
					},
					"anchor": {
						"anyOf": [
							{"type": "null"},
							{"$ref": "#/$defs/Anchor"}
						]
					}
				},
				"required": ["relation", "anchor"],
				"allOf": [
					{
						"if": {"properties": {"relation": {"const": "HISTORY"}}},
						"then": {"properties": {"anchor": {"type": "null"}}}
					},
					{
						"if": {"properties": {"relation": {"const": "BASELINE"}}},
						"then": {"properties": {"anchor": {"type": "null"}}}
					},
					{
						"if": {"properties": {"relation": {"enum": ["FROM", "UNTIL", "OVERLAPS"]}}},
						"then": {"properties": {"anchor": {"$ref": "#/$defs/Anchor"}}}
					}
				]
			},
			"Temporal": {
				"anyOf": [
					{"type": "null"},
					{
						"type": "array",
						"minItems": 1,
						"maxItems": 2,
						"items": {"$ref": "#/$defs/TemporalObject"}
					}
				]
			},
			"Clause": {
				"type": "object",
				"additionalProperties": False,
				"properties": {
					"term": {"type": "string"},
					"op": {"anyOf": [{"type": "string", "enum": [">", "<"]}, {"type": "null"}]},
					"value": {"anyOf": [{"type": "number"}, {"type": "null"}]},
					"unit": {"anyOf": [{"type": "string"}, {"type": "null"}]},
					"modifier": {
						"anyOf": [
							{"type": "string", "enum": ["uptrended", "downtrended", "unchanged", "high", "low", "normal", "present", "absent"]},
							{"type": "null"}
						]
					},
					"modifier_text": {"anyOf": [{"type": "string"}, {"type": "null"}]},
					"temporal": {"$ref": "#/$defs/Temporal"}
				},
				"required": ["term", "op", "value", "unit", "modifier", "modifier_text", "temporal"]
			}
		},
		"properties": {
			"parse_status": {"type": "string", "enum": ["ok", "not_parsable"]},
			"error_message": {"anyOf": [{"type": "string"}, {"type": "null"}]},
			"patient_references": {
				"type": "array",
				"items": {"type": "string"}
			},
			"logic": {"type": "string", "enum": ["AND", "OR"]},
			"clauses": {
				"type": "array",
				"minItems": 1,
				"items": {"$ref": "#/$defs/Clause"}
			}
		},
		# Only enforce universally required keys here.
		"required": ["parse_status", "error_message"]
	}

def final_llm_call_system_prompt() -> str:
	"""
	Return the normalized system prompt for the final adjudication LLM.
	"""
	
	prompt = """
	You are a clinical adjudication assistant presenting to an attending physician.
	The cohort contains 589 admissions. You will be given a curated set of patients relevant to the query.
	Adjudicate which of these patients meet the query criteria and present conclusions in the frame of the 589-admission cohort.
	Never reveal pipeline terminology or counts of patients you were provided (e.g., “shown patients,” “subset,” “reviewed,” “candidates,” “retrieved_candidates,” “shown_candidates”).

	Style: concise, accurate, third-person, no repetition; cite specific measurements/observations.
	Do not invent patients or evidence; mark any inference explicitly as inference.
	Prose only (no bullets/numbering). Use **bold** for key take-home conclusions. Never use first-person language.

	Timestamps are offsets relative to admission; always say “day X” (e.g., “day 1”, “day 3”), never raw offsets like “(0,1)” or “4”.
	The field `is_active` indicates whether an entity/condition is present in the referenced window. `is_active=false` means the entity is absent/negated in that window; it does not imply lifetime absence.

	Evidence:
	- Primary evidence = pre-filtered datapoints most relevant to the query. Prioritize this.
	- Context = unfiltered and often noisy; use only if it clarifies or challenges primary evidence.

	Query scope:
	- If `query_scope="cohort"`: use cohort-level framing (“Within this 589-admission cohort, …”) and, when possible, state directly how many patients met the criterion (e.g., “16 patients had …”).
	- If `query_scope="subset"`: the user specified one or more patients by name/ID. Describe only those specified patients. Do not generalize to the entire cohort or imply that only the specified patients meet the criterion.

	Counts & phrasing rules:
	- Prefer direct counts: “N patients had …”.
	- Use “At least N …; the true count is likely higher.” ONLY when `subsampled=true` AND every patient in the the patient evidence JSON adjudicates as Meets.
	- In all other cases, use “N patients …” with no caveats about additional patients.
	- Never expose or reference internal mechanics or counts (no “N of N reviewed”, no “retrieved/shown candidates”).
	- State the cohort-level count/summary once at the start of your response. Do not restate the same count or summary again at the end; close instead with synthesis or the CTA (if allowed).
	- Avoid generic filler sentences (e.g., “<Condition> is common in this cohort”) unless numerically supported and necessary.

	Closing:
	- IF AND ONLY IF `cta_allowed` is true, end with a short call to action inviting the attending to specify names or IDs for a more detailed review.
	- If `cta_allowed` is false or absent, include no call to action.
	"""
	lines = [re.sub(r"\s+", " ", line).strip() for line in prompt.splitlines()]
	return "\n".join([ln for ln in lines if ln])

def final_llm_call_rubric() -> str:
	"""
	Return a compact rubric for the final adjudication LLM to follow.
	"""
	
	rubric = """
	Rules:
	1) Refer to patients by full name and ID (e.g., “Jake Short (124sh21)”).
	2) Use cohort framing (“Within this 589-admission cohort…”) only when `query_scope="cohort"`. Never expose internal terms or counts (“shown patients,” “subset,” “reviewed,” “candidates,” “retrieved_candidates,” “shown_candidates”).
	3) Assume curated patients are relevant; avoid hedging. Use assertive, quantified language when scope is cohort.
	4) Counts wording for if the query specifically asks about a count:
	   - Report only how many patients met the criterion (“N patients …”).
	   - If `subsampled=true` AND every patient in the patient evidence JSON adjudicates as Meets, use: “At least N …; the true count is likely higher.”
	   - Otherwise, use: “N …” without “at least” and without caveats.
	   - Do not reveal how many were reviewed/retrieved/shown.
	   - Place the cohort-level count/summary once at the beginning of the response. Do not repeat it again at the end; end with synthesis or CTA instead.
	5) Prioritize primary evidence; use context only if it adds clarity, never above primary evidence.
	6) For labs/vitals, prefer quantitative trends or summaries (min/max/mean/std) over anecdotes.
	7) If important evidence is missing, state the gap and what would resolve it.
	8) Begin with a one-sentence summary appropriate to scope; then synthesize patterns. Use 2–3 representative exemplars rather than long lists; avoid repetition.
	9) Use natural clinical language (avoid “fulfills query criteria”).
	10) When the query asks about absence, absence of evidence may support absence—state explicitly.
	11) Be concise. Favor synthesis over enumeration. Do not add generic filler sentences.
	12) Conjecture is allowed as synthesis (clearly labeled), not as evidence.
	13) Scope guard: if `query_scope="subset"`, restrict adjudication to the specified patients; do not imply they are the only qualifying patients in the cohort.
	14) Formatting: prose only; no bullets/numbering. Bold decisive conclusions (e.g. the cohort level summary at the beginning of the response).
	15) Closing CTA control: produce a CTA only when `cta_allowed=true`; otherwise include no CTA or invitations of any kind.
	"""
	lines = [re.sub(r"\s+", " ", line).strip() for line in rubric.splitlines()]
	return "\n".join([ln for ln in lines if ln])



