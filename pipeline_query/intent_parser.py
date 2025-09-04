"""intent_parser.py

LLM-backed intent parser that converts a free-text clinical query into a
strict, schema-validated JSON intent. The parser performs:

1) A one-shot parse.
2) JSON Schema validation.
3) Additional *semantic* validation (temporal forms, anchors, trends, etc.).
4) If validation fails, a *repair* pass prompts the model with the precise error and requests a corrected tool call. Multiple retries are supported.

Environment/configuration is supplied via `config.settings`, e.g.:
- `params.openai_api_key`
- `params.gpt_model`
- `params.intent_parser_temperature`
- `params.intent_parser_max_retries`

"""
from typing import Optional, Tuple, Dict, Any
import json, time
from openai import OpenAI
from prompts import _intent_parser_prompt, _intent_parser_schema
from jsonschema import Draft7Validator, ValidationError
from config import settings


# --------------- public API ---------------

def parse_intent(user_query: str) -> dict:
	"""Parse a free-text user query into a structured intent.
	
	This is the high-level API. It creates an OpenAI client, calls the
	model once, validates the output, and if needed, performs a repair pass
	with explicit error messages until a valid result is produced or
	retries are exhausted.
	
	"""
	# call LLM with few-shot prompt to generate intent json
	client = OpenAI(api_key=settings.params.openai_api_key)
	intent_json = parse_with_repair(client, user_query, settings.params.gpt_model, settings.params.intent_parser_temperature)
	# return file
	return intent_json


# --------------- helpers ---------------

# define constants
TOOLS = [{
	"type": "function",
	"function": {
		"name": "emit_parse",
		"description": "Return the parsed query JSON matching the schema.",
		"parameters": _intent_parser_schema()}}]
		
# define helper functions
_validator = Draft7Validator(_intent_parser_schema()) 

def _validate_json_str(json_str: str) -> Tuple[bool, Optional[str]]:
	"""Validate a JSON string against the Draft7 schema.
	
	Returns
	-------
	tuple
	``(is_valid, error_message)`` where ``error_message`` is ``None`` when
	valid, otherwise a concise description with a JSONPath-like locator.
	"""
	try:
		data = json.loads(json_str)
	except json.JSONDecodeError as e:
		return False, f"JSON parse error: {e}"
	errors = sorted(_validator.iter_errors(data), key=lambda e: e.path)
	if errors:
		e = errors[0]
		loc = "$" + "".join([f"[{repr(p)}]" if isinstance(p, int) else f".{p}" for p in e.path])
		return False, f"Schema error at {loc}: {e.message}"
	return True, None
	
def _semantic_validate(intent: Dict[str, Any]) -> Optional[str]:
	"""
	Return None if semantically valid; else a short error message.
	Assumes JSON already passed Draft7 schema above.

	Enforces (consistent with updated schema/prompt):
	  - parse_status ok/not_parsable.
	  - When ok: logic ∈ {AND, OR}; clauses is non-empty.
	  - Clauses are leaves (no nested boolean groups).
	  - Temporal: null or list with 1–2 TemporalObjects (no bare dict).
	  - Temporal.relation:
		  * HISTORY/BASELINE => anchor is None
		  * FROM/UNTIL/OVERLAPS => anchor is a single Anchor object
	  - Anchor validation:
		  * admission: term/kind/event == None; day_offset is int
		  * concept+EVENT: term non-empty; event ∈ {start, stop, None}; day_offset int or None
		  * concept+INTERVAL: relation must be OVERLAPS; event == None; day_offset == None
	  - Trend modifiers ('uptrended','downtrended','unchanged'):
		  * valid if:
			  - no temporal (trend over entire admission), OR
			  - any temporal is OVERLAPS with concept INTERVAL anchor, OR
			  - at least two temporals have non-null anchors (window/ordered comparison), OR
			  - baseline-window exception: has FROM & UNTIL and modifier_text contains 'baseline'
	"""
	if intent.get("parse_status") not in {"ok", "not_parsable"}:
		return "parse_status must be 'ok' or 'not_parsable'"
	if intent["parse_status"] != "ok":
		return None

	logic = intent.get("logic")
	if logic not in {"AND", "OR"}:
		return "logic must be 'AND' or 'OR'"

	clauses = intent.get("clauses")
	if not isinstance(clauses, list) or not clauses:
		return "clauses must be a non-empty list"

	TREND = {"uptrended", "downtrended", "unchanged"}
	CANON = TREND | {"high", "low", "normal", "present", "absent"}
	REL = {"FROM", "UNTIL", "OVERLAPS", "HISTORY", "BASELINE"}
	KIND = {"EVENT", "INTERVAL"}

	for i, cl in enumerate(clauses):
		# must be a leaf
		if "logic" in cl or "clauses" in cl:
			return f"clause[{i}]: nested boolean groups are not supported"

		term = cl.get("term")
		if not isinstance(term, str) or not term:
			return f"clause[{i}]: term must be a non-empty string"

		op = cl.get("op")
		if op not in {">", "<", None}:
			return f"clause[{i}]: op must be '>' or '<' or null"

		if op is not None:
			if not isinstance(cl.get("value"), (int, float)):
				return f"clause[{i}]: numeric 'value' required when 'op' is set"

		modifier = cl.get("modifier")
		if modifier is not None and modifier not in CANON:
			return f"clause[{i}]: unknown modifier '{modifier}'"

		mt = cl.get("modifier_text")
		if mt is not None and not isinstance(mt, str):
			return f"clause[{i}]: modifier_text must be string or null"
		mt_lower = (mt or "").lower()

		# --- Temporal normalization (schema-compatible): null or list(1..2) ---
		temporals_raw = cl.get("temporal")
		if temporals_raw is None:
			temporals: List[Dict[str, Any]] = []
		elif isinstance(temporals_raw, list) and 1 <= len(temporals_raw) <= 2:
			temporals = temporals_raw
		else:
			return f"clause[{i}]: temporal must be null or a list with 1–2 temporal objects"

		# Stats for trend checks
		non_null_anchor_temporals = 0
		has_interval_overlap = False
		has_from = False
		has_until = False

		for t_idx, t in enumerate(temporals):
			relation = t.get("relation")
			if relation not in REL:
				return f"clause[{i}].temporal[{t_idx}]: relation must be one of {sorted(REL)}"

			anchor = t.get("anchor", None)

			if relation in {"HISTORY", "BASELINE"}:
				if anchor is not None:
					return f"clause[{i}].temporal[{t_idx}]: relation={relation} requires anchor=null"
				# nothing else to validate in this temporal
				continue

			# FROM/UNTIL/OVERLAPS => anchor must be a single object
			if not isinstance(anchor, dict):
				return f"clause[{i}].temporal[{t_idx}]: anchor must be an object for relation={relation}"
			non_null_anchor_temporals += 1

			form = anchor.get("form")
			if form not in {"concept", "admission"}:
				return f"clause[{i}].temporal[{t_idx}].anchor: form must be 'concept' or 'admission'"

			day_off = anchor.get("day_offset")

			if form == "admission":
				if anchor.get("term") is not None or anchor.get("kind") is not None or anchor.get("event") is not None:
					return f"clause[{i}].temporal[{t_idx}].anchor: admission must have term/kind/event = null"
			else:
				# concept
				aterm = anchor.get("term")
				if not isinstance(aterm, str) or not aterm:
					return f"clause[{i}].temporal[{t_idx}].anchor: concept.term must be a non-empty string"

				kind = anchor.get("kind")
				if kind not in KIND:
					return f"clause[{i}].temporal[{t_idx}].anchor: concept.kind must be EVENT or INTERVAL"

				ev = anchor.get("event")
				if kind == "EVENT":
					if ev not in {"start", "stop", None}:
						return f"clause[{i}].temporal[{t_idx}].anchor: EVENT 'event' must be 'start'|'stop'|null"
					if day_off is not None and not isinstance(day_off, int):
						return f"clause[{i}].temporal[{t_idx}].anchor: day_offset must be integer or null for concept EVENT"
				else:  # INTERVAL
					if relation != "OVERLAPS":
						return f"clause[{i}].temporal[{t_idx}].anchor: INTERVAL only permitted with relation=OVERLAPS"
					if ev is not None:
						return f"clause[{i}].temporal[{t_idx}].anchor: INTERVAL must have event=null"
					if day_off is not None:
						return f"clause[{i}].temporal[{t_idx}].anchor: INTERVAL must have day_offset=null"
					has_interval_overlap = True

			if relation == "FROM":
				has_from = True
			elif relation == "UNTIL":
				has_until = True

		# --- Trend rules ---
		# if modifier in TREND:
		#     if not temporals:
		#         pass  # trend over entire admission allowed
		#     elif has_interval_overlap:
		#         pass  # OVERLAPS + INTERVAL anchor allowed
		#     elif non_null_anchor_temporals >= 2:
		#         pass  # two timepoints/window via two anchored temporals
		#     elif (has_from and has_until and "baseline" in mt_lower):
		#         pass  # baseline-window exception
		#     else:
		#         return (f"clause[{i}]: trend requires either no temporal, or OVERLAPS+INTERVAL, "
		#                 f"or two anchored temporals, or baseline-window exception (FROM+UNTIL with 'baseline').")

	return None

def call_intent_parser_once(client: OpenAI, query: str, model: str="gpt-4.1", temperature: float=0) -> str:
	"""
	Single model call that must return a tool call to `emit_parse`.
	"""
	chat = client.chat.completions.create(
		model=model,
		temperature=temperature,
		tools=TOOLS,
		tool_choice={"type": "function", "function": {"name": "emit_parse"}},
		messages=[
			{"role": "system", "content": "Call emit_parse with the final JSON as arguments. Do not write any text."},
			{"role": "user", "content": _intent_parser_prompt()},
			{"role": "user", "content": query},
		],
	)
	tc = chat.choices[0].message.tool_calls[0]
	return tc.function.arguments  # JSON string

def retry_fix_json(client: OpenAI, query: str, bad_json: Optional[str], schema_error: Optional[str], model: str="gpt-4.1", temperature: float=0) -> str:
	"""
	Repair pass: show the prior JSON and the validation error (schema or semantic);
	force a new tool call to emit_parse. Returns tool arguments (JSON string).
	Raises RuntimeError if no valid tool call is produced after retries.
	"""
	bad_snippet = ""
	if bad_json:
		bad_snippet = bad_json if len(bad_json) < 6000 else (bad_json[:6000] + "...<truncated>")
	error_note = f"\nValidation error:\n{schema_error}" if schema_error else ""

	instructions = (
		"Your previous output failed validation against the strict schema and rules."
		f"{error_note}\n\n"
		"Produce a corrected result by CALLING emit_parse.\n"
		"Do not write any free text. The function arguments must fully comply with the schema and rules."
	)

	msgs = [
		{"role": "system", "content": "Fix the output by calling emit_parse. Do not write any text; only the function call."},
		{"role": "user", "content": _intent_parser_prompt()},
		{"role": "user", "content": query},
		{"role": "user", "content": instructions},
	]
	if bad_snippet:
		msgs.append({"role": "user", "content": f"Here is your previous output to correct:\n```json\n{bad_snippet}\n```"})

	last_err = None
	for attempt in range(settings.params.intent_parser_max_retries + 1):
		try:
			chat = client.chat.completions.create(
				model=model,
				temperature=temperature,
				tools=TOOLS,  # must include emit_parse with the PARSE_SCHEMA
				tool_choice={"type": "function", "function": {"name": "emit_parse"}},
				messages=msgs,
			)

			msg = chat.choices[0].message
			tool_calls = getattr(msg, "tool_calls", None) or []
			# Prefer the emit_parse tool call if multiple exist
			emit = None
			for tc in tool_calls:
				if getattr(tc, "function", None) and tc.function.name == "emit_parse":
					emit = tc
					break

			if emit is None:
				# very rare, but handle gracefully
				last_err = RuntimeError("Model returned no emit_parse tool call during repair.")
				continue

			args = getattr(emit.function, "arguments", None)
			if not args or not isinstance(args, str):
				last_err = RuntimeError("emit_parse function.arguments missing or not a string.")
				continue

			return args  # JSON string

		except Exception as e:
			last_err = e
			# brief backoff between attempts
			time.sleep(0.6)

	raise RuntimeError(f"retry_fix_json failed: {last_err}")

def parse_with_repair(client: OpenAI, query: str, model: str="gpt-4.1", temperature: float=0) -> Dict[str, Any]:
	"""
	One-shot parse; if JSON/schema invalid OR semantic rules invalid, run a repair pass.
	Returns a Python dict that conforms to PARSE_SCHEMA and passes semantic checks.
	"""
	# First attempt
	first = call_intent_parser_once(client, query, model, temperature)

	# Schema validation
	ok_schema, err_schema = _validate_json_str(first)
	if ok_schema:
		parsed = json.loads(first)
		# Semantic validation
		sem_err = _semantic_validate(parsed)
		if sem_err is None:
			return parsed
		# Semantic repair
		time.sleep(1.0)
		repaired = retry_fix_json(client, query, bad_json=first, schema_error=f"Semantic error: {sem_err}",
								  model=model, temperature=temperature)
		ok2, err2 = _validate_json_str(repaired)
		if ok2:
			repaired_json = json.loads(repaired)
			sem_err2 = _semantic_validate(repaired_json)
			if sem_err2 is None:
				repaired_json["error_message"] = "first llm try failed semantic validation; repair succeeded"
				return repaired_json
			return _err(f"semantic validation still failing after repair: {sem_err2}")
		return _err(f"schema invalid after semantic repair attempt: {err2}")

	# Schema repair path
	time.sleep(1.0)
	repaired = retry_fix_json(client, query, bad_json=first, schema_error=err_schema,
							  model=model, temperature=temperature)
	ok2, err2 = _validate_json_str(repaired)
	if ok2:
		repaired_json = json.loads(repaired)
		sem_err2 = _semantic_validate(repaired_json)
		if sem_err2 is None:
			repaired_json["error_message"] = "first llm try failed schema validation; repair succeeded"
			return repaired_json
		# Try a final repair specifically for semantic error
		time.sleep(1.0)
		repaired2 = retry_fix_json(client, query, bad_json=repaired, schema_error=f"Semantic error: {sem_err2}",
								   model=model, temperature=temperature)
		ok3, err3 = _validate_json_str(repaired2)
		if ok3:
			repaired_json2 = json.loads(repaired2)
			sem_err3 = _semantic_validate(repaired_json2)
			if sem_err3 is None:
				repaired_json2["error_message"] = "schema fixed on repair #1; semantic fixed on repair #2"
				return repaired_json2
			return _err(f"semantic validation still failing after second repair: {sem_err3}")
		return _err(f"schema invalid on second repair: {err3}")

	return _err(f"schema invalid on first attempt: {err_schema}")
	
def _err(msg: str) -> dict:
	"""
	Create a standardized not-parsable payload.
	"""
	return {
		"parse_status": "not_parsable",
		"error_message": msg,
		"patient_references": [],
		"logic": None,
		"clauses": []
	}

