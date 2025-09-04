"""
sql_engine.py

Translates an extended intent "trace" (from the abstraction layer) into DuckDB SQL for:
  1) Candidate retrieval — which patients satisfy the boolean/temporal logic.
  2) Evidence retrieval — rows that support adjudication for the final LLM.

Expected DuckDB tables
----------------------
- patient_df(patient_id, patient_name, age, sex, ...)
- event_df(patient_id, normalized_term, value, unit, is_active, is_started, is_ended, history, day_offset_start, day_offset_end, source_entity, ...)
- term_concept_class_map_df(term, concept_class)
- baselines_df(patient_id, normalized_term, anchor, value, unit)
- trends_df(patient_id, normalized_term, anchor1, anchor2, change, anchor1_start, anchor1_end, anchor2_start, anchor2_end, val_start, val_end, delta, ...)

Temporal semantics
------------------
The engine compiles temporal constraints into one of:
- absolute window (admission-relative integer day bounds), or
- per-patient window computed from anchor concepts (via CTEs), supporting
  EVENT ("start"/"stop") and INTERVAL (min start / max end) anchors.

It also supports special cases:
- HISTORY   → no anchor, draws from `event_df.history = TRUE`
- BASELINE  → baseline-at-admission semantics (via `baselines_df` or early admission events (<= day 2) or history)
- Trend predicates ("uptrended"/"downtrended"/"unchanged") use `trends_df` and an anchor-pair resolver that understands pump-related spans.

Notes
-----
- `pump_canonical` is provided by the abstraction layer and used to resolve pump-related trend anchors into canonical (anchor1, anchor2) labels.
- `_cte_map` caches anchor CTEs so multiple leaves can reuse the same JOIN.

"""

import pandas as pd
import duckdb
from dataclasses import dataclass
import re
import sqlparse

@dataclass
class WindowSpec:
	"""
	Concrete temporal window specification for predicate compilation.

	Attributes
	----------
	mode : str
		"absolute" for admission-relative numeric bounds; "per_patient" when
		bounds depend on per-patient anchors (via CTEs).
	d_from : int | None
		Left bound (inclusive) when mode == "absolute".
	d_until : int | None
		Right bound (inclusive) when mode == "absolute".
	join : str | None
		SQL JOIN clause that attaches the anchor CTE to the target table
		(e.g., "JOIN cte_alias a ON a.patient_id = p.patient_id").
	left_expr : str | None
		SQL expression for the lower bound when mode == "per_patient".
	right_expr : str | None
		SQL expression for the upper bound when mode == "per_patient".
	extra_params : list
		Parameter list associated with `left_expr`/`right_expr` (e.g., offsets).
	include_history : bool
		When True, also check HISTORY rows in addition to the window, used for
		"prior to admission" semantics (e.g., UNTIL admission day 0).
	"""
	mode: str                 # "absolute" | "per_patient"
	d_from: int | None        # used when mode == "absolute"
	d_until: int | None       # used when mode == "absolute"
	join: str | None          # "JOIN cte_alias a ON a.patient_id = p.patient_id" or None
	left_expr: str | None     # SQL expr for lower bound (used in predicates)
	right_expr: str | None    # SQL expr for upper bound
	extra_params: list        # params for left/right exprs (e.g., integer offsets)
	include_history: bool = False # ensures history entries are searched for 'prior to admission' constraint

class SqlEngine:
	"""
	Compiles the abstraction-layer trace into SQL and executes against DuckDB.

	Parameters
	----------
	duckdb_path : str
		Path to the DuckDB database file.
	pump_canonical :
		Tuple returned by `QueryCompiler.resolve("left heart pump")`. The
		engine expects the first element to be the `ResolvedTarget` for the pump
		term and uses it to resolve pump-specific trend anchors.

	Attributes
	----------
	db : str
		DuckDB path used for all connections.
	error_str : str | None
		Error sentinel for certain unsupported constructs (e.g., trend anchors).
	_cte_map : dict[str, tuple[str, list]]
		Registry of anchor CTEs: name -> (cte_sql, params).
	pump_canonical :
		The `ResolvedTarget` for the pump term, used by trend anchor resolver.
	"""
	def __init__(self, duckdb_path: str, pump_canonical):
		self.db = duckdb_path
		self.error_str = None
		self._cte_map = {}     
		# store canonical term for pump 
		self.pump_canonical,_,_,_ = pump_canonical
		assert self.pump_canonical!=None # ensure that its canonical could be resolved

	# candidate retrieval
	def build_candidate_sql(self, trace) -> tuple[str, list]:
		"""
		Compile the candidate-retrieval SQL for a given extended trace.

		Parameters
		----------
		trace : dict
			Extended intent trace produced by the abstraction layer.

		Returns
		-------
		(sql, params) : tuple[str, list]
			SQL string with '?' placeholders and the parameter list.
		"""
		return self._compile_candidate_sql(trace)

	# returns patient ids satisfying the query as a list
	def run_candidate_sql(self, sql: str, params: list[int|float|str|bool]) -> list[int]:
		"""
		Execute candidate SQL and return matching patient IDs.

		Parameters
		----------
		sql : str
			Query string with '?' placeholders.
		params : list
			Parameter values in positional order.

		Returns
		-------
		list[int]
			Patient IDs that satisfy the compiled predicate.
		"""
		with duckdb.connect(self.db, read_only=True) as con:
			rows = con.execute(sql, params).fetchall()
		return [r[0] for r in rows] 

	# evidence retrieval
	def build_evidence_sql(self, trace, patient_ids: list[str]) -> tuple[str | None, list | None, str | None, list | None]:
		"""
		Compile evidence SQL for events and trends constrained by patient IDs.

		Parameters
		----------
		trace : dict
			Extended intent trace.
		patient_ids : list[str]
			Candidate patient IDs to restrict evidence selection.

		Returns
		-------
		(sql_event, params_event, sql_trend, params_trend)
			Each may be None if there is nothing to select for that source.
		"""
		sql_event, params_event = self._compile_event_evidence_sql(trace, patient_ids)
		sql_trend, params_trend = self._compile_trend_evidence_sql(trace, patient_ids)
		return sql_event, params_event, sql_trend, params_trend
		
	def run_evidence_sql(self, patient_ids: list[str], sql_event: str, params_event: list[int|float|str|bool], sql_trend: str, params_trend: list[int|float|str|bool]) -> dict[str, pd.DataFrame]:
		"""
		Execute evidence queries and return cleaned DataFrames.

		Parameters
		----------
		patient_ids : list[str]
			IDs to include when building context fallback.
		sql_event, params_event : str, list
			Event evidence SQL + params (may be None).
		sql_trend, params_trend : str, list
			Trend evidence SQL + params (may be None).

		Returns
		-------
		(event_df, trend_df) : tuple[pd.DataFrame | None, pd.DataFrame | None]
			Cleaned tables (None if no rows).
		"""
		df_dicts = self.run_evidence_dfs(patient_ids, sql_event, params_event, sql_trend, params_trend, len(patient_ids)<=10) # also retrieves context if n candidates <= 10
		event_df, trend_df = df_dicts.get('event_df'), df_dicts.get('trend_df')
		first_cols = ["patient_id", "patient_name", "age", "sex"]
		if event_df is not None:
			event_df = event_df.sort_values(by=["context","patient_id","day_offset_start","day_offset_end","normalized_term"], ascending=[True,True,True,True,True]).reset_index(drop=True)
			event_df = event_df.drop(columns=["is_worsening","level_entry"], errors="ignore")
			event_df = event_df[first_cols + [c for c in event_df.columns if c not in first_cols]]
			event_df = event_df[event_df['source_entity'].str.strip('"')!="extrapolated pump entry"] # only show final llm real, unextrapolated datapoints (remove for github repo)
		if trend_df is not None:
			trend_df = trend_df[first_cols + [c for c in trend_df.columns if c not in first_cols]]
		return event_df, trend_df
			
	# convert sql query to string for debug trace
		"""
		Build a readable SQL string with parameters inlined for debugging.

		Parameters
		----------
		sql1, params1 : str | None, list | None
			First statement and params (typically event or candidate SQL).
		sql2, params2 : str | None, list | None
			Optional second statement (e.g., trend SQL) and its params.

		Returns
		-------
		str
			Pretty-printed SQL with `'?'` replaced by literal representations.
		"""
	def build_query_str_for_debug_trace(self, sql1=None, params1=None, sql2=None, params2=None) -> str:
		output_str = ''
		if sql1:
			debug_sql = sql1
			for p in params1:
				debug_sql = debug_sql.replace("?", repr(p), 1)
			debug_sql = sqlparse.format(debug_sql, reindent=True, keyword_case='upper')
			output_str += debug_sql
		if sql2:
			debug_sql = sql2
			for p in params2:
				debug_sql = debug_sql.replace("?", repr(p), 1)
			debug_sql = sqlparse.format(debug_sql, reindent=True, keyword_case='upper')
			if sql1: output_str += "\n\n/* -------- precomputed trend lookups -------- */\n"
			output_str += debug_sql
		return output_str
		
		
		
	########### HELPER FUNCTIONS ###########
		
	def _string_normalizer(self, s: str) -> str:
		return re.sub(r"\s+", " ", re.sub(r"[-_]", " ", s)).strip().lower()
		
	def _compile_patient_refs(self, refs):
		"""
		Compile an optional patient-reference filter (IDs or names).

		Parameters
		----------
		refs : list[str] | None
			Mixed patient IDs/names from `intent.patient_references`.

		Returns
		-------
		(sql, params) : tuple[str, list]
			WHERE fragment and parameters; returns "TRUE" when empty.
		"""
		if not refs:
			return "TRUE", []
		parts, params = [], []
		for r in refs:
			r = self._string_normalizer(r)
			parts.append("(CAST(p.patient_id AS VARCHAR) = ? OR p.patient_name ILIKE '%' || ? || '%')")
			params.extend([str(r), str(r)])
		return "(" + " OR ".join(parts) + ")", params
	
	def _compile_node(self, node, parent_temporal):
		"""
		Recursively compile a boolean node/leaf into a WHERE fragment.

		Inherits the closest `temporal` context from ancestors. Leaves are
		"resolved" terms/classes (set by the abstraction layer). Boolean nodes
		combine children with AND/OR. Neutral children ("TRUE") are skipped.

		Returns
		-------
		(sql, params) : tuple[str, list]
			WHERE fragment and parameters.
		"""
		# Non-dict or empty → neutral TRUE (so parents can combine cleanly)
		if not isinstance(node, dict):
			return "TRUE", []
	
		# Inherit temporal
		temporal_ctx = node.get("temporal") or parent_temporal
	
		# Resolved leaf?
		if ("clauses" not in node) and (node.get("resolved_term") or node.get("resolved_class")):
			return self._compile_leaf(node, temporal_ctx)
	
		# Boolean node
		logic = (node.get("logic") or "AND").upper()
		parts, params = [], []
		for ch in (node.get("clauses") or []):
			s, p = self._compile_node(ch, temporal_ctx)
			if s and s != "TRUE":               # skip neutral children
				parts.append(f"({s})")
				params.extend(p)
	
		if not parts:                           # empty subtree
			return ("FALSE", []) if logic == "OR" else ("TRUE", [])
	
		if logic == "OR":
			return " OR ".join(parts), params
		# default AND (covers unknown logic too)
		return " AND ".join(parts), params
		
	def _temporal_mode(self, temporal_ctx):
		"""
		Classify a temporal context into a coarse mode.

		Returns
		-------
		str
			"history" if exactly one temporal with relation=HISTORY,
			"baseline" if exactly one temporal with relation=BASELINE,
			otherwise "window".
		"""
		# Returns "history" | "baseline" | "window"
		if temporal_ctx and len(temporal_ctx) == 1:
			rel = (temporal_ctx[0].get("relation") or "").upper()
			if rel == "HISTORY":  return "history"
			if rel == "BASELINE": return "baseline"
		return "window"
		
	def _compile_leaf(self, leaf, temporal_ctx):
		"""
		Compile a resolved leaf (term/class) into a WHERE fragment.

		Handles:
		  - trend modifiers → trends_df
		  - present/numeric/absent with HISTORY/BASELINE/window modes
		"""
		mod = (leaf.get("modifier") or "").lower().strip()
		if mod in {"uptrended", "downtrended", "unchanged"}:
			return self._compile_trend_predicate(leaf, temporal_ctx)
	
		term = leaf.get("resolved_term")
		concept_class = leaf.get("resolved_class")
		mode = self._temporal_mode(temporal_ctx)
	
		# --------------------------- ABSENT semantics ---------------------------
		if mod == "absent":
			disj_sql, disj_params = [], []
			if mode == "history":
				if term:
					s, p = self._term_history_absent_predicate(term); disj_sql.append(s); disj_params += p
				if concept_class:
					s, p = self._class_history_absent_predicate(concept_class); disj_sql.append(s); disj_params += p
				return (" AND ".join(f"({s})" for s in disj_sql) or "TRUE", disj_params)
	
			if mode == "baseline":
				if term:
					s, p = self._term_baseline_absent_predicate(term); disj_sql.append(s); disj_params += p
				if concept_class:
					s, p = self._class_baseline_absent_predicate(concept_class); disj_sql.append(s); disj_params += p
				return (" AND ".join(f"({s})" for s in disj_sql) or "TRUE", disj_params)
	
			spec = self._resolve_temporal_day_window(temporal_ctx)
			if term:
				s, p = self._term_absent_predicate(term, spec); disj_sql.append(s); disj_params += p
			if concept_class:
				s, p = self._class_absent_predicate(concept_class, spec); disj_sql.append(s); disj_params += p
			return (" AND ".join(f"({s})" for s in disj_sql) or "TRUE", disj_params)
	
		# --------------------------- PRESENT / numeric --------------------------
		disj_sql, disj_params = [], []
	
		if mode == "history":
			if term:
				s, p = self._term_history_predicate(leaf, term); disj_sql.append(s); disj_params += p
			if concept_class:
				s, p = self._class_history_predicate(leaf, concept_class); disj_sql.append(s); disj_params += p
			return (" OR ".join(f"({s})" for s in disj_sql) or "TRUE", disj_params)
	
		if mode == "baseline":
			if term:
				s, p = self._term_baseline_predicate(leaf, term); disj_sql.append(s); disj_params += p
			if concept_class:
				s, p = self._class_baseline_predicate(leaf, concept_class); disj_sql.append(s); disj_params += p
			return (" OR ".join(f"({s})" for s in disj_sql) or "TRUE", disj_params)
	
		# window mode
		spec = self._resolve_temporal_day_window(temporal_ctx)
		if term:
			s, p = self._term_exists_predicate(leaf, term, spec); disj_sql.append(s); disj_params += p
		if concept_class:
			s, p = self._class_exists_predicate(leaf, concept_class, spec); disj_sql.append(s); disj_params += p
		return (" OR ".join(f"({s})" for s in disj_sql) or "TRUE", disj_params)
		
	def _resolve_temporal_day_window(self, temporal_ctx, patient_alias: str = "p") -> WindowSpec:
		"""
		Convert a temporal context into a `WindowSpec`.

		- No context → unbounded absolute window (-INF, +INF).
		- Single FROM/UNTIL/OVERLAPS:
			* admission anchors → absolute bounds
			* concept EVENT/INTERVAL → per-patient bounds via CTEs
		- Paired FROM+UNTIL:
			* both admission → absolute (d_from, d_until)
			* otherwise → pair of per-patient expressions (with JOINs)

		Parameters
		----------
		temporal_ctx : list[dict] | None
			Clause-level temporal array (normalized).
		patient_alias : str
			Alias of the patient-bearing table for building JOINs.

		Returns
		-------
		WindowSpec
			Ready to be consumed by overlap predicates.
		"""
		INF = 10**9
		if not temporal_ctx:
			return WindowSpec("absolute", -INF, +INF, None, None, None, [])
	
		if len(temporal_ctx) == 1:
			rel = (temporal_ctx[0].get("relation") or "").upper()
			a   = temporal_ctx[0].get("anchor") or {}
	
			# FROM
			if rel == "FROM":
				if a.get("form") == "admission":
					return WindowSpec("absolute", int(a.get("day_offset") or 0), +INF, None, None, None, [])
				left_expr, join, params = self._anchor_expr(a, side="from", patient_alias=patient_alias)
				return WindowSpec("per_patient", None, None, join, left_expr, str(+INF), params)
	
			# UNTIL  (lower bound is -INF, upper bound is the anchor)
			if rel == "UNTIL":
				if a.get("form") == "admission":
					return WindowSpec("absolute", -INF, int(a.get("day_offset") or 0), None, None, None, [], include_history=(int(a.get("day_offset") or 0) <= 0)) # last entry set to true for the special case: UNTIL admission day 0
				right_expr, join, params = self._anchor_expr(a, side="until", patient_alias=patient_alias)
				return WindowSpec("per_patient", None, None, join, str(-INF), right_expr, params)
	
			# OVERLAPS 
			if rel == "OVERLAPS":
				if a.get("form") == "admission":
					k = int(a.get("day_offset") or 0)
					return WindowSpec("absolute", k, k, None, None, None, [])
			
				if a.get("form") == "concept":
					kind = (a.get("kind") or "").upper()
					if kind == "INTERVAL":
						# --- NEW: single join + two bounds from the same alias ---
						term = self._string_normalizer(a.get("resolved_term") or a.get("term") or "")
						term_slug = term.replace(' ', '_')
						cte_name  = f"anchor_interval_{term_slug}"
						if cte_name not in self._cte_map:
							sql = (
								"SELECT e.patient_id, "
								"MIN(e.day_offset_start) AS anchor_from, "
								"MAX(COALESCE(e.day_offset_end, e.day_offset_start)) AS anchor_until "
								"FROM event_df e "
								"WHERE e.normalized_term = ? AND e.is_active = TRUE "
								"GROUP BY e.patient_id"
							)
							self._cte_map[cte_name] = (sql, [term])
			
						alias = "a"
						join = f"JOIN {cte_name} {alias} ON {alias}.patient_id = {patient_alias}.patient_id"
			
						# the same offset applies to both bounds (use two params)
						offset = int(a.get("day_offset") or 0)
						left_expr  = f"{alias}.anchor_from + ?"
						right_expr = f"{alias}.anchor_until + ?"
						extras = [offset, offset]
			
						return WindowSpec("per_patient", None, None, join, left_expr, right_expr, extras)
			
					if kind == "EVENT":
						# unchanged from before, but ensure single join usage here as well
						pt_expr, join, p = self._anchor_expr(a, side="from", patient_alias=patient_alias)
						return WindowSpec("per_patient", None, None, join, pt_expr, pt_expr, p + p)
			
				return WindowSpec("absolute", 0, 0, None, None, None, [])
	
		# FROM + UNTIL 
		if (len(temporal_ctx) == 2 and
			(temporal_ctx[0].get("relation") or "").upper() == "FROM" and
			(temporal_ctx[1].get("relation") or "").upper() == "UNTIL"):
			a_from  = temporal_ctx[0].get("anchor") or {}
			a_until = temporal_ctx[1].get("anchor") or {}
		
			# both admission → absolute bounds
			if a_from.get("form") == "admission" and a_until.get("form") == "admission":
				d_from  = int(a_from.get("day_offset")  or 0)
				d_until = int(a_until.get("day_offset") or 0)
				return WindowSpec("absolute", d_from, d_until, None, None, None, [])
		
		left_params, right_params = [], []
		joins = []
		
		# left bound
		if a_from.get("form") == "admission":
			left_expr = str(int(a_from.get("day_offset") or 0))
		else:
			left_expr, j1, p1 = self._anchor_expr(a_from, side="from", alias="a_from", patient_alias=patient_alias)
			if j1: joins.append(j1)
			left_params = p1  # keep separate
		
		# right bound
		if a_until.get("form") == "admission":
			right_expr = str(int(a_until.get("day_offset") or 0))
		else:
			right_expr, j2, p2 = self._anchor_expr(a_until, side="until", alias="a_until", patient_alias=patient_alias)
			if j2: joins.append(j2)
			right_params = p2  # keep separate
		
		join = " ".join(joins) if joins else None
		extras = (right_params or []) + (left_params or [])  # RIGHT then LEFT
		return WindowSpec("per_patient", None, None, join, left_expr, right_expr, extras)
	
	def _term_exists_predicate(self, leaf, term, spec: WindowSpec):
		# main (window-overlap) branch
		conds, params = [], []
		conds.append("e.patient_id = p.patient_id")
		conds.append("e.normalized_term = ?"); params.append(term)
		conds.append("e.is_active = TRUE")
		if leaf.get("op") and (leaf.get("value") is not None):
			conds.append(f"e.value {leaf['op']} ?"); params.append(leaf["value"])
		if leaf.get("unit"):
			conds.append("e.unit ILIKE '%' || ? || '%'"); params.append(str(leaf["unit"]))
		ov_sql, ov_params = self._temporal_overlap_sql(spec)
		conds.append(ov_sql); params += ov_params
		join_sql = f" {spec.join} " if (spec.mode == "per_patient" and spec.join) else " "
		sql_main = "EXISTS (SELECT 1 FROM event_df e" + join_sql + "WHERE " + " AND ".join(conds) + ")"
	
		# optional history branch (same numeric/unit filters, no overlap)
		if spec.include_history:
			h_conds, h_params = [], []
			h_conds.append("e.patient_id = p.patient_id")
			h_conds.append("e.normalized_term = ?"); h_params.append(term)
			h_conds.append("e.is_active = TRUE")
			h_conds.append("e.history = TRUE")
			if leaf.get("op") and (leaf.get("value") is not None):
				h_conds.append(f"e.value {leaf['op']} ?"); h_params.append(leaf["value"])
			if leaf.get("unit"):
				h_conds.append("e.unit ILIKE '%' || ? || '%'"); h_params.append(str(leaf["unit"]))
			sql_hist = "EXISTS (SELECT 1 FROM event_df e WHERE " + " AND ".join(h_conds) + ")"
			return f"(({sql_main}) OR ({sql_hist}))", params + h_params
	
		return sql_main, params
	
	def _class_exists_predicate(self, leaf, concept_class, spec: WindowSpec):
		"""
		Build EXISTS(...) predicate for a single term within a temporal window.

		Applies optional numeric/unit filters. When `include_history` is set on
		the window spec, an OR-branch for HISTORY is added.
		"""
		# main (window-overlap) branch
		base_join = (
			"FROM event_df e "
			"JOIN term_concept_class_map_df m "
			"ON m.term = e.normalized_term AND m.concept_class = ?"
		)
		anchor_join = f" {spec.join} " if (spec.mode == "per_patient" and spec.join) else " "
	
		conds, params = [], []
		params.append(concept_class)
		conds.append("e.patient_id = p.patient_id")
		conds.append("e.is_active = TRUE")
		if leaf.get("op") and (leaf.get("value") is not None):
			conds.append(f"e.value {leaf['op']} ?"); params.append(leaf["value"])
		if leaf.get("unit"):
			conds.append("e.unit ILIKE '%' || ? || '%'"); params.append(str(leaf["unit"]))
		ov_sql, ov_params = self._temporal_overlap_sql(spec)
		conds.append(ov_sql); params += ov_params
		sql_main = f"EXISTS (SELECT 1 {base_join}{anchor_join}WHERE " + " AND ".join(conds) + ")"
	
		# optional history branch (same numeric/unit filters, no overlap)
		if spec.include_history:
			h_conds, h_params = [], []
			h_params.append(concept_class)
			h_conds.append("e.patient_id = p.patient_id")
			h_conds.append("e.is_active = TRUE")
			h_conds.append("e.history = TRUE")
			if leaf.get("op") and (leaf.get("value") is not None):
				h_conds.append(f"e.value {leaf['op']} ?"); h_params.append(leaf["value"])
			if leaf.get("unit"):
				h_conds.append("e.unit ILIKE '%' || ? || '%'"); h_params.append(str(leaf["unit"]))
			sql_hist = (
				"EXISTS (SELECT 1 FROM event_df e "
				"JOIN term_concept_class_map_df m "
				"ON m.term = e.normalized_term AND m.concept_class = ? "
				"WHERE " + " AND ".join(h_conds) + ")"
			)
			return f"(({sql_main}) OR ({sql_hist}))", params + h_params
	
		return sql_main, params
	
	def _term_history_predicate(self, leaf, term):
		"""Build EXISTS(...) for a term in HISTORY (no temporal overlap check)."""
		conds  = ["e.patient_id = p.patient_id", "e.normalized_term = ?", "e.history = TRUE"]
		params = [term]
		mod = (leaf.get("modifier") or "").lower().strip()
		conds.append("e.is_active = FALSE" if mod == "absent" else "e.is_active = TRUE")
		if leaf.get("op") and (leaf.get("value") is not None):
			conds.append(f"e.value {leaf['op']} ?"); params.append(leaf["value"])
		if leaf.get("unit"):
			conds.append("e.unit ILIKE '%' || ? || '%'"); params.append(str(leaf["unit"]))
		sql = "EXISTS (SELECT 1 FROM event_df e WHERE " + " AND ".join(conds) + ")"
		return sql, params
	
	
	def _class_history_predicate(self, leaf, concept_class):
		"""Build EXISTS(...) for any class term in HISTORY (no temporal overlap check)."""
		conds  = ["e.patient_id = p.patient_id", "e.history = TRUE"]
		params = [concept_class]
	
		join = ("FROM event_df e "
				"JOIN term_concept_class_map_df m "
				"ON m.term = e.normalized_term AND m.concept_class = ?")
	
		mod = (leaf.get("modifier") or "").lower()
		if mod == "absent":
			conds.append("e.is_active = FALSE")
		else:
			conds.append("e.is_active = TRUE")
	
		if leaf.get("op") and (leaf.get("value") is not None):
			conds.append(f"e.value {leaf['op']} ?"); params.append(leaf["value"])
		if leaf.get("unit"):
			conds.append("e.unit ILIKE '%' || ? || '%'"); params.append(str(leaf["unit"]))
	
		sql = f"EXISTS (SELECT 1 {join} WHERE " + " AND ".join(conds) + ")"
		return sql, params
	
	def _term_baseline_predicate(self, leaf, term):
		"""
		Build BASELINE predicate:
		numeric → baselines_df at admission_start,
		non-numeric → early admission (<= day 2) OR history.
		"""
		has_numeric = bool(leaf.get("op") and (leaf.get("value") is not None))
		if has_numeric:
			conds, params = [], []
			conds.append("b.patient_id = p.patient_id")
			conds.append("b.normalized_term = ?"); params.append(term)
			conds.append("b.anchor = 'admission_start'")
			conds.append(f"b.value {leaf['op']} ?"); params.append(leaf["value"])
			if leaf.get("unit"):
				conds.append("b.unit ILIKE '%' || ? || '%'"); params.append(str(leaf["unit"]))
			sql = "EXISTS (SELECT 1 FROM baselines_df b WHERE " + " AND ".join(conds) + ")"
			return sql, params
	
		# Non-numeric: early admission (<=2) or history
		conds, params = [], []
		conds.append("e.patient_id = p.patient_id")
		conds.append("e.normalized_term = ?"); params.append(term)
		mod = (leaf.get("modifier") or "").lower().strip()
		conds.append("e.is_active = FALSE" if mod == "absent" else "e.is_active = TRUE")
		if leaf.get("unit"):
			conds.append("e.unit ILIKE '%' || ? || '%'"); params.append(str(leaf["unit"]))
		conds.append("(COALESCE(e.day_offset_end, e.day_offset_start) <= 2 OR e.history = TRUE)")
		sql = "EXISTS (SELECT 1 FROM event_df e WHERE " + " AND ".join(conds) + ")"
		return sql, params
	
	def _class_baseline_predicate(self, leaf, concept_class):
		"""
		Build BASELINE predicate for a class:
		numeric → baselines_df joined with class map,
		non-numeric → early admission (<= day 2) OR history joined with class map.
		"""
		has_numeric = bool(leaf.get("op") and (leaf.get("value") is not None))
	
		if has_numeric:
			# Compare class members' baseline values at admission_start
			join = ("FROM baselines_df b "
					"JOIN term_concept_class_map_df m "
					"ON m.term = b.normalized_term AND m.concept_class = ?")
			conds, params  = [], []
			params.append(concept_class)  # join param
			conds.append("b.patient_id = p.patient_id")
			conds.append("b.anchor = 'admission_start'")
			conds.append(f"b.value {leaf['op']} ?"); params.append(leaf["value"])
			if leaf.get("unit"):
				conds.append("b.unit ILIKE '%' || ? || '%'"); params.append(str(leaf["unit"]))
			sql = f"EXISTS (SELECT 1 {join} WHERE " + " AND ".join(conds) + ")"
			return sql, params
	
		# Non-numeric baseline using early-admission events or history
		join = ("FROM event_df e "
				"JOIN term_concept_class_map_df m "
				"ON m.term = e.normalized_term AND m.concept_class = ?")
		conds, params  = [], []
		params.append(concept_class)  # join param
		conds.append("e.patient_id = p.patient_id")
	
		# Modifier: default to present unless 'absent'
		mod = (leaf.get("modifier") or "").lower().strip()
		if mod == "absent":
			conds.append("e.is_active = FALSE")
		else:
			conds.append("e.is_active = TRUE")
	
		if leaf.get("unit"):
			conds.append("e.unit ILIKE '%' || ? || '%'"); params.append(str(leaf["unit"]))
	
		conds.append("(COALESCE(e.day_offset_end, e.day_offset_start) <= 2 OR e.history = TRUE)")
	
		sql = f"EXISTS (SELECT 1 {join} WHERE " + " AND ".join(conds) + ")"
		return sql, params
		
	def _anchor_expr(self, anchor: dict, side: str, alias: str = "a", patient_alias: str = "p"):
		"""
		Build a per-patient bound expression and JOIN clause from an anchor.

		Parameters
		----------
		anchor : dict
			Normalized anchor object with form/kind/event/day_offset.
		side : str
			'from' or 'until' — for INTERVALs picks min(start) vs max(end).
		alias : str
			Alias for the anchor CTE/table.
		patient_alias : str
			Alias for the table holding patient_id being filtered.

		Returns
		-------
		(expr, join_clause, params) : tuple[str, str | None, list]
			An SQL expression (e.g., "a.anchor_day + ?"), an optional JOIN
			clause to attach the CTE, and the parameter list (offset).
		"""
		form = anchor.get("form")
		offset = int(anchor.get("day_offset") or 0)
	
		# Admission handled by caller (literal)
		if form == "admission":
			return str(offset), None, []
	
		if form != "concept":
			# Unknown → treat as day 0 literal
			return "0", None, []
	
		term = self._string_normalizer(anchor.get("resolved_term") or anchor.get("term") or "")
		kind = (anchor.get("kind") or "").upper()
		event = (anchor.get("event") or "").lower().strip()  # "start" | "stop" | ""
	
		# INTERVAL: build min(start), max(end)
		if kind == "INTERVAL":
			cte_name = f"anchor_interval_{term.replace(' ','_')}"
			if cte_name not in self._cte_map:
				sql = (
					"SELECT e.patient_id, "
					"MIN(e.day_offset_start) AS anchor_from, "
					"MAX(COALESCE(e.day_offset_end, e.day_offset_start)) AS anchor_until "
					"FROM event_df e "
					"WHERE e.normalized_term = ? AND e.is_active = TRUE "
					"GROUP BY e.patient_id"
				)
				self._cte_map[cte_name] = (sql, [term])
			join_clause = f"JOIN {cte_name} {alias} ON {alias}.patient_id = {patient_alias}.patient_id"
			expr = f"{alias}.anchor_from + ?" if side == "from" else f"{alias}.anchor_until + ?"
			return expr, join_clause, [offset]
	
		# EVENT: start/stop points
		if kind == "EVENT":
			if event == "start":
				cte_name = f"anchor_start_{term.replace(' ','_')}"
				if cte_name not in self._cte_map:
					sql = (
						"SELECT patient_id, MIN(day_offset_start) AS anchor_day "
						"FROM event_df "
						"WHERE normalized_term = ? AND is_started = TRUE "
						"GROUP BY patient_id"
					)
					self._cte_map[cte_name] = (sql, [term])
			elif event == "stop":
				cte_name = f"anchor_stop_{term.replace(' ','_')}"
				if cte_name not in self._cte_map:
					sql = (
						"SELECT patient_id, MAX(day_offset_end) AS anchor_day "
						"FROM event_df "
						"WHERE normalized_term = ? AND is_ended = TRUE "
						"GROUP BY patient_id"
					)
					self._cte_map[cte_name] = (sql, [term])
			else:
				# Unknown event kind → treat as 0
				return "0", None, []
	
			join_clause = f"JOIN {cte_name} {alias} ON {alias}.patient_id = {patient_alias}.patient_id"
			expr = f"{alias}.anchor_day + ?"
			return expr, join_clause, [offset]
	
		# default
		return "0", None, []
	
	def _temporal_overlap_sql(self, spec: WindowSpec):
		"""Return overlap condition SQL + params for this window spec."""
		start_expr = "COALESCE(e.day_offset_start, e.day_offset_end, 0)"
		end_expr   = "COALESCE(e.day_offset_end,   e.day_offset_start, 0)"
	
		if spec.mode == "absolute":
			sql = f"{start_expr} <= ? AND {end_expr} >= ?"
			return sql, [spec.d_until, spec.d_from]
	
		right = spec.right_expr or str(10**9)
		left  = spec.left_expr  or str(-10**9)
		sql = f"{start_expr} <= ({right}) AND {end_expr} >= ({left})"
		return sql, spec.extra_params
		
	def _term_absent_predicate(self, term, spec: WindowSpec):
		"""
		Build NOT EXISTS(...) for a term over a temporal window (absence).

		When `include_history` is set, also forbids any HISTORY row.
		"""
		# window branch: no active datapoint overlapping window
		neg_conds, neg_params = ["e.patient_id = p.patient_id", "e.normalized_term = ?"], [term]
		ov_sql, ov_params = self._temporal_overlap_sql(spec)
		neg_conds.append("e.is_active = TRUE")
		neg_conds.append(ov_sql); neg_params += ov_params
		join_sql = f" {spec.join} " if (spec.mode == "per_patient" and spec.join) else " "
		sql_main = "NOT EXISTS (SELECT 1 FROM event_df e" + join_sql + "WHERE " + " AND ".join(neg_conds) + ")"
	
		if spec.include_history:
			h_conds, h_params = [], []
			h_conds.append("e.patient_id = p.patient_id")
			h_conds.append("e.normalized_term = ?"); h_params.append(term)
			h_conds.append("e.is_active = TRUE")
			h_conds.append("e.history = TRUE")
			sql_hist = "NOT EXISTS (SELECT 1 FROM event_df e WHERE " + " AND ".join(h_conds) + ")"
			return f"(({sql_main}) AND ({sql_hist}))", neg_params + h_params
	
		return sql_main, neg_params
	
	def _class_absent_predicate(self, concept_class, spec: WindowSpec):
		"""
		Build NOT EXISTS(...) for any term in a class over a temporal window (absence).

		Adds a HISTORY branch when `include_history` is set on the window spec.
		"""
		ov_sql, ov_params = self._temporal_overlap_sql(spec)
		anchor_join = f" {spec.join} " if (spec.mode == "per_patient" and spec.join) else " "
		sql_main = (
			"NOT EXISTS (SELECT 1 FROM event_df e "
			"JOIN term_concept_class_map_df m ON m.term = e.normalized_term AND m.concept_class = ?"
			f"{anchor_join}"
			"WHERE e.patient_id = p.patient_id AND e.is_active = TRUE AND " + ov_sql + ")"
		)
		params = [concept_class] + ov_params
	
		if spec.include_history:
			sql_hist = (
				"NOT EXISTS (SELECT 1 FROM event_df e "
				"JOIN term_concept_class_map_df m ON m.term = e.normalized_term AND m.concept_class = ? "
				"WHERE e.patient_id = p.patient_id AND e.is_active = TRUE AND e.history = TRUE)"
			)
			return f"(({sql_main}) AND ({sql_hist}))", params + [concept_class]
	
		return sql_main, params
	
	def _term_history_absent_predicate(self, term):
		"""Build NOT EXISTS(...) forbidding HISTORY rows for a term."""
		sql = (
			"NOT EXISTS (SELECT 1 FROM event_df e "
			"WHERE e.patient_id = p.patient_id AND e.normalized_term = ? "
			"AND e.history = TRUE AND e.is_active = TRUE)"
		)
		return sql, [term]
	
	def _class_history_absent_predicate(self, concept_class):
		"""Build NOT EXISTS(...) forbidding HISTORY rows for any class term."""
		sql = (
			"NOT EXISTS (SELECT 1 FROM event_df e "
			"JOIN term_concept_class_map_df m ON m.term = e.normalized_term AND m.concept_class = ? "
			"WHERE e.patient_id = p.patient_id AND e.history = TRUE AND e.is_active = TRUE)"
		)
		return sql, [concept_class]
	
	def _term_baseline_absent_predicate(self, term):
		"""
		Build NOT EXISTS(...) forbidding early-admission (<=2) or HISTORY rows
		for a term (baseline absence).
		"""
		sql = (
			"NOT EXISTS (SELECT 1 FROM event_df e "
			"WHERE e.patient_id = p.patient_id AND e.normalized_term = ? "
			"AND (COALESCE(e.day_offset_end, e.day_offset_start) <= 2 OR e.history = TRUE) "
			"AND e.is_active = TRUE)"
		)
		return sql, [term]
	
	def _class_baseline_absent_predicate(self, concept_class):
		"""
		Build NOT EXISTS(...) forbidding early-admission (<=2) or HISTORY rows
		for any term in a class (baseline absence).
		"""
		sql = (
			"NOT EXISTS (SELECT 1 FROM event_df e "
			"JOIN term_concept_class_map_df m ON m.term = e.normalized_term AND m.concept_class = ? "
			"WHERE e.patient_id = p.patient_id "
			"AND (COALESCE(e.day_offset_end, e.day_offset_start) <= 2 OR e.history = TRUE) "
			"AND e.is_active = TRUE)"
		)
		return sql, [concept_class]
		
	def _resolve_trend_anchors(self, temporal_ctx) -> tuple[str, str] | None:
		"""
		Resolve (anchor1, anchor2) labels for trend predicates.

		The mapping recognizes admission boundaries and pump-related spans
		using `self.pump_canonical.canonical`. Returns a pair like
		("pump_earliest", "pump_latest"), ("prepump_latest", "pump_earliest"),
		("admission_start", "admission_end"), etc.

		Returns
		-------
		tuple[str, str] | None
			(anchor1, anchor2) if resolvable, else None (marks unsupported).
		"""
		def pick_anchors(tctx):
			if not tctx:
				return "admission_start", "latest"
	
			if len(tctx)==1:
				rel = (tctx[0].get("relation") or "").upper()
				a = tctx[0].get("anchor") or {}
				a_term = a.get("resolved_term") or a.get("term")
				a_event = a.get("event")
				a_day_offset = a.get("day_offset")
	
				if a_term == self.pump_canonical.canonical and (a_day_offset is None or a_day_offset == 0):
					if rel == "OVERLAPS":
						if not a_event:
							return "pump_earliest", "pump_latest"
						if a_event == "start":
							return "prepump_latest", "pump_earliest"
						if a_event == "stop":
							return "pump_latest", "postpump_earliest"
					if rel == "UNTIL":
						if a_event == "stop":
							return "prepump", "pump"
						if a_event == "start":
							return "admission_start", "prepump_latest"
						return "admission_start", "pump"
					if rel == "FROM":
						if a_event == "start":
							return "pump", "admission_end"
						if a_event == "stop":
							return "postpump_earliest", "admission_end"
						return "pump", "admission_end"
	
				if a.get("form") == "admission" and (a_day_offset is None or a_day_offset in (0, 999)):
					return "admission_start", "admission_end"
	
				return None
	
			elif len(tctx) == 2:
				rel0, rel1 = (tctx[0].get("relation") or "").upper(), (tctx[1].get("relation") or "").upper()
				a0, a1 = tctx[0].get("anchor") or {}, tctx[1].get("anchor") or {}
				a_term0, a_term1 = a0.get("resolved_term") or a0.get("term"), a1.get("resolved_term") or a1.get("term")
				a_event0, a_event1 = a0.get("event"), a1.get("event")
				a_day_offset0, a_day_offset1 = a0.get("day_offset"), a1.get("day_offset")
				
				if (a_term0==self.pump_canonical.canonical and (a_day_offset0 is None or a_day_offset0==0)) and (a_term1==self.pump_canonical.canonical and (a_day_offset1 is None or a_day_offset1==0)):
					if rel0 == "FROM" and rel1 == "UNTIL":
						if a_event0 == "start" and a_event1 == "stop":
							return "pump_earliest", "pump_latest"
							
				return None
				
			return None
	
		return pick_anchors(temporal_ctx)
				
	def _compile_trend_predicate(self, leaf, temporal_ctx):
		"""
		Compile a trend leaf into EXISTS(...) over `trends_df`.

		If anchors cannot be resolved, sets `self.error_str` and returns a
		neutral predicate ("TRUE") so the caller can short-circuit gracefully.
		"""
		anchors = self._resolve_trend_anchors(temporal_ctx)
		if not anchors:
			self.error_str = "unsupported trend anchor"
			return "TRUE", []
		anchor1, anchor2 = anchors
	
		term = leaf.get("resolved_term")
		concept_class = leaf.get("resolved_class")
		change = leaf.get("modifier")
	
		parts, params = [], []
		if term:
			sql = (
				"EXISTS (SELECT 1 FROM trends_df t "
				"WHERE t.patient_id = p.patient_id "
				"AND t.normalized_term = ? "
				"AND t.anchor1 = ? AND t.anchor2 = ? "
				"AND t.change = ?)"
			)
			parts.append(sql)
			params += [term, anchor1, anchor2, change]
	
		if concept_class:
			sql = (
				"EXISTS (SELECT 1 FROM trends_df t "
				"JOIN term_concept_class_map_df m "
				"  ON m.term = t.normalized_term AND m.concept_class = ? "
				"WHERE t.patient_id = p.patient_id "
				"AND t.anchor1 = ? AND t.anchor2 = ? "
				"AND t.change = ?)"
			)
			parts.append(sql)
			params += [concept_class, anchor1, anchor2, change]
	
		if not parts:
			return "TRUE", []
		return (" OR ".join(parts), params)
		
	def _emit_event_leaf_selects(self, leaf: dict, temporal_ctx, pid_clause: str, pid_params: list, selects: list[tuple[str, list]]):
		"""
		Append SELECT statements that lift event rows serving as evidence/context.

		The logic mirrors predicate compilation:
		  - trends → lift without temporal filters
		  - history/baseline → appropriate WHEREs
		  - window mode → temporal overlap + optional history pass-through

		Parameters
		----------
		leaf : dict
			Resolved term/class leaf.
		temporal_ctx : list | None
			Inherited temporal context for the leaf.
		pid_clause : str
			"e.patient_id IN (?, ?, ...)" WHERE fragment.
		pid_params : list
			Patient ID parameters.
		selects : list[tuple[str, list]]
			Output collection of (sql, params) to UNION later.
		"""
		mod   = (leaf.get("modifier") or "").lower().strip()
		term  = leaf.get("resolved_term")
		concept_class = leaf.get("resolved_class")
	
		# Common numeric/unit filters
		num_where, num_params = [], []
		if leaf.get("op") and (leaf.get("value") is not None):
			num_where.append(f"e.value {leaf['op']} ?"); num_params.append(leaf["value"])
		if leaf.get("unit"):
			num_where.append("e.unit ILIKE '%' || ? || '%'"); num_params.append(str(leaf["unit"]))
	
		def add_term_select(extra_where: list[str], extra_params: list, anchor_join: str = " "):
			where = [pid_clause, "e.normalized_term = ?"] + extra_where + num_where
			params = list(pid_params) + [term] + extra_params + num_params
			sql = "SELECT e.* FROM event_df e" + anchor_join + "WHERE " + " AND ".join(where)
			selects.append((sql, params))
	
		def add_class_select(extra_where: list[str], extra_params: list, anchor_join: str = " "):
			join = (
				"FROM event_df e "
				"JOIN term_concept_class_map_df m "
				"ON m.term = e.normalized_term AND m.concept_class = ?"
			)
			where = [pid_clause] + extra_where + num_where
			params = [concept_class] + list(pid_params) + extra_params + num_params
			sql = f"SELECT e.* {join}{anchor_join}WHERE " + " AND ".join(where)
			selects.append((sql, params))
	
		# Trends → no temporal window (lift all rows)
		if mod in {"uptrended", "downtrended", "unchanged"}:
			if term:
				add_term_select(extra_where=[], extra_params=[])
			if concept_class:
				add_class_select(extra_where=[], extra_params=[])
			return ""
	
		# Non-trend → same temporal semantics
		mode = self._temporal_mode(temporal_ctx)
	
		if mode == "history":
			if term:
				add_term_select(extra_where=["e.history = TRUE"], extra_params=[])
			if concept_class:
				add_class_select(extra_where=["e.history = TRUE"], extra_params=[])
			return ""
	
		if mode == "baseline":
			early_or_hist = "(COALESCE(e.day_offset_end, e.day_offset_start) <= 2 OR e.history = TRUE)"
			if term:
				add_term_select(extra_where=[early_or_hist], extra_params=[])
			if concept_class:
				add_class_select(extra_where=[early_or_hist], extra_params=[])
			return ""
	
		# window mode
		spec = self._resolve_temporal_day_window(temporal_ctx, patient_alias="e")
		ov_sql, ov_params = self._temporal_overlap_sql(spec)
		anchor_join = f" {spec.join} " if (spec.mode == "per_patient" and spec.join) else " "
	
		if term:
			add_term_select(extra_where=[ov_sql], extra_params=ov_params, anchor_join=anchor_join)
			if spec.include_history:
				add_term_select(extra_where=["e.history = TRUE"], extra_params=[])
	
		if concept_class:
			add_class_select(extra_where=[ov_sql], extra_params=ov_params, anchor_join=anchor_join)
			if spec.include_history:
				add_class_select(extra_where=["e.history = TRUE"], extra_params=[])
	
		return ""
		
	def _compile_event_evidence_sql(self, trace: dict, patient_ids: list[str]) -> tuple[str | None, list | None]:
		"""
		Build `UNION ALL` of event evidence SELECTs for non-trend leaves.

		Also "lifts" anchor concept terms referenced in temporal contexts so the
		final LLM can see anchor rows. Returns (sql, params) or (None, None).
		"""
		if not patient_ids:
			return None, None
	
		self._cte_map.clear()
	
		def collect_anchor_terms_from_temporal_ctx(temporal_ctx):
			terms = []
			if not temporal_ctx:
				return terms
			for obj in temporal_ctx:
				a = (obj.get("anchor") or {})
				if a.get("form") == "concept":
					t = self._string_normalizer(a.get("resolved_term") or a.get("term") or "")
					if t:
						terms.append(t)
			return terms
	
		selects: list[tuple[str, list]] = []
		placeholders = ",".join("?" for _ in patient_ids)
		pid_clause, pid_params = f"e.patient_id IN ({placeholders})", list(patient_ids)
	
		anchor_terms = set()
	
		def walk(node, parent_temporal):
			if not isinstance(node, dict):
				return
			temporal_ctx = node.get("temporal") or parent_temporal
	
			# collect anchor terms
			for t in collect_anchor_terms_from_temporal_ctx(temporal_ctx):
				anchor_terms.add(t)
	
			# leaf?
			if ("clauses" not in node) and (node.get("resolved_term") or node.get("resolved_class")):
				mod = (node.get("modifier") or "").lower().strip()
				if mod in {"uptrended", "downtrended", "unchanged"}:
					# skip here; handled by trend compiler
					return
				self._emit_event_leaf_selects(node, temporal_ctx, pid_clause, pid_params, selects)
				return
			# boolean
			for ch in (node.get("clauses") or []):
				walk(ch, temporal_ctx)
	
		walk(trace, parent_temporal=None)
	
		# add one SELECT per anchor concept term (from event_df)
		for t in anchor_terms:
			sql = (
				"SELECT e.* FROM event_df e "
				"WHERE " + pid_clause + " AND e.normalized_term = ?"
			)
			selects.append((sql, pid_params + [t]))
	
		if not selects:
			return None, None
	
		# Build CTE list (anchor CTEs) + unioned CTE
		ctes, cte_params = [], []
		for name, (cte_sql, cparams) in self._cte_map.items():
			ctes.append(f"{name} AS ({cte_sql})")
			cte_params.extend(cparams)
	
		union_body_sql = "\nUNION ALL\n".join(s for s, _ in selects)
		unioned_cte = f"unioned AS (\n{union_body_sql}\n)"
		with_sql = "WITH " + ", ".join(ctes + [unioned_cte]) if ctes else "WITH " + unioned_cte
	
		# Join demographics at the end for convenience
		final_sql = (
			f"{with_sql}\n"
			"SELECT u.*, p.patient_name, p.sex, p.age\n"
			"FROM unioned u\n"
			"JOIN patient_df p ON p.patient_id = u.patient_id"
		)
		params = cte_params + [p for _, ps in selects for p in ps]
		return final_sql, params
		
	def _compile_trend_evidence_sql(self, trace: dict, patient_ids: list[str]) -> tuple[str | None, list | None]:
		"""
		Build `UNION ALL` of trend evidence SELECTs for trend leaves only.

		Constrained by patient IDs and resolved (anchor1, anchor2, change).
		Returns (sql, params) or (None, None) if no trend leaves.
		"""
		if not patient_ids:
			return None, None
	
		selects: list[tuple[str, list]] = []
		placeholders = ",".join("?" for _ in patient_ids)
		pid_clause, pid_params = f"t.patient_id IN ({placeholders})", list(patient_ids)
	
		def walk(node, parent_temporal):
			if not isinstance(node, dict):
				return
			temporal_ctx = node.get("temporal") or parent_temporal
	
			# leaf?
			if ("clauses" not in node) and (node.get("resolved_term") or node.get("resolved_class")):
				mod = (node.get("modifier") or "").lower().strip()
				if mod not in {"uptrended", "downtrended", "unchanged"}:
					return  # not a trend leaf
	
				anchors = self._resolve_trend_anchors(temporal_ctx)
				if not anchors:
					return
				anchor1, anchor2 = anchors
	
				term  = node.get("resolved_term")
				concept_class = node.get("resolved_class")
	
				base_where = [pid_clause, "t.anchor1 = ?", "t.anchor2 = ?", "t.change = ?"]
				base_params = pid_params + [anchor1, anchor2, mod]
	
				if term:
					sql = (
						"SELECT t.*, p.patient_name, p.sex, p.age "
						"FROM trends_df t "
						"JOIN patient_df p ON p.patient_id = t.patient_id "
						"WHERE " + " AND ".join(base_where + ["t.normalized_term = ?"])
					)
					selects.append((sql, base_params + [term]))
	
				if concept_class:
					sql = (
						"SELECT t.*, p.patient_name, p.sex, p.age "
						"FROM trends_df t "
						"JOIN term_concept_class_map_df m "
						"  ON m.term = t.normalized_term AND m.concept_class = ? "
						"JOIN patient_df p ON p.patient_id = t.patient_id "
						"WHERE " + " AND ".join(base_where)
					)
					selects.append((sql, [concept_class] + base_params))
	
				return
	
			# boolean
			for ch in (node.get("clauses") or []):
				walk(ch, temporal_ctx)
	
		walk(trace, parent_temporal=None)
	
		if not selects:
			return None, None
	
		union_body_sql = "\nUNION ALL\n".join(s for s, _ in selects)
		# No extra CTEs needed; trends don’t rely on anchor CTEs.
		return union_body_sql, [p for _, ps in selects for p in ps]
		
	def _compile_all_events_context_sql(self, patient_ids: list[str]) -> tuple[str, list]:
		"""
		Fetch all events for these patients as context (no boolean trace logic).

		Returns
		-------
		(sql, params)
		"""
		placeholders = ",".join("?" for _ in patient_ids)
		params: list = list(patient_ids)
	
		where_clauses = [f"e.patient_id IN ({placeholders})"]
	
		base_select = (
			"SELECT e.*, p.patient_name, p.sex, p.age "
			"FROM event_df e "
			"JOIN patient_df p ON p.patient_id = e.patient_id "
			"WHERE " + " AND ".join(where_clauses)
		)
	
		return base_select, params
	
	def _compile_candidate_sql(self, trace):
		"""
		Build the full candidate retrieval SQL with optional anchor CTEs.

		Steps
		-----
		1) Reset CTE registry.
		2) Compile boolean tree to a WHERE predicate (populates `_cte_map`).
		3) Compile patient-references prefilter from `patient_references`.
		4) WITH-clause includes anchor CTEs + `candidates` CTE.
		5) Final SELECT DISTINCT patient_id from `candidates` filtered by WHERE.
		6) Return (sql, params) with proper parameter ordering.
		"""
		# 1) reset CTE stash
		self._cte_map.clear()
	
		# 2) compile boolean tree (may populate _cte_map via temporal anchors)
		where_sql, where_params = self._compile_node(trace, parent_temporal=None)
	
		# 3) patient prefilter (id or name matches any reference)
		pr_sql, pr_params = self._compile_patient_refs(trace.get("patient_references"))
	
		# 4) assemble WITH list: anchor CTEs + candidates
		ctes = []
		cte_params = []
		for name, (cte_sql, cparams) in self._cte_map.items():
			ctes.append(f"{name} AS ({cte_sql})")
			cte_params.extend(cparams)
	
		candidates_cte = (
			"candidates AS ("
			"  SELECT p.patient_id FROM patient_df p "
			f"  WHERE {pr_sql}"
			")"
		)
		if ctes:
			with_clause = "WITH " + ", ".join(ctes + [candidates_cte])
		else:
			with_clause = "WITH " + candidates_cte
	
		# 5) final query: filter candidates by the compiled predicate
		sql = (
			f"{with_clause} "
			"SELECT DISTINCT p.patient_id "
			"FROM candidates p "
			f"WHERE {where_sql}"
		)
	
		# 6) params: CTE params first, then prefilter params, then boolean-tree params
		params = cte_params + pr_params + where_params
		return sql, params
		
	def run_evidence_dfs(self, patient_ids: list[str], sql_event: str, params_event: list[int|float|str|bool], sql_trend: str, params_trend: list[int|float|str|bool], include_context: bool) -> dict[str, pd.DataFrame]:
		"""
		Execute evidence retrieval as separate sources and return a dict:
			{'event_df': ..., 'trend_df': ..., 'context_df': ...}
		Keys are present only if that source produced rows.

		Parameters
		----------
		patient_ids : list[str]
			For context compilation when `include_context` is True.
		sql_event, params_event : str | None, list | None
			Event evidence query and parameters.
		sql_trend, params_trend : str | None, list | None
			Trend evidence query and parameters.
		include_context : bool
			When True, fetch all events as context (no boolean filtering).

		Returns
		-------
		dict[str, pd.DataFrame]
			Available keys: 'event_df', 'trend_df', optionally 'context_df'.
		"""
		out: dict[str, pd.DataFrame] = {}
	
		with duckdb.connect(self.db, read_only=True) as con:
			df_event = pd.DataFrame()
			df_trend = pd.DataFrame()
			df_ctx   = pd.DataFrame()
		
			# EVENT evidence
			if sql_event:
				df_event = con.execute(sql_event, params_event).fetchdf()
				if not df_event.empty:
					df_event["context"] = False  # mark event rows as non-context
		
			# TREND evidence (unchanged)
			if sql_trend:
				df_trend = con.execute(sql_trend, params_trend).fetchdf()
				if not df_trend.empty:
					out["trend_df"] = df_trend
		
			# CONTEXT
			if include_context:
				sql_ctx, params_ctx = self._compile_all_events_context_sql(patient_ids)
				df_ctx = con.execute(sql_ctx, params_ctx).fetchdf()
				if not df_ctx.empty:
					df_ctx["context"] = True  # mark context rows
		
			# Merge EVENT + CONTEXT into a single event_df, preferring EVENT on duplicates
			if not df_event.empty or not df_ctx.empty:
				if df_event.empty:
					combined = df_ctx.copy()
				elif df_ctx.empty:
					combined = df_event.copy()
				else:
					combined = pd.concat([df_event, df_ctx], ignore_index=True)
					combined = combined.drop_duplicates(subset=["patient_id","normalized_term","day_offset_start","day_offset_end"], keep="first").reset_index(drop=True)
		
				out["event_df"] = combined
				
		return out