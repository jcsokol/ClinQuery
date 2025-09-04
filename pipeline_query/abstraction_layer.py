"""abstraction_layer.py

High-level concept resolution and intent expansion layer.

This module compiles a user intent (lightweight JSON) into an extended
"trace" that includes resolved targets (terms, concept classes, or
custom concepts), per-clause temporal anchors, and helpful rollups for
later SQL compilation. It does **not** build or execute SQL.

Key features
------------
- Fuzzy matching (RapidFuzz) with tunable thresholds.
- ANN fallback (FAISS) over pre-normalized alias embeddings.
- Support for YAML-defined custom concepts with polarity logic. These override terms/classes.

External dependencies
---------------------
- FAISS index file (vector rows correspond 1:1 to `alias_ids` array)
- NumPy array `alias_ids` mapping FAISS row → alias_id
- An embedding function `embed_fn: Callable[[str], np.ndarray]` that returns a 1D vector.

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
import duckdb
import numpy as np
import faiss
from rapidfuzz import process, fuzz
from config import settings
import re
from collections import defaultdict
import copy
from pathlib import Path
import yaml


# ----------------- Public data structures -----------------

@dataclass
class QueryPlan:
	"""Container for the compiled result.
	
	Attributes
	----------
	trace:
	Extended intent dictionary enriched with per-clause resolutions,
	temporal anchor resolutions, and rollups.
	error:
	A human-readable message set **only** when no top-level clause matched
	anything (term/class/custom). Otherwise ``None``.
	"""
	trace: Dict[str, Any]         
	error: Optional[str] = None    


@dataclass
class ResolvedTarget:
	"""Represents a resolved target for a query token.
	
	Attributes
	----------
	kind:
	One of {"term", "concept_class", "custom", "unresolved"}.
	canonical:
	Canonical string for the target.
	matched_alias:
	The alias string that was selected by the matcher.
	query_term:
	The normalized user-provided token.
	alias_id:
	Integer identifier from the alias table (``-1`` for custom/direct lookup).
	method:
	One of {"fuzzy", "faiss", "direct lookup", "none"} describing how it was matched.
	score:
	Score on a 0–100 scale. Fuzzy and ANN are both normalized to this range.
	"""
	kind: str                 
	canonical: str
	matched_alias: str     
	query_term: str          
	alias_id: int
	method: str            
	score: float           


# ----------------- Core compiler -----------------

class QueryCompiler:
	"""
	Build an extended 'trace' dict from an input intent:
	  - For each clause: resolved_term / resolved_class / resolved_custom (+ score/method)
	  - For each 'concept' temporal anchor: resolved_term / resolved_class / resolved_custom (+ score/method)
	  - Top-level rollups: unresolved_concepts_list, resolved_mappings,
						   resolved_concept_class_term_expansions
	Does NOT build or execute SQL (yet).

	Expected DuckDB tables:
	  - alias_to_canonical_df(alias_id, alias, target_kind, canonical, status, priority)
		* target_kind ∈ {'term','concept_class'}
	  - term_concept_class_map_df(term, concept_class, status, priority)
	"""
	def __init__(self, duckdb_path: str, faiss_path: str, idmap_path: str,
				 embed_fn: Callable[[str], np.ndarray]):
		# Load alias + class maps into RAM once; close DB
		con = duckdb.connect(duckdb_path, read_only=True)

		adf = con.execute("""
		  SELECT alias_id, alias, target_kind, canonical, COALESCE(priority,100) AS priority
		  FROM alias_to_canonical_df
		  WHERE COALESCE(status,'active')='active'
		""").df()

		cdf = con.execute("""
		  SELECT concept_class, term, COALESCE(priority,100) AS priority
		  FROM term_concept_class_map_df
		  WHERE COALESCE(status,'active')='active'
		  ORDER BY concept_class, priority ASC, term ASC
		""").df()

		con.close()
		
		# FAISS + id map (row -> alias_id); FAISS vectors pre-normalized to unit length
		self.faiss = faiss.read_index(faiss_path)
		self.alias_ids = np.load(idmap_path, mmap_mode="r")
		if self.faiss.ntotal != len(self.alias_ids):
			raise RuntimeError(f"FAISS ntotal ({self.faiss.ntotal}) != alias_vector_ids length ({len(self.alias_ids)})")

		# id -> (alias, kind, canonical, priority)
		self.alias_meta: Dict[int, Tuple[str, str, str, int]] = {}
		# alias -> [alias_id,...]
		self.alias_to_ids: Dict[str, List[int]] = {}
		for r in adf.itertuples(index=False):
			aid = int(r.alias_id)
			self.alias_meta[aid] = (r.alias, r.target_kind, r.canonical, int(r.priority))
			self.alias_to_ids.setdefault(r.alias, []).append(aid)

		# class -> [terms] (for rollups / expansions)
		self.class_terms_map: Dict[str, List[str]] = {}
		for cc, grp in cdf.groupby("concept_class", sort=False):
			self.class_terms_map[cc] = grp["term"].tolist()

		# Fuzzy universe = unique alias strings
		self.fuzzy_choices: List[str] = list(self.alias_to_ids.keys())

		self.embed_fn = embed_fn
		
		# sanity check: ensure that there are only 'term'+'concept_class'+'custom' entries in adf['target_kind']
		if not set(adf['target_kind']) <= {'term','concept_class'}:
			raise RuntimeError("alias_to_canonical_df['target_kind'] contains entries other than 'term'+'concept_class'")
			
		# read in yaml files (custom concept definitions)
		self.custom_concepts, self.custom_alias_to_key = self.load_custom_concepts(settings.params.custom_concepts_path)
		
		
	# ---------- Public API ----------
	
	def compile(self, intent: Dict[str, Any]) -> QueryPlan:
		"""
		Returns QueryPlan with:
		  - trace: extended copy of the input intent with per-clause and per-anchor resolutions
		  - error: set only if no top-level clause term resolved to any of term/concept_class/custom
		"""
		trace, any_clause_resolved = self.extend_intent(intent)

		plan_error = None
		if not any_clause_resolved and len(trace.get("clauses") or []) > 0:
			plan_error = "No queried concept matched either a term or a concept class or a custom mapping"

		return QueryPlan(trace=trace, error=plan_error)


	# ---------- Intent → Extended Trace ----------
	
	def extend_intent(self, intent: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
		"""
		Make a deep copy of the input intent and extend in-place:
		  - For each clause: resolved_term / resolved_class / resolved_custom (+ score/method)
		  - For each 'concept' temporal anchor: resolved_term / resolved_class / resolved_custom (+ score/method)
		  - Add top-level:
			  * unresolved_concepts_list
			  * resolved_mappings (list of tuples: (query_concept, canonical, kind))
			  * resolved_concept_class_term_expansions ({class: [terms]})
		  - Custom terms can currently only be matched to clauses and not temporal anchors (by design)
		Returns (trace, any_clause_resolved).
		"""
		trace: Dict[str, Any] = copy.deepcopy(intent)

		unresolved_concepts: List[str] = []
		resolved_mappings: List[Tuple[str, str, str]] = []  # (query_concept, canonical, kind)
		class_expansions: Dict[str, set] = defaultdict(set)

		clauses = trace.get("clauses") or []
		any_clause_resolved = False

		for cl in clauses:
			# ----- Resolve the clause term itself -----
			q_raw = cl.get("term") or ""
			best_term, best_class, best_custom, class_terms = self.resolve(q_raw)

			# term
			if best_term is not None and best_custom is None:
				cl["resolved_term"] = best_term.canonical
				cl["resolved_term_score"] = float(best_term.score)
				cl["resolved_term_method"] = best_term.method
				any_clause_resolved = True
				resolved_mappings.append((best_term.query_term, best_term.canonical, "term"))
			else:
				pass

			# class
			if best_class is not None and best_custom is None:
				cl["resolved_class"] = best_class.canonical
				cl["resolved_class_score"] = float(best_class.score)
				cl["resolved_class_method"] = best_class.method
				any_clause_resolved = True
				resolved_mappings.append((best_class.query_term, best_class.canonical, "concept_class"))
				for t in class_terms.get(best_class.canonical, []):
					class_expansions[best_class.canonical].add(t)
			else:
				pass

			# custom
			if best_custom is not None:
				cl["resolved_custom"] = best_custom.canonical
				cl["resolved_custom_score"] = float(best_custom.score)
				cl["resolved_custom_method"] = best_custom.method
				any_clause_resolved = True
				resolved_mappings.append((best_custom.query_term, best_custom.canonical, "custom"))
			else:
				pass

			if (best_term is None) and (best_class is None) and (best_custom is None):
				q_norm = (q_raw or "").strip().lower()
				if q_norm and (q_norm not in unresolved_concepts):
					unresolved_concepts.append(q_norm)

			# ----- Resolve temporal anchors (if any) -----
			temporals = cl.get("temporal") or []
			for t in temporals:
				a = (t or {}).get("anchor")
				if not isinstance(a, dict):
					continue
				form = (a.get("form") or "").lower()
				if form == "concept":
					anchor_q = a.get("term") or ""
					a_term, a_class, _, a_terms_map = self.resolve(anchor_q)

					# term
					if a_term is not None:
						a["resolved_term"] = a_term.canonical
						a["resolved_term_score"] = float(a_term.score)
						a["resolved_term_method"] = a_term.method
						resolved_mappings.append((a_term.query_term, a_term.canonical, "term"))
					else:
						pass

					# class
					if a_class is not None:
						a["resolved_class"] = a_class.canonical
						a["resolved_class_score"] = float(a_class.score)
						a["resolved_class_method"] = a_class.method
						resolved_mappings.append((a_class.query_term, a_class.canonical, "concept_class"))
						for tcanon in a_terms_map.get(a_class.canonical, []):
							class_expansions[a_class.canonical].add(tcanon)
					else:
						pass

					if (a_term is None) and (a_class is None):
						qn = (anchor_q or "").strip().lower()
						if qn and (qn not in unresolved_concepts):
							unresolved_concepts.append(qn)
				else:
					pass
						
		# extend constraints using yaml file if any of the terms within any clause were matched to a custom term 
		trace = self.inject_custom_constraints_into_trace(trace)
		
		# Final rollups
		trace["unresolved_concepts_list"] = sorted(list(set(unresolved_concepts)))
		trace["resolved_mappings"] = sorted(set(resolved_mappings))
		trace["resolved_concept_class_term_expansions"] = {cc: sorted(list(terms)) for cc, terms in class_expansions.items()}

		return trace, any_clause_resolved


	# ---------- Resolver (normalize, fuzzy-first, ANN fallback; returns BOTH term and class) ----------
	
	def resolve(self, q_raw: str) -> Tuple[Optional[ResolvedTarget], Optional[ResolvedTarget], Optional[ResolvedTarget], Dict[str, List[str]]]:
		"""
		Returns (best_term, best_class, best_custom, class_terms_by_class).
		  - best_*: ResolvedTarget or None
		  - class_terms_by_class: {canonical_class: [terms]}
		No exceptions; placeholders are None when unmatched.
		"""
		q = self.string_normalizer(q_raw)

		# comparator for "best"
		def better(a: Optional[ResolvedTarget], b: ResolvedTarget) -> bool:
			if a is None:
				return True
			if b.score != a.score:
				return b.score > a.score
			# tie-break with stored priority (lower is better) than canonical
			_, _, canon_b, prio_b = self.alias_meta[b.alias_id]
			_, _, canon_a, prio_a = self.alias_meta[a.alias_id]
			if prio_b != prio_a:
				return prio_b < prio_a
			return canon_b < canon_a

		best_term: Optional[ResolvedTarget] = None
		best_class: Optional[ResolvedTarget] = None
		best_custom: Optional[ResolvedTarget] = None
		
		# do keyword based custom concept matching
		if q in self.custom_alias_to_key: 
			best_custom = ResolvedTarget("custom", self.custom_alias_to_key[q], q, q, -1, "direct lookup", 100.0)

		# ---------- FUZZY ----------
		hits = process.extract(q, self.fuzzy_choices, scorer=fuzz.ratio, limit=64)
		for alias_str, score, _ in hits:
			s = float(score)  # 0..100
			thr = self.fuzzy_min_for(q, alias_str)
			if s < thr:
				continue
			for aid in self.alias_to_ids[alias_str]:
				a_str, kind, canonical, _prio = self.alias_meta[aid]
				rt = ResolvedTarget(kind, canonical, a_str, q, aid, "fuzzy", s)
				if kind == "term":
					if better(best_term, rt): best_term = rt
				elif kind == "concept_class":
					if better(best_class, rt): best_class = rt

		# ---------- ANN search ----------
		if len(q) >= settings.params.min_search_term_len and best_term is None and best_class is None: # only do ANN search if no term AND no class have been matched

			qv = self.embed_fn(q)  # embed_fn applies device/dtype consistently
			q2 = np.ascontiguousarray(qv.reshape(1, -1))
			faiss.normalize_L2(q2)
			D, I = self.faiss.search(q2, settings.params.ann_topk)

			seen = set()
			for d, row in zip(D[0], I[0]):
				if row < 0:
					continue
				aid = int(self.alias_ids[row])
				if aid in seen:
					continue
				seen.add(aid)
				a_str, kind, canonical, _prio = self.alias_meta[aid]
				s = float(d) * 100.0  # 0..100
				if s < settings.params.ann_min:
					continue
				rt = ResolvedTarget(kind, canonical, a_str, q, aid, "faiss", s)
				if kind == "term":
					if better(best_term, rt): best_term = rt
				elif kind == "concept_class":
					if better(best_class, rt): best_class = rt

		class_terms = {}
		if best_class is not None:
			class_terms[best_class.canonical] = self.class_terms_map.get(best_class.canonical, [])

		return best_term, best_class, best_custom, class_terms


	# ---------- Utilities ----------

	def fuzzy_min_for(self, q: str, alias_str: str) -> float:
		"""Compute the fuzzy threshold for a pair of short/long strings.
		
		Applies a small bump when *both* strings are shorter than
		``settings.params.min_search_term_len``.
		"""
		bump = settings.params.short_bump if (len(q) < settings.params.min_search_term_len and len(alias_str) < settings.params.min_search_term_len) else 0
		return float(settings.params.fuzzy_min + bump)
		
	def string_normalizer(self, s: str) -> str:
		return re.sub(r"\s+", " ", re.sub(r"[-_]", " ", s)).strip().lower()
		
	def load_custom_concepts(self, dir_path: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Set[str]]]:
		"""
		Load all YAML concept files in a directory into normalized in-memory structures.
	
		Returns:
		  custom_concepts: {
			 <concept_key>: {
				"aliases": term -> modifier -> polarity mappings,
				"definition_polarity": "positive"|"negative",
				"lookback_window": {"days": int, "hours": int},
				"definition": rule tree,
				"source_path": "..."
			 }, ...
		  }
		  custom_alias_to_key: alias -> concept_key mappings
		"""
		base = Path(dir_path)
		if not base.exists() or not base.is_dir():
			raise FileNotFoundError(f"Directory not found: {dir_path}")
	
		custom_concepts: Dict[str, Dict[str, Any]] = {}
		custom_alias_to_key: Dict[str, str] = {}
	
		files = list(base.glob("*.yaml")) + list(base.glob("*.yml"))
		for fp in files:
			with fp.open("r", encoding="utf-8") as f:
				y = yaml.safe_load(f) or {}
	
			# Polarity (default = definition_polarity must be present)
			pol = y.get("definition_polarity", "positive")
			if pol not in {"positive", "negative"}:
				raise ValueError(f"{fp.name}: 'definition_polarity' must be 'positive' or 'negative'")
	
			# Window (ints)
			win = y.get("lookback_window", {"days":1,"hours":0})
			lookback_window = {"days": int(win.get("days", 0)), "hours": int(win.get("hours", 0))}
	
			# Build term→modifier→polarity alias map
			aliases_raw = y.get("aliases")
			aliases: Dict[str, Dict[str, str]] = {}
			
			for polarity in ["positive","negative"]:
				items = aliases_raw.get(polarity,[])
				for item in items:
					if isinstance(item, dict):
						aliases.setdefault(self.string_normalizer(item['term']), {})
						aliases[self.string_normalizer(item['term'])][self.string_normalizer(item.get('modifier',''))] = polarity
					else:
						aliases.setdefault(self.string_normalizer(item), {})
						aliases[self.string_normalizer(item)][''] = polarity
						
			# Check that definition exists
			if "definition" not in y:
				raise ValueError(f"{fp.name}: missing required key 'definition'")
			
			concept_key = self.string_normalizer(fp.stem)  # filename stem is the canonical/ID
			custom_concepts[concept_key] = {"aliases": aliases, "definition_polarity": pol, "lookback_window": lookback_window, "definition": y["definition"], "source_path": str(fp.resolve())}
	
			# Populate term-only router: term -> concept_key mappings for all terms
			for term in aliases.keys():
				if term in custom_alias_to_key.keys(): print('warning: overlapping aliases bewteen yaml files.')
				custom_alias_to_key[term] = concept_key
	
		return custom_concepts, custom_alias_to_key
		
	def combine_into_single_trace(self, trace: dict, custom_trees: list) -> dict:
		"""Combine the original (possibly non-nested) trace with injected custom trees.
		
		If ``custom_trees`` is non-empty, returns a new top-level AND node that
		contains the original trace as the first child and each injected custom
		tree as additional children. Otherwise, returns the original shape.
		"""
		keep = {k: v for k, v in trace.items() if k not in ("logic", "clauses")}
		orig = {"logic": trace.get("logic", "AND"), "clauses": trace.get("clauses", [])}
		if not custom_trees:  # nothing to add
			return {**keep, **orig}
		return {**keep, "logic": "AND", "clauses": [orig, *custom_trees]}
		
	def polarity_is_opposite(self, def_pol, clause_term, clause_mod) -> bool:
		"""Return ``True`` if the clause polarity is opposite the YAML definition's polarity.
		
		The clause's term and modifier are normalized. We look up the clause term in
		the concept's alias map to find the polarity associated with the most specific
		matching modifier (longest substring first), falling back to an empty
		modifier if present. If the associated alias polarity differs from the
		concept's ``definition_polarity``, the tree should be polarity-flipped.
		"""
		clause_term = self.string_normalizer(clause_term)
		clause_mod = self.string_normalizer(clause_mod)
	
		modifier_to_pol_dict = self.custom_concepts[self.custom_alias_to_key[clause_term]]['aliases'][clause_term]
		modifiers = list(modifier_to_pol_dict)
		modifiers.sort(key = lambda m: (len(m), m), reverse=True)
		
		if clause_mod=="" and "" in modifiers:
			if modifier_to_pol_dict[""] != def_pol: 
				return True
			else:
				return False
		elif clause_mod=="":
			return False
	
		for m in modifiers:
			if m in clause_mod:       
				if modifier_to_pol_dict[m] != def_pol:
					return True
				else:
					return False  

		return False
		
	def flip_polarity_in_place(self, node: dict):
		"""Flip presence/absence and numeric comparators **in place**.
		
		- Leaf nodes: toggle ``modifier`` (present↔absent) and invert ``op``.
		- Boolean nodes: swap ``logic`` (AND↔OR) and recurse into children.
		"""
		# Leaf: flip modifier and numeric comparator
		if "clauses" not in node and "term" in node:
			m = node.get("modifier")
			if m == "present":
				node["modifier"] = "absent"
			elif m == "absent":
				node["modifier"] = "present"
	
			op = node.get("op")
			if op:
				node["op"] = {">":"<=", "<":">=", ">=":"<", "<=":">", "=":"!=", "!=":"="}.get(op, op)
			return
	
		# Internal node: flip boolean operator and recurse
		logic = node.get("logic")
		if logic == "AND":
			node["logic"] = "OR"
		elif logic == "OR":
			node["logic"] = "AND"
	
		for ch in node.get("clauses", []):
			self.flip_polarity_in_place(ch)
				
	def normalize_yaml_definition_to_intent_tree(self, custom_key: str, yaml_node: dict, lookback_window: dict, temporal: dict | None, unresolved_set: set):
		"""Normalize a YAML definition subtree into the internal intent format.
		
		
		The function converts ``any_of``/``all_of`` boolean groups and validates
		leaves by resolving features to canonical terms. Leaves that cannot be
		resolved are pruned and collected in ``unresolved_set``.
		
		
		The returned root node (if any) also carries the original ``temporal``
		constraint and a ``custom_concept_expansion`` marker for traceability.
		"""
		def norm(node):
			# Boolean nodes
			if "any_of" in node:
				kids = [norm(c) for c in node["any_of"]]
				kids = [k for k in kids if k is not None]
				if not kids: return None
				return {"logic": "OR", "clauses": kids}
			if "all_of" in node:
				kids = [norm(c) for c in node["all_of"]]
				kids = [k for k in kids if k is not None]
				if not kids: return None
				return {"logic": "AND", "clauses": kids}
	
			# Leaf (terms only)
			feat_raw = (node.get("feature"))
			feat = self.string_normalizer(feat_raw)
			if not feat:
				return None
			best_term, _, _, _ = self.resolve(feat)
			if best_term is None:
				unresolved_set.add(feat)
				return None
	
			leaf = {"term": feat, "resolved_term": best_term.canonical}
			if "modifier" in node:
				leaf["modifier"] = node["modifier"]
			if "op" in node and "value" in node:
				leaf["op"] = node["op"]
				leaf["value"] = node["value"]
				if "unit" in node:
					leaf["unit"] = node["unit"]
			return leaf
	
		normalized_tree = norm(yaml_node)
	
		if normalized_tree is not None:
			normalized_tree["temporal"] = temporal
			normalized_tree["custom_concept_expansion"] = custom_key
	
		return normalized_tree, unresolved_set
								
	def inject_custom_constraints_into_trace(self, trace) -> Dict[str, Any]:
		"""Inject normalized YAML-defined custom constraints into the trace.
		
		
		For each clause that resolved to a custom concept, we:
		- Fetch the YAML definition & lookback window
		- Normalize it to the internal intent tree
		- Flip polarity if the clause's alias polarity is opposite the definition's
		- Register the injected tree for AND-combination with the original trace
		
		
		Adds ``unresolved_custom_concepts_list`` to the trace. Returns the
		modified trace.
		"""
		# initialize datastructures
		custom_trees = [] 
		unresolved_custom = set() 
		
		# now traverse all clauses (input is not nested per requirement that user query can't produce nested logic) and create new nested tree if clause contains custom concept
		clauses = trace.get("clauses")
		for cl_i, cl in enumerate(clauses):
	
			custom_key = cl.get("resolved_custom")
			if custom_key is None:
				continue
	
			# fetch YAML-defined custom concept; if missing, record and skip
			cc = self.custom_concepts.get(custom_key)
	
			# get temporal constraint of original clause
			tree_temporal = cl.get("temporal")
	
			# normalize yaml 'definition' into new nested clause tree
			# returns either a normalized node {"logic", "clauses", "temporal"} or a leaf {"term", "temporal", ...} or None (pruned)
			normalized_root, unresolved_custom = self.normalize_yaml_definition_to_intent_tree(custom_key, cc["definition"], cc['lookback_window'], tree_temporal, unresolved_custom)
	
			# if everything pruned away (e.g., all leaves unresolved), skip injection
			if normalized_root is None:
				continue
				
			# flip tree polarity if necessary
			def_pol = cc.get("definition_polarity", "positive")
			clause_term, clause_modifier = cl.get("term"), cl.get("modifier_text")
			if self.polarity_is_opposite(def_pol, clause_term, clause_modifier):
				self.flip_polarity_in_place(normalized_root)
				normalized_root['flipped_polarity'] = True
	
			# register the injected tree
			custom_trees.append(normalized_root)
			
		# finalize outputs
		trace = self.combine_into_single_trace(trace,custom_trees)
		trace["unresolved_custom_concepts_list"] = list(unresolved_custom)
		
		# return modified trace
		return trace