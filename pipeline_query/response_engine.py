"""
response_engine.py

Assembles patient-level evidence and generates the final natural-language answer for a clinical query using an LLM.

Pipeline (high level)
---------------------
1) `_package_evidence(...)` converts raw retrieval outputs (`event_df`, `trend_df`) into a compact JSON payload with:
   - scope info (cohort vs subset), subsampling flags, CTA control
   - per-patient `evidence` (events/trends) and optional `context`
   - optional summaries to respect `max_*_datapoints` caps from `settings.params`
2) `_generate_final_answer(...)` sends a system prompt, the user query, the packaged evidence JSON, and a rubric to the model to produce the final prose.
3) `generate_response(...)` orchestrates both steps and returns the model text plus the evidence JSON actually supplied to the model.

Notes
-----
- Counts like `retrieved_candidates` / `shown_candidates` are included in the evidence package for internal control but the system prompt forbids the model from exposing them in the final prose.
- DataFrames are copied defensively before mutation and may be summarized when the number of datapoints exceeds configured caps.

"""

from openai import OpenAI
import json
from config import settings
from prompts import final_llm_call_system_prompt, final_llm_call_rubric
import pandas as pd
import random


# ---------------- Public API ----------------

def generate_response(user_query, candidate_pt_ids, event_df, trend_df, is_subset_query) -> tuple[str, dict]:
	"""
	Package evidence and obtain a final natural-language answer from the LLM.

	Parameters
	----------
	user_query : str
		The original natural-language question from the user.
	candidate_pt_ids : list
		List of candidate patient IDs produced by the retrieval step.
	event_df : pd.DataFrame
		Event-style evidence table (may include `context` boolean column).
		Expected columns (subset): patient_id, patient_name, age, sex, normalized_term, value, unit, is_active, is_started, is_ended, day_offset_start, day_offset_end, context (optional).
	trend_df : pd.DataFrame
		Trend-style evidence table with two anchors per item.
		Expected columns (subset): patient_id, patient_name, age, sex, normalized_term, unit, anchor1_start, anchor1_end, anchor2_start, anchor2_end, val_start, val_end, delta, change.
	is_subset_query : bool
		True if the user asked about specific patients; controls scope and CTA.

	Returns
	-------
	(str, dict)
		- Final model response text.
		- The exact evidence JSON payload sent to the model.
	"""
	packaged_evidence_dict = _package_evidence(candidate_pt_ids, event_df, trend_df, is_subset_query)
	client = OpenAI(api_key=settings.params.openai_api_key)
	model_response_str = _generate_final_answer(client, user_query, packaged_evidence_dict)
	return model_response_str, packaged_evidence_dict
	
	
# ---------------- Private functions ----------------
	
def _package_evidence(candidate_pt_ids, event_df, trend_df, is_subset_query) -> dict:
	"""
	Convert raw retrieval outputs into a compact evidence JSON for the LLM.

	Behavior
	--------
	- Splits events into qualifying evidence (`context==False`) vs context.
	- Enforces datapoint caps from `settings.params`:
		* `max_qe_datapoints` for qualifying evidence
		* `max_context_datapoints` for context
	- Summarizes event datapoints per (patient, term, unit, active flags) when needed to stay under caps; otherwise includes individual items.
	- If still too large, subsamples candidate patients (tracking `subsampled`/`shown_candidates`) and uses summaries.
	- Builds per-patient objects with `evidence` (events/trends) and `context`.

	Parameters
	----------
	candidate_pt_ids : list
		All candidate patient IDs before any subsampling.
	event_df : pd.DataFrame
		Event evidence; may be empty/None.
	trend_df : pd.DataFrame
		Trend evidence; may be empty/None.
	is_subset_query : bool
		Controls `query_scope` and `cta_allowed`.

	Returns
	-------
	dict
		Evidence package containing scope flags, counts, per-patient items,
		and summary statistics used by the final prompt.
	"""
	
	# initialize variables & preprocess inputs
	evidence_package = {"retrieved_candidates":len(candidate_pt_ids),"query_scope":'subset' if is_subset_query else 'cohort',"subsampled":False,"cta_allowed":is_subset_query==False and len(candidate_pt_ids)>4}
	event_df = event_df.copy(deep=True) if event_df is not None else pd.DataFrame(columns=['patient_id'])
	no_unit_str = "_none_"
	if not event_df.empty: event_df['unit'] = event_df['unit'].fillna(no_unit_str) # necessary because None's throw groupby off
	trend_df = trend_df.copy(deep=True) if trend_df is not None else pd.DataFrame(columns=['patient_id'])
	event_df_qe = event_df.loc[event_df["context"] == False] if "context" in event_df else event_df
	event_df_ctx = event_df.loc[event_df["context"] == True] if "context" in event_df else event_df
	patients_map = {}

	###### POPULATE QUALIFIED EVIDENCE ######
	if len(event_df_qe)+len(trend_df) <= settings.params.max_qe_datapoints: # enough space for all entries
		if not event_df_qe.empty:
			for (pid,pname,page,psex), grp in event_df_qe.groupby(["patient_id","patient_name","age","sex"],sort=False):
				if pid not in patients_map: patients_map[pid] = {"id":str(pname)+' ('+str(pid)+')',"age":page,"sex":psex,"evidence":[],"context":[]}
				patients_map[pid]["evidence"] += [_create_event_term_entry(r,no_unit_str) for r in grp.itertuples(index=False)]
		if not trend_df.empty:
			for (pid,pname,page,psex), grp in trend_df.groupby(["patient_id","patient_name","age","sex"],sort=False):
				if pid not in patients_map: patients_map[pid] = {"id":str(pname)+' ('+str(pid)+')',"age":page,"sex":psex,"evidence":[],"context":[]}
				patients_map[pid]["evidence"] += [_create_trend_term_entry(r,no_unit_str) for r in grp.itertuples(index=False)]
	elif (0 if event_df_qe.empty else event_df_qe.groupby(['patient_id','normalized_term','unit','is_active','is_started','is_ended'], sort=False).ngroups) + len(trend_df) <= settings.params.max_qe_datapoints: # enough space when events are summarized
		if not event_df_qe.empty:
			for (pid,pname,page,psex,_,_,_,_,_), df_pt in event_df_qe.groupby(['patient_id','patient_name',"age","sex",'normalized_term','unit','is_active','is_started','is_ended'], sort=False):
				if pid not in patients_map: patients_map[pid] = {"id":str(pname)+' ('+str(pid)+')',"age":page,"sex":psex,"evidence":[],"context":[]}
				patients_map[pid]["evidence"].append(_create_event_summary_entry(df_pt,no_unit_str))
		if not trend_df.empty:
			for (pid,pname,page,psex), grp in trend_df.groupby(["patient_id","patient_name","age","sex"],sort=False):
				if pid not in patients_map: patients_map[pid] = {"id":str(pname)+' ('+str(pid)+')',"age":page,"sex":psex,"evidence":[],"context":[]}
				patients_map[pid]["evidence"] += [_create_trend_term_entry(r,no_unit_str) for r in grp.itertuples(index=False)]
	else: # subsample retrieved candidates such that we will hit *approximately* settings.params.max_qe_datapoints using summarized datapoints
		items_per_pt = ((event_df_qe.groupby(['patient_id','normalized_term','unit','is_active','is_started','is_ended'], sort=False).ngroups/event_df_qe['patient_id'].nunique() if not event_df_qe.empty else 0)+(len(trend_df)/trend_df['patient_id'].nunique() if not trend_df.empty else 0))
		candidate_pt_ids = random.sample(candidate_pt_ids, min(int(0.9*settings.params.max_qe_datapoints/items_per_pt),len(candidate_pt_ids)))
		evidence_package['subsampled'] = True
		evidence_package['shown_candidates'] = len(candidate_pt_ids)
		event_df_qe = event_df_qe[event_df_qe['patient_id'].isin(candidate_pt_ids)]
		event_df_ctx = event_df_ctx[event_df_ctx['patient_id'].isin(candidate_pt_ids)]
		trend_df = trend_df[trend_df['patient_id'].isin(candidate_pt_ids)]
		if not event_df_qe.empty:
			for (pid,pname,page,psex,_,_,_,_,_), df_pt in event_df_qe.groupby(['patient_id','patient_name',"age","sex",'normalized_term','unit','is_active','is_started','is_ended'], sort=False):
				if pid not in patients_map: patients_map[pid] = {"id":str(pname)+' ('+str(pid)+')',"age":page,"sex":psex,"evidence":[],"context":[]}
				patients_map[pid]["evidence"].append(_create_event_summary_entry(df_pt,no_unit_str))		
		if not trend_df.empty:
			for (pid,pname,page,psex), grp in trend_df.groupby(["patient_id","patient_name","age","sex"],sort=False):
				if pid not in patients_map: patients_map[pid] = {"id":str(pname)+' ('+str(pid)+')',"age":page,"sex":psex,"evidence":[],"context":[]}
				patients_map[pid]["evidence"] += [_create_trend_term_entry(r,no_unit_str) for r in grp.itertuples(index=False)]
	
	###### POPULATE CONTEXT ######
	qualifying_terms = (set(event_df_qe['normalized_term']) if not event_df_qe.empty else set()).union(set(trend_df['normalized_term']) if not trend_df.empty else set())
	if len(event_df_ctx) <= settings.params.max_context_datapoints: # enough space for all context entries
		if not event_df_ctx.empty:
			for (pid,pname,page,psex), grp in event_df_ctx.groupby(["patient_id","patient_name","age","sex"],sort=False):
				if pid not in patients_map: patients_map[pid] = {"id":str(pname)+' ('+str(pid)+')',"age":page,"sex":psex,"evidence":[],"context":[]}
				patients_map[pid]["context"] += [_create_event_term_entry(r,no_unit_str) for r in grp.itertuples(index=False)]
	elif event_df_ctx.groupby(['patient_id','normalized_term','unit','is_active','is_started','is_ended'], sort=False).ngroups <= settings.params.max_context_datapoints: # enough space for all context when summarized
		if not event_df_ctx.empty:
			for (pid,pname,page,psex,_,_,_,_,_),df_pt in event_df_ctx.groupby(['patient_id','patient_name',"age","sex",'normalized_term','unit','is_active','is_started','is_ended'], sort=False):
				if pid not in patients_map: patients_map[pid] = {"id":str(pname)+' ('+str(pid)+')',"age":page,"sex":psex,"evidence":[],"context":[]}
				patients_map[pid]["context"].append(_create_event_summary_entry(df_pt,no_unit_str))
	# check if there is enough space for summarized entries from the context, only considering the terms that appear in the qualifying evidence
	elif event_df_ctx[event_df_ctx['normalized_term'].isin(qualifying_terms)].groupby(['patient_id','normalized_term','unit','is_active','is_started','is_ended'], sort=False).ngroups <= settings.params.max_context_datapoints:
		event_df_ctx = event_df_ctx[event_df_ctx['normalized_term'].isin(qualifying_terms)]
		if not event_df_ctx.empty:
			for (pid,pname,page,psex,_,_,_,_,_),df_pt in event_df_ctx.groupby(['patient_id','patient_name',"age","sex",'normalized_term','unit','is_active','is_started','is_ended'], sort=False):
				if pid not in patients_map: patients_map[pid] = {"id":str(pname)+' ('+str(pid)+')',"age":page,"sex":psex,"evidence":[],"context":[]}
				patients_map[pid]["context"].append(_create_event_summary_entry(df_pt,no_unit_str))
	else: 
		pass # ignore context

	# count datapoints for provenance and finally return evidence package
	evidence_package["patients"] = list(patients_map.values())
	evidence_package["datapoints"] = sum([len(patients_map[pt]['evidence']) for pt in patients_map.keys()])
	evidence_package["contextual_datapoints"] = sum([len(patients_map[pt]['context']) for pt in patients_map.keys()])
	return evidence_package


def _create_event_term_entry(r,no_unit_str) -> dict:
	"""
	Build a single 'event' evidence item from one `event_df` row-group tuple.

	Returns
	-------
	dict
		Event item with fields:
		type='event', term, unit, timestamp (single int or (start,end)),
		value, is_active, is_started, is_ended.
	"""
	
	ts_start = getattr(r, "day_offset_start", None)
	if pd.isna(ts_start)==False: ts_start = int(ts_start)
	ts_end = getattr(r, "day_offset_end", ts_start)
	if pd.isna(ts_end)==False: ts_end = int(ts_end)
	item = {
		"type": "event",
		"term": getattr(r, "normalized_term", None),
		"unit": getattr(r, "unit", None),
		"timestamp": ts_start if ts_start==ts_end else (ts_start,ts_end),
		"value": getattr(r, "value", None),
		"is_active": getattr(r, "is_active", None),  
		"is_started": getattr(r, "is_started", None),  
		"is_ended": getattr(r, "is_ended", None)
	}
	if item["unit"]==no_unit_str: item["unit"]=None
	return item

def _create_trend_term_entry(r,no_unit_str) -> dict:
	"""
	Build a single 'trend' evidence item from one `trend_df` row-group tuple.

	Returns
	-------
	dict
		Trend item with fields:
		type='trend', term, unit, anchor1_day, anchor2_day,
		anchor1_median, anchor2_median, delta, change.
	"""
	
	ts1_start = getattr(r, "anchor1_start", None)
	if pd.isna(ts1_start)==False: ts1_start = int(ts1_start)
	ts1_end = getattr(r, "anchor1_end", ts1_start)
	if pd.isna(ts1_end)==False: ts1_end = int(ts1_end)
	ts2_start = getattr(r, "anchor2_start", None)
	if pd.isna(ts2_start)==False: ts2_start = int(ts2_start)
	ts2_end = getattr(r, "anchor2_end", ts2_start)
	if pd.isna(ts2_end)==False: ts2_end = int(ts2_end)  
	item = {
		"type": "trend",
		"term": getattr(r, "normalized_term", None),
		"unit": getattr(r, "unit", None),
		"anchor1_day": ts1_start if ts1_start==ts1_end else (ts1_start,ts1_end),
		"anchor2_day": ts2_start if ts2_start==ts2_end else (ts2_start,ts2_end),
		"anchor1_median": getattr(r, "val_start", None),
		"anchor2_median": getattr(r, "val_end", None),
		"delta": getattr(r, "delta", None),
		"change": getattr(r, "change", None),
	}
	if item["unit"]==no_unit_str: item["unit"]=None
	return item
	
def _create_event_summary_entry(g: pd.DataFrame, no_unit_str: str) -> dict:
	"""
	Build a summarized 'event' evidence item from a grouped `event_df` slice. The summary aggregates values across a (patient, term, unit, activity flags) group to keep payload sizes within limits.

	Returns
	-------
	dict
		Summary item with min/max/median/std (when numeric), earliest/latest day,
		and counts of total vs numeric entries.
	"""
	
	vals = g["value"]
	vals_num = pd.to_numeric(vals, errors="coerce")
	num_mask = vals_num.notna()
	vals_num = vals_num[num_mask].astype(float)
	n_numeric = int(vals_num.size)
	n_total = int(len(g))
	
	start_min = g["day_offset_start"].min(skipna=True)
	start_max = g["day_offset_start"].max(skipna=True)
	end_max   = g["day_offset_end"].max(skipna=True)
	earliest_timestamp = int(start_min) if pd.notna(start_min) else None
	latest_timestamp = int(end_max) if pd.notna(end_max) else (int(start_max) if pd.notna(start_max) else None)
	
	item = {
		"type": "summary",
		"term": str(g['normalized_term'].iloc[0]),
		"is_active": bool(g['is_active'].iloc[0]),
		"is_started": bool(g['is_started'].iloc[0]),
		"is_ended": bool(g['is_ended'].iloc[0]),
		"unit": str(g['unit'].iloc[0]),
		"stats": {"n_total": int(n_total), "n_numeric": int(n_numeric)},
		"earliest_day": int(earliest_timestamp) if earliest_timestamp is not None else None,
		"latest_day": int(latest_timestamp) if latest_timestamp is not None else None
	}
	if item["unit"]==no_unit_str: item["unit"]=None
	ts = g["day_offset_end"].fillna(g["day_offset_start"])
	if n_numeric>0:
		item["stats"]["min"] = float(vals_num.min())
		item["stats"]["max"] = float(vals_num.max())
		if n_numeric>1: item["stats"]["min_entry_timestamp"] = int(ts.loc[vals_num.idxmin()]) if pd.notna(ts.loc[vals_num.idxmin()]) else None
		if n_numeric>1: item["stats"]["max_entry_timestamp"] = int(ts.loc[vals_num.idxmax()]) if pd.notna(ts.loc[vals_num.idxmax()]) else None
		item["stats"]["median"] = float(vals_num.median())
		item["stats"]["std"] = round(float(vals_num.std(ddof=1)),3) if n_numeric>1 else None
		
	return item


def _generate_final_answer(client, user_query, packaged_evidence_dict) -> str:
	"""
	Call the LLM to produce the final adjudicated answer.

	Parameters
	----------
	client : OpenAI
		Initialized OpenAI client.
	user_query : str
		Original user question.
	packaged_evidence_dict : dict
		Evidence payload produced by `_package_evidence(...)`.

	Returns
	-------
	str
		Final response text generated by the model.
	"""

	user_payload_json = json.dumps(packaged_evidence_dict, ensure_ascii=False, separators=(',', ':'))

	messages = [
		{"role": "system", "content": final_llm_call_system_prompt()},
		{"role": "user", "content": f"User query:\n{user_query}"},
		{"role": "user", "content": "Patient evidence JSON:\n```json\n" + user_payload_json + "\n```"},
		{"role": "user", "content": final_llm_call_rubric()}
	]

	resp = client.chat.completions.create(model=settings.params.gpt_model, temperature=settings.params.final_response_temperature, messages=messages)
	return resp.choices[0].message.content