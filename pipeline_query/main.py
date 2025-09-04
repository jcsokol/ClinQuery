"""
main.py

Core orchestrator for executing a user query end-to-end.

Pipeline
--------
1. Parse the natural language query into a structured intent (LLM).
2. Run the abstraction layer (concept resolution).
3. Build and run candidate retrieval SQL.
4. Build and run evidence retrieval SQL.
5. Package evidence and generate the final response (LLM).
6. Return a full payload dict with traces, latencies, and answer text.

Notes
-----
- All major stages are timed and latencies stored in the payload.
- Exceptions are caught and logged into the payload under `errors`.
- Logging is **disabled**: the call to `log_payload(...)` is commented out. To enable persistence to Postgres, remove the comment in the `finally` block.

Public API
----------
- `run_query(user_query: str, session_id: str, query_compiler: Callable)`

"""

from typing import Dict, Callable
import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timezone
from intent_parser import parse_intent
import time
from config import settings
from sql_engine import SqlEngine
import traceback
from response_engine import generate_response
from logger import log_payload



def run_query(user_query: str, session_id: str, query_compiler: Callable):
	"""
	Execute a single query end-to-end and return a payload dictionary.

	Parameters
	----------
	user_query : str
		Natural-language query provided by the user.
	session_id : str
		Session identifier (used for grouping related queries in a session).
	query_compiler : Callable
		Instance of QueryCompiler (abstraction layer) initialized with embeddings and concept maps.

	Returns
	-------
	Dict
		Payload dictionary containing:
		- raw query and request metadata
		- parsed intent and abstraction-layer trace
		- SQL queries and retrieved evidence
		- concept mappings and expansions
		- final LLM answer
		- timing breakdowns and run status
		- error logs if any stage failed
	"""
	# Set timer
	timer_start = time.perf_counter()
	# Initialize payload datastructure
	request_id = str(uuid.uuid4())
	request_timestamp = datetime.now(timezone.utc).isoformat() + "Z"
	payload_dict = initialize_payload_dict(user_query,session_id,request_id,request_timestamp)
	
	try: # try running entire query
	
		# Parse intent
		payload_dict['parsed_intent'] = parse_intent(user_query)
		payload_dict['intent_generation_latency'] = time.perf_counter()-timer_start; timer_start_int = time.perf_counter()
		if payload_dict['parsed_intent']['parse_status']!='ok':
			payload_dict['run_status'] = 'intent parsing failed'
			payload_dict['final_answer'] = settings.error_messages.failed_intent_parsing
			return payload_dict
		# Run abstraction layer
		compiled_query = query_compiler.compile(payload_dict['parsed_intent'])
		payload_dict['abstraction_layer_latency'] = time.perf_counter()-timer_start_int; timer_start_int = time.perf_counter()
		if len(compiled_query.trace["unresolved_concepts_list"])>0:
			payload_dict['run_status'] = 'some queried concepts not resolved'
			payload_dict['final_answer'] = settings.error_messages.unrecognized_term(compiled_query.trace["unresolved_concepts_list"])
			return payload_dict
		payload_dict['concept_mappings'] = "  \n".join(f"***{t}*** mapped to *{c}* ({ty})" for t, c, ty in compiled_query.trace["resolved_mappings"]).replace("concept_class","class")
		payload_dict['concept_expansions'] = "  \n".join(f"***{cls}*** expanded to {', '.join(f'*{t}*' for t in terms)}" for cls, terms in compiled_query.trace["resolved_concept_class_term_expansions"].items())
		payload_dict['abstraction_layer_trace'] = compiled_query.trace
		# Build & run CANDIDATE retrieval query
		sql_engine = SqlEngine(settings.params.duckdb_path,query_compiler.resolve('left heart pump'))
		cand_sql, cand_params = sql_engine.build_candidate_sql(payload_dict['abstraction_layer_trace'])
		if sql_engine.error_str == 'unsupported trend anchor': 
			payload_dict['final_answer'] = settings.error_messages.unsupported_trend_anchor
			payload_dict['run_status'] = 'sql query failed'
			return payload_dict 
		payload_dict['sql']['cand_retrieval_query'] = sql_engine.build_query_str_for_debug_trace(cand_sql, cand_params)
		candidate_pt_ids = sql_engine.run_candidate_sql(cand_sql, cand_params)
		payload_dict['n_candidates'] = len(candidate_pt_ids)
		if payload_dict['n_candidates']==0:
			payload_dict['final_answer'] = settings.error_messages.no_retrieved_patients
			payload_dict['run_status'] = 'no patients matched query'
			return payload_dict
		payload_dict['candidate_patient_ids'] = ', '.join(map(str, candidate_pt_ids))
		# Build & run EVIDENCE retrieval query
		sql_engine = SqlEngine(settings.params.duckdb_path,query_compiler.resolve('left heart pump'))
		sql_event, params_event, sql_trend, params_trend = sql_engine.build_evidence_sql(payload_dict['abstraction_layer_trace'],candidate_pt_ids)
		payload_dict['sql']['evidence_retrieval_query'] = sql_engine.build_query_str_for_debug_trace(sql_event, params_event, sql_trend, params_trend)
		payload_dict['sql']['retrieved_event_df'], payload_dict['sql']['retrieved_trend_df'] = sql_engine.run_evidence_sql(candidate_pt_ids, sql_event, params_event, sql_trend, params_trend)
		payload_dict['sql_query_latency'] = time.perf_counter()-timer_start_int; timer_start_int = time.perf_counter()
		# Evidence packaging & final llm call
		is_subset_query = len(payload_dict['parsed_intent']['patient_references'])>0
		payload_dict['final_answer'], payload_dict['final_llm_call_input'] = generate_response(user_query, candidate_pt_ids, payload_dict['sql']['retrieved_event_df'], payload_dict['sql']['retrieved_trend_df'], is_subset_query)
		payload_dict['final_llm_call_latency'] = time.perf_counter()-timer_start_int; timer_start_int = time.perf_counter()
		payload_dict['run_status'] = 'ok'
		
	except Exception as e: # store exception if anything failed in an unaccounted for way
	
		payload_dict['run_status'] = 'exception'
		payload_dict['errors'].append({"type": type(e).__name__,"message": str(e),"stacktrace": traceback.format_exc()})
		payload_dict['final_answer'] = settings.error_messages.exception 
		return payload_dict
		
	finally: # now log everything
	
		try:
			# remove dataframes from payload to be logged
			payload_dict_copy = dict(payload_dict)
			payload_dict_copy['sql'] = dict(payload_dict_copy.get('sql') or {})
			payload_dict_copy['sql'].pop('retrieved_event_df', None)
			payload_dict_copy['sql'].pop('retrieved_trend_df', None)
			# log payload and also retrieve lifetime queries
			# payload_dict['lifetime_queries'] = log_payload(payload=payload_dict_copy) ### LOGGING DISABLED (undo comment out to enable it) ###
			
		except Exception as log_err:
			payload_dict['errors'].append({"type": "LoggingError", "message": str(log_err)})
		
	
		# Return payload for UI
		payload_dict['total_latency'] = time.perf_counter()-timer_start
		return payload_dict




# ---------------- Helper functions ----------------

def initialize_payload_dict(user_query: str, session_id: str, request_id: str, request_timestamp: str) -> Dict:
	"""
	Initialize a structured payload dictionary with defaults.
	"""
	return {
		"user_query":user_query,
		"request_id":request_id,
		"request_timestamp":request_timestamp,
		"parsed_intent":None, # shows intent outputted by first llm call
		"concept_mappings":None, # shows what each entity was normalized to
		"concept_expansions":None, # for expansions via either concept classes or manually defined expansions
		"abstraction_layer_trace":None,
		"sql": {"cand_retrieval_query":None,
				"evidence_retrieval_query":None,
				"retrieved_event_df": None,
				"retrieved_trend_df": None,
				},
		"n_candidates":0,
		"candidate_patient_ids":None,
		"final_llm_call_input":None,
		"final_answer":None,
		"total_latency":0,
		"intent_generation_latency":0,
		"abstraction_layer_latency":0,
		"sql_query_latency":0,
		"final_llm_call_latency":0,
		"errors":[],
		"intent_parser_temperature":settings.params.intent_parser_temperature,
		"final_response_temperature":settings.params.final_response_temperature,
		"llm_model":settings.params.gpt_model,
		"NER_model":settings.displays.NER_model,
		"relations_model":settings.displays.relations_model,
		"db_id":settings.displays.db_version,
		"query_engine_version":settings.displays.query_engine_version,
		"run_status":'error',
		"lifetime_queries":None,
	}
