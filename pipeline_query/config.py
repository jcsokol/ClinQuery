"""
config.py

Centralized configuration for the query system.

This module provides:
- **Params**: core system paths, model choices, and thresholds
- **Displays**: version strings and model identifiers for reporting
- **ErrorMessages**: standardized error messages for user-facing responses
- **Settings**: convenience wrapper aggregating all of the above

Environment variables used:
- `OPENAI_API_KEY`: OpenAI authentication key
- `PG_DSN`: optional Postgres DSN string for persistence/logging
"""

from dataclasses import dataclass
from typing import List
import os

@dataclass(frozen=True)
class Params:
	openai_api_key = os.getenv("OPENAI_API_KEY", "")
	duckdb_path = './db/master_db_v1.duckdb'
	faiss_path = "./db/alias_vectors.faiss"
	idmap_path = "./db/alias_vector_ids.npy"
	embedder_model = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext' 
	custom_concepts_path = './db/custom_concepts' 
	pg_dsn = os.getenv("PG_DSN", "")
	gpt_model = 'gpt-4.1'
	intent_parser_max_retries = 1
	intent_parser_temperature = 0
	final_response_temperature = 0.1
	fuzzy_min = 92
	short_bump = 5
	min_search_term_len = 6
	ann_topk = 32
	ann_min = 90
	max_qe_datapoints = 120 # maximum qualifying evidence datapoints to feed final llm call (soft cap)
	max_context_datapoints = 400 # maximum context datapoints to feed final llm call (soft cap)
	
@dataclass(frozen=True)
class Displays:
	db_version = '1.0'
	query_engine_version = '1.0'
	NER_model = 'UFNLP/gatortron-base'
	relations_model = 'michiyasunaga/BioLinkBERT-large'
	
@dataclass(frozen=True)
class ErrorMessages:
	failed_intent_parsing = """Your request could not be interpreted. This may be because:

1. **Out of scope** – The query is not clinical in nature.  
2. **Unsupported structure** – Current limitations include:  
   - **AND/OR clauses**: only ***all*** *must be present* or ***any*** *must be present* supported (nested logic not allowed).  
   - **Equality checks**: asking “where X = Y” is not supported (please provide a **range** instead).  
   - **Numeric deltas**: asking for “increase of 10” is not supported (please ask for a **qualitative change**, e.g. “uptrended”/“downtrended”/“stable”)."""
   
	unsupported_trend_anchor = """Currently supported trends are limited to:  

- **Across an interval**: throughout admission, prepump, while on pump, or postpump  
- **Relative to an anchor**: pump placement, while on pump, pump removal

Your query seems to involve a trend relative to an unsupported time point (for example: *“Did temperature decrease after antibiotics were stopped?”*). These comparisons are not yet supported but will be in a future update. """

	no_retrieved_patients = """No patients matched your query. Try either changing or relaxing your constraints."""
	
	exception = """An exception occured that has now been logged. It will be reviewed asap!"""

	@staticmethod
	def unrecognized_term(unmapped_terms: List[str]) -> str:
		terms_list = ", ".join(f"***{t}***" for t in unmapped_terms)
		return f"""The following terms could not be mapped to any concept in the database: {terms_list}

This may be because:

1. **No match in data** – The term (or its aliases) does not exist within the digested EHR notes.  
2. **Too abstract** – The term may be too general and has not yet been defined in the abstraction layer.  
   - Try replacing it with more concrete constraints (e.g. *signs of liver failure* → worsened liver function panel OR coagulopathy OR ascites ...).  
   - Soon you’ll be able to define your own YAML files on-the-fly with nested AND/OR logic and feature weighting. Try a query with ***clinical stability*** or ***end-organ underperfusion*** to see this in action right now.  
3. **Alias/abbreviation issue** – Occasionally the system may miss aliases. Try rephrasing terms (e.g. *nsaid* → *non-steroidal anti-inflammatory drug*)."""

@dataclass(frozen=True)
class Settings:
	params: Params = Params()
	displays: Displays = Displays()
	error_messages: ErrorMessages = ErrorMessages()

settings = Settings()

