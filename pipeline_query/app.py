"""EHR Query Engine Demo (Streamlit UI)

This script provides a thin UI layer over the core pipeline in `main.run_query`. It accepts a free-text clinical question, runs intent parsing + abstraction + SQL evidence retrieval, and renders both the final answer and a trace (parsed intent, SQL, retrieved evidence, latency, etc.).

Notes:
- The heavy lifting happens in `run_query` (see `main.py`).
- This app caches the `QueryCompiler` resources via `@st.cache_resource` so that the embedding model needs to only be loaded once on cold startup. 
"""

import streamlit as st
import uuid
from datetime import datetime, timezone
from main import run_query
from config import settings
from abstraction_layer import QueryCompiler
st.set_page_config(page_title="EHR Query Engine Demo", layout="wide")

# --- CSS modifications ---
st.markdown("""
<style>
/* Only style expanders that live INSIDE two expanders */
[data-testid="stExpander"] [data-testid="stExpander"] [data-testid="stExpander"] > details {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
}
/* Reduce padding inside the clickable summary row */
[data-testid="stExpander"] [data-testid="stExpander"] [data-testid="stExpander"] > details > summary {
  padding-top: 0.0rem !important;
  padding-bottom: 0.0rem !important;
  padding-left: 0.0 !important;
  line-height: 0.6rem !important;     /* reduce row height */
  min-height: 0 !important;         /* prevent default min-height */
  font-size: 0.6rem !important;     /* smaller text if needed */
}
/* Also tighten the body container inside innermost expanders */
[data-testid="stExpander"] [data-testid="stExpander"] [data-testid="stExpander"] {
  padding-top: 0.0rem !important;
  padding-bottom: 0.0rem !important;
  margin: 0 !important;
}
</style>
""", unsafe_allow_html=True)
	
# --- Session setup ---
if "session_id" not in st.session_state:
	st.session_state.session_id = str(uuid.uuid4())
if "last_result" not in st.session_state:
	st.session_state.last_result = None
if "last_query" not in st.session_state:
	st.session_state.last_query = ""

# --- Title + user-facing summary ---
st.markdown("""<div style="font-size: 2.5rem;font-weight: 600;line-height: 1.2;margin: 0 0 1.00rem 0;">EHR Query Engine Demo</div>""",unsafe_allow_html=True)
st.markdown(
	"""
**Query 589 synthetic ICU patient timelines, built from raw EHR notes using custom NER + relation models**  
This tool converts unstructured notes into a structured, timestamped database, enabling **temporally precise** and **concept-aware** clinical queries. It handles nuanced terms like *end-organ underperfusion* or *clinical stability* and resolves time windows (e.g., “between 1-2 days after pump removal”) directly against each patients' timeline — not just by text search. Most simulated admissions involved temporary heart pump placement.

**Examples you can try:**  
- *Of all patients with a history of diabetes, who had an A1C ≥6.5 during this admission?*
- *Did any patients with baseline creatinine ≤1.0 develop creatinine ≥1.5 within 72 hours?*
- *Which liver function markers downtrended after pump placement?*
- *Which patients exhibited end-organ hypoperfusion from 3 days post pump placement up to 1 day prior to pump removal?* \*\*\*
- *Which patients developed acute kidney injury on days 2–4 post-pump placement?*
- *How many patients had RIGHT heart failure after successful LEFT-sided pump weaning?*  
- *Show me liver function markers for Greta Schowlater (78b8a2b2) between 1 day post pump placement and pump removal.*
- *How did cbc and chem7 panels evolve from pump placement to pump removal?*
- *Did any patients with no history of myocardial infarctions or hypertension have a myocardial infarction during this admission?*""")

st.caption("[Technical walkthrough](https://github.com/jcsokol/ClinQuery) | Contact: jsokol\u200B@alumni.stanford.edu")

# --- Load datastructures into RAM on cold-start ---
@st.cache_resource(show_spinner="App cold start...")
def load_compiler() -> QueryCompiler:
	"""Load and cache the `QueryCompiler` and its sentence-transformer embedder.

	This is called once per process (thanks to `@st.cache_resource`), so model
	weights and FAISS/DuckDB handles are reused across runs and users.

	Returns
	-------
	QueryCompiler
		Fully initialized compiler that provides concept resolution and ANN search.
	"""
	from sentence_transformers import SentenceTransformer, models
	word_emb = models.Transformer(settings.params.embedder_model)
	pooling = models.Pooling(word_emb.get_word_embedding_dimension())
	model = SentenceTransformer(modules=[word_emb, pooling], device="cpu")
	def embed_fn(s: str):
		return model.encode([s], normalize_embeddings=True, convert_to_numpy=True)[0].astype("float32")
	return QueryCompiler(duckdb_path=settings.params.duckdb_path, faiss_path=settings.params.faiss_path, idmap_path=settings.params.idmap_path, embed_fn=embed_fn)
query_compiler = load_compiler()

# --- Input + Run ---
with st.form("query_form", clear_on_submit=False):
	c1, c2 = st.columns([8, 1])  # adjust ratios to taste
	with c1:
		user_query = st.text_input(
			"Ask any question about the 589 synthetic EHR records (e.g., “did creatinine rise after pump removal in any patient?”)",
			label_visibility="collapsed",  # or "collapsed" if you want tighter spacing
			placeholder="Type your question..."
		)
	with c2:
		run = st.form_submit_button("Run", use_container_width=True)

# --- Compute only when Run is pressed ---
if run and user_query.strip():
	with st.spinner("Reasoning..."):
		result = run_query(
			user_query=user_query,
			session_id=st.session_state.session_id,
			query_compiler=query_compiler
		)

	# Ensure identifiers even if main.py didn't set them
	result.setdefault("request_id", str(uuid.uuid4()))
	result.setdefault("request_timestamp", datetime.now(timezone.utc).isoformat() + "Z")

	# Persist the latest run
	st.session_state.last_result = result
	st.session_state.last_query = user_query

# --- Always render the most recent run (no recompute on toggle) ---
if st.session_state.last_result:
	result = st.session_state.last_result

	final_answer = result.get("final_answer") or "No answer generated."
	route = result.get("route") or "unknown"
	routing_decision = result.get("routing_decision") or ""
	ts = result.get("request_timestamp")  # your dict uses this key

# 	st.markdown("### Answer:")
	st.write(final_answer)

	# Clickable label to reveal reasoning (no checkbox, no tabs)
	with st.expander("Show reasoning trace", expanded=False):
		result = st.session_state.last_result  # already saved on Run
		
		# -------- Summary section --------
		st.markdown('**Status:** '+str(result.get("run_status"))+'  ·  '+'**Retrieved candidates:** '+str(result.get("n_candidates"))+'  ·  '+'**Latency:** '+str(round(result.get("total_latency"),2))+'s')
			
		# -------- Query parser --------
		with st.expander("Query parser", expanded=False):
			st.markdown("**User query**")
			st.text(result.get("user_query"))
			st.markdown("**Parsed intent**")
			st.json(result.get("parsed_intent"), expanded=False)
			st.markdown("**Abstraction layer trace**")
			st.json(result.get("abstraction_layer_trace"), expanded=False)
			st.markdown("**Concept mappings**")
			st.markdown(result.get("concept_mappings"))
			if isinstance(result.get("concept_expansions"), str) and len(result.get("concept_expansions"))>0:
				st.markdown("**Concept expansions**")
				st.markdown(result.get("concept_expansions"))
	
		# -------- SQL query --------
		with st.expander("SQL query", expanded=False):
			st.markdown("**Candidate retrieval query**")
			sql_block = result.get("sql")
			with st.expander("", expanded=False):
				st.code(sql_block.get("cand_retrieval_query"),language="sql")
			if int(result.get('n_candidates'))>0: # skip if no candidates were retrieved
				st.markdown("**Retrieved patient ids:**")
				if int(result.get('n_candidates'))>200:
					with st.container(height=200, border=False):
						st.text(result.get("candidate_patient_ids"))
				else:
					st.text(result.get("candidate_patient_ids"))
				st.markdown("**Evidence retrieval query**")
				
				with st.expander("", expanded=False):
					st.code(sql_block.get("evidence_retrieval_query"),language="sql")
				st.markdown("**Retrieved evidence**")
				event_df = sql_block.get("retrieved_event_df")
				trend_df = sql_block.get("retrieved_trend_df")
				if event_df is not None:
					st.dataframe(event_df, use_container_width=True, hide_index=True)
					st.caption(f"Row count: "+str(len(event_df)))
				if trend_df is not None: 
					if event_df is not None: st.markdown("**Retrieved precomputed trends**")
					st.dataframe(trend_df, use_container_width=True, hide_index=True)
					st.caption(f"Row count: "+str(len(trend_df)))
	
		# -------- Generation I/O --------
		if int(result.get('n_candidates'))>0: # skip if no candidates were retrieved
			with st.expander("Generation I/O", expanded=False):
				st.markdown("**Final LLM input**")
				st.json(result.get("final_llm_call_input"), expanded=False)
		
		# -------- Performance --------
		with st.expander("Performance & Reproducibility", expanded=False):
			st.markdown("**Total latency (s)**")
			st.text(round(result.get("total_latency"),2))
			st.markdown("**Intent parser latency (s)**")
			st.text(round(result.get("intent_generation_latency"),2))
			st.markdown("**Abstraction layer latency (s)**")
			st.text(round(result.get("abstraction_layer_latency"),2))
			st.markdown("**SQL query latency (s)**")
			st.text(round(result.get("sql_query_latency"),2))
			st.markdown("**Final LLM call latency (s)**")
			st.text(round(result.get("final_llm_call_latency"),2))
			st.markdown("**LLM model**")
			st.text(result.get("llm_model"))
			st.markdown("**NER model**")
			st.text(result.get("NER_model"))
			st.markdown("**Relations model**")
			st.text(result.get("relations_model"))
			st.markdown("**SQL database version**")
			st.text(result.get("db_id"))
			st.markdown("**Query engine version**")
			st.text(result.get("query_engine_version"))
			st.markdown("**Request timestamp**")
			st.text(result.get("request_timestamp"))
			st.markdown("**Request ID**")
			st.text(result.get("request_id"))
			st.markdown("**Lifetime queries**")
			st.text(result.get("lifetime_queries"))