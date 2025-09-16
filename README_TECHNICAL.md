# Technical README

## 1. Overview
Two modular pipelines:

1. **Ingestion Pipeline** (`pipeline_ingest/`)  
   - Converts ICU-style EHR notes into a canonicalized SQL database of patient timelines  
   - Precomputes baselines, trends, and stores structured events

2. **Query Pipeline** (`pipeline_query/`)  
   - Interprets natural language clinical queries  
   - Generates SQL queries over the database  
   - Emits reasoning/debug trace
	
---

## 2. Folder Structure
```text
├── pipeline_ingest/
│   ├── db
│   ├── ingest.py
│   ├── ner.py
│   ├── relations.py
│   ├── normalize.py
│   └── sql_writer.py
│
├── pipeline_query/
│   ├── db
│   ├── cli.py
│   ├── app.py
│   ├── config.py
│   ├── main.py
│   ├── intent_parser.py
│   ├── abstraction_layer.py
│   ├── sql_engine.py
│   ├── response_engine.py
│   └── logger.py
│
├── notebooks/
│   ├── db
│   ├── train_ner_model.ipynb
│   ├── train_relations_model.ipynb
│   └── *_helpers.py
│
├── raw_ehr_records
│
├── logging_setup.py
├── requirements.yml
├── README_TECHNICAL
└── README
```

---

## 3. Component Breakdown

### 3.1 Named Entity Recognition & Relation Extraction
- **NER model**: GatorTron 
- Extracts:
  - Clinical concepts (labs, meds, diagnoses, interventions)
  - Timestamps (absolute & relative)
  - Negations
  - Admission time
  - Sex
  - DOB
  - Age
- Trained on 589 GPT-annotated examples and evaluated using 20 hand-annotated simulated notes (20 notes expand into ~4000 spans)
- F1 scores of final model far outperformed F1 scores of GPT-annotations
- **Relations model**: BioLinkBERT
  - Concept ↔ Timestamp
  - Concept ↔ Negation
- Trained on 589 GPT-annotated examples and evaluated using 20 hand-annotated simulated notes (20 notes expand into >10,000 positive relation examples)
- F1 scores of final model again far outperformed F1 scores of GPT-annotations

---

### 3.2 Term Normalization & Timeline Construction
- **Term mapping**: uses fuzzy+embedding matching to UMLS terms
- **Concept class linking**: assigns concept classes to each term (e.g. ACEs/ARBs will have an 'antihypertensive drug' concept class along with many other classes)
- **Modifier normalization**: qualitative (↑/↓/stable) and quantitative fusion
- **Timestamp/negation extraction**: conflict resolution & prioritization
- **Precomputations**:
  - Baselines: admission, admission end, latest, prepump, during-pump, postpump, etc..
  - Trends: admission→latest, pre→during pump, during→post pump, etc..
- Outputs to `baselines` and `trends` SQL tables
- Note on normalizations: I had to make some custom fixes to UMLS terms and did not include these here to keep this pipeline generalizable. I also had to fuse synonymous terms using a fuzzy+embedding-based collapser I didn't include here. I encourage you to review your own extracted terms + concept classes and to experiment with ways to reduce ontology noise. 

---

### 3.3 SQL Schema
- **events** — all patient events with normalized terms
- **patients** — demographics & identifiers  
- **baselines** — precomputed baselines  
- **trends** — precomputed trends
- **alias to canonical** — alias → canonical mappings
- **concept class map** — concept class → term expansions

---

### 3.4 Query Interpreter + Clinical Abstraction Layer
- First LLM pass:  
  - Extracts query intent (query terms, modifiers, temporal constraints)
- Abstraction layer:  
  - Maps to canonical terms/modifiers
  - Custom logic for concepts like clinical stability or 'watcher status' patients
  - Allows for the user to create new concept definitions on-the-fly and for these to be read in without restarting a query engine instance

---

### 3.5 Frontend
- Simple UI for entering queries  
- Displays:
  - final synthesized answer
  - reasoning trace (user query, parsed intent, abstraction layer trace, SQL query+result, final packaged evidence for LLM)
- Has logger to audit performance

---

## 4. Data
- 589 synthetic ICU EHR notes generated via GPT
- Based on [Synthea](https://github.com/synthetichealth/synthea) patients to ensure a diverse demographic & diverse symptoms
- Simulates pump placement/removal, renal changes, hemodynamics, etc..

---

## 5. Installation & Running

### Installation (linux, gpu support)
```
# create environment
conda env create -f requirements.yml
conda activate clinquery_environment

# (optional) register notebook kernel
python -m ipykernel install --user --name=clinquery_environment

# set API key
export OPENAI_API_KEY=...
```
*Note: if you only intend to use the query pipeline you can comment out several packages in `requirements.yml` (they are labeled).*
### Running the Ingestion Pipeline
```
python -m pipeline_ingest.ingest \
  ./pipeline_ingest/db/input.jsonl \
  ./pipeline_query/db \
  --mrconso_rrf /path/to/MRCONSO.RRF \
  --mrrel_rrf   /path/to/MRREL.RRF \
  --mrsty_rrf   /path/to/MRSTY.RRF
```
*Note: you will have to obtain your own UMLS license and download mrconso.rrf+mrrel.rrf+mrsty.rrf, which aren't provided in this repo.*
### Running the Query Pipeline from the command line
<pre>
python -m pipeline_query.cli "did creatinine rise after pump removal in any patient?"
</pre>

### Running the Query Pipeline using the frontend
```
streamlit run pipeline_query/app.py
```

*All commands must be run from the repo root using the '-m' flag where included.*

---


## 6. Design Decisions & Tradeoffs

**Entity classifications:** Used a broad clinical entity NER label (conditions, meds, procedures, devices, observations) to reduce misses. Disambiguation happens later in normalization. 

**Negative example tuning for relation extraction:** Performed best when negatives were sampled with distances similar to positives. Careful tuning was critical. 

**Ontology coverage vs. precision:** UMLS provides broad coverage with many clinically meaningful classes (e.g., “antihypertensive drug”), but also includes noisy or irrelevant groupings. I added an abstraction layer to override UMLS classes, balancing breadth with precision. 

**Modifiers and values:** Parsed separately before normalization for generality, though this makes parsing accuracy critical. Terms that encode both (e.g. extubated) need special handling.

**Evidence extrapolation policy:** No assumptions beyond documented events, except pumps (assumed active between start/stop).

**Evidence retrieval policy:** I retrieve any candidate that *could* satisfy the query (including cases with conflicting evidence) so the LLM adjudicator can decide. The risk is flooding the adjudicator with marginal candidates.

**NER/relations dependency:** Recall of the entire system is directly tied to NER and relation extraction performance. Given ~4 weeks of development time, I opted for small language models (SLMs). They provided good recall, no hallucinations, and reasonable compute costs. Large language models (e.g., Google’s LangExtract) may boost recall but risk hallucination and require more resources.  

---

## 8. Failure Modes & Future Extensions

**Ontology & concept classes:** Too many irrelevant classes surface; pruning and better synonym collapsing are needed. Second, occasional misclassifications (e.g., furosemide as an antibiotic) highlight the need for automated checks to scale. 

**Query expressivity:** Extend to support open-ended summaries and demographic filters. 

**Evidence retrieval:** Add RAG fallback for recall, better numeric handling for qualitative constraints, and a relevance function that scales with token budget to avoid flooding the LLM.

**Performance:** The ingestion pipeline's runtime and memory usage can be optimized in several places to enable scaling to very large datasets (see performance & reproducibility). 

> ## **Vision:**
> Build a **“Wolfram Alpha for physicians”**, a system that transparently maps patient data onto evidence-based clinical algorithms and guidelines. Physicians don’t just want answers; they want to see the reasoning path, step by step, the same way they are trained to think. This tool could transform how doctors interact with patient data, and my background in both clinical medicine and ML engineering puts me in a unique position to build it.

---

## 9. Performance & Reproducibility
- **Ingestion**: ~589 notes / 90k events in ~2 hrs; parallelization can scale to millions. Bottleneck = relations + normalization loops. 
- **Query pipeline**: <15s latency, <1GB RAM; main costs are two LLM calls.
- **Dependencies**: Requires UMLS license/files.
- **NER + relation model weights** NER and relation weights (~1.3 GB each) excluded; contact me for access or fine-tune via notebooks.

---

## 10. License & Contact
License: MIT   
Contact: <span>jsokol</span><span>@</span><span>alumni.stanford.edu</span>