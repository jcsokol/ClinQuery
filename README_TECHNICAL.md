# Technical README

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open-informational)](https://clinquery-live.onrender.com/)

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
├── requirements.yml
├── README_TECHNICAL
└── README
```

*final scripts for ingestion pipeline will be pushed this week. 

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
python pipeline_ingest/ingest.py \
  --input_json ./db/input.jsonl \
  --output_path ./pipeline_query/db \ 
  --mrconso_rrf PATH \ 
  --mrrel_rrf PATH \ 
  --mrsty_rrf PATH
```
*Note: you will have to obtain your own UMLS license and download mrconso.rrf+mrrel.rrf+mrsty.rrf, which aren't provided in this repo.*
### Running the Query Pipeline from the command line
<pre>
python pipeline_query/cli.py "did creatinine rise after pump removal in any patient?"
</pre>

### Running the Query Pipeline using the frontend
```
streamlit run pipeline_query/app.py
```

---


## 6. Design Decisions & Tradeoffs

**Entity classifications:** I pushed most classification to the normalization layer by using a catch-all *clinical entity* label in NER (covering observations, conditions, devices, medications, procedures). This reduced false negatives from over-cautious NER at the cost of pushing disambiguation to the normalizer.   

**Negative example tuning for relation extraction:** The relations model only performed well when negative examples (entity–time/negation pairs) were selected with distances that matched those of positive examples. This tuning step was crucial to model performance.  

**Ontology coverage vs. precision:** UMLS provides broad coverage with many clinically meaningful classes (e.g., “antihypertensive drug”), but also includes noisy or irrelevant groupings. I added an abstraction layer to override UMLS classes, balancing breadth with clinical precision. 

**Modifiers and values:** These can either be parsed separately *before* normalization, or entities can be normalized *with* modifiers included (which requires explicit ontology entries). I chose the first approach, since it generalizes to unseen entity–modifier pairs, at the cost of needing very accurate parsing. A drawback was that at times terms that encode both a modifier and concept (e.g. 'extubated') aren't parsed well. Another example of a concept that requires special treatment is 'high blood pressure'. 

**Evidence extrapolation policy:** I avoided fabricating evidence by not assuming persistence of events beyond what was documented. For example, if an event occurred on day X, it was not automatically extended to X+1. The only exception: I assumed heart pumps remained active between “start” and “stop” events, since this is a safe clinical assumption and enables temporal constraints around pump-related outcomes. 

**Evidence retrieval policy:** I retrieve any candidate that *could* satisfy the query (including cases with conflicting evidence) so the LLM adjudicator can decide. The risk is flooding the adjudicator with marginal candidates.

**NER/relations dependency:** Recall of the entire system is directly tied to NER and relation extraction performance. Given ~4 weeks of development time, I opted for small language models (SLMs). They provided good recall, no hallucinations, and reasonable compute costs. Large language models (e.g., Google’s LangExtract) may boost recall but risk hallucination and require more resources.  

---

## 8. Failure Modes & Future Extensions

**Ontology & concept classes:**  
- Too many clinically irrelevant classes currently surface; pruning strategies are needed.  
- Equivalent terms/classes are occasionally not collapsed despite a merge step.  
- Misclassifications occur (e.g., *furosemide* miscategorized as an antibiotic). I can spot these errors instantly, but the system will eventually need automated checks on some level. Thus, a scalable system will need dedicated focus towards curating an error-free ontology. Realisticaly, this will take months to get right, but is worth the time. 
- **I currently prune concepts absent from the ingested dataset to reduce complexity**. This can cause queries to miss otherwise valid concepts and is probably the first thing I will target next. I would prefer retaining the concept index so the system can say a concept exists but wasn’t identified in any note.

**Query expressivity:**  
- System requires at least one entity in the query; it should be extended to support open-ended requests like “summarize patient X’s hospital course.”  
- Support for demographic subsetting (age, sex, etc.) is not yet implemented.  

**Evidence retrieval:**  
- RAG fallback not yet built; this would improve recall, though SQL-based retrieval is the priority for temporally precise queries.  
- Very rare parsing errors in values/units/timestamps may cause missed candidates.  
- When queries return very few datapoints, the system should retrieve additional supporting evidence using a relevance-based function. Currently I am using a two-stage approach that retrieves all datapoints, or all queried datapoints irrespective of temporal constraints, depending on which fits into the token limit. A continuous tuned relevance function that scales the retrieved context w.r.t. the remaining token limit would enable more accurate responses.  
- Frequently the system retrieves too many candidates, not all of which satisfy the constraints. This happens partly because my intent parser can’t yet convert qualitative constraints (e.g., “low,” “high”) into numeric thresholds. Fixing this will protect the LLM adjudicator from being flooded with irrelevant candidates.  

The ingestion pipeline's runtime and memory usage can be optimized in several places to enable scaling to very large datasets (see performance & reproducibility). 

> ## **Vision:**
> The long-term goal is a **“Wolfram Alpha for medicine”:** a system that shows transparently how it interpreted the data in response to a query using guideline-aware navigation. In medicine this would be invaluable, since clinical guidelines and algorithms are referenced all the time as physicians work up and treat patients. Ideally the system would surface the relevant guideline, traverse it step-by-step, and show where the patient lands. Physicians would **love** this, and my recent hospital years give me an edge for building it.

---

## 9. Performance & Reproducibility
- **Ingestion pipeline**: processes ~589 notes / ~90k events in ~2 hours, but this can be sped up via parallelization to handle ~millions of pages. Currently bottlenecked by relations predictions and entity normalization steps that loop over each entity using a single worker.  
- **Query pipeline**: typical query executes in <15s with <1GB RAM. Memory load comes mainly from the embedder model (initialized once at startup and shared across users). Latency is bottlenecked by two LLM calls.  
- Certain optimizations have been omitted here to keep the public version generalizable. Contact me if you’d like details.  
- You will need your own UMLS license (free) to run the ingestion pipeline.  
- **NER + relation model weights** (~1.3 GB each) are not included to keep this repo lightweight. Contact me for model weights, or fine-tune your own models via the provided notebooks.

---

## 10. License & Contact
License: MIT   
Contact: <span>jsokol</span><span>@</span><span>alumni.stanford.edu</span>
